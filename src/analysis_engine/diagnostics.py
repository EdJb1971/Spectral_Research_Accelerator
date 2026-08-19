import torch
import numpy as np
from typing import Dict, Any, Tuple, List
from src.physical_core.field import PhysicalField
from src.transform_engine.transforms import SpectralTransformEngine
from src.synthetic_generator.perturbation import PerturbationEngine

class SpectralSpatialAnalysisEngine:
    """
    Calculates diagnostics including power spectral density, wavelet energy, SSIM, and gradient errors.
    """
    @staticmethod
    def compute_radial_psd(data: torch.Tensor) -> Tuple[List[float], List[float]]:
        H, W = data.shape
        fft_coeffs = torch.fft.fft2(data)
        fft_shifted = torch.fft.fftshift(fft_coeffs)
        power_spectrum = torch.abs(fft_shifted) ** 2 / (H * W)
        
        y = torch.arange(H, dtype=torch.float32, device=data.device) - H // 2
        x = torch.arange(W, dtype=torch.float32, device=data.device) - W // 2
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(grid_y**2 + grid_x**2)
        
        r_int = torch.round(r).to(torch.long)
        max_r = int(min(H, W) // 2)
        
        psd = []
        wavenumbers = []
        for k in range(0, max_r + 1):
            mask = (r_int == k)
            if torch.any(mask):
                psd.append(torch.mean(power_spectrum[mask]).item())
                wavenumbers.append(float(k))
                
        return wavenumbers, psd

    @staticmethod
    def compute_spectral_coherence(forecast: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[List[float], List[float]]:
        H, W = forecast.shape
        f_fft = torch.fft.fftshift(torch.fft.fft2(forecast))
        g_fft = torch.fft.fftshift(torch.fft.fft2(ground_truth))
        
        csd = f_fft * torch.conj(g_fft)
        psd_f = torch.abs(f_fft) ** 2
        psd_g = torch.abs(g_fft) ** 2
        
        y = torch.arange(H, dtype=torch.float32, device=forecast.device) - H // 2
        x = torch.arange(W, dtype=torch.float32, device=forecast.device) - W // 2
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(grid_y**2 + grid_x**2)
        r_int = torch.round(r).to(torch.long)
        max_r = int(min(H, W) // 2)
        
        coherence = []
        wavenumbers = []
        for k in range(0, max_r + 1):
            mask = (r_int == k)
            if torch.any(mask):
                num = torch.abs(torch.mean(csd[mask])) ** 2
                den = torch.mean(psd_f[mask]) * torch.mean(psd_g[mask]) + 1e-8
                coherence.append((num / den).item())
                wavenumbers.append(float(k))
                
        return wavenumbers, coherence

    @staticmethod
    def compute_gradient_errors(forecast: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        dy_f, dx_f = torch.gradient(forecast)
        dy_g, dx_g = torch.gradient(ground_truth)
        
        mag_f = torch.sqrt(dx_f**2 + dy_f**2)
        mag_g = torch.sqrt(dx_g**2 + dy_g**2)
        
        mag_mae = torch.mean(torch.abs(mag_f - mag_g)).item()
        
        angle_f = torch.atan2(dy_f, dx_f)
        angle_g = torch.atan2(dy_g, dx_g)
        
        diff_angle = torch.abs(angle_f - angle_g)
        diff_angle = torch.where(diff_angle > np.pi, 2 * np.pi - diff_angle, diff_angle)
        
        mean_angular_error = torch.mean(diff_angle).item()
        
        return {
            "gradient_magnitude_mae": mag_mae,
            "gradient_direction_mae_rad": mean_angular_error,
            "gradient_direction_mae_deg": np.degrees(mean_angular_error)
        }

    @staticmethod
    def compute_wavelet_energy(field: PhysicalField, levels: int = 3) -> Dict[str, Any]:
        coeffs = SpectralTransformEngine.apply_dwt2d(field, levels=levels)
        energy = {}
        
        ll_energy = torch.sum(coeffs["LL"] ** 2).item()
        energy["LL"] = ll_energy
        
        total_energy = ll_energy
        for lvl in range(1, levels + 1):
            level_coeffs = coeffs[f"level_{lvl}"]
            lvl_energy = {}
            for subband in ["LH", "HL", "HH"]:
                sub_energy = torch.sum(level_coeffs[subband] ** 2).item()
                lvl_energy[subband] = sub_energy
                total_energy += sub_energy
            energy[f"level_{lvl}"] = lvl_energy
            
        energy["total"] = total_energy
        return energy

    @classmethod
    def compute_diagnostics(cls, forecast: PhysicalField, ground_truth: PhysicalField) -> Dict[str, Any]:
        f_data = forecast.data
        g_data = ground_truth.data
        
        if f_data.shape != g_data.shape:
            raise ValueError(f"Forecast and Ground Truth shapes must match. Got {f_data.shape} and {g_data.shape}")
            
        mse = torch.mean((f_data - g_data) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(f_data - g_data)).item()
        bias = torch.mean(f_data - g_data).item()
        
        ssim = PerturbationEngine.compute_ssim(f_data, g_data)
        
        f_flat = f_data.flatten()
        g_flat = g_data.flatten()
        f_mean = torch.mean(f_flat)
        g_mean = torch.mean(g_flat)
        cov = torch.mean((f_flat - f_mean) * (g_flat - g_mean))
        std_f = torch.std(f_flat)
        std_g = torch.std(g_flat)
        correlation = (cov / (std_f * std_g + 1e-8)).item()
        
        spatial_metrics = {
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
            "bias": bias,
            "structural_similarity_index": ssim,
            "pearson_correlation": correlation
        }
        
        grad_errors = cls.compute_gradient_errors(f_data, g_data)
        
        wn_f, psd_f = cls.compute_radial_psd(f_data)
        _, psd_g = cls.compute_radial_psd(g_data)
        _, coherence = cls.compute_spectral_coherence(f_data, g_data)
        
        # Fit turbulence spectral slopes (Kolmogorov 5/3 vs Charney 3.0 enstrophy)
        k_limit = float(min(f_data.shape) // 2)
        forecast_slope = cls.fit_spectral_slope(wn_f, psd_f, k_min=2.0, k_max=k_limit)
        gt_slope = cls.fit_spectral_slope(wn_f, psd_g, k_min=2.0, k_max=k_limit)
        
        spectral_diagnostics = {
            "wavenumbers": wn_f,
            "forecast_psd": psd_f,
            "ground_truth_psd": psd_g,
            "spectral_coherence": coherence,
            "forecast_slope_analysis": forecast_slope,
            "ground_truth_slope_analysis": gt_slope
        }
        
        min_dim = min(f_data.shape)
        levels = min(3, int(np.log2(min_dim)) - 1)
        levels = max(1, levels)
        
        forecast_energy = cls.compute_wavelet_energy(forecast, levels=levels)
        ground_truth_energy = cls.compute_wavelet_energy(ground_truth, levels=levels)
        
        wavelet_energy = {
            "levels": levels,
            "forecast_energy": forecast_energy,
            "ground_truth_energy": ground_truth_energy
        }
        
        return {
            "spatial_metrics": spatial_metrics,
            "gradient_errors": grad_errors,
            "spectral_diagnostics": spectral_diagnostics,
            "wavelet_energy": wavelet_energy
        }
        
    @staticmethod
    def fit_spectral_slope(wavenumbers: List[float], psd: List[float], k_min: float = 2.0, k_max: float = 12.0) -> Dict[str, Any]:
        """
        Fits a power-law E(k) = C * k^(-beta) in log-log space to estimate the spectral energy cascade slope.
        ln(E(k)) = ln(C) - beta * ln(k)
        """
        k_arr = np.array(wavenumbers)
        psd_arr = np.array(psd)
        
        # Filter indices within band and where values are valid for log
        valid_mask = (k_arr >= k_min) & (k_arr <= k_max) & (k_arr > 0) & (psd_arr > 0)
        k_fit = k_arr[valid_mask]
        psd_fit = psd_arr[valid_mask]
        
        if len(k_fit) < 3:
            return {
                "slope_beta": 0.0,
                "intercept_ln_c": 0.0,
                "r_squared": 0.0,
                "regime_interpretation": "Insufficient data points for stable fit"
            }
            
        ln_k = np.log(k_fit)
        ln_psd = np.log(psd_fit)
        
        # Fit linear polynomial: y = m*x + c
        # Here y = ln_psd, x = ln_k, so m = -beta, c = ln_C
        slope, intercept = np.polyfit(ln_k, ln_psd, 1)
        beta = -slope
        
        # Compute R-squared
        residuals = ln_psd - (slope * ln_k + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ln_psd - np.mean(ln_psd))**2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        # Interpretation
        if abs(beta - 3.0) < 0.4:
            interpretation = "Charney/Kraichnan 2D Enstrophy Cascade (beta ~ 3.0)"
        elif abs(beta - 1.67) < 0.3:
            interpretation = "Kolmogorov 3D/Mesoscale Kinetic Energy Cascade (beta ~ 5/3)"
        elif beta < 1.0:
            interpretation = "Flat Spectrum (Likely Dominated by Noise/Artifacts)"
        else:
            interpretation = "Mixed/Intermediate Turbulent Cascade"
            
        return {
            "slope_beta": float(beta),
            "intercept_ln_c": float(intercept),
            "r_squared": float(r_squared),
            "regime_interpretation": interpretation
        }
