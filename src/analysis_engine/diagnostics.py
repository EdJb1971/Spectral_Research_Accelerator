import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core import operators
from src.analysis_engine import spectra
from src.transform_engine.transforms import SpectralTransformEngine
from src.synthetic_generator.perturbation import PerturbationEngine

class SpectralSpatialAnalysisEngine:
    """
    Calculates diagnostics including power spectral density, wavelet energy, SSIM, and gradient errors.
    """
    @staticmethod
    def compute_radial_psd(
        data,
        grid: Optional[GridSpec] = None,
        units: str = "rad_per_m",
        convention: str = "energy_1d",
    ) -> Tuple[List[float], List[float]]:
        """Isotropically-averaged power spectrum. Delegates to `analysis_engine.spectra`.

        Defects D13 and D26. The wavenumber axis is now physical (metres, not pixel index)
        and the returned power is the 1D energy spectrum E(k) by default - the convention in
        which Kolmogorov is 5/3. Pass ``units="legacy_pixel"`` to reproduce the previous
        integer-radius, S(k)-in-pixel-units behaviour exactly.

        On a pixel grid ``rad_per_m`` degrades gracefully to radians per pixel, which is
        well-defined; ``spectra.radial_power_spectrum`` returns the unit label so a plot
        axis cannot be mislabelled.
        """
        record = spectra.radial_power_spectrum(data, grid=grid, units=units,
                                               convention=convention)
        return record["k"], record["power"]

    @staticmethod
    def compute_radial_psd_full(data, grid: Optional[GridSpec] = None,
                                **kwargs: Any) -> Dict[str, Any]:
        """Full spectral record: k, power, counts, units, isotropy, variance captured."""
        return spectra.radial_power_spectrum(data, grid=grid, **kwargs)

    @staticmethod
    def compute_spectral_coherence(
        forecast: torch.Tensor,
        ground_truth: torch.Tensor,
        grid: Optional[GridSpec] = None,
        units: str = "rad_per_m",
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[List[float], List[float]]:
        """Magnitude-squared coherence per wavenumber band, on a physical k axis.

        Defect D17: the annulus reduction is now `torch.bincount` passes rather than a
        Python loop over bins, so cost no longer scales with the bin count.

        Interpretation caveat, stated because it is easy to over-read: with a *single* field
        pair each Fourier mode gives one cross-spectral sample, and single-sample coherence
        is identically 1 by construction. Coherence is only meaningful once averaged over
        many modes, which is what the annulus average supplies - so the first few bands,
        where an annulus holds few modes, are biased high. Mode counts for the same binning
        come from `compute_radial_psd_full`.
        """
        if forecast.shape != ground_truth.shape:
            raise ValueError(
                "coherence needs matching shapes, got %r and %r"
                % (tuple(forecast.shape), tuple(ground_truth.shape))
            )
        gspec = grid if grid is not None else GridSpec.pixel(tuple(forecast.shape))
        H, W = forecast.shape
        f = forecast.to(dtype)
        g = ground_truth.to(dtype)

        f_fft = torch.fft.fftshift(torch.fft.fft2(f - f.mean()))
        g_fft = torch.fft.fftshift(torch.fft.fft2(g - g.mean()))
        csd = f_fft * torch.conj(g_fft)
        psd_f = f_fft.real ** 2 + f_fft.imag ** 2
        psd_g = g_fft.real ** 2 + g_fft.imag ** 2

        if units == "legacy_pixel":
            y = torch.arange(H, dtype=dtype, device=f.device) - H // 2
            x = torch.arange(W, dtype=dtype, device=f.device) - W // 2
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            kmag = torch.sqrt(gy ** 2 + gx ** 2)
            idx = torch.round(kmag).to(torch.long)
            n_bins = int(min(H, W) // 2) + 1
            valid = idx < n_bins
        else:
            kmag = gspec.wavenumber_magnitude(units="rad_per_m", device=f.device,
                                              dtype=dtype, shifted=True)
            n_bins = max(int(min(H, W) // 2), 2)
            dk = gspec.isotropic_k_max("rad_per_m") / n_bins
            idx = torch.floor(kmag / dk).to(torch.long)
            valid = (idx < n_bins) & (kmag > 0)

        flat = idx[valid].reshape(-1)
        counts = torch.bincount(flat, minlength=n_bins).to(dtype)
        csd_re = torch.bincount(flat, weights=csd.real[valid].reshape(-1), minlength=n_bins)
        csd_im = torch.bincount(flat, weights=csd.imag[valid].reshape(-1), minlength=n_bins)
        sf = torch.bincount(flat, weights=psd_f[valid].reshape(-1), minlength=n_bins)
        sg = torch.bincount(flat, weights=psd_g[valid].reshape(-1), minlength=n_bins)
        kk = torch.bincount(flat, weights=kmag[valid].reshape(-1), minlength=n_bins)

        occupied = counts > 0
        c = counts[occupied]
        num = (csd_re[occupied] ** 2 + csd_im[occupied] ** 2) / c ** 2
        den = (sf[occupied] / c) * (sg[occupied] / c)
        coh = torch.where(den > 0, num / den, torch.zeros_like(num))
        k_out = kk[occupied] / c

        if units not in ("legacy_pixel", "rad_per_m"):
            rad = gspec.wavenumber_axes("rad_per_m", shifted=False)[1].abs().max()
            tgt = gspec.wavenumber_axes(units, shifted=False)[1].abs().max()
            k_out = k_out * float(tgt / rad)

        return [float(v) for v in k_out], [float(v) for v in coh]

    @staticmethod
    def compute_gradient_errors(
        forecast: torch.Tensor,
        ground_truth: torch.Tensor,
        grid: Optional[GridSpec] = None,
    ) -> Dict[str, Any]:
        """Gradient magnitude and direction error, in physical units where available.

        Defect D13. Previously both gradients came from `torch.gradient` at unit spacing,
        which on a lat/lon grid is wrong by a latitude-dependent factor and so distorts
        direction as well as magnitude. Now the grid metric is applied, the statistics are
        area-weighted, and the first-order edge ring is excluded via ``valid_mask`` so the
        reported number describes the second-order interior.

        The angular error also uses a proper circular difference wrapped to [0, pi]. The
        previous version folded `atan2` outputs with a single ``> pi`` test, which
        mishandles pairs straddling the branch cut.
        """
        gspec = grid if grid is not None else GridSpec.pixel(tuple(forecast.shape))
        gf = operators.gradient(forecast, gspec)
        gg = operators.gradient(ground_truth, gspec)
        mask = gf["valid_mask"]

        mag_f, mag_g = gf["magnitude"], gg["magnitude"]
        weights = gspec.area_weights(device=mag_f.device, dtype=mag_f.dtype) * mask
        total = torch.sum(weights)
        if float(total) <= 0:
            raise ValueError(
                "gradient error needs a second-order interior, but a %r field has none. "
                "Use a field at least 3x3." % (tuple(forecast.shape),)
            )
        weights = weights / total

        mag_mae = float(torch.sum(weights * torch.abs(mag_f - mag_g)))

        diff = gf["direction_rad"] - gg["direction_rad"]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()   # wrap to [0, pi]
        mean_ang = float(torch.sum(weights * diff))

        return {
            "gradient_magnitude_mae": mag_mae,
            "gradient_direction_mae_rad": mean_ang,
            "gradient_direction_mae_deg": float(np.degrees(mean_ang)),
            "units": gf["units"],
            "is_physical": gf["is_physical"],
            "area_weighted": True,
            "edge_ring_excluded": True,
            "grid": gf["grid"],
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

        # Defect D13: verification scores are area-weighted. On a lat/lon grid an
        # unweighted mean over-weights poleward rows in proportion to 1/cos(lat), which
        # silently changes the score. Both weighted and unweighted are returned so the size
        # of that effect is visible in the record rather than only in this comment.
        gspec = getattr(forecast, "grid", None) or GridSpec.pixel(tuple(f_data.shape))
        weighted = operators.area_weighted_error_metrics(f_data, g_data, grid=gspec)
        mse = weighted["rmse"] ** 2
        rmse = weighted["rmse"]
        mae = weighted["mae"]
        bias = weighted["bias"]

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
            "pearson_correlation": correlation,
            "root_mean_squared_error_unweighted": weighted["rmse_unweighted"],
            "mean_absolute_error_unweighted": weighted["mae_unweighted"],
            "bias_unweighted": weighted["bias_unweighted"],
            "area_weighted": True,
            "weighting_effect_ratio": weighted["weighting_effect_ratio"],
            "variable_units": weighted["variable_units"],
        }

        grad_errors = cls.compute_gradient_errors(f_data, g_data, grid=gspec)

        # Defects D13 and D26: physical wavenumber axis, 1D energy convention.
        spec_f = cls.compute_radial_psd_full(f_data, grid=gspec)
        spec_g = cls.compute_radial_psd_full(g_data, grid=gspec)
        wn_f, psd_f = spec_f["k"], spec_f["power"]
        psd_g = spec_g["power"]
        _, coherence = cls.compute_spectral_coherence(f_data, g_data, grid=gspec)

        # k_max is the largest wavenumber resolved on *both* axes, not the longer axis's
        # Nyquist: beyond it an annulus is only partly sampled and its mean is biased by
        # the missing sectors. k_min skips the first two bins, which hold too few modes to
        # constrain a log-log slope.
        k_lo = wn_f[min(2, len(wn_f) - 1)] if wn_f else None
        k_hi = spec_f["k_max_isotropic"]
        forecast_slope = cls.fit_spectral_slope(wn_f, psd_f, k_min=k_lo, k_max=k_hi,
                                                counts=spec_f["counts"],
                                                convention=spec_f["convention"])
        gt_slope = cls.fit_spectral_slope(wn_f, psd_g, k_min=k_lo, k_max=k_hi,
                                          counts=spec_g["counts"],
                                          convention=spec_g["convention"])

        spectral_diagnostics = {
            "wavenumbers": wn_f,
            "forecast_psd": psd_f,
            "ground_truth_psd": psd_g,
            "spectral_coherence": coherence,
            "forecast_slope_analysis": forecast_slope,
            "ground_truth_slope_analysis": gt_slope,
            "k_units": spec_f["k_units"],
            "power_units": spec_f["power_units"],
            "convention": spec_f["convention"],
            "convention_note": spec_f["convention_note"],
            "mode_counts": spec_f["counts"],
            "variance_captured_fraction": spec_f["variance_captured_fraction"],
            "isotropy": spec_f["isotropy"],
            "warnings": spec_f["warnings"],
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
            "wavelet_energy": wavelet_energy,
            "grid": gspec.to_provenance(),
        }

    @staticmethod
    def fit_spectral_slope(
        wavenumbers: List[float],
        psd: List[float],
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
        counts: Optional[List[float]] = None,
        convention: str = "energy_1d",
    ) -> Dict[str, Any]:
        """Fit P(k) = C k^-beta in log-log space. Delegates to `spectra.fit_power_law`.

        Defect D26: ``convention`` now decides how the exponent is interpreted, defaulting
        to the 1D energy convention because that is the one in which the reference
        exponents 5/3 and 3 are defined. Keeps the original keys and adds
        ``slope_standard_error``, ``beta_energy_1d``, ``beta_density_2d``, ``n_points`` and
        ``assumptions``. The old ``k_min=2.0`` / ``k_max=12.0`` defaults are gone: they were
        pixel radii and are meaningless on a physical axis.
        """
        return spectra.fit_power_law(wavenumbers, psd, counts=counts, k_min=k_min,
                                     k_max=k_max, convention=convention)
