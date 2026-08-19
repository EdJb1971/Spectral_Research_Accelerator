import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.physical_core.field import PhysicalField

class PerturbationEngine:
    """
    Applies controlled spatial perturbations (rotation, translation, noise) to measure representation sensitivity.
    """
    @staticmethod
    def apply_affine(data: torch.Tensor, angle_degrees: float, tx: float, ty: float) -> torch.Tensor:
        H, W = data.shape
        angle_rad = np.deg2rad(angle_degrees)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty]
        ], dtype=torch.float32, device=data.device).unsqueeze(0)
        
        x = data.unsqueeze(0).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=True)
        out = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out.squeeze(0).squeeze(0)

    @classmethod
    def rotate(cls, field: PhysicalField, angle_degrees: float) -> PhysicalField:
        rotated_data = cls.apply_affine(field.data, angle_degrees, 0.0, 0.0)
        return PhysicalField(rotated_data, coords=field.coords, metadata={**field.metadata, "perturbation": "rotation", "angle": angle_degrees})

    @classmethod
    def translate(cls, field: PhysicalField, shift_x: float, shift_y: float) -> PhysicalField:
        translated_data = cls.apply_affine(field.data, 0.0, shift_x, shift_y)
        return PhysicalField(translated_data, coords=field.coords, metadata={**field.metadata, "perturbation": "translation", "shift": (shift_x, shift_y)})

    @staticmethod
    def add_noise(field: PhysicalField, noise_type: str = "gaussian", level: float = 0.1) -> PhysicalField:
        data = field.data.clone()
        std_val = torch.std(data).item() if torch.std(data).item() > 0 else 1.0
        
        if noise_type == "gaussian":
            noise = torch.randn_like(data) * (level * std_val)
            perturbed_data = data + noise
        elif noise_type == "uniform":
            noise = (torch.rand_like(data) - 0.5) * 2.0 * (level * std_val)
            perturbed_data = data + noise
        elif noise_type == "salt_pepper":
            perturbed_data = data.clone()
            mask = torch.rand_like(data)
            perturbed_data[mask < (level / 2.0)] = torch.min(data)
            perturbed_data[mask > (1.0 - level / 2.0)] = torch.max(data)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
            
        return PhysicalField(perturbed_data, coords=field.coords, metadata={**field.metadata, "perturbation": "noise", "noise_type": noise_type, "level": level})

    @staticmethod
    def compute_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        var_x = torch.var(x)
        var_y = torch.var(y)
        cov_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        L = torch.max(x).item() - torch.min(x).item()
        if L == 0:
            L = 1.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
        return (num / den).item()

    @classmethod
    def compute_sensitivity_metrics(cls, original: PhysicalField, perturbed: PhysicalField) -> Dict[str, float]:
        orig = original.data
        pert = perturbed.data
        
        mse = torch.mean((orig - pert) ** 2).item()
        rmse = np.sqrt(mse)
        
        max_val = torch.max(orig).item()
        min_val = torch.min(orig).item()
        range_val = max_val - min_val if max_val > min_val else 1.0
        psnr = 20 * np.log10(range_val / (rmse + 1e-8))
        
        ssim = cls.compute_ssim(orig, pert)
        
        orig_fft = torch.abs(torch.fft.rfft2(orig))
        pert_fft = torch.abs(torch.fft.rfft2(pert))
        spectral_mse = torch.mean((orig_fft - pert_fft) ** 2).item()
        
        return {
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "peak_signal_to_noise_ratio": psnr,
            "structural_similarity_index": ssim,
            "spectral_energy_shift": spectral_mse
        }