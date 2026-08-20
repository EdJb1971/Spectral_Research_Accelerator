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

        # dtype follows the *data*, not a literal. A hard-coded float32 theta raised
        # "expected scalar type Double but found Float" the moment the API stopped
        # downcasting incoming fields (D36) - `grid_sample` requires both to match. Deriving
        # it from the input is also the only version that keeps working on a float16 or
        # bfloat16 device later.
        theta = torch.tensor([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty]
        ], dtype=data.dtype, device=data.device).unsqueeze(0)

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
    def add_noise(
        field: PhysicalField,
        noise_type: str = "gaussian",
        level: float = 0.1,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PhysicalField:
        """Add noise to a field, reproducibly when asked.

        Defect D12 / roadmap T3.5.12 / standard E4. This previously used `torch.randn_like`,
        which draws from the *global* torch RNG. Two consequences, both bad for a research
        platform: a perturbation experiment could not be re-run to the same numbers from its
        lineage record, and running the same sweep under a thread or process executor
        (T3.5.19) would give different answers depending on interleaving - a result that
        depends on the scheduler is not a result.

        Pass ``seed`` for a reproducible draw, or ``generator`` to thread an existing stream
        through (which is what lets a sweep derive one independent substream per run via
        `SeedSequence.spawn` rather than reusing one global state). Passing neither keeps
        the old global-RNG behaviour, and the returned metadata records ``seeded: False`` so
        an unreproducible run is visibly labelled rather than silently indistinguishable
        from a reproducible one.
        """
        if seed is not None and generator is not None:
            raise ValueError(
                "pass either seed or generator, not both - two sources of randomness would "
                "make it ambiguous which one produced the result")

        data = field.data.clone()
        std_val = torch.std(data).item() if torch.std(data).item() > 0 else 1.0

        gen = generator
        if seed is not None:
            gen = torch.Generator(device=data.device)
            gen.manual_seed(int(seed))

        def _randn():
            return torch.randn(data.shape, generator=gen, dtype=data.dtype,
                               device=data.device) if gen is not None else torch.randn_like(data)

        def _rand():
            return torch.rand(data.shape, generator=gen, dtype=data.dtype,
                              device=data.device) if gen is not None else torch.rand_like(data)

        if noise_type == "gaussian":
            perturbed_data = data + _randn() * (level * std_val)
        elif noise_type == "uniform":
            perturbed_data = data + (_rand() - 0.5) * 2.0 * (level * std_val)
        elif noise_type == "salt_pepper":
            perturbed_data = data.clone()
            mask = _rand()
            perturbed_data[mask < (level / 2.0)] = torch.min(data)
            perturbed_data[mask > (1.0 - level / 2.0)] = torch.max(data)
        else:
            raise ValueError(
                f"Unsupported noise type: {noise_type}. "
                "Expected 'gaussian', 'uniform' or 'salt_pepper'.")

        return PhysicalField(
            perturbed_data, coords=field.coords, grid=field.grid,
            metadata={**field.metadata, "perturbation": "noise", "noise_type": noise_type,
                      "level": level, "seed": seed,
                      "seeded": seed is not None or generator is not None})

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