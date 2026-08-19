import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.physical_core.field import PhysicalField

class BoundaryConditionLab:
    """
    Applies periodic, zero-padding, reflection, and windowing treatments to regional fields
    and quantifies boundary artefacts as a function of distance.
    """
    @staticmethod
    def apply_window(data: torch.Tensor, window_type: str = "tukey", alpha: float = 0.1) -> torch.Tensor:
        H, W = data.shape
        
        def get_1d_window(N: int) -> torch.Tensor:
            if N <= 1:
                return torch.ones(N, dtype=torch.float32)
            n = torch.arange(N, dtype=torch.float32)
            if window_type == "hann":
                return 0.5 * (1.0 - torch.cos(2 * np.pi * n / (N - 1)))
            elif window_type == "hamming":
                return 0.54 - 0.46 * torch.cos(2 * np.pi * n / (N - 1))
            elif window_type == "tukey":
                w = torch.ones(N, dtype=torch.float32)
                if alpha <= 0:
                    return w
                if alpha >= 1:
                    return 0.5 * (1.0 - torch.cos(2 * np.pi * n / (N - 1)))
                
                # Vectorised, and uses numpy for the scalar taper: torch.cos()
                # rejects Python floats (the previous per-element loop raised TypeError).
                limit = int(alpha * (N - 1) / 2)
                idx = torch.arange(limit + 1, dtype=torch.float32)
                taper = 0.5 * (1.0 + torch.cos(
                    torch.tensor(np.pi, dtype=torch.float32) * (2.0 * idx / (alpha * (N - 1)) - 1.0)
                ))
                w[: limit + 1] = taper
                w[N - 1 - limit:] = torch.flip(taper, dims=[0])
                return w
            else:
                return torch.ones(N, dtype=torch.float32)
                
        w_y = get_1d_window(H).to(data.device)
        w_x = get_1d_window(W).to(data.device)
        window_2d = torch.outer(w_y, w_x)
        return data * window_2d

    @classmethod
    def apply_boundary_treatment(
        cls,
        field: PhysicalField,
        treatment: str,
        pad_width: int,
        window_type: Optional[str] = None,
        window_alpha: float = 0.1
    ) -> torch.Tensor:
        data = field.data
        
        if window_type is not None:
            data = cls.apply_window(data, window_type, window_alpha)
            
        treatment = treatment.lower()
        data_4d = data.unsqueeze(0).unsqueeze(0)
        if treatment == "periodic":
            padded_4d = torch.nn.functional.pad(data_4d, (pad_width, pad_width, pad_width, pad_width), mode="circular")
        elif treatment == "zero":
            padded_4d = torch.nn.functional.pad(data_4d, (pad_width, pad_width, pad_width, pad_width), mode="constant", value=0.0)
        elif treatment == "reflect":
            padded_4d = torch.nn.functional.pad(data_4d, (pad_width, pad_width, pad_width, pad_width), mode="reflect")
        elif treatment == "replicate":
            padded_4d = torch.nn.functional.pad(data_4d, (pad_width, pad_width, pad_width, pad_width), mode="replicate")
        else:
            raise ValueError(f"Unsupported boundary treatment: {treatment}")
            
        return padded_4d.squeeze(0).squeeze(0)

    @classmethod
    def analyze_boundary_artefacts(
        cls,
        field: PhysicalField,
        treatment: str,
        pad_width: int,
        window_type: Optional[str] = None,
        window_alpha: float = 0.1,
        reference_field: Optional[PhysicalField] = None
    ) -> Dict[str, Any]:
        padded_data = cls.apply_boundary_treatment(field, treatment, pad_width, window_type, window_alpha)
        
        H, W = field.data.shape
        padded_H, padded_W = padded_data.shape
        
        y_indices = torch.arange(padded_H, dtype=torch.float32, device=padded_data.device)
        x_indices = torch.arange(padded_W, dtype=torch.float32, device=padded_data.device)
        grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing="ij")
        
        y_min, y_max = pad_width, pad_width + H - 1
        x_min, x_max = pad_width, pad_width + W - 1
        
        dist_left = grid_x - x_min
        dist_right = x_max - grid_x
        dist_top = grid_y - y_min
        dist_bottom = y_max - grid_y
        
        dist_inside = torch.stack([dist_left, dist_right, dist_top, dist_bottom], dim=0).min(dim=0)[0]
        dist_outside_x = torch.clamp(x_min - grid_x, min=0) + torch.clamp(grid_x - x_max, min=0)
        dist_outside_y = torch.clamp(y_min - grid_y, min=0) + torch.clamp(grid_y - y_max, min=0)
        dist_outside = torch.sqrt(dist_outside_x**2 + dist_outside_y**2)
        
        dist_to_boundary = torch.where(dist_inside >= 0, dist_inside, dist_outside)
        
        # D13 note, and a deliberate exception to it. This gradient stays in *pixel* units
        # and that is the correct frame here: a boundary artefact is a property of the
        # padding operation and the grid, not of the atmosphere, and the distance profile
        # it feeds is binned in cells from the padded edge. Converting to K/m would divide
        # every profile by a constant (or, on a lat/lon grid, by a latitude-dependent
        # factor) without changing which distances show elevated gradients - while making
        # the artefact's magnitude harder to compare across treatments.
        #
        # What was wrong before was not the choice but its invisibility: the units are now
        # stated in the returned record, so a reader cannot mistake these for physical
        # gradients. `gradient_units` is asserted in test_boundary_synthetic.
        dy, dx = torch.gradient(padded_data.to(torch.float64), edge_order=1)
        grad_mag = torch.sqrt(dx**2 + dy**2)
        
        has_ref = reference_field is not None
        if has_ref:
            ref_data = reference_field.data
            if ref_data.shape != padded_data.shape:
                ref_field_scaled = reference_field.scale_resolution((padded_H, padded_W))
                ref_data = ref_field_scaled.data
            abs_error = torch.abs(padded_data - ref_data)
        else:
            abs_error = torch.zeros_like(padded_data)
            
        max_dist = int(torch.max(dist_to_boundary).item())
        distance_profiles = []
        
        for d in range(0, max_dist + 1):
            mask = (dist_to_boundary >= d) & (dist_to_boundary < d + 1)
            if torch.any(mask):
                mean_grad = torch.mean(grad_mag[mask]).item()
                max_grad = torch.max(grad_mag[mask]).item()
                mean_err = torch.mean(abs_error[mask]).item() if has_ref else 0.0
                max_err = torch.max(abs_error[mask]).item() if has_ref else 0.0
                
                distance_profiles.append({
                    "distance": float(d),
                    "mean_gradient": mean_grad,
                    "max_gradient": max_grad,
                    "mean_absolute_error": mean_err,
                    "max_absolute_error": max_err
                })
                
        orig_fft = torch.abs(torch.fft.rfft2(field.data))
        padded_fft = torch.abs(torch.fft.rfft2(padded_data))
        
        orig_fft_norm = orig_fft / (torch.sum(orig_fft) + 1e-8)
        padded_fft_norm = padded_fft / (torch.sum(padded_fft) + 1e-8)
        
        orig_cutoff = int(orig_fft_norm.shape[0] * 0.7)
        padded_cutoff = int(padded_fft_norm.shape[0] * 0.7)
        
        orig_high_freq_energy = torch.sum(orig_fft_norm[orig_cutoff:]).item()
        padded_high_freq_energy = torch.sum(padded_fft_norm[padded_cutoff:]).item()
        
        spectral_leakage = padded_high_freq_energy / (orig_high_freq_energy + 1e-8)
        
        return {
            "padded_field": padded_data.tolist(),
            "distance_profiles": distance_profiles,
            "gradient_units": "value per pixel",
            "distance_units": "pixel",
            "gradient_frame_note": (
                "Boundary-artefact gradients are intentionally in pixel units: an artefact "
                "is a property of the padding and the grid, so the pixel frame is the one "
                "in which it is defined. Use physical_core.operators.gradient for "
                "atmospheric gradients."
            ),
            "grid": field.grid.to_provenance(),
            "spectral_leakage": spectral_leakage,
            "has_reference": has_ref
        }