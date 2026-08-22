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
                return torch.ones(N, dtype=torch.float64)
            n = torch.arange(N, dtype=torch.float64)
            if window_type == "hann":
                return 0.5 * (1.0 - torch.cos(2 * np.pi * n / (N - 1)))
            elif window_type == "hamming":
                return 0.54 - 0.46 * torch.cos(2 * np.pi * n / (N - 1))
            elif window_type == "tukey":
                w = torch.ones(N, dtype=torch.float64)
                if alpha <= 0:
                    return w
                if alpha >= 1:
                    return 0.5 * (1.0 - torch.cos(2 * np.pi * n / (N - 1)))
                
                # Vectorised, and uses numpy for the scalar taper: torch.cos()
                # rejects Python floats (the previous per-element loop raised TypeError).
                limit = int(alpha * (N - 1) / 2)
                idx = torch.arange(limit + 1, dtype=torch.float64)
                taper = 0.5 * (1.0 + torch.cos(
                    torch.tensor(np.pi, dtype=torch.float64) * (2.0 * idx / (alpha * (N - 1)) - 1.0)
                ))
                w[: limit + 1] = taper
                w[N - 1 - limit:] = torch.flip(taper, dims=[0])
                return w
            else:
                return torch.ones(N, dtype=torch.float64)
                
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
        
        y_indices = torch.arange(padded_H, dtype=torch.float64, device=padded_data.device)
        x_indices = torch.arange(padded_W, dtype=torch.float64, device=padded_data.device)
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
            
        # Group every pixel once by floor(distance), then reduce all rings together.  The
        # old loop allocated and scanned a full padded-field mask for every distance, an
        # O(number_of_rings * H * W) path inside boundary sweeps.  Means use ``bincount``;
        # maxima use the matching grouped scatter reduction.  The distance construction
        # above is unchanged, including Euclidean distance around padded corners.
        labels = torch.floor(dist_to_boundary).to(torch.int64).reshape(-1)
        n_bins = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=n_bins)

        flat_grad = grad_mag.reshape(-1)
        gradient_sum = torch.bincount(labels, weights=flat_grad, minlength=n_bins)
        gradient_max = torch.full(
            (n_bins,), -torch.inf, dtype=flat_grad.dtype, device=flat_grad.device)
        gradient_max.scatter_reduce_(0, labels, flat_grad, reduce="amax", include_self=True)

        if has_ref:
            flat_error = abs_error.reshape(-1)
            error_sum = torch.bincount(labels, weights=flat_error, minlength=n_bins)
            error_max = torch.full(
                (n_bins,), -torch.inf, dtype=flat_error.dtype, device=flat_error.device)
            error_max.scatter_reduce_(0, labels, flat_error, reduce="amax", include_self=True)
        else:
            error_sum = torch.zeros(n_bins, dtype=flat_grad.dtype, device=flat_grad.device)
            error_max = torch.zeros(n_bins, dtype=flat_grad.dtype, device=flat_grad.device)

        valid = torch.nonzero(counts, as_tuple=False).flatten()
        gradient_mean = gradient_sum[valid] / counts[valid]
        error_mean = error_sum[valid] / counts[valid]
        distance_profiles = [{
            "distance": float(distance.item()),
            "mean_gradient": float(gradient_mean[index].item()),
            "max_gradient": float(gradient_max[distance].item()),
            "mean_absolute_error": float(error_mean[index].item()),
            "max_absolute_error": float(error_max[distance].item()),
        } for index, distance in enumerate(valid)]
                
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
