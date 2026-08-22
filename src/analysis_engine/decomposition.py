import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.physical_core.field import PhysicalField

class ErrorDecompositionEngine:
    """
    Decomposes forecast errors across physical scales, lead times, and distance from boundary.
    """
    @staticmethod
    def decompose_by_scale(
        forecast: torch.Tensor,
        ground_truth: torch.Tensor,
        k_low: Optional[float] = None,
        k_high: Optional[float] = None
    ) -> Dict[str, float]:
        H, W = forecast.shape
        error = forecast - ground_truth
        
        err_fft = torch.fft.fft2(error)
        err_fft_shifted = torch.fft.fftshift(err_fft)
        
        y = torch.arange(H, dtype=torch.float64, device=forecast.device) - H // 2
        x = torch.arange(W, dtype=torch.float64, device=forecast.device) - W // 2
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(grid_y**2 + grid_x**2)
        
        max_r = float(min(H, W) // 2)
        if k_low is None:
            k_low = 0.15 * max_r
        if k_high is None:
            k_high = 0.4 * max_r
            
        mask_low = (r <= k_low)
        mask_mid = (r > k_low) & (r <= k_high)
        mask_high = (r > k_high)
        
        def reconstruct_band(mask):
            filtered_fft = err_fft_shifted * mask
            filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft)
            recon = torch.fft.ifft2(filtered_fft_unshifted).real
            return recon
            
        recon_low = reconstruct_band(mask_low)
        recon_mid = reconstruct_band(mask_mid)
        recon_high = reconstruct_band(mask_high)
        
        rmse_low = torch.sqrt(torch.mean(recon_low**2)).item()
        rmse_mid = torch.sqrt(torch.mean(recon_mid**2)).item()
        rmse_high = torch.sqrt(torch.mean(recon_high**2)).item()
        rmse_total = torch.sqrt(torch.mean(error**2)).item()
        
        return {
            "low_scale_rmse": rmse_low,
            "mid_scale_rmse": rmse_mid,
            "high_scale_rmse": rmse_high,
            "total_rmse": rmse_total
        }

    @staticmethod
    def decompose_by_boundary(
        forecast: torch.Tensor,
        ground_truth: torch.Tensor,
        boundary_width: int = 8
    ) -> List[Dict[str, Any]]:
        H, W = forecast.shape
        error = forecast - ground_truth

        # Every cell belongs to exactly one integer-distance ring.  The former
        # implementation rebuilt a full H x W boolean mask for every ring, making this
        # O(boundary_width * H * W).  Compute the labels once and reduce all rings in one
        # pass instead.  ``bincount`` also avoids materialising four H x W coordinate
        # grids merely to take their minimum.
        y = torch.arange(H, dtype=torch.int64, device=forecast.device)
        x = torch.arange(W, dtype=torch.int64, device=forecast.device)
        y_distance = torch.minimum(y, (H - 1) - y)
        x_distance = torch.minimum(x, (W - 1) - x)
        distance_bins = torch.minimum(y_distance[:, None], x_distance[None, :])

        selected = distance_bins <= int(boundary_width)
        labels = distance_bins[selected].reshape(-1)
        if labels.numel() == 0:
            return []

        values = error[selected].reshape(-1)
        n_bins = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=n_bins)
        squared_sum = torch.bincount(labels, weights=values.square(), minlength=n_bins)
        absolute_sum = torch.bincount(labels, weights=values.abs(), minlength=n_bins)
        valid = torch.nonzero(counts, as_tuple=False).flatten()

        means_squared = squared_sum[valid] / counts[valid]
        means_absolute = absolute_sum[valid] / counts[valid]
        return [{
            "distance": float(distance.item()),
            "rmse": float(torch.sqrt(means_squared[index]).item()),
            "mean_absolute_error": float(means_absolute[index].item()),
        } for index, distance in enumerate(valid)]

    @staticmethod
    def decompose_by_lead_time(
        forecast_series: List[torch.Tensor],
        ground_truth_series: List[torch.Tensor],
        lead_times: List[float]
    ) -> List[Dict[str, Any]]:
        results = []
        for f, g, lt in zip(forecast_series, ground_truth_series, lead_times):
            rmse = torch.sqrt(torch.mean((f - g)**2)).item()
            mae = torch.mean(torch.abs(f - g)).item()
            results.append({
                "lead_time": lt,
                "rmse": rmse,
                "mean_absolute_error": mae
            })
        return results
