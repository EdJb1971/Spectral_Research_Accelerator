import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.physical_core.field import PhysicalField

class SyntheticFieldGenerator:
    """
    Generates deterministic 2D physical fields (sinusoids, vortices, fronts) based on physical parameters.
    """
    @staticmethod
    def generate_sinusoid(
        height: int,
        width: int,
        frequencies: List[Any] = [(2.0, 2.0)],
        amplitudes: List[float] = [1.0],
        phases: List[Any] = [(0.0, 0.0)]
    ) -> PhysicalField:
        y = torch.linspace(0.0, 1.0, height)
        x = torch.linspace(0.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        
        n = len(frequencies)
        if len(amplitudes) < n:
            amplitudes = list(amplitudes) + [1.0] * (n - len(amplitudes))
        if len(phases) < n:
            phases = list(phases) + [(0.0, 0.0)] * (n - len(phases))
            
        data = torch.zeros((height, width), dtype=torch.float32)
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            fx, fy = freq[0], freq[1]
            px, py = phase[0], phase[1]
            term = amp * torch.sin(2 * np.pi * fx * grid_x + px) * torch.cos(2 * np.pi * fy * grid_y + py)
            data += term
            
        coords = {"x": x, "y": y}
        metadata = {
            "type": "sinusoid",
            "frequencies": [list(f) for f in frequencies],
            "amplitudes": amplitudes,
            "phases": [list(p) for p in phases]
        }
        return PhysicalField(data, coords=coords, metadata=metadata)

    @staticmethod
    def generate_vortex(
        height: int,
        width: int,
        centers: List[Any] = [(0.5, 0.5)],
        amplitudes: List[float] = [1.0],
        core_radii: List[float] = [0.1]
    ) -> PhysicalField:
        y = torch.linspace(0.0, 1.0, height)
        x = torch.linspace(0.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        
        n = len(centers)
        if len(amplitudes) < n:
            amplitudes = list(amplitudes) + [1.0] * (n - len(amplitudes))
        if len(core_radii) < n:
            core_radii = list(core_radii) + [0.1] * (n - len(core_radii))
            
        data = torch.zeros((height, width), dtype=torch.float32)
        for center, amp, r_core in zip(centers, amplitudes, core_radii):
            cx, cy = center[0], center[1]
            r2 = (grid_x - cx)**2 + (grid_y - cy)**2
            term = amp * torch.exp(-r2 / (2 * r_core**2))
            data += term
            
        coords = {"x": x, "y": y}
        metadata = {
            "type": "vortex",
            "centers": [list(c) for c in centers],
            "amplitudes": amplitudes,
            "core_radii": core_radii
        }
        return PhysicalField(data, coords=coords, metadata=metadata)

    @staticmethod
    def generate_front(
        height: int,
        width: int,
        angle: float = 0.0,
        offset: float = 0.0,
        width_param: float = 0.1,
        amplitude: float = 1.0
    ) -> PhysicalField:
        y = torch.linspace(0.0, 1.0, height)
        x = torch.linspace(0.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        
        angle_rad = np.deg2rad(angle)
        proj = (grid_x - 0.5) * np.cos(angle_rad) + (grid_y - 0.5) * np.sin(angle_rad) - offset
        data = amplitude * torch.tanh(proj / width_param)
        
        coords = {"x": x, "y": y}
        metadata = {
            "type": "front",
            "angle": angle,
            "offset": offset,
            "width_param": width_param,
            "amplitude": amplitude
        }
        return PhysicalField(data, coords=coords, metadata=metadata)