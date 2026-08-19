import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

class PhysicalField:
    """
    Domain-independent physical-field abstraction representing a 2D spatial grid.
    Manages coordinate mappings, multi-resolution scaling, and train/val/test split guardrails.
    """
    def __init__(
        self,
        data: torch.Tensor,  # Shape: (H, W)
        coords: Optional[Dict[str, torch.Tensor]] = None,  # e.g., {"lat": ..., "lon": ...} or {"x": ..., "y": ...}
        metadata: Optional[Dict[str, Any]] = None,
        split: Optional[str] = None  # "train", "val", "test"
    ):
        if len(data.shape) != 2:
            raise ValueError(f"PhysicalField data must be 2D. Got shape {data.shape}")
        
        self.data = data.float()
        self.height, self.width = data.shape
        
        # Default coordinates if not provided
        if coords is None:
            x = torch.linspace(0.0, 1.0, self.width, device=data.device)
            y = torch.linspace(0.0, 1.0, self.height, device=data.device)
            self.coords = {"x": x, "y": y}
        else:
            self.coords = coords
            
        self.metadata = metadata or {}
        self.split = split  # Guardrail tracking

    def scale_resolution(self, target_shape: Tuple[int, int], mode: str = "bilinear") -> "PhysicalField":
        """
        Scales the physical field to a target resolution using spatial interpolation.
        """
        # Add batch and channel dimensions for interpolate: (1, 1, H, W)
        temp_tensor = self.data.unsqueeze(0).unsqueeze(0)
        scaled_tensor = torch.nn.functional.interpolate(
            temp_tensor, size=target_shape, mode=mode, align_corners=True if mode != "nearest" else None
        )
        scaled_data = scaled_tensor.squeeze(0).squeeze(0)
        
        # Interpolate coordinates
        new_coords = {}
        for key, val in self.coords.items():
            if len(val.shape) == 1:
                new_len = target_shape[0] if key in ["y", "lat"] else target_shape[1]
                temp_coord = val.unsqueeze(0).unsqueeze(0)
                scaled_coord = torch.nn.functional.interpolate(
                    temp_coord, size=(new_len,), mode="linear", align_corners=True
                )
                new_coords[key] = scaled_coord.squeeze(0).squeeze(0)
            else:
                temp_coord = val.unsqueeze(0).unsqueeze(0)
                scaled_coord = torch.nn.functional.interpolate(
                    temp_coord, size=target_shape, mode=mode, align_corners=True if mode != "nearest" else None
                )
                new_coords[key] = scaled_coord.squeeze(0).squeeze(0)
                
        return PhysicalField(scaled_data, coords=new_coords, metadata=self.metadata, split=self.split)

    def split_field(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Dict[str, "PhysicalField"]:
        """
        Splits the field into train, val, and test regions along the X-axis (or Y-axis)
        with strict guardrails to prevent data leakage.
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("Sum of train_ratio and val_ratio must be less than 1.0")
            
        w = self.width
        train_end = int(w * train_ratio)
        val_end = int(w * (train_ratio + val_ratio))
        
        train_data = self.data[:, :train_end]
        val_data = self.data[:, train_end:val_end]
        test_data = self.data[:, val_end:]
        
        def slice_coords(start: int, end: int) -> Dict[str, torch.Tensor]:
            sliced = {}
            for k, v in self.coords.items():
                if len(v.shape) == 1:
                    if k in ["x", "lon"]:
                        sliced[k] = v[start:end]
                    else:
                        sliced[k] = v.clone()
                else:
                    sliced[k] = v[:, start:end]
            return sliced

        return {
            "train": PhysicalField(train_data, coords=slice_coords(0, train_end), metadata=self.metadata, split="train"),
            "val": PhysicalField(val_data, coords=slice_coords(train_end, val_end), metadata=self.metadata, split="val"),
            "test": PhysicalField(test_data, coords=slice_coords(val_end, w), metadata=self.metadata, split="test")
        }

    def validate_split_guardrails(self, other: "PhysicalField") -> bool:
        """
        Enforces strict guardrails: ensures no spatial overlap or data leakage between splits.
        """
        if self.split is None or other.split is None:
            return True
        if self.split == other.split:
            return True
            
        for k in self.coords:
            if k in other.coords:
                v1 = self.coords[k]
                v2 = other.coords[k]
                if len(v1.shape) == 1 and len(v2.shape) == 1:
                    min1, max1 = v1.min().item(), v1.max().item()
                    min2, max2 = v2.min().item(), v2.max().item()
                    if not (max1 < min2 or max2 < min1):
                        raise ValueError(
                            f"Data Leakage Guardrail Violated! Split '{self.split}' overlaps with split '{other.split}' "
                            f"on coordinate '{k}' (Interval 1: [{min1}, {max1}], Interval 2: [{min2}, {max2}])"
                        )
        return True