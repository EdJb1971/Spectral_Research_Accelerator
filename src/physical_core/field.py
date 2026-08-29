import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.core.axes import resolve_axis_roles
from src.physical_core.grid import GridSpec

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
        split: Optional[str] = None,  # "train", "val", "test"
        dtype: Optional[torch.dtype] = None,
        grid: Optional[GridSpec] = None,
        units: Optional[str] = None,
        # TG1.1: which coordinate is the row and which the column, when the caller knows.
        # Without it the names are consulted, exactly as before; with it they are labels.
        axis_roles: Optional[Dict[str, str]] = None
    ):
        if len(data.shape) != 2:
            raise ValueError(f"PhysicalField data must be 2D. Got shape {data.shape}")

        # Defect D16: this previously called data.float() unconditionally, silently
        # downcasting float64 input to float32. That is fine for display but it caps the
        # precision of everything downstream - it limited a tight-frame validation to
        # ~3e-7 relative error, which is float32 epsilon rather than any property of the
        # transform under test. Phase 4C surrogate statistics, log-log power-law fits and
        # mutual-information estimation all need float64 to be available.
        #
        # Policy: honour an explicit dtype; otherwise preserve a floating input dtype and
        # promote non-floating input (int, bool) to float32.
        if dtype is not None:
            self.data = data.to(dtype)
        elif data.is_floating_point():
            self.data = data
        else:
            self.data = data.float()
        self.height, self.width = data.shape
        
        # Default coordinates if not provided
        if coords is None:
            x = torch.linspace(0.0, 1.0, self.width, device=data.device)
            y = torch.linspace(0.0, 1.0, self.height, device=data.device)
            self.coords = {"x": x, "y": y}
        else:
            self.coords = coords
            
        self.axis_roles = dict(axis_roles) if axis_roles else None
        self.metadata = metadata or {}
        if units is not None:
            self.metadata.setdefault("units", units)
        self.split = split  # Guardrail tracking

        # Defect D13: physical geometry is now a first-class attribute rather than an
        # unstated assumption. The default is a *pixel* grid, deliberately not None:
        # "lengths here are array indices" then travels with the data and is printed in
        # the units field of every derived quantity, instead of being rediscovered when
        # someone wonders why a gradient is off by a factor of six million.
        if grid is not None:
            if tuple(grid.shape) != tuple(self.data.shape):
                raise ValueError(
                    "grid %s describes shape %r but the field data is %r. A GridSpec "
                    "belongs to one specific array; use grid.subset(...) after a crop or "
                    "grid.resampled(...) after an interpolation."
                    % (grid.describe(), tuple(grid.shape), tuple(self.data.shape))
                )
            self.grid = grid
        else:
            self.grid = GridSpec.from_coords(self.coords, tuple(self.data.shape), self.metadata)

    def _coordinate_axis(self, name: str) -> Optional[str]:
        """``"row"``, ``"column"`` or ``None`` for a 1-D coordinate (TG1.1, standard E14).

        This replaces two hardcoded name lists - ``["y","lat"]`` in `scale_resolution` and
        ``["x","lon"]`` in `split_field` - with one resolution that a caller can override by
        declaring `axis_roles`. Those lists disagreed with each other about what an
        unrecognised coordinate was: the first treated it as a column, the second as a row.
        Both quirks are preserved at the call sites, because a coordinate nobody has named or
        declared is genuinely unknown and changing it silently would move existing results.

        What does change: ``latitude``/``longitude`` (spelled in full, as CF and ERA5 spell
        them) now resolve. Before, `latitude` fell through the ``["y","lat"]`` test and was
        interpolated to the *column* count, and `longitude` fell through ``["x","lon"]`` and
        was cloned rather than sliced by a split - leaving a coordinate vector that no longer
        described its own data. That is defect **D56**, and it is fixed here.
        """
        resolution = resolve_axis_roles(tuple(self.coords), self.axis_roles,
                                        allow_positional_inference=False)
        row, column = resolution.spatial_pair()
        if name == row:
            return "row"
        if name == column:
            return "column"
        return None

    @property
    def units(self) -> Optional[str]:
        """Units of the field *values* (e.g. "K"), distinct from the grid's length units."""
        return self.metadata.get("units", self.grid.variable_units)

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
                # An unresolved coordinate is treated as a column, which is what the
                # hardcoded `key in ["y","lat"]` test did with everything it did not know.
                new_len = (target_shape[0] if self._coordinate_axis(key) == "row"
                           else target_shape[1])
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
                
        return PhysicalField(scaled_data, coords=new_coords, metadata=self.metadata,
                             split=self.split, grid=self.grid.resampled(target_shape),
                             axis_roles=self.axis_roles)

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
                    # The split is along the column axis, so only the column coordinate is
                    # cut. An unresolved coordinate is copied whole, as the hardcoded
                    # `k in ["x","lon"]` test did with everything it did not know.
                    if self._coordinate_axis(k) == "column":
                        sliced[k] = v[start:end]
                    else:
                        sliced[k] = v.clone()
                else:
                    sliced[k] = v[:, start:end]
            return sliced

        return {
            "train": PhysicalField(train_data, coords=slice_coords(0, train_end),
                                   metadata=self.metadata, split="train",
                                   grid=self.grid.subset(col_start=0, col_stop=train_end),
                                   axis_roles=self.axis_roles),
            "val": PhysicalField(val_data, coords=slice_coords(train_end, val_end),
                                 metadata=self.metadata, split="val",
                                 grid=self.grid.subset(col_start=train_end, col_stop=val_end),
                                 axis_roles=self.axis_roles),
            "test": PhysicalField(test_data, coords=slice_coords(val_end, w),
                                  metadata=self.metadata, split="test",
                                  grid=self.grid.subset(col_start=val_end, col_stop=w),
                                  axis_roles=self.axis_roles)
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