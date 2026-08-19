"""A plugin adding one data source and one pipeline action - in this file and nowhere else.

This exists to make T3.5.15's acceptance criterion executable rather than aspirational:

    "a new data source and a new pipeline action are each added in a **new file only**, with
    zero edits to engine.py, adapters.py or main.py, and both appear automatically in
    GET /api/v1/actions and GET /api/v1/data/datasets."

Nothing in `src/` imports this module. `test_registries.py` imports it and then checks that
both additions are live *and* that the three core files are byte-identical afterwards.

It doubles as the worked example for anyone extending the platform: a source needs
`can_serve`/`fetch` (and ideally `why_not`), an action needs a `(args, device)` callable, and
both need one decorator carrying the metadata the discovery endpoints render.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import xarray as xr

from src.core.errors import MissingParameterError
from src.data_layer.sources import register_source
from src.experiment_engine.actions import register_action
from src.physical_core.field import PhysicalField

DEMO_DATASET_ID = "demo_checkerboard"


@register_source(
    "checkerboard_demo",
    description="A deterministic checkerboard dataset, for demonstrating the plugin seam.",
    priority=500,
    is_simulated=True,
    kind="demo",
    capabilities={"dataset_ids": [DEMO_DATASET_ID], "streaming": False,
                  "requires_network": False, "observational": False},
)
class CheckerboardSource:
    """Serves one synthetic dataset with no I/O of any kind."""

    @staticmethod
    def can_serve(dataset_id: str) -> bool:
        return dataset_id == DEMO_DATASET_ID

    @staticmethod
    def why_not(dataset_id: str) -> str:
        return "this demo source only serves %r" % DEMO_DATASET_ID

    @staticmethod
    def fetch(dataset_id: str, **kwargs: Any) -> xr.Dataset:
        lat = np.linspace(60.0, 50.0, 16)
        lon = np.linspace(0.0, 10.0, 16)
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        board = ((np.arange(16)[:, None] + np.arange(16)[None, :]) % 2).astype("float32")
        values = np.stack([board, 1.0 - board])
        return xr.Dataset(
            {"checker": (("time", "lat", "lon"), values,
                         {"units": "1", "long_name": "Checkerboard pattern"})},
            coords={"time": time, "lat": lat, "lon": lon},
            attrs={"description": "Deterministic checkerboard for the plugin-seam test."},
        )


@register_action(
    "count_extrema",
    description="Count local maxima and minima in a field.",
    params={"field": "field reference", "threshold": "float, default 0.0"},
    node_type="metrics",
)
def count_extrema(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Count strict local extrema on the interior of a field."""
    field = args.get("field")
    if field is None:
        raise MissingParameterError("field", action="count_extrema", required=["field"])
    data = field.data if isinstance(field, PhysicalField) else torch.as_tensor(field)
    threshold = float(args.get("threshold", 0.0))

    centre = data[1:-1, 1:-1]
    neighbours = torch.stack([
        data[:-2, 1:-1], data[2:, 1:-1], data[1:-1, :-2], data[1:-1, 2:],
        data[:-2, :-2], data[:-2, 2:], data[2:, :-2], data[2:, 2:],
    ])
    is_max = (centre > neighbours.max(dim=0).values) & (centre.abs() > threshold)
    is_min = (centre < neighbours.min(dim=0).values) & (centre.abs() > threshold)
    return {
        "maxima": int(is_max.sum()),
        "minima": int(is_min.sum()),
        "extrema_count": int(is_max.sum() + is_min.sum()),
        "threshold": threshold,
    }
