"""The data sources that ship with the platform (T3.5.15, standard E2).

Two sources, in priority order:

*   ``netcdf_local`` (priority 10) - reads a real `.nc` file from `data/`.
*   ``simulated`` (priority 900) - fabricates a plausible field so the platform stays usable
    offline. Deliberately last, and deliberately labelled `is_simulated=True`, so a run that
    falls back to it is visibly not observational.

The priority gap is wide on purpose: `T3.5.18`'s Zarr/ERA5 source will slot in at ~20 without
anyone renumbering anything, and a future observational source can come in below that.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.core.errors import DataSourceError
from src.data_layer.sources import register_source

#: Dataset ids the built-in sources know about.
BUILTIN_DATASET_IDS = ("era5_reanalysis", "gfs_forecast", "toy_climate_model")


@register_source(
    "netcdf_local",
    description="Reads a NetCDF file from the local data directory.",
    priority=10,
    is_simulated=False,
    kind="netcdf",
    capabilities={"dataset_ids": list(BUILTIN_DATASET_IDS), "streaming": False,
                  "requires_network": False, "observational": True},
)
class LocalNetCDFSource:
    """Local NetCDF files, one per dataset id."""

    DATA_DIR = "data"

    @classmethod
    def _path(cls, dataset_id: str) -> str:
        from src.data_layer.adapters import MeteorologicalDataAdapter
        directory = getattr(MeteorologicalDataAdapter, "DATA_DIR", cls.DATA_DIR)
        return os.path.join(directory, "%s.nc" % dataset_id)

    @classmethod
    def can_serve(cls, dataset_id: str) -> bool:
        # Cheap and side-effect free, as the protocol requires: a path test, no open().
        return dataset_id in BUILTIN_DATASET_IDS and os.path.exists(cls._path(dataset_id))

    @classmethod
    def why_not(cls, dataset_id: str) -> str:
        if dataset_id not in BUILTIN_DATASET_IDS:
            return "this source only serves %s" % ", ".join(BUILTIN_DATASET_IDS)
        return ("no file at %s - drop a NetCDF file there to use real data instead of the "
                "simulated fallback" % cls._path(dataset_id))

    @classmethod
    def fetch(cls, dataset_id: str, **kwargs: Any):
        import xarray as xr

        path = cls._path(dataset_id)
        if not os.path.exists(path):
            raise DataSourceError(
                "no NetCDF file at %r. Drop a file there, or let the request fall through "
                "to the simulated source." % path, dataset_id=dataset_id, path=path)
        try:
            return xr.open_dataset(path)
        except Exception as exc:
            raise DataSourceError(
                "%r exists but could not be opened as NetCDF (%s: %s). Check the file is "
                "not truncated and that its engine is installed."
                % (path, type(exc).__name__, exc), dataset_id=dataset_id, path=path)


@register_source(
    "simulated",
    description="Fabricates a plausible field so the platform works offline. NOT real data.",
    priority=900,
    is_simulated=True,
    kind="simulated",
    capabilities={"dataset_ids": list(BUILTIN_DATASET_IDS), "streaming": False,
                  "requires_network": False, "observational": False},
)
class SimulatedSource:
    """Synthetic stand-ins for the three built-in dataset ids."""

    @staticmethod
    def can_serve(dataset_id: str) -> bool:
        return dataset_id in BUILTIN_DATASET_IDS

    @staticmethod
    def fetch(dataset_id: str, **kwargs: Any):
        from src.data_layer.adapters import (
            create_simulated_era5,
            create_simulated_gfs,
            create_simulated_toy,
        )

        builders = {
            "era5_reanalysis": create_simulated_era5,
            "gfs_forecast": create_simulated_gfs,
            "toy_climate_model": create_simulated_toy,
        }
        if dataset_id not in builders:
            raise DataSourceError(
                "the simulated source has no builder for %r; it knows %s"
                % (dataset_id, ", ".join(sorted(builders))), dataset_id=dataset_id)
        return builders[dataset_id]()
