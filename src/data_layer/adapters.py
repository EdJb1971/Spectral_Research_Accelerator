import os
import logging
import torch
import numpy as np
import pandas as pd
import xarray as xr
from src.data_layer import sources as data_sources
from src.data_layer import builtin_sources as _builtin  # noqa: F401

logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Optional, Sequence, Tuple
from src.physical_core.field import PhysicalField

def create_simulated_era5() -> xr.Dataset:
    times = pd.date_range("2023-01-01", periods=5, freq="D")
    levels = [1000.0, 850.0, 500.0, 300.0, 200.0]
    lats = np.linspace(-90.0, 90.0, 45)
    lons = np.linspace(-180.0, 180.0, 90)
    
    coords = {
        "time": times,
        "level": levels,
        "lat": lats,
        "lon": lons
    }
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    t2m_base = 25.0 * np.cos(np.deg2rad(lat_grid)) + 273.15
    
    t2m_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx in range(len(times)):
        wave = 5.0 * np.sin(4 * np.deg2rad(lon_grid)) * np.cos(np.deg2rad(lat_grid))
        t2m_data[t_idx] = t2m_base + wave + 2.0 * np.sin(t_idx * np.pi / 4)
        
    z_data = np.zeros((len(times), len(levels), len(lats), len(lons)), dtype=np.float32)
    for l_idx, lvl in enumerate(levels):
        base_height = 8000.0 * np.log(1013.25 / lvl)
        for t_idx in range(len(times)):
            wave = 200.0 * np.sin(3 * np.deg2rad(lon_grid)) * np.cos(2 * np.deg2rad(lat_grid))
            z_data[t_idx, l_idx] = base_height + wave + 50.0 * np.sin(t_idx * np.pi / 4)
            
    ds = xr.Dataset(
        data_vars={
            "t2m": (["time", "lat", "lon"], t2m_data, {"units": "K", "long_name": "2 metre temperature"}),
            "z": (["time", "level", "lat", "lon"], z_data, {"units": "m**2 s**-2", "long_name": "Geopotential"})
        },
        coords=coords,
        attrs={"description": "Simulated ERA5 Reanalysis Dataset for SpectralEarth"}
    )
    return ds

def create_simulated_gfs() -> xr.Dataset:
    times = pd.date_range("2023-01-01", periods=6, freq="6h")
    levels = [1000.0, 850.0, 500.0, 300.0]
    lats = np.linspace(-90.0, 90.0, 45)
    lons = np.linspace(-180.0, 180.0, 90)
    
    coords = {
        "time": times,
        "level": levels,
        "lat": lats,
        "lon": lons
    }
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    t2m_base = 24.0 * np.cos(np.deg2rad(lat_grid)) + 273.15
    t2m_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx in range(len(times)):
        wave = 4.0 * np.sin(3 * np.deg2rad(lon_grid)) * np.cos(np.deg2rad(lat_grid))
        t2m_data[t_idx] = t2m_base + wave + 1.5 * np.sin(t_idx * np.pi / 3)
        
    z_data = np.zeros((len(times), len(levels), len(lats), len(lons)), dtype=np.float32)
    for l_idx, lvl in enumerate(levels):
        base_height = 8000.0 * np.log(1013.25 / lvl)
        for t_idx in range(len(times)):
            wave = 150.0 * np.sin(2 * np.deg2rad(lon_grid)) * np.cos(2 * np.deg2rad(lat_grid))
            z_data[t_idx, l_idx] = base_height + wave + 30.0 * np.sin(t_idx * np.pi / 3)
            
    u_data = np.zeros((len(times), len(levels), len(lats), len(lons)), dtype=np.float32)
    v_data = np.zeros((len(times), len(levels), len(lats), len(lons)), dtype=np.float32)
    for l_idx, lvl in enumerate(levels):
        wind_speed = 40.0 * (1.0 - np.abs(lvl - 300.0) / 1000.0)
        for t_idx in range(len(times)):
            u_data[t_idx, l_idx] = wind_speed * np.exp(-(lat_grid - 45.0)**2 / 200.0) + 5.0 * np.random.randn(*lat_grid.shape)
            v_data[t_idx, l_idx] = 10.0 * np.sin(4 * np.deg2rad(lon_grid)) * np.exp(-(lat_grid - 45.0)**2 / 200.0)
            
    ds = xr.Dataset(
        data_vars={
            "t2m": (["time", "lat", "lon"], t2m_data, {"units": "K", "long_name": "2 metre temperature"}),
            "z": (["time", "level", "lat", "lon"], z_data, {"units": "m**2 s**-2", "long_name": "Geopotential"}),
            "u": (["time", "level", "lat", "lon"], u_data, {"units": "m s**-1", "long_name": "U component of wind"}),
            "v": (["time", "level", "lat", "lon"], v_data, {"units": "m s**-1", "long_name": "V component of wind"})
        },
        coords=coords,
        attrs={"description": "Simulated GFS Forecast Dataset for SpectralEarth"}
    )
    return ds

def create_simulated_toy() -> xr.Dataset:
    times = pd.date_range("2023-01-01", periods=10, freq="D")
    lats = np.linspace(-90.0, 90.0, 32)
    lons = np.linspace(-180.0, 180.0, 64)
    
    coords = {
        "time": times,
        "lat": lats,
        "lon": lons
    }
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    
    sst_base = 28.0 * np.cos(np.deg2rad(lat_grid)) + 2.0
    sst_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx in range(len(times)):
        wave = 2.0 * np.sin(5 * np.deg2rad(lon_grid)) * np.cos(np.deg2rad(lat_grid))
        sst_data[t_idx] = sst_base + wave + 0.5 * np.sin(t_idx * np.pi / 5)
        
    msl_base = 1013.25 + 10.0 * np.sin(np.deg2rad(lat_grid))
    msl_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx in range(len(times)):
        wave = 15.0 * np.cos(3 * np.deg2rad(lon_grid)) * np.sin(2 * np.deg2rad(lat_grid))
        msl_data[t_idx] = msl_base + wave + 3.0 * np.sin(t_idx * np.pi / 5)
        
    ds = xr.Dataset(
        data_vars={
            "sst": (["time", "lat", "lon"], sst_data, {"units": "C", "long_name": "Sea surface temperature"}),
            "msl": (["time", "lat", "lon"], msl_data, {"units": "hPa", "long_name": "Mean sea level pressure"})
        },
        coords=coords,
        attrs={"description": "Simulated Toy Climate Model for SpectralEarth"}
    )
    return ds

class MeteorologicalDataAdapter:
    """Resolve dataset ids through the source registry and slice them onto the core spine.

    GRIB is NOT supported: there is no cfgrib branch here and cfgrib is not a dependency.
    Local files use ``.nc``; the separately registered Zarr source accepts explicit,
    provenance-carrying regional ERA5 crop specifications.
    """

    _datasets: Dict[str, xr.Dataset] = {}
    # (source_kind, path_or_None, mtime_or_None) per dataset id, so a newly added file is
    # picked up without an API restart (D9) and so the resolved source is auditable.
    _sources: Dict[str, Dict[str, Any]] = {}

    DATA_DIR = "data"

    @classmethod
    def _resolve_path(cls, dataset_id: str) -> Optional[str]:
        path = os.path.join(cls.DATA_DIR, f"{dataset_id}.nc")
        return path if os.path.exists(path) else None

    @classmethod
    def get_dataset(cls, dataset_id: str,
                    source_options: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """Resolve a dataset through the registered source chain (T3.5.15, standard E2).

        Sources are tried in priority order and the full attempt record is kept, so a run
        served by the simulated fallback is a visible provenance fact rather than something
        inferable from a log line. The mtime-based cache invalidation of D9 is preserved:
        dropping a new `.nc` file into `data/` is still picked up without a restart.
        """
        dataset_id = dataset_id.lower()
        options = dict(source_options or {})

        # Parameterised sources such as ERA5 Zarr do not denote one stable dataset by id:
        # the crop specification *is part of the identity*. Their own content-addressed cache
        # handles reuse, so do not put one crop in the legacy id-only process cache where a
        # later request for another crop could silently receive the first one (D42).
        if options:
            resolution = data_sources.resolve(dataset_id, **options)
            cls._sources[dataset_id] = {
                "kind": resolution.kind,
                "path": None,
                "mtime": None,
                "fallback_reason": resolution.fallback_reason,
                "source_name": resolution.source_name,
                "is_simulated": resolution.is_simulated,
                "provenance": resolution.to_provenance(),
                "request": options,
            }
            return resolution.dataset

        path = cls._resolve_path(dataset_id)
        mtime = os.path.getmtime(path) if path else None

        cached = cls._sources.get(dataset_id)
        if cached is not None and dataset_id in cls._datasets:
            if cached.get("path") == path and cached.get("mtime") == mtime:
                return cls._datasets[dataset_id]
            try:
                cls._datasets[dataset_id].close()
            except Exception:
                pass

        resolution = data_sources.resolve(dataset_id)
        cls._datasets[dataset_id] = resolution.dataset
        cls._sources[dataset_id] = {
            "kind": resolution.kind,
            "path": path,
            "mtime": mtime,
            "fallback_reason": resolution.fallback_reason,
            "source_name": resolution.source_name,
            "is_simulated": resolution.is_simulated,
            "provenance": resolution.to_provenance(),
        }
        return cls._datasets[dataset_id]


    @classmethod
    def get_source_info(cls, dataset_id: str) -> Dict[str, Any]:
        """Where this dataset actually came from. Surfaced through the API so a run is
        never silently satisfied by simulated data when the researcher expected ERA5."""
        dataset_id = dataset_id.lower()
        # Always re-resolve: get_dataset() performs the path/mtime check that detects a
        # newly added or changed file (D9). Trusting a cached _sources entry here would
        # report a stale source forever.
        cls.get_dataset(dataset_id)
        return dict(cls._sources[dataset_id])

    @classmethod
    def invalidate_cache(cls, dataset_id: Optional[str] = None) -> None:
        """Drop cached handles so the next access re-resolves from disk."""
        ids = [dataset_id.lower()] if dataset_id else list(cls._datasets.keys())
        for d in ids:
            ds = cls._datasets.pop(d, None)
            cls._sources.pop(d, None)
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

    @classmethod
    def _get_simulated_fallback(cls, dataset_id: str) -> xr.Dataset:
        if dataset_id == "era5_reanalysis":
            return create_simulated_era5()
        elif dataset_id == "gfs_forecast":
            return create_simulated_gfs()
        elif dataset_id == "toy_climate_model":
            return create_simulated_toy()
        else:
            raise ValueError(f"Unknown dataset ID: {dataset_id}")

    @classmethod
    def known_dataset_ids(cls) -> List[str]:
        """Every dataset id any registered source declares it can serve."""
        ids = set()
        for entry in data_sources.SOURCES.entries():
            ids.update(entry.capabilities.get("dataset_ids", []))
        return sorted(ids)

    @classmethod
    def list_datasets(cls) -> List[Dict[str, Any]]:
        # Dataset ids come from the registered sources' declared capabilities, not from a
        # literal here. That is what makes the acceptance criterion of T3.5.15 achievable:
        # a source added in a new file shows up in this listing with no edit to adapters.py.
        for d_id in cls.known_dataset_ids():
            try:
                cls.get_dataset(d_id)
            except Exception as exc:
                logger.warning("dataset %s could not be resolved: %s", d_id, exc)
            
        result = []
        for d_id, ds in cls._datasets.items():
            vars_info = {}
            for v_name, v_da in ds.data_vars.items():
                vars_info[v_name] = {
                    "units": v_da.attrs.get("units", "unknown"),
                    "long_name": v_da.attrs.get("long_name", v_name),
                    "dims": list(v_da.dims)
                }
                
            source = cls._sources.get(d_id, {"kind": "unknown"})
            metadata = {
                "id": d_id,
                "name": d_id.replace("_", " ").title(),
                "source_kind": source.get("kind", "unknown"),
                "source_path": source.get("path"),
                "fallback_reason": source.get("fallback_reason"),
                # Read the flag the source *declared*, not a string comparison on `kind`.
                # The literal `kind == "simulated"` test silently reported any simulated
                # source with a different kind label (a demo source, a future model-output
                # source) as real observational data - found by the T3.5.15 acceptance test
                # the moment a third source existed.
                "is_simulated": bool(source.get("is_simulated",
                                                source.get("kind") == "simulated")),
                "description": ds.attrs.get("description", ""),
                "variables": list(ds.data_vars.keys()),
                "variables_metadata": vars_info,
                "pressure_levels": list(ds.coords["level"].values.tolist()) if "level" in ds.coords else None,
                "time_range": [str(ds.coords["time"].values[0]), str(ds.coords["time"].values[-1])],
                "spatial_resolution": "2.0 degree" if d_id != "toy_climate_model" else "5.0 degree",
                "bounding_box": {
                    "lat_min": float(ds.coords["lat"].values.min()),
                    "lat_max": float(ds.coords["lat"].values.max()),
                    "lon_min": float(ds.coords["lon"].values.min()),
                    "lon_max": float(ds.coords["lon"].values.max())
                }
            }
            result.append(metadata)
        return result

    @classmethod
    def slice_dataset(
        cls,
        dataset_id: str,
        variable: str,
        time: Optional[str] = None,
        level: Optional[float] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        source_options: Optional[Dict[str, Any]] = None,
    ) -> PhysicalField:
        ds = cls.get_dataset(dataset_id, source_options=source_options)
        
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset '{dataset_id}'. Available: {list(ds.data_vars.keys())}")
            
        da = ds[variable]
        
        if "time" in da.dims:
            if time is not None:
                da = da.sel(time=time, method="nearest")
            else:
                da = da.isel(time=0)
                
        if "level" in da.dims:
            if level is not None:
                da = da.sel(level=level, method="nearest")
            else:
                if 500.0 in da.coords["level"].values:
                    da = da.sel(level=500.0)
                else:
                    da = da.isel(level=len(da.coords["level"]) // 2)
                    
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"

        if lat_range is not None:
            lat_min, lat_max = sorted(lat_range)
            latitude = da.coords[lat_name].values
            lat_slice = (slice(lat_max, lat_min)
                         if len(latitude) > 1 and latitude[0] > latitude[-1]
                         else slice(lat_min, lat_max))
            da = da.sel({lat_name: lat_slice})
        if lon_range is not None:
            lon_min, lon_max = sorted(lon_range)
            da = da.sel({lon_name: slice(lon_min, lon_max)})
            
        da = da.squeeze()
        if len(da.dims) != 2:
            raise ValueError(f"Slicing did not result in a 2D field. Remaining dimensions: {da.dims}")
            
        data_tensor = torch.tensor(da.values, dtype=torch.float32)
        if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
        lat_coords = torch.tensor(da.coords[lat_name].values, dtype=torch.float32)
        lon_coords = torch.tensor(da.coords[lon_name].values, dtype=torch.float32)
        
        coords = {"lat": lat_coords, "lon": lon_coords}
        metadata = {
            "dataset_id": dataset_id,
            "variable": variable,
            "units": da.attrs.get("units", "unknown"),
            "long_name": da.attrs.get("long_name", variable),
            "dims": list(da.dims)
        }
        if "level" in ds.coords and level is not None:
            metadata["level"] = float(level)
        if "time" in ds.coords and time is not None:
            metadata["time"] = str(time)
            
        return PhysicalField(data_tensor, coords=coords, metadata=metadata)

    @classmethod
    def slice_sequence(
        cls,
        dataset_id: str,
        variable: str,
        time_range: Optional[Tuple[Any, Any]] = None,
        level: Optional[float] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        max_frames: int = 2000,
        source_options: Optional[Dict[str, Any]] = None,
    ) -> "FieldSequence":
        """Crop a dataset over **time** rather than at one instant (roadmap T4A.4).

        The counterpart of `slice_dataset`, and the entry point through which every Phase 4
        stage gets its data. `slice_dataset` picks one timestep and throws the time axis away;
        everything above 4A needs that axis kept.

        ``max_frames`` is a guard, not a preference. A whole ERA5 record at 6-hourly cadence is
        ~93,000 frames; materialising that as a list of `PhysicalField`s is tens of gigabytes
        of Python objects, and the failure mode is a killed process rather than an error
        message. The limit refuses with the count and the fix instead.
        """
        from src.physical_core.sequence import FieldSequence

        ds = cls.get_dataset(dataset_id, source_options=source_options)
        if variable not in ds.data_vars:
            raise ValueError(
                f"Variable '{variable}' not found in dataset '{dataset_id}'. "
                f"Available: {list(ds.data_vars.keys())}")

        da = ds[variable]
        if "time" not in da.dims:
            raise ValueError(
                f"'{variable}' in '{dataset_id}' has no time dimension (dims {list(da.dims)}), "
                f"so it cannot be sliced as a sequence. Use slice_dataset for a single field.")

        if time_range is not None:
            start, end = time_range
            da = da.sel(time=slice(start, end))
            if da.sizes.get("time", 0) == 0:
                coverage = ds["time"].values
                raise ValueError(
                    f"time_range {time_range} selects no frames. '{dataset_id}' covers "
                    f"{str(coverage[0])[:19]} to {str(coverage[-1])[:19]}.")

        if "level" in da.dims:
            if level is not None:
                da = da.sel(level=level, method="nearest")
            elif 500.0 in da.coords["level"].values:
                da = da.sel(level=500.0)
            else:
                da = da.isel(level=len(da.coords["level"]) // 2)

        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"

        if lat_range is not None:
            lat_min, lat_max = sorted(lat_range)
            latitude = da.coords[lat_name].values
            lat_slice = (slice(lat_max, lat_min)
                         if len(latitude) > 1 and latitude[0] > latitude[-1]
                         else slice(lat_min, lat_max))
            da = da.sel({lat_name: lat_slice})
        if lon_range is not None:
            lon_min, lon_max = sorted(lon_range)
            da = da.sel({lon_name: slice(lon_min, lon_max)})

        n_frames = int(da.sizes["time"])
        if n_frames > max_frames:
            raise ValueError(
                f"this crop selects {n_frames} frames, over the {max_frames}-frame limit. "
                f"Materialising them as PhysicalField objects would use roughly "
                f"{n_frames * da.sizes.get(lat_name, 1) * da.sizes.get(lon_name, 1) * 8 / 1e9:.1f} GB "
                f"before any analysis runs. Narrow time_range, or raise max_frames "
                f"deliberately if the memory is genuinely available.")

        values = da.values
        if values.ndim != 3:
            raise ValueError(
                f"slicing produced a {values.ndim}D array with dims {list(da.dims)}; a "
                f"sequence needs exactly (time, lat, lon) after level selection.")

        # nan_to_num is applied per frame *and counted*, because silently replacing missing
        # data with zeros changes every statistic computed afterwards - a zero is a value, not
        # an absence, and a spectrum of a field with zeroed gaps has structure the atmosphere
        # does not.
        n_nonfinite = int(np.count_nonzero(~np.isfinite(values)))

        lat_coords = torch.tensor(da.coords[lat_name].values, dtype=torch.float64)
        lon_coords = torch.tensor(da.coords[lon_name].values, dtype=torch.float64)
        coords = {"lat": lat_coords, "lon": lon_coords}

        base_metadata = {
            "dataset_id": dataset_id,
            "variable": variable,
            "units": da.attrs.get("units", "unknown"),
            "long_name": da.attrs.get("long_name", variable),
        }
        if level is not None:
            base_metadata["level"] = float(level)

        fields = []
        for index in range(n_frames):
            frame = torch.tensor(values[index], dtype=torch.float64)
            if not torch.isfinite(frame).all():
                frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            fields.append(PhysicalField(
                frame, coords=coords,
                metadata={**base_metadata, "frame_index": index}))

        source = cls._sources.get(dataset_id, {})
        metadata = {
            **base_metadata,
            "n_nonfinite_replaced": n_nonfinite,
            "nonfinite_policy": (
                "non-finite values replaced with 0.0 and counted. A zero is a value, not an "
                "absence: any spectrum or gradient computed here includes structure the "
                "replacement introduced." if n_nonfinite else "no non-finite values present"),
            "is_simulated": bool(source.get("is_simulated", True)),
            "source_kind": source.get("kind", "unknown"),
            "fallback_reason": source.get("fallback_reason"),
            "source_provenance": source.get("provenance"),
            "source_request": source.get("request"),
        }
        return FieldSequence(fields, da.coords["time"].values, metadata=metadata)

    @classmethod
    def slice_level_sequences(
        cls,
        dataset_id: str,
        variable: str,
        levels: Sequence[float],
        time_range: Optional[Tuple[Any, Any]] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        max_frames: int = 2000,
    ) -> Dict[float, "FieldSequence"]:
        """One `FieldSequence` per pressure level, sharing a grid and a time axis (T4B.4).

        The vertical axis has until now been a *selector* - "analyse 500 hPa" - which makes the
        canonical atmospheric precursor relationship, an upper-level trough preceding surface
        cyclogenesis, impossible to express: two levels were two unrelated runs with nothing
        tying their time axes together.

        The levels are sliced in one pass here rather than by repeated calls so the shared time
        axis is a property of the construction rather than something a caller has to remember
        to arrange. A duplicate level is refused rather than deduplicated: it almost always
        means a typo in a config, and quietly collapsing it would make the returned bank a
        different shape than the one that was asked for.
        """
        if not levels:
            raise ValueError(
                "at least one pressure level is required; pass a single value to "
                "slice_sequence instead if the vertical axis is not part of the analysis.")
        requested = [float(level) for level in levels]
        duplicates = sorted({lev for lev in requested if requested.count(lev) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate pressure levels {duplicates}. A level bank indexed by pressure "
                f"cannot hold the same level twice, and silently deduplicating would return "
                f"a bank of a different size than the config asked for.")

        sequences: Dict[float, "FieldSequence"] = {}
        for level in requested:
            sequences[level] = cls.slice_sequence(
                dataset_id=dataset_id, variable=variable, time_range=time_range,
                level=level, lat_range=lat_range, lon_range=lon_range,
                max_frames=max_frames)

        # `sel(method="nearest")` inside slice_sequence means a requested level that the
        # dataset does not carry silently becomes its neighbour. Two requested levels can
        # therefore land on the *same* stored level, which would make a cross-level lead-lag
        # a comparison of a field with itself - a correlation of 1.0 that means nothing.
        reference = sequences[requested[0]]
        for level, sequence in sequences.items():
            if level is requested[0]:
                continue
            if torch.equal(sequence.to_tensor(), reference.to_tensor()):
                raise ValueError(
                    f"levels {requested[0]} and {level} hPa returned identical data. The "
                    f"dataset does not carry both, and nearest-level selection has mapped "
                    f"them onto one stored level; a cross-level statistic computed from this "
                    f"would be correlating a field with itself. Check which levels "
                    f"'{dataset_id}' actually provides.")
        return sequences
