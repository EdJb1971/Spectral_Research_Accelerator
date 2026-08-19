import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Any, List, Optional, Tuple
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
    """Resolves dataset ids to xarray Datasets, preferring real NetCDF over simulation.

    GRIB is NOT supported: there is no cfgrib branch here and cfgrib is not a dependency.
    Only ``.nc`` is probed. Remote sources (CDS API, S3) and Zarr are Task 3.5.20.
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
    def get_dataset(cls, dataset_id: str) -> xr.Dataset:
        dataset_id = dataset_id.lower()
        path = cls._resolve_path(dataset_id)
        mtime = os.path.getmtime(path) if path else None

        cached = cls._sources.get(dataset_id)
        # Invalidate when the file appears, disappears, or changes on disk (D9).
        if cached is not None and dataset_id in cls._datasets:
            if cached.get("path") == path and cached.get("mtime") == mtime:
                return cls._datasets[dataset_id]
            try:
                cls._datasets[dataset_id].close()
            except Exception:
                pass

        if path is not None:
            try:
                cls._datasets[dataset_id] = xr.open_dataset(path)
                cls._sources[dataset_id] = {
                    "kind": "netcdf", "path": path, "mtime": mtime, "fallback_reason": None,
                }
                return cls._datasets[dataset_id]
            except Exception as e:
                # E2: a fallback is a provenance fact, not a silent convenience.
                cls._datasets[dataset_id] = cls._get_simulated_fallback(dataset_id)
                cls._sources[dataset_id] = {
                    "kind": "simulated", "path": path, "mtime": mtime,
                    "fallback_reason": f"{type(e).__name__} while opening {path}",
                }
                return cls._datasets[dataset_id]

        cls._datasets[dataset_id] = cls._get_simulated_fallback(dataset_id)
        cls._sources[dataset_id] = {
            "kind": "simulated", "path": None, "mtime": None,
            "fallback_reason": f"no file at {os.path.join(cls.DATA_DIR, dataset_id + '.nc')}",
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
    def list_datasets(cls) -> List[Dict[str, Any]]:
        for d_id in ["era5_reanalysis", "gfs_forecast", "toy_climate_model"]:
            cls.get_dataset(d_id)
            
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
                "is_simulated": source.get("kind") == "simulated",
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
        lon_range: Optional[Tuple[float, float]] = None
    ) -> PhysicalField:
        ds = cls.get_dataset(dataset_id)
        
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
                    
        if lat_range is not None:
            lat_min, lat_max = sorted(lat_range)
            da = da.sel(lat=slice(lat_min, lat_max))
        if lon_range is not None:
            lon_min, lon_max = sorted(lon_range)
            da = da.sel(lon=slice(lon_min, lon_max))
            
        da = da.squeeze()
        if len(da.dims) != 2:
            raise ValueError(f"Slicing did not result in a 2D field. Remaining dimensions: {da.dims}")
            
        data_tensor = torch.tensor(da.values, dtype=torch.float32)
        if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
        lat_coords = torch.tensor(da.coords["lat"].values, dtype=torch.float32)
        lon_coords = torch.tensor(da.coords["lon"].values, dtype=torch.float32)
        
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
