import torch
import numpy as np
import pytest
from src.physical_core.field import PhysicalField
from src.data_layer.adapters import MeteorologicalDataAdapter
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine
from src.analysis_engine.decomposition import ErrorDecompositionEngine

def test_meteorological_data_adapter_list():
    datasets = MeteorologicalDataAdapter.list_datasets()
    assert len(datasets) == 3
    ids = [d["id"] for d in datasets]
    assert "era5_reanalysis" in ids
    assert "gfs_forecast" in ids
    assert "toy_climate_model" in ids

def test_meteorological_data_adapter_slice():
    field = MeteorologicalDataAdapter.slice_dataset(
        dataset_id="era5_reanalysis",
        variable="t2m",
        lat_range=(-45.0, 45.0),
        lon_range=(-90.0, 90.0)
    )
    assert isinstance(field, PhysicalField)
    assert field.data.shape[0] < 45
    assert field.data.shape[1] < 90
    assert "lat" in field.coords
    assert "lon" in field.coords
    assert field.metadata["variable"] == "t2m"

def test_analysis_engine_diagnostics():
    forecast = torch.randn(32, 32)
    ground_truth = forecast + 0.1 * torch.randn(32, 32)
    
    f_field = PhysicalField(forecast)
    g_field = PhysicalField(ground_truth)
    
    diagnostics = SpectralSpatialAnalysisEngine.compute_diagnostics(f_field, g_field)
    
    assert "spatial_metrics" in diagnostics
    assert "gradient_errors" in diagnostics
    assert "spectral_diagnostics" in diagnostics
    assert "wavelet_energy" in diagnostics
    
    assert diagnostics["spatial_metrics"]["root_mean_squared_error"] > 0.0
    assert diagnostics["spatial_metrics"]["structural_similarity_index"] > 0.0
    assert diagnostics["gradient_errors"]["gradient_magnitude_mae"] > 0.0
    assert len(diagnostics["spectral_diagnostics"]["forecast_psd"]) > 0

def test_error_decomposition():
    forecast = torch.randn(32, 32)
    ground_truth = forecast + 0.1 * torch.randn(32, 32)
    
    scale_decomp = ErrorDecompositionEngine.decompose_by_scale(forecast, ground_truth)
    assert "low_scale_rmse" in scale_decomp
    assert "mid_scale_rmse" in scale_decomp
    assert "high_scale_rmse" in scale_decomp
    
    boundary_decomp = ErrorDecompositionEngine.decompose_by_boundary(forecast, ground_truth, boundary_width=4)
    assert len(boundary_decomp) == 5
    assert boundary_decomp[0]["distance"] == 0.0
    assert boundary_decomp[0]["rmse"] > 0.0

def test_api_endpoints_analysis_data(client):
    
    resp = client.get("/api/v1/data/datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    assert len(datasets) == 3
    
    resp = client.post("/api/v1/data/slice", json={
        "dataset_id": "toy_climate_model",
        "variable": "sst",
        "lat_range": [-30.0, 30.0],
        "lon_range": [-60.0, 60.0]
    })
    assert resp.status_code == 200
    slice_data = resp.json()
    assert "field_data" in slice_data
    assert "coords" in slice_data
    
    f_data = torch.randn(16, 16).tolist()
    g_data = torch.randn(16, 16).tolist()
    resp = client.post("/api/v1/analysis/diagnostics", json={
        "forecast_data": f_data,
        "ground_truth_data": g_data
    })
    assert resp.status_code == 200
    diag_data = resp.json()
    assert "spatial_metrics" in diag_data
    assert "gradient_errors" in diag_data
    
    resp = client.post("/api/v1/analysis/error-decomposition", json={
        "forecast_data": f_data,
        "ground_truth_data": g_data,
        "boundary_width": 4
    })
    assert resp.status_code == 200
    decomp_data = resp.json()
    assert "scale_decomposition" in decomp_data
    assert "boundary_decomposition" in decomp_data

def test_error_decomposition_validation(client):
    
    # Test too large single field
    large_data = torch.randn(1025, 10).tolist()
    resp = client.post("/api/v1/analysis/error-decomposition", json={
        "forecast_data": large_data,
        "ground_truth_data": large_data
    })
    assert resp.status_code == 422
    
    # Test too large series length
    series_data = [torch.randn(10, 10).tolist()] * 101
    resp = client.post("/api/v1/analysis/error-decomposition", json={
        "forecast_series": series_data,
        "ground_truth_series": series_data,
        "lead_times": list(range(101))
    })
    assert resp.status_code == 422
