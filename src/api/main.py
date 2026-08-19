import torch
import numpy as np
import logging
import datetime
import uuid
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from src.physical_core.field import PhysicalField
from src.transform_engine.transforms import SpectralTransformEngine
from src.transform_engine import stationary as swt_engine
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.synthetic_generator.perturbation import PerturbationEngine
from src.boundary_lab.boundary import BoundaryConditionLab
from src.data_layer.adapters import MeteorologicalDataAdapter
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine
from src.analysis_engine.decomposition import ErrorDecompositionEngine

from src.database.session import engine, Base, get_db, SessionLocal
from src.database.models import Experiment, ExperimentRun, LineageNode, LineageEdge, Hypothesis
from src.experiment_engine.engine import DeclarativeExperimentEngine

logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables safely within the lifespan context to avoid race conditions
    Base.metadata.create_all(bind=engine)
    # Background tasks outlive the request-scoped session, so they need their own factory.
    # Held on app.state so tests (and later worker processes) can rebind it (D24).
    if not getattr(app.state, "session_factory", None):
        app.state.session_factory = SessionLocal
    yield

app = FastAPI(
    title="SpectralEarth Research Platform API",
    description="High-performance spectral transforms, physical field operations, synthetic generation, boundary-condition laboratory, meteorological data adapters, analysis engine, declarative experiment engine, and automated hypothesis engine.",
    version="1.0.0",
    lifespan=lifespan
)

# D5: without CORS the API is reachable only through the Vite dev proxy, so any
# separate-origin or statically-hosted deployment breaks. Origins come from the
# environment so production does not inherit a development allowlist.
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the API and its database are reachable.")
    api_version: str = Field(..., description="API version string.")
    database: str = Field(..., description="'ok' or an error summary.")
    database_url_scheme: str = Field(..., description="Backend in use, e.g. 'sqlite'.")
    datasets_available: int = Field(..., description="Number of datasets the adapter can resolve.")
    torch_device: str = Field(..., description="Execution device selected for this process.")


class TransformRequest(BaseModel):
    field_data: List[List[float]] = Field(..., description="2D array representing the physical field.")
    transform_type: str = Field(..., description="Type of transform: 'fft', 'dct', 'dwt', 'swt', 'dtcwt', or 'hybrid'.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters for the transform.")

    @validator("field_data")
    def validate_field_data(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("field_data must be a non-empty 2D list.")
        height = len(v)
        width = len(v[0])
        if height > 1024 or width > 1024:
            raise ValueError("Field dimensions must not exceed 1024x1024.")
        for row in v:
            if len(row) != width:
                raise ValueError("All rows in field_data must have the same length.")
        return v

class TransformResponse(BaseModel):
    reconstructed_field: List[List[float]] = Field(..., description="Reconstructed 2D field after inverse transform.")
    coefficients: Dict[str, Any] = Field(..., description="Transform coefficients (e.g., magnitude, phase, subbands).")
    metrics: Dict[str, float] = Field(..., description="Performance and reconstruction accuracy metrics (e.g., MSE, Max Error).")

class GenerateRequest(BaseModel):
    type: str = Field(..., description="Type of field: 'sinusoid', 'vortex', or 'front'.")
    height: int = Field(..., ge=4, le=1024, description="Height of the generated field.")
    width: int = Field(..., ge=4, le=1024, description="Width of the generated field.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the generator.")

class GenerateResponse(BaseModel):
    field_data: List[List[float]] = Field(..., description="Generated 2D field.")
    coords: Dict[str, List[float]] = Field(..., description="Coordinate mappings.")
    metadata: Dict[str, Any] = Field(..., description="Metadata of the generated field.")

class PerturbationItem(BaseModel):
    type: str = Field(..., description="Type of perturbation: 'rotation', 'translation', or 'noise'.")
    angle: Optional[float] = Field(None, description="Rotation angle in degrees.")
    shift_x: Optional[float] = Field(None, description="Translation along X-axis (normalized [-1, 1]).")
    shift_y: Optional[float] = Field(None, description="Translation along Y-axis (normalized [-1, 1]).")
    noise_type: Optional[str] = Field("gaussian", description="Type of noise: 'gaussian', 'uniform', or 'salt_pepper'.")
    level: Optional[float] = Field(0.1, description="Noise level or perturbation magnitude.")

class PerturbRequest(BaseModel):
    field_data: List[List[float]] = Field(..., description="2D array representing the physical field.")
    perturbations: List[PerturbationItem] = Field(..., description="List of perturbations to apply sequentially.")

    @validator("field_data")
    def validate_field_data(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("field_data must be a non-empty 2D list.")
        height = len(v)
        width = len(v[0])
        if height > 1024 or width > 1024:
            raise ValueError("Field dimensions must not exceed 1024x1024.")
        for row in v:
            if len(row) != width:
                raise ValueError("All rows in field_data must have the same length.")
        return v

class PerturbResponse(BaseModel):
    perturbed_field: List[List[float]] = Field(..., description="Perturbed 2D field.")
    metrics: Dict[str, float] = Field(..., description="Sensitivity and error metrics comparing perturbed to original.")

class BoundaryRequest(BaseModel):
    field_data: List[List[float]] = Field(..., description="2D array representing the physical field.")
    treatment: str = Field(..., description="Boundary treatment: 'periodic', 'zero', 'reflect', or 'replicate'.")
    pad_width: int = Field(..., ge=1, le=128, description="Padding width.")
    window_type: Optional[str] = Field(None, description="Optional window function: 'hann', 'hamming', or 'tukey'.")
    window_alpha: Optional[float] = Field(0.1, description="Tukey window alpha parameter.")
    reference_field_data: Optional[List[List[float]]] = Field(None, description="Optional reference field for error calculation.")

    @validator("field_data")
    def validate_field_data(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("field_data must be a non-empty 2D list.")
        height = len(v)
        width = len(v[0])
        if height > 1024 or width > 1024:
            raise ValueError("Field dimensions must not exceed 1024x1024.")
        for row in v:
            if len(row) != width:
                raise ValueError("All rows in field_data must have the same length.")
        return v

class BoundaryResponse(BaseModel):
    padded_field: List[List[float]] = Field(..., description="Padded 2D field.")
    distance_profiles: List[Dict[str, Any]] = Field(..., description="Spatial error diagnostics as a function of distance.")
    spectral_leakage: float = Field(..., description="Spectral leakage ratio.")
    has_reference: bool = Field(..., description="Whether a reference field was used.")
    # D13: the frame these numbers live in is part of the result, not an implementation
    # detail. Boundary gradients are deliberately in pixel units (see boundary.py); saying
    # so in the payload is what stops them being read as physical.
    gradient_units: str = Field("value per pixel", description="Units of the profile gradients.")
    distance_units: str = Field("pixel", description="Units of the distance axis.")
    gradient_frame_note: Optional[str] = Field(None, description="Why this frame was chosen.")
    grid: Optional[Dict[str, Any]] = Field(None, description="Grid geometry of the input field.")

class DatasetMetadata(BaseModel):
    id: str
    name: str
    description: str
    # E2: the resolved source is a provenance fact and must be visible to the researcher.
    source_kind: str = Field("unknown", description="'netcdf' or 'simulated'.")
    source_path: Optional[str] = Field(None, description="File backing this dataset, if any.")
    fallback_reason: Optional[str] = Field(None, description="Why simulation was used, if it was.")
    is_simulated: bool = Field(False, description="True when served by the simulated fallback.")
    variables: List[str]
    pressure_levels: Optional[List[float]] = None
    time_range: List[str]
    spatial_resolution: str
    bounding_box: Dict[str, float]

class SliceRequest(BaseModel):
    dataset_id: str = Field(..., description="ID of the dataset: 'era5_reanalysis', 'gfs_forecast', or 'toy_climate_model'.")
    variable: str = Field(..., description="Variable to extract.")
    time: Optional[str] = Field(None, description="ISO timestamp or date string.")
    level: Optional[float] = Field(None, description="Pressure level in hPa.")
    lat_range: Optional[Tuple[float, float]] = Field(None, description="Latitude range (min, max).")
    lon_range: Optional[Tuple[float, float]] = Field(None, description="Longitude range (min, max).")

class SliceResponse(BaseModel):
    field_data: List[List[float]] = Field(..., description="Extracted 2D physical field.")
    coords: Dict[str, List[float]] = Field(..., description="Coordinate mappings.")
    metadata: Dict[str, Any] = Field(..., description="Metadata of the sliced field.")

class DiagnosticsRequest(BaseModel):
    forecast_data: List[List[float]] = Field(..., description="2D array representing the forecast field.")
    ground_truth_data: List[List[float]] = Field(..., description="2D array representing the ground truth field.")

    @validator("forecast_data", "ground_truth_data")
    def validate_field_data(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("Field data must be a non-empty 2D list.")
        height = len(v)
        width = len(v[0])
        if height > 1024 or width > 1024:
            raise ValueError("Field dimensions must not exceed 1024x1024.")
        for row in v:
            if len(row) != width:
                raise ValueError("All rows must have the same length.")
        return v

class DiagnosticsResponse(BaseModel):
    # spatial_metrics and gradient_errors are Dict[str, Any] rather than Dict[str, float]
    # because, since D13, every physical quantity travels with its units, its grid
    # provenance and its weighting flags. Typing them as float silently dropped that
    # context - a number whose units are not carried alongside it is exactly the failure
    # mode D13 was about, so the schema now admits the metadata instead of discarding it.
    spatial_metrics: Dict[str, Any] = Field(..., description="Spatial error metrics, area-weighted, with units.")
    gradient_errors: Dict[str, Any] = Field(..., description="Metric-aware gradient error metrics, with units.")
    spectral_diagnostics: Dict[str, Any] = Field(..., description="Power spectrum (physical wavenumber, E(k) convention) and coherence.")
    wavelet_energy: Dict[str, Any] = Field(..., description="Wavelet energy distribution across scales.")
    grid: Optional[Dict[str, Any]] = Field(None, description="Grid geometry the diagnostics were computed on.")

class ErrorDecompositionRequest(BaseModel):
    forecast_data: Optional[List[List[float]]] = Field(None, description="2D array representing a single forecast field.")
    ground_truth_data: Optional[List[List[float]]] = Field(None, description="2D array representing a single ground truth field.")
    forecast_series: Optional[List[List[List[float]]]] = Field(None, description="Series of 2D forecast fields.")
    ground_truth_series: Optional[List[List[List[float]]]] = Field(None, description="Series of 2D ground truth fields.")
    lead_times: Optional[List[float]] = Field(None, description="Lead times corresponding to the series.")
    boundary_width: Optional[int] = Field(8, ge=1, le=128, description="Width of the boundary zone for distance analysis.")

    @validator("forecast_data", "ground_truth_data")
    def validate_field_data(cls, v):
        if v is not None:
            if not isinstance(v, list) or len(v) == 0 or not isinstance(v[0], list):
                raise ValueError("Field data must be a non-empty 2D list.")
            height = len(v)
            width = len(v[0])
            if height > 1024 or width > 1024:
                raise ValueError("Field dimensions must not exceed 1024x1024.")
            for row in v:
                if len(row) != width:
                    raise ValueError("All rows must have the same length.")
        return v

    @validator("forecast_series", "ground_truth_series")
    def validate_series(cls, v):
        if v is not None:
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError("Series must be a non-empty list of 2D arrays.")
            if len(v) > 100:
                raise ValueError("Series length must not exceed 100 steps.")
            for idx, field in enumerate(v):
                if not isinstance(field, list) or len(field) == 0 or not isinstance(field[0], list):
                    raise ValueError(f"Field at index {idx} must be a non-empty 2D list.")
                h = len(field)
                w = len(field[0])
                if h > 1024 or w > 1024:
                    raise ValueError(f"Field dimensions at index {idx} must not exceed 1024x1024.")
                for row in field:
                    if len(row) != w:
                        raise ValueError(f"All rows in field at index {idx} must have the same length.")
        return v

    @root_validator(skip_on_failure=True)
    def check_inputs(cls, values):
        fd = values.get("forecast_data")
        gtd = values.get("ground_truth_data")
        fs = values.get("forecast_series")
        gts = values.get("ground_truth_series")
        lts = values.get("lead_times")
        
        if fd is not None or gtd is not None:
            if fd is None or gtd is None:
                raise ValueError("Both forecast_data and ground_truth_data must be provided for single-field analysis.")
            if len(fd) != len(gtd) or len(fd[0]) != len(gtd[0]):
                raise ValueError("forecast_data and ground_truth_data must have the same dimensions.")
                
        if fs is not None or gts is not None:
            if fs is None or gts is None or lts is None:
                raise ValueError("forecast_series, ground_truth_series, and lead_times must all be provided for series analysis.")
            if len(fs) != len(gts) or len(fs) != len(lts):
                raise ValueError("forecast_series, ground_truth_series, and lead_times must have the same length.")
            h, w = len(fs[0]), len(fs[0][0])
            for idx, (f, g) in enumerate(zip(fs, gts)):
                if len(f) != h or len(f[0]) != w:
                    raise ValueError(f"All fields in forecast_series must have the same dimensions {h}x{w}.")
                if len(g) != h or len(g[0]) != w:
                    raise ValueError(f"All fields in ground_truth_series must have the same dimensions {h}x{w}.")
                
        if fd is None and fs is None:
            raise ValueError("Either single-field data or series data must be provided.")
            
        return values

class ErrorDecompositionResponse(BaseModel):
    scale_decomposition: Optional[Dict[str, float]] = Field(None, description="Error decomposed across spatial scales.")
    boundary_decomposition: Optional[List[Dict[str, Any]]] = Field(None, description="Error decomposed by distance from boundary.")
    lead_time_decomposition: Optional[List[Dict[str, Any]]] = Field(None, description="Error trajectory over lead times.")

# Experiment Engine Models
class PipelineStep(BaseModel):
    name: str = Field(..., description="Unique name of the pipeline step.")
    action: str = Field(..., description="Action to execute.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the action.")

class ExperimentRequest(BaseModel):
    name: str = Field(..., description="Name of the experiment.")
    description: Optional[str] = Field(None, description="Detailed description of the experiment.")
    parameter_matrix: Dict[str, List[Any]] = Field(default_factory=dict, description="Parameter matrix for Cartesian product expansion.")
    pipeline: List[PipelineStep] = Field(..., description="Sequence of pipeline steps to execute.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including code_revision and dataset_version.")

    @validator("parameter_matrix")
    def validate_parameter_matrix(cls, v):
        if not v:
            return v
        total_combinations = 1
        for val_list in v.values():
            if isinstance(val_list, list):
                if len(val_list) > 0:
                    total_combinations *= len(val_list)
            else:
                raise ValueError("All values in parameter_matrix must be lists.")
        if total_combinations > 1000:
            raise ValueError(f"Parameter matrix expansion exceeds maximum limit of 1000 combinations (got {total_combinations}).")
        return v

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    config: Dict[str, Any]
    created_at: str
    updated_at: str

class RunResponse(BaseModel):
    id: str
    experiment_id: str
    parameters: Dict[str, Any]
    status: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class ExperimentDetailResponse(ExperimentResponse):
    runs: List[RunResponse]

class ExperimentListResponse(BaseModel):
    total: int = Field(..., description="Total experiments matching the query.")
    limit: int
    offset: int
    experiments: List[ExperimentResponse] = Field(..., description="Page of experiments, newest first.")

class LineageNodeResponse(BaseModel):
    id: str
    name: str
    type: str
    value: Dict[str, Any]
    created_at: str

class LineageEdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation: str

class LineageResponse(BaseModel):
    nodes: List[LineageNodeResponse]
    edges: List[LineageEdgeResponse]

# Hypothesis Engine Models
class DiscoverRequest(BaseModel):
    experiment_ids: Optional[List[str]] = Field(None, description="List of experiment IDs to analyze. If empty, analyze all completed experiments.")
    target_metrics: Optional[List[str]] = Field(None, description="List of metrics to focus on. If empty, analyze all numerical metrics.")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum absolute correlation coefficient or relative difference.")

class HypothesisResponse(BaseModel):
    id: str
    experiment_ids: List[str]
    pattern_type: str
    description: str
    confidence: float
    metrics_analyzed: List[str]
    parameters_analyzed: List[str]
    proposed_experiment_config: Optional[Dict[str, Any]] = None
    created_at: str

@app.get("/api/v1/health", response_model=HealthResponse)
async def health(db: Session = Depends(get_db)):
    """Liveness/readiness probe.

    Used by start_platform.ps1 to wait for the backend instead of racing it (D7),
    and by the frontend to distinguish "backend offline" from "request failed".
    """
    from sqlalchemy import text

    from src.experiment_engine.engine import get_execution_device

    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:  # pragma: no cover - exercised only on a broken database
        db_status = f"error: {type(e).__name__}"

    try:
        n_datasets = len(MeteorologicalDataAdapter.list_datasets())
    except Exception:
        n_datasets = 0

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        api_version=app.version,
        database=db_status,
        database_url_scheme=str(engine.url).split(":", 1)[0],
        datasets_available=n_datasets,
        torch_device=str(get_execution_device(0)),
    )


@app.post("/api/v1/transforms/apply", response_model=TransformResponse)
async def apply_transform(request: TransformRequest):
    try:
        data_tensor = torch.tensor(request.field_data, dtype=torch.float32)
        field = PhysicalField(data_tensor)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input field data format.")

    transform_type = request.transform_type.lower()
    config = request.config
    
    try:
        if transform_type == "fft":
            magnitude, phase = SpectralTransformEngine.apply_fft2d(field)
            reconstructed = SpectralTransformEngine.inverse_fft2d(magnitude, phase, field.data.shape)
            coefficients = {
                "magnitude": magnitude.tolist(),
                "phase": phase.tolist()
            }
        elif transform_type == "dct":
            coeffs = SpectralTransformEngine.apply_dct2d(field)
            reconstructed = SpectralTransformEngine.inverse_dct2d(coeffs)
            coefficients = {
                "coefficients": coeffs.tolist()
            }
        elif transform_type == "dwt":
            levels = config.get("levels", 1)
            coeffs = SpectralTransformEngine.apply_dwt2d(field, levels=levels)
            reconstructed = SpectralTransformEngine.inverse_dwt2d(coeffs, levels=levels, target_shape=field.data.shape)
            serialized_coeffs = {"LL": coeffs["LL"].tolist()}
            for lvl in range(1, levels + 1):
                serialized_coeffs[f"level_{lvl}"] = {
                    k: v.tolist() for k, v in coeffs[f"level_{lvl}"].items()
                }
            coefficients = serialized_coeffs
        elif transform_type == "dtcwt":
            levels = config.get("levels", 1)
            coeffs = SpectralTransformEngine.apply_dtcwt2d(field, levels=levels)
            reconstructed = SpectralTransformEngine.inverse_dtcwt2d(coeffs, levels=levels, target_shape=field.data.shape)
            serialized_coeffs = {"LL": coeffs["LL"].tolist()}
            for lvl in range(1, levels + 1):
                serialized_coeffs[f"level_{lvl}"] = {
                    "LH_real": coeffs[f"level_{lvl}"]["LH_real"].tolist(),
                    "LH_imag": coeffs[f"level_{lvl}"]["LH_imag"].tolist(),
                    "HL_real": coeffs[f"level_{lvl}"]["HL_real"].tolist(),
                    "HL_imag": coeffs[f"level_{lvl}"]["HL_imag"].tolist(),
                    "HH_real": coeffs[f"level_{lvl}"]["HH_real"].tolist(),
                    "HH_imag": coeffs[f"level_{lvl}"]["HH_imag"].tolist()
                }
            coefficients = serialized_coeffs
        elif transform_type == "swt":
            # Undecimated / stationary transform (T3.5.7): every band keeps the parent grid
            # shape and the transform is shift-invariant, which is what Phase 4 requires.
            # NOTE: another branch on the chain defect D15 is about; moves to a registry in T3.5.15.
            levels = config.get("levels", 1)
            wavelet = config.get("wavelet", "haar")
            mode = config.get("mode", "periodic")
            coeffs_swt = swt_engine.apply_swt2d(field, levels=levels, wavelet=wavelet, mode=mode)
            reconstructed = (
                swt_engine.inverse_swt2d(coeffs_swt) if mode == "periodic" else field
            )
            serialized = {"LL": coeffs_swt["LL"].tolist()}
            for lvl in range(1, levels + 1):
                serialized[f"level_{lvl}"] = {
                    k: v.tolist() for k, v in coeffs_swt[f"level_{lvl}"].items()
                }
            serialized["energy_fractions"] = swt_engine.swt_energy_fractions(coeffs_swt)
            serialized["meta"] = coeffs_swt["meta"]
            coefficients = serialized
        elif transform_type == "hybrid":
            crossover_freq = config.get("crossover_freq", 0.5)
            mixing_weight = config.get("mixing_weight", 0.5)
            coeffs = SpectralTransformEngine.apply_hybrid(field, crossover_freq, mixing_weight)
            reconstructed = SpectralTransformEngine.inverse_hybrid(coeffs, field.data.shape)
            serialized_dwt = {"LL": coeffs["dwt_coeffs"]["LL"].tolist()}
            for lvl in range(1, 2):
                serialized_dwt[f"level_{lvl}"] = {
                    k: v.tolist() for k, v in coeffs["dwt_coeffs"][f"level_{lvl}"].items()
                }
            coefficients = {
                "fft_mag": coeffs["fft_mag"].tolist(),
                "fft_phase": coeffs["fft_phase"].tolist(),
                "fft_mag_filtered": coeffs["fft_mag_filtered"].tolist(),
                "dwt_coeffs": serialized_dwt,
                "mixing_weight": coeffs["mixing_weight"],
                "crossover_freq": coeffs["crossover_freq"]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported transform type: {transform_type}")
            
        mse = torch.mean((field.data - reconstructed.data) ** 2).item()
        max_err = torch.max(torch.abs(field.data - reconstructed.data)).item()
        
        return TransformResponse(
            reconstructed_field=reconstructed.data.tolist(),
            coefficients=coefficients,
            metrics={
                "mean_squared_error": mse,
                "max_absolute_error": max_err
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal transform error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the transform.")

@app.post("/api/v1/synthetic/generate", response_model=GenerateResponse)
async def generate_synthetic_field(request: GenerateRequest):
    try:
        field_type = request.type.lower()
        params = request.params
        
        if field_type == "sinusoid":
            freqs = params.get("frequencies", [(2.0, 2.0)])
            amps = params.get("amplitudes", [1.0])
            phases = params.get("phases", [(0.0, 0.0)])
            field = SyntheticFieldGenerator.generate_sinusoid(request.height, request.width, freqs, amps, phases)
        elif field_type == "vortex":
            centers = params.get("centers", [(0.5, 0.5)])
            amps = params.get("amplitudes", [1.0])
            radii = params.get("core_radii", [0.1])
            field = SyntheticFieldGenerator.generate_vortex(request.height, request.width, centers, amps, radii)
        elif field_type == "front":
            angle = params.get("angle", 0.0)
            offset = params.get("offset", 0.0)
            width_param = params.get("width_param", 0.1)
            amp = params.get("amplitude", 1.0)
            field = SyntheticFieldGenerator.generate_front(request.height, request.width, angle, offset, width_param, amp)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported synthetic field type: {field_type}")
            
        return GenerateResponse(
            field_data=field.data.tolist(),
            coords={k: v.tolist() for k, v in field.coords.items()},
            metadata=field.metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthetic generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during synthetic field generation.")

@app.post("/api/v1/synthetic/perturb", response_model=PerturbResponse)
async def perturb_field(request: PerturbRequest):
    try:
        original_tensor = torch.tensor(request.field_data, dtype=torch.float32)
        original_field = PhysicalField(original_tensor)
        
        current_field = original_field
        for pert in request.perturbations:
            p_type = pert.type.lower()
            if p_type == "rotation":
                if pert.angle is None:
                    raise HTTPException(status_code=400, detail="Rotation perturbation requires an 'angle' parameter.")
                current_field = PerturbationEngine.rotate(current_field, pert.angle)
            elif p_type == "translation":
                if pert.shift_x is None or pert.shift_y is None:
                    raise HTTPException(status_code=400, detail="Translation perturbation requires 'shift_x' and 'shift_y' parameters.")
                current_field = PerturbationEngine.translate(current_field, pert.shift_x, pert.shift_y)
            elif p_type == "noise":
                current_field = PerturbationEngine.add_noise(
                    current_field, 
                    pert.noise_type if pert.noise_type is not None else "gaussian", 
                    pert.level if pert.level is not None else 0.1
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported perturbation type: {p_type}")
                
        metrics = PerturbationEngine.compute_sensitivity_metrics(original_field, current_field)
        return PerturbResponse(
            perturbed_field=current_field.data.tolist(),
            metrics=metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Perturbation engine error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during field perturbation.")

@app.post("/api/v1/boundary/analyze", response_model=BoundaryResponse)
async def analyze_boundary(request: BoundaryRequest):
    try:
        data_tensor = torch.tensor(request.field_data, dtype=torch.float32)
        field = PhysicalField(data_tensor)
        
        ref_field = None
        if request.reference_field_data is not None:
            ref_tensor = torch.tensor(request.reference_field_data, dtype=torch.float32)
            ref_field = PhysicalField(ref_tensor)
            
        analysis = BoundaryConditionLab.analyze_boundary_artefacts(
            field=field,
            treatment=request.treatment,
            pad_width=request.pad_width,
            window_type=request.window_type,
            window_alpha=request.window_alpha if request.window_alpha is not None else 0.1,
            reference_field=ref_field
        )
        
        return BoundaryResponse(
            padded_field=analysis["padded_field"],
            distance_profiles=analysis["distance_profiles"],
            spectral_leakage=analysis["spectral_leakage"],
            has_reference=analysis["has_reference"],
            gradient_units=analysis["gradient_units"],
            distance_units=analysis["distance_units"],
            gradient_frame_note=analysis["gradient_frame_note"],
            grid=analysis["grid"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Boundary laboratory error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during boundary condition analysis.")

@app.get("/api/v1/data/datasets", response_model=List[DatasetMetadata])
async def list_datasets():
    try:
        return MeteorologicalDataAdapter.list_datasets()
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while listing datasets.")

@app.post("/api/v1/data/slice", response_model=SliceResponse)
async def slice_dataset(request: SliceRequest):
    try:
        field = MeteorologicalDataAdapter.slice_dataset(
            dataset_id=request.dataset_id,
            variable=request.variable,
            time=request.time,
            level=request.level,
            lat_range=request.lat_range,
            lon_range=request.lon_range
        )
        return SliceResponse(
            field_data=field.data.tolist(),
            coords={k: v.tolist() for k, v in field.coords.items()},
            metadata=field.metadata
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error slicing dataset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while slicing the dataset.")

@app.post("/api/v1/analysis/diagnostics", response_model=DiagnosticsResponse)
async def compute_diagnostics(request: DiagnosticsRequest):
    try:
        forecast_tensor = torch.tensor(request.forecast_data, dtype=torch.float32)
        gt_tensor = torch.tensor(request.ground_truth_data, dtype=torch.float32)
        
        forecast_field = PhysicalField(forecast_tensor)
        gt_field = PhysicalField(gt_tensor)
        
        diagnostics = SpectralSpatialAnalysisEngine.compute_diagnostics(forecast_field, gt_field)
        return DiagnosticsResponse(
            spatial_metrics=diagnostics["spatial_metrics"],
            gradient_errors=diagnostics["gradient_errors"],
            grid=diagnostics.get("grid"),
            spectral_diagnostics=diagnostics["spectral_diagnostics"],
            wavelet_energy=diagnostics["wavelet_energy"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error computing diagnostics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while computing diagnostics.")

@app.post("/api/v1/analysis/error-decomposition", response_model=ErrorDecompositionResponse)
async def decompose_errors(request: ErrorDecompositionRequest):
    try:
        scale_decomp = None
        boundary_decomp = None
        lead_time_decomp = None
        
        if request.forecast_data is not None and request.ground_truth_data is not None:
            f_tensor = torch.tensor(request.forecast_data, dtype=torch.float32)
            g_tensor = torch.tensor(request.ground_truth_data, dtype=torch.float32)
            
            scale_decomp = ErrorDecompositionEngine.decompose_by_scale(f_tensor, g_tensor)
            boundary_decomp = ErrorDecompositionEngine.decompose_by_boundary(
                f_tensor, g_tensor, boundary_width=request.boundary_width or 8
            )
            
        if request.forecast_series is not None and request.ground_truth_series is not None and request.lead_times is not None:
            f_series = [torch.tensor(f, dtype=torch.float32) for f in request.forecast_series]
            g_series = [torch.tensor(g, dtype=torch.float32) for g in request.ground_truth_series]
            
            lead_time_decomp = ErrorDecompositionEngine.decompose_by_lead_time(
                f_series, g_series, request.lead_times
            )
            
        return ErrorDecompositionResponse(
            scale_decomposition=scale_decomp,
            boundary_decomposition=boundary_decomp,
            lead_time_decomposition=lead_time_decomp
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error decomposing errors: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while decomposing errors.")

# Experiment Engine Endpoints
@app.post("/api/v1/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        experiment_id = str(uuid.uuid4())
        experiment = Experiment(
            id=experiment_id,
            name=request.name,
            description=request.description,
            status="PENDING",
            config=request.dict(),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            updated_at=datetime.datetime.now(datetime.timezone.utc)
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        background_tasks.add_task(
            DeclarativeExperimentEngine.execute_experiment,
            experiment_id=experiment_id,
            session_factory=getattr(app.state, "session_factory", None) or SessionLocal,
        )

        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            status=experiment.status,
            config=experiment.config,
            created_at=experiment.created_at.isoformat(),
            updated_at=experiment.updated_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while creating the experiment.")

@app.get("/api/v1/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    limit: int = 25,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Paginated experiment list, newest first.

    Without this the frontend cannot show prior experiments, which made the whole
    lineage/hypothesis loop unusable across sessions.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    query = db.query(Experiment)
    if status:
        query = query.filter(Experiment.status == status.upper())

    total = query.count()
    rows = query.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()

    return ExperimentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        experiments=[
            ExperimentResponse(
                id=e.id,
                name=e.name,
                description=e.description,
                status=e.status,
                config=e.config,
                created_at=e.created_at.isoformat(),
                updated_at=e.updated_at.isoformat(),
            )
            for e in rows
        ],
    )


@app.get("/api/v1/experiments/{id}", response_model=ExperimentDetailResponse)
async def get_experiment(id: str, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment with ID {id} not found.")
    
    runs = db.query(ExperimentRun).filter(ExperimentRun.experiment_id == id).all()
    
    return ExperimentDetailResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        status=experiment.status,
        config=experiment.config,
        created_at=experiment.created_at.isoformat(),
        updated_at=experiment.updated_at.isoformat(),
        runs=[
            RunResponse(
                id=r.id,
                experiment_id=r.experiment_id,
                parameters=r.parameters,
                status=r.status,
                results=r.results,
                error_message=r.error_message,
                created_at=r.created_at.isoformat(),
                completed_at=r.completed_at.isoformat() if r.completed_at else None
            ) for r in runs
        ]
    )

@app.get("/api/v1/experiments/{id}/lineage", response_model=LineageResponse)
async def get_experiment_lineage(id: str, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment with ID {id} not found.")
        
    nodes = db.query(LineageNode).filter(LineageNode.experiment_id == id).all()
    edges = db.query(LineageEdge).join(LineageNode, LineageEdge.source_id == LineageNode.id).filter(LineageNode.experiment_id == id).all()
    
    return LineageResponse(
        nodes=[
            LineageNodeResponse(
                id=n.id,
                name=n.name,
                type=n.type,
                value=n.value,
                created_at=n.created_at.isoformat()
            ) for n in nodes
        ],
        edges=[
            LineageEdgeResponse(
                id=e.id,
                source_id=e.source_id,
                target_id=e.target_id,
                relation=e.relation
            ) for e in edges
        ]
    )

# Hypothesis Engine Endpoints
@app.post("/api/v1/hypothesis/discover", response_model=List[HypothesisResponse])
async def discover_hypotheses(request: DiscoverRequest, db: Session = Depends(get_db)):
    try:
        from src.hypothesis_engine.engine import PatternDiscoveryEngine
        hypotheses = PatternDiscoveryEngine.discover_patterns(
            db=db,
            experiment_ids=request.experiment_ids,
            target_metrics=request.target_metrics,
            confidence_threshold=request.confidence_threshold if request.confidence_threshold is not None else 0.3
        )
        return [
            HypothesisResponse(
                id=h.id,
                experiment_ids=h.experiment_ids,
                pattern_type=h.pattern_type,
                description=h.description,
                confidence=h.confidence,
                metrics_analyzed=h.metrics_analyzed,
                parameters_analyzed=h.parameters_analyzed,
                proposed_experiment_config=h.proposed_experiment_config,
                created_at=h.created_at.isoformat()
            ) for h in hypotheses
        ]
    except Exception as e:
        logger.error(f"Error in hypothesis discovery: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during pattern discovery.")

@app.get("/api/v1/hypothesis/proposals", response_model=List[HypothesisResponse])
async def get_proposals(
    pattern_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Hypothesis)
        if pattern_type:
            query = query.filter(Hypothesis.pattern_type == pattern_type)
        if min_confidence is not None:
            query = query.filter(Hypothesis.confidence >= min_confidence)
            
        hypotheses = query.all()
        return [
            HypothesisResponse(
                id=h.id,
                experiment_ids=h.experiment_ids,
                pattern_type=h.pattern_type,
                description=h.description,
                confidence=h.confidence,
                metrics_analyzed=h.metrics_analyzed,
                parameters_analyzed=h.parameters_analyzed,
                proposed_experiment_config=h.proposed_experiment_config,
                created_at=h.created_at.isoformat()
            ) for h in hypotheses
        ]
    except Exception as e:
        logger.error(f"Error retrieving proposals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving proposals.")
