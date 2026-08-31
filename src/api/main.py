import torch
import numpy as np
import json
import logging
import datetime
import uuid
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (BaseModel, Field, StrictFloat, StrictInt, root_validator,
                      validator)
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlalchemy.orm import Session

from src.physical_core.field import PhysicalField
from src.transform_engine.transforms import SpectralTransformEngine
from src.transform_engine import registry as transform_registry
from src.experiment_engine import actions as pipeline_actions
from src.data_layer import sources as data_sources
from src.data_layer import builtin_sources as _builtin_sources  # noqa: F401
# Defect D35: registration is an import side effect, so a source module that is only
# imported lazily inside a handler is **absent from the fallback chain** until some
# unrelated request happens to import it. Measured on the running platform:
# GET /data/sources returned ['netcdf_local', 'simulated'] on a fresh process and
# ['netcdf_local', 'era5_zarr', 'simulated'] after visiting the ERA5 tab - so which
# sources `resolve()` considered depended on the order the researcher clicked.
from src.data_layer import zarr_source as _zarr_source  # noqa: F401
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.dataset_capabilities import build_profile
from src.core.onboarding import DOMAIN_ONBOARDINGS, geometry_offers_metric
from src.transform_engine import dtcwt as RealDTCWT
from src.transform_engine import stationary as swt_engine
from src.transform_engine.training import training_representation_catalogue, RepresentationError
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.synthetic_generator.perturbation import PerturbationEngine
from src.boundary_lab.boundary import BoundaryConditionLab
from src.data_layer.adapters import MeteorologicalDataAdapter
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine
from src.analysis_engine.decomposition import ErrorDecompositionEngine
from src.forecasting.adapter import ForecastContractError
from src.forecasting.evaluation_report import EvaluationReceiptStore, MAX_RECEIPT_BYTES

# Defect D36: every incoming field was cast to float32 at this boundary, so the platform's
# double-precision core was discarded the moment a request arrived. Measured on the running
# platform: an FFT round trip is 2.8e-16 in float64 and 1.9e-07 in float32 - nine orders of
# magnitude, on the identical field. D27 was fixed precisely because a float32 `fftfreq` capped
# Parseval at 5.8e-8; casting the data itself to float32 gave all of that back. The whole API
# now works in float64, which is what `PhysicalField`, the grid metrics, the spectra and the
# benchmarks have always assumed.
from src.database.session import engine, get_db, SessionLocal
from src.database import migrate as schema_migrate
from src.database.models import Experiment, ExperimentRun, LineageNode, LineageEdge, Hypothesis
from src.experiment_engine.engine import DeclarativeExperimentEngine

logger = logging.getLogger("api")


def _evaluation_receipts() -> EvaluationReceiptStore:
    """Resolve at request time so laptop/HPC deployments and tests can relocate evidence."""
    directory = getattr(app.state, "evaluation_receipt_dir", None)
    return EvaluationReceiptStore(directory)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema management is Alembic's, not create_all's (T3.5.8). create_all adds missing
    # *tables* and silently ignores missing *columns*, so every database created before a
    # slice that added a column stayed broken while startup reported success - measured:
    # `no such column: experiment_runs.seed` on the real spectral_earth.db.
    # `ensure_schema` adopts such a database by inspection and upgrades it.
    #
    # Bound to whichever engine this app is actually using: the test suite rebinds
    # session_factory to an in-memory database, and migrating the module-level engine
    # while serving a different one is exactly the mistake env.py's single-source-of-truth
    # URL exists to prevent.
    bound_engine = engine
    if getattr(app.state, "session_factory", None) is not None:
        bound_engine = app.state.session_factory.kw.get("bind", engine)
    app.state.schema = schema_migrate.ensure_schema(bound_engine)
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

# TG9.1: the cross-domain claim surface. Imported eagerly at module scope, not lazily inside a
# handler, so glossary registration cannot depend on which route a researcher happens to visit
# first - that was defect D35, and `DOMAIN_GLOSSARIES` has exactly the shape that caused it.
from src.api.findings import router as findings_router  # noqa: E402
from src.api.channels import router as channels_router  # noqa: E402
from src.api.acquisitions import router as acquisitions_router  # noqa: E402
from src.api.profiles import router as profiles_router  # noqa: E402
from src.api.lightcurves import router as lightcurves_router  # noqa: E402
from src.api.ingress import router as ingress_router  # noqa: E402
from src.api.analysis import router as domain_analysis_router  # noqa: E402
from src.api.preregistration import router as preregistration_router  # noqa: E402
from src.api.evidence import router as evidence_router  # noqa: E402
from src.api.mining import router as mining_router  # noqa: E402
from src.api.cross_domain import router as cross_domain_router  # noqa: E402
from src.api.reviews import router as reviews_router  # noqa: E402
from src.api.experiment_composer import router as experiment_composer_router  # noqa: E402
from src.api.experiment_runs import router as experiment_runs_router  # noqa: E402
from src.api.comparison_views import router as comparison_views_router  # noqa: E402
from src.api.experiment_receipts import router as experiment_receipts_router  # noqa: E402
from src.api.experiment_qualification import router as experiment_qualification_router  # noqa: E402

app.include_router(findings_router)
# TG8.4. Mounted here for the same reason the findings router is: registration must not depend
# on which handler happened to run first (D35). A domain that appears only after a researcher
# visits the right tab is a domain a record silently cannot be read under.
app.include_router(channels_router)
# TG10.2: domain-first projection over the existing domain/source contracts. Mounted eagerly so
# plugin registrations are visible without a researcher first visiting another route (D35).
app.include_router(acquisitions_router)
app.include_router(profiles_router)
app.include_router(lightcurves_router)
app.include_router(ingress_router)
# TG11.1: stateless access to the existing domain-analysis engine. It re-reads the selected
# full record, stores nothing and cannot move a claim rung (R22).
app.include_router(domain_analysis_router)
# TG11.2: the generate/confirm split (R18). Mounted after the analysis router because the
# ordering it enforces is on that router's gate operation: a sweep may not be launched against
# a held-out partition that has already been spent.
app.include_router(preregistration_router)
# TG11.3: the evidence write path. Mounted last of the workflow routers because it records
# what the earlier ones produce, and it is the only one that writes toward a claim: the rung
# in every one of its responses is recomputed by the ladder and never accepted from a client.
app.include_router(evidence_router)
# TG11.4: structure mining. Mounted after the evidence router because its outputs are what that
# router records, and because it depends on both of the routers above it - seals go into
# TG11.2's store and the held-out ledger it reads is TG11.2's ledger. It accepts no feature and
# no rung: scenes are extracted here from admitted fields, and a confirmation receipt is an
# input to the write path rather than a claim (R22).
app.include_router(mining_router)
# TG11.4b: the cross-domain record. Mounted beside the mining router because it shares the same
# two dependencies - TG11.2's seal store and its one held-out ledger - and last of the analysis
# surfaces because it is the only one whose lag family is declared in seconds rather than in
# frames of a single record's clock. It aligns two native clocks by exact intersection and never
# by interpolation, and it records no evidence and moves no rung (R22).
app.include_router(cross_domain_router)
# TG11.5: the review store is read only and stays outside the findings routes on purpose.
# Recorded argument can be inspected beside a selected study, but never shares an endpoint or
# a response object with translated claim text (R22/R23).
app.include_router(reviews_router)
# TG17.1: the first no-glue experiment surface. It stores only content-addressed manifest
# revisions and performs metadata-only planning; no route here acquires values or creates a claim.
app.include_router(experiment_composer_router)
app.include_router(experiment_runs_router)
# TG17.8: the linked comparison views. Read-only over a manifest and, when one already exists at
# that manifest's address, its run receipt. No route here opens a run, acquires a value or admits
# evidence; a view that would put two domains' native magnitudes on one axis refuses instead.
app.include_router(comparison_views_router)
app.include_router(experiment_receipts_router)
app.include_router(experiment_qualification_router)


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the API and its database are reachable.")
    api_version: str = Field(..., description="API version string.")
    database: str = Field(..., description="'ok' or an error summary.")
    database_url_scheme: str = Field(..., description="Backend in use, e.g. 'sqlite'.")
    datasets_available: int = Field(..., description="Number of datasets the adapter can resolve.")
    torch_device: str = Field(..., description="Execution device selected for this process.")
    # T3.5.19 / D18: what this deployment can actually do. A researcher choosing an
    # execution backend needs to know how many cores are available and whether the database
    # is configured for concurrent writes; both were previously invisible.
    execution: Dict[str, Any] = Field(default_factory=dict,
                                      description="Devices, thread budget and executor backends available.")
    database_settings: Dict[str, Any] = Field(default_factory=dict,
                                              description="SQLite pragmas actually in force (WAL, busy_timeout).")
    # T3.5.8: the stamped schema revision. A stored result is only reproducible if the
    # schema that stored it is identifiable, and "the schema is behind the code" is a
    # readiness fact a probe must be able to report rather than a surprise at query time.
    # Named `schema_state` rather than `schema`: pydantic v1 refuses a field that shadows
    # `BaseModel.schema()`, and an alias would leave the JSON key colliding with OpenAPI's
    # own `schema` in anything that walks the response generically.
    schema_state: Dict[str, Any] = Field(default_factory=dict,
                                         description="Alembic revision, head, and any pending migrations.")


class TransformRequest(BaseModel):
    field_data: List[List[float]] = Field(..., description="2D array representing the physical field.")
    transform_type: str = Field(..., description="Type of transform: 'fft', 'dct', 'dwt', 'swt', 'dtcwt', or 'hybrid'.")
    level1: Optional[str] = Field(None, description="DTCWT level-1 filter set (near_sym_a/near_sym_b/legall).")
    qshift: Optional[str] = Field(None, description="DTCWT q-shift filter set for levels >= 2 (qshift_a..d).")
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
    # Defect D34. `PerturbationEngine.add_noise` has taken a seed since T3.5.12, and this
    # endpoint never passed one - so every perturbation requested over HTTP was drawn from
    # the global RNG and could not be reproduced. The engine even recorded `seeded: False`
    # in its own metadata, and nothing surfaced it. Reproducibility that exists only in the
    # Python API is reproducibility the platform does not have.
    seed: Optional[int] = Field(None, description="Seed for a reproducible noise draw. Omit for an unseeded (irreproducible) draw.")

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
    # Declared before the existing fields so it reads first in the OpenAPI schema: a
    # perturbed field without its provenance is not a scientific object.
    perturbed_field: List[List[float]] = Field(..., description="Perturbed 2D field.")
    metrics: Dict[str, float] = Field(..., description="Sensitivity and error metrics comparing perturbed to original.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list,
                                             description="Per-perturbation record: type, seed, and whether it was seeded at all.")
    reproducible: bool = Field(True, description="False when any stochastic step ran unseeded, so this field cannot be regenerated.")

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


DEFAULT_BENCHMARK_SEED = 20260819


def _jsonable(value):
    """Make a known-answer dict JSON-serialisable without losing information.

    Known answers contain tuples, numpy scalars and torch values. Converting them here
    keeps the benchmark modules free of serialisation concerns, and refuses silently
    dropping anything - an unrepresentable value becomes its repr rather than vanishing.
    """
    import numpy as _np
    import torch as _torch

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (_np.integer,)):
        return int(value)
    if isinstance(value, (_np.floating,)):
        return float(value)
    if isinstance(value, _np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, _torch.Tensor):
        return _jsonable(value.tolist())
    return repr(value)


class BenchmarkCheckResponse(BaseModel):
    stage: str = Field(..., description="Pipeline stage this check gates.")
    outcome: str = Field(..., description="PASS, FAIL or NOT_YET_RUNNABLE.")
    detail: str = Field(..., description="What was measured, in words.")
    measured: Optional[Dict[str, Any]] = Field(None, description="Numbers behind the verdict.")

class BenchmarkResponse(BaseModel):
    name: str
    kind: str
    description: str
    gates: List[str]
    is_null: bool = Field(..., description="True when the correct answer is 'nothing here'.")
    known_answer: Dict[str, Any] = Field(..., description="The analytic truth, declared up front.")
    checks: List[BenchmarkCheckResponse] = Field(default_factory=list)

class BenchmarkSuiteResponse(BaseModel):
    root_seed: int
    passed: int
    failed: int
    not_yet_runnable: int
    null_failures: List[str] = Field(
        default_factory=list,
        description="Failures on null benchmarks. Non-empty means the platform is "
                    "reporting structure in data that contains none; treat every finding "
                    "it has produced as suspect.")
    benchmarks: List[BenchmarkResponse]

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
    # Defect D8. `confidence` is an effect size; on its own it is what produced 9 spurious
    # "discoveries" from a 9-run sweep. A reported pattern must travel with its p-value, its
    # multiplicity-corrected q-value, the size of the family the correction covered, and the
    # dependence assumption of the procedure used - otherwise the client cannot tell a
    # finding from an artefact, and neither can the researcher.
    p_value: Optional[float] = Field(None, description="Uncorrected p-value for this test.")
    q_value: Optional[float] = Field(None, description="Multiplicity-corrected p-value (FDR).")
    n_tests: Optional[int] = Field(None, description="Family size the correction covered.")
    statistics: Optional[Dict[str, Any]] = Field(
        None, description="Test used, correction procedure, its assumption, and caveats.")

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

    from src.core import device as device_policy
    from src.core.executor import BACKENDS
    from src.database.session import sqlite_settings

    try:
        db_settings = sqlite_settings()
    except Exception:  # pragma: no cover - only on a database that cannot be queried
        db_settings = {}

    # Reported per request rather than cached from startup: a migration applied by another
    # process while this one is running should show up here, and a cached "up_to_date": true
    # would be a stale reassurance - the one kind of health output that is worse than none.
    try:
        schema_state = schema_migrate.describe(db.get_bind())
    except Exception as e:  # pragma: no cover - only on a database that cannot be queried
        schema_state = {"error": "%s: %s" % (type(e).__name__, e)}

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        api_version=app.version,
        database=db_status,
        database_url_scheme=str(engine.url).split(":", 1)[0],
        datasets_available=n_datasets,
        torch_device=str(device_policy.select_device()),
        execution={
            "backends": list(BACKENDS),
            "default_backend": "serial",
            "default_backend_rationale": (
                "PyTorch already parallelises FFT and BLAS across cores, so run-level "
                "workers compete for cores it is using. Measured on this class of workload: "
                "serial beat thread(4) and process(4). Raise n_workers only when tasks are "
                "small and numerous, or set threads_per_worker to divide the cores."),
            "devices": device_policy.available_devices(),
            "cpu_count": __import__("os").cpu_count(),
            "torch_num_threads": __import__("torch").get_num_threads(),
        },
        database_settings=db_settings,
        schema_state=schema_state,
    )


@app.get("/api/v1/evaluation/receipts")
async def list_evaluation_receipts():
    """List only verified, accepted real-source reports; an empty list is the honest empty state."""
    return _evaluation_receipts().list()


@app.get("/api/v1/evaluation/receipts/{report_id}")
async def get_evaluation_receipt(report_id: str):
    try:
        return _evaluation_receipts().get(report_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="verified evaluation report not found") from None
    except ForecastContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/v1/evaluation/receipts/import")
async def import_evaluation_receipt(file: UploadFile = File(...)):
    """Verify and content-address one receipt upload; server filesystem paths are never accepted."""
    filename = file.filename or "receipt.json"
    if not filename.lower().endswith(".json"):
        raise HTTPException(status_code=415, detail="evaluation receipt must be a .json file")
    payload = await file.read(MAX_RECEIPT_BYTES + 1)
    try:
        return _evaluation_receipts().import_bytes(payload)
    except ForecastContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.post("/api/v1/transforms/apply", response_model=TransformResponse)
async def apply_transform(request: TransformRequest):
    try:
        data_tensor = torch.tensor(request.field_data, dtype=torch.float64)
        field = PhysicalField(data_tensor)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input field data format.")

    transform_type = request.transform_type.lower()
    config = request.config
    
    try:
        # T3.5.15 / D15: dispatch through the transform registry. This was a six-branch
        # `if/elif` that duplicated the one in `engine.py`, so the two could - and did -
        # drift apart on defaults. There is now a single definition of what each
        # transform name means, and an unknown name produces a 404 listing the valid
        # ones rather than a bare 400.
        result = transform_registry.apply_transform(transform_type, field, config)
        reconstructed = result["reconstructed"]
        coefficients = result["summary"]
        mse = result["reconstruction_mse"]
        max_err = float(torch.max(torch.abs(field.data.to(reconstructed.data.dtype)
                                            - reconstructed.data)))
        
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
        # T3.5.14 / D14: the error kind decides the status code, not this handler guessing.
        # A bad transform name or parameter is the caller's, and they get told exactly what
        # was wrong; a genuine internal fault stays opaque and is logged in full.
        info = classify(e)
        if info["status_code"] >= 500:
            logger.error(f"Internal transform error: {str(e)}", exc_info=True)
        else:
            logger.info(f"Rejected transform request: {str(e)}")
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])

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
        original_tensor = torch.tensor(request.field_data, dtype=torch.float64)
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
                    pert.level if pert.level is not None else 0.1,
                    seed=pert.seed,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported perturbation type: {p_type}")
                
        metrics = PerturbationEngine.compute_sensitivity_metrics(original_field, current_field)
        # The provenance travels with the field. `seeded: false` is reported rather than
        # hidden, because an unseeded draw is a real property of the result: nobody, including
        # the person who ran it, can reproduce it.
        provenance = [
            {"type": p.type.lower(),
             "seed": p.seed,
             "seeded": p.seed is not None,
             **({"noise_type": p.noise_type, "level": p.level}
                if p.type.lower() == "noise" else {})}
            for p in request.perturbations
        ]
        return PerturbResponse(
            perturbed_field=current_field.data.tolist(),
            metrics=metrics,
            provenance=provenance,
            reproducible=all(p["seeded"] for p in provenance
                             if p["type"] == "noise") or not any(
                                 p["type"] == "noise" for p in provenance),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Perturbation engine error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during field perturbation.")

@app.post("/api/v1/boundary/analyze", response_model=BoundaryResponse)
async def analyze_boundary(request: BoundaryRequest):
    try:
        data_tensor = torch.tensor(request.field_data, dtype=torch.float64)
        field = PhysicalField(data_tensor)
        
        ref_field = None
        if request.reference_field_data is not None:
            ref_tensor = torch.tensor(request.reference_field_data, dtype=torch.float64)
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

@app.get("/api/v1/actions")
async def list_actions():
    """Every registered pipeline action, generated from the registry (T3.5.15).

    Generated, never hand-maintained: a hand-written list is a document that goes stale, and
    this one would go stale precisely when someone adds an action - the moment it matters.
    """
    return [
        {**entry.to_dict(), "node_type": entry.value.node_type}
        for entry in pipeline_actions.ACTIONS.entries()
    ]

@app.get("/api/v1/transforms")
async def list_transforms():
    """Every registered transform, with its parameters and declared capabilities.

    `capabilities` is what makes the T4B.2 wavelet bank selectable by property rather than
    by name - "every shift-invariant multiscale transform" instead of a hard-coded list.
    """
    return transform_registry.TRANSFORMS.describe()


@app.get("/api/v1/training/representations")
async def list_training_representations(
    levels: int = Query(3, ge=1, le=8),
    wavelet: str = Query("db2"),
    height: int = Query(120, ge=1, le=4096),
    width: int = Query(80, ge=1, le=4096),
):
    """Training readiness is distinct from availability as a single-field transform.

    Returns accepted batched/autograd capabilities plus the selected SWT redundancy and R13
    edge budget. Entries that exist only for 2D analysis are named as such rather than being
    omitted or accidentally advertised as model-ready.
    """
    try:
        return training_representation_catalogue(levels, wavelet, (height, width))
    except RepresentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/api/v1/data/sources")
async def list_data_sources():
    """The data-source fallback chain, in priority order (standard E2)."""
    return data_sources.describe_sources()

class ZarrAnalysisRequest(BaseModel):
    """Transform identity used only to plan scientific crop support, not to select data."""

    transform_family: str = Field("dtcwt", description="Registered transform with R13 support.")
    wavelet: str = Field("db2", description="SWT wavelet: haar, db2 or db3.")
    boundary_mode: str = Field("periodic", description="SWT boundary mode.")
    dtcwt_level1: str = Field("near_sym_b", description="DTCWT level-1 filter family.")
    dtcwt_qshift: str = Field("qshift_b", description="DTCWT q-shift filter family.")


class ZarrCropRequest(BaseModel):
    """A regional crop of a cloud Zarr archive (T3.5.18)."""

    store: str = Field("era5_0p25_6h",
                       description="Catalogue id, or a raw Zarr URI (gs://... or a local path).")
    variables: List[str] = Field(..., description="Variable names, e.g. ['temperature'].")
    time_start: str = Field(..., description="ISO date or datetime, inclusive.")
    time_end: str = Field(..., description="ISO date or datetime, inclusive.")
    lat_min: float = Field(..., description="Southern edge, degrees north.")
    lat_max: float = Field(..., description="Northern edge, degrees north.")
    lon_min: float = Field(..., description="Western edge, degrees east.")
    lon_max: float = Field(..., description="Eastern edge, degrees east.")
    levels: List[Union[StrictInt, StrictFloat]] = Field(
        default_factory=list,
        description=("Exact values on the store's declared vertical axis: integer pressure "
                     "levels for ERA5 or fractional negative-metre elevations for GLORYS."))
    n_levels_analysis: int = Field(4, ge=1, le=8,
                                  description="Wavelet levels the crop must support (R13).")
    analysis: ZarrAnalysisRequest = Field(
        default_factory=ZarrAnalysisRequest,
        description=("Transform/filter identity used for metadata-only R13 planning. It does "
                     "not change which source values are selected."))


class ZarrProbeRequest(BaseModel):
    """Probe one store and record the result, whatever the result is (TG10.3)."""

    uri: str = Field(..., description="Catalogue id, raw Zarr URI, or a local store path.")
    variables: List[str] = Field(
        default_factory=list,
        description="Variables to describe. Empty means every variable the store carries.")
    crop: Optional[ZarrCropRequest] = Field(
        None,
        description=("Optional crop to cost. An amplification is a fact about one access "
                     "pattern, so it is only computed when the pattern is stated."))
    persist: bool = Field(
        True, description="Write the record to the probe directory as well as the ledger.")


def _zarr_support_request(request: ZarrCropRequest):
    """Bind the wire model to the implementation-owned transform support contract."""
    from src.data_layer.crop_planner import TransformSupportRequest

    return TransformSupportRequest(
        transform_family=request.analysis.transform_family,
        levels=request.n_levels_analysis,
        wavelet=request.analysis.wavelet,
        boundary_mode=request.analysis.boundary_mode,
        dtcwt_level1=request.analysis.dtcwt_level1,
        dtcwt_qshift=request.analysis.dtcwt_qshift,
    )


class ExportFieldRequest(BaseModel):
    """Export a 2D field with its coordinates and provenance (T3.5.23)."""

    field_data: List[List[float]] = Field(..., description="2D array to export.")
    format: str = Field("csv", description="csv, json, netcdf or zarr.")
    coords: Dict[str, List[float]] = Field(default_factory=dict,
                                           description="Coordinate vectors, e.g. {'lat': [...], 'lon': [...]}.")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="Provenance to embed IN the file: source, seed, is_simulated, grid, units.")
    variable: str = Field("field", description="Variable name used inside NetCDF/Zarr.")
    units: Optional[str] = Field(None, description="Physical units of the values.")
    name: str = Field("field", description="Filename stem; a UTC timestamp is appended.")


class ExportTableRequest(BaseModel):
    """Export a list of records - hypotheses, benchmarks, metrics, a spectrum."""

    rows: List[Dict[str, Any]] = Field(..., description="Records to export.")
    format: str = Field("csv", description="csv or json.")
    columns: Optional[List[str]] = Field(None, description="Column order; inferred from the rows if omitted.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provenance to embed in the file.")
    name: str = Field("table", description="Filename stem; a UTC timestamp is appended.")


@app.post("/api/v1/import/inspect")
async def import_inspect(file: UploadFile = File(...)):
    """Describe an uploaded file **without** committing to a 2D slice of it (T3.5.24).

    Two calls on purpose. An ERA5 NetCDF is `(time, level, lat, lon)`; there is no single field
    in it, and picking `[0, 0]` on the researcher's behalf would import a slice they did not
    choose while every statistic downstream described that arbitrary timestep. This reports the
    variables and which dimensions still need an index; `/import/field` then reads the one
    they name.
    """
    from src.data_layer import importers

    payload = await file.read()
    try:
        return importers.inspect(payload, file.filename or "upload")
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    except Exception as e:
        logger.error("Import inspection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400,
                            detail="Could not read this file: %s" % type(e).__name__)


@app.post("/api/v1/import/field")
async def import_field(
    file: UploadFile = File(...),
    variable: Optional[str] = Form(None),
    selection: Optional[str] = Form(None),
):
    """Read one 2D field out of an uploaded file, with reconstructed provenance.

    `selection` is a JSON object pinning every non-spatial dimension by index, e.g.
    `{"time": 0, "level": 2}`. The returned provenance records the filename, a content hash of
    the exact bytes, the variable and the selection - so a finding can name the file it came
    from rather than "a NetCDF someone uploaded".

    `is_simulated` comes back **null** for a file of unknown origin rather than false: the
    platform did not produce this data and asserting it is observational would be inventing a
    fact. Where the file carries our own provenance block, the flag is inherited from it, so a
    round trip cannot launder a simulated field into an apparently real one.
    """
    from src.data_layer import importers

    payload = await file.read()
    try:
        parsed = json.loads(selection) if selection else None
        if parsed is not None and not isinstance(parsed, dict):
            raise InvalidParameterError("selection", parsed,
                                        'a JSON object such as {"time": 0, "level": 2}')
        return importers.read_field(payload, file.filename or "upload",
                                    variable=variable, selection=parsed)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail='`selection` must be a JSON object such as {"time": 0, "level": 2}.')
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    except Exception as e:
        logger.error("Import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400,
                            detail="Could not read this file: %s" % type(e).__name__)


@app.post("/api/v1/export/field")
async def export_field_endpoint(request: ExportFieldRequest):
    """Serialise a field to CSV, JSON, NetCDF4 or a zipped Zarr store.

    **Why the server does this rather than the browser.** CSV and JSON a browser could build;
    NetCDF4 and Zarr it cannot - both need real binary writers, and a hand-rolled approximation
    would produce files that open in some tools and not others, which is worse than none. Doing
    all four here also means one code path decides what provenance is embedded, so a CSV and a
    NetCDF of the same field carry the same record.

    PNG and SVG are deliberately **not** here: those are rendered client-side from the live
    plot, because a server-side re-render would be a different picture from the one on screen.
    """
    from fastapi.responses import Response

    from src.data_layer import exporters

    try:
        payload = exporters.export_field(
            request.field_data, request.format, coords=request.coords or None,
            metadata=request.metadata, variable=request.variable, units=request.units)
        name = exporters.filename(request.name, request.format.strip().lower())
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Export failed: %s" % type(e).__name__)

    return Response(
        content=payload,
        media_type=exporters.MEDIA_TYPES[request.format.strip().lower()],
        headers={"Content-Disposition": 'attachment; filename="%s"' % name,
                 # Without this the browser cannot read the header it needs to name the file.
                 "Access-Control-Expose-Headers": "Content-Disposition"},
    )


@app.post("/api/v1/export/table")
async def export_table_endpoint(request: ExportTableRequest):
    """Serialise a table of records to CSV or JSON, provenance embedded."""
    from fastapi.responses import Response

    from src.data_layer import exporters

    try:
        payload = exporters.export_table(
            request.rows, request.format, columns=request.columns,
            metadata=request.metadata)
        name = exporters.filename(request.name, request.format.strip().lower())
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    except Exception as e:
        logger.error("Table export failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Export failed: %s" % type(e).__name__)

    return Response(
        content=payload,
        media_type=exporters.MEDIA_TYPES[request.format.strip().lower()],
        headers={"Content-Disposition": 'attachment; filename="%s"' % name,
                 "Access-Control-Expose-Headers": "Content-Disposition"},
    )


@app.get("/api/v1/data/zarr/catalogue")
async def zarr_catalogue():
    """Known cloud ERA5 stores, and whether this deployment may reach them (T3.5.18).

    The catalogue carries a `note` per store saying what it is good and bad *for*, because
    the difference between the 0.25 degree and 1.5 degree stores is not resolution alone: one
    is chunked one timestep at a time and is hostile to regional crops, the other is not.

    Since TG10.1 it is generated from the `GRIDDED_STORES` registry rather than from a literal,
    so a store added in a new file appears here with no edit to this function. Each entry also
    now carries the **domain** it belongs to, its access requirement, its vertical axis name
    and how its chunk figures were obtained - a store nobody has measured says so rather than
    reading like one that was.
    """
    from src.data_layer import stores as stores_module
    from src.data_layer import zarr_source as zarr_adapter
    from src.transform_engine.registry import TRANSFORMS

    support_transforms = {}
    for entry in TRANSFORMS.entries():
        if entry.value.support is not None:
            support_transforms[entry.name] = {
                "description": entry.description,
                "params": dict(entry.params),
                "capabilities": dict(entry.capabilities),
            }

    return {
        "stores": zarr_adapter.catalogue_payload(),
        "store_domains": stores_module.domains_with_stores(),
        "access_requirements": stores_module.ACCESS_REQUIREMENTS,
        "network_enabled": zarr_adapter.network_enabled(),
        "network_env_var": zarr_adapter.NETWORK_ENV_VAR,
        "missing_dependencies": zarr_adapter.missing_dependencies(),
        "cache_dir": zarr_adapter.DEFAULT_CACHE_DIR,
        "r13_minimum_crop": {str(n): zarr_adapter.minimum_crop_size(n)
                             for n in range(1, 7)},
        "analysis_transforms": support_transforms,
        "r13_legacy_note": ("r13_minimum_crop is the pre-planner conservative 14-tap table "
                            "kept for API compatibility. Use the request-specific acquisition "
                            "plan returned by /inspect for a scientific decision."),
        "note": ("Network access is opt-in: reaching the internet must never be a side "
                 "effect of running a sweep, and a mistyped bounding box against a 0.25 "
                 "degree store can move tens of gigabytes."),
    }


@app.get("/api/v1/data/zarr/cached")
async def zarr_cached_crops():
    """Crops already materialised locally. Available with no network at all."""
    from src.data_layer import zarr_source as zarr_adapter
    from src.data_layer.regional_forecast import assess_manifest_readiness

    crops = zarr_adapter.cached_crops()
    return {
        "count": len(crops),
        "cache_dir": zarr_adapter.DEFAULT_CACHE_DIR,
        "crops": [
            {
                "content_key": c.get("content_key"),
                "content_hash": c.get("content_hash"),
                "spec": c.get("spec"),
                "shape": c.get("shape"),
                "megabytes_transferred": c.get("megabytes_transferred"),
                "elapsed_s": c.get("elapsed_s"),
                "regional_forecast_readiness": assess_manifest_readiness(c),
            }
            for c in crops
        ],
    }


@app.get("/api/v1/data/zarr/probes")
async def zarr_probes():
    """Every recorded probe, newest first, with the transcription debt stated (TG10.3).

    `transcribed` counts records this code did not produce - inspections run before the probe
    existed, transcribed rather than deleted or re-invented. It is published rather than kept
    internal so the number can only fall where anyone can see it.
    """
    from src.data_layer import store_probe

    ledger = store_probe.probe_ledger()
    return {
        "count": len(ledger),
        "transcribed": len(store_probe.transcribed_probes()),
        "probe_dir": store_probe.DEFAULT_PROBE_DIR,
        "outcomes": store_probe.PROBE_OUTCOMES,
        "evidence_kinds": store_probe.EVIDENCE_KINDS,
        "probes": ledger,
        "note": ("A probe records what a store is, and records `unreachable`, `needs "
                 "credentials` or `network is switched off` just as readily - those are "
                 "results about a store, not failures of the probe."),
    }


@app.post("/api/v1/data/zarr/probe")
async def zarr_probe(request: ZarrProbeRequest):
    """Open a store, record its structure and cost, and record the refusal if it will not open.

    **Metadata only.** The store is opened lazily and an amplification is chunk arithmetic, so
    nothing of the data crosses the wire. Unlike `/inspect`, this does **not** return 409 when
    network access is off: "network is switched off here" is a recorded outcome, because a
    deployment that cannot reach a store needs that written down rather than raised.

    The one thing that is still an error is a request that makes no sense - an empty URI, or a
    crop that is not a crop - because that is a fault in the request rather than a fact about
    a store.
    """
    from src.data_layer import store_probe
    from src.data_layer import zarr_source as zarr_adapter

    crop = None
    if request.crop is not None:
        try:
            crop = zarr_adapter.CropSpec(
                store=request.crop.store, variables=tuple(request.crop.variables),
                time_start=request.crop.time_start, time_end=request.crop.time_end,
                lat_min=request.crop.lat_min, lat_max=request.crop.lat_max,
                lon_min=request.crop.lon_min, lon_max=request.crop.lon_max,
                levels=tuple(request.crop.levels),
                n_levels_analysis=request.crop.n_levels_analysis,
                vertical_dim=zarr_adapter.vertical_dim_for_store(request.crop.store),
            )
        except SpectralEarthError as e:
            info = classify(e)
            raise HTTPException(status_code=info["status_code"], detail=info["detail"])

    uri = zarr_adapter.uri_for(request.uri) or request.uri
    try:
        probe = store_probe.probe_store(
            uri, variables=list(request.variables) or None, crop=crop)
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])

    store_probe.record_probe(probe)
    saved = store_probe.save_probe(probe) if request.persist else None
    return {"probe": probe.to_dict(), "saved_to": saved,
            "requested": request.uri, "resolved_uri": uri}


@app.post("/api/v1/data/zarr/inspect")
async def zarr_inspect(request: ZarrCropRequest):
    """Report a store's chunk structure and whether this crop is chunk-hostile.

    **Metadata only - nothing of the data is transferred.** This is deliberately a separate
    endpoint from materialisation, because it is the call a researcher should make *first*:
    against WeatherBench 2's 0.25 degree archive a one-year 256x256 four-level crop must move
    79 GB to deliver 1.55 GB, and finding that out from a progress bar an hour in is not a
    design. Materialisation is not exposed over HTTP at all yet: it is a minutes-to-hours job
    that needs the Phase 4A artifact store and a job record, and a request that silently holds
    a connection open for two hours would be a worse answer than no endpoint.
    """
    from src.data_layer import zarr_source as zarr_adapter
    from src.data_layer import crop_planner

    try:
        spec = zarr_adapter.crop_for_store(
            request.store, variables=tuple(request.variables),
            time_start=request.time_start, time_end=request.time_end,
            lat_min=request.lat_min, lat_max=request.lat_max,
            lon_min=request.lon_min, lon_max=request.lon_max,
            levels=tuple(request.levels), n_levels_analysis=request.n_levels_analysis,
        )
        analysis = _zarr_support_request(request)
    except SpectralEarthError as e:
        # classify() also returns `error` and `context`, which HTTPException does not take;
        # only the status and the client-safe detail cross the wire.
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])

    remote = "://" in spec.uri
    if remote and not zarr_adapter.network_enabled():
        raise HTTPException(
            status_code=409,
            detail=("Inspecting %s requires network access, which is disabled. Set %s=1 to "
                    "enable it. Reading store metadata costs a few hundred kilobytes, but "
                    "enabling it is still a deliberate choice rather than a default."
                    % (spec.uri, zarr_adapter.NETWORK_ENV_VAR)))

    try:
        dataset, _counter = zarr_adapter.open_dataset(spec.uri, chunks={})
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    try:
        structure = zarr_adapter.describe_store(dataset, spec.variables)
        assessment = zarr_adapter.assess_access_pattern(dataset, spec)
        acquisition_plan = crop_planner.plan_acquisition(
            dataset, spec, analysis, structure=structure)
        geometry = acquisition_plan["geometry"]
    except SpectralEarthError as e:
        info = classify(e)
        raise HTTPException(status_code=info["status_code"], detail=info["detail"])
    finally:
        dataset.close()

    analysis_flags = "--analysis-levels %d --analysis-transform %s" % (
        request.n_levels_analysis, request.analysis.transform_family)
    if request.analysis.transform_family == "swt":
        analysis_flags += " --wavelet %s --boundary-mode %s" % (
            request.analysis.wavelet, request.analysis.boundary_mode)
    else:
        analysis_flags += " --dtcwt-level1 %s --dtcwt-qshift %s" % (
            request.analysis.dtcwt_level1, request.analysis.dtcwt_qshift)

    store_declaration = (zarr_adapter.store_for(request.store)
                         if request.store in zarr_adapter.GRIDDED_STORES else None)
    domain_name = store_declaration.domain if store_declaration is not None else None
    onboarding = DOMAIN_ONBOARDINGS.get(domain_name) if domain_name is not None else None
    capability_profile = build_profile(
        kind="gridded_crop", phase="planned",
        identity=acquisition_plan["plan_sha256"], domain=domain_name,
        facts={
            "sample_table": False, "channel_series": False, "profile_collection": False,
            "spatial_grid_2d": True,
            "physical_metric": (geometry_offers_metric(onboarding.geometry)
                                if onboarding is not None else None),
            "ordered_time_axis": True,
            "regular_cadence": (bool(store_declaration.cadence_hours)
                                if store_declaration is not None else None),
            "irregular_support": False,
            "transform_compatible": bool(geometry["meets_recommended_minimum"]),
            "precedence_admissible": (store_declaration.declaration().precedence_admissible
                                      if store_declaration is not None else None),
            "independent_samples": False,
        },
        basis={"plan_sha256": acquisition_plan["plan_sha256"],
               "source_observation_sha256": acquisition_plan["source_observation_sha256"],
               "geometry": onboarding.geometry if onboarding is not None else None,
               "crop_geometry": geometry,
               "note": "Planned from metadata and coordinates; no field value was read."})
    return {
        "spec": spec.to_provenance(),
        "cached": zarr_adapter.is_cached(spec),
        "structure": structure,
        "assessment": assessment,
        "geometry": geometry,
        "acquisition_plan": acquisition_plan,
        "capability_profile": capability_profile,
        "cli": ("python -m src.data_layer.zarr_source materialise --store %s --variables %s "
                "--start %s --end %s --lat %g %g --lon %g %g --levels %s %s"
                % (request.store, ",".join(request.variables), request.time_start,
                   request.time_end, request.lat_min, request.lat_max,
                   request.lon_min, request.lon_max,
                   ",".join(str(v) for v in request.levels), analysis_flags)),
    }


@app.get("/api/v1/benchmarks", response_model=List[BenchmarkResponse])
async def list_benchmarks():
    """The Ground-Truth Benchmark Suite: what is tested, and what the right answer is.

    Exposed through the API deliberately (standard E7). A researcher must be able to see
    what the platform has been proved to get right - and which gates are declared but not
    yet enforceable - without reading the test suite.
    """
    from src.benchmarks import all_benchmarks
    try:
        return [
            BenchmarkResponse(
                name=b.name, kind=b.kind, description=b.description,
                gates=list(b.gates), is_null=b.is_null,
                known_answer=_jsonable(b.truth()), checks=[])
            for b in all_benchmarks()
        ]
    except Exception as e:
        logger.error(f"Benchmark listing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not list benchmarks.")

@app.post("/api/v1/benchmarks/run", response_model=BenchmarkSuiteResponse)
async def run_benchmarks(root_seed: int = DEFAULT_BENCHMARK_SEED,
                         name: Optional[List[str]] = Query(None)):
    """Run the suite and report the three outcomes separately.

    NOT_YET_RUNNABLE is never folded into PASS: a summary that did so would let "all green"
    mean "we never looked".
    """
    from src.benchmarks import all_benchmarks, run_all
    from src.benchmarks.runner import null_failures, summarise
    try:
        known = {b.name for b in all_benchmarks()}
        # An unknown name previously filtered to nothing and returned 200 with an empty
        # suite - a silent no-op that reads as "everything passed". A gate that can be
        # made to disappear by a typo is not a gate.
        if name:
            unknown = [n for n in name if n not in known]
            if unknown:
                raise HTTPException(
                    status_code=404,
                    detail="Unknown benchmark(s): %s. Available: %s"
                           % (", ".join(sorted(unknown)), ", ".join(sorted(known))))
        results = run_all(root_seed=root_seed, names=name)
        by_name = {b.name: b for b in all_benchmarks()}
        counts = summarise(results)
        payload = []
        for bench_name, checks in results.items():
            b = by_name[bench_name]
            payload.append(BenchmarkResponse(
                name=b.name, kind=b.kind, description=b.description,
                gates=list(b.gates), is_null=b.is_null,
                known_answer=_jsonable(b.truth()),
                checks=[BenchmarkCheckResponse(
                    stage=c.stage, outcome=c.outcome.value, detail=c.detail,
                    measured=_jsonable(c.measured) if c.measured else None)
                    for c in checks]))
        return BenchmarkSuiteResponse(
            root_seed=root_seed,
            passed=counts["PASS"], failed=counts["FAIL"],
            not_yet_runnable=counts["NOT_YET_RUNNABLE"],
            null_failures=["%s / %s: %s" % (n, c.stage, c.detail)
                           for n, c in null_failures(results)],
            benchmarks=payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark run error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Benchmark suite failed to run.")

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
        forecast_tensor = torch.tensor(request.forecast_data, dtype=torch.float64)
        gt_tensor = torch.tensor(request.ground_truth_data, dtype=torch.float64)
        
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
            f_tensor = torch.tensor(request.forecast_data, dtype=torch.float64)
            g_tensor = torch.tensor(request.ground_truth_data, dtype=torch.float64)
            
            scale_decomp = ErrorDecompositionEngine.decompose_by_scale(f_tensor, g_tensor)
            boundary_decomp = ErrorDecompositionEngine.decompose_by_boundary(
                f_tensor, g_tensor, boundary_width=request.boundary_width or 8
            )
            
        if request.forecast_series is not None and request.ground_truth_series is not None and request.lead_times is not None:
            f_series = [torch.tensor(f, dtype=torch.float64) for f in request.forecast_series]
            g_series = [torch.tensor(g, dtype=torch.float64) for g in request.ground_truth_series]
            
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
                created_at=h.created_at.isoformat(),
            p_value=h.p_value,
            q_value=h.q_value,
            n_tests=h.n_tests,
            statistics=h.statistics,
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
                created_at=h.created_at.isoformat(),
            p_value=h.p_value,
            q_value=h.q_value,
            n_tests=h.n_tests,
            statistics=h.statistics,
        ) for h in hypotheses
        ]
    except Exception as e:
        logger.error(f"Error retrieving proposals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving proposals.")
