"""HTTP acquisition and reduction surface for irregular profiles (TG12.2b-d)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.analysis_engine.cross_scale import masked_frame_lag_assessment
from src.core.errors import SpectralEarthError, classify
from src.core.dataset_capabilities import build_profile
from src.core.onboarding import DOMAIN_ONBOARDINGS, geometry_offers_metric
from src.data_layer.argo_source import (ARGO_DOI, NETWORK_ENV_VAR, acquire_profiles,
                                        inspect_profile_query, network_enabled,
                                        persist_collection, register_argo_source)
from src.data_layer.profile_reductions import PROFILE_REDUCTIONS, reduce_profiles
from src.data_layer.profiles import PROFILE_SOURCES, ProfileSpec


router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])
REGISTERED_SOURCE = register_argo_source()


class ProfileSpecRequest(BaseModel):
    source: str = "argo_gdac_erddap"
    time_start: str
    time_end: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    pressure_min_dbar: float = 0.0
    pressure_max_dbar: float = 200.0
    variables: List[str] = Field(default_factory=lambda: ["temperature"])
    max_profiles: int = 200


class ProfileReductionRequest(BaseModel):
    name: str = "per_float_at_pressure"
    configuration: Dict[str, Any] = Field(
        default_factory=lambda: {"variable": "temperature", "pressure_dbar": 100.0,
                                 "max_interpolation_gap_dbar": 25.0})


class ProfileAcquisitionRequest(BaseModel):
    spec: ProfileSpecRequest
    reduction: ProfileReductionRequest = Field(default_factory=ProfileReductionRequest)


def _spec(value: ProfileSpecRequest) -> ProfileSpec:
    data = value.dict()
    # Registry lookup is the refusal and discovery seam; the first implementation happens to
    # be Argo, but the request is not an if/elif over source names.
    PROFILE_SOURCES.get(str(data["source"]))
    return ProfileSpec(**data)


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


def _analysis_readiness(result: Any) -> Dict[str, Any]:
    series, declaration = result.series, result.declaration
    present = series.present
    if present is not None and not bool(np.all(np.asarray(present, dtype=bool))):
        assessment = masked_frame_lag_assessment(
            np.asarray(present, dtype=bool), lags=[1], estimator="mutual_information")
        return {
            "frame_lag_admissible": False, "decision": "refuse_frame_lags",
            "basis": ["irregular_sampling", "non_stationary_support"],
            "reason": (
                "The reduction preserves floats starting and stopping on an irregular union "
                "clock. A shift in observed positions is not a shift in physical time; "
                "compacting, binning or interpolation would invent adjacency or simultaneity. "
                "The record is valid and reproducible, but the current frame-lag estimator "
                "must refuse it until a physical-time estimator exists."),
            "contiguous_candidate": assessment,
        }
    if "irregular_sampling" in declaration.violations \
            and not bool(series.provenance.get("clock_regular")):
        return {
            "frame_lag_admissible": False, "decision": "refuse_frame_lags",
            "basis": ["irregular_sampling"],
            "reason": (
                "The reduced clock is irregular, so a lag in rows is not a duration. The "
                "instrument does not resample it merely to make the existing estimator run."),
        }
    return {"frame_lag_admissible": True, "decision": "admissible_for_declared_lag_audit",
            "basis": [], "reason": "No data-shape refusal was triggered at this boundary."}


def _preview(result: Any, cap: int = 400) -> Dict[str, Any]:
    series = result.series
    measure_name = next(iter(series.measures))
    matrix = series.to_matrix(measure_name)
    stop = min(series.n_times, cap)
    present = None if series.present is None else np.asarray(series.present, dtype=bool)
    channels = []
    for column, record in enumerate(series.channel_records):
        values = matrix[:stop, column]
        channels.append({
            "name": str(series.channels[column]), "usable": bool(record["usable"]),
            "present_count": int(record.get("present_count", np.isfinite(matrix[:, column]).sum())),
            "absent_count": int(record.get("absent_count", 0)),
            "presence": None if present is None else present[:stop, column].tolist(),
            "values": [None if not np.isfinite(value) else float(value) for value in values],
        })
    return {"measure": measure_name, "n_rows": series.n_times, "preview_rows": stop,
            "rows_withheld": series.n_times - stop,
            "times_seconds": np.asarray(series.times_seconds)[:stop].tolist(),
            "channels": channels}


@router.get("")
async def capabilities() -> Dict[str, Any]:
    return {
        "schema": "spectral.profiles.capabilities.v1",
        "sources": [entry.value.describe() for entry in PROFILE_SOURCES],
        "reductions": PROFILE_REDUCTIONS.describe(),
        "network_enabled": network_enabled(), "network_env_var": NETWORK_ENV_VAR,
        "source_doi": ARGO_DOI,
        "claim_boundary": (
            "A profile source is an acquisition path and a reduction is recorded arithmetic. "
            "Neither is an oceanographic finding; analysis refusals remain valid outcomes."),
    }


@router.post("/inspect")
async def inspect(request: ProfileAcquisitionRequest) -> Dict[str, Any]:
    try:
        spec = _spec(request.spec)
        # Validate the exact reduction configuration before even the metadata query.
        reduction = PROFILE_REDUCTIONS.get(request.reduction.name)
        # Full normalisation depends on the collection; registry lookup here still refuses a
        # misspelled reduction before network access.
        plan = inspect_profile_query(spec)
    except SpectralEarthError as error:
        raise _handle(error)
    return {**plan, "requested_reduction": {
        "name": request.reduction.name,
        "configuration": dict(request.reduction.configuration),
        "description": PROFILE_REDUCTIONS.entry(request.reduction.name).description,
        "capabilities": PROFILE_REDUCTIONS.entry(request.reduction.name).capabilities,
    }}


@router.post("/acquire")
async def acquire(request: ProfileAcquisitionRequest) -> Dict[str, Any]:
    try:
        spec = _spec(request.spec)
        collection = acquire_profiles(spec)
        publication = persist_collection(collection)
        from extensions.argo_float import ARGO
        result = reduce_profiles(collection, ARGO, request.reduction.name,
                                 request.reduction.configuration)
    except SpectralEarthError as error:
        raise _handle(error)
    readiness = _analysis_readiness(result)
    onboarding = DOMAIN_ONBOARDINGS.get(result.declaration.name)
    profile = build_profile(
        kind="profile_collection", phase="acquired",
        identity=str(result.describe()["content_sha256"]), domain=result.declaration.name,
        facts={
            "sample_table": False, "channel_series": False, "profile_collection": True,
            "spatial_grid_2d": False,
            "physical_metric": geometry_offers_metric(onboarding.geometry),
            "ordered_time_axis": True,
            "regular_cadence": bool(result.describe()["clock"]["regular"]),
            "irregular_support": True, "transform_compatible": False,
            "precedence_admissible": bool(readiness["frame_lag_admissible"]),
            "independent_samples": False,
        },
        basis={"collection_sha256": collection.collection_sha256(),
               "reduction": result.describe()["reduction"],
               "geometry": onboarding.geometry,
               "record_specific_lag_decision": readiness})
    return {
        "schema": "spectral.profile-acquisition.v1",
        "collection": collection.describe(), "publication": publication,
        "reduction": result.describe(), "preview": _preview(result),
        "analysis_readiness": readiness, "capability_profile": profile,
        "source_doi": ARGO_DOI,
        "claim_boundary": (
            "This receipt proves which public profile response was acquired, which QC policy "
            "was applied and which registered reduction produced the series. It does not "
            "establish an oceanographic result. A scientific refusal is not converted into "
            "a success state by the acquisition surface."),
    }


__all__ = ["ProfileAcquisitionRequest", "ProfileReductionRequest", "ProfileSpecRequest",
           "router"]
