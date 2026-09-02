"""Metadata-only browser planning for bounded Copernicus CDS requests (TG18.1).

Planning is deliberately separate from acquisition.  These routes validate and hash the exact
request, enumerate its monthly queue shards and conservatively price its storage footprint.  No
CDS client is constructed and no network operation is possible from this module.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.errors import SpectralEarthError, classify
from src.data_layer.cds_source import (
    CDS_DATASET,
    CDS_VARIABLES,
    PRESSURE_LEVELS,
    CDSRegionalRequest,
    estimate_cds_storage,
    plan_monthly_shards,
)
from src.data_layer.zarr_source import NETWORK_ENV_VAR, check_crop_size


router = APIRouter(prefix="/api/v1/data/cds", tags=["cds"])


class CDSPlanRequest(BaseModel):
    variables: List[str] = Field(default_factory=lambda: ["t"])
    date_start: str = "2018-01-01"
    date_end: str = "2023-12-31"
    hours_utc: List[int] = Field(default_factory=lambda: [0, 6, 12, 18])
    lat_min: float = -60.0
    lat_max: float = -20.0
    lon_min: float = 140.0
    lon_max: float = 180.0
    pressure_levels: List[int] = Field(default_factory=lambda: [850])
    grid_degrees: float = 0.25
    n_levels_analysis: int = 3


def _spec(value: CDSPlanRequest) -> CDSRegionalRequest:
    record = value.dict()
    record["variables"] = tuple(record["variables"])
    record["hours_utc"] = tuple(record["hours_utc"])
    record["pressure_levels"] = tuple(record["pressure_levels"])
    return CDSRegionalRequest(**record)


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


@router.get("")
async def cds_capabilities() -> Dict[str, Any]:
    """The complete planner vocabulary and its hard no-network boundary."""
    defaults = CDSPlanRequest()
    return {
        "schema": "cds-browser-planner/v1",
        "dataset": CDS_DATASET,
        "variables": [
            {"id": name, "cds_name": CDS_VARIABLES[name]}
            for name in CDS_VARIABLES
        ],
        "pressure_levels": list(PRESSURE_LEVELS),
        "defaults": defaults.dict(),
        "network_env_var": NETWORK_ENV_VAR,
        "network_enabled": os.environ.get(NETWORK_ENV_VAR, "").strip().lower()
        in ("1", "true", "yes"),
        "planner_network_used": False,
        "execution_status": "NOT_MOUNTED",
        "workflow": [
            "Configure the exact regional request.",
            "Validate its grid, monthly shards, frame count and conservative storage ceiling.",
            "Review the request digest before creating a durable acquisition job.",
        ],
        "claim_boundary": (
            "A valid plan is request provenance, not acquired data, source agreement, analysis, "
            "evidence or a finding. This endpoint never contacts CDS."
        ),
    }


@router.post("/plan")
async def plan_cds_request(value: CDSPlanRequest) -> Dict[str, Any]:
    """Validate and price one frozen request without constructing a CDS client."""
    try:
        spec = _spec(value)
        shards = plan_monthly_shards(spec)
        estimate = estimate_cds_storage(spec, shards=shards)
        height = int(estimate["latitude_points_upper_bound"])
        width = int(estimate["longitude_points_upper_bound"])
        try:
            geometry = {
                "status": "MEETS_GENERIC_R13_HEURISTIC",
                "assessment": check_crop_size(height, width, spec.n_levels_analysis),
            }
        except SpectralEarthError as error:
            # Acquisition geometry is still a valid request.  This is an analysis-readiness
            # refusal, kept beside (not confused with) the transfer plan.
            geometry = {
                "status": "DOES_NOT_MEET_GENERIC_R13_HEURISTIC",
                "assessment": classify(error),
            }
        return {
            "schema": "cds-browser-plan/v1",
            "request": spec.to_provenance(),
            "request_sha256": spec.request_sha256(),
            "monthly_shards": [shard.to_provenance() for shard in shards],
            "storage_estimate": estimate,
            "analysis_geometry": geometry,
            "network_used": False,
            "execution_status": "NOT_MOUNTED",
            "next_action": (
                "Review this immutable plan. Durable browser execution, progress and resume "
                "are a separate job surface and are not mounted yet."
            ),
            "claim_boundary": (
                "This is metadata-only planning. It acquired no values and establishes no "
                "agreement, effect, evidence or finding."
            ),
        }
    except SpectralEarthError as error:
        raise _handle(error) from error
