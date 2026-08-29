"""Bounded TESS light-curve acquisition API (TG13.1)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt, StrictStr, validator

from extensions.tess_lightcurve import TESS, register as register_tess_domain
from src.core.errors import SpectralEarthError, classify
from src.core.domain import refusals_for
from src.core.dataset_capabilities import build_profile
from src.core.onboarding import DOMAIN_ONBOARDINGS, geometry_offers_metric
from src.data_layer.lightcurves import LightCurveSpec, persist_collection
from src.data_layer.tess_source import (acquire_tess, inspect_tess_query,
                                        register_tess_source)

router = APIRouter(prefix="/api/v1/lightcurves", tags=["lightcurves"])
REGISTERED_DOMAIN = register_tess_domain()
REGISTERED_SOURCE = register_tess_source()


class LightCurveSpecRequest(BaseModel):
    target_id: StrictStr
    sectors: List[StrictInt] = Field(..., min_items=1, max_items=8)
    flux_column: StrictStr = "PDCSAP_FLUX"
    quality_policy: StrictStr = "quality_zero"
    max_products: StrictInt = 4
    max_download_bytes: StrictInt = 64 * 1024 * 1024

    @validator("target_id", "flux_column", "quality_policy")
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    def contract(self) -> LightCurveSpec:
        return LightCurveSpec(**self.dict())


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


@router.get("")
async def capabilities() -> Dict[str, Any]:
    return {
        "schema": "spectral.lightcurve-capabilities.v1",
        "domain": TESS.describe(), "refusals": list(refusals_for(TESS)),
        "source": REGISTERED_SOURCE.describe(),
        "workflow": ["inspect_metadata", "acquire_bounded_products", "review_quality_flags"],
        "claim_boundary": (
            "A catalogue row or acquisition receipt is not a photometric finding. "
            "No default calendar is fitted, no gap is interpolated and no claim rung moves."),
    }


@router.post("/inspect")
async def inspect(request: LightCurveSpecRequest) -> Dict[str, Any]:
    try:
        return inspect_tess_query(request.contract())
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/acquire")
async def acquire(request: LightCurveSpecRequest) -> Dict[str, Any]:
    try:
        collection = acquire_tess(request.contract())
        publication = persist_collection(collection)
    except SpectralEarthError as error:
        raise _handle(error)
    onboarding = DOMAIN_ONBOARDINGS.get(TESS.name)
    profile = build_profile(
        kind="lightcurve_collection", phase="acquired",
        identity=collection.collection_sha256(), domain=TESS.name,
        facts={
            "sample_table": False, "channel_series": False, "profile_collection": False,
            "spatial_grid_2d": False,
            "physical_metric": geometry_offers_metric(onboarding.geometry),
            "ordered_time_axis": True, "regular_cadence": False,
            "irregular_support": True, "transform_compatible": False,
            "precedence_admissible": False, "independent_samples": False,
        },
        basis={"collection_sha256": collection.collection_sha256(),
               "geometry": onboarding.geometry, "time_scale": "BJD_TDB",
               "quality_flags_retained": collection.describe()["quality_flagged"]})
    return {
        "schema": "spectral.lightcurve-acquisition.v1",
        "collection": collection.describe(), "publication": publication,
        "analysis_readiness": {
            "association": "available after an explicit channel assembly and quality policy",
            "precedence": "refused: tess_lightcurve declares lag_policy=none",
            "calendar_climatology": "refused: no_natural_cycle",
            "frame_lags": "refused: irregular_sampling",
            "sector_gaps": "retained; never interpolated",
        },
        "capability_profile": profile,
        "claim_boundary": (
            "This receipt identifies calibrated archive bytes and retained quality flags. "
            "It establishes no variability, period, association, precursor or mechanism."),
    }


__all__ = ["LightCurveSpecRequest", "router"]
