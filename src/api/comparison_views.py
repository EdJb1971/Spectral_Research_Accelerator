"""TG17.8 HTTP surface for the linked scientific comparison views.

Every route here takes a manifest and returns a *reading aid*: what a view asks, the axes it is
drawn on, the legend that distinguishes a candidate from a confirmation, the numbers behind the
picture and the sentence saying what may not be concluded from it. No route opens an archive, and
no route reports a measured value.

Two decisions worth stating, because both were the alternative that would have looked simpler:

*   **The run is looked for, never opened.** Exactly as `POST /experiment-composer/path/state`
    does, and for exactly the same reason: `RunStore.open` publishes a frozen manifest, and
    asking to *look at* a plan must never be the act that freezes it.

*   **`POST /readings/check` exists as its own route.** A client could infer admissibility from
    the contract's two lists, but then every client would carry its own copy of the rule and the
    refusal reason would be written twice. The reason a reading is refused is the scientifically
    interesting part of the refusal, so it is served rather than reconstructed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.adapters import register_all_adapters
from src.core.comparison_views import (COMPARISON_VIEWS, ViewContext, describe_views,
                                       assert_reading_admissible, linked_selection,
                                       ordered_encodings, render_all, render_view)
from src.core.errors import SpectralEarthError
from src.core.experiment_manifest import CrossDomainExperimentSpec, preflight_manifest
from src.core.experiment_run import RunStore, run_identity


router = APIRouter(prefix="/api/v1/comparison-views", tags=["comparison-views"])

#: Eager and idempotent, for the reason defect D35 recorded: a view's provenance row reads the
#: registered adapter, and a row that appears only after some other tab has been opened is a
#: provenance row that can be missing exactly when someone is checking it.
REGISTERED_ADAPTERS = register_all_adapters()


class SelectionRequest(BaseModel):
    manifest: Dict[str, Any]
    window: str = Field(..., min_length=1)
    domains: List[str] = Field(default_factory=list)


class ReadingRequest(BaseModel):
    mode: str = Field(..., min_length=1)
    reading: str = Field(..., min_length=1)


def _run_store(request: Request) -> RunStore:
    root = getattr(request.app.state, "experiment_run_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_RUN_DIR", "data/experiment_runs")
    return RunStore(Path(root))


def _receipt(request: Request, spec: CrossDomainExperimentSpec) -> Optional[Dict[str, Any]]:
    """The receipt of the run at this manifest's address, if one already exists.

    Looked for on disk rather than opened. A view is a read.
    """
    store = _run_store(request)
    identity = run_identity(spec)
    if not (store.root / "runs" / identity["run_id"] / "manifest.json").exists():
        return None
    try:
        return store.load(identity["run_id"]).receipt()
    except (SpectralEarthError, OSError, ValueError):
        return None


def _context(request: Request, spec: CrossDomainExperimentSpec) -> ViewContext:
    return ViewContext(spec=spec, preflight=preflight_manifest(spec),
                       receipt=_receipt(request, spec))


@router.get("")
async def views_contract() -> Dict[str, Any]:
    """The contract, generated from the registries so it cannot drift from what is served."""
    contract = describe_views()
    contract["routes"] = {
        "render_all": "POST /api/v1/comparison-views/render",
        "render_one": "POST /api/v1/comparison-views/render/{view_id}",
        "linked_selection": "POST /api/v1/comparison-views/linked-selection",
        "check_reading": "POST /api/v1/comparison-views/readings/check",
    }
    contract["not_yet_available"] = ["measured values from executed stage workers"]
    return contract


@router.get("/encodings")
async def encodings() -> Dict[str, Any]:
    """The legend, in its declared order, with the definition of each role."""
    return {"schema": describe_views()["schema"],
            "encodings": [encoding.describe() for encoding in ordered_encodings()],
            "why_three_channels": ("Colour, marker and word carry the same distinction. A reader "
                                   "who cannot use one channel still has two, and the difference "
                                   "between a candidate and a confirmation is the entire "
                                   "scientific content of the generate/confirm split.")}


@router.post("/render")
async def render_every_view(spec: CrossDomainExperimentSpec,
                            request: Request) -> Dict[str, Any]:
    try:
        return render_all(_context(request, spec))
    except SpectralEarthError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/render/{view_id}")
async def render_one_view(view_id: str, spec: CrossDomainExperimentSpec,
                          request: Request) -> Dict[str, Any]:
    if view_id not in COMPARISON_VIEWS:
        raise HTTPException(status_code=404,
                            detail="unknown comparison view %r; registered: %s"
                                   % (view_id, COMPARISON_VIEWS.names()))
    try:
        return render_view(view_id, _context(request, spec))
    except SpectralEarthError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/linked-selection")
async def selection(body: SelectionRequest, request: Request) -> Dict[str, Any]:
    """What one selected window contributes, in every domain's own native terms."""
    try:
        spec = CrossDomainExperimentSpec.parse_obj(body.manifest)
        return linked_selection(_context(request, spec), window=body.window,
                                domains=body.domains)
    except SpectralEarthError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/readings/check")
async def check_reading(body: ReadingRequest) -> Dict[str, Any]:
    """Whether a mode may draw a reading, and the reason when it may not."""
    try:
        assert_reading_admissible(body.mode, body.reading)
    except SpectralEarthError as exc:
        return {"mode": body.mode, "reading": body.reading, "renderable": False,
                "reason": str(exc)}
    return {"mode": body.mode, "reading": body.reading, "renderable": True,
            "reason": "",
            "claim_boundary": ("A renderable reading is one a picture may express. Whether this "
                               "study measured it is a separate question the run answers.")}
