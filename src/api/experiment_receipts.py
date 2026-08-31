"""HTTP transport for TG17.9 experiment bundles and their read-only replay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from src.core.errors import SpectralEarthError
from src.core.experiment_receipt import (capability_snapshot, publish_bundle, verify_bundle)
from src.core.experiment_run import RunStore


router = APIRouter(prefix="/api/v1/experiment-receipts", tags=["experiment-receipts"])


def _root(request: Request) -> Path:
    value = getattr(request.app.state, "experiment_run_dir", None)
    return Path(value if value is not None else os.getenv("EXPERIMENT_RUN_DIR",
                                                          "data/experiment_runs"))


def _run(request: Request, run_id: str):
    try:
        return RunStore(_root(request)).load(run_id)
    except SpectralEarthError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """Generated trust surface for every operation, adapter, refusal and receipt field."""
    return capability_snapshot()


@router.post("/runs/{run_id}/export")
async def export_run(run_id: str, request: Request) -> Dict[str, Any]:
    try:
        published = publish_bundle(_run(request, run_id), _root(request))
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    bundle = published["bundle"]
    return {"schema": "experiment-receipt-export/v1", "integrity": "VERIFIED",
            "bundle_sha256": bundle["bundle_sha256"],
            "report_sha256": bundle["methods_report"]["sha256"], "bundle": bundle,
            "claim_boundary": bundle["claim_boundary"]}


@router.get("/runs/{run_id}/methods", response_class=PlainTextResponse)
async def methods_report(run_id: str, request: Request) -> str:
    try:
        return publish_bundle(_run(request, run_id), _root(request))["bundle"][
            "methods_report"]["text"]
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/replay")
async def replay_bundle(body: Dict[str, Any]) -> Dict[str, Any]:
    """Verify and reconstruct an audit projection. This writes no run or evidence state."""
    try:
        return verify_bundle(body)
    except (SpectralEarthError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
