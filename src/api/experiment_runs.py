"""TG17.6 HTTP surface for the content-addressed, resumable experiment orchestrator.

Every route here is idempotent in the same sense the state machine is. `POST /experiment-runs`
with a manifest that already has a run resumes it and returns the same identity rather than
starting a second one, which is what makes a browser refresh - or a second tab, or a retried
request after a dropped connection - harmless.

The one thing a route may not do is decide what work happens. Stage workers come from the
registered suites in `src.core.run_workers`, named by the caller from a list the server publishes;
a request cannot supply behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.errors import SpectralEarthError
from src.core.experiment_manifest import CrossDomainExperimentSpec, ManifestStore
from src.core.experiment_run import (RunStore, describe_state_machine, run_identity)
from src.core.run_workers import build_suite, describe_suites


router = APIRouter(prefix="/api/v1/experiment-runs", tags=["experiment-runs"])


class ExecuteRequest(BaseModel):
    worker_suite: str = Field(..., min_length=1)


class CancelRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


class EditableCopyRequest(BaseModel):
    draft_id: str = Field(..., min_length=1, max_length=64)


def _root(request: Request) -> Path:
    root = getattr(request.app.state, "experiment_run_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_RUN_DIR", "data/experiment_runs")
    return Path(root)


def _store(request: Request) -> RunStore:
    return RunStore(_root(request))


def _manifest_store(request: Request) -> ManifestStore:
    root = getattr(request.app.state, "experiment_manifest_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_MANIFEST_DIR", "data/experiment_manifests")
    return ManifestStore(Path(root))


def _load(request: Request, run_id: str):
    try:
        return _store(request).load(run_id)
    except SpectralEarthError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("")
async def contract(request: Request) -> Dict[str, Any]:
    """The machine, the suites and the runs on disk - generated, never hand-listed."""
    return {"schema": "experiment-runs/v1",
            "state_machine": describe_state_machine(),
            "worker_suites": list(describe_suites()),
            "runs": _store(request).list_runs(),
            "available_now": ["create or resume a run from a manifest",
                              "preflight and freeze", "execute a registered rehearsal suite",
                              "retry an operational failure", "cancel",
                              "an editable copy of a refused run", "progress and receipt"],
            "not_yet_available": ["live acquisition and translation workers"],
            "claim_boundary": "A completed run is an executed plan, not admitted evidence. The "
                              "registered suites acquire nothing."}


@router.post("")
async def create_or_resume(spec: CrossDomainExperimentSpec, request: Request) -> Dict[str, Any]:
    """Open the run this manifest identifies, preflight it, and report where it stands.

    Not `create`. The identity is the manifest, so posting the same manifest twice cannot produce
    two runs, and the response says which of the two it was.
    """
    store = _store(request)
    run = store.open(spec)
    existed = run.state != "DRAFT"
    try:
        run.preflight()
    except SpectralEarthError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**run_identity(spec), "resumed": existed, "progress": run.progress(),
            "receipt": run.receipt()}


@router.get("/{run_id}")
async def receipt(run_id: str, request: Request) -> Dict[str, Any]:
    return _load(request, run_id).receipt()


@router.get("/{run_id}/progress")
async def progress(run_id: str, request: Request) -> Dict[str, Any]:
    """What a watcher may see mid-run: stage, component, digest, bounded work, remediation."""
    return _load(request, run_id).progress()


@router.post("/{run_id}/execute")
async def execute(run_id: str, body: ExecuteRequest, request: Request) -> Dict[str, Any]:
    run = _load(request, run_id)
    try:
        return run.execute(build_suite(body.worker_suite, run=run))
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{run_id}/retry")
async def retry(run_id: str, body: ExecuteRequest, request: Request) -> Dict[str, Any]:
    run = _load(request, run_id)
    try:
        return run.retry(build_suite(body.worker_suite, run=run))
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, body: CancelRequest, request: Request) -> Dict[str, Any]:
    run = _load(request, run_id)
    try:
        return run.cancel(body.reason)
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{run_id}/editable-copy")
async def editable_copy(run_id: str, body: EditableCopyRequest,
                        request: Request) -> Dict[str, Any]:
    """The remedy for a refusal. The frozen run is not touched, and the response says so."""
    run = _load(request, run_id)
    try:
        return run.editable_copy(_manifest_store(request), body.draft_id)
    except (SpectralEarthError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
