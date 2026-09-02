"""Metadata-only browser planning for bounded Copernicus CDS requests (TG18.1).

Planning is deliberately separate from acquisition.  These routes validate and hash the exact
request, enumerate its monthly queue shards and conservatively price its storage footprint.  No
CDS client is constructed and no network operation is possible from this module.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StrictBool

from src.core.errors import SpectralEarthError, classify
from src.core.cds_job import CDSJobStore
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
_threads: Dict[str, threading.Thread] = {}
_threads_lock = threading.RLock()


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


class CDSSubmitRequest(BaseModel):
    request: CDSPlanRequest
    confirm_request_sha256: str = Field(..., min_length=64, max_length=64)
    confirm_network_access: StrictBool


class CDSCancelRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


def _spec(value: CDSPlanRequest) -> CDSRegionalRequest:
    record = value.dict()
    record["variables"] = tuple(record["variables"])
    record["hours_utc"] = tuple(record["hours_utc"])
    record["pressure_levels"] = tuple(record["pressure_levels"])
    return CDSRegionalRequest(**record)


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


def _root(request: Request) -> Path:
    root = getattr(request.app.state, "cds_job_dir", None)
    return Path(root or os.getenv("CDS_JOB_DIR", "data/cds_jobs"))


def _store(request: Request) -> CDSJobStore:
    return CDSJobStore(_root(request))


def _network_enabled() -> bool:
    return os.environ.get(NETWORK_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def _active_job_ids() -> set[str]:
    with _threads_lock:
        return {job_id for job_id, worker in _threads.items() if worker.is_alive()}


def _launch(request: Request, store: CDSJobStore, job_id: str) -> None:
    with _threads_lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        runner = getattr(request.app.state, "cds_job_runner", None)
        worker = threading.Thread(
            target=store.execute, kwargs={"job_id": job_id, "runner": runner},
            name="cds-job-%s" % job_id[-12:], daemon=True)
        _threads[job_id] = worker
        worker.start()


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
        "execution_status": "AVAILABLE" if _network_enabled() else "NETWORK_DISABLED",
        "workflow": [
            "Configure the exact regional request.",
            "Validate its grid, monthly shards, frame count and conservative storage ceiling.",
            "Confirm the exact request digest and explicit network use before submission.",
            "Monitor durable monthly progress; cancel or resume without discarding verified shards.",
            "Inspect the acquisition record only after every shard passes integrity checks.",
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
            "execution_status": "READY_TO_SUBMIT" if _network_enabled() else "NETWORK_DISABLED",
            "submission_confirmation": {
                "confirm_request_sha256": spec.request_sha256(),
                "confirm_network_access": True,
                "statement": "I reviewed this exact request and authorize network acquisition.",
            },
            "next_action": (
                "Review the immutable digest and storage ceiling, then explicitly authorize "
                "network submission. Planning itself has still acquired nothing."
            ),
            "claim_boundary": (
                "This is metadata-only planning. It acquired no values and establishes no "
                "agreement, effect, evidence or finding."
            ),
        }
    except SpectralEarthError as error:
        raise _handle(error) from error


@router.get("/jobs")
async def list_cds_jobs(request: Request) -> Dict[str, Any]:
    jobs = _store(request).list(active_job_ids=_active_job_ids())
    return {"schema": "cds-acquisition-jobs/v1", "jobs": jobs,
            "network_enabled": _network_enabled(),
            "claim_boundary": "Jobs and acquisition records are operational provenance, not evidence."}


@router.post("/jobs", status_code=202)
async def submit_cds_job(body: CDSSubmitRequest, request: Request) -> Dict[str, Any]:
    """Persist and launch only after an exact digest and network acknowledgement."""
    try:
        spec = _spec(body.request)
        digest = spec.request_sha256()
        if body.confirm_request_sha256 != digest:
            raise HTTPException(
                status_code=409,
                detail="Submission confirmation does not match the exact planned request digest.")
        if body.confirm_network_access is not True:
            raise HTTPException(
                status_code=409, detail="Explicit network-access confirmation is required.")
        if not _network_enabled():
            raise HTTPException(
                status_code=409,
                detail="CDS network execution is disabled on this server; planning remains available.")
        store = _store(request)
        job, resumed = store.submit(spec)
        if resumed and job["state"] in {"QUEUED", "RUNNING", "CANCELLING"} \
                and job["job_id"] not in _active_job_ids():
            job = store.load(job["job_id"], reconcile=True)
        if job["state"] == "QUEUED":
            _launch(request, store, job["job_id"])
        return {**job, "existing_job": resumed}
    except SpectralEarthError as error:
        raise _handle(error) from error


@router.get("/jobs/{job_id}")
async def get_cds_job(job_id: str, request: Request) -> Dict[str, Any]:
    try:
        return _store(request).load(job_id, reconcile=job_id not in _active_job_ids())
    except SpectralEarthError as error:
        raise _handle(error) from error


@router.post("/jobs/{job_id}/cancel")
async def cancel_cds_job(job_id: str, body: CDSCancelRequest, request: Request) -> Dict[str, Any]:
    try:
        store = _store(request)
        store.load(job_id, reconcile=job_id not in _active_job_ids())
        return store.cancel(job_id, body.reason)
    except SpectralEarthError as error:
        raise _handle(error) from error


@router.post("/jobs/{job_id}/resume", status_code=202)
async def resume_cds_job(job_id: str, request: Request) -> Dict[str, Any]:
    if not _network_enabled():
        raise HTTPException(
            status_code=409, detail="CDS network execution is disabled; this job remains resumable.")
    try:
        store = _store(request)
        store.load(job_id, reconcile=job_id not in _active_job_ids())
        job = store.prepare_resume(job_id)
        if job["state"] == "QUEUED":
            _launch(request, store, job_id)
        return job
    except SpectralEarthError as error:
        raise _handle(error) from error


@router.get("/jobs/{job_id}/record")
async def get_cds_acquisition_record(job_id: str, request: Request) -> Dict[str, Any]:
    try:
        job = _store(request).load(job_id, reconcile=job_id not in _active_job_ids())
        if job["state"] != "COMPLETE" or not job.get("acquisition_record"):
            raise HTTPException(
                status_code=409,
                detail="No acquisition record exists until every planned shard is verified.")
        return job["acquisition_record"]
    except SpectralEarthError as error:
        raise _handle(error) from error
