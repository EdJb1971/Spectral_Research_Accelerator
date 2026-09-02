"""Durable, content-addressed orchestration for browser-submitted CDS acquisitions.

The request digest is the job identity.  Paths are chosen by the server, state is atomically
persisted, and a process that disappears while a transfer is running leaves an INTERRUPTED job
that can be explicitly resumed.  A completed job alone receives an acquisition record.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_source import (
    CDSAcquisitionCancelled,
    CDSRegionalRequest,
    acquire_cds_shards,
    plan_monthly_shards,
    preflight_cds_storage,
)


JOB_SCHEMA = "cds-acquisition-job/v1"
RECORD_SCHEMA = "cds-acquisition-record/v1"
ACTIVE_STATES = {"QUEUED", "RUNNING", "CANCELLING"}
RESUMABLE_STATES = {"INTERRUPTED", "CANCELLED", "FAILED"}
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class CDSJobStore:
    """A small durable job journal rooted entirely in server-owned storage."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def job_id(spec: CDSRegionalRequest) -> str:
        return "cds-%s" % spec.request_sha256()

    def _directory(self, job_id: str) -> Path:
        expected = "cds-"
        digest = job_id[len(expected):] if job_id.startswith(expected) else ""
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise InvalidParameterError("job_id", job_id, "a complete CDS content-addressed job id")
        return self.root / job_id

    def _path(self, job_id: str) -> Path:
        return self._directory(job_id) / "job.json"

    def download_dir(self, job_id: str) -> Path:
        return self._directory(job_id) / "downloads"

    def _read(self, job_id: str) -> Dict[str, Any]:
        path = self._path(job_id)
        if not path.exists():
            raise InvalidParameterError("job_id", job_id, "an existing CDS acquisition job")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataSourceError("CDS job journal is unreadable", job_id=job_id) from exc
        if value.get("schema") != JOB_SCHEMA or value.get("job_id") != job_id:
            raise DataSourceError("CDS job journal identity is invalid", job_id=job_id)
        request = CDSRegionalRequest.from_provenance(value.get("request", {}))
        if request.request_sha256() != value.get("request_sha256") or self.job_id(request) != job_id:
            raise DataSourceError("CDS job journal request digest is invalid", job_id=job_id)
        record = value.get("acquisition_record")
        if record is not None:
            unsigned = dict(record)
            recorded_digest = unsigned.pop("record_sha256", None)
            if recorded_digest != _digest(unsigned):
                raise DataSourceError("CDS acquisition record SHA-256 is invalid", job_id=job_id)
        if (value.get("state") == "COMPLETE") != (record is not None):
            raise DataSourceError(
                "CDS completion state and acquisition record disagree", job_id=job_id)
        return value

    def _public(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(value)
        result.pop("cancel_requested", None)
        result["storage"] = {
            "ownership": "SERVER_MANAGED",
            "namespace": "cds_acquisition_jobs",
            "job_id": result["job_id"],
            "client_path_accepted": False,
        }
        result["progress"] = {
            "completed_shards": int(result.get("completed_shards", 0)),
            "total_shards": int(result["total_shards"]),
            "fraction": (int(result.get("completed_shards", 0)) /
                         max(1, int(result["total_shards"]))),
            "current_shard": result.get("current_shard"),
        }
        result["resumable"] = result["state"] in RESUMABLE_STATES
        result["cancellation_boundary"] = (
            "Cancellation is cooperative between monthly shards; an in-flight CDS request may "
            "finish before the job stops. Verified shards are retained for resume."
        )
        result["claim_boundary"] = (
            "A completed acquisition record proves bounded transfer and shard integrity only. "
            "It is not analysis, evidence, agreement, an effect or a finding."
        )
        return result

    def load(self, job_id: str, *, reconcile: bool = True) -> Dict[str, Any]:
        with _LOCK:
            value = self._read(job_id)
            if reconcile and value["state"] in ACTIVE_STATES:
                value.update(state="INTERRUPTED", updated_at=_now(), current_shard=None,
                             message="The server process ended while this job was active. Resume explicitly.")
                _atomic_write(self._path(job_id), value)
            return self._public(value)

    def list(self, *, active_job_ids: Optional[set[str]] = None) -> list[Dict[str, Any]]:
        active = active_job_ids or set()
        jobs = []
        for path in sorted(self.root.glob("cds-*/job.json")):
            try:
                jobs.append(self.load(path.parent.name, reconcile=path.parent.name not in active))
            except (DataSourceError, InvalidParameterError):
                continue
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    def submit(self, spec: CDSRegionalRequest) -> tuple[Dict[str, Any], bool]:
        job_id = self.job_id(spec)
        with _LOCK:
            path = self._path(job_id)
            if path.exists():
                return self._public(self._read(job_id)), True
            preflight = preflight_cds_storage(spec, download_dir=self.download_dir(job_id))
            shards = plan_monthly_shards(spec)
            now = _now()
            value = {
                "schema": JOB_SCHEMA, "job_id": job_id, "request_sha256": spec.request_sha256(),
                "request": spec.to_provenance(), "state": "QUEUED", "created_at": now,
                "updated_at": now, "started_at": None, "completed_at": None,
                "total_shards": len(shards), "completed_shards": 0, "current_shard": None,
                "downloaded_shards_this_attempt": 0, "resumed_shards_this_attempt": 0,
                "attempts": 0, "cancel_requested": False, "message": "Queued for CDS acquisition.",
                "error": None, "storage_preflight": preflight, "acquisition_record": None,
            }
            _atomic_write(path, value)
            return self._public(value), False

    def prepare_resume(self, job_id: str) -> Dict[str, Any]:
        with _LOCK:
            value = self._read(job_id)
            if value["state"] == "COMPLETE":
                return self._public(value)
            if value["state"] in ACTIVE_STATES:
                raise InvalidParameterError("job state", value["state"],
                                            "a stopped FAILED, CANCELLED or INTERRUPTED job")
            if value["state"] not in RESUMABLE_STATES:
                raise InvalidParameterError("job state", value["state"], "a resumable job")
            spec = CDSRegionalRequest.from_provenance(value["request"])
            value["storage_preflight"] = preflight_cds_storage(
                spec, download_dir=self.download_dir(job_id))
            value.update(state="QUEUED", updated_at=_now(), cancel_requested=False,
                         current_shard=None, message="Queued to resume verified monthly shards.",
                         error=None)
            _atomic_write(self._path(job_id), value)
            return self._public(value)

    def cancel(self, job_id: str, reason: str) -> Dict[str, Any]:
        if not reason.strip():
            raise InvalidParameterError("reason", reason, "a non-empty cancellation reason")
        with _LOCK:
            value = self._read(job_id)
            if value["state"] == "COMPLETE":
                raise InvalidParameterError("job state", "COMPLETE", "an active or resumable job")
            if value["state"] in {"CANCELLED", "FAILED", "INTERRUPTED"}:
                return self._public(value)
            value.update(state="CANCELLING" if value["state"] == "RUNNING" else "CANCELLED",
                         cancel_requested=True, updated_at=_now(), message=reason.strip())
            _atomic_write(self._path(job_id), value)
            return self._public(value)

    def cancellation_requested(self, job_id: str) -> bool:
        with _LOCK:
            return bool(self._read(job_id).get("cancel_requested"))

    def _progress(self, job_id: str, progress: Mapping[str, Any]) -> None:
        with _LOCK:
            value = self._read(job_id)
            value.update(completed_shards=int(progress["completed_shards"]),
                         current_shard=progress.get("current_shard"), updated_at=_now())
            _atomic_write(self._path(job_id), value)

    def execute(self, job_id: str, *, runner: Optional[Callable[..., Mapping[str, Any]]] = None) -> None:
        """Run in a worker thread. Every transition is durable before network can occur."""
        with _LOCK:
            value = self._read(job_id)
            if value["state"] != "QUEUED":
                return
            value.update(state="RUNNING", attempts=int(value["attempts"]) + 1,
                         started_at=value.get("started_at") or _now(), updated_at=_now(),
                         message="CDS acquisition is running.")
            _atomic_write(self._path(job_id), value)
            spec = CDSRegionalRequest.from_provenance(value["request"])
        operation = runner or acquire_cds_shards
        try:
            acquired = operation(
                spec, self.download_dir(job_id), allow_network=True,
                progress_callback=lambda item: self._progress(job_id, item),
                cancellation_requested=lambda: self.cancellation_requested(job_id),
            )
            run = acquired["run"]
            completed = acquired.get("completed_shards", {})
            planned = {shard.filename: shard for shard in plan_monthly_shards(spec)}
            if run.get("complete") is not True or set(completed) != set(planned):
                raise DataSourceError(
                    "CDS worker returned before every planned shard was verified", job_id=job_id)
            for name, item in completed.items():
                sha = item.get("sha256")
                if (item.get("request_sha256") != planned[name].request_sha256
                        or not isinstance(sha, str) or len(sha) != 64
                        or any(char not in "0123456789abcdef" for char in sha)
                        or isinstance(item.get("bytes"), bool) or int(item.get("bytes", -1)) < 0):
                    raise DataSourceError(
                        "CDS worker returned invalid shard integrity provenance", job_id=job_id,
                        shard=name)
            observed_total = sum(int(item["bytes"]) for item in completed.values())
            if int(run.get("total_bytes", -1)) != observed_total:
                raise DataSourceError(
                    "CDS worker byte total disagrees with verified shards", job_id=job_id)
            record = {
                "schema": RECORD_SCHEMA, "job_id": job_id,
                "request_sha256": spec.request_sha256(), "request": spec.to_provenance(),
                "completed_at": _now(), "completed_shards": len(completed),
                "total_bytes": observed_total,
                "shards": [{"filename": name, "sha256": item["sha256"],
                            "bytes": int(item["bytes"]), "request_sha256": item["request_sha256"]}
                           for name, item in sorted(completed.items())],
                "claim_boundary": "Transfer and integrity provenance only; not analysis or evidence.",
            }
            record["record_sha256"] = _digest(record)
            with _LOCK:
                value = self._read(job_id)
                value.update(state="COMPLETE", completed_at=record["completed_at"],
                             updated_at=record["completed_at"], completed_shards=len(completed),
                             current_shard=None, cancel_requested=False,
                             downloaded_shards_this_attempt=int(run["downloaded_shards"]),
                             resumed_shards_this_attempt=int(run["resumed_shards"]),
                             message="Acquisition complete; immutable record available.",
                             acquisition_record=record, error=None)
                _atomic_write(self._path(job_id), value)
        except CDSAcquisitionCancelled:
            with _LOCK:
                value = self._read(job_id)
                value.update(state="CANCELLED", updated_at=_now(), current_shard=None,
                             message="Cancelled between monthly shards; verified shards retained.")
                _atomic_write(self._path(job_id), value)
        except Exception as exc:
            with _LOCK:
                value = self._read(job_id)
                value.update(state="FAILED", updated_at=_now(), current_shard=None,
                             message="Acquisition failed; verified shards remain resumable.",
                             error={"type": type(exc).__name__, "detail": str(exc)})
                _atomic_write(self._path(job_id), value)
