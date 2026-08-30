"""TG17.1 HTTP surface for composing and preflighting one scientific manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from src.adapters import register_all_adapters
from src.benchmarks.structural_trajectory import known_answer_native, known_answer_preview
from src.core.adapter_conformance import ConformanceCase, run_conformance
from src.core.errors import SpectralEarthError
from src.core.experiment_adapter import EXPERIMENT_ADAPTERS, adapter_for
from src.core.experiment_manifest import (CrossDomainExperimentSpec, ManifestStore,
                                          flagship_recipe, manifest_envelope,
                                          manifest_sha256, preflight_manifest)


router = APIRouter(prefix="/api/v1/experiment-composer", tags=["experiment-composer"])

#: Eager and idempotent (defect D35): the domain selector must not depend on which tab a
#: researcher opened first. A domain that appears only after some other route has run is a
#: domain whose declared refusals can be missed.
REGISTERED_ADAPTERS = register_all_adapters()


def _store(request: Request) -> ManifestStore:
    root = getattr(request.app.state, "experiment_manifest_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_MANIFEST_DIR", "data/experiment_manifests")
    return ManifestStore(Path(root))


@router.get("")
async def composer_contract() -> Dict[str, Any]:
    return {
        "schema": "experiment-composer/v1",
        "modes": {
            "calendar_aligned": "Shared UTC support permits co-occurrence analysis only.",
            "scale_shape_aligned": "Normalized structural recurrence; no simultaneity or precedence.",
        },
        "duration_presets": ["week", "three_months", "six_months"],
        "domains": [{"domain": entry.value.domain, "adapter_id": entry.value.adapter_id,
                     "label": entry.value.declaration.description}
                    for entry in EXPERIMENT_ADAPTERS.entries()],
        "workflow": ["compose", "save immutable revision", "metadata preflight",
                     "inspect canonical known-answer contract"],
        "available_now": ["manifest", "saved draft", "metadata-only preflight",
                          "StructuralTrajectory contract and known-answer preview",
                          "registered domain adapters, their controls and conformance"],
        "not_yet_available": ["live adapter translation", "acquire quartet", "run experiment"],
        "claim_boundary": "A ready preflight is not acquisition, analysis, evidence or a result.",
    }


@router.get("/recipes")
async def recipes() -> Dict[str, Any]:
    spec = flagship_recipe()
    envelope = manifest_envelope(spec)
    return {"recipes": [{"recipe_id": "g17-flagship-calendar", "title": spec.title,
                          "mode": spec.mode, "domains": [item.domain for item in spec.observations],
                          "manifest_sha256": envelope["manifest_sha256"]}],
            "note": "Recipes are complete manifests, not code generators or hidden defaults."}


@router.get("/recipes/g17-flagship-calendar")
async def flagship() -> Dict[str, Any]:
    return {"recipe_id": "g17-flagship-calendar", **manifest_envelope(flagship_recipe())}


@router.get("/adapters")
async def adapters() -> Dict[str, Any]:
    """Every registered domain adapter and its typed controls, generated from the registry.

    The Composer renders this rather than a hardcoded form per domain. A fifth adapter reaches
    the selector by registering, and this route does not learn its name.
    """
    return {"schema": "experiment-composer-adapters/v1",
            "adapters": [entry.value.describe() for entry in EXPERIMENT_ADAPTERS.entries()],
            "note": ("Each row carries the whole adapter description, including its domain "
                     "declaration and refusals. There is deliberately no per-adapter route: an "
                     "endpoint nothing reaches is an untested contract."),
            "claim_boundary": ("A registered adapter is a declared translation contract. It is "
                               "not evidence that its archive was read or that its declaration "
                               "describes that archive correctly.")}


@router.post("/adapters/{adapter_id}/conformance")
async def adapter_conformance(adapter_id: str,
                              parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the conformance kit against this adapter's deterministic known-answer record.

    Visible rather than a build-time check: an adapter author, and a reader of a result, can both
    see which declared invariances were executed and which are merely declared.
    """
    try:
        item = adapter_for(adapter_id)
    except SpectralEarthError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        record = known_answer_native(item.domain)
    except KeyError as error:
        raise HTTPException(status_code=422, detail=(
            "no deterministic known-answer record is registered for domain %r, so this adapter "
            "cannot be conformance-tested without acquiring data" % item.domain)) from error
    from datetime import datetime, timezone

    class _Window:
        name = "conformance_week"
        start_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_utc = datetime(2026, 1, 8, tzinfo=timezone.utc)

    report = run_conformance(item, ConformanceCase(
        parameters=dict(parameters or {}), windows=[_Window()],
        maximum_planned_bytes=4 * 1024 ** 3, native_record=record))
    return {**report.describe(), "record_kind": "deterministic_known_answer_not_acquired_data"}


@router.post("/manifests/validate")
async def validate_manifest(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    return manifest_envelope(spec)


@router.post("/manifests/preflight")
async def preflight(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    return preflight_manifest(spec)


@router.post("/manifests/representation-preview")
async def representation_preview(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """Inspect TG17.2 on frozen fixtures; never imply that manifest sources were acquired."""
    required = {"reanalysis", "argo_float", "tess_lightcurve"}
    observations = {item.domain: item for item in spec.observations}
    if not required.issubset(observations):
        raise HTTPException(status_code=422, detail=(
            "known-answer preview requires declared reanalysis, Argo and TESS observations"))
    preview = known_answer_preview(spec.seeds.get("family", 20260830))
    preview["manifest_sha256"] = manifest_sha256(spec)
    preview["selected_observations"] = sorted(required)
    return preview


@router.put("/drafts/{draft_id}")
async def save_draft(draft_id: str, spec: CrossDomainExperimentSpec,
                     request: Request) -> Dict[str, Any]:
    try:
        return _store(request).save(draft_id, spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/drafts/{draft_id}")
async def load_draft(draft_id: str, request: Request) -> Dict[str, Any]:
    try:
        spec = _store(request).load_draft(draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail="saved experiment draft not found")
    return {"draft_id": draft_id, **manifest_envelope(spec)}


@router.get("/manifests/{manifest_sha256}")
async def load_manifest(manifest_sha256: str, request: Request) -> Dict[str, Any]:
    try:
        spec = _store(request).load_manifest(manifest_sha256)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="immutable experiment manifest not found")
    return manifest_envelope(spec)


__all__ = ["router"]
