"""TG17.1 HTTP surface for composing and preflighting one scientific manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from src.benchmarks.structural_trajectory import known_answer_preview
from src.core.experiment_manifest import (CrossDomainExperimentSpec, ManifestStore,
                                          flagship_recipe, manifest_envelope,
                                          manifest_sha256, preflight_manifest)


router = APIRouter(prefix="/api/v1/experiment-composer", tags=["experiment-composer"])


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
        "workflow": ["compose", "save immutable revision", "metadata preflight",
                     "inspect canonical known-answer contract"],
        "available_now": ["manifest", "saved draft", "metadata-only preflight",
                          "StructuralTrajectory contract and known-answer preview"],
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
