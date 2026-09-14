"""File-first generic ingress and fixed-family representation audit (G14.1/G14.2)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.errors import SpectralEarthError, classify
from src.data_layer.dataset_ingress import (SampleTableDeclaration,
                                            freeze_external_subspace_transfer,
                                            freeze_stable_subspace_confirmation,
                                            plan_conditional_information_audit,
                                            plan_sample_table_handoff,
                                            plan_redundancy_structure_audit,
                                            plan_representation_audit,
                                            plan_stable_subspace_generation,
                                            probe_delimited,
                                            publish_stable_subspace_candidate,
                                            run_conditional_information_audit,
                                            run_external_subspace_certification,
                                            run_redundancy_structure_audit,
                                            run_representation_audit,
                                            run_stable_subspace_confirmation,
                                            run_stable_subspace_generation,
                                            sample_table_capability_profile)
from src.api.preregistration import held_out_ledger, load_seal, store_seal
from src.core.preregistration import Seal

router = APIRouter(prefix="/api/v1/ingress", tags=["ingress"])


def _candidate_root() -> Path:
    return Path(os.environ.get("SPECTRAL_PREREGISTRATION_ROOT",
                               str(Path("data") / "preregistrations"))) / "subspaces"


def _store_candidate(candidate: Dict[str, Any]) -> None:
    root = _candidate_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (candidate["candidate_sha256"] + ".json")
    if not path.exists():
        path.write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")


def _load_candidate(candidate_sha256: str) -> Dict[str, Any]:
    digest = str(candidate_sha256).lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise HTTPException(status_code=404, detail="Not a published candidate digest.")
    path = _candidate_root() / (digest + ".json")
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="No published subspace candidate %s is stored here." % digest)
    return json.loads(path.read_text(encoding="utf-8"))


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


def _object(raw: str, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail="`%s` must be a JSON object." % name) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400,
                            detail="`%s` must be a JSON object." % name)
    return value


@router.post("/probe")
async def probe(file: UploadFile = File(...), delimiter: str = Form(",")) -> Dict[str, Any]:
    payload = await file.read()
    try:
        return probe_delimited(payload, filename=file.filename or "upload", delimiter=delimiter)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/plan")
async def plan(file: UploadFile = File(...), delimiter: str = Form(","),
               declaration: str = Form(...), representations: str = Form('["identity","pca"]'),
               pca_components: int = Form(3), bins: int = Form(4),
               permutations: int = Form(4999), generate_fraction: float = Form(0.7),
               seed: int = Form(1729), alpha: float = Form(0.05)) -> Dict[str, Any]:
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        representation_names = json.loads(representations)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail="`representations` must be a JSON array.") from exc
    if not isinstance(representation_names, list):
        raise HTTPException(status_code=400,
                            detail="`representations` must be a JSON array.")
    try:
        return plan_representation_audit(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared),
            representations=representation_names, pca_components=pca_components,
            bins=bins, permutations=permutations, generate_fraction=generate_fraction,
            seed=seed, alpha=alpha)
    except (TypeError, SpectralEarthError) as error:
        if isinstance(error, SpectralEarthError):
            raise _handle(error)
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")


@router.post("/capabilities")
async def capabilities(file: UploadFile = File(...), delimiter: str = Form(","),
                       declaration: str = Form(...)) -> Dict[str, Any]:
    """Derive paths from explicit semantics and exact bytes; infer no column meaning."""
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        return sample_table_capability_profile(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared))
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/planning-handoff")
async def planning_handoff(file: UploadFile = File(...), delimiter: str = Form(","),
                           declaration: str = Form(...),
                           operation: str = Form(...)) -> Dict[str, Any]:
    """Bind an exact declared object to one registered planner without executing it."""
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        return plan_sample_table_handoff(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared), operation=operation)
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/audit")
async def audit(file: UploadFile = File(...), delimiter: str = Form(","),
                plan: str = Form(...)) -> Dict[str, Any]:
    payload = await file.read()
    frozen = _object(plan, "plan")
    try:
        return run_representation_audit(payload, filename=file.filename or "upload",
                                        delimiter=delimiter, plan=frozen)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/structure/plan")
async def structure_plan(file: UploadFile = File(...), delimiter: str = Form(","),
                         declaration: str = Form(...), bins: int = Form(4),
                         permutations: int = Form(4999), seed: int = Form(16101),
                         alpha: float = Form(0.05)) -> Dict[str, Any]:
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        return plan_redundancy_structure_audit(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared), bins=bins,
            permutations=permutations, seed=seed, alpha=alpha)
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/structure/audit")
async def structure_audit(file: UploadFile = File(...), delimiter: str = Form(","),
                          plan: str = Form(...)) -> Dict[str, Any]:
    payload = await file.read()
    frozen = _object(plan, "plan")
    try:
        return run_redundancy_structure_audit(
            payload, filename=file.filename or "upload", delimiter=delimiter, plan=frozen)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/conditional/plan")
async def conditional_plan(file: UploadFile = File(...), delimiter: str = Form(","),
                           declaration: str = Form(...), bins: int = Form(3),
                           permutations: int = Form(4999), seed: int = Form(16201),
                           alpha: float = Form(0.05)) -> Dict[str, Any]:
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        return plan_conditional_information_audit(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared), bins=bins,
            permutations=permutations, seed=seed, alpha=alpha)
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/conditional/audit")
async def conditional_audit(file: UploadFile = File(...), delimiter: str = Form(","),
                            plan: str = Form(...)) -> Dict[str, Any]:
    payload = await file.read()
    frozen = _object(plan, "plan")
    try:
        return run_conditional_information_audit(
            payload, filename=file.filename or "upload", delimiter=delimiter, plan=frozen)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/plan")
async def subspace_plan(file: UploadFile = File(...), delimiter: str = Form(","),
                        declaration: str = Form(...), dimensions: str = Form("[1]"),
                        regularizations: str = Form("[0.01,0.1,1.0]"),
                        permutations: int = Form(4999), generate_fraction: float = Form(0.7),
                        restarts: int = Form(6), iterations: int = Form(48),
                        perturbations: int = Form(6), perturbation_scale: float = Form(0.10),
                        stability_threshold: float = Form(0.10),
                        nuisance_penalty: float = Form(1.0),
                        variance_weight: float = Form(0.05), seed: int = Form(16301),
                        alpha: float = Form(0.05)) -> Dict[str, Any]:
    payload = await file.read()
    declared = _object(declaration, "declaration")
    try:
        dimension_values = json.loads(dimensions)
        regularization_values = json.loads(regularizations)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail="`dimensions` and `regularizations` must be JSON arrays.") from exc
    if not isinstance(dimension_values, list) or not isinstance(regularization_values, list):
        raise HTTPException(status_code=400,
                            detail="`dimensions` and `regularizations` must be JSON arrays.")
    try:
        return plan_stable_subspace_generation(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            declaration=SampleTableDeclaration(**declared), dimensions=dimension_values,
            regularizations=regularization_values, permutations=permutations,
            generate_fraction=generate_fraction, restarts=restarts, iterations=iterations,
            perturbations=perturbations, perturbation_scale=perturbation_scale,
            stability_threshold=stability_threshold, nuisance_penalty=nuisance_penalty,
            variance_weight=variance_weight, seed=seed, alpha=alpha)
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/generate")
async def subspace_generate(file: UploadFile = File(...), delimiter: str = Form(","),
                            plan: str = Form(...)) -> Dict[str, Any]:
    payload = await file.read()
    frozen = _object(plan, "plan")
    try:
        return run_stable_subspace_generation(
            payload, filename=file.filename or "upload", delimiter=delimiter, plan=frozen)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/freeze")
async def subspace_freeze(file: UploadFile = File(...), delimiter: str = Form(","),
                          plan: str = Form(...), generation: str = Form(...),
                          confirmation_permutations: int = Form(4999),
                          confirmation_seed: int = Form(16401)) -> Dict[str, Any]:
    """Freeze all confirmation settings before opening the reserved outcomes."""
    payload = await file.read()
    frozen_plan = _object(plan, "plan")
    generated = _object(generation, "generation")
    try:
        result = freeze_stable_subspace_confirmation(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            plan=frozen_plan, generation=generated,
            sealed_at=datetime.now(timezone.utc).isoformat(),
            confirmation_permutations=confirmation_permutations,
            confirmation_seed=confirmation_seed, ledger=held_out_ledger())
        store_seal(Seal.from_mapping(result["seal"]))
        return result
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/confirm")
async def subspace_confirm(file: UploadFile = File(...), delimiter: str = Form(","),
                           seal_sha256: str = Form(...),
                           published_sha256: str | None = Form(None)) -> Dict[str, Any]:
    """Apply the frozen family unchanged and spend its held-out partition once."""
    payload = await file.read()
    stored = load_seal(seal_sha256)
    frozen = {
        "schema": "spectral.stable-subspace-confirmation-seal.v1",
        "seal": stored.to_mapping(), "seal_sha256": stored.seal_sha256,
    }
    try:
        return run_stable_subspace_confirmation(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            seal=frozen, ledger=held_out_ledger(),
            opened_at=datetime.now(timezone.utc).isoformat(),
            published_sha256=published_sha256)
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/publish")
async def subspace_publish(seal_sha256: str = Form(...),
                           label: str = Form(...)) -> Dict[str, Any]:
    """Publish one generated candidate definition; this does not claim replication."""
    try:
        candidate = publish_stable_subspace_candidate(
            confirmation_seal=load_seal(seal_sha256), label=label,
            published_at=datetime.now(timezone.utc).isoformat())
        _store_candidate(candidate)
        return candidate
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/transfer/freeze")
async def subspace_transfer_freeze(
        candidate_sha256: str = Form(...), target_content_sha256: str = Form(...),
        target_n_rows: int = Form(...), target_declaration: str = Form(...),
        target_provenance: str = Form(...), permutations: int = Form(4999),
        seed: int = Form(16501), alpha: float = Form(0.05)) -> Dict[str, Any]:
    """Bind published definitions to target metadata before target values are supplied."""
    try:
        digests = json.loads(candidate_sha256)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail="`candidate_sha256` must be a JSON array.") from exc
    if not isinstance(digests, list):
        raise HTTPException(status_code=400,
                            detail="`candidate_sha256` must be a JSON array.")
    declared = _object(target_declaration, "target_declaration")
    provenance = _object(target_provenance, "target_provenance")
    try:
        result = freeze_external_subspace_transfer(
            candidates=[_load_candidate(str(value)) for value in digests],
            target_content_sha256=target_content_sha256, target_n_rows=target_n_rows,
            target_declaration=SampleTableDeclaration(**declared),
            target_provenance=provenance,
            sealed_at=datetime.now(timezone.utc).isoformat(), permutations=permutations,
            seed=seed, alpha=alpha, ledger=held_out_ledger())
        store_seal(Seal.from_mapping(result["seal"]))
        return result
    except TypeError:
        raise HTTPException(status_code=400,
                            detail="Target declaration needs roles, sample_relationship and units.")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/subspace/transfer/certify")
async def subspace_transfer_certify(
        file: UploadFile = File(...), delimiter: str = Form(","),
        transfer_seal_sha256: str = Form(...),
        published_sha256: str = Form(...)) -> Dict[str, Any]:
    """Spend and open one target, then execute its no-adaptation transfer contract once."""
    try:
        stored = load_seal(transfer_seal_sha256)
        frozen = {"schema": "spectral.external-subspace-transfer-seal.v1",
                  "seal": stored.to_mapping(), "seal_sha256": stored.seal_sha256}
        payload = await file.read()
        return run_external_subspace_certification(
            payload, filename=file.filename or "upload", delimiter=delimiter,
            seal=frozen, ledger=held_out_ledger(),
            opened_at=datetime.now(timezone.utc).isoformat(),
            published_sha256=published_sha256)
    except SpectralEarthError as error:
        raise _handle(error)


__all__ = ["router"]
