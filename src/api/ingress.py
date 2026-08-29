"""File-first generic ingress and fixed-family representation audit (G14.1/G14.2)."""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.errors import SpectralEarthError, classify
from src.data_layer.dataset_ingress import (SampleTableDeclaration,
                                            plan_conditional_information_audit,
                                            plan_redundancy_structure_audit,
                                            plan_representation_audit,
                                            plan_stable_subspace_generation,
                                            probe_delimited,
                                            run_conditional_information_audit,
                                            run_redundancy_structure_audit,
                                            run_representation_audit,
                                            run_stable_subspace_generation,
                                            sample_table_capability_profile)

router = APIRouter(prefix="/api/v1/ingress", tags=["ingress"])


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


__all__ = ["router"]
