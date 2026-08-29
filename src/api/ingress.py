"""File-first generic ingress and fixed-family representation audit (G14.1/G14.2)."""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.errors import SpectralEarthError, classify
from src.data_layer.dataset_ingress import (SampleTableDeclaration,
                                            plan_representation_audit,
                                            probe_delimited,
                                            run_representation_audit)

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


__all__ = ["router"]
