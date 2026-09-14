"""Frozen G17 successor declaration for exact scale/shape pool substitution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, root_validator

from src.benchmarks.pool_calibration import (
    ALPHA,
    CALIBRATION_CONTRACT,
    CALIBRATION_FAMILY_SIZE,
    CORRECTION,
    ORIENTATION,
)
from src.core.correspondence_estimand import minimum_pool_size


SCHEMA = "g17-scale-shape-successor/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_FILE = REPO_ROOT / "data" / "studies" / "g17_scale_shape_successor.json"


class _FrozenModel(BaseModel):
    class Config:
        allow_mutation = False
        allow_population_by_field_name = True
        extra = "forbid"


class RealPoolInventory(_FrozenModel):
    status: Literal["UNRESOLVED", "CURATED"]
    record_ids: List[str]
    exchangeability: Literal["NOT_ESTABLISHED", "ESTABLISHED"]

    @root_validator
    def _curation_cannot_be_implied(cls, values):
        status = values.get("status")
        records = values.get("record_ids") or []
        exchangeability = values.get("exchangeability")
        if status == "UNRESOLVED" and (records or exchangeability != "NOT_ESTABLISHED"):
            raise ValueError("an unresolved inventory has no adopted records or exchangeability")
        if status == "CURATED" and (not records or exchangeability != "ESTABLISHED"):
            raise ValueError("a curated inventory must name records and establish exchangeability")
        if len(records) != len(set(records)):
            raise ValueError("real pool record identities must be unique")
        return values


class ScaleShapeSuccessorDeclaration(_FrozenModel):
    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")
    study_id: str = Field(..., min_length=1)
    mode: Literal["scale_shape_aligned"]
    estimand: Literal["per_correspondence"]
    inference: Literal["exact_pool_substitution"]
    statistic: Literal["src.benchmarks.shape_calibration:shape_recurrence"]
    orientation: Literal["larger_is_more_similar"]
    correction: Literal["benjamini_yekutieli"]
    alpha: float = Field(..., gt=0.0, lt=1.0)
    family_size: int = Field(..., gt=0)
    minimum_pool_size_per_correspondence: int = Field(..., gt=0)
    admission_contract: Dict[str, Any]
    real_pool_inventory: RealPoolInventory
    claim_boundary: str = Field(..., min_length=1)

    @root_validator
    def _matches_calibrated_contract(cls, values):
        family_size = values.get("family_size")
        if family_size != CALIBRATION_FAMILY_SIZE:
            raise ValueError("family_size must match the calibrated successor family")
        if values.get("minimum_pool_size_per_correspondence") != minimum_pool_size(family_size):
            raise ValueError("minimum pool size must be derived for the declared family")
        if values.get("admission_contract") != CALIBRATION_CONTRACT.describe():
            raise ValueError("admission contract must match the calibrated successor contract")
        if values.get("orientation") != ORIENTATION:
            raise ValueError("orientation must match the calibrated successor statistic")
        if values.get("alpha") != ALPHA or values.get("correction") != CORRECTION:
            raise ValueError("alpha and correction must match the calibrated successor family")
        inventory = values.get("real_pool_inventory")
        minimum = values.get("minimum_pool_size_per_correspondence")
        if inventory and inventory.status == "CURATED" and len(inventory.record_ids) < minimum:
            raise ValueError("a curated inventory must meet the minimum pool size")
        return values

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.json(
            by_alias=True, sort_keys=True, separators=(",", ":"), exclude_none=False
        ).encode("utf-8")
        ).hexdigest()

    def describe(self) -> Dict[str, Any]:
        return {**json.loads(self.json(by_alias=True)), "declaration_sha256": self.digest}


def successor_scale_shape_manifest(path: Path = DECLARATION_FILE) \
        -> ScaleShapeSuccessorDeclaration:
    """Read and validate the frozen successor without resolving its real inventory."""
    return ScaleShapeSuccessorDeclaration.parse_obj(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "DECLARATION_FILE",
    "SCHEMA",
    "RealPoolInventory",
    "ScaleShapeSuccessorDeclaration",
    "successor_scale_shape_manifest",
]