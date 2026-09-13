import json

import pytest
from pydantic import ValidationError

from src.benchmarks.pool_calibration import (
    ALPHA,
    CALIBRATION_CONTRACT,
    CALIBRATION_FAMILY_SIZE,
    CORRECTION,
    ORIENTATION,
)
from src.core.correspondence_estimand import minimum_pool_size
from src.core.scale_shape_successor import (
    DECLARATION_FILE,
    ScaleShapeSuccessorDeclaration,
    successor_scale_shape_manifest,
)


def test_frozen_successor_matches_calibration_without_claiming_a_real_pool():
    declaration = successor_scale_shape_manifest()

    assert declaration.family_size == CALIBRATION_FAMILY_SIZE == 6
    assert declaration.minimum_pool_size_per_correspondence == minimum_pool_size(6) == 48
    assert declaration.admission_contract == CALIBRATION_CONTRACT.describe()
    assert declaration.orientation == ORIENTATION
    assert declaration.alpha == ALPHA and declaration.correction == CORRECTION
    assert declaration.real_pool_inventory.status == "UNRESOLVED"
    assert declaration.real_pool_inventory.record_ids == []
    assert declaration.real_pool_inventory.exchangeability == "NOT_ESTABLISHED"
    assert declaration.describe()["declaration_sha256"] == declaration.digest


@pytest.mark.parametrize("field,value", [
    ("family_size", 5),
    ("minimum_pool_size_per_correspondence", 47),
    ("alpha", 0.1),
])
def test_calibrated_successor_fields_cannot_drift(field, value):
    body = json.loads(DECLARATION_FILE.read_text(encoding="utf-8"))
    body[field] = value
    with pytest.raises(ValidationError):
        ScaleShapeSuccessorDeclaration.parse_obj(body)


def test_unresolved_inventory_cannot_carry_records_or_claim_exchangeability():
    body = json.loads(DECLARATION_FILE.read_text(encoding="utf-8"))
    body["real_pool_inventory"] = {
        "status": "UNRESOLVED",
        "record_ids": ["plausible-looking-placeholder"],
        "exchangeability": "ESTABLISHED",
    }
    with pytest.raises(ValidationError, match="unresolved inventory"):
        ScaleShapeSuccessorDeclaration.parse_obj(body)


def test_admission_contract_cannot_drift_from_the_calibrated_marginals():
    body = json.loads(DECLARATION_FILE.read_text(encoding="utf-8"))
    body["admission_contract"]["minimum_coverage"] = 0.4
    with pytest.raises(ValidationError, match="admission contract"):
        ScaleShapeSuccessorDeclaration.parse_obj(body)


def test_curated_inventory_must_reach_the_derived_pool_minimum():
    body = json.loads(DECLARATION_FILE.read_text(encoding="utf-8"))
    body["real_pool_inventory"] = {
        "status": "CURATED",
        "record_ids": ["record-%02d" % index for index in range(47)],
        "exchangeability": "ESTABLISHED",
    }
    with pytest.raises(ValidationError, match="minimum pool size"):
        ScaleShapeSuccessorDeclaration.parse_obj(body)

    body["real_pool_inventory"]["record_ids"].append("record-47")
    assert ScaleShapeSuccessorDeclaration.parse_obj(body).real_pool_inventory.status == "CURATED"