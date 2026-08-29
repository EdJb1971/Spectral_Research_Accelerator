"""TG16.1 frozen-family redundancy structure audit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis_engine.representation_structure import (audit_pair_structure,
                                                           candidate_pairs)
from src.api.main import app
from src.core.errors import InvalidParameterError
from src.data_layer.dataset_ingress import (SampleTableDeclaration,
                                            plan_redundancy_structure_audit,
                                            run_redundancy_structure_audit,
                                            sample_table_capability_profile)


def _payload(kind: str, *, seed: int = 19, n: int = 360) -> bytes:
    rng = np.random.default_rng(seed)
    first = rng.standard_normal(n)
    if kind == "duplicate":
        second, target = first.copy(), first.copy()
    elif kind == "complementary":
        second = rng.standard_normal(n)
        target = first + second + 0.15 * rng.standard_normal(n)
    else:
        raise ValueError(kind)
    lines = ["sample,target,first,second"]
    lines.extend("%d,%.12g,%.12g,%.12g" % row for row in
                 zip(range(n), target, first, second))
    return ("\n".join(lines) + "\n").encode()


def _declaration(relationship: str = "independent") -> SampleTableDeclaration:
    role = "sample_id" if relationship == "independent" else (
        "group" if relationship == "grouped" else "ordering")
    return SampleTableDeclaration(
        roles={"sample": role, "target": "target", "first": "feature",
               "second": "feature"},
        sample_relationship=relationship,
        units={"target": "dimensionless", "first": "dimensionless",
               "second": "dimensionless"})


def test_complete_pair_family_is_sorted_and_never_selected_from_results():
    assert candidate_pairs(("c", "a", "b")) == (("a", "b"), ("a", "c"), ("b", "c"))


def test_joint_estimator_distinguishes_redundancy_complementarity_xor_and_null():
    rng = np.random.default_rng(71)
    n = 512
    latent = rng.standard_normal(n)
    cases = {
        "supported_redundancy": (
            {"first": latent + 0.2 * rng.standard_normal(n),
             "second": latent + 0.2 * rng.standard_normal(n)}, latent),
        "supported_complementarity": (
            {"first": (first := rng.standard_normal(n)),
             "second": (second := rng.standard_normal(n))},
            first + second + 0.2 * rng.standard_normal(n)),
        "unresolved": ({"first": rng.standard_normal(n), "second": rng.standard_normal(n)},
                       rng.standard_normal(n)),
    }
    for expected, (features, target) in cases.items():
        result = audit_pair_structure(features, target, bins=3, permutations=109,
                                      seed=801, alpha=0.05)
        assert result["pairs"][0]["outcome"] == expected
    first = rng.integers(0, 2, n).astype(float)
    second = rng.integers(0, 2, n).astype(float)
    xor = np.logical_xor(first.astype(bool), second.astype(bool)).astype(float)
    result = audit_pair_structure({"first": first, "second": second}, xor, bins=2,
                                  permutations=109, seed=802, alpha=0.05)
    assert result["pairs"][0]["outcome"] == "supported_complementarity"
    assert max(result["pairs"][0]["singleton_mi_nats"].values()) < 0.02


def test_plan_seals_every_method_choice_and_refuses_unresolvable_or_dependent_families():
    payload = _payload("duplicate")
    plan = plan_redundancy_structure_audit(
        payload, filename="pairs.csv", delimiter=",", declaration=_declaration(), bins=3,
        permutations=109, seed=33)
    assert plan["candidate_groups"] == [["first", "second"]]
    assert plan["n_tests"] == 3
    assert "target-within-other-candidate-bin" in plan["null"]
    assert plan["claim_boundary"].endswith("computes no structure result.")
    with pytest.raises(InvalidParameterError, match="smallest attainable p-value"):
        plan_redundancy_structure_audit(
            payload, filename="pairs.csv", delimiter=",", declaration=_declaration(), bins=3,
            permutations=99)
    with pytest.raises(InvalidParameterError, match="group-held-out confirmation"):
        plan_redundancy_structure_audit(
            payload, filename="pairs.csv", delimiter=",",
            declaration=_declaration("grouped"), bins=3, permutations=109)
    grouped = sample_table_capability_profile(
        payload, filename="pairs.csv", delimiter=",", declaration=_declaration("grouped"))
    decision = grouped["operations"]["redundancy_structure_audit"]
    assert decision["available"] is False
    assert "group-held-out benchmarked nulls" in decision["reason"]


def test_run_reports_candidate_structure_without_feature_removal_language():
    payload = _payload("duplicate")
    plan = plan_redundancy_structure_audit(
        payload, filename="pairs.csv", delimiter=",", declaration=_declaration(), bins=3,
        permutations=109)
    report = run_redundancy_structure_audit(
        payload, filename="pairs.csv", delimiter=",", plan=plan)
    pair = report["structure_map"][0]
    assert pair["outcome"] == "supported_redundancy"
    assert pair["exact_duplicate"] is True
    assert report["family"]["n_tests"] == 3
    assert report["stored"] is False and report["rung_moved"] is False
    assert "not a partial-information decomposition" in report["claim_boundary"]
    assert "no outcome instructs removal" in report["claim_boundary"]


def test_changed_bytes_and_tampered_method_are_refused_before_enumeration():
    payload = _payload("duplicate")
    plan = plan_redundancy_structure_audit(
        payload, filename="pairs.csv", delimiter=",", declaration=_declaration(), bins=3,
        permutations=109)
    with pytest.raises(InvalidParameterError, match="exact file bytes"):
        run_redundancy_structure_audit(
            payload + b"\n", filename="pairs.csv", delimiter=",", plan=plan)
    with pytest.raises(InvalidParameterError, match="complete frozen structure plan"):
        run_redundancy_structure_audit(
            payload, filename="pairs.csv", delimiter=",", plan={**plan, "bins": 4})


def test_capability_and_http_workflow_expose_the_earned_operation():
    payload = _payload("complementary")
    profile = sample_table_capability_profile(
        payload, filename="pairs.csv", delimiter=",", declaration=_declaration())
    assert profile["operations"]["redundancy_structure_audit"]["available"] is True
    client = TestClient(app)
    planned = client.post(
        "/api/v1/ingress/structure/plan",
        files={"file": ("pairs.csv", payload, "text/csv")},
        data={"delimiter": ",", "declaration": json.dumps(_declaration().canonical()),
              "bins": "3", "permutations": "109", "seed": "92"})
    assert planned.status_code == 200, planned.text
    audited = client.post(
        "/api/v1/ingress/structure/audit",
        files={"file": ("pairs.csv", payload, "text/csv")},
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    assert audited.status_code == 200, audited.text
    assert audited.json()["structure_map"][0]["outcome"] == "supported_complementarity"
