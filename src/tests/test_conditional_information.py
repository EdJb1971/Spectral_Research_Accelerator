"""TG16.2 frozen-family conditional-information audit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis_engine.conditional_information import (
    audit_conditional_information, conditional_support)
from src.api.main import app
from src.core.errors import InvalidParameterError
from src.data_layer.dataset_ingress import (
    SampleTableDeclaration, plan_conditional_information_audit,
    run_conditional_information_audit, sample_table_capability_profile)


def _arrays(kind: str, *, seed: int = 73, n: int = 360):
    rng = np.random.default_rng(seed)
    nuisance = rng.standard_normal(n)
    if kind == "survives":
        candidate = 0.4 * nuisance + rng.standard_normal(n)
        target = 1.2 * candidate + 1.5 * nuisance + 0.3 * rng.standard_normal(n)
    elif kind == "nuisance_only":
        candidate = nuisance + 0.25 * rng.standard_normal(n)
        target = nuisance + 0.25 * rng.standard_normal(n)
    elif kind == "conditional_null":
        candidate = rng.standard_normal(n)
        target = nuisance + 0.25 * rng.standard_normal(n)
    elif kind == "collider":
        candidate = rng.standard_normal(n)
        target = rng.standard_normal(n)
        nuisance = candidate + target + 0.2 * rng.standard_normal(n)
    else:
        raise ValueError(kind)
    return candidate, target, nuisance


def _payload(kind: str, *, seed: int = 73, n: int = 360) -> bytes:
    candidate, target, nuisance = _arrays(kind, seed=seed, n=n)
    lines = ["sample,target,nuisance,candidate"]
    lines.extend("%d,%.12g,%.12g,%.12g" % row for row in
                 zip(range(n), target, nuisance, candidate))
    return ("\n".join(lines) + "\n").encode()


def _declaration(relationship: str = "independent", *, nuisance: bool = True):
    sample_role = "sample_id" if relationship == "independent" else (
        "group" if relationship == "grouped" else "ordering")
    roles = {"sample": sample_role, "target": "target", "candidate": "feature",
             "nuisance": "nuisance" if nuisance else "ignore"}
    return SampleTableDeclaration(
        roles=roles, sample_relationship=relationship,
        units={"target": "dimensionless", "nuisance": "dimensionless",
               "candidate": "dimensionless"})


def test_estimator_calibrates_nuisance_nulls_and_detects_survival_and_collider():
    expected = {
        "survives": "supported_conditional_association",
        "nuisance_only": "unresolved", "conditional_null": "unresolved",
        "collider": "supported_conditional_association",
    }
    for index, (kind, outcome) in enumerate(expected.items()):
        candidate, target, nuisance = _arrays(kind)
        result = audit_conditional_information(
            {"candidate": candidate}, target, nuisance, bins=3,
            permutations=39, seed=900 + index, alpha=0.05)
        assert result["candidates"][0]["outcome"] == outcome
    assert "Collider" in result["claim_boundary"]
    assert "not proof of confounding" in result["claim_boundary"]


def test_support_rule_requires_overlap_levels_and_effective_joint_support():
    candidate, target, nuisance = _arrays("survives")
    admitted = conditional_support(
        {"candidate": candidate}, target, nuisance, bins=3)
    assert admitted["admitted"] is True
    refused = conditional_support(
        {"candidate": np.ones(candidate.size)}, target, nuisance, bins=3)
    assert refused["admitted"] is False
    assert refused["candidates"][0]["admitted"] is False
    nonlinear = conditional_support(
        {"candidate": candidate}, nuisance * nuisance, nuisance, bins=3)
    assert nonlinear["admitted"] is False
    assert nonlinear["conditional_model"]["admitted"] is False


def test_plan_seals_conditional_model_support_and_complete_family_with_refusals():
    payload = _payload("survives")
    plan = plan_conditional_information_audit(
        payload, filename="conditional.csv", delimiter=",", declaration=_declaration(),
        bins=3, permutations=39, seed=42)
    assert plan["quantity"] == "I(candidate; target | declared nuisance)"
    assert plan["candidates"] == ["candidate"] and plan["n_tests"] == 1
    assert plan["support_admission"]["admitted"] is True
    assert "reconstruct and rediscretise" in plan["null"]
    with pytest.raises(InvalidParameterError, match="smallest attainable p-value"):
        plan_conditional_information_audit(
            payload, filename="conditional.csv", delimiter=",", declaration=_declaration(),
            bins=3, permutations=18)
    with pytest.raises(InvalidParameterError, match="exactly one researcher-declared"):
        plan_conditional_information_audit(
            payload, filename="conditional.csv", delimiter=",",
            declaration=_declaration(nuisance=False), bins=3, permutations=39)
    with pytest.raises(InvalidParameterError, match="group-held-out confirmation"):
        plan_conditional_information_audit(
            payload, filename="conditional.csv", delimiter=",",
            declaration=_declaration("grouped"), bins=3, permutations=39)


def test_plan_refuses_inadequate_overlap_before_any_result():
    payload = _payload("survives")
    lines = payload.decode().splitlines()
    constant = [lines[0]] + [",".join(row.split(",")[:3] + ["1"]) for row in lines[1:]]
    with pytest.raises(InvalidParameterError, match="overlap/effective support"):
        plan_conditional_information_audit(
            ("\n".join(constant) + "\n").encode(), filename="constant.csv",
            delimiter=",", declaration=_declaration(), bins=3, permutations=39)


def test_run_is_content_bound_and_reports_only_conditional_association():
    payload = _payload("survives")
    plan = plan_conditional_information_audit(
        payload, filename="conditional.csv", delimiter=",", declaration=_declaration(),
        bins=3, permutations=39)
    report = run_conditional_information_audit(
        payload, filename="conditional.csv", delimiter=",", plan=plan)
    assert report["conditional_associations"][0]["outcome"] == \
        "supported_conditional_association"
    assert report["stored"] is False and report["rung_moved"] is False
    assert "does not mean 'confounding removed'" in report["claim_boundary"]
    assert "not causal" in report["claim_boundary"]
    with pytest.raises(InvalidParameterError, match="exact file bytes"):
        run_conditional_information_audit(
            payload + b"\n", filename="conditional.csv", delimiter=",", plan=plan)
    with pytest.raises(InvalidParameterError, match="complete frozen"):
        run_conditional_information_audit(
            payload, filename="conditional.csv", delimiter=",",
            plan={**plan, "bins": 4})


def test_capability_and_multipart_workflow_expose_only_declared_nuisance_recipe():
    payload = _payload("survives")
    profile = sample_table_capability_profile(
        payload, filename="conditional.csv", delimiter=",", declaration=_declaration())
    assert profile["operations"]["conditional_information_audit"]["available"] is True
    no_nuisance = sample_table_capability_profile(
        payload, filename="conditional.csv", delimiter=",",
        declaration=_declaration(nuisance=False))
    decision = no_nuisance["operations"]["conditional_information_audit"]
    assert decision["available"] is False
    assert decision["reason_code"] == "requires_declared_nuisance"

    client = TestClient(app)
    planned = client.post(
        "/api/v1/ingress/conditional/plan",
        files={"file": ("conditional.csv", payload, "text/csv")},
        data={"delimiter": ",", "declaration": json.dumps(_declaration().canonical()),
              "bins": "3", "permutations": "39", "seed": "92"})
    assert planned.status_code == 200, planned.text
    audited = client.post(
        "/api/v1/ingress/conditional/audit",
        files={"file": ("conditional.csv", payload, "text/csv")},
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    assert audited.status_code == 200, audited.text
    assert audited.json()["conditional_associations"][0]["outcome"] == \
        "supported_conditional_association"
