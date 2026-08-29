"""TG16.3 bounded generate-only stable-subspace search."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis_engine.stable_subspace import (
    generate_stable_subspaces, projector, projector_distance)
from src.api.main import app
from src.core.errors import InvalidParameterError
from src.data_layer.dataset_ingress import (
    SampleTableDeclaration, plan_stable_subspace_generation,
    run_stable_subspace_generation, sample_table_capability_profile)


def _arrays(kind: str, *, seed: int = 83, n: int = 180):
    rng = np.random.default_rng(seed)
    first, second = rng.standard_normal((2, n))
    nuisance = rng.standard_normal(n)
    if kind == "signal":
        target = first + second + 0.2 * rng.standard_normal(n)
    elif kind == "null":
        target = rng.standard_normal(n)
    elif kind == "nuisance_stable":
        first = first + 0.3 * nuisance
        target = first + second + 0.5 * nuisance + 0.2 * rng.standard_normal(n)
    else:
        raise ValueError(kind)
    return first, second, target, nuisance


def _payload(kind: str = "signal", *, nuisance: bool = False) -> bytes:
    first, second, target, nuisance_values = _arrays(kind)
    header = "sample,target,first,second" + (",nuisance" if nuisance else "")
    lines = [header]
    values = zip(range(target.size), target, first, second, nuisance_values)
    for sample, target_value, first_value, second_value, nuisance_value in values:
        row = "%d,%.12g,%.12g,%.12g" % (
            sample, target_value, first_value, second_value)
        lines.append(row + (",%.12g" % nuisance_value if nuisance else ""))
    return ("\n".join(lines) + "\n").encode()


def _declaration(relationship: str = "independent", *, nuisance: bool = False):
    relationship_role = ("sample_id" if relationship == "independent" else
                         ("group" if relationship == "grouped" else "ordering"))
    roles = {"sample": relationship_role, "target": "target", "first": "feature",
             "second": "feature"}
    units = {"target": "dimensionless", "first": "dimensionless",
             "second": "dimensionless"}
    if nuisance:
        roles["nuisance"] = "nuisance"
        units["nuisance"] = "dimensionless"
    return SampleTableDeclaration(roles=roles, sample_relationship=relationship, units=units)


def test_projector_identity_ignores_basis_sign_and_rotation():
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    rotated = basis @ np.asarray([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(projector(basis), projector(-rotated))
    assert projector_distance(basis, -rotated) < 1e-12


def test_generator_detects_stable_linear_signal_and_leaves_null_unresolved():
    for kind, expected, data_seed in (("signal", True, 83), ("null", False, 84)):
        first, second, target, _nuisance = _arrays(kind, seed=data_seed)
        result = generate_stable_subspaces(
            {"first": first, "second": second}, target, dimensions=(1,),
            regularizations=(0.1,), permutations=39, restarts=3, iterations=24,
            perturbations=3, seed=101)
        assert bool(result["compact_candidates"]) is expected
    assert "projectors" in result["claim_boundary"]
    assert "not optimal representations" in result["claim_boundary"]


def test_nuisance_region_stability_is_generate_only_and_not_nuisance_removal():
    first, second, target, nuisance = _arrays("nuisance_stable")
    result = generate_stable_subspaces(
        {"first": first, "second": second}, target, nuisance,
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=102)
    diagnostic = result["subspaces"][0]["nuisance_stability"]
    assert diagnostic["admitted"] is True and len(diagnostic["region_sizes"]) == 3
    assert "not conditional information" in diagnostic["claim_boundary"]
    assert "not evidence that nuisance was removed" in diagnostic["claim_boundary"]
    with pytest.raises(ValueError, match="every nuisance region"):
        generate_stable_subspaces(
            {"first": first, "second": second}, target, nuisance > 0,
            dimensions=(1,), regularizations=(0.1,), permutations=39,
            restarts=3, iterations=24, perturbations=3)
    with pytest.raises(ValueError, match="target must vary"):
        generate_stable_subspaces(
            {"first": first, "second": second}, np.ones(target.size),
            dimensions=(1,), regularizations=(0.1,), permutations=39,
            restarts=3, iterations=24, perturbations=3)


def test_plan_seals_complete_family_optimizer_partitions_and_refuses_bad_recipes():
    payload = _payload()
    plan = plan_stable_subspace_generation(
        payload, filename="linear.csv", delimiter=",", declaration=_declaration(),
        dimensions=(1,), regularizations=(0.1, 0.01), permutations=109,
        restarts=3, iterations=24, perturbations=3, seed=44)
    assert plan["family_members"] == ["dimension=1;ridge=0.01",
                                       "dimension=1;ridge=0.1"]
    assert plan["n_tests"] == 2 and plan["optimizer"] == "seeded block power iteration"
    assert plan["partitions"]["confirmation_opened"] is False
    assert plan["claim_boundary"].endswith("does not open the reserved confirmation values.")
    with pytest.raises(InvalidParameterError, match="smallest attainable p-value"):
        plan_stable_subspace_generation(
            payload, filename="linear.csv", delimiter=",", declaration=_declaration(),
            dimensions=(1,), regularizations=(0.01, 0.1), permutations=38)
    with pytest.raises(InvalidParameterError, match="group-held-out confirmation"):
        plan_stable_subspace_generation(
            payload, filename="linear.csv", delimiter=",",
            declaration=_declaration("grouped"), dimensions=(1,),
            regularizations=(0.1,), permutations=39)


def test_run_is_content_bound_returns_projectors_and_keeps_confirmation_unopened():
    payload = _payload()
    plan = plan_stable_subspace_generation(
        payload, filename="linear.csv", delimiter=",", declaration=_declaration(),
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=45)
    report = run_stable_subspace_generation(
        payload, filename="linear.csv", delimiter=",", plan=plan)
    assert report["candidate_compact_stable_subspaces"]
    assert report["partitions"]["confirmation_opened"] is False
    assert report["stored"] is False and report["rung_moved"] is False
    assert len(report["subspaces"][0]["projector"]) == 2
    assert len(report["subspaces"][0]["multi_seed_perturbations"]) == 3
    assert "confirmation partition remains unopened" in report["claim_boundary"]
    with pytest.raises(InvalidParameterError, match="exact file bytes"):
        run_stable_subspace_generation(
            payload + b"\n", filename="linear.csv", delimiter=",", plan=plan)
    with pytest.raises(InvalidParameterError, match="complete frozen subspace plan"):
        run_stable_subspace_generation(
            payload, filename="linear.csv", delimiter=",",
            plan={**plan, "regularizations": [1.0]})


def test_capability_and_multipart_plan_generation_workflow():
    payload = _payload()
    profile = sample_table_capability_profile(
        payload, filename="linear.csv", delimiter=",", declaration=_declaration())
    assert profile["operations"]["stable_subspace_generation"]["available"] is True
    client = TestClient(app)
    planned = client.post(
        "/api/v1/ingress/subspace/plan",
        files={"file": ("linear.csv", payload, "text/csv")},
        data={"delimiter": ",", "declaration": json.dumps(_declaration().canonical()),
              "dimensions": "[1]", "regularizations": "[0.1]",
              "permutations": "39", "restarts": "3", "iterations": "24",
              "perturbations": "3", "seed": "46"})
    assert planned.status_code == 200, planned.text
    generated = client.post(
        "/api/v1/ingress/subspace/generate",
        files={"file": ("linear.csv", payload, "text/csv")},
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    assert generated.status_code == 200, generated.text
    assert generated.json()["candidate_compact_stable_subspaces"]
