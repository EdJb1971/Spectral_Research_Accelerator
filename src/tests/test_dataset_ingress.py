"""G14.1/G14.2 generic ingress and fixed-family representation audit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.errors import InvalidParameterError
from src.data_layer.dataset_ingress import (SampleTableDeclaration,
                                            plan_representation_audit,
                                            probe_delimited,
                                            require_independent_samples,
                                            run_representation_audit,
                                            sample_table_capability_profile)


def dataset(seed: int = 17, n: int = 240) -> bytes:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    nuisance = rng.uniform(-1, 1, size=n)
    target = 3.0 * signal + 0.15 * rng.normal(size=n)
    noise = rng.normal(size=n)
    lines = ["sample,alpha,b,signal,noise"]
    lines.extend("%d,%.12g,%.12g,%.12g,%.12g" % row for row in
                 zip(range(n), target, nuisance, signal, noise))
    return ("\n".join(lines) + "\n").encode()


def declaration() -> SampleTableDeclaration:
    return SampleTableDeclaration(
        roles={"sample": "sample_id", "alpha": "target", "b": "nuisance",
               "signal": "feature", "noise": "feature"},
        sample_relationship="independent",
        units={"alpha": "dimensionless", "b": "dimensionless",
               "signal": "dimensionless", "noise": "dimensionless"})


def test_probe_reports_storage_facts_and_does_not_guess_meaning_from_names():
    report = probe_delimited(dataset(), filename="experiment.csv")
    assert report["semantic_inference"] is False
    assert report["n_rows"] == 240
    assert {row["storage_type"] for row in report["columns"]} == {"numeric"}
    assert all("role" not in row for row in report["columns"])


def test_declaration_requires_explicit_roles_relationship_and_units():
    probe = probe_delimited(dataset(), filename="experiment.csv")
    broken = SampleTableDeclaration(
        roles={**declaration().roles, "alpha": "feature"},
        sample_relationship="independent", units=declaration().units)
    with pytest.raises(InvalidParameterError, match="exactly one target"):
        broken.validate(probe)
    no_units = SampleTableDeclaration(roles=declaration().roles,
                                      sample_relationship="independent", units={})
    with pytest.raises(InvalidParameterError, match="explicit unit"):
        no_units.validate(probe)


def test_plan_freezes_complete_family_and_refuses_underpowered_permutation_resolution():
    payload = dataset()
    with pytest.raises(InvalidParameterError, match="smallest attainable p-value"):
        plan_representation_audit(
            payload, filename="experiment.csv", delimiter=",", declaration=declaration(),
            representations=("identity", "pca"), pca_components=1, permutations=99)
    plan = plan_representation_audit(
        payload, filename="experiment.csv", delimiter=",", declaration=declaration(),
        representations=("identity", "pca"), pca_components=1, permutations=199)
    assert plan["candidates"] == [
        "identity:noise", "identity:signal", "pca:component_1"]
    assert plan["n_tests_per_partition"] == 3
    assert plan["claim_boundary"].endswith("opens no held-out values.")


def test_capability_profile_routes_independent_tables_and_explains_spatial_refusals():
    profile = sample_table_capability_profile(
        dataset(), filename="experiment.csv", delimiter=",", declaration=declaration())
    assert profile["identity"] == probe_delimited(
        dataset(), filename="experiment.csv")["content_sha256"]
    assert profile["operations"]["representation_audit"]["available"] is True
    assert "association_redundancy" not in profile["operations"]
    boundary = profile["operations"]["boundary_lab"]
    assert boundary["status"] == "unavailable"
    assert boundary["reason_code"] == "requires_spatial_grid_2d"
    assert "no declared 2D grid" in boundary["reason"]
    assert len(profile["profile_sha256"]) == 64


def test_grouped_and_ordered_tables_name_the_safe_split_they_need_and_never_plan():
    payload = dataset()
    for relationship, role, phrase in (
        ("grouped", "group", "group-held-out confirmation"),
        ("ordered", "ordering", "blocked and embargoed confirmation"),
    ):
        declared = SampleTableDeclaration(
            roles={**declaration().roles, "sample": role},
            sample_relationship=relationship, units=declaration().units)
        profile = sample_table_capability_profile(
            payload, filename="experiment.csv", delimiter=",", declaration=declared)
        decision = profile["operations"]["representation_audit"]
        assert decision["available"] is False
        assert phrase in decision["reason"]
        with pytest.raises(InvalidParameterError, match=phrase):
            plan_representation_audit(
                payload, filename="experiment.csv", delimiter=",", declaration=declared,
                representations=("identity",), pca_components=1, permutations=99)


def test_g16_admission_is_independent_only_and_names_the_missing_resampling_contract():
    require_independent_samples(declaration(), recipe="G16 representation structure")
    for relationship, phrase in (
        ("grouped", "group-held-out confirmation with benchmarked nulls"),
        ("ordered", "blocked and embargoed confirmation with benchmarked nulls"),
    ):
        declared = SampleTableDeclaration(
            roles={**declaration().roles, "sample":
                   "group" if relationship == "grouped" else "ordering"},
            sample_relationship=relationship, units=declaration().units)
        with pytest.raises(InvalidParameterError, match=phrase):
            require_independent_samples(declared, recipe="G16 representation structure")
    invalid = SampleTableDeclaration(
        roles=declaration().roles, sample_relationship="unknown", units=declaration().units)
    with pytest.raises(InvalidParameterError, match="one of"):
        require_independent_samples(invalid, recipe="G16 representation structure")


def test_audit_finds_planted_candidate_on_generate_and_confirmation_without_claiming_truth():
    payload = dataset()
    plan = plan_representation_audit(
        payload, filename="experiment.csv", delimiter=",", declaration=declaration(),
        representations=("identity", "pca"), pca_components=1, permutations=199)
    report = run_representation_audit(payload, filename="experiment.csv", delimiter=",", plan=plan)
    by_name = {row["candidate"]: row for row in report["candidates"]}
    assert by_name["identity:signal"]["survives_both"] is True
    assert by_name["identity:signal"]["confirm_mi_nats"] > by_name["identity:noise"]["confirm_mi_nats"]
    assert by_name["identity:signal"]["nuisance_stability"][0]["nuisance"] == "b"
    assert "not conditional" in by_name["identity:signal"]["nuisance_stability"][0]["interpretation"]
    assert report["family"]["n_tests_per_partition"] == 3
    assert report["partitions"]["held_out_opened_once"] is True
    assert report["stored"] is False and report["rung_moved"] is False
    assert "not certified invariants" in report["claim_boundary"]


def test_audit_refuses_changed_file_or_tampered_plan_before_computation():
    payload = dataset()
    plan = plan_representation_audit(
        payload, filename="experiment.csv", delimiter=",", declaration=declaration(),
        representations=("identity",), pca_components=1, permutations=99)
    with pytest.raises(InvalidParameterError, match="exact file bytes"):
        run_representation_audit(payload + b"\n", filename="changed.csv", delimiter=",", plan=plan)
    altered = {**plan, "bins": 7}
    with pytest.raises(InvalidParameterError, match="complete frozen plan"):
        run_representation_audit(payload, filename="experiment.csv", delimiter=",", plan=altered)


def test_http_workflow_is_file_first_and_reuses_the_exact_sealed_plan():
    payload = dataset()
    client = TestClient(app)
    probe = client.post("/api/v1/ingress/probe",
                        files={"file": ("experiment.csv", payload, "text/csv")},
                        data={"delimiter": ","})
    assert probe.status_code == 200
    declared = declaration().canonical()
    routed = client.post(
        "/api/v1/ingress/capabilities",
        files={"file": ("experiment.csv", payload, "text/csv")},
        data={"delimiter": ",", "declaration": json.dumps(declared)})
    assert routed.status_code == 200, routed.text
    assert routed.json()["operations"]["representation_audit"]["available"] is True
    planned = client.post(
        "/api/v1/ingress/plan", files={"file": ("experiment.csv", payload, "text/csv")},
        data={"delimiter": ",", "declaration": json.dumps(declared),
              "representations": json.dumps(["identity", "pca"]),
              "pca_components": "1", "permutations": "199"})
    assert planned.status_code == 200, planned.text
    audited = client.post(
        "/api/v1/ingress/audit", files={"file": ("experiment.csv", payload, "text/csv")},
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    assert audited.status_code == 200, audited.text
    assert audited.json()["plan_sha256"] == planned.json()["plan_sha256"]
