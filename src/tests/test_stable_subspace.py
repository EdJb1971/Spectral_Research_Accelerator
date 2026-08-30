"""TG16.3 bounded generate-only stable-subspace search."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis_engine.stable_subspace import (
    certify_external_subspaces, confirm_stable_subspaces, generate_stable_subspaces,
    projector, projector_distance)
from src.api.main import app
from src.core.errors import InvalidParameterError
from src.data_layer.dataset_ingress import (
    SampleTableDeclaration, freeze_external_subspace_transfer,
    freeze_stable_subspace_confirmation, plan_stable_subspace_generation,
    publish_stable_subspace_candidate, run_external_subspace_certification,
    run_stable_subspace_confirmation, run_stable_subspace_generation,
    sample_table_capability_profile)
from src.core.preregistration import HeldOutLedger, Seal


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


def _payload(kind: str = "signal", *, nuisance: bool = False, seed: int = 83) -> bytes:
    first, second, target, nuisance_values = _arrays(kind, seed=seed)
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
    assert profile["operations"]["stable_subspace_confirmation"]["available"] is True
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


def test_confirmation_applies_frozen_span_and_corrects_complete_family():
    first, second, target, _nuisance = _arrays("signal", n=220)
    generated = generate_stable_subspaces(
        {"first": first[:150], "second": second[:150]}, target[:150],
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=201)
    confirmed = confirm_stable_subspaces(
        {"first": first[150:], "second": second[150:]}, target[150:],
        preprocessing=generated["preprocessing"],
        frozen_subspaces=generated["subspaces"],
        family_members=generated["family"]["members"], permutations=39, seed=202)
    assert confirmed["correction_unit"] == 1
    assert confirmed["internally_replicated_candidates"]
    assert confirmed["subspaces"][0]["outcome"] == "internally_replicated_candidate"
    assert "not an external replication certificate" in confirmed["claim_boundary"]


def test_confirmation_refuses_inadequate_frozen_nuisance_region_overlap():
    first, second, target, nuisance = _arrays("nuisance_stable", n=180)
    generated = generate_stable_subspaces(
        {"first": first[:120], "second": second[:120]}, target[:120], nuisance[:120],
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=203)
    cuts = np.quantile(nuisance[:120], (1 / 3, 2 / 3))
    with pytest.raises(ValueError, match="every frozen nuisance region"):
        confirm_stable_subspaces(
            {"first": first[120:], "second": second[120:]}, target[120:],
            preprocessing=generated["preprocessing"],
            frozen_subspaces=generated["subspaces"],
            family_members=generated["family"]["members"],
            nuisance=np.full(60, cuts[0] - 1), nuisance_region_cuts=cuts,
            permutations=39, seed=204)


def test_confirmation_seal_is_content_bound_and_partition_opens_once():
    payload = _payload()
    plan = plan_stable_subspace_generation(
        payload, filename="linear.csv", delimiter=",", declaration=_declaration(),
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=205)
    generation = run_stable_subspace_generation(
        payload, filename="linear.csv", delimiter=",", plan=plan)
    ledger = HeldOutLedger()
    seal = freeze_stable_subspace_confirmation(
        payload, filename="linear.csv", delimiter=",", plan=plan,
        generation=generation, sealed_at="2026-08-30T00:00:00Z",
        confirmation_permutations=39, confirmation_seed=206, ledger=ledger)
    assert seal["confirmation_opened"] is False and seal["correction_unit"] == 1
    with pytest.raises(InvalidParameterError, match="published"):
        run_stable_subspace_confirmation(
            payload, filename="linear.csv", delimiter=",", seal=seal, ledger=ledger,
            opened_at="2026-08-30T00:00:30Z", published_sha256="0" * 64)
    assert not ledger.records
    receipt = run_stable_subspace_confirmation(
        payload, filename="linear.csv", delimiter=",", seal=seal, ledger=ledger,
        opened_at="2026-08-30T00:01:00Z", published_sha256=seal["seal_sha256"])
    assert receipt["confirmation_opened"] is True
    assert receipt["internally_replicated_candidates"]
    assert receipt["stored"] is False and receipt["rung_moved"] is False
    with pytest.raises(InvalidParameterError, match="held-out partition that has not been tested"):
        run_stable_subspace_confirmation(
            payload, filename="linear.csv", delimiter=",", seal=seal, ledger=ledger,
            opened_at="2026-08-30T00:02:00Z")
    with pytest.raises(InvalidParameterError, match="complete generation result"):
        freeze_stable_subspace_confirmation(
            payload, filename="linear.csv", delimiter=",", plan=plan,
            generation={**generation, "preprocessing": {}},
            sealed_at="2026-08-30T00:03:00Z", confirmation_permutations=39)


def test_multipart_confirmation_freeze_and_confirm_spend_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAL_PREREGISTRATION_ROOT", str(tmp_path))
    payload = _payload()
    client = TestClient(app)
    common_file = {"file": ("linear.csv", payload, "text/csv")}
    planned = client.post(
        "/api/v1/ingress/subspace/plan", files=common_file,
        data={"delimiter": ",", "declaration": json.dumps(_declaration().canonical()),
              "dimensions": "[1]", "regularizations": "[0.1]", "permutations": "39",
              "restarts": "3", "iterations": "24", "perturbations": "3", "seed": "207"})
    generated = client.post(
        "/api/v1/ingress/subspace/generate", files=common_file,
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    frozen = client.post(
        "/api/v1/ingress/subspace/freeze", files=common_file,
        data={"delimiter": ",", "plan": json.dumps(planned.json()),
              "generation": json.dumps(generated.json()),
              "confirmation_permutations": "39", "confirmation_seed": "208"})
    assert frozen.status_code == 200, frozen.text
    confirmed = client.post(
        "/api/v1/ingress/subspace/confirm", files=common_file,
        data={"delimiter": ",", "seal_sha256": frozen.json()["seal_sha256"],
              "published_sha256": frozen.json()["seal_sha256"]})
    assert confirmed.status_code == 200, confirmed.text
    repeated = client.post(
        "/api/v1/ingress/subspace/confirm", files=common_file,
        data={"delimiter": ",", "seal_sha256": frozen.json()["seal_sha256"]})
    assert repeated.status_code == 400
    assert "second confirmation" in repeated.text


def test_external_certification_uses_unchanged_published_span_and_spends_target_once():
    source = _payload(seed=301)
    target = _payload(seed=302)
    declaration = _declaration()
    plan = plan_stable_subspace_generation(
        source, filename="source.csv", delimiter=",", declaration=declaration,
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=303)
    generation = run_stable_subspace_generation(
        source, filename="source.csv", delimiter=",", plan=plan)
    confirmation_seal = freeze_stable_subspace_confirmation(
        source, filename="source.csv", delimiter=",", plan=plan,
        generation=generation, sealed_at="2026-08-30T00:00:00Z",
        confirmation_permutations=39, confirmation_seed=304)
    candidate = publish_stable_subspace_candidate(
        confirmation_seal=Seal.from_mapping(confirmation_seal["seal"]),
        label=generation["candidate_compact_stable_subspaces"][0]["label"],
        published_at="2026-08-30T00:01:00Z")
    ledger = HeldOutLedger()
    transfer = freeze_external_subspace_transfer(
        candidates=[candidate], target_content_sha256=hashlib.sha256(target).hexdigest(),
        target_n_rows=180, target_declaration=declaration,
        target_provenance={"acquisition_id": "independent-302",
                           "acquired_at": "2026-08-29T00:00:00Z",
                           "source": "independent fixture", "independent_of_origin": True},
        sealed_at="2026-08-30T00:02:00Z", permutations=39, seed=305, ledger=ledger)
    with pytest.raises(InvalidParameterError, match="published"):
        run_external_subspace_certification(
            target, filename="target.csv", delimiter=",", seal=transfer, ledger=ledger,
            opened_at="2026-08-30T00:03:00Z", published_sha256="0" * 64)
    assert not ledger.records
    receipt = run_external_subspace_certification(
        target, filename="target.csv", delimiter=",", seal=transfer, ledger=ledger,
        opened_at="2026-08-30T00:04:00Z",
        published_sha256=transfer["seal_sha256"])
    assert receipt["external_replication_receipts"]
    assert receipt["subspaces"][0]["outcome"] == "external_replication_receipt"
    assert receipt["target_provenance"]["independent_of_origin"] is True
    assert "does not prove the acquisition declaration" in receipt["claim_boundary"]
    with pytest.raises(InvalidParameterError, match="held-out partition"):
        run_external_subspace_certification(
            target, filename="renamed.csv", delimiter=",", seal=transfer, ledger=ledger,
            opened_at="2026-08-30T00:05:00Z",
            published_sha256=transfer["seal_sha256"])


def test_external_contract_refuses_source_bytes_adaptation_and_tampered_candidate():
    source = _payload(seed=311)
    declaration = _declaration()
    plan = plan_stable_subspace_generation(
        source, filename="source.csv", delimiter=",", declaration=declaration,
        dimensions=(1,), regularizations=(0.1,), permutations=39,
        restarts=3, iterations=24, perturbations=3, seed=312)
    generation = run_stable_subspace_generation(
        source, filename="source.csv", delimiter=",", plan=plan)
    seal = freeze_stable_subspace_confirmation(
        source, filename="source.csv", delimiter=",", plan=plan, generation=generation,
        sealed_at="2026-08-30T01:00:00Z", confirmation_permutations=39)
    candidate = publish_stable_subspace_candidate(
        confirmation_seal=Seal.from_mapping(seal["seal"]),
        label=generation["candidate_compact_stable_subspaces"][0]["label"],
        published_at="2026-08-30T01:01:00Z")
    provenance = {"acquisition_id": "declared-independent", "acquired_at": "2026-08-29",
                  "source": "fixture archive", "independent_of_origin": True}
    with pytest.raises(InvalidParameterError, match="different from the source"):
        freeze_external_subspace_transfer(
            candidates=[candidate], target_content_sha256=hashlib.sha256(source).hexdigest(),
            target_n_rows=180, target_declaration=declaration,
            target_provenance=provenance, sealed_at="2026-08-30T01:02:00Z",
            permutations=39)
    tampered = {**candidate, "preprocessing": {"mean": [0, 0], "scale": [1, 1]}}
    with pytest.raises(InvalidParameterError, match="candidate_sha256"):
        freeze_external_subspace_transfer(
            candidates=[tampered], target_content_sha256="1" * 64,
            target_n_rows=180, target_declaration=declaration,
            target_provenance=provenance, sealed_at="2026-08-30T01:03:00Z",
            permutations=39)


def test_multipart_publish_freeze_and_external_certify_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAL_PREREGISTRATION_ROOT", str(tmp_path))
    source, target = _payload(seed=321), _payload(seed=322)
    client = TestClient(app)
    source_file = {"file": ("source.csv", source, "text/csv")}
    planned = client.post(
        "/api/v1/ingress/subspace/plan", files=source_file,
        data={"delimiter": ",", "declaration": json.dumps(_declaration().canonical()),
              "dimensions": "[1]", "regularizations": "[0.1]", "permutations": "39",
              "restarts": "3", "iterations": "24", "perturbations": "3", "seed": "323"})
    generated = client.post(
        "/api/v1/ingress/subspace/generate", files=source_file,
        data={"delimiter": ",", "plan": json.dumps(planned.json())})
    frozen = client.post(
        "/api/v1/ingress/subspace/freeze", files=source_file,
        data={"delimiter": ",", "plan": json.dumps(planned.json()),
              "generation": json.dumps(generated.json()),
              "confirmation_permutations": "39", "confirmation_seed": "324"})
    confirmed = client.post(
        "/api/v1/ingress/subspace/confirm", files=source_file,
        data={"delimiter": ",", "seal_sha256": frozen.json()["seal_sha256"],
              "published_sha256": frozen.json()["seal_sha256"]})
    assert confirmed.status_code == 200, confirmed.text
    label = confirmed.json()["internally_replicated_candidates"][0]["label"]
    published = client.post(
        "/api/v1/ingress/subspace/publish",
        data={"seal_sha256": frozen.json()["seal_sha256"], "label": label})
    assert published.status_code == 200, published.text
    transfer = client.post(
        "/api/v1/ingress/subspace/transfer/freeze",
        data={"candidate_sha256": json.dumps([published.json()["candidate_sha256"]]),
              "target_content_sha256": hashlib.sha256(target).hexdigest(),
              "target_n_rows": "180",
              "target_declaration": json.dumps(_declaration().canonical()),
              "target_provenance": json.dumps({
                  "acquisition_id": "external-322", "acquired_at": "2026-08-29",
                  "source": "independent fixture", "independent_of_origin": True}),
              "permutations": "39", "seed": "325"})
    assert transfer.status_code == 200, transfer.text
    target_file = {"file": ("target.csv", target, "text/csv")}
    certified = client.post(
        "/api/v1/ingress/subspace/transfer/certify", files=target_file,
        data={"delimiter": ",", "transfer_seal_sha256": transfer.json()["seal_sha256"],
              "published_sha256": transfer.json()["seal_sha256"]})
    assert certified.status_code == 200, certified.text
    assert certified.json()["external_replication_receipts"]
    repeated = client.post(
        "/api/v1/ingress/subspace/transfer/certify", files=target_file,
        data={"delimiter": ",", "transfer_seal_sha256": transfer.json()["seal_sha256"],
              "published_sha256": transfer.json()["seal_sha256"]})
    assert repeated.status_code == 400
    assert "held-out partition" in repeated.text
