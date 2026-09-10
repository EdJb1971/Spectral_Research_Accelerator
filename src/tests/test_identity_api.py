"""T4E.8 slice 4: the identity declaration served, with its refusals at equal weight.

The surface exists so the most consequential scientific choice in the atmospheric sequence
stops being made by hand-editing JSON. These tests pin what it must show -- including the cell
a researcher most needs and a permissive surface would omit -- and what it must not offer.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client(tmp_path):
    app.state.identity_audit_dir = str(tmp_path)
    with TestClient(app) as test_client:
        yield test_client
    app.state.identity_audit_dir = None


def _write(directory, name, body):
    (directory / name).write_text(json.dumps(body), encoding="utf-8")


# ------------------------------------------------------------------ the admissibility matrix

def test_every_target_is_served_with_what_it_does_not_license(client):
    body = client.get("/api/v1/identity/targets").json()
    names = {row["identity_target"] for row in body["targets"]}
    assert names == {"track_continuity", "spatial_persistence", "kind_recurrence"}
    for row in body["targets"]:
        assert row["recognises"]
        assert row["does_not_license"]


def test_the_circular_pairing_is_served_as_a_refusal_not_omitted(client):
    """The cell a permissive surface would leave out is the one that matters most."""
    body = client.get("/api/v1/identity/targets").json()
    row = next(r for r in body["targets"] if r["identity_target"] == "kind_recurrence")
    cell = next(c for c in row["evidence"] if c["evidence_class"] == "record_derived_proxy")
    assert cell["admitted"] is False
    assert "validated against itself" in cell["refusal"]


def test_an_admitted_pairing_still_carries_its_caveat(client):
    body = client.get("/api/v1/identity/targets").json()
    row = next(r for r in body["targets"] if r["identity_target"] == "track_continuity")
    cell = next(c for c in row["evidence"] if c["evidence_class"] == "record_derived_proxy")
    assert cell["admitted"] is True
    assert "agreement with the tracker" in cell["caveat"]


def test_the_matrix_covers_every_target_against_every_evidence_class(client):
    body = client.get("/api/v1/identity/targets").json()
    classes = {item["evidence_class"] for item in body["evidence_classes"]}
    for row in body["targets"]:
        assert {cell["evidence_class"] for cell in row["evidence"]} == classes


def test_the_surface_says_it_does_not_choose(client):
    body = client.get("/api/v1/identity/targets").json()
    assert "does not choose" in body["choosing_is_not_automated"]
    assert any("scientific declaration" in refusal for refusal in body["refusals"])
    assert body["network_used"] is False


# ------------------------------------------------------------------------------ the receipts

def test_an_empty_store_reads_as_an_absence_of_runs(client):
    body = client.get("/api/v1/identity/audits").json()
    assert body["audits"] == []
    assert body["unreadable"] == []


def test_a_declared_receipt_surfaces_its_target_and_boundary(client, tmp_path):
    _write(tmp_path, "declared.json", {
        "status": "DISCRIMINATION_CRITERIA_NOT_MET", "approved_mining_radius": None,
        "frozen_radius": 0.138, "windows": [{}, {}, {}],
        "claim_boundary": "no radius is approved",
        "identity_declaration": {"identity_target": "spatial_persistence",
                                 "evidence_class": "record_derived_proxy",
                                 "caveat": "geometry drifts", "label_boundary": "proxy labels"},
    })
    audit = client.get("/api/v1/identity/audits").json()["audits"][0]
    assert audit["identity_declaration"]["identity_target"] == "spatial_persistence"
    assert audit["approved_mining_radius"] is None
    assert audit["claim_boundary"] == "no radius is approved"
    assert audit["windows"] == 3


def test_a_receipt_without_a_declaration_is_listed_and_named_as_undeclared(client, tmp_path):
    """Receipts written before slice 3 must not be hidden; the store would look uniform."""
    _write(tmp_path, "legacy.json", {"status": "REFUSED", "approved_mining_radius": None})
    body = client.get("/api/v1/identity/audits").json()
    assert body["audits"][0]["identity_declaration"] is None
    assert body["audits_without_a_declared_target"] == ["legacy.json"]
    assert "hiding them" in body["undeclared_note"]


def test_an_unreadable_receipt_is_reported_rather_than_skipped(client, tmp_path):
    """A store that drops what it could not parse shows a shorter list that looks complete."""
    _write(tmp_path, "good.json", {"status": "ok", "approved_mining_radius": None})
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    body = client.get("/api/v1/identity/audits").json()
    assert [item["file"] for item in body["audits"]] == ["good.json"]
    assert [item["file"] for item in body["unreadable"]] == ["broken.json"]


def test_a_json_document_that_is_not_an_object_is_unreadable_not_empty(client, tmp_path):
    (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    body = client.get("/api/v1/identity/audits").json()
    assert body["audits"] == []
    assert "not an object" in body["unreadable"][0]["error"]


def test_one_receipt_is_served_whole(client, tmp_path):
    _write(tmp_path, "one.json", {"status": "ok", "approved_mining_radius": None,
                                  "windows": [], "extra": {"kept": True}})
    body = client.get("/api/v1/identity/audits/one.json").json()
    assert body["audit"]["extra"]["kept"] is True
    assert body["summary"]["file"] == "one.json"


@pytest.mark.parametrize("name", ["../secrets.json", "a/b.json", ".hidden"])
def test_a_path_is_refused_where_a_file_name_was_required(client, name):
    assert client.get("/api/v1/identity/audits/%s" % name).status_code in (400, 404)


def test_a_missing_receipt_is_a_404_naming_what_was_asked_for(client):
    response = client.get("/api/v1/identity/audits/absent.json")
    assert response.status_code == 404
    assert "absent.json" in response.json()["detail"]


def test_an_unparseable_receipt_asked_for_by_name_is_422_not_500(client, tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    response = client.get("/api/v1/identity/audits/bad.json")
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


# ---------------------------------------------------------------------------- what it refuses

@pytest.mark.parametrize("method,path", [
    ("post", "/api/v1/identity/targets"),
    ("post", "/api/v1/identity/audits"),
    ("put", "/api/v1/identity/audits/one.json"),
    ("delete", "/api/v1/identity/audits/one.json"),
])
def test_the_surface_is_read_only(client, method, path):
    assert getattr(client, method)(path).status_code in (404, 405)


def test_the_refusals_are_published_rather_than_implied(client):
    body = client.get("/api/v1/identity/audits").json()
    assert any("approves a mining radius" in refusal for refusal in body["refusals"])
    assert any("can reach a network" in refusal for refusal in body["refusals"])


# ------------- T4E.22: the measurements, and the chain that joins a question to its answer
#
# Until this slice the router served declarations and nothing else, so a reader could see that a
# study had been DECLARED and never what it MEASURED. A declaration without its result is a
# promise; a result without its declaration is an assertion; only the pair is evidence.
#
# These tests build their own store rather than reading the repository's. A test that asserts
# against live evidence files is an assertion about today's contents, not about the code, and it
# passes for the wrong reason the moment somebody adds a record.


@pytest.fixture()
def study_store(tmp_path):
    """A declaration store and a measurement store, both isolated and both populated."""
    audits = tmp_path / "audits"
    measurements = tmp_path / "measurements"
    audits.mkdir()
    measurements.mkdir()

    _write(audits, "t4e30-example-declaration.json", {
        "schema": "example/v1", "status": "declared_before_measurement",
        "claim_boundary": "Synthetic scenes settle a property of the criterion."})
    _write(audits, "t4e30-example-adoption.json", {
        "schema": "example-adoption/v1", "status": "adopted"})
    _write(measurements, "t4e30_example.json", {
        "VERDICT": "FALSIFIED on condition 2.",
        "what_this_does_not_settle": ["Not that the signature is inadequate."],
        "claim_boundary": "A measurement settles an instrument, not the atmosphere."})

    # a question fixed and deliberately left unanswered, with its withdrawal recorded
    _write(audits, "t4e31-withdrawn-declaration.json", {
        "schema": "example/v1", "status": "WITHDRAWN_BEFORE_ADOPTION_BY_DERIVATION",
        "withdrawal": {"why": "the null could reassemble the signal"}})

    # a measurement predating the boundary convention, and a file belonging to no study
    _write(measurements, "t4e32_unbounded.json", {"result": 41})
    _write(measurements, "extension_conformance.json", {"schema": "extension-evidence/v1"})

    app.state.identity_audit_dir = str(audits)
    app.state.measurement_dir = str(measurements)
    with TestClient(app) as test_client:
        yield test_client
    app.state.identity_audit_dir = None
    app.state.measurement_dir = None


def test_a_measurement_is_served_with_what_it_may_not_be_used_for(study_store):
    """PLAN section 5: no view states a number without its boundary."""
    body = study_store.get("/api/v1/identity/measurements").json()

    assert body["network_used"] is False
    served = {m["file"]: m for m in body["measurements"]}
    example = served["t4e30_example.json"]
    assert example["verdict"].startswith("FALSIFIED")
    keys = {clause["key"] for clause in example["boundaries"]}
    assert {"claim_boundary", "what_this_does_not_settle"} <= keys


def test_a_measurement_without_a_boundary_is_listed_rather_than_passed_over(study_store):
    """Some records predate the convention. Hiding them would make the store look uniform."""
    body = study_store.get("/api/v1/identity/measurements").json()

    assert "t4e32_unbounded.json" in body["measurements_without_a_stated_boundary"]
    assert "none are hidden" in body["boundary_note"]


def test_a_corrected_record_carries_the_mark_in_its_summary(study_store, tmp_path):
    """A corrected record that reads as current is the dangerous case."""
    _write(tmp_path / "measurements", "t4e33_corrected.json",
           {"VERDICT": "superseded", "GATE_CORRECTION": {"what_happened": "the gate was wrong"},
            "claim_boundary": "settles an instrument"})

    body = study_store.get("/api/v1/identity/measurements").json()

    assert "t4e33_corrected.json" in body["corrected_or_superseded"]
    served = {m["file"]: m for m in body["measurements"]}
    assert "GATE_CORRECTION" in served["t4e33_corrected.json"]["corrected_or_superseded"]
    assert "corrected or superseded in the open" in body["correction_note"]


def test_the_studies_view_joins_a_declaration_to_the_measurement_that_answered_it(study_store):
    """The chain the work itself runs: declaration, adoption, measurement, outcome."""
    body = study_store.get("/api/v1/identity/studies").json()

    studies = {study["task"]: study for study in body["studies"]}
    assert "T4E30" in studies
    joined = studies["T4E30"]
    assert len(joined["declarations"]) == 1
    assert len(joined["adoptions"]) == 1
    assert len(joined["measurements"]) == 1
    assert joined["has_a_result"] is True
    assert joined["declared_before_measured"] is True


def test_a_declaration_with_no_result_is_shown_and_not_filtered(study_store):
    """A question fixed and deliberately left unanswered is a legitimate state here.

    T4E.16 was withdrawn before adoption by derivation. A view that filtered such a study would
    hide the cheapest result the programme produced.
    """
    body = study_store.get("/api/v1/identity/studies").json()

    assert "t4e31" in body["declared_but_not_measured"]
    assert "shown rather than filtered" in body["declared_but_not_measured_note"]
    studies = {study["task"]: study for study in body["studies"]}
    assert studies["T4E31"]["has_a_result"] is False
    assert "withdrawal" in studies["T4E31"]["corrected_or_superseded"]


def test_a_file_belonging_to_no_study_is_reported_not_dropped(study_store):
    """`extension_conformance.json` is evidence about something else, and is named as such."""
    body = study_store.get("/api/v1/identity/studies").json()

    assert "extension_conformance.json" in body["files_outside_any_study"]


def test_the_study_key_takes_the_task_and_not_the_first_token():
    """`t4e19_positional_error.json` belongs to t4e19, not to a study of its own.

    Splitting on the first separator made every measurement its own study and every declaration
    read as unanswered, which is a view worse than none.
    """
    from src.api.identity import _study_key

    assert _study_key("t4e19-positional-error-declaration.json") == "t4e19"
    assert _study_key("t4e19_positional_error.json") == "t4e19"
    assert _study_key("t4e12-consistency-adoption.json") == "t4e12"
    assert _study_key("extension_conformance.json") is None


def test_one_measurement_can_be_read_in_full_and_a_path_is_refused(study_store):
    """The name is a file name in the store and never a path."""
    full = study_store.get("/api/v1/identity/measurements/t4e30_example.json")

    assert full.status_code == 200
    assert full.json()["measurement"]["VERDICT"].startswith("FALSIFIED")
    assert full.json()["summary"]["file"] == "t4e30_example.json"
    assert study_store.get("/api/v1/identity/measurements/nothing-here.json").status_code == 404
