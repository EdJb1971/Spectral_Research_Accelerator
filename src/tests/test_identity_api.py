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
