"""The read-only HTTP surface over the T4C gate record (T4C.5j).

The artifacts these routes serve were already tested where they are produced. What is new, and
what is tested here, is that **serving** them cannot weaken them:

*   a retired campaign must still be readable, with its defect intact, and must be labelled
    retired before any of its design is shown;
*   the retirement must be derived from content rather than from a file name, an identifier or
    a flag, because a frozen artifact cannot carry a flag without being edited;
*   an empty receipt store must say that no run has happened, since an empty list rendered as a
    result is the opposite claim;
*   the surface must serve no verb that could acquire, edit or re-freeze anything.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.analysis_engine.gate_campaign import (load_gate_campaign,
                                               load_gate_campaign_supersession,
                                               save_gate_campaign)
from src.analysis_engine.gate_run import RECEIPT_SCHEMA, _canonical_json, _sha256
from src.api.main import app


CAMPAIGNS = Path(__file__).resolve().parents[2] / "campaigns"
V1 = CAMPAIGNS / "t4c6_nz_era5_temperature_850_v1.json"
V2 = CAMPAIGNS / "t4c6_nz_era5_temperature_850_v2.json"
SUPERSESSION = CAMPAIGNS / "t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json"


@pytest.fixture
def client(tmp_path):
    """A client whose stores are empty by default, so a test says what it put in them."""
    app.state.gate_campaign_dir = tmp_path / "campaigns"
    app.state.gate_receipt_dir = tmp_path / "receipts"
    (tmp_path / "campaigns").mkdir()
    (tmp_path / "receipts").mkdir()
    with TestClient(app) as instance:
        yield instance
    app.state.gate_campaign_dir = None
    app.state.gate_receipt_dir = None


def _install(directory: Path, *sources: Path, names=None) -> None:
    for index, source in enumerate(sources):
        name = source.name if names is None else names[index]
        (directory / name).write_bytes(source.read_bytes())


def _receipt(path: Path, *, gate_verdict: str, scientific_verdict: str,
             power_applied: bool, reason: str) -> str:
    """A receipt that authenticates, written without running a gate.

    This is a **transport fixture and not evidence**. It exists so the route can be tested for
    what it shows a reviewer; it reports nothing about the atmosphere, and no real gate has run.
    """
    # The plan is the checked-in successor's own, re-identified. Every field of a
    # `real_era5_gate` plan is validated on construction -- the crop must be a CDS URI, the
    # scales must match the crop, the surrogate count must be able to reject after correction --
    # so inventing one from scratch only reproduces that design badly. Only the study identity
    # differs, and it differs so this fixture cannot be mistaken for a run of the campaign.
    source = load_gate_campaign(V2).gate_plan
    protocol = dataclasses.replace(source.protocol, study_id="transport-fixture")
    plan = dataclasses.replace(source, study_id="transport-fixture", protocol=protocol)
    receipt = {
        "schema": RECEIPT_SCHEMA, "plan": plan.to_mapping(),
        "plan_sha256": plan.fingerprint(),
        "gate": {"verdict": gate_verdict,
                 "protocol_fingerprint": plan.protocol.fingerprint()},
        "power_adjudication": {"scientific_verdict": scientific_verdict,
                               "power_applied": power_applied, "reason": reason},
        "scientific_verdict": scientific_verdict,
        "claim_boundary": "A transport fixture. It is not atmospheric evidence.",
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    path.write_bytes(_canonical_json(receipt) + b"\n")
    return receipt["receipt_sha256"]


# ======================================================== the surface refuses rather than lacks

def test_the_surface_publishes_what_it_will_not_do(client, tmp_path):
    _install(tmp_path / "campaigns", V1, V2, SUPERSESSION)
    body = client.get("/api/v1/gate").json()
    assert body["campaigns"] == 2 and body["supersessions"] == 1
    assert body["retired_campaigns"] == 1
    assert body["measurement_status"] == "NOT_YET_MEASURED"
    assert body["network_used"] is False
    joined = " ".join(body["refusals"]).lower()
    for expected in ("read-only", "preflight", "acquisition", "network"):
        assert expected in joined, expected


def test_no_route_on_this_surface_accepts_a_verb_that_changes_anything():
    """A read-only surface is a property of the routing table, not of the handlers' good will."""
    methods = set()
    for route in app.routes:
        if getattr(route, "path", "").startswith("/api/v1/gate"):
            methods |= (getattr(route, "methods", set()) - {"HEAD", "OPTIONS"})
    assert methods == {"GET"}, methods


# ================================================================= a retirement is about content

def test_a_retired_campaign_is_labelled_retired_and_still_served_in_full(client, tmp_path):
    _install(tmp_path / "campaigns", V1, V2, SUPERSESSION)
    rows = {row["campaign_id"]: row for row in
            client.get("/api/v1/gate/campaigns").json()["campaigns"]}
    retired = rows["t4c6-nz-era5-temperature-850-campaign-v1"]
    live = rows["t4c6-nz-era5-temperature-850-campaign-v2"]
    assert retired["status"] == "RETIRED" and live["status"] == "ACTIVE"
    assert retired["retired_by"]["successor_campaign_id"] == live["campaign_id"]
    assert retired["retired_by"]["acquisition"] == "REFUSED"
    assert live["retired_by"] is None

    # The defect is the reason the campaign was retired, so it must remain readable against the
    # campaign it belongs to. A store that hid it would have destroyed the record.
    body = client.get("/api/v1/gate/campaigns/%s" % retired["campaign_id"]).json()
    assert body["status"] == "RETIRED"
    assert body["scientific_design"]["resolvable"] is False
    assert body["scientific_design"]["full_frames"] == 7304
    assert body["campaign_sha256"] == retired["campaign_sha256"]
    assert client.get("/api/v1/gate/campaigns/%s" % live["campaign_id"]
                      ).json()["scientific_design"]["resolvable"] is True


def test_retirement_survives_a_renamed_file_because_it_is_matched_on_content(client, tmp_path):
    """A retirement keyed to a file name is defeated by `cp`, which is not a scientific act."""
    _install(tmp_path / "campaigns", V1, V2, SUPERSESSION,
             names=["design-a.json", "design-b.json", "retirement.json"])
    rows = {row["campaign_id"]: row for row in
            client.get("/api/v1/gate/campaigns").json()["campaigns"]}
    assert rows["t4c6-nz-era5-temperature-850-campaign-v1"]["status"] == "RETIRED"
    assert rows["t4c6-nz-era5-temperature-850-campaign-v1"]["file"] == "design-a.json"


def test_a_campaign_no_supersession_names_is_not_retired_by_proximity(client, tmp_path):
    """Being older, or sharing a lineage, is not a retirement. Only a record retires a design."""
    _install(tmp_path / "campaigns", V1, V2)
    rows = {row["campaign_id"]: row for row in
            client.get("/api/v1/gate/campaigns").json()["campaigns"]}
    assert {row["status"] for row in rows.values()} == {"ACTIVE"}
    assert rows["t4c6-nz-era5-temperature-850-campaign-v1"]["resolvable"] is False, (
        "the defect must still be reported; not being retired is not being sound")


def test_a_file_that_does_not_authenticate_is_listed_rather_than_dropped(client, tmp_path):
    """A shorter list that looks complete hides exactly the file whose integrity is in doubt."""
    store = tmp_path / "campaigns"
    _install(store, V2)
    envelope = json.loads(V1.read_text(encoding="utf-8"))
    envelope["campaign_sha256"] = "0" * 64
    (store / "tampered.json").write_text(json.dumps(envelope), encoding="utf-8")

    body = client.get("/api/v1/gate/campaigns").json()
    assert [row["campaign_id"] for row in body["campaigns"]] == [
        "t4c6-nz-era5-temperature-850-campaign-v2"]
    assert [entry["file"] for entry in body["unreadable"]] == ["tampered.json"]
    assert "authenticate" in body["unreadable"][0]["as_campaign"]
    assert client.get("/api/v1/gate").json()["unreadable"], (
        "the surface summary must not report a clean store when a file did not load")


# ============================================================ the retirement re-runs its checks

def test_the_supersession_review_re_runs_every_reason_against_both_campaigns(client, tmp_path):
    _install(tmp_path / "campaigns", V1, V2, SUPERSESSION)
    identifier = client.get("/api/v1/gate/supersessions").json()[
        "supersessions"][0]["supersession_id"]
    body = client.get("/api/v1/gate/supersessions/%s" % identifier).json()

    assert body["supersession_sha256"] == (
        "054592339d99141a1560b8fbecad6daaf02844a9813513331b982722f10e6711")
    assert body["reasons"], "a retirement with no checked reason is an assertion"
    for reason in body["reasons"]:
        assert reason["superseded"]["passes"] is False, reason["check"]
        assert reason["successor"]["passes"] is True, reason["check"]
    for invariant in body["preserved"]:
        assert invariant["superseded"]["passes"] is True
        assert invariant["successor"]["passes"] is True

    # The honesty field. A retirement that reads as though it closed its defects is the failure
    # this record exists to prevent, so the route must carry what it does not settle.
    assert {entry["defect"] for entry in body["deferred_to_run"]} == {"D84", "D85", "D43"}
    assert body["successor_review"]["scientific_design"]["resolvable"] is True
    assert body["network_used"] is False


def test_an_absent_record_is_a_404_rather_than_an_empty_reading(client, tmp_path):
    _install(tmp_path / "campaigns", V2)
    assert client.get("/api/v1/gate/campaigns/not-a-campaign").status_code == 404
    assert client.get("/api/v1/gate/supersessions/not-a-record").status_code == 404
    assert client.get("/api/v1/gate/receipts/not-a-receipt").status_code == 404


# ================================================================================== receipts

def test_no_receipts_is_reported_as_no_runs_and_not_as_no_findings(client):
    body = client.get("/api/v1/gate/receipts").json()
    assert body["receipts"] == [] and body["status"] == "NOT_YET_MEASURED"
    assert "absence of runs, not an absence of findings" in body["statement"]
    assert client.get("/api/v1/gate").json()["measurement_status"] == "NOT_YET_MEASURED"


def test_a_receipt_shows_both_verdicts_and_which_rule_moved_the_second(client, tmp_path):
    """An unpowered absence is INVALID, and a reader must be able to see that it was not a FAIL.

    Serving the scientific verdict alone would hide the boundary; serving the gate's alone would
    publish an absence that is a property of the crop as a negative finding about the atmosphere.
    """
    _receipt(tmp_path / "receipts" / "underpowered.json", gate_verdict="FAIL",
             scientific_verdict="INVALID", power_applied=True,
             reason="the absence is not adequately powered, so it is not a negative finding")

    index = client.get("/api/v1/gate/receipts").json()
    assert index["status"] == "MEASURED"
    assert index["receipts"][0]["gate_verdict"] == "FAIL"
    assert index["receipts"][0]["scientific_verdict"] == "INVALID"
    assert index["receipts"][0]["power_applied"] is True

    body = client.get("/api/v1/gate/receipts/underpowered").json()
    assert body["integrity"] == "VERIFIED"
    assert body["gate_verdict"] == "FAIL" and body["scientific_verdict"] == "INVALID"
    assert "not adequately powered" in body["power_adjudication"]["reason"]


def test_a_tampered_receipt_is_refused_rather_than_shown(client, tmp_path):
    path = tmp_path / "receipts" / "edited.json"
    _receipt(path, gate_verdict="FAIL", scientific_verdict="INVALID", power_applied=True,
             reason="unpowered")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["scientific_verdict"] = "PASS"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert client.get("/api/v1/gate/receipts/edited").status_code == 409
    body = client.get("/api/v1/gate/receipts").json()
    assert body["receipts"] == [] and body["unreadable"][0]["file"] == "edited.json"
    assert body["status"] == "NOT_YET_MEASURED", (
        "a store holding only a receipt that did not authenticate has measured nothing")


def test_a_receipt_id_cannot_escape_its_store(client, tmp_path):
    (tmp_path / "elsewhere.json").write_text("{}", encoding="utf-8")
    assert client.get("/api/v1/gate/receipts/..%2Felsewhere").status_code == 404


# ================================================== the checked-in store, served as it stands

def test_the_repository_store_serves_the_real_campaigns_and_their_retirement():
    """The default store is the repository's own, and this is what a reviewer opening it sees."""
    app.state.gate_campaign_dir = None
    app.state.gate_receipt_dir = None
    with TestClient(app) as instance:
        head = instance.get("/api/v1/gate").json()
        assert head["campaigns"] == 2 and head["retired_campaigns"] == 1
        assert head["measurement_status"] == "NOT_YET_MEASURED", (
            "no gate has run; if this ever fails, a receipt exists and the docs must say so")
        rows = {row["campaign_id"]: row for row in
                instance.get("/api/v1/gate/campaigns").json()["campaigns"]}
    assert rows["t4c6-nz-era5-temperature-850-campaign-v1"]["campaign_sha256"] == (
        load_gate_campaign(V1).fingerprint())
    assert rows["t4c6-nz-era5-temperature-850-campaign-v2"]["campaign_sha256"] == (
        load_gate_campaign(V2).fingerprint())
    assert (rows["t4c6-nz-era5-temperature-850-campaign-v1"]["retired_by"]["supersession_id"]
            == load_gate_campaign_supersession(SUPERSESSION).supersession_id)
    assert save_gate_campaign  # the write path exists in the module and is not routed
