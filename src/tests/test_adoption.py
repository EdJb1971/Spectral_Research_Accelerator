"""T4E.32/T4E.33: signing and convening, performed through a surface rather than a text editor.

The rule is unchanged -- code does not sign a scientific declaration for a person -- and these
tests are mostly about what the surface REFUSES, because that is where the rule now lives. A
maintainer who reads a declaration, types their name and types the affirmation has signed it; a
form that filled any of that in would be signing on their behalf while appearing to ask.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.adoption import (
    PLACEHOLDER_NAMES,
    REQUIRED_AFFIRMATION,
    AdoptionRefused,
    adoption_name,
    adoption_state,
    sign_declaration,
)

GOOD = {"adopted_by": "Edward Jonathan Bentley, maintainer",
        "adopted_as": "A_TEST_ADOPTION",
        "what_was_adopted": "the declaration as written",
        "affirmation": REQUIRED_AFFIRMATION}


@pytest.fixture()
def store(tmp_path):
    (tmp_path / "t4e99-example-declaration.json").write_text(
        json.dumps({"schema": "x/v1", "task": "T4E.99"}), encoding="utf-8")
    return tmp_path


# ---------------- what the signature binds


def test_signing_writes_an_adoption_bound_to_the_declaration_digest(store):
    adoption = sign_declaration(store, "t4e99-example-declaration.json", **GOOD)
    body = json.loads(adoption.path.read_text(encoding="utf-8"))

    import hashlib
    expected = hashlib.sha256(
        (store / "t4e99-example-declaration.json").read_bytes()).hexdigest()
    assert body["adopts_sha256"] == expected
    assert body["adopted_by"] == GOOD["adopted_by"]
    assert adoption.path.name == "t4e99-example-adoption.json"


def test_the_adoption_records_that_a_person_signed_it_and_not_the_software(store):
    adoption = sign_declaration(store, "t4e99-example-declaration.json", **GOOD)

    assert "Code does not sign a scientific declaration for anyone" in adoption.body[
        "signed_through"]
    assert "does not make the declaration correct" in adoption.body[
        "what_this_adoption_does_not_do"]


def test_the_adoption_name_follows_the_repository_convention():
    assert adoption_name("t4e28-join-rerun-declaration.json") == "t4e28-join-rerun-adoption.json"
    assert adoption_name("t4e18-vorticity-acquisition-design.json") == \
        "t4e18-vorticity-acquisition-adoption.json"


# ---------------- what it refuses


def test_one_click_cannot_sign_anything(store):
    """A signature that can be produced by one press is one that can be produced by accident."""
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "t4e99-example-declaration.json",
                         **dict(GOOD, affirmation=""))
    assert REQUIRED_AFFIRMATION in str(caught.value)
    assert not list(store.glob("*adoption*"))


def test_an_approximate_affirmation_is_refused(store):
    with pytest.raises(AdoptionRefused):
        sign_declaration(store, "t4e99-example-declaration.json",
                         **dict(GOOD, affirmation="i adopt it"))


def test_a_trailing_full_stop_is_tolerated_because_it_is_not_a_different_intention(store):
    adoption = sign_declaration(store, "t4e99-example-declaration.json",
                                **dict(GOOD, affirmation=REQUIRED_AFFIRMATION + "."))
    assert adoption.path.is_file()


@pytest.mark.parametrize("name", ["", "me", "Claude", "the maintainer", "AI", "unknown"])
def test_a_placeholder_is_not_a_person(store, name):
    """An adoption is an attribution. A placeholder where the person should be voids it."""
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "t4e99-example-declaration.json", **dict(GOOD, adopted_by=name))
    assert "is not a person" in str(caught.value)
    assert name.strip().lower() in PLACEHOLDER_NAMES or len(name.strip()) < 3


def test_an_adoption_that_does_not_say_what_was_adopted_is_refused(store):
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "t4e99-example-declaration.json",
                         **dict(GOOD, what_was_adopted="   "))
    assert "records only that a button was pressed" in str(caught.value)


def test_a_missing_declaration_is_refused_rather_than_signed_into_existence(store):
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "t4e00-nothing-declaration.json", **GOOD)
    assert "would bind a digest of nothing" in str(caught.value)


def test_a_path_is_refused_rather_than_sanitised(store):
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "../secrets.json", **GOOD)
    assert "a path is refused rather than sanitised" in str(caught.value)


def test_an_adoption_is_written_once_and_never_overwritten(store):
    sign_declaration(store, "t4e99-example-declaration.json", **GOOD)
    with pytest.raises(AdoptionRefused) as caught:
        sign_declaration(store, "t4e99-example-declaration.json", **GOOD)
    assert "two different scientific acts" in str(caught.value)


# ---------------- drift


def test_a_declaration_amended_after_signing_no_longer_carries_the_signature(store):
    sign_declaration(store, "t4e99-example-declaration.json", **GOOD)
    (store / "t4e99-example-declaration.json").write_text(
        json.dumps({"schema": "x/v1", "task": "T4E.99", "added": "later"}), encoding="utf-8")
    state = adoption_state(store, "t4e99-example-declaration.json")

    assert state["adopted"] is True
    assert state["signature_still_reaches_the_declaration"] is False
    assert "what was signed is not what is on disk" in state["drift"]


def test_an_unamended_declaration_still_carries_its_signature(store):
    sign_declaration(store, "t4e99-example-declaration.json", **GOOD)
    state = adoption_state(store, "t4e99-example-declaration.json")

    assert state["signature_still_reaches_the_declaration"] is True
    assert "drift" not in state


# ---------------- the surfaces


@pytest.fixture()
def client():
    return TestClient(app)


def test_the_declaration_index_shows_what_is_unsigned_and_refuses_to_supply_a_name(client):
    body = client.get("/api/v1/identity/declarations").json()
    rows = {row["declaration"]: row for row in body["declarations"]}

    assert body["required_affirmation"] == REQUIRED_AFFIRMATION
    assert "signing on the maintainer's behalf" in body["what_this_surface_will_not_supply"]
    assert rows["t4e28-join-rerun-declaration.json"]["adopted"] is False
    assert rows["t4e27-position-tolerance-declaration.json"]["adopted"] is True


def test_the_sign_route_refuses_a_bad_affirmation_and_writes_nothing(client, tmp_path):
    response = client.post("/api/v1/identity/declarations/sign", json={
        "declaration": "t4e28-join-rerun-declaration.json", "adopted_by": "A Person",
        "adopted_as": "X", "what_was_adopted": "y", "affirmation": "yes"})

    assert response.status_code == 400
    assert REQUIRED_AFFIRMATION in response.json()["detail"]
    assert not Path("data/identity_calibration/t4e28-join-rerun-adoption.json").exists()


def test_convening_refuses_without_authorisation_and_sends_nothing(client):
    response = client.post("/api/v1/reviews/studies/t4e28-join-rerun/round-robin", json={})

    assert response.status_code == 400
    assert "Nothing was sent" in response.json()["detail"]
    assert "costs money" in response.json()["detail"]


def test_convening_refuses_without_a_key_and_never_takes_one_from_the_request(client,
                                                                             monkeypatch):
    from src.api.reviews import KEY_VARIABLES

    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    response = client.post("/api/v1/reviews/studies/t4e28-join-rerun/round-robin",
                           json={"i_authorise_paid_calls": True})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "never from the request" in detail
    assert "never in a browser, a log" in detail


def test_the_panel_plan_states_the_call_count_before_anything_is_sent(client):
    body = client.get("/api/v1/reviews/panel-plan").json()

    assert body["calls_if_every_turn_is_taken"] == 8
    assert body["calls_if_nothing_is_dissented_from"] == 7
    assert "rebuttal of nothing" in body["why_that_differs"]
    assert body["network_used"] is False
    assert [seat["role"] for seat in body["roles"]][0] == "candidate_synthesis"


def test_an_unknown_seat_is_refused_with_the_roles_that_exist(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-used-because-the-seat-fails-first")
    response = client.post("/api/v1/reviews/studies/t4e28-join-rerun/round-robin", json={
        "i_authorise_paid_calls": True, "seats": {"closing_argument": "a-model"}})

    assert response.status_code == 400
    assert "is not a review role" in response.json()["detail"]
