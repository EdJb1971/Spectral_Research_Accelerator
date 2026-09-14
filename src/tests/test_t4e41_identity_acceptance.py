"""T4E.41: the acceptance must be decidable, and undecidable by software alone.

Half of these tests try to make the module accept T4E.8 -- by registering evidence, by
adopting the registration, by forging the declaration's own verdict semantics. The point of the
slice is that none of them can, and that each failed route fails for a stated reason rather
than by an assertion that happens to hold today.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import UserInputError
from src.core.identity_acceptance import (
    AWAITING_HUMAN_ACCEPTANCE, CONDITION_IDS, CONDITION_MET, CONDITION_NOT_MET,
    DEFAULT_DECLARATION, EVIDENCE_CLAIMED_NOT_ADOPTED, FORBIDDEN_COMPUTED_VERDICT,
    INSUFFICIENT_EVIDENCE, NOT_ACCEPTED, NO_EVIDENCE, REGISTER_SCHEMA, acceptance_state,
    load_acceptance_declaration, verify_bound_evidence)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declaration_copy(tmp_path: Path) -> Path:
    path = tmp_path / "t4e41-identity-acceptance-declaration.json"
    path.write_bytes(DEFAULT_DECLARATION.read_bytes())
    return path


def _measurement(tmp_path: Path, name: str = "candidate.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"schema": "example/v1", "result": "measured"}), encoding="utf-8")
    return path


def _register(register_dir: Path, declaration: Path, measurement: Path, condition: str,
              *, outcome: str = "MET", name: str = None) -> Path:
    register_dir.mkdir(parents=True, exist_ok=True)
    path = register_dir / (name or ("%s-%s.json" % (condition.lower(), measurement.stem)))
    path.write_text(json.dumps({
        "schema": REGISTER_SCHEMA, "task": "T4E.41", "condition": condition,
        "acceptance_declaration": str(declaration).replace("\\", "/"),
        "acceptance_declaration_sha256": _sha256(declaration),
        "measurement": str(measurement).replace("\\", "/"),
        "measurement_sha256": _sha256(measurement),
        "addresses": "a claim written for this test",
        "outcome": outcome, "status": "CLAIMED_NOT_ADOPTED",
        "claim_boundary": "not a finding",
    }, indent=2), encoding="utf-8")
    return path


def _adopt(path: Path) -> None:
    sign_declaration(
        path.parent, path.name, adopted_by="Ed Bentley", adopted_as="REGISTERED",
        what_was_adopted="this condition evidence entry", affirmation=REQUIRED_AFFIRMATION)


def test_the_declaration_states_a_bar_no_existing_evidence_can_clear():
    declaration = load_acceptance_declaration()

    assert declaration["status"] == "DRAFTED_NOT_ADOPTED"
    assert tuple(item["id"] for item in declaration["acceptance_conditions"]) == CONDITION_IDS
    assert declaration["target_and_scope"]["identity_target"] == "kind_recurrence"
    assert declaration["target_and_scope"]["admissible_evidence_class"] == "external_reference"

    # Each condition must say what would license it AND what would not. A bar that only says
    # what it wants is the prose this slice replaces.
    for condition in declaration["acceptance_conditions"]:
        assert condition["licensed_by"].strip()
        assert condition["insufficient"].strip()

    # The two results that could most plausibly be mistaken for a pass are named as not one.
    insufficient = " ".join(item["insufficient"] for item in declaration["acceptance_conditions"])
    assert "deterministic majority" in insufficient
    assert "calibrated on" in insufficient
    assert any("finite deterministic majority" in item
               for item in declaration["what_does_not_discharge_this"])


def test_the_bar_is_bound_to_the_record_it_was_set_against():
    state = verify_bound_evidence(load_acceptance_declaration())

    assert state["checked"] == 7
    assert state["all_verified"] is True
    assert state["drifted"] == [] and state["absent"] == []


def test_the_current_state_is_insufficient_evidence_and_not_a_failure():
    state = acceptance_state()

    assert state["VERDICT"] == INSUFFICIENT_EVIDENCE
    assert state["VERDICT"] != NOT_ACCEPTED
    assert state["conditions_met"] == 0
    assert all(item["status"] == NO_EVIDENCE for item in state["conditions"])
    assert state["code_may_emit_accepted"] is False
    assert state["acceptance_adopted"] is False
    assert state["network_used"] is False


def test_a_claim_is_not_evidence_until_a_named_person_adopts_it(tmp_path):
    declaration = _declaration_copy(tmp_path)
    measurement = _measurement(tmp_path)
    register_dir = tmp_path / "register"
    entry = _register(register_dir, declaration, measurement, "C1")

    state = acceptance_state(declaration, register_dir)
    condition = next(item for item in state["conditions"] if item["id"] == "C1")
    assert condition["status"] == EVIDENCE_CLAIMED_NOT_ADOPTED
    assert "not adopted" in condition["why"]
    assert state["VERDICT"] == INSUFFICIENT_EVIDENCE

    _adopt(entry)
    state = acceptance_state(declaration, register_dir)
    condition = next(item for item in state["conditions"] if item["id"] == "C1")
    assert condition["status"] == CONDITION_MET
    assert condition["evidence"][0]["adopted_by"] == "Ed Bentley"
    # One condition met is still not seven others.
    assert state["VERDICT"] == INSUFFICIENT_EVIDENCE


def test_adopted_evidence_stops_counting_when_its_measurement_changes(tmp_path):
    declaration = _declaration_copy(tmp_path)
    measurement = _measurement(tmp_path)
    register_dir = tmp_path / "register"
    _adopt(_register(register_dir, declaration, measurement, "C2"))
    assert next(item for item in acceptance_state(declaration, register_dir)["conditions"]
                if item["id"] == "C2")["status"] == CONDITION_MET

    measurement.write_text(json.dumps({"schema": "example/v1", "result": "edited"}),
                           encoding="utf-8")

    condition = next(item for item in acceptance_state(declaration, register_dir)["conditions"]
                     if item["id"] == "C2")
    assert condition["status"] == EVIDENCE_CLAIMED_NOT_ADOPTED
    assert "bytes have changed" in condition["why"]

    measurement.unlink()
    condition = next(item for item in acceptance_state(declaration, register_dir)["conditions"]
                     if item["id"] == "C2")
    assert "absent" in condition["why"]


def test_a_registered_failure_counts_against_the_condition_rather_than_being_dropped(tmp_path):
    declaration = _declaration_copy(tmp_path)
    register_dir = tmp_path / "register"
    passing = _measurement(tmp_path, "passing.json")
    failing = _measurement(tmp_path, "failing.json")
    _adopt(_register(register_dir, declaration, passing, "C3"))
    _adopt(_register(register_dir, declaration, failing, "C3", outcome="NOT_MET"))

    state = acceptance_state(declaration, register_dir)
    condition = next(item for item in state["conditions"] if item["id"] == "C3")
    assert condition["status"] == CONDITION_NOT_MET
    assert state["VERDICT"] == NOT_ACCEPTED


def test_evidence_registered_against_another_acceptance_is_refused(tmp_path):
    declaration = _declaration_copy(tmp_path)
    measurement = _measurement(tmp_path)
    register_dir = tmp_path / "register"
    entry = _register(register_dir, declaration, measurement, "C4")
    body = json.loads(entry.read_text(encoding="utf-8"))

    entry.write_text(json.dumps(
        {**body, "acceptance_declaration_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(UserInputError, match="different acceptance declaration"):
        acceptance_state(declaration, register_dir)

    entry.write_text(json.dumps({**body, "condition": "C9"}), encoding="utf-8")
    with pytest.raises(UserInputError, match="not C1..C8"):
        acceptance_state(declaration, register_dir)

    entry.write_text(json.dumps({**body, "schema": "something-else/v1"}), encoding="utf-8")
    with pytest.raises(UserInputError, match="is not a %s entry" % REGISTER_SCHEMA):
        acceptance_state(declaration, register_dir)


def test_no_combination_of_evidence_lets_code_accept_t4e8(tmp_path):
    declaration = _declaration_copy(tmp_path)
    register_dir = tmp_path / "register"
    for index, condition in enumerate(CONDITION_IDS):
        measurement = _measurement(tmp_path, "evidence-%d.json" % index)
        _adopt(_register(register_dir, declaration, measurement, condition))

    state = acceptance_state(declaration, register_dir)

    # Every condition is met, by adopted evidence, and the verdict still is not acceptance.
    assert state["conditions_met"] == len(CONDITION_IDS)
    assert state["VERDICT"] == AWAITING_HUMAN_ACCEPTANCE
    assert state["VERDICT"] != FORBIDDEN_COMPUTED_VERDICT
    assert FORBIDDEN_COMPUTED_VERDICT not in json.dumps(state["conditions"])
    assert state["code_may_emit_accepted"] is False

    # Adopting the acceptance declaration itself does not change that either.
    sign_declaration(
        declaration.parent, declaration.name, adopted_by="Ed Bentley",
        adopted_as="ACCEPTANCE_BAR_ADOPTED", what_was_adopted="the T4E.41 acceptance bar",
        affirmation=REQUIRED_AFFIRMATION)
    adopted = acceptance_state(declaration, register_dir)
    assert adopted["acceptance_adopted"] is True
    assert adopted["VERDICT"] == AWAITING_HUMAN_ACCEPTANCE


def test_a_declaration_that_softens_the_bar_is_refused(tmp_path):
    body = json.loads(DEFAULT_DECLARATION.read_text(encoding="utf-8"))
    path = tmp_path / "declaration.json"

    dropped = dict(body)
    dropped["acceptance_conditions"] = body["acceptance_conditions"][:-1]
    path.write_text(json.dumps(dropped), encoding="utf-8")
    with pytest.raises(UserInputError, match="exactly 8 conditions"):
        load_acceptance_declaration(path)

    reordered = dict(body)
    reordered["acceptance_conditions"] = list(reversed(body["acceptance_conditions"]))
    path.write_text(json.dumps(reordered), encoding="utf-8")
    with pytest.raises(UserInputError, match="not C1..C8 in order"):
        load_acceptance_declaration(path)

    # The clause that forbids software from accepting is itself part of the contract.
    permissive = json.loads(json.dumps(body))
    permissive["verdict_semantics"][FORBIDDEN_COMPUTED_VERDICT] = "compute it when all met"
    path.write_text(json.dumps(permissive), encoding="utf-8")
    with pytest.raises(UserInputError, match="must forbid code from emitting"):
        load_acceptance_declaration(path)

    hollow = json.loads(json.dumps(body))
    hollow["acceptance_conditions"][2]["insufficient"] = "   "
    path.write_text(json.dumps(hollow), encoding="utf-8")
    with pytest.raises(UserInputError, match="condition C3 lacks insufficient"):
        load_acceptance_declaration(path)


def test_the_bar_is_signed_through_the_ordinary_adoption_surface(tmp_path):
    """The UI must be able to record an adoption. It must not be able to originate one.

    Both halves are asserted here against the same endpoint the panel calls: a wrong
    affirmation writes nothing, and a correct one -- typed by a person, carried by code --
    adopts the bar while leaving every condition and the verdict exactly where they were.
    """
    declaration = _declaration_copy(tmp_path)
    app.state.identity_audit_dir = str(tmp_path)
    app.state.identity_acceptance_declaration = str(declaration)
    app.state.identity_acceptance_register = str(tmp_path / "register")
    client = TestClient(app)
    body = {
        "declaration": declaration.name,
        "adopted_by": "Ed Bentley",
        "adopted_as": "ACCEPTANCE_BAR_ADOPTED",
        "what_was_adopted": "the T4E.41 acceptance bar",
    }
    try:
        # The bar is reachable from the same surface that lists unsigned declarations.
        listed = client.get("/api/v1/identity/declarations").json()["declarations"]
        row = next(item for item in listed if item["declaration"] == declaration.name)
        assert row["adopted"] is False

        refused = client.post("/api/v1/identity/declarations/sign",
                              json={**body, "affirmation": "yes, adopt it"})
        assert refused.status_code == 400
        assert not list(tmp_path.glob("*-adoption.json"))
        assert client.get("/api/v1/identity/acceptance").json()["acceptance_adopted"] is False

        adopted = client.post("/api/v1/identity/declarations/sign",
                              json={**body, "affirmation": REQUIRED_AFFIRMATION})
        assert adopted.status_code == 200
        assert adopted.json()["adopted"]["adopted_by"] == "Ed Bentley"

        state = client.get("/api/v1/identity/acceptance").json()
        assert state["acceptance_adopted"] is True
        # Adopting the bar fixes the standard. It meets nothing and accepts nothing.
        assert state["VERDICT"] == INSUFFICIENT_EVIDENCE
        assert state["conditions_met"] == 0
        assert state["code_may_emit_accepted"] is False
    finally:
        for name in ("identity_audit_dir", "identity_acceptance_declaration",
                     "identity_acceptance_register"):
            delattr(app.state, name)


def test_the_surface_carries_the_declaration_a_signer_affirms_having_read(tmp_path):
    """`I have read this declaration` must be honourable where it is typed."""
    state = acceptance_state()

    assert state["declaration_file"].endswith(".json")
    assert state["why_this_exists"] and state["what_this_declaration_is_not"]
    assert state["target_and_scope"]["identity_target"] == "kind_recurrence"
    assert state["what_acceptance_would_and_would_not_license"]["would_not"]
    assert state["verdict_semantics"][FORBIDDEN_COMPUTED_VERDICT]
