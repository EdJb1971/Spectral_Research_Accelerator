"""T4E.34: composing a declaration, and committing it alone so the ordering can be proved.

T4E.30 found six studies that cannot be carried into an evidence bundle, and not one was refused
for being declared after the fact -- in every case the declaration and the measurement entered git
in the SAME commit. These tests pin the structure that makes the provable path the easy one, and
pin what the composer refuses, which is where the discipline actually lives.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.declaration_composer import (
    TASK,
    CompositionRefused,
    GateEntry,
    Prediction,
    commit_alone,
    compose,
    declaration_name,
)

GATE = [GateEntry(quantity="median offset", declared_value="52.1 km", tolerance="0.1 km")]
PREDICTIONS = [Prediction(name="reproduces", statement="the median will be 52.1 km",
                          what_would_falsify_it="any median outside 52.0 to 52.2")]
GOOD = dict(task="T4E.99", artefact="A trial study", declared_by="Edward Jonathan Bentley",
            why_this_exists="to check the composer end to end",
            what_this_is_not="not a measurement and not an adoption",
            the_inputs="the committed catalogue and the 48-shard record",
            claim_boundary="settles what this repository regenerates and nothing about any world",
            gate=GATE, predictions=PREDICTIONS)


# ---------------- names


def test_the_task_pattern_accepts_this_repository_s_own_identifiers():
    for identifier in ("T4E.28", "TG19.6", "T4E.9", "T4F.7", "t4e34"):
        assert TASK.match(identifier), identifier
    for rejected in ("nope", "T4E", "19", "", "E4.2"):
        assert not TASK.match(rejected), rejected


def test_the_file_name_is_the_one_the_rest_of_the_repository_looks_for():
    assert declaration_name("T4E.34", "The declaration composer") == \
        "t4e34-the-declaration-composer-declaration.json"


# ---------------- what it writes


def test_a_composed_declaration_is_drafted_and_not_adopted(tmp_path):
    written = compose(tmp_path, **GOOD)
    body = json.loads(written.read_text(encoding="utf-8"))

    assert body["status"] == "DRAFTED_NOT_ADOPTED"
    assert body["task"] == "T4E.99"
    assert body["declared_by"] == "Edward Jonathan Bentley"
    assert body["the_gate_declared_before_the_run"][0]["declared_value"] == "52.1 km"
    assert body["the_predictions_declared_before_the_run"][0]["what_would_falsify_it"]


def test_the_record_says_the_science_is_the_declarer_s_and_the_structure_is_not(tmp_path):
    body = json.loads(compose(tmp_path, **GOOD).read_text(encoding="utf-8"))

    assert "every word of the science below is the declarer's" in body["composed_through"]
    assert "No field was pre-filled" in body["composed_through"]


def test_the_record_names_what_would_make_it_worthless(tmp_path):
    body = json.loads(compose(tmp_path, **GOOD).read_text(encoding="utf-8"))
    assert "Adjusting the gate after seeing the result" in body["what_would_make_this_worthless"]


# ---------------- what it refuses


def test_a_prediction_without_a_falsifier_is_refused_with_the_reason(tmp_path):
    """T4E.27 declared an improvement that was impossible before the run, and it read as risk."""
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, predictions=[
            Prediction(name="p", statement="something", what_would_falsify_it="  ")]))

    assert "A prediction that cannot fail is not a prediction" in str(caught.value)
    assert not list(tmp_path.glob("*.json"))


def test_a_declaration_with_no_prediction_fixes_nothing(tmp_path):
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, predictions=[]))
    assert "can then be read as having confirmed whatever it produced" in str(caught.value)


def test_a_declaration_with_no_gate_has_nothing_to_judge_the_run_against(tmp_path):
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, gate=[]))
    assert "nothing to judge the run against" in str(caught.value)


@pytest.mark.parametrize("field", ["artefact", "why_this_exists", "what_this_is_not",
                                   "the_inputs", "claim_boundary"])
def test_every_required_narrative_field_is_refused_when_empty(tmp_path, field):
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, **{field: "   "}))
    assert field in str(caught.value)
    assert not list(tmp_path.glob("*.json"))


def test_a_placeholder_declarer_is_refused(tmp_path):
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, declared_by="maintainer"))
    assert "is not a person" in str(caught.value)


def test_a_task_without_an_identifier_is_orphaned_from_the_study_trail(tmp_path):
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **dict(GOOD, task="my study"))
    assert "orphaned from the study trail" in str(caught.value)


def test_a_declaration_is_written_once(tmp_path):
    compose(tmp_path, **GOOD)
    with pytest.raises(CompositionRefused) as caught:
        compose(tmp_path, **GOOD)
    assert "how a gate becomes whatever the result was" in str(caught.value)


# ---------------- committing it alone


@pytest.fixture()
def repository(tmp_path, monkeypatch):
    """A throwaway repository, so no test ever commits into the real one."""
    # A scoped global config, so marking this throwaway repo safe never touches the real one.
    config = tmp_path / "gitconfig"
    config.write_text("[safe]\n\tdirectory = *\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*arguments):
        return subprocess.run(("git",) + arguments, cwd=tmp_path, capture_output=True,
                              text=True, check=False)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "A Test")
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "seed.txt")
    git("commit", "-q", "-m", "seed")
    return tmp_path, git


def test_committing_alone_takes_the_declaration_and_nothing_else(repository, monkeypatch):
    """The whole point: a declaration committed with its measurement cannot be shown to
    predate it, and six of sixteen studies here are refused a bundle for exactly that."""
    root, git = repository
    monkeypatch.chdir(root)
    written = compose(root / "declarations", **GOOD)
    # Something else is dirty at the same time, exactly as in a real working session.
    (root / "unrelated.txt").write_text("work in progress\n", encoding="utf-8")
    git("add", "unrelated.txt")

    result = commit_alone(written)

    assert result.committed is True
    assert result.files_in_commit == ["declarations/" + written.name]
    # The unrelated file is still staged and uncommitted, untouched.
    assert "unrelated.txt" in git("diff", "--cached", "--name-only").stdout


def test_the_commit_explains_why_it_is_alone(repository, monkeypatch):
    root, git = repository
    monkeypatch.chdir(root)
    written = compose(root / "declarations", **GOOD)

    commit_alone(written)
    message = git("log", "-1", "--format=%B").stdout

    assert "before the run it judges" in message
    assert "cannot be shown to predate it" in message


def test_committing_an_unchanged_file_reports_nothing_to_commit(repository, monkeypatch):
    root, _git = repository
    monkeypatch.chdir(root)
    written = compose(root / "declarations", **GOOD)
    commit_alone(written)

    again = commit_alone(written)

    assert again.committed is False
    assert "nothing to commit" in again.refusal


def test_committing_a_missing_file_is_refused(repository, monkeypatch):
    root, _git = repository
    monkeypatch.chdir(root)
    result = commit_alone(root / "declarations" / "absent.json")

    assert result.committed is False
    assert "no file at" in result.refusal


def test_the_result_explains_why_alone_matters():
    result = commit_alone(Path("nothing-here.json"))
    assert "cannot be shown to predate it" in result.describe()["why_alone"]


# ---------------- the surface


@pytest.fixture()
def client():
    return TestClient(app)


def test_the_compose_route_refuses_a_prediction_with_no_falsifier_and_writes_nothing(client):
    response = client.post("/api/v1/identity/declarations/compose", json={
        "task": "T4E.99", "artefact": "x", "declared_by": "A Real Person",
        "why_this_exists": "y", "what_this_is_not": "z", "the_inputs": "i",
        "claim_boundary": "b",
        "gate": [{"quantity": "q", "declared_value": "1"}],
        "predictions": [{"name": "p", "statement": "s", "what_would_falsify_it": ""}]})

    assert response.status_code == 400
    assert "cannot fail is not a prediction" in response.json()["detail"]
    assert list(Path("data/identity_calibration").glob("t4e99-*")) == []


def test_the_compose_route_does_not_adopt_what_it_composes(client, tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTITY_AUDIT_DIR", str(tmp_path))
    response = client.post("/api/v1/identity/declarations/compose", json={
        "task": "T4E.98", "artefact": "A surface trial", "declared_by": "A Real Person",
        "why_this_exists": "y", "what_this_is_not": "z", "the_inputs": "i",
        "claim_boundary": "b",
        "gate": [{"quantity": "q", "declared_value": "1"}],
        "predictions": [{"name": "p", "statement": "s", "what_would_falsify_it": "f"}],
        "commit_it_alone": False})

    body = response.json()
    assert response.status_code == 200
    # The write must land in the temporary store. An earlier version of this test set the wrong
    # environment variable, fell back to the real calibration directory, and left a stray
    # declaration in it -- a test that pollutes the store it is testing is worse than no test.
    assert str(tmp_path).replace("\\", "/") in body["path"]
    assert body["status"] == "DRAFTED_NOT_ADOPTED"
    assert "signing at the moment of drafting" in body["composing_is_not_adopting"]
    assert "before the run, nothing can prove it predates the measurement" in body[
        "not_committed"]
