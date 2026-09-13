"""T4E.47: the ninth candidate, declared and unmeasured.

Eight criteria before it were falsified or withdrawn, and three of those failures were derivable
before adoption and were not derived. So most of these tests are refusals: they check that the
declaration cannot lose the reasoning that justifies it, that it has been measured on nothing, and
that declaring it opens neither reserve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.errors import UserInputError
from src.core.partial_recurrence_criterion import (
    DEFAULT_DECLARATION, REQUIRED_CONDITIONS, REQUIRED_CONSTRAINTS, RESERVED_FAMILIES,
    candidate_state, load_candidate_declaration)


def _copy(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_bytes(DEFAULT_DECLARATION.read_bytes())
    return path


def _edited(tmp_path: Path, mutate) -> Path:
    body = json.loads(DEFAULT_DECLARATION.read_text(encoding="utf-8"))
    mutate(body)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def test_the_candidate_is_declared_against_every_derived_constraint():
    declaration = load_candidate_declaration()
    constraints = declaration["constraints_this_candidate_is_declared_against"]

    for name in REQUIRED_CONSTRAINTS:
        assert any(key.startswith(name + "_") for key in constraints)
    # K1 is answered structurally rather than by argument: there is no surrogate to reconstruct.
    assert "no surrogate" in constraints["K1_a_surrogate_must_not_reconstruct_the_set_it_tests"]
    # K2 is the one not yet established, and the declaration says so rather than claiming it.
    assert "not yet established" in constraints[
        "K2_a_null_that_changes_distances_per_replicate_is_not_runnable"]


def test_the_design_records_why_the_two_derivations_forced_its_shape():
    """A rank statistic and a tightest-subset reference are consequences, not preferences."""
    declaration = load_candidate_declaration()

    rank = declaration["criterion"]["why_rank_and_not_distance"]
    assert "1.88x" in rank and "K4" in rank

    reference = declaration["the_null_and_how_the_bar_is_set"]
    assert "NOT the minimum over chance CONSISTENT sets" in reference[
        "the_reference_is_the_tightest_one_per_scene_subset"]
    forced = reference["why_that_distinction_is_forced_and_not_a_choice"]
    assert "zero consistent sets at m = 5" in forced
    assert "impostor" in forced


def test_the_extremes_are_derived_before_adoption_including_the_one_not_derivable():
    """Candidates 1, 4 and 5 failed at their own extremes. This one states all three cases."""
    derivation = load_candidate_declaration()["what_is_derivable_before_measuring"]

    assert "candidate 4" in derivation["at_m_equal_2_the_bar_is_severe"]
    assert "candidate 2" in derivation["at_m_equal_S_the_bar_is_weak_and_that_is_the_intended_direction"]
    # The honest half: the comparison the result actually turns on is not derivable here.
    assert "NOT derivable" in derivation["the_attachment_case_is_what_this_must_actually_decide"]
    assert "may admit nothing" in derivation["non_inertness_is_not_derivable"]


def test_the_feasibility_refusal_is_explicit_because_candidate_5_died_of_it():
    feasibility = load_candidate_declaration()[
        "feasibility_requirement_to_be_measured_before_adoption"]

    assert "1.1e14" in feasibility["the_requirement"]
    assert "one hour" in feasibility["the_budget"]
    assert "withdrawn on feasibility" in feasibility["the_refusal"]
    # The feasibility run must not itself be a measurement of the criterion.
    assert "spends no signal evidence" in feasibility["why_this_is_not_a_measurement_of_the_criterion"]


def test_nothing_has_been_measured_and_no_reserve_is_open():
    state = candidate_state()

    assert state["declaration_status"] == "DRAFTED_NOT_ADOPTED"
    assert state["adopted"] is False
    assert state["has_been_measured"] is False
    assert state["feasibility_measured"] is False
    assert state["reserves_open"] is False
    assert state["measured_on"] is None
    assert "Not a reserve" in state["what_adoption_would_authorise"]


def test_the_reserves_are_named_exactly_and_left_shut(tmp_path):
    declaration = load_candidate_declaration()
    confirmatory = declaration["confirmatory"]

    assert tuple(tuple(block) for block in confirmatory["blocks"]) == RESERVED_FAMILIES
    assert "UNOPENED" in confirmatory["status"]
    assert "does not open them" in confirmatory["status"]

    def open_them(body):
        body["confirmatory"]["status"] = "OPEN for the development pass"
    with pytest.raises(UserInputError, match="unopened"):
        load_candidate_declaration(_edited(tmp_path, open_them))


def test_a_declaration_that_loses_its_reasoning_is_refused(tmp_path):
    """The failure mode is editing the justification out after the numbers appear."""
    def drop_constraint(body):
        del body["constraints_this_candidate_is_declared_against"][
            "K4_no_cross_partition_calibration_of_an_absolute_bar"]
    with pytest.raises(UserInputError, match="K4"):
        load_candidate_declaration(_edited(tmp_path, drop_constraint))

    def drop_condition(body):
        del body["acceptance"]["condition_2_admission_on_partial_presence"]
    with pytest.raises(UserInputError, match="condition_2"):
        load_candidate_declaration(_edited(tmp_path, drop_condition))

    def drop_extremes(body):
        body["what_is_derivable_before_measuring"] = {"note": "trust me"}
    with pytest.raises(UserInputError, match="extremes"):
        load_candidate_declaration(_edited(tmp_path, drop_extremes))

    def drop_refusal(body):
        body["feasibility_requirement_to_be_measured_before_adoption"]["the_refusal"] = ""
    with pytest.raises(UserInputError, match="feasibility budget"):
        load_candidate_declaration(_edited(tmp_path, drop_refusal))


def test_the_acceptance_keeps_the_condition_this_candidate_is_most_likely_to_fail():
    """Candidate 3 failed condition 2 at 0.40 to 0.80. It is named as the reason this exists."""
    acceptance = load_candidate_declaration()["acceptance"]

    assert set(REQUIRED_CONDITIONS).issubset(acceptance)
    assert "0.40 to 0.80" in acceptance["condition_2_admission_on_partial_presence"]
    assert "do not trade against one another" in acceptance["all_five_required"]


def test_the_claim_boundary_refuses_the_reading_that_declaring_is_progress():
    declaration = load_candidate_declaration()
    boundary = declaration["claim_boundary"]

    assert "measured on nothing" in boundary
    assert "does not meet C6" in boundary
    would_not = declaration["what_a_success_would_and_would_not_license"]["would_not"]
    assert "mining radius" in would_not and "T4E.41 bar" in would_not
    failure = declaration["what_a_failure_would_and_would_not_license"]["would_not"]
    assert "discovery is impossible" in failure
