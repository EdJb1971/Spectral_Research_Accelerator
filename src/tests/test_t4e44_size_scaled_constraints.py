"""T4E.44: the derivation that has to come before a ninth criterion.

T4E.16 declared a size-scaled criterion in full and withdrew it when a derivation showed its
surrogate could reassemble the set it was testing. The lesson recorded there is the one candidate
4's declaration should have applied and did not: ask what a rule does at the extremes of its own
domain *before* adopting it. These tests hold that derivation as computation rather than as prose,
and hold this slice to being a derivation -- not a candidate wearing one as a disguise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.errors import InvalidParameterError
from src.core.size_scaled_null import (
    DECLARED_REPLICATES, FROZEN_SCENES, SCHEMA, T4E16_DECLARATION, exclusion_variant,
    reassembly_rate, scene_count_sensitivity, simulate_reassembly, size_scaled_constraints)


def test_the_recorded_withdrawal_figure_is_reproduced_from_first_principles():
    """T4E.16's number is checked, not believed: the record has to survive being recomputed."""
    frozen = reassembly_rate()

    assert frozen["scenes"] == FROZEN_SCENES and frozen["replicates"] == DECLARED_REPLICATES
    # S!/S^S at S = 6 is 720/46656. T4E.16 recorded 0.0154.
    assert frozen["per_replicate"] == pytest.approx(0.015432, abs=5e-6)
    # Its simulations gave 0.0153 to 0.0168 per replicate, and 0.9535 to 0.9657 over 199.
    assert 0.94 < frozen["over_replicates"] < 0.97


def test_the_closed_form_agrees_with_a_simulation_of_the_same_quantity():
    """A closed form nobody checked is an assertion. This one is cheap to check, so it is."""
    exact = reassembly_rate()["per_replicate"]
    simulated = simulate_reassembly(trials=40000, seed=0)["per_replicate"]

    assert simulated == pytest.approx(exact, abs=0.002)


def test_the_defect_shrinks_with_partition_size_which_is_why_widening_it_is_forbidden():
    """The tempting repair must be visible in the numbers and refused in the open."""
    rows = {row["scenes"]: row["over_replicates"] for row in scene_count_sensitivity()}

    assert rows[6] > 0.94
    assert rows[12] < 0.02
    # Monotone in S, so the incentive to widen is real rather than hypothetical.
    assert all(rows[a] > rows[b] for a, b in zip(sorted(rows)[:-1], sorted(rows)[1:]))

    constraint = next(item for item in size_scaled_constraints()["constraints"]
                      if item["id"] == "K5")
    assert "tuned to the rule" in constraint["measured"]
    assert "widening S" in constraint["forbids"]


def test_the_exclusion_repair_states_its_bias_rather_than_reporting_a_measurement():
    """Zero by construction is a derivation. Simulating it would be a loop pretending to measure."""
    variant = exclusion_variant()

    assert variant["reassembly_rate"] == 0.0
    assert variant["reassembly_is_zero_by_construction"] is True
    assert "would report zero by construction" in variant["not_measured_because"]
    # The cost of the repair is stated in the direction that matters.
    assert variant["bias_direction"] == "TOWARD_FALSE_ADMISSION"
    assert variant["cached_matrix_survives"] is True
    assert "before adoption" in variant["what_a_candidate_using_this_owes"]


def test_every_constraint_names_what_established_it_and_what_it_forbids():
    result = size_scaled_constraints()

    assert [item["id"] for item in result["constraints"]] == [
        "K1", "K2", "K3", "K4", "K5", "K6"]
    for item in result["constraints"]:
        assert item["constraint"].strip()
        assert item["established_by"].strip()
        assert item["measured"].strip()
        assert item["forbids"].strip()

    assert result["t4e16_status"] == "WITHDRAWN_BEFORE_ADOPTION_BY_DERIVATION"
    if T4E16_DECLARATION.exists():
        assert result["t4e16_declaration_sha256"]
        assert "reassemble" in (result["t4e16_withdrawal_reason"] or "")


def test_this_slice_is_a_derivation_and_does_not_pretend_to_meet_c6():
    """The failure mode of a derivation slice is being registered as evidence for the thing."""
    result = size_scaled_constraints()

    assert result["status"] == "DERIVATION_ONLY_NO_CANDIDATE_DECLARED"
    assert result["schema"] == SCHEMA
    assert result["network_used"] is False
    assert any("category error" in item for item in result["what_this_is_not"])
    assert any("not a criterion" in item for item in result["what_this_is_not"])
    # No candidate, no acceptance, no reserve: the words a reader would look for are absent.
    body = json.dumps(result).lower()
    assert "candidate_9" not in body and "candidate 9" not in body
    assert "adopted" not in body


def test_the_published_measurement_matches_what_the_module_derives_now():
    published = Path("measurements/t4e44_size_scaled_constraints.json")
    if not published.exists():
        pytest.skip("the T4E.44 measurement has not been published in this checkout")

    recorded = json.loads(published.read_text(encoding="utf-8"))
    assert recorded["receipt_sha256"] == size_scaled_constraints()["receipt_sha256"]


def test_scene_counts_and_replicates_are_validated():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(InvalidParameterError):
            reassembly_rate(bad)
    with pytest.raises(InvalidParameterError):
        reassembly_rate(FROZEN_SCENES, 0)
