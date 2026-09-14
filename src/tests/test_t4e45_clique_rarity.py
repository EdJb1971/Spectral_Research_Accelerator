"""T4E.45: the extremes test that refuted the shape it was written to support.

T4E.44 recommended Shape B -- take the size scaling from the matching graph's combinatorics --
because its behaviour at the extremes could be derived before adoption rather than measured after.
It was, and it went both ways: the small end works and the large end is refuted by measurements
already on disk. These tests hold both halves, and hold the slice to reporting the refutation
rather than the half that flattered the recommendation.
"""

from __future__ import annotations

import json
from math import comb

import pytest

from src.core.clique_rarity import (
    COINCIDENTAL_FIVE_CLIQUES_OBSERVED, RECORDED_MATCHINGS, clique_rarity_derivation,
    edge_probability, expected_cliques_under_independence, independence_refutation,
    small_end_behaviour)
from src.core.errors import InvalidParameterError
from src.core.size_scaled_null import FROZEN_SCENES


def test_the_edge_density_reproduces_the_recorded_candidate_pair_count():
    """The arithmetic is anchored to a number candidate 2 published, not to an assumption."""
    # Candidate 2 at richness 6: 15 scene pairs, 20 configurations each, 6000 candidate pairs.
    assert comb(FROZEN_SCENES, 2) * 20 * 20 == 6000
    assert edge_probability(FROZEN_SCENES, 20, 121) == pytest.approx(121 / 6000)


def test_the_expectation_at_size_two_is_the_edge_count_itself():
    """A sanity check with teeth: if this were wrong the whole derivation would be."""
    for richness, configurations, edges in RECORDED_MATCHINGS:
        rows = expected_cliques_under_independence(FROZEN_SCENES, configurations, edges)
        pair = next(row for row in rows if row["size"] == 2)
        assert pair["expected_under_independence"] == pytest.approx(float(edges))


def test_the_bar_scales_with_size_without_a_parameter():
    """The expectation must collapse as the group grows, or there is no size scaling at all."""
    rows = expected_cliques_under_independence(FROZEN_SCENES, 20, 121)
    values = [row["expected_under_independence"] for row in rows]

    assert all(later < earlier for earlier, later in zip(values[:-1], values[1:]))
    # Super-exponential, not merely decreasing: six orders of magnitude per step by the top end.
    assert values[0] / values[-1] > 1e18
    assert [row["pairs_required"] for row in rows] == [comb(m, 2)
                                                       for m in range(2, FROZEN_SCENES + 1)]


def test_the_small_end_refuses_pairs_by_arithmetic_rather_than_by_a_special_case():
    """Candidate 4 needed a patch for a group of two. This shape does not."""
    small = small_end_behaviour()

    assert small["every_pair_expected_by_chance"] is True
    assert all(row["expected_pairs_by_chance"] > 100 for row in small["rows"])
    assert all(row["a_pair_is_evidence"] is False for row in small["rows"])
    assert "candidate 4" in small["reading"].lower()


def test_the_large_end_refutes_the_independence_null_using_recorded_measurements():
    """The refutation comes from candidate 3's numbers, not from a new claim about anything."""
    large = independence_refutation()

    assert large["independence_is_refuted"] is True
    assert large["direction"] == "ANTI_CONSERVATIVE"
    by_richness = {row["richness"]: row for row in large["rows"]}

    # Independence says a coincidental five-clique is effectively impossible ...
    assert by_richness[9]["predicted_five_cliques_under_independence"] < 1e-10
    assert by_richness[12]["predicted_five_cliques_under_independence"] < 1e-10
    # ... and candidate 3 measured them in half the development blocks.
    assert by_richness[9]["blocks_with_a_coincidental_five_clique"] == 2
    assert by_richness[12]["blocks_with_a_coincidental_five_clique"] == 4
    assert by_richness[12]["independence_understates_by_at_least"] > 1e12

    observed = sum(count for count, _ in COINCIDENTAL_FIVE_CLIQUES_OBSERVED.values())
    measured = sum(total for _, total in COINCIDENTAL_FIVE_CLIQUES_OBSERVED.values())
    assert (observed, measured) == (6, 12)


def test_the_derivation_reports_the_half_that_went_against_the_recommendation():
    """A derivation that only published its encouraging half would be worse than none."""
    result = clique_rarity_derivation()

    assert result["status"] == "DERIVATION_ONLY_NO_CANDIDATE_DECLARED"
    assert result["network_used"] is False
    survives = " ".join(result["what_survives"]).lower()
    does_not = " ".join(result["what_does_not_survive"]).lower()

    assert "size scaling" in survives and "pair" in survives
    assert "independence" in does_not
    # The recommendation is named as refuted, not quietly dropped.
    assert "recommendation" in does_not
    assert result["large_end"]["independence_is_refuted"] is True


def test_the_open_question_is_narrowed_rather_than_restated():
    result = clique_rarity_derivation()
    question = result["what_the_open_question_became"]

    assert question["was"] != question["is_now"]
    assert "transitivity" in question["is_now"]
    # The honest observation that both shapes may share one obstacle is kept, not buried.
    assert "K1" in question["note"]


def test_the_inputs_are_validated():
    for bad in ((1, 20, 121), (6, 0, 121), (6, 20, -1)):
        with pytest.raises(InvalidParameterError):
            edge_probability(*bad)


def test_the_published_measurement_matches_what_the_module_derives_now():
    from pathlib import Path
    published = Path("measurements/t4e45_clique_rarity.json")
    if not published.exists():
        pytest.skip("the T4E.45 measurement has not been published in this checkout")

    recorded = json.loads(published.read_text(encoding="utf-8"))
    assert recorded["receipt_sha256"] == clique_rarity_derivation()["receipt_sha256"]
