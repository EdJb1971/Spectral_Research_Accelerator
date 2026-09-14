"""T4E.46: the conditional settled, and the correction to T4E.45 held in place.

T4E.45 reported that the independence null was refuted by 1e12 and, on that basis, raised the
possibility that the discovery family was exhausted. It was not a refutation: the rates it used
were measured in planted blocks, where a motif exists by construction, while independence predicts
what chance alone does. The controlled comparison was already in the record -- motif-free null
blocks, same generator and richness, admitting zero at m = 5 and m = 6.

These tests hold the corrected reading, and hold the slice to stating the correction rather than
quietly shipping a better answer.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest

from src.core.clique_rarity import clique_rarity_derivation
from src.core.motif_free_calibration import (
    NULL_BLOCKS, PLANTED_EDGES, independence_against_the_motif_free_null,
    motif_footprint_on_graph_density, motif_free_calibration)
from src.core.size_scaled_null import FROZEN_SCENES


def test_independence_is_consistent_with_the_only_evidence_that_isolates_chance():
    """Motif-free blocks are where a chance model may legitimately be judged."""
    chance = independence_against_the_motif_free_null()

    assert chance["independence_is_refuted_by_chance_evidence"] is False
    for row in chance["rows"]:
        assert row["observed_admissions_at_five"] == 0
        assert row["observed_admissions_at_six"] == 0
        assert row["predicted_five_cliques"] < 1e-9
        assert row["consistent"] is True


def test_the_motif_does_not_set_the_graphs_density():
    """If it did, a motif-free block would be a different object and calibration there moot."""
    footprint = motif_footprint_on_graph_density()

    assert footprint["motif_is_below_block_to_block_variation"] is True
    for row in footprint["rows"]:
        assert row["motif_edges"] == comb(FROZEN_SCENES, 2) == 15
        assert row["planted_block_spread"] > row["motif_edges"]
        assert row["null_density_is_comparable"] is True
    # The null block's density is not systematically below the planted ones: at richness 9 it is
    # above every planted block, which is the opposite of a motif-driven density.
    at_nine = next(row for row in footprint["rows"] if row["richness"] == 9)
    assert at_nine["null_edges"] > max(PLANTED_EDGES[9])


def test_the_conditional_is_answered_no_and_says_what_follows():
    result = motif_free_calibration()
    conditional = result["the_conditional"]

    assert conditional["answer"] == "NO"
    assert "motif-free" in conditional["because"]
    assert "NOT forced" in conditional["consequence"]
    assert result["status"] == "DERIVATION_ONLY_NO_CANDIDATE_DECLARED"
    assert result["network_used"] is False


def test_the_correction_names_the_withdrawn_claim_and_why_it_was_wrong():
    """A correction that does not say what was wrong is a quiet replacement."""
    correction = motif_free_calibration()["corrects"]

    assert correction["task"] == "T4E.45"
    assert "1e12" in correction["the_claim_withdrawn"]
    assert "share one obstacle" in correction["the_claim_withdrawn"]
    assert "planted blocks" in correction["why_it_was_wrong"]
    assert "left unedited" in correction["what_survives_from_t4e45"]


def test_the_superseded_derivation_is_left_standing_rather_than_rewritten():
    """T4E.45's receipt must still reproduce: a corrected record keeps the thing corrected."""
    superseded = clique_rarity_derivation()

    assert superseded["status"] == "DERIVATION_ONLY_NO_CANDIDATE_DECLARED"
    # Its small-end finding is untouched by the correction and still holds.
    assert superseded["small_end"]["every_pair_expected_by_chance"] is True

    published = Path("measurements/t4e45_clique_rarity.json")
    if published.exists():
        recorded = json.loads(published.read_text(encoding="utf-8"))
        assert recorded["receipt_sha256"] == superseded["receipt_sha256"]


def test_the_real_remaining_problem_is_named_and_is_not_the_null():
    result = motif_free_calibration()
    problem = result["what_the_real_remaining_problem_is"]

    assert "motif-dependent" in problem["statement"]
    assert "one absence" in problem["mechanism"]
    assert "one scene wide" in problem["already_in_the_record"]
    assert "not about the null" in problem["why_this_matters"]


def test_the_slice_does_not_overclaim_what_it_settled():
    result = motif_free_calibration()
    disclaimers = " ".join(result["what_this_does_not_establish"]).lower()

    assert "no candidate is declared" in disclaimers
    assert "candidate 2 should be revived" in disclaimers
    assert "atmosphere" in disclaimers
    assert "declares no criterion" in result["claim_boundary"]


def test_the_null_block_counts_match_what_both_candidates_recorded():
    """Both candidate 2 and candidate 3 measured these same blocks and recorded zero."""
    assert [row[0] for row in NULL_BLOCKS] == [6, 9, 12]
    assert [row[2] for row in NULL_BLOCKS] == [116, 595, 1486]
    assert all(row[3] == 0 and row[4] == 0 for row in NULL_BLOCKS)


def test_the_published_measurement_matches_what_the_module_derives_now():
    published = Path("measurements/t4e46_motif_free_calibration.json")
    if not published.exists():
        pytest.skip("the T4E.46 measurement has not been published in this checkout")

    recorded = json.loads(published.read_text(encoding="utf-8"))
    assert recorded["receipt_sha256"] == motif_free_calibration()["receipt_sha256"]
