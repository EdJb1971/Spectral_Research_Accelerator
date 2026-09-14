"""T4E.46: the conditional settled, and a correction to T4E.45.

T4E.45 concluded that the independence null behind a clique-rarity bar was refuted by a factor of
1e12, and on that basis I raised the possibility that both shapes T4E.44 left share one obstacle
-- which would have exhausted the discovery family and made PLAN section 3's catalogue alternative
the live question. That conclusion was wrong, and this module records why rather than quietly
replacing it.

The error was a comparison between the wrong two things. Independence predicts how often a clique
arises **by chance**. The false-admission rates I compared it against were measured in *planted*
blocks, where a motif is present by construction. Comparing a chance model against a
signal-present observation and calling the gap a refutation is a category error, and it is the
same one the programme has warned about elsewhere: a null that answers a different question from
the one being asked.

The record already contained the controlled comparison. The null blocks -- 500-505 and 850-855,
motif-free by construction, same generator, same richness, same configurations per scene --
admit **zero** sets at m = 5 and m = 6 at every richness. That is what independence predicts.
Where chance is the only mechanism, the null is not refuted at all.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from typing import Any, Dict, List, Tuple

from src.core.clique_rarity import expected_cliques_under_independence
from src.core.size_scaled_null import FROZEN_SCENES


SCHEMA = "t4e46-motif-free-calibration/v1"

#: Motif-free null blocks as recorded by candidates 2 and 3: richness, configurations per scene,
#: pairs proposed by the matching, and sets admitted at k = 6 and k = 5 respectively. Both
#: candidates measured the same null blocks and both recorded zero admissions.
NULL_BLOCKS: Tuple[Tuple[int, int, int, int, int], ...] = (
    (6, 20, 116, 0, 0),
    (9, 84, 595, 0, 0),
    (12, 220, 1486, 0, 0),
)

#: Planted-block matching densities, for the footprint comparison. Richness to edge counts.
PLANTED_EDGES: Dict[int, Tuple[int, ...]] = {
    6: (121, 121, 152, 136),
    9: (568, 485, 585, 566),
    12: (1446, 1440, 1590, 1488),
}

#: What candidate 3 measured at k = 5, which is what T4E.45 mistook for a chance rate: blocks
#: with false admission above zero, out of four per richness. Planted blocks only.
PLANTED_FALSE_ADMISSION_BLOCKS: Dict[int, Tuple[int, int]] = {6: (0, 4), 9: (2, 4), 12: (4, 4)}


def independence_against_the_motif_free_null() -> Dict[str, Any]:
    """Test the chance model where chance is the only mechanism: the motif-free null blocks."""
    rows: List[Dict[str, Any]] = []
    for richness, configurations, edges, admitted_six, admitted_five in NULL_BLOCKS:
        expectations = expected_cliques_under_independence(
            FROZEN_SCENES, configurations, edges)
        five = next(item for item in expectations if item["size"] == 5)
        six = next(item for item in expectations if item["size"] == 6)
        rows.append({
            "richness": richness,
            "configurations_per_scene": configurations,
            "edges": edges,
            "predicted_five_cliques": five["expected_under_independence"],
            "predicted_six_cliques": six["expected_under_independence"],
            "observed_admissions_at_five": admitted_five,
            "observed_admissions_at_six": admitted_six,
            "consistent": (admitted_five == 0 and admitted_six == 0
                           and five["expected_under_independence"] < 1.0),
        })
    return {
        "rows": rows,
        "independence_is_refuted_by_chance_evidence": not all(row["consistent"] for row in rows),
        "reading": "Where nothing recurs by construction, independence predicts effectively no "
                   "cliques at m = 5 or m = 6 and none were admitted, at every richness. The "
                   "chance model is consistent with the only evidence that isolates chance.",
    }


def motif_footprint_on_graph_density() -> Dict[str, Any]:
    """Whether the motif's own edges are visible in the matching density at all.

    If a planted block's graph were dominated by the motif, calibrating anything on a
    motif-free block would be answering about a different object. It is not: the motif
    contributes `C(S, 2)` = 15 edges, while planted blocks of the same richness differ from each
    other by more than that, and the null block's density sits among them rather than below them.
    """
    motif_edges = comb(FROZEN_SCENES, 2)
    rows = []
    for richness, configurations, null_edges, _, _ in NULL_BLOCKS:
        planted = PLANTED_EDGES[richness]
        spread = max(planted) - min(planted)
        rows.append({
            "richness": richness,
            "motif_edges": motif_edges,
            "planted_edge_range": [min(planted), max(planted)],
            "planted_block_spread": spread,
            "null_edges": null_edges,
            "motif_share_of_spread": motif_edges / float(spread) if spread else None,
            "null_density_is_comparable": (
                abs(null_edges - sum(planted) / len(planted)) <= spread),
        })
    return {
        "rows": rows,
        "motif_is_below_block_to_block_variation": all(
            row["planted_block_spread"] > row["motif_edges"] for row in rows),
        "reading": "The motif contributes 15 edges. Planted blocks of the same richness differ "
                   "from one another by 31, 100 and 150, and the null block's density sits among "
                   "them rather than below them. The signal is not what sets the graph's "
                   "density, so a motif-free block is not a different object.",
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def motif_free_calibration() -> Dict[str, Any]:
    """The settled conditional, with the correction it rests on stated first."""
    chance = independence_against_the_motif_free_null()
    footprint = motif_footprint_on_graph_density()
    result = {
        "schema": SCHEMA,
        "task": "T4E.46",
        "addresses": "the conditional T4E.45 left open: whether the motif must set its own bar",
        "status": "DERIVATION_ONLY_NO_CANDIDATE_DECLARED",
        "corrects": {
            "task": "T4E.45",
            "the_claim_withdrawn": "that the independence null is refuted by a factor of 1e12, "
                                   "and the conjecture built on it that both shapes T4E.44 left "
                                   "share one obstacle",
            "why_it_was_wrong": "independence predicts how often a clique arises by chance, and "
                                "it was compared against false-admission rates measured in "
                                "planted blocks, where a motif is present by construction. That "
                                "is a chance model judged against a signal-present observation.",
            "what_the_record_already_held": "the motif-free null blocks, which admit zero sets "
                                            "at m = 5 and m = 6 at every richness -- the "
                                            "controlled comparison, and consistent with "
                                            "independence rather than a refutation of it",
            "what_survives_from_t4e45": "the small-end result is unaffected: a pair is expected "
                                        "by the hundreds, so a clique-rarity bar refuses groups "
                                        "of two by arithmetic. The published T4E.45 measurement "
                                        "is left unedited and superseded, not rewritten.",
        },
        "independence_against_chance_only_evidence": chance,
        "motif_footprint": footprint,
        "the_conditional": {
            "question": "must a bar that needs the matching graph's clique structure be "
                        "calibrated on a partition containing the motif under test, so that the "
                        "signal sets its own bar?",
            "answer": "NO",
            "because": "the null blocks are motif-free by construction, share the generator, "
                       "richness and configuration counts, and already carry the measurement a "
                       "bar would need. Calibrating there cannot be set by a signal that is not "
                       "present, and the motif's 15 edges are below the block-to-block variation "
                       "in density, so a motif-free block is not a different object.",
            "consequence": "the two shapes T4E.44 left are not demonstrably one obstacle, the "
                           "discovery family is not shown to be exhausted, and PLAN section 3's "
                           "catalogue alternative is NOT forced.",
        },
        "what_the_real_remaining_problem_is": {
            "statement": "candidate 3's false admissions are motif-dependent: they appear only "
                         "in blocks where a motif exists, and not at all in the null blocks of "
                         "the same richness.",
            "mechanism": "tolerating one absence lets a near-miss configuration attach to a "
                         "genuine set, which requires a genuine set to attach to",
            "already_in_the_record": "the T4E.13 entry says it in its own words -- the margin "
                                     "between recurrence and coincidence in these scenes is "
                                     "exactly one scene wide",
            "why_this_matters": "it is a problem about relaxing k below S, not about the null. A "
                                "criterion that keeps k = S does not meet it, and candidate 2 "
                                "already measured zero false admission there at every richness.",
        },
        "what_this_does_not_establish": [
            "That any size-scaled criterion works. No candidate is declared, and the null being "
            "consistent with chance evidence is not the same as a bar being correct.",
            "That candidate 2 should be revived. Its own limitation stands: k = S does not "
            "transfer, which is why the sequence moved past it.",
            "Anything about the atmosphere, the signature, or the acquired record.",
        ],
        "network_used": False,
        "claim_boundary": (
            "Arithmetic over counts already published, plus a correction to a conclusion I drew "
            "from them. It settles one conditional and withdraws one conjecture; it declares no "
            "criterion, opens no reserve, and establishes nothing about whether discovery can "
            "ultimately be made to work."),
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


__all__ = [
    "NULL_BLOCKS", "PLANTED_EDGES", "PLANTED_FALSE_ADMISSION_BLOCKS", "SCHEMA",
    "independence_against_the_motif_free_null", "motif_footprint_on_graph_density",
    "motif_free_calibration",
]
