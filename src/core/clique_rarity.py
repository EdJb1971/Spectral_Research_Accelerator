"""T4E.45: the extremes test for a clique-rarity bar, before one is declared.

T4E.44 left two shapes for a size-scaled criterion and the recommendation was Shape B: take the
scaling from the combinatorics of the matching graph rather than from a surrogate null, on the
grounds that its behaviour at the extremes could be derived on paper before adoption. This module
performs that derivation, on the frozen S = 6 structure, using matching densities already recorded
by candidates 2 and 3.

It settles the question in both directions, and one of them is against the recommendation. At the
small end the shape works and derives the refusal of pairs that candidate 4 had to be patched for.
At m = 5 the independence null that makes the arithmetic tractable is refuted by measurements
already on disk, by ten to fifteen orders of magnitude, in the direction that admits coincidences.

Nothing here touches a signature, a partition or a distance. It is arithmetic over counts the
record already published, and it declares no criterion.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from typing import Any, Dict, List, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.size_scaled_null import FROZEN_SCENES


SCHEMA = "t4e45-clique-rarity-derivation/v1"

#: Matching densities recorded by candidate 2 and candidate 3 on the development blocks. Each row
#: is (richness, configurations per scene, pairs proposed by the matching), taken from
#: `measurements/t4e12_candidate_2.json` and `measurements/t4e13_candidate_3.json`.
RECORDED_MATCHINGS: Tuple[Tuple[int, int, int], ...] = (
    (6, 20, 121), (6, 20, 152), (6, 20, 136),
    (9, 84, 568), (9, 84, 485), (9, 84, 585), (9, 84, 566),
    (12, 220, 1446), (12, 220, 1440), (12, 220, 1590), (12, 220, 1488),
)

#: What candidate 3 measured at k = S - 1 = 5: blocks whose false admission exceeded zero, which
#: is to say blocks where at least one coincidental five-clique existed. Four blocks per richness.
COINCIDENTAL_FIVE_CLIQUES_OBSERVED: Dict[int, Tuple[int, int]] = {
    6: (0, 4),
    9: (2, 4),
    12: (4, 4),
}


def edge_probability(scenes: int, configurations_per_scene: int, edges: int) -> float:
    """The matching's density over the cross-scene pairs it could have proposed."""
    if scenes < 2 or configurations_per_scene < 1 or edges < 0:
        raise InvalidParameterError(
            "matching", (scenes, configurations_per_scene, edges),
            "at least two scenes, one configuration per scene and a non-negative edge count")
    candidate_pairs = comb(scenes, 2) * configurations_per_scene * configurations_per_scene
    return edges / float(candidate_pairs)


def expected_cliques_under_independence(
    scenes: int, configurations_per_scene: int, edges: int,
) -> List[Dict[str, Any]]:
    """Expected number of one-per-scene cliques of each size, if edges were independent.

    A set of size `m` picks `m` scenes and one configuration from each, so there are
    `C(S, m) * c**m` of them, and a clique needs all `C(m, 2)` of its pairs to be edges. Under
    independence that gives `E_m = C(S, m) * c**m * p**C(m, 2)`.

    The size scaling is the point and it needs no parameter: the count grows as `c**m` while the
    probability falls as `p**(m(m-1)/2)`, so the expectation collapses super-exponentially. That
    is a bar that changes with the group because of what a group of that size is, not because a
    threshold was tuned.
    """
    p = edge_probability(scenes, configurations_per_scene, edges)
    rows = []
    for size in range(2, scenes + 1):
        sets = comb(scenes, size) * configurations_per_scene ** size
        rows.append({
            "size": size,
            "one_per_scene_sets": sets,
            "pairs_required": comb(size, 2),
            "expected_under_independence": sets * p ** comb(size, 2),
        })
    return rows


def small_end_behaviour(matchings: Sequence[Tuple[int, int, int]] = RECORDED_MATCHINGS,
                        ) -> Dict[str, Any]:
    """What the bar does at the smallest group it admits -- the test candidate 4 failed.

    Candidate 4's closure rule was vacuous for a group of two and had to be refused after the
    fact. A clique-rarity bar refuses pairs by arithmetic instead: at `m = 2` the expectation is
    the edge count itself, so hundreds of pairs are expected by chance and no pair can clear any
    bar worth stating. The rule does not need a special case for its own smallest group.
    """
    rows = []
    for richness, configurations, edges in matchings:
        expectations = expected_cliques_under_independence(
            FROZEN_SCENES, configurations, edges)
        pair = next(item for item in expectations if item["size"] == 2)
        rows.append({
            "richness": richness,
            "configurations_per_scene": configurations,
            "edges": edges,
            "expected_pairs_by_chance": pair["expected_under_independence"],
            "a_pair_is_evidence": pair["expected_under_independence"] < 1.0,
        })
    return {
        "rows": rows,
        "every_pair_expected_by_chance": all(not row["a_pair_is_evidence"] for row in rows),
        "reading": "At m = 2 the expectation equals the observed edge count, so a pair is "
                   "expected by the hundreds and cannot be evidence under any bar. Candidate 4's "
                   "defect -- a group of two admitted on the same terms as a group of six -- is "
                   "excluded by the arithmetic rather than by a special case.",
    }


def independence_refutation(
    matchings: Sequence[Tuple[int, int, int]] = RECORDED_MATCHINGS,
    observed: Dict[int, Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """The large end, where the derivation goes against the recommendation that prompted it.

    Independence makes the arithmetic above tractable. Candidate 3 measured what actually happens
    at `m = 5`: coincidental five-cliques in half the development blocks, and up to 0.8 false
    admission at richness 12. Independence predicts they are effectively impossible. The gap is
    not a discrepancy to be calibrated away -- it is the assumption failing, in the direction that
    admits coincidences as recurrences.

    The mechanism is not mysterious. A matching graph built from nearest neighbours is transitive
    by construction: if `a` is among `b`'s closest and `b` among `c`'s, `a` and `c` are usually
    close too. Triangles, and therefore cliques, are far more common than independent edges of the
    same density would give.
    """
    observed = dict(observed or COINCIDENTAL_FIVE_CLIQUES_OBSERVED)
    by_richness: Dict[int, List[float]] = {}
    for richness, configurations, edges in matchings:
        expectations = expected_cliques_under_independence(
            FROZEN_SCENES, configurations, edges)
        five = next(item for item in expectations if item["size"] == 5)
        by_richness.setdefault(richness, []).append(five["expected_under_independence"])

    rows = []
    for richness in sorted(by_richness):
        predicted = max(by_richness[richness])
        blocks_with, blocks_total = observed.get(richness, (0, 0))
        rows.append({
            "richness": richness,
            "predicted_five_cliques_under_independence": predicted,
            "blocks_with_a_coincidental_five_clique": blocks_with,
            "blocks_measured": blocks_total,
            "independence_understates_by_at_least": (
                (blocks_with / float(blocks_total)) / predicted
                if blocks_with and predicted > 0 else None),
        })
    refuted = any(row["blocks_with_a_coincidental_five_clique"] > 0 for row in rows)
    return {
        "rows": rows,
        "independence_is_refuted": refuted,
        "direction": "ANTI_CONSERVATIVE" if refuted else "NOT_REFUTED",
        "mechanism": "a nearest-neighbour matching is transitive by construction, so cliques are "
                     "far more common than independent edges of the same density predict",
        "source_of_the_observation": "candidate 3's recorded false admission at k = S - 1 = 5, "
                                     "which is a measurement and not a new claim",
        "reading": "The independence null is not a conservative approximation here. It understates "
                   "coincidental cliques by ten to fifteen orders of magnitude at m = 5, in the "
                   "direction that admits a coincidence as a recurrence.",
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def clique_rarity_derivation() -> Dict[str, Any]:
    """The whole extremes test, with what survives it and what does not."""
    small = small_end_behaviour()
    large = independence_refutation()
    expectations = {
        richness: expected_cliques_under_independence(FROZEN_SCENES, configurations, edges)
        for richness, configurations, edges in
        ((6, 20, 121), (9, 84, 568), (12, 220, 1446))
    }
    result = {
        "schema": SCHEMA,
        "task": "T4E.45",
        "addresses": "the extremes test for T4E.44's Shape B, before any candidate is declared",
        "status": "DERIVATION_ONLY_NO_CANDIDATE_DECLARED",
        "scenes": FROZEN_SCENES,
        "expected_cliques_under_independence": expectations,
        "small_end": small,
        "large_end": large,
        "what_survives": [
            "The shape of the bar. The count of one-per-scene sets grows as c**m while the "
            "probability of a clique falls as p**(m(m-1)/2), so the expectation collapses "
            "super-exponentially with size. That is size scaling with no free parameter.",
            "The small end. A pair is expected by the hundreds, so a clique-rarity bar refuses "
            "groups of two by arithmetic rather than by the special case candidate 4 needed.",
        ],
        "what_does_not_survive": [
            "The independence null that makes the expectation computable. Candidate 3 measured "
            "coincidental five-cliques in half the development blocks; independence predicts "
            "between 1e-10 and 1e-15 of them. The assumption fails anti-conservatively.",
            "The recommendation that prompted this derivation, in its simple form. Shape B was "
            "recommended because its extremes could be derived before adoption. They were, and "
            "they refuted it.",
        ],
        "what_the_open_question_became": {
            "was": "whether the matching graph has a chance structure estimable without a "
                   "surrogate",
            "is_now": "whether the graph's transitivity -- the quantity that makes cliques common "
                      "-- can be estimated from a partition that contains the motif whose clique "
                      "is under test, without the estimate being set by the signal",
            "why_that_is_narrower": "it names one measurable quantity and one contamination "
                                    "question, rather than an open modelling problem",
            "note": "That contamination question is K1 in another form, which is worth saying "
                    "plainly: the two shapes T4E.44 left may share a single underlying obstacle.",
        },
        "cost_of_this_derivation": "arithmetic over counts already published. No partition, "
                                   "signature or distance was touched, no reserve opened, and no "
                                   "candidate declared or adopted.",
        "network_used": False,
        "claim_boundary": (
            "This is an extremes test of a proposed shape, not a criterion and not evidence for "
            "C6. It refutes one null for one bar; it establishes nothing about the atmosphere, "
            "the signature, or whether any size-scaled criterion can work."),
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


__all__ = [
    "COINCIDENTAL_FIVE_CLIQUES_OBSERVED", "RECORDED_MATCHINGS", "SCHEMA",
    "clique_rarity_derivation", "edge_probability", "expected_cliques_under_independence",
    "independence_refutation", "small_end_behaviour",
]
