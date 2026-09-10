"""Validity gates for synthetic evidence, written so the code IS the declaration.

T4E.20 declared that a synthetic background yielding "hundreds, or none" of features cannot
stand in for the record, and its gate tested only the upper bound. A background yielding none
passed, the slice ran on a field where a planted vortex faced no competition, and the
quantitative comparison it wanted to make was forbidden after the fact rather than before it.

The lesson is not "check both bounds". It is that a gate expressed only in prose is not a gate,
and that the case which slipped through is the one a test must pin. Both are here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

#: T4E.21's declared band, from the real record under identical raw extraction: a median of 7
#: features per frame with a range of 0 to 13. The band is deliberately wider than the record's
#: median alone -- it exists to reject a background that is empty or swarming, not to demand the
#: record's exact distribution, which N Gaussians on a randomised spectrum will never reproduce.
DECLARED_MEDIAN_BAND: Tuple[int, int] = (4, 10)
DECLARED_FRAME_CEILING: int = 40


def feature_density_gate(counts: Sequence[int],
                         *, median_band: Tuple[int, int] = DECLARED_MEDIAN_BAND,
                         frame_ceiling: int = DECLARED_FRAME_CEILING) -> Dict[str, object]:
    """Does this synthetic evidence carry features at the density the record carries?

    Three conditions, all declared and all checked here rather than in prose:

      * the median count lies inside the declared band,
      * no single frame exceeds the ceiling, and
      * the median is strictly greater than zero.

    The third is redundant against a band starting at 4 and is kept anyway, because it is the
    condition T4E.20 declared and did not implement, and a redundant check that names the
    failure it was written for is worth more than a tidy one that does not.
    """
    values = [int(value) for value in counts]
    if not values:
        return {"passed": False, "reasons": ["no frames were measured, so nothing was checked"],
                "median": None, "maximum": None, "n_frames": 0}
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    maximum = max(ordered)
    low, high = int(median_band[0]), int(median_band[1])
    reasons: List[str] = []
    if median <= 0:
        reasons.append(
            "the median frame yields no features at all, so a planted feature would face no "
            "competition and the synthetic problem would be easier than the real one. This is "
            "the case T4E.20's gate declared and did not implement")
    if not low <= median <= high:
        reasons.append(
            "median %d features per frame is outside the declared band [%d, %d] taken from the "
            "record" % (median, low, high))
    if maximum > int(frame_ceiling):
        reasons.append(
            "a frame yields %d features against a ceiling of %d; a swarming background is as "
            "unrepresentative as an empty one" % (maximum, int(frame_ceiling)))
    return {"passed": not reasons, "reasons": reasons, "median": median,
            "maximum": maximum, "n_frames": len(values)}
