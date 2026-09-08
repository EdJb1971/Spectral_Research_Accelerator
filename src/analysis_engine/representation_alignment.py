"""Mutual k-nearest-neighbour alignment between two kernels, measured against a null.

The metric is the one used by the Platonic Representation Hypothesis (arXiv:2405.07987):
a representation is reduced to a kernel over datapoints, and two representations are compared
by the mean intersection of the k-nearest-neighbour sets the two kernels induce, normalised
by k. That paper reports a cross-modal value of 0.16 on a metric whose maximum is 1, and asks
in its own limitations section whether 0.16 is strong alignment or poor alignment. It does not
answer that question, and nothing in it establishes a reference against which 0.16 could be
read: the words `baseline`, `chance`, `null`, `shuffled` and `surrogate` do not occur in the
paper, and there is no significance test anywhere in it.

This module supplies the two references that question needs, and keeps them distinct because
they are not equally strong:

`chance_alignment` is closed form. Two *independent uniform* k-subsets of the n-1 other
points intersect in k^2/(n-1) points on average, so the normalised metric has expectation
k/(n-1). It costs nothing and depends only on the shape of the problem. Measured against
permutation nulls over kernels with strong clusters, heavy duplication and none of either,
it estimates the null *mean* to within about half a percent in each case, so it is a
reliable quick reference and not merely an idealisation. What it does not give is a spread,
and a floor without a spread cannot set a threshold.

`permutation_null` supplies that, and is the reference to run on real data. It permutes the
correspondence between the two point sets -- the shuffled-caption control the paper never
runs -- and recomputes the metric. Both kernels keep their own internal geometry and only the
pairing is destroyed, so the excess is attributable to correspondence. It also catches what
the closed form cannot see: pairing structure that is real but not semantic. If two point
sets are both ordered by topic, unrelated representations of them share that ordering, and
only a permutation knows to destroy it. This is D98's lesson for a different subject -- a
null must destroy the thing under test and nothing else.

No claim about convergence, reality or semantics lives here. Structural comparison never
licenses semantic comparison (R19); this module compares distance structures and says so.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from src.core.errors import InvalidParameterError


#: What this measurement is and is not, carried with every report.
ALIGNMENT_BOUNDARY = (
    "Mutual k-NN agreement between two kernels over a fixed, ordered point set. "
    "It compares how two representations measure distance; it licenses no claim about "
    "what either represents.")


def _kernel(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InvalidParameterError(name, getattr(matrix, "shape", None),
                                    "a square (n, n) kernel over one ordered point set")
    if matrix.shape[0] < 3:
        raise InvalidParameterError(name, matrix.shape[0],
                                    "at least three points; a k-NN set needs somewhere to look")
    if not np.all(np.isfinite(matrix)):
        raise InvalidParameterError(name, "non-finite",
                                    "finite similarities; an absent similarity is not a distant one")
    return matrix


def _neighbour_budget(k: int, n: int) -> int:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise InvalidParameterError("k", k, "an integer neighbour count")
    if not 1 <= k <= n - 1:
        raise InvalidParameterError(
            "k", k, "a neighbour count in [1, n-1] for n=%d points; k=n-1 makes every point "
                    "everyone's neighbour and the metric identically 1" % n)
    return int(k)


def knn_sets(kernel: Any, k: int) -> np.ndarray:
    """Indices of each point's k nearest neighbours under a similarity kernel, self excluded.

    Higher kernel values are nearer. Ties are broken by ascending index, deterministically,
    so that two callers of this function on the same kernel always get the same sets.
    """
    matrix = _kernel(kernel, "kernel")
    n = matrix.shape[0]
    budget = _neighbour_budget(k, n)
    scored = matrix.copy()
    np.fill_diagonal(scored, -np.inf)
    # Sort by (-similarity, index): argsort on the negated matrix is stable, so equal
    # similarities resolve to the lower index rather than to whatever the partition left.
    order = np.argsort(-scored, axis=1, kind="stable")
    return order[:, :budget]


def mutual_knn_alignment(left: Any, right: Any, k: int) -> float:
    """Mean intersection of the two kernels' k-NN sets, normalised by k. In [0, 1]."""
    first, second = _kernel(left, "left"), _kernel(right, "right")
    if first.shape != second.shape:
        raise InvalidParameterError(
            "right", second.shape,
            "the same shape as `left` (%s); the metric compares two kernels over one ordered "
            "point set, and a different point set is not a different representation of it"
            % (first.shape,))
    budget = _neighbour_budget(k, first.shape[0])
    a, b = knn_sets(first, budget), knn_sets(second, budget)
    membership = np.zeros((first.shape[0], first.shape[0]), dtype=bool)
    np.put_along_axis(membership, b, True, axis=1)
    return float(np.mean(np.take_along_axis(membership, a, axis=1).sum(axis=1) / budget))


def chance_alignment(n: int, k: int) -> float:
    """Expected alignment of two independent uniform k-NN assignments: k/(n-1).

    It assumes both neighbour sets are uniform over the other n-1 points. Kernels are not
    uniform, but under a *uniform permutation* the permuted neighbour sets land uniformly
    regardless, so this estimates the permutation null's mean closely -- measured to under
    one percent against strongly clustered and heavily duplicated kernels alike.

    It is still not a threshold. It gives a location and no spread, and it cannot see pairing
    structure that is real but not semantic. Run `permutation_null` before reading a result.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 3:
        raise InvalidParameterError("n", n, "at least three points")
    return _neighbour_budget(k, int(n)) / (int(n) - 1)


def permutation_null(left: Any, right: Any, *, k: int, permutations: int = 200,
                     seed: int = 0) -> Dict[str, Any]:
    """Measured alignment against a null that destroys only the correspondence.

    Each draw relabels the right-hand point set by a random permutation and recomputes the
    metric. Both kernels keep their internal geometry, so the null answers exactly the
    question the measurement raises: how aligned would these two representations look if
    their points were not paired?

    Reports the measured value, the null distribution's location and spread, and the
    empirical exceedance -- the fraction of null draws reaching the measured value, which is
    a permutation p-value over the pairing and over nothing else. It is not a claim about
    convergence, and a permutation test on one point set is not a claim about other data.
    """
    first, second = _kernel(left, "left"), _kernel(right, "right")
    if first.shape != second.shape:
        raise InvalidParameterError("right", second.shape,
                                    "the same shape as `left` (%s)" % (first.shape,))
    n = first.shape[0]
    budget = _neighbour_budget(k, n)
    if not isinstance(permutations, (int, np.integer)) or isinstance(permutations, bool) \
            or permutations < 1:
        raise InvalidParameterError("permutations", permutations,
                                    "at least one permutation; an unmeasured null is not a null")

    measured = mutual_knn_alignment(first, second, budget)
    rng = np.random.default_rng(seed)
    draws = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        order = rng.permutation(n)
        draws[index] = mutual_knn_alignment(first, second[np.ix_(order, order)], budget)

    # Add-one so a p-value is never reported as exactly zero on a finite null.
    exceedance = float((np.count_nonzero(draws >= measured) + 1) / (permutations + 1))
    return {
        "measured_alignment": measured,
        "k": budget,
        "n_points": n,
        "permutations": int(permutations),
        "null_mean": float(np.mean(draws)),
        "null_std": float(np.std(draws, ddof=1)) if permutations > 1 else None,
        "null_max": float(np.max(draws)),
        "null_quantile_95": float(np.quantile(draws, 0.95)),
        "excess_over_null_mean": measured - float(np.mean(draws)),
        "permutation_exceedance": exceedance,
        "analytic_chance_alignment": chance_alignment(n, budget),
        "boundary": ALIGNMENT_BOUNDARY,
        "null_boundary": ("Permuting the pairing preserves each kernel's own geometry and "
                          "destroys only the correspondence, so the excess is attributable "
                          "to pairing. It is not evidence about any other point set."),
    }


def describe_measurement(report: Dict[str, Any], *,
                         label: Optional[str] = None) -> Dict[str, Any]:
    """A reading of an alignment report that states what it does and does not support.

    Deliberately returns `unresolved` rather than a verdict when the measured value sits
    inside the null. A number that a shuffled pairing reproduces is not a finding.
    """
    measured = report["measured_alignment"]
    if not math.isfinite(measured):
        raise InvalidParameterError("report.measured_alignment", measured, "a finite alignment")
    inside = measured <= report["null_quantile_95"]
    return {
        "label": label,
        "outcome": "unresolved" if inside else "alignment_exceeds_permuted_pairing",
        "measured_alignment": measured,
        "null_mean": report["null_mean"],
        "permutation_exceedance": report["permutation_exceedance"],
        "reading": (
            "The measured alignment is within the range a permuted pairing reaches, so this "
            "point set does not distinguish the two representations from unpaired ones."
            if inside else
            "The measured alignment exceeds the 95th percentile of permuted pairings on this "
            "point set. That is evidence about correspondence, not about convergence."),
        "boundary": ALIGNMENT_BOUNDARY,
    }
