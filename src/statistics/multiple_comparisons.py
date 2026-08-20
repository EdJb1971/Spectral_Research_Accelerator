"""Multiple-comparison control (roadmap R5 / T4C.5, defect D8).

**The defect this closes, measured.** The end-to-end smoke sweep - 9 runs over
`grid_size x transform_type` - made the hypothesis engine emit **9 "discoveries"**, including
*"strong positive correlation (r = 0.96) between grid size and floating-point reconstruction
error"*. The engine reported every pair whose |r| exceeded a threshold, with no p-value at all,
let alone a correction. On 9 samples that is not discovery; it is a random-number generator with
a narrative layer.

**Why a correction is not optional here.** An automated discovery engine's whole purpose is to
test a great many hypotheses. Testing 500 independent nulls at alpha = 0.05 yields ~25 false
positives *by construction*. Any pipeline that scans and reports without controlling for
multiplicity will therefore always produce findings, on any data, including noise - which is
exactly what the null benchmarks in `src/benchmarks/` exist to detect.

**Which procedure, and the dependence assumption - stated, because it is the part that gets
hidden.**

*   ``bonferroni`` controls the family-wise error rate under *any* dependence. Valid always,
    and badly underpowered for hundreds of tests.
*   ``holm`` also controls FWER under any dependence and uniformly dominates Bonferroni. Use it
    when a single false positive is unacceptable.
*   ``benjamini_hochberg`` controls the *false discovery rate* under independence or **positive
    regression dependence** (PRDS). This is the right default for exploratory scanning, and the
    PRDS caveat matters: spectral statistics across neighbouring scales are positively
    correlated, which PRDS covers, but a mixture of positively and negatively correlated tests
    is not guaranteed.
*   ``benjamini_yekutieli`` controls FDR under **arbitrary** dependence, at the cost of a
    ``sum(1/i)`` penalty (~7.5x for 1000 tests). This is the defendable choice when the
    dependence structure is unknown, which for cross-scale spectral tests it usually is.

Every function returns the procedure name and its assumption in the result, so a reported
q-value can never be separated from the condition under which it is valid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROCEDURES = ("bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli")

ASSUMPTIONS = {
    "bonferroni": "controls FWER under arbitrary dependence",
    "holm": "controls FWER under arbitrary dependence; dominates Bonferroni",
    "benjamini_hochberg": ("controls FDR under independence or positive regression "
                           "dependence (PRDS)"),
    "benjamini_yekutieli": "controls FDR under arbitrary dependence",
}


class MultipleComparisonError(ValueError):
    """Raised when a correction cannot be computed honestly from what was supplied."""


def _validate(p_values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(p_values), dtype=np.float64)
    if arr.size == 0:
        raise MultipleComparisonError(
            "no p-values supplied. A correction over an empty family is not 'nothing to "
            "correct' - it usually means the scan silently produced no tests, which is a "
            "different problem worth surfacing.")
    if np.any(~np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise MultipleComparisonError(
            "%d of %d p-values are NaN or infinite. Dropping them silently would shrink the "
            "family size and inflate significance; decide explicitly whether those tests "
            "were performed." % (bad, arr.size))
    if np.any(arr < 0) or np.any(arr > 1):
        raise MultipleComparisonError(
            "p-values must lie in [0, 1]; got range [%.4g, %.4g]" % (arr.min(), arr.max()))
    return arr


def adjust(
    p_values: Sequence[float],
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
    n_tests: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Correct a family of p-values for multiplicity.

    Parameters
    ----------
    n_tests
        The size of the family, if larger than ``len(p_values)``. **This is the parameter
        that keeps a scan honest.** If a pipeline computes 500 tests, discards the 480 that
        looked uninteresting, and corrects only the surviving 20, the correction is
        meaningless - the selection already used the data. Pass the true number of tests
        performed.
    labels
        Optional names, carried through so a corrected result can be traced to its test.

    Returns adjusted p-values (q-values for the FDR procedures) in the input order, the
    rejection mask at ``alpha``, and the procedure's dependence assumption.
    """
    if method not in PROCEDURES:
        raise MultipleComparisonError(
            "unknown method %r; expected one of %s" % (method, ", ".join(PROCEDURES)))
    if not 0 < alpha < 1:
        raise MultipleComparisonError("alpha must lie in (0, 1); got %r" % (alpha,))

    p = _validate(p_values)
    m_observed = p.size
    m = int(n_tests) if n_tests is not None else m_observed
    if m < m_observed:
        raise MultipleComparisonError(
            "n_tests=%d is smaller than the %d p-values supplied. The family cannot be "
            "smaller than what it contains." % (m, m_observed))

    warnings: List[str] = []
    if m > m_observed:
        warnings.append(
            "correcting %d p-values as part of a family of %d. The %d absent tests are "
            "assumed to have been performed and not reported; if they were never performed, "
            "n_tests is wrong and the correction is too conservative."
            % (m_observed, m, m - m_observed))

    order = np.argsort(p, kind="stable")
    ranked = p[order]

    if method == "bonferroni":
        adjusted_sorted = np.minimum(ranked * m, 1.0)

    elif method == "holm":
        # Step-down: multiply by (m - i), then enforce monotonicity forwards.
        factors = m - np.arange(m_observed)
        adjusted_sorted = np.minimum.accumulate(
            np.maximum.accumulate(ranked * factors))[::1]
        adjusted_sorted = np.maximum.accumulate(np.minimum(ranked * factors, 1.0))

    else:
        # Step-up (BH / BY): multiply by m/i, then enforce monotonicity *backwards*, which
        # is what makes the adjusted values non-decreasing in the original p-order. Getting
        # the direction of this cumulative minimum wrong is the classic BH implementation
        # bug and it silently makes results anti-conservative.
        ranks = np.arange(1, m_observed + 1)
        penalty = 1.0
        if method == "benjamini_yekutieli":
            penalty = float(np.sum(1.0 / np.arange(1, m + 1)))
        scaled = ranked * m * penalty / ranks
        adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha

    result: Dict[str, Any] = {
        "method": method,
        "assumption": ASSUMPTIONS[method],
        "alpha": alpha,
        "n_tests": m,
        "n_p_values": m_observed,
        "adjusted": [float(v) for v in adjusted],
        "rejected": [bool(v) for v in rejected],
        "n_rejected": int(np.sum(rejected)),
        "min_adjusted": float(np.min(adjusted)),
        "warnings": warnings,
    }
    if method == "benjamini_yekutieli":
        result["dependence_penalty"] = float(np.sum(1.0 / np.arange(1, m + 1)))
    if labels is not None:
        if len(labels) != m_observed:
            raise MultipleComparisonError(
                "labels has %d entries but %d p-values were supplied"
                % (len(labels), m_observed))
        result["labels"] = list(labels)
        result["rejected_labels"] = [l for l, r in zip(labels, rejected) if r]
    return result


def resolution_limit(n_surrogates: int) -> float:
    """Smallest p-value a surrogate test with ``n_surrogates`` draws can *ever* produce.

    A surrogate p-value is ``(1 + k) / (1 + n)``, so its floor is ``1 / (1 + n)``. With 99
    surrogates nothing below 0.01 is representable.
    """
    return 1.0 / (1.0 + max(0, int(n_surrogates)))


def required_surrogates(n_tests: int, alpha: float = 0.05,
                        method: str = "benjamini_hochberg") -> int:
    """How many surrogates a family of ``n_tests`` needs to be *capable* of a rejection.

    **The trap this exists to prevent.** After correcting for 500 tests, a significant result
    needs a raw p-value around ``alpha / n_tests``. A surrogate test with 99 draws cannot
    produce a p-value below 0.01, so such a study is arithmetically incapable of finding
    anything - and will duly report nothing, which looks like a clean negative result rather
    than a broken instrument. Worse, it is indistinguishable from one.
    """
    if method in ("bonferroni", "holm"):
        needed_p = alpha / max(1, n_tests)
    elif method == "benjamini_yekutieli":
        penalty = float(np.sum(1.0 / np.arange(1, max(1, n_tests) + 1)))
        needed_p = alpha / (max(1, n_tests) * penalty)
    else:
        # BH: the most permissive threshold is alpha at rank m, i.e. the smallest p needs
        # only beat alpha/m in the worst case (rank 1).
        needed_p = alpha / max(1, n_tests)
    return int(math.ceil(1.0 / needed_p)) - 1


def check_power(n_surrogates: int, n_tests: int, alpha: float = 0.05,
                method: str = "benjamini_hochberg") -> Dict[str, Any]:
    """Report whether a surrogate study *can* reject anything after correction."""
    floor = resolution_limit(n_surrogates)
    needed = required_surrogates(n_tests, alpha, method)
    capable = n_surrogates >= needed
    return {
        "n_surrogates": n_surrogates,
        "n_tests": n_tests,
        "p_value_floor": floor,
        "surrogates_required": needed,
        "can_reject_after_correction": capable,
        "warning": None if capable else (
            "This study cannot produce a significant result. %d surrogates give a p-value "
            "floor of %.4g, but rejecting one of %d tests under %s at alpha=%.3g needs a raw "
            "p below roughly %.4g. Increase surrogates to at least %d, or reduce the number "
            "of tests. A null result from this configuration says nothing about the data."
            % (n_surrogates, floor, n_tests, method, alpha, alpha / max(1, n_tests), needed)),
    }
