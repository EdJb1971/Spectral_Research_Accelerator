"""T4E.10: choosing an identity radius that still holds on data it has not seen.

T4E.9 measured the failure with the atmosphere out of the way. On synthetic scenes whose motif
is known by construction, with AUC 1.0 and zero false admissions in 11,985 pairs, a radius
calibrated at 90% recall on 15 cross-scene motif pairs and frozen still split **46.7%** of motif
pairs in a partition it had never seen. Perfect labels, total separation, and the operating
point still did not transfer. So the estimator is the defect, not the record.

**The estimator was answering the wrong question.** `recall_radius` is an empirical quantile:
r90 of 15 samples is its 14th smallest value. That estimates where the population's 90th
percentile *is*. What the acceptance needs is a bound that will still admit 90% of pairs it has
never seen, which is a one-sided nonparametric tolerance bound. The two objects differ, and the
second is strictly wider -- an empirical quantile is below the true one about half the time, and
every one of those times the split rate on fresh data exceeds its declared bound.

**And the support can be insufficient before any radius is computed.** For the k-th smallest of
n samples the covered proportion is distributed `Beta(k, n-k+1)`, so a bound covering proportion
p with confidence gamma exists only where some k satisfies `P(Beta(k, n-k+1) >= p) >= gamma`.
Taking the maximum gives `1 - p**n >= gamma`, so `n >= log(1-gamma)/log(p)`. At p = gamma = 0.9
that is **n >= 22**, and T4E.9 had 15. No choice of order statistic repairs that, and neither
does a wider quantile: the guarantee was unavailable before the data was looked at.

Estimators are registered rather than selected by a branch, each declaring what it guarantees
and what it needs. A refusal is a first-class outcome here: an estimator that cannot support its
declared coverage returns a refusal naming the support that would, because a number issued
without its guarantee is exactly what T4E.9 caught.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import beta as _beta

from src.core.errors import InvalidParameterError
from src.core.registry import Registry


@dataclass(frozen=True)
class OperatingPoint:
    """A radius, or a refusal to name one, with the guarantee it carries either way."""

    radius: Optional[float]
    estimator: str
    guarantee: str
    support: int
    refused: bool = False
    refusal: Optional[str] = None
    detail: Dict[str, Any] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "estimator": self.estimator, "radius": self.radius, "support": self.support,
            "guarantee": self.guarantee, "refused": self.refused, "refusal": self.refusal,
            **(self.detail or {}),
        }


@dataclass(frozen=True)
class Estimator:
    """A declared way of turning observed same-configuration distances into a radius."""

    fn: Callable[..., OperatingPoint]
    guarantees: str
    refuses_on_insufficient_support: bool


ESTIMATORS: Registry[Estimator] = Registry("identity operating-point estimator")


def _positives(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise InvalidParameterError("operating_point.distances", "invalid",
                                    "a one-dimensional sequence of finite nonnegative distances")
    return np.sort(result)


def _bounds(coverage: float, confidence: float) -> None:
    for name, value in (("coverage", coverage), ("confidence", confidence)):
        if not math.isfinite(value) or not 0 < value < 1:
            raise InvalidParameterError("operating_point." + name, value, "a value in (0, 1)")


def required_support(coverage: float, confidence: float) -> int:
    """Smallest n for which *any* order statistic carries the declared coverage/confidence.

    From `1 - coverage**n >= confidence`, which is the maximum's guarantee and therefore the
    weakest requirement any order statistic can meet.
    """
    _bounds(coverage, confidence)
    return int(math.ceil(math.log1p(-confidence) / math.log(coverage)))


def _order_statistic(support: int, coverage: float, confidence: float) -> Optional[int]:
    """Largest usable k (1-indexed, smallest first), or None when the support cannot carry it."""
    for k in range(support, 0, -1):
        if 1.0 - float(_beta.cdf(coverage, k, support - k + 1)) >= confidence:
            return k
    return None


def empirical_quantile(distances: Sequence[float], *, coverage: float = 0.9,
                       confidence: float = 0.9) -> OperatingPoint:
    """The present behaviour, kept so that its failure stays measurable rather than remembered.

    `confidence` is accepted and deliberately ignored: an empirical quantile carries no
    confidence statement, and the guarantee string says so rather than implying one.
    """
    _bounds(coverage, confidence)
    values = _positives(distances)
    if not len(values):
        return OperatingPoint(None, "empirical_quantile", _EMPIRICAL_GUARANTEE, 0, True,
                              "no observations; an unmeasured population has no quantile")
    index = math.ceil(coverage * len(values)) - 1
    return OperatingPoint(
        float(values[index]), "empirical_quantile", _EMPIRICAL_GUARANTEE, len(values),
        detail={"order_statistic": index + 1,
                "confidence_requested_and_ignored": confidence})


def nonparametric_tolerance_bound(distances: Sequence[float], *, coverage: float = 0.9,
                                  confidence: float = 0.9) -> OperatingPoint:
    """A bound covering `coverage` of unseen pairs with `confidence`, or a refusal naming n.

    The refusal is the point. Where the support cannot carry the declared guarantee, no radius
    exists that carries it, and returning one anyway is what T4E.9 caught happening.
    """
    _bounds(coverage, confidence)
    values = _positives(distances)
    needed = required_support(coverage, confidence)
    if len(values) < needed:
        return OperatingPoint(
            None, "nonparametric_tolerance_bound", _TOLERANCE_GUARANTEE, len(values), True,
            "support of %d cannot carry coverage %.3f at confidence %.3f; %d observations are "
            "required and no order statistic of %d supplies it. A radius issued here would "
            "carry no guarantee at all." % (len(values), coverage, confidence, needed,
                                            len(values)),
            {"required_support": needed})
    k = _order_statistic(len(values), coverage, confidence)
    if k is None:  # pragma: no cover - unreachable while `needed` is the maximum's requirement
        return OperatingPoint(
            None, "nonparametric_tolerance_bound", _TOLERANCE_GUARANTEE, len(values), True,
            "no order statistic of %d carries the declared guarantee" % len(values),
            {"required_support": needed})
    return OperatingPoint(
        float(values[k - 1]), "nonparametric_tolerance_bound", _TOLERANCE_GUARANTEE, len(values),
        detail={"order_statistic": k, "required_support": needed,
                "achieved_confidence": float(1.0 - _beta.cdf(coverage, k, len(values) - k + 1))})


def bootstrap_upper(distances: Sequence[float], *, coverage: float = 0.9,
                    confidence: float = 0.9, resamples: int = 2000,
                    seed: int = 0) -> OperatingPoint:
    """The upper confidence limit of the coverage-quantile, by percentile bootstrap.

    Offered so that refusing is a choice rather than the only option: this widens as the
    support falls instead of declining to answer. It is the weaker guarantee, and its
    guarantee string says so -- the bootstrap is asymptotic, and at the supports this
    programme actually has, that is precisely where it is least trustworthy.
    """
    _bounds(coverage, confidence)
    values = _positives(distances)
    if len(values) < 2:
        return OperatingPoint(None, "bootstrap_upper", _BOOTSTRAP_GUARANTEE, len(values), True,
                              "fewer than two observations; a resample would be the sample")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(resamples), len(values)), replace=True)
    quantiles = np.quantile(draws, coverage, axis=1)
    return OperatingPoint(
        float(np.quantile(quantiles, confidence)), "bootstrap_upper", _BOOTSTRAP_GUARANTEE,
        len(values), detail={"resamples": int(resamples), "seed": int(seed)})


_EMPIRICAL_GUARANTEE = (
    "None. This is the sample's own coverage-quantile and carries no statement about pairs "
    "not in the sample; it is below the population quantile about half the time.")
_TOLERANCE_GUARANTEE = (
    "Covers at least `coverage` of the population with probability at least `confidence`, "
    "distribution-free, provided the observations are exchangeable draws from it.")
_BOOTSTRAP_GUARANTEE = (
    "An asymptotic upper confidence limit on the coverage-quantile. Not distribution-free and "
    "not exact at small support, which is where this programme operates.")

ESTIMATORS.add("empirical_quantile",
               Estimator(empirical_quantile, _EMPIRICAL_GUARANTEE, False),
               description="T4E.8's behaviour, retained as the measurable baseline.")
ESTIMATORS.add("nonparametric_tolerance_bound",
               Estimator(nonparametric_tolerance_bound, _TOLERANCE_GUARANTEE, True),
               description="Distribution-free coverage guarantee; refuses on thin support.")
ESTIMATORS.add("bootstrap_upper",
               Estimator(bootstrap_upper, _BOOTSTRAP_GUARANTEE, False),
               description="Widens rather than refusing; asymptotic, so weakest where it is "
                           "most needed.")


def estimate(name: str, distances: Sequence[float], **kwargs: Any) -> OperatingPoint:
    """Apply a declared estimator. An unknown name is refused with its correction."""
    return ESTIMATORS.get(name).fn(distances, **kwargs)
