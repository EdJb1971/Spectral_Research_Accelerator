"""Calibrating the TG17.0 fixtures at the frozen family level (TG17.5).

**What this measures, and what it does not.** TG17.5's acceptance asks that the false-alignment
fixtures calibrate *at the frozen family level* and that the planted cases meet stated power.
Both of those are properties of a whole declared search, not of one pair: a screen that rejects
one of six pairs at an uncorrected 0.02 has found nothing, and a study that reports it has
reported a family it corrected after looking. So every case here is run as one family of six
pairs, corrected once under Benjamini-Yekutieli at the declared alpha, and what is counted is
rejections *after* that correction.

**The statistic compares support, not rows.** The four fixtures sample at 30-minute, 2-hour,
3-hour and 12-hour cadences over the same 28 days. Binning them onto a common grid would
manufacture the simultaneity TG17.4 exists to refuse, and correlating row `i` against row `i`
would compare a light curve's first fortnight against a float's first three years of nothing.
`support_weighted_correlation` intersects the two records' declared `[start, end)` supports and
weights each contribution by the seconds those supports actually share — so a pair's statistic
is built from overlap duration, and a record rewritten at ten times the row density with the
same support produces the same number.

**The null is the declared one.** Surrogates come from the registered `NullFamily` the manifest
names, applied through `bind_null`, so the calibration exercises the same object a study would
declare rather than a shuffle written here for convenience. `global_value_shuffle` is refused
before it can be used, which is the point of registering it.

Nothing in this module is evidence about the world. It is a measurement of whether a declared
family behaves as declared on fixtures whose answers were fixed in TG17.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.benchmarks.multidomain_flagship import DOMAIN_NAMES, build_multidomain_flagship
from src.benchmarks.seeding import derive
from src.benchmarks.structural_trajectory import (known_answer_declaration, native_from_fixture)
from src.core.structural_nulls import bind_null
from src.core.structural_trajectory import StructuralTrajectory, translate_standardized_level
from src.statistics.multiple_comparisons import adjust


#: The frozen calibration family: every unordered pair of the four flagship domains, one
#: window, one channel, one scale, one relationship. Six members, which 999 replications can
#: resolve under Benjamini-Yekutieli at alpha 0.05 (293 are required).
CALIBRATION_FAMILY_SIZE = 6
DEFAULT_REPLICATIONS = 999
ALPHA = 0.05
CORRECTION = "benjamini_yekutieli"
CHANNEL = "standardized_level"


@dataclass(frozen=True)
class CalibrationResult:
    case: str
    expectation: str
    pairs: Tuple[str, ...]
    statistics: Tuple[float, ...]
    p_values: Tuple[float, ...]
    adjusted: Tuple[float, ...]
    rejected: Tuple[bool, ...]
    family_size: int
    replications: int
    null_family: str

    @property
    def n_rejected(self) -> int:
        return int(sum(self.rejected))

    def describe(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "expectation": self.expectation,
            "family_size": self.family_size,
            "replications": self.replications,
            "null_family": self.null_family,
            "correction": CORRECTION,
            "alpha": ALPHA,
            "n_rejected_after_correction": self.n_rejected,
            "pairs": [
                {"pair": pair, "statistic": statistic, "p_value": p,
                 "adjusted_p_value": q, "rejected": bool(flag)}
                for pair, statistic, p, q, flag in zip(
                    self.pairs, self.statistics, self.p_values, self.adjusted, self.rejected)],
            "claim_boundary": ("A calibration on fixtures with known answers. It says the "
                               "declared family behaves as declared; it is not evidence about "
                               "any domain."),
        }


# ------------------------------------------------------------------------------ statistic


def _shared_support(left: StructuralTrajectory, right: StructuralTrajectory) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every interval the two records both declare, with the value each carried there.

    A linear sweep over two sorted interval sequences. Rows whose validity is false are skipped
    rather than filled: a gap is an absence of observation, and the honest treatment of an
    absence is to contribute no weight to the comparison.
    """
    a_start = np.asarray(left.support_start_seconds, dtype=np.float64)
    a_end = np.asarray(left.support_end_seconds, dtype=np.float64)
    b_start = np.asarray(right.support_start_seconds, dtype=np.float64)
    b_end = np.asarray(right.support_end_seconds, dtype=np.float64)
    a_value = np.asarray(left.channels[CHANNEL], dtype=np.float64)
    b_value = np.asarray(right.channels[CHANNEL], dtype=np.float64)
    a_valid = np.asarray(left.valid_mask, dtype=bool)
    b_valid = np.asarray(right.valid_mask, dtype=bool)

    weights: List[float] = []
    lefts: List[float] = []
    rights: List[float] = []
    i = j = 0
    while i < len(a_start) and j < len(b_start):
        start = max(a_start[i], b_start[j])
        end = min(a_end[i], b_end[j])
        if end > start and a_valid[i] and b_valid[j]:
            weights.append(float(end - start))
            lefts.append(float(a_value[i]))
            rights.append(float(b_value[j]))
        if a_end[i] <= b_end[j]:
            i += 1
        else:
            j += 1
    return (np.asarray(weights, dtype=np.float64), np.asarray(lefts, dtype=np.float64),
            np.asarray(rights, dtype=np.float64))


def support_weighted_correlation(left: StructuralTrajectory,
                                 right: StructuralTrajectory) -> float:
    """Correlation over shared declared support, weighted by the seconds actually shared.

    The magnitude is returned. A shared structure that runs in opposite directions is still a
    shared structure, and this comparison mode is not entitled to a sign: nothing here makes the
    two records' native magnitudes comparable, so "one goes up when the other goes down" is a
    statement about standardized level and not about the world.
    """
    weights, a, b = _shared_support(left, right)
    total = float(np.sum(weights))
    if total <= 0.0 or len(weights) < 2:
        return 0.0
    a_mean = float(np.sum(weights * a) / total)
    b_mean = float(np.sum(weights * b) / total)
    a_dev, b_dev = a - a_mean, b - b_mean
    covariance = float(np.sum(weights * a_dev * b_dev) / total)
    a_var = float(np.sum(weights * a_dev * a_dev) / total)
    b_var = float(np.sum(weights * b_dev * b_dev) / total)
    if a_var <= 0.0 or b_var <= 0.0:
        return 0.0
    return float(abs(covariance / np.sqrt(a_var * b_var)))


# ---------------------------------------------------------------------------- calibration


@lru_cache(maxsize=8)
def _flagship(focus: str, seed: int):
    """Cached because every case in a focus rebuilds the same four fixtures deterministically.

    The seed is part of the key, so caching cannot silently return a different study's records.
    """
    return build_multidomain_flagship(derive("tg17.5-calibration", seed), focus=focus)


def case_trajectories(case_name: str, *, focus: str,
                      seed: int = 20260831) -> Dict[str, StructuralTrajectory]:
    case = _flagship(focus, seed).cases[case_name]
    result: Dict[str, StructuralTrajectory] = {}
    for fixture_domain in DOMAIN_NAMES:
        native = native_from_fixture(case.domains[fixture_domain])
        result[native.domain] = translate_standardized_level(
            native, known_answer_declaration(native.domain))
    return result


def calibrate_case(case_name: str, *, focus: str, expectation: str,
                   null_family: str = "independent_native_clock_shift",
                   parameters: Mapping[str, Any] = (),
                   replications: int = DEFAULT_REPLICATIONS,
                   seed: int = 20260831) -> CalibrationResult:
    """Run one fixture case as one corrected family of six pairs."""
    bound = bind_null(null_family, dict(parameters or {}), mode="calendar_aligned")
    trajectories = case_trajectories(case_name, focus=focus, seed=seed)
    domains = sorted(trajectories)
    pairs = [(a, b) for index, a in enumerate(domains) for b in domains[index + 1:]]
    if len(pairs) != CALIBRATION_FAMILY_SIZE:  # pragma: no cover - four domains give six pairs
        raise ValueError("the frozen calibration family is %d pairs, not %d"
                         % (CALIBRATION_FAMILY_SIZE, len(pairs)))

    # Every replication shifts both records of every pair, so a pair's null distribution is the
    # distribution of its own statistic under the declared surrogate rather than a distribution
    # borrowed from some other pair.
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2 ** 31 - 1, size=(replications, len(domains)))
    surrogates = {domain: [bound.apply(trajectories[domain], int(seeds[index][position]))
                           for index in range(replications)]
                  for position, domain in enumerate(domains)}

    statistics: List[float] = []
    p_values: List[float] = []
    for a, b in pairs:
        observed = support_weighted_correlation(trajectories[a], trajectories[b])
        null = np.asarray([support_weighted_correlation(surrogates[a][index],
                                                        surrogates[b][index])
                           for index in range(replications)], dtype=np.float64)
        # The surrogate p-value, with the observation counted in both numerator and
        # denominator: it cannot be zero, and its floor is what R18 prices the family against.
        statistics.append(observed)
        p_values.append(float((1 + int(np.sum(null >= observed))) / (1 + replications)))

    corrected = adjust(p_values, method=CORRECTION, alpha=ALPHA,
                       n_tests=CALIBRATION_FAMILY_SIZE,
                       labels=["%s|%s" % pair for pair in pairs])
    return CalibrationResult(
        case=case_name, expectation=expectation,
        pairs=tuple("%s|%s" % pair for pair in pairs),
        statistics=tuple(statistics), p_values=tuple(p_values),
        adjusted=tuple(corrected["adjusted"]), rejected=tuple(corrected["rejected"]),
        family_size=CALIBRATION_FAMILY_SIZE, replications=replications,
        null_family=null_family)


#: What each case must do at the frozen family level. Written here rather than in the test, so
#: the expected answer travels with the fixture and a later slice cannot quietly relax it.
CALIBRATION_CASES: Tuple[Dict[str, Any], ...] = (
    {"case": "shared_calendar_event", "focus": "planted", "minimum_rejections": 6,
     "maximum_rejections": 6,
     "expectation": "one shared calendar event: every pair rejects after correction"},
    {"case": "same_window_unrelated", "focus": "safeguards", "minimum_rejections": 0,
     "maximum_rejections": 0,
     "expectation": "identical outer interval, independent construction: no pair rejects"},
    {"case": "gap_alias", "focus": "safeguards", "minimum_rejections": 0,
     "maximum_rejections": 0,
     "expectation": "shared observation gaps must not become shared structure"},
    {"case": "inadmissible_precedence", "focus": "safeguards", "minimum_rejections": 0,
     "maximum_rejections": 6,
     "expectation": ("association may be tested; the precedence members of the family are "
                     "unavailable and are not counted here")},
)


def calibrate_family(*, replications: int = DEFAULT_REPLICATIONS,
                     seed: int = 20260831) -> Dict[str, Any]:
    """Every calibration case, with its declared expectation and what it did."""
    results = []
    for entry in CALIBRATION_CASES:
        result = calibrate_case(entry["case"], focus=entry["focus"],
                                expectation=entry["expectation"],
                                replications=replications, seed=seed)
        results.append({**result.describe(),
                        "minimum_rejections": entry["minimum_rejections"],
                        "maximum_rejections": entry["maximum_rejections"],
                        "met": (entry["minimum_rejections"] <= result.n_rejected
                                <= entry["maximum_rejections"])})
    return {"schema": "family-calibration/v1", "replications": replications,
            "family_size": CALIBRATION_FAMILY_SIZE, "alpha": ALPHA, "correction": CORRECTION,
            "cases": results, "all_met": all(row["met"] for row in results),
            "claim_boundary": ("A frozen-family calibration on TG17.0 fixtures. It is not a "
                               "result about any domain and not a licence to mine.")}


__all__ = ["ALPHA", "CALIBRATION_CASES", "CALIBRATION_FAMILY_SIZE", "CHANNEL", "CORRECTION",
           "DEFAULT_REPLICATIONS", "CalibrationResult", "calibrate_case", "calibrate_family",
           "case_trajectories", "support_weighted_correlation"]
