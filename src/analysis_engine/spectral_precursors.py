"""Precursor tests: the first null this phase draws (`spectral_precursors.py`, T4F.3).

T4F.2 counted. Nothing it emits is evidence of anything, and its own receipt says so: a
confidence of 1.000 is a fact about how often one pattern was followed by another and not a
fact about whether that following means anything. This module is where the question is asked.
It is the roadmap's central one, stated precisely: **when a small attributed configuration
appears at `t`, does the probability of another configuration appearing by `t + delta` rise
above its base rate, above a surrogate ensemble, at admissible lags, under FDR control?**

Five things have to be right, and each of them is a way this measurement lies if it is not.

**1. The base rate has to be measured over the same shape of window the confidence was.**
Confidence here is a *window* probability -- the chance that a declared window following an
occurrence contains the consequent. Comparing it against the fraction of frames carrying the
consequent would divide a window probability by a frame probability and call the ratio a lift,
which inflates with the width of the window and with nothing else. So the base rate is computed
by dropping the *same window* at every searched position on the grid whose window was wholly
observed, and asking how often the consequent falls inside it. `base_rate_positions` publishes
how many positions that was, because a base rate is an estimate and its own sample size is the
first thing a reader needs.

That reference set contains the antecedent's own occurrences. It is not purged of them, and the
direction of that choice is stated rather than hidden: including them pulls lift toward 1 for a
common antecedent, which is the conservative direction, and purging them would make the
reference depend on which rule is being tested, so two rules sharing a consequent would be
divided by different numbers.

**2. The null has to preserve what the claim is not about.** The claim is about *alignment*
between two patterns at a lag. So the default null is `circular_antecedent_shift`: the
antecedent's occurrence times are rotated on the searched lattice, which preserves its count
exactly, preserves its clustering and every gap inside it, and destroys only the alignment
between it and the consequent. The consequent is never touched, so the base rate is invariant
under this null and confidence and lift rank identically -- the p-value is the same statistic
either way, which is worth knowing before someone asks which one was tested.

`uniform_antecedent_relocation` is provided and is **anti-conservative**, deliberately, in the
way `surrogate_null.per_frame_phase` is: it scatters the antecedent's occurrences uniformly over
the searched frames, which preserves the count and destroys the antecedent's own bursting. A
bursty antecedent then beats it on burstiness alone, with no relationship to the consequent
whatsoever. It is here because it is the obvious thing to reach for and because the failure is
worth being able to demonstrate; the acceptance suite measures both on the same data.

Shifts smaller than the window itself are not drawn. A shift of one frame against a window three
frames wide leaves most of the alignment under test in place, so such a surrogate is not a null
draw but a slightly blurred copy of the observation. The floor is derived from the declared
window and the grid's cadence rather than chosen.

**3. A lag chosen by the data is a search, and the null has to be over the choice.**
`src/core/precedence.py` found this the hard way at the band level: the maximum of twelve lagged
statistics is not one statistic, and testing the winner against the null of a single lag prices
a search of twelve as one test. So this module reports two families and corrects each on its own.
The per-lag family is every (antecedent, consequent, window) triple, every one of them reported
whether it looked interesting or not, so no selection happens before the correction. The
selected-lag family is one member per pair, and its p-value comes from the distribution of the
**maximum across the declared lags** -- which is available exactly because each surrogate draw
uses one shift across the whole lag family rather than a fresh shift per lag.

**4. A design that cannot reject is refused before it is run, not after.** With `n` surrogates a
p-value cannot go below `1 / (1 + n)`, and a family of `m` tests under Benjamini-Yekutieli needs
a raw p below roughly `alpha / (m * H_m)`. A study that fails that arithmetic will duly report
nothing and the nothing will be indistinguishable from a clean negative result. `check_power`
already knows this and it is asked *before* any counting happens, so the refusal is a fact about
the declared design and can never be a suppressed absence. T4C.5i's boundary -- an inadequately
powered absence is not a negative finding -- is honoured by never producing the absence.

**5. The vocabulary stops below causation (R7).** A rejected rule is a `PRECURSOR_SIGNATURE`:
the antecedent's appearance was followed by the consequent more often than the record's own
re-alignments of it were. Not a cause, not a driver, not a mechanism, not a forecast. Every rule
carries R9's six figures or declares which of them is undefined and why, and
`PrecursorRule.figures()` builds the programme's existing `AssociationFigures` rather than a
seventh private home for the same six numbers -- so a rule that cannot satisfy R9's contract is
refused by the contract itself instead of by a check in this file.

Nothing here is reimplemented. The counting and its eligibility rule are T4F.2's
`count_sequence`; the empirical p-value is `significance.surrogate_p_value`; the correction and
the power check are `multiple_comparisons.adjust` and `check_power`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.spectral_events import (
    COVERAGE_UNDECLARED, EventSeries, ObservationGrid, WINDOW_MEASURED,
)
from src.analysis_engine.spectral_mining import MiningBudget, MiningBudgetExceededError
from src.analysis_engine.spectral_sequences import (
    CONFIDENCE_MEASURED, TransitionWindow, count_sequence, times_by_pattern,
)
from src.core.errors import InvalidParameterError
from src.core.translation import AssociationFigures
from src.statistics.multiple_comparisons import PROCEDURES, adjust, check_power
from src.statistics.significance import surrogate_p_value


PRECURSOR_SCHEMA = "spectral-precursor-rules/v1"

#: What one member of the per-lag family is, spelled out, because "n_tests" is the number a
#: reader has to trust and the thing it counts is easy to get wrong.
TEST_UNIT = "one ordered (antecedent, consequent) pair at one declared lag window"

NULL_CIRCULAR_SHIFT = "circular_antecedent_shift"
NULL_UNIFORM_RELOCATION = "uniform_antecedent_relocation"
PRECURSOR_NULLS = (NULL_CIRCULAR_SHIFT, NULL_UNIFORM_RELOCATION)

NULL_PRESERVES: Dict[str, Tuple[str, ...]] = {
    NULL_CIRCULAR_SHIFT: (
        "the antecedent's occurrence count exactly",
        "the antecedent's own clustering: every gap between its consecutive occurrences, up to "
        "the one gap the wrap point falls in",
        "the consequent entirely, and therefore the base rate",
        "the searched grid, so eligibility is decided for a surrogate by the same rule that "
        "decided it for the observation",
    ),
    NULL_UNIFORM_RELOCATION: (
        "the antecedent's occurrence count exactly",
        "the consequent entirely, and therefore the base rate",
        "the searched grid",
    ),
}

NULL_DESTROYS: Dict[str, Tuple[str, ...]] = {
    NULL_CIRCULAR_SHIFT: (
        "the alignment between the antecedent and the consequent at every lag at once",
    ),
    NULL_UNIFORM_RELOCATION: (
        "the alignment between the antecedent and the consequent",
        "**the antecedent's own temporal clustering** -- which is why this null is "
        "anti-conservative for any bursty pattern, related to the consequent or not",
    ),
}

NULL_CAUTION: Dict[str, Optional[str]] = {
    NULL_CIRCULAR_SHIFT: None,
    NULL_UNIFORM_RELOCATION: (
        "**Anti-conservative.** This null does not preserve the antecedent's bursting, so a "
        "pattern that fires in "
        "clumps clears it on the shape of its own occurrence times with no relation to the "
        "consequent at all. It is provided so that failure can be demonstrated, and it should "
        "not be used to support a claim."),
}

#: The rule was rejected against its null after correction. This is a *signature*: the
#: antecedent's appearance preceded the consequent more often than the record's own
#: re-alignments of that antecedent did. It is not a cause and not a forecast (R7).
RULE_PRECURSOR = "PRECURSOR_SIGNATURE"
#: Tested, and the observation sits inside what the null produced.
RULE_NOT_DISTINGUISHED = "NOT_DISTINGUISHED_FROM_NULL"
#: Every occurrence of the antecedent was censored or holed, so there is no denominator.
RULE_NO_ELIGIBLE_ANTECEDENT = "NO_ELIGIBLE_ANTECEDENT"
#: The consequent never fell inside any observed window anywhere on the grid, so there is no
#: base rate and lift is not a number.
RULE_BASE_RATE_ZERO = "CONSEQUENT_NEVER_OBSERVED_IN_A_WINDOW"
#: No cadence was declared, so eligibility itself is undecidable and nothing downstream of it
#: can be computed. The count still stands; the ratio does not.
RULE_UNDECIDABLE = COVERAGE_UNDECLARED

INTERVAL_METHOD = "wilson_score"

#: Said on every interval rather than in a docstring, because this is the assumption most
#: likely to be forgotten by whoever reads the number.
INTERVAL_ASSUMPTION = (
    "A Wilson interval treats the eligible antecedent occurrences as independent trials. "
    "Overlapping windows on an autocorrelated record are not independent, so where "
    "overlapping_eligible_windows is above zero this interval is narrower than the truth and "
    "is a lower bound on the uncertainty rather than a bound on the effect.")

PRECURSOR_CLAIM_BOUNDARY = (
    "A rejected rule is a precursor signature and nothing more: the antecedent was followed by "
    "the consequent within the declared window more often than the record's own re-alignments "
    "of that antecedent were. It is not a cause, a driver, a mechanism, a trigger or a "
    "forecast, and no intervention is implied or supported (R7). The null is a statement about "
    "alignment, so what is rejected is the hypothesis that the two patterns are unaligned at "
    "this lag -- both could follow a third thing this record does not contain, and a "
    "correlation surviving a surrogate ensemble is still a correlation. The p-value is "
    "referenced to the named null and means nothing without it; the lift interval assumes an "
    "independence the record does not have; and the base rate is estimated from this record "
    "alone, so a rule is a statement about this record and not about the world.")

SELECTION_NOTE = (
    "The lag was chosen by the data, so the choice is part of what is tested: this p-value is "
    "referenced to the distribution of the maximum across the whole declared lag family, not to "
    "the distribution at the chosen lag. Testing the winner against a single-lag null would "
    "price a search of several lags as one test.")


def _positive_int(name: str, value: Any, expectation: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidParameterError(name, value, expectation)
    return int(value)


@dataclass(frozen=True)
class LagFamily:
    """The admissible lags, declared before the record is read, as whole windows.

    A lag family is a hypothesis family, and the number of members in it is what the correction
    is paid on. It is a frozen object rather than a loop variable so that the family a report
    was corrected against is the family the report carries, and so that a caller cannot widen
    the search after seeing a result without the arithmetic changing with it.
    """

    windows: Tuple[TransitionWindow, ...]

    def __post_init__(self) -> None:
        windows = tuple(self.windows)
        if not windows:
            raise InvalidParameterError(
                "LagFamily.windows", windows,
                "at least one declared lag window. A precursor test with no admissible lag is "
                "not a weaker test, it is not a test")
        for index, window in enumerate(windows):
            if not isinstance(window, TransitionWindow):
                raise InvalidParameterError("LagFamily.windows[%d]" % index,
                                            type(window).__name__, "a TransitionWindow")
        units = {window.time_units for window in windows}
        if len(units) > 1:
            raise InvalidParameterError(
                "LagFamily.windows", sorted(units),
                "one time unit across the whole family. Lags declared in two units are two "
                "families, and correcting them together would be arithmetic across a "
                "conversion nobody performed")
        seen = [(float(w.minimum_lag), float(w.maximum_lag)) for w in windows]
        if len(set(seen)) != len(seen):
            raise InvalidParameterError(
                "LagFamily.windows", seen,
                "distinct lag windows. A window declared twice is one hypothesis counted twice "
                "in the correction, which pays for a test nobody ran")
        object.__setattr__(self, "windows", windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[TransitionWindow]:
        return iter(self.windows)

    @property
    def time_units(self) -> str:
        return self.windows[0].time_units

    @property
    def overlaps(self) -> bool:
        """Do two declared windows share an admissible lag? Reported, not refused.

        Overlapping windows are not independent tests. Benjamini-Yekutieli assumes arbitrary
        dependence and so remains valid, which is exactly why it is the default here rather
        than Benjamini-Hochberg; but a reader counting tests should know the family is not as
        wide as its member count suggests.
        """
        ordered = sorted((float(w.minimum_lag), float(w.maximum_lag)) for w in self.windows)
        return any(later[0] <= earlier[1] for earlier, later in zip(ordered, ordered[1:]))

    def describe(self) -> Dict[str, Any]:
        return {
            "n_windows": len(self.windows),
            "time_units": self.time_units,
            "windows": [window.describe() for window in self.windows],
            "windows_overlap": self.overlaps,
        }


@dataclass(frozen=True)
class PrecursorRule:
    """One ordered pair at one declared lag, with the six figures or the reason there are five."""

    antecedent: int
    consequent: int
    window: TransitionWindow
    status: str

    support: int
    eligible_antecedents: int
    antecedent_occurrences: int
    ineligible_truncated: int
    ineligible_unobserved: int
    confidence: Optional[float]

    base_rate: Optional[float]
    base_rate_positions: int
    base_rate_hits: int

    lift: Optional[float]
    lift_interval: Optional[Tuple[float, float]]
    overlapping_eligible_windows: int

    null_mean_confidence: Optional[float]
    surrogate_corrected_lift: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float]
    n_surrogates: int
    n_surrogates_at_least_as_extreme: Optional[int]

    antecedent_cardinality: Optional[int]
    consequent_cardinality: Optional[int]

    #: The confidence every surrogate produced, in draw order. Published rather than summarised
    #: away because a p-value is a statement about a distribution and the distribution is the
    #: part a reader can check; and because the draws are shared across the declared lag family,
    #: this is what makes the null of the maximum checkable rather than merely asserted.
    null_confidences: Tuple[float, ...]

    @property
    def rejected(self) -> bool:
        return self.status == RULE_PRECURSOR

    def null_quantiles(self) -> Optional[Dict[str, float]]:
        if not self.null_confidences:
            return None
        ordered = sorted(self.null_confidences)
        def _at(fraction: float) -> float:
            return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]
        return {"minimum": ordered[0], "median": _at(0.5), "upper_95": _at(0.95),
                "maximum": ordered[-1]}

    def figures(self) -> Optional[AssociationFigures]:
        """R9's six figures as the programme's own carrier, or `None` with a status saying why.

        This does not validate the six numbers itself. `AssociationFigures` already refuses a
        zero base rate, a lift that is not the ratio it claims to be, and a support of nothing,
        and a second copy of those rules here would be a second thing to keep in step. If this
        returns `None`, `status` names which figure is missing.
        """
        if self.support < 1 or self.confidence is None or not self.base_rate \
                or self.lift is None or self.lift_interval is None \
                or self.surrogate_corrected_lift is None:
            return None
        return AssociationFigures(
            support=int(self.support), confidence=float(self.confidence),
            base_rate=float(self.base_rate), lift=float(self.lift),
            lift_interval=(float(self.lift_interval[0]), float(self.lift_interval[1])),
            surrogate_corrected_lift=float(self.surrogate_corrected_lift))

    def describe(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "window": self.window.describe(),
            "status": self.status,
            "support": self.support,
            "eligible_antecedents": self.eligible_antecedents,
            "antecedent_occurrences": self.antecedent_occurrences,
            "ineligible_truncated_by_record": self.ineligible_truncated,
            "ineligible_unobserved_time": self.ineligible_unobserved,
            "confidence": self.confidence,
            "base_rate": self.base_rate,
            "base_rate_positions": self.base_rate_positions,
            "base_rate_hits": self.base_rate_hits,
            "base_rate_basis": (
                "the same window dropped at every searched position whose window was wholly "
                "observed, including the antecedent's own occurrences"),
            "lift": self.lift,
            "lift_interval": (None if self.lift_interval is None
                              else list(self.lift_interval)),
            "lift_interval_method": INTERVAL_METHOD,
            "lift_interval_assumption": INTERVAL_ASSUMPTION,
            "overlapping_eligible_windows": self.overlapping_eligible_windows,
            "null_mean_confidence": self.null_mean_confidence,
            "surrogate_corrected_lift": self.surrogate_corrected_lift,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "n_surrogates": self.n_surrogates,
            "n_surrogates_at_least_as_extreme": self.n_surrogates_at_least_as_extreme,
            "null_quantiles": self.null_quantiles(),
            "antecedent_cardinality": self.antecedent_cardinality,
            "consequent_cardinality": self.consequent_cardinality,
            "consequent_is_larger": (
                None if self.antecedent_cardinality is None
                or self.consequent_cardinality is None
                else self.consequent_cardinality > self.antecedent_cardinality),
        }


@dataclass(frozen=True)
class SelectedLagRule:
    """One pair's strongest declared lag, tested against the null of that maximum."""

    antecedent: int
    consequent: int
    window: TransitionWindow
    status: str
    confidence: Optional[float]
    base_rate: Optional[float]
    lift: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float]
    n_surrogates: int
    n_surrogates_at_least_as_extreme: Optional[int]
    n_lags_searched: int
    #: The maximum confidence each surrogate reached anywhere in the declared lag family. It is
    #: the elementwise maximum of the per-lag ensembles precisely because one draw is shared
    #: across the family, and that identity is what a test can hold this to.
    null_maximum: Tuple[float, ...]

    @property
    def rejected(self) -> bool:
        return self.status == RULE_PRECURSOR

    def describe(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "selected_window": self.window.describe(),
            "status": self.status,
            "confidence": self.confidence,
            "base_rate": self.base_rate,
            "lift": self.lift,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "n_surrogates": self.n_surrogates,
            "n_surrogates_at_least_as_extreme": self.n_surrogates_at_least_as_extreme,
            "n_lags_searched": self.n_lags_searched,
            "null_maximum_quantiles": (
                None if not self.null_maximum
                else {"minimum": min(self.null_maximum),
                      "median": sorted(self.null_maximum)[len(self.null_maximum) // 2],
                      "maximum": max(self.null_maximum)}),
            "selection_note": SELECTION_NOTE,
        }


@dataclass(frozen=True)
class PrecursorReport:
    """Every rule the declared family contains, under one null, one alpha and one correction."""

    rules: Tuple[PrecursorRule, ...]
    selected: Tuple[SelectedLagRule, ...]
    lags: LagFamily
    null_method: str
    n_surrogates: int
    alpha: float
    correction: str
    seed: int
    minimum_shift: float
    n_shifts_available: int
    power: Mapping[str, Any]
    per_lag_correction: Mapping[str, Any]
    selected_lag_correction: Mapping[str, Any]
    counts_performed: int

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[PrecursorRule]:
        return iter(self.rules)

    @property
    def signatures(self) -> Tuple[PrecursorRule, ...]:
        return tuple(rule for rule in self.rules if rule.rejected)

    def rule(self, antecedent: int, consequent: int,
             window: TransitionWindow) -> PrecursorRule:
        for item in self.rules:
            if item.antecedent == int(antecedent) and item.consequent == int(consequent) \
                    and item.window == window:
                return item
        raise InvalidParameterError("(antecedent, consequent, window)",
                                    (antecedent, consequent, window.describe()),
                                    "a pair and window this report tested")

    def selected_rule(self, antecedent: int, consequent: int) -> SelectedLagRule:
        for item in self.selected:
            if item.antecedent == int(antecedent) and item.consequent == int(consequent):
                return item
        raise InvalidParameterError("(antecedent, consequent)", (antecedent, consequent),
                                    "a pair this report tested")

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": PRECURSOR_SCHEMA,
            "n_rules": len(self.rules),
            "n_signatures": len(self.signatures),
            "test_unit": TEST_UNIT,
            "lags": self.lags.describe(),
            "null": {
                "method": self.null_method,
                "n_surrogates": self.n_surrogates,
                "seed": self.seed,
                "preserves": list(NULL_PRESERVES[self.null_method]),
                "destroys": list(NULL_DESTROYS[self.null_method]),
                "caution": NULL_CAUTION[self.null_method],
                "minimum_shift": self.minimum_shift,
                "minimum_shift_basis": (
                    "one cadence step beyond the widest declared lag, so no surrogate retains "
                    "any part of the alignment under test"),
                "n_shifts_available": self.n_shifts_available,
                "base_rate_invariant": (
                    "the consequent is not touched by this null, so the base rate is identical "
                    "for every surrogate and confidence and lift rank the ensemble identically"),
            },
            "alpha": self.alpha,
            "correction": self.correction,
            "power": dict(self.power),
            "per_lag_correction": dict(self.per_lag_correction),
            "selected_lag_correction": dict(self.selected_lag_correction),
            "counts_performed": self.counts_performed,
            "rules": [rule.describe() for rule in self.rules],
            "selected_lags": [item.describe() for item in self.selected],
            "claim_boundary": PRECURSOR_CLAIM_BOUNDARY,
        }


# --------------------------------------------------------------------------------------------
# the base rate, and the interval
# --------------------------------------------------------------------------------------------


def window_base_rate(series: EventSeries, consequent: int,
                     window: TransitionWindow) -> Tuple[Optional[float], int, int]:
    """How often the declared window contains the consequent, dropped anywhere it was observed.

    The universe is the searched lattice, not the antecedent's occurrences: this is the number
    a confidence has to beat, and it must not depend on which antecedent is being tested or two
    rules sharing a consequent would be divided by different denominators.
    """
    grid = series.grid
    if grid.cadence is None:
        return None, 0, 0
    occurrences = sorted(float(event.time) for event in series.for_pattern(int(consequent)))
    positions = 0
    hits = 0
    low = float(window.minimum_lag)
    high = float(window.maximum_lag)
    for frame in grid.frames:
        anchor = float(frame)
        status, _observed, _missing = grid.observation_of_window(anchor + low, anchor + high)
        if status != WINDOW_MEASURED:
            continue
        positions += 1
        if any(anchor + low <= time <= anchor + high for time in occurrences):
            hits += 1
    if positions == 0:
        return None, 0, 0
    return hits / float(positions), positions, hits


def _wilson(successes: int, trials: int, level: float) -> Optional[Tuple[float, float]]:
    if trials <= 0:
        return None
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    p = successes / float(trials)
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def _overlapping_windows(anchors: Sequence[float], window: TransitionWindow) -> int:
    """How many eligible anchors have a window overlapping the previous one's.

    This is the measurement that makes the interval's independence assumption checkable rather
    than merely disclosed: an anchor whose window overlaps its predecessor's is not a fresh
    trial, and a reader can see how many of the trials were not fresh.
    """
    ordered = sorted(float(value) for value in anchors)
    low = float(window.minimum_lag)
    high = float(window.maximum_lag)
    return sum(1 for earlier, later in zip(ordered, ordered[1:])
               if later + low <= earlier + high)


# --------------------------------------------------------------------------------------------
# the null
# --------------------------------------------------------------------------------------------


def _shift_floor(lags: LagFamily, grid: ObservationGrid) -> float:
    """The smallest shift that removes every part of the alignment under test."""
    widest = max(float(window.maximum_lag) for window in lags)
    cadence = float(grid.cadence)
    steps = int(math.ceil(widest / cadence - 1e-9)) + 1
    return steps * cadence


def _available_shifts(grid: ObservationGrid, floor: float) -> Tuple[float, ...]:
    """Every lattice rotation that is at least `floor` away from the identity, both ways round.

    A rotation of `extent - s` is a rotation of `s` in the other direction, so a shift close to
    the full extent is as small a disturbance as a shift close to zero and is excluded with it.
    """
    cadence = float(grid.cadence)
    extent = float(grid.span) + cadence
    steps = int(round(extent / cadence))
    return tuple(step * cadence for step in range(1, steps)
                 if step * cadence >= floor - 1e-9
                 and extent - step * cadence >= floor - 1e-9)


def _rotate(times: Sequence[float], shift: float, origin: float,
            extent: float) -> List[float]:
    return sorted(origin + ((float(time) - origin + shift) % extent) for time in times)


def _relocate(count: int, frames: Sequence[float], rng: Any) -> List[float]:
    chosen = rng.choice(np.asarray([float(frame) for frame in frames], dtype=np.float64),
                        size=count, replace=False)
    return sorted(float(value) for value in chosen)


# --------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------


def precursor_report(
        series: EventSeries, *, lags: LagFamily, null: str = NULL_CIRCULAR_SHIFT,
        n_surrogates: int, alpha: float = 0.05,
        correction: str = "benjamini_yekutieli", seed: int,
        interval_confidence: float = 0.95,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        budget: Optional[MiningBudget] = None) -> PrecursorReport:
    """Test every declared pair at every declared lag, against a null, under FDR control.

    Every argument that changes what a p-value means is required rather than defaulted: the
    ensemble size, the seed and the lag family. `alpha` and `correction` carry the programme's
    defaults because they are declared identically everywhere else in it.
    """
    if not isinstance(series, EventSeries):
        raise InvalidParameterError("series", type(series).__name__, "a T4F.1 EventSeries")
    if not isinstance(lags, LagFamily):
        raise InvalidParameterError(
            "lags", type(lags).__name__,
            "a declared LagFamily. A lag family assembled at the call site is a family whose "
            "size the correction never sees")
    if null not in PRECURSOR_NULLS:
        raise InvalidParameterError("null", null, "one of %s" % (list(PRECURSOR_NULLS),))
    if correction not in PROCEDURES:
        raise InvalidParameterError("correction", correction,
                                    "one of %s" % (list(PROCEDURES),))
    n_surrogates = _positive_int("n_surrogates", n_surrogates, "a positive ensemble size")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidParameterError(
            "seed", seed,
            "a declared integer seed. An ensemble nobody can redraw is an ensemble nobody can "
            "check")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise InvalidParameterError("alpha", alpha, "a significance level in (0, 1)")
    if not isinstance(interval_confidence, (int, float)) \
            or isinstance(interval_confidence, bool) or not 0 < interval_confidence < 1:
        raise InvalidParameterError("interval_confidence", interval_confidence,
                                    "an interval level in (0, 1)")

    grid = series.grid
    if grid.cadence is None:
        raise InvalidParameterError(
            "series.grid.cadence", None,
            "a declared cadence. Without one, eligibility is undecidable, the base rate has no "
            "universe of positions to be estimated over, and a circular shift has no lattice to "
            "rotate on. T4F.2 can still count on such a grid; nothing here can be inferred "
            "from it")
    for window in lags:
        if window.time_units != grid.time_units:
            raise InvalidParameterError(
                "LagFamily.time_units", window.time_units,
                "the grid's own unit of %r. A lag declared in one unit against frames declared "
                "in another is refused rather than converted" % grid.time_units)

    identities = series.pattern_ids
    if pairs is None:
        declared = tuple((earlier, later) for earlier in identities for later in identities
                         if earlier != later)
    else:
        declared = tuple((int(a), int(b)) for a, b in pairs)
        for antecedent, consequent in declared:
            if antecedent == consequent:
                raise InvalidParameterError(
                    "pairs", (antecedent, consequent),
                    "an ordered pair of distinct patterns. A pattern preceding itself is a "
                    "recurrence question, and T4F.2's recurrence_report answers it with a "
                    "denominator built for that question rather than for this one")
            for name, value in (("antecedent", antecedent), ("consequent", consequent)):
                if value not in identities:
                    raise InvalidParameterError("pairs.%s" % name, value,
                                                "a pattern this series contains")
    if not declared:
        raise InvalidParameterError(
            "series.pattern_ids", list(identities),
            "at least two distinct patterns, so that an ordered pair exists to test")

    family_size = len(declared) * len(lags)
    power = check_power(n_surrogates, family_size, alpha=alpha, method=correction)
    if not power["can_reject_after_correction"]:
        raise InvalidParameterError(
            "n_surrogates", n_surrogates,
            "an ensemble this declared family could reject one of. %s This is refused before "
            "any counting happens, so it is a fact about the design and not a result: an "
            "under-powered study reports an absence that is indistinguishable from a real one, "
            "and that absence must never be produced in the first place" % power["warning"])

    counts_required = family_size * (1 + n_surrogates)
    if budget is not None:
        if not isinstance(budget, MiningBudget):
            raise InvalidParameterError("budget", type(budget).__name__, "a T4E.4 MiningBudget")
        if counts_required > budget.max_candidates:
            # Priced before anything is counted, because the arithmetic is exactly known in
            # advance: a family corrected against a size it did not run is corrected against a
            # fiction, so the sweep is refused whole rather than truncated.
            raise MiningBudgetExceededError(
                "precursor sequence count", counts_required, budget.max_candidates)

    floor = _shift_floor(lags, grid)
    shifts = _available_shifts(grid, floor)
    if null == NULL_CIRCULAR_SHIFT and len(shifts) < n_surrogates:
        raise InvalidParameterError(
            "n_surrogates", n_surrogates,
            "an ensemble no larger than the %d distinct rotations this record admits. Shifts "
            "below %g %s are excluded because they leave part of the alignment under test in "
            "place, and drawing the same rotation twice would price one null draw as several"
            % (len(shifts), floor, grid.time_units))
    if null == NULL_UNIFORM_RELOCATION and len(grid.frames) < 1:
        raise InvalidParameterError("series.grid.frames", 0, "a grid with searched frames")

    rng = np.random.default_rng(int(seed))
    times = times_by_pattern(series)
    cardinalities = {event.pattern_id: int(event.cardinality) for event in series}
    origin = float(grid.frames[0])
    extent = float(grid.span) + float(grid.cadence)

    # One draw per antecedent per surrogate index, shared across the whole lag family. That
    # sharing is what makes the maximum-across-lags null a valid null of the maximum: a fresh
    # draw per lag would give the maximum of independent nulls, which is wider than the null of
    # the maximum and would make the selected-lag test anti-conservative.
    antecedents = sorted({pair[0] for pair in declared})
    surrogate_times: Dict[int, List[List[float]]] = {}
    for antecedent in antecedents:
        observed = times.get(antecedent, [])
        draws: List[List[float]] = []
        if null == NULL_CIRCULAR_SHIFT:
            picked = rng.choice(np.asarray(shifts, dtype=np.float64), size=n_surrogates,
                                replace=False)
            draws = [_rotate(observed, float(shift), origin, extent) for shift in picked]
        else:
            draws = [_relocate(len(observed), grid.frames, rng) for _ in range(n_surrogates)]
        surrogate_times[antecedent] = draws

    base_rates: Dict[Tuple[int, int], Tuple[Optional[float], int, int]] = {}
    for _, consequent in declared:
        for index, window in enumerate(lags):
            key = (consequent, index)
            if key not in base_rates:
                base_rates[key] = window_base_rate(series, consequent, window)

    performed = 0
    provisional: List[Dict[str, Any]] = []
    null_by_pair: Dict[Tuple[int, int], List[List[float]]] = {}
    for antecedent, consequent in declared:
        per_lag_nulls: List[List[float]] = []
        for index, window in enumerate(lags):
            counted = count_sequence((antecedent, consequent), series, times, window, 1)
            performed += 1
            rate, positions, hits = base_rates[(consequent, index)]

            null_confidences: List[float] = []
            for draw in surrogate_times[antecedent]:
                shifted = dict(times)
                shifted[antecedent] = draw
                surrogate = count_sequence((antecedent, consequent), series, shifted, window, 1)
                performed += 1
                null_confidences.append(
                    0.0 if surrogate.confidence is None else float(surrogate.confidence))
            per_lag_nulls.append(null_confidences)

            eligible_anchors = [time for time in times.get(antecedent, [])
                                if grid.observation_of_window(
                                    time + float(window.minimum_lag),
                                    time + float(window.maximum_lag))[0] == WINDOW_MEASURED]
            overlapping = _overlapping_windows(eligible_anchors, window)

            if counted.confidence_status != CONFIDENCE_MEASURED:
                status = (RULE_UNDECIDABLE if counted.confidence is None
                          and grid.cadence is None else RULE_NO_ELIGIBLE_ANTECEDENT)
                provisional.append({
                    "pair": (antecedent, consequent), "window": window, "counted": counted,
                    "rate": rate, "positions": positions, "hits": hits, "status": status,
                    "lift": None, "interval": None, "overlapping": overlapping,
                    "null": null_confidences, "p": None, "extreme": None,
                    "null_mean": None, "corrected": None})
                continue

            if not rate:
                provisional.append({
                    "pair": (antecedent, consequent), "window": window, "counted": counted,
                    "rate": rate, "positions": positions, "hits": hits,
                    "status": RULE_BASE_RATE_ZERO, "lift": None, "interval": None,
                    "overlapping": overlapping, "null": null_confidences, "p": None,
                    "extreme": None, "null_mean": None, "corrected": None})
                continue

            confidence = float(counted.confidence)
            interval = _wilson(counted.support, counted.eligible_antecedents,
                               float(interval_confidence))
            lift_interval = None if interval is None else (interval[0] / rate,
                                                           interval[1] / rate)
            outcome = surrogate_p_value(confidence, null_confidences, alternative="greater")
            null_mean = float(outcome["null_mean"])
            provisional.append({
                "pair": (antecedent, consequent), "window": window, "counted": counted,
                "rate": rate, "positions": positions, "hits": hits,
                "status": RULE_NOT_DISTINGUISHED, "lift": confidence / rate,
                "interval": lift_interval, "overlapping": overlapping,
                "null": null_confidences, "p": float(outcome["p_value"]),
                "extreme": int(outcome["n_exceeding"]), "null_mean": null_mean,
                "corrected": (None if null_mean <= 0.0 else confidence / null_mean)})
        null_by_pair[(antecedent, consequent)] = per_lag_nulls

    testable = [index for index, item in enumerate(provisional) if item["p"] is not None]
    per_lag_correction: Dict[str, Any] = {
        "method": correction, "alpha": alpha, "n_tests": family_size,
        "n_p_values": len(testable), "n_rejected": 0,
        "note": ("every member of the declared family is reported whether it looked "
                 "interesting or not, so no selection precedes this correction"),
    }
    if testable:
        adjusted = adjust([provisional[index]["p"] for index in testable], method=correction,
                          alpha=alpha, n_tests=family_size,
                          labels=["%d->%d@[%g,%g]"
                                  % (provisional[index]["pair"][0],
                                     provisional[index]["pair"][1],
                                     provisional[index]["window"].minimum_lag,
                                     provisional[index]["window"].maximum_lag)
                                  for index in testable])
        per_lag_correction = dict(adjusted)
        per_lag_correction["note"] = (
            "every member of the declared family is reported whether it looked interesting or "
            "not, so no selection precedes this correction")
        for position, index in enumerate(testable):
            provisional[index]["q"] = float(adjusted["adjusted"][position])
            if adjusted["rejected"][position]:
                provisional[index]["status"] = RULE_PRECURSOR

    rules = tuple(
        PrecursorRule(
            antecedent=item["pair"][0], consequent=item["pair"][1], window=item["window"],
            status=item["status"], support=item["counted"].support,
            eligible_antecedents=item["counted"].eligible_antecedents,
            antecedent_occurrences=item["counted"].antecedent_occurrences,
            ineligible_truncated=item["counted"].ineligible_truncated,
            ineligible_unobserved=item["counted"].ineligible_unobserved,
            confidence=item["counted"].confidence, base_rate=item["rate"],
            base_rate_positions=item["positions"], base_rate_hits=item["hits"],
            lift=item["lift"], lift_interval=item["interval"],
            overlapping_eligible_windows=item["overlapping"],
            null_mean_confidence=item["null_mean"],
            surrogate_corrected_lift=item["corrected"], p_value=item["p"],
            q_value=item.get("q"), n_surrogates=n_surrogates,
            n_surrogates_at_least_as_extreme=item["extreme"],
            antecedent_cardinality=cardinalities.get(item["pair"][0]),
            consequent_cardinality=cardinalities.get(item["pair"][1]),
            null_confidences=tuple(float(value) for value in item["null"]))
        for item in provisional)

    selected = _select_lags(rules, declared, lags, null_by_pair, n_surrogates, alpha,
                            correction)

    return PrecursorReport(
        rules=rules, selected=selected[0], lags=lags, null_method=null,
        n_surrogates=n_surrogates, alpha=float(alpha), correction=correction, seed=int(seed),
        minimum_shift=floor, n_shifts_available=len(shifts), power=power,
        per_lag_correction=per_lag_correction, selected_lag_correction=selected[1],
        counts_performed=performed)


def _select_lags(rules: Sequence[PrecursorRule], declared: Sequence[Tuple[int, int]],
                 lags: LagFamily, null_by_pair: Mapping[Tuple[int, int], List[List[float]]],
                 n_surrogates: int, alpha: float,
                 correction: str) -> Tuple[Tuple[SelectedLagRule, ...], Dict[str, Any]]:
    """One member per pair: its strongest lag, against the null of the maximum over the family.

    Kept apart from the per-lag family because it is a different question corrected against a
    different family size, and reporting a selected winner inside a per-lag correction would be
    the exact double-dip `precedence.py` names.
    """
    by_pair: Dict[Tuple[int, int], List[PrecursorRule]] = {}
    for rule in rules:
        by_pair.setdefault((rule.antecedent, rule.consequent), []).append(rule)

    provisional: List[Dict[str, Any]] = []
    for pair in declared:
        members = by_pair.get(pair, [])
        testable = [rule for rule in members if rule.confidence is not None]
        if not testable:
            provisional.append({"pair": pair, "rule": members[0] if members else None,
                                "p": None, "extreme": None,
                                "status": (members[0].status if members
                                           else RULE_NO_ELIGIBLE_ANTECEDENT)})
            continue
        best = max(testable, key=lambda rule: (float(rule.confidence), -rule.window.maximum_lag))
        maxima = [max(values) for values in zip(*null_by_pair[pair])]
        outcome = surrogate_p_value(float(best.confidence), maxima, alternative="greater")
        provisional.append({"pair": pair, "rule": best, "p": float(outcome["p_value"]),
                            "extreme": int(outcome["n_exceeding"]), "maxima": maxima,
                            "status": RULE_NOT_DISTINGUISHED})

    tested = [index for index, item in enumerate(provisional) if item["p"] is not None]
    record: Dict[str, Any] = {"method": correction, "alpha": alpha, "n_tests": len(declared),
                              "n_p_values": len(tested), "n_rejected": 0,
                              "note": SELECTION_NOTE}
    if tested:
        adjusted = adjust([provisional[index]["p"] for index in tested], method=correction,
                          alpha=alpha, n_tests=len(declared),
                          labels=["%d->%d" % provisional[index]["pair"] for index in tested])
        record = dict(adjusted)
        record["note"] = SELECTION_NOTE
        for position, index in enumerate(tested):
            provisional[index]["q"] = float(adjusted["adjusted"][position])
            if adjusted["rejected"][position]:
                provisional[index]["status"] = RULE_PRECURSOR

    selected = tuple(
        SelectedLagRule(
            antecedent=item["pair"][0], consequent=item["pair"][1],
            window=(item["rule"].window if item["rule"] is not None else lags.windows[0]),
            status=item["status"],
            confidence=(item["rule"].confidence if item["rule"] is not None else None),
            base_rate=(item["rule"].base_rate if item["rule"] is not None else None),
            lift=(item["rule"].lift if item["rule"] is not None else None),
            p_value=item["p"], q_value=item.get("q"), n_surrogates=n_surrogates,
            n_surrogates_at_least_as_extreme=item["extreme"], n_lags_searched=len(lags),
            null_maximum=tuple(item.get("maxima") or ()))
        for item in provisional)
    return selected, record


__all__ = [
    "PRECURSOR_SCHEMA", "TEST_UNIT", "PRECURSOR_CLAIM_BOUNDARY", "SELECTION_NOTE",
    "INTERVAL_METHOD", "INTERVAL_ASSUMPTION",
    "NULL_CIRCULAR_SHIFT", "NULL_UNIFORM_RELOCATION", "PRECURSOR_NULLS",
    "NULL_PRESERVES", "NULL_DESTROYS", "NULL_CAUTION",
    "RULE_PRECURSOR", "RULE_NOT_DISTINGUISHED", "RULE_NO_ELIGIBLE_ANTECEDENT",
    "RULE_BASE_RATE_ZERO", "RULE_UNDECIDABLE",
    "LagFamily", "PrecursorRule", "SelectedLagRule", "PrecursorReport",
    "precursor_report", "window_base_rate",
]
