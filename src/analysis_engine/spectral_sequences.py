"""T4F.2: frequent sequences over timed pattern events, and whether the gap between them recurs.

T4F.1 built the substrate and deliberately counted nothing. This module does the counting the
phase was built for -- `A4 -> A8 -> B8 -> C16` as an ordered chain with a support and a
confidence attached -- and stops before the question T4F.3 asks. Nothing here is a null test, a
p-value, a base rate, a lift or a precursor claim. A sequence that occurs often in one record is
a frequent sequence in one record.

**A transition needs a declared window, because without one every pair in the record is one.**
`TransitionWindow` names a minimum and a maximum lag in the grid's own unit. The minimum is
strictly positive, so two events on one frame can never form a step: simultaneity is not order,
and T4F.1 refused to sort it into one for the same reason. A window that admits no lag on the
grid's own lattice is refused rather than returning zeros, because a zero count from an
unreachable window measures the window and not the record.

**The denominator is the honest part.** Confidence is the fraction of antecedent occurrences that
were followed, and an occurrence belongs in that fraction only if the whole window it would have
been followed in was actually searched. Two things disqualify one, and they are counted apart:

*   `WINDOW_TRUNCATED_BY_RECORD` -- the record ended before the window did. This is the classical
    right-censoring bias, and it is not symmetric: including these occurrences drags every
    confidence down by exactly the tail of the record, so a pattern that fires late looks less
    predictive than one that fires early for a reason that is about the record's edge and not
    about the pattern.
*   `WINDOW_SPANS_UNOBSERVED_TIME` -- an instant inside the window was never read, so a
    consequent may have occurred there and gone unseen.

Completions seen at ineligible antecedents are real observations and are published as
`observed_completions_including_ineligible`; they are simply not admissible into a ratio whose
denominator they cannot join. Without a declared cadence eligibility is undecidable altogether,
and confidence is reported as `None` under `COVERAGE_UNDECLARED` rather than computed against a
denominator nobody checked. The support count survives that case, because a pair that was seen
was seen.

**Support is antimonotone under extension, and the scan uses it.** Extending a sequence lengthens
the window an antecedent must have observed, so the eligible set can only shrink, and a chain
that completes for `s + (q,)` completes for `s` by taking its prefix. Candidates are therefore
generated only by extending sequences that met the minimum, which is a pruning rule this module
can prove rather than a heuristic.

**Recurrence is a repeated gap, not a period.** `recurrence_report` tallies T4F.1's spans on the
cadence lattice, which is the finest interval the pass can resolve. A span marked
`SPANS_UNOBSERVED_TIME` is an *upper bound* on an inter-occurrence interval rather than a
measurement of one -- an unseen occurrence inside it would split it in two -- so it is excluded
from the tally and counted as excluded. The report publishes how many lattice values the record
was long enough to admit, because with few of those a repeat is expected under no structure at
all, and the reader is entitled to see the denominator of that coincidence rather than a
concentration on its own.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from src.analysis_engine.spectral_events import (
    COVERAGE_INCOMPLETE, COVERAGE_UNDECLARED, EventSeries, ObservationGrid,
    WINDOW_INCOMPLETE, WINDOW_MEASURED, WINDOW_TRUNCATED,
)
from src.analysis_engine.spectral_mining import MiningBudget, MiningBudgetExceededError
from src.core.errors import InvalidParameterError


SEQUENCE_SCHEMA = "spectral-transition-sequences/v1"
RECURRENCE_SCHEMA = "spectral-recurrence-intervals/v1"

SUPPORT_UNIT = "eligible antecedent occurrence completed within the declared window"

#: Confidence statuses. Only the first carries a number.
CONFIDENCE_MEASURED = "MEASURED"
CONFIDENCE_NO_ELIGIBLE_ANTECEDENT = "NO_ELIGIBLE_ANTECEDENT"
CONFIDENCE_UNDECIDABLE = COVERAGE_UNDECLARED

#: Sequence statuses.
SEQUENCE_SUPPORTED = "SUPPORTED"
SEQUENCE_BELOW_MINIMUM = "BELOW_MINIMUM_SUPPORT"

#: Recurrence statuses.
RECURRENCE_TOO_FEW_SPANS = "TOO_FEW_MEASURED_SPANS"
RECURRENCE_UNMEASURED = "NO_MEASURED_SPAN"
RECURRENCE_NO_REPEAT = "NO_REPEATED_INTERVAL"
RECURRENCE_REPEATS = "INTERVAL_REPEATS"

SEQUENCE_CLAIM_BOUNDARY = (
    "support is a count of antecedent occurrences that were followed within a declared window "
    "and whose window was wholly searched; confidence is that count over those occurrences. "
    "Neither is a base rate, a lift, a surrogate comparison, a p-value, a precursor or a cause; "
    "those begin at T4F.3")

RECURRENCE_CLAIM_BOUNDARY = (
    "a repeated interval is a count of inter-occurrence gaps that agree to the cadence in this "
    "record. It is not a period, a frequency, an oscillation or evidence of one: no null was "
    "drawn and repeats are expected among few admissible lattice values under no structure at "
    "all. Whether a concentration exceeds chance is T4F.3's question")


@dataclass(frozen=True)
class TransitionWindow:
    """The lags at which one occurrence may be said to follow another, in the grid's unit."""

    minimum_lag: float
    maximum_lag: float
    time_units: str

    def __post_init__(self) -> None:
        if not isinstance(self.time_units, str) or not self.time_units.strip():
            raise InvalidParameterError(
                "TransitionWindow.time_units", self.time_units,
                "a non-empty unit name matching the grid's. A lag without a unit is not a lag")
        for name, value in (("minimum_lag", self.minimum_lag),
                            ("maximum_lag", self.maximum_lag)):
            if not math.isfinite(float(value)):
                raise InvalidParameterError(
                    "TransitionWindow.%s" % name, value, "a finite lag")
        if float(self.minimum_lag) <= 0.0:
            raise InvalidParameterError(
                "TransitionWindow.minimum_lag", self.minimum_lag,
                "a strictly positive lag. A zero minimum would let two events on one frame form "
                "a step, and a succession taken from simultaneity is the arrow this phase exists "
                "not to fabricate")
        if float(self.maximum_lag) < float(self.minimum_lag):
            raise InvalidParameterError(
                "TransitionWindow.maximum_lag", self.maximum_lag,
                "a maximum lag at or above the minimum (%s)" % self.minimum_lag)

    def admits(self, lag: float) -> bool:
        return float(self.minimum_lag) <= float(lag) <= float(self.maximum_lag)

    def describe(self) -> Dict[str, Any]:
        return {"minimum_lag": float(self.minimum_lag), "maximum_lag": float(self.maximum_lag),
                "time_units": self.time_units}


def _check_window_against_grid(window: TransitionWindow, grid: ObservationGrid) -> None:
    if window.time_units != grid.time_units:
        raise InvalidParameterError(
            "window.time_units", window.time_units,
            "the unit the grid declares (%r). A lag in one unit against frames in another "
            "would make every transition a mixture of the two" % grid.time_units)
    if grid.cadence is None:
        return
    cadence = float(grid.cadence)
    steps = max(int(math.ceil(float(window.minimum_lag) / cadence - 1e-9)), 1)
    if steps * cadence > float(window.maximum_lag) + 1e-9:
        raise InvalidParameterError(
            "window", window.describe(),
            "a window admitting at least one lag on the grid's own cadence of %s %s. No frame "
            "can fall between %s and %s, so every count this window returned would be a zero "
            "that measures the window rather than the record"
            % (cadence, grid.time_units, window.minimum_lag, window.maximum_lag))


@dataclass(frozen=True)
class SequenceSupport:
    """One candidate sequence, what it was counted over, and what it could not be counted over."""

    sequence: Tuple[int, ...]
    status: str
    support: int
    eligible_antecedents: int
    antecedent_occurrences: int
    observed_completions_including_ineligible: int
    ineligible_truncated: int
    ineligible_unobserved: int
    confidence: Optional[float]
    confidence_status: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    def describe(self) -> Dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "length": self.length,
            "status": self.status,
            "support": self.support,
            "support_unit": SUPPORT_UNIT,
            "eligible_antecedents": self.eligible_antecedents,
            "antecedent_occurrences": self.antecedent_occurrences,
            "observed_completions_including_ineligible":
                self.observed_completions_including_ineligible,
            "ineligible_truncated_by_record": self.ineligible_truncated,
            "ineligible_unobserved_time": self.ineligible_unobserved,
            "confidence": self.confidence,
            "confidence_status": self.confidence_status,
        }


@dataclass(frozen=True)
class SequenceMiningResult:
    """Every candidate the scan reached, with the budget and window it reached them under."""

    sequences: Tuple[SequenceSupport, ...]
    window: TransitionWindow
    minimum_support: int
    maximum_length: int
    budget: MiningBudget
    grid: ObservationGrid
    n_events: int
    candidates_examined: int
    candidates_pruned_by_antimonotonicity: int
    elapsed_seconds_unasserted: float

    @property
    def supported(self) -> Tuple[SequenceSupport, ...]:
        return tuple(item for item in self.sequences if item.status == SEQUENCE_SUPPORTED)

    def __len__(self) -> int:
        return len(self.supported)

    def of_length(self, length: int) -> Tuple[SequenceSupport, ...]:
        return tuple(item for item in self.supported if item.length == int(length))

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": SEQUENCE_SCHEMA,
            "algorithm": ("breadth-first extension of sequences meeting the minimum, pruned by "
                          "the antimonotonicity of support under extension"),
            "window": self.window.describe(),
            "grid": self.grid.describe(),
            "n_events": self.n_events,
            "minimum_support": self.minimum_support,
            "maximum_length": self.maximum_length,
            "support_unit": SUPPORT_UNIT,
            "eligibility_basis": (
                "an antecedent counts in a denominator only if the whole window it could have "
                "been followed in was searched; one the record ended before is truncated rather "
                "than unfollowed, and one containing an unread instant is unknown rather than "
                "empty"),
            "budget": self.budget.describe(),
            "candidates_examined": self.candidates_examined,
            "candidates_pruned_by_antimonotonicity":
                self.candidates_pruned_by_antimonotonicity,
            "supported_count": len(self.supported),
            "elapsed_seconds_unasserted": self.elapsed_seconds_unasserted,
            "sequences": [item.describe() for item in self.sequences],
            "claim_boundary": SEQUENCE_CLAIM_BOUNDARY,
        }


def _times_by_pattern(series: EventSeries) -> Dict[int, List[float]]:
    times: Dict[int, List[float]] = {}
    for event in series:
        times.setdefault(event.pattern_id, []).append(float(event.time))
    for values in times.values():
        values.sort()
    return times


def _completes(sequence: Sequence[int], anchor: float, times: Dict[int, List[float]],
               window: TransitionWindow) -> bool:
    """Does a chain start at `anchor` whose every step lands inside the window?"""
    reachable = [anchor]
    for pattern_id in sequence[1:]:
        candidates = times.get(int(pattern_id))
        if not candidates:
            return False
        found: List[float] = []
        for earlier in reachable:
            low = bisect.bisect_left(candidates, earlier + float(window.minimum_lag))
            high = bisect.bisect_right(candidates, earlier + float(window.maximum_lag))
            found.extend(candidates[low:high])
        if not found:
            return False
        reachable = sorted(set(found))
    return True


def _count(sequence: Tuple[int, ...], series: EventSeries, times: Dict[int, List[float]],
           window: TransitionWindow, minimum_support: int) -> SequenceSupport:
    grid = series.grid
    anchors = times.get(sequence[0], [])
    reach = (len(sequence) - 1) * float(window.maximum_lag)
    support = 0
    eligible = 0
    completions = 0
    truncated = 0
    unobserved = 0
    for anchor in anchors:
        status, _observed, _missing = grid.observation_of_window(
            anchor + float(window.minimum_lag), anchor + reach)
        completed = _completes(sequence, anchor, times, window)
        if completed:
            completions += 1
        if status == WINDOW_MEASURED:
            eligible += 1
            if completed:
                support += 1
        elif status == WINDOW_TRUNCATED:
            truncated += 1
        elif status == WINDOW_INCOMPLETE:
            unobserved += 1

    if grid.cadence is None:
        confidence: Optional[float] = None
        confidence_status = CONFIDENCE_UNDECIDABLE
    elif eligible == 0:
        confidence = None
        confidence_status = CONFIDENCE_NO_ELIGIBLE_ANTECEDENT
    else:
        confidence = support / float(eligible)
        confidence_status = CONFIDENCE_MEASURED

    status = SEQUENCE_SUPPORTED if support >= minimum_support else SEQUENCE_BELOW_MINIMUM
    return SequenceSupport(
        sequence=tuple(int(item) for item in sequence), status=status, support=support,
        eligible_antecedents=eligible, antecedent_occurrences=len(anchors),
        observed_completions_including_ineligible=completions,
        ineligible_truncated=truncated, ineligible_unobserved=unobserved,
        confidence=confidence, confidence_status=confidence_status)


def mine_frequent_sequences(
        series: EventSeries, *, window: TransitionWindow, minimum_support: int,
        maximum_length: int, budget: MiningBudget,
        clock: Callable[[], float] = time.monotonic) -> SequenceMiningResult:
    """Count ordered chains of pattern occurrences, or refuse the whole sweep."""
    if not isinstance(series, EventSeries):
        raise InvalidParameterError(
            "series", type(series).__name__, "a T4F.1 EventSeries on a declared grid")
    if not isinstance(window, TransitionWindow):
        raise InvalidParameterError(
            "window", type(window).__name__,
            "an explicit TransitionWindow. Without declared lags every pair of occurrences "
            "anywhere in the record would be a transition")
    if isinstance(minimum_support, bool) or int(minimum_support) != minimum_support \
            or minimum_support < 1:
        raise InvalidParameterError(
            "minimum_support", minimum_support, "a positive integer occurrence count")
    if isinstance(maximum_length, bool) or int(maximum_length) != maximum_length \
            or maximum_length < 2:
        raise InvalidParameterError(
            "maximum_length", maximum_length,
            "a sequence length of at least two. A length-one sequence is an occurrence count, "
            "which T4E.4 already reports, and calling it a transition would name one event a "
            "succession")
    if not isinstance(budget, MiningBudget):
        raise InvalidParameterError(
            "budget", type(budget).__name__, "an explicit MiningBudget with both hard limits")
    _check_window_against_grid(window, series.grid)

    started = float(clock())
    times = _times_by_pattern(series)
    pattern_ids = tuple(sorted(times))

    results: List[SequenceSupport] = []
    examined = 0
    pruned = 0
    frontier: List[Tuple[int, ...]] = [(item,) for item in pattern_ids]
    for length in range(2, int(maximum_length) + 1):
        candidates = [prefix + (item,) for prefix in frontier for item in pattern_ids]
        if examined + len(candidates) > budget.max_candidates:
            raise MiningBudgetExceededError(
                "candidate-count", examined + len(candidates), budget.max_candidates)
        survivors: List[Tuple[int, ...]] = []
        for candidate in candidates:
            elapsed = float(clock()) - started
            if elapsed > budget.max_seconds:
                raise MiningBudgetExceededError("wall-clock", elapsed, budget.max_seconds)
            counted = _count(candidate, series, times, window, int(minimum_support))
            examined += 1
            results.append(counted)
            if counted.status == SEQUENCE_SUPPORTED:
                survivors.append(candidate)
        if length < int(maximum_length):
            # Support cannot rise under extension, so each candidate that missed the minimum
            # takes its whole subtree with it: the extensions it would have had are never
            # built. Counting the prefixes instead of the candidates would understate that by
            # the width of the level, and agree with the truth only at the first one.
            pruned += (len(candidates) - len(survivors)) * len(pattern_ids)
        frontier = survivors
        if not frontier:
            break

    elapsed = float(clock()) - started
    if elapsed > budget.max_seconds:
        raise MiningBudgetExceededError("wall-clock", elapsed, budget.max_seconds)
    results.sort(key=lambda item: (item.length, item.sequence))
    return SequenceMiningResult(
        sequences=tuple(results), window=window, minimum_support=int(minimum_support),
        maximum_length=int(maximum_length), budget=budget, grid=series.grid,
        n_events=len(series), candidates_examined=examined,
        candidates_pruned_by_antimonotonicity=pruned, elapsed_seconds_unasserted=elapsed)


@dataclass(frozen=True)
class RecurrenceInterval:
    """One pattern's inter-occurrence gaps, tallied on the lattice that can resolve them."""

    pattern_id: int
    status: str
    spans_total: int
    spans_measured: int
    spans_bounded_above_only: int
    spans_undecidable: int
    resolution: Optional[float]
    time_units: str
    modal_interval: Optional[float]
    modal_count: int
    concentration: Optional[float]
    distinct_intervals: int
    shortest: Optional[float]
    longest: Optional[float]

    def describe(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "status": self.status,
            "spans_total": self.spans_total,
            "spans_measured": self.spans_measured,
            "spans_bounded_above_only": self.spans_bounded_above_only,
            "spans_undecidable": self.spans_undecidable,
            "resolution": self.resolution,
            "time_units": self.time_units,
            "modal_interval": self.modal_interval,
            "modal_count": self.modal_count,
            "concentration": self.concentration,
            "distinct_intervals": self.distinct_intervals,
            "shortest": self.shortest,
            "longest": self.longest,
        }


@dataclass(frozen=True)
class RecurrenceReport:
    """Every pattern's gaps, with the coincidence denominator the record itself allows."""

    intervals: Tuple[RecurrenceInterval, ...]
    minimum_spans: int
    grid: ObservationGrid
    admissible_lattice_values: Optional[int]

    def __iter__(self) -> Iterator[RecurrenceInterval]:
        return iter(self.intervals)

    def for_pattern(self, pattern_id: int) -> RecurrenceInterval:
        for interval in self.intervals:
            if interval.pattern_id == int(pattern_id):
                return interval
        raise InvalidParameterError(
            "pattern_id", pattern_id, "a pattern present in this series")

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": RECURRENCE_SCHEMA,
            "minimum_spans": self.minimum_spans,
            "grid": self.grid.describe(),
            "admissible_lattice_values": self.admissible_lattice_values,
            "coincidence_basis": (
                "the record is long enough to admit %s distinct interval values at its own "
                "cadence; a repeat among that many is the denominator of any concentration "
                "reported here" % self.admissible_lattice_values),
            "exclusion_basis": (
                "a gap containing an unread instant bounds an interval from above rather than "
                "measuring it, because an unseen occurrence inside would split it in two, so it "
                "is excluded from the tally and counted as excluded"),
            "intervals": [interval.describe() for interval in self.intervals],
            "claim_boundary": RECURRENCE_CLAIM_BOUNDARY,
        }


def recurrence_report(series: EventSeries, *, minimum_spans: int) -> RecurrenceReport:
    """Tally each pattern's measured inter-occurrence gaps on the cadence lattice."""
    if not isinstance(series, EventSeries):
        raise InvalidParameterError(
            "series", type(series).__name__, "a T4F.1 EventSeries on a declared grid")
    if isinstance(minimum_spans, bool) or int(minimum_spans) != minimum_spans \
            or minimum_spans < 2:
        raise InvalidParameterError(
            "minimum_spans", minimum_spans,
            "at least two measured spans. One gap between two occurrences cannot recur; "
            "calling it a recurrence interval would name a single distance a repetition")

    grid = series.grid
    cadence = None if grid.cadence is None else float(grid.cadence)
    lattice = None if cadence is None else int(math.floor(grid.span / cadence))

    intervals: List[RecurrenceInterval] = []
    for pattern_id in series.pattern_ids:
        spans = series.spans(pattern_id)
        measured: List[float] = []
        bounded = 0
        undecidable = 0
        for span in spans:
            if span.coverage == COVERAGE_UNDECLARED:
                undecidable += 1
            elif span.coverage == COVERAGE_INCOMPLETE:
                bounded += 1
            else:
                measured.append(float(span.duration))

        counts: Dict[int, int] = {}
        for duration in measured:
            steps = int(round(duration / cadence)) if cadence else 0
            counts[steps] = counts.get(steps, 0) + 1

        if not measured:
            status = RECURRENCE_UNMEASURED
        elif len(measured) < int(minimum_spans):
            status = RECURRENCE_TOO_FEW_SPANS
        elif max(counts.values()) < 2:
            status = RECURRENCE_NO_REPEAT
        else:
            status = RECURRENCE_REPEATS

        if status == RECURRENCE_REPEATS:
            best = max(counts.items(), key=lambda item: (item[1], -item[0]))
            modal_interval: Optional[float] = best[0] * cadence
            modal_count = best[1]
            concentration: Optional[float] = modal_count / float(len(measured))
        else:
            modal_interval = None
            modal_count = 0
            concentration = None

        intervals.append(RecurrenceInterval(
            pattern_id=int(pattern_id), status=status, spans_total=len(spans),
            spans_measured=len(measured), spans_bounded_above_only=bounded,
            spans_undecidable=undecidable, resolution=cadence, time_units=grid.time_units,
            modal_interval=modal_interval, modal_count=modal_count,
            concentration=concentration, distinct_intervals=len(counts),
            shortest=min(measured) if measured else None,
            longest=max(measured) if measured else None))

    return RecurrenceReport(intervals=tuple(intervals), minimum_spans=int(minimum_spans),
                            grid=grid, admissible_lattice_values=lattice)


__all__ = [
    "SEQUENCE_SCHEMA", "RECURRENCE_SCHEMA", "SUPPORT_UNIT",
    "SEQUENCE_CLAIM_BOUNDARY", "RECURRENCE_CLAIM_BOUNDARY",
    "CONFIDENCE_MEASURED", "CONFIDENCE_NO_ELIGIBLE_ANTECEDENT", "CONFIDENCE_UNDECIDABLE",
    "SEQUENCE_SUPPORTED", "SEQUENCE_BELOW_MINIMUM",
    "RECURRENCE_TOO_FEW_SPANS", "RECURRENCE_UNMEASURED", "RECURRENCE_NO_REPEAT",
    "RECURRENCE_REPEATS",
    "TransitionWindow", "SequenceSupport", "SequenceMiningResult",
    "RecurrenceInterval", "RecurrenceReport",
    "mine_frequent_sequences", "recurrence_report",
]
