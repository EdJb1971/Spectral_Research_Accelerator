"""The five refusals (`src/core/refusal.py`, TG4.2, `ed-dev`).

The proposal's validation strategy names five things this engine has to be able to do, and
four of them are refusals. It must recover a planted relationship without being told where it
is; **reject** a convincing but artificial correlation; **distinguish** independent
observations from autocorrelated repetitions; **avoid** a motif that belongs to the
representation rather than to the world; and **report nothing** when there is nothing. TG4.1
built the recovery and one of the refusals. This module builds the other three, and it builds
them the only way that means anything: by first constructing a record on which the engine as
it stood **produced the false discovery**, and then declaring the rule that refuses it.

Two of the three traps were holes, not demonstrations. That is written down here because a
slice that only confirmed what the previous slice already did would be worth nothing.

**A single band, read through a redundant transform, manufactures four relationships.** The
trap is one modulation exciting one spatial band, with no second process anywhere in the
record. A stationary wavelet spreads that band's energy across every level, so every level's
series is a smeared copy of one process, and because the process has memory the copies stay
correlated at a lag. TG4.1's basis control does not catch it: that control excites both bands
independently and asks which *pairs* the transform confuses, and it reports levels 1 and 4 as
separable at 0.023 - which they are, when both are excited. Run the TG4.1 pipeline unchanged
on the single-band record and it confirms **all four admissible members at q = 0.0104, at
every one of five root seeds**, each of them at the shortest admissible lag. Nothing was
planted between two bands. The engine reported four relationships.

**The rule that refuses it is a ceiling, not a threshold.** Instantaneous leakage of one
process into two bands produces `r(k) = r(0) * rho(k)` where `rho` is the process's own
autocorrelation - maximal at zero, decaying from there, and never larger than the
*simultaneous* correlation `r(0)`. So a lead is claimable only when it is **stronger than the
simultaneous relationship it might be a smeared copy of**. There is nothing to tune: `r(0)` is
measured on the same partition as the candidate, `rho <= 1` makes it the supremum of what a
leak can produce, and the lag the family may never contain becomes the reference every member
is measured against. On the single-band record **0 of 48 members survive** at all five seeds -
the family is emptied, which is the correct answer and a different answer from "nothing was
significant". On the planted record the true member survives at all five seeds and confirms at
q = 0.0050, unchanged: the ceiling costs the true finding nothing, because a real lead is
stronger at its lag than at zero.

This is selection on the training partition and it is applied there only, before the seal.
It does not narrow the declared family - 48 members are priced and 48 are measured - and it
never runs on the held-out partition, where the lag is frozen and no choice remains.

**A shared cycle defeats the surrogate that was supposed to be enough.** The second trap is
two independently modulated bands that both carry one deterministic cycle, the fine band's
crest arriving three frames before the coarse band's. Nothing is coupled. TG4.1's
circular-shift surrogate ought to be the classical defence, because a shifted copy of a
periodic series is still periodic - and it is not enough: the shift moves the phase, most
shifts misalign the crests, and the observed alignment still looks surprising. Measured over
five root seeds, the unguarded pipeline **confirms three or four of the four members at every
one of them**, at q of 0.0104 or better, with the strongest member carrying a naive p-value
between 1.7e-71 and 6.4e-54 and an ESS-corrected one between 3.6e-13 and 2.8e-10. Neither
rule R12 nor the surrogate refuses this.

**The calendar is metadata, not a discovery.** What refuses it is rule R11 lifted into this
pipeline: the periods a record can carry are computed from its own cadence - a day is 24
frames at hourly cadence whatever the data does - fitted as harmonics **on the training
partition** and subtracted from both partitions on one shared clock, and `carry_forward`
refuses to hand a single candidate on to a freeze from a record that still has its calendar in
it. With that removal in place the same five seeds confirm nothing, the smallest corrected q
being 0.105.

The one place this refusal has been seen to leak is recorded rather than tuned away. On a
longer record - 384 frames with a 24-frame cycle - the residual confirms a lag-1 relationship
at one root seed in five, and it does so at one, two and three harmonics alike, which is what
says it is a false positive of a 0.05 test rather than an unremoved cycle.

Detecting the period from the data instead was tried and measured, and it does not work at
this record length. Fitting one harmonic per candidate period on the first half of the
training partition and scoring it by the variance it removes from the second half - the
strictest of the four variants tried - separates a real cycle at 0.13 to 0.58 from a
red-noise fluke at up to 0.29, over ten seeds and five record types. The distributions
overlap, so a detector would either miss real calendars or invent them. The clock does not
overlap with anything.

**And what that leaves undefended is said out loud.** A periodic confound at a period the
clock does not name is not refused by anything in this module, and the measurement above is
exactly the measurement of how badly it would go: the engine confirms it. The refusal here is
of the *calendar*, which is the confound this programme will actually meet, and not of
periodicity in general.

**The third trap was already refused, and the benchmark exists to show by how much.** Two
independent bands at `phi = 0.95` over 288 frames confirm nothing at all five seeds, while
between 22 and 48 of the 48 members carry a naive p-value below 0.05 and as many as 20 carry
an ESS-corrected one below 0.05. On a record drawn the same way the winner has been seen at
`level_2>level_4@5`, `r = 0.48` - the same label, at the same lag, that the planted
benchmark confirms, on two bands that were never related at all. What
refuses it is the held-out surrogate, and the value of the benchmark is the size of the gap
between the number a naive reading would have reported and the number this pipeline reports.

**The register is machinery, not a list in a document.** `REFUSALS` names all five, the gate
that carries each and the benchmark that gate runs on, and `refusal_coverage` checks that
against the live benchmark registry: five benchmarks, at least three of them nulls, every
named gate declared by the benchmark that claims it. TG4.2's acceptance criterion is therefore
computed rather than asserted, and a refusal whose benchmark is deleted stops being a refusal
that day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.precedence import (
    PrecedenceCandidate,
    Record,
    ScaleSeries,
    SweepResult,
    _fisher_error,
    _fisher_z,
    affordable_member_count,
    deduplicate_candidates,
    lagged_correlation,
    pair_label,
)

#: Seconds in the two cycles a geophysical record is always exposed to. There is no third
#: entry: a period this list does not name is not refused by anything here, and pretending
#: otherwise by adding "likely" periods would be the detector this module measured and
#: rejected, wearing a hard-coded disguise.
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
CALENDAR_SECONDS: Tuple[Tuple[str, float], ...] = (
    ("diurnal", SECONDS_PER_DAY),
    ("annual", SECONDS_PER_YEAR),
)

#: How many complete turns of a cycle a record must contain before that cycle is removable.
#: Below this the harmonic fit is interpolating a trend rather than estimating a cycle, and
#: subtracting it would remove signal the record never separated from it. A record too short
#: to see a cycle twice does not have that cycle *removed*; it has it declared unremovable,
#: which is the honest state and is recorded as such.
MIN_CYCLES_TO_REMOVE = 2.0

#: Harmonics per calendar period. One, and the count was measured rather than assumed: the
#: calendar does not reach a band's energy series as a sinusoid - it modulates an amplitude
#: that is read as an energy in logs - so more harmonics looked likely to help, and on 384
#: frames at a 24-frame cycle two and three of them removed *less* of the spurious lead than
#: one did, not more. Every additional harmonic is another parameter fitted on the training
#: partition and subtracted from the held-out one, so the cheapest model that works is the
#: one that ships.
CALENDAR_HARMONICS = 1


class EverythingRefusedError(InvalidParameterError):
    """Every member of the declared family was explained by something other than the world.

    This is a result, and it is not the same result as "nothing was significant". The family
    was declared, the sweep ran, and each member was refused for a stated reason before any
    ensemble was consulted. A study that reaches this state has learned that its record
    cannot carry the question it was asked, which is worth reporting.
    """

    def __init__(self, n_members: int, reason: str) -> None:
        super().__init__(
            "family", n_members,
            "at least one member the record does not already explain. All %d were refused: "
            "%s. This is a refusal of the family, not a null result about the world"
            % (n_members, reason))


class CalendarNotRemovedError(InvalidParameterError):
    """A record long enough to carry a calendar cycle was swept with the cycle still in it."""

    def __init__(self, record: str, periods: Sequence[str]) -> None:
        super().__init__(
            "record", record,
            "a record whose calendar has been removed. This one can carry %s and its "
            "provenance does not record their removal, so any relationship found in it may "
            "be the calendar's (rule R11). Call `remove_calendar` on the partitions before "
            "carrying a candidate forward" % (", ".join(periods),))


# ------------------------------------------------------------------- the leakage ceiling


@dataclass(frozen=True)
class LagProfile:
    """One ordered band pair's strength as a function of lag, starting at lag zero.

    Index 0 is the *simultaneous* correlation, which is not a member of any precedence family
    and is the whole point: it is the reference, not a hypothesis. The profile is measured
    over every lag up to `max_lag` whether or not the family admits them, because a shape is
    not a claim and an admissibility rule that shortens the shape would blind the reference.
    """

    driver: str
    driven: str
    strengths: Tuple[float, ...]

    @property
    def simultaneous(self) -> float:
        return self.strengths[0]

    @property
    def peak_lag(self) -> int:
        return int(max(range(len(self.strengths)), key=lambda k: self.strengths[k]))

    def at(self, lag: int) -> float:
        if not 0 <= lag < len(self.strengths):
            raise InvalidParameterError(
                "lag", lag, "a lag this profile covers (0 to %d)" % (len(self.strengths) - 1))
        return self.strengths[lag]

    def rises_above_simultaneous(self, lag: int) -> bool:
        """The leakage ceiling: is this lead stronger than the same-frame relationship?"""
        return self.at(lag) > self.simultaneous

    def describe(self) -> Dict[str, Any]:
        return {"pair": pair_label((self.driver, self.driven)),
                "simultaneous": float(self.simultaneous),
                "peak_lag": self.peak_lag,
                "strengths": [float(v) for v in self.strengths]}


def simultaneous_strength(record: Record, driver: str, driven: str) -> float:
    """The two bands' correlation within the same frame - lag zero, the forbidden member."""
    a, b = record.array(driver), record.array(driven)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def lag_profile(record: Record, driver: str, driven: str, *, max_lag: int) -> LagProfile:
    """Strength at lags 0 through `max_lag`, on one partition."""
    if max_lag < 1:
        raise InvalidParameterError(
            "max_lag", max_lag, "at least one lag beyond the simultaneous reference")
    a, b = record.array(driver), record.array(driven)
    strengths = [simultaneous_strength(record, driver, driven)]
    strengths.extend(abs(lagged_correlation(a, b, k)) for k in range(1, int(max_lag) + 1))
    return LagProfile(driver=driver, driven=driven, strengths=tuple(strengths))


def explained_by_leakage(record: Record, candidate: PrecedenceCandidate) -> bool:
    """True when instantaneous leakage plus memory could have produced this member.

    One process leaking into two bands gives `r(k) = r(0) * rho(k)` with `rho <= 1`, so
    `r(0)` is the ceiling of everything a leak can manufacture at any lag. A member at or
    below it is indistinguishable from one band read twice; a member above it is not, whatever
    else it may turn out to be.
    """
    profile = lag_profile(record, candidate.driver, candidate.driven, max_lag=candidate.lag)
    return not profile.rises_above_simultaneous(candidate.lag)


def refuse_leaked_members(
        record: Record,
        candidates: Sequence[PrecedenceCandidate],
) -> Tuple[Tuple[PrecedenceCandidate, ...], Tuple[PrecedenceCandidate, ...]]:
    """Split a sweep's candidates into those that build and those a leak explains."""
    kept: List[PrecedenceCandidate] = []
    refused: List[PrecedenceCandidate] = []
    for candidate in candidates:
        (refused if explained_by_leakage(record, candidate) else kept).append(candidate)
    return tuple(kept), tuple(refused)


# ------------------------------------------------------------------------- the calendar


def calendar_periods(record: Record) -> Tuple[Tuple[str, float], ...]:
    """The cycles this record's own clock says it is exposed to, in frames.

    Computed from `cadence_seconds` and the record's length and nothing else. A record of
    hourly frames carries a 24-frame day whatever its values look like, which is why this
    needs no threshold and produces no false positives: it is a fact about when the record was
    sampled, established before a single value is read.
    """
    out: List[Tuple[str, float]] = []
    for name, seconds in CALENDAR_SECONDS:
        frames = float(seconds) / float(record.cadence_seconds)
        if frames < 2.0:
            continue                       # faster than the sampling; not resolvable at all
        if record.length / frames < MIN_CYCLES_TO_REMOVE:
            continue                       # too short to see it turn twice
        out.append((name, frames))
    return tuple(out)


def _harmonic_design(times: np.ndarray, periods: Sequence[float],
                     harmonics: int = CALENDAR_HARMONICS) -> np.ndarray:
    columns = [np.ones_like(times)]
    for period in periods:
        for h in range(1, int(harmonics) + 1):
            columns.append(np.cos(2.0 * math.pi * h * times / float(period)))
            columns.append(np.sin(2.0 * math.pi * h * times / float(period)))
    return np.column_stack(columns)


def _absolute_times(record: Record) -> np.ndarray:
    """Frame indices in the parent record's numbering, so two partitions share one clock."""
    start = int(record.frames[0])
    return np.arange(start, start + record.length, dtype=np.float64)


def remove_calendar(train: Record, *records: Record,
                    harmonics: int = CALENDAR_HARMONICS) -> Tuple[Record, ...]:
    """Fit the calendar on the training partition, subtract it from every partition given.

    Fitted on train and applied elsewhere, never refitted (rules R6 and R11): a climatology
    fitted on all the frames has seen the held-out ones, and a held-out partition whose own
    mean has been removed using itself is not held out. The two partitions are placed on one
    clock through their absolute frame numbers, so the phase the training partition estimated
    is the phase subtracted from the held-out one - which is what makes the removal testable
    rather than cosmetic.
    """
    periods = calendar_periods(train)
    if not periods:
        raise InvalidParameterError(
            "train", train.name,
            "a record long enough to carry a calendar cycle at its own cadence. This one is "
            "not, so there is no calendar to remove and none to refuse")
    values = [p for _, p in periods]
    design = _harmonic_design(_absolute_times(train), values, harmonics)
    coefficients = {
        level: np.linalg.lstsq(design, train.array(level), rcond=None)[0]
        for level in train.levels
    }
    out: List[Record] = []
    for record in (train,) + records:
        matrix = _harmonic_design(_absolute_times(record), values, harmonics)
        series = tuple(
            ScaleSeries(level, record.array(level) - matrix @ coefficients[level])
            for level in record.levels)
        provenance = dict(record.provenance)
        provenance["calendar_removed"] = [name for name, _ in periods]
        provenance["calendar_periods_frames"] = [float(p) for p in values]
        provenance["calendar_fitted_on"] = train.name
        provenance["calendar_harmonics"] = int(harmonics)
        out.append(Record(name=record.name, series=series,
                          cadence_seconds=record.cadence_seconds, frames=record.frames,
                          provenance=provenance))
    return tuple(out)


def require_calendar_removed(record: Record) -> None:
    """Refuse a record that can carry a calendar and does not record having lost it."""
    periods = calendar_periods(record)
    if not periods:
        return
    removed = record.provenance.get("calendar_removed")
    missing = [name for name, _ in periods if not removed or name not in removed]
    if missing:
        raise CalendarNotRemovedError(record.name, missing)


# ------------------------------------------------------- frames are not observations


def naive_versus_effective(result: SweepResult, alpha: float = 0.05) -> Dict[str, Any]:
    """How many members a naive frame count would have called significant, and how many do.

    Neither number is a finding. The gap between them is the size of rule R12 on this record,
    and a benchmark that reports the corrected number without the naive one has not shown
    that the correction did anything.
    """
    naive = [c for c in result.candidates if c.p_naive < alpha]
    effective = [c for c in result.candidates if c.p_effective < alpha]
    ratios = [c.n_effective / c.n_pairs for c in result.candidates if c.n_pairs]
    return {
        "alpha": float(alpha),
        "n_members": result.n_examined,
        "naive_significant": len(naive),
        "effective_significant": len(effective),
        "median_ess_fraction": float(np.median(ratios)) if ratios else 1.0,
        "claim_boundary": ("Uncorrected counts over a declared family. Both are quantities "
                           "about the sweep, and neither is a result about the record"),
    }


# ------------------------------------------------------------------------ the gatekeeper


def carry_forward(result: SweepResult, record: Record, *,
                  n_candidates: Optional[int] = None) -> Tuple[PrecedenceCandidate, ...]:
    """What the generate stage may hand to a freeze, after the refusals have had their say.

    The order matters. The calendar is refused first, because a record that still contains it
    has no trustworthy candidate to rank; then the leakage ceiling, which is a property of
    each member; then TG4.1's own two rules - one member per ordered pair, and only those the
    training partition could not tell apart from the winner.

    This is the single door between a sweep and a seal, which is where a refusal has to live
    if it is to be machinery rather than discipline: there is no path from a swept record to
    a frozen claim that does not pass through here.
    """
    require_calendar_removed(record)
    kept, refused = refuse_leaked_members(record, result.ranked)
    if not kept:
        raise EverythingRefusedError(
            result.n_examined,
            "every member is at or below its own simultaneous correlation, so a single "
            "process leaking into two bands would produce all of them")
    pool = deduplicate_candidates(kept)
    ceiling = affordable_member_count(
        result.specification.n_surrogates, alpha=result.specification.alpha,
        correction=result.specification.correction)
    wanted = ceiling if n_candidates is None else int(n_candidates)
    if wanted < 1:
        raise InvalidParameterError(
            "n_candidates", wanted,
            "at least one member to carry forward. A confirmatory family of none is a study "
            "that cannot report anything, which is not the same as a null result")
    if wanted > ceiling:
        raise InvalidParameterError(
            "n_candidates", wanted,
            "at most the %d members %d surrogates can reject one of after %s correction at "
            "alpha %g" % (ceiling, result.specification.n_surrogates,
                          result.specification.correction, result.specification.alpha))
    if n_candidates is None:
        best = pool[0]
        error = _fisher_error(best.n_effective)
        top = _fisher_z(best.strength)
        pool = tuple(c for c in pool if top - _fisher_z(c.strength) <= error)
    return tuple(pool[:min(wanted, len(pool))])


def refusal_report(result: SweepResult, record: Record) -> Dict[str, Any]:
    """What the refusals did to this sweep, whether or not anything survived them."""
    kept, refused = refuse_leaked_members(record, result.ranked)
    strongest = refused[0] if refused else None
    return {
        "n_members": result.n_examined,
        "n_survived": len(kept),
        "n_refused_as_leakage": len(refused),
        "family_emptied": not kept,
        "strongest_refused": strongest.describe() if strongest is not None else None,
        "calendar_periods": [name for name, _ in calendar_periods(record)],
        "calendar_removed": list(record.provenance.get("calendar_removed", ())),
        "reading": ("Members refused because their strength does not exceed the same pair's "
                    "simultaneous correlation, which is the ceiling of what leakage of one "
                    "process into two bands can produce"),
    }


# --------------------------------------------------------------------- the five refusals


@dataclass(frozen=True)
class Refusal:
    """One of the five things the engine must be able to do, and what carries it."""

    name: str
    must: str
    gate: str
    benchmark: str
    is_null: bool
    carried_by: Tuple[str, ...]

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "must": self.must, "gate": self.gate,
                "benchmark": self.benchmark, "is_null": self.is_null,
                "carried_by": list(self.carried_by)}


REFUSALS: Tuple[Refusal, ...] = (
    Refusal(
        name="unknown_location",
        must="recover a planted relationship without being told its scale or its lag",
        gate="4F.precedence_recovery",
        benchmark="planted_precedence",
        is_null=False,
        carried_by=("precedence.precedence_search_specification", "precedence.sweep",
                    "precedence.freeze_precedence", "precedence.confirm_precedence"),
    ),
    Refusal(
        name="artificial_correlation",
        must="reject a convincing correlation manufactured by a cycle both bands follow",
        gate="4F.refusal_calendar",
        benchmark="shared_cycle_precedence",
        is_null=True,
        carried_by=("refusal.calendar_periods", "refusal.remove_calendar",
                    "refusal.require_calendar_removed"),
    ),
    Refusal(
        name="autocorrelated_repetitions",
        must="distinguish independent observations from autocorrelated repetitions",
        gate="4F.refusal_autocorrelation",
        benchmark="slow_independent_precedence",
        is_null=True,
        carried_by=("precedence.admissible_lags", "precedence.circular_shift",
                    "refusal.naive_versus_effective"),
    ),
    Refusal(
        name="representation_artefact",
        must="avoid a relationship that belongs to the transform rather than to the world",
        gate="4F.refusal_representation",
        benchmark="leaked_band_precedence",
        is_null=True,
        carried_by=("precedence.measure_basis_coupling", "refusal.lag_profile",
                    "refusal.explained_by_leakage", "refusal.carry_forward"),
    ),
    Refusal(
        name="no_relationship",
        must="report no relationship when there is none",
        gate="4F.precedence_null",
        benchmark="precedence_null",
        is_null=True,
        carried_by=("precedence.confirm_precedence", "preregistration.confirm_on_held_out"),
    ),
)

#: TG4.2's acceptance criterion, as numbers rather than as prose.
REQUIRED_REFUSALS = 5
REQUIRED_NULLS = 3


def refusal_coverage() -> Dict[str, Any]:
    """Check the register against the live benchmark registry.

    Named benchmarks must exist, must declare the gate the refusal claims they carry, and
    must agree about whether they are nulls. Then the arithmetic the roadmap asked for: five
    refusals, at least three of them null benchmarks. A refusal whose benchmark is deleted or
    renamed stops being a refusal the moment this runs.
    """
    from src.benchmarks.core import get_benchmark            # local: core must not need it
    import src.benchmarks.sequences                          # noqa: F401  (registers them)

    problems: List[str] = []
    for refusal in REFUSALS:
        try:
            bench = get_benchmark(refusal.benchmark)
        except KeyError:
            problems.append("refusal %r names benchmark %r, which is not registered"
                            % (refusal.name, refusal.benchmark))
            continue
        if refusal.gate not in bench.gates:
            problems.append("refusal %r claims gate %r, which benchmark %r does not declare "
                            "(it declares %s)"
                            % (refusal.name, refusal.gate, bench.name, list(bench.gates)))
        if bool(bench.is_null) != bool(refusal.is_null):
            problems.append("refusal %r calls benchmark %r %s and the benchmark calls itself "
                            "%s" % (refusal.name, bench.name,
                                    "a null" if refusal.is_null else "not a null",
                                    "a null" if bench.is_null else "not a null"))
    names = [r.name for r in REFUSALS]
    if len(set(names)) != len(names):
        problems.append("two refusals share a name: %s" % (names,))
    benchmarks = [r.benchmark for r in REFUSALS]
    if len(set(benchmarks)) != len(benchmarks):
        problems.append("two refusals name the same benchmark: %s" % (benchmarks,))
    if len(REFUSALS) != REQUIRED_REFUSALS:
        problems.append("the validation strategy names %d refusals and this register holds %d"
                        % (REQUIRED_REFUSALS, len(REFUSALS)))
    n_nulls = sum(1 for r in REFUSALS if r.is_null)
    if n_nulls < REQUIRED_NULLS:
        problems.append("%d of the five benchmarks are nulls; at least %d are required, "
                        "because the load-bearing result is a null returning null"
                        % (n_nulls, REQUIRED_NULLS))
    return {
        "n_refusals": len(REFUSALS),
        "n_nulls": n_nulls,
        "benchmarks": benchmarks,
        "gates": [r.gate for r in REFUSALS],
        "problems": problems,
        "covered": not problems,
    }


__all__ = [
    "SECONDS_PER_DAY", "SECONDS_PER_YEAR", "CALENDAR_SECONDS", "CALENDAR_HARMONICS",
    "MIN_CYCLES_TO_REMOVE",
    "EverythingRefusedError", "CalendarNotRemovedError",
    "LagProfile", "simultaneous_strength", "lag_profile", "explained_by_leakage",
    "refuse_leaked_members",
    "calendar_periods", "remove_calendar", "require_calendar_removed",
    "naive_versus_effective",
    "carry_forward", "refusal_report",
    "Refusal", "REFUSALS", "REQUIRED_REFUSALS", "REQUIRED_NULLS", "refusal_coverage",
]
