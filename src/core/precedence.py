"""Recovering a relationship nobody pointed at (`src/core/precedence.py`, TG4.1, `ed-dev`).

Phase 4C's central claim is that fine-scale activity at `t` precedes coarse-scale activity at
`t + lag`, and `src/benchmarks/sequences.py` has carried a benchmark for it since T4C.3. That
benchmark's check reads `driver_level` and `driven_level` **out of the known answer** and then
takes an `argmax` over lag. It recovers the injected lag, which is worth knowing, and it is not
the claim: a search told which two bands to look at is a search of one band pair, and a study
that could only be run by someone who already knew the answer has not demonstrated recovery.

This module runs the same recovery without being told. The band pair and the lag are both
declared axes of one family, the family is priced by TG3.1 before it enumerates anything, and
what comes out of the sweep is candidates rather than findings. Nine things had to be got
right, and each of them is a way the untold version manufactures a relationship that the told
version never could.

**The decomposition relates bands to each other before the world does.** This one was found by
running the null and watching it confirm a relationship at q = 0.0075 on a record with nothing
in it. A redundant transform does not hand back independent bands, it hands back a smear, and
two levels that are largely the same band produce a beautiful lagged relationship that is a
fact about the wavelet. `measure_basis_coupling` reads that off a **control record built with
no temporal structure at all**, where any correlation between bands is the decomposition
talking about itself: measured there, levels 1 and 2 correlate at 0.99 within a frame whatever
is planted, levels 3 and 4 at 0.78 to 0.84 and levels 1 and 3 at 0.49 to 0.53, while the
largest cross-band correlation at any admissible lag is 0.06 to 0.18. The limit is not chosen -
a basis has no dynamics, so what it produces across frames is its own noise floor, and two
bands are coupled when it relates them within a frame more strongly than it relates anything
across one. Of twelve ordered pairs, four survive. Narrowing a family is normally what R18
refuses; this narrowing is admissible because it is derived from a control rather than from the
record under test, so it is fixed before the study's data is read. And note which lag does the
work: zero, the one lag `admissible_lags` may never return, because a band seen twice is seen
twice *simultaneously*.

**Being told where to look is a family of one.** The 4C check tests a single pair at a single
recovered lag. Declared honestly, the same question over `L` bands and `|lags|` admissible lags
is `L(L-1) * |lags|` members - ordered pairs, because *fine leads coarse* and *coarse leads
fine* are two different hypotheses and a search that only asks one of them was told the answer.
Four bands and twelve lags is 144 members, which TG3.1 prices at 15,985 surrogates before one
of them could be rejected; six bands and twelve lags is 360 members and 46,545, and eight bands
and twenty-four lags is 1,344 and 209,153. So this slice is a generate/confirm study for the
same reason TG3.5 was, and the gate refuses a pass if its own generate family ever becomes
affordable in one stage.

**The lag is chosen by the data, so the null has to be over the choice.** The maximum of twelve
lagged correlations is not one correlation: its null distribution is the distribution of a
maximum, and testing the winner against the null of a *single* lag prices a search of twelve at
the cost of one. This is the largest effect in the slice, and it is measured on records with
**nothing planted in them**: the winner of the sweep, tested against a single-lag null, is
significant at 0.05 in six runs out of six, and against the null of the same maximum in one out
of six. `precedence_p_value` takes `selected_over` for exactly that. It is necessary and not
sufficient - the maximum prices the choice of *lag* and not the choice of band pair, which is
why the pass ends in a corrected confirmation on a partition it did not select on, where the
same twenty null records produce **no** confirmation at all. What makes that confirmation
affordable is that the lag was **frozen on train**: a frozen lag is not a selection, so the
held-out null is over one statistic and needs no maximum. The generate/confirm split is not
only how the correction is paid here, it is how the null becomes computable.

**A surrogate must keep the autocorrelation.** The honest surrogate is a circular shift of the
driver band's series: it preserves every value, the whole autocorrelation function and the
marginal distribution exactly, and destroys only the alignment - the one thing the hypothesis is
about. Shuffling the series instead destroys the autocorrelation too, and two red series are far
more aligned at a random offset than two white ones, so a shuffled null sits closer to zero than
the truth and the observed alignment looks surprising when it is ordinary. Measured on twelve
records with nothing planted, at the same lag: at `phi = 0` the two surrogates agree exactly
(median p 0.64 and 0.65, no false positives either way), at `phi = 0.7` the shuffle's median p
has fallen to 0.34 against the shift's 0.55, and at `phi = 0.9` the shuffle rejects three times
in twelve at 0.05 where the shift rejects none and their median p-values are 0.19 and 0.59. The
error grows with exactly the quantity the surrogate discards. `permuted_series` is public and
unregistered so that this can be measured rather than asserted.

**Frames are not samples.** The modulation is red by construction, and a correlation between two
autocorrelated series has far fewer independent pairs than frames. Every candidate carries both
p-values, naive and effective-sample-size corrected, because the ratio is how much serial
dependence was inflating it (rule R12). The ESS also decides which lags are admissible at all:
a lag of `k` on a partition of `n` frames is tested on `n - k` pairs, so the far end of the lag
axis is tested with less data than the near end, and a family whose largest lag leaves fewer
than `MIN_EFFECTIVE_PAIRS` effective pairs is refused rather than quietly tested at a power
nobody declared.

**One relationship enters the family once per lag it survives at.** A driver with an
autocorrelation time of several frames correlates with its target at `k - 1`, `k` and `k + 1`,
so the top of the ranking is one relationship wearing three labels. Freezing all three is a
correction unit of three paid for one hypothesis, so `deduplicate_candidates` keeps the best lag
per ordered band pair. Which lag is *not* thereby claimed to be the true one: the confirmation
tests the lag that was frozen, and a neighbouring lag would have been a different member.

**The embargo is hygiene, and it is not what protects the confirmation.** Splitting a time
series in two and calling the second half held out looked like the temporal version of testing
on the training data - the first frames of the test half are the immediate future of the last
frames of the train half, and a red driver carries across the join - so `split_with_embargo`
drops frames between them and `freeze_precedence` refuses to seal a partition cut closer than
the record's own memory. Then the effect was measured, and on twenty records per setting whose
bands are independent but slow it is not there: at `phi = 0.9` the train winner reaches the
held-out partition at a median |r| of 0.150 with no embargo and 0.152 with one, rejecting in 1
and 2 runs of 20 respectively, and at `phi = 0.95` it is 0.161 against 0.188 and 2 against 3.
The reason is that the held-out test is a surrogate test drawn from the held-out record itself,
so that record's own autocorrelation is already in its null - the protection comes from the
surrogate, not from the gap. The refusal stays, because it costs a few frames and it is the
kind of thing that stops mattering only until the two partitions genuinely overlap, but it is
recorded here as a declared constraint whose benefit this tree has not been able to measure,
not as a fix for something that was going wrong.

**A record can be too slow to carry any lagged claim at all.** The same admissibility rule that
drops over-long lags refuses every lag when the memory is long enough: an `AR(1)` record of 360
frames at `phi = 0.98` has a measured memory of anywhere between 26 and 115 frames depending on
the realisation - the estimate is itself unreliable there, which is part of the point - and
fewer than eight effective pairs at any lag, so `admissible_lags` empties the family and says
so, before a sweep is run and before a single correlation is computed. A near-unit-root record
does not produce a weak result here; it produces a refusal.

**Direction has to be in the family, not in the analyst's head.** Correlation is symmetric, so
"fine leads coarse at lag k" and "coarse leads fine at lag k" are computed from the same two
series and differ only in which one is shifted. Both are members. Recovering the direction means
the forward member survives the correction and the reverse one does not - which is a result
about the data, where declaring the direction in advance is a result about the analyst.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.family import (
    SearchAxis,
    SearchSpecification,
    SearchTerm,
    max_affordable_family,
)
from src.core.preregistration import (
    HeldOutLedger,
    PartitionIdentity,
    Seal,
    confirm_on_held_out,
    freeze_confirmatory_family,
    report_generation,
)
from src.statistics.significance import (
    correlation_test,
    effective_sample_size,
    surrogate_p_value,
)

#: The shortest record a lagged correlation may be estimated from at all. Below this the
#: effective sample size is smaller than the number of lags being searched, and the sweep is
#: fitting the record rather than measuring it.
MIN_FRAMES = 24

#: Effective pairs a member must retain after its lag is subtracted. A lag that leaves fewer is
#: not tested at reduced power, it is refused: the family declares equal treatment of its
#: members and a member tested on a third of the data is not the same test.
MIN_EFFECTIVE_PAIRS = 8.0

#: Precedence needs a lead. Lag zero is a simultaneous correlation, which is a real thing to
#: measure and is not this thing (rule R21).
MIN_ADMISSIBLE_LAG = 1

#: The honest surrogate.
DEFAULT_SURROGATE = "circular_shift"

#: The dishonest one, kept public so the difference can be measured.
PERMUTED = "permutation"


# ------------------------------------------------------------------------------ refusals


class LagInadmissibleError(InvalidParameterError):
    """A declared lag that rule R21 does not permit a precedence reading of."""

    def __init__(self, lag: int, reason: str) -> None:
        super().__init__(
            "lag", lag,
            "a lag this record can carry a precedence reading at. %s. A lag admitted here "
            "becomes a member of the family and is corrected for, so admitting one that "
            "cannot be read as a lead spends correction power on a hypothesis that was never "
            "available" % reason)
        self.lag = int(lag)


class ShiftInfeasibleError(InvalidParameterError):
    """The record is too short to be circularly shifted clear of its own autocorrelation."""

    def __init__(self, length: int, minimum_shift: int) -> None:
        super().__init__(
            "record length", length,
            "a record long enough to shift clear of its own memory. A circular shift smaller "
            "than the autocorrelation time (%d frames here) leaves the surrogate still "
            "aligned with the original, so the null contains the alternative and every "
            "p-value drawn from it is too large; %d frames leaves no admissible shift. The "
            "answer is a longer record, not a smaller shift" % (minimum_shift, length))
        self.length = int(length)
        self.minimum_shift = int(minimum_shift)


class ControlNotStaticError(InvalidParameterError):
    """A basis control that has dynamics of its own, and so measures more than the basis."""

    def __init__(self, level: str, memory: float) -> None:
        super().__init__(
            "control", level,
            "a control record with no temporal structure at all. Band %r has a memory of "
            "%.1f frames, so this record's bands are related across frames as well as within "
            "them - and the floor read off it would be the record's own dynamics rather than "
            "the basis's noise. The control exists to isolate what the decomposition relates "
            "when nothing else does; one with a memory cannot do that, and a lower floor "
            "would hide the problem rather than fix it" % (level, memory))
        self.level = str(level)
        self.memory = float(memory)


class EmbargoTooShortError(InvalidParameterError):
    """A held-out partition cut too close to the data the candidates were mined from."""

    def __init__(self, embargo: int, recommended: int, name: str) -> None:
        super().__init__(
            "embargo", embargo,
            "at least the %d frames this record's own autocorrelation time asks for. The "
            "held-out partition %r begins %d frames after the training data ends, so its "
            "first frames are the immediate future of the last frames the candidates were "
            "selected on. Measured on this tree's own records the gap changes nothing - the "
            "held-out null is drawn from the held-out record and already carries its memory "
            "- so this is a declared constraint rather than a repair, and it is refused "
            "here because a study that quietly narrows its own gap has changed what its "
            "seal means" % (recommended, name, embargo))
        self.embargo = int(embargo)
        self.recommended = int(recommended)


# ------------------------------------------------------------------------------- records


@dataclass(frozen=True)
class ScaleSeries:
    """One band's activity as a function of time."""

    level: str
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", str(self.level))
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if not self.level.strip():
            raise InvalidParameterError(
                "level", self.level,
                "a name for the band. An unnamed band cannot appear in a member's label, and "
                "a member whose label does not name its operands is not checkable")
        array = np.asarray(self.values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise InvalidParameterError(
                "values", self.level,
                "a finite series. A missing frame is a fact about the record and has to be "
                "handled where it arises, not averaged away inside a correlation")

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Record:
    """Several bands observed over the same frames - one partition of one study."""

    name: str
    series: Tuple[ScaleSeries, ...]
    cadence_seconds: float = 1.0
    frames: Tuple[int, int] = (0, 0)
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if len(self.series) < 2:
            raise InvalidParameterError(
                "series", len(self.series),
                "at least two bands. A precedence family is built from ordered pairs of "
                "distinct bands, and one band yields no pair at all - the sweep would then "
                "report nothing because there was nothing to test")
        lengths = {len(s) for s in self.series}
        if len(lengths) != 1:
            raise InvalidParameterError(
                "series", sorted(lengths),
                "bands observed over the same frames. Series of different lengths cannot be "
                "lagged against each other without choosing an alignment, and the alignment "
                "is the thing under test")
        length = lengths.pop()
        if length < MIN_FRAMES:
            raise InvalidParameterError(
                "series length", length,
                "at least %d frames. Below that the effective sample size of an "
                "autocorrelated series is comparable to the number of lags being searched, "
                "and the sweep is fitting the record rather than measuring it" % MIN_FRAMES)
        names = [s.level for s in self.series]
        if len(set(names)) != len(names):
            raise InvalidParameterError(
                "levels", names,
                "distinct band names. Two bands under one name are one axis value, and the "
                "family would price two hypotheses as one")
        if self.frames == (0, 0):
            object.__setattr__(self, "frames", (0, int(length)))
        start, stop = int(self.frames[0]), int(self.frames[1])
        object.__setattr__(self, "frames", (start, stop))
        if stop - start != length:
            raise InvalidParameterError(
                "frames", (start, stop),
                "a half-open frame range spanning the %d frames this record holds. A "
                "partition whose lineage and content disagree is two partitions, and the "
                "seal would bind whichever the reader believed" % length)
        if not math.isfinite(self.cadence_seconds) or self.cadence_seconds <= 0:
            raise InvalidParameterError("cadence_seconds", self.cadence_seconds,
                                        "a positive frame spacing")

    @property
    def levels(self) -> Tuple[str, ...]:
        return tuple(s.level for s in self.series)

    @property
    def length(self) -> int:
        return len(self.series[0])

    def array(self, level: str) -> np.ndarray:
        for s in self.series:
            if s.level == level:
                return s.array()
        raise InvalidParameterError(
            "level", level, "one of the bands this record holds: %s" % (list(self.levels),))

    def identity(self, name: Optional[str] = None) -> PartitionIdentity:
        """The partition this record is, described without reading a single value."""
        return PartitionIdentity(
            name=str(name or self.name), n_times=self.length, n_channels=len(self.series),
            channel_labels=self.levels, frames=self.frames, provenance=self.provenance)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "levels": list(self.levels), "n_frames": self.length,
                "frames": list(self.frames), "cadence_seconds": float(self.cadence_seconds),
                "provenance": dict(self.provenance)}


# ----------------------------------------------------------------------------- the clock


def autocorrelation_time(values: Sequence[float]) -> float:
    """The AR(1)-equivalent memory of a series, in frames.

    Read off the same lag-1 autocorrelation the effective sample size uses, so the memory that
    sets the embargo and the memory that discounts the sample are the same number and cannot
    drift apart into two different stories about the same series.
    """
    v = np.asarray(list(values), dtype=np.float64)
    v = v - v.mean()
    denom = float(np.sum(v * v))
    if denom <= 0:
        return 1.0
    r1 = float(np.sum(v[1:] * v[:-1]) / denom)
    r1 = max(min(r1, 0.999), 0.0)
    return float((1.0 + r1) / (1.0 - r1))


def record_memory(record: Record) -> int:
    """The slowest band's memory, in whole frames - what an embargo has to clear."""
    return int(math.ceil(max(autocorrelation_time(s.values) for s in record.series)))


# --------------------------------------------------------------------------- the family


def admissible_lags(record: Record, *, max_lag: int,
                    minimum_lag: int = MIN_ADMISSIBLE_LAG,
                    min_effective_pairs: float = MIN_EFFECTIVE_PAIRS) -> Tuple[int, ...]:
    """Which lags this record may be read for precedence at (rule R21), before any sweep.

    Two refusals, both before the family is priced. Lag zero is not a lead. And a lag long
    enough to leave fewer than `min_effective_pairs` effective pairs is refused outright
    rather than tested at a power the declaration does not mention, because the family
    presents its members as equivalent tests and a member with a third of the data is not.
    """
    if minimum_lag < MIN_ADMISSIBLE_LAG:
        raise LagInadmissibleError(
            minimum_lag,
            "Lag %d is not a lead: the two bands are read at the same instant, and a "
            "simultaneous correlation is a statement about shared modulation rather than "
            "about precedence" % minimum_lag)
    if max_lag < minimum_lag:
        raise InvalidParameterError(
            "max_lag", max_lag,
            "a maximum lag at least the minimum of %d. A lag axis with no values empties the "
            "family" % minimum_lag)
    kept: List[int] = []
    for lag in range(int(minimum_lag), int(max_lag) + 1):
        pairs = record.length - lag
        if pairs < 4:
            break
        worst = min(effective_sample_size(record.array(level)[: pairs])
                    for level in record.levels)
        if worst < min_effective_pairs:
            break
        kept.append(lag)
    if not kept:
        raise LagInadmissibleError(
            max_lag,
            "No lag between %d and %d leaves %g effective pairs on a record of %d frames, so "
            "there is no admissible member to search at all"
            % (minimum_lag, max_lag, min_effective_pairs, record.length))
    return tuple(kept)


def ordered_pairs_of(levels: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """Every ordered pair of distinct bands, in TG3.1's `ordered_pairs` enumeration order.

    Ordered, because *A leads B* and *B leads A* are computed from the same two series and
    are two hypotheses; a search that asks only one of them was told the direction.
    """
    return tuple((a, b) for a in levels for b in levels if str(a) != str(b))


def pair_label(pair: Tuple[str, str]) -> str:
    return "%s>%s" % (pair[0], pair[1])


@dataclass(frozen=True)
class BasisCoupling:
    """What the analysis basis relates to what, measured where nothing else can.

    The control is a record built by the same pipeline with **no temporal structure at all**:
    every frame independent, so a correlation between two of its bands is the decomposition
    talking about itself. A redundant transform does not hand back independent bands, it
    hands back a smear, and two levels that are largely the same band produce a beautiful
    lagged relationship that is a fact about the wavelet and not about the world.

    The limit is not chosen, it is read off the same control: a basis has no dynamics, so
    whatever correlation it produces *across* frames is its own noise floor, and two bands are
    coupled by the basis when it relates them within a frame more strongly than it relates
    anything across one. Measured on this tree's own control that floor is 0.06 to 0.18 over
    seeds, against simultaneous couplings of 0.99 between the two finest levels, 0.78 to 0.84
    between the two coarsest, and 0.49 to 0.53 between the finest and the second coarsest -
    so of the twelve ordered pairs four survive, and the pass is left asking about band pairs
    the transform can actually tell apart.
    """

    simultaneous: Mapping[str, float]
    floor: float
    control: str

    def couples(self, a: str, b: str) -> bool:
        return float(self.simultaneous.get(pair_label((a, b)), 0.0)) > float(self.floor)

    def describe(self) -> Dict[str, Any]:
        return {"control": self.control, "floor": float(self.floor),
                "simultaneous": {k: float(v) for k, v in sorted(self.simultaneous.items())},
                "coupled_pairs": sorted(k for k, v in self.simultaneous.items()
                                        if float(v) > float(self.floor)),
                "claim_boundary": ("A property of the decomposition, measured on a record "
                                   "with no temporal structure. It says which bands this "
                                   "basis cannot separate; it says nothing about any record "
                                   "under test, and no data from one was read to compute it")}


#: How much memory a control record may have before it stops being a control. One frame is
#: the memoryless case exactly; the allowance is for estimator noise on a finite record.
MAX_CONTROL_MEMORY = 1.5


def measure_basis_coupling(control: Record, *, lags: Sequence[int]) -> BasisCoupling:
    """Read the basis's own couplings off a control record, before any study is declared.

    The control is checked to *be* one first. A record with dynamics of its own relates its
    bands across frames as well as within them, and the across-frame floor read off it would
    be the record's memory rather than the basis's noise - which would raise the floor and
    quietly stop excluding the pairs this exists to exclude.
    """
    lag_values = tuple(int(lag) for lag in lags)
    for series in control.series:
        memory = autocorrelation_time(series.values)
        if memory > MAX_CONTROL_MEMORY:
            raise ControlNotStaticError(series.level, memory)
    if not lag_values:
        raise InvalidParameterError(
            "lags", [], "the lags the study will search, so the basis's across-frame floor "
                        "is measured over the same range the study will ask about")
    simultaneous: Dict[str, float] = {}
    floor = 0.0
    for a, b in ordered_pairs_of(control.levels):
        x, y = control.array(a), control.array(b)
        if np.std(x) == 0 or np.std(y) == 0:
            simultaneous[pair_label((a, b))] = 0.0
            continue
        simultaneous[pair_label((a, b))] = abs(float(np.corrcoef(x, y)[0, 1]))
        floor = max(floor, max(precedence_strength(x, y, lag) for lag in lag_values))
    return BasisCoupling(simultaneous=simultaneous, floor=float(floor), control=control.name)


def admissible_pairs(record: Record, *,
                     coupling: Optional[BasisCoupling] = None) -> Tuple[Tuple[str, str], ...]:
    """The band pairs this study may ask about: every ordered pair the basis can separate.

    Narrowing a family is normally the thing R18 refuses, and this narrowing is admissible
    for one reason: it is derived from a control record rather than from the record under
    test, so it is fixed before the study's data is read and it cannot have been chosen to
    suit an answer. A pair the transform couples is not a hypothesis this instrument can
    test, and pricing it as though it were spends correction power on a question about the
    wavelet.
    """
    pairs = ordered_pairs_of(record.levels)
    if coupling is not None:
        pairs = tuple(p for p in pairs if not coupling.couples(*p))
    if not pairs:
        raise InvalidParameterError(
            "pairs", [],
            "at least one band pair this basis can separate. Every ordered pair of these %d "
            "bands is coupled by the decomposition itself, so there is no relationship "
            "between them that this instrument could distinguish from its own smear - which "
            "is a fact about the transform, and the answer to it is a different transform "
            "rather than a lower threshold" % len(record.levels))
    return pairs


def precedence_search_specification(record: Record, *, lags: Sequence[int], n_surrogates: int,
                                    pairs: Optional[Sequence[Tuple[str, str]]] = None,
                                    alpha: float = 0.05,
                                    correction: str = "benjamini_yekutieli",
                                    study_id: str = "") -> SearchSpecification:
    """The declared family: every admissible band pair, at every admissible lag.

    Two terms multiplied - the pairs, and the lags, because neither was known. The pairs are
    a declared axis rather than TG3.1's `ordered_pairs` combinator applied to the bands,
    because the basis excludes some of them and an exclusion cannot be expressed as an axis
    of bands; with nothing excluded the two price the family identically, and
    `test_precedence.py` checks that they do rather than leaving it as a claim.
    """
    lag_values = tuple(int(lag) for lag in lags)
    pair_values = tuple(pairs) if pairs is not None else ordered_pairs_of(record.levels)
    if not pair_values:
        raise InvalidParameterError(
            "pairs", [], "at least one ordered band pair. A family with no pair axis is a "
                         "study with nothing to test")
    for a, b in pair_values:
        if str(a) == str(b):
            raise InvalidParameterError(
                "pairs", (a, b),
                "pairs of distinct bands. A band lagged against itself is its own "
                "autocorrelation, which every red series has and which is not a "
                "relationship between scales")
        for name in (a, b):
            if name not in record.levels:
                raise InvalidParameterError(
                    "pairs", name, "bands this record holds: %s" % (list(record.levels),))
    if not lag_values:
        raise InvalidParameterError(
            "lags", [], "at least one admissible lag. A family with no lag axis is not a "
                        "narrowing, it is a study with nothing to test")
    if len(set(lag_values)) != len(lag_values):
        raise InvalidParameterError("lags", list(lag_values),
                                    "distinct lags. A lag declared twice is a member counted "
                                    "twice, and the correction unit is then wrong")
    for lag in lag_values:
        if lag < MIN_ADMISSIBLE_LAG:
            raise LagInadmissibleError(lag, "Lag %d is not a lead" % lag)
    return SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("band_pair",
                                                 tuple(pair_label(p) for p in pair_values)),)),
               SearchTerm("product", (SearchAxis("lag", lag_values),))),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}@{1}", study_id=study_id,
        notes={"stage": "generate", "record": record.name,
               "cadence_seconds": float(record.cadence_seconds),
               "reading": ("each member asks whether the first band at t precedes the second "
                           "at t + lag. The pair is ordered because the reverse reading is a "
                           "different hypothesis, and it is a member too")})


# ------------------------------------------------------------------------- the statistic


def lagged_pair(driver: np.ndarray, driven: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """The overlapping frames: the driver up to `-lag`, the driven from `lag` on."""
    if lag < 1:
        raise LagInadmissibleError(lag, "Lag %d is not a lead" % lag)
    if lag >= driver.size:
        raise LagInadmissibleError(
            lag, "Lag %d leaves no overlapping frames on a series of %d"
                 % (lag, driver.size))
    return driver[: driver.size - lag], driven[lag:]


def lagged_correlation(driver: np.ndarray, driven: np.ndarray, lag: int) -> float:
    """Pearson correlation of the driver at `t` with the driven at `t + lag`."""
    a, b = lagged_pair(driver, driven, lag)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def precedence_strength(driver: np.ndarray, driven: np.ndarray, lag: int) -> float:
    """The statistic: the *magnitude* of the lagged correlation.

    Magnitude, because the sign of a correlation is not the direction of a precedence. A band
    whose activity suppresses another's later activity leads it exactly as much as one that
    excites it, and a signed statistic would report the first as evidence against itself. The
    cost is paid honestly: the surrogate null is over the same magnitude, so a two-sided
    question is priced as one.
    """
    return abs(lagged_correlation(driver, driven, lag))


@dataclass(frozen=True)
class PrecedenceCandidate:
    """One member of the family, measured on one partition. Not a finding."""

    driver: str
    driven: str
    lag: int
    correlation: float
    n_pairs: int
    n_effective: float
    p_naive: float
    p_effective: float

    @property
    def label(self) -> str:
        return "%s@%d" % (pair_label((self.driver, self.driven)), self.lag)

    @property
    def strength(self) -> float:
        return abs(self.correlation)

    @property
    def reverse_label(self) -> str:
        """The same two bands read the other way at the same lag - a different member."""
        return "%s@%d" % (pair_label((self.driven, self.driver)), self.lag)

    def describe(self) -> Dict[str, Any]:
        return {"label": self.label, "driver": self.driver, "driven": self.driven,
                "lag": int(self.lag), "correlation": float(self.correlation),
                "n_pairs": int(self.n_pairs), "n_effective": float(self.n_effective),
                "p_naive": float(self.p_naive), "p_effective": float(self.p_effective),
                "claim_boundary": ("A member of a declared family measured on one partition. "
                                   "Neither p-value here is corrected for the family, and "
                                   "the family is priced at more surrogates than any "
                                   "ensemble in this study")}


@dataclass(frozen=True)
class SweepResult:
    """Every declared member, measured. The generate stage's whole output."""

    record: str
    specification: SearchSpecification
    candidates: Tuple[PrecedenceCandidate, ...]
    lags: Tuple[int, ...]
    pairs: Tuple[Tuple[str, str], ...]
    affordable_here: bool

    @property
    def n_examined(self) -> int:
        return len(self.candidates)

    @property
    def ranked(self) -> Tuple[PrecedenceCandidate, ...]:
        """Strongest first, ties broken by the shorter lag and then by label.

        The shorter lag deliberately: two lags of one relationship are the same relationship,
        and when they are indistinguishable the one that claims less - a lead of two frames
        rather than of five - is the one to carry forward.
        """
        return tuple(sorted(self.candidates,
                            key=lambda c: (-c.strength, c.lag, c.label)))

    @property
    def best(self) -> PrecedenceCandidate:
        return self.ranked[0]

    def member(self, label: str) -> PrecedenceCandidate:
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate
        raise InvalidParameterError(
            "label", label, "a member of this sweep's declared family")

    def describe(self) -> Dict[str, Any]:
        return {
            "record": self.record,
            "n_examined": self.n_examined,
            "family_size": self.specification.family_size,
            "lags": list(self.lags),
            "band_pairs": [pair_label(p) for p in self.pairs],
            "specification_sha256": self.specification.fingerprint(),
            "affordable_in_one_stage": bool(self.affordable_here),
            "best": self.best.describe(),
            "claim_boundary": ("%d members measured on the training partition. The strongest "
                               "is the strongest of %d, which is a different quantity from a "
                               "strong one, and nothing here has been corrected"
                               % (self.n_examined, self.specification.family_size)),
        }


def sweep(record: Record, *, lags: Sequence[int], n_surrogates: int,
          pairs: Optional[Sequence[Tuple[str, str]]] = None, alpha: float = 0.05,
          correction: str = "benjamini_yekutieli", study_id: str = "") -> SweepResult:
    """Measure every declared member, and check that they are the declared members.

    The family and the pass are compared label for label, in order, rather than by count: two
    different searches of the same size would agree on a count and disagree on what was
    tested, and it is what was tested that the correction unit describes.
    """
    specification = precedence_search_specification(
        record, lags=lags, n_surrogates=n_surrogates, pairs=pairs, alpha=alpha,
        correction=correction, study_id=study_id)
    lag_values = tuple(int(lag) for lag in lags)
    pair_values = tuple(pairs) if pairs is not None else ordered_pairs_of(record.levels)
    candidates: List[PrecedenceCandidate] = []
    for driver, driven in pair_values:
        a_full, b_full = record.array(driver), record.array(driven)
        for lag in lag_values:
            a, b = lagged_pair(a_full, b_full, lag)
            measured = correlation_test(a, b)
            r = float(measured["r"]) if math.isfinite(measured["r"]) else 0.0
            candidates.append(PrecedenceCandidate(
                driver=driver, driven=driven, lag=int(lag), correlation=r,
                n_pairs=int(a.size), n_effective=float(measured.get("n_effective", a.size)),
                p_naive=float(measured.get("p_value_naive", 1.0)),
                p_effective=float(measured.get("p_value", 1.0))))
    emitted = tuple(c.label for c in candidates)
    declared = specification.labels()
    if emitted != declared:
        mismatch = [(d, e) for d, e in zip(declared, emitted) if d != e]
        raise InvalidParameterError(
            "sweep", mismatch[:4],
            "a pass over exactly the declared family, in the declared order. %d of %d "
            "members differ between what was priced and what was measured, so the correction "
            "unit describes a search that did not happen"
            % (len(mismatch) or abs(len(declared) - len(emitted)), len(declared)))
    ceiling = affordable_member_count(n_surrogates, alpha=alpha, correction=correction)
    return SweepResult(record=record.name, specification=specification,
                       candidates=tuple(candidates), lags=lag_values,
                       pairs=pair_values,
                       affordable_here=specification.family_size <= ceiling)


# ------------------------------------------------------------------------- the surrogate


def circular_shift(values: np.ndarray, *, rng: np.random.Generator,
                   minimum_shift: int) -> np.ndarray:
    """The same series, started somewhere else.

    Every value, the whole autocorrelation function and the marginal distribution survive
    exactly; only the alignment with the other band is destroyed, and the alignment is the
    hypothesis. The shift is drawn clear of the series' own memory in both directions,
    because a shift of one frame on a series with a memory of six is still very nearly the
    original series and a null built from it contains the alternative.
    """
    n = int(values.size)
    low = int(max(1, minimum_shift))
    if n - low <= low:
        raise ShiftInfeasibleError(n, low)
    offset = int(rng.integers(low, n - low + 1))
    return np.roll(values, offset)


def permuted_series(values: np.ndarray, *, rng: np.random.Generator,
                    minimum_shift: int = 0) -> np.ndarray:
    """The wrong surrogate, kept public so its wrongness can be measured.

    An i.i.d. shuffle preserves the marginal distribution and destroys the autocorrelation
    along with the alignment. Two red series are far more correlated at a random alignment
    than two white ones, so a null built by shuffling sits much closer to zero than the truth
    and every p-value drawn from it is too small. It is not registered anywhere: unlike
    `circular_shift` it cannot be reached by name, and the only thing it is for is the
    comparison in `test_precedence.py`.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    rng.shuffle(out)
    return out


SURROGATES = {DEFAULT_SURROGATE: circular_shift, PERMUTED: permuted_series}


@dataclass(frozen=True)
class PrecedenceEvidence:
    """What a surrogate ensemble says about one member, on one partition."""

    label: str
    driver: str
    driven: str
    lag: int
    strength: float
    p_value: float
    n_surrogates: int
    null_mean: float
    surrogate: str
    selected_over: Tuple[int, ...] = ()

    @property
    def power_floor(self) -> float:
        """The smallest p-value this ensemble could ever have produced."""
        return 1.0 / (1.0 + max(0, int(self.n_surrogates)))

    def vacuous(self, alpha: float) -> bool:
        """True when the test could not have rejected at `alpha` however strong the data.

        TG2.4's rule, applied here: an ensemble whose floor sits above the corrected alpha is
        an instrument that cannot register the thing it was pointed at, and its silence is a
        fact about the ensemble rather than about the record.
        """
        return self.power_floor > float(alpha)

    def describe(self) -> Dict[str, Any]:
        return {"label": self.label, "driver": self.driver, "driven": self.driven,
                "lag": int(self.lag), "strength": float(self.strength),
                "p_value": float(self.p_value), "n_surrogates": int(self.n_surrogates),
                "null_mean": float(self.null_mean), "surrogate": self.surrogate,
                "selected_over": list(self.selected_over),
                "power_floor": self.power_floor}


def precedence_p_value(record: Record, *, driver: str, driven: str, lag: int,
                       n_surrogates: int, seed: int,
                       surrogate: str = DEFAULT_SURROGATE,
                       minimum_shift: Optional[int] = None,
                       selected_over: Optional[Sequence[int]] = None) -> PrecedenceEvidence:
    """Test one member against an ensemble that keeps everything except the alignment.

    `selected_over` is what makes a *chosen* lag testable. If the lag being reported was the
    best of a sweep over several, the statistic being tested is the maximum of that sweep and
    the null has to be the distribution of that maximum; passing the lag set computes it.
    A lag frozen on the training partition was not chosen on this one, so the confirmation
    leaves `selected_over` empty and tests a single statistic - which is the affordable case,
    and is affordable *because* of the freeze.
    """
    if surrogate not in SURROGATES:
        raise InvalidParameterError("surrogate", surrogate,
                                    "one of %s" % sorted(SURROGATES))
    a = record.array(driver)
    b = record.array(driven)
    shift = int(minimum_shift if minimum_shift is not None else record_memory(record))
    lags = tuple(int(x) for x in (selected_over if selected_over else (lag,)))
    if lag not in lags:
        raise InvalidParameterError(
            "selected_over", list(lags),
            "a lag set containing the lag being reported. A statistic tested against the "
            "null of a maximum it was not part of is being priced for a search it did not "
            "come from")

    def statistic(driver_values: np.ndarray) -> float:
        return max(precedence_strength(driver_values, b, k) for k in lags)

    observed = statistic(a)
    rng = np.random.default_rng(seed)
    draw = SURROGATES[surrogate]
    null = [statistic(draw(a, rng=rng, minimum_shift=shift)) for _ in range(int(n_surrogates))]
    result = surrogate_p_value(observed, null, alternative="greater")
    return PrecedenceEvidence(
        label="%s@%d" % (pair_label((driver, driven)), lag),
        driver=driver, driven=driven, lag=int(lag),
        strength=float(observed), p_value=float(result["p_value"]),
        n_surrogates=int(n_surrogates), null_mean=float(result["null_mean"]),
        surrogate=surrogate,
        selected_over=lags if selected_over else ())


# -------------------------------------------------------------------------- the partition


def split_with_embargo(record: Record, *, fraction: float = 0.5,
                       embargo: Optional[int] = None) -> Tuple[Record, Record]:
    """Cut a record into a training and a held-out partition, with a gap between them.

    The gap is between them because two adjacent halves of an autocorrelated record are not
    quite two partitions: the first frames of the second half are the immediate future of the
    last frames of the first. Whether that matters was measured rather than assumed, and on
    records whose bands are independent but slow it does not - the held-out surrogate null
    already carries the held-out record's memory. The default embargo is the record's own
    memory anyway, and both the embargo used and the one recommended travel in the held-out
    partition's provenance, so that `freeze_precedence` can refuse a seal cut closer than the
    study declared.
    """
    if not 0.1 <= fraction <= 0.9:
        raise InvalidParameterError(
            "fraction", fraction,
            "a split somewhere between a tenth and nine tenths. A partition holding almost "
            "nothing cannot confirm anything, and calling it held out does not change that")
    recommended = record_memory(record)
    gap = int(recommended if embargo is None else embargo)
    if gap < 0:
        raise InvalidParameterError("embargo", gap, "a non-negative number of frames to drop")
    cut = int(round(record.length * fraction))
    start = cut + gap
    if cut < MIN_FRAMES or record.length - start < MIN_FRAMES:
        raise InvalidParameterError(
            "fraction", fraction,
            "a split leaving at least %d frames on each side after a %d-frame embargo. A "
            "record of %d frames cannot carry both" % (MIN_FRAMES, gap, record.length))
    offset = record.frames[0]

    def cut_out(lo: int, hi: int, name: str, split: str) -> Record:
        provenance = dict(record.provenance)
        provenance.update({
            "split": split, "parent": record.name,
            "split_frames": [offset + lo, offset + hi],
            "embargo_frames": gap, "recommended_embargo_frames": recommended,
        })
        return Record(name=name,
                      series=tuple(ScaleSeries(s.level, s.values[lo:hi]) for s in record.series),
                      cadence_seconds=record.cadence_seconds,
                      frames=(offset + lo, offset + hi), provenance=provenance)

    return (cut_out(0, cut, "%s.train" % record.name, "train"),
            cut_out(start, record.length, "%s.held_out" % record.name, "held_out"))


# -------------------------------------------------------------- generate, freeze, confirm


def affordable_member_count(n_surrogates: int, alpha: float = 0.05,
                            correction: str = "benjamini_yekutieli") -> int:
    """How many members this ensemble can still reject one of, after correction (TG3.1)."""
    return int(max_affordable_family(n_surrogates, alpha=alpha, method=correction))


def deduplicate_candidates(
        candidates: Sequence[PrecedenceCandidate]) -> Tuple[PrecedenceCandidate, ...]:
    """One member per ordered band pair - the best lag it was seen at.

    A relationship with a memory of several frames appears in the ranking at the true lag and
    at its neighbours, because the driver at `k - 1` is very nearly the driver at `k`. Those
    are one hypothesis wearing several labels, and freezing all of them is a correction unit
    of several paid for one test. Greedy over the ranking, so the lag kept is the strongest
    on train - which is not thereby a claim that it is the true lag: it is the lag that will
    be frozen, and the confirmation tests that lag and no other.
    """
    kept: List[PrecedenceCandidate] = []
    seen = set()
    for candidate in candidates:
        pair = (candidate.driver, candidate.driven)
        if pair in seen:
            continue
        seen.add(pair)
        kept.append(candidate)
    return tuple(kept)


def indistinguishable_from_best(result: SweepResult) -> Tuple[PrecedenceCandidate, ...]:
    """The candidates whose training strength is within one standard error of the best.

    A correlation estimated from `n_eff` effective pairs has a standard error of about
    `1 / sqrt(n_eff - 3)` on Fisher's z scale, and two members closer together than that were
    not distinguished by the data. Taking only the best would be pretending to a precision
    the record does not have; taking everything down to the ceiling spends correction power
    on members nobody proposed, which TG3.5 measured the cost of. This is the middle: the
    members the training partition could not tell apart from the winner.
    """
    ranked = deduplicate_candidates(result.ranked)
    best = ranked[0]
    error = _fisher_error(best.n_effective)
    top = _fisher_z(best.strength)
    return tuple(c for c in ranked if top - _fisher_z(c.strength) <= error)


def _fisher_z(r: float) -> float:
    return float(np.arctanh(min(abs(r), 0.999999)))


def _fisher_error(n_effective: float) -> float:
    return float(1.0 / math.sqrt(max(4.0, n_effective) - 3.0))


def choose_candidates(result: SweepResult,
                      n_candidates: Optional[int] = None) -> Tuple[PrecedenceCandidate, ...]:
    """What the generate stage carries forward: distinct pairs, undistinguished from the best.

    Selecting on training strength is selection, and it is the selection the generate stage
    exists to perform: it happens before the seal, on the partition that was swept, and
    nothing it produces is a claim.
    """
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
            "alpha %g. Freezing more is freezing a family the confirmation cannot pay for"
            % (ceiling, result.specification.n_surrogates,
               result.specification.correction, result.specification.alpha))
    pool = (indistinguishable_from_best(result) if n_candidates is None
            else deduplicate_candidates(result.ranked))
    return pool[:min(wanted, len(pool))]


def report_precedence_generation(result: SweepResult, *, train: PartitionIdentity,
                                 n_candidates: Optional[int] = None) -> Dict[str, Any]:
    """The generation receipt, through TG3.2's own checker."""
    chosen = choose_candidates(result, n_candidates=n_candidates)
    report = report_generation(
        result.specification, train=train, candidates=[c.label for c in chosen],
        p_values={c.label: c.p_effective for c in chosen})
    report["sweep"] = result.describe()
    return report


def confirmatory_specification(chosen: Sequence[PrecedenceCandidate], *, n_surrogates: int,
                               alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli",
                               study_id: str = "") -> SearchSpecification:
    """The frozen family: these band pairs, at these lags, and nothing else.

    Each member keeps the label it had in the generate family, which is what makes the
    confirmatory family demonstrably a subset of the generated one - and what makes a
    silently changed lag visible, because a lag is part of the name.
    """
    labels = [c.label for c in chosen]
    if not labels:
        raise InvalidParameterError(
            "chosen", labels, "at least one member to freeze")
    if len(set(labels)) != len(labels):
        raise InvalidParameterError(
            "chosen", labels,
            "distinct members. One relationship frozen twice is a correction unit inflated "
            "by a duplicate")
    return SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("relationship", tuple(labels)),)),),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}", study_id=study_id,
        notes={"stage": "confirm",
               "reading": ("each member is one band pair at one lag, frozen before the "
                           "held-out partition was opened and tested there once")})


def freeze_precedence(result: SweepResult, *, held_out: PartitionIdentity, sealed_at: str,
                      n_surrogates: int,
                      chosen: Optional[Sequence[PrecedenceCandidate]] = None,
                      n_candidates: Optional[int] = None,
                      ledger: Optional[HeldOutLedger] = None,
                      study_id: str = "") -> Tuple[Seal, Tuple[PrecedenceCandidate, ...]]:
    """Freeze the confirmatory family, or refuse to.

    One refusal beyond TG3.2's own: a held-out partition whose provenance records an embargo
    shorter than the record's memory is not held out. That check lives here rather than in
    the splitter because this is where the claim is made - a study may cut a record any way
    it likes as long as it does not then seal a claim against the offcut.
    """
    embargo = held_out.provenance.get("embargo_frames")
    recommended = held_out.provenance.get("recommended_embargo_frames")
    if embargo is not None and recommended is not None and int(embargo) < int(recommended):
        raise EmbargoTooShortError(int(embargo), int(recommended), held_out.name)
    picked = tuple(chosen) if chosen is not None else choose_candidates(
        result, n_candidates=n_candidates)
    confirm = confirmatory_specification(
        picked, n_surrogates=n_surrogates, alpha=result.specification.alpha,
        correction=result.specification.correction, study_id=study_id)
    seal = freeze_confirmatory_family(
        result.specification, confirm, held_out=held_out, sealed_at=sealed_at,
        study_id=study_id, ledger=ledger)
    order = {label: i for i, label in enumerate(seal.confirm_labels)}
    return seal, tuple(sorted(picked, key=lambda c: order[c.label]))


def confirm_precedence(seal: Seal, chosen: Sequence[PrecedenceCandidate], *,
                       record: Record, held_out: PartitionIdentity, ledger: HeldOutLedger,
                       opened_at: str, seed: int, surrogate: str = DEFAULT_SURROGATE,
                       n_surrogates: Optional[int] = None) -> Dict[str, Any]:
    """Test the frozen relationships on the held-out record, once, at the frozen size.

    The lag is read from the frozen label rather than re-chosen here, which is the whole
    difference between this stage and the sweep: no maximum is taken on the held-out
    partition, so the null is the null of one statistic and the ensemble can afford it.
    """
    frozen = list(seal.confirm_labels)
    supplied = [c.label for c in chosen]
    if supplied != frozen:
        raise InvalidParameterError(
            "chosen", supplied[:8],
            "the members the seal froze, in the order it froze them: %s. A relationship "
            "tested under a frozen label at a different lag is a redefinition, and the lag "
            "is part of the name it was sealed under" % frozen[:8])
    ensemble = int(n_surrogates if n_surrogates is not None
                   else seal.confirm.get("n_surrogates", 0))
    evidence: List[PrecedenceEvidence] = []
    for offset, candidate in enumerate(chosen):
        measured = precedence_p_value(
            record, driver=candidate.driver, driven=candidate.driven, lag=candidate.lag,
            n_surrogates=ensemble, seed=seed + offset, surrogate=surrogate)
        evidence.append(dc_replace(measured, label=candidate.label))
    receipt = confirm_on_held_out(
        seal, p_values={e.label: e.p_value for e in evidence}, held_out=held_out,
        ledger=ledger, opened_at=opened_at)
    alpha = float(seal.confirm.get("alpha", 0.05))
    receipt["relationships"] = [e.describe() for e in evidence]
    receipt["strengths"] = [e.strength for e in evidence]
    receipt["surrogate"] = surrogate
    receipt["minimum_shift"] = record_memory(record)
    receipt["n_frames"] = record.length
    receipt["vacuous"] = [e.label for e in evidence if e.vacuous(alpha)]
    return receipt


__all__ = [
    "MIN_FRAMES", "MIN_EFFECTIVE_PAIRS", "MIN_ADMISSIBLE_LAG",
    "DEFAULT_SURROGATE", "PERMUTED", "SURROGATES",
    "LagInadmissibleError", "ShiftInfeasibleError", "EmbargoTooShortError",
    "ControlNotStaticError", "MAX_CONTROL_MEMORY",
    "ScaleSeries", "Record", "PrecedenceCandidate", "SweepResult", "PrecedenceEvidence",
    "autocorrelation_time", "record_memory",
    "admissible_lags", "ordered_pairs_of", "pair_label", "BasisCoupling",
    "measure_basis_coupling", "admissible_pairs", "precedence_search_specification",
    "lagged_pair", "lagged_correlation", "precedence_strength", "sweep",
    "circular_shift", "permuted_series", "precedence_p_value",
    "split_with_embargo",
    "affordable_member_count", "deduplicate_candidates", "indistinguishable_from_best",
    "choose_candidates", "report_precedence_generation", "confirmatory_specification",
    "freeze_precedence", "confirm_precedence",
]
