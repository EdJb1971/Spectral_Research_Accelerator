"""The scale/shape statistic, and the invariance it may claim only once measured (TG17.11).

**Why this module exists.** `experiment_qualification.py`'s `scale_shape_calibration` gate reads
`NOT_IMPLEMENTED` rather than `NOT_RUN`, and the distinction is exact: G17's other declared mode
has a method that has merely not been executed, while this one had no scientific statistic at all.
`structural_alignment.py` enumerates which normalized coordinates two records share, which is a
lookup and not a test; `structural_nulls.py` supplies the surrogate. What was missing is the
number in between.

**What the mode asks, and what that forces the statistic to be.** Scale/shape mode asks whether a
frozen shape recurs across *native* scales - whether the profile a light curve traces over 1.8
hours is the profile a float traces over 46 days. There is no shared clock, so the comparison
cannot be made in seconds, and the statistic must be invariant to each record's native duration.
That invariance is the mode's entire content, which is why it is measured here rather than
asserted: `measure_scale_invariance` re-presents one record across a range of native durations
and reports the largest relative movement the statistic shows. It varies nothing else, because a
sweep that changed the sampling at the same time would report one number for two effects and make
the invariance claim unfalsifiable; row density is measured separately by
`measure_cadence_dependence`, where movement is expected rather than a fault.

**Dividing by a declared quantity is not the same as dividing by a modelled one, and the
difference is the whole argument.** `invariance.py` records why `scale_normalised` is kept as a
function and left out of `MATCHERS`: its denominator is the extractor's *estimated* spatial scale,
which drifts from about +3.6% to -2.6% across a sixfold range, and the whole of that drift lands
in the quotient. The denominator here is different in kind. A record's native duration is
`StructuralScale.native_value`, a quantity the adapter *declares* and the manifest carries, not
one this module estimates from the data it is about to test. Nothing is inferred and so nothing
drifts. That argument is still not accepted on its own: `measure_normalisation_sensitivity`
reports how far the statistic moves when the declared duration is wrong by a stated percentage,
so the cost of a mis-declaration is a number in the record rather than an assumption.

**Comparison is over shared phase, not over a grid and not over rows.** Resampling both records
onto a common phase grid would manufacture the correspondence being tested, exactly as binning
onto a common calendar grid would manufacture simultaneity - the reason `family_calibration.py`
weights by shared support *duration* rather than aligning row `i` to row `i`. The same treatment
carries over with phase in place of seconds: intersect the two records' declared supports in
phase, weight each contribution by the phase actually shared, and a record rewritten at ten times
the row density over the same span produces the same number. `measure_scale_invariance` is what
demonstrates that rather than what claims it.

Nothing in this module is evidence about the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.structural_trajectory import StructuralTrajectory


CHANNEL = "standardized_level"

#: The declared native duration must arrive in seconds. A unit this module does not know is
#: refused rather than converted by guess: a wrong factor here rescales the phase axis silently
#: and would present as a shape that does not recur.
SUPPORTED_NATIVE_UNITS = ("seconds",)


class ShapeComparisonRefusal(InvalidParameterError):
    """A scale/shape comparison this module will not compute, named with what it lacked."""


@dataclass(frozen=True)
class ShapeRecord:
    """One record's canonical values on its own declared native scale.

    Holds only what the comparison is entitled to use: the declared supports, the values, the
    validity, and the *declared* native duration the phase axis is built from. It deliberately
    does not carry the trajectory, so no part of the statistic can reach past the declaration
    into the record's clock.
    """

    label: str
    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    native_seconds: float
    native_units: str = "seconds"
    origin_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("starts", "ends", "values"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        object.__setattr__(self, "valid", np.asarray(self.valid, dtype=bool))
        if self.native_units not in SUPPORTED_NATIVE_UNITS:
            raise ShapeComparisonRefusal(
                "native_units", self.native_units,
                "a declared native duration in %s. Converting an unrecognised unit by guess "
                "would rescale the phase axis silently" % " or ".join(SUPPORTED_NATIVE_UNITS))
        if not np.isfinite(self.native_seconds) or self.native_seconds <= 0.0:
            raise ShapeComparisonRefusal(
                "native_seconds", self.native_seconds,
                "a positive declared native duration to build a phase axis from")
        if not (len(self.starts) == len(self.ends) == len(self.values) == len(self.valid)):
            raise ShapeComparisonRefusal(
                "record", self.label, "supports, values and validity of equal length")
        if self.origin_seconds is None:
            object.__setattr__(self, "origin_seconds",
                               float(self.starts[0]) if len(self.starts) else 0.0)

    @property
    def phase_starts(self) -> np.ndarray:
        return (self.starts - self.origin_seconds) / self.native_seconds

    @property
    def phase_ends(self) -> np.ndarray:
        return (self.ends - self.origin_seconds) / self.native_seconds

    @property
    def phase_span(self) -> float:
        """How many native cycles this record's declared support actually covers."""
        if not len(self.starts):
            return 0.0
        return float(self.phase_ends[-1] - self.phase_starts[0])

    @property
    def rows_per_cycle(self) -> float:
        """How finely the record resolves one native cycle, which bounds what shape it can carry."""
        span = self.phase_span
        return float(len(self.starts) / span) if span > 0.0 else 0.0


def shape_record_from_trajectory(trajectory: StructuralTrajectory, *, scale_index: int = 0,
                                 channel: str = CHANNEL) -> ShapeRecord:
    """Read one record's declared native scale off the trajectory that declares it."""
    if not trajectory.structural_scales:
        raise ShapeComparisonRefusal(
            "structural_scales", trajectory.domain,
            "at least one declared structural scale. A record that declares no native duration "
            "has no phase axis, and inventing one would be this module choosing the answer")
    scale = trajectory.structural_scales[scale_index]
    return ShapeRecord(
        label=trajectory.domain,
        starts=np.asarray(trajectory.support_start_seconds, dtype=np.float64),
        ends=np.asarray(trajectory.support_end_seconds, dtype=np.float64),
        values=np.asarray(trajectory.channels[channel], dtype=np.float64),
        valid=np.asarray(trajectory.valid_mask, dtype=bool),
        native_seconds=float(scale.native_value), native_units=str(scale.native_units))


# ------------------------------------------------------------------------------- statistic


def _shared_phase(left: ShapeRecord, right: ShapeRecord, cycles: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every phase interval both records declare inside the window, with the values there.

    A linear sweep over two sorted interval sequences, in phase rather than in seconds. Invalid
    rows contribute no weight: a gap is an absence of observation, and filling one would be this
    module supplying the very agreement it is measuring.
    """
    a_start, a_end = left.phase_starts, left.phase_ends
    b_start, b_end = right.phase_starts, right.phase_ends
    weights: List[float] = []
    lefts: List[float] = []
    rights: List[float] = []
    i = j = 0
    while i < len(a_start) and j < len(b_start):
        start = max(float(a_start[i]), float(b_start[j]), 0.0)
        end = min(float(a_end[i]), float(b_end[j]), cycles)
        if end > start and left.valid[i] and right.valid[j]:
            weights.append(end - start)
            lefts.append(float(left.values[i]))
            rights.append(float(right.values[j]))
        if a_end[i] <= b_end[j]:
            i += 1
        else:
            j += 1
    return (np.asarray(weights, dtype=np.float64), np.asarray(lefts, dtype=np.float64),
            np.asarray(rights, dtype=np.float64))


#: The resolution floor, in rows per native cycle, below which a comparison is refused rather
#: than returned. It is an operating point read off `measure_cadence_dependence`, not a law: on
#: the reference shape the statistic is unmoved to machine precision at 16 rows per cycle and
#: above, moves 2.7% at 8, 11.7% at 4 and 15.6% at 2. Those last figures are not a fault in the
#: statistic — 2 rows per cycle cannot represent a second harmonic at all, so the agreement it
#: reports is genuinely lower — but a comparison run there is measuring the sampling as much as
#: the shape, and a number that mixes the two is worse than a refusal that names the reason.
MINIMUM_ROWS_PER_CYCLE = 8.0


def shape_recurrence(left: ShapeRecord, right: ShapeRecord, *,
                     cycles: Optional[float] = None,
                     minimum_rows_per_cycle: float = MINIMUM_ROWS_PER_CYCLE) -> float:
    """How far two records trace the same profile per native cycle, on shared phase.

    The magnitude is returned, for the reason calendar mode returns one: nothing here makes two
    records' native magnitudes comparable, so a sign would be a statement about standardized
    level rather than about the world.

    `cycles` is the declared phase window. Left unset it is the shorter record's own phase span,
    so the comparison never extends past where one of the two stops declaring support - which
    would score a shape against nothing and count the absence as agreement.

    A record too coarse to resolve a cycle is refused rather than scored, at the declared floor.
    """
    if cycles is None:
        cycles = min(left.phase_span, right.phase_span)
    if not np.isfinite(cycles) or cycles <= 0.0:
        raise ShapeComparisonRefusal(
            "cycles", cycles,
            "a positive phase window both records declare support over. One of these records "
            "spans no whole native cycle, so there is no shape to compare")
    for record in (left, right):
        if record.rows_per_cycle < minimum_rows_per_cycle:
            raise ShapeComparisonRefusal(
                "rows_per_cycle", "%s: %.2f" % (record.label, record.rows_per_cycle),
                "at least %.1f rows per declared native cycle. Below that the comparison is "
                "reading the sampling as much as the shape, and a record whose declared native "
                "scale is its own row cadence carries no shape at that scale at all"
                % minimum_rows_per_cycle)
    weights, a, b = _shared_phase(left, right, float(cycles))
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


# ------------------------------------------------------------------- re-presenting a record


def represent_at(record: ShapeRecord, *, native_factor: float = 1.0,
                 cadence_factor: float = 1.0) -> ShapeRecord:
    """The same shape, declared at a different native duration and sampled at a different cadence.

    `native_factor` stretches the record's clock *and* its declared native duration together, so
    the shape per native cycle is unchanged and only the seconds it occupies differ - the
    presentation a genuinely scale-free statistic must not be able to see. `cadence_factor`
    rewrites the record at a different row density over the same span, each new row carrying the
    support-duration-weighted mean of the rows it covers, which changes the row structure without
    changing the value integral. A statistic that moves under either is reading the presentation.
    """
    if not (native_factor > 0.0) or not (cadence_factor > 0.0):
        raise ShapeComparisonRefusal(
            "factor", (native_factor, cadence_factor), "positive re-presentation factors")
    origin = float(record.origin_seconds)
    starts = origin + (record.starts - origin) * native_factor
    ends = origin + (record.ends - origin) * native_factor
    native = record.native_seconds * native_factor
    values, valid = record.values, record.valid

    if cadence_factor != 1.0:
        span_start, span_end = float(starts[0]), float(ends[-1])
        rows = max(2, int(round(len(starts) / cadence_factor)))
        edges = np.linspace(span_start, span_end, rows + 1)
        new_starts, new_ends = edges[:-1], edges[1:]
        new_values = np.zeros(rows, dtype=np.float64)
        new_valid = np.zeros(rows, dtype=bool)
        for index in range(rows):
            overlap = np.minimum(ends, new_ends[index]) - np.maximum(starts, new_starts[index])
            weight = np.where(valid, np.clip(overlap, 0.0, None), 0.0)
            total = float(np.sum(weight))
            if total > 0.0:
                new_values[index] = float(np.sum(weight * values) / total)
                new_valid[index] = True
        starts, ends, values, valid = new_starts, new_ends, new_values, new_valid

    return ShapeRecord(label=record.label, starts=starts, ends=ends, values=values, valid=valid,
                       native_seconds=native, native_units=record.native_units,
                       origin_seconds=origin)


# ------------------------------------------------------------- measuring what it may claim


#: Re-presentations the invariance measurement sweeps. The native range is sixfold, matching the
#: range over which `invariance.py` found its estimated-scale denominator to drift, so the two
#: measurements answer the same question over the same span.
NATIVE_FACTORS: Tuple[float, ...] = (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 6.0)
CADENCE_FACTORS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: How wrong a declared native duration is allowed to be in the sensitivity sweep, as fractions.
NORMALISATION_ERRORS: Tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)


def _relative(statistic: float, reference: float) -> float:
    return abs(statistic - reference) / abs(reference) if reference else abs(statistic - reference)


def measure_scale_invariance(left: ShapeRecord, right: ShapeRecord, *,
                             cycles: Optional[float] = None,
                             native_factors: Sequence[float] = NATIVE_FACTORS,
                             ) -> Dict[str, Any]:
    """Re-present the right record across native durations only, and report the movement.

    This is the measurement the mode's whole content rests on: the same shape, declared at a
    different native duration, must score the same. Nothing else is varied, because a sweep that
    changed the sampling at the same time would report one number for two effects and the
    invariance claim would be unfalsifiable — which is how `measure_cadence_dependence` came to be
    a separate function rather than another axis of this one.

    Reported as the largest *relative* deviation from the reference presentation, the quantity
    `invariance.py` reports for its own matchers, so the two are directly comparable. Nothing
    here decides whether the number is small enough: a caller declares its threshold.

    The phase window is pinned to the reference comparison rather than recomputed per
    presentation, so a re-presentation cannot quietly change *what* was compared as well.
    """
    if cycles is None:
        cycles = min(left.phase_span, right.phase_span)
    reference = shape_recurrence(left, right, cycles=cycles)
    presentations: List[Dict[str, Any]] = []
    for native_factor in native_factors:
        statistic = shape_recurrence(left, represent_at(right, native_factor=native_factor),
                                     cycles=cycles)
        presentations.append({"native_factor": float(native_factor), "statistic": statistic,
                              "relative_deviation": _relative(statistic, reference)})
    worst = max(presentations, key=lambda row: row["relative_deviation"])
    return {"schema": "shape-scale-invariance/v1", "reference_statistic": reference,
            "cycles": float(cycles), "n_presentations": len(presentations),
            "native_range": (float(min(native_factors)), float(max(native_factors))),
            "max_relative_deviation": worst["relative_deviation"], "worst_presentation": worst,
            "presentations": tuple(presentations),
            "claim_boundary": ("A measurement of one statistic under re-presentation. It is not "
                               "evidence that any shape recurs anywhere.")}


def measure_cadence_dependence(left: ShapeRecord, right: ShapeRecord, *,
                               cycles: Optional[float] = None,
                               cadence_factors: Sequence[float] = CADENCE_FACTORS,
                               ) -> Dict[str, Any]:
    """How the statistic moves when a record is rewritten at a different row density.

    Kept separate from the invariance measurement, and *not* an invariance claim, because the two
    say different things. Rewriting a record at a coarser cadence over the same span does not
    only change its presentation: past a point it destroys the shape's own detail, and a statistic
    that reported unchanged agreement there would be inventing structure the rows no longer carry.
    So movement here is expected and reported, while movement under native rescaling would be a
    fault. What this establishes is where the row density stops mattering, which is what
    `MINIMUM_ROWS_PER_CYCLE` is read off.
    """
    if cycles is None:
        cycles = min(left.phase_span, right.phase_span)
    reference = shape_recurrence(left, right, cycles=cycles)
    rows: List[Dict[str, Any]] = []
    for cadence_factor in cadence_factors:
        presented = represent_at(right, cadence_factor=cadence_factor)
        try:
            statistic = shape_recurrence(left, presented, cycles=cycles)
        except ShapeComparisonRefusal:
            rows.append({"cadence_factor": float(cadence_factor),
                         "rows_per_cycle": presented.rows_per_cycle,
                         "statistic": None, "relative_deviation": None, "refused": True})
            continue
        rows.append({"cadence_factor": float(cadence_factor),
                     "rows_per_cycle": presented.rows_per_cycle, "statistic": statistic,
                     "relative_deviation": _relative(statistic, reference), "refused": False})
    scored = [row for row in rows if not row["refused"]]
    worst = max(scored, key=lambda row: row["relative_deviation"]) if scored else None
    return {"schema": "shape-cadence-dependence/v1", "reference_statistic": reference,
            "cycles": float(cycles),
            "max_relative_deviation": worst["relative_deviation"] if worst else None,
            "worst": worst, "rows": tuple(rows),
            "claim_boundary": ("A measurement of where row density stops changing the statistic. "
                               "It is not an invariance claim and not evidence about any record.")}


#: Phase windows the sensitivity sweep reports over. A phase comparison is phase-locked, so a
#: mis-declared native duration drifts the two records apart cumulatively; how much it costs
#: therefore depends on how many cycles were compared, and reporting one window would hide that.
SENSITIVITY_CYCLES: Tuple[float, ...] = (1.0, 2.0, 6.0)


def measure_normalisation_sensitivity(left: ShapeRecord, right: ShapeRecord, *,
                                      cycles: Optional[float] = None,
                                      errors: Sequence[float] = NORMALISATION_ERRORS,
                                      ) -> Dict[str, Any]:
    """What a wrongly declared native duration costs, as a number rather than an assumption.

    The phase axis divides by a declared quantity, not a modelled one, so it imports no estimator
    drift. It does import the declaration's own error, and this reports how much: the right
    record's declared native duration is perturbed while its clock is left alone, which is
    exactly the situation of an adapter that declared the wrong native scale.
    """
    if cycles is None:
        cycles = min(left.phase_span, right.phase_span)
    reference = shape_recurrence(left, right, cycles=cycles)
    rows: List[Dict[str, Any]] = []
    for error in errors:
        mis = ShapeRecord(label=right.label, starts=right.starts, ends=right.ends,
                          values=right.values, valid=right.valid,
                          native_seconds=right.native_seconds * (1.0 + error),
                          native_units=right.native_units, origin_seconds=right.origin_seconds)
        statistic = shape_recurrence(left, mis, cycles=cycles)
        rows.append({"declared_error": float(error), "statistic": statistic,
                     "relative_deviation": _relative(statistic, reference)})
    worst = max(rows, key=lambda row: row["relative_deviation"])
    return {"schema": "shape-normalisation-sensitivity/v1", "reference_statistic": reference,
            "cycles": float(cycles), "max_relative_deviation": worst["relative_deviation"],
            "worst": worst, "rows": tuple(rows),
            "claim_boundary": ("A measurement of how far the statistic moves when the declared "
                               "native duration is wrong. It licenses no claim about any record.")}


def measure_phase_window_tradeoff(left: ShapeRecord, right: ShapeRecord, *,
                                  errors: Sequence[float] = NORMALISATION_ERRORS,
                                  window_cycles: Sequence[float] = SENSITIVITY_CYCLES,
                                  ) -> Dict[str, Any]:
    """The cost of a mis-declared native duration, as a function of how much was compared.

    This is the design trade-off the mode cannot escape and must therefore state. Comparing more
    native cycles buys statistical support, and buys it at the price of tolerance to error in the
    declared duration: the two records drift apart by the declared error once per cycle, so the
    same 10% mis-declaration that costs little over one cycle destroys the comparison over six.
    A study that chose its phase window without seeing this table would be choosing its own
    sensitivity by accident.
    """
    table: List[Dict[str, Any]] = []
    for cycles in window_cycles:
        reference = shape_recurrence(left, right, cycles=float(cycles))
        row: Dict[str, Any] = {"cycles": float(cycles), "reference_statistic": reference,
                               "costs": []}
        for error in errors:
            mis = ShapeRecord(label=right.label, starts=right.starts, ends=right.ends,
                              values=right.values, valid=right.valid,
                              native_seconds=right.native_seconds * (1.0 + error),
                              native_units=right.native_units,
                              origin_seconds=right.origin_seconds)
            statistic = shape_recurrence(left, mis, cycles=float(cycles))
            row["costs"].append({"declared_error": float(error), "statistic": statistic,
                                 "relative_deviation": _relative(statistic, reference)})
        row["costs"] = tuple(row["costs"])
        table.append(row)
    return {"schema": "shape-phase-window-tradeoff/v1", "table": tuple(table),
            "claim_boundary": ("A measurement of one trade-off in one statistic. It selects no "
                               "phase window and licenses no claim.")}


__all__ = ["CADENCE_FACTORS", "CHANNEL", "MINIMUM_ROWS_PER_CYCLE", "NATIVE_FACTORS",
           "NORMALISATION_ERRORS", "SENSITIVITY_CYCLES", "SUPPORTED_NATIVE_UNITS",
           "ShapeComparisonRefusal", "ShapeRecord", "measure_cadence_dependence",
           "measure_normalisation_sensitivity", "measure_phase_window_tradeoff",
           "measure_scale_invariance", "represent_at", "shape_record_from_trajectory",
           "shape_recurrence"]
