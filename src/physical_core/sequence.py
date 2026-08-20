"""`FieldSequence`: the time axis the rest of Phase 4 rests on (roadmap T4A.1, T4A.2).

**Why this did not exist before, and why nothing above it could.** `PhysicalField` is strictly
2D and raises on any other rank. Every transform, statistic and benchmark in the platform
operates on one snapshot. The research question the whole project exists to answer -
*which small configurations at t precede which large structures at t+delta* - could not even be
**stated**, because nothing represented "t". `decompose_by_lead_time` only appeared to work
because the caller assembled the list of fields itself, so the time axis lived in a local
variable and vanished the moment the function returned.

A `FieldSequence` is an ordered set of `PhysicalField`s plus a real time coordinate, with the
consistency the rest of Phase 4 will assume checked **once, here**, rather than re-derived
(differently) by every consumer:

*   every frame has the same shape, or the sequence is refused;
*   every frame has a compatible grid, or the sequence is refused - a sequence whose frames sit
    on different grids is not a sequence of the same region, and averaging across it is
    meaningless in a way no downstream test would catch;
*   times are strictly increasing, so "the next frame" is unambiguous and a lag in frames can be
    converted to a lag in hours.

**Irregular sampling is allowed but never silently.** ERA5 is 1-hourly or 6-hourly, but a
concatenation of two crops, or a record with a gap, is not. `cadence` reports the *modal* spacing
and `is_regular` says whether every step matches it; anything converting a lag in frames to a
physical lag must consult those rather than assume.

**R6 lives here** (`split_temporal`). A predictive claim evaluated without a strict temporal
split and an embargo is trivially fakeable, and retrofitting a split after seeing results is how
a platform talks itself into a discovery. The API deliberately mirrors the existing *spatial*
guardrail (`PhysicalField.split_field` / `validate_split_guardrails`) so the discipline reads the
same on both axes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.physical_core.field import PhysicalField

#: Fractional tolerance for calling two sampling intervals "the same". Floating-point
#: timestamps and datetime64 conversions do not land on exactly equal deltas, and demanding
#: exact equality would report every real ERA5 record as irregular.
CADENCE_RTOL = 1e-6

TimeLike = Union[Sequence[float], Sequence[Any], np.ndarray, torch.Tensor]


def _to_seconds(times: TimeLike) -> Tuple[np.ndarray, str]:
    """Normalise a time coordinate to float seconds, reporting what it was.

    Accepts numeric offsets (kept as-is, unit unknown) or numpy datetime64 / anything
    `numpy` can coerce to it. Returning the *kind* alongside the values matters: a lag of
    "6" means six hours in one case and six arbitrary units in the other, and the difference
    has to survive into the provenance rather than be assumed by whoever reads it next.
    """
    if isinstance(times, torch.Tensor):
        arr = times.detach().cpu().numpy()
    else:
        arr = np.asarray(times)

    if arr.dtype.kind == "M":  # datetime64
        seconds = arr.astype("datetime64[ns]").astype("int64") / 1e9
        return seconds.astype(np.float64), "datetime64"
    if arr.dtype.kind in "iuf":
        return arr.astype(np.float64), "numeric"
    # Strings and objects: let numpy try datetime64 before giving up, because an ISO date
    # list is the most common hand-written case.
    try:
        seconds = np.asarray(arr, dtype="datetime64[ns]").astype("int64") / 1e9
        return seconds.astype(np.float64), "datetime64"
    except (TypeError, ValueError):
        raise InvalidParameterError(
            "times", arr.dtype,
            "numeric offsets or datetime64-compatible timestamps (ISO strings are accepted)")


class FieldSequence:
    """An ordered sequence of `PhysicalField`s sharing a grid, plus a time coordinate."""

    def __init__(
        self,
        fields: Sequence[PhysicalField],
        times: TimeLike,
        metadata: Optional[Dict[str, Any]] = None,
        split: Optional[str] = None,
    ) -> None:
        fields = list(fields)
        if not fields:
            raise InvalidParameterError(
                "fields", [], "at least one PhysicalField. An empty sequence has no shape, "
                              "no grid and no time axis, so nothing downstream can validate "
                              "against it")

        seconds, time_kind = _to_seconds(times)
        if seconds.ndim != 1:
            raise InvalidParameterError("times", seconds.shape, "a 1D time coordinate")
        if len(seconds) != len(fields):
            raise ShapeMismatchError("times", (len(seconds),), "fields", (len(fields),),
                                     fix="Supply exactly one timestamp per frame.")

        reference = fields[0]
        for index, field in enumerate(fields):
            if not isinstance(field, PhysicalField):
                raise InvalidParameterError(
                    "fields[%d]" % index, type(field).__name__, "a PhysicalField")
            if field.data.shape != reference.data.shape:
                raise ShapeMismatchError(
                    "frame 0", tuple(reference.data.shape),
                    "frame %d" % index, tuple(field.data.shape),
                    fix="Every frame of a sequence must be the same shape; a ragged "
                        "sequence cannot be stacked, differenced or transformed as a block.")
            # Grid equality is checked on the *metric*, not on object identity: two frames
            # cropped from the same archive are separate GridSpec instances describing the
            # same region, and refusing those would make the class unusable on real data.
            if not _same_grid(reference.grid, field.grid):
                raise InvalidParameterError(
                    "fields[%d].grid" % index, field.grid.describe(),
                    "the same grid as frame 0 (%s). Frames on different grids are not "
                    "observations of one region, and any statistic computed across them "
                    "would be averaging different places together"
                    % reference.grid.describe())

        order = np.argsort(seconds, kind="stable")
        if not np.array_equal(order, np.arange(len(seconds))):
            raise InvalidParameterError(
                "times", "unsorted",
                "strictly increasing timestamps. Sorting them here would silently reorder "
                "the caller's frames, and a sequence whose order was corrected without "
                "anyone noticing is worse than one that is refused")
        deltas = np.diff(seconds)
        if len(deltas) and np.any(deltas <= 0):
            duplicated = int(np.argmin(deltas)) + 1
            raise InvalidParameterError(
                "times", float(seconds[duplicated]),
                "strictly increasing timestamps; frame %d repeats or precedes frame %d. "
                "Duplicate timestamps make 'the next frame' ambiguous, and every lag "
                "computed from this sequence would be wrong by an unknown amount"
                % (duplicated, duplicated - 1))

        self.fields: List[PhysicalField] = fields
        self.times_seconds = seconds
        self.time_kind = time_kind
        self.raw_times = times
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.split = split

    # ------------------------------------------------------------------ shape and access

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, index):
        """Integer index returns a frame; a slice returns a sub-sequence, not a list.

        Returning a bare list from a slice would drop the time coordinate, which is exactly
        the failure this class exists to prevent.
        """
        if isinstance(index, slice):
            return FieldSequence(self.fields[index], self.raw_times_slice(index),
                                 metadata=dict(self.metadata), split=self.split)
        return self.fields[index]

    def raw_times_slice(self, index: slice):
        """The original time values for a slice, preserving their original type."""
        if isinstance(self.raw_times, torch.Tensor):
            return self.raw_times[index]
        return np.asarray(self.raw_times)[index]

    @property
    def n_frames(self) -> int:
        return len(self.fields)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """`(T, H, W)` - the shape `to_tensor()` produces."""
        return (len(self.fields), self.fields[0].height, self.fields[0].width)

    @property
    def grid(self):
        return self.fields[0].grid

    @property
    def units(self) -> Optional[str]:
        return self.fields[0].units

    def at(self, t: int) -> PhysicalField:
        """Frame by index, with an error that names the range rather than an IndexError."""
        if not -len(self.fields) <= t < len(self.fields):
            raise InvalidParameterError(
                "t", t, "a frame index in 0..%d (negative indexing allowed)"
                        % (len(self.fields) - 1))
        return self.fields[t]

    def at_time(self, value: Any, tolerance: Optional[float] = None) -> PhysicalField:
        """Frame by *timestamp*, refusing an inexact match rather than returning the nearest.

        Returning the nearest frame is the tempting behaviour and the wrong one: a caller who
        asked for 06:00 and silently received 12:00 has a six-hour error in a lag calculation
        with nothing to indicate it.
        """
        target, _kind = _to_seconds([value])
        deltas = np.abs(self.times_seconds - target[0])
        nearest = int(np.argmin(deltas))
        allowed = tolerance if tolerance is not None else 0.0
        if deltas[nearest] > allowed:
            raise InvalidParameterError(
                "value", value,
                "a timestamp present in the sequence. The nearest frame is %d, off by %.6g "
                "seconds; pass `tolerance` explicitly if an approximate match is intended"
                % (nearest, float(deltas[nearest])))
        return self.fields[nearest]

    # ------------------------------------------------------------------ cadence

    @property
    def deltas_seconds(self) -> np.ndarray:
        return np.diff(self.times_seconds)

    @property
    def cadence_seconds(self) -> Optional[float]:
        """The modal sampling interval, or None for a single frame."""
        deltas = self.deltas_seconds
        if len(deltas) == 0:
            return None
        return float(np.median(deltas))

    @property
    def is_regular(self) -> bool:
        """Whether every step matches the cadence to within `CADENCE_RTOL`.

        Reported rather than enforced. A concatenation of two crops, or an archive with a gap,
        is a legitimate sequence - but converting "3 frames" into "18 hours" is only valid when
        this is true, so the fact has to be available to whoever does the converting.
        """
        deltas = self.deltas_seconds
        if len(deltas) == 0:
            return True
        cadence = self.cadence_seconds or 0.0
        if cadence <= 0:
            return False
        return bool(np.all(np.abs(deltas - cadence) <= CADENCE_RTOL * cadence))

    def lag_to_seconds(self, lag_frames: int) -> float:
        """Convert a lag in frames to a lag in seconds, refusing when that is meaningless."""
        if not self.is_regular:
            raise InvalidParameterError(
                "lag_frames", lag_frames,
                "a lag on a regularly sampled sequence. This sequence has gaps (intervals "
                "%s seconds), so N frames is not a fixed duration and any lag expressed in "
                "frames would mean different things in different parts of the record"
                % sorted({round(float(d), 3) for d in self.deltas_seconds}))
        cadence = self.cadence_seconds
        if cadence is None:
            raise InvalidParameterError(
                "lag_frames", lag_frames, "a sequence with at least two frames")
        return float(lag_frames) * cadence

    # ------------------------------------------------------------------ transformation

    def map(self, fn: Callable[[PhysicalField], PhysicalField],
            metadata: Optional[Dict[str, Any]] = None) -> "FieldSequence":
        """Apply `fn` per frame, keeping the time axis attached.

        The point of the method is that the result is a `FieldSequence` and not a list: a
        `[fn(f) for f in seq]` comprehension is one character shorter and loses the times,
        which is exactly how the time axis went missing before this class existed.
        """
        mapped = []
        for index, field in enumerate(self.fields):
            result = fn(field)
            if not isinstance(result, PhysicalField):
                raise InvalidParameterError(
                    "fn(frame %d)" % index, type(result).__name__,
                    "a PhysicalField. `map` preserves the sequence structure, so the "
                    "function must return a field; use `to_tensor()` for arbitrary output")
            mapped.append(result)
        merged = dict(self.metadata)
        merged.update(metadata or {})
        return FieldSequence(mapped, self.raw_times, metadata=merged, split=self.split)

    def to_tensor(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Stack to `(T, H, W)`."""
        stacked = torch.stack([f.data for f in self.fields], dim=0)
        return stacked.to(dtype) if dtype is not None else stacked

    def mean_field(self) -> PhysicalField:
        """The temporal mean as a `PhysicalField`, carrying the grid and units forward."""
        data = self.to_tensor().mean(dim=0)
        return PhysicalField(data, coords=self.fields[0].coords,
                             metadata={**self.fields[0].metadata,
                                       "reduction": "temporal_mean",
                                       "n_frames": len(self.fields)},
                             grid=self.grid)

    def anomalies(self) -> "FieldSequence":
        """Frames with the temporal mean removed - the usual first step before any spectrum.

        Not climatology removal: that is `analysis_engine.climatology`, which fits a harmonic
        model and needs a long record. This is the plain mean, and it says so in the metadata
        so the two cannot be confused in a lineage row.
        """
        mean = self.mean_field().data
        return self.map(
            lambda f: PhysicalField(f.data - mean, coords=f.coords,
                                    metadata={**f.metadata, "anomaly": "temporal_mean_removed"},
                                    grid=f.grid),
            metadata={"anomaly": "temporal_mean_removed"})

    # ------------------------------------------------------------------ provenance

    def summary(self) -> Dict[str, Any]:
        """A lineage-safe description: shape and statistics, never the payload."""
        stacked = self.to_tensor()
        return {
            "type": "FieldSequence",
            "n_frames": len(self.fields),
            "shape": list(self.shape),
            "grid": self.grid.to_provenance(),
            "units": self.units,
            "time_kind": self.time_kind,
            "time_start": _format_time(self.raw_times, 0),
            "time_end": _format_time(self.raw_times, len(self.fields) - 1),
            "cadence_seconds": self.cadence_seconds,
            "is_regular": self.is_regular,
            "split": self.split,
            "min": float(stacked.min()),
            "max": float(stacked.max()),
            "mean": float(stacked.mean()),
            "metadata": dict(self.metadata),
        }


def _same_grid(a, b) -> bool:
    """Grid equality on the metric, not on identity."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    if a.kind != b.kind:
        return False
    for attr in ("dx", "dy", "lat0", "lon0"):
        left, right = getattr(a, attr, None), getattr(b, attr, None)
        if left is None and right is None:
            continue
        if left is None or right is None:
            return False
        if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12):
            return False
    return True


def _format_time(times, index: int) -> str:
    if isinstance(times, torch.Tensor):
        return str(times[index].item())
    return str(np.asarray(times)[index])


# --------------------------------------------------------------------------- R6: the split

def split_temporal(
    sequence: FieldSequence,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    embargo_frames: int = 0,
) -> Dict[str, FieldSequence]:
    """Split a sequence in time with an embargo gap between the parts (rule R6).

    **Why an embargo and not just a boundary.** Atmospheric fields are strongly autocorrelated:
    the frame immediately after the train/test boundary is nearly a copy of the frame
    immediately before it. Testing on it measures persistence, not prediction. The embargo is
    the number of frames *discarded* at each boundary, and it must be at least as long as the
    longest lag under test - otherwise the target of a training example lies inside the test
    window, and the model has been shown its own answer.

    The discarded frames are returned under ``embargo_train_val`` and ``embargo_val_test``
    rather than silently dropped, so a lineage record shows what was excluded and a reader can
    check the gap was real.
    """
    if not 0 < train_ratio < 1:
        raise InvalidParameterError("train_ratio", train_ratio, "a fraction in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise InvalidParameterError("val_ratio", val_ratio, "a fraction in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise InvalidParameterError(
            "train_ratio + val_ratio", train_ratio + val_ratio,
            "a total below 1.0, leaving a non-empty test window")
    if embargo_frames < 0:
        raise InvalidParameterError("embargo_frames", embargo_frames, "a non-negative count")

    n = len(sequence)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    n_gaps = 1 if val_ratio == 0 else 2
    if train_end <= 0 or val_end >= n or (val_ratio > 0 and val_end <= train_end):
        raise InvalidParameterError(
            "ratios", (train_ratio, val_ratio),
            "ratios that leave every requested window non-empty for %d frames" % n)
    if embargo_frames * n_gaps >= n - train_end:
        raise InvalidParameterError(
            "embargo_frames", embargo_frames,
            "an embargo that leaves data after it. %d frames x %d gap(s) consumes everything "
            "after the training window (%d frames). Use a longer record or a smaller embargo - "
            "but do NOT shrink the embargo below the longest lag under test, because that "
            "reintroduces exactly the leakage it exists to prevent."
            % (embargo_frames, n_gaps, n - train_end))

    def part(start: int, stop: int, name: Optional[str]) -> FieldSequence:
        sub = FieldSequence(sequence.fields[start:stop],
                            sequence.raw_times_slice(slice(start, stop)),
                            metadata=dict(sequence.metadata), split=name)
        return sub

    result: Dict[str, FieldSequence] = {}
    if val_ratio > 0:
        result["train"] = part(0, train_end, "train")
        if embargo_frames:
            result["embargo_train_val"] = part(train_end, train_end + embargo_frames, "embargo")
        result["val"] = part(train_end + embargo_frames, val_end, "val")
        if embargo_frames:
            result["embargo_val_test"] = part(val_end, val_end + embargo_frames, "embargo")
        result["test"] = part(val_end + embargo_frames, n, "test")
    else:
        result["train"] = part(0, train_end, "train")
        if embargo_frames:
            result["embargo_train_test"] = part(train_end, train_end + embargo_frames, "embargo")
        result["test"] = part(train_end + embargo_frames, n, "test")

    for name, sub in result.items():
        if len(sub) == 0:
            raise InvalidParameterError(
                "ratios/embargo", (train_ratio, val_ratio, embargo_frames),
                "a configuration leaving every window non-empty; %r came out empty" % name)
        sub.metadata["temporal_split"] = {
            "name": name, "embargo_frames": embargo_frames,
            "train_ratio": train_ratio, "val_ratio": val_ratio,
            "n_frames_total": n,
        }
    return result


def validate_temporal_guardrails(
    a: FieldSequence,
    b: FieldSequence,
    max_lag_frames: int = 0,
) -> bool:
    """Raise if two splits overlap in time, or if the gap is shorter than the longest lag.

    The counterpart of `PhysicalField.validate_split_guardrails` on the time axis, and the
    thing T4A.2's acceptance criterion asks for: a deliberate leakage attempt must raise.

    Two distinct failures, because they have different causes and different fixes:

    *   **Overlap** - the windows share timestamps. Someone sliced them wrongly.
    *   **Insufficient embargo** - the windows are disjoint but adjacent, and a training
        example at the end of the train window has its *target* inside the test window. The
        split looks clean and leaks anyway, which is why this one needs checking explicitly
        rather than being assumed away by "they do not overlap".
    """
    if a.split is not None and b.split is not None and a.split == b.split:
        return True

    a_start, a_end = float(a.times_seconds[0]), float(a.times_seconds[-1])
    b_start, b_end = float(b.times_seconds[0]), float(b.times_seconds[-1])

    if not (a_end < b_start or b_end < a_start):
        raise ValueError(
            "Temporal leakage guardrail violated: split %r spans [%s, %s] and split %r spans "
            "[%s, %s] - they overlap. A predictive claim evaluated across overlapping windows "
            "is measuring memorisation."
            % (a.split, _format_time(a.raw_times, 0), _format_time(a.raw_times, -1),
               b.split, _format_time(b.raw_times, 0), _format_time(b.raw_times, -1)))

    if max_lag_frames > 0:
        earlier, later = (a, b) if a_end < b_start else (b, a)
        gap_seconds = float(later.times_seconds[0] - earlier.times_seconds[-1])
        cadence = earlier.cadence_seconds or later.cadence_seconds
        if cadence is None or cadence <= 0:
            raise ValueError(
                "Cannot verify the embargo: neither split has enough frames to establish a "
                "cadence, so 'a gap of %d frames' has no duration to compare against."
                % max_lag_frames)
        required = max_lag_frames * cadence
        # Strictly less, with a tolerance: a gap of exactly one lag means the last training
        # target lands on the first test frame, which is the leak.
        if gap_seconds <= required * (1.0 + CADENCE_RTOL):
            raise ValueError(
                "Temporal leakage guardrail violated: the gap between split %r and split %r "
                "is %.6g seconds (%.2f frames), but the longest lag under test is %d frames "
                "(%.6g seconds). A training example near the boundary has its target inside "
                "the other window, so the split is clean-looking and leaks. Increase the "
                "embargo to more than %d frames."
                % (earlier.split, later.split, gap_seconds, gap_seconds / cadence,
                   max_lag_frames, required, max_lag_frames))
    return True
