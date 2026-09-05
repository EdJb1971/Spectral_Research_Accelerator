"""Clock, support and coverage semantics for cross-domain comparison (TG17.4, `ed-dev`).

**What this module refuses to let happen.** Two records that arrive as arrays of equal length
look aligned. They are not. A row index is a position in a file; a *support* is the half-open
interval `[start, end)` of world over which one observation is the observation it claims to be.
Comparing by row index silently asserts that the two files were written on the same clock, and
nothing in either file says so. Every function here compares support, and the arithmetic is
interval arithmetic.

The consequence that matters is stated as an invariant and tested as one:

    **Changing row density alone cannot manufacture support.**

Splitting every hourly record into sixty minutely rows with the same support produces sixty
times the rows, the same occupied duration, the same overlap and the same effective sample
size. A pipeline that counted overlapping *pairs* would report a 3600-fold increase in evidence
for a file that gained no information at all. `effective_sample_size` is therefore derived from
overlap **duration** divided by the coarser of the two native scales, never from a row count.

**Half-open, and it bites at the boundary.** `[a, b)` and `[b, c)` do not overlap. They abut.
A closed-interval convention would report a coincidence at every boundary in every regularly
sampled record, which is the single most productive way to manufacture cross-domain structure
out of nothing.

**Nothing bins, compacts, forward-fills or interpolates by default.** The only kernel that runs
without being named is `exact_support_overlap`, which transforms nothing. Every kernel that
widens, snaps, carries or invents support is a declared adapter operation: it must be named in
the frozen manifest, it must be admitted by every participating adapter, and its consequence —
how many seconds of overlap it created that the records do not contain — is reported in
preflight before any value is opened. A kernel that invents values is refused outright over a
domain declaring `irregular_sampling`, because interpolating across an irregular clock
manufactures precisely the simultaneity the experiment exists to test for.

**The two modes cannot borrow each other's vocabulary.** Calendar mode compares UTC support and
may speak of co-occurrence; it may not silently search normalized scale ratios. Scale/shape mode
compares a normalized structural-scale coordinate that retains its mapping back to each native
duration; it may not emit simultaneity, precedence or causal language at any confidence. The
guard is a name lookup, not a convention: `assert_mode_admits_relationship` refuses the wrong
pairing by name. A study wanting both declares both and pays for the combined family.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.registry import Registry


SCHEMA = "structural-alignment/v1"

#: Relationships that only mean anything on a shared calendar, and only in calendar mode.
CALENDAR_RELATIONSHIPS: Tuple[str, ...] = (
    "co_occurrence", "precedence", "lead_lag", "causality",
)

#: Relationships that live on the normalized structural-scale coordinate. None of them is a
#: statement about when anything happened.
SCALE_SHAPE_RELATIONSHIPS: Tuple[str, ...] = (
    "scale_ratio_similarity", "shape_recurrence", "motif_recurrence",
)

MODE_RELATIONSHIPS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "calendar_aligned": CALENDAR_RELATIONSHIPS,
    "scale_shape_aligned": SCALE_SHAPE_RELATIONSHIPS,
})

#: What each mode may never say, quoted back to the researcher when it tries.
MODE_FORBIDS: Mapping[str, str] = MappingProxyType({
    "calendar_aligned": ("shared UTC support establishes co-occurrence only; semantic "
                         "equivalence, magnitude equivalence and normalized scale-ratio search "
                         "are outside this mode"),
    "scale_shape_aligned": ("a normalized structural-scale coordinate carries no clock; "
                            "simultaneity, precedence and causal language are outside this "
                            "mode at any confidence"),
})


class AlignmentRefusal(InvalidParameterError):
    """An alignment was refused by name rather than approximated."""

    def __init__(self, requirement: str, subject: Any, detail: str) -> None:
        super().__init__("alignment.%s" % requirement, subject, detail, requirement=requirement)
        self.requirement = requirement


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------- half-open interval algebra


Interval = Tuple[float, float]


def merge_intervals(starts: Sequence[float], ends: Sequence[float],
                    valid: Optional[Sequence[bool]] = None) -> Tuple[Interval, ...]:
    """The union of half-open supports, as a sorted tuple of disjoint `[start, end)` intervals.

    A union rather than a sum. Overlapping supports — a profile whose bins overlap, a light
    curve whose sectors share a month — occupy the world once, and adding their durations would
    report more coverage than the window contains.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.shape != ends.shape or starts.ndim != 1:
        raise AlignmentRefusal("support_shape", (starts.shape, ends.shape),
                               "equal-length one-dimensional support bounds")
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != starts.shape:
            raise AlignmentRefusal("support_shape", mask.shape,
                                   "a validity mask the same length as the supports")
        starts, ends = starts[mask], ends[mask]
    if starts.size == 0:
        return ()
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise AlignmentRefusal("support_finite", "non-finite",
                               "finite support bounds; an unbounded support is not a coverage "
                               "claim that can be intersected with anything")
    if np.any(ends <= starts):
        raise AlignmentRefusal(
            "support_positive", float(np.min(ends - starts)),
            "every support to satisfy start < end. A zero-width support is an instant, and an "
            "instant never intersects another half-open interval, so it would silently "
            "contribute nothing while still being counted as an observation")
    order = np.argsort(starts, kind="stable")
    merged: List[Interval] = []
    for index in order:
        start, end = float(starts[index]), float(ends[index])
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def clip_intervals(intervals: Sequence[Interval], window: Interval) -> Tuple[Interval, ...]:
    """The part of a union that lies inside a window, under half-open semantics."""
    low, high = float(window[0]), float(window[1])
    if not high > low:
        raise AlignmentRefusal("window_positive", window,
                               "a window with start < end")
    out: List[Interval] = []
    for start, end in intervals:
        lo, hi = max(start, low), min(end, high)
        if hi > lo:
            out.append((lo, hi))
    return tuple(out)


def intersect_intervals(left: Sequence[Interval],
                        right: Sequence[Interval]) -> Tuple[Interval, ...]:
    """The half-open intersection of two unions. Abutting intervals produce nothing."""
    out: List[Interval] = []
    i = j = 0
    left = list(left)
    right = list(right)
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            out.append((lo, hi))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def complement_intervals(intervals: Sequence[Interval], window: Interval) -> Tuple[Interval, ...]:
    """The gaps: the part of the window no support covers."""
    low, high = float(window[0]), float(window[1])
    out: List[Interval] = []
    cursor = low
    for start, end in clip_intervals(intervals, window):
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < high:
        out.append((cursor, high))
    return tuple(out)


def subtract_intervals(left: Sequence[Interval],
                       right: Sequence[Interval]) -> Tuple[Interval, ...]:
    """What is in `left` and not in `right`.

    Needed because "how much of this overlap did the kernel create" is not a difference of two
    durations. A kernel that snaps support onto a grid can drop one second and add another, and
    a net figure would report zero for a comparison in which nothing shared is where the records
    said it was. The honest number is the measure of the part that is new.
    """
    out: List[Interval] = []
    for start, end in left:
        cursor = start
        for other_start, other_end in right:
            if other_end <= cursor or other_start >= end:
                continue
            if other_start > cursor:
                out.append((cursor, min(other_start, end)))
            cursor = max(cursor, other_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return tuple(item for item in out if item[1] > item[0])


def occupied_seconds(intervals: Sequence[Interval]) -> float:
    return float(sum(end - start for start, end in intervals))


# ------------------------------------------------------------------------------ support profiles


@dataclass(frozen=True)
class SupportProfile:
    """What one record actually covers inside one window, and what it does not.

    `raw_row_count` is carried and reported and is used by nothing. It is here so that a report
    can show a reader the number they would have reached for, next to the number that is
    actually evidence.
    """

    label: str
    domain: str
    window: Interval
    intervals: Tuple[Interval, ...]
    native_scale_seconds: float
    raw_row_count: int
    valid_row_count: int
    support_duration_min_seconds: Optional[float]
    support_duration_max_seconds: Optional[float]
    kernel: str = "exact_support_overlap"
    kernel_parameters: Mapping[str, Any] = dc_field(default_factory=dict)
    manufactured_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", tuple((float(a), float(b))
                                                    for a, b in self.intervals))
        object.__setattr__(self, "window", (float(self.window[0]), float(self.window[1])))
        object.__setattr__(self, "kernel_parameters",
                           MappingProxyType(dict(self.kernel_parameters)))
        if not self.native_scale_seconds > 0:
            raise AlignmentRefusal("native_scale", self.native_scale_seconds,
                                   "a positive native scale; without one there is no unit in "
                                   "which overlap duration can be counted as evidence")

    # ------------------------------------------------------------------ derived facts

    @property
    def window_seconds(self) -> float:
        return self.window[1] - self.window[0]

    @property
    def occupied_seconds(self) -> float:
        return occupied_seconds(self.intervals)

    @property
    def covered_fraction(self) -> float:
        return self.occupied_seconds / self.window_seconds if self.window_seconds else 0.0

    @property
    def gaps(self) -> Tuple[Interval, ...]:
        return complement_intervals(self.intervals, self.window)

    @property
    def largest_gap_seconds(self) -> float:
        gaps = self.gaps
        return max((end - start for start, end in gaps), default=0.0)

    @property
    def support_is_stationary(self) -> bool:
        """Whether every support has the same duration.

        Non-stationary support is not an error and is not corrected. It is reported, because an
        effective sample size computed from a single governing scale is an upper bound when the
        supports underneath it are not all the same size.
        """
        low, high = self.support_duration_min_seconds, self.support_duration_max_seconds
        if low is None or high is None:
            return False
        return math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA, "label": self.label, "domain": self.domain,
            "window_start_seconds": self.window[0], "window_end_seconds": self.window[1],
            "window_seconds": self.window_seconds,
            "occupied_seconds": self.occupied_seconds,
            "covered_fraction": self.covered_fraction,
            "intervals": [list(item) for item in self.intervals],
            "gaps": [list(item) for item in self.gaps],
            "gap_count": len(self.gaps),
            "largest_gap_seconds": self.largest_gap_seconds,
            "native_scale_seconds": self.native_scale_seconds,
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "rows_are_not_evidence": ("row counts are reported and used by nothing; overlap is "
                                      "measured in seconds of support"),
            "support_is_stationary": self.support_is_stationary,
            "support_duration_min_seconds": self.support_duration_min_seconds,
            "support_duration_max_seconds": self.support_duration_max_seconds,
            "kernel": self.kernel,
            "kernel_parameters": dict(self.kernel_parameters),
            "manufactured_seconds": self.manufactured_seconds,
        }


def support_profile(*, label: str, domain: str, starts: Sequence[float], ends: Sequence[float],
                    window: Interval, native_scale_seconds: float,
                    valid: Optional[Sequence[bool]] = None) -> SupportProfile:
    """One record's coverage of one window, from its declared supports alone."""
    raw = int(np.asarray(starts).size)
    mask = None if valid is None else np.asarray(valid, dtype=bool)
    valid_rows = raw if mask is None else int(mask.sum())
    merged = merge_intervals(starts, ends, valid)
    clipped = clip_intervals(merged, window)
    widths = (np.asarray(ends, dtype=np.float64) - np.asarray(starts, dtype=np.float64))
    if mask is not None:
        widths = widths[mask]
    return SupportProfile(
        label=label, domain=domain, window=window, intervals=clipped,
        native_scale_seconds=float(native_scale_seconds), raw_row_count=raw,
        valid_row_count=valid_rows,
        support_duration_min_seconds=float(widths.min()) if widths.size else None,
        support_duration_max_seconds=float(widths.max()) if widths.size else None)


def profile_from_trajectory(trajectory: Any, window: Interval, *,
                            native_scale_seconds: Optional[float] = None,
                            label: Optional[str] = None) -> SupportProfile:
    """A `StructuralTrajectory`'s coverage, taken from the support it already carries.

    TG17.2 made the canonical record preserve exact native `[start, end)` support precisely so
    that this function would never have to reconstruct it from a cadence.
    """
    starts = np.asarray(trajectory.support_start_seconds, dtype=np.float64)
    ends = np.asarray(trajectory.support_end_seconds, dtype=np.float64)
    scale = native_scale_seconds
    if scale is None:
        widths = ends - starts
        scale = float(np.median(widths)) if widths.size else 0.0
    return support_profile(label=label or trajectory.trajectory_id, domain=trajectory.domain,
                           starts=starts, ends=ends, window=window,
                           native_scale_seconds=scale,
                           valid=np.asarray(trajectory.valid_mask, dtype=bool))


# ------------------------------------------------------------------------------ declared kernels


@dataclass(frozen=True)
class AlignmentKernel:
    """A declared, frozen operation on support. Never a default, never implicit.

    `manufactures_simultaneity` is the field that decides whether the report has to say how many
    seconds of the overlap were created rather than observed. `invents_values` marks the kernels
    that do not merely re-describe support but produce measurements nobody recorded.
    """

    name: str
    summary: str
    required_parameters: Tuple[str, ...]
    transform: Any
    manufactures_simultaneity: bool = False
    invents_values: bool = False
    refused_over_violations: Tuple[str, ...] = ()

    def resolve(self, parameters: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        supplied = dict(parameters or {})
        unknown = sorted(set(supplied) - set(self.required_parameters))
        if unknown:
            raise AlignmentRefusal(
                "kernel_parameters", unknown,
                "only the parameters kernel %r declares (%s). An ignored parameter is a setting "
                "a researcher believes is in force and is not" %
                (self.name, ", ".join(self.required_parameters) or "none"))
        resolved: Dict[str, float] = {}
        for required in self.required_parameters:
            if required not in supplied:
                raise AlignmentRefusal(
                    "kernel_parameters", required,
                    "parameter %r, which kernel %r requires. A tolerance or bin width with a "
                    "framework default would be a scientific choice nobody made" %
                    (required, self.name))
            value = float(supplied[required])
            if not math.isfinite(value) or value <= 0:
                raise AlignmentRefusal("kernel_parameters", supplied[required],
                                       "a positive finite value for %r" % required)
            resolved[required] = value
        if "minimum_bin_fraction" in resolved and resolved["minimum_bin_fraction"] > 1.0:
            raise AlignmentRefusal("kernel_parameters", resolved["minimum_bin_fraction"],
                                   "a minimum bin fraction in (0, 1]")
        return resolved

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "summary": self.summary,
                "required_parameters": list(self.required_parameters),
                "manufactures_simultaneity": self.manufactures_simultaneity,
                "invents_values": self.invents_values,
                "refused_over_violations": list(self.refused_over_violations)}


ALIGNMENT_KERNELS: Registry[AlignmentKernel] = Registry("alignment kernel")


def register_alignment_kernel(kernel: AlignmentKernel) -> AlignmentKernel:
    ALIGNMENT_KERNELS.add(kernel.name, kernel, description=kernel.summary,
                          capabilities={"manufactures_simultaneity":
                                        kernel.manufactures_simultaneity,
                                        "invents_values": kernel.invents_values})
    return kernel


def _exact(profile: SupportProfile, parameters: Mapping[str, float]) -> SupportProfile:
    return profile


def _tolerance(profile: SupportProfile, parameters: Mapping[str, float]) -> SupportProfile:
    """Widen every support by half the tolerance on each side, then re-merge.

    Half on each side rather than the whole tolerance on each side: the declared number is the
    total slack permitted *between* two records, and applying it to both profiles in full would
    quietly double it.
    """
    slack = parameters["tolerance_seconds"] / 2.0
    widened = merge_intervals([start - slack for start, _ in profile.intervals],
                              [end + slack for _, end in profile.intervals])
    return clip_to(profile, widened)


def _grid_aggregate(profile: SupportProfile, parameters: Mapping[str, float]) -> SupportProfile:
    """Snap support onto a common grid anchored at the window start.

    A bin counts as occupied only when the record actually covers at least
    `minimum_bin_fraction` of it. A bin the record barely touches is dropped rather than
    promoted: promoting it is exactly how a coarse grid manufactures agreement between records
    that share nothing but a bin edge.
    """
    width = parameters["bin_seconds"]
    floor = parameters["minimum_bin_fraction"]
    low, high = profile.window
    count = int(math.ceil((high - low) / width))
    kept: List[Interval] = []
    for index in range(count):
        edge = low + index * width
        stop = min(edge + width, high)
        covered = occupied_seconds(intersect_intervals(profile.intervals, [(edge, stop)]))
        if stop > edge and covered / (stop - edge) >= floor:
            kept.append((edge, stop))
    return clip_to(profile, merge_intervals([a for a, _ in kept], [b for _, b in kept]))


def _carry_forward(profile: SupportProfile, parameters: Mapping[str, float]) -> SupportProfile:
    """Extend each support forward by at most `maximum_carry_seconds` toward the next one."""
    carry = parameters["maximum_carry_seconds"]
    starts = [start for start, _ in profile.intervals]
    ends = [min(end + carry, profile.window[1]) for _, end in profile.intervals]
    return clip_to(profile, merge_intervals(starts, ends))


def clip_to(profile: SupportProfile, intervals: Sequence[Interval]) -> SupportProfile:
    """A profile carrying transformed support, with the seconds the transform created."""
    clipped = clip_intervals(intervals, profile.window)
    created = occupied_seconds(subtract_intervals(clipped, profile.intervals))
    return SupportProfile(
        label=profile.label, domain=profile.domain, window=profile.window, intervals=clipped,
        native_scale_seconds=profile.native_scale_seconds,
        raw_row_count=profile.raw_row_count, valid_row_count=profile.valid_row_count,
        support_duration_min_seconds=profile.support_duration_min_seconds,
        support_duration_max_seconds=profile.support_duration_max_seconds,
        kernel=profile.kernel, kernel_parameters=profile.kernel_parameters,
        manufactured_seconds=profile.manufactured_seconds + created)


register_alignment_kernel(AlignmentKernel(
    name="exact_support_overlap",
    summary=("Compare the supports the records declare and nothing else. The only kernel that "
             "runs without being named in the manifest."),
    required_parameters=(), transform=_exact))

register_alignment_kernel(AlignmentKernel(
    name="symmetric_tolerance",
    summary=("Treat supports within a declared total slack as overlapping. Widens both records "
             "by half the slack, so the declared number is the slack between them."),
    required_parameters=("tolerance_seconds",), transform=_tolerance,
    manufactures_simultaneity=True))

register_alignment_kernel(AlignmentKernel(
    name="common_grid_aggregate",
    summary=("Snap support onto a shared grid anchored at the window start; a bin counts only "
             "when the record covers at least the declared fraction of it."),
    required_parameters=("bin_seconds", "minimum_bin_fraction"), transform=_grid_aggregate,
    manufactures_simultaneity=True))

register_alignment_kernel(AlignmentKernel(
    name="carry_forward",
    summary=("Extend each support forward by at most a declared duration. Asserts that the last "
             "observation still held, which is a claim about the world, not about the file."),
    required_parameters=("maximum_carry_seconds",), transform=_carry_forward,
    manufactures_simultaneity=True, invents_values=True,
    refused_over_violations=("irregular_sampling", "aggregated_values")))


@dataclass(frozen=True)
class BoundKernel:
    """One kernel with its parameters frozen, and the digest that freezing produces."""

    kernel: AlignmentKernel
    parameters: Mapping[str, float]
    freeze_sha256: str = dc_field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "freeze_sha256", _digest(
            {"schema": SCHEMA, "kernel": self.kernel.name,
             "parameters": dict(self.parameters)}))

    @property
    def name(self) -> str:
        return self.kernel.name

    def apply(self, profile: SupportProfile) -> SupportProfile:
        transformed = self.kernel.transform(profile, self.parameters)
        return SupportProfile(
            label=transformed.label, domain=transformed.domain, window=transformed.window,
            intervals=transformed.intervals,
            native_scale_seconds=transformed.native_scale_seconds,
            raw_row_count=transformed.raw_row_count,
            valid_row_count=transformed.valid_row_count,
            support_duration_min_seconds=transformed.support_duration_min_seconds,
            support_duration_max_seconds=transformed.support_duration_max_seconds,
            kernel=self.kernel.name, kernel_parameters=dict(self.parameters),
            manufactured_seconds=transformed.manufactured_seconds)

    def describe(self) -> Dict[str, Any]:
        return {**self.kernel.describe(), "parameters": dict(self.parameters),
                "freeze_sha256": self.freeze_sha256}


def bind_kernel(name: str, parameters: Optional[Mapping[str, Any]] = None, *,
                declarations: Sequence[Any] = (),
                admissible_by_adapter: Optional[Mapping[str, Sequence[str]]] = None,
                ) -> BoundKernel:
    """Freeze one kernel against the domains that will run under it, or refuse by name."""
    if name not in ALIGNMENT_KERNELS:
        raise AlignmentRefusal(
            "kernel", name,
            "one of the registered kernels (%s). An unregistered kernel is not applied as a "
            "best effort: a comparison whose alignment operation nobody can name is a "
            "comparison nobody can reproduce" % ", ".join(sorted(ALIGNMENT_KERNELS.names())))
    kernel = ALIGNMENT_KERNELS.get(name)
    resolved = kernel.resolve(parameters)
    for declaration in declarations:
        violations = tuple(getattr(declaration, "violations", ()) or ())
        blocked = sorted(set(kernel.refused_over_violations) & set(violations))
        if blocked:
            raise AlignmentRefusal(
                "kernel_admissibility", name,
                "a kernel this domain admits. %r declares %s, and kernel %r would %s across "
                "it — which manufactures exactly the simultaneity the experiment is testing "
                "for" % (getattr(declaration, "name", "?"), " and ".join(blocked), name,
                         "invent values" if kernel.invents_values else "widen support"))
    if admissible_by_adapter:
        for adapter_id, admitted in sorted(admissible_by_adapter.items()):
            if name not in tuple(admitted):
                raise AlignmentRefusal(
                    "kernel_admissibility", name,
                    "a kernel adapter %r admits (%s). An alignment kernel is an adapter "
                    "operation: the domain that knows what its support means decides which "
                    "transformations of it remain that domain's data"
                    % (adapter_id, ", ".join(sorted(admitted)) or "none"))
    return BoundKernel(kernel=kernel, parameters=resolved)


# ------------------------------------------------------------------------------ pairwise overlap


@dataclass(frozen=True)
class PairwiseOverlap:
    """What two records share, in seconds of support, and what that is worth as evidence."""

    left: str
    right: str
    kernel: str
    kernel_parameters: Mapping[str, Any]
    intervals: Tuple[Interval, ...]
    exact_intervals: Tuple[Interval, ...]
    overlap_seconds: float
    exact_overlap_seconds: float
    left_occupied_seconds: float
    right_occupied_seconds: float
    governing_scale_seconds: float
    left_row_count: int
    right_row_count: int
    supports_are_stationary: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_parameters",
                           MappingProxyType(dict(self.kernel_parameters)))

    @property
    def manufactured_overlap_seconds(self) -> float:
        """Shared support the kernel produced that the records do not contain.

        The measure of the new part, not the difference of two totals: a kernel that drops one
        second of real overlap and adds another has manufactured a second, and a net figure
        would have reported none.
        """
        return occupied_seconds(subtract_intervals(self.intervals, self.exact_intervals))

    @property
    def effective_sample_size(self) -> float:
        """Overlap duration in units of the coarser native scale.

        The coarser scale governs. Two records overlapping for a day, one hourly and one
        monthly, share one month-scale observation and not twenty-four hourly ones: the finer
        record cannot lend resolution the coarser record does not have.
        """
        return self.overlap_seconds / self.governing_scale_seconds

    @property
    def overlap_fraction_of_shorter(self) -> float:
        shorter = min(self.left_occupied_seconds, self.right_occupied_seconds)
        return self.overlap_seconds / shorter if shorter > 0 else 0.0

    def describe(self) -> Dict[str, Any]:
        return {
            "left": self.left, "right": self.right, "kernel": self.kernel,
            "kernel_parameters": dict(self.kernel_parameters),
            "overlap_seconds": self.overlap_seconds,
            "exact_overlap_seconds": self.exact_overlap_seconds,
            "manufactured_overlap_seconds": self.manufactured_overlap_seconds,
            "lost_overlap_seconds": occupied_seconds(
                subtract_intervals(self.exact_intervals, self.intervals)),
            "overlap_intervals": [list(item) for item in self.intervals],
            "left_occupied_seconds": self.left_occupied_seconds,
            "right_occupied_seconds": self.right_occupied_seconds,
            "overlap_fraction_of_shorter": self.overlap_fraction_of_shorter,
            "governing_scale_seconds": self.governing_scale_seconds,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_size_basis": (
                "overlap seconds divided by the coarser native scale"
                if self.supports_are_stationary else
                "overlap seconds divided by the coarser native scale; an upper bound, because "
                "at least one record's support duration is not constant"),
            "row_counts_not_used": {"left": self.left_row_count, "right": self.right_row_count},
        }


def pairwise_overlap(left: SupportProfile, right: SupportProfile, *,
                     kernel: Optional[BoundKernel] = None) -> PairwiseOverlap:
    """The shared support of two records, under a kernel if one was declared."""
    if left.window != right.window:
        raise AlignmentRefusal(
            "window", (left.window, right.window),
            "both records profiled against the same window. Comparing coverage of two "
            "different intervals produces a number that describes neither")
    exact = intersect_intervals(left.intervals, right.intervals)
    a, b = (kernel.apply(left), kernel.apply(right)) if kernel else (left, right)
    shared = intersect_intervals(a.intervals, b.intervals)
    return PairwiseOverlap(
        left=left.label, right=right.label,
        kernel=kernel.name if kernel else "exact_support_overlap",
        kernel_parameters=dict(kernel.parameters) if kernel else {},
        intervals=shared, exact_intervals=exact,
        overlap_seconds=occupied_seconds(shared),
        exact_overlap_seconds=occupied_seconds(exact),
        left_occupied_seconds=a.occupied_seconds, right_occupied_seconds=b.occupied_seconds,
        governing_scale_seconds=max(left.native_scale_seconds, right.native_scale_seconds),
        left_row_count=left.raw_row_count, right_row_count=right.raw_row_count,
        supports_are_stationary=left.support_is_stationary and right.support_is_stationary)


# ----------------------------------------------------------------------------- the calendar itself


def elapsed_seconds(start: datetime, end: datetime) -> float:
    """True elapsed UTC seconds between two aware instants.

    Not `days * 86400`. A local calendar day is 23 or 25 hours across a daylight-saving
    transition, and a coverage denominator of 86400 misstates such a window by four percent in
    the direction that flatters coverage.
    """
    for label, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise AlignmentRefusal(
                "window_instant", label,
                "an instant carrying a UTC offset. A naive local timestamp is not a point in "
                "time until somebody supplies the zone, and supplying it silently is how a "
                "daylight-saving transition becomes an hour of manufactured coverage")
    return (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()


def window_bounds_seconds(start: datetime, end: datetime) -> Interval:
    """A declared window as epoch seconds, which is the only clock this module has."""
    if elapsed_seconds(start, end) <= 0:
        raise AlignmentRefusal("window_positive", (start, end), "a window with start < end")
    return (start.astimezone(timezone.utc).timestamp(),
            end.astimezone(timezone.utc).timestamp())


def nominal_day_discrepancy(start: datetime, end: datetime) -> Dict[str, Any]:
    """How far a nominal 86400-second day is from the window a researcher actually declared."""
    actual = elapsed_seconds(start, end)
    nominal = round(actual / 86400.0) * 86400.0
    return {"elapsed_seconds": actual, "nominal_seconds": nominal,
            "discrepancy_seconds": actual - nominal,
            "clock_is_uniform": math.isclose(actual, nominal, rel_tol=0.0, abs_tol=1e-6)}


# ------------------------------------------------------------------------- mode and vocabulary


def assert_mode_admits_relationship(mode: str, relationship: str) -> None:
    """Refuse the wrong mode for a relationship by name, rather than by convention."""
    if mode not in MODE_RELATIONSHIPS:
        raise AlignmentRefusal("mode", mode,
                               "one of %s" % ", ".join(sorted(MODE_RELATIONSHIPS)))
    admitted = MODE_RELATIONSHIPS[mode]
    if relationship in admitted:
        return
    other = [name for name, group in MODE_RELATIONSHIPS.items() if relationship in group]
    if other:
        raise AlignmentRefusal(
            "mode_relationship", relationship,
            "a relationship mode %r admits (%s). %r belongs to %r, and %s"
            % (mode, ", ".join(admitted), relationship, other[0], MODE_FORBIDS[mode]))
    raise AlignmentRefusal(
        "mode_relationship", relationship,
        "a declared relationship. Registered: %s"
        % ", ".join(sorted(CALENDAR_RELATIONSHIPS + SCALE_SHAPE_RELATIONSHIPS)))


def combined_family_multiplier(modes: Sequence[str]) -> Dict[str, Any]:
    """What declaring both modes costs, stated before the freeze rather than after it."""
    unique = sorted(set(modes))
    for mode in unique:
        if mode not in MODE_RELATIONSHIPS:
            raise AlignmentRefusal("mode", mode,
                                   "one of %s" % ", ".join(sorted(MODE_RELATIONSHIPS)))
    return {"modes": unique, "multiplier": len(unique),
            "correction_scope": ("one family over both modes" if len(unique) > 1
                                 else "one family over one mode"),
            "why": ("A study that looks for calendar co-occurrence and for normalized scale "
                    "recurrence has searched both, whichever one it reports. The correction is "
                    "over the union of what was searched.")}


# ---------------------------------------------------------------------------- scale/shape mode


@dataclass(frozen=True)
class ScaleShapeCorrespondence:
    """A normalized-scale match that still knows what native durations it came from."""

    left: str
    right: str
    coordinate: float
    left_native_value: float
    left_native_units: str
    right_native_value: float
    right_native_units: str
    mapping: str

    @property
    def native_ratio(self) -> float:
        return self.left_native_value / self.right_native_value

    def describe(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right, "coordinate": self.coordinate,
                "left_native_value": self.left_native_value,
                "left_native_units": self.left_native_units,
                "right_native_value": self.right_native_value,
                "right_native_units": self.right_native_units,
                "native_ratio": self.native_ratio, "mapping": self.mapping,
                "emits_simultaneity": False, "emits_precedence": False,
                "claim_boundary": MODE_FORBIDS["scale_shape_aligned"]}


def scale_shape_correspondences(left_label: str, left_scales: Sequence[Any],
                                right_label: str, right_scales: Sequence[Any], *,
                                coordinate_tolerance: float = 1e-9,
                                ) -> Tuple[ScaleShapeCorrespondence, ...]:
    """Match two records on the normalized coordinate, keeping both native durations.

    The mapping back to native duration is retained rather than discarded, so that a normalized
    match can always be reported as what it is — a shape recurring at 1.8 hours here and 46 days
    there — instead of as an unqualified similarity.
    """
    out: List[ScaleShapeCorrespondence] = []
    for left in left_scales:
        for right in right_scales:
            if abs(float(left.coordinate) - float(right.coordinate)) > coordinate_tolerance:
                continue
            out.append(ScaleShapeCorrespondence(
                left=left_label, right=right_label, coordinate=float(left.coordinate),
                left_native_value=float(left.native_value),
                left_native_units=str(left.native_units),
                right_native_value=float(right.native_value),
                right_native_units=str(right.native_units),
                mapping="%s | %s" % (left.mapping, right.mapping)))
    return tuple(out)


# ---------------------------------------------------------------------------------- the report


def alignment_report(profiles: Sequence[SupportProfile], *, mode: str,
                     kernel: Optional[BoundKernel] = None,
                     minimum_overlap_seconds: float = 0.0,
                     minimum_effective_samples: float = 0.0,
                     relationships: Sequence[str] = ()) -> Dict[str, Any]:
    """Every pair's actual shared support, and every reason a pair cannot be compared."""
    if mode not in MODE_RELATIONSHIPS:
        raise AlignmentRefusal("mode", mode,
                               "one of %s" % ", ".join(sorted(MODE_RELATIONSHIPS)))
    refusals: List[Dict[str, Any]] = []
    for relationship in relationships:
        try:
            assert_mode_admits_relationship(mode, relationship)
        except AlignmentRefusal as error:
            refusals.append({"pair": None, "relationship": relationship, "reason": str(error)})
    pairs: List[Dict[str, Any]] = []
    for index, left in enumerate(profiles):
        for right in profiles[index + 1:]:
            overlap = pairwise_overlap(left, right, kernel=kernel)
            row = overlap.describe()
            pair_refusals: List[str] = []
            if overlap.overlap_seconds <= 0.0:
                pair_refusals.append(
                    "these records share no support in this window. Half-open supports that "
                    "abut do not overlap, and no amount of resampling makes them")
            if overlap.overlap_seconds < minimum_overlap_seconds:
                pair_refusals.append(
                    "shared support is %.1f s and the manifest requires at least %.1f s"
                    % (overlap.overlap_seconds, minimum_overlap_seconds))
            if overlap.effective_sample_size < minimum_effective_samples:
                pair_refusals.append(
                    "effective sample size is %.3f at the governing scale of %.1f s and the "
                    "manifest requires at least %.3f"
                    % (overlap.effective_sample_size, overlap.governing_scale_seconds,
                       minimum_effective_samples))
            row["status"] = "REFUSED" if pair_refusals else "COMPARABLE"
            row["refusals"] = pair_refusals
            pairs.append(row)
            for reason in pair_refusals:
                refusals.append({"pair": [left.label, right.label], "relationship": None,
                                 "reason": reason})
    return {
        "schema": SCHEMA, "mode": mode,
        "mode_forbids": MODE_FORBIDS[mode],
        "admitted_relationships": list(MODE_RELATIONSHIPS[mode]),
        "kernel": kernel.describe() if kernel else
            ALIGNMENT_KERNELS.get("exact_support_overlap").describe(),
        "coverage": [profile.describe() for profile in profiles],
        "pairs": pairs, "refusals": refusals,
        "status": "REFUSED" if refusals else "COMPARABLE",
        "row_indices_were_not_compared": True,
        "claim_boundary": ("Shared support is a precondition for a comparison, not a result of "
                           "one."),
    }


__all__ = [
    "ALIGNMENT_KERNELS", "AlignmentKernel", "AlignmentRefusal", "BoundKernel",
    "CALENDAR_RELATIONSHIPS", "MODE_FORBIDS", "MODE_RELATIONSHIPS", "PairwiseOverlap",
    "SCALE_SHAPE_RELATIONSHIPS", "SCHEMA", "ScaleShapeCorrespondence", "SupportProfile",
    "alignment_report", "assert_mode_admits_relationship", "bind_kernel", "clip_intervals",
    "combined_family_multiplier", "complement_intervals", "elapsed_seconds",
    "intersect_intervals", "merge_intervals", "nominal_day_discrepancy", "occupied_seconds",
    "pairwise_overlap", "profile_from_trajectory", "register_alignment_kernel",
    "scale_shape_correspondences", "subtract_intervals", "support_profile",
    "window_bounds_seconds",
]
