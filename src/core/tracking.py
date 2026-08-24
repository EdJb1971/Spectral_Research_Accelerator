"""TG2.3: frame-to-frame association, and the numbers a gate is not allowed to invent.

TG2.1 settled what a feature is and TG2.2 settled who decides one is there. This module
settles which feature in one frame is *the same object* as a feature in the next, which is
the first claim in the tree that is not a measurement of a single array. Nothing in a pair of
frames says two blobs are one object; association is an inference, and the whole design here
is about making the inference's assumptions visible rather than baking them into constants.

**The framework keeps what must not vary, exactly as in TG2.2.** A registered associator
receives a cost matrix and a mask of admissible pairs and returns pairs. It does not see the
features, so it cannot invent a gate, cannot bridge a gap, cannot decide what a birth is and
cannot attach anything to a record. Gating, cost construction, the R19 refusals, the track
bookkeeping and the receipt are all here. `ASSOCIATORS` holds `greedy_nearest` and
`hungarian`, and the two disagree on a case with a known answer - which is why there are two
of them rather than one with a comment.

**The gate is derived, never chosen.** A tracker's one free number is usually "how far may a
feature move between frames", and it decides how many objects exist in the same way TG2.2's
suppression radius decided how many features exist. Here the ceiling is a *coincidence*
radius: the distance at which the expected number of **unrelated** features from the target
frame falling inside the search ball equals the same `alpha` the extraction was already
calibrated at. It is computed from the frame's own feature density and the declared extent of
the search volume, so it moves when the field or the crowding moves, and it reuses a number
the receipt already carries instead of adding one.

Everything stricter than that is **declared physics** (`MotionBounds`), not tuning: a maximum
speed, a maximum scale-doubling rate, a maximum turn rate. Declared bounds are per unit time
and are multiplied by the actual elapsed time of the link, so an irregular clock does not
silently widen or narrow the gate. Nothing is declared by default, and the receipt says so.

**A gate on a quantity the extractor does not measure is refused.** The acceptance criterion
for this slice names scale and orientation gating, and the only extractor registered in
TG2.2 declares `reports_orientation: False`. Gating on an absent orientation by treating it
as "no evidence against" is the failure that would make the gate decorative: it would appear
in the receipt, refuse nothing, and be indistinguishable from a gate that was doing work.

**A missed frame ends a track.** Bridging a gap requires a motion model good enough to
predict where the object was while it was invisible, and this tree has measured no such
model. A death followed by a birth is a fact a receipt can carry and a reader can argue with;
a bridged gap is an assertion that nothing happened in between.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np

from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.extraction import DEFAULT_ALPHA, ExtractionField
from src.core.feature import (
    FeatureSet, Quantity, SemanticComparisonError, SpectralFeature,
)
from src.core.registry import Registry

DEFAULT_ASSOCIATOR = "hungarian"

#: Larger than any admissible cost by a margin no accumulation of gate fractions can reach.
_INADMISSIBLE = 1.0e9


# ------------------------------------------------------------------------ the search volume


@dataclass(frozen=True)
class SearchVolume:
    """How big the searched region is, on each declared axis, in that axis's units.

    Required rather than optional, and deliberately not inferable from the features. The
    coincidence gate is a statement about *density*, and a density needs an area; a tracker
    that guessed one from the spread of the features it happened to find would tighten its
    own gate every time it found fewer objects, which is the direction that manufactures
    tracks. For a periodic axis this length is also its period, which is the number
    `FeatureLocation.separation_to` refuses to proceed without.
    """

    extent: Mapping[str, float]
    units: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.extent:
            raise InvalidParameterError(
                "SearchVolume.extent", dict(self.extent),
                "the length of each declared axis. Without one there is no density, and "
                "without a density the gate is a number somebody picked")
        cleaned: Dict[str, float] = {}
        for name, value in self.extent.items():
            length = float(value)
            if not math.isfinite(length) or length <= 0.0:
                raise InvalidParameterError(
                    "SearchVolume.extent", value,
                    "a positive finite length for axis %r" % name)
            cleaned[str(name)] = length
        if self.units is not None and not str(self.units).strip():
            raise InvalidParameterError(
                "SearchVolume.units", self.units,
                "the units the extents are in, or None when they are not recorded")
        object.__setattr__(self, "extent", cleaned)

    @classmethod
    def of(cls, field: ExtractionField) -> "SearchVolume":
        """The volume an `ExtractionField` was searched over - its own axis lengths."""
        units = {a.units for a in field.axes}
        return cls(extent={a.name: field.axis_length(a.name) for a in field.axes},
                   units=units.pop() if len(units) == 1 else None)

    def measure(self, axis_names: Sequence[str]) -> float:
        """The product of the extents, in the shared axis units raised to the axis count."""
        total = 1.0
        for name in axis_names:
            total *= self.length(name)
        return total

    def periods(self, axes: Sequence[AxisSpec]) -> Dict[str, float]:
        """The `periods` mapping a separation asks for - only for axes that declare it."""
        return {a.name: self.length(a.name) for a in axes if a.periodic}

    def length(self, name: str) -> float:
        if name not in self.extent:
            raise InvalidParameterError(
                "SearchVolume.extent", sorted(self.extent),
                "a length for declared axis %r. The gate is computed per axis and a "
                "missing one cannot be treated as unbounded" % name)
        return float(self.extent[name])

    def describe(self) -> Dict[str, Any]:
        return {"extent": dict(self.extent), "units": self.units}


# ------------------------------------------------------------------------- declared physics


@dataclass(frozen=True)
class MotionBounds:
    """What the domain says is possible, per unit time. Declared, never inferred.

    Each bound is `None` by default, which means "not declared" and not "unbounded by
    assumption" - the receipt reports which gates were active, so a run with no declared
    physics cannot be mistaken for one whose physics happened to be satisfied.

    The bounds are rates. They are multiplied by the *actual* elapsed time of each link, so
    a sequence sampled every hour and one sampled every six hours get gates six times apart
    without either of them being re-tuned.
    """

    #: Maximum displacement per unit time, in the axes' units per the clock's units.
    max_speed: Optional[float] = None
    #: Maximum |log2(scale ratio)| per unit time - "doublings per unit time".
    max_doublings: Optional[float] = None
    #: Maximum orientation change per unit time, in degrees, under the shared convention.
    max_turn_degrees: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("max_speed", "max_doublings", "max_turn_degrees"):
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise InvalidParameterError(
                    "MotionBounds.%s" % name, value,
                    "a positive finite rate per unit time, or None for 'not declared'. "
                    "Zero is not a loose bound, it is a gate nothing can pass")
            object.__setattr__(self, name, float(value))

    @property
    def declared(self) -> Tuple[str, ...]:
        return tuple(n for n in ("max_speed", "max_doublings", "max_turn_degrees")
                     if getattr(self, n) is not None)

    def describe(self) -> Dict[str, Any]:
        return {"max_speed": self.max_speed, "max_doublings": self.max_doublings,
                "max_turn_degrees": self.max_turn_degrees,
                "declared": list(self.declared)}


NO_BOUNDS = MotionBounds()


# ------------------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class GateReport:
    """What a single frame-to-frame step gated on, and what it refused.

    A tracker's silence is as ambiguous as an extractor's: "no link" can mean the object left,
    or that the gate was too tight to admit the one true partner. This says which.
    """

    time_from: float
    time_to: float
    elapsed: float
    open_tracks: int
    observations: int
    coincidence_radius: float
    speed_radius: Optional[float]
    radius: float
    active_gates: Tuple[str, ...]
    linked: int
    refused_by_gate: int
    born: int
    died: int

    def describe(self) -> Dict[str, Any]:
        return {
            "time_from": self.time_from, "time_to": self.time_to,
            "elapsed": self.elapsed, "open_tracks": self.open_tracks,
            "observations": self.observations,
            "coincidence_radius": self.coincidence_radius,
            "speed_radius": self.speed_radius, "radius": self.radius,
            "active_gates": list(self.active_gates), "linked": self.linked,
            "refused_by_gate": self.refused_by_gate, "born": self.born, "died": self.died,
        }


def coincidence_radius(alpha: float, count: int, measure: float, dimensions: int) -> float:
    """The distance at which an unrelated neighbour is no longer surprising.

    `count` features scattered uniformly over a region of size `measure` put an expected
    `count * V_d * r**d / measure` of them inside a ball of radius `r`. Solving that for
    `alpha` gives the radius at which admitting a link on proximity alone has the same
    false-alarm rate the extraction was already calibrated at. It is not a model of how the
    object moves; it is the point past which proximity stops being evidence.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidParameterError(
            "coincidence_radius.alpha", alpha, "a false-alarm rate strictly inside (0, 1)")
    if int(count) < 1:
        raise InvalidParameterError(
            "coincidence_radius.count", count, "at least one candidate to be confused with")
    if not math.isfinite(float(measure)) or float(measure) <= 0.0:
        raise InvalidParameterError(
            "coincidence_radius.measure", measure, "a positive searched volume")
    d = int(dimensions)
    if d < 1:
        raise InvalidParameterError("coincidence_radius.dimensions", dimensions, "at least 1")
    unit_ball = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
    return float((float(alpha) * float(measure) / (int(count) * unit_ball)) ** (1.0 / d))


# ------------------------------------------------------------------ the associator registry


#: `(cost, admissible) -> pairs`. The cost matrix is tracks x observations and already
#: normalised so that an admissible pair costs at most one per active gate; `admissible` is a
#: boolean mask of the same shape. An associator sees no features, no units and no times.
Associator = Callable[[np.ndarray, np.ndarray], Sequence[Tuple[int, int]]]

ASSOCIATORS: Registry[Associator] = Registry("frame-to-frame associator")


@ASSOCIATORS.register(
    "greedy_nearest",
    description="Cheapest admissible pair first, then the next, skipping tracks and "
                "observations already used.",
    capabilities={"optimal": False, "deterministic": True, "complexity": "k log k"},
    tags=["association"])
def greedy_nearest(cost: np.ndarray, admissible: np.ndarray) -> List[Tuple[int, int]]:
    """The obvious algorithm, and it is wrong often enough to be worth keeping honest.

    Greedy takes the globally cheapest admissible pair and never revisits it, so a track that
    is *slightly* nearer to the wrong observation takes it and strands the right one. On two
    features whose paths cross it swaps both identities where the Hungarian assignment keeps
    them. It is kept registered because it is what a reader expects, and because a case with
    a known answer showing it fail is worth more than a paragraph saying it can.
    """
    order = sorted(
        ((float(cost[i, j]), i, j)
         for i in range(cost.shape[0]) for j in range(cost.shape[1]) if admissible[i, j]),
        key=lambda item: (item[0], item[1], item[2]))     # ties broken by index (E4)
    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for _, i, j in order:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((i, j))
    return pairs


@ASSOCIATORS.register(
    "hungarian",
    description="Minimum-total-cost assignment over the admissible pairs "
                "(scipy.optimize.linear_sum_assignment).",
    capabilities={"optimal": True, "deterministic": True, "complexity": "n^3"},
    tags=["association"])
def hungarian(cost: np.ndarray, admissible: np.ndarray) -> List[Tuple[int, int]]:
    """Globally optimal assignment, with inadmissible pairs priced out rather than deleted.

    Inadmissible pairs are given a cost far above any admissible one instead of being removed
    from the matrix, because a rectangular assignment with holes has no solution and the
    library would raise where the right answer is "these two do not link". Any pair the
    solver returns that is still inadmissible is dropped afterwards, so the price is a
    convenience and never a decision.
    """
    from scipy.optimize import linear_sum_assignment      # declared in requirements.txt

    if not admissible.any():
        return []
    priced = np.where(admissible, cost, _INADMISSIBLE)
    rows, cols = linear_sum_assignment(priced)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if admissible[i, j]]


# ------------------------------------------------------------------------------- the track


@dataclass(frozen=True)
class Track:
    """One object, observed in consecutive frames, plus what its history supports.

    The observations are a `FeatureSet`, so every refusal TG2.1 built is inherited rather
    than restated: a track cannot span two domains, two variables, two representations or
    two clocks, because the set it is made of will not hold them.
    """

    track_id: int
    observations: FeatureSet
    periods: Mapping[str, float] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observations, FeatureSet):
            raise InvalidParameterError(
                "Track.observations", self.observations,
                "a FeatureSet, so that the homogeneity refusals are inherited and not "
                "reimplemented")
        times = self.observations.times
        for earlier, later in zip(times, times[1:]):
            if not later > earlier:
                raise InvalidParameterError(
                    "Track.observations", list(times),
                    "strictly increasing observation times. Two observations of one object "
                    "at one instant is a contradiction, not a fast-moving object")
        object.__setattr__(self, "periods", dict(self.periods))

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[SpectralFeature]:
        return iter(self.observations)

    def __getitem__(self, index: int) -> SpectralFeature:
        return self.observations[index]

    # ------------------------------------------------------------------ when

    @property
    def birth_time(self) -> float:
        return self.observations.times[0]

    @property
    def last_time(self) -> float:
        return self.observations.times[-1]

    @property
    def time_units(self) -> Optional[str]:
        return self.observations[0].time_units

    def lifetime(self) -> Quantity:
        """Time from first to last observation, in the clock's own units."""
        return Quantity(self.last_time - self.birth_time, self._time_units())

    # ------------------------------------------------------------------ how fast

    def displacement(self) -> Dict[str, float]:
        """Total signed displacement per axis, accumulated link by link.

        Accumulated rather than taken end to end, because on a periodic axis the end-to-end
        difference is only defined modulo the period: an object that has been round twice and
        one that has not moved have the same endpoints, and the shortest-path reading of that
        is a confidently wrong zero.
        """
        axes = self.observations[0].location.axes
        totals = {a.name: 0.0 for a in axes}
        for earlier, later in zip(self.observations, self.observations[1:]):
            for axis in axes:
                delta = (later.location.coords[axis.name]
                         - earlier.location.coords[axis.name])
                if axis.periodic:
                    period = self._period(axis.name)
                    delta = (delta + period / 2.0) % period - period / 2.0
                totals[axis.name] += delta
        return totals

    def velocity(self) -> Dict[str, Quantity]:
        """Mean signed velocity per axis, in axis units per clock unit."""
        elapsed = self._elapsed()
        units = self._velocity_units()
        return {name: Quantity(total / elapsed, units)
                for name, total in self.displacement().items()}

    def speed(self) -> Quantity:
        """Magnitude of the mean velocity - refused when the axes carry mixed units."""
        components = self.velocity()
        units = {q.units for q in components.values()}
        if len(units) != 1:
            raise SemanticComparisonError(
                "velocity", str(sorted(str(u) for u in units)), "one shared unit",
                "The squares of a speed in cells per frame and one in metres per frame do "
                "not add to a speed.")
        return Quantity(math.sqrt(sum(q.value * q.value for q in components.values())),
                        units.pop())

    def scale_velocity(self) -> float:
        """Doublings of spatial scale per unit time, by least squares on log2(scale).

        Dimensionless by construction: it is built from ratios of the track's own scales,
        which TG2.1 makes the one form in which a scale may leave its domain. Refuses a
        track whose observations do not all carry a scale, because a fit over the subset
        that happens to have one is a fit over a different object's history.
        """
        self._require_length("scale_velocity")
        scales = []
        for item in self.observations:
            if item.spatial_scale is None:
                raise InvalidParameterError(
                    "Track.scale_velocity", item.time,
                    "a spatial scale on every observation. A growth rate fitted to the "
                    "frames that happened to report one is not this track's growth rate")
            scales.append(item.spatial_scale)
        units = {q.units for q in scales}
        if len(units) != 1:
            raise SemanticComparisonError(
                "spatial_scale", str(sorted(str(u) for u in units)), "one shared unit",
                "A ratio of a scale in cells to one in metres is not dimensionless.")
        times = np.asarray(self.observations.times, dtype=float)
        logs = np.log2(np.asarray([q.value for q in scales], dtype=float))
        return float(np.polyfit(times, logs, 1)[0])

    def doubling_time(self) -> Quantity:
        """The clock time in which the scale doubles - refused for a shrinking track."""
        rate = self.scale_velocity()
        if rate <= 0.0:
            raise InvalidParameterError(
                "Track.doubling_time", rate,
                "a positive growth rate. A shrinking or steady feature has no doubling "
                "time, and reporting a negative one invites it to be read as a magnitude")
        return Quantity(1.0 / rate, self._time_units())

    # ------------------------------------------------------------------ receipts

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "track_id": self.track_id,
            "length": len(self),
            "birth_time": self.birth_time,
            "last_time": self.last_time,
            "time_units": self.time_units,
            "lifetime": self.lifetime().describe() if len(self) > 1 else None,
            "domain": self.observations.domain,
            "dataset": self.observations.dataset,
            "variable": self.observations.variable,
            "representation": self.observations.representation,
            "periods": dict(self.periods),
        }
        record["velocity"] = (
            {n: q.describe() for n, q in self.velocity().items()} if len(self) > 1 else None)
        try:
            record["scale_velocity_doublings_per_time"] = (
                self.scale_velocity() if len(self) > 1 else None)
        except (InvalidParameterError, SemanticComparisonError) as exc:
            record["scale_velocity_doublings_per_time"] = None
            record["scale_velocity_refused"] = str(exc)
        return record

    # ------------------------------------------------------------------ internals

    def _require_length(self, what: str) -> None:
        if len(self) < 2:
            raise InvalidParameterError(
                "Track.%s" % what, len(self),
                "at least two observations. A single sighting has no velocity, and "
                "returning zero for one is a measurement that was never made")

    def _elapsed(self) -> float:
        self._require_length("velocity")
        return self.last_time - self.birth_time

    def _time_units(self) -> str:
        units = self.time_units
        if units is None:
            raise InvalidParameterError(
                "Track.time_units", None,
                "a clock with recorded units. 'cells per unrecorded unit' is not a rate, "
                "and a lifetime whose units are unknown cannot be compared with another")
        return units

    def _velocity_units(self) -> Optional[str]:
        axis_units = {a.units for a in self.observations[0].location.axes}
        if len(axis_units) != 1 or axis_units == {None}:
            return None
        return "%s/%s" % (axis_units.pop(), self._time_units())

    def _period(self, name: str) -> float:
        if name not in self.periods:
            raise InvalidParameterError(
                "Track.periods", name,
                "the length of periodic axis %r. A displacement across the seam cannot be "
                "computed without it, and the un-wrapped answer is wrong by an axis" % name)
        return float(self.periods[name])


# ------------------------------------------------------------------------------ the result


@dataclass(frozen=True)
class TrackingResult:
    """Every track, plus the accounting a reader needs to argue with them."""

    features: FeatureSet
    associator: str
    alpha: float
    bounds: MotionBounds
    volume: SearchVolume
    #: Every frame that was searched, including the ones the extractor found nothing in.
    times: Tuple[float, ...] = ()
    tracks: Tuple[Track, ...] = ()
    steps: Tuple[GateReport, ...] = ()

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self.tracks)

    def __getitem__(self, index: int) -> Track:
        return self.tracks[index]

    @property
    def frame_times(self) -> Tuple[float, ...]:
        return tuple(self.times)

    @property
    def empty_frames(self) -> Tuple[float, ...]:
        """Searched frames the extractor reported nothing in - a fact, not an absence."""
        found = set(self.features.times)
        return tuple(t for t in self.times if t not in found)

    @property
    def births(self) -> int:
        """Every track begins once, including the ones present in the first frame."""
        return len(self.tracks)

    @property
    def deaths(self) -> int:
        """Tracks that stopped before the last frame. A track still running is not a death."""
        if not self.frame_times:
            return 0
        last = self.frame_times[-1]
        return sum(1 for t in self.tracks if t.last_time < last)

    @property
    def longest(self) -> Optional[Track]:
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: (len(t), -t.track_id))

    def describe(self) -> Dict[str, Any]:
        return {
            "associator": self.associator,
            "alpha": self.alpha,
            "bounds": self.bounds.describe(),
            "volume": self.volume.describe(),
            "frames": len(self.frame_times),
            "empty_frames": list(self.empty_frames),
            "observations": len(self.features),
            "track_count": len(self.tracks),
            "births": self.births,
            "deaths": self.deaths,
            "gap_bridging": "none: a missed frame ends a track",
            "tracks": [t.describe() for t in self.tracks],
            "steps": [s.describe() for s in self.steps],
        }


# -------------------------------------------------------------------------------- tracking


def track(
    features: FeatureSet,
    *,
    volume: SearchVolume,
    times: Sequence[float],
    associator: str = DEFAULT_ASSOCIATOR,
    alpha: float = DEFAULT_ALPHA,
    bounds: MotionBounds = NO_BOUNDS,
) -> TrackingResult:
    """Link features frame to frame, refusing every link the declaration cannot support.

    The order of operations is the design. Gates are computed from the frame and the
    declaration; the cost is built from the gates; the registered associator chooses pairs
    *within* what the gates already allowed; and its answer is checked before it is believed.
    An associator that returned an inadmissible pair would otherwise be able to reopen every
    gate this module exists to close.

    **`times` is every frame that was searched, and it is required.** The first version of
    this function read the clock off the features it was given, and that version could not
    report a death at the end of a run: a vortex advected out of a 24-frame sequence after
    frame 5 produced a `FeatureSet` whose last frame *was* frame 5, so the track ran to the
    end of its own evidence and looked complete. Worse, an empty frame in the middle
    vanished entirely and the two frames either side were linked across it - silently
    bridging exactly the gap this module's docstring promises never to bridge. A frame in
    which the extractor found nothing is a fact about the search, and the search is the only
    thing that knows it happened.
    """
    if not isinstance(features, FeatureSet):
        raise InvalidParameterError(
            "track.features", features,
            "a FeatureSet. Tracking is arithmetic on coordinates and scales, and a bare "
            "list can hold two domains")
    if not isinstance(volume, SearchVolume):
        raise InvalidParameterError("track.volume", volume, "a SearchVolume")
    if not isinstance(bounds, MotionBounds):
        raise InvalidParameterError("track.bounds", bounds, "a MotionBounds")
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidParameterError(
            "track.alpha", alpha,
            "a false-alarm rate strictly inside (0, 1) - the same one the extraction was "
            "calibrated at, because that is what the coincidence radius means")
    entry = ASSOCIATORS.entry(associator)          # UnknownNameError, with a did-you-mean

    axes = features[0].location.axes
    axis_names = tuple(a.name for a in axes)
    _require_shared_axes(features, axis_names)
    for name in axis_names:
        volume.length(name)                        # refuse a missing extent up front
    if volume.units is not None:
        declared = {a.units for a in axes}
        if declared != {volume.units}:
            raise SemanticComparisonError(
                "search volume", str(volume.units),
                str(sorted(str(u) for u in declared)),
                "The searched extent and the feature axes are in different units, so the "
                "density the gate is derived from is not features per unit of this area.")
    periods = volume.periods(axes)
    measure = volume.measure(axis_names)
    _require_measurable_gates(features, bounds)
    clock = _require_clock(features, times)

    open_tracks: List[List[SpectralFeature]] = []
    closed: List[List[SpectralFeature]] = []
    steps: List[GateReport] = []
    previous_time: Optional[float] = None

    for time in clock:
        observations = list(features.at_time(time))
        if previous_time is None:
            open_tracks = [[item] for item in observations]
            steps.append(GateReport(
                time_from=time, time_to=time, elapsed=0.0, open_tracks=0,
                observations=len(observations), coincidence_radius=float("inf"),
                speed_radius=None, radius=float("inf"), active_gates=(), linked=0,
                refused_by_gate=0, born=len(observations), died=0))
            previous_time = time
            continue

        elapsed = float(time) - float(previous_time)
        heads = [chain[-1] for chain in open_tracks]
        radius_alpha = (coincidence_radius(alpha, len(observations), measure, len(axes))
                        if observations else float("inf"))
        radius_speed = (bounds.max_speed * elapsed
                        if bounds.max_speed is not None else None)
        radius = min(r for r in (radius_alpha, radius_speed) if r is not None)

        cost, admissible, active = _score(
            heads, observations, periods=periods, radius=radius,
            elapsed=elapsed, bounds=bounds)
        pairs = _associate(entry.value, associator, cost, admissible)

        linked_rows = {i for i, _ in pairs}
        linked_cols = {j for _, j in pairs}
        next_open: List[List[SpectralFeature]] = []
        for i, chain in enumerate(open_tracks):
            if i not in linked_rows:
                closed.append(chain)
        for i, j in pairs:
            next_open.append(open_tracks[i] + [observations[j]])
        for j, item in enumerate(observations):
            if j not in linked_cols:
                next_open.append([item])

        steps.append(GateReport(
            time_from=float(previous_time), time_to=float(time), elapsed=elapsed,
            open_tracks=len(open_tracks), observations=len(observations),
            coincidence_radius=radius_alpha, speed_radius=radius_speed, radius=radius,
            active_gates=active, linked=len(pairs),
            refused_by_gate=int(admissible.size - int(admissible.sum())),
            born=len(observations) - len(linked_cols),
            died=len(open_tracks) - len(linked_rows)))
        open_tracks = next_open
        previous_time = time

    chains = closed + open_tracks
    chains.sort(key=lambda c: (c[0].time,
                               tuple(c[0].location.coords[n] for n in axis_names)))
    tracks = tuple(
        Track(track_id=i, observations=FeatureSet(chain), periods=periods)
        for i, chain in enumerate(chains))
    return TrackingResult(features=features, associator=associator, alpha=float(alpha),
                          bounds=bounds, volume=volume, times=clock, tracks=tracks,
                          steps=tuple(steps))


def track_extractions(results: Sequence[Any], **kwargs: Any) -> Optional[TrackingResult]:
    """Track a sequence of `ExtractionResult`s, keeping the frames that found nothing.

    The convenience that exists because forgetting the empty frames is the mistake: this
    takes the searched clock and the searched volume from the results themselves, so the
    caller cannot supply a clock shorter than the run. Returns `None` when no frame produced
    a feature, because a `FeatureSet` refuses to be empty and "nothing anywhere" is a
    statement about the search rather than a tracking result with zero tracks.
    """
    ordered = sorted(results, key=lambda r: r.field.time)
    if not ordered:
        raise InvalidParameterError(
            "track_extractions.results", [],
            "at least one searched frame. A clock with no frames on it is not a sequence")
    found = [f for r in ordered for f in r]
    if not found:
        return None
    kwargs.setdefault("volume", SearchVolume.of(ordered[0].field))
    kwargs.setdefault("times", [r.field.time for r in ordered])
    return track(FeatureSet(found), **kwargs)


# ------------------------------------------------------------------------------ internals


def _require_clock(features: FeatureSet, times: Sequence[float]) -> Tuple[float, ...]:
    """The searched frames, ascending and distinct, covering every feature's time."""
    clock = tuple(float(t) for t in times)
    if not clock:
        raise InvalidParameterError(
            "track.times", [],
            "every frame that was searched. Inferring the clock from the features found "
            "makes the last frame with a feature in it the end of the run, and no track "
            "can then die at the end of one")
    for earlier, later in zip(clock, clock[1:]):
        if not later > earlier:
            raise InvalidParameterError(
                "track.times", list(clock),
                "strictly increasing searched times; %r does not follow %r"
                % (later, earlier))
    known = set(clock)
    missing = sorted({f.time for f in features} - known)
    if missing:
        raise InvalidParameterError(
            "track.times", missing,
            "a clock containing every feature's time. A feature at a frame the run does "
            "not list is a feature from another run, and it would be tracked into this one")
    return clock


def _require_shared_axes(features: FeatureSet, axis_names: Tuple[str, ...]) -> None:
    for item in features:
        if item.location.axis_names != axis_names:
            raise SemanticComparisonError(
                "location", ", ".join(axis_names), ", ".join(item.location.axis_names),
                "The features are on different axes, so no two of them have a separation.")


def _require_measurable_gates(features: FeatureSet, bounds: MotionBounds) -> None:
    """A gate on a quantity no feature carries is refused before a single link is made.

    Treating a missing scale or orientation as "no evidence against the link" makes the gate
    appear in the receipt while refusing nothing, and the run then looks like one whose
    physics was satisfied rather than one whose physics was never tested.
    """
    if bounds.max_doublings is not None:
        missing = [f.time for f in features if f.spatial_scale is None]
        if missing:
            raise InvalidParameterError(
                "track.bounds.max_doublings", bounds.max_doublings,
                "a scale gate only when every feature carries a spatial scale; %d of %d "
                "do not (first at time %r). The extractor that produced them declares "
                "whether it reports one" % (len(missing), len(features), missing[0]))
    if bounds.max_turn_degrees is not None:
        missing = [f.time for f in features if f.orientation is None]
        if missing:
            raise InvalidParameterError(
                "track.bounds.max_turn_degrees", bounds.max_turn_degrees,
                "an orientation gate only when every feature carries an orientation; %d of "
                "%d do not (first at time %r). An extractor declaring "
                "reports_orientation=False cannot be gated on one"
                % (len(missing), len(features), missing[0]))


def _score(heads: Sequence[SpectralFeature], observations: Sequence[SpectralFeature], *,
           periods: Mapping[str, float], radius: float, elapsed: float,
           bounds: MotionBounds) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Cost and admissibility for every (open track, observation) pair.

    Each active gate contributes the square of the fraction of itself that the pair used, so
    an admissible pair costs at most one per gate and the gates *are* the weights. There is
    no relative weighting to choose, which is the point: a hand-set trade-off between "how
    far it moved" and "how much it grew" is another free number deciding how many objects
    exist.
    """
    rows, cols = len(heads), len(observations)
    cost = np.zeros((rows, cols), dtype=float)
    admissible = np.ones((rows, cols), dtype=bool)
    active: List[str] = ["position"]
    if bounds.max_doublings is not None:
        active.append("scale")
    if bounds.max_turn_degrees is not None:
        active.append("orientation")

    for i, head in enumerate(heads):
        for j, item in enumerate(observations):
            fraction = item.separation_to(head, periods=periods).value / radius
            total = fraction * fraction
            ok = fraction <= 1.0
            if bounds.max_doublings is not None:
                share = (abs(math.log2(item.scale_ratio_to(head)))
                         / (bounds.max_doublings * elapsed))
                total += share * share
                ok = ok and share <= 1.0
            if bounds.max_turn_degrees is not None:
                share = (item.orientation.separation_to(head.orientation)
                         / (bounds.max_turn_degrees * elapsed))
                total += share * share
                ok = ok and share <= 1.0
            cost[i, j] = total
            admissible[i, j] = ok
    return cost, admissible, tuple(active)


def _associate(associator: Associator, name: str, cost: np.ndarray,
               admissible: np.ndarray) -> List[Tuple[int, int]]:
    """Run a registered associator and refuse an answer the gates did not allow."""
    if cost.size == 0:
        return []
    pairs = [(int(i), int(j)) for i, j in associator(cost, admissible)]
    rows: Set[int] = set()
    cols: Set[int] = set()
    for i, j in pairs:
        if not 0 <= i < cost.shape[0] or not 0 <= j < cost.shape[1]:
            raise InvalidParameterError(
                "associator.%s" % name, (i, j),
                "a pair of indices inside the %dx%d cost matrix" % cost.shape)
        if not admissible[i, j]:
            raise InvalidParameterError(
                "associator.%s" % name, (i, j),
                "a pair the gates admitted. An associator that can widen a gate makes "
                "every refusal in this module advisory")
        if i in rows or j in cols:
            raise InvalidParameterError(
                "associator.%s" % name, (i, j),
                "each track and each observation used at most once. One observation "
                "continuing two tracks is a merge, and this module does not claim one")
        rows.add(i)
        cols.add(j)
    return sorted(pairs)
