"""T4E.1: a tracking pass, read as the attributed graphs `src.core.constellation` already knows.

**Why this is a bridge and not a second graph.** The attributed graph exists, in TG3.3, and it is
not a sketch: eight typed relations, each of which either divides by something the two features
carry themselves or declares that it cannot; a `RelationValue` that is dimensionless by
construction or is `None`, with no third case; relations that **refuse by name** rather than
treating an absent quantity as agreement; a carried record that R19 protects and that never
enters a comparison; and a matcher that refuses above `MAX_MATCH_NODES` instead of approximating.
Writing a second graph here would mean writing a second set of those refusals, and the second set
is the one that would be weaker. It would also produce edges in cells, which is precisely the
number TG3.3 exists to refuse.

What was genuinely missing is the enumeration, and the attributes a *track* has that a feature
does not. `constellations_from_set` takes `C(n, k)` over a whole set with no notion of a frame:
over the 8,764 frames of the T4C.5m record that is not a sweep anybody can run, and it would pair
a maximum in January with one in March. The unit of co-occurrence here is the searched frame, and
the unit of identity is the track, which is what makes an onset, an age, a velocity and a rate of
change available at all. So this module enumerates the co-present tracks of each searched frame,
hands each combination to `constellation()`, and carries the track-derived facts alongside the
graph rather than inside it.

**The split is the design.** `FrameConstellation.graph` is the comparable half: dimensionless,
domain-free, and the only half a match or a cluster may read. `FrameConstellation.nodes` and
`.relations` are the **carried** half: cells, frames, raw coefficient magnitudes, band labels.
They are exactly the numbers R19 refuses across a domain boundary, and they are here because a
researcher reading one record needs them and because T4F.5 has to project a pattern back onto the
map. Adding one of them to a comparison is a visible edit, not the consequence of a key that was
never removed -- which is the rule TG3.3 already states about its own two halves.

**Found on the way in: D90.** `distance` -- the one relation that makes a configuration
recognisable at another location -- refused on every feature the 4D line produces. T4D.2 named
the position unit `cells` and the scale unit `parent-grid px`, and TG3.3 correctly declines to
divide a separation in one unit by a scale in another. On an undecimated bank whose levels are
all mapped to the parent grid those are two spellings of one unit, so the one relation that could
have carried geometry was refusing for a spelling. Fixed in `_scale_quantity`.

**Five things the carried half is not permitted to mean.**

*A separation is between two flanks, not two structures.* T4D.2 measured why: a detail wavelet is
derivative-like, so a symmetric blob has zero response at its centre and two maxima on its flanks
about one analysing width out along the axis that band high-passes. Two nodes from different
bands are two flanks, displaced by the difference of two offsets that both grow with the
structure. On the T4D.2 vortex the `L4/LH` and `L4/HL` maxima of one vortex sit 1.44 scale lengths
apart at frame 9. That number is reported because it is the geometry the coefficients have; it is
not called a separation of structures anywhere, and `same_band` is on every relation so a caller
can decline the cross-band ones.

*A cross-band strength ratio is a ratio of filter gains until it is normalised.* Bands of a
redundant bank have no common gain, and the detection's own per-band threshold proves it: the
threshold is `sigma` times that band's RMS, so two bands with different thresholds respond on
different scales to one record. `strength_ratio_sigma` divides each strength by its own band's
RMS first. On this vortex the two disagree about the *sign* of the comparison -- raw 1.76, so the
coarse band looks stronger by three quarters; normalised 0.55, so it is the weaker of the two --
which is why both are carried and neither is called the strength ratio.

*An onset offset can be a lower bound rather than a measurement.* A track alive in the first
searched frame did not begin there; the record did. Its onset is left-censored, and an offset
with a censored end is a bound. TG3.3's `succession` is not this quantity: it orders the two
*observations*, which are in the same frame by construction, so within a constellation it is
always false. The ordering that carries information is between onsets, and it is carried here.

*A bearing in the carried half is not a compass.* It is degrees in the row-column plane, from
`+row` toward `+col`, with no grid consulted. Whether `+row` is north depends on the sign of `dy`,
and whether a column step is as long as a row step depends on the cosine of the latitude -- T4D.3
measured that omitting it rotates 45 degrees to 26.6 at 60N. The compass lives in
`spectral_narrative`, which has a grid to ask; TG3.3's `direction` is the invariant form and
refuses here, because these features carry no orientation to measure a bearing from.

*Co-occurrence in a frame is not interaction.* Two maxima share a constellation because both were
found in the same searched frame under the same threshold, and for no other reason. There is no
null, no support count and no significance in this module: T4E.3 clusters these and T4E.4 counts
support, and a constellation extracted here is one observation of one arrangement.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_tracking import COL, ROW
from src.core.constellation import (
    RELATIONS, AttributedGraph, RelationContext, constellation, measurable_relations,
    relation_for,
)
from src.core.errors import InvalidParameterError
from src.core.feature import SemanticComparisonError

CONSTELLATION_SCHEMA = "spectral-frame-constellation/v1"

#: The only cardinalities this module will build. Pairs are O(n^2) and triples O(n^3); a fourth
#: node is the beginning of frequent-subgraph mining and is refused rather than defaulted.
ALLOWED_CARDINALITIES = (2, 3)

#: Degrees in the array plane, said in full on every relation so no reader has to remember it.
BEARING_BASIS = ("degrees in the row-column plane, from +row toward +col, wrapped to [0, 360). "
                 "No grid was consulted, so this is not a compass: whether +row is north "
                 "depends on the sign of dy, and whether a column step is as long as a row step "
                 "depends on the cosine of the latitude. The invariant form is TG3.3's "
                 "'direction', which measures a bearing from a feature's own orientation")

CLAIM_BOUNDARY = (
    "A constellation is a set of coefficient maxima found in one searched frame of one "
    "transform. Co-occurrence in a frame is the only relation asserted: there is no null, no "
    "support count, no significance and no claim of interaction. The comparable half is the "
    "TG3.3 attributed graph; everything in `nodes` and `relations` is carried, in this record's "
    "own units, and R19 refuses it across a domain boundary. A separation is between two "
    "coefficient maxima, and a detail maximum sits on its structure's flank rather than its "
    "centre, so a cross-band one is a distance between two flanks and not between two "
    "structures. A raw strength ratio across bands carries the ratio of two filter gains; "
    "strength_ratio_sigma is the band-normalised form. An onset offset with "
    "onset_offset_censored set is a lower bound, because a track alive in the first searched "
    "frame did not begin there.")


# ------------------------------------------------------------------------------ small helpers


def _band(feature) -> str:
    """The band label a node came from, in the short form T4D.3 already renders."""
    provenance = dict(getattr(feature, "provenance", {}) or {})
    scale = provenance.get("scale_label")
    orientation = provenance.get("orientation_label")
    if scale is None or orientation is None:
        return "unlabelled"
    return "L%s/%s" % (scale, orientation)


def _band_rms(feature) -> Optional[float]:
    """The RMS this band's threshold was built from, or None when it was not recorded.

    The detection publishes `threshold = sigma * rms` per band together with the `sigma` it
    used, so the RMS comes back exactly by division. It is the only per-band scale the pass
    measured, and dividing a strength by it is what makes two bands' strengths comparable.
    """
    provenance = dict(getattr(feature, "provenance", {}) or {})
    threshold = provenance.get("threshold")
    sigma = provenance.get("threshold_sigma")
    try:
        threshold = float(threshold)
        sigma = float(sigma)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(threshold) and math.isfinite(sigma)) or sigma == 0.0:
        return None
    rms = threshold / sigma
    return rms if rms > 0.0 else None


def _wrap(delta: float, period: Optional[float]) -> float:
    """A signed difference on an axis that may close on itself."""
    if period is None or period <= 0.0:
        return delta
    return (delta + period / 2.0) % period - period / 2.0


def _local_rate(values: Sequence[float], times: Sequence[float], index: int,
                period: Optional[float] = None) -> Optional[float]:
    """A rate at one observation: central where it can be, one-sided at an end.

    A node is at a frame, so its rate is a local quantity. Fitting the whole track and hanging
    the slope on every node would report the same number at the beginning and the end of a
    history that changed, which is exactly the behaviour a node attribute exists to keep.
    Returns None for a single sighting, where the true statement is that no rate was measured.
    """
    if len(values) < 2:
        return None
    low = max(0, index - 1)
    high = min(len(values) - 1, index + 1)
    span = float(times[high]) - float(times[low])
    if span <= 0.0:
        return None
    return _wrap(float(values[high]) - float(values[low]), period) / span


def _axis_units(feature) -> Optional[str]:
    units = {axis.units for axis in feature.location.axes}
    if len(units) != 1:
        raise SemanticComparisonError(
            "location.axes", str(sorted(str(unit) for unit in units)), "one shared unit",
            "A separation is the square root of a sum of squared axis differences, and a "
            "difference in cells does not add to a difference in metres.")
    return units.pop()


# ------------------------------------------------------------------------------ the node


@dataclass(frozen=True)
class ConstellationNode:
    """What one track supplies at one frame that a feature alone does not -- all of it carried.

    Nothing here is dimensionless and nothing here may enter a comparison across a domain
    boundary: positions are in the grid's cells, rates are per frame of this record's clock,
    and a strength is a coefficient magnitude of this bank. The comparable view of the same node
    is `AttributedGraph.attributes`, built by TG3.3's `node_attributes`.

    Every optional field is optional for a stated reason rather than for convenience:
    `velocity` and `strength_rate` need a neighbouring observation, `orientation` needs a
    transform whose band labels are angles, and `strength_sigma` needs a detection that recorded
    its per-band threshold.
    """

    track_id: int
    time: float
    coords: Mapping[str, float]
    coord_units: Optional[str]
    band: str
    scale: float
    scale_units: Optional[str]
    strength: float
    strength_units: Optional[str]
    onset: float
    age: float
    onset_censored: bool
    orientation: Optional[float] = None
    phase: Optional[float] = None
    strength_sigma: Optional[float] = None
    strength_rate: Optional[float] = None
    scale_velocity: Optional[float] = None
    held_one_scale: bool = False
    velocity: Optional[Mapping[str, float]] = None
    observations: int = 1
    cross_scale_comparable: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", dict(self.coords))
        if self.velocity is not None:
            object.__setattr__(self, "velocity", dict(self.velocity))

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "track_id": self.track_id,
            "time": self.time,
            "coords": dict(self.coords),
            "coord_units": self.coord_units,
            "band": self.band,
            "scale": self.scale,
            "scale_units": self.scale_units,
            "strength": self.strength,
            "strength_units": self.strength_units,
            "strength_sigma": self.strength_sigma,
            "orientation": self.orientation,
            "phase": self.phase,
            "strength_rate": self.strength_rate,
            "scale_velocity": self.scale_velocity,
            "held_one_scale": self.held_one_scale,
            "velocity": dict(self.velocity) if self.velocity is not None else None,
            "onset": self.onset,
            "onset_censored": self.onset_censored,
            "age": self.age,
            "observations": self.observations,
            "carried": ("this record's own units, never compared across a domain boundary "
                        "(R19). The comparable view of this node is the graph's attributes"),
        }
        if self.strength_sigma is None:
            record["strength_sigma_refused"] = (
                "the detection did not record a per-band threshold and sigma, so this band's "
                "RMS is unknown and its strength cannot be put on a common footing")
        if self.strength_rate is None:
            record["strength_rate_refused"] = (
                "a single sighting has no rate of change, and reporting zero for one is a "
                "measurement that was never made")
        return record


# ------------------------------------------------------------------------------ the relation


@dataclass(frozen=True)
class TrackRelation:
    """The carried, track-derived record for one pair, oriented low `track_id` to high.

    The orientation is fixed rather than chosen so that a pair produces one record with one set
    of signs, and so that every field means the same thing in every constellation without a
    convention having to be looked up. None of this is the graph: the graph's edge for the same
    pair holds the dimensionless relations, and these are the numbers in cells and frames that
    a researcher reading this one record needs and that R19 refuses to export.
    """

    source: int
    target: int
    separation: float
    separation_units: Optional[str]
    offset: Mapping[str, float]
    same_band: bool
    bearing_deg: Optional[float] = None
    strength_ratio: Optional[float] = None
    strength_ratio_sigma: Optional[float] = None
    radial_velocity: Optional[float] = None
    onset_offset: Optional[float] = None
    onset_offset_censored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", dict(self.offset))

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "separation": self.separation,
            "separation_units": self.separation_units,
            "offset": dict(self.offset),
            "bearing_deg": self.bearing_deg,
            "bearing_basis": BEARING_BASIS,
            "strength_ratio": self.strength_ratio,
            "strength_ratio_sigma": self.strength_ratio_sigma,
            "radial_velocity": self.radial_velocity,
            "radial_velocity_sign": ("positive is diverging: the rate of change of the "
                                     "separation, projected on the line between the two nodes"),
            "onset_offset": self.onset_offset,
            "onset_offset_censored": self.onset_offset_censored,
            "same_band": self.same_band,
            "carried": ("this record's own units, never compared across a domain boundary "
                        "(R19). The comparable view of this pair is the graph's edge"),
        }
        if self.bearing_deg is None:
            record["bearing_refused"] = (
                "the two nodes are at the same point, and the direction from a point to "
                "itself is undefined rather than zero")
        if self.radial_velocity is None:
            record["radial_velocity_refused"] = (
                "both nodes need a local velocity and a non-zero separation to project it on")
        if self.onset_offset_censored:
            record["onset_offset_note"] = (
                "one or both onsets are at the first searched frame, so this is a lower bound "
                "on the offset and not a measurement of it")
        return record


# ------------------------------------------------------------------------------ the graph


@dataclass(frozen=True)
class FrameConstellation:
    """One TG3.3 attributed graph, plus the carried half its tracks supply.

    `graph` is the comparable half and the only half a match or a cluster may read.
    `nodes` and `relations` are in this record's units and are aligned with the graph by
    position: `nodes[i]` is the track behind `graph.attributes[i]`.

    `features` holds the `SpectralFeature` each node was built from, in the same order and
    carried for the same reason the nodes are: a later pass that wants a quantity this one
    did not compute -- T4E.2's invariant signature is the first -- must measure it from the
    coefficient rather than reconstruct it from a summary that has already rounded. It is
    absent from `describe()`, because a receipt records what was measured and the features
    are the input to a measurement, not one.
    """

    graph: AttributedGraph
    nodes: Tuple[ConstellationNode, ...]
    relations: Tuple[TrackRelation, ...]
    time: float
    time_units: Optional[str]
    features: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.features and len(self.features) != len(self.nodes):
            raise InvalidParameterError(
                "FrameConstellation.features", len(self.features),
                "one feature per node (%d), or none at all. The features are kept so a later "
                "pass can re-measure the configuration rather than re-derive it from the "
                "carried summary, and a mismatch would measure one track through another "
                "track's coefficient" % len(self.nodes))
        if len(self.nodes) not in ALLOWED_CARDINALITIES:
            raise InvalidParameterError(
                "FrameConstellation.nodes", len(self.nodes),
                "two or three nodes. Beyond three the enumeration stops being a bounded sweep "
                "and becomes frequent-subgraph mining, which is a different task with "
                "different budgets (T4E.4)")
        if self.graph.n_nodes != len(self.nodes):
            raise InvalidParameterError(
                "FrameConstellation.graph", self.graph.n_nodes,
                "a graph with one node per track (%d). The carried half is aligned with the "
                "comparable half by position, so a mismatch would attach one track's cells to "
                "another track's dimensionless attributes" % len(self.nodes))
        times = {node.time for node in self.nodes}
        if len(times) != 1:
            raise InvalidParameterError(
                "FrameConstellation.nodes", sorted(times),
                "one shared frame. Co-occurrence is the relation this graph asserts, and "
                "nodes from different frames do not co-occur")

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[ConstellationNode]:
        return iter(self.nodes)

    @property
    def cardinality(self) -> int:
        return len(self.nodes)

    @property
    def track_ids(self) -> Tuple[int, ...]:
        return tuple(node.track_id for node in self.nodes)

    @property
    def bands(self) -> Tuple[str, ...]:
        return tuple(node.band for node in self.nodes)

    def key(self) -> Tuple[Any, ...]:
        """A deterministic identity: which tracks, in which frame. Hashable, and stable."""
        return (self.time,) + self.track_ids

    def spans_scales(self) -> bool:
        return len({node.scale for node in self.nodes}) > 1

    def relation_between(self, source: int, target: int) -> Optional[TrackRelation]:
        low, high = sorted((int(source), int(target)))
        for item in self.relations:
            if (item.source, item.target) == (low, high):
                return item
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": CONSTELLATION_SCHEMA,
            "cardinality": self.cardinality,
            "time": self.time,
            "time_units": self.time_units,
            "track_ids": list(self.track_ids),
            "bands": list(self.bands),
            "spans_scales": self.spans_scales(),
            "graph": self.graph.describe(),
            "nodes": [node.describe() for node in self.nodes],
            "relations": [item.describe() for item in self.relations],
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class ConstellationSet:
    """Everything one extraction produced, with the budget it ran under attached."""

    constellations: Tuple[FrameConstellation, ...]
    frames_searched: Tuple[float, ...]
    cardinalities: Tuple[int, ...]
    relations: Tuple[str, ...] = ()
    unmeasurable: Mapping[str, str] = dc_field(default_factory=dict)
    budget: Mapping[str, Any] = dc_field(default_factory=dict)
    census: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget", dict(self.budget))
        object.__setattr__(self, "unmeasurable", dict(self.unmeasurable))
        object.__setattr__(self, "census", tuple(dict(item) for item in self.census))

    def __len__(self) -> int:
        return len(self.constellations)

    def __iter__(self) -> Iterator[FrameConstellation]:
        return iter(self.constellations)

    def __getitem__(self, index: int) -> FrameConstellation:
        return self.constellations[index]

    def of_cardinality(self, cardinality: int) -> Tuple[FrameConstellation, ...]:
        return tuple(item for item in self if item.cardinality == cardinality)

    def at(self, time: float) -> Tuple[FrameConstellation, ...]:
        return tuple(item for item in self if item.time == float(time))

    def describe(self) -> Dict[str, Any]:
        counts: Dict[int, int] = {}
        for item in self:
            counts[item.cardinality] = counts.get(item.cardinality, 0) + 1
        return {
            "schema": CONSTELLATION_SCHEMA,
            "constellations": len(self),
            "by_cardinality": {str(k): counts.get(k, 0) for k in self.cardinalities},
            "cardinalities": list(self.cardinalities),
            "relations": list(self.relations),
            "unmeasurable_relations": dict(self.unmeasurable),
            "frames_searched": len(self.frames_searched),
            "frames_with_a_constellation": len({item.time for item in self}),
            "budget": dict(self.budget),
            "census": [dict(item) for item in self.census],
            "claim_boundary": CLAIM_BOUNDARY,
            "support": ("not counted here. A constellation in this set is one observation of "
                        "one arrangement in one frame; recurrence is T4E.3's clustering and "
                        "T4E.4's support threshold, and neither has run"),
        }


# ------------------------------------------------------------------------------ extraction


def _build_node(track, index: int, first_frame: float) -> ConstellationNode:
    """One node from one observation of `track`, with the track supplying the derivatives."""
    feature = track[index]
    axes = feature.location.axes
    units = _axis_units(feature)
    times = list(track.observations.times)
    coords = {axis.name: float(feature.location.coords[axis.name]) for axis in axes}

    velocity: Optional[Dict[str, float]] = None
    if len(track) > 1:
        velocity = {}
        for axis in axes:
            period = track.periods.get(axis.name) if axis.periodic else None
            series = [float(item.location.coords[axis.name]) for item in track.observations]
            rate = _local_rate(series, times, index, period)
            if rate is None:
                velocity = None
                break
            velocity[axis.name] = rate

    strengths = [float(item.magnitude.value) for item in track.observations]
    scales = {None if item.spatial_scale is None else
              (item.spatial_scale.value, item.spatial_scale.units)
              for item in track.observations}
    held_one_scale = len(scales) == 1 and None not in scales
    scale_velocity = None
    if held_one_scale and len(track) > 1:
        # The fit is over a constant, so its slope is zero exactly. Reporting the least-squares
        # residue -- around 1e-17 on this bank -- would put a direction on a track that never
        # changed level, and a receipt reader has no way to tell that from a slow shrink.
        scale_velocity = 0.0
    elif len(track) > 1:
        try:
            scale_velocity = float(track.scale_velocity())
        except (InvalidParameterError, SemanticComparisonError):
            scale_velocity = None

    rms = _band_rms(feature)
    provenance = dict(getattr(feature, "provenance", {}) or {})
    scale = feature.spatial_scale
    if scale is None:
        raise InvalidParameterError(
            "ConstellationNode.scale", track.track_id,
            "an observation carrying a spatial scale. TG3.3's distance and relative_scale are "
            "the two relations these features can measure at all, and both divide by a scale; "
            "a node without one would put a hole in every edge it touches")

    return ConstellationNode(
        track_id=int(track.track_id),
        time=float(feature.time),
        coords=coords,
        coord_units=units,
        band=_band(feature),
        scale=float(scale.value),
        scale_units=scale.units,
        strength=float(feature.magnitude.value),
        strength_units=feature.magnitude.units,
        onset=float(track.birth_time),
        age=float(feature.time) - float(track.birth_time),
        onset_censored=bool(float(track.birth_time) <= float(first_frame)),
        orientation=(None if feature.orientation is None
                     else float(feature.orientation.degrees)),
        phase=(float(provenance["phase"]) if provenance.get("phase") is not None else None),
        strength_sigma=(None if rms is None else float(feature.magnitude.value) / rms),
        strength_rate=_local_rate(strengths, times, index),
        scale_velocity=scale_velocity,
        held_one_scale=bool(held_one_scale),
        velocity=velocity,
        observations=len(track),
        cross_scale_comparable=provenance.get("alignment_cross_scale_comparable"),
    )


def _build_relation(source: ConstellationNode, target: ConstellationNode,
                    periods: Mapping[str, float]) -> TrackRelation:
    names = sorted(set(source.coords) & set(target.coords))
    if len(names) != len(source.coords) or len(names) != len(target.coords):
        raise InvalidParameterError(
            "TrackRelation.offset", sorted(set(source.coords) ^ set(target.coords)),
            "two nodes on the same axes. A separation over axes only one of them has is a "
            "distance in a space neither node lives in")
    if source.coord_units != target.coord_units:
        raise SemanticComparisonError(
            "coord_units", str(source.coord_units), str(target.coord_units),
            "A separation between a position in cells and one in metres is not a length.")

    offset = {name: _wrap(target.coords[name] - source.coords[name], periods.get(name))
              for name in names}
    separation = math.sqrt(sum(value * value for value in offset.values()))

    bearing = None
    if separation > 0.0 and ROW in offset and COL in offset:
        bearing = math.degrees(math.atan2(offset[COL], offset[ROW])) % 360.0

    radial = None
    if separation > 0.0 and source.velocity is not None and target.velocity is not None:
        radial = sum((target.velocity[name] - source.velocity[name]) * offset[name]
                     for name in names) / separation

    strength_ratio = (None if source.strength == 0.0
                      else target.strength / source.strength)
    ratio_sigma = None
    if (source.strength_sigma not in (None, 0.0)) and target.strength_sigma is not None:
        ratio_sigma = target.strength_sigma / source.strength_sigma

    return TrackRelation(
        source=source.track_id,
        target=target.track_id,
        separation=separation,
        separation_units=source.coord_units,
        offset=offset,
        bearing_deg=bearing,
        strength_ratio=strength_ratio,
        strength_ratio_sigma=ratio_sigma,
        radial_velocity=radial,
        onset_offset=target.onset - source.onset,
        onset_offset_censored=bool(source.onset_censored or target.onset_censored),
        same_band=bool(source.band == target.band and source.band != "unlabelled"),
    )


def _require_registered(nodes: Sequence[ConstellationNode]) -> None:
    """Refuse a cross-scale constellation whose bands were never put on a common frame (D88).

    Silent when the tracking pass did not record the flag: an absent receipt is not a passing
    one, but inventing a refusal from a missing key would refuse every hand-built feature set
    in the programme for a fact nobody stated.
    """
    if len({node.scale for node in nodes}) <= 1:
        return
    offenders = sorted(node.band for node in nodes if node.cross_scale_comparable is False)
    if offenders:
        raise InvalidParameterError(
            "extract_constellations.tracking", offenders,
            "a decomposition whose scales are registered to one another (D88). Every relation "
            "here divides a separation between two bands by a scale, so an unregistered pair "
            "would report a distance that is mostly the transform's analysis delay -- tens of "
            "pixels at coarse levels, and growing with level")


def _combinations(n: int, k: int) -> int:
    return 0 if n < k else math.comb(n, k)


def _shared_periods(tracks: Sequence[Any]) -> Dict[str, float]:
    """The wrap length of each periodic axis, refused when two tracks disagree about it."""
    periods: Dict[str, float] = {}
    for track in tracks:
        for name, value in dict(track.periods).items():
            if name in periods and periods[name] != float(value):
                raise SemanticComparisonError(
                    "periods.%s" % name, str(periods[name]), str(float(value)),
                    "Two tracks on the same axis cannot disagree about where it closes.")
            periods[name] = float(value)
    return periods


def _relation_census(features: Sequence[Any],
                     names: Optional[Sequence[str]]) -> Tuple[Tuple[str, ...],
                                                              Dict[str, str]]:
    """Which of TG3.3's relations these features can support, and why the rest cannot.

    Declared once over the whole pass rather than per frame, because a family whose relation
    axis changes frame by frame has a declared size that is not the number of tests that ran --
    which is the failure `relation_axis` exists to prevent.
    """
    chosen = list(RELATIONS.names()) if names is None else list(names)
    supported = list(measurable_relations(features, chosen))
    unmeasurable: Dict[str, str] = {}
    for name in chosen:
        if name in supported:
            continue
        rule = relation_for(name)
        missing = sorted({field for feature in features for field in rule.missing_on(feature)})
        unmeasurable[name] = (
            "refused by name: these features carry no %s, and treating an absent quantity as "
            "agreement would put an edge in the receipt that constrains nothing"
            % ", ".join(missing))
    return tuple(supported), unmeasurable


def extract_constellations(
    result,
    *,
    cardinalities: Sequence[int] = (2,),
    relations: Optional[Sequence[str]] = None,
    max_nodes_per_frame: int = 24,
    max_constellations: int = 200000,
) -> ConstellationSet:
    """Enumerate the co-present tracks of every searched frame as TG3.3 attributed graphs.

    `result` is a `src.core.tracking.TrackingResult`. Every track with an observation at a
    frame is a node in that frame, and every combination of the requested sizes is emitted.
    The enumeration is complete rather than sampled, and the budgets are refusals rather than
    truncations: a run that would exceed one stops and says by how much, because a silently
    truncated sweep produces a support count in T4E.4 that is a count of what fitted.

    The graphs are built with `strict=False`, so a relation that refuses on one pair records
    the refusal on that graph instead of ending the pass. The relations *declared* are the ones
    every feature in the pass carries the fields for; the rest are reported as unmeasurable in
    the receipt rather than silently dropped.
    """
    if result is None:
        raise InvalidParameterError(
            "extract_constellations.result", None,
            "a TrackingResult. There is no constellation of nothing, and an empty set here "
            "would be indistinguishable from a frame in which nothing was found")
    sizes = tuple(int(value) for value in cardinalities)
    if not sizes or sorted(set(sizes)) != sorted(sizes) or any(
            size not in ALLOWED_CARDINALITIES for size in sizes):
        raise InvalidParameterError(
            "extract_constellations.cardinalities", list(cardinalities),
            "a set of distinct sizes drawn from %s. Pairs are O(n^2) and triples O(n^3); a "
            "fourth node is frequent-subgraph mining, which needs its own budget and its own "
            "pruning rather than one more loop here" % (list(ALLOWED_CARDINALITIES),))
    if int(max_nodes_per_frame) < max(sizes):
        raise InvalidParameterError(
            "extract_constellations.max_nodes_per_frame", max_nodes_per_frame,
            "a cap of at least the largest requested cardinality (%d), or nothing can ever be "
            "built" % max(sizes))

    tracks = list(result)
    frames = [float(t) for t in result.frame_times]
    if not tracks:
        raise InvalidParameterError(
            "extract_constellations.result", 0,
            "a tracking result with at least one track. A pass that linked nothing has no "
            "co-occurrence to report, and saying so at the call site is clearer than an "
            "empty set that could equally mean the frames were empty")
    first_frame = frames[0] if frames else min(track.birth_time for track in tracks)

    every = [item for track in tracks for item in track.observations]
    declared, unmeasurable = _relation_census(every, relations)

    # One index of observations per track, so a frame lookup is not a scan of the whole history.
    by_time: Dict[float, List[Tuple[Any, int]]] = {}
    for track in tracks:
        for index, item in enumerate(track.observations):
            by_time.setdefault(float(item.time), []).append((track, index))

    time_units = tracks[0].time_units
    built: List[FrameConstellation] = []
    census: List[Dict[str, Any]] = []
    for frame in frames:
        present = sorted(by_time.get(frame, ()), key=lambda pair: int(pair[0].track_id))
        count = len(present)
        would_emit = sum(_combinations(count, size) for size in sizes)
        census.append({"time": frame, "tracks_present": count,
                       "constellations": would_emit})
        if count == 0:
            continue
        if count > int(max_nodes_per_frame):
            raise InvalidParameterError(
                "extract_constellations.max_nodes_per_frame", count,
                "at most %d co-present tracks in a frame, or a raised cap. Frame %s holds %d, "
                "which would emit %d constellations from that frame alone. The cap refuses "
                "rather than samples, because a sampled frame makes T4E.4's support a count "
                "of what fitted in the budget"
                % (int(max_nodes_per_frame), frame, count, would_emit))
        nodes = [_build_node(track, index, first_frame) for track, index in present]
        features = [track[index] for track, index in present]
        context = RelationContext(_shared_periods([track for track, _ in present]))
        periods = dict(context.periods)
        for size in sizes:
            for chosen in itertools.combinations(range(count), size):
                picked = [nodes[i] for i in chosen]
                _require_registered(picked)
                graph = constellation([features[i] for i in chosen], relations=declared,
                                      context=context, strict=False)
                pairs = tuple(_build_relation(picked[a], picked[b], periods)
                              for a, b in itertools.combinations(range(size), 2))
                built.append(FrameConstellation(
                    graph=graph, nodes=tuple(picked), relations=pairs, time=frame,
                    time_units=time_units,
                    features=tuple(features[i] for i in chosen)))
                if len(built) > int(max_constellations):
                    raise InvalidParameterError(
                        "extract_constellations.max_constellations", len(built),
                        "a sweep that fits in %d constellations, or a raised budget. It was "
                        "exceeded at frame %s of %d searched, so the pass is refused whole "
                        "rather than returned as a prefix that looks complete"
                        % (int(max_constellations), frame, len(frames)))

    return ConstellationSet(
        constellations=tuple(built), frames_searched=tuple(frames), cardinalities=sizes,
        relations=declared, unmeasurable=unmeasurable,
        budget={"max_nodes_per_frame": int(max_nodes_per_frame),
                "max_constellations": int(max_constellations),
                "largest_frame": max((item["tracks_present"] for item in census), default=0),
                "enumeration": "complete; every combination of every searched frame"},
        census=tuple(census))


def describe_constellations(constellations: ConstellationSet) -> Dict[str, Any]:
    """The receipt for an extraction pass, with the band census that makes it readable."""
    body = dict(constellations.describe())
    pairs: Dict[Tuple[str, ...], int] = {}
    for item in constellations:
        pairs[tuple(item.bands)] = pairs.get(tuple(item.bands), 0) + 1
    body["by_bands"] = {"+".join(key): value
                        for key, value in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))}
    return body


__all__ = [
    "ALLOWED_CARDINALITIES", "BEARING_BASIS", "CLAIM_BOUNDARY", "CONSTELLATION_SCHEMA",
    "ConstellationNode", "ConstellationSet", "FrameConstellation", "TrackRelation",
    "describe_constellations", "extract_constellations",
]
