"""T4E.2: the invariant signature of a frame constellation, and the toggle that is a claim.

T4E.1 read a tracking pass as TG3.3's attributed graphs. This slice asks what survives when
the same configuration is presented somewhere else on the grid, turned, or resized -- and it
answers with TG3.4's matchers rather than with a second set of normalisations, for the same
reason T4E.1 did not write a second graph.

**Why this is a bridge and not a second normalisation.**

TG3.4 already measured what T4E.2's specification asks for, and its findings are not the ones
the specification assumes:

*   *"distances normalised by the participating features' scales"* is TG3.3's `distance`
    relation, and TG3.4 measured that it is **not** rescaling-invariant on real extracted
    features. It divides a separation by an *estimated* spatial scale, the estimate drifts
    from about +3.6% at sigma 3 to about -2.6% at sigma 18, and the whole of that drift lands
    in the quotient: 4.8% movement at `scale_factor=3` against a 3.3% noise floor. TG3.4
    deliberately left it out of `MATCHERS` because a registry entry carries a declaration and
    this one has none it can demonstrate.
*   *"translation and rotation invariance therefore fall out of the representation"* is true,
    and rescaling invariance does not fall out with them. What survives a rescaling is a
    separation divided by *another separation* -- TG3.4's `relative_geometry`, measured to
    reproduce to 0.37%. The price is that it needs three features: two features have one
    separation, and its ratio to itself is 1 for every configuration in the world.

So the two halves of the roadmap's paragraph are two registered matchers that already exist,
and the honest thing to add here is the **toggle** that chooses between them, the two blocks
neither of them measures (bearings and strengths), and the count of what each mode costs.

**The toggle, precisely.**

`scale_invariant=False` -- *scale-specific*. Geometry is `distance`: separation over the
geometric mean of the members' own spatial scales. Available at cardinality 2 and 3.
Translation- and rotation-invariant; **not** rescaling-invariant, for the reason above. The
scale block carries the members' absolute scales in this record's units, so the signature does
not cross a domain boundary.

`scale_invariant=True` -- *the universality hook*. Geometry is `shape_ratio`: separation over
the geometric mean of every separation in the configuration. No estimated quantity enters, so
translation, rotation and rescaling invariance are all by construction. The scale block keeps
only the ratios between the members' scales, which is dimensionless, so this is the mode -- the
only mode -- that can be compared across a domain boundary. It needs three nodes.

The scale block moves with the toggle too, and that is the second half of the question the
specification asks. Scale-specific keeps the members' *absolute* scales in the vector, so a
configuration of the same shape two octaves down is a different configuration -- "does this
recur at *this* scale?". Scale-invariant keeps only the ratios between them, so it is the same
one -- "does this recur at *other* scales?". The absolute scales are carried in both modes and
comparable in only one, which is why the scale-specific signature stops at the domain boundary:
its vector holds a length in cells, and a length in cells has no ratio to a length in metres.

**What the toggle costs, measured.** On the T4D.2 vortex pass, turning it on removes **87 of
135** constellations from the pass -- every pair -- because a pair has no shape. That is the
comparison the specification asks to be able to run, and it has a number rather than an
argument.

**Found on the way in: the benchmark's own configuration has no principal axis.**

*"Bearings measured relative to the constellation's own principal axis"* assumes the
configuration has one. `planted_configuration` -- the benchmark the programme supplies for
invariance, and the one TG3.4's gate runs on -- is an **equilateral** triangle. Its position
covariance is isotropic, so the axis is whatever the noise decided. Measured over 24 field-noise
realisations of the same planting: the anisotropy `lambda_1 / lambda_2` stays between 1.0077 and
1.0421, while the recovered axis angle scatters from 0.78 to 158.08 degrees -- effectively
uniform over the half-circle, a circular standard deviation near 50 degrees -- and the shape
ratios over the same replicates reproduce to 0.218%. A bearing block written without a guard
would have emitted a confident angle that was pure noise, on the exact configuration the
roadmap nominates for testing invariance.

`AXIS_ISOTROPY_FLOOR` is that measurement and not a choice: below the largest anisotropy a
known-isotropic configuration produced under noise, the observed elongation is not
distinguishable from noise and the axis is refused by name. Clearing the floor is a minimum and
not a precision claim -- a configuration just above it still has a poorly determined angle. The
vortex triples clear it by two orders of magnitude (smallest observed 85.22), which is why the
bearings on this record are usable at all.

**Bearings are a re-expression at these cardinalities, not new information.** Three points'
pairwise separations determine the triangle up to similarity and reflection, so the angles the
edges make with the principal axis are a function of the shape block rather than an addition to
it. They are kept because they are the readable form and because T4F must project a
configuration back onto a map, not because they add a degree of freedom. At four nodes they
would; the enumeration stops at three.

**Reflection.** An edge is unordered and a principal axis has no sign, so the only well-defined
angle between them lies in [0, 90] degrees. That makes the bearing block invariant to
reflection as well as rotation -- a consequence, not a preference. Distinguishing a
configuration from its mirror image would need an orientation convention on the grid, and these
features declare none: `node_attributes` reports `has_orientation: False` for every one of them,
which is the same refusal T4D.3 made when it declined to give a compass word to a grid that
never said which way was north.

**Canonical order, and why the blocks are not sorted independently.** A signature must not
depend on which track happened to be listed first, so the vector is minimised
lexicographically over all node permutations -- six of them at most, since the enumeration
stops at three. Sorting each block on its own would be cheaper and wrong: two configurations
could then agree on sorted separations and sorted strengths with no single correspondence that
makes both true at once, which is a matcher that reports agreement it cannot exhibit.

**What this slice does not do.** It does not cluster, count support, or decide that two
signatures are the same -- deciding sameness needs a tolerance, a tolerance must be calibrated
against replicates rather than chosen (TG3.4's `calibrate_match_tolerance`), and that is T4E.3.
Nothing here reports a p-value, a null or a discovery. The comparison between the two modes is
a comparison of what each can *express and refuse*, not of what either found.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_constellation import ConstellationSet, FrameConstellation
from src.core.constellation import RelationContext
from src.core.errors import InvalidParameterError
from src.core.invariance import SHAPE_RATIO, relative_geometry, scale_normalised
from src.core.registry import Registry


@dataclass(frozen=True)
class SignatureMode:
    """Declared identity semantics; adding a mode uses this registry, not a matcher fork."""

    geometry: Callable
    relation: str
    invariance: Tuple[str, ...]
    strengths: bool = True
    scales: bool = True
    scale_ratios: bool = False
    requires_scope: bool = False


SIGNATURE_MODES: Registry[SignatureMode] = Registry("spectral signature mode")

#: The receipt schema for one signature.
SIGNATURE_SCHEMA = "spectral-constellation-signature/v1"

#: The two modes, and the matcher each one keys on.
SCALE_MODES: Mapping[str, str] = {
    "scale_specific": "distance",
    "scale_invariant": SHAPE_RATIO,
}

#: What each mode is invariant to. `scale_specific` omits rescaling because TG3.4 measured
#: that it does not have it, not because it was not tried.
MODE_INVARIANCE: Mapping[str, Tuple[str, ...]] = {
    "scale_specific": ("translation", "rotation", "reflection"),
    "scale_invariant": ("translation", "rotation", "reflection", "rescaling"),
}

#: The largest anisotropy `lambda_1 / lambda_2` that 24 field-noise realisations of a
#: *known-isotropic* configuration produced on the `planted_configuration` benchmark, with the
#: extraction pipeline that feeds it. Over those replicates the anisotropy stayed in
#: [1.0077, 1.0421] while the axis angle scattered from 0.78 to 158.08 degrees. Below this
#: value an observed elongation is not distinguishable from noise, so the axis is refused.
#: `calibrate_axis_admission` re-measures it; it is a floor and not a precision claim.
AXIS_ISOTROPY_FLOOR = 1.0421

#: How many replicates the floor above was measured over.
AXIS_FLOOR_REPLICATES = 24

BEARING_BASIS = (
    "degrees between an edge and the configuration's own principal axis, folded to [0, 90]. "
    "An edge is unordered and an axis has no sign, so no larger range is defined; the fold "
    "makes the block invariant to reflection as well as to rotation, which is a consequence "
    "of the grid declaring no orientation rather than a choice.")

STRENGTH_BASIS = (
    "each member's coefficient magnitude divided by its own band's RMS, then divided by the "
    "geometric mean of the members' band-normalised strengths. The band normalisation is not "
    "optional: a raw ratio across two bands is a ratio of filter gains, and on this record the "
    "raw and normalised ratios disagree about which member is the stronger.")

CLAIM_BOUNDARY = (
    "A signature is a description of one configuration in one frame. It is not a match, not a "
    "cluster, not a count of support and not a finding. Two equal signatures are two "
    "descriptions that agree; deciding whether that agreement means anything needs a tolerance "
    "calibrated against replicates and a null, and neither is measured here.")


# --------------------------------------------------------------------- the principal axis


def _coord_axes(constellation: FrameConstellation) -> Tuple[str, ...]:
    """The spatial axis names the members are located on.

    There is deliberately no check here that the members agree about their axes or their
    units. A constellation whose members disagree cannot be built: T4E.1 pools the frame's
    features into a `FeatureSet`, and a position in cells beside one in metres is refused there
    by name -- `coord_units: 'm vs cells'` -- before anything reaches this module. A second
    copy of that refusal would be a branch no call can take, which is a worse defect than the
    duplication, because it reads like a guard and guards nothing. A configuration on more
    than two axes does reach here, and `principal_axis` refuses it.
    """
    return tuple(sorted(constellation.nodes[0].coords))


def principal_axis(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """The anisotropy and the axis angle of a set of planar positions.

    Returns `(lambda_1 / lambda_2, angle_deg)` with the angle in [0, 180) measured from the
    first axis toward the second. The anisotropy is infinite for exactly collinear points,
    which is a well-determined axis rather than a failure; it is 1 for an isotropic
    arrangement, where the angle means nothing and `admit_axis` refuses it.
    """
    rows = [tuple(float(value) for value in point) for point in points]
    if len(rows) < 2 or any(len(row) != 2 for row in rows):
        raise InvalidParameterError(
            "principal_axis.points", [len(row) for row in rows],
            "at least two positions, each on exactly two axes. A principal axis in three "
            "dimensions is a different quantity and this module does not measure it")
    centre = [sum(row[i] for row in rows) / len(rows) for i in (0, 1)]
    centred = [(row[0] - centre[0], row[1] - centre[1]) for row in rows]
    cxx = sum(a * a for a, _ in centred) / len(rows)
    cyy = sum(b * b for _, b in centred) / len(rows)
    cxy = sum(a * b for a, b in centred) / len(rows)
    half = (cxx + cyy) / 2.0
    root = math.sqrt(max(((cxx - cyy) / 2.0) ** 2 + cxy * cxy, 0.0))
    major, minor = half + root, half - root
    if major <= 0.0:
        raise InvalidParameterError(
            "principal_axis.points", 0.0,
            "positions that are not all identical. Points at one place have no direction "
            "between them")
    ratio = float("inf") if minor <= 0.0 else major / minor
    if abs(cxy) < 1e-15 and abs(cxx - cyy) < 1e-15:
        angle = 0.0
    else:
        angle = math.degrees(math.atan2(major - cxx, cxy)) if abs(cxy) > 0.0 else (
            0.0 if cxx >= cyy else 90.0)
    return ratio, angle % 180.0


@dataclass(frozen=True)
class AxisAdmission:
    """Whether the configuration has a principal axis, and the measurement that decided it."""

    admitted: bool
    anisotropy: float
    floor: float
    angle_deg: Optional[float]
    refusal: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "admitted": bool(self.admitted),
            "anisotropy": None if math.isinf(self.anisotropy) else float(self.anisotropy),
            "collinear": bool(math.isinf(self.anisotropy)),
            "floor": float(self.floor),
            "floor_basis": ("the largest anisotropy %d noise realisations of a known-isotropic "
                            "configuration produced; clearing it is a minimum, not a precision "
                            "claim" % AXIS_FLOOR_REPLICATES),
            "angle_deg": None if self.angle_deg is None else float(self.angle_deg),
        }
        if self.refusal is not None:
            body["refused"] = self.refusal
        return body


def admit_axis(constellation: FrameConstellation, *,
               floor: float = AXIS_ISOTROPY_FLOOR) -> AxisAdmission:
    """Measure the configuration's principal axis, and refuse it when it is noise.

    The refusal is by name and carries the number that produced it, so a reader can tell a
    configuration that had no axis from one whose axis was never asked for.
    """
    axes = _coord_axes(constellation)
    points = [[node.coords[name] for name in axes] for node in constellation.nodes]
    ratio, angle = principal_axis(points)
    if constellation.cardinality < 3:
        return AxisAdmission(
            False, ratio, float(floor), None,
            "two members lie on their own separation, so the principal axis is that edge and "
            "every bearing relative to it is zero by construction")
    if ratio <= float(floor):
        return AxisAdmission(
            False, ratio, float(floor), None,
            "anisotropy %.4f does not clear the isotropy floor %.4f, so the axis direction is "
            "not distinguishable from the one noise alone produces on a configuration with no "
            "axis at all" % (ratio, float(floor)))
    return AxisAdmission(True, ratio, float(floor), angle)


def calibrate_axis_admission(
        replicates: Sequence[Sequence[Sequence[float]]]) -> Tuple[float, Dict[str, Any]]:
    """Re-measure the isotropy floor from replicates of one configuration.

    The floor is the largest anisotropy the replicates produced, in the same spirit as TG3.4's
    `calibrate_match_tolerance`: the operating point is what the noise did, not what an author
    thought reasonable. Hand it replicates of an *isotropic* planting -- calibrating on an
    elongated one produces a floor that refuses everything.
    """
    rows = [principal_axis(points) for points in replicates]
    if len(rows) < 2:
        raise InvalidParameterError(
            "calibrate_axis_admission.replicates", len(rows),
            "at least two replicates. One realisation measures a configuration; a floor is a "
            "statement about the spread of realisations and one of them has none")
    ratios = [ratio for ratio, _ in rows]
    if any(math.isinf(ratio) for ratio in ratios):
        raise InvalidParameterError(
            "calibrate_axis_admission.replicates", "collinear",
            "replicates that are not exactly collinear. A collinear replicate has infinite "
            "anisotropy, and a floor at infinity refuses every configuration there is")
    angles = [angle for _, angle in rows]
    resultant = abs(sum(complex(math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a)))
                        for a in angles) / len(angles))
    spread = (float("inf") if resultant <= 0.0
              else math.degrees(math.sqrt(max(-2.0 * math.log(resultant), 0.0))) / 2.0)
    return max(ratios), {
        "replicates": len(rows),
        "anisotropy_min": min(ratios),
        "anisotropy_max": max(ratios),
        "angle_min_deg": min(angles),
        "angle_max_deg": max(angles),
        "angle_circular_sd_deg": spread,
        "basis": "largest anisotropy observed; the angle spread says what it was hiding",
    }


# ------------------------------------------------------------------------- the signature


@dataclass(frozen=True)
class ConstellationSignature:
    """One configuration, expressed so that moving it does not change it.

    `vector()` is the comparable half and the only half a match may read. Everything named
    `*_carried` or held in `scales` under the scale-specific mode is in this record's own
    units and stops at the domain boundary.
    """

    key: Tuple[Any, ...]
    time: float
    cardinality: int
    mode: str
    order: Tuple[int, ...]
    geometry: Tuple[float, ...]
    geometry_relation: str
    bearings: Optional[Tuple[float, ...]]
    strengths: Tuple[float, ...]
    scales: Tuple[float, ...]
    scale_units: Optional[str]
    axis: AxisAdmission
    track_ids: Tuple[int, ...]
    bands: Tuple[str, ...]
    refusals: Mapping[str, str] = dc_field(default_factory=dict)
    time_units: Optional[str] = None
    comparison_scope: Optional[str] = None
    geometry_units: Optional[str] = None
    strengths_carried: Tuple[Optional[float], ...] = ()
    comparison_source: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in SIGNATURE_MODES:
            raise InvalidParameterError("ConstellationSignature.mode", self.mode,
                                        "one of %s. The mode names which quantity the geometry "
                                        "was divided by, or declares raw record separation; "
                                        "unknown semantics cannot be compared" % SIGNATURE_MODES.names())
        specification = SIGNATURE_MODES.get(self.mode)
        if specification.requires_scope and (
                not isinstance(self.comparison_scope, str) or not self.comparison_scope.strip()
                or not self.geometry_units or not self.comparison_source):
            raise InvalidParameterError(
                "ConstellationSignature.comparison_scope", self.comparison_scope,
                "a declared record/grid comparison scope and geometry units; spatial "
                "separations cannot be compared just because two records both say cells")
        expected = self.cardinality * (self.cardinality - 1) // 2
        if len(self.geometry) != expected:
            raise InvalidParameterError(
                "ConstellationSignature.geometry", len(self.geometry),
                "one value per unordered pair (%d for %d members)"
                % (expected, self.cardinality))
        if self.bearings is not None and len(self.bearings) != expected:
            raise InvalidParameterError(
                "ConstellationSignature.bearings", len(self.bearings),
                "one bearing per edge (%d), or none at all when the axis was refused"
                % expected)

    @property
    def scale_invariant(self) -> bool:
        return self.mode == "scale_invariant"

    @property
    def invariant_to(self) -> Tuple[str, ...]:
        return SIGNATURE_MODES.get(self.mode).invariance

    @property
    def cross_domain_comparable(self) -> bool:
        """Only the scale-invariant mode is. The other carries absolute scales in cells."""
        return self.scale_invariant

    def vector(self) -> Tuple[float, ...]:
        """The comparable numbers, in canonical order: geometry, bearings, strengths, scales.

        The scale block is where the toggle bites a second time, and it is the difference
        between the two questions the specification names. Under `scale_invariant` it holds the
        *ratios* between the members' scales, so a configuration of the same shape two octaves
        down is the same configuration -- "does this recur at other scales?". Under
        `scale_specific` it holds the members' **absolute** scales, in this record's cells, so
        that configuration is a different one -- "does this recur at this scale?".

        That is why only one of the two modes crosses a domain boundary, and it is a fact about
        the vector rather than a policy about it: the scale-specific vector contains a length in
        cells, and a length in cells has no ratio to a length in metres. T4E.3's clustering
        declares a weighting per attribute, which is where a block in cells acquires one.
        """
        return tuple(value for block in self.comparison_blocks().values() for value in block)

    def comparison_blocks(self) -> Dict[str, Tuple[float, ...]]:
        """One authority for vector(), scalar matching and accelerated matching.

        In spatial_geometry the detector's bands and strengths remain carried metadata.
        They are absent blocks, not zero-weight observations or fabricated unit values.
        """
        specification = SIGNATURE_MODES.get(self.mode)
        return {
            "geometry": self.geometry,
            "bearings": () if self.bearings is None else self.bearings,
            "strengths": self.strengths if specification.strengths else (),
            "scales": ((self._scale_ratios() if specification.scale_ratios else self.scales)
                       if specification.scales else ()),
        }

    def _scale_ratios(self) -> Tuple[float, ...]:
        reference = math.exp(sum(math.log(value) for value in self.scales) / len(self.scales))
        return tuple(value / reference for value in self.scales)

    def describe(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema": SIGNATURE_SCHEMA,
            "key": list(self.key),
            "time": self.time,
            "time_units": self.time_units,
            "cardinality": self.cardinality,
            "mode": self.mode,
            "scale_invariant": self.scale_invariant,
            "geometry": list(self.geometry),
            "geometry_relation": self.geometry_relation,
            "geometry_basis": (
                "separation over the geometric mean of every separation in the configuration; "
                "no estimated quantity enters" if self.scale_invariant else
                "separation over the geometric mean of the members' estimated spatial scales; "
                "TG3.4 measured that the estimate drifts with the scale being estimated, so "
                "this quantity is not invariant to a rescaling"),
            "bearings_deg": None if self.bearings is None else list(self.bearings),
            "bearings_basis": BEARING_BASIS,
            "bearings_independence": (
                "at three members the bearings are a function of the geometry block rather "
                "than an addition to it; they are kept because they are the readable form"),
            "strengths": list(self.strengths),
            "strengths_basis": STRENGTH_BASIS,
            "scales": list(self.scales),
            "scale_units": self.scale_units,
            "scale_ratios": list(self._scale_ratios()),
            "scale_block_in_vector": "ratios" if self.scale_invariant else "absolute, in cells",
            "axis": self.axis.describe(),
            "canonical_order": list(self.order),
            "canonical_order_basis": (
                "the node permutation that minimises the whole vector lexicographically; the "
                "blocks are not sorted independently, because that would let two "
                "configurations agree with no single correspondence that makes both blocks "
                "true at once"),
            "invariant_to": list(self.invariant_to),
            "cross_domain_comparable": self.cross_domain_comparable,
            "track_ids": list(self.track_ids),
            "bands": list(self.bands),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        if self.refusals:
            body["refusals"] = dict(self.refusals)
        if SIGNATURE_MODES.get(self.mode).requires_scope:
            body.update({
                "schema": "spectral-constellation-signature/v2",
                "comparison_scope": self.comparison_scope,
                "comparison_source": list(self.comparison_source),
                "geometry_units": self.geometry_units,
                "geometry_basis": "pairwise spatial separations in the declared record/grid",
                "strengths_carried": list(self.strengths_carried),
                "strengths_basis": "band-RMS-normalised strengths carried only; not compared",
                "scale_block_in_vector": "absent; detector scales are carried only",
                "comparable_blocks": list(name for name, block in
                                           self.comparison_blocks().items() if block),
                "claim_boundary": (
                    "Exploratory spatial identity within one declared record/grid. "
                    "Invariant to detector band and magnitude at fixed positions, not to "
                    "physical evolution, rescaling, latitude-dependent grid metrics or "
                    "cross-domain transfer. Track continuity is a proxy label, not ground truth."),
            })
        return body


@dataclass(frozen=True)
class SignatureSet:
    """Everything one signing pass produced, with what it could not sign counted beside it."""

    signatures: Tuple[ConstellationSignature, ...]
    mode: str
    considered: int
    refused: Mapping[str, int] = dc_field(default_factory=dict)
    axis_refused: int = 0
    floor: float = AXIS_ISOTROPY_FLOOR
    comparison_scope: Optional[str] = None

    def __len__(self) -> int:
        return len(self.signatures)

    def __iter__(self) -> Iterator[ConstellationSignature]:
        return iter(self.signatures)

    def __getitem__(self, index: int) -> ConstellationSignature:
        return self.signatures[index]

    @property
    def scale_invariant(self) -> bool:
        return self.mode == "scale_invariant"

    def of_cardinality(self, cardinality: int) -> Tuple[ConstellationSignature, ...]:
        return tuple(item for item in self.signatures
                     if item.cardinality == int(cardinality))

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": SIGNATURE_SCHEMA,
            "mode": self.mode,
            "scale_invariant": self.scale_invariant,
            "geometry_relation": SIGNATURE_MODES.get(self.mode).relation,
            "considered": self.considered,
            "signed": len(self.signatures),
            "refused": dict(self.refused),
            "axis_refused": self.axis_refused,
            "isotropy_floor": self.floor,
            "invariant_to": list(SIGNATURE_MODES.get(self.mode).invariance),
            "cross_domain_comparable": self.scale_invariant,
            "by_cardinality": {str(size): len(self.of_cardinality(size))
                               for size in sorted({item.cardinality
                                                   for item in self.signatures})},
            "claim_boundary": CLAIM_BOUNDARY,
            **({"comparison_scope": self.comparison_scope,
                "schema": "spectral-constellation-signature/v2"}
               if SIGNATURE_MODES.get(self.mode).requires_scope else {}),
        }


# ------------------------------------------------------------------------- the measurement


def _edge_pairs(size: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(itertools.combinations(range(size), 2))


def _geometry_values(constellation: FrameConstellation, mode: str,
                     context: RelationContext) -> Dict[Tuple[int, int], float]:
    """The edge quantity for one mode, taken from TG3.4's matcher rather than recomputed.

    Nothing here handles a missing edge. Both matchers raise rather than record: T4E.1 builds
    its graphs with `strict=False` so that one refused pair does not end a whole sweep, but a
    *signature* is the configuration as a whole, and `relative_geometry` and `scale_normalised`
    both refuse the configuration outright when a separation cannot be taken or a quotient is
    not dimensionless. A shape assembled from the edges that happened to work would be a shape
    with a hole in it, and neither matcher will hand one over.
    """
    features = list(constellation.features)
    graph = (relative_geometry(features, context) if mode == "scale_invariant"
             else scale_normalised(features, context))
    wanted = SCALE_MODES[mode]
    values: Dict[Tuple[int, int], float] = {}
    for pair in _edge_pairs(len(features)):
        edge = graph.edges.get(pair) or graph.edges.get((pair[1], pair[0]))
        values[pair] = float(edge[wanted].value)
    return values


def _bearing(constellation: FrameConstellation, axes: Sequence[str], pair: Tuple[int, int],
             axis_deg: float) -> float:
    left, right = constellation.nodes[pair[0]], constellation.nodes[pair[1]]
    delta = [right.coords[name] - left.coords[name] for name in axes]
    edge_deg = math.degrees(math.atan2(delta[1], delta[0])) % 180.0
    difference = abs(edge_deg - axis_deg) % 180.0
    return min(difference, 180.0 - difference)


def _strength_block(constellation: FrameConstellation) -> Tuple[float, ...]:
    values = []
    for index, node in enumerate(constellation.nodes):
        if node.strength_sigma is None or node.strength_sigma <= 0.0:
            raise InvalidParameterError(
                "signature_for.constellation", index,
                "every member's strength expressed in its own band's RMS. Member %d has no "
                "band RMS recorded, and a raw magnitude ratio across two bands is a ratio of "
                "filter gains rather than a comparison of two structures" % index)
        values.append(float(node.strength_sigma))
    reference = math.exp(sum(math.log(value) for value in values) / len(values))
    return tuple(value / reference for value in values)


def _spatial_geometry(constellation: FrameConstellation,
                      context: RelationContext) -> Dict[Tuple[int, int], float]:
    """O(k^2) separations via the existing unit/domain/period-aware location contract.

    These are grid separations, not kilometres. No band estimate enters the denominator.
    A pair retains its separation and therefore does not suffer the scale-free pair collapse.
    """
    if any(axis.periodic for feature in constellation.features for axis in feature.location.axes):
        raise InvalidParameterError("spatial_geometry.axes", "periodic",
                                    "a nonperiodic planar record; principal-axis bearings "
                                    "have no declared wrapped-coordinate convention")
    values = {}
    for left, right in _edge_pairs(constellation.cardinality):
        distance = constellation.features[left].separation_to(
            constellation.features[right], periods=context.periods)
        if not distance.units or not math.isfinite(distance.value) or distance.value <= 0:
            raise InvalidParameterError(
                "spatial_geometry.separation", distance.value,
                "a positive finite separation with declared units; coincident detections "
                "do not define the positive relative-distance identity used here")
        values[left, right] = float(distance.value)
    return values


SIGNATURE_MODES.add("scale_specific", SignatureMode(
    lambda constellation, context: _geometry_values(constellation, "scale_specific", context),
    SCALE_MODES["scale_specific"], MODE_INVARIANCE["scale_specific"]))
SIGNATURE_MODES.add("scale_invariant", SignatureMode(
    lambda constellation, context: _geometry_values(constellation, "scale_invariant", context),
    SCALE_MODES["scale_invariant"], MODE_INVARIANCE["scale_invariant"], scale_ratios=True))
SIGNATURE_MODES.add("spatial_geometry", SignatureMode(
    _spatial_geometry, "record_separation", ("translation", "rotation", "reflection"),
    strengths=False, scales=False, requires_scope=True),
    description="Spatial configuration identity within a declared record/grid; bands and "
                "strengths are carried only, and absolute separation remains discriminating.",
    capabilities={"cross_domain": False, "band_independent": True, "requires_scope": True})


def signature_for(constellation: FrameConstellation, *, scale_invariant: bool = False,
                  axis_floor: float = AXIS_ISOTROPY_FLOOR,
                  context: Optional[RelationContext] = None,
                  mode: Optional[str] = None,
                  comparison_scope: Optional[str] = None) -> ConstellationSignature:
    """The invariant signature of one frame constellation, in one of the two modes.

    Refuses rather than approximates: a pair asked for the scale-invariant mode is refused by
    name, because its one separation over its own geometric mean is 1 for every pair there has
    ever been, and a signature that matches everything reports a discovery on every pair it is
    shown.
    """
    if constellation is None:
        raise InvalidParameterError(
            "signature_for.constellation", None,
            "a FrameConstellation. There is no signature of nothing")
    if not constellation.features:
        raise InvalidParameterError(
            "signature_for.constellation", 0,
            "a constellation carrying its features. The geometry is measured from the "
            "coefficients rather than from the carried summary, which has already been "
            "rounded for reading")
    mode = mode or ("scale_invariant" if scale_invariant else "scale_specific")
    specification = SIGNATURE_MODES.get(mode)
    if scale_invariant and mode != "scale_invariant":
        raise InvalidParameterError("signature_for.mode", mode,
                                    "a mode consistent with scale_invariant=True")
    scale_invariant = specification.scale_ratios
    if specification.requires_scope and (not isinstance(comparison_scope, str) or not comparison_scope.strip()):
        raise InvalidParameterError("signature_for.comparison_scope", comparison_scope,
                                    "an explicit record/grid scope before spatial signing")
    size = constellation.cardinality
    if scale_invariant and size < 3:
        raise InvalidParameterError(
            "signature_for.constellation", size,
            "at least three members for the scale-invariant mode. A pair has one separation "
            "and its ratio to itself is 1, so the scale-free shape of a pair is the same "
            "number for every pair in every domain. Run this pair in the scale-specific mode, "
            "where the separation is divided by the members' own scales, and read the result "
            "as the not-rescaling-invariant quantity it is")

    context = context or RelationContext()
    axes = _coord_axes(constellation)
    geometry = specification.geometry(constellation, context)
    strengths = _strength_block(constellation) if specification.strengths else ()
    scales = tuple(float(node.scale) for node in constellation.nodes)
    scale_units = constellation.nodes[0].scale_units
    admission = admit_axis(constellation, floor=axis_floor)

    bearings: Optional[Dict[Tuple[int, int], float]] = None
    refusals: Dict[str, str] = {}
    if admission.admitted and admission.angle_deg is not None:
        bearings = {pair: _bearing(constellation, axes, pair, admission.angle_deg)
                    for pair in _edge_pairs(size)}
    else:
        refusals["bearings"] = admission.refusal or "no principal axis"
    if mode == "scale_specific":
        refusals["rescaling"] = (
            "this mode divides by an estimated spatial scale, and TG3.4 measured that the "
            "estimate drifts with the scale being estimated, so the signature is not claimed "
            "invariant to a rescaling")
        refusals["cross_domain"] = (
            "the scale block is in %s. A ratio of scales in cells to scales in metres is not "
            "a number, so this signature stops at the domain boundary"
            % (scale_units if scale_units else "this record's own units"))
    if specification.requires_scope:
        refusals["rescaling"] = "absolute spatial separation changes under rescaling"
        refusals["cross_domain"] = "spatial identity is restricted to the declared record/grid scope"

    best: Optional[Tuple[Tuple[float, ...], Tuple[int, ...]]] = None
    for order in itertools.permutations(range(size)):
        edges = _edge_pairs(size)
        block: List[float] = []
        for a, b in edges:
            pair = (order[a], order[b])
            block.append(geometry.get(pair, geometry.get((pair[1], pair[0]), math.nan)))
        if bearings is not None:
            for a, b in edges:
                pair = (order[a], order[b])
                block.append(bearings.get(pair, bearings.get((pair[1], pair[0]), math.nan)))
        if specification.strengths:
            block.extend(strengths[index] for index in order)
        if specification.scales:
            block.extend(scales[index] for index in order)
        candidate = (tuple(block), tuple(order))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    order = best[1]
    edges = _edge_pairs(size)
    ordered_geometry = tuple(
        geometry.get((order[a], order[b]), geometry.get((order[b], order[a]), math.nan))
        for a, b in edges)
    ordered_bearings = None if bearings is None else tuple(
        bearings.get((order[a], order[b]), bearings.get((order[b], order[a]), math.nan))
        for a, b in edges)

    return ConstellationSignature(
        key=constellation.key(), time=constellation.time, cardinality=size, mode=mode,
        order=order, geometry=ordered_geometry, geometry_relation=specification.relation,
        bearings=ordered_bearings,
        strengths=tuple(strengths[index] for index in order) if strengths else (),
        scales=tuple(scales[index] for index in order), scale_units=scale_units,
        axis=admission,
        track_ids=tuple(constellation.track_ids[index] for index in order),
        bands=tuple(constellation.bands[index] for index in order),
        refusals=refusals, time_units=constellation.time_units,
        comparison_scope=comparison_scope if specification.requires_scope else None,
        geometry_units=(constellation.nodes[0].coord_units if specification.requires_scope else None),
        strengths_carried=(tuple(constellation.nodes[index].strength_sigma for index in order)
                           if not specification.strengths else ()),
        comparison_source=(tuple(constellation.features[0].semantic_key)
                           + (constellation.features[0].representation,
                              repr(tuple((a.name, a.role, a.units, a.periodic)
                                         for a in constellation.features[0].location.axes)))
                           if specification.requires_scope else ()))


def sign_constellations(constellations: ConstellationSet, *, scale_invariant: bool = False,
                        axis_floor: float = AXIS_ISOTROPY_FLOOR,
                        mode: Optional[str] = None,
                        comparison_scope: Optional[str] = None) -> SignatureSet:
    """Sign a whole extraction pass, counting by name what could not be signed.

    A constellation the mode cannot express is skipped and counted rather than raised, because
    the count is the answer to the roadmap's question -- what does turning the toggle on cost?
    -- and an exception would make that question unanswerable without a try/except at the call
    site.
    """
    if constellations is None:
        raise InvalidParameterError(
            "sign_constellations.constellations", None,
            "a ConstellationSet from T4E.1's extraction")
    mode = mode or ("scale_invariant" if scale_invariant else "scale_specific")
    specification = SIGNATURE_MODES.get(mode)
    if scale_invariant and mode != "scale_invariant":
        raise InvalidParameterError("sign_constellations.mode", mode,
                                    "a mode consistent with scale_invariant=True")
    if specification.requires_scope and (not isinstance(comparison_scope, str) or not comparison_scope.strip()):
        raise InvalidParameterError("sign_constellations.comparison_scope", comparison_scope,
                                    "an explicit record/grid scope before spatial signing")
    signed: List[ConstellationSignature] = []
    refused: Dict[str, int] = {}
    axis_refused = 0
    for item in constellations:
        try:
            signature = signature_for(item, scale_invariant=scale_invariant,
                                      axis_floor=axis_floor, mode=mode,
                                      comparison_scope=comparison_scope)
        except InvalidParameterError:
            reason = ("cardinality %d: no shape without a second separation" % item.cardinality
                      if scale_invariant and item.cardinality < 3 else
                      "cardinality %d: the relation refused on an edge" % item.cardinality)
            refused[reason] = refused.get(reason, 0) + 1
            continue
        if signature.bearings is None:
            axis_refused += 1
        signed.append(signature)
    return SignatureSet(signatures=tuple(signed), mode=mode,
                        considered=len(constellations), refused=refused,
                        axis_refused=axis_refused, floor=float(axis_floor),
                        comparison_scope=comparison_scope if specification.requires_scope else None)


def compare_scale_modes(constellations: ConstellationSet, *,
                        axis_floor: float = AXIS_ISOTROPY_FLOOR) -> Dict[str, Any]:
    """Run the same pass with the toggle on and off, and report what the difference is.

    The specification asks for exactly this comparison, and the honest form of it at T4E.2 is
    a comparison of *expressive reach* -- how much of the pass each mode can describe, and what
    each one refuses. It is not a comparison of findings: deciding that two signatures are the
    same needs a calibrated tolerance, which is T4E.3.
    """
    off = sign_constellations(constellations, scale_invariant=False, axis_floor=axis_floor)
    on = sign_constellations(constellations, scale_invariant=True, axis_floor=axis_floor)
    lost = sorted({item.key for item in off} - {item.key for item in on})
    return {
        "schema": SIGNATURE_SCHEMA,
        "considered": len(constellations),
        "scale_specific": off.describe(),
        "scale_invariant": on.describe(),
        "signable_in_both": len(on),
        "lost_by_turning_it_on": len(lost),
        "lost_by_cardinality": {
            str(size): sum(1 for key in lost
                           if any(item.key == key and item.cardinality == size
                                  for item in off))
            for size in sorted({item.cardinality for item in off})},
        "what_the_toggle_buys": (
            "invariance to a rescaling, and a signature whose every entry is dimensionless, "
            "which is the only form that could be compared with a configuration from another "
            "domain"),
        "what_the_toggle_costs": (
            "every configuration with fewer than three members, because a scale-free shape is "
            "a ratio between separations and a pair has only one"),
        "not_compared": (
            "how many distinct configurations each mode sees. That is a count of clusters, it "
            "needs a tolerance calibrated against replicates, and it is T4E.3"),
    }


def describe_signatures(signatures: SignatureSet) -> Dict[str, Any]:
    """The receipt for a signing pass, with the band census that makes it readable."""
    body = dict(signatures.describe())
    census: Dict[Tuple[str, ...], int] = {}
    for item in signatures:
        key = tuple(sorted(item.bands))
        census[key] = census.get(key, 0) + 1
    body["by_bands"] = {"+".join(key): value
                        for key, value in sorted(census.items(),
                                                 key=lambda kv: (-kv[1], kv[0]))}
    anisotropies = [item.axis.anisotropy for item in signatures
                    if not math.isinf(item.axis.anisotropy)]
    body["anisotropy"] = ({"min": min(anisotropies), "max": max(anisotropies)}
                          if anisotropies else None)
    return body


__all__ = [
    "AXIS_FLOOR_REPLICATES", "AXIS_ISOTROPY_FLOOR", "BEARING_BASIS", "CLAIM_BOUNDARY",
    "MODE_INVARIANCE", "SCALE_MODES", "SIGNATURE_SCHEMA", "STRENGTH_BASIS",
    "AxisAdmission", "ConstellationSignature", "SignatureSet", "admit_axis",
    "calibrate_axis_admission", "compare_scale_modes", "describe_signatures",
    "principal_axis", "sign_constellations", "signature_for",
]
