"""TG3.3: constellations as attributed graphs, and the relations that refuse to travel.

TG2.1 settled what a feature is, TG2.2 who decides one is there and TG2.3 which feature in
one frame is the same object as one in the next. This module settles what a *configuration*
of features is: `k` features plus the typed relations between them, expressed so that the
description survives leaving its domain.

**The whole problem is in one contrast already present in TG2.1.** `scale_ratio_to` is
permitted across a domain boundary and `separation_to` is refused across one, because a
ratio of two lengths is a number about the world and a distance in cells is a number about
an array. A relation is therefore not "a quantity computed from two features" - it is a
quantity computed from two features *and then divided by something the two features carry
themselves*, so that the unit cancels. Forty cells at a scale of six cells and twelve
hundred metres at a scale of a hundred and eighty metres are the same relation; forty and
twelve hundred are not. Everything registered here either performs that division or declares
that it cannot.

**Relations are computed within a domain and compared across one.** A constellation is drawn
from a single `FeatureSet`, which TG2.1 already constrains to one domain, one dataset, one
variable and one representation - so the coordinates the relations read are coordinates in a
space that exists. What crosses the boundary afterwards is the `AttributedGraph`, whose
comparable view is built from the dimensionless results rather than filtered out of the full
record. Adding a carried field back into the comparison is a visible edit to
`AttributedGraph.attributes` rather than an invisible consequence of a key not being removed.

**Eight relations, and the split between them is a finding rather than a taxonomy.** The
roadmap names distance, relative scale, temporal lag, direction, convergence, containment,
succession and co-occurrence. Sorting them by what each needs before it can be made
dimensionless does not produce eight of a kind:

*   `succession` needs nothing but a shared clock. An ordering is already dimensionless -
    "this one came first" is the same statement in frames and in hours.
*   `co_occurrence` needs a temporal scale, because simultaneity is a tolerance, and a
    tolerance in seconds is not a statement any other domain can read.
*   `distance` and `containment` need a spatial scale and an extent respectively, for the
    same reason.
*   `direction` and `convergence` need an orientation, which the only extractor registered
    in TG2.2 declares it does not report.

So two of the eight cannot be measured at all on today's features, and the design ruling of
this slice is that they register anyway and **refuse by name**. That is TG2.3's rule about
gates carried into relations: a relation that treats an absent quantity as "no evidence
against" would appear in a receipt, constrain nothing, and be indistinguishable from one
that was doing work. `measurable_relations` reports which of the eight a given set can
actually support, and `relation_axis` turns that answer into a TG3.1 `SearchAxis` - because
a family priced over eight relations when some of them will refuse is a family whose declared
size was never the number of tests that ran.

**Matching is exhaustive or refused.** Two attributed graphs describe the same configuration
if some correspondence of their nodes makes every node attribute and every edge relation
agree, and finding that correspondence is `k!` comparisons. Above `MAX_MATCH_NODES` this
module refuses rather than falling back on a greedy or heuristic assignment: an approximate
match that returns `True` is a claim, and this tree does not make claims it has not
measured. TG3.4's invariant matcher is what will make the larger case tractable; until it
exists the honest answer to a nine-node constellation is a refusal.

**What this slice deliberately does not do.** It does not match a configuration under
rotation, rescaling and translation - that is TG3.4, whose whole acceptance is the
`4E.invariance` gate moving off `NOT_YET_RUNNABLE`, and claiming it here would leave that
slice with nothing to prove. It does not mine for repeated configurations either; TG3.5 does
that, and only through TG3.1's declared family and TG3.2's split.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field
from typing import (
    Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple,
)

from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.family import SearchAxis
from src.core.feature import (
    FeatureSet, SemanticComparisonError, SpectralFeature, convention_for,
)
from src.core.registry import Registry

#: Above this many nodes, `AttributedGraph.matches` refuses rather than approximating.
#: 8! is 40,320 correspondences, which is a fraction of a second; 9! is nine times that and
#: the number that follows is not a wall this module should be discovering at runtime.
MAX_MATCH_NODES = 8

#: Default tolerance when comparing two dimensionless relation values. Relations are ratios
#: of measured quantities, so this is a *relative* tolerance, not an absolute one.
DEFAULT_TOLERANCE = 1e-9


class RelationUnmeasurableError(InvalidParameterError):
    """A relation was asked for a quantity the features do not carry.

    Distinct from `SemanticComparisonError`, which means the quantities are present and
    refuse to be compared. This one means the question cannot be asked at all - and it is an
    error rather than a `None` because a relation silently absent from a graph would make
    two graphs match on the strength of what neither of them measured.
    """

    def __init__(self, relation: str, requirement: str, detail: str) -> None:
        super().__init__(
            "relation %r" % relation, requirement,
            "features carrying %s. %s" % (requirement, detail),
            relation=relation, requirement=requirement)
        self.relation = relation
        self.requirement = requirement


class GraphTooLargeError(InvalidParameterError):
    """A node correspondence search was asked for more permutations than it will run."""

    def __init__(self, n_nodes: int) -> None:
        super().__init__(
            "AttributedGraph.matches", n_nodes,
            "at most %d nodes. Matching is exhaustive over node correspondences, and this "
            "module refuses to approximate one: a heuristic assignment that reports a match "
            "is a claim about a configuration, and an approximate claim is the failure mode "
            "this tree exists to prevent. TG3.4's invariant matcher is what makes the "
            "larger case tractable" % MAX_MATCH_NODES,
            n_nodes=n_nodes, limit=MAX_MATCH_NODES)


# --------------------------------------------------------------------------- the results


@dataclass(frozen=True)
class RelationValue:
    """One measured relation between one ordered pair of features.

    `value` is dimensionless by construction or it is `None`; there is no third case. A
    relation that could only report a number in the originating units reports nothing, and
    `basis` says in words what the number was divided by, because "6.67" is unreadable
    without "cells of separation over cells of scale".
    """

    relation: str
    value: Optional[float]
    holds: Optional[bool]
    basis: str

    def __post_init__(self) -> None:
        if self.value is None and self.holds is None:
            raise InvalidParameterError(
                "RelationValue", self.relation,
                "a value, a predicate, or both. A relation that measured neither is an "
                "edge that would make two graphs agree about nothing")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise InvalidParameterError(
                "RelationValue.value", self.value,
                "a finite dimensionless value. A NaN relation compares equal to nothing "
                "and unequal to nothing, so a match over it would report False for a "
                "reason that is not the geometry")
        if not str(self.basis).strip():
            raise InvalidParameterError(
                "RelationValue.basis", self.basis,
                "a statement of what the quantity was divided by. A dimensionless number "
                "whose derivation is not recorded cannot be checked by a reader")
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))

    def describe(self) -> Dict[str, Any]:
        return {"relation": self.relation, "value": self.value, "holds": self.holds,
                "basis": self.basis}


@dataclass(frozen=True)
class RelationContext:
    """What pairwise arithmetic needs that the features themselves do not carry.

    Only the periodic-axis lengths, so far. TG2.1 refuses a separation across a periodic
    axis whose length was not declared, and that refusal has to be reachable from here
    rather than worked around with a default of "not periodic".
    """

    periods: Mapping[str, float] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "periods",
                           {str(k): float(v) for k, v in dict(self.periods).items()})

    def describe(self) -> Dict[str, Any]:
        return {"periods": dict(self.periods)}


# ------------------------------------------------------------------------- the registry


Measure = Callable[[SpectralFeature, SpectralFeature, RelationContext], RelationValue]


@dataclass(frozen=True)
class Relation:
    """One typed relation, and the declaration of what it cannot do without.

    `requires` names fields on `SpectralFeature`, checked before `measure` runs, so a
    relation body never has to decide what an absent scale means. `symmetric` records
    whether the relation is a property of the unordered pair: a distance is, a scale ratio
    is not, and a graph that stored one direction of an asymmetric relation and inferred the
    other would be inventing an inverse it never measured.
    """

    name: str
    kind: str
    requires: Tuple[str, ...]
    symmetric: bool
    measure: Measure
    description: str = ""

    KINDS = ("continuous", "predicate")

    def __post_init__(self) -> None:
        if self.kind not in Relation.KINDS:
            raise UnknownNameError("relation kind", self.kind, list(Relation.KINDS))
        object.__setattr__(self, "requires", tuple(self.requires))
        for name in self.requires:
            if not hasattr(SpectralFeature, "__dataclass_fields__") or \
                    name not in SpectralFeature.__dataclass_fields__:
                raise UnknownNameError(
                    "feature field", name,
                    sorted(SpectralFeature.__dataclass_fields__))

    def missing_on(self, feature: SpectralFeature) -> Tuple[str, ...]:
        """The declared requirements this feature does not carry."""
        return tuple(name for name in self.requires if getattr(feature, name) is None)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "requires": list(self.requires),
                "symmetric": self.symmetric, "description": self.description}


#: The typed relations (standard E1). A ninth registers from outside `src/`.
RELATIONS: Registry[Relation] = Registry("constellation relation")


def relation_for(name: str) -> Relation:
    return RELATIONS.get(name)


def _require(relation: str, left: SpectralFeature, right: SpectralFeature) -> None:
    """Refuse before measuring, naming the field and the feature that lacks it."""
    rule = relation_for(relation)
    for feature, side in ((left, "left"), (right, "right")):
        missing = rule.missing_on(feature)
        if missing:
            raise RelationUnmeasurableError(
                relation, missing[0],
                "The %s feature (%s/%s at t=%g) does not carry %s, and treating an absent "
                "quantity as 'no evidence against' would put an edge in the graph that "
                "constrains nothing."
                % (side, feature.domain, feature.variable, feature.time, missing[0]))


def _geometric_mean(left: float, right: float, *, what: str) -> float:
    """The reference scale a separation is divided by.

    Geometric rather than arithmetic because the quantity being formed is a ratio, and the
    mean of two scales that differ by a factor of ten should sit between them
    multiplicatively - an arithmetic mean is dominated by the larger and would make the same
    configuration read differently depending on which feature happened to be bigger.
    """
    if left <= 0.0 or right <= 0.0:
        raise InvalidParameterError(
            what, (left, right),
            "positive %s on both features. A ratio against a zero reference is not a "
            "large relation, it is an undefined one" % what)
    return math.sqrt(left * right)


def _shared_scale_units(relation: str, left: Any, right: Any, *, what: str) -> str:
    if left.units != right.units:
        raise SemanticComparisonError(
            what, left.describe_short(), right.describe_short(),
            "Relation %r divides one by the other, and %r over %r is not dimensionless."
            % (relation, left.units, right.units))
    return left.units


def _separation(relation: str, left: SpectralFeature, right: SpectralFeature,
                context: RelationContext) -> Any:
    """The raw separation, which is exactly the number that must not leave the domain."""
    return left.separation_to(right, periods=context.periods)


def _bearing_degrees(left: SpectralFeature, right: SpectralFeature,
                     relation: str) -> float:
    """The compass angle of the displacement left -> right, in the grid's own frame.

    Not a relation on its own: a bearing is measured from the grid's north, which is a
    property of how the array was stored rather than of the world. It becomes transferable
    only when subtracted from an angle the feature itself carries, which is what `direction`
    and `convergence` do with it.
    """
    axes = [axis for axis in left.location.axes if axis.role == "space"]
    if len(axes) != 2:
        raise RelationUnmeasurableError(
            relation, "two spatial axes",
            "The location declares %d spatial axis/axes, and an angle in a plane needs "
            "exactly two. A bearing computed from one axis is a sign, and one computed "
            "from three is a projection nobody chose." % len(axes))
    ordered = sorted(axes, key=lambda a: (a.ordinal if a.ordinal is not None else 0))
    row, col = ordered[0].name, ordered[1].name
    if any(axis.periodic for axis in axes):
        raise RelationUnmeasurableError(
            relation, "non-periodic spatial axes",
            "Axis wrapping makes the displacement ambiguous in direction even where the "
            "wrapped distance is well defined: the short way round and the long way round "
            "point opposite ways, and this module will not pick one for you.")
    d_row = right.location.coords[row] - left.location.coords[row]
    d_col = right.location.coords[col] - left.location.coords[col]
    if d_row == 0.0 and d_col == 0.0:
        raise RelationUnmeasurableError(
            relation, "a non-zero displacement",
            "The two features are at the same location, and the direction from a point to "
            "itself is not zero degrees, it is undefined.")
    return math.degrees(math.atan2(d_col, d_row))


# ------------------------------------------------------------------- the eight relations


def _distance(left: SpectralFeature, right: SpectralFeature,
              context: RelationContext) -> RelationValue:
    _require("distance", left, right)
    separation = _separation("distance", left, right, context)
    units = _shared_scale_units("distance", left.spatial_scale, right.spatial_scale,
                                what="spatial_scale")
    if separation.units != units:
        raise SemanticComparisonError(
            "distance", separation.describe_short(), left.spatial_scale.describe_short(),
            "The separation is in %r and the scales are in %r, so the quotient is not a "
            "number of scale lengths. A location in cells beside a scale in metres is a "
            "real state of this record and it is refused rather than coerced."
            % (separation.units, units))
    reference = _geometric_mean(left.spatial_scale.value, right.spatial_scale.value,
                                what="spatial_scale")
    return RelationValue("distance", separation.value / reference, None,
                         "separation over the geometric mean of the two spatial scales, "
                         "both in %s" % (units if units is not None else "no units"))


def _relative_scale(left: SpectralFeature, right: SpectralFeature,
                    context: RelationContext) -> RelationValue:
    _require("relative_scale", left, right)
    return RelationValue("relative_scale", left.scale_ratio_to(right), None,
                         "the left spatial scale over the right, units cancelling")


def _temporal_lag(left: SpectralFeature, right: SpectralFeature,
                  context: RelationContext) -> RelationValue:
    _require("temporal_lag", left, right)
    units = _shared_scale_units("temporal_lag", left.temporal_scale, right.temporal_scale,
                                what="temporal_scale")
    if units != left.time_units:
        raise SemanticComparisonError(
            "temporal_lag", str(left.time_units), str(units),
            "The clock is in %r and the temporal scales are in %r, so the quotient is not "
            "a number of characteristic times." % (left.time_units, units))
    reference = _geometric_mean(left.temporal_scale.value, right.temporal_scale.value,
                                what="temporal_scale")
    return RelationValue("temporal_lag", right.elapsed_to(left) / reference, None,
                         "elapsed time from left to right over the geometric mean of the "
                         "two temporal scales, both in %s"
                         % (units if units is not None else "no units"))


def _direction(left: SpectralFeature, right: SpectralFeature,
               context: RelationContext) -> RelationValue:
    _require("direction", left, right)
    bearing = _bearing_degrees(left, right, "direction")
    convention = left.orientation.convention_object()
    if left.orientation.convention != right.orientation.convention:
        raise SemanticComparisonError(
            "direction", left.orientation.convention, right.orientation.convention,
            "The two features declare different orientation conventions, so the angles "
            "they carry are not the same quantity.")
    return RelationValue(
        "direction", convention.difference(bearing, left.orientation.degrees), None,
        "the bearing of the displacement measured from the left feature's own "
        "orientation, wrapped at %g degrees. The grid's north cancels."
        % convention.period_degrees)


def _convergence(left: SpectralFeature, right: SpectralFeature,
                 context: RelationContext) -> RelationValue:
    _require("convergence", left, right)
    if left.orientation.convention != right.orientation.convention:
        raise SemanticComparisonError(
            "convergence", left.orientation.convention, right.orientation.convention,
            "The two features declare different orientation conventions.")
    convention = left.orientation.convention_object()
    if not convention.directed:
        raise RelationUnmeasurableError(
            "convergence", "a directed orientation",
            "Convention %r wraps at %g degrees and declares directed=False, which is an "
            "undirected axis. An axis does not point, so it cannot point towards anything: "
            "a ridge at 170 degrees and one at 350 are the same orientation, and asking "
            "whether they converge is asking a question the quantity cannot answer."
            % (left.orientation.convention, convention.period_degrees))
    towards = _bearing_degrees(left, right, "convergence")
    back = _bearing_degrees(right, left, "convergence")
    inward = (math.cos(math.radians(convention.difference(
                  towards, left.orientation.degrees)))
              + math.cos(math.radians(convention.difference(
                  back, right.orientation.degrees))))
    return RelationValue(
        "convergence", inward, inward > 0.0,
        "the sum of the cosines of each feature's orientation against the bearing to the "
        "other, in [-2, 2]; positive when both point inward")


def _containment(left: SpectralFeature, right: SpectralFeature,
                 context: RelationContext) -> RelationValue:
    _require("containment", left, right)
    separation = _separation("containment", left, right, context)
    units = _shared_scale_units("containment", left.extent, right.extent, what="extent")
    if separation.units != units:
        raise SemanticComparisonError(
            "containment", separation.describe_short(), left.extent.describe_short(),
            "The separation is in %r and the extents are in %r, so one cannot be "
            "subtracted from the other." % (separation.units, units))
    if left.extent.value <= 0.0:
        raise InvalidParameterError(
            "containment", left.extent.value, "a positive extent on the containing feature")
    margin = (left.extent.value - (separation.value + right.extent.value)) \
        / left.extent.value
    return RelationValue(
        "containment", margin, margin >= 0.0,
        "the left extent less the separation and the right extent, over the left extent; "
        "non-negative when the right feature lies wholly inside the left")


def _succession(left: SpectralFeature, right: SpectralFeature,
                context: RelationContext) -> RelationValue:
    _require("succession", left, right)
    elapsed = right.elapsed_to(left)
    return RelationValue(
        "succession", None, elapsed > 0.0,
        "the sign of the elapsed time from left to right on the shared clock. An ordering "
        "needs no characteristic scale to survive a change of units: what came first came "
        "first in frames and in hours alike")


def _co_occurrence(left: SpectralFeature, right: SpectralFeature,
                   context: RelationContext) -> RelationValue:
    _require("co_occurrence", left, right)
    units = _shared_scale_units("co_occurrence", left.temporal_scale,
                                right.temporal_scale, what="temporal_scale")
    if units != left.time_units:
        raise SemanticComparisonError(
            "co_occurrence", str(left.time_units), str(units),
            "The clock is in %r and the temporal scales are in %r, so the tolerance and "
            "the separation are not in the same units." % (left.time_units, units))
    reference = _geometric_mean(left.temporal_scale.value, right.temporal_scale.value,
                                what="temporal_scale")
    ratio = abs(right.elapsed_to(left)) / reference
    return RelationValue(
        "co_occurrence", ratio, ratio <= 1.0,
        "the absolute elapsed time over the geometric mean of the two temporal scales; "
        "co-occurring within one characteristic time. Unlike an ordering, simultaneity is "
        "a tolerance, and a tolerance in seconds is not a statement another domain can read")


RELATIONS.add(
    "distance", Relation(
        "distance", "continuous", ("spatial_scale",), True, _distance,
        "Separation in units of the features' own scale."),
    description="Separation in units of the features' own spatial scale.",
    capabilities={"requires": ["spatial_scale"], "symmetric": True,
                  "kind": "continuous", "dimensionless_by": "own_spatial_scale"})

RELATIONS.add(
    "relative_scale", Relation(
        "relative_scale", "continuous", ("spatial_scale",), False, _relative_scale,
        "The ratio of the two spatial scales."),
    description="The ratio of the two spatial scales, which is already dimensionless.",
    capabilities={"requires": ["spatial_scale"], "symmetric": False,
                  "kind": "continuous", "dimensionless_by": "ratio"})

RELATIONS.add(
    "temporal_lag", Relation(
        "temporal_lag", "continuous", ("temporal_scale",), False, _temporal_lag,
        "Signed elapsed time in units of the features' own temporal scale."),
    description="Signed elapsed time in units of the features' own temporal scale.",
    capabilities={"requires": ["temporal_scale"], "symmetric": False,
                  "kind": "continuous", "dimensionless_by": "own_temporal_scale"})

RELATIONS.add(
    "direction", Relation(
        "direction", "continuous", ("orientation",), False, _direction,
        "The bearing to the other feature, measured from this one's own orientation."),
    description=("The bearing to the other feature measured from this one's own "
                 "orientation, so the grid's north cancels."),
    capabilities={"requires": ["orientation"], "symmetric": False,
                  "kind": "continuous", "dimensionless_by": "own_orientation"})

RELATIONS.add(
    "convergence", Relation(
        "convergence", "predicate", ("orientation",), True, _convergence,
        "Whether both features point towards each other; requires a directed angle."),
    description=("Whether both features point towards one another. Refused on an "
                 "undirected axis convention, which does not point."),
    capabilities={"requires": ["orientation"], "symmetric": True,
                  "kind": "predicate", "needs_directed_orientation": True,
                  "dimensionless_by": "own_orientation"})

RELATIONS.add(
    "containment", Relation(
        "containment", "predicate", ("extent",), False, _containment,
        "Whether the right feature lies wholly inside the left."),
    description="Whether the right feature lies wholly inside the left, in units of extent.",
    capabilities={"requires": ["extent"], "symmetric": False,
                  "kind": "predicate", "dimensionless_by": "own_extent"})

RELATIONS.add(
    "succession", Relation(
        "succession", "predicate", (), False, _succession,
        "Whether the right feature follows the left on the shared clock."),
    description=("Whether the right feature follows the left. The only relation needing no "
                 "characteristic scale: an ordering is dimensionless already."),
    capabilities={"requires": [], "symmetric": False, "kind": "predicate",
                  "dimensionless_by": "ordering_is_already_dimensionless"})

RELATIONS.add(
    "co_occurrence", Relation(
        "co_occurrence", "predicate", ("temporal_scale",), True, _co_occurrence,
        "Whether the two features fall within one characteristic time of each other."),
    description=("Whether the two fall within one characteristic time of each other. "
                 "Simultaneity is a tolerance, so unlike an ordering it needs a scale."),
    capabilities={"requires": ["temporal_scale"], "symmetric": True,
                  "kind": "predicate", "dimensionless_by": "own_temporal_scale"})


# --------------------------------------------------------- what a set can actually support


def measurable_relations(features: Sequence[SpectralFeature],
                         names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """The registered relations every feature in `features` carries the fields for.

    Membership is a property of the fields, not of an attempt: a relation whose
    requirements are present can still refuse on a particular pair - two features at the
    same location have no direction between them - and that refusal is a fact about the
    pair rather than a reason to drop the relation from the declaration.
    """
    if not features:
        raise InvalidParameterError(
            "measurable_relations", [],
            "at least one feature. 'Every feature carries it' is vacuously true of no "
            "features, and a family priced on that answer would declare eight relations "
            "and run none")
    chosen = RELATIONS.names() if names is None else list(names)
    supported = []
    for name in chosen:
        rule = relation_for(name)
        if all(not rule.missing_on(feature) for feature in features):
            supported.append(name)
    return tuple(supported)


def relation_axis(features: Sequence[SpectralFeature],
                  names: Optional[Sequence[str]] = None) -> SearchAxis:
    """A TG3.1 search axis over exactly the relations this set can support.

    The point of routing the declaration through here rather than through
    `RELATIONS.names()` is that a family priced over eight relations when two of them will
    refuse has declared a size that is not the number of tests that ran - which is R18's
    failure in the one direction R18 does not catch, since the declared family was too
    *large* and the correction therefore too conservative. Conservative is not honest: the
    receipt would name tests that never happened.
    """
    supported = measurable_relations(features, names)
    if not supported:
        raise RelationUnmeasurableError(
            "relation_axis", "any measurable relation",
            "None of the %d relations asked for can be measured on these features, so "
            "there is no axis to declare. A family with an empty relation axis is empty, "
            "and an empty family reports nothing for a reason that is not the data."
            % (len(RELATIONS) if names is None else len(list(names))))
    return SearchAxis("relation", supported)


# ------------------------------------------------------------------- the attributed graph


def _node_attributes(feature: SpectralFeature) -> Dict[str, Any]:
    """The dimensionless view of one node, built rather than filtered.

    Deliberately *not* `structural_signature()`. That view carries `representation`, which
    is a name rather than a number and belongs with the carried record, and it carries an
    absolute `orientation_degrees`, which is measured from the grid's north - a property of
    how the array was stored. What survives here is the presence of each quantity, the
    convention an angle wraps under, the significance and the extent expressed in scales.
    Absolute angles re-enter as *differences* on the edges, which is where they belong.
    """
    signature = feature.structural_signature()
    return {
        "has_spatial_scale": signature["has_spatial_scale"],
        "has_temporal_scale": signature["has_temporal_scale"],
        "has_orientation": feature.orientation is not None,
        "orientation_convention": signature["orientation_convention"],
        "significance": signature["significance"],
        "significance_basis": signature["significance_basis"],
        "extent_in_scales": signature["extent_in_scales"],
    }


def _carried(feature: SpectralFeature) -> Dict[str, Any]:
    """What R19 protects: travels with the graph, never enters a comparison."""
    return {"domain": feature.domain, "dataset": feature.dataset,
            "variable": feature.variable, "representation": feature.representation,
            "units": feature.units, "time_units": feature.time_units}


@dataclass(frozen=True)
class MatchReport:
    """The result of comparing two graphs, including what was *not* compared.

    A bare `True` would be the weakest possible statement about a cross-domain match. The
    report names the correspondence that succeeded, the relations that were compared, and
    the carried records of both sides - so a reader can see that two graphs matched while
    describing a temperature field and a pressure field, which is the entire point of R19,
    and can see whether the two came from the same representation, which is R8's.
    """

    matched: bool
    correspondence: Optional[Tuple[int, ...]]
    relations_compared: Tuple[str, ...]
    tolerance: float
    left_carried: Tuple[Mapping[str, Any], ...]
    right_carried: Tuple[Mapping[str, Any], ...]
    reason: str = ""

    def __bool__(self) -> bool:
        return self.matched

    @property
    def representations(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Both sides' representations, stated rather than compared (rule R8)."""
        return (tuple(sorted({str(c["representation"]) for c in self.left_carried})),
                tuple(sorted({str(c["representation"]) for c in self.right_carried})))

    def describe(self) -> Dict[str, Any]:
        left, right = self.representations
        return {
            "matched": self.matched,
            "correspondence": None if self.correspondence is None
            else list(self.correspondence),
            "relations_compared": list(self.relations_compared),
            "tolerance": self.tolerance,
            "left_carried": [dict(c) for c in self.left_carried],
            "right_carried": [dict(c) for c in self.right_carried],
            "left_representations": list(left),
            "right_representations": list(right),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttributedGraph:
    """`k` features and the typed relations between them, in two separated halves.

    `attributes` is what may cross a domain boundary; `carried` is what R19 protects and
    what nothing in `matches` reads. The separation is structural rather than documented:
    `matches` has no access to a carried value that is not passed to it, and passing one
    would be a visible edit to this class.
    """

    attributes: Tuple[Mapping[str, Any], ...]
    carried: Tuple[Mapping[str, Any], ...]
    edges: Mapping[Tuple[int, int], Mapping[str, RelationValue]]
    relations: Tuple[str, ...]
    refusals: Mapping[Tuple[int, int], Mapping[str, str]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.attributes) < 2:
            raise InvalidParameterError(
                "AttributedGraph.attributes", len(self.attributes),
                "at least two nodes. A constellation of one feature has no relations, and "
                "a graph with no edges is a feature with extra ceremony")
        if len(self.attributes) != len(self.carried):
            raise InvalidParameterError(
                "AttributedGraph.carried", len(self.carried),
                "one carried record per node")
        object.__setattr__(self, "attributes",
                           tuple(dict(a) for a in self.attributes))
        object.__setattr__(self, "carried", tuple(dict(c) for c in self.carried))
        object.__setattr__(self, "edges",
                           {tuple(k): dict(v) for k, v in dict(self.edges).items()})
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "refusals",
                           {tuple(k): dict(v) for k, v in dict(self.refusals).items()})

    @property
    def n_nodes(self) -> int:
        return len(self.attributes)

    def edge(self, left: int, right: int) -> Mapping[str, RelationValue]:
        if (left, right) not in self.edges:
            raise UnknownNameError(
                "graph edge", "%d->%d" % (left, right),
                ["%d->%d" % pair for pair in sorted(self.edges)])
        return self.edges[(left, right)]

    def value(self, left: int, right: int, relation: str) -> RelationValue:
        edge = self.edge(left, right)
        if relation not in edge:
            raise UnknownNameError(
                "relation on edge %d->%d" % (left, right), relation, sorted(edge))
        return edge[relation]

    # -------------------------------------------------------------------- comparison

    def matches(self, other: "AttributedGraph", *,
                tolerance: float = DEFAULT_TOLERANCE,
                relations: Optional[Sequence[str]] = None) -> MatchReport:
        """Is there a node correspondence under which every attribute and edge agrees?

        Exhaustive over the `k!` correspondences, or refused - see `MAX_MATCH_NODES`. Only
        `attributes` and `edges` are read; `carried` is reported and never compared, which
        is what lets a temperature field and a pressure field match while remaining two
        different things (R19).
        """
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise InvalidParameterError(
                "matches.tolerance", tolerance, "a non-negative finite relative tolerance")
        compared = tuple(relations) if relations is not None else self.relations
        for name in compared:
            if name not in self.relations or name not in other.relations:
                raise UnknownNameError(
                    "relation measured on both graphs", name,
                    sorted(set(self.relations) & set(other.relations)))
        for graph, side in ((self, "this graph"), (other, "the other graph")):
            graph._require_measured_everywhere(compared, side)
        if self.n_nodes != other.n_nodes:
            return MatchReport(
                False, None, compared, tolerance, self.carried, other.carried,
                "the graphs have %d and %d nodes" % (self.n_nodes, other.n_nodes))
        if self.n_nodes > MAX_MATCH_NODES:
            raise GraphTooLargeError(self.n_nodes)
        for permutation in itertools.permutations(range(other.n_nodes)):
            if self._agrees_under(other, permutation, compared, tolerance):
                return MatchReport(True, permutation, compared, tolerance,
                                   self.carried, other.carried,
                                   "node i of this graph corresponds to node "
                                   "correspondence[i] of the other")
        return MatchReport(
            False, None, compared, tolerance, self.carried, other.carried,
            "no correspondence of %d nodes makes every attribute and every relation in %s "
            "agree to a relative tolerance of %g"
            % (self.n_nodes, ", ".join(compared) or "(none)", tolerance))

    def _require_measured_everywhere(self, compared: Sequence[str], side: str) -> None:
        """A hole in the record is not a disagreement about the configuration.

        A graph built with `strict=False` declares relations it may not carry on every
        edge, and comparing one without saying so returns a `False` whose stated reason is
        the geometry - which is a measurement nobody made. It is a refusal instead, and the
        caller narrows `relations` to what was actually measured.
        """
        for name in compared:
            for pair, edge in sorted(self.edges.items()):
                if name not in edge:
                    reason = self.refusals.get(pair, {}).get(name, "it was not measured")
                    raise RelationUnmeasurableError(
                        name, "an edge on every pair",
                        "Edge %d->%d of %s has no %r: %s. A comparison over a relation "
                        "that is missing here would report a mismatch and name the "
                        "configuration as the cause, which is a measurement nobody made - "
                        "narrow `relations` to what was measured instead."
                        % (pair[0], pair[1], side, name, reason))

    def _agrees_under(self, other: "AttributedGraph", permutation: Sequence[int],
                      compared: Sequence[str], tolerance: float) -> bool:
        for index, mapped in enumerate(permutation):
            if not _attributes_agree(self.attributes[index], other.attributes[mapped],
                                     tolerance):
                return False
        for (left, right), edge in self.edges.items():
            counterpart = other.edges.get((permutation[left], permutation[right]))
            if counterpart is None:
                return False
            for name in compared:
                if name not in edge or name not in counterpart:
                    return False
                if not _values_agree(edge[name], counterpart[name], tolerance):
                    return False
        return True

    def describe(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "relations": list(self.relations),
            "attributes": [dict(a) for a in self.attributes],
            "carried": [dict(c) for c in self.carried],
            "edges": {"%d->%d" % pair: {n: v.describe() for n, v in edge.items()}
                      for pair, edge in sorted(self.edges.items())},
            "refusals": {"%d->%d" % pair: dict(reasons)
                         for pair, reasons in sorted(self.refusals.items())},
        }


def _close(left: Optional[float], right: Optional[float], tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _attributes_agree(left: Mapping[str, Any], right: Mapping[str, Any],
                      tolerance: float) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, float) or isinstance(b, float):
            if not _close(None if a is None else float(a),
                          None if b is None else float(b), tolerance):
                return False
        elif a != b:
            return False
    return True


def _values_agree(left: RelationValue, right: RelationValue, tolerance: float) -> bool:
    if left.holds != right.holds:
        return False
    return _close(left.value, right.value, tolerance)


# ------------------------------------------------------------------------ construction


def constellation(features: Sequence[SpectralFeature], *,
                  relations: Optional[Sequence[str]] = None,
                  context: Optional[RelationContext] = None,
                  strict: bool = True) -> AttributedGraph:
    """Build the attributed graph of `k` features under the named relations.

    `strict=True` - the default - propagates a relation's refusal, because a graph missing
    an edge it was asked for is a graph two comparisons could agree about for a reason
    neither of them measured. `strict=False` records the refusal on the graph instead,
    which is what a mining pass over heterogeneous features needs; the refusals are part of
    `describe()` so they cannot be lost between the pass and the receipt.
    """
    features = tuple(features)
    if len(features) < 2:
        raise InvalidParameterError(
            "constellation", len(features),
            "at least two features. A configuration is a statement about relations, and "
            "one feature has none")
    if len(set(id(f) for f in features)) != len(features):
        raise InvalidParameterError(
            "constellation", len(features),
            "distinct features. The same feature twice has a distance of zero from itself "
            "and would put a spurious edge in every graph it appeared in")
    first = features[0]
    for feature in features[1:]:
        if feature.semantic_key != first.semantic_key:
            raise SemanticComparisonError(
                "constellation", "/".join(first.semantic_key),
                "/".join(feature.semantic_key),
                "A constellation is a configuration within one dataset. Its relations read "
                "coordinates, and two datasets do not share a coordinate system merely by "
                "having coordinates - what crosses a domain boundary is the graph, not the "
                "features it was built from.")
    chosen = tuple(relations) if relations is not None else measurable_relations(features)
    for name in chosen:
        relation_for(name)                      # refuse an unregistered relation by name
    context = context or RelationContext()

    edges: Dict[Tuple[int, int], Dict[str, RelationValue]] = {}
    refusals: Dict[Tuple[int, int], Dict[str, str]] = {}
    for left, right in itertools.permutations(range(len(features)), 2):
        edge: Dict[str, RelationValue] = {}
        for name in chosen:
            rule = relation_for(name)
            if rule.symmetric and left > right:
                mirror = edges.get((right, left), {})
                if name in mirror:
                    edge[name] = mirror[name]
                    continue
            try:
                edge[name] = rule.measure(features[left], features[right], context)
            except (RelationUnmeasurableError, SemanticComparisonError,
                    InvalidParameterError) as exc:
                if strict:
                    raise
                refusals.setdefault((left, right), {})[name] = str(exc)
        edges[(left, right)] = edge
    return AttributedGraph(
        attributes=tuple(_node_attributes(f) for f in features),
        carried=tuple(_carried(f) for f in features),
        edges=edges, relations=chosen, refusals=refusals)


def constellations_from_set(features: FeatureSet, size: int, **kwargs: Any) \
        -> Iterator[AttributedGraph]:
    """Every unordered `size`-subset of a set, as attributed graphs.

    The enumeration order is `itertools.combinations` over the set's own time ordering, so
    it is the same shape TG3.1's `unordered_pairs` counts and the shape a `k`-feature
    declaration will have to price. This yields rather than returns: the count is
    `C(n, k)`, which is exactly the number TG3.1 exists to refuse when it is too large, and
    materialising it here would spend the memory before the refusal could be consulted.
    """
    if size < 2:
        raise InvalidParameterError(
            "constellations_from_set.size", size,
            "a configuration of at least two features")
    if size > len(features):
        raise InvalidParameterError(
            "constellations_from_set.size", size,
            "a size no larger than the %d features in the set" % len(features))
    for chosen in itertools.combinations(list(features), size):
        yield constellation(chosen, **kwargs)


__all__ = [
    "MAX_MATCH_NODES", "DEFAULT_TOLERANCE", "RelationUnmeasurableError",
    "GraphTooLargeError", "RelationValue", "RelationContext", "Relation", "RELATIONS",
    "relation_for", "measurable_relations", "relation_axis", "MatchReport",
    "AttributedGraph", "constellation", "constellations_from_set",
]
