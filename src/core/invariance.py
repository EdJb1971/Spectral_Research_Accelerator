"""TG3.4: invariant matching, and the tolerance that is measured rather than chosen.

The `4E.invariance` gate asks a question TG3.3 could not answer on its own: is the same
configuration, presented rotated, translated and rescaled, recognised as the same
configuration? Three findings shaped this module, and each of them is a measurement rather
than a design preference.

**A matcher is not an algorithm here, it is a choice of what to measure.** Every entry in
`MATCHERS` is a function from features to an `AttributedGraph`, and the comparison is
always TG3.3's exhaustive `AttributedGraph.matches`. That separation matters: it means a
registered matcher cannot fail by being sloppy about the node correspondence, because it
does not get to decide the correspondence. It can only fail by keying on the wrong
quantity, which is the failure the gate is about.

**Dividing by a modelled quantity imports that quantity's bias.** TG3.3's `distance`
relation is a separation over the geometric mean of the two features' spatial scales, and
it is dimensionless, so it looked scale-invariant. Measured against the benchmark across a
sixfold range of `scale_factor` it is not: the extractor's scale estimate drifts from about
+3.6% at sigma 3 to about -2.6% at sigma 18, and since the separation itself is recovered
to better than 1%, the whole of that drift lands in the quotient. The relation moves by 4.4%
under a threefold rescaling while replicates of the identical configuration move by 0.9%.
`scale_normalised` is kept as a function and left out of `MATCHERS`, because a registry
entry carries a declaration and this one has no declaration that can be demonstrated - the
drift is real at `scale_factor=3` and too mild across the rest of the range for a rank test
to reject at a corrected alpha.

**The invariant that survives divides one measurement by another of the same kind.** A
separation over a separation cancels its units exactly and involves no estimated scale at
all. Normalising each edge by the geometric mean of every edge in the graph gives a shape
that reproduces to 0.37% across all three transforms - inside the replicate noise. The
price is that it needs three features: with two there is one edge, its ratio to its own
mean is 1, and the only dimensionless quantity available carries no information. That is
why TG3.3 divided by the modelled scale, and it was the right choice for two features.

**The tolerance is calibrated, not chosen.** A matcher compared at a tolerance its author
picked is a matcher tuned until it passed - the same defect a fixed suppression radius
already cost this tree once in TG2.2. `calibrate_match_tolerance` re-measures the *same*
configuration under different noise realisations and reports the smallest tolerance at
which it agrees with itself. Anything wider is unearned, and a matcher that needs wider is
not invariant, it is blurred.

**Invariance is a claim about a matcher, and it needs a test rather than a threshold.**
`MatchTolerance` is an operating point for one comparison - the largest deviation the noise
produced - and thresholding a family of presentations at it would reject a genuinely
invariant matcher about one time in five, which is R18's subject exactly: thirteen chances
at a one-in-sixty-seven event. `measure_invariance` compares the presentations against the
whole null distribution with a rank test, corrects alpha across matchers and transforms, and
carries TG2.4's `vacuous` rule - a test whose smallest possible p-value sits above its own
alpha reports invariance because it could not have reported anything else.

**A declared capability is measured, not trusted.** `audit_declared_invariance` runs every
registered matcher against every presentation and compares what it survived with what it
declared. Overclaiming fails; so does understating, for the reason TG3.1's relation axis
gives - a receipt naming a property the run did not have is wrong in either direction, and
conservatism is not honesty. The position-memorising matcher the roadmap asks to see fail
is registered here as `absolute_position`, declaring no invariance at all, and it fails the
audit not by a special case but by the general rule applied to it.

What this module deliberately does not do: recover a scale ratio across a domain boundary.
A ratio of separations in cells to separations in metres is not a number, and TG3.5's
motif mining is where a cross-domain configuration will need one - if it turns out to need
one at all.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from scipy.stats import mannwhitneyu

from src.core.constellation import (
    AttributedGraph,
    MatchReport,
    RelationContext,
    RelationValue,
    carried_record,
    constellation,
    node_attributes,
)
from src.core.errors import InvalidParameterError
from src.core.feature import SemanticComparisonError, SpectralFeature
from src.core.registry import Registry

#: Family-wise error rate for the invariance audit, divided across matchers and transforms.
DEFAULT_ALPHA = 0.05

#: The transforms `build_planted_configuration` can apply, and the gate's vocabulary.
TRANSFORMS: Tuple[str, ...] = ("rotation", "translation", "rescaling")

#: The edge quantity `relative_geometry` keys on. Deliberately *not* in `RELATIONS`: a
#: `Relation` is a function of two features, and this one is normalised by every edge in
#: the graph, so it cannot be measured on a pair in isolation.
SHAPE_RATIO = "shape_ratio"

#: The edge quantity `absolute_position` keys on - a raw separation, which TG2.1 refuses to
#: compare across a domain boundary. That refusal is the point: see `absolute_position`.
RAW_SEPARATION = "raw_separation"


class ScaleRatioUnavailableError(InvalidParameterError):
    """Raised when a scale ratio is asked for where one is not a number.

    Two graphs whose separations are in different units have no ratio, and neither do two
    that were never matched - a ratio between arbitrary orderings of unmatched nodes is a
    number without a referent.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__("scale_ratio", reason, detail)
        self.reason = reason


# --------------------------------------------------------------------- measured records


@dataclass(frozen=True)
class MatchTolerance:
    """The smallest tolerance at which a configuration agrees with itself.

    `value` is a *measurement of the pipeline*, not a parameter of the matcher: it is the
    largest relative disagreement observed between replicates of one configuration under
    different noise. A match at a wider tolerance than this has not been shown to mean
    anything, and a match at a narrower one would be rejecting noise.

    It is a floor, and honestly so. More replicates can only widen it, never narrow it, so
    `n_replicates` is carried and a reader can see how thin the estimate is.
    """

    value: float
    n_replicates: int
    components: int
    worst_component: str
    basis: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0.0:
            raise InvalidParameterError(
                "MatchTolerance.value", self.value,
                "a finite non-negative relative tolerance")
        if self.n_replicates < 2:
            raise InvalidParameterError(
                "MatchTolerance.n_replicates", self.n_replicates,
                "at least two replicates. One realisation agrees with itself exactly, and "
                "a tolerance of zero derived that way would be a statement about having "
                "measured once")
        if not str(self.basis).strip():
            raise InvalidParameterError(
                "MatchTolerance.basis", self.basis,
                "a stated basis. A tolerance whose derivation is not written down is "
                "indistinguishable from one that was tuned")

    def describe(self) -> Dict[str, Any]:
        return {"value": self.value, "n_replicates": self.n_replicates,
                "components": self.components, "worst_component": self.worst_component,
                "basis": self.basis}


@dataclass(frozen=True)
class ScaleRatio:
    """How much bigger the right configuration is than the left, or why that is not asked.

    Recovered from the separations rather than from the features' spatial scales, because
    the separations are what the extractor measures without a model - across the benchmark's
    sixfold range this recovers `scale_factor` to better than 1%, where the scale estimate
    drifts by 6%.
    """

    value: Optional[float]
    units: Optional[str]
    basis: str
    unavailable: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) == (not self.unavailable):
            raise InvalidParameterError(
                "ScaleRatio", (self.value, self.unavailable),
                "either a value or a stated reason there is none, and not both")
        if self.value is not None and (not math.isfinite(self.value) or self.value <= 0.0):
            raise InvalidParameterError(
                "ScaleRatio.value", self.value, "a positive finite ratio")

    def __bool__(self) -> bool:
        return self.value is not None

    def describe(self) -> Dict[str, Any]:
        return {"value": self.value, "units": self.units, "basis": self.basis,
                "unavailable": self.unavailable}


@dataclass(frozen=True)
class InvariantMatch:
    """One comparison: which matcher, what it decided, and the scale it recovered."""

    matcher: str
    report: MatchReport
    scale_ratio: ScaleRatio
    tolerance: MatchTolerance

    def __bool__(self) -> bool:
        return bool(self.report)

    def describe(self) -> Dict[str, Any]:
        return {"matcher": self.matcher, "match": self.report.describe(),
                "scale_ratio": self.scale_ratio.describe(),
                "tolerance": self.tolerance.describe()}


@dataclass(frozen=True)
class Presentation:
    """The same configuration, shown under a named set of transforms."""

    name: str
    transforms: Tuple[str, ...]
    features: Tuple[SpectralFeature, ...]
    expected_scale_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        unknown = [t for t in self.transforms if t not in TRANSFORMS]
        if unknown:
            raise InvalidParameterError(
                "Presentation.transforms", unknown,
                "transforms drawn from %s" % ", ".join(TRANSFORMS))
        if self.expected_scale_ratio is not None and (
                not math.isfinite(self.expected_scale_ratio)
                or self.expected_scale_ratio <= 0.0):
            raise InvalidParameterError(
                "Presentation.expected_scale_ratio", self.expected_scale_ratio,
                "a positive finite ratio, or None where the presentation was not resized")
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "features", tuple(self.features))

    @property
    def is_single(self) -> bool:
        """One transform at a time - the only presentations that can attribute a failure."""
        return len(self.transforms) == 1


@dataclass(frozen=True)
class InvarianceTest:
    """One transform, tested: is this matcher's deviation under it bigger than noise?

    `p_value` answers "how often would noise alone put these presentations this high", and
    `power_floor` answers the question a p-value cannot: could this test have failed at all?
    A test that could not have failed reports invariance because it was never able to report
    anything else, and `vacuous` says so rather than letting the pass stand.
    """

    transform: str
    presentations: Tuple[str, ...]
    deviations: Tuple[float, ...]
    n_null: int
    p_value: float
    alpha: float
    power_floor: float

    @property
    def vacuous(self) -> bool:
        """True when no result could have failed, so a pass means nothing."""
        return self.power_floor > self.alpha

    @property
    def invariant(self) -> bool:
        return not self.vacuous and self.p_value >= self.alpha

    def describe(self) -> Dict[str, Any]:
        return {"transform": self.transform, "presentations": list(self.presentations),
                "deviations": list(self.deviations), "n_null": self.n_null,
                "p_value": self.p_value, "alpha": self.alpha,
                "power_floor": self.power_floor, "vacuous": self.vacuous,
                "invariant": self.invariant}


@dataclass(frozen=True)
class ScaleRecoveryTest:
    """Did the matcher recover how much bigger the configuration was, or only that it was?

    The bar is the recovery's own noise floor, not the shape's. A matcher blind to scale and
    a matcher that measures scale and states it both match a rescaled triangle; only one of
    them produces this number, and only a number checked against the right null says
    anything.
    """

    presentations: Tuple[str, ...]
    expected: Tuple[float, ...]
    recovered: Tuple[Optional[float], ...]
    errors: Tuple[float, ...]
    n_null: int
    p_value: float
    alpha: float
    power_floor: float

    @property
    def vacuous(self) -> bool:
        return self.power_floor > self.alpha

    @property
    def accurate(self) -> bool:
        return not self.vacuous and self.p_value >= self.alpha

    def describe(self) -> Dict[str, Any]:
        return {"presentations": list(self.presentations), "expected": list(self.expected),
                "recovered": list(self.recovered), "errors": list(self.errors),
                "n_null": self.n_null, "p_value": self.p_value, "alpha": self.alpha,
                "power_floor": self.power_floor, "vacuous": self.vacuous,
                "accurate": self.accurate}


@dataclass(frozen=True)
class InvarianceReport:
    """What a matcher declared, what it survived, and the gap between them."""

    matcher: str
    declared: Tuple[str, ...]
    measured: Tuple[str, ...]
    tolerance: MatchTolerance
    matches: Mapping[str, InvariantMatch] = dc_field(default_factory=dict)
    tests: Mapping[str, InvarianceTest] = dc_field(default_factory=dict)
    scale_recovery: Optional[ScaleRecoveryTest] = None

    @property
    def vacuous(self) -> Tuple[str, ...]:
        """Transforms whose test could not have failed. A pass on one of those is not one."""
        return tuple(name for name, test in sorted(self.tests.items()) if test.vacuous)

    @property
    def overclaimed(self) -> Tuple[str, ...]:
        return tuple(t for t in self.declared if t not in self.measured)

    @property
    def understated(self) -> Tuple[str, ...]:
        return tuple(t for t in self.measured if t not in self.declared)

    @property
    def honest(self) -> bool:
        """Declared exactly what it survived - no more, and no less.

        Understating fails for the reason TG3.1's relation axis gives: a receipt naming a
        property the run did not have is wrong in either direction, and a matcher that
        under-declares makes a caller reach for a heavier one it did not need.
        """
        return (not self.overclaimed and not self.understated and not self.vacuous)

    def describe(self) -> Dict[str, Any]:
        return {
            "matcher": self.matcher,
            "declared": list(self.declared),
            "measured": list(self.measured),
            "overclaimed": list(self.overclaimed),
            "understated": list(self.understated),
            "honest": self.honest,
            "vacuous": list(self.vacuous),
            "tolerance": self.tolerance.describe(),
            "tests": {name: t.describe() for name, t in sorted(self.tests.items())},
            "scale_recovery": (None if self.scale_recovery is None
                               else self.scale_recovery.describe()),
            "matches": {name: m.describe() for name, m in sorted(self.matches.items())},
        }


# -------------------------------------------------------------------- the separations


def _separations(features: Sequence[SpectralFeature],
                 context: RelationContext) -> Tuple[Dict[Tuple[int, int], float],
                                                    Optional[str]]:
    """Every ordered pair's raw separation, and the units they are all in.

    The units are read off the quantities rather than assumed: a configuration whose axes
    disagree with each other would produce separations that cannot be divided by one
    another, and the shape would then be a number with no meaning rather than a refusal.
    """
    values: Dict[Tuple[int, int], float] = {}
    units: Optional[str] = None
    seen_units = False
    for left, right in itertools.permutations(range(len(features)), 2):
        if (right, left) in values:
            values[(left, right)] = values[(right, left)]
            continue
        quantity = features[left].separation_to(features[right], periods=context.periods)
        if seen_units and quantity.units != units:
            raise SemanticComparisonError(
                "separation", str(units), str(quantity.units),
                "The separations within one configuration are in two different units, so "
                "their ratio is not dimensionless and the shape they describe is not a "
                "shape.")
        units, seen_units = quantity.units, True
        if quantity.value <= 0.0:
            raise InvalidParameterError(
                "separation", quantity.value,
                "a positive separation between distinct features %d and %d. Two features "
                "at the same place have no shape between them" % (left, right))
        values[(left, right)] = quantity.value
    return values, units


def _geometric_mean_separation(values: Mapping[Tuple[int, int], float]) -> float:
    """The graph's own length unit.

    Geometric rather than arithmetic for the reason TG3.3 gives for its reference scale: a
    ratio wants a multiplicative centre, or the same configuration reads differently
    depending on which edge happened to be longest.
    """
    unique = [v for (a, b), v in values.items() if a < b]
    return math.exp(sum(math.log(v) for v in unique) / len(unique))


# ------------------------------------------------------------------------ the matchers


def _graph(features: Sequence[SpectralFeature],
           attributes: Sequence[Mapping[str, Any]],
           edges: Mapping[Tuple[int, int], Mapping[str, RelationValue]],
           relation: str,
           extra_carried: Mapping[int, Mapping[str, Any]]) -> AttributedGraph:
    carried = []
    for index, feature in enumerate(features):
        record = dict(carried_record(feature))
        record.update(extra_carried.get(index, {}))
        carried.append(record)
    return AttributedGraph(attributes=tuple(attributes), carried=tuple(carried),
                           edges=edges, relations=(relation,))


def relative_geometry(features: Sequence[SpectralFeature],
                      context: RelationContext) -> AttributedGraph:
    """Each edge over the geometric mean of every edge: the shape, and nothing else.

    Units cancel exactly, because both halves are separations measured the same way. No
    estimated quantity enters, which is what makes this survive a rescaling that the
    scale-normalised relation does not.
    """
    separations, units = _separations(features, context)
    reference = _geometric_mean_separation(separations)
    basis = ("separation over the geometric mean of all %d separations in the "
             "configuration, both in %s"
             % (len(features) * (len(features) - 1) // 2,
                units if units is not None else "no units"))
    edges = {pair: {SHAPE_RATIO: RelationValue(SHAPE_RATIO, value / reference, None, basis)}
             for pair, value in separations.items()}
    extra = {index: {"separation_units": units,
                     "separation_geometric_mean": reference}
             for index in range(len(features))}
    return _graph(features, [node_attributes(f) for f in features], edges,
                  SHAPE_RATIO, extra)


def scale_normalised(features: Sequence[SpectralFeature],
                     context: RelationContext) -> AttributedGraph:
    """TG3.3's `distance` relation, unchanged - and deliberately **not** in `MATCHERS`.

    It is the comparison that works on two features and the one that crosses a domain
    boundary, so it is public and TG3.5 will want it: `relative_geometry` needs three
    features, and a pair has no shape.

    It is unregistered because a registry entry carries a declaration and there is no true
    value to write in this one. Measured directly, its rescaling deviation reaches 4.8% at
    `scale_factor=3` against a noise floor of 3.3%, so it is not rescaling-invariant. Tested
    the way `measure_invariance` tests, over seven rescalings, the evidence comes to
    `p = 0.016` - short of a family-corrected alpha, because the drift is mild across most
    of the range and a rank test cannot see a trend it is not looking for. So the claim
    "not invariant to rescaling" is true and unproven at once, and neither declaring it nor
    omitting it would be a statement this module could stand behind. What can be stood
    behind is the refusal to register it, and the reason.
    """
    graph = constellation(features, relations=["distance"], context=context)
    separations, units = _separations(features, context)
    reference = _geometric_mean_separation(separations)
    carried = tuple(dict(c, separation_units=units, separation_geometric_mean=reference)
                    for c in graph.carried)
    return AttributedGraph(attributes=graph.attributes, carried=carried,
                           edges=graph.edges, relations=graph.relations)


def absolute_position(features: Sequence[SpectralFeature],
                      context: RelationContext) -> AttributedGraph:
    """The matcher the roadmap asks to see fail: it remembers where things were.

    Nothing here is dishonest except by omission, and the omission is declared - it claims
    no invariance. It is the implementation someone writes first, and it works perfectly on
    a configuration nobody moved, which is exactly why the gate has to exist.
    """
    separations, units = _separations(features, context)
    attributes = []
    for feature in features:
        record = dict(node_attributes(feature))
        record.update({"coord_%s" % name: float(value)
                       for name, value in sorted(feature.location.coords.items())})
        attributes.append(record)
    basis = "the separation itself, in %s" % (units if units is not None else "no units")
    edges = {pair: {RAW_SEPARATION: RelationValue(RAW_SEPARATION, value, None, basis)}
             for pair, value in separations.items()}
    extra = {index: {"separation_units": units,
                     "separation_geometric_mean": _geometric_mean_separation(separations)}
             for index in range(len(features))}
    return _graph(features, attributes, edges, RAW_SEPARATION, extra)


Matcher = Callable[[Sequence[SpectralFeature], RelationContext], AttributedGraph]

MATCHERS: Registry[Matcher] = Registry("constellation matcher")

MATCHERS.add(
    "relative_geometry", relative_geometry,
    description="Each separation over the geometric mean of all of them.",
    capabilities={"invariant_to": ("rotation", "translation", "rescaling"),
                  "min_nodes": 3, "crosses_domains": True,
                  "keys_on": SHAPE_RATIO})
MATCHERS.add(
    "absolute_position", absolute_position,
    description="Coordinates and raw separations - the position-memorising control.",
    capabilities={"invariant_to": (), "min_nodes": 2, "crosses_domains": False,
                  "keys_on": RAW_SEPARATION})


def matcher_for(name: str) -> Matcher:
    return MATCHERS.get(name)


def declared_invariance(name: str) -> Tuple[str, ...]:
    """What a registered matcher claims. `audit_declared_invariance` checks it."""
    return tuple(MATCHERS.entry(name).capabilities.get("invariant_to", ()))


def build_signature(name: str, features: Sequence[SpectralFeature], *,
                    context: Optional[RelationContext] = None) -> AttributedGraph:
    """Run one registered matcher's measurement, refusing a configuration too small for it.

    The minimum is not a performance guard. `relative_geometry` on two features would
    normalise the single edge by itself and return 1.0 for every configuration in the
    world, which is a matcher that matches everything - and a matcher that matches
    everything reports a discovery on every pair it is shown.
    """
    entry = MATCHERS.entry(name)                # UnknownNameError, with a did-you-mean
    minimum = int(entry.capabilities.get("min_nodes", 2))
    features = tuple(features)
    if len(features) < minimum:
        raise InvalidParameterError(
            "build_signature.features", len(features),
            "at least %d features for matcher %r: %s"
            % (minimum, name,
               "a shape is a ratio between separations, and %d feature(s) provide %d "
               "separation(s) to take a ratio of"
               % (len(features), len(features) * (len(features) - 1) // 2)))
    return entry.value(features, context or RelationContext())


# ------------------------------------------------------------------------- calibration


def _deviation_under(left: AttributedGraph, right: AttributedGraph,
                     permutation: Sequence[int]) -> Tuple[float, str]:
    """The largest relative disagreement between two graphs under one correspondence.

    Node attributes as well as edges: a matcher whose node attributes wobble between
    realisations is as unreliable as one whose edges do, and calibrating on the edges alone
    would produce a tolerance the attributes then violate. A non-numeric disagreement is
    infinite rather than large, because no tolerance makes two different conventions the
    same convention.

    The denominator is `max(1, |a|, |b|)`, matching TG3.3's `_close` exactly. A tolerance
    measured under one rule and applied under another would be a number that means
    something slightly different from what it is used for.
    """
    worst, key = 0.0, "(none)"
    for index, mapped in enumerate(permutation):
        a_attr, b_attr = left.attributes[index], right.attributes[mapped]
        if set(a_attr) != set(b_attr):
            return math.inf, "node%d.(different attributes)" % index
        for name in sorted(a_attr):
            a, b = a_attr[name], b_attr[name]
            numeric = (not isinstance(a, bool) and not isinstance(b, bool)
                       and isinstance(a, (int, float)) and isinstance(b, (int, float)))
            if not numeric:
                if a != b:
                    return math.inf, "node%d.%s" % (index, name)
                continue
            deviation = abs(float(a) - float(b)) / max(1.0, abs(float(a)), abs(float(b)))
            if deviation > worst:
                worst, key = deviation, "node%d.%s" % (index, name)
    for (i, j), edge in sorted(left.edges.items()):
        counterpart = right.edges.get((permutation[i], permutation[j]))
        if counterpart is None:
            return math.inf, "edge%d-%d.(absent)" % (i, j)
        for name in sorted(edge):
            if name not in counterpart:
                return math.inf, "edge%d-%d.%s" % (i, j, name)
            a_value, b_value = edge[name], counterpart[name]
            if a_value.holds != b_value.holds:
                return math.inf, "edge%d-%d.%s" % (i, j, name)
            if a_value.value is None or b_value.value is None:
                if (a_value.value is None) != (b_value.value is None):
                    return math.inf, "edge%d-%d.%s" % (i, j, name)
                continue
            deviation = (abs(a_value.value - b_value.value)
                         / max(1.0, abs(a_value.value), abs(b_value.value)))
            if deviation > worst:
                worst, key = deviation, "edge%d-%d.%s" % (i, j, name)
    return worst, key


def best_deviation(left: AttributedGraph, right: AttributedGraph) -> Tuple[float, str]:
    """The smallest tolerance at which these two graphs would match, and what set it.

    Minimised over the same `k!` correspondences `AttributedGraph.matches` searches, and for
    the same reason the null needs it: a deterministic extractor does not return its
    features in a stable order between noise realisations, because the order follows which
    blob happened to be brightest. Comparing replicate node 0 with replicate node 0 measures
    that shuffle instead of the measurement. It is not a small effect - it put the position
    matcher's noise floor at 0.27, wide enough to call a 37-degree rotation the same picture.
    """
    if left.n_nodes != right.n_nodes:
        return math.inf, "(different node counts)"
    best, key = math.inf, "(none)"
    for permutation in itertools.permutations(range(right.n_nodes)):
        deviation, component = _deviation_under(left, right, permutation)
        if deviation < best:
            best, key = deviation, component
    return best, key


def _component_count(graph: AttributedGraph) -> int:
    """How many numbers a comparison of this graph reads - for the receipt, not the rule."""
    nodes = sum(1 for a in graph.attributes for v in a.values()
                if not isinstance(v, bool) and isinstance(v, (int, float)))
    return nodes + sum(len(e) for e in graph.edges.values())


def _replicate_graphs(replicates: Sequence[Sequence[SpectralFeature]], matcher: str,
                      context: Optional[RelationContext]) -> Tuple[AttributedGraph, ...]:
    prepared = [tuple(r) for r in replicates]
    if len(prepared) < 2:
        raise InvalidParameterError(
            "replicates", len(prepared),
            "at least two realisations of the same configuration. A noise floor measured "
            "from one realisation is zero, and zero is not a measurement of noise")
    sizes = {len(r) for r in prepared}
    if len(sizes) != 1:
        raise InvalidParameterError(
            "replicates", sorted(sizes),
            "the same number of features in every replicate. Replicates that disagree "
            "about how many features there are disagree about more than tolerance, and "
            "averaging over that would bury a detection failure inside a measurement error")
    return tuple(build_signature(matcher, r, context=context) for r in prepared)


def null_deviations(replicates: Sequence[Sequence[SpectralFeature]], *, matcher: str,
                    context: Optional[RelationContext] = None) -> Tuple[float, ...]:
    """How far apart two measurements of the *same* configuration land, over every pair.

    This is the null the whole slice is measured against: the configuration is identical in
    every replicate, so any deviation here is the pipeline's own noise. `C(R, 2)` values
    from `R` replicates, and the count matters - it sets the smallest p-value the invariance
    test can ever return, which is what `InvarianceTest.vacuous` is about.
    """
    graphs = _replicate_graphs(replicates, matcher, context)
    values = []
    for left, right in itertools.combinations(graphs, 2):
        deviation, key = best_deviation(left, right)
        if not math.isfinite(deviation):
            raise InvalidParameterError(
                "replicates", key,
                "replicates that differ only in noise. Two realisations of one "
                "configuration disagreed on %r in a way no tolerance can close, so what "
                "would be measured here is a detection failure wearing the name of a "
                "measurement error" % key)
        values.append(deviation)
    return tuple(values)


def calibrate_match_tolerance(replicates: Sequence[Sequence[SpectralFeature]], *,
                              matcher: str,
                              context: Optional[RelationContext] = None) -> MatchTolerance:
    """The operating point for a *single* comparison: the largest deviation noise produced.

    There is no safety factor, because a safety factor is a free parameter and a free
    parameter is where a tuned result hides. The false-rejection rate that comes with it is
    neither zero nor hidden - it is about `1/(n_null + 1)`, since a fresh pair of noise
    realisations exceeds the largest of `n_null` previous ones about that often, and the
    number is written into the basis.

    **This is not how invariance is decided.** Thresholding a dozen presentations at this
    value would reject a genuinely invariant matcher about one time in five, for exactly the
    reason R18 exists: thirteen chances at a one-in-sixty-seven event. `measure_invariance`
    tests the presentations against the whole null distribution instead.
    """
    graphs = _replicate_graphs(replicates, matcher, context)
    components = _component_count(graphs[0])
    worst, worst_component, pairs = 0.0, "(none)", 0
    for left, right in itertools.combinations(graphs, 2):
        deviation, key = best_deviation(left, right)
        pairs += 1
        if deviation > worst:
            worst, worst_component = deviation, key
    return MatchTolerance(
        value=worst, n_replicates=len(graphs), components=components,
        worst_component=worst_component,
        basis="the largest relative disagreement between %d replicates of one "
              "configuration under matcher %r, over %d measured components, minimised "
              "over the same node correspondences a match searches; false-rejection rate "
              "about %.3g for a single comparison"
              % (len(graphs), matcher, components, 1.0 / (pairs + 1)))


# ---------------------------------------------------------------------------- matching


def _geometric_mean_ratio(left: AttributedGraph, right: AttributedGraph) -> float:
    return (float(right.carried[0]["separation_geometric_mean"])
            / float(left.carried[0]["separation_geometric_mean"]))


def _carries_separation_scale(graph: AttributedGraph) -> bool:
    """Did the matcher that built this graph record a length to take a ratio of?

    Not every matcher does. TG3.5's constant-signature control measures nothing at all, and
    asking it how much bigger one configuration is than another is a question it has no
    number for - which is a refusal by name rather than a `KeyError`, because the two read
    very differently to whoever has to act on it.
    """
    return all("separation_geometric_mean" in record for record in graph.carried)


def _log_error(found: float, expected: float) -> float:
    """`|ln(found/expected)|` - symmetric, so recovering 2x and 0.5x are equally wrong.

    A plain relative error is not: reporting 1.5 where the answer is 3 is a 50% error and
    reporting 3 where the answer is 1.5 is a 100% one, and the two are the same mistake.
    """
    return abs(math.log(found / expected))


def scale_recovery_null(replicates: Sequence[Sequence[SpectralFeature]], *, matcher: str,
                        context: Optional[RelationContext] = None) -> Tuple[float, ...]:
    """How wrong the recovered ratio is between two measurements of one configuration.

    Every replicate is the same size, so the true ratio is 1 and every deviation here is
    the recovery's own noise. This is a *different* null from `null_deviations`, and it has
    to be: that one measures the shape, this one measures the size, and using the shape's
    floor to judge a size - which is what the first version of the 4E gate did - compares a
    number with the noise of a different number.
    """
    graphs = _replicate_graphs(replicates, matcher, context)
    return tuple(_log_error(_geometric_mean_ratio(a, b), 1.0)
                 for a, b in itertools.combinations(graphs, 2))


def recover_scale_ratio(left: AttributedGraph, right: AttributedGraph,
                        report: MatchReport) -> ScaleRatio:
    """How much bigger `right` is than `left`, once they have been found to be the same.

    Takes the `MatchReport` rather than a flag, so it structurally cannot run before a
    decision has been made. That ordering is what keeps R19 intact: the separation scale
    lives in `carried`, nothing in `matches` can read it, and this reads it only to
    *describe* a match that was already decided without it.
    """
    if not report.matched:
        raise ScaleRatioUnavailableError(
            "an established correspondence",
            "The two graphs did not match, so there is no pairing of their nodes and a "
            "ratio of their sizes would be a number relating two different things.")
    if not (_carries_separation_scale(left) and _carries_separation_scale(right)):
        return ScaleRatio(
            None, None,
            "a ratio of the two configurations' geometric mean separations",
            unavailable="At least one of these graphs carries no separation scale, so its "
                        "matcher measured no length and there is nothing to take a ratio "
                        "of. A "
                        "matcher can recognise two configurations as the same shape "
                        "without ever having measured how big either of them was.")
    left_units = {c.get("separation_units") for c in left.carried}
    right_units = {c.get("separation_units") for c in right.carried}
    if left_units != right_units:
        return ScaleRatio(
            None, None,
            "a ratio of the two configurations' geometric mean separations",
            unavailable="The separations are in %s and %s. A configuration in one unit is "
                        "not a number of times bigger than a configuration in another, and "
                        "the match that crossed that boundary crossed it precisely because "
                        "it had divided the units out."
                        % (sorted(map(str, left_units)), sorted(map(str, right_units))))
    units = next(iter(left_units))
    return ScaleRatio(
        _geometric_mean_ratio(left, right), units,
        "the right configuration's geometric mean separation over the left's, both in %s; "
        "recovered from the separations rather than from the features' estimated scales, "
        "which drift with the scale being estimated"
        % (units if units is not None else "no units"))


def match(left: Sequence[SpectralFeature], right: Sequence[SpectralFeature], *,
          matcher: str, tolerance: MatchTolerance,
          context: Optional[RelationContext] = None) -> InvariantMatch:
    """Compare two configurations under one registered matcher at a measured tolerance.

    `tolerance` is a `MatchTolerance` and not a float on purpose. A bare number here would
    accept a value chosen because it made the comparison pass, and the argument of this
    module is that such a number cannot be told apart from a measured one after the fact.
    """
    if not isinstance(tolerance, MatchTolerance):
        raise InvalidParameterError(
            "match.tolerance", type(tolerance).__name__,
            "a MatchTolerance from `calibrate_match_tolerance`. A tolerance that was "
            "chosen rather than measured cannot be distinguished from one that was tuned "
            "until the comparison passed")
    context = context or RelationContext()
    left_graph = build_signature(matcher, left, context=context)
    right_graph = build_signature(matcher, right, context=context)
    report = left_graph.matches(right_graph, tolerance=tolerance.value)
    if report.matched:
        ratio = recover_scale_ratio(left_graph, right_graph, report)
    else:
        ratio = ScaleRatio(
            None, None, "a ratio of the two configurations' geometric mean separations",
            unavailable="The graphs did not match, so there is no correspondence to "
                        "measure a ratio along.")
    return InvariantMatch(matcher=matcher, report=report, scale_ratio=ratio,
                          tolerance=tolerance)


# --------------------------------------------------------------------- is it invariant?


def _rank_sum_p_value(observed: Sequence[float], null: Sequence[float]) -> float:
    """One-sided: how often noise alone would put the presentations this high.

    A rank test rather than a threshold because the presentations and the null replicates
    measure the same thing - two realisations of one configuration - whenever the matcher is
    invariant. Under that hypothesis they are exchangeable, and the question is whether the
    presentation sample sits systematically higher, not whether any single member of it
    happened to clear a line.
    """
    return float(mannwhitneyu(list(observed), list(null), alternative="greater").pvalue)


def _power_floor(n_observed: int, n_null: int) -> float:
    """The smallest p-value this comparison could possibly return.

    TG2.4's `vacuous` rule, applied to a rank test: a plane whose null nothing could exceed
    was counted as clean, and a test whose floor sits above its own alpha is the same
    failure - it reports invariance because it could not have reported anything else.
    """
    return 1.0 / float(math.comb(n_observed + n_null, n_observed))


def measure_invariance(matcher: str, reference: Sequence[SpectralFeature],
                       presentations: Sequence[Presentation],
                       replicates: Sequence[Sequence[SpectralFeature]], *,
                       alpha: float = DEFAULT_ALPHA,
                       context: Optional[RelationContext] = None) -> InvarianceReport:
    """Which transforms this matcher survives, against what it declared.

    Only presentations applying a single transform can attribute a result to that transform.
    Combined presentations are measured and reported but confer nothing on their own: a
    combined presentation that matched could have matched because two errors cancelled, and
    a cancellation is not a property.

    `alpha` is divided across the tests actually run - one per transform, plus the scale
    recovery where any presentation says how much bigger it was built. That is R18 at its
    smallest scale: four chances to declare a matcher broken is a family, and an uncorrected
    alpha over it calls an honest matcher a liar about one run in six.
    """
    context = context or RelationContext()
    null = null_deviations(replicates, matcher=matcher, context=context)
    reference_graph = build_signature(matcher, reference, context=context)
    tolerance = calibrate_match_tolerance(replicates, matcher=matcher, context=context)

    single: Dict[str, List[Presentation]] = {}
    for presentation in presentations:
        if presentation.is_single:
            single.setdefault(presentation.transforms[0], []).append(presentation)
    if not single:
        raise InvalidParameterError(
            "measure_invariance.presentations", [p.name for p in presentations],
            "at least one presentation applying a single transform. Combined presentations "
            "can show that something is wrong but cannot say which transform is wrong, and "
            "a gate that cannot name what failed is a gate nobody can act on")
    resized = [p for p in presentations if p.expected_scale_ratio is not None]
    per_test = alpha / float(len(single) + (1 if resized else 0))

    matches = {p.name: match(reference, p.features, matcher=matcher, tolerance=tolerance,
                             context=context)
               for p in presentations}
    tests: Dict[str, InvarianceTest] = {}
    for transform, group in sorted(single.items()):
        deviations = tuple(
            best_deviation(reference_graph,
                           build_signature(matcher, p.features, context=context))[0]
            for p in group)
        tests[transform] = InvarianceTest(
            transform=transform, presentations=tuple(p.name for p in group),
            deviations=deviations, n_null=len(null),
            p_value=_rank_sum_p_value(deviations, null), alpha=per_test,
            power_floor=_power_floor(len(deviations), len(null)))
    recovery = None
    if resized and _carries_separation_scale(reference_graph):
        null_ratio = scale_recovery_null(replicates, matcher=matcher, context=context)
        found = tuple(_geometric_mean_ratio(
            reference_graph, build_signature(matcher, p.features, context=context))
            for p in resized)
        errors = tuple(_log_error(f, p.expected_scale_ratio)
                       for f, p in zip(found, resized))
        recovery = ScaleRecoveryTest(
            presentations=tuple(p.name for p in resized),
            expected=tuple(float(p.expected_scale_ratio) for p in resized),
            recovered=found, errors=errors, n_null=len(null_ratio),
            p_value=_rank_sum_p_value(errors, null_ratio), alpha=per_test,
            power_floor=_power_floor(len(errors), len(null_ratio)))

    measured = tuple(t for t in TRANSFORMS if t in tests and tests[t].invariant)
    return InvarianceReport(
        matcher=matcher, declared=declared_invariance(matcher), measured=measured,
        tolerance=tolerance, matches=matches, tests=tests, scale_recovery=recovery)


def audit_declared_invariance(reference: Sequence[SpectralFeature],
                              presentations: Sequence[Presentation],
                              replicates: Sequence[Sequence[SpectralFeature]], *,
                              alpha: float = DEFAULT_ALPHA,
                              names: Optional[Sequence[str]] = None,
                              context: Optional[RelationContext] = None) \
        -> Dict[str, InvarianceReport]:
    """Every registered matcher, measured against its own declaration.

    Each matcher gets its own null and its own tolerance, measured through its own
    signature. Sharing one across matchers would compare a matcher keying on coordinates in
    cells with one keying on a dimensionless ratio, and whichever quantity happened to be
    noisier would set the bar for both.

    `alpha` is divided across the matchers as well as, inside each, across the transforms:
    auditing three matchers is three times as many chances to call one of them a liar.

    Matchers whose minimum size the configuration cannot meet are skipped rather than
    failed. A matcher that was never run has not overclaimed anything.
    """
    chosen = list(names) if names is not None else MATCHERS.names()
    runnable = [name for name in chosen
                if len(tuple(reference)) >= int(
                    MATCHERS.entry(name).capabilities.get("min_nodes", 2))]
    if not runnable:
        raise InvalidParameterError(
            "audit_declared_invariance.reference", len(tuple(reference)),
            "a configuration large enough for at least one registered matcher")
    per_matcher = alpha / float(len(runnable))
    return {name: measure_invariance(name, reference, presentations, replicates,
                                     alpha=per_matcher, context=context)
            for name in runnable}


__all__ = [
    "TRANSFORMS", "SHAPE_RATIO", "RAW_SEPARATION", "DEFAULT_ALPHA",
    "ScaleRatioUnavailableError", "MatchTolerance", "ScaleRatio", "InvariantMatch",
    "Presentation", "InvarianceTest", "InvarianceReport", "MATCHERS", "matcher_for",
    "ScaleRecoveryTest", "declared_invariance", "build_signature", "best_deviation",
    "null_deviations", "scale_recovery_null",
    "calibrate_match_tolerance", "recover_scale_ratio", "match", "measure_invariance",
    "audit_declared_invariance", "relative_geometry", "scale_normalised",
    "absolute_position",
]
