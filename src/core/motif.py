"""TG3.5: recurring motifs - mining repeated configurations, priced and split.

A motif is a configuration that occurs more than once. Everything hard about mining for
one is a consequence of that sentence being about *counting*, and this module is mostly
the accounting that counting honestly requires.

**The family is the number of configurations looked at, and it is enormous.** Mining
`k`-feature configurations over `s` scenes of `n` features each examines `s * C(n, k)` of
them, and every one of those is a chance for a repeat to look surprising. Six scenes of six
features at `k = 3` is 120 members, which TG3.1 prices at 12,885 surrogates before its
ceiling lets a single member be rejected; eight scenes of twelve features is 1,760 members
and 283,380 surrogates. That is not a tuning problem: no achievable ensemble pays for a
mining pass in one stage. TG3.2's generate/confirm split is
therefore not an optimisation here, it is the only affordable shape, and
`motif_search_specification` computes the refusal rather than asserting it.

**A motif is an exemplar, not a cluster.** Matching under a tolerance is not transitive: A
matches B and B matches C without A matching C, and single-linkage clustering silently
promotes that chain into one motif with a support of three. So a candidate is one
occurrence, and its members are the occurrences that match *that* occurrence - a star, not
a chain. `MiningResult.intransitive_pairs` counts how often the difference would have
mattered, because a design decision whose consequence is never measured is a preference.

**Overlapping occurrences within a scene are one piece of evidence, not several.** Two
triangles in one scene sharing two of their three features are very nearly the same
observation, and counting them separately inflates support without adding data. Support is
therefore the number of *scenes* in which a motif occurs, and the occurrence count is
carried beside it as description.

**Invariance is not sufficiency.** `always_matches` is a matcher whose signature is a
constant. It is perfectly invariant to rotation, translation and rescaling - genuinely so,
not by a trick - and `test_motif.py` registers it into TG3.4's `MATCHERS` and shows that
the `4E.invariance` audit calls it honest. It is also useless: it reports every
configuration as a repeat of every other. A gate on invariance alone cannot see the
difference, which is why the motif gate is a null: the thing that separates a real motif
from a constant is that the constant recurs exactly as often in surrogate scenes.

**A surrogate has to be drawable by the process that produced the data.** The null asks how
often this much repetition happens by chance, and it answers with scenes whose features are
placed at random. Placed at random *without constraint*, two features can land closer
together than any real pair - closer than the extractor could have resolved them - and the
triangles they make are near-degenerate slivers the pipeline could never have returned. The
null then fills with shapes unlike anything the data contains, supports the motif less often
than a real arrangement would, and makes the observed support look more surprising than it
is: measured over five seeds, dropping the constraint roughly halves the p-value, from
0.20-0.25 to 0.07-0.14. `surrogate_scene` enforces the smallest separation actually
observed, and `test_motif.py` measures the difference rather than asserting it.

**Support on the training partition is descriptive and nothing else.** The exemplar is one
of the occurrences it is counted among, so its support starts at one by construction, and
it was chosen because its support was high. Both are selection, and neither is evidence.
The only thing the generate stage produces is a confirmatory family to freeze; the p-values
are computed on the held-out partition, over the frozen set, corrected at its frozen size.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.constellation import (
    AttributedGraph,
    RelationContext,
    RelationValue,
    carried_record,
)
from src.core.errors import InvalidParameterError
from src.core.family import (
    FAMILY_COMBINATORS,
    Combinator,
    SearchAxis,
    SearchSpecification,
    SearchTerm,
    max_affordable_family,
)
from src.core.feature import FeatureLocation, SpectralFeature
from src.core.invariance import best_deviation, build_signature
from src.core.preregistration import (
    HeldOutLedger,
    PartitionIdentity,
    Seal,
    confirm_on_held_out,
    freeze_confirmatory_family,
    report_generation,
)

# --------------------------------------------------------------------------- combinators

#: TG3.1's test asserted that the next search shape would be a registration rather than an
#: edit to `family.py`, and named `k`-feature constellations as the shape that would prove
#: it. This is that registration, made from the module that needs it.
_SUBSET_SIZES = {3: ("unordered_triples", "three"), 4: ("unordered_quadruples", "four")}


def _subset_combinator(size: int) -> Combinator:
    return Combinator(
        count=lambda axes, k=size: math.comb(len(_only_axis(axes, k)), k),
        enumerate=lambda axes, k=size: itertools.combinations(_only_axis(axes, k).values, k),
        n_axes=1,
        components=lambda axes, k=size: k,
    )


def _only_axis(axes: Sequence[SearchAxis], size: int) -> SearchAxis:
    if len(axes) != 1:
        raise InvalidParameterError(
            "unordered subsets of %d axes" % len(axes), [axis.name for axis in axes],
            "exactly one axis. Every member of a configuration is drawn from the same "
            "declared set of features, and a second axis would be a different question")
    return axes[0]


for _size, (_name, _word) in sorted(_SUBSET_SIZES.items()):
    if _name not in FAMILY_COMBINATORS:
        FAMILY_COMBINATORS.add(
            _name, _subset_combinator(_size),
            description=("Every unordered set of %s distinct values from one axis - the "
                         "shape a %s-feature configuration has." % (_word, _word)),
            capabilities={"ordered": False, "distinct": True, "axes": 1, "size": _size},
        )

#: The sizes a motif search can be declared at, and the combinator that prices each.
CONFIGURATION_COMBINATORS: Dict[int, str] = {
    2: "unordered_pairs", 3: "unordered_triples", 4: "unordered_quadruples"}

#: The matcher a motif search uses unless told otherwise. TG3.4 measured this one to be
#: invariant to all three transforms, which is what lets the same shape in two scenes be
#: recognised as the same shape.
DEFAULT_MATCHER = "relative_geometry"

#: How many attempts `surrogate_scene` makes to place one feature under the separation
#: constraint before it reports the constraint infeasible rather than quietly relaxing it.
MAX_PLACEMENT_ATTEMPTS = 200


# ------------------------------------------------------------------------------ refusals


class MotifSizeUnavailableError(InvalidParameterError):
    """A configuration size with no registered combinator to price it.

    Deliberately not a formula. TG3.1's whole argument is that a family is enumerated by a
    registered rule that also counts it, so that the count and the enumeration cannot drift
    apart; computing `C(n, k)` here for an arbitrary `k` would be the second formula that
    argument refuses.
    """

    def __init__(self, size: int) -> None:
        super().__init__(
            "configuration size", size,
            "a size with a registered family combinator: %s. A size priced by a formula "
            "written at the point of use is a family size nothing enumerates, and TG3.1 "
            "exists because those two must be the same rule"
            % ", ".join("%d (%s)" % (k, v) for k, v in sorted(
                CONFIGURATION_COMBINATORS.items())))


class SurrogateInfeasibleError(InvalidParameterError):
    """The null could not be drawn under the constraint the data itself implies."""

    def __init__(self, minimum_separation: float, bounds: Mapping[str, Tuple[float, float]],
                 n_features: int) -> None:
        super().__init__(
            "surrogate placement", minimum_separation,
            "a minimum separation that %d features can satisfy inside %s. The constraint is "
            "not a parameter to relax: it is the smallest separation the real scenes "
            "contained, and a surrogate that violates it is a configuration the extractor "
            "could not have returned - a null built from shapes the data cannot contain "
            "supports the motif less often than a real arrangement would, and every p-value "
            "computed against it comes out too small"
            % (n_features, ", ".join("%s in [%.1f, %.1f]" % (name, lo, hi)
                                     for name, (lo, hi) in sorted(bounds.items()))),
            max_attempts=MAX_PLACEMENT_ATTEMPTS)


# ------------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Scene:
    """One frame's features, named. The name is what a support count counts."""

    name: str
    features: Tuple[SpectralFeature, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidParameterError(
                "Scene.name", self.name,
                "a name. Support is a count of distinct scenes, and unnamed scenes cannot "
                "be counted distinctly")
        if "|" in self.name:
            raise InvalidParameterError(
                "Scene.name", self.name,
                "a name without '|', which separates the scene from the configuration in a "
                "family member's label")

    def __len__(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class Occurrence:
    """One `k`-subset of one scene, and the signature the matcher measured for it."""

    scene: str
    indices: Tuple[int, ...]
    graph: AttributedGraph

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(int(i) for i in self.indices))
        if len(set(self.indices)) != len(self.indices):
            raise InvalidParameterError(
                "Occurrence.indices", list(self.indices),
                "distinct feature indices within one configuration")
        if list(self.indices) != sorted(self.indices):
            raise InvalidParameterError(
                "Occurrence.indices", list(self.indices),
                "indices in ascending order, so one subset has one label. The same three "
                "features under two orderings would be two members of a priced family and "
                "one configuration in the data")

    @property
    def label(self) -> str:
        """The family member this occurrence is, in the specification's own vocabulary."""
        return "%s|%s" % (self.scene, "-".join(str(i) for i in self.indices))

    def describe(self) -> Dict[str, Any]:
        return {"scene": self.scene, "indices": list(self.indices), "label": self.label,
                "n_nodes": self.graph.n_nodes}


@dataclass(frozen=True)
class MotifCandidate:
    """An exemplar, and every occurrence that matched *it*.

    Not a cluster: membership is decided against the exemplar alone. See the module
    docstring for why, and `MiningResult.intransitive_pairs` for how often it mattered.
    """

    exemplar: Occurrence
    occurrences: Tuple[Occurrence, ...]
    n_examined: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrences", tuple(self.occurrences))
        if self.exemplar.label not in {o.label for o in self.occurrences}:
            raise InvalidParameterError(
                "MotifCandidate.occurrences", self.exemplar.label,
                "a member list containing the exemplar. A configuration that does not match "
                "itself is a tolerance below the noise, not a motif")
        if self.n_examined < len(self.occurrences):
            raise InvalidParameterError(
                "MotifCandidate.n_examined", self.n_examined,
                "at least as many configurations examined as matched, and %d matched"
                % len(self.occurrences))

    @property
    def label(self) -> str:
        return self.exemplar.label

    @property
    def scenes(self) -> Tuple[str, ...]:
        return tuple(sorted({o.scene for o in self.occurrences}))

    @property
    def support(self) -> int:
        """Distinct scenes, not occurrences.

        Two overlapping subsets of one scene are very nearly the same observation, and a
        support that counted both would grow fastest exactly where the evidence is weakest.
        """
        return len(self.scenes)

    @property
    def specificity(self) -> float:
        """The fraction of examined configurations this exemplar does *not* match.

        Description, not a threshold. A motif that matches everything is priced by the null
        rather than by a cutoff written here: it matches everything in the surrogates too,
        so its p-value is about 1 - which `test_motif.py` measures with `always_matches`.
        """
        return 1.0 - len(self.occurrences) / float(self.n_examined)

    def describe(self) -> Dict[str, Any]:
        return {"label": self.label, "support": self.support,
                "n_occurrences": len(self.occurrences), "scenes": list(self.scenes),
                "specificity": self.specificity,
                "occurrences": [o.label for o in self.occurrences]}


@dataclass(frozen=True)
class MiningResult:
    """What the generate stage produced - candidates, and explicitly no claims."""

    scenes: Tuple[str, ...]
    size: int
    matcher: str
    tolerance: float
    specification: SearchSpecification
    candidates: Tuple[MotifCandidate, ...]
    occurrences: Tuple[Occurrence, ...]
    intransitive_pairs: int
    affordable_here: bool

    @property
    def n_examined(self) -> int:
        return len(self.occurrences)

    @property
    def ranked(self) -> Tuple[MotifCandidate, ...]:
        """Candidates by support, then by *fewest* occurrences, then by label.

        The second key is the one that is easy to get backwards, and getting it backwards
        is what puts a promiscuous shape at the top: support is capped at the number of
        scenes and saturates, so the ties at the cap are broken by how many configurations
        the exemplar matched - and a shape matching three triples per scene is a looser
        shape than one matching exactly one, not a stronger finding. Label last, so the
        ranking is total and does not change with the enumeration order.
        """
        return tuple(sorted(self.candidates,
                            key=lambda c: (-c.support, len(c.occurrences), c.label)))

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": "motif-mining/v1",
            "scenes": list(self.scenes), "size": self.size, "matcher": self.matcher,
            "tolerance": self.tolerance,
            "generate_family_size": self.specification.family_size,
            "generate_sha256": self.specification.fingerprint(),
            "n_examined": self.n_examined,
            "n_candidates": len(self.candidates),
            "intransitive_pairs": self.intransitive_pairs,
            "affordable_in_one_stage": self.affordable_here,
            "top": [c.describe() for c in self.ranked[:8]],
            "claim_boundary": (
                "Support here is descriptive. Every exemplar is one of the occurrences it "
                "is counted among, so its support begins at one by construction, and it "
                "was ranked highly because its support was high - both are selection. The "
                "only output of this stage is a confirmatory family to freeze."),
        }


@dataclass(frozen=True)
class MotifEvidence:
    """One frozen motif, tested once on a partition it was not mined from."""

    label: str
    support: int
    n_scenes: int
    n_surrogates: int
    p_value: float
    surrogate_support: Tuple[int, ...] = dc_field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surrogate_support",
                           tuple(int(v) for v in self.surrogate_support))
        if self.n_surrogates < 1:
            raise InvalidParameterError(
                "MotifEvidence.n_surrogates", self.n_surrogates,
                "at least one surrogate. A support with no null beside it is a count")
        if not 0.0 < self.p_value <= 1.0:
            raise InvalidParameterError(
                "MotifEvidence.p_value", self.p_value,
                "a p-value in (0, 1]. The floor is 1/(n+1) and is never zero, because a "
                "finite ensemble cannot exclude a possibility it did not draw")

    @property
    def power_floor(self) -> float:
        """The smallest p-value this ensemble could return, whatever the data did."""
        return 1.0 / (self.n_surrogates + 1.0)

    def vacuous(self, alpha: float) -> bool:
        """TG2.4's rule: a test that could not have rejected confers nothing."""
        return self.power_floor > alpha

    def describe(self) -> Dict[str, Any]:
        return {"label": self.label, "support": self.support, "n_scenes": self.n_scenes,
                "p_value": self.p_value, "n_surrogates": self.n_surrogates,
                "power_floor": self.power_floor}


# --------------------------------------------------------------------------- the family


def _prepare(scenes: Sequence[Scene]) -> Tuple[Scene, ...]:
    prepared = tuple(scenes)
    if len(prepared) < 2:
        raise InvalidParameterError(
            "scenes", len(prepared),
            "at least two scenes. Recurrence is a statement about more than one scene, and "
            "a motif found twice inside a single frame is a statement about that frame")
    names = [s.name for s in prepared]
    if len(set(names)) != len(names):
        raise InvalidParameterError(
            "scenes", names,
            "distinct scene names. Support counts scenes, and two scenes sharing a name "
            "count once however different they are")
    sizes = {len(s) for s in prepared}
    if len(sizes) != 1:
        raise InvalidParameterError(
            "scenes", sorted(sizes),
            "the same number of features in every scene. The declared family is scenes "
            "times C(n, k), and scenes of different n have no single such number - a family "
            "priced at the mean would be a price for a search nobody ran")
    return prepared


def motif_search_specification(scenes: Sequence[Scene], *, size: int, n_surrogates: int,
                               alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli",
                               study_id: str = "") -> SearchSpecification:
    """The family a mining pass over these scenes would search, priced before it runs.

    Two terms, multiplied: which scene, and which `size`-subset of its features. That is
    the search exactly - every subset of every scene is a chance for a repeat to look
    surprising - and the number it produces is the reason the generate/confirm split is not
    optional here.
    """
    prepared = _prepare(scenes)
    if size not in CONFIGURATION_COMBINATORS:
        raise MotifSizeUnavailableError(size)
    n_features = len(prepared[0])
    if size > n_features:
        raise InvalidParameterError(
            "size", size,
            "a configuration no larger than the %d features each scene holds" % n_features)
    scene_axis = SearchAxis("scene", tuple(s.name for s in prepared))
    feature_axis = SearchAxis("feature", tuple(range(n_features)))
    return SearchSpecification(
        terms=(SearchTerm("product", (scene_axis,)),
               SearchTerm(CONFIGURATION_COMBINATORS[size], (feature_axis,))),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}|" + "-".join("{%d}" % (i + 1) for i in range(size)),
        study_id=study_id,
        notes={"stage": "generate", "size": size,
               "reading": ("every %d-subset of every scene, which is what a mining pass "
                           "examines and therefore what it must be priced at" % size)})


# ----------------------------------------------------------------------- the occurrences


def enumerate_occurrences(scenes: Sequence[Scene], *, size: int,
                          matcher: str = DEFAULT_MATCHER,
                          context: Optional[RelationContext] = None) -> Tuple[Occurrence, ...]:
    """Every `size`-subset of every scene, measured by one matcher.

    The enumeration order is `itertools.combinations` over each scene in turn, which is the
    order `motif_search_specification` prices, and `mine` checks the two label sets against
    each other rather than trusting that they agree.
    """
    prepared = _prepare(scenes)
    context = context or RelationContext()
    found: List[Occurrence] = []
    for scene in prepared:
        for indices in itertools.combinations(range(len(scene)), size):
            graph = build_signature(matcher, [scene.features[i] for i in indices],
                                    context=context)
            found.append(Occurrence(scene.name, indices, graph))
    return tuple(found)


def signatures_match(left: AttributedGraph, right: AttributedGraph,
                     tolerance: float) -> bool:
    """Do these two configurations agree to within the tolerance, under some correspondence?

    `best_deviation` rather than `AttributedGraph.matches` because mining needs the number
    as well as the verdict - the same comparison, minimised over the same `k!`
    correspondences, with the deviation kept.
    """
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise InvalidParameterError(
            "tolerance", tolerance,
            "a finite non-negative tolerance, measured from replicates by TG3.4's "
            "`calibrate_match_tolerance` rather than chosen here")
    deviation, _ = best_deviation(left, right)
    return deviation <= tolerance


def candidates_from(occurrences: Sequence[Occurrence], *,
                    tolerance: float) -> Tuple[MotifCandidate, ...]:
    """One candidate per occurrence: the star of everything that matches it.

    Every occurrence is an exemplar, so the ranking is over the whole examined family and
    not over an arbitrary subset of it. That is `O(m^2)` comparisons and it is the honest
    cost of not having picked the exemplars in advance.
    """
    prepared = tuple(occurrences)
    if not prepared:
        raise InvalidParameterError(
            "occurrences", 0, "at least one examined configuration")
    matched: List[List[Occurrence]] = [[] for _ in prepared]
    for i, left in enumerate(prepared):
        matched[i].append(left)
        for j in range(i + 1, len(prepared)):
            if signatures_match(left.graph, prepared[j].graph, tolerance):
                matched[i].append(prepared[j])
                matched[j].append(left)
    return tuple(MotifCandidate(exemplar=prepared[i], occurrences=tuple(members),
                                n_examined=len(prepared))
                 for i, members in enumerate(matched))


def count_intransitive(candidates: Sequence[MotifCandidate]) -> int:
    """How many pairs would have been merged by a chain but are not the same shape.

    A matching relation with a tolerance is reflexive and symmetric but not transitive, and
    single-linkage clustering assumes it is. This counts the pairs where that assumption is
    false: both match a common exemplar, and do not match each other.
    """
    by_label = {c.label: c for c in candidates}
    broken = set()
    for candidate in candidates:
        members = candidate.occurrences
        for left, right in itertools.combinations(members, 2):
            partner = by_label.get(left.label)
            if partner is None:
                continue
            if right.label not in {o.label for o in partner.occurrences}:
                broken.add(tuple(sorted((left.label, right.label))))
    return len(broken)


def mine(scenes: Sequence[Scene], *, size: int, tolerance: float, n_surrogates: int,
         matcher: str = DEFAULT_MATCHER, alpha: float = 0.05,
         correction: str = "benjamini_yekutieli", study_id: str = "",
         context: Optional[RelationContext] = None) -> MiningResult:
    """Mine one partition for repeated configurations. Produces candidates, never findings.

    The specification is built and priced first, and its label set is compared against the
    labels the enumeration actually emits - in order - because a family of 1,760 agreeing
    with an enumeration of 1,760 for two different reasons would pass a count check and
    describe a different search.

    Affordability is *reported*, not enforced. `SearchSpecification.declare()` would refuse
    this family and be right to if this were a one-stage result; what happens instead is
    that the refusal it would have raised is recorded as the reason the confirmatory stage
    exists, and `freeze_motifs` applies TG3.1's gate at the frozen size where it can be paid.
    """
    prepared = _prepare(scenes)
    specification = motif_search_specification(
        prepared, size=size, n_surrogates=n_surrogates, alpha=alpha,
        correction=correction, study_id=study_id)
    occurrences = enumerate_occurrences(prepared, size=size, matcher=matcher,
                                        context=context)
    declared = specification.labels()
    emitted = tuple(o.label for o in occurrences)
    if declared != emitted:
        disagreement = next((pair for pair in zip(declared, emitted) if pair[0] != pair[1]),
                            (len(declared), len(emitted)))
        raise InvalidParameterError(
            "enumeration", disagreement,
            "an enumeration matching the declared family member for member, in order. The "
            "family was priced at %d and the pass examined %d; a search whose members are "
            "not the ones that were priced has an unknown correction unit"
            % (len(declared), len(emitted)))
    candidates = candidates_from(occurrences, tolerance=tolerance)
    return MiningResult(
        scenes=tuple(s.name for s in prepared), size=size, matcher=matcher,
        tolerance=float(tolerance), specification=specification,
        candidates=candidates, occurrences=occurrences,
        intransitive_pairs=count_intransitive(candidates),
        affordable_here=bool(specification.account().affordable))


# ---------------------------------------------------------------------------- the control


#: What a constant signature calls every edge.
CONSTANT = "constant"


def always_matches(features: Sequence[SpectralFeature],
                   context: RelationContext) -> AttributedGraph:
    """A matcher whose signature does not depend on the features. The promiscuity control.

    Public, and deliberately not in TG3.4's `MATCHERS` - but for the opposite reason to
    `scale_normalised`, which is unregistered because its declaration cannot be
    demonstrated. This one's declaration is demonstrable and true: it is exactly invariant
    to rotation, translation and rescaling, because it is invariant to everything.
    `test_motif.py` registers it temporarily and shows that `audit_declared_invariance`
    calls it honest.

    That is the point of it. Invariance is necessary for recognising the same configuration
    in two scenes and it is nowhere near sufficient, and a gate built on invariance alone
    cannot tell this apart from a matcher that measures something. What separates them is
    the null: this one supports every motif in every surrogate scene as readily as in the
    real ones, so its p-value goes to 1 and nothing it proposes is ever confirmed.
    """
    del context
    count = len(features)
    edges = {(left, right): {CONSTANT: RelationValue(
        CONSTANT, 1.0, None, "a constant: this matcher measures nothing")}
        for left, right in itertools.permutations(range(count), 2)}
    return AttributedGraph(
        attributes=tuple({CONSTANT: 1.0} for _ in range(count)),
        carried=tuple(carried_record(f) for f in features),
        edges=edges, relations=(CONSTANT,))


# ------------------------------------------------------------------------- the surrogate


def _spatial_axes(feature: SpectralFeature) -> Tuple[str, ...]:
    return tuple(axis.name for axis in feature.location.axes if axis.role == "space")


def _bounds(scenes: Sequence[Scene]) -> Dict[str, Tuple[float, float]]:
    """Where features were seen, per spatial axis - the region the null draws from.

    The observed extent rather than the grid: a null that scatters features across regions
    the data never populated is answering a question about the grid, not about the scenes.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    for scene in scenes:
        for feature in scene.features:
            for name in _spatial_axes(feature):
                value = float(feature.location.coords[name])
                lo, hi = bounds.get(name, (value, value))
                bounds[name] = (min(lo, value), max(hi, value))
    if not bounds:
        raise InvalidParameterError(
            "scenes", "(no spatial axes)",
            "features located on at least one spatial axis. A surrogate repositions "
            "features, and features with no position cannot be repositioned")
    return bounds


def observed_minimum_separation(scenes: Sequence[Scene],
                                context: Optional[RelationContext] = None) -> float:
    """The closest two features ever came, across every scene.

    This is the constraint the surrogate has to respect, and it is read off the data rather
    than declared, because it stands for something physical the data already demonstrates:
    the extractor resolved two features this close and no closer.
    """
    context = context or RelationContext()
    closest = math.inf
    for scene in scenes:
        for left, right in itertools.combinations(scene.features, 2):
            quantity = left.separation_to(right, periods=context.periods)
            closest = min(closest, float(quantity.value))
    if not math.isfinite(closest):
        raise InvalidParameterError(
            "scenes", "(no pairs)",
            "at least one scene holding two features, so a minimum separation exists")
    return closest


def surrogate_scene(scene: Scene, *, rng: np.random.Generator,
                    bounds: Mapping[str, Tuple[float, float]],
                    minimum_separation: float,
                    context: Optional[RelationContext] = None) -> Scene:
    """The same features, the same count, positions drawn at random - and no closer than real.

    Everything except position is carried over unchanged: magnitude, scale, orientation,
    provenance. The null is about *arrangement*, and a surrogate that also randomised the
    features' properties would be testing a different hypothesis while wearing this one's
    name.
    """
    context = context or RelationContext()
    placed: List[SpectralFeature] = []
    for feature in scene.features:
        axes = _spatial_axes(feature)
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            coords = dict(feature.location.coords)
            for name in axes:
                lo, hi = bounds[name]
                coords[name] = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
            moved = dc_replace(feature, location=FeatureLocation(
                coords=coords, axes=feature.location.axes,
                uncertainty=feature.location.uncertainty))
            if all(float(moved.separation_to(other, periods=context.periods).value)
                   >= minimum_separation for other in placed):
                placed.append(moved)
                break
        else:
            raise SurrogateInfeasibleError(minimum_separation, bounds, len(scene.features))
    return Scene(scene.name, tuple(placed))


def support_of(exemplar: AttributedGraph, scenes: Sequence[Scene], *, size: int,
               tolerance: float, matcher: str = DEFAULT_MATCHER,
               context: Optional[RelationContext] = None) -> int:
    """In how many of these scenes does this exemplar occur at least once?

    Stops at the first match within a scene, because support counts scenes: a second
    occurrence in a scene already counted adds nothing to the statistic and the null must
    be computed under the same rule as the observation.
    """
    context = context or RelationContext()
    found = 0
    for scene in scenes:
        for indices in itertools.combinations(range(len(scene)), size):
            graph = build_signature(matcher, [scene.features[i] for i in indices],
                                    context=context)
            if signatures_match(exemplar, graph, tolerance):
                found += 1
                break
    return found


def motif_p_value(exemplar: AttributedGraph, scenes: Sequence[Scene], *, size: int,
                  tolerance: float, n_surrogates: int, seed: int,
                  matcher: str = DEFAULT_MATCHER,
                  minimum_separation: Optional[float] = None,
                  context: Optional[RelationContext] = None) -> MotifEvidence:
    """How often does an arrangement drawn at random support this motif as well?

    The `(1 + count) / (1 + n)` form, for the reason it is used everywhere else in this
    tree: a finite ensemble cannot exclude what it did not draw, so the floor is `1/(n+1)`
    and never zero.
    """
    prepared = _prepare(scenes)
    context = context or RelationContext()
    if minimum_separation is None:
        minimum_separation = observed_minimum_separation(prepared, context)
    observed = support_of(exemplar, prepared, size=size, tolerance=tolerance,
                          matcher=matcher, context=context)
    bounds = _bounds(prepared)
    rng = np.random.default_rng(seed)
    null: List[int] = []
    for _ in range(n_surrogates):
        drawn = [surrogate_scene(scene, rng=rng, bounds=bounds,
                                 minimum_separation=minimum_separation, context=context)
                 for scene in prepared]
        null.append(support_of(exemplar, drawn, size=size, tolerance=tolerance,
                               matcher=matcher, context=context))
    at_least = sum(1 for value in null if value >= observed)
    return MotifEvidence(
        label="(exemplar)", support=observed, n_scenes=len(prepared),
        n_surrogates=n_surrogates,
        p_value=(1.0 + at_least) / (1.0 + n_surrogates),
        surrogate_support=tuple(null))


# -------------------------------------------------------------- generate, freeze, confirm


def report_motif_generation(result: MiningResult, *, train: PartitionIdentity,
                            n_candidates: Optional[int] = None) -> Dict[str, Any]:
    """The generation receipt, through TG3.2's own checker.

    The candidate labels are the exemplars' labels, which are members of the generate
    family by construction - and `report_generation` verifies that rather than taking it on
    trust, which is what catches an exemplar that came from somewhere else.
    """
    chosen = choose_candidates(result, n_candidates=n_candidates)
    report = report_generation(result.specification, train=train,
                               candidates=[c.label for c in chosen])
    report["mining"] = result.describe()
    return report


def affordable_candidate_count(n_surrogates: int, alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli") -> int:
    """How many motifs this ensemble can still reject one of, after correction (TG3.1)."""
    return int(max_affordable_family(n_surrogates, alpha=alpha, method=correction))


def deduplicate_candidates(candidates: Sequence[MotifCandidate], *,
                           tolerance: float) -> Tuple[MotifCandidate, ...]:
    """One representative per distinct shape, keeping the highest-ranked of each.

    Every occurrence is an exemplar, so a motif that genuinely recurs in six scenes enters
    the ranking six times - once under each scene's label - and freezing all six would
    spend six slots of a correction unit on one hypothesis. Greedy over the ranking, and
    membership decided against the representative rather than against the group, for the
    same non-transitivity reason a candidate is a star: a chain of near-misses would
    otherwise absorb shapes the representative does not match.
    """
    kept: List[MotifCandidate] = []
    for candidate in candidates:
        if any(signatures_match(other.exemplar.graph, candidate.exemplar.graph, tolerance)
               for other in kept):
            continue
        kept.append(candidate)
    return tuple(kept)


def choose_candidates(result: MiningResult,
                      n_candidates: Optional[int] = None) -> Tuple[MotifCandidate, ...]:
    """The distinct top-ranked candidates, capped at what the ensemble can pay for.

    Three rules, all forced rather than preferred. Only the shapes that recurred *most* on
    train, because a family topped up to the ceiling with candidates the mining pass did
    not favour spends correction power on members nobody proposed - it moved the planted
    motif's corrected q from 0.005 to 0.042 in this tree's own benchmark, which is the
    difference between a clear result and a marginal one. Distinct, because the same shape
    appears in the ranking once per scene it occurs in and six names for one hypothesis is
    a correction unit of six paid for one test. And capped at TG3.1's ceiling, because
    freezing more motifs than the confirmatory ensemble can reject one of produces a family
    whose correction cannot be paid, and the confirmation would then be a stage that could
    not have rejected anything.

    Selecting on train support is selection, and it is the selection the generate stage
    exists to perform: it happens before the seal, on the partition that was mined, and
    nothing it produces is a claim.
    """
    ceiling = affordable_candidate_count(
        result.specification.n_surrogates, alpha=result.specification.alpha,
        correction=result.specification.correction)
    wanted = ceiling if n_candidates is None else int(n_candidates)
    if wanted < 1:
        raise InvalidParameterError(
            "n_candidates", wanted,
            "at least one candidate to carry forward. A confirmatory family of none is a "
            "study that cannot report anything, which is not the same as a null result")
    if wanted > ceiling:
        raise InvalidParameterError(
            "n_candidates", wanted,
            "at most the %d members %d surrogates can reject one of after %s correction at "
            "alpha %g. Freezing more is freezing a family the confirmation cannot pay for"
            % (ceiling, result.specification.n_surrogates,
               result.specification.correction, result.specification.alpha))
    ranked = result.ranked
    if n_candidates is None:
        best = ranked[0].support
        ranked = tuple(c for c in ranked if c.support == best)
    distinct = deduplicate_candidates(ranked, tolerance=result.tolerance)
    return distinct[:min(wanted, len(distinct))]


def confirmatory_specification(chosen: Sequence[MotifCandidate], *, n_surrogates: int,
                               alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli",
                               study_id: str = "",
                               notes: Optional[Mapping[str, Any]] = None
                               ) -> SearchSpecification:
    """The frozen family: these motifs, named by the exemplar that defines each one.

    A motif *is* its exemplar, so the label that identifies it on the held-out partition is
    the label it had on train. That is what makes the confirmatory family a subset of the
    generated one - and it is also what makes redefinition visible, because a motif tested
    under a label whose exemplar has changed is a different signature under the same name.

    `notes` are merged *beside* the two this function writes, never over them, and they are
    part of the specification's fingerprint - so anything recorded there is sealed with the
    family rather than stored next to it. TG11.4 needs that: a confirmatory run driven over
    HTTP has to read every setting back out of the seal, and a setting kept in a file beside
    the seal is a setting the seal does not bind (standard E12 - the parameter generalises
    the function beside its existing behaviour and no caller that omits it sees a change).
    """
    labels = [c.label for c in chosen]
    if len(set(labels)) != len(labels):
        raise InvalidParameterError(
            "chosen", labels, "distinct exemplars. One motif frozen twice is a correction "
                              "unit inflated by a duplicate")
    return SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("motif", tuple(labels)),)),),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}", study_id=study_id,
        notes=dict(dict(notes or {}),
                   stage="confirm",
                   reading=("each member is one motif, defined by the exemplar named in its "
                            "label and tested on a partition it was not mined from")))


def freeze_motifs(result: MiningResult, *, held_out: PartitionIdentity, sealed_at: str,
                  n_surrogates: int, chosen: Optional[Sequence[MotifCandidate]] = None,
                  n_candidates: Optional[int] = None,
                  ledger: Optional[HeldOutLedger] = None,
                  study_id: str = "",
                  notes: Optional[Mapping[str, Any]] = None
                  ) -> Tuple[Seal, Tuple[MotifCandidate, ...]]:
    """Freeze the confirmatory family before the held-out partition is opened.

    Returns the seal *and* the candidates it froze, in the seal's own label order, because
    the confirmation needs the exemplar graphs and the seal carries only their names. A
    caller that reconstructs the graphs from somewhere else has redefined the motif, and
    `confirm_motifs` checks the labels it is given against the labels the seal froze.
    """
    picked = tuple(chosen) if chosen is not None else choose_candidates(
        result, n_candidates=n_candidates)
    confirm = confirmatory_specification(
        picked, n_surrogates=n_surrogates, alpha=result.specification.alpha,
        correction=result.specification.correction, study_id=study_id, notes=notes)
    seal = freeze_confirmatory_family(
        result.specification, confirm, held_out=held_out, sealed_at=sealed_at,
        study_id=study_id, ledger=ledger)
    order = {label: i for i, label in enumerate(seal.confirm_labels)}
    return seal, tuple(sorted(picked, key=lambda c: order[c.label]))


def confirm_motifs(seal: Seal, chosen: Sequence[MotifCandidate], *,
                   scenes: Sequence[Scene], held_out: PartitionIdentity,
                   ledger: HeldOutLedger, opened_at: str, size: int, tolerance: float,
                   seed: int, matcher: str = DEFAULT_MATCHER,
                   n_surrogates: Optional[int] = None,
                   context: Optional[RelationContext] = None) -> Dict[str, Any]:
    """Test the frozen motifs on the held-out scenes, once, and correct at the frozen size.

    Each motif gets its own surrogate ensemble under the same seed stream, and the receipt
    carries the supports beside the p-values so a reader can see that a motif rejected at
    q < alpha was rejected for occurring more often than chance and not merely for occurring.
    """
    frozen = list(seal.confirm_labels)
    supplied = [c.label for c in chosen]
    if supplied != frozen:
        raise InvalidParameterError(
            "chosen", supplied[:8],
            "the exemplars the seal froze, in the order it froze them: %s. A motif tested "
            "under a frozen label whose signature came from elsewhere is a redefinition, "
            "and the seal names the definition it was sealed with" % frozen[:8])
    ensemble = int(n_surrogates if n_surrogates is not None
                   else seal.confirm.get("n_surrogates", 0))
    prepared = _prepare(scenes)
    minimum = observed_minimum_separation(prepared, context)
    evidence: List[MotifEvidence] = []
    for offset, candidate in enumerate(chosen):
        measured = motif_p_value(
            candidate.exemplar.graph, prepared, size=size, tolerance=tolerance,
            n_surrogates=ensemble, seed=seed + offset, matcher=matcher,
            minimum_separation=minimum, context=context)
        evidence.append(dc_replace(measured, label=candidate.label))
    receipt = confirm_on_held_out(
        seal, p_values={e.label: e.p_value for e in evidence}, held_out=held_out,
        ledger=ledger, opened_at=opened_at)
    alpha = float(seal.confirm.get("alpha", 0.05))
    receipt["motifs"] = [e.describe() for e in evidence]
    receipt["supports"] = [e.support for e in evidence]
    receipt["n_scenes"] = len(prepared)
    receipt["minimum_separation"] = minimum
    receipt["vacuous"] = [e.label for e in evidence if e.vacuous(alpha)]
    return receipt


__all__ = [
    "CONFIGURATION_COMBINATORS", "DEFAULT_MATCHER", "MAX_PLACEMENT_ATTEMPTS",
    "MotifSizeUnavailableError", "SurrogateInfeasibleError",
    "Scene", "Occurrence", "MotifCandidate", "MiningResult", "MotifEvidence",
    "motif_search_specification", "enumerate_occurrences", "signatures_match",
    "candidates_from", "count_intransitive", "mine", "deduplicate_candidates",
    "CONSTANT", "always_matches",
    "observed_minimum_separation", "surrogate_scene", "support_of", "motif_p_value",
    "report_motif_generation", "affordable_candidate_count", "choose_candidates",
    "confirmatory_specification", "freeze_motifs", "confirm_motifs",
]
