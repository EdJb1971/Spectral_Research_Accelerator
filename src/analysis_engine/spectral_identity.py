"""The identity step at record scale, without changing what a pattern is (`T4E.5`, fixes D96).

T4E.3 defines a pattern by clustering signed configurations: complete linkage under a calibrated
tolerance, with the extra condition that a cluster's radius from its own centroid also stays
inside that tolerance. That definition is the scientific content and this module does not touch
it. What it replaces is an implementation that enumerates every pair of surviving clusters and
recomputes every cross-distance from scratch on every merge -- measured at about O(n^3) on real
signatures, 347 s at n=200, against a training record presenting some 843,000 configurations
(D96).

**Every function here returns exactly what `cluster_signatures` returns.** Not a comparable
clustering, not one that agrees to within a tolerance: the same patterns, with the same members,
the same centroids and the same radii. That is the only honest way to make a scientific component
faster, and the suite holds it to that by running both on the same inputs and comparing the
receipts, rather than by comparing summary statistics. Where the exact answer is out of reach
this module says so and refuses; it does not quietly return an approximate one.

Four things make it possible.

**1. A cluster can never span two components of the tolerance graph.** Complete linkage requires
*every* pair inside a cluster to be within the tolerance, so two configurations further apart
than that are never in one cluster, and neither are two configurations with no chain of
within-tolerance neighbours between them. So the connected components of the graph
`{(i,j) : d(i,j) <= tolerance}` partition the problem exactly, and each component can be
clustered on its own. This needs no assumption about the metric at all -- not even the triangle
inequality, which this metric is not known to satisfy.

**2. The metric's own algebra gives an exact box in log coordinates.** The relative difference
this metric uses on every positive block is

    delta(a, b) = |a - b| / ((a + b) / 2) = 2 * tanh(|log a - log b| / 2)

which is *monotone* in the absolute difference of the logarithms. So `delta <= tau` holds exactly
when `|log a - log b| <= 2 * artanh(tau / 2)`, with no constraint at all once `tau >= 2`, since
the relative difference of two positive numbers never reaches 2. And because the metric is a
weighted root-mean-square over blocks, `d <= T` forces every single component to satisfy
`delta_i <= T * sqrt(n_block * sum_of_weights / weight_of_block)`. Those two facts together turn
"which configurations could possibly be within tolerance" into an axis-aligned box query in log
space -- a **superset** of the true neighbours, containing every one of them, obtained without
computing a single distance.

**3. The superset is then verified exactly.** Every candidate the box returns is measured with
the real metric and dropped unless it is genuinely within tolerance. The box is a filter, never
an answer, so a loose bound costs time and can never cost correctness.

**4. Within a component, the greedy is the same greedy, done once.** Complete linkage admits an
exact incremental update -- the cross-distance from a merged cluster to any third is the maximum
of the two it was merged from -- so the cross-distances are computed once and maintained rather
than rebuilt. The centroid-radius condition admits no such update, because a centroid is a medoid
search over the merged members; so instead of testing it on every candidate pair, candidates are
taken in the order the original takes them and the first that passes wins. That is the same
winner the original picks, reached without testing the ones it would have discarded anyway.

What this does **not** fix: a component that is itself enormous. The decomposition is exact, not
magic, and if the tolerance is wide enough that most of a record falls into one component then
that component costs what it costs. `plan_clustering` measures the components before any
clustering happens and reports what the work will be, so an infeasible run is refused in advance
with the number that makes it infeasible, rather than discovered hours later.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.spectral_clustering import (
    ConstellationPattern, PatternCatalogue, SignatureMetric, SignaturePoint,
    SignatureTolerance, _centroid, _member_key, _points,
)
from src.core.errors import InvalidParameterError

IDENTITY_SCHEMA = "spectral-identity-plan/v1"

#: The relative difference of two positive numbers is `2 * tanh(|dlog| / 2)`, which approaches 2
#: and never reaches it. A per-component bound at or above this constrains nothing, and the box
#: side for that component is infinite rather than very large.
RELATIVE_DIFFERENCE_SUPREMUM = 2.0

#: Bearings are compared as an absolute folded-angle difference divided by this, so their box
#: side is linear rather than logarithmic. Kept as a named constant because it is the one block
#: whose geometry differs, and a reader should not have to infer that from an arithmetic.
BEARING_SCALE_DEGREES = 90.0

EQUIVALENCE_NOTE = (
    "This returns exactly what `cluster_signatures` returns: the same patterns with the same "
    "members, centroids and radii, not a comparable clustering. The decomposition is exact "
    "because complete linkage requires every pair inside a cluster to be within the tolerance, "
    "so no cluster can span two components of the tolerance graph; and the box filter is exact "
    "because every candidate it admits is afterwards measured with the real metric. A loose "
    "bound costs time and cannot cost correctness.")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidParameterError(name, value, "a positive whole number")
    return int(value)


# --------------------------------------------------------------------------------------------
# the exact box the metric's own algebra permits
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentBound:
    """The widest one component of one block may differ and still allow `d <= tolerance`."""

    block: str
    #: The largest per-component value of this block's own difference measure.
    difference: float
    #: The box half-width in the coordinate the index is built on: log units for a relative
    #: block, degrees for bearings. Infinite when the bound constrains nothing.
    half_width: float
    basis: str

    def as_record(self) -> Dict[str, Any]:
        return {"block": self.block, "max_component_difference": self.difference,
                "box_half_width": (None if math.isinf(self.half_width) else self.half_width),
                "constrains": math.isfinite(self.half_width), "basis": self.basis}


def component_bounds(point: SignaturePoint, metric: SignatureMetric,
                     tolerance: float) -> Dict[str, ComponentBound]:
    """How far one component of each block may differ, given the whole distance is bounded.

    The metric reduces to `sqrt(sum_b w_b * mean_i delta_bi^2 / sum_b w_b)`. Requiring that to
    be at most `T` requires, for every block `b` and every component `i` of it,

        w_b * delta_bi^2 / n_b <= T^2 * sum_of_weights

    because every other term in the sum is non-negative. Rearranged, that is the bound below.
    It is necessary and not sufficient, which is exactly what a filter needs to be.
    """
    if not isinstance(point, SignaturePoint):
        raise InvalidParameterError("point", type(point).__name__, "a SignaturePoint")
    if not isinstance(metric, SignatureMetric):
        raise InvalidParameterError("metric", type(metric).__name__, "a SignatureMetric")
    limit = float(tolerance)
    if not math.isfinite(limit) or limit < 0.0:
        raise InvalidParameterError("tolerance", tolerance, "a finite non-negative radius")

    weights = metric.weights.as_mapping()
    blocks = {name: values for name, values in point.blocks().items() if values}
    active = {name: weights[name] for name in blocks if weights[name] > 0.0}
    if not active:
        raise InvalidParameterError(
            "metric.weights", weights,
            "a positive weight on at least one block this signature family carries")
    total = sum(active.values())

    bounds: Dict[str, ComponentBound] = {}
    for name, values in blocks.items():
        weight = weights[name]
        if weight <= 0.0:
            # A block the metric ignores constrains nothing, and pretending otherwise would
            # filter out neighbours the metric itself would have admitted.
            bounds[name] = ComponentBound(
                name, float("inf"), float("inf"),
                "this block carries zero weight, so it cannot bound the distance")
            continue
        difference = limit * math.sqrt(len(values) * total / weight)
        if name == "bearings":
            half = difference * BEARING_SCALE_DEGREES
            basis = ("a folded-angle difference divided by %g degrees, so the box side is that "
                     "difference in degrees" % BEARING_SCALE_DEGREES)
        elif difference >= RELATIVE_DIFFERENCE_SUPREMUM:
            half = float("inf")
            basis = ("a relative difference of two positive numbers is below %g, so a bound of "
                     "%.6g excludes nothing" % (RELATIVE_DIFFERENCE_SUPREMUM, difference))
        else:
            half = 2.0 * math.atanh(difference / RELATIVE_DIFFERENCE_SUPREMUM)
            basis = ("the relative difference is 2*tanh(|dlog|/2), so this bound is exactly "
                     "|dlog| <= 2*artanh(%.6g/2)" % difference)
        bounds[name] = ComponentBound(name, difference, half, basis)
    return bounds


def _coordinates(points: Sequence[SignaturePoint]) -> Dict[str, np.ndarray]:
    """Each block as an array in the coordinate its bound is expressed in."""
    stacked: Dict[str, np.ndarray] = {}
    for name in ("geometry", "bearings", "strengths", "scales"):
        rows = [point.blocks()[name] for point in points]
        if not rows or not rows[0]:
            continue
        array = np.asarray(rows, dtype=np.float64)
        stacked[name] = array if name == "bearings" else np.log(array)
    return stacked


def _permutation_coordinates(points: Sequence[SignaturePoint], order: Tuple[int, ...]
                             ) -> Dict[str, np.ndarray]:
    return _coordinates([point.permuted(order) for point in points])


def pairwise_distance_matrix(points: Sequence[SignaturePoint], metric: SignatureMetric,
                             block: int = 1024) -> np.ndarray:
    """Every pairwise distance at once, computed to agree with `metric.distance` exactly.

    The metric is a weighted root-mean-square of per-component differences, minimised over node
    correspondences. Every part of that vectorises: the relative difference is elementwise, the
    weighted reduction is a dot product, and the correspondence minimum is a minimum over the
    handful of permutations a cardinality admits. The suite holds this to the scalar metric
    element by element rather than in aggregate, because a distance that is nearly right would
    put a configuration in nearly the right pattern.
    """
    if not points:
        raise InvalidParameterError("points", 0, "at least one signature point")
    family = points[0].family
    if any(point.family != family for point in points):
        raise InvalidParameterError("points", "mixed", "one signature family")
    weights = metric.weights.as_mapping()
    blocks = {name: values for name, values in points[0].blocks().items() if values}
    active = [name for name in blocks if weights[name] > 0.0]
    if not active:
        raise InvalidParameterError(
            "metric.weights", weights,
            "a positive weight on at least one block this signature family carries")
    total = sum(weights[name] for name in active)
    size = _positive_int("block", block)

    straight = {name: np.asarray([point.blocks()[name] for point in points], dtype=np.float64)
                for name in active}
    orders = tuple(itertools.permutations(range(family.cardinality)))
    permuted = {
        order: {name: np.asarray([point.permuted(order).blocks()[name] for point in points],
                                 dtype=np.float64)
                for name in active}
        for order in orders}

    n = len(points)
    out = np.zeros((n, n), dtype=np.float64)
    for start in range(0, n, size):
        stop = min(start + size, n)
        best = None
        for order in orders:
            accumulated = np.zeros((stop - start, n), dtype=np.float64)
            for name in active:
                left = straight[name][start:stop]
                right = permuted[order][name]
                if name == "bearings":
                    delta = (np.abs(left[:, None, :] - right[None, :, :])
                             / BEARING_SCALE_DEGREES)
                else:
                    denominator = (np.abs(left[:, None, :])
                                   + np.abs(right[None, :, :])) / 2.0
                    difference = np.abs(left[:, None, :] - right[None, :, :])
                    delta = np.divide(difference, denominator,
                                      out=np.zeros_like(difference),
                                      where=denominator != 0.0)
                accumulated += weights[name] * np.mean(delta * delta, axis=2)
            candidate = np.sqrt(accumulated / total)
            best = candidate if best is None else np.minimum(best, candidate)
        out[start:stop, :] = best
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------------------------
# the tolerance graph, exactly
# --------------------------------------------------------------------------------------------


class _Union:
    """Union-find with path halving. Small enough to keep here rather than to depend on."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = {}
        for item in range(len(self._parent)):
            buckets.setdefault(self.find(item), []).append(item)
        return [sorted(group) for group in
                sorted(buckets.values(), key=lambda group: (len(group), group[0]))]


@dataclass(frozen=True)
class NeighbourGraph:
    """Which configurations are close enough to ever share a pattern, and what that cost."""

    components: Tuple[Tuple[int, ...], ...]
    n_points: int
    #: How many times two components were joined -- **not** the number of within-tolerance
    #: pairs. Finding components does not need the edge set, so a pair whose ends are already
    #: in one component is skipped without measuring it, and this count is therefore a spanning
    #: structure: it equals `n_points - n_components` exactly. Calling it an edge count would
    #: invite a reader to divide it by the pairs and call the result a density, which it is not.
    n_unions: int
    n_candidate_pairs: int
    n_exact_distances: int
    bounds: Mapping[str, ComponentBound]

    @property
    def largest_component(self) -> int:
        return max((len(item) for item in self.components), default=0)

    def as_record(self) -> Dict[str, Any]:
        sizes = sorted((len(item) for item in self.components), reverse=True)
        return {
            "n_points": self.n_points,
            "n_components": len(self.components),
            "largest_component": self.largest_component,
            "component_sizes_largest_first": sizes[:16],
            "n_unions_performed": self.n_unions,
            "n_unions_basis": (
                "components joined, which is n_points - n_components and not the number of "
                "within-tolerance pairs: a pair already inside one component is skipped "
                "without being measured, because it cannot change the decomposition"),
            "n_candidate_pairs_the_box_admitted": self.n_candidate_pairs,
            "n_exact_distances_computed": self.n_exact_distances,
            "distances_not_computed": (self.n_points * (self.n_points - 1)) // 2
                                      - self.n_exact_distances,
            "bounds": {name: bound.as_record() for name, bound in self.bounds.items()},
            "exactness": EQUIVALENCE_NOTE,
        }


def neighbour_graph(points: Sequence[SignaturePoint], *, metric: SignatureMetric,
                    tolerance: float, block: int = 4096) -> NeighbourGraph:
    """Every within-tolerance pair, found through the box and then verified exactly.

    The box is a necessary condition derived from the metric, so it admits a superset; every
    admitted pair is then measured with `metric.distance` and kept only if it really is within
    tolerance. `n_exact_distances` against the number of pairs there are is what the box bought,
    and it is published rather than assumed. A pair whose two ends already share a component is
    without being measured, because no distance between them can change the decomposition -- so
    what this returns is the components, and deliberately not the edge set.
    """
    if not points:
        raise InvalidParameterError("points", 0, "at least one signature point to relate")
    family = points[0].family
    if any(point.family != family for point in points):
        raise InvalidParameterError(
            "points", "mixed", "one signature family; a distance between different measured "
                               "quantities is not a number")
    size = _positive_int("block", block)
    limit = float(tolerance)
    bounds = component_bounds(points[0], metric, limit)

    base = _coordinates(points)
    orders = tuple(itertools.permutations(range(family.cardinality)))
    permuted = {order: _permutation_coordinates(points, order) for order in orders}

    union = _Union(len(points))
    unions = candidates = exact = 0
    n = len(points)
    for start in range(0, n, size):
        stop = min(start + size, n)
        # Only the upper triangle: every pair is considered once, from the earlier index.
        for other_start in range(start, n, size):
            other_stop = min(other_start + size, n)
            admissible = np.zeros((stop - start, other_stop - other_start), dtype=bool)
            for order in orders:
                inside = np.ones_like(admissible)
                for name, bound in bounds.items():
                    if math.isinf(bound.half_width) or name not in base:
                        continue
                    left = base[name][start:stop]
                    right = permuted[order][name][other_start:other_stop]
                    delta = np.abs(left[:, None, :] - right[None, :, :])
                    inside &= np.all(delta <= bound.half_width + 1e-12, axis=2)
                    if not inside.any():
                        break
                admissible |= inside
            rows, cols = np.nonzero(admissible)
            for row, col in zip(rows.tolist(), cols.tolist()):
                i, j = start + row, other_start + col
                if i >= j:
                    continue
                candidates += 1
                if union.find(i) == union.find(j):
                    # Already joined, so the exact distance cannot change the components.
                    continue
                exact += 1
                if metric.distance(points[i], points[j]) <= limit:
                    unions += 1
                    union.union(i, j)

    return NeighbourGraph(
        components=tuple(tuple(group) for group in union.groups()), n_points=n,
        n_unions=unions, n_candidate_pairs=candidates, n_exact_distances=exact, bounds=bounds)


# --------------------------------------------------------------------------------------------
# the same greedy, done once
# --------------------------------------------------------------------------------------------


def _cluster_component(indices: Sequence[int], points: Sequence[SignaturePoint],
                       metric: SignatureMetric, tolerance: float) -> List[Tuple[int, ...]]:
    """T4E.3's greedy over one component, as a heap rather than a rescan.

    Three exact changes to the same algorithm.

    **The cross-distances are maintained, not rebuilt.** Complete linkage has an incremental
    identity -- the cross-distance from a merged cluster to a third is the larger of the two it
    came from -- so the pairwise distances are computed once, vectorised, and updated in place.

    **The candidate pairs live in a heap, not in a list rebuilt every iteration.** Rescanning
    every surviving pair on every merge is what makes the original cubic: there are up to n
    merges and each rescan is quadratic. Every admissible pair is pushed once and the minimum
    popped; a pair naming a cluster that has since merged away is stale and discarded on sight.

    **The radius condition is tested on candidates in the order the original considers them.**
    It has no incremental identity, since a centroid is a medoid search over the merged members.
    The original filters every candidate by it and then takes the smallest survivor; popping in
    increasing order and taking the first that passes reaches the same one. A pair that fails
    cannot pass later without one of its clusters changing, and if either changes the pair no
    longer exists -- so discarding a failure is safe.
    """
    local = list(indices)
    clusters: Dict[int, Tuple[int, ...]] = {position: (item,)
                                            for position, item in enumerate(local)}
    size = len(local)
    if size == 1:
        return [clusters[0]]

    cross = pairwise_distance_matrix([points[item] for item in local], metric)

    heap: List[Tuple[float, Tuple[int, ...], int, int]] = []
    rows, cols = np.nonzero(np.triu(cross <= tolerance, k=1))
    for a, b in zip(rows.tolist(), cols.tolist()):
        heapq.heappush(heap, (float(cross[a, b]),
                              tuple(sorted(clusters[a] + clusters[b])), a, b))

    alive = set(range(size))
    while len(alive) > 1 and heap:
        value, merged, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive:
            continue                      # one side has already been merged away
        if tuple(sorted(clusters[a] + clusters[b])) != merged:
            continue                      # this pair has moved on since it was pushed
        candidate_points = [points[item] for item in merged]
        centre = _centroid(candidate_points, metric)
        if max(metric.distance(point, centre) for point in candidate_points) > tolerance:
            continue                      # fails the radius condition, and cannot later pass
        clusters[a] = merged
        alive.discard(b)
        del clusters[b]
        for other in alive:
            if other == a:
                continue
            updated = max(cross[a, other], cross[b, other])
            cross[a, other] = cross[other, a] = updated
            if updated <= tolerance:
                heapq.heappush(heap, (float(updated),
                                      tuple(sorted(clusters[a] + clusters[other])), a, other))
    return [clusters[item] for item in sorted(alive)]


@dataclass(frozen=True)
class ClusteringPlan:
    """What the work will be, measured before any of it is done."""

    graph: NeighbourGraph
    estimated_distance_evaluations: int
    largest_component: int
    feasible: bool
    note: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema": IDENTITY_SCHEMA,
            "graph": self.graph.as_record(),
            "largest_component": self.largest_component,
            "estimated_distance_evaluations": self.estimated_distance_evaluations,
            "feasible": self.feasible,
            "note": self.note,
        }


def plan_clustering(signatures: Sequence[Any], *, metric: SignatureMetric,
                    tolerance: SignatureTolerance,
                    max_component: int = 20000) -> ClusteringPlan:
    """Measure the decomposition before clustering, so an impossible run is refused in advance.

    The decomposition is exact but it is not magic: if the tolerance is wide enough that most of
    a record falls into one component, that component costs what it costs. Knowing that before
    the run rather than after is the difference between a refusal and a wasted afternoon.
    """
    points = _points(sorted(signatures, key=_member_key))
    graph = neighbour_graph(points, metric=metric, tolerance=tolerance.value)
    largest = graph.largest_component
    estimate = sum(len(item) * (len(item) - 1) // 2 for item in graph.components)
    feasible = largest <= _positive_int("max_component", max_component)
    if feasible:
        note = ("The largest component holds %d of %d configurations, so the exact greedy runs "
                "on %d pairs rather than the %d there are."
                % (largest, graph.n_points, estimate,
                   graph.n_points * (graph.n_points - 1) // 2))
    else:
        note = ("The largest component holds %d configurations against a declared ceiling of "
                "%d. The decomposition is exact and this component is genuinely that large: "
                "the tolerance admits a chain of neighbours joining all of them, so no exact "
                "method avoids the cost. Raise the ceiling deliberately, or reduce what is "
                "clustered, but do not read this as a reason to widen the tolerance -- a wider "
                "tolerance makes this component larger, not smaller."
                % (largest, max_component))
    return ClusteringPlan(graph=graph, estimated_distance_evaluations=estimate,
                          largest_component=largest, feasible=feasible, note=note)


def cluster_signatures_scalable(signatures: Sequence[Any], *, metric: SignatureMetric,
                                tolerance: SignatureTolerance,
                                max_component: int = 20000) -> PatternCatalogue:
    """T4E.3's clustering, decomposed exactly. Same patterns, same centroids, same radii.

    Refuses rather than approximates when the decomposition leaves a component too large to
    cluster exactly: an approximate identity returned under the same name as an exact one would
    make every pattern downstream mean something the caller did not ask for.
    """
    ordered = tuple(sorted(signatures, key=_member_key))
    if not ordered:
        raise InvalidParameterError(
            "signatures", 0, "at least one signed constellation to cluster")
    points = _points(ordered)
    if metric.digest != tolerance.metric_digest:
        raise InvalidParameterError(
            "metric", metric.digest,
            "the exact declared metric used to calibrate this tolerance (%s)"
            % tolerance.metric_digest)
    if any(point.family != tolerance.family for point in points):
        raise InvalidParameterError(
            "signatures", [point.family.describe() for point in points],
            "the signature family on which the tolerance was measured (%s)"
            % tolerance.family.describe())

    plan = plan_clustering(ordered, metric=metric, tolerance=tolerance,
                           max_component=max_component)
    if not plan.feasible:
        raise InvalidParameterError(
            "signatures", plan.largest_component,
            "a set whose tolerance graph decomposes into components this can cluster exactly. "
            + plan.note)

    clusters: List[Tuple[int, ...]] = []
    for component in plan.graph.components:
        clusters.extend(_cluster_component(component, points, metric, tolerance.value))

    # The ordering is T4E.3's own -- by centroid vector, then by member key -- and not by
    # index, because pattern_id is part of what a receipt names and must not depend on which
    # component a cluster happened to come out of.
    rows = []
    for indices in clusters:
        member_points = [points[index] for index in indices]
        centre = _centroid(member_points, metric)
        observed = max(metric.distance(point, centre) for point in member_points)
        members = tuple(ordered[index] for index in indices)
        rows.append((centre.vector(), members, centre, observed))
    rows.sort(key=lambda row: (row[0], tuple(_member_key(item) for item in row[1])))
    records = tuple(
        ConstellationPattern(
            pattern_id=index + 1, members=row[1], centroid=row[2],
            tolerance_radius=tolerance.value, observed_radius=row[3])
        for index, row in enumerate(rows))
    return PatternCatalogue(patterns=records, metric=metric, tolerance=tolerance,
                            n_signatures=len(ordered))


__all__ = [
    "IDENTITY_SCHEMA", "EQUIVALENCE_NOTE", "RELATIVE_DIFFERENCE_SUPREMUM",
    "BEARING_SCALE_DEGREES",
    "ComponentBound", "component_bounds", "NeighbourGraph", "neighbour_graph",
    "ClusteringPlan", "plan_clustering", "cluster_signatures_scalable",
    "pairwise_distance_matrix",
]
