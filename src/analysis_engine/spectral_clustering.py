"""T4E.3: approximate constellation matching with a measured radius.

T4E.2 turns one attributed graph into four canonical, comparable blocks: geometry,
bearings, relative strength and scale.  This module does not turn those continuous
measurements back into exact labels.  It declares how much each block contributes,
measures the largest within-configuration distance over replicate pairs, and uses that
measurement as the only admissible clustering radius.

The clustering is deterministic complete-link agglomeration.  Complete link is deliberate:
single link can join A to C through B even when A and C are farther apart than the measured
noise floor.  Every resulting member pair, and every member-to-centroid distance, must remain
inside the calibrated radius.  A pattern consequently has a readable centroid, the calibrated
tolerance radius, and the smaller observed radius of the members actually assigned to it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_invariance import ConstellationSignature
from src.core.errors import InvalidParameterError
from src.core.invariance import MatchTolerance


CLUSTER_SCHEMA = "spectral-constellation-clusters/v1"
METRIC_SCHEMA = "spectral-constellation-metric/v1"


@dataclass(frozen=True)
class AttributeWeights:
    """Declared contribution of every comparable attribute block.

    There are intentionally no defaults.  A caller must state all four weights, including a
    zero, so an omitted scientific choice cannot silently become a matcher configuration.
    A block weight is shared over that block's components; adding another edge therefore does
    not make geometry count more merely because the graph has one more number.
    """

    geometry: float
    bearings: float
    strengths: float
    scales: float

    def __post_init__(self) -> None:
        values = self.as_mapping()
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise InvalidParameterError(
                    "AttributeWeights.%s" % name, value,
                    "a finite non-negative declared weight")
        if not any(value > 0.0 for value in values.values()):
            raise InvalidParameterError(
                "AttributeWeights", values,
                "at least one positive weight. A zero metric calls every constellation the "
                "same pattern")

    def as_mapping(self) -> Dict[str, float]:
        return {
            "geometry": float(self.geometry),
            "bearings": float(self.bearings),
            "strengths": float(self.strengths),
            "scales": float(self.scales),
        }


@dataclass(frozen=True)
class SignatureFamily:
    """The quantities that must agree before a numerical distance has meaning."""

    mode: str
    cardinality: int
    has_bearings: bool
    scale_units: Optional[str]

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "cardinality": self.cardinality,
            "has_bearings": self.has_bearings,
            "scale_units": self.scale_units,
        }


@dataclass(frozen=True)
class SignaturePoint:
    """Only the comparable half of a T4E.2 signature, split into named blocks."""

    family: SignatureFamily
    geometry: Tuple[float, ...]
    bearings: Optional[Tuple[float, ...]]
    strengths: Tuple[float, ...]
    scales: Tuple[float, ...]

    def __post_init__(self) -> None:
        expected_edges = self.family.cardinality * (self.family.cardinality - 1) // 2
        if len(self.geometry) != expected_edges:
            raise InvalidParameterError(
                "SignaturePoint.geometry", len(self.geometry),
                "%d edge values for cardinality %d" %
                (expected_edges, self.family.cardinality))
        if (self.bearings is None) == self.family.has_bearings:
            raise InvalidParameterError(
                "SignaturePoint.bearings", self.bearings,
                "a bearing block exactly when family.has_bearings is true")
        if self.bearings is not None and len(self.bearings) != expected_edges:
            raise InvalidParameterError(
                "SignaturePoint.bearings", len(self.bearings),
                "%d edge bearings for cardinality %d" %
                (expected_edges, self.family.cardinality))
        if len(self.strengths) != self.family.cardinality or len(self.scales) != self.family.cardinality:
            raise InvalidParameterError(
                "SignaturePoint", (len(self.strengths), len(self.scales)),
                "one strength and one scale value per constellation member")
        for name, block in self.blocks().items():
            if any(not math.isfinite(value) for value in block):
                raise InvalidParameterError(
                    "SignaturePoint.%s" % name, block, "finite measurements")
            if name != "bearings" and any(value <= 0.0 for value in block):
                raise InvalidParameterError(
                    "SignaturePoint.%s" % name, block,
                    "positive measurements for a symmetric relative distance")
        if self.bearings is not None and any(value < 0.0 or value > 90.0
                                             for value in self.bearings):
            raise InvalidParameterError(
                "SignaturePoint.bearings", self.bearings,
                "folded bearings between 0 and 90 degrees")

    @classmethod
    def from_signature(cls, signature: ConstellationSignature) -> "SignaturePoint":
        if not isinstance(signature, ConstellationSignature):
            raise InvalidParameterError(
                "signature", type(signature).__name__, "a ConstellationSignature from T4E.2")
        if signature.scale_invariant:
            reference = math.exp(sum(math.log(value) for value in signature.scales)
                                 / len(signature.scales))
            scales = tuple(value / reference for value in signature.scales)
            units = None
        else:
            scales = tuple(signature.scales)
            units = signature.scale_units
        return cls(
            family=SignatureFamily(
                mode=signature.mode, cardinality=signature.cardinality,
                has_bearings=signature.bearings is not None, scale_units=units),
            geometry=tuple(signature.geometry),
            bearings=None if signature.bearings is None else tuple(signature.bearings),
            strengths=tuple(signature.strengths), scales=scales)

    def blocks(self) -> Dict[str, Tuple[float, ...]]:
        return {
            "geometry": self.geometry,
            "bearings": () if self.bearings is None else self.bearings,
            "strengths": self.strengths,
            "scales": self.scales,
        }

    def vector(self) -> Tuple[float, ...]:
        values = list(self.geometry)
        if self.bearings is not None:
            values.extend(self.bearings)
        values.extend(self.strengths)
        values.extend(self.scales)
        return tuple(values)

    def describe(self) -> Dict[str, Any]:
        return {
            "family": self.family.describe(),
            "geometry": list(self.geometry),
            "bearings_deg": None if self.bearings is None else list(self.bearings),
            "strengths": list(self.strengths),
            "scales": list(self.scales),
        }

    def permuted(self, order: Sequence[int]) -> "SignaturePoint":
        """Relabel nodes while keeping every edge attribute attached to its endpoints."""
        order = tuple(int(index) for index in order)
        if sorted(order) != list(range(self.family.cardinality)):
            raise InvalidParameterError(
                "SignaturePoint.permuted.order", order,
                "a permutation of every node index exactly once")
        pairs = tuple(itertools.combinations(range(self.family.cardinality), 2))
        geometry = {pair: value for pair, value in zip(pairs, self.geometry)}
        bearings = (None if self.bearings is None else
                    {pair: value for pair, value in zip(pairs, self.bearings)})

        def edge_block(values: Mapping[Tuple[int, int], float]) -> Tuple[float, ...]:
            output = []
            for left, right in pairs:
                old_pair = tuple(sorted((order[left], order[right])))
                output.append(values[old_pair])
            return tuple(output)

        return SignaturePoint(
            family=self.family, geometry=edge_block(geometry),
            bearings=None if bearings is None else edge_block(bearings),
            strengths=tuple(self.strengths[index] for index in order),
            scales=tuple(self.scales[index] for index in order))


def _relative(left: float, right: float) -> float:
    denominator = (abs(left) + abs(right)) / 2.0
    return 0.0 if denominator == 0.0 else abs(left - right) / denominator


@dataclass(frozen=True)
class SignatureMetric:
    """A dimensionless weighted RMS distance over the four declared blocks."""

    weights: AttributeWeights

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": METRIC_SCHEMA,
            "weights": self.weights.as_mapping(),
            "component_distance": {
                "geometry": "symmetric relative difference",
                "bearings": "absolute folded-angle difference divided by 90 degrees",
                "strengths": "symmetric relative difference",
                "scales": "symmetric relative difference; absolute scales in scale-specific "
                          "mode and geometric-mean-normalised ratios in scale-invariant mode",
            },
            "block_reduction": "root mean square within each block",
            "metric_reduction": "weighted root mean square over available positive-weight blocks",
            "node_correspondence": (
                "minimum over every node permutation (six at cardinality three); edge and "
                "node attributes move together under a permutation"),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _raw_breakdown(self, left: SignaturePoint,
                       right: SignaturePoint) -> Dict[str, float]:
        if left.family != right.family:
            raise InvalidParameterError(
                "SignatureMetric.distance.family",
                (left.family.describe(), right.family.describe()),
                "the same mode, cardinality, bearing availability and scale units. Distances "
                "between different measured quantities are not numbers")
        output: Dict[str, float] = {}
        for name, left_block in left.blocks().items():
            right_block = right.blocks()[name]
            if not left_block:
                continue
            if name == "bearings":
                differences = [abs(a - b) / 90.0 for a, b in zip(left_block, right_block)]
            else:
                differences = [_relative(a, b) for a, b in zip(left_block, right_block)]
            output[name] = math.sqrt(sum(value * value for value in differences)
                                     / len(differences))
        return output

    def _reduce(self, blocks: Mapping[str, float]) -> float:
        weights = self.weights.as_mapping()
        weighted = [(weights[name], value)
                    for name, value in blocks.items()
                    if weights[name] > 0.0]
        if not weighted:
            raise InvalidParameterError(
                "SignatureMetric.weights", self.weights.as_mapping(),
                "a positive weight on at least one attribute available in this signature "
                "family")
        numerator = sum(weight * value * value for weight, value in weighted)
        return math.sqrt(numerator / sum(weight for weight, _ in weighted))

    def alignment(self, left: SignaturePoint, right: SignaturePoint
                  ) -> Tuple[SignaturePoint, Dict[str, float], float]:
        """Return the right-node correspondence with the smallest attributed distance."""
        if left.family != right.family:
            # Route this through the one detailed refusal rather than failing in permutations.
            self._raw_breakdown(left, right)
        candidates = []
        for order in itertools.permutations(range(right.family.cardinality)):
            aligned = right.permuted(order)
            breakdown = self._raw_breakdown(left, aligned)
            distance = self._reduce(breakdown)
            candidates.append((distance, aligned.vector(), aligned, breakdown))
        distance, _, aligned, breakdown = min(candidates, key=lambda row: (row[0], row[1]))
        return aligned, breakdown, distance

    def breakdown(self, left: SignaturePoint, right: SignaturePoint) -> Dict[str, float]:
        return self.alignment(left, right)[1]

    def distance(self, left: SignaturePoint, right: SignaturePoint) -> float:
        return self.alignment(left, right)[2]


#: The admission rate was measured against a real contrast population.
DISCRIMINATION_MEASURED = "MEASURED"
#: No contrast was supplied, so what this radius admits is unknown. This is not zero and it is
#: not small; it is unmeasured, and a radius carrying it is not defensible on that ground.
DISCRIMINATION_UNMEASURED = "ADMISSION_RATE_NOT_MEASURED"

DISCRIMINATION_NOTE = (
    "A radius has two error rates and this programme measured only one of them until T4E.6. "
    "The false-split rate -- how often two measurements of one configuration land outside the "
    "radius -- is what a replicate calibration reports. The admission rate -- how often two "
    "configurations that are not the same land inside it -- is the one that decides whether a "
    "pattern means anything, and it cannot be computed from replicates alone because replicates "
    "contain no example of two different things. Both are reported here, or the second is "
    "reported as unmeasured; neither is ever defaulted to a number nobody measured.")


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def at(fraction: float) -> float:
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]

    return {"minimum": ordered[0], "q05": at(0.05), "median": at(0.5), "q95": at(0.95),
            "maximum": ordered[-1]}


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


@dataclass(frozen=True)
class Discrimination:
    """What a radius splits and what it admits, or a named refusal for the second."""

    status: str
    radius: Optional[float]
    n_same_pairs: int
    n_different_pairs: int
    same: Mapping[str, float]
    different: Mapping[str, float]
    false_split_rate: Optional[float]
    admission_rate: Optional[float]
    separation: Optional[float]
    note: str

    @property
    def measured(self) -> bool:
        return self.status == DISCRIMINATION_MEASURED

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "status": self.status,
            "radius": self.radius,
            "n_same_configuration_pairs": self.n_same_pairs,
            "same_configuration_distances": dict(self.same),
            "false_split_rate": self.false_split_rate,
            "basis": DISCRIMINATION_NOTE,
            "note": self.note,
        }
        if self.measured:
            record.update({
                "n_different_configuration_pairs": self.n_different_pairs,
                "different_configuration_distances": dict(self.different),
                "admission_rate": self.admission_rate,
                "standardised_separation": self.separation,
            })
        return record


def measure_discrimination(same_distances: Sequence[float],
                           different_distances: Sequence[float],
                           radius: float) -> Discrimination:
    """Both error rates of one radius, against two measured distance populations.

    `same_distances` are distances between measurements of one configuration; a value above the
    radius is a **split** of something that should have stayed together. `different_distances`
    are distances between configurations that are not the same one; a value at or below the
    radius is an **admission** of something the radius does not distinguish. Neither rate is a
    property of the radius alone -- both are properties of the radius *and* the populations, so
    both populations travel in the record.
    """
    same = [float(value) for value in same_distances]
    if not same:
        raise InvalidParameterError(
            "same_distances", 0,
            "at least one distance between measurements of one configuration")
    limit = float(radius)
    if not math.isfinite(limit) or limit < 0.0:
        raise InvalidParameterError("radius", radius, "a finite non-negative radius")
    split = sum(1 for value in same if value > limit) / float(len(same))

    different = [float(value) for value in different_distances]
    if not different:
        return Discrimination(
            status=DISCRIMINATION_UNMEASURED, radius=limit, n_same_pairs=len(same),
            n_different_pairs=0, same=_quantiles(same), different={},
            false_split_rate=split, admission_rate=None, separation=None,
            note=("No contrast population was supplied, so what this radius admits has not been "
                  "measured. That is not the same as admitting nothing: a replicate calibration "
                  "contains no example of two different configurations and therefore cannot "
                  "produce this number by itself."))
    admitted = sum(1 for value in different if value <= limit) / float(len(different))
    pooled = math.sqrt(0.5 * (_variance(same) + _variance(different)))
    same_median = sorted(same)[len(same) // 2]
    different_median = sorted(different)[len(different) // 2]
    separation = (0.0 if pooled == 0.0 else (different_median - same_median) / pooled)
    return Discrimination(
        status=DISCRIMINATION_MEASURED, radius=limit, n_same_pairs=len(same),
        n_different_pairs=len(different), same=_quantiles(same),
        different=_quantiles(different), false_split_rate=split, admission_rate=admitted,
        separation=separation,
        note=("At this radius, %.2f%% of same-configuration pairs are split and %.2f%% of "
              "different-configuration pairs are admitted. The two medians are %.4f and %.4f, "
              "%.3f pooled standard deviations apart."
              % (100.0 * split, 100.0 * admitted, same_median, different_median, separation)))


def discrimination_curve(same_distances: Sequence[float],
                         different_distances: Sequence[float],
                         radii: Optional[Sequence[float]] = None) -> Tuple[Discrimination, ...]:
    """Both rates across a sweep of radii, so the whole trade-off is visible at once.

    A single operating point can flatter a metric. The curve is what makes a later change to the
    signature judgeable: if it does not move this, it did not improve discrimination.
    """
    if radii is None:
        pool = sorted(set(float(value) for value in same_distances)
                      | set(float(value) for value in different_distances))
        if not pool:
            raise InvalidParameterError("radii", None, "distances to sweep over, or explicit radii")
        step = max(1, len(pool) // 64)
        radii = pool[::step]
    return tuple(measure_discrimination(same_distances, different_distances, float(radius))
                 for radius in radii)


def best_operating_point(same_distances: Sequence[float],
                         different_distances: Sequence[float],
                         target_rate: float) -> Optional[Discrimination]:
    """The smallest radius holding **both** rates at or below `target_rate`, or `None`.

    `None` is the answer that matters. It says no radius separates these two populations at the
    rate asked for, which is a fact about the signature and not about the search -- and it is
    what a caller must be able to discover before building an identity on top of it.
    """
    if not 0.0 < float(target_rate) < 1.0:
        raise InvalidParameterError("target_rate", target_rate, "a rate strictly between 0 and 1")
    for row in discrimination_curve(same_distances, different_distances):
        if not row.measured:
            return None
        if row.false_split_rate <= float(target_rate) \
                and row.admission_rate <= float(target_rate):
            return row
    return None


@dataclass(frozen=True)
class SignatureTolerance:
    """A TG3.4 ``MatchTolerance`` bound to one metric and one signature family."""

    measurement: MatchTolerance
    metric_digest: str
    family: SignatureFamily
    pair_distances: Tuple[float, ...]
    #: What this radius admits as well as what it splits (T4E.6). Never absent: where no
    #: contrast population was supplied it carries a named refusal rather than a silence, so a
    #: reader cannot mistake "not measured" for "measured and small".
    discrimination: Optional[Discrimination] = None

    @property
    def value(self) -> float:
        return self.measurement.value

    def describe(self) -> Dict[str, Any]:
        body = self.measurement.describe()
        body.update({
            "metric_digest": self.metric_digest,
            "family": self.family.describe(),
            "pair_distances": list(self.pair_distances),
            "discrimination": (
                Discrimination(
                    status=DISCRIMINATION_UNMEASURED, radius=self.value,
                    n_same_pairs=len(self.pair_distances), n_different_pairs=0,
                    same=_quantiles(self.pair_distances), different={},
                    false_split_rate=None, admission_rate=None, separation=None,
                    note=("This tolerance was built without a discrimination measurement at "
                          "all, so neither rate is attached to it.")).as_record()
                if self.discrimination is None else self.discrimination.as_record()),
        })
        return body


def _points(signatures: Iterable[ConstellationSignature]) -> Tuple[SignaturePoint, ...]:
    return tuple(SignaturePoint.from_signature(item) for item in signatures)


def calibrate_signature_tolerance(
        replicates: Sequence[ConstellationSignature], *,
        metric: SignatureMetric,
        contrast: Sequence[ConstellationSignature] = ()) -> SignatureTolerance:
    """Measure the single-comparison radius from known same-configuration replicates.

    This is the signature-space counterpart of TG3.4's ``calibrate_match_tolerance`` and
    deliberately reuses its measured record type.  The maximum of all replicate-pair
    distances is a noise floor, not an estimated population quantile; its finite-replicate
    false-rejection limitation is retained in the basis.

    **`contrast` decides whether the returned radius is defensible (T4E.6).** It is a population
    of signatures known *not* to be the configuration the replicates measure, and without it the
    admission rate -- how often this radius calls two different things one pattern -- cannot be
    computed, because replicates contain no example of two different things. Supplying none is
    permitted and is recorded as a refusal on the returned tolerance rather than passed over: a
    reader of `describe()` always learns whether the question was asked.

    See D97 for why this matters and for what a real record does to the assumption underneath
    it: an atmospheric record contains no replicates at all, and the radius this returns there is
    dominated by physical evolution rather than by measurement noise.
    """
    points = _points(replicates)
    if len(points) < 2:
        raise InvalidParameterError(
            "replicates", len(points),
            "at least two measurements of the same physical configuration")
    family = points[0].family
    if any(point.family != family for point in points[1:]):
        raise InvalidParameterError(
            "replicates", [point.family.describe() for point in points],
            "one signature family. A mode, cardinality, axis admission or unit change is "
            "not measurement jitter")

    rows = []
    worst_component = "(none)"
    for left, right in itertools.combinations(points, 2):
        distance = metric.distance(left, right)
        rows.append(distance)
        if distance >= max(rows):
            breakdown = metric.breakdown(left, right)
            weighted = {name: metric.weights.as_mapping()[name] * value * value
                        for name, value in breakdown.items()}
            worst_component = max(weighted, key=weighted.get)
    components = len(points[0].vector())
    pairs = len(rows)
    measurement = MatchTolerance(
        value=max(rows), n_replicates=len(points), components=components,
        worst_component=worst_component,
        basis=("largest declared-weight signature distance among %d measurements of one "
               "physical configuration (%d replicate pairs, %d numeric attributes); this "
               "is the measured noise floor and has an approximate single-comparison "
               "false-rejection rate of %.3g"
               % (len(points), pairs, components, 1.0 / (pairs + 1))))
    contrast_points = _points(contrast) if contrast else ()
    for point in contrast_points:
        if point.family != family:
            raise InvalidParameterError(
                "contrast", point.family.describe(),
                "the signature family the replicates were measured in (%s). A distance between "
                "different measured quantities is not a number, so it cannot be a contrast for "
                "this radius either" % family.describe())
    across = [metric.distance(left, right)
              for left in points for right in contrast_points]
    return SignatureTolerance(
        measurement=measurement, metric_digest=metric.digest, family=family,
        pair_distances=tuple(rows),
        discrimination=measure_discrimination(rows, across, measurement.value))


def _centroid(points: Sequence[SignaturePoint], metric: SignatureMetric) -> SignaturePoint:
    family = points[0].family

    # A medoid supplies a real, deterministic correspondence frame.  Averaging each point's
    # canonical vector directly is wrong at a canonical-order boundary: infinitesimal noise can
    # exchange two nodes and attach a strength or scale to the wrong vertex.
    reference = min(
        points,
        key=lambda candidate: (
            sum(metric.distance(candidate, other) for other in points), candidate.vector()))
    aligned = [metric.alignment(reference, point)[0] for point in points]

    def mean_block(name: str) -> Tuple[float, ...]:
        blocks = [point.blocks()[name] for point in aligned]
        return tuple(sum(values) / len(values) for values in zip(*blocks))

    bearings = mean_block("bearings") if family.has_bearings else None
    return SignaturePoint(
        family=family, geometry=mean_block("geometry"), bearings=bearings,
        strengths=mean_block("strengths"), scales=mean_block("scales"))


def _member_key(signature: ConstellationSignature) -> Tuple[Any, ...]:
    point = SignaturePoint.from_signature(signature)
    return (point.vector(), repr(signature.key), float(signature.time), signature.track_ids)


@dataclass(frozen=True)
class ConstellationPattern:
    """One approximate pattern: centroid plus its measured admissible radius."""

    pattern_id: int
    members: Tuple[ConstellationSignature, ...]
    centroid: SignaturePoint
    tolerance_radius: float
    observed_radius: float

    @property
    def support(self) -> int:
        """Member count only; minimum-support decisions belong to T4E.4."""
        return len(self.members)

    def describe(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "support_observed": self.support,
            "centroid": self.centroid.describe(),
            "tolerance_radius": self.tolerance_radius,
            "observed_radius": self.observed_radius,
            "member_keys": [list(item.key) for item in self.members],
            "claim_boundary": (
                "support_observed is membership in this supplied batch, not minimum-support "
                "mining, recurrence evidence, a p-value or a discovery; those begin at T4E.4"),
        }


@dataclass(frozen=True)
class PatternCatalogue:
    patterns: Tuple[ConstellationPattern, ...]
    metric: SignatureMetric
    tolerance: SignatureTolerance
    n_signatures: int

    def __len__(self) -> int:
        return len(self.patterns)

    def __iter__(self):
        return iter(self.patterns)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": CLUSTER_SCHEMA,
            "algorithm": "deterministic complete-link agglomeration",
            "merge_rule": (
                "all cross-member distances and all candidate member-to-centroid distances "
                "must be no greater than the replicate-calibrated tolerance"),
            "n_signatures": self.n_signatures,
            "n_patterns": len(self.patterns),
            "metric": self.metric.describe(),
            "metric_digest": self.metric.digest,
            "tolerance": self.tolerance.describe(),
            "patterns": [pattern.describe() for pattern in self.patterns],
        }


def cluster_signatures(
        signatures: Sequence[ConstellationSignature], *, metric: SignatureMetric,
        tolerance: SignatureTolerance) -> PatternCatalogue:
    """Cluster compatible signatures without allowing single-link bridge chains."""
    ordered = tuple(sorted(signatures, key=_member_key))
    if not ordered:
        raise InvalidParameterError(
            "signatures", 0, "at least one signed constellation to cluster")
    points = _points(ordered)
    if metric.digest != tolerance.metric_digest:
        raise InvalidParameterError(
            "metric", metric.digest,
            "the exact declared metric used to calibrate this tolerance (%s)" %
            tolerance.metric_digest)
    if any(point.family != tolerance.family for point in points):
        raise InvalidParameterError(
            "signatures", [point.family.describe() for point in points],
            "the signature family on which the tolerance was measured (%s)" %
            tolerance.family.describe())

    clusters: Tuple[Tuple[int, ...], ...] = tuple((index,) for index in range(len(points)))
    while True:
        candidates = []
        for left_index, right_index in itertools.combinations(range(len(clusters)), 2):
            left, right = clusters[left_index], clusters[right_index]
            cross = [metric.distance(points[a], points[b]) for a in left for b in right]
            if max(cross) > tolerance.value:
                continue
            merged = tuple(sorted(left + right))
            centre = _centroid([points[index] for index in merged], metric)
            radii = [metric.distance(points[index], centre) for index in merged]
            if max(radii) > tolerance.value:
                continue
            candidates.append((max(cross), merged, left_index, right_index))
        if not candidates:
            break
        _, merged, left_index, right_index = min(candidates)
        clusters = tuple(cluster for index, cluster in enumerate(clusters)
                         if index not in (left_index, right_index)) + (merged,)
        clusters = tuple(sorted(clusters))

    patterns = []
    for indices in clusters:
        member_points = [points[index] for index in indices]
        centre = _centroid(member_points, metric)
        observed = max(metric.distance(point, centre) for point in member_points)
        members = tuple(ordered[index] for index in indices)
        patterns.append((centre.vector(), members, centre, observed))
    patterns.sort(key=lambda row: (row[0], tuple(_member_key(item) for item in row[1])))
    records = tuple(
        ConstellationPattern(
            pattern_id=index + 1, members=row[1], centroid=row[2],
            tolerance_radius=tolerance.value, observed_radius=row[3])
        for index, row in enumerate(patterns))
    return PatternCatalogue(
        patterns=records, metric=metric, tolerance=tolerance,
        n_signatures=len(ordered))


__all__ = [
    "AttributeWeights", "CLUSTER_SCHEMA", "ConstellationPattern", "METRIC_SCHEMA",
    "PatternCatalogue", "SignatureFamily", "SignatureMetric", "SignaturePoint",
    "SignatureTolerance", "calibrate_signature_tolerance", "cluster_signatures",
    "Discrimination", "DISCRIMINATION_MEASURED", "DISCRIMINATION_UNMEASURED",
    "DISCRIMINATION_NOTE", "measure_discrimination", "discrimination_curve",
    "best_operating_point",
]
