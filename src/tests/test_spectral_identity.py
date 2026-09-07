"""T4E.5: the identity step at record scale, and the one thing it is not allowed to change.

1.  **The test that matters is equality, not similarity.** A faster clustering that returns a
    *comparable* answer has changed what a pattern is, and every support count, every lift and
    every gate verdict downstream would silently mean something else. So the suite runs both
    implementations on the same inputs and compares their **receipts** -- pattern ids, member
    keys, centroid vectors and observed radii -- rather than counts or sizes.

2.  **The decomposition is exact for a stated reason, and the reason is tested.** Complete
    linkage requires every pair inside a cluster to be within the tolerance, so no cluster can
    span two components of the tolerance graph. The suite constructs a record whose graph
    genuinely splits and asserts that no returned pattern draws members from two components.

3.  **The box is a filter and never an answer.** It is derived from the metric's own algebra:
    the relative difference is `2*tanh(|dlog|/2)`, which the suite verifies numerically, and the
    weighted RMS forces a per-component bound. Every pair the box admits is then measured
    exactly, so the suite asserts that the box never excludes a true neighbour -- checked
    against brute force on every pair.

4.  **A bound that constrains nothing says so.** A zero-weight block cannot bound the distance
    and a per-component bound at or above 2 excludes nothing, because a relative difference of
    two positive numbers never reaches 2. Both are asserted, because a filter that silently
    dropped a block the metric ignores would drop real neighbours.

5.  **Refusing beats approximating.** When the decomposition leaves a component too large to
    cluster exactly, the module refuses and names the number. It does not return an approximate
    identity under the same name as an exact one.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureFamily, SignatureMetric, SignaturePoint, SignatureTolerance,
    calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_identity import (
    BEARING_SCALE_DEGREES, EQUIVALENCE_NOTE, IDENTITY_SCHEMA, RELATIVE_DIFFERENCE_SUPREMUM,
    cluster_signatures_scalable, component_bounds, neighbour_graph, plan_clustering,
)
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_constellation import extract_constellations
from src.core.errors import InvalidParameterError

METRIC = SignatureMetric(AttributeWeights(1.0, 1.0, 1.0, 1.0))
SEED = 20260907

_CACHE: dict = {}


def _record():
    """A synthetic tracked record giving real signed constellations, built once.

    Real signatures rather than hand-made `SignaturePoint`s, because what is under test is
    agreement with T4E.3 on the things T4E.3 is actually given.
    """
    if "record" not in _CACHE:
        import torch
        from src.analysis_engine.spectral_tracking import track_spectral_features
        from src.physical_core.field import PhysicalField
        from src.physical_core.grid import GridSpec
        from src.physical_core.sequence import FieldSequence
        from src.transform_engine.coefficient_field import decompose_sequence

        rows, cols, steps = 96, 96, 40
        grid = GridSpec.cartesian((rows, cols), dy_m=31000.0)
        generator = torch.Generator().manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        fields = []
        # Three well-separated families of blobs, so the tolerance graph has a real chance of
        # splitting rather than collapsing to one component.
        # Widths and amplitudes far enough apart, and jitter tight enough, that the calibrated
        # tolerance does *not* swallow the whole record. A fixture whose graph is one component
        # would exercise the decomposition without ever testing that it decomposes: measured,
        # this one gives components of 96, 14, 7 and 3 at n=120.
        centres = [(24.0, 24.0), (24.0, 72.0), (72.0, 48.0)]
        widths = (2.5, 6.0, 10.0)
        amplitudes = (1.0, 2.0, 3.2)
        for step in range(steps):
            data = torch.zeros((rows, cols), dtype=torch.float64)
            for index, (cy, cx) in enumerate(centres):
                width = widths[index]
                jitter = rng.normal(0.0, 0.02, size=2)
                yy, xx = torch.meshgrid(
                    torch.arange(rows, dtype=torch.float64) - (cy + jitter[0]),
                    torch.arange(cols, dtype=torch.float64) - (cx + jitter[1]), indexing="ij")
                data = data + amplitudes[index] * torch.exp(
                    -(yy ** 2 + xx ** 2) / (2.0 * width ** 2))
            fields.append(PhysicalField(
                data + 0.02 * torch.randn(rows, cols, generator=generator,
                                          dtype=torch.float64),
                grid=grid, units="dimensionless"))
        sequence = FieldSequence(fields, np.arange(steps, dtype=float))
        field = decompose_sequence(sequence, "swt",
                                   {"levels": 4, "wavelet": "haar"}).select(scales=[2, 3])
        tracked = track_spectral_features(
            field, domain="synthetic", dataset="identity", variable="amplitude",
            time_units="frames", threshold_sigma=3.0)
        constellations = extract_constellations(tracked, cardinalities=(2,),
                                                max_nodes_per_frame=32)
        signatures = list(sign_constellations(constellations,
                                              scale_invariant=False).signatures)
        _CACHE["record"] = signatures
    return _CACHE["record"]


@pytest.fixture(scope="module")
def signatures():
    return _record()


@pytest.fixture(scope="module")
def tolerance(signatures):
    """Calibrated the way the function intends: repeated observations of one configuration.

    The same set of tracks, co-present at successive frames, is one physical configuration
    measured again. Calibrating from arbitrary signatures instead is what made the first
    measurement of D96 pessimistic by a factor of two, and the mistake is not repeated here.
    """
    from collections import defaultdict
    runs = defaultdict(list)
    for item in signatures:
        runs[tuple(sorted(item.track_ids))].append(item)
    longest = max(runs.values(), key=len)
    assert len(longest) >= 3, "the record must observe one configuration several times"
    return calibrate_signature_tolerance(sorted(longest, key=lambda s: s.time),
                                         metric=METRIC)


def _receipt(catalogue):
    return json.dumps(catalogue.describe(), sort_keys=True, default=str)


def _points_sorted(signatures):
    from src.analysis_engine.spectral_clustering import _member_key, _points
    return _points(sorted(signatures, key=_member_key))


# ---------------------------------------------------------------------------------------------
# section 1: the same clustering, not a comparable one
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 12, 30, 60, 120])
def test_the_fast_clustering_is_the_same_clustering(signatures, tolerance, n):
    """Receipts compared whole. A count or a size agreeing proves nothing about identity."""
    sample = signatures[:n]
    slow = cluster_signatures(sample, metric=METRIC, tolerance=tolerance)
    fast = cluster_signatures_scalable(sample, metric=METRIC, tolerance=tolerance)
    assert _receipt(fast) == _receipt(slow)


def test_the_members_of_every_pattern_are_the_same_objects(signatures, tolerance):
    sample = signatures[:80]
    slow = cluster_signatures(sample, metric=METRIC, tolerance=tolerance)
    fast = cluster_signatures_scalable(sample, metric=METRIC, tolerance=tolerance)
    assert len(fast.patterns) == len(slow.patterns)
    for left, right in zip(slow.patterns, fast.patterns):
        assert left.pattern_id == right.pattern_id
        assert [item.key for item in left.members] == [item.key for item in right.members]
        assert left.centroid.vector() == right.centroid.vector()
        assert left.observed_radius == right.observed_radius
        assert left.tolerance_radius == right.tolerance_radius


def test_agreement_holds_when_the_tolerance_is_wide_enough_to_merge_almost_everything(
        signatures, tolerance):
    """The regime that made D96 slow is the regime the decomposition helps least in."""
    wide = SignatureTolerance(
        measurement=type(tolerance.measurement)(
            value=tolerance.value * 4.0, n_replicates=tolerance.measurement.n_replicates,
            components=tolerance.measurement.components,
            worst_component=tolerance.measurement.worst_component,
            basis="widened four-fold by this test to force a single component"),
        metric_digest=tolerance.metric_digest, family=tolerance.family,
        pair_distances=tolerance.pair_distances)
    sample = signatures[:40]
    plan = plan_clustering(sample, metric=METRIC, tolerance=wide)
    assert plan.largest_component > len(sample) // 2, "the tolerance did not merge the graph"
    assert _receipt(cluster_signatures_scalable(sample, metric=METRIC, tolerance=wide)) == \
        _receipt(cluster_signatures(sample, metric=METRIC, tolerance=wide))


def test_a_tolerance_of_zero_leaves_every_configuration_its_own_pattern(signatures,
                                                                        tolerance):
    tight = SignatureTolerance(
        measurement=type(tolerance.measurement)(
            value=0.0, n_replicates=tolerance.measurement.n_replicates,
            components=tolerance.measurement.components,
            worst_component=tolerance.measurement.worst_component,
            basis="zero, by this test"),
        metric_digest=tolerance.metric_digest, family=tolerance.family,
        pair_distances=tolerance.pair_distances)
    sample = signatures[:40]
    fast = cluster_signatures_scalable(sample, metric=METRIC, tolerance=tight)
    assert _receipt(fast) == _receipt(cluster_signatures(sample, metric=METRIC,
                                                         tolerance=tight))


# ---------------------------------------------------------------------------------------------
# section 2: why the decomposition is exact
# ---------------------------------------------------------------------------------------------


def test_no_pattern_ever_draws_members_from_two_components(signatures, tolerance):
    sample = signatures[:120]
    plan = plan_clustering(sample, metric=METRIC, tolerance=tolerance)
    assert len(plan.graph.components) > 1, "this record's graph must actually split"
    where = {}
    for number, component in enumerate(plan.graph.components):
        for index in component:
            where[index] = number
    keys = [item.key for item in
            sorted(sample, key=lambda s: __import__(
                "src.analysis_engine.spectral_clustering", fromlist=["_member_key"]
            )._member_key(s))]
    catalogue = cluster_signatures_scalable(sample, metric=METRIC, tolerance=tolerance)
    position = {key: index for index, key in enumerate(keys)}
    for pattern in catalogue:
        components = {where[position[item.key]] for item in pattern.members}
        assert len(components) == 1, "a pattern spanned two components of the tolerance graph"


def test_two_configurations_in_different_components_are_never_within_tolerance(signatures,
                                                                               tolerance):
    """The property the decomposition rests on, checked against every cross-component pair."""
    from src.analysis_engine.spectral_clustering import _member_key, _points
    sample = sorted(signatures[:100], key=_member_key)
    points = _points(sample)
    graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
    assert len(graph.components) > 1
    where = {index: number for number, comp in enumerate(graph.components) for index in comp}
    checked = 0
    for left, right in itertools.combinations(range(len(points)), 2):
        if where[left] == where[right]:
            continue
        assert METRIC.distance(points[left], points[right]) > tolerance.value
        checked += 1
    assert checked > 0, "there were no cross-component pairs to check"


# ---------------------------------------------------------------------------------------------
# section 3: the box is a filter, and never an answer
# ---------------------------------------------------------------------------------------------


def test_the_relative_difference_is_exactly_two_tanh_of_half_the_log_difference():
    """The identity the whole box rests on, checked numerically over the positive quadrant."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(1e-3, 1e3, 20000)
    b = rng.uniform(1e-3, 1e3, 20000)
    relative = np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2.0)
    identity = 2.0 * np.tanh(np.abs(np.log(a) - np.log(b)) / 2.0)
    assert np.abs(relative - identity).max() < 1e-12
    assert relative.max() < RELATIVE_DIFFERENCE_SUPREMUM


def test_the_box_never_excludes_a_true_neighbour(signatures, tolerance):
    """Checked against brute force: every pair the metric admits, the box must have admitted."""
    from src.analysis_engine.spectral_clustering import _member_key, _points
    sample = sorted(signatures[:90], key=_member_key)
    points = _points(sample)
    graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
    where = {index: number for number, comp in enumerate(graph.components) for index in comp}
    true_edges = 0
    for left, right in itertools.combinations(range(len(points)), 2):
        if METRIC.distance(points[left], points[right]) <= tolerance.value:
            true_edges += 1
            assert where[left] == where[right], (
                "a genuinely within-tolerance pair was separated, so the box excluded it")
    assert true_edges > 0


def test_the_box_admits_more_than_it_keeps_and_publishes_both(signatures, tolerance):
    from src.analysis_engine.spectral_clustering import _member_key, _points
    points = _points(sorted(signatures[:120], key=_member_key))
    graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
    record = graph.as_record()
    assert record["n_exact_distances_computed"] <= \
        record["n_candidate_pairs_the_box_admitted"]
    assert record["distances_not_computed"] >= 0
    assert record["exactness"] == EQUIVALENCE_NOTE


def test_the_union_count_is_a_spanning_structure_and_says_so(signatures, tolerance):
    """It counts components joined, not pairs within tolerance, and the receipt must not
    let a reader divide it by the pairs and call the result a density.

    A pair whose ends already share a component is skipped without being measured, which is
    what makes the decomposition cheap -- and it means the count that survives is exactly
    `n_points - n_components`, however dense the true neighbourhood is.
    """
    for n in (40, 80, 120):
        points = _points_sorted(signatures[:n])
        graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
        assert graph.n_unions == graph.n_points - len(graph.components)
        record = graph.as_record()
        assert record["n_unions_performed"] == graph.n_unions
        assert "not the number of within-tolerance pairs" in record["n_unions_basis"]
        assert "n_edges_within_tolerance" not in record
    # And the true neighbourhood really is denser than that count, or it would prove nothing.
    points = _points_sorted(signatures[:120])
    truly_within = sum(
        1 for left, right in itertools.combinations(range(len(points)), 2)
        if METRIC.distance(points[left], points[right]) <= tolerance.value)
    graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
    assert truly_within > graph.n_unions


def test_a_block_the_metric_ignores_cannot_bound_the_distance(signatures):
    """A zero weight means that block is not in the distance, so it must not filter on it.

    Filtering on a block the metric ignores would exclude configurations the metric itself
    calls neighbours, which is the one way a filter can be wrong rather than merely slow.
    """
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    ignored = SignatureMetric(AttributeWeights(0.0, 1.0, 1.0, 1.0))
    bounds = component_bounds(point, ignored, 0.5)
    assert not bounds["geometry"].as_record()["constrains"]
    assert "zero weight" in bounds["geometry"].basis
    assert bounds["scales"].as_record()["constrains"]
    # And the filter must actually admit what the metric admits under those weights.
    points = _points(signatures[:60])
    graph = neighbour_graph(points, metric=ignored, tolerance=0.25)
    where = {index: number for number, comp in enumerate(graph.components) for index in comp}
    for left, right in itertools.combinations(range(len(points)), 2):
        if ignored.distance(points[left], points[right]) <= 0.25:
            assert where[left] == where[right]


def test_this_record_carries_no_bearings_and_a_family_that_does_is_bounded_in_degrees(
        signatures):
    """The bearing branch has no data here, so it is exercised on a point that does have one."""
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    assert not point.family.has_bearings
    assert "bearings" not in component_bounds(point, METRIC, 0.2)
    bearing_point = SignaturePoint(
        family=SignatureFamily(mode=point.family.mode, cardinality=2, has_bearings=True,
                               scale_units=point.family.scale_units),
        geometry=(12.0,), bearings=(35.0,), strengths=(1.0, 2.0), scales=(4.0, 8.0))
    bounds = component_bounds(bearing_point, METRIC, 0.2)
    assert bounds["bearings"].half_width == pytest.approx(
        bounds["bearings"].difference * BEARING_SCALE_DEGREES)
    assert "degrees" in bounds["bearings"].basis
    assert bounds["bearings"].as_record()["constrains"]


def test_a_bound_at_or_above_two_excludes_nothing(signatures):
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    generous = component_bounds(point, METRIC, 1.5)
    assert generous["geometry"].difference >= RELATIVE_DIFFERENCE_SUPREMUM
    assert not generous["geometry"].as_record()["constrains"]
    assert "excludes nothing" in generous["geometry"].basis
    tight = component_bounds(point, METRIC, 0.05)
    assert tight["geometry"].as_record()["constrains"]


def test_a_relative_block_is_bounded_in_log_units_by_the_artanh_identity(signatures):
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    bounds = component_bounds(point, METRIC, 0.2)
    assert bounds["scales"].half_width == pytest.approx(
        2.0 * np.arctanh(bounds["scales"].difference / RELATIVE_DIFFERENCE_SUPREMUM))
    assert "artanh" in bounds["scales"].basis


def test_the_bound_is_the_weighted_rms_rearranged(signatures):
    """Derived, not fitted: T * sqrt(n_block * sum_weights / weight)."""
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    weights = {"geometry": 1.0, "bearings": 2.0, "strengths": 0.5, "scales": 3.0}
    metric = SignatureMetric(AttributeWeights(**weights))
    blocks = {name: values for name, values in point.blocks().items() if values}
    total = sum(weights[name] for name in blocks)
    bounds = component_bounds(point, metric, 0.05)
    for name, values in blocks.items():
        expected = 0.05 * np.sqrt(len(values) * total / weights[name])
        assert bounds[name].difference == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------
# section 4: refusing beats approximating
# ---------------------------------------------------------------------------------------------


def test_a_component_too_large_to_cluster_exactly_is_refused_not_approximated(signatures,
                                                                              tolerance):
    sample = signatures[:60]
    plan = plan_clustering(sample, metric=METRIC, tolerance=tolerance, max_component=2)
    assert not plan.feasible
    assert "no exact method avoids the cost" in plan.note
    assert "do not read this as a reason to widen the tolerance" in plan.note
    with pytest.raises(InvalidParameterError) as excinfo:
        cluster_signatures_scalable(sample, metric=METRIC, tolerance=tolerance,
                                    max_component=2)
    assert "cluster exactly" in str(excinfo.value)


def test_the_plan_is_measured_before_any_clustering_and_says_what_the_work_will_be(
        signatures, tolerance):
    plan = plan_clustering(signatures[:120], metric=METRIC, tolerance=tolerance)
    record = plan.as_record()
    assert record["schema"] == IDENTITY_SCHEMA
    assert record["feasible"] is True
    assert record["estimated_distance_evaluations"] == sum(
        len(item) * (len(item) - 1) // 2 for item in plan.graph.components)
    assert record["estimated_distance_evaluations"] < 120 * 119 // 2
    assert "rather than the" in plan.note


def test_the_metric_and_tolerance_must_be_the_pair_that_was_calibrated(signatures, tolerance):
    other = SignatureMetric(AttributeWeights(1.0, 1.0, 1.0, 2.0))
    with pytest.raises(InvalidParameterError) as excinfo:
        cluster_signatures_scalable(signatures[:20], metric=other, tolerance=tolerance)
    assert "used to calibrate this tolerance" in str(excinfo.value)


def test_an_empty_input_is_refused_exactly_as_the_original_refuses_it(tolerance):
    with pytest.raises(InvalidParameterError) as excinfo:
        cluster_signatures_scalable([], metric=METRIC, tolerance=tolerance)
    assert "at least one signed constellation" in str(excinfo.value)
    with pytest.raises(InvalidParameterError):
        cluster_signatures([], metric=METRIC, tolerance=tolerance)


def test_mixed_families_are_refused_rather_than_related(signatures, tolerance):
    from src.analysis_engine.spectral_clustering import _points
    points = list(_points(signatures[:4]))
    odd = SignaturePoint(
        family=SignatureFamily(mode=points[0].family.mode, cardinality=3,
                               has_bearings=points[0].family.has_bearings,
                               scale_units=points[0].family.scale_units),
        geometry=(1.0, 1.0, 1.0),
        bearings=(10.0, 20.0, 30.0) if points[0].family.has_bearings else None,
        strengths=(1.0, 1.0, 1.0), scales=(1.0, 1.0, 1.0))
    with pytest.raises(InvalidParameterError) as excinfo:
        neighbour_graph(points + [odd], metric=METRIC, tolerance=tolerance.value)
    assert "one signature family" in str(excinfo.value)


def test_the_things_that_are_not_what_they_claim_to_be_are_refused(signatures, tolerance):
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    with pytest.raises(InvalidParameterError):
        component_bounds(object(), METRIC, 0.5)
    with pytest.raises(InvalidParameterError):
        component_bounds(point, object(), 0.5)
    with pytest.raises(InvalidParameterError):
        component_bounds(point, METRIC, -1.0)
    with pytest.raises(InvalidParameterError):
        component_bounds(point, METRIC, float("inf"))
    with pytest.raises(InvalidParameterError):
        neighbour_graph([], metric=METRIC, tolerance=0.5)
    with pytest.raises(InvalidParameterError):
        neighbour_graph(_points(signatures[:4]), metric=METRIC, tolerance=0.5, block=0)


def test_a_metric_with_no_positive_weight_on_any_available_block_is_refused(signatures):
    from src.analysis_engine.spectral_clustering import _points
    point = _points(signatures[:1])[0]
    # `AttributeWeights` refuses an all-zero metric before this module can see it, which is
    # the better place for that refusal; this pins where it happens rather than duplicating it.
    with pytest.raises(InvalidParameterError) as excinfo:
        AttributeWeights(0.0, 0.0, 0.0, 0.0)
    assert "calls every constellation the same pattern" in str(excinfo.value)
    # The refusal this module owns is the one the weights cannot catch: every positive weight
    # sitting on a block this particular signature family does not carry.
    bearings_only = SignatureMetric(AttributeWeights(0.0, 1.0, 0.0, 0.0))
    with pytest.raises(InvalidParameterError) as excinfo:
        component_bounds(point, bearings_only, 0.5)
    assert "at least one block" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 5: what it bought
# ---------------------------------------------------------------------------------------------


def test_the_decomposition_avoids_work_and_the_receipt_says_how_much(signatures, tolerance):
    from src.analysis_engine.spectral_clustering import _member_key, _points
    points = _points(sorted(signatures[:150], key=_member_key))
    graph = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value)
    every_pair = len(points) * (len(points) - 1) // 2
    assert graph.n_exact_distances < every_pair, "the box saved no distance computations"
    assert graph.as_record()["distances_not_computed"] == every_pair - graph.n_exact_distances
    assert graph.n_points == len(points)


def test_the_block_size_changes_the_work_but_never_the_answer(signatures, tolerance):
    from src.analysis_engine.spectral_clustering import _member_key, _points
    points = _points(sorted(signatures[:100], key=_member_key))
    wide = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value, block=4096)
    narrow = neighbour_graph(points, metric=METRIC, tolerance=tolerance.value, block=7)
    assert wide.components == narrow.components
    assert wide.n_unions == narrow.n_unions
