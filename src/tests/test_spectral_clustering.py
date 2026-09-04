"""T4E.3: measured approximate matching of noisy constellation signatures."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.analysis_engine.spectral_clustering import (
    CLUSTER_SCHEMA, METRIC_SCHEMA, AttributeWeights, SignatureMetric, SignaturePoint,
    calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_invariance import signature_for
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.feature import FeatureLocation, FeatureSet, Quantity, SpectralFeature
from src.core.tracking import MotionBounds, SearchVolume, Track, TrackingResult


ROW, COL = "row", "col"
AXES = (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
        AxisSpec(COL, "space", units="cells", periodic=False, ordinal=1))
BOUNDS = MotionBounds(max_doublings=0.5)
SCALENE = ((0.0, 0.0), (0.0, 30.0), (40.0, 0.0))
WEIGHTS = AttributeWeights(geometry=1.0, bearings=1.0, strengths=1.0, scales=1.0)
METRIC = SignatureMetric(WEIGHTS)


def _transform(points, *, rotate_deg=0.0, shift=(0.0, 0.0), scale=1.0):
    angle = math.radians(rotate_deg)
    output = []
    for row, col in points:
        row, col = row * scale, col * scale
        output.append((row * math.cos(angle) - col * math.sin(angle) + shift[0] + 200.0,
                       row * math.sin(angle) + col * math.cos(angle) + shift[1] + 200.0))
    return tuple(output)


def _signature(*, scale_invariant, points=SCALENE, rotate_deg=0.0, shift=(0.0, 0.0),
               position_scale=1.0, position_jitter=((0.0, 0.0),) * 3,
               magnitudes=(3.0, 5.0, 7.0), spatial_scales=(8.0, 8.0, 16.0),
               key_time=0.0, scale_units="cells"):
    located = _transform(points, rotate_deg=rotate_deg, shift=shift, scale=position_scale)
    located = tuple((row + jitter[0], col + jitter[1])
                    for (row, col), jitter in zip(located, position_jitter))
    features = []
    for index, ((row, col), magnitude, spatial_scale) in enumerate(
            zip(located, magnitudes, spatial_scales)):
        features.append(SpectralFeature(
            domain="synthetic", dataset="t4e3_acceptance", variable="amplitude",
            magnitude=Quantity(float(magnitude), None),
            location=FeatureLocation({ROW: row, COL: col}, AXES),
            time=float(key_time), time_units="frames", representation="manual",
            spatial_scale=Quantity(float(spatial_scale), scale_units),
            provenance={
                "scale_label": index + 1, "orientation_label": "LH",
                "threshold": 1.0, "threshold_sigma": 3.0, "phase": None,
                "alignment_cross_scale_comparable": True,
            }))
    tracks = tuple(Track(index, FeatureSet([feature]), {})
                   for index, feature in enumerate(features))
    tracking = TrackingResult(
        features=FeatureSet(features), associator="hungarian", alpha=0.05,
        bounds=BOUNDS, volume=SearchVolume({ROW: 1024.0, COL: 1024.0}, units="cells"),
        times=(float(key_time),), tracks=tracks)
    constellations = extract_constellations(tracking, cardinalities=(3,))
    assert len(constellations) == 1
    return signature_for(constellations[0], scale_invariant=scale_invariant)


def _calibration_replicates(scale_invariant):
    """Known same-configuration measurements spanning the declared 10% perturbation."""
    return (
        _signature(scale_invariant=scale_invariant, key_time=10.0),
        _signature(
            scale_invariant=scale_invariant,
            position_jitter=((0.0, 0.0), (0.0, 3.0), (-4.0, 0.0)),
            magnitudes=(3.3, 4.5, 7.7), spatial_scales=(8.8, 7.2, 17.6), key_time=11.0),
        _signature(
            scale_invariant=scale_invariant,
            position_jitter=((0.0, 0.0), (0.0, -3.0), (4.0, 0.0)),
            magnitudes=(2.7, 5.5, 6.3), spatial_scales=(7.2, 8.8, 14.4), key_time=12.0),
    )


def test_every_attribute_weight_must_be_finite_non_negative_and_not_all_zero():
    with pytest.raises(InvalidParameterError, match="finite non-negative"):
        AttributeWeights(geometry=-1.0, bearings=1.0, strengths=1.0, scales=1.0)
    with pytest.raises(InvalidParameterError, match="zero metric"):
        AttributeWeights(geometry=0.0, bearings=0.0, strengths=0.0, scales=0.0)


def test_metric_is_symmetric_dimensionless_and_reports_each_declared_block():
    left, right = _calibration_replicates(False)[:2]
    a, b = SignaturePoint.from_signature(left), SignaturePoint.from_signature(right)
    assert METRIC.distance(a, b) == pytest.approx(METRIC.distance(b, a))
    assert 0.0 < METRIC.distance(a, b) < 1.0
    assert set(METRIC.breakdown(a, b)) == {"geometry", "bearings", "strengths", "scales"}
    report = METRIC.describe()
    assert report["schema"] == METRIC_SCHEMA
    assert report["weights"] == WEIGHTS.as_mapping()


def test_distance_refuses_different_modes_cardinalities_axis_states_and_units():
    specific = SignaturePoint.from_signature(_signature(scale_invariant=False))
    invariant = SignaturePoint.from_signature(_signature(scale_invariant=True))
    with pytest.raises(InvalidParameterError, match="different measured quantities"):
        METRIC.distance(specific, invariant)
    other_units = replace(specific, family=replace(specific.family, scale_units="m"))
    with pytest.raises(InvalidParameterError, match="different measured quantities"):
        METRIC.distance(specific, other_units)
    no_axis = replace(specific, family=replace(specific.family, has_bearings=False), bearings=None)
    with pytest.raises(InvalidParameterError, match="different measured quantities"):
        METRIC.distance(specific, no_axis)


def test_a_weight_only_on_an_unavailable_bearing_block_is_refused():
    signature = _signature(scale_invariant=True)
    point = SignaturePoint.from_signature(replace(signature, bearings=None))
    metric = SignatureMetric(AttributeWeights(
        geometry=0.0, bearings=1.0, strengths=0.0, scales=0.0))
    with pytest.raises(InvalidParameterError, match="available"):
        metric.distance(point, point)


def test_tolerance_requires_two_known_same_configuration_measurements():
    with pytest.raises(InvalidParameterError, match="at least two"):
        calibrate_signature_tolerance([_signature(scale_invariant=False)], metric=METRIC)


def test_tolerance_refuses_to_average_a_mode_or_unit_change_into_noise():
    with pytest.raises(InvalidParameterError, match="not measurement jitter"):
        calibrate_signature_tolerance(
            [_signature(scale_invariant=False), _signature(scale_invariant=True)], metric=METRIC)
    metres = replace(_signature(scale_invariant=False, key_time=2.0), scale_units="m")
    with pytest.raises(InvalidParameterError, match="not measurement jitter"):
        calibrate_signature_tolerance(
            [_signature(scale_invariant=False), metres], metric=METRIC)


def test_tolerance_is_the_largest_all_pair_replicate_distance_and_reuses_tg34_record():
    replicates = _calibration_replicates(False)
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC)
    assert tolerance.value == max(tolerance.pair_distances)
    assert len(tolerance.pair_distances) == 3
    assert tolerance.measurement.n_replicates == 3
    assert tolerance.measurement.components == len(SignaturePoint.from_signature(replicates[0]).vector())
    assert "measured noise floor" in tolerance.measurement.basis


def test_node_correspondence_is_researched_again_when_noise_flips_canonical_order():
    left, _, right = _calibration_replicates(False)
    assert left.order != right.order  # This is the boundary that exposed D91.
    left_point = SignaturePoint.from_signature(left)
    right_point = SignaturePoint.from_signature(right)
    aligned, breakdown, distance = METRIC.alignment(left_point, right_point)
    assert aligned.scales != right_point.scales
    assert breakdown["scales"] < 0.11
    assert distance < 0.11


def test_location_rotation_and_ten_percent_attribute_jitter_land_in_one_cluster():
    calibration = _calibration_replicates(False)
    tolerance = calibrate_signature_tolerance(calibration, metric=METRIC)
    base = _signature(scale_invariant=False, key_time=20.0)
    moved = _signature(scale_invariant=False, shift=(317.0, -83.0), key_time=21.0)
    rotated = _signature(scale_invariant=False, rotate_deg=73.0, key_time=22.0)
    jittered = _signature(
        scale_invariant=False, rotate_deg=31.0, shift=(17.0, 29.0),
        position_jitter=((0.0, 0.0), (1.5, -1.5), (-2.0, 2.0)),
        magnitudes=(3.24, 4.6, 7.35), spatial_scales=(8.4, 7.6, 16.8), key_time=23.0)
    catalogue = cluster_signatures(
        [jittered, moved, base, rotated], metric=METRIC, tolerance=tolerance)
    assert len(catalogue) == 1
    assert catalogue.patterns[0].support == 4


def test_absolute_rescaling_joins_only_when_scale_invariance_is_enabled():
    specific_base = _signature(scale_invariant=False, key_time=30.0)
    specific_scaled = _signature(
        scale_invariant=False, position_scale=2.0,
        spatial_scales=(16.0, 16.0, 32.0), key_time=31.0)
    specific_tolerance = calibrate_signature_tolerance(
        _calibration_replicates(False), metric=METRIC)
    assert len(cluster_signatures(
        [specific_base, specific_scaled], metric=METRIC,
        tolerance=specific_tolerance)) == 2

    invariant_base = _signature(scale_invariant=True, key_time=32.0)
    invariant_scaled = _signature(
        scale_invariant=True, position_scale=2.0,
        spatial_scales=(16.0, 16.0, 32.0), key_time=33.0)
    invariant_tolerance = calibrate_signature_tolerance(
        _calibration_replicates(True), metric=METRIC)
    invariant = cluster_signatures(
        [invariant_base, invariant_scaled], metric=METRIC,
        tolerance=invariant_tolerance)
    assert len(invariant) == 1
    assert invariant.patterns[0].support == 2


def test_a_tolerance_cannot_be_reused_after_the_declared_weights_change():
    tolerance = calibrate_signature_tolerance(_calibration_replicates(False), metric=METRIC)
    changed = SignatureMetric(AttributeWeights(
        geometry=4.0, bearings=1.0, strengths=1.0, scales=1.0))
    with pytest.raises(InvalidParameterError, match="exact declared metric"):
        cluster_signatures(
            [_signature(scale_invariant=False)], metric=changed, tolerance=tolerance)


def test_complete_link_prevents_a_bridge_chain_from_becoming_one_pattern():
    base = _signature(scale_invariant=False)
    metric = SignatureMetric(AttributeWeights(
        geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))

    def geometry_factor(value, time):
        return replace(base, key=("bridge", time), time=float(time),
                       geometry=tuple(component * value for component in base.geometry))

    calibration = [geometry_factor(1.0, 100), geometry_factor(1.1052631579, 101)]
    tolerance = calibrate_signature_tolerance(calibration, metric=metric)
    signatures = [geometry_factor(1.0, 1), geometry_factor(1.09, 2), geometry_factor(1.18, 3)]
    assert metric.distance(SignaturePoint.from_signature(signatures[0]),
                           SignaturePoint.from_signature(signatures[1])) < tolerance.value
    assert metric.distance(SignaturePoint.from_signature(signatures[1]),
                           SignaturePoint.from_signature(signatures[2])) < tolerance.value
    assert metric.distance(SignaturePoint.from_signature(signatures[0]),
                           SignaturePoint.from_signature(signatures[2])) > tolerance.value
    catalogue = cluster_signatures(signatures, metric=metric, tolerance=tolerance)
    assert sorted(pattern.support for pattern in catalogue) == [1, 2]


def test_clustering_and_pattern_ids_are_deterministic_under_input_permutation():
    tolerance = calibrate_signature_tolerance(_calibration_replicates(False), metric=METRIC)
    signatures = [
        _signature(scale_invariant=False, key_time=41.0),
        _signature(scale_invariant=False, shift=(20.0, 50.0), key_time=42.0),
        _signature(scale_invariant=False, position_scale=2.0,
                   spatial_scales=(16.0, 16.0, 32.0), key_time=43.0),
    ]
    forward = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance).describe()
    reverse = cluster_signatures(list(reversed(signatures)), metric=METRIC,
                                 tolerance=tolerance).describe()
    assert forward == reverse


def test_every_pattern_exposes_a_centroid_and_calibrated_and_observed_radii():
    tolerance = calibrate_signature_tolerance(_calibration_replicates(False), metric=METRIC)
    catalogue = cluster_signatures(
        [_signature(scale_invariant=False, key_time=50.0),
         _signature(scale_invariant=False, rotate_deg=15.0, key_time=51.0)],
        metric=METRIC, tolerance=tolerance)
    pattern = catalogue.patterns[0]
    assert pattern.tolerance_radius == tolerance.value
    assert pattern.observed_radius <= pattern.tolerance_radius
    assert len(pattern.centroid.vector()) == tolerance.measurement.components


def test_report_states_algorithm_weights_and_the_t4e4_claim_boundary():
    tolerance = calibrate_signature_tolerance(_calibration_replicates(True), metric=METRIC)
    report = cluster_signatures(
        [_signature(scale_invariant=True)], metric=METRIC, tolerance=tolerance).describe()
    assert report["schema"] == CLUSTER_SCHEMA
    assert report["algorithm"] == "deterministic complete-link agglomeration"
    assert report["metric"]["weights"] == WEIGHTS.as_mapping()
    assert report["tolerance"]["metric_digest"] == METRIC.digest
    assert "not minimum-support mining" in report["patterns"][0]["claim_boundary"]


def test_an_empty_clustering_request_is_refused_instead_of_reporting_zero_patterns():
    tolerance = calibrate_signature_tolerance(_calibration_replicates(False), metric=METRIC)
    with pytest.raises(InvalidParameterError, match="at least one"):
        cluster_signatures([], metric=METRIC, tolerance=tolerance)
