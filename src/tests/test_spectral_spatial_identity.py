"""T4E.8: geometry identity has no detector-band or coefficient-magnitude input.

Analytic configurations exercise the scientific definition independently of any ERA5 labels.
The acquired-record evaluation is exploratory and is not a passing test expectation.
"""
from dataclasses import replace
import itertools
import json

import numpy as np
import pytest

from src.analysis_engine.spectral_invariance import (
    SIGNATURE_MODES, sign_constellations, signature_for,
)
from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, SignaturePoint, calibrate_signature_tolerance,
    cluster_signatures,
)
from src.analysis_engine.spectral_identity import pairwise_distance_matrix, cluster_signatures_scalable
from src.analysis_engine.spectral_identity_audit import labelled_errors, recall_radius, radius_feasibility
from src.core.errors import InvalidParameterError
from src.tests.test_spectral_invariance import _configuration, _transform, SCALENE

SCOPE = "analytic-cartesian-grid/v1"
METRIC = SignatureMetric(AttributeWeights(1.0, 1.0, 1.0, 1.0))


def spatial(configuration, scope=SCOPE):
    return signature_for(configuration, mode="spatial_geometry", comparison_scope=scope)


@pytest.mark.parametrize("points", [((0., 0.), (3., 4.)), SCALENE])
def test_band_and_magnitude_changes_do_not_change_identity(points):
    original = spatial(_configuration(points))
    changed = spatial(_configuration(points, scales=[1, 7, 2][:len(points)],
                                     magnitudes=[800., 0.2, 50.][:len(points)]))
    assert original.scales != changed.scales
    assert original.strengths_carried != changed.strengths_carried
    assert original.vector() == changed.vector()
    assert METRIC.distance(SignaturePoint.from_signature(original),
                           SignaturePoint.from_signature(changed)) == 0


def test_pair_retains_analytic_separation_and_distinguishes_size():
    one = spatial(_configuration(((0., 0.), (3., 4.))))
    two = spatial(_configuration(((0., 0.), (6., 8.))))
    assert one.geometry == (5.,)
    assert two.geometry == (10.,)
    assert one.vector() == (5.,)
    assert METRIC.distance(SignaturePoint.from_signature(one),
                           SignaturePoint.from_signature(two)) == pytest.approx(2 / 3)
    assert not one.cross_domain_comparable
    assert "rescaling" not in one.invariant_to


@pytest.mark.parametrize("angle", [0., 31., 90., 180.])
def test_geometry_is_translation_rotation_reflection_and_permutation_invariant(angle):
    original = spatial(_configuration(SCALENE))
    moved = _configuration(_transform(SCALENE, rotate_deg=angle, shift=(97., -19.), reflect=True))
    for order in itertools.permutations(range(3)):
        permuted = replace(moved, nodes=tuple(moved.nodes[i] for i in order),
                           features=tuple(moved.features[i] for i in order))
        assert spatial(permuted).vector() == pytest.approx(original.vector(), abs=1e-11)


def test_missing_strength_normalisation_does_not_prevent_spatial_identity():
    configuration = _configuration(SCALENE)
    missing = replace(configuration, nodes=tuple(replace(n, strength_sigma=None)
                                                 for n in configuration.nodes))
    result = spatial(missing)
    assert result.strengths_carried == (None, None, None)
    assert result.vector() == spatial(configuration).vector()
    with pytest.raises(InvalidParameterError):
        signature_for(missing)


def test_carried_attributes_are_absent_from_both_comparison_surfaces():
    signature = spatial(_configuration(SCALENE))
    point = SignaturePoint.from_signature(signature)
    assert point.vector() == signature.vector()
    assert point.strengths == point.scales == ()
    assert signature.scales
    assert signature.strengths_carried
    receipt = json.loads(json.dumps(signature.describe()))
    assert receipt["comparison_scope"] == SCOPE
    assert receipt["geometry_units"] == "cells"
    assert receipt["scale_block_in_vector"].startswith("absent")
    assert receipt["comparable_blocks"] == ["geometry", "bearings"]
    assert "not ground truth" in receipt["claim_boundary"]


@pytest.mark.parametrize("scope", [None, "", "   ", 7, False, []])
def test_record_scope_is_required_before_signing(scope):
    with pytest.raises(InvalidParameterError, match="scope"):
        spatial(_configuration(SCALENE), scope)


def test_incompatible_modes_and_scopes_cannot_share_a_radius():
    configuration = _configuration(SCALENE)
    reference = SignaturePoint.from_signature(spatial(configuration))
    for other in (spatial(configuration, "different-grid/v1"), signature_for(configuration),
                  signature_for(configuration, scale_invariant=True)):
        with pytest.raises(InvalidParameterError, match="family"):
            METRIC.distance(reference, SignaturePoint.from_signature(other))


def test_reusing_scope_cannot_cross_a_source_boundary():
    configuration = _configuration(SCALENE)
    changed = replace(configuration, features=tuple(replace(f, dataset="another_record")
                                                    for f in configuration.features))
    left = SignaturePoint.from_signature(spatial(configuration))
    right = SignaturePoint.from_signature(spatial(changed))
    assert left.family.comparison_scope == right.family.comparison_scope
    with pytest.raises(InvalidParameterError, match="family"):
        METRIC.distance(left, right)


def test_carried_only_weight_cannot_turn_into_a_zero_distance():
    point = SignaturePoint.from_signature(spatial(_configuration(SCALENE)))
    metric = SignatureMetric(AttributeWeights(0., 0., 1., 1.))
    with pytest.raises(InvalidParameterError, match="positive weight"):
        metric.distance(point, point)
    with pytest.raises(InvalidParameterError, match="positive weight"):
        pairwise_distance_matrix([point], metric)


def test_forged_comparable_carried_blocks_are_refused():
    point = SignaturePoint.from_signature(spatial(_configuration(SCALENE)))
    with pytest.raises(InvalidParameterError):
        replace(point, strengths=(1., 1., 1.))
    with pytest.raises(InvalidParameterError):
        replace(point, scales=(1., 1., 1.))
    with pytest.raises(InvalidParameterError):
        replace(point, family=replace(point.family, comparison_scope=None))


def test_conflicting_toggle_is_not_silently_ignored():
    with pytest.raises(InvalidParameterError, match="consistent"):
        signature_for(_configuration(SCALENE), mode="spatial_geometry",
                      comparison_scope=SCOPE, scale_invariant=True)


def test_spatial_mode_is_discoverable_and_does_not_replace_legacy_modes():
    assert set(SIGNATURE_MODES.names()) == {"scale_specific", "scale_invariant", "spatial_geometry"}
    assert SIGNATURE_MODES.entry("spatial_geometry").capabilities["band_independent"] is True
    assert signature_for(_configuration(SCALENE)).mode == "scale_specific"


def test_periodic_bearings_are_refused_without_a_planar_convention():
    configuration = _configuration(SCALENE)
    features = tuple(replace(f, location=replace(f.location,
                     axes=tuple(replace(a, periodic=True) for a in f.location.axes)))
                     for f in configuration.features)
    with pytest.raises(InvalidParameterError, match="nonperiodic"):
        spatial(replace(configuration, features=features))


def test_unknown_or_unscoped_batch_fails_before_swallowing_a_configuration():
    # The public batch validates its declaration before it touches the iterable.
    with pytest.raises(InvalidParameterError, match="scope"):
        sign_constellations((), mode="spatial_geometry")
    with pytest.raises(InvalidParameterError, match="consistent"):
        sign_constellations((), mode="spatial_geometry", comparison_scope=SCOPE, scale_invariant=True)


def test_accelerated_metric_preserves_the_new_definition():
    signatures = [spatial(_configuration(((0., 0.), (3. * f, 4. * f))))
                  for f in (1., 1.01, 2., 2.02)]
    points = [SignaturePoint.from_signature(s) for s in signatures]
    expected = np.array([[METRIC.distance(a, b) for b in points] for a in points])
    np.testing.assert_allclose(pairwise_distance_matrix(points, METRIC, block=2), expected,
                               rtol=1e-14, atol=1e-14)


def test_spatial_units_do_not_get_relabelled_as_band_units():
    signature = spatial(_configuration(SCALENE, scale_units="m"))
    assert signature.geometry_units == "cells"
    assert SignaturePoint.from_signature(signature).family.scale_units == "cells"


def test_clustering_consumes_spatial_identity_and_rejects_a_legacy_radius():
    signatures = [replace(spatial(_configuration(((0., 0.), (3. * f, 4. * f)))),
                          time=float(i), key=(float(i), 0, 1))
                  for i, f in enumerate((1., 1.01, 2., 2.02))]
    tolerance = calibrate_signature_tolerance(signatures[:2], metric=METRIC,
                                               contrast=signatures[2:])
    original = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
    fast = cluster_signatures_scalable(signatures, metric=METRIC, tolerance=tolerance)
    assert len(original.patterns) >= 2
    assert fast.describe() == original.describe()
    legacy = [signature_for(_configuration(SCALENE)),
              signature_for(_configuration(_transform(SCALENE, shift=(1., 0.))))]
    old_radius = calibrate_signature_tolerance(legacy, metric=METRIC)
    for cluster in (cluster_signatures, cluster_signatures_scalable):
        with pytest.raises(InvalidParameterError, match="family"):
            cluster(signatures, metric=METRIC, tolerance=old_radius)


def test_empirical_errors_include_radius_ties_and_auc_half_ties():
    result = labelled_errors([0., 1., 2.], [1., 2., 3.], 1.)
    assert result["false_split_count"] == 1
    assert result["false_admission_count"] == 1
    assert result["false_split_rate"] == result["false_admission_rate"] == 1 / 3
    assert result["auc"] == pytest.approx(7 / 9)
    assert recall_radius([3., 1., 2.], 2 / 3) == 2.
    assert recall_radius([3., 1., 2.], .5) == 2.


def test_empty_label_populations_remain_unmeasured():
    assert recall_radius([], .9) is None
    assert labelled_errors([], [1.], .5)["false_split_rate"] is None
    assert labelled_errors([1.], [], .5)["false_admission_rate"] is None
    report = labelled_errors([1.], [2.], None)
    assert report["false_split_rate"] is report["false_admission_rate"] is None
    assert report["auc"] == 1.


@pytest.mark.parametrize("bad", [[-1.], [float("nan")], [float("inf")], [[1.]]])
def test_invalid_distances_are_refused(bad):
    with pytest.raises(InvalidParameterError):
        labelled_errors(bad, [1.], .5)
    with pytest.raises(InvalidParameterError):
        labelled_errors([1.], bad, .5)
    with pytest.raises(InvalidParameterError):
        recall_radius(bad, .9)


@pytest.mark.parametrize("bad", [0., -1., 1.1, float("nan"), float("inf")])
def test_invalid_recall_is_refused(bad):
    with pytest.raises(InvalidParameterError):
        recall_radius([1.], bad)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_radius_is_refused(bad):
    with pytest.raises(InvalidParameterError):
        labelled_errors([1.], [2.], bad)


def test_empirical_feasibility_is_a_monotone_bound_not_a_radius_search():
    yes = radius_feasibility([1., 1.], [2., 3.], max_split=.1, max_admission=.1)
    assert yes["any_radius_meets_both_empirical_bounds"] is True
    no = radius_feasibility([2., 3.], [1., 2.], max_split=.1, max_admission=.1)
    assert no["any_radius_meets_both_empirical_bounds"] is False
    assert no["smallest_split_feasible_radius"] == 3.
    assert radius_feasibility([], [1.], max_split=.1, max_admission=.1)[
        "any_radius_meets_both_empirical_bounds"] is None
    assert radius_feasibility([1.], [], max_split=.1, max_admission=.1)[
        "any_radius_meets_both_empirical_bounds"] is None
    assert radius_feasibility(list(range(1, 11)), [12.], max_split=.7, max_admission=.1)[
        "smallest_split_feasible_radius"] == 3.


@pytest.mark.parametrize("value", [-.1, 1., float("nan"), float("inf")])
def test_invalid_feasibility_bounds_are_refused(value):
    with pytest.raises(InvalidParameterError):
        radius_feasibility([1.], [2.], max_split=value, max_admission=.1)
    with pytest.raises(InvalidParameterError):
        radius_feasibility([1.], [2.], max_split=.1, max_admission=value)
