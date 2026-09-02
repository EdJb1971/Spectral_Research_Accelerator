"""T4E.2: the invariant signature, and the axis the benchmark does not have.

1.  **Invariance is measured, not declared.** The same configuration is translated, rotated,
    reflected and relabelled, and the signature vector is asserted identical to floating-point
    precision in both modes. Nothing here is a tolerance.
2.  **The toggle is priced.** Turning scale invariance on costs 87 of the vortex pass's 135
    constellations -- every pair -- and buys the only mode whose entries are all dimensionless.
    The two modes are separated by an estimator that is wrong by the amount TG3.4 measured.
3.  **The benchmark's own configuration has no principal axis.** `planted_configuration` is
    equilateral. Over 24 field-noise realisations its anisotropy stays under 1.05 while the
    recovered axis angle scatters across the half-circle, so a bearing block without a guard
    would have emitted noise on the exact configuration the roadmap nominates for invariance.
4.  **The canonical order is a minimisation over correspondences, not a sort.** A test shows
    the difference: two configurations that agree on independently sorted blocks with no
    single correspondence making both true at once.
5.  **Refusals are by name and reachable.** Each one is pinned by the call that triggers it,
    and the pair refusals are pinned twice -- once for the shape and once for the axis --
    because they refuse for two different reasons.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_invariance import (
    AXIS_FLOOR_REPLICATES, AXIS_ISOTROPY_FLOOR, MODE_INVARIANCE, SCALE_MODES,
    SIGNATURE_SCHEMA, admit_axis, calibrate_axis_admission, compare_scale_modes,
    describe_signatures, principal_axis, sign_constellations, signature_for,
)
from src.analysis_engine.spectral_tracking import COL, ROW, track_spectral_features
from src.benchmarks.core import get_benchmark
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.extraction import ExtractionField, extract
from src.core.feature import (
    FeatureLocation, FeatureSet, Quantity, SemanticComparisonError, SpectralFeature,
)
from src.core.tracking import MotionBounds, SearchVolume, Track, TrackingResult
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import decompose_sequence

WAVELET = "haar"
LEVELS = 5
TRACKED_SCALES = [4, 5]
THRESHOLD_SIGMA = 4.0
BOUNDS = MotionBounds(max_doublings=0.5)

AXES = (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
        AxisSpec(COL, "space", units="cells", periodic=False, ordinal=1))
METRE_AXES = (AxisSpec(ROW, "space", units="m", periodic=False, ordinal=0),
              AxisSpec(COL, "space", units="m", periodic=False, ordinal=1))
REPRESENTATION = "wavelet[swt,wavelet=haar,levels=5]"

#: The `planted_configuration` benchmark's own axes, in its own units.
PLANTED_AXES = (AxisSpec("row", "space", "cells", ordinal=0),
                AxisSpec("col", "space", "cells", ordinal=1))

_CACHE = {}


# ------------------------------------------------------------------------------- fixtures


def _tracked():
    """The T4D.2 acceptance pass, tracked once and reused."""
    if "tracked" not in _CACHE:
        built = get_benchmark("advected_vortex_sequence").make()
        sequence = FieldSequence(built.fields, np.arange(len(built.fields), dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET}).select(
                scales=TRACKED_SCALES)
        _CACHE["tracked"] = track_spectral_features(
            field, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", bounds=BOUNDS,
            threshold_sigma=THRESHOLD_SIGMA)
    return _CACHE["tracked"]


def _extracted():
    if "extracted" not in _CACHE:
        _CACHE["extracted"] = extract_constellations(_tracked(), cardinalities=(2, 3))
    return _CACHE["extracted"]


def _feature(time, row, col, magnitude, *, scale=4, orientation="LH", dataset="manual",
             threshold=1.0, sigma=4.0, axes=AXES, scale_units="cells"):
    provenance = {"scale_label": scale, "orientation_label": orientation,
                  "threshold": threshold, "threshold_sigma": sigma, "phase": None,
                  "alignment_cross_scale_comparable": True}
    return SpectralFeature(
        domain="synthetic", dataset=dataset, variable="amplitude",
        magnitude=Quantity(float(magnitude), None),
        location=FeatureLocation({ROW: float(row), COL: float(col)}, axes),
        time=float(time), time_units="frames", representation=REPRESENTATION,
        spatial_scale=Quantity(float(2 ** (scale - 1)), scale_units),
        provenance=provenance)


def _configuration(points, *, magnitudes=None, scales=None, axes=AXES,
                   scale_units="cells", time=0.0):
    """One frame constellation over hand-placed features, so the truth is exact."""
    magnitudes = magnitudes or [3.0, 5.0, 7.0][:len(points)]
    scales = scales or [4, 4, 5][:len(points)]
    features = [_feature(time, row, col, magnitude, scale=scale, axes=axes,
                         orientation="LH" if index % 2 == 0 else "HL",
                         scale_units=scale_units)
                for index, ((row, col), magnitude, scale)
                in enumerate(zip(points, magnitudes, scales))]
    tracks = [Track(index, FeatureSet([item]), {}) for index, item in enumerate(features)]
    result = TrackingResult(
        features=FeatureSet(list(features)), associator="hungarian", alpha=0.05,
        bounds=BOUNDS, volume=SearchVolume({ROW: 4096.0, COL: 4096.0}, units="cells"),
        times=(float(time),), tracks=tuple(tracks))
    extracted = extract_constellations(result, cardinalities=(len(points),))
    assert len(extracted) == 1
    return extracted[0]


#: A scalene triangle: no symmetry, so exactly one correspondence recovers it.
SCALENE = ((0.0, 0.0), (0.0, 30.0), (40.0, 0.0))


def _transform(points, *, rotate_deg=0.0, shift=(0.0, 0.0), scale=1.0, reflect=False):
    angle = math.radians(rotate_deg)
    out = []
    for row, col in points:
        row, col = row * scale, col * scale
        if reflect:
            col = -col
        out.append((row * math.cos(angle) - col * math.sin(angle) + shift[0] + 2048.0,
                    row * math.sin(angle) + col * math.cos(angle) + shift[1] + 2048.0))
    return tuple(out)


def _vector(points, *, scale_invariant, **kwargs):
    return signature_for(_configuration(_transform(points, **kwargs)),
                         scale_invariant=scale_invariant).vector()


def _planted_positions(root_seed, **overrides):
    """Extract the three planted blobs and return their row/col positions."""
    bench = get_benchmark("planted_configuration")
    data = np.asarray(bench.make(root_seed, **overrides).data.numpy(), dtype=np.float64)
    frame = ExtractionField(
        values=data, axes=PLANTED_AXES, domain="synthetic",
        dataset="planted_configuration", variable="amplitude", units=None,
        time=0.0, time_units="frames", representation="identity")
    features = list(extract(frame, n_surrogates=99, seed=17))
    assert len(features) == 3
    return [[item.location.coords["row"], item.location.coords["col"]] for item in features]


def _planted_replicates():
    key = "planted"
    if key not in _CACHE:
        _CACHE[key] = [_planted_positions(seed)
                       for seed in range(1, AXIS_FLOOR_REPLICATES + 1)]
    return _CACHE[key]


# =============================================================================== section 1
# The principal axis is measured, and it agrees with the eigendecomposition it stands for.


def test_the_principal_axis_agrees_with_the_covariance_eigendecomposition():
    rng = np.random.default_rng(3)
    for _ in range(50):
        points = rng.normal(size=(3, 2)) * np.array([4.0, 1.0])
        ratio, angle = principal_axis(points)
        centred = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(centred.T @ centred / 3.0)
        order = np.argsort(values)[::-1]
        expected_ratio = values[order][0] / values[order][1]
        vector = vectors[:, order[0]]
        expected_angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
        assert ratio == pytest.approx(expected_ratio, rel=1e-9)
        difference = abs(angle - expected_angle) % 180.0
        assert min(difference, 180.0 - difference) < 1e-7


def test_exactly_collinear_points_report_an_infinite_anisotropy_rather_than_a_failure():
    ratio, angle = principal_axis([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    assert math.isinf(ratio)
    assert angle == pytest.approx(45.0)


def test_points_at_one_place_have_no_direction_between_them():
    with pytest.raises(InvalidParameterError) as error:
        principal_axis([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])
    assert "no direction between them" in str(error.value)


def test_a_principal_axis_in_three_dimensions_is_refused_rather_than_projected():
    with pytest.raises(InvalidParameterError) as error:
        principal_axis([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
    assert "three dimensions is a different quantity" in str(error.value)


# =============================================================================== section 2
# The benchmark the roadmap supplies for invariance is the one with no axis.


def test_the_planted_benchmark_is_isotropic_and_its_axis_angle_is_noise():
    floor, report = calibrate_axis_admission(_planted_replicates())
    assert report["replicates"] == AXIS_FLOOR_REPLICATES
    # The elongation never gets far from 1: this is an equilateral triangle.
    assert report["anisotropy_max"] < 1.05
    assert report["anisotropy_min"] > 1.0
    # And the angle it implies wanders across the whole half-circle.
    assert report["angle_min_deg"] < 5.0
    assert report["angle_max_deg"] > 150.0
    assert report["angle_circular_sd_deg"] > 30.0
    assert floor == pytest.approx(report["anisotropy_max"])


def test_the_isotropy_floor_in_the_module_is_the_number_that_measurement_produced():
    floor, _ = calibrate_axis_admission(_planted_replicates())
    assert floor == pytest.approx(AXIS_ISOTROPY_FLOOR, abs=5e-5)


def test_a_configuration_at_the_benchmarks_own_anisotropy_is_refused_an_axis():
    points = _planted_replicates()[0]
    refused = admit_axis(_configuration([(row, col) for row, col in points]))
    assert refused.admitted is False
    assert refused.angle_deg is None
    assert "does not clear the isotropy floor" in refused.refusal


def test_the_vortex_triples_clear_the_floor_by_two_orders_of_magnitude():
    ratios = [admit_axis(item).anisotropy
              for item in _extracted().of_cardinality(3)]
    assert len(ratios) == 48
    assert min(ratios) > 85.0
    assert all(admit_axis(item).admitted for item in _extracted().of_cardinality(3))


def test_calibrating_on_one_realisation_is_refused_because_a_floor_is_about_spread():
    with pytest.raises(InvalidParameterError) as error:
        calibrate_axis_admission([_planted_replicates()[0]])
    assert "one of them has none" in str(error.value)


def test_calibrating_on_collinear_replicates_is_refused_rather_than_returning_infinity():
    collinear = [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]] * 3
    with pytest.raises(InvalidParameterError) as error:
        calibrate_axis_admission(collinear)
    assert "refuses every configuration there is" in str(error.value)


# =============================================================================== section 3
# Invariance, measured on exact geometry where the truth has no noise in it.


@pytest.mark.parametrize("scale_invariant", [False, True])
def test_moving_the_configuration_does_not_move_its_signature(scale_invariant):
    reference = _vector(SCALENE, scale_invariant=scale_invariant)
    moved = _vector(SCALENE, scale_invariant=scale_invariant, shift=(311.0, -207.0))
    assert moved == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize("scale_invariant", [False, True])
@pytest.mark.parametrize("angle", [17.0, 90.0, 233.5])
def test_turning_the_configuration_does_not_turn_its_signature(scale_invariant, angle):
    reference = _vector(SCALENE, scale_invariant=scale_invariant)
    turned = _vector(SCALENE, scale_invariant=scale_invariant, rotate_deg=angle)
    assert turned == pytest.approx(reference, abs=1e-8)


@pytest.mark.parametrize("scale_invariant", [False, True])
def test_the_signature_is_invariant_to_reflection_because_the_grid_declares_no_orientation(
        scale_invariant):
    reference = _vector(SCALENE, scale_invariant=scale_invariant)
    mirrored = _vector(SCALENE, scale_invariant=scale_invariant, reflect=True)
    assert mirrored == pytest.approx(reference, abs=1e-8)
    assert "reflection" in MODE_INVARIANCE[
        "scale_invariant" if scale_invariant else "scale_specific"]


@pytest.mark.parametrize("scale_invariant", [False, True])
def test_listing_the_same_tracks_in_another_order_gives_the_same_signature(scale_invariant):
    reference = _vector(SCALENE, scale_invariant=scale_invariant)
    for order in ((1, 2, 0), (2, 0, 1), (2, 1, 0)):
        shuffled = _configuration([SCALENE[index] for index in order],
                                  magnitudes=[[3.0, 5.0, 7.0][index] for index in order],
                                  scales=[[4, 4, 5][index] for index in order])
        moved = signature_for(shuffled, scale_invariant=scale_invariant).vector()
        assert moved == pytest.approx(reference, abs=1e-9)


def test_a_uniform_rescaling_leaves_the_scale_free_shape_alone():
    reference = _vector(SCALENE, scale_invariant=True)
    for factor in (0.5, 2.0, 7.5):
        assert _vector(SCALENE, scale_invariant=True, scale=factor) == pytest.approx(
            reference, abs=1e-9)


def test_the_scale_specific_mode_moves_when_the_scale_estimate_drifts_and_the_other_does_not():
    """The difference between the two modes, priced off the drift TG3.4 measured.

    On exact geometry both modes survive a rescaling, because a perfect estimator returns a
    perfectly rescaled scale. The estimator is not perfect: TG3.4 measured `distance` moving
    4.8% at `scale_factor=3` against a 3.3% noise floor, because the extractor's scale
    estimate runs high at small sigma and low at large. Here the positions are rescaled by
    two and the recorded scales are left where they were -- an estimator that missed the
    rescaling entirely -- and only one of the two modes notices.
    """
    reference_positions = _transform(SCALENE)
    stretched = _transform(SCALENE, scale=2.0)
    held = _configuration(stretched, scales=[4, 4, 5])
    base = _configuration(reference_positions, scales=[4, 4, 5])

    invariant = signature_for(held, scale_invariant=True).vector()
    assert invariant == pytest.approx(
        signature_for(base, scale_invariant=True).vector(), abs=1e-9)

    specific = np.array(signature_for(held, scale_invariant=False).geometry)
    specific_base = np.array(signature_for(base, scale_invariant=False).geometry)
    assert specific == pytest.approx(2.0 * specific_base, rel=1e-9)


def test_scaling_the_amplitudes_leaves_the_strength_block_alone():
    reference = _vector(SCALENE, scale_invariant=True)
    louder = signature_for(
        _configuration(_transform(SCALENE), magnitudes=[30.0, 50.0, 70.0]),
        scale_invariant=True).vector()
    assert louder == pytest.approx(reference, abs=1e-9)


def test_a_different_triangle_gives_a_different_signature():
    reference = _vector(SCALENE, scale_invariant=True)
    other = signature_for(
        _configuration(_transform(((0.0, 0.0), (0.0, 30.0), (12.0, 15.0)))),
        scale_invariant=True).vector()
    assert other != pytest.approx(reference, abs=1e-3)


# =============================================================================== section 4
# The canonical order is a minimisation over correspondences, not a sort of each block.


def test_the_canonical_order_is_recorded_and_the_blocks_follow_it():
    signature = signature_for(_configuration(_transform(SCALENE)), scale_invariant=True)
    assert sorted(signature.order) == [0, 1, 2]
    assert signature.track_ids == tuple(
        _configuration(_transform(SCALENE)).track_ids[index] for index in signature.order)
    assert signature.bands == tuple(
        _configuration(_transform(SCALENE)).bands[index] for index in signature.order)


def test_sorting_the_blocks_separately_would_match_two_configurations_a_correspondence_cannot():
    """Why the vector is minimised whole rather than sorted block by block.

    Both configurations are the same scalene triangle and carry the same three strengths. They
    differ only in *which vertex* holds which strength, and no relabelling of one produces the
    other. Sorted independently, the geometry blocks agree and the strength blocks agree, so a
    sorted signature would call them the same configuration. Minimised whole, they do not
    agree, because there is no single correspondence that makes both blocks true at once.
    """
    left = _configuration(_transform(SCALENE), magnitudes=[3.0, 5.0, 7.0])
    right = _configuration(_transform(SCALENE), magnitudes=[5.0, 3.0, 7.0])
    one = signature_for(left, scale_invariant=True)
    two = signature_for(right, scale_invariant=True)

    assert sorted(one.geometry) == pytest.approx(sorted(two.geometry), abs=1e-9)
    assert sorted(one.strengths) == pytest.approx(sorted(two.strengths), abs=1e-9)
    assert one.vector() != pytest.approx(two.vector(), abs=1e-6)


# =============================================================================== section 5
# The toggle, and what it costs on the record it was run on.


def test_the_toggle_costs_every_pair_and_that_is_the_comparison_the_slice_owes():
    report = compare_scale_modes(_extracted())
    assert report["considered"] == 135
    assert report["scale_specific"]["signed"] == 135
    assert report["scale_invariant"]["signed"] == 48
    assert report["lost_by_turning_it_on"] == 87
    assert report["lost_by_cardinality"] == {"2": 87, "3": 0}
    assert report["signable_in_both"] == 48
    assert "T4E.3" in report["not_compared"]


def test_only_the_scale_invariant_mode_could_cross_a_domain_boundary():
    off = sign_constellations(_extracted())
    on = sign_constellations(_extracted(), scale_invariant=True)
    assert off.describe()["cross_domain_comparable"] is False
    assert on.describe()["cross_domain_comparable"] is True
    assert all(not item.cross_domain_comparable for item in off)
    assert all(item.cross_domain_comparable for item in on)
    assert "rescaling" not in MODE_INVARIANCE["scale_specific"]
    assert "rescaling" in MODE_INVARIANCE["scale_invariant"]


def test_the_scale_specific_mode_names_the_two_things_it_does_not_claim():
    signature = signature_for(_extracted().of_cardinality(2)[0])
    assert "rescaling" in signature.refusals
    assert "drifts with the scale being estimated" in signature.refusals["rescaling"]
    assert "cross_domain" in signature.refusals
    assert "cells" in signature.refusals["cross_domain"]


def test_the_scale_invariant_mode_claims_neither_refusal():
    signature = signature_for(_extracted().of_cardinality(3)[0], scale_invariant=True)
    assert "rescaling" not in signature.refusals
    assert "cross_domain" not in signature.refusals


def test_every_signed_pair_has_its_axis_refused_and_every_signed_triple_has_one():
    off = sign_constellations(_extracted())
    assert off.axis_refused == 87
    assert sum(1 for item in off if item.bearings is None) == 87
    assert all(item.bearings is not None for item in off if item.cardinality == 3)
    on = sign_constellations(_extracted(), scale_invariant=True)
    assert on.axis_refused == 0


def test_the_relations_each_mode_keys_on_are_the_two_tg34_matchers():
    assert SCALE_MODES == {"scale_specific": "distance", "scale_invariant": "shape_ratio"}
    off = signature_for(_extracted().of_cardinality(3)[0])
    on = signature_for(_extracted().of_cardinality(3)[0], scale_invariant=True)
    assert off.geometry_relation == "distance"
    assert on.geometry_relation == "shape_ratio"


# =============================================================================== section 6
# What each block is, on the record itself.


def test_the_geometry_block_holds_one_value_per_unordered_pair():
    for item in sign_constellations(_extracted()):
        expected = item.cardinality * (item.cardinality - 1) // 2
        assert len(item.geometry) == expected
        assert item.bearings is None or len(item.bearings) == expected


def test_the_bearings_lie_in_the_only_range_an_unordered_edge_and_an_unsigned_axis_define():
    for item in sign_constellations(_extracted()):
        for bearing in item.bearings or ():
            assert 0.0 <= bearing <= 90.0


def test_the_strength_block_is_band_normalised_and_multiplies_to_one():
    for item in sign_constellations(_extracted()):
        product = math.exp(sum(math.log(value) for value in item.strengths))
        assert product == pytest.approx(1.0, abs=1e-9)


def test_the_scale_block_is_absolute_under_one_mode_and_a_ratio_under_the_other():
    """The toggle's second bite, and the reason only one mode crosses a domain boundary."""
    item = _extracted().of_cardinality(3)[0]
    off = signature_for(item)
    on = signature_for(item, scale_invariant=True)
    assert off.scale_units == "cells"
    assert off.describe()["scale_block_in_vector"] == "absolute, in cells"
    assert on.describe()["scale_block_in_vector"] == "ratios"
    # Scale-specific carries a length in cells, which is what stops it at the boundary.
    assert set(off.vector()[-3:]) == set(off.scales)
    assert max(off.vector()[-3:]) >= 8.0
    # Scale-invariant carries ratios, which multiply to one.
    assert math.exp(sum(math.log(value) for value in on.vector()[-3:])) == pytest.approx(1.0)
    assert off.cross_domain_comparable is False and on.cross_domain_comparable is True


def test_the_same_shape_two_octaves_down_is_one_configuration_in_one_mode_and_two_in_the_other():
    """The two questions the specification names, made arithmetic.

    Same shape, same relative strengths, members moved from levels 4/4/5 to 2/2/3. The
    scale-invariant mode calls it the same configuration; the scale-specific mode does not,
    because its vector holds the levels themselves.
    """
    here = _configuration(_transform(SCALENE), scales=[4, 4, 5])
    lower = _configuration(_transform(SCALENE), scales=[2, 2, 3])
    assert signature_for(here, scale_invariant=True).vector() == pytest.approx(
        signature_for(lower, scale_invariant=True).vector(), abs=1e-9)
    assert signature_for(here).vector() != pytest.approx(
        signature_for(lower).vector(), abs=1e-3)


def test_the_vortex_signatures_only_ever_hold_two_levels_so_universality_is_not_tested_here():
    """The toggle is exercised on this record and it is not evidenced by it.

    The tracked bank has two levels, so "does this configuration recur at *other* scales?" has
    almost no room to be answered: every triple's scale block is one of two ratio patterns.
    """
    patterns = {tuple(sorted(item.scales))
                for item in sign_constellations(_extracted(), scale_invariant=True)}
    assert patterns == {(8.0, 8.0, 16.0), (8.0, 16.0, 16.0)}


# =============================================================================== section 7
# Refusals, each reached by the call that triggers it.


def test_a_pair_asked_for_the_scale_free_shape_is_refused_by_name():
    with pytest.raises(InvalidParameterError) as error:
        signature_for(_extracted().of_cardinality(2)[0], scale_invariant=True)
    assert "its ratio to itself is 1" in str(error.value)
    assert "scale-specific mode" in str(error.value)


def test_a_pairs_axis_is_refused_for_a_different_reason_than_an_isotropic_triples():
    pair = admit_axis(_extracted().of_cardinality(2)[0])
    assert pair.admitted is False
    assert "lie on their own separation" in pair.refusal
    assert "isotropy floor" not in pair.refusal


def test_a_constellation_without_its_features_cannot_be_signed():
    from dataclasses import replace
    stripped = replace(_extracted().of_cardinality(3)[0], features=())
    with pytest.raises(InvalidParameterError) as error:
        signature_for(stripped)
    assert "measured from the coefficients" in str(error.value)


def test_signing_nothing_is_refused():
    with pytest.raises(InvalidParameterError) as error:
        signature_for(None)
    assert "no signature of nothing" in str(error.value)
    with pytest.raises(InvalidParameterError) as error:
        sign_constellations(None)
    assert "ConstellationSet" in str(error.value)


def test_a_position_in_metres_beside_a_scale_in_cells_refuses_one_mode_and_not_the_other():
    """D90's rule, still doing its job -- and the two modes part company on it.

    The scale-specific mode divides a separation by an estimated scale, so a separation in
    metres over a scale in cells is refused by name. The scale-invariant mode divides a
    separation by another separation in the same units, which is exactly what R19's refusal
    message tells the caller to do instead: *compare structure - scale ratios, orientation
    differences, relative geometry*. So the configuration is still signable, and only in the
    mode whose entries are all dimensionless.
    """
    item = _configuration(_transform(SCALENE), axes=METRE_AXES, scale_units="cells")
    assert item.graph.refusals, "distance should have been refused on every edge"
    with pytest.raises(SemanticComparisonError) as error:
        signature_for(item, scale_invariant=False)
    assert "not a number of scale lengths" in str(error.value)
    signed = signature_for(item, scale_invariant=True)
    assert len(signed.geometry) == 3


def test_members_in_two_different_position_units_are_refused_before_this_module_sees_them():
    """There is no second copy of this refusal in the signature module, and no need for one."""
    features = [
        _feature(0.0, 2048.0, 2048.0, 3.0, axes=METRE_AXES, scale_units="m"),
        _feature(0.0, 2048.0, 2078.0, 5.0, axes=AXES, orientation="HL"),
        _feature(0.0, 2088.0, 2048.0, 7.0, scale=5, axes=METRE_AXES, scale_units="m"),
    ]
    tracks = [Track(index, FeatureSet([item]), {}) for index, item in enumerate(features)]
    with pytest.raises(SemanticComparisonError) as error:
        extract_constellations(TrackingResult(
            features=FeatureSet(list(features)), associator="hungarian", alpha=0.05,
            bounds=BOUNDS, volume=SearchVolume({ROW: 4096.0, COL: 4096.0}, units="m"),
            times=(0.0,), tracks=tuple(tracks)), cardinalities=(3,))
    assert "coord_units" in str(error.value)


def test_a_member_with_no_band_rms_is_refused_rather_than_compared_as_a_filter_gain():
    features = [_feature(0.0, row, col, magnitude, scale=scale,
                         orientation="LH" if index % 2 == 0 else "HL",
                         sigma=(0.0 if index == 1 else 4.0))
                for index, ((row, col), magnitude, scale)
                in enumerate(zip(_transform(SCALENE), [3.0, 5.0, 7.0], [4, 4, 5]))]
    tracks = [Track(index, FeatureSet([item]), {}) for index, item in enumerate(features)]
    result = TrackingResult(
        features=FeatureSet(list(features)), associator="hungarian", alpha=0.05,
        bounds=BOUNDS, volume=SearchVolume({ROW: 4096.0, COL: 4096.0}, units="cells"),
        times=(0.0,), tracks=tuple(tracks))
    item = extract_constellations(result, cardinalities=(3,))[0]
    with pytest.raises(InvalidParameterError) as error:
        signature_for(item, scale_invariant=True)
    assert "ratio of filter gains" in str(error.value)


def test_an_unknown_mode_cannot_be_constructed():
    signature = signature_for(_extracted().of_cardinality(3)[0], scale_invariant=True)
    from dataclasses import replace
    with pytest.raises(InvalidParameterError) as error:
        replace(signature, mode="whatever")
    assert "names which quantity the geometry was divided by" in str(error.value)


def test_a_signature_whose_blocks_disagree_about_cardinality_is_refused():
    from dataclasses import replace
    signature = signature_for(_extracted().of_cardinality(3)[0], scale_invariant=True)
    with pytest.raises(InvalidParameterError) as error:
        replace(signature, geometry=signature.geometry[:2])
    assert "one value per unordered pair" in str(error.value)
    with pytest.raises(InvalidParameterError) as error:
        replace(signature, bearings=(1.0,))
    assert "one bearing per edge" in str(error.value)


# =============================================================================== section 8
# The receipts, and what they refuse to say.


def test_a_signature_receipt_is_serialisable_and_says_what_it_does_not_establish():
    body = signature_for(_extracted().of_cardinality(3)[0], scale_invariant=True).describe()
    assert body["schema"] == SIGNATURE_SCHEMA
    assert "not a finding" in body["claim_boundary"]
    assert "tolerance calibrated against replicates" in body["claim_boundary"]
    assert "a function of the geometry block" in body["bearings_independence"]
    assert "no estimated quantity enters" in body["geometry_basis"]
    assert json.loads(json.dumps(body))["mode"] == "scale_invariant"


def test_the_scale_specific_receipt_says_the_estimator_is_why_it_is_not_invariant():
    body = signature_for(_extracted().of_cardinality(3)[0]).describe()
    assert "drifts with the scale being estimated" in body["geometry_basis"]
    assert body["cross_domain_comparable"] is False


def test_the_set_receipt_carries_the_floor_the_axis_was_judged_against():
    body = describe_signatures(sign_constellations(_extracted()))
    assert body["isotropy_floor"] == AXIS_ISOTROPY_FLOOR
    assert body["considered"] == 135
    assert body["signed"] == 135
    assert body["axis_refused"] == 87
    assert body["anisotropy"]["min"] > 0.0
    assert sum(body["by_bands"].values()) == 135
    assert json.loads(json.dumps(body))["mode"] == "scale_specific"


def test_the_axis_receipt_says_that_clearing_the_floor_is_not_a_precision_claim():
    body = admit_axis(_extracted().of_cardinality(3)[0]).describe()
    assert body["admitted"] is True
    assert body["collinear"] is False
    assert "not a precision claim" in body["floor_basis"]
    assert body["angle_deg"] is not None
