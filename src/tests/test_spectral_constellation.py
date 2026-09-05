"""T4E.1: the bridge from a tracking pass to TG3.3's attributed graphs.

1.  **The graph is TG3.3's, not a second one.** Every constellation carries a real
    `AttributedGraph`, and the relations it declares are the ones `measurable_relations` says
    these features support -- three of the eight, with the other five refused by name and the
    reason recorded rather than dropped.
2.  **D90.** `distance` refused on every 4D feature because the position unit was spelled
    `cells` and the scale unit `parent-grid px`. The regression is pinned on the units
    themselves, not on the symptom.
3.  **The carried half is checked against the tracks it came from**, and the enumeration
    against the combinatorics of the frame census, so it cannot quietly be a sample.
4.  **A cross-band strength ratio is a ratio of filter gains until it is normalised.** On this
    record the raw ratio and the band-normalised one disagree about the *sign*.
5.  **An onset at the first searched frame is censored**, and `succession` is not that
    quantity: it orders two observations that are in one frame by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis_engine.spectral_constellation import (
    ALLOWED_CARDINALITIES, BEARING_BASIS, CONSTELLATION_SCHEMA, FrameConstellation,
    describe_constellations, extract_constellations,
)
from src.analysis_engine.spectral_tracking import COL, ROW, track_spectral_features
from src.benchmarks.core import get_benchmark
from src.core.constellation import RELATIONS, constellation, measurable_relations
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
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
WRAPPED_AXES = (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
                AxisSpec(COL, "space", units="cells", periodic=True, ordinal=1))
REPRESENTATION = "wavelet[swt,wavelet=haar,levels=5]"

_CACHE = {}


def _tracked():
    """The T4D.2 acceptance pass, tracked once and reused."""
    if "tracked" not in _CACHE:
        built = get_benchmark("advected_vortex_sequence").make()
        sequence = FieldSequence(built.fields, np.arange(len(built.fields), dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET}).select(
                scales=TRACKED_SCALES)
        result = track_spectral_features(
            field, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", bounds=BOUNDS,
            threshold_sigma=THRESHOLD_SIGMA)
        _CACHE["tracked"] = (field, result)
    return _CACHE["tracked"]


def _extracted(cardinalities=(2, 3)):
    key = ("extracted", cardinalities)
    if key not in _CACHE:
        _, result = _tracked()
        _CACHE[key] = extract_constellations(result, cardinalities=cardinalities)
    return _CACHE[key]


def _feature(time, row, col, magnitude, *, scale=4, orientation="LH", dataset="manual",
             threshold=1.0, sigma=4.0, axes=AXES, comparable=True):
    provenance = {"scale_label": scale, "orientation_label": orientation,
                  "threshold": threshold, "threshold_sigma": sigma, "phase": None,
                  "alignment_cross_scale_comparable": comparable}
    return SpectralFeature(
        domain="synthetic", dataset=dataset, variable="amplitude",
        magnitude=Quantity(float(magnitude), None),
        location=FeatureLocation({ROW: float(row), COL: float(col)}, axes),
        time=float(time), time_units="frames", representation=REPRESENTATION,
        spatial_scale=Quantity(float(2 ** (scale - 1)), "cells"),
        provenance=provenance)


def _track(track_id, features, periods=None):
    return Track(track_id, FeatureSet(list(features)), periods or {})


def _result(tracks, times):
    """A hand-built tracking result, so a refusal can be reached without a whole transform."""
    features = FeatureSet([item for track in tracks for item in track.observations])
    return TrackingResult(
        features=features, associator="hungarian", alpha=0.05, bounds=BOUNDS,
        volume=SearchVolume({ROW: 128.0, COL: 128.0}, units="cells"),
        times=tuple(float(t) for t in times), tracks=tuple(tracks))


def _pair(bands, time):
    """The one two-node constellation of `bands` at `time`."""
    found = [item for item in _extracted().at(time)
             if item.cardinality == 2 and item.bands == bands]
    assert len(found) == 1, "expected exactly one %s pair at %s" % (bands, time)
    return found[0]


# =============================================================================== section 1
# The graph is TG3.3's, and it declares what these features can actually support.


def test_every_constellation_carries_a_real_attributed_graph():
    for item in _extracted():
        assert item.graph.n_nodes == item.cardinality
        assert set(item.graph.relations) == set(_extracted().relations)
        assert len(item.graph.carried) == item.cardinality


def test_three_of_the_eight_relations_are_measurable_and_five_refuse_by_name():
    extracted = _extracted()
    assert len(RELATIONS.names()) == 8
    assert list(extracted.relations) == ["distance", "relative_scale", "succession"]
    assert sorted(extracted.unmeasurable) == [
        "co_occurrence", "containment", "convergence", "direction", "temporal_lag"]
    assert "temporal_scale" in extracted.unmeasurable["temporal_lag"]
    assert "orientation" in extracted.unmeasurable["direction"]
    assert "extent" in extracted.unmeasurable["containment"]
    for reason in extracted.unmeasurable.values():
        assert "constrains nothing" in reason


def test_the_declared_relations_are_exactly_what_measurable_relations_says():
    _, result = _tracked()
    every = [item for track in result for item in track.observations]
    assert tuple(_extracted().relations) == measurable_relations(every)


def test_the_comparable_half_is_dimensionless_and_says_what_it_divided_by():
    graph = _pair(("L4/LH", "L4/HL"), 9.0).graph
    edge = graph.edge(0, 1)
    assert edge["distance"].value == pytest.approx(1.4447, abs=0.001)
    assert "geometric mean of the two spatial scales" in edge["distance"].basis
    assert edge["relative_scale"].value == 1.0


def test_succession_is_always_false_inside_a_frame_which_is_why_onsets_are_carried():
    """The ordering TG3.3 measures is between observations, and they share a frame."""
    for item in _extracted():
        for left in range(item.cardinality):
            for right in range(item.cardinality):
                if left != right:
                    assert item.graph.value(left, right, "succession").holds is False
    relation = _pair(("L4/LH", "L5/LH"), 9.0).relations[0]
    assert relation.onset_offset == pytest.approx(9.0)


# =============================================================================== section 2
# D90: the unit spelling that disabled the one relation carrying geometry.


def test_a_features_position_and_its_scale_are_named_in_the_same_unit_d90():
    _, result = _tracked()
    observation = result[0][0]
    axis_units = {axis.units for axis in observation.location.axes}
    assert axis_units == {"cells"}
    assert observation.spatial_scale.units == "cells", (
        "a separation divided by a scale is refused when the two unit names differ, even "
        "when they denote one unit (D90)")


def test_distance_is_measured_rather_than_refused_on_every_pair():
    for item in _extracted():
        assert not item.graph.describe()["refusals"], (
            "no pair of this pass refuses a declared relation")
        for left in range(item.cardinality):
            for right in range(item.cardinality):
                if left != right:
                    assert item.graph.value(left, right, "distance").value > 0.0


def test_a_scale_named_in_another_unit_is_still_refused_which_is_the_check_working():
    """The D90 fix aligned two names; it did not weaken the rule that unlike units refuse."""
    left = _feature(0.0, 0.0, 0.0, 1.0)
    right = SpectralFeature(
        domain="synthetic", dataset="manual", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({ROW: 5.0, COL: 5.0}, AXES), time=0.0,
        time_units="frames", representation=REPRESENTATION,
        spatial_scale=Quantity(8.0, "metres"),
        provenance={"scale_label": 4, "orientation_label": "HL"})
    graph = constellation([left, right], relations=["distance"], strict=False)
    assert graph.describe()["refusals"], "a scale in metres beside a location in cells refuses"


# =============================================================================== section 3
# The carried half, against the tracks it was built from.


def test_every_node_carries_the_numbers_of_the_observation_it_was_built_from():
    _, result = _tracked()
    extracted = _extracted()
    by_id = {track.track_id: track for track in result}
    checked = 0
    for item in extracted:
        for node in item.nodes:
            track = by_id[node.track_id]
            index = list(track.observations.times).index(node.time)
            observation = track[index]
            assert node.strength == pytest.approx(observation.magnitude.value)
            assert node.coords[ROW] == pytest.approx(observation.location.coords[ROW])
            assert node.coords[COL] == pytest.approx(observation.location.coords[COL])
            assert node.scale == pytest.approx(observation.spatial_scale.value)
            assert node.onset == pytest.approx(track.birth_time)
            assert node.age == pytest.approx(node.time - track.birth_time)
            assert node.observations == len(track)
            checked += 1
    assert checked == 87 * 2 + 48 * 3 == 318, (
        "every node of every constellation is checked against its own track")


def test_the_carried_half_is_labelled_as_carried_in_the_receipt():
    body = _extracted()[0].describe()
    assert "never compared across a domain boundary" in body["nodes"][0]["carried"]
    assert "never compared across a domain boundary" in body["relations"][0]["carried"]
    assert "R19 refuses it across a domain boundary" in body["claim_boundary"]


def test_the_enumeration_is_complete_and_the_census_says_so():
    extracted = _extracted()
    expected = 0
    for row in extracted.census:
        present = row["tracks_present"]
        pairs = math.comb(present, 2) if present >= 2 else 0
        triples = math.comb(present, 3) if present >= 3 else 0
        assert row["constellations"] == pairs + triples
        expected += pairs + triples
        assert len(extracted.at(row["time"])) == pairs + triples
    assert len(extracted) == expected == 135
    assert len(extracted.of_cardinality(2)) == 87
    assert len(extracted.of_cardinality(3)) == 48


def test_a_triple_carries_three_pair_records_and_a_pair_carries_one():
    for item in _extracted():
        assert len(item.relations) == math.comb(item.cardinality, 2)
        assert len(set(item.track_ids)) == item.cardinality
        assert len({node.time for node in item.nodes}) == 1


def test_a_pair_record_is_oriented_low_track_id_to_high():
    for item in _extracted():
        for relation in item.relations:
            assert relation.source < relation.target
            assert item.relation_between(relation.target, relation.source) is relation


def test_the_receipt_names_the_budget_the_bands_and_the_absence_of_support():
    body = describe_constellations(_extracted())
    assert body["schema"] == CONSTELLATION_SCHEMA
    assert body["frames_searched"] == 24
    assert body["budget"]["largest_frame"] == 4
    assert body["budget"]["enumeration"].startswith("complete")
    assert body["by_bands"]["L4/LH+L4/HL"] == 24
    assert "not counted here" in body["support"]
    assert "no support count" in body["claim_boundary"]


def test_a_constellation_identity_is_the_frame_and_the_tracks_and_it_is_hashable():
    extracted = _extracted()
    keys = {item.key() for item in extracted}
    assert len(keys) == len(extracted), "no two constellations share an identity"
    first = extracted[0]
    assert first.key() == (first.time,) + first.track_ids


# =============================================================================== section 4
# A separation is between two flanks.


def test_two_bands_following_one_vortex_are_measurably_apart():
    """The flank offset T4D.2 measured, expressed as the thing an edge would otherwise hide."""
    separations = [item.relations[0].separation for item in _extracted().of_cardinality(2)
                   if item.bands == ("L4/LH", "L4/HL")]
    assert len(separations) == 24
    # One vortex, one level, two orientations: the separation is entirely flank geometry.
    assert min(separations) > 2.0
    assert max(separations) > 10.0


def test_a_pair_record_says_whether_its_two_nodes_came_from_the_same_band():
    for item in _extracted().of_cardinality(2):
        assert item.relations[0].same_band == (item.bands[0] == item.bands[1])
    assert not any(relation.same_band for item in _extracted().of_cardinality(2)
                   for relation in item.relations), (
        "the vortex produces one track per band, so no pair shares a band")


def test_the_claim_boundary_says_the_separation_is_between_flanks():
    body = _extracted()[0].describe()
    assert "flank" in body["claim_boundary"]
    assert "not between two structures" in body["claim_boundary"]


# =============================================================================== section 5
# A cross-band strength ratio is a ratio of filter gains until it is normalised.


def test_the_raw_and_normalised_strength_ratios_disagree_about_the_sign_of_the_comparison():
    """The measurement that justifies carrying both numbers instead of one."""
    relation = _pair(("L4/LH", "L5/LH"), 9.0).relations[0]
    # Raw: the coarse band's coefficient is the larger number by three quarters.
    assert relation.strength_ratio == pytest.approx(1.755, abs=0.01)
    assert relation.strength_ratio > 1.0
    # Each divided by its own band's RMS first: the coarse band is the weaker of the two.
    assert relation.strength_ratio_sigma == pytest.approx(0.554, abs=0.01)
    assert relation.strength_ratio_sigma < 1.0


def test_the_band_rms_comes_back_exactly_from_the_threshold_and_its_sigma():
    _, result = _tracked()
    by_id = {track.track_id: track for track in result}
    for item in _extracted():
        for node in item.nodes:
            track = by_id[node.track_id]
            observation = track[list(track.observations.times).index(node.time)]
            rms = (observation.provenance["threshold"]
                   / observation.provenance["threshold_sigma"])
            assert node.strength_sigma == pytest.approx(node.strength / rms)


def test_a_detection_that_recorded_no_threshold_gets_no_normalised_strength_and_says_why():
    bare = [SpectralFeature(
        domain="synthetic", dataset="manual", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({ROW: float(t), COL: 0.0}, AXES), time=float(t),
        time_units="frames", representation=REPRESENTATION,
        spatial_scale=Quantity(8.0, "cells"),
        provenance={"scale_label": 4, "orientation_label": "LH"}) for t in (0.0, 1.0)]
    other = [_feature(t, 5.0, 5.0, 2.0) for t in (0.0, 1.0)]
    extracted = extract_constellations(_result([_track(0, bare), _track(1, other)], [0.0, 1.0]))
    node = extracted[0].nodes[0]
    assert node.strength_sigma is None
    assert "cannot be put on a common footing" in node.describe()["strength_sigma_refused"]
    assert extracted[0].relations[0].strength_ratio_sigma is None
    assert extracted[0].relations[0].strength_ratio == pytest.approx(2.0)


# =============================================================================== section 6
# An onset at the first searched frame is censored.


def test_a_track_alive_in_the_first_searched_frame_is_marked_left_censored():
    _, result = _tracked()
    censored = {node.band: node.onset_censored
                for item in _extracted() for node in item.nodes}
    assert censored == {"L4/LH": True, "L4/HL": True, "L5/LH": False, "L5/HL": False}
    assert result.frame_times[0] == 0.0


def test_the_nine_frame_offset_is_reported_as_a_bound_because_one_end_is_censored():
    relation = _pair(("L4/LH", "L5/LH"), 9.0).relations[0]
    assert relation.onset_offset == pytest.approx(9.0)
    assert relation.onset_offset_censored is True
    assert "lower bound" in relation.describe()["onset_offset_note"]


def test_an_offset_between_two_uncensored_onsets_carries_no_bound_note():
    early = [_feature(t, t, 0.0, 1.0) for t in (1.0, 2.0, 3.0)]
    late = [_feature(t, t, 10.0, 1.0, orientation="HL") for t in (2.0, 3.0)]
    result = _result([_track(0, early), _track(1, late)], [0.0, 1.0, 2.0, 3.0])
    relation = extract_constellations(result).at(2.0)[0].relations[0]
    assert relation.onset_offset == pytest.approx(1.0)
    assert relation.onset_offset_censored is False
    assert "onset_offset_note" not in relation.describe()


# =============================================================================== section 7
# A bearing in the carried half is not a compass.


@pytest.mark.parametrize("d_row, d_col, expected", [
    (1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0), (0.0, -1.0, 270.0),
    (1.0, 1.0, 45.0), (1.0, -1.0, 315.0),
])
def test_the_plane_angle_runs_from_plus_row_toward_plus_col(d_row, d_col, expected):
    first = [_feature(t, 10.0, 10.0, 1.0) for t in (0.0, 1.0)]
    second = [_feature(t, 10.0 + d_row, 10.0 + d_col, 1.0, orientation="HL")
              for t in (0.0, 1.0)]
    result = _result([_track(0, first), _track(1, second)], [0.0, 1.0])
    relation = extract_constellations(result).at(0.0)[0].relations[0]
    assert relation.bearing_deg == pytest.approx(expected)
    assert relation.separation == pytest.approx(math.hypot(d_row, d_col))


def test_the_bearing_says_in_the_receipt_that_no_grid_was_consulted():
    body = _extracted()[0].relations[0].describe()
    assert body["bearing_basis"] == BEARING_BASIS
    assert "not a compass" in body["bearing_basis"]
    assert "cosine of the latitude" in body["bearing_basis"]


def test_two_nodes_at_one_point_are_given_no_direction_rather_than_zero_degrees():
    first = [_feature(t, 4.0, 4.0, 1.0) for t in (0.0, 1.0)]
    second = [_feature(t, 4.0, 4.0, 2.0, orientation="HL") for t in (0.0, 1.0)]
    relation = extract_constellations(
        _result([_track(0, first), _track(1, second)], [0.0, 1.0])).at(0.0)[0].relations[0]
    assert relation.separation == 0.0
    assert relation.bearing_deg is None
    assert relation.radial_velocity is None
    assert "undefined rather than zero" in relation.describe()["bearing_refused"]


def test_a_separation_wraps_on_a_periodic_axis_instead_of_going_the_long_way():
    left = [_feature(t, 0.0, 1.0, 1.0, axes=WRAPPED_AXES) for t in (0.0, 1.0)]
    right = [_feature(t, 0.0, 99.0, 1.0, axes=WRAPPED_AXES, orientation="HL")
             for t in (0.0, 1.0)]
    result = _result([_track(0, left, {COL: 100.0}), _track(1, right, {COL: 100.0})],
                     [0.0, 1.0])
    relation = extract_constellations(result).at(0.0)[0].relations[0]
    assert relation.offset[COL] == pytest.approx(-2.0)
    assert relation.separation == pytest.approx(2.0)


def test_two_tracks_disagreeing_about_where_an_axis_closes_are_refused():
    left = [_feature(t, 0.0, 1.0, 1.0, axes=WRAPPED_AXES) for t in (0.0, 1.0)]
    right = [_feature(t, 0.0, 9.0, 1.0, axes=WRAPPED_AXES, orientation="HL")
             for t in (0.0, 1.0)]
    result = _result([_track(0, left, {COL: 100.0}), _track(1, right, {COL: 360.0})],
                     [0.0, 1.0])
    with pytest.raises(SemanticComparisonError) as excinfo:
        extract_constellations(result)
    assert "where it closes" in str(excinfo.value)


# =============================================================================== section 8
# Rates are local to the node, not fitted once and hung on every frame.


def test_the_rate_of_change_is_local_so_two_frames_of_one_track_can_differ():
    grower = [_feature(t, 0.0, 0.0, m) for t, m in ((0.0, 1.0), (1.0, 1.0), (2.0, 5.0))]
    other = [_feature(t, 8.0, 0.0, 1.0, orientation="HL") for t in (0.0, 1.0, 2.0)]
    extracted = extract_constellations(
        _result([_track(0, grower), _track(1, other)], [0.0, 1.0, 2.0]))
    rates = {item.time: item.nodes[0].strength_rate for item in extracted}
    assert rates[0.0] == pytest.approx(0.0)          # one-sided over 1.0 -> 1.0
    assert rates[1.0] == pytest.approx(2.0)          # central over 1.0 -> 5.0 in two frames
    assert rates[2.0] == pytest.approx(4.0)          # one-sided over 1.0 -> 5.0


def test_a_single_sighting_gets_no_rate_and_no_velocity_and_says_why():
    lone = [_feature(1.0, 0.0, 0.0, 1.0)]
    other = [_feature(t, 8.0, 0.0, 1.0, orientation="HL") for t in (0.0, 1.0, 2.0)]
    extracted = extract_constellations(
        _result([_track(0, lone), _track(1, other)], [0.0, 1.0, 2.0]))
    node = extracted.at(1.0)[0].nodes[0]
    assert node.observations == 1
    assert node.strength_rate is None
    assert node.velocity is None
    assert node.scale_velocity is None
    assert "never made" in node.describe()["strength_rate_refused"]
    assert extracted.at(1.0)[0].relations[0].radial_velocity is None


def test_a_track_that_held_one_level_reports_a_scale_velocity_of_exactly_zero():
    for item in _extracted():
        for node in item.nodes:
            assert node.held_one_scale is True
            assert node.scale_velocity == 0.0, (
                "not a least-squares residue of 1e-17, which reads as a direction")


def test_the_radial_velocity_is_signed_and_positive_means_diverging():
    still = [_feature(t, 0.0, 0.0, 1.0) for t in (0.0, 1.0, 2.0)]
    approaching = [_feature(t, 0.0, 20.0 - 4.0 * t, 1.0, orientation="HL")
                   for t in (0.0, 1.0, 2.0)]
    relation = extract_constellations(
        _result([_track(0, still), _track(1, approaching)],
                [0.0, 1.0, 2.0])).at(1.0)[0].relations[0]
    assert relation.radial_velocity == pytest.approx(-4.0)
    assert "positive is diverging" in relation.describe()["radial_velocity_sign"]


# =============================================================================== section 9
# What an extraction refuses to be.


@pytest.mark.parametrize("cardinalities", [(1,), (4,), (2, 4), (2, 2), ()])
def test_only_pairs_and_triples_are_built(cardinalities):
    _, result = _tracked()
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(result, cardinalities=cardinalities)
    message = str(excinfo.value)
    assert "frequent-subgraph mining" in message or "distinct sizes" in message
    assert list(ALLOWED_CARDINALITIES) == [2, 3]


def test_a_frame_over_the_node_cap_refuses_rather_than_sampling():
    _, result = _tracked()
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(result, cardinalities=(2, 3), max_nodes_per_frame=3)
    message = str(excinfo.value)
    assert "refuses rather than samples" in message
    assert "a count of what fitted" in message
    assert "holds 4" in message


def test_a_sweep_over_the_constellation_budget_is_refused_whole_not_returned_as_a_prefix():
    _, result = _tracked()
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(result, cardinalities=(2, 3), max_constellations=10)
    assert "rather than returned as a prefix" in str(excinfo.value)


def test_a_tracking_result_of_none_is_refused_instead_of_yielding_an_empty_set():
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(None)
    assert "no constellation of nothing" in str(excinfo.value)


def test_r19_is_enforced_by_tg3_3_rather_than_a_second_time_here():
    """Two datasets are refused, and the refusal is the one `constellation()` already makes."""
    one = [_feature(t, 0.0, 0.0, 1.0, dataset="era5") for t in (0.0, 1.0)]
    two = [_feature(t, 5.0, 5.0, 1.0, dataset="argo", orientation="HL") for t in (0.0, 1.0)]
    inconsistent = TrackingResult(
        features=FeatureSet(one), associator="hungarian", alpha=0.05, bounds=BOUNDS,
        volume=SearchVolume({ROW: 128.0, COL: 128.0}, units="cells"),
        times=(0.0, 1.0), tracks=(_track(0, one), _track(1, two)))
    with pytest.raises(SemanticComparisonError) as excinfo:
        extract_constellations(inconsistent)
    assert "a configuration within one dataset" in str(excinfo.value)


def test_a_cross_scale_pair_from_an_unregistered_decomposition_is_refused_d88():
    fine = [_feature(t, 0.0, 0.0, 1.0, scale=4, comparable=False) for t in (0.0, 1.0)]
    coarse = [_feature(t, 5.0, 5.0, 1.0, scale=5, orientation="HL", comparable=False)
              for t in (0.0, 1.0)]
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(_result([_track(0, fine), _track(1, coarse)], [0.0, 1.0]))
    assert "registered to one another (D88)" in str(excinfo.value)


def test_a_same_scale_pair_from_an_unregistered_decomposition_is_allowed():
    """Registration is a cross-scale question, so one level does not need it."""
    first = [_feature(t, 0.0, 0.0, 1.0, scale=4, comparable=False) for t in (0.0, 1.0)]
    second = [_feature(t, 5.0, 5.0, 1.0, scale=4, orientation="HL", comparable=False)
              for t in (0.0, 1.0)]
    extracted = extract_constellations(
        _result([_track(0, first), _track(1, second)], [0.0, 1.0]))
    assert len(extracted) == 2
    assert extracted[0].spans_scales() is False


def test_a_decomposition_that_recorded_no_registration_is_not_refused_for_a_missing_key():
    """An absent receipt is not a passing one, but it is not a failing one either."""
    def bare(scale, row, col, orientation):
        return [SpectralFeature(
            domain="synthetic", dataset="manual", variable="amplitude",
            magnitude=Quantity(1.0, None),
            location=FeatureLocation({ROW: row + t, COL: col}, AXES), time=float(t),
            time_units="frames", representation=REPRESENTATION,
            spatial_scale=Quantity(float(2 ** (scale - 1)), "cells"),
            provenance={"scale_label": scale, "orientation_label": orientation})
            for t in (0.0, 1.0)]
    extracted = extract_constellations(
        _result([_track(0, bare(4, 0.0, 0.0, "LH")), _track(1, bare(5, 9.0, 9.0, "HL"))],
                [0.0, 1.0]))
    assert extracted[0].spans_scales() is True
    assert extracted[0].nodes[0].cross_scale_comparable is None


def test_an_observation_with_no_spatial_scale_is_refused_rather_than_left_a_hole():
    bare = [SpectralFeature(
        domain="synthetic", dataset="manual", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({ROW: float(t), COL: 0.0}, AXES), time=float(t),
        time_units="frames", representation=REPRESENTATION,
        provenance={"scale_label": 4, "orientation_label": "LH"}) for t in (0.0, 1.0)]
    other = [_feature(t, 9.0, 9.0, 1.0, orientation="HL") for t in (0.0, 1.0)]
    with pytest.raises(InvalidParameterError) as excinfo:
        extract_constellations(_result([_track(0, bare), _track(1, other)], [0.0, 1.0]))
    assert "a hole in every edge it touches" in str(excinfo.value)


def test_a_graph_of_one_node_or_four_is_refused_at_construction():
    item = _extracted().of_cardinality(2)[0]
    for count in (1, 4):
        nodes = (item.nodes * 4)[:count]
        with pytest.raises(InvalidParameterError) as excinfo:
            FrameConstellation(graph=item.graph, nodes=nodes, relations=(),
                               time=item.time, time_units=item.time_units)
        assert "two or three nodes" in str(excinfo.value)


def test_the_carried_half_must_be_the_same_size_as_the_comparable_half():
    triple = _extracted().of_cardinality(3)[0]
    pair = _extracted().of_cardinality(2)[0]
    with pytest.raises(InvalidParameterError) as excinfo:
        FrameConstellation(graph=triple.graph, nodes=pair.nodes, relations=(),
                           time=pair.time, time_units=pair.time_units)
    assert "one node per track" in str(excinfo.value)


def test_there_is_no_self_loop_check_because_a_track_cannot_be_in_a_frame_twice():
    """`Track` refuses two observations at one instant, so a self-pair cannot be built."""
    with pytest.raises(InvalidParameterError) as excinfo:
        _track(0, [_feature(0.0, 0.0, 0.0, 1.0),
                   _feature(0.0, 1.0, 1.0, 1.0, orientation="HL")])
    assert "strictly increasing observation times" in str(excinfo.value)
