# -*- coding: utf-8 -*-
"""T4E.7: a clustering radius calibrated against a null, and every way it refuses.

The module under test replaces a calibration that needed replicates a real record cannot
supply (D97) and whose one published error rate was zero by construction (T4E.6). These tests
hold it to the bar those two findings set: the radius must not be arbitrary, the rates must not
be tautological, and the refusals must be reachable on data rather than only in principle.

Most of the suite works on **constructed distance populations** rather than on signatures. That
is deliberate: it makes the mixture exact, so an estimate can be checked against a number rather
than against an impression, and it keeps the three tests that do run real signatures through the
metric affordable enough to assert something. Those three are the ones that matter most -- a
planted grouping recovered without labels and then checked against them, the same machinery
refusing on a record with no recurrence, and the contamination rate measured against labels and
shown to fall below the truth -- so they use cardinality-2 signatures where the cardinality is
not what is being tested, because a distance minimises over the node permutations and three
nodes cost three times two.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureFamily, SignatureMetric, SignaturePoint,
)
from src.analysis_engine.spectral_null_calibration import (
    ATTRIBUTE_PERMUTATION_NULL, CALIBRATION_MEASURED, CALIBRATION_NO_OPERATING_POINT,
    CALIBRATION_NO_SEPARATION, CALIBRATION_STATUSES, CALIBRATION_UNSTABLE,
    DEFAULT_NULL_FRACTION_CAP, DistanceEnsemble, ExcessPoint, NULL_CALIBRATION_SCHEMA,
    SPLIT_RATE_NOTE,
    SURROGATE_RECORD_NULL, SignatureNull, calibrate_against_null, distance_ensemble,
    estimate_recurrence_fraction, excess_curve, grid_resolution, permuted_signature_population,
    pair_at, radii_for_resolution, radius_grid, signature_distances,
    separation_test,
)
from src.core.errors import InvalidParameterError

NULL = SignatureNull(name="constructed", preserves=("the unrelated-pair distribution",),
                     destroys=("recurrence",))

FAMILY = SignatureFamily(mode="scale_specific", cardinality=3, has_bearings=False,
                         scale_units="km")
#: Two nodes rather than three, so a distance minimises over two node permutations instead of
#: six. Used where a test needs many thousands of real distances and the cardinality is not
#: what it is testing.
PAIR_FAMILY = SignatureFamily(mode="scale_specific", cardinality=2, has_bearings=False,
                              scale_units="km")
METRIC = SignatureMetric(AttributeWeights(1.0, 0.0, 1.0, 1.0))


# --------------------------------------------------------------------------------------------
# constructed populations
# --------------------------------------------------------------------------------------------


def unrelated(rng, n):
    """Distances between configurations that are not the same one."""
    return rng.uniform(0.30, 1.00, n)


def mixture(rng, n, pi):
    """`pi` of the pairs are one recurring configuration, the rest are unrelated."""
    n_same = int(round(pi * n))
    return np.concatenate([rng.uniform(0.0, 0.05, n_same), unrelated(rng, n - n_same)])


def ensemble_of(observed, n_members=19, n=6000, seed=4, null=NULL):
    rng = np.random.default_rng(seed)
    return distance_ensemble(
        observed, [unrelated(rng, n) for _ in range(n_members)], null=null,
        metric_digest="digest")


@pytest.fixture(scope="module")
def planted():
    """A record where 6% of pairs are one recurring configuration, exactly."""
    rng = np.random.default_rng(20260908)
    return ensemble_of(mixture(rng, 6000, 0.06))


@pytest.fixture(scope="module")
def pure_null():
    """A record with no recurrence at all, drawn from the very distribution of its null."""
    rng = np.random.default_rng(11)
    return ensemble_of(unrelated(rng, 6000))


# --------------------------------------------------------------------------------------------
# the null's identity travels with the number
# --------------------------------------------------------------------------------------------


def test_a_null_without_a_name_is_refused():
    with pytest.raises(InvalidParameterError) as raised:
        SignatureNull(name="  ", preserves=("a",), destroys=("b",))
    assert "cannot be interpreted" in str(raised.value)


@pytest.mark.parametrize("preserves,destroys", [((), ("b",)), (("a",), ())])
def test_a_null_that_preserves_or_destroys_nothing_is_refused(preserves, destroys):
    with pytest.raises(InvalidParameterError):
        SignatureNull(name="n", preserves=preserves, destroys=destroys)


def test_the_cheap_null_carries_its_own_warning_and_the_honest_one_does_not():
    # The anti-conservative null is provided because a caller will reach for it, and it is named
    # for its failure exactly as surrogate_null.per_frame_phase is.
    assert "ANTI-CONSERVATIVE" in (ATTRIBUTE_PERMUTATION_NULL.caveat or "")
    assert SURROGATE_RECORD_NULL.caveat is None
    assert "caveat" in ATTRIBUTE_PERMUTATION_NULL.describe()
    assert "caveat" not in SURROGATE_RECORD_NULL.describe()


# --------------------------------------------------------------------------------------------
# the ensemble
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_a_distance_that_is_not_a_distance_is_refused(bad):
    with pytest.raises(InvalidParameterError):
        distance_ensemble([0.1, bad], [[0.5]], null=NULL)


def test_an_ensemble_with_no_null_members_is_refused():
    with pytest.raises(InvalidParameterError) as raised:
        distance_ensemble([0.1, 0.2], [], null=NULL)
    assert "An excess is a comparison" in str(raised.value)


def test_an_ensemble_needs_a_declared_null_not_a_string():
    with pytest.raises(InvalidParameterError):
        DistanceEnsemble(observed=(0.1,), null_members=((0.5,),), null="phase randomised")


def test_the_p_value_floor_is_one_over_one_plus_the_ensemble(planted):
    assert planted.p_value_floor() == pytest.approx(1.0 / 20.0)


def test_a_null_far_smaller_than_the_record_is_warned_about():
    ensemble = distance_ensemble(list(np.linspace(0.1, 1.0, 500)),
                                 [[0.4, 0.5, 0.6]] * 19, null=NULL)
    assert any("median null population" in row for row in ensemble.warnings())


def test_too_few_null_members_is_warned_about_with_the_floor_named():
    rng = np.random.default_rng(3)
    ensemble = ensemble_of(unrelated(rng, 500), n_members=5, n=500)
    warned = [row for row in ensemble.warnings() if "floor any p-value" in row]
    assert warned and "0.167" in warned[0]


def test_a_caveated_null_surfaces_its_caveat_in_the_ensemble_warnings():
    rng = np.random.default_rng(3)
    ensemble = ensemble_of(unrelated(rng, 500), n_members=19, n=500,
                           null=ATTRIBUTE_PERMUTATION_NULL)
    assert any("ANTI-CONSERVATIVE" in row for row in ensemble.warnings())


def test_a_null_that_destroyed_the_detections_is_warned_about_through_the_sampling():
    """The warning pair counts cannot give, because sampling caps both at the same number.

    On the acquired record the surrogate null produced about a quarter of the record's
    signatures. Both populations were then sampled to 6,000 pairs, so every pair count matched
    and nothing about the shortfall was visible in them.
    """
    rng = np.random.default_rng(12)
    ensemble = distance_ensemble(
        unrelated(rng, 6000), [unrelated(rng, 6000) for _ in range(19)], null=NULL,
        n_observed_signatures=69580, n_null_signatures=[16000] * 19)
    warned = [row for row in ensemble.warnings() if "removed the structure" in row]
    assert warned and "69580" in warned[0] and "16000" in warned[0]
    assert not any("median null population" in row for row in ensemble.warnings())
    assert ensemble.describe()["n_observed_signatures"] == 69580


def test_a_signature_count_for_every_member_or_none_at_all():
    rng = np.random.default_rng(12)
    with pytest.raises(InvalidParameterError) as raised:
        distance_ensemble(unrelated(rng, 100), [unrelated(rng, 100) for _ in range(19)],
                          null=NULL, n_observed_signatures=100, n_null_signatures=[10, 20])
    assert "one signature count per null member" in str(raised.value)


def test_the_ensemble_record_names_every_member_size(planted):
    record = planted.describe()
    assert record["n_observed_pairs"] == 6000
    assert len(record["null_pairs_per_member"]) == 19
    assert record["null"]["name"] == "constructed"


# --------------------------------------------------------------------------------------------
# distances between signatures
# --------------------------------------------------------------------------------------------


def signature(rng, centre, jitter=0.0, family=FAMILY):
    geometry, strengths, scales = centre

    def jittered(values):
        return tuple(np.asarray(values) * np.exp(rng.normal(0.0, jitter, len(values))))

    return SignaturePoint(family=family, geometry=jittered(geometry), bearings=None,
                          strengths=jittered(strengths), scales=jittered(scales))


def centre(rng, family=FAMILY):
    nodes = family.cardinality
    edges = nodes * (nodes - 1) // 2
    return (rng.uniform(0.5, 3.0, edges), rng.uniform(0.5, 3.0, nodes),
            rng.uniform(50.0, 400.0, nodes))


def population(rng, n_recurring, per_group, jitter, family=FAMILY):
    points, labels = [], []
    for group in range(n_recurring):
        anchor = centre(rng, family)
        for _ in range(per_group):
            points.append(signature(rng, anchor, jitter, family))
            labels.append(group)
    return points, labels


@pytest.mark.parametrize("n_points", [2, 3, 7, 40])
def test_the_pair_index_agrees_with_enumerating_the_pairs(n_points):
    assert [pair_at(n_points, index)
            for index in range(n_points * (n_points - 1) // 2)] == list(
        itertools.combinations(range(n_points), 2))


def test_the_pair_index_round_trips_at_the_scale_it_exists_for():
    """69,580 signatures make 2.42 billion pairs, and the inversion takes a float square root.

    An off-by-one at some index would not fail; it would silently pair the wrong signatures and
    report a distance distribution of the wrong population. So the formula is checked at the
    size it was written for, against the triangular number it inverts.
    """
    rng = np.random.default_rng(0)
    for n_points in (16000, 69580, 200000):
        total = n_points * (n_points - 1) // 2
        indices = [0, 1, total - 2, total - 1] + [
            int(value) for value in rng.integers(0, total, 2000)]
        for index in indices:
            left, right = pair_at(n_points, index)
            assert 0 <= left < right < n_points
            back = (total - (n_points - left) * (n_points - left - 1) // 2
                    + (right - left - 1))
            assert back == index


@pytest.mark.parametrize("index", [-1, 15])
def test_a_pair_index_outside_the_pair_space_is_refused(index):
    with pytest.raises(InvalidParameterError):
        pair_at(6, index)


def test_fewer_than_two_points_have_no_pairs():
    with pytest.raises(InvalidParameterError):
        pair_at(1, 0)


def test_full_enumeration_matches_the_pairs_it_claims_to_enumerate():
    rng = np.random.default_rng(5)
    points = [signature(rng, centre(rng)) for _ in range(8)]
    expected = [METRIC.distance(left, right)
                for left, right in itertools.combinations(points, 2)]
    assert list(signature_distances(points, metric=METRIC)) == expected


def test_sampling_pairs_without_a_seed_is_refused():
    rng = np.random.default_rng(5)
    points = [signature(rng, centre(rng)) for _ in range(20)]
    with pytest.raises(InvalidParameterError) as raised:
        signature_distances(points, metric=METRIC, max_pairs=10)
    assert "nobody can redraw" in str(raised.value)


def test_sampling_pairs_is_reproducible_and_returns_what_was_asked_for():
    rng = np.random.default_rng(5)
    points = [signature(rng, centre(rng)) for _ in range(30)]
    first = signature_distances(points, metric=METRIC, max_pairs=40, seed=1)
    again = signature_distances(points, metric=METRIC, max_pairs=40, seed=1)
    other = signature_distances(points, metric=METRIC, max_pairs=40, seed=2)
    assert len(first) == 40 and first == again and first != other


def test_sampling_draws_from_the_whole_pair_space_and_not_from_a_prefix():
    # Observation order is time. A prefix of the pair index space is the earliest part of the
    # record, so a sampler that took the first `max_pairs` pairs would report the distance
    # distribution of the record's opening rather than of the record.
    rng = np.random.default_rng(5)
    points = [signature(rng, centre(rng)) for _ in range(60)]
    tail = signature(rng, centre(rng))
    points.append(tail)
    # Every pair involving the last point sits at the very end of the index space.
    with_tail = set(round(METRIC.distance(tail, other), 12) for other in points[:-1])
    sampled = set(round(value, 12)
                  for value in signature_distances(points, metric=METRIC,
                                                   max_pairs=300, seed=9))
    assert with_tail & sampled


def test_one_signature_cannot_make_a_pair():
    rng = np.random.default_rng(5)
    with pytest.raises(InvalidParameterError):
        signature_distances([signature(rng, centre(rng))], metric=METRIC)


def test_the_permutation_null_keeps_every_component_marginal():
    rng = np.random.default_rng(7)
    points = [signature(rng, centre(rng)) for _ in range(40)]
    permuted = permuted_signature_population(points, seed=3)
    for name in ("geometry", "strengths", "scales"):
        for index in range(3):
            before = sorted(round(point.blocks()[name][index], 12) for point in points)
            after = sorted(round(point.blocks()[name][index], 12) for point in permuted)
            assert before == after


def test_the_permutation_null_refuses_to_mix_families():
    rng = np.random.default_rng(7)
    pair_family = SignatureFamily(mode="scale_specific", cardinality=2, has_bearings=False,
                                  scale_units="km")
    other = SignaturePoint(family=pair_family, geometry=(1.0,), bearings=None,
                           strengths=(1.0, 2.0), scales=(10.0, 20.0))
    with pytest.raises(InvalidParameterError) as raised:
        permuted_signature_population([signature(rng, centre(rng)), other], seed=1)
    assert "different quantities" in str(raised.value)


# --------------------------------------------------------------------------------------------
# the sweep and what it can resolve
# --------------------------------------------------------------------------------------------


def test_the_grid_reaches_the_closest_pair_rather_than_the_first_percentile(planted):
    grid = radius_grid(planted, n_radii=64)
    smallest = min(planted.observed)
    assert grid[0] == pytest.approx(smallest)
    # Most of the sweep sits in the close-pair tail, which is where the two mixture components
    # are separated. A grid spaced evenly in quantile would put its first point at 1/64 -- above
    # the entire same-configuration mass of a record whose recurrence is 1% of pairs.
    tenth = float(np.quantile(planted.observed, 0.10))
    assert sum(1 for value in grid if value <= tenth) > len(grid) // 2
    assert grid[0] < float(np.quantile(planted.observed, 1.0 / 64.0))


def test_a_sweep_of_one_radius_is_a_point_and_is_refused(planted):
    with pytest.raises(InvalidParameterError):
        radius_grid(planted, n_radii=1)


def test_the_grid_resolution_is_its_quantile_step_and_inverts():
    assert grid_resolution(4000, 64) == pytest.approx(4000 ** (1.0 / 63.0) - 1.0)
    needed = radii_for_resolution(4000, 0.10)
    assert grid_resolution(4000, needed) <= 0.10
    assert grid_resolution(4000, needed - 1) > 0.10


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2])
def test_a_resolution_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(InvalidParameterError):
        radii_for_resolution(4000, bad)


def test_the_excess_is_the_two_fractions_subtracted_and_nothing_else():
    ensemble = distance_ensemble([0.1, 0.2, 0.3, 0.4],
                                 [[0.15, 0.5, 0.6, 0.7], [0.05, 0.5, 0.6, 0.7]], null=NULL)
    row, = excess_curve(ensemble, radii=[0.25])
    assert row.observed_fraction == pytest.approx(0.5)
    assert row.null_fraction == pytest.approx(0.25)
    assert row.excess == pytest.approx(0.25)


def test_a_per_radius_p_value_is_one_plus_k_over_one_plus_n_and_never_k_over_n():
    ensemble = distance_ensemble([0.1, 0.2], [[0.9, 0.95]] * 4, null=NULL)
    row, = excess_curve(ensemble, radii=[0.5])
    # No null member reaches the observed fraction, so k = 0 and the p-value is at its floor
    # rather than at zero, which no finite ensemble can support.
    assert row.p_value == pytest.approx(1.0 / 5.0)


# --------------------------------------------------------------------------------------------
# does the record depart from the null at all?
# --------------------------------------------------------------------------------------------


def test_a_record_drawn_from_its_own_null_does_not_separate(pure_null):
    points = excess_curve(pure_null, n_radii=64)
    assert not separation_test(pure_null, points).significant


def test_a_record_with_a_planted_recurrence_separates(planted):
    points = excess_curve(planted, n_radii=64)
    found = separation_test(planted, points)
    assert found.significant and found.p_value == pytest.approx(1.0 / 20.0)
    # The maximum excess recovers the planted fraction: 6% of pairs were made close.
    assert found.statistic == pytest.approx(0.06, abs=0.01)


def test_one_null_member_cannot_supply_its_own_leave_one_out_distribution():
    ensemble = distance_ensemble([0.1, 0.2], [[0.5, 0.6]], null=NULL)
    with pytest.raises(InvalidParameterError) as raised:
        separation_test(ensemble, excess_curve(ensemble, n_radii=4))
    assert "no others to be scored against" in str(raised.value)


def test_a_member_is_scored_against_the_others_and_not_against_itself():
    # One member is far closer than the rest. Scored against the mean of all members -- itself
    # included -- its own contribution would flatter it and shrink its statistic; scored
    # leave-one-out it stands out, which is what makes the band honest.
    members = [[0.05] * 100] + [[0.9] * 100 for _ in range(9)]
    ensemble = distance_ensemble([0.9] * 100, members, null=NULL)
    found = separation_test(ensemble, excess_curve(ensemble, radii=[0.5]))
    assert max(found.null_statistics) == pytest.approx(1.0)


def test_the_band_is_the_upper_quantile_of_the_leave_one_out_statistics(planted):
    points = excess_curve(planted, n_radii=32)
    found = separation_test(planted, points, alpha=0.05)
    assert found.critical_value == pytest.approx(
        float(np.quantile(np.asarray(found.null_statistics), 0.95, method="higher")))


def test_the_report_says_how_many_surrogates_a_corrected_sweep_would_have_needed(planted):
    found = separation_test(planted, excess_curve(planted, n_radii=64))
    # This is why the sweep is one maximum statistic rather than 64 corrected tests: the
    # ensemble that a corrected sweep needs is thousands of pipeline re-runs.
    assert found.surrogates_a_corrected_sweep_would_need > 1000
    assert "clean-looking negative" in found.describe()["why_not_a_corrected_sweep"] \
        or "clean negative" in found.describe()["why_not_a_corrected_sweep"]


@pytest.mark.parametrize("bad", [0.0, 1.0, 2.0])
def test_a_significance_level_outside_the_unit_interval_is_refused(planted, bad):
    with pytest.raises(InvalidParameterError):
        separation_test(planted, excess_curve(planted, n_radii=8), alpha=bad)


# --------------------------------------------------------------------------------------------
# the mixture fraction
# --------------------------------------------------------------------------------------------


def test_the_mixture_fraction_recovers_a_planted_one(planted):
    points = excess_curve(planted, n_radii=128)
    found = estimate_recurrence_fraction(
        points, critical_value=separation_test(planted, points).critical_value)
    assert found.stable
    assert found.value == pytest.approx(0.06, abs=0.005)


def test_a_fraction_read_at_one_radius_alone_is_unstable_rather_than_perfect(planted):
    points = excess_curve(planted, n_radii=128)
    band = max(row.excess for row in points) * 0.999
    found = estimate_recurrence_fraction(points, critical_value=band)
    assert not found.stable and found.spread == 0.0
    assert "nothing to be checked against" in found.reason


def test_a_band_no_radius_clears_leaves_nowhere_to_read(planted):
    points = excess_curve(planted, n_radii=64)
    found = estimate_recurrence_fraction(points, critical_value=0.99)
    assert not found.stable and found.read_at == ()
    assert "nowhere on this curve" in found.reason
    # What it reports is what the curve reached, not zero. Zero would read as "no recurrence"
    # when the truth is that the fraction was not read at all.
    assert found.value == found.maximum_excess > 0.0


def test_the_cap_keeps_the_read_out_of_the_bulk(planted):
    points = excess_curve(planted, n_radii=128)
    band = separation_test(planted, points).critical_value
    capped = estimate_recurrence_fraction(points, critical_value=band,
                                          null_fraction_cap=DEFAULT_NULL_FRACTION_CAP)
    uncapped = estimate_recurrence_fraction(points, critical_value=band,
                                            null_fraction_cap=0.999)
    # Reading into the bulk adds radii, and every one of them is read where the record's own
    # unrelated component is a sample of however many configurations it happens to contain --
    # an error of order 1/sqrt(C) against a fraction of order 1/C.
    assert len(uncapped.read_at) > len(capped.read_at)
    assert all(fraction <= DEFAULT_NULL_FRACTION_CAP for _r, fraction, _v in capped.read_at)
    assert any(fraction > DEFAULT_NULL_FRACTION_CAP for _r, fraction, _v in uncapped.read_at)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_a_cap_outside_the_unit_interval_is_refused(planted, bad):
    with pytest.raises(InvalidParameterError):
        estimate_recurrence_fraction(excess_curve(planted, n_radii=8),
                                     critical_value=0.01, null_fraction_cap=bad)


def test_a_non_positive_stability_tolerance_is_refused(planted):
    with pytest.raises(InvalidParameterError):
        estimate_recurrence_fraction(excess_curve(planted, n_radii=8),
                                     critical_value=0.01, tolerance=0.0)


def test_an_excess_that_is_never_positive_has_no_mixture_to_describe():
    rng = np.random.default_rng(2)
    ensemble = distance_ensemble(list(rng.uniform(0.6, 1.0, 400)),
                                 [list(rng.uniform(0.0, 1.0, 400)) for _ in range(19)],
                                 null=NULL)
    found = estimate_recurrence_fraction(excess_curve(ensemble, n_radii=32),
                                         critical_value=0.0)
    assert not found.stable and found.maximum_excess <= 0.0


# --------------------------------------------------------------------------------------------
# the calibration, and every way it refuses
# --------------------------------------------------------------------------------------------


def test_a_record_with_no_recurrence_gets_no_radius(pure_null):
    found = calibrate_against_null(pure_null, target_rate=0.10)
    assert found.status == CALIBRATION_NO_SEPARATION
    assert found.radius is None and not found.measured
    assert "not a failure of the search" in found.note


def test_a_planted_recurrence_gets_a_radius_with_both_rates(planted):
    found = calibrate_against_null(planted, target_rate=0.10)
    assert found.status == CALIBRATION_MEASURED and found.measured
    assert found.radius is not None
    assert found.rates.contamination_rate <= 0.10
    assert found.rates.relative_loss <= 0.10
    # The planted close pairs run to 0.05 and the unrelated ones start at 0.30.
    assert 0.03 <= found.radius <= 0.30


def test_the_radius_chosen_is_the_smallest_that_qualifies_not_the_first_or_the_best(planted):
    found = calibrate_against_null(planted, target_rate=0.10)
    qualifying = [row for row in found.curve
                  if row.above_band and row.contamination_rate <= 0.10
                  and row.relative_loss <= 0.10]
    assert found.radius == min(row.radius for row in qualifying)


def test_a_target_no_radius_holds_is_refused_with_the_best_achievable_published():
    # The two components overlap, so every radius that reaches the recurring pairs also admits
    # unrelated ones. There is no clean operating point, and that is a fact about the signature
    # rather than about the search.
    rng = np.random.default_rng(31)
    overlapping = np.concatenate([rng.uniform(0.0, 0.60, 360), unrelated(rng, 5640)])
    found = calibrate_against_null(ensemble_of(overlapping, seed=32), target_rate=0.05)
    assert found.status == CALIBRATION_NO_OPERATING_POINT
    assert found.radius is None
    assert found.best_achievable is not None
    assert found.best_achievable.worst_rate > 0.05
    assert "at a laxer level" in found.note


def test_a_fraction_that_cannot_be_read_stably_is_refused(planted):
    # A stability tolerance of nothing at all: no two radii will agree to that precision.
    found = calibrate_against_null(planted, target_rate=0.10, stability_tolerance=1e-12)
    assert found.status == CALIBRATION_UNSTABLE
    assert found.radius is None
    assert "would inherit that arbitrariness" in found.note


def test_every_status_the_module_publishes_is_one_it_can_actually_return():
    assert len(set(CALIBRATION_STATUSES)) == len(CALIBRATION_STATUSES) == 4


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, True])
def test_a_target_rate_that_is_not_a_rate_is_refused(planted, bad):
    with pytest.raises(InvalidParameterError):
        calibrate_against_null(planted, target_rate=bad)


def test_an_ensemble_of_the_wrong_type_is_refused():
    with pytest.raises(InvalidParameterError):
        calibrate_against_null([0.1, 0.2], target_rate=0.1)


def test_a_grid_too_coarse_for_the_target_is_refused_and_says_how_coarse(planted):
    with pytest.raises(InvalidParameterError) as raised:
        calibrate_against_null(planted, target_rate=0.10, n_radii=64)
    message = str(raised.value)
    assert "at least %d radii" % radii_for_resolution(6000, 0.10) in message
    assert "about the grid rather than about the record" in message


def test_the_grid_defaults_to_whatever_resolves_the_declared_target(planted):
    found = calibrate_against_null(planted, target_rate=0.10)
    assert found.grid_resolution <= 0.10
    assert len(found.curve) <= radii_for_resolution(6000, 0.10)


def test_the_receipt_carries_the_whole_sweep_it_chose_from(planted):
    # A receipt stating a radius without the sweep behind it asks to be believed rather than
    # checked, and every refusal here is a claim about the shape of that sweep.
    found = calibrate_against_null(planted, target_rate=0.10)
    record = found.describe()
    assert len(record["curve"]) == len(found.curve)
    assert [row["radius"] for row in record["curve"]] == [row.radius for row in found.curve]
    assert record["curve"][0]["contamination_rate"] == found.curve[0].contamination_rate


def test_the_record_carries_the_schema_the_null_and_the_boundary(planted):
    record = calibrate_against_null(planted, target_rate=0.10).describe()
    assert record["schema"] == NULL_CALIBRATION_SCHEMA
    assert record["ensemble"]["null"]["name"] == "constructed"
    assert "identifiable" in record["claim_boundary"]
    assert record["separation"]["p_value"] <= 0.05
    assert record["recurrence_fraction"]["value"] > 0.0


def test_no_absolute_split_rate_is_published_anywhere_in_the_record(planted):
    # T4E.6 found the replicate calibration's split rate to be zero by construction. The fix is
    # not a different estimate of the same quantity: without labels or replicates the absolute
    # split rate is not identifiable at all, and publishing one optimistic by six-fold would be
    # worse than publishing none. If a later task makes it identifiable, this test should fail.
    record = calibrate_against_null(planted, target_rate=0.10).describe()
    assert "estimated_split_rate" not in record["rates"]
    assert record["split_rate"] == SPLIT_RATE_NOTE
    assert "not identifiable" in SPLIT_RATE_NOTE
    assert all("estimated_split_rate" not in row for row in record["curve"])


def test_the_contamination_estimate_uses_the_null_and_the_fraction_and_nothing_else(planted):
    # contamination = F_null * (1 - pi) / F_obs, capped at one. Nothing about the search enters
    # it: it is arithmetic on two measured fractions and one estimated one.
    found = calibrate_against_null(planted, target_rate=0.10)
    for row in found.curve:
        if row.observed_fraction > 0.0:
            assert row.contamination_rate <= 1.0
            assert row.contamination_rate >= min(
                1.0, row.admission_rate * (1.0 - found.recurrence.value)
                / row.observed_fraction) - 1e-12


def test_both_rates_move_in_opposite_directions_as_the_radius_grows(planted):
    # A radius that admits more must lose less. If both moved the same way the two rates would
    # not be a trade-off and there would be nothing for an operating point to balance.
    found = calibrate_against_null(planted, target_rate=0.10)
    losses = [row.relative_loss for row in found.curve]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(losses, losses[1:]))
    assert losses[0] > losses[-1]
    admissions = [row.admission_rate for row in found.curve]
    assert all(later >= earlier - 1e-12
               for earlier, later in zip(admissions, admissions[1:]))


def test_the_answer_depends_on_the_null_it_was_measured_against(planted):
    # An excess over one null is not an excess over another. A calibration that returned the
    # same radius whatever the null would not be using the null at all.
    rng = np.random.default_rng(77)
    closer = distance_ensemble(
        planted.observed, [mixture(rng, 6000, 0.06) for _ in range(19)], null=NULL)
    assert calibrate_against_null(closer, target_rate=0.10).status == CALIBRATION_NO_SEPARATION



# --------------------------------------------------------------------------------------------
# boundaries, exactly on them
# --------------------------------------------------------------------------------------------


def point(radius, observed, null, members=(0.0,)):
    """One row of an excess curve, built by hand so a boundary can be landed on exactly."""
    return ExcessPoint(radius=radius, observed_fraction=observed, null_fraction=null,
                       excess=observed - null, null_member_fractions=tuple(members),
                       p_value=0.05)


def test_an_excess_exactly_at_the_band_is_not_read():
    # The band is the level an excess must *clear*. An excess equal to it is the level itself,
    # which the null ensemble reached, so reading a mixture fraction there would read the band.
    rows = [point(0.1, 0.20, 0.00), point(0.2, 0.40, 0.00)]
    found = estimate_recurrence_fraction(rows, critical_value=0.20)
    assert [radius for radius, _f, _v in found.read_at] == [0.2]


def test_a_null_fraction_exactly_at_the_cap_is_still_read():
    # The cap is the largest null fraction the mixture is still conditioned at, so a radius
    # sitting on it is inside the region rather than past it.
    rows = [point(0.1, 0.50, 0.25), point(0.2, 0.60, 0.26), point(0.3, 0.70, 0.20)]
    found = estimate_recurrence_fraction(rows, critical_value=0.10, null_fraction_cap=0.25)
    assert sorted(radius for radius, _f, _v in found.read_at) == [0.1, 0.3]


def test_a_radius_whose_excess_equals_the_band_is_not_above_it():
    # Identical null members put every leave-one-out statistic at exactly zero, and fractions
    # that are exact binary quarters keep it there, so the band is zero and every radius where
    # the record matches the null sits exactly on it rather than near it.
    ensemble = distance_ensemble([0.1, 0.2, 0.3, 0.4], [[0.1, 0.2, 0.3, 0.9]] * 19, null=NULL)
    found = calibrate_against_null(ensemble, target_rate=0.5)
    assert found.separation.critical_value == 0.0
    on_the_band = [row for row in found.curve if row.excess == 0.0]
    assert on_the_band
    assert all(not row.above_band for row in on_the_band)
    assert any(row.above_band for row in found.curve)


def test_a_null_member_that_ties_the_observed_fraction_counts_against_it():
    # A tie is not evidence against the null, so it is counted for it. Counting only strict
    # exceedances would make every tie a point in the record's favour.
    ensemble = distance_ensemble([0.1, 0.2], [[0.1, 0.9]] + [[0.8, 0.9]] * 3, null=NULL)
    row, = excess_curve(ensemble, radii=[0.15])
    assert row.observed_fraction == pytest.approx(0.5)
    assert row.p_value == pytest.approx(2.0 / 5.0)


def test_a_distance_exactly_at_a_radius_is_inside_it():
    # Every radius on the grid *is* an observed distance, so excluding the boundary would put
    # each grid point one pair below where it says it is.
    ensemble = distance_ensemble([0.1, 0.2, 0.3], [[0.9]], null=NULL)
    row, = excess_curve(ensemble, radii=[0.2])
    assert row.observed_fraction == pytest.approx(2.0 / 3.0)


def test_the_sweep_never_visits_the_same_radius_twice(planted):
    grid = radius_grid(planted, n_radii=200)
    assert list(grid) == sorted(set(grid))
    assert all(later > earlier for earlier, later in zip(grid, grid[1:]))


# --------------------------------------------------------------------------------------------
# the rates, at their exact values rather than under a bound
# --------------------------------------------------------------------------------------------


def test_the_contamination_estimate_is_tightened_by_the_mixture_fraction(planted):
    # Chance explains F_null of all pairs, but only 1 - pi of the record's pairs are chance's
    # to explain. Dropping that factor would leave the bound loose by exactly pi.
    found = calibrate_against_null(planted, target_rate=0.10)
    row = max((row for row in found.curve if row.observed_fraction > 0.0),
              key=lambda row: row.admission_rate)
    assert row.contamination_rate == pytest.approx(
        min(1.0, row.admission_rate * (1.0 - found.recurrence.value)
            / row.observed_fraction))
    assert found.recurrence.value > 0.01


def test_the_contamination_estimate_is_capped_at_certainty():
    # A record whose pairs sit further apart than its null has fewer close pairs than chance
    # would give, so the ratio exceeds one wherever that holds. A rate above one is not a rate.
    rng = np.random.default_rng(88)
    ensemble = distance_ensemble(rng.uniform(0.40, 1.00, 3000),
                                 [rng.uniform(0.30, 1.00, 3000) for _ in range(19)], null=NULL)
    found = calibrate_against_null(ensemble, target_rate=0.10)
    share = 1.0 - (0.0 if found.recurrence is None else found.recurrence.value)
    uncapped = [row.admission_rate * share / row.observed_fraction
                for row in found.curve if row.observed_fraction > 0.0]
    assert max(uncapped) > 1.0
    assert all(row.contamination_rate <= 1.0 for row in found.curve)


def test_the_contamination_estimate_can_sit_below_the_truth_and_is_not_a_bound():
    """The limitation, pinned as a measurement rather than left as a caveat.

    An earlier draft of the module called this rate an upper bound. It is not one. The
    expression puts the null's close-pair fraction where the record's *own* unrelated-pair
    distribution belongs, and a record made of few enough distinct configurations samples that
    distribution too coarsely for the two to agree. Here fifteen configurations are each
    observed ten times, the calibration sees no labels, and the published rate is compared
    against the truth at every radius on its own sweep.

    If a later change makes the rate a genuine bound, this test will fail, and the claim in the
    module docstring should be strengthened rather than this test deleted.
    """
    rng = np.random.default_rng(2024)
    points, labels = population(rng, 15, 10, 0.02, PAIR_FAMILY)
    nulls = [population(np.random.default_rng(300 + index), 150, 1, 0.02, PAIR_FAMILY)[0]
             for index in range(19)]
    ensemble = distance_ensemble(
        signature_distances(points, metric=METRIC),
        [signature_distances(member, metric=METRIC, max_pairs=1200, seed=index)
         for index, member in enumerate(nulls)],
        null=SURROGATE_RECORD_NULL, metric_digest=METRIC.digest)
    found = calibrate_against_null(ensemble, target_rate=0.10)

    distances = np.asarray([METRIC.distance(first, second) for first, second
                            in itertools.combinations(points, 2)])
    same = np.asarray([labels[left] == labels[right] for left, right
                       in itertools.combinations(range(len(points)), 2)])
    below = 0
    for row in found.curve:
        inside = distances <= row.radius
        if not inside.any():
            continue
        truth = float((inside & ~same).sum()) / float(inside.sum())
        if row.contamination_rate < truth - 1e-12:
            below += 1
    assert below > 0, (
        "the published contamination rate never fell below the truth on this record, which "
        "would make it a bound here; the module says it is an estimate")


def test_the_loss_is_never_negative_even_where_the_excess_overshoots(planted):
    # The excess peaks above the fraction it is read as, because a maximum of a noisy
    # difference is biased upwards. A radius reaching more than everything is not reaching
    # minus something.
    found = calibrate_against_null(planted, target_rate=0.10)
    assert max(row.excess for row in found.curve) > found.recurrence.value
    assert all(row.relative_loss >= 0.0 for row in found.curve)


def test_the_loss_names_the_denominator_it_was_actually_measured_against(planted, pure_null):
    # On the measured path the denominator is the estimated fraction; on a refusal path no
    # fraction exists and the largest excess stands in for it. A row that claimed the first
    # while using the second would be describing a number it did not report.
    measured = calibrate_against_null(planted, target_rate=0.10)
    assert measured.status == CALIBRATION_MEASURED
    assert "estimated recurrence fraction" in measured.curve[0].loss_reference
    assert "estimated recurrence fraction" in measured.curve[0].describe()["relative_loss_is"]

    refused = calibrate_against_null(pure_null, target_rate=0.10)
    assert refused.status == CALIBRATION_NO_SEPARATION
    assert "no mixture fraction having been estimated" in refused.curve[0].loss_reference


def test_the_best_achievable_is_the_smallest_worst_rate_and_not_the_largest(planted):
    found = calibrate_against_null(planted, target_rate=0.10)
    eligible = [row for row in found.curve if row.above_band]
    assert found.best_achievable.worst_rate == min(row.worst_rate for row in eligible)


def test_a_spread_exactly_at_the_tolerance_is_stable(planted):
    # The tolerance is how far the fraction may move, so a fraction that moves exactly that far
    # has not moved further.
    points = excess_curve(planted, n_radii=128)
    band = separation_test(planted, points).critical_value
    measured = estimate_recurrence_fraction(points, critical_value=band)
    assert measured.spread > 0.0
    on_the_line = estimate_recurrence_fraction(points, critical_value=band,
                                               tolerance=measured.spread)
    assert on_the_line.stable
    assert not estimate_recurrence_fraction(
        points, critical_value=band, tolerance=measured.spread * 0.999).stable


def test_a_non_positive_mixture_fraction_is_a_status_nothing_can_reach():
    # It was one until mutation testing asked what would produce it. The largest observed
    # distance puts F_obs at 1 and F_null at most 1, so the maximum excess is non-negative on
    # every input; it is zero only when every null distance lies inside the record's own range,
    # and then every leave-one-out statistic is non-negative too and the p-value is 1. A
    # refusal nobody can reach is not a refusal, so the status was removed.
    from src.analysis_engine import spectral_null_calibration as module
    assert not hasattr(module, "CALIBRATION_NO_RECURRENCE")
    assert len(CALIBRATION_STATUSES) == 4
    rng = np.random.default_rng(6)
    for _ in range(50):
        ensemble = distance_ensemble(
            rng.uniform(0.0, 1.0, 300),
            [rng.uniform(0.0, 1.2, 300) for _ in range(19)], null=NULL)
        points = excess_curve(ensemble, n_radii=32)
        assert max(row.excess for row in points) >= 0.0

# --------------------------------------------------------------------------------------------
# end to end, on signatures
# --------------------------------------------------------------------------------------------


def test_the_chosen_radius_recovers_a_planted_grouping_in_signature_space():
    """The acceptance test: real signatures, a planted identity, and no labels in sight.

    Twelve configurations are each observed ten times with 2% jitter; the null populations are
    the same number of signatures with no configuration observed twice. The calibration sees
    only distances. It is then checked against the labels it never had.
    """
    rng = np.random.default_rng(20260908)
    points, labels = population(rng, 12, 10, 0.02)
    nulls = [population(np.random.default_rng(900 + index), 120, 1, 0.02)[0]
             for index in range(19)]
    ensemble = distance_ensemble(
        signature_distances(points, metric=METRIC),
        [signature_distances(member, metric=METRIC, max_pairs=1200, seed=index)
         for index, member in enumerate(nulls)],
        null=SignatureNull(name="independent_configuration_draw",
                           preserves=("the distribution configurations are drawn from",),
                           destroys=("recurrence: no configuration is observed twice",)),
        metric_digest=METRIC.digest)
    found = calibrate_against_null(ensemble, target_rate=0.10)
    assert found.status == CALIBRATION_MEASURED, found.note

    same_inside = same = different_inside = different = 0
    for (left, first), (right, second) in itertools.combinations(list(enumerate(points)), 2):
        inside = METRIC.distance(first, second) <= found.radius
        if labels[left] == labels[right]:
            same += 1
            same_inside += inside
        else:
            different += 1
            different_inside += inside
    # The radius was chosen without labels; against the labels it groups the planted
    # configurations and admits none of the pairs that are not one.
    assert different_inside == 0
    assert same_inside >= 0.85 * same
    assert found.recurrence.value == pytest.approx(same / (same + different), abs=0.01)


def test_a_signature_record_with_no_recurrence_is_refused_end_to_end():
    """The refusal path on the same machinery: every configuration observed exactly once."""
    rng = np.random.default_rng(4242)
    points, _labels = population(rng, 120, 1, 0.02)
    nulls = [population(np.random.default_rng(700 + index), 120, 1, 0.02)[0]
             for index in range(19)]
    ensemble = distance_ensemble(
        signature_distances(points, metric=METRIC, max_pairs=1200, seed=1),
        [signature_distances(member, metric=METRIC, max_pairs=1200, seed=index)
         for index, member in enumerate(nulls)],
        null=SURROGATE_RECORD_NULL, metric_digest=METRIC.digest)
    found = calibrate_against_null(ensemble, target_rate=0.10)
    assert found.status == CALIBRATION_NO_SEPARATION
    assert found.radius is None


def test_the_metric_digest_travels_so_a_radius_cannot_be_read_under_another_metric(planted):
    assert planted.describe()["metric_digest"] == "digest"
    assert len(METRIC.digest) == 64
    assert math.isfinite(grid_resolution(6000, 100))
