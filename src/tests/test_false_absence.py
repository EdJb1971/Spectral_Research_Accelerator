"""T4E.24: the false-absence measurement, and the properties its numbers depend on.

Every test here pins something a wrong answer would quietly corrupt: a pairing rule that lets
one extraction answer for two plantings would under-count absences; a configuration redrawn per
scene would make the presence count meaningless; an admission rate that is not monotone in the
tolerance would be arithmetically impossible and therefore a bug.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmarks.false_absence import (
    DECLARED_COUNT_RANGE,
    DECLARED_EDGE_MARGIN_CELLS,
    DECLARED_MINIMUM_SEPARATION_CELLS,
    DECLARED_PAIRING_RADIUS_CELLS,
    DECLARED_RATIO_RANGE,
    DECLARED_SCALE_RANGE,
    DECLARED_SCENES_PER_CONFIGURATION,
    DECLARED_STRETCHES,
    PlantedFeature,
    admission_rates,
    background_scale,
    dispersion,
    draw_configuration,
    pair_plantings,
    plant,
    presence_distribution,
)


def _feature(row, col, sigma=3.0, ratio=16.0, stretch=1.0, angle=0.0):
    return PlantedFeature(row=row, col=col, sigma=sigma, ratio=ratio,
                          stretch=stretch, angle=angle)


# ---------------- the envelope is T4E.21's, unchanged


def test_the_planting_envelope_is_the_one_t4e21_declared():
    """This result speaks to T4E.21's regime only if it plants into the same one."""
    assert DECLARED_COUNT_RANGE == (3, 10)
    assert DECLARED_SCALE_RANGE == (1.73, 11.32)
    assert DECLARED_RATIO_RANGE == (8.0, 32.0)
    assert DECLARED_STRETCHES == (1.0, 1.5, 2.5)
    assert DECLARED_MINIMUM_SEPARATION_CELLS == 8.0
    assert DECLARED_EDGE_MARGIN_CELLS == 12.0


def test_the_pairing_radius_equals_the_minimum_separation():
    """The reason the rule works: a planting can never reach its neighbour's feature."""
    assert DECLARED_PAIRING_RADIUS_CELLS == DECLARED_MINIMUM_SEPARATION_CELLS


def test_the_partition_size_is_the_one_the_criterion_is_measured_at():
    assert DECLARED_SCENES_PER_CONFIGURATION == 6


# ---------------- the configuration is drawn once, which is the whole design


def test_a_configuration_respects_the_declared_separation_and_margin():
    rng = np.random.default_rng(11)
    for _ in range(20):
        features = draw_configuration(rng, (161, 161))
        for i, f in enumerate(features):
            assert DECLARED_EDGE_MARGIN_CELLS <= f.row <= 161 - DECLARED_EDGE_MARGIN_CELLS
            assert DECLARED_EDGE_MARGIN_CELLS <= f.col <= 161 - DECLARED_EDGE_MARGIN_CELLS
            for g in features[i + 1:]:
                assert math.hypot(f.row - g.row, f.col - g.col) >= (
                    DECLARED_MINIMUM_SEPARATION_CELLS - 1e-9)


def test_every_drawn_parameter_lands_inside_its_declared_range():
    rng = np.random.default_rng(12)
    for _ in range(20):
        for f in draw_configuration(rng, (161, 161)):
            assert DECLARED_SCALE_RANGE[0] <= f.sigma <= DECLARED_SCALE_RANGE[1]
            assert DECLARED_RATIO_RANGE[0] <= f.ratio <= DECLARED_RATIO_RANGE[1]
            assert f.stretch in DECLARED_STRETCHES
            assert 0.0 <= f.angle <= math.pi


def test_the_same_configuration_is_planted_into_every_scene():
    """The background is the only thing allowed to vary between the S scenes."""
    rng = np.random.default_rng(13)
    features = draw_configuration(rng, (64, 64))
    scenes = [plant(np.random.default_rng(s).normal(size=(64, 64)), features) for s in range(6)]
    for values in scenes:
        assert values.shape == (64, 64)
    # The geometry is a property of `features`, not of any scene, so it cannot drift.
    assert all(f is g for f, g in zip(features, features))


def test_a_configuration_is_dropped_short_rather_than_crowded():
    """A frame with no room refuses to place the last feature instead of merging two."""
    rng = np.random.default_rng(14)
    features = draw_configuration(rng, (30, 30), count=DECLARED_COUNT_RANGE[1])
    for i, f in enumerate(features):
        for g in features[i + 1:]:
            assert math.hypot(f.row - g.row, f.col - g.col) >= (
                DECLARED_MINIMUM_SEPARATION_CELLS - 1e-9)


def test_a_frame_smaller_than_the_margins_is_refused_by_name():
    with pytest.raises(ValueError):
        draw_configuration(np.random.default_rng(15), (20, 20))


# ---------------- planting


def test_planting_puts_a_maximum_where_the_feature_was_planted():
    background = np.zeros((64, 64))
    values = plant(background, [_feature(30.0, 40.0, sigma=3.0)], scale=1.0)
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    assert peak == (30, 40)


def test_planting_leaves_the_background_untouched():
    background = np.random.default_rng(16).normal(size=(48, 48))
    original = background.copy()
    plant(background, [_feature(24.0, 24.0)], scale=1.0)
    assert np.array_equal(background, original)


def test_an_empty_configuration_returns_the_background():
    background = np.random.default_rng(17).normal(size=(32, 32))
    assert np.array_equal(plant(background, []), background)


def test_a_stretched_feature_is_wider_along_its_own_axis():
    """Stretch is applied along `angle`, so the axis it widens is the one declared."""
    values = plant(np.zeros((81, 81)), [_feature(40.0, 40.0, sigma=3.0, stretch=2.5, angle=0.0)],
                   scale=1.0)
    along = values[40, 40] - values[52, 40]
    across = values[40, 40] - values[40, 52]
    assert along < across, "the stretched axis should fall off more slowly"


def test_the_background_scale_is_robust_to_the_peaks_planted_on_it():
    background = np.random.default_rng(18).normal(size=(64, 64))
    before = background_scale(background)
    after = background_scale(plant(background, [_feature(32.0, 32.0, ratio=100.0)], scale=before))
    assert abs(after - before) / before < 0.15


# ---------------- pairing: the rule that decides what counts as an absence


def test_a_planting_with_nothing_inside_the_radius_is_an_absence():
    paired, offsets = pair_plantings([_feature(10.0, 10.0)], [(40.0, 40.0)])
    assert paired == (None,)
    assert math.isnan(offsets[0])


def test_one_extraction_cannot_answer_for_two_plantings():
    """Claiming is what stops a single strong feature hiding a genuine absence."""
    features = [_feature(20.0, 20.0), _feature(20.0, 28.0)]
    paired, _ = pair_plantings(features, [(20.0, 24.0)])
    assert sorted(p is None for p in paired) == [False, True]
    assert len([p for p in paired if p is not None]) == 1


def test_the_closest_pair_wins_regardless_of_planting_order():
    """Planting order is arbitrary, so it must not decide which planting is recorded absent."""
    a, b = _feature(20.0, 20.0), _feature(20.0, 26.0)
    forward, _ = pair_plantings([a, b], [(20.0, 25.0)])
    backward, _ = pair_plantings([b, a], [(20.0, 25.0)])
    assert forward[1] == 0 and forward[0] is None
    assert backward[0] == 0 and backward[1] is None


def test_a_planting_is_never_paired_beyond_the_declared_radius():
    edge = DECLARED_PAIRING_RADIUS_CELLS
    inside, _ = pair_plantings([_feature(20.0, 20.0)], [(20.0, 20.0 + edge - 0.01)])
    outside, _ = pair_plantings([_feature(20.0, 20.0)], [(20.0, 20.0 + edge + 0.01)])
    assert inside[0] == 0
    assert outside[0] is None


def test_no_extractions_makes_every_planting_an_absence():
    paired, offsets = pair_plantings([_feature(10.0, 10.0), _feature(30.0, 30.0)], [])
    assert paired == (None, None)
    assert all(math.isnan(o) for o in offsets)


def test_no_plantings_pairs_nothing():
    assert pair_plantings([], [(5.0, 5.0)]) == ((), ())


# ---------------- the reported quantities


def test_the_distribution_covers_every_count_from_zero_to_s():
    """Acceptance condition 2: the full distribution, never only its mean."""
    histogram = presence_distribution([0, 6, 6, 3], scenes=6)
    assert sorted(histogram) == sorted(str(k) for k in range(7))
    assert histogram["6"] == 2 and histogram["0"] == 1 and histogram["3"] == 1
    assert sum(histogram.values()) == 4


def test_admission_is_monotone_in_the_tolerance():
    """Admissions can only grow as k falls; anything else is a bug, not a finding."""
    counts = [6, 6, 5, 4, 3, 0, 2, 6]
    rates = admission_rates(counts, scenes=6)
    series = [rates["a=%d" % a]["admitted"] for a in (0, 1, 2, 3)]
    assert series == sorted(series)


def test_admission_rates_are_counts_over_the_observed_features():
    rates = admission_rates([6, 6, 5, 0], scenes=6)
    assert rates["a=0"] == {"k": 6, "admitted": 2, "of": 4, "rate": 0.5}
    assert rates["a=1"]["admitted"] == 3


def test_admission_on_no_features_is_not_a_number():
    assert math.isnan(admission_rates([], scenes=6)["a=1"]["rate"])


def test_independent_absences_land_inside_the_bootstrap_band():
    """The `if_independent_instead` arm of the declared prediction, on data built that way."""
    rng = np.random.default_rng(19)
    counts = rng.binomial(6, 0.68, size=400)
    result = dispersion(counts, scenes=6, draws=299, seed=5)
    assert result["outside_the_band"] is False
    assert result["direction"] == "inside the band"


def test_concentrated_absences_read_as_over_dispersed():
    """The `if_concentrated_as_predicted` arm: mass at 0 and at S, nothing between."""
    counts = [6] * 200 + [0] * 100
    result = dispersion(counts, scenes=6, draws=299, seed=5)
    assert result["outside_the_band"] is True
    assert result["direction"] == "over-dispersed"


def test_the_dispersion_report_carries_its_own_caveat():
    """The mechanism claim is the weak half and must not travel without saying so."""
    result = dispersion([6, 5, 4], scenes=6, draws=99, seed=5)
    assert "close to built in" in result["what_this_is_not"]
    assert result["what_this_is_not"].startswith("a test with a stated level")


def test_dispersion_on_no_features_refuses_rather_than_returning_zero():
    result = dispersion([], scenes=6)
    assert result["observed_variance"] is None
    assert "no features" in result["reason"]
