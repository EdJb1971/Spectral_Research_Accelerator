"""T4E.26: peeled-null calibration, and the properties that keep its numbers honest.

The risk this procedure carries is that it manufactures the features it then reports. Most of
what is pinned here is therefore about the contamination split -- the thing that stops a residual
lobe hiding inside a background rate -- and about the subtraction using only what the pipeline
actually published rather than a second, unaudited fit.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmarks.false_absence import PlantedFeature
from src.benchmarks.peeled_null import (
    ARTEFACT_RADIUS_IN_FITTED_WIDTHS,
    fitted_gaussian,
    gap_closed,
    nearest_planting_stretch,
    peel,
    split_contamination,
)


class _Feature:
    """The parts of a SpectralFeature this module reads, and nothing else."""

    def __init__(self, row, col, width, amplitude, baseline=0.0, drop=()):
        self.magnitude = type("Q", (), {"value": amplitude + baseline})()
        self.provenance = {"index_position": [row, col], "scale_cells": width,
                           "baseline": baseline}
        for key in drop:
            del self.provenance[key]


# ---------------- the shape that is subtracted is the one that was published


def test_the_subtracted_shape_peaks_where_the_feature_was_reported():
    field = fitted_gaussian((41, 41), (20.0, 25.0), amplitude=3.0, width=2.0)
    assert np.unravel_index(int(np.argmax(field)), field.shape) == (20, 25)
    assert field[20, 25] == pytest.approx(3.0)


def test_a_feature_with_no_usable_width_is_left_in_place_and_counted():
    """A subtraction with an invented width removes power the pipeline never claimed."""
    values = np.zeros((32, 32))
    residual, subtracted = peel(values, [_Feature(10, 10, 2.0, 1.0, drop=("scale_cells",))])

    assert subtracted == ()
    assert np.array_equal(residual, values)


def test_a_zero_or_negative_width_subtracts_nothing_rather_than_dividing_by_zero():
    assert not fitted_gaussian((9, 9), (4.0, 4.0), amplitude=1.0, width=0.0).any()
    assert not fitted_gaussian((9, 9), (4.0, 4.0), amplitude=1.0, width=-1.0).any()


def test_peeling_removes_the_amplitude_above_the_baseline_not_the_magnitude():
    """Subtracting the magnitude would remove the baseline too, at every feature."""
    values = np.full((33, 33), 5.0) + fitted_gaussian(
        (33, 33), (16.0, 16.0), amplitude=2.0, width=2.0)
    assert values[16, 16] == pytest.approx(7.0)

    residual, subtracted = peel(values, [_Feature(16, 16, 2.0, amplitude=2.0, baseline=5.0)])

    # The peak is removed and the baseline is left standing. Subtracting the magnitude instead
    # would have taken the baseline with it, at every feature in the frame.
    assert subtracted[0]["amplitude"] == pytest.approx(2.0)
    assert residual[16, 16] == pytest.approx(5.0)


def test_peeling_leaves_the_input_untouched():
    values = np.random.default_rng(1).normal(size=(24, 24))
    original = values.copy()
    peel(values, [_Feature(12, 12, 2.0, 1.0)])
    assert np.array_equal(values, original)


def test_peeling_lowers_a_planted_peak():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(41, 41)) * 0.01
    values += fitted_gaussian((41, 41), (20.0, 20.0), amplitude=4.0, width=3.0)
    residual, subtracted = peel(values, [_Feature(20.0, 20.0, 3.0, 4.0)])

    assert len(subtracted) == 1
    assert abs(residual[20, 20]) < abs(values[20, 20])


# ---------------- the contamination split, which is the whole safeguard


def test_a_spurious_feature_beside_a_peeled_one_is_an_artefact():
    subtracted = [{"row": 20.0, "col": 20.0, "width": 3.0, "amplitude": 1.0}]
    split = split_contamination([(20.0, 24.0)], subtracted)

    assert split["residual_artefacts"] == 1
    assert split["elsewhere"] == 0


def test_a_spurious_feature_far_from_every_peeled_one_is_not():
    subtracted = [{"row": 20.0, "col": 20.0, "width": 3.0, "amplitude": 1.0}]
    split = split_contamination([(80.0, 80.0)], subtracted)

    assert split["residual_artefacts"] == 0
    assert split["elsewhere"] == 1
    assert split["artefact_fraction_of_spurious"] == 0.0


def test_the_artefact_radius_scales_with_the_feature_it_came_from():
    """A wide feature leaves its lobes further out; a fixed radius in cells would miss them."""
    narrow = [{"row": 0.0, "col": 0.0, "width": 1.0, "amplitude": 1.0}]
    wide = [{"row": 0.0, "col": 0.0, "width": 10.0, "amplitude": 1.0}]

    assert split_contamination([(0.0, 15.0)], narrow)["residual_artefacts"] == 0
    assert split_contamination([(0.0, 15.0)], wide)["residual_artefacts"] == 1


def test_the_declared_artefact_radius_is_the_one_the_measurement_used():
    """Declared before the counts existed and not moved afterwards."""
    assert ARTEFACT_RADIUS_IN_FITTED_WIDTHS == 2.0


def test_no_spurious_features_reports_no_fraction_rather_than_zero():
    """Zero of zero is not zero per cent, and reporting it as such would invent a clean run."""
    split = split_contamination([], [{"row": 0.0, "col": 0.0, "width": 2.0, "amplitude": 1.0}])
    assert split["artefact_fraction_of_spurious"] is None


def test_with_nothing_peeled_every_spurious_feature_is_ordinary_contamination():
    split = split_contamination([(5.0, 5.0), (9.0, 9.0)], [])
    assert split["residual_artefacts"] == 0
    assert split["elsewhere"] == 2


# ---------------- the gap, and refusing to invent one


def test_the_gap_closed_is_the_fraction_of_the_distance_to_the_oracle():
    assert gap_closed(20.0, 12.0, 4.0) == pytest.approx(0.5)
    assert gap_closed(20.0, 4.0, 4.0) == pytest.approx(1.0)
    assert gap_closed(20.0, 20.0, 4.0) == pytest.approx(0.0)


def test_a_round_zero_already_at_the_oracle_has_no_gap_to_close():
    """A fraction of a non-existent gap is not zero, and zero would invent a failure."""
    assert gap_closed(4.0, 4.0, 4.0) is None
    assert gap_closed(3.0, 3.0, 4.0) is None


def test_overshooting_the_oracle_is_reported_rather_than_clipped():
    """A peeled cut below the oracle is a real possibility and must not be silently capped."""
    assert gap_closed(20.0, 2.0, 4.0) > 1.0


# ---------------- the mechanism description


def test_the_nearest_planting_supplies_the_stretch_of_an_artefact():
    plantings = [PlantedFeature(row=10.0, col=10.0, sigma=3.0, ratio=16.0,
                                stretch=2.5, angle=0.0),
                 PlantedFeature(row=80.0, col=80.0, sigma=3.0, ratio=16.0,
                                stretch=1.0, angle=0.0)]
    assert nearest_planting_stretch((12.0, 12.0), plantings) == 2.5
    assert nearest_planting_stretch((79.0, 81.0), plantings) == 1.0


def test_no_plantings_gives_no_stretch_rather_than_a_default():
    assert nearest_planting_stretch((1.0, 1.0), []) is None
