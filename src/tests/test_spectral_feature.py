"""T4D.1 feature detection: what a located maximum is, and what it refuses to be."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis_engine.spectral_feature import (
    DEFAULT_MAX_FEATURES_PER_BAND,
    DETECTION_SCHEMA,
    detect_features,
)
from src.core.errors import InvalidParameterError
from src.physical_core.grid import GridSpec
from src.transform_engine.coefficient_field import CoefficientField

SIZE = 64


def _blob(height: int, width: int, y: float, x: float, amplitude: float = 10.0,
          sigma: float = 1.6) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    return amplitude * np.exp(
        -(((grid_y - y) ** 2 + (grid_x - x) ** 2) / (2.0 * sigma ** 2)))


def _field(frames: np.ndarray, *, family: str = "swt", scales=(1,),
           orientations=("LH",)) -> CoefficientField:
    """`frames` is `(time, scale, orientation, y, x)` already."""
    data = torch.as_tensor(np.asarray(frames, dtype=np.float64))
    return CoefficientField(
        data, wavelet_family=family, scales=list(scales),
        orientations=list(orientations),
        times=np.arange(data.shape[0], dtype=np.float64),
        grid=GridSpec.pixel((int(data.shape[-2]), int(data.shape[-1]))))


def _single(frame: np.ndarray, **kwargs) -> CoefficientField:
    return _field(frame[None, None, None, :, :], **kwargs)


# ------------------------------------------------------------------ the acceptance test

@pytest.mark.parametrize("true_y,true_x", [(32.0, 32.0), (30.4, 41.7), (25.5, 25.5)])
def test_a_planted_maximum_is_recovered_to_better_than_one_pixel(true_y, true_x):
    """T4D.1 acceptance: a known blob is located, sub-pixel, in the parent frame.

    Sub-pixel matters because T4D.2 associates detections frame to frame; a position quantised
    to the pixel would make a slow drift look like a stationary feature that jumps.
    """
    field = _single(_blob(SIZE, SIZE, true_y, true_x))
    result = detect_features(field, interior=False)
    assert result["schema"] == DETECTION_SCHEMA
    assert result["n_features"] == 1
    found = result["features"][0]
    assert abs(found["y"] - true_y) < 1.0 and abs(found["x"] - true_x) < 1.0
    # The integer peak alone would be up to half a pixel out; the refinement has to beat that
    # on the deliberately worst case, a blob centred exactly between two samples.
    assert found["strength"] > found["threshold"]


def test_the_refinement_is_better_than_the_integer_peak_it_starts_from():
    true_y, true_x = 30.5, 41.5
    field = _single(_blob(SIZE, SIZE, true_y, true_x))
    found = detect_features(field, interior=False)["features"][0]
    integer_error = max(abs(round(found["y"]) - true_y), abs(round(found["x"]) - true_x))
    refined_error = max(abs(found["y"] - true_y), abs(found["x"] - true_x))
    assert refined_error < integer_error


# ------------------------------------------------------------------ R13, the margin

def test_a_maximum_inside_the_contaminated_margin_is_not_reported():
    """R13. A tracker fed edge detections reports births and deaths that are artefacts.

    The same blob is planted twice: once inside the filter-support margin and once outside it.
    Only the second is a statement about the field rather than about where the crop was cut.
    """
    near_edge = _single(_blob(SIZE, SIZE, 1.0, 1.0), scales=(3,))
    assert detect_features(near_edge, interior=True)["n_features"] == 0
    # Not because the detector cannot see it -- with the mask off, it is found.
    assert detect_features(near_edge, interior=False)["n_features"] == 1

    interior = _single(_blob(SIZE, SIZE, 32.0, 32.0), scales=(3,))
    result = detect_features(interior, interior=True)
    assert result["n_features"] == 1
    assert result["interior"]["applied"] is True
    assert result["interior"]["margin_parent_px"]["3"] > 0


def test_the_margin_is_recorded_even_when_it_is_not_applied():
    field = _single(_blob(SIZE, SIZE, 32.0, 32.0))
    result = detect_features(field, interior=False)
    assert result["interior"]["applied"] is False
    assert result["interior"]["margin_parent_px"] == {"1": 0}
    assert "R13" in result["interior"]["basis"]


# ------------------------------------------------------------------ the threshold discipline

def test_the_threshold_is_fitted_once_over_the_record_not_per_frame():
    """A per-frame threshold would make a quiet frame and a loud one report alike.

    Two frames, one with a blob at ten times the amplitude of the other. Fitted over the record,
    the quiet frame's maximum falls below the shared threshold; that is the point. A threshold
    refitted per frame would find one feature in each and the count would say nothing.
    """
    loud = _blob(SIZE, SIZE, 20.0, 20.0, amplitude=100.0)
    quiet = _blob(SIZE, SIZE, 40.0, 40.0, amplitude=1.0)
    field = _field(np.stack([loud, quiet])[:, None, None, :, :])
    result = detect_features(field, interior=False)
    times = sorted(record["time_index"] for record in result["features"])
    assert times == [0]
    assert result["threshold_source"] == "fitted from this complete coefficient record"


def test_frozen_thresholds_are_used_verbatim_and_recorded_as_inherited():
    """The leakage-safe path: held-out frames inherit the yardstick, they do not set it."""
    field = _single(_blob(SIZE, SIZE, 32.0, 32.0))
    fitted = detect_features(field, interior=False)
    inherited = detect_features(
        field, interior=False, threshold_values=[fitted["thresholds"]["1"]])
    assert inherited["threshold_source"] == "supplied frozen per-scale thresholds"
    assert inherited["thresholds"] == fitted["thresholds"]
    assert inherited["n_features"] == fitted["n_features"]

    # A threshold above the peak yields nothing, which is how we know it was actually applied.
    silenced = detect_features(field, interior=False, threshold_values=[1e6])
    assert silenced["n_features"] == 0


def test_a_threshold_that_would_accept_everything_is_refused():
    field = _single(_blob(SIZE, SIZE, 32.0, 32.0))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(InvalidParameterError):
            detect_features(field, threshold_sigma=bad)
    with pytest.raises(InvalidParameterError):
        detect_features(field, threshold_values=[1.0, 2.0])       # wrong scale count
    with pytest.raises(InvalidParameterError):
        detect_features(field, threshold_values=[-1.0])           # negative
    with pytest.raises(InvalidParameterError):
        detect_features(field, max_features_per_band=0)


# ------------------------------------------------------------------ honesty of the output

def test_truncation_is_recorded_rather_than_silent():
    """A silently truncated detection set would make a count of features meaningless."""
    rng = np.random.default_rng(4406)
    # A lattice of separated blobs, so there are genuinely many strict maxima.
    frame = np.zeros((SIZE, SIZE))
    for y in range(4, SIZE - 4, 4):
        for x in range(4, SIZE - 4, 4):
            frame += _blob(SIZE, SIZE, y + rng.uniform(-0.3, 0.3),
                           x + rng.uniform(-0.3, 0.3), amplitude=10.0, sigma=0.8)
    field = _single(frame)
    capped = detect_features(field, interior=False, max_features_per_band=5)
    assert capped["n_features"] == 5
    assert capped["truncated_bands"], "truncation must be reported, never silent"
    record = capped["truncated_bands"][0]
    assert record["kept"] == 5 and record["found"] > 5
    # The strongest are the ones kept.
    uncapped = detect_features(field, interior=False)
    strongest = sorted((f["strength"] for f in uncapped["features"]), reverse=True)[:5]
    assert sorted((f["strength"] for f in capped["features"]), reverse=True) == strongest


def test_a_real_family_reports_no_phase_rather_than_inventing_zero():
    field = _single(_blob(SIZE, SIZE, 32.0, 32.0))
    result = detect_features(field, interior=False)
    assert result["features"][0]["phase"] is None
    assert result["provenance"]["complex_family"] is False


def test_a_complex_family_reports_the_phase_it_actually_has():
    magnitude = _blob(SIZE, SIZE, 32.0, 32.0)
    values = magnitude * np.exp(1j * 0.75)
    data = torch.as_tensor(values[None, None, None, :, :])
    field = CoefficientField(
        data, wavelet_family="dtcwt", scales=[1], orientations=["15"],
        times=np.array([0.0]), grid=GridSpec.pixel((SIZE, SIZE)))
    result = detect_features(field, interior=False)
    assert result["provenance"]["complex_family"] is True
    assert result["features"][0]["phase"] == pytest.approx(0.75, abs=1e-9)
    assert result["features"][0]["strength"] == pytest.approx(magnitude.max(), rel=1e-9)


def test_a_plateau_is_one_detection_at_its_centre_carrying_its_own_imprecision():
    """A flat top is one feature, not sixteen, and it says how flat it was.

    Reporting one detection per tied pixel would let a tracker manufacture sixteen associations
    out of one feature. Reporting none would lose it entirely -- and would also lose every
    feature sitting exactly between two samples, which is where a smoothly advecting feature
    spends half its time. So: one detection, positioned at the group's centroid, carrying the
    group's size so a consumer can see it is located less sharply than a single peak.
    """
    frame = np.zeros((SIZE, SIZE))
    frame[30:34, 30:34] = 50.0
    result = detect_features(_single(frame), interior=False)
    assert result["n_features"] == 1
    found = result["features"][0]
    assert found["plateau_pixels"] == 16
    assert found["y"] == pytest.approx(31.5) and found["x"] == pytest.approx(31.5)


def test_a_feature_exactly_between_two_samples_is_found_at_the_midpoint():
    """The case a strict-maximum detector loses, and the reason the rule is regional.

    A blob centred on a half-pixel produces two exactly equal samples. Strictness would reject
    both and the feature would blink out -- twice per pixel of travel, for anything advecting.
    """
    field = _single(_blob(SIZE, SIZE, 32.0, 31.5))
    result = detect_features(field, interior=False)
    assert result["n_features"] == 1
    found = result["features"][0]
    assert found["plateau_pixels"] == 2
    assert found["y"] == pytest.approx(32.0, abs=1e-9)
    assert found["x"] == pytest.approx(31.5, abs=1e-9)


def test_a_plateau_on_a_slope_is_not_a_maximum():
    """Equal to its neighbours in one direction, lower in another: not a maximum at all."""
    frame = np.zeros((SIZE, SIZE))
    frame[20:40, 20:40] = 10.0
    frame[30:34, 30:34] = 50.0
    frame[31, 34] = 90.0          # something strictly higher adjoins the flat top
    result = detect_features(_single(frame), interior=False)
    positions = {(record["y"], record["x"]) for record in result["features"]}
    assert (31.5, 31.5) not in positions


def test_the_claim_boundary_travels_with_the_detection():
    field = _single(_blob(SIZE, SIZE, 32.0, 32.0))
    result = detect_features(field, interior=False)
    assert "not a physical object" in result["claim_boundary"]
    assert result["max_features_per_band"] == DEFAULT_MAX_FEATURES_PER_BAND
    assert result["coordinate_frame"].startswith("parent-grid pixels")


def test_every_band_is_visited_and_labelled():
    """Two scales and two orientations: a feature says which band produced it."""
    frames = np.zeros((1, 2, 2, SIZE, SIZE))
    frames[0, 0, 0] = _blob(SIZE, SIZE, 20.0, 20.0)
    frames[0, 1, 1] = _blob(SIZE, SIZE, 40.0, 40.0)
    field = _field(frames, scales=(1, 2), orientations=("LH", "HL"))
    result = detect_features(field, interior=False)
    labelled = {(record["scale"], record["orientation"]) for record in result["features"]}
    assert labelled == {("1", "LH"), ("2", "HL")}
    assert set(result["thresholds"]) == {"1", "2"}
