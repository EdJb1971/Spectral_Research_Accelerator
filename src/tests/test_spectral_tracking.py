"""T4D.2: linking coefficient maxima frame to frame, and what the link is allowed to mean.

1.  **The acceptance criterion, against an answer recorded before the tracker.**
    `advected_vortex_sequence` carries the trajectory, the velocity and the doubling time. A
    detail coefficient peaks at a structure's *flank*, so what the tracks recover is stated
    exactly: the transverse coordinate to under a pixel, and the velocity as advection plus
    the structure's own growth along the axis its band high-passes.
2.  **D88, the registration the parent-grid claim needed.** Two levels of one decomposition
    are displaced relative to each other by the analysis delay. Pooling them without
    subtracting it associates on a separation that is mostly filter, so a bank that cannot
    register its own levels is refused rather than tracked.
3.  **The translation is lossy in stated ways.** A separable band label is not an angle; a
    threshold crossing is not a significance; a plateau is located less sharply than a peak.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis_engine.spectral_feature import detect_features
from src.analysis_engine.spectral_tracking import (
    describe_tracking, spectral_feature_set, track_spectral_features,
)
from src.benchmarks.core import get_benchmark
from src.core.errors import InvalidParameterError
from src.core.tracking import MotionBounds
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import decompose_sequence

#: A linear-phase filter, because D88 says only such a bank registers its own levels exactly.
WAVELET = "haar"
LEVELS = 5
#: Levels 4 and 5 are the ones a vortex of sigma 5 to 13 cells excites; finer levels see the
#: benchmark's declared noise and coarser ones do not fit the crop's valid interior.
TRACKED_SCALES = [4, 5]
THRESHOLD_SIGMA = 4.0
#: One octave per frame is impossible for a structure whose width doubles every sixteen, so
#: this gate separates the levels without being tuned to the answer.
BOUNDS = MotionBounds(max_doublings=0.5)

_CACHE = {}


def _vortex():
    """The benchmark sequence, decomposed once and reused."""
    if "vortex" not in _CACHE:
        bench = get_benchmark("advected_vortex_sequence")
        built = bench.make()
        sequence = FieldSequence(built.fields,
                                 np.arange(len(built.fields), dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET})
        _CACHE["vortex"] = (field, bench.truth())
    return _CACHE["vortex"]


def _tracked():
    if "tracked" not in _CACHE:
        field, truth = _vortex()
        band = field.select(scales=TRACKED_SCALES)
        result = track_spectral_features(
            band, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", bounds=BOUNDS,
            threshold_sigma=THRESHOLD_SIGMA)
        _CACHE["tracked"] = (band, truth, result)
    return _CACHE["tracked"]


def _label(track):
    first = track[0]
    return (first.provenance["scale_label"], first.provenance["orientation_label"])


def _longest_per_band(result):
    best = {}
    for item in result:
        key = _label(item)
        if key not in best or len(item) > len(best[key]):
            best[key] = item
    return best


def _blob_sequence(n=128, centre=(64.0, 64.0), sigma=6.0, frames=3):
    grid_y, grid_x = np.mgrid[0:n, 0:n]
    values = np.exp(-(((grid_y - centre[0]) ** 2 + (grid_x - centre[1]) ** 2)
                      / (2.0 * sigma ** 2)))
    fields = [PhysicalField(__import__("torch").as_tensor(values.copy()),
                            grid=GridSpec.pixel((n, n)))
              for _ in range(frames)]
    return FieldSequence(fields, np.arange(frames, dtype=float))


# ============================================ 1. the acceptance criterion

def test_the_structure_is_one_unbroken_track_per_excited_band():
    """D1's concern, stated as this representation actually presents it.

    A shift-variant transform makes a tracker report births and deaths as a feature drifts
    across grid parity. Here the two level-4 bands run the whole sequence without a break,
    which is the property the undecimated bank exists to give. There is more than one track
    because the bank is redundant and a detail band peaks on a flank, not because anything
    was lost -- and the module says so rather than reporting a count of one.
    """
    band, truth, result = _tracked()
    assert result is not None
    n_frames = band.n_times
    per_band = _longest_per_band(result)
    for orientation in ("LH", "HL"):
        assert len(per_band[("4", orientation)]) == n_frames, (
            "the level-4 %s track breaks, which is the aliasing D1 describes" % orientation)
    # Every track is one of the bands' own; none is a fragment of another band's.
    assert set(per_band) <= {("4", "LH"), ("4", "HL"), ("4", "HH"),
                             ("5", "LH"), ("5", "HL"), ("5", "HH")}


def test_the_transverse_position_is_within_a_pixel_of_the_recorded_trajectory():
    """The roadmap's `< 1 px`, on the coordinate for which it is a meaningful claim.

    An `LH` band high-passes columns, so it localises the row and displaces the column by
    about the analysing width. The row is therefore a statement about the vortex; the column
    is a statement about its flank. Both are checked, and only the first is claimed.
    """
    band, truth, result = _tracked()
    transverse = {"LH": "row", "HL": "col"}
    checked = 0
    for track in _longest_per_band(result).values():
        scale, orientation = _label(track)
        axis = transverse[orientation]
        index = 0 if axis == "row" else 1
        errors = [abs(item.location.coords[axis]
                      - truth["positions_rowcol"][int(item.time)][index])
                  for item in track]
        assert max(errors) < 1.0, (scale, orientation, max(errors))
        checked += 1
    assert checked == 4, "every band's track is checked, not a chosen one"


def test_the_velocity_is_advection_plus_the_structures_own_growth():
    """The trajectory and the scale evolution, both recovered, neither supplied.

    The tracker was given the coefficients, the axes, an alpha and one octave-per-frame gate.
    It was not given the velocity or the doubling time. A flank sits about one width out
    along the axis its band high-passes, so as the vortex grows the flank moves at the
    advection *plus* the growth rate on that axis alone -- a prediction with no free
    parameter, since both numbers were recorded before this module existed.
    """
    band, truth, result = _tracked()
    known_row, known_col = truth["velocity_cells_per_step"]
    sigma = truth["sigma_cells"]
    longitudinal = {"LH": "col", "HL": "row"}
    checked = 0
    for track in _longest_per_band(result).values():
        scale, orientation = _label(track)
        axis = longitudinal[orientation]
        first, last = int(track[0].time), int(track[-1].time)
        growth = (sigma[last] - sigma[first]) / float(last - first)
        predicted = {"row": known_row, "col": known_col}
        predicted[axis] += growth
        measured = track.velocity()
        error = math.hypot(measured["row"].value - predicted["row"],
                           measured["col"].value - predicted["col"])
        # A flank sits about, not exactly, one sigma out: the filter's own width contributes
        # too. The tolerance is that approximation's slack, not a fitted threshold.
        assert error < 0.15, (scale, orientation, error, predicted, measured)
        checked += 1
    assert checked == 4, "every band's track is checked, not a chosen one"


def test_the_coarse_band_is_excited_later_and_never_earlier():
    """"Correct scale-doubling detection", as a redundant bank can actually express it.

    The vortex starts too narrow to excite level 5 and grows into it. So level-5 tracks are
    born after level-4 tracks and never before -- the direction of the scale migration, read
    off the population of tracks, because in a redundant bank the coarse level lights up
    *beside* the fine one rather than instead of it.
    """
    band, truth, result = _tracked()
    births = {}
    for item in result:
        scale, _ = _label(item)
        births.setdefault(scale, []).append(item.birth_time)
    assert min(births["4"]) == 0.0
    assert "5" in births, "the vortex doubles in width; the coarse band has to light up"
    assert min(births["5"]) > min(births["4"])


def test_no_track_migrates_between_levels_and_the_module_says_why():
    """The scale gate refuses an octave step, so a level-5 detection cannot steal a level-4
    track. Without it the two bands' detections sit a few pixels apart and the association
    would swap them, reporting a scale change that is really a band confusion."""
    band, truth, result = _tracked()
    for item in result:
        scales = {record.provenance["scale_label"] for record in item}
        assert len(scales) == 1, "a track spans two levels; the octave gate did not hold"
    body = describe_tracking(result, field=band)
    assert "does not claim a merge" in body["claim_boundary"]
    assert body["bands"]["scales"] == ["4", "5"]


# ============================================ 2. D88, registration

def test_levels_of_a_linear_phase_bank_agree_on_where_a_blob_is():
    """The delay subtracted, one structure lands in one place at every level.

    Undecimated bands have the parent grid's *shape*; that was never the same as being
    registered to it. Before D88 was subtracted, level 4 and level 5 put the same blob 8
    pixels apart, and pooling them for association compared filter against filter.
    """
    sequence = _blob_sequence()
    field = decompose_sequence(sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET})
    detection = detect_features(field.select(scales=[3, 4]), interior=True, align=True)
    assert detection["alignment"]["cross_scale_comparable"] is True
    rows = {}
    for record in detection["features"]:
        if record["orientation"] != "LH":
            continue
        rows.setdefault(record["scale"], []).append(record["y"])
    # LH localises the row; a blob at row 64 should be at row 64 in both levels.
    for scale, values in rows.items():
        assert min(abs(value - 64.0) for value in values) < 1.0, (scale, values)

    raw = detect_features(field.select(scales=[3, 4]), interior=True, align=False)
    assert raw["alignment"]["cross_scale_comparable"] is False
    assert "NOT comparable" in raw["coordinate_frame"]
    shifts = detection["alignment"]["shift_parent_px"]
    assert shifts["4"] > shifts["3"] > 0.0


def test_a_bank_that_cannot_register_its_levels_is_refused_not_tracked():
    """db2 is orthogonal and therefore neither symmetric nor antisymmetric.

    Its delay is not one number, so subtracting the common shift leaves a residual that grows
    with the level's dilation. Pooling such levels would associate detections on a separation
    that is mostly transform, and the error is not even constant, so it could not be quoted as
    an uncertainty either. The refusal names the filter and the way out.
    """
    sequence = _blob_sequence()
    field = decompose_sequence(sequence, "swt", {"levels": 4, "wavelet": "db2"})
    with pytest.raises(InvalidParameterError) as excinfo:
        spectral_feature_set(field.select(scales=[3, 4]), domain="synthetic",
                             dataset="blob", variable="amplitude", time_units="frames")
    message = str(excinfo.value)
    assert "D88" in message and "linear-phase" in message

    # One level at a time is fine: within a band there is nothing to be out of register with.
    single = spectral_feature_set(field.select(scales=[3]), domain="synthetic",
                                  dataset="blob", variable="amplitude", time_units="frames")
    assert single is not None and len(single) > 0


def test_a_decimated_family_cannot_declare_an_alignment_at_all():
    """DTCWT reaches the parent grid by nearest-neighbour replication.

    An aligned index is then a label for the block one native coefficient covers, and no
    single shift turns a label into a location. The field says so, and pooling is refused for
    that reason rather than for the delay.
    """
    sequence = _blob_sequence()
    field = decompose_sequence(sequence, "dtcwt", {"levels": 3})
    record = field.parent_alignment(2)
    assert record["shift_px"] is None and record["exact"] is False
    assert "nearest-neighbour replication" in record["basis"]
    with pytest.raises(InvalidParameterError):
        spectral_feature_set(field.select(scales=[2, 3]), domain="synthetic",
                             dataset="blob", variable="amplitude", time_units="frames")


# ============================================ 3. what the translation refuses to invent

def test_a_separable_band_label_does_not_become_an_angle():
    """`HH` responds to both diagonal signs. An orientation gate on it would refuse nothing.

    So the features carry no orientation and the tracker's own check declines the gate. That
    refusal is the point: a gate that appears in a receipt while testing nothing makes a run
    with no declared physics look like one whose physics was satisfied.
    """
    band, truth, result = _tracked()
    features = spectral_feature_set(band, domain="synthetic", dataset="vortex",
                                    variable="amplitude", time_units="frames",
                                    threshold_sigma=THRESHOLD_SIGMA)
    assert all(item.orientation is None for item in features)
    with pytest.raises(InvalidParameterError) as excinfo:
        track_spectral_features(band, domain="synthetic", dataset="vortex",
                                variable="amplitude", time_units="frames",
                                threshold_sigma=THRESHOLD_SIGMA,
                                bounds=MotionBounds(max_doublings=0.5,
                                                    max_turn_degrees=30.0))
    assert "orientation" in str(excinfo.value)


def test_a_complex_family_passes_its_declared_angle_through():
    sequence = _blob_sequence()
    field = decompose_sequence(sequence, "dtcwt", {"levels": 3}).select(scales=[2])
    features = spectral_feature_set(field, domain="synthetic", dataset="blob",
                                    variable="amplitude", time_units="frames")
    assert features is not None
    angles = {item.orientation.degrees for item in features}
    assert angles and all(item.orientation.convention == "axis_180" for item in features)
    assert angles <= set(float(o) % 180.0 for o in field.orientations)


def test_a_threshold_crossing_is_not_reported_as_a_significance():
    band, truth, result = _tracked()
    features = spectral_feature_set(band, domain="synthetic", dataset="vortex",
                                    variable="amplitude", time_units="frames",
                                    threshold_sigma=THRESHOLD_SIGMA)
    assert all(item.significance is None for item in features)
    assert "not established" in features[0].provenance["significance"]
    assert features[0].provenance["threshold"] > 0.0


def test_the_representation_names_the_filter_and_not_only_the_family():
    """R8's audit asks which representation manufactured a feature. `swt` is not an answer:
    a Haar bank and a db2 bank of that family make different edges."""
    band, truth, result = _tracked()
    features = spectral_feature_set(band, domain="synthetic", dataset="vortex",
                                    variable="amplitude", time_units="frames",
                                    threshold_sigma=THRESHOLD_SIGMA)
    assert "wavelet=haar" in features.representation
    assert "swt" in features.representation


def test_a_plateau_is_carried_as_positional_uncertainty():
    band, truth, result = _tracked()
    features = spectral_feature_set(band, domain="synthetic", dataset="vortex",
                                    variable="amplitude", time_units="frames",
                                    threshold_sigma=THRESHOLD_SIGMA)
    for item in features:
        expected = 0.5 * math.sqrt(item.provenance["plateau_pixels"])
        assert item.location.uncertainty["row"] == pytest.approx(expected)


# ============================================ 4. the declarations the bridge will not guess

def test_the_domain_and_the_dataset_are_required_not_defaulted():
    band, truth, result = _tracked()
    for kwargs in ({"domain": "", "dataset": "vortex"},
                   {"domain": "synthetic", "dataset": "  "}):
        with pytest.raises(InvalidParameterError):
            spectral_feature_set(band, variable="amplitude", **kwargs)


def test_a_field_that_never_recorded_its_variable_has_to_be_told():
    band, truth, result = _tracked()
    with pytest.raises(InvalidParameterError) as excinfo:
        spectral_feature_set(band, domain="synthetic", dataset="vortex")
    assert "variable" in str(excinfo.value)


def test_a_grid_that_closes_in_longitude_cannot_be_declared_flat():
    """D59's lesson, at this boundary. A seam crossing is one track on a torus and two on a
    plane, so the topology decides the count of objects and may not be assumed away."""
    grid = GridSpec(kind="latlon", shape=(64, 128), dy=-1.40625, dx=2.8125,
                    lat0=45.0, lon0=0.0)
    torch = __import__("torch")
    values = np.zeros((64, 128))
    values[32, 64] = 1.0
    fields = [PhysicalField(torch.as_tensor(values.copy()), grid=grid) for _ in range(2)]
    sequence = FieldSequence(fields, np.arange(2, dtype=float))
    field = decompose_sequence(sequence, "swt", {"levels": 2, "wavelet": WAVELET})
    with pytest.raises(InvalidParameterError) as excinfo:
        spectral_feature_set(field.select(scales=[1]), domain="synthetic", dataset="global",
                             variable="amplitude", time_units="frames", periodic=False)
    assert "360" in str(excinfo.value)


def test_two_sets_of_detection_settings_are_refused():
    """One of them did not run, and a receipt describing the pass that did not happen is
    worse than no receipt."""
    band, truth, result = _tracked()
    detection = detect_features(band, threshold_sigma=THRESHOLD_SIGMA)
    with pytest.raises(InvalidParameterError):
        spectral_feature_set(band, detection, domain="synthetic", dataset="vortex",
                             variable="amplitude", time_units="frames",
                             threshold_sigma=9.0)


def test_a_search_that_found_nothing_is_not_a_tracking_result():
    band, truth, result = _tracked()
    assert track_spectral_features(
        band, domain="synthetic", dataset="vortex", variable="amplitude",
        time_units="frames", threshold_values=[1e9, 1e9]) is None


def test_the_clock_handed_to_the_tracker_is_every_frame_that_was_searched():
    """A frame that found nothing ends a track rather than being bridged across."""
    band, truth, result = _tracked()
    assert result.times == tuple(float(t) for t in band.times)
    assert len(result.times) == band.n_times
