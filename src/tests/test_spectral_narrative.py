"""T4D.3: the sentences, and the four things they are not permitted to say.

1.  **The acceptance sentence, against the vortex T4D.2 already tracked.** Every number a
    sentence contains is checked against the track it came from, so the prose cannot drift
    from the measurement it describes.
2.  **Scale growth is in the population.** The roadmap's "dominant scale doubled" is not a
    statement any one of these tracks supports; what is measured is level 4 weakening while
    level 5 strengthens and is excited nine frames later, and that is what gets written.
3.  **North has to be earned.** A cartesian grid gets axis-relative wording; a latlon grid
    gets a bearing whose sign comes from `dy` and whose east component is shortened by the
    cosine of the latitude.
4.  **R7 is enforced on the rendered half.** The one list of words the programme will not say
    is imported, not restated, and a causal word reaching a sentence is refused wherever it
    came from -- including from a caller's own dataset name.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from src.analysis_engine.spectral_narrative import (
    ENTITLEMENT, NARRATIVE_SCHEMA, assert_no_causal_language, narrate_population,
    narrate_track, narrate_tracking, render_text,
)
from src.analysis_engine.spectral_tracking import COL, ROW, track_spectral_features
from src.benchmarks.core import get_benchmark
from src.core.claim_ladder import OUTSIDE_THE_LADDER
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.feature import FeatureLocation, FeatureSet, Quantity, SpectralFeature
from src.core.tracking import MotionBounds, Track
from src.physical_core.grid import GridError, GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import decompose_sequence

WAVELET = "haar"
LEVELS = 5
TRACKED_SCALES = [4, 5]
THRESHOLD_SIGMA = 4.0
BOUNDS = MotionBounds(max_doublings=0.5)

AXES = (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
        AxisSpec(COL, "space", units="cells", periodic=False, ordinal=1))
REPRESENTATION = "wavelet[swt,wavelet=haar,levels=5]"

_CACHE = {}


def _narrated():
    """The T4D.2 acceptance pass, narrated once and reused."""
    if "narrated" not in _CACHE:
        bench = get_benchmark("advected_vortex_sequence")
        built = bench.make()
        sequence = FieldSequence(built.fields, np.arange(len(built.fields), dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET})
        band = field.select(scales=TRACKED_SCALES)
        result = track_spectral_features(
            band, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", bounds=BOUNDS,
            threshold_sigma=THRESHOLD_SIGMA)
        _CACHE["narrated"] = (band, result, narrate_tracking(result, field=band))
    return _CACHE["narrated"]


def _feature(time, row, col, magnitude, *, scale=4, orientation="LH", dataset="manual"):
    return SpectralFeature(
        domain="synthetic", dataset=dataset, variable="amplitude",
        magnitude=Quantity(float(magnitude), None),
        location=FeatureLocation({ROW: float(row), COL: float(col)}, AXES),
        time=float(time), time_units="frames", representation=REPRESENTATION,
        spatial_scale=Quantity(float(2 ** (scale - 1)), "parent-grid px"),
        provenance={"scale_label": scale, "orientation_label": orientation})


def _straight_track(*, d_row, d_col, magnitudes=(1.0, 1.0), dataset="manual", scale=4):
    """A hand-built two-point track, so a bearing can be checked against arithmetic."""
    features = [_feature(index, index * d_row, index * d_col, magnitude,
                         scale=scale, dataset=dataset)
                for index, magnitude in enumerate(magnitudes)]
    return Track(0, FeatureSet(features))


def _latlon(dy, *, lat0=0.0, shape=(64, 64)):
    return GridSpec("latlon", shape, dy=dy, dx=abs(dy), lat0=lat0, lon0=0.0)


def _number_after(sentence, pattern):
    match = re.search(pattern, sentence)
    assert match is not None, "sentence did not carry %r: %s" % (pattern, sentence)
    return float(match.group(1))


# =============================================================================== section 1
# The sentence, and the numbers underneath it.


def test_every_track_gets_one_sentence_carrying_the_numbers_it_was_built_from():
    _, result, narrative = _narrated()
    assert narrative["schema"] == NARRATIVE_SCHEMA
    assert len(narrative["tracks"]) == len(result) == 4
    for item, track in zip(narrative["tracks"], result):
        assert item["track_id"] == track.track_id
        assert item["observations"] == len(track)
        assert item["displacement_cells"].keys() == {ROW, COL}
        assert item["speed"]["value"] == pytest.approx(track.speed().value)
        assert item["magnitude"]["first"] == pytest.approx(track[0].magnitude.value)
        assert item["magnitude"]["last"] == pytest.approx(track[-1].magnitude.value)


def test_the_speed_in_the_prose_is_the_speed_the_track_actually_has():
    _, result, narrative = _narrated()
    checked = 0
    for item, track in zip(narrative["tracks"], result):
        spoken = _number_after(item["sentence"], r"at a mean ([0-9.]+) cells")
        assert spoken == pytest.approx(track.speed().value, abs=0.005)
        checked += 1
    assert checked == 4, "every track's prose is checked against its own track"


def test_the_subject_of_every_sentence_is_the_maximum_and_not_the_structure():
    _, _, narrative = _narrated()
    for item in narrative["tracks"]:
        assert "the coefficient maximum was followed" in item["sentence"]
    text = render_text(narrative).lower()
    for forbidden in ("the vortex", "the storm", "the object travelled"):
        assert forbidden not in text


def test_the_frame_count_is_of_frames_searched_not_only_frames_found():
    _, result, narrative = _narrated()
    assert narrative["frames_searched"] == len(result.frame_times) == 24
    shortest = min(narrative["tracks"], key=lambda item: item["observations"])
    assert "%d of 24 searched frames" % shortest["observations"] in shortest["sentence"]


# =============================================================================== section 2
# Scale growth lives in the population, and the prose says so.


def test_no_track_of_a_growing_vortex_claims_its_own_scale_doubled():
    _, result, narrative = _narrated()
    for item, track in zip(narrative["tracks"], result):
        assert item["scale"]["held_level"] is not None
        assert item["scale"]["scale_velocity_doublings_per_time"] == 0.0
        assert item["scale"]["doubling_time"] is None
        assert "doubl" not in item["sentence"].lower()
        assert "no change of scale" in item["sentence"]


def test_the_growth_that_did_happen_is_visible_as_one_band_yielding_to_the_next():
    """The honest replacement for "dominant scale doubled", and it is a measurement.

    The vortex widens from sigma 5 to sigma 13.5 cells. Level 4's maxima therefore weaken
    over the record while level 5's strengthen, and level 5 is not excited at all until the
    structure has grown into it. Neither fact is available from any single track.
    """
    _, _, narrative = _narrated()
    by_level = {}
    for item in narrative["tracks"]:
        by_level.setdefault(str(item["band"]["scale_label"]), []).append(item)
    assert sorted(by_level) == ["4", "5"]
    for item in by_level["4"]:
        assert item["magnitude"]["magnitude_change_pct"] < 0.0
    for item in by_level["5"]:
        assert item["magnitude"]["magnitude_change_pct"] > 0.0
    bands = {record["band"]: record for record in narrative["population"]["bands"]}
    assert min(r["first_seen"] for b, r in bands.items() if b.startswith("L4")) == 0.0
    assert min(r["first_seen"] for b, r in bands.items() if b.startswith("L5")) == 9.0


def test_the_band_ordering_is_offered_as_a_candidate_precursor_and_nothing_above_it():
    _, _, narrative = _narrated()
    sentence = narrative["population"]["sentence"]
    assert "candidate precursor relationship" in sentence
    assert "was not tested against a null" in sentence
    assert "no merge is claimed" in sentence
    for forbidden in ("led to", "triggered", "produced by", "gave rise"):
        assert forbidden not in sentence.lower()


def test_one_band_supports_no_ordering():
    result = _narrated()[1]
    single = [t for t in result
              if (str(t[0].provenance["scale_label"]),
                  t[0].provenance["orientation_label"]) == ("4", "LH")]
    assert len(single) == 1
    population = narrate_population(single)
    assert "no ordering and no candidate precursor relationship" in population["sentence"]


# =============================================================================== section 3
# North has to be earned.


def test_a_grid_that_does_not_know_where_north_is_gets_axis_relative_wording():
    band, _, narrative = _narrated()
    assert band.grid.kind == "cartesian"
    assert narrative["compass"]["available"] is False
    assert "declares no orientation" in narrative["compass"]["reason"]
    for item in narrative["tracks"]:
        assert item["direction"]["point"] is None
        assert "increasing row" in item["sentence"] or "decreasing row" in item["sentence"]
        for forbidden in ("north", "south", "east", "west"):
            assert forbidden not in item["sentence"].lower()


def test_the_sign_that_makes_a_row_northward_is_read_from_the_grid_not_assumed():
    """One displacement, two grids, opposite compass points -- which is the ERA5 case."""
    track = _straight_track(d_row=1.0, d_col=0.0)
    south_to_north = narrate_track(track, grid=_latlon(+0.25))["direction"]
    north_to_south = narrate_track(track, grid=_latlon(-0.25))["direction"]
    assert south_to_north["point"] == "north"
    assert north_to_south["point"] == "south"
    assert south_to_north["bearing_deg"] == pytest.approx(0.0)
    assert north_to_south["bearing_deg"] == pytest.approx(180.0)


def test_a_degree_of_longitude_is_shortened_by_the_cosine_before_the_bearing_is_taken():
    """At 60 degrees north a degree of longitude is half a degree of latitude.

    Equal displacements in row and column are therefore *not* north-east. Taking the bearing
    from degrees would say 45; taking it in metres says 26.6, and the difference is a rotation
    rather than a scaling, so it changes the compass word rather than only a number.
    """
    track = _straight_track(d_row=1.0, d_col=1.0)
    direction = narrate_track(track, grid=_latlon(+0.25, lat0=60.0))["direction"]
    expected = math.degrees(math.atan2(math.cos(math.radians(60.0 + 0.125)), 1.0))
    assert direction["bearing_deg"] == pytest.approx(expected, abs=0.05)
    assert abs(direction["bearing_deg"] - 45.0) > 15.0
    assert direction["point"] == "north-east"
    assert "cosine at the track's mean latitude" in direction["basis"]


def test_a_track_that_returned_to_where_it_started_is_given_no_bearing():
    """A net displacement of zero has no direction, and "north" is not a rounding of it."""
    track = _straight_track(d_row=0.0, d_col=0.0)
    direction = narrate_track(track, grid=_latlon(+0.25))["direction"]
    assert direction["point"] is None
    assert "net displacement is exactly zero" in direction["refused"]


def test_the_bearing_needs_no_guard_for_a_missing_lat0_and_this_is_why():
    """The unreachable-branch test: a latlon grid without a latitude origin cannot exist.

    `_bearing` reads `lat0` without checking it, which would be a defect if a caller could
    supply a latlon grid that lacked one. `GridSpec` refuses to build that grid, so the check
    would be a branch no input reaches -- and an unreachable refusal reads, in a receipt, like
    a case that was considered and covered.
    """
    with pytest.raises(GridError):
        GridSpec("latlon", (64, 64), dy=0.25, dx=0.25, lat0=None, lon0=None)


# =============================================================================== section 4
# Magnitude is not energy.


def test_energy_is_the_square_and_the_two_figures_are_reported_under_their_own_names():
    """The roadmap's own example sentence, checked: 43% in magnitude is 104% in energy."""
    track = _straight_track(d_row=1.0, d_col=1.0, magnitudes=(1.0, 1.43))
    narrative = narrate_track(track)
    assert narrative["magnitude"]["magnitude_change_pct"] == pytest.approx(43.0)
    assert narrative["magnitude"]["energy_change_pct"] == pytest.approx(104.49)
    assert "a change of 43.0%" in narrative["sentence"]
    assert "104.5% in coefficient energy" in narrative["sentence"]
    assert "energy being the square" in narrative["sentence"]


def test_a_change_from_a_first_magnitude_of_zero_is_refused_not_rendered_infinite():
    track = _straight_track(d_row=1.0, d_col=0.0, magnitudes=(0.0, 2.0))
    narrative = narrate_track(track)
    assert narrative["magnitude"]["magnitude_change_pct"] is None
    assert "undefined, not infinite" in narrative["magnitude"]["refused"]
    assert "%" not in narrative["sentence"]


# =============================================================================== section 5
# R7, on the rendered half.


def test_the_guard_uses_the_programmes_one_list_of_words_it_will_not_say():
    for word in OUTSIDE_THE_LADDER:
        with pytest.raises(InvalidParameterError) as excinfo:
            assert_no_causal_language(["the level 4 band %s this" % word])
        assert "R7" in str(excinfo.value)


def test_a_causal_word_in_a_callers_own_dataset_name_is_refused_before_a_reader_sees_it():
    track = _straight_track(d_row=1.0, d_col=0.0, dataset="co2_causes_warming")
    with pytest.raises(InvalidParameterError) as excinfo:
        narrate_track(track)
    message = str(excinfo.value)
    assert "causes" in message and "R7" in message


def test_the_guard_states_its_limit_rather_than_over_reaching_into_substrings():
    """A word run together with another is not caught, and a substring scan is worse.

    `assert_no_causal_language` flattens punctuation so an identifier is screened, but it
    still matches whole words. Falling back to a substring test would refuse every sentence
    containing "becausewhat", "causeway" or "mechanisms of the filter bank" -- and a guard
    that fires on innocent text is a guard that gets turned off.
    """
    assert_no_causal_language(["the co2causeswarming record"])
    assert_no_causal_language(["the causeway band"])
    with pytest.raises(InvalidParameterError):
        assert_no_causal_language(["the co2_causes_warming record"])


def test_the_entitlement_may_name_the_boundary_that_the_sentences_may_not_cross():
    """The split is deliberate, and this is the test that says so.

    The entitlement contains two words the guard refuses, and it contains them in order to
    refuse them. Scanning it would refuse the sentence whose whole purpose is to hold the
    line, which is why the curated half and the rendered half are scanned differently.
    """
    assert "mechanism" in ENTITLEMENT and "cause" in ENTITLEMENT
    with pytest.raises(InvalidParameterError):
        assert_no_causal_language([ENTITLEMENT])
    narrative = narrate_track(_straight_track(d_row=1.0, d_col=0.0))
    assert narrative["entitlement"] == ENTITLEMENT
    assert ENTITLEMENT in render_text(_narrated()[2])


def test_the_entitlement_appears_once_however_many_tracks_there_are():
    narrative = _narrated()[2]
    text = render_text(narrative)
    assert len(narrative["tracks"]) == 4
    assert text.count("provides no framework that could license one (R7)") == 1


# =============================================================================== section 6
# What a narrative refuses to be.


def test_a_single_sighting_supports_no_direction_speed_or_growth():
    track = Track(7, FeatureSet([_feature(3.0, 10.0, 10.0, 1.0)]))
    narrative = narrate_track(track)
    assert "seen once" in narrative["sentence"]
    assert "no direction, no speed and no growth" in narrative["sentence"]
    assert "speed" not in narrative
    assert narrative["structural_signature"].endswith("n=1")


def test_a_search_that_found_nothing_is_not_narrated_as_an_empty_list_of_sentences():
    with pytest.raises(InvalidParameterError) as excinfo:
        narrate_tracking(None)
    assert "say so at the call site" in str(excinfo.value)


def test_the_structural_signature_names_no_variable_dataset_or_units():
    _, _, narrative = _narrated()
    for item in narrative["tracks"]:
        signature = item["structural_signature"]
        for forbidden in ("advected_vortex_sequence", "amplitude", "synthetic", "cells"):
            assert forbidden not in signature
        assert signature.startswith("L")


def test_the_narrative_names_the_filter_and_not_only_the_family():
    _, _, narrative = _narrated()
    assert narrative["representation"] == "wavelet[swt,wavelet=haar,levels=5]"


def test_the_empty_frames_of_the_pass_travel_with_the_narrative():
    _, result, narrative = _narrated()
    assert narrative["empty_frames"] == list(result.empty_frames)
    assert narrative["track_count"] == len(result)
