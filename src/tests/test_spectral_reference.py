"""T4F.6: the physical gate, and the five ways a gate like this stops being one.

1.  **A gate that cannot fail is not a gate, and this one is made to fail on purpose.** The
    same seven patterns and the same ranked report return PASS against a catalogue whose
    envelopes fit them, FAIL against one whose envelopes do not, and INVALID against one so wide
    that nothing could have fallen outside it. All three verdicts are exercised here on real
    coefficients rather than on constructed inputs, because a FAIL path that only a stub can
    reach is not evidence that the real one works.

2.  **The record's own geometry sets the ceiling on discrimination, and it is measured.** On the
    benchmark's cartesian grid a cell is 31 km exactly, so a level-4 footprint is 496 km and a
    level-5 one is 992 km and the two are cleanly separable. On the programme's own ERA5 crop
    the same 16 cells are anywhere from 222 to 445 km, because the zonal metric runs 13.90 to
    26.12 km across 40 degrees of latitude. Both numbers are pinned, because the second is what
    stops the first being read as the accuracy of the method.

3.  **Unassessable is not unrecognised.** An entry needing a 500 hPa field is unassessable on a
    record of 850 hPa temperature for every pattern and forever, and the shipped draft catalogue
    contains exactly one such entry so that this is exercised rather than described.

4.  **A lead in hours needs the record's clock, not somebody else's.** The cadence is measured
    from the record's own timestamps, and a rule counted on an event series whose frames are not
    the record's frames has its lead refused by name.

5.  **The catalogue is a declaration and this code will not sign it.** The digest covers the
    entries and not the signature, so freezing an unchanged catalogue leaves it traceable to the
    draft that was reviewed; and `physical_gate` refuses a draft outright.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, ConstellationPattern, PatternCatalogue, SignatureMetric, SignaturePoint,
    calibrate_signature_tolerance,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_events import EventSeries, ObservationGrid, PatternEvent
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_precursors import LagFamily, precursor_report
from src.analysis_engine.spectral_projection import (
    evidence_index, geography_of, project_pattern,
)
from src.analysis_engine.spectral_reference import (
    CATALOGUE_DRAFT, CATALOGUE_FROZEN, CRITERION_INSIDE, CRITERION_OUTSIDE,
    CRITERION_UNASSESSABLE, GATE_CLAIM_BOUNDARY, GATE_FAIL, GATE_INVALID, GATE_PASS,
    GATE_SCHEMA, LABEL_RECOGNISED, LABEL_UNASSESSABLE, LABEL_UNRECOGNISED,
    MATCH_CONSISTENT, MATCH_EXCLUDED, MATCH_UNASSESSABLE, MINIMUM_ASSESSED_CRITERIA,
    RANK_KEYS, RECOGNITION_NOTE, SCALE_FILTER_SUPPORT, SCALE_OCTAVE_LABEL,
    SOUTHERN_OCEAN_OBSERVABLE, Envelope, GateDeclaration, Phenomenon, ReferenceCatalogue,
    cross_reference, describe_gate, discrimination, draft_southern_ocean_reference,
    label_pattern, lead_for, match_phenomenon, measure_pattern, physical_bridge, physical_gate,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.analysis_engine.spectral_tracking import track_spectral_features
from src.benchmarks.core import get_benchmark
from src.core.errors import InvalidParameterError
from src.core.level_axis import PRESSURE_HPA
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import CoefficientField, decompose_sequence

WAVELET = "haar"
LEVELS = 5
SCALES = [4, 5]
THRESHOLD_SIGMA = 4.0
METRIC = SignatureMetric(AttributeWeights(geometry=1.0, bearings=1.0, strengths=1.0,
                                          scales=1.0))

#: The benchmark's grid is 31 km per cell and isotropic, so a footprint's size in kilometres is
#: one number rather than a range, and the two tracked levels are 496 km and 992 km apart in
#: scale. Written here so the tests below read a number rather than recompute the code.
CELL_KM = 31.0
LEVEL4_KM = 16 * CELL_KM
LEVEL5_KM = 32 * CELL_KM

#: The T4C.5k ERA5 record's own crop: 850 hPa temperature, 0.25 degrees, six-hourly, latitudes
#: -60 to -20 and longitudes 140 to 180.
ERA5_SHAPE = (161, 161)
ERA5_LAT0, ERA5_DLAT = -20.0, -0.25
ERA5_LON0, ERA5_DLON = 140.0, 0.25

CADENCE_HOURS = 6
_CACHE: dict = {}


# ---------------------------------------------------------------------------------------------
# the record: one planted vortex on a dated clock, decomposed, tracked and clustered once
# ---------------------------------------------------------------------------------------------


def _record():
    """The T4D.2 acceptance sequence on a six-hourly calendar, carried as far as T4E.3.

    Dated, unlike T4F.5's copy of it, because a cadence is the one thing this module needs that
    a bare frame index cannot give: the calendar is what makes a lag in frames a lead in hours.
    """
    if "record" not in _CACHE:
        bench = get_benchmark("advected_vortex_sequence")
        built = bench.make()
        stamps = (np.datetime64("2021-06-01T00", "ns")
                  + np.arange(len(built.fields)) * np.timedelta64(CADENCE_HOURS, "h"))
        sequence = FieldSequence(built.fields, stamps)
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET}).select(scales=SCALES)
        tracked = track_spectral_features(
            field, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", threshold_sigma=THRESHOLD_SIGMA)
        constellations = extract_constellations(tracked, cardinalities=(2,))
        _CACHE["record"] = (sequence, field, constellations)
    return _CACHE["record"]


def _catalogue(constellations):
    """One pattern per band pair, assembled rather than agglomerated (as T4F.5's suite does)."""
    signatures = sign_constellations(constellations, scale_invariant=False).signatures
    by_key = {tuple(item.key()): item for item in constellations}
    groups: dict = {}
    for signature in signatures:
        groups.setdefault(by_key[tuple(signature.key)].bands, []).append(signature)
    replicates = sorted(groups.values(), key=len)[-1][:3]
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC)
    patterns = []
    for index, bands in enumerate(sorted(groups)):
        members = tuple(sorted(groups[bands], key=lambda item: (item.time, item.key)))
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=index + 1, members=members, centroid=points[0],
            tolerance_radius=tolerance.value,
            observed_radius=max(METRIC.distance(point, points[0]) for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=METRIC, tolerance=tolerance,
                            n_signatures=len(signatures))


@pytest.fixture(scope="module")
def record():
    return _record()


@pytest.fixture(scope="module")
def geography(record):
    return geography_of(record[1], time_units="frames", values=record[0])


@pytest.fixture(scope="module")
def bridge(geography):
    return physical_bridge(geography, variable="amplitude")


@pytest.fixture(scope="module")
def projections(record, geography):
    catalogue = _catalogue(record[2])
    index = evidence_index(catalogue, record[2], geography)
    return tuple(project_pattern(pattern.pattern_id, index) for pattern in catalogue.patterns)


def _regrid(field, grid, *, level=None, level_axis=None, times=None):
    """The same coefficients under another geometry or clock, so each one's answer can be read.

    `times` is here because `geography_of` insists a supplied record's frames are the frames the
    decomposition ran on -- rightly -- and the clock is exactly what several tests below need to
    vary. Re-decomposing to change a timestamp would measure the transform, not the bridge.
    """
    return CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations,
        times=(field.times if times is None else times), grid=grid,
        config=dict(field.config), level=level, level_axis=level_axis)


def _era5_record():
    """A two-frame decomposition on the T4C.5k crop's own grid, for the geometry alone.

    161x161 at 0.25 degrees from -20 to -60 north and 140 to 180 east, which is the record
    T4C.5k acquired. The values are noise and no result is read from them: what is read is what
    that grid does to a length, which depends on the crop and not on the field.
    """
    if "era5" not in _CACHE:
        grid = GridSpec.latlon(ERA5_SHAPE, lat0=ERA5_LAT0, dlat=ERA5_DLAT,
                               lon0=ERA5_LON0, dlon=ERA5_DLON, variable_units="K")
        generator = torch.Generator().manual_seed(20260906)
        fields = [PhysicalField(torch.randn(ERA5_SHAPE, generator=generator,
                                            dtype=torch.float64), grid=grid)
                  for _ in range(2)]
        stamps = (np.datetime64("2021-06-01T00", "ns")
                  + np.arange(2) * np.timedelta64(CADENCE_HOURS, "h"))
        sequence = FieldSequence(fields, stamps)
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET}).select(scales=SCALES)
        field = CoefficientField(
            field.data, wavelet_family=field.wavelet_family, scales=field.scales,
            orientations=field.orientations, times=field.times, grid=grid,
            config=dict(field.config), level=850.0, level_axis=PRESSURE_HPA)
        _CACHE["era5"] = physical_bridge(
            geography_of(field, time_units="frames", values=sequence), variable="t")
    return _CACHE["era5"]


# ---------------------------------------------------------------------------------------------
# the declared catalogue used by most of the tests below
# ---------------------------------------------------------------------------------------------

OBSERVABLE = "amplitude@no-declared-level"

MESOSCALE = Phenomenon(
    name="mesoscale-vortex-pair",
    citation="advected_vortex_sequence, the benchmark's own known_answer",
    observables=(OBSERVABLE,), horizontal_scale=Envelope(300.0, 700.0, "km"),
    lead=Envelope(3.0, 24.0, "hours"), cardinality=2,
    separation=Envelope(200.0, 700.0, "km"))
PLANETARY = Phenomenon(
    name="planetary-wave-train",
    citation="a declared decoy: nothing this size was planted in this benchmark",
    observables=(OBSERVABLE,), horizontal_scale=Envelope(3000.0, 8000.0, "km"),
    lead=Envelope(72.0, 240.0, "hours"), cardinality=2)
ELSEWHERE = Phenomenon(
    name="upper-level-precursor", citation="needs a level this record has not got",
    observables=("z@500 pressure_hpa",), horizontal_scale=Envelope(800.0, 3000.0, "km"),
    lead=Envelope(12.0, 72.0, "hours"), cardinality=2)
#: Three entries that exist only to be *nearly* the mesoscale one. Overlap is a conjunction --
#: two entries collide only when their scales meet *and* their leads meet -- and without a pair
#: that agrees on exactly one of the two, dropping either half of that test changes nothing.
SAME_SCALE_OTHER_LEAD = Phenomenon(
    name="same-scale-other-lead", citation="c", observables=(OBSERVABLE,),
    horizontal_scale=Envelope(400.0, 900.0, "km"), lead=Envelope(100.0, 200.0, "hours"),
    cardinality=2)
SAME_LEAD_OTHER_SCALE = Phenomenon(
    name="same-lead-other-scale", citation="c", observables=(OBSERVABLE,),
    horizontal_scale=Envelope(3000.0, 5000.0, "km"), lead=Envelope(6.0, 30.0, "hours"),
    cardinality=2)
#: A triple, so that the cardinality criterion has something to exclude on a record whose every
#: constellation has two members.
TRIPLE = Phenomenon(
    name="three-member-configuration", citation="c", observables=(OBSERVABLE,),
    horizontal_scale=Envelope(1.0, 100000.0, "km"), lead=Envelope(0.0, 10000.0, "hours"),
    cardinality=3)
#: The mesoscale envelope with its separation criterion removed, so that the patterns separation
#: alone excludes can be identified by difference.
NO_SEPARATION_LIMIT = Phenomenon(
    name="mesoscale-without-a-separation-limit", citation="c", observables=(OBSERVABLE,),
    horizontal_scale=MESOSCALE.horizontal_scale, lead=MESOSCALE.lead, cardinality=2)

ANYTHING = Phenomenon(
    name="anything-at-all", citation="a deliberately useless envelope",
    observables=(OBSERVABLE,), horizontal_scale=Envelope(1.0, 100000.0, "km"),
    lead=Envelope(0.0, 10000.0, "hours"))


def _declared(*phenomena, status=CATALOGUE_FROZEN):
    catalogue = ReferenceCatalogue(phenomena=tuple(phenomena), declared_by="the T4F.6 suite",
                                   declared_on="2026-09-06")
    return catalogue.freeze(signed_by="the suite") if status == CATALOGUE_FROZEN else catalogue


@pytest.fixture(scope="module")
def catalogue():
    return _declared(MESOSCALE, PLANETARY)


# ---------------------------------------------------------------------------------------------
# the event series and the ranked report the gate adjudicates
# ---------------------------------------------------------------------------------------------

FRAMES = 120
B_TIMES = [8, 15, 31, 36, 41, 50, 55, 64, 68, 77, 91, 99, 108, 119]
A_TIMES = [time - 2 for time in B_TIMES[:7]] + [66, 60, 118, 119]
DECOY_TIMES = list(range(0, FRAMES, 7))
WINDOW = TransitionWindow(1.0, 3.0, "frames")
SURROGATES = 113
SEED = 20260906

#: The two patterns the planted series names. `1` is the level-4 pair whose 496 km footprint
#: sits inside the mesoscale envelope; `7` is the level-5 pair whose 992 km footprint does not.
ANTECEDENT, CONSEQUENT, DECOY = 1, 2, 7


def _series(frames=None):
    grid = ObservationGrid(
        frames=tuple(float(item) for item in (frames if frames is not None
                                              else range(FRAMES))),
        time_units="frames", cadence=1.0)
    searched = set(grid.frames)
    events = []
    for pattern_id, times in ((ANTECEDENT, A_TIMES), (CONSEQUENT, B_TIMES),
                              (DECOY, DECOY_TIMES)):
        for time in times:
            if float(time) not in searched:
                continue
            events.append(PatternEvent(
                pattern_id=pattern_id, time=float(time), identity="p%d@%d" % (pattern_id, time),
                cardinality=2, mode="scale_specific"))
    events.sort(key=lambda item: (item.time, item.pattern_id))
    return EventSeries(events=tuple(events), grid=grid, mode="scale_specific",
                       units_confirmed_by_members=len(events), units_uncarried_by_members=0)


@pytest.fixture(scope="module")
def ranked():
    series = _series()
    report = precursor_report(series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES,
                              seed=SEED, pairs=[(ANTECEDENT, CONSEQUENT), (DECOY, CONSEQUENT)])
    return series, report


def _declaration(catalogue, *, rank_by="q_value", top_n=2):
    return GateDeclaration(
        catalogue=catalogue, rank_by=rank_by, top_n=top_n,
        period_justification=("synthetic mechanism check on the advected-vortex benchmark; no "
                              "documented atmospheric event is claimed"),
        declared_by="the T4F.6 suite", declared_on="2026-09-06")


# ---------------------------------------------------------------------------------------------
# section 1: what a declaration must contain before it can gate anything
# ---------------------------------------------------------------------------------------------


def test_an_envelope_with_a_reversed_range_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Envelope(700.0, 300.0, "km")
    assert "reversed range is empty" in str(excinfo.value)


def test_an_envelope_with_an_infinite_end_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Envelope(1.0, float("inf"), "km")
    assert "excludes nothing" in str(excinfo.value)


def test_an_envelope_without_a_unit_is_refused():
    with pytest.raises(InvalidParameterError):
        Envelope(1.0, 2.0, "  ")


def test_an_envelope_overlaps_inclusively_at_both_ends():
    envelope = Envelope(300.0, 700.0, "km")
    assert envelope.overlaps(700.0, 900.0)
    assert envelope.overlaps(100.0, 300.0)
    assert not envelope.overlaps(700.001, 900.0)


def test_the_fraction_inside_tells_a_bare_overlap_from_a_containment():
    envelope = Envelope(300.0, 700.0, "km")
    assert envelope.fraction_of(400.0, 600.0) == pytest.approx(1.0)
    assert envelope.fraction_of(500.0, 900.0) == pytest.approx(0.5)
    assert envelope.fraction_of(496.0, 496.0) == pytest.approx(1.0)
    assert envelope.fraction_of(992.0, 992.0) == pytest.approx(0.0)


def test_an_entry_without_a_citation_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Phenomenon(name="x", citation="   ", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "km"), lead=Envelope(1.0, 2.0, "hours"))
    assert "recognise whatever it was written to recognise" in str(excinfo.value)


def test_an_entry_naming_no_observable_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Phenomenon(name="x", citation="c", observables=(),
                   horizontal_scale=Envelope(1.0, 2.0, "km"), lead=Envelope(1.0, 2.0, "hours"))
    assert "checkable against every record" in str(excinfo.value)


def test_an_entry_whose_scale_is_not_in_kilometres_is_refused():
    with pytest.raises(InvalidParameterError):
        Phenomenon(name="x", citation="c", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "cells"),
                   lead=Envelope(1.0, 2.0, "hours"))


def test_an_entry_whose_lead_is_not_in_hours_is_refused():
    with pytest.raises(InvalidParameterError):
        Phenomenon(name="x", citation="c", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "km"),
                   lead=Envelope(1.0, 2.0, "frames"))


def test_a_negative_declared_lead_is_refused_because_the_window_cannot_produce_one():
    with pytest.raises(InvalidParameterError) as excinfo:
        Phenomenon(name="x", citation="c", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "km"),
                   lead=Envelope(-6.0, 2.0, "hours"))
    assert "consequent preceding its antecedent" in str(excinfo.value)


def test_an_unknown_scale_convention_is_refused_by_name():
    with pytest.raises(InvalidParameterError) as excinfo:
        Phenomenon(name="x", citation="c", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "km"), lead=Envelope(1.0, 2.0, "hours"),
                   scale_convention="whatever")
    assert "wrong by that factor" in str(excinfo.value)


def test_a_catalogue_with_two_entries_of_one_name_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        ReferenceCatalogue(phenomena=(MESOSCALE, MESOSCALE), declared_by="a", declared_on="b")
    assert "mesoscale-vortex-pair" in str(excinfo.value)


def test_an_empty_catalogue_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        ReferenceCatalogue(phenomena=(), declared_by="a", declared_on="b")
    assert "labels every pattern unassessable" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 2: the catalogue as a declaration -- digest, signature, overlaps
# ---------------------------------------------------------------------------------------------


def test_editing_an_envelope_changes_the_catalogues_digest():
    before = ReferenceCatalogue(phenomena=(MESOSCALE,), declared_by="a", declared_on="b")
    widened = Phenomenon(
        name=MESOSCALE.name, citation=MESOSCALE.citation, observables=MESOSCALE.observables,
        horizontal_scale=Envelope(300.0, 1400.0, "km"), lead=MESOSCALE.lead,
        cardinality=MESOSCALE.cardinality, separation=MESOSCALE.separation)
    after = ReferenceCatalogue(phenomena=(widened,), declared_by="a", declared_on="b")
    assert before.digest != after.digest


def test_signing_an_unchanged_catalogue_leaves_its_digest_alone():
    """Otherwise a receipt could not be traced back to the draft that was reviewed."""
    draft = ReferenceCatalogue(phenomena=(MESOSCALE,), declared_by="a", declared_on="b")
    frozen = draft.freeze(signed_by="a maintainer")
    assert frozen.status == CATALOGUE_FROZEN and draft.status == CATALOGUE_DRAFT
    assert frozen.signed_by == "a maintainer"
    assert frozen.digest == draft.digest


def test_freezing_without_a_signatory_is_refused():
    draft = ReferenceCatalogue(phenomena=(MESOSCALE,), declared_by="a", declared_on="b")
    with pytest.raises(InvalidParameterError) as excinfo:
        draft.freeze(signed_by="  ")
    assert "draft that has stopped saying it is one" in str(excinfo.value)


def test_two_entries_whose_envelopes_meet_are_published_as_a_pair():
    catalogue = _declared(MESOSCALE, PLANETARY)
    assert catalogue.overlapping_pairs() == ()
    both = ReferenceCatalogue(
        phenomena=(MESOSCALE, Phenomenon(
            name="another", citation="c", observables=(OBSERVABLE,),
            horizontal_scale=Envelope(400.0, 900.0, "km"), lead=Envelope(6.0, 30.0, "hours"))),
        declared_by="a", declared_on="b")
    pairs = both.overlapping_pairs()
    assert len(pairs) == 1
    assert pairs[0]["entries"] == ["mesoscale-vortex-pair", "another"]
    assert "evidence for neither" in pairs[0]["note"]


def test_the_shipped_southern_ocean_reference_is_a_draft_nobody_has_signed():
    catalogue = draft_southern_ocean_reference()
    assert catalogue.status == CATALOGUE_DRAFT
    assert catalogue.signed_by is None
    assert "none of them is a value quoted from a paper" in catalogue.note
    assert all(item.citation.strip() for item in catalogue)


def test_the_shipped_reference_contains_an_entry_this_record_can_never_check():
    """The single-level record has no upper level, and that is a permanent unassessable."""
    catalogue = draft_southern_ocean_reference()
    needing_another_level = [item for item in catalogue
                             if SOUTHERN_OCEAN_OBSERVABLE not in item.observables]
    assert [item.name for item in needing_another_level] == ["upper-level-pv-precursor"]


# ---------------------------------------------------------------------------------------------
# section 3: the bridge -- what this record can turn a cell and a frame into
# ---------------------------------------------------------------------------------------------


def test_an_isotropic_cartesian_grid_gives_one_cell_size_and_not_a_range(bridge):
    assert bridge.has_length
    assert bridge.cell_km_low == pytest.approx(CELL_KM)
    assert bridge.cell_km_high == pytest.approx(CELL_KM)
    assert bridge.anisotropy["is_isotropic"] is True


def test_the_cadence_is_measured_from_the_records_own_timestamps(bridge):
    assert bridge.has_clock
    assert bridge.cadence_hours == pytest.approx(float(CADENCE_HOURS))


def test_the_observable_is_spelled_from_what_the_record_declares(bridge):
    assert bridge.observable == OBSERVABLE


def test_a_level_set_without_its_axis_is_spelled_as_a_number_not_as_hectopascals(record):
    """TG1.5, arriving here: an entry asking for 850 hPa must not match a bare 850."""
    field = _regrid(record[1], record[1].grid, level=850.0, level_axis=None)
    bare = physical_bridge(geography_of(field, time_units="frames", values=record[0]),
                           variable="t")
    assert bare.observable == "t@level-850-with-no-declared-axis"
    declared = physical_bridge(
        geography_of(_regrid(record[1], record[1].grid, level=850.0, level_axis=PRESSURE_HPA),
                     time_units="frames", values=record[0]), variable="t")
    assert declared.observable == SOUTHERN_OCEAN_OBSERVABLE


def test_a_pixel_grid_refuses_kilometres_rather_than_assuming_a_spacing(record):
    field = _regrid(record[1], GridSpec.pixel(record[1].grid.shape))
    bare = physical_bridge(geography_of(field, time_units="frames", values=record[0]),
                           variable="amplitude")
    assert not bare.has_length
    assert "recognise a phenomenon by a number nobody measured" in bare.cell_refusal
    assert "cell_km" not in bare.as_record()


def _undated(record):
    """The same coefficients on a clock of bare frame numbers."""
    numeric = FieldSequence(record[0].fields, np.arange(len(record[0].fields), dtype=float))
    field = _regrid(record[1], record[1].grid, times=numeric.times_seconds)
    return physical_bridge(geography_of(field, time_units="frames", values=numeric),
                           variable="amplitude")


def test_a_record_with_no_calendar_refuses_hours_rather_than_naming_the_frame_one(record):
    bare = _undated(record)
    assert not bare.has_clock
    assert "carries no calendar" in bare.cadence_refusal
    assert "cadence_hours" not in bare.as_record()


def test_a_record_whose_frames_are_not_evenly_spaced_has_no_cadence(record):
    stamps = (np.datetime64("2021-06-01T00", "ns")
              + np.arange(len(record[0].fields)) * np.timedelta64(CADENCE_HOURS, "h"))
    stamps[3] = stamps[3] + np.timedelta64(1, "h")
    gappy = FieldSequence(record[0].fields, stamps)
    field = _regrid(record[1], record[1].grid, times=gappy.times_seconds)
    bare = physical_bridge(geography_of(field, time_units="frames", values=gappy),
                           variable="amplitude")
    assert not bare.has_clock
    assert "has no one cadence" in bare.cadence_refusal


def test_the_real_era5_crop_turns_one_cell_into_a_range_of_kilometres():
    """The measurement that limits this gate: 16 cells is 222 to 445 km on that crop."""
    era5 = _era5_record()
    assert era5.observable == SOUTHERN_OCEAN_OBSERVABLE
    assert era5.cell_km_low == pytest.approx(13.899, abs=1e-3)
    assert era5.cell_km_high == pytest.approx(27.799, abs=1e-3)
    assert era5.cell_km_high / era5.cell_km_low == pytest.approx(2.0, abs=0.01)
    span = era5.length("horizontal_scale", 16.0, 16.0)
    assert span.low == pytest.approx(222.4, abs=0.1)
    assert span.high == pytest.approx(444.8, abs=0.1)
    assert era5.anisotropy["dx_variation_pct"] == pytest.approx(61.08, abs=0.01)
    assert any("zonal metric varies" in text for text in era5.anisotropy["warnings"])


def test_the_bridge_refuses_a_record_that_cannot_say_what_field_it_holds(geography):
    with pytest.raises(InvalidParameterError) as excinfo:
        physical_bridge(geography, variable="   ")
    assert "matches everything" in str(excinfo.value)


def test_the_bridge_refuses_anything_that_is_not_a_bound_geography():
    with pytest.raises(InvalidParameterError):
        physical_bridge(object(), variable="t")


# ---------------------------------------------------------------------------------------------
# section 4: measuring a pattern off its own footprints
# ---------------------------------------------------------------------------------------------


def test_every_pattern_is_measured_as_the_footprints_it_was_drawn_with(projections, bridge):
    """Level 4 is 496 km, level 5 is 992 km, and a cross-level pattern spans both."""
    seen = {}
    for projection in projections:
        measured = measure_pattern(projection, bridge)
        span = measured[SCALE_FILTER_SUPPORT]
        seen[projection.pattern_id] = (round(span.low), round(span.high))
    assert set(seen.values()) == {(LEVEL4_KM, LEVEL4_KM), (LEVEL4_KM, LEVEL5_KM),
                                  (LEVEL5_KM, LEVEL5_KM)}
    assert seen[ANTECEDENT] == (LEVEL4_KM, LEVEL4_KM)
    assert seen[DECOY] == (LEVEL5_KM, LEVEL5_KM)


def test_the_octave_label_is_carried_beside_the_support_and_is_half_of_it(projections, bridge):
    """T4F.5's two numbers about different things, and why an entry declares which it means."""
    measured = measure_pattern(projections[0], bridge)
    support = measured[SCALE_FILTER_SUPPORT]
    octave = measured[SCALE_OCTAVE_LABEL]
    assert support.low == pytest.approx(2.0 * octave.low)
    assert support.high == pytest.approx(2.0 * octave.high)


def test_an_entry_declared_against_the_octave_label_reads_the_other_number(projections, bridge):
    measured = measure_pattern(projections[0], bridge)
    octave_entry = Phenomenon(
        name="o", citation="c", observables=(OBSERVABLE,),
        horizontal_scale=Envelope(200.0, 300.0, "km"), lead=Envelope(1.0, 2.0, "hours"),
        cardinality=2, scale_convention=SCALE_OCTAVE_LABEL)
    match = match_phenomenon(measured, octave_entry, bridge)
    scale = [item for item in match.criteria if item.name == "horizontal_scale"][0]
    assert scale.status == CRITERION_INSIDE          # 248 km, the octave label
    support_entry = Phenomenon(
        name="s", citation="c", observables=(OBSERVABLE,),
        horizontal_scale=Envelope(200.0, 300.0, "km"), lead=Envelope(1.0, 2.0, "hours"),
        cardinality=2, scale_convention=SCALE_FILTER_SUPPORT)
    other = match_phenomenon(measured, support_entry, bridge)
    assert [item for item in other.criteria
            if item.name == "horizontal_scale"][0].status == CRITERION_OUTSIDE


def test_the_measured_separation_is_read_off_the_footprint_centres(projections, bridge):
    measured = measure_pattern(projections[ANTECEDENT - 1], bridge)
    separation = measured["separation"]
    assert separation.measured
    assert separation.units == "km"
    assert 0.0 < separation.low <= separation.high
    assert "through this grid's own metric" in separation.basis


def test_every_pattern_of_this_record_has_two_members(projections, bridge):
    for projection in projections:
        assert measure_pattern(projection, bridge)["cardinality"] == [2]


def test_measuring_something_that_is_not_a_projection_is_refused(bridge):
    with pytest.raises(InvalidParameterError) as excinfo:
        measure_pattern(object(), bridge)
    assert "only extent this transform actually justifies" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 5: the three labels, and the boundary between two of them
# ---------------------------------------------------------------------------------------------


def test_the_planted_pair_is_recognised_and_the_level_five_pair_is_not(projections, bridge,
                                                                      catalogue):
    labels = {item.pattern_id: item for item in cross_reference(catalogue, projections, bridge)}
    assert labels[ANTECEDENT].label == LABEL_RECOGNISED
    assert labels[ANTECEDENT].consistent_with == ("mesoscale-vortex-pair",)
    assert labels[ANTECEDENT].excluded_by == ("planetary-wave-train",)
    # Consistent with one entry and excluded by another: this label *did* separate something,
    # so it must not carry the caution that says it separated nothing.
    assert labels[ANTECEDENT].non_discriminating is False
    assert "caution" not in labels[ANTECEDENT].as_record()
    statuses = {item.phenomenon: item.status for item in labels[ANTECEDENT].matches}
    assert statuses == {"mesoscale-vortex-pair": MATCH_CONSISTENT,
                        "planetary-wave-train": MATCH_EXCLUDED}
    assert labels[DECOY].label == LABEL_UNRECOGNISED
    assert labels[DECOY].consistent_with == ()
    assert set(labels[DECOY].excluded_by) == {"mesoscale-vortex-pair", "planetary-wave-train"}


def test_an_entry_needing_a_field_this_record_lacks_is_unassessable_not_unrecognised(
        projections, bridge):
    labels = cross_reference(_declared(ELSEWHERE), projections, bridge)
    assert {item.label for item in labels} == {LABEL_UNASSESSABLE}
    for item in labels:
        assert item.unassessable_for == ("upper-level-precursor",)
        assert item.excluded_by == ()
        match = item.matches[0]
        assert match.status == MATCH_UNASSESSABLE
        assert "would record the crop's contents as a fact about the atmosphere" in \
            match.criteria[0].note


def test_the_observable_check_runs_before_every_other_criterion(projections, bridge):
    """A scale verdict on the wrong field would be a verdict about something else entirely."""
    match = match_phenomenon(measure_pattern(projections[0], bridge), ELSEWHERE, bridge)
    assert [item.name for item in match.criteria] == ["observables"]


def test_consistency_on_one_criterion_alone_is_not_recognition(projections, bridge):
    """`ANYTHING` matches on scale and on nothing else, so it recognises nothing."""
    measured = measure_pattern(projections[0], bridge)
    match = match_phenomenon(measured, ANYTHING, bridge)
    substantive = [item for item in match.criteria if item.name != "observables"]
    assert sum(1 for item in substantive if item.status == CRITERION_INSIDE) == 1
    assert MINIMUM_ASSESSED_CRITERIA == 2
    assert match.status == MATCH_UNASSESSABLE
    assert label_pattern(1, measured, _declared(ANYTHING), bridge).label == LABEL_UNASSESSABLE


def test_the_observable_criterion_is_never_counted_as_evidence(projections, bridge):
    """Otherwise 'this record holds the right field' would be half of every recognition."""
    measured = measure_pattern(projections[0], bridge)
    match = match_phenomenon(measured, ANYTHING, bridge)
    observable = [item for item in match.criteria if item.name == "observables"][0]
    assert observable.status == CRITERION_INSIDE
    assert match.status == MATCH_UNASSESSABLE


def test_a_pattern_labelled_outside_a_rule_has_no_lead_to_be_judged_on(projections, bridge,
                                                                      catalogue):
    labels = cross_reference(catalogue, projections, bridge)
    match = labels[0].matches[0]
    lead = [item for item in match.criteria if item.name == "lead"][0]
    assert lead.status == CRITERION_UNASSESSABLE
    assert "a lead is a property of a rule and not of a pattern" in lead.note


def test_a_lead_supplied_from_a_rule_is_judged_and_can_exclude(projections, bridge):
    measured = measure_pattern(projections[ANTECEDENT - 1], bridge)
    inside = match_phenomenon(measured, MESOSCALE, bridge,
                              lead=bridge.lead(TransitionWindow(1.0, 3.0, "frames")))
    assert [item for item in inside.criteria
            if item.name == "lead"][0].status == CRITERION_INSIDE      # 6 to 18 hours
    outside = match_phenomenon(measured, MESOSCALE, bridge,
                               lead=bridge.lead(TransitionWindow(10.0, 20.0, "frames")))
    assert [item for item in outside.criteria
            if item.name == "lead"][0].status == CRITERION_OUTSIDE     # 60 to 120 hours
    assert outside.status == MATCH_EXCLUDED


def test_a_pattern_consistent_with_every_entry_is_flagged_as_separating_nothing(projections,
                                                                                bridge):
    wide = Phenomenon(name="wide", citation="c", observables=(OBSERVABLE,),
                      horizontal_scale=Envelope(1.0, 100000.0, "km"),
                      lead=Envelope(0.0, 10000.0, "hours"), cardinality=2)
    label = label_pattern(1, measure_pattern(projections[0], bridge), _declared(wide), bridge)
    assert label.label == LABEL_RECOGNISED
    assert label.non_discriminating is True
    assert "separates it from nothing" in label.as_record()["caution"]


def test_every_label_carries_the_sentence_that_stops_it_being_read_as_a_detection(projections,
                                                                                 bridge,
                                                                                 catalogue):
    for label in cross_reference(catalogue, projections, bridge):
        record = label.as_record()
        assert record["recognition_note"] == RECOGNITION_NOTE
        assert "not a detection of the named phenomenon" in record["recognition_note"]
        assert "'unrecognised' is not a discovery" in record["recognition_note"]


def test_cross_reference_refuses_anything_that_is_not_a_reference_catalogue(projections,
                                                                           bridge):
    with pytest.raises(InvalidParameterError):
        cross_reference(object(), projections, bridge)


# ---------------------------------------------------------------------------------------------
# section 6: could it have failed? measured before anything is adjudicated
# ---------------------------------------------------------------------------------------------


def test_the_catalogue_that_excludes_things_is_recorded_as_discriminating(projections, bridge,
                                                                          catalogue):
    labels = cross_reference(catalogue, projections, bridge)
    power = discrimination(catalogue, labels, bridge)
    assert power.verdict == "DISCRIMINATING"
    assert power.n_patterns == len(projections)
    assert power.patterns_excluding_something == len(projections)
    assert power.n_exclusions >= len(projections)
    assert "capable of refusing recognition" in power.reason


def test_a_catalogue_that_excluded_nothing_is_invalid_rather_than_passing(projections, bridge):
    """Recognition by imprecision: the failure mode this whole task invites."""
    wide = Phenomenon(name="wide", citation="c", observables=(OBSERVABLE,),
                      horizontal_scale=Envelope(1.0, 100000.0, "km"),
                      lead=Envelope(0.0, 10000.0, "hours"), cardinality=2)
    catalogue = _declared(wide)
    power = discrimination(catalogue, cross_reference(catalogue, projections, bridge), bridge)
    assert power.verdict == GATE_INVALID
    assert power.n_exclusions == 0
    assert "could not have failed anything" in power.reason


def test_the_two_reasons_a_cross_reference_can_be_wholly_unassessable_are_told_apart(
        projections, bridge):
    absent = discrimination(_declared(ELSEWHERE),
                            cross_reference(_declared(ELSEWHERE), projections, bridge), bridge)
    assert absent.verdict == GATE_INVALID
    assert "%d of them because no entry asks for" % len(projections) in absent.reason
    thin = discrimination(_declared(ANYTHING),
                          cross_reference(_declared(ANYTHING), projections, bridge), bridge)
    assert thin.verdict == GATE_INVALID
    assert "0 of them because no entry asks for" in thin.reason
    assert "fewer than %d of each entry's criteria" % MINIMUM_ASSESSED_CRITERIA in thin.reason


def test_the_discrimination_record_publishes_the_grids_own_cell_range(projections, bridge,
                                                                      catalogue):
    power = discrimination(catalogue, cross_reference(catalogue, projections, bridge), bridge)
    record = power.as_record()
    assert record["cell_km"]["low"] == pytest.approx(CELL_KM)
    assert record["cell_km"]["ratio"] == pytest.approx(1.0)
    assert "cannot exclude anything" in record["cell_km"]["note"]
    assert record["cadence_hours"] == pytest.approx(float(CADENCE_HOURS))


# ---------------------------------------------------------------------------------------------
# section 7: the gate
# ---------------------------------------------------------------------------------------------


def test_the_gate_passes_when_a_recognised_pattern_leads_the_declared_ranking(ranked,
                                                                             projections,
                                                                             bridge,
                                                                             catalogue):
    series, report = ranked
    gate = physical_gate(report, _declaration(catalogue), projections, bridge, series)
    assert gate.status == GATE_PASS
    assert gate.phase_4g_may_start is True
    assert [item.rank for item in gate.ranked] == [1, 2]
    assert gate.ranked[0].antecedent == ANTECEDENT
    assert gate.ranked[0].label.label == LABEL_RECOGNISED
    assert gate.ranked[1].antecedent == DECOY
    assert gate.ranked[1].label.label == LABEL_UNRECOGNISED
    assert any("rank 1 of the q_value ranking" in text for text in gate.reasons)


def test_the_gate_fails_against_a_catalogue_whose_envelopes_do_not_fit(ranked, projections,
                                                                      bridge):
    """The path that matters: R10's presumption is that the pipeline is broken, and it is said."""
    series, report = ranked
    gate = physical_gate(report, _declaration(_declared(PLANETARY)), projections, bridge,
                         series)
    assert gate.status == GATE_FAIL
    assert gate.phase_4g_may_start is False
    assert gate.discrimination.verdict == "DISCRIMINATING"
    assert any("the pipeline is broken (R10)" in text for text in gate.reasons)
    assert any("Phase 4G does not start" in text for text in gate.reasons)


def test_a_failing_gate_is_not_reached_by_a_catalogue_that_could_not_have_passed(ranked,
                                                                                projections,
                                                                                bridge):
    series, report = ranked
    gate = physical_gate(report, _declaration(_declared(ANYTHING)), projections, bridge, series)
    assert gate.status == GATE_INVALID
    assert gate.phase_4g_may_start is False
    assert any("This is not a FAIL" in text for text in gate.reasons)


def test_a_record_that_cannot_carry_any_entrys_observable_is_invalid(ranked, projections,
                                                                    bridge):
    series, report = ranked
    gate = physical_gate(report, _declaration(_declared(ELSEWHERE)), projections, bridge,
                         series)
    assert gate.status == GATE_INVALID
    assert all(item.label.label == LABEL_UNASSESSABLE for item in gate.ranked)


def test_the_gate_refuses_a_catalogue_no_maintainer_has_signed(ranked, projections, bridge):
    series, report = ranked
    draft = _declared(MESOSCALE, PLANETARY, status=CATALOGUE_DRAFT)
    with pytest.raises(InvalidParameterError) as excinfo:
        physical_gate(report, _declaration(draft), projections, bridge, series)
    assert "a draft is a proposal nobody has signed" in str(excinfo.value)


def test_the_shipped_reference_cannot_gate_anything_until_it_is_reviewed(ranked, projections,
                                                                        bridge):
    series, report = ranked
    with pytest.raises(InvalidParameterError):
        physical_gate(report, _declaration(draft_southern_ocean_reference()), projections,
                      bridge, series)


def test_a_declaration_without_a_documented_period_is_refused(catalogue):
    with pytest.raises(InvalidParameterError) as excinfo:
        GateDeclaration(catalogue=catalogue, rank_by="q_value", top_n=1,
                        period_justification="   ", declared_by="a", declared_on="b")
    assert "some period contained something" in str(excinfo.value)


def test_a_declaration_naming_an_unknown_ranking_key_is_refused(catalogue):
    with pytest.raises(InvalidParameterError) as excinfo:
        GateDeclaration(catalogue=catalogue, rank_by="confidence", top_n=1,
                        period_justification="j", declared_by="a", declared_on="b")
    assert "must name its key before it is run" in str(excinfo.value)
    assert set(RANK_KEYS) == {"q_value", "p_value", "lift", "support"}


def test_a_declaration_with_a_nonsensical_top_n_is_refused(catalogue):
    for value in (0, -1, True, 1.5):
        with pytest.raises(InvalidParameterError):
            GateDeclaration(catalogue=catalogue, rank_by="q_value", top_n=value,
                            period_justification="j", declared_by="a", declared_on="b")


def test_narrowing_the_top_n_to_one_still_finds_the_planted_rule(ranked, projections, bridge,
                                                                 catalogue):
    series, report = ranked
    gate = physical_gate(report, _declaration(catalogue, top_n=1), projections, bridge, series)
    assert gate.status == GATE_PASS
    assert len(gate.ranked) == 1


def test_the_declared_ranking_key_decides_which_rules_are_looked_at(ranked, projections,
                                                                    bridge, catalogue):
    """T4F.4's finding carried into the gate: the key is part of the declaration, not a detail."""
    series, report = ranked
    by_q = physical_gate(report, _declaration(catalogue, rank_by="q_value", top_n=1),
                         projections, bridge, series)
    by_support = physical_gate(report, _declaration(catalogue, rank_by="support", top_n=1),
                               projections, bridge, series)
    assert by_q.ranked[0].antecedent == ANTECEDENT
    assert by_support.ranked[0].antecedent == DECOY
    assert by_q.status == GATE_PASS and by_support.status == GATE_FAIL
    assert by_q.declaration_digest != by_support.declaration_digest


def test_the_declaration_digest_moves_when_any_part_of_the_declaration_does(catalogue):
    base = _declaration(catalogue)
    assert base.digest == _declaration(catalogue).digest
    assert base.digest != _declaration(catalogue, top_n=3).digest
    assert base.digest != GateDeclaration(
        catalogue=catalogue, rank_by="q_value", top_n=2,
        period_justification="a different period entirely", declared_by="the T4F.6 suite",
        declared_on="2026-09-06").digest


def test_a_lead_is_refused_when_the_series_is_not_on_the_records_clock(ranked, bridge):
    """The 120-frame planted series is not the 24-frame record, and the hours say so."""
    series, _report = ranked
    lead = lead_for(bridge, series, WINDOW)
    assert not lead.measured
    assert "counted on a different clock" in lead.refusal
    assert bridge.lead(WINDOW).measured           # the bridge alone would have converted it


def test_a_series_on_the_records_own_frames_does_get_its_lead_in_hours(record, bridge):
    on_clock = _series(frames=[float(item) for item in record[1].times])
    lead = lead_for(bridge, on_clock, WINDOW)
    assert lead.measured
    assert lead.units == "hours"
    assert (lead.low, lead.high) == (pytest.approx(6.0), pytest.approx(18.0))
    assert "this record's own measured cadence" in lead.basis


def test_a_gate_on_a_record_with_no_clock_still_adjudicates_on_what_it_can(record, projections,
                                                                          ranked, catalogue):
    """The lead refuses; scale, cardinality and separation are still three real criteria."""
    undated = _undated(record)
    series, report = ranked
    gate = physical_gate(report, _declaration(catalogue), projections, undated, series)
    assert gate.status == GATE_PASS
    assert not gate.ranked[0].lead.measured
    assert "carries no calendar" in gate.ranked[0].lead.refusal


def test_rules_the_report_did_not_measure_are_counted_and_not_ranked_as_though_zero(
        projections, bridge, catalogue):
    series = _series()
    report = precursor_report(series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES,
                              seed=SEED, pairs=[(ANTECEDENT, CONSEQUENT), (DECOY, CONSEQUENT)])
    gate = physical_gate(report, _declaration(catalogue, rank_by="lift", top_n=2), projections,
                         bridge, series)
    measured = [rule for rule in report.rules if rule.lift is not None]
    assert gate.unmeasured_rules == len(report.rules) - len(measured)
    assert len(gate.ranked) <= len(measured)


def test_the_gate_refuses_a_report_or_a_declaration_of_the_wrong_type(ranked, projections,
                                                                     bridge, catalogue):
    series, report = ranked
    with pytest.raises(InvalidParameterError):
        physical_gate(object(), _declaration(catalogue), projections, bridge, series)
    with pytest.raises(InvalidParameterError):
        physical_gate(report, object(), projections, bridge, series)


# ---------------------------------------------------------------------------------------------
# section 8: the receipt, and what it may not be read as
# ---------------------------------------------------------------------------------------------


def test_the_receipt_names_the_catalogue_and_the_declaration_it_was_adjudicated_under(
        ranked, projections, bridge, catalogue):
    series, report = ranked
    gate = physical_gate(report, _declaration(catalogue), projections, bridge, series)
    record = describe_gate(gate)
    assert record["schema"] == GATE_SCHEMA
    assert record["catalogue_digest"] == catalogue.digest
    assert record["declaration_digest"] == _declaration(catalogue).digest
    assert record["status"] == GATE_PASS
    assert record["phase_4g_may_start"] is True


def test_the_receipt_carries_the_grid_and_its_warnings_rather_than_only_the_verdict(
        ranked, projections, bridge, catalogue):
    series, report = ranked
    record = describe_gate(physical_gate(report, _declaration(catalogue), projections, bridge,
                                         series))
    assert record["record"]["observable"] == OBSERVABLE
    assert record["record"]["cell_km"]["low"] == pytest.approx(CELL_KM)
    assert record["record"]["cadence_hours"] == pytest.approx(float(CADENCE_HOURS))
    assert "grid_warnings" in record["record"]


def test_the_receipt_lists_every_pattern_and_not_only_the_ranked_ones(ranked, projections,
                                                                     bridge, catalogue):
    series, report = ranked
    record = describe_gate(physical_gate(report, _declaration(catalogue), projections, bridge,
                                         series))
    assert len(record["labels"]) == len(projections)
    assert len(record["ranking"]) == 2


def test_the_claim_boundary_refuses_the_six_words_by_name(ranked, projections, bridge,
                                                          catalogue):
    series, report = ranked
    record = describe_gate(physical_gate(report, _declaration(catalogue), projections, bridge,
                                         series))
    boundary = record["claim_boundary"]
    assert boundary == GATE_CLAIM_BOUNDARY
    for word in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert word in boundary
    assert "INVALID means the question could not be asked" in boundary
    assert "must never be read or reported as a FAIL" in boundary


def test_the_receipt_says_what_pass_licenses_and_it_is_only_phase_4g(ranked, projections,
                                                                    bridge, catalogue):
    series, report = ranked
    record = describe_gate(physical_gate(report, _declaration(catalogue), projections, bridge,
                                         series))
    assert "licenses the start of Phase 4G and claims nothing about the atmosphere" in \
        record["claim_boundary"]


def test_describing_something_that_is_not_a_gate_is_refused():
    with pytest.raises(InvalidParameterError):
        describe_gate(object())


# ---------------------------------------------------------------------------------------------
# section 9: the twelve claims mutation testing found nothing was holding
# ---------------------------------------------------------------------------------------------


def test_an_entry_of_one_member_is_refused():
    """T4E.1 never builds a constellation of one, and a configuration of one has no geometry."""
    with pytest.raises(InvalidParameterError) as excinfo:
        Phenomenon(name="x", citation="c", observables=(OBSERVABLE,),
                   horizontal_scale=Envelope(1.0, 2.0, "km"), lead=Envelope(1.0, 2.0, "hours"),
                   cardinality=1)
    assert "A constellation of one has no geometry" in str(excinfo.value)


def test_two_entries_agreeing_on_only_one_criterion_are_not_an_overlapping_pair():
    """Overlap is a conjunction: agreeing on scale alone does not make two entries collide."""
    scale_only = ReferenceCatalogue(phenomena=(MESOSCALE, SAME_SCALE_OTHER_LEAD),
                                    declared_by="a", declared_on="b")
    assert scale_only.overlapping_pairs() == ()
    lead_only = ReferenceCatalogue(phenomena=(MESOSCALE, SAME_LEAD_OTHER_SCALE),
                                   declared_by="a", declared_on="b")
    assert lead_only.overlapping_pairs() == ()
    both = ReferenceCatalogue(
        phenomena=(MESOSCALE, Phenomenon(
            name="both", citation="c", observables=(OBSERVABLE,),
            horizontal_scale=Envelope(400.0, 900.0, "km"), lead=Envelope(6.0, 30.0, "hours"))),
        declared_by="a", declared_on="b")
    assert [pair["entries"] for pair in both.overlapping_pairs()] == \
        [["mesoscale-vortex-pair", "both"]]


def test_a_cell_is_measured_by_its_shorter_side_on_an_anisotropic_cartesian_grid(record):
    """`dy` is the shorter side here, and every grid in the suite above hid that it could be."""
    tall = GridSpec.cartesian(record[1].grid.shape, dy_m=10000.0, dx_m=40000.0)
    bridge = physical_bridge(
        geography_of(_regrid(record[1], tall), time_units="frames", values=record[0]),
        variable="amplitude")
    assert bridge.cell_km_low == pytest.approx(10.0)
    assert bridge.cell_km_high == pytest.approx(40.0)


def test_a_latlon_grid_whose_longitudes_are_coarser_uses_both_metrics(record):
    """At 0.25 degrees square the zonal side is always shorter; at 1.0 by 0.25 it is not."""
    wide = GridSpec.latlon(record[1].grid.shape, lat0=-5.0, dlat=-0.25, lon0=140.0, dlon=1.0)
    bridge = physical_bridge(
        geography_of(_regrid(record[1], wide), time_units="frames", values=record[0]),
        variable="t")
    dx = wide.dx_metres().numpy() / 1000.0
    dy = wide.dy_metres().numpy() / 1000.0
    assert float(dy.min()) < float(dx.min())          # the meridional side is the shorter one
    assert bridge.cell_km_low == pytest.approx(float(min(dx.min(), dy.min())))
    assert bridge.cell_km_high == pytest.approx(float(max(dx.max(), dy.max())))


def test_the_separation_is_the_distance_between_centres_and_not_one_axis_of_it(projections,
                                                                               bridge):
    """Recomputed here from the footprints, so a projection onto one axis cannot pass."""
    projection = projections[ANTECEDENT - 1]
    pairs = [occurrence.footprints for occurrence in projection.occurrences]
    distances = [float(np.hypot(left.centre.row - right.centre.row,
                                left.centre.col - right.centre.col)) for left, right in pairs]
    rows_only = [abs(left.centre.row - right.centre.row) for left, right in pairs]
    measured = measure_pattern(projection, bridge)["separation"]
    assert measured.low == pytest.approx(min(distances) * CELL_KM)
    assert measured.high == pytest.approx(max(distances) * CELL_KM)
    assert min(distances) > max(rows_only)


def test_a_pattern_with_nothing_on_the_map_is_refused_rather_than_measured(projections, bridge):
    empty = dataclasses.replace(projections[0], occurrences=(), n_members=0)
    with pytest.raises(InvalidParameterError) as excinfo:
        measure_pattern(empty, bridge)
    assert "labelling an empty set" in str(excinfo.value)


def test_the_cardinality_criterion_can_exclude_on_its_own(projections, bridge):
    """Every constellation of this record has two members, so a triple excludes all of them."""
    labels = cross_reference(_declared(TRIPLE), projections, bridge)
    assert {item.label for item in labels} == {LABEL_UNRECOGNISED}
    for item in labels:
        assert item.excluded_by == ("three-member-configuration",)
        verdict = [criterion for criterion in item.matches[0].criteria
                   if criterion.name == "cardinality"][0]
        assert verdict.status == CRITERION_OUTSIDE


def test_the_separation_criterion_can_exclude_on_its_own(projections, bridge):
    """Some patterns clear the mesoscale envelope on scale and are excluded only by separation."""
    limited = cross_reference(_declared(MESOSCALE), projections, bridge)
    with_limit = {item.pattern_id: item.label for item in limited}
    without = {item.pattern_id: item.label
               for item in cross_reference(_declared(NO_SEPARATION_LIMIT), projections, bridge)}
    only_separation = sorted(pattern for pattern in with_limit
                             if with_limit[pattern] != without[pattern])
    assert only_separation, "no pattern is excluded by separation alone; the test is vacuous"
    by_pattern = {item.pattern_id: item for item in limited}
    for pattern in only_separation:
        assert with_limit[pattern] == LABEL_UNRECOGNISED
        assert without[pattern] == LABEL_RECOGNISED
        match = by_pattern[pattern].matches[0]
        outside = [item.name for item in match.criteria if item.status == CRITERION_OUTSIDE]
        assert outside == ["separation"]


#: A fourth pattern whose every occurrence sits at the very end of the record, so its window is
#: always truncated, its confidence is never defined and its lift is therefore not a number.
UNMEASURED = 4
UNMEASURED_TIMES = [117, 118, 119]


def _series_with_an_unmeasured_pair():
    grid = ObservationGrid(frames=tuple(float(item) for item in range(FRAMES)),
                           time_units="frames", cadence=1.0)
    events = []
    for pattern_id, times in ((ANTECEDENT, A_TIMES), (CONSEQUENT, B_TIMES),
                              (DECOY, DECOY_TIMES), (UNMEASURED, UNMEASURED_TIMES)):
        for time in times:
            events.append(PatternEvent(
                pattern_id=pattern_id, time=float(time), identity="p%d@%d" % (pattern_id, time),
                cardinality=2, mode="scale_specific"))
    events.sort(key=lambda item: (item.time, item.pattern_id))
    return EventSeries(events=tuple(events), grid=grid, mode="scale_specific",
                       units_confirmed_by_members=len(events), units_uncarried_by_members=0)


@pytest.fixture(scope="module")
def with_unmeasured():
    series = _series_with_an_unmeasured_pair()
    report = precursor_report(
        series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES, seed=SEED,
        pairs=[(ANTECEDENT, CONSEQUENT), (UNMEASURED, CONSEQUENT)])
    return series, report


def test_a_rule_the_record_could_not_measure_is_counted_and_left_out_of_the_ranking(
        with_unmeasured, projections, bridge, catalogue):
    """Ranking a missing lift as zero would put an unmeasurable rule above a measured one."""
    series, report = with_unmeasured
    assert report.rule(UNMEASURED, CONSEQUENT, WINDOW).lift is None
    gate = physical_gate(report, _declaration(catalogue, rank_by="lift", top_n=5), projections,
                         bridge, series)
    assert gate.unmeasured_rules == 1
    assert [item.antecedent for item in gate.ranked] == [ANTECEDENT]
    assert gate.status == GATE_PASS


def test_a_report_with_no_measured_figure_at_all_is_invalid_rather_than_empty(
        with_unmeasured, projections, bridge, catalogue):
    series, _report = with_unmeasured
    only_unmeasurable = precursor_report(
        series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES, seed=SEED,
        pairs=[(UNMEASURED, CONSEQUENT)])
    gate = physical_gate(only_unmeasurable, _declaration(catalogue, rank_by="lift"),
                         projections, bridge, series)
    assert gate.status == GATE_INVALID
    assert gate.ranked == ()
    assert any("no rule in this report carries a lift" in text for text in gate.reasons)


def test_a_series_that_names_no_searched_frames_gets_no_lead(bridge):
    """There is then nothing to check the two clocks against each other with."""
    lead = lead_for(bridge, object(), WINDOW)
    assert not lead.measured
    assert "nothing to check its clock against" in lead.refusal


def test_the_ranked_label_reads_the_measurement_the_cross_reference_already_took(
        ranked, projections, bridge, catalogue):
    """One measurement, several readers: the gate must not measure a pattern a second time."""
    series, report = ranked
    gate = physical_gate(report, _declaration(catalogue), projections, bridge, series)
    by_pattern = {item.pattern_id: item for item in gate.labels}
    for entry in gate.ranked:
        assert entry.label is not None
        assert entry.label.measured == by_pattern[entry.antecedent].measured
        assert entry.label.excluded_by == by_pattern[entry.antecedent].excluded_by


def test_the_receipt_counts_only_the_criteria_the_decision_rule_counts(projections, bridge):
    """`ANYTHING` decides one substantive criterion, and the receipt must not say two."""
    measured = measure_pattern(projections[0], bridge)
    thin = match_phenomenon(measured, ANYTHING, bridge)
    assert thin.n_assessed == 1
    assert thin.as_record()["substantive_criteria_decided"] == 1
    assert thin.as_record()["minimum_for_consistency"] == MINIMUM_ASSESSED_CRITERIA
    assert thin.status == MATCH_UNASSESSABLE
    full = match_phenomenon(measured, MESOSCALE, bridge)
    assert full.n_assessed == 3          # scale, cardinality, separation; the lead has no rule
    assert full.status == MATCH_CONSISTENT


def test_the_receipt_calls_an_exclusion_count_what_it_is(projections, bridge, catalogue):
    """One pattern excluded by one entry on three criteria is one exclusion, not three."""
    labels = cross_reference(catalogue, projections, bridge)
    power = discrimination(catalogue, labels, bridge)
    record = power.as_record()
    assert record["pattern_entry_exclusions"] == sum(len(item.excluded_by) for item in labels)
    assert "criterion_exclusions" not in record
    both = Phenomenon(name="wrong-on-two-counts", citation="c", observables=(OBSERVABLE,),
                      horizontal_scale=Envelope(1.0, 100.0, "km"),
                      lead=Envelope(0.0, 10000.0, "hours"), cardinality=3)
    one_entry = _declared(both)
    single = cross_reference(one_entry, projections, bridge)
    outside = [criterion.name for item in single for match in item.matches
               for criterion in match.criteria if criterion.status == CRITERION_OUTSIDE]
    assert len(outside) == 2 * len(projections)          # both criteria fail, on every pattern
    assert discrimination(one_entry, single, bridge).n_exclusions == len(projections)


def test_an_entry_the_record_cannot_check_does_not_rescue_a_label_that_separates_nothing(
        projections, bridge):
    """An unassessable entry discriminates in neither direction, so it cannot make a label do so."""
    wide = Phenomenon(name="wide", citation="c", observables=(OBSERVABLE,),
                      horizontal_scale=Envelope(1.0, 100000.0, "km"),
                      lead=Envelope(0.0, 10000.0, "hours"), cardinality=2)
    label = label_pattern(1, measure_pattern(projections[0], bridge),
                          _declared(wide, ELSEWHERE), bridge)
    assert label.label == LABEL_RECOGNISED
    assert label.consistent_with == ("wide",)
    assert label.unassessable_for == ("upper-level-precursor",)
    assert label.non_discriminating is True
    assert "every entry this record could check it against" in label.as_record()["caution"]
