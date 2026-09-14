"""T4F.7: where a rule holds and where it does not, and the six ways that question goes wrong.

1.  **The acceptance is a record built to disagree with itself.** Five boxes of one 260-frame
    field: `A` where the rule is found, `B` built identically, `C` built with the same pair of
    structures at a longer lag, `D` built with the consequent *before* the antecedent, and `E`
    left empty. The rule holds in `A`, `B` and `C`, does not hold in `D`, and `E` never took the
    test -- and the module has to say all four of those different things.

2.  **A region where nothing occurred did not fail.** `E` comes back `NOT_ASSESSABLE`, not
    `DOES_NOT_HOLD`. That distinction is the whole task: rendering an absence of data as
    evidence of locality is how "we found a thing about the Alps" gets written down as a fact
    about the atmosphere.

3.  **Membership is decided on footprints, and there are three ways of not having one.** Of
    this record's 1,063 configurations, 281 sit wholly inside a declared box; 364 straddle two of
    them, 397 reach out of the one box they touch into undeclared ground, and 21 are outside
    every box. All four are counted, and none is quietly assigned to whichever region held more
    of it.

4.  **Identity may not be fitted on the ground it is re-tested on.** `match_into_catalogue`
    holds the centroids and the calibrated radius fixed, and `identity_leakage` refuses
    `general` when the pattern being re-tested was partly defined by the held-out regions.

5.  **Adjacent boxes are not independent.** The gaps are published in cells and kilometres, and
    `general` is refused while any held-out region is closer to the discovery region than a
    declared decorrelation length -- or while no such length has been declared at all.

6.  **A measurement that shaped the design is pinned here.** T4E.2's signature is invariant to
    rotation *by construction*, so it cannot tell an `HL` configuration from an `LH` one: on this
    record the same `HL` signatures sit a median 0.447 from their own centroid and 0.585 from the
    other one, both far inside a calibrated radius of 0.959. A catalogue whose patterns differ
    only by band orientation therefore cannot be matched into by signature distance, which is why
    the identity used below is declared rather than fitted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, ConstellationPattern, PatternCatalogue, SignatureMetric, SignaturePoint,
    calibrate_signature_tolerance,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_events import ObservationGrid
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_projection import geography_of
from src.analysis_engine.spectral_reference import physical_bridge
from src.analysis_engine.spectral_regions import (
    GENERALISATION_CLAIM_BOUNDARY, GENERALISATION_SCHEMA, MEMBERSHIP_INSIDE,
    MEMBERSHIP_OUTSIDE, MEMBERSHIP_PARTLY_OUTSIDE, MEMBERSHIP_STRADDLES,
    MINIMUM_HELD_OUT_ASSESSED, REGION_FAILS,
    REGION_HOLDS, REGION_UNASSESSABLE, ROLE_DISCOVERY, ROLE_HELD_OUT, VERDICT_GENERAL,
    VERDICT_REGIONAL, VERDICT_UNASSESSABLE, Physiography, Region, RegionPartition,
    assign_regions, cross_region_generalisation, describe_generalisation, identity_leakage,
    match_into_catalogue, region_series, restrict_catalogue,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.analysis_engine.spectral_tracking import track_spectral_features
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import decompose_sequence

ROWS, COLS, STEPS = 128, 184, 260
NARROW, LONG = 2.0, 24.0
SURROGATES, SEED = 240, 20260907
WINDOW = TransitionWindow(1.0, 3.0, "frames")
METRIC = SignatureMetric(AttributeWeights(1.0, 1.0, 1.0, 1.0))
ANTECEDENT, CONSEQUENT = 1, 2

#: The declared boxes. `E` sits over ground no structure was ever drawn on.
BOXES = {"A": ((16, 56), (16, 56)), "B": ((16, 56), (72, 112)), "C": ((16, 56), (128, 168)),
         "D": ((72, 112), (16, 56)), "E": ((72, 112), (72, 112))}
CENTRE = {name: ((rows[0] + rows[1]) / 2.0, (cols[0] + cols[1]) / 2.0)
          for name, (rows, cols) in BOXES.items()}

_rng = np.random.default_rng(SEED)
EPISODES, _cursor = [], 0
while _cursor + 10 < STEPS:
    EPISODES.append(int(_cursor))
    _cursor += int(_rng.integers(10, 15))

#: `A` and `B` run the chain at a lag of two frames, `C` at three, and `D` carries both
#: structures with the consequent six frames *before* the antecedent -- far enough from the next
#: episode that neither follows the other inside the declared window. `E` is left empty. The
#: episode starts are jittered off any lattice on purpose: a periodic schedule is reproduced
#: exactly by every rotation that is a multiple of its period, which would put a mode of the
#: shifting null on the observed value and make a real effect look like chance.
ZONAL_AT = {"A": [t + 1 for t in EPISODES], "B": [t + 1 for t in EPISODES],
            "C": [t + 2 for t in EPISODES], "D": [t + 6 for t in EPISODES]}
MERID_AT = {"A": [t + 3 for t in EPISODES], "B": [t + 3 for t in EPISODES],
            "C": [t + 5 for t in EPISODES], "D": [t + 0 for t in EPISODES]}

SAME = Physiography("uniform-plane", "built with the same ridge pair at the same lag",
                    "the fixture's own construction, below")
LATER = Physiography("delayed-plane", "built with the same ridge pair at a longer lag",
                     "the fixture's own construction, below")
REVERSED = Physiography("reversed-plane", "built with the consequent before the antecedent",
                        "the fixture's own construction, below")
PHYSIOGRAPHY = {"A": SAME, "B": SAME, "C": LATER, "D": REVERSED, "E": SAME}

_CACHE: dict = {}


def _ridge(cy, cx, sy, sx):
    yy, xx = torch.meshgrid(torch.arange(ROWS, dtype=torch.float64) - cy,
                            torch.arange(COLS, dtype=torch.float64) - cx, indexing="ij")
    return torch.exp(-(yy ** 2) / (2.0 * sy ** 2) - (xx ** 2) / (2.0 * sx ** 2))


def _record_tracked():
    """The tracking pass behind the record, for the one test that needs another cardinality."""
    _record()
    return _CACHE["tracked"]


def _record():
    """The five-region field, decomposed, tracked and enumerated once (about four seconds)."""
    if "record" not in _CACHE:
        grid = GridSpec.cartesian((ROWS, COLS), dy_m=31000.0)
        generator = torch.Generator().manual_seed(SEED)
        fields = []
        for time in range(STEPS):
            data = torch.zeros((ROWS, COLS), dtype=torch.float64)
            for name, times in ZONAL_AT.items():
                if time in times:
                    data = data + _ridge(*CENTRE[name], NARROW, LONG)
            for name, times in MERID_AT.items():
                if time in times:
                    data = data + _ridge(*CENTRE[name], LONG, NARROW)
            fields.append(PhysicalField(
                data + 0.02 * torch.randn(ROWS, COLS, generator=generator,
                                          dtype=torch.float64),
                grid=grid, units="dimensionless"))
        sequence = FieldSequence(fields, np.arange(STEPS, dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": 4, "wavelet": "haar"}).select(scales=[3])
        tracked = track_spectral_features(
            field, domain="synthetic", dataset="five_regions", variable="amplitude",
            time_units="frames", threshold_sigma=3.5)
        constellations = extract_constellations(tracked, cardinalities=(2,))
        geography = geography_of(field, time_units="frames", values=sequence)
        _CACHE["tracked"] = tracked
        _CACHE["record"] = (sequence, field, constellations, geography)
    return _CACHE["record"]


def _regions(names="ABCDE", *, physiography=True):
    return tuple(
        Region(name, BOXES[name][0], BOXES[name][1],
               ROLE_DISCOVERY if name == "A" else ROLE_HELD_OUT,
               PHYSIOGRAPHY[name] if physiography else None)
        for name in names)


def _partition(names="ABCDE", **kwargs):
    return RegionPartition(_regions(names, **kwargs), declared_by="the T4F.7 suite",
                           declared_on="2026-09-07")


@pytest.fixture(scope="module")
def record():
    return _record()


@pytest.fixture(scope="module")
def geography(record):
    return record[3]


@pytest.fixture(scope="module")
def constellations(record):
    return record[2]


@pytest.fixture(scope="module")
def bridge(geography):
    return physical_bridge(geography, variable="amplitude")


@pytest.fixture(scope="module")
def partition():
    return _partition()


@pytest.fixture(scope="module")
def assignment(constellations, partition, geography):
    return assign_regions(constellations, partition, geography)


@pytest.fixture(scope="module")
def signatures(constellations):
    return sign_constellations(constellations, scale_invariant=False).signatures


@pytest.fixture(scope="module")
def catalogue(constellations, signatures):
    """One pattern per band pair, declared rather than fitted, as T4F.5's suite assembles one.

    Declared identity is what makes `fitted=()` the truthful answer below: no clustering ran, so
    no region -- held out or otherwise -- helped to define what a pattern is.
    """
    by_key = {tuple(item.key()): item for item in constellations}
    groups: dict = {}
    for signature in signatures:
        bands = by_key[tuple(signature.key)].bands
        if len(set(bands)) != 1:
            continue
        groups.setdefault(bands, []).append(signature)
    tolerance = calibrate_signature_tolerance(sorted(groups.values(), key=len)[-1][:3],
                                              metric=METRIC)
    patterns = []
    for index, bands in enumerate(sorted(groups)):
        members = tuple(sorted(groups[bands], key=lambda item: (item.time, item.key)))
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=index + 1, members=members, centroid=points[0],
            tolerance_radius=tolerance.value,
            observed_radius=max(METRIC.distance(point, points[0]) for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=METRIC, tolerance=tolerance,
                            n_signatures=sum(len(item.members) for item in patterns))


@pytest.fixture(scope="module")
def grid():
    return ObservationGrid(frames=tuple(float(i) for i in range(STEPS)), time_units="frames",
                           cadence=1.0)


def _run(catalogue, constellations, partition, geography, grid, bridge, **kwargs):
    options = {"antecedent": ANTECEDENT, "consequent": CONSEQUENT, "window": WINDOW,
               "n_surrogates": SURROGATES, "seed": SEED, "bridge": bridge,
               "decorrelation_km": 400.0, "fitted": ()}
    options.update(kwargs)
    return cross_region_generalisation(catalogue, constellations, partition=partition,
                                       geography=geography, grid=grid, **options)


@pytest.fixture(scope="module")
def result(catalogue, constellations, partition, geography, grid, bridge):
    return _run(catalogue, constellations, partition, geography, grid, bridge)


# ---------------------------------------------------------------------------------------------
# section 1: what a design must declare before it can re-test anything
# ---------------------------------------------------------------------------------------------


def test_a_physiography_without_a_source_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Physiography("coastal", "a coastline runs through it", "   ")
    assert "because someone typed the same word" in str(excinfo.value)


def test_a_region_with_an_unknown_role_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Region("X", (0, 10), (0, 10), "somewhere")
    assert "re-testing a rule on the region it was found in is not a re-test" in str(
        excinfo.value)


def test_a_region_with_a_reversed_or_negative_box_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Region("X", (10, 0), (0, 10), ROLE_HELD_OUT)
    assert "empty region tests nothing" in str(excinfo.value)
    with pytest.raises(InvalidParameterError):
        Region("X", (-1, 10), (0, 10), ROLE_HELD_OUT)


def test_a_partition_needs_exactly_one_discovery_region():
    with pytest.raises(InvalidParameterError) as excinfo:
        RegionPartition((Region("A", (0, 10), (0, 10), ROLE_HELD_OUT),
                         Region("B", (20, 30), (0, 10), ROLE_HELD_OUT)),
                        declared_by="a", declared_on="b")
    assert "nothing to generalise from" in str(excinfo.value)
    with pytest.raises(InvalidParameterError):
        RegionPartition((Region("A", (0, 10), (0, 10), ROLE_DISCOVERY),
                         Region("B", (20, 30), (0, 10), ROLE_DISCOVERY)),
                        declared_by="a", declared_on="b")


def test_a_partition_with_no_held_out_region_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        RegionPartition((Region("A", (0, 10), (0, 10), ROLE_DISCOVERY),),
                        declared_by="a", declared_on="b")
    assert "the same measurement reported twice" in str(excinfo.value)


def test_overlapping_regions_are_refused_because_one_would_confirm_the_other():
    with pytest.raises(InvalidParameterError) as excinfo:
        RegionPartition((Region("A", (0, 20), (0, 20), ROLE_DISCOVERY),
                         Region("B", (10, 30), (10, 30), ROLE_HELD_OUT)),
                        declared_by="a", declared_on="b")
    assert "the very occurrences it was discovered from" in str(excinfo.value)


def test_two_regions_of_one_name_are_refused():
    with pytest.raises(InvalidParameterError):
        RegionPartition((Region("A", (0, 10), (0, 10), ROLE_DISCOVERY),
                         Region("A", (20, 30), (0, 10), ROLE_HELD_OUT)),
                        declared_by="a", declared_on="b")


def test_the_partition_digest_moves_when_the_declared_design_does(partition):
    assert partition.digest == _partition().digest
    assert partition.digest != _partition("ABCE").digest
    assert partition.digest != _partition(physiography=False).digest


def test_the_partition_names_its_own_discovery_and_held_out_regions(partition):
    assert partition.discovery.name == "A"
    assert [item.name for item in partition.held_out] == ["B", "C", "D", "E"]
    assert partition.region("C").physiography.label == "delayed-plane"
    with pytest.raises(InvalidParameterError):
        partition.region("Z")


# ---------------------------------------------------------------------------------------------
# section 2: where a configuration is, decided on the footprints it was drawn with
# ---------------------------------------------------------------------------------------------


def test_every_configuration_is_placed_or_explicitly_not_placed(assignment, constellations):
    record = assignment.as_record()
    assert record["configurations"] == len(constellations)
    placed = sum(record["inside_each_region"].values())
    assert (placed + record["straddling_two_or_more_regions"]
            + record["reaching_outside_the_one_region_they_touch"]
            + record["outside_every_region"]) == len(constellations)


def test_the_acceptance_record_populates_four_boxes_and_leaves_the_fifth_empty(assignment):
    inside = assignment.as_record()["inside_each_region"]
    assert inside == {"A": 89, "B": 102, "C": 43, "D": 47, "E": 0}


def test_the_three_ways_of_not_being_in_a_region_are_kept_apart(assignment):
    """Giving a configuration to whichever box held more of it would put it in the wrong one.

    The three are different statements about the *declaration*: straddling two declared boxes
    says they were drawn closer together than this transform's own footprints, reaching outside
    the one box it touches says the configuration extends into undeclared ground, and being
    outside every box says it is somewhere else altogether.
    """
    record = assignment.as_record()
    assert record["straddling_two_or_more_regions"] == 364
    assert record["reaching_outside_the_one_region_they_touch"] == 397
    assert record["outside_every_region"] == 21
    for status, minimum, maximum in ((MEMBERSHIP_STRADDLES, 2, 99),
                                     (MEMBERSHIP_PARTLY_OUTSIDE, 1, 1),
                                     (MEMBERSHIP_OUTSIDE, 0, 0)):
        found = [item for item in assignment.memberships.values() if item.status == status]
        assert found, status
        for item in found:
            assert item.region is None
            assert minimum <= len(item.touched) <= maximum
    assert "not on the cell its maximum landed in" in record["note"]


def test_a_configuration_outside_every_declared_box_touches_nothing(assignment):
    outside = [item for item in assignment.memberships.values()
               if item.status == MEMBERSHIP_OUTSIDE]
    assert outside
    for item in outside:
        assert item.region is None and item.touched == ()


def test_membership_is_the_whole_footprint_and_not_the_maximum(assignment, partition,
                                                              constellations, geography):
    """The recorded box is the union of the projected footprints, recomputed here to say so.

    Asserting only that the box is wide would pass for a box with one bound taken from a
    maximum and the other from a footprint, which is exactly the mistake worth catching: at a
    boundary that puts a configuration in the wrong region by up to half a filter width.
    """
    from src.analysis_engine.spectral_projection import project_occurrence
    checked = 0
    for constellation in constellations:
        item = assignment.of(constellation.key())
        projection = project_occurrence(constellation, geography=geography, pattern_id=0)
        segments = [seg for print_ in projection.footprints for seg in print_.columns]
        assert item.rows == (min(p.rows[0] for p in projection.footprints),
                             max(p.rows[1] for p in projection.footprints))
        assert item.columns == (min(seg[0] for seg in segments),
                                max(seg[1] for seg in segments))
        centres = [int(p.centre.row_cell) for p in projection.footprints]
        assert item.rows[0] <= min(centres) and max(centres) <= item.rows[1]
        checked += 1
        if checked >= 40:
            break
    assert checked == 40
    for item in assignment.memberships.values():
        if item.status != MEMBERSHIP_INSIDE:
            continue
        region = partition.region(item.region)
        assert region.rows[0] <= item.rows[0] and item.rows[1] <= region.rows[1]
        assert region.columns[0] <= item.columns[0] and item.columns[1] <= region.columns[1]


def test_keys_in_a_region_are_exactly_the_ones_placed_there(assignment):
    for name in "ABCDE":
        keys = set(assignment.keys_in(name))
        expected = {key for key, item in assignment.memberships.items()
                    if item.status == MEMBERSHIP_INSIDE and item.region == name}
        assert keys == expected


def test_assigning_regions_refuses_anything_that_is_not_the_bound_map(constellations,
                                                                     partition, geography):
    with pytest.raises(InvalidParameterError):
        assign_regions(object(), partition, geography)
    with pytest.raises(InvalidParameterError):
        assign_regions(constellations, object(), geography)
    with pytest.raises(InvalidParameterError) as excinfo:
        assign_regions(constellations, partition, object())
    assert "rather than by the cell its maximum landed in" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 3: identity, and the ground it may not be fitted on
# ---------------------------------------------------------------------------------------------


def test_restricting_a_catalogue_keeps_the_identity_and_changes_only_the_occurrences(
        catalogue, assignment):
    restricted = restrict_catalogue(catalogue, assignment, "B")
    assert restricted.metric is catalogue.metric
    assert restricted.tolerance is catalogue.tolerance
    by_id = {item.pattern_id: item for item in catalogue}
    for pattern in restricted:
        assert pattern.centroid == by_id[pattern.pattern_id].centroid
        assert pattern.tolerance_radius == by_id[pattern.pattern_id].tolerance_radius
        assert len(pattern.members) < len(by_id[pattern.pattern_id].members)
        for member in pattern.members:
            assert assignment.of(member.key).region == "B"


def test_a_region_with_no_occurrence_gets_no_series_rather_than_an_empty_one(catalogue,
                                                                            assignment, grid):
    """An empty series would be counted as a region that was searched and found nothing."""
    assert region_series(catalogue, assignment, "E", grid=grid) is None
    assert region_series(catalogue, assignment, "B", grid=grid) is not None


def test_a_declared_identity_has_no_fitted_members_and_therefore_no_leakage(
        catalogue, assignment, partition):
    leakage = identity_leakage(catalogue, assignment, partition, fitted=())
    assert leakage.n_members == 0 and leakage.clean is True


def test_a_catalogue_fitted_on_held_out_ground_is_reported_as_leaking(catalogue, assignment,
                                                                     partition):
    """The conservative default: a catalogue that cannot say what defined it was defined by all."""
    leakage = identity_leakage(catalogue, assignment, partition)
    assert leakage.n_members == sum(len(item.members) for item in catalogue)
    assert leakage.n_from_held_out > 0
    assert leakage.clean is False
    assert set(leakage.per_pattern) <= {item.pattern_id for item in catalogue}
    assert "R6's leak with a map in place of a calendar" in leakage.as_record()["note"]


def test_only_the_fitted_members_count_towards_leakage(catalogue, assignment, partition):
    discovery_keys = set(assignment.keys_in("A"))
    fitted = [member.key for pattern in catalogue for member in pattern.members
              if tuple(member.key) in discovery_keys]
    leakage = identity_leakage(catalogue, assignment, partition, fitted=fitted)
    assert leakage.n_members == len(fitted)
    assert leakage.n_from_held_out == 0 and leakage.clean is True


def test_the_matcher_holds_the_identity_fixed_and_admits_only_what_the_radius_contains(
        catalogue, signatures):
    fitted_only = PatternCatalogue(
        patterns=tuple(ConstellationPattern(
            pattern_id=item.pattern_id, members=item.members[:3], centroid=item.centroid,
            tolerance_radius=item.tolerance_radius, observed_radius=item.observed_radius)
            for item in catalogue),
        metric=catalogue.metric, tolerance=catalogue.tolerance, n_signatures=6)
    matched = match_into_catalogue(fitted_only, signatures)
    assert matched.n_matched + matched.n_unmatched == len(signatures) - len(matched.fitted)
    by_id = {item.pattern_id: item for item in fitted_only}
    for pattern in matched.catalogue:
        assert pattern.centroid == by_id[pattern.pattern_id].centroid
        assert pattern.tolerance_radius == by_id[pattern.pattern_id].tolerance_radius
        # Only the *matched* members must lie inside the radius. A fitted member defines the
        # pattern and need not sit within a tolerance calibrated from replicates of it -- on
        # this record one of them sits at 1.033 against a radius of 0.959.
        fitted = set(matched.fitted)
        for member in pattern.members:
            if tuple(member.key) in fitted:
                continue
            point = SignaturePoint.from_signature(member)
            assert (catalogue.metric.distance(point, pattern.centroid)
                    <= pattern.tolerance_radius + 1e-12)
        assert any(tuple(member.key) in fitted for member in pattern.members)
    assert "Nothing is re-clustered" in matched.as_record()["note"]


def test_the_matcher_refuses_an_empty_catalogue_because_it_would_be_clustering(catalogue,
                                                                              signatures):
    empty = PatternCatalogue(patterns=(), metric=catalogue.metric,
                             tolerance=catalogue.tolerance, n_signatures=0)
    with pytest.raises(InvalidParameterError) as excinfo:
        match_into_catalogue(empty, signatures)
    assert "clustering under another name" in str(excinfo.value)


def test_the_signature_cannot_tell_two_band_orientations_apart_and_that_is_by_construction(
        catalogue, constellations, signatures):
    """T4E.2's signature is invariant to rotation, so `HL` and `LH` are the same shape to it.

    Measured rather than asserted, because it is the fact that decided how this suite builds its
    catalogue: a set of patterns that differ only by band orientation cannot be matched into by
    signature distance, and a caller who tried would get occurrences of one assigned to the
    other without any error being raised.
    """
    by_key = {tuple(item.key()): item for item in constellations}
    centroids = {item.pattern_id: item.centroid for item in catalogue}
    distances = {}
    for signature in signatures:
        bands = by_key[tuple(signature.key)].bands
        if len(set(bands)) != 1:
            continue
        point = SignaturePoint.from_signature(signature)
        for pattern_id, centroid in centroids.items():
            if centroid.family != point.family:
                continue
            distances.setdefault((bands[0], pattern_id), []).append(
                METRIC.distance(point, centroid))
    radius = catalogue.patterns[0].tolerance_radius
    own = float(np.median(distances[("L3/HL", 1)]))
    other = float(np.median(distances[("L3/HL", 2)]))
    assert radius == pytest.approx(0.9591, abs=0.001)
    assert own == pytest.approx(0.4473, abs=0.001)
    assert other == pytest.approx(0.5847, abs=0.001)
    # Both sit well inside the calibrated radius, and the gap between them is a fifth of it:
    # an `HL` configuration matches the `LH` centroid about as readily as its own.
    assert own < radius and other < radius
    assert (other - own) < 0.2 * radius


# ---------------------------------------------------------------------------------------------
# section 4: how far apart the boxes are, and whether that was declared to be far enough
# ---------------------------------------------------------------------------------------------


def test_the_gaps_between_regions_are_published_in_cells_and_kilometres(partition, bridge):
    gaps = {tuple(item["regions"]): item for item in partition.gaps(bridge)}
    assert gaps[("A", "B")]["gap_cells"] == pytest.approx(16.0)
    assert gaps[("A", "C")]["gap_cells"] == pytest.approx(72.0)
    assert gaps[("A", "B")]["gap_km"]["low"] == pytest.approx(16.0 * 31.0)
    assert gaps[("A", "E")]["gap_cells"] == pytest.approx(np.hypot(16.0, 16.0))


def test_a_grid_with_no_length_gets_a_gap_in_cells_and_a_refusal_for_kilometres(record,
                                                                               partition):
    from src.transform_engine.coefficient_field import CoefficientField
    field = record[1]
    pixel = CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times,
        grid=GridSpec.pixel(field.grid.shape), config=dict(field.config))
    bare = physical_bridge(geography_of(pixel, time_units="frames", values=record[0]),
                           variable="amplitude")
    gaps = partition.gaps(bare)
    assert all("gap_km" not in item for item in gaps)
    assert all("no size in kilometres" in item["gap_km_refusal"] for item in gaps)


def test_independence_is_unestablished_when_no_decorrelation_length_is_declared(
        catalogue, constellations, partition, geography, grid, bridge):
    outcome = _run(catalogue, constellations, partition, geography, grid, bridge,
                   decorrelation_km=None)
    assert outcome.independence["status"] == "UNESTABLISHED"
    assert "may be the discovery repeated rather than reproduced" in \
        outcome.independence["note"]


def test_a_held_out_region_closer_than_the_declared_decorrelation_length_is_named(
        catalogue, constellations, partition, geography, grid, bridge):
    outcome = _run(catalogue, constellations, partition, geography, grid, bridge,
                   decorrelation_km=1500.0)
    assert outcome.independence["status"] == "NOT_ESTABLISHED"
    named = outcome.independence[
        "closer_to_the_discovery_region_than_the_declared_decorrelation_length"]
    assert set(named) == {"B", "D", "E"}         # 496 km and 701 km, against a declared 1500
    assert "C" not in named                      # 2,232 km at its widest reading


# ---------------------------------------------------------------------------------------------
# section 5: the acceptance -- four different answers from one record
# ---------------------------------------------------------------------------------------------


def test_the_rule_holds_where_it_was_built_to_and_not_where_it_was_not(result):
    outcomes = {item.region: item for item in result.outcomes}
    assert outcomes["A"].status == REGION_HOLDS and outcomes["A"].role == ROLE_DISCOVERY
    assert outcomes["B"].status == REGION_HOLDS
    assert outcomes["C"].status == REGION_HOLDS
    assert outcomes["D"].status == REGION_FAILS
    assert outcomes["E"].status == REGION_UNASSESSABLE


def test_the_measured_figures_of_each_region_are_pinned(result):
    outcomes = {item.region: item for item in result.outcomes}
    assert (outcomes["A"].support, outcomes["A"].eligible_antecedents) == (38, 76)
    assert (outcomes["B"].support, outcomes["B"].eligible_antecedents) == (55, 81)
    assert (outcomes["C"].support, outcomes["C"].eligible_antecedents) == (14, 23)
    assert (outcomes["D"].support, outcomes["D"].eligible_antecedents) == (0, 24)
    assert outcomes["A"].lift == pytest.approx(3.89, abs=0.02)
    assert outcomes["B"].lift == pytest.approx(4.47, abs=0.02)
    assert outcomes["C"].lift == pytest.approx(3.72, abs=0.02)
    assert outcomes["D"].lift == pytest.approx(0.0)
    for name in ("B", "C"):
        assert outcomes[name].q_value == pytest.approx(0.0346, abs=0.001)
    assert outcomes["D"].q_value == pytest.approx(1.0)


def test_an_empty_region_never_took_the_test_and_is_not_a_failure(result):
    empty = [item for item in result.outcomes if item.region == "E"][0]
    assert empty.status == REGION_UNASSESSABLE
    assert empty.n_occurrences_of_antecedent == 0
    assert empty.support is None and empty.p_value is None and empty.lift is None
    assert "never put to the test here" in empty.reason
    assert "That is not the rule failing in this region" in empty.reason


def test_a_region_carrying_the_antecedent_but_never_the_consequent_is_also_unassessable(
        catalogue, constellations, geography, grid, bridge, assignment):
    """A base rate of zero is not a lift of zero, and the two are different findings."""
    restricted = restrict_catalogue(catalogue, assignment, "D")
    only_antecedent = PatternCatalogue(
        patterns=tuple(item for item in restricted if item.pattern_id == ANTECEDENT),
        metric=catalogue.metric, tolerance=catalogue.tolerance,
        n_signatures=len(restricted.patterns[0].members))
    outcome = _run(only_antecedent, constellations, _partition("ABDE"), geography, grid, bridge)
    by_region = {item.region: item for item in outcome.outcomes}
    assert by_region["D"].status == REGION_UNASSESSABLE
    assert by_region["D"].n_occurrences_of_antecedent > 0
    assert by_region["D"].n_occurrences_of_consequent == 0
    assert "the configuration was absent" in by_region["D"].reason.lower()


def test_the_discovery_region_is_reported_but_stays_out_of_the_corrected_family(result):
    discovery = [item for item in result.outcomes if item.role == ROLE_DISCOVERY][0]
    assert "counting it again is not a re-test" in discovery.reason
    assert result.correction["regions_declared_held_out"] == 4
    assert result.correction["regions_returning_a_p_value"] == 3


def test_the_family_is_the_regions_declared_and_not_the_ones_that_reported(result):
    """`E` returned no p-value and was still part of the design."""
    assert result.correction["regions_declared_held_out"] == 4
    assert result.correction["regions_returning_a_p_value"] == 3
    assert "would price a search as though it had not happened" in result.correction["note"]


# ---------------------------------------------------------------------------------------------
# section 6: the verdict, and every reason it is not `general`
# ---------------------------------------------------------------------------------------------


def test_a_rule_that_fails_in_one_assessable_region_is_regional(result):
    assert result.verdict == VERDICT_REGIONAL
    assert any("did not hold in D" in text for text in result.reasons)


def test_a_rule_holding_in_every_assessable_held_out_region_is_general(
        catalogue, constellations, geography, grid, bridge):
    """A different declared design -- three held-out boxes rather than four -- not a narrowing."""
    outcome = _run(catalogue, constellations, _partition("ABCE"), geography, grid, bridge)
    assert outcome.verdict == VERDICT_GENERAL
    holds = [item.region for item in outcome.held_out_outcomes
             if item.status == REGION_HOLDS]
    assert sorted(holds) == ["B", "C"]
    assert any("held in all 2 held-out regions" in text for text in outcome.reasons)


def test_one_assessable_held_out_region_is_not_enough_to_claim_generality(
        catalogue, constellations, geography, grid, bridge):
    outcome = _run(catalogue, constellations, _partition("ABE"), geography, grid, bridge)
    assert outcome.verdict == VERDICT_UNASSESSABLE
    assert MINIMUM_HELD_OUT_ASSESSED == 2
    assert any("one re-test is a second measurement" in text for text in outcome.reasons)


def test_a_rule_re_tested_only_where_nothing_occurred_is_unassessable(
        catalogue, constellations, geography, grid, bridge):
    outcome = _run(catalogue, constellations, _partition("AE"), geography, grid, bridge)
    assert outcome.verdict == VERDICT_UNASSESSABLE
    assert any("has not been re-tested anywhere" in text for text in outcome.reasons)


def test_generality_is_refused_when_the_identity_was_fitted_on_the_held_out_ground(
        catalogue, constellations, geography, grid, bridge):
    outcome = _run(catalogue, constellations, _partition("ABCE"), geography, grid, bridge,
                   fitted=None)
    assert outcome.leakage.clean is False
    assert outcome.verdict == VERDICT_REGIONAL
    assert any("partly fitted on the data they were re-tested on" in text
               for text in outcome.reasons)


def test_generality_is_refused_while_the_regions_independence_is_unestablished(
        catalogue, constellations, geography, grid, bridge):
    outcome = _run(catalogue, constellations, _partition("ABCE"), geography, grid, bridge,
                   decorrelation_km=None)
    assert outcome.verdict == VERDICT_REGIONAL
    assert any("independence of the held-out regions is unestablished" in text
               for text in outcome.reasons)


def test_generality_is_refused_when_no_region_declares_its_physiography(
        catalogue, constellations, geography, grid, bridge):
    """R14 asks for similar *and* dissimilar ground, and an undeclared surface answers neither."""
    outcome = _run(catalogue, constellations, _partition("ABCE", physiography=False), geography,
                   grid, bridge)
    assert outcome.verdict == VERDICT_REGIONAL
    assert any("declare no physiography" in text for text in outcome.reasons)


def test_generality_is_refused_when_every_assessed_region_is_of_one_physiography(
        catalogue, constellations, geography, grid, bridge):
    same = tuple(Region(name, BOXES[name][0], BOXES[name][1],
                        ROLE_DISCOVERY if name == "A" else ROLE_HELD_OUT, SAME)
                 for name in "ABCE")
    outcome = _run(catalogue, constellations,
                   RegionPartition(same, declared_by="a", declared_on="b"),
                   geography, grid, bridge)
    assert outcome.verdict == VERDICT_REGIONAL
    assert any("every assessed held-out region is of one declared physiography" in text
               for text in outcome.reasons)


def test_a_pattern_may_not_be_generalised_against_itself(catalogue, constellations, partition,
                                                         geography, grid, bridge):
    with pytest.raises(InvalidParameterError) as excinfo:
        _run(catalogue, constellations, partition, geography, grid, bridge,
             consequent=ANTECEDENT)
    assert "T4F.2 counts it under that name" in str(excinfo.value)


def test_the_run_refuses_a_partition_or_a_grid_of_the_wrong_type(catalogue, constellations,
                                                                 partition, geography, bridge,
                                                                 grid):
    with pytest.raises(InvalidParameterError):
        _run(catalogue, constellations, object(), geography, grid, bridge)
    with pytest.raises(InvalidParameterError) as excinfo:
        _run(catalogue, constellations, partition, geography, object(), bridge)
    assert "not recoverable from a catalogue" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 7: the receipt, and what it may not be read as
# ---------------------------------------------------------------------------------------------


def test_the_receipt_carries_the_design_it_was_reached_under(result, partition):
    record = describe_generalisation(result)
    assert record["schema"] == GENERALISATION_SCHEMA
    assert record["partition_digest"] == partition.digest
    assert record["verdict"] == VERDICT_REGIONAL
    assert [item["region"] for item in record["regions"]] == list("ABCDE")
    assert record["rule"]["antecedent"] == ANTECEDENT
    assert record["assignment"]["straddling_two_or_more_regions"] == 364


def test_the_receipt_reports_every_region_including_the_ones_that_could_not_be_assessed(result):
    record = describe_generalisation(result)
    outcomes = {item["region"]: item for item in record["regions"]}
    assert outcomes["E"]["outcome"] == REGION_UNASSESSABLE
    assert "support" not in outcomes["E"]
    assert outcomes["D"]["support"] == 0
    assert outcomes["B"]["physiography"] == "uniform-plane"


def test_the_claim_boundary_refuses_the_six_words_and_states_what_general_means(result):
    boundary = describe_generalisation(result)["claim_boundary"]
    assert boundary == GENERALISATION_CLAIM_BOUNDARY
    for word in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert word in boundary
    assert "not a claim that the rule holds anywhere it has not been tested" in boundary
    assert "licenses nothing at all" in boundary


def test_the_receipt_carries_the_precursor_boundary_it_inherits(result):
    record = describe_generalisation(result)
    assert record["inherited_claim_boundary"]
    assert record["inherited_claim_boundary"] != record["claim_boundary"]


def test_describing_something_that_is_not_a_verdict_is_refused():
    with pytest.raises(InvalidParameterError):
        describe_generalisation(object())



# ---------------------------------------------------------------------------------------------
# section 8: the six claims mutation testing found nothing was holding
# ---------------------------------------------------------------------------------------------


def test_the_matcher_takes_the_nearest_pattern_and_not_the_first(catalogue, signatures):
    """Rotation invariance puts many signatures inside both radii, so 'nearest' has to mean it."""
    fitted_only = PatternCatalogue(
        patterns=tuple(ConstellationPattern(
            pattern_id=item.pattern_id, members=item.members[:3], centroid=item.centroid,
            tolerance_radius=item.tolerance_radius, observed_radius=item.observed_radius)
            for item in catalogue),
        metric=catalogue.metric, tolerance=catalogue.tolerance, n_signatures=6)
    matched = match_into_catalogue(fitted_only, signatures)
    fitted = set(matched.fitted)
    centroids = {item.pattern_id: item.centroid for item in fitted_only}
    contested = 0
    for pattern in matched.catalogue:
        for member in pattern.members:
            if tuple(member.key) in fitted:
                continue
            point = SignaturePoint.from_signature(member)
            distances = {pattern_id: catalogue.metric.distance(point, centroid)
                         for pattern_id, centroid in centroids.items()
                         if centroid.family == point.family}
            assert distances[pattern.pattern_id] == min(distances.values())
            inside = [key for key, value in distances.items()
                      if value <= fitted_only.patterns[0].tolerance_radius]
            if len(inside) > 1:
                contested += 1
    assert contested > 0, "no signature lay inside two radii, so 'nearest' was never tested"


def test_the_matcher_never_puts_a_signature_into_a_pattern_of_another_family(catalogue,
                                                                            constellations):
    """A cardinality-3 configuration is not a candidate for a cardinality-2 pattern."""
    triples = sign_constellations(
        extract_constellations(_record_tracked(), cardinalities=(3,)),
        scale_invariant=False).signatures
    assert triples, "the record carries no triple to test the family guard with"
    assert {item.cardinality for item in triples} == {3}
    matched = match_into_catalogue(catalogue, list(triples))
    assert matched.n_matched == 0
    assert matched.n_unmatched == len(triples)
    for pattern in matched.catalogue:
        assert all(member.cardinality == 2 for member in pattern.members)


def test_a_pixel_grid_cannot_establish_independence_however_it_is_declared(
        record, catalogue, constellations, partition, geography, grid):
    from src.transform_engine.coefficient_field import CoefficientField
    field = record[1]
    pixel = CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times,
        grid=GridSpec.pixel(field.grid.shape), config=dict(field.config))
    bare = physical_bridge(geography_of(pixel, time_units="frames", values=record[0]),
                           variable="amplitude")
    outcome = _run(catalogue, constellations, partition, geography, grid, bare,
                   decorrelation_km=400.0)
    assert outcome.independence["status"] == "UNESTABLISHED"
    assert "this grid declares no length" in outcome.independence["note"]


def test_closeness_is_judged_on_the_widest_kilometre_reading_the_grid_supports(
        record, catalogue, constellations, partition, geography, grid):
    """On a lat/lon crop a gap in cells is a range of kilometres, and the two ends disagree."""
    from src.transform_engine.coefficient_field import CoefficientField
    field = record[1]
    spherical = GridSpec.latlon(field.grid.shape, lat0=-20.0, dlat=-40.0 / ROWS,
                                lon0=140.0, dlon=40.0 / ROWS)
    regridded = CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times, grid=spherical,
        config=dict(field.config))
    curved = physical_bridge(geography_of(regridded, time_units="frames", values=record[0]),
                             variable="amplitude")
    gap = [item for item in partition.gaps(curved)
           if item["regions"] == ["A", "B"]][0]["gap_km"]
    assert gap["low"] < 400.0 < gap["high"], (gap, "the two readings must disagree at 400 km")
    outcome = _run(catalogue, constellations, partition, geography, grid, curved,
                   decorrelation_km=400.0)
    named = outcome.independence[
        "closer_to_the_discovery_region_than_the_declared_decorrelation_length"]
    assert "B" not in named


def test_a_bogus_grid_is_refused_even_when_no_region_carries_anything(catalogue,
                                                                     constellations,
                                                                     geography, bridge):
    """The only path on which nothing downstream would ever look at the grid."""
    empty = RegionPartition(
        (Region("E", (72, 112), (72, 112), ROLE_DISCOVERY, SAME),
         Region("F", (72, 112), (128, 168), ROLE_HELD_OUT, SAME)),
        declared_by="the T4F.7 suite", declared_on="2026-09-07")
    assignment = assign_regions(constellations, empty, geography)
    assert assignment.as_record()["inside_each_region"] == {"E": 0, "F": 0}
    with pytest.raises(InvalidParameterError) as excinfo:
        _run(catalogue, constellations, empty, geography, object(), bridge)
    assert "not recoverable from a catalogue" in str(excinfo.value)
