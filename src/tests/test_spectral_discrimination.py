"""T4E.6: a radius has two error rates, and this programme measured one of them.

1.  **The arithmetic is tested on distance populations, not through the pipeline.** What is
    under test is what a radius does to two populations, and the honest unit for that is two
    lists of distances. Running them through tracking and signing first would test the pipeline
    and leave the arithmetic inferred.

2.  **The two rates must move in opposite directions, always.** Widening a radius can only split
    fewer same-configuration pairs and admit more different ones. The suite asserts both are
    monotone across a full sweep, because a rate that moved the other way would mean the
    populations had been swapped.

3.  **`None` is the answer that matters.** `best_operating_point` returns no radius when none
    holds both rates at the asked-for level. That is a fact about the signature rather than
    about the search, and a caller must be able to discover it before building an identity on it.

4.  **An unmeasured admission rate is recorded as a refusal, never as a silence.** A tolerance
    calibrated without a contrast population carries `ADMISSION_RATE_NOT_MEASURED` and no
    number, so a reader cannot mistake "not measured" for "measured and small". This is the half
    of D97 this task closes.

5.  **D97's phenomenon is reproduced here.** On this repository's own tracked fixture the two
    populations overlap so heavily that no radius achieves both rates at 20%, and the suite pins
    that rather than hiding it. That is the measurement T4E.7 exists to act on.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from src.analysis_engine.spectral_clustering import (
    DISCRIMINATION_MEASURED, DISCRIMINATION_NOTE, DISCRIMINATION_UNMEASURED, AttributeWeights,
    SignatureFamily, SignatureMetric, SignaturePoint, best_operating_point,
    calibrate_signature_tolerance, discrimination_curve, measure_discrimination,
)
from src.core.errors import InvalidParameterError
from src.tests.test_spectral_identity import _record

METRIC = SignatureMetric(AttributeWeights(1.0, 1.0, 1.0, 1.0))

#: Two populations that genuinely separate: everything close is below 0.2, everything far is
#: above 0.6, and nothing lies between. A radius exists here and the suite must find it.
TIGHT_SAME = [0.01, 0.02, 0.04, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.19]
TIGHT_DIFFERENT = [0.61, 0.66, 0.70, 0.74, 0.79, 0.83, 0.88, 0.92, 0.97, 1.02]

#: Two populations that do not separate: they are drawn over the same span. No radius holds both
#: rates low, and the honest answer is that there is none.
OVERLAP_SAME = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
OVERLAP_DIFFERENT = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


# ---------------------------------------------------------------------------------------------
# section 1: what a radius does to two populations
# ---------------------------------------------------------------------------------------------


def test_both_rates_are_counted_against_their_own_population():
    result = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.4)
    assert result.status == DISCRIMINATION_MEASURED
    assert result.measured
    assert result.false_split_rate == 0.0        # every same pair is below 0.4
    assert result.admission_rate == 0.0          # every different pair is above it
    assert result.n_same_pairs == 10 and result.n_different_pairs == 10
    tight = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.1)
    assert tight.false_split_rate == pytest.approx(0.4)   # 0.11, 0.13, 0.16, 0.19 are above
    assert tight.admission_rate == 0.0
    wide = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.8)
    assert wide.false_split_rate == 0.0
    assert wide.admission_rate == pytest.approx(0.5)


def test_the_rates_move_in_opposite_directions_across_the_whole_sweep():
    """Widening a radius can only split fewer and admit more. A rate that moved the other way
    would mean the two populations had been exchanged."""
    curve = discrimination_curve(OVERLAP_SAME, OVERLAP_DIFFERENT)
    assert len(curve) > 4
    radii = [row.radius for row in curve]
    assert radii == sorted(radii)
    splits = [row.false_split_rate for row in curve]
    admits = [row.admission_rate for row in curve]
    assert splits == sorted(splits, reverse=True), "the split rate must not rise with the radius"
    assert admits == sorted(admits), "the admission rate must not fall with the radius"
    assert splits[0] > splits[-1] and admits[0] < admits[-1], "the sweep moved nothing"


def test_a_radius_of_zero_splits_everything_and_a_wide_one_admits_everything():
    tightest = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.0)
    assert tightest.false_split_rate == 1.0 and tightest.admission_rate == 0.0
    widest = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 99.0)
    assert widest.false_split_rate == 0.0 and widest.admission_rate == 1.0


def test_the_separation_is_reported_and_is_larger_when_the_populations_are_apart():
    apart = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.4)
    together = measure_discrimination(OVERLAP_SAME, OVERLAP_DIFFERENT, 0.4)
    assert apart.separation > together.separation
    assert apart.separation > 2.0
    assert abs(together.separation) < 0.5


# ---------------------------------------------------------------------------------------------
# section 2: the answer that matters is None
# ---------------------------------------------------------------------------------------------


def test_an_operating_point_is_found_where_the_populations_separate():
    best = best_operating_point(TIGHT_SAME, TIGHT_DIFFERENT, 0.05)
    assert best is not None
    assert best.false_split_rate <= 0.05 and best.admission_rate <= 0.05
    assert 0.19 <= best.radius < 0.61, "the radius must land in the gap between the populations"


def test_no_operating_point_exists_where_they_do_not_separate():
    """The refusal is the point: no radius holds both rates low, and saying so is the finding."""
    for target in (0.05, 0.10, 0.20):
        assert best_operating_point(OVERLAP_SAME, OVERLAP_DIFFERENT, target) is None


def test_the_smallest_sufficient_radius_is_the_one_returned():
    best = best_operating_point(TIGHT_SAME, TIGHT_DIFFERENT, 0.05)
    tighter = [row for row in discrimination_curve(TIGHT_SAME, TIGHT_DIFFERENT)
               if row.radius < best.radius]
    for row in tighter:
        assert row.false_split_rate > 0.05 or row.admission_rate > 0.05


#: A population where several radii qualify, so "the smallest" is a real choice rather than the
#: only option. Ten same-distances mean a 20% target tolerates two above the radius.
MANY_SAME = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
MANY_DIFFERENT = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def test_the_first_qualifying_radius_is_returned_where_several_qualify():
    """With one qualifying radius, first and last are the same answer and prove nothing."""
    qualifying = [row for row in discrimination_curve(MANY_SAME, MANY_DIFFERENT)
                  if row.false_split_rate <= 0.20 and row.admission_rate <= 0.20]
    assert len(qualifying) >= 3, "this population must offer a choice of radii"
    best = best_operating_point(MANY_SAME, MANY_DIFFERENT, 0.20)
    assert best.radius == min(row.radius for row in qualifying)
    assert best.radius < max(row.radius for row in qualifying)


def test_a_distance_exactly_at_the_radius_is_admitted_and_not_split():
    """The boundary belongs to the radius on both sides: a pair exactly at it is inside."""
    exact = measure_discrimination([0.30], [0.30], 0.30)
    assert exact.false_split_rate == 0.0, "a same pair at the radius is not split"
    assert exact.admission_rate == 1.0, "a different pair at the radius is admitted"
    just_outside = measure_discrimination([0.3000001], [0.3000001], 0.30)
    assert just_outside.false_split_rate == 1.0
    assert just_outside.admission_rate == 0.0


def test_an_operating_point_needs_a_contrast_population_to_exist_at_all():
    assert best_operating_point(TIGHT_SAME, [], 0.05) is None


# ---------------------------------------------------------------------------------------------
# section 3: an unmeasured rate is a refusal, not a silence
# ---------------------------------------------------------------------------------------------


def test_no_contrast_gives_a_named_refusal_and_no_number():
    result = measure_discrimination(TIGHT_SAME, [], 0.4)
    assert result.status == DISCRIMINATION_UNMEASURED
    assert not result.measured
    assert result.admission_rate is None
    assert result.separation is None
    assert result.false_split_rate == 0.0, "the rate that *can* be measured still is"
    assert "not the same as admitting nothing" in result.note


def test_the_record_omits_the_figures_it_did_not_measure():
    """A key present with a null invites a reader to treat it as zero; the key is absent."""
    unmeasured = measure_discrimination(TIGHT_SAME, [], 0.4).as_record()
    assert "admission_rate" not in unmeasured
    assert "different_configuration_distances" not in unmeasured
    assert "standardised_separation" not in unmeasured
    assert unmeasured["status"] == DISCRIMINATION_UNMEASURED
    assert unmeasured["basis"] == DISCRIMINATION_NOTE
    measured = measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, 0.4).as_record()
    assert measured["admission_rate"] == 0.0
    assert measured["standardised_separation"] > 2.0


def test_the_basis_says_why_replicates_cannot_produce_the_second_rate():
    assert "replicates contain no example of two different things" in DISCRIMINATION_NOTE
    assert "decides whether a pattern means anything" in DISCRIMINATION_NOTE


# ---------------------------------------------------------------------------------------------
# section 4: the tolerance carries it
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signatures():
    return _record()


@pytest.fixture(scope="module")
def other_family():
    """The same record enumerated at cardinality three: a real signature of another family.

    The scale-invariant mode would also be another family, but on this record it signs nothing
    at all -- T4E.2's universality hook drops configurations whose members do not share a unit,
    and here that is all of them. A triple is the family difference this record can actually
    supply.
    """
    from src.analysis_engine.spectral_constellation import extract_constellations
    from src.analysis_engine.spectral_invariance import sign_constellations
    from src.tests.test_spectral_identity import _CACHE, _record
    _record()
    triples = extract_constellations(_CACHE["tracked"], cardinalities=(3,),
                                     max_nodes_per_frame=32)
    return list(sign_constellations(triples, scale_invariant=False).signatures)


@pytest.fixture(scope="module")
def replicates_and_contrast(signatures):
    runs = defaultdict(list)
    for item in signatures:
        runs[tuple(sorted(item.track_ids))].append(item)
    longest = max(runs.values(), key=len)
    key = tuple(sorted(longest[0].track_ids))
    contrast = [group[0] for other, group in runs.items() if other != key][:120]
    assert len(longest) >= 3 and len(contrast) >= 20
    return sorted(longest, key=lambda s: s.time), contrast


def test_a_tolerance_calibrated_without_contrast_records_that_it_was(replicates_and_contrast):
    replicates, _contrast = replicates_and_contrast
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC)
    assert tolerance.discrimination.status == DISCRIMINATION_UNMEASURED
    assert tolerance.discrimination.admission_rate is None
    record = tolerance.describe()["discrimination"]
    assert record["status"] == DISCRIMINATION_UNMEASURED
    assert "admission_rate" not in record


def test_a_tolerance_calibrated_with_contrast_carries_both_rates(replicates_and_contrast):
    replicates, contrast = replicates_and_contrast
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC, contrast=contrast)
    found = tolerance.discrimination
    assert found.status == DISCRIMINATION_MEASURED
    assert found.radius == tolerance.value
    assert 0.0 <= found.false_split_rate <= 1.0
    assert 0.0 <= found.admission_rate <= 1.0
    assert found.n_different_pairs == len(replicates) * len(contrast)
    assert tolerance.describe()["discrimination"]["admission_rate"] == found.admission_rate


def test_the_radius_itself_is_unchanged_by_measuring_what_it_admits(replicates_and_contrast):
    """T4E.6 measures the radius; it does not move it. Moving it is T4E.7's question."""
    replicates, contrast = replicates_and_contrast
    without = calibrate_signature_tolerance(replicates, metric=METRIC)
    with_ = calibrate_signature_tolerance(replicates, metric=METRIC, contrast=contrast)
    assert without.value == with_.value
    assert without.pair_distances == with_.pair_distances


def test_a_contrast_of_another_family_is_refused_rather_than_measured(replicates_and_contrast,
                                                                      other_family):
    """A distance between different measured quantities is not a number, so it is not a
    contrast either. The other family here is real: the same configurations signed in the
    scale-invariant mode, whose scale units are dimensionless rather than cells."""
    replicates, _contrast = replicates_and_contrast
    assert other_family[0].cardinality == 3 and replicates[0].cardinality == 2
    with pytest.raises(InvalidParameterError) as excinfo:
        calibrate_signature_tolerance(replicates, metric=METRIC, contrast=other_family[:5])
    assert "cannot be a contrast" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 5: what this repository's own record actually does
# ---------------------------------------------------------------------------------------------


def test_this_records_signatures_do_not_separate_and_the_suite_says_so(replicates_and_contrast):
    """D97's phenomenon, reproduced on the fixture rather than only on the acquired record.

    This is not a failure of the measurement -- it is the measurement working. The populations
    overlap, no radius holds both rates at a fifth, and that is the number T4E.7 exists to act
    on. If a later change to the signature makes this test fail, the signature improved and the
    assertion should be tightened rather than removed.
    """
    replicates, contrast = replicates_and_contrast
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC, contrast=contrast)
    found = tolerance.discrimination
    assert found.admission_rate > 0.2, (
        "the calibrated radius admits %.1f%% of contrast pairs" % (100 * found.admission_rate))
    assert found.separation < 1.5
    same = list(tolerance.pair_distances)
    different = _cross_distances(replicates, contrast)
    for target in (0.05, 0.20):
        assert best_operating_point(same, different, target) is None


def test_the_split_rate_at_the_calibrated_radius_is_zero_by_construction(
        replicates_and_contrast):
    """The radius *is* the largest replicate distance, so no replicate pair can exceed it.

    The measured false-split rate at the calibrated radius is therefore identically zero on any
    input whatever, and it says nothing about the instrument. The `1/(pairs+1)` figure the
    calibration quotes is an assumed rate for *future* comparisons, not something measured here.
    This is the second limb of D97 and it is pinned so that a future calibration which reports a
    real, non-tautological split rate will break this test and have to replace it.
    """
    replicates, contrast = replicates_and_contrast
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC, contrast=contrast)
    assert tolerance.discrimination.false_split_rate == 0.0
    assert tolerance.value == max(tolerance.pair_distances)
    # True for any subset too, because the property is arithmetic and not a fact about data.
    for size in (3, 4, 5):
        if len(replicates) < size:
            continue
        subset = calibrate_signature_tolerance(replicates[:size], metric=METRIC,
                                               contrast=contrast)
        assert subset.discrimination.false_split_rate == 0.0


def test_the_radius_depends_on_which_configuration_supplied_the_replicates(signatures):
    """Same instrument, same record, same metric -- and a materially different answer.

    Measured on the acquired ERA5 record the calibrated radius spans a factor of 6.7 across 25
    configurations, with the admission rate running from 8.9% to 84.5%. This asserts the same
    instability on the fixture, because a radius that is a property of which replicate sample
    was reached for is not a property of the instrument.
    """
    runs = defaultdict(list)
    for item in signatures:
        runs[tuple(sorted(item.track_ids))].append(item)
    usable = sorted((sorted(v, key=lambda s: s.time) for v in runs.values() if len(v) >= 4),
                    key=len, reverse=True)[:12]
    assert len(usable) >= 6, "the fixture must supply several repeated configurations"
    radii = [calibrate_signature_tolerance(run, metric=METRIC).value for run in usable]
    assert max(radii) / min(radii) > 2.0, (
        "the calibrated radius should vary with the replicate sample; got %s"
        % [round(value, 4) for value in radii])


def test_the_curve_is_what_makes_a_later_signature_judgeable(replicates_and_contrast):
    replicates, contrast = replicates_and_contrast
    same = list(calibrate_signature_tolerance(replicates, metric=METRIC).pair_distances)
    curve = discrimination_curve(same, _cross_distances(replicates, contrast))
    assert all(row.measured for row in curve)
    best = min(curve, key=lambda row: max(row.false_split_rate, row.admission_rate))
    assert max(best.false_split_rate, best.admission_rate) > 0.2, (
        "the best achievable operating point on this record is %.1f%%"
        % (100 * max(best.false_split_rate, best.admission_rate)))


# ---------------------------------------------------------------------------------------------
# section 6: refusals
# ---------------------------------------------------------------------------------------------


def test_the_things_that_are_not_measurable_are_refused():
    with pytest.raises(InvalidParameterError):
        measure_discrimination([], TIGHT_DIFFERENT, 0.4)
    with pytest.raises(InvalidParameterError):
        measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, -1.0)
    with pytest.raises(InvalidParameterError):
        measure_discrimination(TIGHT_SAME, TIGHT_DIFFERENT, float("nan"))
    for target in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(InvalidParameterError):
            best_operating_point(TIGHT_SAME, TIGHT_DIFFERENT, target)
    with pytest.raises(InvalidParameterError):
        discrimination_curve([], [])


# ---------------------------------------------------------------------------------------------


def _cross_distances(replicates, contrast):
    from src.analysis_engine.spectral_clustering import _points
    left = _points(replicates)
    right = _points(contrast)
    return [METRIC.distance(a, b) for a in left for b in right]


