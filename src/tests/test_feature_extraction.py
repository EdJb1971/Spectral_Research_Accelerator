"""TG2.2: feature extraction as a registry, and the first extractor in it.

Four groups, in the order the slice has to be believed:

1.  **It finds what was planted.** `planted_configuration` records three features, their
    positions, their width and their separation, and it recorded them before this module
    existed. The extractor is run against that answer across a six-fold range of scales and
    under rotation, translation and rescaling.
2.  **It finds nothing where there is nothing.** The two null benchmarks are the reason the
    threshold is calibrated rather than chosen. An extractor that reports a feature on white
    noise or on fBm is not a detector, and neither is one that reports nothing because its
    threshold happened to be high.
3.  **Extraction is a registry.** A second extractor is registered from this module and
    drives `extract` without an edit to `src`, and the parts the framework keeps -
    significance, representation, the R19 refusals - arrive on its records anyway.
4.  **The declaration changes the arithmetic.** The same array, declared periodic and
    declared not, is two different problems, and the extractor answers both correctly rather
    than answering one of them twice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmarks.core import get_benchmark
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.extraction import (
    DEFAULT_SURROGATE_METHOD, EXTRACTORS, Candidate, ExtractionField, ExtractorReport,
    NullCalibration, calibrate, extract,
)
from src.core.feature import FeatureSet, SemanticComparisonError, SpectralFeature
from src.core.registry import restore, snapshot

ROW = AxisSpec("row", "space", units="cells", ordinal=0)
COL = AxisSpec("col", "space", units="cells", ordinal=1)
N_SURROGATES = 99          # enough to resolve alpha = 0.05; small enough to run in CI
SEED = 1234


def _field(values, axes=(ROW, COL), **kwargs):
    params = dict(
        domain="synthetic", dataset="planted_configuration", variable="amplitude",
        units=None, time=0.0, representation="identity", time_units="frames")
    params.update(kwargs)
    return ExtractionField(values=np.asarray(values, dtype=np.float64), axes=axes, **params)


def _planted(**overrides):
    """The benchmark's field and its recorded answer - the answer is never fed forward."""
    bench = get_benchmark("planted_configuration")
    data = np.asarray(bench.make(**overrides).data.numpy(), dtype=np.float64)
    return _field(data), bench.truth(**overrides)


def _position_errors(result, truth):
    """Distance from each reported feature to the nearest recorded position, in cells."""
    return [min(math.hypot(f.location.coords["row"] - ty, f.location.coords["col"] - tx)
                for ty, tx in truth["positions_rowcol"]) for f in result]


def _run(field, **kwargs):
    kwargs.setdefault("n_surrogates", N_SURROGATES)
    kwargs.setdefault("seed", SEED)
    return extract(field, **kwargs)


# ============================================================ 1. it finds what was planted


def test_the_extractor_finds_exactly_the_three_features_the_benchmark_planted():
    field, truth = _planted()
    result = _run(field)
    assert result.found == truth["feature_count"] == 3
    assert max(_position_errors(result, truth)) < 1.0


@pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0, 3.0])
def test_the_count_holds_across_a_sixfold_range_of_feature_scales(scale_factor):
    """The number of features must not be a function of how wide they are.

    This is the test that killed a fixed suppression radius. Tuned at `scale_factor=1.0`, a
    fixed radius reports eight features at `scale_factor=2.0` - three planted and five noise
    maxima on their shoulders, each above the calibrated threshold and each indistinguishable
    in a receipt from a discovery. Suppressing at a multiple of the feature's *own* measured
    scale removes the parameter that was doing the inventing.
    """
    field, truth = _planted(scale_factor=scale_factor)
    result = _run(field)
    assert result.found == 3
    assert max(_position_errors(result, truth)) < 1.0


@pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0, 3.0])
def test_the_measured_scale_recovers_the_planted_width_across_that_range(scale_factor):
    field, truth = _planted(scale_factor=scale_factor)
    result = _run(field)
    expected = truth["feature_sigma_cells"]
    for feature in result:
        assert feature.spatial_scale.units == "cells"
        assert abs(feature.spatial_scale.value / expected - 1.0) < 0.06


def test_the_measured_separations_recover_the_planted_triangle():
    """The pairwise distance is what a constellation matcher will key on in phase G3."""
    field, truth = _planted()
    result = _run(field)
    features = list(result)
    distances = [features[i].separation_to(features[j]).value
                 for i in range(3) for j in range(i + 1, 3)]
    for d in distances:
        assert abs(d - truth["pairwise_distance_cells"]) < 1.0


@pytest.mark.parametrize("overrides", [
    {"rotation_deg": 37.0},
    {"translation": (25.0, -30.0)},
    {"rotation_deg": 17.0, "translation": (-20.0, 40.0), "scale_factor": 1.5},
])
def test_the_geometry_survives_rotation_translation_and_rescaling(overrides):
    """Not the 4E invariance gate - that is TG3.4 and needs a matcher.

    What is checked here is the prerequisite nobody had checked: that the *extractor* returns
    the same configuration when the same configuration is presented transformed. A matcher
    tested on features from an extractor that loses one under rotation would be measuring the
    extractor.
    """
    field, truth = _planted(**overrides)
    result = _run(field)
    assert result.found == 3
    features = list(result)
    distances = [features[i].separation_to(features[j]).value
                 for i in range(3) for j in range(i + 1, 3)]
    for d in distances:
        assert abs(d - truth["pairwise_distance_cells"]) < 1.0


def test_the_reported_magnitude_is_the_amplitude_and_not_the_brightest_sample():
    """The brightest cell overstates a unit-amplitude feature, increasingly as it broadens.

    At `scale_factor=3.0` the peak sample of a feature of amplitude 1 reads about 1.09,
    because it is the largest of several hundred noisy samples across a nearly flat top.
    Carried into the integral estimator that becomes a scale biased low, in a direction that
    does not cancel in a ratio because it is a function of the feature's own size. The
    de-biased disc mean is kept as the magnitude and the raw sample is kept in provenance,
    so the correction is visible rather than merely applied.
    """
    field, _ = _planted(scale_factor=3.0)
    result = _run(field)
    for feature in result:
        sample = feature.provenance["peak_sample"]
        assert abs(feature.magnitude.value - 1.0) < 0.04
        assert sample > feature.magnitude.value
        assert sample - 1.0 > 0.05


def test_every_feature_carries_the_surrogate_p_value_and_its_ensemble_size():
    field, _ = _planted()
    result = _run(field)
    for feature in result:
        assert feature.significance.basis == "surrogate_p_value"
        assert feature.significance.n_surrogates == N_SURROGATES
        assert feature.significance.value >= feature.significance.resolution_floor
        assert feature.significance.value <= result.calibration.alpha


def test_the_result_is_a_homogeneous_feature_set_the_tracker_can_consume():
    field, _ = _planted()
    result = _run(field)
    fs = result.feature_set()
    assert isinstance(fs, FeatureSet)
    assert fs.representation == "identity"
    assert len(fs.frames()) == 1        # one frame, three features at one time


# ======================================================= 2. nothing where there is nothing


@pytest.mark.parametrize("name", ["white_noise_field", "fractional_brownian"])
@pytest.mark.parametrize("root_seed", [20260819, 7, 99])
def test_the_null_benchmarks_yield_no_features(name, root_seed):
    """The false-positive floor. These two benchmarks' known answer is 'nothing here'."""
    bench = get_benchmark(name)
    data = np.asarray(bench.make(root_seed=root_seed).data.numpy(), dtype=np.float64)
    result = _run(_field(data, dataset=name), seed=root_seed)
    assert result.found == 0
    assert result.is_empty


def test_finding_nothing_is_a_result_with_a_receipt_rather_than_an_empty_list():
    """`FeatureSet` refuses to be empty; this is the object its docstring points at.

    "No features" and "four thousand local maxima, none of which cleared the calibrated cut"
    are different statements about a field, and TG2.4's audit is a question about the second.
    """
    bench = get_benchmark("white_noise_field")
    data = np.asarray(bench.make().data.numpy(), dtype=np.float64)
    result = _run(_field(data, dataset="white_noise_field"))
    assert result.is_empty
    assert result.rejected["below_threshold"] > 100
    described = result.describe()
    assert described["found"] == 0
    assert described["calibration"]["n_surrogates"] == N_SURROGATES
    assert described["field"]["dataset"] == "white_noise_field"
    with pytest.raises(InvalidParameterError) as exc:
        result.feature_set()
    assert "found none" in str(exc.value)


def test_the_null_field_still_contains_maxima_so_the_silence_is_the_threshold_working():
    """A detector that reports nothing because it looked at nothing is not a detector."""
    bench = get_benchmark("white_noise_field")
    data = np.asarray(bench.make().data.numpy(), dtype=np.float64)
    result = _run(_field(data, dataset="white_noise_field"))
    assert result.notes["raw_local_maxima"] > 1000
    assert result.found == 0


def test_dropping_the_threshold_to_the_floor_turns_the_null_field_into_findings():
    """The same field, the same extractor, an alpha the ensemble can barely resolve.

    This is the control the null tests need: the silence above is a property of the
    calibration, not of an extractor that cannot find anything. It is also the shape of the
    mistake rule R3 exists to prevent - the number that decides what is real was chosen.
    """
    bench = get_benchmark("white_noise_field")
    data = np.asarray(bench.make().data.numpy(), dtype=np.float64)
    field = _field(data, dataset="white_noise_field")
    strict = _run(field)
    loose = _run(field, alpha=0.5)
    assert strict.found == 0
    assert loose.calibration.threshold < strict.calibration.threshold
    assert loose.found > strict.found


def test_an_alpha_finer_than_the_ensemble_can_resolve_is_refused_before_the_run():
    """99 surrogates cannot resolve p < 0.01, so a cut at 0.001 is unreachable.

    Without this the extractor returns nothing and the receipt says the field was empty -
    the most expensive failure mode available, because it is indistinguishable from a
    genuine null and it is silent.
    """
    with pytest.raises(InvalidParameterError) as exc:
        calibrate(np.zeros((16, 16)), alpha=0.001, n_surrogates=99)
    message = str(exc.value)
    assert "resolve" in message
    assert "999" in message          # names the ensemble size the level actually needs


def test_the_threshold_is_an_order_statistic_of_the_ensemble_not_an_interpolated_quantile():
    """The cut must be a value some observation could actually achieve `p <= alpha` at."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal((48, 48))
    cal = calibrate(data, alpha=0.05, n_surrogates=99, seed=SEED)
    assert cal.threshold in cal.null_maxima
    assert cal.p_value(cal.threshold * 1.5).value == pytest.approx(cal.resolution_floor)


def test_the_comparison_against_the_threshold_is_strict_because_a_tie_is_not_an_exceedance():
    """The off-by-one this test exists to hold shut.

    `k` counts null maxima *at least as large as* the observation, so an observation sitting
    exactly on the threshold ties with it and reports `p` one step above alpha. Admitting it
    with `>=` would report features at a level the receipt does not claim - 0.06 under a
    heading that says 0.05.
    """
    rng = np.random.default_rng(5)
    cal = calibrate(rng.standard_normal((48, 48)), alpha=0.05, n_surrogates=99, seed=SEED)
    assert not cal.clears(cal.threshold)
    assert cal.p_value(cal.threshold).value > cal.alpha
    just_above = float(np.nextafter(cal.threshold, np.inf))
    assert cal.clears(just_above)
    assert cal.p_value(just_above).value == pytest.approx(cal.alpha)


# ============================================================ 3. extraction is a registry


def test_the_first_extractor_is_the_first_and_declares_what_it_assumes():
    entry = EXTRACTORS.entry("local_maximum")
    assert entry.capabilities["dimensions"] == 2
    assert entry.capabilities["shape_model"] == "isotropic_gaussian_on_a_flat_baseline"
    assert entry.capabilities["honours_periodic_axes"] is True
    assert EXTRACTORS.with_capability("sub_cell")


def test_an_unregistered_extractor_raises_with_a_did_you_mean():
    field, _ = _planted()
    with pytest.raises(UnknownNameError) as exc:
        _run(field, extractor="local_maxima")
    assert "local_maximum" in str(exc.value)


def test_a_second_extractor_drives_the_pipeline_without_editing_src():
    """The whole point of TG2.2. `local_maximum` is one opinion about where features are.

    This one is a different opinion - the single brightest cell, no sub-cell localisation,
    no scale - and it produces valid records through the same entry point. Nothing in
    `src/core/extraction.py` knows it exists.
    """
    state = snapshot(EXTRACTORS)
    try:
        @EXTRACTORS.register(
            "brightest_cell", description="The single brightest cell above the cut.",
            capabilities={"dimensions": 2, "needs_calibration": True, "sub_cell": False})
        def brightest(field, calibration, **params):
            flat = int(np.argmax(field.values))
            i, j = divmod(flat, field.values.shape[1])
            peak = float(field.values[i, j])
            if peak < calibration.threshold:
                return ExtractorReport(rejected={"below_threshold": 1})
            return ExtractorReport(candidates=(Candidate(
                coords={"row": float(i), "col": float(j)},
                magnitude=peak, peak_sample=peak),))

        field, truth = _planted()
        result = _run(field, extractor="brightest_cell")
        assert result.found == 1
        assert result.extractor == "brightest_cell"
        assert min(_position_errors(result, truth)) < 1.5
        assert result.describe()["extractor_capabilities"]["sub_cell"] is False
    finally:
        restore(EXTRACTORS, state)


def test_the_framework_and_not_the_extractor_attaches_significance_and_representation():
    """A registered extractor cannot forget the three things TG2.1 exists to enforce.

    `Candidate` carries no significance, no representation and no domain, so an extractor
    has no way to declare its own. That is why the record-building lives in `extract` and
    not behind the registry: the parts that must not vary are the parts a plug-in would vary.
    """
    state = snapshot(EXTRACTORS)
    try:
        @EXTRACTORS.register("bare", capabilities={"dimensions": 2})
        def bare(field, calibration, **params):
            return ExtractorReport(candidates=(Candidate(
                coords={"row": 10.0, "col": 20.0}, magnitude=5.0, peak_sample=5.0),))

        field, _ = _planted()
        feature = _run(field, extractor="bare")[0]
        assert isinstance(feature, SpectralFeature)
        assert feature.representation == "identity"
        assert feature.domain == "synthetic"
        assert feature.significance.basis == "surrogate_p_value"
        assert feature.provenance["extractor"] == "bare"
        assert feature.provenance["surrogate_method"] == DEFAULT_SURROGATE_METHOD
    finally:
        restore(EXTRACTORS, state)


def test_an_extractor_returning_a_coordinate_the_axes_do_not_declare_is_refused():
    state = snapshot(EXTRACTORS)
    try:
        @EXTRACTORS.register("wrong_axes", capabilities={"dimensions": 2})
        def wrong(field, calibration, **params):
            return ExtractorReport(candidates=(Candidate(
                coords={"x": 10.0, "y": 20.0}, magnitude=5.0, peak_sample=5.0),))

        field, _ = _planted()
        with pytest.raises(InvalidParameterError) as exc:
            _run(field, extractor="wrong_axes")
        assert "row" in str(exc.value) and "col" in str(exc.value)
    finally:
        restore(EXTRACTORS, state)


def test_an_extractor_that_does_not_return_a_report_is_refused():
    state = snapshot(EXTRACTORS)
    try:
        @EXTRACTORS.register("sloppy", capabilities={"dimensions": 2})
        def sloppy(field, calibration, **params):
            return [(10.0, 20.0)]

        field, _ = _planted()
        with pytest.raises(InvalidParameterError):
            _run(field, extractor="sloppy")
    finally:
        restore(EXTRACTORS, state)


def test_a_candidate_is_not_a_feature_and_cannot_pretend_to_be_one():
    """`Candidate` has no domain, so no R19 refusal can be evaded by constructing one."""
    c = Candidate(coords={"row": 1.0, "col": 2.0}, magnitude=1.0, peak_sample=1.1)
    assert not hasattr(c, "domain")
    assert not hasattr(c, "significance")


def test_an_extractor_declaring_two_dimensions_refuses_a_three_dimensional_field():
    axes = (AxisSpec("level", "level", units="index"), ROW, COL)
    field = _field(np.zeros((4, 16, 16)), axes=axes)
    with pytest.raises(InvalidParameterError) as exc:
        _run(field)
    assert "dimensions=2" in str(exc.value)


# ================================================ 4. the declaration changes the arithmetic


def _seam_blob(centre_col, periodic, n=128, sigma=5.0, noise=0.05, seed=3):
    """One Gaussian on a torus. The *data* wraps; whether the reader knows is the test."""
    idx = np.arange(n, dtype=np.float64)
    d0 = idx[:, None] - 64.0
    d1 = idx[None, :] - centre_col
    d0 = (d0 + n / 2) % n - n / 2
    d1 = (d1 + n / 2) % n - n / 2
    blob = np.exp(-(d0 ** 2 + d1 ** 2) / (2 * sigma ** 2))
    return blob + noise * np.random.default_rng(seed).standard_normal((n, n))


PERIODIC_AXES = (AxisSpec("row", "space", units="cells", periodic=True, ordinal=0),
                 AxisSpec("col", "space", units="cells", periodic=True, ordinal=1))


@pytest.mark.parametrize("centre_col", [64.0, 2.0, 126.5])
def test_a_periodic_axis_finds_the_seam_feature_once_and_in_the_right_place(centre_col):
    """The reason `periodic` is on `AxisSpec` and not in a comment.

    The neighbourhood comparison, the localisation window and the reported coordinate all
    wrap together. Averaging wrapped indices instead is how a feature on the seam acquires a
    position in the middle of the frame; averaging unwrapped ones and forgetting to wrap back
    is how it acquires a position off the end of it.
    """
    data = _seam_blob(centre_col, periodic=True)
    result = _run(_field(data, axes=PERIODIC_AXES, dataset="seam"), seed=11)
    assert result.found == 1
    got = result[0].location.coords["col"]
    offset = abs((got - centre_col + 64.0) % 128.0 - 64.0)
    assert offset < 1.0
    assert abs(result[0].spatial_scale.value - 5.0) < 0.5


def test_the_same_array_declared_non_periodic_refuses_rather_than_answering_wrongly():
    """R13, sized by the feature rather than by a fixed margin.

    Declared non-periodic, the seam feature has half its integral outside the frame. The
    truncation correction then applies for the wrong reason and returns a position three
    cells out and a scale a quarter too small, with nothing in the record to say so. Refused
    and counted instead - a missing feature is a fact a receipt can carry, and a confidently
    mismeasured one is not.
    """
    data = _seam_blob(2.0, periodic=True)
    periodic = _run(_field(data, axes=PERIODIC_AXES, dataset="seam"), seed=11)
    plain = _run(_field(data, axes=(ROW, COL), dataset="seam"), seed=11)
    assert periodic.found == 1
    assert plain.found == 0
    assert plain.rejected["outside_valid_interior"] > 0


def test_a_feature_in_the_middle_is_found_identically_either_way():
    """The declaration must change the answer only where the topology actually differs."""
    data = _seam_blob(64.0, periodic=True)
    a = _run(_field(data, axes=PERIODIC_AXES, dataset="seam"), seed=11)
    b = _run(_field(data, axes=(ROW, COL), dataset="seam"), seed=11)
    assert a.found == b.found == 1
    assert a[0].location.coords["col"] == pytest.approx(b[0].location.coords["col"])


def test_the_field_supplies_the_periods_a_separation_across_the_seam_needs():
    """TG2.1 refuses a separation on a periodic axis without its length; this is where it
    comes from. The axis length is a property of the data, not of the feature."""
    axes = PERIODIC_AXES
    left = _run(_field(_seam_blob(2.0, True), axes=axes, dataset="seam", time=0.0), seed=11)
    right = _run(_field(_seam_blob(126.0, True), axes=axes, dataset="seam", time=1.0), seed=11)
    assert left.field.periods() == {"row": 128.0, "col": 128.0}
    separation = left[0].separation_to(right[0], periods=left.field.periods())
    assert separation.value == pytest.approx(4.0, abs=0.5)
    naive = left[0].location.coords["col"] - right[0].location.coords["col"]
    assert abs(naive) > 120.0        # what the same pair looks like without the declaration


def test_a_coordinate_vector_puts_the_position_and_the_scale_in_the_axiss_own_units():
    """Index space is a default, not an assumption. A declared coordinate converts both."""
    data = _seam_blob(64.0, periodic=True)
    axes = (AxisSpec("y", "space", units="m", ordinal=0),
            AxisSpec("x", "space", units="m", ordinal=1))
    metres = np.arange(128, dtype=np.float64) * 250.0
    field = _field(data, axes=axes, dataset="seam",
                   coordinates={"y": metres, "x": metres})
    result = _run(field, seed=11)
    assert result.found == 1
    assert result[0].location.coords["x"] == pytest.approx(64.0 * 250.0, abs=250.0)
    assert result[0].spatial_scale.units == "m"
    assert result[0].spatial_scale.value == pytest.approx(5.0 * 250.0, rel=0.1)


def test_a_stretched_coordinate_axis_is_refused_rather_than_averaged():
    """A width in cells converts through one spacing. A stretched axis has no such number,
    and the honest answer depends on where the feature is - which is what the conversion
    would hide."""
    stretched = np.cumsum(np.linspace(1.0, 3.0, 128))
    with pytest.raises(InvalidParameterError) as exc:
        _field(np.zeros((128, 128)), axes=(ROW, AxisSpec("x", "space", units="m")),
               coordinates={"row": np.arange(128.0), "x": stretched})
    assert "uniformly spaced" in str(exc.value)


def test_axes_in_different_units_get_no_single_isotropic_scale():
    """One number cannot be a width in cells and a width in metres at the same time."""
    data = _seam_blob(64.0, periodic=True)
    axes = (AxisSpec("row", "space", units="cells", ordinal=0),
            AxisSpec("x", "space", units="m", ordinal=1))
    result = _run(_field(data, axes=axes, dataset="seam"), seed=11)
    assert result.found == 1
    assert result[0].spatial_scale is None


def test_a_field_that_will_not_name_its_domain_is_refused():
    with pytest.raises(InvalidParameterError):
        _field(np.zeros((16, 16)), domain=" ")


def test_a_time_axis_inside_a_frame_is_refused_as_a_second_clock():
    with pytest.raises(InvalidParameterError) as exc:
        _field(np.zeros((16, 16)), axes=(AxisSpec("t", "time", units="hours"), COL))
    assert "second clock" in str(exc.value)


def test_a_field_with_non_finite_values_is_refused_before_any_maximum_is_taken():
    data = np.zeros((16, 16))
    data[3, 3] = np.nan
    with pytest.raises(InvalidParameterError) as exc:
        _field(data)
    assert "NaN" in str(exc.value)


def test_axes_that_do_not_match_the_array_are_refused():
    with pytest.raises(InvalidParameterError) as exc:
        _field(np.zeros((16, 16)), axes=(ROW,))
    assert "one axis per array dimension" in str(exc.value)


# -------------------------------------------------------------------------- determinism


def test_the_same_field_and_seed_give_the_same_features_twice():
    """Standard E4. The surrogate ensemble is the only randomness, and it is seeded."""
    field, _ = _planted()
    a, b = _run(field), _run(field)
    assert a.calibration.threshold == b.calibration.threshold
    assert [f.location.coords for f in a] == [f.location.coords for f in b]


def test_a_precomputed_calibration_can_be_reused_across_frames():
    """TG2.3 tracks across a sequence; recalibrating per frame would make the threshold
    move under the tracker, and a feature could then be born by the null shifting."""
    field, _ = _planted()
    cal = calibrate(field.values, alpha=0.05, n_surrogates=N_SURROGATES, seed=SEED)
    assert isinstance(cal, NullCalibration)
    result = extract(field, calibration=cal)
    assert result.calibration is cal
    assert result.found == 3


# --------------------------------------------------------------- 5. the handoff to TG2.3


def _vortex_features():
    """Every frame of `advected_vortex_sequence`, extracted under one calibration."""
    bench = get_benchmark("advected_vortex_sequence")
    sequence = bench.make()
    frames = [np.asarray(f.data.numpy(), dtype=np.float64) for f in sequence.fields]
    cal = calibrate(frames[0], alpha=0.05, n_surrogates=N_SURROGATES, seed=7)
    results = [extract(_field(arr, dataset="advected_vortex_sequence", time=float(t)),
                       calibration=cal) for t, arr in enumerate(frames)]
    return results, bench.truth()


def test_the_sequence_the_tracker_will_consume_is_one_feature_in_every_frame():
    """TG2.3's acceptance is *1 track, 1 birth, 0 deaths*, and a tracker cannot reach it
    from an extractor that finds two objects in one frame and none in the next. Everything
    left between here and that criterion is association."""
    results, truth = _vortex_features()
    assert [r.found for r in results] == [1] * len(results)
    for t, r in enumerate(results):
        ty, tx = truth["positions_rowcol"][t]
        assert math.hypot(r[0].location.coords["row"] - ty,
                          r[0].location.coords["col"] - tx) < 1.0


def test_the_measured_scales_recover_the_benchmarks_known_doubling_schedule():
    """The vortex doubles every 16 steps, and the extractor was not told so."""
    results, truth = _vortex_features()
    scales = [r[0].spatial_scale for r in results]
    for measured, known in zip(scales, truth["sigma_cells"]):
        assert abs(measured.value / known - 1.0) < 0.05
    doubling = int(truth["scale_doubling_steps"])
    assert scales[doubling].ratio_to(scales[0]) == pytest.approx(2.0, rel=0.05)


def test_one_calibration_covers_the_whole_sequence_so_a_birth_cannot_be_the_threshold_moving():
    """Every frame in that sequence was cut at the same number, and the receipt says so."""
    results, _ = _vortex_features()
    thresholds = {r.calibration.threshold for r in results}
    assert len(thresholds) == 1
    combined = FeatureSet([f for r in results for f in r])
    assert len(combined.frames()) == len(results)


# ------------- A diverged scale is refused by name, not acted on
#
# `_localise` corrects the integral estimator for the fraction of a Gaussian its window
# captures, and that correction is a fixed point: a larger sigma captures less of itself, and
# dividing by the smaller capture returns a larger sigma. When the excess is broader than the
# window, or when the refined amplitude collapses toward zero, it diverges.
#
# Found on a real ERA5 SWT level_1/HH plane, where it produced sigma 72,404 cells on a 161-cell
# frame and then asked for a 42.5 TiB index array, killing the pass. Every frame of the acquired
# record was unextractable because of it. The finite check upstream does not catch this: a
# runaway here is a large finite number, not an infinity.


def test_a_diverged_scale_is_refused_rather_than_acted_on():
    """The guard, exercised on the path that actually diverged.

    On later refinement rounds the amplitude is the refined disc amplitude rather than the peak
    height, so an amplitude collapsing toward zero sends sigma up without bound. The refusal
    must fire before that width reaches `_disc_amplitude`, because the allocation there is
    proportional to it.
    """
    import numpy as np

    import src.core.extraction as extraction

    field = np.zeros((48, 48))
    field[16:32, 16:32] = 1.0
    field[24, 24] = 1.0000001

    offered = []
    original = extraction._disc_amplitude

    def collapsing(values, centre, sigma, baseline, periodic):
        offered.append(sigma)
        return 1e-12                      # a refined amplitude that has collapsed

    extraction._disc_amplitude = collapsing
    try:
        measured = extraction._localise(
            field, (24, 24), float(field[24, 24]), float(np.median(field)),
            (False, False), 2.0, 8)
    finally:
        extraction._disc_amplitude = original

    assert measured is None, "a diverged scale was returned instead of refused"
    limit = extraction._MAX_MEASURABLE_SCALE_CELLS * max(field.shape)
    assert all(sigma <= limit for sigma in offered), (
        "a width past the frame reached _disc_amplitude, where the allocation is proportional "
        "to it: %r" % (offered,))


def test_the_refusal_is_counted_by_name_and_does_not_end_the_pass():
    """`unmeasurable_scale` already existed; it simply was not reachable.

    A missing feature is a fact a receipt can carry. A pass that dies on one plane is not.
    """
    import numpy as np

    import src.core.extraction as extraction
    from src.core.domain import AxisSpec
    from src.core.extraction import ExtractionField, extract

    rng = np.random.default_rng(11)
    values = rng.normal(0.0, 1.0, (48, 48))
    values[24, 24] += 40.0

    original = extraction._disc_amplitude
    extraction._disc_amplitude = lambda *a, **k: 1e-12
    try:
        result = extract(
            ExtractionField(
                values=values,
                axes=(AxisSpec("row", "space", units="cells", ordinal=0),
                      AxisSpec("col", "space", units="cells", ordinal=1)),
                domain="synthetic", dataset="diverged_scale", variable="amplitude",
                units=None, time=0.0, time_units="frames", representation="identity"),
            n_surrogates=19, seed=3)
    finally:
        extraction._disc_amplitude = original

    rejected = result.describe()["rejected"]
    assert rejected["unmeasurable_scale"] >= 1, rejected
    assert isinstance(list(result), list), "the pass did not complete"


def test_the_bound_is_generous_enough_not_to_trim_a_real_feature():
    """It exists to stop a diverged fixed point, not to narrow the extractor.

    A Gaussian whose width equals the frame is already unmeasurable from that frame, so a
    limit at the frame's own extent cannot reject anything a real extraction produces.
    """
    import numpy as np

    from src.core.domain import AxisSpec
    from src.core.extraction import (
        _MAX_MEASURABLE_SCALE_CELLS, ExtractionField, extract)

    assert _MAX_MEASURABLE_SCALE_CELLS == 1.0

    size, sigma = 64, 3.0
    rows, cols = np.mgrid[0:size, 0:size]
    values = 5.0 * np.exp(-(((rows - 32.0) ** 2 + (cols - 32.0) ** 2) / (2.0 * sigma ** 2)))
    result = extract(
        ExtractionField(
            values=values,
            axes=(AxisSpec("row", "space", units="cells", ordinal=0),
                  AxisSpec("col", "space", units="cells", ordinal=1)),
            domain="synthetic", dataset="one_clean_gaussian", variable="amplitude",
            units=None, time=0.0, time_units="frames", representation="identity"),
        n_surrogates=99, seed=5)

    features = list(result)
    assert len(features) == 1, [f.location.coords for f in features]
    assert result.describe()["rejected"]["unmeasurable_scale"] == 0
    measured = float(features[0].spatial_scale.value)
    assert measured < _MAX_MEASURABLE_SCALE_CELLS * size
    assert 1.0 < measured < 12.0, measured
