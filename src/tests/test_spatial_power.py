"""T4C.5i: the spatial sample count a crop actually supplies (fixes D84).

The tests that matter here are the known-answer ones. A decorrelation estimator can always be
made to return a plausible number; what it must do is return the *right* number on a field whose
correlation length was set by construction, and refuse to return one at all when the field does
not decorrelate inside the crop it was given.
"""

import math

import numpy as np
import pytest

from src.analysis_engine.spatial_power import (
    AREA_EXPONENT_BOUNDS,
    DECORRELATION_THRESHOLD,
    REMEDY_AXES,
    MIN_EFFECTIVE_SAMPLES,
    TRUST_HORIZON_FRACTION,
    StreamedAttenuation,
    assess_interior,
    attenuation_curve,
    centred_subcrop,
    subcrop_sizes,
    effective_spatial_samples,
    energy_density_series,
    extrapolate_attenuation,
    crop_for_effect,
    detection_rank,
    family_for_effect,
    frames_for_resolution,
    spatial_power_refusal,
    minimum_detectable_effect,
    power_verdict,
    spatial_decorrelation,
)
from src.core.errors import InvalidParameterError


def _white(shape, seed=20260901):
    return np.random.default_rng(seed).standard_normal(shape)


def _correlated(shape, length_px, seed=20260901):
    """A field whose autocorrelation crosses 1/e at approximately ``length_px``.

    Smoothing white noise with a Gaussian kernel of standard deviation ``s`` gives a field whose
    autocorrelation is Gaussian with standard deviation ``s * sqrt(2)``. A Gaussian of standard
    deviation ``d`` crosses 1/e at lag ``d * sqrt(2)``, not at ``d``. So the crossing sits at
    ``2 * s``, and a kernel of ``length_px / 2`` puts it exactly where the caller asked.

    Getting this wrong is how a decorrelation estimator gets "calibrated" against its own
    mistake: the first version of this helper was off by sqrt(2) and would have been tuned away
    by widening the tolerance until the module agreed with it.
    """
    rng = np.random.default_rng(seed)
    sigma = length_px / 2.0
    radius = int(math.ceil(4 * sigma))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel /= kernel.sum()
    pad = radius
    noise = rng.standard_normal((shape[0] + 2 * pad, shape[1] + 2 * pad))
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 0, noise)
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, smoothed)
    return smoothed[pad:pad + shape[0], pad:pad + shape[1]]


# ------------------------------------------------------------------ known answers

def test_white_noise_decorrelates_at_one_pixel():
    result = spatial_decorrelation(_white((139, 139)))
    assert result["saturated"] is False
    assert [row["decorrelation_px"] for row in result["axes"]] == [1, 1]


@pytest.mark.parametrize("length_px", [4, 8, 16])
def test_a_constructed_correlation_length_is_recovered(length_px):
    """The estimator is calibrated, not merely monotone: it lands on the built-in length."""
    result = spatial_decorrelation(_correlated((256, 256), length_px))
    assert result["saturated"] is False
    for row in result["axes"]:
        assert abs(row["decorrelation_px"] - length_px) <= max(2, length_px // 3), row


def test_effective_samples_fall_as_the_structure_grows():
    counts = []
    for length_px in (2, 8, 32):
        assessment = assess_interior(_correlated((256, 256), length_px))
        counts.append(assessment["effective_samples"]["effective_samples"])
    assert counts[0] > counts[1] > counts[2]
    # 256px of interior at 32px structure is ~8x8 independent patches, not 65,536 samples.
    assert counts[-1] < 100


def test_pixel_count_is_never_treated_as_the_sample_count():
    """The failure this module exists to prevent, stated as an assertion."""
    interior = _correlated((139, 139), 16)
    ess = assess_interior(interior)["effective_samples"]["effective_samples"]
    assert ess < interior.size / 100


# ------------------------------------------------------------------ the model-free refusal

def test_a_crop_that_never_decorrelates_reports_saturation_rather_than_a_length():
    """One structure per crop. No constant is needed to know that is not a sample."""
    result = spatial_decorrelation(_correlated((24, 24), 40))
    assert result["saturated"] is True
    assert all(row["decorrelation_px"] is None for row in result["axes"])
    assert "longer than this crop can measure" in result["axes"][0]["reason"]


def test_saturation_yields_no_effective_sample_count_and_is_inadequate():
    assessment = assess_interior(_correlated((24, 24), 40))
    ess = assessment["effective_samples"]
    assert ess["effective_samples"] is None
    assert ess["adequate"] is False
    assert "one independent structure" in ess["reason"]


def test_the_searched_limit_is_never_substituted_for_an_unmeasured_length():
    """Reporting "as long as we looked" would convert ignorance into an optimistic number."""
    result = spatial_decorrelation(_correlated((24, 24), 40))
    searched = result["axes"][0]["searched_to_px"]
    assert searched == int(24 * TRUST_HORIZON_FRACTION) == 6
    assert result["axes"][0]["decorrelation_px"] is None
    assert effective_spatial_samples(result)["effective_samples"] is None


def test_mean_centring_cannot_manufacture_a_decorrelation_length():
    """The artefact this module's trust horizon exists to refuse.

    A field far smoother than its window still shows a falling sample autocorrelation at long
    lags, because the window was centred on its own mean. Searching that far would return a
    confident length for a crop holding one structure -- the optimistic direction.
    """
    size, structure_px = 40, 60
    interior = _correlated((size, size), structure_px)

    # Searching to half the interior -- what the temporal estimator can afford over thousands of
    # frames -- finds a crossing at 19 px in a field whose real structure is 60 px. That number
    # is the window's own mean-centring, and nothing about it looks wrong.
    centred = interior - interior.mean()
    variance = float((centred ** 2).sum())
    artefact = [lag for lag in range(1, size // 2 + 1)
                if abs(float((centred[:-lag] * centred[lag:]).sum()) / variance)
                < DECORRELATION_THRESHOLD]
    assert artefact, "the artefact must be present, or this test proves nothing"
    assert artefact[0] > int(size * TRUST_HORIZON_FRACTION)
    assert artefact[0] < structure_px, (
        "the artefact reports a length shorter than the structure, so it flatters the "
        "sample count in exactly the optimistic direction")

    # Within the horizon the module declines to name a length at all.
    assert spatial_decorrelation(interior)["saturated"] is True


def test_a_large_enough_crop_measures_what_a_small_one_could_not():
    """The same field: saturated at 24px of interior, measured at 256px. Size is the variable."""
    small = spatial_decorrelation(_correlated((24, 24), 12, seed=7))
    large = spatial_decorrelation(_correlated((256, 256), 12, seed=7))
    assert small["saturated"] is True
    assert large["saturated"] is False
    assert all(row["decorrelation_px"] >= 8 for row in large["axes"])


# ------------------------------------------------------------------ contracts

def test_a_constant_interior_is_one_sample_not_an_error():
    result = spatial_decorrelation(np.full((64, 64), 3.5))
    assert [row["decorrelation_px"] for row in result["axes"]] == [1, 1]
    assert "constant" in result["axes"][0]["reason"]


def test_a_masked_interior_is_refused_rather_than_silently_dropped():
    band = _white((64, 64))
    band[10, 10] = np.nan
    with pytest.raises(InvalidParameterError, match="fully finite interior"):
        spatial_decorrelation(band)


def test_a_one_dimensional_series_is_refused():
    with pytest.raises(InvalidParameterError, match="2-D valid-interior array"):
        spatial_decorrelation(np.arange(64.0))


def test_the_threshold_matches_the_temporal_theiler_convention():
    from src.analysis_engine.cross_scale import decorrelation_frames

    assert DECORRELATION_THRESHOLD == pytest.approx(1.0 / math.e)
    # The spatial estimator must agree with the temporal one on the same 1-D structure.
    series = _correlated((512, 4), 8)[:, 0]
    temporal = decorrelation_frames(series)
    spatial = spatial_decorrelation(_correlated((512, 512), 8))["axes"][0]["decorrelation_px"]
    assert abs(temporal - spatial) <= 3


def test_the_effective_sample_floor_is_arithmetic_not_a_judgement_constant():
    """MIN_EFFECTIVE_SAMPLES replaces MIN_VALID_INTERIOR only where counting is meaningless."""
    assert MIN_EFFECTIVE_SAMPLES == 2.0
    result = spatial_decorrelation(_correlated((64, 64), 24))
    assessment = effective_spatial_samples(result)
    if assessment["effective_samples"] is not None:
        assert assessment["adequate"] == (
            assessment["effective_samples"] >= MIN_EFFECTIVE_SAMPLES)


# ============================================================ planted attenuation (T4C.5i step 3)

def _planted_pair(frames=1200, size=96, structure_px=2, lag=3, coupling=0.3, seed=20260901):
    """Two field stacks with a known directed coupling in their per-frame energy density.

    A latent AR(1) drive sets the source's energy density; the target's is driven by the source's
    value ``lag`` frames earlier plus its own independent variation. The default coupling is
    deliberately weak: the attenuation model is a weak-dependence approximation, and a strong
    coupling drives the transfer entropy toward the ``log(bins)`` ceiling where no such model
    holds. Detecting weak effects is also the regime a gate actually operates in. Each frame's spatial pattern
    is freshly drawn correlated noise, so averaging over a smaller window resolves the latent
    amplitude less precisely -- which is exactly the attenuation being measured, arising from the
    geometry rather than being added by hand as a noise term.
    """
    rng = np.random.default_rng(seed)
    drive = np.zeros(frames)
    for t in range(1, frames):
        drive[t] = 0.85 * drive[t - 1] + rng.standard_normal()
    drive = drive - drive.min() + 1.0

    target_level = np.empty(frames)
    innovation = rng.standard_normal(frames)
    for t in range(frames):
        past = drive[t - lag] if t >= lag else drive[0]
        target_level[t] = coupling * past + (1.0 - coupling) * abs(innovation[t]) + 1.0

    def stack(levels, offset):
        out = np.empty((frames, size, size))
        for t in range(frames):
            pattern = _correlated((size, size), structure_px, seed=seed + offset + t)
            pattern = pattern / np.sqrt(np.mean(pattern ** 2))
            out[t] = math.sqrt(levels[t]) * pattern
        return out

    return stack(drive, 100_000), stack(target_level, 500_000)


@pytest.fixture(scope="module")
def planted():
    source, target = _planted_pair()
    return {"source": source, "target": target,
            "curve": attenuation_curve(source, target, lag=3, bins=4)}


def test_the_planted_effect_attenuates_as_the_crop_shrinks(planted):
    """The premise of the whole task, demonstrated rather than assumed."""
    rows = sorted(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    measurable = [row for row in rows if row["effective_samples"] is not None]
    assert len(measurable) >= 3, "the fixture must span crops whose sample count is readable"
    smallest, largest = measurable[0], measurable[-1]
    assert smallest["transfer_entropy_nats"] < largest["transfer_entropy_nats"]
    assert smallest["effective_samples"] < largest["effective_samples"]


def test_a_subcrop_below_its_own_trust_horizon_is_reported_not_silently_counted(planted):
    """A crop too small to measure its own structure keeps its row and loses its number."""
    for row in planted["curve"]["curve"]:
        if row["effective_samples"] is None:
            assert row["interior_px"] < planted["curve"]["full_interior_px"]
            break


def test_only_spatial_precision_varies_across_the_curve(planted):
    """If frames, bins or lag moved, the curve would not be readable as attenuation."""
    curve = planted["curve"]
    assert curve["bins"] == 4 and curve["lag"] == 3
    sizes = [row["interior_px"] for row in curve["curve"]]
    assert sizes == sorted(set(sizes))
    assert energy_density_series(planted["source"]).size == planted["source"].shape[0]
    assert energy_density_series(
        centred_subcrop(planted["source"], 16)).size == planted["source"].shape[0]


def test_the_extrapolation_exceeds_every_measured_crop(planted):
    """Attenuation is toward zero, so the unlimited-crop estimate cannot sit below the data."""
    extrapolation = extrapolate_attenuation(planted["curve"])
    assert extrapolation["extrapolated_transfer_entropy_nats"] is not None
    assert extrapolation["fit_r_squared"] >= 0.8
    largest = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    assert (extrapolation["extrapolated_transfer_entropy_nats"]
            >= largest["transfer_entropy_nats"] * 0.99)
    assert 0 < extrapolation["retained_fraction"] <= 1.0


def test_the_same_field_yields_different_verdicts_when_only_the_crop_changes(planted):
    """T4C.5i's load-bearing acceptance criterion.

    An adequately powered absence and an underpowered absence must be distinguishable. Nothing
    varies here but how much of one identical field is averaged.
    """
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    measured = full["transfer_entropy_nats"]
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]

    # A threshold the full crop already clears: the verdict stands on its own evidence.
    below = power_verdict(planted["curve"], detection_threshold_nats=measured * 0.5)
    assert below["verdict"] == "ADEQUATE"

    # A threshold between what this crop sees and what an unlimited crop would: the absence is a
    # property of the crop, and calling it a negative result would be wrong.
    straddling = 0.5 * (measured + extrapolated)
    assert measured < straddling < extrapolated
    verdict = power_verdict(planted["curve"], detection_threshold_nats=straddling)
    assert verdict["verdict"] == "INVALID"
    assert "property of the crop" in verdict["reason"]

    # A threshold neither crop could reach: absent whatever the crop, so an adequately powered
    # FAIL rather than a power failure.
    above = power_verdict(planted["curve"], detection_threshold_nats=extrapolated * 2.0)
    assert above["verdict"] == "ADEQUATE"


def test_a_curve_still_climbing_refuses_and_says_so_distinctly():
    """A near-ceiling coupling: an excellent fit that still must not yield a number.

    The refusal here is not a fit failure -- R^2 is high. The fitted line reaches zero at a
    finite sample count, meaning no plateau is in view at any crop measured. Reporting that as
    "poor fit" would misdiagnose the strongest evidence of inadequacy the curve can give.
    """
    source, target = _planted_pair(structure_px=6, coupling=0.8, seed=4242)
    extrapolation = extrapolate_attenuation(attenuation_curve(source, target, lag=3, bins=4))
    assert extrapolation["extrapolated_transfer_entropy_nats"] is None
    assert extrapolation["plateau_in_view"] is False
    assert extrapolation["fit_r_squared"] > 0.8, "the fit is fine; the curve is not"
    assert "still rising steeply" in extrapolation["reason"]


def test_a_readable_curve_has_a_plateau_in_view(planted):
    extrapolation = extrapolate_attenuation(planted["curve"])
    assert extrapolation["plateau_in_view"] is True
    assert extrapolation["fit_r_squared"] >= 0.8


def test_an_unreadable_curve_refuses_rather_than_naming_an_intercept():
    """A poorly determined extrapolation must not be dressed up as a power correction."""
    curve = {"lag": 3, "bins": 4, "full_interior_px": 64, "curve": [
        {"interior_px": 16, "effective_samples": 4.0, "transfer_entropy_nats": 0.0},
        {"interior_px": 32, "effective_samples": 16.0, "transfer_entropy_nats": 0.0},
        {"interior_px": 64, "effective_samples": 64.0, "transfer_entropy_nats": 0.0}]}
    extrapolation = extrapolate_attenuation(curve)
    assert extrapolation["extrapolated_transfer_entropy_nats"] is None
    assert "no trend can be read" in extrapolation["reason"]
    assert power_verdict(curve, detection_threshold_nats=0.01)["verdict"] == "INVALID"


def test_a_subcrop_is_concentric_and_bounded():
    stack = _white((5, 32, 32)).reshape(5, 32, 32)
    assert centred_subcrop(stack, 8).shape == (5, 8, 8)
    np.testing.assert_allclose(centred_subcrop(stack, 32), stack)
    with pytest.raises(InvalidParameterError, match="no larger than the interior"):
        centred_subcrop(stack, 33)


def test_energy_density_matches_the_signature_definition():
    """The curve must consume the same statistic the gate does, not a lookalike."""
    stack = _white((3, 8, 8))
    expected = [float(np.mean(frame ** 2)) for frame in stack]
    np.testing.assert_allclose(energy_density_series(stack), expected)


def test_a_detection_threshold_must_be_a_positive_effect():
    curve = {"lag": 3, "bins": 4, "full_interior_px": 64, "curve": [
        {"interior_px": 64, "effective_samples": 64.0, "transfer_entropy_nats": 0.1}]}
    with pytest.raises(InvalidParameterError, match="positive transfer entropy"):
        power_verdict(curve, detection_threshold_nats=0.0)


# ================================================ minimum detectable effect (T4C.5i step 4)

CAMPAIGN_TESTS = 36            # t4c6_nz_era5_temperature_850_v1: 6 ordered scale pairs x 6 lags
CAMPAIGN_SURROGATES = 4999
CAMPAIGN_ALPHA = 0.05


def _null(size=CAMPAIGN_SURROGATES, seed=20260901):
    """A plausible circular-shift null: non-negative, right-skewed, bounded away from zero.

    The shape is not load-bearing -- every claim below is an order statistic of whatever the
    ensemble actually produced -- but a symmetric null would let a normal-theory bug pass.
    """
    return np.random.default_rng(seed).gamma(2.0, 0.01, size)


def test_the_campaign_design_admits_exactly_one_ranking():
    """36 tests, BY, alpha 0.05 and 4,999 shifts leave no room for a single exceedance."""
    rank = detection_rank(n_tests=CAMPAIGN_TESTS, n_surrogates=CAMPAIGN_SURROGATES)
    assert rank["can_detect"]
    assert rank["max_exceedances"] == 0
    assert rank["p_value_floor"] == pytest.approx(1.0 / 5000.0)
    # The observation must beat every surrogate, and only just: the floor clears the required
    # level by a factor of 1.66, so the design has no margin to spend on a second exceedance.
    assert rank["required_raw_p"] > rank["p_value_floor"]
    assert 2.0 / 5000.0 > rank["required_raw_p"]


def test_the_dependence_penalty_is_the_repositorys_own():
    """If this module's BY penalty ever drifts from the one that screens the family, the derived
    threshold would describe a procedure the gate does not run."""
    from src.statistics.multiple_comparisons import adjust

    rank = detection_rank(n_tests=CAMPAIGN_TESTS, n_surrogates=CAMPAIGN_SURROGATES)
    reference = adjust([0.5] * CAMPAIGN_TESTS, method="benjamini_yekutieli",
                       alpha=CAMPAIGN_ALPHA)
    assert rank["dependence_penalty"] == pytest.approx(reference["dependence_penalty"])


def test_the_required_level_agrees_with_the_existing_power_check():
    """`required_surrogates` inverts the same arithmetic; the two must not disagree."""
    from src.statistics.multiple_comparisons import required_surrogates

    rank = detection_rank(n_tests=CAMPAIGN_TESTS, n_surrogates=CAMPAIGN_SURROGATES)
    needed = required_surrogates(CAMPAIGN_TESTS, CAMPAIGN_ALPHA, "benjamini_yekutieli")
    assert needed == math.ceil(1.0 / rank["required_raw_p"]) - 1
    assert (CAMPAIGN_SURROGATES >= needed) is rank["can_detect"]


def test_a_design_that_cannot_reject_has_no_minimum_detectable_effect():
    """The distinction that matters: no threshold at all, rather than a very large one."""
    rank = detection_rank(n_tests=CAMPAIGN_TESTS, n_surrogates=199)
    assert not rank["can_detect"]
    assert rank["max_exceedances"] < 0
    assert "no detection at all" in rank["reason"]

    effect = minimum_detectable_effect(_null(199), n_tests=CAMPAIGN_TESTS)
    assert effect["minimum_detectable_effect_nats"] is None
    assert "no detection at all" in effect["reason"]


def test_the_threshold_is_the_ensemble_maximum_at_rank_one():
    null = _null()
    effect = minimum_detectable_effect(null, n_tests=CAMPAIGN_TESTS)
    assert effect["order_statistic_rank"] == 1
    assert effect["minimum_detectable_effect_nats"] == pytest.approx(float(null.max()))
    assert effect["minimum_detectable_excess_nats"] == pytest.approx(
        float(null.max() - null.mean()))


def test_a_smaller_family_reaches_deeper_into_the_ensemble():
    """Fewer declared tests buy a laxer corrected level, which is a lower threshold in nats."""
    null = _null()
    small = minimum_detectable_effect(null, n_tests=2)
    large = minimum_detectable_effect(null, n_tests=CAMPAIGN_TESTS)
    assert small["order_statistic_rank"] > large["order_statistic_rank"]
    assert (small["minimum_detectable_effect_nats"]
            < large["minimum_detectable_effect_nats"])
    ordered = np.sort(null)[::-1]
    assert small["minimum_detectable_effect_nats"] == pytest.approx(
        float(ordered[small["order_statistic_rank"] - 1]))


def test_the_threshold_is_exactly_where_the_real_screen_changes_its_mind():
    """T4C.5i step 4's load-bearing check.

    The derived number is only worth anything if it is the *same* boundary the gate applies. So
    this reproduces the gate's own arithmetic -- `p = (1 + k) / (1 + n)` from `cross_scale`, then
    `screen` over the declared family -- and confirms the rejection flips across the threshold
    and nowhere else. An observation a hair above it is significant; a hair below, and the
    identical machinery reports nothing.
    """
    from src.statistics.significance import screen

    null = _null()
    threshold = minimum_detectable_effect(
        null, n_tests=CAMPAIGN_TESTS)["minimum_detectable_effect_nats"]

    def rejected(observed):
        k = int(np.sum(null >= observed))
        p = (1.0 + k) / (1.0 + null.size)
        # The rest of the declared family saw nothing, which is the case rank 1 must survive.
        family = [{"label": "tested", "p_value": p}] + [
            {"label": "other%d" % i, "p_value": 1.0}
            for i in range(CAMPAIGN_TESTS - 1)]
        result = screen(family, alpha=CAMPAIGN_ALPHA, method="benjamini_yekutieli")
        return result["n_significant"] >= 1

    assert rejected(threshold * 1.000001)
    assert not rejected(threshold)
    assert not rejected(threshold * 0.999999)


def test_non_finite_surrogates_are_discarded_and_counted():
    """A silently shortened ensemble would shift the rank without saying so."""
    null = _null(100)
    spoiled = np.concatenate([null, [np.nan, np.inf]])
    effect = minimum_detectable_effect(spoiled, n_tests=2)
    assert effect["n_finite_surrogates"] == 100
    assert effect["n_discarded_surrogates"] == 2
    assert effect["detection_rank"]["n_surrogates"] == 100


def test_the_derived_threshold_drives_the_power_verdict(planted):
    """Step 4 supplies exactly what step 3 left to the caller."""
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    # A null whose maximum sits between the measured and extrapolated effects, which is the
    # configuration that must come back INVALID rather than as a negative result.
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    straddle = 0.5 * (full["transfer_entropy_nats"] + extrapolated)
    null = _null() * (straddle / float(_null().max()))
    threshold = minimum_detectable_effect(
        null, n_tests=CAMPAIGN_TESTS)["minimum_detectable_effect_nats"]
    assert threshold == pytest.approx(straddle)
    verdict = power_verdict(planted["curve"], detection_threshold_nats=threshold)
    assert verdict["verdict"] == "INVALID"


@pytest.mark.parametrize("kwargs, match", [
    ({"n_tests": 0, "n_surrogates": 999}, "positive number of declared tests"),
    ({"n_tests": 36, "n_surrogates": 0}, "positive ensemble size"),
    ({"n_tests": 36, "n_surrogates": 999, "alpha": 0.0}, "significance level"),
    ({"n_tests": 36, "n_surrogates": 999, "correction": "sidak"}, "one of"),
])
def test_a_design_that_was_not_declared_is_refused(kwargs, match):
    with pytest.raises(InvalidParameterError, match=match):
        detection_rank(**kwargs)


def test_an_ensemble_too_small_to_order_is_refused():
    with pytest.raises(InvalidParameterError, match="at least two finite"):
        minimum_detectable_effect([0.1], n_tests=2)


# ============================================ refusing on the derived quantity (T4C.5i step 5)

# The frozen campaign's own split, from the preregistered protocol: 4,382 train frames and 2,914
# test frames at a lag family starting at 6.
CAMPAIGN_TRAIN_FRAMES = 4382
CAMPAIGN_TEST_FRAMES = 2914
CAMPAIGN_LAG = 6


def test_a_refusal_may_only_name_an_axis_the_study_can_change():
    assert REMEDY_AXES == ("crop_size", "frame_count", "scale_count")


def test_the_train_partition_resolves_the_corrected_level():
    resolution = frames_for_resolution(
        n_frames=CAMPAIGN_TRAIN_FRAMES, lag=CAMPAIGN_LAG, theiler=24,
        n_tests=CAMPAIGN_TESTS)
    assert resolution["resolves_corrected_level"]
    assert resolution["frames_required"] is None
    assert resolution["attainable_exact_p"] < resolution["required_raw_p"]


def test_the_test_partition_cannot_resolve_the_corrected_level():
    """Defect D85, found by this step and recorded rather than repaired here.

    The confirmatory half of the frozen campaign is 2,914 frames. Excluding the tested and
    simultaneous alignments leaves fewer distinct admissible shifts than the declared 36-test
    family needs, so the exact test's attainable p-value never reaches the corrected level --
    however many of the 4,999 surrogates are drawn, because they are drawn *with replacement*
    from that same small reference set. `check_power` does not see this: it counts draws.
    """
    resolution = frames_for_resolution(
        n_frames=CAMPAIGN_TEST_FRAMES, lag=CAMPAIGN_LAG, theiler=24,
        n_tests=CAMPAIGN_TESTS)
    assert not resolution["resolves_corrected_level"]
    assert resolution["attainable_exact_p"] > resolution["required_raw_p"]
    assert resolution["frames_required"] > CAMPAIGN_TEST_FRAMES
    assert "however many surrogates are drawn" in resolution["reason"]


def test_the_unresolvable_partition_is_confirmed_by_the_gates_own_screen():
    """The claim above is not this module's arithmetic; it is the gate's verdict.

    Feeding the best p-value the test partition can attain into `screen` over the declared
    family returns q = 0.0516 against alpha = 0.05 -- not significant. The same construction on
    the train partition returns q = 0.0347 and is significant.
    """
    from src.analysis_engine.cross_scale import admissible_shifts
    from src.statistics.significance import screen

    def best_q(frames):
        distinct = int(admissible_shifts(frames, CAMPAIGN_LAG, 24).size)
        family = [{"label": "tested", "p_value": 1.0 / (1.0 + distinct)}] + [
            {"label": "other%d" % i, "p_value": 1.0} for i in range(CAMPAIGN_TESTS - 1)]
        result = screen(family, alpha=CAMPAIGN_ALPHA, method="benjamini_yekutieli")
        return [row for row in result["results"] if row["label"] == "tested"][0]

    assert best_q(CAMPAIGN_TRAIN_FRAMES)["significant"]
    assert not best_q(CAMPAIGN_TEST_FRAMES)["significant"]


def test_drawing_more_surrogates_does_not_repair_a_short_record():
    """The distinction the whole function exists for: draws are not resolution."""
    few = frames_for_resolution(n_frames=CAMPAIGN_TEST_FRAMES, lag=CAMPAIGN_LAG, theiler=24,
                                n_tests=CAMPAIGN_TESTS)
    assert few["distinct_admissible_shifts"] < few["distinct_shifts_required"]
    # `check_power` is satisfied by the campaign's 4,999 requested draws on the same partition.
    from src.statistics.multiple_comparisons import check_power
    assert check_power(CAMPAIGN_SURROGATES, CAMPAIGN_TESTS, CAMPAIGN_ALPHA,
                       "benjamini_yekutieli")["can_reject_after_correction"]


def test_a_wider_decorrelation_window_costs_resolution():
    narrow = frames_for_resolution(n_frames=3200, lag=CAMPAIGN_LAG, theiler=1,
                                   n_tests=CAMPAIGN_TESTS)
    wide = frames_for_resolution(n_frames=3200, lag=CAMPAIGN_LAG, theiler=120,
                                 n_tests=CAMPAIGN_TESTS)
    assert wide["distinct_admissible_shifts"] < narrow["distinct_admissible_shifts"]
    assert wide["attainable_exact_p"] > narrow["attainable_exact_p"]


def test_the_crop_that_would_close_the_deficit_is_larger_than_the_one_measured(planted):
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    target = 0.5 * (full["transfer_entropy_nats"] + extrapolated)
    crop = crop_for_effect(planted["curve"], target_nats=target)
    assert crop["interior_px_required"] > crop["interior_px_now"]
    assert crop["effective_samples_required"] > crop["effective_samples_now"]
    assert AREA_EXPONENT_BOUNDS[0] <= crop["measured_area_exponent"] <= AREA_EXPONENT_BOUNDS[1]


def test_a_target_above_the_ceiling_names_no_crop_at_all(planted):
    """The refusal that matters most: no crop size closes an absence."""
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    crop = crop_for_effect(planted["curve"], target_nats=extrapolated * 1.5)
    assert crop["interior_px_required"] is None
    assert "the effect is not attenuated, it is absent" in crop["reason"]


def test_a_larger_target_needs_a_larger_crop(planted):
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    low = crop_for_effect(planted["curve"],
                          target_nats=0.3 * full["transfer_entropy_nats"]
                          + 0.7 * extrapolated)
    high = crop_for_effect(planted["curve"],
                           target_nats=0.05 * full["transfer_entropy_nats"]
                           + 0.95 * extrapolated)
    assert high["interior_px_required"] > low["interior_px_required"]


def test_reducing_the_declared_family_is_reported_and_refused():
    null = _null()
    measured = float(np.sort(null)[::-1][8])       # would clear a family of about nine
    family = family_for_effect(null, measured_nats=measured, n_tests=CAMPAIGN_TESTS)
    assert family["admissible_after_seeing_data"] is False
    assert family["largest_family_that_would_detect"] < CAMPAIGN_TESTS
    assert "how a null result is converted into a finding" in family["reason"]
    # The reported family really is the largest that would have detected it.
    largest = family["largest_family_that_would_detect"]
    assert measured > minimum_detectable_effect(
        null, n_tests=largest)["minimum_detectable_effect_nats"]
    assert measured <= minimum_detectable_effect(
        null, n_tests=largest + 1)["minimum_detectable_effect_nats"]


def test_an_effect_no_family_would_detect_says_so():
    null = _null()
    family = family_for_effect(null, measured_nats=float(null.min()) * 0.5,
                               n_tests=CAMPAIGN_TESTS)
    assert family["largest_family_that_would_detect"] is None
    assert "the design's reach is not what limited it" in family["reason"]


def test_a_short_record_is_refused_before_the_threshold_is_consulted(planted):
    """Resolution binds first: a record that cannot reach the level has no threshold to miss."""
    refusal = spatial_power_refusal(
        planted["curve"], null_nats=_null(), n_tests=CAMPAIGN_TESTS,
        n_frames=CAMPAIGN_TEST_FRAMES, theiler=24)
    assert refusal["verdict"] == "INVALID"
    assert refusal["deficit"] == "resolution"
    assert [remedy["axis"] for remedy in refusal["remedies"]] == ["frame_count"]
    assert refusal["remedies"][0]["required"] > CAMPAIGN_TEST_FRAMES
    assert "power_verdict" not in refusal


def test_an_attenuation_deficit_names_the_crop_and_refuses_the_family(planted):
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    straddle = 0.5 * (full["transfer_entropy_nats"] + extrapolated)
    null = _null() * (straddle / float(_null().max()))
    refusal = spatial_power_refusal(
        planted["curve"], null_nats=null, n_tests=CAMPAIGN_TESTS,
        n_frames=CAMPAIGN_TRAIN_FRAMES, theiler=24)
    assert refusal["verdict"] == "INVALID"
    assert refusal["deficit"] == "attenuation"
    axes = {remedy["axis"]: remedy for remedy in refusal["remedies"]}
    assert set(axes) == {"crop_size", "scale_count"}
    assert axes["crop_size"]["required"] > axes["crop_size"]["current"]
    assert axes["crop_size"]["admissible_after_seeing_data"] is True
    assert axes["scale_count"]["admissible_after_seeing_data"] is False


def test_an_adequate_crop_is_cleared_to_report_an_absence(planted):
    """ADEQUATE is not a finding; it is permission to call an absence FAIL rather than INVALID."""
    full = max(planted["curve"]["curve"], key=lambda row: row["interior_px"])
    extrapolated = extrapolate_attenuation(
        planted["curve"])["extrapolated_transfer_entropy_nats"]
    null = _null() * ((extrapolated * 2.0) / float(_null().max()))
    refusal = spatial_power_refusal(
        planted["curve"], null_nats=null, n_tests=CAMPAIGN_TESTS,
        n_frames=CAMPAIGN_TRAIN_FRAMES, theiler=24)
    assert refusal["verdict"] == "ADEQUATE"
    assert refusal["deficit"] is None
    assert refusal["remedies"] == []
    assert refusal["measured_transfer_entropy_nats"] == pytest.approx(
        full["transfer_entropy_nats"])


def test_no_refusal_this_module_produces_names_a_bare_constant(planted):
    """T4C.5i's stated contract: a refusal names an axis to acquire, never a number to type."""
    refusal = spatial_power_refusal(
        planted["curve"], null_nats=_null(), n_tests=CAMPAIGN_TESTS,
        n_frames=CAMPAIGN_TEST_FRAMES, theiler=24)
    for remedy in refusal["remedies"]:
        assert remedy["axis"] in REMEDY_AXES
        assert remedy["unit"]
        assert "MIN_VALID_INTERIOR" not in remedy["note"]
        assert "power of two" not in remedy["note"]
    assert "128" not in refusal["reason"]
    assert "256" not in refusal["reason"]


def test_a_record_too_short_to_shuffle_is_refused():
    with pytest.raises(InvalidParameterError, match="at least three frames"):
        frames_for_resolution(n_frames=2, lag=1, theiler=1, n_tests=CAMPAIGN_TESTS)


def test_a_target_that_is_not_an_effect_is_refused(planted):
    with pytest.raises(InvalidParameterError, match="positive transfer entropy"):
        crop_for_effect(planted["curve"], target_nats=0.0)


# ======================================================= streaming the curve (T4C.5i step 7)


def _stream(source, target, *, lag, bins):
    """Build the same curve one frame at a time, as a gate run has to."""
    accumulator = StreamedAttenuation(
        interior_px=int(min(source.shape[1], source.shape[2])),
        n_frames=int(source.shape[0]), lag=lag, bins=bins)
    for index in range(source.shape[0]):
        accumulator.add(index, [source[index]], [target[index]])
    return accumulator.curve()


def test_the_streamed_curve_is_the_array_curve_exactly(planted):
    """Not "close to": the same dict.

    The gate cannot hold the stack -- 4,382 frames of a 139 px interior is 677 MB per scale
    before the orientations are counted -- so it builds the curve by accumulation. That path is
    only trustworthy if it computes the quantity this module's other tests characterise, and the
    honest way to establish it is equality rather than a tolerance.
    """
    assert _stream(planted["source"], planted["target"], lag=3, bins=4) == planted["curve"]


def test_orientations_collapse_the_way_a_signature_collapses_them(planted):
    """Several planes per frame reduce to the mean of squares over the concatenation.

    `scale_signature` concatenates every orientation's interior and takes one mean of squares
    over the whole thing, so a scale with three orientations is not three measurements. Feeding
    the same plane twice must therefore change nothing at all.
    """
    source, target = planted["source"], planted["target"]
    accumulator = StreamedAttenuation(
        interior_px=int(min(source.shape[1], source.shape[2])),
        n_frames=int(source.shape[0]), lag=3, bins=4)
    for index in range(source.shape[0]):
        accumulator.add(index, [source[index], source[index]],
                        [target[index], target[index]])
    doubled = accumulator.curve()
    single = _stream(source, target, lag=3, bins=4)
    assert [row["transfer_entropy_nats"] for row in doubled["curve"]] == \
        [row["transfer_entropy_nats"] for row in single["curve"]]


def test_a_curve_cannot_be_read_from_a_subset_of_the_record():
    """A partial curve is a different measurement wearing the same name."""
    rng = np.random.default_rng(77)
    accumulator = StreamedAttenuation(interior_px=16, n_frames=8, lag=1, bins=3)
    for index in range(5):
        accumulator.add(index, [rng.standard_normal((16, 16))],
                        [rng.standard_normal((16, 16))])
    with pytest.raises(InvalidParameterError, match="every declared frame"):
        accumulator.curve()


def test_a_frame_cannot_be_counted_twice():
    rng = np.random.default_rng(78)
    accumulator = StreamedAttenuation(interior_px=16, n_frames=4, lag=1, bins=3)
    accumulator.add(0, [rng.standard_normal((16, 16))], [rng.standard_normal((16, 16))])
    with pytest.raises(InvalidParameterError, match="not already recorded"):
        accumulator.add(0, [rng.standard_normal((16, 16))], [rng.standard_normal((16, 16))])


def test_the_two_paths_cannot_choose_different_sizes():
    """`subcrop_sizes` exists so a shared curve cannot be built on unshared rows."""
    assert subcrop_sizes(100) == (25, 40, 55, 70, 85, 100)
    # Duplicates collapse rather than repeat, so a small interior yields fewer rows.
    assert subcrop_sizes(6) == (4, 5, 6)
    with pytest.raises(InvalidParameterError, match="at least four pixels"):
        subcrop_sizes(3)
