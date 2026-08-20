"""Tests for multiplicity control, surrogate nulls and significance (defect D8, R1/R5/R12).

This is the file that decides whether a reported finding is defendable. Two properties matter
more than any other and both are asserted by *measurement* rather than by inspection:

*   **Calibration** - on data where the null is true by construction, the test must reject at
    close to its nominal rate. A test that rejects 76% of the time at alpha = 0.05 is not a
    conservative test, it is a broken one, and that is what an early version did.
*   **Power** - a correction that rejects everything is not a fix. The engine must still find
    a real effect, or D8 would have been "closed" by making the platform permanently silent.

Several thresholds here come from calibration runs recorded in `VERIFICATION.md`, not from
convenience (rule R16).
"""

import math

import numpy as np
import pytest

from src.statistics.multiple_comparisons import (
    ASSUMPTIONS,
    MultipleComparisonError,
    PROCEDURES,
    adjust,
    check_power,
    required_surrogates,
    resolution_limit,
)
from src.statistics.significance import (
    UNIT_ROOT_RHO,
    correlation_test,
    effective_sample_size,
    screen,
    stationarity_check,
    surrogate_p_value,
    surrogate_test,
)
from src.statistics.surrogates import (
    METHODS,
    SurrogateError,
    aaft,
    block_bootstrap,
    circular_shift,
    generate,
    iaaft,
    phase_randomise,
)


def ar1(rng, phi, n=256):
    scale = math.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + scale * rng.standard_normal()
    return x


def mean_abs_diff(f):
    return float(np.abs(np.diff(np.asarray(f).ravel())).mean())


# ============================================================== multiple comparisons

@pytest.mark.parametrize("method", ["benjamini_hochberg", "benjamini_yekutieli"])
@pytest.mark.parametrize("n", [5, 20, 100])
def test_fdr_matches_scipys_reference_implementation(method, n):
    """Independent oracle. A hand-rolled BH is easy to get subtly wrong."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, n)
    mine = np.array(adjust(p, method)["adjusted"])
    reference = scipy_stats.false_discovery_control(
        p, method="bh" if method == "benjamini_hochberg" else "by")
    assert np.abs(mine - reference).max() < 1e-12


def test_bonferroni_and_holm_are_correct_by_hand():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    bonf = adjust(p, "bonferroni")["adjusted"]
    assert bonf == pytest.approx([0.005, 0.04, 0.195, 0.205, 1.0])
    holm = adjust(p, "holm")["adjusted"]
    # Step-down factors 5,4,3,2,1 with enforced monotonicity.
    assert holm == pytest.approx([0.005, 0.032, 0.117, 0.117, 0.9])
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonf)), "Holm must dominate Bonferroni"


def test_adjusted_values_are_monotone_in_the_original_order():
    """The classic BH bug is a cumulative minimum applied in the wrong direction."""
    rng = np.random.default_rng(1)
    p = np.sort(rng.uniform(0, 1, 50))
    for method in PROCEDURES:
        adjusted = adjust(p, method)["adjusted"]
        assert all(adjusted[i] <= adjusted[i + 1] + 1e-12 for i in range(len(adjusted) - 1)), \
            method


def test_by_is_more_conservative_than_bh():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.05, 200)
    bh = adjust(p, "benjamini_hochberg")
    by = adjust(p, "benjamini_yekutieli")
    assert by["n_rejected"] < bh["n_rejected"]
    assert by["dependence_penalty"] > 5.0
    assert "arbitrary dependence" in by["assumption"]
    assert "positive regression dependence" in bh["assumption"]


def test_every_procedure_declares_its_dependence_assumption():
    """A q-value quoted without its assumption is not interpretable."""
    for method in PROCEDURES:
        result = adjust([0.01, 0.5], method)
        assert result["assumption"] == ASSUMPTIONS[method]
        assert result["assumption"]


def test_family_size_can_exceed_the_reported_p_values():
    """Correcting only the survivors of a selection is not a correction at all."""
    result = adjust([0.001, 0.002], "benjamini_hochberg", n_tests=500)
    assert result["n_tests"] == 500
    assert result["warnings"], "the discrepancy must be flagged"
    tight = adjust([0.001, 0.002], "benjamini_hochberg")
    assert result["adjusted"][0] > tight["adjusted"][0]


def test_a_family_smaller_than_its_contents_is_refused():
    with pytest.raises(MultipleComparisonError, match="cannot be smaller"):
        adjust([0.1, 0.2, 0.3], n_tests=2)


def test_non_finite_p_values_are_refused_rather_than_dropped():
    """Silently dropping them shrinks the family and inflates significance."""
    with pytest.raises(MultipleComparisonError, match="NaN or infinite"):
        adjust([0.01, float("nan"), 0.4])


def test_empty_and_out_of_range_input_is_refused():
    with pytest.raises(MultipleComparisonError, match="no p-values"):
        adjust([])
    with pytest.raises(MultipleComparisonError, match=r"\[0, 1\]"):
        adjust([0.5, 1.5])
    with pytest.raises(MultipleComparisonError, match="unknown method"):
        adjust([0.5], method="my_own_procedure")


# ============================================================== the power trap

def test_a_surrogate_p_value_has_a_hard_floor():
    assert resolution_limit(99) == pytest.approx(0.01)
    assert resolution_limit(999) == pytest.approx(0.001)


def test_an_underpowered_surrogate_study_says_so():
    """A study that *cannot* reject must not be mistaken for a clean negative result."""
    power = check_power(99, 500, method="benjamini_yekutieli")
    assert power["can_reject_after_correction"] is False
    assert power["warning"]
    assert "says nothing about the data" in power["warning"]
    assert power["surrogates_required"] > 99


def test_an_adequately_powered_study_reports_no_warning():
    power = check_power(required_surrogates(20) + 10, 20)
    assert power["can_reject_after_correction"] is True
    assert power["warning"] is None


# ============================================================== surrogates

def test_phase_randomisation_preserves_the_spectrum_exactly():
    rng = np.random.default_rng(3)
    for data in (rng.standard_normal(256), rng.standard_normal((48, 48))):
        surrogate = phase_randomise(data, seed=1)
        ref = np.abs(np.fft.fftn(data))
        got = np.abs(np.fft.fftn(surrogate))
        assert np.abs(got - ref).max() / ref.max() < 1e-12
        assert np.isrealobj(surrogate)


def test_phase_randomisation_preserves_mean_and_variance():
    """Regression: forcing the DC phase to 0 flipped the sign of a negative mean."""
    rng = np.random.default_rng(4)
    for data in (rng.standard_normal(256) - 5.0, rng.standard_normal((40, 40)) - 3.0):
        surrogate = phase_randomise(data, seed=2)
        assert abs(surrogate.mean() - data.mean()) < 1e-9
        assert abs(surrogate.var() - data.var()) / data.var() < 1e-12


def test_surrogate_phases_are_uniform_not_merely_antisymmetric():
    """The bug that made the fBm null benchmark fail.

    Antisymmetrising independent uniform phases by averaging gives an antisymmetric field -
    so the surrogate is real and its spectrum is exactly preserved, passing every structural
    check - but the phase density is *triangular, peaked at zero*, biasing every surrogate
    toward the phase-aligned configuration. Measured effect: the excess kurtosis of level-2
    wavelet detail was 90.0 for those surrogates against 0.75 for the field. Preserving the
    spectrum is necessary but not sufficient; the phases must also be uniform.
    """
    rng = np.random.default_rng(5)
    field = rng.standard_normal((64, 64))
    phases = np.angle(np.fft.fft2(phase_randomise(field, seed=7)))
    # Standard deviation of a uniform distribution on [-pi, pi) is pi/sqrt(3) = 1.8138.
    assert abs(float(np.std(phases)) - math.pi / math.sqrt(3.0)) < 0.06
    assert abs(float(np.mean(phases))) < 0.1


def test_aaft_and_iaaft_preserve_the_amplitude_distribution():
    rng = np.random.default_rng(6)
    data = np.exp(rng.standard_normal(256))     # strongly non-Gaussian
    for fn in (aaft, iaaft):
        surrogate = fn(data, seed=1)
        assert np.allclose(np.sort(surrogate.ravel()), np.sort(data.ravel()))


def test_iaaft_preserves_the_spectrum_better_than_aaft():
    """The iteration must earn its cost."""
    rng = np.random.default_rng(7)
    data = np.cumsum(rng.standard_normal(256))
    ref = np.abs(np.fft.fft(data))
    err = {}
    for name, fn in (("aaft", aaft), ("iaaft", iaaft)):
        got = np.abs(np.fft.fft(fn(data, seed=1)))
        err[name] = float(np.abs(got - ref).max() / ref.max())
    assert err["iaaft"] < err["aaft"] / 2.0, err


def test_circular_shift_preserves_the_autocorrelation_exactly():
    """The right null for a lagged claim: only the alignment is destroyed."""
    rng = np.random.default_rng(8)
    data = ar1(rng, 0.8)
    surrogate = circular_shift(data, seed=1)
    assert np.allclose(np.sort(surrogate), np.sort(data))
    assert not np.allclose(surrogate, data)
    assert abs(effective_sample_size(surrogate) - effective_sample_size(data)) \
        / effective_sample_size(data) < 0.35


def test_block_bootstrap_returns_the_right_length_and_reports_its_block():
    rng = np.random.default_rng(9)
    data = ar1(rng, 0.7, n=200)
    surrogate = block_bootstrap(data, seed=1)
    assert surrogate.shape == data.shape
    with pytest.raises(SurrogateError, match="shorter than the series"):
        block_bootstrap(data, block_length=500)


def test_ensemble_is_reproducible_independent_and_declares_what_it_preserves():
    rng = np.random.default_rng(10)
    data = rng.standard_normal(128)
    first = generate(data, "phase_randomise", n=12, seed=5)
    second = generate(data, "phase_randomise", n=12, seed=5)
    assert np.array_equal(first["members"][0], second["members"][0])
    assert len({m.tobytes() for m in first["members"]}) == 12
    assert first["p_value_floor"] == pytest.approx(1 / 13)
    for method in METHODS:
        assert generate(data, method, n=2, seed=1)["preserves"]


def test_unknown_surrogate_method_lists_the_valid_ones():
    with pytest.raises(SurrogateError, match="iaaft"):
        generate(np.zeros(32), method="bootstrapping")


# ============================================================== calibration

def test_surrogate_test_is_calibrated_on_a_true_null():
    """**The property that makes a p-value mean anything.**

    White noise: the null is true by construction, so rejection should occur at close to the
    nominal rate. An early version of the phase-randomisation code gave 0.765 here.
    """
    rng = np.random.default_rng(11)
    p_values = []
    for i in range(200):
        series = rng.standard_normal(256)
        p_values.append(surrogate_test(series, mean_abs_diff, method="phase_randomise",
                                       n_surrogates=99, seed=3000 + i,
                                       alternative="two_sided")["p_value"])
    rate = float((np.array(p_values) <= 0.05).mean())
    # 200 trials give a standard error of ~0.015 on a rate of 0.05.
    assert rate < 0.12, "false-positive rate %.3f is inflated; nominal is 0.05" % rate


def test_surrogate_p_value_uses_the_plus_one_convention():
    """`k/n` would allow p = 0, claiming infinite evidence from a finite ensemble."""
    result = surrogate_p_value(100.0, [1.0, 2.0, 3.0])
    assert result["p_value"] == pytest.approx(1 / 4)
    assert result["p_value_floor"] == pytest.approx(1 / 4)
    assert result["p_value"] > 0


def test_surrogate_p_value_alternatives_and_empty_ensemble():
    null = list(range(10))
    assert surrogate_p_value(20, null, "greater")["p_value"] < \
        surrogate_p_value(20, null, "less")["p_value"]
    with pytest.raises(ValueError, match="empty"):
        surrogate_p_value(1.0, [])
    with pytest.raises(ValueError, match="two_sided"):
        surrogate_p_value(1.0, [1.0], alternative="sideways")


# ============================================================== stationarity gate

def test_stationarity_gate_separates_reliable_from_unreliable_series():
    """Thresholds calibrated against measured false-positive rate, not chosen for neatness.

    Measured surrogate FPR: 0.05 at phi = 0, 0.06 at 0.7, 0.11 at 0.8, 0.16 at 0.85,
    0.39 at 0.95. The gate must pass the first two and flag the rest.
    """
    reliable, unreliable = (0.0, 0.5, 0.7), (0.85, 0.95)
    for phi in reliable:
        rng = np.random.default_rng(int(phi * 100) + 1)
        flagged = sum(not stationarity_check(ar1(rng, phi))["is_stationary"]
                      for _ in range(60)) / 60
        assert flagged < 0.35, "phi=%.2f flagged %.0f%% but its FPR is nominal" % (
            phi, 100 * flagged)
    for phi in unreliable:
        rng = np.random.default_rng(int(phi * 100) + 1)
        flagged = sum(not stationarity_check(ar1(rng, phi))["is_stationary"]
                      for _ in range(60)) / 60
        assert flagged > 0.7, "phi=%.2f flagged only %.0f%% but its FPR is inflated" % (
            phi, 100 * flagged)


def test_a_random_walk_is_always_flagged():
    rng = np.random.default_rng(12)
    for _ in range(20):
        check = stationarity_check(np.cumsum(rng.standard_normal(256)))
        assert not check["is_stationary"]
        assert check["recommendation"] and "difference" in check["recommendation"]


def test_the_threshold_is_where_the_measurement_says_it_is():
    assert 0.70 <= UNIT_ROOT_RHO <= 0.80, (
        "the gate is calibrated to the measured FPR crossover at rho1 ~ 0.75-0.79")


def test_surrogate_test_attaches_the_stationarity_verdict():
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.standard_normal(256))
    result = surrogate_test(walk, mean_abs_diff, n_surrogates=19, seed=1)
    assert result["valid"] is False
    assert result["warnings"]
    assert "anti-conservative" in result["warnings"][0]

    clean = surrogate_test(rng.standard_normal(256), mean_abs_diff, n_surrogates=19, seed=1)
    assert clean["valid"] is True
    assert clean["warnings"] == []


def test_surrogate_test_records_which_null_it_used():
    """"p = 0.004" means different things against different nulls."""
    rng = np.random.default_rng(14)
    result = surrogate_test(rng.standard_normal(128), mean_abs_diff,
                            method="iaaft", n_surrogates=19, seed=1)
    assert result["surrogate_method"] == "iaaft"
    assert result["null_preserves"]
    assert "null_hypothesis" in result


# ============================================================== correlation and ESS

def test_correlation_reports_both_naive_and_corrected_p_values():
    rng = np.random.default_rng(15)
    x = ar1(rng, 0.9, n=200)
    y = ar1(rng, 0.9, n=200)
    result = correlation_test(x, y)
    assert result["n_effective"] < result["n"]
    assert result["p_value"] >= result["p_value_naive"]
    assert any("effective sample size" in a for a in result["assumptions"])


def test_correlation_declines_gracefully_on_degenerate_input():
    assert math.isnan(correlation_test([1.0, 2.0], [1.0, 2.0])["p_value"])
    assert math.isnan(correlation_test([1.0] * 10, list(range(10)))["p_value"])
    with pytest.raises(ValueError, match="same length"):
        correlation_test([1, 2, 3], [1, 2])


def test_effective_sample_size_matches_the_ar1_formula():
    phi = 0.8
    rng = np.random.default_rng(16)
    x = ar1(rng, phi, n=4000)
    predicted = 4000 * (1 - phi ** 2) / (1 + phi ** 2)
    assert abs(effective_sample_size(x) - predicted) / predicted < 0.2


# ============================================================== screening

def test_screening_rejects_everything_on_pure_noise():
    """**The D8 scenario.** 20 pairs of 9-sample noise - the smoke sweep's shape."""
    rng = np.random.default_rng(17)
    tests = []
    for i in range(20):
        a, b = rng.standard_normal(9), rng.standard_normal(9)
        result = correlation_test(a, b, account_for_autocorrelation=False)
        tests.append({"label": "pair_%d" % i, "p_value": result["p_value"],
                      "r": result["r"]})
    big_effect = sum(abs(t["r"]) >= 0.5 for t in tests)
    assert big_effect >= 3, "the old |r| >= 0.5 rule should fire on noise, or this test is vacuous"
    assert screen(tests)["n_significant"] == 0


def test_screening_still_finds_a_real_effect():
    """A correction that rejects everything is not a fix, it is a mute button."""
    rng = np.random.default_rng(18)
    tests = []
    x = np.arange(40, dtype=float)
    real = correlation_test(x, 0.5 * x + rng.standard_normal(40),
                           account_for_autocorrelation=False)
    tests.append({"label": "real", "p_value": real["p_value"], "r": real["r"]})
    for i in range(19):
        a = rng.standard_normal(40)
        noise = correlation_test(x, a, account_for_autocorrelation=False)
        tests.append({"label": "noise_%d" % i, "p_value": noise["p_value"],
                      "r": noise["r"]})
    screened = screen(tests)
    assert screened["n_significant"] == 1
    assert [r["label"] for r in screened["results"] if r["significant"]] == ["real"]


def test_screening_reports_its_procedure_and_handles_empty_input():
    rng = np.random.default_rng(19)
    tests = [{"label": "t%d" % i, "p_value": float(rng.uniform())} for i in range(10)]
    screened = screen(tests, method="benjamini_yekutieli")
    assert screened["correction"]["method"] == "benjamini_yekutieli"
    assert "arbitrary dependence" in screened["correction"]["assumption"]
    empty = screen([])
    assert empty["n_significant"] == 0 and empty["warnings"]


def test_screening_flags_an_underpowered_surrogate_family():
    tests = [{"label": "t%d" % i, "p_value": 0.01, "n_surrogates": 99} for i in range(200)]
    screened = screen(tests)
    assert screened["power_check"] is not None
    assert screened["power_check"]["can_reject_after_correction"] is False
    assert any("cannot produce a significant result" in w for w in screened["warnings"])
