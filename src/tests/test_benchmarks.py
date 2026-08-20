"""Tests for the Ground-Truth Benchmark Suite, seed discipline and climatology removal.

Roadmap T3.5.17 (standard E7) and T3.5.12 (defect D12, standard E4).

The most important test in this file is `test_no_null_benchmark_ever_fails`. A stage that
reports structure in data containing none is not a bug in one function, it invalidates every
finding the platform has produced - so that check is separated from the general pass/fail
count and given its own failure message.
"""

import math

import numpy as np
import pytest
import torch

from src.analysis_engine.climatology import (
    ClimatologyError,
    HOURS_PER_YEAR,
    harmonic_design_matrix,
    remove_climatology,
)
from src.benchmarks import (
    all_benchmarks,
    benchmarks_gating,
    get_benchmark,
    null_benchmarks,
    run_all,
    format_report,
)
from src.benchmarks.core import (
    Benchmark,
    CheckResult,
    Outcome,
    register_benchmark,
)
from src.benchmarks.runner import failures, null_failures, summarise
from src.benchmarks.seeding import DEFAULT_ROOT_SEED, derive, stable_label_hash
from src.benchmarks.sequences import effective_sample_size, correlation_p_value


# ============================================================== seed discipline

def test_same_label_and_seed_reproduce_exactly():
    a, b = derive("thing", 42), derive("thing", 42)
    assert a == b
    x = torch.randn(64, generator=a.torch_generator(), dtype=torch.float64)
    y = torch.randn(64, generator=b.torch_generator(), dtype=torch.float64)
    assert torch.equal(x, y)
    assert np.array_equal(a.numpy_generator().standard_normal(32),
                          b.numpy_generator().standard_normal(32))


def test_different_labels_give_independent_streams():
    """Without this, "same seed reproduces" would pass for a generator ignoring its seed."""
    a, b = derive("alpha", 42), derive("beta", 42)
    assert a.torch_seed != b.torch_seed
    x = torch.randn(512, generator=a.torch_generator(), dtype=torch.float64)
    y = torch.randn(512, generator=b.torch_generator(), dtype=torch.float64)
    assert not torch.equal(x, y)
    # ...and not merely different: uncorrelated.
    assert abs(float(np.corrcoef(x.numpy(), y.numpy())[0, 1])) < 0.15


def test_different_root_seeds_give_different_streams():
    assert derive("thing", 1).torch_seed != derive("thing", 2).torch_seed


def test_label_hash_is_stable_not_pythons_randomised_hash():
    """`hash()` on a str is salted per process, which breaks cross-session reproducibility.

    Asserting a *literal* value is the point: if the derivation ever changes to something
    process-dependent, this test fails on the second machine it runs on rather than silently
    producing different benchmark data.
    """
    assert stable_label_hash("fbm") == 3620960832
    assert stable_label_hash("") == 0
    assert stable_label_hash("a") != stable_label_hash("b")


def test_seed_bundle_records_its_own_derivation():
    prov = derive("x", 7).to_provenance()
    assert prov["root_seed"] == 7
    assert prov["label"] == "x"
    assert "SeedSequence" in prov["derivation"]


# ============================================================== registry

def test_every_benchmark_declares_gates_and_a_known_answer():
    assert len(all_benchmarks()) >= 8
    for b in all_benchmarks():
        assert b.gates, "%s declares no gate" % b.name
        assert b.kind in ("field", "sequence")
        truth = b.truth()
        assert isinstance(truth, dict) and truth, "%s has an empty known answer" % b.name
        assert b.description


def test_duplicate_benchmark_names_are_refused():
    existing = all_benchmarks()[0]
    with pytest.raises(ValueError, match="already registered"):
        register_benchmark(Benchmark(
            name=existing.name, kind="field", description="dup",
            gates=("x",), build=lambda bundle: None, known_answer=lambda: {}))


def test_unknown_benchmark_name_lists_the_available_ones():
    with pytest.raises(KeyError, match="registered:"):
        get_benchmark("does_not_exist")


def test_benchmarks_are_deterministic_across_builds():
    for name in ("fractional_brownian", "white_noise_field", "pure_noise_sequence"):
        b = get_benchmark(name)
        first, second = b.make(), b.make()
        a = first.stack() if hasattr(first, "stack") else first.data
        c = second.stack() if hasattr(second, "stack") else second.data
        assert torch.equal(a, c), "%s is not reproducible" % name


def test_changing_the_root_seed_changes_the_data():
    b = get_benchmark("fractional_brownian")
    assert not torch.equal(b.make(1).data, b.make(2).data)


def test_null_benchmarks_are_flagged_and_cover_the_key_traps():
    names = {b.name for b in null_benchmarks()}
    assert {"fractional_brownian", "white_noise_field", "pure_noise_sequence",
            "seasonal_diurnal_sequence", "red_noise_sequence"} <= names


def test_gating_lookup_finds_benchmarks_by_stage():
    assert get_benchmark("advected_vortex_sequence") in benchmarks_gating("4D")
    assert get_benchmark("planted_configuration") in benchmarks_gating("4E")
    assert benchmarks_gating("4C")


# ============================================================== the suite itself

@pytest.fixture(scope="module")
def suite_results():
    return run_all()


def test_no_null_benchmark_ever_fails(suite_results):
    """**The false-positive floor.** Read the message before changing anything else.

    A failure here means a stage reported structure in data that was constructed to contain
    none. That is not a local bug: every finding the platform has produced is suspect until
    it is resolved.
    """
    bad = null_failures(suite_results)
    assert not bad, "\n".join(
        "NULL BENCHMARK FAILURE %s / %s: %s" % (n, c.stage, c.detail) for n, c in bad)


def test_the_whole_suite_passes(suite_results):
    bad = failures(suite_results)
    assert not bad, "\n".join("%s / %s: %s" % (n, c.stage, c.detail) for n, c in bad)


def test_pending_gates_are_reported_not_hidden(suite_results):
    """NOT_YET_RUNNABLE must be its own outcome, never folded into PASS.

    If this ever reads zero because the outcome was quietly reclassified, "all green" would
    start meaning "we did not look".
    """
    counts = summarise(suite_results)
    # Was 3 pending; `4C.surrogate_null` became enforceable in T4C.5 when the surrogate
    # machinery landed, so it moved from NOT_YET_RUNNABLE to PASS. That transition is the
    # point of the three-valued outcome: a gate becoming real should change this number.
    assert counts["NOT_YET_RUNNABLE"] >= 2
    assert counts["PASS"] >= 12
    report = format_report(suite_results)
    assert "NOT_YET_RUNNABLE" in report
    assert "Gates defined but not yet enforceable" in report
    for stage in ("4D.tracking", "4E.invariance"):
        assert stage in report
    # ...and the one that graduated must now be a genuine PASS, not silently absent.
    surrogate = [c for c in suite_results["fractional_brownian"]
                 if c.stage == "4C.surrogate_null"]
    assert surrogate and surrogate[0].outcome is Outcome.PASS, (
        "the fBm surrogate-null gate must be enforced, not pending")


def test_a_check_that_crashes_is_a_failure_not_an_error():
    """A benchmark whose check raises must be reported, never swallowed."""
    def exploding(data, truth):
        raise RuntimeError("boom")
    exploding.stage = "test.stage"
    b = Benchmark(name="_tmp_explode", kind="field", description="d", gates=("x",),
                  build=lambda bundle: None, known_answer=lambda: {"a": 1},
                  checks=(exploding,))
    results = b.run()
    assert results[0].outcome is Outcome.FAIL
    assert "boom" in results[0].detail


def test_outcome_ok_treats_pending_as_not_failed_but_report_separates_it():
    assert CheckResult("s", Outcome.PASS, "").ok is True
    assert CheckResult("s", Outcome.NOT_YET_RUNNABLE, "").ok is True
    assert CheckResult("s", Outcome.FAIL, "").ok is False


# ============================================================== known answers

def test_fbm_slope_follows_two_h_plus_one():
    """Ground truth: for a 2D fBm, E(k) ~ k^-(2H+1). Verified across the H range."""
    from src.analysis_engine import spectra
    b = get_benchmark("fractional_brownian")
    for hurst in (0.3, 0.5, 0.7, 0.9):
        field = b.make(hurst=hurst, n=192)
        got = spectra.spectral_slope(field)["beta_energy_1d"]
        assert abs(got - (2 * hurst + 1)) < 0.08, "H=%.1f gave %.4f" % (hurst, got)


def test_white_noise_energy_exponent_is_minus_one_not_zero():
    """Rule R15 in miniature: 'flat' means beta=0 for S(k) and beta=-1 for E(k)."""
    truth = get_benchmark("white_noise_field").truth()
    assert truth["beta_energy_1d"] == -1.0
    assert truth["beta_density_2d"] == 0.0


def test_sinusoid_wavelength_is_recovered_in_kilometres():
    b = get_benchmark("pure_sinusoid")
    for cells in (8, 16, 32):
        results = b.run(wavelength_cells=cells, n=256)
        peak = [r for r in results if r.stage == "4C.scale_signature"][0]
        assert peak.outcome is Outcome.PASS, peak.detail
        assert peak.measured["relative_error"] < 0.1


def test_planted_configuration_geometry_is_what_it_claims():
    b = get_benchmark("planted_configuration")
    truth = b.truth(triangle_side=40.0)
    pts = truth["positions_rowcol"]
    dists = [math.dist(pts[i], pts[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    assert all(abs(d - 40.0) < 1e-6 for d in dists), dists


def test_planted_configuration_transformations_preserve_relative_geometry():
    """The 4E invariance target: translation and rotation must not change the signature."""
    b = get_benchmark("planted_configuration")
    base = b.truth(triangle_side=40.0)
    moved = b.truth(triangle_side=40.0, translation=(30.0, -20.0), rotation_deg=37.0)
    assert abs(moved["pairwise_distance_cells"] - base["pairwise_distance_cells"]) < 1e-9
    # ...while a genuine rescale must be visible, since scale is a real difference.
    scaled = b.truth(triangle_side=40.0, scale_factor=2.0)
    assert abs(scaled["pairwise_distance_cells"] - 80.0) < 1e-9
    assert scaled["scale_ratio_vs_reference"] == 2.0


def test_vortex_trajectory_truth_matches_the_generated_field():
    b = get_benchmark("advected_vortex_sequence")
    results = {r.stage: r for r in b.run()}
    assert results["4D.position"].outcome is Outcome.PASS
    assert results["4D.position"].measured["worst_error_cells"] < 1.0
    assert results["4D.scale_evolution"].measured["late_centroid"] > \
        results["4D.scale_evolution"].measured["early_centroid"]


def test_cascade_recovers_the_injected_lag():
    b = get_benchmark("coupled_cascade_sequence")
    for lag in (4, 6, 9):
        res = [r for r in b.run(lag=lag, steps=80) if r.stage == "4C.cross_scale"][0]
        assert res.outcome is Outcome.PASS, res.detail
        assert abs(res.measured["recovered_lag"] - lag) <= 1


# ============================================================== R12, effective sample size

def test_effective_sample_size_matches_the_ar1_formula():
    phi = 0.8
    rng = np.random.default_rng(0)
    scale = math.sqrt(1 - phi ** 2)
    x = np.empty(4000)
    x[0] = rng.standard_normal()
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + scale * rng.standard_normal()
    predicted = len(x) * (1 - phi ** 2) / (1 + phi ** 2)
    assert abs(effective_sample_size(x) - predicted) / predicted < 0.2


def test_independent_series_are_not_deflated():
    """Positive control: white noise must keep essentially its full sample size."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    assert effective_sample_size(x) > 0.85 * len(x)


def test_naive_significance_over_rejects_on_autocorrelated_data():
    """The R12 trap, measured. This is why frames cannot be counted as samples."""
    res = [r for r in get_benchmark("red_noise_sequence").run()
           if r.stage == "4C.r12_effective_sample_size"][0]
    assert res.outcome is Outcome.PASS, res.detail
    assert res.measured["naive_false_positive_rate"] > 0.15
    assert res.measured["ess_false_positive_rate"] < 0.12


# ============================================================== climatology (R11 + R6)

def test_harmonic_declimatology_beats_a_bin_climatology_on_a_short_record():
    """The measured reason harmonic regression is used - a factor of ~900 here."""
    res = [r for r in get_benchmark("seasonal_diurnal_sequence").run()
           if r.stage == "4C.r11_anomaly"][0]
    assert res.outcome is Outcome.PASS, res.detail
    assert res.measured["harmonic_ratio"] < 0.01
    assert res.measured["binned_ratio"] > 0.1
    assert res.measured["binned_ratio"] > 50 * res.measured["harmonic_ratio"]


def test_climatology_can_be_fitted_on_training_frames_only():
    res = [r for r in get_benchmark("seasonal_diurnal_sequence").run()
           if r.stage == "4C.r11_split_aware"][0]
    assert res.outcome is Outcome.PASS, res.detail
    assert res.measured["fitted_on_all"] is False


def test_raw_cycles_look_like_a_strong_finding():
    """Positive control: without this, the anomaly test would prove nothing."""
    res = [r for r in get_benchmark("seasonal_diurnal_sequence").run()
           if r.stage == "4C.r11_raw_is_deceptive"][0]
    assert abs(res.measured["raw_correlation"]) > 0.5
    assert res.measured["raw_p"] < 1e-6


def test_design_matrix_columns_are_labelled():
    t = np.arange(200, dtype=float)
    design, labels = harmonic_design_matrix(t, (24.0, HOURS_PER_YEAR), n_harmonics=2)
    assert design.shape == (200, 1 + 2 * 2 * 2)
    assert labels[0] == "intercept"
    assert sum("24" in l for l in labels) == 4


def test_climatology_refuses_an_underdetermined_fit_with_a_fix():
    stack = torch.randn(6, 8, 8, dtype=torch.float64)
    with pytest.raises(ClimatologyError, match="reduce n_harmonics"):
        remove_climatology(stack, np.arange(6, dtype=float), n_harmonics=2)


def test_climatology_warns_when_the_record_is_shorter_than_the_period():
    stack = torch.randn(120, 4, 4, dtype=torch.float64)
    out = remove_climatology(stack, np.arange(120, dtype=float) * 6.0, n_harmonics=1)
    assert out["warnings"], "a 30-day record cannot observe an annual cycle silently"
    assert any("extrapolation" in w for w in out["warnings"])


def test_climatology_rejects_mismatched_times():
    with pytest.raises(ClimatologyError, match="has 5 entries but the stack has 10"):
        remove_climatology(torch.randn(10, 4, 4), np.arange(5, dtype=float))


def test_climatology_rejects_a_non_sequence():
    with pytest.raises(ClimatologyError, match="time, height, width"):
        remove_climatology(torch.randn(8, 8), np.arange(8, dtype=float))


def test_climatology_removes_a_pure_cycle_essentially_completely():
    """Analytic case: a field that IS the climatology must leave nothing behind."""
    t = np.arange(400, dtype=float) * 6.0
    pattern = torch.randn(6, 6, dtype=torch.float64)
    signal = torch.stack([
        pattern * (3.0 * math.sin(2 * math.pi * h / 24.0)
                   + 5.0 * math.cos(2 * math.pi * h / HOURS_PER_YEAR)) for h in t])
    out = remove_climatology(signal, t, n_harmonics=1)
    assert out["residual_variance_ratio"] < 1e-20
    assert out["variance_explained_by_climatology"] > 0.999999


def test_climatology_preserves_a_signal_that_is_not_a_cycle():
    """Guard against over-fitting: weather must survive the anomaly step.

    A climatology with too many harmonics eats the very signal it is meant to expose, so
    this asserts a non-cyclic perturbation is still there afterwards.
    """
    t = np.arange(400, dtype=float) * 6.0
    pattern = torch.randn(6, 6, dtype=torch.float64)
    cycle = torch.stack([pattern * 5.0 * math.sin(2 * math.pi * h / 24.0) for h in t])
    weather = torch.zeros_like(cycle)
    weather[150:170] = 4.0
    out = remove_climatology(cycle + weather, t, n_harmonics=2)
    recovered = out["anomalies"][150:170].mean()
    assert float(recovered) > 3.0, "the anomaly step absorbed the non-cyclic signal"


# ============================================================== determinism regressions

def test_climatology_solve_is_deterministic_regardless_of_prior_work():
    """Regression for a defect that only appeared when other work ran first.

    The harmonic basis for a short record is near rank-deficient (condition number 8.4e13
    before column normalisation), and `torch.linalg.lstsq`'s default driver made a rank
    decision that flipped depending on prior BLAS state: the same data and seed gave a
    residual of 0.000172 in one ordering and 0.157639 in another. The rank decision is now
    ours and explicit.

    This test deliberately does unrelated tensor work first, which is what triggered it.
    """
    b = get_benchmark("seasonal_diurnal_sequence")
    seq = b.make()
    first = remove_climatology(seq.stack(), seq.times_hours, n_harmonics=2)

    for _ in range(3):
        torch.linalg.svd(torch.randn(300, 40, dtype=torch.float64))
        torch.linalg.lstsq(torch.randn(200, 12, dtype=torch.float64),
                           torch.randn(200, 5, dtype=torch.float64))
    again = remove_climatology(seq.stack(), seq.times_hours, n_harmonics=2)

    assert torch.equal(first["anomalies"], again["anomalies"])
    assert first["residual_variance_ratio"] == again["residual_variance_ratio"]


def test_column_normalisation_keeps_the_design_well_conditioned():
    """The conditioning fix, measured: 8.4e13 -> ~1.9e4 on the same basis."""
    b = get_benchmark("seasonal_diurnal_sequence")
    seq = b.make()
    raw, _ = harmonic_design_matrix(seq.times_hours, n_harmonics=2)
    assert float(torch.linalg.cond(raw)) > 1e10
    out = remove_climatology(seq.stack(), seq.times_hours, n_harmonics=2)
    assert out["condition_number"] < 1e6
    assert out["effective_rank"] == out["n_parameters"]


def test_rank_deficient_basis_is_reported_not_silently_pseudo_inverted():
    """Asking for a cycle the record cannot resolve must say so."""
    t = np.arange(90, dtype=float) * 6.0
    stack = torch.randn(90, 4, 4, dtype=torch.float64)
    out = remove_climatology(stack, t, periods_hours=(24.0, 24.0), n_harmonics=1)
    assert out["effective_rank"] < out["n_parameters"]
    assert any("rank-deficient" in w for w in out["warnings"])
    assert any("not identifiable" in w for w in out["warnings"])


# ============================================================== D12 seed discipline

def test_add_noise_is_reproducible_when_seeded():
    from src.physical_core.field import PhysicalField
    from src.synthetic_generator.perturbation import PerturbationEngine

    field = PhysicalField(torch.randn(24, 24, generator=torch.Generator().manual_seed(3),
                                      dtype=torch.float64))
    for kind in ("gaussian", "uniform", "salt_pepper"):
        a = PerturbationEngine.add_noise(field, noise_type=kind, seed=11)
        b = PerturbationEngine.add_noise(field, noise_type=kind, seed=11)
        c = PerturbationEngine.add_noise(field, noise_type=kind, seed=12)
        assert torch.equal(a.data, b.data), "%s not reproducible" % kind
        assert not torch.equal(a.data, c.data), "%s ignores its seed" % kind
        assert a.metadata["seeded"] is True
        assert a.metadata["seed"] == 11


def test_unseeded_noise_is_labelled_as_unreproducible():
    """An unreproducible run must be visibly different from a reproducible one."""
    from src.physical_core.field import PhysicalField
    from src.synthetic_generator.perturbation import PerturbationEngine

    out = PerturbationEngine.add_noise(PhysicalField(torch.randn(8, 8)))
    assert out.metadata["seeded"] is False
    assert out.metadata["seed"] is None


def test_add_noise_accepts_a_threaded_generator_and_refuses_two_sources():
    from src.physical_core.field import PhysicalField
    from src.synthetic_generator.perturbation import PerturbationEngine

    field = PhysicalField(torch.randn(8, 8, dtype=torch.float64))
    g1 = torch.Generator(); g1.manual_seed(5)
    g2 = torch.Generator(); g2.manual_seed(5)
    assert torch.equal(PerturbationEngine.add_noise(field, generator=g1).data,
                       PerturbationEngine.add_noise(field, generator=g2).data)
    with pytest.raises(ValueError, match="not both"):
        PerturbationEngine.add_noise(field, seed=1, generator=g1)


def test_add_noise_preserves_the_grid():
    """A perturbed field must not silently lose its physical metric (D13)."""
    from src.physical_core.field import PhysicalField
    from src.physical_core.grid import GridSpec
    from src.synthetic_generator.perturbation import PerturbationEngine

    grid = GridSpec.latlon((16, 16), 50.0, -0.25, 0.0, 0.25, variable_units="K")
    field = PhysicalField(torch.randn(16, 16, dtype=torch.float64), grid=grid)
    assert PerturbationEngine.add_noise(field, seed=1).grid == grid


def test_add_noise_rejects_an_unknown_type_with_the_valid_options():
    from src.physical_core.field import PhysicalField
    from src.synthetic_generator.perturbation import PerturbationEngine
    with pytest.raises(ValueError, match="salt_pepper"):
        PerturbationEngine.add_noise(PhysicalField(torch.randn(8, 8)), noise_type="pink")
