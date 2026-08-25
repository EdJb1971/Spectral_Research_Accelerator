"""What the engine refuses, and what each refusal cost to establish (TG4.2).

The module under test is `src/core/refusal.py`. Two of its three rules exist because the
pipeline TG4.1 shipped produced a false discovery without them, so several tests here build
the record that broke it and check the break as well as the fix - a refusal whose trap is
never shown to spring is a preference with a docstring.

The third, the effective sample size, was already carried by `precedence.py`; what is tested
here is that both numbers travel, because a corrected count with no naive count beside it does
not show that the correction did anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.errors import InvalidParameterError
from src.core.precedence import (
    PrecedenceCandidate,
    Record,
    ScaleSeries,
    admissible_lags,
    choose_candidates,
    lagged_correlation,
    sweep,
)
from src.core.refusal import (
    CALENDAR_HARMONICS,
    CALENDAR_SECONDS,
    MIN_CYCLES_TO_REMOVE,
    REFUSALS,
    REQUIRED_NULLS,
    REQUIRED_REFUSALS,
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    CalendarNotRemovedError,
    EverythingRefusedError,
    Refusal,
    calendar_periods,
    carry_forward,
    explained_by_leakage,
    lag_profile,
    naive_versus_effective,
    refusal_coverage,
    refusal_report,
    refuse_leaked_members,
    remove_calendar,
    require_calendar_removed,
    simultaneous_strength,
)

SURROGATES = 199
LAGS = tuple(range(1, 13))


# ------------------------------------------------------------------------------- helpers


def _red(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    out[0] = rng.standard_normal()
    innovation = math.sqrt(max(1.0 - phi * phi, 1e-12))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovation * rng.standard_normal()
    return out


def _leaked_record(*, seed: int = 3, n: int = 400, phi: float = 0.9,
                   noise: float = 0.15, cadence: float = 1.0) -> Record:
    """One process, read four times. There is no relationship in this record to find.

    This is the shape a redundant transform hands back when a single spatial band is excited:
    every level's series is the same modulation plus a little independent noise, so every
    ordered pair correlates strongly *within* a frame and, because the modulation has memory,
    stays correlated at a lag.
    """
    rng = np.random.default_rng(seed)
    shared = _red(n, phi, rng)
    return Record(name="leaked", cadence_seconds=cadence, series=tuple(
        ScaleSeries("level_%d" % j, shared + noise * rng.standard_normal(n))
        for j in range(1, 5)))


def _planted_record(*, seed: int = 11, n: int = 400, lag: int = 5, phi: float = 0.7,
                    noise: float = 0.6, cadence: float = 1.0) -> Record:
    """`level_1` at t sets `level_4` at t + lag. A real lead, stronger at its lag than at 0."""
    rng = np.random.default_rng(seed)
    driver = _red(n + lag, phi, rng)
    fine = driver[lag:] + noise * rng.standard_normal(n)
    coarse = driver[:-lag] + noise * rng.standard_normal(n)
    return Record(name="planted", cadence_seconds=cadence, series=(
        ScaleSeries("level_1", fine),
        ScaleSeries("level_2", _red(n, phi, rng)),
        ScaleSeries("level_3", _red(n, phi, rng)),
        ScaleSeries("level_4", coarse)))


def _cycle_record(*, seed: int = 7, n: int = 480, period: float = 12.0, offset: int = 3,
                  amplitude: float = 3.0, phi: float = 0.7,
                  cadence: float = 7200.0) -> Record:
    """Two independent bands that both follow one deterministic cycle. Nothing is coupled."""
    rng = np.random.default_rng(seed)
    clock = np.arange(n, dtype=float)
    fine = _red(n, phi, rng) + amplitude * np.sin(2 * np.pi * clock / period)
    coarse = _red(n, phi, rng) + amplitude * np.sin(2 * np.pi * (clock - offset) / period)
    return Record(name="cycle", cadence_seconds=cadence, series=(
        ScaleSeries("level_1", fine),
        ScaleSeries("level_2", _red(n, phi, rng)),
        ScaleSeries("level_4", coarse)))


@pytest.fixture(scope="module")
def leaked() -> Record:
    return _leaked_record()


@pytest.fixture(scope="module")
def planted() -> Record:
    return _planted_record()


@pytest.fixture(scope="module")
def cycle() -> Record:
    return _cycle_record()


# ------------------------------------------------------- 1. the profile, and lag zero


def test_the_profile_starts_at_the_lag_the_family_may_not_contain(planted):
    profile = lag_profile(planted, "level_1", "level_4", max_lag=12)
    assert len(profile.strengths) == 13
    assert profile.at(0) == pytest.approx(
        simultaneous_strength(planted, "level_1", "level_4"))


def test_the_simultaneous_strength_is_symmetric(planted):
    assert simultaneous_strength(planted, "level_1", "level_4") == pytest.approx(
        simultaneous_strength(planted, "level_4", "level_1"))


def test_a_lead_is_not_symmetric(planted):
    forward = lag_profile(planted, "level_1", "level_4", max_lag=12)
    reverse = lag_profile(planted, "level_4", "level_1", max_lag=12)
    assert forward.at(5) > reverse.at(5)


def test_the_planted_lead_peaks_at_the_planted_lag(planted):
    assert lag_profile(planted, "level_1", "level_4", max_lag=12).peak_lag == 5


def test_a_leaked_pair_peaks_at_lag_zero(leaked):
    assert lag_profile(leaked, "level_1", "level_2", max_lag=12).peak_lag == 0


def test_a_leaked_profile_decays_from_zero(leaked):
    profile = lag_profile(leaked, "level_1", "level_2", max_lag=6)
    assert profile.strengths[0] > profile.strengths[1] > profile.strengths[2]


def test_the_profile_covers_lags_the_family_would_refuse(planted):
    """Shape is measured over every lag; admissibility governs hypotheses, not references."""
    admissible = admissible_lags(planted, max_lag=12)
    profile = lag_profile(planted, "level_1", "level_4", max_lag=len(admissible) + 20)
    assert len(profile.strengths) > len(admissible)


def test_a_profile_refuses_a_lag_it_does_not_cover(planted):
    profile = lag_profile(planted, "level_1", "level_4", max_lag=4)
    with pytest.raises(InvalidParameterError):
        profile.at(9)


def test_a_profile_of_no_lags_is_refused(planted):
    with pytest.raises(InvalidParameterError):
        lag_profile(planted, "level_1", "level_4", max_lag=0)


def test_a_constant_band_correlates_with_nothing():
    record = Record(name="flat", series=(
        ScaleSeries("level_1", np.zeros(60)),
        ScaleSeries("level_2", np.arange(60, dtype=float))))
    assert simultaneous_strength(record, "level_1", "level_2") == 0.0


def test_the_profile_describes_itself(planted):
    described = lag_profile(planted, "level_1", "level_4", max_lag=6).describe()
    assert described["pair"] == "level_1>level_4"
    assert described["peak_lag"] == 5
    assert len(described["strengths"]) == 7


# --------------------------------------------------------------- 2. the leakage ceiling


def _candidate(record: Record, driver: str, driven: str, lag: int) -> PrecedenceCandidate:
    r = lagged_correlation(record.array(driver), record.array(driven), lag)
    return PrecedenceCandidate(driver=driver, driven=driven, lag=lag, correlation=r,
                               n_pairs=record.length - lag, n_effective=50.0,
                               p_naive=0.0, p_effective=0.0)


def test_a_real_lead_is_not_explained_by_leakage(planted):
    assert not explained_by_leakage(planted, _candidate(planted, "level_1", "level_4", 5))


def test_every_lag_of_a_leaked_pair_is_explained_by_leakage(leaked):
    for lag in LAGS:
        assert explained_by_leakage(leaked, _candidate(leaked, "level_1", "level_2", lag))


def test_the_ceiling_is_the_simultaneous_correlation(leaked):
    """Not a tuned threshold: `r(k) = r(0) * rho(k)` and `rho <= 1`, so `r(0)` is the sup."""
    ceiling = simultaneous_strength(leaked, "level_1", "level_2")
    profile = lag_profile(leaked, "level_1", "level_2", max_lag=12)
    assert max(profile.strengths[1:]) < ceiling


def test_the_rule_reads_only_the_partition_it_is_given(planted):
    """The ceiling is a training-partition quantity, so a different partition may differ."""
    half = Record(name="half", series=tuple(
        ScaleSeries(s.level, s.values[:200]) for s in planted.series))
    assert simultaneous_strength(half, "level_1", "level_4") != pytest.approx(
        simultaneous_strength(planted, "level_1", "level_4"))


def test_refuse_leaked_members_partitions_the_family(planted):
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    kept, refused = refuse_leaked_members(planted, result.candidates)
    assert len(kept) + len(refused) == result.n_examined
    assert kept and refused


def test_the_planted_member_survives_the_ceiling(planted):
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    kept, _ = refuse_leaked_members(planted, result.candidates)
    assert "level_1>level_4@5" in [c.label for c in kept]


def test_a_leaked_family_is_emptied(leaked):
    result = sweep(leaked, lags=LAGS, n_surrogates=SURROGATES)
    kept, refused = refuse_leaked_members(leaked, result.candidates)
    assert not kept
    assert len(refused) == result.n_examined


def test_the_report_says_what_was_refused_and_why(leaked):
    result = sweep(leaked, lags=LAGS, n_surrogates=SURROGATES)
    report = refusal_report(result, leaked)
    assert report["family_emptied"] is True
    assert report["n_survived"] == 0
    assert report["n_refused_as_leakage"] == result.n_examined
    assert report["strongest_refused"]["label"]


def test_the_report_is_available_when_nothing_was_refused(planted):
    report = refusal_report(sweep(planted, lags=LAGS, n_surrogates=SURROGATES), planted)
    assert report["n_survived"] > 0
    assert report["family_emptied"] is False


# ------------------------------------------------ 3. the false discovery it prevents


def test_the_unguarded_pipeline_ranks_a_leak_first(leaked):
    """The trap, measured: without the ceiling the strongest member of the family is a leak."""
    result = sweep(leaked, lags=LAGS, n_surrogates=SURROGATES)
    assert result.best.strength > 0.5
    assert result.best.lag == min(LAGS)


def test_the_unguarded_pipeline_would_carry_leaks_forward(leaked):
    chosen = choose_candidates(sweep(leaked, lags=LAGS, n_surrogates=SURROGATES))
    assert chosen
    assert all(explained_by_leakage(leaked, c) for c in chosen)


def test_the_guarded_pipeline_carries_nothing_forward(leaked):
    result = sweep(leaked, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(EverythingRefusedError):
        carry_forward(result, leaked)


def test_an_emptied_family_is_not_a_null_result(leaked):
    result = sweep(leaked, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(EverythingRefusedError) as raised:
        carry_forward(result, leaked)
    assert "not a null result" in str(raised.value)


# ---------------------------------------------------- 4. what the clock names


def test_a_second_by_second_record_carries_no_calendar():
    record = _planted_record(cadence=1.0)
    assert calendar_periods(record) == ()


def test_an_hourly_record_carries_a_day():
    record = _planted_record(n=400, cadence=3600.0)
    names = [n for n, _ in calendar_periods(record)]
    assert names == ["diurnal"]


def test_the_day_is_twenty_four_frames_at_hourly_cadence():
    record = _planted_record(n=400, cadence=3600.0)
    assert dict(calendar_periods(record))["diurnal"] == pytest.approx(24.0)


def test_a_two_hourly_record_carries_a_twelve_frame_day():
    record = _planted_record(n=400, cadence=7200.0)
    assert dict(calendar_periods(record))["diurnal"] == pytest.approx(12.0)


def test_a_record_too_short_to_see_a_cycle_twice_does_not_carry_it():
    record = _planted_record(n=30, cadence=3600.0)
    assert calendar_periods(record) == ()


def test_the_threshold_is_two_turns_not_one():
    """A single turn of a cycle is a trend, and subtracting it would remove the record."""
    period_frames = SECONDS_PER_DAY / 3600.0
    just_under = _planted_record(n=int(period_frames * MIN_CYCLES_TO_REMOVE) - 1,
                                 cadence=3600.0)
    just_over = _planted_record(n=int(period_frames * MIN_CYCLES_TO_REMOVE) + 1,
                                cadence=3600.0)
    assert calendar_periods(just_under) == ()
    assert [n for n, _ in calendar_periods(just_over)] == ["diurnal"]


def test_a_long_enough_record_carries_both_cycles():
    """Twelve-hourly frames over four years: a two-frame day and a 730-frame year."""
    record = _planted_record(n=1600, cadence=SECONDS_PER_DAY / 2.0)
    assert [n for n, _ in calendar_periods(record)] == ["diurnal", "annual"]


def test_a_cycle_faster_than_the_sampling_is_not_carried():
    record = _planted_record(n=800, cadence=SECONDS_PER_YEAR / 24.0)
    assert [n for n, _ in calendar_periods(record)] == ["annual"]


def test_the_calendar_is_two_cycles_and_no_others():
    assert [n for n, _ in CALENDAR_SECONDS] == ["diurnal", "annual"]
    assert SECONDS_PER_YEAR == pytest.approx(365.25 * SECONDS_PER_DAY)


def test_the_calendar_is_a_fact_about_the_clock_not_the_values():
    """Two records with the same cadence carry the same cycles whatever they contain."""
    a = calendar_periods(_planted_record(n=400, cadence=3600.0))
    b = calendar_periods(_leaked_record(n=400, cadence=3600.0))
    assert a == b


# ------------------------------------------------------- 5. removing it, once, on train


def test_removal_records_what_it_removed(cycle):
    train, = remove_calendar(cycle)
    assert train.provenance["calendar_removed"] == ["diurnal"]
    assert train.provenance["calendar_harmonics"] == CALENDAR_HARMONICS


def test_removal_leaves_the_partition_identity_alone(cycle):
    train, = remove_calendar(cycle)
    assert train.frames == cycle.frames
    assert train.levels == cycle.levels
    assert train.length == cycle.length


def test_removal_collapses_the_spurious_lead(cycle):
    before = lag_profile(cycle, "level_1", "level_4", max_lag=12)
    train, = remove_calendar(cycle)
    after = lag_profile(train, "level_1", "level_4", max_lag=12)
    assert max(after.strengths[1:]) < 0.5 * max(before.strengths[1:])


def test_the_model_is_fitted_on_train_and_applied_to_the_other_partition(cycle):
    """Rule R6: the held-out partition never sees its own coefficients."""
    train = Record(name="train", cadence_seconds=cycle.cadence_seconds, frames=(0, 300),
                   series=tuple(ScaleSeries(s.level, s.values[:300]) for s in cycle.series))
    held = Record(name="held", cadence_seconds=cycle.cadence_seconds, frames=(300, 480),
                  series=tuple(ScaleSeries(s.level, s.values[300:]) for s in cycle.series))
    train_out, held_out = remove_calendar(train, held)
    assert held_out.provenance["calendar_fitted_on"] == "train"
    assert np.std(held_out.array("level_1")) < np.std(held.array("level_1"))


def test_the_two_partitions_share_one_clock(cycle):
    """Absolute frame numbers, so the phase estimated on train is the phase subtracted."""
    train = Record(name="train", cadence_seconds=cycle.cadence_seconds, frames=(0, 300),
                   series=tuple(ScaleSeries(s.level, s.values[:300]) for s in cycle.series))
    shifted = Record(name="held", cadence_seconds=cycle.cadence_seconds, frames=(300, 480),
                     series=tuple(ScaleSeries(s.level, s.values[300:]) for s in cycle.series))
    mislabelled = Record(name="held", cadence_seconds=cycle.cadence_seconds, frames=(0, 180),
                         series=shifted.series)
    _, honest = remove_calendar(train, shifted)
    _, wrong = remove_calendar(train, mislabelled)
    assert np.std(honest.array("level_1")) < np.std(wrong.array("level_1"))


def test_removing_a_calendar_the_record_cannot_carry_is_refused():
    with pytest.raises(InvalidParameterError):
        remove_calendar(_planted_record(cadence=1.0))


def test_removal_does_not_touch_a_record_without_a_calendar(planted):
    assert calendar_periods(planted) == ()
    require_calendar_removed(planted)          # no calendar, nothing to refuse


def test_the_residual_still_carries_the_relationship_that_was_planted():
    """The removal must cost a real lead nothing: it takes the calendar, not the signal."""
    record = _planted_record(n=480, cadence=7200.0)
    before = lag_profile(record, "level_1", "level_4", max_lag=12)
    anomalies, = remove_calendar(record)
    after = lag_profile(anomalies, "level_1", "level_4", max_lag=12)
    assert after.peak_lag == before.peak_lag == 5
    assert after.at(5) > 0.9 * before.at(5)


# ------------------------------------------------------- 6. machinery, not discipline


def test_a_record_with_its_calendar_in_it_is_refused(cycle):
    with pytest.raises(CalendarNotRemovedError):
        require_calendar_removed(cycle)


def test_the_refusal_names_the_cycle_it_found(cycle):
    with pytest.raises(CalendarNotRemovedError) as raised:
        require_calendar_removed(cycle)
    assert "diurnal" in str(raised.value)
    assert "R11" in str(raised.value)


def test_nothing_can_be_carried_forward_from_a_record_with_a_calendar(cycle):
    result = sweep(cycle, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(CalendarNotRemovedError):
        carry_forward(result, cycle)


def test_the_unguarded_path_would_have_carried_it_forward(cycle):
    """The trap: TG4.1's own selection function has no opinion about the calendar."""
    result = sweep(cycle, lags=LAGS, n_surrogates=SURROGATES)
    assert choose_candidates(result)


def test_a_partly_removed_calendar_is_still_refused():
    record = _planted_record(n=800, cadence=SECONDS_PER_YEAR / 24.0)
    provenance = dict(record.provenance)
    provenance["calendar_removed"] = ["diurnal"]
    partial = Record(name=record.name, series=record.series,
                     cadence_seconds=record.cadence_seconds, frames=record.frames,
                     provenance=provenance)
    with pytest.raises(CalendarNotRemovedError) as raised:
        require_calendar_removed(partial)
    assert "annual" in str(raised.value)


def test_a_removed_calendar_passes(cycle):
    train, = remove_calendar(cycle)
    require_calendar_removed(train)


# ---------------------------------------------- 7. frames are not observations


def test_both_counts_travel(planted):
    counts = naive_versus_effective(sweep(planted, lags=LAGS, n_surrogates=SURROGATES))
    assert counts["n_members"] == 144
    assert counts["naive_significant"] >= counts["effective_significant"]


def _slow_record(phi: float, *, seed: int = 4, n: int = 400) -> Record:
    rng = np.random.default_rng(seed)
    return Record(name="slow", series=tuple(
        ScaleSeries("level_%d" % j, _red(n, phi, rng)) for j in range(1, 5)))


def test_the_gap_grows_with_memory():
    """The size of rule R12 on a record is the gap between the two counts."""
    def gap(phi: float) -> int:
        counts = naive_versus_effective(
            sweep(_slow_record(phi), lags=LAGS, n_surrogates=SURROGATES))
        return counts["naive_significant"] - counts["effective_significant"]
    assert gap(0.9) > gap(0.0)


def test_the_effective_fraction_is_reported(planted):
    counts = naive_versus_effective(sweep(planted, lags=LAGS, n_surrogates=SURROGATES))
    assert 0.0 < counts["median_ess_fraction"] <= 1.0


def test_neither_count_is_offered_as_a_result(planted):
    counts = naive_versus_effective(sweep(planted, lags=LAGS, n_surrogates=SURROGATES))
    assert "neither is a result" in counts["claim_boundary"]


def test_an_independent_slow_record_is_significant_naively_and_not_effectively():
    result = sweep(_slow_record(0.95), lags=LAGS, n_surrogates=SURROGATES)
    counts = naive_versus_effective(result)
    assert counts["naive_significant"] > counts["effective_significant"]


# ------------------------------------------------------------------ 8. the gatekeeper


def test_carry_forward_keeps_one_member_per_pair(planted):
    chosen = carry_forward(sweep(planted, lags=LAGS, n_surrogates=SURROGATES), planted)
    pairs = [(c.driver, c.driven) for c in chosen]
    assert len(pairs) == len(set(pairs))


def test_carry_forward_agrees_with_the_unguarded_choice_when_nothing_is_refused(planted):
    """On a record with a real lead and no leakage the ceiling changes nothing at the top."""
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    assert carry_forward(result, planted)[0].label == choose_candidates(result)[0].label


def test_carry_forward_refuses_a_family_larger_than_the_ensemble_can_pay_for(planted):
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(InvalidParameterError):
        carry_forward(result, planted, n_candidates=10_000)


def test_carry_forward_refuses_an_empty_confirmatory_family(planted):
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(InvalidParameterError):
        carry_forward(result, planted, n_candidates=0)


def test_an_explicit_count_takes_the_ranking_rather_than_the_indistinguishable_set(planted):
    result = sweep(planted, lags=LAGS, n_surrogates=SURROGATES)
    assert len(carry_forward(result, planted, n_candidates=3)) == 3


def test_the_calendar_is_refused_before_the_ranking_is_read(cycle):
    """Order matters: a record with its calendar in it has no trustworthy ranking to rank."""
    result = sweep(cycle, lags=LAGS, n_surrogates=SURROGATES)
    with pytest.raises(CalendarNotRemovedError):
        carry_forward(result, cycle, n_candidates=0)     # would otherwise raise about count


# ------------------------------------------------------------ 9. the register itself


def test_there_are_five_refusals():
    assert len(REFUSALS) == REQUIRED_REFUSALS == 5


def test_at_least_three_of_them_are_nulls():
    assert sum(1 for r in REFUSALS if r.is_null) >= REQUIRED_NULLS >= 3


def test_the_register_covers_the_live_benchmark_registry():
    coverage = refusal_coverage()
    assert coverage["problems"] == []
    assert coverage["covered"] is True


def test_every_refusal_names_a_distinct_benchmark():
    benchmarks = [r.benchmark for r in REFUSALS]
    assert len(set(benchmarks)) == len(benchmarks)


def test_every_refusal_names_a_distinct_gate():
    gates = [r.gate for r in REFUSALS]
    assert len(set(gates)) == len(gates)


def test_every_refusal_names_the_machinery_that_carries_it():
    assert all(r.carried_by for r in REFUSALS)


def test_the_five_are_the_five_the_proposal_named():
    assert {r.name for r in REFUSALS} == {
        "unknown_location", "artificial_correlation", "autocorrelated_repetitions",
        "representation_artefact", "no_relationship"}


def test_a_refusal_describes_itself():
    described = REFUSALS[0].describe()
    assert described["gate"] and described["benchmark"] and described["must"]


def test_coverage_notices_a_gate_a_benchmark_does_not_declare(monkeypatch):
    import src.core.refusal as module
    broken = Refusal(name="invented", must="do something", gate="4F.not_a_gate",
                     benchmark="precedence_null", is_null=True, carried_by=("nothing",))
    monkeypatch.setattr(module, "REFUSALS", REFUSALS[:-1] + (broken,))
    coverage = module.refusal_coverage()
    assert any("4F.not_a_gate" in p for p in coverage["problems"])


def test_coverage_notices_a_benchmark_that_does_not_exist(monkeypatch):
    import src.core.refusal as module
    broken = Refusal(name="invented", must="do something", gate="4F.precedence_null",
                     benchmark="benchmark_that_was_deleted", is_null=True,
                     carried_by=("nothing",))
    monkeypatch.setattr(module, "REFUSALS", REFUSALS[:-1] + (broken,))
    assert any("not registered" in p for p in module.refusal_coverage()["problems"])


def test_coverage_notices_a_null_that_calls_itself_something_else(monkeypatch):
    import src.core.refusal as module
    lying = Refusal(name="no_relationship", must="report nothing",
                    gate="4F.precedence_null", benchmark="precedence_null", is_null=False,
                    carried_by=("nothing",))
    monkeypatch.setattr(module, "REFUSALS", REFUSALS[:-1] + (lying,))
    assert any("calls itself" in p for p in module.refusal_coverage()["problems"])


def test_coverage_notices_a_missing_refusal(monkeypatch):
    import src.core.refusal as module
    monkeypatch.setattr(module, "REFUSALS", REFUSALS[:-1])
    problems = module.refusal_coverage()["problems"]
    assert any("names 5 refusals" in p or "refusals and this register holds" in p
               for p in problems)


def test_coverage_notices_too_few_nulls(monkeypatch):
    import src.core.refusal as module
    weakened = tuple(Refusal(name=r.name, must=r.must, gate=r.gate, benchmark=r.benchmark,
                             is_null=False, carried_by=r.carried_by) for r in REFUSALS)
    monkeypatch.setattr(module, "REFUSALS", weakened)
    assert any("load-bearing" in p for p in module.refusal_coverage()["problems"])
