"""What recovering a relationship nobody pointed at costs, and what it refuses (TG4.1).

The module under test is `src/core/precedence.py`. Its claim is narrow and worth stating
plainly: the existing 4C cascade check recovers a planted lag *after being told which two
bands to look at*, and this one is not told. Everything expensive here follows from that -
the family, the selection-aware null, the surrogate that keeps the autocorrelation, and the
control that says which bands the decomposition can separate at all.

Several tests measure rather than assert. Where a design decision was made because the
alternative is wrong, the test computes both and shows the difference, because a decision
whose consequence is never measured is a preference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.errors import InvalidParameterError
from src.core.family import SearchAxis, SearchSpecification, SearchTerm
from src.core.precedence import (
    DEFAULT_SURROGATE,
    MIN_ADMISSIBLE_LAG,
    MIN_FRAMES,
    PERMUTED,
    BasisCoupling,
    EmbargoTooShortError,
    LagInadmissibleError,
    PrecedenceCandidate,
    Record,
    ScaleSeries,
    ShiftInfeasibleError,
    admissible_lags,
    admissible_pairs,
    affordable_member_count,
    autocorrelation_time,
    choose_candidates,
    circular_shift,
    confirm_precedence,
    confirmatory_specification,
    deduplicate_candidates,
    freeze_precedence,
    indistinguishable_from_best,
    lagged_correlation,
    measure_basis_coupling,
    ordered_pairs_of,
    pair_label,
    permuted_series,
    precedence_p_value,
    precedence_search_specification,
    precedence_strength,
    record_memory,
    report_precedence_generation,
    split_with_embargo,
    sweep,
)
from src.core.preregistration import HeldOutLedger, PartitionMismatchError

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


def _record(*, coupling: float = 1.0, lag: int = 5, phi: float = 0.7, noise: float = 0.6,
            seed: int = 11, n: int = 400, name: str = "probe") -> Record:
    """Four bands. `level_1` at t sets `level_4` at t + lag, unless `coupling` is zero."""
    rng = np.random.default_rng(seed)
    driver = _red(n + lag, phi, rng)
    independent = _red(n + lag, phi, rng)
    fine = driver[lag:] + noise * rng.standard_normal(n)
    mixed = (coupling * driver[:-lag]
             + math.sqrt(max(1.0 - coupling ** 2, 0.0)) * independent[:-lag])
    coarse = mixed + noise * rng.standard_normal(n)
    return Record(name=name, series=(
        ScaleSeries("level_1", fine),
        ScaleSeries("level_2", _red(n, phi, rng)),
        ScaleSeries("level_3", _red(n, phi, rng)),
        ScaleSeries("level_4", coarse)))


def _leaky_record(*, seed: int = 5, n: int = 300) -> Record:
    """A control: no temporal structure at all, and two of its three bands are one band.

    White in time on purpose - that is what makes it a control. Anything it relates within a
    frame is the decomposition talking about itself, and anything it relates across frames is
    the floor that talking sits on.
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n)
    return Record(name="leaky", series=(
        ScaleSeries("level_1", shared + 0.05 * rng.standard_normal(n)),
        ScaleSeries("level_2", shared + 0.05 * rng.standard_normal(n)),
        ScaleSeries("level_3", rng.standard_normal(n))))


@pytest.fixture(scope="module")
def planted() -> Record:
    return _record(coupling=1.0)


@pytest.fixture(scope="module")
def empty() -> Record:
    return _record(coupling=0.0, seed=23)


@pytest.fixture(scope="module")
def swept(planted: Record):
    return sweep(planted, lags=LAGS, n_surrogates=SURROGATES)


# ------------------------------------------------- 1. what the search costs to declare


def test_the_family_is_pairs_times_lags(planted):
    spec = precedence_search_specification(planted, lags=LAGS, n_surrogates=SURROGATES)
    assert spec.family_size == 4 * 3 * len(LAGS) == 144


def test_the_pair_axis_prices_the_family_exactly_as_tg31s_ordered_pairs_would(planted):
    """The declared pair axis is a different shape from `ordered_pairs`, not a different family."""
    mine = precedence_search_specification(planted, lags=LAGS, n_surrogates=SURROGATES)
    theirs = SearchSpecification(
        terms=(SearchTerm("ordered_pairs", (SearchAxis("band", planted.levels),)),
               SearchTerm("product", (SearchAxis("lag", LAGS),))),
        n_surrogates=SURROGATES)
    assert mine.family_size == theirs.family_size
    assert [tuple(m) for m in mine.enumerate_family()] == [
        (pair_label((a, b)), lag) for a, b, lag in theirs.enumerate_family()]


def test_being_told_the_band_pair_is_a_family_of_one(planted):
    """The quantity this slice exists to make visible."""
    told = SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("lag", LAGS),)),), n_surrogates=SURROGATES)
    untold = precedence_search_specification(planted, lags=LAGS, n_surrogates=SURROGATES)
    assert untold.family_size == 12 * told.family_size


def test_the_family_is_unaffordable_in_one_stage(swept):
    account = swept.specification.account()
    assert not account.affordable
    assert account.surrogates_required == 15985
    assert not swept.affordable_here


def test_a_bigger_study_is_worse_not_better(planted):
    rng = np.random.default_rng(3)
    wide = Record(name="wide", series=tuple(
        ScaleSeries("level_%d" % i, _red(400, 0.7, rng)) for i in range(1, 9)))
    spec = precedence_search_specification(wide, lags=tuple(range(1, 25)),
                                           n_surrogates=SURROGATES)
    assert spec.family_size == 8 * 7 * 24 == 1344
    assert spec.account().surrogates_required == 209153


def test_the_sweep_measures_exactly_the_declared_members_in_order(swept):
    assert tuple(c.label for c in swept.candidates) == swept.specification.labels()
    assert swept.n_examined == swept.specification.family_size


def test_a_sweep_that_measured_something_else_is_refused(planted, monkeypatch):
    """A count check would pass here - same size, different search; the labels are compared."""
    import src.core.precedence as module
    declared = module.precedence_search_specification

    def reversed_declaration(record, *, pairs=None, **kwargs):
        ordered = tuple(pairs) if pairs is not None else module.ordered_pairs_of(record.levels)
        return declared(record, pairs=tuple(reversed(ordered)), **kwargs)

    monkeypatch.setattr(module, "precedence_search_specification", reversed_declaration)
    with pytest.raises(InvalidParameterError, match="did not happen"):
        sweep(planted, lags=LAGS, n_surrogates=SURROGATES)


def test_a_control_with_dynamics_of_its_own_is_not_a_control(planted):
    """Its floor would be the record's memory rather than the basis's noise."""
    from src.core.precedence import ControlNotStaticError
    with pytest.raises(ControlNotStaticError, match="no temporal structure"):
        measure_basis_coupling(planted, lags=LAGS)


def test_a_member_names_both_bands_and_the_lag(swept):
    assert swept.member("level_1>level_4@5").driver == "level_1"
    assert swept.member("level_1>level_4@5").lag == 5


# -------------------------------------------------------- 2. what a lag is allowed to be


def test_lag_zero_is_not_precedence(planted):
    with pytest.raises(LagInadmissibleError, match="not a lead"):
        admissible_lags(planted, max_lag=6, minimum_lag=0)
    with pytest.raises(LagInadmissibleError):
        precedence_search_specification(planted, lags=(0, 1), n_surrogates=SURROGATES)


def test_the_admissible_lags_stop_where_the_data_does(planted):
    lags = admissible_lags(planted, max_lag=12)
    assert lags == LAGS
    assert lags[0] == MIN_ADMISSIBLE_LAG


def test_a_lag_that_leaves_too_few_effective_pairs_is_dropped(planted):
    short = Record(name="short", series=tuple(
        ScaleSeries(s.level, s.values[:40]) for s in planted.series))
    assert max(admissible_lags(short, max_lag=39)) < 39


def test_a_record_too_slow_to_carry_any_lagged_claim_is_refused():
    """A near-unit-root record produces a refusal here, not a weak result."""
    rng = np.random.default_rng(4)
    slow = Record(name="slow", series=tuple(
        ScaleSeries("level_%d" % i, _red(360, 0.98, rng)) for i in range(1, 5)))
    assert record_memory(slow) > 20
    with pytest.raises(LagInadmissibleError, match="no admissible member"):
        admissible_lags(slow, max_lag=12)


def test_a_lag_declared_twice_is_refused(planted):
    with pytest.raises(InvalidParameterError, match="counted"):
        precedence_search_specification(planted, lags=(1, 2, 2), n_surrogates=SURROGATES)


def test_a_band_lagged_against_itself_is_not_a_relationship(planted):
    with pytest.raises(InvalidParameterError, match="autocorrelation"):
        precedence_search_specification(planted, lags=LAGS, n_surrogates=SURROGATES,
                                        pairs=(("level_1", "level_1"),))


def test_the_memory_and_the_effective_sample_size_are_the_same_number(planted):
    """One story about how much this series remembers, not two."""
    rng = np.random.default_rng(9)
    series = _red(2000, 0.8, rng)
    assert autocorrelation_time(series) == pytest.approx(9.0, rel=0.25)


# --------------------------------------------- 3. what the basis relates to what


def test_the_basis_control_finds_the_bands_it_cannot_separate():
    leaky = _leaky_record()
    basis = measure_basis_coupling(leaky, lags=LAGS)
    assert basis.couples("level_1", "level_2")
    assert not basis.couples("level_1", "level_3")


def test_the_limit_is_read_off_the_control_rather_than_chosen():
    basis = measure_basis_coupling(_leaky_record(), lags=LAGS)
    assert 0.0 < basis.floor < 0.5
    assert basis.simultaneous[pair_label(("level_1", "level_2"))] > 0.9


def test_a_coupled_pair_is_excluded_from_the_family():
    leaky = _leaky_record()
    basis = measure_basis_coupling(leaky, lags=LAGS)
    pairs = admissible_pairs(leaky, coupling=basis)
    assert ("level_1", "level_2") not in pairs
    assert ("level_1", "level_3") in pairs
    assert len(pairs) < len(ordered_pairs_of(leaky.levels))


def test_excluding_pairs_shrinks_the_priced_family_and_nothing_else():
    leaky = _leaky_record()
    basis = measure_basis_coupling(leaky, lags=LAGS)
    narrow = precedence_search_specification(
        leaky, lags=LAGS, n_surrogates=SURROGATES, pairs=admissible_pairs(leaky, coupling=basis))
    wide = precedence_search_specification(leaky, lags=LAGS, n_surrogates=SURROGATES)
    assert narrow.family_size < wide.family_size
    assert set(narrow.labels()) < set(wide.labels())


def test_a_basis_that_separates_nothing_is_refused_rather_than_thresholded():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal(200)
    one_band = Record(name="one", series=tuple(
        ScaleSeries("level_%d" % i, shared + 0.01 * rng.standard_normal(200))
        for i in range(1, 4)))
    basis = measure_basis_coupling(one_band, lags=LAGS)
    with pytest.raises(InvalidParameterError, match="different transform"):
        admissible_pairs(one_band, coupling=basis)


def test_the_control_reads_no_data_from_the_record_under_test():
    """The narrowing is preregistered because it comes from somewhere else entirely."""
    basis = measure_basis_coupling(_leaky_record(), lags=LAGS)
    assert basis.control == "leaky"
    assert "claim_boundary" in basis.describe()
    assert set(basis.describe()["coupled_pairs"]) <= set(basis.simultaneous)


def test_the_coupling_lives_at_lag_zero_which_the_family_may_not_contain():
    """The one lag precedence excludes is exactly the one that identifies a redundant band."""
    leaky = _leaky_record()
    basis = measure_basis_coupling(leaky, lags=LAGS)
    simultaneous = basis.simultaneous[pair_label(("level_1", "level_2"))]
    assert simultaneous > basis.floor
    assert MIN_ADMISSIBLE_LAG > 0


# ----------------------------------------------------------- 4. what a surrogate keeps


def test_a_circular_shift_keeps_every_value_and_the_autocorrelation(planted):
    rng = np.random.default_rng(1)
    values = planted.array("level_1")
    shifted = circular_shift(values, rng=rng, minimum_shift=record_memory(planted))
    assert sorted(shifted) == pytest.approx(sorted(values))
    assert autocorrelation_time(shifted) == pytest.approx(autocorrelation_time(values),
                                                          rel=0.15)


def test_a_shuffle_keeps_the_values_and_destroys_the_memory(planted):
    rng = np.random.default_rng(1)
    values = planted.array("level_1")
    shuffled = permuted_series(values, rng=rng)
    assert sorted(shuffled) == pytest.approx(sorted(values))
    assert autocorrelation_time(shuffled) < 0.4 * autocorrelation_time(values)


def test_the_shift_is_drawn_clear_of_the_series_own_memory(planted):
    rng = np.random.default_rng(7)
    values = planted.array("level_1")
    for _ in range(50):
        shifted = circular_shift(values, rng=rng, minimum_shift=20)
        assert not np.allclose(shifted[:20], values[:20])


def test_a_record_too_short_to_shift_clear_is_refused():
    with pytest.raises(ShiftInfeasibleError, match="longer record"):
        circular_shift(np.arange(30, dtype=float), rng=np.random.default_rng(0),
                       minimum_shift=20)


@pytest.mark.parametrize("phi,expected_gap", [(0.0, False), (0.9, True)])
def test_the_shuffle_null_is_wrong_by_exactly_the_memory_it_discards(phi, expected_gap):
    """Measured, not asserted: the error grows with the autocorrelation the shuffle removes."""
    shift_p, perm_p = [], []
    for seed in range(40, 50):
        rng = np.random.default_rng(seed)
        rec = Record(name="null", series=(
            ScaleSeries("level_1", _red(400, phi, rng)),
            ScaleSeries("level_2", _red(400, phi, rng))))
        shift_p.append(precedence_p_value(rec, driver="level_1", driven="level_2", lag=5,
                                          n_surrogates=SURROGATES, seed=3).p_value)
        perm_p.append(precedence_p_value(rec, driver="level_1", driven="level_2", lag=5,
                                         n_surrogates=SURROGATES, seed=3,
                                         surrogate=PERMUTED).p_value)
    shift, perm = float(np.median(shift_p)), float(np.median(perm_p))
    if expected_gap:
        assert perm < 0.6 * shift
    else:
        assert perm == pytest.approx(shift, abs=0.25)


def test_an_unregistered_surrogate_is_refused(planted):
    with pytest.raises(InvalidParameterError):
        precedence_p_value(planted, driver="level_1", driven="level_4", lag=5,
                           n_surrogates=9, seed=1, surrogate="whatever")


# ---------------------------------------------------- 5. the lag was chosen by the data


def test_the_winner_of_a_sweep_is_not_one_correlation(empty):
    """On a record with nothing planted, a single-lag null calls the sweep's winner real."""
    result = sweep(empty, lags=LAGS, n_surrogates=SURROGATES)
    best = result.best
    single = precedence_p_value(empty, driver=best.driver, driven=best.driven, lag=best.lag,
                                n_surrogates=SURROGATES, seed=5).p_value
    maximum = precedence_p_value(empty, driver=best.driver, driven=best.driven, lag=best.lag,
                                 n_surrogates=SURROGATES, seed=5, selected_over=LAGS).p_value
    assert single < maximum


def test_the_null_of_a_maximum_must_contain_the_statistic_being_reported(planted):
    with pytest.raises(InvalidParameterError, match="did not come from"):
        precedence_p_value(planted, driver="level_1", driven="level_4", lag=5,
                           n_surrogates=9, seed=1, selected_over=(7, 8))


def test_a_frozen_lag_needs_no_maximum(planted):
    """Which is why the confirmation is affordable: the selection happened on the other half."""
    evidence = precedence_p_value(planted, driver="level_1", driven="level_4", lag=5,
                                  n_surrogates=SURROGATES, seed=5)
    assert evidence.selected_over == ()


def test_pricing_the_lag_choice_is_necessary_and_not_sufficient(empty):
    """It prices the choice of lag; the choice of band pair is paid for by the held-out stage."""
    result = sweep(empty, lags=LAGS, n_surrogates=SURROGATES)
    best = result.best
    maximum = precedence_p_value(empty, driver=best.driver, driven=best.driven, lag=best.lag,
                                 n_surrogates=SURROGATES, seed=5, selected_over=LAGS)
    assert maximum.selected_over == LAGS
    assert result.specification.family_size > len(LAGS)


# --------------------------------------------------------- 6. frames are not samples


def test_the_effective_sample_size_is_well_below_the_frame_count(swept):
    """Measured here at 262 effective pairs out of 395 frames - two thirds, at phi = 0.7."""
    member = swept.member("level_1>level_4@5")
    assert member.n_effective < 0.75 * member.n_pairs


def test_both_p_values_travel_together(swept):
    """The ratio is how much serial dependence was inflating the result (rule R12)."""
    member = swept.member("level_1>level_4@5")
    assert member.p_naive < member.p_effective


def test_a_lag_costs_data_and_the_record_says_so(swept):
    near = swept.member("level_1>level_4@1")
    far = swept.member("level_1>level_4@12")
    assert far.n_pairs == near.n_pairs - 11


# ------------------------------------------------- 7. generate, freeze, confirm


def test_one_relationship_enters_the_ranking_once_per_lag(swept):
    """Neighbouring lags of one relationship are one hypothesis wearing several labels."""
    top = swept.ranked[:4]
    assert len({(c.driver, c.driven) for c in top}) < len(top)
    distinct = deduplicate_candidates(swept.ranked)
    assert len({(c.driver, c.driven) for c in distinct}) == len(distinct)
    assert len(distinct) == len(swept.pairs)


def test_the_frozen_set_is_what_the_training_data_could_not_tell_apart(swept):
    chosen = choose_candidates(swept)
    assert chosen[0].label == "level_1>level_4@5"
    assert set(chosen) <= set(indistinguishable_from_best(swept))


def test_a_record_with_nothing_planted_freezes_more_than_one_candidate(empty):
    """When nothing stands out, several members are undistinguished - and none confirm."""
    result = sweep(empty, lags=LAGS, n_surrogates=SURROGATES)
    assert len(choose_candidates(result)) > 1


def test_more_members_than_the_ensemble_can_reject_one_of_is_refused(swept):
    ceiling = affordable_member_count(SURROGATES)
    with pytest.raises(InvalidParameterError, match="cannot pay for"):
        choose_candidates(swept, n_candidates=ceiling + 1)


def test_a_confirmatory_family_of_none_is_not_a_null_result(swept):
    with pytest.raises(InvalidParameterError, match="not the same as a null result"):
        choose_candidates(swept, n_candidates=0)


def test_the_generation_report_produces_candidates_and_no_claims(planted, swept):
    train, _ = split_with_embargo(planted, fraction=0.6)
    report = report_precedence_generation(swept, train=train.identity())
    assert report["stage"] == "generate"
    assert "no p-value here is evidence" in report["claim_boundary"].lower()


def test_mining_the_held_out_partition_is_refused(planted, swept):
    _, held_out = split_with_embargo(planted, fraction=0.6)
    with pytest.raises(InvalidParameterError, match="held-out"):
        report_precedence_generation(swept, train=held_out.identity())


def test_the_lag_is_part_of_the_name_a_member_is_frozen_under(swept):
    chosen = choose_candidates(swept)
    spec = confirmatory_specification(chosen, n_surrogates=SURROGATES)
    assert all("@" in label for label in spec.labels())


def test_a_held_out_partition_cut_too_close_is_not_sealed(planted, swept):
    _, close = split_with_embargo(planted, fraction=0.6, embargo=0)
    with pytest.raises(EmbargoTooShortError, match="declared constraint"):
        freeze_precedence(swept, held_out=close.identity(), sealed_at="2026-08-25T00:00Z",
                          n_surrogates=SURROGATES)


def test_the_split_records_both_the_embargo_used_and_the_one_recommended(planted):
    _, held_out = split_with_embargo(planted, fraction=0.6, embargo=3)
    assert held_out.provenance["embargo_frames"] == 3
    assert held_out.provenance["recommended_embargo_frames"] == record_memory(planted)


def test_a_split_that_leaves_no_partition_is_refused(planted):
    with pytest.raises(InvalidParameterError, match=str(MIN_FRAMES)):
        split_with_embargo(planted, fraction=0.9, embargo=60)


def test_the_partitions_do_not_overlap(planted):
    train, held_out = split_with_embargo(planted, fraction=0.6)
    assert train.frames[1] + record_memory(planted) == held_out.frames[0]


# ------------------------------------------------------------- 8. the whole pass


@pytest.fixture(scope="module")
def confirmed(planted):
    train, held_out = split_with_embargo(planted, fraction=0.6)
    result = sweep(train, lags=admissible_lags(train, max_lag=12), n_surrogates=SURROGATES)
    ledger = HeldOutLedger()
    seal, chosen = freeze_precedence(result, held_out=held_out.identity(),
                                     sealed_at="2026-08-25T00:00Z", n_surrogates=SURROGATES,
                                     ledger=ledger, study_id="tg41")
    receipt = confirm_precedence(seal, chosen, record=held_out, held_out=held_out.identity(),
                                 ledger=ledger, opened_at="2026-08-25T00:01Z", seed=99)
    return {"seal": seal, "chosen": chosen, "receipt": receipt, "held_out": held_out,
            "result": result, "ledger": ledger}


def test_the_planted_relationship_is_recovered_without_being_told(confirmed):
    assert confirmed["receipt"]["rejected_labels"] == ["level_1>level_4@5"]


def test_the_recovered_lag_and_direction_are_both_the_planted_ones(confirmed):
    entry = confirmed["receipt"]["relationships"][0]
    assert (entry["driver"], entry["driven"], entry["lag"]) == ("level_1", "level_4", 5)


def test_the_correction_is_over_what_was_frozen(confirmed):
    assert confirmed["receipt"]["correction_unit"] == len(confirmed["seal"].confirm_labels)


def test_the_confirmation_is_not_vacuous(confirmed):
    assert confirmed["receipt"]["vacuous"] == []


def test_the_reverse_member_is_in_the_family_and_does_not_confirm(confirmed):
    """Direction is recovered, not declared."""
    forward = confirmed["chosen"][0]
    assert forward.reverse_label in confirmed["result"].specification.labels()
    assert forward.reverse_label not in confirmed["receipt"]["rejected_labels"]


def test_the_held_out_partition_is_spent_once(confirmed):
    with pytest.raises(InvalidParameterError):
        confirm_precedence(confirmed["seal"], confirmed["chosen"],
                           record=confirmed["held_out"],
                           held_out=confirmed["held_out"].identity(),
                           ledger=confirmed["ledger"], opened_at="2026-08-25T01:00Z", seed=1)


def test_a_member_tested_under_a_frozen_label_at_another_lag_is_a_redefinition(confirmed):
    frozen = confirmed["chosen"][0]
    moved = PrecedenceCandidate(driver=frozen.driver, driven=frozen.driven, lag=frozen.lag + 1,
                                correlation=frozen.correlation, n_pairs=frozen.n_pairs,
                                n_effective=frozen.n_effective, p_naive=frozen.p_naive,
                                p_effective=frozen.p_effective)
    with pytest.raises(InvalidParameterError, match="redefinition"):
        confirm_precedence(confirmed["seal"], (moved,), record=confirmed["held_out"],
                           held_out=confirmed["held_out"].identity(),
                           ledger=HeldOutLedger(), opened_at="2026-08-25T01:00Z", seed=1)


def test_confirming_against_a_partition_the_seal_did_not_name(confirmed, planted):
    other, _ = split_with_embargo(planted, fraction=0.5)
    with pytest.raises(PartitionMismatchError):
        confirm_precedence(confirmed["seal"], confirmed["chosen"], record=other,
                           held_out=other.identity(), ledger=HeldOutLedger(),
                           opened_at="2026-08-25T01:00Z", seed=1)


def test_the_same_pass_over_a_record_with_nothing_planted_confirms_nothing(empty):
    train, held_out = split_with_embargo(empty, fraction=0.6)
    result = sweep(train, lags=admissible_lags(train, max_lag=12), n_surrogates=SURROGATES)
    ledger = HeldOutLedger()
    seal, chosen = freeze_precedence(result, held_out=held_out.identity(),
                                     sealed_at="2026-08-25T00:00Z", n_surrogates=SURROGATES,
                                     ledger=ledger)
    receipt = confirm_precedence(seal, chosen, record=held_out, held_out=held_out.identity(),
                                 ledger=ledger, opened_at="2026-08-25T00:01Z", seed=99)
    assert chosen, "a null that proposed nothing is a pass obtained by not looking"
    assert receipt["rejected_labels"] == []


# ------------------------------------------------------------------ 9. the records


def test_bands_observed_over_different_frames_are_refused():
    with pytest.raises(InvalidParameterError, match="same frames"):
        Record(name="ragged", series=(ScaleSeries("a", np.zeros(50)),
                                      ScaleSeries("b", np.zeros(40))))


def test_a_single_band_yields_no_pair_and_is_refused():
    with pytest.raises(InvalidParameterError, match="at least two bands"):
        Record(name="one", series=(ScaleSeries("a", np.zeros(50)),))


def test_a_missing_frame_is_a_fact_not_an_average():
    values = np.zeros(50)
    values[3] = np.nan
    with pytest.raises(InvalidParameterError, match="finite"):
        ScaleSeries("a", values)


def test_a_record_carries_its_own_lineage(planted):
    identity = planted.identity()
    assert identity.n_channels == 4
    assert identity.channel_labels == planted.levels
    assert identity.frames == (0, planted.length)


def test_a_partition_whose_lineage_and_content_disagree_is_refused():
    with pytest.raises(InvalidParameterError, match="two partitions"):
        Record(name="wrong", series=(ScaleSeries("a", np.zeros(50)),
                                     ScaleSeries("b", np.zeros(50))), frames=(0, 40))
