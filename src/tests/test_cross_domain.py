"""A relationship that crosses units, semantics and clocks without erasing any of them.

TG4.3 is not satisfied by concatenating arrays.  These tests make the scientific boundary
executable: exact timestamp intersection instead of interpolation; physical lags instead of
native frame counts; both directions and every cross-domain channel pair in the family; and
original units, meanings, licences and clocks bound into the held-out seal.

The planted and null gates are the same generator at coupling one and zero.  Recovery matters,
but the null returning null is the load-bearing result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.benchmarks import get_benchmark
from src.benchmarks.core import Outcome
from src.benchmarks.cross_domain import build_cross_domain_pair
from src.benchmarks.seeding import derive
from src.core.cross_domain import (
    CHANNEL_SEPARATOR,
    CROSS_DOMAIN_SCHEMA,
    DomainChannel,
    DomainTimeSeries,
    ExactClockRequiredError,
    PhysicalLagRequiredError,
    align_exact,
    confirm_cross_domain,
    cross_domain_metadata,
    cross_domain_pairs,
    physical_lags,
    sweep_cross_domain,
)
from src.core.domain import AxisSpec, DomainDeclaration, PrecedenceNotAdmissibleError
from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.core.precedence import (
    Record,
    ScaleSeries,
    choose_candidates,
    freeze_precedence,
    split_with_embargo,
)
from src.core.preregistration import (
    HeldOutAlreadyOpenedError,
    HeldOutLedger,
    PartitionMismatchError,
)


ROOT_SEED = 20260825
SURROGATES = 199
LAG_SECONDS = (10800.0, 21600.0, 32400.0, 43200.0)
TRUE_LABEL = ("synthetic_thermal_observatory::thermal_gradient>"
              "synthetic_demand_ledger::demand_pressure@2")


def _declaration(name: str, *, floor: int = 1, violations=None,
                 lag_policy: str = "declared") -> DomainDeclaration:
    if violations is None:
        violations = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle",
                      "unordered_channels")
    return DomainDeclaration(
        name=name, description="test domain", licence="synthetic-test-licence",
        axes=(AxisSpec("clock", "time", units="s"),
              AxisSpec("channel", "category", ordered=False)),
        violations=violations, lag_policy=lag_policy,
        declared_floor_frames=floor if lag_policy == "declared" else None,
        declared_floor_basis=("instrument response" if lag_policy == "declared" else None))


def _series(name: str, *, cadence: float = 10.0, start: float = 0.0, n: int = 80,
            floor: int = 1, violations=None, lag_policy: str = "declared",
            aggregation_window_seconds=None) -> DomainTimeSeries:
    times = tuple(start + np.arange(n, dtype=float) * cadence)
    return DomainTimeSeries(
        declaration=_declaration(name, floor=floor, violations=violations,
                                 lag_policy=lag_policy),
        dataset_id="%s-record" % name,
        times_seconds=times,
        channels=(DomainChannel("signal", "%s primary signal" % name, "%s-unit" % name,
                                tuple(np.arange(n, dtype=float))),),
        aggregation_window_seconds=aggregation_window_seconds)


def _study(coupling: float = 1.0):
    data = build_cross_domain_pair(derive("tg4.3-tests", ROOT_SEED), coupling=coupling)
    aligned = align_exact(data.first, data.second, name="tg4.3-test")
    train, held_out = split_with_embargo(aligned, fraction=0.55)
    generated = sweep_cross_domain(
        train, lag_seconds=LAG_SECONDS, n_surrogates=SURROGATES,
        study_id="tg4.3-test")
    chosen = choose_candidates(generated.result)
    return data, aligned, train, held_out, generated, chosen


@pytest.fixture(scope="module")
def planted_study():
    return _study(1.0)


@pytest.fixture(scope="module")
def null_study():
    return _study(0.0)


@pytest.fixture(scope="module")
def confirmed(planted_study):
    _, _, _, held_out, generated, chosen = planted_study
    ledger = HeldOutLedger()
    seal, frozen = freeze_precedence(
        generated.result, held_out=held_out.identity(),
        sealed_at="2026-08-25T00:00:00Z", n_surrogates=SURROGATES,
        chosen=chosen, ledger=ledger, study_id="tg4.3-test")
    receipt = confirm_cross_domain(
        seal, frozen, record=held_out, held_out=held_out.identity(), ledger=ledger,
        opened_at="2026-08-25T00:00:01Z", seed=ROOT_SEED)
    return seal, frozen, held_out, ledger, receipt


# ------------------------------------------------------------ the source records


def test_the_two_native_clocks_are_genuinely_different(planted_study):
    data = planted_study[0]
    assert data.first.cadence_seconds == 3600.0
    assert data.second.cadence_seconds == 10800.0


def test_the_two_operands_keep_different_units_and_semantics(planted_study):
    data = planted_study[0]
    first = data.first.channels[0]
    second = data.second.channels[0]
    assert first.units == "K" and second.units == "MW"
    assert first.semantics != second.semantics


def test_a_channel_requires_a_name():
    with pytest.raises(InvalidParameterError, match="non-empty channel name"):
        DomainChannel(" ", "meaning", "unit", (1.0, 2.0))


@pytest.mark.parametrize("label", ["a::b", "a>b", "a@2"])
def test_a_channel_name_cannot_steal_the_relationship_delimiters(label):
    with pytest.raises(InvalidParameterError, match="reserved relationship delimiters"):
        DomainChannel(label, "meaning", "unit", (1.0, 2.0))


def test_semantics_are_not_optional_r19():
    with pytest.raises(InvalidParameterError, match="source quantity's meaning"):
        DomainChannel("a", " ", "unit", (1.0, 2.0))


def test_units_are_not_optional_r19():
    with pytest.raises(InvalidParameterError, match="units may never be dropped"):
        DomainChannel("a", "meaning", " ", (1.0, 2.0))


def test_non_finite_values_are_refused_before_alignment():
    with pytest.raises(InvalidParameterError, match="finite scalar"):
        DomainChannel("a", "meaning", "unit", (1.0, np.nan))


def test_a_source_record_requires_a_stable_identity():
    with pytest.raises(InvalidParameterError, match="stable source-record identifier"):
        replace(_series("a"), dataset_id=" ")


def test_an_unordered_native_clock_is_refused():
    source = _series("a")
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        replace(source, times_seconds=(0.0, 2.0, 1.0))


def test_a_channel_and_its_native_clock_must_line_up():
    with pytest.raises(ShapeMismatchError, match="actual observation time"):
        replace(_series("a"), channels=(DomainChannel(
            "short", "short channel", "u", tuple(range(12))),))


def test_duplicate_native_channel_names_are_refused():
    source = _series("a")
    channel = source.channels[0]
    with pytest.raises(InvalidParameterError, match="distinct channel labels"):
        replace(source, channels=(channel, channel))


def test_an_aggregate_must_name_its_physical_window():
    violations = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle",
                  "unordered_channels", "aggregated_values")
    with pytest.raises(InvalidParameterError, match="positive physical window"):
        _series("a", violations=violations)


def test_an_aggregation_window_cannot_hide_behind_an_instantaneous_declaration():
    with pytest.raises(InvalidParameterError, match="claiming instantaneous"):
        _series("a", aggregation_window_seconds=20.0)


# --------------------------------------------------------- exact clock alignment


def test_alignment_is_the_exact_timestamp_intersection(planted_study):
    data, aligned = planted_study[:2]
    assert aligned.length == len(data.second.times_seconds) == 360
    assert aligned.cadence_seconds == data.second.cadence_seconds


def test_alignment_never_interpolates(planted_study):
    aligned = planted_study[1]
    block = cross_domain_metadata(aligned)
    assert block["alignment"] == "exact_timestamp_intersection"
    assert block["interpolation"] == "none"


def test_the_faster_clock_reports_what_was_discarded(planted_study):
    block = cross_domain_metadata(planted_study[1])
    assert block["retained_native_observations"]["synthetic_thermal_observatory"] == 360
    assert block["discarded_native_observations"]["synthetic_thermal_observatory"] == 720
    assert block["discarded_native_observations"]["synthetic_demand_ledger"] == 0


def test_aligned_values_are_observations_from_the_native_record(planted_study):
    data, aligned = planted_study[:2]
    native = np.asarray(data.first.channels[0].values)
    observed = aligned.array("synthetic_thermal_observatory::thermal_gradient")
    assert np.array_equal(observed, native[::3])


def test_offset_clocks_are_refused_instead_of_interpolated():
    with pytest.raises(ExactClockRequiredError, match="Interpolation is not offered"):
        align_exact(_series("a", cadence=10.0), _series("b", cadence=10.0, start=5.0),
                    name="no-overlap")


def test_an_irregular_exact_intersection_is_not_a_physical_clock():
    first = _series("a", cadence=10.0, n=80)
    second = replace(_series("b", cadence=10.0, n=80),
                     times_seconds=tuple(list(np.arange(40) * 20.0)
                                         + list(790.0 + np.arange(40) * 30.0)))
    with pytest.raises(ExactClockRequiredError, match="regular native clock"):
        align_exact(first, second, name="irregular")


def test_a_domain_declaring_irregular_sampling_is_refused_precedence():
    violations = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle",
                  "unordered_channels", "irregular_sampling")
    irregular = _series("a", violations=violations)
    with pytest.raises(ExactClockRequiredError, match="association may be measured"):
        align_exact(irregular, _series("b"), name="irregular-declared")


def test_two_records_from_one_domain_are_not_cross_domain():
    first = _series("a")
    with pytest.raises(InvalidParameterError, match="two distinct domain declarations"):
        align_exact(first, replace(first, dataset_id="another"), name="same-domain")


def test_a_domain_name_cannot_steal_the_operand_delimiter():
    with pytest.raises(InvalidParameterError, match="reversible operand label"):
        align_exact(_series("a::nested"), _series("b"), name="ambiguous-domain")


def test_a_domain_with_no_lag_basis_is_refused_r21():
    no_floor = _series("a", lag_policy="none")
    with pytest.raises(PrecedenceNotAdmissibleError, match="rule R21 forbids"):
        align_exact(no_floor, _series("b"), name="unfloored")


def test_a_plain_precedence_record_cannot_be_laundered_as_cross_domain():
    plain = Record("plain", series=(ScaleSeries("a", tuple(range(30))),
                                     ScaleSeries("b", tuple(range(30)))))
    with pytest.raises(InvalidParameterError, match="does not retain the domains"):
        cross_domain_metadata(plain)


def test_the_clock_identity_is_deterministic(planted_study):
    data, aligned = planted_study[:2]
    again = align_exact(data.first, data.second, name="another-name")
    assert cross_domain_metadata(aligned)["clock_sha256"] == \
        cross_domain_metadata(again)["clock_sha256"]


# ---------------------------------------------------- the family and physical lag


def test_the_family_contains_only_cross_domain_pairs(planted_study):
    aligned = planted_study[1]
    metadata = cross_domain_metadata(aligned)["channels"]
    assert all(metadata[a]["domain"] != metadata[b]["domain"]
               for a, b in cross_domain_pairs(aligned))


def test_every_cross_domain_direction_is_in_the_family(planted_study):
    pairs = cross_domain_pairs(planted_study[1])
    assert len(pairs) == 8
    assert all((b, a) in pairs for a, b in pairs)


def test_physical_durations_become_common_clock_frames(planted_study):
    assert physical_lags(planted_study[2], LAG_SECONDS) == (1, 2, 3, 4)


def test_a_lag_below_either_domains_floor_is_refused_whole(planted_study):
    with pytest.raises(PhysicalLagRequiredError, match="refused rather than silently"):
        physical_lags(planted_study[2], (3600.0, 10800.0))


def test_an_in_between_lag_is_not_interpolated(planted_study):
    with pytest.raises(PhysicalLagRequiredError, match="No interpolation"):
        physical_lags(planted_study[2], (14400.0,))


def test_a_physical_lag_declared_twice_is_a_duplicate_member(planted_study):
    with pytest.raises(PhysicalLagRequiredError, match="duplicated family member"):
        physical_lags(planted_study[2], (10800.0, 10800.0))


def test_a_frame_count_is_not_accepted_in_place_of_physical_time(planted_study):
    with pytest.raises(PhysicalLagRequiredError, match="physical floor"):
        physical_lags(planted_study[2], (1.0, 2.0))


def test_an_aggregation_window_raises_the_physical_floor():
    violations = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle",
                  "unordered_channels", "aggregated_values")
    first = _series("a", cadence=10.0, violations=violations,
                    aggregation_window_seconds=30.0)
    aligned = align_exact(first, _series("b", cadence=10.0), name="aggregated")
    assert cross_domain_metadata(aligned)["physical_lag_floor_seconds"] == 30.0
    with pytest.raises(PhysicalLagRequiredError, match="30"):
        physical_lags(aligned, (20.0,))


def test_the_declared_family_is_eight_directions_by_four_lags(planted_study):
    generated = planted_study[4]
    assert generated.result.specification.family_size == 32
    assert generated.result.n_examined == 32
    assert set(generated.result.specification.labels()) == {
        "%s>%s@%d" % (a, b, lag)
        for a, b in generated.pairs for lag in generated.lag_frames}


def test_a_domain_with_an_unremoved_natural_cycle_is_refused_before_sweep():
    violations = ("no_physical_metric", "no_propagation_speed", "unordered_channels")
    aligned = align_exact(_series("a", violations=violations), _series("b"), name="cycle")
    with pytest.raises(InvalidParameterError, match="native-clock calendar removal"):
        sweep_cross_domain(aligned, lag_seconds=(10.0,), n_surrogates=99)


# ------------------------------------------------ generate, freeze, confirm


def test_the_planted_pair_is_selected_without_being_named(planted_study):
    chosen = planted_study[5]
    assert chosen[0].label == TRUE_LABEL


def test_the_planted_relationship_confirms_at_the_physical_lag(confirmed):
    receipt = confirmed[4]
    assert receipt["rejected_labels"] == [TRUE_LABEL]
    row = {item["label"]: item for item in receipt["relationships"]}[TRUE_LABEL]
    assert row["lag_seconds"] == 21600.0


def test_the_confirmed_operands_retain_both_source_vocubularies(confirmed):
    row = {item["label"]: item for item in confirmed[4]["relationships"]}[TRUE_LABEL]
    assert row["driver_operand"]["units"] == "K"
    assert row["driven_operand"]["units"] == "MW"
    assert row["driver_operand"]["semantics"] == "instantaneous thermal state"
    assert row["driven_operand"]["semantics"] == "reported electrical demand"


def test_the_cross_domain_statistic_is_dimensionless(confirmed):
    receipt = confirmed[4]
    assert receipt["cross_domain"]["statistic_units"] == "dimensionless"
    assert all(row["statistic_units"] == "dimensionless"
               for row in receipt["relationships"])


def test_precedence_is_not_reported_as_causation(confirmed):
    boundary = confirmed[4]["claim_boundary"]
    assert "never compared" in boundary
    assert "does not identify a causal mechanism" in boundary


def test_changing_a_units_declaration_after_the_seal_breaks_the_partition(planted_study):
    _, _, _, held_out, generated, chosen = planted_study
    ledger = HeldOutLedger()
    seal, frozen = freeze_precedence(
        generated.result, held_out=held_out.identity(),
        sealed_at="2026-08-25T00:00:00Z", n_surrogates=SURROGATES,
        chosen=chosen, ledger=ledger)
    provenance = deepcopy(held_out.provenance)
    channel = "synthetic_thermal_observatory::thermal_gradient"
    provenance["cross_domain"]["channels"][channel]["units"] = "degrees-ish"
    tampered = Record(held_out.name, series=held_out.series,
                      cadence_seconds=held_out.cadence_seconds,
                      frames=held_out.frames, provenance=provenance)
    with pytest.raises(PartitionMismatchError):
        confirm_cross_domain(
            seal, frozen, record=tampered, held_out=tampered.identity(), ledger=ledger,
            opened_at="2026-08-25T00:00:01Z", seed=ROOT_SEED)


def test_a_within_domain_member_cannot_be_laundered_through_this_boundary(planted_study):
    _, _, _, held_out, generated, chosen = planted_study
    ledger = HeldOutLedger()
    seal, frozen = freeze_precedence(
        generated.result, held_out=held_out.identity(),
        sealed_at="2026-08-25T00:00:00Z", n_surrogates=SURROGATES,
        chosen=chosen, ledger=ledger)
    candidate = frozen[0]
    within = replace(candidate, driven="synthetic_thermal_observatory::pressure_tendency")
    with pytest.raises(InvalidParameterError, match="cannot be laundered"):
        confirm_cross_domain(
            seal, (within,), record=held_out, held_out=held_out.identity(), ledger=ledger,
            opened_at="2026-08-25T00:00:01Z", seed=ROOT_SEED)


def test_the_held_out_cross_domain_partition_can_be_spent_only_once(confirmed):
    seal, frozen, held_out, ledger, _ = confirmed
    with pytest.raises(HeldOutAlreadyOpenedError):
        confirm_cross_domain(
            seal, frozen, record=held_out, held_out=held_out.identity(), ledger=ledger,
            opened_at="2026-08-25T00:00:02Z", seed=ROOT_SEED)


def test_the_same_builder_at_coupling_zero_confirms_nothing(null_study):
    _, _, _, held_out, generated, chosen = null_study
    ledger = HeldOutLedger()
    seal, frozen = freeze_precedence(
        generated.result, held_out=held_out.identity(),
        sealed_at="2026-08-25T00:00:00Z", n_surrogates=SURROGATES,
        chosen=chosen, ledger=ledger)
    receipt = confirm_cross_domain(
        seal, frozen, record=held_out, held_out=held_out.identity(), ledger=ledger,
        opened_at="2026-08-25T00:00:01Z", seed=ROOT_SEED)
    assert receipt["rejected_labels"] == []
    assert len(frozen) >= 1, "the null must select candidates on train to be a real control"


# ----------------------------------------------------------- registered gates


@pytest.mark.parametrize("name", ["planted_cross_domain", "cross_domain_null"])
def test_both_cross_domain_gates_are_live_and_green(name):
    results = get_benchmark(name).run(ROOT_SEED)
    assert len(results) == 1
    assert results[0].outcome is Outcome.PASS, results[0].detail


def test_the_registered_null_is_marked_as_a_null():
    assert get_benchmark("cross_domain_null").is_null is True
    assert get_benchmark("planted_cross_domain").is_null is False


def test_the_cross_domain_schema_is_carried_to_confirmation(confirmed):
    assert confirmed[4]["cross_domain"]["schema"] == CROSS_DOMAIN_SCHEMA
    assert CHANNEL_SEPARATOR in TRUE_LABEL
