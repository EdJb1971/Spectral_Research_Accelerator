"""TG12.2a / D69: per-sample presence is meaning, not a NaN convention."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis_engine import cross_scale as cs
from src.analysis_engine.domain_analysis import association_only
from src.api.channels import _read_payload
from src.core.channel_series import (ChannelSeries, assert_presence_contract,
                                     from_channel_series, split_channel_series)
from src.core.domain import AxisSpec, DomainDeclaration, declaration_for
from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.core.preregistration import PartitionIdentity
from src.data_layer.tabular_source import read_tabular_channels


def _masked_series(n=20):
    present = np.zeros((n, 2), dtype=bool)
    present[:15, 0] = True
    present[5:, 1] = True
    values = np.full((n, 2), np.nan)
    values[present] = np.arange(int(present.sum()), dtype=np.float64)
    return ChannelSeries(
        channels=("float-a", "float-b"), times_seconds=np.arange(n, dtype=float),
        measures={"value": values}, present=present, support_parent_px=(1, 1))


def _domain(*, declares_presence=True):
    violations = ["no_physical_metric", "no_propagation_speed"]
    if declares_presence:
        violations.append("non_stationary_support")
    return DomainDeclaration(
        name="synthetic_float_array",
        description="An asynchronous float fixture, not a real archive.",
        axes=(AxisSpec("time", "time", units="s"),
              AxisSpec("float", "category", ordered=False)),
        licence="CC0 synthetic fixture",
        violations=tuple(violations), lag_policy="none")


@pytest.mark.parametrize("bad", [np.ones((20, 2), dtype=np.int8),
                                  np.ones((20, 2), dtype=np.float64)])
def test_presence_is_boolean_and_has_the_exact_time_channel_shape(bad):
    with pytest.raises(InvalidParameterError, match="boolean"):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(20.0),
                      measures={"value": np.ones((20, 2))}, present=bad)
    with pytest.raises(ShapeMismatchError, match="Presence must state"):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(20.0),
                      measures={"value": np.ones((20, 2))},
                      present=np.ones((20, 1), dtype=bool))


def test_presence_is_binding_but_observed_invalid_remains_distinct_from_absence():
    present = np.ones((10, 2), dtype=bool)
    values = np.ones((10, 2), dtype=float)
    values[3, 0] = np.nan  # present and invalid is permitted, and remains distinguishable
    series = ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                           measures={"value": values}, present=present)
    assert series.present[3, 0] and np.isnan(series.to_matrix("value")[3, 0])

    present[4, 1] = False
    with pytest.raises(InvalidParameterError, match="finite value.*presence is false"):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                      measures={"value": values}, present=present)


def test_a_usable_channel_needs_two_present_samples_and_counts_reach_the_receipt():
    present = np.zeros((10, 2), dtype=bool)
    present[:1, 0], present[:5, 1] = True, True
    values = np.full((10, 2), np.nan)
    with pytest.raises(InvalidParameterError, match="at least two present samples"):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                      measures={"value": values}, present=present)

    series = _masked_series()
    assert [(row["present_count"], row["absent_count"])
            for row in series.channel_records] == [(15, 5), (15, 5)]
    assert from_channel_series(series)["present_count_by_channel"] == {
        "float-a": 15, "float-b": 15}


def test_presence_is_enforced_in_both_directions_at_the_domain_boundary():
    series = _masked_series()
    assert_presence_contract(series, _domain().violations, _domain().name)
    with pytest.raises(InvalidParameterError, match="cannot travel as undeclared metadata"):
        assert_presence_contract(series, _domain(declares_presence=False).violations,
                                 _domain(declares_presence=False).name)

    unmasked = ChannelSeries(
        channels=("a", "b"), times_seconds=np.arange(20.0),
        measures={"value": np.ones((20, 2))}, support_parent_px=(1, 1))
    with pytest.raises(InvalidParameterError, match="cannot say where"):
        assert_presence_contract(unmasked, _domain().violations, _domain().name)


def test_the_tabular_reader_cannot_claim_non_stationary_support_without_a_mask(tmp_path):
    path = tmp_path / "record.csv"
    path.write_text("t,a,b\n0,1,2\n1,2,3\n2,3,4\n", encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="per-sample boolean presence mask"):
        read_tabular_channels(
            str(path), domain_name="masked_table", description="fixture",
            licence="CC0 fixture", violations=("non_stationary_support",),
            time_column="t", lag_policy="none")


def test_split_slices_presence_and_refuses_a_channel_missing_from_train_or_test():
    series = _masked_series()
    # Put the two-frame embargo wholly inside a gap. It is still returned and counted as clock
    # time, while both channels are explicitly unusable within the embargo itself.
    present = np.asarray(series.present).copy()
    present[10:12, :] = False
    values = series.to_matrix("value").copy()
    values[10:12, :] = np.nan
    gapped = ChannelSeries(
        channels=series.channels, times_seconds=series.times_seconds,
        measures={"value": values}, present=present, support_parent_px=(1, 1))
    parts = split_channel_series(gapped, train_ratio=0.5, embargo_frames=2)
    assert parts["train"].present.shape == (10, 2)
    assert parts["test"].present.shape == (8, 2)
    assert [row["present_count"] for row in parts["embargo"].channel_records] == [0, 0]
    assert not any(row["usable"] for row in parts["embargo"].channel_records)

    absent_after_split = np.ones((20, 2), dtype=bool)
    absent_after_split[10:, 0] = False
    absent_values = np.ones((20, 2), dtype=float)
    absent_values[~absent_after_split] = np.nan
    source = ChannelSeries(
        channels=("dies-early", "whole-record"), times_seconds=np.arange(20.0),
        measures={"value": absent_values}, present=absent_after_split)
    with pytest.raises(InvalidParameterError, match="test partition") as refusal:
        split_channel_series(source, train_ratio=0.5, embargo_frames=2)
    assert "dies-early" in str(refusal.value)


def test_gap_correct_decorrelation_never_compacts_asynchronous_samples():
    # Alternating observations have no genuinely adjacent pairs. Compaction invents an adjacent
    # series and understates the window; the presence-aware path measures physical frame gaps.
    values = np.linspace(-1.0, 1.0, 16)
    present = np.zeros(16, dtype=bool)
    present[::2] = True
    assert cs.decorrelation_frames(values[present]) == 2
    assert cs.decorrelation_frames(values, present=present) == 6

    # This second pattern has five observations but no lag represented by four genuine pairs.
    # Here refusal, rather than a guessed default, is the only supported result.
    sparse = np.zeros(40, dtype=bool)
    sparse[[0, 1, 4, 10, 20]] = True
    with pytest.raises(cs.CrossScaleError, match="no physical frame lag"):
        cs.decorrelation_frames(np.linspace(-1.0, 1.0, 40), present=sparse)
    assert cs.decorrelation_frames(values, present=np.ones(16, dtype=bool)) == \
        cs.decorrelation_frames(values)


def test_fixed_n_shift_null_takes_the_overlap_before_every_surrogate():
    source = np.linspace(-1.0, 1.0, 80)
    target = np.sin(np.linspace(0.0, 4.0, 80))
    joint = np.zeros(80, dtype=bool)
    joint[10:60] = True
    seen = []

    def statistic(left, right, lag, bins, wrap):
        seen.append((left.size, right.size))
        return float(np.mean(left * right))

    null = cs._shift_null(source, target, lag=2, bins=4, wrap=True,
                          statistic=statistic, n_surrogates=31, seed=7,
                          theiler=2, joint_present=joint)
    assert null.shape == (31,)
    assert set(seen) == {(50, 50)}


def test_argo_like_union_clock_selects_refusal_not_contiguous_frame_lags():
    n_floats, cycles = 20, 146
    events = sorted((cycle * 240 + 12 * float_index, float_index)
                    for float_index in range(n_floats) for cycle in range(cycles))
    clock = sorted({moment for moment, _ in events})
    positions = {moment: row for row, moment in enumerate(clock)}
    present = np.zeros((len(clock), n_floats), dtype=bool)
    for moment, channel in events:
        present[positions[moment], channel] = True

    assessment = cs.masked_frame_lag_assessment(
        present, lags=(1, 2, 3), estimator="transfer_entropy", theiler=1)
    assert len(clock) == 2920 and len(assessment["pairs"]) == 380
    assert assessment["n_pairs_with_any_admissible_lag"] == 0
    assert {row["n_effective"] for row in assessment["pairs"]} == {0}
    assert {row["longest_contiguous_joint_run"] for row in assessment["pairs"]} == {0}


def test_partial_presence_is_refused_before_a_frame_lag_can_be_reported():
    series = _masked_series(20)
    with pytest.raises(InvalidParameterError, match="future physical-time estimator") as refusal:
        association_only(series, _domain(), lags=(1, 2), cadence_seconds=1.0,
                         measure="value", estimator="mutual_information",
                         bins=3, n_surrogates=19)
    assert refusal.value.context["masked_lag_decision"] == "refuse_frame_lags"
    assessment = refusal.value.context["contiguous_candidate"]
    assert all("n_effective" in row for row in assessment["pairs"])


def test_an_all_true_declared_mask_leaves_the_accepted_sweep_byte_identical():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(120, 2))
    kwargs = dict(channels=("a", "b"), times_seconds=np.arange(120.0),
                  measures={"value": values}, support_parent_px=(1, 1))
    unmasked = ChannelSeries(**kwargs)
    masked = ChannelSeries(**kwargs, present=np.ones((120, 2), dtype=bool))
    common = dict(lags=(1,), cadence_seconds=1.0, measure="value",
                  estimator="mutual_information", bins=3, n_surrogates=19, seed=5)
    assert cs.cross_scale_dependency(unmasked, **common) == \
        cs.cross_scale_dependency(masked, **common)


class _MaskOnlySeries:
    n_times = 8
    n_channels = 2
    channels = ("a", "b")
    provenance = {"split": "test", "split_frames": [8, 16]}

    def __init__(self, present):
        self.present = present

    @property
    def measures(self):
        raise AssertionError("partition identity must not read values")


def test_partition_identity_binds_the_presence_pattern_without_reading_values():
    first = np.ones((8, 2), dtype=bool)
    second = first.copy()
    second[3, 1] = False
    a = PartitionIdentity.from_series(_MaskOnlySeries(first), name="held-out")
    b = PartitionIdentity.from_series(_MaskOnlySeries(second), name="held-out")
    assert a.digest() != b.digest()
    assert a.provenance["presence_sha256"] != b.provenance["presence_sha256"]
    assert b.provenance["present_count_by_channel"] == [8, 7]


def test_channel_api_distinguishes_absent_from_observed_invalid_and_reports_counts():
    series = _masked_series()
    values = series.to_matrix("value").copy()
    values[6, 0] = np.nan  # observed but invalid
    series = ChannelSeries(
        channels=series.channels, times_seconds=series.times_seconds,
        measures={"value": values}, present=series.present, support_parent_px=(1, 1),
        provenance={"path_basename": "floats.csv", "content_sha256": "a" * 64})
    declaration = declaration_for("order_book")
    payload = _read_payload(series, declaration, "order_book")
    first = payload["channels"][0]
    assert first["present_count"] == 15 and first["absent_count"] == 5
    assert first["presence"][6] is True and first["values"][6] is None
    assert first["presence"][16] is False and first["values"][16] is None
