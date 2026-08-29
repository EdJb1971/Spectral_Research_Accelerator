"""TG0.1: the channel-series contract, and the claim that the inference layer is domain-free.

The load-bearing test in this file is
`test_a_plain_channel_series_reproduces_the_signature_result_exactly`. Everything else is the
validation surface around it.

That test matters because the whole cross-domain programme rests on one assertion: that
`cross_scale_dependency` — surrogate-referenced, BY-corrected, replication-gated — does not
actually depend on the atmosphere, on a wavelet decomposition, or on a 2D grid, and only ever
needed labelled scalar series with a validity mask and a lag basis. That is an easy thing to
assert in a design document and a cheap thing to check in code, so it is checked here rather
than assumed: the same numbers, carried by a domain-neutral container that has never heard of a
wavelet, must produce a byte-identical result.

If this test ever fails, the abstraction is not real and Phase G0 has done its job.
"""

import numpy as np
import pytest

from src.analysis_engine import cross_scale as cs
from src.analysis_engine.scale_signature import scale_signature
from src.core.channel_series import (ChannelSeries, ChannelSeriesLike, from_channel_series)
from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.synthetic_generator.cascade import build_cascade
from src.transform_engine.coefficient_field import decompose_sequence

CADENCE = 3600.0
MEASURE = "energy_fraction"


def _signature(levels=3, frames=96):
    sequence = build_cascade(n_frames=frames, seed=4242)
    return scale_signature(decompose_sequence(
        sequence, "swt", {"levels": levels, "wavelet": "db2"}, keep_native=False))


def _mirror(signature) -> ChannelSeries:
    """A domain-neutral series carrying exactly the signature's own numbers.

    Nothing is recomputed. This is deliberately a transcription, because the question under
    test is whether the *container* matters, not whether two decompositions agree.
    """
    records = signature.channel_records
    return ChannelSeries(
        channels=list(signature.channels),
        times_seconds=np.asarray(signature.times_seconds, dtype=np.float64),
        measures={MEASURE: signature.to_matrix(MEASURE)},
        usable=[bool(record["usable"]) for record in records],
        support_parent_px=[record.get("support_parent_px") for record in records],
        provenance=dict(signature.provenance),
    )


# --------------------------------------------------------------------- the contract holds

def test_scale_signature_satisfies_the_channel_series_contract():
    """The contract was extracted from the code, so the original must already satisfy it."""
    signature = _signature()
    assert isinstance(signature, ChannelSeriesLike)
    assert list(signature.channels) == list(signature.scales)
    assert list(signature.channel_records) == list(signature.interior)


def test_a_plain_channel_series_satisfies_the_contract():
    assert isinstance(_mirror(_signature()), ChannelSeriesLike)


# ------------------------------------------------------- the Phase G0 claim, as a test

def test_a_plain_channel_series_reproduces_the_signature_result_exactly():
    """The inference layer does not care that its input came from a wavelet transform.

    Identical inputs, two containers, one result. Compared including the configuration hash
    and every per-test p-value, effect size and surrogate mean, because "close enough" would
    hide exactly the kind of container-dependent path this test exists to rule out.
    """
    signature = _signature()
    kwargs = dict(lags=(2, 3, 4), cadence_seconds=CADENCE, measure=MEASURE,
                  n_surrogates=199, bins=4, seed=20260823)

    from_signature = cs.cross_scale_dependency(signature, **kwargs)
    from_series = cs.cross_scale_dependency(_mirror(signature), **kwargs)

    assert from_series["analysis_config_sha256"] == from_signature["analysis_config_sha256"]
    assert from_series["n_tests"] == from_signature["n_tests"] > 0
    assert from_series["n_excluded"] == from_signature["n_excluded"]
    assert from_series["n_significant"] == from_signature["n_significant"]

    for expected, actual in zip(from_signature["results"], from_series["results"]):
        assert expected["label"] == actual["label"]
        for key in ("observed_nats", "surrogate_mean_nats", "excess_nats", "p_value",
                    "effect_size", "q_value", "theiler_window_frames",
                    "support_floor_frames"):
            left, right = expected[key], actual[key]
            if isinstance(left, float) and np.isnan(left):
                assert np.isnan(right), key
            else:
                assert left == right, key


def test_the_support_floor_is_identical_for_both_containers():
    signature = _signature()
    assert (cs.support_floor(_mirror(signature), CADENCE, advection_speed_m_s=12.0)
            == cs.support_floor(signature, CADENCE, advection_speed_m_s=12.0))


# ------------------------------------------------------------------- rule R21's refusal

def test_a_channel_without_declared_support_is_refused_a_geometric_floor():
    """R21: no declared representation support means no geometric floor, and no invention.

    A domain whose representation has no spatial footprint must be refused rather than given
    a plausible-looking default — the same reasoning that denies `advection_speed_m_s` a
    default. The refusal names the channel record so an adapter author can find it.
    """
    signature = _signature()
    series = ChannelSeries(
        channels=list(signature.channels),
        times_seconds=np.asarray(signature.times_seconds, dtype=np.float64),
        measures={MEASURE: signature.to_matrix(MEASURE)},
        provenance=dict(signature.provenance),
    )
    assert all("support_parent_px" not in record for record in series.channel_records)
    with pytest.raises(InvalidParameterError, match="support_parent_px"):
        cs.support_floor(series, CADENCE, advection_speed_m_s=12.0)


def test_a_series_with_no_physical_grid_reports_a_one_frame_floor_and_says_why():
    series = ChannelSeries(
        channels=("a", "b"),
        times_seconds=np.arange(64, dtype=np.float64) * CADENCE,
        measures={MEASURE: np.random.default_rng(0).normal(size=(64, 2))},
        support_parent_px=(4, 8),
    )
    floors = cs.support_floor(series, CADENCE, advection_speed_m_s=12.0)
    assert floors["enforced"] is False
    assert floors["floor_by_scale"] == {"a": 1, "b": 1}
    assert any("no physical spacing" in warning for warning in floors["warnings"])


# ------------------------------------------------------------------------- validation

def test_measures_must_be_sampled_on_the_declared_grid():
    with pytest.raises(ShapeMismatchError):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                      measures={MEASURE: np.zeros((9, 2))})


def test_duplicate_channel_labels_are_refused():
    with pytest.raises(InvalidParameterError, match="distinct channel labels"):
        ChannelSeries(channels=("a", "a"), times_seconds=np.arange(10.0),
                      measures={MEASURE: np.zeros((10, 2))})


def test_a_single_channel_is_refused():
    with pytest.raises(InvalidParameterError, match="at least two channels"):
        ChannelSeries(channels=("a",), times_seconds=np.arange(10.0),
                      measures={MEASURE: np.zeros((10, 1))})


def test_an_unordered_clock_is_refused_rather_than_sorted():
    clock = np.arange(10.0)
    clock[4] = 99.0
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        ChannelSeries(channels=("a", "b"), times_seconds=clock,
                      measures={MEASURE: np.zeros((10, 2))})


def test_an_undeclared_measure_is_refused_by_name():
    series = ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                           measures={MEASURE: np.zeros((10, 2))})
    with pytest.raises(InvalidParameterError, match="energy_fraction"):
        series.to_matrix("participation_ratio")


def test_a_per_channel_mask_must_line_up_with_the_channels():
    with pytest.raises(ShapeMismatchError):
        ChannelSeries(channels=("a", "b"), times_seconds=np.arange(10.0),
                      measures={MEASURE: np.zeros((10, 2))}, usable=(True,))


def test_an_unusable_channel_carries_its_reason():
    series = ChannelSeries(
        channels=("a", "b"), times_seconds=np.arange(10.0),
        measures={MEASURE: np.zeros((10, 2))}, usable=(True, False),
        unusable_reason={"b": "sensor offline for the whole record"})
    records = series.channel_records
    assert records[0] == {"scale": "a", "usable": True}
    assert records[1]["reason"] == "sensor offline for the whole record"


def test_the_lineage_summary_holds_no_arrays():
    summary = from_channel_series(_mirror(_signature()))
    assert summary["n_channels"] == 3
    assert summary["usable"] and summary["with_declared_support"]
    assert all(not isinstance(value, np.ndarray) for value in summary.values())
