"""`FieldSequence`, the temporal split, and the artifact store (roadmap T4A.1-T4A.4).

Phase 4A is the foundation the rest of Phase 4 stands on, and two of its guarantees are the
kind that fail silently if they are not tested directly:

*   **A sequence whose frames disagree** - different shapes, different grids, unsorted or
    duplicated timestamps - produces numbers rather than errors. Averaging across frames from
    different regions is arithmetic that succeeds and means nothing.
*   **A temporal split without an embargo** looks clean and leaks. Adjacent windows are
    disjoint, so an overlap check passes; but a training example near the boundary has its
    *target* inside the test window. R6 exists because "wavelet B predicts t+6" is trivially
    fakeable without one, and T4A.2's acceptance criterion is explicitly that a deliberate
    leakage attempt raises.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import (FieldSequence, split_temporal,
                                        validate_temporal_guardrails)


SIX_HOURS = 6 * 3600.0


def _fields(n=20, shape=(8, 8), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [PhysicalField(torch.randn(*shape, generator=generator, dtype=torch.float64))
            for _ in range(n)]


def _times(n=20, step=SIX_HOURS, start=0.0):
    return np.arange(n) * step + start


def _sequence(n=20, **kwargs):
    return FieldSequence(_fields(n), _times(n), **kwargs)


# ======================================================== construction and validation

def test_a_sequence_carries_shape_grid_and_time():
    seq = _sequence(12)
    assert len(seq) == 12
    assert seq.shape == (12, 8, 8)
    assert seq.to_tensor().shape == (12, 8, 8)
    assert seq.cadence_seconds == SIX_HOURS
    assert seq.is_regular is True


def test_frames_of_different_shapes_are_refused():
    """A ragged sequence cannot be stacked, differenced or transformed as a block."""
    fields = _fields(3) + [PhysicalField(torch.zeros(8, 9, dtype=torch.float64))]
    with pytest.raises(ShapeMismatchError) as excinfo:
        FieldSequence(fields, _times(4))
    assert "frame 3" in str(excinfo.value)


def test_frames_on_different_grids_are_refused():
    """The failure this prevents produces numbers, not errors.

    Two frames on different grids are not observations of one region. Every statistic computed
    across them - a mean, a spectrum, a gradient - would be averaging different places
    together, and nothing downstream can detect it.
    """
    a = PhysicalField(torch.zeros(8, 8, dtype=torch.float64),
                      grid=GridSpec(kind="cartesian", shape=(8, 8), dy=1000.0, dx=1000.0))
    b = PhysicalField(torch.zeros(8, 8, dtype=torch.float64),
                      grid=GridSpec(kind="cartesian", shape=(8, 8), dy=2000.0, dx=2000.0))
    with pytest.raises(InvalidParameterError) as excinfo:
        FieldSequence([a, b], _times(2))
    assert "not observations of one region" in str(excinfo.value)


def test_frames_on_equal_but_distinct_grids_are_accepted():
    """Grid equality is on the metric, not on object identity.

    Two frames cropped from the same archive are separate `GridSpec` instances describing the
    same region; refusing those would make the class unusable on real data.
    """
    grid_a = GridSpec(kind="cartesian", shape=(8, 8), dy=1000.0, dx=1000.0)
    grid_b = GridSpec(kind="cartesian", shape=(8, 8), dy=1000.0, dx=1000.0)
    assert grid_a is not grid_b
    fields = [PhysicalField(torch.zeros(8, 8, dtype=torch.float64), grid=g)
              for g in (grid_a, grid_b)]
    assert len(FieldSequence(fields, _times(2))) == 2


def test_unsorted_times_are_refused_rather_than_sorted():
    """Sorting silently would reorder the caller's frames without anyone noticing."""
    with pytest.raises(InvalidParameterError) as excinfo:
        FieldSequence(_fields(3), [0.0, 2.0, 1.0])
    assert "would silently reorder" in str(excinfo.value)


def test_duplicate_timestamps_are_refused():
    """"The next frame" must be unambiguous, or every lag is wrong by an unknown amount."""
    with pytest.raises(InvalidParameterError) as excinfo:
        FieldSequence(_fields(3), [0.0, 1.0, 1.0])
    assert "repeats or precedes" in str(excinfo.value)


def test_a_time_per_frame_is_required():
    with pytest.raises(ShapeMismatchError):
        FieldSequence(_fields(3), _times(2))


def test_an_empty_sequence_is_refused():
    with pytest.raises(InvalidParameterError):
        FieldSequence([], [])


def test_iso_timestamps_are_accepted():
    seq = FieldSequence(_fields(3), ["2020-01-01", "2020-01-02", "2020-01-03"])
    assert seq.time_kind == "datetime64"
    assert seq.cadence_seconds == 86400.0


def test_numeric_and_datetime_times_are_distinguished():
    """A lag of "6" means six hours in one case and six unknown units in the other."""
    assert FieldSequence(_fields(3), [0.0, 1.0, 2.0]).time_kind == "numeric"
    assert FieldSequence(_fields(3), np.arange("2020-01-01", "2020-01-04",
                                               dtype="datetime64[D]")).time_kind == "datetime64"


# ======================================================== cadence

def test_an_irregular_sequence_is_reported_not_refused():
    """A concatenation of two crops is legitimate; converting frames to hours is not."""
    seq = FieldSequence(_fields(4), [0.0, SIX_HOURS, 2 * SIX_HOURS, 10 * SIX_HOURS])
    assert seq.is_regular is False
    with pytest.raises(InvalidParameterError) as excinfo:
        seq.lag_to_seconds(2)
    assert "has gaps" in str(excinfo.value)


def test_lag_conversion_on_a_regular_sequence():
    assert _sequence(10).lag_to_seconds(2) == 2 * SIX_HOURS


def test_a_single_frame_has_no_cadence():
    seq = FieldSequence(_fields(1), [0.0])
    assert seq.cadence_seconds is None
    assert seq.is_regular is True


# ======================================================== access

def test_slicing_returns_a_sequence_not_a_list():
    """A list would drop the time axis - the exact failure this class exists to prevent."""
    sub = _sequence(10)[2:5]
    assert isinstance(sub, FieldSequence)
    assert len(sub) == 3
    assert sub.times_seconds[0] == 2 * SIX_HOURS


def test_at_names_the_valid_range():
    with pytest.raises(InvalidParameterError) as excinfo:
        _sequence(5).at(99)
    assert "0..4" in str(excinfo.value)


def test_at_time_refuses_an_inexact_match_by_default():
    """Returning the nearest frame silently is a six-hour error with nothing to indicate it."""
    seq = _sequence(10)
    assert seq.at_time(2 * SIX_HOURS) is seq.at(2)
    with pytest.raises(InvalidParameterError) as excinfo:
        seq.at_time(2 * SIX_HOURS + 60)
    assert "nearest frame" in str(excinfo.value)
    # An approximate match is available, but only when asked for explicitly.
    assert seq.at_time(2 * SIX_HOURS + 60, tolerance=120) is seq.at(2)


# ======================================================== transformation

def test_map_preserves_the_time_axis():
    seq = _sequence(6)
    doubled = seq.map(lambda f: PhysicalField(f.data * 2, coords=f.coords, grid=f.grid))
    assert isinstance(doubled, FieldSequence)
    assert np.array_equal(doubled.times_seconds, seq.times_seconds)
    assert torch.allclose(doubled.at(0).data, seq.at(0).data * 2)


def test_map_refuses_a_function_that_does_not_return_a_field():
    with pytest.raises(InvalidParameterError):
        _sequence(3).map(lambda f: f.data)


def test_anomalies_remove_the_temporal_mean():
    seq = _sequence(8)
    anomalies = seq.anomalies()
    assert torch.allclose(anomalies.to_tensor().mean(dim=0),
                          torch.zeros(8, 8, dtype=torch.float64), atol=1e-12)
    assert anomalies.metadata["anomaly"] == "temporal_mean_removed"


def test_summary_is_lineage_safe():
    """Statistics and shape, never the payload."""
    summary = FieldSequence(_fields(100, (64, 64)), _times(100)).summary()
    encoded = len(json.dumps(summary, default=str).encode())
    assert encoded < 4096, "a sequence summary must fit in a lineage row (%d bytes)" % encoded
    assert summary["shape"] == [100, 64, 64]
    assert "min" in summary and "max" in summary


# ======================================================== R6: the temporal split

def test_split_temporal_partitions_in_order():
    parts = split_temporal(_sequence(20), 0.6, 0.2, embargo_frames=0)
    assert len(parts["train"]) == 12
    assert len(parts["val"]) == 4
    assert len(parts["test"]) == 4
    assert parts["train"].times_seconds[-1] < parts["val"].times_seconds[0]
    assert parts["val"].times_seconds[-1] < parts["test"].times_seconds[0]


def test_the_embargo_is_returned_not_silently_dropped():
    """A lineage record must show what was excluded, so a reader can check the gap was real."""
    parts = split_temporal(_sequence(20), 0.6, 0.2, embargo_frames=2)
    assert len(parts["embargo_train_val"]) == 2
    assert len(parts["embargo_val_test"]) == 2
    assert sum(len(p) for p in parts.values()) == 20


def test_every_split_records_how_it_was_made():
    parts = split_temporal(_sequence(20), 0.6, 0.2, embargo_frames=2)
    record = parts["test"].metadata["temporal_split"]
    assert record["embargo_frames"] == 2
    assert record["n_frames_total"] == 20


def test_an_embargo_that_consumes_the_record_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        split_temporal(_sequence(10), 0.6, 0.2, embargo_frames=5)
    message = str(excinfo.value)
    assert "consumes everything" in message
    # And it must say the wrong fix out loud.
    assert "do NOT shrink the embargo" in message


def test_ratios_that_leave_an_empty_window_are_refused():
    with pytest.raises(InvalidParameterError):
        split_temporal(_sequence(20), 0.95, 0.049)


def test_a_two_way_split_needs_no_val_window():
    parts = split_temporal(_sequence(20), 0.7, 0.0, embargo_frames=2)
    assert set(parts) == {"train", "embargo_train_test", "test"}


# ======================================================== R6: the leakage guardrails

def test_overlapping_windows_raise():
    """T4A.2's acceptance criterion, half one: a deliberate overlap must raise."""
    seq = _sequence(20)
    a = FieldSequence(seq.fields[:12], seq.raw_times_slice(slice(0, 12)), split="train")
    b = FieldSequence(seq.fields[8:], seq.raw_times_slice(slice(8, 20)), split="test")
    with pytest.raises(ValueError) as excinfo:
        validate_temporal_guardrails(a, b)
    assert "they overlap" in str(excinfo.value)
    assert "measuring memorisation" in str(excinfo.value)


def test_an_embargo_shorter_than_the_longest_lag_raises():
    """T4A.2's acceptance criterion, half two - and the subtle half.

    The windows are disjoint, so an overlap check passes. But a training example at the end of
    the train window has its target *inside* the test window, so the split looks clean and
    leaks. This is the failure that has to be checked explicitly rather than assumed away by
    "they do not overlap".
    """
    parts = split_temporal(_sequence(20), 0.6, 0.0, embargo_frames=1)
    # No overlap - this passes without a lag argument.
    assert validate_temporal_guardrails(parts["train"], parts["test"]) is True
    # ...and leaks the moment the lag under test exceeds the gap.
    with pytest.raises(ValueError) as excinfo:
        validate_temporal_guardrails(parts["train"], parts["test"], max_lag_frames=4)
    message = str(excinfo.value)
    assert "clean-looking and leaks" in message
    assert "Increase the embargo" in message


def test_an_embargo_exactly_equal_to_the_lag_still_raises():
    """A gap of exactly one lag puts the last training target on the first test frame."""
    parts = split_temporal(_sequence(30), 0.6, 0.0, embargo_frames=2)
    with pytest.raises(ValueError):
        validate_temporal_guardrails(parts["train"], parts["test"], max_lag_frames=3)


def test_a_sufficient_embargo_passes():
    parts = split_temporal(_sequence(30), 0.6, 0.0, embargo_frames=5)
    assert validate_temporal_guardrails(parts["train"], parts["test"],
                                        max_lag_frames=3) is True


def test_two_windows_of_the_same_split_are_not_compared():
    """Mirrors `PhysicalField.validate_split_guardrails`: same split, no guardrail."""
    seq = _sequence(20)
    a = FieldSequence(seq.fields[:10], seq.raw_times_slice(slice(0, 10)), split="train")
    b = FieldSequence(seq.fields[5:15], seq.raw_times_slice(slice(5, 15)), split="train")
    assert validate_temporal_guardrails(a, b) is True


# ======================================================== T4A.4: adapter sequence slicing

def test_slice_sequence_returns_a_sequence_with_time(tmp_path):
    """The entry point through which every Phase 4 stage gets its data."""
    from src.data_layer.adapters import MeteorologicalDataAdapter

    seq = MeteorologicalDataAdapter.slice_sequence(
        "era5_reanalysis", "t2m", time_range=("2023-01-01", "2023-01-03"))
    assert isinstance(seq, FieldSequence)
    assert len(seq) >= 2
    assert seq.units == "K"
    assert seq.grid.kind == "latlon"


def test_slice_sequence_carries_the_simulated_flag():
    """A sequence served by the fallback source must say so, exactly as a single field does."""
    from src.data_layer.adapters import MeteorologicalDataAdapter

    seq = MeteorologicalDataAdapter.slice_sequence("era5_reanalysis", "t2m")
    assert seq.metadata["is_simulated"] is True
    assert "source_kind" in seq.metadata


def test_slice_sequence_counts_replaced_non_finite_values():
    """A zero is a value, not an absence.

    Replacing missing data with zeros changes every statistic computed afterwards - a spectrum
    of a field with zeroed gaps has structure the atmosphere does not. The count is recorded so
    the fact survives into the provenance rather than being invisible.
    """
    from src.data_layer.adapters import MeteorologicalDataAdapter

    seq = MeteorologicalDataAdapter.slice_sequence("era5_reanalysis", "t2m")
    assert "n_nonfinite_replaced" in seq.metadata
    assert "nonfinite_policy" in seq.metadata


def test_slice_sequence_refuses_an_empty_time_range():
    from src.data_layer.adapters import MeteorologicalDataAdapter

    with pytest.raises(ValueError) as excinfo:
        MeteorologicalDataAdapter.slice_sequence(
            "era5_reanalysis", "t2m", time_range=("1990-01-01", "1990-01-02"))
    assert "covers" in str(excinfo.value), "the refusal must name the actual coverage"


def test_slice_sequence_guards_against_materialising_the_archive():
    """A whole ERA5 record is ~93,000 frames; the failure mode is a killed process."""
    from src.data_layer.adapters import MeteorologicalDataAdapter

    with pytest.raises(ValueError) as excinfo:
        MeteorologicalDataAdapter.slice_sequence("era5_reanalysis", "t2m", max_frames=2)
    message = str(excinfo.value)
    assert "over the 2-frame limit" in message
    assert "GB" in message, "the refusal must quantify what it prevented"


def test_the_slice_sequence_action_stores_a_handle_not_a_payload(tmp_path, monkeypatch):
    """T4A.3 + T4A.4 together: the frames go to the store, the reference goes to lineage."""
    import torch as _torch

    from src.artifact_store import store as store_module
    from src.artifact_store.store import ArtifactStore
    from src.experiment_engine import actions

    monkeypatch.setattr(store_module, "_DEFAULT", ArtifactStore(str(tmp_path / "artifacts")))
    args = {"dataset_id": "era5_reanalysis", "variable": "t2m",
            "time_range": ["2023-01-01", "2023-01-05"]}
    result = actions.execute("slice_sequence", args, _torch.device("cpu"))
    assert result["sequence_ref"].startswith("artifact://")
    assert "field_data" not in result, "the payload must not travel between steps"

    node = actions.summarise("slice_sequence", result, args)
    encoded = len(json.dumps(node, default=str).encode())
    assert encoded < 4096, "lineage row is %d bytes" % encoded
    assert node["sequence"]["n_frames"] >= 2
