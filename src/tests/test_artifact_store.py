"""Content-addressed artifact store (roadmap T4A.3).

T4A.3's acceptance criteria, stated in the roadmap, are asserted here directly: a 100-frame
64x64 sequence round-trips, every lineage row stays under 4 KB, and `analyze_boundary` no
longer embeds a full padded field in its node.

The properties worth testing beyond those are the ones whose failure is *silent*:

*   **Checksum verification.** An artifact whose bytes no longer match its handle is not an
    error anywhere unless someone checks - it is simply a different array, and every number
    derived from it is wrong with nothing to indicate it.
*   **Handle size.** A handle that grew a payload back into its summary would defeat the whole
    purpose while still looking exactly like a handle.
*   **Refusal to pickle.** An artifact only the code that wrote it can read is not
    reproducible data, whatever the lineage row says.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.artifact_store.store import (MAX_HANDLE_BYTES, ArtifactError, ArtifactHandle,
                                      ArtifactStore, dereference, is_ref, summarise)
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.sequence import FieldSequence


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(str(tmp_path / "artifacts"))


# ======================================================== T4A.3 acceptance criteria

def test_a_100_frame_64x64_sequence_round_trips(store):
    """The stated acceptance criterion, and the size Phase 4B actually needs."""
    generator = torch.Generator().manual_seed(0)
    fields = [PhysicalField(torch.randn(64, 64, generator=generator, dtype=torch.float64))
              for _ in range(100)]
    sequence = FieldSequence(fields, np.arange(100) * 21600.0)

    handle = store.put(sequence, name="era5_like")
    loaded = store.load(handle)

    assert loaded.shape == (100, 64, 64)
    assert np.array_equal(loaded, sequence.to_tensor().numpy()), (
        "the round trip must be exact; a lossy store would silently change every result")
    assert handle.summary["source"] == "FieldSequence"
    assert handle.summary["n_frames"] == 100


def test_a_lineage_row_stays_under_four_kilobytes(store):
    """The point of the whole exercise: the payload does not go in the database."""
    generator = torch.Generator().manual_seed(0)
    fields = [PhysicalField(torch.randn(64, 64, generator=generator, dtype=torch.float64))
              for _ in range(100)]
    sequence = FieldSequence(fields, np.arange(100) * 21600.0)
    handle = store.put(sequence, name="era5_like")

    row = {"action": "slice_sequence", "sequence_ref": handle.ref,
           "sequence": sequence.summary(), "handle": handle.to_dict()}
    encoded = len(json.dumps(row, default=str).encode("utf-8"))
    assert encoded < 4096, "lineage row is %d bytes" % encoded
    # For contrast, the payload it replaces:
    assert sequence.to_tensor().numpy().nbytes > 3_000_000


def test_analyze_boundary_does_not_embed_a_padded_field():
    """The third acceptance criterion - and it was already satisfied, by the summariser layer.

    Asserted rather than assumed: `_summarise_analyze_boundary` returns the treatment, the pad
    width and the leakage metric, so the full padded field never reaches a lineage node even
    though the action returns it. This test exists so a later change to that summariser cannot
    quietly reintroduce the payload.
    """
    from src.experiment_engine import actions

    result = {"padded_field": [[0.0] * 64 for _ in range(64)],
              "spectral_leakage": 0.01, "treatment": "zero"}
    node = actions.summarise("analyze_boundary", result,
                             {"treatment": "zero", "pad_width": 4})
    encoded = json.dumps(node, default=str)
    assert "padded_field" not in encoded
    assert len(encoded.encode()) < 4096


# ======================================================== content addressing

def test_identical_content_produces_one_artifact(store):
    """Content addressing is the deduplication, not a filing convention."""
    array = np.arange(1000, dtype=np.float64).reshape(10, 100)
    first = store.put(array, name="a")
    second = store.put(array.copy(), name="b")
    assert first.ref == second.ref
    assert store.stats()["n_artifacts"] == 1


def test_different_content_produces_different_artifacts(store):
    a = store.put(np.zeros((4, 4)), name="a")
    b = store.put(np.ones((4, 4)), name="b")
    assert a.ref != b.ref
    assert store.stats()["n_artifacts"] == 2


def test_a_tampered_artifact_fails_its_checksum(store):
    """The failure this catches is silent: a different array, not an error."""
    handle = store.put(np.arange(64, dtype=np.float64), name="x")
    path = store.path_of(handle)
    with open(path, "rb") as fh:
        payload = fh.read()
    # Flip a bit rather than overwriting the tail with zeros. The last bytes of a zip are the
    # end-of-central-directory comment length, which is *already* zero - so the first version
    # of this test rewrote the file byte-identically and passed for the wrong reason. It
    # proved nothing about the checksum, which is precisely the property under test.
    middle = len(payload) // 2
    tampered = payload[:middle] + bytes([payload[middle] ^ 0xFF]) + payload[middle + 1:]
    assert tampered != payload, "the tamper must actually change the bytes"
    with open(path, "wb") as fh:
        fh.write(tampered)
    with pytest.raises(ArtifactError) as excinfo:
        store.load(handle)
    assert "fails its checksum" in str(excinfo.value)


def test_a_handle_that_disagrees_with_the_store_is_refused(store):
    handle = store.put(np.zeros((4, 4)), name="x")
    lying = ArtifactHandle(ref=handle.ref, shape=handle.shape, dtype=handle.dtype,
                           sha256="0" * 64, format=handle.format,
                           bytes_on_disk=handle.bytes_on_disk)
    with pytest.raises(ArtifactError):
        store.load(lying)


def test_a_missing_artifact_names_the_remedy(store):
    handle = store.put(np.zeros((4, 4)), name="x")
    os.remove(store.path_of(handle))
    with pytest.raises(ArtifactError) as excinfo:
        store.load(handle)
    assert "must be re-run" in str(excinfo.value)


def test_artifacts_are_sharded_by_prefix(store):
    """A flat directory of tens of thousands of files is unusable on NTFS."""
    handle = store.put(np.zeros((2, 2)), name="x")
    relative = os.path.relpath(store.path_of(handle), store.directory)
    assert relative.split(os.sep)[0] == handle.sha256[:2]


def test_no_temporary_files_are_left_behind(store):
    """Writes are atomic; a crash must never leave a half-written artifact."""
    store.put(np.zeros((8, 8)), name="x")
    leftovers = [name for _root, _dirs, files in os.walk(store.directory)
                 for name in files if name.endswith(".tmp")]
    assert leftovers == []


# ======================================================== handles

def test_a_handle_is_small(store):
    handle = store.put(np.random.default_rng(0).standard_normal((100, 64, 64)), name="big")
    assert handle.check_size() < MAX_HANDLE_BYTES


def test_a_handle_carrying_a_payload_is_refused(store):
    """A handle with data smuggled into its summary defeats the purpose while looking right."""
    handle = store.put(np.zeros((2, 2)), name="x")
    bloated = ArtifactHandle(
        ref=handle.ref, shape=handle.shape, dtype=handle.dtype, sha256=handle.sha256,
        format=handle.format, bytes_on_disk=handle.bytes_on_disk,
        summary={"payload": list(range(5000))})
    with pytest.raises(ArtifactError) as excinfo:
        bloated.check_size()
    assert "statistics, not data" in str(excinfo.value)


def test_a_handle_round_trips_through_json(store):
    handle = store.put(np.arange(9.0).reshape(3, 3), name="x")
    restored = ArtifactHandle.from_dict(json.loads(json.dumps(handle.to_dict())))
    assert restored == handle
    assert np.array_equal(store.load(restored), np.arange(9.0).reshape(3, 3))


def test_is_ref_distinguishes_a_pointer_from_data():
    assert is_ref("artifact://abc.npz")
    assert not is_ref("a normal string")
    assert not is_ref(42)


# ======================================================== summaries

def test_the_summary_counts_nan_and_infinity():
    """An array that is 3% NaN and one that is entirely NaN look identical in shape alone."""
    array = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
    record = summarise(array)
    assert record["n_nan"] == 1
    assert record["n_inf"] == 2
    assert record["n_finite"] == 2
    assert record["min"] == 1.0 and record["max"] == 5.0


def test_an_all_nan_array_reports_that_rather_than_nan_statistics():
    record = summarise(np.full(10, np.nan))
    assert record["n_finite"] == 0
    assert "min" not in record
    assert "no finite values" in record["note"]


# ======================================================== accepted payloads

def test_a_physical_field_carries_its_grid_into_the_summary(store):
    field = PhysicalField(torch.arange(16, dtype=torch.float64).reshape(4, 4), units="K")
    handle = store.put(field, name="f")
    assert handle.summary["source"] == "PhysicalField"
    assert handle.summary["units"] == "K"
    assert "grid" in handle.summary


def test_torch_and_numpy_produce_the_same_artifact(store):
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert store.put(values).ref == store.put(torch.from_numpy(values)).ref


def test_a_nested_list_is_accepted(store):
    handle = store.put([[1.0, 2.0], [3.0, 4.0]], name="x")
    assert np.array_equal(store.load(handle), [[1.0, 2.0], [3.0, 4.0]])


def test_a_ragged_list_is_refused_rather_than_pickled(store):
    with pytest.raises(InvalidParameterError) as excinfo:
        store.put([[1.0, 2.0], [3.0]], name="x")
    assert "only this exact version of the code could read" in str(excinfo.value)


def test_an_arbitrary_object_is_refused(store):
    """An artifact only the writing code can read is not reproducible data."""
    with pytest.raises(InvalidParameterError) as excinfo:
        store.put({"not": "an array"}, name="x")
    assert "does not pickle arbitrary objects" in str(excinfo.value)


# ======================================================== dereferencing

def test_dereference_replaces_refs_and_leaves_everything_else(store):
    handle = store.put(np.arange(4.0), name="x")
    payload = {"field": handle.ref, "levels": [850, 500], "name": "keep me"}
    resolved = dereference(payload, store)
    assert np.array_equal(resolved["field"], np.arange(4.0))
    assert resolved["levels"] == [850, 500]
    assert resolved["name"] == "keep me"


def test_resolve_value_dereferences_a_handle(store, monkeypatch):
    """The engine seam: an action written before the store existed still gets an array."""
    from src.artifact_store import store as store_module
    from src.experiment_engine.engine import resolve_value

    monkeypatch.setattr(store_module, "_DEFAULT", store)
    handle = store.put(np.arange(12.0).reshape(3, 4), name="x")
    assert resolve_value(handle.ref, {}, {}).shape == (3, 4)
    assert resolve_value(handle.to_dict(), {}, {}).shape == (3, 4)
    # And a plain value must be untouched - the branch order matters, because a bare
    # "artifact://..." is a string and the `{placeholder}` handling would return it unchanged.
    assert resolve_value("hello", {}, {}) == "hello"
    assert resolve_value({"a": 1}, {}, {}) == {"a": 1}


# ======================================================== housekeeping

def test_clear_removes_everything_and_reports_the_count(store):
    store.put(np.zeros((2, 2)), name="a")
    store.put(np.ones((2, 2)), name="b")
    assert store.clear() == 2
    assert store.stats()["n_artifacts"] == 0


def test_stats_reports_size(store):
    store.put(np.zeros((100, 100)), name="a")
    stats = store.stats()
    assert stats["n_artifacts"] == 1
    assert stats["bytes"] > 0


# ======================================================== complex payloads (defect D38)

def test_a_complex_array_is_summarised_on_its_magnitude():
    """D38, found while building Phase 4B.

    The original `summarise` called `float(values.min())` unconditionally. For a complex array
    that does not raise: numpy casts to real, discards the imaginary part, and emits a warning
    nobody reads. `[1+2j, 3-1j]` reported `min = 1.0`. Every DTCWT coefficient field is
    complex, so the lineage rows of an entire phase would have described only the real part.
    """
    record = summarise(np.array([1 + 2j, 3 - 1j]))

    assert record["is_complex"] is True
    assert record["statistic_of"] == "magnitude"
    assert record["min"] == pytest.approx(abs(1 + 2j))   # sqrt(5)  ~ 2.236
    assert record["max"] == pytest.approx(abs(3 - 1j))   # sqrt(10) ~ 3.162
    # And the value the old code reported, so the regression is named rather than described:
    assert record["min"] != 1.0
    assert record["mean_real"] == pytest.approx(2.0)
    assert record["mean_imag"] == pytest.approx(0.5)


def test_a_real_array_is_not_labelled_complex():
    record = summarise(np.array([1.0, 3.0]))
    assert "is_complex" not in record
    assert record["min"] == 1.0


def test_a_complex_artifact_round_trips_exactly(store):
    values = np.array([[1 + 2j, 3 - 1j], [0.5j, -4 + 0j]])
    handle = store.put(values, name="complex")
    assert np.array_equal(store.load(handle), values)
    assert handle.dtype == "complex128"


# ======================================================== self-describing artifacts

def test_a_stored_sequence_keeps_its_time_axis(store):
    """A bare `(T, H, W)` array is not a FieldSequence.

    The time coordinate is the whole reason the class exists, and a caller who rebuilt one by
    assuming a regular cadence would silently mis-date every frame of an irregular record -
    which is precisely the record type `is_regular` exists to flag.
    """
    irregular = [0.0, 21600.0, 90000.0, 100000.0]
    fields = [PhysicalField(torch.full((8, 8), float(i), dtype=torch.float64))
              for i in range(4)]
    sequence = FieldSequence(fields, irregular)

    handle = store.put(sequence, name="s")
    restored = store.load_sequence(handle)

    assert np.array_equal(restored.times_seconds, np.array(irregular))
    assert restored.is_regular is False
    assert torch.equal(restored.to_tensor(), sequence.to_tensor())
    assert tuple(restored.grid.shape) == (8, 8)


def test_a_stored_coefficient_field_keeps_its_labels(store):
    from src.transform_engine.coefficient_field import decompose_sequence

    fields = [PhysicalField(torch.randn(32, 32, dtype=torch.float64)) for _ in range(2)]
    sequence = FieldSequence(fields, [0.0, 21600.0])
    field = decompose_sequence(sequence, "dtcwt", {"levels": 2})

    restored = store.load_coefficient_field(store.put(field, name="cf"))

    assert restored.scales == field.scales
    assert restored.orientations == field.orientations
    assert restored.resampled_to_parent is True
    assert restored.native_shapes == field.native_shapes
    assert torch.equal(restored.data, field.data)


def test_loading_the_wrong_kind_is_refused_by_name(store):
    """Rebuilding a sequence from a coefficient field would mean inventing axes."""
    handle = store.put(np.zeros((4, 8, 8)), name="bare")
    with pytest.raises(ArtifactError) as excinfo:
        store.load_sequence(handle)
    assert "bare array" in str(excinfo.value)


def test_a_bare_array_has_no_auxiliary_axes(store):
    """Not an error - only a statement that this artifact was never anything richer."""
    assert store.load_aux(store.put(np.zeros((2, 2)), name="x")) == {}


def test_the_auxiliary_members_do_not_bloat_the_handle(store):
    """The axes go in the archive, not in the database row."""
    fields = [PhysicalField(torch.zeros(16, 16, dtype=torch.float64)) for _ in range(200)]
    sequence = FieldSequence(fields, np.arange(200) * 3600.0)
    handle = store.put(sequence, name="long")
    assert handle.check_size() < MAX_HANDLE_BYTES
