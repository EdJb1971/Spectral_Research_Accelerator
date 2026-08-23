"""Content-addressed artifact store (roadmap T4A.3, standard E5).

**The wall this removes.** Lineage `value` columns and inter-step `step_outputs` carry full
payloads as nested Python lists - `analyze_boundary` puts an entire padded field into its node.
`architecture.md` has listed that as a hard scaling limit since the first audit, and Phase 4B is
where it stops being theoretical: a `CoefficientField` over 10 frames x 64x64 x 4 scales x
6 orientations is about **10 million floats**. As JSON-encoded nested lists that is roughly
200 MB in a database column, per step, per run.

So steps exchange an `ArtifactHandle` - a reference plus shape, dtype, checksum and a small
statistical summary - and the bytes live on disk under `artifacts/`, keyed by the SHA-256 of
their own content.

**Content addressing is not a filing convention here, it is the deduplication and the integrity
check at once.** Two steps that produce identical output write one file. A handle names the
exact bytes it was created from, so `load` can verify them and a corrupted or truncated artifact
is an error rather than a subtly wrong array. That check is cheap and it is the difference
between "the file is there" and "the file is what the lineage record says it is".

**What goes in the database is the handle and the summary, never the payload.** The summary is
deliberately small and deliberately *statistical* - shape, dtype, min/max/mean/std, NaN and
infinity counts - because a reader looking at a lineage row six months later needs to know
whether an array was all zeros or full of NaNs without materialising 200 MB to find out.

One format, ``.npz``: portable, readable by anything with numpy, compressed, and
- unlike a torch checkpoint - not a pickle. Torch tensors are converted on the way in and
handed back through `load_tensor`, which is a deliberate narrowing: a `.pt` file can only be
read by the torch version that wrote it, which is not a property reproducible data should have.

**An artifact carries its own axes.** A bare `(T, H, W)` array is not a `FieldSequence` and a
bare `(T, S, O, Y, X)` array is not a `CoefficientField`: the time coordinate, the grid, the
scale and orientation labels are what make the numbers mean anything, and an artifact that
dropped them would force every reader to find the lineage row that described it. So the archive
carries auxiliary members alongside `array` - the time axis as an array, the rest as one JSON
member - and `load_sequence` / `load_coefficient_field` rebuild the real object.

Both are written **atomically** (temporary file, then rename), because a store that can be left
holding a half-written artifact by a crash or a killed sweep is a store whose checksums start
failing for reasons nobody can reconstruct.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.errors import InvalidParameterError, SpectralEarthError

#: Where artifacts live. Content-addressed, so this directory is safe to share between runs
#: and safe to delete wholesale (at the cost of recomputation, never of correctness).
DEFAULT_ARTIFACT_DIR = "artifacts"

#: A handle is meant to fit comfortably in a database row alongside the rest of a lineage
#: record. T4A.3's acceptance criterion is 4 KB per row; this is the budget for the handle
#: itself, checked by `ArtifactHandle.check_size`.
MAX_HANDLE_BYTES = 4096

#: Prefix marking a string as an artifact reference in `step_outputs` and lineage values.
#: Chosen to be visually unmistakable in a JSON dump: someone reading a lineage row must be
#: able to tell instantly that they are looking at a pointer and not at data.
REF_PREFIX = "artifact://"


class ArtifactError(SpectralEarthError):
    """Raised when an artifact cannot be written, found, or verified."""

    status_code = 500


def _cache_dir(value: Optional[str] = None) -> str:
    """Resolved at call time, not bound into a default argument (the D-lesson from T3.5.18)."""
    return value if value is not None else DEFAULT_ARTIFACT_DIR


@dataclass(frozen=True)
class ArtifactHandle:
    """A reference to stored bytes, plus everything a reader needs *without* loading them."""

    ref: str
    shape: Tuple[int, ...]
    dtype: str
    sha256: str
    format: str
    bytes_on_disk: int
    summary: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        record["shape"] = list(self.shape)
        return record

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "ArtifactHandle":
        known = {f for f in ("ref", "shape", "dtype", "sha256", "format",
                             "bytes_on_disk", "summary")}
        missing = known - set(record) - {"summary"}
        if missing:
            raise InvalidParameterError(
                "handle", sorted(record), "an artifact handle containing %s" % sorted(missing))
        return cls(
            ref=record["ref"], shape=tuple(record["shape"]), dtype=record["dtype"],
            sha256=record["sha256"], format=record["format"],
            bytes_on_disk=int(record["bytes_on_disk"]), summary=record.get("summary", {}))

    def check_size(self, limit: int = MAX_HANDLE_BYTES) -> int:
        """Assert the handle itself is small. Returns its encoded size.

        A handle that grew a payload back into its summary would defeat the entire point
        while still looking like a handle, so the budget is checked rather than assumed.
        """
        encoded = len(json.dumps(self.to_dict(), default=str).encode("utf-8"))
        if encoded > limit:
            raise ArtifactError(
                "this artifact handle encodes to %d bytes, over the %d-byte budget. Something "
                "has put payload into the summary - the summary is for statistics, not data."
                % (encoded, limit), encoded_bytes=encoded, limit=limit)
        return encoded


def is_ref(value: Any) -> bool:
    """Whether a value is an artifact reference string."""
    return isinstance(value, str) and value.startswith(REF_PREFIX)


# --------------------------------------------------------------------------- summarising

def summarise(array: np.ndarray) -> Dict[str, Any]:
    """A small statistical description that survives into the lineage row.

    **NaN and infinity counts are first-class, not diagnostics.** An array that is 3% NaN
    produces a mean of NaN and poisons everything downstream; an array that is *entirely*
    NaN looks identical in shape and dtype to a good one. Recording the counts means a
    lineage row shows which happened, months later, without reloading the payload.

    **Complex arrays are summarised on their magnitude, and say so** (defect D38). The first
    version of this function called `float(values.min())` unconditionally, which for a complex
    array does not raise - numpy casts to real and discards the imaginary part with a warning
    nobody reads. `[1+2j, 3-1j]` reported `min = 1.0`. Every DTCWT coefficient field is
    complex, so Phase 4B would have written lineage rows whose statistics silently described
    only the real part.
    """
    is_complex = array.dtype.kind == "c"
    floating = array.dtype.kind in "fc"
    finite = np.isfinite(array)
    n_finite = int(finite.sum())
    record: Dict[str, Any] = {
        "n_elements": int(array.size),
        "n_nan": int(np.isnan(array).sum()) if floating else 0,
        "n_inf": int(np.isinf(array).sum()) if floating else 0,
        "n_finite": n_finite,
    }
    if is_complex:
        record["is_complex"] = True
        record["statistic_of"] = "magnitude"
    if n_finite:
        values = array[finite]
        if is_complex:
            magnitude = np.abs(values)
            record.update({
                "min": float(magnitude.min()),
                "max": float(magnitude.max()),
                "mean": float(magnitude.mean()),
                "std": float(magnitude.std()),
                "mean_real": float(values.real.mean()),
                "mean_imag": float(values.imag.mean()),
            })
        else:
            record.update({
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            })
    else:
        record["note"] = ("no finite values - every element is NaN or infinite, so no "
                          "statistics are reportable")
    return record


# --------------------------------------------------------------------------- the store

class ArtifactStore:
    """Content-addressed storage for arrays too large to travel through JSON."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = _cache_dir(directory)

    # -- paths ------------------------------------------------------------------

    def _path_for(self, digest: str, fmt: str) -> str:
        # Two-character shard prefix: a flat directory with tens of thousands of entries is
        # slow to list on every filesystem this is likely to meet, and unbearable on NTFS.
        return os.path.join(self.directory, digest[:2], "%s.%s" % (digest, fmt))

    def path_of(self, handle_or_ref: Union[ArtifactHandle, str]) -> str:
        ref = handle_or_ref.ref if isinstance(handle_or_ref, ArtifactHandle) else handle_or_ref
        if not is_ref(ref):
            raise InvalidParameterError(
                "ref", ref, "a reference beginning %r" % REF_PREFIX)
        body = ref[len(REF_PREFIX):]
        digest, _, fmt = body.partition(".")
        if not digest or not fmt:
            raise InvalidParameterError("ref", ref, "a reference of the form "
                                                    "artifact://<sha256>.<format>")
        return self._path_for(digest, fmt)

    # -- writing ----------------------------------------------------------------

    def put(self, value: Any, name: str = "artifact",
            metadata: Optional[Dict[str, Any]] = None) -> ArtifactHandle:
        """Store an array or tensor and return its handle.

        Accepts a numpy array, a torch tensor, a `PhysicalField`, a `FieldSequence`, or a plain
        nested list. Anything else is refused by name rather than pickled: this store is for
        numeric payloads, and silently accepting arbitrary objects would turn a reproducible
        artifact into an unversioned pickle nobody can read without the exact code that wrote
        it.
        """
        array, fmt, extra, aux = _coerce(value)
        payload = _encode(array, fmt, aux)
        digest = hashlib.sha256(payload).hexdigest()
        path = self._path_for(digest, fmt)

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Atomic: write beside the target, then rename. A crash mid-write leaves a
            # temporary file, never a half-written artifact whose checksum will fail later
            # for reasons nobody can reconstruct.
            handle_fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            try:
                with os.fdopen(handle_fd, "wb") as fh:
                    fh.write(payload)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

        summary = summarise(array)
        summary["name"] = name
        summary.update(extra)
        if metadata:
            summary["metadata"] = metadata

        handle = ArtifactHandle(
            ref="%s%s.%s" % (REF_PREFIX, digest, fmt),
            shape=tuple(int(s) for s in array.shape),
            dtype=str(array.dtype),
            sha256=digest,
            format=fmt,
            bytes_on_disk=len(payload),
            summary=summary,
        )
        handle.check_size()
        return handle

    # -- reading ----------------------------------------------------------------

    def load(self, handle_or_ref: Union[ArtifactHandle, str],
             verify: bool = True) -> np.ndarray:
        """Load an artifact, verifying its checksum by default.

        `verify=False` exists for a hot loop that has already checked once; it is not the
        default, because an artifact whose bytes no longer match its handle is *silently* a
        different array, and every number derived from it would be wrong with nothing to
        indicate it.
        """
        path = self.path_of(handle_or_ref)
        ref = handle_or_ref.ref if isinstance(handle_or_ref, ArtifactHandle) else handle_or_ref
        if not os.path.exists(path):
            raise ArtifactError(
                "artifact %r is not in the store at %r. Content-addressed artifacts are safe "
                "to delete but not to move: if the store was cleared, the step that produced "
                "this must be re-run." % (ref, self.directory), ref=ref, path=path)
        with open(path, "rb") as fh:
            payload = fh.read()

        if verify:
            expected = ref[len(REF_PREFIX):].partition(".")[0]
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise ArtifactError(
                    "artifact %r fails its checksum: the file on disk hashes to %s. The bytes "
                    "have changed since they were written, so this is not the array the "
                    "lineage record describes." % (ref, actual[:16]), ref=ref)
            if isinstance(handle_or_ref, ArtifactHandle) and handle_or_ref.sha256 != actual:
                raise ArtifactError(
                    "handle claims sha256 %s but the stored bytes hash to %s"
                    % (handle_or_ref.sha256[:16], actual[:16]), ref=ref)

        fmt = ref[len(REF_PREFIX):].rpartition(".")[2]
        return _decode(payload, fmt)

    def load_tensor(self, handle_or_ref, verify: bool = True):
        import torch

        return torch.from_numpy(np.ascontiguousarray(self.load(handle_or_ref, verify)))

    # -- self-describing payloads -----------------------------------------------

    def load_aux(self, handle_or_ref, verify: bool = True) -> Dict[str, Any]:
        """The axes stored alongside the array: times, labels, grid, metadata.

        Returns `{}` for an artifact written as a bare array - which is not an error, only a
        statement that this artifact was never anything richer.
        """
        path = self.path_of(handle_or_ref)
        ref = handle_or_ref.ref if isinstance(handle_or_ref, ArtifactHandle) else handle_or_ref
        # Route through `load` first so the checksum is verified exactly once, by the code
        # that owns that check, rather than re-implemented here where it could drift.
        self.load(handle_or_ref, verify=verify)
        with open(path, "rb") as fh:
            payload = fh.read()
        fmt = ref[len(REF_PREFIX):].rpartition(".")[2]
        members = _decode_aux(payload, fmt)
        record: Dict[str, Any] = {}
        if "meta_json" in members:
            record.update(json.loads(str(members["meta_json"])))
        if "times_seconds" in members:
            record["times_seconds"] = np.asarray(members["times_seconds"], dtype=np.float64)
        return record

    def _require_kind(self, record: Dict[str, Any], kind: str, ref: Any):
        if record.get("kind") != kind:
            raise ArtifactError(
                "artifact %r was stored as %s, not as a %s. Rebuilding one from the other "
                "would mean inventing the axes the original never carried."
                % (ref, record.get("kind") or "a bare array", kind), ref=str(ref))

    def load_sequence(self, handle_or_ref, verify: bool = True):
        """Rebuild the `FieldSequence` - frames, grid and **time axis** - not a bare array.

        `load` returns `(T, H, W)` floats, which is not a sequence: the time coordinate is the
        whole reason `FieldSequence` exists, and a caller who reconstructed one by assuming a
        regular cadence would silently mis-date every frame of an irregular record.
        """
        from src.physical_core.field import PhysicalField
        from src.physical_core.grid import GridSpec
        from src.physical_core.sequence import FieldSequence

        record = self.load_aux(handle_or_ref, verify=verify)
        self._require_kind(record, "FieldSequence", handle_or_ref)
        array = self.load(handle_or_ref, verify=False)
        grid = GridSpec.from_provenance(record["grid"])
        import torch

        fields = [PhysicalField(torch.from_numpy(np.ascontiguousarray(frame)), grid=grid)
                  for frame in array]
        return FieldSequence(fields, record["times_seconds"],
                             metadata=record.get("metadata") or {},
                             split=record.get("split"))

    def load_coefficient_field(self, handle_or_ref, verify: bool = True):
        """Rebuild a `CoefficientField` with its scale and orientation labels intact.

        The native coefficients are **not** restored - they are not stored, because they are
        the largest part of the payload and are only needed for reconstruction. The rebuilt
        field therefore reports `reconstructable: False` and refuses `reconstruct()` by name
        rather than returning an inverse of the aligned view.
        """
        import torch

        from src.physical_core.grid import GridSpec
        from src.transform_engine.coefficient_field import CoefficientField

        record = self.load_aux(handle_or_ref, verify=verify)
        self._require_kind(record, "CoefficientField", handle_or_ref)
        array = self.load(handle_or_ref, verify=False)
        return CoefficientField(
            torch.from_numpy(np.ascontiguousarray(array)),
            wavelet_family=record["wavelet_family"],
            scales=record["scales"],
            orientations=record["orientations"],
            times=record["times_seconds"],
            grid=GridSpec.from_provenance(record["grid"]),
            source_variable=record.get("source_variable"),
            orientation_convention=record.get("orientation_convention", "unspecified"),
            resampled_to_parent=record.get("resampled_to_parent", False),
            native_shapes={int(k): tuple(v)
                           for k, v in (record.get("native_shapes") or {}).items()},
            native=None,
            config=record.get("config") or {},
            level=record.get("level"),
            level_axis=record.get("level_axis"),
            metadata={**(record.get("metadata") or {}), "restored_from_artifact": True})

    def exists(self, handle_or_ref) -> bool:
        try:
            return os.path.exists(self.path_of(handle_or_ref))
        except InvalidParameterError:
            return False

    # -- housekeeping ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Count and total size of the store, for the health endpoint and for operators."""
        count = 0
        total = 0
        if os.path.isdir(self.directory):
            for root, _dirs, files in os.walk(self.directory):
                for name in files:
                    if name.endswith(".tmp"):
                        continue
                    count += 1
                    total += os.path.getsize(os.path.join(root, name))
        return {"directory": self.directory, "n_artifacts": count,
                "bytes": total, "megabytes": round(total / 1e6, 3)}

    def clear(self) -> int:
        """Delete every artifact. Returns how many were removed.

        Safe by construction: artifacts are content-addressed derived data, so the worst
        consequence is recomputation. Lineage rows keep their handles and summaries, and a
        subsequent `load` of a cleared artifact raises an error naming the step to re-run
        rather than returning something wrong.
        """
        removed = self.stats()["n_artifacts"]
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory, ignore_errors=True)
        return removed


# --------------------------------------------------------------------------- coercion

def _coerce(value: Any) -> Tuple[np.ndarray, str, Dict[str, Any], Dict[str, np.ndarray]]:
    """Turn a supported payload into `(array, format, extra summary fields, aux members)`.

    `aux` is what lets an artifact describe itself: the axes that make the array meaningful,
    stored inside the archive rather than only in the lineage row that points at it.
    """
    # Imported lazily so this module does not force torch onto a caller that only has numpy.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the platform
        torch = None

    extra: Dict[str, Any] = {}
    aux: Dict[str, np.ndarray] = {}

    # PhysicalField / FieldSequence / CoefficientField are duck-typed rather than imported, to keep this module
    # free of a dependency on physical_core (which would make the import graph circular the
    # moment a field wants to store itself).
    if hasattr(value, "data") and hasattr(value, "scales") and hasattr(value, "orientations"):
        extra["source"] = "CoefficientField"
        extra["wavelet_family"] = value.wavelet_family
        extra["grid"] = value.grid.to_provenance()
        aux["times_seconds"] = np.asarray(value.times, dtype=np.float64)
        aux["meta_json"] = _meta_member({
            "kind": "CoefficientField",
            "wavelet_family": value.wavelet_family,
            "scales": list(value.scales),
            "orientations": list(value.orientations),
            "orientation_convention": value.orientation_convention,
            "grid": value.grid.to_provenance(),
            "source_variable": value.source_variable,
            "resampled_to_parent": value.resampled_to_parent,
            "native_shapes": {str(k): list(v) for k, v in value.native_shapes.items()},
            "level": value.level,
            "level_axis": value.level_axis,
            "config": dict(value.config),
            "metadata": dict(value.metadata),
        })
        value = value.data
    elif hasattr(value, "data") and hasattr(value, "grid") and hasattr(value, "coords"):
        extra["source"] = "PhysicalField"
        extra["grid"] = value.grid.to_provenance()
        extra["units"] = getattr(value, "units", None)
        value = value.data
    elif hasattr(value, "fields") and hasattr(value, "times_seconds"):
        extra["source"] = "FieldSequence"
        extra["n_frames"] = len(value.fields)
        extra["grid"] = value.grid.to_provenance()
        extra["units"] = value.units
        aux["times_seconds"] = np.asarray(value.times_seconds, dtype=np.float64)
        aux["meta_json"] = _meta_member({
            "kind": "FieldSequence",
            "grid": value.grid.to_provenance(),
            "units": value.units,
            "time_kind": value.time_kind,
            "split": value.split,
            "metadata": dict(value.metadata),
        })
        value = value.to_tensor()

    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array, "npz", extra, aux
    if isinstance(value, np.ndarray):
        return value, "npz", extra, aux
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except ValueError:
            # numpy >= 1.24 raises here rather than silently building an object array. Both
            # outcomes are handled, because the message a researcher gets should not depend
            # on which numpy they happen to have installed.
            array = np.empty((), dtype=object)
        if array.dtype == object:
            raise InvalidParameterError(
                "value", "ragged nested sequence",
                "a rectangular array. A ragged list cannot be stored as an array, and "
                "storing it as an object pickle would produce an artifact only this exact "
                "version of the code could read")
        return array, "npz", extra, aux
    raise InvalidParameterError(
        "value", type(value).__name__,
        "a numpy array, torch tensor, PhysicalField, FieldSequence or nested numeric list. "
        "This store deliberately does not pickle arbitrary objects: an artifact that only "
        "the code that wrote it can read is not reproducible data")


def _meta_member(record: Dict[str, Any]) -> np.ndarray:
    """One JSON string as a 0-d unicode array - an npz member, never a pickle."""
    return np.array(json.dumps(record, default=str))


def _encode(array: np.ndarray, fmt: str,
            aux: Optional[Dict[str, np.ndarray]] = None) -> bytes:
    if fmt != "npz":
        raise InvalidParameterError("format", fmt, "npz")
    if aux and "array" in aux:
        raise InvalidParameterError(
            "aux", "array", "member names other than 'array', which holds the payload")
    buffer = io.BytesIO()
    # `savez_compressed` rather than `savez`: coefficient fields are highly compressible and
    # the store is the thing that stops the database being the bottleneck, so trading CPU for
    # bytes is the right way round here. Determinism matters too - the digest must depend only
    # on the data, so the archive member name is fixed rather than derived from a variable.
    np.savez_compressed(buffer, array=np.ascontiguousarray(array), **(aux or {}))
    return buffer.getvalue()


def _decode(payload: bytes, fmt: str) -> np.ndarray:
    if fmt != "npz":
        raise InvalidParameterError("format", fmt, "npz")
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        return loaded["array"]


def _decode_aux(payload: bytes, fmt: str) -> Dict[str, np.ndarray]:
    """Every member except the payload. `allow_pickle=False` throughout, deliberately."""
    if fmt != "npz":
        raise InvalidParameterError("format", fmt, "npz")
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files if name != "array"}


# --------------------------------------------------------------------------- default store

_DEFAULT: Optional[ArtifactStore] = None


def get_store(directory: Optional[str] = None) -> ArtifactStore:
    """The process-wide store, or a fresh one for an explicit directory."""
    global _DEFAULT
    if directory is not None:
        return ArtifactStore(directory)
    if _DEFAULT is None:
        _DEFAULT = ArtifactStore()
    return _DEFAULT


def set_default_store(store: Optional[ArtifactStore]) -> None:
    """Rebind the process-wide store (tests, and a deployment with its own artifact path)."""
    global _DEFAULT
    _DEFAULT = store


def dereference(value: Any, store: Optional[ArtifactStore] = None) -> Any:
    """Replace artifact references with their arrays, leaving everything else untouched.

    This is what `resolve_value` gains: a step declaring `{previous.field}` gets the array,
    not the pointer, without every action having to know the store exists.
    """
    store = store or get_store()
    if is_ref(value):
        return store.load(value)
    if isinstance(value, dict):
        if is_ref(value.get("ref")):
            return store.load(value["ref"])
        return {k: dereference(v, store) for k, v in value.items()}
    if isinstance(value, list):
        return [dereference(v, store) for v in value]
    return value
