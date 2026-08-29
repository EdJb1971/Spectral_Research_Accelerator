"""Cloud-native ERA5 over Zarr, with a rechunked local cache (roadmap T3.5.18, standard E2).

**Why Zarr and not a NetCDF download.** WeatherBench 2 publishes ERA5 as public Zarr on GCS.
`xarray.open_zarr` over `fsspec` opens the whole archive lazily and fetches only the chunks a
crop touches, so "crop aggressively" stops being a separate pipeline step and becomes simply
how the store works. No credentials are needed: the buckets are readable anonymously.

**The trap, now measured rather than predicted.** SpectralEarth's access pattern is *many
timesteps over a small region*. WeatherBench 2's 0.25 degree stores are chunked
``(1, 13, 721, 1440)`` - one timestep, every pressure level, the whole planet, **54.0 MB per
chunk**. A 256x256 four-level crop wants 1.05 MB per timestep. Measured against the real store
(`VERIFICATION.md`, slice 11): 2 timesteps of exactly that crop took 16.6 s and moved
**108 MB to deliver 2.11 MB - an amplification of 51.1x**, which is within 0.1% of the figure
`assess_access_pattern` predicts from chunk metadata alone, before transferring anything.

So the adapter has two stages and the second is not optional:

1.  **Query** the remote store lazily for the requested region, window, levels and variables.
2.  **Materialise once** into a local Zarr cache **rechunked time-contiguous for that region**,
    content-hashed, with bytes-transferred recorded. Everything downstream reads the cache.

**Selecting fewer levels does not reduce transfer, and that is the whole point of stage 2.**
Level is *inside* the chunk on the 0.25 degree stores, so asking for 4 of 13 levels still pulls
all 13. Only the variable axis and the time axis actually reduce what crosses the network.
A researcher cannot be expected to know that; the adapter reports it.

**Provenance (E5).** A crop is exactly specifiable as ``{store URI, variables, time range,
bbox, levels}`` plus a content hash of the materialised snapshot. That round-trips through the
lineage record, so any finding is re-materialisable by anyone with an internet connection -
categorically stronger than "someone put a `.nc` in a folder".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.core.errors import DataSourceError, FieldTooSmallError, InvalidParameterError
from src.core.level_axis import PRESSURE_HPA
# `GriddedStore` and `store_for` are re-exported deliberately rather than used here: this
# module is the adapter facade a caller already has in hand, and making them reach for a
# second import to name the thing they just looked up is friction with no benefit (TG10.1).
from src.data_layer.stores import (  # noqa: F401  (GriddedStore, store_for re-exported)
    GRIDDED_STORES, CatalogueView, GriddedStore, catalogue_payload,
    register_builtin_stores, store_for, uri_for)
# A real extension module, kept outside the catalogue and adapter implementations. It loads
# checked-in probes and registers GLORYS without importing a credential-minting client.
from src.data_layer import glorys_store as _glorys_store  # noqa: F401,E402

# --------------------------------------------------------------------------- catalogue

#: Public WeatherBench 2 ERA5 stores, and every other registered gridded store, as a
#: **read-only mapping view** over `stores.GRIDDED_STORES` (TG10.1, standard E1).
#:
#: This was a module-level dictionary of four ERA5 stores until TG10.1, which is why its
#: value shape is ERA5's own. The entries now live in a registry, each declaring the domain
#: it belongs to, how it is reached, its vertical axis name and how its chunk figures were
#: obtained; `to_dict` keeps every key the dictionary carried, so the provenance, overlap
#: and reporting readers below are unaffected. `describe_store` still works on any Zarr URI,
#: listed or not, and an uncatalogued name is still passed through as a raw URI.
#:
#: The view does not support assignment. Writing to it would put a store in the catalogue
#: without a domain, an access requirement or a chunk record, which are the three things
#: `register_store` exists to require.
CATALOGUE: Mapping[str, Dict[str, Any]] = CatalogueView()

#: Where materialised crops live. Content-addressed, so two identical specs share one entry.
DEFAULT_CACHE_DIR = os.path.join("data", "zarr_cache")


def _cache_dir(value: Optional[str] = None) -> str:
    """Resolve the cache directory **at call time**.

    Every cache function takes ``cache_dir=None`` rather than
    ``cache_dir=DEFAULT_CACHE_DIR``, because a default argument binds once at import. With
    the module-level default baked into the signature, reconfiguring the cache location -
    which a test, a notebook or a deployment all legitimately do - changed the constant and
    changed nothing else, so `can_serve` kept consulting the original directory. That
    reported "no crop has been materialised" about a cache that was full.
    """
    return value if value is not None else DEFAULT_CACHE_DIR

#: Amplification above which an access pattern is called hostile and warned about.
#: 4x is the point at which the rechunked cache pays for itself on a second read; below it
#: the extra local copy is arguably not worth the disk.
HOSTILE_AMPLIFICATION = 4.0

# --------------------------------------------------------------------------- R13 geometry

#: Taps in the DTCWT q-shift filter, which sets the edge exclusion. 14 is the Kingsbury
#: q-shift length used by `transform_engine/dtcwt.py`; the SWT's db2/db3 are shorter, so
#: sizing from 14 is conservative in the right direction.
DEFAULT_FILTER_TAPS = 14

#: Smallest valid interior worth analysing, in pixels per side. This is a judgement, so it
#: is a named constant rather than a literal: with the accumulated cascade support, 128
#: valid pixels requires 512 for four levels and 1024 for five (D44).
MIN_VALID_INTERIOR = 128


def edge_exclusion(level: int, taps: int = DEFAULT_FILTER_TAPS) -> int:
    """Pixels contaminated per side at an undecimated level (roadmap R13).

    The recursive cascade support is ``1 + (L - 1) * (2**j - 1)`` pixels. Conservatively exclude
    ``support // 2`` per side, which is equivalent to taking the ceiling of the possibly
    half-integer radius.  This deliberately matches
    :func:`transform_engine.stationary.valid_interior_halfwidth`; flooring would declare a
    contaminated pixel valid for an even-length filter at level 1 (D41).

    For L=14 the margins at levels 1-4 are 7, 20, 46 and 98 pixels per side.
    """
    if level < 1:
        raise InvalidParameterError("level", level, "an integer >= 1")
    support = 1 + (taps - 1) * (2 ** level - 1)
    return int(support // 2)


def valid_interior(size: int, level: int, taps: int = DEFAULT_FILTER_TAPS) -> int:
    """Analysable width after excluding contaminated edges. May be zero or negative."""
    return int(size - 2 * edge_exclusion(level, taps))


def minimum_crop_size(levels: int, taps: int = DEFAULT_FILTER_TAPS,
                      min_interior: int = MIN_VALID_INTERIOR) -> int:
    """Smallest crop that leaves ``min_interior`` valid pixels at the *coarsest* level.

    Rounded up to a power of two, because the dyadic transforms want one and because it
    matches how a researcher thinks about crop sizes. With the accumulated cascade support it
    gives 512 for four levels and 1024 for five (D44).
    """
    needed = min_interior + 2 * edge_exclusion(levels, taps)
    size = 1
    while size < needed:
        size *= 2
    return size


def check_crop_size(height: int, width: int, levels: int,
                    taps: int = DEFAULT_FILTER_TAPS) -> Dict[str, Any]:
    """Refuse a crop too small for the requested number of scales, naming the minimum.

    **Why this refuses rather than warns.** A 64x64 crop has *zero* valid interior at level 4
    and about 12x12 px at level 3. Cross-scale analysis on it is not merely noisy, it is
    arithmetically impossible - and every coefficient near the edge looks like a strong,
    localised, edge-aligned feature, which is precisely what a discovery engine would report.
    Silently proceeding would manufacture findings out of padding.
    """
    minimum = minimum_crop_size(levels, taps)
    interiors = {j: valid_interior(min(height, width), j, taps)
                 for j in range(1, levels + 1)}
    if min(height, width) < minimum:
        raise FieldTooSmallError(
            "a %d-level cross-scale analysis" % levels,
            (height, width),
            (minimum, minimum),
            remedy=(
                "at level %d a %d-tap filter contaminates %d px per side, leaving a valid "
                "interior of %d px from a %d px crop. R13 requires at least %d px of valid "
                "interior, so the crop must be at least %dx%d. Constrain the number of "
                "frames or the bank breadth instead - never the grid: a smaller grid does "
                "not make the analysis cheaper, it makes it wrong."
                % (levels, taps, edge_exclusion(levels, taps),
                   valid_interior(min(height, width), levels, taps),
                   min(height, width), MIN_VALID_INTERIOR, minimum, minimum)),
            valid_interior_by_level=interiors,
        )
    return {
        "ok": True,
        "levels": levels,
        "filter_taps": taps,
        "minimum_size": minimum,
        "edge_exclusion_by_level": {j: edge_exclusion(j, taps)
                                    for j in range(1, levels + 1)},
        "valid_interior_by_level": interiors,
    }


# --------------------------------------------------------------------------- crop spec

@dataclass(frozen=True)
class CropSpec:
    """The exact, hashable specification of a regional crop (standard E5).

    Everything a re-materialisation needs and nothing that varies between machines - no local
    paths, no timestamps. Two identical specs therefore produce the same ``content_key`` on
    any machine, which is what makes the cache shareable and the provenance record a
    reproduction recipe rather than a description.
    """

    store: str
    variables: Tuple[str, ...]
    time_start: str
    time_end: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Numeric values on the declared vertical axis. ERA5 uses integer pressure levels;
    # GLORYS uses fractional negative-metre elevations. Keeping the historical field name
    # preserves every existing crop identity while allowing the axis declaration to be real.
    levels: Tuple[Union[int, float], ...] = ()
    #: Declared analysis depth, used for the R13 floor check. Not part of the data selection.
    n_levels_analysis: int = 4
    #: Name of the store's vertical dimension - `level` for ERA5's pressure levels, `depth`
    #: for an ocean product. The one axis TG10.1 generalises, and the only generalisation the
    #: crop path needs to accept a non-atmospheric gridded store: everything else about a
    #: region-and-time slice is already store-agnostic. It keeps ERA5's default because every
    #: catalogued store today is ERA5 and a required argument here would break every existing
    #: provenance record, but a store that disagrees says so through `vertical_dim_for_store`.
    vertical_dim: str = "level"

    def __post_init__(self) -> None:
        if not self.variables:
            raise InvalidParameterError("variables", self.variables,
                                        "at least one variable name")
        if self.lat_min >= self.lat_max:
            raise InvalidParameterError(
                "lat_min/lat_max", (self.lat_min, self.lat_max),
                "lat_min < lat_max, in degrees north")
        if self.lon_min >= self.lon_max:
            raise InvalidParameterError(
                "lon_min/lon_max", (self.lon_min, self.lon_max),
                "lon_min < lon_max, in degrees east")
        if not str(self.vertical_dim).strip():
            raise InvalidParameterError(
                "vertical_dim", self.vertical_dim,
                "the name of the store's vertical dimension, e.g. 'level' or 'depth'")
        for value in self.levels:
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise InvalidParameterError(
                    "levels", self.levels,
                    "finite numeric values on the declared vertical axis")

    @property
    def uri(self) -> str:
        """Resolve a catalogue id, or pass a raw URI through unchanged."""
        return uri_for(self.store) or self.store

    def canonical(self) -> Dict[str, Any]:
        """Order-independent, machine-independent form - the thing that gets hashed."""
        record: Dict[str, Any] = {
            "store": self.store,
            "uri": self.uri,
            "variables": sorted(self.variables),
            "time_start": self.time_start,
            "time_end": self.time_end,
            "bbox": [self.lat_min, self.lat_max, self.lon_min, self.lon_max],
            "levels": sorted(self.levels),
        }
        # The vertical dimension name enters the key **only when it is not ERA5's**, and that
        # is a compatibility decision taken deliberately rather than a purity lapse. Adding it
        # unconditionally would change the content key of every crop ever materialised,
        # orphaning the whole local cache and making every recorded provenance record resolve
        # to a different key than the one it names - a high price for a distinction that
        # distinguishes nothing, since `store` is already in the key and the store determines
        # its own vertical axis. A store whose axis is not `level` is fully separated.
        if self.vertical_dim != "level":
            record["vertical_dim"] = self.vertical_dim
        return record

    def content_key(self) -> str:
        """Stable 16-hex-character key. Sorted keys and separators so it cannot drift."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_provenance(self) -> Dict[str, Any]:
        """The lineage record. **Byte-identical to the pre-TG10.1 record for an ERA5 crop.**

        `vertical_dim` is omitted when it is `level`, for the same reason `canonical()` omits
        it and for one sharper reason found by defect D63: this record is embedded in
        authenticated artefacts - a gate campaign's preregistration is fingerprinted over it -
        so emitting a new key retroactively changed the fingerprint of a signed record that
        was written before the field existed. A schema that grows under an artefact already
        signed is not a schema; a store whose axis is not `level` still says so.
        """
        record = dict(asdict(self))
        record["variables"] = list(self.variables)
        record["levels"] = list(self.levels)
        if self.vertical_dim == "level":
            record.pop("vertical_dim", None)
        record["uri"] = self.uri
        record["content_key"] = self.content_key()
        return record

    @classmethod
    def from_provenance(cls, record: Dict[str, Any]) -> "CropSpec":
        """Rebuild a spec from a lineage record. The round trip is asserted in the tests.

        Extra keys are ignored on purpose: the lineage record also carries the resolved URI,
        the content key and the transfer statistics, none of which are part of the *request*.
        Failing on them would make provenance records un-replayable the moment anything was
        added to them.
        """
        fields = {"store", "variables", "time_start", "time_end", "lat_min", "lat_max",
                  "lon_min", "lon_max", "levels", "n_levels_analysis", "vertical_dim"}
        kwargs = {k: v for k, v in record.items() if k in fields}
        missing = fields - set(kwargs) - {"levels", "n_levels_analysis", "vertical_dim"}
        if missing:
            raise InvalidParameterError(
                "record", sorted(record), "a crop record containing %s" % sorted(missing))
        kwargs["variables"] = tuple(kwargs["variables"])
        kwargs["levels"] = tuple(kwargs.get("levels", ()))
        return cls(**kwargs)


def vertical_dim_for_store(store: str, default: str = "level") -> str:
    """The vertical axis name a catalogued store declares, or `default` for a raw URI.

    A convenience, not an inference: it reads a declaration a registration already made. For
    an uncatalogued URI there is nothing to read, so the caller's default stands and the
    caller remains responsible for it - a name guessed from the data would be exactly the
    axis-role inference standard E14 exists to forbid.
    """
    if store in GRIDDED_STORES:
        declared = GRIDDED_STORES.get(store).vertical_dim
        if declared:
            return declared
    return default


def crop_for_store(store: str, **kwargs: Any) -> "CropSpec":
    """Build a `CropSpec` whose vertical axis comes from the store's own declaration."""
    kwargs.setdefault("vertical_dim", vertical_dim_for_store(store))
    return CropSpec(store=store, **kwargs)


def _parse_level(text: str) -> Union[int, float]:
    """One value on a store's vertical axis, integer-first (D72).

    Integer-first is not a style choice. ERA5's pressure levels are integers and enter the
    content key as integers, so parsing `850` as `850.0` would move every pinned crop identity
    and orphan the cache. A value that is not an integer literal -- a GLORYS elevation such as
    `-0.49402499198913574` -- is kept as a float, which `CropSpec.levels` has accepted since
    D68 and which the CLI alone still refused.
    """
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        raise InvalidParameterError(
            "--levels", stripped,
            "a value on the store's declared vertical axis: an integer pressure level such as "
            "850, or a fractional value such as an ocean elevation in metres") from None


# --------------------------------------------------------------------------- byte counting

class CountingStore(MutableMapping):
    """A Zarr store wrapper that counts the bytes actually fetched.

    **Why the count is taken here and not from a timer or a progress bar.** The acceptance
    criterion is that a repeat request hits the cache with *zero network traffic*, and the
    only way to assert zero rather than "it felt fast" is to count. Wrapping the store also
    means the count includes the metadata reads (`.zmetadata`, `.zarray`) that a naive
    data-only measurement misses, so it never understates.

    ``getitems`` is delegated rather than left to the `MutableMapping` default, because zarr
    uses it to fetch many chunks concurrently; without it every read would serialise and the
    measured throughput would be an artefact of this wrapper.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.bytes_read = 0
        self.keys_read = 0
        self.bytes_written = 0

    # -- counted reads
    def __getitem__(self, key):
        value = self.inner[key]
        self.bytes_read += len(value) if value is not None else 0
        self.keys_read += 1
        return value

    def getitems(self, keys, **kwargs):
        getter = getattr(self.inner, "getitems", None)
        if getter is None:
            out = {}
            for key in keys:
                try:
                    out[key] = self[key]
                except KeyError:
                    pass
            return out
        result = getter(keys, **kwargs)
        for value in result.values():
            self.bytes_read += len(value) if value is not None else 0
            self.keys_read += 1
        return result

    # -- pass-through
    def __setitem__(self, key, value):
        self.bytes_written += len(value) if value is not None else 0
        self.inner[key] = value

    def __delitem__(self, key):
        del self.inner[key]

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def __contains__(self, key):
        return key in self.inner

    def listdir(self, path=""):
        return self.inner.listdir(path) if hasattr(self.inner, "listdir") else []

    def rmdir(self, path=""):
        if hasattr(self.inner, "rmdir"):
            self.inner.rmdir(path)

    def report(self) -> Dict[str, Any]:
        return {
            "bytes_read": int(self.bytes_read),
            "megabytes_read": round(self.bytes_read / 1e6, 3),
            "chunks_read": int(self.keys_read),
        }


# --------------------------------------------------------------------------- store opening

def _storage_options(uri: str, storage_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    options = dict(storage_options or {})
    # WeatherBench 2's buckets are public. Without `token="anon"` gcsfs looks for default
    # credentials and fails with an authentication error, which reads as "you need a Google
    # account" when in fact nothing is needed at all - a wrong and discouraging message.
    if uri.startswith("gs://") and "token" not in options:
        options["token"] = "anon"
    return options


def open_store(uri: str, storage_options: Optional[Dict[str, Any]] = None) -> CountingStore:
    """Open a Zarr store (local path or remote URI) wrapped in a byte counter."""
    import zarr

    if "://" in uri:
        try:
            inner = zarr.storage.FSStore(uri, **_storage_options(uri, storage_options))
        except Exception as exc:
            raise DataSourceError(
                "could not open the remote Zarr store %r (%s: %s). Check the URI and that "
                "`fsspec` has a driver for its scheme - for gs:// that is `gcsfs`."
                % (uri, type(exc).__name__, exc), uri=uri)
    else:
        if not os.path.exists(uri):
            raise DataSourceError(
                "no Zarr store at %r. A local store is a *directory* containing .zarray "
                "metadata, not a single file." % uri, uri=uri)
        inner = zarr.storage.DirectoryStore(uri)
    return CountingStore(inner)


def open_dataset(uri: str, storage_options: Optional[Dict[str, Any]] = None,
                 chunks: Any = None) -> Tuple[Any, CountingStore]:
    """Open a store lazily. Returns the dataset and its byte counter."""
    import xarray as xr

    store = open_store(uri, storage_options)
    try:
        # decode_timedelta is pinned because xarray's default changed between versions and
        # the resulting warning is noise in every single call.
        dataset = xr.open_zarr(store, chunks=chunks, decode_timedelta=False,
                               consolidated=None)
    except Exception as exc:
        raise DataSourceError(
            "%r opened as a store but xarray could not read it as Zarr (%s: %s)."
            % (uri, type(exc).__name__, exc), uri=uri)
    return dataset, store


# --------------------------------------------------------------------------- chunk reports

def _var_chunks(dataarray) -> Optional[Tuple[int, ...]]:
    """The *store's* chunk shape, not dask's. `encoding` is the on-disk truth."""
    chunks = dataarray.encoding.get("chunks")
    if chunks is None and dataarray.chunks is not None:
        chunks = tuple(c[0] for c in dataarray.chunks)
    return tuple(int(c) for c in chunks) if chunks else None


def describe_store(dataset, variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Report the store's dimensions and its per-variable chunk structure.

    The first half of the acceptance criterion. Chunk *shape* alone is not actionable, so the
    bytes per chunk are computed too - "one timestep, all levels, whole globe" only becomes
    alarming once it reads "54.0 MB".
    """
    import numpy as np

    names = list(variables) if variables else list(dataset.data_vars)
    report: Dict[str, Any] = {"dimensions": {k: int(v) for k, v in dataset.sizes.items()},
                              "n_data_vars": len(dataset.data_vars), "variables": {}}
    for name in names:
        if name not in dataset.data_vars:
            continue
        var = dataset[name]
        chunks = _var_chunks(var)
        itemsize = int(var.dtype.itemsize)
        entry: Dict[str, Any] = {
            "dims": list(var.dims),
            "shape": [int(s) for s in var.shape],
            "dtype": str(var.dtype),
            "chunks": list(chunks) if chunks else None,
        }
        if chunks:
            chunk_elements = int(np.prod(chunks))
            entry["chunk_bytes"] = chunk_elements * itemsize
            entry["chunk_megabytes"] = round(chunk_elements * itemsize / 1e6, 3)
            entry["n_chunks"] = int(np.prod(
                [int(np.ceil(s / c)) for s, c in zip(var.shape, chunks)]))
        report["variables"][name] = entry
    return report


def assess_access_pattern(dataset, spec: "CropSpec") -> Dict[str, Any]:
    """Predict, from chunk metadata alone, how much the network will over-deliver.

    Amplification is ``bytes the store must hand over / bytes the crop actually wants``,
    computed per variable from the chunk grid the selection touches. Nothing is transferred
    to work this out, which is the point: the warning has to arrive *before* the download,
    not as an explanation afterwards.

    **Byte counts here are uncompressed**, so for a compressed store they are an upper bound
    on what crosses the wire; the *ratio* is unaffected, because both halves are counted the
    same way. Stated because the measured and predicted numbers otherwise appear to disagree:
    on random float32 test data the wire bytes came in 6% below the prediction, which is the
    compressor, not an error.

    Validated against the real WeatherBench 2 store (`VERIFICATION.md`, slice 11).
    """
    import numpy as np

    lat_name = "latitude" if "latitude" in dataset.sizes else "lat"
    lon_name = "longitude" if "longitude" in dataset.sizes else "lon"

    selection, selected_positions = _selection_plan(
        dataset, spec, lat_name, lon_name)
    per_variable: Dict[str, Any] = {}
    worst = 0.0
    total_fetch = 0
    total_want = 0

    for name in spec.variables:
        if name not in dataset.data_vars:
            continue
        var = dataset[name]
        chunks = _var_chunks(var)
        if not chunks:
            continue
        itemsize = int(var.dtype.itemsize)
        want_elements = 1
        fetch_elements = 1
        detail = {}
        for dim, size, chunk in zip(var.dims, var.shape, chunks):
            wanted = selection.get(dim, int(size))
            positions = selected_positions.get(str(dim))
            if positions is None:
                touched = int(np.ceil(size / chunk))
            else:
                # Exact chunk-grid arithmetic. Counting only ``wanted`` cannot distinguish a
                # selection inside one chunk from the same-width selection straddling two.
                touched = len({int(position) // int(chunk) for position in positions})
            touched = max(1, min(touched, int(np.ceil(size / chunk))))
            want_elements *= max(1, wanted)
            fetch_elements *= touched * chunk
            detail[dim] = {"wanted": int(wanted), "chunk": int(chunk),
                           "chunks_touched": int(touched)}
        want_bytes = want_elements * itemsize
        fetch_bytes = fetch_elements * itemsize
        amp = fetch_bytes / want_bytes if want_bytes else float("inf")
        per_variable[name] = {
            "bytes_wanted": int(want_bytes),
            "bytes_fetched": int(fetch_bytes),
            "amplification": round(amp, 2),
            "by_dimension": detail,
        }
        worst = max(worst, amp)
        total_fetch += fetch_bytes
        total_want += want_bytes

    hostile = worst >= HOSTILE_AMPLIFICATION
    result: Dict[str, Any] = {
        "selection": selection,
        "per_variable": per_variable,
        "bytes_wanted": int(total_want),
        "bytes_fetched_estimate": int(total_fetch),
        "megabytes_fetched_estimate": round(total_fetch / 1e6, 2),
        "byte_basis": ("uncompressed; an upper bound on wire bytes for a compressed store. "
                       "The amplification ratio is unaffected."),
        "amplification": round(worst, 2),
        "chunk_hostile": bool(hostile),
        "threshold": HOSTILE_AMPLIFICATION,
        "warning": None,
        "advice": [],
    }
    if hostile:
        result["warning"] = (
            "CHUNK-HOSTILE ACCESS: this store must hand over %.2f GB to deliver %.2f GB of "
            "requested data - an amplification of %.1fx. The chunks are larger than the crop "
            "in the dimensions being subset, so each one is fetched whole and mostly "
            "discarded. Materialise once into the local rechunked cache and read that; "
            "repeating this query against the remote store will repeat the whole transfer."
            % (total_fetch / 1e9, total_want / 1e9, worst))
        offenders = sorted(
            ((v["amplification"], name) for name, v in per_variable.items()), reverse=True)
        for _amp, name in offenders[:3]:
            detail = per_variable[name]["by_dimension"]
            fixed = [d for d, v in detail.items()
                     if v["chunks_touched"] * v["chunk"] > v["wanted"] * 2]
            if fixed:
                result["advice"].append(
                    "%s: the %s dimension(s) cannot be narrowed by selection because the "
                    "chunk spans more than the request - subsetting them saves nothing over "
                    "the network, only memory." % (name, ", ".join(sorted(fixed))))
    return result


def _selection_plan(dataset, spec: "CropSpec", lat_name: str,
                    lon_name: str) -> Tuple[Dict[str, int], Dict[str, Tuple[int, ...]]]:
    """Selected sizes and source positions, computed from coordinate indexes alone.

    Apply the same xarray indexers as :func:`select`, but only to the one-dimensional coordinate
    arrays.  The old implementation ran ``select(dataset, spec)`` and therefore asked dask to
    build a sliced task graph for every selected data variable merely to read ``subset.sizes``.
    GLORYS made that graph large enough to raise ``MemoryError`` (D67), even though the answer is
    entirely coordinate and chunk arithmetic.

    Keeping xarray in this path is important: a partial date such as ``2020-01-01`` selects the
    whole day, exactly as the materialising path does.  Reimplementing date comparisons with
    numpy previously under-counted a twelve-step fixture by three frames.
    """
    _validate_requested_variables(dataset, spec)
    indexers = _selection_indexers(dataset, spec, lat_name, lon_name)

    selected_dims = {
        str(dim)
        for name in spec.variables
        for dim in dataset[name].dims
    }
    sizes = {dim: int(dataset.sizes[dim]) for dim in selected_dims}
    positions: Dict[str, Tuple[int, ...]] = {}
    for dim, indexer in indexers.items():
        if dim not in selected_dims:
            continue
        coordinate = dataset[dim]
        selected = coordinate.sel({dim: indexer})
        sizes[dim] = int(selected.size)
        source_index = dataset.get_index(dim)
        locations = source_index.get_indexer(selected.values)
        if bool((locations < 0).any()):
            raise DataSourceError(
                "the selected %s coordinates could not be mapped back to the store's chunk "
                "grid" % dim, dimension=dim)
        positions[dim] = tuple(int(value) for value in locations.tolist())

    _validate_nonempty_selection(dataset, spec, sizes, lat_name, lon_name)
    return sizes, positions


def _validate_requested_variables(dataset, spec: "CropSpec") -> None:
    missing = [v for v in spec.variables if v not in dataset.data_vars]
    if missing:
        raise DataSourceError(
            "the store does not contain %s. Available (first 15): %s"
            % (", ".join(missing), ", ".join(sorted(dataset.data_vars)[:15])),
            missing=missing)


def _selection_indexers(dataset, spec: "CropSpec", lat_name: str,
                        lon_name: str) -> Dict[str, Any]:
    """Build the shared coordinate indexers without touching any data variable."""
    import numpy as np

    lat = np.asarray(dataset[lat_name].values)
    descending = bool(lat.size > 1 and lat[0] > lat[-1])
    lat_slice = (slice(spec.lat_max, spec.lat_min) if descending
                 else slice(spec.lat_min, spec.lat_max))

    lon = np.asarray(dataset[lon_name].values)
    lon_min, lon_max = spec.lon_min, spec.lon_max
    if float(lon.min()) >= 0.0 and lon_min < 0.0:
        raise InvalidParameterError(
            "lon_min", lon_min,
            "a longitude in [0, 360) for this store, whose longitude axis runs %.2f..%.2f. "
            "A region spanning the prime meridian must be requested as two crops and "
            "joined, because one monotonic slice cannot express the wrap."
            % (float(lon.min()), float(lon.max())))

    indexers: Dict[str, Any] = {
        lat_name: lat_slice,
        lon_name: slice(lon_min, lon_max),
    }
    vertical = spec.vertical_dim
    if spec.levels and vertical in dataset.coords:
        available_values = np.asarray(dataset[vertical].values)
        available = available_values.tolist()
        unknown = [lev for lev in spec.levels
                   if not bool(np.any(available_values == lev))]
        if unknown:
            raise InvalidParameterError(
                "levels", list(spec.levels),
                "%s values present in the store; %s not among %s"
                % (vertical, unknown, sorted(available)))
        indexers[vertical] = list(spec.levels)
    if "time" in dataset.coords:
        indexers["time"] = slice(spec.time_start, spec.time_end)
    return indexers


def _validate_nonempty_selection(dataset, spec: "CropSpec", sizes: Dict[str, int],
                                 lat_name: str, lon_name: str) -> None:
    if "time" in dataset.coords and int(sizes.get("time", 0)) == 0:
        times = dataset["time"].values
        raise InvalidParameterError(
            "time_start/time_end", (spec.time_start, spec.time_end),
            "a window inside the store's coverage, %s to %s"
            % (str(times[0])[:16], str(times[-1])[:16]))
    for dim, name in ((lat_name, "latitude"), (lon_name, "longitude")):
        if int(sizes.get(dim, 0)) == 0:
            raise InvalidParameterError(
                name, (spec.lat_min, spec.lat_max) if dim == lat_name
                else (spec.lon_min, spec.lon_max),
                "a range overlapping the store's %s axis (%.3f to %.3f)"
                % (name, float(dataset[dim].values.min()),
                   float(dataset[dim].values.max())))


# --------------------------------------------------------------------------- selection

def select(dataset, spec: "CropSpec"):
    """Apply a crop spec to an open dataset, tolerating either latitude ordering.

    ERA5 stores latitude descending (90 -> -90) in some layouts and ascending in others.
    `sel(slice(...))` on a descending axis with an ascending slice silently returns **zero
    rows**, which downstream looks like a region with no data rather than a reversed axis -
    so the direction is detected rather than assumed.
    """
    lat_name = "latitude" if "latitude" in dataset.sizes else "lat"
    lon_name = "longitude" if "longitude" in dataset.sizes else "lon"

    _validate_requested_variables(dataset, spec)
    indexers = _selection_indexers(dataset, spec, lat_name, lon_name)
    subset = dataset[list(spec.variables)]
    subset = subset.sel(indexers)
    _validate_nonempty_selection(
        dataset, spec, {str(k): int(v) for k, v in subset.sizes.items()},
        lat_name, lon_name)
    return subset


# --------------------------------------------------------------------------- cache

def cache_path(spec: "CropSpec", cache_dir: Optional[str] = None) -> str:
    """Content-addressed location for a materialised crop."""
    return os.path.join(_cache_dir(cache_dir), "%s.zarr" % spec.content_key())


def manifest_path(spec: "CropSpec", cache_dir: Optional[str] = None) -> str:
    return os.path.join(_cache_dir(cache_dir), "%s.json" % spec.content_key())


def is_cached(spec: "CropSpec", cache_dir: Optional[str] = None) -> bool:
    return os.path.exists(manifest_path(spec, cache_dir)) and os.path.isdir(
        cache_path(spec, cache_dir))


def _content_hash(dataset) -> str:
    """Hash of the materialised values, for provenance (standard E5).

    Hashes each variable's bytes in name order after loading, so it is independent of chunk
    layout: the same crop rechunked differently must produce the same hash, or the hash would
    be describing the cache rather than the data.
    """
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(dataset.data_vars):
        values = np.ascontiguousarray(dataset[name].values)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()[:32]


def streaming_content_hash(dataset, time_block: int = 32) -> str:
    """Return the canonical data hash without materialising a whole long record.

    This emits the same logical byte stream as :func:`_content_hash`: variables in name
    order, followed by each name, full shape and contiguous C-order values.  Splitting an
    array along its leading ``time`` dimension does not change that byte stream, so neither
    the read block nor the Zarr chunk layout can change the identity.
    """
    import numpy as np

    if isinstance(time_block, bool) or int(time_block) != time_block or time_block < 1:
        raise InvalidParameterError("time_block", time_block, "a positive integer")
    digest = hashlib.sha256()
    for name in sorted(dataset.data_vars):
        variable = dataset[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(int(value) for value in variable.shape)).encode("utf-8"))
        if variable.dims and variable.dims[0] == "time":
            for start in range(0, int(variable.sizes["time"]), int(time_block)):
                stop = min(start + int(time_block), int(variable.sizes["time"]))
                values = np.ascontiguousarray(variable.isel(time=slice(start, stop)).values)
                digest.update(values.tobytes())
        else:
            digest.update(np.ascontiguousarray(variable.values).tobytes())
    return digest.hexdigest()[:32]


def _raise_transform_crop_refusal(geometry: Mapping[str, Any],
                                  plan: Optional[Mapping[str, Any]] = None) -> None:
    """Translate a planner verdict into the established client-safe R13 refusal."""
    current = tuple(int(v) for v in geometry["current_shape"])
    required = tuple(int(v) for v in geometry["recommended_minimum"]["shape"])
    analysis = geometry["analysis"]
    suggestion = ((plan or {}).get("suggestions") or {}).get("recommended") or {}
    bounds = suggestion.get("bounds")
    remedy = (
        "%s level %d leaves this crop below the statistically recommended R13 interior. "
        "The absolute minimum %s is technical computability only; it is not licensed for "
        "meaningful spatial statistics. "
        % (analysis["transform_family"].upper(), int(analysis["levels"]),
           tuple(geometry["absolute_minimum"]["shape"])))
    if suggestion.get("feasible") and bounds:
        remedy += (
            "Expand on the store's native coordinate grid to lat %.12g..%.12g and lon "
            "%.12g..%.12g, which yields %s; inspect that exact suggestion to review its "
            "revised transfer and storage cost before materialising."
            % (bounds["lat_min"], bounds["lat_max"], bounds["lon_min"], bounds["lon_max"],
               tuple(suggestion["actual_shape"])))
    else:
        remedy += str(suggestion.get("reason") or
                      "Use a larger crop, a shallower preregistered analysis, or another "
                      "registered transform; do not weaken the grid after seeing the cost.")
    raise FieldTooSmallError(
        "%s level-%d statistically recommended cross-scale analysis"
        % (analysis["transform_family"].upper(), int(analysis["levels"])),
        current, required, remedy=remedy,
        geometry=dict(geometry), acquisition_plan_sha256=(plan or {}).get("plan_sha256"))


def materialise(spec: "CropSpec", cache_dir: Optional[str] = None,
                 storage_options: Optional[Dict[str, Any]] = None,
                 time_chunk: Optional[int] = None,
                 check_size: bool = True,
                 force: bool = False,
                 analysis: Any = None) -> Dict[str, Any]:
    """Fetch a crop once and write it to a local, time-contiguous Zarr cache.

    Parameters
    ----------
    time_chunk
        Timesteps per chunk in the cache. Default is the whole window: the downstream access
        pattern is "all frames at this location", so one chunk per variable makes a cross-scale
        read a single seek. This is the *inverse* of the remote layout, and reversing it is the
        entire justification for keeping a second copy.
    check_size
        Enforce the R13 floor against ``analysis`` when supplied, otherwise the legacy
        conservative 14-tap check against ``spec.n_levels_analysis``. Off only for deliberately
        small test crops.

    Returns a manifest: the spec, the resolved shape, bytes transferred, the content hash and
    the chunk report. The manifest is what goes into lineage.
    """
    cache_dir = _cache_dir(cache_dir)
    key = spec.content_key()
    path = cache_path(spec, cache_dir)
    if is_cached(spec, cache_dir) and not force:
        with open(manifest_path(spec, cache_dir), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if check_size and analysis is not None:
            from src.data_layer.crop_planner import assess_shape
            shape = manifest.get("shape") or {}
            lat_name = "latitude" if "latitude" in shape else "lat"
            lon_name = "longitude" if "longitude" in shape else "lon"
            geometry = assess_shape(int(shape.get(lat_name, 0)),
                                    int(shape.get(lon_name, 0)), analysis)
            if not geometry["meets_recommended_minimum"]:
                _raise_transform_crop_refusal(geometry)
            manifest["requested_analysis_geometry"] = geometry
        manifest["cache_hit"] = True
        # Zero, not "small". The claim being made is that a repeat costs no network at all.
        manifest["bytes_transferred"] = 0
        return manifest

    started = time.time()
    dataset, counter = open_dataset(spec.uri, storage_options, chunks={})
    try:
        structure = describe_store(dataset, spec.variables)
        assessment = assess_access_pattern(dataset, spec)
        acquisition_plan = None
        if check_size and analysis is not None:
            from src.data_layer.crop_planner import plan_acquisition
            acquisition_plan = plan_acquisition(
                dataset, spec, analysis, structure=structure)
            geometry = acquisition_plan["geometry"]
            if not geometry["meets_recommended_minimum"]:
                _raise_transform_crop_refusal(geometry, acquisition_plan)
        else:
            selection = assessment["selection"]
            lat_key = "latitude" if "latitude" in selection else "lat"
            lon_key = "longitude" if "longitude" in selection else "lon"
            geometry = (check_crop_size(
                int(selection[lat_key]), int(selection[lon_key]), spec.n_levels_analysis)
                        if check_size else {"ok": None, "skipped": "check_size=False"})

        # The scientific geometry refusal above uses coordinate/chunk metadata only. Construct
        # the data-variable selection only after it passes, and load values later still.
        subset = select(dataset, spec)
        lat_name = "latitude" if "latitude" in subset.sizes else "lat"
        lon_name = "longitude" if "longitude" in subset.sizes else "lon"

        bytes_before = counter.bytes_read
        loaded = subset.load()
        transferred = counter.bytes_read - bytes_before

        n_time = int(loaded.sizes.get("time", 1))
        chunking = {"time": int(time_chunk or n_time)}
        for dim in (lat_name, lon_name, spec.vertical_dim):
            if dim in loaded.sizes:
                chunking[dim] = int(loaded.sizes[dim])
        rechunked = loaded.chunk(chunking)

        # **The inherited-encoding trap.** `to_zarr` prefers each variable's
        # `encoding["chunks"]`, which xarray copied from the *remote* store - one timestep
        # per chunk. So `.chunk()` set the dask graph, the write ignored it, and the cache
        # came out with the very layout it exists to escape. Nothing errored; the manifest
        # still recorded the requested chunking, and the only visible symptom was a cache
        # read costing 32 chunk fetches instead of 1. The encoding is therefore cleared and
        # the target chunks passed explicitly.
        encoding = {}
        for name in rechunked.data_vars:
            for key_to_drop in ("chunks", "preferred_chunks"):
                rechunked[name].encoding.pop(key_to_drop, None)
            encoding[name] = {"chunks": tuple(
                int(chunking.get(str(dim), rechunked.sizes[dim]))
                for dim in rechunked[name].dims)}

        os.makedirs(cache_dir, exist_ok=True)
        if os.path.isdir(path) and force:
            import shutil
            shutil.rmtree(path)
        rechunked.to_zarr(path, mode="w", consolidated=True, encoding=encoding)

        manifest = {
            "content_key": key,
            "cache_path": path,
            "cache_hit": False,
            "spec": spec.to_provenance(),
            "shape": {k: int(v) for k, v in loaded.sizes.items()},
            "variables": sorted(loaded.data_vars),
            "content_hash": _content_hash(loaded),
            "bytes_transferred": int(transferred),
            "megabytes_transferred": round(transferred / 1e6, 3),
            "chunks_transferred": int(counter.keys_read),
            "elapsed_s": round(time.time() - started, 2),
            "remote_chunk_structure": structure,
            "access_assessment": assessment,
            "geometry": geometry,
            "cache_chunking": chunking,
            "rechunk_rationale": (
                "the remote store is chunked one timestep at a time; the cache is chunked "
                "time-contiguous for this region, so reading all frames at one location is "
                "one read instead of %d" % n_time),
        }
        # Preserve the legacy manifest schema exactly unless the caller explicitly requested
        # transform-aware planning. New science records get the complete immutable plan;
        # historical callers do not acquire a semantically empty null field.
        if acquisition_plan is not None:
            manifest["acquisition_plan"] = acquisition_plan
        with open(manifest_path(spec, cache_dir), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest
    finally:
        dataset.close()


def load_cached(spec: "CropSpec", cache_dir: Optional[str] = None
                ) -> Tuple[Any, Dict[str, Any]]:
    """Open a materialised crop, asserting that no remote bytes are touched.

    The returned manifest carries the counter's own report, so the "zero network traffic on a
    repeat request" claim is a measurement in the record rather than an assurance in a
    docstring.
    """
    cache_dir = _cache_dir(cache_dir)
    if not is_cached(spec, cache_dir):
        raise DataSourceError(
            "crop %s is not in the cache at %r. Call materialise() first; loading is "
            "deliberately not allowed to fall back to a silent remote fetch, because the "
            "cost difference between the two is four orders of magnitude."
            % (spec.content_key(), cache_dir), content_key=spec.content_key())
    dataset, counter = open_dataset(cache_path(spec, cache_dir))
    dataset.load()
    with open(manifest_path(spec, cache_dir), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["local_read"] = counter.report()
    manifest["remote_bytes"] = 0
    return dataset, manifest


def open_cached_lazy(spec: "CropSpec", cache_dir: Optional[str] = None
                     ) -> Tuple[Any, Dict[str, Any]]:
    """Open a materialised crop without loading its arrays into host memory.

    This is the bounded-memory counterpart to :func:`load_cached`.  It has the same strict
    local-only precondition, but leaves value reads lazy so a consumer can stream statistics
    or select individual training windows.  The returned xarray object owns an open local
    store and must be closed by the caller.
    """
    cache_dir = _cache_dir(cache_dir)
    if not is_cached(spec, cache_dir):
        raise DataSourceError(
            "crop %s is not in the cache at %r. Call materialise() first; lazy opening is "
            "deliberately not allowed to fall back to the network."
            % (spec.content_key(), cache_dir), content_key=spec.content_key())
    dataset, counter = open_dataset(cache_path(spec, cache_dir), chunks=None)
    with open(manifest_path(spec, cache_dir), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["local_read"] = counter.report()
    manifest["remote_bytes"] = 0
    manifest["value_access"] = "lazy_local_zarr"
    return dataset, manifest


class CachedFieldReader:
    """Random-access physical frames from one authenticated local crop selection.

    This is the bounded bridge into the Phase 4 streaming analysis.  It opens only an
    existing local Zarr cache, selects one exact variable and pressure level, and reads one
    time index per call.  Machine paths never enter ``source_provenance``.
    """

    def __init__(
        self,
        spec: CropSpec,
        variable: str,
        *,
        level_hpa: Optional[float] = None,
        cache_dir: Optional[str] = None,
        maximum_source_chunk_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        import numpy as np
        import torch

        if isinstance(maximum_source_chunk_bytes, bool) \
                or int(maximum_source_chunk_bytes) != maximum_source_chunk_bytes \
                or maximum_source_chunk_bytes < 1:
            raise InvalidParameterError(
                "maximum_source_chunk_bytes", maximum_source_chunk_bytes,
                "a positive integer byte ceiling")
        self.dataset, manifest = open_cached_lazy(spec, cache_dir)
        try:
            if manifest.get("content_key") != spec.content_key():
                raise DataSourceError("cached manifest content key does not match the crop")
            if variable not in self.dataset.data_vars:
                raise DataSourceError(
                    "variable %r is absent from the cached crop; available variables are %s"
                    % (variable, sorted(str(name) for name in self.dataset.data_vars)))
            array = self.dataset[variable]
            if "level" in array.dims:
                if level_hpa is None:
                    raise DataSourceError(
                        "cached variable %r has a pressure-level axis; select level_hpa exactly"
                        % variable)
                levels = np.asarray(array.level.values, dtype=np.float64)
                if not np.any(levels == float(level_hpa)):
                    raise DataSourceError(
                        "level_hpa %.6g is absent; available levels are %s"
                        % (float(level_hpa), levels.tolist()))
                array = array.sel(level=float(level_hpa))
            elif level_hpa is not None:
                raise DataSourceError(
                    "level_hpa was supplied but cached variable %r has no level axis" % variable)

            lat_name = "latitude" if "latitude" in array.dims else "lat"
            lon_name = "longitude" if "longitude" in array.dims else "lon"
            expected_dims = {"time", lat_name, lon_name}
            if set(array.dims) != expected_dims:
                raise DataSourceError(
                    "cached field selection must have exactly time/latitude/longitude; got %s"
                    % (list(array.dims),))
            self.array = array.transpose("time", lat_name, lon_name)
            self.times = np.asarray(self.array.time.values)
            if self.times.ndim != 1 or self.times.size < 2:
                raise DataSourceError("cached field reader requires at least two time frames")
            nanoseconds = self.times.astype("datetime64[ns]").astype("int64")
            deltas = np.diff(nanoseconds)
            if np.any(deltas <= 0) or not np.all(deltas == deltas[0]):
                raise DataSourceError(
                    "cached time axis must be strictly increasing and exactly regular")
            self.latitude = np.asarray(self.array[lat_name].values, dtype=np.float64)
            self.longitude = np.asarray(self.array[lon_name].values, dtype=np.float64)
            self._coords = {
                "lat": torch.as_tensor(self.latitude, dtype=torch.float64),
                "lon": torch.as_tensor(self.longitude, dtype=torch.float64),
            }
            stored_time_chunk = int((manifest.get("cache_chunking") or {}).get("time", 0))
            if stored_time_chunk < 1:
                raise DataSourceError(
                    "cached manifest does not record a positive time chunk; bounded reads "
                    "cannot be established")
            source_chunk_bytes = int(
                stored_time_chunk * self.latitude.size * self.longitude.size
                * np.dtype(self.array.dtype).itemsize)
            if source_chunk_bytes > int(maximum_source_chunk_bytes):
                raise DataSourceError(
                    "one stored source chunk expands to %d bytes, above the %d-byte ceiling; "
                    "rematerialise with a smaller time_chunk"
                    % (source_chunk_bytes, int(maximum_source_chunk_bytes)))
            coordinate_digest = hashlib.sha256()
            coordinate_digest.update(np.ascontiguousarray(nanoseconds).tobytes())
            coordinate_digest.update(np.ascontiguousarray(self.latitude).tobytes())
            coordinate_digest.update(np.ascontiguousarray(self.longitude).tobytes())
            self.variable = str(variable)
            self.level_hpa = float(level_hpa) if level_hpa is not None else None
            self.units = str(self.array.attrs.get("units", "unknown"))
            source_is_portable = (spec.store in CATALOGUE or "://" in spec.store
                                  or spec.store.startswith("cds:"))
            self.source_provenance = {
                "content_key": manifest.get("content_key"),
                "content_hash": manifest.get("content_hash"),
                "crop_spec": manifest.get("spec"),
                "shape": manifest.get("shape"),
                "variables": manifest.get("variables"),
                "cache_chunking": manifest.get("cache_chunking"),
                "source_route": manifest.get("source_route", "catalogued Zarr cache"),
                "independent_overlap_check": manifest.get(
                    "independent_overlap_check", "NOT DECLARED"),
                "independent_overlap_receipt_sha256": manifest.get(
                    "independent_overlap_receipt_sha256"),
                "independent_overlap_receipt": manifest.get(
                    "independent_overlap_receipt"),
                "acquisition_request_sha256": (
                    ((manifest.get("acquisition") or {}).get("request") or {}).get(
                        "request_sha256")),
                "variable": self.variable,
                "level_hpa": self.level_hpa,
                "units": self.units,
                "coordinate_sha256": coordinate_digest.hexdigest(),
                "cadence_seconds": float(deltas[0] / 1e9),
                "source_chunk_bytes": source_chunk_bytes,
                "maximum_source_chunk_bytes": int(maximum_source_chunk_bytes),
                "network_used": False,
                "machine_paths_included": not source_is_portable,
            }
        except BaseException:
            self.dataset.close()
            raise

    def __len__(self) -> int:
        return int(self.times.size)

    def read_frame(self, index: int):
        import numpy as np
        import torch

        from src.physical_core.field import PhysicalField

        if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < len(self):
            raise InvalidParameterError(
                "index", index, "an integer frame index in 0..%d" % (len(self) - 1))
        values = np.ascontiguousarray(self.array.isel(time=int(index)).values)
        return PhysicalField(
            torch.from_numpy(values),
            coords=self._coords,
            metadata={
                "variable": self.variable,
                "level": self.level_hpa,
                # TG1.5: the unit travels with the number from the one place that
                # knows it. This reader selects an ERA5 pressure level by name, so it
                # is entitled to declare the coordinate; nothing downstream has to
                # infer hPa from an attribute name.
                "level_axis": PRESSURE_HPA,
                "units": self.units,
                "source_content_hash": self.source_provenance["content_hash"],
                "source_coordinate_sha256": self.source_provenance["coordinate_sha256"],
                "frame_index": int(index),
                "is_simulated": False,
            },
            units=self.units,
        )

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "CachedFieldReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def rematerialise_from_provenance(record: Dict[str, Any],
                                  cache_dir: Optional[str] = None,
                                  **kwargs: Any) -> Dict[str, Any]:
    """Reproduce a crop from its lineage record alone (standard E5).

    The last acceptance criterion, and the one that makes the rest worth having: a finding is
    reproducible only if the *exact* input can be rebuilt from what was recorded, by someone
    who was not there.
    """
    spec = CropSpec.from_provenance(record.get("spec", record))
    return materialise(spec, cache_dir=cache_dir, **kwargs)


# --------------------------------------------------------------------------- source seam

#: Dataset id this source answers to. Distinct from the three built-in NetCDF ids because it
#: is a *different kind of thing*: those name a file, this names a regional crop of an archive.
ZARR_DATASET_ID = "era5_zarr_regional"

#: Network access is **opt-in**. A research platform must not reach the internet as a side
#: effect of running a test or a sweep: it makes results depend on connectivity, turns an
#: offline machine into a stream of confusing failures, and lets a mistyped bounding box
#: transfer tens of gigabytes. Set SPECTRALEARTH_ALLOW_NETWORK=1 to enable.
NETWORK_ENV_VAR = "SPECTRALEARTH_ALLOW_NETWORK"


def network_enabled() -> bool:
    return os.getenv(NETWORK_ENV_VAR, "0").strip().lower() in ("1", "true", "yes", "on")


def missing_dependencies() -> List[str]:
    """Which optional packages are absent, checked by import rather than by version pin."""
    missing = []
    for module in ("zarr", "dask", "gcsfs"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def cached_crops(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every materialised crop, read from its manifest. Works with no network at all."""
    cache_dir = _cache_dir(cache_dir)
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(cache_dir, name), "r", encoding="utf-8") as handle:
                out.append(json.load(handle))
        except (OSError, ValueError):
            # A half-written manifest is a cache problem, not a reason to fail the listing.
            continue
    return out


def _register():
    """Register the Zarr source. Separated from module scope so tests can re-register it."""
    from src.data_layer.sources import register_source

    @register_source(
        "era5_zarr",
        description=("Streams regional ERA5 crops from public WeatherBench 2 Zarr on GCS "
                     "into a rechunked local cache. Real observational data."),
        # Between netcdf_local (10) and simulated (900), exactly as builtin_sources.py
        # anticipated: a file the researcher deliberately placed outranks a remote fetch.
        priority=20,
        is_simulated=False,
        kind="zarr",
        # `crop_dataset_ids`, deliberately NOT `dataset_ids`. That key means "ids this
        # source can hand over as a whole dataset", and `list_datasets` promises that every
        # id declared under it appears in the dataset listing with concrete variables, a
        # time range, a bounding box and a resolution. A crop *family* has none of those
        # until a crop is specified - the underlying archive is 64 years of the whole
        # planet - so advertising it there would list a dataset that `/data/slice` cannot
        # serve. Materialised crops, which do have concrete extents, are listed by
        # `GET /api/v1/data/zarr/cached` instead. Caught by T3.5.15's acceptance test, which
        # asserts exactly that invariant.
        capabilities={"crop_dataset_ids": [ZARR_DATASET_ID], "streaming": True,
                      "requires_network": True, "observational": True,
                      "regional_crops": True},
    )
    class ERA5ZarrSource:
        """ERA5 over Zarr, as a registered fallback-chain source."""

        @staticmethod
        def can_serve(dataset_id: str) -> bool:
            # Cheap, as the protocol demands: imports and a directory listing, never a
            # network call. A cached crop makes this source available *offline*, which is
            # the entire point of stage 2.
            if dataset_id != ZARR_DATASET_ID or missing_dependencies():
                return False
            return network_enabled() or bool(cached_crops())

        @staticmethod
        def why_not(dataset_id: str) -> str:
            if dataset_id != ZARR_DATASET_ID:
                return "this source only serves %r" % ZARR_DATASET_ID
            missing = missing_dependencies()
            if missing:
                return ("the Zarr stack is not installed (missing %s); pip install %s"
                        % (", ".join(missing), " ".join(missing)))
            return ("network access is disabled (set %s=1) and no crop has been "
                    "materialised into %s yet, so there is nothing to read offline"
                    % (NETWORK_ENV_VAR, DEFAULT_CACHE_DIR))

        @staticmethod
        def fetch(dataset_id: str, crop: Optional[Dict[str, Any]] = None,
                  cache_dir: Optional[str] = None, **kwargs: Any):
            if crop is None:
                raise DataSourceError(
                    "%r needs a crop specification - a store, variables, a time window, a "
                    "bounding box and levels. Unlike a NetCDF file there is no single "
                    "'the dataset' to return: the archive is 64 years of the whole planet."
                    % ZARR_DATASET_ID, dataset_id=dataset_id)
            spec = CropSpec.from_provenance(crop)
            if is_cached(spec, cache_dir):
                dataset, _manifest = load_cached(spec, cache_dir)
                return dataset
            if not network_enabled():
                raise DataSourceError(
                    "crop %s is not cached and network access is disabled. Set %s=1 to "
                    "materialise it, or point cache_dir at a directory that already "
                    "contains it." % (spec.content_key(), NETWORK_ENV_VAR),
                    content_key=spec.content_key())
            materialise(spec, cache_dir=cache_dir, **kwargs)
            dataset, _manifest = load_cached(spec, cache_dir)
            return dataset

    return ERA5ZarrSource


ERA5ZarrSource = _register()


# --------------------------------------------------------------------------- operator CLI

def _main(argv: Optional[Sequence[str]] = None) -> int:
    """python -m src.data_layer.zarr_source <command>

    ``inspect`` reads only metadata - no data transfer - and is the command to run *before*
    committing to a download. ``materialise`` performs the transfer.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ERA5 Zarr crop tool (T3.5.18)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("catalogue", help="list known stores")
    sub.add_parser("cached", help="list materialised crops")

    for name in ("inspect", "materialise"):
        crop_parser = sub.add_parser(name, help="%s a crop" % name)
        crop_parser.add_argument("--store", default="era5_0p25_6h")
        crop_parser.add_argument("--variables", default="temperature")
        crop_parser.add_argument("--start", required=True)
        crop_parser.add_argument("--end", required=True)
        crop_parser.add_argument("--lat", nargs=2, type=float, required=True,
                                 metavar=("MIN", "MAX"))
        crop_parser.add_argument("--lon", nargs=2, type=float, required=True,
                                 metavar=("MIN", "MAX"))
        crop_parser.add_argument("--levels", default="850,700,500,300")
        crop_parser.add_argument("--analysis-levels", type=int, default=4)
        crop_parser.add_argument("--analysis-transform", default="dtcwt")
        crop_parser.add_argument("--wavelet", default="db2")
        crop_parser.add_argument("--boundary-mode", default="periodic")
        crop_parser.add_argument("--dtcwt-level1", default="near_sym_b")
        crop_parser.add_argument("--dtcwt-qshift", default="qshift_b")
        crop_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
        crop_parser.add_argument("--time-chunk", type=int, default=None)
        crop_parser.add_argument("--no-size-check", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "catalogue":
        print(json.dumps(catalogue_payload(), indent=2))
        return 0
    if args.command == "cached":
        print(json.dumps([{"content_key": c["content_key"], "spec": c["spec"],
                           "shape": c.get("shape")} for c in cached_crops()], indent=2))
        return 0
    if args.command not in ("inspect", "materialise"):
        parser.print_help()
        return 2

    # `crop_for_store`, not `CropSpec`, so the vertical axis comes from the store's own
    # declaration rather than from ERA5's default -- the CLI was building a `level` selection
    # for every store, including one whose axis is `elevation` (D72). Levels are parsed
    # integer-first so that ERA5's `850,700,500,300` stays integral and every pinned content
    # key is byte-identical, while a fractional GLORYS elevation is kept as a float instead of
    # raising `invalid literal for int()`.
    spec = crop_for_store(
        args.store,
        variables=tuple(v.strip() for v in args.variables.split(",") if v.strip()),
        time_start=args.start, time_end=args.end,
        lat_min=args.lat[0], lat_max=args.lat[1],
        lon_min=args.lon[0], lon_max=args.lon[1],
        levels=tuple(_parse_level(v) for v in args.levels.split(",") if v.strip()),
        n_levels_analysis=args.analysis_levels,
    )
    from src.data_layer.crop_planner import TransformSupportRequest, plan_acquisition
    analysis = TransformSupportRequest(
        transform_family=args.analysis_transform, levels=args.analysis_levels,
        wavelet=args.wavelet, boundary_mode=args.boundary_mode,
        dtcwt_level1=args.dtcwt_level1, dtcwt_qshift=args.dtcwt_qshift)
    if args.command == "inspect":
        dataset, _counter = open_dataset(spec.uri, chunks={})
        try:
            structure = describe_store(dataset, spec.variables)
            report = {
                "spec": spec.to_provenance(),
                "structure": structure,
                "assessment": assess_access_pattern(dataset, spec),
                "acquisition_plan": plan_acquisition(
                    dataset, spec, analysis, structure=structure),
            }
        finally:
            dataset.close()
        print(json.dumps(report, indent=2, default=str))
        return 0

    manifest = materialise(spec, cache_dir=args.cache_dir, time_chunk=args.time_chunk,
                           check_size=not args.no_size_check, analysis=analysis)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
