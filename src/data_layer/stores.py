"""Gridded stores as a registry rather than a literal (roadmap TG10.1, standard E1).

**What was wrong with the dict.** `zarr_source.CATALOGUE` was four ERA5 stores in a
module-level dictionary whose value shape was ERA5's own: `resolution_deg`, `cadence_hours`,
`levels`. Standard E1 exists to forbid exactly that. A fifth store — an ocean product on a
`depth` axis, a store behind a free account — could not be added without editing `src/`, and
nothing in the shape had anywhere to record *which domain the store belongs to* or *what it
costs to crop*. Both omissions are load-bearing rather than cosmetic:

*   **A store belongs to a domain.** Rule R17 says a domain that violates nothing is not a
    second domain, and a catalogue that lists stores without saying whose they are invites
    exactly the opposite reading — three entries looking like three domains. Declaring the
    domain here means the acquisition surface can ask *what can I acquire for this domain*
    rather than hard-coding which stores exist (standard E2).
*   **Chunk facts are measured, not assumed.** Defect D43 is what registering a store nobody
    had measured looks like: the 0.703125 degree store reads as the reasonable middle option
    and is not laptop-feasible for a long regional record. Every entry therefore carries how
    its figures were obtained and, where a live inspection produced them, the date it ran.
    `ChunkFacts` refuses a live-inspection claim with no date attached, because an undated
    measurement of an archive that rechunks is a description of nothing.

**What this module deliberately does not do.** It does not open a store, reach the network,
or verify that any recorded figure is still true. Registering a store is a declaration; TG10.3
turns probing into a recorded act and makes a probe a precondition of registration. Until then
a note saying "not measured" is an honest entry and an undated "live inspection" is not.

`zarr_source.CATALOGUE` survives as a **read-only mapping view** over this registry, so the
provenance, overlap and reporting code that only ever asked "is this store catalogued, and
what is its URI" keeps working unchanged. The view is deliberately not mutable: a write to it
would bypass every check below, and a registry whose contents depend on who wrote to it last
is defect D35 rebuilt by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple

from src.core.domain import DOMAIN_DECLARATIONS, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.registry import Registry

#: How a store is reached. Declared per store rather than discovered, so a deployment can be
#: told what it will need *before* a request fails with a credential error.
ACCESS_REQUIREMENTS: Dict[str, str] = {
    "anonymous": "readable with no credentials; the bucket is public",
    "credentials": ("requires an account. Credentials are supplied per request from the "
                    "environment, never stored in a spec, a manifest, a log or a repr"),
    "local": "a path on this machine; nothing crosses the network",
}

#: How a chunk figure was obtained. The distinction is the point: two of these are
#: observations and one is an admission.
MEASUREMENT_METHODS: Dict[str, str] = {
    "live inspection": "opened against the real archive and recorded on a stated date",
    "store metadata": "read from the store's own chunk metadata without transferring data",
    "not measured": "nobody has looked; the store's cost for a regional crop is unknown",
}

#: Vertical axis names this programme has seen. Not a closed set — a store declares its own —
#: but listed so an author reaches for an existing name before inventing a synonym.
KNOWN_VERTICAL_DIMENSIONS: Dict[str, str] = {
    "level": "pressure levels, in hPa, as ERA5 publishes them",
    "depth": "depth below the surface, in metres, as ocean reanalyses publish it",
}


@dataclass(frozen=True)
class ChunkFacts:
    """What one chunk of this store costs, and where that figure came from.

    Separated from the store itself because the store's identity is stable and its chunking
    is not: WeatherBench 2 could rechunk tomorrow, and when it does, the thing that should
    change is this record and its date, not the entry's existence.
    """

    megabytes_per_chunk: Optional[float]
    method: str
    measured_on: str = ""
    shape: Optional[Tuple[int, ...]] = None
    dims: Optional[Tuple[str, ...]] = None
    #: Bytes moved divided by bytes wanted, for a stated regional crop. The single most
    #: useful number for a researcher, and the one nothing else in the entry implies.
    regional_amplification: Optional[float] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.method not in MEASUREMENT_METHODS:
            raise InvalidParameterError(
                "method", self.method, "one of %s" % sorted(MEASUREMENT_METHODS))
        if self.method == "not measured":
            if self.megabytes_per_chunk is not None:
                raise InvalidParameterError(
                    "megabytes_per_chunk", self.megabytes_per_chunk,
                    "None when the method is 'not measured'; a figure nobody measured is "
                    "an assumption wearing a measurement's clothes")
        elif self.megabytes_per_chunk is None or self.megabytes_per_chunk <= 0:
            raise InvalidParameterError(
                "megabytes_per_chunk", self.megabytes_per_chunk,
                "a positive size in megabytes for method %r" % self.method)
        if self.method == "live inspection" and not self.measured_on.strip():
            raise InvalidParameterError(
                "measured_on", self.measured_on,
                "an ISO date; an undated live inspection of an archive that may rechunk "
                "cannot be checked against the archive later")
        if self.shape is not None and self.dims is not None:
            if len(self.shape) != len(self.dims):
                raise InvalidParameterError(
                    "shape/dims", (self.shape, self.dims),
                    "one dimension name per chunk extent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "megabytes_per_chunk": self.megabytes_per_chunk,
            "method": self.method,
            "method_means": MEASUREMENT_METHODS[self.method],
            "measured_on": self.measured_on,
            "shape": list(self.shape) if self.shape is not None else None,
            "dims": list(self.dims) if self.dims is not None else None,
            "regional_amplification": self.regional_amplification,
            "note": self.note,
        }


@dataclass(frozen=True)
class GriddedStore:
    """One acquirable gridded store: what it is, whose domain it is, and what it costs.

    `vertical_dim` has **no default**, for the reason `AxisSpec.role` has none (standard
    E14): the vertical axis of a store is declared by whoever registers it, never inferred
    from the fact that most stores so far happened to call it `level`. A store with no
    vertical axis declares `None` and says so.
    """

    name: str
    uri: str
    domain: str
    access: str
    vertical_dim: Optional[str]
    grid: Tuple[int, int]
    resolution_deg: float
    cadence_hours: float
    levels: int
    note: str
    chunks: ChunkFacts
    variables_note: str = ""
    extra: Dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise InvalidParameterError("name", self.name, "a non-empty store name")
        if not str(self.uri).strip():
            raise InvalidParameterError("uri", self.uri, "a non-empty URI or local path")
        if self.access != "local" and "://" not in self.uri:
            # A scheme is required for anything that crosses the network, because a bare
            # string there is a name nobody can resolve. A `local` store is exempt: a path
            # on this machine legitimately has no scheme, and demanding `file://` would
            # rule out the fixture stores the offline tests are built on.
            raise InvalidParameterError(
                "uri", self.uri,
                "a URI with a scheme, e.g. gs://bucket/path.zarr, for access %r"
                % self.access)
        if self.access not in ACCESS_REQUIREMENTS:
            raise InvalidParameterError(
                "access", self.access, "one of %s" % sorted(ACCESS_REQUIREMENTS))
        if self.vertical_dim is not None and not str(self.vertical_dim).strip():
            raise InvalidParameterError(
                "vertical_dim", self.vertical_dim,
                "a dimension name such as %s, or None for a store with no vertical axis"
                % sorted(KNOWN_VERTICAL_DIMENSIONS))
        if len(self.grid) != 2 or any(int(n) <= 0 for n in self.grid):
            raise InvalidParameterError(
                "grid", self.grid, "a (rows, columns) pair of positive integers")
        if self.levels < 0:
            raise InvalidParameterError("levels", self.levels, "a level count >= 0")
        if self.vertical_dim is None and self.levels:
            raise InvalidParameterError(
                "levels", self.levels,
                "0 when vertical_dim is None; a store cannot carry levels on an axis it "
                "declares it does not have")
        if not self.note.strip():
            raise InvalidParameterError(
                "note", self.note,
                "a note saying what this store is good and bad FOR. The 0.25 and 1.5 degree "
                "ERA5 stores differ by more than resolution, and a catalogue that does not "
                "say so sends a researcher to the wrong one")

    def declaration(self) -> DomainDeclaration:
        """The domain this store belongs to, resolved at call time.

        Resolved late rather than held as an object, so a store entry never pins a stale copy
        of a declaration that was re-onboarded after registration (TG8.1 allows exactly that).
        """
        return DOMAIN_DECLARATIONS.get(self.domain)

    def to_dict(self) -> Dict[str, Any]:
        """The catalogue's public shape.

        The five keys the pre-TG10.1 dictionary carried — `uri`, `resolution_deg`,
        `cadence_hours`, `grid`, `levels`, `note` — are kept verbatim so the existing
        provenance, reporting and frontend readers are unaffected by the change.
        """
        payload: Dict[str, Any] = {
            "uri": self.uri,
            "resolution_deg": self.resolution_deg,
            "cadence_hours": self.cadence_hours,
            "grid": list(self.grid),
            "levels": self.levels,
            "note": self.note,
            "domain": self.domain,
            "access": self.access,
            "access_means": ACCESS_REQUIREMENTS[self.access],
            "vertical_dim": self.vertical_dim,
            "chunks": self.chunks.to_dict(),
        }
        if self.variables_note:
            payload["variables_note"] = self.variables_note
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


#: Registered gridded stores (standard E1). A fifth store is a new file, not an edit here.
GRIDDED_STORES: Registry[GriddedStore] = Registry("gridded store")


def _ensure_domains() -> None:
    """Make the built-in domains present before a store claims one.

    Imported inside the function on purpose. `src.core.builtin_domains` pulls in the
    onboarding contract and the glossaries, and importing that chain at module scope would
    make whether a store registers depend on which module a process happened to import first
    — defect D35, in a registry that decides what a researcher may acquire.
    """
    from src.core.builtin_domains import register_builtin_domains

    register_builtin_domains()


def register_store(store: GriddedStore, *, replace: bool = False,
                   tags: Optional[List[str]] = None) -> GriddedStore:
    """Register one store, refusing a domain nothing has declared.

    The domain check is the reason this is a function rather than a bare `Registry.add`. A
    store naming a domain that was never onboarded would appear in the catalogue, be
    selectable in the acquisition surface, and then fail at the point where its refusals were
    needed — which is the latest possible moment for the failure and the worst.
    """
    _ensure_domains()
    if store.domain not in DOMAIN_DECLARATIONS:
        # Raises UnknownNameError, which lists the valid names and offers a "did you mean".
        DOMAIN_DECLARATIONS.get(store.domain)
    return GRIDDED_STORES.add(
        store.name, store, description=store.note.split(".")[0].strip(),
        capabilities={
            "domain": store.domain,
            "access": store.access,
            "vertical_dim": store.vertical_dim,
            "requires_network": store.access != "local",
            "requires_credentials": store.access == "credentials",
            "chunk_measured": store.chunks.method != "not measured",
            "acquisition_shape": "grid_crop",
        },
        tags=list(tags or []) + [store.domain, store.access],
        replace=replace)


def store_for(name: str) -> GriddedStore:
    """One registered store, or `UnknownNameError` listing the ones there are."""
    return GRIDDED_STORES.get(name)


def uri_for(name: str) -> Optional[str]:
    """The store's URI, or `None` if the name is not catalogued.

    Returns `None` rather than raising because the caller — `CropSpec.uri` — treats an
    uncatalogued name as a raw URI to pass through, which is what keeps an unlisted store
    usable.
    """
    if name in GRIDDED_STORES:
        return GRIDDED_STORES.get(name).uri
    return None


def stores_for_domain(domain: str) -> List[GriddedStore]:
    """Every store declared as belonging to one domain (standard E2)."""
    return [entry.value for entry in GRIDDED_STORES.entries()
            if entry.value.domain == domain]


def catalogue_payload() -> Dict[str, Dict[str, Any]]:
    """The whole registry in the catalogue's public shape, name-ordered."""
    return {entry.name: entry.value.to_dict() for entry in GRIDDED_STORES.entries()}


def domains_with_stores() -> List[str]:
    return sorted({entry.value.domain for entry in GRIDDED_STORES.entries()})


# --------------------------------------------------------------------------- built-ins

#: The four public WeatherBench 2 ERA5 stores, verified reachable anonymously on 2026-08-20
#: by listing `gs://weatherbench2/datasets/era5` (24 stores). Only the ones this platform has
#: a use for are listed; `describe_store` works on any Zarr URI, listed or not.
#:
#: The figures below are transcribed from the notes the pre-TG10.1 catalogue carried, with
#: their provenance now stated per figure rather than buried in prose. Where a chunk shape was
#: never recorded it is `None` rather than reconstructed, because a plausible shape written
#: from a megabyte total is a guess that would read as a measurement.
BUILTIN_STORES: Tuple[GriddedStore, ...] = (
    GriddedStore(
        name="era5_0p25_6h",
        uri=("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-wb13-6h-1440x721.zarr"),
        domain="reanalysis",
        access="anonymous",
        vertical_dim="level",
        grid=(721, 1440),
        resolution_deg=0.25,
        cadence_hours=6,
        levels=13,
        note=("Full-resolution ERA5, 13 pressure levels. Chunked one timestep x all "
              "levels x whole globe (54.0 MB/chunk), so regional crops are severely "
              "chunk-hostile - measured 51.1x amplification. Use for the R13 spatial "
              "floor with a short window, not for long records."),
        chunks=ChunkFacts(
            megabytes_per_chunk=54.0,
            method="live inspection",
            measured_on="2026-08-20",
            shape=(1, 13, 721, 1440),
            dims=("time", "level", "latitude", "longitude"),
            regional_amplification=51.1,
            note=("VERIFICATION.md slice 11: 2 timesteps of a 256x256 four-level crop moved "
                  "108 MB to deliver 2.11 MB, within 0.1% of what `assess_access_pattern` "
                  "predicts from chunk metadata alone."),
        ),
    ),
    GriddedStore(
        name="era5_0p25_1h_full37",
        uri=("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"),
        domain="reanalysis",
        access="anonymous",
        vertical_dim="level",
        grid=(721, 1440),
        resolution_deg=0.25,
        cadence_hours=1,
        levels=37,
        note=("Hourly, 37 levels, 153.6 MB/chunk. The most chunk-hostile store in the "
              "catalogue for regional work; 561,264 timesteps."),
        chunks=ChunkFacts(
            megabytes_per_chunk=153.6,
            method="store metadata",
            note=("Chunk shape was not recorded when the figure was taken, so it is left "
                  "unstated rather than reconstructed from the total."),
        ),
    ),
    GriddedStore(
        name="era5_1p5_6h",
        uri=("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
        domain="reanalysis",
        access="anonymous",
        vertical_dim="level",
        grid=(121, 240),
        resolution_deg=1.5,
        cadence_hours=6,
        levels=13,
        note=("Coarse but chunked 8 timesteps deep (12.1 MB/chunk), so long records are "
              "cheap. Global grid is 121x240, which is BELOW the R13 256x256 floor - "
              "usable for whole-globe work, not for a regional cross-scale crop."),
        chunks=ChunkFacts(
            megabytes_per_chunk=12.1,
            method="store metadata",
            note="Eight timesteps deep, which is why long records are affordable here.",
        ),
    ),
    GriddedStore(
        name="era5_0p7_6h",
        uri=("gs://weatherbench2/datasets/era5/"
             "1959-2022-6h-512x256_equiangular_conservative.zarr"),
        domain="reanalysis",
        access="anonymous",
        vertical_dim="level",
        grid=(256, 512),
        resolution_deg=0.703125,
        cadence_hours=6,
        levels=13,
        note=("Time chunks are eight frames deep, but each still spans every level and "
              "the full 512x256 globe. Live inspection on 2026-08-21 estimated 29.88 GB "
              "for a three-year, one-variable 255x255 crop (26.2x amplification). This "
              "is not a laptop-feasible long-record regional source for T4C.6 (D43)."),
        chunks=ChunkFacts(
            megabytes_per_chunk=8.0,
            method="live inspection",
            measured_on="2026-08-21",
            regional_amplification=26.2,
            note=("The store that reads as the reasonable middle option and is not one. "
                  "Defect D43 is open against this figure: 29.88 GB for a three-year "
                  "one-variable 255x255 crop is not laptop-feasible."),
        ),
    ),
)


def register_builtin_stores() -> Tuple[str, ...]:
    """Register the built-in stores, once, and return the names.

    Eager and idempotent for the reason `register_builtin_domains` is: a catalogue whose
    contents depend on which handler happened to run first is defect D35, and a store that
    appears only after a researcher visits the right tab is a store whose access requirement
    and chunk cost can be missed.
    """
    for store in BUILTIN_STORES:
        if store.name in GRIDDED_STORES:
            continue
        register_store(store)
    return tuple(store.name for store in BUILTIN_STORES)


class CatalogueView(Mapping):
    """A read-only mapping over `GRIDDED_STORES`, in the pre-TG10.1 dictionary's shape.

    Read-only is the whole point. Every consumer of the old `CATALOGUE` only ever asked
    whether a store was catalogued and what its URI was; none of them wrote. Keeping it
    unwritable means a store still cannot enter the catalogue without passing
    `register_store`, so the domain check and the chunk-facts check cannot be sidestepped by
    assigning a bare dictionary — which is precisely what a test was doing before TG10.1.
    """

    def __getitem__(self, key: str) -> Dict[str, Any]:
        # KeyError rather than the registry's UnknownNameError: `Mapping.get` and every
        # `dict(...)` conversion in the standard library catch KeyError and nothing else, and
        # `CATALOGUE.get(store)` on an uncatalogued name is a supported question with the
        # answer "no" - the path a raw URI takes.
        if not isinstance(key, str) or key not in GRIDDED_STORES:
            raise KeyError(key)
        return GRIDDED_STORES.get(key).to_dict()

    def __iter__(self):
        return iter(GRIDDED_STORES.names())

    def __len__(self) -> int:
        return len(GRIDDED_STORES)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in GRIDDED_STORES

    def __repr__(self) -> str:
        return "CatalogueView(%s)" % GRIDDED_STORES.names()


__all__ = ["ACCESS_REQUIREMENTS", "BUILTIN_STORES", "GRIDDED_STORES",
           "KNOWN_VERTICAL_DIMENSIONS", "MEASUREMENT_METHODS", "CatalogueView",
           "ChunkFacts", "GriddedStore", "catalogue_payload", "domains_with_stores",
           "register_builtin_stores", "register_store", "store_for", "stores_for_domain",
           "uri_for"]
