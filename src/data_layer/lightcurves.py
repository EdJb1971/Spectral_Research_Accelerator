"""Immutable per-target photometry contracts (TG13.1).

Light curves are neither grid crops nor profile scatters. The acquisition unit is a set of
archive products for one target and an explicitly bounded sector family. Archive bytes and
every admitted sample enter the content identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.publication import publish_new_bytes
from src.core.registry import Registry

LIGHTCURVE_SCHEMA = "spectral.lightcurve-collection.v1"
DEFAULT_LIGHTCURVE_ROOT = Path("data/lightcurve_collections")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _tic_id(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("TIC"):
        text = text[3:].strip()
    if not text.isdigit() or int(text) <= 0:
        raise InvalidParameterError("target_id", value,
                                    "a positive TIC identifier, with or without the TIC prefix")
    return str(int(text))


@dataclass(frozen=True)
class LightCurveSpec:
    """One bounded, reproducible TESS product request."""

    target_id: str
    sectors: Sequence[int]
    flux_column: str = "PDCSAP_FLUX"
    quality_policy: str = "quality_zero"
    max_products: int = 4
    max_download_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _tic_id(self.target_id))
        if isinstance(self.sectors, (str, bytes)):
            raise InvalidParameterError("sectors", self.sectors,
                                        "one to eight distinct positive sector numbers")
        try:
            sectors = tuple(int(value) for value in self.sectors)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError("sectors", self.sectors,
                                        "one to eight distinct positive sector numbers") from exc
        if not sectors or len(sectors) > 8 or any(value <= 0 for value in sectors) \
                or len(set(sectors)) != len(sectors):
            raise InvalidParameterError("sectors", list(sectors),
                                        "one to eight distinct positive sector numbers")
        object.__setattr__(self, "sectors", tuple(sorted(sectors)))
        flux = str(self.flux_column).strip().upper()
        if flux not in ("PDCSAP_FLUX", "SAP_FLUX"):
            raise InvalidParameterError("flux_column", self.flux_column,
                                        "PDCSAP_FLUX or SAP_FLUX")
        object.__setattr__(self, "flux_column", flux)
        if self.quality_policy not in ("quality_zero", "retain_all"):
            raise InvalidParameterError("quality_policy", self.quality_policy,
                                        "quality_zero or retain_all")
        if isinstance(self.max_products, bool) or not 1 <= int(self.max_products) <= 16:
            raise InvalidParameterError("max_products", self.max_products,
                                        "an integer from 1 through 16")
        if isinstance(self.max_download_bytes, bool) \
                or not 1 <= int(self.max_download_bytes) <= 512 * 1024 * 1024:
            raise InvalidParameterError("max_download_bytes", self.max_download_bytes,
                                        "an integer from 1 through 536870912")

    def canonical(self) -> Dict[str, Any]:
        return {"source": "mast_tess_spoc", "target_id": self.target_id,
                "sectors": list(self.sectors), "flux_column": self.flux_column,
                "quality_policy": self.quality_policy,
                "max_products": int(self.max_products),
                "max_download_bytes": int(self.max_download_bytes)}

    def request_sha256(self) -> str:
        return stable_sha256(self.canonical())


@dataclass(frozen=True)
class LightCurveProduct:
    observation_id: str
    data_uri: str
    filename: str
    sector: int
    size_bytes: int
    provenance_name: str = "SPOC"

    def __post_init__(self) -> None:
        if not str(self.observation_id).strip() or not str(self.data_uri).startswith("mast:"):
            raise InvalidParameterError("light-curve product", self.data_uri,
                                        "a named observation and a MAST data URI")
        if int(self.sector) <= 0 or int(self.size_bytes) <= 0:
            raise InvalidParameterError("light-curve product", self.describe(),
                                        "a positive sector and declared byte size")

    def describe(self) -> Dict[str, Any]:
        return {"observation_id": self.observation_id, "data_uri": self.data_uri,
                "filename": self.filename, "sector": int(self.sector),
                "size_bytes": int(self.size_bytes), "provenance_name": self.provenance_name}


@dataclass(frozen=True)
class LightCurveCollection:
    """Validated TESS samples with exact archive lineage."""

    spec: LightCurveSpec
    products: Sequence[LightCurveProduct]
    times_bjd_tdb: Sequence[float]
    flux: Sequence[float]
    flux_error: Sequence[float]
    quality: Sequence[int]
    sectors: Sequence[int]
    source_sha256: Sequence[str]
    target_ra_deg: float
    target_dec_deg: float
    #: Cadences the archive emitted with no timestamp, dropped before this collection
    #: was built. Counted rather than discarded silently: a sample with no time cannot
    #: be placed on a clock, and how many there were is a property of the record (D95).
    unclocked_samples_dropped: int = 0

    def __post_init__(self) -> None:
        arrays = {"times_bjd_tdb": np.asarray(self.times_bjd_tdb, dtype=np.float64),
                  "flux": np.asarray(self.flux, dtype=np.float64),
                  "flux_error": np.asarray(self.flux_error, dtype=np.float64),
                  "quality": np.asarray(self.quality), "sectors": np.asarray(self.sectors)}
        lengths = {name: int(value.size) for name, value in arrays.items()}
        if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
            raise InvalidParameterError("LightCurveCollection sample lengths", lengths,
                                        "five non-empty arrays of equal length")
        if any(value.ndim != 1 for value in arrays.values()):
            raise InvalidParameterError(
                "LightCurveCollection sample shapes",
                {name: value.shape for name, value in arrays.items()},
                "one-dimensional sample arrays")
        times = arrays["times_bjd_tdb"]
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise InvalidParameterError("times_bjd_tdb", "non-finite or non-increasing",
                                        "a finite strictly increasing BJD_TDB clock")
        if arrays["quality"].dtype.kind not in "iu" or arrays["sectors"].dtype.kind not in "iu":
            raise InvalidParameterError("quality/sectors", "non-integer values",
                                        "integer archive flags and sector numbers")
        if np.any(arrays["sectors"] <= 0) or not set(map(int, arrays["sectors"])) \
                <= set(self.spec.sectors):
            raise InvalidParameterError("sample sectors",
                                        sorted(set(map(int, arrays["sectors"]))),
                                        "only sectors declared by the request")
        if len(self.products) != len(self.source_sha256) or not self.products:
            raise InvalidParameterError("source_sha256", len(self.source_sha256),
                                        "one SHA-256 for every acquired product")
        if any(len(str(value)) != 64 or any(ch not in "0123456789abcdef" for ch in str(value))
               for value in self.source_sha256):
            raise InvalidParameterError("source_sha256", list(self.source_sha256),
                                        "lowercase SHA-256 digests")
        if not math.isfinite(float(self.target_ra_deg)) \
                or not -90 <= float(self.target_dec_deg) <= 90:
            raise InvalidParameterError("target coordinates",
                                        [self.target_ra_deg, self.target_dec_deg],
                                        "finite ICRS degrees with declination in [-90, 90]")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "products", tuple(self.products))
        object.__setattr__(self, "source_sha256", tuple(self.source_sha256))

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.times_bjd_tdb).size)

    def canonical(self) -> Dict[str, Any]:
        def finite_or_none(value: float) -> Optional[float]:
            return float(value) if math.isfinite(float(value)) else None
        return {"schema": LIGHTCURVE_SCHEMA, "spec": self.spec.canonical(),
                "products": [value.describe() for value in self.products],
                "samples": {
                    "time_bjd_tdb": [float(value) for value in self.times_bjd_tdb],
                    "flux": [finite_or_none(value) for value in self.flux],
                    "flux_error": [finite_or_none(value) for value in self.flux_error],
                    "quality": [int(value) for value in self.quality],
                    "sector": [int(value) for value in self.sectors]},
                "source_sha256": list(self.source_sha256),
                "unclocked_samples_dropped": int(self.unclocked_samples_dropped),
                "target": {"tic_id": self.spec.target_id,
                           "ra_deg": float(self.target_ra_deg),
                           "dec_deg": float(self.target_dec_deg), "frame": "ICRS"}}

    def collection_sha256(self) -> str:
        return stable_sha256(self.canonical())

    def describe(self) -> Dict[str, Any]:
        quality = np.asarray(self.quality)
        return {"schema": LIGHTCURVE_SCHEMA, "request_sha256": self.spec.request_sha256(),
                "collection_sha256": self.collection_sha256(), "n_samples": self.n_samples,
                "n_products": len(self.products),
                "sectors": sorted(set(map(int, self.sectors))),
                "quality_admitted": int(np.sum(quality == 0)),
                "quality_flagged": int(np.sum(quality != 0)),
                "unclocked_samples_dropped": int(self.unclocked_samples_dropped),
                "target": self.canonical()["target"], "time_scale": "BJD_TDB",
                "flux_column": self.spec.flux_column}

    @classmethod
    def from_canonical(cls, value: Mapping[str, Any]) -> "LightCurveCollection":
        if value.get("schema") != LIGHTCURVE_SCHEMA:
            raise InvalidParameterError("light-curve schema", value.get("schema"),
                                        LIGHTCURVE_SCHEMA)
        spec = LightCurveSpec(**{key: item for key, item in value["spec"].items()
                                 if key != "source"})
        samples, target = value["samples"], value["target"]
        return cls(spec=spec,
                   products=[LightCurveProduct(**item) for item in value["products"]],
                   times_bjd_tdb=samples["time_bjd_tdb"],
                   flux=[float("nan") if item is None else item for item in samples["flux"]],
                   flux_error=[float("nan") if item is None else item
                               for item in samples["flux_error"]],
                   quality=samples["quality"], sectors=samples["sector"],
                   source_sha256=value["source_sha256"], target_ra_deg=target["ra_deg"],
                   target_dec_deg=target["dec_deg"],
                   unclocked_samples_dropped=int(value.get("unclocked_samples_dropped", 0)))


class LightCurveSource:
    name = ""
    domain = ""
    access = "anonymous"

    def inspect(self, spec: LightCurveSpec) -> Mapping[str, Any]:
        raise NotImplementedError

    def fetch(self, spec: LightCurveSpec) -> LightCurveCollection:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "access": self.access}


LIGHTCURVE_SOURCES: Registry[LightCurveSource] = Registry("lightcurve source")


def persist_collection(collection: LightCurveCollection,
                       root: Path = DEFAULT_LIGHTCURVE_ROOT) -> Dict[str, Any]:
    digest = collection.collection_sha256()
    path = Path(root) / (digest + ".json")
    payload = canonical_json(collection.canonical()) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise InvalidParameterError("light-curve collection", str(path),
                                        "canonical bytes matching its content digest")
        publication = "already-present-identical"
    else:
        publication = publish_new_bytes(path, payload, "light-curve collection")
    return {"path": str(path), "collection_sha256": digest,
            "bytes": len(payload), "publication": publication}


def load_collection(collection_sha256: str,
                    root: Path = DEFAULT_LIGHTCURVE_ROOT) -> LightCurveCollection:
    digest = str(collection_sha256).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise InvalidParameterError("collection_sha256", collection_sha256,
                                    "a 64-character SHA-256")
    path = Path(root) / (digest + ".json")
    if not path.exists():
        raise InvalidParameterError("collection_sha256", digest,
                                    "a locally materialised light-curve collection")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidParameterError("light-curve collection", str(path),
                                    "canonical JSON") from exc
    collection = LightCurveCollection.from_canonical(value)
    if collection.collection_sha256() != digest \
            or raw != canonical_json(collection.canonical()) + b"\n":
        raise InvalidParameterError("light-curve collection", str(path),
                                    "canonical bytes matching the addressed digest")
    return collection


__all__ = ["DEFAULT_LIGHTCURVE_ROOT", "LIGHTCURVE_SCHEMA", "LIGHTCURVE_SOURCES",
           "LightCurveCollection", "LightCurveProduct", "LightCurveSource", "LightCurveSpec",
           "canonical_json", "load_collection", "persist_collection", "stable_sha256"]
