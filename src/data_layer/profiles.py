"""Immutable irregular-profile acquisition contracts (TG12.2b).

``CropSpec`` is intentionally not reused here.  A crop describes a rectangular array;
an ocean-profile query returns a scatter whose pressure vector may have a different length
for every observation.  Naming that distinction prevents a later adapter from padding the
scatter into an array and silently changing what its axes mean.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.core.registry import Registry


PROFILE_SPEC_SCHEMA = "spectral.profile-spec.v1"
PROFILE_COLLECTION_SCHEMA = "spectral.profile-collection.v1"
PROFILE_VARIABLES = ("temperature", "salinity")


@dataclass(frozen=True)
class ProfileSource:
    """Discoverable producer of :class:`ProfileCollection` objects."""

    name: str
    domain: str
    access: str
    description: str
    licence: str
    variables: Sequence[str]
    defaults: Mapping[str, Any]

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "shape": "profile_query",
                "access": self.access, "description": self.description,
                "licence": self.licence, "variables": list(self.variables),
                "defaults": dict(self.defaults)}


PROFILE_SOURCES: Registry[ProfileSource] = Registry("profile source")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _json_vector(value: np.ndarray) -> list[Any]:
    return [None if not np.isfinite(item) else float(item) for item in value]


def _utc(value: str, field: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidParameterError(field, value, "an ISO-8601 date or timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return parsed.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ProfileSpec:
    """A bounded, machine-independent request for an irregular profile scatter."""

    time_start: str
    time_end: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    pressure_min_dbar: float
    pressure_max_dbar: float
    variables: Sequence[str] = ("temperature", "salinity")
    max_profiles: int = 200
    source: str = "argo_gdac_erddap"

    def __post_init__(self) -> None:
        start, end = _utc(self.time_start, "time_start"), _utc(self.time_end, "time_end")
        if start >= end:
            raise InvalidParameterError("time range", [start, end],
                                        "a start strictly before the end")
        object.__setattr__(self, "time_start", start)
        object.__setattr__(self, "time_end", end)
        for name in ("lat_min", "lat_max", "lon_min", "lon_max",
                     "pressure_min_dbar", "pressure_max_dbar"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise InvalidParameterError(name, value, "a finite coordinate")
            object.__setattr__(self, name, value)
        if not -90 <= self.lat_min < self.lat_max <= 90:
            raise InvalidParameterError("latitude bounds", [self.lat_min, self.lat_max],
                                        "-90 <= minimum < maximum <= 90")
        if not -180 <= self.lon_min < self.lon_max <= 180:
            raise InvalidParameterError(
                "longitude bounds", [self.lon_min, self.lon_max],
                "-180 <= minimum < maximum <= 180. Dateline-crossing queries must be split "
                "into two explicit requests so their identity is unambiguous")
        if not 0 <= self.pressure_min_dbar < self.pressure_max_dbar:
            raise InvalidParameterError(
                "pressure bounds", [self.pressure_min_dbar, self.pressure_max_dbar],
                "0 <= minimum < maximum in dbar")
        variables = tuple(str(value).strip().lower() for value in self.variables)
        if not variables or len(set(variables)) != len(variables) \
                or any(value not in PROFILE_VARIABLES for value in variables):
            raise InvalidParameterError("variables", list(self.variables),
                                        "distinct names drawn from %s" % (PROFILE_VARIABLES,))
        object.__setattr__(self, "variables", variables)
        if isinstance(self.max_profiles, bool) or not 2 <= int(self.max_profiles) <= 500:
            raise InvalidParameterError(
                "max_profiles", self.max_profiles,
                "an integer from 2 through 500. The preflight refuses before a query can "
                "turn a browser action into an unbounded archive transfer")
        object.__setattr__(self, "max_profiles", int(self.max_profiles))
        if self.source != "argo_gdac_erddap":
            raise InvalidParameterError("source", self.source, "'argo_gdac_erddap'")

    def canonical(self) -> Dict[str, Any]:
        return {
            "schema": PROFILE_SPEC_SCHEMA, "source": self.source,
            "time_start": self.time_start, "time_end": self.time_end,
            "latitude": [self.lat_min, self.lat_max],
            "longitude": [self.lon_min, self.lon_max],
            "pressure_dbar": [self.pressure_min_dbar, self.pressure_max_dbar],
            "variables": list(self.variables), "max_profiles": self.max_profiles,
        }

    def request_sha256(self) -> str:
        return stable_sha256(self.canonical())

    def content_key(self) -> str:
        return self.request_sha256()[:16]


def _readonly_vector(value: Any, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    vector = np.array(value, dtype=dtype, copy=True)
    if vector.ndim != 1:
        raise ShapeMismatchError(name, vector.shape, "a one-dimensional profile vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ProfileCollection:
    """One immutable scatter of profiles, with no rectangular-grid fiction.

    Position and time are one value per profile. Pressure and every measured variable are
    tuples of one-dimensional vectors, one vector per profile. Missing/failed measurements
    remain ``NaN`` where the profile was observed; absence between profiles is introduced only
    by a later declared reduction.
    """

    spec: ProfileSpec
    profile_ids: Sequence[str]
    platform_ids: Sequence[str]
    cycle_numbers: Sequence[int]
    times_seconds: Sequence[float]
    latitude: Sequence[float]
    longitude: Sequence[float]
    pressure_dbar: Sequence[np.ndarray]
    measures: Mapping[str, Sequence[np.ndarray]]
    source_files: Sequence[str]
    index_sha256: str
    response_sha256: str
    qc_policy: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.profile_ids)
        if n < 1:
            raise InvalidParameterError("profile_ids", [],
                                        "at least one observed profile")
        if len(set(str(value) for value in self.profile_ids)) != n:
            raise InvalidParameterError("profile_ids", list(self.profile_ids),
                                        "distinct stable profile identities")
        for name in ("platform_ids", "cycle_numbers", "times_seconds", "latitude",
                     "longitude", "pressure_dbar", "source_files"):
            if len(getattr(self, name)) != n:
                raise ShapeMismatchError(name, (len(getattr(self, name)),),
                                         "one entry per profile", (n,))
        times = _readonly_vector(self.times_seconds, "times_seconds")
        lat = _readonly_vector(self.latitude, "latitude")
        lon = _readonly_vector(self.longitude, "longitude")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(lat)) \
                or not np.all(np.isfinite(lon)):
            raise InvalidParameterError("profile coordinates", "non-finite",
                                        "finite time, latitude and longitude per profile")
        pressures = tuple(_readonly_vector(value, "pressure_dbar[%d]" % index)
                          for index, value in enumerate(self.pressure_dbar))
        for index, pressure in enumerate(pressures):
            if pressure.size < 1 or not np.all(np.isfinite(pressure)) \
                    or np.any(np.diff(pressure) <= 0):
                raise InvalidParameterError(
                    "pressure_dbar[%d]" % index, pressure.tolist(),
                    "a non-empty, finite, strictly increasing pressure vector. Profiles are "
                    "not silently sorted or deduplicated after acquisition")
        measures: Dict[str, Tuple[np.ndarray, ...]] = {}
        if tuple(sorted(self.measures)) != tuple(sorted(self.spec.variables)):
            raise InvalidParameterError("measures", sorted(self.measures),
                                        "exactly the variables requested by ProfileSpec")
        for variable, profiles in self.measures.items():
            if len(profiles) != n:
                raise ShapeMismatchError(variable, (len(profiles),),
                                         "one vector per profile", (n,))
            vectors = tuple(_readonly_vector(value, "%s[%d]" % (variable, index))
                            for index, value in enumerate(profiles))
            for index, vector in enumerate(vectors):
                if vector.shape != pressures[index].shape:
                    raise ShapeMismatchError(
                        "%s[%d]" % (variable, index), vector.shape,
                        "the profile's pressure vector", pressures[index].shape)
            measures[str(variable)] = vectors
        for field_name in ("index_sha256", "response_sha256"):
            digest = str(getattr(self, field_name)).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise InvalidParameterError(field_name, digest, "a 64-character SHA-256")
        object.__setattr__(self, "profile_ids", tuple(str(value) for value in self.profile_ids))
        object.__setattr__(self, "platform_ids", tuple(str(value) for value in self.platform_ids))
        object.__setattr__(self, "cycle_numbers", tuple(int(value) for value in self.cycle_numbers))
        object.__setattr__(self, "times_seconds", times)
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "pressure_dbar", pressures)
        object.__setattr__(self, "measures", measures)
        object.__setattr__(self, "source_files", tuple(str(value) for value in self.source_files))
        object.__setattr__(self, "qc_policy", dict(self.qc_policy))

    @property
    def n_profiles(self) -> int:
        return len(self.profile_ids)

    def canonical(self) -> Dict[str, Any]:
        return {
            "schema": PROFILE_COLLECTION_SCHEMA,
            "spec": self.spec.canonical(), "request_sha256": self.spec.request_sha256(),
            "profile_ids": list(self.profile_ids), "platform_ids": list(self.platform_ids),
            "cycle_numbers": list(self.cycle_numbers),
            "times_seconds": self.times_seconds.tolist(),
            "latitude": self.latitude.tolist(), "longitude": self.longitude.tolist(),
            "pressure_dbar": [_json_vector(value) for value in self.pressure_dbar],
            "measures": {name: [_json_vector(value) for value in profiles]
                         for name, profiles in sorted(self.measures.items())},
            "source_files": list(self.source_files),
            "index_sha256": self.index_sha256, "response_sha256": self.response_sha256,
            "qc_policy": dict(self.qc_policy),
        }

    def collection_sha256(self) -> str:
        return stable_sha256(self.canonical())

    def describe(self) -> Dict[str, Any]:
        counts = {name: int(sum(np.isfinite(value).sum() for value in profiles))
                  for name, profiles in self.measures.items()}
        return {
            "schema": PROFILE_COLLECTION_SCHEMA,
            "collection_sha256": self.collection_sha256(),
            "request_sha256": self.spec.request_sha256(), "n_profiles": self.n_profiles,
            "n_platforms": len(set(self.platform_ids)), "variables": list(self.measures),
            "valid_value_count": counts,
            "time_range_seconds": [float(self.times_seconds.min()),
                                   float(self.times_seconds.max())],
            "latitude_range": [float(self.latitude.min()), float(self.latitude.max())],
            "longitude_range": [float(self.longitude.min()), float(self.longitude.max())],
            "pressure_range_dbar": [float(min(v[0] for v in self.pressure_dbar)),
                                    float(max(v[-1] for v in self.pressure_dbar))],
            "index_sha256": self.index_sha256, "response_sha256": self.response_sha256,
            "qc_policy": dict(self.qc_policy),
        }

    @classmethod
    def from_canonical(cls, value: Mapping[str, Any]) -> "ProfileCollection":
        spec_value = value["spec"]
        spec = ProfileSpec(
            time_start=spec_value["time_start"], time_end=spec_value["time_end"],
            lat_min=spec_value["latitude"][0], lat_max=spec_value["latitude"][1],
            lon_min=spec_value["longitude"][0], lon_max=spec_value["longitude"][1],
            pressure_min_dbar=spec_value["pressure_dbar"][0],
            pressure_max_dbar=spec_value["pressure_dbar"][1],
            variables=spec_value["variables"], max_profiles=spec_value["max_profiles"],
            source=spec_value["source"])
        return cls(
            spec=spec, profile_ids=value["profile_ids"], platform_ids=value["platform_ids"],
            cycle_numbers=value["cycle_numbers"], times_seconds=value["times_seconds"],
            latitude=value["latitude"], longitude=value["longitude"],
            pressure_dbar=[np.asarray(v, dtype=np.float64) for v in value["pressure_dbar"]],
            measures={name: [np.asarray([np.nan if item is None else item for item in v],
                                       dtype=np.float64) for v in profiles]
                      for name, profiles in value["measures"].items()},
            source_files=value["source_files"], index_sha256=value["index_sha256"],
            response_sha256=value["response_sha256"], qc_policy=value.get("qc_policy", {}))


__all__ = ["PROFILE_COLLECTION_SCHEMA", "PROFILE_SOURCES", "PROFILE_SPEC_SCHEMA",
           "PROFILE_VARIABLES", "ProfileCollection", "ProfileSource", "ProfileSpec",
           "canonical_json", "stable_sha256"]
