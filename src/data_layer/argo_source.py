"""Bounded access to the official Argo GDAC ERDDAP view (TG12.2d).

The Argo Data Management Team lists the Ifremer ERDDAP service as an access path over the
GDAC collection.  A cheap distinct-profile query is always made first; the level data request
is refused when that observed count exceeds ``ProfileSpec.max_profiles``.  Network access uses
the same explicit opt-in as the Zarr acquisition path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.publication import publish_new_bytes
from src.data_layer.profiles import (PROFILE_SOURCES, ProfileCollection, ProfileSource,
                                     ProfileSpec, canonical_json)


ARGO_ERDDAP_DATASET = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats"
ARGO_DOI = "https://doi.org/10.17882/42182"
NETWORK_ENV_VAR = "SPECTRALEARTH_ALLOW_NETWORK"
DEFAULT_PROFILE_ROOT = Path("data") / "profile_collections"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
GOOD_QC = ("1", "2")

Fetch = Callable[[str, int], bytes]

_HEADER_FIELDS = ("fileNumber", "platform_number", "cycle_number", "direction", "time",
                  "latitude", "longitude")
_VALUE_FIELDS = _HEADER_FIELDS + (
    "pres", "pres_qc", "pres_adjusted", "pres_adjusted_qc",
    "temp", "temp_qc", "temp_adjusted", "temp_adjusted_qc",
    "psal", "psal_qc", "psal_adjusted", "psal_adjusted_qc")


ARGO_SOURCE = ProfileSource(
    name="argo_gdac_erddap", domain="argo_float", access="public",
    description=("Core-Argo profile levels from the official GDAC collection through the "
                 "Ifremer ERDDAP view."),
    licence=("Argo data are freely available; cite the Argo Program, contributing national "
             "programmes, and DOI 10.17882/42182."),
    variables=("temperature", "salinity"),
    defaults={"time_start": "2026-01-01", "time_end": "2026-03-01",
              "lat_min": -46.0, "lat_max": -34.0, "lon_min": 165.0, "lon_max": 180.0,
              "pressure_min_dbar": 0.0, "pressure_max_dbar": 200.0,
              "variables": ["temperature"], "max_profiles": 200})


def register_argo_source() -> ProfileSource:
    """Register the profile producer and its external domain atomically enough for discovery."""
    from extensions import argo_float  # noqa: F401 - onboarding is the import's explicit act
    if ARGO_SOURCE.name not in PROFILE_SOURCES:
        PROFILE_SOURCES.add(
            ARGO_SOURCE.name, ARGO_SOURCE, description=ARGO_SOURCE.description,
            capabilities={"profile_query": True, "preflight": True,
                          "official_gdac_view": True},
            params={"variables": list(ARGO_SOURCE.variables)})
    return PROFILE_SOURCES.get(ARGO_SOURCE.name)


def network_enabled() -> bool:
    return os.getenv(NETWORK_ENV_VAR, "0").strip().lower() in ("1", "true", "yes", "on")


def _fetch(url: str, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    request = Request(url, headers={
        "User-Agent": "SpectralEarth/1.0 scientific-profile-acquisition",
        "Accept": "text/csv",
    })
    with urlopen(request, timeout=120) as response:  # nosec B310 - fixed HTTPS host
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > limit:
            raise InvalidParameterError(
                "Argo response bytes", int(declared),
                "at most %d bytes. Narrow the profile query; the client does not start an "
                "unbounded archive transfer" % limit)
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise InvalidParameterError("Argo response bytes", len(payload),
                                    "at most %d bytes" % limit)
    return payload


def _constraints(spec: ProfileSpec) -> Tuple[str, ...]:
    return (
        "time>=%s" % spec.time_start, "time<=%s" % spec.time_end,
        "latitude>=%s" % format(spec.lat_min, ".17g"),
        "latitude<=%s" % format(spec.lat_max, ".17g"),
        "longitude>=%s" % format(spec.lon_min, ".17g"),
        "longitude<=%s" % format(spec.lon_max, ".17g"),
    )


def query_url(spec: ProfileSpec, *, values: bool) -> str:
    fields = _VALUE_FIELDS if values else _HEADER_FIELDS
    pieces = [",".join(fields)] + [quote(value, safe="=:-._")
                                      for value in _constraints(spec)]
    if not values:
        pieces.append("distinct()")
    return ARGO_ERDDAP_DATASET + ".csv0?" + "&".join(pieces)


def _rows(payload: bytes, fields: Sequence[str]) -> List[Dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidParameterError("Argo response", "non-UTF-8",
                                    "the GDAC ERDDAP CSV encoding") from exc
    rows: List[Dict[str, str]] = []
    for position, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or (len(row) == 1 and row[0].startswith("Error")):
            continue
        if len(row) != len(fields):
            raise InvalidParameterError(
                "Argo response row", {"row": position, "columns": len(row)},
                "%d columns matching the declared ERDDAP projection" % len(fields))
        rows.append(dict(zip(fields, row)))
    return rows


def _profile_id(row: Mapping[str, str]) -> str:
    return "%s:%s:%s:%s" % (row["platform_number"].strip(), row["cycle_number"].strip(),
                             row["direction"].strip() or "?", row["time"].strip())


def inspect_profile_query(spec: ProfileSpec, *, fetch: Fetch = _fetch) -> Dict[str, Any]:
    """Observe candidate profile metadata without transferring level values."""
    if fetch is _fetch and not network_enabled():
        raise InvalidParameterError(
            NETWORK_ENV_VAR, os.getenv(NETWORK_ENV_VAR),
            "1 before live Argo access. Profile metadata and values are remote; network "
            "access is never enabled by a browser action")
    url = query_url(spec, values=False)
    payload = fetch(url, 8 * 1024 * 1024)
    rows = _rows(payload, _HEADER_FIELDS)
    candidates = []
    seen = set()
    for row in rows:
        identifier = _profile_id(row)
        if identifier in seen:
            continue
        seen.add(identifier)
        candidates.append({
            "profile_id": identifier, "platform_id": row["platform_number"].strip(),
            "cycle_number": int(row["cycle_number"]), "direction": row["direction"].strip(),
            "time": row["time"].strip(), "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]), "source_file": row["fileNumber"].strip(),
        })
    candidates.sort(key=lambda value: (value["time"], value["profile_id"]))
    count = len(candidates)
    return {
        "schema": "spectral.profile-query-plan.v1", "request": spec.canonical(),
        "request_sha256": spec.request_sha256(), "source": "Argo GDAC via Ifremer ERDDAP",
        "source_url": ARGO_ERDDAP_DATASET, "source_doi": ARGO_DOI,
        "metadata_query_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_profiles": count, "candidate_platforms": len({
            value["platform_id"] for value in candidates}),
        "within_profile_cap": count <= spec.max_profiles,
        "max_profiles": spec.max_profiles, "estimated_level_response_cap_bytes": MAX_RESPONSE_BYTES,
        "profiles": candidates[:50], "profiles_withheld": max(0, count - 50),
        "claim_boundary": (
            "This is a coordinate/index observation from the official GDAC view. It has not "
            "fetched profile-level measurements and makes no claim about data quality or "
            "scientific viability."),
    }


def _number(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _admitted(raw: str, raw_qc: str, adjusted: str, adjusted_qc: str) -> float:
    """Prefer QC-admitted adjusted data, then QC-admitted raw data; otherwise observed-invalid."""
    adjusted_value = _number(adjusted)
    if adjusted_qc.strip() in GOOD_QC and np.isfinite(adjusted_value):
        return adjusted_value
    raw_value = _number(raw)
    if raw_qc.strip() in GOOD_QC and np.isfinite(raw_value):
        return raw_value
    return float("nan")


def _unix(value: str) -> float:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def acquire_profiles(spec: ProfileSpec, *, fetch: Fetch = _fetch) -> ProfileCollection:
    """Fetch the smallest bounded level response after an exact profile-count preflight."""
    plan = inspect_profile_query(spec, fetch=fetch)
    if plan["candidate_profiles"] == 0:
        raise InvalidParameterError(
            "profile query", spec.canonical(),
            "at least one Argo profile. Expand the region/time range or inspect another span")
    if not plan["within_profile_cap"]:
        raise InvalidParameterError(
            "candidate_profiles", plan["candidate_profiles"],
            "at most max_profiles=%d before profile values are fetched. Narrow the region/time "
            "range or deliberately raise the cap (never above 500)" % spec.max_profiles,
            request_sha256=spec.request_sha256())
    url = query_url(spec, values=True)
    payload = fetch(url, MAX_RESPONSE_BYTES)
    rows = _rows(payload, _VALUE_FIELDS)
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_profile_id(row), []).append(row)
    if len(grouped) > spec.max_profiles:
        raise InvalidParameterError(
            "profiles returned", len(grouped),
            "at most max_profiles=%d. The live archive changed between metadata preflight and "
            "value fetch; no partial result is accepted" % spec.max_profiles)

    records: List[Dict[str, Any]] = []
    rejected_nonmonotonic: List[str] = []
    for identifier, profile_rows in grouped.items():
        first = profile_rows[0]
        pressure: List[float] = []
        variables: Dict[str, List[float]] = {name: [] for name in spec.variables}
        for row in profile_rows:
            p = _admitted(row["pres"], row["pres_qc"], row["pres_adjusted"],
                          row["pres_adjusted_qc"])
            if not np.isfinite(p) or not spec.pressure_min_dbar <= p <= spec.pressure_max_dbar:
                continue
            pressure.append(p)
            if "temperature" in variables:
                variables["temperature"].append(_admitted(
                    row["temp"], row["temp_qc"], row["temp_adjusted"],
                    row["temp_adjusted_qc"]))
            if "salinity" in variables:
                variables["salinity"].append(_admitted(
                    row["psal"], row["psal_qc"], row["psal_adjusted"],
                    row["psal_adjusted_qc"]))
        p_array = np.asarray(pressure, dtype=np.float64)
        if p_array.size == 0:
            continue
        if np.any(np.diff(p_array) <= 0):
            rejected_nonmonotonic.append(identifier)
            continue
        records.append({
            "profile_id": identifier, "platform_id": first["platform_number"].strip(),
            "cycle_number": int(first["cycle_number"]), "time": _unix(first["time"]),
            "latitude": float(first["latitude"]), "longitude": float(first["longitude"]),
            "pressure": p_array, "variables": variables,
            "source_file": first["fileNumber"].strip(),
        })
    records.sort(key=lambda value: (value["time"], value["profile_id"]))
    if not records:
        raise InvalidParameterError(
            "profile values", {"returned_profiles": len(grouped),
                               "nonmonotonic_rejected": rejected_nonmonotonic},
            "at least one profile with QC-admitted, strictly increasing pressure inside the "
            "requested range")
    return ProfileCollection(
        spec=spec, profile_ids=[value["profile_id"] for value in records],
        platform_ids=[value["platform_id"] for value in records],
        cycle_numbers=[value["cycle_number"] for value in records],
        times_seconds=[value["time"] for value in records],
        latitude=[value["latitude"] for value in records],
        longitude=[value["longitude"] for value in records],
        pressure_dbar=[value["pressure"] for value in records],
        measures={name: [np.asarray(value["variables"][name], dtype=np.float64)
                         for value in records] for name in spec.variables},
        source_files=[value["source_file"] for value in records],
        index_sha256=plan["metadata_query_sha256"],
        response_sha256=hashlib.sha256(payload).hexdigest(),
        qc_policy={
            "name": "argo-adjusted-preferred-good-probably-good/v1",
            "accepted_qc_flags": list(GOOD_QC),
            "selection": "adjusted when admitted, otherwise raw when admitted",
            "failed_value": "NaN means observed but not QC-admitted",
            "nonmonotonic_profiles_rejected": rejected_nonmonotonic,
        })


def persist_collection(collection: ProfileCollection,
                       root: Path = DEFAULT_PROFILE_ROOT) -> Dict[str, Any]:
    """Publish canonical collection bytes under their observed-content digest."""
    digest = collection.collection_sha256()
    path = Path(root) / (digest + ".json")
    payload = canonical_json(collection.canonical()) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise InvalidParameterError("profile collection", str(path),
                                        "bytes matching its content-addressed digest")
        primitive = "already-present-identical"
    else:
        primitive = publish_new_bytes(path, payload, "profile collection")
    return {"path": str(path), "collection_sha256": digest,
            "bytes": len(payload), "publication": primitive}


def load_collection(collection_sha256: str,
                    root: Path = DEFAULT_PROFILE_ROOT) -> ProfileCollection:
    digest = str(collection_sha256).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise InvalidParameterError("collection_sha256", collection_sha256,
                                    "a 64-character SHA-256")
    path = Path(root) / (digest + ".json")
    if not path.exists():
        raise InvalidParameterError("collection_sha256", digest,
                                    "a locally materialised profile collection")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidParameterError("profile collection", str(path),
                                    "canonical JSON") from exc
    collection = ProfileCollection.from_canonical(value)
    if collection.collection_sha256() != digest \
            or raw != canonical_json(collection.canonical()) + b"\n":
        raise InvalidParameterError("profile collection", str(path),
                                    "canonical bytes matching the addressed digest")
    return collection


__all__ = ["ARGO_DOI", "ARGO_ERDDAP_DATASET", "ARGO_SOURCE", "DEFAULT_PROFILE_ROOT",
           "NETWORK_ENV_VAR", "acquire_profiles", "inspect_profile_query", "load_collection",
           "network_enabled", "persist_collection", "query_url", "register_argo_source"]
