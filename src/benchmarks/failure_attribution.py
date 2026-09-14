"""T4E.37: attribute fixed T4E.36 join failures before changing the instrument.

The diagnostic inspects the raw local-maximum candidates that already precede the calibrated
extractor threshold.  It does not introduce an alternative extractor or turn those candidates
into features.  A real evaluation is guarded by a digest-bound human adoption.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

from src.benchmarks.catalogue_join import great_circle_km, join_row
from src.benchmarks.wrapped_join import (
    _field,
    _load_record_receipt,
    _observation,
    _positions,
)
from src.core.adoption import adoption_state
from src.core.errors import DataSourceError
from src.core.extraction import _neighbour_maximum, extract
from src.core.publication import publish_new_bytes
from src.data_layer.zarr_source import streaming_content_sha256


FAILURE_ATTRIBUTION_SCHEMA = "t4e37-failure-attribution/v1"
FAILURE_DECLARATION_SCHEMA = "t4e37-failure-attribution-declaration/v1"
DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e37-failure-attribution-declaration.json")
DEFAULT_OUTPUT = Path("measurements/t4e37_failure_attribution.json")
SURROGATES = 99
SEED = 1234


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_failure_declaration(
    path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "T4E.37 declaration is unreadable: %s" % exc, path=str(source)) from exc
    if (body.get("schema") != FAILURE_DECLARATION_SCHEMA
            or body.get("task") != "T4E.37"):
        raise DataSourceError("unsupported T4E.37 failure-attribution declaration")
    population = body.get("frozen_population", {})
    diagnostic = body.get("frozen_diagnostic", {})
    classification = body.get("classification_declared_before_measurement", {})
    if (population.get("expected_rows") != 13
            or len(population.get("identities", [])) != 13):
        raise DataSourceError("T4E.37 declaration must freeze exactly 13 failed rows")
    if ("3x3" not in str(diagnostic.get("sampled_candidate", ""))
            or "n_surrogates=99" not in str(diagnostic.get("extractor_reproduction", ""))
            or "seed=1234" not in str(diagnostic.get("extractor_reproduction", ""))):
        raise DataSourceError("T4E.37 declaration lacks the frozen candidate or extractor rule")
    if ("at least 7" not in str(classification.get("decision", ""))
            or "Neither outcome is PASS" not in str(
                classification.get("no_success_gate", ""))):
        raise DataSourceError("T4E.37 declaration lacks its majority decision boundary")
    return body


def require_current_adoption(
    declaration_path: Union[str, os.PathLike[str]],
) -> Dict[str, Any]:
    source = Path(declaration_path)
    state = adoption_state(source.parent, source.name)
    if not state.get("adopted"):
        raise DataSourceError(
            "T4E.37 evaluation is refused: the declaration has not been adopted by a "
            "maintainer", declaration=source.name, adoption_file=state["adoption_file"])
    if not state.get("signature_still_reaches_the_declaration"):
        raise DataSourceError(
            "T4E.37 evaluation is refused: the adoption does not bind the current declaration",
            declaration=source.name, adoption_file=state["adoption_file"])
    return state


def sampled_local_maxima(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> List[Dict[str, Any]]:
    """Return the extractor's native-grid candidates before thresholding or localisation."""
    array = np.asarray(values, dtype=np.float64)
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    if (array.ndim != 2 or array.shape != (len(lats), len(lons))
            or not np.isfinite(array).all()
            or not np.isfinite(lats).all() or not np.isfinite(lons).all()):
        raise DataSourceError(
            "T4E.37 requires one finite 2D field matching finite latitude and longitude axes")
    mask = array >= _neighbour_maximum(array, (False, False))
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    rows, columns = np.nonzero(mask)
    return [
        {
            "grid_index": [int(row), int(column)],
            "lat": float(lats[row]),
            "lon": float(lons[column]),
            "field_value": float(array[row, column]),
        }
        for row, column in zip(rows, columns)
    ]


def classify_failure(inside_sampled_maxima: int) -> str:
    if not isinstance(inside_sampled_maxima, (int, np.integer)) or inside_sampled_maxima < 0:
        raise DataSourceError("inside sampled-maximum count must be a non-negative integer")
    return ("EXTRACTOR_FILTERING_CANDIDATE" if inside_sampled_maxima
            else "FIELD_REFERENCE_SEPARATION_CANDIDATE")


def majority_decision(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "EXTRACTOR_FILTERING_CANDIDATE": 0,
        "FIELD_REFERENCE_SEPARATION_CANDIDATE": 0,
    }
    for row in rows:
        label = row.get("classification")
        if label not in counts:
            raise DataSourceError("T4E.37 row has no declared binary classification")
        counts[str(label)] += 1
    if len(rows) != 13:
        raise DataSourceError("T4E.37 majority decision requires exactly 13 rows")
    dominant = ("EXTRACTOR_FILTERING_DOMINANT"
                if counts["EXTRACTOR_FILTERING_CANDIDATE"] >= 7
                else "FIELD_REFERENCE_SEPARATION_DOMINANT")
    return {"outcome": dominant, "counts": counts, "majority_needed": 7,
            "is_acceptance_verdict": False}


def _verified_json(path: Path, expected_file_sha256: str, label: str) -> Dict[str, Any]:
    try:
        if _file_sha256(path) != expected_file_sha256:
            raise DataSourceError("%s bytes do not match the declaration" % label, path=str(path))
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("%s is unreadable: %s" % (label, exc), path=str(path)) from exc


def _same_distances(observed: List[float], expected: List[float]) -> bool:
    return len(observed) == len(expected) and bool(np.allclose(
        observed, expected, rtol=0.0, atol=1e-9))


def evaluate_failure_attribution(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    *,
    output_path: Union[str, os.PathLike[str]] = DEFAULT_OUTPUT,
    hash_block_frames: int = 32,
) -> Dict[str, Any]:
    """Evaluate only after adoption, reproducing T4E.36 before attributing its failures."""
    declaration_path = Path(declaration_path)
    declaration = load_failure_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    evidence = declaration["source_evidence"]
    measurement_path = Path(evidence["wrapped_join"])
    measurement = _verified_json(
        measurement_path, evidence["wrapped_join_file_sha256"], "T4E.36 measurement")
    unsigned = dict(measurement)
    supplied = unsigned.pop("receipt_sha256", None)
    if (supplied != _sha256(unsigned)
            or supplied != evidence["wrapped_join_receipt_sha256"]
            or measurement.get("VERDICT") != "FAIL"):
        raise DataSourceError("T4E.36 measurement receipt or FAIL state is invalid")

    record_receipt_path = Path(evidence["wrapped_record_receipt"])
    if _file_sha256(record_receipt_path) != evidence["wrapped_record_receipt_file_sha256"]:
        raise DataSourceError("wrapped-record receipt bytes do not match the declaration")
    receipt = _load_record_receipt(record_receipt_path)
    if (receipt["receipt_sha256"] != evidence["wrapped_record_receipt_sha256"]
            or receipt["record_sha256"] != evidence["wrapped_record_sha256"]):
        raise DataSourceError("wrapped record identity does not match the T4E.37 declaration")

    raw_rows = measurement.get("paths", {}).get("raw_field", {}).get("rows", [])
    failed = [row for row in raw_rows if row.get("inside_radius") == 0]
    identities = [[row["sid"], row["time"], row["storm"]] for row in failed]
    if identities != declaration["frozen_population"]["identities"]:
        raise DataSourceError("T4E.36 failed-row population does not match the T4E.37 declaration")

    rows: List[Dict[str, Any]] = []
    record_path = Path(receipt["record_path"])
    with xr.open_zarr(record_path, consolidated=True) as dataset:
        if streaming_content_sha256(dataset, time_block=hash_block_frames) != receipt["record_sha256"]:
            raise DataSourceError("wrapped record content does not match its receipt")
        latitudes = np.asarray(dataset.latitude.values, dtype=np.float64)
        longitudes = np.asarray(dataset.longitude.values, dtype=np.float64)
        for expected in failed:
            observation = _observation(expected)
            stamp = np.datetime64(observation.time.replace(" ", "T"))
            try:
                frame = dataset.vo.sel(time=stamp).squeeze(drop=True)
            except (KeyError, ValueError) as exc:
                raise DataSourceError(
                    "wrapped record lacks fixed storm-time %s" % observation.time) from exc
            values = -np.asarray(frame.values, dtype=np.float64)
            extraction = extract(_field(values), n_surrogates=SURROGATES, seed=SEED)
            reproduced = join_row(
                observation, _positions(extraction, latitudes, longitudes)).describe()
            if (reproduced["features"] != expected["features"]
                    or reproduced["inside_radius"] != expected["inside_radius"]
                    or not _same_distances(
                        reproduced["distances_km"], expected["distances_km"])):
                raise DataSourceError(
                    "unchanged extractor does not reproduce T4E.36 row %s" % observation.name)

            maxima = sampled_local_maxima(values, latitudes, longitudes)
            for maximum in maxima:
                maximum["distance_km"] = great_circle_km(
                    observation.centre, (maximum["lat"], maximum["lon"]))
            maxima.sort(key=lambda item: (item["distance_km"], item["grid_index"]))
            inside = [dict(item) for item in maxima
                      if item["distance_km"] <= observation.radius_km]
            threshold = float(extraction.calibration.threshold)
            for maximum in inside:
                maximum["clears_calibrated_threshold"] = bool(
                    maximum["field_value"] > threshold)
            rows.append({
                "storm": observation.name,
                "sid": observation.sid,
                "time": observation.time,
                "catalogue_radius_km": observation.radius_km,
                "sampled_local_maxima": len(maxima),
                "nearest_sampled_local_maximum_km": (
                    maxima[0]["distance_km"] if maxima else None),
                "inside_radius_sampled_local_maxima": inside,
                "inside_radius_sampled_count": len(inside),
                "published_extractor_nearest_km": reproduced["nearest_km"],
                "published_extractor_features": reproduced["features"],
                "calibration_threshold": threshold,
                "extractor_rejected": dict(extraction.rejected),
                "classification": classify_failure(len(inside)),
            })

    decision = majority_decision(rows)
    result = {
        "schema": FAILURE_ATTRIBUTION_SCHEMA,
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": _file_sha256(declaration_path),
        "adoption": adoption,
        "sources": evidence,
        "population": {
            "rows": len(rows), "passing_rows_reclassified": 0,
            "catalogue_census_rerun": False, "forecast_test_opened": False,
        },
        "diagnostic": {
            "negated": True, "neighbourhood": "3x3", "boundary_margin": 1,
            "longitude_periodic": False, "position": "native_grid_sample",
            "distance_function": "catalogue_join.great_circle_km",
            "n_surrogates": SURROGATES, "seed": SEED,
        },
        "rows": rows,
        "decision": decision,
        "OUTCOME": decision["outcome"],
        "VERDICT": "NOT_AN_ACCEPTANCE",
        "claim_boundary": declaration["claim_boundary"],
    }
    result["receipt_sha256"] = _sha256(result)
    try:
        publish_new_bytes(
            output_path,
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "T4E.37 failure-attribution measurement")
    except FileExistsError as exc:
        raise DataSourceError(str(exc), path=str(output_path)) from exc
    return result
