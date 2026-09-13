"""Fixed-population T4E.36 join over a verified wrapped record."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from src.benchmarks.catalogue_join import Observation, aggregates, join_row
from src.core.domain import AxisSpec
from src.core.errors import DataSourceError
from src.core.extraction import ExtractionField, extract
from src.core.publication import publish_new_bytes
from src.core.representation import plane_field, representation_planes
from src.data_layer.wrapped_record import (
    WRAPPED_RECORD_SCHEMA,
    load_wrapped_declaration,
    require_current_adoption,
)
from src.data_layer.zarr_source import streaming_content_sha256


WRAPPED_JOIN_SCHEMA = "t4e36-wrapped-join/v1"
DEFAULT_BASELINE = Path("measurements/t4e28_join_rerun.json")
DEFAULT_OUTPUT = Path("measurements/t4e36_wrapped_join.json")
SURROGATES = 99
SEED = 1234
SWT_CONFIG = {"wavelet": "db2", "level": 3}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positions(features: Sequence[Any], latitudes: np.ndarray,
               longitudes: np.ndarray) -> List[Tuple[float, float]]:
    rows = np.arange(len(latitudes))
    columns = np.arange(len(longitudes))
    return [
        (float(np.interp(float(feature.location.coords["row"]), rows, latitudes)),
         float(np.interp(float(feature.location.coords["col"]), columns, longitudes)))
        for feature in features
    ]


def _field(values: np.ndarray) -> ExtractionField:
    axes = (AxisSpec("row", "space", units="cells", ordinal=0),
            AxisSpec("col", "space", units="cells", ordinal=1))
    return ExtractionField(
        values=values, axes=axes, domain="reanalysis",
        dataset="era5_sp_wrapped_850hPa_vo",
        variable="relative_vorticity_negated", units="s**-1",
        time=0.0, time_units="s", representation="identity")


def _raw_positions(field: ExtractionField, latitudes: np.ndarray,
                   longitudes: np.ndarray) -> List[Tuple[float, float]]:
    return _positions(
        extract(field, n_surrogates=SURROGATES, seed=SEED), latitudes, longitudes)


def _swt_positions(field: ExtractionField, latitudes: np.ndarray,
                   longitudes: np.ndarray) -> List[Tuple[float, float]]:
    found: List[Tuple[float, float]] = []
    for plane in representation_planes("swt", field, SWT_CONFIG):
        if plane.extractable:
            found.extend(_positions(
                extract(plane_field(field, "swt", plane, SWT_CONFIG),
                        n_surrogates=SURROGATES, seed=SEED),
                latitudes, longitudes))
    return found


def _observation(row: Mapping[str, Any]) -> Observation:
    return Observation(
        sid=str(row["sid"]), name=str(row["storm"]), time=str(row["time"]),
        lat=float(row["lat"]), lon=float(row["lon"]), nature=str(row["nature"]),
        radius_km=float(row["radius_km"]), agency_fixes=2)


def _load_record_receipt(path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    source = Path(path)
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("wrapped-record receipt is unreadable: %s" % exc) from exc
    if receipt.get("schema") != WRAPPED_RECORD_SCHEMA:
        raise DataSourceError("unsupported wrapped-record receipt schema")
    unsigned = dict(receipt)
    supplied = unsigned.pop("receipt_sha256", None)
    if supplied != _sha256(unsigned):
        raise DataSourceError("wrapped-record receipt digest is missing or invalid")
    return receipt


def evaluate_wrapped_join(
    declaration_path: Union[str, os.PathLike[str]],
    record_receipt_path: Union[str, os.PathLike[str]],
    *,
    baseline_path: Union[str, os.PathLike[str]] = DEFAULT_BASELINE,
    output_path: Union[str, os.PathLike[str]] = DEFAULT_OUTPUT,
    hash_block_frames: int = 32,
) -> Dict[str, Any]:
    """Evaluate exactly the 18 T4E.28 rows; never rerun the wider catalogue census."""
    declaration_path = Path(declaration_path)
    declaration = load_wrapped_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    receipt = _load_record_receipt(record_receipt_path)
    if receipt.get("declaration_sha256") != hashlib.sha256(
            declaration_path.read_bytes()).hexdigest():
        raise DataSourceError("wrapped record is bound to a different declaration")

    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        old_paths = {
            name: baseline["paths"][name]["rows"]
            for name in ("raw_field", "swt_planes")}
        old_rows = old_paths["raw_field"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataSourceError("T4E.28 fixed-population record is unreadable: %s" % exc) from exc
    if len(old_rows) != 18:
        raise DataSourceError(
            "T4E.36 requires exactly the 18 T4E.28 rows; found %d" % len(old_rows))
    identities = [(row["sid"], row["time"]) for row in old_rows]
    if any(len(old_paths[name]) != 18 or
           [(row["sid"], row["time"]) for row in old_paths[name]] != identities
           for name in old_paths):
        raise DataSourceError("T4E.28 extraction paths do not describe one identical 18-row population")

    record_path = Path(receipt["record_path"])
    with xr.open_zarr(record_path, consolidated=True) as dataset:
        observed_hash = streaming_content_sha256(dataset, time_block=hash_block_frames)
        if observed_hash != receipt.get("record_sha256"):
            raise DataSourceError("wrapped record content does not match its receipt")
        latitudes = np.asarray(dataset.latitude.values, dtype=np.float64)
        longitudes = np.asarray(dataset.longitude.values, dtype=np.float64)
        results: Dict[str, Any] = {}
        for name, extractor in (("raw_field", _raw_positions),
                                ("swt_planes", _swt_positions)):
            rows = []
            join_rows = []
            for index, old in enumerate(old_rows):
                observation = _observation(old)
                stamp = np.datetime64(observation.time.replace(" ", "T"))
                try:
                    frame = dataset.vo.sel(time=stamp).squeeze(drop=True)
                except (KeyError, ValueError) as exc:
                    raise DataSourceError(
                        "wrapped record lacks fixed storm-time %s" % observation.time) from exc
                values = -np.asarray(frame.values, dtype=np.float64)
                row = join_row(
                    observation, extractor(_field(values), latitudes, longitudes))
                join_rows.append(row)
                described = row.describe()
                described["baseline_same_path"] = old_paths[name][index]
                described["stress_row"] = row.storm in set(
                    declaration["frozen_population"]["east_edge_stress_rows"])
                described["retained_counterexample"] = row.storm == "RUBY"
                rows.append(described)
            results[name] = {"rows": rows, "aggregates": aggregates(join_rows)}

    raw = results["raw_field"]["aggregates"]
    condition_1 = raw["condition_1_at_least_one_inside_radius"]
    condition_2 = raw["condition_2_three_inside_radius"]
    verdict = "PASS" if condition_1["met"] >= 9 and condition_2["met"] >= 9 else "FAIL"
    measurement = {
        "schema": WRAPPED_JOIN_SCHEMA,
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
        "adoption": adoption,
        "wrapped_record_receipt": str(record_receipt_path).replace("\\", "/"),
        "wrapped_record_receipt_sha256": receipt["receipt_sha256"],
        "fixed_population": {
            "source": str(Path(baseline_path)).replace("\\", "/"),
            "source_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
            "rows": len(old_rows), "catalogue_census_rerun": False,
        },
        "extraction": {
            "negated": True, "n_surrogates": SURROGATES, "seed": SEED,
            "raw_field": "representation 'identity'",
            "swt_planes": SWT_CONFIG,
        },
        "paths": results,
        "gate": {
            "verdict": verdict,
            "primary_path": "raw_field",
            "condition_1": {**condition_1, "passes": condition_1["met"] >= 9},
            "condition_2": {**condition_2, "passes": condition_2["met"] >= 9},
            "swt_can_rescue_raw_failure": False,
        },
        "VERDICT": verdict,
        "claim_boundary": declaration["claim_boundary"],
    }
    measurement["receipt_sha256"] = _sha256(measurement)
    try:
        publish_new_bytes(
            output_path,
            json.dumps(measurement, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "T4E.36 wrapped-join measurement")
    except FileExistsError as exc:
        raise DataSourceError(str(exc), path=str(output_path)) from exc
    return measurement
