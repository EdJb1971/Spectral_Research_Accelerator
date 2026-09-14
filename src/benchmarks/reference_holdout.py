"""Pre-opening T4E.39 temporal-holdout declaration and census boundary."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from src.benchmarks.catalogue_join import Selection, deepest_per_storm, read_observations
from src.core.adoption import adoption_state
from src.core.errors import DataSourceError
from src.core.publication import publish_new_bytes
from src.data_layer.signed_reference import resolve


HOLDOUT_DECLARATION_SCHEMA = "t4e39-reference-holdout-declaration/v1"
HOLDOUT_CENSUS_SCHEMA = "t4e39-reference-holdout-census/v1"
DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e39-reference-holdout-declaration.json")
DEFAULT_CENSUS = Path("measurements/t4e39_reference_holdout_census.json")
T4E17_DISCLOSED_BOX = {
    "lat_min": -60.0, "lat_max": -20.0, "lon_min": 140.0, "lon_max": 180.0,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_disclosed_t4e17_population(path: Path, dates: list[str]) -> Dict[str, Any]:
    """Reproduce the historical T4E.17 disclosure in the box it described.

    T4E.17 disclosed 393 rows / 13 storms while its matching domain was -60..-20.
    Its later development-domain amendment moved to -58..-18, which T4E.39 adopts.
    The disclosure is therefore a provenance check, not an expected count for the new box.
    """
    rows = 0
    storms = set()
    bounds = T4E17_DISCLOSED_BOX
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            next(reader)  # units row beneath the header
            for row in reader:
                try:
                    lat = float(row["LAT"])
                    lon = float(row["LON"])
                except (ValueError, KeyError):
                    continue
                if lon < 0:
                    lon += 360.0
                stamp = row.get("ISO_TIME", "")
                if (stamp and dates[0] <= stamp[:10] <= dates[1]
                        and bounds["lat_min"] <= lat <= bounds["lat_max"]
                        and bounds["lon_min"] <= lon <= bounds["lon_max"]):
                    rows += 1
                    storms.add(row.get("SID", ""))
    except OSError as exc:
        raise DataSourceError(
            "T4E.39 cannot reproduce the T4E.17 disclosed population: %s" % exc,
            path=str(path)) from exc
    return {
        "source": "T4E.17 pre-amendment matching domain",
        "catalogue_box": bounds,
        "dates": dates,
        "in_box_observations": rows,
        "distinct_storms": len(storms),
    }


def _bound_json(path: Path, expected_sha256: str, label: str) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise DataSourceError("%s bytes do not match the T4E.39 declaration" % label)
        return json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataSourceError("%s is unreadable: %s" % (label, exc), path=str(path)) from exc


def _validate_receipt(body: Mapping[str, Any], expected: str, label: str) -> None:
    unsigned = dict(body)
    supplied = unsigned.pop("receipt_sha256", None)
    if supplied != _sha256(unsigned) or supplied != expected:
        raise DataSourceError("%s receipt identity is invalid" % label)


def load_holdout_declaration(
    path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("T4E.39 declaration is unreadable: %s" % exc) from exc
    population = body.get("holdout_population", {})
    acquisition = body.get("future_exact_acquisition", {})
    decision = body.get("decision_declared_before_holdout_opening", {})
    bounds = population.get("catalogue_inclusion_box", {})
    segments = acquisition.get("field_segments", [])
    if body.get("schema") != HOLDOUT_DECLARATION_SCHEMA or body.get("task") != "T4E.39":
        raise DataSourceError("unsupported T4E.39 holdout declaration")
    if (population.get("dates") != ["2022-01-01", "2023-12-31"]
            or bounds != {"lat_min": -58.0, "lat_max": -18.0,
                          "lon_min": 140.0, "lon_max": 180.0}
            or population.get("synoptic_hours_utc") != [0, 6, 12, 18]
            or population.get("minimum_agency_fixes") != 2
            or population.get("interior_margin_degrees") != 2.0
            or population.get("minimum_selected_storms") != 10):
        raise DataSourceError("T4E.39 declaration does not preserve the holdout selection")
    if (acquisition.get("grid_degrees") != 0.25
            or acquisition.get("expected_segment_shape") != [161, 161]
            or acquisition.get("expected_joined_longitude_points") != 321
            or len(segments) != 2
            or [item.get("name") for item in segments] != ["parent", "complement"]
            or acquisition.get("mslp", {}).get("canonical_variable") != "msl"
            or acquisition.get("vorticity", {}).get("canonical_variable") != "vo"
            or acquisition.get("vorticity", {}).get("pressure_level_hpa") != 850
            or acquisition.get("vorticity", {}).get("negated") is not True):
        raise DataSourceError("T4E.39 declaration lacks the exact two-field acquisition")
    if (decision.get("majority") != "floor(selected_storms / 2) + 1"
            or "strict majority" not in str(decision.get(
                "HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE", ""))
            or "NOT_AN_ACCEPTANCE" not in str(decision.get("no_acceptance", ""))):
        raise DataSourceError("T4E.39 declaration lacks the frozen holdout decision")
    return body


def require_current_adoption(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    source = Path(declaration_path)
    state = adoption_state(source.parent, source.name)
    if not state.get("adopted"):
        raise DataSourceError(
            "T4E.39 holdout opening is refused: the declaration is not adopted")
    if not state.get("signature_still_reaches_the_declaration"):
        raise DataSourceError(
            "T4E.39 holdout opening is refused: adoption does not bind this declaration")
    return state


def plan_reference_holdout(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    """Validate identities and readiness without parsing any holdout catalogue row."""
    declaration_path = Path(declaration_path)
    declaration = load_holdout_declaration(declaration_path)
    evidence = declaration["source_evidence"]
    if _file_sha256(Path(evidence["t4e38_declaration"])) != evidence[
            "t4e38_declaration_sha256"]:
        raise DataSourceError("T4E.38 declaration bytes drifted from the holdout declaration")
    measurement = _bound_json(
        Path(evidence["t4e38_measurement"]),
        evidence["t4e38_measurement_file_sha256"], "T4E.38 measurement")
    _validate_receipt(
        measurement, evidence["t4e38_measurement_receipt_sha256"], "T4E.38 measurement")
    if (measurement.get("OUTCOME") != evidence["required_t4e38_outcome"]
            or measurement.get("VERDICT") != evidence["required_t4e38_verdict"]):
        raise DataSourceError("T4E.38 source does not carry the declared result and boundary")
    reference = resolve(evidence["signed_reference"], verify_digest=True)
    if (reference.expected_sha256 != evidence["signed_reference_sha256"]
            or reference.expected_bytes != evidence["signed_reference_bytes"]):
        raise DataSourceError("signed-reference terms drifted from the holdout declaration")
    population = declaration["holdout_population"]
    return {
        "task": "T4E.39",
        "status": declaration["status"],
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": _file_sha256(declaration_path),
        "source_outcome": measurement["OUTCOME"],
        "source_verdict": measurement["VERDICT"],
        "selection": population,
        "strict_majority_formula": "floor(n / 2) + 1",
        "minimum_selected_storms": int(population["minimum_selected_storms"]),
        "signed_reference": reference.describe(),
        "catalogue_bytes_read_for_digest_only": bool(reference.available),
        "catalogue_rows_read": False,
        "holdout_identities_enumerated": False,
        "era5_record_opened": False,
        "network_used": False,
        "next_action": "human adoption before the catalogue-only census",
    }


def build_reference_holdout_census(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    *,
    output_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
) -> Dict[str, Any]:
    """After adoption, freeze holdout identities without opening or acquiring ERA5."""
    declaration_path = Path(declaration_path)
    declaration = load_holdout_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    plan = plan_reference_holdout(declaration_path)
    evidence = declaration["source_evidence"]
    catalogue = resolve(evidence["signed_reference"], verify_digest=True)
    catalogue_path = catalogue.require()
    population = declaration["holdout_population"]
    bounds = population["catalogue_inclusion_box"]
    selection = Selection(
        start_date=population["dates"][0], end_date=population["dates"][1],
        south=float(bounds["lat_min"]), north=float(bounds["lat_max"]),
        west=float(bounds["lon_min"]), east=float(bounds["lon_max"]),
        interior_margin=float(population["interior_margin_degrees"]),
        minimum_agencies=int(population["minimum_agency_fixes"]))
    observations, census = read_observations(catalogue_path, selection)
    disclosed = population["disclosed_before_this_declaration"]
    disclosed_check = _count_disclosed_t4e17_population(
        catalogue_path, population["dates"])
    if (disclosed_check["in_box_observations"]
            != disclosed["in_box_observations_before_exact_refusals"]
            or disclosed_check["distinct_storms"]
            != disclosed["distinct_storms_before_exact_refusals"]):
        raise DataSourceError(
            "T4E.39 catalogue does not reproduce the historical T4E.17 "
            "393-row / 13-storm disclosure")
    before_agency_refusal = len(observations) + int(census["too_few_agencies"])
    selected = deepest_per_storm(observations, selection)
    identities = [(item.sid, item.time) for item in selected]
    if (len(identities) != len(set(identities))
            or [item.time for item in selected] != sorted(item.time for item in selected)):
        raise DataSourceError("T4E.39 selected rows are not unique and chronological")
    minimum = int(population["minimum_selected_storms"])
    status = ("READY_FOR_EXACT_ACQUISITION" if len(selected) >= minimum
              else "INSUFFICIENT_HOLDOUT_POPULATION")
    rows = [{
        "sid": item.sid, "storm": item.name, "time": item.time,
        "lat": item.lat, "lon": item.lon, "agency_fixes": item.agency_fixes,
        "catalogue_radius_km": item.radius_km,
    } for item in selected]
    result = {
        "schema": HOLDOUT_CENSUS_SCHEMA,
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": plan["declaration_sha256"],
        "adoption": adoption,
        "signed_reference": catalogue.describe(),
        "disclosed_population_check": disclosed_check,
        "selection": selection.describe(),
        "census": {
            **census,
            "admitted_before_interior_selection": len(observations),
            "in_box_before_agency_refusal": before_agency_refusal,
        },
        "selected_rows": rows,
        "selected_storms": len(rows),
        "minimum_selected_storms": minimum,
        "future_majority_needed": math.floor(len(rows) / 2) + 1,
        "status": status,
        "era5_record_opened": False,
        "network_used": False,
        "claim_boundary": (
            "Catalogue-only census. It freezes holdout identities and exact request times; it "
            "measures no ERA5 field and supplies no comparison or scientific result."),
    }
    result["receipt_sha256"] = _sha256(result)
    try:
        publish_new_bytes(
            output_path,
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "T4E.39 holdout census")
    except FileExistsError as exc:
        raise DataSourceError(str(exc), path=str(output_path)) from exc
    return result
