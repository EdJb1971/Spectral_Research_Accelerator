"""A dated four-domain live-source measurement, or an honestly closed release gate.

TG17.10 deliberately left ``live_sources`` at ``NOT_RUN``. Acquisition code and isolated
opt-in checks existed, but there was no evidence channel from a complete, dated live run into
the release ledger. This module is that channel; it does not perform network access.

The distinction between the four domains is part of the contract. Reanalysis, Argo and TESS
name public or credentialled archives and a passing outcome for them must say that network was
used. The bespoke order-book family has no archive by design: its passing live binding is a
researcher-supplied, content-addressed local record and must say that network was *not* used.
Calling all four "public archives" would contradict the adapter the gate is meant to qualify.

As with the calendar, browser and extension evidence channels, recording and deciding are
separate. A future opt-in runner writes only what happened. ``read_live_source_evidence``
decides whether that recording is complete, current and passing. Absence, source drift and a
partial or structurally invalid recording mean the measurement has not happened for this
checkout and read ``NOT_RUN``; an operational refusal reads ``REFUSED``; an executed check that
found a broken contract reads ``FAIL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.experiment_manifest import flagship_recipe


SCHEMA = "live-source-evidence/v1"
RECORD_KIND = "four_domain_live_source_acceptance_not_scientific_result"
REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_DIR = "measurements"
RECORD_NAME = "live_sources.json"

PUBLIC_NETWORK_DOMAINS: Tuple[str, ...] = (
    "reanalysis", "argo_float", "tess_lightcurve")
LOCAL_RECORD_DOMAINS: Tuple[str, ...] = ("order_book",)

# Small, historical, provider-stable acceptance requests. They qualify transport and record
# binding, not the flagship's scientific duration matrix, so paying for a six-month tail here
# would add cost without testing another contract. They are frozen before any response is seen.
FROZEN_REQUESTS: Mapping[str, Mapping[str, Any]] = {
    "reanalysis": {
        "variables": ["t"], "date_start": "2025-01-01", "date_end": "2025-01-01",
        "hours_utc": [0, 6, 12, 18], "lat_min": -46.0, "lat_max": -44.0,
        "lon_min": 170.0, "lon_max": 172.0, "pressure_levels": [850],
        "grid_degrees": 0.25, "n_levels_analysis": 1,
    },
    "argo_float": {
        "time_start": "2026-01-01", "time_end": "2026-03-01",
        "lat_min": -46.0, "lat_max": -34.0, "lon_min": 165.0, "lon_max": 180.0,
        "pressure_min_dbar": 0.0, "pressure_max_dbar": 200.0,
        "variables": ["temperature"], "max_profiles": 200,
    },
    "tess_lightcurve": {
        "target_id": "261136679", "sectors": [1], "flux_column": "PDCSAP_FLUX",
        "quality_policy": "quality_zero", "max_products": 2,
        "max_download_bytes": 64 * 1024 * 1024,
    },
    "order_book": {
        "time_column": "t", "time_units": "s", "delimiter": ",",
        "record_selection": "researcher_supplied_before_execution",
    },
}

# Every implementation that decides how one of the four records is addressed or admitted.
# Normalising line endings keeps a recording portable across Windows and POSIX checkouts.
SOURCE_FILES: Tuple[str, ...] = (
    "src/adapters/reanalysis.py",
    "extensions/argo_float.py",
    "extensions/tess_lightcurve.py",
    "src/adapters/bespoke_record.py",
    "src/adapters/standardized_level_adapter.py",
    "src/data_layer/cds_source.py",
    "src/data_layer/argo_source.py",
    "src/data_layer/tess_source.py",
    "src/data_layer/tabular_source.py",
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "ABSENT"
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _content_digest(path: Path) -> str:
    """Exact record identity; unlike source identity, line endings are measurement bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _utc(value: Any) -> Optional[datetime]:
    text = str(value or "")
    if not text.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        return None


def source_digests(root: Optional[Path] = None) -> Dict[str, str]:
    base = Path(root) if root is not None else REPO_ROOT
    return {name: _file_digest(base / name) for name in SOURCE_FILES}


def live_source_contract() -> Dict[str, Any]:
    """The source identities the live tail must cover, derived from the flagship manifest."""
    domains = []
    for observation in flagship_recipe().observations:
        domain = observation.domain
        local = domain in LOCAL_RECORD_DOMAINS
        domains.append({
            "domain": domain,
            "source_id": observation.acquisition.source_id,
            "source_version": observation.acquisition.source_version,
            "access_kind": "content_addressed_local_record" if local else "archive",
            "network_required_for_pass": not local,
            "content_identity_required": True,
            "coverage_must_be_recorded": True,
            "request": dict(FROZEN_REQUESTS[domain]),
        })
    contract = {
        "schema": SCHEMA,
        "record_kind": RECORD_KIND,
        "domains": domains,
        "public_network_domains": list(PUBLIC_NETWORK_DOMAINS),
        "local_record_domains": list(LOCAL_RECORD_DOMAINS),
        "allowed_outcomes": ["PASS", "REFUSED", "FAIL"],
        "claim_boundary": (
            "This qualifies bounded acquisition and record binding in four domains. It is not "
            "a cross-domain scientific result, evidence, replication or claim promotion."),
    }
    return {**contract, "contract_sha256": _digest(contract)}


def record_file(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else REPO_ROOT
    return base / MEASUREMENT_DIR / RECORD_NAME


def live_source_plan(root: Optional[Path] = None,
                     order_book_record: Optional[Path] = None) -> Dict[str, Any]:
    """Render the exact run before permission is sought; this function is network-dark."""
    base = Path(root) if root is not None else REPO_ROOT
    contract = live_source_contract()
    local = None
    if order_book_record is not None:
        candidate = Path(order_book_record)
        if candidate.is_file():
            local = {"path_supplied": True, "bytes": candidate.stat().st_size,
                     "sha256": _content_digest(candidate)}
        else:
            local = {"path_supplied": True, "exists": False}
    era5 = FROZEN_REQUESTS["reanalysis"]
    latitude_points = int(round(
        (float(era5["lat_max"]) - float(era5["lat_min"]))
        / float(era5["grid_degrees"]))) + 1
    longitude_points = int(round(
        (float(era5["lon_max"]) - float(era5["lon_min"]))
        / float(era5["grid_degrees"]))) + 1
    return {
        "schema": SCHEMA,
        "contract": contract,
        "source_sha256": source_digests(base),
        "order_book_record": local or {"path_supplied": False},
        "network_used": False,
        "network_requirements": {
            "argument": "confirm_network_access=True",
            "environment": "SPECTRALEARTH_ALLOW_NETWORK=1",
            "both_required": True,
        },
        "resource_caps": {
            "reanalysis_values": (len(era5["hours_utc"])
                                  * len(era5["pressure_levels"])
                                  * len(era5["variables"])
                                  * latitude_points * longitude_points),
            "argo_profiles": FROZEN_REQUESTS["argo_float"]["max_profiles"],
            "tess_products": FROZEN_REQUESTS["tess_lightcurve"]["max_products"],
            "tess_download_bytes":
                FROZEN_REQUESTS["tess_lightcurve"]["max_download_bytes"],
        },
        "record_path": str(record_file(base)),
        "claim_boundary": contract["claim_boundary"],
    }


Probe = Callable[[Mapping[str, Any], Path, Path], Mapping[str, Any]]


def _reanalysis_probe(declared: Mapping[str, Any], base: Path,
                      _local_record: Path) -> Mapping[str, Any]:
    from src.data_layer.cds_source import CDSRegionalRequest, materialise_cds

    request = dict(declared["request"])
    for name in ("variables", "hours_utc", "pressure_levels"):
        request[name] = tuple(request[name])
    spec = CDSRegionalRequest(**request)
    scratch = base / "data" / "live_source_qualification"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reanalysis-", dir=str(scratch)) as directory:
        temporary = Path(directory)
        result = materialise_cds(
            spec, download_dir=temporary / "downloads", cache_dir=str(temporary / "cache"),
            time_chunk=4, check_size=False, allow_network=True)
    shape = dict(result["shape"])
    observed = len(result["variables"])
    for name in ("time", "level", "latitude", "longitude"):
        observed *= int(shape[name])
    return {
        "network_used": not result.get("cache_hit") and int(result["bytes_transferred"]) > 0,
        "coverage": {"start": spec.date_start + "T00:00:00Z",
                     "end": spec.date_end + "T18:00:00Z", "exact": True},
        "content": {"sha256": result["content_hash"], "observed_values": observed,
                    "bytes": int(result["bytes_transferred"])},
        "detail": {"request_sha256": spec.request_sha256(), "shape": shape,
                   "source_route": result["source_route"]},
    }


def _argo_probe(declared: Mapping[str, Any], _base: Path,
                _local_record: Path) -> Mapping[str, Any]:
    from src.data_layer.argo_source import acquire_profiles
    from src.data_layer.profiles import ProfileSpec

    request = dict(declared["request"])
    request["variables"] = tuple(request["variables"])
    spec = ProfileSpec(**request)
    collection = acquire_profiles(spec)
    description = collection.describe()
    return {
        "network_used": True,
        "coverage": {"start": spec.time_start, "end": spec.time_end, "exact": False},
        "content": {"sha256": description["collection_sha256"],
                    "observed_values": sum(description["valid_value_count"].values()),
                    "records": description["n_profiles"]},
        "detail": {"request_sha256": description["request_sha256"],
                   "platforms": description["n_platforms"],
                   "response_sha256": description["response_sha256"]},
    }


def _tess_probe(declared: Mapping[str, Any], _base: Path,
                _local_record: Path) -> Mapping[str, Any]:
    from src.data_layer.lightcurves import LightCurveSpec
    from src.data_layer.tess_source import acquire_tess

    request = dict(declared["request"])
    request["sectors"] = tuple(request["sectors"])
    spec = LightCurveSpec(**request)
    collection = acquire_tess(spec)
    description = collection.describe()
    finite = int(sum(math.isfinite(float(value)) for value in collection.flux))
    return {
        "network_used": True,
        "coverage": {"start": str(float(collection.times_bjd_tdb.min())),
                     "end": str(float(collection.times_bjd_tdb.max())), "exact": True,
                     "time_scale": "BJD_TDB"},
        "content": {"sha256": description["collection_sha256"],
                    "observed_values": finite, "records": description["n_products"]},
        "detail": {"request_sha256": description["request_sha256"],
                   "sectors": description["sectors"], "target": description["target"]},
    }


def _order_book_probe(declared: Mapping[str, Any], _base: Path,
                      local_record: Path) -> Mapping[str, Any]:
    import numpy as np

    from src.core.builtin_domains import register_builtin_domains
    from src.data_layer.tabular_source import read_channels_for_domain

    register_builtin_domains()
    text = local_record.read_bytes().decode("utf-8")
    request = declared["request"]
    series, _declaration = read_channels_for_domain(
        text, source_name=local_record.name, domain="order_book",
        time_column=str(request["time_column"]), time_units=str(request["time_units"]),
        delimiter=str(request["delimiter"]))
    present = (int(np.asarray(series.present, dtype=bool).sum()) if series.present is not None
               else series.n_times * series.n_channels)
    return {
        "network_used": False,
        "coverage": {"start": str(float(series.times_seconds[0])),
                     "end": str(float(series.times_seconds[-1])), "exact": True,
                     "time_units": "s"},
        "content": {"sha256": series.provenance["content_sha256"],
                    "observed_values": present, "records": series.n_times},
        "detail": {"channels": list(series.channels),
                   "domain_attribution": series.provenance["domain_attribution"]},
    }


DEFAULT_PROBES: Mapping[str, Probe] = {
    "reanalysis": _reanalysis_probe,
    "argo_float": _argo_probe,
    "tess_lightcurve": _tess_probe,
    "order_book": _order_book_probe,
}


def _now_utc() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def measure_live_sources(*, order_book_record: Path, order_book_provenance: str,
                         order_book_licence: str, root: Optional[Path] = None,
                         confirm_network_access: bool = False,
                         probes: Optional[Mapping[str, Probe]] = None) -> Dict[str, Any]:
    """Perform and record the bounded run, behind two explicit network permissions.

    Supplying a probe mapping exists for offline contract tests. It does not weaken either lock.
    Provider/declared refusals are recorded separately from unexpected implementation failures.
    """
    if confirm_network_access is not True:
        raise PermissionError("live-source measurement requires confirm_network_access=True")
    if os.environ.get("SPECTRALEARTH_ALLOW_NETWORK") != "1":
        raise PermissionError(
            "live-source measurement also requires SPECTRALEARTH_ALLOW_NETWORK=1")
    local = Path(order_book_record)
    if not local.is_file():
        raise FileNotFoundError("the content-addressed bespoke record does not exist: %s" % local)

    base = Path(root) if root is not None else REPO_ROOT
    from src.core.errors import InvalidParameterError, SpectralEarthError
    if not str(order_book_provenance).strip():
        raise InvalidParameterError(
            "order_book_provenance", order_book_provenance,
            "a researcher declaration of where the record came from")
    if not str(order_book_licence).strip():
        raise InvalidParameterError(
            "order_book_licence", order_book_licence,
            "the licence or authority under which the record may be used")
    local_digest = _content_digest(local)
    demo_root = base / "data" / "channels"
    demo_digests = {_content_digest(path) for path in demo_root.glob("*.csv") if path.is_file()}
    if local_digest in demo_digests:
        raise InvalidParameterError(
            "order_book_record", local_digest,
            "a non-fixture researcher-supplied record. The committed channel CSVs are explicitly "
            "fabricated demonstrations and cannot satisfy a live-source gate")

    contract = live_source_contract()
    implementations = dict(DEFAULT_PROBES if probes is None else probes)
    outcomes: List[Dict[str, Any]] = []

    for declared in contract["domains"]:
        domain = declared["domain"]
        started = _now_utc()
        common = {"domain": domain, "source_id": declared["source_id"],
                  "source_version": declared["source_version"], "started_utc": started}
        try:
            measured = dict(implementations[domain](declared, base, local))
            if domain == "order_book":
                detail = dict(measured.get("detail") or {})
                detail["operator_declaration"] = {
                    "provenance": str(order_book_provenance).strip(),
                    "licence": str(order_book_licence).strip(),
                    "content_sha256": local_digest,
                }
                measured["detail"] = detail
            outcome = {**common, **measured, "status": "PASS", "reason": ""}
        except SpectralEarthError as error:
            outcome = {**common, "status": "REFUSED", "network_used": None,
                       "network_attempted": domain in PUBLIC_NETWORK_DOMAINS,
                       "coverage": {"start": "REFUSED", "end": "REFUSED", "exact": False},
                       "reason": "%s: %s" % (type(error).__name__, error.message)}
        except Exception as error:  # an implementation failure is not an archive refusal
            reason = str(error).replace(str(base), "<repository>").replace(
                str(local), "<bespoke-record>")
            outcome = {**common, "status": "FAIL", "network_used": None,
                       "network_attempted": domain in PUBLIC_NETWORK_DOMAINS,
                       "coverage": {"start": "FAILED", "end": "FAILED", "exact": False},
                       "reason": "%s: %s" % (type(error).__name__, reason)}
        outcome["finished_utc"] = _now_utc()
        outcomes.append(outcome)

    record = {
        "schema": SCHEMA, "record_kind": RECORD_KIND, "recorded_utc": _now_utc(),
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": source_digests(base), "outcomes": outcomes,
        "bespoke_record": {"content_sha256": local_digest,
                           "provenance": str(order_book_provenance).strip(),
                           "licence": str(order_book_licence).strip()},
        "claim_boundary": contract["claim_boundary"],
    }
    record["record_sha256"] = _digest(record)
    path = record_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return record


def _structural_reasons(record: Mapping[str, Any], contract: Mapping[str, Any],
                        sources: Mapping[str, str]) -> List[str]:
    reasons: List[str] = []
    if record.get("schema") != SCHEMA:
        reasons.append("the recording has schema %r rather than %s"
                       % (record.get("schema"), SCHEMA))
    if record.get("record_kind") != RECORD_KIND:
        reasons.append("the recording is not labelled as live-source acceptance")
    supplied_digest = record.get("record_sha256")
    digest_body = {key: value for key, value in record.items() if key != "record_sha256"}
    if supplied_digest != _digest(digest_body):
        reasons.append("the live-source record digest does not match its contents")
    if record.get("contract_sha256") != contract["contract_sha256"]:
        reasons.append("the declared live-source contract has changed since it was recorded")
    if _utc(record.get("recorded_utc")) is None:
        reasons.append("the recording has no exact UTC recording time")

    recorded_sources = dict(record.get("source_sha256") or {})
    moved = sorted(name for name, digest in sources.items()
                   if recorded_sources.get(name) != digest)
    if moved:
        reasons.append("the source of %s has changed since the run was recorded"
                       % ", ".join(moved))

    expected = [row["domain"] for row in contract["domains"]]
    outcomes = list(record.get("outcomes") or [])
    observed = [row.get("domain") for row in outcomes]
    if sorted(observed, key=str) != sorted(expected) or len(observed) != len(set(observed)):
        absent = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected), key=str)
        reasons.append(
            "the recorded domain inventory is incomplete or duplicated"
            + (": missing %s" % ", ".join(absent) if absent else "")
            + ("; unexpected %s" % ", ".join(map(str, extra)) if extra else ""))

    by_domain = {row["domain"]: row for row in contract["domains"]}
    bespoke = dict(record.get("bespoke_record") or {})
    if not str(bespoke.get("provenance") or "").strip() \
            or not str(bespoke.get("licence") or "").strip() \
            or not _is_sha256(bespoke.get("content_sha256")):
        reasons.append("the bespoke record has no complete provenance, licence and content binding")
    for outcome in outcomes:
        domain = outcome.get("domain")
        declared = by_domain.get(domain)
        if declared is None:
            continue
        if outcome.get("source_id") != declared["source_id"]:
            reasons.append("%s was measured against a different source identity" % domain)
        if outcome.get("source_version") != declared["source_version"]:
            reasons.append("%s was measured against a different source version" % domain)
        if outcome.get("status") not in contract["allowed_outcomes"]:
            reasons.append("%s has no recognised measured outcome" % domain)
        started, finished = _utc(outcome.get("started_utc")), _utc(outcome.get("finished_utc"))
        if started is None or finished is None:
            reasons.append("%s does not record its exact measurement interval" % domain)
        elif finished < started:
            reasons.append("%s finishes before its measurement starts" % domain)
        coverage = outcome.get("coverage")
        if not isinstance(coverage, Mapping) or not coverage.get("start") \
                or not coverage.get("end") or "exact" not in coverage:
            reasons.append("%s does not record its archive/record coverage" % domain)

        if outcome.get("status") == "PASS":
            if outcome.get("network_used") is not declared["network_required_for_pass"]:
                expected_network = "network use" if declared["network_required_for_pass"] \
                    else "no network use"
                reasons.append("%s PASS does not demonstrate %s" % (domain, expected_network))
            content = outcome.get("content")
            if not isinstance(content, Mapping) or not _is_sha256(content.get("sha256")) \
                    or int(content.get("observed_values") or 0) <= 0:
                reasons.append("%s PASS has no non-empty content-addressed record" % domain)
            if domain == "order_book" and isinstance(content, Mapping) \
                    and content.get("sha256") != bespoke.get("content_sha256"):
                reasons.append("order_book outcome does not match its declared record content")
        elif outcome.get("status") == "REFUSED" and not outcome.get("reason"):
            reasons.append("%s refusal gives no reason" % domain)
        elif outcome.get("status") == "FAIL" and not outcome.get("reason"):
            reasons.append("%s failure gives no reason" % domain)
    return reasons


def read_live_source_evidence(root: Optional[Path] = None) -> Dict[str, Any]:
    """Return the gate status earned by the recorded four-domain live-source run."""
    base = Path(root) if root is not None else REPO_ROOT
    contract = live_source_contract()
    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "gate": "live_sources",
        "measured_by": "an explicit opt-in four-domain acquisition run",
        "executed_here": False,
        "network_used_here": False,
        "record_path": "%s/%s" % (MEASUREMENT_DIR, RECORD_NAME),
        "contract_sha256": contract["contract_sha256"],
        "domains": [row["domain"] for row in contract["domains"]],
        "claim_boundary": contract["claim_boundary"],
    }
    path = record_file(base)
    if not path.is_file():
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["no four-domain live-source run has been recorded for this checkout"]}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["the live-source recording could not be read: %s" % error]}

    reasons = _structural_reasons(record, contract, source_digests(base))
    summary = {
        "recorded_utc": record.get("recorded_utc"),
        "outcomes": [{"domain": row.get("domain"), "status": row.get("status"),
                      "network_used": row.get("network_used")}
                     for row in record.get("outcomes") or []],
    }
    facts = {**facts, "recorded": summary}
    if reasons:
        return {**facts, "status": "NOT_RUN", "reasons": reasons}

    outcomes = list(record["outcomes"])
    failures = [row for row in outcomes if row["status"] == "FAIL"]
    refusals = [row for row in outcomes if row["status"] == "REFUSED"]
    if failures:
        return {**facts, "status": "FAIL",
                "reasons": ["%s failed: %s" % (row["domain"], row["reason"])
                            for row in failures]}
    if refusals:
        return {**facts, "status": "REFUSED",
                "reasons": ["%s refused: %s" % (row["domain"], row["reason"])
                            for row in refusals]}
    return {**facts, "status": "PASS", "reasons": []}


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly run the bounded TG17.14 live-source qualification")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the exact network-dark contract")
    plan.add_argument("--order-book-record", type=Path)
    run = commands.add_parser("run", help="perform the bounded acquisition and write evidence")
    run.add_argument("--order-book-record", type=Path, required=True)
    run.add_argument("--order-book-provenance", required=True)
    run.add_argument("--order-book-licence", required=True)
    run.add_argument(
        "--confirm-network-access", required=True,
        choices=("I_AUTHORIZE_BOUNDED_ARCHIVE_REQUESTS",),
        help="exact acknowledgement required in addition to SPECTRALEARTH_ALLOW_NETWORK=1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _cli_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = live_source_plan(order_book_record=args.order_book_record)
    else:
        try:
            result = measure_live_sources(
                order_book_record=args.order_book_record,
                order_book_provenance=args.order_book_provenance,
                order_book_licence=args.order_book_licence,
                confirm_network_access=True)
        except (PermissionError, FileNotFoundError) as error:
            parser.error(str(error))
        except Exception as error:
            from src.core.errors import SpectralEarthError
            if isinstance(error, SpectralEarthError):
                parser.error(error.message)
            raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["DEFAULT_PROBES", "FROZEN_REQUESTS", "LOCAL_RECORD_DOMAINS", "MEASUREMENT_DIR",
           "PUBLIC_NETWORK_DOMAINS", "RECORD_KIND", "RECORD_NAME", "REPO_ROOT", "SCHEMA",
           "SOURCE_FILES", "live_source_contract", "live_source_plan", "measure_live_sources",
           "main", "read_live_source_evidence", "record_file", "source_digests"]


if __name__ == "__main__":
    raise SystemExit(main())
