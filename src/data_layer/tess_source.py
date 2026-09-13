"""Bounded anonymous access to calibrated TESS-SPOC light curves at MAST (TG13.1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Mapping, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.publication import publish_new_bytes
from src.data_layer.lightcurves import (LIGHTCURVE_SOURCES, LightCurveCollection,
                                        LightCurveProduct, LightCurveSource, LightCurveSpec)

NETWORK_ENV_VAR = "SPECTRALEARTH_ALLOW_NETWORK"
MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
MAST_TIMEOUT_SECONDS = 90
MAST_TRANSPORT_ATTEMPTS = 2
MAX_METADATA_ROWS = 200
DISCOVERY_SCHEMA = "spectral.tess-pool-discovery.v1"
Query = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Download = Callable[[str, int], bytes]


def network_enabled() -> bool:
    return os.environ.get(NETWORK_ENV_VAR, "") == "1"


def _require_network() -> None:
    if not network_enabled():
        raise InvalidParameterError(
            "network access", "disabled",
            "explicit opt-in with %s=1 before contacting MAST" % NETWORK_ENV_VAR)


def _mast_query(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Submit one bounded MAST request, including its documented long-poll protocol."""
    _require_network()
    body = dict(request)
    body.setdefault("format", "json")
    body.setdefault("pagesize", MAX_METADATA_ROWS)
    body.setdefault("page", 1)
    body.setdefault("timeout", 60)
    # A stable cachebreaker is required when polling: changing it creates another job.
    body.setdefault("cachebreaker", hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:20])
    encoded = urlencode({"request": json.dumps(body, separators=(",", ":"))}).encode()
    for _attempt in range(6):
        req = Request(MAST_INVOKE_URL, data=encoded,
                      headers={"Content-Type": "application/x-www-form-urlencoded",
                               "Accept": "application/json",
                               "User-Agent": "SpectralEarth/1.0 TG13.1"})
        failure: Exception | None = None
        for transport_attempt in range(MAST_TRANSPORT_ATTEMPTS):
            try:
                with urlopen(req, timeout=MAST_TIMEOUT_SECONDS) as response:
                    payload = response.read(8 * 1024 * 1024 + 1)
                failure = None
                break
            except Exception as exc:
                failure = exc
                if transport_attempt + 1 < MAST_TRANSPORT_ATTEMPTS:
                    time.sleep(0.5)
        if failure is not None:
            raise InvalidParameterError(
                "MAST query", body.get("service"),
                "a response after %d bounded transport attempts of %d seconds each; MAST "
                "did not complete this metadata request, so no empty result was invented"
                % (MAST_TRANSPORT_ATTEMPTS, MAST_TIMEOUT_SECONDS)) from failure
        if len(payload) > 8 * 1024 * 1024:
            raise InvalidParameterError("MAST metadata", len(payload),
                                        "at most 8 MiB; narrow the target or sector family")
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidParameterError("MAST response", "non-JSON", "valid JSON metadata") from exc
        status = str(result.get("status", "")).upper()
        if status == "COMPLETE":
            if len(result.get("data", ())) > MAX_METADATA_ROWS:
                raise InvalidParameterError("MAST metadata rows", len(result["data"]),
                                            "at most %d" % MAX_METADATA_ROWS)
            return result
        if status not in ("EXECUTING", "PENDING"):
            raise InvalidParameterError("MAST query status", result,
                                        "COMPLETE; archive failures are not empty results")
        time.sleep(0.25)
    raise InvalidParameterError("MAST query status", "still executing",
                                "completion within six bounded polls")


def _download(data_uri: str, maximum: int) -> bytes:
    _require_network()
    req = Request(MAST_DOWNLOAD_URL + "?" + urlencode({"uri": data_uri}),
                  headers={"Accept": "application/fits",
                           "User-Agent": "SpectralEarth/1.0 TG13.1"})
    try:
        with urlopen(req, timeout=MAST_TIMEOUT_SECONDS) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > maximum:
                raise InvalidParameterError("MAST product bytes", int(declared),
                                            "at most %d bytes" % maximum)
            payload = response.read(maximum + 1)
    except InvalidParameterError:
        raise
    except Exception as exc:
        raise InvalidParameterError("MAST product", data_uri,
                                    "an anonymous download within the timeout") from exc
    if len(payload) > maximum:
        raise InvalidParameterError("MAST product bytes", len(payload),
                                    "at most %d bytes" % maximum)
    return payload


def _catalogue_target(spec: LightCurveSpec, query: Query) -> Tuple[float, float]:
    result = query({
        "service": "Mast.Catalogs.Filtered.Tic",
        "params": {"columns": "ID,ra,dec",
                   "filters": [{"paramName": "ID", "values": [spec.target_id]}]},
        "format": "json", "pagesize": 2, "page": 1})
    rows = list(result.get("data", ()))
    if len(rows) != 1 or str(rows[0].get("ID")) != spec.target_id:
        raise InvalidParameterError("target_id", spec.target_id,
                                    "one exact TIC catalogue record")
    try:
        ra, dec = float(rows[0]["ra"]), float(rows[0]["dec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidParameterError("TIC coordinates", rows[0], "finite ICRS degrees") from exc
    if not np.isfinite(ra) or not np.isfinite(dec) or not -90 <= dec <= 90:
        raise InvalidParameterError("TIC coordinates", [ra, dec], "finite ICRS degrees")
    return ra, dec


def _observations(spec: LightCurveSpec, ra: float, dec: float, query: Query) -> List[Mapping[str, Any]]:
    filters = [{"paramName": "obs_collection", "values": ["TESS"]},
               {"paramName": "dataproduct_type", "values": ["timeseries"]},
               {"paramName": "sequence_number",
                "values": [str(value) for value in spec.sectors]}]
    result = query({
        "service": "Mast.Caom.Filtered.Position", "format": "json",
        "params": {"columns": "obsid,obs_id,target_name,sequence_number,provenance_name",
                   "filters": filters, "position": "%.12g, %.12g, 0.001" % (ra, dec)},
        "pagesize": MAX_METADATA_ROWS, "page": 1})
    rows = [row for row in result.get("data", ())
            if int(row.get("sequence_number", -1)) in spec.sectors]
    # MAST position search may return neighbours. Product filename matching below is the exact
    # TIC boundary; no target is admitted merely because it lies in the cone.
    if not rows:
        raise InvalidParameterError("TESS observations", spec.canonical(),
                                    "at least one public timeseries in the requested sectors")
    return rows


def _products(spec: LightCurveSpec, observations: Sequence[Mapping[str, Any]],
              query: Query) -> List[LightCurveProduct]:
    target_token = spec.target_id.zfill(16)
    products: Dict[str, LightCurveProduct] = {}
    for observation in observations:
        result = query({"service": "Mast.Caom.Products",
                        "params": {"obsid": observation["obsid"]},
                        "format": "json", "pagesize": MAX_METADATA_ROWS, "page": 1})
        for row in result.get("data", ()):
            filename = str(row.get("productFilename", ""))
            if str(row.get("productType", "")).upper() != "SCIENCE" \
                    or str(row.get("productSubGroupDescription", "")).upper() != "LC" \
                    or target_token not in filename:
                continue
            product = LightCurveProduct(
                observation_id=str(row.get("obs_id") or observation.get("obs_id") or
                                   observation.get("obsid")),
                data_uri=str(row.get("dataURI", "")), filename=filename,
                sector=int(observation["sequence_number"]), size_bytes=int(row.get("size", 0)),
                provenance_name=str(observation.get("provenance_name") or "SPOC"))
            products[product.data_uri] = product
    ordered = sorted(products.values(), key=lambda value: (value.sector, value.filename))
    if not ordered:
        raise InvalidParameterError("TESS light-curve products", spec.canonical(),
                                    "at least one exact-TIC public LC product")
    return ordered


def inspect_tess_query(spec: LightCurveSpec, *, query: Query = _mast_query) -> Dict[str, Any]:
    """Resolve exact products and bytes without downloading a FITS value."""
    ra, dec = _catalogue_target(spec, query)
    observations = _observations(spec, ra, dec, query)
    products = _products(spec, observations, query)
    total = sum(value.size_bytes for value in products)
    return {"schema": "spectral.lightcurve-plan.v1", "request": spec.canonical(),
            "request_sha256": spec.request_sha256(),
            "target": {"tic_id": spec.target_id, "ra_deg": ra, "dec_deg": dec,
                       "frame": "ICRS"},
            "candidate_products": len(products), "predicted_download_bytes": total,
            "within_product_cap": len(products) <= spec.max_products,
            "within_byte_cap": total <= spec.max_download_bytes,
            "products": [value.describe() for value in products],
            "metadata_only": True,
            "claim_boundary": "Archive metadata only; no light-curve value was transferred."}


def _tic_from_observation(row: Mapping[str, Any]) -> str | None:
    target = str(row.get("target_name", "")).strip()
    match = re.fullmatch(r"(?:TIC\s*)?0*(\d+)", target, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"tess-s\d+-(\d{16})(?:-|$)",
                          str(row.get("obs_id", "")), flags=re.IGNORECASE)
    if not match or int(match.group(1)) <= 0:
        return None
    return str(int(match.group(1)))


def discover_tess_targets(*, sector: int, target_limit: int = 48,
                          query: Query = _mast_query) -> Dict[str, Any]:
    """Find exact public SPOC light-curve products without downloading their values."""
    if isinstance(sector, bool) or int(sector) != sector or int(sector) <= 0:
        raise InvalidParameterError("sector", sector, "a positive integer sector number")
    if isinstance(target_limit, bool) or int(target_limit) != target_limit \
            or not 1 <= int(target_limit) <= MAX_METADATA_ROWS:
        raise InvalidParameterError(
            "target_limit", target_limit,
            "an integer from 1 through %d" % MAX_METADATA_ROWS)
    sector = int(sector)
    target_limit = int(target_limit)
    request = {
        "service": "Mast.Caom.Filtered", "format": "json",
        "params": {
            "columns": "obsid,obs_id,target_name,sequence_number,provenance_name",
            "filters": [
                {"paramName": "obs_collection", "values": ["TESS"]},
                {"paramName": "dataproduct_type", "values": ["timeseries"]},
                {"paramName": "sequence_number", "values": [str(sector)]},
                {"paramName": "provenance_name", "values": ["SPOC"]},
            ],
        },
        "pagesize": MAX_METADATA_ROWS, "page": 1,
    }
    result = query(request)
    rows = [row for row in result.get("data", ())
            if int(row.get("sequence_number", -1)) == sector
            and str(row.get("provenance_name", "")).upper() == "SPOC"]
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    unnamed = 0
    for row in rows:
        target_id = _tic_from_observation(row)
        if target_id is None:
            unnamed += 1
            continue
        grouped.setdefault(target_id, []).append(row)

    selected = []
    refused = []
    for target_id in sorted(grouped, key=int):
        if len(selected) >= target_limit:
            break
        spec = LightCurveSpec(target_id=target_id, sectors=(sector,), max_products=16,
                              max_download_bytes=512 * 1024 * 1024)
        try:
            products = _products(spec, grouped[target_id], query)
        except InvalidParameterError as error:
            refused.append({"tic_id": target_id, "reason": str(error)})
            continue
        selected.append({
            "tic_id": target_id,
            "observation_ids": sorted({str(row.get("obsid"))
                                       for row in grouped[target_id]}),
            "products": [product.describe() for product in products],
            "predicted_download_bytes": sum(product.size_bytes for product in products),
        })

    encoded = json.dumps(
        {"request": request, "rows": rows}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": DISCOVERY_SCHEMA,
        "source": "MAST TESS-SPOC",
        "sector": sector,
        "target_limit": target_limit,
        "metadata_rows": len(rows),
        "metadata_row_cap": MAX_METADATA_ROWS,
        "metadata_page_full": len(rows) == MAX_METADATA_ROWS,
        "unnamed_observations": unnamed,
        "candidate_targets_in_page": len(grouped),
        "selected_targets": len(selected),
        "shortfall": max(0, target_limit - len(selected)),
        "predicted_download_bytes": sum(
            row["predicted_download_bytes"] for row in selected),
        "targets": selected,
        "refused_targets": refused,
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "selection_basis": (
            "numeric TIC order among exact SPOC light-curve products in the first bounded "
            "archive metadata page; no photometric values were opened"),
        "metadata_only": True,
        "claim_boundary": (
            "Catalogue and product availability only. Selection establishes neither target "
            "independence, profile validity, admission nor exchangeability."),
    }


def discover_periodic_tess_products(
        period_catalogue: Mapping[str, Any], *, target_limit: int = 64,
    excluded_target_ids: Collection[str] = (), workers: int = 4,
        query: Query = _mast_query) -> Dict[str, Any]:
    """Join externally declared periods to exact SPOC products without opening values."""
    if period_catalogue.get("schema") != "spectral.tess-period-candidates/v1":
        raise InvalidParameterError(
            "period catalogue schema", period_catalogue.get("schema"),
            "spectral.tess-period-candidates/v1")
    candidates = period_catalogue.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise InvalidParameterError("period candidates", candidates, "a non-empty candidate list")
    if isinstance(target_limit, bool) or not 1 <= int(target_limit) <= 200:
        raise InvalidParameterError("target_limit", target_limit, "an integer from 1 through 200")
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= int(workers) <= 8:
        raise InvalidParameterError("workers", workers, "an integer from 1 through 8")
    excluded = {str(value).strip() for value in excluded_target_ids}
    if any(not value.isdigit() or int(value) <= 0 for value in excluded):
        raise InvalidParameterError(
            "excluded_target_ids", sorted(excluded), "positive TIC identifiers")
    candidates = [row for row in candidates if str(row.get("tic_id", "")).strip() not in excluded]

    def inspect_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        target_id = str(candidate.get("tic_id", "")).strip()
        result = query({
            "service": "Mast.Caom.Filtered", "format": "json",
            "params": {
                "columns": "obsid,obs_id,target_name,sequence_number,provenance_name",
                "filters": [
                    {"paramName": "obs_collection", "values": ["TESS"]},
                    {"paramName": "dataproduct_type", "values": ["timeseries"]},
                    {"paramName": "provenance_name", "values": ["SPOC"]},
                    {"paramName": "target_name", "values": [target_id]},
                ],
            },
            "pagesize": MAX_METADATA_ROWS, "page": 1,
        })
        observations = sorted(
            [row for row in result.get("data", ())
             if str(row.get("target_name", "")).strip() == target_id
             and str(row.get("provenance_name", "")).upper() == "SPOC"],
            key=lambda row: (int(row.get("sequence_number", 0)), str(row.get("obsid", ""))))
        chosen = None
        for observation in observations:
            sector = int(observation.get("sequence_number", 0))
            if sector <= 0:
                continue
            spec = LightCurveSpec(target_id=target_id, sectors=(sector,), max_products=16,
                                  max_download_bytes=512 * 1024 * 1024)
            try:
                products = _products(spec, [observation], query)
            except InvalidParameterError:
                continue
            chosen = products[0]
            break
        if chosen is None:
            return {"tic_id": target_id,
                    "reason": "no exact public SPOC LC product found"}
        return {
            **candidate,
            "sector": chosen.sector,
            "product": chosen.describe(),
            "predicted_download_bytes": chosen.size_bytes,
        }

    inspected = []
    selected = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for start in range(0, len(candidates), int(workers)):
            batch = list(executor.map(
                inspect_candidate, candidates[start:start + int(workers)]))
            inspected.extend(batch)
            selected.extend(row for row in batch if "product" in row)
            if len(selected) >= int(target_limit):
                break
    selected = selected[:int(target_limit)]
    refused = [row for row in inspected if "product" not in row]

    return {
        "schema": "spectral.tess-period-product-discovery/v1",
        "period_catalogue_response_sha256": period_catalogue.get("response_sha256"),
        "target_limit": int(target_limit),
        "excluded_target_count": len(excluded),
        "excluded_target_ids_sha256": hashlib.sha256(json.dumps(
            sorted(excluded, key=int), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "selected_targets": len(selected),
        "shortfall_from_g17_minimum": max(0, 48 - len(selected)),
        "predicted_download_bytes": sum(row["predicted_download_bytes"] for row in selected),
        "targets": selected,
        "refused_targets": refused,
        "metadata_only": True,
        "selection_basis": (
            "unique-period TIC order from the frozen NASA Exoplanet Archive response, then "
            "the earliest sector with an exact public SPOC LC product"),
        "claim_boundary": (
            "External period and archive-product availability only. No FITS values were opened; "
            "actual cycle coverage, profile validity, independence and exchangeability remain "
            "unassessed."),
    }


def acquire_discovered_tess(
        discovery: Mapping[str, Any], *, root: Path,
        maximum_total_bytes: int = 128 * 1024 * 1024,
    workers: int = 4,
        download: Download = _download) -> Dict[str, Any]:
    """Acquire exactly one metadata-preflighted target set into a raw immutable cache."""
    discovery_schema = discovery.get("schema")
    supported_schemas = (DISCOVERY_SCHEMA, "spectral.tess-period-product-discovery/v1")
    if discovery_schema not in supported_schemas:
        raise InvalidParameterError(
            "discovery schema", discovery_schema, "one of %s" % (supported_schemas,))
    targets = discovery.get("targets")
    if not isinstance(targets, list) or not targets \
            or discovery.get("selected_targets") != len(targets):
        raise InvalidParameterError(
            "discovery targets", discovery.get("selected_targets"),
            "a non-empty target list matching selected_targets")
    declared_total = discovery.get("predicted_download_bytes")
    if isinstance(declared_total, bool) or not isinstance(declared_total, int) \
            or declared_total <= 0:
        raise InvalidParameterError(
            "predicted_download_bytes", declared_total, "a positive integer byte count")
    if isinstance(maximum_total_bytes, bool) or int(maximum_total_bytes) < declared_total:
        raise InvalidParameterError(
            "maximum_total_bytes", maximum_total_bytes,
            "at least %d from the frozen discovery before any FITS product is downloaded"
            % declared_total)
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= int(workers) <= 8:
        raise InvalidParameterError("workers", workers, "an integer from 1 through 8")

    product_total = 0
    products_by_target = []
    seen_targets = set()
    seen_uris = set()
    for target in targets:
        target_id = str(target.get("tic_id", "")).strip()
        if not target_id.isdigit() or int(target_id) <= 0 or target_id in seen_targets:
            raise InvalidParameterError(
                "discovery tic_id", target_id, "one unique positive TIC identifier per target")
        raw_products = ([target["product"]] if "product" in target
                        else target.get("products", ()))
        products = [LightCurveProduct(**value) for value in raw_products]
        if not products:
            raise InvalidParameterError(
                "discovery products[%s]" % target_id,
                len(products), "at least one exact SPOC light-curve product")
        token = target_id.zfill(16)
        if any(token not in product.filename for product in products):
            raise InvalidParameterError(
                "discovery products[%s]" % target_id,
                [product.filename for product in products],
                "filenames containing the exact zero-padded TIC identifier")
        for product in products:
            if product.data_uri in seen_uris:
                raise InvalidParameterError(
                    "discovery data_uri", product.data_uri,
                    "one unique archive product in the acquisition")
            seen_uris.add(product.data_uri)
            product_total += product.size_bytes
        if target.get("predicted_download_bytes") != sum(
                product.size_bytes for product in products):
            raise InvalidParameterError(
                "target predicted_download_bytes", target.get("predicted_download_bytes"),
                "the sum of its declared product sizes")
        seen_targets.add(target_id)
        products_by_target.append((target, target_id, products))
    if product_total != declared_total:
        raise InvalidParameterError(
            "predicted_download_bytes", declared_total,
            "the exact sum of all declared product sizes (%d)" % product_total)

    flat_products = [
        product for _, _, products in products_by_target for product in products]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        payloads = dict(zip(
            [product.data_uri for product in flat_products],
            executor.map(lambda product: download(product.data_uri, product.size_bytes),
                         flat_products)))

    root = Path(root)
    acquired = []
    for target, target_id, products in products_by_target:
        spec = LightCurveSpec(
            target_id=target_id, sectors=tuple(sorted({product.sector for product in products})),
            max_products=len(products),
            max_download_bytes=sum(product.size_bytes for product in products))
        timestamped = 0
        finite_quality_zero = 0
        product_records = []
        for product in products:
            payload = payloads[product.data_uri]
            if len(payload) != product.size_bytes:
                raise InvalidParameterError(
                    "downloaded product bytes", len(payload),
                    "the preflighted size of %d within the aggregate cap" % product.size_bytes)
            parsed = parse_spoc_fits(payload, spec, product)
            clocked = np.isfinite(parsed["time"])
            timestamped += int(np.count_nonzero(clocked))
            finite_quality_zero += int(np.count_nonzero(
                clocked & np.isfinite(parsed["flux"]) & (parsed["quality"] == 0)))
            digest = hashlib.sha256(payload).hexdigest()
            path = root / (digest + ".fits")
            if path.exists():
                if path.read_bytes() != payload:
                    raise InvalidParameterError(
                        "raw TESS cache", str(path), "bytes matching its content digest")
                publication = "already-present-identical"
            else:
                publication = publish_new_bytes(path, payload, "raw TESS FITS product")
            product_records.append({
                **product.describe(), "content_sha256": digest,
                "cache_file": path.name, "publication": publication,
                "fits_checksum": parsed["fits_checksum"],
            })
        acquired.append({
            "tic_id": target_id, "timestamped_samples": timestamped,
            "finite_quality_zero_flux": finite_quality_zero,
            **({
                "toi": target["toi"],
                "disposition": target["disposition"],
                "orbital_period_seconds": target["orbital_period_seconds"],
                "period_error_upper_days": target["period_error_upper_days"],
                "period_error_lower_days": target["period_error_lower_days"],
            } if discovery_schema == "spectral.tess-period-product-discovery/v1" else {}),
            "products": product_records,
        })

    discovery_digest = hashlib.sha256(json.dumps(
        discovery, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema": "spectral.tess-pool-acquisition/v1",
            "discovery_schema": discovery_schema,
        "discovery_sha256": discovery_digest,
        "target_count": len(acquired),
        "downloaded_bytes": declared_total,
        "raw_cache": root.as_posix(),
        "targets": acquired,
        "claim_boundary": (
            "This receipt validates and binds raw archive products only. It does not derive "
            "G17 profile marginals or establish target independence, admission or exchangeability."),
    }


def parse_spoc_fits(payload: bytes, spec: LightCurveSpec, product: LightCurveProduct
                    ) -> Dict[str, Any]:
    """Read one calibrated SPOC LC product and retain its archive flags."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise InvalidParameterError("astronomy dependency", "astropy missing",
                                    "install requirements-astronomy.txt") from exc
    try:
        with warnings.catch_warnings(record=True) as checksum_warnings:
            warnings.simplefilter("always")
            hdus = fits.open(io.BytesIO(payload), memmap=False, checksum=True)
        with hdus:
            primary, table = hdus[0].header, hdus[1]
            names = set(table.columns.names)
            required = {"TIME", spec.flux_column, spec.flux_column + "_ERR", "QUALITY"}
            if not required <= names:
                raise InvalidParameterError("FITS columns", sorted(names),
                                            "columns %s" % sorted(required))
            tic = str(primary.get("TICID", "")).strip()
            if tic and str(int(tic)) != spec.target_id:
                raise InvalidParameterError("FITS TICID", tic, spec.target_id)
            time_values = np.asarray(table.data["TIME"], dtype=np.float64)
            reference = float(table.header.get("BJDREFI", primary.get("BJDREFI", 0))) \
                + float(table.header.get("BJDREFF", primary.get("BJDREFF", 0)))
            scale = str(table.header.get("TIMESYS", primary.get("TIMESYS", "TDB"))).upper()
            if scale != "TDB" or reference <= 0:
                raise InvalidParameterError("FITS time system", [scale, reference],
                                            "TIME + a positive BJDREF in TDB")
            checksum = [{
                "hdu": str(hdu.header.get("EXTNAME", "PRIMARY")),
                "checksum_present": "CHECKSUM" in hdu.header,
                "checksum_valid": (
                    None if getattr(hdu, "_checksum_valid", None) is None
                    else bool(hdu._checksum_valid)),
                "datasum_present": "DATASUM" in hdu.header,
                "datasum_valid": (
                    None if getattr(hdu, "_datasum_valid", None) is None
                    else bool(hdu._datasum_valid)),
            } for hdu in hdus]
            return {
                "time": time_values + reference,
                "flux": np.asarray(table.data[spec.flux_column], dtype=np.float64),
                "flux_error": np.asarray(table.data[spec.flux_column + "_ERR"], dtype=np.float64),
                "quality": np.asarray(table.data["QUALITY"], dtype=np.int64),
                "ra_deg": float(primary.get("RA_OBJ")), "dec_deg": float(primary.get("DEC_OBJ")),
                "sector": np.full(time_values.shape, product.sector, dtype=np.int64),
                "fits_checksum": checksum,
                "fits_checksum_warnings": [str(item.message) for item in checksum_warnings]}
    except InvalidParameterError:
        raise
    except Exception as exc:
        raise InvalidParameterError("TESS FITS", product.filename,
                                    "a readable SPOC light curve with checksum state recorded") from exc


def acquire_tess(spec: LightCurveSpec, *, query: Query = _mast_query,
                 download: Download = _download) -> LightCurveCollection:
    plan = inspect_tess_query(spec, query=query)
    if not plan["within_product_cap"] or not plan["within_byte_cap"]:
        raise InvalidParameterError(
            "TESS acquisition plan",
            {"products": plan["candidate_products"],
             "bytes": plan["predicted_download_bytes"]},
            "at most max_products=%d and max_download_bytes=%d before values are fetched"
            % (spec.max_products, spec.max_download_bytes))
    products = [LightCurveProduct(**value) for value in plan["products"]]
    parsed, digests = [], []
    remaining = int(spec.max_download_bytes)
    for product in products:
        payload = download(product.data_uri, remaining)
        remaining -= len(payload)
        if remaining < 0:
            raise InvalidParameterError("downloaded product bytes", spec.max_download_bytes - remaining,
                                        "at most %d" % spec.max_download_bytes)
        parsed.append(parse_spoc_fits(payload, spec, product))
        digests.append(hashlib.sha256(payload).hexdigest())
    time_values = np.concatenate([value["time"] for value in parsed])
    # SPOC emits a row for every cadence in the window, including the ones photometry was
    # not produced for, and those carry a non-finite TIME. They are dropped here and counted
    # rather than carried: a sample with no timestamp cannot be placed on a clock, so it is
    # not a measurement this collection can hold. The count travels in the receipt (D95).
    clocked = np.isfinite(time_values)
    unclocked = int(time_values.size - int(np.count_nonzero(clocked)))
    if not np.any(clocked):
        raise InvalidParameterError(
            "TESS product clocks", "no finite timestamps in %d rows" % time_values.size,
            "at least one cadence the archive actually timestamped")
    kept = np.flatnonzero(clocked)
    # The order array both sorts and selects. The duplicate check below was never blind to
    # these rows -- numpy sorts NaN to the end, so the finite prefix was always compared
    # correctly -- and it is checked here after the drop for the plainer reason that a
    # duplicate among timestamped cadences is what it exists to catch.
    order = kept[np.argsort(time_values[kept], kind="stable")]
    if np.any(np.diff(time_values[order]) <= 0):
        raise InvalidParameterError("TESS product clocks", "duplicate or reversed samples",
                                    "a strictly increasing union without silently deduplicating")
    ra = [value["ra_deg"] for value in parsed]
    dec = [value["dec_deg"] for value in parsed]
    if not np.allclose(ra, plan["target"]["ra_deg"], atol=1 / 3600, rtol=0) \
            or not np.allclose(dec, plan["target"]["dec_deg"], atol=1 / 3600, rtol=0):
        raise InvalidParameterError("FITS target coordinates", [ra, dec],
                                    "the resolved TIC position within one arcsecond")
    def joined(name: str) -> np.ndarray:
        return np.concatenate([value[name] for value in parsed])[order]
    return LightCurveCollection(
        spec=spec, products=products, times_bjd_tdb=time_values[order],
        flux=joined("flux"), flux_error=joined("flux_error"), quality=joined("quality"),
        sectors=joined("sector"), source_sha256=digests,
        target_ra_deg=plan["target"]["ra_deg"], target_dec_deg=plan["target"]["dec_deg"],
        unclocked_samples_dropped=unclocked)


class TessMastSource(LightCurveSource):
    name = "mast_tess_spoc"
    domain = "tess_lightcurve"
    access = "anonymous"

    def inspect(self, spec: LightCurveSpec) -> Mapping[str, Any]:
        return inspect_tess_query(spec)

    def fetch(self, spec: LightCurveSpec) -> LightCurveCollection:
        return acquire_tess(spec)

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "archive": "MAST", "collection": "TESS",
                "pipeline": "SPOC", "product_subgroup": "LC",
                "network_env_var": NETWORK_ENV_VAR, "metadata_preflight": True,
                "maximum_products": 16, "maximum_download_bytes": 512 * 1024 * 1024}


def register_tess_source() -> LightCurveSource:
    from extensions.tess_lightcurve import register as register_tess_domain

    register_tess_domain()
    if TessMastSource.name not in LIGHTCURVE_SOURCES:
        LIGHTCURVE_SOURCES.add(
            TessMastSource.name, TessMastSource(),
            description="Calibrated TESS-SPOC LC products from the official MAST archive.",
            capabilities={"metadata_preflight": True, "bounded_download": True,
                          "quality_flags": True, "multi_sector": True})
    return LIGHTCURVE_SOURCES.get(TessMastSource.name)


__all__ = ["DISCOVERY_SCHEMA", "MAST_DOWNLOAD_URL", "MAST_INVOKE_URL", "NETWORK_ENV_VAR",
           "TessMastSource", "acquire_discovered_tess", "acquire_tess",
           "discover_periodic_tess_products", "discover_tess_targets", "inspect_tess_query",
           "network_enabled", "parse_spoc_fits", "register_tess_source"]
