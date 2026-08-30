"""Bounded anonymous access to calibrated TESS-SPOC light curves at MAST (TG13.1)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from src.core.errors import InvalidParameterError
from src.data_layer.lightcurves import (LIGHTCURVE_SOURCES, LightCurveCollection,
                                        LightCurveProduct, LightCurveSource, LightCurveSpec)

NETWORK_ENV_VAR = "SPECTRALEARTH_ALLOW_NETWORK"
MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
MAST_TIMEOUT_SECONDS = 90
MAST_TRANSPORT_ATTEMPTS = 2
MAX_METADATA_ROWS = 200
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


def parse_spoc_fits(payload: bytes, spec: LightCurveSpec, product: LightCurveProduct
                    ) -> Dict[str, Any]:
    """Read one calibrated SPOC LC product and retain its archive flags."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise InvalidParameterError("astronomy dependency", "astropy missing",
                                    "install requirements-astronomy.txt") from exc
    try:
        with fits.open(io.BytesIO(payload), memmap=False, checksum=True) as hdus:
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
            return {
                "time": time_values + reference,
                "flux": np.asarray(table.data[spec.flux_column], dtype=np.float64),
                "flux_error": np.asarray(table.data[spec.flux_column + "_ERR"], dtype=np.float64),
                "quality": np.asarray(table.data["QUALITY"], dtype=np.int64),
                "ra_deg": float(primary.get("RA_OBJ")), "dec_deg": float(primary.get("DEC_OBJ")),
                "sector": np.full(time_values.shape, product.sector, dtype=np.int64)}
    except InvalidParameterError:
        raise
    except Exception as exc:
        raise InvalidParameterError("TESS FITS", product.filename,
                                    "a readable checksum-valid SPOC light curve") from exc


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
    order = np.argsort(time_values, kind="stable")
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
        target_ra_deg=plan["target"]["ra_deg"], target_dec_deg=plan["target"]["dec_deg"])


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


__all__ = ["MAST_DOWNLOAD_URL", "MAST_INVOKE_URL", "NETWORK_ENV_VAR", "TessMastSource",
           "acquire_tess", "inspect_tess_query", "network_enabled", "parse_spoc_fits",
           "register_tess_source"]
