"""Bounded public catalogue access for externally declared TESS native periods."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from typing import Any, Callable, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.benchmarks.pool_calibration import SPAN_CYCLES
from src.core.errors import InvalidParameterError


TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
NETWORK_ENV_VAR = "SPECTRALEARTH_ALLOW_NETWORK"
SCHEMA = "spectral.tess-period-candidates/v1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
Fetch = Callable[[str, int], bytes]


def _fetch(url: str, maximum: int = MAX_RESPONSE_BYTES) -> bytes:
    if os.getenv(NETWORK_ENV_VAR, "0").strip().lower() not in ("1", "true", "yes", "on"):
        raise InvalidParameterError(
            NETWORK_ENV_VAR, os.getenv(NETWORK_ENV_VAR),
            "1 before public NASA Exoplanet Archive access")
    request = Request(url, headers={
        "Accept": "text/csv",
        "User-Agent": "SpectralEarth/1.0 G17-period-catalogue",
    })
    with urlopen(request, timeout=90) as response:  # nosec B310 - fixed HTTPS host
        payload = response.read(maximum + 1)
    if len(payload) > maximum:
        raise InvalidParameterError(
            "period catalogue bytes", len(payload), "at most %d" % maximum)
    return payload


def discover_periodic_tess_candidates(
        *, candidate_limit: int = 200, nominal_sector_days: float = 27.0,
        minimum_cycles: float = SPAN_CYCLES, fetch: Fetch = _fetch) -> Dict[str, Any]:
    """Read TOI periods fixed outside the light curves later compared by G17."""
    if isinstance(candidate_limit, bool) or not 1 <= int(candidate_limit) <= 500:
        raise InvalidParameterError("candidate_limit", candidate_limit, "an integer from 1 to 500")
    if not math.isfinite(float(nominal_sector_days)) or nominal_sector_days <= 0.0:
        raise InvalidParameterError("nominal_sector_days", nominal_sector_days, "a positive span")
    if not math.isfinite(float(minimum_cycles)) or minimum_cycles < SPAN_CYCLES:
        raise InvalidParameterError(
            "minimum_cycles", minimum_cycles,
            "at least the calibrated %.1f-cycle span" % SPAN_CYCLES)
    maximum_period_days = float(nominal_sector_days) / float(minimum_cycles)
    query = (
        "select top %d tid,toi,tfopwg_disp,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
        "pl_orbperlim from toi where pl_orbper is not null and pl_orbper>0 and "
        "pl_orbper<=%.17g and tfopwg_disp<>'FP' and "
        "(pl_orbperlim is null or pl_orbperlim=0) order by tid"
        % (int(candidate_limit), maximum_period_days))
    url = TAP_URL + "?" + urlencode({"query": query, "format": "csv"})
    payload = fetch(url, MAX_RESPONSE_BYTES)
    try:
        text = payload.decode("utf-8-sig")
        fields = ("tid", "toi", "tfopwg_disp", "pl_orbper", "pl_orbpererr1",
                  "pl_orbpererr2", "pl_orbperlim")
        parsed = list(csv.reader(io.StringIO(text)))
        if parsed and tuple(parsed[0]) == fields:
            parsed = parsed[1:]
        if any(len(row) != len(fields) for row in parsed):
            raise ValueError("expected seven TOI period columns")
        by_target: Dict[str, List[Dict[str, Any]]] = {}
        for values in parsed:
            row = dict(zip(fields, values))
            target_id = str(row["tid"]).strip()
            period = float(row["pl_orbper"])
            if not target_id.isdigit() or int(target_id) <= 0 \
                    or not math.isfinite(period) or not 0.0 < period <= maximum_period_days:
                raise ValueError("invalid TIC period row")
            by_target.setdefault(str(int(target_id)), []).append({
                "tic_id": str(int(target_id)), "toi": str(row["toi"]).strip(),
                "disposition": str(row["tfopwg_disp"]).strip(),
                "orbital_period_days": period,
                "orbital_period_seconds": period * 86400.0,
                "period_error_upper_days": (
                    float(row["pl_orbpererr1"]) if row["pl_orbpererr1"].strip() else None),
                "period_error_lower_days": (
                    float(row["pl_orbpererr2"]) if row["pl_orbpererr2"].strip() else None),
            })
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise InvalidParameterError(
            "TOI period response", "malformed", "the declared CSV columns and finite periods") \
            from error
    ambiguous = [{
        "tic_id": target_id,
        "reason": "multiple catalogue orbital periods for one stellar light curve",
        "toi_periods": [{"toi": row["toi"], "orbital_period_days": row["orbital_period_days"]}
                        for row in values],
    } for target_id, values in sorted(by_target.items(), key=lambda item: int(item[0]))
        if len(values) != 1]
    rows = [values[0] for _, values in sorted(by_target.items(), key=lambda item: int(item[0]))
            if len(values) == 1]
    return {
        "schema": SCHEMA,
        "source": "NASA Exoplanet Archive TOI table",
        "source_url": TAP_URL,
        "query": query,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_limit": int(candidate_limit),
        "candidate_count": len(rows),
        "ambiguous_host_count": len(ambiguous),
        "refused_ambiguous_hosts": ambiguous,
        "nominal_sector_days": float(nominal_sector_days),
        "minimum_cycles": float(minimum_cycles),
        "maximum_period_days": maximum_period_days,
        "candidates": rows,
        "native_seconds_basis": (
            "NASA Exoplanet Archive TOI pl_orbper converted from days to seconds; not estimated "
            "from the SPOC flux used by the correspondence statistic"),
        "claim_boundary": (
            "Catalogue candidates only. A TOI disposition and period do not establish that a "
            "usable SPOC light curve exists, that its flux traces the orbital shape, or that "
            "records are independent or exchangeable."),
    }


__all__ = ["SCHEMA", "TAP_URL", "discover_periodic_tess_candidates"]