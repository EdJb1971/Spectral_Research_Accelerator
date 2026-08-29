"""The ingestion seam for channel records (TG8.4, `ed-dev`).

TG8.1 made a domain **declarable** from outside `src/`. A declaration is not a connection: until
this router existed there was no path from any file to any declared domain, so `/findings/domains`
described vocabularies and limits for data nobody could load. Meanwhile
`src/data_layer/tabular_source.py` had read delimited channel records since TG0.2 and was
referenced zero times from `src/api/` and `frontend/src/` — tested infrastructure with no seam.

**Two calls, and the reason is the same one `/import/inspect` has.** That route exists because an
ERA5 file is `(time, level, lat, lon)` and *"picking `[0, 0]` on the researcher's behalf would
import a slice they did not choose while every statistic downstream described that arbitrary
timestep."* Here the arbitrary choices are the clock column and the domain, and neither is made
for the caller:

*   ``POST /inspect`` reports what the file *is* — columns, row count, which columns could serve
    as a clock, whether that clock is strictly increasing and regular — and turns those facts
    into **obligations**: the violations whichever domain the caller picks must already declare.
    It never fills one in. It also reports, for every onboarded domain, whether it admits this
    file and the exact wording of the refusal if it does not, so a refusal is visible before it
    is hit.
*   ``POST /read`` reads the record against a **named onboarded domain**, or refuses by name.

**What a successful read does and does not mean.** It means the domain's declaration admits the
file's shape: its declared axes are ones a channel table can supply, its violations cover what
the clock and the supports require. It does **not** mean the file came from that domain, and no
check here could establish that — the same gap `DOMAIN_ATTRIBUTION_CAVEAT` records for findings.
The caveat is served with every read for that reason, and the view renders the sentence rather
than restating it.

**Stateless.** The series is returned and nothing is stored, exactly as `/import/field` returns a
field the browser then holds. A server-side store would be a second place a record could go stale
against the file it came from.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np

from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT, refuse_bare_confidence
from src.core.builtin_domains import register_builtin_domains
from src.core.domain import declaration_for, refusals_for
from src.core.errors import SpectralEarthError, classify
from src.core.onboarding import DOMAIN_ONBOARDINGS, is_onboarded
from src.data_layer.tabular_source import (VALUE_MEASURE, clock_facts, inspect_delimited,
                                           read_channels_for_domain)

router = APIRouter(prefix="/api/v1/channels", tags=["channels"])

#: Eager, once, at import of this module rather than inside a handler (D35). A domain that
#: appears only after a researcher has visited the right tab is a domain whose refusals can be
#: missed, and here it would also be a domain a file silently could not be read under.
REGISTERED_DOMAINS = register_builtin_domains()

#: The largest preview this router will serve back per channel. A record above it is **not**
#: refused — it loaded fine — but the browser is sent a stated, contiguous head of it rather than
#: the whole thing, and the payload says how many rows it held back. Downsampling would be the
#: wrong answer here for the same reason it is wrong upstream: a thinned series that is not
#: labelled as thinned is indistinguishable from a record that was always that shape.
PREVIEW_ROWS = 5_000


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=("%r is not UTF-8 text. This reads delimited channel records; a binary file "
                    "belongs to /api/v1/import/inspect, which reads NetCDF and Zarr."
                    % filename))


def _json_field(raw: Optional[str], name: str, example: str) -> Optional[Any]:
    if raw is None or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400,
                            detail="`%s` must be JSON, such as %s." % (name, example))


def _handle(error: SpectralEarthError) -> HTTPException:
    """A refusal from the adapter is a 400 carrying the adapter's own words.

    The message is not rewritten. Every refusal in `tabular_source` states what was expected and
    why it matters — that an imputed value upstream of a dependence estimator becomes a finding,
    that a silently reordered record has provenance that no longer describes it — and a handler
    that replaced those with "invalid file" would throw away the only part a researcher needs.
    """
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


@router.post("/inspect")
async def inspect_record(
    file: UploadFile = File(...),
    delimiter: str = Form(","),
    time_column: Optional[str] = Form(None),
    time_units: str = Form("s"),
) -> Dict[str, Any]:
    """Describe an uploaded channel record and the obligations its shape creates.

    Chooses nothing. If the file's first column cannot serve as a clock, the inspection stops and
    says so rather than promoting whichever other column happens to increase — substituting a
    price column for a timestamp because the timestamp ran backwards would be invisible in every
    result downstream.
    """
    text = _decode(await file.read(), file.filename or "upload")
    try:
        report = inspect_delimited(text, source_name=file.filename or "upload",
                                   delimiter=delimiter, time_column=time_column,
                                   time_units=time_units)
    except SpectralEarthError as error:
        raise _handle(error)
    report["attribution_caveat"] = DOMAIN_ATTRIBUTION_CAVEAT
    return refuse_bare_confidence(report, where="channels.inspect")


@router.post("/read")
async def read_record(
    file: UploadFile = File(...),
    domain: str = Form(...),
    time_column: str = Form(...),
    time_units: str = Form("s"),
    delimiter: str = Form(","),
    channel_columns: Optional[str] = Form(None),
    support_parent_px: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Read an uploaded record against a declared domain, or refuse by name.

    `support_parent_px` is a JSON object giving a channel's parent-axis footprint in samples,
    e.g. `{"trade_count": 60}` for a value aggregated over sixty clock samples. It has no
    default beyond one sample per channel, which is a fact about an instantaneous reading rather
    than a guess; and declaring a footprint above one obliges the domain to have declared
    `aggregated_values`, which the adapter enforces.
    """
    text = _decode(await file.read(), file.filename or "upload")
    columns = _json_field(channel_columns, "channel_columns", '["bid", "ask"]')
    supports = _json_field(support_parent_px, "support_parent_px", '{"trade_count": 60}')
    if columns is not None and not isinstance(columns, list):
        raise HTTPException(status_code=400,
                            detail='`channel_columns` must be a JSON array of column names.')
    if supports is not None and not isinstance(supports, dict):
        raise HTTPException(
            status_code=400,
            detail='`support_parent_px` must be a JSON object such as {"trade_count": 60}.')

    try:
        series, declaration = read_channels_for_domain(
            text, source_name=file.filename or "upload", domain=domain,
            time_column=time_column, time_units=time_units, channel_columns=columns,
            support_parent_px=supports, delimiter=delimiter)
    except SpectralEarthError as error:
        raise _handle(error)

    return refuse_bare_confidence(_read_payload(series, declaration, domain),
                                  where="channels.read")


def _read_payload(series: Any, declaration: Any, domain: str) -> Dict[str, Any]:
    """The loaded record, its clock, and what the domain it was read under refuses.

    Channels are a **list of named entries** rather than a mapping keyed by channel name, and
    that is deliberate rather than stylistic: `refuse_bare_confidence` screens every mapping key
    on the way out, so a record with a column honestly named `confidence` would be refused as a
    bare confidence figure if channel names became keys. A column heading is not a claim.
    """
    values = series.to_matrix(VALUE_MEASURE)
    total_rows = int(values.shape[0])
    shown = min(total_rows, PREVIEW_ROWS)
    supports = list(series.support_parent_px or [1.0] * len(series.channels))
    present = (None if getattr(series, "present", None) is None
               else np.asarray(series.present, dtype=bool))

    entries: List[Dict[str, Any]] = []
    for index, name in enumerate(series.channels):
        channel_present = (np.ones(total_rows, dtype=bool) if present is None
                           else present[:, index])
        present_count = int(channel_present.sum())
        entries.append({
            "name": str(name),
            "support_parent_px": float(supports[index]),
            "is_aggregate": float(supports[index]) > 1.0,
            "present_count": present_count,
            "absent_count": total_rows - present_count,
            "presence": (None if present is None
                         else [bool(value) for value in channel_present[:shown]]),
            # JSON has no NaN. `presence` distinguishes absent from observed-but-invalid;
            # both render as a gap, but only the former reduces effective N.
            "values": [float(value) if np.isfinite(value) else None
                       for value in values[:shown, index]],
        })

    facts = clock_facts(series.times_seconds)
    return {
        "source_name": series.provenance.get("path_basename"),
        "content_sha256": series.provenance.get("content_sha256"),
        "domain": domain,
        "onboarding_sha256": DOMAIN_ONBOARDINGS.get(domain).onboarding_sha256
        if is_onboarded(domain) else None,
        "n_rows": total_rows,
        "preview_rows": shown,
        "rows_withheld": total_rows - shown,
        "preview_note": (
            "A plot of a record is not an analysis of it. Nothing has been mined, no claim "
            "exists, and no rung has moved (R22)."),
        "clock": facts,
        "times_seconds": [float(t) for t in series.times_seconds[:shown]],
        "channels": entries,
        "domain_limits": {
            "declared": declaration.describe(),
            "precedence_admissible": declaration.precedence_admissible,
            "refuses": [dict(item) for item in refusals_for(declaration_for(domain))],
            "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
        },
        "provenance": dict(series.provenance),
    }


__all__ = ["router", "PREVIEW_ROWS", "REGISTERED_DOMAINS"]
