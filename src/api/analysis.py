"""Read-only HTTP boundary for the cross-domain analysis engine (TG11.1).

The browser retains the original file after TG11.0 because ``/channels/read`` returns only a
bounded preview.  This route deliberately reads that file again under the same named domain: an
analysis of a preview would be an analysis of a truncated record, while a server-side record
cache would create a second, staleable source of truth.

Nothing in this module implements an estimator, correction, lag floor or replication verdict.
Those remain in :mod:`src.analysis_engine.domain_analysis`; this is only the wire boundary that
turns explicit multipart fields into its three operations.  It stores nothing and cannot accept
a study rung, claim level or confidence from a client (R22).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.analysis_engine.domain_analysis import (analyse_precedence, association_only,
                                                  run_domain_gate)
from src.analysis_engine.cross_scale import GateProtocol
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.preregistration import HeldOutAlreadyOpenedError
from src.data_layer.tabular_source import VALUE_MEASURE, read_channels_for_domain
from src.api.preregistration import held_out_identity_for


router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

OPERATIONS = {
    "association": (
        "Test the declared lag family as association only. Available only when the domain "
        "cannot justify precedence; the result carries the R21 warning."),
    "precedence": (
        "Test the declared lag family as precedence. Refused before computation unless the "
        "domain's registered lag policy supplies an admissible floor (R21)."),
    "domain_gate": (
        "Split the full record with an embargo, run the same frozen family independently on "
        "train and test, and return PASS, FAIL or INVALID."),
}

COMMON_KEYS = {
    "lags", "estimator", "bins", "n_surrogates", "alpha", "correction", "seed",
    "lag_policy_params",
}
GATE_KEYS = COMMON_KEYS | {
    "study_id", "train_ratio", "embargo_frames", "require_advection_floor",
}


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=("%r is not UTF-8 text. Cross-domain analysis currently accepts admitted "
                    "channel tables; inspect a gridded file through its acquisition adapter."
                    % filename))


def _configuration(raw: str, operation: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`configuration` must be a JSON object.")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="`configuration` must be a JSON object.")
    allowed = GATE_KEYS if operation == "domain_gate" else COMMON_KEYS
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InvalidParameterError(
            "configuration keys", unknown,
            "only %s for operation %r. Unknown settings are refused rather than ignored, "
            "because an ignored family setting would make the receipt describe a different "
            "analysis" % (sorted(allowed), operation))
    return value


def _common(config: Mapping[str, Any]) -> Dict[str, Any]:
    lags = config.get("lags", [1])
    if not isinstance(lags, list):
        raise InvalidParameterError("lags", lags, "a JSON array of positive frame lags")
    policy = config.get("lag_policy_params", {})
    if not isinstance(policy, dict):
        raise InvalidParameterError("lag_policy_params", policy, "a JSON object")
    return {
        "lags": lags,
        "estimator": config.get("estimator", "mutual_information"),
        "bins": config.get("bins", 4),
        "n_surrogates": config.get("n_surrogates", 499),
        "alpha": config.get("alpha", 0.05),
        "correction": config.get("correction", "benjamini_yekutieli"),
        "seed": config.get("seed", 20260827),
        "lag_policy_params": policy,
    }


def _refuse_a_spent_partition(series: Any, *, train_ratio: Any, embargo_frames: Any) -> None:
    """Read the held-out ledger before the gate opens the partition the gate would create."""
    identity, record = held_out_identity_for(series, train_ratio=float(train_ratio),
                                             embargo_frames=int(embargo_frames))
    if record is not None:
        raise HeldOutAlreadyOpenedError(record, "(unsealed gate run)")


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    # A spent partition is a state conflict rather than a malformed request: nothing the
    # caller could edit would make this call legitimate.
    status = 409 if isinstance(error, HeldOutAlreadyOpenedError) else info["status_code"]
    return HTTPException(status_code=status, detail=info["detail"])


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """Describe the reachable operations without implying that any one is admissible."""
    return {
        "operations": OPERATIONS,
        "input": "an admitted full channel-table file plus its explicit domain declaration",
        "measure": VALUE_MEASURE,
        "read_only": True,
        "stores": False,
        "moves_rung": False,
        "claim_boundary": (
            "This surface computes candidate analysis results. It records no evidence, moves "
            "no claim rung (R22), and does not preregister or spend held-out data; those are "
            "separate workflow steps beginning in TG11.2."),
    }


@router.post("/run")
async def run_analysis(
    file: UploadFile = File(...),
    operation: str = Form(...),
    domain: str = Form(...),
    time_column: str = Form(...),
    configuration: str = Form("{}"),
    time_units: str = Form("s"),
    delimiter: str = Form(","),
    support_parent_px: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Run one engine operation against the complete uploaded record, without writing state."""
    if operation not in OPERATIONS:
        raise HTTPException(
            status_code=404,
            detail="Unknown analysis operation %r. Available: %s."
                   % (operation, ", ".join(sorted(OPERATIONS))))
    filename = file.filename or "upload"
    text = _decode(await file.read(), filename)
    try:
        config = _configuration(configuration, operation)
        supports = json.loads(support_parent_px) if support_parent_px else None
        if supports is not None and not isinstance(supports, dict):
            raise InvalidParameterError("support_parent_px", supports, "a JSON object")
        series, declaration = read_channels_for_domain(
            text, source_name=filename, domain=domain, time_column=time_column,
            time_units=time_units, support_parent_px=supports, delimiter=delimiter)
        cadence = series.provenance.get("cadence_seconds")
        if not isinstance(cadence, (int, float)) or cadence <= 0:
            raise InvalidParameterError(
                "record clock", cadence,
                "a regular clock with a finite positive cadence. A lag in frames is not a "
                "physical duration on an irregular clock, so this surface will not invent one")

        common = _common(config)
        if operation == "association":
            result = association_only(series, declaration, cadence_seconds=float(cadence),
                                      measure=VALUE_MEASURE, **common)
        elif operation == "precedence":
            result = analyse_precedence(series, declaration, cadence_seconds=float(cadence),
                                        measure=VALUE_MEASURE, **common)
        else:
            policy_params = common.pop("lag_policy_params")
            # TG11.2. The gate splits internally and reports a verdict on its own test
            # partition, so running it is a test of that partition. If the partition has
            # already been spent under a seal, this is a second, uncorrected test of the same
            # held-out data - refused here rather than reported (R18). Nothing is *recorded*
            # by running the gate: the ledger is only read.
            _refuse_a_spent_partition(
                series, train_ratio=config.get("train_ratio", 0.6),
                embargo_frames=config.get("embargo_frames", 1))
            protocol = GateProtocol(
                study_id=str(config.get("study_id", "workbench-domain-gate")),
                n_scales=len(series.channels), lags=tuple(int(v) for v in common.pop("lags")),
                expected_frames=series.n_times, cadence_seconds=float(cadence),
                train_ratio=config.get("train_ratio", 0.6),
                embargo_frames=config.get("embargo_frames", 1),
                measure=VALUE_MEASURE,
                require_advection_floor=config.get("require_advection_floor", False),
                **common)
            result = run_domain_gate(series, declaration, protocol,
                                     lag_policy_params=policy_params)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Malformed analysis configuration: %s" % error)
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": "spectral.analysis.http.v1",
        "operation": operation,
        "source": {
            "name": series.provenance.get("path_basename"),
            "content_sha256": series.provenance.get("content_sha256"),
            "domain": declaration.name,
            "frames": series.n_times,
            "channels": list(series.channels),
            "cadence_seconds": float(cadence),
        },
        "read_only": True,
        "stored": False,
        "rung_moved": False,
        "result": result,
        "claim_boundary": (
            "This HTTP response is read-only compute over this exact record. It stores no "
            "evidence, moves no claim rung (R22), and is not a preregistration or a held-out "
            "confirmation."),
    }


__all__ = ["router", "OPERATIONS"]
