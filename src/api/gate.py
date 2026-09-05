"""Read-only HTTP transport for the T4C atmospheric gate line (T4C.5j).

**Why this router exists.** Every artifact in the T4C line -- the frozen campaign, the checked
supersession that retires one, the immutable receipt a run publishes -- was reachable only from
the command line and the filesystem. A reviewer had to know which file to open, and the two
distinctions the line exists to draw were the ones buried deepest: that a *retired* campaign is
still readable but must not be acquired, and that a FAIL is a negative finding only where the
derived spatial-power record shows the absence was detectable. Those are exactly what a reviewer
needs to *see*.

**What it deliberately does not do.** It serves GET and nothing else. There is no route that
writes a campaign, edits one, records a supersession, runs a preflight or starts an acquisition:
the preflight probes local storage and credential configuration, and acquisition spends a 2.8 GB
transfer, so neither belongs behind a browser button. Every response carries `network_used:
false` because none of these paths can reach a network at all.

**Retirement is derived, not declared.** A campaign is retired here if and only if some
supersession in the store names it *by content hash* -- not by identifier, and not by a flag in
the campaign file, which a frozen artifact could not carry without being edited. This mirrors
`preflight_gate_campaign`, which refuses on the same fingerprint comparison. The retired campaign
is still served in full, with its defect intact, because a defect that cannot be read against the
artifact it belongs to has not been recorded anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request

from src.analysis_engine.gate_campaign import (
    CampaignSupersession, GateCampaign, load_gate_campaign,
    load_gate_campaign_supersession, review_gate_campaign,
    review_gate_campaign_supersession)
from src.analysis_engine.gate_run import load_gate_receipt
from src.core.errors import SpectralEarthError

router = APIRouter(prefix="/api/v1/gate", tags=["gate"])

#: What this surface refuses to do, published rather than merely implemented. A reader who can
#: see a campaign and cannot see why there is no "acquire" button is entitled to the reason.
REFUSALS = (
    "No route here writes, edits or publishes any artifact; the store is read-only.",
    "No route runs a preflight: it probes local storage and credential configuration, which is "
    "a property of a machine rather than of the science, and belongs on the command line.",
    "No route starts an acquisition. The mandatory order is WeatherBench overlap, then CDS "
    "canary, then the full record, each gated on the previous one passing.",
    "No route can reach a network, so nothing served here can be a live measurement.",
)


def _campaign_dir(request: Request) -> Path:
    value = getattr(request.app.state, "gate_campaign_dir", None)
    return Path(value if value is not None else os.getenv("GATE_CAMPAIGN_DIR", "campaigns"))


def _receipt_dir(request: Request) -> Path:
    value = getattr(request.app.state, "gate_receipt_dir", None)
    return Path(value if value is not None else
                os.getenv("GATE_RECEIPT_DIR", "data/gate_receipts"))


def _scan(directory: Path) -> Tuple[Dict[str, Tuple[Path, GateCampaign]],
                                    Dict[str, Tuple[Path, CampaignSupersession]],
                                    List[Dict[str, str]]]:
    """Load every campaign and supersession in a directory, keeping what would not load.

    Unreadable files are returned rather than skipped. A store that silently drops the file it
    could not authenticate shows a reviewer a shorter list that looks complete, and the missing
    entry is precisely the one whose integrity is in question.
    """
    campaigns: Dict[str, Tuple[Path, GateCampaign]] = {}
    supersessions: Dict[str, Tuple[Path, CampaignSupersession]] = {}
    unreadable: List[Dict[str, str]] = []
    if not directory.is_dir():
        return campaigns, supersessions, unreadable
    for path in sorted(directory.glob("*.json")):
        try:
            record = load_gate_campaign_supersession(path)
            supersessions[record.supersession_id] = (path, record)
            continue
        except (SpectralEarthError, ValueError, TypeError, KeyError) as superseding:
            first = str(superseding)
        try:
            campaign = load_gate_campaign(path)
            campaigns[campaign.campaign_id] = (path, campaign)
        except (SpectralEarthError, ValueError, TypeError, KeyError) as exc:
            unreadable.append({"file": path.name, "as_campaign": str(exc),
                               "as_supersession": first})
    return campaigns, supersessions, unreadable


def _retirement(campaign: GateCampaign,
                supersessions: Dict[str, Tuple[Path, CampaignSupersession]]
                ) -> Optional[Dict[str, str]]:
    """The supersession that retires this campaign, matched on content rather than on name."""
    fingerprint = campaign.fingerprint()
    for identifier, (_, record) in sorted(supersessions.items()):
        if record.superseded.fingerprint() == fingerprint:
            return {"supersession_id": identifier,
                    "successor_campaign_id": record.successor.campaign_id,
                    "successor_campaign_sha256": record.successor.fingerprint(),
                    "acquisition": "REFUSED",
                    "statement": ("This design is retired. It is served in full, with its "
                                  "defect intact, because a defect has not been recorded "
                                  "anywhere if it cannot be read against the artifact it "
                                  "belongs to. preflight_gate_campaign refuses to spend on "
                                  "it when the supersession is supplied.")}
    return None


@router.get("")
async def gate_surface(request: Request) -> Dict[str, Any]:
    """What this surface holds and what it will not do."""
    campaigns, supersessions, unreadable = _scan(_campaign_dir(request))
    directory = _receipt_dir(request)
    receipts = sorted(directory.glob("*.json")) if directory.is_dir() else []
    return {
        "schema": "cross-scale-gate-surface/v1",
        "campaigns": len(campaigns),
        "supersessions": len(supersessions),
        "retired_campaigns": sum(
            1 for _, campaign in campaigns.values()
            if _retirement(campaign, supersessions) is not None),
        "receipts": len(receipts),
        "measurement_status": "MEASURED" if receipts else "NOT_YET_MEASURED",
        "unreadable": unreadable,
        "refusals": list(REFUSALS),
        "claim_boundary": (
            "A campaign is a preregistered design and a supersession is a record about two "
            "designs. Neither is a result. Only a receipt reports a run, and no receipt "
            "reports one that has not happened."),
        "network_used": False,
    }


@router.get("/campaigns")
async def list_campaigns(request: Request) -> Dict[str, Any]:
    campaigns, supersessions, unreadable = _scan(_campaign_dir(request))
    entries = []
    for identifier, (path, campaign) in sorted(campaigns.items()):
        retired = _retirement(campaign, supersessions)
        design = review_gate_campaign(campaign)["scientific_design"]
        entries.append({
            "campaign_id": identifier,
            "campaign_sha256": campaign.fingerprint(),
            "study_plan_sha256": campaign.gate_plan.fingerprint(),
            "file": path.name,
            "status": "RETIRED" if retired else "ACTIVE",
            "retired_by": retired,
            "variable": campaign.gate_plan.variable,
            "level_hpa": float(campaign.gate_plan.level_hpa),
            "date_start": campaign.full_acquisition.date_start,
            "date_end": campaign.full_acquisition.date_end,
            "expected_frames": design["full_frames"],
            "resolvable": design["resolvable"],
            "hypothesis_family_size": design["hypothesis_family_size"],
            # D86: a campaign that declares no criterion cannot be acquired, so the absence
            # belongs on the row rather than three clicks away inside the review.
            "overlap_criterion": (campaign.overlap_criterion.to_mapping()
                                  if campaign.overlap_criterion is not None else None),
        })
    return {"schema": "cross-scale-gate-campaign-index/v1", "campaigns": entries,
            "unreadable": unreadable, "network_used": False}


@router.get("/campaigns/{campaign_id}")
async def campaign_review(campaign_id: str, request: Request) -> Dict[str, Any]:
    campaigns, supersessions, _ = _scan(_campaign_dir(request))
    if campaign_id not in campaigns:
        raise HTTPException(status_code=404,
                            detail="no campaign %r in the store" % campaign_id)
    path, campaign = campaigns[campaign_id]
    review = review_gate_campaign(campaign)
    retired = _retirement(campaign, supersessions)
    return {**review, "file": path.name, "status": "RETIRED" if retired else "ACTIVE",
            "retired_by": retired, "refusals": list(REFUSALS)}


@router.get("/supersessions")
async def list_supersessions(request: Request) -> Dict[str, Any]:
    _, supersessions, unreadable = _scan(_campaign_dir(request))
    entries = []
    for identifier, (path, record) in sorted(supersessions.items()):
        entries.append({
            "supersession_id": identifier,
            "supersession_sha256": record.fingerprint(),
            "file": path.name,
            "superseded_campaign_id": record.superseded.campaign_id,
            "successor_campaign_id": record.successor.campaign_id,
            "defects": sorted({str(reason["defect"]) for reason in record.reasons}),
            "reason_count": len(record.reasons),
            "preserved_count": len(record.preserved),
            "deferred_to_run": [str(entry["defect"]) for entry in record.deferred_to_run],
        })
    return {"schema": "cross-scale-gate-supersession-index/v1", "supersessions": entries,
            "unreadable": unreadable, "network_used": False}


@router.get("/supersessions/{supersession_id}")
async def supersession_review(supersession_id: str, request: Request) -> Dict[str, Any]:
    """Re-run every stated reason against both campaigns and publish the outcomes side by side.

    The review is computed here rather than read from the file, so what a reader sees is the
    checks passing now against the two campaigns as stored -- not an assertion the record made
    about itself when it was written.
    """
    _, supersessions, _ = _scan(_campaign_dir(request))
    if supersession_id not in supersessions:
        raise HTTPException(status_code=404,
                            detail="no supersession %r in the store" % supersession_id)
    path, record = supersessions[supersession_id]
    try:
        review = review_gate_campaign_supersession(record)
    except SpectralEarthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {**review, "file": path.name,
            "superseded_campaign_id": record.superseded.campaign_id,
            "successor_campaign_id": record.successor.campaign_id,
            "refusals": list(REFUSALS)}


@router.get("/receipts")
async def list_receipts(request: Request) -> Dict[str, Any]:
    """Every published gate receipt, and an explicit statement when there are none.

    An empty list rendered without this reads as "no relationship was found". The distinction
    between an absence of findings and an absence of runs is the first thing a reviewer of this
    line needs, so the status says which one this is.
    """
    directory = _receipt_dir(request)
    entries: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, str]] = []
    for path in (sorted(directory.glob("*.json")) if directory.is_dir() else []):
        try:
            receipt = load_gate_receipt(path)
        except (SpectralEarthError, ValueError, TypeError, KeyError) as exc:
            unreadable.append({"file": path.name, "error": str(exc)})
            continue
        entries.append({
            "receipt_id": path.stem,
            "receipt_sha256": receipt["receipt_sha256"],
            "plan_sha256": receipt["plan_sha256"],
            "study_id": receipt["plan"].get("study_id"),
            "evidence_role": receipt["plan"].get("evidence_role"),
            "gate_verdict": receipt["gate"]["verdict"],
            "scientific_verdict": receipt["scientific_verdict"],
            "power_applied": receipt["power_adjudication"]["power_applied"],
        })
    return {
        "schema": "cross-scale-gate-receipt-index/v1",
        "receipts": entries, "unreadable": unreadable,
        "status": "MEASURED" if entries else "NOT_YET_MEASURED",
        "statement": ("Each entry reports one completed run." if entries else
                      "No gate has run, so there is no verdict here of any kind. This is an "
                      "absence of runs, not an absence of findings."),
        "network_used": False,
    }


@router.get("/receipts/{receipt_id}")
async def receipt_detail(receipt_id: str, request: Request) -> Dict[str, Any]:
    """One receipt, with the gate's verdict and the scientific verdict shown side by side.

    They are not the same thing and the receipt does not pretend they are. The gate reports what
    the replication rule returned; the scientific verdict is what a reviewer should read, and
    where the two differ it is the derived spatial-power record that moved it. Serving one
    without the other would hide the FAIL/INVALID boundary this line exists to draw.
    """
    path = _receipt_dir(request) / ("%s.json" % receipt_id)
    if ".." in receipt_id or "/" in receipt_id or "\\" in receipt_id or not path.is_file():
        raise HTTPException(status_code=404, detail="no gate receipt %r" % receipt_id)
    try:
        receipt = load_gate_receipt(path)
    except (SpectralEarthError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"schema": "cross-scale-gate-receipt-view/v1", "receipt_id": receipt_id,
            "integrity": "VERIFIED", "receipt": receipt,
            "gate_verdict": receipt["gate"]["verdict"],
            "scientific_verdict": receipt["scientific_verdict"],
            "power_adjudication": receipt["power_adjudication"],
            "claim_boundary": receipt["claim_boundary"], "network_used": False}
