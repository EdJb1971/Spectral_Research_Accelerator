"""Read-only HTTP transport for the identity declaration and its audits (T4E.8 slice 4).

**Why this router exists.** T4E.8 slice 3 made the identity target a declared, refusable input:
what identity is meant to recognise -- continuity of a tracked constellation, persistence of a
spatial configuration, or recurrence of the same physical kind -- is the most consequential
scientific choice in the atmospheric sequence, and choosing it wrongly is how a programme
validates a definition against itself. Until now that choice was made by hand-editing a JSON
design, and nothing in `src/api` or `frontend/src` could read an identity-calibration receipt at
all. A choice a researcher cannot see is a choice they cannot make knowingly.

**Refusals rank equal to values here.** `/targets` publishes the whole admissibility matrix: for
every target and evidence class it states whether the pairing is admitted and, when it is not,
the refusal in full. The most important cell in that matrix is a refusal -- `kind_recurrence`
against record-derived proxy labels -- and a surface that showed only the workable combinations
would hide the one thing a researcher most needs to understand. Caveats travel the same way: a
pairing can be admitted and still owe the reader a warning, and `track_continuity` on tracked
keys does.

**What it deliberately does not do.** It serves GET and nothing else. There is no route that
chooses a target, writes a design, runs an audit or approves a radius. Choosing is a scientific
act and code does not perform it for a person; running an audit reads a multi-gigabyte record
and belongs on the command line. Every response carries `network_used: false` because no path
here can reach a network.

**Unreadable receipts are reported, never skipped.** A store that silently drops the file it
could not parse shows a shorter list that looks complete, and the missing entry is the one whose
integrity is in question.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from src.analysis_engine.spectral_identity_audit import (
    EVIDENCE_CLASSES, IDENTITY_TARGETS, declare_identity_target,
)
from src.core.errors import SpectralEarthError

router = APIRouter(prefix="/api/v1/identity", tags=["identity"])

#: What this surface refuses to do, published rather than merely implemented.
REFUSALS = (
    "No route chooses an identity target. Which question identity answers is a scientific "
    "declaration and code does not make it for a person.",
    "No route writes or amends an audit design, and no route approves a mining radius.",
    "No route runs an audit: it reads the acquired record and belongs on the command line.",
    "No route can reach a network, so nothing served here is a live measurement.",
)


def _audit_dir(request: Request) -> Path:
    value = getattr(request.app.state, "identity_audit_dir", None)
    return Path(value if value is not None else
                os.getenv("IDENTITY_AUDIT_DIR", "data/identity_calibration"))


def _admissibility() -> List[Dict[str, Any]]:
    """Every target against every evidence class, admitted or refused, with the reason.

    Built by asking `declare_identity_target` rather than by re-reading the registry, so this
    matrix cannot drift away from the rule the audit actually enforces.
    """
    matrix: List[Dict[str, Any]] = []
    for target in IDENTITY_TARGETS.names():
        specification = IDENTITY_TARGETS.get(target)
        row: Dict[str, Any] = {
            "identity_target": target,
            "recognises": specification.recognises,
            "does_not_license": specification.does_not_license,
            "evidence": [],
        }
        for evidence in EVIDENCE_CLASSES.names():
            try:
                declaration = declare_identity_target(target, evidence)
            except SpectralEarthError as refusal:
                row["evidence"].append({
                    "evidence_class": evidence, "admitted": False,
                    "refusal": str(refusal),
                    "independent_of_record":
                        EVIDENCE_CLASSES.get(evidence).independent_of_record,
                })
                continue
            row["evidence"].append({
                "evidence_class": evidence, "admitted": True, "refusal": None,
                "caveat": declaration["caveat"],
                "label_boundary": declaration["label_boundary"],
                "independent_of_record": declaration["evidence_independent_of_record"],
                "evidence_provenance": declaration["evidence_provenance"],
            })
        matrix.append(row)
    return matrix


@router.get("/targets")
def read_targets() -> Dict[str, Any]:
    """The declared identity targets, and which evidence may validate each one."""
    return {
        "targets": _admissibility(),
        "evidence_classes": [
            {"evidence_class": name,
             "provenance": EVIDENCE_CLASSES.get(name).provenance,
             "independent_of_record": EVIDENCE_CLASSES.get(name).independent_of_record,
             "label_boundary": EVIDENCE_CLASSES.get(name).label_boundary}
            for name in EVIDENCE_CLASSES.names()],
        "choosing_is_not_automated": (
            "This surface states what each target claims and what evidence may validate it. "
            "It does not choose, and no default is supplied."),
        "refusals": list(REFUSALS),
        "network_used": False,
    }


def _summarise(path: Path, body: Dict[str, Any]) -> Dict[str, Any]:
    """One receipt reduced to what a reader must see, boundaries and verdict included."""
    windows = body.get("windows") or []
    return {
        "file": path.name,
        "schema": body.get("schema"),
        "status": body.get("status"),
        "approved_mining_radius": body.get("approved_mining_radius"),
        "frozen_radius": body.get("frozen_radius"),
        "identity_declaration": body.get("identity_declaration"),
        "code_revision": body.get("code_revision"),
        "code_dirty": body.get("code_dirty"),
        "design_sha256": body.get("design_sha256"),
        "windows": len(windows),
        "claim_boundary": body.get("claim_boundary"),
    }


def _load(directory: Path):
    readable: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, str]] = []
    if not directory.is_dir():
        return readable, unreadable
    for path in sorted(directory.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            unreadable.append({"file": path.name, "error": str(exc)})
            continue
        if not isinstance(body, dict):
            unreadable.append({"file": path.name,
                               "error": "top level is %s, not an object" % type(body).__name__})
            continue
        readable.append(_summarise(path, body))
    return readable, unreadable


@router.get("/audits")
def read_audits(request: Request) -> Dict[str, Any]:
    """Every identity-calibration receipt in the store, with what could not be read."""
    readable, unreadable = _load(_audit_dir(request))
    undeclared = [item["file"] for item in readable if not item.get("identity_declaration")]
    return {
        "audits": readable,
        "unreadable": unreadable,
        "audits_without_a_declared_target": undeclared,
        "undeclared_note": (
            "Receipts written before T4E.8 slice 3 carry no identity declaration. They are "
            "shown because hiding them would make the store look uniformly declared."),
        "refusals": list(REFUSALS),
        "network_used": False,
    }


@router.get("/audits/{name}")
def read_audit(request: Request, name: str) -> Dict[str, Any]:
    """One receipt in full. The name is a file name in the store and never a path."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400,
                            detail="audit name must be a file name in the store, not a path")
    path = _audit_dir(request) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no identity audit named %r" % name)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422,
                            detail="identity audit %r could not be read: %s" % (name, exc))
    return {"audit": body, "summary": _summarise(path, body),
            "refusals": list(REFUSALS), "network_used": False}
