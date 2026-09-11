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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from src.analysis_engine.spectral_identity_audit import (
    EVIDENCE_CLASSES, IDENTITY_TARGETS, declare_identity_target,
)
from src.analysis_engine.position_tolerance import (
    DECLARED_GRID_KM_PER_CELL, DECLARED_LOCALISATION_CELLS, localisation_km, tolerance_for,
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


def _measurement_dir(request: Request) -> Path:
    value = getattr(request.app.state, "measurement_dir", None)
    return Path(value if value is not None else
                os.getenv("MEASUREMENT_DIR", "measurements"))


#: The several names this programme has used for "what this result may not be used for". A
#: reader must never be shown a number without it, and a surface that recognised only one
#: spelling would silently drop the boundary from most records. Collected rather than
#: normalised, because renaming the keys in forty committed records to suit a viewer would
#: rewrite evidence to fit its display.
BOUNDARY_KEYS = (
    "claim_boundary",
    # `boundary` and `acceptance_boundary` were missed on the first pass, and the panel told a
    # reader that seven records "predate the convention" when the clause was right there under
    # the plainest name of all. A viewer that under-reports a boundary is worse than one that
    # omits the field: it makes a false statement about the evidence, on screen.
    "boundary",
    "acceptance_boundary",
    "is_a_diagnostic_not_a_criterion",
    "what_this_does_not_settle",
    "what_this_slice_does_not_settle",
    "what_it_does_NOT_license",
    "what_this_does_NOT_license",
    "what_this_does_NOT_do",
    "what_this_signature_does_NOT_do",
    "what_this_slice_will_not_do",
    "what_this_acquisition_does_NOT_authorise",
    "what_a_failure_would_and_would_not_license",
    "what_a_success_would_and_would_not_license",
    "what_this_still_does_not_establish",
    "what_was_NOT_done",
    "what_this_evidence_does_not_supply",
    "what_this_evidence_still_does_not_supply",
)

#: A record that has been corrected or superseded and does not say so in its summary is the
#: dangerous case: it reads as current. These are the marks this programme leaves when it
#: supersedes something in the open rather than editing it away.
CORRECTION_KEYS = ("CORRECTION_2026_09_10", "GATE_CORRECTION", "withdrawal",
                   "superseded_signature", "superseded_figures", "amendment",
                   "the_first_diagnosis_was_wrong")


def _boundaries(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every "does not license" clause a record carries, under whichever name it uses."""
    found: List[Dict[str, Any]] = []
    for key in BOUNDARY_KEYS:
        if key in body and body[key]:
            found.append({"key": key, "text": body[key]})
    return found


#: An amendment is recognised by what its key SAYS, not by a prefix. T4E.28 appended
#: `CONDITION_2_LIFTED_2026_09_11_BY_T4E_28` to the T4E.27 receipt -- a refusal lifted by new
#: evidence, which is exactly the kind of change a reader must not miss -- and the panel reported
#: that record as never amended, because the key did not begin with `CORRECTION`. That is the
#: same failure as the under-reported boundary above, and the same principle applies: a viewer
#: that under-reports an amendment makes a false statement about the evidence, on screen.
#:
#: Matched as whole words against the key's tokens, so `what_would_lift_the_refusal` -- a clause
#: describing a refusal that still STANDS -- is not mistaken for one that has been lifted.
AMENDMENT_WORDS = frozenset((
    "correction", "corrections", "corrected", "amendment", "amended", "amend",
    "withdrawn", "withdrawal", "superseded", "supersedes", "falsified", "lifted",
    "retracted",
))
#: `restated` is deliberately ABSENT. T4E.27's own `condition_1_restated` and
#: `condition_2_restated` are the measurement's results, not marks that it was amended, and
#: listing them as amendments would be the opposite error: a record that reports itself corrected
#: when nothing about it has changed.


def _corrections(body: Dict[str, Any]) -> List[str]:
    """Marks that this record was corrected, lifted or superseded in the open."""
    marks = [key for key in CORRECTION_KEYS if key in body and body[key]]
    for key in body:
        if not body[key]:
            continue
        tokens = {token for token in re.split(r"[^A-Za-z]+", key.lower()) if token}
        if tokens & AMENDMENT_WORDS:
            marks.append(key)
    return sorted(set(marks))


def _verdict(body: Dict[str, Any]) -> Optional[Any]:
    for key in ("VERDICT", "verdict", "status", "outcome"):
        if key in body and body[key]:
            return body[key]
    return None


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
        # T4E.22. A summary that could not hold a verdict showed a schema and a status for
        # records whose entire content was a result, and a reader saw nulls where the finding
        # was. PLAN section 5 asks for the opposite: no view states a number without what it
        # may not be used for.
        "verdict": _verdict(body),
        "boundaries": _boundaries(body),
        "corrected_or_superseded": _corrections(body),
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


# ---------------------------------------------------------------------------------------------
# T4E.22: the measurements, and the chain that joins a question to its answer.
#
# Until now this router served `data/identity_calibration` and nothing else, so a reader could
# see that a study had been DECLARED and never what it MEASURED: the offsets, the falsified
# predictions, the corrected diagnosis all lived in `measurements/` and in git. For a research
# tool that is the wrong half. A declaration without its result is a promise; a result without
# its declaration is an assertion; only the pair is evidence.


def _study_key(name: str) -> Optional[str]:
    """The task a file belongs to -- `t4e19` from either naming convention, or None.

    Declarations are named `t4e19-positional-error-declaration.json` and measurements
    `t4e19_positional_error.json`, so the key is the leading token under either separator.
    """
    stem = (name[:-5] if name.lower().endswith(".json") else name).lower()
    match = re.match(r"^(t4[a-z]\d+)", stem)
    return match.group(1) if match else None


@router.get("/measurements")
def read_measurements(request: Request) -> Dict[str, Any]:
    """Every measurement record, with its verdict and what it may not be used for.

    A measurement is served with its boundaries attached rather than beside them, because the
    boundary is the part a reader is most likely to skip and the part this programme most often
    found itself needing.
    """
    readable, unreadable = _load(_measurement_dir(request))
    corrected = [item["file"] for item in readable if item.get("corrected_or_superseded")]
    unbounded = [item["file"] for item in readable if not item.get("boundaries")]
    return {
        "measurements": readable,
        "unreadable": unreadable,
        "corrected_or_superseded": corrected,
        "correction_note": (
            "These records were corrected or superseded in the open rather than edited away. "
            "A corrected record that reads as current is the dangerous case, so the mark is "
            "carried in the summary and not only in the body."),
        "measurements_without_a_stated_boundary": unbounded,
        "boundary_note": (
            "A measurement with no clause saying what it may not be used for is listed here "
            "rather than passed over. Some predate the convention; none are hidden."),
        "refusals": list(REFUSALS),
        "network_used": False,
    }


@router.get("/measurements/{name}")
def read_measurement(request: Request, name: str) -> Dict[str, Any]:
    """One measurement in full. The name is a file name in the store and never a path."""
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400,
                            detail="measurement name must be a file name, not a path")
    path = _measurement_dir(request) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no measurement named %r" % name)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422,
                            detail="measurement %r could not be read: %s" % (name, exc))
    return {"measurement": body, "summary": _summarise(path, body),
            "refusals": list(REFUSALS), "network_used": False}


@router.get("/studies")
def read_studies(request: Request) -> Dict[str, Any]:
    """Each task as a chain: what was declared, what was signed or adopted, what was measured.

    This is the view the work itself asks for. Every study in this programme runs
    declaration -> adoption -> measurement -> outcome, and a surface that showed the three
    separately would leave a reader to reconstruct by filename which result answered which
    question, and whether the question was fixed before the answer was known.
    """
    audits, audit_unreadable = _load(_audit_dir(request))
    measurements, measurement_unreadable = _load(_measurement_dir(request))
    studies: Dict[str, Dict[str, Any]] = {}
    unfiled: List[str] = []
    for item, role in ([(a, "declaration") for a in audits]
                       + [(m, "measurement") for m in measurements]):
        key = _study_key(item["file"])
        if key is None:
            unfiled.append(item["file"])
            continue
        study = studies.setdefault(key, {"task": key.upper(), "declarations": [],
                                         "measurements": []})
        name = item["file"].lower()
        if role == "declaration" and ("adoption" in name or "signature" in name
                                      or "amendment" in name):
            study.setdefault("adoptions", []).append(item)
        elif role == "declaration":
            study["declarations"].append(item)
        else:
            study["measurements"].append(item)
    for study in studies.values():
        study["has_a_result"] = bool(study["measurements"])
        study["declared_before_measured"] = bool(study["declarations"]) and study["has_a_result"]
        study["corrected_or_superseded"] = sorted({
            mark for group in ("declarations", "adoptions", "measurements")
            for item in study.get(group, []) for mark in item.get("corrected_or_superseded", [])})
    declared_only = sorted(k for k, v in studies.items() if not v["has_a_result"])
    return {
        "studies": [studies[key] for key in sorted(studies)],
        "declared_but_not_measured": declared_only,
        "declared_but_not_measured_note": (
            "A declaration with no measurement is a question that has been fixed and not yet "
            "answered. That is a legitimate state in this programme -- several were declared "
            "and deliberately left unmeasured -- and it is shown rather than filtered."),
        "files_outside_any_study": sorted(unfiled),
        "unreadable": audit_unreadable + measurement_unreadable,
        "refusals": list(REFUSALS),
        "network_used": False,
    }


# ---------------------------------------------------------------- TG19.2: the join's own bar
#
# T4E.27 built a tolerance that publishes its parts. It was reachable only by importing a Python
# module, which means a researcher could read what bar this programme used and could not see what
# theirs would be. A bar that can only be accepted or rejected is not an instrument; one whose
# components, provenance and refusals are on the wire can be disagreed with specifically, and
# disagreeing specifically is how somebody else's question gets asked with the same tool.
#
# These routes COMPUTE a tolerance. They do not decide an acceptance, and there is no route that
# writes one, stores one, or approves a join.


@router.get("/tolerance/components")
def read_tolerance_components() -> Dict[str, Any]:
    """What a position tolerance is made of, before any observation is supplied.

    Served without parameters so the contract can be read first and applied second. The excluded
    component is published beside the included ones, because what a bar leaves out determines
    what its residual means.
    """
    example = tolerance_for("<none supplied>", 1.0)
    return {
        "components": [
            {"name": name, "source": source,
             "supplied_by": ("the caller, per observation" if name == "catalogue_uncertainty"
                             else "this instrument, measured")}
            for name, _value, source in example.components],
        "combined_by": (
            "quadrature. These are independent contributions to a separation; a plain sum would "
            "double-count and a maximum would discard the others entirely."),
        "estimator_localisation_km": localisation_km(),
        "estimator_localisation_cells": DECLARED_LOCALISATION_CELLS,
        "grid_km_per_cell": DECLARED_GRID_KM_PER_CELL,
        "excluded": example.describe()["what_is_not_included"],
        "why_a_missing_uncertainty_is_refused": (
            "A catalogue that reports no uncertainty arrives as 0.0. Used as a bar it demands a "
            "separation of exactly zero, which no measurement can supply, so an observation "
            "carrying it could never pass however good the instrument was. It is refused by "
            "name and leaves the denominator instead of scoring as a failed detection."),
        "refusals": list(REFUSALS),
        "network_used": False,
    }


@router.get("/tolerance")
def compute_tolerance(catalogue_radius_km: Optional[float] = None,
                      separation_km: Optional[float] = None,
                      observation: str = "unnamed") -> Dict[str, Any]:
    """One observation's tolerance, with every part of it shown.

    `separation_km` is optional. Supplied, the response also says whether the separation is
    admitted and what the justified components fail to explain -- which is the measure of the
    component this bar deliberately omits. Omitted, only the bar is returned, so a caller can
    see what they would be judged against before judging anything.
    """
    tolerance = tolerance_for(observation, catalogue_radius_km)
    payload: Dict[str, Any] = dict(tolerance.describe())
    payload["network_used"] = False
    payload["refusals"] = list(REFUSALS)
    if separation_km is None:
        payload["separation_km"] = None
        payload["admitted"] = None
        payload["unexplained_residual_km"] = None
        payload["no_separation_supplied"] = (
            "The bar is returned without a verdict, because none was asked for.")
        return payload

    admitted = tolerance.admits(float(separation_km))
    payload.update({
        "separation_km": float(separation_km),
        "admitted": admitted,
        "unexplained_residual_km": tolerance.unexplained_residual(float(separation_km)),
    })
    if admitted is None:
        payload["why_no_verdict"] = (
            "The tolerance was refused, so this separation is not judged. That is not a "
            "failure: `we could not say` and `no` are different answers, and reporting this as "
            "a miss would count a missing catalogue uncertainty against the instrument.")
    return payload
