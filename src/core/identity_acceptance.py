"""T4E.41: T4E.8's acceptance, made decidable.

T4E.8's acceptance has been a sentence of prose while eight criteria were declared and
falsified around it. This module turns that sentence into eight conditions whose state can be
computed, and -- more importantly -- it fixes who may say a condition is met. Code may read a
register entry, recompute the digest of the measurement it names and check that a named person's
signature still reaches it. Code may not decide that the evidence is good enough, and it may not
emit `T4E8_ACCEPTED` under any combination of inputs: the furthest state reachable here is
`ALL_CONDITIONS_MET_AWAITING_HUMAN_ACCEPTANCE`.

The asymmetry is deliberate. Every falsification in this sequence was recorded because a
declaration was fixed before the numbers appeared; an acceptance that software could award
itself would be the one place in the programme where that ordering is reversed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.core.adoption import adoption_state
from src.core.errors import UserInputError


SCHEMA = "t4e41-identity-acceptance-declaration/v1"
REGISTER_SCHEMA = "t4e41-condition-evidence/v1"

DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e41-identity-acceptance-declaration.json")
DEFAULT_REGISTER_DIR = Path("data/identity_calibration/t4e41-condition-evidence")

CONDITION_IDS: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")

NO_EVIDENCE = "NO_EVIDENCE"
EVIDENCE_CLAIMED_NOT_ADOPTED = "EVIDENCE_CLAIMED_NOT_ADOPTED"
CONDITION_MET = "CONDITION_MET"
CONDITION_NOT_MET = "CONDITION_NOT_MET"

NOT_ACCEPTED = "T4E8_NOT_ACCEPTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
AWAITING_HUMAN_ACCEPTANCE = "ALL_CONDITIONS_MET_AWAITING_HUMAN_ACCEPTANCE"

#: The verdict this module exists to be incapable of producing.
FORBIDDEN_COMPUTED_VERDICT = "T4E8_ACCEPTED"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, label: str) -> Dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("%s is unreadable: %s" % (label, error)) from error
    if not isinstance(body, dict):
        raise UserInputError("%s does not contain a JSON object" % label)
    return body


def load_acceptance_declaration(
    path: Union[str, os.PathLike] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    """Load the acceptance and refuse a declaration whose bar has drifted.

    The structural checks are narrow on purpose. They do not police the wording of a
    condition -- that is the maintainer's -- but they do refuse a declaration that has lost a
    condition, gained one, or acquired a verdict semantics under which code could accept.
    """
    path = Path(path)
    body = _read(path, "T4E.41 acceptance declaration")
    if body.get("schema") != SCHEMA or body.get("task") != "T4E.41":
        raise UserInputError("unsupported T4E.41 acceptance declaration")

    conditions = body.get("acceptance_conditions")
    if not isinstance(conditions, list) or len(conditions) != len(CONDITION_IDS):
        raise UserInputError(
            "T4E.41 acceptance must carry exactly %d conditions" % len(CONDITION_IDS))
    if tuple(str(item.get("id")) for item in conditions) != CONDITION_IDS:
        raise UserInputError("T4E.41 acceptance conditions are not C1..C8 in order")
    for item in conditions:
        for field in ("name", "requires", "licensed_by", "insufficient"):
            if not str(item.get(field, "")).strip():
                raise UserInputError(
                    "T4E.41 condition %s lacks %s" % (item.get("id"), field))

    semantics = body.get("verdict_semantics", {})
    for verdict in (FORBIDDEN_COMPUTED_VERDICT, NOT_ACCEPTED, INSUFFICIENT_EVIDENCE,
                    AWAITING_HUMAN_ACCEPTANCE):
        if not str(semantics.get(verdict, "")).strip():
            raise UserInputError("T4E.41 verdict semantics lack %s" % verdict)
    if "never emit this verdict" not in str(semantics.get(FORBIDDEN_COMPUTED_VERDICT)):
        raise UserInputError(
            "T4E.41 acceptance must forbid code from emitting %s" % FORBIDDEN_COMPUTED_VERDICT)
    if not str(body.get("claim_boundary", "")).strip():
        raise UserInputError("T4E.41 acceptance lacks a claim boundary")
    return body


def verify_bound_evidence(
    declaration: Mapping[str, Any], *, root: Union[str, os.PathLike] = ".",
) -> Dict[str, Any]:
    """Recompute the digest of every artefact the bar was set against.

    A bar set against a record that has since changed was set against nothing, so drift is
    reported rather than tolerated -- but it is reported, not raised: a missing artefact is a
    fact about this checkout, and refusing to describe the declaration because of it would
    hide the bar as well as the drift.
    """
    root = Path(root)
    bound = declaration.get("bound_evidence", {})
    checks: List[Dict[str, Any]] = []
    for key, expected in sorted(bound.items()):
        if not key.endswith("_sha256"):
            continue
        # Two spellings are in use across this programme's receipts: `<name>_sha256` and
        # `<name>_file_sha256`. Both point at the same artefact and both must be checked.
        target = None
        for suffix in ("_file_sha256", "_sha256"):
            if key.endswith(suffix):
                candidate = bound.get(key[: -len(suffix)])
                if isinstance(candidate, str):
                    target = candidate
                    break
        if not isinstance(target, str) or not isinstance(expected, str):
            continue
        path = root / target
        if not path.exists():
            checks.append({"artefact": target, "status": "ABSENT",
                           "expected_sha256": expected, "observed_sha256": None})
            continue
        observed = _file_sha256(path)
        checks.append({
            "artefact": target,
            "status": "VERIFIED" if observed == expected else "DRIFTED",
            "expected_sha256": expected, "observed_sha256": observed,
        })
    drifted = [item["artefact"] for item in checks if item["status"] == "DRIFTED"]
    absent = [item["artefact"] for item in checks if item["status"] == "ABSENT"]
    return {
        "checked": len(checks), "checks": checks,
        "all_verified": not drifted and not absent,
        "drifted": drifted, "absent": absent,
    }


@dataclass(frozen=True)
class ConditionEvidence:
    """One claim that a named measurement addresses one condition."""

    condition: str
    measurement: str
    measurement_sha256: str
    addresses: str
    register_file: str
    register_sha256: str
    adopted: bool
    adopted_by: Optional[str]
    adopted_on: Optional[str]
    signature_current: bool
    measurement_present: bool
    measurement_digest_matches: bool
    outcome: Optional[str]


def _load_register_entry(path: Path, declaration_sha256: str,
                         root: Path) -> ConditionEvidence:
    body = _read(path, "T4E.41 condition evidence")
    if body.get("schema") != REGISTER_SCHEMA:
        raise UserInputError("%s is not a %s entry" % (path.name, REGISTER_SCHEMA))
    condition = str(body.get("condition", ""))
    if condition not in CONDITION_IDS:
        raise UserInputError("%s names condition %r, which is not C1..C8" % (
            path.name, condition))
    if body.get("acceptance_declaration_sha256") != declaration_sha256:
        raise UserInputError(
            "%s binds a different acceptance declaration than the one in force" % path.name)
    for field in ("measurement", "measurement_sha256", "addresses"):
        if not str(body.get(field, "")).strip():
            raise UserInputError("%s lacks %s" % (path.name, field))

    measurement = root / str(body["measurement"])
    present = measurement.exists()
    matches = present and _file_sha256(measurement) == body["measurement_sha256"]
    state = adoption_state(path.parent, path.name)
    return ConditionEvidence(
        condition=condition,
        measurement=str(body["measurement"]),
        measurement_sha256=str(body["measurement_sha256"]),
        addresses=str(body["addresses"]),
        register_file=path.name,
        register_sha256=_file_sha256(path),
        adopted=bool(state.get("adopted")),
        adopted_by=state.get("adopted_by"),
        adopted_on=state.get("adopted_on"),
        signature_current=bool(state.get("signature_still_reaches_the_declaration")),
        measurement_present=present,
        measurement_digest_matches=bool(matches),
        outcome=body.get("outcome"),
    )


def _condition_status(entries: Sequence[ConditionEvidence]) -> Tuple[str, Optional[str]]:
    """Decide one condition's state, and say which fact decided it.

    A claim is not evidence, an adopted claim whose measurement has changed is not evidence,
    and an adopted claim whose own signature no longer reaches it is not evidence. Only the
    remaining case counts, and a registered failure counts against the condition rather than
    being quietly dropped.
    """
    if not entries:
        return NO_EVIDENCE, None
    reasons: List[str] = []
    met = False
    for entry in entries:
        if not entry.measurement_present:
            reasons.append("%s names a measurement that is absent" % entry.register_file)
        elif not entry.measurement_digest_matches:
            reasons.append("%s names a measurement whose bytes have changed" % entry.register_file)
        elif not entry.adopted:
            reasons.append("%s is claimed but not adopted" % entry.register_file)
        elif not entry.signature_current:
            reasons.append("%s has an adoption that no longer reaches it" % entry.register_file)
        elif str(entry.outcome).upper() == "NOT_MET":
            return CONDITION_NOT_MET, "%s records NOT_MET" % entry.register_file
        else:
            met = True
    if met:
        return CONDITION_MET, None
    return EVIDENCE_CLAIMED_NOT_ADOPTED, "; ".join(reasons)


def acceptance_state(
    declaration_path: Union[str, os.PathLike] = DEFAULT_DECLARATION,
    register_dir: Union[str, os.PathLike] = DEFAULT_REGISTER_DIR,
    *,
    root: Union[str, os.PathLike] = ".",
) -> Dict[str, Any]:
    """Compute T4E.8's acceptance state. This function cannot return `T4E8_ACCEPTED`."""
    declaration_path = Path(declaration_path)
    declaration = load_acceptance_declaration(declaration_path)
    declaration_sha256 = _file_sha256(declaration_path)
    root = Path(root)

    register_dir = Path(register_dir)
    entries: List[ConditionEvidence] = []
    if register_dir.is_dir():
        for path in sorted(register_dir.glob("*.json")):
            if path.name.endswith("-adoption.json"):
                continue
            entries.append(_load_register_entry(path, declaration_sha256, root))

    by_condition: Dict[str, List[ConditionEvidence]] = {name: [] for name in CONDITION_IDS}
    for entry in entries:
        by_condition[entry.condition].append(entry)

    conditions = []
    for declared in declaration["acceptance_conditions"]:
        name = str(declared["id"])
        status, why = _condition_status(by_condition[name])
        conditions.append({
            "id": name,
            "name": declared["name"],
            "requires": declared["requires"],
            "licensed_by": declared["licensed_by"],
            "insufficient": declared["insufficient"],
            "status": status,
            "why": why,
            "evidence": [{
                "register_file": item.register_file,
                "register_sha256": item.register_sha256,
                "measurement": item.measurement,
                "measurement_sha256": item.measurement_sha256,
                "measurement_present": item.measurement_present,
                "measurement_digest_matches": item.measurement_digest_matches,
                "addresses": item.addresses,
                "outcome": item.outcome,
                "adopted": item.adopted,
                "adopted_by": item.adopted_by,
                "adopted_on": item.adopted_on,
                "signature_still_reaches_the_entry": item.signature_current,
            } for item in by_condition[name]],
        })

    statuses = [item["status"] for item in conditions]
    if CONDITION_NOT_MET in statuses:
        verdict = NOT_ACCEPTED
    elif all(status == CONDITION_MET for status in statuses):
        verdict = AWAITING_HUMAN_ACCEPTANCE
    else:
        verdict = INSUFFICIENT_EVIDENCE

    adoption = adoption_state(declaration_path.parent, declaration_path.name)
    result = {
        "schema": "t4e41-identity-acceptance-state/v1",
        "task": "T4E.41",
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": declaration_sha256,
        "declaration_status": declaration.get("status"),
        "acceptance_adopted": bool(adoption.get("adopted")),
        "adoption": adoption,
        "identity_target": declaration.get("target_and_scope", {}).get("identity_target"),
        # The narrative sections travel with the computed state so that a reader can honour
        # "I have read this declaration" where they sign it, rather than being asked to affirm
        # a document the surface only summarised.
        "why_this_exists": declaration.get("why_this_exists", {}),
        "what_this_declaration_is_not": declaration.get("what_this_declaration_is_not", []),
        "target_and_scope": declaration.get("target_and_scope", {}),
        "what_acceptance_would_and_would_not_license": declaration.get(
            "what_acceptance_would_and_would_not_license", {}),
        "declaration_file": declaration_path.name,
        "bound_evidence": verify_bound_evidence(declaration, root=root),
        "conditions": conditions,
        "conditions_met": sum(1 for status in statuses if status == CONDITION_MET),
        "conditions_total": len(conditions),
        "VERDICT": verdict,
        "verdict_semantics": declaration.get("verdict_semantics", {}),
        "code_may_emit_accepted": False,
        "what_does_not_discharge_this": declaration.get("what_does_not_discharge_this", []),
        "network_used": False,
        "claim_boundary": declaration["claim_boundary"],
    }
    result["receipt_sha256"] = _sha256(result)
    if result["VERDICT"] == FORBIDDEN_COMPUTED_VERDICT:  # pragma: no cover - unreachable
        raise UserInputError("T4E.41 state computed a verdict only a person may give")
    return result


__all__ = [
    "AWAITING_HUMAN_ACCEPTANCE", "CONDITION_IDS", "CONDITION_MET", "CONDITION_NOT_MET",
    "DEFAULT_DECLARATION", "DEFAULT_REGISTER_DIR", "EVIDENCE_CLAIMED_NOT_ADOPTED",
    "FORBIDDEN_COMPUTED_VERDICT", "INSUFFICIENT_EVIDENCE", "NOT_ACCEPTED", "NO_EVIDENCE",
    "REGISTER_SCHEMA", "SCHEMA", "ConditionEvidence", "acceptance_state",
    "load_acceptance_declaration", "verify_bound_evidence",
]
