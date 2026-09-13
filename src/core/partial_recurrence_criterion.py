"""T4E.47: the ninth criterion, drafted and not adopted.

Eight criteria have been declared in this sequence and every one of them was falsified or
withdrawn. This module loads the ninth and refuses it in the ways the previous eight taught: a
declaration that has lost a constraint mapping, an acceptance condition, its extremes derivation
or its feasibility refusal is not the declaration that was reasoned about, and a loader that
accepted it would let the reasoning be edited out after the fact.

It loads and validates. It measures nothing, and it cannot report a result: there is no function
here that takes evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from src.core.adoption import adoption_state
from src.core.errors import UserInputError
from src.core.size_scaled_null import FROZEN_SCENES


SCHEMA = "t4e47-partial-recurrence-criterion-declaration/v1"

DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e47-partial-recurrence-criterion-declaration.json")

#: The constraints T4E.44 derived. A candidate declaration must say how it meets each one.
REQUIRED_CONSTRAINTS = ("K1", "K2", "K3", "K4", "K5", "K6")

#: The acceptance conditions this candidate is judged against. Named here so a declaration
#: cannot quietly drop the one it is most likely to fail.
REQUIRED_CONDITIONS = (
    "condition_1_recall_on_partial_presence",
    "condition_2_admission_on_partial_presence",
    "condition_3_no_hallucinated_presence",
    "condition_4_null",
    "condition_5_the_pair_extreme",
)

#: The reserved confirmatory families. Declaring a candidate does not open them.
RESERVED_FAMILIES = (
    (720, 721, 722, 723, 724, 725), (730, 731, 732, 733, 734, 735),
    (880, 881, 882, 883, 884, 885), (890, 891, 892, 893, 894, 895),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_declaration(
    path: Union[str, os.PathLike] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    """Load the ninth candidate, refusing a declaration whose reasoning has been edited out."""
    path = Path(path)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("T4E.47 declaration is unreadable: %s" % error) from error
    if not isinstance(body, dict) or body.get("schema") != SCHEMA or body.get("task") != "T4E.47":
        raise UserInputError("unsupported T4E.47 candidate declaration")

    constraints = body.get("constraints_this_candidate_is_declared_against", {})
    for name in REQUIRED_CONSTRAINTS:
        stated = [key for key in constraints if key.startswith(name + "_")]
        if not stated or not str(constraints[stated[0]]).strip():
            raise UserInputError(
                "T4E.47 must state how it meets %s; a candidate that does not answer a derived "
                "constraint has not been declared against it" % name)

    acceptance = body.get("acceptance", {})
    for condition in REQUIRED_CONDITIONS:
        if not str(acceptance.get(condition, "")).strip():
            raise UserInputError("T4E.47 acceptance lacks %s" % condition)

    derivation = body.get("what_is_derivable_before_measuring", {})
    if len(derivation) < 4 or not any("candidate 4" in str(value) for value in derivation.values()):
        raise UserInputError(
            "T4E.47 must derive what the rule does at the extremes of its own domain before "
            "adoption, which is the check candidate 4's declaration omitted")

    feasibility = body.get("feasibility_requirement_to_be_measured_before_adoption", {})
    if not str(feasibility.get("the_refusal", "")).strip():
        raise UserInputError(
            "T4E.47 must state what happens if the feasibility budget is exceeded; candidate 5 "
            "was withdrawn on exactly that and the refusal may not be left implicit")

    confirmatory = body.get("confirmatory", {})
    declared = tuple(tuple(item) for item in confirmatory.get("blocks", []))
    if declared != RESERVED_FAMILIES:
        raise UserInputError("T4E.47 must name the reserved families exactly and leave them shut")
    if "UNOPENED" not in str(confirmatory.get("status", "")):
        raise UserInputError("T4E.47 must record the reserves as unopened")
    if not str(body.get("claim_boundary", "")).strip():
        raise UserInputError("T4E.47 lacks a claim boundary")
    return body


def candidate_state(
    path: Union[str, os.PathLike] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    """Describe the candidate and what has and has not happened to it."""
    path = Path(path)
    declaration = load_candidate_declaration(path)
    state = adoption_state(path.parent, path.name)
    return {
        "schema": "t4e47-candidate-state/v1",
        "task": "T4E.47",
        "candidate": declaration["candidate"],
        "declaration": str(path).replace("\\", "/"),
        "declaration_sha256": _file_sha256(path),
        "declaration_status": declaration["status"],
        "adopted": bool(state.get("adopted")),
        "adoption": state,
        "scenes": FROZEN_SCENES,
        "constraints_answered": list(REQUIRED_CONSTRAINTS),
        "acceptance_conditions": list(REQUIRED_CONDITIONS),
        "measured_on": None,
        "has_been_measured": False,
        "feasibility_measured": False,
        "reserves_open": False,
        "what_adoption_would_authorise": (
            "the feasibility measurement on motif-free blocks, and then one development pass. "
            "Not a reserve, not a claim, and not C6 of the T4E.41 acceptance."),
        "network_used": False,
        "claim_boundary": declaration["claim_boundary"],
    }


__all__ = [
    "DEFAULT_DECLARATION", "REQUIRED_CONDITIONS", "REQUIRED_CONSTRAINTS", "RESERVED_FAMILIES",
    "SCHEMA", "candidate_state", "load_candidate_declaration",
]
