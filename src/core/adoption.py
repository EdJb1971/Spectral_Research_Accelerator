"""T4E.32: adoption as an act a person performs, through a surface rather than a text editor.

**What was true until now.** Every adoption in this repository was a JSON file written by hand.
The rule that produced them is right and is not being relaxed: *code does not sign a scientific
declaration for a person.* But that rule was being enforced by the awkwardness of the medium,
which is a poor place to keep a principle. A maintainer who clicks a button carrying their own
name, having read the declaration and typed the words that say they adopt it, **has signed it**.
That is what a signature is. Editing a file by hand is not more deliberate than that; it is only
slower, and slower is not a safeguard.

**What is kept, exactly.** An adoption binds the declaration's `sha256`, so it cannot drift onto
an amended declaration. It names who adopted it and when. It is written once and never
overwritten. And it cannot be produced without an affirmation typed out in full -- a single
click cannot sign anything here, because a button that signs on one press is a button that signs
by accident.

**What this module will not do.** It will not write the declaration. It will not decide that a
declaration is worth adopting, fill in why, or supply a default for the maintainer's name. Every
one of those is the scientific act itself, and a surface that offered a plausible default for
them would be signing on the maintainer's behalf while appearing to ask.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

from src.core.errors import InvalidParameterError, UserInputError

#: The words a maintainer must type for an adoption to be written. Not a checkbox and not a
#: single click: typing a sentence is the smallest act that cannot happen by accident, and an
#: adoption that could happen by accident is worth nothing as a signature.
REQUIRED_AFFIRMATION = "I have read this declaration and I adopt it"

#: A name that is not a name. Refused so that an adoption never carries a placeholder where the
#: person should be -- the whole artefact is an attribution.
PLACEHOLDER_NAMES = frozenset({
    "", "-", "n/a", "na", "none", "null", "unknown", "anonymous", "me", "user", "maintainer",
    "test", "tester", "claude", "assistant", "ai", "chatgpt", "gpt", "llm", "the maintainer",
})

_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")


class AdoptionRefused(UserInputError):
    """The adoption was not written, and the reason names what would allow it."""

    status_code = 400
    client_safe = True


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adoption_name(declaration: str) -> str:
    """The adoption file that belongs to a declaration, by the repository's own convention."""
    stem = declaration[:-5] if declaration.lower().endswith(".json") else declaration
    for suffix in ("-declaration", "-design", "_declaration", "_design"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return "%s-adoption.json" % stem


@dataclass(frozen=True)
class Adoption:
    """One signed adoption, and the declaration it is bound to by content."""

    path: Path
    declaration: str
    declaration_sha256: str
    adopted_by: str
    adopted_on: str
    body: Dict[str, Any]

    def describe(self) -> Dict[str, Any]:
        return {"file": self.path.name, "adopts": self.declaration,
                "adopts_sha256": self.declaration_sha256, "adopted_by": self.adopted_by,
                "adopted_on": self.adopted_on}


def sign_declaration(directory: Path, declaration: str, *, adopted_by: str, adopted_as: str,
                     what_was_adopted: str, affirmation: str,
                     why: Optional[str] = None, today: Optional[str] = None) -> Adoption:
    """Write one adoption, bound to the declaration's digest, or refuse and write nothing.

    Every refusal says what would lift it. A maintainer told only that the adoption failed learns
    nothing about what they are being asked for.
    """
    if not _FILE_NAME.match(declaration or ""):
        raise AdoptionRefused(
            "%r is not a declaration file name in the calibration store. Give the file name "
            "alone, such as 't4e28-join-rerun-declaration.json'; a path is refused rather than "
            "sanitised." % declaration, declaration=declaration)

    source = directory / declaration
    if not source.is_file():
        raise AdoptionRefused(
            "there is no declaration named %r to adopt. Nothing was written. An adoption whose "
            "declaration does not exist would bind a digest of nothing." % declaration,
            declaration=declaration)

    if (affirmation or "").strip().rstrip(".") != REQUIRED_AFFIRMATION:
        raise AdoptionRefused(
            "the affirmation must be typed exactly as %r. Nothing was written. This is not "
            "ceremony: a signature that can be produced by one click is a signature that can be "
            "produced by accident, and this artefact is an attribution to a person."
            % REQUIRED_AFFIRMATION,
            required=REQUIRED_AFFIRMATION)

    name = (adopted_by or "").strip()
    if name.lower() in PLACEHOLDER_NAMES or len(name) < 3:
        raise AdoptionRefused(
            "%r is not a person. An adoption is an attribution and must carry the name of "
            "whoever is making the scientific choice, not a placeholder and not the name of a "
            "tool. Nothing was written." % adopted_by, adopted_by=adopted_by)

    for label, value in (("adopted_as", adopted_as), ("what_was_adopted", what_was_adopted)):
        if not (value or "").strip():
            raise AdoptionRefused(
                "%s is required and was empty. Nothing was written. An adoption that does not "
                "say what was adopted records only that a button was pressed." % label,
                field=label)

    target = directory / adoption_name(declaration)
    if target.exists():
        raise AdoptionRefused(
            "%s already exists. An adoption is written once and never overwritten: a "
            "declaration that has been adopted and then re-adopted on different terms is two "
            "different scientific acts, and the first does not disappear because the second "
            "happened. Nothing was written." % target.name, existing=target.name)

    digest = digest_of(source)
    when = today or date.today().isoformat()
    body: Dict[str, Any] = {
        "schema": "maintainer-adoption/v1",
        "adopts": str(source).replace("\\", "/"),
        "adopts_sha256": digest,
        "adopted_by": name,
        "adopted_on": when,
        "adopted_as": adopted_as.strip(),
        "what_was_adopted": what_was_adopted.strip(),
        "affirmation": REQUIRED_AFFIRMATION,
        "signed_through": (
            "the adoption surface, by a person who typed the affirmation above. Code does not "
            "sign a scientific declaration for anyone; this record exists because a named "
            "maintainer performed the act, and the surface refuses to supply any part of it."),
        "what_this_adoption_does_not_do": (
            "It does not make the declaration correct, does not permit any claim, and does not "
            "adopt any amendment made after the digest above. A declaration edited after this "
            "adoption is a different declaration and this signature does not reach it."),
    }
    if (why or "").strip():
        body["why_the_maintainer_adopted_it"] = why.strip()

    try:
        with open(target, "xb") as handle:
            handle.write((json.dumps(body, indent=2) + "\n").encode("utf-8"))
    except FileExistsError:
        raise AdoptionRefused(
            "%s was created while this adoption was being written; nothing was overwritten."
            % target.name, existing=target.name) from None

    return Adoption(path=target, declaration=body["adopts"], declaration_sha256=digest,
                    adopted_by=name, adopted_on=when, body=body)


def adoption_state(directory: Path, declaration: str) -> Dict[str, Any]:
    """Whether a declaration is adopted, and whether the signature still reaches it.

    A declaration amended after adoption is a different declaration. That is reported here rather
    than left for a reader to notice, because an adoption whose digest no longer matches is the
    quietest possible way for a signed record to become untrue.
    """
    source = directory / declaration
    target = directory / adoption_name(declaration)
    state: Dict[str, Any] = {
        "declaration": declaration,
        "declaration_exists": source.is_file(),
        "adoption_file": target.name,
        "adopted": target.is_file(),
        "required_affirmation": REQUIRED_AFFIRMATION,
    }
    if not source.is_file():
        return state
    state["declaration_sha256"] = digest_of(source)
    if not target.is_file():
        return state
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        state["unreadable"] = str(exc)
        return state
    signed = body.get("adopts_sha256")
    state["adopted_by"] = body.get("adopted_by")
    state["adopted_on"] = body.get("adopted_on")
    state["adopts_sha256"] = signed
    state["signature_still_reaches_the_declaration"] = signed == state["declaration_sha256"]
    if not state["signature_still_reaches_the_declaration"]:
        state["drift"] = (
            "the declaration has changed since it was adopted. The signature binds %s and the "
            "file now hashes to %s, so what was signed is not what is on disk. The adoption is "
            "not evidence about the current text."
            % (signed, state["declaration_sha256"]))
    return state
