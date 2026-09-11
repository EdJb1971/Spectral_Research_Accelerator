"""T4E.34: composing a declaration, and committing it alone so the ordering can be proved.

**Why this is not a form over a text editor.** T4E.30 surveyed every study in this repository and
found ten that could be carried into an evidence bundle and six that could not -- and **not one of
the six was refused for being declared after the fact**. In every case the declaration and the
measurement entered git in the *same commit*. Those declarations were almost certainly written
first; the sessions show it; git cannot separate them, so nothing downstream can check it.

That is not a discipline problem. It is what happens when writing a declaration means opening an
editor in the middle of a working session: it gets saved with everything else. The composer exists
to make the *provable* path the easy one. It writes the declaration, and then commits **that file
and nothing else**, which is the act that makes a later bundle possible.

**What it will not do.** It will not write the science. There is no template text, no suggested
prediction, no default claim boundary, and no example filled into a field. Every word of what is
declared comes from the person declaring it. What the composer supplies is *structure and
refusals*: a prediction with no stated falsifier is refused, an empty claim boundary is refused,
and a declaration that would overwrite an existing one is refused.

**A prediction that cannot fail is not a prediction.** TG19.5 was written because T4E.27 declared
an improvement that was arithmetically impossible before the run, and nothing caught it. The
general form of that check cannot be automated -- it needs the bars and the values -- but the
question can be forced: every prediction here must carry what would falsify it, in the declarer's
own words, and the refusal says why.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from src.core.adoption import PLACEHOLDER_NAMES
from src.core.errors import UserInputError

#: `t4e34`, `tg19`, `t4f7`. The prefix is how every other surface finds a study's declaration,
#: its measurement and its adoption, so a declaration that does not carry one is orphaned from
#: the study trail the moment it is written.
TASK = re.compile(r"^(?:[Tt]4[A-Za-z]|[Tt][Gg])\.?\d{1,3}(?:\.\d{1,3})?$")

_SLUG = re.compile(r"[^a-z0-9]+")


class CompositionRefused(UserInputError):
    """The declaration was not written, and the reason names what is missing."""

    status_code = 400
    client_safe = True


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git",) + arguments, capture_output=True, text=True,
                          timeout=60, check=False)


def slug(text: str) -> str:
    return _SLUG.sub("-", (text or "").strip().lower()).strip("-")


def declaration_name(task: str, artefact: str) -> str:
    """The file name the rest of this repository will look for."""
    return "%s-%s-declaration.json" % (task.lower().replace(".", ""), slug(artefact)[:60])


@dataclass(frozen=True)
class GateEntry:
    """One figure the run will be judged against, fixed before it runs."""

    quantity: str
    declared_value: str
    tolerance: Optional[str] = None

    def to_mapping(self) -> Dict[str, Any]:
        return {"quantity": self.quantity, "declared_value": self.declared_value,
                "tolerance": self.tolerance}


@dataclass(frozen=True)
class Prediction:
    """One prediction, and what would show it wrong."""

    name: str
    statement: str
    what_would_falsify_it: str

    def to_mapping(self) -> Dict[str, Any]:
        return {"name": self.name, "statement": self.statement,
                "what_would_falsify_it": self.what_would_falsify_it}


def _required(value: Optional[str], field: str, why: str) -> str:
    text = (value or "").strip()
    if not text:
        raise CompositionRefused("%s is required and was empty. %s Nothing was written."
                                 % (field, why), field=field)
    return text


def compose(directory: Path, *, task: str, artefact: str, declared_by: str,
            why_this_exists: str, what_this_is_not: str, the_inputs: str,
            claim_boundary: str, gate: Sequence[GateEntry],
            predictions: Sequence[Prediction], today: Optional[str] = None) -> Path:
    """Write one declaration, or refuse and write nothing.

    The status is always `DRAFTED_NOT_ADOPTED`: composing a declaration is not adopting it, and a
    surface that did both in one action would be signing at the moment of drafting.
    """
    if not TASK.match((task or "").strip()):
        raise CompositionRefused(
            "%r is not a task identifier such as 'T4E.34' or 'TG19.6'. Every other surface finds "
            "a study's declaration, measurement and adoption by this prefix, so one without it "
            "is orphaned from the study trail the moment it is written. Nothing was written."
            % task, task=task)

    name = (declared_by or "").strip()
    if name.lower() in PLACEHOLDER_NAMES or len(name) < 3:
        raise CompositionRefused(
            "%r is not a person. A declaration records whose scientific choice this is, and a "
            "placeholder or a tool's name where the person should be makes the record worthless. "
            "Nothing was written." % declared_by, declared_by=declared_by)

    artefact_text = _required(
        artefact, "artefact",
        "A declaration must say what is being made or measured, in one line.")
    why = _required(
        why_this_exists, "why_this_exists",
        "A declaration that does not say why it exists cannot be argued with later.")
    not_text = _required(
        what_this_is_not, "what_this_is_not",
        "Stating what a piece of work is NOT is how this programme has caught its own "
        "overreach; leaving it blank is where the overreach goes.")
    inputs = _required(
        the_inputs, "the_inputs",
        "Name the data, the population and the parameters. A run whose inputs were never "
        "declared cannot be reproduced or refused by name.")
    boundary = _required(
        claim_boundary, "claim_boundary",
        "No number in this repository is recorded without what it may not be used for.")

    if not predictions:
        raise CompositionRefused(
            "a declaration with no prediction fixes nothing before the run, and the run can then "
            "be read as having confirmed whatever it produced. State at least one. Nothing was "
            "written.", field="predictions")

    for index, prediction in enumerate(predictions, 1):
        if not prediction.statement.strip() or not prediction.name.strip():
            raise CompositionRefused(
                "prediction %d has no name or no statement. Nothing was written." % index,
                field="predictions")
        if not prediction.what_would_falsify_it.strip():
            raise CompositionRefused(
                "prediction %r does not say what would falsify it. A prediction that cannot fail "
                "is not a prediction: T4E.27 declared an improvement that was arithmetically "
                "impossible before the run, and it read as a risk that had been taken. Say what "
                "result would show this wrong. Nothing was written." % prediction.name,
                field="predictions", prediction=prediction.name)

    if not gate:
        raise CompositionRefused(
            "a declaration with no gate has nothing to judge the run against. State at least one "
            "quantity and the value it must meet. Nothing was written.", field="gate")
    for entry in gate:
        if not entry.quantity.strip() or not entry.declared_value.strip():
            raise CompositionRefused(
                "every gate entry needs a quantity and the value declared for it. Nothing was "
                "written.", field="gate")

    target = directory / declaration_name(task, artefact_text)
    if target.exists():
        raise CompositionRefused(
            "%s already exists. A declaration is written once: editing one after a run is how a "
            "gate becomes whatever the result was. Nothing was written." % target.name,
            existing=target.name)

    body = {
        "schema": "composed-declaration/v1",
        "status": "DRAFTED_NOT_ADOPTED",
        "task": task.strip().upper(),
        "artefact": artefact_text,
        "declared_by": name,
        "drafted_on": today or date.today().isoformat(),
        "composed_through": (
            "the declaration composer. The structure and the refusals are the instrument's; "
            "every word of the science below is the declarer's. No field was pre-filled."),
        "why_this_exists": why,
        "what_this_is_not": not_text,
        "the_inputs_declared_before_the_run": inputs,
        "the_gate_declared_before_the_run": [entry.to_mapping() for entry in gate],
        "the_predictions_declared_before_the_run": [p.to_mapping() for p in predictions],
        "claim_boundary": boundary,
        "what_would_make_this_worthless": (
            "Adjusting the gate after seeing the result, or adopting this declaration after the "
            "run. Both are visible in the git history of this file, which is why it is committed "
            "on its own before anything is measured."),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with open(target, "xb") as handle:
        handle.write((json.dumps(body, indent=2) + "\n").encode("utf-8"))
    return target


@dataclass(frozen=True)
class CommitResult:
    """What the commit did, or why it was refused."""

    committed: bool
    commit: Optional[str]
    committed_at: Optional[str]
    refusal: Optional[str]
    files_in_commit: List[str]

    def describe(self) -> Dict[str, Any]:
        return {"committed": self.committed, "commit": self.commit,
                "committed_at": self.committed_at, "refusal": self.refusal,
                "files_in_commit": self.files_in_commit,
                "why_alone": (
                    "A declaration committed alongside its measurement cannot be shown to "
                    "predate it, and T4E.30 refuses to bundle such a study. Six of this "
                    "repository's sixteen are refused for exactly that. Committing the "
                    "declaration by itself, before the run, is what makes the ordering "
                    "provable.")}


def commit_alone(path: Path, *, message: Optional[str] = None) -> CommitResult:
    """Commit this declaration and nothing else, so its registration moment is unambiguous.

    Other staged or modified files are left exactly as they are: `git commit -- <path>` takes
    only this path, whatever else is in the index.
    """
    text = str(path).replace("\\", "/")
    if not Path(path).is_file():
        return CommitResult(False, None, None,
                            "there is no file at %s to commit" % text, [])

    added = _git("add", "--", text)
    if added.returncode != 0:
        return CommitResult(False, None, None,
                            "git could not stage %s: %s" % (text, added.stderr.strip()), [])

    subject = message or ("%s declared: the gate, committed before anything is measured"
                          % Path(path).stem)
    body = ("The declaration is committed on its own, before the run it judges, so that the "
            "commit that registered it provably precedes the commit that records the "
            "measurement. A declaration committed alongside its measurement cannot be shown to "
            "predate it, and is refused when a study is carried into an evidence bundle.")
    done = _git("commit", "-m", subject, "-m", body, "--", text)
    if done.returncode != 0:
        output = (done.stdout + done.stderr).strip()
        if any(phrase in output for phrase in
               ("nothing to commit", "nothing added to commit", "no changes added")):
            return CommitResult(False, None, None,
                                "git reports nothing to commit for %s; it is already committed "
                                "and unchanged." % text, [])
        return CommitResult(False, None, None, "git refused the commit: %s" % output, [])

    sha = _git("rev-parse", "HEAD").stdout.strip() or None
    when = _git("log", "-1", "--format=%cI").stdout.strip() or None
    files = [line for line in _git("show", "--name-only", "--format=", "HEAD")
             .stdout.splitlines() if line.strip()]
    return CommitResult(True, sha, when, None, files)
