"""TG18.5 slice 4: a rendered browser run reaching the release gate, or the gate staying shut.

TG17.10 registered two gates it could not award itself.  ``browser_no_glue`` reads ``NOT_RUN``
because "a deterministic backend rehearsal cannot observe a rendered browser and must not award
itself a gate on someone else's evidence", and ``scientist_actions`` reads ``NOT_MEASURED`` for the
action count and the refusal-explanation time.  This module is the channel those two need, and it
is built to the constraint TG18.5 set for itself: *the ledger may ingest a measurement with its
provenance, and may never synthesize one it did not receive.*

**Recording and deciding are separate, and separately owned.**  ``frontend/e2e/`` writes what a run
did - `qualification-reporter.ts` for the run itself, `scientist-actions.spec.ts` for the two
counts.  Neither decides anything.  This module decides, and every one of its decisions can be a
refusal.

**What a recording is bound to.**  The same shape TG17.12 established for a calibration, because
the problem is the same: a receipt saying "it passed" survives every later change to the thing it
describes.

* The **source of every spec in the suite**.  A recording is a measurement of a particular set of
  assertions.  Weaken a spec and the recording is no longer about the suite that now exists, so
  the gate returns to ``NOT_RUN`` rather than let a pass earned by stronger assertions be spent by
  weaker ones.
* The **completeness of the run**.  A single-spec invocation is the normal way to work on a test,
  and it writes a recording that looks green.  The reporter therefore records both the specs that
  ran and the whole inventory it found, and a run that did not cover the inventory is refused.

Neither binding is tamper-evidence against someone editing this repository.  What it is proof
against is drift, which is the failure that actually happens: a suite quietly weakened, or a
partial run mistaken for a full one.

Nothing here is evidence about the world.  A browser run is apparatus behaviour: it says a page
rendered and a path was walkable, never that a number on it is right.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA = "browser-evidence/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_DIR = "measurements"
SPEC_DIR = "frontend/e2e"

BROWSER_RUN = "browser_run.json"
SCIENTIST_ACTIONS = "scientist_actions.json"

#: The counts TG18.5 slice 3 froze. They live here as well as in the spec because a gate that
#: reads whatever number it is handed is not a gate: a path that grew two actions would be
#: reported rather than refused. The spec asserts them at measurement time; this asserts that the
#: measurement it is reading is the one that was frozen.
DECLARED_PLAN_ACTIONS = 14
DECLARED_ACTIONS_TO_REFUSAL = 3
DECLARED_REFUSAL_TO_REMEDIATION = 4


def _file_digest(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():  # pragma: no cover - a missing spec is a broken checkout
        return "ABSENT"
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def spec_digests(root: Optional[Path] = None) -> Dict[str, str]:
    """Every spec in the suite, digested from source as the reporter digests them."""
    base = Path(root) if root is not None else REPO_ROOT
    directory = base / SPEC_DIR
    if not directory.is_dir():  # pragma: no cover - a checkout without the frontend
        return {}
    return {path.name: _file_digest(base, "%s/%s" % (SPEC_DIR, path.name))
            for path in sorted(directory.glob("*.spec.ts"))}


def _read(root: Path, name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    path = root / MEASUREMENT_DIR / name
    if not path.is_file():
        return None, "no %s has been recorded for this checkout" % name
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (ValueError, OSError) as error:  # pragma: no cover - unreadable recording
        return None, "%s could not be read: %s" % (name, error)


def _run_reasons(record: Dict[str, Any], digests: Dict[str, str]) -> List[str]:
    """Every reason this recording may not qualify anything. Order is stable, not prioritised."""
    reasons: List[str] = []

    recorded = dict(record.get("spec_sha256") or {})
    moved = sorted(name for name, digest in digests.items() if recorded.get(name) != digest)
    if moved:
        reasons.append("the source of %s has changed since the run was recorded"
                       % ", ".join(moved))
    vanished = sorted(set(recorded) - set(digests))
    if vanished:
        reasons.append("%s was recorded but no longer exists" % ", ".join(vanished))

    inventory = sorted(record.get("spec_inventory") or [])
    ran = sorted(record.get("specs_that_ran") or [])
    if not ran:
        reasons.append("the recorded run executed no tests")
    elif inventory != ran:
        missing = sorted(set(inventory) - set(ran))
        reasons.append(
            "the recorded run covered %d of %d specs and is a partial run: %s did not run"
            % (len(ran), len(inventory), ", ".join(missing) or "some specs"))

    return reasons


def browser_run_evidence(root: Optional[Path] = None) -> Dict[str, Any]:
    """What the ``browser_no_glue`` gate is entitled to say, and why.

    ``PASS`` requires a complete run of the suite this checkout actually contains, with nothing
    failed.  Every other outcome is blocking, and the three blocking kinds are kept apart: a run
    that is absent or stale or partial has *not happened* for this code and reads ``NOT_RUN``; a
    run that happened and failed reads ``FAIL``.
    """
    base = Path(root) if root is not None else REPO_ROOT
    digests = spec_digests(base)
    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "measured_by": "frontend/e2e (Playwright, Chromium)",
        "executed_here": False,
        "record_path": "%s/%s" % (MEASUREMENT_DIR, BROWSER_RUN),
        "specs_in_this_checkout": len(digests),
        "claim_boundary": (
            "Apparatus behaviour in a rendered browser: a page rendered and a path was walkable. "
            "It is not evidence that any workspace computes anything correctly."),
    }
    record, unreadable = _read(base, BROWSER_RUN)
    if record is None:
        return {**facts, "status": "NOT_RUN", "recorded": None, "reasons": [unreadable]}

    reasons = _run_reasons(record, digests)
    totals = dict(record.get("totals") or {})
    summary = {
        "recorded_utc": record.get("recorded_utc"),
        "playwright_status": record.get("status"),
        "passed": int(totals.get("passed", 0)),
        "failed": int(totals.get("failed", 0)),
        "skipped": int(totals.get("skipped", 0)),
        "specs_that_ran": len(record.get("specs_that_ran") or []),
        "artefacts": len(record.get("artefacts") or []),
    }
    facts = {**facts, "recorded": summary}

    if reasons:
        return {**facts, "status": "NOT_RUN", "reasons": reasons}
    if summary["failed"] or record.get("status") != "passed":
        return {**facts, "status": "FAIL",
                "reasons": ["the recorded run reported %d failed of %d and a status of %r"
                            % (summary["failed"],
                               summary["passed"] + summary["failed"], record.get("status"))]}
    return {**facts, "status": "PASS", "reasons": []}


def scientist_action_evidence(root: Optional[Path] = None) -> Dict[str, Any]:
    """The two counts, or an honest ``NOT_MEASURED``; never a number this process invented.

    The wall-clock figures the measurement carries are passed through under names that say they
    are unasserted, and nothing here compares them to anything.  A duration is a property of the
    machine that ran the suite, and a gate that turned on one would fail for reasons unrelated to
    the interface while passing on a fast machine as the interface got slower.
    """
    base = Path(root) if root is not None else REPO_ROOT
    unmeasured = {
        "status": "NOT_MEASURED",
        "definition": "visible researcher actions from a clean browser session",
        "adapter_specific_framework_edits": "NOT_MEASURED",
        "refusal_explanation_time_seconds": "NOT_MEASURED",
    }
    record, unreadable = _read(base, SCIENTIST_ACTIONS)
    if record is None:
        return {**unmeasured, "reasons": [unreadable]}

    plan = dict(record.get("declared_plan_to_completed_run") or {})
    refusal = dict(record.get("refusal_to_remediation") or {})
    declared = {
        "actions_to_a_completed_run": DECLARED_PLAN_ACTIONS,
        "actions_to_reach_the_refusal": DECLARED_ACTIONS_TO_REFUSAL,
        "actions_from_refusal_to_remediation": DECLARED_REFUSAL_TO_REMEDIATION,
    }
    measured = {
        "actions_to_a_completed_run": plan.get("actions"),
        "actions_to_reach_the_refusal": refusal.get("actions_to_reach_the_refusal"),
        "actions_from_refusal_to_remediation":
            refusal.get("actions_from_refusal_to_remediation"),
    }
    drifted = sorted(name for name, value in declared.items() if measured.get(name) != value)
    if drifted:
        return {**unmeasured,
                "reasons": ["the measured %s does not match the count frozen in this module"
                            % ", ".join(drifted)],
                "measured": measured, "declared": declared}

    return {
        "status": "MEASURED",
        "definition": record.get("definition") or unmeasured["definition"],
        "counted": record.get("counted"),
        "not_counted": record.get("not_counted"),
        # Still not measured, and still not by a browser: a source-edit audit belongs to the
        # `synthetic_fifth_adapter` gate, which no rendered run can observe.
        "adapter_specific_framework_edits": "NOT_MEASURED",
        **measured,
        # Carried, never compared. The names say so where a reader will see them.
        "refusal_explanation_time_seconds_unasserted":
            refusal.get("refusal_explanation_time_seconds_unasserted"),
        "completed_run_elapsed_seconds_unasserted": plan.get("elapsed_seconds_unasserted"),
        "remediation": refusal.get("remediation"),
        "claim_boundary": record.get("claim_boundary"),
        "reasons": [],
    }


__all__ = ["BROWSER_RUN", "DECLARED_ACTIONS_TO_REFUSAL", "DECLARED_PLAN_ACTIONS",
           "DECLARED_REFUSAL_TO_REMEDIATION", "MEASUREMENT_DIR", "REPO_ROOT", "SCHEMA",
           "SCIENTIST_ACTIONS", "SPEC_DIR", "browser_run_evidence",
           "scientist_action_evidence", "spec_digests"]
