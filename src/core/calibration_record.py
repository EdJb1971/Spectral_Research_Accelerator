"""TG17.12 recording a calibration that is measured outside orchestration, for a release gate.

**The problem this solves.** `src.benchmarks.family_calibration.calibrate_family` runs the frozen
calendar family against fixtures whose answers were fixed in TG17.0, and it passes.  It has always
passed.  The `calendar_calibration` release gate nevertheless read `NOT_RUN`, because nothing
carried that measurement into the qualification record - the gate and the measurement had no
channel between them.  `NOT_RUN` was therefore true of the record and false of the world, which is
the one combination a gate must never be in for long.

**Why the gate does not simply run it.** A calibration is a scientific measurement and
`experiment_qualification` is a release gate; running four fixture cases at 999 replications inside
plan assembly would make an HTTP route's cost depend on a benchmark's.  This is the same separation
`scale_shape_calibration` states (TG17.11), and this module is the channel that separation needs.

**What a recorded calibration is bound to.** A receipt that says only "it passed" is a sentence,
not evidence: it survives every later change to the thing it describes.  A recording here binds
two digests and is refused as unrun if either moves.

* The **declared contract** - the cases and the rejection counts frozen with them, the family
  size, alpha, correction, replications, channel, null family and seed.  Relaxing an expectation
  changes this digest, which is precisely what `CALIBRATION_CASES` says it exists to prevent.
* The **source** of the modules that decide what the measurement is: the calibration itself, the
  fixtures it builds, the null it draws from and the correction it applies.

Neither digest makes a recording tamper-evident against someone editing this repository, and the
source binding covers those four files rather than the whole transitive import graph.  It is not
meant to; the backstop is that the live calibration runs in the test suite on every pass, and a
guard there compares what it produced against what is recorded here.  A recording that drifts from
reality fails a test rather than quietly qualifying a release.

Nothing here is evidence about the world.  It records whether a declared family behaved as
declared on fixtures with known answers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCHEMA = "calibration-record/v1"

#: The repository root, from this file's own location. The recording is source, not an artefact:
#: it travels with the code it attests to, and a checkout without it reads `NOT_RUN`.
REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = "calibration"

#: The modules whose source decides what a calendar calibration measures. Not the whole import
#: graph - see the module docstring for why that boundary is stated rather than hidden.
CALENDAR_SOURCES: Tuple[str, ...] = (
    "src/benchmarks/family_calibration.py",
    "src/benchmarks/multidomain_flagship.py",
    "src/core/structural_nulls.py",
    "src/statistics/multiple_comparisons.py",
)

CALENDAR_ENTRY_POINT = "src.benchmarks.family_calibration:calibrate_family"
CALENDAR_SEED = 20260831


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_digest(root: Path, relative: str) -> str:
    """Newline-normalised, so a Windows checkout and a POSIX one agree about the same source."""
    path = root / relative
    if not path.is_file():  # pragma: no cover - a missing source module is a broken checkout
        return "ABSENT"
    body = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(body).hexdigest()


def calendar_contract() -> Dict[str, Any]:
    """The declared scientific contract of the calendar calibration, as its own digestible fact.

    Read from the calibration module rather than restated, so that a later slice relaxing an
    expectation moves this digest instead of leaving a stale sentence agreeing with it.
    """
    from src.benchmarks.family_calibration import (
        ALPHA, CALIBRATION_CASES, CALIBRATION_FAMILY_SIZE, CHANNEL, CORRECTION,
        DEFAULT_REPLICATIONS,
    )

    contract = {
        "entry_point": CALENDAR_ENTRY_POINT,
        "mode": "calendar_aligned",
        "null_family": "independent_native_clock_shift",
        "family_size": CALIBRATION_FAMILY_SIZE,
        "replications": DEFAULT_REPLICATIONS,
        "alpha": ALPHA,
        "correction": CORRECTION,
        "channel": CHANNEL,
        "seed": CALENDAR_SEED,
        "cases": [{"case": row["case"], "focus": row["focus"],
                   "minimum_rejections": row["minimum_rejections"],
                   "maximum_rejections": row["maximum_rejections"],
                   "expectation": row["expectation"]}
                  for row in CALIBRATION_CASES],
    }
    return {**contract, "contract_sha256": _digest(contract)}


def source_digests(root: Optional[Path] = None) -> Dict[str, str]:
    base = Path(root) if root is not None else REPO_ROOT
    return {name: _file_digest(base, name) for name in CALENDAR_SOURCES}


def calendar_applicability() -> Dict[str, Any]:
    """Whether the calendar null can resolve anything, at the two sizes that matter, before power.

    The counterpart of `scale_shape_applicability`, and the interesting thing is that it comes out
    the other way.  Both modes have a p-value floor.  The scale/shape null's floor is fixed by the
    inventory - a `k`-pairing family enumerates `k` partners and cannot go below `1/k` - so it is
    bought with domains, and TG17.11 measured that the sizes required and the sizes drawable do
    not overlap.  The calendar null's floor is `1 / (1 + replications)`: its surrogates come from
    clock shifts the record itself supports, so there is no enumeration ceiling and the floor is
    bought with computation.  Two bounds that do not meet, against two that meet with room to
    spare.

    Both configurations are checked against the real correction rather than asserted: the
    calibration's own family, and the calendar plan the qualification manifests actually declare.
    """
    from src.benchmarks.family_calibration import ALPHA, CALIBRATION_FAMILY_SIZE, CORRECTION
    from src.core.experiment_family import correction_plan
    from src.core.experiment_manifest import flagship_recipe
    from src.statistics.multiple_comparisons import check_power

    calibration = check_power(calendar_contract()["replications"], CALIBRATION_FAMILY_SIZE,
                              alpha=ALPHA, method=CORRECTION)
    declared = correction_plan(flagship_recipe())
    return {
        "floor_is_bought_with": "replications",
        "enumeration_ceiling": None,
        "calibration_family": {
            "members": CALIBRATION_FAMILY_SIZE,
            "replications": calibration["n_surrogates"],
            "p_value_floor": calibration["p_value_floor"],
            "replications_required": calibration["surrogates_required"],
            "can_reject_after_correction": bool(calibration["can_reject_after_correction"]),
        },
        "declared_plan": {
            "members": declared["correction_unit_members"],
            "declared_search_members": declared["declared_search_members"],
            "replications": declared["n_surrogates"],
            "p_value_floor": declared["p_value_floor"],
            "replications_required": declared["surrogates_required"],
            "can_reject_after_correction": bool(declared["affordable"]),
        },
        "alpha": ALPHA,
        "correction": CORRECTION,
        "claim_boundary": (
            "Applicability, not power: whether a rejection is arithmetically reachable at the "
            "declared configuration. Whether one occurs on fixtures with known answers is the "
            "recorded calibration, and whether one occurs on acquired data is neither."),
    }


def record_file(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else REPO_ROOT
    return base / RECORD_DIR / "calendar_calibration.json"


def record_calendar_calibration(root: Optional[Path] = None, *,
                                replications: Optional[int] = None) -> Dict[str, Any]:
    """Run the frozen calendar calibration and write what it measured, bound to what it measured
    it against.  Minutes, not milliseconds: this is the scientific measurement, run deliberately.
    """
    from src.benchmarks.family_calibration import calibrate_family

    base = Path(root) if root is not None else REPO_ROOT
    contract = calendar_contract()
    measured = calibrate_family(
        replications=int(replications if replications is not None else contract["replications"]),
        seed=CALENDAR_SEED)
    record = {
        "schema": SCHEMA,
        "calibration": "calendar_calibration",
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": source_digests(base),
        "all_met": bool(measured["all_met"]),
        "replications": int(measured["replications"]),
        "family_size": int(measured["family_size"]),
        "cases": [{"case": row["case"],
                   "expectation": row["expectation"],
                   "minimum_rejections": row["minimum_rejections"],
                   "maximum_rejections": row["maximum_rejections"],
                   "n_rejected_after_correction": row["n_rejected_after_correction"],
                   "met": bool(row["met"])}
                  for row in measured["cases"]],
        "claim_boundary": measured["claim_boundary"],
    }
    path = record_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit newlines: a recording is committed source, and a Windows run must not rewrite
    # every line of it as a diff. The digests it carries are over the source modules, not over
    # this file, so line endings change nothing scientific - only what a reviewer has to read.
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return record


def _stale_reasons(record: Mapping[str, Any], contract: Mapping[str, Any],
                   sources: Mapping[str, str]) -> Tuple[str, ...]:
    reasons: List[str] = []
    if record.get("contract_sha256") != contract["contract_sha256"]:
        reasons.append("the declared calibration contract has changed since it was recorded")
    recorded_sources = record.get("source_sha256") or {}
    moved = sorted(name for name, digest in sources.items()
                   if recorded_sources.get(name) != digest)
    if moved:
        reasons.append("the source of %s has changed since it was recorded" % ", ".join(moved))
    return tuple(reasons)


def read_calendar_calibration(root: Optional[Path] = None) -> Dict[str, Any]:
    """What the release gate is entitled to say about the calendar calibration, and why.

    Four outcomes, and three of them are blocking.  A recording that is absent, that was made
    against a different declared contract, or that was made against different source is not a
    weaker pass: it is a measurement of something else, and the gate reads `NOT_RUN` rather than
    inheriting a verdict from it.
    """
    base = Path(root) if root is not None else REPO_ROOT
    contract = calendar_contract()
    sources = source_digests(base)
    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "entry_point": CALENDAR_ENTRY_POINT,
        "executed_here": False,
        "record_path": "%s/calendar_calibration.json" % RECORD_DIR,
        "contract_sha256": contract["contract_sha256"],
        "applicability": calendar_applicability(),
    }
    path = record_file(base)
    if not path.is_file():
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["no calibration has been recorded for this checkout"]}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:  # pragma: no cover - unreadable recording
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["the recorded calibration could not be read: %s" % (error,)]}

    stale = _stale_reasons(record, contract, sources)
    facts = {**facts, "recorded": record, "recorded_utc": record.get("recorded_utc")}
    if stale:
        return {**facts, "status": "NOT_RUN", "reasons": list(stale)}
    if not record.get("all_met"):
        unmet = [row["case"] for row in record.get("cases", []) if not row.get("met")]
        return {**facts, "status": "FAIL",
                "reasons": ["%s did not meet the answer frozen with the fixture" % case
                            for case in unmet]}
    return {**facts, "status": "PASS", "reasons": []}


__all__ = ["CALENDAR_ENTRY_POINT", "CALENDAR_SEED", "CALENDAR_SOURCES", "RECORD_DIR",
           "REPO_ROOT", "SCHEMA", "calendar_applicability", "calendar_contract",
           "read_calendar_calibration", "record_calendar_calibration", "record_file",
           "source_digests"]
