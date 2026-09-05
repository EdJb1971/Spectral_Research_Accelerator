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

**TG17.15 slice 5 adds a second recording and one thing the first did not need: a supersession.**
The scale/shape gate has refused since TG17.11, and the refusal was about a *method* - the declared
joint-reassignment null cannot reject at any inventory size its enumerator will reach.  TG17.15
built a different null for a different estimand and calibrated it.  That does not make the old
refusal wrong, and deleting it would be the easiest possible way to lose the finding: a reader of a
repository with no refusal in it cannot tell a limit that was overcome from one that was edited
away.

So the old record is kept and superseded, and the supersession is **checked rather than asserted**.
`scale_shape_supersession` recomputes the predecessor's claim from the same primitives the
predecessor turned on, and states three outcomes rather than one:

* `SUPERSEDED` - the old claim is still true on its own terms, and a successor recording exists, is
  bound to its contract and source, and passed.  The old refusal keeps its meaning; the new record
  says what it replaces and, as importantly, what it does not.
* `NOT_SUPERSEDED` - the successor is absent, stale or failed.  The old refusal stands alone, which
  is the state this repository was in before this slice.
* `VOID` - the predecessor's claim is **no longer true**.  Someone raised the enumeration cap or
  changed the correction, so the old refusal was retired on its own terms rather than superseded,
  and this record describes a world that no longer holds.  It must be re-derived, not kept.

`VOID` is the outcome that makes the word "checked" mean anything.  A supersession nobody
recomputes is a sentence about the past that keeps agreeing with itself.

**What the successor does not supersede, and why the gate still refuses.** The calibration measures
a null the declared qualification manifests do not request: they declare
`scale_partner_reassignment` with a replication count, and the calibrated method is exact pool
substitution over a declared partner pool.  That is checked against the manifests rather than
restated.  And the calibration's own claim boundary says it is evidence about fixtures and not
about whether a pool of real records is exchangeable.  Neither is discharged by measuring harder,
so the gate reads `REFUSED` for a narrower and now-measured reason rather than turning green.

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
SUPERSESSION_SCHEMA = "calibration-supersession/v1"

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

#: The modules whose source decides what a pool-substitution calibration measures: the calibration
#: itself, the statistic being ranked, the estimand that fixes alpha and the correction, the pool
#: the reference set is drawn from, the null, and the correction's implementation.
SCALE_SHAPE_SOURCES: Tuple[str, ...] = (
    "src/benchmarks/pool_calibration.py",
    "src/benchmarks/shape_calibration.py",
    "src/core/correspondence_estimand.py",
    "src/core/partner_pool.py",
    "src/core/pool_substitution_null.py",
    "src/statistics/multiple_comparisons.py",
)

SCALE_SHAPE_ENTRY_POINT = "src.benchmarks.pool_calibration:calibrate_pool_substitution"
SCALE_SHAPE_SEED = 20260904


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


# ----------------------------------------------- TG17.15 slice 5: the pool-substitution recording


def scale_shape_contract() -> Dict[str, Any]:
    """The declared contract of the pool-substitution calibration, as its own digestible fact.

    Read from the calibration module rather than restated.  The declared *expectations* are part of
    it, so relaxing one - lowering a rank-one floor, raising a family-wise ceiling, allowing more
    refusals - moves this digest and returns the gate to `NOT_RUN`, instead of leaving a recording
    made against a stricter contract standing beside a looser one.
    """
    from src.benchmarks.pool_calibration import (
        ALPHA, CALIBRATION_CONTRACT, CALIBRATION_FAMILY_SIZE, CANDIDATES_OFFERED,
        CASE_EXPECTATIONS, CORRECTION, DEFAULT_LADDER_REALISATIONS, DEFAULT_REALISATIONS,
        EFFECT_WEIGHTS, ORIENTATION, WITNESS_REALISATIONS,
    )

    contract = {
        "entry_point": SCALE_SHAPE_ENTRY_POINT,
        "mode": "scale_shape",
        "estimand": "per_correspondence",
        "null_family": "exact_pool_substitution",
        "statistic": "src.benchmarks.shape_calibration:shape_recurrence",
        "orientation": ORIENTATION,
        "family_size": CALIBRATION_FAMILY_SIZE,
        "realisations": DEFAULT_REALISATIONS,
        "ladder_realisations": DEFAULT_LADDER_REALISATIONS,
        "candidates_offered": CANDIDATES_OFFERED,
        "effect_weights": list(EFFECT_WEIGHTS),
        "witness_realisations": WITNESS_REALISATIONS,
        "alpha": ALPHA,
        "correction": CORRECTION,
        "admission_contract": CALIBRATION_CONTRACT.describe(),
        "seed": SCALE_SHAPE_SEED,
        "cases": {case: dict(expectation)
                  for case, expectation in sorted(CASE_EXPECTATIONS.items())},
    }
    return {**contract, "contract_sha256": _digest(contract)}


def scale_shape_source_digests(root: Optional[Path] = None) -> Dict[str, str]:
    base = Path(root) if root is not None else REPO_ROOT
    return {name: _file_digest(base, name) for name in SCALE_SHAPE_SOURCES}


def scale_shape_record_file(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else REPO_ROOT
    return base / RECORD_DIR / "scale_shape_calibration.json"


def _trim_case(row: Mapping[str, Any]) -> Dict[str, Any]:
    """What a release gate is entitled to read back, which is less than the calibration produced.

    The bounds and the witness, not the prose the benchmark writes for a reader of the benchmark.
    A recording that copied every explanatory sentence would be a second place for them to drift.
    """
    return {
        "case": row["case"],
        "within_expectation": bool(row["within_expectation"]),
        "certifies_rate": bool(row["certifies_rate"]),
        "realisations": int(row["realisations"]),
        "refusals": int(row["refusals"]),
        "refusal_rate_upper": row["refusal_rate"]["one_sided_upper"],
        "family_wise_error": {
            key: row["family_wise_error"][key]
            for key in ("successes", "trials", "point", "one_sided_upper")},
        "members_at_their_pool_floor": {
            key: row["members_at_their_pool_floor"][key]
            for key in ("successes", "trials", "one_sided_lower")},
        "members_at_their_floor_that_did_not_reject":
            int(row["members_at_their_floor_that_did_not_reject"]),
        "independent_uniformity": {
            key: value for key, value in row["independent_uniformity"].items()
            if key in ("n", "mean", "ks_statistic", "ks_p_value")},
        "pool_size_range": list(row["pool_size_range"]),
        "reproduction_witness": dict(row["reproduction_witness"]),
        "seconds": row["seconds"],
    }


def record_scale_shape_calibration(root: Optional[Path] = None, *,
                                   realisations: Optional[int] = None,
                                   ladder_realisations: Optional[int] = None) -> Dict[str, Any]:
    """Run the pool-substitution calibration and write what it measured, bound to what against.

    Tens of minutes, not minutes: five cases at two hundred realisations each, plus an eight-rung
    effect ladder.  That cost is why this is recorded rather than executed by the gate, and it is
    also why the recording carries a reproduction witness - see `WITNESS_REALISATIONS` for why a
    short run at the same seed is a *prefix* of this one rather than merely similar to it.
    """
    from src.benchmarks.pool_calibration import calibrate_pool_substitution

    base = Path(root) if root is not None else REPO_ROOT
    contract = scale_shape_contract()
    measured = calibrate_pool_substitution(
        realisations=int(realisations if realisations is not None
                         else contract["realisations"]),
        ladder_realisations=int(ladder_realisations if ladder_realisations is not None
                                else contract["ladder_realisations"]),
        seed=SCALE_SHAPE_SEED)
    ladder = measured["detection_profile"]
    record = {
        "schema": SCHEMA,
        "calibration": "scale_shape_calibration",
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": scale_shape_source_digests(base),
        "all_met": bool(measured["calibrated"]),
        "realisations": int(measured["realisations"]),
        "family_size": int(measured["family_size"]),
        "seed": int(measured["seed"]),
        "cases_failing_expectation": list(measured["cases_failing_expectation"]),
        "cases_whose_run_cannot_certify_a_rate":
            list(measured["cases_whose_run_cannot_certify_a_rate"]),
        "cases": [_trim_case(row) for row in measured["cases"]],
        "detection": {
            # One digest if the ladder held its inventory fixed, more than one if it did not.
            # Recorded as the set rather than as a boolean, so a reader sees which it was.
            "inventory_sha256": sorted({rung["inventory_sha256"] for rung in ladder["rungs"]}),
            "rungs": [{"weight": rung["weight"],
                       "realisations": rung["realisations"],
                       "refusals": rung["refusals"],
                       "member_detection": rung["member_detection"]["point"],
                       "family_detection": rung["family_detection"]["point"],
                       "member_uncorrected": rung["member_uncorrected"]["point"],
                       "members_at_their_pool_floor":
                           rung["members_at_their_pool_floor"]["point"],
                       "members_at_their_floor_that_did_not_reject":
                           rung["members_at_their_floor_that_did_not_reject"]}
                      for rung in ladder["rungs"]],
        },
        "claim_boundary": measured["claim_boundary"],
    }
    path = scale_shape_record_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return record


def read_scale_shape_calibration(root: Optional[Path] = None) -> Dict[str, Any]:
    """What the release gate is entitled to say about the pool-substitution calibration.

    The same four outcomes as the calendar one, blocking on three, and one difference stated rather
    than glossed.  The calendar calibration is cheap enough that the live measurement runs in the
    test suite on every pass and a guard compares it against the recording; this one costs tens of
    minutes and does not.  Its backstop is the reproduction witness the recording carries: the
    suite re-runs the leading realisations of every case at the recorded seed and compares digests.
    That is a smaller claim than the calendar one's, and it is described as the smaller claim it is
    rather than as an equivalent.
    """
    base = Path(root) if root is not None else REPO_ROOT
    contract = scale_shape_contract()
    sources = scale_shape_source_digests(base)
    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "entry_point": SCALE_SHAPE_ENTRY_POINT,
        "executed_here": False,
        "record_path": "%s/scale_shape_calibration.json" % RECORD_DIR,
        "contract_sha256": contract["contract_sha256"],
        "reproduction_basis": (
            "bound to the declared contract and to the source of the modules that decide what the "
            "measurement is, and carrying a per-case reproduction witness the test suite "
            "recomputes. The whole measurement is not re-run on every suite pass, which is a "
            "weaker backstop than the calendar recording's, deliberately and stated as such"),
    }
    path = scale_shape_record_file(base)
    if not path.is_file():
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["no pool-substitution calibration has been recorded for this "
                            "checkout"]}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:  # pragma: no cover - unreadable recording
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["the recorded calibration could not be read: %s" % (error,)]}

    stale = _stale_reasons(record, contract, sources)
    facts = {**facts, "recorded": record, "recorded_utc": record.get("recorded_utc")}
    if stale:
        return {**facts, "status": "NOT_RUN", "reasons": list(stale)}
    reasons: List[str] = []
    for row in record.get("cases", []):
        if not row.get("within_expectation"):
            reasons.append("%s did not do what the case declares it does" % row["case"])
        elif not row.get("certifies_rate"):
            reasons.append("%s was run too few times to bound its own error rate" % row["case"])
    if reasons or not record.get("all_met"):
        return {**facts, "status": "FAIL",
                "reasons": reasons or ["the recorded run did not meet its declared contract"]}
    return {**facts, "status": "PASS", "reasons": []}


def scale_shape_supersession(root: Optional[Path] = None) -> Dict[str, Any]:
    """Whether TG17.11's refusal is superseded, recomputed rather than remembered.

    The predecessor's claim is not read back from a document.  It is recomputed from the two
    primitives it turned on - the smallest family the correction can resolve, and the largest
    inventory the enumerator will draw from uniformly - using the same functions the gate's own
    applicability section uses, so the two cannot disagree about the world.
    """
    from src.benchmarks.pool_calibration import CALIBRATION_FAMILY_SIZE
    from src.benchmarks.shape_fixtures import minimum_resolvable_family
    from src.core.correspondence_estimand import ESTIMANDS, minimum_pool_size
    from src.core.structural_nulls import MAX_REASSIGNABLE_PAIRINGS

    minimum_family = minimum_resolvable_family()
    predecessor_holds = MAX_REASSIGNABLE_PAIRINGS < minimum_family
    successor = read_scale_shape_calibration(root)
    pool_needed = minimum_pool_size(CALIBRATION_FAMILY_SIZE)

    if not predecessor_holds:
        status = "VOID"
        reasons = [
            "the superseded refusal is no longer true on its own terms: the declared null now "
            "draws from inventories of up to %d and %d are resolvable, so it was retired rather "
            "than superseded and this record must be re-derived"
            % (MAX_REASSIGNABLE_PAIRINGS, minimum_family)]
    elif successor["status"] != "PASS":
        status = "NOT_SUPERSEDED"
        reasons = list(successor["reasons"])
    else:
        status = "SUPERSEDED"
        reasons = []

    return {
        "schema": SUPERSESSION_SCHEMA,
        "status": status,
        "reasons": reasons,
        "superseded": {
            "record": "TG17.11 slice 5",
            "estimand": "joint_structure",
            "inference": "drawn joint reassignment with a replication count",
            "claim": ("the declared scale/shape null cannot reject at any inventory size its "
                      "enumerator will draw from uniformly: no family smaller than %d pairings "
                      "can reject under the declared correction, and no inventory larger than %d "
                      "is drawn from rather than refused"
                      % (minimum_family, MAX_REASSIGNABLE_PAIRINGS)),
            "recomputed": {
                "minimum_resolvable_family": int(minimum_family),
                "largest_drawable_inventory": int(MAX_REASSIGNABLE_PAIRINGS),
            },
            "still_true": bool(predecessor_holds),
            "why_it_is_kept": (
                "a refusal deleted once a way round it is found leaves a repository in which a "
                "limit that was overcome and a limit that was edited away read the same"),
        },
        "superseding": {
            "record": "TG17.15 slices 1-4",
            "estimand": "per_correspondence",
            "inference": "exact pool substitution over a declared partner pool",
            "entry_point": SCALE_SHAPE_ENTRY_POINT,
            "status": successor["status"],
            "recorded_utc": successor.get("recorded_utc"),
            "tested_correspondences": int(CALIBRATION_FAMILY_SIZE),
            "minimum_pool_size": int(pool_needed),
            "why_the_bound_moves": (
                "the predecessor's inventory supplies both the hypotheses and the alternatives, "
                "so resolution and multiplicity are one knob and the crossover H_k/k <= alpha "
                "puts the answer at %d. A declared pool is not itself under test, so the two "
                "separate: %d correspondences need a pool of %d, and margin is bought by "
                "enlarging the pool at no correction cost"
                % (minimum_family, CALIBRATION_FAMILY_SIZE, pool_needed)),
        },
        "replaces": (
            "which inference the scale/shape gate decides applicability on, and nothing else"),
        "does_not_replace": (
            "the predecessor's own claim, recomputed above and still holding; the estimand it "
            "answered, which is a different question and stays registered and refused by name "
            "among %s; and the exchangeability of any pool of real records, which no measurement "
            "on built fixtures can establish"
            % ", ".join(sorted(ESTIMANDS.names()))),
    }


__all__ = ["CALENDAR_ENTRY_POINT", "CALENDAR_SEED", "CALENDAR_SOURCES", "RECORD_DIR",
           "REPO_ROOT", "SCHEMA", "calendar_applicability", "calendar_contract",
           "read_calendar_calibration", "record_calendar_calibration", "record_file",
           "source_digests",
           "SCALE_SHAPE_ENTRY_POINT", "SCALE_SHAPE_SEED", "SCALE_SHAPE_SOURCES",
           "SUPERSESSION_SCHEMA", "read_scale_shape_calibration",
           "record_scale_shape_calibration", "scale_shape_contract", "scale_shape_record_file",
           "scale_shape_source_digests", "scale_shape_supersession"]
