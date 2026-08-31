"""TG17.10 deterministic qualification of the configurable experiment apparatus.

This module is deliberately a *release gate*, not a scientific worker.  It exercises the six
frozen duration/mode declarations through metadata preflight, the resumable orchestrator,
portable receipt export and read-only replay.  Every artefact it creates is labelled as a
known-answer rehearsal, and the overall verdict remains ``NOT_RELEASEABLE`` while browser,
mode-specific calibration and live-source gates have not been recorded separately.

That distinction is load-bearing.  A green state-machine rehearsal proves that the apparatus
preserves one plan.  It says nothing about whether an archive was reachable, a statistic was
calibrated, an effect was detected, or a claim should move.

It is also load-bearing that a cell may come back ``REFUSED``.  The frozen quartet includes the
bespoke order-book domain, whose adapter declares that it cannot carry ``scale_partner_reassignment``
because a depositor-supplied record has no native duration worth comparing shapes across.  The
three scale/shape cells are therefore refused at preflight, and the gate records that refusal as
the qualification result rather than widening a framework default until the matrix turns green.
A domain's declared refusal is the science; a passing matrix obtained by overruling it would be
the failure this gate exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.benchmarks.multidomain_flagship import EXPERIMENT_CONTRACT
from src.benchmarks.structural_trajectory import known_answer_native
from src.core.experiment_manifest import (
    CrossDomainExperimentSpec,
    canonical_bytes,
    flagship_recipe,
    manifest_sha256,
    preflight_manifest,
)
from src.core.experiment_receipt import publish_bundle, verify_bundle
from src.core.experiment_run import RunStore
from src.core.run_workers import build_suite


SCHEMA = "experiment-qualification/v1"
RECORD_KIND = "deterministic_known_answer_rehearsal_not_acquired_data"
MODES: Tuple[str, ...] = ("calendar_aligned", "scale_shape_aligned")
DURATIONS: Tuple[str, ...] = tuple(EXPERIMENT_CONTRACT["duration_presets"])


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def qualification_manifest(duration: str, mode: str, *, suffix: str = "") \
        -> CrossDomainExperimentSpec:
    """One frozen matrix cell, derived without inventing a second flagship configuration."""
    if duration not in DURATIONS:
        raise ValueError("duration must be one of %s" % (list(DURATIONS),))
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (list(MODES),))

    body = json.loads(canonical_bytes(flagship_recipe()))
    body["study_id"] = "g17_qualification_%s_%s%s" % (
        mode, duration, ("_" + suffix) if suffix else "")
    body["mode"] = mode
    body["title"] = "G17 qualification: %s / %s" % (
        mode.replace("_", " "), duration.replace("_", " "))
    body["windows"] = [row for row in body["windows"] if row["name"] == duration]
    body["confirmation"]["held_out_partition"] = "g17_qualification_heldout_%s_%s%s" % (
        mode, duration, ("_" + suffix) if suffix else "")

    # The order-book fixture is a real content-addressed record.  Binding its digest here is
    # what lets metadata plan the complete quartet without pretending it has a public archive.
    for observation in body["observations"]:
        observation["adapter"]["parameters"]["source_binding"] = "benchmark_known_answer"
        if observation["domain"] == "order_book":
            observation["acquisition"]["identity"]["content_sha256"] = (
                known_answer_native("order_book").content_sha256)

    if mode == "calendar_aligned":
        body["scale_normalization"] = None
        body["family"]["relationships"] = ["co_occurrence"]
        body["nulls"] = [{"name": "domain_preserving_shift",
                          "method": "independent_native_clock_shift",
                          "replications": 200, "parameters": {}}]
        body["notes"]["claim_boundary"] = "co-occurrence only"
    else:
        body["scale_normalization"] = {
            "method": "native_scale_ratio", "reference": "within_domain"}
        body["family"]["relationships"] = ["shape_recurrence"]
        body["nulls"] = [{"name": "scale_partner_reassignment",
                          "method": "scale_partner_reassignment",
                          "replications": 200, "parameters": {}}]
        body["notes"]["claim_boundary"] = (
            "shape recurrence only; no simultaneity, precedence or causality")
    body["notes"]["qualification_record_kind"] = RECORD_KIND
    body["notes"]["duration_selected_before_results"] = True
    return CrossDomainExperimentSpec.parse_obj(body)


def _gate(gate_id: str, title: str, status: str, detail: str,
          *, blocking: bool = True) -> Dict[str, Any]:
    return {"gate_id": gate_id, "title": title, "status": status,
            "blocking": blocking and status != "PASS", "detail": detail}


def qualification_plan() -> Dict[str, Any]:
    """The complete gate before anything is executed; omissions are impossible to hide."""
    cells = []
    for duration in DURATIONS:
        for mode in MODES:
            spec = qualification_manifest(duration, mode)
            window = spec.windows[0]
            cells.append({
                "cell_id": "%s:%s" % (duration, mode),
                "duration": duration,
                "mode": mode,
                "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
                "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
                "manifest_sha256": manifest_sha256(spec),
                "family_correction": spec.correction,
                "record_kind": RECORD_KIND,
                "status": "NOT_RUN",
            })
    gates = [
        _gate("offline_matrix", "Six-cell known-answer apparatus matrix", "NOT_RUN",
              "Run all three frozen durations in both comparison modes."),
        _gate("restart_recovery", "Single remote-failure restart recovery", "NOT_RUN",
              "Resume the same manifest after a process-boundary reload and retry only the "
              "failed component."),
        _gate("browser_no_glue", "Clean-browser no-glue path", "NOT_RUN",
              "Must be measured by a rendered browser test through visible controls."),
        _gate("synthetic_fifth_adapter", "Synthetic fifth-adapter no-edit test", "NOT_RUN",
              "Must be measured by the extension conformance test and source-edit audit."),
        _gate("calendar_calibration", "Calendar null calibration and planted power", "NOT_RUN",
              "The frozen family calibration must run separately from orchestration."),
        _gate("scale_shape_calibration", "Scale/shape null calibration and planted power",
              "NOT_IMPLEMENTED", "No registered scale/shape mining calibration currently "
              "produces a scientific statistic; rehearsal cannot satisfy this gate."),
        _gate("live_sources", "Four-domain live-source tail", "NOT_RUN",
              "Network remains opt-in and archive coverage/refusals require a separately "
              "dated live record."),
    ]
    result = {
        "schema": SCHEMA,
        "qualification_sha256": "",
        "verdict": "NOT_RELEASEABLE",
        "record_kind": RECORD_KIND,
        "matrix": cells,
        "gates": gates,
        "scientist_actions": {
            "status": "NOT_MEASURED",
            "definition": "visible researcher actions from a clean browser session",
            "adapter_specific_framework_edits": "NOT_MEASURED",
            "refusal_explanation_time_seconds": "NOT_MEASURED",
        },
        "claim_boundary": (
            "This plan and its offline rehearsal qualify apparatus behaviour only. They are "
            "not acquired observations, scientific results, evidence, replication or claims."),
    }
    result["qualification_sha256"] = _digest({**result, "qualification_sha256": ""})
    return result


def _cell_result(spec: CrossDomainExperimentSpec, root: Path) -> Dict[str, Any]:
    preflight = preflight_manifest(spec)
    window = spec.windows[0]
    if preflight["refusals"]:
        # No run is opened for a plan the registered declarations refuse. Opening one would
        # manufacture a run identity for an experiment that may not be conducted, and a later
        # reader could not tell a refused plan from an unexecuted one.
        return {
            "cell_id": "%s:%s" % (window.name, spec.mode),
            "duration": window.name,
            "mode": spec.mode,
            "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
            "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
            "manifest_sha256": manifest_sha256(spec),
            "preflight_status": preflight["status"],
            "family_correction": spec.correction,
            "record_kind": RECORD_KIND,
            "refusals": list(preflight["refusals"]),
            "status": "REFUSED",
        }
    run = RunStore(root).open(spec)
    receipt = run.execute(build_suite("fixture_dry_run", run=run))
    exported = publish_bundle(run, root)["bundle"]
    replay = verify_bundle(exported)
    checks = {
        "no_preflight_refusal": not preflight["refusals"],
        "complete_run": receipt["state"] == "COMPLETE",
        "same_manifest_everywhere": (
            preflight["manifest_sha256"] == manifest_sha256(spec)
            == exported["run_identity"]["manifest_sha256"]
            == replay["run_receipt"]["manifest_sha256"]),
        "receipt_integrity_verified": replay["integrity"] == "VERIFIED",
        "fixture_not_measured": (
            exported["results"]["measured"] is False
            and exported["results"]["artefacts"] == {}),
        "no_evidence_or_claim_promotion": (
            exported["evidence_handoff"]["automatic_actions"] == []
            and all(row["status"] == "ABSENT"
                    for row in exported["evidence_handoff"]["categories"]
                    if row["category"] in ("measured_results", "registered_hypothesis",
                                           "admitted_evidence", "independent_replication",
                                           "claim_promotion"))),
    }
    return {
        "cell_id": "%s:%s" % (window.name, spec.mode),
        "duration": window.name,
        "mode": spec.mode,
        "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
        "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
        "manifest_sha256": manifest_sha256(spec),
        "run_id": receipt["run_id"],
        "bundle_sha256": exported["bundle_sha256"],
        "preflight_status": preflight["status"],
        "family_correction": spec.correction,
        "record_kind": RECORD_KIND,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _recovery_result(root: Path) -> Dict[str, Any]:
    spec = qualification_manifest("week", "calendar_aligned", suffix="recovery")
    first_store = RunStore(root)
    run = first_store.open(spec)
    if run.state not in ("FAILED", "COMPLETE"):
        run.execute(build_suite("fixture_single_remote_failure", run=run))

    # A fresh store and loaded run is the process-boundary contract: the worker's memory is gone.
    resumed = RunStore(root).load(run.run_id)
    if resumed.state == "FAILED":
        resumed.retry(build_suite("fixture_single_remote_failure", run=resumed))
    final = resumed.receipt()
    failed_component = "tess_lightcurve"
    attempts_before = {item.domain: (1 if item.domain == failed_component else 1)
                       for item in spec.observations}
    attempts_after = {item.domain: resumed.attempts("ACQUIRING", item.domain)
                      for item in spec.observations}
    transitions = [event.get("to_state") for event in resumed.events
                   if event.get("kind") == "transition"]
    retries = [event for event in resumed.events if event.get("kind") == "retry"]
    checks = {
        "first_attempt_failed_operationally": "FAILED" in transitions,
        "same_run_after_restart": final["run_id"] == run.run_id,
        "retry_completed": final["state"] == "COMPLETE",
        "retry_named_only_failed_component": (
            len(retries) == 1 and retries[0].get("components") == [failed_component]),
        "only_failed_component_retried": (
            attempts_after == {name: 2 if name == failed_component else 1
                               for name in attempts_before}),
        "replay_verified_after_recovery": (
            verify_bundle(publish_bundle(resumed, root)["bundle"])["integrity"] == "VERIFIED"),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "run_id": final["run_id"], "failed_component": failed_component,
            "attempts_before_restart": attempts_before,
            "attempts_after_retry": attempts_after, "checks": checks}


def execute_offline_qualification(root: Path) -> Dict[str, Any]:
    """Execute only the deterministic gates and return an honestly blocked release record."""
    root = Path(root)
    matrix = [_cell_result(qualification_manifest(duration, mode), root)
              for duration in DURATIONS for mode in MODES]
    recovery = _recovery_result(root)
    plan = qualification_plan()
    gates = []
    for row in plan["gates"]:
        if row["gate_id"] == "offline_matrix":
            passed = sum(cell["status"] == "PASS" for cell in matrix)
            refused = [cell for cell in matrix if cell["status"] == "REFUSED"]
            if passed == len(matrix):
                status = "PASS"
            elif refused and passed + len(refused) == len(matrix):
                # Not a broken apparatus: a declared scientific refusal the matrix cannot
                # overrule. It blocks release exactly as a failure does, and says why.
                status = "REFUSED"
            else:
                status = "FAIL"
            detail = "%d of %d deterministic cells passed." % (passed, len(matrix))
            if refused:
                reasons = sorted({str(item.get("reason", "")) for cell in refused
                                  for item in cell["refusals"]})
                detail += " %d refused before execution: %s" % (
                    len(refused), " ".join(reasons))
            gates.append(_gate(row["gate_id"], row["title"], status, detail))
        elif row["gate_id"] == "restart_recovery":
            gates.append(_gate(row["gate_id"], row["title"], recovery["status"],
                               "One acquisition component timed out; a fresh store resumed the "
                               "same run and retried only that component."))
        else:
            gates.append(row)
    record = {**plan, "qualification_sha256": "", "matrix": matrix, "gates": gates,
              "recovery": recovery}
    record["verdict"] = ("RELEASEABLE" if all(row["status"] == "PASS" for row in gates)
                         else "NOT_RELEASEABLE")
    record["qualification_sha256"] = _digest(record)
    return record


def verify_qualification_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the record's outer seal without turning its unrun gates into passes."""
    supplied = str(record.get("qualification_sha256", ""))
    body = dict(record)
    body["qualification_sha256"] = ""
    expected = _digest(body)
    if supplied != expected:
        raise ValueError("qualification_sha256 does not match the complete qualification record")
    if record.get("verdict") == "RELEASEABLE" and any(
            row.get("status") != "PASS" for row in record.get("gates", [])):
        raise ValueError("a RELEASEABLE qualification record contains a gate that did not pass")
    return {"integrity": "VERIFIED", "qualification_sha256": supplied,
            "verdict": record.get("verdict"), "gates": list(record.get("gates", []))}


__all__ = ["DURATIONS", "MODES", "RECORD_KIND", "SCHEMA", "execute_offline_qualification",
           "qualification_manifest", "qualification_plan", "verify_qualification_record"]
