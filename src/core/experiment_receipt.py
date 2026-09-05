"""TG17.9 immutable experiment export, methods report and evidence handoff.

The orchestrator's ``receipt()`` is a current projection of an append-only journal.  This module
creates the different object a reviewer needs: a sealed, portable bundle which contains every
scientific identity needed to reconstruct that projection without the browser, the run store or
the adapter registry that happens to be installed later.

Import is deliberately read-only.  A replay bundle can reconstruct an audit view; it cannot
create an EvidenceBundle, promote a claim, or repopulate a run journal.  Those are separate acts
with separate ledgers, and collapsing them would make "I can replay it" mean "I accepted it".
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.adapters import register_all_adapters
from src.core.errors import InvalidParameterError
from src.core.experiment_adapter import EXPERIMENT_ADAPTERS, adapter_for
from src.core.experiment_manifest import (CrossDomainExperimentSpec, canonical_bytes,
                                          manifest_sha256, preflight_manifest)
from src.core.experiment_run import (TRANSITIONS, WORK_STAGES, ExperimentRun, run_identity,
                                     stage_components)
from src.core.publication import publish_new_bytes


BUNDLE_SCHEMA = "cross-domain-experiment-bundle/v1"
REPORT_SCHEMA = "cross-domain-methods-report/v1"
CAPABILITY_SCHEMA = "cross-domain-receipt-capabilities/v1"
SOFTWARE_VERSION = "spectralearth-ed-dev/tg17.9"

CLAIM_BOUNDARY = (
    "This is an experiment receipt: it proves which declared plan the recorded runner executed. "
    "It is not automatically a study, finding, EvidenceBundle, publication, independent "
    "replication or claim promotion."
)


def _json_numbers(value: Any) -> Any:
    """Normalise the JSON number model before hashing.

    JavaScript has one number type: parsing ``1.0`` and serialising it again produces ``1``.
    Python distinguishes those spellings even though JSON does not attach a type distinction to
    them.  Without this normalisation a bundle exported by the API failed integrity after the
    browser merely read and re-emitted it (D81).  Non-integral numbers retain their value; large
    content identities are strings and never pass through a JSON number.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numbers(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_numbers(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(mapping: Mapping[str, Any], fields: Sequence[str], label: str) -> Dict[str, Any]:
    actual, expected = set(mapping), set(fields)
    if actual != expected:
        raise InvalidParameterError(
            label, sorted(actual),
            "exactly the registered fields; missing=%s unexpected=%s"
            % (sorted(expected - actual), sorted(actual - expected)))
    return dict(mapping)


def execution_environment() -> Dict[str, Any]:
    """A small, serialisable environment snapshot captured in the run journal at freeze time."""
    packages: Dict[str, str] = {}
    for name in ("fastapi", "numpy", "pydantic", "scipy", "torch", "xarray"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "software_version": SOFTWARE_VERSION,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def capture_archival_context(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """Capture registry-derived contracts when the manifest becomes immutable."""
    register_all_adapters()
    return {"schema": "experiment-archival-context/v1",
            "preflight": preflight_manifest(spec),
            "sources": _source_contracts(spec),
            "adapters": _adapter_contracts(spec),
            "environment": execution_environment()}


# This registry is the documentation contract.  The capability endpoint and both UIs render it,
# and the documentation test compares it with every top-level bundle field.  Adding a field to an
# export without explaining it therefore fails before it can become an invisible trust claim.
RECEIPT_FIELDS: Tuple[Dict[str, str], ...] = (
    {"name": "schema", "label": "Bundle schema", "meaning": "Version of the portable replay contract."},
    {"name": "bundle_sha256", "label": "Bundle digest", "meaning": "SHA-256 recomputed over every other bundle field."},
    {"name": "run_identity", "label": "Run identity", "meaning": "Content address derived only from the manifest."},
    {"name": "manifest", "label": "Exact manifest", "meaning": "The byte-stable scientific configuration shared by API, runner and UI."},
    {"name": "preflight", "label": "Coverage decision", "meaning": "Metadata-only coverage, family, null and resource decision bound before execution."},
    {"name": "sources", "label": "Sources", "meaning": "Declared acquisition identities and source plans for every domain."},
    {"name": "adapters", "label": "Adapter contracts", "meaning": "Versioned declarations and definition digests used for native-to-structural translation."},
    {"name": "artefacts", "label": "Artefact lineage", "meaning": "Native, canonical, mining and confirmation digests, separated by scientific role."},
    {"name": "inference", "label": "Inference declaration", "meaning": "Complete family, nulls, correction, alpha, confirmation policy and seeds."},
    {"name": "environment", "label": "Execution environment", "meaning": "Software and runtime identity captured when the manifest was frozen."},
    {"name": "events", "label": "Stage events", "meaning": "The exact append-only event sequence from which run state is folded."},
    {"name": "run_receipt", "label": "Run projection", "meaning": "Terminal state, decisions, missing components and bounded work reconstructed from the journal."},
    {"name": "results", "label": "Result identities", "meaning": "Digests of measured result artefacts, or an explicit statement that none exist."},
    {"name": "refusals", "label": "Refusals and absences", "meaning": "Preflight, component and terminal limitations that must travel with the result."},
    {"name": "evidence_handoff", "label": "Evidence handoff", "meaning": "Reviewable next actions and absent evidence categories; never an automatic admission."},
    {"name": "methods_report", "label": "Methods report", "meaning": "Digest and text of the scientist-readable methods and limitations account."},
    {"name": "claim_boundary", "label": "Claim boundary", "meaning": "What completion and replay do not establish."},
)

BUNDLE_FIELDS: Tuple[str, ...] = tuple(item["name"] for item in RECEIPT_FIELDS)


def _environment_from_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    freezes = [event for event in events if event.get("kind") == "freeze"]
    if not freezes or "environment" not in freezes[-1]:
        return {"software_version": "NOT_RECORDED",
                "limitation": "This run predates TG17.9 environment capture."}
    return dict(freezes[-1]["environment"])


def _archival_context(run: ExperimentRun) -> Dict[str, Any]:
    freezes = [event for event in run.events if event.get("kind") == "freeze"]
    if freezes and freezes[-1].get("archival_context"):
        return dict(freezes[-1]["archival_context"])
    # Compatibility for TG17.6 runs frozen before the archival context existed. Re-derivation is
    # accepted only when it hashes to the preflight decision recorded in their journal.
    context = capture_archival_context(run.spec)
    preflights = [event for event in run.events if event.get("kind") == "preflight"]
    recorded = preflights[-1].get("preflight_sha256") if preflights else None
    legacy_digest = hashlib.sha256(json.dumps(
        context["preflight"], sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()
    if recorded != legacy_digest:
        raise InvalidParameterError(
            "archival_context", "NOT_RECORDED",
            "a freeze-time context, or a current preflight byte-identical to the digest this "
            "legacy run recorded. The installed adapters or planner have changed, so exporting "
            "their current description as historical fact is refused")
    context["legacy_rederived"] = True
    context["limitation"] = ("This TG17.6 run predates freeze-time archival context; its current "
                             "contracts were admitted only because preflight identity is unchanged.")
    return context


def _artefact_roles(receipt: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {
        "native": {}, "canonical": {}, "mining": {}, "confirmation": {}}
    roles = {"ACQUIRING": "native", "TRANSLATING": "canonical",
             "MINING": "mining", "CONFIRMING": "confirmation"}
    for key, digest in sorted(dict(receipt.get("artefacts", {})).items()):
        stage, _, component = key.partition("/")
        if stage in roles:
            groups[roles[stage]][component] = digest
    return groups


def _source_contracts(spec: CrossDomainExperimentSpec) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for observation in spec.observations:
        adapter = adapter_for(observation.adapter.adapter_id)
        parameters = adapter.resolve(observation.adapter.parameters)
        plan = adapter.plan_acquisition(parameters, dict(observation.acquisition.identity))
        rows.append({
            "domain": observation.domain,
            "acquisition_identity": observation.acquisition.dict(),
            "source_plan": plan.describe(),
        })
    return rows


def _adapter_contracts(spec: CrossDomainExperimentSpec) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for observation in spec.observations:
        adapter = adapter_for(observation.adapter.adapter_id)
        if adapter.adapter_version != observation.adapter.adapter_version:
            raise InvalidParameterError(
                "adapter_version", observation.adapter.adapter_version,
                "the registered version %s for %s; exporting against another installed adapter "
                "would describe a translation this run did not declare"
                % (adapter.adapter_version, adapter.adapter_id))
        rows.append({"domain": observation.domain,
                     "manifest_binding": observation.adapter.dict(),
                     "contract": adapter.describe(),
                     "translator_config": adapter.config(observation.adapter.parameters)})
    return rows


def evidence_handoff(receipt: Mapping[str, Any], results: Mapping[str, Any]) -> Dict[str, Any]:
    has_results = bool(results.get("artefacts"))
    complete = receipt.get("state") == "COMPLETE"
    categories = [
        {"category": "declared_plan", "status": "PRESENT", "source": "manifest"},
        {"category": "execution_lineage", "status": "PRESENT", "source": "events"},
        {"category": "measured_results", "status": "PRESENT" if has_results else "ABSENT",
         "source": "results.artefacts" if has_results else None},
        {"category": "registered_hypothesis", "status": "ABSENT", "source": None},
        {"category": "admitted_evidence", "status": "ABSENT", "source": None},
        {"category": "independent_replication", "status": "ABSENT", "source": None},
        {"category": "claim_promotion", "status": "ABSENT", "source": None},
    ]
    return {
        "run_complete": complete,
        "eligible_actions": (["open_reviewable_study_draft", "inspect_findings_read_only"]
                             if complete else []),
        "automatic_actions": [],
        "categories": categories,
        "proposed_study_id": str(receipt.get("study_id", "")),
        "claim_boundary": "A handoff proposes where this receipt may be reviewed. It writes no evidence and moves no rung.",
    }


def _refusals(preflight: Mapping[str, Any], receipt: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = [{"kind": "preflight", "detail": str(item)}
            for item in preflight.get("refusals", [])]
    rows.extend({"kind": "component", **dict(item)}
                for item in receipt.get("missing_components", []))
    if receipt.get("state") in ("REFUSED", "FAILED", "CANCELLED"):
        history = list(receipt.get("history", []))
        rows.append({"kind": "terminal_state", "state": receipt.get("state"),
                     "detail": history[-1].get("reason", "") if history else ""})
    return rows


def render_methods_report(body: Mapping[str, Any]) -> str:
    """Render reviewable Markdown from bundle fields, never from a second configuration."""
    manifest = body["manifest"]
    receipt = body["run_receipt"]
    inference = body["inference"]
    sources = body["sources"]
    result_count = len(body["results"]["artefacts"])
    refusal_count = len(body["refusals"])
    lines = [
        "# Cross-domain experiment methods and limitations",
        "",
        "## Identity",
        "",
        "- Study: `%s`" % receipt["study_id"],
        "- Run: `%s`" % receipt["run_id"],
        "- Manifest SHA-256: `%s`" % receipt["manifest_sha256"],
        "- State: **%s**" % receipt["state"],
        "- Software: `%s`" % body["environment"].get("software_version", "NOT_RECORDED"),
        "",
        "## Declared method",
        "",
        "The experiment used **%s** comparison over %d declared domain observations and %d "
        "window(s). The complete declared family contained %s members. Inference used `%s` "
        "correction at alpha %s; the null definitions and their parameters are retained in the "
        "machine-readable bundle."
        % (manifest["mode"], len(manifest["observations"]), len(manifest["windows"]),
           body["preflight"]["family"].get("family_size", "the manifest-defined number"),
           inference["correction"], inference["alpha"]),
        "",
        "## Sources and translation",
        "",
    ]
    for source in sources:
        plan = source["source_plan"]
        lines.append("- **%s:** `%s` (%s); network used by the declared plan: %s."
                     % (source["domain"], plan["source_id"], plan["access"],
                        str(plan["network_used"]).lower()))
    lines.extend([
        "",
        "Each native record and canonical trajectory remains separately content-addressed. Raw "
        "magnitudes from different domains were not placed on a shared quantitative axis.",
        "",
        "## Results and limitations",
        "",
        "The receipt identifies %d mining/confirmation result artefact(s) and records %d refusal "
        "or absence item(s). An artefact digest establishes identity, not scientific validity."
        % (result_count, refusal_count),
        "",
        CLAIM_BOUNDARY,
        "",
        "No publication-readiness or independent-replication claim is made by this report.",
    ])
    return "\n".join(lines) + "\n"


def build_bundle(run: ExperimentRun) -> Dict[str, Any]:
    """Build and self-hash a portable export for a completed run."""
    if run.state != "COMPLETE":
        raise InvalidParameterError(
            "run.state", run.state,
            "COMPLETE. A failed, refused, cancelled or in-progress run has a live receipt and "
            "limitations, but it is not a completed experiment export")
    register_all_adapters()
    spec, receipt, events = run.spec, run.receipt(), run.events
    context = _archival_context(run)
    preflight = context["preflight"]
    roles = _artefact_roles(receipt)
    stage_results = {**{"MINING/%s" % k: v for k, v in roles["mining"].items()},
                     **{"CONFIRMING/%s" % k: v for k, v in roles["confirmation"].items()}}
    fixture = bool(events) and all(
        str(event.get("outcome", {}).get("detail", "")).startswith("fixture:")
        for event in events if event.get("kind") in ("step", "step_reused"))
    results = {"measured": bool(stage_results) and not fixture,
               "artefacts": {} if fixture else stage_results,
               "fixture_artefacts": stage_results if fixture else {},
               "interpretation": (("The registered rehearsal produced addressed stage markers "
                                   "but opened no measurement value; they are fixture artefacts, "
                                   "not measured results.") if fixture else
                                  ("Result artefact identities are present. Values remain in the "
                                   "content-addressed artefacts; this index does not reinterpret them."))}
    body: Dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "run_identity": run_identity(spec),
        "manifest": json.loads(canonical_bytes(spec)),
        "preflight": preflight,
        "sources": context["sources"],
        "adapters": context["adapters"],
        "artefacts": roles,
        "inference": {"family": spec.family.dict(), "nulls": [item.dict() for item in spec.nulls],
                      "correction": spec.correction, "alpha": spec.alpha,
                      "confirmation": spec.confirmation.dict(), "seeds": dict(spec.seeds)},
        "environment": context["environment"],
        "events": events,
        "run_receipt": receipt,
        "results": results,
        "refusals": _refusals(preflight, receipt),
        "evidence_handoff": evidence_handoff(receipt, results),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report = render_methods_report(body)
    body["methods_report"] = {"schema": REPORT_SCHEMA, "format": "text/markdown",
                              "sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
                              "text": report}
    body["bundle_sha256"] = _digest(body)
    return body


def _replay_receipt(spec: CrossDomainExperimentSpec,
                    events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Fold exported events without a filesystem, reproducing the TG17.6 receipt projection."""
    identity = run_identity(spec)
    state = "DRAFT"
    outcomes: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    decisions: List[Mapping[str, Any]] = []
    history: List[Dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("schema") != "experiment-run/v1":
            raise InvalidParameterError("events[%d].schema" % index, event.get("schema"),
                                        "experiment-run/v1")
        if event.get("run_sha256") != identity["run_sha256"]:
            raise InvalidParameterError("events[%d].run_sha256" % index,
                                        event.get("run_sha256"), identity["run_sha256"])
        kind = event.get("kind")
        if kind == "transition":
            if event.get("from_state") != state or event.get("to_state") not in TRANSITIONS[state]:
                raise InvalidParameterError(
                    "events[%d].transition" % index,
                    "%s -> %s" % (event.get("from_state"), event.get("to_state")),
                    "a transition from replay state %s to one of %s"
                    % (state, list(TRANSITIONS[state])))
            history.append({"at": event["at"], "from": state, "to": event["to_state"],
                            "reason": event.get("reason", "")})
            state = event["to_state"]
        elif kind in ("step", "step_reused"):
            stage, component = event.get("stage"), event.get("component")
            if stage not in WORK_STAGES or component not in stage_components(spec, stage):
                raise InvalidParameterError("events[%d].step" % index,
                                            "%s/%s" % (stage, component),
                                            "a component declared by the exact manifest")
            outcomes[(stage, component)] = event
        elif kind == "stage_decision":
            decisions.append(event)
    missing = [{"stage": stage, "component": component,
                "status": event["outcome"]["status"],
                "detail": event["outcome"]["detail"],
                "remediation": event["outcome"]["remediation"]}
               for (stage, component), event in sorted(outcomes.items())
               if event["outcome"]["status"] != "COMPLETE"]
    artefacts = {"%s/%s" % (stage, component): event["outcome"]["artifact_sha256"]
                 for (stage, component), event in sorted(outcomes.items())
                 if event["outcome"]["status"] == "COMPLETE"}
    total = sum(len(stage_components(spec, stage)) for stage in WORK_STAGES)
    completed = sum(1 for event in outcomes.values()
                    if event["outcome"]["status"] == "COMPLETE")
    bounded = {"completed_steps": completed, "total_steps": total,
               "fraction": round(completed / float(total), 6) if total else 0.0,
               "bytes_read": sum(int(event["outcome"].get("bytes_read", 0))
                                 for event in outcomes.values())}
    return {"schema": "experiment-run/v1", "run_id": identity["run_id"],
            "run_sha256": identity["run_sha256"],
            "manifest_sha256": identity["manifest_sha256"], "study_id": spec.study_id,
            "state": state, "history": history, "artefacts": artefacts,
            "missing_components": missing,
            "stage_decisions": [{"stage": event["stage"], "verdict": event["verdict"],
                                 "reason": event["reason"]} for event in decisions],
            "bounded_work": bounded,
            "coverage_policy": {"requirement": spec.coverage_policy.requirement,
                                "minimum_fraction": spec.coverage_policy.minimum_fraction},
            "confirmation": spec.confirmation.dict(), "events": len(events),
            "claim_boundary": "A completed run is an executed plan, not admitted evidence."}


def verify_bundle(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Strictly verify a bundle and reconstruct its immutable audit projection."""
    record = _exact(value, BUNDLE_FIELDS, "experiment replay bundle")
    claimed = record.pop("bundle_sha256")
    if record.get("schema") != BUNDLE_SCHEMA:
        raise InvalidParameterError("schema", record.get("schema"), BUNDLE_SCHEMA)
    actual = _digest(record)
    if claimed != actual:
        raise InvalidParameterError("bundle_sha256", claimed,
                                    "the SHA-256 recomputed from every exported field (%s)" % actual)
    spec = CrossDomainExperimentSpec.parse_obj(record["manifest"])
    identity = run_identity(spec)
    if record["run_identity"] != identity:
        raise InvalidParameterError("run_identity", record["run_identity"],
                                    "the identity derived from the exact manifest")
    receipt = record["run_receipt"]
    if receipt.get("manifest_sha256") != manifest_sha256(spec):
        raise InvalidParameterError("run_receipt.manifest_sha256",
                                    receipt.get("manifest_sha256"), manifest_sha256(spec))
    replayed_receipt = _replay_receipt(spec, record["events"])
    if receipt != replayed_receipt:
        raise InvalidParameterError(
            "run_receipt", receipt,
            "the receipt reconstructed from the exported append-only events. Embedded state, "
            "artefacts or conclusions may not disagree with replay")
    report = record["methods_report"]
    report_sha = hashlib.sha256(report.get("text", "").encode("utf-8")).hexdigest()
    if report.get("sha256") != report_sha:
        raise InvalidParameterError("methods_report.sha256", report.get("sha256"), report_sha)
    reconstructed = dict(record, bundle_sha256=claimed)
    return {"schema": BUNDLE_SCHEMA, "integrity": "VERIFIED", "bundle_sha256": claimed,
            "manifest": record["manifest"], "run_identity": identity,
            "run_receipt": receipt, "results": record["results"],
            "refusals": record["refusals"], "evidence_handoff": record["evidence_handoff"],
            "methods_report": report, "claim_boundary": CLAIM_BOUNDARY,
            "bundle": reconstructed}


def publish_bundle(run: ExperimentRun, root: Path) -> Dict[str, Any]:
    """Publish canonical bundle and report once, returning their immutable addresses."""
    bundle = build_bundle(run)
    digest = bundle["bundle_sha256"]
    directory = Path(root) / "exports"
    bundle_path = directory / (digest + ".json")
    report_path = directory / (bundle["methods_report"]["sha256"] + ".md")
    reused = bundle_path.exists() and report_path.exists()
    payload = _canonical(bundle)
    for path, raw, label in ((bundle_path, payload, "experiment replay bundle"),
                             (report_path, bundle["methods_report"]["text"].encode("utf-8"),
                              "experiment methods report")):
        try:
            publish_new_bytes(path, raw, label)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise InvalidParameterError(path.name, "existing bytes",
                                            "the bytes already published at this content address")
    return {"bundle": bundle, "bundle_path": str(bundle_path), "report_path": str(report_path),
            "reused": reused}


def capability_snapshot() -> Dict[str, Any]:
    """The generated trust surface: operations, adapters, refusals and every receipt field."""
    register_all_adapters()
    adapters = [entry.value.describe() for entry in EXPERIMENT_ADAPTERS.entries()]
    return {
        "schema": CAPABILITY_SCHEMA,
        "software_version": SOFTWARE_VERSION,
        "operations": [
            {"name": "export_completed_run", "effect": "publishes immutable bundle and report", "writes_evidence": False},
            {"name": "verify_and_replay_bundle", "effect": "reconstructs a read-only audit projection", "writes_evidence": False},
            {"name": "open_reviewable_study_draft", "effect": "navigates to a separate evidence workflow", "writes_evidence": False},
        ],
        "adapters": adapters,
        "refusals": [
            {"name": "non_complete_export", "reason": "Only a COMPLETE run can be sealed as a completed experiment."},
            {"name": "digest_mismatch", "reason": "A changed bundle, manifest or methods report is refused."},
            {"name": "automatic_evidence_admission", "reason": "Replay and handoff never create evidence or move a claim rung."},
            {"name": "publication_readiness", "reason": "A methods report is reviewable documentation, not certification or replication."},
        ],
        "receipt_fields": [dict(item) for item in RECEIPT_FIELDS],
        "lineage": ["manifest", "preflight", "run journal", "content-addressed artefacts",
                    "replay bundle", "optional evidence workflow"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = ["BUNDLE_FIELDS", "BUNDLE_SCHEMA", "CAPABILITY_SCHEMA", "CLAIM_BOUNDARY",
           "RECEIPT_FIELDS", "REPORT_SCHEMA", "SOFTWARE_VERSION", "build_bundle",
           "capability_snapshot", "capture_archival_context", "evidence_handoff", "execution_environment",
           "publish_bundle", "render_methods_report", "verify_bundle"]
