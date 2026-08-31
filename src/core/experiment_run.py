"""TG17.6 Content-addressed, resumable experiment orchestrator.

One state machine executes a frozen manifest:

    DRAFT -> PREFLIGHTED -> FROZEN -> ACQUIRING -> TRANSLATING -> MINING -> CONFIRMING -> COMPLETE

with `REFUSED`, `FAILED` and `CANCELLED` as the explicit ways out. The states are a table, not a
sequence of `if` branches, because the interesting question about a long run is never "what
happens next" - it is "what was allowed to happen next, and who says so".

**Why an orchestrator is a scientific component and not plumbing.** A four-domain study is a long
job over remote archives. Long jobs get killed: a laptop sleeps, a token expires, a browser is
refreshed, TESS times out. Every one of those interruptions offers the same tempting recovery -
start again, and quietly run whatever is available this time. That recovery silently substitutes a
different experiment for the declared one, and nothing downstream can tell, because the run that
finishes looks exactly like a run that was always going to finish that way. Three properties are
what stop it:

1.  **Run identity is the manifest.** `run_sha256` is a content address over the schema and the
    manifest digest and nothing else - not the wall clock, not a UUID, not the machine. Executing
    an identical manifest twice therefore *is* the same run: the second execution resumes the
    first rather than starting a rival copy of it. A run identity carrying a timestamp would make
    "did I already acquire this?" unanswerable, which is the same question as "am I about to
    re-download 40 GB and call it a second result?".

2.  **Every step is content-addressed and idempotent.** A step key is the digest of the run, the
    stage, the component and the digests of that step's declared inputs. A completed step is
    published immutably (`src.core.publication.publish_new_bytes`) and replayed from disk rather
    than recomputed, so a resumed run does no duplicate network acquisition. Because the key
    contains the input digests, it is also the drift check: if an upstream artefact changes, the
    downstream key changes and the step re-runs instead of pairing a new native record with a
    stale translation.

3.  **A retry may not author a new plan.** `retry` re-executes the components that failed
    *operationally* and refuses a manifest whose digest differs from the frozen one. The
    scientifically interesting failure - a permanent coverage refusal - is not retryable at all:
    it produces `REFUSED`, and the remedy is `editable_copy`, which writes a new mutable draft and
    leaves the frozen run exactly as it stands. Editing a frozen run in place is how a study
    acquires a plan nobody ever declared.

**Partial acquisition is a policy decision, never a default.** When a component is missing, the
frozen `CoveragePolicy` alone decides. `complete_required` refuses; `partial_permitted` admits the
run only while the completed fraction stays at or above the declared minimum, and the receipt
lists every missing component by name either way. The orchestrator has no opinion of its own here,
which is the point: silently running a smaller experiment is the failure this stage exists to make
impossible.

**Progress cannot leak an unopened result, by construction rather than by discipline.** A
`ComponentOutcome` has nowhere to put a measurement value. It carries a status, a digest, a bounded
work estimate and a remediation string, so a progress feed watched during `MINING` can report that
mining is happening and cannot report what it found. A field that could hold a p-value would
eventually hold one.

**What this slice does not do.** It ships the state machine, not the domain workers: stage workers
are supplied by the caller, and a stage with no registered worker refuses by name. Live acquisition
and translation arrive with the slices that own them. Confirmation openings are recorded here
against the run that opened them, and the full `src.core.preregistration` seal - partition
identity, sealed digests, `HeldOutLedger` - remains the authority when a real confirmatory pass is
executed; this module's job is to refuse a *second* opening of a partition a different run already
spent.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UserInputError
from src.core.experiment_manifest import (CrossDomainExperimentSpec, canonical_bytes,
                                          manifest_sha256, preflight_manifest)
from src.core.publication import publish_new_bytes


SCHEMA = "experiment-run/v1"

#: Every state the machine can occupy. `FAILED` is deliberately not terminal: an operational
#: failure is the one kind a retry may legitimately clear.
STATES: Tuple[str, ...] = ("DRAFT", "PREFLIGHTED", "FROZEN", "ACQUIRING", "TRANSLATING",
                           "MINING", "CONFIRMING", "COMPLETE", "REFUSED", "FAILED", "CANCELLED")

#: The stages that do work, in the order they are executed.
WORK_STAGES: Tuple[str, ...] = ("ACQUIRING", "TRANSLATING", "MINING", "CONFIRMING")

#: States nothing can leave. `REFUSED` is here because the remedy for a refusal is a new draft,
#: not a nudge to the frozen run that earned it.
TERMINAL_STATES: Tuple[str, ...] = ("COMPLETE", "REFUSED", "CANCELLED")

#: The allowed successors of each state, as data. A transition absent from this table cannot be
#: performed by any code path in this module, which is the property that makes the machine
#: reviewable without reading the code that drives it.
TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "DRAFT": ("PREFLIGHTED", "REFUSED", "CANCELLED"),
    "PREFLIGHTED": ("FROZEN", "REFUSED", "CANCELLED"),
    "FROZEN": ("ACQUIRING", "CANCELLED"),
    "ACQUIRING": ("TRANSLATING", "REFUSED", "FAILED", "CANCELLED"),
    "TRANSLATING": ("MINING", "REFUSED", "FAILED", "CANCELLED"),
    "MINING": ("CONFIRMING", "REFUSED", "FAILED", "CANCELLED"),
    "CONFIRMING": ("COMPLETE", "REFUSED", "FAILED", "CANCELLED"),
    "COMPLETE": (),
    "REFUSED": (),
    "FAILED": ("ACQUIRING", "TRANSLATING", "MINING", "CONFIRMING", "CANCELLED"),
    "CANCELLED": (),
}

#: What a stage worker is allowed to report about one component.
#:
#: The split that matters is `FAILED`/`TIMED_OUT` against `REFUSED`/`MISSING`. The first pair are
#: operational: the archive was unreachable, the request timed out, the process died. They are
#: retryable, and retrying them changes nothing scientific. The second pair are the archive
#: answering the question - this coverage does not exist, this component cannot be produced - and
#: retrying them is just asking the same question again until the answer is convenient.
COMPONENT_STATUSES: Tuple[str, ...] = ("COMPLETE", "MISSING", "REFUSED", "FAILED", "TIMED_OUT")

#: Statuses a retry is permitted to re-execute.
RETRYABLE_STATUSES: Tuple[str, ...] = ("FAILED", "TIMED_OUT")

#: Statuses that end the run scientifically rather than operationally.
REFUSING_STATUSES: Tuple[str, ...] = ("REFUSED",)

RUN_ID = re.compile(r"[0-9a-f]{32}")


class RunStateError(InvalidParameterError):
    """A transition the table does not allow, named together with the ones it does."""

    def __init__(self, state: str, requested: str) -> None:
        allowed = TRANSITIONS.get(state, ())
        super().__init__(
            "run state transition", "%s -> %s" % (state, requested),
            "one of the declared successors of %s, which are %s"
            % (state, ", ".join(allowed) if allowed else "(none - it is a terminal state)"),
            state=state, requested=requested)
        self.state, self.requested = state, requested


class RetryNotPermittedError(UserInputError):
    """A retry of something that did not fail operationally.

    Deliberately not an `InvalidParameterError`: nothing about the caller's parameters is wrong.
    The run is simply in a state where retrying is not the remedy, and saying which one it is in
    is more useful than naming a parameter.
    """


class PlanChangedError(InvalidParameterError):
    """A resumed or retried run was handed a different manifest.

    The digest is the whole of the argument. A run is its manifest; presenting a different one and
    calling it the same run is how a study ends up reporting a plan it never declared.
    """

    def __init__(self, frozen: str, presented: str) -> None:
        super().__init__(
            "manifest_sha256", presented,
            "the manifest this run executes, %s. A retry or a resume may not carry a new "
            "scientific plan: a run is its manifest, and presenting a different one under the "
            "same identity is how a study reports a plan nobody declared. Compose the change as "
            "a new manifest - editable_copy writes one from this run without touching it."
            % frozen, frozen_manifest=frozen)
        self.frozen, self.presented = frozen, presented


class SpentTargetError(InvalidParameterError):
    """A held-out partition another run already opened."""

    def __init__(self, partition: str, opened_by: str) -> None:
        super().__init__(
            "held_out_partition", partition,
            "a partition no run has opened. This one was opened by run %s, and a held-out "
            "partition is confirmatory exactly once: a retry cannot spend it again. Declare a "
            "new partition, or run a generating pass." % opened_by, opened_by=opened_by)
        self.partition, self.opened_by = partition, opened_by


class MissingStageWorkerError(InvalidParameterError):
    """No worker was supplied for a stage the run has reached."""

    def __init__(self, stage: str, supplied: Sequence[str]) -> None:
        super().__init__(
            "workers", sorted(supplied),
            "a worker for stage %s. TG17.6 ships the orchestrator and its state machine, not the "
            "domain workers: src.core.run_workers registers the rehearsal suites, and live "
            "acquisition and translation arrive with the slices that own them." % stage,
            stage=stage)
        self.stage = stage


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def run_identity(spec: CrossDomainExperimentSpec) -> Dict[str, str]:
    """The content address of a run: the schema and the manifest, and nothing else.

    Nothing else is the specification. A clock reading or a random identifier here would make two
    executions of one declared study two different runs, which is precisely the mistake that turns
    a resumed acquisition into a second acquisition.
    """
    manifest = manifest_sha256(spec)
    body = {"schema": SCHEMA, "manifest_sha256": manifest}
    run_sha256 = _digest(body)
    return {"schema": SCHEMA, "manifest_sha256": manifest, "run_sha256": run_sha256,
            "run_id": run_sha256[:32], "study_id": spec.study_id}


def stage_components(spec: CrossDomainExperimentSpec, stage: str) -> Tuple[str, ...]:
    """The units of work a stage is made of, named so a receipt can list what is missing.

    Acquisition and translation are per domain, because that is the granularity at which an
    archive fails and at which a partial result is or is not admissible. Mining and confirmation
    are one component each, because a family corrected in pieces is not the family that was
    declared.
    """
    if stage in ("ACQUIRING", "TRANSLATING"):
        return tuple(observation.domain for observation in spec.observations)
    if stage == "MINING":
        return ("candidate_family",)
    if stage == "CONFIRMING":
        return (("held_out_confirmation",)
                if spec.confirmation.stage == "generate_then_confirm" else ("declared_family",))
    raise InvalidParameterError("stage", stage,
                                "one of the work stages %s" % ", ".join(WORK_STAGES))


def decide_stage(spec: CrossDomainExperimentSpec, stage: str,
                 statuses: Mapping[str, str]) -> Dict[str, Any]:
    """What a stage's component statuses permit, read from the frozen policy alone.

    A pure function of the manifest and the statuses, deliberately: the decision that turns a
    partial acquisition into either a smaller experiment or a refusal is the one a reader will
    want to check without reconstructing a run, and a decision reachable only through a state
    machine is a decision nobody audits.

    The ordering is not arbitrary. A refusal outranks a failure because the archive's answer does
    not become provisional just because something else also timed out, and a failure outranks a
    missing component because "we could not ask" is not "the answer is no".
    """
    components = tuple(stage_components(spec, stage))
    unknown = sorted(set(statuses) - set(components))
    if unknown:
        raise InvalidParameterError(
            "statuses", unknown,
            "components %s declares, which are %s" % (stage, ", ".join(components)))
    missing_status = [name for name in components if name not in statuses]
    if missing_status:
        raise InvalidParameterError(
            "statuses", sorted(statuses),
            "a status for every component of %s. Nothing was reported for %s, and a stage cannot "
            "be decided from a partial report - an unreported component is not a completed one."
            % (stage, ", ".join(missing_status)))
    complete = [name for name, status in statuses.items() if status == "COMPLETE"]
    missing = [name for name, status in statuses.items() if status == "MISSING"]
    refused = [name for name, status in statuses.items() if status in REFUSING_STATUSES]
    failed = [name for name, status in statuses.items() if status in RETRYABLE_STATUSES]
    fraction = len(complete) / float(len(components))
    policy = spec.coverage_policy
    decision: Dict[str, Any] = {
        "stage": stage, "statuses": dict(statuses), "complete": sorted(complete),
        "missing": sorted(missing), "refused": sorted(refused), "failed": sorted(failed),
        "completed_fraction": round(fraction, 6),
        "coverage_requirement": policy.requirement,
        "minimum_fraction": policy.minimum_fraction}
    if refused:
        decision.update(verdict="REFUSED",
                        reason="the source refused %s" % ", ".join(sorted(refused)))
    elif failed:
        decision.update(verdict="FAILED",
                        reason="%s failed operationally and can be retried"
                               % ", ".join(sorted(failed)))
    elif missing and policy.requirement == "complete_required":
        decision.update(verdict="REFUSED",
                        reason="the frozen policy requires complete coverage and %s is missing"
                               % ", ".join(sorted(missing)))
    elif missing and fraction < policy.minimum_fraction:
        decision.update(verdict="REFUSED",
                        reason="the frozen policy permits partial coverage at %.3g and this stage "
                               "reached %.3g" % (policy.minimum_fraction, fraction))
    elif missing:
        decision.update(verdict="ADVANCE_PARTIAL",
                        reason="the frozen policy permits partial coverage at %.3g; %s is missing "
                               "and is named on the receipt"
                               % (policy.minimum_fraction, ", ".join(sorted(missing))))
    else:
        decision.update(verdict="ADVANCE", reason="every declared component completed")
    return decision


@dataclass(frozen=True)
class ComponentOutcome:
    """What one worker reports about one component.

    There is deliberately nowhere here to put a measurement value, a statistic or a p-value. The
    orchestrator hands this straight to the progress feed, and a field that could carry a result
    would carry one out of an unopened stage sooner or later.
    """

    status: str
    artifact_sha256: Optional[str] = None
    detail: str = ""
    remediation: str = ""
    bytes_read: int = 0
    seconds: float = 0.0
    network_used: bool = False

    def __post_init__(self) -> None:
        if self.status not in COMPONENT_STATUSES:
            raise InvalidParameterError(
                "status", self.status,
                "one of the declared component statuses %s" % ", ".join(COMPONENT_STATUSES))
        if self.status == "COMPLETE" and not self.artifact_sha256:
            raise InvalidParameterError(
                "artifact_sha256", self.artifact_sha256,
                "the digest of the artefact a COMPLETE component produced; a completion with no "
                "digest cannot be reused, verified or resumed")
        if self.status in RETRYABLE_STATUSES and not self.remediation:
            raise InvalidParameterError(
                "remediation", self.remediation,
                "what a researcher should do about a %s; an operational failure nobody can act "
                "on is indistinguishable from a refusal" % self.status)
        if self.status in REFUSING_STATUSES and not self.detail:
            raise InvalidParameterError(
                "detail", self.detail,
                "the reason the source refused; a refusal without a reason is an outage")

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUSES

    def describe(self) -> Dict[str, Any]:
        return {"status": self.status, "artifact_sha256": self.artifact_sha256,
                "detail": self.detail, "remediation": self.remediation,
                "bytes_read": int(self.bytes_read), "seconds": round(float(self.seconds), 6),
                "network_used": bool(self.network_used), "retryable": self.retryable}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ComponentOutcome":
        return cls(status=mapping["status"], artifact_sha256=mapping.get("artifact_sha256"),
                   detail=mapping.get("detail", ""), remediation=mapping.get("remediation", ""),
                   bytes_read=int(mapping.get("bytes_read", 0)),
                   seconds=float(mapping.get("seconds", 0.0)),
                   network_used=bool(mapping.get("network_used", False)))


@dataclass(frozen=True)
class StepRequest:
    """One unit of work, addressed by its content.

    `inputs` is what makes the address a drift check rather than a cache key: it holds the digests
    of everything upstream this step consumed, so a changed native artefact produces a changed
    translation key and the translation is redone instead of being paired with a stale record.
    """

    run_sha256: str
    stage: str
    component: str
    inputs: Mapping[str, str] = dc_field(default_factory=dict)
    parameters: Mapping[str, Any] = dc_field(default_factory=dict)

    @property
    def step_sha256(self) -> str:
        return _digest({"schema": SCHEMA, "run": self.run_sha256, "stage": self.stage,
                        "component": self.component, "inputs": dict(self.inputs),
                        "parameters": dict(self.parameters)})

    def describe(self) -> Dict[str, Any]:
        return {"run_sha256": self.run_sha256, "stage": self.stage, "component": self.component,
                "inputs": dict(self.inputs), "parameters": dict(self.parameters),
                "step_sha256": self.step_sha256}


#: A stage worker takes one addressed request and reports one outcome. It is given no database
#: handle, no run object and no way to move the state machine: a worker that could advance the run
#: it is executing is a worker that can decide the experiment finished.
StageWorker = Callable[[StepRequest], ComponentOutcome]


class HeldOutOpenings:
    """Which run spent which held-out partition.

    Small on purpose. `src.core.preregistration.HeldOutLedger` is the authority on what an opening
    *means* - the sealed partition identity, the frozen confirmatory family, the digests that make
    the seal checkable. This records only the fact an orchestrated run needs at retry time: that a
    partition has a first opener, and that it is not this run.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def records(self) -> Dict[str, Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def open(self, partition: str, *, run_sha256: str, at: str) -> Dict[str, Any]:
        records = self.records()
        existing = records.get(partition)
        if existing is not None:
            #: Re-opening within the same run is a resume, not a second spending. That distinction
            #: is the entire reason the opener is recorded rather than a bare boolean.
            if existing["run_sha256"] != run_sha256:
                raise SpentTargetError(partition, existing["run_sha256"][:32])
            return dict(existing)
        record = {"partition": partition, "run_sha256": run_sha256, "opened_at": at}
        records[partition] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(records, sort_keys=True, indent=2).encode("utf-8")
        temporary = self.path.with_suffix(".tmp")
        with io.open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(self.path))
        return dict(record)


class RunJournal:
    """An append-only event log that is the run's only source of truth.

    Folding a log is what makes kill/restart cheap to reason about: there is no separate state file
    to fall out of step with it, and a crash can only ever lose the tail. A torn final line - the
    process died mid-write - is skipped on replay rather than raising, because a half-written
    event is an event that did not happen, and the step it described is still keyed by its content
    and will simply be executed.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        with io.open(self.path, "a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return dict(event)

    def read(self) -> List[Dict[str, Any]]:
        try:
            raw = self.path.read_text("utf-8")
        except FileNotFoundError:
            return []
        events: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                #: Only a truncated tail can be unparseable, and only the last line can be
                #: truncated. Anything after it is unreachable, so stop.
                break
        return events


class ExperimentRun:
    """One frozen manifest, executed as a resumable state machine."""

    def __init__(self, spec: CrossDomainExperimentSpec, root: Path,
                 openings: Optional[HeldOutOpenings] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.spec = spec
        self.identity = run_identity(spec)
        self.run_sha256 = self.identity["run_sha256"]
        self.run_id = self.identity["run_id"]
        self.manifest_sha256 = self.identity["manifest_sha256"]
        self.root = Path(root)
        self.directory = self.root / "runs" / self.run_id
        self.journal = RunJournal(self.directory / "journal.jsonl")
        self.openings = openings or HeldOutOpenings(self.root / "held_out_openings.json")
        self._clock = clock or time.time

    # ---------------------------------------------------------------- replay

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self.journal.read()

    @property
    def state(self) -> str:
        state = "DRAFT"
        for event in self.events:
            if event.get("kind") == "transition":
                state = event["to_state"]
        return state

    def _outcomes(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """The latest recorded outcome per (stage, component), folded from the journal."""
        outcomes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for event in self.events:
            if event.get("kind") in ("step", "step_reused"):
                outcomes[(event["stage"], event["component"])] = event
        return outcomes

    def attempts(self, stage: str, component: str) -> int:
        """How many times this component has actually been executed.

        Folded from the journal rather than counted in memory, so it survives a restart. Replays
        are not attempts: a step that was reused was not requested again, and counting it as an
        attempt would make a resumed run look like a run that kept retrying.
        """
        return sum(1 for event in self.events
                   if event.get("kind") == "step" and event["stage"] == stage
                   and event["component"] == component)

    def _artifact(self, stage: str, component: str) -> Optional[str]:
        record = self._outcomes().get((stage, component))
        return None if record is None else record["outcome"].get("artifact_sha256")

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock()))

    def _emit(self, kind: str, **body: Any) -> Dict[str, Any]:
        return self.journal.append({"schema": SCHEMA, "kind": kind, "at": self._now(),
                                    "run_sha256": self.run_sha256, **body})

    def _transition(self, to_state: str, *, reason: str = "") -> Dict[str, Any]:
        state = self.state
        if to_state not in TRANSITIONS.get(state, ()):
            raise RunStateError(state, to_state)
        return self._emit("transition", from_state=state, to_state=to_state, reason=reason)

    # ------------------------------------------------------------ transitions

    def preflight(self) -> Dict[str, Any]:
        """DRAFT -> PREFLIGHTED, or straight to REFUSED. Idempotent."""
        state = self.state
        if state != "DRAFT":
            return {"run_id": self.run_id, "state": state, "repeated": True,
                    "report": self._recorded_preflight()}
        report = preflight_manifest(self.spec)
        self._emit("preflight", status=report["status"], refusals=report["refusals"],
                   family_size=report["family"]["family_size"],
                   planned_bytes=report["planned_bytes"],
                   preflight_sha256=_digest(report))
        if report["status"] == "REFUSED":
            self._transition("REFUSED", reason="preflight refused the declared manifest")
        else:
            self._transition("PREFLIGHTED", reason="preflight status %s" % report["status"])
        return {"run_id": self.run_id, "state": self.state, "repeated": False, "report": report}

    def _recorded_preflight(self) -> Optional[Dict[str, Any]]:
        for event in reversed(self.events):
            if event.get("kind") == "preflight":
                return event
        return None

    def freeze(self) -> Dict[str, Any]:
        """PREFLIGHTED -> FROZEN. Idempotent, and refuses to freeze what preflight refused."""
        state = self.state
        if state in ("FROZEN",) + WORK_STAGES + ("COMPLETE",):
            return {"run_id": self.run_id, "state": state, "repeated": True,
                    "manifest_sha256": self.manifest_sha256}
        self._transition("FROZEN", reason="manifest frozen at %s" % self.manifest_sha256)
        self._emit("freeze", manifest_sha256=self.manifest_sha256,
                   manifest_bytes=len(canonical_bytes(self.spec)))
        return {"run_id": self.run_id, "state": self.state, "repeated": False,
                "manifest_sha256": self.manifest_sha256}

    def cancel(self, reason: str) -> Dict[str, Any]:
        if not reason.strip():
            raise InvalidParameterError("reason", reason,
                                        "a stated reason for cancelling this run")
        self._transition("CANCELLED", reason=reason)
        return {"run_id": self.run_id, "state": self.state, "reason": reason}

    # -------------------------------------------------------------- execution

    def _request(self, stage: str, component: str) -> StepRequest:
        """Build one step's content address, including everything upstream it consumes."""
        spec = self.spec
        windows = _digest([{"name": w.name, "start": w.start_utc, "end": w.end_utc,
                            "stride_seconds": w.stride_seconds} for w in spec.windows])
        observations = {item.domain: item for item in spec.observations}
        if stage == "ACQUIRING":
            observation = observations[component]
            inputs = {"acquisition": _digest(observation.acquisition.dict()), "windows": windows}
            parameters: Dict[str, Any] = {"mode": spec.mode}
        elif stage == "TRANSLATING":
            observation = observations[component]
            native = self._artifact("ACQUIRING", component)
            inputs = {"native": native or "", "adapter": _digest(observation.adapter.dict())}
            parameters = {"measure": observation.measure, "role": observation.role,
                          "units": observation.units}
        elif stage == "MINING":
            inputs = {domain: (self._artifact("TRANSLATING", domain) or "")
                      for domain in observations}
            inputs["family"] = _digest(spec.family.dict())
            inputs["nulls"] = _digest([null.dict() for null in spec.nulls])
            parameters = {"correction": spec.correction, "alpha": spec.alpha,
                          "seeds": dict(spec.seeds)}
        else:
            inputs = {"candidates": self._artifact("MINING", "candidate_family") or "",
                      "confirmation": _digest(spec.confirmation.dict())}
            parameters = {"correction": spec.correction, "alpha": spec.alpha,
                          "stage": spec.confirmation.stage}
        return StepRequest(run_sha256=self.run_sha256, stage=stage, component=component,
                           inputs=inputs, parameters=parameters)

    def _step_path(self, request: StepRequest) -> Path:
        return self.directory / "steps" / (request.step_sha256 + ".json")

    def _recorded_step(self, request: StepRequest) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self._step_path(request).read_text("utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def _publish_step(self, request: StepRequest, outcome: ComponentOutcome) -> Dict[str, Any]:
        """Publish determinate answers immutably; leave operational failures in the journal.

        A step's address is a claim about its *inputs*, so what it identifies must be a function of
        them. `COMPLETE`, `MISSING` and `REFUSED` are: the archive holds this coverage, or it does
        not, or it declines. A timeout is not - it is a fact about a network at a moment, and
        publishing it under the input address would make a retry that succeeds look like the same
        address disagreeing with itself. Failures are events, and the journal is where events live.
        """
        record = {"schema": SCHEMA, "request": request.describe(), "outcome": outcome.describe()}
        if outcome.status in RETRYABLE_STATUSES:
            return record
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            publish_new_bytes(self._step_path(request), payload, "experiment step")
        except FileExistsError:
            #: Two executions of one addressed step must agree. If they do not, the address is
            #: lying about what it identifies, and that is a defect rather than a race to resolve.
            existing = self._step_path(request).read_bytes()
            if existing != payload:
                raise InvalidParameterError(
                    "step_sha256", request.step_sha256,
                    "an address that identifies one outcome. This one already recorded a "
                    "different one, and a content-addressed step must be a function of its "
                    "address - two answers here is a defect, not a race to resolve")
        return record

    def _execute_component(self, stage: str, component: str, workers: Mapping[str, StageWorker],
                           *, force: bool = False) -> Dict[str, Any]:
        request = self._request(stage, component)
        recorded = None if force else self._recorded_step(request)
        if recorded is not None and recorded["outcome"]["status"] == "COMPLETE":
            #: The whole point of the address: completed work is replayed, never repeated. This is
            #: what "resume without duplicate network acquisition" means concretely.
            return self._emit("step_reused", stage=stage, component=component,
                              step_sha256=request.step_sha256, outcome=recorded["outcome"])
        worker = workers.get(stage)
        if worker is None:
            raise MissingStageWorkerError(stage, list(workers))
        started = self._clock()
        outcome = worker(request)
        if not isinstance(outcome, ComponentOutcome):
            raise InvalidParameterError(
                "%s worker result" % stage, type(outcome).__name__,
                "a ComponentOutcome, which is the type with nowhere to put a result value")
        if outcome.seconds == 0.0:
            outcome = ComponentOutcome(
                status=outcome.status, artifact_sha256=outcome.artifact_sha256,
                detail=outcome.detail, remediation=outcome.remediation,
                bytes_read=outcome.bytes_read, seconds=self._clock() - started,
                network_used=outcome.network_used)
        self._publish_step(request, outcome)
        return self._emit("step", stage=stage, component=component,
                          step_sha256=request.step_sha256, outcome=outcome.describe())

    def _decide_stage(self, stage: str) -> Dict[str, Any]:
        outcomes = self._outcomes()
        statuses = {name: outcomes[(stage, name)]["outcome"]["status"]
                    for name in stage_components(self.spec, stage)}
        return decide_stage(self.spec, stage, statuses)

    def execute(self, workers: Mapping[str, StageWorker], *,
                spec: Optional[CrossDomainExperimentSpec] = None) -> Dict[str, Any]:
        """Drive the machine as far as the frozen plan and the workers allow. Resumable.

        Calling this on a run that already reached a terminal state performs no work and returns
        the receipt: replay is the normal case, not the exception.
        """
        if spec is not None and manifest_sha256(spec) != self.manifest_sha256:
            raise PlanChangedError(self.manifest_sha256, manifest_sha256(spec))
        if self.state == "DRAFT":
            self.preflight()
        if self.state == "PREFLIGHTED":
            self.freeze()
        if self.state in TERMINAL_STATES:
            return self.receipt()
        if self.state == "FAILED":
            raise RetryNotPermittedError(
                "this run is FAILED; call retry(), which re-executes only the components that "
                "failed operationally and refuses to author a new plan")
        for stage in WORK_STAGES:
            if not self._stage_is_pending(stage):
                continue
            self._enter(stage)
            if stage == "CONFIRMING":
                self._open_held_out()
            for component in stage_components(self.spec, stage):
                self._execute_component(stage, component, workers)
            decision = self._decide_stage(stage)
            self._emit("stage_decision", **decision)
            if decision["verdict"] == "REFUSED":
                self._transition("REFUSED", reason=decision["reason"])
                return self.receipt()
            if decision["verdict"] == "FAILED":
                self._transition("FAILED", reason=decision["reason"])
                return self.receipt()
        self._transition("COMPLETE", reason="every declared stage completed")
        return self.receipt()

    def _stage_is_pending(self, stage: str) -> bool:
        """Has this stage already been decided? Folded from the journal, not tracked separately."""
        verdict = None
        for event in self.events:
            #: The *last* decision, not the first: a stage that failed and was retried has two,
            #: and reading the earlier one would re-execute a stage that has since advanced.
            if event.get("kind") == "stage_decision" and event["stage"] == stage:
                verdict = event["verdict"]
        return verdict not in ("ADVANCE", "ADVANCE_PARTIAL")

    def _enter(self, stage: str) -> None:
        if self.state != stage:
            self._transition(stage, reason="entering %s" % stage)

    def _open_held_out(self) -> Optional[Dict[str, Any]]:
        confirmation = self.spec.confirmation
        if confirmation.stage != "generate_then_confirm" or not confirmation.held_out_partition:
            return None
        record = self.openings.open(confirmation.held_out_partition,
                                    run_sha256=self.run_sha256, at=self._now())
        self._emit("held_out_opened", partition=confirmation.held_out_partition,
                   opened_by=record["run_sha256"], opened_at=record["opened_at"],
                   confirmatory_members=confirmation.confirmatory_members)
        return record

    def retry(self, workers: Mapping[str, StageWorker], *,
              spec: Optional[CrossDomainExperimentSpec] = None) -> Dict[str, Any]:
        """Re-execute only what failed operationally. Never a new plan, never a spent target."""
        if spec is not None and manifest_sha256(spec) != self.manifest_sha256:
            raise PlanChangedError(self.manifest_sha256, manifest_sha256(spec))
        state = self.state
        if state != "FAILED":
            raise RetryNotPermittedError(
                "only a FAILED run can be retried; this run is %s. A refusal is the archive "
                "answering the question, and asking it again does not change the answer - "
                "`editable_copy` is the remedy for a refusal." % state)
        stage = self._failed_stage()
        outcomes = self._outcomes()
        components = [name for name in stage_components(self.spec, stage)
                      if outcomes[(stage, name)]["outcome"]["status"] in RETRYABLE_STATUSES]
        self._transition(stage, reason="retrying %s in %s" % (", ".join(components), stage))
        self._emit("retry", stage=stage, components=components)
        for component in components:
            self._execute_component(stage, component, workers, force=True)
        decision = self._decide_stage(stage)
        self._emit("stage_decision", **decision)
        if decision["verdict"] == "REFUSED":
            self._transition("REFUSED", reason=decision["reason"])
            return self.receipt()
        if decision["verdict"] == "FAILED":
            self._transition("FAILED", reason=decision["reason"])
            return self.receipt()
        return self.execute(workers)

    def _failed_stage(self) -> str:
        for event in reversed(self.events):
            if event.get("kind") == "stage_decision" and event["verdict"] == "FAILED":
                return event["stage"]
        raise RetryNotPermittedError("no stage recorded an operational failure to retry")

    # ---------------------------------------------------------------- reports

    def editable_copy(self, store: Any, draft_id: str) -> Dict[str, Any]:
        """The remedy for a refusal: a new mutable draft, and the frozen run left alone.

        Deliberately not `unfreeze`. A frozen run that can be edited after seeing how it went is a
        plan chosen with knowledge of the result, and the fact that the edit was well-intentioned
        is not recoverable from the artefacts six months later.
        """
        before = len(self.events)
        saved = store.save(draft_id, self.spec)
        return {"draft_id": draft_id, "manifest_sha256": saved["manifest_sha256"],
                "copied_from_run": self.run_id, "frozen_run_state": self.state,
                "frozen_run_untouched": len(self.events) == before,
                "note": "The refused run is immutable. This draft is editable and will become a "
                        "different run identity the moment its manifest changes."}

    def bounded_work(self) -> Dict[str, Any]:
        """A work estimate that is bounded because the plan is declared, not discovered."""
        total = sum(len(stage_components(self.spec, stage)) for stage in WORK_STAGES)
        outcomes = self._outcomes()
        completed = sum(1 for key, event in outcomes.items()
                        if key[0] in WORK_STAGES and event["outcome"]["status"] == "COMPLETE")
        return {"completed_steps": completed, "total_steps": total,
                "fraction": round(completed / float(total), 6) if total else 0.0,
                "bytes_read": sum(int(event["outcome"].get("bytes_read", 0))
                                  for event in outcomes.values())}

    def progress(self) -> Dict[str, Any]:
        """What a watcher may see while a stage is still running.

        Stage, component status, artefact digest, bounded work and remediation. No result value
        appears here because `ComponentOutcome` has no field one could occupy.
        """
        outcomes = self._outcomes()
        stages = []
        for stage in WORK_STAGES:
            components = []
            for name in stage_components(self.spec, stage):
                event = outcomes.get((stage, name))
                components.append({"component": name,
                                   "status": event["outcome"]["status"] if event else "PENDING",
                                   "artifact_sha256": (event["outcome"]["artifact_sha256"]
                                                       if event else None),
                                   "remediation": event["outcome"]["remediation"] if event else "",
                                   "reused": bool(event and event["kind"] == "step_reused")})
            stages.append({"stage": stage, "components": components})
        return {"schema": SCHEMA, "run_id": self.run_id, "state": self.state,
                "manifest_sha256": self.manifest_sha256, "stages": stages,
                "bounded_work": self.bounded_work(),
                "retryable": self.state == "FAILED",
                "results_visible": self.state == "COMPLETE",
                "claim_boundary": "Progress reports which stage is running and which artefacts "
                                  "exist. It never reports what an unopened stage found."}

    def receipt(self) -> Dict[str, Any]:
        """The immutable account of what ran, including every component that did not."""
        outcomes = self._outcomes()
        missing = [{"stage": stage, "component": component,
                    "status": event["outcome"]["status"],
                    "detail": event["outcome"]["detail"],
                    "remediation": event["outcome"]["remediation"]}
                   for (stage, component), event in sorted(outcomes.items())
                   if event["outcome"]["status"] != "COMPLETE"]
        artefacts = {"%s/%s" % (stage, component): event["outcome"]["artifact_sha256"]
                     for (stage, component), event in sorted(outcomes.items())
                     if event["outcome"]["status"] == "COMPLETE"}
        decisions = [event for event in self.events if event.get("kind") == "stage_decision"]
        history = [{"at": event["at"], "from": event["from_state"], "to": event["to_state"],
                    "reason": event.get("reason", "")}
                   for event in self.events if event.get("kind") == "transition"]
        return {"schema": SCHEMA, "run_id": self.run_id, "run_sha256": self.run_sha256,
                "manifest_sha256": self.manifest_sha256, "study_id": self.spec.study_id,
                "state": self.state, "history": history,
                "artefacts": artefacts, "missing_components": missing,
                "stage_decisions": [{"stage": event["stage"], "verdict": event["verdict"],
                                     "reason": event["reason"]} for event in decisions],
                "bounded_work": self.bounded_work(),
                "coverage_policy": {"requirement": self.spec.coverage_policy.requirement,
                                    "minimum_fraction": self.spec.coverage_policy.minimum_fraction},
                "confirmation": self.spec.confirmation.dict(),
                "events": len(self.events),
                "claim_boundary": "A completed run is an executed plan, not admitted evidence."}


class RunStore:
    """Content-addressed runs on disk. `open` creates or resumes; it never forks."""

    def __init__(self, root: Path, clock: Optional[Callable[[], float]] = None) -> None:
        self.root = Path(root)
        self._clock = clock

    def open(self, spec: CrossDomainExperimentSpec) -> ExperimentRun:
        identity = run_identity(spec)
        run = ExperimentRun(spec, self.root, clock=self._clock)
        directory = run.directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        raw = canonical_bytes(spec)
        try:
            publish_new_bytes(path, raw, "frozen run manifest")
        except FileExistsError:
            if path.read_bytes() != raw:
                #: The run identity is a digest of the manifest, so this cannot happen without a
                #: SHA-256 collision - which is worth saying out loud rather than overwriting.
                raise InvalidParameterError(
                    "run_id", identity["run_id"],
                    "a run holding the manifest it is addressed by. This one holds different "
                    "bytes, and since the identity is a SHA-256 content address that is a "
                    "collision rather than something to overwrite")
        return run

    def load(self, run_id: str) -> ExperimentRun:
        if not RUN_ID.fullmatch(run_id):
            raise InvalidParameterError("run_id", run_id,
                                        "32 hexadecimal characters of a run content address")
        path = self.root / "runs" / run_id / "manifest.json"
        if not path.exists():
            raise InvalidParameterError("run_id", run_id,
                                        "a run recorded under %s" % self.root)
        spec = CrossDomainExperimentSpec.parse_raw(path.read_bytes())
        return ExperimentRun(spec, self.root, clock=self._clock)

    def list_runs(self) -> List[Dict[str, Any]]:
        directory = self.root / "runs"
        if not directory.exists():
            return []
        rows = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or not RUN_ID.fullmatch(entry.name):
                continue
            run = self.load(entry.name)
            rows.append({"run_id": run.run_id, "study_id": run.spec.study_id,
                         "title": run.spec.title, "state": run.state,
                         "manifest_sha256": run.manifest_sha256,
                         "bounded_work": run.bounded_work()})
        return rows


def describe_state_machine() -> Dict[str, Any]:
    """The machine as data, so the browser draws the states the backend actually enforces."""
    return {"schema": SCHEMA, "states": list(STATES), "work_stages": list(WORK_STAGES),
            "terminal_states": list(TERMINAL_STATES),
            "transitions": {state: list(successors) for state, successors in TRANSITIONS.items()},
            "component_statuses": list(COMPONENT_STATUSES),
            "retryable_statuses": list(RETRYABLE_STATUSES),
            "note": "A retry re-executes operational failures only. A refusal is answered with an "
                    "editable copy, never by editing the frozen run."}


__all__ = ["COMPONENT_STATUSES", "ComponentOutcome", "ExperimentRun", "HeldOutOpenings",
           "MissingStageWorkerError", "PlanChangedError", "RETRYABLE_STATUSES",
           "RetryNotPermittedError", "RunJournal", "RunStateError", "RunStore", "SCHEMA",
           "STATES", "SpentTargetError", "StepRequest", "TERMINAL_STATES", "TRANSITIONS",
           "WORK_STAGES", "decide_stage", "describe_state_machine", "run_identity",
           "stage_components"]
