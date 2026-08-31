"""Registered stage-worker suites for the TG17.6 orchestrator (standard E1).

A suite is a named set of workers, one per work stage, that the orchestrator can be driven with.
It is a registry rather than an argument to a route because the alternative - a route that accepts
executable behaviour from its caller - is a route that lets the request decide what an experiment
does.

**Everything registered here acquires nothing.** These are rehearsal suites: they exercise the
state machine, the content-addressed step store, the journal and the receipt without opening an
archive or reading a measurement value. Each one reports `network_used: false` on every component
and stamps `fixture: true` into the detail line, so a receipt produced by one cannot be mistaken
for a receipt produced by data. The domain workers that acquire and translate for real arrive with
the slices that own those operations; when they do, they register here beside these and the
orchestrator does not change.

Two suites, because the two things worth rehearsing before a long remote job are different:

* `fixture_dry_run` completes every component, which is what proves a plan can be executed at all
  and what a browser needs in order to show the whole path from FROZEN to COMPLETE.
* `fixture_transient_failure` times out the first attempt at each acquisition component and
  succeeds on the second. An operator who has never seen the retry path before the night a remote
  archive times out is an operator who will reach for "start again", and starting again is the
  recovery this whole slice exists to make unnecessary.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Set, Tuple

from src.core.experiment_run import WORK_STAGES, ComponentOutcome, StepRequest
from src.core.registry import Registry


#: Named worker suites. A suite is `Mapping[stage, StageWorker]`.
WORKER_SUITES: Registry[Any] = Registry("run worker suite")

FIXTURE_DETAIL = ("fixture: the orchestrator executed this step; no archive was opened and no "
                  "measurement value was read")


def _fixture_digest(request: StepRequest) -> str:
    """An artefact digest derived from the step address, so replay is bit-identical.

    Deriving it from the address rather than from a clock or a counter is what makes a rehearsal
    resumable in exactly the way a real run is: the same declared plan produces the same digests
    on the same machine and on a different one.
    """
    return hashlib.sha256(("fixture-artifact/v1:" + request.step_sha256).encode("utf-8")).hexdigest()


def _complete(request: StepRequest) -> ComponentOutcome:
    return ComponentOutcome(status="COMPLETE", artifact_sha256=_fixture_digest(request),
                            detail=FIXTURE_DETAIL, bytes_read=0, network_used=False)


def dry_run_suite(run: Any = None) -> Dict[str, Any]:
    """Every component completes. Nothing is acquired."""
    return {stage: _complete for stage in WORK_STAGES}


def transient_failure_suite(run: Any = None) -> Dict[str, Any]:
    """Acquisition times out on the first attempt at each component, then completes on retry.

    "First attempt" is read from the run's own journal when one is supplied, not from a counter in
    this process. That is what makes the rehearsal work the way the thing it rehearses works: the
    retry arrives as a separate HTTP request, in a separate process after a restart, and a suite
    that remembered its failures only in memory would fail the same component forever and teach an
    operator that retrying does not help.

    The remediation text is the part that matters most. An operational failure a researcher cannot
    act on is indistinguishable from a refusal, and the orchestrator refuses to record one without
    it.
    """
    seen: Set[str] = set()

    def already_attempted(request: StepRequest) -> bool:
        if run is not None:
            return run.attempts(request.stage, request.component) > 0
        return request.step_sha256 in seen

    def acquire(request: StepRequest) -> ComponentOutcome:
        if not already_attempted(request):
            seen.add(request.step_sha256)
            return ComponentOutcome(
                status="TIMED_OUT",
                detail="rehearsal: the %s request exceeded its deadline" % request.component,
                remediation="Retry this run. The completed components are replayed from their "
                            "content addresses and are not requested again.",
                network_used=False)
        return _complete(request)

    suite = {stage: _complete for stage in WORK_STAGES}
    suite["ACQUIRING"] = acquire
    return suite


WORKER_SUITES.add("fixture_dry_run", dry_run_suite,
                  description="Completes every stage without acquiring anything, so a frozen plan "
                              "can be rehearsed end to end before a byte is requested.",
                  capabilities={"acquires": False, "network_used": False, "fixture": True})

WORKER_SUITES.add("fixture_transient_failure", transient_failure_suite,
                  description="Times out each acquisition component once and completes it on "
                              "retry, so the retry path is exercised before a real archive "
                              "exercises it.",
                  capabilities={"acquires": False, "network_used": False, "fixture": True,
                                "fails_first_attempt": True})


def build_suite(name: str, run: Any = None) -> Mapping[str, Any]:
    """Instantiate a registered suite, optionally bound to the run it will be driven with.

    Binding is what lets a suite read the run's journal instead of keeping its own memory, so a
    rehearsal behaves identically whether it is driven from one process or from four HTTP
    requests across a restart.
    """
    return WORKER_SUITES.get(name)(run=run)


def describe_suites() -> Tuple[Dict[str, Any], ...]:
    return tuple(entry.to_dict() for entry in WORKER_SUITES.entries())


__all__ = ["FIXTURE_DETAIL", "WORKER_SUITES", "build_suite", "describe_suites",
           "dry_run_suite", "transient_failure_suite"]
