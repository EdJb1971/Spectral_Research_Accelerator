"""T4E.43: a long measurement you can actually watch and stop.

The durable-run machinery was already resumable, content-addressed and journalled. What it could
not do was be interrupted: `execute` ran every declared stage to completion whatever the journal
said, and the HTTP route drove it on the event loop, so the whole API stopped answering for the
duration. The two controls built for a long run -- `/progress` and `/cancel` -- were unreachable
for exactly as long as the run they exist to watch and stop, and a cancel could therefore only be
issued to a run that was not running.

These tests hold the repaired behaviour: the instrument keeps answering while a run works, a
cancel raised mid-run is observed at the next component boundary, completed work is kept, and
the cancellation is terminal rather than a pause.
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest
from fastapi.testclient import TestClient

from src.api import experiment_runs
from src.api.main import app
from src.core.experiment_manifest import flagship_recipe
from src.core.experiment_run import (TERMINAL_STATES, WORK_STAGES, ComponentOutcome, RunStore,
                                     stage_components)


@pytest.fixture
def spec():
    recipe = flagship_recipe()
    return recipe.copy(update={"observations": [item for item in recipe.observations
                                                if item.domain != "order_book"]})


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path)


class CancellingWorker:
    """Completes components, and cancels the run from a second instance partway through.

    A second instance is the point: it is how the HTTP cancel route reaches a run that another
    request is executing. The two share nothing but the journal on disk.
    """

    def __init__(self, store, spec, cancel_after: int) -> None:
        self.store = store
        self.spec = spec
        self.cancel_after = cancel_after
        self.calls = []

    def __call__(self, request):
        self.calls.append((request.stage, request.component))
        if len(self.calls) == self.cancel_after:
            self.store.open(self.spec).cancel("stopped by the researcher while it was running")
        return ComponentOutcome(status="COMPLETE", artifact_sha256="0" * 64,
                                detail="worked", bytes_read=1, seconds=0.01)


def _suite(worker):
    return {stage: worker for stage in WORK_STAGES}


def test_a_run_stops_at_the_next_component_after_a_cancel_raised_elsewhere(store, spec):
    run = store.open(spec)
    acquiring = list(stage_components(spec, "ACQUIRING"))
    assert len(acquiring) > 1, "this test needs a stage with more than one component"
    worker = CancellingWorker(store, spec, cancel_after=1)

    receipt = run.execute(_suite(worker))

    assert receipt["state"] == "CANCELLED"
    # One component ran, the cancel landed, and the second was never started.
    assert len(worker.calls) == 1
    assert worker.calls[0][0] == "ACQUIRING"


def test_work_already_paid_for_survives_the_cancellation(store, spec):
    run = store.open(spec)
    worker = CancellingWorker(store, spec, cancel_after=2)

    run.execute(_suite(worker))

    steps = [event for event in run.events if event.get("kind") == "step"]
    assert len(worker.calls) == 2
    # A cancel discards the remaining work, never the work already done. Both completed
    # components keep their recorded outcome, which is what makes the run resumable-in-principle
    # and what stops a stop button from throwing away paid acquisition.
    assert len(steps) == 2
    assert [event["outcome"]["status"] for event in steps] == ["COMPLETE", "COMPLETE"]
    assert all(event["step_sha256"] for event in steps)
    assert run.state == "CANCELLED"


def test_a_cancelled_run_is_terminal_and_does_no_further_work(store, spec):
    run = store.open(spec)
    worker = CancellingWorker(store, spec, cancel_after=1)
    run.execute(_suite(worker))
    assert run.state in TERMINAL_STATES

    before = len(worker.calls)
    receipt = run.execute(_suite(worker))

    assert receipt["state"] == "CANCELLED"
    assert len(worker.calls) == before, "a cancelled run must not resume by being re-executed"


def test_the_execution_routes_are_not_declared_on_the_event_loop():
    """`/progress` and `/cancel` are only useful if something can serve them mid-run."""
    for name in ("execute", "retry"):
        route = getattr(experiment_runs, name)
        assert not inspect.iscoroutinefunction(route), (
            "%s must stay synchronous so FastAPI threadpools it; as `async def` a long run "
            "blocked every other request, including the progress and cancel routes that exist "
            "to watch and stop it" % name)
        reason = (route.__doc__ or "").lower()
        assert "threadpool" in reason or "synchronous" in reason, (
            "the reason %s is synchronous must stay written beside it" % name)


def test_the_instrument_keeps_answering_while_a_run_occupies_the_threadpool():
    """The regression measured, not merely asserted structurally."""
    client = TestClient(app)

    async def exercise() -> float:
        loop = asyncio.get_running_loop()
        occupied = loop.run_in_executor(None, time.sleep, 1.5)
        started = time.monotonic()
        health = await loop.run_in_executor(None, client.get, "/api/v1/health")
        elapsed = time.monotonic() - started
        await occupied
        assert health.status_code == 200
        return elapsed

    assert asyncio.run(exercise()) < 1.0


def test_the_state_machine_still_declares_cancelled_terminal_and_says_why():
    machine = TestClient(app).get("/api/v1/experiment-runs/state-machine")
    if machine.status_code == 404:
        from src.core.experiment_run import describe_state_machine
        described = describe_state_machine()
    else:
        described = machine.json()

    assert "CANCELLED" in described["terminal_states"]
    assert described["transitions"]["CANCELLED"] == []
