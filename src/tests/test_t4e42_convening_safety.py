"""T4E.42: convening must not take the instrument down, and must not be reachable by accident.

Two defects met on 2026-09-13, and they compounded. A browser test clicked convene expecting the
no-key refusal, but this machine supplies a key, so an eight-seat panel actually ran and recorded
itself. It went unnoticed because the handler was declared `async def` over a blocking transport,
so the call sat on the event loop and the whole API -- `/health` included -- stopped answering
until the process was killed. Every panel in the browser suite then failed to load, which read as
a broken frontend rather than as a convening in progress.

These tests hold both halves. Neither one makes a network call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import reviews


ARCHIVE = Path("data/superseded/t4e42_unintended_panel_20260913")


def test_the_convening_route_is_not_declared_on_the_event_loop():
    """A blocking transport in an `async def` route blocks every other request.

    The transport's own default batch timeout is 86,400 seconds, so the worst case is not a slow
    response -- it is an instrument that answers nothing for a day. FastAPI runs a plain `def`
    route in a threadpool, so this is asserted structurally rather than by timing a real call.
    """
    assert not inspect.iscoroutinefunction(reviews.convene_round_robin), (
        "convene_round_robin must stay a synchronous route so FastAPI runs it in a threadpool; "
        "as `async def` its blocking transport froze the whole API, /health included")
    assert "threadpool" in (reviews.convene_round_robin.__doc__ or ""), (
        "the reason this route is synchronous must stay written down beside it")


def test_an_unauthorised_convening_refuses_before_any_key_is_read_and_sends_nothing():
    """The authorisation refusal is decided before the key, so it is safe on any machine."""
    response = TestClient(app).post(
        "/api/v1/reviews/studies/t4e28-join-rerun/round-robin",
        json={"i_authorise_paid_calls": False})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Nothing was sent" in detail
    assert "costs money" in detail


def test_the_panel_plan_reports_whether_a_key_is_present_without_revealing_it():
    """The browser suite needs this to check its own precondition instead of assuming it."""
    plan = TestClient(app).get("/api/v1/reviews/panel-plan").json()

    assert isinstance(plan["key_present"], bool)
    assert plan["key_variables"] == ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    assert plan["network_used"] is False
    # The value is reported, never the key: no variable's content may appear in the response.
    body = json.dumps(plan)
    assert "GEMINI_API_KEY=" not in body
    assert plan.get("key") is None


def test_the_api_keeps_answering_while_a_convening_is_in_flight():
    """The regression, measured rather than asserted structurally.

    A convening is simulated by occupying the same threadpool path with a blocking call; the
    point is that an unrelated request is still served while it runs. Under the `async def`
    version this test could not have passed, because nothing else was served at all.
    """
    client = TestClient(app)

    async def exercise() -> float:
        loop = asyncio.get_running_loop()
        blocked = loop.run_in_executor(None, time.sleep, 1.5)
        started = time.monotonic()
        health = await loop.run_in_executor(None, client.get, "/api/v1/health")
        elapsed = time.monotonic() - started
        await blocked
        assert health.status_code == 200
        return elapsed

    assert asyncio.run(exercise()) < 1.0, (
        "an unrelated request waited on the blocking call, which is the defect this slice fixed")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="the unintended panel archive is not present")
def test_the_unintended_panel_is_preserved_outside_the_study_record():
    """Paid calls are never discarded, and a test harness never becomes a study's reviewer."""
    outcome = json.loads((ARCHIVE / "t4e28-join-rerun.round-robin.json").read_text(
        encoding="utf-8"))

    assert outcome["study_id"] == "t4e28-join-rerun"
    assert outcome["record_sha256"]
    assert (ARCHIVE / "README.md").read_text(encoding="utf-8").strip()

    # It must not be readable as the study's own completed panel.
    live = Path("data/reviews")
    assert not (live / "t4e28-join-rerun.round-robin.json").exists()
    assert not (live / "t4e28-join-rerun.review.json").exists()
