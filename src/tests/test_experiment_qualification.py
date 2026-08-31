"""TG17.10a: deterministic qualification must stay visibly below scientific release."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.experiment_adapter import adapter_for_domain
from src.core.experiment_manifest import preflight_manifest
from src.core.experiment_qualification import (
    DURATIONS,
    MODES,
    RECORD_KIND,
    execute_offline_qualification,
    qualification_manifest,
    qualification_plan,
    verify_qualification_record,
)
from src.core.experiment_run import RunStore
from src.core.run_workers import build_suite, describe_suites


@pytest.fixture(scope="module")
def qualified(tmp_path_factory):
    return execute_offline_qualification(tmp_path_factory.mktemp("g17-qualification"))


def test_plan_is_exactly_three_durations_by_two_separate_modes():
    plan = qualification_plan()
    assert len(plan["matrix"]) == 6
    assert {(row["duration"], row["mode"]) for row in plan["matrix"]} == {
        (duration, mode) for duration in DURATIONS for mode in MODES}
    assert all(row["status"] == "NOT_RUN" for row in plan["matrix"])


def test_exact_dates_and_complete_family_correction_are_frozen_before_results():
    plan = qualification_plan()
    windows = {row["duration"]: (row["start_utc"], row["end_utc"])
               for row in plan["matrix"]}
    assert windows == {
        "week": ("2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"),
        "three_months": ("2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
        "six_months": ("2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    }
    assert {row["family_correction"] for row in plan["matrix"]} == {
        "benjamini_yekutieli"}


@pytest.mark.parametrize("duration", DURATIONS)
def test_every_calendar_matrix_manifest_preflights_without_a_refusal(duration):
    report = preflight_manifest(qualification_manifest(duration, "calendar_aligned"))
    assert report["status"] == "PARTIAL"
    assert report["refusals"] == []
    assert report["network_used"] is False
    assert report["measurement_values_opened"] is False


@pytest.mark.parametrize("duration", DURATIONS)
def test_the_scale_shape_quartet_is_refused_by_the_order_books_own_declaration(duration):
    """The quartet cannot be qualified in scale/shape mode, and that is the measurement.

    `order_book.bespoke_record` declares that it cannot carry `scale_partner_reassignment`: a
    depositor-supplied record's native scale is whatever the depositor wrote down, so there is
    no native duration to compare shapes across. Preflight refuses before anything executes.
    """
    report = preflight_manifest(qualification_manifest(duration, "scale_shape_aligned"))
    assert report["refusals"]
    assert any("order_book.bespoke_record" in str(row.get("reason", ""))
               for row in report["refusals"])
    assert report["network_used"] is False
    assert report["measurement_values_opened"] is False


def test_modes_have_different_relationship_null_and_language_contracts():
    calendar = qualification_manifest("week", "calendar_aligned")
    scale = qualification_manifest("week", "scale_shape_aligned")
    assert calendar.family.relationships == ["co_occurrence"]
    assert calendar.nulls[0].method == "independent_native_clock_shift"
    assert calendar.scale_normalization is None
    assert scale.family.relationships == ["shape_recurrence"]
    assert scale.nulls[0].method == "scale_partner_reassignment"
    assert scale.scale_normalization == {
        "method": "native_scale_ratio", "reference": "within_domain"}
    assert "no simultaneity" in scale.notes["claim_boundary"]


def test_order_book_fixture_is_bound_by_content_not_filename():
    spec = qualification_manifest("week", "calendar_aligned")
    order_book = next(row for row in spec.observations if row.domain == "order_book")
    assert len(order_book.acquisition.identity["content_sha256"]) == 64
    assert order_book.adapter.parameters["source_binding"] == "benchmark_known_answer"


def test_the_scale_partner_null_is_admitted_per_domain_and_never_by_default():
    """A framework default must not answer a question only the adapter author can answer.

    Three flagship domains declare that their support can carry the scale/shape null, each with
    a stated reason. The bespoke record declines. Widening the shared default until the matrix
    turned green would have silently overruled that refusal for every adapter, including a fifth
    one nobody has written yet -- which is what the full suite caught before this slice landed.
    """
    from src.core.experiment_adapter import DomainExperimentAdapter

    admitting = {row.domain for row in qualification_manifest(
        "week", "scale_shape_aligned").observations
        if "scale_partner_reassignment" in adapter_for_domain(row.domain).admissible_nulls}
    assert admitting == {"reanalysis", "argo_float", "tess_lightcurve"}
    assert "scale_partner_reassignment" not in adapter_for_domain(
        "order_book").admissible_nulls
    assert DomainExperimentAdapter.admissible_nulls == ("independent_native_clock_shift",)


def test_offline_matrix_executes_exports_and_replays_every_admissible_cell(qualified):
    executed = [row for row in qualified["matrix"] if row["status"] == "PASS"]
    refused = [row for row in qualified["matrix"] if row["status"] == "REFUSED"]
    assert [row["mode"] for row in executed] == ["calendar_aligned"] * 3
    assert [row["mode"] for row in refused] == ["scale_shape_aligned"] * 3
    assert all(row["checks"]["receipt_integrity_verified"] for row in executed)
    assert all(row["checks"]["fixture_not_measured"] for row in executed)
    assert all(row["record_kind"] == RECORD_KIND for row in qualified["matrix"])


def test_a_refused_cell_opens_no_run_and_keeps_the_reason_it_was_refused(qualified):
    for row in qualified["matrix"]:
        if row["status"] != "REFUSED":
            continue
        assert row["refusals"]
        # A refused plan must not acquire a run identity: a later reader could not otherwise
        # distinguish an experiment that was declined from one that was merely never executed.
        assert "run_id" not in row and "bundle_sha256" not in row
        assert len(row["manifest_sha256"]) == 64


def test_one_failure_restarts_and_retries_only_tess(qualified):
    recovery = qualified["recovery"]
    assert recovery["status"] == "PASS"
    assert recovery["failed_component"] == "tess_lightcurve"
    assert recovery["attempts_before_restart"] == {
        "reanalysis": 1, "argo_float": 1, "tess_lightcurve": 1, "order_book": 1}
    assert recovery["attempts_after_retry"] == {
        "reanalysis": 1, "argo_float": 1, "tess_lightcurve": 2, "order_book": 1}
    assert all(recovery["checks"].values())


def test_single_failure_suite_is_registered_and_fixture_only():
    row = next(item for item in describe_suites()
               if item["name"] == "fixture_single_remote_failure")
    assert row["capabilities"]["failed_components_per_rehearsal"] == 1
    assert row["capabilities"]["acquires"] is False
    assert set(build_suite(row["name"])) == {"ACQUIRING", "TRANSLATING", "MINING", "CONFIRMING"}


def test_green_offline_rehearsal_cannot_make_the_release_verdict_green(qualified):
    gates = {row["gate_id"]: row for row in qualified["gates"]}
    # REFUSED, not FAIL: nothing in the apparatus broke. A declared scientific refusal blocks
    # release exactly as a failure does, and the two must stay distinguishable.
    assert gates["offline_matrix"]["status"] == "REFUSED"
    assert gates["offline_matrix"]["blocking"] is True
    assert "order_book.bespoke_record" in gates["offline_matrix"]["detail"]
    assert gates["restart_recovery"]["status"] == "PASS"
    assert gates["scale_shape_calibration"]["status"] == "NOT_IMPLEMENTED"
    assert gates["live_sources"]["status"] == "NOT_RUN"
    assert qualified["verdict"] == "NOT_RELEASEABLE"


def test_scientist_action_measurements_are_not_invented(qualified):
    assert set(qualified["scientist_actions"].values()) >= {
        "NOT_MEASURED", "visible researcher actions from a clean browser session"}


def test_qualification_record_is_self_hashed_and_tampering_is_detected(qualified):
    assert verify_qualification_record(qualified)["integrity"] == "VERIFIED"
    tampered = copy.deepcopy(qualified)
    tampered["gates"][-1]["status"] = "PASS"
    with pytest.raises(ValueError, match="qualification_sha256"):
        verify_qualification_record(tampered)


def test_repeating_the_qualification_resumes_identical_runs(tmp_path):
    first = execute_offline_qualification(tmp_path)
    second = execute_offline_qualification(tmp_path)
    assert second["qualification_sha256"] == first["qualification_sha256"]
    assert [row.get("run_id") for row in second["matrix"]] == [
        row.get("run_id") for row in first["matrix"]]
    # Three executed calendar cells plus the recovery rehearsal. The refused cells opened none.
    assert len(RunStore(tmp_path).list_runs()) == 4


def test_http_plan_and_rehearsal_keep_the_unrun_gates_visible(tmp_path):
    app.state.experiment_run_dir = str(tmp_path / "runs")
    with TestClient(app) as client:
        plan = client.get("/api/v1/experiment-qualification")
        assert plan.status_code == 200
        assert plan.json()["verdict"] == "NOT_RELEASEABLE"
        response = client.post("/api/v1/experiment-qualification/rehearse")
        assert response.status_code == 200
        body = response.json()
    assert sum(row["status"] == "PASS" for row in body["matrix"]) == 3
    assert sum(row["status"] == "REFUSED" for row in body["matrix"]) == 3
    assert any(row["status"] == "NOT_RUN" for row in body["gates"])
    assert body["record_kind"] == RECORD_KIND
