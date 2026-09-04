"""TG17.10a: deterministic qualification must stay visibly below scientific release."""

import copy
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.experiment_adapter import adapter_for_domain
from src.core.experiment_manifest import preflight_manifest
from src.core.experiment_qualification import (
    DURATIONS,
    MODES,
    RECORD_KIND,
    SCALE_SHAPE_CALIBRATION,
    SCALE_SHAPE_DOMAIN_FAMILIES,
    execute_offline_qualification,
    qualification_manifest,
    qualification_plan,
    scale_shape_applicability,
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
    # TG17.11 moved this off NOT_IMPLEMENTED: a registered calibration now exists, and what
    # blocks the gate is that no declared plan can reach it. Still blocking, for a stated reason.
    assert gates["scale_shape_calibration"]["status"] == "REFUSED"
    # TG17.12: the calendar calibration ran, met every frozen answer, and now reaches the record.
    # It is the first scientific gate to clear, and the verdict is unmoved by it.
    assert gates["calendar_calibration"]["status"] == "PASS"
    assert gates["calendar_calibration"]["blocking"] is False
    # TG17.13: the fifth-adapter gate no longer states a sentence about a measurement nobody made.
    # It reads a live source-edit audit and a recorded acceptance run, and it may still refuse.
    assert gates["synthetic_fifth_adapter"]["status"] in ("PASS", "FAIL", "NOT_RUN")
    # TG17.14: the four-domain live acquisition ran and its record is bound to this tree, so
    # the last unmeasured scientific gate now reads PASS. Asserted as the recorded state
    # rather than pinned to PASS forever: the record binds source digests, and an edit to any
    # acquisition module correctly returns this to NOT_RUN until the run is repeated.
    assert gates["live_sources"]["status"] in ("PASS", "REFUSED", "FAIL", "NOT_RUN")
    assert gates["live_sources"]["blocking"] is False
    # The verdict is unmoved, and by exactly one thing: TG17.11's declared scientific limit.
    # Five of seven gates now clear and the release is still refused, which is the point --
    # a refusal blocks as a failure does, and clearing four archives does not buy past it.
    assert gates["scale_shape_calibration"]["status"] == "REFUSED"
    assert qualified["verdict"] == "NOT_RELEASEABLE"


def test_the_fifth_adapter_gate_publishes_the_glue_count_it_does_not_block_on(qualified):
    """TG17.13. Two numbers, and collapsing them would lose the one that is actionable.

    The installation claim is asserted and is zero: no framework source names the fifth adapter.
    The standing glue count is reported, because "glue must trend to zero rather than merely move
    files" is a property of the whole surface over time rather than of one installation. A gate
    that blocked on it would make an unrelated archive's acquisition semantics a release decision;
    a gate that passed without printing it would let glue accumulate behind a green light.
    """
    evidence = qualified["extension_evidence"]
    gate = {item["gate_id"]: item for item in qualified["gates"]}["synthetic_fifth_adapter"]
    assert evidence["audit_read_live_here"] is True

    if evidence["status"] != "PASS":
        assert evidence["reasons"], "a gate that has not cleared must say why"
        assert "Must be measured by the extension conformance test" in gate["detail"]
        return

    recorded = evidence["recorded"]
    assert recorded["installation_required_framework_edits"] == 0
    standing = recorded["adapter_specific_framework_edits"]
    assert isinstance(standing, int)
    # The published count is in the detail a reader of the record actually sees.
    assert "installation required 0 framework edits" in gate["detail"]
    assert "Standing adapter-specific glue in those surfaces is %d" % standing in gate["detail"]
    for item in recorded["glue"]:
        assert item["source"] in gate["detail"], "a counted instance is named, not just totalled"


def test_scientist_action_measurements_are_not_invented(qualified):
    """TG18.5 slice 4 gave this a channel, and the guard is now about provenance, not absence.

    Before the channel existed the only honest answer was `NOT_MEASURED`, and this test asserted
    that literal. A measurement arriving does not weaken the rule it was protecting: the record
    may report a number a rendered run handed it, and may still never produce one itself. So the
    two admissible shapes are asserted, and the field no browser can observe is asserted to stay
    unmeasured in both of them.
    """
    actions = qualified["scientist_actions"]
    assert actions["definition"] == "visible researcher actions from a clean browser session"
    # A source-edit audit belongs to `synthetic_fifth_adapter`. No browser run may award it.
    assert actions["adapter_specific_framework_edits"] == "NOT_MEASURED"

    if actions["status"] == "NOT_MEASURED":
        assert actions["refusal_explanation_time_seconds"] == "NOT_MEASURED"
        assert actions["reasons"], "an unmeasured count must say why it has none"
    else:
        assert actions["status"] == "MEASURED"
        assert isinstance(actions["actions_to_a_completed_run"], int)
        # Durations are carried, and only under a name that says nothing asserts them.
        assert not any(key.endswith("_seconds") for key in actions)
        assert any(key.endswith("_seconds_unasserted") for key in actions)


def test_qualification_record_is_self_hashed_and_tampering_is_detected(qualified):
    assert verify_qualification_record(qualified)["integrity"] == "VERIFIED"
    tampered = copy.deepcopy(qualified)
    # The edit must actually change the record. This assignment used to read `= "PASS"`, and
    # when TG17.14's live run turned the last gate PASS the tamper became a no-op writing the
    # value already there: the record verified, and the test failed for the one reason a
    # tamper test must never fail. The sentinel cannot collide with a real status.
    assert tampered["gates"][-1]["status"] != "TAMPERED"
    tampered["gates"][-1]["status"] = "TAMPERED"
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
    # The rehearsal resolves the two gates it performs and must not quietly absorb the ones
    # it does not. Until TG17.14 the survivor was `live_sources`, still unmeasured; now that
    # it has run, the gate the rehearsal cannot touch is the refused one, and a rehearsal
    # that returned an all-clear body would be the failure this assertion guards against.
    unresolved = [row for row in body["gates"] if row["status"] != "PASS"]
    assert unresolved, "the rehearsal must not present every gate as passing"
    assert any(row["status"] == "REFUSED" for row in unresolved)
    assert body["verdict"] == "NOT_RELEASEABLE"
    assert body["record_kind"] == RECORD_KIND


# ------------------------------------------ TG17.11 slice 5: the scale/shape calibration gate


def test_the_scale_shape_gate_is_refused_rather_than_unimplemented_or_passed(qualified):
    """What the gate has actually earned, which is none of the three easy answers.

    It is no longer `NOT_IMPLEMENTED`: a registered calibration exists. It is not `PASS`: that
    calibration is a measurement on built fixtures at an inventory size no declared plan reaches,
    and this module does not read its result. It is not `FAIL`: nothing broke. `REFUSED` is the
    status this apparatus already reserves for a declared scientific limit that blocks release
    exactly as a failure does, and it is the honest one here.
    """
    gate = {row["gate_id"]: row for row in qualified["gates"]}["scale_shape_calibration"]
    assert gate["status"] == "REFUSED"
    assert gate["blocking"] is True
    assert SCALE_SHAPE_CALIBRATION in gate["detail"]
    assert qualified["verdict"] == "NOT_RELEASEABLE"


def test_the_gate_states_applicability_and_does_not_claim_the_calibration_s_result(qualified):
    """The two facts slice 5 exists to keep apart.

    A calibration that runs and meets its targets on fixtures is one fact; whether a declared plan
    can use the method is another. Reporting the first as the second is how an unusable mode gets
    a green gate, so the record carries the entry point, says it did not execute it, and turns on
    a quantity it computed itself.
    """
    facts = qualified["scale_shape_calibration"]
    assert facts["calibration"] == SCALE_SHAPE_CALIBRATION
    assert facts["calibration_executed_here"] is False
    assert "does not assert one" in facts["claim_boundary"]
    assert not any(key.startswith(("power", "rejection", "full_recovery")) for key in facts), (
        "the gate must not carry a power number it did not measure")


def test_the_declared_null_cannot_reject_at_any_inventory_size_it_will_draw_from(qualified):
    """The finding the gate turns on, asserted as the inequality rather than as two constants.

    The resolution bound rises with the correction's stringency and the enumeration bound is where
    a uniform draw stops being demonstrable. They do not meet, and the gap is an order of
    magnitude: every inventory the declared null will draw from is too small to reject, and every
    inventory large enough to reject is one it refuses to draw from. That is a property of the
    declared null, not of the fixtures, so no fixture work can move it.
    """
    facts = qualified["scale_shape_calibration"]
    assert facts["largest_drawable_inventory"] < facts["minimum_resolvable_family"]
    assert facts["declared_null_can_ever_reject"] is False
    assert facts["minimum_resolvable_family"] > 10 * facts["largest_drawable_inventory"]
    assert facts["inference_the_calibration_uses"] != facts["inference_the_manifests_declare"]


def test_both_declared_domain_families_carry_the_null_s_own_refusal_verbatim(qualified):
    """Neither candidate family is refused for its size; each is refused before size is reached.

    The reasons are taken from the null rather than restated here, so a change to either refusal
    surfaces as a changed qualification record instead of as a sentence that quietly went stale.
    """
    families = qualified["scale_shape_calibration"]["declared_families"]
    assert [row["pairings"] for row in families] == [6, 3]
    assert all(row["null_refusal"] and not row["reaches_resolvable_size"] for row in families)
    assert "admits exactly one" in families[0]["null_refusal"]
    assert "admits none" in families[1]["null_refusal"]
    for name, domains in SCALE_SHAPE_DOMAIN_FAMILIES:
        adapters = [adapter_for_domain(domain) for domain in domains]
        if name == "admitting_triple_all_pairs":
            assert all("scale_partner_reassignment" in item.admissible_nulls
                       for item in adapters), (
                "the triple must be the domains that do admit the null, or the finding is about "
                "the wrong inventory")


def test_the_gate_is_decided_without_acquiring_or_calibrating_anything():
    """Applicability is decidable in milliseconds, which is why it is decided before power.

    A gate that had to run the calibration would put a three-minute benchmark inside an HTTP
    route, and a gate that skipped the question would report an absent method rather than an
    inapplicable one.
    """
    started = time.perf_counter()
    facts = scale_shape_applicability()
    assert time.perf_counter() - started < 1.0
    assert facts == qualification_plan()["scale_shape_calibration"]
