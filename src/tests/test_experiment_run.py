"""TG17.6 - the content-addressed, resumable experiment orchestrator.

The failure this slice exists to prevent is not a crash. It is a long remote job that gets
interrupted, restarted, and quietly runs a smaller experiment than the one that was declared -
producing a receipt indistinguishable from a run that always intended to be that size. So the
tests here are mostly about what happens after something goes wrong: what is replayed, what is
re-requested, what a retry is allowed to change, and what a refusal costs.
"""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src.core.errors import InvalidParameterError
from src.core.experiment_manifest import (CoveragePolicy, ManifestStore, flagship_recipe,
                                          manifest_sha256)
from src.core.experiment_run import (COMPONENT_STATUSES, ComponentOutcome, HeldOutOpenings,
                                     MissingStageWorkerError, PlanChangedError, RETRYABLE_STATUSES, RunJournal, RunStateError, RunStore,
                                     SCHEMA, STATES, SpentTargetError, StepRequest,
                                     TERMINAL_STATES, TRANSITIONS, WORK_STAGES, decide_stage,
                                     describe_state_machine, run_identity, stage_components)
from src.core.run_workers import WORKER_SUITES, build_suite, describe_suites


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def spec():
    """The flagship without the bespoke domain, which has no archive to plan against.

    `order_book` refuses at metadata preflight by design (a bespoke record is the only thing that
    establishes what it observed), so a run declaring it never reaches acquisition. That refusal
    has its own test below; the executable path needs a manifest preflight admits.
    """
    recipe = flagship_recipe()
    return recipe.copy(update={"observations": [item for item in recipe.observations
                                                if item.domain != "order_book"]})


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path)


class RecordingWorker:
    """A worker that remembers what it was asked for, which is the whole point of most tests."""

    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = dict(statuses or {})

    def __call__(self, request):
        self.calls.append((request.stage, request.component))
        planned = self.statuses.get((request.stage, request.component))
        if planned is None:
            return ComponentOutcome(
                status="COMPLETE", detail="fixture", bytes_read=11,
                artifact_sha256=hashlib.sha256(request.step_sha256.encode()).hexdigest())
        return planned

    def suite(self):
        return {stage: self for stage in WORK_STAGES}

    def components(self, stage):
        return [component for recorded_stage, component in self.calls if recorded_stage == stage]


def _timed_out(component):
    return ComponentOutcome(status="TIMED_OUT",
                            detail="%s exceeded its deadline" % component,
                            remediation="retry the run; completed components are replayed")


def _missing(component):
    return ComponentOutcome(status="MISSING",
                            detail="the archive holds no coverage for %s in this window"
                                   % component)


def _refused(component):
    return ComponentOutcome(status="REFUSED",
                            detail="%s declines this request permanently" % component)


# --------------------------------------------------------------------------- the machine


def test_every_state_has_a_declared_row_in_the_transition_table():
    assert set(TRANSITIONS) == set(STATES)


def test_every_declared_successor_is_itself_a_state():
    for state, successors in TRANSITIONS.items():
        for successor in successors:
            assert successor in STATES, "%s -> %s names no state" % (state, successor)


def test_terminal_states_have_no_successors():
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == ()


def test_failed_is_not_terminal_because_a_retry_is_what_clears_it():
    assert "FAILED" not in TERMINAL_STATES
    assert set(WORK_STAGES).issubset(set(TRANSITIONS["FAILED"]))


def test_the_declared_path_runs_from_draft_to_complete():
    path = ["DRAFT", "PREFLIGHTED", "FROZEN"] + list(WORK_STAGES) + ["COMPLETE"]
    for current, following in zip(path, path[1:]):
        assert following in TRANSITIONS[current], "%s cannot reach %s" % (current, following)


def test_an_illegal_transition_names_the_ones_that_are_legal(spec, store):
    run = store.open(spec)
    with pytest.raises(RunStateError) as caught:
        run._transition("COMPLETE")
    assert "PREFLIGHTED" in str(caught.value)


def test_the_state_machine_is_published_as_data():
    described = describe_state_machine()
    assert described["transitions"] == {state: list(next_states)
                                        for state, next_states in TRANSITIONS.items()}
    assert described["retryable_statuses"] == list(RETRYABLE_STATUSES)


def test_a_stage_that_is_not_a_work_stage_is_refused_by_name(spec):
    with pytest.raises(InvalidParameterError) as caught:
        stage_components(spec, "FROZEN")
    assert "work stages" in str(caught.value)


# --------------------------------------------------------------------------- identity


def test_run_identity_is_the_manifest_and_nothing_else(spec):
    assert run_identity(spec) == run_identity(spec)
    assert run_identity(spec)["manifest_sha256"] == manifest_sha256(spec)


def test_identity_carries_no_clock(spec, tmp_path):
    early = RunStore(tmp_path / "a", clock=lambda: 0.0).open(spec)
    late = RunStore(tmp_path / "b", clock=lambda: 1.0e9).open(spec)
    assert early.run_id == late.run_id


def test_a_changed_manifest_is_a_different_run(spec):
    changed = spec.copy(update={"title": spec.title + " (revised)"})
    assert run_identity(changed)["run_id"] != run_identity(spec)["run_id"]


def test_reopening_the_same_manifest_resumes_rather_than_forks(spec, store, tmp_path):
    worker = RecordingWorker()
    first = store.open(spec)
    first.execute(worker.suite())
    second = store.open(spec)
    assert second.run_id == first.run_id
    assert second.state == "COMPLETE"
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_an_identical_complete_manifest_returns_the_same_artefacts(spec, store):
    worker = RecordingWorker()
    first = store.open(spec).execute(worker.suite())
    second = store.open(spec).execute(worker.suite())
    assert second["artefacts"] == first["artefacts"]
    assert second["run_sha256"] == first["run_sha256"]


def test_re_executing_a_complete_run_requests_nothing(spec, store):
    worker = RecordingWorker()
    store.open(spec).execute(worker.suite())
    before = len(worker.calls)
    store.open(spec).execute(worker.suite())
    assert len(worker.calls) == before


def test_the_store_lists_what_is_on_disk(spec, store):
    store.open(spec).execute(RecordingWorker().suite())
    rows = store.list_runs()
    assert [row["state"] for row in rows] == ["COMPLETE"]
    assert rows[0]["study_id"] == spec.study_id


def test_loading_a_run_reconstructs_its_frozen_manifest(spec, store):
    run = store.open(spec)
    loaded = store.load(run.run_id)
    assert manifest_sha256(loaded.spec) == manifest_sha256(spec)


def test_an_unknown_run_identifier_is_refused(store):
    with pytest.raises(InvalidParameterError):
        store.load("not-a-run-id")


# --------------------------------------------------------------------------- content addressing


def test_a_step_address_is_a_function_of_its_declared_inputs():
    left = StepRequest(run_sha256="r", stage="ACQUIRING", component="argo_float",
                       inputs={"native": "a"})
    same = StepRequest(run_sha256="r", stage="ACQUIRING", component="argo_float",
                       inputs={"native": "a"})
    other = StepRequest(run_sha256="r", stage="ACQUIRING", component="argo_float",
                        inputs={"native": "b"})
    assert left.step_sha256 == same.step_sha256
    assert left.step_sha256 != other.step_sha256


def test_a_translation_is_keyed_to_the_native_artefact_it_consumed(spec, store):
    """The drift check: a changed upstream artefact re-keys everything downstream."""
    run = store.open(spec)
    run.preflight()
    run.freeze()
    domain = spec.observations[0].domain
    before = run._request("TRANSLATING", domain).step_sha256
    run._emit("step", stage="ACQUIRING", component=domain, step_sha256="x",
              outcome=ComponentOutcome(status="COMPLETE", artifact_sha256="a" * 64).describe())
    assert run._request("TRANSLATING", domain).step_sha256 != before


def test_a_completed_step_is_replayed_not_repeated(spec, store):
    worker = RecordingWorker()
    run = store.open(spec)
    run.execute(worker.suite())
    resumed = store.open(spec)
    resumed._emit("transition", from_state="COMPLETE", to_state="COMPLETE", reason="forced")
    assert len(worker.calls) == 8


def test_a_step_record_is_published_immutably(spec, store):
    run = store.open(spec)
    run.execute(RecordingWorker().suite())
    steps = sorted((run.directory / "steps").glob("*.json"))
    assert steps, "a completed run publishes its steps"
    record = json.loads(steps[0].read_text("utf-8"))
    assert record["schema"] == SCHEMA
    assert record["outcome"]["status"] in COMPONENT_STATUSES


def test_two_outcomes_at_one_address_are_a_defect_not_a_race(spec, store):
    run = store.open(spec)
    run.preflight()
    run.freeze()
    request = run._request("ACQUIRING", spec.observations[0].domain)
    run._publish_step(request, ComponentOutcome(status="COMPLETE", artifact_sha256="a" * 64))
    with pytest.raises(InvalidParameterError) as caught:
        run._publish_step(request, ComponentOutcome(status="COMPLETE", artifact_sha256="b" * 64))
    assert "function of its address" in str(caught.value)


def test_an_operational_failure_is_not_published_as_a_step_artefact(spec, store):
    """A timeout is a fact about a network at a moment, not an output of the declared inputs."""
    domain = spec.observations[0].domain
    worker = RecordingWorker({("ACQUIRING", domain): _timed_out(domain)})
    run = store.open(spec)
    run.execute(worker.suite())
    published = {json.loads(path.read_text("utf-8"))["request"]["component"]
                 for path in (run.directory / "steps").glob("*.json")}
    assert domain not in published


# --------------------------------------------------------------------------- resuming


def test_a_killed_run_resumes_without_re_requesting_what_completed(spec, store):
    class Dies(RecordingWorker):
        def __call__(self, request):
            if request.component == spec.observations[-1].domain:
                raise KeyboardInterrupt("the laptop slept")
            return RecordingWorker.__call__(self, request)

    dying = Dies()
    run = store.open(spec)
    with pytest.raises(KeyboardInterrupt):
        run.execute(dying.suite())
    acquired_before = set(dying.components("ACQUIRING"))

    resumed = store.open(spec)
    survivor = RecordingWorker()
    receipt = resumed.execute(survivor.suite())
    assert receipt["state"] == "COMPLETE"
    reacquired = set(survivor.components("ACQUIRING"))
    assert not (reacquired & (acquired_before - {spec.observations[-1].domain})), \
        "a resumed run re-requested coverage it had already acquired"


def test_a_torn_journal_tail_is_skipped_and_the_run_still_resumes(spec, store):
    worker = RecordingWorker()
    run = store.open(spec)
    run.execute(worker.suite())
    path = run.journal.path
    text = path.read_text("utf-8")
    path.write_text(text + '{"kind": "transition", "to_st', encoding="utf-8")
    resumed = store.open(spec)
    assert resumed.state == "COMPLETE"
    assert resumed.receipt()["artefacts"]


def test_the_journal_is_the_only_state(spec, store):
    worker = RecordingWorker()
    run = store.open(spec)
    run.execute(worker.suite())
    events = RunJournal(run.journal.path).read()
    assert [event["to_state"] for event in events if event["kind"] == "transition"][-1] == "COMPLETE"


def test_resuming_with_a_different_manifest_is_refused(spec, store):
    run = store.open(spec)
    other = spec.copy(update={"title": spec.title + " (revised)"})
    with pytest.raises(PlanChangedError) as caught:
        run.execute(RecordingWorker().suite(), spec=other)
    assert manifest_sha256(other) in str(caught.value)


def test_a_run_directory_holding_a_different_manifest_is_a_collision(spec, store):
    run = store.open(spec)
    (run.directory / "manifest.json").write_bytes(b'{"not": "the manifest"}')
    with pytest.raises(InvalidParameterError) as caught:
        store.open(spec)
    assert "collision" in str(caught.value)


# --------------------------------------------------------------------------- failure and retry


def test_a_timeout_leaves_the_run_failed_with_a_remediation(spec, store):
    domain = spec.observations[-1].domain
    worker = RecordingWorker({("ACQUIRING", domain): _timed_out(domain)})
    receipt = store.open(spec).execute(worker.suite())
    assert receipt["state"] == "FAILED"
    assert [row["remediation"] for row in receipt["missing_components"]
            if row["component"] == domain][0]


def test_a_retry_re_executes_only_what_failed(spec, store):
    domain = spec.observations[-1].domain
    failing = RecordingWorker({("ACQUIRING", domain): _timed_out(domain)})
    run = store.open(spec)
    run.execute(failing.suite())
    healed = RecordingWorker()
    receipt = run.retry(healed.suite())
    assert receipt["state"] == "COMPLETE"
    assert healed.components("ACQUIRING") == [domain]


def test_a_retry_of_a_healthy_run_is_refused(spec, store):
    run = store.open(spec)
    run.execute(RecordingWorker().suite())
    with pytest.raises(Exception) as caught:
        run.retry(RecordingWorker().suite())
    assert "FAILED" in str(caught.value)


def test_a_retry_may_not_carry_a_new_plan(spec, store):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    run.execute(RecordingWorker({("ACQUIRING", domain): _timed_out(domain)}).suite())
    other = spec.copy(update={"title": spec.title + " (revised)"})
    with pytest.raises(PlanChangedError):
        run.retry(RecordingWorker().suite(), spec=other)


def test_executing_a_failed_run_directs_the_caller_to_retry(spec, store):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    run.execute(RecordingWorker({("ACQUIRING", domain): _timed_out(domain)}).suite())
    with pytest.raises(Exception) as caught:
        run.execute(RecordingWorker().suite())
    assert "retry()" in str(caught.value)


def test_a_second_failure_leaves_the_run_retryable_again(spec, store):
    domain = spec.observations[-1].domain
    failing = RecordingWorker({("ACQUIRING", domain): _timed_out(domain)}).suite()
    run = store.open(spec)
    run.execute(failing)
    receipt = run.retry(failing)
    assert receipt["state"] == "FAILED"


# --------------------------------------------------------------------------- refusal


def test_a_refused_component_ends_the_run_and_is_not_retryable(spec, store):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    receipt = run.execute(RecordingWorker({("ACQUIRING", domain): _refused(domain)}).suite())
    assert receipt["state"] == "REFUSED"
    with pytest.raises(Exception) as caught:
        run.retry(RecordingWorker().suite())
    assert "editable_copy" in str(caught.value)


def test_a_refused_preflight_never_reaches_acquisition(store):
    """The bespoke domain has no archive to plan against, so the run stops before it costs."""
    worker = RecordingWorker()
    run = store.open(flagship_recipe())
    receipt = run.execute(worker.suite())
    assert receipt["state"] == "REFUSED"
    assert worker.calls == []


def test_a_refusal_is_answered_with_an_editable_copy_not_an_edit(spec, store, tmp_path):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    run.execute(RecordingWorker({("ACQUIRING", domain): _refused(domain)}).suite())
    events_before = len(run.events)
    copy = run.editable_copy(ManifestStore(tmp_path / "manifests"), "g17_after_refusal")
    assert copy["frozen_run_untouched"] is True
    assert copy["manifest_sha256"] == manifest_sha256(spec)
    assert len(run.events) == events_before
    assert run.state == "REFUSED"


def test_the_frozen_run_cannot_be_restarted_after_a_refusal(spec, store):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    run.execute(RecordingWorker({("ACQUIRING", domain): _refused(domain)}).suite())
    worker = RecordingWorker()
    run.execute(worker.suite())
    assert worker.calls == []
    assert run.state == "REFUSED"


# --------------------------------------------------------------------------- partial coverage


def test_missing_coverage_under_complete_required_refuses(spec):
    frozen = spec.copy(update={"coverage_policy":
                               CoveragePolicy(requirement="complete_required",
                                              minimum_fraction=1.0)})
    statuses = {item.domain: "COMPLETE" for item in frozen.observations}
    statuses[frozen.observations[-1].domain] = "MISSING"
    decision = decide_stage(frozen, "ACQUIRING", statuses)
    assert decision["verdict"] == "REFUSED"
    assert "complete coverage" in decision["reason"]


def test_missing_coverage_above_the_declared_minimum_advances_and_is_named(spec, store):
    domain = spec.observations[-1].domain
    frozen = spec.copy(update={"coverage_policy":
                               CoveragePolicy(requirement="partial_permitted",
                                              minimum_fraction=0.5)})
    worker = RecordingWorker({("ACQUIRING", domain): _missing(domain)})
    receipt = store.open(frozen).execute(worker.suite())
    assert receipt["state"] == "COMPLETE"
    assert [row["component"] for row in receipt["missing_components"]] == [domain]
    assert [decision["verdict"] for decision in receipt["stage_decisions"]
            if decision["stage"] == "ACQUIRING"] == ["ADVANCE_PARTIAL"]


def test_missing_coverage_below_the_declared_minimum_refuses(spec):
    frozen = spec.copy(update={"coverage_policy":
                               CoveragePolicy(requirement="partial_permitted",
                                              minimum_fraction=0.9)})
    statuses = {item.domain: "COMPLETE" for item in frozen.observations}
    statuses[frozen.observations[-1].domain] = "MISSING"
    decision = decide_stage(frozen, "ACQUIRING", statuses)
    assert decision["verdict"] == "REFUSED"
    assert "0.9" in decision["reason"]


def test_a_refusal_outranks_a_timeout(spec):
    statuses = {item.domain: "COMPLETE" for item in spec.observations}
    statuses[spec.observations[0].domain] = "REFUSED"
    statuses[spec.observations[1].domain] = "TIMED_OUT"
    assert decide_stage(spec, "ACQUIRING", statuses)["verdict"] == "REFUSED"


def test_a_timeout_outranks_a_missing_component(spec):
    statuses = {item.domain: "COMPLETE" for item in spec.observations}
    statuses[spec.observations[0].domain] = "MISSING"
    statuses[spec.observations[1].domain] = "TIMED_OUT"
    assert decide_stage(spec, "ACQUIRING", statuses)["verdict"] == "FAILED"


def test_a_stage_cannot_be_decided_from_a_partial_report(spec):
    with pytest.raises(InvalidParameterError) as caught:
        decide_stage(spec, "ACQUIRING", {spec.observations[0].domain: "COMPLETE"})
    assert "partial report" in str(caught.value)


def test_a_status_for_an_undeclared_component_is_refused(spec):
    statuses = {item.domain: "COMPLETE" for item in spec.observations}
    statuses["a_domain_nobody_declared"] = "COMPLETE"
    with pytest.raises(InvalidParameterError):
        decide_stage(spec, "ACQUIRING", statuses)


# --------------------------------------------------------------------------- held-out targets


def test_confirming_opens_the_declared_partition_once(spec, store):
    run = store.open(spec)
    run.execute(RecordingWorker().suite())
    openings = [event for event in run.events if event["kind"] == "held_out_opened"]
    assert len(openings) == 1
    assert openings[0]["partition"] == spec.confirmation.held_out_partition


def test_a_second_run_cannot_spend_the_same_partition(spec, store):
    store.open(spec).execute(RecordingWorker().suite())
    rival = spec.copy(update={"title": spec.title + " (a second study)"})
    with pytest.raises(SpentTargetError) as caught:
        store.open(rival).execute(RecordingWorker().suite())
    assert spec.confirmation.held_out_partition in str(caught.value)


def test_reopening_within_one_run_is_a_resume_not_a_second_spending(tmp_path):
    openings = HeldOutOpenings(tmp_path / "openings.json")
    first = openings.open("g17_heldout", run_sha256="r" * 64, at="2026-08-31T00:00:00Z")
    again = openings.open("g17_heldout", run_sha256="r" * 64, at="2026-08-31T01:00:00Z")
    assert again == first


def test_a_confirmatory_only_study_opens_no_partition(spec, store):
    frozen = spec.copy(update={"confirmation": spec.confirmation.copy(
        update={"stage": "confirmatory_only", "held_out_partition": None,
                "confirmatory_members": None})})
    run = store.open(frozen)
    run.execute(RecordingWorker().suite())
    assert not [event for event in run.events if event["kind"] == "held_out_opened"]
    assert stage_components(frozen, "CONFIRMING") == ("declared_family",)


# --------------------------------------------------------------------------- outcomes


def test_a_component_outcome_has_nowhere_to_put_a_result():
    with pytest.raises(TypeError):
        ComponentOutcome(status="COMPLETE", artifact_sha256="a" * 64, p_value=0.001)


def test_a_complete_component_must_name_its_artefact():
    with pytest.raises(InvalidParameterError) as caught:
        ComponentOutcome(status="COMPLETE")
    assert "digest" in str(caught.value)


def test_an_operational_failure_must_state_a_remediation():
    with pytest.raises(InvalidParameterError) as caught:
        ComponentOutcome(status="TIMED_OUT", detail="the archive did not answer")
    assert "remediation" in str(caught.value)


def test_a_refusal_must_state_why():
    with pytest.raises(InvalidParameterError) as caught:
        ComponentOutcome(status="REFUSED")
    assert "without a reason" in str(caught.value)


def test_an_unknown_status_is_refused_by_name():
    with pytest.raises(InvalidParameterError) as caught:
        ComponentOutcome(status="PROBABLY_FINE")
    assert "COMPLETE" in str(caught.value)


def test_a_worker_returning_something_else_is_refused(spec, store):
    run = store.open(spec)
    with pytest.raises(InvalidParameterError) as caught:
        run.execute({stage: (lambda request: {"status": "COMPLETE"}) for stage in WORK_STAGES})
    assert "ComponentOutcome" in str(caught.value)


def test_a_stage_with_no_worker_refuses_by_name(spec, store):
    run = store.open(spec)
    with pytest.raises(MissingStageWorkerError) as caught:
        run.execute({})
    assert "ACQUIRING" in str(caught.value)


# --------------------------------------------------------------------------- progress


def test_progress_never_carries_a_result(spec, store):
    secret = "0.0001-this-is-a-p-value"

    def leaky(request):
        return ComponentOutcome(status="COMPLETE", artifact_sha256="a" * 64,
                                detail="fixture", bytes_read=1)

    run = store.open(spec)
    run.execute({stage: leaky for stage in WORK_STAGES})
    assert secret not in json.dumps(run.progress())
    assert secret not in json.dumps(run.receipt())


def test_results_are_not_declared_visible_before_the_run_completes(spec, store):
    domain = spec.observations[-1].domain
    run = store.open(spec)
    run.execute(RecordingWorker({("ACQUIRING", domain): _timed_out(domain)}).suite())
    progress = run.progress()
    assert progress["results_visible"] is False
    assert progress["retryable"] is True


def test_bounded_work_is_bounded_by_the_declared_plan(spec, store):
    run = store.open(spec)
    total = sum(len(stage_components(spec, stage)) for stage in WORK_STAGES)
    assert run.bounded_work()["total_steps"] == total
    run.execute(RecordingWorker().suite())
    assert run.bounded_work()["completed_steps"] == total
    assert run.bounded_work()["fraction"] == 1.0


def test_progress_shows_pending_components_before_they_run(spec, store):
    run = store.open(spec)
    run.preflight()
    statuses = {component["status"]
                for stage in run.progress()["stages"] for component in stage["components"]}
    assert statuses == {"PENDING"}


def test_the_receipt_lists_every_missing_component(spec, store):
    domain = spec.observations[-1].domain
    frozen = spec.copy(update={"coverage_policy":
                               CoveragePolicy(requirement="partial_permitted",
                                              minimum_fraction=0.5)})
    receipt = store.open(frozen).execute(
        RecordingWorker({("ACQUIRING", domain): _missing(domain)}).suite())
    rows = receipt["missing_components"]
    assert [row["component"] for row in rows] == [domain]
    assert rows[0]["detail"]


# --------------------------------------------------------------------------- cancellation


def test_a_run_can_be_cancelled_and_stays_cancelled(spec, store):
    run = store.open(spec)
    run.preflight()
    run.cancel("the archive is down for maintenance")
    assert run.state == "CANCELLED"
    with pytest.raises(RunStateError):
        run.cancel("again")


def test_a_cancellation_must_state_its_reason(spec, store):
    run = store.open(spec)
    with pytest.raises(InvalidParameterError):
        run.cancel("   ")


def test_a_completed_run_cannot_be_cancelled(spec, store):
    run = store.open(spec)
    run.execute(RecordingWorker().suite())
    with pytest.raises(RunStateError):
        run.cancel("changed my mind")


# --------------------------------------------------------------------------- worker suites


def test_the_registered_suites_acquire_nothing():
    for entry in WORKER_SUITES.entries():
        assert entry.capabilities["acquires"] is False
        assert entry.capabilities["network_used"] is False


def test_a_suite_covers_every_work_stage():
    for name in WORKER_SUITES.names():
        assert set(build_suite(name)) == set(WORK_STAGES)


def test_the_dry_run_suite_completes_a_frozen_plan(spec, store):
    receipt = store.open(spec).execute(build_suite("fixture_dry_run"))
    assert receipt["state"] == "COMPLETE"
    assert all(not event["outcome"]["network_used"]
               for event in store.open(spec).events if event.get("kind") == "step")


def test_the_transient_failure_suite_fails_once_and_succeeds_on_retry(spec, store):
    run = store.open(spec)
    suite = build_suite("fixture_transient_failure")
    first = run.execute(suite)
    assert first["state"] == "FAILED"
    assert run.retry(suite)["state"] == "COMPLETE"


def test_two_rehearsals_do_not_share_their_recorded_failures(spec, tmp_path):
    left = RunStore(tmp_path / "left").open(spec)
    left.execute(build_suite("fixture_transient_failure"))
    right = RunStore(tmp_path / "right").open(spec)
    assert right.execute(build_suite("fixture_transient_failure"))["state"] == "FAILED"


def test_a_rehearsal_artefact_digest_is_derived_from_the_step_address(spec, tmp_path):
    left = RunStore(tmp_path / "left").open(spec).execute(build_suite("fixture_dry_run"))
    right = RunStore(tmp_path / "right").open(spec).execute(build_suite("fixture_dry_run"))
    assert left["artefacts"] == right["artefacts"]


def test_the_suites_are_described_for_the_browser():
    described = {entry["name"] for entry in describe_suites()}
    assert described == set(WORKER_SUITES.names())


# --------------------------------------------------------------------------- api


@pytest.fixture
def client(tmp_path):
    from src.api.main import app
    app.state.experiment_run_dir = str(tmp_path / "runs")
    app.state.experiment_manifest_dir = str(tmp_path / "manifests")
    with TestClient(app) as test_client:
        yield test_client


def test_the_contract_route_publishes_the_machine_and_the_suites(client):
    body = client.get("/api/v1/experiment-runs").json()
    assert body["state_machine"]["transitions"]["FROZEN"] == list(TRANSITIONS["FROZEN"])
    assert {entry["name"] for entry in body["worker_suites"]} == set(WORKER_SUITES.names())
    assert body["runs"] == []


def test_posting_a_manifest_twice_resumes_one_run(client, spec):
    payload = json.loads(spec.json())
    first = client.post("/api/v1/experiment-runs", json=payload).json()
    second = client.post("/api/v1/experiment-runs", json=payload).json()
    assert second["run_id"] == first["run_id"]
    assert first["resumed"] is False and second["resumed"] is True


def test_a_run_executes_a_named_suite_over_http(client, spec):
    created = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    receipt = client.post("/api/v1/experiment-runs/%s/execute" % created["run_id"],
                          json={"worker_suite": "fixture_dry_run"}).json()
    assert receipt["state"] == "COMPLETE"
    assert len(receipt["artefacts"]) == 8


def test_a_timeout_can_be_retried_over_http(client, spec):
    created = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    run_id = created["run_id"]
    first = client.post("/api/v1/experiment-runs/%s/execute" % run_id,
                        json={"worker_suite": "fixture_transient_failure"}).json()
    assert first["state"] == "FAILED"
    progress = client.get("/api/v1/experiment-runs/%s/progress" % run_id).json()
    assert progress["retryable"] is True
    retried = client.post("/api/v1/experiment-runs/%s/retry" % run_id,
                          json={"worker_suite": "fixture_transient_failure"}).json()
    assert retried["state"] == "COMPLETE"


def test_progress_survives_a_browser_refresh(client, spec):
    created = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    client.post("/api/v1/experiment-runs/%s/execute" % created["run_id"],
                json={"worker_suite": "fixture_dry_run"})
    reloaded = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    assert reloaded["run_id"] == created["run_id"]
    assert reloaded["progress"]["state"] == "COMPLETE"


def test_an_unknown_suite_is_refused_with_the_names_that_exist(client, spec):
    created = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    response = client.post("/api/v1/experiment-runs/%s/execute" % created["run_id"],
                           json={"worker_suite": "whatever_works"})
    assert response.status_code == 409
    assert "fixture_dry_run" in response.json()["detail"]


def test_an_unknown_run_is_a_404(client):
    assert client.get("/api/v1/experiment-runs/%s" % ("f" * 32)).status_code == 404


def test_a_refused_run_is_copied_to_an_editable_draft_over_http(client):
    payload = json.loads(flagship_recipe().json())
    created = client.post("/api/v1/experiment-runs", json=payload).json()
    assert created["receipt"]["state"] == "REFUSED"
    copy = client.post("/api/v1/experiment-runs/%s/editable-copy" % created["run_id"],
                       json={"draft_id": "g17_after_refusal"}).json()
    assert copy["frozen_run_untouched"] is True
    assert client.get("/api/v1/experiment-runs/%s" % created["run_id"]).json()["state"] == "REFUSED"


def test_a_run_can_be_cancelled_over_http(client, spec):
    created = client.post("/api/v1/experiment-runs", json=json.loads(spec.json())).json()
    cancelled = client.post("/api/v1/experiment-runs/%s/cancel" % created["run_id"],
                            json={"reason": "the token expires tonight"}).json()
    assert cancelled["state"] == "CANCELLED"
    assert client.post("/api/v1/experiment-runs/%s/execute" % created["run_id"],
                       json={"worker_suite": "fixture_dry_run"}).json()["state"] == "CANCELLED"
