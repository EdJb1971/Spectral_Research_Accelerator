"""TG17.7 - the guided path, the ladder, and the single next legitimate action.

The failure this slice exists to prevent is not a broken button. It is a workbench that offers
four acquisition pages and a page of instructions, leaving the order of operations in the
researcher's head - because the order of operations *is* the scientific discipline. A family
priced after acquisition is priced knowing what the data looked like; a null chosen after the
statistic exists is not a null. A UI that permits those in any order has not made an error, it
has made the error undetectable, since no receipt can distinguish a plan that was declared from
one that was assembled.

So most of these tests are about what the path *refuses* to let a researcher slide past: that
one action is next rather than several, that a blocked step keeps its reason rather than
disappearing, that reading where a plan stands never starts it, and that walking the whole path
produces an executed plan and stops well short of evidence.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.core.composer_path import (COMPOSER_PATH, DURATION_PRESETS, EnvelopeIntegrityError,
                                    NO_TEMPLATE_REASON, SCHEMA, STAGE_LADDER, STEP_STATUSES,
                                    ComposerState, PathStep, StepVerdict, compose_state,
                                    describe_path, domain_menu, export_envelope, import_envelope,
                                    observation_templates, ordered_steps,
                                    preregistration_summary, stage_ladder, window_presets)
from src.core.errors import InvalidParameterError
from src.core.experiment_manifest import flagship_recipe, manifest_sha256, preflight_manifest


@pytest.fixture
def spec():
    """The flagship without the bespoke domain, which has no archive to plan against."""
    recipe = flagship_recipe()
    return recipe.copy(update={"observations": [item for item in recipe.observations
                                                if item.domain != "order_book"]})


@pytest.fixture
def state(spec):
    return compose_state(spec)


def _step(state, step_id):
    return next(row for row in state["steps"] if row["step_id"] == step_id)


# ------------------------------------------------------------------------- the path


def test_the_path_is_a_registry_rather_than_a_list_in_a_component():
    """Standard E1. An eighth step registers; it is not remembered in three places."""
    assert len(COMPOSER_PATH) == 7
    assert [step.step_id for step in ordered_steps()] == [
        "question", "domains", "observation", "preflight", "analysis",
        "freeze_and_run", "interpret"]


def test_the_path_is_ordered_by_its_ordinals_and_not_by_how_names_happen_to_sort():
    """The registry sorts by name, and `domains` sorts before `question`.

    Reading the workflow out of a registry in registry order would have silently reversed the
    first two steps - which is exactly the class of mistake the path exists to make impossible.
    """
    assert sorted(COMPOSER_PATH.names()) != [step.step_id for step in ordered_steps()]
    assert [step.ordinal for step in ordered_steps()] == [1, 2, 3, 4, 5, 6, 7]


def test_every_step_states_the_question_it_settles_and_what_it_does_not_license():
    for step in ordered_steps():
        assert step.question.endswith("?"), step.step_id
        assert step.settles and step.claim_boundary and step.controls, step.step_id
        assert step.action_label and step.action_route, step.step_id


def test_the_served_contract_carries_the_path_the_statuses_and_the_ladder():
    contract = describe_path()
    assert contract["schema"] == SCHEMA
    assert [row["step_id"] for row in contract["steps"]] == \
        [step.step_id for step in ordered_steps()]
    assert contract["step_statuses"] == list(STEP_STATUSES)
    assert contract["duration_presets"] == sorted(DURATION_PRESETS)
    assert [row["rung"] for row in contract["ladder"]] == [row["rung"] for row in STAGE_LADDER]


def test_a_step_verdict_cannot_carry_an_invented_status():
    with pytest.raises(InvalidParameterError):
        StepVerdict("NEARLY", "close enough")


# --------------------------------------------------------------- the single next action


def test_exactly_one_next_action_is_named_and_it_is_the_first_unsatisfied_step(state):
    """`next_action` is one field because two enabled buttons meaning two different
    scientific commitments cannot both be the next legitimate act."""
    unsatisfied = [row for row in state["steps"] if row["status"] != "SATISFIED"]
    assert state["next_action"]["step_id"] == unsatisfied[0]["step_id"]
    assert state["next_action"]["why"] == unsatisfied[0]["reason"]


def test_the_next_action_names_the_route_that_performs_it(state):
    action = state["next_action"]
    step = _step(state, action["step_id"])
    assert action["route"] == step["action_route"]
    assert action["label"] == step["action_label"]


def test_a_blocked_step_is_reported_as_blocked_rather_than_as_merely_not_done(spec):
    """"You have not done this" and "this cannot be done yet" are different sentences.

    Only one of them is the researcher's move, and a UI shown the wrong one sends them looking
    for a control that will not help.
    """
    state = compose_state(spec)
    assert _step(state, "interpret")["status"] == "BLOCKED"
    assert _step(state, "freeze_and_run")["status"] == "ACTION_REQUIRED"


def test_the_path_is_walkable_end_to_end_for_a_plan_metadata_admits(spec, tmp_path):
    """Steps one to five satisfied by declaration alone, before anything is acquired."""
    state = compose_state(spec)
    for step_id in ("question", "domains", "observation", "preflight", "analysis"):
        assert _step(state, step_id)["status"] == "SATISFIED", step_id
    assert state["satisfied"] == 5


def test_the_whole_flagship_is_blocked_at_preflight_and_says_which_domain_and_why():
    """The fourth domain has no public archive, so its coverage cannot be planned.

    This is the refusal remaining visible rather than the domain quietly disappearing from a
    study that then reports itself complete.
    """
    state = compose_state(flagship_recipe())
    step = _step(state, "preflight")
    assert step["status"] == "BLOCKED"
    assert any("order_book" in reason for reason in step["detail"]["refusals"])
    assert state["next_action"]["blocked"] is True


# ------------------------------------------------------------------- individual gates


def test_the_question_step_carries_what_its_mode_forbids(state):
    detail = _step(state, "question")["detail"]
    assert detail["mode"] == "calendar_aligned"
    assert "co-occurrence only" in detail["forbids"]


def test_the_domains_step_reports_what_each_declared_domain_breaks(state):
    detail = _step(state, "domains")["detail"]
    assert {row["domain"] for row in detail["domains"]} == \
        {"reanalysis", "argo_float", "tess_lightcurve"}
    assert "irregular_sampling" in detail["assumptions_broken"]
    for row in detail["domains"]:
        assert row["licence"], row["domain"]


def test_a_domain_this_instance_cannot_translate_blocks_rather_than_being_dropped(spec):
    """A manifest may name a domain this instance has no adapter for - after an import, say.

    The refusal is that its declared violations are unknown, so they cannot be inherited. The
    dangerous alternative is running the other three and calling it the declared study.
    """
    observations = [row.copy(update={"domain": "not_a_registered_domain"}) if index == 0 else row
                    for index, row in enumerate(spec.observations)]
    state = compose_state(spec.copy(update={"observations": observations}))
    step = _step(state, "domains")
    assert step["status"] == "BLOCKED"
    assert "not_a_registered_domain" in step["reason"]


def test_the_observation_step_reports_the_preset_a_window_happens_to_match(state):
    windows = {row["name"]: row for row in _step(state, "observation")["detail"]["windows"]}
    assert windows["week"]["preset"] == "week"
    assert windows["six_months"]["preset"] == "six_months"


def test_partial_coverage_is_the_frozen_policy_s_call_and_not_this_view_s(spec):
    """The same metadata verdict is satisfactory under one coverage rule and not the other."""
    strict = compose_state(spec.copy(update={
        "coverage_policy": spec.coverage_policy.copy(update={"requirement": "complete_required",
                                                             "minimum_fraction": 1.0})}))
    step = _step(strict, "preflight")
    assert step["status"] in ("SATISFIED", "ACTION_REQUIRED", "BLOCKED")
    if step["detail"]["status"] == "PARTIAL":
        assert step["status"] == "ACTION_REQUIRED"
        assert "does not admit" in step["reason"]


def test_the_analysis_step_refuses_a_null_a_declared_domain_does_not_admit(spec):
    """A null is what a p-value is measured against, so only the domain can admit it."""
    nulls = [spec.nulls[0].copy(update={"method": "seasonal_block_shift"})] + list(spec.nulls[1:])
    state = compose_state(spec.copy(update={"nulls": nulls}))
    step = _step(state, "analysis")
    assert step["status"] == "ACTION_REQUIRED"
    assert step["detail"]["inadmissible_nulls"], "the offending domain/null pairs must be named"


def test_the_analysis_step_reports_the_family_it_would_correct(state):
    detail = _step(state, "analysis")["detail"]
    assert detail["family"]["declared_members"] > 0
    assert detail["correction"] == "benjamini_yekutieli"
    assert detail["seeds"], "the seeds are part of the declaration, not of the run"


def test_the_freeze_step_says_that_opening_a_run_resumes_rather_than_creates(state):
    assert "resumable" in _step(state, "freeze_and_run")["reason"]


def test_interpretation_is_blocked_until_the_frozen_plan_has_actually_run(spec):
    blocked = compose_state(spec)
    assert _step(blocked, "interpret")["status"] == "BLOCKED"
    complete = compose_state(spec, run={"state": "COMPLETE", "run_id": "a" * 32,
                                        "artefacts": {"ACQUIRING/reanalysis": "d" * 64}})
    assert _step(complete, "interpret")["status"] == "ACTION_REQUIRED"
    assert "separate, recorded act" in _step(complete, "interpret")["reason"]


def test_a_terminal_run_blocks_the_freeze_step_rather_than_inviting_an_edit(spec):
    refused = compose_state(spec, run={"state": "REFUSED", "run_id": "b" * 32})
    step = _step(refused, "freeze_and_run")
    assert step["status"] == "BLOCKED"
    assert "editable copy" in step["reason"]


# ------------------------------------------------------------------------- the ladder


def test_the_ladder_separates_four_things_a_researcher_can_possess():
    assert [row["rung"] for row in STAGE_LADDER] == \
        ["acquired_material", "experiment_run", "finding", "admitted_evidence"]
    for rung in STAGE_LADDER:
        assert rung["is"] and rung["is_not"] and rung["gate"], rung["rung"]


def test_no_rung_is_reached_by_a_plan_that_has_not_run(spec):
    ladder = compose_state(spec)["ladder"]
    assert not any(row["reached"] for row in ladder)
    assert all(row["why_not"] for row in ladder)


def test_acquiring_material_does_not_by_itself_reach_an_executed_run(spec):
    """Downloading four archives feels like having a study. It is the first rung only."""
    ladder = {row["rung"]: row for row in compose_state(spec, run={
        "state": "ACQUIRING", "run_id": "c" * 32,
        "artefacts": {"ACQUIRING/reanalysis": "e" * 64}})["ladder"]}
    assert ladder["acquired_material"]["reached"] is True
    assert ladder["experiment_run"]["reached"] is False


def test_a_completed_run_reaches_neither_a_finding_nor_admitted_evidence(spec):
    """The two rungs this surface cannot move you across, whatever it shows."""
    ladder = {row["rung"]: row for row in compose_state(spec, run={
        "state": "COMPLETE", "run_id": "f" * 32,
        "artefacts": {"ACQUIRING/reanalysis": "1" * 64}})["ladder"]}
    assert ladder["experiment_run"]["reached"] is True
    assert ladder["finding"]["reached"] is False
    assert ladder["admitted_evidence"]["reached"] is False
    assert "findings instrument" in ladder["finding"]["why_not"]


def test_the_state_never_reports_a_measured_value(state):
    """A path status that could carry a result is a path status that can leak one.

    The distinction the check has to make is between arithmetic and measurement. The family
    expansion legitimately reports a `p_value_floor` and a `surrogates_required`, and neither is
    a result: both are computable from the declared ensemble before a single byte exists, and
    reporting them *before* acquisition is the whole point of pricing a family. What must never
    appear is a value that required data - a measured statistic, an observed p-value, an effect.
    """
    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                for found in keys(item):
                    yield found
        elif isinstance(value, list):
            for item in value:
                for found in keys(item):
                    yield found

    found = set(keys(state))
    for forbidden in ("observed_p_value", "measurement", "effect_size", "correlation",
                      "test_statistic", "rejected", "significant"):
        assert forbidden not in found, forbidden
    text = json.dumps(state)
    assert "p_value_floor" in text, (
        "the declared floor is arithmetic about the ensemble and belongs here; if this ever "
        "starts meaning an observed p-value, this test must fail rather than be updated")


# ------------------------------------------------------------------ presets and menu


def test_a_preset_resolves_to_explicit_instants_on_the_server():
    """A manifest storing "six months" would mean different things on different days, and its
    content address would not change when it did."""
    presets = {row["preset"]: row for row in
               window_presets("2024-01-01T00:00:00Z")["presets"]}
    assert set(presets) == set(DURATION_PRESETS)
    assert presets["week"]["start_utc"] == "2024-01-01T00:00:00Z"
    assert presets["week"]["end_utc"] == "2024-01-08T00:00:00Z"
    assert presets["six_months"]["end_utc"] == "2024-07-01T00:00:00Z"


def test_a_preset_anchor_that_is_not_an_instant_is_refused_by_name():
    with pytest.raises(InvalidParameterError):
        window_presets("next tuesday")


def test_every_registered_domain_appears_in_the_menu_with_what_it_breaks():
    """Nothing is filtered out: a control that vanishes when it becomes inadmissible is
    indistinguishable from one that was never offered."""
    from src.core.experiment_adapter import EXPERIMENT_ADAPTERS

    menu = domain_menu(selected=["argo_float"])
    assert {row["domain"] for row in menu["domains"]} == \
        {entry.value.domain for entry in EXPERIMENT_ADAPTERS.entries()}
    assert menu["minimum_domains"] == 2
    selected = [row for row in menu["domains"] if row["selected"]]
    assert [row["domain"] for row in selected] == ["argo_float"]
    for row in menu["domains"]:
        assert "breaks" in row and "licence" in row


def test_a_domain_with_no_declared_observation_is_disabled_with_the_reason_not_hidden():
    """An adapter says how a domain is translated; it does not say what is measured, in which
    units, in which role, from which record. A menu that guessed would invent the observation."""
    menu = domain_menu()
    for row in menu["domains"]:
        assert row["selectable"] is (row["observation"] is not None)
        assert bool(row["unavailable_reason"]) is (row["observation"] is None)
    assert "inventing the observation" in NO_TEMPLATE_REASON
    assert set(observation_templates()) <= {row["domain"] for row in menu["domains"]}


# --------------------------------------------------------------- preregistration text


def test_the_preregistration_summary_is_generated_from_the_bytes_that_are_hashed(spec):
    summary = preregistration_summary(spec)
    assert summary["manifest_sha256"] == manifest_sha256(spec)
    joined = " ".join(summary["sentences"])
    assert spec.correction in joined
    assert str(spec.alpha) in joined
    assert spec.nulls[0].method in joined
    assert "seeds" in joined
    assert "co-occurrence" in joined


def test_the_summary_says_a_frozen_plan_is_never_edited_after_seeing_how_it_went(spec):
    joined = " ".join(preregistration_summary(spec)["sentences"])
    assert "editable copy" in joined
    assert "quietly narrowed" in joined


def test_the_summary_changes_when_the_plan_changes(spec):
    """A summary that did not track the plan would be a preregistration of the summary."""
    before = preregistration_summary(spec)
    after = preregistration_summary(spec.copy(update={"alpha": 0.01}))
    assert before["sentences"] != after["sentences"]
    assert before["manifest_sha256"] != after["manifest_sha256"]


# ---------------------------------------------------------------- export and import


def test_an_exported_manifest_round_trips_to_the_same_content_address(spec):
    envelope = export_envelope(spec)
    assert envelope["manifest_sha256"] == manifest_sha256(spec)
    assert manifest_sha256(import_envelope(envelope)) == manifest_sha256(spec)


def test_an_envelope_edited_in_a_text_editor_is_refused_rather_than_imported(spec):
    """A run's identity is the content address of its plan, so an envelope whose digest does not
    address its body would give the run an identity for something it would not execute."""
    envelope = dict(export_envelope(spec))
    envelope["manifest_sha256"] = "0" * 64
    with pytest.raises(EnvelopeIntegrityError):
        import_envelope(envelope)


def test_something_that_is_not_an_envelope_is_refused_by_shape(spec):
    with pytest.raises(InvalidParameterError):
        import_envelope({"study_id": "not an envelope"})


def test_the_exported_envelope_carries_a_plan_and_no_data(spec):
    envelope = export_envelope(spec)
    assert "no data" in envelope["claim_boundary"]
    assert set(envelope["manifest"]) == set(json.loads(spec.json()))


# ------------------------------------------------------------------------------- api


def test_the_path_route_serves_the_registered_steps(client):
    body = client.get("/api/v1/experiment-composer/path").json()
    assert [row["step_id"] for row in body["steps"]] == [s.step_id for s in ordered_steps()]


def test_the_composer_contract_no_longer_keeps_its_own_copy_of_the_workflow(client):
    """Two descriptions of an order of operations are two different experiments waiting."""
    body = client.get("/api/v1/experiment-composer").json()
    assert body["workflow"] == [step.step_id for step in ordered_steps()]
    assert body["path_route"] == "/api/v1/experiment-composer/path"


def test_the_state_route_answers_for_a_manifest_that_has_never_been_executed(client, spec):
    response = client.post("/api/v1/experiment-composer/path/state",
                           json=json.loads(spec.json()))
    assert response.status_code == 200
    body = response.json()
    assert body["manifest_sha256"] == manifest_sha256(spec)
    assert body["run_state"] is None
    assert body["next_action"]["step_id"] == "freeze_and_run"


def test_reading_where_a_plan_stands_never_starts_it(client, spec, tmp_path):
    """`RunStore.open` publishes a frozen manifest and creates the run directory.

    A read of the path must look for a run, never open one, or a researcher inspecting their
    draft would have frozen it by looking at it.
    """
    runs = tmp_path / "experiment_runs" / "runs"
    client.post("/api/v1/experiment-composer/path/state", json=json.loads(spec.json()))
    assert not runs.exists() or not any(runs.iterdir())


def test_an_opened_run_is_folded_into_the_path_at_the_manifest_s_own_address(client, spec):
    """The composer never asks for a run id: a run identity *is* the content address of the
    plan, and a composer that took an id could be pointed at a run executing a different one."""
    payload = json.loads(spec.json())
    opened = client.post("/api/v1/experiment-runs", json=payload)
    assert opened.status_code == 200, opened.text
    body = client.post("/api/v1/experiment-composer/path/state", json=payload).json()
    assert body["run_state"] == opened.json()["progress"]["state"]


def test_a_completed_run_moves_the_ladder_but_not_past_the_second_rung(client, spec):
    payload = json.loads(spec.json())
    run_id = client.post("/api/v1/experiment-runs", json=payload).json()["run_id"]
    executed = client.post("/api/v1/experiment-runs/%s/execute" % run_id,
                           json={"worker_suite": "fixture_dry_run"})
    assert executed.status_code == 200, executed.text
    assert executed.json()["state"] == "COMPLETE", executed.text
    body = client.post("/api/v1/experiment-composer/path/state", json=payload).json()
    ladder = {row["rung"]: row for row in body["ladder"]}
    assert ladder["acquired_material"]["reached"] is True
    assert ladder["experiment_run"]["reached"] is True
    assert ladder["finding"]["reached"] is False
    assert ladder["admitted_evidence"]["reached"] is False


def test_the_window_preset_route_refuses_an_anchor_that_is_not_an_instant(client):
    ok = client.get("/api/v1/experiment-composer/window-presets",
                    params={"anchor_utc": "2026-01-01T00:00:00Z"})
    assert ok.status_code == 200 and len(ok.json()["presets"]) == len(DURATION_PRESETS)
    bad = client.get("/api/v1/experiment-composer/window-presets",
                     params={"anchor_utc": "whenever"})
    assert bad.status_code == 422


def test_the_domain_menu_route_marks_the_selected_domains(client):
    body = client.get("/api/v1/experiment-composer/domain-menu",
                      params={"selected": "argo_float,reanalysis"}).json()
    assert {row["domain"] for row in body["domains"] if row["selected"]} == \
        {"argo_float", "reanalysis"}


def test_the_preregistration_route_returns_the_plan_in_sentences(client, spec):
    body = client.post("/api/v1/experiment-composer/preregistration-summary",
                       json=json.loads(spec.json())).json()
    assert body["manifest_sha256"] == manifest_sha256(spec)
    assert len(body["sentences"]) >= 8


def test_the_export_and_import_routes_round_trip_over_http(client, spec):
    envelope = client.post("/api/v1/experiment-composer/manifests/export",
                           json=json.loads(spec.json())).json()
    imported = client.post("/api/v1/experiment-composer/manifests/import", json=envelope)
    assert imported.status_code == 200
    assert imported.json()["manifest_sha256"] == manifest_sha256(spec)


def test_a_tampered_envelope_is_refused_over_http_with_both_digests(client, spec):
    envelope = client.post("/api/v1/experiment-composer/manifests/export",
                           json=json.loads(spec.json())).json()
    envelope["manifest_sha256"] = "0" * 64
    refused = client.post("/api/v1/experiment-composer/manifests/import", json=envelope)
    assert refused.status_code == 422
    assert manifest_sha256(spec) in refused.json()["detail"]


def test_the_state_payload_is_shaped_as_the_frontend_type_declares(client, spec):
    """The contract check that TG11.6 exists for: the browser reads these keys by name."""
    body = client.post("/api/v1/experiment-composer/path/state",
                       json=json.loads(spec.json())).json()
    assert set(body) >= {"schema", "manifest_sha256", "study_id", "steps", "satisfied",
                         "next_action", "ladder", "run_state", "claim_boundary"}
    assert set(body["steps"][0]) >= {"step_id", "ordinal", "title", "question", "settles",
                                     "controls", "action_label", "action_route",
                                     "claim_boundary", "status", "reason", "detail"}
    assert set(body["next_action"]) >= {"step_id", "label", "route", "status", "why", "blocked"}
    assert set(body["ladder"][0]) >= {"rung", "title", "is", "is_not", "gate", "reached",
                                      "why_not"}


def test_the_menu_and_preset_payloads_are_shaped_as_the_frontend_types_declare(client):
    menu = client.get("/api/v1/experiment-composer/domain-menu").json()
    assert set(menu) >= {"schema", "domains", "minimum_domains", "note", "claim_boundary"}
    assert set(menu["domains"][0]) >= {"domain", "adapter_id", "label", "licence", "breaks",
                                       "lag_policy", "precedence_admissible",
                                       "admissible_kernels", "admissible_nulls", "selected",
                                       "selectable", "unavailable_reason", "observation",
                                       "onboarding_cost"}
    presets = client.get("/api/v1/experiment-composer/window-presets",
                         params={"anchor_utc": "2026-01-01T00:00:00Z"}).json()
    assert set(presets) >= {"schema", "anchor_utc", "presets", "note", "claim_boundary"}
    assert set(presets["presets"][0]) >= {"preset", "days", "start_utc", "end_utc",
                                          "stride_seconds", "label"}
