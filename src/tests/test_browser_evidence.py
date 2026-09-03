"""TG18.5 slice 4: a rendered run reaches the release gate, or the gate stays shut.

The two gates this feeds are the ones TG17.10 refused to award itself. These guards exist so the
opposite failure cannot replace that refusal: a gate reading `PASS` because something once ran, on
a suite that has since changed, or on a run that only covered one spec of fifteen.
"""

import json
import shutil

import pytest

from src.core.browser_evidence import (
    BROWSER_RUN,
    DECLARED_PLAN_ACTIONS,
    MEASUREMENT_DIR,
    REPO_ROOT,
    SCIENTIST_ACTIONS,
    SPEC_DIR,
    browser_run_evidence,
    scientist_action_evidence,
    spec_digests,
)
from src.core.experiment_qualification import qualification_plan


def _checkout(root, *, specs=True, run=True, actions=True):
    """A throwaway checkout carrying only what the evidence is bound to."""
    if specs:
        (root / SPEC_DIR).mkdir(parents=True, exist_ok=True)
        for path in (REPO_ROOT / SPEC_DIR).glob("*.spec.ts"):
            shutil.copy2(path, root / SPEC_DIR / path.name)
    (root / MEASUREMENT_DIR).mkdir(parents=True, exist_ok=True)
    for name, wanted in ((BROWSER_RUN, run), (SCIENTIST_ACTIONS, actions)):
        source = REPO_ROOT / MEASUREMENT_DIR / name
        if wanted and source.is_file():
            shutil.copy2(source, root / MEASUREMENT_DIR / name)
    return root


def _run_record(root):
    return json.loads((root / MEASUREMENT_DIR / BROWSER_RUN).read_text(encoding="utf-8"))


def _write_run(root, record):
    (root / MEASUREMENT_DIR / BROWSER_RUN).write_text(
        json.dumps(record, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def checkout(tmp_path_factory):
    return _checkout(tmp_path_factory.mktemp("checkout"))


# --------------------------------------------------------------- the browser run evidence


def test_a_checkout_with_no_recorded_run_reads_not_run(tmp_path):
    facts = browser_run_evidence(_checkout(tmp_path, run=False, actions=False))
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is None
    assert BROWSER_RUN in facts["reasons"][0]


def test_a_complete_recorded_run_beside_the_specs_it_ran_reads_pass(checkout):
    """The control for the refusals below: the channel does work when nothing has moved."""
    facts = browser_run_evidence(checkout)
    assert facts["status"] == "PASS", facts["reasons"]
    assert facts["reasons"] == []
    assert facts["executed_here"] is False
    assert facts["recorded"]["failed"] == 0
    assert facts["recorded"]["passed"] > 0


def test_a_partial_run_cannot_qualify_the_gate_however_green_it_is(checkout, tmp_path):
    """The failure this channel most needed to prevent.

    Running one spec is the normal way to work on a test, and it writes a recording that says
    `passed`. A gate that read only the verdict would be awarded by it.
    """
    root = _checkout(tmp_path)
    record = _run_record(root)
    record["specs_that_ran"] = record["specs_that_ran"][:1]
    record["totals"] = {"passed": 4, "failed": 0, "skipped": 0}
    _write_run(root, record)

    facts = browser_run_evidence(root)
    assert facts["status"] == "NOT_RUN"
    assert "partial run" in " ".join(facts["reasons"])
    assert browser_run_evidence(checkout)["status"] == "PASS"


def test_weakening_a_spec_returns_the_gate_to_not_run_and_names_the_file(tmp_path):
    """A recording is a measurement of a particular set of assertions, not a permanent pass."""
    root = _checkout(tmp_path)
    spec = root / SPEC_DIR / "product-modes.spec.ts"
    spec.write_bytes(spec.read_bytes() + b"\n// a change to what this spec asserts\n")

    facts = browser_run_evidence(root)
    assert facts["status"] == "NOT_RUN"
    assert "product-modes.spec.ts" in " ".join(facts["reasons"])
    assert facts["recorded"] is not None, "the recording is readable; it is only unbound"


def test_a_run_that_happened_and_failed_reads_fail_rather_than_not_run(tmp_path):
    """The distinction this apparatus keeps everywhere, applied to a browser."""
    root = _checkout(tmp_path)
    record = _run_record(root)
    record["status"] = "failed"
    record["totals"] = {**record["totals"], "passed": record["totals"]["passed"] - 1,
                        "failed": 1}
    _write_run(root, record)

    facts = browser_run_evidence(root)
    assert facts["status"] == "FAIL"
    assert "1 failed" in " ".join(facts["reasons"])


def test_a_spec_recorded_but_since_deleted_is_named_rather_than_ignored(tmp_path):
    root = _checkout(tmp_path)
    (root / SPEC_DIR / "product-modes.spec.ts").unlink()

    facts = browser_run_evidence(root)
    assert facts["status"] == "NOT_RUN"
    assert "product-modes.spec.ts" in " ".join(facts["reasons"])


def test_the_evidence_states_what_a_browser_run_is_not(checkout):
    boundary = browser_run_evidence(checkout)["claim_boundary"]
    assert "not evidence that any workspace computes anything correctly" in boundary


# ------------------------------------------------------------- the scientist-action counts


def test_the_action_counts_are_measured_and_match_what_this_module_froze(checkout):
    facts = scientist_action_evidence(checkout)
    assert facts["status"] == "MEASURED"
    assert facts["actions_to_a_completed_run"] == DECLARED_PLAN_ACTIONS
    assert facts["reasons"] == []


def test_a_drifted_count_reads_unmeasured_rather_than_reporting_the_new_number(tmp_path):
    """A gate that reports whatever number it is handed is not a gate.

    A path that grew two actions is a change to how much a researcher must do, and it has to be
    re-measured and re-frozen deliberately rather than absorbed.
    """
    root = _checkout(tmp_path)
    path = root / MEASUREMENT_DIR / SCIENTIST_ACTIONS
    record = json.loads(path.read_text(encoding="utf-8"))
    record["declared_plan_to_completed_run"]["actions"] = DECLARED_PLAN_ACTIONS + 2
    path.write_text(json.dumps(record), encoding="utf-8")

    facts = scientist_action_evidence(root)
    assert facts["status"] == "NOT_MEASURED"
    assert "does not match the count frozen" in " ".join(facts["reasons"])
    assert facts["measured"]["actions_to_a_completed_run"] == DECLARED_PLAN_ACTIONS + 2


def test_an_absent_measurement_reads_unmeasured_with_every_field_intact(tmp_path):
    facts = scientist_action_evidence(_checkout(tmp_path, run=False, actions=False))
    assert facts["status"] == "NOT_MEASURED"
    assert facts["adapter_specific_framework_edits"] == "NOT_MEASURED"
    assert facts["refusal_explanation_time_seconds"] == "NOT_MEASURED"


def test_the_source_edit_audit_stays_unmeasured_even_when_the_browser_ran(checkout):
    """No rendered run can observe a source-edit audit, so a rendered run may not award it."""
    assert scientist_action_evidence(checkout)["adapter_specific_framework_edits"] \
        == "NOT_MEASURED"


def test_wall_clock_reaches_the_record_only_under_a_name_that_says_it_is_unasserted(checkout):
    facts = scientist_action_evidence(checkout)
    timed = [key for key in facts if "seconds" in key]
    assert timed, "the durations must be carried, not dropped"
    for key in timed:
        assert key.endswith("_unasserted"), key


# ------------------------------------------------------------------------- the gate itself


def test_the_gate_reads_the_recording_and_does_not_open_a_browser():
    import time

    started = time.perf_counter()
    plan = qualification_plan()
    elapsed = time.perf_counter() - started

    gate = {row["gate_id"]: row for row in plan["gates"]}["browser_no_glue"]
    assert gate["status"] in {"PASS", "FAIL", "NOT_RUN"}
    assert plan["browser_evidence"]["executed_here"] is False
    assert elapsed < 1.0, "the plan assembled in %.3fs; it must not be running a browser" % elapsed


def test_the_qualification_record_carries_the_counts_or_says_it_has_none():
    actions = qualification_plan()["scientist_actions"]
    assert actions["status"] in {"MEASURED", "NOT_MEASURED"}
    if actions["status"] == "NOT_MEASURED":
        assert actions["refusal_explanation_time_seconds"] == "NOT_MEASURED"
    else:
        assert actions["actions_to_a_completed_run"] == DECLARED_PLAN_ACTIONS


def test_no_gate_passing_makes_the_apparatus_releaseable_on_its_own():
    plan = qualification_plan()
    assert plan["verdict"] == "NOT_RELEASEABLE"
    assert any(row["blocking"] for row in plan["gates"])


def test_every_spec_in_the_suite_is_bound_not_merely_the_ones_a_run_chose():
    """The inventory is read from disk, so adding a spec cannot leave it silently unbound."""
    digests = spec_digests()
    names = {path.name for path in (REPO_ROOT / SPEC_DIR).glob("*.spec.ts")}
    assert set(digests) == names
    assert len(names) >= 15
