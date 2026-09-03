"""TG17.12: a calibration measured elsewhere reaches a release gate, or the gate stays shut.

The gate this feeds used to read `NOT_RUN` while the calibration it names ran and passed on every
test pass. These guards exist so the opposite failure cannot replace it: a gate that reads `PASS`
because something once passed, against a contract or a statistic that has since moved.
"""

import json
import shutil
import time

import pytest

from src.core.calibration_record import (
    CALENDAR_ENTRY_POINT,
    CALENDAR_SOURCES,
    calendar_applicability,
    calendar_contract,
    read_calendar_calibration,
    record_file,
    source_digests,
)
from src.core.experiment_qualification import qualification_plan


@pytest.fixture(scope="module")
def checkout(tmp_path_factory):
    """A throwaway checkout carrying only what a recording is bound to, so it can be moved."""
    root = tmp_path_factory.mktemp("checkout")
    for name in CALENDAR_SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_file().parent.parent / name, target)
    (root / "calibration").mkdir()
    shutil.copy2(record_file(), root / "calibration" / "calendar_calibration.json")
    return root


def test_the_recorded_calendar_calibration_in_this_checkout_meets_every_frozen_answer():
    """The fact the gate is entitled to state, asserted against the recording itself."""
    facts = read_calendar_calibration()
    assert facts["status"] == "PASS"
    assert facts["reasons"] == []
    recorded = facts["recorded"]
    assert recorded["all_met"] is True
    planted = next(row for row in recorded["cases"] if row["case"] == "shared_calendar_event")
    assert planted["n_rejected_after_correction"] == planted["minimum_rejections"] == 6
    for row in recorded["cases"]:
        assert row["minimum_rejections"] <= row["n_rejected_after_correction"] \
            <= row["maximum_rejections"]


def test_a_checkout_with_no_recording_reads_not_run_rather_than_inheriting_this_one(tmp_path):
    facts = read_calendar_calibration(tmp_path)
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is None
    assert "no calibration has been recorded" in facts["reasons"][0]


def test_a_recording_copied_beside_the_source_it_was_made_against_still_reads_pass(checkout):
    """The control for the two staleness guards below: the channel does work when nothing moved."""
    assert read_calendar_calibration(checkout)["status"] == "PASS"


def test_changing_the_statistic_returns_the_gate_to_not_run_and_names_the_file(checkout):
    """A recording is a measurement of particular code, not a permanent certificate."""
    moved = checkout / "src/benchmarks/family_calibration.py"
    original = moved.read_bytes()
    try:
        moved.write_bytes(original + b"\n# a change to what the calibration measures\n")
        facts = read_calendar_calibration(checkout)
    finally:
        moved.write_bytes(original)
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is not None, "the recording is still readable; it is only unbound"
    assert "src/benchmarks/family_calibration.py" in " ".join(facts["reasons"])
    assert read_calendar_calibration(checkout)["status"] == "PASS"


def test_relaxing_a_frozen_expectation_returns_the_gate_to_not_run(checkout, monkeypatch):
    """`CALIBRATION_CASES` says it exists so a later slice cannot quietly relax an answer.

    Relaxing one here must not leave a recorded pass agreeing with the weaker declaration.
    """
    import src.benchmarks.family_calibration as calibration_module

    relaxed = tuple({**row, "minimum_rejections": 0} if row["case"] == "shared_calendar_event"
                    else row for row in calibration_module.CALIBRATION_CASES)
    monkeypatch.setattr(calibration_module, "CALIBRATION_CASES", relaxed)
    facts = read_calendar_calibration(checkout)
    assert facts["status"] == "NOT_RUN"
    assert any("contract has changed" in reason for reason in facts["reasons"])


def test_a_recording_whose_cases_did_not_meet_their_answers_reads_fail_not_not_run(tmp_path):
    """A calibration that ran and failed is a different fact from one that did not run."""
    for name in CALENDAR_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_file().parent.parent / name, target)
    recorded = json.loads(record_file().read_text(encoding="utf-8"))
    recorded["all_met"] = False
    for row in recorded["cases"]:
        if row["case"] == "shared_calendar_event":
            row["n_rejected_after_correction"] = 0
            row["met"] = False
    recorded["source_sha256"] = source_digests(tmp_path)
    recorded["contract_sha256"] = calendar_contract()["contract_sha256"]
    path = tmp_path / "calibration" / "calendar_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recorded), encoding="utf-8")

    facts = read_calendar_calibration(tmp_path)
    assert facts["status"] == "FAIL"
    assert "shared_calendar_event" in " ".join(facts["reasons"])


def test_a_source_digest_ignores_line_endings_so_two_checkouts_agree(tmp_path):
    """Otherwise every Windows clone would read its own recording as stale."""
    name = CALENDAR_SOURCES[0]
    body = (record_file().parent.parent / name).read_bytes().replace(b"\r\n", b"\n")
    for variant in (body, body.replace(b"\n", b"\r\n")):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(variant)
        assert source_digests(tmp_path)[name] == source_digests()[name]


def test_the_gate_reads_the_recording_and_does_not_run_the_calibration():
    """The separation the whole channel exists for: a release gate is not a scientific worker."""
    started = time.perf_counter()
    plan = qualification_plan()
    elapsed = time.perf_counter() - started
    gate = {row["gate_id"]: row for row in plan["gates"]}["calendar_calibration"]
    assert gate["status"] == "PASS"
    assert gate["blocking"] is False
    assert CALENDAR_ENTRY_POINT in gate["detail"]
    assert plan["calendar_calibration"]["executed_here"] is False
    assert elapsed < 1.0, "the gate assembled in %.3fs; it must not be running a calibration" % \
        elapsed


def test_one_passing_gate_does_not_release_the_apparatus():
    """A gate moving to PASS must move the verdict only when it is the last one."""
    plan = qualification_plan()
    assert plan["verdict"] == "NOT_RELEASEABLE"
    assert any(row["blocking"] for row in plan["gates"])


def test_the_calendar_null_resolves_at_both_the_calibration_and_the_declared_plan():
    """Asserted as the inequality the correction gives, not as the numbers it happens to give."""
    facts = calendar_applicability()
    for configuration in (facts["calibration_family"], facts["declared_plan"]):
        assert configuration["replications"] >= configuration["replications_required"]
        assert configuration["can_reject_after_correction"] is True
        assert configuration["p_value_floor"] == pytest.approx(
            1.0 / (1 + configuration["replications"]))


def test_the_two_modes_buy_their_floor_differently_and_only_one_of_them_can_reject():
    """The TG17.11/TG17.12 contrast, computed here rather than asserted in prose.

    Both nulls have a p-value floor. The calendar null buys it with replications and has no
    enumeration ceiling; the scale/shape null buys it with domains and has one, an order of
    magnitude below what its correction needs.
    """
    from src.core.experiment_qualification import scale_shape_applicability

    calendar = calendar_applicability()
    scale_shape = scale_shape_applicability()
    assert calendar["enumeration_ceiling"] is None
    assert calendar["floor_is_bought_with"] == "replications"
    assert calendar["declared_plan"]["can_reject_after_correction"] is True
    assert scale_shape["declared_null_can_ever_reject"] is False
    assert scale_shape["largest_drawable_inventory"] < scale_shape["minimum_resolvable_family"]


def test_the_gate_states_the_recording_and_claims_nothing_about_acquired_data():
    facts = read_calendar_calibration()
    boundary = facts["applicability"]["claim_boundary"]
    assert "not power" in boundary
    assert facts["recorded"]["claim_boundary"].startswith("A frozen-family calibration")
    assert "not a licence to mine" in facts["recorded"]["claim_boundary"]
