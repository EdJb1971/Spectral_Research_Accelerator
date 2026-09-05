"""TG17.12: a calibration measured elsewhere reaches a release gate, or the gate stays shut.

The gate this feeds used to read `NOT_RUN` while the calibration it names ran and passed on every
test pass. These guards exist so the opposite failure cannot replace it: a gate that reads `PASS`
because something once passed, against a contract or a statistic that has since moved.

TG17.15 slice 5 adds a second recording and the supersession that carries it into the same gate.
The guards below it are aimed at one failure the calendar half could not have: a supersession that
keeps agreeing with itself. The predecessor's claim is recomputed, so the interesting test is not
that it holds today but that the record says `VOID` on the day it stops.
"""

import json
import shutil
import time

import pytest

from src.core.calibration_record import (
    CALENDAR_ENTRY_POINT,
    CALENDAR_SOURCES,
    SCALE_SHAPE_ENTRY_POINT,
    SCALE_SHAPE_SEED,
    SCALE_SHAPE_SOURCES,
    calendar_applicability,
    calendar_contract,
    read_calendar_calibration,
    read_scale_shape_calibration,
    record_file,
    scale_shape_contract,
    scale_shape_record_file,
    scale_shape_source_digests,
    scale_shape_supersession,
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


# ========================= TG17.15 slice 5: a second recording, and a checked supersession


@pytest.fixture(scope="module")
def pool_checkout(tmp_path_factory):
    """A throwaway checkout carrying only what the pool-substitution recording is bound to."""
    root = tmp_path_factory.mktemp("pool_checkout")
    for name in SCALE_SHAPE_SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_file().parent.parent / name, target)
    (root / "calibration").mkdir()
    shutil.copy2(scale_shape_record_file(), root / "calibration" / "scale_shape_calibration.json")
    return root


def test_the_recorded_pool_calibration_in_this_checkout_met_its_declared_contract():
    """The fact the gate is entitled to read back, asserted against the recording itself."""
    facts = read_scale_shape_calibration()
    assert facts["status"] == "PASS"
    assert facts["reasons"] == []
    assert facts["executed_here"] is False
    recorded = facts["recorded"]
    assert recorded["all_met"] is True
    assert recorded["cases_failing_expectation"] == []
    assert recorded["cases_whose_run_cannot_certify_a_rate"] == []
    for row in recorded["cases"]:
        assert row["within_expectation"] and row["certifies_rate"]


def test_every_recorded_error_rate_clears_alpha_on_its_bound_and_not_on_its_point():
    """The distinction slice 4 was built on, checked where a gate would read it.

    A point estimate of zero is not a measured rate of zero, and a recording that carried only
    point estimates would let a run of twenty realisations look like a run of two hundred.
    """
    from src.benchmarks.pool_calibration import ALPHA, CASE_EXPECTATIONS

    recorded = read_scale_shape_calibration()["recorded"]
    scoring = [row for row in recorded["cases"]
               if CASE_EXPECTATIONS[row["case"]]["outcome"] == "does_not_reject"]
    assert scoring, "the true null and both adversarial nulls must be recorded"
    for row in scoring:
        bound = row["family_wise_error"]["one_sided_upper"]
        assert bound <= ALPHA
        assert bound > row["family_wise_error"]["point"], (
            "an upper bound that is not above the point estimate is not a bound")


def test_the_recorded_witness_reproduces_from_the_seed_it_names():
    """The backstop that makes a tens-of-minutes recording checkable in seconds.

    The calendar calibration is cheap enough to re-run whole on every suite pass, and is. This one
    is not, so the recording carries a per-case digest of its leading realisations and the seed
    that addresses them. Recomputing those realisations is the smaller claim honestly available
    here: not that the whole recorded run would come back, but that the code path it names still
    produces, realisation for realisation, what it produced then.
    """
    from src.benchmarks.pool_calibration import calibrate_case

    recorded = read_scale_shape_calibration()["recorded"]
    assert recorded["seed"] == SCALE_SHAPE_SEED
    for row in recorded["cases"]:
        witness = row["reproduction_witness"]
        again = calibrate_case(row["case"], realisations=witness["realisations"],
                               seed=witness["seed"])
        assert again.witness == witness["sha256"], (
            "%s no longer reproduces the leading realisations it was recorded from" % row["case"])


def test_the_recorded_ladder_ran_every_rung_against_one_inventory():
    """`held_fixed` is the ladder's whole claim, and the recording keeps it checkable.

    A detection curve whose rungs saw different inventories measures effect size confounded with
    whatever the redraw changed. The recording stores the digests as a set, so a reader sees how
    many there were rather than a boolean somebody computed for them.
    """
    detection = read_scale_shape_calibration()["recorded"]["detection"]
    assert len(detection["inventory_sha256"]) == 1
    assert len(detection["rungs"]) >= 8
    assert detection["rungs"][0]["weight"] > detection["rungs"][-1]["weight"]


def test_a_checkout_with_no_pool_recording_reads_not_run(tmp_path):
    facts = read_scale_shape_calibration(tmp_path)
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is None
    assert "no pool-substitution calibration has been recorded" in facts["reasons"][0]


def test_a_pool_recording_beside_the_source_it_was_made_against_still_reads_pass(pool_checkout):
    assert read_scale_shape_calibration(pool_checkout)["status"] == "PASS"


def test_changing_the_null_the_calibration_ranks_against_returns_the_gate_to_not_run(
        pool_checkout):
    """The source binding covers the null, not only the benchmark that drives it.

    A recording bound to the benchmark alone would survive a change to the thing being measured,
    which is the failure the calendar half of this module already names.
    """
    moved = pool_checkout / "src/core/pool_substitution_null.py"
    original = moved.read_bytes()
    try:
        moved.write_bytes(original + b"\n# a change to what the calibration measures\n")
        facts = read_scale_shape_calibration(pool_checkout)
    finally:
        moved.write_bytes(original)
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is not None, "the recording is still readable; it is only unbound"
    assert "src/core/pool_substitution_null.py" in " ".join(facts["reasons"])
    assert read_scale_shape_calibration(pool_checkout)["status"] == "PASS"


def test_relaxing_a_declared_expectation_returns_the_pool_gate_to_not_run(pool_checkout,
                                                                         monkeypatch):
    """Slice 4's expectations were the defect once already, and were replaced deliberately.

    Whatever replaces one next must not be able to inherit a recording made against the stricter
    declaration, so the expectations are inside the contract digest rather than beside it.
    """
    import src.benchmarks.pool_calibration as calibration_module

    relaxed = {case: ({**row, "maximum_family_wise_error": 0.5}
                      if row["outcome"] == "does_not_reject" else row)
               for case, row in calibration_module.CASE_EXPECTATIONS.items()}
    monkeypatch.setattr(calibration_module, "CASE_EXPECTATIONS", relaxed)
    facts = read_scale_shape_calibration(pool_checkout)
    assert facts["status"] == "NOT_RUN"
    assert any("contract has changed" in reason for reason in facts["reasons"])


def test_a_pool_recording_whose_case_missed_its_expectation_reads_fail_not_not_run(tmp_path):
    """A calibration that ran and failed is a different fact from one that did not run.

    Slice 4's first recorded run *did* fail, so this is not a hypothetical branch.
    """
    for name in SCALE_SHAPE_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_file().parent.parent / name, target)
    recorded = json.loads(scale_shape_record_file().read_text(encoding="utf-8"))
    recorded["all_met"] = False
    recorded["cases_failing_expectation"] = ["no_correspondence"]
    for row in recorded["cases"]:
        if row["case"] == "no_correspondence":
            row["within_expectation"] = False
    recorded["source_sha256"] = scale_shape_source_digests(tmp_path)
    recorded["contract_sha256"] = scale_shape_contract()["contract_sha256"]
    path = tmp_path / "calibration" / "scale_shape_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recorded), encoding="utf-8")

    facts = read_scale_shape_calibration(tmp_path)
    assert facts["status"] == "FAIL"
    assert "no_correspondence" in " ".join(facts["reasons"])


def test_a_run_too_small_to_bound_its_own_rate_reads_fail_rather_than_pass(tmp_path):
    """`certifies_rate` is separate from `within_expectation` in the benchmark, and stays separate
    here. A case that met its expectation on too few realisations has not shown what the gate
    would be quoting it for, and a recording that collapsed the two would hide exactly that."""
    for name in SCALE_SHAPE_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_file().parent.parent / name, target)
    recorded = json.loads(scale_shape_record_file().read_text(encoding="utf-8"))
    for row in recorded["cases"]:
        if row["case"] == "no_correspondence":
            row["certifies_rate"] = False
    recorded["source_sha256"] = scale_shape_source_digests(tmp_path)
    recorded["contract_sha256"] = scale_shape_contract()["contract_sha256"]
    path = tmp_path / "calibration" / "scale_shape_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recorded), encoding="utf-8")

    facts = read_scale_shape_calibration(tmp_path)
    assert facts["status"] == "FAIL"
    assert "too few times to bound its own error rate" in " ".join(facts["reasons"])


def test_the_supersession_holds_only_while_the_predecessor_s_claim_is_recomputed_true():
    """The state this checkout is in, and the two numbers it turns on."""
    supersession = scale_shape_supersession()
    assert supersession["status"] == "SUPERSEDED"
    superseded = supersession["superseded"]
    assert superseded["still_true"] is True
    assert superseded["recomputed"]["largest_drawable_inventory"] < \
        superseded["recomputed"]["minimum_resolvable_family"]
    assert supersession["superseding"]["entry_point"] == SCALE_SHAPE_ENTRY_POINT
    assert supersession["superseding"]["status"] == "PASS"


def test_a_supersession_whose_predecessor_stopped_being_true_reads_void_not_superseded(
        monkeypatch):
    """The outcome that makes "checked" mean anything, and the one no run today produces.

    If someone lifts the enumeration cap past the resolvable size, TG17.11's refusal has been
    retired *on its own terms* - it is simply no longer true - and calling that a supersession
    would credit this phase with removing a limit that removed itself. The record must say the
    supersession is describing a world that no longer holds, and must be re-derived.
    """
    import src.core.calibration_record as record_module
    import src.core.structural_nulls as nulls_module

    monkeypatch.setattr(nulls_module, "MAX_REASSIGNABLE_PAIRINGS", 10000)
    supersession = record_module.scale_shape_supersession()
    assert supersession["status"] == "VOID"
    assert supersession["superseded"]["still_true"] is False
    assert "no longer true on its own terms" in " ".join(supersession["reasons"])
    assert "re-derived" in " ".join(supersession["reasons"])


def test_a_supersession_without_a_passing_successor_leaves_the_old_refusal_standing(tmp_path):
    """The state this repository was in before this slice, reachable and named rather than lost."""
    supersession = scale_shape_supersession(tmp_path)
    assert supersession["status"] == "NOT_SUPERSEDED"
    assert supersession["superseded"]["still_true"] is True
    assert "no pool-substitution calibration has been recorded" in " ".join(
        supersession["reasons"])


def test_the_supersession_states_what_it_does_not_replace():
    """The half a supersession is most likely to leave out, and the half that keeps it honest."""
    supersession = scale_shape_supersession()
    kept = supersession["does_not_replace"]
    assert "joint_structure" in kept
    assert "exchangeability" in kept
    assert "real records" in kept
    assert "applicability" in supersession["replaces"]
    assert supersession["superseded"]["estimand"] != supersession["superseding"]["estimand"]


def test_the_two_estimands_need_different_inventories_and_the_record_says_by_how_much():
    """The reason the rebuild was worth doing, as the two numbers rather than as an adjective."""
    supersession = scale_shape_supersession()
    family = supersession["superseded"]["recomputed"]["minimum_resolvable_family"]
    pool = supersession["superseding"]["minimum_pool_size"]
    assert pool < family
    recorded = read_scale_shape_calibration()["recorded"]
    smallest = min(row["pool_size_range"][0] for row in recorded["cases"]
                   if row["pool_size_range"][1])
    assert smallest >= pool, (
        "the calibration must actually have reached the pool size its estimand requires")


def test_the_trimming_that_builds_a_recording_carries_the_witness_it_will_be_checked_by():
    """Mutation testing found this gap, and it is the kind a slow measurement hides.

    Every guard above reads the recording already on disk, so a break in the code that *writes* one
    would pass every test and surface only after the next fifty-minute re-record - by which point
    the recording it produced would be a receipt with an unusable witness. `_trim_case` is
    therefore exercised directly, on a one-realisation outcome that costs a second.
    """
    from src.benchmarks.pool_calibration import calibrate_case
    from src.core.calibration_record import _trim_case

    outcome = calibrate_case("no_correspondence", realisations=1, seed=SCALE_SHAPE_SEED)
    trimmed = _trim_case(outcome.describe())
    assert trimmed["reproduction_witness"]["sha256"] == outcome.witness
    assert trimmed["reproduction_witness"]["seed"] == SCALE_SHAPE_SEED
    assert trimmed["case"] == "no_correspondence"
    assert trimmed["family_wise_error"]["one_sided_upper"] > 0.0

    recorded = read_scale_shape_calibration()["recorded"]["cases"][0]
    assert set(trimmed) == set(recorded), (
        "what a recording is written with and what the gate reads back must be one shape")
