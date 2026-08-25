"""Phase G5, TG5.3: corrected motif-to-organisation transfer on train and test."""

import inspect
import json
import math
from dataclasses import replace

import pytest

from src.core.domain import AxisSpec, DomainDeclaration, PrecedenceNotAdmissibleError
from src.core.errors import InvalidParameterError
from src.core.feature import FeatureLocation, Quantity, Significance, SpectralFeature
from src.core.motif import Scene, mine
from src.core.motif_freeze import freeze_motif
from src.core.motif_relationship import (
    RELATIONSHIP_LEDGER_SCHEMA,
    RelationshipFrame,
    RelationshipLedger,
    RelationshipPlan,
    load_relationship_plan,
    plan_relationship_transfer,
    save_relationship_plan,
    test_relationship_transfer as run_relationship_transfer,
)
from src.core.preregistration import PartitionIdentity


SOURCE_AXES = (AxisSpec("row", "space", "cells", ordinal=0),
               AxisSpec("col", "space", "cells", ordinal=1))
TARGET_AXES = (AxisSpec("northing", "space", "metres", ordinal=0),
               AxisSpec("easting", "space", "metres", ordinal=1))


def _feature(first, second, *, domain, axes, dataset, variable):
    return SpectralFeature(
        domain=domain, dataset=dataset, variable=variable, magnitude=Quantity(1.0, None),
        location=FeatureLocation({axes[0].name: first, axes[1].name: second}, axes),
        time=0.0, representation="identity", spatial_scale=Quantity(3.0, axes[0].units),
        significance=Significance(0.01, "surrogate_quantile"))


def _triangle(centre, angle, scale, *, domain, axes, dataset, variable):
    return tuple(_feature(
        centre[0] + scale * radius * math.sin(math.radians(angle + offset)),
        centre[1] + scale * radius * math.cos(math.radians(angle + offset)),
        domain=domain, axes=axes, dataset=dataset, variable=variable)
        for radius, offset in zip((12.0, 19.0, 27.0), (0.0, 120.0, 240.0)))


def _different(index):
    return tuple(_feature(500 + index * 2 + step * step * 7, 800 + step * 23,
                          domain="synthetic_pressure", axes=TARGET_AXES,
                          dataset="pressure_v1", variable="pressure")
                 for step in range(3))


def _domain(name, axes, *, target=False):
    values = dict(
        name=name, description="TG5.3 synthetic domain",
        axes=(AxisSpec("time", "time", "hours"),) + axes,
        licence="repository synthetic fixture", violations=("no_natural_cycle",),
        provenance={"builder": "test_motif_relationship", "version": 1})
    if target:
        values.update(lag_policy="declared", declared_floor_frames=1,
                      declared_floor_basis="one-frame synthetic response floor")
    else:
        values.update(lag_policy="none", violations=("no_propagation_speed", "no_natural_cycle"))
    return DomainDeclaration(**values)


@pytest.fixture(scope="module")
def frozen():
    scenes = tuple(Scene("source%d" % i, _triangle(
        (50 + i * 4, 70 + i * 5), i * 29, 1.0, domain="synthetic_shapes",
        axes=SOURCE_AXES, dataset="shape_v1", variable="brightness")) for i in range(4))
    result = mine(scenes, size=3, tolerance=1e-9, n_surrogates=199,
                  study_id="tg5.3-tests")
    source = PartitionIdentity("source_train", 4, 1, ("brightness",), (0, 4),
                               {"split": "train", "source_sha256": "a" * 64})
    return freeze_motif(
        result.ranked[0], result=result, origin_domain=_domain("synthetic_shapes", SOURCE_AXES),
        train=source, frozen_at="2026-08-26T08:00:00+12:00", study_id="tg5.3-tests")


def _partitions(n=60):
    return (
        PartitionIdentity("target_train", n, 1, ("organisation",), (0, n),
                          {"split": "train", "archive": "sealed-target-v1"}),
        PartitionIdentity("target_test", n, 1, ("organisation",), (n, 2 * n),
                          {"split": "test", "archive": "sealed-target-v1"}))


def _plan(frozen, **overrides):
    train, test = _partitions()
    values = dict(
        target_domain=_domain("synthetic_pressure", TARGET_AXES, target=True),
        train=train, test=test, outcomes=("organisation",), lags=(1,),
        n_surrogates=199, root_seed=731, planned_at="2026-08-26T08:01:00+12:00")
    values.update(overrides)
    return plan_relationship_transfer(frozen, **values)


def _frames(*, planted=True, all_absent=False, n=60):
    presence = [False if all_absent else i % 3 == 0 for i in range(n)]
    outcomes = [0.0] * n
    if planted:
        for i in range(n - 1):
            outcomes[i + 1] = 10.0 if presence[i] else 0.0
    else:
        outcomes = [float((i * 17) % 11) for i in range(n)]
    frames = []
    for i, present in enumerate(presence):
        features = (_triangle((500 + i * 3, 900 + i * 2), i * 37, 2.0,
                              domain="synthetic_pressure", axes=TARGET_AXES,
                              dataset="pressure_v1", variable="pressure")
                    if present else _different(i))
        frames.append(RelationshipFrame(Scene("target%d" % i, features),
                                        {"organisation": outcomes[i]}))
    return tuple(frames)


def _run(frozen, tmp_path, **overrides):
    plan = overrides.pop("plan", _plan(frozen))
    values = dict(
        published_motif_sha256=frozen.motif_sha256,
        published_plan_sha256=plan.plan_sha256,
        train_opener=lambda: _frames(), test_opener=lambda: _frames(),
        ledger=RelationshipLedger(tmp_path / "relationship-ledger.json"),
        bound_at="2026-08-26T08:02:00+12:00",
        train_opened_at="2026-08-26T08:03:00+12:00",
        test_opened_at="2026-08-26T08:04:00+12:00")
    values.update(overrides)
    return run_relationship_transfer(frozen, plan, **values)


def test_plan_binds_the_complete_corrected_family_and_both_splits(frozen):
    plan = _plan(frozen)
    assert plan.labels == ("organisation|lag=1",)
    assert plan.specification["notes"]["alternative"] == "greater"
    assert plan.train.provenance["split"] == "train"
    assert plan.test.provenance["split"] == "test"
    assert plan.n_surrogates == 199


def test_execution_has_no_relationship_redefinition_surface():
    names = set(inspect.signature(run_relationship_transfer).parameters)
    assert not names.intersection({"outcomes", "lags", "n_surrogates", "root_seed", "alpha",
                                   "correction", "matcher", "tolerance"})


def test_both_openings_are_durably_committed_before_train_is_seen(frozen, tmp_path):
    path = tmp_path / "relationship-ledger.json"
    seen = []

    def train_opener():
        value = json.loads(path.read_text(encoding="utf-8"))
        record = next(iter(value["records"].values()))
        seen.append(record["state"])
        return _frames()

    receipt = _run(frozen, tmp_path, train_opener=train_opener)
    assert seen == ["both_openings_committed"]
    assert receipt["chronology"]["published_plan_sha256"] == _plan(frozen).plan_sha256


def test_the_same_positive_relationship_must_survive_correction_twice(frozen, tmp_path):
    receipt = _run(frozen, tmp_path)
    assert receipt["verdict"] == "PASS"
    assert receipt["replicated_labels"] == ["organisation|lag=1"]
    for split in ("train", "test"):
        result = receipt[split]["results"][0]
        assert result["effect"] == 10.0
        assert result["rejected"] is True
        assert receipt[split]["family_size"] == receipt["correction_unit"] == 1


def test_an_adequately_powered_null_is_a_complete_fail(frozen, tmp_path):
    receipt = _run(frozen, tmp_path, train_opener=lambda: _frames(planted=False),
                   test_opener=lambda: _frames(planted=False))
    assert receipt["verdict"] == "FAIL"
    assert receipt["n_replicated"] == 0
    assert receipt["train"]["results"][0]["p_value"] > 0
    assert "adequately powered null" in receipt["claim_boundary"]


def test_no_target_motif_is_counted_inside_the_family_not_silently_dropped(frozen, tmp_path):
    receipt = _run(frozen, tmp_path, train_opener=lambda: _frames(all_absent=True),
                   test_opener=lambda: _frames(all_absent=True))
    result = receipt["test"]["results"][0]
    assert receipt["verdict"] == "FAIL"
    assert result["testable"] is False
    assert result["p_value"] == 1.0
    assert result["adjusted"] == 1.0


def test_wrong_published_plan_never_reaches_either_opener(frozen, tmp_path):
    opened = []
    with pytest.raises(InvalidParameterError, match="published_plan_sha256"):
        _run(frozen, tmp_path, published_plan_sha256="0" * 64,
             train_opener=lambda: opened.append("train"),
             test_opener=lambda: opened.append("test"))
    assert opened == []
    assert not (tmp_path / "relationship-ledger.json").exists()


def test_a_failed_train_load_still_spends_both_bound_partitions(frozen, tmp_path):
    path = tmp_path / "relationship-ledger.json"
    with pytest.raises(RuntimeError, match="train read failed"):
        _run(frozen, tmp_path, ledger=RelationshipLedger(path),
             train_opener=lambda: (_ for _ in ()).throw(RuntimeError("train read failed")))
    with pytest.raises(InvalidParameterError, match="already been opened"):
        _run(frozen, tmp_path, ledger=RelationshipLedger(path))


def test_post_open_subset_and_outcome_redefinition_are_refused_after_spending(frozen, tmp_path):
    path = tmp_path / "relationship-ledger.json"
    with pytest.raises(InvalidParameterError, match="distinct frames"):
        _run(frozen, tmp_path, ledger=RelationshipLedger(path),
             train_opener=lambda: _frames()[:-1])
    assert RelationshipLedger(path).records

    other = tmp_path / "other-ledger.json"
    changed = list(_frames())
    changed[0] = RelationshipFrame(changed[0].scene, {"chosen_after_open": 1.0})
    with pytest.raises(InvalidParameterError, match="every frozen finite outcome"):
        _run(frozen, tmp_path, ledger=RelationshipLedger(other), train_opener=lambda: changed)
    assert RelationshipLedger(other).records


def test_train_and_test_must_be_explicit_nonoverlapping_splits(frozen):
    train, test = _partitions()
    with pytest.raises(InvalidParameterError, match="non-overlapping"):
        _plan(frozen, test=replace(test, frames=(30, 90)))
    with pytest.raises(InvalidParameterError, match="explicit train and test"):
        _plan(frozen, test=replace(test, provenance={"split": "train"}))
    with pytest.raises(InvalidParameterError, match="shorter than both"):
        _plan(frozen, lags=(60,))


def test_target_must_license_precedence_and_its_lag_floor(frozen):
    no_floor = _domain("no_floor_target", TARGET_AXES)
    with pytest.raises(PrecedenceNotAdmissibleError):
        _plan(frozen, target_domain=no_floor)
    floor_two = DomainDeclaration(
        name="floor_two", description="synthetic", axes=(AxisSpec("time", "time", "hours"),) + TARGET_AXES,
        licence="synthetic", violations=("no_natural_cycle",), lag_policy="declared",
        declared_floor_frames=2, declared_floor_basis="fixture")
    with pytest.raises(InvalidParameterError, match="floor of 2"):
        _plan(frozen, target_domain=floor_two, lags=(1,))


def test_an_unaffordable_family_is_refused_before_a_plan_exists(frozen):
    with pytest.raises(InvalidParameterError, match="cannot report anything"):
        _plan(frozen, outcomes=("a", "b"), lags=(1, 2), n_surrogates=19)


def test_plan_round_trip_is_canonical_and_published_digest_checked(frozen, tmp_path):
    plan = _plan(frozen)
    path = tmp_path / "relationship-plan.json"
    assert save_relationship_plan(path, plan) == plan.plan_sha256
    loaded = load_relationship_plan(path, published_sha256=plan.plan_sha256)
    assert loaded == plan
    with pytest.raises(FileExistsError):
        save_relationship_plan(path, plan)


def test_plan_and_ledger_tampering_are_detected(frozen, tmp_path):
    plan = _plan(frozen)
    record = plan.to_mapping()
    record["lags"] = [2]
    with pytest.raises(InvalidParameterError, match="relationship specification"):
        RelationshipPlan.from_mapping(record)

    bypass = plan.to_mapping()
    bypass["target_domain"]["precedence_admissible"] = False
    body = {key: value for key, value in bypass.items() if key != "plan_sha256"}
    bypass["plan_sha256"] = __import__("hashlib").sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode()).hexdigest()
    with pytest.raises(InvalidParameterError, match="target_domain"):
        RelationshipPlan.from_mapping(bypass)

    _run(frozen, tmp_path)
    path = tmp_path / "relationship-ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["schema"] == RELATIONSHIP_LEDGER_SCHEMA
    next(iter(ledger["records"].values()))["test_sha256"] = "0" * 64
    path.write_text(json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="recomputed digest"):
        RelationshipLedger(path)


def test_a_partition_cannot_be_reused_with_a_different_counterpart(frozen, tmp_path):
    path = tmp_path / "relationship-ledger.json"
    first = _plan(frozen)
    _run(frozen, tmp_path, plan=first, ledger=RelationshipLedger(path))
    replacement_test = replace(first.test, name="second_test", frames=(120, 180),
                               provenance={"split": "test", "archive": "sealed-target-v1"})
    second = _plan(frozen, train=first.train, test=replacement_test)
    with pytest.raises(InvalidParameterError, match="never opened"):
        _run(frozen, tmp_path, plan=second, ledger=RelationshipLedger(path))


def test_receipt_is_content_addressed_and_keeps_the_claim_boundary(frozen, tmp_path):
    receipt = _run(frozen, tmp_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    import hashlib
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert receipt["receipt_sha256"] == expected
    assert "causality" in receipt["claim_boundary"]
