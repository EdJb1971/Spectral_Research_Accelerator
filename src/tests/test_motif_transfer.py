"""Phase G5, TG5.2: the published definition is bound before target access."""

import hashlib
import inspect
import json
import math

import pytest

from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.feature import FeatureLocation, Quantity, Significance, SpectralFeature
from src.core.motif import Scene, mine
from src.core.motif_freeze import FrozenMotif, FrozenMotifIntegrityError, freeze_motif
from src.core.motif_transfer import (
    TRANSFER_LEDGER_SCHEMA,
    TRANSFER_RECEIPT_SCHEMA,
    TargetAlreadyOpenedError,
    TransferLedger,
    blind_transfer,
)
from src.core.preregistration import PartitionIdentity


SOURCE_AXES = (
    AxisSpec("row", "space", "cells", ordinal=0),
    AxisSpec("col", "space", "cells", ordinal=1),
)
TARGET_AXES = (
    AxisSpec("northing", "space", "metres", ordinal=0),
    AxisSpec("easting", "space", "metres", ordinal=1),
)


def _feature(first, second, *, domain, axes, dataset, variable, units=None):
    return SpectralFeature(
        domain=domain, dataset=dataset, variable=variable,
        magnitude=Quantity(1.0, units),
        location=FeatureLocation({axes[0].name: first, axes[1].name: second}, axes),
        time=0.0, representation="identity", spatial_scale=Quantity(3.0, axes[0].units),
        significance=Significance(0.01, "surrogate_quantile"),
    )


def _triangle(centre, angle, scale, *, domain, axes, dataset, variable):
    features = []
    for radius, offset in zip((12.0, 19.0, 27.0), (0.0, 120.0, 240.0)):
        theta = math.radians(angle + offset)
        features.append(_feature(
            centre[0] + scale * radius * math.sin(theta),
            centre[1] + scale * radius * math.cos(theta),
            domain=domain, axes=axes, dataset=dataset, variable=variable))
    return tuple(features)


def _domain(name, axes, *, builder):
    return DomainDeclaration(
        name=name, description="Synthetic domain for %s" % builder,
        axes=(AxisSpec("time", "time", "hours"),) + axes,
        licence="repository synthetic fixture",
        violations=("no_propagation_speed", "no_natural_cycle"),
        lag_policy="none", provenance={"builder": builder, "version": 1})


@pytest.fixture(scope="module")
def frozen():
    scenes = tuple(Scene(
        "source%d" % index,
        _triangle((50 + index * 4, 70 + index * 5), index * 29, 1.0,
                  domain="synthetic_shapes", axes=SOURCE_AXES,
                  dataset="shape_camera_v1", variable="brightness"))
                   for index in range(4))
    result = mine(scenes, size=3, tolerance=1e-9, n_surrogates=199,
                  study_id="tg5.2-tests")
    train = PartitionIdentity(
        "source_train", 4, 1, ("brightness",), (0, 4),
        {"split": "train", "source_sha256": "a" * 64})
    return freeze_motif(
        result.ranked[0], result=result,
        origin_domain=_domain("synthetic_shapes", SOURCE_AXES, builder="source"),
        train=train, frozen_at="2026-08-26T08:00:00+12:00",
        study_id="tg5.2-tests")


def _target():
    return PartitionIdentity(
        "unopened_target", 4, 1, ("pressure",), (0, 4),
        {"role": "blind_transfer_target", "archive_key": "sealed-fixture-2"})


def _target_domain(name="synthetic_pressure"):
    return _domain(name, TARGET_AXES, builder="target")


def _matching_scenes(domain="synthetic_pressure"):
    return tuple(Scene(
        "target%d" % index,
        _triangle((500 + index * 60, 900 + index * 80), 17 + index * 41,
                  2.0 + index * 0.4, domain=domain, axes=TARGET_AXES,
                  dataset="pressure_array_v2", variable="pressure"))
                 for index in range(4))


def _run(frozen, tmp_path, **overrides):
    values = dict(
        published_sha256=frozen.motif_sha256, target=_target(),
        target_domain=_target_domain(), opener=_matching_scenes,
        ledger=TransferLedger(tmp_path / "transfer-ledger.json"),
        bound_at="2026-08-26T08:01:00+12:00",
        opened_at="2026-08-26T08:02:00+12:00",
    )
    values.update(overrides)
    return blind_transfer(frozen, **values)


def test_binding_is_durable_before_the_only_target_opener_runs(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    seen = []

    def opener():
        stored = json.loads(path.read_text(encoding="utf-8"))
        record = stored["records"][_target().digest()]
        seen.append((record["published_sha256"], record["state"]))
        return _matching_scenes()

    receipt = _run(frozen, tmp_path, opener=opener,
                   ledger=TransferLedger(path))
    assert seen == [(frozen.motif_sha256, "opening_committed")]
    assert receipt["schema"] == TRANSFER_RECEIPT_SCHEMA
    assert receipt["n_matches"] == 4
    assert receipt["support"] == 4


def test_search_parameters_exist_only_in_the_frozen_definition(frozen, tmp_path):
    parameters = inspect.signature(blind_transfer).parameters
    assert not {"size", "matcher", "tolerance", "relations"} & set(parameters)
    receipt = _run(frozen, tmp_path)
    assert receipt["search"] == {
        "size": frozen.definition["size"],
        "matcher": frozen.definition["matcher"],
        "tolerance": frozen.definition["tolerance"],
        "relations": list(frozen.graph.relations),
        "source": "frozen_motif",
    }


def test_matches_keep_both_domains_semantically_visible(frozen, tmp_path):
    receipt = _run(frozen, tmp_path)
    comparison = receipt["matches"][0]["comparison"]
    assert {node["domain"] for node in comparison["left_carried"]} == {
        "synthetic_shapes"}
    assert {node["domain"] for node in comparison["right_carried"]} == {
        "synthetic_pressure"}
    assert receipt["source_domain"] != receipt["target_domain"]["name"]


def test_a_wrong_published_digest_cannot_reach_the_target_callback(frozen, tmp_path):
    called = []
    with pytest.raises(FrozenMotifIntegrityError, match="published_sha256"):
        _run(frozen, tmp_path, published_sha256="0" * 64,
             opener=lambda: called.append(True))
    assert called == []
    assert not (tmp_path / "transfer-ledger.json").exists()


@pytest.mark.parametrize("bound_at,opened_at", [
    ("2026-08-26T07:59:00+12:00", "2026-08-26T08:02:00+12:00"),
    ("2026-08-26T08:02:00+12:00", "2026-08-26T08:01:00+12:00"),
    ("2026-08-26T08:01:00", "2026-08-26T08:02:00+12:00"),
])
def test_invalid_chronology_cannot_reach_the_target_callback(
        frozen, tmp_path, bound_at, opened_at):
    called = []
    with pytest.raises(InvalidParameterError, match="chronology|UTC offset"):
        _run(frozen, tmp_path, bound_at=bound_at, opened_at=opened_at,
             opener=lambda: called.append(True))
    assert called == []


def test_another_partition_of_the_source_domain_is_not_transfer(frozen, tmp_path):
    called = []
    with pytest.raises(InvalidParameterError, match="replication, not cross-domain"):
        _run(frozen, tmp_path,
             target_domain=_target_domain(name="synthetic_shapes"),
             opener=lambda: called.append(True))
    assert called == []


def test_a_matcher_that_did_not_declare_cross_domain_use_is_refused(frozen, tmp_path):
    definition = dict(frozen.definition)
    definition["matcher"] = "absolute_position"
    local = FrozenMotif(
        study_id=frozen.study_id, frozen_at=frozen.frozen_at,
        source=frozen.source, source_partition=frozen.source_partition,
        source_partition_sha256=frozen.source_partition_sha256,
        origin_domain=frozen.origin_domain, origin_nodes=frozen.origin_nodes,
        definition=definition)
    called = []
    with pytest.raises(InvalidParameterError, match="crosses_domains=True"):
        _run(local, tmp_path, published_sha256=local.motif_sha256,
             opener=lambda: called.append(True))
    assert called == []


def test_one_target_cannot_be_reopened_under_any_motif(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    _run(frozen, tmp_path, ledger=TransferLedger(path))
    with pytest.raises(TargetAlreadyOpenedError, match="already been opened"):
        _run(frozen, tmp_path, ledger=TransferLedger(path),
             bound_at="2026-08-26T08:03:00+12:00",
             opened_at="2026-08-26T08:04:00+12:00")


def test_a_callback_failure_still_spends_the_target(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"

    def broken():
        raise RuntimeError("archive read failed after access")

    with pytest.raises(RuntimeError, match="archive read failed"):
        _run(frozen, tmp_path, ledger=TransferLedger(path), opener=broken)
    with pytest.raises(TargetAlreadyOpenedError):
        _run(frozen, tmp_path, ledger=TransferLedger(path),
             bound_at="2026-08-26T08:03:00+12:00",
             opened_at="2026-08-26T08:04:00+12:00")


def test_mislabelled_target_values_are_refused_after_being_spent(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    with pytest.raises(InvalidParameterError, match="declared target domain"):
        _run(frozen, tmp_path, ledger=TransferLedger(path),
             opener=lambda: _matching_scenes(domain="an_unbound_domain"))
    assert _target().digest() in TransferLedger(path).records


def test_a_post_open_subset_of_target_scenes_is_refused_and_spent(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    with pytest.raises(InvalidParameterError, match="post-open subset"):
        _run(frozen, tmp_path, ledger=TransferLedger(path),
             opener=lambda: _matching_scenes()[:2])
    assert _target().digest() in TransferLedger(path).records


def test_a_null_match_search_is_a_complete_descriptive_receipt(frozen, tmp_path):
    def different():
        scenes = []
        for index in range(4):
            points = tuple(_feature(
                100.0 * index + first, 200.0 + second,
                domain="synthetic_pressure", axes=TARGET_AXES,
                dataset="pressure_array_v2", variable="pressure")
                for first, second in ((0.0, 0.0), (1.0, 0.0), (30.0, 2.0)))
            scenes.append(Scene("different%d" % index, points))
        return tuple(scenes)

    receipt = _run(frozen, tmp_path, opener=different)
    assert receipt["n_examined"] == 4
    assert receipt["n_matches"] == 0
    assert receipt["support"] == 0
    assert receipt["matches"] == []


def test_every_target_configuration_is_examined_without_post_open_selection(frozen, tmp_path):
    scenes = tuple(Scene(
        scene.name, scene.features + (_feature(
            9000 + index, 12000 + index, domain="synthetic_pressure", axes=TARGET_AXES,
            dataset="pressure_array_v2", variable="pressure"),))
        for index, scene in enumerate(_matching_scenes()))
    receipt = _run(frozen, tmp_path, opener=lambda: scenes)
    assert receipt["n_examined"] == 4 * math.comb(4, 3)


def test_ledger_is_canonical_reloadable_and_detects_record_tampering(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    receipt = _run(frozen, tmp_path, ledger=TransferLedger(path))
    reloaded = TransferLedger(path)
    assert reloaded.records[_target().digest()]["record_sha256"] == (
        receipt["chronology"]["record_sha256"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["records"][_target().digest()]["opened_at"] = "2026-08-26T09:00:00+12:00"
    path.write_text(json.dumps(stored, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="recomputed transfer-opening"):
        TransferLedger(path)


def test_unknown_ledger_meaning_is_refused(frozen, tmp_path):
    path = tmp_path / "transfer-ledger.json"
    _run(frozen, tmp_path, ledger=TransferLedger(path))
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["records"][_target().digest()]["can_redefine"] = False
    path.write_text(json.dumps(stored, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="unrecognised meaning"):
        TransferLedger(path)


def test_receipt_and_bound_target_declaration_are_content_addressed(frozen, tmp_path):
    receipt = _run(frozen, tmp_path)
    body = dict(receipt)
    recorded = body.pop("receipt_sha256")
    assert recorded == hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    chronology = receipt["chronology"]
    assert chronology["target_domain"] == _target_domain().describe()
    assert chronology["published_sha256"] == frozen.motif_sha256
    assert TRANSFER_LEDGER_SCHEMA in (
        tmp_path / "transfer-ledger.json").read_text(encoding="utf-8")
