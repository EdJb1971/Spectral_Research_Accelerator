"""Phase G5, TG5.1: a motif survives the process that discovered it unchanged."""

import json
import math
from dataclasses import replace

import pytest

from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.feature import FeatureLocation, Quantity, Significance, SpectralFeature
from src.core.motif import MotifCandidate, Occurrence, Scene, mine
from src.core.motif_freeze import (
    FROZEN_MOTIF_SCHEMA,
    FrozenMotif,
    FrozenMotifIntegrityError,
    freeze_motif,
    load_frozen_motif,
    save_frozen_motif,
)
from src.core.preregistration import PartitionIdentity


SPATIAL_AXES = (
    AxisSpec("row", "space", "cells", ordinal=0),
    AxisSpec("col", "space", "cells", ordinal=1),
)


def _feature(row, col, *, domain="synthetic_origin"):
    return SpectralFeature(
        domain=domain, dataset="planted_shapes_v1", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({"row": row, "col": col}, SPATIAL_AXES),
        time=0.0, representation="identity", spatial_scale=Quantity(3.0, "cells"),
        significance=Significance(0.01, "surrogate_quantile"),
    )


def _triangle(row, col, angle):
    points = []
    for radius, offset in zip((12.0, 19.0, 27.0), (0.0, 120.0, 240.0)):
        theta = math.radians(angle + offset)
        points.append(_feature(row + radius * math.sin(theta),
                               col + radius * math.cos(theta)))
    return tuple(points)


@pytest.fixture(scope="module")
def mined():
    scenes = tuple(Scene("train%d" % i, _triangle(50 + i * 5, 70 + i * 4, i * 31))
                   for i in range(4))
    result = mine(scenes, size=3, tolerance=1e-9, n_surrogates=199,
                  study_id="tg5.1-tests")
    return result, result.ranked[0]


def _domain(**overrides):
    values = dict(
        name="synthetic_origin", description="Planted configurations for TG5.1",
        axes=(AxisSpec("time", "time", "hours"),) + SPATIAL_AXES,
        licence="repository synthetic fixture",
        violations=("no_propagation_speed", "no_natural_cycle"),
        lag_policy="none", provenance={"builder": "test_motif_freezing", "version": 1},
    )
    values.update(overrides)
    return DomainDeclaration(**values)


def _train(**provenance):
    record = {"split": "train", "source_sha256": "a" * 64}
    record.update(provenance)
    return PartitionIdentity("train", 4, 1, ("amplitude",), (0, 4), record)


def _frozen(mined, **overrides):
    result, candidate = mined
    values = dict(
        result=result, origin_domain=_domain(), train=_train(),
        frozen_at="2026-08-26T08:00:00+12:00", study_id="tg5.1-tests",
    )
    values.update(overrides)
    return freeze_motif(candidate, **values)


def test_the_structural_definition_and_origin_are_separate_and_both_hashed(mined):
    motif = _frozen(mined)
    assert motif.schema == FROZEN_MOTIF_SCHEMA
    assert motif.definition["matcher"] == "relative_geometry"
    assert motif.definition["size"] == 3
    assert "carried" not in motif.definition
    assert motif.origin_domain["name"] == "synthetic_origin"
    assert {node["dataset"] for node in motif.origin_nodes} == {"planted_shapes_v1"}
    assert motif.definition_sha256 != motif.motif_sha256
    assert motif.verify()["matches"] is True


def test_the_graph_round_trips_without_the_mining_process(mined, tmp_path):
    motif = _frozen(mined)
    path = tmp_path / "motif.json"
    assert save_frozen_motif(path, motif) == motif.motif_sha256
    loaded = load_frozen_motif(path, published_sha256=motif.motif_sha256)
    assert loaded.to_mapping() == motif.to_mapping()
    assert loaded.graph.describe() == motif.graph.describe()


def test_serialization_is_canonical_and_stable(mined, tmp_path):
    first = _frozen(mined)
    second = _frozen(mined)
    assert first.motif_sha256 == second.motif_sha256
    path = tmp_path / "motif.json"
    save_frozen_motif(path, first)
    assert path.read_bytes() == (
        json.dumps(first.to_mapping(), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def test_the_in_memory_definition_has_no_mutation_path(mined):
    motif = _frozen(mined)
    with pytest.raises(TypeError):
        motif.definition["matcher"] = "absolute_position"
    with pytest.raises(TypeError):
        motif.definition["attributes"][0]["extent_in_scales"] = 999
    with pytest.raises(TypeError):
        motif.origin_domain["provenance"]["version"] = 2


def test_publication_refuses_to_replace_even_identical_bytes(mined, tmp_path):
    path = tmp_path / "motif.json"
    motif = _frozen(mined)
    save_frozen_motif(path, motif)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_frozen_motif(path, motif)


def test_a_changed_structural_number_breaks_the_definition_hash(mined):
    record = _frozen(mined).to_mapping()
    edge = next(iter(record["definition"]["edges"].values()))
    relation = next(iter(edge.values()))
    relation["value"] += 0.1
    with pytest.raises(FrozenMotifIntegrityError, match="definition_sha256"):
        FrozenMotif.from_mapping(record)


def test_a_changed_origin_breaks_the_outer_hash_even_when_structure_is_unchanged(mined):
    record = _frozen(mined).to_mapping()
    record["origin_domain"]["licence"] = "a different licence"
    with pytest.raises(FrozenMotifIntegrityError, match="motif_sha256"):
        FrozenMotif.from_mapping(record)


def test_a_wholesale_honest_rewrite_is_caught_against_the_published_digest(mined):
    first = _frozen(mined)
    second = _frozen(mined, frozen_at="2026-08-26T08:01:00+12:00")
    assert second.verify()["matches"] is True
    with pytest.raises(FrozenMotifIntegrityError, match="published_sha256"):
        second.verify_published(first.motif_sha256)


def test_a_held_out_partition_cannot_be_named_as_the_mining_origin(mined):
    held = PartitionIdentity("test", 4, 1, ("amplitude",), (4, 8), {"split": "test"})
    with pytest.raises(InvalidParameterError, match="already used the data"):
        _frozen(mined, train=held)


def test_a_label_cannot_be_reused_for_a_different_graph(mined):
    result, candidate = mined
    other = Occurrence(candidate.exemplar.scene, candidate.exemplar.indices,
                       replace(candidate.exemplar.graph, attributes=(
                           dict(candidate.exemplar.graph.attributes[0], extent_in_scales=99.0),
                           *candidate.exemplar.graph.attributes[1:])))
    redefined = MotifCandidate(other, (other,), candidate.n_examined)
    with pytest.raises(InvalidParameterError, match="motif redefinition"):
        freeze_motif(
            redefined, result=result, origin_domain=_domain(), train=_train(),
            frozen_at="2026-08-26T08:00:00+12:00", study_id="tg5.1-tests")


def test_a_mixed_or_misnamed_origin_domain_is_refused(mined):
    with pytest.raises(InvalidParameterError, match="domain carried by every exemplar node"):
        _frozen(mined, origin_domain=_domain(name="some_other_domain"))


def test_the_freeze_time_must_support_the_future_opening_order(mined):
    with pytest.raises(InvalidParameterError, match="explicit UTC offset"):
        _frozen(mined, frozen_at="2026-08-26T08:00:00")


def test_unknown_serialized_meaning_is_refused(mined):
    record = _frozen(mined).to_mapping()
    record["definition"]["helpful_new_field"] = True
    with pytest.raises(InvalidParameterError, match="unrecognised meaning"):
        FrozenMotif.from_mapping(record)

    incomplete = _frozen(mined).to_mapping()
    del incomplete["origin_domain"]["licence"]
    with pytest.raises(InvalidParameterError, match="omitted meaning"):
        FrozenMotif.from_mapping(incomplete)


def test_noncanonical_bytes_are_not_an_alternative_artifact_identity(mined, tmp_path):
    motif = _frozen(mined)
    path = tmp_path / "motif.json"
    path.write_text(json.dumps(motif.to_mapping(), indent=2), encoding="utf-8")
    with pytest.raises(InvalidParameterError, match="canonical JSON"):
        load_frozen_motif(path)
