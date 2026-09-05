"""TG17.2 canonical structural-trajectory contract and visible Composer preview."""

import dataclasses
import inspect

import numpy as np
import pytest

from src.benchmarks.structural_trajectory import (
    known_answer_declaration, known_answer_preview, known_answer_trajectories,
)
from src.core.structural_trajectory import (
    StructuralConformanceError, assert_structural_conformance, mine_structural_peak,
)


def test_weather_argo_and_tess_share_one_domain_blind_mining_interface():
    trajectories = known_answer_trajectories()
    assert tuple(trajectories) == ("reanalysis", "argo_float", "tess_lightcurve")
    results = [mine_structural_peak(item) for item in trajectories.values()]
    assert all(result["channel"] == "standardized_level" for result in results)
    source = inspect.getsource(mine_structural_peak)
    assert "reanalysis" not in source and "argo" not in source and "tess" not in source


def test_translation_preserves_native_clock_intervals_gaps_and_raw_record_identity():
    preview = known_answer_preview()
    for row in preview["trajectories"]:
        assert len(row["support_start_seconds"]) == len(row["support_end_seconds"])
        assert len(row["valid_mask"]) == len(row["values"])
        assert np.all(np.diff(row["support_start_seconds"]) > 0)
        assert np.all(np.asarray(row["support_end_seconds"]) > row["support_start_seconds"])
        assert row["native_record"]["retained"] is True
        assert row["native_record"]["locator"].startswith("benchmark://")


def test_every_canonical_value_reconstructs_from_declared_lineage():
    from src.benchmarks.multidomain_flagship import build_multidomain_flagship
    from src.benchmarks.seeding import derive
    from src.benchmarks.structural_trajectory import MANIFEST_TO_FIXTURE, native_from_fixture
    data = build_multidomain_flagship(
        derive("g17.2-structural-preview", 20260830), focus="planted")
    for domain, trajectory in known_answer_trajectories().items():
        fixture = data.cases["shared_calendar_event"].domains[MANIFEST_TO_FIXTURE[domain]]
        native = native_from_fixture(fixture)
        assert_structural_conformance(trajectory, native, known_answer_declaration(domain))
        lineage = trajectory.lineage["standardized_level"]
        rebuilt = ((native.values - lineage.parameters["valid_native_mean"]) /
                   lineage.parameters["valid_native_population_std"])
        assert np.array_equal(lineage.source_indices, np.arange(len(native.values)))
        assert np.allclose(rebuilt, trajectory.channels["standardized_level"],
                           rtol=0, atol=1e-12)


def test_semantic_leakage_fails_before_translation():
    trajectory = known_answer_trajectories()["reanalysis"]
    declaration = known_answer_declaration("reanalysis")
    from src.core.structural_trajectory import NativeStructuralRecord, translate_standardized_level
    native = NativeStructuralRecord(
        domain="reanalysis", source_id="fixture", variable="salinity",
        semantics="practical salinity", units="1e-3",
        sample_times_seconds=trajectory.support_start_seconds,
        values=trajectory.channels["standardized_level"], valid_mask=trajectory.valid_mask,
        native_scale_seconds=10800, native_locator="benchmark://wrong-semantics")
    with pytest.raises(StructuralConformanceError, match="semantic leakage"):
        translate_standardized_level(native, declaration)


def test_undeclared_interpolation_or_clock_change_fails_conformance():
    from src.benchmarks.multidomain_flagship import build_multidomain_flagship
    from src.benchmarks.seeding import derive
    from src.benchmarks.structural_trajectory import native_from_fixture
    fixture = build_multidomain_flagship(
        derive("g17.2-structural-preview", 20260830), focus="planted"
    ).cases["shared_calendar_event"].domains["argo"]
    native = native_from_fixture(fixture)
    declaration = known_answer_declaration("argo_float")
    trajectory = known_answer_trajectories()["argo_float"]
    shifted = dataclasses.replace(
        trajectory, support_start_seconds=trajectory.support_start_seconds + 1.0)
    with pytest.raises(StructuralConformanceError, match="interpolation, compaction or support"):
        assert_structural_conformance(shifted, native, declaration)


def test_adapter_contract_declares_scientific_behavior_and_only_benchmarked_channels():
    declaration = known_answer_declaration("tess_lightcurve")
    assert declaration.required_axes == ("native_time",)
    assert declaration.required_roles == ("observation", "validity")
    assert declaration.output_clock == "identity_native_clock"
    assert "never fill" in declaration.missing_data_behavior
    assert "coverage_pattern_similarity" in declaration.leakage_risks
    assert "undeclared_interpolation" in declaration.refused_operations
    assert declaration.channels["standardized_level"].benchmark_id
    assert len(declaration.definition_sha256) == 64
    from src.benchmarks.multidomain_flagship import build_multidomain_flagship
    from src.benchmarks.seeding import derive
    from src.benchmarks.structural_trajectory import native_from_fixture
    from src.core.structural_trajectory import translate_standardized_level
    fixture = build_multidomain_flagship(
        derive("g17.2-structural-preview", 20260830), focus="planted"
    ).cases["shared_calendar_event"].domains["tess"]
    with pytest.raises(StructuralConformanceError, match="only its benchmarked"):
        translate_standardized_level(native_from_fixture(fixture), declaration,
                                     config={"ddof": 1})


def test_canonical_plot_payload_cannot_drop_clock_units_support_or_adapter_digest():
    preview = known_answer_preview()
    assert preview["domain_branch_in_mining"] is False
    for row in preview["trajectories"]:
        assert row["support_start_seconds"] and row["support_end_seconds"]
        assert row["native_units"] and row["channel_units"]
        assert len(row["adapter"]["definition_sha256"]) == 64
        assert len(row["adapter"]["config_sha256"]) == 64
        assert len(row["native_record"]["content_sha256"]) == 64
        assert len(row["lineage"]["source_indices"]) == len(row["values"])


def test_structural_arrays_are_immutable_copies_and_do_not_overwrite_native_data():
    trajectory = known_answer_trajectories()["reanalysis"]
    with pytest.raises(ValueError):
        trajectory.channels["standardized_level"][0] = 99
    with pytest.raises(ValueError):
        trajectory.support_start_seconds[0] = 99


def test_composer_representation_preview_is_manifest_bound_and_honest(client):
    recipe = client.get(
        "/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    response = client.post(
        "/api/v1/experiment-composer/manifests/representation-preview",
        json=recipe["canonical_manifest"])
    assert response.status_code == 200
    body = response.json()
    assert body["manifest_sha256"] == recipe["manifest_sha256"]
    assert body["kind"] == "deterministic_known_answer_not_acquired_data"
    assert [row["domain"] for row in body["trajectories"]] == [
        "reanalysis", "argo_float", "tess_lightcurve"]
    assert "not live acquisition" in body["claim_boundary"]


def test_composer_contract_distinguishes_canonical_contract_from_live_translation(client):
    contract = client.get("/api/v1/experiment-composer").json()
    assert "StructuralTrajectory contract and known-answer preview" in contract["available_now"]
    assert "live adapter translation" in contract["not_yet_available"]
