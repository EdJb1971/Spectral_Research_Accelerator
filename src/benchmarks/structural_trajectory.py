"""TG17.2 known-answer structural adapters and an inspectable preview payload.

These are deliberately benchmark adapters, not the live acquisition registrations promised by
TG17.3.  They prove that three unlike native records can enter one domain-blind mining seam while
retaining exact native support, meaning, units and lineage.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from src.benchmarks.multidomain_flagship import NativeDomainFixture, build_multidomain_flagship
from src.benchmarks.seeding import derive
from src.core.structural_trajectory import (
    NativeStructuralRecord, StructuralAdapterDeclaration, StructuralChannelDefinition,
    StructuralTrajectory, assert_structural_conformance, mine_structural_peak,
    translate_standardized_level,
)


FIXTURE_TO_MANIFEST = {"reanalysis": "reanalysis", "argo": "argo_float",
                       "tess": "tess_lightcurve", "order_book": "order_book"}
MANIFEST_TO_FIXTURE = {value: key for key, value in FIXTURE_TO_MANIFEST.items()}

_NATIVE = {
    "reanalysis": ("air temperature anomaly", "K", "air_temperature"),
    "argo_float": ("practical salinity profile structure", "1e-3", "salinity"),
    "tess_lightcurve": ("relative stellar flux", "dimensionless", "relative_flux"),
    "order_book": ("aggregated traded volume", "shares", "aggregated_volume"),
}


def known_answer_declaration(domain: str) -> StructuralAdapterDeclaration:
    semantics, units, _ = _NATIVE[domain]
    channel = StructuralChannelDefinition(
        name="standardized_level",
        semantics="within-record standardized structural level; native meaning is not erased",
        units="dimensionless",
        formula="(native_value - valid_native_mean) / valid_native_population_std",
        benchmark_id="g17.2.standardized-level-v1",
    )
    return StructuralAdapterDeclaration(
        adapter_id="%s.structural.known-answer" % domain,
        adapter_version="tg17.2-v1", domain=domain,
        accepted_semantics=semantics, accepted_units=units,
        required_axes=("native_time",), required_roles=("observation", "validity"),
        invariances=("native_value_translation", "native_value_positive_scaling"),
        consumed_information=("native_value", "native_time", "validity", "native_scale"),
        output_clock="identity_native_clock",
        output_support="[native_time_i, native_time_i + declared native scale)",
        missing_data_behavior="preserve validity exactly; never fill, compact, bin or interpolate",
        legitimate_null_family="domain-preserving native-clock shift with gaps preserved",
        leakage_risks=("semantic_equivalence", "native_magnitude_comparability",
                       "coverage_pattern_similarity"),
        refused_operations=("causality", "semantic_equivalence", "raw_magnitude_comparison",
                            "undeclared_interpolation"),
        channels={channel.name: channel},
    )


def native_from_fixture(fixture: NativeDomainFixture) -> NativeStructuralRecord:
    domain = FIXTURE_TO_MANIFEST[fixture.domain]
    _, _, variable = _NATIVE[domain]
    return NativeStructuralRecord(
        domain=domain, source_id="g17.0:%s" % fixture.domain, variable=variable,
        semantics=fixture.semantics, units=fixture.units,
        sample_times_seconds=fixture.sample_times_seconds, values=fixture.values,
        valid_mask=fixture.valid_mask, native_scale_seconds=fixture.native_scale_seconds,
        native_locator="benchmark://multidomain_flagship/shared_calendar_event/%s" % fixture.domain,
        assumption_violations=fixture.violations,
    )


def known_answer_native(domain: str, seed: int = 20260830) -> NativeStructuralRecord:
    """The deterministic TG17.0 native record for one manifest domain (TG17.3).

    Registered adapters bind to this by an explicit `source_binding` control, so a record that
    carries a planted answer can never be mistaken for an acquired observation: the choice is
    visible in the manifest and travels with every result derived from it.
    """
    fixture_domain = MANIFEST_TO_FIXTURE[domain]
    data = build_multidomain_flagship(derive("g17.2-structural-preview", seed), focus="planted")
    return native_from_fixture(data.cases["shared_calendar_event"].domains[fixture_domain])


def known_answer_trajectories(seed: int = 20260830) -> Mapping[str, StructuralTrajectory]:
    data = build_multidomain_flagship(derive("g17.2-structural-preview", seed), focus="planted")
    case = data.cases["shared_calendar_event"]
    result: Dict[str, StructuralTrajectory] = {}
    for fixture_domain in ("reanalysis", "argo", "tess"):
        native = native_from_fixture(case.domains[fixture_domain])
        declaration = known_answer_declaration(native.domain)
        trajectory = translate_standardized_level(native, declaration)
        assert_structural_conformance(trajectory, native, declaration)
        result[native.domain] = trajectory
    return result


def _plot_contract(trajectory: StructuralTrajectory) -> Dict[str, Any]:
    definition = trajectory.channel_definitions["standardized_level"]
    lineage = trajectory.lineage["standardized_level"]
    return {
        "trajectory_id": trajectory.trajectory_id,
        "domain": trajectory.domain,
        "source_id": trajectory.source_id,
        "variable": trajectory.variable,
        "native_semantics": trajectory.native_semantics,
        "native_units": trajectory.native_units,
        "support_start_seconds": trajectory.support_start_seconds.tolist(),
        "support_end_seconds": trajectory.support_end_seconds.tolist(),
        "valid_mask": trajectory.valid_mask.tolist(),
        "channel": definition.name,
        "channel_semantics": definition.semantics,
        "channel_units": definition.units,
        "values": trajectory.channels[definition.name].tolist(),
        "structural_scales": [vars(scale) for scale in trajectory.structural_scales],
        "adapter": {"id": trajectory.adapter_id, "version": trajectory.adapter_version,
                    "definition_sha256": trajectory.adapter_definition_sha256,
                    "config_sha256": trajectory.adapter_config_sha256},
        "native_record": {"content_sha256": trajectory.native_record_sha256,
                          "locator": trajectory.native_record_locator,
                          "retained": trajectory.native_record_retained},
        "lineage": {"operation": lineage.operation,
                    "source_variable": lineage.source_variable,
                    "source_indices": lineage.source_indices.tolist(),
                    "parameters": dict(lineage.parameters),
                    "output_sha256": lineage.output_sha256},
        "assumption_violations": list(trajectory.assumption_violations),
        "peak": dict(mine_structural_peak(trajectory)),
    }


def known_answer_preview(seed: int = 20260830) -> Dict[str, Any]:
    trajectories = known_answer_trajectories(seed)
    return {
        "schema": "structural-trajectory-preview/v1",
        "kind": "deterministic_known_answer_not_acquired_data",
        "seed": seed,
        "mining_interface": "mine_structural_peak(StructuralTrajectory, channel)",
        "domain_branch_in_mining": False,
        "trajectories": [_plot_contract(trajectory) for trajectory in trajectories.values()],
        "claim_boundary": (
            "This proves the canonical record and domain-blind mining seam on deterministic "
            "TG17 fixtures. It is not live acquisition, a cross-domain result, evidence, or "
            "a claim that native meanings or magnitudes are comparable."),
    }


__all__ = ["FIXTURE_TO_MANIFEST", "MANIFEST_TO_FIXTURE", "known_answer_declaration",
           "known_answer_native", "known_answer_preview", "known_answer_trajectories",
           "native_from_fixture"]
