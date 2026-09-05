"""The shared standardized-level adapter factory (TG17.3, `ed-dev`).

**This module is the onboarding-cost measurement.** TG17.3's acceptance requires that adapter
onboarding cost separate *unavoidable domain mathematics* from *framework glue*, and that the
glue trend to zero rather than merely move files. Four domains all running the same benchmarked
`standardized_level` translation should therefore share one implementation, and each domain
module should contain only what its own archive and declaration make different:

*   its `DomainDeclaration` — what it is and what it breaks;
*   its controls — what a researcher must choose to address the archive;
*   its `AcquisitionPlan` — native support kind, cadence, exactness, access and cost;
*   the native semantics and units its translator accepts;
*   its content-addressed record binding.

Everything else — the translation contract, the null, capability derivation, provenance
rendering — comes from here and is identical across domains by construction. When a domain needs
different structural mathematics it does not edit this factory: it registers its own adapter,
which is what the synthetic fifth adapter in the conformance suite does, and which is the only
thing that proves the seam is a seam rather than a shared helper with four callers.

**The live/fixture boundary is a control, not a hidden mode.** `source_binding` is an enum a
researcher selects. The benchmark binding returns the deterministic TG17.0 known-answer record
and says so in its provenance; the live binding refuses with the slice that will supply it.
Neither can be reached by accident, and a result carries which one produced it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.experiment_adapter import (AcquisitionPlan, ControlField, ControlSchema,
                                         DomainExperimentAdapter)
from src.core.structural_nulls import circular_clock_shift, describe_null
from src.core.structural_trajectory import (NativeStructuralRecord, StructuralAdapterDeclaration,
                                            StructuralTrajectory,
                                            standardized_level_declaration,
                                            translate_standardized_level)


ADAPTER_VERSION = "tg17.3-v1"

BENCHMARK_BINDING = "benchmark_known_answer"
LIVE_BINDING = "live_archive"

#: The one control every standardized-level adapter contributes, so that which record a result
#: came from is a visible choice rather than an environment variable or an import-time default.
SOURCE_BINDING = ControlField(
    name="source_binding", label="Record binding", kind="enum",
    help=("Which record this domain reads. The deterministic known-answer record is TG17.0 "
          "fixture data with a planted answer, not an observation; the live archive binding "
          "requires the acquisition path that lands with TG17.6."),
    choices=(BENCHMARK_BINDING, LIVE_BINDING), default=BENCHMARK_BINDING)


def build_standardized_level_adapter(
        *, declaration: DomainDeclaration,
        adapter_id: str,
        accepted_semantics: str,
        accepted_units: str,
        controls: Tuple[ControlField, ...],
        plan: Callable[[Mapping[str, Any]], AcquisitionPlan],
        fixture_record: Optional[Callable[[Mapping[str, Any]], NativeStructuralRecord]] = None,
        live_refusal: str = "",
        extra_leakage_risks: Tuple[str, ...] = (),
        extra_refused_operations: Tuple[str, ...] = (),
        domain_mathematics: Tuple[str, ...] = (),
        admissible_kernels: Tuple[str, ...] = ("exact_support_overlap",),
        admissible_nulls: Tuple[str, ...] = ("independent_native_clock_shift",),
        adapter_version: str = ADAPTER_VERSION) -> DomainExperimentAdapter:
    """One conforming adapter over the benchmarked standardized-level channel."""

    schema = ControlSchema((SOURCE_BINDING,) + tuple(controls))

    def structural_declaration(parameters: Mapping[str, Any]) -> StructuralAdapterDeclaration:
        return standardized_level_declaration(
            adapter_id=adapter_id, adapter_version=adapter_version, domain=declaration.name,
            accepted_semantics=accepted_semantics, accepted_units=accepted_units,
            extra_leakage_risks=extra_leakage_risks,
            extra_refused_operations=extra_refused_operations
            + (() if declaration.precedence_admissible else ("precedence",)))

    def translate(record: NativeStructuralRecord,
                  structural: StructuralAdapterDeclaration,
                  parameters: Mapping[str, Any]) -> StructuralTrajectory:
        return translate_standardized_level(record, structural)

    def materialize(parameters: Mapping[str, Any],
                    window: Any = None) -> NativeStructuralRecord:
        resolved = schema.validate(parameters)
        if resolved.get("source_binding") == LIVE_BINDING or fixture_record is None:
            raise InvalidParameterError(
                "source_binding", resolved.get("source_binding"),
                live_refusal or ("a binding this slice can materialise. Live acquisition for "
                                 "domain %r is not wired to the canonical translator yet; the "
                                 "deterministic known-answer binding is available and is "
                                 "labelled as fixture data wherever it is shown"
                                 % declaration.name))
        return fixture_record(resolved)

    def derive_capabilities(record: NativeStructuralRecord) -> Dict[str, Any]:
        """Facts about this record, stated as yes/no/not-established rather than as scores."""
        intervals = np.diff(np.asarray(record.sample_times_seconds))
        regular = bool(intervals.size and np.allclose(intervals, intervals[0], rtol=0.0,
                                                      atol=1e-9 * max(abs(float(intervals[0])), 1.0)))
        valid = int(np.count_nonzero(record.valid_mask))
        return {
            "domain": declaration.name,
            "native_record_sha256": record.content_sha256,
            "supports": len(record.sample_times_seconds),
            "valid_supports": valid,
            "complete_support": valid == len(record.sample_times_seconds),
            "regular_clock": regular,
            "cadence_seconds": float(intervals[0]) if regular else None,
            "precedence_admissible": declaration.precedence_admissible,
            "lag_policy": declaration.lag_policy,
            "violations": list(declaration.violations),
            "standardizable": bool(valid >= 2 and
                                   float(np.std(np.asarray(record.values)[record.valid_mask])) > 0),
            "not_established": ["archive coverage beyond this record",
                                "comparability of native magnitudes with any other domain"],
        }

    def build_null(trajectory: StructuralTrajectory, seed: int) -> StructuralTrajectory:
        return circular_clock_shift(trajectory, seed)

    def render_provenance(trajectory: StructuralTrajectory) -> Dict[str, Any]:
        lineage = {name: {"operation": item.operation,
                          "parameters": dict(item.parameters),
                          "source_indices": item.source_indices.tolist(),
                          "output_sha256": item.output_sha256}
                   for name, item in trajectory.lineage.items()}
        return {
            "domain": trajectory.domain, "source_id": trajectory.source_id,
            "variable": trajectory.variable,
            "native_semantics": trajectory.native_semantics,
            "native_units": trajectory.native_units,
            "native_record": {"sha256": trajectory.native_record_sha256,
                              "locator": trajectory.native_record_locator,
                              "retained": trajectory.native_record_retained},
            "adapter": {"id": trajectory.adapter_id, "version": trajectory.adapter_version,
                        "definition_sha256": trajectory.adapter_definition_sha256,
                        "config_sha256": trajectory.adapter_config_sha256},
            "structural_scales": [vars(scale) for scale in trajectory.structural_scales],
            "lineage": lineage,
            "assumption_violations": list(trajectory.assumption_violations),
            "null_family": describe_null(admissible_nulls[0]),
            "claim_boundary": (
                "Provenance shows how each canonical value arose. It does not make native "
                "magnitudes comparable across domains and does not license precedence."),
        }

    return DomainExperimentAdapter(
        adapter_id=adapter_id, adapter_version=adapter_version, declaration=declaration,
        controls=schema, plan_acquisition=plan,
        structural_declaration=structural_declaration, translate=translate,
        translator_config=lambda parameters: {"ddof": 0},
        materialize=materialize, derive_capabilities=derive_capabilities,
        build_null=build_null, render_provenance=render_provenance,
        admissible_kernels=admissible_kernels, admissible_nulls=admissible_nulls,
        onboarding_cost={
            "domain_mathematics": list(domain_mathematics),
            "framework_glue_lines": 0,
            "shared_from": "src/adapters/standardized_level_adapter.py",
            "supplied_by_domain": ["declaration", "controls", "acquisition plan",
                                   "accepted semantics and units", "record binding",
                                   "admissible alignment kernels"],
        })


__all__ = ["ADAPTER_VERSION", "BENCHMARK_BINDING", "LIVE_BINDING", "SOURCE_BINDING",
           "build_standardized_level_adapter"]
