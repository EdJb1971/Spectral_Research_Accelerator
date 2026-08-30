"""Executable Argo domain extension, deliberately outside :mod:`src` (TG12.2d)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.adapters.standardized_level_adapter import build_standardized_level_adapter
from src.benchmarks.structural_trajectory import known_answer_native
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.experiment_adapter import (AcquisitionPlan, ControlField, EXPERIMENT_ADAPTERS,
                                         register_experiment_adapter)
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain

DOMAIN_NAME = "argo_float"
ADAPTER_ID = "argo_float.standardized-level"

ARGO = DomainDeclaration(
    name=DOMAIN_NAME,
    description=("Autonomous profiling floats reporting temperature and salinity from the "
                 "upper ocean on a nominal park-and-profile cycle they do not keep exactly."),
    axes=(AxisSpec(name="cycle_time", role="time", units="s"),
          AxisSpec(name="latitude", role="space", units="m", ordinal=0),
          AxisSpec(name="longitude", role="space", units="m", ordinal=1),
          AxisSpec(name="pressure", role="level", units="dbar"),
          AxisSpec(name="float_id", role="category", ordered=False)),
    licence=("Argo data are freely available under the Argo Data Management licence; the Argo "
             "Program and the national programmes that contribute floats must be cited."),
    violations=("irregular_sampling", "non_stationary_support"),
    lag_policy="declared", declared_floor_frames=1,
    declared_floor_basis=(
        "one park-and-profile cycle. A float reports once per ascent, so two profiles from the "
        "same float are the shortest interval at which that float says anything twice; a lag "
        "shorter than one cycle is an interval no instrument in this array can resolve."),
    provenance={
        "declared_by": "extensions/argo_float.py",
        "archive": "Argo Global Data Assembly Centre",
        "archive_doi": "https://doi.org/10.17882/42182",
        "note": ("The floor is the array's reporting interval, not a transport time: nothing "
                 "here claims to know how fast structure moves in the ocean."),
    })

ARGO_PHRASES: Dict[str, str] = {
    "rung.observation": "a value returned by a float and written down",
    "rung.association": "a link between two float quantities, of a stated size, over the cycles examined",
    "rung.robust_association": "a link that held over a second, separate span of the array once known contaminating influences had been dealt with",
    "rung.candidate_precursor": "a link whose earlier and later parts are ordered in cycle time and whose working can be traced back to the float that reported it",
    "rung.demonstrated_predictive_utility": "a link that was tested on cycles held back from the fitting and still held",
    "gate.observation.no_failed_or_invalid_evidence": "the check that no float value was written down as failed or unusable",
    "gate.observation.no_standing_contradiction": "the check that nothing in the array record still stands against the idea",
    "gate.association.observation_recorded": "the check that at least one float value was returned and kept",
    "gate.association.effect_size_estimated": "the check that the size of the link was estimated and not merely asserted",
    "gate.association.uncertainty_quantified": "the check that a range was put around that estimate",
    "gate.robust_association.replicated": "the check that the work was repeated over a separate span of the array",
    "gate.robust_association.confounders_addressed": "the check that contaminating influences were named and each one dealt with",
    "gate.candidate_precursor.provenance_auditable": "the check that the working can be traced back to the floats it came from",
    "gate.candidate_precursor.temporal_precedence_recorded": "the check that the earlier part was written down as falling earlier in cycle time",
    "gate.demonstrated_predictive_utility.out_of_sample": "the check that performance was taken on cycles held back from the fitting",
    "alternative.nothing_measured": "no float value was returned at all, so there is nothing here to account for",
    "alternative.no_estimated_effect": "no size was put on the link, so it has no stated magnitude",
    "alternative.chance": "the apparent link is the ordinary drift of a finite span of the array",
    "alternative.sample_specific": "the link belongs to the span of the array it was found in and to no other",
    "alternative.confounding": "a third ocean influence moves both sides of the link at once",
    "alternative.reverse_or_simultaneous_order": "the later part does not in fact fall after the earlier part in cycle time",
    "alternative.unauditable_origin": "the working cannot be traced back to the floats it is said to rest on",
    "alternative.in_sample_optimism": "performance was taken on the very cycles the fitting used",
    "category.observations": "a value returned by a float",
    "category.effect_sizes": "an estimate of how large the link is",
    "category.uncertainty": "a range put around an estimate",
    "category.null_results": "an attempt that found nothing, kept rather than dropped",
    "category.replication_results": "an attempt repeated over a separate span of the array",
    "category.holdout_performance": "performance taken on cycles held back from the fitting",
    "category.provenance": "the trail back to the floats and the working behind a result",
    "category.confounders": "a contaminating influence, and what was done about it",
    "category.contradictory_evidence": "something in the array record that stands against the idea",
    "category.failure_states": "a way the work is known to have gone wrong",
    "status.PASS": "held, against the array",
    "status.FAIL": "did not hold, against the array",
    "status.INVALID": "could not be relied upon at all",
    "status.INCONCLUSIVE": "settled nothing either way",
    "status.NOT_APPLICABLE": "did not arise for this work",
}

if DOMAIN_NAME in DOMAIN_ONBOARDINGS:
    ONBOARDED = DOMAIN_ONBOARDINGS.get(DOMAIN_NAME)
else:
    ONBOARDED = onboard_domain(
        ARGO, ARGO_PHRASES, geometry="latlon",
        glossary_description=("Profiling-float wording for the structural vocabulary, for a "
                              "reader who knows the array and not the claim ladder."))

# ---------------------------------------------------------------- TG17.3 experiment adapter

#: Argo's addressing is a sparse point support: floats report on a nominal park-and-profile
#: cycle they do not keep exactly, so a requested interval cannot be turned into an expected
#: sample count from a product description. `coverage_exact=False` and a `None` cadence are the
#: honest answer, and the framework then reports "unknown for native sparse/irregular support"
#: rather than inventing a number the array never promised.
ARGO_BYTES_PER_DAY = 24_000

ADAPTER_CONTROLS = (
    ControlField(name="pressure_dbar", label="Pressure level", kind="number", units="dbar",
                 help=("The pressure surface to read from each profile. A profile is indexed "
                       "by pressure, so this is part of what a result is about."),
                 default=100.0, minimum=0.0, maximum=6000.0),
)


def adapter_plan(parameters: Mapping[str, Any],
                 identity: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    return AcquisitionPlan(
        source_id="profile_query:argo_gdac_erddap", source_version="Argo-GDAC",
        support_kind="sparse_point_support", access="public_network",
        access_means="Public network access through the official Argo GDAC view.",
        coverage_exact=False, native_cadence_seconds=None,
        estimated_bytes_per_day=ARGO_BYTES_PER_DAY,
        identity={"licence_scope": "source-declared",
                  "pressure_dbar": parameters.get("pressure_dbar")})


def register_adapter() -> Any:
    """Register Argo's experiment adapter through the supported seam, from outside `src`."""
    if ADAPTER_ID in EXPERIMENT_ADAPTERS:
        return EXPERIMENT_ADAPTERS.get(ADAPTER_ID)
    if DOMAIN_NAME not in DOMAIN_ONBOARDINGS:
        onboard_domain(ARGO, ARGO_PHRASES, geometry="latlon",
                       glossary_description="Profiling-float wording.")
    return register_experiment_adapter(build_standardized_level_adapter(
        declaration=ARGO, adapter_id=ADAPTER_ID,
        accepted_semantics="practical salinity profile structure", accepted_units="1e-3",
        controls=ADAPTER_CONTROLS, plan=adapter_plan,
        fixture_record=lambda parameters: known_answer_native(DOMAIN_NAME),
        live_refusal=("a binding this slice can materialise. The Argo profile query exists "
                      "(see the Acquire tab) but is not yet wired to the canonical "
                      "translator; that is TG17.6's orchestrator."),
        # A float ascends when it ascends. A declared tolerance around a profile time is a
        # legitimate statement about how long an ascent takes; a fixed grid is not, because
        # the array does not keep the nominal cycle a grid would assume.
        admissible_kernels=("exact_support_overlap", "symmetric_tolerance"),
        domain_mathematics=(
            "the sparse point support and its consequences: a nominal cycle that the array "
            "does not keep means no expected sample count may be derived from an interval, "
            "and non-stationary support means the effective sample size differs per float.",)))


__all__ = ["ADAPTER_CONTROLS", "ADAPTER_ID", "ARGO", "ARGO_PHRASES", "ARGO_BYTES_PER_DAY",
           "DOMAIN_NAME", "ONBOARDED", "adapter_plan", "register_adapter"]
