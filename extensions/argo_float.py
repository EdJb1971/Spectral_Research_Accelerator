"""Executable Argo domain extension, deliberately outside :mod:`src` (TG12.2d)."""

from __future__ import annotations

from typing import Dict

from src.core.domain import AxisSpec, DomainDeclaration
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain

DOMAIN_NAME = "argo_float"

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

__all__ = ["ARGO", "ARGO_PHRASES", "DOMAIN_NAME", "ONBOARDED"]
