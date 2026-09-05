"""Executable TESS light curve domain extension, outside :mod:src (TG13.1)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.adapters.standardized_level_adapter import build_standardized_level_adapter
from src.benchmarks.structural_trajectory import known_answer_native
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.experiment_adapter import (AcquisitionPlan, ControlField, EXPERIMENT_ADAPTERS,
                                         register_experiment_adapter)
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.physical_core.geometry import GEOMETRIES, Geometry

DOMAIN_NAME = "tess_lightcurve"
ADAPTER_ID = "tess_lightcurve.standardized-level"

TESS = DomainDeclaration(
    name=DOMAIN_NAME,
    description=("TESS photometric light curves of celestial targets, observed in sectors "
                 "with gaps, irregular sampling and targets entering/leaving."),
    axes=(AxisSpec(name="time", role="time", units="d"),
          AxisSpec(name="ra", role="space", units="deg", ordinal=0),
          AxisSpec(name="dec", role="space", units="deg", ordinal=1),
          AxisSpec(name="target_id", role="category", ordered=False)),
    licence=("NASA/STScI public archive data. Publications using MAST data must acknowledge "
             "MAST and cite the producing TESS pipeline/product documentation."),
    violations=("no_natural_cycle", "irregular_sampling", "non_stationary_support"),
    lag_policy="none",
    declared_floor_frames=None,
    declared_floor_basis=None,
    provenance={
        "declared_by": "extensions/tess_lightcurve.py",
        "archive": "MAST (Mikulski Archive for Space Telescopes)",
        "time_scale": "BJD_TDB",
        "note": "Angular separation is a metric, but this domain has no advective propagation."
    }
)

TESS_PHRASES: Dict[str, str] = {
    "rung.observation": "a flux measurement taken from the telescope and written down",
    "rung.association": "a link between two photometric quantities, of a stated size, in the sectors examined",
    "rung.robust_association": "a link that held on a second, separate stretch of observations once known contaminating influences had been dealt with",
    "rung.candidate_precursor": "a link whose earlier and later parts are ordered in time and whose working can be traced back to the target it came from",
    "rung.demonstrated_predictive_utility": "a link that was tested on sectors held back from the fitting and still held",
    "gate.observation.no_failed_or_invalid_evidence": "the check that no flux value was written down as failed or unusable",
    "gate.observation.no_standing_contradiction": "the check that nothing in the record still stands against the idea",
    "gate.association.observation_recorded": "the check that at least one measurement was returned and kept",
    "gate.association.effect_size_estimated": "the check that the size of the link was estimated and not merely asserted",
    "gate.association.uncertainty_quantified": "the check that a range was put around that estimate",
    "gate.robust_association.replicated": "the check that the work was repeated over a separate span of the observations",
    "gate.robust_association.confounders_addressed": "the check that contaminating influences were named and each one dealt with",
    "gate.candidate_precursor.provenance_auditable": "the check that the working can be traced back to the targets it came from",
    "gate.candidate_precursor.temporal_precedence_recorded": "the check that the earlier changes actually happened first",
    "gate.demonstrated_predictive_utility.out_of_sample": "the check that the link was tested on sectors held back from the fitting",
    "alternative.nothing_measured": "no flux was returned at all",
    "alternative.no_estimated_effect": "no size was put on the link",
    "alternative.chance": "the link could be a fluke",
    "alternative.sample_specific": "the link might only hold for these specific targets",
    "alternative.confounding": "a third factor might be driving both quantities",
    "alternative.reverse_or_simultaneous_order": "the quantities might be changing together, or in reverse order",
    "alternative.unauditable_origin": "the origin of the data cannot be audited",
    "alternative.in_sample_optimism": "the link might fit the training data well but fail on new sectors",
    "category.observations": "a value returned by a telescope",
    "category.effect_sizes": "an estimate of how large the link is",
    "category.uncertainty": "a range put around an estimate",
    "category.null_results": "an attempt that found nothing, kept rather than dropped",
    "category.replication_results": "an attempt repeated over a separate span of observations",
    "category.holdout_performance": "performance taken on sectors held back from the fitting",
    "category.provenance": "the trail back to the telescope and the working behind a result",
    "category.confounders": "a contaminating influence, and what was done about it",
    "category.contradictory_evidence": "something in the record that stands against the idea",
    "category.failure_states": "a way the work is known to have gone wrong",
    "status.PASS": "held, against the targets",
    "status.FAIL": "did not hold, against the targets",
    "status.INVALID": "could not be relied upon at all",
    "status.INCONCLUSIVE": "settled nothing either way",
    "status.NOT_APPLICABLE": "did not arise for this work",
}

class AngularSkyGeometry(Geometry):
    """Point-set angular geometry; deliberately unavailable to grid differential operators."""

    name = "angular_sky"
    coord_priority = 30

    def validate(self, spec: Any) -> None:
        from src.physical_core.grid import GridError
        raise GridError(
            "angular_sky describes celestial point locations, not a raster GridSpec. "
            "Angular separation is available to point-set consumers; grid derivatives are not.")


def register() -> Any:
    """Register the TESS domain and its wording."""
    if "angular_sky" not in GEOMETRIES:
        GEOMETRIES.add(
            "angular_sky", AngularSkyGeometry(),
            description=("Great-circle angular geometry for celestial point locations; "
                         "not a raster differential geometry."),
            capabilities={"physical_metric": True, "uniform_metric": None, "spherical": True,
                          "has_latitude": False, "length_units": "deg",
                          "grid_compatible": False, "point_metric": "great_circle"})
    if DOMAIN_NAME in DOMAIN_ONBOARDINGS:
        return DOMAIN_ONBOARDINGS.get(DOMAIN_NAME)
    return onboard_domain(
        TESS, TESS_PHRASES, geometry="angular_sky",
        glossary_description=("TESS photometry wording for a reader who knows targets, "
                              "sectors and quality flags rather than the claim ladder."))


# ---------------------------------------------------------------- TG17.3 experiment adapter

#: A sector that intersects a requested window says an observation exists somewhere inside it,
#: never that the window is covered. TG17.1 froze that distinction and the conformance kit's
#: `coverage_honesty` check now enforces it: this plan may state its nominal short cadence and
#: must still declare `coverage_exact=False`.
TESS_BYTES_PER_DAY = 80_000
SHORT_CADENCE_SECONDS = 120.0
LONG_CADENCE_SECONDS = 1_800.0

ADAPTER_CONTROLS = (
    ControlField(name="cadence", label="Cadence", kind="enum", choices=("short", "long"),
                 default="short",
                 help=("Which SPOC product this observation reads. The cadence changes what "
                       "structure is resolvable at all, so it is a scientific choice rather "
                       "than a performance setting.")),
)


def adapter_plan(parameters: Mapping[str, Any],
                 identity: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    cadence = (SHORT_CADENCE_SECONDS if parameters.get("cadence") == "short"
               else LONG_CADENCE_SECONDS)
    return AcquisitionPlan(
        source_id="lightcurve_query:mast_tess_spoc", source_version="MAST-TESS-SPOC",
        support_kind="intersecting_observational_sectors", access="public_network",
        access_means="Public network access through MAST/AWS.",
        coverage_exact=False, native_cadence_seconds=cadence,
        estimated_bytes_per_day=TESS_BYTES_PER_DAY,
        identity={"licence_scope": "source-declared", "cadence": parameters.get("cadence")})


def register_adapter() -> Any:
    """Register TESS's experiment adapter through the supported seam, from outside `src`."""
    if ADAPTER_ID in EXPERIMENT_ADAPTERS:
        return EXPERIMENT_ADAPTERS.get(ADAPTER_ID)
    register()
    return register_experiment_adapter(build_standardized_level_adapter(
        declaration=TESS, adapter_id=ADAPTER_ID,
        accepted_semantics="relative stellar flux", accepted_units="dimensionless",
        controls=ADAPTER_CONTROLS, plan=adapter_plan,
        fixture_record=lambda parameters: known_answer_native(DOMAIN_NAME),
        live_refusal=("a binding this slice can materialise. The MAST light-curve query "
                      "exists (see the Acquire tab) but is not yet wired to the canonical "
                      "translator; that is TG17.6's orchestrator."),
        # Cadence is a real property of a SPOC product, so binning to a coarser shared grid
        # is a statement about it. A tolerance is not: widening sector support would blur
        # exactly the observational gap that decides whether a target was observed at all.
        admissible_kernels=("exact_support_overlap", "common_grid_aggregate"),
        # A light curve is delivered per sector, and a sector is a real group: the
        # instrument, the pointing and the systematics change at its boundary. A shift
        # that moved flux across it would produce a curve no spacecraft could record.
        # No annual cycle is claimed for a target, so the whole-cycle shift is not
        # admitted: admitting a null whose preserved feature this domain does not have
        # would let a study declare a seasonal guarantee nothing here supports.
        admissible_nulls=("independent_native_clock_shift", "within_group_clock_shift",
                          "scale_partner_reassignment"),
        domain_mathematics=(
            "sector-bounded observational support: coverage is established by which sectors "
            "actually observed a target, never by an interval intersecting a sector, and no "
            "physical metric or propagation speed exists to floor a lag.",)))


__all__ = ["ADAPTER_CONTROLS", "ADAPTER_ID", "AngularSkyGeometry", "DOMAIN_NAME",
           "LONG_CADENCE_SECONDS", "SHORT_CADENCE_SECONDS", "TESS", "TESS_BYTES_PER_DAY",
           "TESS_PHRASES", "adapter_plan", "register", "register_adapter"]
