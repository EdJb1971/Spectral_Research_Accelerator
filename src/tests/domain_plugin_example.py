"""A third domain, onboarded in this file and nowhere else (TG8.1's acceptance criterion).

This is to `onboard_domain` what `plugin_example.py` is to the source and action registries: the
executable form of the claim that a domain can be added **without editing `src/`**. Nothing in
`src/` imports this module. `test_domain_onboarding.py` imports it, then checks that the domain
is live in every registry *and* that the files it would otherwise have had to touch are
byte-identical afterwards.

**Why Argo, and not a fifth trading venue.** Rule R17: *a domain that violates nothing is not a
second domain.* The same reasoning refuses a domain that violates only what an existing one
already violates — `physical_core/geometry.py` makes the point about geometries, where *"a fourth
geometry that is merely `cartesian` with a different name proves nothing."* Of the seven entries
in `KNOWN_VIOLATIONS`, the two registered domains between them break five; `non_stationary_support`
has never been broken by any registered domain, so no refusal that depends on it has ever fired
against a declared source.

**Corrected after the fact, and worth reading as a lesson rather than a tidy-up.** This paragraph
originally claimed *two* previously-unbroken violations, `irregular_sampling` among them. TG8.4
found that `order_book` had described an "irregular trading clock" from its first commit while
omitting `irregular_sampling` from its violation tuple (**D61**), so one of the two was really a
gap in an existing declaration rather than a contribution from this one. Nothing caught it for
four slices because nothing had yet tried to read data under the declaration.

Argo declares both, and declares them for reasons that are physically real rather than
contrived:

*   **`irregular_sampling`** — a float parks at depth, drifts, and surfaces to report on a
    nominal cycle it does not keep precisely. The interval between two profiles is a property of
    the float and the ocean, not of a clock anyone set. (Shared with `order_book` since D61 was
    fixed; still true of Argo, and still declared here on its own account.)
*   **`non_stationary_support`** — floats are deployed, fail, run out of battery and are
    replaced throughout the record. Channels genuinely start and stop mid-series, so the
    effective sample size differs per float and per pair.

And it is the first declared domain in which **precedence is admissible while the clock is
irregular**. Neither built-in has that shape: `reanalysis` is regular and advective, `order_book`
has no floor at all. Argo therefore uses `lag_policy="declared"` — the only policy no built-in
exercises — with a floor whose basis is a real instrument behaviour rather than a plausible
number, which is the distinction `DeclaredPolicy` refuses a declaration for missing.

It also supplies a **physical metric**: float positions are latitudes and longitudes on a sphere,
so `geometry="latlon"` and the declaration must *not* claim `no_physical_metric`. That exercises
the onboarding biconditional in the direction the built-ins do not — `reanalysis` has a metric
and no violations at all, `order_book` renounces the metric — so the pairing here is the case
where a domain breaks assumptions *and* keeps its lengths.
"""

from __future__ import annotations

from typing import Dict

from src.core.domain import AxisSpec, DomainDeclaration
from src.core.onboarding import onboard_domain

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
    lag_policy="declared",
    declared_floor_frames=1,
    declared_floor_basis=(
        "one park-and-profile cycle. A float reports once per ascent, so two profiles from the "
        "same float are the shortest interval at which that float says anything twice; a lag "
        "shorter than one cycle is an interval no instrument in this array can resolve."),
    provenance={
        "declared_by": "src/tests/domain_plugin_example.py",
        "archive": "Argo Global Data Assembly Centre",
        "note": ("Declared from outside `src/` to demonstrate TG8.1's acceptance criterion. "
                 "The floor is the array's reporting interval, not a transport time: nothing "
                 "here claims to know how fast structure moves in the ocean."),
    },
)


#: Float wording for the closed structural vocabulary. Written against TG7.4's four registration
#: screens rather than fixed up afterwards — no digit, no term from `OUTSIDE_THE_LADDER`, no
#: comparative asserting a relation of size, and no rung borrowing wording reserved above it.
#: The last screen is the one that bites: "signal" is reserved to `candidate_precursor`, which
#: rules out the phrase an oceanographer would reach for first.
ARGO_PHRASES: Dict[str, str] = {
    "rung.observation":
        "a value returned by a float and written down",
    "rung.association":
        "a link between two float quantities, of a stated size, over the cycles examined",
    "rung.robust_association":
        "a link that held over a second, separate span of the array once known contaminating "
        "influences had been dealt with",
    "rung.candidate_precursor":
        "a link whose earlier and later parts are ordered in cycle time and whose working can "
        "be traced back to the float that reported it",
    "rung.demonstrated_predictive_utility":
        "a link that was tested on cycles held back from the fitting and still held",

    "gate.observation.no_failed_or_invalid_evidence":
        "the check that no float value was written down as failed or unusable",
    "gate.observation.no_standing_contradiction":
        "the check that nothing in the array record still stands against the idea",
    "gate.association.observation_recorded":
        "the check that at least one float value was returned and kept",
    "gate.association.effect_size_estimated":
        "the check that the size of the link was estimated and not merely asserted",
    "gate.association.uncertainty_quantified":
        "the check that a range was put around that estimate",
    "gate.robust_association.replicated":
        "the check that the work was repeated over a separate span of the array",
    "gate.robust_association.confounders_addressed":
        "the check that contaminating influences were named and each one dealt with",
    "gate.candidate_precursor.provenance_auditable":
        "the check that the working can be traced back to the floats it came from",
    "gate.candidate_precursor.temporal_precedence_recorded":
        "the check that the earlier part was written down as falling earlier in cycle time",
    "gate.demonstrated_predictive_utility.out_of_sample":
        "the check that performance was taken on cycles held back from the fitting",

    "alternative.nothing_measured":
        "no float value was returned at all, so there is nothing here to account for",
    "alternative.no_estimated_effect":
        "no size was put on the link, so it has no stated magnitude",
    "alternative.chance":
        "the apparent link is the ordinary drift of a finite span of the array",
    "alternative.sample_specific":
        "the link belongs to the span of the array it was found in and to no other",
    "alternative.confounding":
        "a third ocean influence moves both sides of the link at once",
    "alternative.reverse_or_simultaneous_order":
        "the later part does not in fact fall after the earlier part in cycle time",
    "alternative.unauditable_origin":
        "the working cannot be traced back to the floats it is said to rest on",
    "alternative.in_sample_optimism":
        "performance was taken on the very cycles the fitting used",

    "category.observations": "a value returned by a float",
    "category.effect_sizes": "an estimate of how large the link is",
    "category.uncertainty": "a range put around an estimate",
    "category.null_results": "an attempt that found nothing, kept rather than dropped",
    "category.replication_results": "an attempt repeated over a separate span of the array",
    "category.holdout_performance": "performance taken on cycles held back from the fitting",
    "category.provenance": "the trail back to the floats and the working behind a result",
    "category.confounders": "a contaminating influence, and what was done about it",
    "category.contradictory_evidence":
        "something in the array record that stands against the idea",
    "category.failure_states": "a way the work is known to have gone wrong",

    "status.PASS": "held, against the array",
    "status.FAIL": "did not hold, against the array",
    "status.INVALID": "could not be relied upon at all",
    "status.INCONCLUSIVE": "settled nothing either way",
    "status.NOT_APPLICABLE": "did not arise for this work",
}


#: Onboarded at import, in one call, from outside `src/`. `geometry` is passed explicitly
#: because `onboard_domain` gives it no default: float positions are on a sphere with a real
#: metre, so the declaration must not claim `no_physical_metric`, and the contract checks that
#: the two statements agree.
ONBOARDED = onboard_domain(
    ARGO, ARGO_PHRASES,
    geometry="latlon",
    glossary_description=("Profiling-float wording for the structural vocabulary, for a reader "
                          "who knows the array and not the claim ladder."),
)


__all__ = ["ARGO", "ARGO_PHRASES", "DOMAIN_NAME", "ONBOARDED"]
