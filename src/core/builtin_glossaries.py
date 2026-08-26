"""Built-in domain glossaries (TG9.1, `ed-dev`).

TG7.4 made domain wording declared data and gave it a registry. It shipped no glossaries,
because the module that defines the vocabulary is not the module that should decide what an
oceanographer calls a confounder. These are the first two, and they exist so `GET /domains`
has something truthful to serve and so the findings surface can demonstrate the property that
matters: **one study, two vocabularies, an identical set of facts.**

**Registration here is eager and explicit, and that is the whole point of the module.** Defect
D35 is the cautionary tale: source registration happened as an import side effect of a module
that was only imported lazily inside its own handler, so `GET /data/sources` returned
`['netcdf_local', 'simulated']` on a fresh process and a longer list after the researcher
happened to visit the ERA5 tab. Which sources the fallback chain considered depended on
browsing order. `DOMAIN_GLOSSARIES` has exactly that shape. `register_builtin_glossaries()` is
therefore called once at API startup, is idempotent, and returns what it registered so a caller
can assert the set rather than hope for it.

**On writing a glossary.** The four registration screens are strict, and deliberately so — the
constraint is the feature. A phrase may carry no digit, nothing from `OUTSIDE_THE_LADDER`, no
comparative asserting a relation of size, and no wording reserved to a rung above the one it
words. That last one bites hardest and most usefully: `RESERVED_WORDING` reserves "signal" to
`candidate_precursor`, so an atmospheric glossary cannot call an association a signal, which is
exactly the slippage the screen exists to catch. Both glossaries below were written against the
screens rather than fixed up afterwards.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.core.translation import DOMAIN_GLOSSARIES, DomainGlossary


#: A reanalysis vocabulary, for the programme's home domain.  Deliberately fluent: the reader
#: this exists for knows what a reanalysis field is and does not know what
#: `robust_association.confounders_addressed` means.
REANALYSIS_PHRASES: Dict[str, str] = {
    "rung.observation":
        "a reading taken from the reanalysis and written down",
    "rung.association":
        "a link between two reanalysis quantities, of a stated size, in the frames examined",
    "rung.robust_association":
        "a link that held on a second, separate stretch of the archive once known "
        "contaminating influences had been dealt with",
    "rung.candidate_precursor":
        "a link whose earlier and later parts are ordered in time and whose working can be "
        "traced back to the archive it came from",
    "rung.demonstrated_predictive_utility":
        "a link that was tested on frames held back from the fitting and still held",

    "gate.observation.no_failed_or_invalid_evidence":
        "the check that no reanalysis reading was written down as failed or unusable",
    "gate.observation.no_standing_contradiction":
        "the check that nothing in the archive record still stands against the idea",
    "gate.association.observation_recorded":
        "the check that at least one reanalysis reading was taken and kept",
    "gate.association.effect_size_estimated":
        "the check that the size of the link was estimated and not merely asserted",
    "gate.association.uncertainty_quantified":
        "the check that a range was put around that estimate",
    "gate.robust_association.replicated":
        "the check that the work was repeated on a separate stretch of the archive",
    "gate.robust_association.confounders_addressed":
        "the check that contaminating influences were named and each one dealt with",
    "gate.candidate_precursor.provenance_auditable":
        "the check that the working can be traced back to the frames it came from",
    "gate.candidate_precursor.temporal_precedence_recorded":
        "the check that the earlier part was written down as falling earlier in the archive",
    "gate.demonstrated_predictive_utility.out_of_sample":
        "the check that performance was taken on frames held back from the fitting",

    "alternative.nothing_measured":
        "no reanalysis reading was taken at all, so there is nothing here to account for",
    "alternative.no_estimated_effect":
        "no size was put on the link, so it has no stated magnitude",
    "alternative.chance":
        "the apparent link is the ordinary wobble of a finite stretch of archive",
    "alternative.sample_specific":
        "the link belongs to the stretch of archive it was found in and to no other",
    "alternative.confounding":
        "a third atmospheric influence moves both sides of the link at once",
    "alternative.reverse_or_simultaneous_order":
        "the later part does not in fact fall after the earlier part in the archive",
    "alternative.unauditable_origin":
        "the working cannot be traced back to the frames it is said to rest on",
    "alternative.in_sample_optimism":
        "performance was taken on the very frames the fitting used",

    "category.observations": "a reading taken from the reanalysis",
    "category.effect_sizes": "an estimate of how large the link is",
    "category.uncertainty": "a range put around an estimate",
    "category.null_results": "an attempt that found nothing, kept rather than dropped",
    "category.replication_results": "an attempt repeated on a separate stretch of archive",
    "category.holdout_performance": "performance taken on frames held back from the fitting",
    "category.provenance": "the trail back to the frames and the working behind a result",
    "category.confounders": "a contaminating influence, and what was done about it",
    "category.contradictory_evidence": "something in the archive that stands against the idea",
    "category.failure_states": "a way the work is known to have gone wrong",

    "status.PASS": "held, against the archive",
    "status.FAIL": "did not hold, against the archive",
    "status.INVALID": "could not be relied upon at all",
    "status.INCONCLUSIVE": "settled nothing either way",
    "status.NOT_APPLICABLE": "did not arise for this work",
}


#: An order-book vocabulary, sharing nothing with the first.  Its job is to make the R19
#: boundary visible: two glossaries over one study must be unable to say anything about each
#: other, and a reader seeing both should find no sentence that compares them.
ORDER_BOOK_PHRASES: Dict[str, str] = {
    "rung.observation":
        "a quantity read off the order book and written down",
    "rung.association":
        "a link between two order-book quantities, of a stated size, over the window examined",
    "rung.robust_association":
        "a link that held over a second, separate window once known contaminating influences "
        "had been dealt with",
    "rung.candidate_precursor":
        "a link whose earlier and later parts are ordered in trading time and whose working "
        "can be traced back to the venue feed",
    "rung.demonstrated_predictive_utility":
        "a link that was tested on sessions held back from the fitting and still held",

    "gate.observation.no_failed_or_invalid_evidence":
        "the control that no order-book quantity was written down as failed or unusable",
    "gate.observation.no_standing_contradiction":
        "the control that nothing in the trade record still stands against the idea",
    "gate.association.observation_recorded":
        "the control that at least one order-book quantity was read and kept",
    "gate.association.effect_size_estimated":
        "the control that the size of the link was estimated and not merely asserted",
    "gate.association.uncertainty_quantified":
        "the control that a range was put around that estimate",
    "gate.robust_association.replicated":
        "the control that the work was repeated over a separate window",
    "gate.robust_association.confounders_addressed":
        "the control that contaminating influences were named and each one dealt with",
    "gate.candidate_precursor.provenance_auditable":
        "the control that the working can be traced back to the venue feed",
    "gate.candidate_precursor.temporal_precedence_recorded":
        "the control that the earlier part was written down as falling earlier in the session",
    "gate.demonstrated_predictive_utility.out_of_sample":
        "the control that performance was taken on sessions held back from the fitting",

    "alternative.nothing_measured":
        "no order-book quantity was read at all, so there is nothing here to account for",
    "alternative.no_estimated_effect":
        "no size was put on the link, so it has no stated magnitude",
    "alternative.chance":
        "the apparent link is the ordinary churn of a finite window of trading",
    "alternative.sample_specific":
        "the link belongs to the window it was found in and to no other",
    "alternative.confounding":
        "a third market influence moves both sides of the link at once",
    "alternative.reverse_or_simultaneous_order":
        "the later part does not in fact fall after the earlier part in the session",
    "alternative.unauditable_origin":
        "the working cannot be traced back to the feed it is said to rest on",
    "alternative.in_sample_optimism":
        "performance was taken on the very sessions the fitting used",

    "category.observations": "a quantity read off the order book",
    "category.effect_sizes": "an estimate of how large the link is",
    "category.uncertainty": "a range put around an estimate",
    "category.null_results": "an attempt that found nothing, kept rather than dropped",
    "category.replication_results": "an attempt repeated over a separate window",
    "category.holdout_performance": "performance taken on sessions held back from the fitting",
    "category.provenance": "the trail back to the feed and the working behind a result",
    "category.confounders": "a contaminating influence, and what was done about it",
    "category.contradictory_evidence":
        "something in the trade record that stands against the idea",
    "category.failure_states": "a way the work is known to have gone wrong",

    "status.PASS": "held, at settlement",
    "status.FAIL": "did not hold, at settlement",
    "status.INVALID": "could not be relied upon at all",
    "status.INCONCLUSIVE": "settled nothing either way",
    "status.NOT_APPLICABLE": "did not arise for this work",
}


BUILTIN_GLOSSARIES = (
    ("reanalysis", REANALYSIS_PHRASES,
     "Reanalysis wording for the structural vocabulary, for a reader who knows the archive "
     "and not the claim ladder."),
    ("order_book", ORDER_BOOK_PHRASES,
     "Order-book wording for the structural vocabulary, sharing no term with the reanalysis "
     "glossary so the R19 boundary is visible to a reader."),
)

#: The same descriptions keyed by name, for `builtin_domains`, which onboards wording and
#: declaration together and needs the description without unpacking a tuple of tuples.
BUILTIN_GLOSSARY_DESCRIPTIONS: Dict[str, str] = {
    name: description for name, _phrases, description in BUILTIN_GLOSSARIES
}


def register_builtin_glossaries() -> Tuple[str, ...]:
    """Register every built-in glossary, once, and return the names registered.

    **Since TG8.1 this delegates to `register_builtin_domains`, and that is the point.** A
    glossary is no longer registrable on its own: `onboard_domain` writes wording, declaration
    and contract record atomically, because wording without a declaration is a domain that
    speaks fluently and refuses nothing — which TG9.1 shipped and served for a slice before
    TG9.3 caught it. Keeping this function as a way in that quietly produced that state again
    would leave the hole open beside the fix for it.

    The name survives because callers use it and because "register the built-in glossaries" is
    still what a reader means. The import is deferred to the call rather than taken at module
    level: `builtin_domains` imports the phrase maps from *this* module, so a top-level import
    in this direction would be a cycle.

    Idempotent, and returns the full built-in set rather than only the newly added ones, so a
    caller can assert what is available instead of inferring it from whether it happened to be
    first.
    """
    from src.core.builtin_domains import register_builtin_domains

    return register_builtin_domains()


__all__ = ["REANALYSIS_PHRASES", "ORDER_BOOK_PHRASES", "BUILTIN_GLOSSARIES",
           "BUILTIN_GLOSSARY_DESCRIPTIONS", "register_builtin_glossaries"]
