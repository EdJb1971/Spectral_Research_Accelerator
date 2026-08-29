"""Built-in domain declarations (TG9.3, `ed-dev`).

TG9.1 shipped a `/domains` route that listed **glossaries** — how a domain speaks — and nothing
about what it refuses. That was half of what the slice declared, and the missing half is the more
important one: a reader can be told a finding in fluent domain words while the domain those words
belong to does not admit the claim being made.

These are the declarations behind the two built-in vocabularies, and they are chosen to make that
tension visible rather than to look tidy:

*   **reanalysis** breaks nothing. It is the domain the analysis layer's assumptions were written
    against, structure is transported across it at a physical speed, and rule R21's advective
    floor therefore exists. Precedence is admissible. Rule R17 permits an empty violation set
    here only because the domain floors something; a domain that broke nothing *and* floored
    nothing would be refused as not being a second domain at all.
*   **order_book** breaks five assumptions and declares no lag policy. There is no length in
    metres, nothing propagates, the channels have no ordering, values are window aggregates
    rather than instantaneous samples, and the trading clock is not regularly spaced. Its lag
    policy is ``none``, which is the honest position for a domain with no propagation mechanism
    and is **not** a failure state: association may still be measured. What is refused is the
    lead-lag *interpretation* (R21).

    ``irregular_sampling`` was **missing until TG8.4 (defect D61)**. The description said
    "irregular trading clock" from the first commit and the violation tuple did not say it, so
    the declaration contradicted its own prose for four slices. Nothing caught it because
    nothing had yet tried to *read data* under the declaration — which is precisely the point of
    R17 and precisely what an ingestion seam is for.

So the same study, read in the second vocabulary, is a study whose domain does not admit a
precedence claim — and the refusal surface has to say so.

**What this cannot do, and it matters.** An `EvidenceBundle` does not record which domain
produced it: its fields are the hypothesis, the ten evidence categories and their digests, and
nothing else. So nothing here checks a study against a domain. Selecting a vocabulary states what
*that domain* refuses; it does not establish that the study came from it. Presenting it as a check
would be a fabrication, and both the API and the view say so in those words.
"""

from __future__ import annotations

from typing import Tuple

from src.core.builtin_glossaries import (BUILTIN_GLOSSARY_DESCRIPTIONS,
                                         ORDER_BOOK_PHRASES, REANALYSIS_PHRASES)
from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.onboarding import (DOMAIN_ONBOARDINGS, OnboardedDomain,
                                 onboard_domain)
from src.core.translation import DOMAIN_GLOSSARIES


REANALYSIS = DomainDeclaration(
    name="reanalysis",
    description=("Gridded atmospheric reanalysis on a regular clock, the domain this "
                 "programme's inherited assumptions were written against."),
    axes=(AxisSpec(name="time", role="time", units="s"),
          AxisSpec(name="latitude", role="space", units="m"),
          AxisSpec(name="longitude", role="space", units="m")),
    licence="Copernicus Climate Change Service licence; attribution required on redistribution.",
    violations=(),
    lag_policy="advective",
    provenance={
        "declared_by": "src/core/builtin_domains.py",
        "note": ("Breaks none of the inherited assumptions because they were derived from this "
                 "domain. That is not a claim of quality, only of provenance."),
    },
)


ORDER_BOOK = DomainDeclaration(
    name="order_book",
    description=("Per-instrument order-book channels on an irregular trading clock, with no "
                 "spatial extent and no mechanism that transports structure."),
    axes=(AxisSpec(name="timestamp", role="time", units="s"),
          AxisSpec(name="instrument", role="category", ordered=False)),
    licence="Venue market-data terms; redistribution of raw depth is generally prohibited.",
    violations=("no_physical_metric", "no_propagation_speed", "unordered_channels",
                "aggregated_values", "irregular_sampling"),
    lag_policy="none",
    provenance={
        "declared_by": "src/core/builtin_domains.py",
        "note": ("lag_policy 'none' is the honest position for a domain with no propagation "
                 "mechanism. Association remains measurable; precedence does not (R21). "
                 "'irregular_sampling' was added in TG8.4 (D61): the description declared an "
                 "irregular clock and the violation tuple did not."),
    },
)


#: Each built-in as the onboarding contract takes it: the declaration, its wording, and the
#: geometry that either supplies a physical metric or is renounced with `no_physical_metric`.
#: `reanalysis` is `latlon` — a real metre, non-uniform across rows. `order_book` supplies
#: `None`, which is why it must and does declare `no_physical_metric` (TG8.1).
BUILTIN_ONBOARDINGS = (
    (REANALYSIS, REANALYSIS_PHRASES, "latlon"),
    (ORDER_BOOK, ORDER_BOOK_PHRASES, None),
)

BUILTIN_DECLARATIONS = tuple(declaration for declaration, _phrases, _geometry
                             in BUILTIN_ONBOARDINGS)


def register_builtin_domains() -> Tuple[str, ...]:
    """Onboard every built-in domain, once, and return the names registered.

    Since TG8.1 this goes through `onboard_domain` rather than adding to `DOMAIN_DECLARATIONS`
    directly, so the built-ins are held to the same contract a third-party adapter is. That is
    not ceremony: the two built-ins are the worked examples an adapter author copies, and a
    worked example that skipped the contract would teach the shortcut rather than the recipe.

    Eager and idempotent for the reason `register_builtin_glossaries` is: defect D35 was a
    registry whose contents depended on which handler had happened to run, and a domain that
    appears only after a researcher visits the right tab is a domain whose refusals can be
    missed. **Repair rather than skip** is deliberate — a name present in some registries and
    not others is exactly the half-onboarded state the contract exists to forbid, so it is
    re-onboarded whole instead of being left as it was found.
    """
    for declaration, phrases, geometry in BUILTIN_ONBOARDINGS:
        name = declaration.name
        complete = (name in DOMAIN_GLOSSARIES and name in DOMAIN_DECLARATIONS
                    and name in DOMAIN_ONBOARDINGS)
        if complete:
            continue
        onboard_domain(declaration, phrases, geometry=geometry,
                       glossary_description=BUILTIN_GLOSSARY_DESCRIPTIONS[name],
                       onboarded_by=__name__, replace=True)
    return tuple(declaration.name for declaration, _phrases, _geometry in BUILTIN_ONBOARDINGS)


def builtin_onboarding(name: str) -> OnboardedDomain:
    """One built-in's completed contract, for a caller that wants its digest or checklist."""
    return DOMAIN_ONBOARDINGS.get(name)


__all__ = ["REANALYSIS", "ORDER_BOOK", "BUILTIN_DECLARATIONS", "BUILTIN_ONBOARDINGS",
           "builtin_onboarding", "register_builtin_domains"]
