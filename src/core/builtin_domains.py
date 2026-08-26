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
*   **order_book** breaks four assumptions and declares no lag policy. There is no length in
    metres, nothing propagates, the channels have no ordering, and values are window aggregates
    rather than instantaneous samples. Its lag policy is ``none``, which is the honest position
    for a domain with no propagation mechanism and is **not** a failure state: association may
    still be measured. What is refused is the lead-lag *interpretation* (R21).

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

from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration


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
                "aggregated_values"),
    lag_policy="none",
    provenance={
        "declared_by": "src/core/builtin_domains.py",
        "note": ("lag_policy 'none' is the honest position for a domain with no propagation "
                 "mechanism. Association remains measurable; precedence does not (R21)."),
    },
)


BUILTIN_DECLARATIONS = (REANALYSIS, ORDER_BOOK)


def register_builtin_domains() -> Tuple[str, ...]:
    """Register every built-in declaration, once, and return the names registered.

    Eager and idempotent for the same reason `register_builtin_glossaries` is: defect D35 was a
    registry whose contents depended on which handler had happened to run, and a domain that
    appears only after a researcher visits the right tab is a domain whose refusals can be missed.
    """
    for declaration in BUILTIN_DECLARATIONS:
        if declaration.name in DOMAIN_DECLARATIONS:
            continue
        DOMAIN_DECLARATIONS.add(
            declaration.name, declaration,
            description=declaration.description,
            capabilities={"builtin": True,
                          "precedence_admissible": declaration.precedence_admissible})
    return tuple(d.name for d in BUILTIN_DECLARATIONS)


__all__ = ["REANALYSIS", "ORDER_BOOK", "BUILTIN_DECLARATIONS", "register_builtin_domains"]
