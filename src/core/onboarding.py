"""The onboarding contract: what a domain must declare before it may be read (TG8.1, `ed-dev`).

**The hole this closes.** A domain reaches this programme through two registries that know
nothing about each other. `DOMAIN_GLOSSARIES` (TG7.4) takes its wording; `DOMAIN_DECLARATIONS`
(TG9.3) takes what it is and what it breaks. Either may be registered without the other, and the
asymmetry is not neutral:

*   **Wording without a declaration** is a domain that speaks fluently and refuses nothing. That
    is not hypothetical — TG9.1 shipped exactly that and served it for a slice before TG9.3
    caught it. A reader is handed a finding in confident domain language while the domain those
    words belong to has never stated whether it admits the claim.
*   **A declaration without wording** is a domain whose refusals exist and cannot be read, which
    fails quietly in the other direction.

`onboard_domain` makes the pair the unit. It registers a glossary, a declaration and the record
of the contract itself **atomically**: if any screen refuses, all three registries are restored
to the state they had before the call, and a half-onboarded domain does not exist. Registries
are process-global, so a partial failure that left one of them written would be a defect of
exactly the D35 family — behaviour depending on which registration happened to have run.

**The recipe, made enumerable rather than prose.** :data:`REQUIRED_DECLARATIONS` is the adapter
recipe the roadmap describes, in a form the API can render and a test can assert against. Each
item names what satisfies it, and every one is checked here or by the object it names:

===========================  ==============================================================
``axes``                     declared roles, never inferred from spelling (E14)
``geometry``                 a registered grid geometry, or none with the metric renounced (E13)
``lag_policy``               how the domain justifies calling a lag admissible (R21)
``violations``               which inherited assumptions this domain breaks (E15, R17)
``licence``                  the source's terms, required rather than optional (TG8.2)
``provenance``               who declared this, and on what basis
``glossary``                 the domain's wording for the structural vocabulary (TG7.4)
===========================  ==============================================================

**The one new check, and why it is the interesting one.** Everything above is already validated
by `DomainDeclaration` or `DomainGlossary` — except the relationship *between* the geometry and
the violations, which no single object can see because they live in different registries. The
contract is a biconditional:

    a domain declares ``no_physical_metric`` **if and only if** its geometry offers no physical
    metric.

Both halves bite. A domain that names ``cartesian`` while declaring ``no_physical_metric`` has
supplied a metric and then renounced it, and downstream code that asks the geometry gets metres
while code that asks the declaration gets a refusal — the two answers disagree and nothing
notices. A domain that supplies no geometry and does *not* declare ``no_physical_metric`` is
claiming lengths it cannot produce, which is the same failure with the sign flipped and is the
more common one: it is what an adapter author writes when they have not thought about it yet.

**What this does not do.** It does not check a domain against its data — no file is read here —
and it does not know whether the wording chosen is wording a practitioner would use. It also
establishes nothing about which domain produced any given `EvidenceBundle`, because a bundle
does not record one; see `DOMAIN_ATTRIBUTION_CAVEAT` in `src/api/findings.py`. The contract is
about completeness and internal agreement of a declaration, and claims nothing further.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.core.domain import DOMAIN_DECLARATIONS, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.registry import Registry, restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES, DomainGlossary

ONBOARDING_SCHEMA = "spectral.onboarding.v1"

#: The adapter recipe, in the order an author works through it. Rendered by
#: ``GET /api/v1/findings/domains`` and asserted by `test_domain_onboarding.py`, so the
#: documented contract and the enforced one are the same tuple rather than two lists that drift.
REQUIRED_DECLARATIONS: Tuple[Tuple[str, str], ...] = (
    ("axes", "declared axis roles, never inferred from an axis name (standard E14)"),
    ("geometry", "a registered grid geometry, or none with `no_physical_metric` declared (E13)"),
    ("lag_policy", "how this domain justifies calling any lag admissible (rule R21)"),
    ("violations", "which inherited assumptions this domain breaks (standard E15, rule R17)"),
    ("licence", "the source's licence or terms of use, required rather than optional"),
    ("provenance", "who declared this domain, and on what basis"),
    ("glossary", "this domain's wording for the closed structural vocabulary (TG7.4)"),
)

#: Names only, for callers that want the checklist without the explanations.
REQUIRED_NAMES: Tuple[str, ...] = tuple(name for name, _why in REQUIRED_DECLARATIONS)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class OnboardingRefused(InvalidParameterError):
    """A domain was offered for onboarding and the contract refused it.

    Subclassed rather than raised as a bare `InvalidParameterError` so a caller assembling
    several domains can distinguish *this domain's declaration disagrees with itself* from the
    lower-level refusals `DomainDeclaration` and `DomainGlossary` raise on their own account.
    """

    def __init__(self, domain: str, parameter: str, value: Any, expected: str,
                 **context: Any) -> None:
        super().__init__(parameter, value, expected, domain=domain, **context)
        self.domain = domain


# --------------------------------------------------------------------------------------------
# The geometry/violation biconditional
# --------------------------------------------------------------------------------------------


def geometry_offers_metric(geometry: Optional[str]) -> bool:
    """Whether a named geometry supplies lengths in metres.

    Asked of the geometry's declared ``physical_metric`` capability rather than of its name
    (standard E2), so a fourth geometry registered from outside `src/` answers for itself. A
    geometry of ``None`` offers no metric, which is the honest reading of supplying none.
    """
    if geometry is None:
        return False
    from src.physical_core.geometry import capability, geometry_for

    geometry_for(geometry)          # `UnknownNameError` naming the registered geometries.
    return bool(capability(geometry, "physical_metric", False))


def assert_geometry_agrees(declaration: DomainDeclaration,
                           geometry: Optional[str]) -> None:
    """Refuse a declaration whose geometry and violations contradict each other (E13/E15).

    Neither `DomainDeclaration` nor the geometry registry can perform this check alone: the
    declaration does not name a geometry and the geometry does not know which domain named it.
    The contract is the only place the two facts meet, which is the reason this module exists
    as something more than a convenience wrapper over two `Registry.add` calls.
    """
    declares_none = "no_physical_metric" in tuple(declaration.violations)
    offers = geometry_offers_metric(geometry)

    if offers and declares_none:
        raise OnboardingRefused(
            declaration.name, "onboarding.geometry", geometry,
            "either a geometry with no physical metric or a declaration without "
            "'no_physical_metric'. Geometry %r declares physical_metric=True, so lengths in "
            "metres are available, while the domain declares that they are not. Downstream "
            "code that asks the geometry would be told metres and code that asks the "
            "declaration would be refused, and nothing reconciles the two answers"
            % geometry,
            violations=list(declaration.violations))

    if not offers and not declares_none:
        supplied = "no geometry" if geometry is None else ("geometry %r" % geometry)
        raise OnboardingRefused(
            declaration.name, "onboarding.violations", list(declaration.violations),
            "'no_physical_metric' among the declared violations, or a geometry that supplies "
            "one. The domain supplies %s, so nothing here can convert an axis step into a "
            "length, yet the declaration does not renounce the metric the analysis layer "
            "inherited. Anything reported per metre from this domain would be a number with "
            "no basis rather than a refusal (standard E15, rule R17)" % supplied,
            geometry=geometry)


# --------------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OnboardedDomain:
    """One completed contract: what was declared, by whom, and the digest that binds it."""

    declaration: DomainDeclaration
    glossary: DomainGlossary
    geometry: Optional[str]
    onboarded_by: str

    @property
    def name(self) -> str:
        return self.declaration.name

    def checklist(self) -> List[Dict[str, Any]]:
        """The recipe beside what satisfied each item, generated from `REQUIRED_DECLARATIONS`.

        Generated rather than written out, so an eighth requirement appears in every rendering
        of the contract the moment it is added to the tuple. A checklist maintained by hand is
        a document that goes stale, which is the failure the documentation audit exists to
        catch elsewhere in this codebase.
        """
        declaration = self.declaration
        satisfied: Dict[str, Any] = {
            "axes": [axis.name for axis in declaration.axes],
            "geometry": self.geometry,
            "lag_policy": declaration.lag_policy,
            "violations": list(declaration.violations),
            "licence": declaration.licence,
            "provenance": dict(declaration.provenance),
            "glossary": self.glossary.glossary_sha256,
        }
        return [{"requirement": name, "why": why, "declared": satisfied[name]}
                for name, why in REQUIRED_DECLARATIONS]

    def describe(self) -> Dict[str, Any]:
        """The record that travels with the domain, and the input to its digest."""
        return {
            "schema": ONBOARDING_SCHEMA,
            "name": self.name,
            "geometry": self.geometry,
            "geometry_offers_physical_metric": geometry_offers_metric(self.geometry),
            "onboarded_by": self.onboarded_by,
            "declaration": self.declaration.describe(),
            "glossary_sha256": self.glossary.glossary_sha256,
            "glossary_term_count": len(self.glossary.phrases),
            "checklist": self.checklist(),
        }

    @property
    def onboarding_sha256(self) -> str:
        """Binds the declaration, the wording and the geometry into one citable digest.

        A result attributed to a domain can then name the exact contract it was read under,
        rather than a domain name that may since have been re-registered with different words.
        """
        return _digest(self.describe())


#: Completed onboarding contracts (standard E1). Distinct from `DOMAIN_DECLARATIONS` on purpose:
#: membership here means *the whole recipe was satisfied in one atomic call*, which is a stronger
#: statement than "a declaration exists", and `audit_onboarding` reports the difference rather
#: than letting a piecemeal domain pass for a complete one.
DOMAIN_ONBOARDINGS: Registry[OnboardedDomain] = Registry("onboarded domain")


def onboarding_for(name: str) -> OnboardedDomain:
    return DOMAIN_ONBOARDINGS.get(name)


def is_onboarded(name: str) -> bool:
    return name in DOMAIN_ONBOARDINGS


# --------------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------------


def _calling_module() -> str:
    """The module that called `onboard_domain`, for attribution.

    `Registry`'s `Entry.defined_in` records ``value.__module__``, which for a `DomainGlossary`
    instance is always `src.core.translation` — the class's home, never the adapter's. That
    answers the wrong question for TG8.1, whose whole acceptance criterion is *which file
    onboarded this domain, and was it inside `src/`*. So the caller is captured here instead.
    """
    frame = inspect.currentframe()
    try:
        outer = frame.f_back if frame else None
        while outer is not None and outer.f_globals.get("__name__") == __name__:
            outer = outer.f_back
        return str(outer.f_globals.get("__name__", "unknown")) if outer else "unknown"
    finally:
        del frame


def onboard_domain(declaration: DomainDeclaration,
                   phrases: Mapping[str, str],
                   *,
                   geometry: Optional[str],
                   glossary_description: str = "",
                   onboarded_by: Optional[str] = None,
                   replace: bool = False) -> OnboardedDomain:
    """Register a complete domain, or leave every registry exactly as it was.

    `geometry` is keyword-only and has **no default**, for the reason ``advection_speed_m_s``
    has none: a default here would be a metric assumption made on the author's behalf, and
    ``None`` has to be a decision rather than an omission. Pass ``None`` explicitly for a domain
    with no spatial extent, and declare ``no_physical_metric`` alongside it.

    Ordering inside the call is deliberate. Every screen that can refuse runs **before** the
    first registry is written, so the rollback path exists for genuinely unexpected failures
    rather than for the ordinary refusals, which never reach it.
    """
    if not isinstance(declaration, DomainDeclaration):
        raise InvalidParameterError(
            "onboard_domain.declaration", type(declaration).__name__,
            "a DomainDeclaration. The declaration validates its own axes, violations and lag "
            "policy on construction, and accepting a mapping here would move those checks to "
            "a second place that could disagree with the first")

    name = declaration.name

    # ---- screens, all of them, before anything is registered -------------------------------
    glossary = DomainGlossary(domain=name, phrases=phrases,
                              description=glossary_description or declaration.description)
    if glossary.domain != name:
        raise OnboardingRefused(
            name, "onboarding.name", name,
            "a domain name that survives `DomainGlossary`'s normalisation unchanged. The "
            "glossary normalised it to %r, so the two registries would be keyed differently "
            "and a lookup that found the wording would miss the limits — the half-onboarded "
            "state in a form no screen further down would notice" % glossary.domain)
    assert_geometry_agrees(declaration, geometry)

    if not replace:
        partial = [label for label, registry in (("glossary", DOMAIN_GLOSSARIES),
                                                 ("declaration", DOMAIN_DECLARATIONS),
                                                 ("onboarding", DOMAIN_ONBOARDINGS))
                   if name in registry]
        if partial:
            raise OnboardingRefused(
                name, "onboarding.name", name,
                "a domain name not already registered. %r is already present as: %s. "
                "Re-onboarding would replace wording or limits that existing results were "
                "read under, so it must be asked for explicitly with replace=True"
                % (name, ", ".join(partial)),
                present=partial)

    attribution = str(onboarded_by).strip() if onboarded_by else _calling_module()
    record = OnboardedDomain(declaration=declaration, glossary=glossary,
                             geometry=geometry, onboarded_by=attribution)

    # ---- write all three, or none of them --------------------------------------------------
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    try:
        DOMAIN_GLOSSARIES.add(
            name, glossary, description=glossary.description,
            capabilities={"onboarded": True, "onboarded_by": attribution},
            replace=replace)
        DOMAIN_DECLARATIONS.add(
            name, declaration, description=declaration.description,
            capabilities={"onboarded": True, "onboarded_by": attribution,
                          "geometry": geometry,
                          "precedence_admissible": declaration.precedence_admissible},
            replace=replace)
        DOMAIN_ONBOARDINGS.add(
            name, record, description=declaration.description,
            capabilities={"geometry": geometry, "onboarded_by": attribution,
                          "violations": list(declaration.violations),
                          "precedence_admissible": declaration.precedence_admissible},
            replace=replace)
    except Exception:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])
        raise
    return record


def audit_onboarding(name: str) -> Dict[str, Any]:
    """What is registered under `name`, and what the contract would still require.

    Reports a domain assembled by separate `Registry.add` calls as **not onboarded**, listing
    what is missing, rather than inferring completeness from whichever pieces happen to be
    present. That distinction is the entire point of a third registry: `DOMAIN_DECLARATIONS`
    can only answer *does a declaration exist*, and the question that matters downstream is
    *was this domain ever checked as a whole*.
    """
    present = {
        "glossary": name in DOMAIN_GLOSSARIES,
        "declaration": name in DOMAIN_DECLARATIONS,
        "onboarding": name in DOMAIN_ONBOARDINGS,
    }
    complete = present["onboarding"]
    missing = sorted(part for part, there in present.items() if not there)
    report: Dict[str, Any] = {
        "name": name,
        "complete": complete,
        "registered": present,
        "missing": missing,
        "required": [{"requirement": item, "why": why} for item, why in REQUIRED_DECLARATIONS],
    }
    if complete:
        record = onboarding_for(name)
        report["onboarded_by"] = record.onboarded_by
        report["onboarding_sha256"] = record.onboarding_sha256
        report["geometry"] = record.geometry
    elif present["glossary"] or present["declaration"]:
        report["note"] = (
            "registered piecemeal rather than through `onboard_domain`, so the geometry and "
            "violations were never checked against each other and no contract digest exists")
    return report


def onboarded_names() -> Tuple[str, ...]:
    return tuple(DOMAIN_ONBOARDINGS.names())


__all__ = ["DOMAIN_ONBOARDINGS", "ONBOARDING_SCHEMA", "OnboardedDomain", "OnboardingRefused",
           "REQUIRED_DECLARATIONS", "REQUIRED_NAMES", "assert_geometry_agrees",
           "audit_onboarding", "geometry_offers_metric", "is_onboarded", "onboard_domain",
           "onboarded_names", "onboarding_for"]
