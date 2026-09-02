"""TG7.4: translation, bounded.

Every layer beneath this one speaks in the programme's own vocabulary — rungs, gates,
categories, statuses, digests.  A domain scientist reading ``candidate_precursor, blocked by
robust_association.confounders_addressed`` learns nothing.  This module renders a finding in
domain language for a reader, and it is the first point in the programme where text is produced
for a human to act on.

That is exactly why it is the point where four rules break at once if nothing stops them.
Domain prose reaches naturally for the semantic comparison R19 forbids; a bare confidence
figure is the way R9 says this platform is most likely to mislead its own author; causal verbs
enter through sentences rather than through claim kinds (R7); and "candidate precursor" becomes
"early warning signal" — a promotion carried out entirely in wording, with no gate touched
(R22).

**Translation is a projection, not a generation.**  There is no model call here.  Domain wording
arrives as declared, content-hashed data that is screened when it is registered, and the
renderer emits only from closed template sets bound to structural facts.  That inherits TG7.1's
design: the dangerous thing is *refused rather than recorded*, at the boundary, once.

Three of the four rules are therefore structural rather than checked:

*   **R9** — only :meth:`AssociationFigures.render` can format a percentage, and
    ``AssociationFigures`` cannot be constructed without all six of R9's figures.  There is no
    code path in this module that produces a lone confidence.
*   **R19** — two disjoint template sets, selected by ``semantic_key`` equality.  The
    cross-domain set may reference only ``structural_signature()``.  A cross-domain magnitude
    sentence is not caught after the fact; it cannot be constructed.
*   **R22** — :func:`translate` takes a ``FiveOutputs``, never a ``ReviewedBundle``.  Review
    commentary has no path in, because the signature does not admit one.

The remaining checks exist as backstops that catch an edit to this module rather than a bad
input, which is what makes them worth mutation-testing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.core.claim_ladder import CLAIM_RUNGS, OUTSIDE_THE_LADDER
from src.core.errors import InvalidParameterError
from src.core.evidence import EVIDENCE_FIELDS, EVIDENCE_STATUSES, EvidenceBundle
from src.core.five_outputs import (
    FiveOutputs,
    RUNG_ENTITLEMENTS,
    STRUCTURAL_ALTERNATIVES,
    summarise_evidence,
)
from src.core.recorded_call import SMUGGLING_FLOOR
from src.core.registry import Registry


TRANSLATION_SCHEMA = "translation/v1"
GLOSSARY_SCHEMA = "domain-glossary/v1"

#: The ladder's gates, restated here rather than imported, because ``claim_ladder._GATES`` is
#: private and this module may not modify it (R22).  A restatement that drifts is worse than no
#: restatement, so ``test_translation`` asserts this tuple equals the gate names the ladder
#: actually produces: adding a gate to the ladder fails that test loudly rather than silently
#: leaving a term with no domain wording.
GATE_NAMES = (
    "observation.no_failed_or_invalid_evidence",
    "observation.no_standing_contradiction",
    "association.observation_recorded",
    "association.effect_size_estimated",
    "association.uncertainty_quantified",
    "robust_association.replicated",
    "robust_association.confounders_addressed",
    "candidate_precursor.provenance_auditable",
    "candidate_precursor.temporal_precedence_recorded",
    "demonstrated_predictive_utility.out_of_sample",
)

#: The eight structural alternatives, taken from the module that defines them so the two cannot
#: disagree.
ALTERNATIVE_NAMES = tuple(sorted(
    identifier for identifier, _description, _closes in STRUCTURAL_ALTERNATIVES.values()))

#: Every structural term a glossary must give domain wording for.  Closed, and total: a glossary
#: missing one is refused at registration rather than falling back to programme jargon in the
#: middle of a domain sentence.
STRUCTURAL_VOCABULARY = tuple(sorted(
    tuple("rung.%s" % rung for rung in CLAIM_RUNGS)
    + tuple("gate.%s" % gate for gate in GATE_NAMES)
    + tuple("alternative.%s" % name for name in ALTERNATIVE_NAMES)
    + tuple("status.%s" % status for status in EVIDENCE_STATUSES)
    + tuple("category.%s" % category for category in EVIDENCE_FIELDS)))

#: Comparative wording that asserts one quantity stands in a relation of size to another.  A
#: glossary phrase carrying one of these would let R19's forbidden comparison in through the
#: vocabulary rather than through a template, which is the one route the template partition
#: cannot close.
COMPARATIVE_VOCABULARY = (
    "stronger than", "weaker than", "larger than", "smaller than", "greater than",
    "less than", "bigger than", "higher than", "lower than", "more than", "twice",
    "half as", "times as", "compared to", "relative to", "outweighs", "exceeds",
)

#: Wording reserved to a rung, in ladder order.  A glossary phrase for one rung may not carry
#: wording reserved to a higher one: that is promotion by vocabulary, and it is the failure this
#: whole module exists to prevent.
RESERVED_WORDING = {
    "observation": ("observed", "recorded", "measured"),
    "association": ("association", "associated", "relationship"),
    "robust_association": ("robust", "replicated", "reproduced"),
    "candidate_precursor": ("precursor", "precedes", "leads", "early warning", "signal",
                            "forerunner", "harbinger"),
    "demonstrated_predictive_utility": ("predictive", "predicts", "prediction", "forecast",
                                        "forecasts", "anticipates"),
}

_DIGIT = re.compile(r"\d")
_NUMERAL = re.compile(r"-?\d+(?:\.\d+)?")

_FIGURE_FIELDS = ("support", "confidence", "base_rate", "lift", "lift_interval",
                  "surrogate_corrected_lift")

#: Lift is checked against confidence / base_rate rather than trusted.  The tolerance is for
#: float round-tripping through JSON, not for disagreement.
_LIFT_TOLERANCE = 1e-9


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(name, value, "a non-empty string")
    return value.strip()


def _fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameterError(name, value, "a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise InvalidParameterError(name, value, "a probability in [0, 1]")
    return number


def _phrases(line: str) -> Tuple[str, ...]:
    """Sentence-sized fragments of a commentary line, at the TG7.1 smuggling floor.

    ``SMUGGLING_FLOOR`` is imported rather than restated so the two checks agree on what counts
    as long enough to be an argument rather than a coincidence.
    """
    fragments = [fragment.strip() for fragment in re.split(r"[.;:]\s*", line)]
    return tuple(fragment for fragment in fragments if len(fragment) >= SMUGGLING_FLOOR)


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameterError(name, value, "a number")
    number = float(value)
    if not number > 0.0:
        raise InvalidParameterError(name, value, "a positive number")
    return number


# --------------------------------------------------------------------------------------------
# R9 given a structure: the six figures, or none of them
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssociationFigures:
    """R9's six figures, kept together because they are separated exactly when it matters.

    The roadmap's example is the whole argument: *"when configuration X occurs, there is an 82%
    historical association with pattern Y developing later"* is not a finding.  If Y occurs 80%
    of the time regardless, 82% is noise dressed as insight.  So confidence is meaningless
    without its base rate, and lift is meaningless without an interval and a surrogate
    comparison.

    Before this class those six numbers had no structured home anywhere in ``src``.  The ladder
    asks only whether an ``effect_sizes`` entry *passes*; it never reads what is inside one, so
    the figures lived unvalidated in a payload mapping.  This is the first slice that must
    actually render them, so it is the slice that has to define the contract.

    ``render`` is the **only** method in this module that formats a percentage.  Nothing else
    can produce one, which is how R9's "a bare confidence must not be renderable" becomes a
    property of the code rather than a rule someone has to remember.
    """

    support: int
    confidence: float
    base_rate: float
    lift: float
    lift_interval: Tuple[float, float]
    surrogate_corrected_lift: float

    def __post_init__(self) -> None:
        if isinstance(self.support, bool) or not isinstance(self.support, int) \
                or self.support < 1:
            raise InvalidParameterError(
                "AssociationFigures.support", self.support,
                "a positive count of occurrences. An association supported by nothing is not an "
                "association")
        object.__setattr__(self, "confidence", _fraction(
            "AssociationFigures.confidence", self.confidence))
        base = _fraction("AssociationFigures.base_rate", self.base_rate)
        if base == 0.0:
            raise InvalidParameterError(
                "AssociationFigures.base_rate", self.base_rate,
                "a non-zero base rate. Lift against a base rate of zero is not a number, and a "
                "confidence with no base rate to beat is exactly what R9 refuses")
        object.__setattr__(self, "base_rate", base)
        interval = tuple(self.lift_interval)
        if len(interval) != 2:
            raise InvalidParameterError("AssociationFigures.lift_interval", self.lift_interval,
                                        "a (low, high) pair")
        low = _positive("AssociationFigures.lift_interval[0]", interval[0])
        high = _positive("AssociationFigures.lift_interval[1]", interval[1])
        if low > high:
            raise InvalidParameterError("AssociationFigures.lift_interval", interval,
                                        "an interval whose low bound does not exceed its high")
        object.__setattr__(self, "lift_interval", (low, high))
        lift = _positive("AssociationFigures.lift", self.lift)
        expected = self.confidence / base
        if abs(lift - expected) > _LIFT_TOLERANCE:
            raise InvalidParameterError(
                "AssociationFigures.lift", self.lift,
                "confidence over base rate, which is %.12g. A lift that is not the ratio it "
                "claims to be is the one figure a reader cannot check for themselves" % expected)
        object.__setattr__(self, "lift", lift)
        object.__setattr__(self, "surrogate_corrected_lift", _positive(
            "AssociationFigures.surrogate_corrected_lift", self.surrogate_corrected_lift))

    @property
    def beats_base_rate(self) -> bool:
        """Whether the interval excludes 1.  A lift of 1.2 spanning 0.8 to 1.9 has not."""
        return self.lift_interval[0] > 1.0

    @property
    def survives_surrogate(self) -> bool:
        """Whether the surrogate-corrected lift still exceeds 1 (R1)."""
        return self.surrogate_corrected_lift > 1.0

    def to_mapping(self) -> Dict[str, Any]:
        return {"support": self.support, "confidence": self.confidence,
                "base_rate": self.base_rate, "lift": self.lift,
                "lift_interval": list(self.lift_interval),
                "surrogate_corrected_lift": self.surrogate_corrected_lift}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AssociationFigures":
        """Read the six figures from an ``effect_sizes`` payload, refusing a partial set by name.

        An author who recorded a bare confidence is refused here, at the point of translation,
        rather than discovering it as a large friendly percentage on a dashboard.
        """
        if not isinstance(payload, Mapping):
            raise InvalidParameterError("effect_sizes payload", type(payload).__name__,
                                        "a JSON object")
        missing = [name for name in _FIGURE_FIELDS if name not in payload]
        if missing:
            raise InvalidParameterError(
                "effect_sizes payload", sorted(payload),
                "all six of R9's figures. Missing: %s. Confidence without a base rate is not a "
                "weaker finding, it is not a finding" % ", ".join(missing))
        interval = payload["lift_interval"]
        if not isinstance(interval, (list, tuple)):
            raise InvalidParameterError("AssociationFigures.lift_interval", interval,
                                        "a (low, high) pair")
        return cls(support=payload["support"], confidence=payload["confidence"],
                   base_rate=payload["base_rate"], lift=payload["lift"],
                   lift_interval=(interval[0], interval[1]) if len(interval) == 2 else tuple(
                       interval),
                   surrogate_corrected_lift=payload["surrogate_corrected_lift"])

    def numerals(self) -> Tuple[str, ...]:
        """Every numeral this block is entitled to put in front of a reader.

        The R19 guard uses this as the whitelist for rendered numbers, so a figure that was
        never recorded cannot appear in a sentence.
        """
        return tuple(_NUMERAL.findall(self.render()))

    def render(self) -> str:
        """The only percentage in this module, and it never travels alone.

        Support, base rate, lift, the interval and the surrogate comparison are one string
        rather than five adjacent ones, so no caller can render the confidence and drop the
        context that makes it mean anything (R9).
        """
        return ("observed in %d occurrences: %.1f%% of the time, against a base rate of %.1f%% "
                "(lift %.2f, interval %.2f to %.2f, %.2f after surrogate correction)"
                % (self.support, 100.0 * self.confidence, 100.0 * self.base_rate,
                   self.lift, self.lift_interval[0], self.lift_interval[1],
                   self.surrogate_corrected_lift))


# --------------------------------------------------------------------------------------------
# The glossary: domain wording as declared, screened, hashed data
# --------------------------------------------------------------------------------------------


def _screen(term: str, phrase: str) -> None:
    """Everything a domain phrase may not contain, refused when it is declared.

    Registration is the right moment for all four of these.  A glossary is written once and
    read on every finding, so a check here runs once per glossary rather than once per sentence,
    and — more to the point — it names the offending phrase to the person who wrote it instead
    of to a reader who cannot fix it.
    """
    lowered = phrase.lower()
    for word in OUTSIDE_THE_LADDER:
        if re.search(r"\b%s\b" % re.escape(word), lowered):
            raise InvalidParameterError(
                "glossary[%s]" % term, phrase,
                "domain wording free of causal vocabulary. %r is outside the ladder at every "
                "rung, and a glossary is not a way back in (R7)" % word)
    if _DIGIT.search(phrase):
        raise InvalidParameterError(
            "glossary[%s]" % term, phrase,
            "domain wording without digits. Every number a reader sees must come from the "
            "record, not from the vocabulary used to describe it (R19)")
    for comparative in COMPARATIVE_VOCABULARY:
        if comparative in lowered:
            raise InvalidParameterError(
                "glossary[%s]" % term, phrase,
                "domain wording that does not compare quantities. %r asserts a relation of size "
                "that the structural evidence does not support (R19)" % comparative)


def _screen_promotion(term: str, phrase: str) -> None:
    """A rung's phrase may not borrow wording reserved to a higher rung.

    This is the failure that motivates the whole module: nothing in the gates moves, no digest
    changes, and the reader is told the finding is an early warning signal when the record says
    it is a candidate precursor.  It is a promotion carried out entirely in vocabulary, so the
    vocabulary is where it has to be refused (R22).
    """
    if not term.startswith("rung."):
        return
    rung = term[len("rung."):]
    if rung not in CLAIM_RUNGS:
        return
    lowered = phrase.lower()
    for higher in CLAIM_RUNGS[CLAIM_RUNGS.index(rung) + 1:]:
        for word in RESERVED_WORDING[higher]:
            if word in lowered:
                raise InvalidParameterError(
                    "glossary[%s]" % term, phrase,
                    "wording for %s that does not borrow from %s. %r is reserved to a rung this "
                    "bundle has not reached, and renaming a rung is not the same as climbing it "
                    "(R22)" % (rung, higher, word))


@dataclass(frozen=True)
class DomainGlossary:
    """One domain's wording for the closed structural vocabulary, declared and hashed.

    A glossary is data an adapter supplies, not code this module contains.  It registers through
    the same :class:`~src.core.registry.Registry` that orientation conventions use, so a new
    domain gets domain-language findings **without editing** ``src`` — the TG8.1 onboarding
    condition, met here rather than promised.

    The map must be total over :data:`STRUCTURAL_VOCABULARY`.  A partial glossary is refused
    because the alternative is worse than a refusal: one untranslated term surfaces as
    programme jargon in the middle of an otherwise fluent domain sentence, which reads as a
    typo rather than as the gap it is.
    """

    domain: str
    phrases: Mapping[str, str]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _nonempty("DomainGlossary.domain", self.domain))
        if not isinstance(self.phrases, Mapping):
            raise InvalidParameterError("DomainGlossary.phrases", type(self.phrases).__name__,
                                        "a mapping of structural term to domain phrase")
        cleaned: Dict[str, str] = {}
        for term, phrase in self.phrases.items():
            name = _nonempty("DomainGlossary.phrases key", term)
            if name not in STRUCTURAL_VOCABULARY:
                raise InvalidParameterError(
                    "glossary[%s]" % name, phrase,
                    "a term of the structural vocabulary. A glossary may reword what the gates "
                    "decided; it may not add a term they never used")
            text = _nonempty("glossary[%s]" % name, phrase)
            _screen(name, text)
            _screen_promotion(name, text)
            cleaned[name] = text
        missing = [term for term in STRUCTURAL_VOCABULARY if term not in cleaned]
        if missing:
            raise InvalidParameterError(
                "DomainGlossary.phrases", "%d of %d terms" % (len(cleaned),
                                                              len(STRUCTURAL_VOCABULARY)),
                "wording for every structural term. Missing: %s. An untranslated term reaches "
                "the reader as programme jargon mid-sentence" % ", ".join(missing[:5]))
        object.__setattr__(self, "phrases", dict(cleaned))
        object.__setattr__(self, "description",
                           "" if not self.description else str(self.description).strip())

    def say(self, term: str) -> str:
        """This domain's wording for one structural term."""
        if term not in self.phrases:
            raise InvalidParameterError("term", term,
                                        "one of the %d structural terms this glossary declares"
                                        % len(self.phrases))
        return self.phrases[term]

    def to_mapping(self) -> Dict[str, Any]:
        return {"schema": GLOSSARY_SCHEMA, "domain": self.domain,
                "phrases": dict(self.phrases), "description": self.description}

    @property
    def glossary_sha256(self) -> str:
        """Binds the whole map, so a translation names the exact wording that produced it."""
        return _digest(self.to_mapping())


#: Domain glossaries (standard E1).  A domain registers one from outside ``src`` (TG8.1).
DOMAIN_GLOSSARIES: Registry[DomainGlossary] = Registry("domain glossary")


def glossary_for(name: str) -> DomainGlossary:
    return DOMAIN_GLOSSARIES.get(name)


# --------------------------------------------------------------------------------------------
# Two disjoint template sets: R19 held by construction rather than by inspection
# --------------------------------------------------------------------------------------------

#: Fields a **within-domain** sentence may reference.  These carry units, magnitude and the
#: variable's identity, which is exactly why they may never cross a domain boundary.
WITHIN_DOMAIN_FIELDS = ("variable", "dataset", "units", "magnitude", "location", "time")

#: Fields a **cross-domain** sentence may reference.  This tuple is the whole of
#: ``SpectralFeature.structural_signature`` — the dimensionless view, built from the fields that
#: survive being stripped of their units rather than filtered out of ``describe()``.  Adding a
#: unit-bearing field here is a visible edit to a named constant, which is the point.
CROSS_DOMAIN_FIELDS = ("representation", "has_spatial_scale", "has_temporal_scale",
                       "orientation_convention", "orientation_degrees", "significance",
                       "significance_basis", "extent_in_scales")


def _number(value: Any) -> str:
    """Format a number exactly as canonical JSON would, so numeral containment is verbatim."""
    return json.dumps(value, allow_nan=False)


def _signature_clause(feature: Any) -> str:
    signature = feature.structural_signature()
    parts = ["described in the %s representation" % signature["representation"]]
    if signature["orientation_degrees"] is not None:
        parts.append("oriented at %s degrees under %s"
                     % (_number(signature["orientation_degrees"]),
                        signature["orientation_convention"]))
    if signature["extent_in_scales"] is not None:
        parts.append("extending over %s of its own scales"
                     % _number(signature["extent_in_scales"]))
    if signature["significance"] is not None:
        parts.append("at significance %s on the %s basis"
                     % (_number(signature["significance"]), signature["significance_basis"]))
    return ", ".join(parts)


def render_comparison(left: Any, right: Any) -> str:
    """Describe two features together, choosing the template set by ``semantic_key`` equality.

    This is the load-bearing line of the module.  Two features of one variable of one dataset of
    one domain may be described with their magnitudes and units, because a difference between
    them is a quantity about the world.  Two features from different domains may be described
    only through ``structural_signature`` — the dimensionless view — since a sentence relating
    their magnitudes would be a sentence about two arrays rather than about anything.

    R19 therefore holds because the cross-domain sentence has no access to a unit-bearing field,
    not because a checker read the sentence afterwards.  The guard below is a backstop that
    catches an edit to *this function*; it is not the mechanism.
    """
    if left.semantic_key == right.semantic_key:
        difference = left.compare_magnitude_to(right)
        return ("both are %s in %s: they differ by %s"
                % (left.variable, left.dataset, difference.describe_short()))
    return ("no quantity of %s is comparable with one of %s, so only their structure is "
            "described: the first is %s; the second is %s"
            % (left.domain, right.domain, _signature_clause(left), _signature_clause(right)))


# --------------------------------------------------------------------------------------------
# The translated document
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationUnit:
    """One structural fact, its domain wording, and what that wording does not license.

    ``licences`` is not a footnote and is not adjacent to ``rendered``: :attr:`text` welds them
    into one string, so a caller cannot render "this is a candidate precursor" while dropping
    "predictive utility is not shown".  R9 makes an equivalent demand of the frontend — a bare
    confidence must not be a renderable component — and this is the same constraint applied to
    prose.
    """

    structural_key: str
    source_sha256: str
    rendered: str
    licences: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_key",
                           _nonempty("TranslationUnit.structural_key", self.structural_key))
        object.__setattr__(self, "source_sha256",
                           _nonempty("TranslationUnit.source_sha256", self.source_sha256))
        object.__setattr__(self, "rendered",
                           _nonempty("TranslationUnit.rendered", self.rendered))
        object.__setattr__(self, "licences", _nonempty(
            "TranslationUnit.licences", self.licences))

    @property
    def text(self) -> str:
        """The wording and its bound, as one string.  There is no way to render half of it."""
        return "%s - %s" % (self.rendered, self.licences)

    def to_mapping(self) -> Dict[str, Any]:
        return {"structural_key": self.structural_key, "source_sha256": self.source_sha256,
                "rendered": self.rendered, "licences": self.licences}


@dataclass(frozen=True)
class TranslatedFinding:
    """A finding in one domain's words, bound to the exact revision and glossary behind it."""

    schema: str
    study_id: str
    bundle_sha256: str
    revision: int
    summary_sha256: str
    domain: str
    glossary_sha256: str
    claimable: Tuple[TranslationUnit, ...]
    not_claimable: Tuple[TranslationUnit, ...]
    contradicting: Tuple[TranslationUnit, ...]
    alternatives: Tuple[TranslationUnit, ...]
    next_observation: Optional[TranslationUnit]
    figures: Optional[AssociationFigures]
    commentary: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("claimable", "not_claimable", "contradicting", "alternatives",
                     "commentary"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def units(self) -> Tuple[TranslationUnit, ...]:
        ordered = list(self.claimable) + list(self.not_claimable) + list(self.contradicting) \
            + list(self.alternatives)
        if self.next_observation is not None:
            ordered.append(self.next_observation)
        return tuple(ordered)

    @property
    def structural_keys(self) -> Tuple[str, ...]:
        """The facts this document states, independent of the words used to state them.

        The acceptance test turns on this: translating one bundle into three glossaries must
        give three different documents with one identical key set.
        """
        return tuple(unit.structural_key for unit in self.units)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "study_id": self.study_id,
            "bundle_sha256": self.bundle_sha256, "revision": self.revision,
            "summary_sha256": self.summary_sha256, "domain": self.domain,
            "glossary_sha256": self.glossary_sha256,
            "claimable": [unit.to_mapping() for unit in self.claimable],
            "not_claimable": [unit.to_mapping() for unit in self.not_claimable],
            "contradicting": [unit.to_mapping() for unit in self.contradicting],
            "alternatives": [unit.to_mapping() for unit in self.alternatives],
            "next_observation": (None if self.next_observation is None
                                 else self.next_observation.to_mapping()),
            "figures": None if self.figures is None else self.figures.to_mapping(),
            "commentary": list(self.commentary),
        }

    @property
    def translation_sha256(self) -> str:
        """Binds the document to the bundle revision *and* the glossary that worded it."""
        return _digest(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TranslatedFinding":
        def unit(item: Mapping[str, Any]) -> TranslationUnit:
            return TranslationUnit(structural_key=item["structural_key"],
                                   source_sha256=item["source_sha256"],
                                   rendered=item["rendered"], licences=item["licences"])
        following = value.get("next_observation")
        figures = value.get("figures")
        return cls(schema=value["schema"], study_id=value["study_id"],
                   bundle_sha256=value["bundle_sha256"], revision=value["revision"],
                   summary_sha256=value["summary_sha256"], domain=value["domain"],
                   glossary_sha256=value["glossary_sha256"],
                   claimable=tuple(unit(i) for i in value["claimable"]),
                   not_claimable=tuple(unit(i) for i in value["not_claimable"]),
                   contradicting=tuple(unit(i) for i in value["contradicting"]),
                   alternatives=tuple(unit(i) for i in value["alternatives"]),
                   next_observation=None if following is None else unit(following),
                   figures=None if figures is None else AssociationFigures(
                       support=figures["support"], confidence=figures["confidence"],
                       base_rate=figures["base_rate"], lift=figures["lift"],
                       lift_interval=tuple(figures["lift_interval"]),
                       surrogate_corrected_lift=figures["surrogate_corrected_lift"]),
                   commentary=tuple(value["commentary"]))

    def render(self) -> str:
        """The document as a reader receives it, in the five outputs' fixed order."""
        lines = ["%s, revision %d (%s)" % (self.study_id, self.revision, self.domain),
                 "", "What can be said:"]
        for unit in self.claimable:
            lines.append("   %s" % unit.text)
        if self.figures is not None:
            lines.append("   %s" % self.figures.render())
        lines.append("What cannot be said:")
        for unit in self.not_claimable:
            lines.append("   %s" % unit.text)
        lines.append("What argues against it:")
        for unit in self.contradicting:
            lines.append("   %s" % unit.text)
        if not self.contradicting:
            lines.append("   none recorded; that is not the same as none existing")
        lines.append("What else could explain it:")
        for unit in self.alternatives:
            lines.append("   %s" % unit.text)
        if not self.alternatives:
            lines.append("   none that this record names; that is not the same as none existing")
        lines.append("What to look at next:")
        lines.append("   %s" % ("none identified; that is not the same as none existing"
                                if self.next_observation is None
                                else self.next_observation.text))
        if self.commentary:
            lines.extend(["", "Commentary, which moved nothing above:"])
            lines.extend("   %s" % line for line in self.commentary)
        return "\n".join(lines)

    def claim_text(self) -> str:
        """Everything the document asserts, with quarantined commentary excluded.

        The guards run over this rather than over :meth:`render`.  Commentary is recorded
        adversarial argument (R23) and may legitimately contain the word "causal" — a
        challenger saying a claim is *not* causal is doing its job.  Refusing that would refuse
        a good review, so commentary is fenced off and labelled instead of scanned.
        """
        parts = [unit.text for unit in self.units]
        if self.figures is not None:
            parts.append(self.figures.render())
        return "\n".join(parts)


# --------------------------------------------------------------------------------------------
# Translation: a pure function of the five outputs and one glossary
# --------------------------------------------------------------------------------------------


def _entitlement_of(rung: str) -> str:
    return RUNG_ENTITLEMENTS[rung]


def _claimable_units(outputs: FiveOutputs, glossary: DomainGlossary,
                     anchor: str) -> Tuple[TranslationUnit, ...]:
    return tuple(
        TranslationUnit(structural_key="rung.%s" % rung, source_sha256=anchor,
                        rendered=glossary.say("rung.%s" % rung),
                        licences=_entitlement_of(rung))
        for rung in outputs.claimable)


def _not_claimable_units(outputs: FiveOutputs, glossary: DomainGlossary,
                         anchor: str) -> Tuple[TranslationUnit, ...]:
    units = []
    for unreachable in outputs.not_claimable:
        blockers = ", ".join(glossary.say("gate.%s" % gate) for gate in unreachable.blocked_by)
        units.append(TranslationUnit(
            structural_key="rung.%s" % unreachable.rung, source_sha256=anchor,
            rendered="not yet %s; blocked by %s" % (glossary.say("rung.%s" % unreachable.rung),
                                                    blockers),
            licences=unreachable.entitlement))
    units.append(TranslationUnit(
        structural_key="outside_the_ladder", source_sha256=anchor,
        rendered=("no reading of this record as one thing bringing about another is licensed "
                  "at any rung"),
        licences="This programme provides no framework that could license one (R7)."))
    return tuple(units)


def _contradicting_units(outputs: FiveOutputs, glossary: DomainGlossary
                         ) -> Tuple[TranslationUnit, ...]:
    units = []
    for entry in outputs.contradicting:
        units.append(TranslationUnit(
            structural_key="entry.%d" % entry.sequence, source_sha256=entry.entry_sha256,
            rendered="%s recorded as %s: %s" % (glossary.say("category.%s" % entry.category),
                                                glossary.say("status.%s" % entry.status),
                                                entry.label),
            licences=("This caps the record at its lowest rung." if entry.caps_at_observation
                      else "This argues against the hypothesis without capping it.")))
    return tuple(units)


def _alternative_units(outputs: FiveOutputs, glossary: DomainGlossary,
                       anchor: str) -> Tuple[TranslationUnit, ...]:
    units = []
    for alternative in outputs.alternatives:
        term = "alternative.%s" % alternative.identifier
        wording = (glossary.say(term) if term in glossary.phrases
                   else alternative.description)
        units.append(TranslationUnit(
            structural_key=term, source_sha256=anchor,
            rendered="%s; closed when %s" % (wording, alternative.closes_when),
            licences=("This alternative is open. That it is listed is not evidence it obtains, "
                      "and that another is absent is not evidence it does not.")))
    return tuple(units)


def _next_unit(outputs: FiveOutputs, anchor: str) -> Optional[TranslationUnit]:
    following = outputs.next_observation
    if following is None:
        return None
    return TranslationUnit(
        structural_key="next_observation.%s" % following.kind, source_sha256=anchor,
        rendered=following.requirement,
        licences=("This is the cheapest thing standing in the way, not the most informative "
                  "experiment. No expected information gain was computed."))


def translate(outputs: FiveOutputs, glossary: DomainGlossary, *,
              features: Sequence[Any] = (),
              figures: Optional[AssociationFigures] = None,
              commentary: Sequence[str] = ()) -> TranslatedFinding:
    """Render the five outputs in one domain's words.

    A pure function of its arguments, like ``summarise_evidence`` beneath it: no clock, no
    filesystem, no environment, no randomness.  The same outputs and the same glossary give the
    same document, byte for byte, to anyone holding them.

    The signature takes a ``FiveOutputs``, **never** a ``ReviewedBundle``.  That is not a
    convenience — it is how R22 is enforced here.  Review commentary cannot reach the claim
    text because there is no parameter through which it could arrive; what a caller passes as
    ``commentary`` is carried in its own quarantined field, excluded from
    :meth:`TranslatedFinding.claim_text`, and rendered under a heading saying it moved nothing.
    """
    if not isinstance(outputs, FiveOutputs):
        raise InvalidParameterError("outputs", type(outputs).__name__, "a FiveOutputs")
    if not isinstance(glossary, DomainGlossary):
        raise InvalidParameterError("glossary", type(glossary).__name__, "a DomainGlossary")
    if figures is not None and not isinstance(figures, AssociationFigures):
        raise InvalidParameterError(
            "figures", type(figures).__name__,
            "an AssociationFigures. R9's six figures travel together or not at all")
    anchor = outputs.summary_sha256
    document = TranslatedFinding(
        schema=TRANSLATION_SCHEMA, study_id=outputs.study_id,
        bundle_sha256=outputs.bundle_sha256, revision=outputs.revision,
        summary_sha256=anchor, domain=glossary.domain,
        glossary_sha256=glossary.glossary_sha256,
        claimable=_claimable_units(outputs, glossary, anchor),
        not_claimable=_not_claimable_units(outputs, glossary, anchor),
        contradicting=_contradicting_units(outputs, glossary),
        alternatives=_alternative_units(outputs, glossary, anchor),
        next_observation=_next_unit(outputs, anchor),
        figures=figures,
        commentary=tuple(_nonempty("commentary", line) for line in commentary))
    assert_no_causal_language(document)
    assert_no_bare_confidence(document)
    assert_no_semantic_comparison(document, outputs, features=features)
    return document


# --------------------------------------------------------------------------------------------
# The guards.  Each catches an edit to this module rather than a bad input, which is what makes
# them worth mutating.
# --------------------------------------------------------------------------------------------


def assert_no_causal_language(document: TranslatedFinding) -> None:
    """Refuse claim text carrying a term the ladder places outside itself at every rung (R7).

    The vocabulary is imported from ``claim_ladder`` rather than restated, so there is one
    definition of what this programme will not say.  Glossaries are screened at registration and
    templates are fixed, so a hit here means a template was edited or an author put a causal
    word in an evidence label — and both are worth stopping before a reader sees them.

    What is scanned is each unit's ``rendered`` half, not its ``licences``, and it is scanned
    with its punctuation flattened to spaces so that an identifier is screened the way prose is
    (D89).  What remains uncaught is a word run together with another without any separator —
    ``co2causeswarming`` — and that limit is stated rather than replaced by a substring match,
    which would refuse "causeway" and every other innocent word containing one of these.

    The entitlements are the programme's own fixed sentences, and several of them name a causal
    term precisely in order to refuse it — *"a statement about prediction, never about
    mechanism"*.  Scanning those would refuse the sentence whose whole purpose is to hold the
    line, which is why the curated half and the authored half are scanned differently rather
    than together.
    """
    for unit in document.units:
        # Punctuation is flattened to spaces before the word boundaries are applied (D89).
        # An underscore is a word character, so ``causes`` does not match inside
        # ``co2_causes_warming`` — and an identifier is exactly the shape of the author-supplied
        # text that reaches ``rendered`` through an entry label or an alternative's wording.
        lowered = re.sub(r"[^a-z0-9]+", " ", unit.rendered.lower())
        for word in OUTSIDE_THE_LADDER:
            if re.search(r"\b%s\b" % re.escape(word), lowered):
                raise InvalidParameterError(
                    "translated unit %s" % unit.structural_key, word,
                    "wording free of causal vocabulary. The ladder places %r outside itself at "
                    "every rung, and rendering it for a reader is how that boundary is lost "
                    "(R7)" % word)


def assert_no_bare_confidence(document: TranslatedFinding) -> None:
    """Refuse a percentage anywhere except inside a complete set of R9's six figures.

    A big friendly percentage on its own is, in the roadmap's words, the single most likely way
    this platform ends up misleading its own author.  ``AssociationFigures.render`` is the only
    thing entitled to produce one, and this refuses any other source of one.

    The per-unit loop and the whole-text check are **not** two independent guards: mutation
    testing showed that removing either leaves the other catching every case.  The loop is kept
    for its diagnostics — it names the offending unit, where the second only reports that the
    document as a whole carries a stray percentage — and it is recorded as redundant here so a
    later reader does not mistake belt for braces.
    """
    permitted = "" if document.figures is None else document.figures.render()
    for unit in document.units:
        if "%" in unit.text:
            raise InvalidParameterError(
                "translated unit %s" % unit.structural_key, unit.text,
                "text without a percentage. A confidence is renderable only beside its support, "
                "base rate, lift, interval and surrogate correction (R9)")
    if "%" in document.claim_text().replace(permitted, ""):
        raise InvalidParameterError(
            "translated claim text", document.claim_text(),
            "no percentage outside a complete AssociationFigures block (R9)")


def _source_numerals(outputs: FiveOutputs, features: Sequence[Any],
                     figures: Optional[AssociationFigures]) -> frozenset:
    """Every numeral the record itself contains, independently of what was rendered.

    Deliberately re-derived from the structural sources rather than collected while rendering:
    a whitelist a renderer builds for itself would accept whatever the renderer chose to emit,
    which is not a check.
    """
    allowed = set(_NUMERAL.findall(_canonical(outputs.to_mapping()).decode("utf-8")))
    for feature in features:
        allowed.update(_NUMERAL.findall(_canonical(feature.describe()).decode("utf-8")))
        allowed.update(_NUMERAL.findall(
            _canonical(feature.structural_signature()).decode("utf-8")))
    if figures is not None:
        allowed.update(figures.numerals())
    return frozenset(allowed)


def assert_no_semantic_comparison(document: TranslatedFinding, outputs: FiveOutputs, *,
                                  features: Sequence[Any] = ()) -> None:
    """Refuse a number the record does not contain (R19).

    The template partition in :func:`render_comparison` is the mechanism; this is the backstop.
    A cross-domain sentence that acquired a magnitude — because someone widened
    ``CROSS_DOMAIN_FIELDS``, or reached past the template sets entirely — puts a numeral in
    front of a reader that is not in the source, and that is what this catches.
    """
    allowed = _source_numerals(outputs, features, document.figures)
    for unit in document.units:
        for numeral in _NUMERAL.findall(unit.text):
            if numeral not in allowed:
                raise InvalidParameterError(
                    "translated unit %s" % unit.structural_key, numeral,
                    "a numeral present in the structural record. %r appears in the rendering "
                    "and nowhere in the evidence it claims to describe, which is a magnitude "
                    "the structure does not support (R19)" % numeral)


def assert_structural_only(text: str, features: Sequence[Any]) -> None:
    """Refuse a cross-domain rendering that leaked a unit-bearing field (R19).

    Applied to the output of :func:`render_comparison` for features whose ``semantic_key``
    differs.  What it asserts is absence: the variable, the dataset and the units of either
    feature must not appear, because a sentence carrying them is a sentence comparing
    quantities that have no common measure.
    """
    lowered = text.lower()
    for feature in features:
        for field in ("variable", "dataset"):
            value = str(getattr(feature, field, "") or "")
            if value and value.lower() in lowered:
                raise InvalidParameterError(
                    "cross-domain rendering", value,
                    "structural fields only. The %s reached a sentence describing two domains, "
                    "which is the comparison R19 forbids" % field)
        units = feature.units
        if units and re.search(r"\b%s\b" % re.escape(str(units).lower()), lowered):
            raise InvalidParameterError(
                "cross-domain rendering", units,
                "structural fields only. Units crossed a domain boundary (R19)")


def verify_translation_independence(document: TranslatedFinding,
                                    bundle: EvidenceBundle) -> str:
    """R22 at the reader's edge: refuse unless the translation moved nothing (R22).

    The analogue of ``verify_claim_independence``, and it checks the two things that could
    actually fail here.  The claim state is recomputed from the bundle's own canonical bytes and
    must still be the one the document was bound to — a document naming a summary digest the
    bundle no longer produces is stale, and a stale translation is a claim about a revision that
    no longer exists.  And no phrase of the *quarantined commentary* may appear in the claim
    text, which is the one way argument could have crossed the fence this module puts around it.
    """
    if not isinstance(document, TranslatedFinding):
        raise InvalidParameterError("document", type(document).__name__,
                                    "a TranslatedFinding")
    if not isinstance(bundle, EvidenceBundle):
        raise InvalidParameterError("bundle", type(bundle).__name__, "an EvidenceBundle")
    rebuilt = summarise_evidence(
        EvidenceBundle.from_mapping(bundle.to_mapping())).summary_sha256
    if rebuilt != document.summary_sha256:
        raise InvalidParameterError(
            "summary_sha256", document.summary_sha256,
            "the claim state recomputed from the bundle's own bytes, %s. This translation "
            "describes a revision this bundle no longer produces" % rebuilt)
    claim = document.claim_text()
    for line in document.commentary:
        for phrase in _phrases(line):
            if phrase in claim:
                raise InvalidParameterError(
                    "translated claim text", phrase,
                    "free of commentary. Wording from the recorded review is inside the text "
                    "presented as the finding, where no LLM output may ever be (R22)")
    return document.summary_sha256


# --------------------------------------------------------------------------------------------
# Persistence: beside the bundle, never over it
# --------------------------------------------------------------------------------------------


def save_translation(path: Union[str, "os.PathLike[str]"], document: TranslatedFinding, *,
                     bundle_path: Optional[Union[str, "os.PathLike[str]"]] = None) -> str:
    """Publish a translation in its own file, exclusively, and never over a bundle."""
    if not isinstance(document, TranslatedFinding):
        raise InvalidParameterError("document", type(document).__name__,
                                    "a TranslatedFinding")
    target = Path(path)
    if bundle_path is not None and target.resolve() == Path(bundle_path).resolve():
        raise InvalidParameterError(
            "path", str(target),
            "a file of its own. A translation is published beside the evidence bundle, never "
            "inside it (R22)")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(document.to_mapping()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("refusing to overwrite translation at %s" % target) from None
    return document.translation_sha256


def load_translation(path: Union[str, "os.PathLike[str]"], *,
                     published_sha256: Optional[str] = None) -> TranslatedFinding:
    target = Path(path)
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidParameterError("translation path", str(target),
                                    "readable UTF-8 JSON: %s" % exc) from exc
    document = TranslatedFinding.from_mapping(value)
    if raw != _canonical(document.to_mapping()) + b"\n":
        raise InvalidParameterError("translation bytes", str(target), "canonical JSON")
    if published_sha256 is not None \
            and str(published_sha256).lower() != document.translation_sha256:
        raise InvalidParameterError(
            "published_sha256", published_sha256,
            "the unchanged translation digest %s" % document.translation_sha256)
    return document


__all__ = [
    "TRANSLATION_SCHEMA", "GLOSSARY_SCHEMA", "GATE_NAMES", "ALTERNATIVE_NAMES",
    "STRUCTURAL_VOCABULARY", "COMPARATIVE_VOCABULARY", "RESERVED_WORDING",
    "WITHIN_DOMAIN_FIELDS", "CROSS_DOMAIN_FIELDS",
    "AssociationFigures", "DomainGlossary", "DOMAIN_GLOSSARIES", "glossary_for",
    "TranslationUnit", "TranslatedFinding", "translate", "render_comparison",
    "assert_no_causal_language", "assert_no_bare_confidence", "assert_no_semantic_comparison",
    "assert_structural_only", "verify_translation_independence",
    "save_translation", "load_translation",
]
