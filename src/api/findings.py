"""TG9.1: the read-only claim surface.

The cross-domain line had no HTTP surface at all. Every route in `api/main.py` belongs to the
atmospheric/transform line, and all twelve G-line modules — `domain` through `translation` —
had zero references there, so nothing a browser could reach knew a claim ladder existed. This
module is that surface, and it is **read-only**: nothing here appends evidence, records a call or
moves a rung. A GET cannot change what may be claimed (R22).

**The governing principle of Phase G9: a client computes and formats no scientific number.** So
this surface does not serve parts a caller would have to assemble into a claim. A translation goes
out already rendered. R9's six figures go out whole or not at all, and
:func:`refuse_bare_confidence` walks every response body to enforce that on the wire rather than
trusting each handler to remember — a `confidence` key travelling without its base rate is the
exact failure R9 was written to prevent, and this is the last place it can be caught before a
client sees it.

**Registration is eager, and that is deliberate.** Defect D35 was a fallback chain that depended
on browsing order, because source registration was an import side effect of a module imported
lazily inside its own handler. `DOMAIN_GLOSSARIES` has that shape, so
`register_builtin_glossaries()` is called when this router is built rather than whenever some
handler first happens to import something.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query

from src.core.builtin_domains import register_builtin_domains
from src.core.builtin_glossaries import register_builtin_glossaries
from src.core.domain import DOMAIN_DECLARATIONS, declaration_for, refusals_for
from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle, load_evidence_bundle
from src.core.claim_ladder import CLAIM_RUNGS
from src.core.five_outputs import summarise_evidence
from src.core.onboarding import (ONBOARDING_SCHEMA, REQUIRED_DECLARATIONS,
                                 audit_onboarding, onboarded_names)
from src.core.translation import (
    DOMAIN_GLOSSARIES,
    AssociationFigures,
    glossary_for,
    translate,
)


#: Where published bundles are read from.  One canonical JSON file per study, written by
#: `save_evidence_bundle`, never written by this module.
STUDY_ROOT_ENV = "SPECTRAL_STUDY_ROOT"
DEFAULT_STUDY_ROOT = Path("data") / "studies"

#: The six figures R9 requires to travel together.  Restated from `translation` deliberately:
#: this guard must keep working if someone edits `AssociationFigures`, and a guard that imported
#: its expectations from the thing it is guarding would agree with any change made to it.
R9_FIGURES = ("support", "confidence", "base_rate", "lift", "lift_interval",
              "surrogate_corrected_lift")


def study_root() -> Path:
    """The directory served, overridable so a test never reads a researcher's real studies."""
    return Path(os.environ.get(STUDY_ROOT_ENV) or DEFAULT_STUDY_ROOT)


# --------------------------------------------------------------------------------------------
# The wire guard
# --------------------------------------------------------------------------------------------


def refuse_bare_confidence(payload: Any, *, where: str = "response") -> Any:
    """Refuse any response carrying a confidence without the five figures that give it meaning.

    R9's rule is usually described as a frontend constraint, which puts it in the one place it
    cannot be enforced. This puts it on the wire: whatever a handler intended, a body containing
    ``confidence`` without ``base_rate`` beside it does not leave the process.

    The check is structural rather than textual, so it survives a handler being rewritten, a new
    route being added by someone who has not read R9, and a nested payload the author did not
    think of.
    """
    if isinstance(payload, Mapping):
        if "confidence" in payload:
            missing = [name for name in R9_FIGURES if name not in payload]
            if missing:
                raise InvalidParameterError(
                    where, sorted(payload),
                    "a confidence travelling with all six of R9's figures. Missing: %s. A bare "
                    "confidence is not a weaker finding, it is not a finding, and it must not be "
                    "renderable by any client (R9)" % ", ".join(missing))
        for key, value in payload.items():
            refuse_bare_confidence(value, where="%s.%s" % (where, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            refuse_bare_confidence(value, where="%s[%d]" % (where, index))
    return payload


# --------------------------------------------------------------------------------------------
# The study store: read-only, and never a writer
# --------------------------------------------------------------------------------------------


class StudyStore:
    """Published evidence bundles on disk, read and never written.

    Deliberately not a database and deliberately not a cache. A bundle is a tamper-evident
    canonical file whose digest is checked on every load by `load_evidence_bundle`; holding a
    parsed copy in memory would mean serving a claim state that no longer matches what is on
    disk, which is exactly the staleness the digests exist to make impossible.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else study_root()

    def paths(self) -> List[Path]:
        if not self.root.is_dir():
            return []
        return sorted(path for path in self.root.glob("*.json") if path.is_file())

    def load(self, study_id: str) -> Tuple[EvidenceBundle, Path]:
        """The study's **latest** revision, chosen by the chain and not by the filename (D66).

        A bundle is immutable, so TG11.3 publishes each revision as its own file rather than
        rewriting one. Returning the first file that happened to match would then serve
        revision zero of a study for ever while the write path reported the revision it had
        actually appended - a read surface disagreeing with the record it reads.
        """
        found: Optional[Tuple[EvidenceBundle, Path]] = None
        for path in self.paths():
            try:
                bundle = load_evidence_bundle(path)
            except InvalidParameterError:
                continue
            if bundle.study_id != study_id:
                continue
            if found is None or bundle.revision > found[0].revision:
                found = (bundle, path)
        if found is None:
            raise KeyError(study_id)
        return found

    def summaries(self) -> List[Dict[str, Any]]:
        """One row per readable study, with the unreadable ones reported rather than skipped.

        A file that will not parse is a fact about the store worth surfacing: silently omitting
        it would let a corrupted bundle look like a study nobody ever ran.

        Earlier revisions of a study are not separate studies, so they are folded into the row
        for their latest one (D66) and the count of revisions behind it is reported. Listing
        them would turn one study that has been worked on into six that have not.
        """
        rows: List[Dict[str, Any]] = []
        latest: Dict[str, int] = {}
        for path in self.paths():
            try:
                bundle = load_evidence_bundle(path)
            except InvalidParameterError as exc:
                rows.append({"study_id": None, "file": path.name, "readable": False,
                             "refused_because": str(exc)})
                continue
            outputs = summarise_evidence(bundle)
            row = {
                "study_id": bundle.study_id,
                "file": path.name,
                "readable": True,
                "revision": bundle.revision,
                "superseded_revisions": 0,
                "bundle_sha256": bundle.bundle_sha256,
                "hypothesis": bundle.hypothesis.statement,
                "rung": outputs.rung,
                "blocked": outputs.blocked,
                "summary_sha256": outputs.summary_sha256,
            }
            index = latest.get(bundle.study_id)
            if index is None:
                latest[bundle.study_id] = len(rows)
                rows.append(row)
                continue
            standing = rows[index]
            superseded = standing["superseded_revisions"] + 1
            rows[index] = (row if bundle.revision > standing["revision"] else standing)
            rows[index]["superseded_revisions"] = superseded
        return rows


# --------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/findings", tags=["findings"])

#: Eager, once, at import of this module rather than inside a handler (D35).
REGISTERED_GLOSSARIES = register_builtin_glossaries()
REGISTERED_DOMAINS = register_builtin_domains()

#: Said wherever a domain's refusals are served, because the alternative is a reader concluding
#: something the record cannot support.  An `EvidenceBundle` does not carry the domain that
#: produced it - its fields are the hypothesis, the ten evidence categories and their digests -
#: so selecting a vocabulary states what *that domain* refuses and establishes nothing whatever
#: about the study being read through it.
DOMAIN_ATTRIBUTION_CAVEAT = (
    "An evidence bundle does not record which domain produced it. These limits describe the "
    "selected domain; they are not a check that this study came from it, and no such check "
    "exists.")


def _declaration_payload(name: str) -> Optional[Dict[str, Any]]:
    """A registered domain's declaration and what it refuses, or None if none is registered.

    A glossary may exist without a declaration - wording is registered separately from what a
    source is - so this returns None rather than inventing a declaration, and the route reports
    the absence rather than implying a domain with no limits.
    """
    if name not in DOMAIN_DECLARATIONS:
        return None
    declaration = declaration_for(name)
    return {
        "declared": declaration.describe(),
        "precedence_admissible": declaration.precedence_admissible,
        "minimum_admissible_lag_frames": declaration.minimum_admissible_lag(),
        "refuses": [dict(item) for item in refusals_for(declaration)],
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
    }


def _figures_for(bundle: EvidenceBundle) -> Optional[AssociationFigures]:
    """R9's six figures, if a passing effect-size entry actually carries all six.

    An entry that carries some of them is not partially rendered: `from_payload` refuses, and the
    finding is served without figures rather than with a subset. That is the whole point of R9 —
    four of six figures is not four-sixths of a finding.
    """
    for entry in bundle.evidence("effect_sizes"):
        if entry.status != "PASS":
            continue
        try:
            return AssociationFigures.from_payload(entry.payload)
        except InvalidParameterError:
            continue
    return None


@router.get("/domains")
async def list_domains() -> List[Dict[str, Any]]:
    """Every domain with a registered glossary, generated from the registry (E1, E2).

    Hand-maintaining this list would make it a document that goes stale, which is the failure the
    documentation guard exists to catch elsewhere in this codebase. A domain that registers a
    glossary appears here without anyone editing `src/api/` — TG8.1's condition carried onto the
    HTTP layer.
    """
    rows = []
    # The union of both registries, not just the glossaries. A domain that registered limits
    # and no wording would otherwise be invisible here — the same class of omission as TG9.1's,
    # with the halves swapped — and the row that is missing is the one carrying the refusals.
    for name in sorted(set(DOMAIN_GLOSSARIES.names()) | set(DOMAIN_DECLARATIONS.names())):
        entry = DOMAIN_GLOSSARIES.entry(name) if name in DOMAIN_GLOSSARIES else None
        glossary = entry.value if entry is not None else None
        rows.append({
            "name": name,
            "domain": glossary.domain if glossary is not None else name,
            "description": ((entry.description or glossary.description)
                            if glossary is not None
                            else DOMAIN_DECLARATIONS.entry(name).description),
            "glossary_sha256": glossary.glossary_sha256 if glossary is not None else None,
            "term_count": len(glossary.phrases) if glossary is not None else 0,
            "capabilities": dict(entry.capabilities) if entry is not None else {},
            "defined_in": entry.defined_in if entry is not None else "",
            # TG9.3: what the domain refuses, beside how it speaks. `None` means a vocabulary
            # was registered without a declaration behind it, which is reported rather than
            # rendered as a domain that happens to forbid nothing.
            "declaration": _declaration_payload(name),
            # TG8.1: whether the whole recipe was satisfied in one atomic call, and by which
            # file. A domain assembled from separate registrations reports `complete: false`
            # and names what is missing, rather than passing for one that was checked whole.
            "onboarding": audit_onboarding(name),
        })
    return refuse_bare_confidence(rows, where="domains")


@router.get("/onboarding")
async def onboarding_contract() -> Dict[str, Any]:
    """The adapter recipe itself, generated from `REQUIRED_DECLARATIONS` (TG8.1).

    Served rather than only documented so the contract a reader is held to and the contract the
    code enforces are the same tuple. A recipe that lives in prose beside the checks is a recipe
    that drifts from them, which is the failure the documentation audit exists to catch.
    """
    return refuse_bare_confidence({
        "schema": ONBOARDING_SCHEMA,
        "required": [{"requirement": name, "why": why}
                     for name, why in REQUIRED_DECLARATIONS],
        "onboarded": [audit_onboarding(name) for name in onboarded_names()],
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
    }, where="onboarding")


@router.get("/glossaries/{name}")
async def get_glossary(name: str) -> Dict[str, Any]:
    """One domain's full wording, so a reader can audit the words a finding was rendered in.

    A glossary is the one place a domain's own vocabulary enters the pipeline, and it is screened
    but not verified: nothing can know whether "anomaly" means to an oceanographer what the
    glossary implies. Publishing it whole is what makes that reviewable rather than hidden.
    """
    try:
        glossary = glossary_for(name)
    except Exception:
        raise HTTPException(status_code=404, detail="no glossary registered as %r" % name)
    return refuse_bare_confidence(glossary.to_mapping(), where="glossary")


@router.get("/studies")
async def list_studies() -> List[Dict[str, Any]]:
    """Every published study, with the rung it stands on and whether anything caps it."""
    return refuse_bare_confidence(StudyStore().summaries(), where="studies")


@router.get("/studies/{study_id}")
async def get_study(study_id: str) -> Dict[str, Any]:
    """One bundle, whole, with its digests.  The evidence itself, before any wording."""
    try:
        bundle, path = StudyStore().load(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no published study %r" % study_id)
    return refuse_bare_confidence(
        {"file": path.name, "bundle": bundle.to_mapping()}, where="study")


@router.get("/studies/{study_id}/outputs")
async def get_outputs(study_id: str) -> Dict[str, Any]:
    """The five outputs in the programme's own vocabulary, before translation.

    Served alongside the translated form on purpose: a reader who wants to check that the domain
    wording did not quietly change the finding needs to be able to see both.
    """
    try:
        bundle, _path = StudyStore().load(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no published study %r" % study_id)
    outputs = summarise_evidence(bundle)
    # `summary_sha256` is the digest *of* the mapping, so it cannot live inside it. It is served
    # beside it because a client that cannot name the exact claim state it was shown has no way
    # to tell a stale view from a current one.
    return refuse_bare_confidence(
        dict(outputs.to_mapping(), summary_sha256=outputs.summary_sha256), where="outputs")


#: The rung at which a claim first asserts temporal ordering.  Below it, a domain that refuses
#: precedence refuses nothing the claim is making.
_PRECEDENCE_RUNG = "candidate_precursor"


def _unadmitted_reading(rung: str, glossary_name: str) -> Optional[Dict[str, Any]]:
    """Report when a rung asserts precedence and the selected domain does not admit one.

    This is worth surfacing and easy to overstate, so the wording is careful. The ladder is
    domain-agnostic: it grades the evidence appended to a bundle and knows nothing about where
    that evidence came from. A bundle carries no domain either. So this is **not** a finding that
    a study is wrong, and nothing here moves a rung (R22).

    What it is: a reader has chosen to read a precedence claim in the words of a domain whose own
    declaration says a lead-lag reading is inadmissible from it (R21). If the study really did
    come from that domain, that is a contradiction someone needs to resolve. If it did not, the
    vocabulary is simply the wrong one to read it in. The surface cannot tell which, and says so.
    """
    if glossary_name not in DOMAIN_DECLARATIONS:
        return None
    declaration = declaration_for(glossary_name)
    if declaration.precedence_admissible:
        return None
    if CLAIM_RUNGS.index(rung) < CLAIM_RUNGS.index(_PRECEDENCE_RUNG):
        return None
    return {
        "rung": rung,
        "domain": declaration.name,
        "lag_policy": declaration.lag_policy,
        "note": (
            "This record stands at a rung that asserts temporal ordering, and the %s domain "
            "declares no admissible lag floor, so R21 does not permit a lead-lag reading from "
            "it. Nothing here changes the rung: the ladder grades the evidence recorded, and "
            "the evidence does not say which domain it came from." % declaration.name),
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
    }


@router.get("/studies/{study_id}/translation")
async def get_translation(
        study_id: str,
        glossary: str = Query(..., description="registered glossary name, from GET /domains"),
) -> Dict[str, Any]:
    """One finding, rendered in one domain's words, ready to display verbatim.

    The response carries `rendered_text` — the whole document as a client should show it —
    beside the structured units, because the point of Phase G9 is that a client displays strings
    rather than assembling them. `structural_keys` travels too, so two vocabularies can be checked
    against each other for the property that actually matters: different words, identical facts.
    """
    try:
        bundle, _path = StudyStore().load(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no published study %r" % study_id)
    try:
        wording = glossary_for(glossary)
    except Exception:
        raise HTTPException(status_code=404, detail="no glossary registered as %r" % glossary)
    outputs = summarise_evidence(bundle)
    document = translate(outputs, wording, figures=_figures_for(bundle))
    payload = dict(document.to_mapping(),
                   domain_limits=_declaration_payload(glossary),
                   unadmitted_reading=_unadmitted_reading(outputs.rung, glossary),
                   rendered_text=document.render(),
                   # The assembled figure line, served so a client never has to build one from
                   # the parts. Without this a view wanting to show the association strength in
                   # its own panel would have to interpolate `support` and `confidence` itself,
                   # and the moment it does that R9 depends on the view author again.
                   figures_text=(None if document.figures is None
                                 else document.figures.render()),
                   structural_keys=list(document.structural_keys),
                   translation_sha256=document.translation_sha256)
    return refuse_bare_confidence(payload, where="translation")


__all__ = ["router", "StudyStore", "refuse_bare_confidence", "study_root",
           "STUDY_ROOT_ENV", "R9_FIGURES", "REGISTERED_GLOSSARIES"]

