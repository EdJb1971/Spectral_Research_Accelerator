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

from src.core.builtin_glossaries import register_builtin_glossaries
from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle, load_evidence_bundle
from src.core.five_outputs import summarise_evidence
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
        for path in self.paths():
            try:
                bundle = load_evidence_bundle(path)
            except InvalidParameterError:
                continue
            if bundle.study_id == study_id:
                return bundle, path
        raise KeyError(study_id)

    def summaries(self) -> List[Dict[str, Any]]:
        """One row per readable study, with the unreadable ones reported rather than skipped.

        A file that will not parse is a fact about the store worth surfacing: silently omitting
        it would let a corrupted bundle look like a study nobody ever ran.
        """
        rows: List[Dict[str, Any]] = []
        for path in self.paths():
            try:
                bundle = load_evidence_bundle(path)
            except InvalidParameterError as exc:
                rows.append({"study_id": None, "file": path.name, "readable": False,
                             "refused_because": str(exc)})
                continue
            outputs = summarise_evidence(bundle)
            rows.append({
                "study_id": bundle.study_id,
                "file": path.name,
                "readable": True,
                "revision": bundle.revision,
                "bundle_sha256": bundle.bundle_sha256,
                "hypothesis": bundle.hypothesis.statement,
                "rung": outputs.rung,
                "blocked": outputs.blocked,
                "summary_sha256": outputs.summary_sha256,
            })
        return rows


# --------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/findings", tags=["findings"])

#: Eager, once, at import of this module rather than inside a handler (D35).
REGISTERED_GLOSSARIES = register_builtin_glossaries()


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
    for entry in DOMAIN_GLOSSARIES.entries():
        glossary = entry.value
        rows.append({
            "name": entry.name,
            "domain": glossary.domain,
            "description": entry.description or glossary.description,
            "glossary_sha256": glossary.glossary_sha256,
            "term_count": len(glossary.phrases),
            "capabilities": dict(entry.capabilities),
            "defined_in": entry.defined_in,
        })
    return refuse_bare_confidence(rows, where="domains")


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

