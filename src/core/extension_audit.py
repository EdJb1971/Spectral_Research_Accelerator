"""TG17.13 slice 1: the source-edit audit `synthetic_fifth_adapter` has never had.

TG17.3's acceptance test is named
``test_synthetic_fifth_adapter_reaches_the_registry_and_conforms_without_framework_edits``
and it proves the first half of its own name. A fifth domain with genuinely different
mathematics reaches the registry, the control schema, the conformance kit and the domain-blind
mining seam from a module the application never imports. **Nothing in the repository proved the
second half.** "without framework edits" was carried in a test name and guarded by no assertion,
which is why `adapter_specific_framework_edits` has read `NOT_MEASURED` since TG17.10: not
because a browser cannot observe it, but because nothing had been written that could.

This module measures it, and the measurement has two parts that must not be run together.

**The installation claim, which is absolute.** Installing the synthetic fifth adapter required no
edit to any framework source, and the evidence is that no framework source names it. There is no
admissible reason for `synthetic_rank_sensor` or `monotone_rank` to appear in the generic seam, so
an occurrence is a failure and no declaration may excuse it. This is the half the gate turns on.

**The standing glue count, which is reported rather than asserted.** G17's acceptance also asks
that per-domain branches in the generic surfaces "trend to zero rather than merely move files".
That is a property of the whole surface over time, not of one adapter's installation, so this
module counts it and names every instance instead of collapsing it to a pass. A count that blocked
release would make an unrelated archive's acquisition semantics a release decision; a count that
went unpublished would let glue accumulate behind a green gate.

**Every occurrence must be declared, and a declaration is a sentence rather than a suppression.**
The scan reports every place a framework source names a registered domain. Each one is either in
`DECLARED_OCCURRENCES` with a stated kind and reason, or the audit refuses. An undeclared
occurrence cannot be waved through by adding it to a list without also writing down why it is
there, and the kind decides whether it counts as glue - `BRANCH` does, prose and recipe content do
not. This is the part that keeps the audit from becoming something that can be massaged into
agreeing with itself.

Nothing here is evidence about the world. It is a statement about this repository's source.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA = "extension-audit/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The names the synthetic fifth adapter is built from. `test_adapter_registry.py` owns them; they
#: are restated here because the audit's whole claim is that they appear in none of the sources
#: below, and a scan that imported them from the test module would be scanning for whatever that
#: module happened to say today.
FIFTH_ADAPTER_NAMES: Tuple[str, ...] = ("synthetic_rank_sensor", "monotone_rank")

#: The generic seam: the surfaces a new domain must reach *through* rather than *by editing*.
#: Enumerated rather than globbed, because a glob would sweep in the domain adapters themselves,
#: which name their own domain for the same reason a recipe does - it is their subject.
FRAMEWORK_SOURCES: Tuple[str, ...] = (
    # The orchestrator and the registry seam.
    "src/core/experiment_run.py",
    "src/core/experiment_adapter.py",
    "src/core/experiment_manifest.py",
    "src/core/experiment_family.py",
    "src/core/composer_path.py",
    "src/core/comparison_views.py",
    "src/core/adapter_conformance.py",
    # The generic API routes.
    "src/api/experiment_composer.py",
    "src/api/experiment_runs.py",
    "src/api/cross_domain.py",
    "src/api/mining.py",
    # The UI source that renders whatever is registered.
    "frontend/src/App.tsx",
    "frontend/src/components/AcquisitionView.tsx",
    "frontend/src/components/AdapterControls.tsx",
    "frontend/src/components/StructureMiningView.tsx",
    "frontend/src/components/ExperimentComposer.tsx",
    "frontend/src/components/ChannelRecords.tsx",
)

BRANCH = "BRANCH"
PROSE = "PROSE"
RECIPE = "RECIPE"
DEFAULT = "DEFAULT"

#: Only `BRANCH` is glue. A framework source that *names* a domain in prose or carries a named
#: recipe's own content has not been edited to make that domain work; a source that *behaves*
#: differently because of the name has.
GLUE_KINDS: Tuple[str, ...] = (BRANCH,)

#: (source, domain, kind, reason). Every occurrence the scan finds must match one of these on
#: source and domain, and the reason is the point: it is what a later reader has to disagree with
#: in order to remove the entry.
DECLARED_OCCURRENCES: Tuple[Tuple[str, str, str, str], ...] = (
    ("src/core/experiment_manifest.py", "reanalysis", RECIPE,
     "TG17.0's flagship quartet is a named scientific recipe, and its content is which four "
     "domains it is. A fifth adapter does not need this edited to work; it would need it edited "
     "only to join the quartet, which is a scientific act and not an installation."),
    ("src/core/experiment_manifest.py", "argo_float", RECIPE,
     "The same flagship recipe row."),
    ("src/core/experiment_manifest.py", "tess_lightcurve", RECIPE,
     "The same flagship recipe row."),
    ("src/core/experiment_manifest.py", "order_book", RECIPE,
     "The same flagship recipe row."),
    ("src/core/comparison_views.py", "reanalysis", PROSE,
     "A module docstring explaining why an unshared colour scale misleads, using a temperature "
     "series and a depth series as the example. Prose naming an example is not a branch."),
    ("src/api/experiment_composer.py", "reanalysis", PROSE,
     "`representation_preview` refuses unless the three known-answer domains are declared, and "
     "names them in the refusal. The route serves one frozen known answer, so the names are that "
     "fixture's identity rather than a dispatch: no branch behaves differently per domain, and a "
     "fifth adapter correctly cannot use this route at all."),
    ("src/api/experiment_composer.py", "argo_float", PROSE,
     "The same known-answer refusal."),
    ("src/api/experiment_composer.py", "tess_lightcurve", PROSE,
     "The same known-answer refusal."),
    ("frontend/src/App.tsx", "reanalysis", PROSE,
     "Workspace copy describing what the reanalysis workspace is for."),
    ("frontend/src/components/AdapterControls.tsx", "reanalysis", PROSE,
     "A comment asserting that no `domain === 'reanalysis'` branch exists in this form. True of "
     "this file, and the audit records that it was not true of the surface when written: see the "
     "AcquisitionView entry below."),
    ("frontend/src/components/StructureMiningView.tsx", "reanalysis", DEFAULT,
     "The domain selector's initial value. It privileges one registered domain in the opening "
     "state and changes no behaviour once a selection is made. Recorded rather than excused: a "
     "default drawn from the registry's first entry would carry no domain name at all."),
    ("frontend/src/components/AcquisitionView.tsx", "reanalysis", BRANCH,
     "`domainName === 'reanalysis' && <CDSPlanner />` renders a bespoke acquisition planner for "
     "one archive. This is glue and is counted as glue. Copernicus acquisition is a long-running "
     "job with queue position, progress and resumption rather than a fetch, so the planner is not "
     "a form the generic control schema can currently express - but that is a reason the glue "
     "exists, not a reason it stops being glue."),
)


def _digest(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        return "ABSENT"
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_digests(root: Optional[Path] = None) -> Dict[str, str]:
    """Every framework source, digested, so a recording can be bound to what it scanned."""
    base = Path(root) if root is not None else REPO_ROOT
    return {name: _digest(base, name) for name in FRAMEWORK_SOURCES}


def _scan(base: Path, names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Every line of every framework source that names one of ``names``."""
    found: List[Dict[str, Any]] = []
    for source in FRAMEWORK_SOURCES:
        path = base / source
        if not path.is_file():
            found.append({"source": source, "domain": None, "line": 0,
                          "text": "", "missing": True})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            for name in names:
                if re.search(r"\b%s\b" % re.escape(name), line):
                    found.append({"source": source, "domain": name, "line": number,
                                  "text": line.strip()[:160], "missing": False})
    return found


def _live_domains() -> Tuple[str, ...]:
    """The registered domains, with the registry actually loaded first.

    The adapter registry is empty until something asks for it, and the first version of this
    module read it cold: it scanned for an empty set of names, found nothing, and reported a
    clean surface. That is the defect this repository has already met three times (D64, D74,
    D75) - a guard passing because it could not see what it was checking - so an empty registry
    is a refusal here rather than a scan with nothing to look for.
    """
    from src.adapters import register_builtin_adapters, register_extension_adapters
    from src.core.experiment_adapter import registered_domains

    register_builtin_adapters()
    register_extension_adapters()
    return tuple(registered_domains())


def audit_framework_edits(root: Optional[Path] = None,
                          domains: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """What `synthetic_fifth_adapter` is entitled to say about framework edits, and why.

    ``MEASURED`` requires that no framework source names the synthetic fifth adapter and that
    every occurrence of a registered domain is declared.  Either failing is a refusal, and the two
    are kept apart in the reasons because they mean different things: the first says the
    installation claim is false, the second says the audit does not know what it is looking at.
    """
    base = Path(root) if root is not None else REPO_ROOT
    if domains is None:
        domains = tuple(_live_domains())

    missing = sorted({item["source"] for item in _scan(base, ("",)) if item.get("missing")})
    installation = [item for item in _scan(base, FIFTH_ADAPTER_NAMES) if not item["missing"]]
    occurrences = [item for item in _scan(base, tuple(domains)) if not item["missing"]]

    declared = {(source, domain): (kind, reason)
                for source, domain, kind, reason in DECLARED_OCCURRENCES}
    undeclared = sorted({(item["source"], item["domain"]) for item in occurrences
                         if (item["source"], item["domain"]) not in declared})

    reasons: List[str] = []
    if not domains:
        reasons.append("no domains are registered, so the scan had no names to look for and its "
                       "silence is not evidence of a clean surface")
    if missing:
        reasons.append("framework sources are absent from this checkout: %s" % ", ".join(missing))
    if installation:
        reasons.append(
            "the synthetic fifth adapter is named in %s, so it was not installed without "
            "framework edits" % ", ".join(sorted({"%s:%d" % (item["source"], item["line"])
                                                  for item in installation})))
    if undeclared:
        reasons.append(
            "these framework sources name a registered domain with no declared reason: %s"
            % ", ".join("%s (%s)" % pair for pair in undeclared))

    glue = [{**item, "kind": declared[(item["source"], item["domain"])][0],
             "reason": declared[(item["source"], item["domain"])][1]}
            for item in occurrences
            if (item["source"], item["domain"]) in declared
            and declared[(item["source"], item["domain"])][0] in GLUE_KINDS]

    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "framework_sources": len(FRAMEWORK_SOURCES),
        "registered_domains": sorted(domains),
        "claim_boundary": (
            "A statement about this repository's source, not about the world. It says which "
            "generic surfaces name a domain, not whether any domain computes anything correctly."),
    }
    if reasons:
        return {**facts, "status": "NOT_MEASURED", "reasons": reasons,
                "adapter_specific_framework_edits": "NOT_MEASURED"}

    return {
        **facts,
        "status": "MEASURED",
        # The number the gate has never had. It is a count of standing glue in the generic
        # surfaces, not a count of edits this installation required - that count is zero, and it
        # is zero by the installation check above rather than by this one.
        "adapter_specific_framework_edits": len(glue),
        "installation_required_framework_edits": 0,
        "glue": [{"source": item["source"], "line": item["line"], "domain": item["domain"],
                  "reason": item["reason"]} for item in glue],
        "declared_occurrences": len(DECLARED_OCCURRENCES),
        "occurrences_found": len(occurrences),
        "reasons": [],
    }


__all__ = ["BRANCH", "DECLARED_OCCURRENCES", "DEFAULT", "FIFTH_ADAPTER_NAMES",
           "FRAMEWORK_SOURCES", "GLUE_KINDS", "PROSE", "RECIPE", "REPO_ROOT", "SCHEMA",
           "audit_framework_edits", "source_digests"]
