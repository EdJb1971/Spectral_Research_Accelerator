"""TG17.13 slice 2: the channel from the extension seam's acceptance test to the release gate.

Slice 1 wrote the source-edit audit `synthetic_fifth_adapter` had never had.  The gate still read
`NOT_RUN`, because an audit nothing carries into the qualification record is a measurement with no
channel to the thing it should decide - the same gap TG17.12 closed for `calendar_calibration` and
TG18.5 slice 4 closed for `browser_no_glue`.  This module is that channel.

**Only half of this gate needs a recording, and saying which half is the point.**

* The **source-edit audit** is decidable here, exactly, in milliseconds, from committed source.
  `read_extension_conformance` re-runs it live rather than trusting a receipt, because a live read
  of a file is strictly better evidence about that file than a recording of an earlier read.  A
  recording of a fact that can be recomputed is only a way to be wrong later.
* The **conformance run** cannot be read here at all.  TG17.3's acceptance requires the fifth
  adapter to be defined in a module *the application never imports*, so it lives in
  `src/tests/test_adapter_registry.py` and nothing under `src/core` may reach it.  That is the
  constraint being tested, not an inconvenience, and it is exactly why the run has to be recorded
  by the test and read back here.

**Recording and deciding stay apart, and deciding is on this side.**  The test supplies apparatus
only it owns - the adapter, the native record, the windows - and this module performs every check
and decides whether they passed.  A test that deleted its own assertions would therefore change
nothing about what gets recorded; the checks are here.  What the test can still do is stop calling
the recorder, and the answer to that is `NOT_RUN`, which is the correct answer.

**What a recording is bound to.**

* The **declared contract**: which names the fifth adapter is built from, which surfaces count as
  the generic seam, which occurrence kinds count as glue, every declared occurrence with its
  stated reason, and the checks a conformance run must pass.  Adding a declaration that excuses
  new glue moves this digest, so a recording cannot be made green by widening the list it is
  judged against.
* The **source** of the modules that decide what the measurement is: every framework source the
  audit scans, the audit itself, the conformance kit, the trajectory contract, and the test module
  that defines the adapter.  A weakened acceptance test moves this digest and returns the gate to
  `NOT_RUN` until the measurement is actually made again.

Neither digest is tamper-evidence against someone editing this repository, and neither is meant to
be.  What they are proof against is drift: a surface that grew a branch after the run, or a
recording that outlived the code it describes.

Nothing here is evidence about the world.  It is a statement about this repository's source and
about whether one synthetic adapter reached the seams a third party's adapter would have to reach.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.extension_audit import (
    DECLARED_OCCURRENCES,
    FIFTH_ADAPTER_NAMES,
    FRAMEWORK_SOURCES,
    GLUE_KINDS,
    audit_framework_edits,
)
from src.core.extension_audit import source_digests as framework_digests


SCHEMA = "extension-evidence/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = "measurements"
RECORD_NAME = "extension_conformance.json"

FIFTH_DOMAIN, RANK_CHANNEL = FIFTH_ADAPTER_NAMES

#: The shared channel the four registered domains produce. The fifth adapter exists to prove the
#: seam carries genuinely different mathematics, so producing *this* would make the acceptance
#: vacuous: it would test the factory's arguments rather than the seam.
SHARED_CHANNEL = "standardized_level"

#: Fixed here rather than passed in, because it is part of what conformance means at this gate
#: and not part of the fixture the test owns.
MAX_PLANNED_BYTES = 4 * 1024 ** 3

#: An adapter installed through the supported seam is defined outside the application. These are
#: the directories that would mean it was not - if the fifth adapter's own mathematics lived in any
#: of them, the acceptance would be measuring a builtin wearing a different name.
#:
#: Directories, resolved from the defining *file*, rather than dotted package names. The first
#: version of this check compared `translate.__module__` against package prefixes, and under
#: pytest's import mode that string is the bare `test_adapter_registry` - which starts with none of
#: them, and would have started with none of them whatever the module was called. It passed
#: because it could not see what it was checking, which is D64, D74 and D75 a fifth time. A path
#: that cannot be resolved into this repository is now a refusal rather than a pass.
APPLICATION_SOURCE_ROOTS: Tuple[str, ...] = ("src/core/", "src/api/", "src/adapters/",
                                             "src/benchmarks/", "src/statistics/",
                                             "src/database/", "extensions/", "frontend/")

#: The modules whose source decides what this measurement is. The framework sources are added at
#: call time from the audit, so the two lists cannot drift apart by being maintained twice.
EXTENSION_SOURCES: Tuple[str, ...] = (
    "src/core/extension_audit.py",
    "src/core/adapter_conformance.py",
    "src/core/experiment_adapter.py",
    "src/core/structural_trajectory.py",
    "src/core/onboarding.py",
    "src/tests/test_adapter_registry.py",
)

#: Every check a conformance run must pass, named in the contract so that a recording carrying
#: fewer of them is refused rather than read as a pass with less in it.
REQUIRED_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("defined_outside_the_application",
     "The adapter's own mathematics is defined in a module the application never imports."),
    ("reaches_the_registry",
     "The domain appears in `registered_domains()` after the third-party registration path."),
    ("resolves_by_domain",
     "`adapter_for_domain` returns this adapter and not something standing in for it."),
    ("describes_controls_generically",
     "The controls reach a renderer through the declared schema, carrying no domain name."),
    ("conforms_to_the_kit",
     "`run_conformance` reports conformant against the declared invariances and lineage."),
    ("carries_different_mathematics",
     "The structural channel is not the shared standardized level the four domains produce."),
    ("reaches_the_domain_blind_mining_seam",
     "`mine_structural_peak` consumes the canonical contract without knowing the domain."),
    ("no_framework_source_names_it",
     "The source-edit audit finds the fifth adapter named in no framework source."),
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_digest(root: Path, relative: str) -> str:
    """Newline-normalised, so a Windows checkout and a POSIX one agree about the same source."""
    path = root / relative
    if not path.is_file():
        return "ABSENT"
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def conformance_contract() -> Dict[str, Any]:
    """The declared contract this gate judges a recording against, as its own digestible fact.

    The declared occurrences are part of it verbatim, reasons included.  That is deliberate: the
    audit's whole discipline is that an occurrence is excused by a written reason rather than by
    membership of a list, and a contract digest that covered only the source and domain would let
    a reason be rewritten under a recording that still looked current.
    """
    contract = {
        "gate": "synthetic_fifth_adapter",
        "fifth_adapter_names": list(FIFTH_ADAPTER_NAMES),
        "shared_channel_that_would_make_it_vacuous": SHARED_CHANNEL,
        "framework_sources": list(FRAMEWORK_SOURCES),
        "glue_kinds": list(GLUE_KINDS),
        "maximum_planned_bytes": MAX_PLANNED_BYTES,
        "application_source_roots": list(APPLICATION_SOURCE_ROOTS),
        "required_checks": [{"check": name, "means": means} for name, means in REQUIRED_CHECKS],
        "declared_occurrences": [
            {"source": source, "domain": domain, "kind": kind, "reason": reason}
            for source, domain, kind, reason in DECLARED_OCCURRENCES],
        "installation_required_framework_edits_must_be": 0,
        "adapter_specific_framework_edits_is": "reported, not asserted",
    }
    return {**contract, "contract_sha256": _digest(contract)}


def source_digests(root: Optional[Path] = None) -> Dict[str, str]:
    """Every source that decides what this measurement is: the audit's, and this gate's own."""
    base = Path(root) if root is not None else REPO_ROOT
    digests = dict(framework_digests(base))
    for name in EXTENSION_SOURCES:
        digests[name] = _file_digest(base, name)
    return digests


def record_file(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else REPO_ROOT
    return base / RECORD_DIR / RECORD_NAME


# --------------------------------------------------------------------------- the measurement

def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def defining_source(adapter: Any, root: Optional[Path] = None) -> Optional[str]:
    """The repository-relative file the adapter's own mathematics is defined in, or ``None``.

    ``None`` means the question could not be answered - a builtin, a C extension, an exec'd
    string, a file outside this checkout - and the caller treats that as a refusal rather than as
    a pass, because an unanswerable question is not a satisfied one.
    """
    import inspect

    base = Path(root) if root is not None else REPO_ROOT
    translate = getattr(adapter, "translate", None)
    try:
        source = inspect.getsourcefile(translate) if translate is not None else None
    except TypeError:  # pragma: no cover - a builtin or a C-level callable
        return None
    if not source or not Path(source).is_file():
        # `getsourcefile` will hand back a pseudo-filename like `<no file>` for code compiled from
        # a string whose module carries a loader, and on Windows that resolves to an absolute path
        # inside the repository without raising. A source that cannot be read is not a located one.
        return None
    try:
        relative = Path(source).resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return None
    return relative.as_posix()


def measure_extension_conformance(adapter: Any, native_record: Any,
                                  windows: Sequence[Any],
                                  root: Optional[Path] = None) -> Dict[str, Any]:
    """Perform every check this gate turns on, against apparatus the caller owns.

    The caller supplies the fifth adapter and its fixture, because only a module outside the
    application can hold them.  Every decision about whether they passed is made here, so a
    caller cannot record a pass by asserting less.
    """
    from src.core.adapter_conformance import ConformanceCase, run_conformance
    from src.core.experiment_adapter import adapter_for_domain, registered_domains
    from src.core.structural_trajectory import mine_structural_peak

    base = Path(root) if root is not None else REPO_ROOT
    declared_domain = getattr(getattr(adapter, "declaration", None), "name", None)
    if declared_domain != FIFTH_DOMAIN:
        return {"status": "REFUSED", "checks": [],
                "reasons": ["the adapter offered declares domain %r, and this gate measures %r; "
                            "a recording made from another adapter would qualify nothing"
                            % (declared_domain, FIFTH_DOMAIN)]}
    if getattr(native_record, "domain", None) != FIFTH_DOMAIN:
        return {"status": "REFUSED", "checks": [],
                "reasons": ["the native record offered belongs to %r rather than %r"
                            % (getattr(native_record, "domain", None), FIFTH_DOMAIN)]}
    if not windows:
        return {"status": "REFUSED", "checks": [],
                "reasons": ["no window was offered, so a conformance case could not be built and "
                            "its silence would not be evidence of anything"]}

    source = defining_source(adapter, base)
    if source is None:
        return {"status": "REFUSED", "checks": [],
                "reasons": ["the adapter's own mathematics could not be traced to a file in this "
                            "checkout, so whether it was defined outside the application is "
                            "unanswerable rather than satisfied"]}
    checks: List[Dict[str, Any]] = [
        _check("defined_outside_the_application",
               not source.startswith(APPLICATION_SOURCE_ROOTS),
               "the adapter's translate is defined in %s" % source)]

    domains = tuple(registered_domains())
    checks.append(_check("reaches_the_registry", FIFTH_DOMAIN in domains,
                         "registered domains: %s" % ", ".join(sorted(domains))))
    try:
        resolved = adapter_for_domain(FIFTH_DOMAIN)
    except Exception as error:  # pragma: no cover - an unregistered domain is check 2's failure
        resolved, resolution = None, "lookup raised %s" % (error,)
    else:
        resolution = "resolved %s" % getattr(resolved, "adapter_id", "an adapter")
    checks.append(_check("resolves_by_domain", resolved is adapter, resolution))

    fields = list((adapter.describe().get("controls") or {}).get("fields") or [])
    named = sorted(item.get("name", "") for item in fields
                   if FIFTH_DOMAIN in str(item.get("name", "")))
    checks.append(_check(
        "describes_controls_generically", bool(fields) and not named,
        "%d declared control field(s), %d carrying the domain name" % (len(fields), len(named))))

    report = run_conformance(adapter, ConformanceCase(
        parameters={}, windows=list(windows), maximum_planned_bytes=MAX_PLANNED_BYTES,
        native_record=native_record))
    failures = sorted(item.name for item in report.failures)
    unprobed = sorted(item.name for item in report.checks if item.status == "NOT_PROBED")
    checks.append(_check(
        "conforms_to_the_kit", bool(report.conformant),
        "%d conformance checks, %d failed, %d unprobed and reported as unverified%s"
        % (len(report.checks), len(failures), len(unprobed),
           (": %s" % ", ".join(failures)) if failures else "")))

    declaration = adapter.structural_declaration({})
    channels = sorted(declaration.channels)
    checks.append(_check(
        "carries_different_mathematics",
        SHARED_CHANNEL not in channels and RANK_CHANNEL in channels,
        "declares channel(s) %s" % ", ".join(channels)))

    trajectory = adapter.translate(native_record, declaration, {})
    try:
        peak = mine_structural_peak(trajectory, channel=RANK_CHANNEL)
    except Exception as error:  # pragma: no cover - a channel the seam cannot consume
        peak, mined = {}, "the mining seam refused: %s" % (error,)
    else:
        mined = "peak on channel %r in %s over [%.1f, %.1f) seconds" % (
            peak.get("channel"), peak.get("units"), peak.get("support_start_seconds", 0.0),
            peak.get("support_end_seconds", 0.0))
    checks.append(_check("reaches_the_domain_blind_mining_seam",
                         bool(peak) and peak.get("units") == "dimensionless", mined))

    audit = audit_framework_edits(base)
    installation = audit.get("installation_required_framework_edits")
    checks.append(_check(
        "no_framework_source_names_it",
        audit.get("status") == "MEASURED" and installation == 0,
        "the source-edit audit reports %s over %d framework sources%s"
        % (audit.get("status"), audit.get("framework_sources", 0),
           (": %s" % " ".join(audit.get("reasons") or [])) if audit.get("reasons") else "")))

    missing = sorted({name for name, _ in REQUIRED_CHECKS}
                     - {item["check"] for item in checks})
    if missing:  # pragma: no cover - a check named in the contract but never performed
        return {"status": "REFUSED", "checks": checks,
                "reasons": ["the contract names checks this measurement did not perform: %s"
                            % ", ".join(missing)]}

    failed = [item["check"] for item in checks if item["status"] != "PASS"]
    return {
        "status": "MEASURED",
        "conformant": not failed,
        "checks": checks,
        "failed_checks": failed,
        # Reported, never asserted. The count of standing glue is a property of the whole surface
        # over time; the count of edits *this installation* required is the check above, and is 0.
        "adapter_specific_framework_edits": audit.get("adapter_specific_framework_edits"),
        "installation_required_framework_edits": installation,
        "glue": audit.get("glue") or [],
        "reasons": [],
    }


def record_extension_conformance(adapter: Any, native_record: Any, windows: Sequence[Any],
                                 root: Optional[Path] = None) -> Dict[str, Any]:
    """Write what the acceptance run measured, bound to what it measured it against.

    The recording is committed source, and this runs on every pass of the test suite, so it is
    written only when its content is new.  `recorded_utc` therefore says when this *measurement*
    was first obtained rather than when the file was last touched - and nothing reads it to decide
    anything.  Staleness is decided by the digests, which move when the code does.
    """
    base = Path(root) if root is not None else REPO_ROOT
    contract = conformance_contract()
    measured = measure_extension_conformance(adapter, native_record, windows, base)
    record = {
        "schema": SCHEMA,
        "gate": "synthetic_fifth_adapter",
        "domain": FIFTH_DOMAIN,
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": source_digests(base),
        "claim_boundary": (
            "One synthetic adapter reached the seams a third party's adapter would have to "
            "reach, and no framework source names it. It is not a result about any domain."),
        **{key: value for key, value in measured.items() if key != "reasons"},
        "reasons": measured.get("reasons") or [],
    }

    path = record_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):  # pragma: no cover - unreadable recording is replaced
            previous = None
        if previous is not None and _digest({**previous, "recorded_utc": ""}) == _digest(
                {**record, "recorded_utc": ""}):
            return previous
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return record


# ------------------------------------------------------------------------------ the decision

def _stale_reasons(record: Mapping[str, Any], contract: Mapping[str, Any],
                   sources: Mapping[str, str]) -> List[str]:
    reasons: List[str] = []
    if record.get("contract_sha256") != contract["contract_sha256"]:
        reasons.append("the declared extension contract has changed since it was recorded")
    recorded = record.get("source_sha256") or {}
    moved = sorted(name for name, digest in sources.items() if recorded.get(name) != digest)
    if moved:
        reasons.append("the source of %s has changed since the run was recorded"
                       % ", ".join(moved))
    return reasons


def read_extension_conformance(root: Optional[Path] = None) -> Dict[str, Any]:
    """What the `synthetic_fifth_adapter` gate is entitled to say, and why.

    `PASS` needs both halves and they fail differently.  The audit is read live, so its refusal
    means the audit does not know what it is looking at and the gate reads `NOT_RUN`; its measured
    finding of a framework source naming the fifth adapter is a *measured* falsehood of the
    installation claim and reads `FAIL`.  The recording is read from disk, so absence and drift
    mean the run has not happened for this code (`NOT_RUN`), and a run that happened with a check
    failing reads `FAIL`.
    """
    base = Path(root) if root is not None else REPO_ROOT
    contract = conformance_contract()
    audit = audit_framework_edits(base)
    facts: Dict[str, Any] = {
        "schema": SCHEMA,
        "gate": "synthetic_fifth_adapter",
        "measured_by": "src/tests/test_adapter_registry.py (the acceptance run) and "
                       "src.core.extension_audit (the source-edit audit, read live here)",
        "executed_here": False,
        "audit_read_live_here": True,
        "record_path": "%s/%s" % (RECORD_DIR, RECORD_NAME),
        "contract_sha256": contract["contract_sha256"],
        "audit": audit,
        "required_checks": len(REQUIRED_CHECKS),
        "claim_boundary": (
            "The extension seam carried one synthetic adapter with different mathematics, and the "
            "generic surfaces name it nowhere. Neither half is evidence about any domain's data."),
    }

    if audit.get("status") != "MEASURED":
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": list(audit.get("reasons") or
                                ["the source-edit audit produced no outcome"])}
    if audit.get("installation_required_framework_edits") != 0:
        return {**facts, "status": "FAIL", "recorded": None,
                "reasons": ["the fifth adapter is named in a framework source, so it was not "
                            "installed without framework edits"]}

    path = record_file(base)
    if not path.is_file():
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["no acceptance run has been recorded for this checkout"]}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:  # pragma: no cover - unreadable recording
        return {**facts, "status": "NOT_RUN", "recorded": None,
                "reasons": ["the recorded acceptance run could not be read: %s" % (error,)]}

    summary = {
        "recorded_utc": record.get("recorded_utc"),
        "checks": len(record.get("checks") or []),
        "failed_checks": list(record.get("failed_checks") or []),
        "adapter_specific_framework_edits": record.get("adapter_specific_framework_edits"),
        "installation_required_framework_edits":
            record.get("installation_required_framework_edits"),
        "glue": record.get("glue") or [],
    }
    facts = {**facts, "recorded": summary}

    reasons = _stale_reasons(record, contract, source_digests(base))
    performed = {item.get("check") for item in record.get("checks") or []}
    absent = sorted({name for name, _ in REQUIRED_CHECKS} - performed)
    if absent:
        reasons.append("the recorded run did not perform %s" % ", ".join(absent))
    if record.get("status") != "MEASURED":
        reasons.append("the recorded run produced no measurement: %s"
                       % " ".join(record.get("reasons") or ["no reason was recorded"]))
    if reasons:
        return {**facts, "status": "NOT_RUN", "reasons": reasons}

    if summary["failed_checks"] or not record.get("conformant"):
        return {**facts, "status": "FAIL",
                "reasons": ["the recorded acceptance run failed %s"
                            % ", ".join(summary["failed_checks"] or ["an unnamed check"])]}
    return {**facts, "status": "PASS", "reasons": []}


__all__ = ["APPLICATION_SOURCE_ROOTS", "EXTENSION_SOURCES", "FIFTH_DOMAIN", "MAX_PLANNED_BYTES",
           "RANK_CHANNEL", "RECORD_DIR", "RECORD_NAME", "REPO_ROOT", "REQUIRED_CHECKS", "SCHEMA",
           "SHARED_CHANNEL", "conformance_contract", "defining_source",
           "measure_extension_conformance", "read_extension_conformance",
           "record_extension_conformance", "record_file", "source_digests"]
