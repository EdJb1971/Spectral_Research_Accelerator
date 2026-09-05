"""The adapter conformance kit (TG17.3, `ed-dev`).

**Why a kit rather than a checklist.** TG17.2 made the translation contract executable: an
adapter declares its invariances, its missing-data behaviour, its legitimate null and the
operations it refuses. Declaring them is not the same as honouring them, and a declaration
nobody executes is prose with a digest attached — which is worse than prose, because the digest
makes it look checked.

So each check here *runs* the thing declared:

=================================  ==========================================================
``controls_schema_agreement``      the schema the UI renders is the schema the code validates
``coverage_honesty``               sparse or intersecting support may never report exactness
``bounded_resource_plan``          the plan's own byte estimate is priced against the cap
``deterministic_translation``      the same native record twice yields the same canonical bytes
``content_addressing``             native, adapter and configuration digests all reconstruct
``axis_and_role_validation``       required axes and roles are declared and match the domain
``gap_preservation``               TG17.2 conformance: exact clock, support, validity, lineage
``refusal_propagation``            what the domain refuses, the adapter refuses
``declared_invariances``           each named invariance is applied and the output compared
``null_suitability``               the null moves the values and preserves the gaps
=================================  ==========================================================

**The invariance probes are the interesting half.** An adapter that declares
``native_value_positive_scaling`` is claiming that multiplying every native value by a positive
constant leaves its canonical channels unchanged. `INVARIANCE_PROBES` applies exactly that
transform and compares the output. A declaration naming an invariance the translator does not
have now fails, where previously it would have been a true-sounding sentence in a digest.

An invariance with no registered probe is reported ``NOT_PROBED`` rather than ``PASS``. Silently
passing an unexecuted claim is the failure this module exists to prevent, and reporting it as
passing would reintroduce it one level up.

**What conformance does not establish.** It says an adapter honours its own declaration. It does
not say the declaration is the right one for the archive, that the units are what the archive
means by them, or that the domain is a legitimate second domain under R17 — that last is
`onboard_domain`'s job and is checked when the adapter is constructed. Conformance is internal
agreement, and claims nothing further.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.experiment_adapter import (AdapterConformanceError, DomainExperimentAdapter,
                                         plan_windows)
from src.core.structural_trajectory import (NativeStructuralRecord, StructuralConformanceError,
                                            assert_structural_conformance)


SCHEMA = "domain-adapter-conformance/v1"

PASS = "PASS"
FAIL = "FAIL"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_PROBED = "NOT_PROBED"

#: Support kinds whose extent cannot establish exact coverage. TG17.1 froze the reason: a TESS
#: sector that intersects a requested window says an observation exists somewhere in it, not that
#: the window is covered, and the same is true of a sparse profile array or a local record.
INEXACT_SUPPORT: Tuple[str, ...] = ("sparse_point_support",
                                    "intersecting_observational_sectors",
                                    "irregular_local_record")

#: Operations an adapter must refuse when its domain declares no admissible lag floor (R21).
PRECEDENCE_REFUSALS: Tuple[str, ...] = ("causality",)


def _scaled(record: NativeStructuralRecord, factor: float) -> NativeStructuralRecord:
    return replace(record, values=np.asarray(record.values) * factor)


def _shifted(record: NativeStructuralRecord, offset: float) -> NativeStructuralRecord:
    return replace(record, values=np.asarray(record.values) + offset)


#: Executable probes for the invariance vocabulary. Each returns a native record the canonical
#: channels must be *unchanged* by. Adding an invariance to the vocabulary without adding its
#: probe leaves it reported `NOT_PROBED`, which is visible; it does not quietly pass.
INVARIANCE_PROBES: Dict[str, Callable[[NativeStructuralRecord], NativeStructuralRecord]] = {
    "native_value_translation": lambda record: _shifted(record, 7.5),
    "native_value_positive_scaling": lambda record: _scaled(record, 3.25),
}


@dataclass(frozen=True)
class ConformanceCase:
    """One adapter, one set of controls, one native record, one resource budget."""

    parameters: Mapping[str, Any]
    windows: Sequence[Any]
    maximum_planned_bytes: int
    native_record: NativeStructuralRecord
    identity: Mapping[str, Any] = dc_field(default_factory=dict)
    null_seed: int = 20260830

    def __post_init__(self) -> None:
        if self.maximum_planned_bytes <= 0:
            raise ValueError("a conformance case needs a positive planned-byte budget")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    evidence: Mapping[str, Any] = dc_field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {"check": self.name, "status": self.status, "detail": self.detail,
                "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class ConformanceReport:
    adapter_id: str
    domain: str
    definition_sha256: str
    checks: Tuple[CheckResult, ...]

    @property
    def failures(self) -> Tuple[CheckResult, ...]:
        return tuple(item for item in self.checks if item.status == FAIL)

    @property
    def conformant(self) -> bool:
        return not self.failures

    def describe(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for item in self.checks:
            counts[item.status] = counts.get(item.status, 0) + 1
        return {"schema": SCHEMA, "adapter_id": self.adapter_id, "domain": self.domain,
                "definition_sha256": self.definition_sha256, "conformant": self.conformant,
                "counts": counts, "checks": [item.describe() for item in self.checks],
                "claim_boundary": (
                    "Conformance establishes that an adapter honours its own declaration. It "
                    "does not establish that the declaration describes the archive correctly, "
                    "and it is not a scientific result.")}


def _check(name: str, condition: bool, ok: str, bad: str,
           evidence: Optional[Mapping[str, Any]] = None) -> CheckResult:
    return CheckResult(name, PASS if condition else FAIL, ok if condition else bad,
                       evidence or {})


def run_conformance(adapter: DomainExperimentAdapter,
                    case: ConformanceCase) -> ConformanceReport:
    """Execute every check, collecting results rather than stopping at the first failure.

    A kit that aborted on the first failure would tell an adapter author one thing per run.
    Onboarding cost is what this slice is measured on, so the report is the whole list.
    """
    checks: List[CheckResult] = []
    record = case.native_record

    # ------------------------------------------------------------- controls and the schema
    try:
        resolved = adapter.resolve(case.parameters)
        again = adapter.resolve(resolved)
        rendered = {item["name"] for item in adapter.controls.describe()["fields"]}
        checks.append(_check(
            "controls_schema_agreement", again == resolved and set(resolved) <= rendered,
            "the rendered schema, the validated parameters and their re-validation agree",
            "resolved parameters are not stable under re-validation, or name a control the "
            "rendered schema does not declare",
            {"resolved": sorted(resolved), "rendered": sorted(rendered)}))
    except Exception as error:  # a refusal here is a genuine failure of the case, not a crash
        resolved = {}
        checks.append(CheckResult("controls_schema_agreement", FAIL,
                                  "declared controls refused their own case: %s" % error))

    # --------------------------------------------------------------------- acquisition plan
    plan = None
    try:
        plan = adapter.plan_acquisition(resolved, dict(case.identity))
    except Exception as error:
        checks.append(CheckResult("coverage_honesty", FAIL,
                                  "the acquisition planner raised: %s" % error))
        checks.append(CheckResult("bounded_resource_plan", FAIL, "no plan was produced"))

    if plan is not None:
        checks.append(_check(
            "coverage_honesty",
            not (plan.coverage_exact and plan.support_kind in INEXACT_SUPPORT),
            "support kind and the exactness claim agree",
            "support kind %r cannot establish exact coverage from an extent; an intersection "
            "says an observation exists somewhere in the window, not that the window is "
            "covered" % plan.support_kind,
            {"support_kind": plan.support_kind, "coverage_exact": plan.coverage_exact}))
        rows = plan_windows(plan, case.windows)
        planned = sum(int(row["estimated_bytes"]) for row in rows)
        checks.append(_check(
            "bounded_resource_plan", planned <= case.maximum_planned_bytes,
            "the plan prices %d bytes against a %d byte cap" % (planned,
                                                                case.maximum_planned_bytes),
            "the plan prices %d bytes, above its %d byte cap, and must refuse before "
            "acquisition rather than during it" % (planned, case.maximum_planned_bytes),
            {"planned_bytes": planned, "maximum_planned_bytes": case.maximum_planned_bytes,
             "windows": rows}))

    # ------------------------------------------------------------------------- translation
    declaration = None
    trajectory = None
    try:
        declaration = adapter.structural_declaration(resolved)
        trajectory = adapter.translate(record, declaration, resolved)
    except Exception as error:
        checks.append(CheckResult("deterministic_translation", FAIL,
                                  "translation raised: %s" % error))

    if trajectory is not None and declaration is not None:
        repeated = adapter.translate(record, adapter.structural_declaration(resolved), resolved)
        same = (repeated.trajectory_id == trajectory.trajectory_id
                and all(np.array_equal(repeated.channels[name], values)
                        for name, values in trajectory.channels.items()))
        checks.append(_check(
            "deterministic_translation", same,
            "the same native record and configuration produce the same canonical bytes",
            "translation is not deterministic; a run identity that does not reproduce cannot "
            "carry a result",
            {"trajectory_id": trajectory.trajectory_id}))

        checks.append(_check(
            "content_addressing",
            (trajectory.native_record_sha256 == record.content_sha256
             and trajectory.adapter_definition_sha256 == declaration.definition_sha256
             and trajectory.native_record_retained),
            "native, adapter and retention identities all reconstruct",
            "the canonical projection does not address the native record and adapter "
            "definition it was produced from",
            {"native_record_sha256": trajectory.native_record_sha256,
             "adapter_definition_sha256": trajectory.adapter_definition_sha256}))

        checks.append(_check(
            "axis_and_role_validation",
            bool(declaration.required_axes) and bool(declaration.required_roles)
            and declaration.domain == adapter.domain,
            "required axes and roles are declared and belong to this domain",
            "the structural declaration omits required axes or roles, or names a different "
            "domain than the adapter's own declaration",
            {"required_axes": list(declaration.required_axes),
             "required_roles": list(declaration.required_roles),
             "structural_domain": declaration.domain, "adapter_domain": adapter.domain}))

        try:
            assert_structural_conformance(trajectory, record, declaration,
                                          config=adapter.config(resolved))
            checks.append(CheckResult(
                "gap_preservation", PASS,
                "the native clock, interval support, validity mask and per-value lineage are "
                "reconstructed exactly"))
        except StructuralConformanceError as error:
            checks.append(CheckResult("gap_preservation", FAIL, str(error)))

        checks.append(_check(
            "refusal_propagation",
            (tuple(trajectory.assumption_violations) == tuple(record.assumption_violations)
             and (adapter.declaration.precedence_admissible
                  or all(item in declaration.refused_operations
                         for item in PRECEDENCE_REFUSALS))),
            "the domain's violations reach the canonical record and its lag policy's refusals "
            "are declared by the adapter",
            "a violation was dropped in translation, or a domain with no admissible lag floor "
            "did not refuse %s (R21)" % ", ".join(PRECEDENCE_REFUSALS),
            {"violations": list(trajectory.assumption_violations),
             "precedence_admissible": adapter.declaration.precedence_admissible,
             "refused_operations": list(declaration.refused_operations)}))

        checks.extend(_invariance_checks(adapter, declaration, record, trajectory, resolved))
        checks.append(_null_check(adapter, trajectory, case))

    return ConformanceReport(adapter.adapter_id, adapter.domain, adapter.definition_sha256,
                             tuple(checks))


def _invariance_checks(adapter: DomainExperimentAdapter, declaration: Any,
                       record: NativeStructuralRecord, trajectory: Any,
                       resolved: Mapping[str, Any]) -> List[CheckResult]:
    """Apply each declared invariance to the native record and compare the canonical output."""
    results: List[CheckResult] = []
    if not declaration.invariances:
        return [CheckResult("declared_invariances", NOT_APPLICABLE,
                            "this adapter declares no invariances")]
    for name in declaration.invariances:
        probe = INVARIANCE_PROBES.get(name)
        if probe is None:
            results.append(CheckResult(
                "declared_invariances:%s" % name, NOT_PROBED,
                "no executable probe is registered for this invariance, so the claim is "
                "declared and unverified rather than checked"))
            continue
        try:
            transformed = adapter.translate(probe(record), declaration, resolved)
            unchanged = all(
                np.allclose(transformed.channels[channel], values, rtol=0.0, atol=1e-9)
                for channel, values in trajectory.channels.items())
        except Exception as error:
            results.append(CheckResult("declared_invariances:%s" % name, FAIL,
                                       "the invariance probe raised: %s" % error))
            continue
        results.append(_check(
            "declared_invariances:%s" % name, unchanged,
            "the transform leaves every canonical channel unchanged, as declared",
            "the adapter declares this invariance and its canonical channels change under it"))
    return results


def _null_check(adapter: DomainExperimentAdapter, trajectory: Any,
                case: ConformanceCase) -> CheckResult:
    """A legitimate null moves the values and leaves the domain's own structure alone."""
    if adapter.build_null is None:
        return CheckResult("null_suitability", NOT_APPLICABLE,
                           "this adapter registers no null builder yet")
    try:
        null = adapter.build_null(trajectory, case.null_seed)
    except Exception as error:
        return CheckResult("null_suitability", FAIL, "the null builder raised: %s" % error)
    support_preserved = (
        np.array_equal(null.support_start_seconds, trajectory.support_start_seconds)
        and np.array_equal(null.support_end_seconds, trajectory.support_end_seconds)
        and np.array_equal(null.valid_mask, trajectory.valid_mask))
    moved = any(not np.array_equal(null.channels[name], values)
                for name, values in trajectory.channels.items())
    return _check(
        "null_suitability", support_preserved and moved,
        "the null preserves the native support and validity while moving the values",
        "a null that changes the clock or gaps tests a different record, and one that changes "
        "nothing is not a null",
        {"support_preserved": bool(support_preserved), "values_moved": bool(moved)})


def assert_conformance(adapter: DomainExperimentAdapter, case: ConformanceCase) -> ConformanceReport:
    """Run the kit and refuse a non-conformant adapter, naming every failure."""
    report = run_conformance(adapter, case)
    if not report.conformant:
        raise AdapterConformanceError(
            adapter.adapter_id, "conformance",
            "; ".join("%s: %s" % (item.name, item.detail) for item in report.failures))
    return report


__all__ = ["CheckResult", "ConformanceCase", "ConformanceReport", "FAIL", "INEXACT_SUPPORT",
           "INVARIANCE_PROBES", "NOT_APPLICABLE", "NOT_PROBED", "PASS", "PRECEDENCE_REFUSALS",
           "SCHEMA", "assert_conformance", "run_conformance"]
