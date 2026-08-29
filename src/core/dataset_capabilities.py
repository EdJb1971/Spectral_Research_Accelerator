"""Record-bound scientific capability profiles.

An operation is available because the admitted representation satisfies registered
requirements, never because the UI recognised a source name.  Profiles bind those decisions to
the exact dataset/declaration identity that produced them.  The backend remains authoritative:
this contract guides the researcher but does not replace entry-point refusals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError


PROFILE_SCHEMA = "spectral.dataset-capability-profile.v1"
PROFILE_PHASES = ("probed", "declared", "planned", "acquired", "admitted")


@dataclass(frozen=True)
class Requirement:
    fact: str
    label: str
    unavailable: str
    unknown: str
    false_status: str = "unavailable"


@dataclass(frozen=True)
class Operation:
    name: str
    description: str
    requirements: Tuple[Requirement, ...]


def _need(fact: str, label: str, unavailable: str, unknown: str,
          false_status: str = "unavailable") -> Requirement:
    return Requirement(fact, label, unavailable, unknown, false_status)


OPERATIONS: Dict[str, Operation] = {
    "representation_audit": Operation(
        "Representation Audit", "Fixed-family target/feature representation audit.", (
            _need("sample_table", "declared sample table",
                  "This representation is not an explicitly declared sample table.",
                  "Declare the dataset representation before auditing it."),
            _need("independent_samples", "independent samples",
                  "The first safe recipe uses a row-random generate/confirm split and is only "
                  "admissible for independent samples. Grouped data need group-held-out "
                  "confirmation; ordered data need blocked and embargoed confirmation.",
                  "Declare whether samples are independent, grouped or ordered."),)),
    "association_redundancy": Operation(
        "Association and redundancy", "Contemporaneous association over independent samples.", (
            _need("sample_table", "declared sample table",
                  "This action belongs to the independent-sample table line.",
                  "Declare the dataset representation first."),
            _need("independent_samples", "independent samples",
                  "This recipe does not account for grouped or ordered dependence.",
                  "Declare how samples are related."),)),
    "cross_domain_analysis": Operation(
        "Cross-domain analysis", "Association and declared-lag analysis of channel series.", (
            _need("channel_series", "admitted channel series",
                  "Cross-domain analysis requires a full admitted channel record.",
                  "Admit a channel record first."),
            _need("ordered_time_axis", "ordered time axis",
                  "The record has no declared ordered time axis.",
                  "Declare the record's time axis."),
            _need("regular_cadence", "regular cadence",
                  "The current engine measures lags in frames and refuses an irregular clock.",
                  "Verify the record cadence before analysis.", "insufficient_support"),)),
    "advection_precedence": Operation(
        "Advection-based precedence", "Lead/lag analysis under the domain's declared policy.", (
            _need("channel_series", "admitted channel series",
                  "Precedence requires an admitted channel record.",
                  "Admit a channel record first."),
            _need("ordered_time_axis", "ordered time axis",
                  "Precedence is meaningless without declared temporal order.",
                  "Declare the time axis."),
            _need("regular_cadence", "regular cadence",
                  "A shift in rows is not a duration on this irregular clock.",
                  "Verify cadence before choosing a lag.", "insufficient_support"),
            _need("precedence_admissible", "declared lag policy",
                  "This domain declares no admissible lag floor, so R21 refuses precedence.",
                  "Bind the dataset to an onboarded domain and lag policy."),)),
    "crop_planning": Operation(
        "Crop planning", "Plan a bounded native-coordinate grid selection.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "Crop planning applies only to a declared 2D grid.",
                  "Inspect or declare the dataset geometry."),)),
    "boundary_lab": Operation(
        "Boundary-condition lab", "Diagnose boundary treatments on a 2D field.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "This dataset has no declared 2D grid.",
                  "Inspect or declare the dataset geometry."),)),
    "dtcwt_spatial": Operation(
        "DTCWT spatial analysis", "Directional multiscale analysis on a supported 2D crop.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "This dataset has no declared 2D grid.",
                  "Inspect or declare the dataset geometry."),
            _need("transform_compatible", "sufficient transform support",
                  "The selected crop does not meet the requested transform's support contract.",
                  "Inspect the exact crop and transform first.", "insufficient_support"),)),
    "radial_psd": Operation(
        "Radial physical PSD", "Radial spectrum in physical wavenumber units.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "A radial spatial spectrum requires a declared 2D grid.",
                  "Inspect or declare the dataset geometry."),
            _need("physical_metric", "physical metric",
                  "This dataset has no declared physical length metric, so physical "
                  "wavenumber is unavailable.",
                  "Bind a geometry with an explicit physical metric."),)),
    "gridded_diagnostics": Operation(
        "Gridded diagnostics", "Spatial diagnostics over a 2D field.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "These diagnostics are spatial-grid specific.",
                  "Inspect or declare the dataset geometry."),)),
    "structure_mining": Operation(
        "Structure mining", "Mine spatial structures under sealed geometry.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "Structure mining currently requires spatial field geometry.",
                  "Admit a spatial field first."),)),
    "forecast_evaluation": Operation(
        "Forecast evaluation", "Evaluate gridded forecast fields.", (
            _need("spatial_grid_2d", "declared 2D spatial grid",
                  "Forecast evaluation currently belongs to the gridded field line.",
                  "Admit a gridded forecast field first."),)),
    "profile_reduction": Operation(
        "Profile reduction", "Registered reductions over native irregular profiles.", (
            _need("profile_collection", "native profile collection",
                  "Profile reductions require a native profile collection.",
                  "Acquire profiles first."),
            _need("irregular_support", "preserved irregular support",
                  "This record does not declare native irregular profile support.",
                  "Verify the collection support."),)),
}


CORE_FACTS: Tuple[Tuple[str, str], ...] = (
    ("spatial_grid_2d", "2D spatial grid"),
    ("physical_metric", "Physical metric"),
    ("ordered_time_axis", "Ordered time axis"),
    ("regular_cadence", "Regular cadence"),
    ("irregular_support", "Irregular support preserved"),
    ("transform_compatible", "Requested transform compatible"),
    ("precedence_admissible", "Lag analysis admissible"),
    ("independent_samples", "Independent samples"),
)


def _decision(operation: Operation, facts: Mapping[str, Optional[bool]]) -> Dict[str, Any]:
    checks = []
    for requirement in operation.requirements:
        value = facts.get(requirement.fact)
        checks.append({"fact": requirement.fact, "label": requirement.label,
                       "satisfied": value})
        if value is None:
            return {"status": "needs_declaration", "available": False,
                    "reason_code": "unknown_%s" % requirement.fact,
                    "reason": requirement.unknown, "requirements": checks}
        if value is not True:
            return {"status": requirement.false_status, "available": False,
                    "reason_code": "requires_%s" % requirement.fact,
                    "reason": requirement.unavailable, "requirements": checks}
    return {"status": "available", "available": True, "reason_code": "requirements_met",
            "reason": "Available — every declared and verified requirement is satisfied.",
            "requirements": checks}


def build_profile(*, kind: str, phase: str, identity: str,
                  facts: Mapping[str, Optional[bool]], domain: Optional[str] = None,
                  basis: Optional[Mapping[str, Any]] = None,
                  operations: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build a canonical profile bound to an exact record, collection or request identity."""
    if not str(kind).strip() or not str(identity).strip():
        raise InvalidParameterError("capability profile identity", [kind, identity],
                                    "non-empty representation kind and dataset identity")
    if phase not in PROFILE_PHASES:
        raise InvalidParameterError("capability profile phase", phase,
                                    "one of %s" % (PROFILE_PHASES,))
    unknown_facts = sorted(set(facts) - {name for name, _label in CORE_FACTS}
                           - {"sample_table", "channel_series", "profile_collection"})
    if unknown_facts:
        raise InvalidParameterError("capability facts", unknown_facts,
                                    "registered capability facts")
    invalid_values = {name: value for name, value in facts.items()
                      if value is not True and value is not False and value is not None}
    if invalid_values:
        raise InvalidParameterError("capability fact values", invalid_values,
                                    "True, False or None for every fact")
    names = tuple(operations) if operations is not None else tuple(OPERATIONS)
    unknown_operations = sorted(set(names) - set(OPERATIONS))
    if unknown_operations:
        raise InvalidParameterError("capability operations", unknown_operations,
                                    "registered operations")
    normalized = {name: facts.get(name) for name, _label in CORE_FACTS}
    normalized.update({name: facts.get(name)
                       for name in ("sample_table", "channel_series", "profile_collection")})
    body = {
        "schema": PROFILE_SCHEMA, "kind": kind, "phase": phase, "identity": identity,
        "domain": domain, "facts": normalized,
        "capabilities": [{"name": name, "label": label, "value": normalized[name]}
                         for name, label in CORE_FACTS],
        "operations": {name: {"name": OPERATIONS[name].name,
                              "description": OPERATIONS[name].description,
                              **_decision(OPERATIONS[name], normalized)} for name in names},
        "basis": dict(basis or {}),
        "claim_boundary": (
            "This profile routes an exact declared representation to admissible operations. "
            "It is not evidence that an operation ran or that a scientific claim is true; "
            "backend entry points repeat their own authoritative refusals."),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    body["profile_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return body


__all__ = ["CORE_FACTS", "OPERATIONS", "PROFILE_SCHEMA", "build_profile"]
