"""Immutable configuration and metadata preflight for G17 experiments.

The manifest is the sole scientific configuration carried by the future orchestrator.  This
slice deliberately plans metadata only: it does not open measurement values, translate a native
record, run a statistic, or create evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator

from src.core.publication import publish_new_bytes


SCHEMA = "cross-domain-experiment/v1"
PREFLIGHT_SCHEMA = "cross-domain-experiment-preflight/v1"
STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FrozenModel(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ExplicitWindow(FrozenModel):
    name: str = Field(..., min_length=1, max_length=64)
    start_utc: datetime
    end_utc: datetime
    stride_seconds: int = Field(..., gt=0)

    @root_validator
    def _ordered_utc_window(cls, values):
        start, end = values.get("start_utc"), values.get("end_utc")
        if start is not None and (start.tzinfo is None or start.utcoffset() is None):
            raise ValueError("start_utc must include a UTC offset")
        if end is not None and (end.tzinfo is None or end.utcoffset() is None):
            raise ValueError("end_utc must include a UTC offset")
        if start is not None and end is not None and end <= start:
            raise ValueError("end_utc must be later than start_utc")
        return values


class AcquisitionIdentity(FrozenModel):
    source_id: str = Field(..., min_length=1)
    source_version: str = Field(..., min_length=1)
    identity: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AdapterIdentity(FrozenModel):
    adapter_id: str = Field(..., min_length=1)
    adapter_version: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ObservationContract(FrozenModel):
    domain: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    measure: str = Field(..., min_length=1)
    semantics: str = Field(..., min_length=1)
    units: str = Field(..., min_length=1)
    acquisition: AcquisitionIdentity
    adapter: AdapterIdentity


class CoveragePolicy(FrozenModel):
    requirement: Literal["complete_required", "partial_permitted"] = "complete_required"
    minimum_fraction: float = Field(1.0, gt=0.0, le=1.0)

    @root_validator
    def _complete_means_complete(cls, values):
        if values.get("requirement") == "complete_required" and values.get("minimum_fraction") != 1.0:
            raise ValueError("complete_required has minimum_fraction 1.0")
        return values


class FamilyDefinition(FrozenModel):
    channels: List[str] = Field(..., min_items=1)
    scales: List[float] = Field(..., min_items=1)
    relationships: List[str] = Field(..., min_items=1)
    maximum_members: int = Field(..., gt=0)

    @validator("channels", "relationships")
    def _unique_labels(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("family labels must be unique")
        return value


class NullDefinition(FrozenModel):
    name: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    replications: int = Field(..., ge=20)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ResourceCaps(FrozenModel):
    maximum_family_members: int = Field(..., gt=0)
    maximum_planned_bytes: int = Field(..., gt=0)
    maximum_runtime_seconds: int = Field(..., gt=0)


class CrossDomainExperimentSpec(FrozenModel):
    schema_id: Literal[SCHEMA] = SCHEMA
    study_id: str
    title: str = Field(..., min_length=1, max_length=200)
    mode: Literal["calendar_aligned", "scale_shape_aligned"]
    windows: List[ExplicitWindow] = Field(..., min_items=1)
    observations: List[ObservationContract] = Field(..., min_items=2)
    coverage_policy: CoveragePolicy
    scale_normalization: Optional[Dict[str, Any]] = None
    family: FamilyDefinition
    nulls: List[NullDefinition] = Field(..., min_items=1)
    correction: Literal["benjamini_yekutieli", "holm", "bonferroni"]
    alpha: float = Field(..., gt=0.0, lt=1.0)
    seeds: Dict[str, int]
    resource_caps: ResourceCaps
    notes: Dict[str, Any] = Field(default_factory=dict)

    @validator("study_id")
    def _safe_study_id(cls, value):
        if not STUDY_ID.fullmatch(value):
            raise ValueError("study_id must be a safe 1-128 character identifier")
        return value

    @root_validator
    def _complete_and_unambiguous(cls, values):
        observations = values.get("observations") or []
        domains = [item.domain for item in observations]
        if len(domains) != len(set(domains)):
            raise ValueError("an experiment may declare each domain only once")
        windows = values.get("windows") or []
        names = [item.name for item in windows]
        if len(names) != len(set(names)):
            raise ValueError("window names must be unique")
        if values.get("mode") == "scale_shape_aligned" and not values.get("scale_normalization"):
            raise ValueError("scale_shape_aligned requires scale_normalization")
        family, caps = values.get("family"), values.get("resource_caps")
        if family and caps and family.maximum_members > caps.maximum_family_members:
            raise ValueError("family maximum exceeds the experiment resource cap")
        if not values.get("seeds"):
            raise ValueError("at least one labelled seed is required")
        return values


def canonical_bytes(spec: CrossDomainExperimentSpec) -> bytes:
    """One byte-stable representation for API, saved recipe, run identity and receipt."""
    return json.dumps(spec.dict(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_value).encode("utf-8")


def _json_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise TypeError("not JSON serializable: %r" % (value,))


def manifest_sha256(spec: CrossDomainExperimentSpec) -> str:
    return hashlib.sha256(canonical_bytes(spec)).hexdigest()


def manifest_envelope(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    raw = canonical_bytes(spec)
    return {"schema": SCHEMA, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "run_identity": "g17:%s" % hashlib.sha256(raw).hexdigest(),
            "canonical_manifest": json.loads(raw), "immutable": True}


def _flagship_observations() -> List[Dict[str, Any]]:
    """TG17.0's quartet, addressed through the adapters registered for each domain.

    The adapter identities are real registrations since TG17.3, not the ``planned-tg17.2``
    placeholders TG17.1 carried, and each domain's request parameters live in the adapter's own
    declared controls rather than in a free-form source dictionary. A manifest whose parameters
    its adapter would refuse now refuses at preflight instead of at run time.
    """
    rows = [
        ("reanalysis", "ERA5 reanalysis", "air_temperature", "air temperature", "K",
         "grid_crop:era5_0p25_1h_full37", "CDS-era5", "reanalysis.standardized-level",
         {"variable": "temperature", "level_hpa": 850}),
        ("argo_float", "Argo floats", "salinity", "practical salinity", "1e-3",
         "profile_query:argo_gdac_erddap", "Argo-GDAC", "argo_float.standardized-level",
         {"pressure_dbar": 100.0}),
        ("tess_lightcurve", "TESS light curves", "relative_flux", "relative stellar flux",
         "dimensionless", "lightcurve_query:mast_tess_spoc", "MAST-TESS-SPOC",
         "tess_lightcurve.standardized-level", {"cadence": "short"}),
        ("order_book", "Order book", "aggregated_volume", "aggregated traded volume", "shares",
         "channel_table:local", "user-supplied-v1", "order_book.bespoke_record",
         {"time_column": "timestamp"}),
    ]
    return [{"domain": domain, "label": label, "role": "structural_observation",
             "measure": measure, "semantics": semantics, "units": units,
             "acquisition": {"source_id": source, "source_version": version,
                             "identity": {"licence_scope": "source-declared"},
                             "parameters": {}},
             "adapter": {"adapter_id": adapter_id, "adapter_version": "tg17.3-v1",
                         "parameters": params}}
            for domain, label, measure, semantics, units, source, version, adapter_id, params
            in rows]


def flagship_recipe() -> CrossDomainExperimentSpec:
    """The TG17.0 quartet expressed in the first real manifest schema."""
    return CrossDomainExperimentSpec.parse_obj({
        "study_id": "g17_flagship_calendar_v1",
        "title": "Four-domain shared-window structural experiment",
        "mode": "calendar_aligned",
        "windows": [
            {"name": "week", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-01-08T00:00:00Z", "stride_seconds": 86400},
            {"name": "three_months", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-04-01T00:00:00Z", "stride_seconds": 604800},
            {"name": "six_months", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-07-01T00:00:00Z", "stride_seconds": 1209600},
        ],
        "observations": _flagship_observations(),
        "coverage_policy": {"requirement": "partial_permitted", "minimum_fraction": 0.5},
        "family": {"channels": ["energy_concentration", "persistence", "entropy", "change_point"],
                   "scales": [1.0, 2.0, 4.0, 8.0], "relationships": ["co_occurrence"],
                   "maximum_members": 10000},
        "nulls": [{"name": "domain_preserving_shift", "method": "independent_native_clock_shift",
                   "replications": 200, "parameters": {"preserve_gaps": True}}],
        "correction": "benjamini_yekutieli", "alpha": 0.05,
        "seeds": {"family": 20260830, "nulls": 20260831},
        "resource_caps": {"maximum_family_members": 10000,
                          "maximum_planned_bytes": 4294967296,
                          "maximum_runtime_seconds": 3600},
        "notes": {"contract": "TG17.0 flagship", "claim_boundary": "co-occurrence only"},
    })


def preflight_manifest(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """Plan archive coverage without network access or measurement-value reads.

    Since TG17.3 every row comes from the domain's registered `DomainExperimentAdapter`. The
    literal ``source_plans`` table this replaced was a fifth place a new domain had to be
    remembered; it is now a place a new domain cannot be forgotten, because a manifest naming a
    domain with no registered adapter refuses by name instead of falling through to a default.
    """
    from src.adapters import register_all_adapters
    from src.core.experiment_adapter import adapter_for_domain, plan_windows

    register_all_adapters()
    rows: List[Dict[str, Any]] = []
    refusals: List[Dict[str, Any]] = []
    for observation in spec.observations:
        row, refusal = _preflight_observation(spec, observation, adapter_for_domain, plan_windows)
        rows.append(row)
        if refusal:
            refusals.append(refusal)
    family_members = (len(spec.observations) * (len(spec.observations) - 1) // 2
                      * len(spec.family.channels) * len(spec.family.scales)
                      * len(spec.windows) * len(spec.family.relationships))
    if family_members > spec.resource_caps.maximum_family_members:
        refusals.append({"domain": None, "reason": "declared family has %d members; cap is %d" %
                         (family_members, spec.resource_caps.maximum_family_members)})
    planned_bytes = sum(int(window["estimated_bytes"])
                        for row in rows for window in row.get("windows", []))
    if planned_bytes > spec.resource_caps.maximum_planned_bytes:
        refusals.append({"domain": None, "reason":
                         "declared windows plan %d bytes; cap is %d" %
                         (planned_bytes, spec.resource_caps.maximum_planned_bytes)})
    partial = [row["domain"] for row in rows if row["status"] == "PARTIAL"]
    if partial and spec.coverage_policy.requirement == "complete_required":
        refusals.append({"domain": None, "reason":
                         "complete coverage is required but metadata cannot establish it for %s"
                         % ", ".join(partial)})
    return {"schema": PREFLIGHT_SCHEMA, "manifest_sha256": manifest_sha256(spec),
            "status": "REFUSED" if refusals else ("PARTIAL" if partial else "READY"),
            "metadata_only": True, "network_used": False, "measurement_values_opened": False,
            "family": {"declared_members": family_members,
                       "maximum_members": spec.resource_caps.maximum_family_members,
                       "windows_are_one_family": len(spec.windows) > 1},
            "planned_bytes": planned_bytes,
            "maximum_planned_bytes": spec.resource_caps.maximum_planned_bytes,
            "coverage": rows, "refusals": refusals,
            "claim_boundary": "Coverage planning is not acquisition, analysis, evidence or a scientific result."}


def _refused_row(observation: Any, reason: str) -> Any:
    return ({"domain": observation.domain, "label": observation.label,
             "source_id": observation.acquisition.source_id, "measure": observation.measure,
             "status": "REFUSED", "reason": reason, "opens_measurement_values": False,
             "windows": []},
            {"domain": observation.domain, "reason": reason})


def _preflight_observation(spec: CrossDomainExperimentSpec, observation: Any,
                           adapter_for_domain: Any, plan_windows: Any) -> Any:
    """One domain's coverage row, or the exact reason it has none."""
    from src.core.errors import SpectralEarthError

    try:
        adapter = adapter_for_domain(observation.domain)
    except SpectralEarthError as error:
        return _refused_row(observation, str(error))
    try:
        parameters = adapter.resolve(observation.adapter.parameters)
    except SpectralEarthError as error:
        return _refused_row(observation, "declared adapter parameters were refused: %s" % error)
    if observation.adapter.adapter_id != adapter.adapter_id:
        return _refused_row(observation, (
            "the manifest names adapter %r for this domain and %r is registered. A result must "
            "record which translation produced it, so the manifest is not silently reassigned"
            % (observation.adapter.adapter_id, adapter.adapter_id)))
    plan = adapter.plan_acquisition(parameters, dict(observation.acquisition.identity))
    if plan.source_id != observation.acquisition.source_id:
        return _refused_row(observation, (
            "the manifest names source %r and this adapter plans %r; the manifest addresses an "
            "archive its own adapter does not reach"
            % (observation.acquisition.source_id, plan.source_id)))
    if plan.refusal:
        row, refusal = _refused_row(observation, plan.refusal)
        row.update({"support_kind": plan.support_kind, "access": plan.access,
                    "native_cadence_seconds": plan.native_cadence_seconds})
        return row, refusal
    status = "READY" if plan.coverage_exact else "PARTIAL"
    reason = ("requested UTC extent is addressable from metadata" if plan.coverage_exact else
              "native support is sparse or sector-bounded; exact coverage is established only "
              "after acquisition")
    return ({"domain": observation.domain, "label": observation.label,
             "source_id": plan.source_id, "measure": observation.measure, "status": status,
             "reason": reason, "support_kind": plan.support_kind,
             "native_cadence_seconds": plan.native_cadence_seconds, "access": plan.access,
             "access_means": plan.access_means,
             "adapter": {"adapter_id": adapter.adapter_id,
                         "adapter_version": adapter.adapter_version,
                         "definition_sha256": adapter.definition_sha256,
                         "parameters": _jsonable(parameters)},
             "native_addressing": {"source_version": plan.source_version,
                                   "identity": dict(plan.identity),
                                   "parameters": _jsonable(parameters)},
             "opens_measurement_values": plan.opens_measurement_values,
             "windows": plan_windows(plan, spec.windows)}, None)


def _jsonable(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {name: (_json_value(value) if isinstance(value, datetime) else value)
            for name, value in parameters.items()}


class ManifestStore:
    """Content-addressed immutable manifests plus mutable pointers to saved drafts."""
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(self, draft_id: str, spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
        if not STUDY_ID.fullmatch(draft_id):
            raise ValueError("draft_id must be a safe identifier")
        digest, raw = manifest_sha256(spec), canonical_bytes(spec)
        manifests = self.root / "manifests"
        drafts = self.root / "drafts"
        manifests.mkdir(parents=True, exist_ok=True)
        drafts.mkdir(parents=True, exist_ok=True)
        path = manifests / (digest + ".json")
        try:
            publish_new_bytes(path, raw, "experiment manifest")
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError("content-address collision")
        pointer = json.dumps({"draft_id": draft_id, "manifest_sha256": digest}, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        pointer_path = drafts / (draft_id + ".json")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % pointer_path.name,
                                                       suffix=".tmp", dir=str(drafts))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(pointer)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, pointer_path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        return {"draft_id": draft_id, **manifest_envelope(spec), "saved": True}

    def load_manifest(self, digest: str) -> CrossDomainExperimentSpec:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid manifest digest")
        return CrossDomainExperimentSpec.parse_raw((self.root / "manifests" / (digest + ".json")).read_bytes())

    def load_draft(self, draft_id: str) -> CrossDomainExperimentSpec:
        if not STUDY_ID.fullmatch(draft_id):
            raise ValueError("invalid draft identifier")
        pointer = json.loads((self.root / "drafts" / (draft_id + ".json")).read_text("utf-8"))
        return self.load_manifest(pointer["manifest_sha256"])


__all__ = ["CrossDomainExperimentSpec", "ManifestStore", "SCHEMA", "canonical_bytes",
           "flagship_recipe", "manifest_envelope", "manifest_sha256", "preflight_manifest"]
