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
ALIGNMENT_SCHEMA = "experiment-preflight-alignment/v1"
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
    """Every axis of the declared search, so the family can be priced before acquisition.

    The axes a study is actually free to vary are the axes it must declare. Since TG17.5 that
    includes which domain combinations are tested, at which lags, in which representations and
    over which motifs: each was a knob that could be turned after seeing a result, and a family
    priced without it is short by exactly the factor nobody wrote down.
    """

    channels: List[str] = Field(..., min_items=1)
    scales: List[float] = Field(..., min_items=1)
    relationships: List[str] = Field(..., min_items=1)
    #: Which sizes of domain combination are tested. Pairs by default; a study that also tests
    #: triples has run both searches and pays for both.
    domain_arities: List[int] = Field(default_factory=lambda: [2], min_items=1)
    lags_seconds: List[float] = Field(default_factory=lambda: [0.0], min_items=1)
    representations: List[str] = Field(default_factory=lambda: ["canonical"], min_items=1)
    motifs: List[str] = Field(default_factory=lambda: ["none"], min_items=1)
    maximum_members: int = Field(..., gt=0)

    @validator("channels", "relationships", "representations", "motifs")
    def _unique_labels(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("family labels must be unique")
        return value

    @validator("domain_arities", "lags_seconds", "scales")
    def _unique_numbers(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("family axis values must be unique; a repeated value is a "
                             "member counted twice and a test run once")
        return value


class NullDefinition(FrozenModel):
    """One declared surrogate construction, named from the registry rather than described.

    `method` must be a registered, admissible `NullFamily`. Free text here would let a study
    declare "domain-preserving shuffle" and leave what it preserved to the reader; the registry
    states the preserved features, and TG17.5 checks the family against the mode and against
    every participating adapter before the manifest is frozen.
    """

    name: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    replications: int = Field(..., ge=20)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("method")
    def _registered_and_admissible(cls, value):
        from src.core.structural_nulls import NULL_FAMILIES

        if value not in NULL_FAMILIES:
            raise ValueError("null method %r is not a registered family; registered: %s"
                             % (value, NULL_FAMILIES.names()))
        family = NULL_FAMILIES.get(value)
        if not family.admissible:
            raise ValueError("null method %r is refused: %s"
                             % (value, family.inadmissible_reason))
        return value


class AlignmentPolicy(FrozenModel):
    """How support is compared, frozen before any measurement value is opened (TG17.4).

    The default transforms nothing. Every other kernel widens, snaps or carries support, which
    is a scientific choice: it is named here, its consequence in seconds is reported in
    preflight, and every participating adapter must admit it.
    """

    kernel: str = "exact_support_overlap"
    parameters: Dict[str, float] = Field(default_factory=dict)
    minimum_overlap_seconds: float = Field(0.0, ge=0.0)
    minimum_effective_samples: float = Field(0.0, ge=0.0)


class ConfirmationPolicy(FrozenModel):
    """Which stage this manifest declares, and therefore what R18 corrects over (TG17.5).

    A four-domain search is large, and rule R18 admits exactly two remedies for a family the
    declared ensemble cannot resolve: declare fewer members before the pass runs, or take
    TG3.2's generate/confirm split. The split is not a cheaper family — the generate stage
    still enumerates and still tests every declared member — it moves the correction onto a
    confirmatory family frozen under a content hash *before* a held-out partition is opened.

    Declaring the stage is what makes the difference checkable. A `confirmatory_only` study is
    priced at its complete declared family and refused if it cannot resolve it. A
    `generate_then_confirm` study is priced at the confirmatory family it names here, and its
    generate stage is marked as producing candidates rather than claims — which is the honest
    description of an uncorrected pass whose selection used the data.
    """

    stage: Literal["confirmatory_only", "generate_then_confirm"] = "confirmatory_only"
    #: The partition the confirmation runs on. Named in the manifest so it can be shown to
    #: have been closed while the candidates were chosen, which is the only thing it claims.
    held_out_partition: Optional[str] = None
    confirmatory_members: Optional[int] = Field(None, gt=0)

    @root_validator
    def _split_declares_what_it_needs(cls, values):
        stage = values.get("stage")
        partition = (values.get("held_out_partition") or "").strip()
        members = values.get("confirmatory_members")
        if stage == "generate_then_confirm":
            if not partition:
                raise ValueError(
                    "a generate/confirm split must name its held-out partition. An unnamed "
                    "one cannot be shown to have been closed while the candidates were "
                    "chosen, and correcting over a small confirmatory family is legitimate "
                    "only because it was")
            if not members:
                raise ValueError(
                    "a generate/confirm split must declare how many members it will confirm. "
                    "That number is the correction unit of the only stage that makes claims, "
                    "and choosing it after seeing the candidates is the move R18 forbids")
        elif partition or members:
            raise ValueError(
                "a confirmatory_only study declares neither a held-out partition nor a "
                "confirmatory member count; it corrects over its complete declared family")
        return values


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
    alignment: AlignmentPolicy = Field(default_factory=AlignmentPolicy)
    scale_normalization: Optional[Dict[str, Any]] = None
    family: FamilyDefinition
    nulls: List[NullDefinition] = Field(..., min_items=1)
    confirmation: ConfirmationPolicy = Field(default_factory=ConfirmationPolicy)
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
        _assert_relationships_belong_to_mode(values)
        _assert_nulls_belong_to_mode(values)
        _assert_arities_fit_the_declared_domains(values)
        family, caps = values.get("family"), values.get("resource_caps")
        if family and caps and family.maximum_members > caps.maximum_family_members:
            raise ValueError("family maximum exceeds the experiment resource cap")
        if not values.get("seeds"):
            raise ValueError("at least one labelled seed is required")
        return values


def _assert_relationships_belong_to_mode(values: Dict[str, Any]) -> None:
    """A mode may not borrow the other mode's vocabulary (TG17.4).

    Calendar mode compares UTC support and may speak of co-occurrence; it may not silently
    search normalized scale ratios. Scale/shape mode compares a normalized coordinate that
    carries no clock, and may not emit simultaneity, precedence or causal language at any
    confidence. Refused here rather than at the point a result is worded, because by then the
    search has already happened.
    """
    from src.core.structural_alignment import (AlignmentRefusal,
                                               assert_mode_admits_relationship)

    mode, family = values.get("mode"), values.get("family")
    if not mode or family is None:
        return
    for relationship in family.relationships:
        try:
            assert_mode_admits_relationship(mode, relationship)
        except AlignmentRefusal as error:
            raise ValueError(str(error))


def _assert_nulls_belong_to_mode(values: Dict[str, Any]) -> None:
    """A calendar null is no null for a question that never referred to a clock (TG17.5)."""
    from src.core.structural_nulls import NullRefusal, assert_null_admits_mode, NULL_FAMILIES

    mode, nulls = values.get("mode"), values.get("nulls") or []
    if not mode:
        return
    for null in nulls:
        try:
            assert_null_admits_mode(null.method, mode)
            NULL_FAMILIES.get(null.method).resolve(null.parameters)
        except NullRefusal as error:
            raise ValueError(str(error))


def _assert_arities_fit_the_declared_domains(values: Dict[str, Any]) -> None:
    """A triple cannot be formed from two domains, and neither can it be priced."""
    from src.core.errors import InvalidParameterError
    from src.core.experiment_family import domain_set_labels

    family, observations = values.get("family"), values.get("observations") or []
    if family is None or not observations:
        return
    try:
        domain_set_labels([item.domain for item in observations], family.domain_arities)
    except InvalidParameterError as error:
        raise ValueError(str(error))


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
        # `preserve_gaps` was declared here and read by nothing: the family preserves gaps by
        # construction, and TG17.5's registry says so in `preserves` where a reader can check
        # it. A parameter a researcher believes is in force and is not is worse than none.
        "nulls": [{"name": "domain_preserving_shift", "method": "independent_native_clock_shift",
                   "replications": 200}],
        # 288 declared tests cannot be corrected at 200 replications: rejecting one member
        # of that family under Benjamini-Yekutieli needs about 35,953 surrogates. TG17.5's
        # accounting made that visible before acquisition, which is what it is for. The
        # flagship therefore declares what it actually is - a search that mines its complete
        # family for candidates and confirms a frozen handful on a partition held closed
        # throughout. Four members is exactly what 200 replications can resolve.
        "confirmation": {"stage": "generate_then_confirm",
                         "held_out_partition": "g17_flagship_heldout_v1",
                         "confirmatory_members": 4},
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
    family = _preflight_family(spec, refusals)
    family_members = int(family["declared_members"])
    planned_bytes = sum(int(window["estimated_bytes"])
                        for row in rows for window in row.get("windows", []))
    if planned_bytes > spec.resource_caps.maximum_planned_bytes:
        refusals.append({"domain": None, "reason":
                         "declared windows plan %d bytes; cap is %d" %
                         (planned_bytes, spec.resource_caps.maximum_planned_bytes)})
    alignment, alignment_refusals = _preflight_alignment(spec, rows)
    refusals.extend(alignment_refusals)
    refusals.extend(_preflight_nulls(spec))
    partial = [row["domain"] for row in rows if row["status"] == "PARTIAL"]
    if partial and spec.coverage_policy.requirement == "complete_required":
        refusals.append({"domain": None, "reason":
                         "complete coverage is required but metadata cannot establish it for %s"
                         % ", ".join(partial)})
    return {"schema": PREFLIGHT_SCHEMA, "manifest_sha256": manifest_sha256(spec),
            "status": "REFUSED" if refusals else ("PARTIAL" if partial else "READY"),
            "metadata_only": True, "network_used": False, "measurement_values_opened": False,
            "family": family,
            "planned_bytes": planned_bytes,
            "maximum_planned_bytes": spec.resource_caps.maximum_planned_bytes,
            "alignment": alignment,
            "coverage": rows, "refusals": refusals,
            "claim_boundary": "Coverage planning is not acquisition, analysis, evidence or a scientific result."}


def _preflight_family(spec: CrossDomainExperimentSpec,
                      refusals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Price the complete declared search, and refuse it here rather than after acquisition.

    Two separate refusals, because they are two different problems. A family over the resource
    cap is too expensive to run; a family the declared ensemble cannot resolve is one that will
    run happily, cost the full amount and be arithmetically incapable of rejecting anything -
    which is the failure R18 exists for, and it is indistinguishable afterwards from a clean
    negative result. Both are decided from metadata, before a byte is acquired.
    """
    from src.core.experiment_family import family_expansion

    expansion = family_expansion(spec)
    declared = int(expansion["family_size"])
    if declared > spec.resource_caps.maximum_family_members:
        refusals.append({"domain": None, "reason": "declared family has %d members; cap is %d"
                         % (declared, spec.resource_caps.maximum_family_members)})
    correction = expansion["correction"]
    if not correction["affordable"]:
        refusals.append({"domain": None, "reason":
                         "R18: %d corrected tests at %d surrogates cannot reject anything under "
                         "%s at alpha=%.3g; about %d surrogates are needed, or the declaration "
                         "must lose members before the pass runs (the largest affordable family "
                         "at this ensemble is %d). A generate/confirm split against a named "
                         "held-out partition is the other remedy rule R18 admits."
                         % (correction["correction_unit_members"],
                            expansion["declared_surrogates"], spec.correction,
                            spec.alpha, correction["surrogates_required"],
                            expansion["largest_affordable_family"])})
    return {**expansion,
            "declared_members": declared,
            "maximum_members": spec.resource_caps.maximum_family_members,
            "windows_are_one_family": len(spec.windows) > 1}


def _preflight_nulls(spec: CrossDomainExperimentSpec) -> List[Dict[str, Any]]:
    """Every declared null, bound against every participating adapter (TG17.5)."""
    from src.core.errors import SpectralEarthError
    from src.core.experiment_adapter import adapter_for_domain
    from src.core.structural_nulls import NullRefusal, bind_null

    admissible: Dict[str, Any] = {}
    for observation in spec.observations:
        try:
            adapter = adapter_for_domain(observation.domain)
        except SpectralEarthError:
            continue  # an unregistered domain is already refused by its coverage row
        admissible[adapter.adapter_id] = adapter.admissible_nulls
    refusals: List[Dict[str, Any]] = []
    for null in spec.nulls:
        try:
            bind_null(null.method, null.parameters, mode=spec.mode,
                      admissible_by_adapter=admissible)
        except NullRefusal as error:
            refusals.append({"domain": None, "reason": str(error)})
    return refusals


def _preflight_alignment(spec: CrossDomainExperimentSpec,
                         rows: List[Dict[str, Any]]) -> Any:
    """What the declared windows and clocks imply about shared support (TG17.4).

    Metadata only, and it says so where it cannot say more. A domain whose plan does not
    establish exact coverage cannot have its overlap stated before acquisition, so its pairs are
    reported as bounded by the window rather than given a number that would later turn out to
    have been a guess. That distinction is the entire point of showing this before the freeze.
    """
    from src.core.errors import SpectralEarthError
    from src.core.experiment_adapter import adapter_for_domain
    from src.core.structural_alignment import (MODE_FORBIDS, MODE_RELATIONSHIPS, bind_kernel,
                                               combined_family_multiplier,
                                               nominal_day_discrepancy, pairwise_overlap,
                                               support_profile, window_bounds_seconds)

    refusals: List[Dict[str, Any]] = []
    declarations, admissible = [], {}
    for row in rows:
        try:
            adapter = adapter_for_domain(row["domain"])
        except SpectralEarthError:
            continue
        declarations.append(adapter.declaration)
        admissible[adapter.adapter_id] = adapter.admissible_kernels
    try:
        kernel = bind_kernel(spec.alignment.kernel, spec.alignment.parameters,
                             declarations=declarations, admissible_by_adapter=admissible)
    except SpectralEarthError as error:
        refusals.append({"domain": None, "reason": str(error)})
        return ({"schema": ALIGNMENT_SCHEMA, "mode": spec.mode, "kernel": None,
                 "kernel_refused": str(error), "windows": []}, refusals)

    comparable = [row for row in rows if row["status"] != "REFUSED"]
    names = [row["domain"] for row in comparable]
    windows: List[Dict[str, Any]] = []
    for window in spec.windows:
        bounds = window_bounds_seconds(window.start_utc, window.end_utc)
        profiles, unestablished = {}, {}
        for row in comparable:
            cadence = row.get("native_cadence_seconds")
            planned = row.get("windows") or []
            if not planned or not cadence or not planned[0]["coverage_exact"]:
                unestablished[row["domain"]] = (
                    "native support is %s; exact coverage of this window is established only "
                    "after acquisition, so the shared support cannot be stated yet"
                    % row.get("support_kind", "not planned"))
                continue
            profiles[row["domain"]] = support_profile(
                label=row["domain"], domain=row["domain"], starts=[bounds[0]],
                ends=[bounds[1]], window=bounds, native_scale_seconds=float(cadence))
        pairs: List[Dict[str, Any]] = []
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                if left in profiles and right in profiles:
                    pairs.append(_planned_pair(spec, window, kernel, profiles[left],
                                               profiles[right], refusals))
                    continue
                blocked = [name for name in (left, right) if name in unestablished]
                pairs.append({
                    "left": left, "right": right, "status": "BOUNDED_BY_WINDOW",
                    "overlap_seconds": None,
                    "maximum_possible_overlap_seconds": bounds[1] - bounds[0],
                    "effective_sample_size": None,
                    "reason": "; ".join("%s: %s" % (name, unestablished[name])
                                        for name in blocked)})
        windows.append({
            "name": window.name, "window_seconds": bounds[1] - bounds[0],
            "clock": nominal_day_discrepancy(window.start_utc, window.end_utc),
            "clock_note": ("elapsed UTC seconds, not days times 86400; a local calendar day "
                           "spanning a daylight-saving transition is 23 or 25 hours"),
            "pairs": pairs})
    return ({"schema": ALIGNMENT_SCHEMA, "mode": spec.mode,
             "mode_forbids": MODE_FORBIDS[spec.mode],
             "admitted_relationships": list(MODE_RELATIONSHIPS[spec.mode]),
             "declared_relationships": list(spec.family.relationships),
             "kernel": kernel.describe(),
             "minimum_overlap_seconds": spec.alignment.minimum_overlap_seconds,
             "minimum_effective_samples": spec.alignment.minimum_effective_samples,
             "family": combined_family_multiplier([spec.mode]),
             "row_indices_were_not_compared": True,
             "measurement_values_opened": False,
             "windows": windows}, refusals)


def _planned_pair(spec: CrossDomainExperimentSpec, window: Any, kernel: Any, left: Any,
                  right: Any, refusals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One pair whose shared support metadata does establish, and the thresholds it must meet."""
    from src.core.structural_alignment import pairwise_overlap

    overlap = pairwise_overlap(left, right, kernel=kernel)
    row = {**overlap.describe(), "status": "ESTABLISHED_FROM_METADATA"}
    reason = None
    if overlap.overlap_seconds < spec.alignment.minimum_overlap_seconds:
        reason = ("planned shared support is %.1f s and the manifest requires at least %.1f s"
                  % (overlap.overlap_seconds, spec.alignment.minimum_overlap_seconds))
    elif overlap.effective_sample_size < spec.alignment.minimum_effective_samples:
        reason = ("planned effective sample size is %.3f at the governing scale of %.1f s and "
                  "the manifest requires at least %.3f"
                  % (overlap.effective_sample_size, overlap.governing_scale_seconds,
                     spec.alignment.minimum_effective_samples))
    if reason:
        row["status"], row["reason"] = "REFUSED", reason
        refusals.append({"domain": None, "reason": "%s/%s in window %r: %s"
                         % (left.label, right.label, window.name, reason)})
    return row


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
