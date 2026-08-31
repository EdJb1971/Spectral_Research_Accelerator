"""Frozen, two-stage acquisition campaign for the real T4C.6 ERA5 gate.

The expensive record must not be the first live test of the CDS decoding route.  A campaign
therefore binds a small CDS canary, an exact WeatherBench overlap, the full regional request and
the final GateStudyPlan.  Its offline preflight validates their scientific identity, budgets all
artifacts together, and reports local dependency/consent/configuration readiness without
constructing a client or touching the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import numpy as np

from src.analysis_engine.gate_run import GateStudyPlan
from src.analysis_engine.cross_scale import support_floor
from src.core.errors import DataSourceError, InvalidParameterError
from src.core.publication import publish_new_bytes
from src.data_layer.cds_source import (
    MINIMUM_FREE_RESERVE_BYTES,
    CDSRegionalRequest,
    estimate_cds_storage,
)
from src.data_layer.regional_forecast import VARIABLE_ALIASES
from src.data_layer.zarr_source import (
    CATALOGUE, MIN_VALID_INTERIOR, NETWORK_ENV_VAR, CropSpec, is_cached)

CAMPAIGN_SCHEMA = "cross-scale-gate-campaign/v1"
CAMPAIGN_ENVELOPE_SCHEMA = "cross-scale-gate-campaign-envelope/v1"
PREFLIGHT_SCHEMA = "cross-scale-gate-campaign-preflight/v1"
SCIENTIFIC_REVIEW_SCHEMA = "cross-scale-gate-scientific-review/v1"


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("campaign", value, "finite JSON values: %s" % exc) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        supplied = set(value) if isinstance(value, Mapping) else set()
        raise InvalidParameterError(
            label, sorted(supplied), "exact fields; missing=%s unknown=%s"
            % (sorted(fields - supplied), sorted(supplied - fields)))
    return value


def _times(spec: CDSRegionalRequest) -> np.ndarray:
    values = []
    start = np.datetime64(spec.date_start, "D")
    end = np.datetime64(spec.date_end, "D")
    for day in np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D")):
        values.extend(day.astype("datetime64[h]") + np.timedelta64(hour, "h")
                      for hour in spec.hours_utc)
    return np.asarray(values, dtype="datetime64[ns]")


def _cadence_seconds(spec: CDSRegionalRequest) -> float:
    times = _times(spec).astype("int64")
    differences = np.diff(times)
    if differences.size == 0 or np.any(differences <= 0) \
            or not np.all(differences == differences[0]):
        raise InvalidParameterError(
            "hours_utc", spec.hours_utc,
            "a cadence regular across day boundaries; frame lags cannot use irregular hours")
    return float(differences[0] / 1e9)


def _same_grid(left: CDSRegionalRequest, right: CDSRegionalRequest) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in (
        "lat_min", "lat_max", "lon_min", "lon_max", "grid_degrees"))


def _request_from_mapping(value: Any, label: str) -> CDSRegionalRequest:
    fields = set(CDSRegionalRequest.__dataclass_fields__) | {
        "request_sha256", "area_order", "credential_policy"}
    record = _exact(value, fields, label)
    request = CDSRegionalRequest.from_provenance(record)
    if dict(record) != request.to_provenance():
        raise InvalidParameterError(label, record,
                                    "canonical CDS provenance with valid derived fields")
    return request


#: Fields a crop record may omit because they postdate records already written and signed.
#: `vertical_dim` arrived with TG10.1; a preregistration written before it is authenticated by
#: its fingerprint, so demanding the field would invalidate a signed artefact retroactively
#: (defect D63). Absence means the ERA5 default, which is what those records meant.
_OPTIONAL_CROP_FIELDS = frozenset({"vertical_dim"})


def _crop_from_mapping(value: Any, label: str) -> CropSpec:
    fields = set(CropSpec.__dataclass_fields__) | {"uri", "content_key"}
    supplied = set(value) if isinstance(value, Mapping) else set()
    record = _exact(value, fields - (_OPTIONAL_CROP_FIELDS - supplied), label)
    crop = CropSpec.from_provenance(dict(record))
    if dict(record) != crop.to_provenance():
        raise InvalidParameterError(label, record,
                                    "canonical CropSpec provenance with valid derived fields")
    return crop



def _resolution_audit(protocol, train_frames: int, test_frames: int):
    """Whether each partition holds enough distinct alignments to resolve the declared family.

    T4C.5i step 5, wired ahead of the data rather than after it. The surrogate draw is *with
    replacement* from `admissible_shifts`, so requesting 4,999 surrogates always returns 4,999
    numbers and a nominal p-value floor of 1/5000 whether or not the record contains 4,999
    distinct admissible shifts. `check_power` counts the draws and is satisfied; the exact test's
    reference set is the shifts, and its attainable p-value is bounded by how many there are.

    The Theiler window is not known before the series exists, so the audit uses the most
    favourable window of one frame. A partition that fails here cannot be rescued by any window,
    which is what makes it safe to refuse on before acquisition.
    """
    from src.analysis_engine.spatial_power import frames_for_resolution

    lag = int(min(protocol.lags))
    audits = {}
    for name, frames in (("train", int(train_frames)), ("test", int(test_frames))):
        audits[name] = frames_for_resolution(
            n_frames=frames, lag=lag, theiler=1, n_tests=protocol.family_size,
            alpha=protocol.alpha, correction=protocol.correction)
    audits["adequate"] = all(audits[name]["resolves_corrected_level"]
                             for name in ("train", "test"))
    audits["basis"] = (
        "distinct admissible circular shifts bound the exact test's attainable p-value; "
        "surrogates are drawn with replacement from that set, so the ensemble size does not")
    return audits


def _resolution_refusal_text(audits) -> str:
    failed = [name for name in ("train", "test")
              if not audits[name]["resolves_corrected_level"]]
    return (
        "the %s partition cannot resolve the declared family: %s This is a frame-count deficit "
        "and only a longer record closes it -- a larger crop adds no alignments to shuffle, and "
        "reducing the declared family after the design was frozen is not a remedy."
        % (" and ".join(failed), " ".join(audits[name]["reason"] for name in failed)))


def _scientific_review(campaign: "GateCampaign") -> Dict[str, Any]:
    request, plan = campaign.full_acquisition, campaign.gate_plan
    height = int(round((request.lat_max - request.lat_min) / request.grid_degrees)) + 1
    width = int(round((request.lon_max - request.lon_min) / request.grid_degrees)) + 1
    if plan.transform_family == "swt":
        from src.transform_engine.stationary import filter_support
        supports = [filter_support(plan.wavelet, level)
                    for level in range(1, plan.protocol.n_scales + 1)]
    else:
        from src.transform_engine.dtcwt import filter_support
        supports = [filter_support(level, plan.dtcwt_level1, plan.dtcwt_qshift)
                    for level in range(1, plan.protocol.n_scales + 1)]
    interiors = [{
        "level": level, "support_parent_px": int(support),
        "margin_parent_px_per_side": int(support // 2),
        "valid_parent_shape": [height - 2 * (support // 2),
                               width - 2 * (support // 2)],
    } for level, support in enumerate(supports, start=1)]
    deficient = [record for record in interiors
                 if min(record["valid_parent_shape"]) < MIN_VALID_INTERIOR]
    if deficient:
        raise InvalidParameterError(
            "full_acquisition geometry", [height, width],
            "at least %d valid parent-grid pixels after the frozen transform support; "
            "failures=%s" % (MIN_VALID_INTERIOR, deficient))

    from src.core.channel_series import ChannelGeometrySpec
    from src.physical_core.grid import GridSpec
    grid = GridSpec.latlon(
        (height, width), lat0=request.lat_max, dlat=-request.grid_degrees,
        lon0=request.lon_min, dlon=request.grid_degrees)
    # Declares the `ChannelGeometry` contract it satisfies instead of duck-typing it with a
    # SimpleNamespace: this audit runs before any data exists, so it supplies the labels,
    # per-channel filter support and grid provenance that a lag floor needs, and nothing else.
    signature = ChannelGeometrySpec(
        channels=list(range(1, plan.protocol.n_scales + 1)),
        support_parent_px=list(supports),
        provenance={"grid": grid.to_provenance()})
    floors = support_floor(
        signature, plan.protocol.cadence_seconds, plan.advection_speed_m_s)
    maximum_floor = max(record["floor_frames"] for record in floors["floors"])
    if min(plan.protocol.lags) < maximum_floor:
        raise InvalidParameterError(
            "protocol.lags", plan.protocol.lags,
            "lags at or above the pre-acquisition physical support floor of %d frames"
            % maximum_floor)
    design = plan.protocol.validate()
    resolution = _resolution_audit(
        plan.protocol, design["train_frames"], design["test_frames"])
    full_times = _times(request)
    train_stop = design["train_frames"]
    test_start = train_stop + plan.protocol.embargo_frames
    train_hours = max(0, design["train_frames"] - 1) * plan.protocol.cadence_seconds / 3600.0
    return {
        "primary_analysis": {
            "variable": plan.variable,
            "level_hpa": float(plan.level_hpa),
            "transform_family": plan.transform_family,
            "transform_config": plan.transform_config(),
            "measure": plan.protocol.measure,
            "estimator": plan.protocol.estimator,
            "lags_frames": [int(value) for value in plan.protocol.lags],
            "lags_hours": [float(value * plan.protocol.cadence_seconds / 3600.0)
                           for value in plan.protocol.lags],
            "multiple_comparison_correction": plan.protocol.correction,
            "alpha": float(plan.protocol.alpha),
        },
        "grid_shape": [height, width], "grid_degrees": request.grid_degrees,
        "valid_parent_interiors": interiors,
        "minimum_valid_parent_pixels": MIN_VALID_INTERIOR,
        "surrogate_resolution": resolution,
        "resolvable": bool(resolution["adequate"]),
        "physical_support_floor": floors,
        "full_frames": len(_times(request)),
        "train_frames": design["train_frames"], "test_frames": design["test_frames"],
        "calendar_split": {
            "train_start": str(full_times[0]),
            "train_end": str(full_times[train_stop - 1]),
            "embargo_start": str(full_times[train_stop]),
            "embargo_end": str(full_times[test_start - 1]),
            "test_start": str(full_times[test_start]),
            "test_end": str(full_times[-1]),
        },
        "train_annual_cycles": train_hours / (365.2422 * 24.0),
        "hypothesis_family_size": plan.protocol.family_size,
        "surrogate_statistic_evaluations_upper_bound": (
            2 * plan.protocol.family_size * plan.protocol.n_surrogates),
        "power": design["power"],
    }


def review_gate_campaign(campaign: "GateCampaign") -> Dict[str, Any]:
    """Return the zero-network scientific preregistration review for a frozen campaign."""
    return {
        "schema": SCIENTIFIC_REVIEW_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.fingerprint(),
        "study_plan_sha256": campaign.gate_plan.fingerprint(),
        "scientific_design": _scientific_review(campaign),
        "decision_rule": (
            "PASS requires the same positive q<=%.9g directed scale/lag relationship in "
            "both frozen partitions; an adequately powered absence is FAIL; any contract, "
            "source, geometry, overlap or power failure is INVALID."
            % campaign.gate_plan.protocol.alpha),
        "claim_boundary": (
            "The result adjudicates only this primary ERA5 relationship family. It does not "
            "establish causality, universality across fields/levels/regions, forecast skill "
            "or operational readiness. No secondary analysis may replace the primary verdict."),
        "network_used": False,
    }


@dataclass(frozen=True)
class GateCampaign:
    """Machine-independent identity and mandatory ordering for one atmospheric gate."""

    campaign_id: str
    full_acquisition: CDSRegionalRequest
    canary_acquisition: CDSRegionalRequest
    weatherbench_overlap: CropSpec
    expected_overlap_frames: int
    gate_plan: GateStudyPlan

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise InvalidParameterError("campaign_id", self.campaign_id, "a non-empty identifier")
        if self.gate_plan.evidence_role != "real_era5_gate":
            raise InvalidParameterError(
                "gate_plan.evidence_role", self.gate_plan.evidence_role, "real_era5_gate")
        if self.gate_plan.crop != self.full_acquisition.to_crop_spec():
            raise InvalidParameterError(
                "gate_plan.crop", self.gate_plan.crop.to_provenance(),
                "the exact CropSpec derived from full_acquisition")
        if self.gate_plan.protocol.expected_frames != len(_times(self.full_acquisition)):
            raise InvalidParameterError(
                "protocol.expected_frames", self.gate_plan.protocol.expected_frames,
                "the full acquisition's %d exact timestamps" % len(_times(self.full_acquisition)))
        cadence = _cadence_seconds(self.full_acquisition)
        if cadence != float(self.gate_plan.protocol.cadence_seconds):
            raise InvalidParameterError(
                "protocol.cadence_seconds", self.gate_plan.protocol.cadence_seconds,
                "the full acquisition cadence %.9g" % cadence)

        full, canary = self.full_acquisition, self.canary_acquisition
        if (canary.variables != full.variables or canary.pressure_levels != full.pressure_levels
                or canary.hours_utc != full.hours_utc or canary.dataset != full.dataset
                or canary.data_format != full.data_format or not _same_grid(full, canary)):
            raise InvalidParameterError(
                "canary_acquisition", canary.to_provenance(),
                "the same variables, levels, hours, product, format and grid as full_acquisition")
        full_times, canary_times = _times(full), _times(canary)
        if canary_times.size < 4 or not set(canary_times.astype("int64")).issubset(
                set(full_times.astype("int64"))):
            raise InvalidParameterError(
                "canary timestamps", int(canary_times.size),
                "at least four exact frames contained in the full acquisition")
        if _cadence_seconds(canary) != cadence:
            raise InvalidParameterError("canary cadence", canary.hours_utc,
                                        "the full acquisition cadence")

        overlap = self.weatherbench_overlap
        if overlap.store not in CATALOGUE:
            raise InvalidParameterError(
                "weatherbench_overlap.store", overlap.store,
                "a catalogued WeatherBench 2 ERA5 store")
        catalogue = CATALOGUE[overlap.store]
        if float(catalogue["resolution_deg"]) != float(full.grid_degrees) \
                or float(catalogue["cadence_hours"] * 3600) != cadence:
            raise InvalidParameterError(
                "weatherbench_overlap.store", overlap.store,
                "a WeatherBench route with the exact CDS grid spacing and cadence")
        if (overlap.lat_min, overlap.lat_max, overlap.lon_min, overlap.lon_max) != (
                full.lat_min, full.lat_max, full.lon_min, full.lon_max):
            raise InvalidParameterError(
                "weatherbench_overlap bounds", overlap.to_provenance(),
                "the exact CDS canary spatial bounds")
        if tuple(overlap.levels) != tuple(canary.pressure_levels):
            raise InvalidParameterError("weatherbench_overlap.levels", overlap.levels,
                                        "the exact CDS canary pressure levels")
        missing = [canonical for canonical in full.variables if len([
            name for name in VARIABLE_ALIASES[canonical] if name in overlap.variables]) != 1]
        if missing:
            raise InvalidParameterError(
                "weatherbench_overlap.variables", overlap.variables,
                "exactly one WeatherBench alias for every CDS variable; missing/ambiguous=%s"
                % missing)
        try:
            overlap_start = np.datetime64(overlap.time_start, "ns")
            overlap_end = np.datetime64(overlap.time_end, "ns")
        except ValueError as exc:
            raise InvalidParameterError(
                "weatherbench overlap times", (overlap.time_start, overlap.time_end),
                "exact ISO timestamps") from exc
        if overlap_start != canary_times[0] or overlap_end != canary_times[-1]:
            raise InvalidParameterError(
                "weatherbench overlap times", (overlap.time_start, overlap.time_end),
                "the exact first and last CDS canary timestamps")
        if isinstance(self.expected_overlap_frames, bool) \
                or int(self.expected_overlap_frames) != self.expected_overlap_frames \
                or self.expected_overlap_frames != len(canary_times):
            raise InvalidParameterError(
                "expected_overlap_frames", self.expected_overlap_frames,
                "the canary's %d exact timestamps" % len(canary_times))
        _scientific_review(self)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id,
            "full_acquisition": self.full_acquisition.to_provenance(),
            "canary_acquisition": self.canary_acquisition.to_provenance(),
            "weatherbench_overlap": self.weatherbench_overlap.to_provenance(),
            "expected_overlap_frames": int(self.expected_overlap_frames),
            "gate_plan": self.gate_plan.to_mapping(),
        }

    def fingerprint(self) -> str:
        return _sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateCampaign":
        record = _exact(value, {
            "schema", "campaign_id", "full_acquisition", "canary_acquisition",
            "weatherbench_overlap", "expected_overlap_frames", "gate_plan",
        }, "gate campaign")
        if record["schema"] != CAMPAIGN_SCHEMA:
            raise InvalidParameterError("schema", record["schema"], CAMPAIGN_SCHEMA)
        return cls(
            campaign_id=record["campaign_id"],
            full_acquisition=_request_from_mapping(
                record["full_acquisition"], "full_acquisition"),
            canary_acquisition=_request_from_mapping(
                record["canary_acquisition"], "canary_acquisition"),
            weatherbench_overlap=_crop_from_mapping(
                record["weatherbench_overlap"], "weatherbench_overlap"),
            expected_overlap_frames=record["expected_overlap_frames"],
            gate_plan=GateStudyPlan.from_mapping(record["gate_plan"]),
        )


def _atomic_write_new(path: Union[str, os.PathLike[str]], payload: bytes) -> None:
    target = Path(path)
    try:
        publish_new_bytes(target, payload, "gate campaign")
    except FileExistsError:
        raise
    except OSError as exc:
        raise DataSourceError(str(exc), path=str(target), operation="immutable-publication") from exc


def save_gate_campaign(path: Union[str, os.PathLike[str]], campaign: GateCampaign) -> str:
    envelope = {
        "schema": CAMPAIGN_ENVELOPE_SCHEMA,
        "campaign": campaign.to_mapping(),
        "campaign_sha256": campaign.fingerprint(),
    }
    _atomic_write_new(path, _canonical_json(envelope) + b"\n")
    return campaign.fingerprint()


def load_gate_campaign(path: Union[str, os.PathLike[str]]) -> GateCampaign:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("gate campaign is unreadable: %s" % exc) from exc
    record = _exact(envelope, {"schema", "campaign", "campaign_sha256"},
                    "gate campaign envelope")
    if record["schema"] != CAMPAIGN_ENVELOPE_SCHEMA:
        raise DataSourceError("unsupported gate campaign envelope schema")
    campaign = GateCampaign.from_mapping(record["campaign"])
    if record["campaign_sha256"] != campaign.fingerprint():
        raise DataSourceError("gate campaign hash does not authenticate its contents")
    return campaign


def _probe_volume(path: Path) -> Tuple[str, Path]:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise DataSourceError("cannot resolve filesystem for campaign storage preflight")
    resolved = candidate.resolve()
    drive = os.path.splitdrive(str(resolved))[0].upper()
    return drive or "device:%s" % os.stat(resolved).st_dev, resolved


def _credential_configuration() -> Dict[str, Any]:
    rc_present = (Path.home() / ".cdsapirc").is_file()
    env_present = bool(os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))
    return {
        "configuration_present": bool(rc_present or env_present),
        "configuration_source": (
            "standard ~/.cdsapirc" if rc_present else
            "CDSAPI_URL/CDSAPI_KEY environment" if env_present else None),
        "secret_values_inspected": False,
        "remote_validity_or_licence_acceptance_proven": False,
    }


def preflight_gate_campaign(
    campaign: GateCampaign,
    *,
    full_download_dir: Union[str, os.PathLike[str]],
    canary_download_dir: Union[str, os.PathLike[str]],
    cache_dir: Union[str, os.PathLike[str]],
    independent_cache_dir: Union[str, os.PathLike[str]],
    minimum_free_reserve_bytes: int = MINIMUM_FREE_RESERVE_BYTES,
) -> Dict[str, Any]:
    """Perform a zero-network readiness and aggregate capacity check for the whole campaign."""
    # T4C.5i step 5, and the reason this refusal sits here rather than in the review: a frozen
    # campaign that cannot resolve its own declared family must remain loadable and reviewable,
    # or the defect could not be recorded against it. What it must not do is spend 2.5 GB.
    resolution = _scientific_review(campaign)["surrogate_resolution"]
    if not resolution["adequate"]:
        raise InvalidParameterError(
            "campaign.gate_plan.protocol", resolution["test"]["n_frames"],
            "a record long enough to resolve the declared family before any data is acquired: "
            + _resolution_refusal_text(resolution))
    if isinstance(minimum_free_reserve_bytes, bool) \
            or int(minimum_free_reserve_bytes) != minimum_free_reserve_bytes \
            or minimum_free_reserve_bytes < 0:
        raise InvalidParameterError("minimum_free_reserve_bytes", minimum_free_reserve_bytes,
                                    "a non-negative integer")
    full = estimate_cds_storage(campaign.full_acquisition)
    canary = estimate_cds_storage(campaign.canary_acquisition)
    n_lat = int(math.ceil((campaign.weatherbench_overlap.lat_max
                          - campaign.weatherbench_overlap.lat_min)
                         / campaign.full_acquisition.grid_degrees - 1e-12)) + 1
    n_lon = int(math.ceil((campaign.weatherbench_overlap.lon_max
                          - campaign.weatherbench_overlap.lon_min)
                         / campaign.full_acquisition.grid_degrees - 1e-12)) + 1
    wb_raw = (campaign.expected_overlap_frames * len(campaign.full_acquisition.pressure_levels)
              * n_lat * n_lon * len(campaign.full_acquisition.variables) * 4)
    wb_cache = int(wb_raw * 2 + 16 * 1024 ** 2)

    artifacts = [
        ("full_download", Path(full_download_dir), full["artifact_bytes_upper_bound"]),
        ("full_cache", Path(cache_dir), full["artifact_bytes_upper_bound"]),
        ("canary_download", Path(canary_download_dir), canary["artifact_bytes_upper_bound"]),
        ("canary_cache", Path(cache_dir), canary["artifact_bytes_upper_bound"]),
        ("weatherbench_overlap_cache", Path(independent_cache_dir), wb_cache),
    ]
    grouped: Dict[str, Dict[str, Any]] = {}
    for role, path, required in artifacts:
        volume, probe = _probe_volume(path)
        entry = grouped.setdefault(volume, {
            "volume": volume, "roles": [], "probe": probe, "working_bytes_required": 0})
        entry["roles"].append(role)
        entry["working_bytes_required"] += int(required)
    volumes = []
    for entry in grouped.values():
        usage = shutil.disk_usage(entry.pop("probe"))
        working = int(entry["working_bytes_required"])
        reserve = max(int(minimum_free_reserve_bytes), int(math.ceil(working * 0.10)))
        volumes.append({
            **entry, "free_bytes": int(usage.free), "reserve_bytes": reserve,
            "total_free_required": working + reserve,
            "passes": int(usage.free) >= working + reserve,
        })
    volumes.sort(key=lambda value: value["volume"])

    dependency = importlib.util.find_spec("cdsapi") is not None
    credentials = _credential_configuration()
    consent = os.environ.get(NETWORK_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
    blockers = []
    if not all(volume["passes"] for volume in volumes):
        blockers.append("insufficient aggregate storage")
    if not dependency:
        blockers.append("optional cdsapi dependency is not installed")
    if not credentials["configuration_present"]:
        blockers.append("standard CDS credential configuration is absent")
    if not consent:
        blockers.append("explicit network consent is disabled")

    full_cached = is_cached(campaign.full_acquisition.to_crop_spec(), str(cache_dir))
    canary_cached = is_cached(campaign.canary_acquisition.to_crop_spec(), str(cache_dir))
    independent_cached = is_cached(campaign.weatherbench_overlap, str(independent_cache_dir))
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY_FOR_CANARY" if not blockers else "BLOCKED",
        "campaign_sha256": campaign.fingerprint(),
        "network_used": False,
        "client_constructed": False,
        "dependency": {"cdsapi_available": dependency},
        "credentials": credentials,
        "network_consent_enabled": consent,
        "storage": {
            "basis": "all canary/full NetCDF and Zarr artifacts plus WeatherBench overlap; no compression credit",
            "full": full, "canary": canary,
            "weatherbench_overlap_cache_bytes_upper_bound": wb_cache,
            "volumes": volumes,
        },
        "scientific_design": _scientific_review(campaign),
        "local_artifacts": {
            "full_cache_present": full_cached,
            "canary_cache_present": canary_cached,
            "weatherbench_overlap_cache_present": independent_cached,
        },
        "blockers": blockers,
        "mandatory_order": [
            "materialise_weatherbench_overlap",
            "materialise_cds_canary",
            "require_canary_overlap_PASS_or_stop",
            "materialise_full_cds_record",
            "require_full_cache_overlap_PASS_or_stop",
            "preflight_and_run_T4C.6",
            "proceed_to_4D_only_on_T4C.6_PASS",
        ],
        "claim_boundary": (
            "READY_FOR_CANARY proves local contract, capacity, dependency, configuration-presence "
            "and consent checks only. It does not validate credentials, licence acceptance, CDS "
            "service availability, ERA5 values or the T4C.6 hypothesis."),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or preflight a two-stage real-ERA5 T4C.6 campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--design", required=True,
                        help="JSON object conforming to cross-scale-gate-campaign/v1")
    freeze.add_argument("--out", required=True, help="new immutable campaign envelope")
    review = commands.add_parser("review")
    review.add_argument("--campaign", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--campaign", required=True)
    preflight.add_argument("--full-download-dir", required=True)
    preflight.add_argument("--canary-download-dir", required=True)
    preflight.add_argument("--cache-dir", required=True)
    preflight.add_argument("--independent-cache-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        try:
            design = json.loads(Path(args.design).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataSourceError("campaign design is unreadable: %s" % exc) from exc
        campaign = GateCampaign.from_mapping(design)
        fingerprint = save_gate_campaign(args.out, campaign)
        result = {"campaign_sha256": fingerprint, "campaign_path": str(Path(args.out))}
        code = 0
    elif args.command == "review":
        campaign = load_gate_campaign(args.campaign)
        result = review_gate_campaign(campaign)
        code = 0
    else:
        campaign = load_gate_campaign(args.campaign)
        result = preflight_gate_campaign(
            campaign, full_download_dir=args.full_download_dir,
            canary_download_dir=args.canary_download_dir, cache_dir=args.cache_dir,
            independent_cache_dir=args.independent_cache_dir)
        code = 0 if result["status"] == "READY_FOR_CANARY" else 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
