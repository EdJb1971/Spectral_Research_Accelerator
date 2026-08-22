"""Portable, content-bound execution of the real-data T4C.6 replication gate.

The statistical test existed before this module, but the scientific *job* did not: crop,
variable, level, transform, climatology, advection floor and source identity could drift between
the train and test calls without changing ``GateProtocol``.  This module freezes that complete
design, preflights an existing local cache without network fallback, runs bounded two-pass
anomaly/signature extraction, and publishes one atomic no-overwrite receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.analysis_engine.climatology import (
    DEFAULT_PERIODS_HOURS,
    fit_harmonic_climatology_stream,
)
from src.analysis_engine.cross_scale import (
    GateProtocol,
    cross_scale_dependency,
    evaluate_replication_gate,
    support_floor,
)
from src.analysis_engine.scale_signature import scale_signature, stream_scale_signature
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.zarr_source import CachedFieldReader, CropSpec, MIN_VALID_INTERIOR
from src.transform_engine.coefficient_field import decompose_field

PLAN_SCHEMA = "cross-scale-gate-study/v1"
RECEIPT_SCHEMA = "cross-scale-gate-receipt/v1"
EVIDENCE_ROLES = ("synthetic_acceptance", "real_era5_gate")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class GateStudyPlan:
    """Complete machine-independent scientific identity for one T4C.6 study."""

    study_id: str
    evidence_role: str
    crop: CropSpec
    protocol: GateProtocol
    variable: str
    level_hpa: float
    transform_family: str = "swt"
    wavelet: str = "db2"
    boundary_mode: str = "periodic"
    dtcwt_level1: str = "near_sym_b"
    dtcwt_qshift: str = "qshift_b"
    threshold_sigma: float = 3.0
    climatology_periods_hours: Tuple[float, ...] = DEFAULT_PERIODS_HOURS
    climatology_harmonics: int = 2
    climatology_rank_rtol: float = 1e-10
    advection_speed_m_s: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise InvalidParameterError("study_id", self.study_id, "a non-empty identifier")
        if self.protocol.study_id != self.study_id:
            raise InvalidParameterError(
                "protocol.study_id", self.protocol.study_id,
                "the plan study_id %r" % self.study_id)
        self.protocol.validate()
        if self.evidence_role not in EVIDENCE_ROLES:
            raise InvalidParameterError("evidence_role", self.evidence_role,
                                        "one of %s" % (list(EVIDENCE_ROLES),))
        if self.variable not in self.crop.variables:
            raise InvalidParameterError(
                "variable", self.variable, "one of the frozen crop variables %s"
                % (list(self.crop.variables),))
        if self.level_hpa not in self.crop.levels:
            raise InvalidParameterError(
                "level_hpa", self.level_hpa, "one of the frozen crop levels %s"
                % (list(self.crop.levels),))
        if self.crop.n_levels_analysis != self.protocol.n_scales:
            raise InvalidParameterError(
                "crop.n_levels_analysis", self.crop.n_levels_analysis,
                "the protocol's %d scales" % self.protocol.n_scales)
        if self.transform_family not in ("swt", "dtcwt"):
            raise InvalidParameterError("transform_family", self.transform_family,
                                        "'swt' or 'dtcwt'")
        if self.transform_family == "swt" and self.wavelet not in ("haar", "db2", "db3"):
            raise InvalidParameterError("wavelet", self.wavelet, "haar, db2 or db3")
        if self.transform_family == "swt" and self.boundary_mode != "periodic":
            raise InvalidParameterError(
                "boundary_mode", self.boundary_mode,
                "periodic for the accepted SWT implementation; another mode is a different study")
        if not math.isfinite(self.threshold_sigma) or self.threshold_sigma <= 0:
            raise InvalidParameterError("threshold_sigma", self.threshold_sigma,
                                        "a finite positive value")
        if (not self.climatology_periods_hours
                or any(not math.isfinite(value) or value <= 0
                       for value in self.climatology_periods_hours)):
            raise InvalidParameterError("climatology_periods_hours",
                                        self.climatology_periods_hours,
                                        "one or more finite positive periods")
        if isinstance(self.climatology_harmonics, bool) \
                or int(self.climatology_harmonics) != self.climatology_harmonics \
                or self.climatology_harmonics < 1:
            raise InvalidParameterError("climatology_harmonics", self.climatology_harmonics,
                                        "an integer >= 1")
        if not math.isfinite(self.climatology_rank_rtol) \
                or self.climatology_rank_rtol <= 0:
            raise InvalidParameterError("climatology_rank_rtol",
                                        self.climatology_rank_rtol,
                                        "a finite positive tolerance")
        if not math.isfinite(self.advection_speed_m_s) or self.advection_speed_m_s <= 0:
            raise InvalidParameterError("advection_speed_m_s", self.advection_speed_m_s,
                                        "a declared finite positive speed")
        if self.evidence_role == "real_era5_gate" \
                and not self.crop.store.startswith("cds:"):
            raise InvalidParameterError(
                "crop.store", self.crop.store,
                "a direct regional CDS crop for the current real gate; a synthetic/local URI "
                "cannot be relabelled as ERA5 evidence")

    def transform_config(self) -> Dict[str, Any]:
        if self.transform_family == "swt":
            return {"levels": self.protocol.n_scales, "wavelet": self.wavelet,
                    "mode": self.boundary_mode}
        return {"levels": self.protocol.n_scales, "level1": self.dtcwt_level1,
                "qshift": self.dtcwt_qshift}

    def to_mapping(self) -> Dict[str, Any]:
        mapping = asdict(self)
        mapping["schema"] = PLAN_SCHEMA
        mapping["crop"] = self.crop.to_provenance()
        mapping["protocol"] = self.protocol.to_mapping()
        mapping["climatology_periods_hours"] = list(self.climatology_periods_hours)
        mapping["transform_config"] = self.transform_config()
        return mapping

    def fingerprint(self) -> str:
        return _sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateStudyPlan":
        expected = set(cls.__dataclass_fields__) | {"schema", "transform_config"}
        if not isinstance(value, Mapping) or set(value) != expected:
            supplied = set(value) if isinstance(value, Mapping) else set()
            raise InvalidParameterError(
                "gate plan fields", sorted(supplied),
                "exact fields; missing=%s unknown=%s"
                % (sorted(expected - supplied), sorted(supplied - expected)))
        if value.get("schema") != PLAN_SCHEMA:
            raise InvalidParameterError("schema", value.get("schema"), PLAN_SCHEMA)
        record = dict(value)
        record.pop("schema")
        declared_transform = record.pop("transform_config")
        record["crop"] = CropSpec.from_provenance(record["crop"])
        record["protocol"] = GateProtocol.from_mapping(dict(record["protocol"]))
        record["climatology_periods_hours"] = tuple(record["climatology_periods_hours"])
        plan = cls(**record)
        if declared_transform != plan.transform_config():
            raise InvalidParameterError(
                "transform_config", declared_transform,
                "the config derived from the plan's explicit transform fields")
        return plan


def save_gate_plan(path: Union[str, os.PathLike[str]], plan: GateStudyPlan) -> str:
    envelope = {"plan": plan.to_mapping(), "plan_sha256": plan.fingerprint()}
    _atomic_write_new(path, _canonical_json(envelope) + b"\n", "gate study plan")
    return plan.fingerprint()


def load_gate_plan(path: Union[str, os.PathLike[str]]) -> GateStudyPlan:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("gate study plan is unreadable: %s" % exc) from exc
    if not isinstance(envelope, Mapping) or set(envelope) != {"plan", "plan_sha256"}:
        raise DataSourceError("gate study plan envelope has unknown or missing fields")
    plan = GateStudyPlan.from_mapping(envelope["plan"])
    if envelope["plan_sha256"] != plan.fingerprint():
        raise DataSourceError("gate study plan hash does not authenticate its contents")
    return plan


def _preflight_with_reader(plan: GateStudyPlan, reader: CachedFieldReader) -> Dict[str, Any]:
    design = plan.protocol.validate()
    if len(reader) != plan.protocol.expected_frames:
        raise DataSourceError(
            "cached crop has %d frames, frozen protocol requires exactly %d"
            % (len(reader), plan.protocol.expected_frames))
    cadence = float(reader.source_provenance["cadence_seconds"])
    if cadence != float(plan.protocol.cadence_seconds):
        raise DataSourceError(
            "cached cadence is %.9g seconds, frozen protocol requires %.9g"
            % (cadence, plan.protocol.cadence_seconds))
    if plan.evidence_role == "real_era5_gate":
        if reader.source_provenance.get("source_route") != "Copernicus Climate Data Store API":
            raise DataSourceError("real gate source is not the direct regional CDS route")
        from src.data_layer.era5_overlap import validate_overlap_evidence
        validate_overlap_evidence(
            reader.source_provenance, variable=plan.variable, level_hpa=plan.level_hpa)
        if reader.units.lower() == "unknown":
            raise DataSourceError("real gate variable units are unknown")
    sample = reader.read_frame(0)
    one = scale_signature(decompose_field(
        sample, family=plan.transform_family, config=plan.transform_config(), keep_native=True))
    if one.n_scales != plan.protocol.n_scales:
        raise DataSourceError("transform returned a different scale count from the protocol")
    unusable = [record for record in one.interior if not record.get("usable")]
    thin = [record for record in one.interior if record.get("thin")]
    if unusable or (thin and plan.evidence_role == "real_era5_gate"):
        raise DataSourceError(
            "transform has unusable or statistically thin valid interiors: %s"
            % _json_safe(unusable + thin))
    parent_shape = (int(reader.latitude.size), int(reader.longitude.size))
    exact_parent_interiors = []
    for record in one.interior:
        margin = int(record["support_parent_px"]) // 2
        valid_shape = [parent_shape[0] - 2 * margin, parent_shape[1] - 2 * margin]
        exact_parent_interiors.append({
            "scale": record["scale"],
            "support_parent_px": int(record["support_parent_px"]),
            "excluded_parent_px_per_side": margin,
            "valid_parent_shape": valid_shape,
        })
    if plan.evidence_role == "real_era5_gate":
        deficient = [record for record in exact_parent_interiors
                     if min(record["valid_parent_shape"]) < MIN_VALID_INTERIOR]
        if deficient:
            raise DataSourceError(
                "real gate violates R13: the frozen transform must retain at least %d valid "
                "parent-grid pixels in each direction at every scale; exact failures: %s"
                % (MIN_VALID_INTERIOR, _json_safe(deficient)))
    floors = support_floor(one, cadence, plan.advection_speed_m_s)
    maximum_floor = max(int(record["floor_frames"]) for record in floors["floors"])
    if min(plan.protocol.lags) < maximum_floor:
        raise DataSourceError(
            "frozen lag family starts at %d frames but the exact transform/grid/advection "
            "support floor reaches %d; change the plan before observing results"
            % (min(plan.protocol.lags), maximum_floor))

    nanoseconds = reader.times.astype("datetime64[ns]")
    train_stop = design["train_frames"]
    test_start = train_stop + plan.protocol.embargo_frames
    return {
        "status": "READY",
        "plan_sha256": plan.fingerprint(),
        "evidence_role": plan.evidence_role,
        "source": dict(reader.source_provenance),
        "transform": {"family": plan.transform_family, **plan.transform_config()},
        "valid_interiors": one.interior,
        "valid_parent_interiors": exact_parent_interiors,
        "minimum_valid_parent_pixels": MIN_VALID_INTERIOR,
        "support_floor": floors,
        "split": {
            "train_frames": design["train_frames"],
            "test_frames": design["test_frames"],
            "embargo_frames": plan.protocol.embargo_frames,
            "train_start": str(nanoseconds[0]),
            "train_end": str(nanoseconds[train_stop - 1]),
            "test_start": str(nanoseconds[test_start]),
            "test_end": str(nanoseconds[-1]),
        },
        "network_used": False,
        "machine_paths_included": bool(
            reader.source_provenance.get("machine_paths_included")),
    }


def preflight_cached_gate(
    plan: GateStudyPlan,
    *,
    cache_dir: Optional[str] = None,
    maximum_source_chunk_bytes: int = 512 * 1024 * 1024,
) -> Dict[str, Any]:
    with CachedFieldReader(
            plan.crop, plan.variable, level_hpa=plan.level_hpa,
            cache_dir=cache_dir,
            maximum_source_chunk_bytes=maximum_source_chunk_bytes) as reader:
        return _preflight_with_reader(plan, reader)


def run_cached_gate(
    plan: GateStudyPlan,
    *,
    cache_dir: Optional[str] = None,
    receipt_path: Optional[Union[str, os.PathLike[str]]] = None,
    maximum_source_chunk_bytes: int = 512 * 1024 * 1024,
) -> Dict[str, Any]:
    if receipt_path is not None and Path(receipt_path).exists():
        raise FileExistsError("refusing to overwrite gate receipt at %s" % receipt_path)
    with CachedFieldReader(
            plan.crop, plan.variable, level_hpa=plan.level_hpa,
            cache_dir=cache_dir,
            maximum_source_chunk_bytes=maximum_source_chunk_bytes) as reader:
        preflight = _preflight_with_reader(plan, reader)
        design = plan.protocol.validate()
        train_stop = design["train_frames"]
        test_start = train_stop + plan.protocol.embargo_frames
        fit_mask = np.arange(len(reader)) < train_stop
        times_ns = reader.times.astype("datetime64[ns]").astype("int64")
        times_hours = times_ns.astype(np.float64) / 3.6e12
        climatology = fit_harmonic_climatology_stream(
            reader.read_frame, times_hours, fit_mask=fit_mask,
            periods_hours=plan.climatology_periods_hours,
            n_harmonics=plan.climatology_harmonics,
            rank_rtol=plan.climatology_rank_rtol)

        train_indices = tuple(range(0, train_stop))
        test_indices = tuple(range(test_start, len(reader)))

        def split_reader(indices: Sequence[int]):
            return lambda local_index: climatology.anomaly(indices[int(local_index)])

        train_signature = stream_scale_signature(
            split_reader(train_indices), reader.times[list(train_indices)],
            family=plan.transform_family, config=plan.transform_config(),
            threshold_sigma=plan.threshold_sigma,
            source_provenance={**reader.source_provenance, "split": "train",
                               "climatology_fit_sha256": climatology.provenance["fit_sha256"]})
        test_signature = stream_scale_signature(
            split_reader(test_indices), reader.times[list(test_indices)],
            family=plan.transform_family, config=plan.transform_config(),
            threshold_sigma=plan.threshold_sigma,
            threshold_values=train_signature.threshold_values,
            source_provenance={**reader.source_provenance, "split": "test",
                               "climatology_fit_sha256": climatology.provenance["fit_sha256"]})

        def dependency(signature, seed):
            result = cross_scale_dependency(
                signature, lags=plan.protocol.lags,
                cadence_seconds=plan.protocol.cadence_seconds,
                measure=plan.protocol.measure, estimator=plan.protocol.estimator,
                bins=plan.protocol.bins, wrap=True,
                advection_speed_m_s=plan.advection_speed_m_s,
                n_surrogates=plan.protocol.n_surrogates,
                alpha=plan.protocol.alpha, correction=plan.protocol.correction,
                seed=seed)
            result.update({
                "protocol_fingerprint": plan.protocol.fingerprint(),
                "plan_sha256": plan.fingerprint(),
            })
            return result

        train_result = dependency(train_signature, plan.protocol.seed)
        test_result = dependency(test_signature, plan.protocol.seed + 1)
        gate = evaluate_replication_gate(train_result, test_result, plan.protocol)

    receipt = _json_safe({
        "schema": RECEIPT_SCHEMA,
        "plan": plan.to_mapping(),
        "plan_sha256": plan.fingerprint(),
        "preflight": preflight,
        "climatology": climatology.summary(),
        "signatures": {
            "train": train_signature.summary(),
            "test": test_signature.summary(),
            "train_input_stream_sha256": train_signature.provenance["input_stream_sha256"],
            "test_input_stream_sha256": test_signature.provenance["input_stream_sha256"],
            "test_thresholds_fitted_on": "train",
        },
        "train": train_result,
        "test": test_result,
        "gate": gate,
        "scientific_verdict": (gate["verdict"]
                               if plan.evidence_role == "real_era5_gate" else "NOT_ESTABLISHED"),
        "claim_boundary": (
            "Synthetic acceptance proves orchestration only; it is not atmospheric evidence."
            if plan.evidence_role == "synthetic_acceptance" else
            "PASS/FAIL adjudicates only the frozen T4C.6 relationship family on this exact "
            "ERA5 crop; it is not causality, universality, forecast skill or operational readiness."),
        "network_used": False,
        "machine_paths_included": bool(
            preflight["source"].get("machine_paths_included")),
    })
    receipt["receipt_sha256"] = _sha256(receipt)
    if receipt_path is not None:
        _atomic_write_new(
            receipt_path, _canonical_json(receipt) + b"\n", "gate receipt")
    return receipt


def load_gate_receipt(path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("gate receipt is unreadable: %s" % exc) from exc
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise DataSourceError("unsupported gate receipt schema")
    expected = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if expected != _sha256(unsigned):
        raise DataSourceError("gate receipt hash does not authenticate its contents")
    plan = GateStudyPlan.from_mapping(receipt.get("plan") or {})
    if receipt.get("plan_sha256") != plan.fingerprint():
        raise DataSourceError("gate receipt is not bound to its embedded plan")
    if (receipt.get("gate") or {}).get("protocol_fingerprint") != plan.protocol.fingerprint():
        raise DataSourceError("gate verdict is not bound to the embedded protocol")
    return receipt


def _atomic_write_new(path: Union[str, os.PathLike[str]], payload: bytes, label: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError("refusing to overwrite %s at %s" % (label, target)) from None
        except OSError as exc:
            raise DataSourceError(
                "cannot atomically publish %s at %s: %s" % (label, target, exc)) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
