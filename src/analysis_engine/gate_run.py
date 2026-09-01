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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

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
from src.analysis_engine.spatial_power import (
    StreamedAttenuation,
    assess_interior,
    median_present,
    spatial_power_refusal,
)
from src.core.errors import DataSourceError, InvalidParameterError
from src.core.publication import publish_new_bytes
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
    resolution = _resolution_audit(
        plan.protocol, design["train_frames"], design["test_frames"])
    if plan.evidence_role == "real_era5_gate" and not resolution["adequate"]:
        raise DataSourceError(
            "real gate is not resolvable as designed (T4C.5i): " + _resolution_refusal_text(
                resolution))
    return {
        "status": "READY",
        "plan_sha256": plan.fingerprint(),
        "evidence_role": plan.evidence_role,
        "source": dict(reader.source_provenance),
        "transform": {"family": plan.transform_family, **plan.transform_config()},
        "valid_interiors": one.interior,
        "valid_parent_interiors": exact_parent_interiors,
        "minimum_valid_parent_pixels": MIN_VALID_INTERIOR,
        "surrogate_resolution": resolution,
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


#: Frames whose spatial decorrelation is measured for the receipt. Every frame is another
#: decomposition and one frame's length is a noisy estimate, so an evenly spaced subsample is
#: taken and the median reported -- the same compromise `spatial_power` makes internally, stated
#: here because the receipt publishes the number and a reader is entitled to know how many
#: frames it came from.
DECORRELATION_SAMPLE_FRAMES = 16


def _audited_test(results: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The family's best shot on the train partition, which is the one worth auditing.

    A spatial-power audit answers one question: if this run reports an absence, could the
    instrument have seen the effect? The test that came closest to surviving is the binding case
    -- if even that one is attenuated below the detection threshold, the absence is a statement
    about the crop. Ranking is by p-value, then by excess, then by label so that ties resolve the
    same way on every machine.

    This selects on the **train** partition, which the design already treats as exploratory. The
    same selection on the test partition would be choosing what to audit after seeing the
    held-out result, which is the move the train/test split exists to prevent.
    """
    rows = [row for row in results
            if np.isfinite(float(row.get("observed_nats", float("nan"))))]
    if not rows:
        return None
    return min(rows, key=lambda row: (float(row["p_value"]),
                                      -float(row.get("excess_nats", 0.0)),
                                      str(row["label"])))


def _interior_planes(field, interiors: Sequence[Mapping[str, Any]]) -> List[List[np.ndarray]]:
    """Every scale's valid interior for one decomposed frame, one 2-D plane per orientation.

    This is the same reduction `scale_signature` performs -- magnitude, then the per-scale
    boundary margin removed -- stopped one step earlier. The signature concatenates the
    orientations and collapses them to a single mean of squares; the attenuation curve needs the
    planes still shaped, because it has to crop them concentrically before collapsing.
    """
    planes: List[List[np.ndarray]] = []
    for index, scale in enumerate(field.scales):
        record = interiors[index]
        if not record.get("usable"):
            planes.append([])
            continue
        margin = int(record["halfwidth_native"])
        per_orientation: List[np.ndarray] = []
        for orientation in field.orientations:
            band = field.native_band(0, scale, orientation)
            magnitude = torch.abs(band) if torch.is_complex(band) else band
            array = magnitude.to(torch.float64).numpy()
            if margin:
                array = array[margin:array.shape[0] - margin,
                              margin:array.shape[1] - margin]
            per_orientation.append(np.ascontiguousarray(array))
        planes.append(per_orientation)
    return planes


def _interior_power_records(samples: Mapping[int, List[Dict[str, Any]]],
                            interiors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per-scale decorrelation length and effective sample size: T4C.5i steps 1 and 2.

    Measured on the train partition only. These are the quantities the removed judgement was
    standing in for, so they are published whether or not they end up moving a verdict: a
    reviewer asked to accept an absence needs to see how many independent structures the crop
    actually held, not a constant that says 128.
    """
    records = []
    for index, interior in enumerate(interiors):
        measured = samples.get(index) or []
        if not measured:
            records.append({
                "scale": interior.get("scale"),
                "interior_shape": interior.get("interior_shape"),
                "measured": False,
                "reason": ("no valid interior at this scale, so it carries no spatial "
                           "decorrelation to measure"),
            })
            continue
        effective = [record["effective_samples"]["effective_samples"] for record in measured]
        rows = [record["decorrelation"]["axes"] for record in measured]
        saturated = sum(1 for record in measured if record["decorrelation"]["saturated"])
        per_axis = []
        for axis in (0, 1):
            lengths = [(None if axes[axis]["saturated"]
                        else float(axes[axis]["decorrelation_px"])) for axes in rows]
            per_axis.append(median_present(lengths))
        records.append({
            "scale": interior.get("scale"),
            "interior_shape": interior.get("interior_shape"),
            "measured": True,
            "samples": len(measured),
            "median_decorrelation_px_per_axis": per_axis,
            "median_effective_samples": median_present(effective),
            "saturated_fraction": saturated / len(measured),
            "basis": ("median over an evenly spaced subsample of train frames and every "
                      "orientation of the scale; a saturated interior contributes no length "
                      "rather than the length that was searched to"),
        })
    return records


#: Transform families whose scales share a native grid. For an undecimated family a concentric
#: window of N native pixels is the same patch of atmosphere at every scale, which is what lets
#: one sub-crop size describe both series in an attenuation row.
UNDECIMATED_FAMILIES = ("swt",)


def _audit_limitations(plan: GateStudyPlan) -> List[str]:
    """What the attenuation curve does not establish, carried in the record that reports it.

    A power claim that travels without its caveats is the failure mode this whole task exists to
    remove, so the limitations are a field rather than a docstring.
    """
    limitations = [
        "the curve is a within-field trend over nested sub-crops of one record, not a set of "
        "independent measurements, and nothing infers across sizes",
        "it characterises the single audited test, so it bounds the family's best case rather "
        "than every member of it",
    ]
    if plan.transform_family not in UNDECIMATED_FAMILIES:
        limitations.append(
            "the %s family decimates, so the two scales' native interiors span different "
            "physical extents and a shared sub-crop size is not a shared area. The rows remain "
            "readable as a precision trend, but the matched window is a coefficient-count match "
            "rather than a geographic one. The frozen T4C.6 transform is undecimated and is not "
            "affected." % plan.transform_family)
    return limitations


def _spatial_power_audit(plan: GateStudyPlan, climatology, indices: Sequence[int],
                         signature, result: Mapping[str, Any],
                         interiors: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The derived power record the receipt publishes and the verdict may consult.

    T4C.5i step 7, and with it the attenuation half of step 5 that was deferred because it needs
    exactly this curve. The work is a second pass over the train partition: for the family's best
    test, transfer entropy is recomputed over concentric sub-crops of the same interiors with the
    frames, bins and lag held fixed, so the only thing varying across the curve is spatial
    precision.

    The detection threshold is not invented here. It is the ``k + 1``-th largest value of the
    surrogate ensemble the sweep itself used, recovered by reseeding
    `cross_scale.shift_null_ensemble` with the sweep's own seed -- and the record reports whether
    that reproduction matched the summary the sweep published, because a threshold read off a
    different ensemble would not be this study's decision boundary.
    """
    from src.analysis_engine.cross_scale import shift_null_ensemble, surrogate_seed

    row = _audited_test(result.get("results", []))
    if row is None:
        return {"status": "NOT_AUDITED",
                "reason": ("no test survived the support floor and the interior mask, so the "
                           "family has no best case whose power could be characterised")}

    channels = [str(value) for value in signature.channels]
    source_index = channels.index(str(row["source_scale"]))
    target_index = channels.index(str(row["target_scale"]))
    lag = int(row["lag_frames"])
    theiler = int(row["theiler_window_frames"])
    matrix = signature.to_matrix(plan.protocol.measure)
    seed = surrogate_seed(plan.protocol.seed, lag, source_index)
    null = shift_null_ensemble(
        matrix[:, source_index], matrix[:, target_index], lag,
        bins=plan.protocol.bins, wrap=True, estimator=plan.protocol.estimator,
        n_surrogates=plan.protocol.n_surrogates, seed=seed, theiler=theiler)
    finite = null[np.isfinite(null)]
    reproduced = bool(
        finite.size == int(row["n_surrogates"])
        and math.isclose(float(finite.mean()), float(row["surrogate_mean_nats"]),
                         rel_tol=1e-12, abs_tol=0.0))

    sides = [int(min(interiors[index]["interior_shape"]))
             for index in (source_index, target_index)]
    matched = int(min(sides))
    accumulator = StreamedAttenuation(
        interior_px=matched, n_frames=len(indices), lag=lag, bins=plan.protocol.bins)
    sampled = set(int(value) for value in np.unique(np.linspace(
        0, len(indices) - 1,
        min(DECORRELATION_SAMPLE_FRAMES, len(indices))).astype(int)).tolist())
    samples: Dict[int, List[Dict[str, Any]]] = {}
    for position, global_index in enumerate(indices):
        field = decompose_field(
            climatology.anomaly(int(global_index)), family=plan.transform_family,
            config=plan.transform_config(), keep_native=True)
        planes = _interior_planes(field, interiors)
        accumulator.add(position, planes[source_index], planes[target_index])
        if position in sampled:
            for index, scale_planes in enumerate(planes):
                for plane in scale_planes:
                    samples.setdefault(index, []).append(assess_interior(plane))

    curve = accumulator.curve()
    refusal = spatial_power_refusal(
        curve, null_nats=finite, n_tests=plan.protocol.family_size, n_frames=len(indices),
        theiler=theiler, alpha=plan.protocol.alpha, correction=plan.protocol.correction)
    full = max(curve["curve"], key=lambda entry: entry["interior_px"])
    return {
        "status": "MEASURED",
        "partition": "train",
        "audited_test": {
            "label": str(row["label"]),
            "source_scale": str(row["source_scale"]),
            "target_scale": str(row["target_scale"]),
            "lag_frames": lag,
            "theiler_window_frames": theiler,
            "p_value": float(row["p_value"]),
            "observed_nats": float(row["observed_nats"]),
            "selection": ("the train partition's smallest p-value; for an absence the family's "
                          "best case is the binding one"),
        },
        "surrogate_ensemble": {
            "reproduces_sweep_summary": reproduced,
            "n_surrogates": int(finite.size),
            "seed": int(seed),
            "basis": ("the sweep's own ensemble, recomputed from its seed, because the minimum "
                      "detectable effect is an order statistic no summary preserves"),
        },
        "matched_interior_px": matched,
        "interior_sides_px": {"source": sides[0], "target": sides[1]},
        "limitations": _audit_limitations(plan),
        "matched_interior_note": (
            "the two scales lose different margins to the same filter, so the curve is measured "
            "on the concentric window both interiors can supply. Where those interiors differ, "
            "the curve's largest row is therefore not the sweep's own estimate, which used each "
            "scale's full interior. Both numbers are reported rather than reconciled, because "
            "an adjustment between them would be a correction nothing measured."),
        "sweep_observed_nats": float(row["observed_nats"]),
        "matched_full_interior_nats": float(full["transfer_entropy_nats"]),
        "attenuation": curve,
        "interiors": _interior_power_records(samples, interiors),
        "refusal": refusal,
        "verdict": refusal["verdict"],
    }


def _power_adjudication(plan: GateStudyPlan, gate: Mapping[str, Any],
                        power: Mapping[str, Any]) -> Dict[str, Any]:
    """How the derived power record is allowed to move the gate's verdict.

    Only one direction is open to it. Spatial imprecision attenuates an estimate *toward* the
    null, so an undersized crop cannot manufacture a PASS -- it can only manufacture an absence
    that is a property of the instrument. A PASS therefore stands on its own and an absence does
    not, and that asymmetry is the whole of the FAIL/INVALID boundary step 5 deferred.

    The gate's own verdict is left exactly as `evaluate_replication_gate` returned it. What this
    decides is the *scientific* verdict, which is the one a reviewer reads, so the receipt shows
    both and says which rule moved which.
    """
    if plan.evidence_role != "real_era5_gate":
        return {
            "scientific_verdict": "NOT_ESTABLISHED",
            "power_applied": False,
            "reason": ("synthetic acceptance adjudicates orchestration, not the atmosphere, so "
                       "the derived power record is published for inspection and adjudicates "
                       "nothing"),
        }
    verdict = str(gate["verdict"])
    if verdict != "FAIL":
        return {
            "scientific_verdict": verdict,
            "power_applied": False,
            "reason": ("a %s verdict is not an absence, and spatial imprecision biases toward "
                       "the null, so the derived power record cannot overturn it" % verdict),
        }
    if power.get("status") != "MEASURED":
        return {
            "scientific_verdict": "INVALID",
            "power_applied": True,
            "reason": ("an absence was reported but its power could not be characterised: %s"
                       % power.get("reason", "the audit did not run")),
        }
    if not power["surrogate_ensemble"]["reproduces_sweep_summary"]:
        return {
            "scientific_verdict": "INVALID",
            "power_applied": True,
            "reason": ("the audit could not reproduce the sweep's surrogate ensemble, so the "
                       "detection threshold it derived is not this study's decision boundary "
                       "and the absence cannot be shown adequately powered"),
        }
    if power["verdict"] != "ADEQUATE":
        return {
            "scientific_verdict": "INVALID",
            "power_applied": True,
            "deficit": power["refusal"].get("deficit"),
            "remedies": power["refusal"].get("remedies", []),
            "reason": ("the absence is not adequately powered, so it is not a negative finding: "
                       "%s" % power["refusal"].get("reason", "")),
        }
    return {
        "scientific_verdict": "FAIL",
        "power_applied": True,
        "reason": ("the absence is adequately powered on the derived criterion, so it is a "
                   "negative finding about the atmosphere rather than about the crop: %s"
                   % power["refusal"].get("reason", "")),
    }


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
        # T4C.5i step 7. This runs inside the reader's lifetime because it needs the train
        # anomalies a second time, and it runs unconditionally because the receipt publishes
        # the derivation whether or not it moves this particular verdict -- a power record that
        # appeared only when it changed the answer would be a record nobody could calibrate.
        spatial_power = _spatial_power_audit(
            plan, climatology, train_indices, train_signature, train_result,
            preflight["valid_interiors"])
    adjudication = _power_adjudication(plan, gate, spatial_power)

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
        "spatial_power": spatial_power,
        "power_adjudication": adjudication,
        "scientific_verdict": adjudication["scientific_verdict"],
        "claim_boundary": (
            "Synthetic acceptance proves orchestration only; it is not atmospheric evidence."
            if plan.evidence_role == "synthetic_acceptance" else
            "PASS/FAIL adjudicates only the frozen T4C.6 relationship family on this exact "
            "ERA5 crop; it is not causality, universality, forecast skill or operational "
            "readiness. FAIL is reported only where the derived spatial-power record shows the "
            "absence was detectable; where it does not, the run is INVALID and is not a "
            "negative finding."),
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
    try:
        publish_new_bytes(target, payload, label)
    except FileExistsError:
        raise
    except OSError as exc:
        raise DataSourceError(str(exc), path=str(target), operation="immutable-publication") from exc
