"""Training-native, leakage-safe regional forecast datasets (roadmap T5.2).

The important boundary in this module is temporal, not computational.  Frames are split and
embargoed *before* histories and lead-time targets are assembled, and normalisation is fitted
from the training frames only.  A sample therefore cannot become valid merely because a
windowing implementation happened to reach across a split boundary.

The implementation accepts an already materialised xarray dataset, or the content-addressed
Zarr cache produced by :mod:`src.data_layer.zarr_source`.  It does not download data as a side
effect.  The returned objects are ordinary ``torch.utils.data.Dataset`` instances and work
unchanged with CPU, CUDA or ROCm training: device placement remains the caller's concern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import (FieldSequence, split_temporal,
                                        validate_temporal_guardrails)


CANONICAL_VARIABLES: Tuple[str, ...] = ("t", "q", "u", "v", "z")
VARIABLE_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "t": ("t", "temperature"),
    "q": ("q", "specific_humidity"),
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "z": ("z", "geopotential"),
}


def _stable_hash(value: Any, length: int = 32) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def _time_strings(values: np.ndarray) -> Tuple[str, ...]:
    values = np.asarray(values)
    if values.dtype.kind == "M":
        return tuple(np.datetime_as_string(v, unit="s") for v in values.astype("datetime64[s]"))
    return tuple(str(v.item() if hasattr(v, "item") else v) for v in values)


@dataclass(frozen=True)
class RegionalForecastConfig:
    """Hashable experiment-independent data contract.

    ``lead_frames`` are positive offsets from the final input frame.  The embargo must be at
    least the longest lead: with regularly sampled fields that makes the distance between the
    last frame in one split and the first frame in the next strictly greater than every tested
    lead, as required by :func:`validate_temporal_guardrails`.
    """

    variables: Tuple[str, ...] = CANONICAL_VARIABLES
    level_hpa: int = 850
    history_frames: int = 2
    lead_frames: Tuple[int, ...] = (1,)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    embargo_frames: int = 1
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise InvalidParameterError("variables", self.variables,
                                        "a non-empty sequence of unique canonical variables")
        unknown = sorted(set(self.variables) - set(CANONICAL_VARIABLES))
        if unknown:
            raise InvalidParameterError("variables", unknown,
                                        "canonical ERA5 names drawn from t/q/u/v/z")
        if self.history_frames < 1:
            raise InvalidParameterError("history_frames", self.history_frames, "an integer >= 1")
        if not self.lead_frames or any(int(v) != v or v < 1 for v in self.lead_frames):
            raise InvalidParameterError("lead_frames", self.lead_frames,
                                        "one or more unique positive integer frame offsets")
        if len(set(self.lead_frames)) != len(self.lead_frames):
            raise InvalidParameterError("lead_frames", self.lead_frames, "unique offsets")
        if tuple(sorted(self.lead_frames)) != self.lead_frames:
            raise InvalidParameterError("lead_frames", self.lead_frames, "strictly increasing offsets")
        if self.embargo_frames < max(self.lead_frames):
            raise InvalidParameterError(
                "embargo_frames", self.embargo_frames,
                "at least the longest lead (%d frames). A shorter embargo gives a clean-looking "
                "split whose training target can enter the next window" % max(self.lead_frames))
        if self.dtype not in ("float32", "float64"):
            raise InvalidParameterError("dtype", self.dtype, "float32 or float64")

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["lead_frames"] = list(self.lead_frames)
        record["contract_hash"] = _stable_hash(record)
        return record


@dataclass(frozen=True)
class NormalisationArtifact:
    """Per-channel population statistics fitted exclusively on the training frames."""

    variables: Tuple[str, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    n_values_per_variable: int
    fitted_split: str
    first_timestamp: str
    last_timestamp: str
    source_content_hash: str

    def __post_init__(self) -> None:
        if self.fitted_split != "train":
            raise InvalidParameterError("fitted_split", self.fitted_split,
                                        "exactly 'train'; validation/test fitting is leakage")
        if len(self.variables) != len(self.mean) or len(self.mean) != len(self.std):
            raise InvalidParameterError("normalisation", "misaligned vectors",
                                        "one mean and standard deviation per variable")
        if any(not np.isfinite(v) or v <= 0 for v in self.std):
            raise InvalidParameterError("std", self.std,
                                        "finite positive training standard deviations")

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["mean"] = list(self.mean)
        record["std"] = list(self.std)
        record["method"] = "per-variable population mean/std over train frames and all grid cells"
        record["ddof"] = 0
        record["artifact_hash"] = _stable_hash(record)
        return record


class RegionalForecastDataset(Dataset):
    """A single temporal split yielding normalised histories and lead-time targets.

    Each item is a dictionary so PyTorch's default collator can batch it without a custom
    function.  ``inputs`` has shape ``(history, channel, height, width)`` and ``targets`` has
    shape ``(lead, channel, height, width)``.  Timestamps remain int64 Unix nanoseconds in the
    batchable item; their ISO forms and complete lineage live in :attr:`provenance`.
    """

    def __init__(self, values: torch.Tensor, times_ns: torch.Tensor, frame_indices: Sequence[int],
                 split: str, config: RegionalForecastConfig,
                 normalisation: NormalisationArtifact, provenance: Mapping[str, Any]) -> None:
        if values.ndim != 4:
            raise InvalidParameterError("values", tuple(values.shape), "(time, channel, height, width)")
        self.values = values
        self.times_ns = times_ns.to(torch.int64)
        self.frame_indices = tuple(int(i) for i in frame_indices)
        self.split = split
        self.config = config
        self.normalisation = normalisation
        self.provenance = dict(provenance)
        self.provenance["split"] = split

        first_anchor = config.history_frames - 1
        final_anchor = len(self.frame_indices) - 1 - max(config.lead_frames)
        self._anchors = tuple(range(first_anchor, final_anchor + 1))
        if not self._anchors:
            raise InvalidParameterError(
                "split", {"name": split, "n_frames": len(self.frame_indices)},
                "enough frames for history=%d and maximum lead=%d; use a longer record, not "
                "a smaller scientifically required embargo"
                % (config.history_frames, max(config.lead_frames)))

        dtype = torch.float32 if config.dtype == "float32" else torch.float64
        self._mean = torch.tensor(normalisation.mean, dtype=dtype).view(1, -1, 1, 1)
        self._std = torch.tensor(normalisation.std, dtype=dtype).view(1, -1, 1, 1)

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        anchor_local = self._anchors[item]
        input_local = range(anchor_local - self.config.history_frames + 1, anchor_local + 1)
        target_local = [anchor_local + lead for lead in self.config.lead_frames]
        input_global = [self.frame_indices[i] for i in input_local]
        target_global = [self.frame_indices[i] for i in target_local]
        inputs = (self.values[input_global] - self._mean) / self._std
        targets = (self.values[target_global] - self._mean) / self._std
        return {
            "inputs": inputs,
            "targets": targets,
            "input_times_ns": self.times_ns[input_global],
            "target_times_ns": self.times_ns[target_global],
            "input_frame_indices": torch.tensor(input_global, dtype=torch.int64),
            "target_frame_indices": torch.tensor(target_global, dtype=torch.int64),
        }


@dataclass(frozen=True)
class RegionalForecastBundle:
    train: RegionalForecastDataset
    val: RegionalForecastDataset
    test: RegionalForecastDataset
    normalisation: NormalisationArtifact
    provenance: Mapping[str, Any]

    def datasets(self) -> Dict[str, RegionalForecastDataset]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _resolve_arrays(dataset: Any, config: RegionalForecastConfig) -> Tuple[np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray,
                                                                          Dict[str, Any]]:
    """Resolve aliases and return a canonical ``(T,C,H,W)`` array."""
    if "time" not in dataset.coords:
        raise InvalidParameterError("dataset", list(dataset.coords), "a time coordinate")
    lat_name = "latitude" if "latitude" in dataset.coords else "lat" if "lat" in dataset.coords else None
    lon_name = "longitude" if "longitude" in dataset.coords else "lon" if "lon" in dataset.coords else None
    if lat_name is None or lon_name is None:
        raise InvalidParameterError("dataset", list(dataset.coords), "latitude/longitude coordinates")

    arrays, resolved, units = [], {}, {}
    reference_coords = None
    for canonical in config.variables:
        candidates = [name for name in VARIABLE_ALIASES[canonical] if name in dataset.data_vars]
        if len(candidates) != 1:
            raise InvalidParameterError(
                "variable %s" % canonical, candidates,
                "exactly one available alias from %s (duplicates are ambiguous)"
                % (VARIABLE_ALIASES[canonical],))
        name = candidates[0]
        array = dataset[name]
        if "level" in array.dims:
            levels = np.asarray(dataset["level"].values)
            matches = np.flatnonzero(np.isclose(levels.astype(float), float(config.level_hpa)))
            if len(matches) != 1:
                raise InvalidParameterError("level_hpa", config.level_hpa,
                                            "exactly one matching pressure level in %s" % levels.tolist())
            array = array.isel(level=int(matches[0]))
        extra = set(array.dims) - {"time", lat_name, lon_name}
        if extra:
            raise InvalidParameterError("%s dimensions" % name, list(array.dims),
                                        "only time, level, latitude and longitude; pin %s first"
                                        % sorted(extra))
        array = array.transpose("time", lat_name, lon_name)
        coords = (np.asarray(array.time.values), np.asarray(array[lat_name].values),
                  np.asarray(array[lon_name].values))
        if reference_coords is not None and any(not np.array_equal(a, b)
                                                for a, b in zip(reference_coords, coords)):
            raise InvalidParameterError("variable alignment", canonical,
                                        "identical timestamps, latitude and longitude for every variable")
        reference_coords = coords
        values = np.asarray(array.values)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise InvalidParameterError("variable %s" % canonical, str(values.dtype),
                                        "finite numeric values; missing/non-finite cells need an explicit policy")
        arrays.append(values)
        resolved[canonical] = name
        units[canonical] = array.attrs.get("units")

    assert reference_coords is not None
    times, lat, lon = reference_coords
    if len(times) < 2 or np.any(np.diff(times.astype("datetime64[ns]").astype("int64")) <= 0):
        raise InvalidParameterError("time", _time_strings(times), "strictly increasing timestamps")
    if len(lat) < 2 or len(lon) < 2 or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)) \
            or not np.all(np.diff(lon) > 0):
        raise InvalidParameterError("grid coordinates", (lat.tolist(), lon.tolist()),
                                    "monotonic latitude and strictly increasing longitude")
    stacked = np.stack(arrays, axis=1)
    return stacked, times, lat, lon, {"resolved_names": resolved, "units": units}


def _timeline_split(times: np.ndarray, config: RegionalForecastConfig) -> Dict[str, FieldSequence]:
    # Reuse the platform's accepted R6 implementation.  A one-pixel sentinel represents only
    # the timeline; no meteorological value is copied into this validation structure.
    sentinel = PhysicalField(torch.zeros((1, 1)), grid=GridSpec.pixel((1, 1)))
    timeline = FieldSequence([sentinel] * len(times), times, metadata={"purpose": "T5.2 timeline"})
    parts = split_temporal(timeline, config.train_ratio, config.val_ratio,
                           embargo_frames=config.embargo_frames)
    validate_temporal_guardrails(parts["train"], parts["val"], max(config.lead_frames))
    validate_temporal_guardrails(parts["val"], parts["test"], max(config.lead_frames))
    return parts


def prepare_regional_forecast_datasets(dataset: Any, config: RegionalForecastConfig,
                                       source_manifest: Mapping[str, Any]) -> RegionalForecastBundle:
    """Prepare all three splits from a materialised xarray dataset.

    ``source_manifest`` is mandatory: accepting an anonymous array here would make the tensors
    usable but the experiment irreproducible.  A manifest must at minimum carry the materialised
    content hash; the Zarr cache manifest already supplies the full crop and chunk record.
    """
    content_hash = str(source_manifest.get("content_hash", ""))
    if not content_hash:
        raise InvalidParameterError("source_manifest", sorted(source_manifest),
                                    "a materialised source record containing content_hash")
    values_np, times, lat, lon, variable_record = _resolve_arrays(dataset, config)
    parts = _timeline_split(times, config)
    time_ns_np = times.astype("datetime64[ns]").astype("int64")
    index_by_time = {int(value): i for i, value in enumerate(time_ns_np)}
    frame_indices = {
        name: tuple(index_by_time[int(v)] for v in
                    np.asarray(part.raw_times).astype("datetime64[ns]").astype("int64"))
        for name, part in parts.items()
    }

    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    values = torch.as_tensor(np.ascontiguousarray(values_np), dtype=dtype)
    train_values = values[list(frame_indices["train"])]
    means = train_values.mean(dim=(0, 2, 3), dtype=torch.float64)
    stds = train_values.to(torch.float64).std(dim=(0, 2, 3), correction=0)
    train_time_strings = _time_strings(times[list(frame_indices["train"])])
    normalisation = NormalisationArtifact(
        variables=config.variables, mean=tuple(float(v) for v in means),
        std=tuple(float(v) for v in stds),
        n_values_per_variable=int(train_values.shape[0] * train_values.shape[2] * train_values.shape[3]),
        fitted_split="train", first_timestamp=train_time_strings[0], last_timestamp=train_time_strings[-1],
        source_content_hash=content_hash)

    split_record = {
        name: {"n_frames": len(indices), "first_frame_index": indices[0],
               "last_frame_index": indices[-1],
               "first_timestamp": _time_strings(times[[indices[0]]])[0],
               "last_timestamp": _time_strings(times[[indices[-1]]])[0]}
        for name, indices in frame_indices.items()
    }
    provenance = {
        "schema": "regional_forecast_dataset/v1",
        "source": dict(source_manifest),
        "source_content_hash": content_hash,
        "config": config.to_provenance(),
        "variables": variable_record,
        "level_hpa": config.level_hpa,
        "timestamps": {"count": len(times), "first": _time_strings(times[:1])[0],
                       "last": _time_strings(times[-1:])[0],
                       "sha256": _stable_hash(_time_strings(times))},
        "grid": {"latitude": [float(v) for v in lat], "longitude": [float(v) for v in lon],
                 "shape": [len(lat), len(lon)],
                 "sha256": _stable_hash({"latitude": lat.tolist(), "longitude": lon.tolist()})},
        "temporal_split": split_record,
        "normalisation": normalisation.to_provenance(),
        "construction_order": "split and embargo frames first; fit train statistics; construct samples within each split",
        "device_policy": "CPU tensors returned; DataLoader/training loop may move batches to CUDA, ROCm or MPS",
    }
    time_tensor = torch.as_tensor(time_ns_np, dtype=torch.int64)
    made = {
        name: RegionalForecastDataset(values, time_tensor, frame_indices[name], name, config,
                                      normalisation, provenance)
        for name in ("train", "val", "test")
    }
    return RegionalForecastBundle(made["train"], made["val"], made["test"],
                                  normalisation, provenance)


def prepare_cached_regional_forecast(spec: Any, config: RegionalForecastConfig,
                                     cache_dir: Optional[str] = None) -> RegionalForecastBundle:
    """Open a previously materialised crop locally and prepare T5.2 datasets.

    No network fallback is permitted.  The manifest returned by ``load_cached`` includes a
    measured ``remote_bytes == 0`` fact, making local laptop and shared HPC cache use the same
    call with only ``cache_dir`` changed.
    """
    from src.data_layer.zarr_source import load_cached

    dataset, manifest = load_cached(spec, cache_dir=cache_dir)
    try:
        return prepare_regional_forecast_datasets(dataset, config, manifest)
    finally:
        dataset.close()


def assess_manifest_readiness(manifest: Mapping[str, Any],
                              config: Optional[RegionalForecastConfig] = None) -> Dict[str, Any]:
    """Cheap, metadata-only T5.2 readiness assessment for the workbench.

    This deliberately reports structural eligibility, not successful preparation.  Means,
    standard deviations and sample boundaries can only be certified after opening the values.
    """
    config = config or RegionalForecastConfig()
    spec = manifest.get("spec", {})
    available = set(manifest.get("variables", ())) | set(spec.get("variables", ()))
    resolved = {}
    missing = []
    ambiguous = []
    for canonical in config.variables:
        found = sorted(available.intersection(VARIABLE_ALIASES[canonical]))
        if len(found) == 1:
            resolved[canonical] = found[0]
        elif not found:
            missing.append(canonical)
        else:
            ambiguous.append(canonical)
    levels = [int(v) for v in spec.get("levels", ())]
    level_available = config.level_hpa in levels
    n_frames = int(manifest.get("shape", {}).get("time", 0))
    minimum_frames = (config.history_frames + max(config.lead_frames) +
                      2 * config.embargo_frames + 3)
    eligible = (bool(manifest.get("content_hash")) and not missing and not ambiguous
                and level_available and n_frames >= minimum_frames)
    return {
        "structurally_eligible": eligible,
        "required_variables": list(config.variables), "resolved_variables": resolved,
        "missing_variables": missing, "ambiguous_variables": ambiguous,
        "required_level_hpa": config.level_hpa, "level_available": level_available,
        "n_frames": n_frames, "minimum_frames_lower_bound": minimum_frames,
        "content_fingerprinted": bool(manifest.get("content_hash")),
        "dataset_prepared": False,
        "train_only_normalisation_verified": False,
        "independent_era5_crosscheck": "NOT RUN",
        "claim_boundary": ("Structural eligibility uses manifest metadata only. Prepare the dataset to "
                           "compute split/sample boundaries and train-only statistics; run an actual "
                           "second ERA5 route before claiming source-value agreement."),
    }


def cross_check_era5_overlap(primary: Any, independent: Any, config: RegionalForecastConfig,
                             rtol: float = 1e-5, atol: float = 1e-6) -> Dict[str, Any]:
    """Compare a small overlap obtained through two independent ERA5 routes.

    This is an acceptance instrument, not an assertion that an independent route has already
    been run.  Coordinates must be exact; values use declared tolerances and report the worst
    discrepancy per canonical variable.
    """
    a, at, alat, alon, _ = _resolve_arrays(primary, config)
    b, bt, blat, blon, _ = _resolve_arrays(independent, config)
    coordinates_equal = (np.array_equal(at, bt) and np.array_equal(alat, blat)
                         and np.array_equal(alon, blon))
    if not coordinates_equal:
        raise ValueError("Independent ERA5 overlap has different timestamps or grid coordinates; "
                         "interpolation is not allowed in a value cross-check.")
    detail = {}
    for index, variable in enumerate(config.variables):
        delta = np.abs(a[:, index].astype("float64") - b[:, index].astype("float64"))
        detail[variable] = {"max_abs_error": float(delta.max()),
                            "mean_abs_error": float(delta.mean()),
                            "allclose": bool(np.allclose(a[:, index], b[:, index], rtol=rtol, atol=atol))}
    passed = all(record["allclose"] for record in detail.values())
    return {"passed": passed, "coordinates_exact": True, "rtol": rtol, "atol": atol,
            "variables": detail, "n_frames": int(a.shape[0]), "shape": list(a.shape),
            "comparison_hash": _stable_hash(detail)}
