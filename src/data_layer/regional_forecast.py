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
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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


def _stable_hash(value: Any, length: int = 64) -> str:
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
    calendar_boundaries: Optional[Tuple[str, str]] = None
    expected_cadence_hours: Optional[float] = None
    embargo_frames: int = 1
    dtype: str = "float32"
    statistics_chunk_frames: int = 32
    longitude_convention: Optional[str] = None

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
        if self.calendar_boundaries is not None:
            if len(self.calendar_boundaries) != 2:
                raise InvalidParameterError(
                    "calendar_boundaries", self.calendar_boundaries,
                    "exactly (validation_start, test_start) ISO timestamps")
            try:
                boundaries = tuple(np.datetime64(value, "ns") for value in self.calendar_boundaries)
            except (TypeError, ValueError):
                raise InvalidParameterError(
                    "calendar_boundaries", self.calendar_boundaries,
                    "two valid ISO timestamps") from None
            if any(np.isnat(value) for value in boundaries) or boundaries[0] >= boundaries[1]:
                raise InvalidParameterError(
                    "calendar_boundaries", self.calendar_boundaries,
                    "validation_start strictly before test_start")
        if self.expected_cadence_hours is not None:
            cadence = float(self.expected_cadence_hours)
            cadence_ns = cadence * 3_600_000_000_000
            if not np.isfinite(cadence) or cadence <= 0 or not cadence_ns.is_integer():
                raise InvalidParameterError(
                    "expected_cadence_hours", self.expected_cadence_hours,
                    "a positive duration exactly representable in integer nanoseconds")
        if self.dtype not in ("float32", "float64"):
            raise InvalidParameterError("dtype", self.dtype, "float32 or float64")
        if int(self.statistics_chunk_frames) != self.statistics_chunk_frames \
                or self.statistics_chunk_frames < 1:
            raise InvalidParameterError("statistics_chunk_frames", self.statistics_chunk_frames,
                                        "a positive integer frame count")
        if self.longitude_convention not in (None, "-180..180", "0..360"):
            raise InvalidParameterError(
                "longitude_convention", self.longitude_convention,
                "'-180..180', '0..360', or None when the source convention is not known")

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["lead_frames"] = list(self.lead_frames)
        if self.calendar_boundaries is not None:
            record["calendar_boundaries"] = list(self.calendar_boundaries)
        record["split_mode"] = ("calendar_boundaries" if self.calendar_boundaries is not None
                                else "ratios")
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
    computation_method: str = "eager float64 population moments"
    chunk_frames: Optional[int] = None

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
        record["method"] = self.computation_method
        record["ddof"] = 0
        record["artifact_hash"] = _stable_hash(record)
        return record


class _LazyZarrValues:
    """Pickle-safe, process-local reader for canonical ``(T,C,H,W)`` windows.

    No xarray object or store handle is serialised.  A DataLoader worker opens the local Zarr
    cache on its first item and reuses that handle only while its process id remains unchanged.
    This matters both for Windows ``spawn`` and Linux HPC ``fork`` workers.
    """

    def __init__(self, path: str, resolved_names: Mapping[str, str], variables: Sequence[str],
                 level_hpa: int, lat_name: str, lon_name: str, shape: Sequence[int],
                 dtype: str) -> None:
        self.path = os.path.abspath(path)
        self.resolved_names = dict(resolved_names)
        self.variables = tuple(variables)
        self.level_hpa = int(level_hpa)
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        self._dataset = None
        self._owner_pid: Optional[int] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_dataset"] = None
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
        self._dataset = None
        self._owner_pid = None

    def _open(self) -> Any:
        pid = os.getpid()
        if self._dataset is None or self._owner_pid != pid:
            self.close()
            from src.data_layer.zarr_source import open_dataset
            self._dataset, _counter = open_dataset(self.path, chunks=None)
            self._owner_pid = pid
        return self._dataset

    def read(self, indices: Sequence[int]) -> torch.Tensor:
        requested = [int(v) for v in indices]
        dataset = self._open()
        channels = []
        for canonical in self.variables:
            array = dataset[self.resolved_names[canonical]]
            if "level" in array.dims:
                levels = np.asarray(dataset["level"].values).astype(float)
                matches = np.flatnonzero(np.isclose(levels, float(self.level_hpa)))
                if len(matches) != 1:
                    raise InvalidParameterError("level_hpa", self.level_hpa,
                                                "exactly one matching cached pressure level")
                array = array.isel(level=int(matches[0]))
            array = array.isel(time=requested).transpose("time", self.lat_name, self.lon_name)
            values = np.asarray(array.values)
            if not np.isfinite(values).all():
                raise InvalidParameterError(
                    "cached values", {"variable": canonical, "frame_indices": requested},
                    "finite numeric values; missing/non-finite cells need an explicit policy")
            channels.append(values)
        stacked = np.stack(channels, axis=1)
        dtype = torch.float32 if self.dtype == "float32" else torch.float64
        return torch.as_tensor(np.ascontiguousarray(stacked), dtype=dtype)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


ValueSource = Union[torch.Tensor, _LazyZarrValues]


class RegionalForecastDataset(Dataset):
    """A single temporal split yielding normalised histories and lead-time targets.

    Each item is a dictionary so PyTorch's default collator can batch it without a custom
    function.  ``inputs`` has shape ``(history, channel, height, width)`` and ``targets`` has
    shape ``(lead, channel, height, width)``.  Timestamps remain int64 Unix nanoseconds in the
    batchable item; their ISO forms and complete lineage live in :attr:`provenance`.
    """

    def __init__(self, values: ValueSource, times_ns: torch.Tensor, frame_indices: Sequence[int],
                 split: str, config: RegionalForecastConfig,
                 normalisation: NormalisationArtifact, provenance: Mapping[str, Any]) -> None:
        shape = tuple(values.shape)
        if len(shape) != 4:
            raise InvalidParameterError("values", shape, "(time, channel, height, width)")
        self.values = values
        self.times_ns = times_ns.to(torch.int64)
        self.frame_indices = tuple(int(i) for i in frame_indices)
        self.split = split
        self.config = config
        self.normalisation = normalisation
        self.provenance = dict(provenance)
        self.provenance["split"] = split
        intervals = torch.diff(self.times_ns)
        unique_intervals = torch.unique(intervals)
        self.time_axis_cadence_ns = (int(unique_intervals[0])
                                     if len(unique_intervals) == 1
                                     and int(unique_intervals[0]) > 0 else 0)

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

    def close(self) -> None:
        """Close this process's lazy store handle; eager datasets require no action."""
        if isinstance(self.values, _LazyZarrValues):
            self.values.close()

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        anchor_local = self._anchors[item]
        input_local = range(anchor_local - self.config.history_frames + 1, anchor_local + 1)
        target_local = [anchor_local + lead for lead in self.config.lead_frames]
        input_global = [self.frame_indices[i] for i in input_local]
        target_global = [self.frame_indices[i] for i in target_local]
        unique_global = list(dict.fromkeys(input_global + target_global))
        if isinstance(self.values, torch.Tensor):
            window = self.values[unique_global]
        else:
            window = self.values.read(unique_global)
        position = {frame: offset for offset, frame in enumerate(unique_global)}
        inputs = (window[[position[v] for v in input_global]] - self._mean) / self._std
        targets = (window[[position[v] for v in target_global]] - self._mean) / self._std
        return {
            "inputs": inputs,
            "targets": targets,
            "input_times_ns": self.times_ns[input_global],
            "target_times_ns": self.times_ns[target_global],
            "lead_durations_ns": (self.times_ns[target_global]
                                  - self.times_ns[input_global[-1]]),
            "time_axis_cadence_ns": torch.tensor(self.time_axis_cadence_ns,
                                                  dtype=torch.int64),
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

    def close(self) -> None:
        """Release any local Zarr handles opened by the current process."""
        for dataset in self.datasets().values():
            dataset.close()

    def __enter__(self) -> "RegionalForecastBundle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


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


def _resolve_lazy_metadata(dataset: Any, config: RegionalForecastConfig
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], str, str]:
    """Validate coordinates/layout without reading any meteorological value chunks."""
    if "time" not in dataset.coords:
        raise InvalidParameterError("dataset", list(dataset.coords), "a time coordinate")
    lat_name = "latitude" if "latitude" in dataset.coords else "lat" if "lat" in dataset.coords else None
    lon_name = "longitude" if "longitude" in dataset.coords else "lon" if "lon" in dataset.coords else None
    if lat_name is None or lon_name is None:
        raise InvalidParameterError("dataset", list(dataset.coords), "latitude/longitude coordinates")

    resolved: Dict[str, str] = {}
    units: Dict[str, Any] = {}
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
                                        "only time, level, latitude and longitude")
        if not np.issubdtype(array.dtype, np.number):
            raise InvalidParameterError("variable %s" % canonical, str(array.dtype),
                                        "a numeric dtype")
        array = array.transpose("time", lat_name, lon_name)
        coords = (np.asarray(array.time.values), np.asarray(array[lat_name].values),
                  np.asarray(array[lon_name].values))
        if reference_coords is not None and any(not np.array_equal(a, b)
                                                for a, b in zip(reference_coords, coords)):
            raise InvalidParameterError("variable alignment", canonical,
                                        "identical timestamps, latitude and longitude for every variable")
        reference_coords = coords
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
    return times, lat, lon, {"resolved_names": resolved, "units": units}, lat_name, lon_name


def _streaming_normalisation(source: _LazyZarrValues, train_indices: Sequence[int],
                             times: np.ndarray, config: RegionalForecastConfig,
                             content_hash: str) -> NormalisationArtifact:
    """Merge float64 block moments with Chan's stable parallel-variance formula."""
    count = 0
    mean = np.zeros(len(config.variables), dtype="float64")
    m2 = np.zeros(len(config.variables), dtype="float64")
    chunk = int(config.statistics_chunk_frames)
    for offset in range(0, len(train_indices), chunk):
        block_indices = train_indices[offset:offset + chunk]
        block = source.read(block_indices).to(torch.float64).numpy()
        flattened = np.moveaxis(block, 1, 0).reshape(len(config.variables), -1)
        block_count = int(flattened.shape[1])
        block_mean = flattened.mean(axis=1, dtype="float64")
        block_m2 = np.square(flattened - block_mean[:, None], dtype="float64").sum(axis=1)
        if count == 0:
            mean, m2, count = block_mean, block_m2, block_count
            continue
        delta = block_mean - mean
        total = count + block_count
        mean = mean + delta * (block_count / total)
        m2 = m2 + block_m2 + delta * delta * (count * block_count / total)
        count = total
    std = np.sqrt(m2 / count)
    train_time_strings = _time_strings(times[list(train_indices)])
    return NormalisationArtifact(
        variables=config.variables, mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std), n_values_per_variable=count,
        fitted_split="train", first_timestamp=train_time_strings[0],
        last_timestamp=train_time_strings[-1], source_content_hash=content_hash,
        computation_method=("per-variable population moments over train frames and all grid "
                            "cells; float64 Chan block merge"), chunk_frames=chunk)


def _cadence_record(times: np.ndarray, config: RegionalForecastConfig) -> Dict[str, Any]:
    """Measure the complete time axis and enforce an explicitly requested cadence."""
    time_ns = times.astype("datetime64[ns]").astype("int64")
    differences = np.diff(time_ns)
    unique = np.unique(differences)
    regular = len(unique) == 1
    actual_ns = int(unique[0]) if regular else None
    expected_ns = (None if config.expected_cadence_hours is None else
                   int(float(config.expected_cadence_hours) * 3_600_000_000_000))
    if expected_ns is not None and (not regular or actual_ns != expected_ns):
        observed = [float(value) / 3_600_000_000_000 for value in unique[:10]]
        raise InvalidParameterError(
            "time cadence", observed,
            "a regular %.12g-hour cadence matching expected_cadence_hours; do not reinterpret "
            "frame offsets as physical lead time" % float(config.expected_cadence_hours))
    return {
        "is_regular": regular,
        "observed_cadence_hours": (None if actual_ns is None else
                                   actual_ns / 3_600_000_000_000),
        "expected_cadence_hours": config.expected_cadence_hours,
        "expectation_checked": expected_ns is not None,
        "unique_interval_hours": [float(value) / 3_600_000_000_000
                                  for value in unique[:10]],
        "interval_count": int(len(differences)),
        "claim_boundary": ("Physical lead durations are computed from sample timestamps. "
                           "A configured cadence is reported as verified only after exact "
                           "agreement with every returned interval."),
    }


def _timeline_split(times: np.ndarray, config: RegionalForecastConfig) -> Dict[str, FieldSequence]:
    # Reuse the platform's accepted R6 implementation.  A one-pixel sentinel represents only
    # the timeline; no meteorological value is copied into this validation structure.
    sentinel = PhysicalField(torch.zeros((1, 1)), grid=GridSpec.pixel((1, 1)))
    timeline = FieldSequence([sentinel] * len(times), times, metadata={"purpose": "T5.2 timeline"})
    if config.calendar_boundaries is None:
        parts = split_temporal(timeline, config.train_ratio, config.val_ratio,
                               embargo_frames=config.embargo_frames)
    else:
        time_ns = times.astype("datetime64[ns]").astype("int64")
        boundaries_ns = tuple(int(np.datetime64(value, "ns").astype("int64"))
                              for value in config.calendar_boundaries)
        missing = [value for value in boundaries_ns if value not in set(time_ns.tolist())]
        if missing:
            raise InvalidParameterError(
                "calendar_boundaries", config.calendar_boundaries,
                "timestamps present exactly on the returned time axis; implicit rounding is refused")
        val_start, test_start = (int(np.flatnonzero(time_ns == value)[0])
                                 for value in boundaries_ns)
        embargo = config.embargo_frames

        def part(start: int, stop: int, name: str) -> FieldSequence:
            if stop <= start:
                raise InvalidParameterError(
                    "calendar split", config.calendar_boundaries,
                    "non-empty train/validation/test windows after the declared embargo")
            return FieldSequence(timeline.fields[start:stop],
                                 timeline.raw_times_slice(slice(start, stop)),
                                 metadata=dict(timeline.metadata), split=name)

        parts = {
            "train": part(0, val_start, "train"),
            "embargo_train_val": part(val_start, val_start + embargo, "embargo"),
            "val": part(val_start + embargo, test_start, "val"),
            "embargo_val_test": part(test_start, test_start + embargo, "embargo"),
            "test": part(test_start + embargo, len(times), "test"),
        }
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
    cadence = _cadence_record(times, config)
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
        "schema": "regional_forecast_dataset/v2",
        "source": dict(source_manifest),
        "source_content_hash": content_hash,
        "config": config.to_provenance(),
        "variables": variable_record,
        "level_hpa": config.level_hpa,
        "timestamps": {"count": len(times), "first": _time_strings(times[:1])[0],
                       "last": _time_strings(times[-1:])[0],
                       "sha256": _stable_hash(_time_strings(times)), "timezone": "UTC"},
        "cadence": cadence,
        "grid": {"latitude": [float(v) for v in lat], "longitude": [float(v) for v in lon],
                 "shape": [len(lat), len(lon)],
                 "sha256": _stable_hash({"latitude": lat.tolist(), "longitude": lon.tolist()}),
                 "longitude_convention": config.longitude_convention},
        "temporal_split": split_record,
        "normalisation": normalisation.to_provenance(),
        "normalisation_contract": {
            "method": "per-variable population z-score",
            "scope": ("each variable over all training times and spatial grid points at the "
                      "selected pressure level"),
            "fitted_split": "train", "ddof": 0,
            "statistics_artifact_sha256": normalisation.to_provenance()["artifact_hash"],
        },
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
    """Prepare a worker-safe, bounded-memory dataset from a local Zarr crop.

    Only coordinates and metadata are read eagerly.  Train statistics are accumulated in
    bounded frame blocks, and sample values are selected on demand by a process-local reader.
    No network fallback is permitted, so local laptop and shared HPC cache use the same call
    with only ``cache_dir`` changed.
    """
    from src.data_layer.zarr_source import cache_path, open_cached_lazy

    dataset, manifest = open_cached_lazy(spec, cache_dir=cache_dir)
    try:
        content_hash = str(manifest.get("content_hash", ""))
        if not content_hash:
            raise InvalidParameterError("source_manifest", sorted(manifest),
                                        "a materialised source record containing content_hash")
        times, lat, lon, variable_record, lat_name, lon_name = _resolve_lazy_metadata(dataset, config)
    finally:
        dataset.close()

    stored_time_chunk = manifest.get("cache_chunking", {}).get("time")
    if stored_time_chunk is None or int(stored_time_chunk) > config.statistics_chunk_frames:
        raise InvalidParameterError(
            "cache time chunk", stored_time_chunk,
            "a recorded on-disk time chunk no larger than statistics_chunk_frames=%d. "
            "Rematerialise this crop with time_chunk=%d or smaller; lazy indexing cannot bound "
            "memory when one Zarr chunk spans a larger interval."
            % (config.statistics_chunk_frames, config.statistics_chunk_frames))

    parts = _timeline_split(times, config)
    cadence = _cadence_record(times, config)
    time_ns_np = times.astype("datetime64[ns]").astype("int64")
    index_by_time = {int(value): i for i, value in enumerate(time_ns_np)}
    frame_indices = {
        name: tuple(index_by_time[int(v)] for v in
                    np.asarray(part.raw_times).astype("datetime64[ns]").astype("int64"))
        for name, part in parts.items()
    }
    path = cache_path(spec, cache_dir)

    def make_source() -> _LazyZarrValues:
        return _LazyZarrValues(
            path=path, resolved_names=variable_record["resolved_names"],
            variables=config.variables, level_hpa=config.level_hpa,
            lat_name=lat_name, lon_name=lon_name,
            shape=(len(times), len(config.variables), len(lat), len(lon)), dtype=config.dtype)

    statistics_source = make_source()
    try:
        normalisation = _streaming_normalisation(
            statistics_source, frame_indices["train"], times, config, content_hash)
    finally:
        statistics_source.close()

    split_record = {
        name: {"n_frames": len(indices), "first_frame_index": indices[0],
               "last_frame_index": indices[-1],
               "first_timestamp": _time_strings(times[[indices[0]]])[0],
               "last_timestamp": _time_strings(times[[indices[-1]]])[0]}
        for name, indices in frame_indices.items()
    }
    provenance = {
        "schema": "regional_forecast_dataset/v2",
        "source": dict(manifest),
        "source_content_hash": content_hash,
        "config": config.to_provenance(),
        "variables": variable_record,
        "level_hpa": config.level_hpa,
        "timestamps": {"count": len(times), "first": _time_strings(times[:1])[0],
                       "last": _time_strings(times[-1:])[0],
                       "sha256": _stable_hash(_time_strings(times)), "timezone": "UTC"},
        "cadence": cadence,
        "grid": {"latitude": [float(v) for v in lat], "longitude": [float(v) for v in lon],
                 "shape": [len(lat), len(lon)],
                 "sha256": _stable_hash({"latitude": lat.tolist(), "longitude": lon.tolist()}),
                 "longitude_convention": config.longitude_convention},
        "temporal_split": split_record,
        "normalisation": normalisation.to_provenance(),
        "normalisation_contract": {
            "method": "per-variable population z-score",
            "scope": ("each variable over all training times and spatial grid points at the "
                      "selected pressure level"),
            "fitted_split": "train", "ddof": 0,
            "statistics_artifact_sha256": normalisation.to_provenance()["artifact_hash"],
        },
        "construction_order": ("split and embargo frames first; stream train-only statistics; "
                               "construct samples within each split"),
        "storage": {"mode": "lazy_local_zarr", "worker_handle_policy": "one handle per process",
                    "statistics_chunk_frames": config.statistics_chunk_frames,
                    "maximum_on_disk_time_chunk_frames": int(stored_time_chunk),
                    "explicit_full_crop_load": False},
        "device_policy": "CPU tensors returned; DataLoader/training loop may move batches to CUDA, ROCm or MPS",
    }
    time_tensor = torch.as_tensor(time_ns_np, dtype=torch.int64)
    made = {
        name: RegionalForecastDataset(make_source(), time_tensor, frame_indices[name], name, config,
                                      normalisation, provenance)
        for name in ("train", "val", "test")
    }
    return RegionalForecastBundle(made["train"], made["val"], made["test"],
                                  normalisation, provenance)


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
        "split_mode": ("calendar_boundaries" if config.calendar_boundaries is not None
                       else "ratios"),
        "calendar_boundaries": (None if config.calendar_boundaries is None
                                else list(config.calendar_boundaries)),
        "expected_cadence_hours": config.expected_cadence_hours,
        "cadence_verified": False,
        "physical_lead_reporting_available": False,
        "dataset_prepared": False,
        "train_only_normalisation_verified": False,
        "independent_era5_crosscheck": "NOT RUN",
        "claim_boundary": ("Structural eligibility uses manifest metadata only. Cadence, physical "
                           "lead durations, split/sample boundaries and train-only statistics are not "
                           "verified until preparation opens the timestamps/values; run an actual "
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
