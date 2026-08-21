"""Offline FourCastNet 3 request/result contracts (roadmap T5.6a).

This module deliberately imports no Earth2Studio, Makani, torch-harmonics, NGC or network
client.  The portable application plans and verifies work; an optional isolated worker may
execute it later.  FCN3 is a global spherical model, so a regional initial condition is never
accepted as a cheaper substitute for its declared global state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from src.forecasting.adapter import ForecastContractError


EXTERNAL_RUN_SCHEMA = "external-forecast-run/fcn3-v1"
EXTERNAL_RESULT_SCHEMA = "external-forecast-result/fcn3-v1"
RUN_FILENAME = "external-forecast-run.json"
RESULT_FILENAME = "external-forecast-result.json"
FCN3_MODEL_NAME = "FourCastNet 3"
FCN3_PARAMETER_COUNT = 710_867_670
FCN3_GRID_SHAPE = (721, 1440)
FCN3_GRID_DEGREES = 0.25
FCN3_CADENCE_HOURS = 6.0

_PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
FCN3_VARIABLES: Tuple[str, ...] = (
    "u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv",
    *("u%d" % level for level in _PRESSURE_LEVELS),
    *("v%d" % level for level in _PRESSURE_LEVELS),
    *("z%d" % level for level in _PRESSURE_LEVELS),
    *("t%d" % level for level in _PRESSURE_LEVELS),
    *("q%d" % level for level in _PRESSURE_LEVELS),
)
NZ_850_VARIABLES: Tuple[str, ...] = ("t850", "q850", "u850", "v850", "z850")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(
    r"^(?:unknown|tbd|todo|n/?a|not[ -]?specified|unspecified|unassigned|\?+)$", re.I)


class _FrozenMap(Mapping[str, Any]):
    def __init__(self, items: Sequence[Tuple[str, Any]]) -> None:
        self._items = tuple(items)
        self._values = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_FrozenMap":
        return self


def _freeze(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            if not isinstance(key, str) or not key.strip():
                raise ForecastContractError("%s contains an invalid JSON key" % path)
            items.append((key, _freeze(value[key], "%s.%s" % (path, key))))
        return _FrozenMap(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, "%s[%d]" % (path, index))
                     for index, item in enumerate(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ForecastContractError("%s must contain only finite JSON values" % path)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError(
            "FCN3 contract must contain finite JSON-serialisable data: %s" % exc) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Any, expected: Sequence[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForecastContractError("%s must be a JSON object" % path)
    wanted, supplied = set(expected), set(value)
    missing, unknown = sorted(wanted - supplied), sorted(supplied - wanted)
    if missing or unknown:
        raise ForecastContractError(
            "%s fields differ: missing=%r unknown=%r" % (path, missing, unknown))
    return value


def _sha(value: Any, path: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ForecastContractError("%s must be a lowercase 64-character SHA-256" % path)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or _PLACEHOLDER.fullmatch(value.strip()):
        raise ForecastContractError("%s must be an explicit non-placeholder string" % path)
    return value


def _positive_int(value: Any, path: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ForecastContractError("%s must be an integer >= %d" % (path, lower))
    return value


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastContractError("%s must be a positive finite number" % path)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ForecastContractError("%s must be a positive finite number" % path)
    return number


def _timestamp(value: Any, path: str) -> datetime:
    _text(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ForecastContractError("%s must be an ISO-8601 timestamp" % path) from None
    if parsed.tzinfo is None:
        raise ForecastContractError("%s must include an explicit UTC offset" % path)
    return parsed


def _unique_strings(value: Any, path: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ForecastContractError("%s must be a non-empty JSON array" % path)
    result = tuple(value)
    for index, item in enumerate(result):
        _text(item, "%s[%d]" % (path, index))
    if len(set(result)) != len(result):
        raise ForecastContractError("%s must not contain duplicates" % path)
    return result


@dataclass(frozen=True)
class FCN3ModelContract:
    provider: str
    name: str
    model_version: str
    parameter_count: int
    package_reference: str
    package_sha256: str
    checkpoint_sha256: str
    model_card_reference: str
    model_card_sha256: str
    license_spdx: str

    def __post_init__(self) -> None:
        for name in ("provider", "name", "model_version", "package_reference",
                     "model_card_reference", "license_spdx"):
            _text(getattr(self, name), "model.%s" % name)
        for name in ("package_sha256", "checkpoint_sha256", "model_card_sha256"):
            _sha(getattr(self, name), "model.%s" % name)
        if self.provider != "NVIDIA" or self.name != FCN3_MODEL_NAME:
            raise ForecastContractError("T5.6a accepts only the declared NVIDIA FourCastNet 3 model")
        if self.parameter_count != FCN3_PARAMETER_COUNT:
            raise ForecastContractError(
                "FCN3 parameter_count must equal the reviewed model-card value %d"
                % FCN3_PARAMETER_COUNT)
        if self.license_spdx != "Apache-2.0":
            raise ForecastContractError("FCN3 license_spdx must match the reviewed Apache-2.0 record")


@dataclass(frozen=True)
class FCN3SoftwareContract:
    runner: str
    earth2studio_version: str
    torch_version: str
    torch_harmonics_version: str
    environment_lock_reference: str
    environment_lock_sha256: str

    def __post_init__(self) -> None:
        for name in ("runner", "earth2studio_version", "torch_version",
                     "torch_harmonics_version", "environment_lock_reference"):
            _text(getattr(self, name), "software.%s" % name)
        _sha(self.environment_lock_sha256, "software.environment_lock_sha256")
        if self.runner != "earth2studio.models.px.FCN3":
            raise ForecastContractError("FCN3 runner must be earth2studio.models.px.FCN3")


@dataclass(frozen=True)
class FCN3InputContract:
    source_name: str
    source_version: str
    source_reference: str
    source_sha256: str
    variables: Tuple[str, ...]
    scope: str
    grid_shape: Tuple[int, int]
    grid_spacing_degrees: float
    latitude_order: str
    longitude_convention: str
    grid_coordinates_sha256: str
    initialization_times: Tuple[str, ...]
    normalisation_reference: str
    normalisation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "grid_shape", tuple(self.grid_shape))
        object.__setattr__(self, "initialization_times", tuple(self.initialization_times))
        for name in ("source_name", "source_version", "source_reference", "scope",
                     "latitude_order", "longitude_convention", "normalisation_reference"):
            _text(getattr(self, name), "input.%s" % name)
        for name in ("source_sha256", "grid_coordinates_sha256", "normalisation_sha256"):
            _sha(getattr(self, name), "input.%s" % name)
        if self.scope != "global":
            raise ForecastContractError(
                "FCN3 input scope must be global; regional initial conditions are invalid")
        if self.variables != FCN3_VARIABLES:
            raise ForecastContractError(
                "FCN3 input variables must contain all 72 channels in canonical model-card order")
        if self.grid_shape != FCN3_GRID_SHAPE:
            raise ForecastContractError(
                "FCN3 input grid must be global 721x1440; crop only after inference")
        if not math.isclose(float(self.grid_spacing_degrees), FCN3_GRID_DEGREES,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ForecastContractError("FCN3 input grid spacing must be exactly 0.25 degrees")
        if self.latitude_order not in ("ascending", "descending"):
            raise ForecastContractError("input.latitude_order must be ascending or descending")
        if self.longitude_convention not in ("-180..180", "0..360"):
            raise ForecastContractError(
                "input.longitude_convention must be '-180..180' or '0..360'")
        if not self.initialization_times:
            raise ForecastContractError("input.initialization_times must not be empty")
        parsed = tuple(_timestamp(value, "input.initialization_times")
                       for value in self.initialization_times)
        if tuple(sorted(parsed)) != parsed or len(set(parsed)) != len(parsed):
            raise ForecastContractError(
                "input.initialization_times must be unique and strictly increasing")


@dataclass(frozen=True)
class FCN3RolloutContract:
    cadence_hours: float
    steps: int
    ensemble_size: int
    member_seeds: Tuple[int, ...]
    stochastic_process: str
    perturbation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_seeds", tuple(self.member_seeds))
        if not math.isclose(float(self.cadence_hours), FCN3_CADENCE_HOURS,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ForecastContractError("FCN3 rollout cadence must be exactly 6 hours")
        _positive_int(self.steps, "rollout.steps")
        _positive_int(self.ensemble_size, "rollout.ensemble_size")
        if len(self.member_seeds) != self.ensemble_size:
            raise ForecastContractError("rollout requires exactly one seed per ensemble member")
        for seed in self.member_seeds:
            _positive_int(seed, "rollout.member_seeds", allow_zero=True)
        if len(set(self.member_seeds)) != len(self.member_seeds):
            raise ForecastContractError("rollout.member_seeds must be unique")
        _text(self.stochastic_process, "rollout.stochastic_process")
        if self.perturbation != "none":
            raise ForecastContractError(
                "FCN3 v1 uses intrinsic hidden-state stochasticity; external perturbation must be none")

    @property
    def lead_hours(self) -> Tuple[float, ...]:
        return tuple(self.cadence_hours * step for step in range(1, self.steps + 1))


@dataclass(frozen=True)
class FCN3OutputContract:
    variables: Tuple[str, ...]
    artifact_format: str
    artifact_reference: str
    crop_policy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _unique_strings(self.variables, "output.variables"))
        unknown = tuple(value for value in self.variables if value not in FCN3_VARIABLES)
        if unknown:
            raise ForecastContractError("output variables are not FCN3 channels: %r" % (unknown,))
        if self.artifact_format not in ("netcdf4", "zarr"):
            raise ForecastContractError("output.artifact_format must be netcdf4 or zarr")
        _text(self.artifact_reference, "output.artifact_reference")
        if self.crop_policy != "global_then_crop":
            raise ForecastContractError("FCN3 output crop_policy must be global_then_crop")


@dataclass(frozen=True)
class FCN3RuntimeContract:
    mode: str
    operating_system: Optional[str]
    accelerator: Optional[str]
    hardware_model: Optional[str]
    precision: str

    def __post_init__(self) -> None:
        if self.mode not in ("prepare_only", "external_worker"):
            raise ForecastContractError("runtime.mode must be prepare_only or external_worker")
        if self.precision not in ("bf16", "float32"):
            raise ForecastContractError("runtime.precision must be bf16 or float32")
        if self.mode == "prepare_only":
            if any(value is not None for value in (
                    self.operating_system, self.accelerator, self.hardware_model)):
                raise ForecastContractError(
                    "prepare_only runtime must leave OS, accelerator and hardware unassigned")
        else:
            for name in ("operating_system", "accelerator", "hardware_model"):
                _text(getattr(self, name), "runtime.%s" % name)
            if self.operating_system != "linux" or self.accelerator != "nvidia_cuda":
                raise ForecastContractError(
                    "accepted FCN3 external workers currently require Linux and NVIDIA CUDA")


_T = TypeVar("_T")


def _build(section: str, cls: Type[_T], value: Any) -> _T:
    fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    record = dict(_exact(value, fields, section))
    tuple_fields = {"variables", "grid_shape", "initialization_times", "member_seeds"}
    for name in tuple_fields & set(record):
        if not isinstance(record[name], (list, tuple)):
            raise ForecastContractError("%s.%s must be a JSON array" % (section, name))
        record[name] = tuple(record[name])
    try:
        return cls(**record)
    except TypeError as exc:
        raise ForecastContractError("invalid %s contract: %s" % (section, exc)) from exc


@dataclass(frozen=True)
class ExternalForecastRun:
    schema: str
    request_revision: int
    run_id: str
    evidence_reference: str
    evidence_sha256: str
    model: FCN3ModelContract
    software: FCN3SoftwareContract
    input: FCN3InputContract
    rollout: FCN3RolloutContract
    output: FCN3OutputContract
    runtime: FCN3RuntimeContract

    def __post_init__(self) -> None:
        if self.schema != EXTERNAL_RUN_SCHEMA:
            raise ForecastContractError("unsupported external forecast run schema")
        _positive_int(self.request_revision, "request_revision")
        _text(self.run_id, "run_id")
        _text(self.evidence_reference, "evidence_reference")
        _sha(self.evidence_sha256, "evidence_sha256")

    def to_mapping(self) -> Dict[str, Any]:
        return _thaw(asdict(self))

    def fingerprint(self) -> str:
        return _fingerprint(self.to_mapping())

    def __hash__(self) -> int:
        return int(self.fingerprint()[:16], 16)

    def to_provenance(self) -> Dict[str, Any]:
        return {"request": self.to_mapping(), "request_sha256": self.fingerprint()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalForecastRun":
        fields = tuple(cls.__dataclass_fields__)
        record = dict(_exact(value, fields, "request"))
        sections = {
            "model": FCN3ModelContract, "software": FCN3SoftwareContract,
            "input": FCN3InputContract, "rollout": FCN3RolloutContract,
            "output": FCN3OutputContract, "runtime": FCN3RuntimeContract,
        }
        for name, section_type in sections.items():
            record[name] = _build(name, section_type, record[name])
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid FCN3 request: %s" % exc) from exc


@dataclass(frozen=True)
class ExternalForecastResult:
    schema: str
    request_sha256: str
    artifact_reference: str
    artifact_format: str
    artifact_kind: str
    artifact_sha256: str
    output_variables: Tuple[str, ...]
    grid_shape: Tuple[int, int]
    initialization_times: Tuple[str, ...]
    cadence_hours: float
    steps: int
    ensemble_size: int
    precision: str
    operating_system: str
    accelerator: str
    hardware_model: str
    peak_vram_bytes: int
    peak_ram_bytes: int
    wall_time_seconds: float
    worker_log_sha256: str
    result_sha256: str
    claim_boundary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_variables", tuple(self.output_variables))
        object.__setattr__(self, "grid_shape", tuple(self.grid_shape))
        object.__setattr__(self, "initialization_times", tuple(self.initialization_times))
        if self.schema != EXTERNAL_RESULT_SCHEMA:
            raise ForecastContractError("unsupported external forecast result schema")
        for name in ("request_sha256", "artifact_sha256", "worker_log_sha256", "result_sha256"):
            _sha(getattr(self, name), name)
        for name in ("artifact_reference", "artifact_format", "artifact_kind", "precision",
                     "operating_system", "accelerator", "hardware_model", "claim_boundary"):
            _text(getattr(self, name), name)
        for name in ("steps", "ensemble_size", "peak_vram_bytes", "peak_ram_bytes"):
            _positive_int(getattr(self, name), name)
        _positive_number(self.cadence_hours, "cadence_hours")
        _positive_number(self.wall_time_seconds, "wall_time_seconds")
        _unique_strings(self.output_variables, "result.output_variables")
        if any(variable not in FCN3_VARIABLES for variable in self.output_variables):
            raise ForecastContractError("result contains variables outside the FCN3 contract")
        if self.grid_shape != FCN3_GRID_SHAPE:
            raise ForecastContractError("FCN3 result grid must remain global 721x1440")
        if not math.isclose(float(self.cadence_hours), FCN3_CADENCE_HOURS,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ForecastContractError("FCN3 result cadence must remain exactly 6 hours")
        if self.artifact_format not in ("netcdf4", "zarr"):
            raise ForecastContractError("result artifact_format must be netcdf4 or zarr")
        expected_kind = "file" if self.artifact_format == "netcdf4" else "tree"
        if self.artifact_kind != expected_kind:
            raise ForecastContractError(
                "%s results must be recorded as a %s artifact" %
                (self.artifact_format, expected_kind))
        if self.operating_system != "linux" or self.accelerator != "nvidia_cuda":
            raise ForecastContractError(
                "accepted FCN3 results currently require a Linux NVIDIA CUDA worker")
        if self.precision not in ("bf16", "float32"):
            raise ForecastContractError("result precision must be bf16 or float32")
        if not self.initialization_times:
            raise ForecastContractError("result initialization_times must not be empty")
        for value in self.initialization_times:
            _timestamp(value, "result.initialization_times")
        body = self.to_mapping(include_result_sha256=False)
        if _fingerprint(body) != self.result_sha256:
            raise ForecastContractError("external forecast result SHA-256 mismatch")

    def to_mapping(self, *, include_result_sha256: bool = True) -> Dict[str, Any]:
        record = _thaw(asdict(self))
        if not include_result_sha256:
            record.pop("result_sha256")
        return record

    def to_provenance(self) -> Dict[str, Any]:
        return self.to_mapping()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalForecastResult":
        fields = tuple(cls.__dataclass_fields__)
        record = dict(_exact(value, fields, "result"))
        for name in ("output_variables", "grid_shape", "initialization_times"):
            if not isinstance(record[name], (list, tuple)):
                raise ForecastContractError("result.%s must be a JSON array" % name)
            record[name] = tuple(record[name])
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid FCN3 result: %s" % exc) from exc


def save_external_forecast_run(
    path: Union[str, os.PathLike[str]], request: ExternalForecastRun,
) -> str:
    if not isinstance(request, ExternalForecastRun):
        raise ForecastContractError("request must be a validated ExternalForecastRun")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(request.to_provenance()) + b"\n")
    except FileExistsError:
        raise FileExistsError("refusing to overwrite FCN3 run request at %s" % target) from None
    return request.fingerprint()


def load_external_forecast_run(path: Union[str, os.PathLike[str]]) -> ExternalForecastRun:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read FCN3 run request: %s" % exc) from exc
    envelope = _exact(envelope, ("request", "request_sha256"), "request envelope")
    request = ExternalForecastRun.from_mapping(envelope["request"])
    if envelope["request_sha256"] != request.fingerprint():
        raise ForecastContractError("FCN3 run request SHA-256 mismatch")
    return request


def artifact_sha256(path: Union[str, os.PathLike[str]]) -> Tuple[str, str]:
    """Hash a file or directory tree deterministically without following symlinks."""
    target = Path(path)
    if target.is_symlink():
        raise ForecastContractError("external forecast artifacts must not be symbolic links")
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(b"file\0")
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return "file", digest.hexdigest()
    if not target.is_dir():
        raise ForecastContractError("external forecast artifact does not exist: %s" % target)
    digest.update(b"tree\0")
    files = sorted(item for item in target.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ForecastContractError("external forecast artifact directory is empty")
    for item in files:
        if item.is_symlink():
            raise ForecastContractError("external forecast artifact trees must not contain symlinks")
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return "tree", digest.hexdigest()


def seal_external_forecast_result(
    request: ExternalForecastRun,
    artifact_path: Union[str, os.PathLike[str]],
    *,
    peak_vram_bytes: int,
    peak_ram_bytes: int,
    wall_time_seconds: float,
    worker_log_sha256: str,
) -> ExternalForecastResult:
    """Bind a completed worker artifact to the exact request and measured resources."""
    if not isinstance(request, ExternalForecastRun):
        raise ForecastContractError("request must be a validated ExternalForecastRun")
    if request.runtime.mode != "external_worker":
        raise ForecastContractError("prepare-only FCN3 requests cannot claim a completed result")
    kind, content_hash = artifact_sha256(artifact_path)
    expected_kind = "file" if request.output.artifact_format == "netcdf4" else "tree"
    if kind != expected_kind:
        raise ForecastContractError(
            "%s output requires a %s artifact, got %s" %
            (request.output.artifact_format, expected_kind, kind))
    body = {
        "schema": EXTERNAL_RESULT_SCHEMA,
        "request_sha256": request.fingerprint(),
        "artifact_reference": request.output.artifact_reference,
        "artifact_format": request.output.artifact_format,
        "artifact_kind": kind,
        "artifact_sha256": content_hash,
        "output_variables": list(request.output.variables),
        "grid_shape": list(FCN3_GRID_SHAPE),
        "initialization_times": list(request.input.initialization_times),
        "cadence_hours": request.rollout.cadence_hours,
        "steps": request.rollout.steps,
        "ensemble_size": request.rollout.ensemble_size,
        "precision": request.runtime.precision,
        "operating_system": str(request.runtime.operating_system),
        "accelerator": str(request.runtime.accelerator),
        "hardware_model": str(request.runtime.hardware_model),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_ram_bytes": peak_ram_bytes,
        "wall_time_seconds": wall_time_seconds,
        "worker_log_sha256": worker_log_sha256,
        "claim_boundary": (
            "Artifact identity and declared worker execution only; this record does not verify "
            "the forecast-array schema, meteorological values, ensemble calibration, spectral "
            "fidelity, or scientific skill."),
    }
    return ExternalForecastResult.from_mapping({**body, "result_sha256": _fingerprint(body)})


def verify_external_forecast_artifact(
    result: ExternalForecastResult, artifact_path: Union[str, os.PathLike[str]],
) -> None:
    if not isinstance(result, ExternalForecastResult):
        raise ForecastContractError("result must be a validated ExternalForecastResult")
    kind, content_hash = artifact_sha256(artifact_path)
    if kind != result.artifact_kind or content_hash != result.artifact_sha256:
        raise ForecastContractError("external forecast artifact SHA-256 mismatch")


def verify_external_forecast_result(
    request: ExternalForecastRun,
    result: ExternalForecastResult,
    artifact_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> None:
    """Verify a returned manifest against its request and optionally its artifact bytes."""
    if not isinstance(request, ExternalForecastRun) or not isinstance(
            result, ExternalForecastResult):
        raise ForecastContractError("request and result must be validated FCN3 contracts")
    expected = {
        "request_sha256": request.fingerprint(),
        "artifact_reference": request.output.artifact_reference,
        "artifact_format": request.output.artifact_format,
        "output_variables": request.output.variables,
        "grid_shape": FCN3_GRID_SHAPE,
        "initialization_times": request.input.initialization_times,
        "cadence_hours": request.rollout.cadence_hours,
        "steps": request.rollout.steps,
        "ensemble_size": request.rollout.ensemble_size,
        "precision": request.runtime.precision,
        "operating_system": request.runtime.operating_system,
        "accelerator": request.runtime.accelerator,
        "hardware_model": request.runtime.hardware_model,
    }
    mismatches = [name for name, value in expected.items() if getattr(result, name) != value]
    if mismatches:
        raise ForecastContractError(
            "external forecast result differs from its request: %s" % ", ".join(mismatches))
    if artifact_path is not None:
        verify_external_forecast_artifact(result, artifact_path)


def save_external_forecast_result(
    path: Union[str, os.PathLike[str]], result: ExternalForecastResult,
) -> str:
    if not isinstance(result, ExternalForecastResult):
        raise ForecastContractError("result must be a validated ExternalForecastResult")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(result.to_mapping()) + b"\n")
    except FileExistsError:
        raise FileExistsError("refusing to overwrite FCN3 result manifest at %s" % target) from None
    return result.result_sha256


def load_external_forecast_result(path: Union[str, os.PathLike[str]]) -> ExternalForecastResult:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read FCN3 result manifest: %s" % exc) from exc
    return ExternalForecastResult.from_mapping(record)
