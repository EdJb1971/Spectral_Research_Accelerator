"""Integrity-checked model artefacts for laboratory forecasters.

The platform deliberately does not import a class named in a manifest.  The laboratory code
constructs its own ``nn.Module`` and this module verifies that the supplied object, declared
configuration and tensor-only checkpoint are the artefact that was recorded.  That keeps the
seam useful for private/HPC models without turning a provenance file into executable code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import nn

from src.forecasting.adapter import ForecasterAdapter, ForecastContractError
from src.transform_engine.training import RepresentationModule


ARTIFACT_SCHEMA = "laboratory-model-artifact/v1"
CHECKPOINT_FILENAME = "state_dict.pt"
MANIFEST_FILENAME = "manifest.json"


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ForecastContractError(
            "model provenance must be finite, JSON-serialisable data: %s" % exc) from exc
    return text.encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualified_class(model: nn.Module) -> str:
    cls = type(model)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


def _state_schema(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
    }


@dataclass(frozen=True)
class LaboratoryModelArtifact:
    """Portable identity record for one tensor-only model checkpoint."""

    schema: str
    model_name: str
    model_class: str
    model_config: Mapping[str, Any]
    representation_config: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    config_sha256: str
    checkpoint_sha256: str
    state_schema_sha256: str
    parameter_count: int
    trainable_parameter_count: int
    checkpoint_format: str
    claim_boundary: str

    def to_provenance(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_provenance(cls, value: Mapping[str, Any]) -> "LaboratoryModelArtifact":
        if not isinstance(value, Mapping):
            raise ForecastContractError("laboratory model manifest must be a JSON object")
        try:
            artifact = cls(**dict(value))
        except TypeError as exc:
            raise ForecastContractError("invalid laboratory model manifest: %s" % exc) from exc
        if artifact.schema != ARTIFACT_SCHEMA:
            raise ForecastContractError(
                "unsupported laboratory model artifact schema %r" % artifact.schema)
        expected = _json_hash({
            "model_config": artifact.model_config,
            "representation_config": artifact.representation_config,
        })
        if artifact.config_sha256 != expected:
            raise ForecastContractError("manifest config hash does not match its configuration")
        if artifact.checkpoint_format != "pytorch-state-dict-weights-only":
            raise ForecastContractError("only tensor-only PyTorch state_dict checkpoints are accepted")
        return artifact


def save_laboratory_artifact(
    directory: Union[str, os.PathLike[str]],
    model: nn.Module,
    *,
    model_config: Mapping[str, Any],
    representation_config: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
    model_name: Optional[str] = None,
) -> LaboratoryModelArtifact:
    """Write a checkpoint and manifest, refusing to overwrite an existing artefact."""
    if not isinstance(model, nn.Module):
        raise ForecastContractError("model must be a torch.nn.Module")
    model_config = dict(model_config)
    representation_config = dict(representation_config)
    training_provenance = dict(training_provenance)
    _canonical_json(model_config)
    _canonical_json(representation_config)
    _canonical_json(training_provenance)

    root = Path(directory)
    checkpoint_path = root / CHECKPOINT_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing laboratory model artefact at %s" % root)
    root.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ForecastContractError("model state_dict must contain only string-to-tensor entries")
    torch.save(state, checkpoint_path)
    parameters = tuple(model.parameters())
    artifact = LaboratoryModelArtifact(
        schema=ARTIFACT_SCHEMA,
        model_name=model_name or type(model).__name__,
        model_class=_qualified_class(model),
        model_config=model_config,
        representation_config=representation_config,
        training_provenance=training_provenance,
        config_sha256=_json_hash({
            "model_config": model_config,
            "representation_config": representation_config,
        }),
        checkpoint_sha256=_file_hash(checkpoint_path),
        state_schema_sha256=_json_hash(_state_schema(state)),
        parameter_count=int(sum(p.numel() for p in parameters)),
        trainable_parameter_count=int(sum(p.numel() for p in parameters if p.requires_grad)),
        checkpoint_format="pytorch-state-dict-weights-only",
        claim_boundary=("Checkpoint identity and configuration provenance only; this artefact "
                        "does not establish training quality or forecast skill."),
    )
    manifest_path.write_bytes(_canonical_json(artifact.to_provenance()) + b"\n")
    return artifact


def load_laboratory_artifact(
    directory: Union[str, os.PathLike[str]],
    model: nn.Module,
    *,
    expected_model_config: Mapping[str, Any],
    expected_representation_config: Mapping[str, Any],
    map_location: Union[str, torch.device] = "cpu",
) -> LaboratoryModelArtifact:
    """Verify an artefact completely, then strictly load it into caller-owned model code."""
    if not isinstance(model, nn.Module):
        raise ForecastContractError("model must be a torch.nn.Module")
    root = Path(directory)
    manifest_path = root / MANIFEST_FILENAME
    checkpoint_path = root / CHECKPOINT_FILENAME
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read laboratory model manifest: %s" % exc) from exc
    artifact = LaboratoryModelArtifact.from_provenance(manifest_value)

    expected_config_hash = _json_hash({
        "model_config": dict(expected_model_config),
        "representation_config": dict(expected_representation_config),
    })
    if artifact.config_sha256 != expected_config_hash:
        raise ForecastContractError("supplied model/representation configuration does not match manifest")
    if artifact.model_class != _qualified_class(model):
        raise ForecastContractError(
            "supplied model class %s does not match manifest %s"
            % (_qualified_class(model), artifact.model_class))
    if not checkpoint_path.is_file():
        raise ForecastContractError("laboratory checkpoint is missing: %s" % checkpoint_path)
    if _file_hash(checkpoint_path) != artifact.checkpoint_sha256:
        raise ForecastContractError("laboratory checkpoint SHA-256 mismatch; file may be corrupted or replaced")

    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ForecastContractError("cannot safely load tensor-only checkpoint: %s" % exc) from exc
    if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ForecastContractError("checkpoint is not a string-to-tensor state_dict")
    if _json_hash(_state_schema(state)) != artifact.state_schema_sha256:
        raise ForecastContractError("checkpoint tensor schema does not match manifest")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ForecastContractError("checkpoint is incompatible with supplied model: %s" % exc) from exc
    if sum(p.numel() for p in model.parameters()) != artifact.parameter_count:
        raise ForecastContractError("supplied model parameter count does not match manifest")
    return artifact


def load_laboratory_forecaster(
    directory: Union[str, os.PathLike[str]],
    model: nn.Module,
    representation: RepresentationModule,
    *,
    expected_model_config: Mapping[str, Any],
    expected_representation_config: Mapping[str, Any],
    map_location: Union[str, torch.device] = "cpu",
) -> tuple[ForecasterAdapter, LaboratoryModelArtifact]:
    """Load verified weights and bind their identity into forecast provenance."""
    if not isinstance(representation, RepresentationModule):
        raise ForecastContractError("representation must be an accepted RepresentationModule")
    declared_name = expected_representation_config.get("name")
    if declared_name != representation.name:
        raise ForecastContractError(
            "representation instance %r does not match declared configuration %r"
            % (representation.name, declared_name))
    artifact = load_laboratory_artifact(
        directory, model, expected_model_config=expected_model_config,
        expected_representation_config=expected_representation_config,
        map_location=map_location)
    adapter = ForecasterAdapter(
        model, representation, model_name=artifact.model_name,
        model_artifact=artifact.to_provenance())
    return adapter, artifact
