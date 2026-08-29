"""TG5.1: durable, content-addressed motif definitions.

TG3.5 freezes a confirmatory *label* inside one process.  That is enough to spend a held-out
partition once, but it is not a transferable definition: the exemplar graph still travels as
a live ``MotifCandidate``.  This module is the process boundary required by rule R20.  It stores
the exact dimensionless graph, matcher and measured tolerance separately from the originating
domain records, binds both to canonical SHA-256 digests, and publishes with exclusive creation.

A content hash detects mutation; it cannot prove when a file was created.  Transfer therefore
has to compare ``motif_sha256`` with a digest published before the target domain was opened.
TG5.2 owns that chronological/opening ledger.  TG5.1 supplies the immutable object it will bind.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.core.constellation import AttributedGraph, RelationValue
from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.motif import MiningResult, MotifCandidate, report_motif_generation
from src.core.preregistration import PartitionIdentity


FROZEN_MOTIF_SCHEMA = "frozen-motif/v1"
_EDGE = re.compile(r"^(\d+)->(\d+)$")
_TOP_LEVEL_FIELDS = (
    "schema", "study_id", "frozen_at", "source", "source_partition",
    "source_partition_sha256", "origin_domain", "origin_nodes", "definition",
    "definition_sha256", "motif_sha256",
)
_DEFINITION_FIELDS = (
    "size", "matcher", "tolerance", "attributes", "relations", "edges", "refusals",
)
_ORIGIN_DOMAIN_FIELDS = (
    "name", "description", "licence", "axes", "violations", "lag_policy",
    "declared_floor_frames", "declared_floor_basis", "precedence_admissible", "provenance",
)
_ORIGIN_NODE_FIELDS = (
    "domain", "dataset", "variable", "representation", "units", "time_units",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively remove mutation paths while retaining a mapping-shaped public record."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _exact_keys(value: Any, expected: Sequence[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidParameterError(path, type(value).__name__, "a JSON object")
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise InvalidParameterError(
            path, {"missing": missing, "extra": extra},
            "exactly these fields: %s. A frozen schema must reject both omitted meaning and "
            "unrecognised meaning" % ", ".join(expected))
    return value


def _timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidParameterError("frozen_at", value, "a timezone-bearing ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidParameterError(
            "frozen_at", value, "a timezone-bearing ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidParameterError(
            "frozen_at", value,
            "a timestamp with an explicit UTC offset. TG5.2 must be able to establish that "
            "the target domain was opened after this definition was frozen")
    return value


def _graph_definition(candidate: MotifCandidate, result: MiningResult) -> Dict[str, Any]:
    graph = candidate.exemplar.graph
    described = graph.describe()
    return {
        "size": int(result.size),
        "matcher": str(result.matcher),
        "tolerance": float(result.tolerance),
        "attributes": described["attributes"],
        "relations": described["relations"],
        "edges": described["edges"],
        "refusals": described["refusals"],
    }


def _partition_from_mapping(value: Mapping[str, Any]) -> PartitionIdentity:
    fields = ("name", "n_times", "n_channels", "channel_labels", "frames", "provenance")
    record = _exact_keys(value, fields, "source_partition")
    frames = record["frames"]
    if not isinstance(frames, (list, tuple)) or len(frames) != 2:
        raise InvalidParameterError("source_partition.frames", frames, "two frame bounds")
    return PartitionIdentity(
        name=record["name"], n_times=int(record["n_times"]),
        n_channels=int(record["n_channels"]),
        channel_labels=tuple(record["channel_labels"]),
        frames=(int(frames[0]), int(frames[1])),
        provenance=dict(record["provenance"]),
    )


def _edge_key(value: str, n_nodes: int, path: str) -> Tuple[int, int]:
    matched = _EDGE.fullmatch(str(value))
    if matched is None:
        raise InvalidParameterError(path, value, "an edge key in the form 'left->right'")
    pair = (int(matched.group(1)), int(matched.group(2)))
    if pair[0] >= n_nodes or pair[1] >= n_nodes or pair[0] == pair[1]:
        raise InvalidParameterError(path, value, "two distinct node indices inside the motif")
    return pair


def _graph_from_records(definition: Mapping[str, Any],
                        origin_nodes: Sequence[Mapping[str, Any]]) -> AttributedGraph:
    definition = _exact_keys(definition, _DEFINITION_FIELDS, "definition")
    attributes = tuple(dict(value) for value in definition["attributes"])
    n_nodes = len(attributes)
    if int(definition["size"]) != n_nodes:
        raise InvalidParameterError(
            "definition.size", definition["size"],
            "%d, the number of nodes in the serialized structural graph" % n_nodes)
    edges: Dict[Tuple[int, int], Dict[str, RelationValue]] = {}
    for name, values in dict(definition["edges"]).items():
        pair = _edge_key(name, n_nodes, "definition.edges")
        edge: Dict[str, RelationValue] = {}
        for relation, raw in dict(values).items():
            record = _exact_keys(raw, ("relation", "value", "holds", "basis"),
                                 "definition.edges.%s.%s" % (name, relation))
            if record["relation"] != relation:
                raise InvalidParameterError(
                    "definition.edges.%s.%s.relation" % (name, relation), record["relation"],
                    "the relation named by its enclosing edge key")
            edge[str(relation)] = RelationValue(
                relation=str(record["relation"]), value=record["value"],
                holds=record["holds"], basis=str(record["basis"]))
        edges[pair] = edge
    refusals = {
        _edge_key(name, n_nodes, "definition.refusals"): dict(values)
        for name, values in dict(definition["refusals"]).items()
    }
    return AttributedGraph(
        attributes=attributes, carried=tuple(dict(value) for value in origin_nodes),
        edges=edges, relations=tuple(definition["relations"]), refusals=refusals,
    )


class FrozenMotifIntegrityError(InvalidParameterError):
    """The serialized motif no longer agrees with one of its recorded content hashes."""

    def __init__(self, field: str, recorded: str, recomputed: str) -> None:
        super().__init__(
            field, recomputed,
            "the recorded SHA-256 %s. The frozen motif changed after its digest was "
            "recorded, so it is not the prior definition rule R20 permits for transfer"
            % recorded,
            recorded_sha256=recorded, recomputed_sha256=recomputed)


@dataclass(frozen=True)
class FrozenMotif:
    """One process-independent motif definition and the origin it must never erase."""

    study_id: str
    frozen_at: str
    source: Mapping[str, Any]
    source_partition: Mapping[str, Any]
    source_partition_sha256: str
    origin_domain: Mapping[str, Any]
    origin_nodes: Tuple[Mapping[str, Any], ...]
    definition: Mapping[str, Any]
    definition_sha256: str = ""
    motif_sha256: str = ""
    schema: str = dc_field(default=FROZEN_MOTIF_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise InvalidParameterError(
                "study_id", self.study_id,
                "a non-empty study identity. A transferable motif without a study cannot "
                "be placed back into the family that selected it")
        object.__setattr__(self, "frozen_at", _timestamp(self.frozen_at))
        object.__setattr__(self, "source", _freeze(dict(self.source)))
        object.__setattr__(self, "source_partition", _freeze(dict(self.source_partition)))
        object.__setattr__(self, "origin_domain", _freeze(dict(self.origin_domain)))
        object.__setattr__(self, "origin_nodes",
                           tuple(_freeze(dict(value)) for value in self.origin_nodes))
        object.__setattr__(self, "definition", _freeze(dict(self.definition)))

        _exact_keys(self.source, (
            "label", "generate_sha256", "generate_family_size", "support",
            "n_occurrences", "n_examined"), "source")
        partition = _partition_from_mapping(self.source_partition)
        if partition.digest() != self.source_partition_sha256:
            raise FrozenMotifIntegrityError(
                "source_partition_sha256", self.source_partition_sha256, partition.digest())
        _exact_keys(self.definition, _DEFINITION_FIELDS, "definition")
        graph = _graph_from_records(self.definition, self.origin_nodes)
        if not math.isfinite(float(self.definition["tolerance"])) \
                or float(self.definition["tolerance"]) < 0.0:
            raise InvalidParameterError(
                "definition.tolerance", self.definition["tolerance"],
                "a non-negative finite tolerance measured before transfer")
        _exact_keys(self.origin_domain, _ORIGIN_DOMAIN_FIELDS, "origin_domain")
        domain_name = self.origin_domain.get("name")
        if not isinstance(domain_name, str) or not domain_name.strip():
            raise InvalidParameterError("origin_domain.name", domain_name, "a named origin domain")
        if not isinstance(self.origin_domain.get("licence"), str) \
                or not str(self.origin_domain["licence"]).strip():
            raise InvalidParameterError(
                "origin_domain.licence", self.origin_domain.get("licence"),
                "the originating domain's licence or terms. Transfer cannot erase the "
                "conditions attached to the source definition")
        for index, record in enumerate(self.origin_nodes):
            missing = sorted(set(_ORIGIN_NODE_FIELDS) - set(record))
            if missing:
                raise InvalidParameterError(
                    "origin_nodes[%d]" % index, {"missing": missing},
                    "the carried semantic fields %s. These records stay outside structural "
                    "comparison, but rule R19 does not allow transfer to discard them"
                    % ", ".join(_ORIGIN_NODE_FIELDS))
        node_domains = {record.get("domain") for record in self.origin_nodes}
        if node_domains != {domain_name}:
            raise InvalidParameterError(
                "origin_nodes.domain", sorted(str(value) for value in node_domains),
                "the single originating domain %r. A cross-domain exemplar cannot be "
                "laundered into a motif said to originate in one domain" % domain_name)
        if graph.n_nodes != len(self.origin_nodes):  # AttributedGraph also checks; explicit message.
            raise InvalidParameterError(
                "origin_nodes", len(self.origin_nodes), "one origin record per structural node")

        definition_sha256 = _digest(self.definition)
        if self.definition_sha256 and self.definition_sha256 != definition_sha256:
            raise FrozenMotifIntegrityError(
                "definition_sha256", self.definition_sha256, definition_sha256)
        object.__setattr__(self, "definition_sha256", definition_sha256)
        motif_sha256 = _digest(self.body())
        if self.motif_sha256 and self.motif_sha256 != motif_sha256:
            raise FrozenMotifIntegrityError("motif_sha256", self.motif_sha256, motif_sha256)
        object.__setattr__(self, "motif_sha256", motif_sha256)

    @property
    def graph(self) -> AttributedGraph:
        """Reconstruct the exact exemplar graph without consulting the originating process."""
        return _graph_from_records(self.definition, self.origin_nodes)

    def body(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "frozen_at": self.frozen_at,
            "source": _thaw(self.source),
            "source_partition": _thaw(self.source_partition),
            "source_partition_sha256": self.source_partition_sha256,
            "origin_domain": _thaw(self.origin_domain),
            "origin_nodes": _thaw(self.origin_nodes),
            "definition": _thaw(self.definition),
            "definition_sha256": self.definition_sha256,
        }

    def to_mapping(self) -> Dict[str, Any]:
        record = self.body()
        record["motif_sha256"] = self.motif_sha256
        return record

    def verify(self) -> Dict[str, Any]:
        definition_sha256 = _digest(self.definition)
        if definition_sha256 != self.definition_sha256:
            raise FrozenMotifIntegrityError(
                "definition_sha256", self.definition_sha256, definition_sha256)
        motif_sha256 = _digest(self.body())
        if motif_sha256 != self.motif_sha256:
            raise FrozenMotifIntegrityError("motif_sha256", self.motif_sha256, motif_sha256)
        return {"definition_sha256": definition_sha256, "motif_sha256": motif_sha256,
                "matches": True}

    def verify_published(self, published_sha256: str) -> Dict[str, Any]:
        self.verify()
        if not isinstance(published_sha256, str) or not published_sha256.strip():
            raise InvalidParameterError(
                "published_sha256", published_sha256,
                "the motif SHA-256 published before the target domain was opened")
        if published_sha256 != self.motif_sha256:
            raise FrozenMotifIntegrityError(
                "published_sha256", published_sha256, self.motif_sha256)
        return {"published_sha256": published_sha256,
                "motif_sha256": self.motif_sha256, "matches": True}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenMotif":
        record = _exact_keys(value, _TOP_LEVEL_FIELDS, "frozen_motif")
        if record["schema"] != FROZEN_MOTIF_SCHEMA:
            raise InvalidParameterError(
                "schema", record["schema"], "the %r schema tag" % FROZEN_MOTIF_SCHEMA)
        return cls(
            study_id=record["study_id"], frozen_at=record["frozen_at"],
            source=dict(record["source"]), source_partition=dict(record["source_partition"]),
            source_partition_sha256=str(record["source_partition_sha256"]),
            origin_domain=dict(record["origin_domain"]),
            origin_nodes=tuple(dict(value) for value in record["origin_nodes"]),
            definition=dict(record["definition"]),
            definition_sha256=str(record["definition_sha256"]),
            motif_sha256=str(record["motif_sha256"]),
        )


def freeze_motif(candidate: MotifCandidate, *, result: MiningResult,
                 origin_domain: DomainDeclaration, train: PartitionIdentity,
                 frozen_at: str, study_id: str) -> FrozenMotif:
    """Turn one selected training motif into the durable definition TG5.2 may transfer."""
    if not isinstance(candidate, MotifCandidate):
        raise InvalidParameterError("candidate", type(candidate).__name__, "a MotifCandidate")
    if not isinstance(result, MiningResult):
        raise InvalidParameterError("result", type(result).__name__, "a MiningResult")
    if not isinstance(origin_domain, DomainDeclaration):
        raise InvalidParameterError(
            "origin_domain", type(origin_domain).__name__, "a validated DomainDeclaration")
    if not isinstance(train, PartitionIdentity):
        raise InvalidParameterError("train", type(train).__name__, "a PartitionIdentity")

    # Reuse TG3.2's split and declared-family refusal rather than restating either rule.
    report_motif_generation(result, train=train)
    matching = [value for value in result.candidates if value.label == candidate.label]
    if len(matching) != 1 or _graph_definition(matching[0], result) != _graph_definition(
            candidate, result):
        raise InvalidParameterError(
            "candidate", candidate.label,
            "the exact exemplar graph selected by this MiningResult. Reusing a declared "
            "label with another signature is motif redefinition before serialization")
    if candidate.exemplar.graph.n_nodes != result.size:
        raise InvalidParameterError(
            "candidate.exemplar.graph", candidate.exemplar.graph.n_nodes,
            "%d nodes, the configuration size the mining family declared" % result.size)

    origin = origin_domain.describe()
    node_domains = {record.get("domain") for record in candidate.exemplar.graph.carried}
    if node_domains != {origin_domain.name}:
        raise InvalidParameterError(
            "origin_domain", origin_domain.name,
            "the domain carried by every exemplar node (%s)" % sorted(
                str(value) for value in node_domains))
    partition = train.to_mapping()
    return FrozenMotif(
        study_id=study_id, frozen_at=frozen_at,
        source={
            "label": candidate.label,
            "generate_sha256": result.specification.fingerprint(),
            "generate_family_size": result.specification.family_size,
            "support": candidate.support,
            "n_occurrences": len(candidate.occurrences),
            "n_examined": candidate.n_examined,
        },
        source_partition=partition, source_partition_sha256=train.digest(),
        origin_domain=origin, origin_nodes=tuple(candidate.exemplar.graph.carried),
        definition=_graph_definition(candidate, result),
    )


def save_frozen_motif(path: Union[str, os.PathLike[str]], motif: FrozenMotif) -> str:
    """Publish canonical bytes exactly once; an existing path is never replaced."""
    if not isinstance(motif, FrozenMotif):
        raise InvalidParameterError("motif", type(motif).__name__, "a FrozenMotif")
    motif.verify()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(motif.to_mapping()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("refusing to overwrite frozen motif at %s" % target) from None
    return motif.motif_sha256


def load_frozen_motif(path: Union[str, os.PathLike[str]], *,
                      published_sha256: Optional[str] = None) -> FrozenMotif:
    """Load strict canonical JSON, verify both hashes, and optionally verify publication."""
    target = Path(path)
    try:
        raw = target.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidParameterError(
            "frozen motif path", str(target), "readable UTF-8 JSON: %s" % exc) from exc
    motif = FrozenMotif.from_mapping(record)
    motif.verify()
    if published_sha256 is not None:
        motif.verify_published(published_sha256)
    if raw != _canonical(motif.to_mapping()) + b"\n":
        raise InvalidParameterError(
            "frozen motif bytes", str(target),
            "the canonical JSON serialization. Multiple byte representations of one digest "
            "make artifact identity ambiguous")
    return motif


__all__ = [
    "FROZEN_MOTIF_SCHEMA", "FrozenMotifIntegrityError", "FrozenMotif",
    "freeze_motif", "save_frozen_motif", "load_frozen_motif",
]
