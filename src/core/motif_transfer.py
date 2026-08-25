"""TG5.2: bind a published motif, then open and search a second domain once.

The target values are deliberately available only behind an ``opener`` callback.  A transfer
record containing the externally published motif digest is durably written before that callback
is invoked.  Once written, the target identity is spent even when loading or search later fails.

Search has no size, matcher, relation or tolerance arguments.  All four come from the verified
``FrozenMotif``.  This is the executable part of rule R20: callers may choose a new target, but
cannot use what they see there to revise the question they transfer.

The ledger establishes ordering inside this API; it cannot establish that a person or another
program did not inspect the target earlier.  That external access-control boundary must be
supplied by the archive or preregistration system named in the target provenance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.invariance import MATCHERS, build_signature
from src.core.motif import Scene
from src.core.motif_freeze import FrozenMotif
from src.core.preregistration import PartitionIdentity


TRANSFER_LEDGER_SCHEMA = "motif-transfer-ledger/v1"
TRANSFER_RECEIPT_SCHEMA = "motif-transfer-search/v1"
_LEDGER_FIELDS = ("schema", "records")
_RECORD_FIELDS = (
    "study_id", "motif_sha256", "published_sha256", "definition_sha256",
    "frozen_at", "bound_at", "opened_at", "source_domain", "target_domain",
    "target_domain_sha256", "target_partition", "target_partition_sha256",
    "state", "record_sha256",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Any, expected: Sequence[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidParameterError(path, type(value).__name__, "a JSON object")
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise InvalidParameterError(
            path, {"missing": missing, "extra": extra},
            "exactly the TG5.2 ledger fields; omitted or unrecognised meaning makes the "
            "opening chronology unverifiable")
    return value


def _moment(name: str, value: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidParameterError(name, value, "a timezone-bearing ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidParameterError(
            name, value, "a timezone-bearing ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidParameterError(
            name, value, "a timestamp with an explicit UTC offset")
    return parsed


def _record_body(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: record[name] for name in _RECORD_FIELDS if name != "record_sha256"}


def _partition_from_record(value: Any) -> PartitionIdentity:
    fields = ("name", "n_times", "n_channels", "channel_labels", "frames", "provenance")
    record = _exact(value, fields, "transfer target_partition")
    frames = record["frames"]
    if not isinstance(frames, (list, tuple)) or len(frames) != 2:
        raise InvalidParameterError(
            "transfer target_partition.frames", frames, "two half-open frame bounds")
    return PartitionIdentity(
        name=record["name"], n_times=int(record["n_times"]),
        n_channels=int(record["n_channels"]),
        channel_labels=tuple(record["channel_labels"]),
        frames=(int(frames[0]), int(frames[1])), provenance=dict(record["provenance"]))


class TargetAlreadyOpenedError(InvalidParameterError):
    """A target identity has already crossed the blind-opening boundary."""

    def __init__(self, record: Mapping[str, Any], motif_sha256: str) -> None:
        super().__init__(
            "target_partition", record.get("target_partition_sha256"),
            "a target that has not already been opened for transfer. This identity was "
            "committed at %s for motif %s and treated as opened at %s; trying another motif "
            "would use target observations to choose the transferred definition"
            % (record.get("bound_at"), record.get("motif_sha256"),
               record.get("opened_at")),
            first_motif_sha256=record.get("motif_sha256"),
            second_motif_sha256=motif_sha256,
            first_opened_at=record.get("opened_at"))


class TransferLedger:
    """Durable, content-checked records of targets committed before their opener ran."""

    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        if not path:
            raise InvalidParameterError(
                "transfer ledger path", path,
                "a durable path. An in-memory chronology disappears with the process and "
                "cannot show that binding preceded target access")
        self.path = Path(path)
        self._records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    @property
    def records(self) -> Dict[str, Dict[str, Any]]:
        return json.loads(json.dumps(self._records))

    def _load(self) -> None:
        try:
            raw = self.path.read_bytes()
            stored = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidParameterError(
                "transfer ledger", str(self.path), "readable canonical UTF-8 JSON: %s" % exc
            ) from exc
        top = _exact(stored, _LEDGER_FIELDS, "transfer ledger")
        if top["schema"] != TRANSFER_LEDGER_SCHEMA:
            raise InvalidParameterError(
                "transfer ledger schema", top["schema"],
                "the %r schema tag" % TRANSFER_LEDGER_SCHEMA)
        if not isinstance(top["records"], Mapping):
            raise InvalidParameterError("transfer ledger records", type(top["records"]).__name__,
                                        "a JSON object keyed by target partition digest")
        checked: Dict[str, Dict[str, Any]] = {}
        for key, value in top["records"].items():
            record = dict(_exact(value, _RECORD_FIELDS, "transfer ledger record"))
            recomputed = _digest(_record_body(record))
            if record["record_sha256"] != recomputed:
                raise InvalidParameterError(
                    "record_sha256", record["record_sha256"],
                    "the recomputed transfer-opening record digest %s" % recomputed)
            if key != record["target_partition_sha256"]:
                raise InvalidParameterError(
                    "transfer ledger key", key,
                    "the target partition digest stored inside its record")
            partition_digest = _partition_from_record(record["target_partition"]).digest()
            if partition_digest != record["target_partition_sha256"]:
                raise InvalidParameterError(
                    "target_partition_sha256", record["target_partition_sha256"],
                    "the digest recomputed from the target partition %s" % partition_digest)
            domain_digest = _digest(record["target_domain"])
            if domain_digest != record["target_domain_sha256"]:
                raise InvalidParameterError(
                    "target_domain_sha256", record["target_domain_sha256"],
                    "the digest recomputed from the target declaration %s" % domain_digest)
            if record["motif_sha256"] != record["published_sha256"]:
                raise InvalidParameterError(
                    "published_sha256", record["published_sha256"],
                    "the verified motif digest stored in the same opening record")
            if record["state"] != "opening_committed":
                raise InvalidParameterError(
                    "transfer state", record["state"], "'opening_committed'")
            frozen = _moment("frozen_at", record["frozen_at"])
            bound = _moment("bound_at", record["bound_at"])
            opened = _moment("opened_at", record["opened_at"])
            if not frozen < bound < opened:
                raise InvalidParameterError(
                    "transfer chronology",
                    [record["frozen_at"], record["bound_at"], record["opened_at"]],
                    "strictly ordered frozen_at < bound_at < opened_at")
            checked[str(key)] = record
        canonical = _canonical({"schema": TRANSFER_LEDGER_SCHEMA, "records": checked}) + b"\n"
        if raw != canonical:
            raise InvalidParameterError(
                "transfer ledger bytes", str(self.path),
                "the canonical JSON serialization; alternative bytes make ledger identity "
                "ambiguous")
        self._records = checked

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical({"schema": TRANSFER_LEDGER_SCHEMA,
                              "records": self._records}) + b"\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def commit_opening(self, motif: FrozenMotif, *, published_sha256: str,
                       target: PartitionIdentity, target_domain: DomainDeclaration,
                       bound_at: str, opened_at: str) -> Dict[str, Any]:
        """Persist the binding and opening intent; the caller may invoke the opener after."""
        motif.verify_published(published_sha256)
        if self.path.exists():
            self._load()  # See records another ledger instance may have committed.
        key = target.digest()
        if key in self._records:
            raise TargetAlreadyOpenedError(self._records[key], motif.motif_sha256)
        frozen = _moment("frozen_at", motif.frozen_at)
        bound = _moment("bound_at", bound_at)
        opened = _moment("opened_at", opened_at)
        if not frozen < bound < opened:
            raise InvalidParameterError(
                "transfer chronology", [motif.frozen_at, bound_at, opened_at],
                "strictly ordered frozen_at < bound_at < opened_at. The published digest "
                "must be committed after definition and before target access")
        if target_domain.name == motif.origin_domain["name"]:
            raise InvalidParameterError(
                "target_domain.name", target_domain.name,
                "a domain different from the motif origin %r. Another partition of one "
                "domain is replication, not cross-domain transfer"
                % motif.origin_domain["name"])
        if key == motif.source_partition_sha256:
            raise InvalidParameterError(
                "target_partition", key,
                "a partition different from the source partition that selected the motif")

        declaration = target_domain.describe()
        body = {
            "study_id": motif.study_id,
            "motif_sha256": motif.motif_sha256,
            "published_sha256": published_sha256,
            "definition_sha256": motif.definition_sha256,
            "frozen_at": motif.frozen_at,
            "bound_at": bound_at,
            "opened_at": opened_at,
            "source_domain": motif.origin_domain["name"],
            "target_domain": declaration,
            "target_domain_sha256": _digest(declaration),
            "target_partition": target.to_mapping(),
            "target_partition_sha256": key,
            # Committed before invoking user code: failure still spends the target.
            "state": "opening_committed",
        }
        record = dict(body, record_sha256=_digest(body))
        self._records[key] = record
        self._save()
        return json.loads(json.dumps(record))


def _target_scenes(value: Any, target_domain: DomainDeclaration) -> Tuple[Scene, ...]:
    try:
        scenes = tuple(value)
    except TypeError:
        raise InvalidParameterError(
            "opener result", type(value).__name__, "a sequence of target-domain Scene objects"
        ) from None
    if len(scenes) < 2 or any(not isinstance(scene, Scene) for scene in scenes):
        raise InvalidParameterError(
            "opener result", [type(scene).__name__ for scene in scenes],
            "at least two target-domain Scene objects")
    names = [scene.name for scene in scenes]
    if len(set(names)) != len(names):
        raise InvalidParameterError(
            "target scenes", names, "distinct names because transfer support counts scenes")
    domains = {feature.domain for scene in scenes for feature in scene.features}
    if domains != {target_domain.name}:
        raise InvalidParameterError(
            "target scene domains", sorted(domains),
            "features carried only by declared target domain %r" % target_domain.name)
    return scenes


def blind_transfer(motif: FrozenMotif, *, published_sha256: str,
                   target: PartitionIdentity, target_domain: DomainDeclaration,
                   opener: Callable[[], Sequence[Scene]], ledger: TransferLedger,
                   bound_at: str, opened_at: str) -> Dict[str, Any]:
    """Bind, spend, open and exhaustively search one target using only frozen parameters."""
    if not isinstance(motif, FrozenMotif):
        raise InvalidParameterError("motif", type(motif).__name__, "a FrozenMotif")
    if not isinstance(target, PartitionIdentity):
        raise InvalidParameterError("target", type(target).__name__, "a PartitionIdentity")
    if not isinstance(target_domain, DomainDeclaration):
        raise InvalidParameterError(
            "target_domain", type(target_domain).__name__, "a DomainDeclaration")
    if not isinstance(ledger, TransferLedger):
        raise InvalidParameterError("ledger", type(ledger).__name__, "a TransferLedger")
    if not callable(opener):
        raise InvalidParameterError(
            "opener", type(opener).__name__, "a zero-argument target-data callback")

    matcher = str(motif.definition["matcher"])
    entry = MATCHERS.entry(matcher)
    if not bool(entry.capabilities.get("crosses_domains", False)):
        raise InvalidParameterError(
            "definition.matcher", matcher,
            "a registered matcher declaring crosses_domains=True. A position- or unit-bound "
            "signature cannot define cross-domain sameness")
    size = int(motif.definition["size"])
    tolerance = float(motif.definition["tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0.0:  # Also checked by FrozenMotif.
        raise InvalidParameterError("definition.tolerance", tolerance,
                                    "a frozen non-negative finite tolerance")

    chronology = ledger.commit_opening(
        motif, published_sha256=published_sha256, target=target,
        target_domain=target_domain, bound_at=bound_at, opened_at=opened_at)
    scenes = _target_scenes(opener(), target_domain)
    if len(scenes) != target.n_times:
        raise InvalidParameterError(
            "target scenes", len(scenes),
            "%d scenes, one for every frame bound before opening. Returning a post-open "
            "subset would let target observations choose which configurations are searched"
            % target.n_times)

    examined = 0
    matches = []
    matched_scenes = set()
    for scene in scenes:
        for indices in itertools.combinations(range(len(scene.features)), size):
            examined += 1
            graph = build_signature(matcher, [scene.features[index] for index in indices])
            report = motif.graph.matches(
                graph, tolerance=tolerance, relations=motif.graph.relations)
            if report.matched:
                label = "%s|%s" % (scene.name, "-".join(str(i) for i in indices))
                matches.append({"label": label, "scene": scene.name,
                                "indices": list(indices), "comparison": report.describe()})
                matched_scenes.add(scene.name)

    search = {
        "size": size,
        "matcher": matcher,
        "tolerance": tolerance,
        "relations": list(motif.graph.relations),
        "source": "frozen_motif",
    }
    body = {
        "schema": TRANSFER_RECEIPT_SCHEMA,
        "study_id": motif.study_id,
        "motif_sha256": motif.motif_sha256,
        "published_sha256": published_sha256,
        "definition_sha256": motif.definition_sha256,
        "source_domain": motif.origin_domain["name"],
        "target_domain": target_domain.describe(),
        "target_partition_sha256": target.digest(),
        "chronology": chronology,
        "search": search,
        "n_scenes": len(scenes),
        "n_examined": examined,
        "n_matches": len(matches),
        "support": len(matched_scenes),
        "matched_scenes": sorted(matched_scenes),
        "matches": matches,
        "claim_boundary": (
            "A descriptive search for one externally published, unchanged motif in one "
            "target committed before opening. Match counts are not a corrected relationship "
            "transfer result; TG5.3 owns that test and its train/test family accounting."),
    }
    return dict(body, receipt_sha256=_digest(body))


__all__ = [
    "TRANSFER_LEDGER_SCHEMA", "TRANSFER_RECEIPT_SCHEMA", "TargetAlreadyOpenedError",
    "TransferLedger", "blind_transfer",
]
