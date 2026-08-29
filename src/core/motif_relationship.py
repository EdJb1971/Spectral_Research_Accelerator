"""TG5.3: preregister and test whether a frozen motif precedes organisation.

The target train and test partitions are both bound before either opener is called.  Analysis
parameters live only in :class:`RelationshipPlan`; execution has no outcome, lag, correction,
alpha, surrogate-count, seed, matcher or tolerance override surface.  Every declared outcome x
lag member is evaluated on both partitions, corrected at the full frozen family size in each,
and reported as transferred only when the same positive relationship rejects in both.

The circular-shift null preserves the motif-presence and outcome series separately while
destroying their alignment.  This supports an association/precedence claim, not mechanism,
causality, or predictive utility.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

import numpy as np

from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.family import SearchAxis, SearchSpecification, SearchTerm
from src.core.invariance import MATCHERS, build_signature
from src.core.motif import Scene
from src.core.motif_freeze import FrozenMotif
from src.core.preregistration import PartitionIdentity
from src.statistics.multiple_comparisons import ASSUMPTIONS, adjust


RELATIONSHIP_PLAN_SCHEMA = "motif-relationship-plan/v1"
RELATIONSHIP_LEDGER_SCHEMA = "motif-relationship-ledger/v1"
RELATIONSHIP_RECEIPT_SCHEMA = "motif-relationship-transfer/v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _moment(name: str, value: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidParameterError(name, value, "a timezone-bearing ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidParameterError(name, value,
                                    "a timezone-bearing ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidParameterError(name, value, "a timestamp with an explicit UTC offset")
    return parsed


def _partition(value: Mapping[str, Any]) -> PartitionIdentity:
    expected = {"name", "n_times", "n_channels", "channel_labels", "frames", "provenance"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise InvalidParameterError("partition", sorted(value) if isinstance(value, Mapping)
                                    else type(value).__name__,
                                    "exactly the PartitionIdentity fields")
    return PartitionIdentity(
        str(value["name"]), int(value["n_times"]), int(value["n_channels"]),
        tuple(value["channel_labels"]), tuple(int(v) for v in value["frames"]),
        dict(value["provenance"]))


def _domain(value: Mapping[str, Any]) -> DomainDeclaration:
    expected = {"name", "description", "licence", "axes", "violations", "lag_policy",
                "declared_floor_frames", "declared_floor_basis", "precedence_admissible",
                "provenance"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise InvalidParameterError("target_domain", sorted(value) if isinstance(value, Mapping)
                                    else type(value).__name__,
                                    "exactly the DomainDeclaration record fields")
    axes = []
    axis_fields = {"name", "role", "units", "periodic", "ordered", "ordinal"}
    for item in value["axes"]:
        if not isinstance(item, Mapping) or set(item) != axis_fields:
            raise InvalidParameterError("target_domain.axes", item,
                                        "exactly the AxisSpec record fields")
        axes.append(AxisSpec(**dict(item)))
    if not isinstance(value["violations"], Mapping):
        raise InvalidParameterError("target_domain.violations", value["violations"],
                                    "the declared violation record")
    declaration = DomainDeclaration(
        name=value["name"], description=value["description"], axes=tuple(axes),
        licence=value["licence"], violations=tuple(value["violations"]),
        lag_policy=value["lag_policy"],
        declared_floor_frames=value["declared_floor_frames"],
        declared_floor_basis=value["declared_floor_basis"], provenance=value["provenance"])
    if declaration.describe() != dict(value):
        raise InvalidParameterError("target_domain", value,
                                    "the record reconstructed from its DomainDeclaration")
    return declaration


def relationship_specification(outcomes: Sequence[str], lags: Sequence[int], *,
                               n_surrogates: int, alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli",
                               study_id: str = "") -> SearchSpecification:
    """Declare the complete outcome x positive-lag family before target access."""
    named = tuple(str(value).strip() for value in outcomes)
    if any(not value for value in named):
        raise InvalidParameterError("outcomes", list(outcomes), "non-empty outcome labels")
    delayed = tuple(int(value) for value in lags)
    if any(isinstance(value, bool) or int(value) != value or int(value) < 1 for value in lags):
        raise InvalidParameterError("lags", list(lags),
                                    "positive integer frame leads fixed before target access")
    return SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("outcome", named),
                                      SearchAxis("lag_frames", delayed))),),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}|lag={1}", study_id=study_id,
        notes={"stage": "target_train_and_test", "alternative": "greater",
               "statistic": "mean(subsequent outcome | motif) - mean(subsequent outcome | no motif)",
               "null": "circular shift of outcome relative to frozen motif presence"})


@dataclass(frozen=True)
class RelationshipPlan:
    """Content-addressed TG5.3 declaration, complete before either target split opens."""

    study_id: str
    motif_sha256: str
    target_domain: Mapping[str, Any]
    train: PartitionIdentity
    test: PartitionIdentity
    specification: Mapping[str, Any]
    labels: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    lags: Tuple[int, ...]
    n_surrogates: int
    alpha: float
    correction: str
    root_seed: int
    planned_at: str
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_domain", json.loads(json.dumps(self.target_domain)))
        object.__setattr__(self, "specification", json.loads(json.dumps(self.specification)))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "lags", tuple(int(v) for v in self.lags))
        if not isinstance(self.train, PartitionIdentity) or not isinstance(self.test, PartitionIdentity):
            raise InvalidParameterError("train/test", [type(self.train).__name__,
                                                        type(self.test).__name__],
                                        "two PartitionIdentity records")
        if isinstance(self.root_seed, bool) or int(self.root_seed) != self.root_seed or self.root_seed < 0:
            raise InvalidParameterError("root_seed", self.root_seed,
                                        "a non-negative integer fixed before target access")
        domain = _domain(self.target_domain)
        domain.assert_precedence_admissible()
        floor = domain.minimum_admissible_lag()
        if floor is not None and any(lag < floor for lag in self.lags):
            raise InvalidParameterError("lags", list(self.lags),
                                        "leads at or above the target domain floor of %d" % floor)
        _validate_split_contract(self.train, self.test, self.lags)
        rebuilt = relationship_specification(
            self.outcomes, self.lags, n_surrogates=self.n_surrogates, alpha=self.alpha,
            correction=self.correction, study_id=self.study_id)
        rebuilt.declare()
        if self.specification != rebuilt.to_mapping() or self.labels != rebuilt.labels():
            raise InvalidParameterError(
                "relationship specification", self.specification,
                "the family reconstructed from the plan's frozen analysis fields")
        _moment("planned_at", self.planned_at)
        if not self.plan_sha256:
            object.__setattr__(self, "plan_sha256", _digest(self.body()))
        elif self.plan_sha256 != _digest(self.body()):
            raise InvalidParameterError("plan_sha256", self.plan_sha256,
                                        "the digest recomputed from the relationship plan")

    def body(self) -> Dict[str, Any]:
        return {
            "schema": RELATIONSHIP_PLAN_SCHEMA, "study_id": self.study_id,
            "motif_sha256": self.motif_sha256, "target_domain": dict(self.target_domain),
            "target_domain_sha256": _digest(self.target_domain),
            "train": self.train.to_mapping(), "train_sha256": self.train.digest(),
            "test": self.test.to_mapping(), "test_sha256": self.test.digest(),
            "specification": dict(self.specification), "labels": list(self.labels),
            "outcomes": list(self.outcomes), "lags": list(self.lags),
            "n_surrogates": int(self.n_surrogates), "alpha": float(self.alpha),
            "correction": self.correction, "root_seed": int(self.root_seed),
            "planned_at": self.planned_at,
            "claim": "positive association between motif presence and subsequent organisation",
        }

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), plan_sha256=self.plan_sha256)

    def verify_published(self, published_sha256: str) -> None:
        recomputed = _digest(self.body())
        if recomputed != self.plan_sha256 or str(published_sha256).strip().lower() != recomputed:
            raise InvalidParameterError("published_plan_sha256", published_sha256,
                                        "the unchanged relationship plan digest %s" % recomputed)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelationshipPlan":
        expected = set(cls_fields()) | {"schema", "target_domain_sha256", "train_sha256",
                                        "test_sha256", "claim"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise InvalidParameterError("relationship plan", sorted(value) if isinstance(value, Mapping)
                                        else type(value).__name__, "exactly the TG5.3 plan fields")
        if value["schema"] != RELATIONSHIP_PLAN_SCHEMA:
            raise InvalidParameterError("schema", value["schema"], RELATIONSHIP_PLAN_SCHEMA)
        plan = cls(
            study_id=value["study_id"], motif_sha256=value["motif_sha256"],
            target_domain=value["target_domain"], train=_partition(value["train"]),
            test=_partition(value["test"]), specification=value["specification"],
            labels=tuple(value["labels"]), outcomes=tuple(value["outcomes"]),
            lags=tuple(value["lags"]), n_surrogates=int(value["n_surrogates"]),
            alpha=float(value["alpha"]), correction=value["correction"],
            root_seed=int(value["root_seed"]), planned_at=value["planned_at"],
            plan_sha256=value["plan_sha256"])
        if value["target_domain_sha256"] != _digest(plan.target_domain):
            raise InvalidParameterError("target_domain_sha256", value["target_domain_sha256"],
                                        "the recomputed target-domain digest")
        if value["train_sha256"] != plan.train.digest() or value["test_sha256"] != plan.test.digest():
            raise InvalidParameterError("partition digest", "mismatch",
                                        "digests recomputed from both bound partitions")
        if value["claim"] != "positive association between motif presence and subsequent organisation":
            raise InvalidParameterError("claim", value["claim"],
                                        "the TG5.3 relationship-plan claim")
        return plan


def cls_fields() -> Tuple[str, ...]:
    return ("study_id", "motif_sha256", "target_domain", "train", "test", "specification",
            "labels", "outcomes", "lags", "n_surrogates", "alpha", "correction",
            "root_seed", "planned_at", "plan_sha256")


def _validate_split_contract(train: PartitionIdentity, test: PartitionIdentity,
                             lags: Sequence[int]) -> None:
    if train.digest() == test.digest() or train.frames[1] > test.frames[0]:
        raise InvalidParameterError(
            "train/test", [train.frames, test.frames],
            "distinct, non-overlapping target partitions with train ending before test begins")
    if str(train.provenance.get("split", "")).lower() != "train" or \
            str(test.provenance.get("split", "")).lower() not in ("test", "held_out", "held-out"):
        raise InvalidParameterError("train/test provenance", [train.provenance, test.provenance],
                                    "explicit train and test split identities")
    if train.channel_labels != test.channel_labels or train.n_channels != test.n_channels:
        raise InvalidParameterError("train/test channels", [train.channel_labels, test.channel_labels],
                                    "the same declared outcome geometry on both splits")
    if any(int(lag) >= min(train.n_times, test.n_times) for lag in lags):
        raise InvalidParameterError("lags", list(lags),
                                    "leads shorter than both bound target partitions")


def plan_relationship_transfer(motif: FrozenMotif, *, target_domain: DomainDeclaration,
                               train: PartitionIdentity, test: PartitionIdentity,
                               outcomes: Sequence[str], lags: Sequence[int], n_surrogates: int,
                               root_seed: int, planned_at: str, alpha: float = 0.05,
                               correction: str = "benjamini_yekutieli",
                               study_id: str = "") -> RelationshipPlan:
    """Freeze the complete relationship family and split accounting."""
    motif.verify()
    if isinstance(root_seed, bool) or int(root_seed) != root_seed or int(root_seed) < 0:
        raise InvalidParameterError("root_seed", root_seed,
                                    "a non-negative integer fixed before target access")
    if target_domain.name == motif.origin_domain["name"]:
        raise InvalidParameterError("target_domain", target_domain.name,
                                    "a domain different from the frozen motif origin")
    target_domain.assert_precedence_admissible()
    floor = target_domain.minimum_admissible_lag()
    if floor is not None and any(int(lag) < floor for lag in lags):
        raise InvalidParameterError(
            "lags", list(lags),
            "leads at or above target domain %r's declared floor of %d frames"
            % (target_domain.name, floor))
    _validate_split_contract(train, test, lags)
    planned = _moment("planned_at", planned_at)
    if not _moment("frozen_at", motif.frozen_at) < planned:
        raise InvalidParameterError("planned_at", planned_at, "a time after motif freezing")
    specification = relationship_specification(
        outcomes, lags, n_surrogates=n_surrogates, alpha=alpha,
        correction=correction, study_id=study_id or motif.study_id)
    account = specification.declare()  # Refuses an arithmetically incapable null study.
    del account
    return RelationshipPlan(
        study_id=study_id or motif.study_id, motif_sha256=motif.motif_sha256,
        target_domain=target_domain.describe(), train=train, test=test,
        specification=specification.to_mapping(), labels=specification.labels(),
        outcomes=tuple(str(v).strip() for v in outcomes), lags=tuple(int(v) for v in lags),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        root_seed=int(root_seed), planned_at=planned_at)


def save_relationship_plan(path: Union[str, os.PathLike[str]], plan: RelationshipPlan) -> str:
    target = Path(path)
    payload = _canonical(plan.to_mapping()) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return plan.plan_sha256


def load_relationship_plan(path: Union[str, os.PathLike[str]], *,
                           published_sha256: str = "") -> RelationshipPlan:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    plan = RelationshipPlan.from_mapping(value)
    if raw != _canonical(plan.to_mapping()) + b"\n":
        raise InvalidParameterError("relationship plan bytes", str(path), "canonical JSON")
    if published_sha256:
        plan.verify_published(published_sha256)
    return plan


@dataclass(frozen=True)
class RelationshipFrame:
    """One target scene and the organisation measures observed at that time."""

    scene: Scene
    outcomes: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", dict(self.outcomes))


class RelationshipLedger:
    """Durable one-use ledger keyed by the bound train/test pair."""

    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        if not path:
            raise InvalidParameterError("relationship ledger", path, "a durable path")
        self.path = Path(path)
        self._records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    @property
    def records(self) -> Dict[str, Dict[str, Any]]:
        return json.loads(json.dumps(self._records))

    def _load(self) -> None:
        raw = self.path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if set(value) != {"schema", "records"} or value["schema"] != RELATIONSHIP_LEDGER_SCHEMA:
            raise InvalidParameterError("relationship ledger", value, RELATIONSHIP_LEDGER_SCHEMA)
        checked = {}
        used_partitions = set()
        for key, record in value["records"].items():
            body = {name: record[name] for name in record if name != "record_sha256"}
            if set(record) != {"plan_sha256", "motif_sha256", "published_motif_sha256",
                              "published_plan_sha256", "train_sha256", "test_sha256",
                              "planned_at", "bound_at", "train_opened_at", "test_opened_at",
                              "state", "record_sha256"} or _digest(body) != record["record_sha256"]:
                raise InvalidParameterError("relationship ledger record", key,
                                            "exact fields and its recomputed digest")
            if key != _digest([record["train_sha256"], record["test_sha256"]]):
                raise InvalidParameterError("relationship ledger key", key,
                                            "the bound train/test pair digest")
            if record["state"] != "both_openings_committed":
                raise InvalidParameterError("relationship ledger state", record["state"],
                                            "both_openings_committed")
            if record["published_motif_sha256"] != record["motif_sha256"] or \
                    record["published_plan_sha256"] != record["plan_sha256"]:
                raise InvalidParameterError(
                    "relationship published digests", key,
                    "the motif and plan identities stored in the same committed record")
            pair = {record["train_sha256"], record["test_sha256"]}
            if used_partitions.intersection(pair):
                raise InvalidParameterError(
                    "relationship ledger partitions", sorted(pair),
                    "each target partition appearing in only one committed train/test pair")
            used_partitions.update(pair)
            if not (_moment("planned_at", record["planned_at"]) < _moment("bound_at", record["bound_at"])
                    < _moment("train_opened_at", record["train_opened_at"])
                    < _moment("test_opened_at", record["test_opened_at"])):
                raise InvalidParameterError("relationship chronology", record,
                                            "planned_at < bound_at < train_opened_at < test_opened_at")
            checked[key] = record
        if raw != _canonical({"schema": RELATIONSHIP_LEDGER_SCHEMA, "records": checked}) + b"\n":
            raise InvalidParameterError("relationship ledger bytes", str(self.path), "canonical JSON")
        self._records = checked

    def commit(self, motif: FrozenMotif, plan: RelationshipPlan, *,
               published_motif_sha256: str, published_plan_sha256: str,
               bound_at: str, train_opened_at: str, test_opened_at: str) -> Dict[str, Any]:
        motif.verify_published(published_motif_sha256)
        plan.verify_published(published_plan_sha256)
        if motif.motif_sha256 != plan.motif_sha256:
            raise InvalidParameterError("motif_sha256", motif.motif_sha256,
                                        "the motif bound by the relationship plan")
        if self.path.exists(): self._load()
        key = _digest([plan.train.digest(), plan.test.digest()])
        if key in self._records:
            raise InvalidParameterError("train/test pair", key,
                                        "a relationship target pair that has not already been opened")
        used = {record[field] for record in self._records.values()
                for field in ("train_sha256", "test_sha256")}
        reused = used.intersection({plan.train.digest(), plan.test.digest()})
        if reused:
            raise InvalidParameterError(
                "target partition", sorted(reused),
                "target partitions never opened in any earlier relationship pair; changing "
                "the counterpart does not make reused target observations held out again")
        times = (_moment("planned_at", plan.planned_at), _moment("bound_at", bound_at),
                 _moment("train_opened_at", train_opened_at),
                 _moment("test_opened_at", test_opened_at))
        if not times[0] < times[1] < times[2] < times[3]:
            raise InvalidParameterError("relationship chronology",
                                        [plan.planned_at, bound_at, train_opened_at, test_opened_at],
                                        "planned_at < bound_at < train_opened_at < test_opened_at")
        body = {"plan_sha256": plan.plan_sha256, "motif_sha256": motif.motif_sha256,
                "published_motif_sha256": published_motif_sha256,
                "published_plan_sha256": published_plan_sha256,
                "train_sha256": plan.train.digest(), "test_sha256": plan.test.digest(),
                "planned_at": plan.planned_at, "bound_at": bound_at,
                "train_opened_at": train_opened_at, "test_opened_at": test_opened_at,
                "state": "both_openings_committed"}
        record = dict(body, record_sha256=_digest(body))
        self._records[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical({"schema": RELATIONSHIP_LEDGER_SCHEMA, "records": self._records}) + b"\n"
        descriptor, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp",
                                                  dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        return json.loads(json.dumps(record))


def _presence(motif: FrozenMotif, frames: Tuple[RelationshipFrame, ...], domain: str) -> np.ndarray:
    matcher = str(motif.definition["matcher"])
    if not MATCHERS.entry(matcher).capabilities.get("crosses_domains", False):
        raise InvalidParameterError("matcher", matcher, "crosses_domains=True")
    size, tolerance = int(motif.definition["size"]), float(motif.definition["tolerance"])
    found = []
    for frame in frames:
        if not isinstance(frame, RelationshipFrame):
            raise InvalidParameterError("opener result", type(frame).__name__, "RelationshipFrame values")
        if any(feature.domain != domain for feature in frame.scene.features):
            raise InvalidParameterError("target scene domain", frame.scene.name, domain)
        match = False
        for indices in itertools.combinations(range(len(frame.scene.features)), size):
            graph = build_signature(matcher, [frame.scene.features[i] for i in indices])
            if motif.graph.matches(graph, tolerance=tolerance,
                                   relations=motif.graph.relations).matched:
                match = True
        found.append(match)
    return np.asarray(found, dtype=bool)


def _split_result(motif: FrozenMotif, frames_value: Sequence[RelationshipFrame], *,
                  partition: PartitionIdentity, plan: RelationshipPlan,
                  split: str) -> Dict[str, Any]:
    frames = tuple(frames_value)
    if len(frames) != partition.n_times or len({f.scene.name for f in frames}) != len(frames):
        raise InvalidParameterError(split, len(frames),
                                    "%d distinct frames fixed before opening" % partition.n_times)
    expected = set(plan.outcomes)
    for frame in frames:
        if set(frame.outcomes) != expected or any(not math.isfinite(float(v)) for v in frame.outcomes.values()):
            raise InvalidParameterError(split + " outcomes", frame.outcomes,
                                        "every frozen finite outcome, with no additions or omissions")
    presence = _presence(motif, frames, str(plan.target_domain["name"]))
    results = []
    for outcome in plan.outcomes:
        values = np.asarray([float(frame.outcomes[outcome]) for frame in frames], dtype=float)
        for lag in plan.lags:
            x, y = presence[:-lag], values[lag:]
            testable = bool(np.any(x) and np.any(~x))
            effect = float(np.mean(y[x]) - np.mean(y[~x])) if testable else None
            seed_text = "%d|%s|%s|%d" % (plan.root_seed, split, outcome, lag)
            seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            exceed = 0
            if testable:
                for _ in range(plan.n_surrogates):
                    shifted = np.roll(y, int(rng.integers(1, len(y))))
                    statistic = float(np.mean(shifted[x]) - np.mean(shifted[~x]))
                    exceed += statistic >= effect
                p_value = (1.0 + exceed) / (1.0 + plan.n_surrogates)
            else:
                p_value = 1.0
            results.append({"label": "%s|lag=%d" % (outcome, lag), "outcome": outcome,
                            "lag_frames": lag, "n_pairs": len(x),
                            "n_motif": int(np.sum(x)), "effect": effect,
                            "alternative": "greater", "p_value": p_value,
                            "testable": testable})
    labels = tuple(item["label"] for item in results)
    if labels != plan.labels:
        raise InvalidParameterError("relationship enumeration", labels,
                                    "the frozen family labels in declaration order")
    corrected = adjust([item["p_value"] for item in results], method=plan.correction,
                       alpha=plan.alpha, n_tests=len(plan.labels), labels=labels)
    for item, q, rejected in zip(results, corrected["adjusted"], corrected["rejected"]):
        item["adjusted"] = q
        item["rejected"] = bool(rejected and item["testable"] and item["effect"] > 0)
    return {"split": split, "partition": partition.to_mapping(),
            "motif_presence": [bool(v) for v in presence], "n_motif_frames": int(np.sum(presence)),
            "family_size": len(plan.labels), "correction": plan.correction,
            "dependence_assumption": ASSUMPTIONS[plan.correction], "alpha": plan.alpha,
            "results": results, "rejected_labels": [r["label"] for r in results if r["rejected"]]}


def test_relationship_transfer(motif: FrozenMotif, plan: RelationshipPlan, *,
                               published_motif_sha256: str, published_plan_sha256: str,
                               train_opener: Callable[[], Sequence[RelationshipFrame]],
                               test_opener: Callable[[], Sequence[RelationshipFrame]],
                               ledger: RelationshipLedger, bound_at: str,
                               train_opened_at: str, test_opened_at: str) -> Dict[str, Any]:
    """Commit both openings, evaluate the frozen family twice, and require replication."""
    chronology = ledger.commit(
        motif, plan, published_motif_sha256=published_motif_sha256,
        published_plan_sha256=published_plan_sha256, bound_at=bound_at,
        train_opened_at=train_opened_at, test_opened_at=test_opened_at)
    train = _split_result(motif, train_opener(), partition=plan.train, plan=plan, split="train")
    test = _split_result(motif, test_opener(), partition=plan.test, plan=plan, split="test")
    replicated = sorted(set(train["rejected_labels"]) & set(test["rejected_labels"]))
    body = {"schema": RELATIONSHIP_RECEIPT_SCHEMA, "study_id": plan.study_id,
            "motif_sha256": motif.motif_sha256, "plan_sha256": plan.plan_sha256,
            "chronology": chronology, "family": dict(plan.specification),
            "correction_unit": len(plan.labels), "train": train, "test": test,
            "replicated_labels": replicated, "n_replicated": len(replicated),
            "verdict": "PASS" if replicated else "FAIL",
            "claim_boundary": ("PASS means the same preregistered positive motif-to-subsequent-"
                               "organisation association survived full-family correction on both "
                               "target splits. FAIL is an adequately powered null transfer. Neither "
                               "verdict establishes mechanism, causality, or predictive utility.")}
    return dict(body, receipt_sha256=_digest(body))


__all__ = ["RELATIONSHIP_PLAN_SCHEMA", "RELATIONSHIP_LEDGER_SCHEMA",
           "RELATIONSHIP_RECEIPT_SCHEMA", "RelationshipPlan", "RelationshipFrame",
           "RelationshipLedger", "relationship_specification", "plan_relationship_transfer",
           "save_relationship_plan", "load_relationship_plan", "test_relationship_transfer"]
