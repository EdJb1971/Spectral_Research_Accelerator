"""The generate/confirm split: mine on train, freeze, test held-out once (TG3.2, rule R18).

**What this module is for.** TG3.1 prices a declared family and refuses it when the declared
surrogate ensemble cannot reject one member of it. For any search worth running that refusal
bites hard - the T4C.6 family of 36 needs 3,005 surrogates, and a constellation sweep of 4,480
needs 805,029. R18 permits exactly two remedies, and this module is the second one.

**It is not a cheaper family.** Nothing here reduces the number of tests performed. The
generate stage still enumerates and still tests, on the *training* partition, and nothing it
produces is a claim: its p-values are uncorrected, its selection used the data, and its output
is a list of candidates. What the split buys is that the *confirmatory* family - a subset of
the generated one, small enough to be affordable - is written down, hashed, and published
**before the held-out partition is opened**. The correction then applies to that family, which
is legitimately small because it was fixed without reference to the held-out data.

Everything here exists to make the two ways of cheating detectable:

*   **Editing the declaration afterwards.** A `Seal` carries a digest per sealed field and a
    digest over that table. `Seal.verify` recomputes both, so an edited field is named, and an
    edited digest table is caught by the outer digest.
*   **Opening the held-out partition more than once.** `HeldOutLedger` records consumption
    keyed by the *partition*, not by the seal. A second confirmatory family against the same
    held-out data is refused even under a fresh, perfectly valid seal - because two seals
    against one partition are two tests of that partition, and only the first was corrected
    for.

**The boundary this module cannot cross, stated plainly.** A content hash detects an
unrecorded edit; it does not prevent one. Anyone who can rewrite the seal file can recompute
every digest in it, and nothing in this process is signed. The binding is only as strong as
where `seal_sha256` was published - a commit, a registry entry, a preregistration record that
the author cannot rewrite. `verify_published` is the check that actually carries weight, and
it needs a digest that came from somewhere else. This module makes that digest, names it, and
refuses to pretend the local copy is evidence about itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.channel_series import _recordable
from src.core.errors import InvalidParameterError
from src.core.family import MAX_ENUMERATED, SearchSpecification
from src.statistics.multiple_comparisons import ASSUMPTIONS, adjust

#: Schema tag written into every seal, so a reader of a stored seal knows what it is.
SEAL_SCHEMA = "generate-confirm-seal/v1"

#: The fields a seal binds. Order is fixed because the digest table is keyed by these names
#: and a reader comparing two seals compares them field by field.
SEALED_FIELDS: Tuple[str, ...] = (
    "schema", "study_id", "sealed_at", "generate_sha256", "generate_family_size",
    "confirm", "confirm_sha256", "confirm_labels", "confirm_family_size",
    "confirm_account", "held_out",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=str).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


# ------------------------------------------------------------------- partition identity


@dataclass(frozen=True)
class PartitionIdentity:
    """What a seal binds itself to, derived **without reading a single value**.

    A seal has to name the held-out partition it will be tested against, and it has to do so
    before that partition is opened. So this record is geometry and lineage only: which
    channels, how many frames, which frames of the parent series, and the provenance the
    split itself wrote. `from_series` touches no measure array, and a test enforces that by
    handing it a series whose values raise on access.
    """

    name: str
    n_times: int
    n_channels: int
    channel_labels: Tuple[str, ...]
    frames: Tuple[int, int]
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_labels", tuple(str(c) for c in self.channel_labels))
        object.__setattr__(self, "frames", (int(self.frames[0]), int(self.frames[1])))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidParameterError(
                "name", self.name, "a name for the partition. An unnamed partition cannot be "
                                   "distinguished from another in a ledger of what was opened")
        if self.n_times < 1 or self.n_channels < 1:
            raise InvalidParameterError(
                "partition shape", (self.n_times, self.n_channels),
                "at least one frame and one channel")
        if self.frames[1] <= self.frames[0]:
            raise InvalidParameterError("frames", self.frames,
                                        "a half-open [start, stop) with stop > start")
        if len(self.channel_labels) != self.n_channels:
            raise InvalidParameterError(
                "channel_labels", list(self.channel_labels),
                "one label per channel, and this record declares %d. A partition whose count "
                "and labels disagree describes two different geometries, and the seal would "
                "bind both of them - so a confirmation on the partition the labels name "
                "would look identical to one on the partition the count names"
                % self.n_channels)

    @classmethod
    def from_series(cls, series: Any, *, name: str) -> "PartitionIdentity":
        """Identify a partition from its geometry and lineage. Reads no measure values."""
        provenance = {str(key): _recordable(value)
                      for key, value in dict(series.provenance).items()}
        frames = provenance.get("split_frames")
        if not (isinstance(frames, (list, tuple)) and len(frames) == 2):
            frames = (0, int(series.n_times))
        return cls(name=name, n_times=int(series.n_times), n_channels=int(series.n_channels),
                   channel_labels=tuple(str(c) for c in series.channels),
                   frames=(int(frames[0]), int(frames[1])), provenance=provenance)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_times": int(self.n_times),
            "n_channels": int(self.n_channels),
            "channel_labels": list(self.channel_labels),
            "frames": list(self.frames),
            "provenance": dict(self.provenance),
        }

    def digest(self) -> str:
        """The partition's identity as a hash - the ledger's key and the seal's binding."""
        return _digest(self.to_mapping())


# ---------------------------------------------------------------------------- refusals


class SealBrokenError(InvalidParameterError):
    """A seal no longer hashes to what it recorded, so the declaration changed after it."""

    def __init__(self, changed: Sequence[str], recorded: str, recomputed: str,
                 *, detail: str) -> None:
        listing = ", ".join(changed) if changed else "(the digest table itself)"
        super().__init__(
            "seal_sha256", recomputed,
            "the digest this seal recorded (%s). %s Changed: %s. A seal is a claim that the "
            "confirmatory family was fixed before the held-out partition was opened; a seal "
            "that does not hash to its own contents is not evidence of that, and the "
            "confirmation it was written for cannot be run. If the edit was legitimate, it "
            "is a new declaration on a new held-out partition, not a repair of this one."
            % (recorded, detail, listing),
            changed_fields=list(changed), recorded_sha256=recorded,
            recomputed_sha256=recomputed)


class HeldOutAlreadyOpenedError(InvalidParameterError):
    """This held-out partition has been tested already; a second test is uncorrected."""

    def __init__(self, record: Mapping[str, Any], seal_sha256: str) -> None:
        super().__init__(
            "held_out", record.get("partition_name"),
            "a held-out partition that has not been tested yet. This one was opened at %s "
            "under seal %s for a confirmatory family of %d. A second confirmation against "
            "the same data is a second test of it, and the first correction did not include "
            "the second family - so the pair is uncorrected however each looks alone. "
            "Remedies: (1) a genuinely held-out partition this study has not touched, or "
            "(2) one seal covering both families, whose combined size is then the correction "
            "unit and must pass the TG3.1 gate."
            % (record.get("opened_at"), record.get("seal_sha256"), record.get("family_size")),
            first_seal_sha256=record.get("seal_sha256"), second_seal_sha256=seal_sha256,
            first_opened_at=record.get("opened_at"),
            partition_digest=record.get("partition_digest"))


class PartitionMismatchError(InvalidParameterError):
    """The partition presented for confirmation is not the one the seal was written for."""

    def __init__(self, sealed: PartitionIdentity, presented: PartitionIdentity) -> None:
        super().__init__(
            "held_out", presented.to_mapping(),
            "the partition this seal bound itself to (%s, %d frames %s, digest %s). The "
            "partition presented is %s, %d frames %s, digest %s. Confirming on a partition "
            "the seal did not name allows the partition itself to be chosen after the "
            "declaration, which is the same freedom the split exists to remove."
            % (sealed.name, sealed.n_times, list(sealed.frames), sealed.digest()[:16],
               presented.name, presented.n_times, list(presented.frames),
               presented.digest()[:16]),
            sealed_digest=sealed.digest(), presented_digest=presented.digest())


# -------------------------------------------------------------------------------- seal


@dataclass(frozen=True)
class Seal:
    """A frozen confirmatory declaration, bound field by field to its own digests."""

    study_id: str
    sealed_at: str
    generate_sha256: str
    generate_family_size: int
    confirm: Mapping[str, Any]
    confirm_sha256: str
    confirm_labels: Tuple[str, ...]
    confirm_family_size: int
    confirm_account: Mapping[str, Any]
    held_out: PartitionIdentity
    field_sha256: Mapping[str, str] = dc_field(default_factory=dict)
    seal_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confirm_labels", tuple(str(x) for x in self.confirm_labels))
        object.__setattr__(self, "confirm", dict(self.confirm))
        object.__setattr__(self, "confirm_account", dict(self.confirm_account))
        if not self.field_sha256:
            object.__setattr__(self, "field_sha256", self.compute_field_digests())
        if not self.seal_sha256:
            object.__setattr__(self, "seal_sha256",
                               _digest({"schema": SEAL_SCHEMA,
                                        "field_sha256": dict(self.field_sha256)}))

    # ------------------------------------------------------------------------ digests

    def body(self) -> Dict[str, Any]:
        """The sealed fields, exactly as they are hashed."""
        return {
            "schema": SEAL_SCHEMA,
            "study_id": self.study_id,
            "sealed_at": self.sealed_at,
            "generate_sha256": self.generate_sha256,
            "generate_family_size": int(self.generate_family_size),
            "confirm": dict(self.confirm),
            "confirm_sha256": self.confirm_sha256,
            "confirm_labels": list(self.confirm_labels),
            "confirm_family_size": int(self.confirm_family_size),
            "confirm_account": dict(self.confirm_account),
            "held_out": self.held_out.to_mapping(),
        }

    def compute_field_digests(self) -> Dict[str, str]:
        body = self.body()
        return {name: _digest(body[name]) for name in SEALED_FIELDS}

    def verify(self) -> Dict[str, Any]:
        """Recompute both digest layers. Raises `SealBrokenError` naming what changed.

        Two distinct edits are detectable here, and they are detected differently. An edited
        *field* no longer matches its recorded digest, so the field is named. An editor who
        also updates the digest table breaks the outer digest instead, which is caught but
        cannot say which field moved - `verify_published` is where that case is settled.
        """
        recomputed = self.compute_field_digests()
        recorded = dict(self.field_sha256)
        changed = sorted(name for name in SEALED_FIELDS
                         if recorded.get(name) != recomputed[name])
        if changed:
            raise SealBrokenError(
                changed, recorded.get(changed[0], ""), recomputed[changed[0]],
                detail="These fields no longer hash to the digests recorded beside them.")
        outer = _digest({"schema": SEAL_SCHEMA, "field_sha256": recorded})
        if outer != self.seal_sha256:
            raise SealBrokenError(
                [], self.seal_sha256, outer,
                detail=("The per-field digests are self-consistent but the seal digest over "
                        "them is not, so the digest table was rewritten."))
        return {"seal_sha256": outer, "field_sha256": recomputed,
                "note": ("Self-consistency only. This says the local copy was not edited "
                         "carelessly; it says nothing about whether it was edited. Compare "
                         "against the digest published before the held-out partition was "
                         "opened - `verify_published` - for a check with weight.")}

    def verify_published(self, published_sha256: str) -> Dict[str, Any]:
        """Check this seal against a digest recorded somewhere the author cannot rewrite."""
        if not isinstance(published_sha256, str) or not published_sha256.strip():
            raise InvalidParameterError(
                "published_sha256", published_sha256,
                "the seal digest as it was published before the held-out partition was "
                "opened. Without an external copy there is nothing for the local seal to "
                "disagree with, and a self-consistent seal is not evidence about itself")
        self.verify()
        published = published_sha256.strip().lower()
        if published != self.seal_sha256:
            raise SealBrokenError(
                sorted(SEALED_FIELDS), published, self.seal_sha256,
                detail=("The seal is internally consistent but does not match the published "
                        "digest, so it was rewritten wholesale after publication."))
        return {"seal_sha256": self.seal_sha256, "published_sha256": published,
                "matches": True}

    # -------------------------------------------------------------------- persistence

    def to_mapping(self) -> Dict[str, Any]:
        record = self.body()
        record["field_sha256"] = dict(self.field_sha256)
        record["seal_sha256"] = self.seal_sha256
        return record

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Seal":
        if mapping.get("schema") != SEAL_SCHEMA:
            raise InvalidParameterError("schema", mapping.get("schema"),
                                        "the %r schema tag" % SEAL_SCHEMA)
        held = dict(mapping["held_out"])
        return cls(
            study_id=mapping["study_id"], sealed_at=mapping["sealed_at"],
            generate_sha256=mapping["generate_sha256"],
            generate_family_size=int(mapping["generate_family_size"]),
            confirm=dict(mapping["confirm"]), confirm_sha256=mapping["confirm_sha256"],
            confirm_labels=tuple(mapping["confirm_labels"]),
            confirm_family_size=int(mapping["confirm_family_size"]),
            confirm_account=dict(mapping["confirm_account"]),
            held_out=PartitionIdentity(
                name=held["name"], n_times=int(held["n_times"]),
                n_channels=int(held["n_channels"]),
                channel_labels=tuple(held["channel_labels"]),
                frames=(int(held["frames"][0]), int(held["frames"][1])),
                provenance=dict(held.get("provenance", {}))),
            field_sha256=dict(mapping.get("field_sha256", {})),
            seal_sha256=str(mapping.get("seal_sha256", "")))


# ------------------------------------------------------------------------ generate stage


def report_generation(specification: SearchSpecification, *, train: PartitionIdentity,
                      candidates: Sequence[str],
                      p_values: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """What the mining stage is allowed to produce: candidates, and no claims.

    The generate stage is where the family is large and the correction is hopeless, and that
    is fine as long as nothing here is reported as a result. Two things are checked. A
    candidate must be a declared member of the generate specification, because a candidate
    that was not declared was not enumerated and its family size is unknown. And the
    partition must not be the held-out one: mining on held-out data spends it.
    """
    if str(train.provenance.get("split", train.name)).strip().lower() in ("test", "held_out",
                                                                         "held-out"):
        raise InvalidParameterError(
            "train", train.name,
            "the training partition. This partition is the held-out one, and a candidate "
            "list mined from it has already used the data the confirmation is supposed to "
            "be independent of")
    declared = set(specification.labels())
    unknown = [label for label in candidates if label not in declared]
    if unknown:
        raise InvalidParameterError(
            "candidates", unknown[:8],
            "candidates drawn from the %d declared members of this specification. %d of the "
            "%d proposed were never declared, so they were never part of a priced family and "
            "the correction unit for them is unknown"
            % (len(declared), len(unknown), len(candidates)))
    return {
        "schema": "generation-report/v1",
        "stage": "generate",
        "partition": train.to_mapping(),
        "generate_sha256": specification.fingerprint(),
        "generate_family_size": specification.family_size,
        "n_candidates": len(candidates),
        "candidates": list(candidates),
        "p_values_uncorrected": ({str(k): float(v) for k, v in p_values.items()}
                                 if p_values else None),
        "claim_boundary": (
            "Candidates, not findings. These were selected on the training partition using "
            "the data, from a declared family of %d whose correction the ensemble cannot "
            "pay. No p-value here is evidence; the only thing this stage can produce is a "
            "confirmatory family to freeze." % specification.family_size),
    }


# ------------------------------------------------------------------------------- freeze


def freeze_confirmatory_family(
    generate: SearchSpecification, confirm: SearchSpecification, *,
    held_out: PartitionIdentity, sealed_at: str, study_id: str = "",
    ledger: Optional["HeldOutLedger"] = None,
) -> Seal:
    """Freeze `confirm` against `held_out`, or refuse to.

    The confirmatory family must be a subset of the generated one - confirming a member that
    was never generated is a new search, priced at nothing - and it must pass TG3.1's gate at
    its own size, which is the whole point: a family of four is affordable at 199 surrogates
    where the family of thirty-six it came from needs 3,005.
    """
    if not isinstance(sealed_at, str) or not sealed_at.strip():
        raise InvalidParameterError(
            "sealed_at", sealed_at,
            "when this declaration was frozen. A seal with no time on it cannot be shown to "
            "predate the held-out partition being opened, which is the only thing it claims")
    if str(held_out.provenance.get("split", "")).strip().lower() == "train":
        raise InvalidParameterError(
            "held_out", held_out.name,
            "a partition that was not mined. Its own lineage records it as the training "
            "split, and confirming on the data the candidates were selected from measures "
            "the selection, not the relationship")
    for name, spec in (("generate", generate), ("confirm", confirm)):
        if spec.family_size > MAX_ENUMERATED:
            raise InvalidParameterError(
                "%s family_size" % name, spec.family_size,
                "a family whose labels can be listed, so the confirmatory set can be shown "
                "to lie inside the generated one. Above %d members that comparison has to "
                "be made term by term" % MAX_ENUMERATED)
    generated = set(generate.labels())
    confirm_labels = confirm.labels()
    outside = sorted(set(confirm_labels) - generated)
    if outside:
        raise InvalidParameterError(
            "confirm", outside[:8],
            "a confirmatory family inside the generated one. %d of %d confirmatory members "
            "are not declared members of the generate specification, so they were never "
            "enumerated on train and this is a fresh search on held-out data under the name "
            "of a confirmation" % (len(outside), len(confirm_labels)),
            generate_family_size=generate.family_size)
    account = confirm.declare()
    if ledger is not None:
        ledger.require_unopened(held_out)
    return Seal(
        study_id=study_id, sealed_at=sealed_at,
        generate_sha256=generate.fingerprint(),
        generate_family_size=generate.family_size,
        confirm=confirm.to_mapping(), confirm_sha256=confirm.fingerprint(),
        confirm_labels=confirm_labels, confirm_family_size=confirm.family_size,
        confirm_account=account.describe(), held_out=held_out)


# ------------------------------------------------------------------------------- ledger


class HeldOutLedger:
    """A record of which held-out partitions have been spent, keyed by the partition.

    Keying on the partition rather than the seal is deliberate and is the substance of "once".
    A ledger keyed by seal would let a study write a second, entirely honest seal - correctly
    hashed, correctly frozen, affordable at its own size - and test the same held-out data
    again. Each confirmation would be individually defensible and the pair would be
    uncorrected, which is exactly the arithmetic R18 exists to stop.

    With a `path` the ledger is a JSON file, so the constraint survives the end of a process;
    without one it lasts as long as the object, which is enough for a test and not enough for
    a study.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: Dict[str, Dict[str, Any]] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            if stored.get("schema") != "held-out-ledger/v1":
                raise InvalidParameterError("ledger schema", stored.get("schema"),
                                            "the 'held-out-ledger/v1' schema tag")
            self._records = {str(k): dict(v) for k, v in stored.get("records", {}).items()}

    @property
    def records(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._records.items()}

    def _save(self) -> None:
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "held-out-ledger/v1", "records": self._records}, handle,
                      indent=2, sort_keys=True)

    def require_unopened(self, held_out: PartitionIdentity,
                         seal_sha256: str = "(unsealed)") -> None:
        record = self._records.get(held_out.digest())
        if record is not None:
            raise HeldOutAlreadyOpenedError(record, seal_sha256)

    def open(self, seal: Seal, held_out: PartitionIdentity, *, opened_at: str) -> Dict[str, Any]:
        """Spend the partition. Refuses if it has been spent, whatever seal spent it."""
        if not isinstance(opened_at, str) or not opened_at.strip():
            raise InvalidParameterError("opened_at", opened_at,
                                        "when the held-out partition was opened")
        self.require_unopened(held_out, seal.seal_sha256)
        record = {
            "partition_digest": held_out.digest(),
            "partition_name": held_out.name,
            "seal_sha256": seal.seal_sha256,
            "confirm_sha256": seal.confirm_sha256,
            "family_size": int(seal.confirm_family_size),
            "sealed_at": seal.sealed_at,
            "opened_at": opened_at,
            "study_id": seal.study_id,
        }
        self._records[held_out.digest()] = record
        self._save()
        return dict(record)


# --------------------------------------------------------------------------- confirm


def confirm_on_held_out(
    seal: Seal, *, p_values: Mapping[str, float], held_out: PartitionIdentity,
    ledger: HeldOutLedger, opened_at: str, published_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Test the frozen confirmatory family once, and correct it at its frozen size.

    Order matters here. The seal is verified, the partition is checked to be the one it named,
    the p-value set is checked to be exactly the frozen label set, and only then is the
    partition spent - so a refusal does not consume the held-out data, while a completed
    confirmation always does.
    """
    if published_sha256 is not None:
        seal.verify_published(published_sha256)
    else:
        seal.verify()
    if held_out.digest() != seal.held_out.digest():
        raise PartitionMismatchError(seal.held_out, held_out)

    frozen = list(seal.confirm_labels)
    supplied = {str(key): float(value) for key, value in p_values.items()}
    missing = [label for label in frozen if label not in supplied]
    extra = sorted(set(supplied) - set(frozen))
    if missing:
        raise InvalidParameterError(
            "p_values", missing[:8],
            "a p-value for every one of the %d frozen members. %d are absent, and a "
            "confirmatory family tested in part is a family chosen after looking: the "
            "correction is computed over what was frozen, so an untested member is a claim "
            "that it was tested and not reported" % (len(frozen), len(missing)),
            frozen_family_size=len(frozen))
    if extra:
        raise InvalidParameterError(
            "p_values", extra[:8],
            "p-values for the frozen members only. %d labels were supplied that this seal "
            "does not contain, so they were tested on the held-out partition without having "
            "been frozen - which is mining on held-out data, whatever it is called"
            % len(extra), frozen_family_size=len(frozen))

    correction = str(seal.confirm.get("correction", "benjamini_yekutieli"))
    alpha = float(seal.confirm.get("alpha", 0.05))
    ordered = [supplied[label] for label in frozen]
    corrected = adjust(ordered, method=correction, alpha=alpha,
                       n_tests=len(frozen), labels=frozen)
    record = ledger.open(seal, held_out, opened_at=opened_at)
    rejected = [label for label, flag in zip(frozen, corrected["rejected"]) if bool(flag)]
    return {
        "schema": "confirmation-receipt/v1",
        "stage": "confirm",
        "seal_sha256": seal.seal_sha256,
        "published_sha256": published_sha256,
        "confirm_sha256": seal.confirm_sha256,
        "generate_sha256": seal.generate_sha256,
        "generate_family_size": int(seal.generate_family_size),
        "correction_unit": len(frozen),
        "correction": correction,
        "dependence_assumption": ASSUMPTIONS.get(correction, "unstated"),
        "alpha": alpha,
        "held_out": held_out.to_mapping(),
        "ledger_record": record,
        "labels": frozen,
        "p_values": ordered,
        "adjusted": [float(q) for q in corrected["adjusted"]],
        "rejected": [bool(flag) for flag in corrected["rejected"]],
        "rejected_labels": rejected,
        "n_rejected": len(rejected),
        "claim_boundary": (
            "Corrected over the %d members frozen at %s, on a partition opened once, at %s. "
            "The %d members mined on train are not corrected for here and are not claims; "
            "this receipt says nothing about them. Nothing above is a statement about the "
            "mechanism - a confirmed member is a relationship that survived a surrogate null "
            "on data it was not selected from."
            % (len(frozen), seal.sealed_at, opened_at, int(seal.generate_family_size))),
    }
