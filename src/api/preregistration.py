"""Preregistration before the sweep: the generate/confirm split over HTTP (TG11.2, rule R18).

`core/preregistration.py` is 574 lines that nothing outside the tests could reach.  TG11.1 made
the analysis engine reachable and, in doing so, made it possible to look at a record first and
declare a family afterwards -- the exact freedom R18 exists to remove.  This module closes that,
and the ordering it enforces is **structural rather than presentational**: a confirmation cannot
be run against a partition whose seal does not already exist, and it is refused by the server
rather than hidden by the client.  An interface that enforced the ordering by which button it
displayed would defeat the module completely while appearing to use it.

**The client cannot tune a confirmatory run.**  Everything the confirmatory sweep needs -- the
lags, the ensemble size, the correction, the estimator, the bins, the seed, the split, the domain
and the clock -- is read back out of the seal.  The only thing ``/confirm`` accepts from a caller
is the record itself, and even that is checked against the partition the seal named.  A knob the
caller could still turn after sealing would be a family member chosen after the declaration.

**Three things this module deliberately does not do.**  It records no evidence and moves no rung
(R22); a confirmation receipt is an input to TG11.3's write path, not a claim.  It does not sign
anything: a seal stored here is a local copy, and `verify_published` against a digest recorded
somewhere the author cannot rewrite is the only check that carries weight -- every response that
hands out a seal says so.  And it does not make the confirmatory family cheaper; it makes it
small *and written down first*, which is a different thing and the only legitimate one.

**Why the partition identity is rebuilt here rather than taken from `from_series` (D65).**  A
`ChannelSeries` read from an upload carries `path_basename` in its provenance, and
`PartitionIdentity.from_series` hashes provenance wholesale.  In process that is honest lineage.
On an HTTP boundary it is a hole: the same held-out bytes re-uploaded as ``data2.csv`` would hash
to a different partition and the ledger's "once" would not fire.  So the identity is built here
from the fields that identify the *data* -- content digest, clock, columns, split window -- and
not from the fields that describe its delivery.  The filename, the adapter name and the domain
under which the file was read are all excluded: two confirmations on the same held-out bytes are
two tests of those bytes, whatever they were called or read as.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.analysis_engine.domain_analysis import analyse_precedence, association_only
from src.core.channel_series import split_channel_series
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.family import SearchAxis, SearchSpecification, SearchTerm
from src.core.preregistration import (HeldOutAlreadyOpenedError, HeldOutLedger,
                                      PartitionIdentity, Seal, confirm_on_held_out,
                                      freeze_confirmatory_family)
from src.data_layer.tabular_source import VALUE_MEASURE, read_channels_for_domain


router = APIRouter(prefix="/api/v1/preregistration", tags=["preregistration"])

#: Where seals and the held-out ledger live.  Overridable so a test never writes into the
#: study's real ledger -- a test that spent a real partition would spend it permanently.
ROOT_ENV = "SPECTRAL_PREREGISTRATION_ROOT"
DEFAULT_ROOT = Path("data") / "preregistrations"

LEDGER_NAME = "held_out_ledger.json"
SEAL_SCHEMA_HTTP = "spectral.preregistration.http.v1"

#: Provenance keys that identify the held-out *data*.  See the module docstring (D65): every
#: other key describes how the bytes arrived, not which bytes they were.
IDENTIFYING_PROVENANCE: Tuple[str, ...] = (
    "content_sha256", "n_rows", "columns", "time_column", "time_units", "cadence_seconds",
)

#: The settings a seal freezes so the confirmatory sweep can be reproduced from it alone.
#: Sealed as `notes` on the confirmatory specification, so an edit to any of them breaks the
#: seal's digest rather than silently producing a different analysis under the same seal.
SEALED_RUN_KEYS: Tuple[str, ...] = (
    "estimator", "bins", "seed", "measure", "cadence_seconds", "domain", "time_column",
    "time_units", "delimiter", "train_ratio", "embargo_frames",
)


# ------------------------------------------------------------------------------- storage


def _root() -> Path:
    return Path(os.environ.get(ROOT_ENV, str(DEFAULT_ROOT)))


def _ledger() -> HeldOutLedger:
    """The ledger, on disk.

    Single-process by construction: `HeldOutLedger` reads, mutates and rewrites a JSON file
    with no lock, so two workers spending two different partitions concurrently can lose one
    record.  That is recorded rather than papered over -- the failure is a *lost* spend, never
    a spend that appears not to have happened to the request that made it, because `open`
    refuses from the copy it just read.  A multi-worker deployment needs a real store.
    """
    # The directory is deliberately not created here.  Reading the ledger is what TG11.1's
    # gate does on every run, and a read that created the study's store would make an empty
    # directory appear as a side effect of an analysis that preregistered nothing.
    # `_store_seal` creates it, because sealing is the first act with anything to keep.
    return HeldOutLedger(path=str(_root() / LEDGER_NAME))


def _seal_path(seal_sha256: str) -> Path:
    if not seal_sha256.isalnum() or len(seal_sha256) != 64:
        raise HTTPException(
            status_code=404,
            detail="%r is not a seal digest. A seal is addressed by the 64-character SHA-256 "
                   "it hashes to." % seal_sha256)
    return _root() / ("%s.json" % seal_sha256.lower())


def _store_seal(seal: Seal) -> Path:
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("%s.json" % seal.seal_sha256)
    if path.exists():
        raise HTTPException(
            status_code=409,
            detail="A seal with digest %s already exists. Two declarations that hash to the "
                   "same digest are the same declaration; sealing it twice would put two "
                   "sealing times on one frozen family, and the earlier one is the only "
                   "honest answer to when it was frozen." % seal.seal_sha256)
    path.write_text(json.dumps(seal.to_mapping(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_seal(seal_sha256: str) -> Seal:
    path = _seal_path(seal_sha256)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No seal %s is stored here. A confirmation runs only against a declaration "
                   "frozen before the held-out partition was opened, so there is nothing to "
                   "confirm." % seal_sha256)
    return Seal.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _stored_seals() -> List[Seal]:
    root = _root()
    if not root.exists():
        return []
    seals = []
    for path in sorted(root.glob("*.json")):
        if path.name == LEDGER_NAME:
            continue
        seals.append(Seal.from_mapping(json.loads(path.read_text(encoding="utf-8"))))
    return seals


def held_out_ledger() -> HeldOutLedger:
    """The programme's one held-out ledger, exported for TG11.4.

    Mining spends held-out *frames* where this module spends held-out *rows*, but "this
    partition has been opened" is one fact about the programme rather than one per surface,
    and two ledgers would let the same data be spent once on each.  Same file, same
    single-process caveat as `_ledger`.
    """
    return _ledger()


def store_seal(seal: Seal) -> Path:
    """Publish a seal into the shared store, exported for TG11.4.

    A motif seal and a lag-family seal are the same object and belong in the same drawer:
    `GET /preregistration/seals` is meant to answer "what has this programme frozen", and an
    answer that omitted every mining declaration would be wrong in the direction that
    matters - it would show a study as having preregistered less than it did.
    """
    return _store_seal(seal)


def load_seal(seal_sha256: str) -> Seal:
    """One stored seal by digest, exported for TG11.4's confirmatory run."""
    return _load_seal(seal_sha256)


# ---------------------------------------------------------------------------- the record


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="%r is not UTF-8 text. Preregistration binds itself to an admitted channel "
                   "table." % filename)


def _json_object(raw: str, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    return value


def _lags(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidParameterError(field, value, "a non-empty JSON array of frame lags")
    try:
        lags = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        raise InvalidParameterError(field, value, "integer frame lags")
    if any(lag < 1 for lag in lags) or tuple(sorted(set(lags))) != lags:
        raise InvalidParameterError(
            field, list(lags),
            "strictly increasing positive lags. A repeated lag is one test counted twice, and "
            "the family would be priced short by exactly that duplication")
    return lags


def _read(text: str, filename: str, *, domain: str, time_column: str, time_units: str,
          delimiter: str) -> Tuple[Any, Any, float]:
    series, declaration = read_channels_for_domain(
        text, source_name=filename, domain=domain, time_column=time_column,
        time_units=time_units, delimiter=delimiter)
    cadence = series.provenance.get("cadence_seconds")
    if not isinstance(cadence, (int, float)) or cadence <= 0:
        raise InvalidParameterError(
            "record clock", cadence,
            "a regular clock with a finite positive cadence. A seal binds a family declared in "
            "frame lags, and a lag in frames is not a duration on an irregular clock")
    return series, declaration, float(cadence)


def _identity(series: Any, *, name: str, split: str, start: int, stop: int,
              train_ratio: float, embargo_frames: int) -> PartitionIdentity:
    """A partition identity built from what identifies the data, and nothing else (D65)."""
    provenance = {key: series.provenance.get(key) for key in IDENTIFYING_PROVENANCE
                  if series.provenance.get(key) is not None}
    provenance.update({"split": split, "split_frames": [int(start), int(stop)],
                       "split_train_ratio": float(train_ratio),
                       "split_embargo_frames": int(embargo_frames)})
    return PartitionIdentity(
        name=name, n_times=int(stop - start), n_channels=int(series.n_channels),
        channel_labels=tuple(str(c) for c in series.channels),
        frames=(int(start), int(stop)), provenance=provenance)


def _partitions(series: Any, *, train_ratio: float, embargo_frames: int
                ) -> Tuple[Dict[str, Any], Dict[str, PartitionIdentity]]:
    """Split, and identify both sides. Reads no measure value into either identity."""
    splits = split_channel_series(series, train_ratio=train_ratio,
                                  embargo_frames=embargo_frames)
    identities = {}
    for split, name in (("train", "train"), ("test", "held_out")):
        part = splits[split]
        start, stop = part.provenance["split_frames"]
        identities[split] = _identity(part, name=name, split=split, start=int(start),
                                      stop=int(stop), train_ratio=train_ratio,
                                      embargo_frames=embargo_frames)
    return splits, identities


def held_out_identity_for(series: Any, *, train_ratio: float, embargo_frames: int
                          ) -> Tuple[PartitionIdentity, Optional[Dict[str, Any]]]:
    """The held-out identity a split would produce, and the ledger record that spent it.

    Exported for TG11.1's gate operation, which creates a test partition of its own and
    reports a verdict on it.  Reading the ledger from there is what makes R18's ordering
    structural rather than presentational: the refusal is the server's, and it does not depend
    on the analysis surface knowing anything about seals beyond whether this data is gone.
    """
    _, parts = _partitions(series, train_ratio=train_ratio, embargo_frames=embargo_frames)
    identity = parts["test"]
    return identity, _ledger().records.get(identity.digest())


# ------------------------------------------------------------------------- declarations


def _specification(*, channels: Sequence[Any], lags: Tuple[int, ...], n_surrogates: int,
                   alpha: float, correction: str, study_id: str, notes: Mapping[str, Any]
                   ) -> SearchSpecification:
    """Ordered distinct channel pairs crossed with lags -- the shape the sweep actually emits.

    The same two terms and the same `label_format` as `GateProtocol.search_specification`, and
    a test asserts the two agree label for label.  The axis is declared over the record's own
    **channel names** rather than over ``1..n``, because that is what `cross_scale_dependency`
    renders into a label: a family declared over positions would freeze labels the sweep never
    emits, and every frozen member would come back missing at confirmation -- a design error
    discovered at the one moment the partition may be opened.

    The confirmatory family is a *lag* narrowing of the generated one, so its labels are a
    genuine subset of the sweep's rather than a set filtered out of it afterwards.  A filtered
    sweep would compute tests on held-out data and then decline to count them, which corrects
    at a smaller family than was actually tested.
    """
    return SearchSpecification(
        terms=(
            SearchTerm("ordered_pairs",
                       (SearchAxis("scale", tuple(channels)),)),
            SearchTerm("product", (SearchAxis("lag_frames", tuple(int(v) for v in lags)),)),
        ),
        n_surrogates=int(n_surrogates), alpha=float(alpha), correction=correction,
        label_format="{0}->{1}@{2}", study_id=study_id, notes=dict(notes))


def _sweep(series: Any, declaration: Any, *, lags: Tuple[int, ...], cadence_seconds: float,
           settings: Mapping[str, Any], n_surrogates: int, alpha: float,
           correction: str) -> Dict[str, Any]:
    """Run the frozen family on one partition, through the domain path that enforces R21."""
    common = dict(
        lags=list(lags), cadence_seconds=cadence_seconds, measure=str(settings["measure"]),
        estimator=str(settings["estimator"]), bins=int(settings["bins"]),
        n_surrogates=int(n_surrogates), alpha=float(alpha), correction=correction,
        seed=int(settings["seed"]))
    if declaration.precedence_admissible:
        return analyse_precedence(series, declaration, **common)
    return association_only(series, declaration, **common)


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    # A spent partition is a state conflict, not a malformed request: the caller sent nothing
    # wrong and there is no edit that would make this call succeed.  It is the one refusal in
    # this module a client must not present as "fix your input and try again".
    status = 409 if isinstance(error, HeldOutAlreadyOpenedError) else info["status_code"]
    return HTTPException(status_code=status, detail=info["detail"])


PUBLICATION_NOTE = (
    "This seal is a local copy and is not evidence about itself: anyone who can rewrite the "
    "file can recompute every digest in it, and nothing here is signed. Publish `seal_sha256` "
    "somewhere you cannot rewrite -- a commit, a registry entry, a preregistration record -- "
    "before the held-out partition is opened, and pass it back as `published_sha256` when you "
    "confirm. That comparison is the only check with weight.")


# --------------------------------------------------------------------------------- routes


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """What a seal binds, and what it does not."""
    return {
        "schema": SEAL_SCHEMA_HTTP,
        "steps": {
            "partition": ("Split the record and see both partition identities, without "
                          "binding anything and without reading a measure value."),
            "seal": ("Freeze a confirmatory family, narrowed by lag from a generated one, "
                     "against the held-out partition. Refused if that partition is already "
                     "spent, and the sealing time is the server's, not the caller's."),
            "confirm": ("Open the held-out partition once. Every setting is read from the "
                        "seal; the only thing accepted from the caller is the record."),
        },
        "narrowing": ("The confirmatory family narrows the generated one by lag. The channel "
                      "pairs are whatever the record has, because the sweep tests every "
                      "ordered pair: a family narrower than that would leave the engine "
                      "computing tests on held-out data that the correction did not count."),
        "once_is_per": ("the held-out data -- its content digest, clock, columns and frame "
                        "window. Not the filename, not the domain it was read under, and not "
                        "the seal: a second seal over the same held-out bytes is a second "
                        "test of them, however honest each seal is alone."),
        "records_evidence": False,
        "moves_rung": False,
        "publication": PUBLICATION_NOTE,
        "claim_boundary": (
            "A seal is a promise about ordering, not a result. It says a family was fixed "
            "before a partition was opened; it says nothing about whether the family is any "
            "good, and a confirmation receipt is an input to the evidence write path (TG11.3) "
            "rather than a claim."),
    }


@router.post("/partition")
async def describe_partition(
    file: UploadFile = File(...),
    domain: str = Form(...),
    time_column: str = Form(...),
    train_ratio: float = Form(0.6),
    embargo_frames: int = Form(1),
    time_units: str = Form("s"),
    delimiter: str = Form(","),
) -> Dict[str, Any]:
    """Show what a seal would bind itself to, before it binds anything."""
    filename = file.filename or "upload"
    text = _decode(await file.read(), filename)
    try:
        series, declaration, cadence = _read(text, filename, domain=domain,
                                             time_column=time_column, time_units=time_units,
                                             delimiter=delimiter)
        _, parts = _partitions(series, train_ratio=float(train_ratio),
                               embargo_frames=int(embargo_frames))
        held_out = parts["test"]
        record = _ledger().records.get(held_out.digest())
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": SEAL_SCHEMA_HTTP,
        "domain": declaration.name,
        "frames": series.n_times,
        "channels": list(series.channels),
        "cadence_seconds": cadence,
        "train": {**parts["train"].to_mapping(), "digest": parts["train"].digest()},
        "held_out": {**held_out.to_mapping(), "digest": held_out.digest()},
        "already_opened": record is not None,
        "opened_record": record,
        "read_only": True,
        "stored": False,
        "claim_boundary": (
            "Geometry and lineage only. No measure value from either partition was read to "
            "build these identities, so seeing this is not seeing the held-out data."),
    }


@router.post("/seal")
async def seal_family(
    file: UploadFile = File(...),
    domain: str = Form(...),
    time_column: str = Form(...),
    study_id: str = Form(...),
    generate: str = Form(...),
    confirm: str = Form(...),
    train_ratio: float = Form(0.6),
    embargo_frames: int = Form(1),
    time_units: str = Form("s"),
    delimiter: str = Form(","),
) -> Dict[str, Any]:
    """Freeze a confirmatory family against the held-out partition, or refuse to."""
    filename = file.filename or "upload"
    text = _decode(await file.read(), filename)
    generated = _json_object(generate, "generate")
    confirmed = _json_object(confirm, "confirm")
    try:
        series, declaration, cadence = _read(text, filename, domain=domain,
                                             time_column=time_column, time_units=time_units,
                                             delimiter=delimiter)
        generate_lags = _lags(generated.get("lags"), "generate.lags")
        confirm_lags = _lags(confirmed.get("lags"), "confirm.lags")
        outside = sorted(set(confirm_lags) - set(generate_lags))
        if outside:
            raise InvalidParameterError(
                "confirm.lags", outside,
                "lags drawn from the generated family. A confirmatory lag that was never "
                "generated was never tested on train, so it is a fresh search on held-out "
                "data under the name of a confirmation")

        settings = {
            "estimator": str(generated.get("estimator", "mutual_information")),
            "bins": int(generated.get("bins", 4)),
            "seed": int(generated.get("seed", 20260827)),
            "measure": VALUE_MEASURE,
            "cadence_seconds": cadence,
            "domain": declaration.name,
            "time_column": time_column,
            "time_units": time_units,
            "delimiter": delimiter,
            "train_ratio": float(train_ratio),
            "embargo_frames": int(embargo_frames),
        }
        _, parts = _partitions(series, train_ratio=float(train_ratio),
                               embargo_frames=int(embargo_frames))

        # The confirmatory lags must clear the domain's registered floor now, at declaration
        # time.  A frozen member the engine would refuse to test is a family that cannot be
        # confirmed, and finding that out during the one permitted opening would waste the
        # partition on a design error.
        _lag_floor_admits(series, declaration, confirm_lags)

        channels = tuple(str(name) for name in series.channels)
        generate_spec = _specification(
            channels=channels, lags=generate_lags,
            n_surrogates=int(generated.get("n_surrogates", 499)),
            alpha=float(generated.get("alpha", 0.05)),
            correction=str(generated.get("correction", "benjamini_yekutieli")),
            study_id=study_id, notes=dict(settings, stage="generate"))
        confirm_spec = _specification(
            channels=channels, lags=confirm_lags,
            n_surrogates=int(confirmed.get("n_surrogates", 499)),
            alpha=float(confirmed.get("alpha", 0.05)),
            correction=str(confirmed.get("correction", "benjamini_yekutieli")),
            study_id=study_id, notes=dict(settings, stage="confirm"))

        seal = freeze_confirmatory_family(
            generate_spec, confirm_spec, held_out=parts["test"],
            sealed_at=datetime.now(timezone.utc).isoformat(), study_id=study_id,
            ledger=_ledger())
    except SpectralEarthError as error:
        raise _handle(error)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400,
                            detail="Malformed preregistration declaration: %s" % error)

    _store_seal(seal)
    return {
        "schema": SEAL_SCHEMA_HTTP,
        "seal_sha256": seal.seal_sha256,
        "sealed_at": seal.sealed_at,
        "sealed_at_source": ("the server clock. A caller-supplied sealing time could be "
                             "written after the partition was opened, and the time is the "
                             "whole of what a seal claims."),
        "seal": seal.to_mapping(),
        "generate_family_size": int(seal.generate_family_size),
        "confirm_family_size": int(seal.confirm_family_size),
        "confirm_labels": list(seal.confirm_labels),
        "held_out_digest": seal.held_out.digest(),
        "records_evidence": False,
        "rung_moved": False,
        "publication": PUBLICATION_NOTE,
        "claim_boundary": (
            "Nothing has been tested. The confirmatory family of %d is affordable at %d "
            "surrogates where the generated family of %d is not, and that is the only thing "
            "this exchange bought."
            % (int(seal.confirm_family_size), int(confirm_spec.n_surrogates),
               int(seal.generate_family_size))),
    }


def _lag_floor_admits(series: Any, declaration: Any, lags: Tuple[int, ...]) -> None:
    """Refuse a confirmatory lag the domain's own policy would exclude at confirmation."""
    from src.analysis_engine.domain_analysis import _bind  # local: refusal helper only

    bound = _bind(declaration, None, {})
    bound.check_lags([int(lag) for lag in lags])


@router.get("/seals")
async def list_seals() -> Dict[str, Any]:
    """Every stored seal, and whether its partition has been spent."""
    ledger = _ledger().records
    rows = []
    for seal in _stored_seals():
        digest = seal.held_out.digest()
        rows.append({
            "seal_sha256": seal.seal_sha256,
            "study_id": seal.study_id,
            "sealed_at": seal.sealed_at,
            "confirm_family_size": int(seal.confirm_family_size),
            "generate_family_size": int(seal.generate_family_size),
            "held_out_digest": digest,
            "held_out_frames": list(seal.held_out.frames),
            "spent": digest in ledger,
            "spent_by": ledger.get(digest, {}).get("seal_sha256"),
        })
    return {"schema": SEAL_SCHEMA_HTTP, "seals": rows, "n_seals": len(rows),
            "publication": PUBLICATION_NOTE}


@router.get("/seals/{seal_sha256}")
async def read_seal(seal_sha256: str, published_sha256: Optional[str] = None) -> Dict[str, Any]:
    """One seal, with its digests recomputed rather than repeated back."""
    seal = _load_seal(seal_sha256)
    try:
        verification = (seal.verify_published(published_sha256) if published_sha256
                        else seal.verify())
    except SpectralEarthError as error:
        raise _handle(error)
    digest = seal.held_out.digest()
    record = _ledger().records.get(digest)
    return {
        "schema": SEAL_SCHEMA_HTTP,
        "seal": seal.to_mapping(),
        "verification": verification,
        "checked_against_publication": published_sha256 is not None,
        "held_out_digest": digest,
        "spent": record is not None,
        "spent_record": record,
        "publication": PUBLICATION_NOTE,
    }


@router.post("/seals/{seal_sha256}/confirm")
async def confirm(
    seal_sha256: str,
    file: UploadFile = File(...),
    published_sha256: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Open the held-out partition once, on the family frozen before it was opened.

    The caller supplies the record and nothing else.  Domain, clock, split, estimator, bins,
    seed, lags, ensemble size, alpha and correction all come out of the seal, so there is no
    setting left that could be chosen after the declaration.  The order inside
    `confirm_on_held_out` is what makes a refusal safe: the seal is verified, the partition is
    matched, the label set is checked, and only then is the ledger written -- so a design error
    costs nothing and a completed confirmation always spends the partition.
    """
    seal = _load_seal(seal_sha256)
    filename = file.filename or "upload"
    text = _decode(await file.read(), filename)
    settings = dict(seal.confirm.get("notes", {}))
    missing = sorted(key for key in SEALED_RUN_KEYS if key not in settings)
    if missing:
        raise HTTPException(
            status_code=409,
            detail="Seal %s does not carry %s, so the confirmatory sweep it describes cannot "
                   "be reproduced from it. It was written by a different version of this "
                   "surface and cannot be confirmed here." % (seal.seal_sha256, missing))

    try:
        series, declaration, cadence = _read(
            text, filename, domain=str(settings["domain"]),
            time_column=str(settings["time_column"]), time_units=str(settings["time_units"]),
            delimiter=str(settings["delimiter"]))
        splits, parts = _partitions(series, train_ratio=float(settings["train_ratio"]),
                                    embargo_frames=int(settings["embargo_frames"]))
        lags = _sealed_lags(seal)
        result = _sweep(splits["test"], declaration, lags=lags, cadence_seconds=cadence,
                        settings=settings,
                        n_surrogates=int(seal.confirm.get("n_surrogates", 499)),
                        alpha=float(seal.confirm.get("alpha", 0.05)),
                        correction=str(seal.confirm.get("correction",
                                                        "benjamini_yekutieli")))
        p_values = {str(row["label"]): float(row["p_value"])
                    for row in result.get("results", []) if "label" in row}
        receipt = confirm_on_held_out(
            seal, p_values=p_values, held_out=parts["test"], ledger=_ledger(),
            opened_at=datetime.now(timezone.utc).isoformat(),
            published_sha256=published_sha256 or None)
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": SEAL_SCHEMA_HTTP,
        "seal_sha256": seal.seal_sha256,
        "checked_against_publication": bool(published_sha256),
        "receipt": receipt,
        "sweep_warnings": list(result.get("warnings", [])),
        "records_evidence": False,
        "rung_moved": False,
        "publication": (PUBLICATION_NOTE if not published_sha256 else
                        "Checked against the published digest supplied with this request."),
        "claim_boundary": (
            "%s The held-out partition is now spent: this surface will refuse every further "
            "confirmation against it, under this seal or any other."
            % receipt["claim_boundary"]),
    }


def _sealed_lags(seal: Seal) -> Tuple[int, ...]:
    """The frozen lags, read back out of the sealed specification rather than re-declared."""
    for term in seal.confirm.get("terms", []):
        for axis in term.get("axes", []):
            if axis.get("name") == "lag_frames":
                return tuple(int(value) for value in axis["values"])
    raise HTTPException(
        status_code=409,
        detail="Seal %s declares no `lag_frames` axis, so there is no frozen family to test."
               % seal.seal_sha256)


__all__ = ["router", "held_out_identity_for", "held_out_ledger", "store_seal", "load_seal",
           "ROOT_ENV", "DEFAULT_ROOT", "IDENTIFYING_PROVENANCE", "SEALED_RUN_KEYS"]
