"""Cross-domain temporal precedence on an exact physical clock, over HTTP (TG11.4b).

`core/cross_domain.py` is the module that makes a lag stop being a frame count.  Everywhere else
in this programme a lag family is declared in frames of one record's clock, which is honest only
while there is one clock.  Two domains sampled hourly and three-hourly have two, and a family
declared in frames of either one is a family the other domain cannot read.  This surface is where
a lag is declared in **seconds** and converted per domain, and it was reachable only from the
benchmarks until now.

**Three things this boundary will not do, all of them things it would be easy to offer.**

*It will not resample.*  Two channel tables are aligned by exact timestamp intersection, and if
the intersection is too small or irregular the request is refused.  The tempting repair -- a
nearest-neighbour join, or a linear fill onto a common grid -- manufactures values at times one
source did not observe and then lets those manufactured values vote in a lagged relationship.
The response reports how many native observations each domain kept and discarded, because a
join that silently kept a twentieth of one record is a different study from the one that was
declared.

*It will not default a unit.*  A channel table carries column names and numbers; it does not
carry what its columns mean.  R19 says the source semantics and units may never be dropped, so
they are required per channel and a table whose columns are not all declared is refused rather
than read under `"unknown"`.  A cross-domain result compares no raw magnitude, and the only
thing that keeps that true is that both magnitudes still say what they are.

*It will not let the caller tune a confirmation.*  As in TG11.2 and TG11.4, `/confirm` reads the
domains, the columns, the delimiters, the declared units, the lag family in seconds, the split,
the embargo, the ensemble, alpha, the correction and the seed back out of the seal.  The only
thing accepted from the caller is the two records, and those are checked -- by the seal's own
digest, which an edit breaks, and by the held-out partition identity, which a different pair of
files cannot reproduce.  Both refusals happen before the ledger is written, so re-uploading the
wrong records confirms nothing and costs nothing.  The frozen members are then re-derived by
re-running the deterministic training sweep rather than reconstructed from their labels, so the
family that is tested is the one this record selects and not one that only shares its names.

**Identity, and one residual (D65).**  The held-out partition identity is the aligned record's
own, which hashes its provenance wholesale.  That is safe here because every field in that
provenance is one this module put there: the two content digests, the two clocks, the shared
clock digest, the retained and discarded counts, the split window and the embargo.  The filename
is not among them, so the same held-out bytes re-uploaded under another name are the same
partition and the ledger's "once" still fires.  What *is* among them is each domain's
declaration, and that is the residual: the same two files aligned under two different registered
domains produce two partition digests, so the ledger would let them be opened twice.  It is
recorded rather than papered over -- the alternative, stripping the domain out of the identity,
would make two genuinely different analyses collide -- and it is stated in the claim boundary
this surface returns.

Nothing here implements an estimator, a null, a correction or a lag floor.  It records no
evidence and moves no rung (R22); a cross-domain confirmation receipt is an input to TG11.3's
write path, exactly as a precedence one is.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.cross_domain import (CROSS_DOMAIN_SCHEMA, DomainChannel, DomainTimeSeries,
                                   align_exact, confirm_cross_domain, cross_domain_pairs,
                                   physical_lags, sweep_cross_domain)
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.precedence import (MIN_FRAMES, PrecedenceCandidate, Record,
                                 affordable_member_count, choose_candidates,
                                 freeze_precedence, report_precedence_generation,
                                 split_with_embargo)
from src.core.preregistration import HeldOutAlreadyOpenedError, Seal
from src.statistics.multiple_comparisons import required_surrogates
from src.data_layer.tabular_source import VALUE_MEASURE, read_channels_for_domain
from src.api.preregistration import held_out_ledger, load_seal, store_seal


router = APIRouter(prefix="/api/v1/cross-domain", tags=["cross-domain"])

CROSS_DOMAIN_SCHEMA_HTTP = "spectral.cross_domain.http.v1"

#: Written into every seal this surface freezes, and checked at `/confirm`. A seal frozen by
#: another surface describes a family this one cannot reproduce, and running it anyway would
#: spend a held-out partition on a guess about what the seal meant.
SURFACE = "cross_domain"

#: The keys a reading of one side is allowed to carry. Unknown keys are refused rather than
#: ignored, because an ignored delimiter or time column would read a different record under
#: the same seal.
SOURCE_KEYS = {"domain", "time_column", "time_units", "delimiter", "channels",
               "aggregation_window_seconds"}

#: What a seal must carry for `/confirm` to reproduce the run from it alone.
SEALED_RUN_KEYS: Tuple[str, ...] = (
    "surface", "first", "second", "name", "lag_seconds", "fraction", "embargo_frames",
    "seed", "n_surrogates", "alpha", "correction",
)

DEFAULT_SEED = 20260827
DEFAULT_SURROGATES = 199

PUBLICATION_NOTE = (
    "This seal is a local copy and is not evidence about itself. Publish `seal_sha256` "
    "somewhere you cannot rewrite before the held-out partition is opened, and pass it back as "
    "`published_sha256` when you confirm; that comparison is the only check with weight.")

CLAIM_BOUNDARY = (
    "Cross-domain structural temporal association only. The two records' raw magnitudes keep "
    "different semantics and units and are never compared, the statistic is dimensionless, and "
    "precedence identifies no causal mechanism (R19, R21). A confirmation receipt records no "
    "evidence and moves no rung (R22). The held-out ledger identifies a partition by the data "
    "and its split; the same two files aligned under two different registered domains are two "
    "partitions to it, and this surface cannot detect that they hold the same rows.")


# --------------------------------------------------------------------------- the request


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="%r is not UTF-8 text. Cross-domain alignment reads two admitted channel "
                   "tables." % filename)


def _json_object(raw: str, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    return value


def _source(raw: str, field: str) -> Dict[str, Any]:
    """One side's reading: which domain, which columns, and what the columns mean."""
    value = _json_object(raw, field)
    unknown = sorted(set(value) - SOURCE_KEYS)
    if unknown:
        raise InvalidParameterError(
            "%s keys" % field, unknown,
            "only %s. An unknown reading setting is refused rather than ignored: an ignored "
            "time column or delimiter would align a different record under the same seal"
            % sorted(SOURCE_KEYS))
    for required in ("domain", "time_column", "channels"):
        if required not in value:
            raise InvalidParameterError(
                "%s.%s" % (field, required), None,
                "a declared value. Neither the domain a record is read under nor what its "
                "columns mean can be guessed from the file")
    channels = value["channels"]
    if not isinstance(channels, dict) or not channels:
        raise InvalidParameterError(
            "%s.channels" % field, channels,
            "a JSON object mapping every column name to its `semantics` and `units`")
    for name, described in channels.items():
        if not isinstance(described, dict) or set(described) != {"semantics", "units"}:
            raise InvalidParameterError(
                "%s.channels.%s" % (field, name), described,
                "exactly `semantics` and `units`. A cross-domain result compares no raw "
                "magnitude, and what keeps that honest is that both magnitudes still say "
                "what they are (R19)")
    return {
        "domain": str(value["domain"]),
        "time_column": str(value["time_column"]),
        "time_units": str(value.get("time_units", "s")),
        "delimiter": str(value.get("delimiter", ",")),
        "channels": {str(k): {"semantics": str(v["semantics"]), "units": str(v["units"])}
                     for k, v in channels.items()},
        "aggregation_window_seconds": value.get("aggregation_window_seconds"),
    }


def _lag_seconds(raw: str, field: str = "lag_seconds") -> Tuple[float, ...]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400,
                            detail="`%s` must be a JSON array of durations in seconds." % field)
    if not isinstance(value, list) or not value:
        raise InvalidParameterError(
            field, value,
            "a non-empty JSON array of physical durations in seconds. A frame count is not a "
            "duration when two clocks are involved, which is the whole reason this surface "
            "exists")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise InvalidParameterError(field, value, "numeric durations in seconds")


def _series(text: str, filename: str, spec: Mapping[str, Any]) -> DomainTimeSeries:
    """Read one side under its declared domain and dress it in its own vocabulary.

    The `dataset_id` is the record's content digest rather than its filename. A dataset
    identifier is meant to say *which observations*, and it travels into the partition
    identity: taking it from the upload's name would make the same bytes two datasets and
    the held-out ledger's "once" would stop firing (D65).
    """
    series, declaration = read_channels_for_domain(
        text, source_name=filename, domain=str(spec["domain"]),
        time_column=str(spec["time_column"]), time_units=str(spec["time_units"]),
        delimiter=str(spec["delimiter"]))

    declared = set(spec["channels"])
    present = set(str(name) for name in series.channels)
    if declared != present:
        raise InvalidParameterError(
            "channels", sorted(declared ^ present),
            "one `semantics` and `units` declaration per column in the file, and no others. "
            "Columns %s were read but not declared, and %s were declared but not read; "
            "neither is defaulted, because a channel whose meaning was guessed is a "
            "magnitude nobody chose to compare (R19)"
            % (sorted(present - declared), sorted(declared - present)))

    matrix = series.to_matrix(VALUE_MEASURE)
    digest = series.provenance.get("content_sha256")
    if not isinstance(digest, str) or not digest:
        raise InvalidParameterError(
            "content_sha256", digest,
            "a content digest from the reader. Without it this record cannot be identified "
            "by what it holds rather than by what it was called")
    channels = tuple(
        DomainChannel(label=str(name),
                      semantics=spec["channels"][str(name)]["semantics"],
                      units=spec["channels"][str(name)]["units"],
                      values=tuple(float(v) for v in matrix[:, index]))
        for index, name in enumerate(series.channels))
    window = spec.get("aggregation_window_seconds")
    return DomainTimeSeries(
        declaration=declaration, dataset_id=digest,
        times_seconds=tuple(float(v) for v in series.times_seconds),
        channels=channels,
        aggregation_window_seconds=None if window is None else float(window),
        # Identifying fields only. Everything describing how the bytes arrived - the
        # filename, the adapter, the upload order - is deliberately absent (D65).
        provenance={"content_sha256": digest,
                    "n_rows": int(series.n_times),
                    "columns": [str(name) for name in series.channels],
                    "time_column": str(spec["time_column"]),
                    "time_units": str(spec["time_units"]),
                    "delimiter": str(spec["delimiter"])})


async def _both(first: UploadFile, second: UploadFile, first_source: str, second_source: str,
                name: str) -> Tuple[Record, Dict[str, Any], Dict[str, Any]]:
    """Read both sides and align them, or refuse. Nothing here is stored."""
    first_spec, second_spec = _source(first_source, "first"), _source(second_source, "second")
    first_text = _decode(await first.read(), first.filename or "first")
    second_text = _decode(await second.read(), second.filename or "second")
    one = _series(first_text, first.filename or "first", first_spec)
    two = _series(second_text, second.filename or "second", second_spec)
    return align_exact(one, two, name=name), first_spec, second_spec


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    # A spent partition is a state conflict, not a malformed request: there is no edit to the
    # request that would make opening it again legitimate.
    status = 409 if isinstance(error, HeldOutAlreadyOpenedError) else info["status_code"]
    return HTTPException(status_code=status, detail=info["detail"])


def _alignment(record: Record) -> Dict[str, Any]:
    """The lineage of an aligned record, with no value from either side read."""
    block = record.provenance["cross_domain"]
    return {
        "schema": block["schema"],
        "name": record.name,
        "alignment": block["alignment"],
        "interpolation": block["interpolation"],
        "clock_sha256": block["clock_sha256"],
        "clock_start_seconds": block["clock_start_seconds"],
        "clock_stop_seconds": block["clock_stop_seconds"],
        "common_cadence_seconds": block["common_cadence_seconds"],
        "n_common_observations": block["n_common_observations"],
        "physical_lag_floor_seconds": block["physical_lag_floor_seconds"],
        "retained_native_observations": dict(block["retained_native_observations"]),
        "discarded_native_observations": dict(block["discarded_native_observations"]),
        "domains": {name: {"dataset_id": domain["dataset_id"],
                           "native_cadence_seconds": domain["native_cadence_seconds"],
                           "minimum_lag_seconds": domain["minimum_lag_seconds"],
                           "aggregation_window_seconds": domain["aggregation_window_seconds"],
                           "n_native_observations": domain["n_native_observations"]}
                    for name, domain in block["domains"].items()},
        "channels": {label: {"domain": meta["domain"], "native_label": meta["native_label"],
                             "semantics": meta["semantics"], "units": meta["units"]}
                     for label, meta in block["channels"].items()},
        "statistic_units": block["statistic_units"],
    }


def _split(record: Record, *, fraction: float, embargo_frames: Optional[int]
           ) -> Tuple[Record, Record]:
    return split_with_embargo(record, fraction=float(fraction),
                              embargo=None if embargo_frames is None else int(embargo_frames))


def _price(record: Record, durations: Sequence[float], *, n_surrogates: int, alpha: float,
           correction: str) -> Dict[str, Any]:
    """What the declared family costs, before anything is measured."""
    frames = physical_lags(record, durations)
    pairs = cross_domain_pairs(record)
    size = len(pairs) * len(frames)
    ceiling = affordable_member_count(int(n_surrogates), alpha=float(alpha),
                                      correction=correction)
    return {
        "lag_seconds": [float(v) for v in durations],
        "lag_frames": [int(v) for v in frames],
        "common_cadence_seconds": float(record.cadence_seconds),
        "cross_domain_pairs": ["%s>%s" % pair for pair in pairs],
        "n_pairs": len(pairs),
        "family_size": int(size),
        "n_surrogates": int(n_surrogates),
        "affordable_member_count": int(ceiling),
        "affordable": bool(size <= ceiling),
        "required_surrogates": int(required_surrogates(size, alpha=float(alpha),
                                                       method=correction)),
        "reading": ("Every ordered pair that crosses the boundary, at every declared "
                    "duration. Within-domain pairs are not members: this family exists to "
                    "test whether the abstraction crosses, and a within-domain member would "
                    "spend correction power on a question it does not ask."),
    }


def _candidates_from(record: Record, labels: Sequence[str], *, durations: Sequence[float],
                     n_surrogates: int, alpha: float, correction: str, study_id: str
                     ) -> Tuple[PrecedenceCandidate, ...]:
    """Re-derive the frozen members by re-running the generate stage, not by parsing labels.

    The sweep is deterministic - `correlation_test` draws nothing - so the training pass can
    be repeated exactly, and the members handed to the confirmatory pass are the ones this
    record's training partition actually selects. Reading them back out of the seal's labels
    would be simpler and would match itself by construction, so a later change to the
    selection rule could confirm a stale family under its old names without anything noticing.

    The refusal below is defensive rather than a client-facing gate, and saying otherwise
    would overstate it: a caller cannot reach it. An edited seal fails its own digest at load,
    and a different record produces a different held-out partition, which `confirm_on_held_out`
    refuses by identity. Both of those happen before the ledger is written, so this costs
    nothing either way.
    """
    generated = sweep_cross_domain(record, lag_seconds=durations, n_surrogates=n_surrogates,
                                   alpha=alpha, correction=correction, study_id=study_id)
    chosen = {candidate.label: candidate for candidate in choose_candidates(generated.result)}
    missing = [label for label in labels if label not in chosen]
    if missing:
        raise HTTPException(
            status_code=409,
            detail="The training partition of these records no longer selects %s. The sweep "
                   "is deterministic, so this is a different record, a different reading of "
                   "it, or a different split - not a confirmation of what was sealed. The "
                   "held-out partition has not been opened." % missing[:4])
    return tuple(chosen[label] for label in labels)


def _sealed_settings(seal: Seal) -> Dict[str, Any]:
    settings = dict(seal.confirm.get("notes", {}))
    if settings.get("surface") != SURFACE:
        raise HTTPException(
            status_code=409,
            detail="Seal %s was not frozen by the cross-domain surface, so the family it "
                   "describes cannot be reproduced here. Confirm it where it was sealed."
                   % seal.seal_sha256)
    missing = sorted(key for key in SEALED_RUN_KEYS if key not in settings)
    if missing:
        raise HTTPException(
            status_code=409,
            detail="Seal %s does not carry %s, so the confirmatory sweep it describes cannot "
                   "be reproduced from it. It was written by a different version of this "
                   "surface." % (seal.seal_sha256, missing))
    return settings


# --------------------------------------------------------------------------------- routes


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """What this surface aligns, what it refuses, and what a result is worth."""
    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "record_schema": CROSS_DOMAIN_SCHEMA,
        "steps": {
            "align": ("Intersect two native clocks exactly and report the shared record's "
                      "lineage. No value is read and nothing is stored."),
            "lags": ("Price a lag family declared in seconds against the exact common "
                     "cadence, before any of it is measured."),
            "partition": ("Split the aligned record with an embargo and show both partition "
                          "identities, including whether the held-out one is already spent."),
            "generate": ("Sweep every cross-domain direction at every declared duration on "
                         "the training partition, and select what would be carried forward."),
            "seal": ("Freeze the selected members against the held-out partition, before it "
                     "is opened. The sealing time is the server's."),
            "confirm": ("Open the held-out partition once, on the frozen family. Every "
                        "setting comes from the seal; the caller supplies only the records."),
        },
        "interpolation": "none",
        "alignment": "exact_timestamp_intersection",
        "minimum_common_observations": int(MIN_FRAMES),
        "lags_declared_in": "seconds",
        "measure": VALUE_MEASURE,
        "requires_per_channel": ["semantics", "units"],
        "records_evidence": False,
        "moves_rung": False,
        "publication": PUBLICATION_NOTE,
        "claim_boundary": CLAIM_BOUNDARY,
    }


@router.post("/align")
async def align(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    first_source: str = Form(...),
    second_source: str = Form(...),
    name: str = Form("cross-domain"),
) -> Dict[str, Any]:
    """Intersect two native clocks exactly, or explain why they cannot be intersected."""
    try:
        record, _, _ = await _both(first, second, first_source, second_source, name)
        pairs = cross_domain_pairs(record)
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "alignment": _alignment(record),
        "cross_domain_pairs": ["%s>%s" % pair for pair in pairs],
        "n_frames": record.length,
        "read_only": True,
        "stored": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


@router.post("/lags")
async def price_lags(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    first_source: str = Form(...),
    second_source: str = Form(...),
    lag_seconds: str = Form(...),
    n_surrogates: int = Form(DEFAULT_SURROGATES),
    alpha: float = Form(0.05),
    correction: str = Form("benjamini_yekutieli"),
    name: str = Form("cross-domain"),
) -> Dict[str, Any]:
    """Convert a family declared in seconds onto the common clock and price it."""
    try:
        record, _, _ = await _both(first, second, first_source, second_source, name)
        durations = _lag_seconds(lag_seconds)
        priced = _price(record, durations, n_surrogates=int(n_surrogates),
                        alpha=float(alpha), correction=str(correction))
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "alignment": _alignment(record),
        "family": priced,
        "read_only": True,
        "stored": False,
        "claim_boundary": (
            "A price, not a result. A duration below either domain's physical floor, or one "
            "that is not a whole number of common frames, is refused rather than rounded: "
            "rounding would report a lead the clock cannot express."),
    }


@router.post("/partition")
async def describe_partition(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    first_source: str = Form(...),
    second_source: str = Form(...),
    fraction: float = Form(0.55),
    embargo_frames: Optional[int] = Form(None),
    name: str = Form("cross-domain"),
) -> Dict[str, Any]:
    """Show what a seal would bind itself to, before it binds anything."""
    try:
        record, _, _ = await _both(first, second, first_source, second_source, name)
        train, held_out = _split(record, fraction=float(fraction),
                                 embargo_frames=embargo_frames)
        identity = held_out.identity()
        spent = held_out_ledger().records.get(identity.digest())
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "alignment": _alignment(record),
        "train": {"frames": list(train.frames), "n_frames": train.length,
                  "digest": train.identity().digest()},
        "held_out": {"frames": list(held_out.frames), "n_frames": held_out.length,
                     "digest": identity.digest()},
        "embargo_frames": int(held_out.provenance["embargo_frames"]),
        "recommended_embargo_frames": int(held_out.provenance["recommended_embargo_frames"]),
        "already_opened": spent is not None,
        "opened_record": spent,
        "read_only": True,
        "stored": False,
        "claim_boundary": (
            "Geometry and lineage. The embargo between the two partitions is the record's "
            "own memory, because the first frames after a cut are the immediate future of "
            "the last frames before it."),
    }


@router.post("/generate")
async def generate(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    first_source: str = Form(...),
    second_source: str = Form(...),
    lag_seconds: str = Form(...),
    study_id: str = Form(...),
    n_surrogates: int = Form(DEFAULT_SURROGATES),
    alpha: float = Form(0.05),
    correction: str = Form("benjamini_yekutieli"),
    fraction: float = Form(0.55),
    embargo_frames: Optional[int] = Form(None),
    name: str = Form("cross-domain"),
) -> Dict[str, Any]:
    """Measure the declared family on the training partition. Nothing here is a finding."""
    try:
        record, _, _ = await _both(first, second, first_source, second_source, name)
        durations = _lag_seconds(lag_seconds)
        train, _held_out = _split(record, fraction=float(fraction),
                                  embargo_frames=embargo_frames)
        swept = sweep_cross_domain(train, lag_seconds=durations,
                                   n_surrogates=int(n_surrogates), alpha=float(alpha),
                                   correction=str(correction), study_id=study_id)
        chosen = choose_candidates(swept.result)
        report = report_precedence_generation(swept.result, train=train.identity())
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "alignment": _alignment(record),
        "family": swept.describe(),
        "n_examined": int(swept.result.n_examined),
        "affordable_here": bool(swept.result.affordable_here),
        "candidates": [candidate.describe() for candidate in chosen],
        "generation": report,
        "read_only": True,
        "stored": False,
        "claim_boundary": (
            "Selection, on the partition that was swept. A member carried forward was chosen "
            "for being strong on training data, which is what the generate stage is for and "
            "is not evidence. Nothing on the held-out partition has been read."),
    }


@router.post("/seal")
async def seal_family(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    first_source: str = Form(...),
    second_source: str = Form(...),
    lag_seconds: str = Form(...),
    study_id: str = Form(...),
    n_surrogates: int = Form(DEFAULT_SURROGATES),
    alpha: float = Form(0.05),
    correction: str = Form("benjamini_yekutieli"),
    fraction: float = Form(0.55),
    embargo_frames: Optional[int] = Form(None),
    seed: int = Form(DEFAULT_SEED),
    name: str = Form("cross-domain"),
) -> Dict[str, Any]:
    """Freeze what the generate stage selected, against a partition not yet opened."""
    try:
        record, first_spec, second_spec = await _both(first, second, first_source,
                                                      second_source, name)
        durations = _lag_seconds(lag_seconds)
        train, held_out = _split(record, fraction=float(fraction),
                                 embargo_frames=embargo_frames)
        swept = sweep_cross_domain(train, lag_seconds=durations,
                                   n_surrogates=int(n_surrogates), alpha=float(alpha),
                                   correction=str(correction), study_id=study_id)
        chosen = choose_candidates(swept.result)
        # Every setting `/confirm` needs, sealed *inside* the confirmatory specification, so
        # an edit to any of them changes the seal's digest rather than quietly producing a
        # different analysis under the same seal.
        settings = {
            "surface": SURFACE,
            "first": first_spec,
            "second": second_spec,
            "name": name,
            "lag_seconds": [float(v) for v in durations],
            "fraction": float(fraction),
            "embargo_frames": (None if embargo_frames is None
                               else int(embargo_frames)),
            "seed": int(seed),
            "n_surrogates": int(n_surrogates),
            "alpha": float(alpha),
            "correction": str(correction),
        }
        seal, frozen = freeze_precedence(
            swept.result, held_out=held_out.identity(),
            sealed_at=datetime.now(timezone.utc).isoformat(),
            n_surrogates=int(n_surrogates), chosen=chosen, ledger=held_out_ledger(),
            study_id=study_id, notes=settings)
    except SpectralEarthError as error:
        raise _handle(error)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400,
                            detail="Malformed cross-domain declaration: %s" % error)

    store_seal(seal)
    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "seal_sha256": seal.seal_sha256,
        "sealed_at": seal.sealed_at,
        "sealed_at_source": ("the server clock. A caller-supplied sealing time could be "
                             "written after the partition was opened, and the time is the "
                             "whole of what a seal claims."),
        "seal": seal.to_mapping(),
        "generate_family_size": int(seal.generate_family_size),
        "confirm_family_size": int(seal.confirm_family_size),
        "confirm_labels": list(seal.confirm_labels),
        "frozen": [candidate.describe() for candidate in frozen],
        "held_out_digest": seal.held_out.digest(),
        "records_evidence": False,
        "rung_moved": False,
        "publication": PUBLICATION_NOTE,
        "claim_boundary": (
            "Nothing has been tested on held-out data. A family of %d is affordable at %d "
            "surrogates where the generated family of %d is not, and that is the only thing "
            "this exchange bought."
            % (int(seal.confirm_family_size), int(n_surrogates),
               int(seal.generate_family_size))),
    }


@router.post("/seals/{seal_sha256}/confirm")
async def confirm(
    seal_sha256: str,
    first: UploadFile = File(...),
    second: UploadFile = File(...),
    published_sha256: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Open the held-out partition once, on the family frozen before it was opened."""
    seal = load_seal(seal_sha256)
    settings = _sealed_settings(seal)
    # Before anything is read, and long before the partition is opened. A published digest
    # that disagrees with the seal is a refusal, and a refusal that had already spent the
    # held-out data would have cost exactly what it was declining to authorise.
    if published_sha256 is not None:
        try:
            seal.verify_published(published_sha256)
        except SpectralEarthError as error:
            raise _handle(error)
    try:
        first_text = _decode(await first.read(), first.filename or "first")
        second_text = _decode(await second.read(), second.filename or "second")
        one = _series(first_text, first.filename or "first", settings["first"])
        two = _series(second_text, second.filename or "second", settings["second"])
        record = align_exact(one, two, name=str(settings["name"]))
        train, held_out = _split(record, fraction=float(settings["fraction"]),
                                 embargo_frames=settings["embargo_frames"])
        # Re-derive the frozen members from the training partition before the held-out one is
        # touched. A mismatch here is a design or upload error and must cost nothing.
        chosen = _candidates_from(
            train, list(seal.confirm_labels), durations=settings["lag_seconds"],
            n_surrogates=int(settings["n_surrogates"]), alpha=float(settings["alpha"]),
            correction=str(settings["correction"]), study_id=seal.study_id)
        receipt = confirm_cross_domain(
            seal, chosen, record=held_out, held_out=held_out.identity(),
            ledger=held_out_ledger(),
            opened_at=datetime.now(timezone.utc).isoformat(),
            seed=int(settings["seed"]))
    except SpectralEarthError as error:
        raise _handle(error)

    return {
        "schema": CROSS_DOMAIN_SCHEMA_HTTP,
        "seal_sha256": seal.seal_sha256,
        "checked_against_publication": published_sha256 is not None,
        "alignment": _alignment(record),
        "receipt": receipt,
        "confirmed_labels": list(receipt["rejected_labels"]),
        "records_evidence": False,
        "rung_moved": False,
        "publication": (PUBLICATION_NOTE if not published_sha256 else
                        "Checked against the published digest supplied with this request."),
        "claim_boundary": (
            "%s The held-out partition is now spent: this surface will refuse every further "
            "confirmation against it, under this seal or any other." % CLAIM_BOUNDARY),
    }


__all__ = ["router", "CROSS_DOMAIN_SCHEMA_HTTP", "SURFACE", "SOURCE_KEYS", "SEALED_RUN_KEYS"]
