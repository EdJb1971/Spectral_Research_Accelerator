"""Structure mining over HTTP: scenes, families, seals and transfer (TG11.4, rules R18/R20).

Six core modules - `motif`, `constellation`, `family`, `invariance`, `motif_freeze` and
`motif_transfer` - are 4,211 lines that nothing outside the test suite could reach.  This is
the boundary that reaches them, and it implements no matcher, no null, no correction and no
p-value.  Every number below is computed by the module that owns it.

**The client cannot draw a motif.**  This is the R22-shaped risk of this slice, and it is not
about rungs.  A motif is a configuration of extracted features, so a surface that accepted
feature coordinates would let a caller type the shape it wanted to find and then have the
programme confirm it.  So no route on this surface accepts a feature.  A caller uploads a
*field* - frames of numbers - and the server extracts, every time, with the settings the
record was admitted under.  Scenes are addressed by the digest of the bytes and the
declaration they are read under, and a caller who wants different features has to present
different data.

**The client cannot choose what counts as the same shape either.**  The match tolerance
decides which configurations are repeats, and a wrong one is not a subtle error: this tree's
own benchmark measured a tolerance built the tempting way at 0.41 against a correct 0.0083,
fifty times too wide, at which point every triangle matched every other.  `/generate`
therefore accepts a *calibrated* tolerance by digest and no bare number.  Calibration is
`invariance.calibrate_match_tolerance` over frames the caller declares to be replicates of
one configuration - a claim about the data that the server cannot check, and the receipt says
so in as many words.

**Frames with different feature counts are refused, not trimmed.**  `motif._prepare` requires
every scene to hold the same number of features because the family is `scenes x C(n, k)` and
scenes of different `n` have no such number.  The tempting repair - take the brightest six -
is refused here rather than offered as an option: magnitude ordering moves between noise
realisations, so "the brightest six" is a different configuration in each frame, and a
tolerance calibrated across frames that disagree about *which* features they contain measures
that disagreement.

**Nothing measured inside a held-out frame leaves this surface before it is spent.**
`/records` reports how many features each frame yielded and what the extractor rejected, which
is admission geometry - exactly what a `PartitionIdentity` has always carried about a channel
table, namely how many channels it has and what they are called.  Nothing measured *from* a
frame is reported: not a position, not a scale, not an amplitude, and not the calibrated
threshold, because the split is declared after admission and a per-frame measurement would
have been on screen while the split was being chosen.  The residual is stated rather than
closed: feature counts must agree across frames anyway or the record is refused, so the counts
carry almost nothing, but a caller does admit first and split afterwards, and the held-out
ledger records what each split spent rather than refusing a second one.

**The confirmatory run is read out of the seal, not out of the request.**  TG11.2 established
that for lag families; this extends it to mining.  `/confirm` takes a seal digest and nothing
else that could change the analysis: the record, the split, the configuration size, the
matcher, the tolerance, the ensemble, the correction and the seed are all sealed as notes on
the confirmatory specification, so editing any of them breaks the seal's digest rather than
quietly producing a different analysis under the same name.  The candidates are re-derived by
re-running the same deterministic mining pass over the same training frames, and
`confirm_motifs` checks the labels it is handed against the labels the seal froze - so a
redefined motif is refused rather than confirmed under an old name.

Seals live in TG11.2's store and the held-out ledger is TG11.2's ledger, because "this
partition has been opened" is one fact about the programme and not one per surface.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Extra

from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT, refuse_bare_confidence
from src.api.preregistration import held_out_ledger, load_seal, store_seal
from src.core.builtin_domains import register_builtin_domains
from src.core.constellation import RELATIONS
from src.core.domain import DomainDeclaration, declaration_for
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.extraction import EXTRACTORS, ExtractionField, extract
from src.core.family import (SearchAxis, SearchSpecification, SearchTerm,
                             max_affordable_family)
from src.statistics.multiple_comparisons import required_surrogates
from src.core.invariance import (MATCHERS, Presentation, TRANSFORMS,
                                 audit_declared_invariance, calibrate_match_tolerance,
                                 declared_invariance)
from src.core.motif import (CONFIGURATION_COMBINATORS, DEFAULT_MATCHER, Scene,
                            affordable_candidate_count, choose_candidates, confirm_motifs,
                            freeze_motifs, mine, report_motif_generation)
from src.core.motif_freeze import (FrozenMotif, freeze_motif, load_frozen_motif,
                                   save_frozen_motif)
from src.core.motif_transfer import TargetAlreadyOpenedError, TransferLedger, blind_transfer
from src.core.preregistration import (HeldOutAlreadyOpenedError, PartitionIdentity,
                                      PartitionMismatchError, Seal)


router = APIRouter(prefix="/api/v1/mining", tags=["mining"])

#: Where admitted fields, published motif definitions and the transfer ledger live.
#: Overridable so a test never publishes into a researcher's store, and never spends a
#: transfer target that a researcher had not spent.
ROOT_ENV = "SPECTRAL_MINING_ROOT"
DEFAULT_ROOT = Path("data") / "mining"

TRANSFER_LEDGER_NAME = "transfer_ledger.json"
MINING_SCHEMA_HTTP = "spectral.mining.http.v1"

#: A record is addressed by the digest of its bytes *and* of the declaration they are read
#: under, because the features are a function of both and a scene is what this surface
#: serves.  Re-reading the same array as a different variable is a different record.
DIGEST = re.compile(r"^[0-9a-f]{64}$")

#: 3-D `.npy` only, and never a pickle.  `allow_pickle=False` is not a convenience: an object
#: array in a `.npy` file is arbitrary code at load time, and this route loads whatever it is
#: given.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 512

#: The settings sealed as notes on the confirmatory specification, so that `/confirm` can be
#: driven from the seal alone and an edit to any of them breaks the seal digest.
SEALED_MINING_KEYS: Tuple[str, ...] = (
    "record_id", "size", "matcher", "tolerance", "tolerance_sha256", "train_ratio",
    "embargo_frames", "n_surrogates", "alpha", "correction", "seed",
)

DEFAULT_SEED = 20260827
DEFAULT_SURROGATES = 199
DEFAULT_CORRECTION = "benjamini_yekutieli"

TOLERANCE_NOTE = (
    "A match tolerance is a measurement of the pipeline's noise, not a parameter of the "
    "matcher, so it is calibrated here from frames the caller declares to be replicates of "
    "one configuration under different noise. That declaration is the caller's and cannot be "
    "checked from the numbers: frames of unrelated configurations calibrated as replicates "
    "produce a tolerance wide enough to match everything, and it would look like a "
    "measurement.")

SEAL_PUBLICATION_NOTE = (
    "This seal is a local copy and is not evidence about itself. Publish `seal_sha256` "
    "somewhere you cannot rewrite before the held-out frames are opened, and pass it back as "
    "`published_sha256` when you confirm; that comparison is the only check with weight.")

# Eager, for the reason D35 gave: a response must not depend on which route was visited first.
REGISTERED_DOMAINS = register_builtin_domains()


# --------------------------------------------------------------------------------- storage


def _root() -> Path:
    return Path(os.environ.get(ROOT_ENV, str(DEFAULT_ROOT)))


def _records_root() -> Path:
    return _root() / "records"


def _motifs_root() -> Path:
    return _root() / "motifs"


def _tolerances_root() -> Path:
    return _root() / "tolerances"


def _transfer_ledger() -> TransferLedger:
    """The transfer ledger, on disk.

    `TransferLedger` writes atomically and re-reads before committing, which is stronger than
    the held-out ledger's read-modify-write - but it is still not a lock, and two processes
    committing two openings of one target concurrently can still both believe they were
    first.  A multi-worker deployment needs a real store.
    """
    return TransferLedger(_root() / TRANSFER_LEDGER_NAME)


def _digest_of(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


#: The last instant this process issued.  See `_now`.
_LAST_MOMENT: List[datetime] = []


def _now() -> str:
    """The server's clock, and never a field on a request: a back-dated seal is a false one.

    Strictly increasing within the process, which is not a cosmetic detail.  A transfer
    record is refused unless `frozen_at < bound_at < opened_at`, and that ordering is the
    whole content of the record - it is what says the definition was fixed before the target
    was opened.  A Windows clock ticks about every 15 ms, so two events that really did
    happen in that order can be issued the same timestamp, and the ledger would then refuse a
    correctly ordered transfer for a reason that is about the clock rather than about the
    science.  So an instant that would repeat or go backwards is advanced by a microsecond:
    the ordering reported is the ordering that happened, at a resolution the clock does not
    have.  Nothing here waits, and nothing is back-dated.
    """
    moment = datetime.now(timezone.utc)
    if _LAST_MOMENT and moment <= _LAST_MOMENT[0]:
        moment = _LAST_MOMENT[0] + timedelta(microseconds=1)
    _LAST_MOMENT[:] = [moment]
    return moment.isoformat()


def _identifier(value: str, what: str) -> str:
    if not isinstance(value, str) or not DIGEST.match(value.strip().lower()):
        raise HTTPException(
            status_code=404,
            detail="%r is not a %s. Everything this surface stores is addressed by the "
                   "64-character SHA-256 of its own content." % (value, what))
    return value.strip().lower()


def _record_path(record_id: str, suffix: str) -> Path:
    return _records_root() / ("%s%s" % (record_id, suffix))


# ------------------------------------------------------------------------------- the field


def _space_axes(declaration: DomainDeclaration) -> Tuple[Any, ...]:
    """The declared spatial axes of a domain, in declaration order (standard E14).

    Taken from the domain rather than from the request.  A caller who could name the axes
    could declare a domain's grid to be something the domain does not claim, and every
    separation measured afterwards would be in units nobody declared.
    """
    axes = tuple(axis for axis in declaration.axes if axis.role == "space")
    ordered = sorted(enumerate(axes),
                     key=lambda pair: (pair[1].ordinal if pair[1].ordinal is not None
                                       else pair[0]))
    return tuple(axis for _index, axis in ordered)


def _require_minable_domain(name: str) -> DomainDeclaration:
    declaration = declaration_for(name)          # UnknownNameError, with a did-you-mean
    axes = _space_axes(declaration)
    if len(axes) != 2:
        raise InvalidParameterError(
            "domain", name,
            "a domain declaring exactly two spatial axes, and %r declares %d (%s). A motif "
            "is a configuration in space; a domain with no spatial extent has no "
            "configuration to recur, and the registered extractor declares dimensions=2"
            % (name, len(axes), ", ".join(axis.name for axis in axes) or "none"))
    return declaration


def _load_array(payload: bytes, filename: str) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="%r is empty." % filename)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="%r is %.1f MB; this surface admits at most %d MB in one record."
                   % (filename, len(payload) / 1e6, MAX_UPLOAD_BYTES // (1024 * 1024)))
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:                                    # numpy raises broadly here
        raise HTTPException(
            status_code=400,
            detail="%r is not a NumPy `.npy` array this surface can read (%s). Object arrays "
                   "are refused rather than loaded: a pickled array is executable content, "
                   "and this route loads whatever it is handed." % (filename, exc))
    if values.ndim != 3:
        raise HTTPException(
            status_code=400,
            detail="%r has shape %s. A mining record is a stack of frames: (frame, %s). A "
                   "single frame is not minable, because recurrence is a statement about "
                   "more than one scene."
                   % (filename, tuple(int(n) for n in values.shape), "row, column"))
    if values.shape[0] > MAX_FRAMES:
        raise HTTPException(
            status_code=413,
            detail="%r holds %d frames; this surface admits at most %d in one record."
                   % (filename, int(values.shape[0]), MAX_FRAMES))
    return np.asarray(values, dtype=np.float64)


def _extraction_settings(raw: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {"extractor", "alpha", "n_surrogates", "surrogate_method", "seed",
               "window_scales", "suppression_scales", "boundary_margin", "max_features",
               "refinements"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise InvalidParameterError(
            "extraction keys", unknown,
            "only %s. An ignored extraction setting would make the record's digest describe "
            "features that were found some other way" % sorted(allowed))
    settings = {
        "extractor": str(raw.get("extractor", "local_maximum")),
        "alpha": float(raw.get("alpha", 0.05)),
        "n_surrogates": int(raw.get("n_surrogates", 199)),
        "surrogate_method": str(raw.get("surrogate_method", "phase_randomise")),
        "seed": int(raw.get("seed", DEFAULT_SEED)),
    }
    params = {key: raw[key] for key in
              ("window_scales", "suppression_scales", "boundary_margin", "max_features",
               "refinements") if key in raw}
    EXTRACTORS.entry(settings["extractor"])      # UnknownNameError, with a did-you-mean
    settings["params"] = params
    return settings


def _store_record(payload: bytes, declaration: Mapping[str, Any]) -> str:
    """Admit one field, addressed by the digest of its bytes and its declaration.

    The sidecar is not trusted on the way back in: `_load_record` recomputes this digest and
    refuses a record whose declaration was edited under a name that was derived from it.
    Otherwise a stored record could be re-declared as another variable of another domain
    after it had been mined, and every receipt naming it would silently change meaning.
    """
    record_id = _digest_of(dict(declaration))
    root = _records_root()
    root.mkdir(parents=True, exist_ok=True)
    array_path = _record_path(record_id, ".npy")
    if not array_path.exists():
        with array_path.open("wb") as handle:
            handle.write(payload)
        _record_path(record_id, ".json").write_text(
            json.dumps(dict(declaration), indent=2, sort_keys=True), encoding="utf-8")
    return record_id


def _load_record(record_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    record_id = _identifier(record_id, "record digest")
    array_path, meta_path = _record_path(record_id, ".npy"), _record_path(record_id, ".json")
    if not array_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No record %s is admitted here. A mining pass runs over a field this "
                   "server extracted, not over features a client supplied, so the field has "
                   "to be admitted first (POST /api/v1/mining/records)." % record_id)
    declaration = json.loads(meta_path.read_text(encoding="utf-8"))
    recomputed = _digest_of(declaration)
    if recomputed != record_id:
        raise HTTPException(
            status_code=409,
            detail="Record %s no longer hashes to its own declaration (%s). The stored "
                   "declaration was edited after the record was admitted, and every receipt "
                   "naming this record described the declaration it was admitted under."
                   % (record_id, recomputed))
    payload = array_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != declaration["content_sha256"]:
        raise HTTPException(
            status_code=409,
            detail="The bytes stored for record %s are not the bytes it was admitted with."
                   % record_id)
    return _load_array(payload, "record %s" % record_id), declaration


def _stored_records() -> List[Dict[str, Any]]:
    root = _records_root()
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        declaration = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"record_id": path.stem, "declaration": declaration})
    return rows


# ------------------------------------------------------------------------------ the scenes

#: Extraction is a pure function of an immutable record, so its result is cached per process.
#: Not an optimisation for its own sake: `/confirm` re-derives the training candidates in
#: order to check them against the seal, and paying for the calibration ensemble three times
#: over would push a caller towards a smaller one.
_SCENES: Dict[str, Tuple[Tuple[Scene, ...], Tuple[Dict[str, Any], ...]]] = {}


def _scenes_for(record_id: str) -> Tuple[Tuple[Scene, ...], Tuple[Dict[str, Any], ...]]:
    """Every frame of one record, extracted under the settings it was admitted with."""
    if record_id in _SCENES:
        return _SCENES[record_id]
    values, declaration = _load_record(record_id)
    domain = _require_minable_domain(declaration["domain"])
    axes = _space_axes(domain)
    settings = declaration["extraction"]
    times = declaration["times"]
    scenes: List[Scene] = []
    reports: List[Dict[str, Any]] = []
    for index in range(int(values.shape[0])):
        field = ExtractionField(
            values=values[index], axes=axes, domain=domain.name,
            dataset=declaration["dataset"], variable=declaration["variable"],
            units=declaration["units"], time=float(times[index]),
            time_units=declaration["time_units"],
            representation=declaration["representation"],
            provenance={"record_id": record_id, "frame": index,
                        "content_sha256": declaration["content_sha256"]})
        result = extract(field, settings["extractor"], alpha=settings["alpha"],
                         surrogate_method=settings["surrogate_method"],
                         n_surrogates=settings["n_surrogates"], seed=settings["seed"],
                         **settings["params"])
        scenes.append(Scene(_scene_name(index), tuple(result.features)))
        # Counts and rejection tallies only. The calibrated threshold is deliberately not
        # reported per frame: the split is declared *after* admission, so anything measured
        # inside a frame that later turns out to be held-out would have been visible while
        # the split was being chosen. A count of what was found is admission geometry; a
        # noise floor measured from the frame's own spectrum is a measurement of it.
        reports.append({"frame": index, "scene": _scene_name(index),
                        "n_features": result.found, "rejected": dict(result.rejected)})
    _SCENES[record_id] = (tuple(scenes), tuple(reports))
    return _SCENES[record_id]


def _scene_name(index: int) -> str:
    return "frame%04d" % index


def _window(scenes: Sequence[Scene], start: int, stop: int, *, what: str) -> Tuple[Scene, ...]:
    chosen = tuple(scenes[start:stop])
    counts = sorted({len(scene) for scene in chosen})
    if len(counts) != 1:
        raise InvalidParameterError(
            "%s frames" % what, counts,
            "the same number of features in every frame; frames %d-%d yielded %s. This is "
            "refused rather than repaired: the declared family is scenes times C(n, k) and "
            "frames of different n have no such number, and the obvious repair - keep the "
            "brightest few - is a different configuration in every frame, because magnitude "
            "ordering moves between noise realisations. Change the extraction settings the "
            "record is admitted under, or admit frames that hold the same configuration"
            % (start, stop, counts))
    if counts[0] < 3:
        raise InvalidParameterError(
            "%s frames" % what, counts[0],
            "at least three features per frame. A configuration is a ratio between "
            "separations, and two features give one separation to take a ratio of")
    return chosen


def _split(n_frames: int, train_ratio: float, embargo_frames: int) -> Dict[str, Tuple[int, int]]:
    """The same split rule `split_channel_series` applies to rows, applied to frames (R6).

    Deliberately the same arithmetic rather than a similar one, and a test asserts the two
    agree frame for frame.  The embargo is discarded, not moved: the frame after a boundary
    is nearly a copy of the frame before it, and a training configuration whose neighbour sits
    in the test window has shown the test its own answer.
    """
    if not 0.0 < train_ratio < 1.0:
        raise InvalidParameterError("train_ratio", train_ratio, "a fraction in (0, 1)")
    if embargo_frames < 0:
        raise InvalidParameterError("embargo_frames", embargo_frames, "a non-negative count")
    train_stop = int(n_frames * train_ratio)
    test_start = train_stop + int(embargo_frames)
    if train_stop < 2 or (n_frames - test_start) < 2:
        raise InvalidParameterError(
            "train_ratio", train_ratio,
            "a split leaving at least two frames on each side of a %d-frame embargo; %d "
            "frames give train=%d and held-out=%d. Two is the floor mining needs, not a "
            "recommendation: recurrence across two scenes is the weakest statement the word "
            "supports" % (embargo_frames, n_frames, train_stop, max(n_frames - test_start, 0)))
    return {"train": (0, train_stop), "embargo": (train_stop, test_start),
            "held_out": (test_start, n_frames)}


def _identity(declaration: Mapping[str, Any], *, name: str, split: str,
              window: Tuple[int, int], n_features: int, train_ratio: float,
              embargo_frames: int) -> PartitionIdentity:
    """What a seal binds itself to, built from what identifies the *data* (D65's rule).

    The record digest, the frame window and the split that produced it - never the filename
    the frames arrived under.  The same held-out frames re-uploaded as another file must hash
    to the same partition, or the ledger's "once" does not fire.

    `n_features` is geometry, not a measurement: it is the count of things found in each
    frame, exactly as a channel table's identity carries how many channels it has and what
    they are called.  No position, scale or amplitude from a held-out frame appears here.
    """
    return PartitionIdentity(
        name=name, n_times=int(window[1] - window[0]), n_channels=int(n_features),
        channel_labels=tuple("feature%d" % i for i in range(int(n_features))),
        frames=(int(window[0]), int(window[1])),
        provenance={"content_sha256": declaration["content_sha256"],
                    "record_id": declaration["record_id"],
                    "domain": declaration["domain"],
                    "split": split, "split_frames": [int(window[0]), int(window[1])],
                    "split_train_ratio": float(train_ratio),
                    "split_embargo_frames": int(embargo_frames)})


# --------------------------------------------------------------------------- the tolerance


def _tolerance_path(digest: str) -> Path:
    return _tolerances_root() / ("%s.json" % digest)


def _store_tolerance(receipt: Mapping[str, Any]) -> str:
    digest = _digest_of(dict(receipt))
    root = _tolerances_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _tolerance_path(digest)
    if not path.exists():
        path.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True), encoding="utf-8")
    return digest


def _load_tolerance(digest: str) -> Dict[str, Any]:
    digest = _identifier(digest, "tolerance digest")
    path = _tolerance_path(digest)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No calibrated tolerance %s is stored here. `/generate` takes a tolerance "
                   "by digest and not as a number: the tolerance decides which configurations "
                   "are repeats, and one chosen rather than measured is where a mining pass "
                   "that finds a motif in anything comes from (POST /api/v1/mining/tolerance)."
                   % digest)
    stored = json.loads(path.read_text(encoding="utf-8"))
    if _digest_of(stored) != digest:
        raise HTTPException(
            status_code=409,
            detail="Tolerance %s no longer hashes to its own receipt." % digest)
    return stored


# ------------------------------------------------------------------------------- refusals


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    # A spent partition and a spent transfer target are state conflicts, not malformed
    # requests: nothing the caller could edit would make either call legitimate again.
    if isinstance(error, (HeldOutAlreadyOpenedError, TargetAlreadyOpenedError,
                          PartitionMismatchError)):
        return HTTPException(status_code=409, detail=info["detail"])
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


def _respond(payload: Dict[str, Any], *, where: str) -> Dict[str, Any]:
    return refuse_bare_confidence(payload, where=where)


# ---------------------------------------------------------------------------- request bodies


class _Body(BaseModel):
    """Unknown fields are refused rather than ignored.

    An ignored setting on this surface would be a family member, a tolerance or a split that
    the receipt does not describe, and the receipt is the only thing a reader has.
    """

    class Config:
        extra = Extra.forbid


class PriceRequest(_Body):
    n_scenes: int
    n_features: int
    size: int = 3
    n_surrogates: int = DEFAULT_SURROGATES
    alpha: float = 0.05
    correction: str = DEFAULT_CORRECTION


class ToleranceRequest(_Body):
    record_id: str
    frames: List[int]
    matcher: str = DEFAULT_MATCHER
    declared_as_replicates_of: str


class MiningRequest(_Body):
    record_id: str
    tolerance_sha256: str
    size: int = 3
    matcher: str = DEFAULT_MATCHER
    n_surrogates: int = DEFAULT_SURROGATES
    alpha: float = 0.05
    correction: str = DEFAULT_CORRECTION
    seed: int = DEFAULT_SEED
    train_ratio: float = 0.5
    embargo_frames: int = 0
    study_id: str = ""


class FreezeRequest(MiningRequest):
    n_candidates: Optional[int] = None


class ConfirmRequest(_Body):
    seal_sha256: str
    published_sha256: Optional[str] = None


class PublishRequest(_Body):
    seal_sha256: str
    label: str


class TransferRequest(_Body):
    motif_sha256: str
    published_sha256: str
    target_record_id: str
    target_domain: str


class InvarianceRequest(_Body):
    record_id: str
    reference_frame: int
    replicate_frames: List[int]
    presentations: List[Dict[str, Any]]
    alpha: float = 0.05


# ----------------------------------------------------------------------------------- routes


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """What is reachable, and what none of it is evidence of."""
    return _respond({
        "schema": MINING_SCHEMA_HTTP,
        "sizes": {str(size): combinator
                  for size, combinator in sorted(CONFIGURATION_COMBINATORS.items())},
        "matchers": [{"name": entry.name, "description": entry.description,
                      "declared_invariance": list(declared_invariance(entry.name)),
                      "capabilities": dict(entry.capabilities)}
                     for entry in MATCHERS.entries()],
        "relations": [{"name": entry.name, "description": entry.description,
                       "capabilities": dict(entry.capabilities)}
                      for entry in RELATIONS.entries()],
        "extractors": [{"name": entry.name, "description": entry.description,
                        "capabilities": dict(entry.capabilities),
                        "params": dict(entry.params or {})}
                       for entry in EXTRACTORS.entries()],
        "transforms": list(TRANSFORMS),
        "minable_domains": sorted(
            name for name in REGISTERED_DOMAINS
            if len(_space_axes(declaration_for(name))) == 2),
        "input": ("a stack of frames as a 3-D NumPy `.npy` array, declared under a "
                  "registered domain. Features are extracted here; no route accepts one."),
        "tolerance": TOLERANCE_NOTE,
        "stores": ["admitted fields", "calibrated tolerances", "published motif definitions",
                   "the transfer ledger"],
        "moves_rung": False,
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
        "claim_boundary": (
            "Mining produces candidates. Support measured on the training frames is "
            "selection and not evidence, because every exemplar is one of the occurrences it "
            "is counted among and it was ranked highly for having been counted often. The "
            "only output of the generate stage is a confirmatory family to freeze; the only "
            "output of this surface that bears on a claim is a confirmation receipt, and "
            "recording one is TG11.3's write path, not this one (R22)."),
    }, where="mining capabilities")


@router.post("/records")
async def admit_record(
    file: UploadFile = File(...),
    domain: str = Form(...),
    dataset: str = Form(...),
    variable: str = Form(...),
    representation: str = Form(...),
    units: str = Form(""),
    time_units: str = Form("frames"),
    times: str = Form(""),
    extraction: str = Form("{}"),
) -> Dict[str, Any]:
    """Admit one stack of frames and report what the extractor found in each.

    Extraction happens here, once, under settings that become part of the record's identity.
    That is what makes every later route a function of a digest rather than of a request: a
    caller who wants other features has to admit other data or other settings, and either way
    the receipts say which.
    """
    try:
        payload = await file.read()
        declaration_of = _require_minable_domain(domain)
        values = _load_array(payload, file.filename or "upload.npy")
        n_frames = int(values.shape[0])
        settings = _extraction_settings(_json_object(extraction, "extraction"))
        clock = _times(times, n_frames)
        declaration = {
            "record_id": "",
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "domain": declaration_of.name,
            "dataset": str(dataset), "variable": str(variable),
            "representation": str(representation),
            "units": str(units) or None,
            "time_units": str(time_units) or None,
            "times": list(clock),
            "shape": [int(n) for n in values.shape],
            "extraction": settings,
        }
        # The record's own id cannot be inside the value it hashes, so it is hashed without
        # it and written back afterwards -- and `_load_record` recomputes it the same way.
        identity = _digest_of(declaration)
        declaration["record_id"] = identity
        record_id = _store_record(payload, declaration)
        scenes, reports = _scenes_for(record_id)
        counts = sorted({len(scene) for scene in scenes})
        return _respond({
            "record_id": record_id,
            "declaration": declaration,
            "n_frames": n_frames,
            "frames": list(reports),
            "feature_counts": counts,
            "minable": len(counts) == 1 and counts[0] >= 3,
            "why_not_minable": (None if len(counts) == 1 and counts[0] >= 3 else
                                ("Mining prices a family of scenes times C(n, k) and needs "
                                 "one n. These frames yielded %s. Nothing here trims a frame "
                                 "to make the counts agree." % counts)),
            "note": ("Feature counts are admission geometry - the same thing a channel "
                     "table's identity carries about its channels. No position, scale or "
                     "amplitude measured inside a held-out frame is served by any route on "
                     "this surface until a seal over it is confirmed."),
            "claim_boundary": ("An admitted field is data this server can extract features "
                               "from. It is not a finding, and the extractor's shape model - "
                               "%s - is an assumption about what a feature looks like."
                               % EXTRACTORS.entry(settings["extractor"]).capabilities.get(
                                   "shape_model", "unstated")),
        }, where="admitted record")
    except SpectralEarthError as error:
        raise _handle(error)


@router.get("/records")
async def list_records() -> Dict[str, Any]:
    """Every admitted field, by digest.  Reads no array and extracts nothing."""
    return _respond({"records": _stored_records(),
                     "root": str(_records_root()),
                     "note": ("Admitted fields are programme state, not a cache: a seal, a "
                              "frozen motif and a confirmation receipt all name a record "
                              "digest, and deleting the record makes those receipts "
                              "uncheckable.")},
                    where="records")


@router.post("/price")
async def price(request: PriceRequest) -> Dict[str, Any]:
    """What a mining pass over this shape would cost, computed before anything is mined.

    The point of exposing this is that the refusal is arithmetic and can be seen in advance:
    six scenes of six features at k=3 is 120 members and needs about 12,885 surrogates before
    one of them could be rejected.  That is why the generate/confirm split is the only
    affordable shape here rather than an optimisation, and a caller can see it without
    spending anything.
    """
    try:
        return _respond(_price_account(request), where="price")
    except SpectralEarthError as error:
        raise _handle(error)


def _price_account(request: PriceRequest) -> Dict[str, Any]:
    """The family a mining pass over this shape would declare, priced without any data.

    Built here rather than through `motif_search_specification` because that function takes
    scenes, and the whole point of this route is to answer the question before a field has
    been admitted. A test asserts the two agree on family size for the same shape.
    """
    if int(request.size) not in CONFIGURATION_COMBINATORS:
        raise InvalidParameterError(
            "size", request.size,
            "a configuration size this surface can price: %s"
            % sorted(CONFIGURATION_COMBINATORS))
    if int(request.n_scenes) < 2:
        raise InvalidParameterError(
            "n_scenes", request.n_scenes,
            "at least two scenes. Recurrence is a statement about more than one scene")
    if int(request.n_features) < int(request.size):
        raise InvalidParameterError(
            "n_features", request.n_features,
            "at least the %d features a configuration of that size needs" % request.size)
    specification = SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis(
            "scene", tuple(_scene_name(i) for i in range(int(request.n_scenes)))),)),
               SearchTerm(CONFIGURATION_COMBINATORS[int(request.size)],
                          (SearchAxis("feature", tuple(range(int(request.n_features)))),))),
        n_surrogates=int(request.n_surrogates), alpha=float(request.alpha),
        correction=request.correction,
        label_format="{0}|" + "-".join("{%d}" % (i + 1) for i in range(int(request.size))),
        notes={"stage": "generate", "priced_without_data": True})
    account = specification.account()
    ceiling = max_affordable_family(int(request.n_surrogates), float(request.alpha),
                                    request.correction)
    return {
        "generate": account.describe(),
        "confirmatory_ceiling": {
            "max_affordable_family": int(ceiling),
            "surrogates_for_one_member": int(required_surrogates(
                1, float(request.alpha), request.correction)),
            "reading": ("at %d surrogates a confirmatory family of at most %d members can "
                        "still have one of them rejected after %s correction at alpha %g"
                        % (request.n_surrogates, ceiling, request.correction, request.alpha)),
        },
        "split_is_not_optional": not account.affordable,
        "claim_boundary": ("An accounting statement about a family nobody has run. "
                           "Affordability is not evidence, and an affordable family is not "
                           "a good one."),
    }


@router.post("/tolerance")
async def calibrate(request: ToleranceRequest) -> Dict[str, Any]:
    """Measure the noise floor from frames the caller declares to be replicates.

    The declaration is the caller's and the server cannot check it, which is stated in the
    receipt rather than assumed away.  What the server can do is refuse to accept a number
    instead of a measurement, and it does: `/generate` takes this receipt's digest.
    """
    try:
        scenes, _reports = _scenes_for(request.record_id)
        frames = _frames(request.frames, len(scenes), what="replicate_frames")
        if len(frames) < 2:
            raise InvalidParameterError(
                "replicate_frames", list(frames),
                "at least two replicate frames. One realisation agrees with itself exactly, "
                "and a tolerance derived that way is a statement about having measured once")
        chosen = [scenes[index] for index in frames]
        counts = sorted({len(scene) for scene in chosen})
        if len(counts) != 1:
            raise InvalidParameterError(
                "replicate_frames", counts,
                "replicates holding the same number of features; these hold %s. Two frames "
                "that disagree about which features they contain measure that disagreement, "
                "not the noise" % counts)
        tolerance = calibrate_match_tolerance(
            [scene.features for scene in chosen], matcher=request.matcher)
        receipt = {
            "schema": MINING_SCHEMA_HTTP,
            "record_id": _identifier(request.record_id, "record digest"),
            "frames": list(frames),
            "matcher": request.matcher,
            "declared_as_replicates_of": str(request.declared_as_replicates_of),
            "tolerance": tolerance.describe(),
            "calibrated_at": _now(),
        }
        digest = _store_tolerance(receipt)
        return _respond({
            "tolerance_sha256": digest, "receipt": receipt,
            "note": TOLERANCE_NOTE,
            "claim_boundary": (
                "A floor, and honestly so: more replicates can only widen it, so %d "
                "replicates is a thin estimate biased in the direction that finds no motif. "
                "It is not how invariance is decided - thresholding a dozen presentations "
                "at this value would reject a genuinely invariant matcher about one time in "
                "five (POST /api/v1/mining/invariance)."
                % tolerance.n_replicates),
        }, where="tolerance")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/generate")
async def generate(request: MiningRequest) -> Dict[str, Any]:
    """Mine the training frames.  Produces candidates and explicitly no claims."""
    try:
        run = _mining_run(request)
        report = report_motif_generation(run["result"], train=run["train_identity"])
        return _respond({
            "record_id": run["record_id"],
            "split": {name: list(window) for name, window in run["split"].items()},
            "train": run["train_identity"].to_mapping(),
            "held_out_identity": run["held_out_identity"].to_mapping(),
            "held_out_spent": run["held_out_spent"],
            "tolerance": run["tolerance_receipt"],
            "mining": run["result"].describe(),
            "generation_report": report,
            "chosen": [candidate.describe() for candidate in run["chosen"]],
            "confirmatory_ceiling": affordable_candidate_count(
                int(request.n_surrogates), float(request.alpha), request.correction),
            "next": ("POST /api/v1/mining/freeze with the same body freezes these candidates "
                     "against the held-out frames named above, before they are opened."),
            "claim_boundary": (
                "Nothing here is a finding. The exemplar of a candidate is one of the "
                "occurrences it is counted among, so its support starts at one by "
                "construction, and it is at the top of this ranking because its support was "
                "high - both are selection, performed on the partition that was mined."),
        }, where="generate")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/freeze")
async def freeze(request: FreezeRequest) -> Dict[str, Any]:
    """Freeze the confirmatory family against the held-out frames, before they are opened."""
    try:
        run = _mining_run(request)
        notes = dict({key: run["sealed"][key] for key in SEALED_MINING_KEYS},
                     stage_source=MINING_SCHEMA_HTTP)
        seal, frozen = freeze_motifs(
            run["result"], held_out=run["held_out_identity"], sealed_at=_now(),
            n_surrogates=int(request.n_surrogates),
            chosen=(None if request.n_candidates is not None else run["chosen"]),
            n_candidates=request.n_candidates, ledger=held_out_ledger(),
            study_id=request.study_id, notes=notes)
        store_seal(seal)
        return _respond({
            "seal_sha256": seal.seal_sha256,
            "seal": seal.to_mapping(),
            "frozen_labels": list(seal.confirm_labels),
            "frozen": [candidate.describe() for candidate in frozen],
            "publication_note": SEAL_PUBLICATION_NOTE,
            "sealed_settings": {key: run["sealed"][key] for key in SEALED_MINING_KEYS},
            "next": ("POST /api/v1/mining/confirm with this seal digest. It takes no other "
                     "setting: everything the confirmatory pass needs is sealed above, so "
                     "there is no knob left to turn after the declaration."),
            "claim_boundary": (
                "A seal is a promise about ordering: this family was fixed before these "
                "frames were opened. It says nothing about whether any member is real, and "
                "it records no evidence and moves no rung (R22)."),
        }, where="freeze")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/confirm")
async def confirm(request: ConfirmRequest) -> Dict[str, Any]:
    """Open the held-out frames once, test the frozen motifs, correct at the frozen size."""
    try:
        seal = load_seal(_identifier(request.seal_sha256, "seal digest"))
        # Before anything is opened. A seal that disagrees with the digest published before
        # the held-out frames existed is not the declaration this confirmation claims to be
        # testing, and finding that out after the partition was spent would spend it for a
        # run whose result nobody may use.
        if request.published_sha256 is not None:
            seal.verify_published(request.published_sha256)
        sealed = _sealed_settings(seal)
        rerun = _rerun(seal, sealed)
        receipt = confirm_motifs(
            seal, rerun["chosen"], scenes=rerun["held_out_scenes"],
            held_out=rerun["held_out_identity"], ledger=held_out_ledger(),
            opened_at=_now(), size=int(sealed["size"]),
            tolerance=float(sealed["tolerance"]), seed=int(sealed["seed"]),
            matcher=str(sealed["matcher"]))
        return _respond({
            "receipt": receipt,
            "sealed_settings": sealed,
            "published_sha256": request.published_sha256,
            "publication_note": SEAL_PUBLICATION_NOTE,
            "vacuous": list(receipt.get("vacuous", [])),
            "claim_boundary": (
                "A confirmed motif is a configuration that recurred on frames it was not "
                "mined from more often than the surrogate null placed it there. It is not a "
                "mechanism, not a cause, and not a claim until something records it - which "
                "is TG11.3's write path (R22). A motif listed under `vacuous` was not "
                "confirmed by an ensemble that could have rejected it (R5)."),
        }, where="confirm")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/motifs")
async def publish_motif(request: PublishRequest) -> Dict[str, Any]:
    """Publish one frozen motif as a durable, process-independent definition.

    The definition carries its origin domain and that domain's licence, because a motif that
    crossed into another domain having forgotten where it came from is exactly the erasure
    R17 exists to prevent.
    """
    try:
        seal = load_seal(_identifier(request.seal_sha256, "seal digest"))
        sealed = _sealed_settings(seal)
        rerun = _rerun(seal, sealed, need_held_out=False)
        chosen = [c for c in rerun["chosen"] if c.label == request.label]
        if not chosen:
            raise InvalidParameterError(
                "label", request.label,
                "one of the exemplars this seal froze: %s. A motif published under a label "
                "the seal does not contain was never frozen against those held-out frames"
                % list(seal.confirm_labels))
        declaration = declaration_for(str(sealed["domain"]))
        motif = freeze_motif(chosen[0], result=rerun["result"], origin_domain=declaration,
                             train=rerun["train_identity"], frozen_at=_now(),
                             study_id=seal.study_id or sealed["record_id"][:12])
        root = _motifs_root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / ("%s.json" % motif.motif_sha256)
        if not path.exists():
            save_frozen_motif(path, motif)
        return _respond({
            "motif_sha256": motif.motif_sha256,
            "definition_sha256": motif.definition_sha256,
            "motif": motif.to_mapping(),
            "origin_licence": motif.origin_domain.get("licence"),
            "publication_note": (
                "Publish `motif_sha256` where you cannot rewrite it before opening any "
                "transfer target. `/transfer` takes it back as `published_sha256`, and a "
                "definition that matches only itself is not evidence about itself."),
            "claim_boundary": (
                "A published definition is a shape and its origin. Whether it recurs "
                "anywhere else is what a transfer searches for, and a match count from a "
                "transfer is descriptive rather than a corrected result."),
        }, where="published motif")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/transfer")
async def transfer(request: TransferRequest) -> Dict[str, Any]:
    """Bind a published motif, spend the target, then open and search it once (R20).

    The order is the whole content of the route.  The binding is written to the transfer
    ledger *before* the target frames are read, so a failure after that point still spends
    the target - which is the honest accounting, because a caller who saw an error message
    after the opener ran has still seen the target.
    """
    try:
        motif = _load_motif(request.motif_sha256)
        target_declaration = _require_minable_domain(request.target_domain)
        record_id = _identifier(request.target_record_id, "record digest")
        _values, declaration = _load_record(record_id)
        if declaration["domain"] != target_declaration.name:
            raise InvalidParameterError(
                "target_record_id", declaration["domain"],
                "a record admitted under the target domain %r. This record was admitted "
                "under %r, and features carry the domain they were extracted from"
                % (target_declaration.name, declaration["domain"]))
        scenes, _reports = _scenes_for(record_id)
        counts = sorted({len(scene) for scene in scenes})
        if len(counts) != 1:
            raise InvalidParameterError(
                "target frames", counts,
                "a target whose frames hold the same number of features; these hold %s"
                % counts)
        target = _identity(declaration, name="transfer_target", split="target",
                           window=(0, len(scenes)), n_features=counts[0],
                           train_ratio=0.5, embargo_frames=0)
        bound_at = _now()
        receipt = blind_transfer(
            motif, published_sha256=request.published_sha256, target=target,
            target_domain=target_declaration, opener=lambda: scenes,
            ledger=_transfer_ledger(), bound_at=bound_at, opened_at=_now())
        return _respond({
            "receipt": receipt,
            "target_record_id": record_id,
            "note": ("The ledger establishes ordering inside this API and nothing more. It "
                     "cannot show that nobody looked at these frames before they were "
                     "admitted; that boundary belongs to the archive the target came from."),
            "claim_boundary": receipt["claim_boundary"],
        }, where="transfer")
    except SpectralEarthError as error:
        raise _handle(error)


@router.post("/invariance")
async def audit_invariance(request: InvarianceRequest) -> Dict[str, Any]:
    """Measure every registered matcher against its own declared invariance.

    A matcher that declares invariance it does not have makes every cross-scene match it
    reports suspect; one that under-declares makes a caller reach for a heavier matcher it
    did not need.  Both are failures here, and a test that could not have failed is reported
    as vacuous rather than as a pass.

    The transforms are the caller's declaration about what was done to the frames, because
    only whoever produced them knows.  What is measured is whether the matcher's signature
    survived them.
    """
    try:
        scenes, _reports = _scenes_for(request.record_id)
        reference = scenes[_frames([request.reference_frame], len(scenes),
                                   what="reference_frame")[0]].features
        replicates = [scenes[i].features for i in
                      _frames(request.replicate_frames, len(scenes),
                              what="replicate_frames")]
        if len(replicates) < 2:
            raise InvalidParameterError(
                "replicate_frames", len(replicates),
                "at least two replicates of the reference configuration under different "
                "noise; the null this audit tests against is built from them")
        presentations = []
        for entry in request.presentations:
            unknown = sorted(set(entry) - {"frame", "transforms", "expected_scale_ratio",
                                           "name"})
            if unknown:
                raise InvalidParameterError(
                    "presentation keys", unknown,
                    "only frame, transforms, name and expected_scale_ratio")
            index = _frames([entry["frame"]], len(scenes), what="presentation frame")[0]
            presentations.append(Presentation(
                str(entry.get("name") or "frame%04d" % index),
                tuple(entry["transforms"]), scenes[index].features,
                entry.get("expected_scale_ratio")))
        if not presentations:
            raise InvalidParameterError(
                "presentations", [],
                "at least one presentation of the reference configuration. An audit with "
                "nothing to compare against reports invariance because it never looked")
        reports = audit_declared_invariance(reference, presentations, replicates,
                                            alpha=float(request.alpha))
        return _respond({
            "record_id": _identifier(request.record_id, "record digest"),
            "alpha": float(request.alpha),
            "reports": {name: report.describe() for name, report in sorted(reports.items())},
            "honest": sorted(name for name, report in reports.items() if report.honest),
            "overclaimed": {name: list(report.overclaimed)
                            for name, report in sorted(reports.items())
                            if report.overclaimed},
            "declared_transforms_are_the_callers": (
                "Which transform each presentation underwent is declared, not measured. A "
                "presentation labelled `rotation` that was in fact a different "
                "configuration would be reported as a matcher failing under rotation."),
            "claim_boundary": (
                "Invariance is necessary for recognising one configuration in two scenes and "
                "nowhere near sufficient. A signature that is constant is perfectly "
                "invariant to everything and matches every configuration to every other; "
                "what separates it from a matcher that measures something is the surrogate "
                "null in `/confirm`, not this audit."),
        }, where="invariance")
    except SpectralEarthError as error:
        raise _handle(error)


# ------------------------------------------------------------------------------- machinery


def _json_object(raw: str, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="`%s` must be a JSON object." % field)
    return value


def _times(raw: str, n_frames: int) -> Tuple[float, ...]:
    if not raw.strip():
        return tuple(float(i) for i in range(n_frames))
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`times` must be a JSON array.")
    if not isinstance(value, list) or len(value) != n_frames:
        raise InvalidParameterError(
            "times", value,
            "one time per frame on the domain's declared clock; this record has %d frames"
            % n_frames)
    times = tuple(float(v) for v in value)
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise InvalidParameterError(
            "times", list(times),
            "strictly increasing times. Frames out of order would put an embargo somewhere "
            "other than between train and held-out")
    return times


def _frames(values: Sequence[int], n_frames: int, *, what: str) -> Tuple[int, ...]:
    try:
        frames = tuple(int(v) for v in values)
    except (TypeError, ValueError):
        raise InvalidParameterError(what, list(values), "integer frame indices")
    outside = [f for f in frames if f < 0 or f >= n_frames]
    if outside:
        raise InvalidParameterError(
            what, outside, "frame indices inside the record's %d frames" % n_frames)
    if len(set(frames)) != len(frames):
        raise InvalidParameterError(
            what, list(frames), "distinct frames; one frame named twice is one observation "
                                "counted twice")
    return frames


def _pipeline(declaration: Mapping[str, Any]) -> Dict[str, Any]:
    """What makes two records the same measurement, for the purpose of a noise floor.

    Not the bytes: replicates of one configuration are by definition *other* frames.  What
    has to match is everything that decides what a feature is - the domain, the dataset, the
    variable, its units, the representation and every extraction setting.  A tolerance
    borrowed across any of those is a measurement of some other pipeline's noise wearing this
    one's name.
    """
    return {key: declaration[key] for key in
            ("domain", "dataset", "variable", "units", "representation", "extraction")}


def _mining_run(request: MiningRequest) -> Dict[str, Any]:
    """Everything `/generate` and `/freeze` share: the split, the mine, the accounting."""
    record_id = _identifier(request.record_id, "record digest")
    scenes, _reports = _scenes_for(record_id)
    _values, declaration = _load_record(record_id)
    split = _split(len(scenes), float(request.train_ratio), int(request.embargo_frames))
    train_scenes = _window(scenes, *split["train"], what="training")
    n_features = len(train_scenes[0])

    tolerance_receipt = _load_tolerance(request.tolerance_sha256)
    calibrated_on = tolerance_receipt["record_id"]
    if _pipeline(_load_record(calibrated_on)[1]) != _pipeline(declaration):
        raise InvalidParameterError(
            "tolerance_sha256", calibrated_on,
            "a tolerance calibrated through the pipeline being mined. A match tolerance is a "
            "measurement of *this* pipeline's noise - the domain, the variable, the "
            "representation and the extraction settings - so it may be measured on other "
            "frames read the same way, and may not be borrowed from frames read another way")
    if tolerance_receipt["matcher"] != request.matcher:
        raise InvalidParameterError(
            "tolerance_sha256", tolerance_receipt["matcher"],
            "a tolerance calibrated under the matcher being used (%r). Each matcher measures "
            "its own quantity, and a tolerance measured through one signature is not a "
            "tolerance for another" % request.matcher)
    spilled = ([f for f in tolerance_receipt["frames"] if f >= split["train"][1]]
               if calibrated_on == record_id else [])
    if spilled:
        raise InvalidParameterError(
            "tolerance_sha256", spilled,
            "a tolerance calibrated inside the training window [0, %d). Frames %s are in the "
            "embargo or the held-out partition, and calibrating on them would let the "
            "held-out frames set the width at which configurations count as repeats"
            % (split["train"][1], spilled))
    tolerance = float(tolerance_receipt["tolerance"]["value"])

    train_identity = _identity(declaration, name="train", split="train",
                               window=split["train"], n_features=n_features,
                               train_ratio=float(request.train_ratio),
                               embargo_frames=int(request.embargo_frames))
    held_window = split["held_out"]
    held_counts = sorted({len(scene) for scene in scenes[held_window[0]:held_window[1]]})
    if held_counts != [n_features]:
        raise InvalidParameterError(
            "held-out frames", held_counts,
            "held-out frames holding the same %d features per frame as the training ones. "
            "The confirmatory family is priced over one configuration size, and frames of "
            "another n are a different search" % n_features)
    held_out_identity = _identity(declaration, name="held_out", split="test",
                                  window=held_window, n_features=n_features,
                                  train_ratio=float(request.train_ratio),
                                  embargo_frames=int(request.embargo_frames))

    result = mine(train_scenes, size=int(request.size), tolerance=tolerance,
                  n_surrogates=int(request.n_surrogates), matcher=request.matcher,
                  alpha=float(request.alpha), correction=request.correction,
                  study_id=request.study_id)
    chosen = choose_candidates(result)
    spent = held_out_ledger().records.get(held_out_identity.digest())
    return {
        "record_id": record_id, "declaration": declaration, "split": split,
        "train_scenes": train_scenes, "train_identity": train_identity,
        "held_out_identity": held_out_identity, "held_out_spent": spent,
        "tolerance_receipt": tolerance_receipt, "result": result, "chosen": chosen,
        "sealed": {
            "record_id": record_id, "size": int(request.size), "matcher": request.matcher,
            "tolerance": tolerance, "tolerance_sha256": _identifier(
                request.tolerance_sha256, "tolerance digest"),
            "train_ratio": float(request.train_ratio),
            "embargo_frames": int(request.embargo_frames),
            "n_surrogates": int(request.n_surrogates), "alpha": float(request.alpha),
            "correction": request.correction, "seed": int(request.seed),
            "domain": declaration["domain"],
        },
    }


def _sealed_settings(seal: Seal) -> Dict[str, Any]:
    """The run this seal froze, read back out of it.

    Not out of a file beside it and not out of the request: a setting the seal does not bind
    is a setting that can be changed after the declaration, which is the freedom R18 exists
    to remove.
    """
    seal.verify()
    notes = dict(seal.confirm.get("notes") or {})
    missing = [key for key in SEALED_MINING_KEYS if key not in notes]
    if missing:
        raise HTTPException(
            status_code=409,
            detail="Seal %s does not carry the mining settings this surface seals (%s "
                   "absent). It was frozen by another surface, and running it here would be "
                   "an analysis this seal never described."
                   % (seal.seal_sha256, ", ".join(missing)))
    settings = {key: notes[key] for key in SEALED_MINING_KEYS}
    settings["domain"] = notes.get("domain", seal.held_out.provenance.get("domain"))
    return settings


def _rerun(seal: Seal, sealed: Mapping[str, Any], *,
           need_held_out: bool = True) -> Dict[str, Any]:
    """Re-derive the training candidates from the sealed settings alone.

    The mining pass is deterministic - the enumeration is combinatorial and the randomness
    enters only at the surrogate ensemble - so the same record, window, size, matcher and
    tolerance produce the same exemplars.  `confirm_motifs` then checks those labels against
    the labels the seal froze, so a record whose declaration changed, or a window that moved,
    is refused rather than confirmed under the old names.
    """
    record_id = _identifier(str(sealed["record_id"]), "record digest")
    scenes, _reports = _scenes_for(record_id)
    _values, declaration = _load_record(record_id)
    split = _split(len(scenes), float(sealed["train_ratio"]), int(sealed["embargo_frames"]))
    train_scenes = _window(scenes, *split["train"], what="training")
    n_features = len(train_scenes[0])
    result = mine(train_scenes, size=int(sealed["size"]),
                  tolerance=float(sealed["tolerance"]),
                  n_surrogates=int(sealed["n_surrogates"]), matcher=str(sealed["matcher"]),
                  alpha=float(sealed["alpha"]), correction=str(sealed["correction"]),
                  study_id=seal.study_id)
    order = {label: index for index, label in enumerate(seal.confirm_labels)}
    by_label = {candidate.label: candidate for candidate in result.candidates}
    missing = [label for label in seal.confirm_labels if label not in by_label]
    if missing:
        raise InvalidParameterError(
            "seal", missing[:8],
            "frozen exemplars that the sealed mining pass still produces. %d of %d frozen "
            "labels are absent when the pass is re-run from the sealed settings, so the "
            "definition behind those names has changed" % (len(missing),
                                                           len(seal.confirm_labels)))
    chosen = tuple(sorted((by_label[label] for label in seal.confirm_labels),
                          key=lambda c: order[c.label]))
    out: Dict[str, Any] = {
        "result": result, "chosen": chosen, "record_id": record_id,
        "train_identity": _identity(declaration, name="train", split="train",
                                    window=split["train"], n_features=n_features,
                                    train_ratio=float(sealed["train_ratio"]),
                                    embargo_frames=int(sealed["embargo_frames"])),
    }
    if need_held_out:
        window = split["held_out"]
        out["held_out_scenes"] = _window(scenes, *window, what="held-out")
        out["held_out_identity"] = _identity(
            declaration, name="held_out", split="test", window=window,
            n_features=n_features, train_ratio=float(sealed["train_ratio"]),
            embargo_frames=int(sealed["embargo_frames"]))
    return out


def _load_motif(motif_sha256: str) -> FrozenMotif:
    digest = _identifier(motif_sha256, "motif digest")
    path = _motifs_root() / ("%s.json" % digest)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No published motif %s is stored here. A transfer carries a definition "
                   "that was frozen and published before the target was opened; there is "
                   "nothing here to transfer." % digest)
    return load_frozen_motif(path)


__all__ = ["router", "ROOT_ENV", "DEFAULT_ROOT", "SEALED_MINING_KEYS",
           "TRANSFER_LEDGER_NAME"]
