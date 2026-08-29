"""TG11.3: the evidence write path — the first place the interface writes toward a claim.

Everything before this slice was read-only by construction.  `api/analysis.py` computes and
stores nothing; `api/preregistration.py` stores an ordering promise that records no evidence and
moves no rung.  This module writes into an `EvidenceBundle`, and a bundle is the thing the claim
ladder grades, so this is the slice where rule R22 stops being a property of the architecture and
becomes a property of these handlers.

**The rung is recomputed, never accepted.**  No request body on this surface has a field for a
rung, a claim level or a confidence, and that is enforced structurally rather than by review:
every model forbids unknown fields, so ``{"rung": "candidate_precursor"}`` is refused rather than
silently ignored.  The rung in every response is `assess_claim_ladder` run over the chain that
was just written, and it is recomputed on each read rather than stored, because a stored rung is
a figure that can disagree with the evidence underneath it.

**The one payload key that could climb a rung by assertion.**  The ladder reads exactly one
reserved key out of an evidence payload: ``temporal_precedence``, on passing ``provenance``
entries, and it is the gate for ``candidate_precursor``.  A write path that accepted a free-form
payload would therefore let a client type its way up a rung -- the most direct R22 hole in the
programme, and an arithmetic-free one, so no estimator would ever notice it.  This surface
refuses that key anywhere in a hand-written payload and offers ``/evidence/precedence`` instead,
which runs `analyse_precedence` server-side and records whatever *that* returns, ``false``
included.  The caller chooses the record and the lag family; it does not choose the answer.

**Appends are compare-and-swap.**  An append states the ``head_sha256`` it believes it is
extending, and a revision is a new file created exclusively, so two writers racing on one study
produce one append and one 409 rather than a lost entry.  This is stronger than TG11.2's ledger,
which takes no lock, and the difference is deliberate: evidence is the thing being protected.

**What this module still cannot do.**  It cannot delete, amend or reorder an entry -- the chain
is append-only and each entry carries its predecessor's digest, so an edit is detectable rather
than merely discouraged.  It cannot record commentary: the ten categories in `EVIDENCE_FIELDS`
are the whole vocabulary and free prose is not one of them (R22).  And a bundle carries no
domain, so nothing written here records which instrument the evidence came from.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Extra, Field

from src.analysis_engine.domain_analysis import analyse_precedence
from src.api.findings import StudyStore, refuse_bare_confidence, study_root
from src.core.claim_ladder import OUTSIDE_THE_LADDER, PRECEDENCE_KEY, assess_claim_ladder
from src.core.errors import InvalidParameterError, SpectralEarthError, classify
from src.core.evidence import (EVIDENCE_FIELDS, EVIDENCE_STATUSES, EvidenceBundle,
                               create_evidence_bundle, save_evidence_bundle)
from src.data_layer.tabular_source import VALUE_MEASURE, read_channels_for_domain


router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])

EVIDENCE_SCHEMA_HTTP = "spectral.evidence.http.v1"

#: A study identifier is also a filename here, so it is constrained rather than escaped: a name
#: that cannot traverse a directory needs no sanitiser downstream to remember.
STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Names a client might reach for when it wants to record a verdict instead of an observation.
#: Refused in a payload by name, because a bundle carrying a `rung` key would be read back as
#: though the record asserted one, whatever the ladder said (R22).
RESERVED_PAYLOAD_KEYS: Tuple[str, ...] = (
    "rung", "rung_index", "claim_rung", "claim_level", "unblocked_rung", "claim_ladder",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle(error: SpectralEarthError) -> HTTPException:
    info = classify(error)
    return HTTPException(status_code=info["status_code"], detail=info["detail"])


# -------------------------------------------------------------------------------- the store


def _study_id(value: str) -> str:
    if not isinstance(value, str) or STUDY_ID.fullmatch(value) is None:
        raise InvalidParameterError(
            "study_id", value,
            "1-64 characters of letters, digits, dot, dash or underscore, beginning with a "
            "letter or digit. A study identifier names a file in the study store, so a name "
            "that could traverse a directory is refused rather than quietly rewritten into "
            "one that does not match what the caller asked for")
    return value


def _revision_path(study_id: str, revision: int) -> Path:
    """One file per revision.

    A bundle is immutable and `save_evidence_bundle` refuses to overwrite, so a revision cannot
    be a rewrite of one file: it is a new one. The exclusive create is also the concurrency
    control - two appends computed from the same head collide on the filename, and the loser is
    told so rather than silently dropped.
    """
    return study_root() / ("%s.r%05d.json" % (study_id, revision))


def _current(study_id: str) -> EvidenceBundle:
    try:
        bundle, _ = StudyStore().load(study_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="No study %r is stored here. Open one before recording evidence against it: "
                   "evidence with no registered hypothesis is a measurement, not a test of "
                   "anything." % study_id)
    return bundle


def _publish(bundle: EvidenceBundle) -> EvidenceBundle:
    path = _revision_path(bundle.study_id, bundle.revision)
    try:
        save_evidence_bundle(path, bundle)
    except FileExistsError:
        raise HTTPException(
            status_code=409,
            detail="Revision %d of study %r already exists. Another writer reached this "
                   "revision first; re-read the head and append to what is now there rather "
                   "than to what was there when this request was composed."
                   % (bundle.revision, bundle.study_id))
    return bundle


# --------------------------------------------------------------------------- what is refused


def _reserved(value: Any, *, where: str = "payload") -> None:
    """Refuse a payload that asserts a verdict instead of recording an observation."""
    if isinstance(value, Mapping):
        for key in value:
            if key == PRECEDENCE_KEY:
                raise InvalidParameterError(
                    "%s.%s" % (where, key), value[key],
                    "evidence, not a precedence verdict. `%s` is the one payload key the claim "
                    "ladder reads, and it is the gate for the candidate_precursor rung, so a "
                    "client that could set it would climb a rung by typing. Record it with "
                    "POST /api/v1/evidence/studies/{study_id}/evidence/precedence, which runs "
                    "the analysis and writes whatever it returns" % PRECEDENCE_KEY)
            if key in RESERVED_PAYLOAD_KEYS:
                raise InvalidParameterError(
                    "%s.%s" % (where, key), value[key],
                    "evidence, not a rung. The rung is recomputed from the whole chain by the "
                    "claim ladder on every read and is never stored, because a stored rung is "
                    "a figure that can disagree with the evidence beneath it (R22)")
        for key, item in value.items():
            _reserved(item, where="%s.%s" % (where, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reserved(item, where="%s[%d]" % (where, index))


def _expect_head(bundle: EvidenceBundle, expected: str) -> None:
    if expected != bundle.head_sha256:
        raise HTTPException(
            status_code=409,
            detail="Study %r is at revision %d with head %s; this append names %s. The chain "
                   "moved under this request, and appending anyway would record an entry whose "
                   "author had not seen what it follows."
                   % (bundle.study_id, bundle.revision, bundle.head_sha256, expected))


# ------------------------------------------------------------------------------- the bodies


class _Body(BaseModel):
    class Config:
        extra = Extra.forbid


class OpenStudy(_Body):
    """Revision zero: a hypothesis, registered before any evidence is attached to it."""

    study_id: str
    hypothesis_id: str
    statement: str
    prediction: str
    provenance: Dict[str, Any] = Field(default_factory=dict)


class AppendEvidence(_Body):
    """One entry, extending a chain the caller has read."""

    expected_head_sha256: str
    category: str
    label: str
    status: str
    summary: str
    payload: Dict[str, Any]
    source_sha256s: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------- the response


def _state(bundle: EvidenceBundle) -> Dict[str, Any]:
    """Where the bundle stands, recomputed from the chain and stored nowhere."""
    assessment = assess_claim_ladder(bundle)
    return {
        "study_id": bundle.study_id,
        "revision": bundle.revision,
        "head_sha256": bundle.head_sha256,
        "bundle_sha256": bundle.bundle_sha256,
        "hypothesis_sha256": bundle.hypothesis.digest(),
        "ladder": assessment.to_mapping(),
        "assessment_sha256": assessment.assessment_sha256,
        "blocked": assessment.blocked,
        "unsatisfied_gates": list(assessment.unsatisfied_gates),
        "rung_source": ("recomputed from the evidence chain by src.core.claim_ladder on this "
                        "request. Not stored, not supplied by the client, and not moved by "
                        "anything this surface accepts (R22)"),
    }


def _respond(bundle: EvidenceBundle, **extra: Any) -> Dict[str, Any]:
    body = dict({"schema": EVIDENCE_SCHEMA_HTTP}, **_state(bundle), **extra)
    # The read surface's wire guard, applied on the way out of the write surface: an echoed
    # payload is as much a response as a rendered finding (R9).
    refuse_bare_confidence(body, where="evidence response")
    return body


# ---------------------------------------------------------------------------------- routes


@router.get("")
async def capabilities() -> Dict[str, Any]:
    """What may be written here, and what is computed rather than accepted."""
    return {
        "schema": EVIDENCE_SCHEMA_HTTP,
        "categories": list(EVIDENCE_FIELDS),
        "statuses": list(EVIDENCE_STATUSES),
        "append_only": True,
        "compare_and_swap": ("every append names the head_sha256 it extends, and a revision is "
                             "a new file created exclusively, so a race loses to a 409 rather "
                             "than to a lost entry"),
        "rung_is_computed": True,
        "accepts_a_rung": False,
        "computed_not_accepted": {
            PRECEDENCE_KEY: ("the candidate_precursor gate. Refused in a hand-written payload "
                             "and written only by /evidence/precedence, from the analysis that "
                             "ran, including when the answer is false"),
            "recorded_at": "the server clock, so the append order and the chronology agree",
        },
        "commentary": ("has no category here. The ten fields are the whole vocabulary, and "
                       "prose about a result is not one of them (R22)"),
        "claim_boundary": (
            "Recording evidence is not establishing a finding. The ladder grades what is in "
            "the chain, and a chain of one favourable observation reaches `observation`. A "
            "bundle carries no domain, so nothing written here records which instrument the "
            "evidence came from. And this surface does not consult the held-out ledger: "
            "running a precedence analysis over data that was preregistered as held out "
            "spends it outside the record, which the ledger cannot see and this note cannot "
            "prevent."),
    }


@router.post("/studies")
async def open_study(body: OpenStudy) -> Dict[str, Any]:
    """Register a hypothesis at revision zero, before any evidence exists to favour it."""
    try:
        study_id = _study_id(body.study_id)
        try:
            StudyStore().load(study_id)
        except KeyError:
            pass
        else:
            raise HTTPException(
                status_code=409,
                detail="Study %r already exists. A second bundle under one identifier would "
                       "let two hypotheses share a name, and the ladder grades one chain at a "
                       "time." % study_id)
        moment = _now()
        bundle = create_evidence_bundle(
            study_id=study_id, hypothesis_id=body.hypothesis_id, statement=body.statement,
            prediction=body.prediction, registered_at=moment, created_at=moment,
            provenance=dict(body.provenance))
    except SpectralEarthError as error:
        raise _handle(error)
    _publish(bundle)
    return _respond(bundle, hypothesis=bundle.hypothesis.to_mapping(),
                    wording_note=_wording_note(body.statement, body.prediction))


@router.get("/studies/{study_id}/head")
async def study_head(study_id: str) -> Dict[str, Any]:
    """The digest an append must name, and where the chain currently stands."""
    bundle = _current(study_id)
    return _respond(bundle, next_sequence=bundle.revision + 1,
                    entries=[entry.to_mapping() for entry in bundle.entries])


@router.post("/studies/{study_id}/evidence")
async def append_evidence(study_id: str, body: AppendEvidence) -> Dict[str, Any]:
    """Append one recorded observation to a chain the caller has read."""
    bundle = _current(study_id)
    _expect_head(bundle, body.expected_head_sha256)
    try:
        _reserved(body.payload)
        refuse_bare_confidence(body.payload, where="payload")
        appended = bundle.append(
            body.category, label=body.label, status=body.status, summary=body.summary,
            recorded_at=_now(), payload=body.payload,
            source_sha256s=tuple(body.source_sha256s))
    except SpectralEarthError as error:
        raise _handle(error)
    _publish(appended)
    return _respond(appended, entry=appended.entries[-1].to_mapping())


@router.post("/studies/{study_id}/evidence/precedence")
async def append_precedence(
    study_id: str,
    expected_head_sha256: str = Form(...),
    label: str = Form(...),
    file: UploadFile = File(...),
    domain: str = Form(...),
    time_column: str = Form(...),
    lags: str = Form(...),
    n_surrogates: int = Form(499),
    alpha: float = Form(0.05),
    correction: str = Form("benjamini_yekutieli"),
    estimator: str = Form("mutual_information"),
    bins: int = Form(4),
    seed: int = Form(20260827),
    time_units: str = Form("s"),
    delimiter: str = Form(","),
) -> Dict[str, Any]:
    """Record a provenance entry whose precedence verdict is computed here, not supplied.

    The caller chooses the record and the lag family. Whether temporal precedence was found is
    read out of the sweep, and a negative is recorded as a negative rather than withheld.
    """
    bundle = _current(study_id)
    _expect_head(bundle, expected_head_sha256)
    filename = file.filename or "upload"
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="%r is not UTF-8 text. This surface records evidence computed over an "
                   "admitted channel table." % filename)
    try:
        frames = _lags(lags)
        series, declaration = read_channels_for_domain(
            text, source_name=filename, domain=domain, time_column=time_column,
            time_units=time_units, delimiter=delimiter)
        cadence = series.provenance.get("cadence_seconds")
        if not isinstance(cadence, (int, float)) or cadence <= 0:
            raise InvalidParameterError(
                "record clock", cadence,
                "a regular clock with a finite positive cadence. A lag in frames is not a "
                "duration on an irregular clock, and precedence is a claim about durations")
        result = analyse_precedence(
            series, declaration, lags=frames, cadence_seconds=float(cadence),
            measure=VALUE_MEASURE, estimator=estimator, bins=int(bins),
            n_surrogates=int(n_surrogates), alpha=float(alpha), correction=correction,
            seed=int(seed))
        entry_payload, status = _precedence_payload(result, series, declaration,
                                                    alpha=float(alpha))
        appended = bundle.append(
            "provenance", label=label, status=status,
            summary=("precedence analysis computed by the server over %d frames of domain %r; "
                     "%d of %d tests survived %s correction at alpha %g"
                     % (series.n_times, declaration.name, entry_payload["n_significant"],
                        entry_payload["n_tests"], entry_payload["correction"], float(alpha))),
            recorded_at=_now(), payload=entry_payload,
            source_sha256s=_sources(series, result))
    except SpectralEarthError as error:
        raise _handle(error)
    _publish(appended)
    return _respond(appended, entry=appended.entries[-1].to_mapping(),
                    computed_here=True,
                    verdict_source=("src.analysis_engine.domain_analysis.analyse_precedence on "
                                    "this request, over the record supplied. The caller chose "
                                    "the record and the lag family and nothing else"))


# --------------------------------------------------------------------------------- helpers


def _lags(raw: str) -> Tuple[int, ...]:
    try:
        frames = tuple(int(part) for part in str(raw).replace(",", " ").split())
    except (TypeError, ValueError):
        raise InvalidParameterError("lags", raw, "whitespace- or comma-separated frame lags")
    if not frames or any(lag < 1 for lag in frames) or len(set(frames)) != len(frames):
        raise InvalidParameterError(
            "lags", list(frames),
            "at least one positive frame lag, each named once. A repeated lag is one test "
            "counted twice, and the correction would be priced short by the duplication")
    return frames


def _sources(series: Any, result: Mapping[str, Any]) -> Tuple[str, ...]:
    """The two digests that make this entry checkable: the bytes, and the configuration."""
    digests = [value for value in (series.provenance.get("content_sha256"),
                                   result.get("analysis_config_sha256"))
               if isinstance(value, str) and len(value) == 64]
    return tuple(dict.fromkeys(digests))


def _precedence_payload(result: Mapping[str, Any], series: Any, declaration: Any, *,
                        alpha: float) -> Tuple[Dict[str, Any], str]:
    rows = [row for row in result.get("results", [])
            if bool(row.get("significant"))
            and float(row.get("q_value", 1.0)) <= alpha
            and float(row.get("excess_nats", 0.0)) > 0.0]
    powered = bool((result.get("power") or {}).get("can_reject_after_correction", False))
    payload = {
        # Written from the sweep, never from the request. False is a result here, not a gap.
        PRECEDENCE_KEY: bool(rows) and result.get("claim_boundary") == "precedence",
        "claim_boundary": result.get("claim_boundary"),
        "domain": declaration.name,
        "record_sha256": series.provenance.get("content_sha256"),
        "frames": int(series.n_times),
        "channels": [str(channel) for channel in series.channels],
        "cadence_seconds": float(result.get("cadence_seconds", 0.0)),
        "lags_frames": [int(lag) for lag in result.get("lags_frames", [])],
        "n_tests": int(result.get("n_tests", 0)),
        "n_significant": int(len(rows)),
        "n_excluded": int(result.get("n_excluded", 0)),
        "alpha": float(alpha),
        "correction": str(result.get("correction")),
        "n_surrogates_requested": int(result.get("n_surrogates_requested", 0)),
        "seed": int(result.get("seed", 0)),
        "analysis_config_sha256": result.get("analysis_config_sha256"),
        "can_reject_after_correction": powered,
        "surviving": [{"label": str(row.get("label")),
                       "q_value": float(row.get("q_value", 1.0)),
                       "excess_nats": float(row.get("excess_nats", 0.0))}
                      for row in rows],
    }
    # An underpowered sweep could not have rejected anything, so its silence is not a negative
    # result and must not open the provenance gate as though the check had been made (R5).
    return payload, ("PASS" if powered else "INCONCLUSIVE")


def _wording_note(statement: str, prediction: str) -> Optional[str]:
    """Say once, at registration, when a hypothesis is worded above the top of the ladder."""
    words = set(re.findall(r"[a-z]+", ("%s %s" % (statement, prediction)).lower()))
    found = sorted(words.intersection(OUTSIDE_THE_LADDER))
    if not found:
        return None
    return ("This hypothesis is worded causally (%s). It is registered as written, but no "
            "revision of this bundle can reach that wording: the ladder tops out at "
            "demonstrated predictive utility, which is a statement about out-of-sample "
            "prediction and not about mechanism (R7)." % ", ".join(found))


__all__ = ["router", "EVIDENCE_SCHEMA_HTTP", "RESERVED_PAYLOAD_KEYS", "STUDY_ID"]
