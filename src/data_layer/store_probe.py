"""Probing a store as a recorded act (roadmap TG10.3, standards E1/E5).

**Why this is a module and not a habit.** Probing already happened: somebody opened the
WeatherBench stores, read their chunk metadata and wrote what they found into a prose note.
That is how the catalogue came to say *"measured 51.1x amplification"*, and it is also how
defect D43 came to be discovered a year late — the 0.703125 degree store reads as the sensible
middle option and is not laptop-feasible for a long regional record. A practice that lives in
somebody's terminal history produces exactly that: figures for the stores anyone happened to
look at, silence for the rest, and no way to tell the two apart afterwards. Defect **D62** is
the same failure one level in, a per-chunk size written into a field that no inspection had
ever filled.

**The result is recorded either way.** `unreachable`, `needs credentials` and `network is
switched off` are *results* about a store, not failures of the probe, and they are the results
a researcher most needs before planning around a store. A probe that raised on them would
leave exactly the stores worth warning about unrecorded.

**Nothing here happens by accident.** `probe_store` reaches the network only when
`SPECTRALEARTH_ALLOW_NETWORK` is set, and refuses in a recorded way rather than an exception
when it is not. Importing this module opens nothing; registering a store opens nothing. The
probe is a deliberate act whose *record* is what registration then requires.

**Two kinds of evidence, and the difference is not cosmetic.** A record with
`evidence="probe run"` was produced by this code opening that URI. A record with
`evidence="prior recorded inspection"` is a transcription of an inspection someone ran before
this module existed — the four ERA5 entries are all of that kind. Transcriptions are accepted
because refusing them would mean either deleting four true records or fabricating four live
runs, and both are worse. They are counted, so the number can only go down where anyone can
see it (`transcribed_probes`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.registry import Registry

PROBE_SCHEMA = "gridded-store-probe/v1"

#: Where probe records live when they are persisted. Content-addressed by digest, so the same
#: probe written twice is one file and a differing probe can never overwrite an existing one.
DEFAULT_PROBE_DIR = os.path.join("data", "store_probes")

#: What a probe can conclude. Four of the five are the store declining to be read, and every
#: one of them is a **result**: a researcher planning a three-year crop needs "this store wants
#: credentials" recorded just as much as a chunk shape.
PROBE_OUTCOMES: Dict[str, str] = {
    "described": "the store opened and its structure was read",
    "network_disabled": ("the store is remote and network access is not enabled here; "
                         "nothing was attempted"),
    "needs_credentials": "the store exists but declined an anonymous read",
    "unreachable": "the store could not be opened from here",
    "not_readable": "the store opened but could not be read as Zarr",
}

#: How a record came to exist. See the module docstring: the distinction is the honest part.
EVIDENCE_KINDS: Dict[str, str] = {
    "probe run": "produced by `probe_store` opening this URI",
    "prior recorded inspection": ("transcribed from an inspection run before this module "
                                  "existed, and counted so the number can only go down"),
}

#: Amplification at or above which a store is characterised as hostile to regional crops.
#: The same threshold `zarr_source.HOSTILE_AMPLIFICATION` uses, restated rather than imported
#: because importing `zarr_source` at module scope would close a cycle: it imports `stores`,
#: and `stores` imports this module to enforce the probe requirement.
HOSTILE_AMPLIFICATION = 4.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@dataclass(frozen=True)
class StoreProbe:
    """One recorded look at one store, successful or not.

    Frozen and content-addressed. `digest` covers everything the probe *observed* and the URI
    it observed it at, but **not** `probed_on`: two probes of an unchanged store on different
    days describe the same store, and letting the date into the digest would produce a new
    record every time anyone looked, which turns a ledger into a log.
    """

    uri: str
    outcome: str
    evidence: str
    probed_on: str = dc_field(default_factory=_utc_date)
    dimensions: Dict[str, int] = dc_field(default_factory=dict)
    variables: Tuple[str, ...] = ()
    #: Per variable: dims, shape, dtype, chunks, chunk_bytes, chunk_megabytes, n_chunks -
    #: exactly what `zarr_source.describe_store` reports, kept in its shape rather than
    #: reduced, because reducing it here would be a second place the reduction could drift.
    variable_structure: Dict[str, Any] = dc_field(default_factory=dict)
    #: The crop the amplification below was computed *for*. An amplification with no crop
    #: attached is not a fact about anything: it is the ratio for one access pattern.
    crop: Optional[Dict[str, Any]] = None
    amplification: Optional[float] = None
    megabytes_per_chunk: Optional[float] = None
    #: Why the store declined, verbatim, for a non-`described` outcome. Kept whole because the
    #: difference between "no such bucket" and "403" is the difference between a typo and an
    #: account, and a summarised message loses exactly that.
    refusal_detail: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.uri).strip():
            raise InvalidParameterError("uri", self.uri, "the URI that was probed")
        if self.outcome not in PROBE_OUTCOMES:
            raise InvalidParameterError(
                "outcome", self.outcome, "one of %s" % sorted(PROBE_OUTCOMES))
        if self.evidence not in EVIDENCE_KINDS:
            raise InvalidParameterError(
                "evidence", self.evidence, "one of %s" % sorted(EVIDENCE_KINDS))
        if not str(self.probed_on).strip():
            raise InvalidParameterError(
                "probed_on", self.probed_on,
                "an ISO date; a probe of an archive that may rechunk is worthless undated")
        if self.outcome == "described":
            # At least one observation, not specifically a per-variable structure. A live
            # probe records all three; a transcription of an older inspection may carry only
            # what that inspection wrote down, and demanding the rest would force whoever
            # transcribes it to invent the difference - which is defect D62 exactly.
            if not (self.variable_structure or self.dimensions
                    or self.megabytes_per_chunk is not None
                    or self.amplification is not None):
                raise InvalidParameterError(
                    "variable_structure", self.variable_structure,
                    "something that was actually observed - a structure, the dimensions, a "
                    "chunk size or an amplification - for outcome 'described'")
        elif not self.refusal_detail.strip():
            raise InvalidParameterError(
                "refusal_detail", self.refusal_detail,
                "why the store declined, for outcome %r. A refusal recorded without its "
                "reason cannot be told apart from a typo" % self.outcome)
        if self.amplification is not None and self.crop is None:
            raise InvalidParameterError(
                "crop", self.crop,
                "the crop the amplification was computed for. An amplification with no crop "
                "attached is not a fact about a store, only about one access pattern")
        for name, value in (("amplification", self.amplification),
                            ("megabytes_per_chunk", self.megabytes_per_chunk)):
            if value is not None and value <= 0:
                raise InvalidParameterError(name, value, "a positive value, or None")

    # ---------------------------------------------------------------- identity

    def canonical(self) -> Dict[str, Any]:
        """What the digest covers. Deliberately excludes `probed_on` and `evidence`."""
        return {
            "schema": PROBE_SCHEMA,
            "uri": self.uri,
            "outcome": self.outcome,
            "dimensions": dict(sorted(self.dimensions.items())),
            "variables": sorted(self.variables),
            "variable_structure": self.variable_structure,
            "crop": self.crop,
            "amplification": self.amplification,
            "megabytes_per_chunk": self.megabytes_per_chunk,
            "refusal_detail": self.refusal_detail,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()[:16]

    @property
    def chunk_hostile(self) -> Optional[bool]:
        """Whether a regional crop of this store over-delivers, or `None` if unknown.

        `None` is a third answer and is never folded into `False`: "nobody measured" and "it is
        fine" are the two states D43 confused, and a store characterised as friendly because
        nothing looked is the failure this whole slice exists to stop.
        """
        if self.amplification is None:
            return None
        return self.amplification >= HOSTILE_AMPLIFICATION

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.canonical())
        payload.update({
            "digest": self.digest(),
            "evidence": self.evidence,
            "evidence_means": EVIDENCE_KINDS[self.evidence],
            "outcome_means": PROBE_OUTCOMES[self.outcome],
            "probed_on": self.probed_on,
            "chunk_hostile": self.chunk_hostile,
            "note": self.note,
        })
        return payload

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "StoreProbe":
        if record.get("schema") not in (None, PROBE_SCHEMA):
            raise InvalidParameterError("schema", record.get("schema"), PROBE_SCHEMA)
        fields = {"uri", "outcome", "evidence", "probed_on", "dimensions", "variables",
                  "variable_structure", "crop", "amplification", "megabytes_per_chunk",
                  "refusal_detail", "note"}
        kwargs = {k: v for k, v in record.items() if k in fields}
        if "variables" in kwargs:
            kwargs["variables"] = tuple(kwargs["variables"])
        return cls(**kwargs)


#: Recorded probes, keyed by digest (standard E1). Keyed by digest and not by URI because a
#: store can be probed more than once and the earlier record does not stop being true.
STORE_PROBES: Registry[StoreProbe] = Registry("store probe")


def record_probe(probe: StoreProbe, *, replace: bool = False) -> StoreProbe:
    """Put a probe in the ledger. Re-recording an identical probe is a no-op, not an error.

    Identical means *same digest*, which means the same observation of the same URI. Two
    people probing the same unchanged store should not collide, and should not produce two
    entries either.
    """
    digest = probe.digest()
    if digest in STORE_PROBES and not replace:
        return STORE_PROBES.get(digest)
    return STORE_PROBES.add(
        digest, probe,
        description="%s: %s" % (probe.outcome, probe.uri),
        capabilities={"uri": probe.uri, "outcome": probe.outcome,
                      "evidence": probe.evidence, "chunk_hostile": probe.chunk_hostile,
                      "probed_on": probe.probed_on},
        tags=[probe.outcome, probe.evidence], replace=True)


def probe_by_digest(digest: str) -> StoreProbe:
    return STORE_PROBES.get(digest)


def probes_for_uri(uri: str) -> List[StoreProbe]:
    """Every recorded probe of one URI, most recently probed first."""
    matching = [entry.value for entry in STORE_PROBES.entries() if entry.value.uri == uri]
    return sorted(matching, key=lambda p: p.probed_on, reverse=True)


def latest_probe_for(uri: str) -> Optional[StoreProbe]:
    found = probes_for_uri(uri)
    return found[0] if found else None


def transcribed_probes() -> List[StoreProbe]:
    """Records that are transcriptions rather than runs. The debt, made countable."""
    return [entry.value for entry in STORE_PROBES.entries()
            if entry.value.evidence == "prior recorded inspection"]


def probe_ledger() -> List[Dict[str, Any]]:
    """The whole ledger, newest first, in the shape the API and the UI read."""
    return [entry.value.to_dict() for entry in
            sorted(STORE_PROBES.entries(), key=lambda e: e.value.probed_on, reverse=True)]


# --------------------------------------------------------------------------- probing

def _classify(exc: Exception) -> Tuple[str, str]:
    """Turn an open failure into an outcome and keep the reason verbatim.

    The text is matched, which is unlovely, but the alternative is worse: `fsspec` raises the
    driver's own exception types, so the class is `gcsfs`'s or `s3fs`'s business and changes
    with the driver. What is stable is that a credential refusal says so. A miss lands on
    `unreachable`, which is the honest default - it says the store could not be opened from
    here and does not claim to know why.
    """
    detail = "%s: %s" % (type(exc).__name__, exc)
    lowered = detail.lower()
    for needle in ("credential", "unauthorized", "unauthorised", "forbidden", "permission",
                   "401", "403", "access denied", "anonymous"):
        if needle in lowered:
            return "needs_credentials", detail
    if "could not read it as zarr" in lowered or "not a zarr" in lowered:
        return "not_readable", detail
    return "unreachable", detail


def probe_store(uri: str, *, variables: Optional[Sequence[str]] = None,
                crop: Any = None, storage_options: Optional[Dict[str, Any]] = None,
                probed_on: Optional[str] = None, note: str = "") -> StoreProbe:
    """Open a store, record what it is, and record the refusal if it will not open.

    Metadata only: `chunks={}` opens lazily, so this reads the store's `.zmetadata` and
    nothing else. Passing `crop` additionally computes what that crop would cost through
    `assess_access_pattern`, which is chunk arithmetic and transfers nothing either - the
    whole point being that the warning arrives *before* the download.

    Never raises for a store that declines. It raises only for a caller error, such as an
    empty URI, because that is a fault in the request rather than a fact about a store.
    """
    from src.data_layer import zarr_source  # deferred: `zarr_source` imports `stores`

    if not str(uri).strip():
        raise InvalidParameterError("uri", uri, "a store URI or local path to probe")
    when = probed_on or _utc_date()
    remote = "://" in uri

    if remote and not zarr_source.network_enabled():
        return StoreProbe(
            uri=uri, outcome="network_disabled", evidence="probe run", probed_on=when,
            refusal_detail=("network access is opt-in and %s is not set, so nothing was "
                            "attempted. This is a fact about this deployment, not about the "
                            "store." % zarr_source.NETWORK_ENV_VAR),
            note=note)

    try:
        dataset, _counter = zarr_source.open_dataset(uri, storage_options, chunks={})
    except Exception as exc:  # noqa: BLE001 - every open failure is a result to record
        outcome, detail = _classify(exc)
        return StoreProbe(uri=uri, outcome=outcome, evidence="probe run", probed_on=when,
                          refusal_detail=detail, note=note)

    try:
        structure = zarr_source.describe_store(dataset, variables)
        amplification: Optional[float] = None
        crop_record: Optional[Dict[str, Any]] = None
        if crop is not None:
            assessment = zarr_source.assess_access_pattern(dataset, crop)
            amplification = float(assessment["amplification"])
            crop_record = crop.to_provenance()
        sizes = [entry.get("chunk_megabytes") for entry in structure["variables"].values()
                 if entry.get("chunk_megabytes")]
        return StoreProbe(
            uri=uri, outcome="described", evidence="probe run", probed_on=when,
            dimensions=dict(structure["dimensions"]),
            variables=tuple(sorted(structure["variables"])),
            variable_structure=structure["variables"],
            crop=crop_record, amplification=amplification,
            # The **worst** chunk, not the mean. A store is as expensive to crop as its
            # most expensive variable, and averaging hides the one that hurts.
            megabytes_per_chunk=max(sizes) if sizes else None,
            note=note)
    finally:
        try:
            dataset.close()
        except Exception:  # noqa: BLE001 - closing a lazy dataset must not mask the result
            pass


def probe_and_record(uri: str, **kwargs: Any) -> StoreProbe:
    """`probe_store` plus the ledger entry, which is what a caller almost always wants."""
    return record_probe(probe_store(uri, **kwargs))


# --------------------------------------------------------------------------- persistence

def _probe_dir(value: Optional[str] = None) -> str:
    return value if value is not None else DEFAULT_PROBE_DIR


def probe_path(probe: StoreProbe, probe_dir: Optional[str] = None) -> str:
    return os.path.join(_probe_dir(probe_dir), "%s.json" % probe.digest())


def save_probe(probe: StoreProbe, probe_dir: Optional[str] = None) -> str:
    """Write a probe atomically, and never over an existing one.

    Content-addressed, so "never over an existing one" costs nothing: a file already at this
    path holds this same observation. Writing through a temporary file and renaming means a
    reader never sees a half-written record, and an interrupted write leaves no record rather
    than a truncated one.
    """
    path = Path(probe_path(probe, probe_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return str(path)
    payload = _canonical_json(probe.to_dict())
    descriptor, name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp",
                                        dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(path)


def load_probes(probe_dir: Optional[str] = None, *, record: bool = True) -> List[StoreProbe]:
    """Read every persisted probe, optionally putting each into the ledger."""
    directory = _probe_dir(probe_dir)
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as handle:
            probe = StoreProbe.from_dict(json.load(handle))
        if record:
            record_probe(probe)
        found.append(probe)
    return found


__all__ = ["DEFAULT_PROBE_DIR", "EVIDENCE_KINDS", "HOSTILE_AMPLIFICATION",
           "PROBE_OUTCOMES", "PROBE_SCHEMA", "STORE_PROBES", "StoreProbe", "load_probes",
           "latest_probe_for", "probe_and_record", "probe_by_digest", "probe_ledger",
           "probe_path", "probe_store", "probes_for_uri", "record_probe", "save_probe",
           "transcribed_probes"]
