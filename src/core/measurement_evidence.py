"""T4E.30: carrying a measurement into an evidence bundle, or refusing to by name.

**The gap this closes.** This repository keeps two stores that have never touched. Measurements,
declarations and adoptions live in `measurements/` and `data/identity_calibration/`, and the
identity router serves them to the study trail. The review machinery -- the adversarial
round-robin, the recorded discussion, the claim ladder -- reads `EvidenceBundle`s from
`data/studies/`, and `summarise_evidence` is by design *"a pure function of the bundle: no clock,
no I/O, no other input"*. So a panel sees exactly what was appended as evidence entries and
nothing else.

On 2026-09-11 there were 28 measurement records, 43 declarations and adoptions, and **zero
bundles**. Every finding this programme has produced -- a falsified range, a lifted refusal, a
reproduction that confirmed parameters recovered from a temp directory -- was invisible to the
stage built to argue about it. The single mention of the word in the whole review layer is
`api/evidence.py`'s *"evidence with no registered hypothesis is a measurement, not a test of
one"*, which names the seam exactly and never crosses it.

**Why crossing it is not a formatting exercise.** A bundle refuses a hypothesis registered after
the evidence it explains, and it is right to. Back-dating a hypothesis onto a measurement that
already exists manufactures a preregistration, and a panel reading that bundle would be arguing
about a risk nobody took. So registration is not asserted here -- it is **proved from git**: the
commit that first added the declaration must precede the commit that first added the measurement,
both files must be tracked and unmodified, and a record that cannot show this is refused by name
rather than bundled with a caveat.

**What this module will not decide.** It does not choose what any measurement means. Which
entries a record yields, in which category and at which status, is a scientific judgement and is
supplied by the caller as `EvidenceClaim`s -- written down, reviewable, and attributable. Nothing
here reads a record and infers that a number is a null result or a contradiction. A tool that
guessed would be authoring the evidence it claims to be transporting, and the panel would be
reviewing this module's opinion of the work rather than the work.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UserInputError
from src.core.evidence import (
    EVIDENCE_FIELDS, EVIDENCE_STATUSES, EvidenceBundle, create_evidence_bundle,
)

#: Keys a payload may not carry. A bundle holding one of these reads back as though the record
#: asserted a claim level, whatever the ladder computed (R22). Mirrored from the API's own list
#: so a bundle built here is refused for the same reasons a posted one would be.
RESERVED_PAYLOAD_KEYS: Tuple[str, ...] = (
    "rung", "rung_index", "claim_rung", "claim_level", "unblocked_rung", "claim_ladder",
)


class RegistrationNotEstablished(UserInputError):
    """The hypothesis cannot be shown to predate the evidence, so no bundle is built."""

    status_code = 400
    client_safe = True


def digest_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _git(*arguments: str) -> str:
    """Run git and return stdout, or an empty string if git cannot answer."""
    try:
        finished = subprocess.run(("git",) + arguments, capture_output=True, text=True,
                                  timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return finished.stdout.strip() if finished.returncode == 0 else ""


@dataclass(frozen=True)
class FileMoment:
    """When a file entered the repository, and whether it still matches what was committed."""

    path: str
    tracked: bool
    commit: Optional[str]
    committed_at: Optional[str]
    modified_since_commit: bool
    sha256: Optional[str]

    def describe(self) -> Dict[str, Any]:
        return {"path": self.path, "tracked": self.tracked, "commit": self.commit,
                "committed_at": self.committed_at,
                "modified_since_commit": self.modified_since_commit, "sha256": self.sha256}


def first_commit(path: Path) -> FileMoment:
    """The commit that first added this file, with whether the working copy still matches it.

    The FIRST commit, not the latest: a declaration amended after a measurement still registered
    its terms when it was written, and using the latest commit would make every later typo look
    like a back-dated hypothesis.
    """
    text = str(path).replace("\\", "/")
    if not path.is_file():
        return FileMoment(text, False, None, None, False, None)
    tracked = _git("ls-files", "--error-unmatch", text) != ""
    if not tracked:
        return FileMoment(text, False, None, None, False, digest_of(path))
    line = _git("log", "--diff-filter=A", "--format=%H %cI", "--", text)
    commit = committed_at = None
    if line:
        # A file added, deleted and re-added has several; the earliest is the registration.
        first = line.splitlines()[-1].split()
        if len(first) == 2:
            commit, committed_at = first[0], first[1]
    dirty = _git("status", "--porcelain", "--", text) != ""
    return FileMoment(text, True, commit, committed_at, dirty, digest_of(path))


@dataclass(frozen=True)
class Registration:
    """Whether a declaration can be shown to predate the measurement it registered."""

    declaration: FileMoment
    measurement: FileMoment
    established: bool
    refusal: Optional[str]

    def describe(self) -> Dict[str, Any]:
        return {"declaration": self.declaration.describe(),
                "measurement": self.measurement.describe(),
                "established": self.established, "refusal": self.refusal,
                "how_this_is_established": (
                    "by the git history, not by a field either file could state about itself. "
                    "The commit that first added the declaration must precede the commit that "
                    "first added the measurement, and both working copies must still match what "
                    "was committed.")}

    def require(self) -> "Registration":
        if not self.established:
            raise RegistrationNotEstablished(
                self.refusal or "registration could not be established",
                declaration=self.declaration.path, measurement=self.measurement.path)
        return self


def check_registration(declaration: Path, measurement: Path) -> Registration:
    """Prove from git that the hypothesis was registered before the evidence, or refuse.

    Every refusal names the file and what would lift it, because a researcher told only that a
    bundle could not be built learns nothing about how to build one.
    """
    first = first_commit(Path(declaration))
    second = first_commit(Path(measurement))

    def refuse(reason: str) -> Registration:
        return Registration(first, second, False, reason)

    for moment, role in ((first, "declaration"), (second, "measurement")):
        if not moment.tracked:
            return refuse(
                "the %s %s is not tracked by git, so nothing fixes when it came into existence. "
                "A hypothesis that cannot be dated cannot be shown to predate its evidence. "
                "Commit it, and if it was written after the measurement say so rather than "
                "bundling it." % (role, moment.path))
        if moment.commit is None:
            return refuse(
                "git reports no commit that added the %s %s, so its registration moment cannot "
                "be read. This usually means the history was rewritten; re-derive the ordering "
                "from whatever record of that rewrite exists rather than assuming it."
                % (role, moment.path))
        if moment.modified_since_commit:
            return refuse(
                "the %s %s has uncommitted changes, so the file on disk is not the file whose "
                "date git reports. Commit or revert it; a bundle built from an edited "
                "declaration would carry a registration time for text that was never registered."
                % (role, moment.path))

    if first.commit == second.commit:
        return refuse(
            "the declaration %s and the measurement %s entered the repository in the SAME "
            "commit (%s), so git cannot separate them and the ordering is unprovable. The "
            "declaration may well have been written first -- most of this programme's were -- "
            "but a bundle asserting so would be asserting something nothing checks, which is "
            "the whole failure this check exists to prevent. What lifts it, for future work: "
            "commit the declaration on its own, before the run, as T4E.28 did. Nothing lifts it "
            "retrospectively."
            % (first.path, second.path, first.commit[:7]))

    if first.committed_at >= second.committed_at:
        return refuse(
            "the declaration %s was committed at %s, which is NOT before the measurement %s at "
            "%s. Registering a hypothesis onto evidence that already exists manufactures a "
            "preregistration, and a panel reading such a bundle would be arguing about a risk "
            "nobody took. This record cannot be bundled on these terms. What would lift it: "
            "nothing retrospective -- the ordering is a fact about the past. A study whose "
            "declaration genuinely came first can be bundled; this one must be discussed as "
            "what it is."
            % (first.path, first.committed_at, second.path, second.committed_at))

    return Registration(first, second, True, None)


@dataclass(frozen=True)
class EvidenceClaim:
    """One entry a caller asserts a measurement yields, in their words and at their risk.

    This is the unit of scientific judgement in this module, and it is deliberately not derived.
    Reading a record and inferring that some number is a null result or a contradiction would
    make the panel a reviewer of this code's opinion rather than of the work.
    """

    category: str
    label: str
    status: str
    summary: str
    payload: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in EVIDENCE_FIELDS:
            raise InvalidParameterError(
                "EvidenceClaim.category", self.category,
                "one of %s. Commentary is not scientific evidence" % ", ".join(EVIDENCE_FIELDS))
        if self.status not in EVIDENCE_STATUSES:
            raise InvalidParameterError("EvidenceClaim.status", self.status,
                                        "one of %s" % ", ".join(EVIDENCE_STATUSES))
        if not str(self.label).strip():
            raise InvalidParameterError("EvidenceClaim.label", self.label, "a non-empty label")
        if not str(self.summary).strip():
            raise InvalidParameterError(
                "EvidenceClaim.summary", self.summary,
                "a non-empty summary. An entry a reader cannot understand without opening the "
                "payload is an entry a panel will argue about without reading")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise InvalidParameterError(
                "EvidenceClaim.payload", self.payload,
                "a non-empty mapping of the figures behind the summary")
        offending = sorted(set(self.payload) & set(RESERVED_PAYLOAD_KEYS))
        if offending:
            raise InvalidParameterError(
                "EvidenceClaim.payload", offending,
                "a payload with no claim-level key. A bundle carrying %s reads back as though "
                "the record asserted a rung, whatever the ladder computed (R22)"
                % ", ".join(offending))


def bundle_from_measurement(*, study_id: str, hypothesis_id: str, statement: str,
                            prediction: str, declaration: Path, measurement: Path,
                            claims: Sequence[EvidenceClaim],
                            registered_at: Optional[str] = None,
                            created_at: Optional[str] = None) -> Tuple[EvidenceBundle,
                                                                       Registration]:
    """An evidence bundle whose hypothesis is dated by the commit that registered it.

    `registered_at` and `created_at` default to the two commit times, so the bundle's own
    chronology is the repository's rather than the clock of whoever ran the tool.
    """
    if not claims:
        raise InvalidParameterError(
            "claims", claims,
            "at least one EvidenceClaim. A bundle with a hypothesis and no evidence asserts "
            "that a question was asked and answered by nothing")
    registration = check_registration(Path(declaration), Path(measurement)).require()

    bundle = create_evidence_bundle(
        study_id=study_id, hypothesis_id=hypothesis_id, statement=statement,
        prediction=prediction,
        registered_at=registered_at or registration.declaration.committed_at,
        created_at=created_at or registration.measurement.committed_at,
        provenance={
            "declaration": registration.declaration.describe(),
            "measurement": registration.measurement.describe(),
            "registration_established_by": (
                "the git history: the commit that first added the declaration precedes the "
                "commit that first added the measurement, and both working copies still match "
                "what was committed"),
            "what_this_provenance_does_not_say": (
                "that the declaration was wise, that the measurement was well designed, or that "
                "the claims below are the right reading of it. It says only that the question "
                "was fixed before the answer existed."),
        })

    sources = tuple(digest for digest in
                    (registration.declaration.sha256, registration.measurement.sha256)
                    if digest)
    for claim in claims:
        bundle = bundle.append(
            claim.category, label=claim.label, status=claim.status, summary=claim.summary,
            recorded_at=registration.measurement.committed_at,
            payload=json.loads(json.dumps(dict(claim.payload))),
            source_sha256s=sources)
    return bundle, registration
