"""A signed external reference, resolved by digest or refused by name.

**The gap this closes.** T4E.17 signed the IBTrACS catalogue on terms naming a specific
population, bound it by sha256, and deliberately did not commit 35.5 MB of third-party data. What
the signed design records is the URL, the digest and the byte count. What it records **no** path,
so nothing in this repository knew where to look. The consequence was not theoretical: the file
spent a day in a session scratchpad under `%TEMP%`, `tools/restate_position_acceptance.py` could
only report that T4E.27's condition 2 was unevaluable because the file was absent, and losing it
would have voided the signature rather than merely costing a download.

That is the lesson T4E.27 drew about a catalogue radius of `0.00`, applied to a file instead of a
field: **a missing input should be refused by name, not discovered later.**

**Three bindings, checked in order, because a chain is only as strong as the link nobody checks.**

1. The **signature** against the **design**. `t4e17-...-signature.json` carries `signs_sha256`. If
   the design has been edited since it was signed, the terms recorded are not the terms signed,
   and every number downstream inherits that drift.
2. The **design** against the **data**. The design carries the digest the data must reproduce.
3. The **byte count**, as a cheap pre-check that names a truncated or partial download before
   thirty-five megabytes are hashed to say the same thing.

**A refusal is a result here, not an exception by default.** `resolve()` returns a record saying
what failed and what would lift it; `require()` raises for callers that cannot proceed. Both
carry the source URL and the expected digest, so a reader who has lost the file learns where to
get it and what it must hash to in the same breath as learning it is gone.

**What this module will not do.** It will not download anything -- nothing here reaches a network,
and acquiring the file is the maintainer's act. It will not fall back to an unverified copy, and
it will not accept a file whose digest differs: a living archive like IBTrACS revises earlier
entries, so a different digest is a *different catalogue* on which no signature holds, and
quietly using it would be the quietest possible wrong number.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.errors import UserInputError

#: Read in chunks so a 35 MB catalogue does not arrive in memory whole.
_CHUNK = 1 << 20


class SignedReferenceUnavailable(UserInputError):
    """A signed external reference is absent, altered, or no longer the one that was signed."""

    status_code = 400
    client_safe = True


def digest_of(path: Path) -> str:
    """sha256 of a file, streamed."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


@dataclass(frozen=True)
class Resolution:
    """What a resolution attempt found, including what it refused and why."""

    name: str
    path: Path
    available: bool
    refusal: Optional[str]
    expected_sha256: str
    observed_sha256: Optional[str]
    expected_bytes: Optional[int]
    observed_bytes: Optional[int]
    source_url: Optional[str]
    citation: Tuple[str, ...]
    signature_verified: Optional[bool]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "available": self.available,
            "refusal": self.refusal,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "expected_bytes": self.expected_bytes,
            "observed_bytes": self.observed_bytes,
            "source_url": self.source_url,
            "signature_verified": self.signature_verified,
            "citation_required": list(self.citation),
        }

    def require(self) -> Path:
        """The path, or a refusal raised by name. Never an unverified file."""
        if not self.available:
            raise SignedReferenceUnavailable(
                self.refusal or "the reference %r could not be resolved" % self.name,
                reference=self.name, path=str(self.path),
                expected_sha256=self.expected_sha256,
                observed_sha256=self.observed_sha256,
                source_url=self.source_url)
        return self.path


@dataclass(frozen=True)
class SignedReference:
    """A file some declaration signed, and the design that records its terms."""

    name: str
    path: Path
    design_path: Path
    #: Where in the design the digest, byte count, URL and citation live.
    design_section: str = "the_catalogue"
    signature_path: Optional[Path] = None

    def _design(self) -> Dict[str, Any]:
        if not self.design_path.exists():
            raise SignedReferenceUnavailable(
                "the design that records this reference's terms is missing at %s. Without it "
                "there is no digest to check against and nothing to resolve"
                % self.design_path, reference=self.name)
        return json.loads(self.design_path.read_text(encoding="utf-8"))

    def _signature_state(self) -> Tuple[Optional[bool], Optional[str]]:
        """Has the design been edited since it was signed?

        `None` when no signature file is declared -- an engineering design need not carry one,
        and an absent signature is not a failed one.
        """
        if self.signature_path is None:
            return None, None
        if not self.signature_path.exists():
            return False, ("the signature at %s is missing, so nothing attests that the design "
                           "recording these terms is the one that was signed"
                           % self.signature_path)
        signature = json.loads(self.signature_path.read_text(encoding="utf-8"))
        expected = signature.get("signs_sha256")
        if not expected:
            return False, ("the signature at %s records no digest for the design it signs, so "
                           "the terms signed cannot be distinguished from the terms recorded"
                           % self.signature_path)
        observed = digest_of(self.design_path)
        if observed != expected:
            return False, (
                "the design at %s has been edited since it was signed: the signature attests "
                "sha256 %s and the file now hashes to %s. The terms recorded are not the terms "
                "signed, so nothing downstream may rely on them until the signature is re-given"
                % (self.design_path, expected, observed))
        return True, None

    def resolve(self, *, verify_digest: bool = True) -> Resolution:
        """Find the file and check every binding, or say precisely which one failed."""
        design = self._design()
        section = design.get(self.design_section, {})
        expected = str(section.get("sha256", ""))
        expected_bytes = section.get("bytes")
        url = section.get("source_url")
        citation = tuple(section.get("citation_required_by_the_publisher", ()) or ())

        def refuse(reason: str, *, observed: Optional[str] = None,
                   observed_bytes: Optional[int] = None,
                   signature: Optional[bool] = None) -> Resolution:
            return Resolution(
                name=self.name, path=self.path, available=False, refusal=reason,
                expected_sha256=expected, observed_sha256=observed,
                expected_bytes=expected_bytes, observed_bytes=observed_bytes,
                source_url=url, citation=citation, signature_verified=signature)

        signature_ok, signature_refusal = self._signature_state()
        if signature_refusal is not None:
            return refuse(signature_refusal, signature=signature_ok)

        if not expected:
            return refuse(
                "the design at %s records no sha256 for %r, so there is no binding to check and "
                "an unverified file must not be used in its place"
                % (self.design_path, self.design_section), signature=signature_ok)

        if not self.path.exists():
            return refuse(
                "the signed reference %r is not at %s. It is bound by sha256 rather than "
                "committed, so it must be placed there by hand. Retrieve it from %s; it must "
                "hash to %s, and a copy that does not is a DIFFERENT reference on which the "
                "signature does not hold -- a living archive revises earlier entries, so a "
                "fresh download reproducing the digest is not guaranteed and replacing this "
                "file is a declaration rather than a copy"
                % (self.name, self.path, url or "the source recorded in the design", expected),
                signature=signature_ok)

        observed_bytes = self.path.stat().st_size
        if expected_bytes is not None and int(observed_bytes) != int(expected_bytes):
            # Named before hashing: a truncated download is the common case, and thirty-five
            # megabytes of sha256 to reach the same conclusion is a slower way to say it.
            return refuse(
                "the file at %s is %d bytes and the design records %d. It is not the signed "
                "reference -- most likely a truncated or partial retrieval. Re-fetch from %s"
                % (self.path, observed_bytes, int(expected_bytes),
                   url or "the recorded source"),
                observed_bytes=int(observed_bytes), signature=signature_ok)

        if not verify_digest:
            return Resolution(
                name=self.name, path=self.path, available=True,
                refusal=None, expected_sha256=expected, observed_sha256=None,
                expected_bytes=expected_bytes, observed_bytes=int(observed_bytes),
                source_url=url, citation=citation, signature_verified=signature_ok)

        observed = digest_of(self.path)
        if observed != expected:
            return refuse(
                "the file at %s does not hash to the signed digest: expected %s, found %s. This "
                "is a DIFFERENT reference from the one that was signed, not a damaged copy of "
                "it, and the signed terms -- the population, the base rates, the claim boundary "
                "-- do not extend to it. Using it would evaluate against an unsigned catalogue"
                % (self.path, expected, observed),
                observed=observed, observed_bytes=int(observed_bytes), signature=signature_ok)

        return Resolution(
            name=self.name, path=self.path, available=True, refusal=None,
            expected_sha256=expected, observed_sha256=observed,
            expected_bytes=expected_bytes, observed_bytes=int(observed_bytes),
            source_url=url, citation=citation, signature_verified=signature_ok)


#: The references this repository knows how to look for. A path recorded here is what the signed
#: design deliberately does not carry -- editing that design to add one would break the sha256
#: the signature rests on, so the path lives beside it rather than inside it.
IBTRACS_SP = SignedReference(
    name="ibtracs_sp_v04r01",
    path=Path("data/catalogues/ibtracs.SP.list.v04r01.csv"),
    design_path=Path("data/identity_calibration/t4e17-external-catalogue-design.json"),
    signature_path=Path("data/identity_calibration/t4e17-external-catalogue-signature.json"),
)

REFERENCES: Dict[str, SignedReference] = {IBTRACS_SP.name: IBTRACS_SP}


def resolve(name: str, *, verify_digest: bool = True) -> Resolution:
    """Resolve a known signed reference by name."""
    if name not in REFERENCES:
        raise SignedReferenceUnavailable(
            "no signed reference named %r; known: %s"
            % (name, ", ".join(sorted(REFERENCES))), reference=name)
    return REFERENCES[name].resolve(verify_digest=verify_digest)
