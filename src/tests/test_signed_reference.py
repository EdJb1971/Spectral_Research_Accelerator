"""A signed external reference resolves by digest or refuses by name.

The failure this module exists to prevent is silence: a file that is absent, truncated, or
quietly a different edition of a living archive, discovered as a confusing result rather than as
a refusal. Every test here is about a way that could happen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data_layer.signed_reference import (
    REFERENCES,
    SignedReference,
    SignedReferenceUnavailable,
    digest_of,
    resolve,
)

PAYLOAD = b"SID,SEASON,BASIN\n2018001S10135,2018,SP\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _fixture(tmp_path: Path, *, payload: bytes = PAYLOAD, sha: str = DIGEST,
             size: int = len(PAYLOAD), sign: bool = True,
             write_data: bool = True) -> SignedReference:
    data = tmp_path / "reference.csv"
    if write_data:
        data.write_bytes(payload)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "the_catalogue": {
            "sha256": sha, "bytes": size,
            "source_url": "https://example.invalid/reference.csv",
            "citation_required_by_the_publisher": ["Somebody et al. (2024)"],
        }}), encoding="utf-8")
    signature = None
    if sign:
        signature = tmp_path / "signature.json"
        signature.write_text(json.dumps({"signs_sha256": digest_of(design)}), encoding="utf-8")
    return SignedReference(name="fixture", path=data, design_path=design,
                           signature_path=signature)


# ---------------- the happy path, and the real reference


def test_a_matching_file_resolves_with_every_binding_checked(tmp_path):
    result = _fixture(tmp_path).resolve()

    assert result.available is True
    assert result.refusal is None
    assert result.signature_verified is True
    assert result.observed_sha256 == DIGEST
    assert result.require() == tmp_path / "reference.csv"


def test_the_citation_travels_with_the_resolution(tmp_path):
    """Open access is granted in exchange for the citation; a loader that drops it loses the
    only obligation the licence imposes."""
    assert _fixture(tmp_path).resolve().citation == ("Somebody et al. (2024)",)


def test_the_registry_knows_the_signed_catalogue():
    assert "ibtracs_sp_v04r01" in REFERENCES
    reference = REFERENCES["ibtracs_sp_v04r01"]
    assert reference.path == Path("data/catalogues/ibtracs.SP.list.v04r01.csv")
    # The path lives here because adding it to the signed design would break the sha256 the
    # signature rests on.
    assert "t4e17" in str(reference.design_path)


def test_an_unknown_reference_is_refused_with_the_known_names():
    with pytest.raises(SignedReferenceUnavailable) as caught:
        resolve("not_a_reference")
    assert "ibtracs_sp_v04r01" in str(caught.value)


# ---------------- absence: the case that actually happened


def test_an_absent_file_names_the_path_the_url_and_the_digest(tmp_path):
    """A reader who has lost the file learns where to get it and what it must hash to."""
    result = _fixture(tmp_path, write_data=False).resolve()

    assert result.available is False
    assert "reference.csv" in result.refusal
    assert "https://example.invalid/reference.csv" in result.refusal
    assert DIGEST in result.refusal
    # And that replacing it is not a copy.
    assert "declaration rather than a copy" in result.refusal


def test_require_raises_by_name_rather_than_returning_an_unverified_path(tmp_path):
    with pytest.raises(SignedReferenceUnavailable):
        _fixture(tmp_path, write_data=False).resolve().require()


# ---------------- the wrong file, named before it is hashed


def test_a_truncated_file_is_named_by_its_size_without_hashing(tmp_path):
    result = _fixture(tmp_path, size=999_999).resolve()

    assert result.available is False
    assert "999999" in result.refusal.replace(",", "")
    assert "truncated or partial" in result.refusal
    # The cheap check fired, so no digest was taken.
    assert result.observed_sha256 is None


def test_a_different_edition_is_refused_as_different_and_not_as_damaged(tmp_path):
    """A living archive revises earlier entries, so a digest mismatch is a different catalogue
    and the signed population, base rates and claim boundary do not extend to it."""
    other = b"SID,SEASON,BASIN\n2018001S10135,2018,SP\n2019002S11140,2019,SP\n"
    result = _fixture(tmp_path, payload=other, size=len(other)).resolve()

    assert result.available is False
    assert "DIFFERENT reference" in result.refusal
    assert "unsigned catalogue" in result.refusal
    assert result.observed_sha256 == hashlib.sha256(other).hexdigest()


def test_a_design_recording_no_digest_refuses_rather_than_admitting_anything(tmp_path):
    result = _fixture(tmp_path, sha="").resolve()

    assert result.available is False
    assert "no sha256" in result.refusal


# ---------------- the link nobody checks: has the design been edited since it was signed?


def test_a_design_edited_after_signature_is_refused(tmp_path):
    reference = _fixture(tmp_path)
    design = json.loads(reference.design_path.read_text(encoding="utf-8"))
    design["the_catalogue"]["bytes"] = 1
    reference.design_path.write_text(json.dumps(design), encoding="utf-8")

    result = reference.resolve()

    assert result.available is False
    assert result.signature_verified is False
    assert "edited since it was signed" in result.refusal
    assert "re-given" in result.refusal


def test_a_missing_signature_is_a_refusal_and_not_a_pass(tmp_path):
    reference = _fixture(tmp_path)
    reference.signature_path.unlink()

    result = reference.resolve()
    assert result.available is False
    assert "signature" in result.refusal


def test_a_signature_with_no_digest_cannot_attest_anything(tmp_path):
    reference = _fixture(tmp_path)
    reference.signature_path.write_text(json.dumps({"signed_by": "someone"}), encoding="utf-8")

    result = reference.resolve()
    assert result.available is False
    assert "records no digest" in result.refusal


def test_an_unsigned_design_is_allowed_and_reported_as_unverified(tmp_path):
    """An engineering design need not carry a signature; an absent one is not a failed one."""
    result = _fixture(tmp_path, sign=False).resolve()

    assert result.available is True
    assert result.signature_verified is None


def test_a_missing_design_is_raised_because_there_is_nothing_to_check_against(tmp_path):
    reference = _fixture(tmp_path)
    reference.design_path.unlink()

    with pytest.raises(SignedReferenceUnavailable):
        reference.resolve()


# ---------------- the cheap path


def test_skipping_the_digest_still_checks_presence_and_size(tmp_path):
    result = _fixture(tmp_path).resolve(verify_digest=False)
    assert result.available is True
    assert result.observed_sha256 is None

    truncated = _fixture(tmp_path, size=12).resolve(verify_digest=False)
    assert truncated.available is False


def test_digest_of_streams_a_file_without_reading_it_whole(tmp_path):
    path = tmp_path / "big.bin"
    payload = b"x" * (3 * (1 << 20) + 7)
    path.write_bytes(payload)
    assert digest_of(path) == hashlib.sha256(payload).hexdigest()
