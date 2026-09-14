"""Preflighted import of legitimate local records into G17 profile inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from src.core.errors import UserInputError
from src.core.real_pool_ingress import RecordProfileDeclaration, build_source_bound_profile
from src.core.real_pool_method_review import load_adopted_method_review


SCHEMA = "g17-profile-import-manifest/v1"
_ENTRY_FIELDS = {"source", "source_sha256", "declaration", "output", "delimiter"}


def _confined_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UserInputError("G17 import %s must be a non-empty relative path" % name)
    relative = Path(value)
    if relative.is_absolute():
        raise UserInputError("G17 import %s must be relative to the manifest" % name)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise UserInputError("G17 import %s escapes the manifest directory" % name)
    return path


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise UserInputError("G17 import %s must be a lowercase SHA-256 digest" % name)
    return value


def _load_object(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserInputError("G17 import %s is unreadable: %s" % (name, error)) from error
    if not isinstance(value, dict):
        raise UserInputError("G17 import %s must contain a JSON object" % name)
    return value


def preflight_profile_import(manifest_path: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Validate a complete manifest and build every profile without writing outputs."""
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = _load_object(manifest_path, "manifest")
    if set(manifest) != {"schema", "method_review", "entries"} \
            or manifest.get("schema") != SCHEMA:
        raise UserInputError("G17 import manifest must be an exact %s object" % SCHEMA)
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise UserInputError("G17 import manifest entries must be a non-empty list")

    review_path = _confined_path(root, manifest["method_review"], "method_review")
    method_review = load_adopted_method_review(review_path)
    planned: List[Tuple[Path, Dict[str, Any]]] = []
    sources = set()
    source_digests = set()
    outputs = set()
    record_ids = set()
    provenance_keys = set()

    for index, entry in enumerate(entries):
        label = "entry %d" % (index + 1)
        if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
            raise UserInputError("G17 import %s has invalid fields" % label)
        source = _confined_path(root, entry["source"], "%s source" % label)
        declaration_path = _confined_path(
            root, entry["declaration"], "%s declaration" % label)
        output = _confined_path(root, entry["output"], "%s output" % label)
        if not output.name.endswith(".partner-profile.json"):
            raise UserInputError("G17 import %s output must end with .partner-profile.json" % label)
        if output.exists():
            raise UserInputError("G17 import output already exists; profiles are immutable")
        if source in sources or output in outputs:
            raise UserInputError("G17 import source and output paths must be unique")

        try:
            payload = source.read_bytes()
        except OSError as error:
            raise UserInputError("G17 import %s source is unreadable: %s" % (label, error)) from error
        actual_digest = hashlib.sha256(payload).hexdigest()
        expected_digest = _digest(entry["source_sha256"], "%s source_sha256" % label)
        if actual_digest != expected_digest:
            raise UserInputError("G17 import %s source digest does not match exact bytes" % label)
        if actual_digest in source_digests:
            raise UserInputError("G17 import source bytes must be unique")

        raw_declaration = _load_object(declaration_path, "%s declaration" % label)
        try:
            declaration = RecordProfileDeclaration(**raw_declaration)
        except TypeError as error:
            raise UserInputError("G17 import %s declaration is invalid: %s" % (label, error)) from error
        if declaration.record_id in record_ids or declaration.provenance_key in provenance_keys:
            raise UserInputError("G17 import record and provenance identities must be unique")
        profile = build_source_bound_profile(
            payload, filename=source.name, delimiter=entry["delimiter"],
            declaration=declaration, method_review=method_review)

        sources.add(source)
        source_digests.add(actual_digest)
        outputs.add(output)
        record_ids.add(declaration.record_id)
        provenance_keys.add(declaration.provenance_key)
        planned.append((output, profile))
    return planned


def import_partner_profiles(manifest_path: Path) -> Dict[str, Any]:
    """Write a fully preflighted manifest, rolling back outputs on write failure."""
    planned = preflight_profile_import(manifest_path)
    staged: List[Tuple[Path, Path]] = []
    committed: List[Path] = []
    try:
        for output, profile in planned:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(".%s.%s.tmp" % (output.name, uuid4().hex))
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(profile, indent=2) + "\n")
            staged.append((temporary, output))
        for temporary, output in staged:
            if os.name == "nt":
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
                temporary.unlink()
            committed.append(output)
    except OSError as error:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for output in committed:
            output.unlink(missing_ok=True)
        raise UserInputError("G17 import could not commit its complete batch: %s" % error) from error

    return {
        "schema": SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_path.resolve().read_bytes()).hexdigest(),
        "profile_count": len(planned),
        "outputs": [str(output.relative_to(manifest_path.resolve().parent).as_posix())
                    for output, _ in planned],
        "claim_boundary": (
            "Import proves exact local bytes were profiled under adopted methods; it does not "
            "establish record admission, pool exchangeability or a scientific result."),
    }


__all__ = ["SCHEMA", "import_partner_profiles", "preflight_profile_import"]