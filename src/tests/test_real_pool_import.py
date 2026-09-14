import hashlib
import json

import pytest

from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from tools.import_g17_partner_profiles import main


def _write_review(root):
    path = root / "g17-marginal-method-review.json"
    if path.exists():
        return path
    path.write_text(json.dumps({
        "schema": "g17-marginal-method-review/v1",
        "study_id": "g17_scale_shape_successor_v1",
        "native_seconds_basis": "catalogue period field v2",
        "effective_sample_size_method": "AR(1) correction, analysis note G17-R12",
        "noise_floor_method": "residual SD / total SD, analysis note G17-N1",
        "applicability": "Delimited scalar records with increasing timestamps.",
        "limitations": "Review does not establish record exchangeability.",
        "claim_boundary": "Adoption approves methods, not profiles or pools.",
    }), encoding="utf-8")
    sign_declaration(
        root, path.name, adopted_by="Ada Reviewer", adopted_as="G17 method reviewer",
        what_was_adopted="The declared single-record marginal methods.",
        affirmation=REQUIRED_AFFIRMATION, today="2026-09-12")
    return path


def _declaration(record_id, provenance_key):
    return {
        "record_id": record_id,
        "provenance_key": provenance_key,
        "time_column": "elapsed_seconds",
        "value_column": "signal",
        "time_units": "seconds",
        "window_seconds": 20.0,
        "native_seconds": 3600.0,
        "native_seconds_basis": "catalogue period field v2",
        "effective_sample_size": 2.0,
        "effective_sample_size_method": "AR(1) correction, analysis note G17-R12",
        "noise_floor": 0.1,
        "noise_floor_method": "residual SD / total SD, analysis note G17-N1",
    }


def _write_manifest(root, *, bad_second_digest=False):
    _write_review(root)
    entries = []
    for index in range(2):
        source = root / ("record-%d.csv" % index)
        source.write_text(
            "elapsed_seconds,signal\n0,%d\n10,%d\n20,%d\n" %
            (index, index + 1, index + 2), encoding="utf-8")
        declaration = root / ("record-%d.declaration.json" % index)
        declaration.write_text(json.dumps(
            _declaration("record-%d" % index, "archive/product-%d" % index)),
            encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if bad_second_digest and index == 1:
            digest = "0" * 64
        entries.append({
            "source": source.name,
            "source_sha256": digest,
            "declaration": declaration.name,
            "output": "profiles/record-%d.partner-profile.json" % index,
            "delimiter": ",",
        })
    manifest = root / "g17-profile-import.json"
    manifest.write_text(json.dumps({
        "schema": "g17-profile-import-manifest/v1",
        "method_review": "g17-marginal-method-review.json",
        "entries": entries,
    }), encoding="utf-8")
    return manifest


def test_manifest_import_preflights_then_writes_source_bound_profiles(tmp_path):
    manifest = _write_manifest(tmp_path)

    assert main([str(manifest)]) == 0

    outputs = sorted((tmp_path / "profiles").glob("*.partner-profile.json"))
    assert len(outputs) == 2
    first = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert first["schema"] == "correspondence-record-profile/v3"
    assert first["source"]["content_sha256"] == hashlib.sha256(
        (tmp_path / "record-0.csv").read_bytes()).hexdigest()


def test_manifest_import_emits_nothing_when_any_source_digest_has_drifted(tmp_path):
    manifest = _write_manifest(tmp_path, bad_second_digest=True)

    with pytest.raises(SystemExit, match="2"):
        main([str(manifest)])

    assert not (tmp_path / "profiles").exists()


def test_manifest_paths_must_be_relative_and_outputs_are_immutable(tmp_path):
    manifest = _write_manifest(tmp_path)
    body = json.loads(manifest.read_text(encoding="utf-8"))
    body["entries"][0]["source"] = str((tmp_path / "record-0.csv").resolve())
    manifest.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(manifest)])

    valid_manifest = _write_manifest(tmp_path)
    assert main([str(valid_manifest)]) == 0
    with pytest.raises(SystemExit, match="2"):
        main([str(valid_manifest)])