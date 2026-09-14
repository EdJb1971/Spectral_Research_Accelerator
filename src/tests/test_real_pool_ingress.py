import hashlib
import json

import pytest

from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import UserInputError
from src.core.real_pool_ingress import RecordProfileDeclaration, build_source_bound_profile
from src.core.real_pool_method_review import load_adopted_method_review
from tools.build_g17_partner_profile import main


def declaration(**changes):
    values = {
        "record_id": "record-a",
        "provenance_key": "archive-a/product-7",
        "time_column": "elapsed_seconds",
        "value_column": "signal",
        "time_units": "seconds",
        "window_seconds": 40.0,
        "native_seconds": 3600.0,
        "native_seconds_basis": "catalogue period field v2",
        "effective_sample_size": 3.0,
        "effective_sample_size_method": "AR(1) correction, analysis note G17-R12",
        "noise_floor": 0.1,
        "noise_floor_method": "residual SD / total SD, analysis note G17-N1",
    }
    values.update(changes)
    return RecordProfileDeclaration(**values)


def adopted_review(tmp_path, **changes):
    body = {
        "schema": "g17-marginal-method-review/v1",
        "study_id": "g17_scale_shape_successor_v1",
        "native_seconds_basis": "catalogue period field v2",
        "effective_sample_size_method": "AR(1) correction, analysis note G17-R12",
        "noise_floor_method": "residual SD / total SD, analysis note G17-N1",
        "applicability": "Delimited scalar records with increasing timestamps.",
        "limitations": "Review does not establish record exchangeability.",
        "claim_boundary": "Adoption approves these methods, not any resulting profile or pool.",
    }
    body.update(changes)
    path = tmp_path / "g17-marginal-method-review.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    sign_declaration(
        tmp_path, path.name, adopted_by="Ada Reviewer", adopted_as="G17 method reviewer",
        what_was_adopted="The declared single-record marginal methods and their limitations.",
        affirmation=REQUIRED_AFFIRMATION, today="2026-09-12")
    return path, load_adopted_method_review(path)


def test_profile_is_bound_to_exact_source_and_derives_only_storage_facts(tmp_path):
    payload = b"elapsed_seconds,signal\n0,1\n10,2\n20,3\n40,4\n"
    _, review = adopted_review(tmp_path)
    result = build_source_bound_profile(
        payload, filename="actual.csv", delimiter=",", declaration=declaration(),
        method_review=review)

    assert result["source"]["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["profile"]["n_samples"] == 4
    assert result["profile"]["cadence_seconds"] == 10.0
    assert result["profile"]["coverage_fraction"] == pytest.approx(0.75)
    assert result["profile"]["effective_sample_size"] == 3.0
    assert result["derivation"]["effective_sample_size"].startswith("AR(1)")
    assert result["method_review"]["adopted_by"] == "Ada Reviewer"


@pytest.mark.parametrize("payload, message", [
    (b"elapsed_seconds,signal\n0,1\n", "at least two"),
    (b"elapsed_seconds,signal\n0,1\n0,2\n", "strictly increasing"),
    (b"elapsed_seconds,other\n0,1\n1,2\n", "missing declared columns"),
    (b"elapsed_seconds,signal\n0,1\n1,nan\n", "non-finite"),
    (b"elapsed_seconds,signal,signal\n0,1,2\n1,2,3\n", "column names must be unique"),
])
def test_unusable_source_record_is_refused(tmp_path, payload, message):
    _, review = adopted_review(tmp_path)
    with pytest.raises(UserInputError, match=message):
        build_source_bound_profile(
            payload, filename="actual.csv", delimiter=",", declaration=declaration(),
            method_review=review)


def test_scientific_marginals_require_named_methods(tmp_path):
    with pytest.raises(Exception, match="noise_floor_method"):
        declaration(noise_floor_method=" ")

    _, review = adopted_review(tmp_path)
    with pytest.raises(Exception, match="window_seconds"):
        build_source_bound_profile(
            b"elapsed_seconds,signal\n0,1\n50,2\n", filename="fragment.csv", delimiter=",",
            declaration=declaration(window_seconds=40.0), method_review=review)


def test_profile_methods_must_exactly_match_the_adopted_review(tmp_path):
    _, review = adopted_review(tmp_path)
    with pytest.raises(UserInputError, match="exactly match"):
        build_source_bound_profile(
            b"elapsed_seconds,signal\n0,1\n10,2\n", filename="actual.csv", delimiter=",",
            declaration=declaration(noise_floor_method="an unreviewed alternative"),
            method_review=review)


def test_cli_refuses_scientific_methods_without_an_adopted_review(tmp_path):
    source = tmp_path / "actual.csv"
    source.write_text("elapsed_seconds,signal\n0,1\n10,2\n20,3\n", encoding="utf-8")
    declared = tmp_path / "declaration.json"
    declared.write_text(json.dumps(declaration().__dict__), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([str(source), str(declared), str(tmp_path / "record-a.partner-profile.json")])


def test_cli_writes_one_immutable_profile(tmp_path):
    source = tmp_path / "actual.csv"
    source.write_text("elapsed_seconds,signal\n0,1\n10,2\n20,3\n", encoding="utf-8")
    declared = tmp_path / "declaration.json"
    declared.write_text(json.dumps(declaration().__dict__), encoding="utf-8")
    output = tmp_path / "record-a.partner-profile.json"
    review_path, _ = adopted_review(tmp_path)
    args = [str(source), str(declared), str(output), "--method-review", str(review_path)]

    assert main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["profile"]["record_id"] == "record-a"
    with pytest.raises(SystemExit):
        main(args)

    invalid = dict(declaration().__dict__)
    invalid["time_units"] = "days"
    invalid_declaration = tmp_path / "invalid.json"
    invalid_declaration.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(source), str(invalid_declaration),
              str(tmp_path / "invalid.partner-profile.json"),
              "--method-review", str(review_path)])


def test_cli_refuses_a_review_changed_after_adoption(tmp_path):
    source = tmp_path / "actual.csv"
    source.write_text("elapsed_seconds,signal\n0,1\n10,2\n", encoding="utf-8")
    declared = tmp_path / "declaration.json"
    declared.write_text(json.dumps(declaration(effective_sample_size=2.0).__dict__), encoding="utf-8")
    review_path, _ = adopted_review(tmp_path)
    review_path.write_text(review_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main([str(source), str(declared), str(tmp_path / "record-a.partner-profile.json"),
              "--method-review", str(review_path)])