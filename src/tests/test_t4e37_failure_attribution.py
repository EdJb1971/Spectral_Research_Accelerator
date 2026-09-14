import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmarks.failure_attribution import (
    DEFAULT_DECLARATION,
    classify_failure,
    evaluate_failure_attribution,
    load_failure_declaration,
    majority_decision,
    require_current_adoption,
    sampled_local_maxima,
)
from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import DataSourceError


def test_declaration_freezes_the_13_t4e36_failures_and_no_acceptance_gate():
    declaration = load_failure_declaration()
    measurement_path = Path(declaration["source_evidence"]["wrapped_join"])
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    failed = [row for row in measurement["paths"]["raw_field"]["rows"]
              if row["inside_radius"] == 0]

    assert declaration["status"] == "DRAFTED_NOT_ADOPTED"
    assert len(failed) == declaration["frozen_population"]["expected_rows"] == 13
    assert [[row["sid"], row["time"], row["storm"]] for row in failed] == \
        declaration["frozen_population"]["identities"]
    assert measurement["receipt_sha256"] == \
        declaration["source_evidence"]["wrapped_join_receipt_sha256"]
    assert hashlib.sha256(measurement_path.read_bytes()).hexdigest() == \
        declaration["source_evidence"]["wrapped_join_file_sha256"]
    assert "Neither outcome is PASS" in \
        declaration["classification_declared_before_measurement"]["no_success_gate"]


def test_sampled_candidates_use_the_extractors_unthresholded_rule_and_boundary():
    values = np.zeros((5, 6), dtype=np.float64)
    values[0, 3] = 20.0                 # excluded outer boundary
    values[2, 2] = values[2, 3] = 9.0  # tied maxima are both retained
    latitudes = np.arange(5, dtype=np.float64)
    longitudes = np.arange(6, dtype=np.float64) + 140.0

    maxima = sampled_local_maxima(values, latitudes, longitudes)

    assert [(row["grid_index"], row["field_value"]) for row in maxima] == [
        ([2, 2], 9.0), ([2, 3], 9.0)]
    assert [row["lon"] for row in maxima] == [142.0, 143.0]


def test_sampled_candidates_refuse_nonfinite_or_misaligned_fields():
    with pytest.raises(DataSourceError, match="finite 2D field"):
        sampled_local_maxima(
            np.asarray([[1.0, np.nan], [0.0, 1.0]]),
            np.asarray([0.0, 1.0]), np.asarray([140.0, 141.0]))
    with pytest.raises(DataSourceError, match="finite 2D field"):
        sampled_local_maxima(
            np.ones((3, 3)), np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0, 3.0]))


def test_binary_classification_and_13_row_majority_are_total_without_a_pass():
    filtering = {"classification": "EXTRACTOR_FILTERING_CANDIDATE"}
    separation = {"classification": "FIELD_REFERENCE_SEPARATION_CANDIDATE"}
    decision = majority_decision([filtering] * 7 + [separation] * 6)
    assert classify_failure(1) == "EXTRACTOR_FILTERING_CANDIDATE"
    assert classify_failure(0) == "FIELD_REFERENCE_SEPARATION_CANDIDATE"
    assert decision == {
        "outcome": "EXTRACTOR_FILTERING_DOMINANT",
        "counts": {
            "EXTRACTOR_FILTERING_CANDIDATE": 7,
            "FIELD_REFERENCE_SEPARATION_CANDIDATE": 6,
        },
        "majority_needed": 7,
        "is_acceptance_verdict": False,
    }
    assert majority_decision([filtering] * 6 + [separation] * 7)["outcome"] == \
        "FIELD_REFERENCE_SEPARATION_DOMINANT"
    with pytest.raises(DataSourceError, match="exactly 13"):
        majority_decision([filtering] * 12)


def test_real_evaluation_requires_a_current_human_adoption_before_opening_record(
        tmp_path, monkeypatch):
    declaration = tmp_path / DEFAULT_DECLARATION.name
    declaration.write_bytes(DEFAULT_DECLARATION.read_bytes())
    opened = []
    monkeypatch.setattr(
        "src.benchmarks.failure_attribution.xr.open_zarr",
        lambda *args, **kwargs: opened.append((args, kwargs)))
    with pytest.raises(DataSourceError, match="has not been adopted"):
        evaluate_failure_attribution(declaration, output_path=tmp_path / "out.json")
    assert opened == []
    with pytest.raises(DataSourceError, match="has not been adopted"):
        require_current_adoption(declaration)

    sign_declaration(
        tmp_path, declaration.name,
        adopted_by="Test Human Reviewer",
        adopted_as="ADOPTED_FOR_TEST",
        what_was_adopted="the copied T4E.37 declaration",
        affirmation=REQUIRED_AFFIRMATION,
    )
    assert require_current_adoption(declaration)["signature_still_reaches_the_declaration"]

    declaration.write_bytes(declaration.read_bytes() + b"\n")
    with pytest.raises(DataSourceError, match="does not bind"):
        require_current_adoption(declaration)
