"""A coverage check anyone can run, and the refusals that keep it from flattering an instrument.

The failure this module exists to prevent is a report that produces a reassuring number on
evidence that does not stand in for the field it claims to describe. So the gate, the refusals
and the claim boundary are pinned here alongside the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.coverage_report import coverage_report
from src.benchmarks.false_absence import survival_by_cardinality
from src.core.errors import InvalidParameterError


def _noise(seed: int, n: int = 64):
    def background_for(index: int) -> np.ndarray:
        return np.random.default_rng(seed + index).normal(size=(n, n))
    return background_for


# ---------------- the unit the pipeline actually consumes


def test_survival_counts_pairs_and_triples_not_whole_configurations():
    """`ALLOWED_CARDINALITIES` is (2, 3), so those are the groups a criterion is handed."""
    result = survival_by_cardinality([[6, 6, 6, 0]], scenes=6)

    assert result["k=2"]["groups"] == 6
    assert result["k=3"]["groups"] == 4
    # The three fully-seen features give one intact triple and three intact pairs; every group
    # containing the never-seen fourth is neither intact nor assemblable.
    assert result["k=3"]["intact_in_every_scene"] == 1
    assert result["k=2"]["intact_in_every_scene"] == 3


def test_a_group_holding_a_never_seen_feature_is_not_assemblable():
    """It cannot be matched by any criterion: the object was never built in any scene."""
    result = survival_by_cardinality([[6, 6, 0]], scenes=6)
    assert result["k=3"]["assemblable_in_at_least_one"] == 0


def test_assemblable_is_never_below_intact():
    rng = np.random.default_rng(3)
    for _ in range(40):
        configurations = [list(rng.integers(0, 7, size=int(rng.integers(3, 11))))
                          for _ in range(6)]
        result = survival_by_cardinality(configurations, scenes=6)
        for k in ("k=2", "k=3"):
            assert (result[k]["assemblable_in_at_least_one"]
                    >= result[k]["intact_in_every_scene"])


def test_a_configuration_smaller_than_the_group_contributes_nothing():
    result = survival_by_cardinality([[6, 6]], scenes=6)
    assert result["k=3"]["groups"] == 0
    assert result["k=3"]["intact_rate"] is None


def test_the_survival_report_says_why_those_sizes():
    assert "ALLOWED_CARDINALITIES" in survival_by_cardinality([[6]], scenes=6)["why_these_sizes"]


# ---------------- the report refuses rather than reassures


def test_an_unrepresentative_density_returns_a_refusal_not_a_coverage_number():
    """The gate T4E.20 declared in prose and did not implement, reached generically."""
    report = coverage_report(_noise(1), domain="synthetic", dataset="d", variable="v",
                             configurations=2, scenes=2, surrogates=19,
                             median_band=(400, 500))

    assert report["VERDICT"] == "EVIDENCE_NOT_REPRESENTATIVE"
    assert "recall" not in report
    assert "Nothing." in report["what_this_licenses"]


def test_an_unregistered_extractor_is_refused_with_the_known_names():
    with pytest.raises(InvalidParameterError) as caught:
        coverage_report(_noise(2), domain="d", dataset="d", variable="v",
                        extractor="no_such_extractor")
    assert "local_maximum" in str(caught.value)


def test_an_unknown_calibration_source_is_refused_by_name():
    with pytest.raises(InvalidParameterError):
        coverage_report(_noise(3), domain="d", dataset="d", variable="v",
                        calibrate_on="whatever")


def test_a_constant_background_is_refused_rather_than_divided_by():
    """A field with no spread has no scale for a peak-to-background ratio to be quoted against."""
    with pytest.raises(InvalidParameterError):
        coverage_report(lambda i: np.zeros((64, 64)), domain="d", dataset="d", variable="v",
                        configurations=1, scenes=1, surrogates=19, median_band=(0, 50))


def test_a_three_dimensional_background_is_refused():
    with pytest.raises(InvalidParameterError):
        coverage_report(lambda i: np.zeros((4, 8, 8)), domain="d", dataset="d", variable="v",
                        configurations=1, scenes=1, surrogates=19)


def test_zero_configurations_is_refused_rather_than_reported_as_perfect():
    with pytest.raises(InvalidParameterError):
        coverage_report(_noise(4), domain="d", dataset="d", variable="v", configurations=0)


# ---------------- what a passing report says, and what it refuses to say


def test_a_measured_report_carries_its_claim_boundary_and_its_instrument():
    report = coverage_report(_noise(5), domain="synthetic", dataset="d", variable="v",
                             configurations=2, scenes=2, surrogates=19, median_band=(0, 60))

    assert report["VERDICT"] == "MEASURED"
    assert "property of an instrument, not of a world" in report["claim_boundary"]
    assert report["instrument"]["extractor"] == "local_maximum"
    assert report["instrument"]["capabilities"]["shape_model"]


def test_the_report_states_that_identical_geometry_is_the_favourable_case():
    """A recall quoted without this reads as an estimate; it is an upper bound."""
    report = coverage_report(_noise(6), domain="d", dataset="d", variable="v",
                             configurations=2, scenes=2, surrogates=19, median_band=(0, 60))
    assert "UPPER bound" in report["design"]["geometry_is_identical_across_scenes"]


def test_the_calibration_choice_is_reported_with_its_measured_cost():
    report = coverage_report(_noise(7), domain="d", dataset="d", variable="v",
                             configurations=2, scenes=2, surrogates=19, median_band=(0, 60))
    assert report["instrument"]["calibrated_on"] == "scene"
    assert "3.53x" in report["instrument"]["why_that_matters"]


def test_recall_and_the_false_absence_rate_are_complements():
    report = coverage_report(_noise(8), domain="d", dataset="d", variable="v",
                             configurations=2, scenes=3, surrogates=19, median_band=(0, 60))
    assert report["recall"] + report["false_absence_rate"] == pytest.approx(1.0)


def test_the_report_is_reproducible_from_its_seed():
    kwargs = dict(domain="d", dataset="d", variable="v", configurations=2, scenes=2,
                  surrogates=19, median_band=(0, 60), seed=11)
    first = coverage_report(_noise(9), **kwargs)
    second = coverage_report(_noise(9), **kwargs)

    assert first["rows"] == second["rows"]
    assert first["recall"] == second["recall"]


def test_the_field_declaration_travels_into_the_report():
    """R19 refuses a comparison without it, so a report that dropped it would be unusable."""
    report = coverage_report(_noise(10), domain="ocean", dataset="mine", variable="ssh",
                             units="m", configurations=2, scenes=2, surrogates=19,
                             median_band=(0, 60))
    assert report["field"] == {"domain": "ocean", "dataset": "mine", "variable": "ssh",
                               "units": "m"}
