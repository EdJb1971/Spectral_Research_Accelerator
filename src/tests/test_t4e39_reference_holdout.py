import json
from pathlib import Path

import pytest

from src.benchmarks.catalogue_join import Observation
from src.benchmarks.reference_holdout import (
    DEFAULT_DECLARATION,
    build_reference_holdout_census,
    load_holdout_declaration,
    plan_reference_holdout,
)
from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import DataSourceError


def test_holdout_plan_binds_result_and_reference_without_opening_rows(monkeypatch):
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.read_observations",
        lambda *args, **kwargs: pytest.fail("offline plan parsed holdout rows"))
    plan = plan_reference_holdout()

    assert plan["task"] == "T4E.39"
    assert plan["status"] == "DRAFTED_NOT_ADOPTED"
    assert plan["source_outcome"] == "VERTICAL_QUANTITY_SEPARATION_DOMINANT"
    assert plan["source_verdict"] == "NOT_AN_ACCEPTANCE"
    assert plan["signed_reference"]["available"] is True
    assert plan["catalogue_rows_read"] is False
    assert plan["holdout_identities_enumerated"] is False
    assert plan["era5_record_opened"] is False and plan["network_used"] is False


def test_holdout_declaration_refuses_a_changed_population_or_decision(tmp_path):
    body = json.loads(DEFAULT_DECLARATION.read_text(encoding="utf-8"))
    body["holdout_population"]["dates"][0] = "2021-01-01"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataSourceError, match="holdout selection"):
        load_holdout_declaration(changed)

    body = json.loads(DEFAULT_DECLARATION.read_text(encoding="utf-8"))
    body["decision_declared_before_holdout_opening"]["majority"] = "chosen later"
    changed.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataSourceError, match="holdout decision"):
        load_holdout_declaration(changed)


def test_census_requires_current_human_adoption_before_parsing_catalogue(
        tmp_path, monkeypatch):
    declaration = tmp_path / DEFAULT_DECLARATION.name
    declaration.write_bytes(DEFAULT_DECLARATION.read_bytes())
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.read_observations",
        lambda *args, **kwargs: pytest.fail("unadopted census parsed holdout rows"))

    with pytest.raises(DataSourceError, match="not adopted"):
        build_reference_holdout_census(
            declaration, output_path=tmp_path / "census.json")


class _Reference:
    expected_sha256 = "631f76b95c77a6a4e409233466a0d501bb4848324e421fc58e228efea2086c44"
    expected_bytes = 35482417
    available = True

    def __init__(self, path):
        self.path = path

    def require(self):
        return self.path

    def describe(self):
        return {
            "name": "ibtracs_sp_v04r01", "path": str(self.path), "available": True,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.expected_sha256,
            "expected_bytes": self.expected_bytes, "observed_bytes": self.expected_bytes,
            "source_url": "https://example.invalid/ibtracs.csv",
            "signature_verified": True, "citation_required": [], "refusal": None,
        }


def _adopted_copy(tmp_path):
    declaration = tmp_path / DEFAULT_DECLARATION.name
    declaration.write_bytes(DEFAULT_DECLARATION.read_bytes())
    sign_declaration(
        tmp_path, declaration.name, adopted_by="Test Human Reviewer",
        adopted_as="ADOPTED_FOR_TEST",
        what_was_adopted="the copied T4E.39 holdout declaration",
        affirmation=REQUIRED_AFFIRMATION)
    return declaration


def _observations(count):
    return [Observation(
        sid="SID%02d" % index, name="STORM%02d" % index,
        time="2022-01-%02d 00:00:00" % (index + 1),
        lat=-30.0, lon=160.0, nature="TS", radius_km=20.0,
        agency_fixes=2) for index in range(count)]


def test_adopted_census_freezes_only_identity_geometry_and_no_era5(
        tmp_path, monkeypatch):
    declaration = _adopted_copy(tmp_path)
    reference = _Reference(tmp_path / "catalogue.csv")
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.resolve", lambda *args, **kwargs: reference)
    census = {
        "rows": 1000, "outside_window": 500, "off_synoptic": 0,
        "outside_box": 107, "too_few_agencies": 383, "unparseable_position": 0,
    }
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.read_observations",
        lambda *args, **kwargs: (_observations(10), census))
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout._count_disclosed_t4e17_population",
        lambda *args, **kwargs: {
            "source": "T4E.17 pre-amendment matching domain",
            "catalogue_box": {"lat_min": -60.0, "lat_max": -20.0,
                              "lon_min": 140.0, "lon_max": 180.0},
            "dates": ["2022-01-01", "2023-12-31"],
            "in_box_observations": 393, "distinct_storms": 13,
        })

    result = build_reference_holdout_census(
        declaration, output_path=tmp_path / "census.json")

    assert result["status"] == "READY_FOR_EXACT_ACQUISITION"
    assert result["selected_storms"] == 10 and result["future_majority_needed"] == 6
    assert result["disclosed_population_check"]["in_box_observations"] == 393
    assert result["disclosed_population_check"]["distinct_storms"] == 13
    assert result["era5_record_opened"] is False and result["network_used"] is False
    assert all("nature" not in row for row in result["selected_rows"])
    assert (tmp_path / "census.json").is_file()


def test_adopted_census_names_an_insufficient_selected_population(
        tmp_path, monkeypatch):
    declaration = _adopted_copy(tmp_path)
    reference = _Reference(tmp_path / "catalogue.csv")
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.resolve", lambda *args, **kwargs: reference)
    census = {
        "rows": 1000, "outside_window": 500, "off_synoptic": 0,
        "outside_box": 108, "too_few_agencies": 384, "unparseable_position": 0,
    }
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout.read_observations",
        lambda *args, **kwargs: (_observations(9), census))
    monkeypatch.setattr(
        "src.benchmarks.reference_holdout._count_disclosed_t4e17_population",
        lambda *args, **kwargs: {
            "source": "T4E.17 pre-amendment matching domain",
            "catalogue_box": {"lat_min": -60.0, "lat_max": -20.0,
                              "lon_min": 140.0, "lon_max": 180.0},
            "dates": ["2022-01-01", "2023-12-31"],
            "in_box_observations": 393, "distinct_storms": 13,
        })

    result = build_reference_holdout_census(
        declaration, output_path=tmp_path / "census.json")
    assert result["selected_storms"] == 9
    assert result["status"] == "INSUFFICIENT_HOLDOUT_POPULATION"
