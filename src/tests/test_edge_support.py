import json
from pathlib import Path

import pytest

from src.benchmarks.edge_support import audit_edge_support
from src.core.errors import InvalidParameterError


def test_edge_support_uses_all_four_boundaries_and_keeps_missing_features():
    rows = [
        {"storm": "EAST", "sid": "1", "lat": -30.0, "lon": 179.5,
         "nearest_km": 400.0, "inside_radius": 0},
        {"storm": "SOUTH", "sid": "2", "lat": -57.5, "lon": 160.0,
         "nearest_km": None, "inside_radius": 0},
        {"storm": "MIDDLE", "sid": "3", "lat": -25.0, "lon": 160.0,
         "nearest_km": 20.0, "inside_radius": 1},
    ]

    result = audit_edge_support(
        rows, south=-58.0, north=-18.0, west=140.0, east=180.0)

    assert [row["nearest_edge"] for row in result["rows"]] == ["east", "south", "north"]
    assert result["strata"]["near_edge"]["rows"] == 2
    assert result["strata"]["near_edge"]["no_feature"] == 1
    assert result["strata"]["interior"]["rows"] == 1
    assert result["worst_rows"][0]["storm"] == "SOUTH"


def test_edge_support_refuses_positions_outside_the_declared_crop():
    with pytest.raises(InvalidParameterError, match="inside the crop bounds"):
        audit_edge_support(
            [{"storm": "OUT", "lat": -30.0, "lon": 181.0,
              "nearest_km": 10.0, "inside_radius": 1}],
            south=-58.0, north=-18.0, west=140.0, east=180.0)


def test_committed_join_receipt_can_be_audited_on_both_declared_paths():
    receipt = json.loads(Path("measurements/t4e28_join_rerun.json").read_text(encoding="utf-8"))

    for path_name in ("raw_field", "swt_planes"):
        result = audit_edge_support(
            receipt["paths"][path_name]["rows"],
            south=-58.0, north=-18.0, west=140.0, east=180.0)

        assert len(result["rows"]) == 18
        assert result["strata"]["near_edge"]["rows"] \
            + result["strata"]["interior"]["rows"] == 18
        assert result["association"]["finite_rows"] \
            == 18 - result["strata"]["near_edge"]["no_feature"] \
            - result["strata"]["interior"]["no_feature"]
        assert "Post-hoc descriptive diagnostic" in result["claim_boundary"]