"""The join whose parameters lived in a temp directory, now pinned where they can be checked.

These tests run on a synthetic catalogue of a few rows. They do not touch the 35 MB signed
catalogue or the 325 MB record -- those belong to the T4E.28 run itself, which is a measurement
with a declared gate rather than a unit test.
"""
from __future__ import annotations

import math

import pytest

from src.benchmarks.catalogue_join import (
    AGENCIES,
    Observation,
    Selection,
    aggregates,
    check_gate,
    compare_rows,
    deepest_per_storm,
    great_circle_km,
    join_row,
    read_observations,
)
from src.core.errors import InvalidParameterError


SELECTION = Selection(start_date="2018-01-01", end_date="2021-12-31",
                      south=-58.0, north=-18.0, west=140.0, east=180.0)


def _catalogue(tmp_path, rows):
    """A catalogue with IBTrACS's header and its units row beneath it."""
    columns = ["SID", "NAME", "ISO_TIME", "NATURE", "LAT", "LON"]
    columns += [a + suffix for a in AGENCIES for suffix in ("_LAT", "_LON")]
    lines = [",".join(columns), ",".join(" " for _ in columns)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in columns))
    path = tmp_path / "catalogue.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(**kwargs):
    row = {"SID": "S1", "NAME": "ALPHA", "ISO_TIME": "2018-02-01 12:00:00", "NATURE": "TS",
           "LAT": -30.0, "LON": 160.0, "USA_LAT": -30.1, "USA_LON": 160.0,
           "TOKYO_LAT": -30.0, "TOKYO_LON": 160.1}
    row.update(kwargs)
    return row


# ---------------- distance


def test_a_degree_of_latitude_is_about_111_km():
    assert great_circle_km((0.0, 0.0), (1.0, 0.0)) == pytest.approx(111.19, abs=0.05)


def test_distance_is_symmetric_and_zero_at_a_point():
    assert great_circle_km((-30.0, 160.0), (-30.0, 160.0)) == 0.0
    there = great_circle_km((-30.0, 160.0), (-31.0, 162.0))
    back = great_circle_km((-31.0, 162.0), (-30.0, 160.0))
    assert there == pytest.approx(back)


# ---------------- selection


def test_the_declared_window_box_and_hours_are_all_applied(tmp_path):
    path = _catalogue(tmp_path, [
        _row(SID="keep"),
        _row(SID="early", ISO_TIME="2017-12-31 12:00:00"),
        _row(SID="offhour", ISO_TIME="2018-02-01 03:00:00"),
        _row(SID="westof", LON=100.0),
        _row(SID="northof", LAT=-10.0),
    ])
    kept, census = read_observations(path, SELECTION)

    assert [o.sid for o in kept] == ["keep"]
    assert census["rows"] == 5
    assert census["outside_window"] == 1
    assert census["off_synoptic"] == 1
    assert census["outside_box"] == 2


def test_a_single_agency_fix_is_refused_rather_than_given_a_zero_radius(tmp_path):
    """A radius of 0.00 reaching a tolerance calculation is the failure T4E.27 named."""
    path = _catalogue(tmp_path, [_row(SID="lonely", TOKYO_LAT="", TOKYO_LON="")])
    kept, census = read_observations(path, SELECTION)

    assert kept == []
    assert census["too_few_agencies"] == 1


def test_the_radius_is_the_furthest_agency_fix_not_the_nearest(tmp_path):
    path = _catalogue(tmp_path, [_row(USA_LAT=-30.0, USA_LON=160.05,
                                      TOKYO_LAT=-30.0, TOKYO_LON=160.5)])
    kept, _ = read_observations(path, SELECTION)

    assert kept[0].radius_km == pytest.approx(great_circle_km((-30.0, 160.0), (-30.0, 160.5)))
    assert kept[0].agency_fixes == 2


def test_the_units_row_beneath_the_header_is_not_read_as_an_observation(tmp_path):
    path = _catalogue(tmp_path, [_row()])
    kept, census = read_observations(path, SELECTION)

    assert len(kept) == 1 and census["rows"] == 1


# ---------------- one observation per storm


def test_the_deepest_observation_is_taken_not_the_first():
    """The first in-box fix is always where the storm entered, at the boundary."""
    edge = Observation("S1", "ALPHA", "2018-02-01 00:00:00", -20.0, 160.0, "TS", 10.0, 2)
    deep = Observation("S1", "ALPHA", "2018-02-02 00:00:00", -38.0, 161.0, "TS", 10.0, 2)
    picked = deepest_per_storm([edge, deep], SELECTION)

    assert [o.time for o in picked] == ["2018-02-02 00:00:00"]


def test_observations_inside_the_margin_of_an_edge_are_excluded():
    just_inside = Observation("S1", "A", "2018-02-01 00:00:00", -19.0, 160.0, "TS", 10.0, 2)
    assert deepest_per_storm([just_inside], SELECTION) == []


def test_one_row_per_storm_ordered_by_time():
    a = Observation("S1", "ALPHA", "2018-03-01 00:00:00", -38.0, 160.0, "TS", 10.0, 2)
    b = Observation("S2", "BRAVO", "2018-02-01 00:00:00", -38.0, 160.0, "TS", 10.0, 2)
    assert [o.storm if hasattr(o, "storm") else o.name
            for o in deepest_per_storm([a, b], SELECTION)] == ["BRAVO", "ALPHA"]


# ---------------- the join


def test_every_distance_is_kept_not_only_the_nearest():
    """T4E.27 needed the distribution and the record held a minimum."""
    observation = Observation("S1", "ALPHA", "t", -30.0, 160.0, "TS", 60.0, 2)
    row = join_row(observation, [(-30.0, 160.2), (-30.0, 161.0), (-30.0, 160.05)])

    assert len(row.distances_km) == 3
    assert row.distances_km == sorted(row.distances_km)
    assert row.nearest_km == row.distances_km[0]
    assert row.inside_radius == 2               # 0.05 and 0.2 degrees are inside 60 km; 1.0 is not


def test_a_frame_with_no_feature_is_a_row_with_no_distance():
    row = join_row(Observation("S1", "GRETEL", "t", -30.0, 160.0, "TS", 106.9, 2), [])

    assert row.features == 0
    assert row.nearest_km is None
    assert row.inside_radius == 0


def test_the_radius_boundary_admits_inclusively():
    observation = Observation("S1", "A", "t", 0.0, 0.0, "TS", 111.19492664455873, 2)
    assert join_row(observation, [(1.0, 0.0)]).inside_radius == 1


# ---------------- aggregates


def test_a_storm_with_no_feature_stays_in_every_denominator():
    """Dropping it would improve every aggregate by removing the worst case."""
    rows = [join_row(Observation("S1", "A", "t", 0.0, 0.0, "TS", 200.0, 2), [(1.0, 0.0)]),
            join_row(Observation("S2", "B", "t", 0.0, 0.0, "TS", 200.0, 2), [])]
    summary = aggregates(rows)

    assert summary["storms"] == 2
    assert summary["storms_with_no_feature"] == 1
    assert summary["condition_1_at_least_one_inside_radius"] == {"met": 1, "of": 2, "needed": 1}
    assert summary["features_per_frame"]["min"] == 0


def test_both_conditions_are_reported_not_only_the_first():
    """The raw path's condition 2 was never computed, which is why it is not optional here."""
    close = [(0.01, 0.0), (0.0, 0.01), (0.01, 0.01)]
    rows = [join_row(Observation("S1", "A", "t", 0.0, 0.0, "TS", 200.0, 2), close)]
    summary = aggregates(rows)

    assert summary["condition_1_at_least_one_inside_radius"]["met"] == 1
    assert summary["condition_2_three_inside_radius"]["met"] == 1


def test_an_empty_population_is_refused_rather_than_reported_as_zeroes():
    with pytest.raises(InvalidParameterError):
        aggregates([])


# ---------------- the gate


def test_a_reproduced_gate_reports_both_numbers_for_every_quantity():
    result = check_gate({"nearest_km_median": 52.14, "inside": "3 of 18"},
                        {"nearest_km_median": 52.1, "inside": "3 of 18"})

    assert result["verdict"] == "REPRODUCED"
    assert all("observed" in c and "declared" in c for c in result["comparisons"])


def test_a_drifted_figure_fails_the_gate_by_name():
    result = check_gate({"nearest_km_median": 61.0}, {"nearest_km_median": 52.1})
    failed = [c for c in result["comparisons"] if not c["agrees"]]

    assert result["verdict"] == "NOT REPRODUCED"
    assert failed[0]["quantity"] == "nearest_km_median"
    assert failed[0]["observed"] == 61.0


def test_a_missing_observed_quantity_fails_rather_than_passing_vacuously():
    result = check_gate({}, {"nearest_km_median": 52.1})
    assert result["verdict"] == "NOT REPRODUCED"
    assert result["comparisons"][0]["observed"] is None


def test_a_zero_tolerance_is_refused_with_the_reason():
    with pytest.raises(InvalidParameterError) as caught:
        check_gate({"a": 1.0}, {"a": 1.0}, tolerance_km=0.0)
    assert "rounds to 0.1 km" in str(caught.value)


def test_the_gate_states_that_reproducing_is_not_being_right():
    result = check_gate({"a": 1.0}, {"a": 1.0})
    assert "were right" in result["what_this_checks"]


# ---------------- recovered parameters against a recorded table


def test_row_comparison_confirms_recovered_parameters_when_every_row_matches():
    observed = [join_row(Observation("S1", "FEHI", "t", 0.0, 0.0, "TS", 8.9, 2), [(1.0, 0.0)])]
    recorded = [{"storm": "FEHI", "features": 1, "nearest_km": observed[0].nearest_km,
                 "inside_radius": 0}]

    assert compare_rows(observed, recorded)["agrees"] is True


def test_a_matching_aggregate_does_not_hide_a_row_that_disagrees():
    """This is the point of comparing rows rather than medians."""
    observed = [join_row(Observation("S1", "FEHI", "t", 0.0, 0.0, "TS", 8.9, 2), [(1.0, 0.0)])]
    recorded = [{"storm": "FEHI", "features": 7, "nearest_km": observed[0].nearest_km,
                 "inside_radius": 0}]
    result = compare_rows(observed, recorded)

    assert result["agrees"] is False
    assert result["disagreements"][0]["why"] == "feature count"


def test_a_storm_missing_from_the_rerun_is_a_disagreement():
    recorded = [{"storm": "GONE", "features": 1, "nearest_km": 5.0, "inside_radius": 0}]
    result = compare_rows([], recorded)

    assert result["agrees"] is False
    assert result["disagreements"][0]["why"] == "absent from the re-run"


def test_a_found_feature_where_none_was_recorded_is_a_disagreement():
    observed = [join_row(Observation("S1", "GRETEL", "t", 0.0, 0.0, "TS", 106.9, 2),
                         [(1.0, 0.0)])]
    recorded = [{"storm": "GRETEL", "features": 1, "nearest_km": None, "inside_radius": 0}]

    assert compare_rows(observed, recorded)["disagreements"][0]["why"] == \
        "one path found no feature"


# ---------------- the committed T4E.28 run


def _rerun():
    import json
    return json.loads(open("measurements/t4e28_join_rerun.json", encoding="utf-8").read())


def test_both_declared_gates_read_reproduced_on_the_committed_run():
    record = _rerun()
    assert record["paths"]["raw_field"]["gate"]["verdict"] == "REPRODUCED"
    assert record["paths"]["swt_planes"]["gate"]["verdict"] == "REPRODUCED"


def test_the_recovered_swt_parameters_are_confirmed_against_all_eighteen_recorded_rows():
    """The parameters existed only in a %TEMP% directory; this is what makes them the real ones."""
    recovery = _rerun()["paths"]["swt_planes"]["parameter_recovery"]

    assert recovery["verdict"] == "CONFIRMED"
    assert recovery["detail"]["rows_recorded"] == recovery["detail"]["rows_observed"] == 18
    assert recovery["detail"]["disagreements"] == []


def test_the_raw_path_is_better_on_distance_and_worse_on_the_count_inside_the_radius():
    """'Markedly better' was a statement about one quantity and does not carry to the other."""
    paths = _rerun()["paths"]
    raw = paths["raw_field"]["aggregates"]
    swt = paths["swt_planes"]["aggregates"]

    assert raw["nearest_km"]["median"] < swt["nearest_km"]["median"]
    assert raw["condition_2_three_inside_radius"]["met"] == 0
    assert swt["condition_2_three_inside_radius"]["met"] == 1


def test_t4e27_condition_2_agrees_with_recomputing_it_from_the_distance_lists():
    """The refusal was lifted by evidence; this pins that the evidence still says so."""
    import json
    restated = json.loads(
        open("measurements/t4e27_restated_acceptance.json", encoding="utf-8").read())
    lift = restated["CONDITION_2_LIFTED_2026_09_11_BY_T4E_28"]
    distances = {row["storm"]: row["distances_km"]
                 for row in _rerun()["paths"]["swt_planes"]["rows"]}

    met = judged = 0
    for row in restated["per_storm"]:
        tolerance = row.get("tolerance_km")
        if tolerance is None:
            continue
        judged += 1
        met += sum(1 for d in distances[row["storm"]] if d <= tolerance) >= 3

    recorded = lift["condition_2_three_features_inside_the_restated_tolerance"]["swt_planes"]
    assert (recorded["met"], recorded["judged"]) == (met, judged)
    assert met < recorded["needed"]
