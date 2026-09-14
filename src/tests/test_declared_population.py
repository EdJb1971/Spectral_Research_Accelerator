"""A record holding two answers must be asked which one.

Every test here is a way the unlabelled read that produced T4E.27's withdrawn conclusion could
happen again: reading without naming, naming something that is not there, and -- the one that
actually matters -- asking for a population the record summarises but does not hold, and getting
the other population's rows under its name.
"""
from __future__ import annotations

import json

import pytest

from src.data_layer.declared_population import (
    MAPS,
    Population,
    PopulationMap,
    UndeclaredPopulationError,
    map_for,
    read_population,
)

RECORD = "measurements/t4e18_acceptance.json"


# ---------------- the refusals that are the point


def test_reading_without_naming_a_population_is_refused():
    """The exact shape of the T4E.27 error: take the table, name no path."""
    with pytest.raises(UndeclaredPopulationError) as caught:
        read_population(RECORD)

    message = str(caught.value)
    assert "without naming which" in message
    # Both names are offered, so the refusal tells the reader what to ask for next.
    assert "swt_planes" in message and "raw_field" in message


def test_the_refusal_names_the_mistake_it_exists_to_prevent():
    with pytest.raises(UndeclaredPopulationError) as caught:
        read_population(RECORD)
    assert "T4E.27" in str(caught.value)


def test_a_population_the_record_only_summarises_is_refused_not_substituted():
    """The dangerous case: returning the SWT rows under the raw-field name."""
    with pytest.raises(UndeclaredPopulationError) as caught:
        read_population(RECORD, "raw_field")

    message = str(caught.value)
    assert "summary only" in message
    assert "no per-storm rows" in message
    # And it says what would produce them, rather than leaving a dead end.
    assert "re-run of the join" in message


def test_an_unknown_population_is_refused_with_the_known_names():
    with pytest.raises(UndeclaredPopulationError) as caught:
        read_population(RECORD, "raw")
    assert "swt_planes, raw_field" in str(caught.value)


def test_a_record_with_no_declared_map_is_refused_rather_than_read_whole():
    with pytest.raises(UndeclaredPopulationError) as caught:
        read_population("measurements/t4e24_false_absence.json", "anything")
    assert "no declared population map" in str(caught.value)


# ---------------- the named read


def test_naming_the_population_returns_its_rows_and_how_it_was_identified():
    rows, population = read_population(RECORD, "swt_planes")

    assert len(rows) == 18
    assert population.name == "swt_planes"
    # The identification is checkable prose, not an assertion to be believed.
    assert "73 to 153" in population.identified_by


def test_the_identification_holds_against_the_record_it_describes():
    """If the rows stopped matching the stated evidence, the map would be a false label."""
    rows, _ = read_population(RECORD, "swt_planes")
    counts = [row["features"] for row in rows]
    assert min(counts) >= 73 and max(counts) <= 153


def test_the_other_pass_is_distinguishable_by_the_same_evidence():
    """The raw pass yields 0 to 13 features; the two ranges do not overlap, which is what makes
    the identification decidable rather than a guess."""
    record = json.loads(open(RECORD, encoding="utf-8").read())
    shown = record["CORRECTION_2026_09_10"]["what_the_measurement_actually_shows"]
    assert shown["raw_field_extraction"]["features_per_frame"] == "0 to 13, median 7"
    assert shown["swt_plane_extraction"]["features_per_frame"] == "73 to 153, median 123"


def test_a_payload_may_be_supplied_so_the_file_is_read_once():
    record = json.loads(open(RECORD, encoding="utf-8").read())
    rows, _ = read_population(RECORD, "swt_planes", payload=record)
    assert len(rows) == 18


# ---------------- the map itself


def test_the_map_records_why_it_exists():
    declared = map_for(RECORD)
    assert declared is not None
    assert "withdrawn" in declared.why_this_map_exists


def test_a_record_absent_from_the_registry_has_no_map():
    assert map_for("measurements/nothing.json") is None


def test_the_map_is_declared_beside_the_record_and_does_not_edit_it():
    """Committed evidence is not rewritten to suit a later reader."""
    record = json.loads(open(RECORD, encoding="utf-8").read())
    assert "populations" not in record
    for row in record["per_storm"]:
        assert "extraction_pass" not in row


def test_every_registered_map_offers_at_least_one_population_with_rows():
    """A map where nothing can be read is a refusal dressed as a registry entry."""
    for declared in MAPS.values():
        assert any(p.has_rows for p in declared.populations), declared.record


def test_a_population_with_no_rows_must_say_what_is_there_instead():
    """A refusal that does not say what IS available sends a reader nowhere."""
    for declared in MAPS.values():
        for population in declared.populations:
            if not population.has_rows:
                assert population.summary_only, (declared.record, population.name)


def test_population_lookup_on_an_empty_map_refuses_cleanly():
    empty = PopulationMap(record="x", why_this_map_exists="none", populations=())
    with pytest.raises(UndeclaredPopulationError):
        empty.population("anything")


def test_describe_carries_everything_a_reader_needs_to_check_the_mapping():
    described = map_for(RECORD).describe()
    names = {p["name"] for p in described["populations"]}
    assert names == {"swt_planes", "raw_field"}
    for population in described["populations"]:
        assert population["identified_by"]
        assert population["description"]


def test_a_population_dataclass_reports_whether_it_has_rows():
    assert Population("a", "d", "rows", "how").has_rows is True
    assert Population("a", "d", None, "how", summary_only="only totals").has_rows is False
