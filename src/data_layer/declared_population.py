"""A record holding more than one population is read by name, or not at all.

**The failure this exists to prevent, which already happened.**
`measurements/t4e18_acceptance.json` holds results from **two** extraction paths -- a raw-field
pass and an SWT-plane pass -- with different distances, different feature counts and different
verdicts. Its `per_storm` table is the SWT one. Nothing in that table says so: its fields are
`storm, lat, nature, time, radius_km, nearest_km, inside_radius, features`, and the only way to
tell which pass produced it is to match its feature counts against a prose block elsewhere in the
same file.

T4E.27 read that table, restated a bar against it, and named no path. The verdict survived; the
sharpest conclusion did not, and had to be withdrawn in the open. The record permitted an
unlabelled read and one was taken.

**Why this module and not a convention.** This repository already enforces exactly this
discipline in the other direction. `declare_identity_target` refuses to run an audit without a
declared target and evidence class, because labels drawn from the pipeline under test validate a
definition against itself. That rule is applied to the *inputs* of a measurement and to nothing
that *reads one back*. This is the same rule, pointed the other way.

**Legacy records are mapped beside themselves, never edited.** Committed evidence is not rewritten
to suit a later reader -- the same reason T4E.17's signed design was left alone and its local path
recorded elsewhere. A `PopulationMap` declares what an existing record holds, how each population
is identified within it, and the evidence for that identification, so the mapping can be checked
rather than believed.

**A population that exists only as a summary is refused as a population.** The raw-field pass is
in the record as four aggregate numbers and no per-storm rows. A reader asking for its rows gets a
refusal naming what is actually there and what would produce the rest, rather than the SWT rows
under a raw-field name -- which is precisely the substitution that went wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import UserInputError


class UndeclaredPopulationError(UserInputError):
    """A record holding more than one population was read without naming which."""

    status_code = 400
    client_safe = True


@dataclass(frozen=True)
class Population:
    """One population inside a record, and how a reader can tell it is that one."""

    name: str
    description: str
    #: Dotted path to the rows, or `None` when the record holds only a summary of this one.
    rows_at: Optional[str]
    #: How this population is distinguished inside the record, in words a reader can check.
    identified_by: str
    #: What is present when `rows_at` is None, and what would produce the rest.
    summary_only: Optional[str] = None

    @property
    def has_rows(self) -> bool:
        return self.rows_at is not None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "has_rows": self.has_rows, "rows_at": self.rows_at,
                "identified_by": self.identified_by, "summary_only": self.summary_only}


@dataclass(frozen=True)
class PopulationMap:
    """What an existing record holds, declared beside it rather than edited into it."""

    record: str
    why_this_map_exists: str
    populations: Tuple[Population, ...] = dc_field(default_factory=tuple)

    def names(self) -> List[str]:
        return [p.name for p in self.populations]

    def population(self, name: str) -> Population:
        for candidate in self.populations:
            if candidate.name == name:
                return candidate
        raise UndeclaredPopulationError(
            "%r holds no population named %r. It holds: %s"
            % (self.record, name, ", ".join(self.names())),
            record=self.record, requested=name, available=self.names())

    def describe(self) -> Dict[str, Any]:
        return {"record": self.record, "why_this_map_exists": self.why_this_map_exists,
                "populations": [p.describe() for p in self.populations]}


def _dig(payload: Any, dotted: str) -> Any:
    node = payload
    for part in dotted.split("."):
        node = node[part]
    return node


#: The maps this repository knows. A record absent from here is read whole and at the reader's
#: risk; a record present here cannot be read unnamed.
MAPS: Dict[str, PopulationMap] = {
    "measurements/t4e28_join_rerun.json": PopulationMap(
        record="measurements/t4e28_join_rerun.json",
        why_this_map_exists=(
            "The re-run holds the same two extraction passes as T4E.18 and now holds ROWS for "
            "both, including the full sorted distance from each catalogue centre to every "
            "extracted feature. A record that holds two answers must still be asked which one, "
            "and having rows for both makes the unnamed read easier to get away with, not "
            "harder."),
        populations=(
            Population(
                name="raw_field",
                description=(
                    "Extraction on the raw negated field, representation 'identity'. Nearest "
                    "feature 16.6 km at minimum and 52.1 at median, 3 of 18 inside the "
                    "catalogue radius and 0 of 18 with three inside it, 0 to 13 features per "
                    "frame. Better than the SWT planes on nearest distance and WORSE on the "
                    "count inside the radius; 'markedly better' was a statement about one of "
                    "those and does not carry to the other."),
                rows_at="paths.raw_field.rows",
                identified_by=(
                    "`extraction.raw_field` in the record reads \"representation 'identity'\", "
                    "and these rows' `features` field runs 0 to 13 against the SWT pass's 73 "
                    "to 153"),
            ),
            Population(
                name="swt_planes",
                description=(
                    "Extraction through the stationary wavelet planes, wavelet db2 at level 3 "
                    "over every extractable plane. Median nearest feature 127.1 km, 2 of 18 "
                    "inside the catalogue radius and 1 of 18 with three inside it, 73 to 153 "
                    "features per frame. This is the population T4E.27 restated against, and "
                    "these rows regenerate T4E.18's recorded table exactly."),
                rows_at="paths.swt_planes.rows",
                identified_by=(
                    "`extraction.swt_planes` in the record reads {'wavelet': 'db2', 'level': "
                    "3}, and `paths.swt_planes.parameter_recovery.verdict` is CONFIRMED "
                    "against all 18 rows T4E.18 recorded"),
            ),
        ),
    ),
    "measurements/t4e18_acceptance.json": PopulationMap(
        record="measurements/t4e18_acceptance.json",
        why_this_map_exists=(
            "The record holds two extraction passes with different distances and different "
            "verdicts, and its per_storm table carries no field saying which pass produced it. "
            "T4E.27 read that table, restated a bar against it and named no path; the verdict "
            "survived and its sharpest conclusion had to be withdrawn. This map makes the "
            "distinction a reader must state rather than one they can miss."),
        populations=(
            Population(
                name="swt_planes",
                description=(
                    "Extraction through the stationary wavelet planes. Median nearest feature "
                    "127.1 km, 2 of 18 inside the catalogue radius, 73 to 153 features per "
                    "frame. This is the population T4E.27 restated against."),
                rows_at="per_storm",
                identified_by=(
                    "the per_storm rows' `features` field runs 73 to 153, matching "
                    "CORRECTION_2026_09_10.what_the_measurement_actually_shows."
                    "swt_plane_extraction.features_per_frame; the raw pass yields 0 to 13"),
            ),
            Population(
                name="raw_field",
                description=(
                    "Extraction on the raw negated field. Nearest feature 16.6 km at minimum "
                    "and 52.1 km at median, 3 of 18 inside the catalogue radius, 0 to 13 "
                    "features per frame. T4E.18's own correction calls this 'markedly BETTER "
                    "for this purpose' than the SWT planes."),
                rows_at=None,
                identified_by=(
                    "CORRECTION_2026_09_10.what_the_measurement_actually_shows."
                    "raw_field_extraction, which is a summary of four numbers"),
                summary_only=(
                    "SUPERSEDED 2026-09-11: T4E.28 re-ran this pass against a gate declared "
                    "beforehand, reproduced every aggregate below, and recorded the per-storm "
                    "rows this record never held. Read "
                    "measurements/t4e28_join_rerun.json population 'raw_field' for them. "
                    "In THIS record: only aggregates are recorded -- nearest_km_min, nearest_km_median, "
                    "inside_catalogue_radius and features_per_frame. There are no per-storm "
                    "rows for this pass anywhere in the record. Producing them needs a re-run "
                    "of the join that records the full per-storm distance list, which is "
                    "separate work. Until then, per-storm questions about the raw pass are "
                    "unanswerable from this record and must be refused rather than answered "
                    "with the SWT rows."),
            ),
        ),
    ),
}


def map_for(record: str) -> Optional[PopulationMap]:
    """The declared map for a record path, or `None` when none is declared."""
    return MAPS.get(str(record).replace("\\", "/"))


def read_population(record: str, population: Optional[str] = None,
                    *, payload: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Population]:
    """The rows of one named population, or a refusal saying why there are none to give.

    `population` is required whenever the record has a declared map. That is the whole point:
    a record holding two answers must be asked which one, and a reader who does not ask gets a
    refusal naming both rather than whichever happens to be stored first.
    """
    declared = map_for(record)
    if declared is None:
        raise UndeclaredPopulationError(
            "%r has no declared population map, so what it holds has not been stated and a "
            "named read cannot be checked. Declare one in MAPS before reading it by name"
            % record, record=record)

    if population is None:
        raise UndeclaredPopulationError(
            "%r holds more than one population and was read without naming which. It holds: "
            "%s. Name one -- reading whichever is stored first is how T4E.27 restated a bar "
            "against the SWT pass and reported it as though the record held one answer"
            % (record, ", ".join(declared.names())),
            record=record, available=declared.names())

    chosen = declared.population(population)
    if not chosen.has_rows:
        raise UndeclaredPopulationError(
            "%r records %r as a summary only, not as rows. %s"
            % (record, population, chosen.summary_only or ""),
            record=record, population=population)

    data = payload if payload is not None else json.loads(
        Path(record).read_text(encoding="utf-8"))
    rows = _dig(data, chosen.rows_at)
    return list(rows), chosen
