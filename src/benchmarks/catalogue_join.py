"""T4E.28: joining a signed catalogue to an extracted record, on terms the repository holds.

**What was wrong with how this was done.** T4E.18 measured a join twice -- once on the raw negated
field, once through SWT planes -- and the adopted record carries per-storm rows for the SWT path
only. The raw path, which that record's own correction calls *markedly better*, survives as four
rounded aggregates: `nearest_km_min 16.6`, `nearest_km_median 52.1`, `3 of 18`, `0 to 13, median
7`. T4E.27 then restated a position bar against the rows that existed rather than the path that
was better, and had to be corrected for it.

Worse, **neither path's parameters were written down anywhere in this repository.** Both runs came
from scripts in a session scratchpad under `%TEMP%` -- not version-controlled, not backed up, and
cleared by the operating system without warning. Figures in an adopted acceptance record were
reproducible only from a directory nobody would think to preserve. This module exists so that the
selection, the pairing and the two extraction paths are things the repository states.

**The full distance list, not the minimum.** Every row records the sorted distance from the
catalogue centre to *every* extracted feature. T4E.27 needed that distribution and found a single
nearest value, which is why its condition 2 could not be evaluated. A minimum answers one question
and forecloses the rest; keeping the list costs a few hundred floats per storm.

**A frame that yields nothing is a row, not a gap.** A storm whose frame produced no feature at
all -- GRETEL, in the original run -- has `nearest_km` of `None` and is counted in the denominator
of every condition. Dropping it would improve every aggregate by removing the worst case, which is
the most flattering possible way to be wrong.

**What this module does not decide.** It does not choose between the two paths, rank them, or
adjudicate an acceptance. It measures a join on a declared population and reports what it found
per path, side by side, in the order declared. The acceptance bar and its verdict belong to the
declaration that set them.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError

#: Sphere used for every distance here. The catalogue's own agency fixes are compared on the same
#: sphere, so the radius cancels out of the ratio that matters and only scales the absolute km.
EARTH_RADIUS_KM: float = 6371.0

#: The agencies whose independent fixes define a storm's positional uncertainty. A centre with
#: fewer than two of them has no radius that means anything, and is refused rather than given a
#: radius of zero -- the failure T4E.27 named when LINDA's `0.00` reached a tolerance calculation.
AGENCIES: Tuple[str, ...] = (
    "USA", "TOKYO", "CMA", "HKO", "KMA", "NEWDELHI", "REUNION", "BOM", "NADI", "WELLINGTON")

#: Synoptic hours. Reanalysis frames exist at these times; interpolating a catalogue fix to a
#: frame time, or a frame to a fix time, would add a positional error to the quantity being
#: measured.
SYNOPTIC_HOURS: Tuple[str, ...] = ("00", "06", "12", "18")

#: How far inside the crop's latitude band an observation must sit to be counted. A storm at the
#: boundary is refused by the off-frame rule for a reason that is an artefact of where the crop
#: was cut, not a property of the extractor.
INTERIOR_MARGIN_DEGREES: float = 2.0


def great_circle_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in km between two (latitude, longitude) points in degrees."""
    (lat_a, lon_a), (lat_b, lon_b) = a, b
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    haversine = (math.sin((phi_b - phi_a) / 2.0) ** 2
                 + math.cos(phi_a) * math.cos(phi_b)
                 * math.sin(math.radians(lon_b - lon_a) / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(haversine)))


@dataclass(frozen=True)
class Selection:
    """The population, declared before any row is read."""

    start_date: str
    end_date: str
    south: float
    north: float
    west: float
    east: float
    interior_margin: float = INTERIOR_MARGIN_DEGREES
    minimum_agencies: int = 2

    def describe(self) -> Dict[str, Any]:
        return {
            "dates": "%s..%s" % (self.start_date, self.end_date),
            "latitude": "%.1f..%.1f" % (self.south, self.north),
            "longitude": "%.1f..%.1f" % (self.west, self.east),
            "synoptic_hours": list(SYNOPTIC_HOURS),
            "interior_margin_degrees": self.interior_margin,
            "minimum_agency_fixes": self.minimum_agencies,
            "radius_definition":
                "the furthest of the agency fixes from the reported centre, great-circle on a "
                "sphere of radius %.1f km" % EARTH_RADIUS_KM,
            "one_observation_per_storm":
                "the deepest interior observation, furthest from either latitude edge -- not the "
                "first, which is always where the storm entered the box",
        }


@dataclass(frozen=True)
class Observation:
    """One catalogue fix that survived the selection."""

    sid: str
    name: str
    time: str
    lat: float
    lon: float
    nature: str
    radius_km: float
    agency_fixes: int

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


def read_observations(path: Path, selection: Selection) -> Tuple[List[Observation], Dict[str, int]]:
    """Every catalogue row meeting the declared selection, and a census of what was refused.

    The census is returned rather than logged: a selection that silently drops most of its
    population looks identical, from the aggregates alone, to one that keeps it.
    """
    kept: List[Observation] = []
    census = {"rows": 0, "outside_window": 0, "off_synoptic": 0, "outside_box": 0,
              "too_few_agencies": 0, "unparseable_position": 0}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        next(reader)                                    # the units row, beneath the header
        for row in reader:
            census["rows"] += 1
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
            except (ValueError, KeyError):
                census["unparseable_position"] += 1
                continue
            if lon < 0:
                lon += 360.0
            stamp = row["ISO_TIME"]
            if not stamp or not (selection.start_date <= stamp[:10] <= selection.end_date):
                census["outside_window"] += 1
                continue
            if stamp[11:13] not in SYNOPTIC_HOURS:
                census["off_synoptic"] += 1
                continue
            if not (selection.west <= lon <= selection.east
                    and selection.south <= lat <= selection.north):
                census["outside_box"] += 1
                continue
            fixes: List[Tuple[float, float]] = []
            for agency in AGENCIES:
                try:
                    fixes.append((float(row[agency + "_LAT"]), float(row[agency + "_LON"])))
                except (ValueError, KeyError):
                    pass
            if len(fixes) < selection.minimum_agencies:
                census["too_few_agencies"] += 1
                continue
            here = (lat, lon if lon <= 180.0 else lon - 360.0)
            kept.append(Observation(
                sid=row["SID"], name=row["NAME"] or "UNNAMED", time=stamp,
                lat=lat, lon=here[1], nature=row["NATURE"],
                radius_km=max(great_circle_km(here, fix) for fix in fixes),
                agency_fixes=len(fixes)))
    return kept, census


def deepest_per_storm(observations: Sequence[Observation],
                      selection: Selection) -> List[Observation]:
    """One observation per storm: the one furthest from either latitude edge.

    Sampling a storm's first in-box fix samples where it entered the box, which is always the
    boundary features are refused at. That is a property of the crop, not of the record.
    """
    margin = selection.interior_margin
    interior = [o for o in observations
                if selection.south + margin <= o.lat <= selection.north - margin]
    deepest: Dict[str, Tuple[float, Observation]] = {}
    for observation in interior:
        depth = min(observation.lat - selection.south, selection.north - observation.lat)
        if observation.sid not in deepest or depth > deepest[observation.sid][0]:
            deepest[observation.sid] = (depth, observation)
    return [pair[1] for pair in sorted(deepest.values(), key=lambda pair: pair[1].time)]


@dataclass
class JoinRow:
    """One storm's join, with the whole distance distribution rather than its minimum."""

    storm: str
    sid: str
    time: str
    lat: float
    lon: float
    nature: str
    radius_km: float
    features: int
    distances_km: List[float] = dataclass_field(default_factory=list)

    @property
    def nearest_km(self) -> Optional[float]:
        return self.distances_km[0] if self.distances_km else None

    @property
    def inside_radius(self) -> int:
        return sum(1 for d in self.distances_km if d <= self.radius_km)

    def describe(self) -> Dict[str, Any]:
        return {
            "storm": self.storm, "sid": self.sid, "time": self.time,
            "lat": self.lat, "lon": self.lon, "nature": self.nature,
            "radius_km": self.radius_km, "features": self.features,
            "nearest_km": self.nearest_km, "inside_radius": self.inside_radius,
            "distances_km": self.distances_km,
        }


def join_row(observation: Observation,
             feature_positions: Sequence[Tuple[float, float]]) -> JoinRow:
    """Pair one observation against every feature extracted from its frame.

    A frame that produced nothing yields a row with no distances, not an absent row.
    """
    distances = sorted(great_circle_km(observation.centre, position)
                       for position in feature_positions)
    return JoinRow(storm=observation.name, sid=observation.sid, time=observation.time,
                   lat=observation.lat, lon=observation.lon, nature=observation.nature,
                   radius_km=observation.radius_km, features=len(feature_positions),
                   distances_km=distances)


def aggregates(rows: Sequence[JoinRow]) -> Dict[str, Any]:
    """The figures an acceptance is read from, over the whole declared population.

    Storms with no feature are in the denominator of every condition and out of the distance
    quantiles, because there is no distance to include -- not because excluding them helps.
    """
    if not rows:
        raise InvalidParameterError(
            "rows", rows,
            "at least one join row; an empty population has no figures, and reporting zeroes "
            "for it would read as a measurement rather than as an absence")
    nearest = sorted(row.nearest_km for row in rows if row.nearest_km is not None)
    counts = sorted(row.features for row in rows)
    total = len(rows)
    needed = (total + 1) // 2
    return {
        "storms": total,
        "storms_with_no_feature": sum(1 for row in rows if row.nearest_km is None),
        "nearest_km": ({"min": nearest[0], "median": nearest[len(nearest) // 2],
                        "max": nearest[-1]} if nearest else None),
        "features_per_frame": {"min": counts[0], "median": counts[len(counts) // 2],
                               "max": counts[-1]},
        "condition_1_at_least_one_inside_radius": {
            "met": sum(1 for row in rows if row.inside_radius >= 1),
            "of": total, "needed": needed},
        "condition_2_three_inside_radius": {
            "met": sum(1 for row in rows if row.inside_radius >= 3),
            "of": total, "needed": needed},
    }


def check_gate(observed: Dict[str, Any], declared: Dict[str, Any],
               tolerance_km: float = 0.1) -> Dict[str, Any]:
    """Adjudicate observed aggregates against figures declared before the run.

    Every comparison is reported with both numbers, whether it passed or failed. A gate that
    printed only its failures would let a reader assume the rest matched exactly rather than
    within a tolerance somebody chose.
    """
    if tolerance_km <= 0 or not math.isfinite(tolerance_km):
        raise InvalidParameterError(
            "tolerance_km", tolerance_km,
            "a positive finite number of kilometres; a zero tolerance demands bit-identical "
            "floats from figures the record rounds to 0.1 km")
    comparisons: List[Dict[str, Any]] = []
    for key, declared_value in sorted(declared.items()):
        observed_value = observed.get(key)
        if observed_value is None:
            comparisons.append({"quantity": key, "declared": declared_value,
                                "observed": None, "agrees": False,
                                "note": "not present in the observed aggregates"})
            continue
        if isinstance(declared_value, float):
            agrees = abs(float(observed_value) - declared_value) <= tolerance_km
        else:
            agrees = observed_value == declared_value
        comparisons.append({"quantity": key, "declared": declared_value,
                            "observed": observed_value, "agrees": agrees})
    return {
        "verdict": "REPRODUCED" if all(c["agrees"] for c in comparisons) else "NOT REPRODUCED",
        "tolerance_km": tolerance_km,
        "comparisons": comparisons,
        "what_this_checks": (
            "whether this repository regenerates figures a record already asserts. It does not "
            "check that those figures were right, that the extraction path was the right one, "
            "or that the acceptance bar was well chosen."),
    }


def compare_rows(observed: Sequence[JoinRow], recorded: Sequence[Dict[str, Any]],
                 tolerance_km: float = 0.1) -> Dict[str, Any]:
    """Row-by-row agreement against a previously recorded per-storm table.

    This is what tests whether a set of extraction parameters recovered from outside the
    repository is the set that produced the record, rather than a plausible guess that happens to
    land near the same aggregates.
    """
    by_name = {row.storm: row for row in observed}
    disagreements: List[Dict[str, Any]] = []
    for entry in recorded:
        name = entry["storm"]
        row = by_name.get(name)
        if row is None:
            disagreements.append({"storm": name, "why": "absent from the re-run"})
            continue
        if row.features != entry["features"]:
            disagreements.append({"storm": name, "why": "feature count",
                                  "recorded": entry["features"], "observed": row.features})
        recorded_near, observed_near = entry.get("nearest_km"), row.nearest_km
        if (recorded_near is None) != (observed_near is None):
            disagreements.append({"storm": name, "why": "one path found no feature",
                                  "recorded": recorded_near, "observed": observed_near})
        elif recorded_near is not None and abs(recorded_near - observed_near) > tolerance_km:
            disagreements.append({"storm": name, "why": "nearest_km",
                                  "recorded": recorded_near, "observed": observed_near})
        if row.inside_radius != entry["inside_radius"]:
            disagreements.append({"storm": name, "why": "inside_radius",
                                  "recorded": entry["inside_radius"],
                                  "observed": row.inside_radius})
    return {
        "rows_recorded": len(recorded),
        "rows_observed": len(observed),
        "agrees": not disagreements and len(recorded) == len(observed),
        "disagreements": disagreements,
    }
