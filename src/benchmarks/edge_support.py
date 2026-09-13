"""Post-hoc crop-edge diagnostics for catalogue join receipts.

This module describes whether join errors cluster near an acquired field's boundaries. It does
not adjudicate an acceptance: the T4E.28 rows were inspected before this diagnostic existed, so
any association reported here is exploratory evidence for the next acquisition decision.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Mapping, Sequence

from scipy.stats import spearmanr

from src.benchmarks.catalogue_join import great_circle_km
from src.core.errors import InvalidParameterError


def _edge_distances(lat: float, lon: float, bounds: Mapping[str, float]) -> Dict[str, float]:
    return {
        "south": great_circle_km((lat, lon), (bounds["south"], lon)),
        "north": great_circle_km((lat, lon), (bounds["north"], lon)),
        "west": great_circle_km((lat, lon), (lat, bounds["west"])),
        "east": great_circle_km((lat, lon), (lat, bounds["east"])),
    }


def _stratum(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    finite = sorted(row["nearest_km"] for row in rows if row["nearest_km"] is not None)
    return {
        "rows": len(rows),
        "no_feature": sum(row["nearest_km"] is None for row in rows),
        "nearest_km": ({"min": finite[0], "median": median(finite), "max": finite[-1]}
                       if finite else None),
        "at_least_one_inside_radius": sum(row["inside_radius"] >= 1 for row in rows),
        "three_inside_radius": sum(row["inside_radius"] >= 3 for row in rows),
    }


def audit_edge_support(rows: Sequence[Mapping[str, Any]], *, south: float, north: float,
                       west: float, east: float, margin_degrees: float = 2.0
                       ) -> Dict[str, Any]:
    """Describe join error against support at all four crop edges.

    Missing features remain in the strata and worst-row list, but are excluded from the rank
    correlation because they have no finite error to rank.
    """
    if not south < north or not west < east:
        raise InvalidParameterError("bounds", [south, north, west, east],
                                    "strictly increasing latitude and longitude bounds")
    if margin_degrees <= 0 or not math.isfinite(margin_degrees):
        raise InvalidParameterError("margin_degrees", margin_degrees,
                                    "a positive finite angular support margin")
    bounds = {"south": south, "north": north, "west": west, "east": east}
    described = []
    for source in rows:
        lat, lon = float(source["lat"]), float(source["lon"])
        angular = {
            "south": lat - south, "north": north - lat,
            "west": lon - west, "east": east - lon,
        }
        if any(distance < 0 for distance in angular.values()):
            raise InvalidParameterError("row", dict(source), "a position inside the crop bounds")
        distances = _edge_distances(lat, lon, bounds)
        nearest_edge = min(distances, key=lambda edge: (distances[edge], edge))
        described.append({
            "storm": source["storm"],
            "sid": source.get("sid"),
            "lat": lat,
            "lon": lon,
            "nearest_km": source.get("nearest_km"),
            "inside_radius": int(source.get("inside_radius", 0)),
            "nearest_edge": nearest_edge,
            "edge_distance_km": distances[nearest_edge],
            "edge_distance_degrees": min(angular.values()),
            "near_edge": min(angular.values()) <= margin_degrees,
            "no_feature": source.get("nearest_km") is None,
        })

    finite = [row for row in described if row["nearest_km"] is not None]
    if len(finite) >= 2 and len({row["edge_distance_km"] for row in finite}) > 1 \
            and len({row["nearest_km"] for row in finite}) > 1:
        correlation = spearmanr(
            [row["edge_distance_km"] for row in finite],
            [row["nearest_km"] for row in finite])
        association = {"finite_rows": len(finite), "spearman_rho": float(correlation.statistic),
                       "two_sided_p_value": float(correlation.pvalue)}
    else:
        association = {"finite_rows": len(finite), "spearman_rho": None,
                       "two_sided_p_value": None}

    near_edge = [row for row in described if row["near_edge"]]
    interior = [row for row in described if not row["near_edge"]]
    worst_first = sorted(
        described,
        key=lambda row: (row["nearest_km"] is None,
                         row["nearest_km"] if row["nearest_km"] is not None else 0.0),
        reverse=True)
    return {
        "bounds": bounds,
        "support_margin_degrees": margin_degrees,
        "rows": described,
        "strata": {"near_edge": _stratum(near_edge), "interior": _stratum(interior)},
        "association": association,
        "worst_rows": worst_first[:min(6, len(worst_first))],
        "claim_boundary": (
            "Post-hoc descriptive diagnostic over previously inspected rows. Correlation and "
            "strata select no threshold, adjudicate no acceptance, and do not establish that "
            "crop truncation caused a join error."),
    }


__all__ = ["audit_edge_support"]