"""T4E.27: how close is close enough, said in parts rather than as a number.

T4E.18 tested whether an extracted feature lands at a catalogue cyclone centre, and used the
catalogue's own per-observation radius as the bar. That bar was wrong for three reasons, none of
which is that it failed:

* **It bounds the wrong quantity.** An agency's radius is its uncertainty about *its own* answer
  -- where the surface centre is. It is not a bound on the separation between two *different*
  quantities, and an 850 hPa relative-vorticity maximum is a different quantity.
* **Part of it could not be satisfied by any measurement.** The radii in that record run from
  0.00 to 145.23 km. A radius of exactly zero admits nothing, so at least one storm was
  unpassable however good the extraction was. A zero there is a missing agency report written
  down as a number, and this programme's rule about named refusals had been applied to its
  outputs and not to this input.
* **It is not the scale of the error.** T4E.19 measured the correlation between offset and
  radius at +0.020, with a leave-one-storm-out range straddling zero.

**Why this is a module and not a constant.** A tolerance that appears as a number can only be
accepted or rejected. One that publishes its components, where each came from, and what it
refused can be disagreed with *specifically* -- and that is what lets somebody ask a question
this programme did not think of, with the same instrument, without editing it.

**What is deliberately missing, and why that is the point.** The separation between a
vorticity maximum and a surface centre is not a component here. No defensible number for it
exists in this repository, and adding one sized to make an acceptance pass would be fitting the
bar to the result. Whatever distance the justified components fail to explain *is* the measure
of that missing term, and `unexplained_residual` reports it rather than absorbing it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

#: T4E.21 measured the extractor's own localisation error against known ground truth, at the
#: record's own feature density: 0.295 cells. It was measured on synthetic scenes for a different
#: purpose before this task existed, and is used as measured rather than re-derived here.
DECLARED_LOCALISATION_CELLS: float = 0.295

#: T4E.18's acquisition design fixed the grid at 0.25 degrees before any of this data existed.
#: One quarter-degree of latitude is 27.75 km, so the localisation error is 8.2 km.
DECLARED_GRID_KM_PER_CELL: float = 27.75

#: Why a radius of exactly zero is refused rather than used. IBTrACS reports no uncertainty for
#: some observations and the field arrives as 0.0; treating that as a tolerance of zero asks a
#: measurement to be exactly right, which nothing can be.
MISSING_RADIUS_SENTINEL: float = 0.0


def localisation_km(*, cells: float = DECLARED_LOCALISATION_CELLS,
                    km_per_cell: float = DECLARED_GRID_KM_PER_CELL) -> float:
    """The estimator component, in kilometres. 0.295 cells at 27.75 km/cell is 8.19 km."""
    return float(cells) * float(km_per_cell)


@dataclass(frozen=True)
class PositionTolerance:
    """One observation's tolerance, and everything it was built from.

    `total_km` is `None` when the tolerance could not be built. That is not a tolerance of zero
    and not a tolerance of infinity: it is a refusal, and `refusal` says why in words.
    """

    observation: str
    catalogue_km: Optional[float]
    estimator_km: float
    total_km: Optional[float]
    refusal: Optional[str]
    components: Tuple[Tuple[str, Optional[float], str], ...]

    @property
    def refused(self) -> bool:
        return self.total_km is None

    def admits(self, distance_km: float) -> Optional[bool]:
        """Is a separation inside this tolerance? `None` when the tolerance was refused.

        A refused tolerance returns `None` rather than `False`, because "we could not say" and
        "no" are different answers and a caller that conflates them will count a missing agency
        report as a failed detection.
        """
        if self.total_km is None:
            return None
        return float(distance_km) <= self.total_km

    def unexplained_residual(self, distance_km: float) -> Optional[float]:
        """How much of a separation the justified components do not account for.

        This is the measure of the component this tolerance deliberately omits -- the difference
        between the two quantities being joined. Negative values are returned as measured rather
        than clipped to zero, because a separation comfortably inside tolerance is information.
        """
        if self.total_km is None:
            return None
        return float(distance_km) - self.total_km

    def describe(self) -> Dict[str, object]:
        return {
            "observation": self.observation,
            "total_km": self.total_km,
            "refused": self.refused,
            "refusal": self.refusal,
            "components": [{"name": name, "km": value, "source": source}
                           for name, value, source in self.components],
            "what_is_not_included": (
                "The separation between an 850 hPa relative-vorticity maximum and a surface "
                "centre. No defensible number for it exists here, and inventing one would fit "
                "the bar to the result. The residual measures it instead."),
        }


def tolerance_for(observation: str, catalogue_radius_km: Optional[float],
                  *, estimator_km: Optional[float] = None) -> PositionTolerance:
    """Build one observation's tolerance from its declared parts.

    The catalogue component is refused when the agency reported nothing. The estimator component
    is always present, because it is a property of this instrument rather than of the catalogue.
    """
    estimator = localisation_km() if estimator_km is None else float(estimator_km)
    catalogue_source = ("IBTrACS per-observation reported radius, T4E.17 signed catalogue")
    estimator_source = (
        "T4E.21 measured localisation error on known ground truth at the record's own feature "
        "density: %.3f cells at %.2f km/cell" % (DECLARED_LOCALISATION_CELLS,
                                                 DECLARED_GRID_KM_PER_CELL))

    radius: Optional[float] = (None if catalogue_radius_km is None
                               else float(catalogue_radius_km))
    if radius is None or not math.isfinite(radius) or radius <= MISSING_RADIUS_SENTINEL:
        return PositionTolerance(
            observation=str(observation), catalogue_km=None, estimator_km=estimator,
            total_km=None,
            refusal=(
                "the catalogue reports no usable position uncertainty for this observation (the "
                "field is %s). A zero is a missing report written down as a number, and using "
                "it would demand a separation of exactly zero, which no measurement can "
                "supply; a non-finite one is a bar that admits everything, which is worse than "
                "no bar at all. Refused by name and excluded from the denominator rather than "
                "defaulted"
                % ("absent" if catalogue_radius_km is None else repr(catalogue_radius_km))),
            components=(("catalogue_uncertainty", None, catalogue_source),
                        ("estimator_localisation", estimator, estimator_source)))

    total = math.sqrt(radius * radius + estimator * estimator)
    return PositionTolerance(
        observation=str(observation), catalogue_km=radius, estimator_km=estimator,
        total_km=total, refusal=None,
        components=(("catalogue_uncertainty", radius, catalogue_source),
                    ("estimator_localisation", estimator, estimator_source)))


def acceptance_over(distances_km: Sequence[float],
                    tolerances: Sequence[PositionTolerance]) -> Dict[str, object]:
    """Count how many observations the bar admits, and say what it could not judge.

    The refused observations leave the denominator. Keeping them in would count a missing agency
    report as a failed detection, which is the error this whole task exists to correct.
    """
    if len(distances_km) != len(tolerances):
        raise ValueError("one distance per tolerance; got %d and %d"
                         % (len(distances_km), len(tolerances)))
    admitted, judged, refused = 0, 0, []
    residuals = []
    for distance, tolerance in zip(distances_km, tolerances):
        verdict = tolerance.admits(distance)
        if verdict is None:
            refused.append(tolerance.observation)
            continue
        judged += 1
        admitted += 1 if verdict else 0
        residuals.append(tolerance.unexplained_residual(distance))
    return {
        "admitted": admitted,
        "judged": judged,
        "refused": len(refused),
        "refused_observations": tuple(refused),
        "rate": (admitted / judged) if judged else None,
        "median_unexplained_residual_km": (
            float(sorted(residuals)[len(residuals) // 2]) if residuals else None),
    }


def acceptance_curve(distances_km: Sequence[float],
                     tolerances: Sequence[PositionTolerance],
                     grid_km: Sequence[float]) -> Tuple[Dict[str, object], ...]:
    """The verdict as a function of a flat tolerance, for inspection only.

    This exists so the declared bar can be argued with rather than merely believed. It is **not**
    a menu: choosing a point from it after seeing the verdicts is the horse race R20 forbids, and
    the declared bar decides the acceptance regardless of what this curve shows.
    """
    judged = [t for t in tolerances if not t.refused]
    keep = [d for d, t in zip(distances_km, tolerances) if not t.refused]
    out = []
    for bar in grid_km:
        admitted = sum(1 for d in keep if float(d) <= float(bar))
        out.append({"tolerance_km": float(bar), "admitted": admitted, "judged": len(judged),
                    "rate": (admitted / len(judged)) if judged else None})
    return tuple(out)
