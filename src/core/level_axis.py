"""Vertical coordinates are declared, and pressure is one of them (TG1.5, standards E1/E14).

**The assumption being localised.** `LevelBank` was keyed on `Dict[float, ...]` where the float
was pressure in hectopascals, `ScaleSignature` carried a field literally called `level_hpa`, and
`CoefficientField.level` carried a bare number with no units at all. The unit was in the
attribute name in one place, in a docstring in another, and nowhere in the third.

**And it was not only a naming problem.** `LevelBank.vertical_offsets` decided direction with

    "direction": "upward" if upper < lower else "downward"

which is true of pressure and of ocean depth, and exactly backwards for geopotential height or
altitude. The convention that makes a smaller number mean *higher* is a fact about the
coordinate, not about verticality, and it was hardcoded at the one place that reports the
direction of a vertical precursor relationship - the sign that distinguishes an upper-level
trough from a surface one. No such coordinate exists in the tree today, so nothing shipped
wrong; the convention is now read from the declared coordinate rather than assumed, which is
what standard E16 asks: the assumption lives where the arithmetic happens.

**What a level coordinate declares.** Its units, and which way is up. That is all, deliberately:
this module does not know about atmospheres, oceans or soil profiles, and adding a fourth
coordinate is a registration rather than an edit.

**Ordering is separate from direction, and both are recorded.** Levels are always ordered by
increasing coordinate value, because that is what `LevelBank` already did and changing it would
reorder the stacked tensor of every existing bank. Which end of that order is the top is then a
property of the coordinate: ascending pressure runs downward, ascending height runs upward.
`describe()` says which, so a reader of a receipt never has to know the convention to read the
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.registry import Registry

#: The default vertical coordinate, and the only one the atmospheric line uses. Named once,
#: here, so the places that carry a level can declare it rather than assume it - and so that
#: searching for the atmospheric assumption finds a constant instead of a docstring.
PRESSURE_HPA = "pressure_hpa"


@dataclass(frozen=True)
class LevelCoordinate:
    """One vertical coordinate: its units, and which way along it is up.

    `increases_upward` is the whole content. Pressure and depth decrease and increase in the
    opposite senses to height, and a system that reports "an upper-level precursor" has to know
    which, because the sign of the offset is the finding.
    """

    units: str
    increases_upward: bool
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.units).strip():
            raise InvalidParameterError(
                "LevelCoordinate.units", self.units,
                "non-empty units. A vertical coordinate with no unit is the situation this "
                "module exists to end: the number would mean whatever the reader assumed")

    # ------------------------------------------------------------------ the convention

    def direction(self, from_level: float, to_level: float) -> str:
        """``"upward"``, ``"downward"``, or ``"same"`` - read from the coordinate, not guessed."""
        if float(to_level) == float(from_level):
            return "same"
        rising = float(to_level) > float(from_level)
        return "upward" if rising == self.increases_upward else "downward"

    def offset(self, from_level: float, to_level: float) -> float:
        """The signed offset in this coordinate's own units, `to - from`."""
        return float(to_level) - float(from_level)

    def ordered(self, values: Sequence[float]) -> Tuple[float, ...]:
        """Levels in ascending coordinate value - the order a bank stacks them in."""
        return tuple(sorted(float(value) for value in values))

    @property
    def ascending_runs(self) -> str:
        """Which way the stacking order travels. The sentence a receipt should carry."""
        return "upward" if self.increases_upward else "downward"

    def axis_spec(self, name: str = "level") -> Any:
        """This coordinate as a declared `level`-role `AxisSpec` (TG1.1, TG1.4).

        The literal form of what the roadmap asked for. Imported lazily because `domain.py`
        pulls in the lag-policy registry, and a vertical coordinate should not depend on the
        admissibility machinery to describe itself.
        """
        from src.core.domain import AxisSpec

        return AxisSpec(name=name, role="level", units=self.units, ordered=True)

    def describe(self, name: str) -> Dict[str, Any]:
        return {
            "level_axis": name,
            "level_units": self.units,
            "increases_upward": self.increases_upward,
            "ascending_order_runs": self.ascending_runs,
        }


#: Registered vertical coordinates. Pressure is the first entry, not the definition; a domain
#: with a vertical axis this registry does not know adds one without editing `src/`.
LEVEL_COORDINATES: Registry[LevelCoordinate] = Registry("level coordinate")

LEVEL_COORDINATES.add(
    PRESSURE_HPA,
    LevelCoordinate("hPa", increases_upward=False,
                    description="Atmospheric pressure. Decreases upward, so 500 hPa is above "
                                "850 hPa and an ascending stack runs from the top down."),
    capabilities={"units": "hPa", "increases_upward": False, "metric": False},
    description="Pressure in hectopascals - the atmospheric line's vertical coordinate.")
LEVEL_COORDINATES.add(
    "height_m",
    LevelCoordinate("m", increases_upward=True,
                    description="Geometric or geopotential height above a reference surface. "
                                "Increases upward, the opposite sense to pressure."),
    capabilities={"units": "m", "increases_upward": True, "metric": True},
    description="Height above a reference surface, in metres.")
LEVEL_COORDINATES.add(
    "depth_m",
    LevelCoordinate("m", increases_upward=False,
                    description="Depth below a reference surface - an ocean profile or a soil "
                                "column. Increases downward, the same sense as pressure and "
                                "for an unrelated reason."),
    capabilities={"units": "m", "increases_upward": False, "metric": True},
    description="Depth below a reference surface, in metres.")


def coordinate_for(name: str) -> LevelCoordinate:
    """The registered coordinate, or an `UnknownNameError` naming the alternatives."""
    return LEVEL_COORDINATES.get(name)


def capability(name: str, key: str, default: Any = None) -> Any:
    return LEVEL_COORDINATES.entry(name).capabilities.get(key, default)


def coordinate_names() -> Tuple[str, ...]:
    return tuple(LEVEL_COORDINATES.names())


def units_of(name: Optional[str]) -> Optional[str]:
    """Units for a declared axis, or `None` for a level whose axis nobody declared.

    `None` is returned rather than a guess, and rather than a refusal: a level that arrived
    without its coordinate is a real state of the tree (a bare `CoefficientField.level` set by
    hand), and the honest report of it is "no units were declared", not "hPa".
    """
    return None if name is None else coordinate_for(name).units


def require_pressure(name: Optional[str], context: str) -> None:
    """Refuse a non-pressure axis where a caller has asked for hectopascals by name.

    The counterpart of `units_of`'s tolerance. Reading a number *as* hPa is a claim, and a
    claim about a height axis is wrong by a factor of nothing - it is simply a different
    quantity - so it is refused rather than converted or returned as `None`. A `None` would
    read as "no level", which is a lie of omission about a level that exists.
    """
    if name == PRESSURE_HPA:
        return
    raise InvalidParameterError(
        "level_axis", name,
        "%r, because %s asks for a level in hectopascals. This record declares %s. Read "
        "`level` together with `level_axis` instead: the value is real, it is simply not a "
        "pressure" % (PRESSURE_HPA, context,
                      "no vertical coordinate at all" if name is None else "%r (%s)"
                      % (name, coordinate_for(name).units)))


__all__ = [
    "PRESSURE_HPA",
    "LevelCoordinate",
    "LEVEL_COORDINATES",
    "coordinate_for",
    "capability",
    "coordinate_names",
    "units_of",
    "require_pressure",
]
