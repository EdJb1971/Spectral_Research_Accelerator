"""The canonical feature record: one structure, described so it can leave its domain (TG2.1).

**What this is for.** Section 2.3 of the cross-domain roadmap records that the entire middle of
the proposed pipeline - feature extraction, canonical description, tracking, constellations,
motif mining - does not exist. This module is the first piece of it, and it is deliberately the
*record* rather than the extractor: what a feature is has to be settled before anything decides
how to find one, because every later phase reads this vocabulary and rule R20 will eventually
freeze a motif expressed in it.

**The rule the record exists to enforce is R19.** Mapping sea-surface temperature and trading
volume into a common structural vocabulary makes their *structures* comparable. It does not make
the quantities comparable. So the canonical description does not replace the original one: the
domain, the dataset, the variable and the units travel with every feature and are never dropped
because a structural description exists. Domain and variable are **carried, never compared
numerically**, and the record refuses the comparison rather than documenting that you should not
make it - a comment is not an invariant.

**Two views, and only one of them crosses a domain boundary.**

*   `describe()` is the full record. It carries units, semantics and provenance, and is what a
    receipt stores and a reader reads.
*   `structural_signature()` is the dimensionless view: scale *ratios*, orientation differences
    and significance, with magnitude, units and variable absent by construction. This is what a
    cross-domain matcher is allowed to see. It is not a filtered `describe()` - it is built from
    the quantities that survive being stripped of their units, which is a much shorter list than
    it first appears.

**Every number carries its units, including the ones that have none.** `Quantity` holds a value,
its units and its uncertainty together, because these three are separated at exactly the moment
someone needs them together. A spatial scale in cells and a spatial scale in metres are not the
same quantity, and a domain declaring `no_physical_metric` has scales in cells and no route to
metres. `units=None` is a legitimate declaration - "this number has no unit" - and it is not the
same as "the unit was not recorded", which is why it must be passed explicitly.

**Orientation is the trap in this record.** An elongated ridge at 170 degrees and one at 350
degrees are the *same axis*; a wind vector at 170 degrees and one at 350 degrees are opposite
directions. The wrap period is a property of the quantity being described and not of the number,
so it is declared, registered and read from the declaration wherever the arithmetic happens
(standard E16). TG2.3's tracker gates association on orientation difference, and that gate is
wrong by a factor of two in one of these two cases if the convention is assumed.

**Why there is no `uncertainty` field.** The roadmap's field list names one, and a single
uncertainty for a record holding a magnitude in kelvin, a location in cells, a scale in metres
and an angle in degrees would be a number with no unit and no referent. Uncertainty lives with
each quantity, and `describe()["uncertainty"]` assembles the per-quantity view - so the field
the roadmap asks for is present in the receipt without a lie in the dataclass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.registry import Registry


class SemanticComparisonError(InvalidParameterError):
    """Raised when two features are compared in a way rule R19 forbids.

    The message names both sides in full, because the mistake this catches is not a typo: it
    is a plausible-looking line of code that averages a magnitude in kelvin with one in
    dimensionless trading volume and produces a number nobody can interpret.
    """

    def __init__(self, quantity: str, left: str, right: str, reason: str) -> None:
        super().__init__(
            "feature.%s" % quantity, "%s vs %s" % (left, right),
            "two features that may be compared on this quantity. %s Rule R19: structural "
            "comparison never licenses semantic comparison, so compare structure - scale "
            "ratios, orientation differences, relative geometry - or state the conversion "
            "explicitly and take responsibility for it." % reason,
            left=left, right=right)


# ---------------------------------------------------------------------------- quantities


@dataclass(frozen=True)
class Quantity:
    """A number, its units and its uncertainty, kept together.

    `units=None` means the quantity is genuinely dimensionless. It is not a default: a
    quantity whose units were never recorded is indistinguishable in a receipt from one that
    has none, and the two license completely different arithmetic.
    """

    value: float
    units: Optional[str]
    uncertainty: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise InvalidParameterError("Quantity.value", self.value, "a real number")
        if not math.isfinite(float(self.value)):
            raise InvalidParameterError(
                "Quantity.value", self.value,
                "a finite number. A NaN magnitude propagates silently through every "
                "aggregate that reads it")
        if self.units is not None and not str(self.units).strip():
            raise InvalidParameterError(
                "Quantity.units", self.units,
                "a unit name, or None for a genuinely dimensionless quantity. An empty "
                "string reads as 'not recorded', which is a third thing")
        if self.uncertainty is not None:
            if not math.isfinite(float(self.uncertainty)) or float(self.uncertainty) < 0.0:
                raise InvalidParameterError(
                    "Quantity.uncertainty", self.uncertainty,
                    "a non-negative finite uncertainty in the same units as the value")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "units", None if self.units is None else str(self.units))
        object.__setattr__(
            self, "uncertainty",
            None if self.uncertainty is None else float(self.uncertainty))

    def ratio_to(self, other: "Quantity", *, context: str = "quantity") -> float:
        """`self / other`, dimensionless - and only when the units actually cancel.

        This is the one operation that turns a quantity into something a cross-domain
        matcher may read, which is exactly why it refuses to run on mismatched units.
        """
        if self.units != other.units:
            raise SemanticComparisonError(
                context, self.describe_short(), other.describe_short(),
                "The units do not cancel: %r over %r is not dimensionless."
                % (self.units, other.units))
        if other.value == 0.0:
            raise InvalidParameterError(
                "Quantity.ratio_to", other.value,
                "a non-zero denominator. A scale ratio against a zero scale is not a "
                "large ratio, it is an undefined one")
        return float(self.value) / float(other.value)

    def difference_to(self, other: "Quantity", *, context: str = "quantity") -> "Quantity":
        """`self - other`, in the shared units, refusing when there are none shared."""
        if self.units != other.units:
            raise SemanticComparisonError(
                context, self.describe_short(), other.describe_short(),
                "A difference is only meaningful in shared units (%r and %r)."
                % (self.units, other.units))
        combined: Optional[float] = None
        if self.uncertainty is not None and other.uncertainty is not None:
            combined = math.hypot(self.uncertainty, other.uncertainty)
        return Quantity(float(self.value) - float(other.value), self.units, combined)

    def describe_short(self) -> str:
        return "%g %s" % (self.value, self.units if self.units is not None else
                          "(dimensionless)")

    def describe(self) -> Dict[str, Any]:
        return {"value": self.value, "units": self.units, "uncertainty": self.uncertainty}


# ---------------------------------------------------------------------------- orientation


@dataclass(frozen=True)
class OrientationConvention:
    """How an angle wraps, and whether it points.

    `period_degrees` is the whole story of the arithmetic: 180 for an undirected axis, 360
    for a direction. `directed` records *why*, so a receipt says which kind of angle it is
    rather than leaving the reader to infer it from the period.
    """

    period_degrees: float
    directed: bool
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.period_degrees) or self.period_degrees <= 0.0:
            raise InvalidParameterError(
                "OrientationConvention.period_degrees", self.period_degrees,
                "a positive wrap period in degrees")

    def wrap(self, degrees: float) -> float:
        """The canonical representative in `[0, period)`."""
        return float(degrees) % float(self.period_degrees)

    def difference(self, left: float, right: float) -> float:
        """The signed shortest turn from `right` to `left`, in `(-period/2, period/2]`."""
        half = self.period_degrees / 2.0
        delta = (float(left) - float(right)) % float(self.period_degrees)
        return delta - self.period_degrees if delta > half else delta

    def separation(self, left: float, right: float) -> float:
        """The unsigned angular separation - what a tracker's orientation gate compares."""
        return abs(self.difference(left, right))

    def describe(self) -> Dict[str, Any]:
        return {"period_degrees": self.period_degrees, "directed": self.directed,
                "description": self.description}


#: Angle conventions (standard E1). A fourth registers from outside `src/`.
ORIENTATION_CONVENTIONS: Registry[OrientationConvention] = Registry("orientation convention")

ORIENTATION_CONVENTIONS.add(
    "axis_180",
    OrientationConvention(180.0, False,
                          "An undirected axis: a ridge at 170 and one at 350 are the same "
                          "orientation, separated by 0 degrees and not by 180."),
    capabilities={"period_degrees": 180.0, "directed": False},
    description="Undirected orientation of an elongated structure, wrapping at 180 degrees.")

ORIENTATION_CONVENTIONS.add(
    "direction_360",
    OrientationConvention(360.0, True,
                          "A pointing direction: a motion at 170 and one at 350 are "
                          "opposed, separated by 180 degrees."),
    capabilities={"period_degrees": 360.0, "directed": True},
    description="Directed angle of a vector quantity such as propagation, wrapping at 360.")


def convention_for(name: str) -> OrientationConvention:
    return ORIENTATION_CONVENTIONS.get(name)


@dataclass(frozen=True)
class Orientation:
    """An angle that knows how it wraps.

    Two orientations under different conventions are not comparable, and the refusal is the
    point: an undirected axis angle and a directed vector angle answer different questions,
    and the tracker that gates on their difference would be silently wrong by up to a factor
    of two rather than loudly wrong once.
    """

    degrees: float
    convention: str = "axis_180"
    uncertainty_degrees: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.degrees)):
            raise InvalidParameterError(
                "Orientation.degrees", self.degrees, "a finite angle in degrees")
        convention_for(self.convention)          # refuse an unregistered convention
        if self.uncertainty_degrees is not None:
            if (not math.isfinite(float(self.uncertainty_degrees))
                    or float(self.uncertainty_degrees) < 0.0):
                raise InvalidParameterError(
                    "Orientation.uncertainty_degrees", self.uncertainty_degrees,
                    "a non-negative uncertainty in degrees")
        object.__setattr__(self, "degrees", self.convention_object().wrap(self.degrees))

    def convention_object(self) -> OrientationConvention:
        return convention_for(self.convention)

    def _require_same_convention(self, other: "Orientation") -> OrientationConvention:
        if self.convention != other.convention:
            raise SemanticComparisonError(
                "orientation", "%g deg (%s)" % (self.degrees, self.convention),
                "%g deg (%s)" % (other.degrees, other.convention),
                "An undirected axis angle and a directed angle are different quantities: "
                "170 and 350 degrees are the same axis and opposite directions.")
        return self.convention_object()

    def difference_to(self, other: "Orientation") -> float:
        """Signed shortest turn from `other` to `self`, under the shared convention."""
        return self._require_same_convention(other).difference(self.degrees, other.degrees)

    def separation_to(self, other: "Orientation") -> float:
        """Unsigned angular separation - the quantity a tracking gate thresholds."""
        return self._require_same_convention(other).separation(self.degrees, other.degrees)

    def describe(self) -> Dict[str, Any]:
        return {"degrees": self.degrees, "convention": self.convention,
                "uncertainty_degrees": self.uncertainty_degrees,
                **self.convention_object().describe()}


# ---------------------------------------------------------------------------- significance


@dataclass(frozen=True)
class SignificanceBasis:
    """How a feature's significance was established.

    `is_p_value` is load-bearing rather than descriptive: it is what makes the resolution
    floor checkable, and a basis that is not a p-value must not be checked against one.
    """

    is_p_value: bool
    description: str = ""

    def describe(self) -> Dict[str, Any]:
        return {"is_p_value": self.is_p_value, "description": self.description}


SIGNIFICANCE_BASES: Registry[SignificanceBasis] = Registry("significance basis")

SIGNIFICANCE_BASES.add(
    "surrogate_p_value",
    SignificanceBasis(True, "Empirical p-value against a surrogate ensemble, on the "
                           "(1 + k) / (1 + n) convention used everywhere in this tree."),
    capabilities={"is_p_value": True, "threshold_free": True},
    description="Surrogate-calibrated empirical p-value.")

SIGNIFICANCE_BASES.add(
    "surrogate_quantile",
    SignificanceBasis(False, "The feature's rank within the surrogate ensemble, reported "
                            "as a quantile in [0, 1] rather than converted to a p-value."),
    capabilities={"is_p_value": False, "threshold_free": True},
    description="Rank of the observed value within the surrogate ensemble.")

SIGNIFICANCE_BASES.add(
    "declared_none",
    SignificanceBasis(False, "No significance was established. Recorded explicitly so a "
                            "receipt distinguishes 'not tested' from 'tested and weak'."),
    capabilities={"is_p_value": False, "threshold_free": True},
    description="Significance was not established for this feature.")


@dataclass(frozen=True)
class Significance:
    """A significance value, how it was obtained, and how finely it could be resolved.

    The resolution check is the reason this is a type rather than a float. An empirical
    p-value from `n` surrogates cannot be smaller than `1 / (1 + n)`, and a record claiming
    `p = 1e-4` from 999 surrogates is reporting a number the ensemble could not have
    produced. That number would then be corrected, ranked and published.
    """

    value: float
    basis: str = "declared_none"
    n_surrogates: Optional[int] = None

    def __post_init__(self) -> None:
        entry = SIGNIFICANCE_BASES.entry(self.basis)   # refuse an unregistered basis
        if not math.isfinite(float(self.value)) or not 0.0 <= float(self.value) <= 1.0:
            raise InvalidParameterError(
                "Significance.value", self.value,
                "a finite value in [0, 1]", basis=self.basis)
        if self.n_surrogates is not None and int(self.n_surrogates) < 0:
            raise InvalidParameterError(
                "Significance.n_surrogates", self.n_surrogates,
                "a non-negative surrogate count")
        if entry.value.is_p_value:
            if self.n_surrogates is None:
                raise InvalidParameterError(
                    "Significance.n_surrogates", None,
                    "the size of the surrogate ensemble, because basis %r is an empirical "
                    "p-value and its resolution floor is 1 / (1 + n)" % self.basis)
            floor = 1.0 / (1.0 + float(self.n_surrogates))
            if float(self.value) < floor - 1e-12:
                raise InvalidParameterError(
                    "Significance.value", self.value,
                    "a p-value no smaller than the ensemble's resolution floor %.6g "
                    "(1 / (1 + %d)). A smaller number was not measured; it was assumed"
                    % (floor, int(self.n_surrogates)),
                    basis=self.basis, n_surrogates=int(self.n_surrogates))
        object.__setattr__(self, "value", float(self.value))

    @property
    def resolution_floor(self) -> Optional[float]:
        if self.n_surrogates is None:
            return None
        return 1.0 / (1.0 + float(self.n_surrogates))

    def describe(self) -> Dict[str, Any]:
        return {"value": self.value, "basis": self.basis,
                "n_surrogates": self.n_surrogates,
                "resolution_floor": self.resolution_floor,
                **SIGNIFICANCE_BASES.get(self.basis).describe()}


# ---------------------------------------------------------------------------- location


@dataclass(frozen=True)
class FeatureLocation:
    """Where a feature is, on declared axes, in the units those axes declare.

    The axes are `AxisSpec`s (standard E14) and not a tuple of numbers, because "row 32,
    column 41" means nothing without knowing that both axes are spatial, that they are in
    cells rather than metres, and that one of them wraps. A separation computed across a
    periodic axis without its period is not slightly wrong; it is wrong by the length of the
    axis, at exactly the frames a tracker most needs to be right.
    """

    coords: Mapping[str, float]
    axes: Sequence[AxisSpec]
    uncertainty: Mapping[str, float] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes:
            raise InvalidParameterError(
                "FeatureLocation.axes", [],
                "at least one declared axis. A location on no axis is not a location")
        for axis in axes:
            if not isinstance(axis, AxisSpec):
                raise InvalidParameterError(
                    "FeatureLocation.axes", axis,
                    "an AxisSpec. A name is not a role (standard E14)")
            if axis.role == "time":
                raise InvalidParameterError(
                    "FeatureLocation.axes", axis.name,
                    "an axis that is not the clock. A feature's time is a field of its own, "
                    "and a second copy inside the location is a second clock that can "
                    "disagree with the first")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise InvalidParameterError(
                "FeatureLocation.axes", names, "distinct axis names")
        if set(self.coords) != set(names):
            raise InvalidParameterError(
                "FeatureLocation.coords", sorted(self.coords),
                "one coordinate per declared axis: %s" % ", ".join(sorted(names)))
        for name, value in self.coords.items():
            if not math.isfinite(float(value)):
                raise InvalidParameterError(
                    "FeatureLocation.coords", value,
                    "a finite coordinate on axis %r" % name)
        for name in self.uncertainty:
            if name not in set(names):
                raise UnknownNameError("location axis", name, names)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(
            self, "coords", {str(k): float(v) for k, v in self.coords.items()})
        object.__setattr__(
            self, "uncertainty", {str(k): float(v) for k, v in self.uncertainty.items()})

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    def axis(self, name: str) -> AxisSpec:
        for spec in self.axes:
            if spec.name == name:
                return spec
        raise UnknownNameError("location axis", name, self.axis_names)

    @property
    def units(self) -> Tuple[Optional[str], ...]:
        return tuple(axis.units for axis in self.axes)

    def separation_to(self, other: "FeatureLocation",
                      periods: Optional[Mapping[str, float]] = None) -> Quantity:
        """Euclidean separation, in the axes' shared units, on the declared topology.

        Refuses three things, all of which produce a plausible number if allowed:

        *   **different axes**, because a separation between differently-shaped locations is
            a coincidence of dimension count;
        *   **mixed units across the axes**, because the square root of cells-squared plus
            metres-squared is not a length; and
        *   **a periodic axis with no declared period**, because the wrap distance cannot be
            computed without the axis length and the un-wrapped answer is confidently wrong.
        """
        if self.axis_names != other.axis_names:
            raise SemanticComparisonError(
                "location", ", ".join(self.axis_names), ", ".join(other.axis_names),
                "The two locations are on different axes.")
        units = set(self.units)
        if len(units) != 1:
            raise SemanticComparisonError(
                "location", str(self.units), str(other.units),
                "The axes carry mixed units %s, so their squares do not add to a length."
                % sorted(str(u) for u in units))
        if self.units != other.units:
            raise SemanticComparisonError(
                "location", str(self.units), str(other.units),
                "The same axis names carry different units on the two locations.")
        periods = dict(periods or {})
        for name in periods:
            if name not in self.axis_names:
                raise UnknownNameError("location axis", name, self.axis_names)
        total = 0.0
        for name in self.axis_names:
            delta = abs(self.coords[name] - other.coords[name])
            spec = self.axis(name)
            if spec.periodic:
                if name not in periods:
                    raise InvalidParameterError(
                        "FeatureLocation.separation_to", name,
                        "the length of periodic axis %r, so the wrap distance can be "
                        "computed. Axis %r declares periodic=True and the separation "
                        "across the seam is not |a - b|" % (name, name),
                        axis=name, periodic=True)
                period = float(periods[name])
                if not math.isfinite(period) or period <= 0.0:
                    raise InvalidParameterError(
                        "FeatureLocation.separation_to", period,
                        "a positive length for periodic axis %r" % name)
                delta = min(delta, period - (delta % period))
            elif name in periods:
                raise InvalidParameterError(
                    "FeatureLocation.separation_to", name,
                    "a period only for an axis that declares periodic=True. Axis %r does "
                    "not, and wrapping it would join two ends that are not joined" % name,
                    axis=name, periodic=False)
            total += delta * delta
        combined: Optional[float] = None
        if self.uncertainty and other.uncertainty:
            left = sum(v * v for v in self.uncertainty.values())
            right = sum(v * v for v in other.uncertainty.values())
            combined = math.sqrt(left + right)
        return Quantity(math.sqrt(total), self.units[0], combined)

    def describe(self) -> Dict[str, Any]:
        return {
            "coords": dict(self.coords),
            "axes": [axis.describe() for axis in self.axes],
            "uncertainty": dict(self.uncertainty),
        }


# ---------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class SpectralFeature:
    """One structure, described canonically without losing what it actually was.

    Fifteen things travel with a feature, and the split between them is the whole design:

    *   **carried, never compared numerically** - `domain`, `dataset`, `variable`, and the
        units inside every quantity. These are what R19 protects.
    *   **structural, and therefore transferable once made dimensionless** - `spatial_scale`,
        `temporal_scale`, `orientation`, `extent`, `significance`.
    *   **positional** - `location` and `time`, which are comparable within a domain and
        meaningless across one.
    *   **accountable** - `representation` and `provenance`. `representation` is required:
        rule R8's lesson is that a representation can manufacture a motif, and TG2.4's audit
        cannot ask *which* representation produced a feature if the feature does not say.
    """

    domain: str
    dataset: str
    variable: str
    magnitude: Quantity
    location: FeatureLocation
    time: float
    representation: str
    time_units: Optional[str] = None
    spatial_scale: Optional[Quantity] = None
    temporal_scale: Optional[Quantity] = None
    orientation: Optional[Orientation] = None
    extent: Optional[Quantity] = None
    significance: Optional[Significance] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("domain", "dataset", "variable", "representation"):
            value = getattr(self, name)
            if not str(value).strip():
                raise InvalidParameterError(
                    "SpectralFeature.%s" % name, value,
                    "a non-empty %s. A feature that cannot name its %s cannot be refused "
                    "by name either, and every R19 refusal in this module is by name"
                    % (name, name))
        if not isinstance(self.magnitude, Quantity):
            raise InvalidParameterError(
                "SpectralFeature.magnitude", self.magnitude,
                "a Quantity carrying the value and its units. A bare float is a magnitude "
                "whose units have to be looked up somewhere else, which is how a kelvin "
                "gets compared with a count")
        if not isinstance(self.location, FeatureLocation):
            raise InvalidParameterError(
                "SpectralFeature.location", self.location, "a FeatureLocation")
        if not math.isfinite(float(self.time)):
            raise InvalidParameterError(
                "SpectralFeature.time", self.time, "a finite time on the declared clock")
        for name in ("spatial_scale", "temporal_scale", "extent"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Quantity):
                raise InvalidParameterError(
                    "SpectralFeature.%s" % name, value, "a Quantity or None")
            if value is not None and value.value <= 0.0:
                raise InvalidParameterError(
                    "SpectralFeature.%s" % name, value.value,
                    "a positive %s. A zero or negative one has no ratio, and a ratio is "
                    "the only form in which it can leave this domain" % name)
        if self.orientation is not None and not isinstance(self.orientation, Orientation):
            raise InvalidParameterError(
                "SpectralFeature.orientation", self.orientation,
                "an Orientation declaring how it wraps, or None")
        if self.significance is not None and not isinstance(self.significance, Significance):
            raise InvalidParameterError(
                "SpectralFeature.significance", self.significance,
                "a Significance declaring its basis, or None")
        if self.time_units is not None and not str(self.time_units).strip():
            raise InvalidParameterError(
                "SpectralFeature.time_units", self.time_units,
                "the units of the clock, or None when the clock is in frames and says so")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "provenance", dict(self.provenance))

    # ------------------------------------------------------------------ identity

    @property
    def semantic_key(self) -> Tuple[str, str, str]:
        """What must match before two magnitudes may be compared at all (R19)."""
        return (self.domain, self.dataset, self.variable)

    @property
    def units(self) -> Optional[str]:
        """The units of the magnitude - the roadmap's `units` field, kept where it is used."""
        return self.magnitude.units

    def same_domain_as(self, other: "SpectralFeature") -> bool:
        return self.domain == other.domain

    # ------------------------------------------------------------------ comparison

    def compare_magnitude_to(self, other: "SpectralFeature") -> Quantity:
        """The difference in magnitude - permitted only within one variable of one dataset.

        This method exists to be refused. Nothing in the tree needs a cross-domain magnitude
        difference; what the tree needs is for the line that computes one to fail loudly the
        first time somebody writes it.
        """
        if self.semantic_key != other.semantic_key:
            raise SemanticComparisonError(
                "magnitude", "%s/%s/%s (%s)" % (self.semantic_key + (self.units,)),
                "%s/%s/%s (%s)" % (other.semantic_key + (other.units,)),
                "Magnitudes are only comparable within one variable of one dataset of one "
                "domain.")
        return self.magnitude.difference_to(other.magnitude, context="magnitude")

    def scale_ratio_to(self, other: "SpectralFeature") -> float:
        """`self.spatial_scale / other.spatial_scale`, dimensionless.

        Permitted across domains, unlike a magnitude difference: a structure twice the size
        of its neighbour is twice the size of its neighbour whatever the neighbour is made
        of. The units must still cancel, so a scale in cells and one in metres are refused -
        that is not a domain boundary, it is an arithmetic one.
        """
        if self.spatial_scale is None or other.spatial_scale is None:
            raise InvalidParameterError(
                "SpectralFeature.spatial_scale", None,
                "a spatial scale on both features. A ratio against an unrecorded scale is "
                "not a ratio")
        return self.spatial_scale.ratio_to(other.spatial_scale, context="spatial_scale")

    def separation_to(self, other: "SpectralFeature",
                      periods: Optional[Mapping[str, float]] = None) -> Quantity:
        """Distance between two features - refused across a domain boundary.

        Two features in different domains have no common space, and a number computed from
        their coordinates would be a number about two arrays rather than about the world.
        """
        if self.domain != other.domain or self.dataset != other.dataset:
            raise SemanticComparisonError(
                "location", "%s/%s" % (self.domain, self.dataset),
                "%s/%s" % (other.domain, other.dataset),
                "Two datasets do not share a coordinate system merely by having "
                "coordinates.")
        return self.location.separation_to(other.location, periods=periods)

    def elapsed_to(self, other: "SpectralFeature") -> float:
        """`self.time - other.time` on a shared clock, refused when the clocks differ."""
        if self.domain != other.domain or self.dataset != other.dataset:
            raise SemanticComparisonError(
                "time", "%s/%s" % (self.domain, self.dataset),
                "%s/%s" % (other.domain, other.dataset),
                "Two datasets do not share a clock merely by having timestamps.")
        if self.time_units != other.time_units:
            raise SemanticComparisonError(
                "time", str(self.time_units), str(other.time_units),
                "The two clocks are in different units.")
        return float(self.time) - float(other.time)

    # ------------------------------------------------------------------ views

    def structural_signature(self) -> Dict[str, Any]:
        """The dimensionless view - what may cross a domain boundary, and nothing else.

        Built from the fields that survive being stripped of their units, not filtered out
        of `describe()`. Magnitude, units, variable, dataset, location and time are absent
        by construction: adding one back is a visible edit to this method rather than an
        invisible consequence of a key not being removed.
        """
        signature: Dict[str, Any] = {
            "representation": self.representation,
            "has_spatial_scale": self.spatial_scale is not None,
            "has_temporal_scale": self.temporal_scale is not None,
            "orientation_convention":
                None if self.orientation is None else self.orientation.convention,
            "orientation_degrees":
                None if self.orientation is None else self.orientation.degrees,
            "significance":
                None if self.significance is None else self.significance.value,
            "significance_basis":
                None if self.significance is None else self.significance.basis,
        }
        if self.spatial_scale is not None and self.extent is not None:
            try:
                signature["extent_in_scales"] = self.extent.ratio_to(
                    self.spatial_scale, context="extent")
            except SemanticComparisonError:
                # Extent and scale in different units is a real state - an extent in cells
                # beside a scale in metres - and it is reported as absent, not as zero.
                signature["extent_in_scales"] = None
        else:
            signature["extent_in_scales"] = None
        return signature

    def describe(self) -> Dict[str, Any]:
        """The full record, with every one of the roadmap's fifteen fields present."""
        return {
            "domain": self.domain,
            "dataset": self.dataset,
            "variable": self.variable,
            "units": self.units,
            "location": self.location.describe(),
            "time": self.time,
            "time_units": self.time_units,
            "spatial_scale":
                None if self.spatial_scale is None else self.spatial_scale.describe(),
            "temporal_scale":
                None if self.temporal_scale is None else self.temporal_scale.describe(),
            "orientation":
                None if self.orientation is None else self.orientation.describe(),
            "magnitude": self.magnitude.describe(),
            "significance":
                None if self.significance is None else self.significance.describe(),
            "extent": None if self.extent is None else self.extent.describe(),
            "uncertainty": self.uncertainty_report(),
            "representation": self.representation,
            "provenance": dict(self.provenance),
        }

    def uncertainty_report(self) -> Dict[str, Any]:
        """Per-quantity uncertainty, in each quantity's own units.

        The roadmap asks for an `uncertainty` field; this is it, and it is a mapping rather
        than a number because the record holds quantities in four different units and a
        single scalar over them would have none.
        """
        report: Dict[str, Any] = {
            "magnitude": self.magnitude.uncertainty,
            "magnitude_units": self.magnitude.units,
            "location": dict(self.location.uncertainty),
            "orientation_degrees":
                None if self.orientation is None else self.orientation.uncertainty_degrees,
        }
        for name in ("spatial_scale", "temporal_scale", "extent"):
            value = getattr(self, name)
            report[name] = None if value is None else value.uncertainty
            report["%s_units" % name] = None if value is None else value.units
        return report


# ---------------------------------------------------------------------------- collections


@dataclass(frozen=True)
class FeatureSet:
    """Features from one dataset, one variable and one representation, ordered in time.

    **Why homogeneity is enforced rather than assumed.** Everything TG2.3 and TG3.3 will do
    with a set - associate frame to frame, measure relative geometry, mine repeated
    configurations - is arithmetic on coordinates, scales and times. A set that quietly
    mixed two domains would let all of that run, and every number it produced would be a
    comparison rule R19 forbids. A cross-domain collection is a deliberate act with a frozen
    definition (R20, phase G5), not a list that happened to have two things in it.

    Mixing *representations* is refused for the same reason and one more: TG2.4's audit asks
    which representation manufactured a feature, and it cannot ask that of a set whose
    features came from several.
    """

    features: Sequence[SpectralFeature]

    def __post_init__(self) -> None:
        features = tuple(self.features)
        if not features:
            raise InvalidParameterError(
                "FeatureSet.features", [],
                "at least one feature. An empty set has no domain, and 'no features found' "
                "is a result that belongs beside the search that produced it")
        for item in features:
            if not isinstance(item, SpectralFeature):
                raise InvalidParameterError(
                    "FeatureSet.features", item, "a SpectralFeature")
        first = features[0]
        for item in features[1:]:
            if item.semantic_key != first.semantic_key:
                raise SemanticComparisonError(
                    "feature_set", "/".join(first.semantic_key),
                    "/".join(item.semantic_key),
                    "A feature set is one variable of one dataset of one domain.")
            if item.representation != first.representation:
                raise SemanticComparisonError(
                    "feature_set", first.representation, item.representation,
                    "A feature set is the output of one representation, so that a "
                    "representation-induced feature can be attributed to it (rule R8).")
            if item.time_units != first.time_units:
                raise SemanticComparisonError(
                    "feature_set", str(first.time_units), str(item.time_units),
                    "The features are on clocks with different units.")
        object.__setattr__(self, "features",
                           tuple(sorted(features, key=lambda f: f.time)))

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[SpectralFeature]:
        return iter(self.features)

    def __getitem__(self, index: int) -> SpectralFeature:
        return self.features[index]

    @property
    def domain(self) -> str:
        return self.features[0].domain

    @property
    def dataset(self) -> str:
        return self.features[0].dataset

    @property
    def variable(self) -> str:
        return self.features[0].variable

    @property
    def representation(self) -> str:
        return self.features[0].representation

    @property
    def times(self) -> Tuple[float, ...]:
        return tuple(f.time for f in self.features)

    def frames(self) -> Tuple[Tuple[float, Tuple[SpectralFeature, ...]], ...]:
        """Features grouped by time, in ascending time order.

        This is the shape frame-to-frame association consumes in TG2.3, and it is built
        here rather than there because the grouping key is a property of the record - a
        feature's time is exact and shared within a frame, never binned.
        """
        grouped: Dict[float, list] = {}
        for item in self.features:
            grouped.setdefault(item.time, []).append(item)
        return tuple((t, tuple(grouped[t])) for t in sorted(grouped))

    def at_time(self, time: float) -> Tuple[SpectralFeature, ...]:
        return tuple(f for f in self.features if f.time == float(time))

    def describe(self) -> Dict[str, Any]:
        """The receipt for a set: what it is, and every feature in it."""
        return {
            "domain": self.domain,
            "dataset": self.dataset,
            "variable": self.variable,
            "representation": self.representation,
            "count": len(self.features),
            "frame_count": len(self.frames()),
            "time_units": self.features[0].time_units,
            "time_span": [self.features[0].time, self.features[-1].time],
            "features": [f.describe() for f in self.features],
        }
