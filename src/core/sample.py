"""The sibling sample spine: structured observations that are not 2D metric grids (TG1.4, E12).

**The obstacle, named in section 2.2 of the cross-domain roadmap.** ``PhysicalField.__init__``
raises on anything that is not exactly 2D, and everything above it inherits that constraint:
`FieldSequence`, `CoefficientField`, `ScaleSignature`, the transform registry, `surrogate_null`.
The audit called it "the single largest obstacle" to a second domain.

**What this module does not do.** It does not relax that constraint. Standard E12 is explicit:
generalise *beside*, never weaken. `PhysicalField` stays strictly 2D, coordinate-aware and
metric-carrying, and a test asserts that it still refuses a 1D series and a 3D volume. The
precedent is already in the tree and it worked - `training.py` put a second ``(B,C,H,W)`` spine
beside `PhysicalField` rather than turning `PhysicalField` into an untyped container.

**What a sample is.** An array of any rank whose axes are *declared*, never inferred. Rank is
the only thing that generalises: a 1D spectrum, a 3D volume, a station-by-band panel and a
2D field are all samples, and none of them is a grid until somebody says so. Axis roles come
from TG1.1's registry with **both inference stages switched off**. That refusal is the point of
the type. An array whose axes are called ``x`` and ``y`` is not a field: a field licenses
per-metre gradients, radial binning and area weighting, and a domain that has no metre must not
acquire one by spelling.

**How a sample reaches the accepted falsification layer.** Not through `PhysicalField`. A
sample is reduced along its non-channel axes to one scalar per channel per frame, giving a
`ChannelSeries` - which is the interface `cross_scale_dependency` was shown in TG0.1 to
actually consume. That path is domain-general and needs no transform, no grid and no physics.
The reductions are a registry (standard E1), each declaring the gate measure it emits.

**The one bridge, and it points the other way.** `to_physical_field` moves a sample *into* the
analysis spine when 2D diagnostics are genuinely wanted, exactly as `training.py` moves a
batch. It refuses anything that is not a declared two-axis spatial pair, and the refusal names
what would have to be true. There is no route in the other direction: nothing in the analysis
spine accepts a sample, so a rank-3 array cannot arrive inside a wavelet decomposition by any
path, deliberate or accidental.

**Deliberate limits, recorded rather than hidden.**

*   **A sample carries no metric of its own.** A geometry is supplied at the bridge, by a
    caller who is asserting one, and `GridSpec.pixel` remains the honest default. Giving
    `StructuredSample` a `GridSpec` field would make every sample a grid-in-waiting and put
    the E12 line back where TG1.4 was asked to move it from.
*   **`support_parent_px` has no default here either.** Same discipline as `ChannelSeries` and
    `advection_speed_m_s`: a fabricated footprint would set every lag floor in every result
    from a number the reader never chose. A domain whose reduction has no footprint declares
    none and, under rule R21, gets a refusal rather than a floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.axes import AxisResolution, resolve_axis_roles
from src.core.channel_series import ChannelSeries, GATE_MEASURES, MeasureSpec
from src.core.domain import AxisSpec
from src.core.errors import (
    InvalidParameterError,
    MissingParameterError,
    ShapeMismatchError,
)
from src.core.registry import Registry


class SampleError(InvalidParameterError):
    """A structured sample violates a declared invariant."""


# --------------------------------------------------------------------------- the sample


@dataclass(frozen=True)
class StructuredSample:
    """One observation of arbitrary rank, whose axes are declared rather than inferred.

    Immutable, because a sample travels into provenance and into a reduction that a receipt
    will cite. `values` is stored as a read-only `numpy` view for the same reason: a sample
    that could be edited in place after its checks had passed would be a record of nothing.
    """

    values: np.ndarray
    axes: Sequence[AxisSpec]
    coords: Mapping[str, np.ndarray] = dc_field(default_factory=dict)
    units: Optional[str] = None
    split: Optional[str] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    # ------------------------------------------------------------------- construction

    def __post_init__(self) -> None:
        array = np.asarray(self.values)
        if array.dtype == object or not (np.issubdtype(array.dtype, np.number)
                                         or np.issubdtype(array.dtype, np.bool_)):
            raise SampleError(
                "values", array.dtype,
                "a numeric array. A sample of objects cannot be reduced to a measure, and "
                "coercing one here would decide silently what its entries meant")
        if array.ndim == 0:
            raise SampleError(
                "values", array.shape,
                "an array of rank at least 1. A scalar has no axes to declare, and a domain "
                "whose observation is a single number wants a `ChannelSeries` directly")
        if array.size == 0:
            raise SampleError(
                "values", array.shape,
                "a non-empty array. An empty sample reduces to a measure with no support, "
                "and a family of those would look like a clean negative result")

        axes = tuple(self.axes)
        if len(axes) != array.ndim:
            raise ShapeMismatchError(
                "axes", (len(axes),), "one declared axis per array dimension",
                (array.ndim,),
                fix="Declare every axis, in array order. An undeclared trailing axis is the "
                    "one a positional convention would have guessed, which is exactly what "
                    "this type exists to refuse.")
        for spec in axes:
            if not isinstance(spec, AxisSpec):
                raise SampleError(
                    "axes", type(spec).__name__,
                    "an `AxisSpec` per axis, carrying name, role and units. A bare string "
                    "would be a name, and a name is not a role (standard E14)")
        names = tuple(spec.name for spec in axes)
        if len(set(names)) != len(names):
            raise SampleError(
                "axes", list(names),
                "distinct axis names. Two axes sharing a name cannot be given separate roles, "
                "and selecting either would silently pick one of them")

        # E14 in full force, and structurally rather than by a check. Both inference stages
        # are off, and an `AxisSpec` cannot exist without a declared role, so there is no
        # constructible sample whose axis roles were guessed - `AxisResolution.basis` is
        # ``declared`` for every axis of every sample, and a caller may rely on that. The
        # alternative, accepting bare names and refusing them afterwards with
        # `require_declared`, would put a runtime check where an unconstructible state does
        # the same work; the refusal a caller actually meets is the `AxisSpec` one above.
        resolution = resolve_axis_roles(
            names, {spec.name: spec for spec in axes},
            allow_name_inference=False, allow_positional_inference=False)

        coords: Dict[str, np.ndarray] = {}
        for name, values in dict(self.coords).items():
            if name not in names:
                raise SampleError(
                    "coords", name,
                    "a coordinate for a declared axis, which are %s. A coordinate for an axis "
                    "the sample does not have describes different data" % (list(names),))
            vector = np.asarray(values)
            axis_index = names.index(name)
            if vector.ndim != 1:
                raise SampleError(
                    "coords[%r]" % name, vector.shape,
                    "a 1-D coordinate vector. A sample declares its axes separately, so a "
                    "multi-dimensional coordinate would be describing a geometry, and a "
                    "geometry is asserted at the bridge rather than carried here")
            if vector.size != array.shape[axis_index]:
                raise ShapeMismatchError(
                    "coords[%r]" % name, (int(vector.size),),
                    "the extent of axis %r" % name, (int(array.shape[axis_index]),),
                    fix="A coordinate that does not span its own axis no longer describes "
                        "the data it is attached to. That is the D56 failure mode.")
            spec = axes[axis_index]
            if spec.ordered and np.issubdtype(vector.dtype, np.number) and vector.size > 1:
                steps = np.diff(vector.astype(np.float64))
                strict = spec.role == "time"
                bad = (bool(np.any(steps <= 0)) if strict
                       else not (bool(np.all(steps >= 0)) or bool(np.all(steps <= 0))))
                if bad:
                    raise SampleError(
                        "coords[%r]" % name, "non-monotonic",
                        ("a strictly increasing clock" if strict else
                         "a monotonic coordinate") +
                        ". Axis %r is declared ordered, and an ordered axis whose coordinate "
                        "wanders is either mislabelled or was assembled out of order; either "
                        "way every lag and every neighbour relation taken along it is wrong"
                        % name)
            vector = vector.copy()
            vector.flags.writeable = False
            coords[name] = vector

        array = array.copy()
        array.flags.writeable = False
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "_resolution", resolution)

    # ------------------------------------------------------------------- interrogation

    @property
    def resolution(self) -> AxisResolution:
        """The declared axis resolution. Every basis is ``declared``, by construction."""
        return getattr(self, "_resolution")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(n) for n in self.values.shape)

    @property
    def rank(self) -> int:
        return int(self.values.ndim)

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.axes)

    def axis(self, name: str) -> AxisSpec:
        for spec in self.axes:
            if spec.name == name:
                return spec
        raise InvalidParameterError(
            "axis", name, "one of the declared axes %s" % (list(self.axis_names),))

    def axis_index(self, name: str) -> int:
        self.axis(name)
        return self.axis_names.index(name)

    def extent(self, name: str) -> int:
        return self.shape[self.axis_index(name)]

    def axes_with_role(self, role: str) -> Tuple[str, ...]:
        """Declared axes carrying `role`, ordered by their position within it."""
        return self.resolution.dims_with_role(role)

    def labels(self, name: str) -> Tuple[Any, ...]:
        """Coordinate labels along an axis, or its integer positions when it has none."""
        vector = self.coords.get(name)
        if vector is None:
            return tuple(range(self.extent(name)))
        return tuple(vector.tolist())

    def structure(self) -> Tuple[Any, ...]:
        """The part of a sample two frames of one record must agree on.

        Names, roles, units and extents - not the values, and not the coordinates along a
        reduced axis. Two frames that disagree here are two different records, and a channel
        assembled across them would change meaning partway down its own clock.
        """
        return tuple((spec.name, spec.role, spec.units, spec.periodic, spec.ordered, size)
                     for spec, size in zip(self.axes, self.shape))

    def describe(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "axes": [spec.describe() for spec in self.axes],
            "coords": sorted(self.coords),
            "units": self.units,
            "split": self.split,
            "roles": {name: self.resolution.roles[name] for name in self.axis_names},
        }

    # ------------------------------------------------------------------- selection

    def select(self, axis: str, index: int) -> "StructuredSample":
        """One slice along `axis`, with that axis dropped. The last axis cannot be dropped."""
        position = self.axis_index(axis)
        size = self.shape[position]
        if not isinstance(index, (int, np.integer)) or not -size <= int(index) < size:
            raise InvalidParameterError(
                "index", index,
                "a position within axis %r, which has extent %d" % (axis, size))
        if self.rank == 1:
            raise SampleError(
                "axis", axis,
                "a sample of rank at least 2 to select from. Dropping the only axis would "
                "leave a scalar, which is not a sample")
        keep = [name for name in self.axis_names if name != axis]
        selected = dict(self.provenance.get("selected", {}))
        selected[axis] = self.labels(axis)[int(index)]
        return StructuredSample(
            values=np.take(np.asarray(self.values), int(index), axis=position),
            axes=[self.axis(name) for name in keep],
            coords={name: self.coords[name] for name in keep if name in self.coords},
            units=self.units, split=self.split,
            provenance=dict(self.provenance, selected=selected))

    # ------------------------------------------------------------------- the one bridge

    def to_physical_field(self, *, grid: Any = None, axis_roles: Any = None) -> Any:
        """Move a 2D spatial sample into the analysis spine, or refuse and say why.

        The counterpart of `training.py`'s batch boundary, in the other direction. This is the
        only route from a sample into `PhysicalField`, and it is deliberately narrow: exactly
        two axes, both declared ``space``, in ordinal order. Everything else - a spectrum, a
        volume, a station panel, a 2D array whose axes are time and category - is refused with
        the condition it failed, because the alternative is a wavelet decomposition of an array
        whose second axis is a list of instruments.

        `grid` is passed through untouched. A sample carries no metric, so asserting one is the
        caller's act and is recorded as theirs.
        """
        space = self.axes_with_role("space")
        if self.rank != 2 or len(space) != 2:
            raise SampleError(
                "sample", "rank %d with %d declared spatial axes" % (self.rank, len(space)),
                "exactly two axes, both declared with role 'space', to enter the analysis "
                "spine. `PhysicalField` is strictly 2D and metric-carrying by design "
                "(standard E12), and it is not being relaxed for this sample. Reduce the "
                "sample to a `ChannelSeries` instead - that is the domain-general path into "
                "the accepted falsification layer, and it needs no grid at all",
                axes=[spec.describe() for spec in self.axes])

        row, column = space
        if (self.axis_index(row), self.axis_index(column)) != (0, 1):
            # Refused, never transposed. `PhysicalField` reads array dimension 0 as the row,
            # so a declaration whose spatial ordinals run against array order describes the
            # transpose of what is in memory. Transposing here would fix the array and leave
            # every orientation and anisotropy statistic computed from it silently wrong -
            # the failure TG1.1's `NameHint.ordinal` exists to prevent, one layer up.
            raise SampleError(
                "axes", [row, column],
                "spatial ordinals in array order: axis %r is declared the row and %r the "
                "column, but they sit at positions %d and %d. Reorder the array, or the "
                "declaration, so the two agree; this is not transposed for you because a "
                "transposed field still looks like a field"
                % (row, column, self.axis_index(row), self.axis_index(column)))

        # Deferred, and deliberately so: the sample spine does not depend on the analysis
        # spine. `src.core` importing `physical_core` at module scope would make every
        # non-gridded domain pull in torch to declare an axis.
        import torch
        from src.physical_core.field import PhysicalField

        coords = {name: torch.as_tensor(np.array(vector, dtype=np.float64, copy=True))
                  for name, vector in self.coords.items()}
        return PhysicalField(
            torch.as_tensor(np.array(self.values, copy=True)),
            coords=coords or None,
            metadata=dict(self.provenance),
            split=self.split,
            grid=grid,
            units=self.units,
            axis_roles=axis_roles if axis_roles is not None
            else {row: "space", column: "space"})


# --------------------------------------------------------------------- reductions


class SampleReduction:
    """One scalar per channel per frame, from whatever the sample's other axes are.

    A reduction is what stands where a wavelet measure stands in the atmospheric line, so it
    declares the gate measure it emits and inherits that measure's rule-R3 admissibility.
    Subclasses implement `apply` and nothing else.
    """

    name: str = ""

    def apply(self, block: np.ndarray) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, block: np.ndarray) -> float:
        array = np.asarray(block, dtype=np.float64).ravel()
        if array.size == 0:
            raise SampleError(
                "reduction %r" % self.name, "an empty block",
                "at least one value per channel per frame")
        return float(self.apply(array))


#: Reductions from a structured sample to one scalar per channel. A domain registers its own
#: - a band integral, a count above a declared instrument threshold, a robust centre - without
#: editing this module, exactly as it registers a lag policy or a geometry.
SAMPLE_REDUCTIONS: Registry[SampleReduction] = Registry("sample reduction")


class _Mean(SampleReduction):
    name = "mean"

    def apply(self, block: np.ndarray) -> float:
        return float(np.mean(block))


class _Variance(SampleReduction):
    name = "variance"

    def apply(self, block: np.ndarray) -> float:
        return float(np.var(block))


class _EnergyDensity(SampleReduction):
    name = "energy_density"

    def apply(self, block: np.ndarray) -> float:
        return float(np.mean(block * block))


SAMPLE_REDUCTIONS.add(
    "mean", _Mean(),
    capabilities={"gate_measure": "mean", "threshold_free": True},
    description="Arithmetic mean over the reduced axes.")
SAMPLE_REDUCTIONS.add(
    "variance", _Variance(),
    capabilities={"gate_measure": "variance", "threshold_free": True},
    description="Population variance over the reduced axes.")
SAMPLE_REDUCTIONS.add(
    "energy_density", _EnergyDensity(),
    capabilities={"gate_measure": "energy_density", "threshold_free": True},
    description="Mean of squares - energy per available cell, the same definition the "
                "wavelet measure of that name carries.")

# The two names a sample reduction introduces that the wavelet line never needed. Registered
# here rather than in `channel_series`, because that is where they come from; `energy_density`
# is already registered and is deliberately not re-registered under a second definition.
GATE_MEASURES.add(
    "mean", MeasureSpec(True, "the first moment of the reduced block; no threshold and no "
                              "normalisation enter it"),
    description="mean of a structured sample over its reduced axes")
GATE_MEASURES.add(
    "variance", MeasureSpec(True, "the second central moment of the reduced block; it moves "
                                  "with the amplitude scale but not with any threshold"),
    description="variance of a structured sample over its reduced axes")


def reduction_for(name: str) -> SampleReduction:
    """The registered reduction, or an `UnknownNameError` naming the alternatives."""
    return SAMPLE_REDUCTIONS.get(name)


def gate_measure_of(name: str) -> str:
    """The gate measure a reduction emits, checked against the measure registry.

    Not decoration. The measure name is what a `GateProtocol` freezes and what
    `require_gate_measure` adjudicates rule R3 on, so a reduction that emitted a name no
    measure registry knows would produce a series that could be analysed and never gated -
    a result with no admissibility statement attached to it. The check is for *existence*,
    not admissibility: a threshold-bearing measure is still allowed to be computed and
    reported, and is refused only where R3 refuses it, as a gate primary.
    """
    entry = SAMPLE_REDUCTIONS.entry(name)
    measure = entry.capabilities.get("gate_measure", name)
    GATE_MEASURES.entry(measure)
    return str(measure)


def reduction_names() -> Tuple[str, ...]:
    return tuple(SAMPLE_REDUCTIONS.names())


# --------------------------------------------------- samples to the falsification layer


def channel_series_from_samples(
    samples: Sequence[StructuredSample],
    *,
    times_seconds: Sequence[float],
    channel_axis: str,
    reductions: Sequence[str] = ("mean",),
    support_parent_px: Optional[Mapping[Any, float]] = None,
    usable: Optional[Mapping[Any, bool]] = None,
    unusable_reason: Optional[Mapping[Any, str]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ChannelSeries:
    """Reduce a record of samples to the ``(time, channel)`` matrices inference consumes.

    One frame per entry in `samples`, one channel per position along `channel_axis`, and one
    named measure per entry in `reductions`. Everything else about the sample - its rank, its
    other axes, whether it has a metric - is reduced away here and never reaches the analysis
    layer, which is the whole reason this path exists.

    The clock is supplied rather than read off a time axis: a sample is one *frame*, and a
    frame's own axes describe its structure, not its position in the record.
    """
    frames = list(samples)
    if len(frames) < 2:
        raise InvalidParameterError(
            "samples", len(frames),
            "at least two frames. A lag needs a clock with more than one reading on it")
    for position, sample in enumerate(frames):
        if not isinstance(sample, StructuredSample):
            raise InvalidParameterError(
                "samples[%d]" % position, type(sample).__name__,
                "a `StructuredSample`. A bare array has no declared axes, so which of its "
                "dimensions was the channel would be decided by position")

    reference = frames[0]
    reference.axis(channel_axis)
    if reference.resolution.roles[channel_axis] == "time":
        raise InvalidParameterError(
            "channel_axis", channel_axis,
            "an axis that is not the clock. The record's clock is `times_seconds`, and "
            "splitting a frame's own time axis into channels would put the same quantity on "
            "both sides of a lag")
    if reference.rank < 2:
        raise InvalidParameterError(
            "samples", "rank 1",
            "samples of rank at least 2. A rank-1 sample has nothing left to reduce once the "
            "channel axis is taken, so its channels are already scalars and belong in a "
            "`ChannelSeries` directly")

    structure = reference.structure()
    for position, sample in enumerate(frames[1:], start=1):
        if sample.structure() != structure:
            raise ShapeMismatchError(
                "samples[%d]" % position, sample.structure(),
                "the structure of samples[0]", structure,
                fix="Every frame of one record must declare the same axes with the same "
                    "roles, units and extents. A channel whose meaning changes partway down "
                    "its own clock is not one series.")

    labels = reference.labels(channel_axis)
    for position, sample in enumerate(frames[1:], start=1):
        if sample.labels(channel_axis) != labels:
            raise InvalidParameterError(
                "samples[%d]" % position, list(sample.labels(channel_axis)),
                "the channel labels of samples[0], which are %s. Reordered channels would be "
                "compared against each other's history" % (list(labels),))

    names = list(reductions)
    if not names:
        raise MissingParameterError(
            "reductions", "at least one registered reduction, from %s"
                          % (list(reduction_names()),))
    if len(set(names)) != len(names):
        raise InvalidParameterError(
            "reductions", names,
            "distinct reduction names; a repeated measure would overwrite itself")
    chosen = [reduction_for(name) for name in names]

    # A measure is keyed by the gate measure the reduction declares, not by the reduction's
    # own name: the name is how a domain refers to its arithmetic, the measure is what a
    # protocol freezes and what rule R3 is adjudicated on.
    measures = [gate_measure_of(name) for name in names]
    if len(set(measures)) != len(measures):
        raise InvalidParameterError(
            "reductions", names,
            "reductions emitting distinct gate measures; %s collide, and the second would "
            "overwrite the first in a matrix a receipt cites by measure name"
            % (sorted({m for m in measures if measures.count(m) > 1}),))

    axis_position = reference.axis_index(channel_axis)
    matrices: Dict[str, np.ndarray] = {
        measure: np.empty((len(frames), len(labels)), dtype=np.float64)
        for measure in measures}
    for row, sample in enumerate(frames):
        moved = np.moveaxis(np.asarray(sample.values, dtype=np.float64), axis_position, 0)
        for column in range(len(labels)):
            block = moved[column]
            for measure, reduction in zip(measures, chosen):
                matrices[measure][row, column] = reduction(block)

    supports: Optional[List[Optional[float]]] = None
    if support_parent_px is not None:
        unknown = [key for key in support_parent_px if key not in labels]
        if unknown:
            raise InvalidParameterError(
                "support_parent_px", unknown,
                "footprints for declared channels, which are %s" % (list(labels),))
        supports = [support_parent_px.get(label) for label in labels]

    mask = None if usable is None else [bool(usable.get(label, True)) for label in labels]

    record = {
        "spine": "structured_sample",
        "sample_rank": reference.rank,
        "sample_shape": list(reference.shape),
        "sample_axes": [spec.describe() for spec in reference.axes],
        "channel_axis": channel_axis,
        "channel_axis_role": reference.resolution.roles[channel_axis],
        "reductions": list(names),
        "reduction_measures": {name: measure for name, measure in zip(names, measures)},
        "sample_units": reference.units,
    }
    return ChannelSeries(
        channels=list(labels),
        times_seconds=np.asarray(times_seconds, dtype=np.float64),
        measures=matrices,
        usable=mask,
        support_parent_px=supports,
        unusable_reason=unusable_reason,
        provenance=dict(provenance or {}, **record))


__all__ = [
    "SampleError",
    "StructuredSample",
    "SampleReduction",
    "SAMPLE_REDUCTIONS",
    "reduction_for",
    "reduction_names",
    "gate_measure_of",
    "channel_series_from_samples",
]
