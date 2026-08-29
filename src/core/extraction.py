"""Feature extraction as a registry (roadmap TG2.2, standards E1/E4/E14/E16, rules R3/R8/R13).

TG2.1 settled what a feature *is*. This module settles who is allowed to decide that one is
there, and the answer is: not this module. `EXTRACTORS` is a registry, and
`local_maximum` is the **first** registered extractor rather than the definition of
extraction. A watershed, a persistence filter and a matched filter would each disagree with
it about the same field, and a tree that hard-codes one of them has quietly asserted that the
disagreement does not matter.

**What the framework keeps, and what the registry gets.** An extractor returns
`Candidate`s - positions, magnitudes and scales in the units of the declared axes - and
nothing else. `extract()` calibrates the null, dispatches, and builds the `SpectralFeature`
records itself. The split is deliberate: attaching the representation (R8), attaching the
surrogate significance with its ensemble size (TG2.1's resolution floor) and refusing to
compare across domains (R19) are the parts that must not vary between extractors, so an
extractor is not given the opportunity to forget them.

**The threshold is calibrated, never declared.** Rule R3 exists because "coefficient > 0.6"
is a number somebody chose, and a discovery pipeline built on chosen numbers discovers the
choices. Here the cut comes from the distribution of the **maximum** of a surrogate field
with the same power spectrum and randomised phases: a peak is reported when it exceeds what
the strongest peak of a structureless field with the same spectrum does, at a declared family
-wise level over the whole frame. The p-value that lands on the record is the same
`(1 + k) / (1 + n)` the rest of the tree uses, so `Significance`'s floor applies without
translation - and an `alpha` finer than that floor is refused before the ensemble is built
rather than after it is quoted.

**The suppression radius is measured, not chosen.** The one free number a peak-finder usually
carries is "how far apart two features must be", and it is the number that decides how many
features exist. Here each accepted feature is localised first, and then suppresses its
neighbourhood out to a multiple of *its own measured scale*. Fixing that radius instead
manufactures features: on `planted_configuration` at `scale_factor=2.0` a radius tuned at
`scale_factor=1.0` reports eight features where three were planted, and every one of the five
extra ones is a noise maximum on the shoulder of a real blob, above the calibrated threshold,
and indistinguishable in a receipt from a discovery.

**The declared axes change the arithmetic.** A periodic axis wraps: the neighbourhood
comparison, the localisation window and the reported coordinate all wrap with it, so a
feature straddling the seam is found once at its true position rather than twice at two false
ones. A non-periodic axis instead gets a boundary margin (rule R13), because a feature whose
window runs off the edge of the frame has a truncated integral and therefore a scale that is
wrong in a direction nobody can see. This is the payoff for standard E14: the same field with
`periodic=True` and `periodic=False` is genuinely two different problems, and the extractor is
told which it has rather than guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.feature import (
    FeatureLocation,
    FeatureSet,
    Orientation,
    Quantity,
    Significance,
    SpectralFeature,
)
from src.core.registry import Registry

DEFAULT_ALPHA = 0.05
DEFAULT_SURROGATES = 999
DEFAULT_SURROGATE_METHOD = "phase_randomise"
DEFAULT_SEED = 20260819


# ------------------------------------------------------------------------------ the input


@dataclass(frozen=True)
class ExtractionField:
    """One frame, plus everything the extractor is forbidden to guess.

    The identity fields are not decoration. R19 is enforced on the *record*, and a record can
    only be refused by name if something named it; the only honest place for that something
    is the call that presents the data, because by the time an array reaches a peak-finder
    every trace of what it measured is gone.
    """

    values: np.ndarray
    axes: Sequence[AxisSpec]
    domain: str
    dataset: str
    variable: str
    units: Optional[str]
    time: float
    representation: str
    time_units: Optional[str] = None
    coordinates: Optional[Mapping[str, Sequence[float]]] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        axes = tuple(self.axes)
        for name in ("domain", "dataset", "variable", "representation"):
            if not str(getattr(self, name)).strip():
                raise InvalidParameterError(
                    "ExtractionField.%s" % name, getattr(self, name),
                    "a non-empty %s. Extraction produces records that R19 refuses by name, "
                    "and an unnamed field produces records nothing can refuse" % name)
        if not axes:
            raise InvalidParameterError(
                "ExtractionField.axes", [], "one declared axis per array dimension")
        for axis in axes:
            if not isinstance(axis, AxisSpec):
                raise InvalidParameterError(
                    "ExtractionField.axes", axis, "an AxisSpec (standard E14)")
            if axis.role == "time":
                raise InvalidParameterError(
                    "ExtractionField.axes", axis.name,
                    "an axis that is not the clock. A frame is at one time, carried by "
                    "`time`; a time axis inside it is a second clock")
        if values.ndim != len(axes):
            raise InvalidParameterError(
                "ExtractionField.axes", [a.name for a in axes],
                "one axis per array dimension: the array has %d, %d were declared"
                % (values.ndim, len(axes)))
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise InvalidParameterError(
                "ExtractionField.axes", names, "distinct axis names")
        if not np.all(np.isfinite(values)):
            raise InvalidParameterError(
                "ExtractionField.values", "non-finite entries",
                "a field with no NaN or infinity. A masked frame needs a declared fill "
                "policy, and silently maximising over NaN is not one")
        if self.units is not None and not str(self.units).strip():
            raise InvalidParameterError(
                "ExtractionField.units", self.units,
                "the units of the values, or None for a genuinely dimensionless field")
        if not math.isfinite(float(self.time)):
            raise InvalidParameterError(
                "ExtractionField.time", self.time, "a finite time on the declared clock")

        coords = None
        if self.coordinates is not None:
            coords = {}
            if set(self.coordinates) != set(names):
                raise InvalidParameterError(
                    "ExtractionField.coordinates", sorted(self.coordinates),
                    "one coordinate vector per declared axis, or None for index space: %s"
                    % ", ".join(sorted(names)))
            for axis, size in zip(axes, values.shape):
                vec = np.asarray(list(self.coordinates[axis.name]), dtype=np.float64)
                if vec.size != size:
                    raise InvalidParameterError(
                        "ExtractionField.coordinates[%r]" % axis.name, vec.size,
                        "a vector of length %d matching the array" % size)
                steps = np.diff(vec)
                if vec.size > 1 and not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
                    raise InvalidParameterError(
                        "ExtractionField.coordinates[%r]" % axis.name,
                        "spacing %g..%g" % (float(steps.min()), float(steps.max())),
                        "a uniformly spaced axis. A sub-cell position and a width in cells "
                        "convert to this axis's units through one spacing; on a stretched "
                        "axis there is no single spacing, and the answer depends on where "
                        "the feature is - which is exactly what the conversion would hide")
                if vec.size > 1 and steps[0] == 0.0:
                    raise InvalidParameterError(
                        "ExtractionField.coordinates[%r]" % axis.name, 0.0,
                        "a non-zero axis spacing")
                coords[axis.name] = vec

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.values.shape)

    def axis(self, name: str) -> AxisSpec:
        for a in self.axes:
            if a.name == name:
                return a
        raise InvalidParameterError("ExtractionField.axis", name, "a declared axis name")

    def spacing(self, name: str) -> float:
        """Coordinate units per index step, `1.0` when the field is in index space."""
        if self.coordinates is None:
            return 1.0
        vec = self.coordinates[name]
        return 1.0 if vec.size < 2 else float(vec[1] - vec[0])

    def to_coordinate(self, name: str, index: float) -> float:
        """A fractional index in the axis's own units."""
        if self.coordinates is None:
            return float(index)
        vec = self.coordinates[name]
        return float(vec[0]) + float(index) * self.spacing(name)

    def axis_length(self, name: str) -> float:
        """The period of a periodic axis, in its own units - what a separation needs."""
        size = dict(zip((a.name for a in self.axes), self.values.shape))[name]
        return float(size) * abs(self.spacing(name))

    def periods(self) -> Dict[str, float]:
        """The `periods` mapping `FeatureLocation.separation_to` asks for."""
        return {a.name: self.axis_length(a.name) for a in self.axes if a.periodic}

    def describe(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "dataset": self.dataset,
            "variable": self.variable,
            "units": self.units,
            "time": self.time,
            "time_units": self.time_units,
            "representation": self.representation,
            "shape": list(self.shape),
            "axes": [{"name": a.name, "role": a.role, "units": a.units,
                      "periodic": bool(a.periodic)} for a in self.axes],
            "indexed": self.coordinates is None,
            "spacing": {a.name: self.spacing(a.name) for a in self.axes},
            "provenance": dict(self.provenance),
        }


# ------------------------------------------------------------------------ the calibration


@dataclass(frozen=True)
class NullCalibration:
    """The cut, and the ensemble that produced it.

    The statistic is the **maximum over the frame**, not the value of a single cell. That
    choice is what makes `alpha` mean something a reader can act on: the probability that a
    structureless field with this power spectrum produces *any* reported feature at all. A
    per-cell threshold at the same nominal level would report tens of features on pure noise
    on a 256 x 256 frame and each of them would carry `p = 0.05` truthfully.
    """

    threshold: float
    alpha: float
    method: str
    n_surrogates: int
    seed: int
    null_maxima: Tuple[float, ...]
    statistic: str = "frame_maximum"
    #: The representation the surrogate ensemble was pushed *through* before the maxima were
    #: taken, when the null was propagated rather than built where the numbers are read
    #: (TG2.4). `None` means the ensemble and the observation are the same kind of array, so
    #: the null hypothesis is the plain one. When it is set, the hypothesis is materially
    #: different - it is about the representation of a structureless field, not about a
    #: structureless field - and `describe()` says so rather than leaving a reader to assume.
    propagated_through: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "null_maxima", tuple(float(v) for v in self.null_maxima))

    @property
    def resolution_floor(self) -> float:
        return 1.0 / (1.0 + float(self.n_surrogates))

    def clears(self, observed):
        """`observed > threshold`, strictly - and the strictness is the whole point.

        An observation *equal* to a null maximum ties with it, and a tie is not an
        exceedance: `k = #{null >= observed}` counts that null, so an observation at the
        threshold reports `p` one step *above* alpha. Accepting it with `>=` is an
        off-by-one that admits features at a level the receipt does not claim. Vectorised so
        the extractor can apply it to the whole frame.
        """
        return observed > self.threshold

    def p_value(self, observed: float) -> Significance:
        """`(1 + k) / (1 + n)` against the null maxima - family-wise over the frame."""
        k = int(sum(1 for v in self.null_maxima if v >= float(observed)))
        p = (1.0 + k) / (1.0 + float(self.n_surrogates))
        return Significance(p, basis="surrogate_p_value", n_surrogates=self.n_surrogates)

    def describe(self) -> Dict[str, Any]:
        arr = np.asarray(self.null_maxima, dtype=np.float64)
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "statistic": self.statistic,
            "surrogate_method": self.method,
            "n_surrogates": self.n_surrogates,
            "seed": self.seed,
            "resolution_floor": self.resolution_floor,
            "null_max_min": float(arr.min()) if arr.size else None,
            "null_max_median": float(np.median(arr)) if arr.size else None,
            "null_max_max": float(arr.max()) if arr.size else None,
            "propagated_through": self.propagated_through,
            "null_hypothesis": (
                "no peak in this frame exceeds the strongest peak of %s, at family-wise "
                "level %g over the whole frame"
                % (("%s applied to a field with the same power spectrum and randomised "
                    "phases" % self.propagated_through) if self.propagated_through
                   else "a field with the same power spectrum and randomised phases",
                   self.alpha)),
        }


def calibrate(
    values: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    method: str = DEFAULT_SURROGATE_METHOD,
    n_surrogates: int = DEFAULT_SURROGATES,
    seed: int = DEFAULT_SEED,
) -> NullCalibration:
    """Build the null-maximum ensemble and take the cut from its order statistics.

    No interpolated quantile. The reported p-value is `(1 + k) / (1 + n)`, so the threshold
    that agrees with it is an order statistic of the same ensemble. An observation clears the
    cut when at most `K = floor(alpha * (1 + n)) - 1` null maxima are at least as large as it
    is, which is exactly the condition `observed > ordered[K]` - so the threshold is the
    `(K + 1)`-th largest null maximum and the comparison against it is strict. An
    interpolating quantile would put the cut between two achievable p-values, and the receipt
    would then hold a threshold no reported number could match.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidParameterError(
            "calibrate.alpha", alpha, "a family-wise level strictly between 0 and 1")
    n = int(n_surrogates)
    if n < 1:
        raise InvalidParameterError(
            "calibrate.n_surrogates", n_surrogates, "at least one surrogate")
    floor = 1.0 / (1.0 + n)
    if float(alpha) < floor - 1e-12:
        raise InvalidParameterError(
            "calibrate.alpha", alpha,
            "a level the ensemble can resolve. %d surrogates resolve no p-value smaller "
            "than %.6g, so a cut at %g cannot be reached by any observation: the extractor "
            "would return nothing and the receipt would say the field was empty. Use at "
            "least %d surrogates for this level"
            % (n, floor, float(alpha), int(math.ceil(1.0 / float(alpha))) - 1),
            n_surrogates=n, resolution_floor=floor)

    from src.statistics import surrogates as surrogate_module   # declared dependency

    arr = np.asarray(values, dtype=np.float64)
    ensemble = surrogate_module.generate(arr, method=method, n=n, seed=int(seed))
    null_max = [float(np.max(np.asarray(m))) for m in ensemble["members"]]
    return calibration_from_maxima(
        null_max, alpha=float(alpha), method=str(method), seed=int(seed))


def calibration_from_maxima(
    null_maxima: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    method: str = DEFAULT_SURROGATE_METHOD,
    seed: int = DEFAULT_SEED,
    propagated_through: Optional[str] = None,
) -> NullCalibration:
    """The cut, from an ensemble of frame maxima somebody else built.

    `calibrate` builds its ensemble by surrogating the array it is handed. TG2.4 needs the
    same order statistic over maxima that were produced a different way - surrogates of the
    *field*, each pushed through a representation, with the maximum taken in the coefficient
    plane - and the arithmetic of the cut must not be reimplemented there, because a second
    copy of `floor(alpha * (1 + n)) - 1` is a second chance to get the off-by-one wrong.
    """
    maxima = [float(v) for v in null_maxima]
    n = len(maxima)
    if n < 1:
        raise InvalidParameterError(
            "calibration_from_maxima.null_maxima", 0, "at least one surrogate maximum")
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidParameterError(
            "calibration_from_maxima.alpha", alpha,
            "a family-wise level strictly between 0 and 1")
    floor = 1.0 / (1.0 + n)
    if float(alpha) < floor - 1e-12:
        raise InvalidParameterError(
            "calibration_from_maxima.alpha", alpha,
            "a level the ensemble can resolve. %d surrogates resolve no p-value smaller "
            "than %.6g" % (n, floor), n_surrogates=n, resolution_floor=floor)
    if not all(math.isfinite(v) for v in maxima):
        raise InvalidParameterError(
            "calibration_from_maxima.null_maxima", "non-finite entries",
            "finite maxima; a non-finite null maximum makes the cut unusable rather than "
            "merely large")
    allowed_k = int(math.floor(float(alpha) * (1.0 + n))) - 1
    ordered = np.sort(np.asarray(maxima, dtype=np.float64))[::-1]
    threshold = float(ordered[allowed_k])
    return NullCalibration(
        threshold=threshold, alpha=float(alpha), method=str(method),
        n_surrogates=n, seed=int(seed), null_maxima=tuple(maxima),
        propagated_through=propagated_through)


# ------------------------------------------------------------------------ what an extractor returns


@dataclass(frozen=True)
class Candidate:
    """One structure an extractor believes in, in the units of the declared axes.

    Deliberately not a `SpectralFeature`. An extractor that built its own records would also
    be choosing its own significance basis, its own representation label and its own idea of
    which comparisons are legal, and those are the three things TG2.1 exists to take away
    from it.
    """

    coords: Mapping[str, float]
    magnitude: float
    peak_sample: float
    scale: Optional[float] = None
    scale_units: Optional[str] = None
    orientation: Optional[Orientation] = None
    extent: Optional[float] = None
    extent_units: Optional[str] = None
    uncertainty: Optional[Mapping[str, float]] = None
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.coords:
            raise InvalidParameterError(
                "Candidate.coords", {}, "a coordinate on every declared axis")
        for name, value in self.coords.items():
            if not math.isfinite(float(value)):
                raise InvalidParameterError(
                    "Candidate.coords[%r]" % name, value, "a finite coordinate")
        for name in ("magnitude", "peak_sample"):
            if not math.isfinite(float(getattr(self, name))):
                raise InvalidParameterError(
                    "Candidate.%s" % name, getattr(self, name), "a finite value")
        if self.scale is not None and not (math.isfinite(float(self.scale))
                                           and float(self.scale) > 0.0):
            raise InvalidParameterError(
                "Candidate.scale", self.scale, "a positive finite scale, or None")
        object.__setattr__(self, "coords", dict(self.coords))
        object.__setattr__(self, "notes", dict(self.notes))
        object.__setattr__(
            self, "uncertainty", None if self.uncertainty is None else dict(self.uncertainty))


@dataclass(frozen=True)
class ExtractorReport:
    """Candidates, plus what was looked at and thrown away.

    `rejected` is not diagnostics. "Nothing was found" and "four hundred maxima were found
    and every one of them fell below the calibrated cut" are different statements about a
    field, and TG2.4's audit is a question about the second one.
    """

    candidates: Sequence[Candidate] = ()
    rejected: Mapping[str, int] = dc_field(default_factory=dict)
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        for item in self.candidates:
            if not isinstance(item, Candidate):
                raise InvalidParameterError(
                    "ExtractorReport.candidates", item,
                    "a Candidate. An extractor returns candidates; `extract` builds the "
                    "records, so that no extractor can choose its own significance basis")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejected",
                           {str(k): int(v) for k, v in dict(self.rejected).items()})
        object.__setattr__(self, "notes", dict(self.notes))


Extractor = Callable[..., ExtractorReport]

EXTRACTORS: Registry[Extractor] = Registry("feature extractor")


# ------------------------------------------------------------------------------ the result


@dataclass(frozen=True)
class ExtractionResult:
    """What one extractor found in one frame, including when it found nothing.

    `FeatureSet` refuses to be empty on purpose - an empty set has no domain to report and no
    representation to blame. This is the object its docstring points at: the empty result
    still knows what was searched, with what, at what threshold, and what was rejected on the
    way. A pipeline that returned `[]` for a null field would be indistinguishable from one
    that crashed before looking.
    """

    field: ExtractionField
    extractor: str
    calibration: NullCalibration
    features: Sequence[SpectralFeature] = ()
    rejected: Mapping[str, int] = dc_field(default_factory=dict)
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "rejected",
                           {str(k): int(v) for k, v in dict(self.rejected).items()})
        object.__setattr__(self, "notes", dict(self.notes))

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[SpectralFeature]:
        return iter(self.features)

    def __getitem__(self, index: int) -> SpectralFeature:
        return self.features[index]

    @property
    def is_empty(self) -> bool:
        return not self.features

    @property
    def found(self) -> int:
        return len(self.features)

    def feature_set(self) -> FeatureSet:
        """The homogeneous set TG2.3 consumes - refused when nothing was found."""
        if not self.features:
            raise InvalidParameterError(
                "ExtractionResult.feature_set", 0,
                "at least one feature. Extractor %r found none in %s/%s/%s at threshold "
                "%.6g (alpha %g, %d surrogates); that is a result, and it lives on this "
                "object rather than in an empty set with nothing to report about itself"
                % (self.extractor, self.field.domain, self.field.dataset,
                   self.field.variable, self.calibration.threshold,
                   self.calibration.alpha, self.calibration.n_surrogates),
                rejected=dict(self.rejected))
        return FeatureSet(self.features)

    def describe(self) -> Dict[str, Any]:
        return {
            "extractor": self.extractor,
            "found": self.found,
            "rejected": dict(self.rejected),
            "field": self.field.describe(),
            "calibration": self.calibration.describe(),
            "extractor_capabilities": dict(EXTRACTORS.entry(self.extractor).capabilities),
            "notes": dict(self.notes),
            "features": [f.describe() for f in self.features],
        }


# ------------------------------------------------------------------------------ entry point


def extract(
    field: ExtractionField,
    extractor: str = "local_maximum",
    *,
    alpha: float = DEFAULT_ALPHA,
    surrogate_method: str = DEFAULT_SURROGATE_METHOD,
    n_surrogates: int = DEFAULT_SURROGATES,
    seed: int = DEFAULT_SEED,
    calibration: Optional[NullCalibration] = None,
    **params: Any,
) -> ExtractionResult:
    """Calibrate, dispatch to a registered extractor, and build the records.

    The record-building is here rather than in the extractor because it is the part R19,
    R8 and TG2.1's resolution floor depend on. A registered extractor decides *where the
    features are*; it does not get to decide what a feature is allowed to be compared with.
    """
    if not isinstance(field, ExtractionField):
        raise InvalidParameterError(
            "extract.field", field,
            "an ExtractionField carrying the array and the declaration it needs")
    entry = EXTRACTORS.entry(extractor)     # UnknownNameError, with a did-you-mean

    ndim = entry.capabilities.get("dimensions")
    if ndim is not None and int(ndim) != field.values.ndim:
        raise InvalidParameterError(
            "extract.field", field.values.ndim,
            "a %d-dimensional field: extractor %r declares dimensions=%d"
            % (int(ndim), extractor, int(ndim)))

    if calibration is None:
        calibration = calibrate(field.values, alpha=alpha, method=surrogate_method,
                                n_surrogates=n_surrogates, seed=seed)
    report = entry.value(field, calibration, **params)
    if not isinstance(report, ExtractorReport):
        raise InvalidParameterError(
            "extract.report", type(report).__name__,
            "an ExtractorReport from extractor %r" % extractor)

    features = tuple(
        _to_feature(field, extractor, calibration, c) for c in report.candidates)
    return ExtractionResult(
        field=field, extractor=extractor, calibration=calibration,
        features=features, rejected=report.rejected, notes=report.notes)


def _to_feature(field: ExtractionField, extractor: str, calibration: NullCalibration,
                candidate: Candidate) -> SpectralFeature:
    """One candidate, wearing everything R19 will later refuse it a comparison without."""
    names = [a.name for a in field.axes]
    if set(candidate.coords) != set(names):
        raise InvalidParameterError(
            "Candidate.coords", sorted(candidate.coords),
            "a coordinate on each declared axis (%s); extractor %r returned %s"
            % (", ".join(sorted(names)), extractor, sorted(candidate.coords)))
    location = FeatureLocation(
        coords=candidate.coords, axes=field.axes,
        uncertainty=dict(candidate.uncertainty or {}))
    scale = None
    if candidate.scale is not None:
        scale = Quantity(candidate.scale, candidate.scale_units)
    extent = None
    if candidate.extent is not None:
        extent = Quantity(candidate.extent, candidate.extent_units)
    provenance = {
        "extractor": extractor,
        "peak_sample": candidate.peak_sample,
        "threshold": calibration.threshold,
        "alpha": calibration.alpha,
        "surrogate_method": calibration.method,
        "surrogate_seed": calibration.seed,
        **dict(field.provenance),
        **dict(candidate.notes),
    }
    return SpectralFeature(
        domain=field.domain, dataset=field.dataset, variable=field.variable,
        magnitude=Quantity(candidate.magnitude, field.units),
        location=location, time=field.time, representation=field.representation,
        time_units=field.time_units, spatial_scale=scale, extent=extent,
        orientation=candidate.orientation,
        significance=calibration.p_value(candidate.peak_sample),
        provenance=provenance)


# --------------------------------------------------------------- the first registered extractor


def _neighbour_maximum(values: np.ndarray, periodic: Sequence[bool]) -> np.ndarray:
    """The largest value in each cell's 3x3 neighbourhood, wrapping where declared.

    Written with shifts rather than a filter so that periodicity is per-axis: a field that
    wraps in longitude and not in latitude is the common case, and a single `mode=` argument
    cannot express it.
    """
    padded = values
    for axis, wraps in enumerate(periodic):
        width = [(0, 0)] * values.ndim
        width[axis] = (1, 1)
        padded = np.pad(padded, width, mode="wrap" if wraps else "edge")
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = np.maximum(out, padded[dy:dy + values.shape[0],
                                         dx:dx + values.shape[1]])
    return out


def _window(values: np.ndarray, centre: Tuple[int, int], radius: int,
            periodic: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A square window about `centre`, wrapping on periodic axes and clipped on the rest.

    Returns the window and the *unwrapped* index vectors, so a centroid computed from them
    lands on a continuous coordinate that may be negative or beyond the axis; `extract`
    wraps it back at the end. Averaging wrapped indices instead is how a feature on the seam
    acquires a position in the middle of the frame.
    """
    rows_raw = np.arange(centre[0] - radius, centre[0] + radius + 1)
    cols_raw = np.arange(centre[1] - radius, centre[1] + radius + 1)
    n0, n1 = values.shape
    if periodic[0]:
        rows = rows_raw % n0
    else:
        keep = (rows_raw >= 0) & (rows_raw < n0)
        rows_raw, rows = rows_raw[keep], rows_raw[keep]
    if periodic[1]:
        cols = cols_raw % n1
    else:
        keep = (cols_raw >= 0) & (cols_raw < n1)
        cols_raw, cols = cols_raw[keep], cols_raw[keep]
    return values[np.ix_(rows, cols)], rows_raw, cols_raw


def _square_capture(radius: float, sigma: float) -> float:
    """Fraction of an isotropic Gaussian's integral inside a square of half-width `radius`.

    `erf(r / (sigma * sqrt2)) ** 2`, exactly - the two axes separate. Without it the
    integral estimator reads a truncated integral as a narrower feature, and the error grows
    with the feature, which is the worst shape an error can have for a *ratio*.
    """
    if sigma <= 0.0:
        return 1.0
    return math.erf(radius / (sigma * math.sqrt(2.0))) ** 2


def _disc_amplitude(values: np.ndarray, centre: Tuple[float, float], sigma: float,
                    baseline: float, periodic: Sequence[bool], frac: float = 0.5
                    ) -> Optional[float]:
    """Amplitude from the mean over a small disc, divided by the Gaussian's disc-mean.

    **The sampled peak is not the amplitude.** It is the largest of many noisy samples, so it
    is biased upward, and the bias grows as the feature broadens and more cells compete to be
    the maximum. On `planted_configuration` the brightest sample overstates a unit-amplitude
    feature by 3% at sigma = 3 cells and by 12% at sigma = 18; carried into the integral
    estimator that becomes a scale biased *low* by up to 9%, in a direction that survives
    every ratio because it is a function of the feature's own size. Averaging over a disc of
    `frac * sigma` divides the noise by the root of the cell count and the analytic disc-mean
    `2 (1 - exp(-f^2/2)) / f^2` removes the curvature, leaving a per-feature error under 5%
    across a six-fold range of scales.
    """
    radius = max(1.0, float(frac) * float(sigma))
    n0, n1 = values.shape
    lo0, hi0 = int(math.floor(centre[0] - radius)), int(math.ceil(centre[0] + radius))
    lo1, hi1 = int(math.floor(centre[1] - radius)), int(math.ceil(centre[1] + radius))
    rows_raw = np.arange(lo0, hi0 + 1)
    cols_raw = np.arange(lo1, hi1 + 1)
    if periodic[0]:
        rows = rows_raw % n0
    else:
        keep = (rows_raw >= 0) & (rows_raw < n0)
        rows_raw, rows = rows_raw[keep], rows_raw[keep]
    if periodic[1]:
        cols = cols_raw % n1
    else:
        keep = (cols_raw >= 0) & (cols_raw < n1)
        cols_raw, cols = cols_raw[keep], cols_raw[keep]
    if rows_raw.size == 0 or cols_raw.size == 0:
        return None
    d0 = (rows_raw - centre[0])[:, None]
    d1 = (cols_raw - centre[1])[None, :]
    mask = (d0 ** 2 + d1 ** 2) <= radius ** 2
    if not mask.any():
        return None
    excess = values[np.ix_(rows, cols)] - baseline
    mean_excess = float(excess[mask].mean())
    f = radius / float(sigma) if sigma > 0 else 1.0
    factor = 2.0 * (1.0 - math.exp(-0.5 * f * f)) / (f * f)
    if factor <= 0.0:
        return None
    return mean_excess / factor


@EXTRACTORS.register(
    "local_maximum",
    description=("Local maxima above a surrogate-calibrated frame-maximum threshold, "
                 "localised sub-cell by a windowed centroid and suppressed at a multiple "
                 "of each feature's own measured scale."),
    params={"window_scales": 2.0, "suppression_scales": 2.0, "boundary_margin": 1,
            "max_features": 256, "refinements": 8},
    capabilities={"dimensions": 2, "needs_calibration": True, "sub_cell": True,
                  "reports_scale": True, "reports_orientation": False,
                  "self_scaling_suppression": True, "honours_periodic_axes": True,
                  "shape_model": "isotropic_gaussian_on_a_flat_baseline"},
    tags=["field", "peaks"],
)
def local_maximum_extractor(
    field: ExtractionField,
    calibration: NullCalibration,
    *,
    window_scales: float = 2.0,
    suppression_scales: float = 2.0,
    boundary_margin: int = 1,
    max_features: int = 256,
    refinements: int = 8,
) -> ExtractorReport:
    """The first registered extractor, and only the first.

    It assumes an isotropic peak on a flat baseline. That assumption is declared in the
    registry entry rather than buried here, because it is false for a front, false for a
    filament and false for anything elongated - and the honest response to that is a second
    registered extractor, not a special case inside this one (standard E12).
    """
    values = field.values
    if values.ndim != 2:
        raise InvalidParameterError(
            "local_maximum.field", values.ndim,
            "a 2-dimensional field; this extractor declares dimensions=2")
    if float(window_scales) <= 0 or float(suppression_scales) <= 0:
        raise InvalidParameterError(
            "local_maximum.window_scales", (window_scales, suppression_scales),
            "positive multiples of the measured scale")
    periodic = tuple(bool(a.periodic) for a in field.axes)
    margin = max(0, int(boundary_margin))
    baseline = float(np.median(values))

    above = calibration.clears(values)
    is_max = values >= _neighbour_maximum(values, periodic)
    interior = np.ones(values.shape, dtype=bool)
    for axis, wraps in enumerate(periodic):
        if wraps or margin == 0:
            continue
        # Rule R13, first pass: a maximum on the very edge has no neighbourhood to be
        # maximal over. The pass that matters happens after localisation, below, because
        # the width of the valid interior is a property of the feature and not of the frame.
        sl: List[Any] = [slice(None)] * values.ndim
        sl[axis] = slice(0, margin)
        interior[tuple(sl)] = False
        sl[axis] = slice(values.shape[axis] - margin, values.shape[axis])
        interior[tuple(sl)] = False

    rejected = {
        "below_threshold": int(np.count_nonzero(is_max & ~above)),
        "outside_valid_interior": int(np.count_nonzero(is_max & above & ~interior)),
        "suppressed_by_a_stronger_feature": 0,
        "unmeasurable_scale": 0,
        "over_max_features": 0,
    }

    rows, cols = np.nonzero(is_max & above & interior)
    order = np.argsort(-values[rows, cols], kind="stable")
    accepted: List[Candidate] = []
    centres: List[Tuple[float, float, float]] = []
    name0, name1 = field.axes[0].name, field.axes[1].name
    n0, n1 = values.shape

    for idx in order:
        i, j = int(rows[idx]), int(cols[idx])
        peak = float(values[i, j])
        if _is_suppressed((i, j), centres, periodic, (n0, n1), float(suppression_scales)):
            rejected["suppressed_by_a_stronger_feature"] += 1
            continue
        if len(accepted) >= int(max_features):
            rejected["over_max_features"] += 1
            continue
        measured = _localise(values, (i, j), peak, baseline, periodic,
                             float(window_scales), int(refinements))
        if measured is None:
            rejected["unmeasurable_scale"] += 1
            continue
        cy, cx, sigma, amplitude, radius = measured
        # Rule R13, the pass that matters. The valid interior is not a fixed margin: it is
        # whatever the feature's own window needs. A peak two cells from a non-periodic edge
        # has half its integral outside the frame, and the truncation correction dutifully
        # applies - for the wrong reason - and returns a scale a quarter too small with
        # nothing in the record to say so. Measured on a wrapped test field declared
        # non-periodic, that is a position 3 cells wrong and a scale 24% low. Refused and
        # counted instead: a missing feature is a fact a receipt can carry, and a confidently
        # mismeasured one is not.
        if _runs_off_the_frame((cy, cx), float(window_scales) * sigma, periodic, (n0, n1)):
            rejected["outside_valid_interior"] += 1
            continue
        centres.append((cy, cx, sigma))
        coord0 = field.to_coordinate(name0, cy % n0 if periodic[0] else cy)
        coord1 = field.to_coordinate(name1, cx % n1 if periodic[1] else cx)
        scale_units = field.axes[0].units
        if field.axes[1].units != scale_units:
            # A single isotropic width cannot be quoted when the two axes are in different
            # units; the number would be a length in neither of them.
            scale_units = None
            sigma_out: Optional[float] = None
        else:
            sigma_out = sigma * abs(field.spacing(name0))
        accepted.append(Candidate(
            coords={name0: coord0, name1: coord1},
            magnitude=amplitude + baseline,
            peak_sample=peak,
            scale=sigma_out,
            scale_units=scale_units,
            notes={
                "baseline": baseline,
                "window_radius_cells": radius,
                "scale_cells": sigma,
                "index_position": [cy, cx],
                "localisation": "windowed_centroid",
                "amplitude_estimator": "disc_mean_debiased",
                "scale_estimator": "integral_over_amplitude_square_truncation_corrected",
            }))

    return ExtractorReport(
        candidates=tuple(accepted), rejected=rejected,
        notes={
            "baseline": baseline,
            "baseline_estimator": "frame_median",
            "raw_local_maxima": int(np.count_nonzero(is_max)),
            "boundary_margin": margin,
            "periodic_axes": [a.name for a in field.axes if a.periodic],
        })


def _runs_off_the_frame(centre: Tuple[float, float], radius: float,
                        periodic: Sequence[bool], shape: Tuple[int, int]) -> bool:
    """True when the feature's own measurement window leaves a non-periodic axis."""
    for axis, wraps in enumerate(periodic):
        if wraps:
            continue
        if centre[axis] - radius < 0.0 or centre[axis] + radius > float(shape[axis] - 1):
            return True
    return False


def _is_suppressed(cell: Tuple[int, int], centres: Sequence[Tuple[float, float, float]],
                   periodic: Sequence[bool], shape: Tuple[int, int],
                   scales: float) -> bool:
    """True when a stronger feature's own measured scale already covers this cell."""
    for cy, cx, sigma in centres:
        d0 = abs(cell[0] - cy)
        d1 = abs(cell[1] - cx)
        if periodic[0]:
            d0 = min(d0, shape[0] - d0)
        if periodic[1]:
            d1 = min(d1, shape[1] - d1)
        if math.hypot(d0, d1) <= scales * sigma:
            return True
    return False


def _localise(values: np.ndarray, cell: Tuple[int, int], peak: float, baseline: float,
              periodic: Sequence[bool], window_scales: float, refinements: int
              ) -> Optional[Tuple[float, float, float, float, int]]:
    """Sub-cell position, scale and amplitude, by a window that sizes itself.

    **The textbook three-point parabola was measured and rejected.** Fitting a parabola to
    the peak cell and its two neighbours recovers a sub-cell offset exactly for a noiseless
    Gaussian, and the offset is `(y- - y+) / (2 (y- - 2 y0 + y+))` - a ratio whose
    denominator is the second difference, which for a feature of width sigma is of order
    `A / sigma^2`. On `planted_configuration` that is 0.028 against a noise amplitude of
    0.05: the denominator is noise. Measured against the benchmark's recorded positions the
    parabola errs by up to 0.68 cells where the windowed centroid errs by 0.27. The parabola
    is right only when the feature is a cell or two wide, and nothing in this tree guarantees
    that.
    """
    radius = 4
    result = None
    for _ in range(max(1, refinements)):
        window, rows_raw, cols_raw = _window(values, cell, radius, periodic)
        if rows_raw.size < 3 or cols_raw.size < 3:
            return None
        excess = np.clip(window - baseline, 0.0, None)
        total = float(excess.sum())
        if total <= 0.0:
            return None
        cy = float((excess.sum(axis=1) * rows_raw).sum() / total)
        cx = float((excess.sum(axis=0) * cols_raw).sum() / total)
        amplitude = peak - baseline if result is None else result[3]
        if amplitude <= 0.0:
            return None
        sigma = math.sqrt(total / (2.0 * math.pi * amplitude))
        for _ in range(8):
            capture = _square_capture(float(radius), sigma)
            if capture <= 0.0:
                return None
            sigma = math.sqrt(total / (2.0 * math.pi * amplitude * capture))
        if not math.isfinite(sigma) or sigma <= 0.0:
            return None
        refined = _disc_amplitude(values, (cy, cx), sigma, baseline, periodic)
        if refined is not None and refined > 0.0:
            amplitude = refined
        result = (cy, cx, sigma, amplitude, radius)
        new_radius = max(2, int(round(window_scales * sigma)))
        if new_radius == radius:
            break
        radius = new_radius
    return result
