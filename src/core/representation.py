"""Representation-induced feature audit (roadmap TG2.4, standards E1/E4/E12/E14, rules R3/R8/R13).

TG2.2 settled who is allowed to decide a feature is there, and measured the false-positive
floor on the **raw** field: structureless data in, nothing out. That is half the floor. Rule
R8 says the other half out loud - *a representation can manufacture a motif* - and a
peak-finder that is honest on an array is not thereby honest on a wavelet band of that array,
because the band is not the array. Every transform in `TRANSFORMS` is a lens, every lens has
its own artefacts, and a discovery pipeline reads the lens, not the field.

**What this module is.** A representation is turned into a set of **planes** - named 2D arrays
with declared axes - and the registered extractor is run on each plane of a field that has no
structure. The correct answer is that it finds nothing, in every plane of every registered
representation. Anything else is a manufactured motif, and it is manufactured in a place a
receipt would attribute to the data.

**The null must be propagated, not rebuilt.** This is the one design decision that decides
whether the audit means anything, and it is a registry with a declared default rather than a
constant, because it is exactly the kind of choice that should be visible and measurable.

*   ``propagate_through_representation`` (the default) surrogates the **field**, pushes every
    surrogate through the same representation with the same configuration, and takes the
    maximum in the coefficient plane. The null then carries the representation, so an artefact
    that appears in every realisation appears in the null as well and cannot be reported.
*   ``randomise_in_representation`` surrogates the **coefficient plane**. It is the obvious
    thing to do, it is what "just calibrate where you measure" means, and it is wrong: the
    plane's own spectrum is a product of the representation, so randomising phases *inside* it
    destroys the very artefact the null was supposed to account for. It is registered so the
    difference can be measured rather than asserted.

**The audit pays for its own family.** Forty-five testable planes read as a single question -
*did any registered lens manufacture a feature?* - is a family of forty-five tests, and rule
R18 arrives two phases early. Measured on the ensemble itself, running them uncorrected at a
nominal 0.05 each rejects on **86%** of structureless fields. `FAMILY_CORRECTIONS` is the third
registry: `max_statistic` reads the family-wise rate off the same surrogate ensemble the cuts
come from, leave-one-out, so dependence between bands of one decomposition is *measured*;
`bonferroni` prices them as independent; `none` is registered to be measured against. The
corrected level is a shared per-plane rank, which keeps every threshold an order statistic of
its own null - `calibrate`'s no-interpolation rule applied to a family - and that has a price:
the strictest cut `n` surrogates can express still rejects about `P / n` of the time over `P`
planes, so a family-wise 0.05 over forty-five planes needs about nine hundred surrogates.
`FamilyTooLargeError` names the family, the achievable rate and the required ensemble. An
unaffordable family is refused before it is run, never quietly reported at a level it cannot
reach.

**A plane with nothing left to search is refused and struck from the family.** A dual-tree
level-3 subband of a 64-cell frame is 8 x 8 samples with a six-sample contaminated margin on
every side, which is no valid interior at all: the extractor would report nothing whatever the
data were, and counting that as a floor is a pass earned by not looking. A test with no
possible outcome must also not make the other tests stricter, so it is not charged to the
family either. The report carries the number of cells actually searched.

**A null the surrogate method cannot violate is refused, not passed.** `phase_randomise`
preserves the amplitude spectrum to machine precision. The FFT magnitude plane *is* the
amplitude spectrum. So every surrogate of the field has a bit-identical magnitude plane, the
null maxima have zero spread, the threshold equals the observation, and a strict `>` reports
nothing - for ever, on any data, structured or not. That is not a floor; it is a test that
cannot fail. Every plane's null is checked for that degeneracy and a vacuous one is recorded
as `vacuous`, never as a pass.

**Not every representation has an extractable plane.** The axes of an FFT magnitude plane are
wavenumbers. Declaring them `space` so a spatial peak-finder will accept them is precisely the
error TG1.1 exists to prevent - an axis is not spatial until somebody says it is, and saying it
here would license a position in cells, a width in cells and a separation in cells, none of
which that plane has. Those planes carry `axes=None` and a stated reason, the audit reports
them as **unauditable rather than clean**, and the gap is a recorded limit of the tree rather
than a silence in a summary.

**Coverage is checked against the transform registry.** A transform that is registered but has
no registered representation builder is listed by name in the report. "All representations were
audited" must never be able to mean "the two we wrote builders for were audited".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.extraction import (
    DEFAULT_ALPHA,
    DEFAULT_SEED,
    DEFAULT_SURROGATE_METHOD,
    ExtractionField,
    ExtractionResult,
    NullCalibration,
    calibration_from_maxima,
    extract,
)
from src.core.registry import Registry

#: Enough to resolve alpha = 0.05 on the `(1 + k) / (1 + n)` convention with room to spare,
#: and few enough that a whole-registry audit runs inside a test suite. The audit costs one
#: transform per surrogate per representation, not one per plane: every plane of a
#: representation is produced by the same decomposition and shares the ensemble.
DEFAULT_AUDIT_SURROGATES = 99

DEFAULT_NULL_STRATEGY = "propagate_through_representation"
DEFAULT_FAMILY_CORRECTION = "max_statistic"

#: A null whose maxima span less than this fraction of their own magnitude cannot be exceeded
#: by anything, so the test it defines has no power. Set far below any spread a real ensemble
#: produces (a wavelet band's null maxima spread by tens of percent) and far above float
#: round-trip noise, so the check separates "identical by construction" from "tight".
VACUITY_TOLERANCE = 1.0e-9


# ------------------------------------------------------------------------------ the plane


@dataclass(frozen=True)
class RepresentationPlane:
    """One named 2D array a representation produces, and what it is allowed to be read as.

    `axes` is the declaration, not a convenience. A plane with axes is a spatial map of the
    same domain the field came from, sampled possibly more coarsely; a plane without them is
    an array whose indices mean something else, and the only honest thing to do with it here
    is to measure its null and refuse to extract from it.
    """

    name: str
    values: np.ndarray
    axes: Optional[Tuple[AxisSpec, ...]] = None
    #: Parent cells per sample, per axis. `1.0` for an undecimated band; `2 ** level` for a
    #: decimated one. It is what turns an index in this plane back into a position in the
    #: field, and it is declared by the transform rather than inferred from the shape ratio,
    #: which padding makes a non-integer.
    parent_cells_per_sample: Tuple[float, ...] = (1.0, 1.0)
    #: The filter's contaminated margin in samples of *this* plane (rule R13). Passed to the
    #: extractor as its boundary margin, so an artefact of the convolution's edge handling
    #: cannot be reported as a feature of the field.
    contaminated_halfwidth: int = 0
    #: Why this plane cannot be extracted from. Required when `axes` is None, forbidden
    #: otherwise: a plane either carries a declaration or carries the reason it has none.
    not_extractable: Optional[str] = None
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise InvalidParameterError(
                "RepresentationPlane.values", values.ndim,
                "a 2-dimensional plane; a representation that produces a rank-3 stack "
                "declares one plane per slice, so each one can be named and refused "
                "separately")
        if not np.all(np.isfinite(values)):
            raise InvalidParameterError(
                "RepresentationPlane.values", "non-finite entries",
                "a finite plane. A transform that returns NaN has failed, and maximising "
                "over it would report the failure as a feature")
        if not str(self.name).strip():
            raise InvalidParameterError(
                "RepresentationPlane.name", self.name,
                "a non-empty plane name; the audit attributes a manufactured feature to it")
        if self.axes is None:
            if not (self.not_extractable or "").strip():
                raise InvalidParameterError(
                    "RepresentationPlane.not_extractable", self.not_extractable,
                    "a stated reason. A plane with no declared axes is refused by the "
                    "audit, and a refusal with no reason is indistinguishable from an "
                    "omission")
            axes: Optional[Tuple[AxisSpec, ...]] = None
        else:
            if self.not_extractable is not None:
                raise InvalidParameterError(
                    "RepresentationPlane.not_extractable", self.not_extractable,
                    "None for a plane that declares its axes. A plane cannot both be "
                    "extractable and carry the reason it is not")
            axes = tuple(self.axes)
            if len(axes) != 2:
                raise InvalidParameterError(
                    "RepresentationPlane.axes", len(axes),
                    "one axis per plane dimension")
        factors = tuple(float(f) for f in self.parent_cells_per_sample)
        if len(factors) != 2 or any(not (math.isfinite(f) and f > 0.0) for f in factors):
            raise InvalidParameterError(
                "RepresentationPlane.parent_cells_per_sample", self.parent_cells_per_sample,
                "one positive finite factor per axis")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "parent_cells_per_sample", factors)
        object.__setattr__(self, "contaminated_halfwidth", max(0, int(self.contaminated_halfwidth)))
        object.__setattr__(self, "notes", dict(self.notes))

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def extractable(self) -> bool:
        return self.axes is not None

    @property
    def decimated(self) -> bool:
        return any(f != 1.0 for f in self.parent_cells_per_sample)

    def describe(self) -> Dict[str, Any]:
        return {
            "plane": self.name,
            "shape": list(self.shape),
            "extractable": self.extractable,
            "not_extractable": self.not_extractable,
            "parent_cells_per_sample": list(self.parent_cells_per_sample),
            "contaminated_halfwidth": self.contaminated_halfwidth,
            "axes": None if self.axes is None else [
                {"name": a.name, "role": a.role, "units": a.units,
                 "periodic": bool(a.periodic)} for a in self.axes],
            "notes": dict(self.notes),
        }


def decimation(parent_shape: Sequence[int], plane_shape: Sequence[int]) -> Tuple[float, ...]:
    """Parent cells per sample, taken from the two shapes rather than from the level number.

    Written after getting it wrong. The dual-tree lowpass of a three-level decomposition
    looks like it should be decimated by eight, because its six subbands are; it is decimated
    by *four*, because the lowpass is not carried through the final quad-to-complex step. The
    audit declared eight, `plane_field` built a coordinate axis twice as long as the field,
    and a feature planted at row 70 of a 128-cell frame came back at row 137 - outside the
    frame it was found in, in the frame's own units, with nothing in the record to say so.

    The ratio is the honest number and the level is the hypothesis. `representation_planes`
    checks every declared factor against its own array, so a builder cannot claim a
    decimation its output does not have.
    """
    return tuple(float(pa) / float(pl) for pa, pl in zip(parent_shape, plane_shape))


def searchable_cells(plane: RepresentationPlane) -> int:
    """Cells rule R13 leaves the extractor to look at, on this plane, at this size.

    The audit's second way of earning a pass by not looking. A dual-tree level-3 subband of a
    64-cell frame is 8 x 8 samples with a six-sample contaminated margin on every side, which
    is *no* valid interior at all: the extractor would dutifully report nothing, and the
    report would count a plane that could not have spoken as a plane that had nothing to say.
    Planes with an empty interior are refused by name and are struck from the family, because
    a test with no possible outcome is not a test and must not make the other tests stricter.

    Only the first R13 pass, the one that depends on the frame. The extractor applies a
    second, wider one sized by each feature's own measured scale, which cannot be predicted
    from the plane alone - so this is an upper bound on what was searched, never a claim about
    what was.
    """
    if plane.axes is None:
        return 0
    margin = max(1, int(plane.contaminated_halfwidth))
    cells = 1
    for size, axis in zip(plane.shape, plane.axes):
        cells *= int(size) if axis.periodic else max(0, int(size) - 2 * margin)
    return int(cells)


PlaneBuilder = Callable[[np.ndarray, Tuple[AxisSpec, ...], Mapping[str, Any]],
                        Sequence[RepresentationPlane]]

#: Representations, as a registry (standard E1). Keyed by the name of the registered transform
#: they read, plus `raw` for the identity - so that the field itself is audited by the same
#: machinery as everything derived from it, and TG2.2's floor is an entry here rather than a
#: separate argument.
REPRESENTATIONS: Registry[PlaneBuilder] = Registry("representation")


# --------------------------------------------------------------------- the null, as a choice


NullStrategy = Callable[..., Dict[str, List[float]]]

#: Where the null lives. The default propagates; the alternative is registered so the audit
#: can measure what rebuilding the null inside the representation costs, rather than this
#: module asserting it in a comment. A strategy returns the ensemble of **frame maxima** per
#: plane and nothing else: it does not get to choose the level, the order statistic or the
#: family, because those are the parts the correction below has to see all of at once.
NULL_STRATEGIES: Registry[NullStrategy] = Registry("representation null strategy")


@NULL_STRATEGIES.register(
    "propagate_through_representation",
    description=("Surrogate the field, push each surrogate through the same representation, "
                 "take the maximum in the plane."),
    capabilities={"null_lives_in": "the field", "carries_the_representation": True,
                  "transforms_per_surrogate": 1, "needs_field_ensemble": True},
)
def propagate_through_representation(
    field: ExtractionField,
    representation: str,
    config: Mapping[str, Any],
    planes: Sequence[RepresentationPlane],
    *,
    method: str,
    n_surrogates: int,
    seed: int,
    field_ensemble: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, List[float]]:
    """The null carries the representation, because the artefact does.

    An artefact of the lens is present in every realisation the lens is pointed at, including
    a structureless one. Push the ensemble through the lens and the artefact is in the null,
    where it belongs: it raises the cut, it is not reported, and the receipt says the cut was
    raised. Do anything else and the artefact is a discovery.

    One decomposition per surrogate serves every plane of the representation, and one
    ensemble of the field serves every representation - `_gather` builds it once, because the
    seed is the same for all of them and regenerating it per lens would be the identical
    arrays computed seven times (standard E4 makes that a waste rather than a difference).
    """
    from src.statistics import surrogates as surrogate_module    # declared dependency

    builder = REPRESENTATIONS.get(representation)
    axes = tuple(field.axes)
    wanted = [p.name for p in planes]
    maxima: Dict[str, List[float]] = {name: [] for name in wanted}
    members = (list(field_ensemble) if field_ensemble is not None
               else surrogate_module.generate(field.values, method=method,
                                              n=int(n_surrogates), seed=int(seed))["members"])
    if len(members) != int(n_surrogates):
        raise InvalidParameterError(
            "propagate_through_representation.field_ensemble", len(members),
            "an ensemble of exactly %d members" % int(n_surrogates))
    for member in members:
        produced = {p.name: p for p in builder(np.asarray(member, dtype=np.float64),
                                               axes, dict(config))}
        for name in wanted:
            if name not in produced:
                raise InvalidParameterError(
                    "REPRESENTATIONS[%r]" % representation, name,
                    "the same plane names for every input of the same shape. Plane %r was "
                    "produced for the observed field and not for a surrogate of it, so the "
                    "null and the observation are not comparable (standard E4)" % name)
            maxima[name].append(float(np.max(produced[name].values)))
    return maxima


@NULL_STRATEGIES.register(
    "randomise_in_representation",
    description=("Surrogate the coefficient plane itself. Registered to be measured against, "
                 "not to be used."),
    capabilities={"null_lives_in": "the coefficient plane",
                  "carries_the_representation": False, "transforms_per_surrogate": 0,
                  "needs_field_ensemble": False},
)
def randomise_in_representation(
    field: ExtractionField,
    representation: str,
    config: Mapping[str, Any],
    planes: Sequence[RepresentationPlane],
    *,
    method: str,
    n_surrogates: int,
    seed: int,
    field_ensemble: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, List[float]]:
    """Calibrate where the numbers are read - the obvious thing, and the wrong one.

    A phase-randomised surrogate of a coefficient plane keeps the plane's power spectrum and
    scatters its localisation. An artefact that sits in one place in every realisation - a
    seam, an edge, a checkerboard - is therefore spread out in the null and concentrated in
    the observation, which is the signature of a discovery. The representation put it there.
    """
    from src.statistics import surrogates as surrogate_module    # declared dependency

    out: Dict[str, List[float]] = {}
    for plane in planes:
        ensemble = surrogate_module.generate(
            plane.values, method=method, n=int(n_surrogates), seed=int(seed))
        out[plane.name] = [float(np.max(np.asarray(m))) for m in ensemble["members"]]
    return out


# ------------------------------------------------------------------ the audit's own family


@dataclass(frozen=True)
class CorrectedLevel:
    """The level each plane is actually tested at, once the family has been counted.

    Rule R18's problem, arriving in the audit itself rather than in the mining phase it was
    written for. One plane of one representation tested at alpha = 0.05 is one test; the
    audit runs twenty-nine of them and asks a single question of the answer - *did any
    registered representation manufacture a feature?* - which is a family-wise question, and
    asking it of twenty-nine uncorrected tests produces the findings arithmetic predicts.
    """

    method: str
    alpha_family: float
    alpha_per_plane: float
    family_size: int
    n_surrogates: int
    rank: Optional[int] = None
    measured_fwer: Optional[float] = None
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", dict(self.notes))

    def describe(self) -> Dict[str, Any]:
        return {
            "correction": self.method,
            "alpha_family": self.alpha_family,
            "alpha_per_plane": self.alpha_per_plane,
            "family_size": self.family_size,
            "n_surrogates": self.n_surrogates,
            "rank": self.rank,
            "measured_fwer": self.measured_fwer,
            "notes": dict(self.notes),
        }


class FamilyTooLargeError(InvalidParameterError):
    """Raised when the ensemble cannot resolve a family-wise level over this many planes."""

    def __init__(self, family_size: int, n_surrogates: int, alpha: float,
                 achievable: float, needed: int) -> None:
        super().__init__(
            "audit.n_surrogates", n_surrogates,
            "an ensemble that can resolve a family-wise level of %g over %d planes. The "
            "strictest cut %d surrogates can express is 'larger than every null maximum', "
            "and over this family that still rejects %.4g of the time - the audit would "
            "report features on structureless data at %.3gx the level it claims. About %d "
            "surrogates are needed. This is rule R18 applied to the audit itself: a family "
            "that cannot be afforded is refused before it is run, not corrected afterwards"
            % (alpha, family_size, n_surrogates, achievable, achievable / alpha, needed),
            family_size=family_size, achievable_fwer=achievable,
            required_surrogates=needed)


FamilyCorrection = Callable[..., CorrectedLevel]

#: How the audit pays for its own family. Registered rather than chosen in a branch, because
#: the three entries give measurably different answers on the same ensemble and a reader is
#: entitled to see which one produced the level on the receipt.
FAMILY_CORRECTIONS: Registry[FamilyCorrection] = Registry("audit family correction")


def _fwer_curve(maxima: np.ndarray) -> np.ndarray:
    """Family-wise rejection rate as a function of the shared per-plane rank.

    `maxima` is `(n_surrogates, n_planes)`. For surrogate `j` and plane `p`, `c[j, p]` counts
    the *other* surrogates whose maximum in that plane is at least as large. A cut placed at
    the `K`-th largest of those other maxima is exceeded exactly when `c[j, p] <= K`, so the
    family-wise rate at rank `K` is the fraction of surrogates for which some plane does.
    Leave-one-out on purpose: reusing a surrogate to build the cut it is then tested against
    is how a calibration flatters itself.
    """
    n = int(maxima.shape[0])
    counts = np.empty(maxima.shape, dtype=np.int64)
    for p in range(maxima.shape[1]):
        column = maxima[:, p]
        counts[:, p] = (column[None, :] >= column[:, None]).sum(axis=1) - 1
    best = counts.min(axis=1)
    return np.array([float(np.count_nonzero(best <= k)) / float(n)
                     for k in range(n)], dtype=np.float64)


@FAMILY_CORRECTIONS.register(
    "max_statistic",
    description=("Family-wise level read off the ensemble itself, so dependence between "
                 "planes is measured rather than assumed away."),
    capabilities={"assumes_independence": False, "uses_the_ensemble": True},
)
def max_statistic_correction(maxima: np.ndarray, *, alpha: float,
                             names: Sequence[str]) -> CorrectedLevel:
    """The loosest shared rank whose measured family-wise rate is still within alpha.

    Bands of one wavelet transform of one field are not independent - level 2 is built from
    level 1's approximation, and six orientations of one dual-tree level see the same
    coefficients through six rotations of one filter. Bonferroni prices that family as if
    they were independent and so pays for tests that were never separate. Here the price is
    read off the same surrogate ensemble the thresholds come from, which is the only place
    the actual dependence between these particular planes is written down.
    """
    curve = _fwer_curve(maxima)
    n = int(maxima.shape[0])
    size = int(maxima.shape[1])
    within = np.nonzero(curve <= float(alpha))[0]
    if within.size == 0:
        needed = int(math.ceil(float(size) / float(alpha)))
        raise FamilyTooLargeError(size, n, float(alpha), float(curve[0]), needed)
    rank = int(within[-1])
    uncorrected_rank = max(0, int(math.floor(float(alpha) * (1.0 + n))) - 1)
    return CorrectedLevel(
        method="max_statistic", alpha_family=float(alpha),
        alpha_per_plane=(rank + 1.0) / (1.0 + n), family_size=size,
        n_surrogates=n, rank=rank, measured_fwer=float(curve[rank]),
        notes={"uncorrected_rank": uncorrected_rank,
               "uncorrected_fwer": float(curve[uncorrected_rank]),
               "basis": "leave-one-out rejection rate of the ensemble the cuts come from"})


@FAMILY_CORRECTIONS.register(
    "bonferroni",
    description="alpha / family size, assuming the planes are independent tests.",
    capabilities={"assumes_independence": True, "uses_the_ensemble": False},
)
def bonferroni_correction(maxima: np.ndarray, *, alpha: float,
                          names: Sequence[str]) -> CorrectedLevel:
    """Correct, conservative, and registered so the cost of the assumption is visible."""
    n = int(maxima.shape[0])
    size = int(maxima.shape[1])
    per_plane = float(alpha) / float(size)
    floor = 1.0 / (1.0 + n)
    if per_plane < floor - 1e-12:
        raise FamilyTooLargeError(size, n, float(alpha), float(size) * floor,
                                  int(math.ceil(float(size) / float(alpha))))
    return CorrectedLevel(
        method="bonferroni", alpha_family=float(alpha), alpha_per_plane=per_plane,
        family_size=size, n_surrogates=n,
        notes={"assumption": "the planes are independent tests, which bands of one "
                             "decomposition of one field are not"})


@FAMILY_CORRECTIONS.register(
    "none",
    description="No correction. Registered to be measured against, not to be used.",
    capabilities={"assumes_independence": False, "uses_the_ensemble": False},
)
def no_correction(maxima: np.ndarray, *, alpha: float,
                  names: Sequence[str]) -> CorrectedLevel:
    """Every plane at the nominal level, and the family paid for by nobody."""
    return CorrectedLevel(
        method="none", alpha_family=float(alpha), alpha_per_plane=float(alpha),
        family_size=int(maxima.shape[1]), n_surrogates=int(maxima.shape[0]),
        notes={"expected_findings_on_a_structureless_field":
                   float(alpha) * float(maxima.shape[1])})


# ------------------------------------------------------------------------ plane -> a field


def plane_field(field: ExtractionField, representation: str, plane: RepresentationPlane,
                config: Optional[Mapping[str, Any]] = None) -> ExtractionField:
    """The plane, wearing the declaration `extract` refuses to work without.

    Two things are carried across and one is deliberately changed. The identity travels
    unchanged, so R19 still refuses a comparison across domains. The **representation label**
    becomes `transform/plane`, because `FeatureSet` refuses to mix representations and TG2.4
    is the reason that refusal exists: a feature manufactured by the level-3 diagonal band
    must be attributable to the level-3 diagonal band and to nothing coarser.

    The coordinates are rebuilt in the *parent's* units on a decimated plane, so a position
    or a width read off a level-3 band comes back in field cells rather than in band samples -
    a number eight times too small, in the same units, with nothing in the record to say so.
    """
    if plane.axes is None:
        raise InvalidParameterError(
            "plane_field.plane", plane.name,
            "a plane with declared axes. %s" % plane.not_extractable)
    coords: Optional[Dict[str, Any]] = None
    if plane.decimated or field.coordinates is not None:
        coords = {}
        for axis, size, factor in zip(plane.axes, plane.shape, plane.parent_cells_per_sample):
            origin = field.to_coordinate(axis.name, 0.0)
            step = field.spacing(axis.name) * float(factor)
            coords[axis.name] = origin + np.arange(size, dtype=np.float64) * step
    provenance = dict(field.provenance)
    provenance.update({
        "parent_representation": field.representation,
        "parent_shape": list(field.shape),
        "representation_config": dict(config or {}),
        "parent_cells_per_sample": list(plane.parent_cells_per_sample),
        "contaminated_halfwidth": plane.contaminated_halfwidth,
        "plane_notes": dict(plane.notes),
    })
    return ExtractionField(
        values=plane.values, axes=plane.axes, domain=field.domain, dataset=field.dataset,
        variable=field.variable, units=field.units, time=field.time,
        representation="%s/%s" % (representation, plane.name),
        time_units=field.time_units, coordinates=coords, provenance=provenance)


def representation_planes(representation: str, field: ExtractionField,
                          config: Optional[Mapping[str, Any]] = None
                          ) -> Tuple[RepresentationPlane, ...]:
    """Every plane one registered representation produces from one frame."""
    builder = REPRESENTATIONS.get(representation)         # UnknownNameError, did-you-mean
    planes = tuple(builder(field.values, tuple(field.axes), dict(config or {})))
    seen = set()
    for plane in planes:
        if not isinstance(plane, RepresentationPlane):
            raise InvalidParameterError(
                "REPRESENTATIONS[%r]" % representation, type(plane).__name__,
                "a RepresentationPlane")
        if plane.name in seen:
            raise InvalidParameterError(
                "REPRESENTATIONS[%r]" % representation, plane.name,
                "distinct plane names; a manufactured feature is attributed by name")
        seen.add(plane.name)
    if not planes:
        raise InvalidParameterError(
            "REPRESENTATIONS[%r]" % representation, 0,
            "at least one plane. A representation that produces nothing to look at cannot "
            "be audited, and reporting it as clean would be a pass earned by not looking")
    parent = field.shape
    for plane in planes:
        if plane.axes is None:
            continue
        for axis, size, factor, whole in zip(plane.axes, plane.shape,
                                             plane.parent_cells_per_sample, parent):
            covered = float(size) * float(factor)
            if abs(covered - float(whole)) > float(factor):
                raise InvalidParameterError(
                    "REPRESENTATIONS[%r] plane %r" % (representation, plane.name), factor,
                    "a decimation its own array has. %d samples at %g parent cells each "
                    "cover %g cells of axis %r, which is %d cells long. A coordinate built "
                    "from this factor runs off the field, and a feature found in this plane "
                    "would be reported at a position the field does not have"
                    % (size, factor, covered, axis.name, whole))
    return planes


# ---------------------------------------------------------------------------- the audit


@dataclass(frozen=True)
class PlaneAudit:
    """What the extractor did with one plane, or why it was not asked."""

    representation: str
    plane: RepresentationPlane
    calibration: Optional[NullCalibration] = None
    result: Optional[ExtractionResult] = None
    refused: Optional[str] = None
    null_spread: Optional[float] = None
    observed_maximum: Optional[float] = None
    searchable_cells: int = 0

    @property
    def audited(self) -> bool:
        return self.result is not None

    @property
    def found(self) -> int:
        return 0 if self.result is None else self.result.found

    @property
    def relative_spread(self) -> Optional[float]:
        if self.null_spread is None or self.observed_maximum is None:
            return None
        return float(self.null_spread) / max(abs(float(self.observed_maximum)), 1.0e-300)

    @property
    def vacuous(self) -> bool:
        """True when the null maxima are identical, so nothing could ever exceed them.

        Measured relative to the observed maximum rather than absolutely: a plane of tiny
        coefficients and a plane of large ones have to be judged on the same scale, and the
        question is whether the ensemble has any spread *compared with what it is gating*.
        """
        spread = self.relative_spread
        return spread is not None and spread <= VACUITY_TOLERANCE

    @property
    def clean(self) -> bool:
        """No manufactured features, from a test that had the power to say otherwise.

        Two ways to have no power, and both are refused rather than counted as a pass: a null
        nothing can exceed (`vacuous`) and a frame with nothing left to look at once R13 has
        taken its margin (`searchable_cells == 0`, which is refused before extraction and so
        never reaches this property as an audited plane).
        """
        return (self.audited and self.found == 0 and not self.vacuous
                and self.searchable_cells > 0)

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "representation": self.representation,
            "audited": self.audited,
            "refused": self.refused,
            "found": self.found,
            "vacuous": self.vacuous,
            "null_spread": self.null_spread,
            "relative_null_spread": self.relative_spread,
            "observed_maximum": self.observed_maximum,
            "searchable_cells": self.searchable_cells,
            **self.plane.describe(),
        }
        if self.calibration is not None:
            out["calibration"] = self.calibration.describe()
        if self.result is not None:
            out["rejected"] = dict(self.result.rejected)
            out["features"] = [f.describe() for f in self.result]
        return out


@dataclass(frozen=True)
class RepresentationAudit:
    """Every plane of one representation, and the settings that produced them."""

    representation: str
    strategy: str
    extractor: str
    level: CorrectedLevel
    method: str
    seed: int
    config: Mapping[str, Any] = dc_field(default_factory=dict)
    planes: Tuple[PlaneAudit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "planes", tuple(self.planes))
        object.__setattr__(self, "config", dict(self.config))

    @property
    def found(self) -> int:
        """Features reported across this representation's planes.

        Named for what it measures, not for what it is used for. On a structureless field it
        *is* the manufactured count, which is the number this module exists to drive to zero;
        on a field with structure the same property counts real findings, and a name that
        called those manufactured would be a lie told by the accessor.
        """
        return sum(p.found for p in self.planes)

    @property
    def audited_planes(self) -> Tuple[PlaneAudit, ...]:
        return tuple(p for p in self.planes if p.audited)

    @property
    def refused_planes(self) -> Tuple[PlaneAudit, ...]:
        return tuple(p for p in self.planes if not p.audited)

    @property
    def vacuous_planes(self) -> Tuple[PlaneAudit, ...]:
        return tuple(p for p in self.planes if p.vacuous)

    @property
    def auditable(self) -> bool:
        """True when at least one plane could be handed to the extractor at all.

        `fft` and `dct` are registered, are looked at, and are `auditable=False`. That is a
        third state and it is kept as one: reporting them as clean would claim a floor that
        was never measured, and reporting them as dirty would blame them for a gap in the
        extractor registry rather than in themselves.
        """
        return bool(self.audited_planes)

    @property
    def clean(self) -> bool:
        return self.auditable and all(p.clean for p in self.audited_planes)

    def plane(self, name: str) -> PlaneAudit:
        for item in self.planes:
            if item.plane.name == name:
                return item
        raise InvalidParameterError(
            "RepresentationAudit.plane", name,
            "a plane of representation %r: %s"
            % (self.representation, ", ".join(p.plane.name for p in self.planes)))

    def describe(self) -> Dict[str, Any]:
        return {
            "representation": self.representation,
            "strategy": self.strategy,
            "extractor": self.extractor,
            "surrogate_method": self.method,
            "seed": self.seed,
            "level": self.level.describe(),
            "config": dict(self.config),
            "planes_total": len(self.planes),
            "planes_audited": len(self.audited_planes),
            "planes_refused": len(self.refused_planes),
            "found": self.found,
            "vacuous": [p.plane.name for p in self.vacuous_planes],
            "clean": self.clean,
            "detail": [p.describe() for p in self.planes],
        }


@dataclass(frozen=True)
class AuditReport:
    """Every registered representation, plus the ones nothing could audit.

    `uncovered_transforms` is the load-bearing field. A registered transform with no
    registered plane builder is a lens this audit never looked through, and a summary that
    said "all clean" without naming it would be exactly the "we did not look" that the
    benchmark harness's three outcomes exist to prevent.
    """

    audits: Tuple[RepresentationAudit, ...] = ()
    uncovered_transforms: Tuple[str, ...] = ()
    level: Optional[CorrectedLevel] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "audits", tuple(self.audits))
        object.__setattr__(self, "uncovered_transforms", tuple(self.uncovered_transforms))

    @property
    def found(self) -> int:
        """Features reported across the whole audit; the manufactured count on a null field."""
        return sum(a.found for a in self.audits)

    @property
    def auditable(self) -> Tuple[RepresentationAudit, ...]:
        return tuple(a for a in self.audits if a.auditable)

    @property
    def unauditable(self) -> Tuple[RepresentationAudit, ...]:
        """Registered representations no registered extractor can be pointed at.

        Named, not dropped. This is the honest shape of the gap R8 leaves open here: the
        frequency-plane lenses are audited for their null and not for their peaks, and the
        thing that would close it is a registered extractor with a spectral shape model.
        """
        return tuple(a for a in self.audits if not a.auditable)

    @property
    def clean(self) -> bool:
        """No manufactured feature, nothing unlooked-at, and every verdict had power.

        A representation with no extractable plane is excluded from the verdict rather than
        counted as a pass - it appears in `unauditable`, which the benchmark prints.
        """
        return (bool(self.auditable) and all(a.clean for a in self.auditable)
                and not self.uncovered_transforms)

    def audit(self, representation: str) -> RepresentationAudit:
        for item in self.audits:
            if item.representation == representation:
                return item
        raise InvalidParameterError(
            "AuditReport.audit", representation,
            "an audited representation: %s"
            % ", ".join(a.representation for a in self.audits))

    def findings(self) -> Tuple[Tuple[str, str, int], ...]:
        """`(representation, plane, features)` for every plane that reported one."""
        return tuple((a.representation, p.plane.name, p.found)
                     for a in self.audits for p in a.planes if p.found)

    def vacuous(self) -> Tuple[str, ...]:
        return tuple("%s/%s" % (a.representation, p.plane.name)
                     for a in self.audits for p in a.vacuous_planes)

    @property
    def searched(self) -> int:
        """Cells the extractor was actually free to report a feature in, across the audit.

        The number that stops "nothing was manufactured" from being a statement about how
        little was looked at.
        """
        return sum(p.searchable_cells for a in self.audits for p in a.audited_planes)

    def describe(self) -> Dict[str, Any]:
        return {
            "representations": [a.representation for a in self.audits],
            "uncovered_transforms": list(self.uncovered_transforms),
            "planes_audited": sum(len(a.audited_planes) for a in self.audits),
            "planes_refused": sum(len(a.refused_planes) for a in self.audits),
            "searchable_cells": self.searched,
            "found": self.found,
            "findings": [list(o) for o in self.findings()],
            "unauditable": [a.representation for a in self.unauditable],
            "vacuous": list(self.vacuous()),
            "level": None if self.level is None else self.level.describe(),
            "clean": self.clean,
            "audits": [a.describe() for a in self.audits],
        }


@dataclass(frozen=True)
class _Gathered:
    """One representation's planes and their null maxima, before any level is chosen."""

    representation: str
    config: Dict[str, Any]
    planes: Tuple[RepresentationPlane, ...]
    maxima: Dict[str, List[float]]


def _gather(field: ExtractionField, names: Sequence[str],
            configs: Optional[Mapping[str, Mapping[str, Any]]], *,
            strategy: str, method: str, n_surrogates: int, seed: int) -> List[_Gathered]:
    """Planes and null maxima for each named representation. No thresholds yet.

    Deliberately separated from the thresholding. The level depends on the size of the whole
    family, and a function that calibrated one representation as it went would have to know
    the family before it had finished counting it.
    """
    entry = NULL_STRATEGIES.entry(strategy)               # UnknownNameError, did-you-mean
    build_null = entry.value
    ensemble: Optional[List[np.ndarray]] = None
    if entry.capabilities.get("needs_field_ensemble"):
        from src.statistics import surrogates as surrogate_module

        ensemble = [np.asarray(m, dtype=np.float64) for m in surrogate_module.generate(
            field.values, method=method, n=int(n_surrogates), seed=int(seed))["members"]]
    out: List[_Gathered] = []
    for name in names:
        config = dict((configs or {}).get(name) or {})
        planes = representation_planes(name, field, config)
        maxima = build_null(field, name, config, planes,
                            method=method, n_surrogates=int(n_surrogates), seed=int(seed),
                            field_ensemble=ensemble)
        missing = [p.name for p in planes if p.name not in maxima]
        if missing:
            raise InvalidParameterError(
                "NULL_STRATEGIES[%r]" % strategy, missing,
                "an ensemble of maxima for every plane the representation produced")
        out.append(_Gathered(representation=name, config=config, planes=planes,
                             maxima=maxima))
    return out


def _level_for(gathered: Sequence[_Gathered], *, correction: str, alpha: float
               ) -> CorrectedLevel:
    """The family is every plane that will actually be tested, counted before any is read.

    Only the planes that will actually be tested: extractable, and with an interior left to
    search. A family is a count of tests, and a plane the extractor is never pointed at is
    not a test - charging the audit for `fft/magnitude` would make every other verdict
    stricter to pay for a question nobody asked. Their nulls are still measured, and their
    vacuity is still reported; they are simply not in the family.
    """
    pairs = [(g, p) for g in gathered for p in g.planes
             if p.extractable and searchable_cells(p) > 0]
    if not pairs:
        raise InvalidParameterError(
            "audit.representations", [g.representation for g in gathered],
            "at least one plane that can be tested. Every plane offered either declares no "
            "axes or has no valid interior left at this size, so there is no test to "
            "correct and nothing to report as a floor")
    names = [(g.representation, p.name) for g, p in pairs]
    columns = [np.asarray(g.maxima[p.name], dtype=np.float64) for g, p in pairs]
    sizes = {c.size for c in columns}
    if len(sizes) != 1:
        raise InvalidParameterError(
            "audit.n_surrogates", sorted(sizes),
            "one ensemble size across the family. A family-wise level cannot be read off "
            "columns of different lengths, because a rank means a different level in each")
    matrix = np.column_stack(columns)
    apply_correction = FAMILY_CORRECTIONS.get(correction)   # UnknownNameError, did-you-mean
    return apply_correction(matrix, alpha=float(alpha),
                            names=["%s/%s" % pair for pair in names])


def _finish(field: ExtractionField, gathered: Sequence[_Gathered], level: CorrectedLevel, *,
            strategy: str, extractor: str, method: str, seed: int,
            params: Mapping[str, Any]) -> Tuple[RepresentationAudit, ...]:
    """Thresholds at the corrected level, then the extractor, then the records."""
    audits: List[RepresentationAudit] = []
    for group in gathered:
        planes: List[PlaneAudit] = []
        for plane in group.planes:
            maxima = group.maxima[plane.name]
            calibration = calibration_from_maxima(
                maxima, alpha=level.alpha_per_plane, method=method, seed=int(seed),
                propagated_through=("%s/%s" % (group.representation, plane.name)
                                    if strategy == "propagate_through_representation"
                                    else None))
            spread = (max(maxima) - min(maxima)) if maxima else 0.0
            observed = float(np.max(plane.values))
            cells = searchable_cells(plane)
            if not plane.extractable:
                planes.append(PlaneAudit(
                    representation=group.representation, plane=plane,
                    calibration=calibration, refused=plane.not_extractable,
                    null_spread=spread, observed_maximum=observed, searchable_cells=cells))
                continue
            if cells <= 0:
                planes.append(PlaneAudit(
                    representation=group.representation, plane=plane,
                    calibration=calibration, null_spread=spread, observed_maximum=observed,
                    searchable_cells=cells,
                    refused=("this plane is %d x %d samples and its filter contaminates %d "
                             "on every side, so rule R13 leaves no valid interior. The "
                             "extractor would report nothing here whatever the data were, "
                             "so the plane is refused rather than counted as a floor - "
                             "auditing this representation at this level needs a larger frame"
                             % (plane.shape[0], plane.shape[1],
                                max(1, plane.contaminated_halfwidth)))))
                continue
            derived = plane_field(field, group.representation, plane, group.config)
            result = extract(derived, extractor, calibration=calibration,
                             boundary_margin=max(1, plane.contaminated_halfwidth), **params)
            planes.append(PlaneAudit(
                representation=group.representation, plane=plane, calibration=calibration,
                result=result, null_spread=spread, observed_maximum=observed,
                searchable_cells=cells))
        audits.append(RepresentationAudit(
            representation=group.representation, strategy=strategy, extractor=extractor,
            level=level, method=method, seed=int(seed), config=group.config,
            planes=tuple(planes)))
    return tuple(audits)


def audit_representation(
    field: ExtractionField,
    representation: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    strategy: str = DEFAULT_NULL_STRATEGY,
    correction: str = DEFAULT_FAMILY_CORRECTION,
    extractor: str = "local_maximum",
    alpha: float = DEFAULT_ALPHA,
    method: str = DEFAULT_SURROGATE_METHOD,
    n_surrogates: int = DEFAULT_AUDIT_SURROGATES,
    seed: int = DEFAULT_SEED,
    **params: Any,
) -> RepresentationAudit:
    """Run one registered extractor over every plane of one registered representation.

    The family here is this representation's own planes. Asking the same question of the
    whole registry is `audit_all`, and it is not the same test: nineteen dual-tree subbands
    corrected among themselves are corrected for a smaller family than the same nineteen
    corrected alongside every other lens the platform owns.
    """
    if not isinstance(field, ExtractionField):
        raise InvalidParameterError(
            "audit_representation.field", type(field).__name__,
            "an ExtractionField carrying the array and its declaration")
    gathered = _gather(field, [representation], {representation: dict(config or {})},
                       strategy=strategy, method=method, n_surrogates=n_surrogates,
                       seed=seed)
    level = _level_for(gathered, correction=correction, alpha=alpha)
    return _finish(field, gathered, level, strategy=strategy, extractor=extractor,
                   method=method, seed=seed, params=params)[0]


def audit_all(
    field: ExtractionField,
    *,
    representations: Optional[Sequence[str]] = None,
    configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strategy: str = DEFAULT_NULL_STRATEGY,
    correction: str = DEFAULT_FAMILY_CORRECTION,
    extractor: str = "local_maximum",
    alpha: float = DEFAULT_ALPHA,
    method: str = DEFAULT_SURROGATE_METHOD,
    n_surrogates: int = DEFAULT_AUDIT_SURROGATES,
    seed: int = DEFAULT_SEED,
    **params: Any,
) -> AuditReport:
    """The audit R8 asks for: every registered representation, and a named gap for the rest.

    The family is every plane of every representation, counted once and corrected once. The
    question being asked is a single one - *did any registered lens manufacture a feature in
    a field that has none?* - and a single question asked of a hundred planes is a hundred
    chances to answer it wrongly unless the family is paid for.

    Coverage is taken from the *transform* registry, not from this module's own, so that
    adding a transform without a plane builder shows up as an uncovered lens instead of
    silently shrinking the audit.
    """
    from src.transform_engine.registry import TRANSFORMS

    if not isinstance(field, ExtractionField):
        raise InvalidParameterError(
            "audit_all.field", type(field).__name__,
            "an ExtractionField carrying the array and its declaration")
    names = (list(representations) if representations is not None
             else list(REPRESENTATIONS.names()))
    gathered = _gather(field, names, configs, strategy=strategy, method=method,
                       n_surrogates=n_surrogates, seed=seed)
    level = _level_for(gathered, correction=correction, alpha=alpha)
    audits = _finish(field, gathered, level, strategy=strategy, extractor=extractor,
                     method=method, seed=seed, params=params)
    uncovered = tuple(sorted(set(TRANSFORMS.names()) - set(names)))
    return AuditReport(audits=audits, uncovered_transforms=uncovered, level=level)


# ------------------------------------------------------------- the registered representations


#: Why a frequency-plane array is not handed to a spatial peak-finder. Written once because
#: the reason is the same for every such plane and a reader should be able to see that it is.
_SPECTRAL_REFUSAL = (
    "the axes of this plane are wavenumbers, not positions. The registered extractors "
    "declare a spatial shape model and report a position, a width and a separation in the "
    "units of the declared axes; declaring these axes `space` so one of them would accept "
    "the plane is the error standard E14 exists to prevent, and it would license three "
    "numbers this plane does not have. Auditing %s for manufactured *spectral* peaks needs "
    "an extractor whose shape model is a spectral one, and none is registered.")


def _physical(values: np.ndarray):
    """A `PhysicalField` in index space, which is what every transform in the tree takes."""
    import torch

    from src.physical_core.field import PhysicalField
    from src.physical_core.grid import GridSpec

    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return PhysicalField(torch.from_numpy(arr), grid=GridSpec(kind="pixel", shape=arr.shape))


def _np(tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


@REPRESENTATIONS.register(
    "raw",
    description="The identity: the field itself, as its own representation.",
    capabilities={"decimated": False, "spatial_planes": True, "planes_per_level": 1,
                  "backed_by_transform": False},
)
def raw_planes(values, axes, config):
    """TG2.2's floor, restated as a registry entry rather than as a special case.

    Its null propagates through the identity, which makes it exactly `calibrate`'s null - so
    the audit's default strategy is verifiably a generalisation of the calibration the rest
    of the tree already uses, not a second opinion about it.
    """
    return (RepresentationPlane(
        name="field", values=values, axes=axes,
        notes={"transform": None, "what_this_is": "the values as given"}),)


@REPRESENTATIONS.register(
    "swt",
    description="Undecimated stationary wavelet bands, every one on the parent grid.",
    capabilities={"decimated": False, "spatial_planes": True, "planes_per_level": 3,
                  "backed_by_transform": True, "shift_invariant": True},
)
def swt_planes(values, axes, config):
    """The detail bands and the final approximation, all at the field's own sampling.

    `mode` is the audit's sharpest question of this transform. `periodic` wraps the
    convolution, so a field whose left and right edges do not match acquires a discontinuity
    that is an artefact of the boundary rule and of nothing else - and it lands in exactly
    the place a seam-crossing structure would.
    """
    from src.transform_engine import stationary as swt_mod

    levels = int(config.get("levels", 3))
    wavelet = str(config.get("wavelet", "db2"))
    mode = str(config.get("mode", "periodic"))
    coeffs = swt_mod.apply_swt2d(_physical(values), levels=levels, wavelet=wavelet, mode=mode)
    margins = coeffs["meta"]["valid_interior_halfwidth"]
    planes: List[RepresentationPlane] = []
    for level in range(1, levels + 1):
        halfwidth = int(margins[level])
        for band in ("LH", "HL", "HH"):
            planes.append(RepresentationPlane(
                name="level_%d/%s" % (level, band),
                values=_np(coeffs["level_%d" % level][band]), axes=axes,
                contaminated_halfwidth=halfwidth,
                notes={"transform": "swt", "wavelet": wavelet, "mode": mode,
                       "level": level, "band": band}))
    planes.append(RepresentationPlane(
        name="LL", values=_np(coeffs["LL"]), axes=axes,
        contaminated_halfwidth=int(margins[levels]),
        notes={"transform": "swt", "wavelet": wavelet, "mode": mode, "band": "LL",
               "what_this_is": "the approximation after %d levels" % levels}))
    return tuple(planes)


@REPRESENTATIONS.register(
    "dwt",
    description="Decimated Haar bands, each on its own coarser grid.",
    capabilities={"decimated": True, "spatial_planes": True, "planes_per_level": 3,
                  "backed_by_transform": True, "shift_invariant": False},
)
def dwt_planes(values, axes, config):
    """Every band on its own grid, with the decimation declared rather than divided out.

    A decimated transform is not shift-invariant: translating the field by one cell changes
    which samples survive, so its bands carry a phase that is a property of where the grid
    happens to start. That is the artefact this plane set exists to expose, and it is why the
    parent-cell factor is on the record - a width of 2 in a level-3 band is a width of 16 in
    the field, and reporting the 2 would be wrong by three octaves in the units of the field.
    """
    from src.transform_engine.transforms import SpectralTransformEngine as STE

    levels = int(config.get("levels", 3))
    parent = tuple(np.asarray(values).shape)
    coeffs = STE.apply_dwt2d(_physical(values), levels=levels)
    planes: List[RepresentationPlane] = []
    for level in range(1, levels + 1):
        for band in ("LH", "HL", "HH"):
            plane = _np(coeffs["level_%d" % level][band])
            planes.append(RepresentationPlane(
                name="level_%d/%s" % (level, band),
                values=plane, axes=axes,
                parent_cells_per_sample=decimation(parent, plane.shape),
                notes={"transform": "dwt", "wavelet": "haar", "level": level, "band": band,
                       "sample_alignment": (
                           "nominal: sample i covers parent cells [i*f, (i+1)*f). No "
                           "sub-cell phase correction is applied, so a position read from "
                           "this plane is good to about half a parent cell of the factor")}))
    approximation = _np(coeffs["LL"])
    planes.append(RepresentationPlane(
        name="LL", values=approximation, axes=axes,
        parent_cells_per_sample=decimation(parent, approximation.shape),
        notes={"transform": "dwt", "wavelet": "haar", "band": "LL",
               "what_this_is": "the approximation after %d levels" % levels}))
    return tuple(planes)


@REPRESENTATIONS.register(
    "dtcwt",
    description="Dual-tree complex wavelet subband magnitudes, six orientations per level.",
    capabilities={"decimated": True, "spatial_planes": True, "planes_per_level": 6,
                  "backed_by_transform": True, "shift_invariant": "approximate",
                  "complex": True},
)
def dtcwt_planes(values, axes, config):
    """Magnitude, not the complex coefficient, and the reason is not presentation.

    A complex band has no maximum: an ordering of complex numbers is a choice, and any choice
    would make the threshold a statement about the branch cut. The magnitude is the shift-
    stable quantity the transform is built to provide, it is real, and its maximum is the same
    quantity the null's maximum is - which is the only condition the calibration needs.
    """
    from src.transform_engine import dtcwt as dtcwt_mod

    levels = int(config.get("levels", 3))
    level1 = str(config.get("level1", "near_sym_b"))
    qshift = str(config.get("qshift", "qshift_b"))
    parent = tuple(np.asarray(values).shape)
    coeffs = dtcwt_mod.apply_dtcwt2d(_physical(values), levels=levels,
                                     level1=level1, qshift=qshift)
    orientations = list(coeffs["feature_orientations_deg"])
    planes: List[RepresentationPlane] = []
    for index, band in enumerate(coeffs["highpass"]):
        level = index + 1
        halfwidth = int(dtcwt_mod.native_halfwidth(level, level1, qshift))
        magnitude = band.abs()
        for k in range(magnitude.shape[2]):
            plane = _np(magnitude[:, :, k])
            planes.append(RepresentationPlane(
                name="level_%d/orientation_%g" % (level, orientations[k]),
                values=plane, axes=axes,
                parent_cells_per_sample=decimation(parent, plane.shape),
                contaminated_halfwidth=halfwidth,
                notes={"transform": "dtcwt", "level1": level1, "qshift": qshift,
                       "level": level, "quantity": "complex magnitude",
                       "feature_orientation_deg": float(orientations[k])}))
    lowpass = _np(coeffs["lowpass"])
    lowpass_factor = decimation(parent, lowpass.shape)
    planes.append(RepresentationPlane(
        name="lowpass", values=lowpass, axes=axes,
        parent_cells_per_sample=lowpass_factor,
        contaminated_halfwidth=int(dtcwt_mod.native_halfwidth(levels, level1, qshift)),
        notes={"transform": "dtcwt", "level1": level1, "qshift": qshift,
               "what_this_is": "the real coarse approximation after %d levels" % levels,
               "decimation_note": (
                   "measured, not assumed. The subbands of level %d are decimated by %d and "
                   "this is decimated by %g, because the lowpass does not go through the "
                   "quad-to-complex step that halves the highpass again"
                   % (levels, 2 ** levels, lowpass_factor[0]))}))
    return tuple(planes)


@REPRESENTATIONS.register(
    "hybrid",
    description="FFT low-pass reconstruction plus decimated Haar bands of the residual.",
    capabilities={"decimated": "mixed", "spatial_planes": True, "planes_per_level": 3,
                  "backed_by_transform": True},
)
def hybrid_planes(values, axes, config):
    """Both halves, on their own terms.

    The low-pass half is reconstructed back into the field's own sampling rather than being
    read in the frequency plane it was built in, because that is the only form in which it is
    a map of the domain. The filtered magnitude *is* carried, as an unextractable plane, so
    the audit measures its null instead of pretending the half does not exist.
    """
    from src.transform_engine.transforms import SpectralTransformEngine as STE

    crossover = float(config.get("crossover_freq", 0.5))
    coeffs = STE.apply_hybrid(_physical(values), crossover_freq=crossover,
                              mixing_weight=float(config.get("mixing_weight", 0.5)))
    shape = tuple(np.asarray(values).shape)
    low = STE.inverse_fft2d(coeffs["fft_mag_filtered"], coeffs["fft_phase"], shape)
    planes: List[RepresentationPlane] = [
        RepresentationPlane(
            name="lowpass_field", values=_np(low.data), axes=axes,
            notes={"transform": "hybrid", "crossover_freq": crossover,
                   "what_this_is": "the low-pass half, reconstructed onto the field's grid"}),
        RepresentationPlane(
            name="fft_magnitude_filtered", values=_np(coeffs["fft_mag_filtered"]),
            not_extractable=_SPECTRAL_REFUSAL % "hybrid",
            notes={"transform": "hybrid", "crossover_freq": crossover}),
    ]
    dwt = coeffs["dwt_coeffs"]
    for band in ("LH", "HL", "HH"):
        detail = _np(dwt["level_1"][band])
        planes.append(RepresentationPlane(
            name="residual/level_1/%s" % band, values=detail, axes=axes,
            parent_cells_per_sample=decimation(shape, detail.shape),
            notes={"transform": "hybrid", "band": band,
                   "what_this_is": "Haar detail of the high-frequency residual"}))
    approximation = _np(dwt["LL"])
    planes.append(RepresentationPlane(
        name="residual/LL", values=approximation, axes=axes,
        parent_cells_per_sample=decimation(shape, approximation.shape),
        notes={"transform": "hybrid", "band": "LL"}))
    return tuple(planes)


@REPRESENTATIONS.register(
    "fft",
    description="2D real FFT magnitude. Measured, never extracted from.",
    capabilities={"decimated": False, "spatial_planes": False, "backed_by_transform": True},
)
def fft_planes(values, axes, config):
    """One plane, unextractable, and audited anyway - for its null rather than its peaks.

    The phase is deliberately absent. It is circular-valued, so its maximum is a fact about
    where the branch cut was put and about nothing in the data, and a null built on it would
    be calibrating an arbitrary convention.
    """
    from src.transform_engine.transforms import SpectralTransformEngine as STE

    magnitude, _phase = STE.apply_fft2d(_physical(values))
    return (RepresentationPlane(
        name="magnitude", values=_np(magnitude),
        not_extractable=_SPECTRAL_REFUSAL % "fft",
        notes={"transform": "fft", "quantity": "|F(k)| on the rfft half-plane",
               "phase_omitted": ("circular-valued: its maximum is a property of the branch "
                                 "cut, not of the field")}),)


@REPRESENTATIONS.register(
    "dct",
    description="2D DCT-II coefficients. Measured, never extracted from.",
    capabilities={"decimated": False, "spatial_planes": False, "backed_by_transform": True},
)
def dct_planes(values, axes, config):
    from src.transform_engine.transforms import SpectralTransformEngine as STE

    return (RepresentationPlane(
        name="coefficients", values=_np(STE.apply_dct2d(_physical(values))),
        not_extractable=_SPECTRAL_REFUSAL % "dct",
        notes={"transform": "dct", "quantity": "DCT-II coefficients"}),)
