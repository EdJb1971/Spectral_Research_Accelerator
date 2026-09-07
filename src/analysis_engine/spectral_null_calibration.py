"""T4E.7: a clustering radius measured against a null, with no replicates anywhere.

T4E.3 measured the radius from **replicates** -- repeated measurements of one physical
configuration -- and D97 recorded why that has no valid input here: a real atmospheric record
contains no repeated measurements of one state, only one state evolving.  T4E.6 then measured
what the resulting radius actually does and found two things.  The radius is **arbitrary**: it
moved by a factor of 6.7 across 25 configurations of the same record with the same metric and
the same instrument.  And the one error rate the calibration reported is a **tautology** at its
own operating point: the radius *is* the largest replicate distance, so no replicate pair can
exceed it and the measured false-split rate there is identically zero on any input whatever.

This module replaces both.  It needs no replicates, it works on a real record, and neither rate
it publishes is zero by construction.

**The question it asks.**  Not "how far apart are two measurements of one thing", which the
record cannot answer, but: *how many pairs of signatures land close together, and how many would
land that close if the record contained no recurring configuration at all?*  The second number
is a null, which is this programme's own idiom (T4C.2's `surrogate_null.py`, T4F.3's precursor
tests).  The first minus the second is what recurrence has to explain.

**The estimator, stated so it can be argued with.**  Write ``F_obs(r)`` for the fraction of
observed signature pairs within distance ``r`` and ``F_null(r)`` for the same fraction on the
null.  Model the observed pairs as a two-component mixture -- a fraction ``pi`` that are the same
recurring configuration, the rest unrelated -- and take the null to stand for the unrelated
component::

    F_obs = pi * F_same + (1 - pi) * F_null
    excess(r) = F_obs(r) - F_null(r) = pi * (F_same(r) - F_null(r))

Three quantities follow, and **they are not the same kind of thing**:

*   The **admission rate** at ``r`` is ``F_null(r)``.  It is *measured*: the fraction of pairs
    known to be unrelated that this radius nonetheless admits.  It is the T4E.6 admission rate
    with the null supplying the contrast population that a real record cannot label by hand.
*   The **contamination rate** at ``r`` is ``F_null(r) * (1 - pi) / F_obs(r)``: of the pairs this
    radius actually admits, the share chance explains.  It is the rate that decides whether a
    pattern means anything, and it is what makes the calibration refuse radii the admission rate
    alone made look clean -- on the synthetic record the replicate-style reading called a radius
    4.6% admitting when 55% of what it admitted was chance.

    **It is an estimate and not a bound**, and an earlier draft of this module said otherwise.
    The ``1 - pi`` factor is conservative, but the whole expression puts ``F_null`` where the
    record's *own* unrelated-pair distribution belongs, and those differ by however well the
    null describes the record.  Measured against labels on a planted synthetic: at 17 of 73
    radii the published rate sat **below** the truth, by up to 0.11 even inside the region a
    radius is chosen from.  It tracked the truth exactly at the radius actually chosen, and it
    is in the bulk -- where no operating point is ever taken -- that it wanders furthest.
*   The **relative loss** at ``r`` is one minus the share of the recurrence-attributable
    close-pair mass this radius reaches, measured against the largest such mass the sweep
    reliably attains.  It is explicitly relative, because the absolute denominator -- how many
    pairs really are the same configuration -- is not identifiable.

**What is deliberately not published: an absolute split rate.**  It would be ``1 - F_same(r)``
with ``F_same = excess / pi + F_null``, and it cannot be trusted, because ``pi`` is read where
the same-configuration component is assumed to have saturated and returns ``pi * F_same`` where
it has not.  On a synthetic record with a planted grouping this module reported 1.91% against a
true 12.53%, measured from labels it never saw.  An error rate that is optimistic by six-fold is
worse than an absent one, so it is absent, and `VERIFICATION.md` carries the measurement.

``pi`` is estimated where ``F_same`` has saturated but ``F_null`` has not yet risen: there
``excess(r) = pi * (1 - F_null(r))``, so ``pi = excess(r) / (1 - F_null(r))``.  That region is
found rather than declared -- it is **every radius whose excess clears the significance band**,
which is to say every radius at which the record demonstrably differs from the null at all --
bounded above by a declared cap on ``F_null`` so the read stays out of the bulk, and the
estimate is taken at each of them so **the spread across them is published**.

Reading ``pi`` this way rather than at a declared quantile of the bulk is not a refinement, it is
the difference between an estimate and noise.  The unrelated-pair distribution of a record with
``C`` distinct configurations is a sample of only ``C`` draws, however many signatures it
contains, so ``F_obs`` in the bulk carries an error of order ``1/sqrt(C)`` while ``pi`` itself is
of order ``1/C``.  In the close-pair tail both distributions are near zero, that error collapses,
and the same quantity is recoverable.  A first implementation read ``pi`` at the null's median and
returned **negative** fractions on a synthetic record whose planted grouping the separation test
detected cleanly; the measurement is in `VERIFICATION.md`.

**Three ways this refuses, all of them reachable.**

1.  ``OBSERVED_DOES_NOT_DEPART_FROM_NULL`` -- the largest excess of close pairs anywhere on the
    sweep is one the null ensemble reaches on its own.  The record shows no recurrence this
    signature can see, which is a finding about the signature and not a failure of the search.
2.  ``RECURRENCE_FRACTION_UNSTABLE`` -- the estimate moves across the radii it is read at by
    more than the declared tolerance, or there is only one such radius and its agreement with
    anything is unmeasured. Every rate derived from it would be arbitrary in exactly the way
    T4E.6 found the replicate radius to be.
3.  ``NO_RADIUS_HOLDS_BOTH_RATES`` -- a fraction was estimated but no radius holds both rates at
    the level asked for.  The best achievable worst-rate is published alongside, because "none"
    is much more useful with a number attached.

A fourth status existed until mutation testing asked whether anything could reach it.  A
non-positive mixture fraction cannot coexist with a significant excess: the largest observed
distance puts ``F_obs`` at 1 while ``F_null`` can only be at most 1, so the maximum excess is
non-negative on every input, and it is zero only when every null distance lies inside the
record's own range -- in which case every leave-one-out statistic is non-negative too and the
p-value is 1.  The status was removed rather than left as a refusal nobody could ever see.
`estimate_recurrence_fraction` still reports the non-positive case, because it is reachable
when that function is called directly with a band of its caller's choosing.

**What this module does not decide.**  It does not generate the null.  The null is the
hypothesis, and generating it means re-running the pipeline -- decomposition, tracking,
constellations, signing -- over surrogate records, which belongs to the run and not to the
estimator.  A caller supplies the null distance populations together with a `SignatureNull`
saying what the null preserves and what it destroys, and that declaration travels in every
record this module produces, because an excess over one null is not an excess over another.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.spectral_clustering import (
    SignatureMetric, SignaturePoint, _points,
)
from src.core.errors import InvalidParameterError
from src.statistics.multiple_comparisons import required_surrogates

NULL_CALIBRATION_SCHEMA = "spectral-identity-null-calibration/v1"

#: A radius was found and both rates are attached to it.
CALIBRATION_MEASURED = "MEASURED"
#: The largest excess anywhere on the sweep is one the null ensemble reaches unaided.
CALIBRATION_NO_SEPARATION = "OBSERVED_DOES_NOT_DEPART_FROM_NULL"
#: The mixture fraction depends on where it is read from by more than the declared tolerance.
CALIBRATION_UNSTABLE = "RECURRENCE_FRACTION_UNSTABLE"
#: A fraction was estimated, but no radius holds both rates at the level asked for.
CALIBRATION_NO_OPERATING_POINT = "NO_RADIUS_HOLDS_BOTH_RATES"

CALIBRATION_STATUSES = (
    CALIBRATION_MEASURED, CALIBRATION_NO_SEPARATION,
    CALIBRATION_UNSTABLE, CALIBRATION_NO_OPERATING_POINT,
)

#: How far the mixture fraction may move across the radii it is read at before the calibration
#: refuses. Absolute, in fraction-of-pairs units, and declared rather than tuned to an answer.
DEFAULT_STABILITY_TOLERANCE = 0.05

#: The largest null fraction at which the mixture fraction may still be read. Past this the
#: estimator is reading the bulk, where the unrelated component of a record with ``C`` distinct
#: configurations departs from the null by order ``1 / sqrt(C)`` however many signatures the
#: record contains -- an error that swamps a fraction of order ``1 / C``. Declared, and the
#: measurement that fixed its value is in `VERIFICATION.md`.
DEFAULT_NULL_FRACTION_CAP = 0.25

CALIBRATION_CLAIM_BOUNDARY = (
    "This calibration claims a radius at which the record's signature pairs are closer together "
    "than the declared null explains, and it publishes two rates for that radius. The "
    "contamination rate estimates the share of what the radius admits that chance explains, and "
    "is an estimate rather than a bound: it stands or falls with how well the declared null "
    "describes the record's own unrelated pairs. The relative loss is measured against the "
    "largest recurrence-attributable mass this sweep attains and is not an "
    "absolute miss rate; no absolute miss rate is published, because none is identifiable from "
    "a record with neither labels nor replicates. Neither rate is a statement about whether the "
    "configurations a radius groups are physically the same thing: that depends on the "
    "signature, on the null, and on what 'the same' was taken to mean, and none of those three "
    "is decided here.")

SPLIT_RATE_NOTE = (
    "An absolute split rate -- the share of same-configuration pairs a radius separates -- is "
    "not identifiable from this data. Estimating it needs the mixture fraction on its own, and "
    "what the excess curve supplies is the product of the fraction with the same-configuration "
    "distribution at the radii it is read at. Where that distribution has not saturated the "
    "product is smaller than the fraction, the implied split rate is correspondingly optimistic, "
    "and a measurement on a planted synthetic record put it at 1.91% against a true 12.53%. The "
    "relative loss published instead says what it is relative to.")

RECURRENCE_NOTE = (
    "The recurrence fraction is the proportion of observed signature pairs the mixture model "
    "attributes to a recurring configuration rather than to chance. It is read at the radii "
    "where the excess clears the significance band, where the same-configuration component has "
    "saturated and the unrelated one has not yet risen, so it is a property of the model as "
    "much as of the record; the spread across those radii is published for exactly that reason. "
    "Where the excess clears the band at a single radius alone there is nothing to check the "
    "estimate against, and it is returned unstable rather than read off one point.")


# --------------------------------------------------------------------------------------------
# the null's identity
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureNull:
    """What a null population of signatures preserves and what it destroys.

    An excess over one null is not an excess over another, so this travels in every record.
    The strings are prose because they are read by a person deciding whether to believe a
    number, and no enumeration of them would survive the next null anyone invents.
    """

    name: str
    preserves: Tuple[str, ...]
    destroys: Tuple[str, ...]
    caveat: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise InvalidParameterError(
                "SignatureNull.name", self.name,
                "a name for the null. A p-value against an unnamed null cannot be interpreted")
        if not self.preserves or not self.destroys:
            raise InvalidParameterError(
                "SignatureNull", (self.preserves, self.destroys),
                "at least one thing preserved and one thing destroyed. A null that destroys "
                "nothing is the data, and a null that preserves nothing is noise")

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": self.name,
            "preserves": list(self.preserves),
            "destroys": list(self.destroys),
        }
        if self.caveat is not None:
            record["caveat"] = self.caveat
        return record


#: The honest null: re-run the whole pipeline over phase-randomised records. Distances must be
#: supplied by the caller because generating them means running the pipeline, not the estimator.
SURROGATE_RECORD_NULL = SignatureNull(
    name="surrogate_record",
    preserves=(
        "the record's full 3D power spectrum, hence its spatial spectrum at every scale",
        "the record's temporal autocorrelation (rule R12)",
        "the whole detection, tracking and signing pipeline, applied identically",
    ),
    destroys=(
        "phase organisation, and with it any coherent feature that could recur",
    ))

#: The cheap null, provided because it is the one a caller will reach for, and named for its
#: failure exactly as `surrogate_null.per_frame_phase` is.
ATTRIBUTE_PERMUTATION_NULL = SignatureNull(
    name="attribute_permutation",
    preserves=(
        "the marginal distribution of every signature component across the population",
        "the signature family, so distances remain between comparable quantities",
    ),
    destroys=(
        "the joint structure within one signature, and with it any recurrence",
    ),
    caveat=(
        "ANTI-CONSERVATIVE. Real signatures are constrained -- three inter-node distances must "
        "make a triangle, strengths and scales co-vary -- so they occupy a lower-dimensional "
        "surface than their marginals allow. Permuting components independently scatters the "
        "null off that surface and pushes its pairs further apart than genuinely unrelated real "
        "pairs are, which inflates the excess. Use it to see the shape of the calculation, and "
        "to demonstrate that the answer depends on the null; do not calibrate a published "
        "radius against it."))


# --------------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------------


def _finite_distances(values: Sequence[float], name: str) -> Tuple[float, ...]:
    rows = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise InvalidParameterError(name, value, "finite non-negative distances")
        rows.append(number)
    if not rows:
        raise InvalidParameterError(name, 0, "at least one distance")
    return tuple(rows)


def pair_at(n_points: int, index: int) -> Tuple[int, int]:
    """The `index`-th pair of `itertools.combinations(range(n_points), 2)`, computed directly.

    Materialising the pair list is what this exists to avoid: the acquired record's 69,580
    signatures make 2.42 billion pairs, and a sample of six thousand of them should not cost
    twenty gigabytes to draw. The closed form inverts the triangular number by solving the
    quadratic, and the square root is taken in floating point -- so it is checked by round-trip
    at record scale rather than trusted, because a formula that is off by one at some index
    would silently pair the wrong signatures rather than fail.
    """
    n = int(n_points)
    total = n * (n - 1) // 2
    position = int(index)
    if n < 2:
        raise InvalidParameterError("n_points", n_points, "at least two points to form a pair")
    if not 0 <= position < total:
        raise InvalidParameterError(
            "index", index, "a pair index in [0, %d) for %d points" % (total, n))
    left = int(n - 2 - math.floor(
        math.sqrt(-8.0 * position + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
    right = int(position + left + 1 - total + (n - left) * ((n - left) - 1) // 2)
    return left, right


def signature_distances(signatures: Sequence[Any], *, metric: SignatureMetric,
                        max_pairs: Optional[int] = None,
                        seed: Optional[int] = None) -> Tuple[float, ...]:
    """Pairwise distances within one population, sampled when the population is large.

    The pair count is quadratic and every distance minimises over the node permutations, so a
    population of a few thousand is not enumerable in practice. When `max_pairs` is given the
    pairs are drawn **uniformly at random without replacement from the full pair index space**,
    which keeps the sampled distances an unbiased sample of the population's distance
    distribution; taking the first `max_pairs` pairs in index order would not, because index
    order follows observation order and observation order is time.

    A seed is required whenever sampling happens. A distribution nobody can redraw is a
    distribution nobody can check.
    """
    points: Tuple[SignaturePoint, ...] = (
        tuple(signatures) if signatures and isinstance(signatures[0], SignaturePoint)
        else _points(signatures))
    if len(points) < 2:
        raise InvalidParameterError(
            "signatures", len(points), "at least two signatures to form one pair")
    total = len(points) * (len(points) - 1) // 2
    if max_pairs is None or total <= int(max_pairs):
        return tuple(metric.distance(left, right)
                     for left, right in itertools.combinations(points, 2))

    limit = int(max_pairs)
    if limit < 1:
        raise InvalidParameterError("max_pairs", max_pairs, "at least one pair, or None")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidParameterError(
            "seed", seed,
            "a declared integer seed whenever pairs are sampled. Sampling %d of %d pairs "
            "without a seed produces a distance distribution nobody can redraw" % (limit, total))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(total, size=limit, replace=False)
    rows = []
    for index in chosen:
        left, right = pair_at(len(points), int(index))
        rows.append(metric.distance(points[left], points[right]))
    return tuple(rows)


def permuted_signature_population(signatures: Sequence[Any], *,
                                  seed: int) -> Tuple[SignaturePoint, ...]:
    """One draw from `ATTRIBUTE_PERMUTATION_NULL`: every component shuffled independently.

    Read that null's caveat before using this for anything but a demonstration. It is here
    because it is computable without re-running the pipeline, and because a calibration that
    gives the same answer against this null and against a surrogate-record null is a
    calibration that is not actually using the null.
    """
    points = (tuple(signatures) if signatures and isinstance(signatures[0], SignaturePoint)
              else _points(signatures))
    if len(points) < 2:
        raise InvalidParameterError(
            "signatures", len(points), "at least two signatures to permute between")
    family = points[0].family
    if any(point.family != family for point in points[1:]):
        raise InvalidParameterError(
            "signatures", [point.family.describe() for point in points],
            "one signature family. Permuting a component across families would exchange "
            "measurements of different quantities")
    rng = np.random.default_rng(int(seed))
    blocks: Dict[str, List[List[float]]] = {}
    for name in ("geometry", "bearings", "strengths", "scales"):
        column_count = len(points[0].blocks()[name])
        columns = []
        for index in range(column_count):
            values = [point.blocks()[name][index] for point in points]
            order = rng.permutation(len(values))
            columns.append([values[position] for position in order])
        blocks[name] = columns

    def column(name: str, row: int) -> Tuple[float, ...]:
        return tuple(values[row] for values in blocks[name])

    return tuple(
        SignaturePoint(
            family=family, geometry=column("geometry", row),
            bearings=(None if not family.has_bearings else column("bearings", row)),
            strengths=column("strengths", row), scales=column("scales", row))
        for row in range(len(points)))


# --------------------------------------------------------------------------------------------
# the ensemble
# --------------------------------------------------------------------------------------------


def _ecdf(sorted_values: np.ndarray, radius: float) -> float:
    return float(np.searchsorted(sorted_values, radius, side="right")) / float(
        sorted_values.size)


@dataclass(frozen=True)
class DistanceEnsemble:
    """Observed pair distances, and the same statistic on each member of a null ensemble.

    The null members are kept **separate** rather than pooled. Pooling would give a smoother
    null CDF and no way to say whether the observed excess is larger than the ensemble's own
    spread, which is the only thing that makes the departure testable.
    """

    observed: Tuple[float, ...]
    null_members: Tuple[Tuple[float, ...], ...]
    null: SignatureNull
    metric_digest: Optional[str] = None
    #: How many signatures each population was formed from, before any pair sampling. Optional
    #: because a caller may hold only distances -- but supplying it is what lets this class
    #: notice that a null destroyed the detections rather than only their recurrence. Pair
    #: counts cannot notice it: sampling caps both populations at the same number.
    n_observed_signatures: Optional[int] = None
    n_null_signatures: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.null, SignatureNull):
            raise InvalidParameterError(
                "DistanceEnsemble.null", type(self.null).__name__,
                "a SignatureNull declaring what the null preserves and destroys")
        if not self.null_members:
            raise InvalidParameterError(
                "DistanceEnsemble.null_members", 0,
                "at least one null population. An excess is a comparison, and there is nothing "
                "here to compare against")

    @property
    def n_null_members(self) -> int:
        return len(self.null_members)

    def p_value_floor(self) -> float:
        return 1.0 / (1.0 + self.n_null_members)

    def warnings(self) -> Tuple[str, ...]:
        """Everything about this ensemble that would make a reader distrust its numbers."""
        rows: List[str] = []
        sizes = sorted(len(member) for member in self.null_members)
        median = sizes[len(sizes) // 2]
        if median < 0.1 * len(self.observed):
            rows.append(
                "the median null population has %d pairs against %d observed. A null that "
                "produces far fewer signatures than the record is a noisier estimate of "
                "F_null, and it may also mean the null destroyed the detections themselves "
                "rather than only their recurrence" % (median, len(self.observed)))
        if self.n_observed_signatures is not None and self.n_null_signatures:
            counted = sorted(int(value) for value in self.n_null_signatures)
            middle = counted[len(counted) // 2]
            if middle < 0.5 * int(self.n_observed_signatures):
                rows.append(
                    "the record produced %d signatures and the median null member produced "
                    "%d, a factor of %.1f. The null was meant to destroy recurrence and keep "
                    "the features; producing far fewer of them means it also removed the "
                    "structure the features are found in, so the excess is partly a comparison "
                    "between a record with features and one without. Pair counts cannot show "
                    "this when the pairs were sampled to a common size"
                    % (int(self.n_observed_signatures), middle,
                       int(self.n_observed_signatures) / max(1.0, float(middle))))
        if self.n_null_members < 19:
            rows.append(
                "%d null members floor any p-value at %.3g, which is coarser than the 0.05 "
                "this programme corrects at" % (self.n_null_members, self.p_value_floor()))
        if self.null.caveat is not None:
            rows.append("the declared null carries a caveat: %s" % self.null.caveat)
        return tuple(rows)

    def describe(self) -> Dict[str, Any]:
        return {
            "n_observed_pairs": len(self.observed),
            "n_null_members": self.n_null_members,
            "null_pairs_per_member": [len(member) for member in self.null_members],
            "n_observed_signatures": self.n_observed_signatures,
            "n_null_signatures": list(self.n_null_signatures),
            "p_value_floor": self.p_value_floor(),
            "null": self.null.describe(),
            "metric_digest": self.metric_digest,
            "warnings": list(self.warnings()),
        }


def distance_ensemble(observed: Sequence[float],
                      null_members: Sequence[Sequence[float]], *,
                      null: SignatureNull,
                      metric_digest: Optional[str] = None,
                      n_observed_signatures: Optional[int] = None,
                      n_null_signatures: Sequence[int] = ()) -> DistanceEnsemble:
    """Validate and freeze the two distance populations the calibration reads.

    `n_observed_signatures` and `n_null_signatures` are the population sizes the distances came
    from. They are optional and they are worth supplying: on the acquired record the null
    produced a quarter of the record's signatures, and no pair count could reveal that because
    both populations had been sampled to the same number of pairs.
    """
    members = tuple(_finite_distances(member, "null_members[%d]" % index)
                    for index, member in enumerate(null_members))
    counts = tuple(int(value) for value in n_null_signatures)
    if counts and len(counts) != len(members):
        raise InvalidParameterError(
            "n_null_signatures", len(counts),
            "one signature count per null member (%d), or none at all" % len(members))
    return DistanceEnsemble(
        observed=_finite_distances(observed, "observed"), null_members=members,
        null=null, metric_digest=metric_digest,
        n_observed_signatures=(None if n_observed_signatures is None
                               else int(n_observed_signatures)),
        n_null_signatures=counts)


# --------------------------------------------------------------------------------------------
# the excess curve
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcessPoint:
    """At one radius: how many pairs the record admits, how many the null does, and the gap."""

    radius: float
    observed_fraction: float
    null_fraction: float
    excess: float
    null_member_fractions: Tuple[float, ...]
    p_value: float

    def describe(self) -> Dict[str, Any]:
        return {
            "radius": self.radius,
            "observed_fraction": self.observed_fraction,
            "null_fraction": self.null_fraction,
            "excess": self.excess,
            "p_value": self.p_value,
        }


def grid_resolution(n_observed_pairs: int, n_radii: int) -> float:
    """The finest relative loss a geometric grid of `n_radii` points can resolve.

    The grid's quantiles are in geometric progression from ``1 / n_observed_pairs`` to 1, so
    consecutive points differ by a fixed *ratio*, and no radius exists between them. A relative
    loss smaller than ``ratio - 1`` therefore cannot be distinguished from the next grid point
    down, and asking a calibration to hold a target finer than this is asking a question the
    sweep cannot answer -- the same trap `multiple_comparisons.resolution_limit` names for
    p-values, in a different coordinate.
    """
    pairs = int(n_observed_pairs)
    if pairs < 2 or int(n_radii) < 2:
        raise InvalidParameterError(
            "grid_resolution", (n_observed_pairs, n_radii),
            "at least two pairs and at least two radii")
    return math.exp(math.log(float(pairs)) / (int(n_radii) - 1)) - 1.0


def radii_for_resolution(n_observed_pairs: int, resolution: float) -> int:
    """The smallest `n_radii` whose grid resolves a relative loss of `resolution`."""
    if not math.isfinite(float(resolution)) or not 0.0 < float(resolution) < 1.0:
        raise InvalidParameterError(
            "resolution", resolution, "a relative loss strictly between 0 and 1 to resolve")
    pairs = int(n_observed_pairs)
    if pairs < 2:
        raise InvalidParameterError("n_observed_pairs", n_observed_pairs, "at least two pairs")
    return max(2, int(math.ceil(1.0 + math.log(float(pairs))
                                / math.log(1.0 + float(resolution)))))


def radius_grid(ensemble: DistanceEnsemble, n_radii: int = 64) -> Tuple[float, ...]:
    """A sweep over the observed distances, spaced **geometrically in quantile**.

    The grid runs from the single closest observed pair to the farthest, with its quantiles in
    geometric rather than arithmetic progression: half the points sit below the record's 1st
    percentile of distance.

    That is not a cosmetic choice. Everything this module estimates lives in the close-pair
    tail, because that is the only place the two mixture components are separated, and a grid
    spaced evenly in quantile puts its first point at ``1/n_radii`` -- above the entire
    same-configuration mass whenever recurrence accounts for less than about 1.5% of pairs. A
    linear grid was the first implementation here and it could not see a planted grouping that
    the geometric grid recovers exactly.
    """
    if n_radii < 2:
        raise InvalidParameterError(
            "n_radii", n_radii, "at least two radii. A curve through one point is a point")
    ordered = np.sort(np.asarray(ensemble.observed, dtype=np.float64))
    floor = 1.0 / float(ordered.size)
    fractions = np.exp(np.linspace(math.log(floor), 0.0, int(n_radii)))
    indices = np.clip((fractions * ordered.size).astype(int) - 1, 0, ordered.size - 1)
    return tuple(sorted(set(float(ordered[index]) for index in indices)))


def excess_curve(ensemble: DistanceEnsemble,
                 radii: Optional[Sequence[float]] = None,
                 n_radii: int = 64) -> Tuple[ExcessPoint, ...]:
    """The observed and null close-pair fractions across a sweep of radii.

    The p-value at each radius is the ensemble rank statistic this programme uses everywhere
    else: ``(1 + k) / (1 + n)`` with ``k`` the number of null members whose own close-pair
    fraction reaches the observed one. It is never ``k / n``, which no finite ensemble supports.

    **These per-radius p-values are descriptive and are never corrected into a decision.**
    See `separation_test` for why: correcting a sweep this size against a surrogate ensemble is
    arithmetically incapable of rejecting anything.
    """
    grid = (radius_grid(ensemble, n_radii) if radii is None
            else tuple(sorted(float(value) for value in radii)))
    if not grid:
        raise InvalidParameterError("radii", radii, "at least one radius to evaluate")
    observed_sorted = np.sort(np.asarray(ensemble.observed, dtype=np.float64))
    member_sorted = [np.sort(np.asarray(member, dtype=np.float64))
                     for member in ensemble.null_members]
    rows = []
    for radius in grid:
        observed_fraction = _ecdf(observed_sorted, radius)
        member_fractions = tuple(_ecdf(member, radius) for member in member_sorted)
        null_fraction = float(np.mean(member_fractions))
        k = sum(1 for value in member_fractions if value >= observed_fraction)
        rows.append(ExcessPoint(
            radius=float(radius), observed_fraction=observed_fraction,
            null_fraction=null_fraction, excess=observed_fraction - null_fraction,
            null_member_fractions=member_fractions,
            p_value=(1.0 + k) / (1.0 + len(member_fractions))))
    return tuple(rows)


# --------------------------------------------------------------------------------------------
# does the record depart from the null at all?
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeparationTest:
    """One test, over the whole sweep, of whether the record has more close pairs than chance.

    The statistic is the **largest excess anywhere on the curve**, and its null distribution
    comes from the ensemble itself by leave-one-out: each null member is scored against the mean
    of the others, exactly as the record is scored against the mean of all of them. Taking the
    maximum is what turns a sweep of radii into a single test, and the resulting critical value
    is a **simultaneous** band -- an excess above it at any radius is significant at `alpha`
    across the whole sweep at once, with no correction owed.
    """

    statistic: float
    radius_at_maximum: float
    null_statistics: Tuple[float, ...]
    p_value: float
    p_value_floor: float
    critical_value: float
    alpha: float
    significant: bool
    surrogates_a_corrected_sweep_would_need: int

    def describe(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "statistic_is": "the largest excess of observed over null close-pair fraction",
            "radius_at_maximum": self.radius_at_maximum,
            "p_value": self.p_value,
            "p_value_floor": self.p_value_floor,
            "critical_value": self.critical_value,
            "critical_value_is": (
                "the %.3g quantile of the leave-one-out null statistics; an excess above it is "
                "significant simultaneously across every radius swept" % (1.0 - self.alpha,)),
            "alpha": self.alpha,
            "n_null_members": len(self.null_statistics),
            "significant": self.significant,
            "why_not_a_corrected_sweep": (
                "Testing each radius separately and correcting would need about %d surrogates "
                "to be capable of a rejection at all, because a surrogate p-value floors at "
                "1/(1+n) and the correction divides the level by the family size. A study that "
                "cannot reject reports nothing and looks like a clean negative result. The "
                "maximum statistic answers the same question with the ensemble in hand."
                % self.surrogates_a_corrected_sweep_would_need),
        }


def separation_test(ensemble: DistanceEnsemble,
                    points: Sequence[ExcessPoint], *,
                    alpha: float = 0.05) -> SeparationTest:
    """Test the whole excess curve at once, with the ensemble supplying its own null.

    **Why not the programme's usual Benjamini-Yekutieli correction.** A surrogate p-value cannot
    go below ``1/(1+n)``. Correcting a sweep of 64 radii under BY needs a raw p-value near
    ``alpha / (64 * H_64)``, which is about ``1.6e-4`` -- some six thousand surrogates, each of
    which here means re-running the tracking and signing pipeline over a whole surrogate record.
    A corrected per-radius sweep against any affordable ensemble is therefore incapable of
    rejecting anything, and would return a clean-looking negative result that is indistinguishable
    from a real one. That is precisely the trap `multiple_comparisons.required_surrogates` exists
    to name, and this function reports the number it gives.
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
            or not 0.0 < float(alpha) < 1.0:
        raise InvalidParameterError("alpha", alpha, "a significance level in (0, 1)")
    if len(ensemble.null_members) < 2:
        raise InvalidParameterError(
            "ensemble.null_members", len(ensemble.null_members),
            "at least two null members. The null distribution of the maximum excess is built "
            "by scoring each member against the mean of the others, and one member has no "
            "others to be scored against")
    if not points:
        raise InvalidParameterError("points", 0, "at least one radius on the excess curve")

    radii = [row.radius for row in points]
    fractions = np.asarray([row.null_member_fractions for row in points], dtype=np.float64)
    n_members = fractions.shape[1]
    totals = fractions.sum(axis=1)
    # Leave-one-out mean of the other members, at every radius, for every member.
    others = (totals[:, None] - fractions) / float(n_members - 1)
    null_statistics = tuple(float(value)
                            for value in np.max(fractions - others, axis=0))

    observed = np.asarray([row.excess for row in points], dtype=np.float64)
    statistic = float(np.max(observed))
    at_maximum = float(radii[int(np.argmax(observed))])
    k = sum(1 for value in null_statistics if value >= statistic)
    p_value = (1.0 + k) / (1.0 + n_members)
    critical = float(np.quantile(np.asarray(null_statistics, dtype=np.float64),
                                 1.0 - float(alpha), method="higher"))
    return SeparationTest(
        statistic=statistic, radius_at_maximum=at_maximum, null_statistics=null_statistics,
        p_value=p_value, p_value_floor=ensemble.p_value_floor(), critical_value=critical,
        alpha=float(alpha), significant=p_value <= float(alpha),
        surrogates_a_corrected_sweep_would_need=required_surrogates(
            len(points), float(alpha), "benjamini_yekutieli"))


# --------------------------------------------------------------------------------------------
# the mixture fraction
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecurrenceFraction:
    """The estimated share of observed pairs that recurrence explains, and its sensitivity."""

    value: float
    read_at: Tuple[Tuple[float, float, float], ...]
    maximum_excess: float
    spread: float
    tolerance: float
    critical_value: float
    null_fraction_cap: float
    stable: bool
    reason: str

    def describe(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "maximum_excess": self.maximum_excess,
            "read_at": [{"radius": r, "null_fraction": f, "estimate": v}
                        for r, f, v in self.read_at],
            "n_radii_read": len(self.read_at),
            "critical_value": self.critical_value,
            "null_fraction_cap": self.null_fraction_cap,
            "spread": self.spread,
            "stability_tolerance": self.tolerance,
            "stable": self.stable,
            "reason": self.reason,
            "basis": RECURRENCE_NOTE,
        }


def estimate_recurrence_fraction(
        points: Sequence[ExcessPoint], *,
        critical_value: float,
        null_fraction_cap: float = DEFAULT_NULL_FRACTION_CAP,
        tolerance: float = DEFAULT_STABILITY_TOLERANCE) -> RecurrenceFraction:
    """``pi = excess(r) / (1 - F_null(r))``, read wherever the excess clears the band.

    Where the same-configuration component has saturated and the unrelated one has not yet
    risen, ``excess(r) = pi * (1 - F_null(r))`` and that ratio is ``pi`` at every such radius.
    The region is the radii whose excess exceeds `critical_value` -- the simultaneous band from
    `separation_test`, so exactly the radii at which the record demonstrably differs from the
    null -- and whose null fraction stays at or below `null_fraction_cap`.

    **The cap is what keeps this an estimate rather than noise.** Without it the read runs up to
    ``F_null = 1 - critical_value / pi``, which for a strong recurring component is most of the
    way into the bulk. A record containing ``C`` distinct configurations supplies only ``C``
    draws of the unrelated-pair distribution however many signatures it has, so ``F_obs`` there
    departs from ``F_null`` by order ``1 / sqrt(C)`` while the fraction being estimated is of
    order ``1 / C``. On the synthetic record in `VERIFICATION.md` an uncapped read turned a true
    fraction of 0.196 into estimates spread over 0.96 -- five times the quantity itself.

    The reported value is the median of the estimates and the spread is their full range.
    **One qualifying radius is not enough**: with nothing to compare the estimate against, its
    stability is not unknown but unmeasured, and it is returned unstable for that reason rather
    than reported as though a single point had agreed with itself.
    """
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise InvalidParameterError(
            "tolerance", tolerance,
            "a positive spread the fraction may move across the qualifying radii before "
            "refusing")
    if not math.isfinite(float(critical_value)):
        raise InvalidParameterError(
            "critical_value", critical_value,
            "the finite simultaneous band from separation_test")
    if not math.isfinite(float(null_fraction_cap)) or not 0.0 < float(null_fraction_cap) < 1.0:
        raise InvalidParameterError(
            "null_fraction_cap", null_fraction_cap,
            "a null fraction strictly in (0, 1) beyond which the mixture is read in the bulk "
            "and the estimate is dominated by how many distinct configurations the record has")
    rows = tuple(points)
    if not rows:
        raise InvalidParameterError("points", 0, "at least one radius on the excess curve")

    maximum = max(row.excess for row in rows)
    qualifying = [(row.radius, row.null_fraction, row.excess / (1.0 - row.null_fraction))
                  for row in rows
                  if row.excess > float(critical_value)
                  and row.null_fraction <= float(null_fraction_cap)]
    if not qualifying:
        return RecurrenceFraction(
            value=maximum, read_at=(), maximum_excess=maximum, spread=0.0,
            tolerance=float(tolerance), critical_value=float(critical_value),
            null_fraction_cap=float(null_fraction_cap), stable=False,
            reason=("No radius both has an excess above the band of %.4f and keeps the null "
                    "fraction at or below the cap of %.2f, so there is nowhere on this curve "
                    "where the record demonstrably differs from the null while the mixture is "
                    "still conditioned. The largest excess anywhere is %.4f."
                    % (float(critical_value), float(null_fraction_cap), maximum)))
    values = sorted(value for _r, _f, value in qualifying)
    median = (values[len(values) // 2] if len(values) % 2 == 1
              else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2]))
    if len(qualifying) < 2:
        return RecurrenceFraction(
            value=median, read_at=tuple(qualifying), maximum_excess=maximum, spread=0.0,
            tolerance=float(tolerance), critical_value=float(critical_value),
            null_fraction_cap=float(null_fraction_cap), stable=False,
            reason=("The excess clears the band of %.4f at exactly one radius of the %d "
                    "swept, below a null fraction of %.2f. A fraction read at a single point has "
                    "nothing to be checked against, so its stability is unmeasured rather than "
                    "perfect." % (float(critical_value), len(rows), float(null_fraction_cap))))
    spread = values[-1] - values[0]
    stable = spread <= float(tolerance)
    return RecurrenceFraction(
        value=median, read_at=tuple(qualifying), maximum_excess=maximum, spread=spread,
        tolerance=float(tolerance), critical_value=float(critical_value),
        null_fraction_cap=float(null_fraction_cap), stable=stable,
        reason=("%d radii of %d clear the band of %.4f below a null fraction of %.2f, and the "
                "fraction they give spans %.4f, %s the tolerance of %.4f."
                % (len(qualifying), len(rows), float(critical_value),
                   float(null_fraction_cap), spread,
                   "within" if stable else "above", float(tolerance))))


# --------------------------------------------------------------------------------------------
# the calibration
# --------------------------------------------------------------------------------------------




@dataclass(frozen=True)
class RadiusRates:
    """Both rates at one radius, with the measured one and the estimated one kept apart."""

    radius: float
    observed_fraction: float
    admission_rate: float
    contamination_rate: float
    relative_loss: float
    loss_reference: str
    excess: float
    p_value: float
    above_band: bool

    @property
    def worst_rate(self) -> float:
        return max(self.contamination_rate, self.relative_loss)

    def describe(self) -> Dict[str, Any]:
        return {
            "radius": self.radius,
            "observed_fraction": self.observed_fraction,
            "admission_rate": self.admission_rate,
            "admission_rate_is": (
                "measured on the declared null population: the share of unrelated pairs this "
                "radius admits"),
            "contamination_rate": self.contamination_rate,
            "contamination_rate_is": (
                "an ESTIMATE of the share of the pairs this radius admits that chance explains, "
                "not a bound: it substitutes the null's close-pair fraction for the record's "
                "own unrelated one, and the two differ by however well the null describes the "
                "record"),
            "relative_loss": self.relative_loss,
            "relative_loss_is": (
                "the share of the recurrence-attributable close-pair mass this radius does not "
                "reach, measured against %s. It is NOT an absolute miss rate: it says nothing "
                "about same-configuration pairs lying beyond every radius swept, and no "
                "absolute miss rate is identifiable from a record without labels or replicates"
                % self.loss_reference),
            "worst_rate": self.worst_rate,
            "excess": self.excess,
            "descriptive_p_value": self.p_value,
            "above_simultaneous_band": self.above_band,
        }


@dataclass(frozen=True)
class NullCalibration:
    """A radius chosen against a null, or a named refusal to choose one."""

    status: str
    radius: Optional[float]
    rates: Optional[RadiusRates]
    separation: SeparationTest
    recurrence: Optional[RecurrenceFraction]
    curve: Tuple[RadiusRates, ...]
    ensemble: DistanceEnsemble
    target_rate: float
    grid_resolution: float
    best_achievable: Optional[RadiusRates]
    note: str

    @property
    def measured(self) -> bool:
        return self.status == CALIBRATION_MEASURED

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "schema": NULL_CALIBRATION_SCHEMA,
            "status": self.status,
            "radius": self.radius,
            "target_rate": self.target_rate,
            "n_radii_swept": len(self.curve),
            "grid_resolution": self.grid_resolution,
            "grid_resolution_is": (
                "the finest relative loss this sweep can distinguish: consecutive radii are a "
                "fixed ratio apart in quantile and no radius exists between them"),
            "n_radii_above_band": sum(1 for row in self.curve if row.above_band),
            "separation": self.separation.describe(),
            "ensemble": self.ensemble.describe(),
            "claim_boundary": CALIBRATION_CLAIM_BOUNDARY,
            "split_rate": SPLIT_RATE_NOTE,
            # The curve is the evidence. A receipt that states a radius without the sweep it was
            # chosen from asks to be believed rather than checked, and the whole point of the
            # refusals here is that a reader can see which one applied and why.
            "curve": [row.describe() for row in self.curve],
            "note": self.note,
        }
        if self.recurrence is not None:
            record["recurrence_fraction"] = self.recurrence.describe()
        if self.rates is not None:
            record["rates"] = self.rates.describe()
        if self.best_achievable is not None:
            record["best_achievable"] = self.best_achievable.describe()
        return record


def _monotone(values: Sequence[float]) -> Tuple[float, ...]:
    """Enforce the one property an estimated CDF must have and the estimator does not give.

    ``F_same`` is a distribution function, so it cannot decrease; the plug-in estimate can,
    because ``excess`` is a difference of two noisy ECDFs. A running maximum is the smallest
    correction that restores monotonicity, and it is applied to the estimate rather than hidden
    inside the rate, so the record shows what was estimated and what was imposed.
    """
    best = 0.0
    rows = []
    for value in values:
        best = max(best, float(value))
        rows.append(min(1.0, best))
    return tuple(rows)


def calibrate_against_null(
        ensemble: DistanceEnsemble, *,
        target_rate: float,
        alpha: float = 0.05,
        n_radii: Optional[int] = None,
        null_fraction_cap: float = DEFAULT_NULL_FRACTION_CAP,
        stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE) -> NullCalibration:
    """Choose a clustering radius from a record and a null, or refuse and say why.

    `target_rate` is the level both rates must hold, and it is required rather than defaulted:
    it is the scientific declaration in this call, and a level the software chose is a level
    nobody declared.

    The returned radius is the **smallest** one that holds both rates and whose excess clears the
    simultaneous band. Smallest, not best-separating: a larger radius admits everything a smaller
    one does, so among the radii that qualify, the smallest makes the weakest claim.

    `n_radii` defaults to whatever resolves `target_rate` on this many observed pairs, because
    the grid's quantile step is a floor on the relative loss it can distinguish: a sweep of 64
    points over 4,000 pairs steps by 13.9%, so a target of 10% is finer than any radius on it
    and the calibration refuses for a reason that has nothing to do with the record. Passing a
    grid too coarse for the target is refused outright rather than silently answered.
    """
    if not isinstance(ensemble, DistanceEnsemble):
        raise InvalidParameterError(
            "ensemble", type(ensemble).__name__,
            "a DistanceEnsemble carrying the observed distances and a declared null")
    if not isinstance(target_rate, (int, float)) or isinstance(target_rate, bool) \
            or not 0.0 < float(target_rate) < 1.0:
        raise InvalidParameterError(
            "target_rate", target_rate,
            "a declared rate strictly between 0 and 1 that both error rates must hold")

    pairs = len(ensemble.observed)
    needed = radii_for_resolution(pairs, float(target_rate))
    if n_radii is None:
        resolved = needed
    else:
        resolved = int(n_radii)
        if grid_resolution(pairs, resolved) > float(target_rate):
            raise InvalidParameterError(
                "n_radii", n_radii,
                "at least %d radii to hold a target rate of %.4f over %d observed pairs. A "
                "geometric grid of %d resolves no relative loss finer than %.4f, so the target "
                "asked for is finer than the sweep can distinguish and the calibration would "
                "refuse for a reason that is about the grid rather than about the record"
                % (needed, float(target_rate), pairs, resolved,
                   grid_resolution(pairs, resolved)))

    points = excess_curve(ensemble, n_radii=resolved)
    separation = separation_test(ensemble, points, alpha=alpha)

    def build(fraction: Optional[float]) -> Tuple[RadiusRates, ...]:
        """Rates at every radius. `fraction` is the mixture estimate, or None before one exists.

        Before a fraction has been estimated the contamination estimate cannot be tightened by
        ``1 - pi``, so it is computed at ``pi = 0``, which is its largest value. That keeps a
        curve available on every refusal path without any path reporting a rate that a later
        estimate would have made worse.

        The loss is measured against the estimated fraction rather than against the largest
        excess on the curve. The largest excess is the maximum of a noisy difference and is
        biased upwards by whatever the noise reaches: on the synthetic record it was 0.0464
        against a fraction of 0.0377, and every loss measured against it inherited that 23%,
        which was enough to refuse a radius whose contamination was zero.
        """
        captured = _monotone([row.excess for row in points])
        reachable = (max(captured) if fraction is None else float(fraction)) if captured else 0.0
        reference = ("the largest excess anywhere on the sweep, no mixture fraction having been "
                     "estimated on this path" if fraction is None
                     else "the estimated recurrence fraction, the largest such mass the sweep "
                          "reliably attains")
        share = 1.0 - (0.0 if fraction is None else float(fraction))
        return tuple(
            RadiusRates(
                radius=row.radius, observed_fraction=row.observed_fraction,
                admission_rate=row.null_fraction,
                contamination_rate=(
                    0.0 if row.observed_fraction <= 0.0
                    else min(1.0, row.null_fraction * share / row.observed_fraction)),
                relative_loss=(1.0 if reachable <= 0.0
                               else max(0.0, 1.0 - reach / reachable)),
                loss_reference=reference,
                excess=row.excess, p_value=row.p_value,
                above_band=row.excess > separation.critical_value)
            for row, reach in zip(points, captured))

    def refuse(status: str, note: str, recurrence: Optional[RecurrenceFraction] = None,
               curve: Tuple[RadiusRates, ...] = (),
               best: Optional[RadiusRates] = None) -> NullCalibration:
        return NullCalibration(
            status=status, radius=None, rates=None, separation=separation,
            recurrence=recurrence, curve=curve, ensemble=ensemble,
            target_rate=float(target_rate),
            grid_resolution=grid_resolution(pairs, resolved),
            best_achievable=best, note=note)

    unsplit = build(None)
    if not separation.significant:
        return refuse(
            CALIBRATION_NO_SEPARATION,
            "The largest excess of close pairs anywhere on the sweep is %.4f at radius %.6f, "
            "which %d of %d null members reach or beat: p = %.3g against alpha = %.3g, with a "
            "floor of %.3g. The record's signature pairs are no closer together than the '%s' "
            "null explains, so there is no radius to choose. This is a finding about the "
            "signature and the null, not a failure of the search."
            % (separation.statistic, separation.radius_at_maximum,
               sum(1 for value in separation.null_statistics
                   if value >= separation.statistic),
               len(separation.null_statistics), separation.p_value, separation.alpha,
               separation.p_value_floor, ensemble.null.name),
            curve=unsplit)

    recurrence = estimate_recurrence_fraction(
        points, critical_value=separation.critical_value,
        null_fraction_cap=null_fraction_cap, tolerance=stability_tolerance)
    if not recurrence.stable:
        return refuse(
            CALIBRATION_UNSTABLE,
            "The mixture fraction cannot be read stably off the excess curve. %s Every rate "
            "below it would inherit that arbitrariness, which is the property T4E.6 found in "
            "the replicate radius and this task exists to avoid. Every reading is published "
            "so the disagreement can be seen rather than averaged away." % recurrence.reason,
            recurrence=recurrence, curve=unsplit)

    curve = build(recurrence.value)

    eligible = [row for row in curve if row.above_band]
    best = min(eligible, key=lambda row: (row.worst_rate, row.radius)) if eligible else None
    qualifying = [row for row in eligible
                  if row.contamination_rate <= float(target_rate)
                  and row.relative_loss <= float(target_rate)]
    if not qualifying:
        return refuse(
            CALIBRATION_NO_OPERATING_POINT,
            "A recurring component of %.1f%% of pairs was estimated, but no radius clearing the "
            "simultaneous band holds both rates at or below %.1f%%. %s"
            % (100.0 * recurrence.value, 100.0 * float(target_rate),
               ("The best any such radius achieves is a worst rate of %.1f%% at radius %.6f. A "
                "radius does exist to be chosen at a laxer level; none exists at the one "
                "declared here." % (100.0 * best.worst_rate, best.radius)) if best is not None
               else ("No radius on the sweep clears the band at all, even though the maximum "
                     "excess does: the maximum sits at a radius the grid resolves only at the "
                     "band's own height.")),
            recurrence=recurrence, curve=curve, best=best)

    chosen = min(qualifying, key=lambda row: row.radius)
    return NullCalibration(
        status=CALIBRATION_MEASURED, radius=chosen.radius, rates=chosen,
        separation=separation, recurrence=recurrence, curve=curve, ensemble=ensemble,
        target_rate=float(target_rate), grid_resolution=grid_resolution(pairs, resolved),
        best_achievable=best,
        note=("Radius %.6f admits %.2f%% of the pairs the '%s' null produces. Of the pairs it "
              "admits from the record, an estimated %.2f%% are chance, and it reaches all but "
              "%.2f%% of the recurrence-attributable close-pair mass this sweep reliably "
              "attains -- both at or below the declared %.1f%%. The excess there is %.4f, above "
              "the simultaneous band of %.4f (whole-sweep p = %.3g), and the recurrence "
              "fraction behind the contamination bound is %.1f%%. No absolute split rate is "
              "published: none is identifiable here."
              % (chosen.radius, 100.0 * chosen.admission_rate, ensemble.null.name,
                 100.0 * chosen.contamination_rate, 100.0 * chosen.relative_loss,
                 100.0 * float(target_rate), chosen.excess, separation.critical_value,
                 separation.p_value, 100.0 * recurrence.value)))


__all__ = [
    "ATTRIBUTE_PERMUTATION_NULL", "CALIBRATION_CLAIM_BOUNDARY", "CALIBRATION_MEASURED",
    "CALIBRATION_NO_OPERATING_POINT", "CALIBRATION_NO_SEPARATION",
    "CALIBRATION_STATUSES", "CALIBRATION_UNSTABLE", "DEFAULT_NULL_FRACTION_CAP",
    "DEFAULT_STABILITY_TOLERANCE", "DistanceEnsemble", "ExcessPoint", "NULL_CALIBRATION_SCHEMA",
    "NullCalibration", "RECURRENCE_NOTE", "RadiusRates", "RecurrenceFraction", "SeparationTest",
    "SPLIT_RATE_NOTE", "SignatureNull", "SURROGATE_RECORD_NULL", "calibrate_against_null",
    "distance_ensemble",
    "excess_curve", "estimate_recurrence_fraction", "permuted_signature_population",
    "grid_resolution", "pair_at", "radii_for_resolution", "radius_grid", "separation_test",
    "signature_distances",
]
