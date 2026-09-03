"""Frozen scale/shape fixtures, and the inference the null can actually support (TG17.11).

**The fixtures could not be borrowed.** TG17.0's four flagship records each declare their
structural scale as their own row cadence, so each resolves 1.00 rows per native cycle and
`shape_recurrence` refuses them before computing anything: a shape needs more than one sample per
cycle to exist. The scale/shape cases are therefore built here, by direct analogy with
`family_calibration.py`'s four calendar cases — one planted, two safeguards, and one degenerate
inventory — with their expected answers written in this module rather than in the test, so a later
slice cannot quietly relax one.

**The family size is derived, not chosen, and it is large.** `reassign_scale_partners` replaces a
pairing's right member with another right member from the same inventory. A member's surrogate
statistic can therefore take only as many values as the inventory has admissible alternative
partners: for `k` disjoint pairings, exactly `k - 1`. That bounds the p-value from below at `1/k`
however many replications are paid for, and Benjamini-Yekutieli then requires `1/k <= alpha/H_k`
before any member can reject. `minimum_resolvable_family` solves that against the real correction
rather than against arithmetic written here, and at alpha 0.05 the answer is **105 pairings**. A
scale/shape family smaller than that cannot reject at the declared alpha no matter how strong the
effect, and no amount of computation changes it.

**The Monte Carlo template is registered here and refused, because it is the trap.** Applying
`family_calibration.py`'s method unchanged — draw 999 whole reassignments, count how many
surrogates reach the observation, divide — produces a p-value of 0.001 for a member that beats its
`k - 1` alternatives, when the exact tail probability of that event is `1/k`. The error is a factor
of `k` and it runs in the *anti-conservative* direction. Measured on a wholly unrelated inventory
of six pairings it rejects at a family-wise rate of **74%** against a nominal 5%, while the exact
partner test rejects at **0%**. Calendar mode is not affected: a clock shift has support far larger
than the replication count, so each replication is a genuinely new surrogate. The distinction is
whether the null's support exceeds the number of draws, which is a property of the null and not of
the code, so `monte_carlo_partner_p_values` is kept, named, and refused — the same treatment
`global_value_shuffle` gets, and for the same reason: an operation this available needs to be a
question already answered rather than one nobody thought about.

Nothing in this module is evidence about the world.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.benchmarks.shape_calibration import ShapeRecord, shape_recurrence
from src.core.errors import InvalidParameterError
from src.core.structural_nulls import NullRefusal, admissible_partners
from src.statistics.multiple_comparisons import adjust


ALPHA = 0.05
CORRECTION = "benjamini_yekutieli"

#: Every fixture record spans this many native cycles at this row density. The density sits well
#: above `shape_calibration.MINIMUM_ROWS_PER_CYCLE`, so no fixture is near the resolution floor and
#: a calibration result cannot be an artefact of coarse sampling.
SPAN_CYCLES = 8.0
ROWS_PER_CYCLE = 20
FIXTURE_ROWS = int(SPAN_CYCLES * ROWS_PER_CYCLE)

#: Native durations are drawn across four orders of magnitude, from a quarter-hour to four months,
#: so every pairing under test spans native scales that no shared clock could align.
NATIVE_SECONDS_RANGE = (9.0e2, 1.0e7)


def minimum_resolvable_family(alpha: float = ALPHA, correction: str = CORRECTION,
                              limit: int = 400) -> int:
    """The smallest family whose exact partner null can reject, solved against the real correction.

    A member of a `k`-pairing family has `k - 1` admissible alternative partners, so its exact
    p-value cannot fall below `1/k`. This puts every member at that floor — the most favourable
    case there is, a perfect planted result — and asks the correction whether it rejects. The
    answer is a property of the null and the correction together, so it is computed rather than
    written down.
    """
    for size in range(2, limit + 1):
        floor = 1.0 / size
        corrected = adjust([floor] * size, method=correction, alpha=alpha, n_tests=size,
                           labels=[str(index) for index in range(size)])
        if all(corrected["rejected"]):
            return size
    raise InvalidParameterError(  # pragma: no cover - 105 is found long before the limit
        "limit", limit, "a search bound large enough to contain a resolvable family size")


FAMILY_SIZE = minimum_resolvable_family()


# ------------------------------------------------------------------------- building records


def _profile(phase: np.ndarray, harmonics: Sequence[Tuple[float, float]]) -> np.ndarray:
    """A smooth periodic profile: the shape a record traces once per native cycle."""
    value = np.zeros_like(phase)
    for order, (amplitude, offset) in enumerate(harmonics):
        value = value + amplitude * np.sin(2.0 * np.pi * (order + 1) * phase + offset)
    return value


def _harmonics(rng: np.random.Generator, count: int = 3) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(rng.uniform(0.5, 1.5)), float(rng.uniform(0.0, 2.0 * np.pi)))
                 for _ in range(count))


def _record(label: str, *, native_seconds: float, harmonics: Sequence[Tuple[float, float]],
            rng: np.random.Generator, noise: float = 0.02,
            grid_artefact: Optional[np.ndarray] = None) -> ShapeRecord:
    """One record, standardized to zero mean and unit variance as every fixture record is.

    The standardization is deliberate and is what `same_normalisation_unrelated` exists to test:
    putting two arbitrary smooth profiles on the same location and spread makes them look alike,
    and a mode that reported that as recurrence would be reporting its own preprocessing.
    """
    edges = np.linspace(0.0, SPAN_CYCLES * native_seconds, FIXTURE_ROWS + 1)
    starts, ends = edges[:-1], edges[1:]
    phase = ((starts + ends) / 2.0) / native_seconds
    values = _profile(phase, harmonics)
    if grid_artefact is not None:
        values = values + grid_artefact
    values = (values - values.mean()) / values.std()
    values = values + noise * rng.standard_normal(FIXTURE_ROWS)
    return ShapeRecord(label=label, starts=starts, ends=ends, values=values,
                       valid=np.ones(FIXTURE_ROWS, dtype=bool), native_seconds=native_seconds)


def _native(rng: np.random.Generator) -> float:
    low, high = NATIVE_SECONDS_RANGE
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# ---------------------------------------------------------------------------- the fixtures


def planted_shape_recurrence(seed: int, size: int = FAMILY_SIZE) \
        -> Tuple[Tuple[ShapeRecord, ShapeRecord], ...]:
    """Each pairing genuinely shares one shape, presented at two unrelated native durations.

    The planted case. Both members of a pairing trace the same profile per native cycle while
    occupying wildly different spans of seconds, which is exactly the claim scale/shape mode
    exists to make and the one a working method must recover.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for index in range(size):
        harmonics = _harmonics(rng)
        pairs.append((_record("planted_left_%d" % index, native_seconds=_native(rng),
                              harmonics=harmonics, rng=rng),
                      _record("planted_right_%d" % index, native_seconds=_native(rng),
                              harmonics=harmonics, rng=rng)))
    return tuple(pairs)


def same_normalisation_unrelated(seed: int, size: int = FAMILY_SIZE) \
        -> Tuple[Tuple[ShapeRecord, ShapeRecord], ...]:
    """Independent shapes put through the identical normalisation, which is what makes them look alike.

    This mode's counterpart to calendar's `same_window_unrelated`. Standardizing two smooth
    profiles to zero mean and unit variance gives them the same location, the same spread and a
    correlation well above zero; nothing about that is recurrence, and a family that rejects here
    is reporting its own preprocessing as a finding.
    """
    rng = np.random.default_rng(seed)
    return tuple(
        (_record("unrelated_left_%d" % index, native_seconds=_native(rng),
                 harmonics=_harmonics(rng), rng=rng),
         _record("unrelated_right_%d" % index, native_seconds=_native(rng),
                 harmonics=_harmonics(rng), rng=rng))
        for index in range(size))


def native_scale_alias(seed: int, size: int = FAMILY_SIZE) \
        -> Tuple[Tuple[ShapeRecord, ShapeRecord], ...]:
    """A coincidence of the sampling grid rather than of shape, this mode's `gap_alias`.

    Every record here is sampled at the same rows per native cycle and carries the same strong
    artefact keyed to position *within* the cycle — the signature of a shared instrument cadence
    rather than of a shared shape. Every pairing therefore correlates highly, and so does every
    reassignment, which is precisely why the null must absorb it. A family that rejects here has
    turned its own sampling grid into a scientific claim.
    """
    rng = np.random.default_rng(seed)
    within_cycle = (np.arange(FIXTURE_ROWS) % ROWS_PER_CYCLE) / float(ROWS_PER_CYCLE)
    artefact = 3.0 * np.sign(np.sin(2.0 * np.pi * 5.0 * within_cycle))
    return tuple(
        (_record("alias_left_%d" % index, native_seconds=_native(rng),
                 harmonics=_harmonics(rng), rng=rng, grid_artefact=artefact),
         _record("alias_right_%d" % index, native_seconds=_native(rng),
                 harmonics=_harmonics(rng), rng=rng, grid_artefact=artefact))
        for index in range(size))


def degenerate_inventory(seed: int, size: int = FAMILY_SIZE) \
        -> Tuple[Tuple[ShapeRecord, ShapeRecord], ...]:
    """An inventory whose pairings share a right member, so no reassignment tests anything.

    D91's case, made a fixture. Every pairing here names the same right record, so no pairing has
    an admissible substitute partner: there is no surrogate, and the honest answer is a refusal
    rather than a p-value of 1.0 that would read as a safeguard passing.
    """
    rng = np.random.default_rng(seed)
    shared = _record("degenerate_shared_right", native_seconds=_native(rng),
                     harmonics=_harmonics(rng), rng=rng)
    return tuple((_record("degenerate_left_%d" % index, native_seconds=_native(rng),
                          harmonics=_harmonics(rng), rng=rng), shared)
                 for index in range(size))


FIXTURE_BUILDERS = {
    "planted_shape_recurrence": planted_shape_recurrence,
    "same_normalisation_unrelated": same_normalisation_unrelated,
    "native_scale_alias": native_scale_alias,
    "degenerate_inventory": degenerate_inventory,
}


# ----------------------------------------------------------------------------- the inference


def statistic_grid(pairings: Sequence[Tuple[ShapeRecord, ShapeRecord]], *,
                   cycles: float = SPAN_CYCLES) -> np.ndarray:
    """Every left member against every right member, which is the whole reference set at once."""
    lefts = [left for left, _ in pairings]
    rights = [right for _, right in pairings]
    return np.asarray([[shape_recurrence(left, right, cycles=cycles) for right in rights]
                       for left in lefts], dtype=np.float64)


def exact_partner_p_values(pairings: Sequence[Tuple[ShapeRecord, ShapeRecord]], *,
                           cycles: float = SPAN_CYCLES) -> Dict[str, Any]:
    """The exact p-value under partner reassignment: how the pairing ranks among its alternatives.

    Every value the null can produce for a member is enumerated rather than sampled, because they
    can be: the reference set is the inventory's admissible alternative partners, and there are
    finitely many. The p-value counts the observation in both numerator and denominator, so it is
    the rank of the declared pairing among the pairings that could legitimately have replaced it.

    An inventory in which some pairing has no admissible partner is refused, not scored — the same
    refusal `reassign_scale_partners` gives, arrived at from the same set, so the test and the null
    cannot disagree about what the family is.
    """
    grid = statistic_grid(pairings, cycles=cycles)
    allowed = admissible_partners(list(pairings))
    empty = [index for index, candidates in enumerate(allowed) if not candidates]
    if empty:
        raise NullRefusal(
            "pairings", len(pairings),
            "an inventory in which every pairing has at least one admissible alternative partner. "
            "Pairings %s have none, so their surrogate would be the observation and a p-value "
            "reported for them would be a foregone 1.0 presented as a safeguard passing"
            % empty[:5])
    observed: List[float] = []
    p_values: List[float] = []
    reference_sizes: List[int] = []
    for index, candidates in enumerate(allowed):
        own = float(grid[index][index])
        alternatives = [float(grid[index][candidate]) for candidate in candidates]
        observed.append(own)
        reference_sizes.append(len(alternatives) + 1)
        p_values.append((1 + sum(1 for value in alternatives if value >= own))
                        / float(len(alternatives) + 1))
    return {"schema": "shape-exact-partner/v1", "observed": tuple(observed),
            "p_values": tuple(p_values), "reference_sizes": tuple(reference_sizes),
            "p_value_floor": 1.0 / max(reference_sizes),
            "claim_boundary": ("An exact rank of each declared pairing among its admissible "
                               "alternatives. It is not evidence about any domain.")}


def monte_carlo_partner_p_values(*args: Any, **kwargs: Any) -> Any:
    """Registered so it can be refused by name; never run.

    Drawing `n` whole reassignments and dividing by `n + 1` is the method `family_calibration.py`
    uses for calendar mode, where it is correct because a clock shift's support is far larger than
    the replication count. Under partner reassignment the support is `k - 1` values per member, so
    the replications resample the same handful of numbers and the denominator asserts a resolution
    the null does not have. A member beating its alternatives is reported at 0.001 when its exact
    tail probability is `1/k`, and the error is anti-conservative: on a wholly unrelated inventory
    of six pairings the family-wise false-positive rate is 74% against a nominal 5%.
    """
    raise NullRefusal(
        "inference", "monte_carlo_partner_p_values",
        "an exact partner test. This null's support is the admissible alternative partners and "
        "is finite, so replications resample the same values and the p-value denominator would "
        "claim a resolution the null does not have: see this function's stated measurement")


def corrected_family(pairings: Sequence[Tuple[ShapeRecord, ShapeRecord]], *,
                     cycles: float = SPAN_CYCLES, alpha: float = ALPHA,
                     correction: str = CORRECTION) -> Dict[str, Any]:
    """One inventory run as one family, corrected once, with rejections counted after correction."""
    exact = exact_partner_p_values(pairings, cycles=cycles)
    p_values = list(exact["p_values"])
    corrected = adjust(p_values, method=correction, alpha=alpha, n_tests=len(p_values),
                       labels=[left.label for left, _ in pairings])
    return {**exact, "schema": "shape-corrected-family/v1", "family_size": len(p_values),
            "alpha": alpha, "correction": correction,
            "adjusted": tuple(corrected["adjusted"]), "rejected": tuple(corrected["rejected"]),
            "n_rejected_after_correction": int(sum(corrected["rejected"]))}


#: What each case must do at the frozen family level, stated from what the case *is* rather than
#: from what it was measured to do. The planted case is the only one carrying a real shared shape,
#: so it is the only one entitled to reject; the two safeguards carry none, so any rejection there
#: is a false one; the degenerate inventory has no surrogate at all and must refuse.
FIXTURE_EXPECTATIONS: Tuple[Dict[str, Any], ...] = (
    {"case": "planted_shape_recurrence", "outcome": "rejects",
     "minimum_rejection_rate": 1.0,
     "expectation": ("one shape genuinely recurring at unrelated native durations: every member "
                     "rejects after correction")},
    {"case": "same_normalisation_unrelated", "outcome": "does_not_reject",
     "maximum_family_wise_error": ALPHA,
     "expectation": ("independent shapes through an identical normalisation: no member rejects, "
                     "and across realisations the family-wise error stays at or under alpha")},
    {"case": "native_scale_alias", "outcome": "does_not_reject",
     "maximum_family_wise_error": ALPHA,
     "expectation": ("a shared sampling grid must not become a shared shape: no member rejects, "
                     "and across realisations the family-wise error stays at or under alpha")},
    {"case": "degenerate_inventory", "outcome": "refuses",
     "expectation": ("an inventory whose pairings share one right member admits no reassignment "
                     "and is refused rather than scored")},
)


@lru_cache(maxsize=16)
def _cached_family(case: str, seed: int, size: int) -> Any:
    return corrected_family(FIXTURE_BUILDERS[case](seed, size))


__all__ = ["ALPHA", "CORRECTION", "FAMILY_SIZE", "FIXTURE_BUILDERS", "FIXTURE_EXPECTATIONS",
           "FIXTURE_ROWS", "NATIVE_SECONDS_RANGE", "ROWS_PER_CYCLE", "SPAN_CYCLES",
           "corrected_family", "degenerate_inventory", "exact_partner_p_values",
           "minimum_resolvable_family", "monte_carlo_partner_p_values", "native_scale_alias",
           "planted_shape_recurrence", "same_normalisation_unrelated", "statistic_grid"]
