"""TG17.15 slice 3: the exact pool-substitution null, and the family size that cannot be chosen after the fact.

Slice 1 declared the question: `per_correspondence` asks whether *this* left member's affinity for
*this* partner is special against a declared pool. Slice 2 built the pool and made admission on the
statistic under test inexpressible. This slice is the inference itself: substitute each admitted
alternative for the observed partner, rank the observation among them, and correct the family once.

**Why exact and never sampled.** The reference set here is a *declared finite inventory*. Every
value the null can produce is enumerable by construction, so sampling from it could only add
variance to a quantity that has an exact value. `monte_carlo_pool_substitution` is registered and
refused by name for that reason -- and for a second, worse one: a Monte Carlo denominator is chosen
by the caller rather than fixed by the pool, so a pool of thirty alternatives could report `p =
0.001` when the smallest probability it can distinguish is `1/31 = 0.032`. That is precisely the
error `monte_carlo_partner_p_values` was refused for in scale/shape mode, arriving here by a
different route.

**One statistic path, so the observation and its alternatives cannot come from different code.**
Slice 1 imports the null's own `_valid_reassignments` rather than reimplementing validity, because
a measurement of a reference set must use the enumerator the reference set is drawn from. The same
requirement holds one level down. This module therefore takes a *callable* and evaluates all `N + 1`
values itself. It never accepts a precomputed observed statistic, because an observed value supplied
alongside alternatives computed here is an observed value that may have been computed under
different normalisation, different windowing, or a different version of the code -- and the
comparison would still return a confident number.

**Orientation is declared and has no default.** A statistic may be a similarity or a distance, and
the two invert the tail. An undeclared orientation would let the implementation choose which end of
the distribution counts as evidence, which is the same failure R15 exists to prevent one level up:
under the wrong orientation every reported p-value is roughly `1 - p`, the result still looks like a
result, and nothing announces it.

**The family size is sealed in the pools, not passed as an argument.** The classic multiplicity
failure is to run five hundred tests, discard the four hundred and eighty that looked unpromising,
and correct the survivors. `adjust` guards against that with `n_tests`, but only if the caller
passes an honest number. Here each `PartnerPool` already carries the `tested_correspondences` it was
admitted for, sealed into its digest before any statistic was computed. This module requires every
pool to declare the same family size *and* requires the number of pools presented to equal it. A
family cannot be narrowed after the p-values are seen, because narrowing it contradicts a number
that was fixed before they existed.

**Resolution is re-verified rather than assumed.** `build_partner_pool` refuses a pool below
`minimum_pool_size(m)`, but `PartnerPool` is a public dataclass and can be constructed directly. A
guarantee that holds only when a particular constructor was used is a guarantee that holds by
convention, so the floor is measured again here against the real correction.

**What this slice does not do.** It produces exact ranks. It does not establish that the ranks are
calibrated on real records -- that is slice 4, and it is the slice that can return an unwelcome
answer. T4C.5h is the standing reminder that a null can preserve exactly the property it is named
after and still get the distribution wrong: its surrogate measured a family-wise false-positive rate
of 0.765 against a nominal 0.05. Nothing here rules that out, and the receipt says so.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from src.core.correspondence_estimand import (
    ALPHA, CORRECTION, minimum_pool_size, minimum_pool_size_for_detected_fraction,
    sparsest_detectable_count,
)
from src.core.errors import InvalidParameterError
from src.core.partner_pool import PartnerPool, RecordProfile
from src.core.structural_nulls import NullRefusal
from src.statistics.multiple_comparisons import adjust


NULL_SCHEMA = "correspondence-pool-substitution/v1"
FAMILY_SCHEMA = "correspondence-pool-family/v1"

#: Which tail of the substituted distribution counts as evidence. Declared, never inferred: a
#: similarity and a distance invert it, and under the wrong one every p-value is about `1 - p`
#: while still looking like a p-value.
ORIENTATIONS: Dict[str, str] = {
    "larger_is_more_similar": (
        "the statistic increases with affinity, so an alternative is at least as extreme as the "
        "observation when its value is greater than or equal to it"),
    "smaller_is_more_similar": (
        "the statistic is a distance or divergence and decreases with affinity, so an alternative "
        "is at least as extreme as the observation when its value is less than or equal to it"),
}


def require_declared_orientation(name: Any) -> str:
    """Resolve a declared orientation, or refuse. There is no default and no inference."""
    if name is None or not str(name).strip():
        raise InvalidParameterError(
            "orientation", name,
            "one of %s, declared explicitly. A similarity and a distance put the evidence in "
            "opposite tails, so an undeclared orientation would let the implementation choose "
            "which end of the null distribution counts as a finding" % sorted(ORIENTATIONS))
    key = str(name).strip()
    if key not in ORIENTATIONS:
        raise InvalidParameterError("orientation", key, "one of %s" % sorted(ORIENTATIONS))
    return key


@dataclass(frozen=True)
class SubstitutionResult:
    """One correspondence ranked among the alternatives its own pool admitted."""

    left_id: str
    observed_partner_id: str
    orientation: str
    observed_statistic: float
    #: In pool order, which is the order candidates were offered. Deterministic, and not sorted:
    #: sorting would discard which alternative produced which value.
    alternative_statistics: Tuple[float, ...]
    alternative_ids: Tuple[str, ...]
    n_strictly_more_extreme: int
    n_tied: int
    pool_sha256: str
    contract_sha256: str

    @property
    def reference_size(self) -> int:
        """The observation plus its alternatives. The observation is always in its own reference
        set, which is what makes the p-value a valid tail probability rather than a rank."""
        return len(self.alternative_statistics) + 1

    @property
    def p_value(self) -> float:
        """`(1 + #{at least as extreme}) / (N + 1)`, ties counted toward the numerator.

        Counting ties as evidence against rejection is the conservative choice and the correct one:
        an alternative that achieves exactly the observed value is an alternative the statistic
        cannot distinguish from the observation, and treating indistinguishable as beaten would
        manufacture resolution the statistic does not have.
        """
        return (1 + self.n_strictly_more_extreme + self.n_tied) / float(self.reference_size)

    @property
    def p_floor(self) -> float:
        return 1.0 / float(self.reference_size)

    @property
    def degenerate(self) -> bool:
        """Every alternative tied the observation, so `p = 1.0` was a foregone conclusion.

        Not refused -- `p = 1.0` is conservative and reporting it is honest. But a statistic that
        cannot separate any member of a curated pool has no power here, and a reader seeing only
        the p-value would read a safeguard passing where there was never a test.
        """
        return bool(self.alternative_statistics) and self.n_tied == len(self.alternative_statistics)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": NULL_SCHEMA,
            "left": self.left_id,
            "observed_partner": self.observed_partner_id,
            "orientation": self.orientation,
            "orientation_meaning": ORIENTATIONS[self.orientation],
            "observed_statistic": float(self.observed_statistic),
            "alternative_ids": list(self.alternative_ids),
            "alternative_statistics": [float(value) for value in self.alternative_statistics],
            "n_strictly_more_extreme": int(self.n_strictly_more_extreme),
            "n_tied": int(self.n_tied),
            "reference_size": self.reference_size,
            "p_value": self.p_value,
            "p_floor": self.p_floor,
            "attained_floor": self.p_value == self.p_floor,
            "degenerate": self.degenerate,
            "pool_sha256": self.pool_sha256,
            "contract_sha256": self.contract_sha256,
            "inference": "exact_pool_substitution",
            "sampling": "none; the reference set is a declared finite inventory and is enumerated",
        }


def _payload(records: Mapping[str, Any], profile: RecordProfile, role: str) -> Any:
    if profile.record_id not in records:
        raise NullRefusal(
            "records[%r]" % profile.record_id, role,
            "a payload for every record the pool admitted. %s is in the sealed pool but absent "
            "from the records offered, and skipping it would shrink the denominator of an exact "
            "p-value without changing the pool digest that p-value is reported against -- an "
            "anticonservative error that leaves no trace" % profile.record_id)
    return records[profile.record_id]


def _evaluate(statistic: Callable[[Any, Any], float], left_payload: Any, right_payload: Any,
              *, right_id: str) -> float:
    """The single path every value in the reference set travels, the observation included."""
    value = statistic(left_payload, right_payload)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise NullRefusal(
            "statistic(left, %s)" % right_id, type(value).__name__,
            "a real-valued statistic. A comparison that cannot be ordered cannot be ranked")
    if not math.isfinite(numeric):
        raise NullRefusal(
            "statistic(left, %s)" % right_id, numeric,
            "a finite value for every member of the reference set. A non-finite alternative "
            "cannot be ranked, and dropping it would shrink the denominator of an exact p-value "
            "and report a resolution the pool does not have")
    return numeric


def exact_pool_substitution(*, pool: PartnerPool, records: Mapping[str, Any],
                            statistic: Callable[[Any, Any], float], orientation: str,
                            verify_determinism: bool = True) -> SubstitutionResult:
    """Rank one observed correspondence among the alternatives its pool admitted.

    The observed statistic is computed here, from the same callable and the same code path as the
    alternatives. It is deliberately not accepted as an argument: an observed value computed
    elsewhere may carry a different normalisation or a different code version, and the resulting
    comparison would be meaningless while still returning a number.
    """
    tail = require_declared_orientation(orientation)
    if not isinstance(pool, PartnerPool):
        raise InvalidParameterError("pool", type(pool).__name__, "a sealed PartnerPool")
    if not callable(statistic):
        raise InvalidParameterError(
            "statistic", type(statistic).__name__,
            "a callable of two record payloads. A precomputed observed value is not accepted: the "
            "observation and its alternatives must travel the same code path or the rank compares "
            "quantities that were never the same quantity")
    if not pool.admitted:
        raise NullRefusal(
            "pool.admitted", 0,
            "at least one admitted alternative. A reference set containing only the observation "
            "yields p = 1.0 by arithmetic, which would present a foregone conclusion as a "
            "safeguard passing")
    if pool.observed_partner.record_id in {item.record_id for item in pool.admitted}:
        raise NullRefusal(
            "pool.admitted", pool.observed_partner.record_id,
            "a pool that excludes the observed partner. The observation is already counted in "
            "both numerator and denominator; admitting it again would count it twice and shift "
            "every p-value upward by exactly one reference slot")

    left_payload = _payload(records, pool.left, "left member")
    observed = _evaluate(statistic, left_payload,
                         _payload(records, pool.observed_partner, "observed partner"),
                         right_id=pool.observed_partner.record_id)
    if verify_determinism:
        repeat = _evaluate(statistic, left_payload,
                           _payload(records, pool.observed_partner, "observed partner"),
                           right_id=pool.observed_partner.record_id)
        if repeat != observed:
            raise NullRefusal(
                "statistic", "%r then %r" % (observed, repeat),
                "a deterministic statistic. The same pair returned two values, so the rank of the "
                "observation among its alternatives depends on evaluation order and the p-value "
                "is not exact. Seed the statistic, or declare it and its draw as part of the null")

    alternatives: List[float] = []
    alternative_ids: List[str] = []
    for candidate in pool.admitted:
        alternatives.append(_evaluate(
            statistic, left_payload, _payload(records, candidate, "admitted alternative"),
            right_id=candidate.record_id))
        alternative_ids.append(candidate.record_id)

    if tail == "larger_is_more_similar":
        strictly = sum(1 for value in alternatives if value > observed)
    else:
        strictly = sum(1 for value in alternatives if value < observed)
    tied = sum(1 for value in alternatives if value == observed)

    return SubstitutionResult(
        left_id=pool.left.record_id, observed_partner_id=pool.observed_partner.record_id,
        orientation=tail, observed_statistic=observed,
        alternative_statistics=tuple(alternatives), alternative_ids=tuple(alternative_ids),
        n_strictly_more_extreme=int(strictly), n_tied=int(tied),
        pool_sha256=pool.digest, contract_sha256=pool.contract.digest)


def monte_carlo_pool_substitution(*args: Any, **kwargs: Any) -> Any:
    """Registered so it can be refused by name; never run.

    Sampling is the right tool when a null's support is large enough that enumerating it is
    infeasible. This null's support is a *declared finite inventory* -- slice 2 sealed it, by
    digest, before any statistic existed. Drawing `n` substitutions from it and dividing by
    `n + 1` therefore adds variance to a quantity with an exact value, and does something worse:
    the denominator becomes a number the caller chose rather than one the pool fixed. A pool of
    thirty alternatives resolves nothing finer than `1/31 = 0.032`, but ten thousand draws would
    report `p = 0.0001` with the same apparent authority.
    """
    raise NullRefusal(
        "inference", "monte_carlo_pool_substitution",
        "the exact pool substitution. The reference set is a sealed finite pool, so sampling adds "
        "variance to an exactly computable value and lets the p-value denominator be chosen by "
        "the caller instead of fixed by the pool: see this function's stated measurement")


@dataclass(frozen=True)
class CorrespondenceFamily:
    """One declared family of correspondences, corrected once, at the size it was declared at."""

    results: Tuple[SubstitutionResult, ...]
    estimand: str
    tested_correspondences: int
    alpha: float
    correction: str
    adjusted: Tuple[float, ...]
    rejected: Tuple[bool, ...]
    warnings: Tuple[str, ...]

    @property
    def n_rejected(self) -> int:
        return int(sum(1 for flag in self.rejected if flag))

    @property
    def digest(self) -> str:
        body = {"results": [item.describe() for item in self.results],
                "estimand": self.estimand, "alpha": float(self.alpha),
                "correction": self.correction,
                "tested_correspondences": int(self.tested_correspondences)}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def resolution(self) -> Dict[str, Any]:
        """Could this family have rejected at all, measured against the real correction?

        `build_partner_pool` already refuses an unresolvable pool, but `PartnerPool` is a public
        dataclass and a guarantee that depends on which constructor was used is a guarantee by
        convention. So the floor is measured again, here, against the same correction the family
        is actually corrected under.

        **Two questions, and the second is the one a reader needs.** The floor check asks whether
        every member could reject *when every member sits at its floor* -- the most favourable
        world there is, and the world `minimum_pool_size` sizes a pool for. A real family is
        mixed, and members that do not correspond consume the Benjamini-Yekutieli step-up ranks
        the genuine ones need. Reporting only the favourable case shows a green light for a family
        that cannot produce a finding: six pools of 58 clear the required 48 and still reject
        nothing unless five of their six correspondences are genuine. `sparsest_detectable_count`
        measures that number from the floors these pools actually have.
        """
        m = int(self.tested_correspondences)
        floors = [item.p_floor for item in self.results]
        at_floor = adjust(floors, method=self.correction, alpha=self.alpha, n_tests=m,
                          labels=["%s:%s" % (item.left_id, item.observed_partner_id)
                                  for item in self.results])
        unresolvable = [
            {"left": item.left_id, "observed_partner": item.observed_partner_id,
             "reference_size": item.reference_size, "p_floor": item.p_floor,
             "required_pool_size": minimum_pool_size(m, alpha=self.alpha,
                                                     correction=self.correction)}
            for item, ok in zip(self.results, at_floor["rejected"]) if not ok]
        sparsest = sparsest_detectable_count(
            floors, tested_correspondences=m, alpha=self.alpha, correction=self.correction)
        return {
            "worst_p_floor": max(floors) if floors else None,
            "adjusted_at_floor": [float(value) for value in at_floor["adjusted"]],
            "every_member_can_reject_at_its_own_floor": all(at_floor["rejected"]),
            "unresolvable_members": unresolvable,
            "required_pool_size": minimum_pool_size(m, alpha=self.alpha,
                                                    correction=self.correction),
            "basis": ("each member placed at the smallest p-value its own pool can produce, then "
                      "corrected as the declared family. A member that cannot reject here could "
                      "not reject at any effect size"),
            # The number above answers the most favourable question there is: it places *every*
            # member at its floor simultaneously. A real family is mixed, and members that do not
            # correspond consume the step-up ranks Benjamini-Yekutieli would otherwise have given
            # to the ones that do. Reporting only the all-genuine case would show a green light
            # for a family that cannot reject anything.
            "sparsest_detectable_count": sparsest,
            "sparsest_detectable_fraction": (None if sparsest is None else sparsest / float(m)),
            "sparsity_basis": (
                "the fewest genuinely corresponding members this family could reject, measured "
                "from the floors these pools actually have. `every_member_can_reject_at_its_own_"
                "floor` is the all-genuine case and is the most favourable world there is; this "
                "is the sparsity the family is actually powered for" if sparsest is not None else
                "no number of genuine correspondences would produce a rejection at these pool "
                "sizes; the family cannot reject at any effect size"),
            "pool_size_for_half_the_family": minimum_pool_size_for_detected_fraction(
                m, 0.5, alpha=self.alpha, correction=self.correction) if m > 1 else None,
        }

    def describe(self) -> Dict[str, Any]:
        resolution = self.resolution()
        return {
            "schema": FAMILY_SCHEMA,
            "estimand": self.estimand,
            "family_size": len(self.results),
            "tested_correspondences": int(self.tested_correspondences),
            "alpha": float(self.alpha),
            "correction": self.correction,
            "correction_basis": (
                "members of one correspondence family are tested against overlapping candidate "
                "inventories and are therefore dependent; Benjamini-Yekutieli is valid under "
                "arbitrary dependence and Benjamini-Hochberg is not"),
            "members": [item.describe() for item in self.results],
            "p_values": [item.p_value for item in self.results],
            "adjusted": [float(value) for value in self.adjusted],
            "rejected": [bool(flag) for flag in self.rejected],
            "n_rejected_after_correction": self.n_rejected,
            "degenerate_members": [item.left_id for item in self.results if item.degenerate],
            "resolution": resolution,
            "powered_for": (
                "this family can produce a rejection only if at least %s of its %d declared "
                "correspondences are genuine. Members that do not correspond consume the "
                "Benjamini-Yekutieli step-up ranks the genuine ones would need"
                % (resolution["sparsest_detectable_count"], len(self.results))
                if resolution["sparsest_detectable_count"] is not None else
                "this family cannot produce a rejection at any effect size: no number of genuine "
                "correspondences clears the correction at these pool sizes"),
            "warnings": list(self.warnings),
            "family_sha256": self.digest,
            "family_size_basis": (
                "the family size was sealed into every pool digest before any statistic was "
                "computed, and the number of correspondences presented must equal it. A family "
                "cannot be narrowed after its p-values are seen"),
            "claim_boundary": (
                "an exact rank of each declared correspondence among the alternatives its own "
                "pool admitted, corrected once at the declared family size. Its validity as a "
                "tail probability rests on the pool being exchangeable under the null, which "
                "slice 2 establishes as a necessary and not a sufficient condition. It is NOT a "
                "calibration: no false-positive rate has been measured for this null on records "
                "with no planted correspondence, and a null can preserve the property it is named "
                "after and still get the distribution wrong"),
        }


def correspondence_family(*, pools: Sequence[PartnerPool], records: Mapping[str, Any],
                          statistic: Callable[[Any, Any], float], orientation: str,
                          alpha: float = ALPHA, correction: str = CORRECTION,
                          verify_determinism: bool = True) -> CorrespondenceFamily:
    """Run every declared correspondence and correct once, at the size the pools were sealed at."""
    if not pools:
        raise InvalidParameterError(
            "pools", 0, "at least one sealed partner pool. An empty family has nothing to correct")
    declared = {int(pool.tested_correspondences) for pool in pools}
    if len(declared) != 1:
        raise NullRefusal(
            "pools[*].tested_correspondences", sorted(declared),
            "one declared family size shared by every pool. Pools admitted for different family "
            "sizes were admitted under different multiplicity burdens and are not one family")
    m = declared.pop()
    if len(pools) != m:
        raise NullRefusal(
            "len(pools)", len(pools),
            "exactly the %d correspondences these pools were sealed for. %s the declared family "
            "after the pools were built would change the multiplicity burden the resolution was "
            "checked against -- and if the p-values have already been seen, it is selection"
            % (m, "Narrowing" if len(pools) < m else "Enlarging"))

    estimands = {pool.estimand for pool in pools}
    if len(estimands) != 1:
        raise NullRefusal(
            "pools[*].estimand", sorted(estimands),
            "one estimand for the whole family. Different estimands ask different questions with "
            "different reference sets and cannot be corrected together")

    seen: Dict[Tuple[str, str], int] = {}
    for index, pool in enumerate(pools):
        key = (pool.left.record_id, pool.observed_partner.record_id)
        if key in seen:
            raise NullRefusal(
                "pools[%d]" % index, "%s -> %s" % key,
                "one pool per declared correspondence. This pairing is also at position %d, and "
                "counting one hypothesis twice would spend multiplicity budget on a duplicate "
                "while presenting it as a second piece of evidence" % seen[key])
        seen[key] = index

    results = tuple(
        exact_pool_substitution(pool=pool, records=records, statistic=statistic,
                                orientation=orientation, verify_determinism=verify_determinism)
        for pool in pools)

    corrected = adjust([item.p_value for item in results], method=correction, alpha=alpha,
                       n_tests=m,
                       labels=["%s:%s" % (item.left_id, item.observed_partner_id)
                               for item in results])
    return CorrespondenceFamily(
        results=results, estimand=estimands.pop(), tested_correspondences=m, alpha=float(alpha),
        correction=str(correction), adjusted=tuple(float(v) for v in corrected["adjusted"]),
        rejected=tuple(bool(flag) for flag in corrected["rejected"]),
        warnings=tuple(corrected.get("warnings", ()) or ()))


__all__ = [
    "NULL_SCHEMA", "FAMILY_SCHEMA", "ORIENTATIONS", "require_declared_orientation",
    "SubstitutionResult", "exact_pool_substitution", "monte_carlo_pool_substitution",
    "CorrespondenceFamily", "correspondence_family",
]
