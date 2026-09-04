"""TG17.15 slice 1: which question a correspondence test asks, declared before it is answered.

`scale_shape_calibration` refuses because the declared null cannot reject at any inventory size
the enumerator can reach. TG17.11 established that as a fact about the null. This module
establishes *why*, and the why turns out not to be about compute.

**Two different questions have been travelling under one name.**

*   `joint_structure` asks whether the **whole correspondence structure** is special: take the
    inventory of pairings and reassign every partner at once, so the surrogate is a different
    global arrangement of the same records.
*   `per_correspondence` asks whether **this** left member's affinity for **this** partner is
    special, against a declared pool of candidate partners that are not themselves hypotheses.

They are not refinements of each other. They have different reference sets, different resolution
bounds and, on the same data, different answers. R15's requirement that a result name its estimand
is the reason this module exists rather than a comment: a test that does not say which of the two
it asked cannot be checked, and the two are easy to conflate because both are "reassign the
partners".

**The measurement that decides between them, computed here rather than asserted.**

`joint_reassignment_resolution(k)` enumerates the null's own valid reassignments and counts, for
one member, how many *distinct partners* it can actually receive. The two numbers diverge hard:

    k = 8 -> 14,833 valid reassignments, and 7 distinct partners for any one member.

The reassignment count is the derangement number and grows factorially. The partner count is
`k - 1`. A member's statistic depends only on which partner it received, so those 14,833 draws
produce seven distinct statistic values, and **using the reassignment count as the reference-set
size counts duplicates as independent evidence**. At k = 8 that is the difference between a floor
of 1/14834 = 6.7e-5 and the true 1/8 = 0.125 -- anticonservative by more than three orders of
magnitude. This is the single most likely error in any reimplementation of this test, so it is
measured by a function and pinned by a test rather than left as a warning.

**The consequence is that the gap TG17.11 found is structural.** Raising
`MAX_REASSIGNABLE_PAIRINGS` above 8 -- which is achievable honestly, since rejection sampling from
uniform permutations is exactly uniform on the valid subset -- raises the reassignment count and
not the partner count. No amount of compute lowers a floor of `1/k`.

**And the cause is one inventory doing two jobs.** Under `joint_structure` the k pairings are both
the hypotheses, which sets the multiplicity burden, and the source of alternative partners, which
sets the resolution. The two scale together and nearly cancel: the crossover is `H_k / k <= alpha`.
`per_correspondence` separates them, which is the whole point of preferring it.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.registry import Registry
# The measurement must use the *same* enumerator the null draws from. Reimplementing the validity
# constraint here would measure a different object and could agree with the null today and diverge
# silently later, which is the failure this whole module exists to make impossible.
from src.core.structural_nulls import MAX_REASSIGNABLE_PAIRINGS, _valid_reassignments
from src.statistics.multiple_comparisons import adjust


ESTIMAND_SCHEMA = "correspondence-estimand/v1"

#: The correction every declared correspondence family is tested under. Benjamini-Yekutieli
#: rather than Benjamini-Hochberg because members of one family share a partner inventory and are
#: therefore dependent; BY is valid under arbitrary dependence and BH is not. The choice is
#: load-bearing, not incidental, so it is named here beside the estimands it governs.
CORRECTION = "benjamini_yekutieli"
ALPHA = 0.05


@dataclass(frozen=True)
class CorrespondenceEstimand:
    """One question a correspondence test can ask, with the reference set that answers it."""

    name: str
    question: str
    reference_set: str
    #: What bounds the smallest p-value this estimand can produce, in words.
    resolution_rule: str
    #: True when the same inventory supplies both the hypotheses and the alternatives, so
    #: resolution cannot be bought without also buying multiplicity.
    resolution_coupled_to_family: bool
    admissible: bool
    reason: str

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": ESTIMAND_SCHEMA,
            "name": self.name,
            "question": self.question,
            "reference_set": self.reference_set,
            "resolution_rule": self.resolution_rule,
            "resolution_coupled_to_family": self.resolution_coupled_to_family,
            "admissible": self.admissible,
            "reason": self.reason,
        }


ESTIMANDS: Registry[CorrespondenceEstimand] = Registry("correspondence estimand")


def register_estimand(estimand: CorrespondenceEstimand) -> CorrespondenceEstimand:
    ESTIMANDS.add(estimand.name, estimand, description=estimand.question,
                  capabilities={"admissible": estimand.admissible,
                                "resolution_coupled_to_family":
                                    estimand.resolution_coupled_to_family})
    return estimand


JOINT_STRUCTURE = register_estimand(CorrespondenceEstimand(
    name="joint_structure",
    question=("Is the whole observed arrangement of partners special, against arrangements of the "
              "same inventory in which every pairing changes?"),
    reference_set=("the valid reassignments of the inventory -- but a single member's statistic "
                   "depends only on the partner it received, so its *effective* reference set is "
                   "the k-1 distinct partners available to it, not the reassignment count"),
    resolution_rule=("1/k for a k-pairing inventory, because a member has k-1 admissible "
                     "alternative partners however many joint reassignments exist"),
    resolution_coupled_to_family=True,
    admissible=False,
    reason=("Registered and refused by name rather than omitted, because it is the natural first "
            "design and its failure is not visible from inside it. The inventory supplies both "
            "the hypotheses and the alternatives, so lowering the 1/k floor requires growing k, "
            "which raises the Benjamini-Yekutieli burden at almost the same rate; the crossover "
            "`H_k / k <= alpha` puts the smallest resolvable family at a size the exact "
            "enumerator cannot reach, and one that arrives with no margin even if it could. "
            "Sampling further reassignments does not help: it raises the reassignment count and "
            "leaves the partner count at k-1."),
))


PER_CORRESPONDENCE = register_estimand(CorrespondenceEstimand(
    name="per_correspondence",
    question=("Is this left member's affinity for this partner special, against a declared pool "
              "of candidate partners that are not themselves under test?"),
    reference_set=("the declared partner pool, of size N, chosen before the test and independent "
                   "of how many correspondences are being tested"),
    resolution_rule=("1/(N+1) for a pool of N candidates, independent of the number of tested "
                     "correspondences m"),
    resolution_coupled_to_family=False,
    admissible=True,
    reason=("Resolution and multiplicity become independent knobs: margin can be bought by "
            "enlarging the pool at no correction cost. It is also the narrower question, and the "
            "narrowing must be declared rather than absorbed -- 'this shape at this native "
            "duration resembles that shape at that one' is a per-correspondence sentence, but it "
            "is not the same sentence as 'this arrangement is special'. The cost moves rather "
            "than disappearing: the pool must be exchangeable under the null, which is a "
            "curation obligation that can be violated silently where the enumeration bound "
            "announced itself."),
))


def joint_reassignment_resolution(k: int) -> Dict[str, Any]:
    """Measure what a k-pairing joint reassignment actually resolves, for one member.

    Computed from the null's own enumerator rather than written down, so the two numbers below
    cannot drift apart from the object they describe.
    """
    if isinstance(k, bool) or int(k) != k or k < 2:
        raise InvalidParameterError("k", k, "an integer inventory size of at least two pairings")
    if k > MAX_REASSIGNABLE_PAIRINGS:
        raise InvalidParameterError(
            "k", k,
            "at most %d, the size the null itself will enumerate. Measuring resolution above the "
            "bound the null refuses would describe a test that cannot be run"
            % MAX_REASSIGNABLE_PAIRINGS)
    pairs = [("L%d" % index, "R%d" % index) for index in range(int(k))]
    assignments = _valid_reassignments(pairs)
    rights = [right for _, right in pairs]
    partners = collections.Counter(rights[order[0]] for order in assignments)
    return {
        "pairings": int(k),
        "valid_reassignments": len(assignments),
        "distinct_partners_for_one_member": len(partners),
        "p_floor_partner_based": 1.0 / (len(partners) + 1),
        "p_floor_if_reassignments_miscounted": 1.0 / (len(assignments) + 1),
        "understatement_factor": ((len(partners) + 1) / (len(assignments) + 1)
                                  if assignments else float("nan")),
        "why": ("a member's statistic depends only on the partner it received, so the "
                "reassignment count is a count of spellings and not of distinguishable evidence"),
    }


def minimum_pool_size(tested_correspondences: int, *, alpha: float = ALPHA,
                      correction: str = CORRECTION, limit: int = 20000) -> int:
    """Smallest partner pool N whose floor `1/(N+1)` all `m` tests can clear under the correction.

    Solved against the real correction rather than a closed form, for the same reason
    `minimum_resolvable_family` is: if the correction changes, the number must change with it.
    """
    m = tested_correspondences
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise InvalidParameterError(
            "tested_correspondences", m, "a positive integer count of declared correspondences")
    labels = [str(index) for index in range(int(m))]
    for size in range(2, int(limit) + 1):
        floor = 1.0 / (size + 1)
        outcome = adjust([floor] * int(m), method=correction, alpha=alpha, n_tests=int(m),
                         labels=labels)
        if all(outcome["rejected"]):
            return size
    raise InvalidParameterError(
        "limit", limit, "a search bound large enough to contain a resolvable pool size")


def require_declared_estimand(name: Optional[str]) -> CorrespondenceEstimand:
    """Resolve a declared estimand, or refuse by name. There is no default.

    A correspondence test with no declared estimand is not under-specified in a way a sensible
    default could rescue: the two registered questions have different reference sets and different
    answers, so choosing one silently would be choosing the result.
    """
    if name is None or not str(name).strip():
        raise InvalidParameterError(
            "estimand", name,
            "one of %s, declared explicitly. A correspondence test has no default question: "
            "`joint_structure` and `per_correspondence` have different reference sets and can "
            "disagree on the same data, so an undeclared estimand would let the implementation "
            "choose the finding" % sorted(ESTIMANDS.names()))
    key = str(name).strip()
    if key not in set(ESTIMANDS.names()):
        raise InvalidParameterError(
            "estimand", key, "one of the registered estimands %s" % sorted(ESTIMANDS.names()))
    estimand = ESTIMANDS.get(key)
    if not estimand.admissible:
        raise InvalidParameterError(
            "estimand", key,
            "an admissible estimand. %s is registered so it can be refused by name rather than "
            "rediscovered: %s" % (key, estimand.reason))
    return estimand


def estimand_report() -> Dict[str, Any]:
    """Everything this slice decided, in one readable body for a receipt or a reviewer."""
    measured = [joint_reassignment_resolution(k)
                for k in range(4, MAX_REASSIGNABLE_PAIRINGS + 1)]
    return {
        "schema": ESTIMAND_SCHEMA,
        "alpha": ALPHA,
        "correction": CORRECTION,
        "correction_basis": (
            "members of one correspondence family share a partner inventory and are therefore "
            "dependent; Benjamini-Yekutieli is valid under arbitrary dependence and "
            "Benjamini-Hochberg is not"),
        "estimands": [ESTIMANDS.get(name).describe() for name in sorted(ESTIMANDS.names())],
        "chosen": PER_CORRESPONDENCE.name,
        "joint_reassignment_resolution": measured,
        "largest_enumerable_inventory": MAX_REASSIGNABLE_PAIRINGS,
        "pool_sizes_required": {
            str(m): minimum_pool_size(m) for m in (1, 3, 5, 6, 10, 20)},
        "claim_boundary": (
            "this declares which question a correspondence test asks and what bounds its "
            "resolution. It is not a calibration, a power analysis on real records, a partner "
            "pool, or a result"),
    }


__all__ = [
    "ESTIMAND_SCHEMA", "ALPHA", "CORRECTION", "CorrespondenceEstimand", "ESTIMANDS",
    "register_estimand", "JOINT_STRUCTURE", "PER_CORRESPONDENCE",
    "joint_reassignment_resolution", "minimum_pool_size", "require_declared_estimand",
    "estimand_report",
]
