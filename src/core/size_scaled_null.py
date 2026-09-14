"""T4E.44: what any size-scaled criterion must survive, derived before one is declared.

C6 of the T4E.41 acceptance asks for a bar that changes with the group's size, so that a group of
two is not evidence on the same terms as a group of six. The obvious way to build one is a
surrogate null that sets the bar per size, and that way has already been tried: T4E.16's candidate
5 was declared in full and withdrawn before adoption when a derivation showed the surrogate could
reassemble the very set it was testing. Its own conclusion was that *a feasible null is invalid
here and a valid null is infeasible*.

This module computes that constraint rather than restating it, so a ninth candidate is written
against measured limits instead of intuition. It adjudicates nothing, declares no criterion and
opens no reserve. Every function here is combinatorics over scene counts; none of it touches a
signature, a partition or any measured distance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from src.core.errors import InvalidParameterError


SCHEMA = "t4e44-size-scaled-constraints/v1"

#: The partition size this programme froze. Named rather than passed, because the whole point of
#: the constraint below is that changing it to rescue a rule would be tuning evidence to the rule.
FROZEN_SCENES = 6

#: The replicate count candidate 5 declared, taken from `_MOTIF_SURROGATES` for its 1/200 floor.
DECLARED_REPLICATES = 199

T4E16_DECLARATION = Path(
    "data/identity_calibration/t4e16-size-scaled-evidence-declaration.json")


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvalidParameterError(name, value, "a positive integer")
    return int(value)


def reassembly_rate(scenes: int = FROZEN_SCENES,
                    replicates: int = DECLARED_REPLICATES) -> Dict[str, Any]:
    """How often a redistribution surrogate rebuilds the set it is supposed to destroy.

    A surrogate that redistributes the partition's *actual* configurations puts each of the S
    motif copies into a uniformly chosen scene. When they happen to land in S distinct scenes the
    motif is reassembled exactly, with exactly its own diameter, and a rule admitting only sets
    strictly tighter than the tightest surrogate set of their size then rejects the motif.

    The per-replicate rate is therefore the probability that S uniform draws from S scenes are a
    permutation: `S! / S**S`. Over R independent replicates only one such landing is needed to
    set the bar, so the rate that matters is `1 - (1 - p)**R`.
    """
    scenes = _positive_int("scenes", scenes)
    replicates = _positive_int("replicates", replicates)
    per_replicate = math.factorial(scenes) / float(scenes ** scenes)
    return {
        "scenes": scenes,
        "replicates": replicates,
        "per_replicate": per_replicate,
        "over_replicates": 1.0 - (1.0 - per_replicate) ** replicates,
        "formula": "S! / S**S per replicate; 1 - (1 - p)**R over R replicates",
    }


def simulate_reassembly(scenes: int = FROZEN_SCENES, *, trials: int = 20000,
                        seed: int = 0) -> Dict[str, Any]:
    """The same quantity by simulation, so the closed form is checked rather than trusted."""
    scenes = _positive_int("scenes", scenes)
    trials = _positive_int("trials", trials)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(trials)
               if len({rng.randrange(scenes) for _ in range(scenes)}) == scenes)
    return {"scenes": scenes, "trials": trials, "seed": seed,
            "per_replicate": hits / float(trials)}


def scene_count_sensitivity(
    counts: Sequence[int] = (4, 6, 8, 10, 12),
    replicates: int = DECLARED_REPLICATES,
) -> List[Dict[str, Any]]:
    """How fast the defect disappears as S grows -- which is why it must not be used.

    Published because the tempting repair is visible in the numbers and must be refused in the
    open: widening the partition drives the reassembly rate down quickly, and a partition size
    chosen to make a criterion pass is evidence tuned to the rule rather than a rule tested
    against evidence.
    """
    return [reassembly_rate(count, replicates) for count in counts]


def exclusion_variant(scenes: int = FROZEN_SCENES) -> Dict[str, Any]:
    """The one repair that keeps the cached distance matrix, and the bias it buys.

    If the surrogate pool excludes the members of the set under test, the set cannot be
    reassembled by construction -- the reassembly rate is exactly zero, and that is provable
    rather than estimated. Pairwise distances are unchanged, so the cached matrix that made the
    design runnable survives and the cost does not return to the measured 189 hours.

    What it costs is stated here rather than discovered later. The excluded configurations are
    precisely the tightest group in the partition, so the surrogate pool is both smaller by `m`
    and stripped of its closest members. The tightest surrogate set of size `m` is therefore
    looser than it would otherwise be, the bar is easier to clear, and the bias runs toward false
    admission -- the direction that matters, because admitting a coincidence as a recurrence is
    the error this whole sequence exists to avoid. A candidate using this repair owes a
    quantification of that bias before it is adopted, not after it passes.
    """
    scenes = _positive_int("scenes", scenes)
    # Not simulated. The rate is zero because the members are absent from the pool, and a
    # simulation reporting zero would be a loop that cannot do otherwise dressed up as evidence.
    return {
        "scenes": scenes,
        "reassembly_rate": 0.0,
        "reassembly_is_zero_by_construction": True,
        "not_measured_because": "a set cannot be drawn from a pool it is not in; simulating this "
                                "would report zero by construction and look like a measurement",
        "why": "the members of the set under test are not in the pool the surrogate draws from, "
               "so no redistribution of that pool can produce them",
        "distances_unchanged": True,
        "cached_matrix_survives": True,
        "bias_direction": "TOWARD_FALSE_ADMISSION",
        "bias_mechanism": "the excluded members are the tightest group in the partition, so the "
                          "surrogate pool loses its closest configurations and the tightest "
                          "surrogate set of the same size is looser than it would otherwise be",
        "what_a_candidate_using_this_owes": "a quantification of that bias, derived before "
                                            "adoption, not a pass reported after it",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def size_scaled_constraints(
    declaration_path: Union[str, os.PathLike] = T4E16_DECLARATION,
) -> Dict[str, Any]:
    """The constraints a ninth candidate must satisfy, each bound to what established it.

    Nothing here is a design. Every entry names a result already in the record and states what it
    forbids, so a candidate can be checked against the constraint set before it is adopted rather
    than after it has spent a reserve.
    """
    declaration_path = Path(declaration_path)
    withdrawal: Mapping[str, Any] = {}
    if declaration_path.exists():
        body = json.loads(declaration_path.read_text(encoding="utf-8"))
        withdrawal = body.get("withdrawal", {})

    frozen = reassembly_rate()
    constraints = [
        {
            "id": "K1",
            "constraint": "A surrogate must not be able to reconstruct the set it is testing.",
            "established_by": "T4E.16's withdrawal, recomputed here",
            "measured": "a redistribution surrogate reassembles the motif at %.6f per replicate "
                        "and %.4f over %d replicates at S = %d"
                        % (frozen["per_replicate"], frozen["over_replicates"],
                           frozen["replicates"], frozen["scenes"]),
            "forbids": "any null built by redistributing the partition's own configurations",
        },
        {
            "id": "K2",
            "constraint": "A null that changes pairwise distances per replicate is not runnable.",
            "established_by": "T4E.16's cost derivation",
            "measured": "matching costs 0.8 s, 13.5 s and 95.3 s per partition at richness 6, 9 "
                        "and 12; a generative null invalidates the cached matrix and returns the "
                        "cost to the measured 189 hours at richness 12 alone",
            "forbids": "a per-replicate generative null at the declared richness levels",
        },
        {
            "id": "K3",
            "constraint": "K1 and K2 together admit only two shapes: a surrogate pool that "
                          "excludes the set under test, or no per-size surrogate at all.",
            "established_by": "the conjunction of K1 and K2",
            "measured": "the exclusion repair has a reassembly rate of exactly zero by "
                        "construction and keeps the cached matrix, at a bias toward false "
                        "admission that a candidate must quantify before adoption",
            "forbids": "presenting either shape as unconstrained; each carries a stated cost",
        },
        {
            "id": "K4",
            "constraint": "A bar calibrated on one partition may not be applied to another.",
            "established_by": "T4E.10 and T4E.11 candidate B",
            "measured": "four blocks from one generator differ 1.88x in mean same-configuration "
                        "distance, while the label-free close-pair normaliser varies only 1.084x "
                        "and measures the nearest unrelated distance rather than the "
                        "same-configuration one",
            "forbids": "cross-partition calibration of an absolute diameter, and repair by that "
                       "normaliser",
        },
        {
            "id": "K5",
            "constraint": "The partition size stays at %d." % FROZEN_SCENES,
            "established_by": "T4E.16's withdrawal",
            "measured": "the reassembly rate falls quickly with S, which is exactly why moving "
                        "it is forbidden: a partition size chosen to make a rule pass is "
                        "evidence tuned to the rule",
            "forbids": "widening S to rescue a criterion",
        },
        {
            "id": "K6",
            "constraint": "A ninth candidate inherits eight declared attempts' multiplicity.",
            "established_by": "T4E.16's own multiplicity section",
            "measured": "eight criteria declared and falsified; two reserves remain unspent and "
                        "unspendable without their own amendments",
            "forbids": "reporting a first success as though it were a single test, and opening "
                       "720-735 or 880-895 without an amendment",
        },
    ]
    result = {
        "schema": SCHEMA,
        "task": "T4E.44",
        "addresses": "C6 of the T4E.41 acceptance -- evidence that scales with the group",
        "status": "DERIVATION_ONLY_NO_CANDIDATE_DECLARED",
        "t4e16_declaration": str(declaration_path).replace("\\", "/"),
        "t4e16_declaration_sha256": (
            _file_sha256(declaration_path) if declaration_path.exists() else None),
        "t4e16_status": "WITHDRAWN_BEFORE_ADOPTION_BY_DERIVATION",
        "t4e16_withdrawal_reason": withdrawal.get("why"),
        "frozen_partition": frozen,
        "scene_count_sensitivity": scene_count_sensitivity(),
        "exclusion_variant": exclusion_variant(),
        "constraints": constraints,
        "what_this_is_not": [
            "It is not a criterion, a candidate, or a design for one.",
            "It adjudicates nothing and measures no signature, partition or distance.",
            "It does not meet C6, and registering it against C6 would be a category error: C6 "
            "asks for a criterion, and this is the set of limits one would have to satisfy.",
            "It opens no reserve and spends no evidence.",
        ],
        "network_used": False,
        "claim_boundary": (
            "This is arithmetic over scene counts, bound to results already in the record. It "
            "states what any size-scaled criterion must survive; it does not propose one, and "
            "satisfying every constraint here would not make a candidate correct."),
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


__all__ = [
    "DECLARED_REPLICATES", "FROZEN_SCENES", "SCHEMA", "T4E16_DECLARATION",
    "exclusion_variant", "reassembly_rate", "scene_count_sensitivity",
    "simulate_reassembly", "size_scaled_constraints",
]
