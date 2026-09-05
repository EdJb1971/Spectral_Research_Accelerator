"""T4E.4: minimum-support decisions over T4E.3 constellation patterns.

Clustering says which noisy observations count as one approximate pattern.  This module asks the
smaller next question: which of those patterns meet a declared minimum number of distinct
constellation occurrences?  It neither creates clusters nor turns support into significance.

Candidates are ordered by decreasing support.  At the first candidate below the threshold every
remaining candidate is known to be below it and is pruned without further eligibility work.  A
candidate-count preflight and a monotonic wall-clock deadline refuse the whole sweep rather than
returning a prefix that could be mistaken for a complete search.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple

from src.analysis_engine.spectral_clustering import (
    ConstellationPattern, PatternCatalogue,
)
from src.core.errors import InvalidParameterError, UserInputError


MINING_SCHEMA = "spectral-constellation-support/v1"
SUPPORT_UNIT = "distinct constellation identity (signature.key)"


class MiningBudgetExceededError(UserInputError):
    """A complete support sweep did not fit its declared resource envelope."""

    def __init__(self, resource: str, observed: float, limit: float) -> None:
        super().__init__(
            "Constellation support mining exceeded its declared %s budget (%s > %s). The "
            "sweep is refused whole; no partial candidate prefix is a result. Raise the budget "
            "or reduce the clustered input."
            % (resource, observed, limit),
            resource=resource, observed=observed, limit=limit, partial_result=False)


@dataclass(frozen=True)
class MiningBudget:
    """Both hard limits are mandatory and travel in the result receipt."""

    max_candidates: int
    max_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.max_candidates, bool) or int(self.max_candidates) != self.max_candidates:
            raise InvalidParameterError(
                "MiningBudget.max_candidates", self.max_candidates,
                "a positive integer hard limit")
        if self.max_candidates < 1:
            raise InvalidParameterError(
                "MiningBudget.max_candidates", self.max_candidates,
                "a positive integer hard limit")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0.0:
            raise InvalidParameterError(
                "MiningBudget.max_seconds", self.max_seconds,
                "a positive finite wall-clock limit")

    def describe(self) -> Dict[str, Any]:
        return {"max_candidates": int(self.max_candidates),
                "max_seconds": float(self.max_seconds)}


def _identity(pattern: ConstellationPattern, member_index: int) -> str:
    key = pattern.members[member_index].key
    return json.dumps(list(key), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class SupportDecision:
    pattern: ConstellationPattern
    support: int
    status: str

    def describe(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern.pattern_id,
            "support": self.support,
            "support_unit": SUPPORT_UNIT,
            "status": self.status,
            "centroid": self.pattern.centroid.describe(),
            "tolerance_radius": self.pattern.tolerance_radius,
            "observed_radius": self.pattern.observed_radius,
        }


@dataclass(frozen=True)
class SupportMiningResult:
    decisions: Tuple[SupportDecision, ...]
    minimum_support: int
    budget: MiningBudget
    candidates_examined: int
    candidates_pruned_early: int
    elapsed_seconds_unasserted: float

    @property
    def supported(self) -> Tuple[ConstellationPattern, ...]:
        return tuple(decision.pattern for decision in self.decisions
                     if decision.status == "SUPPORTED")

    def __len__(self) -> int:
        return len(self.supported)

    def __iter__(self) -> Iterator[ConstellationPattern]:
        return iter(self.supported)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": MINING_SCHEMA,
            "algorithm": "decreasing-support scan with early tail pruning",
            "minimum_support": self.minimum_support,
            "support_unit": SUPPORT_UNIT,
            "candidate_count": len(self.decisions),
            "candidates_examined": self.candidates_examined,
            "candidates_pruned_early": self.candidates_pruned_early,
            "supported_count": len(self.supported),
            "budget": self.budget.describe(),
            "elapsed_seconds_unasserted": self.elapsed_seconds_unasserted,
            "patterns": [decision.describe() for decision in self.decisions],
            "claim_boundary": (
                "minimum support is a deterministic frequency filter over this supplied batch; "
                "it is not recurrence significance, a null test, a p-value, predictive evidence "
                "or a discovery"),
        }


def _validate_distinct_identities(catalogue: PatternCatalogue) -> None:
    seen: Dict[str, int] = {}
    for pattern in catalogue:
        for index in range(len(pattern.members)):
            identity = _identity(pattern, index)
            if identity in seen:
                raise InvalidParameterError(
                    "catalogue.patterns.members.key", identity,
                    "one occurrence of each constellation identity. The same key appears in "
                    "patterns %d and %d; counting it twice would manufacture support"
                    % (seen[identity], pattern.pattern_id))
            seen[identity] = pattern.pattern_id


def mine_supported_patterns(
        catalogue: PatternCatalogue, *, minimum_support: int, budget: MiningBudget,
        clock: Callable[[], float] = time.monotonic) -> SupportMiningResult:
    """Filter a complete catalogue, or refuse without returning a partial prefix."""
    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a complete T4E.3 PatternCatalogue")
    if isinstance(minimum_support, bool) or int(minimum_support) != minimum_support:
        raise InvalidParameterError(
            "minimum_support", minimum_support, "a positive integer occurrence count")
    if minimum_support < 1:
        raise InvalidParameterError(
            "minimum_support", minimum_support, "a positive integer occurrence count")
    if not isinstance(budget, MiningBudget):
        raise InvalidParameterError(
            "budget", type(budget).__name__, "an explicit MiningBudget with both hard limits")
    if len(catalogue.patterns) > budget.max_candidates:
        raise MiningBudgetExceededError(
            "candidate-count", len(catalogue.patterns), budget.max_candidates)

    started = float(clock())
    _validate_distinct_identities(catalogue)
    ordered = sorted(catalogue.patterns, key=lambda pattern: (-pattern.support,
                                                               pattern.pattern_id))
    after_preflight = float(clock())
    if after_preflight - started > budget.max_seconds:
        raise MiningBudgetExceededError(
            "wall-clock", after_preflight - started, budget.max_seconds)

    decisions = []
    examined = 0
    pruned = 0
    for index, pattern in enumerate(ordered):
        elapsed = float(clock()) - started
        if elapsed > budget.max_seconds:
            raise MiningBudgetExceededError("wall-clock", elapsed, budget.max_seconds)
        if pattern.support < minimum_support:
            tail = ordered[index:]
            decisions.extend(SupportDecision(item, item.support, "PRUNED_BELOW_MINIMUM")
                             for item in tail)
            pruned += len(tail)
            break
        decisions.append(SupportDecision(pattern, pattern.support, "SUPPORTED"))
        examined += 1

    elapsed = float(clock()) - started
    if elapsed > budget.max_seconds:
        raise MiningBudgetExceededError("wall-clock", elapsed, budget.max_seconds)
    # Receipts retain stable pattern order even though the scan order is support-first.
    decisions.sort(key=lambda decision: decision.pattern.pattern_id)
    return SupportMiningResult(
        decisions=tuple(decisions), minimum_support=int(minimum_support), budget=budget,
        candidates_examined=examined, candidates_pruned_early=pruned,
        elapsed_seconds_unasserted=elapsed)


__all__ = [
    "MINING_SCHEMA", "SUPPORT_UNIT", "MiningBudget", "MiningBudgetExceededError",
    "SupportDecision", "SupportMiningResult", "mine_supported_patterns",
]
