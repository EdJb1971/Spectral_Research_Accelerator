"""TG6.2: the claim ladder over a TG6.1 evidence bundle.

`observation -> association -> robust association -> candidate precursor -> demonstrated
predictive utility`.  Each rung has deterministic entry conditions computed from the bundle and
nothing else, so rung assignment is a pure function of the evidence chain: no clock, no
filesystem, no randomness, no configuration and no argument other than the bundle.

Two rules dominate the ladder.  Any evidence entry recorded as ``FAIL`` or ``INVALID``, and any
standing contradiction or failure state recorded as ``PASS``, caps the bundle at ``observation``:
negative evidence is load-bearing here rather than decorative, and no quantity of favourable
evidence can outvote it.  And causal claims are outside the ladder entirely: the top rung is
demonstrated predictive utility, which is a statement about out-of-sample prediction and not
about mechanism or cause (R7).

Free commentary cannot reach these gates.  The gates read only the category, the status and, for
temporal precedence alone, one reserved boolean payload key of entries that TG6.1 already
constrains to its ten first-class scientific fields (R22).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle


CLAIM_LADDER_SCHEMA = "claim-ladder/v1"

CLAIM_RUNGS = (
    "observation",
    "association",
    "robust_association",
    "candidate_precursor",
    "demonstrated_predictive_utility",
)

#: Claim kinds this programme does not license at any rung, for want of a causal-inference
#: framework it does not provide (R7).  Asking whether the ladder permits one is an error, not
#: a question with a ``False`` answer, because a ``False`` invites a later "not yet" reading.
OUTSIDE_THE_LADDER = (
    "causal", "causation", "cause", "causes", "caused",
    "mechanism", "mechanistic", "efficacy", "cure", "explains", "because",
)

#: An entry carrying either status caps the bundle at ``observation``.
BLOCKING_STATUSES = ("FAIL", "INVALID")

#: Fields whose ``PASS`` entries assert that a contradiction or failure stands.
STANDING_NEGATIVE_FIELDS = ("contradictory_evidence", "failure_states")

#: The one reserved payload key the gates read, on ``provenance`` entries only.
PRECEDENCE_KEY = "temporal_precedence"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class GateResult:
    """One deterministic entry condition and whether this bundle satisfies it."""

    name: str
    rung: str
    satisfied: bool
    requirement: str

    def to_mapping(self) -> Dict[str, Any]:
        return {"name": self.name, "rung": self.rung, "satisfied": self.satisfied,
                "requirement": self.requirement}


@dataclass(frozen=True)
class _Gate:
    name: str
    rung: str
    requirement: str
    blocking: bool
    test: Callable[["_Census"], bool]


@dataclass(frozen=True)
class _Census:
    """Everything the gates are allowed to see, extracted once from the bundle."""

    counts: Mapping[Tuple[str, str], int]
    precedence_recorded: bool
    failed_sequences: Tuple[int, ...]
    standing_negative_sequences: Tuple[int, ...]

    def passes(self, category: str) -> int:
        return self.counts.get((category, "PASS"), 0)

    @property
    def blocking_sequences(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.failed_sequences) | set(self.standing_negative_sequences)))


def _census(bundle: EvidenceBundle) -> _Census:
    counts: Dict[Tuple[str, str], int] = {}
    failed = []
    standing = []
    precedence = False
    for entry in bundle.entries:
        key = (entry.category, entry.status)
        counts[key] = counts.get(key, 0) + 1
        if entry.status in BLOCKING_STATUSES:
            failed.append(entry.sequence)
        if entry.status == "PASS" and entry.category in STANDING_NEGATIVE_FIELDS:
            standing.append(entry.sequence)
        if entry.category == "provenance" and entry.status == "PASS" \
                and entry.payload.get(PRECEDENCE_KEY) is True:
            precedence = True
    return _Census(counts=counts, precedence_recorded=precedence,
                   failed_sequences=tuple(failed), standing_negative_sequences=tuple(standing))


_GATES: Tuple[_Gate, ...] = (
    _Gate("observation.no_failed_or_invalid_evidence", "observation",
          "no evidence entry is recorded as FAIL or INVALID", True,
          lambda census: not census.failed_sequences),
    _Gate("observation.no_standing_contradiction", "observation",
          "no contradictory evidence or failure state is recorded as an established PASS", True,
          lambda census: not census.standing_negative_sequences),
    _Gate("association.observation_recorded", "association",
          "at least one observations entry passes", False,
          lambda census: census.passes("observations") >= 1),
    _Gate("association.effect_size_estimated", "association",
          "at least one effect_sizes entry passes", False,
          lambda census: census.passes("effect_sizes") >= 1),
    _Gate("association.uncertainty_quantified", "association",
          "at least one uncertainty entry passes; a point estimate alone is not an association",
          False,
          lambda census: census.passes("uncertainty") >= 1),
    _Gate("robust_association.replicated", "robust_association",
          "at least one replication_results entry passes", False,
          lambda census: census.passes("replication_results") >= 1),
    _Gate("robust_association.confounders_addressed", "robust_association",
          "at least one confounders entry passes", False,
          lambda census: census.passes("confounders") >= 1),
    _Gate("candidate_precursor.provenance_auditable", "candidate_precursor",
          "at least one provenance entry passes", False,
          lambda census: census.passes("provenance") >= 1),
    _Gate("candidate_precursor.temporal_precedence_recorded", "candidate_precursor",
          "a passing provenance entry records %s exactly true" % PRECEDENCE_KEY, False,
          lambda census: census.precedence_recorded),
    _Gate("demonstrated_predictive_utility.out_of_sample", "demonstrated_predictive_utility",
          "at least one holdout_performance entry passes", False,
          lambda census: census.passes("holdout_performance") >= 1),
)


@dataclass(frozen=True)
class ClaimLadderAssessment:
    """Where one bundle stands on the ladder, and exactly why it stands no higher."""

    schema: str
    study_id: str
    hypothesis_sha256: str
    bundle_sha256: str
    revision: int
    rung: str
    rung_index: int
    unblocked_rung: str
    gates: Tuple[GateResult, ...]
    blocking_entries: Tuple[int, ...]

    @property
    def blocked(self) -> bool:
        return self.rung != self.unblocked_rung or bool(self.blocking_entries)

    @property
    def unsatisfied_gates(self) -> Tuple[str, ...]:
        return tuple(gate.name for gate in self.gates if not gate.satisfied)

    def permits(self, claim: str) -> bool:
        """Whether the bundle reaches ``claim``.  Causal claim kinds are refused, not denied."""
        if not isinstance(claim, str) or not claim.strip():
            raise InvalidParameterError("claim", claim, "a non-empty claim name")
        name = claim.strip().lower()
        if name in OUTSIDE_THE_LADDER:
            raise InvalidParameterError(
                "claim", claim,
                "a rung of the claim ladder. Causal claims are outside the ladder entirely and "
                "are unreachable without an explicit causal-inference framework this programme "
                "does not provide (R7)")
        if name not in CLAIM_RUNGS:
            raise InvalidParameterError("claim", claim,
                                        "one of %s" % ", ".join(CLAIM_RUNGS))
        return self.rung_index >= CLAIM_RUNGS.index(name)

    def to_mapping(self) -> Dict[str, Any]:
        return {"schema": self.schema, "study_id": self.study_id,
                "hypothesis_sha256": self.hypothesis_sha256,
                "bundle_sha256": self.bundle_sha256, "revision": self.revision,
                "rung": self.rung, "rung_index": self.rung_index,
                "unblocked_rung": self.unblocked_rung,
                "gates": [gate.to_mapping() for gate in self.gates],
                "blocking_entries": list(self.blocking_entries)}

    @property
    def assessment_sha256(self) -> str:
        """A digest binding this verdict to the exact bundle revision it was computed from."""
        return hashlib.sha256(_canonical(self.to_mapping())).hexdigest()


def _climb(gates: Tuple[GateResult, ...]) -> str:
    """The highest rung whose gates, and every lower rung's gates, are satisfied."""
    reached = CLAIM_RUNGS[0]
    for rung in CLAIM_RUNGS[1:]:
        if not all(gate.satisfied for gate in gates if gate.rung == rung):
            return reached
        reached = rung
    return reached


def assess_claim_ladder(bundle: EvidenceBundle) -> ClaimLadderAssessment:
    """Assign a rung.  A pure function of the bundle: no clock, no I/O, no other input."""
    if not isinstance(bundle, EvidenceBundle):
        raise InvalidParameterError("bundle", type(bundle).__name__, "an EvidenceBundle")
    census = _census(bundle)
    gates = tuple(GateResult(name=gate.name, rung=gate.rung, satisfied=bool(gate.test(census)),
                             requirement=gate.requirement)
                  for gate in _GATES)
    unblocked = _climb(gates)
    blocked = not all(gate.satisfied for gate in gates
                      if gate.rung == CLAIM_RUNGS[0])
    rung = CLAIM_RUNGS[0] if blocked else unblocked
    return ClaimLadderAssessment(
        schema=CLAIM_LADDER_SCHEMA, study_id=bundle.study_id,
        hypothesis_sha256=bundle.hypothesis.digest(), bundle_sha256=bundle.bundle_sha256,
        revision=bundle.revision, rung=rung, rung_index=CLAIM_RUNGS.index(rung),
        unblocked_rung=unblocked, gates=gates,
        blocking_entries=census.blocking_sequences)


__all__ = [
    "CLAIM_LADDER_SCHEMA", "CLAIM_RUNGS", "OUTSIDE_THE_LADDER", "BLOCKING_STATUSES",
    "STANDING_NEGATIVE_FIELDS", "PRECEDENCE_KEY", "GateResult", "ClaimLadderAssessment",
    "assess_claim_ladder",
]
