"""TG6.3: the five outputs of an evidence bundle.

For any TG6.1 bundle this module states, deterministically and from the bundle alone: what can be
claimed, what cannot, what evidence contradicts it, which alternative explanations remain, and
which single observation would most efficiently distinguish between them.

Like the TG6.2 ladder it sits on, ``summarise_evidence`` is a pure function of the bundle: no
clock, no filesystem, no environment, no randomness and no second argument, so the same snapshot
yields the same five outputs for anyone holding it.

Two boundaries are structural rather than stylistic.  Membership of every output is decided by
category, status and the gate table alone, so labels and summaries are carried to the reader but
can never move a verdict (R22).  And no output licenses a causal reading: ``permits`` refuses
causal claim kinds exactly as the ladder does, at every rung (R7).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle
from src.core.claim_ladder import (
    BLOCKING_STATUSES,
    CLAIM_RUNGS,
    STANDING_NEGATIVE_FIELDS,
    ClaimLadderAssessment,
    GateResult,
    assess_claim_ladder,
)


FIVE_OUTPUTS_SCHEMA = "five-outputs/v1"

#: What each rung entitles a reader to say, in the programme's own words rather than the study's.
RUNG_ENTITLEMENTS = {
    "observation": "Something was measured and recorded. No relationship may be asserted.",
    "association": "A relationship of a stated size, with stated uncertainty, was found in this "
                   "sample. It may not be described as robust, as a precursor, or as predictive.",
    "robust_association": "The relationship survived replication and a recorded treatment of "
                          "confounders. It may not be described as a precursor or as predictive.",
    "candidate_precursor": "The relationship is auditable and its temporal ordering is recorded, "
                           "so it is a candidate precursor. Predictive utility is not shown.",
    "demonstrated_predictive_utility": "Out-of-sample performance was recorded and passed. This "
                                       "is a statement about prediction, never about mechanism.",
}

#: Fields whose entries argue against the hypothesis.  Deliberately wider than the ladder's
#: blocking set: the ladder asks what may be claimed, this asks what argues against it, and a
#: recorded null result is contrary evidence even where it caps nothing.
CONTRARY_FIELDS = (*STANDING_NEGATIVE_FIELDS, "null_results")

#: One structural alternative explanation per climbing gate, total over the gate table.  Each
#: remains open exactly while its gate is unsatisfied, so the rule is one line rather than a
#: judgement.  These are the only alternatives the module can name unaided; every other
#: alternative must have been recorded by someone.
STRUCTURAL_ALTERNATIVES = {
    "association.observation_recorded": (
        "nothing_measured",
        "No passing observation is recorded, so there is no finding to explain.",
        "Record the observation itself as a passing observations entry."),
    "association.effect_size_estimated": (
        "no_estimated_effect",
        "No effect size is recorded, so the finding has no stated magnitude.",
        "Record an effect size estimate."),
    "association.uncertainty_quantified": (
        "chance",
        "The apparent relationship is sampling noise.",
        "Record an interval or other quantified uncertainty around the estimate."),
    "robust_association.replicated": (
        "sample_specific",
        "The relationship holds only in the sample it was discovered in.",
        "Replicate on an independent partition and record the result."),
    "robust_association.confounders_addressed": (
        "confounding",
        "A third factor drives both sides of the relationship.",
        "Record which confounders were considered and how each was addressed."),
    "candidate_precursor.provenance_auditable": (
        "unauditable_origin",
        "The result cannot be traced to the data and procedure that produced it.",
        "Record provenance for the result."),
    "candidate_precursor.temporal_precedence_recorded": (
        "reverse_or_simultaneous_order",
        "The outcome does not in fact follow the putative precursor in time.",
        "Record temporal precedence explicitly on a passing provenance entry."),
    "demonstrated_predictive_utility.out_of_sample": (
        "in_sample_optimism",
        "Performance was measured where the model was fitted.",
        "Measure performance on held-out data and record it."),
}

#: The three kinds of next observation, in the fixed precedence the selector applies.
OBSERVATION_KINDS = ("resolve_blocking_entry", "close_structural_gate",
                     "resolve_recorded_alternative")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ContraryEntry:
    """One recorded entry that argues against the hypothesis."""

    sequence: int
    category: str
    status: str
    label: str
    summary: str
    entry_sha256: str
    caps_at_observation: bool

    def to_mapping(self) -> Dict[str, Any]:
        return {"sequence": self.sequence, "category": self.category, "status": self.status,
                "label": self.label, "summary": self.summary,
                "entry_sha256": self.entry_sha256,
                "caps_at_observation": self.caps_at_observation}


@dataclass(frozen=True)
class Alternative:
    """One explanation, other than the hypothesis, that the bundle has not closed off."""

    identifier: str
    origin: str
    description: str
    closes_when: str
    sequence: Optional[int]

    def to_mapping(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "origin": self.origin,
                "description": self.description, "closes_when": self.closes_when,
                "sequence": self.sequence}


@dataclass(frozen=True)
class UnreachableRung:
    """A rung the bundle does not reach, and precisely what stands between."""

    rung: str
    entitlement: str
    blocked_by: Tuple[str, ...]
    requirements: Tuple[str, ...]

    def to_mapping(self) -> Dict[str, Any]:
        return {"rung": self.rung, "entitlement": self.entitlement,
                "blocked_by": list(self.blocked_by), "requirements": list(self.requirements)}


@dataclass(frozen=True)
class NextObservation:
    """The one observation this module nominates next, and why it was chosen."""

    kind: str
    target: str
    requirement: str
    addresses: Tuple[str, ...]
    sequence: Optional[int]

    def to_mapping(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "requirement": self.requirement,
                "addresses": list(self.addresses), "sequence": self.sequence}


@dataclass(frozen=True)
class FiveOutputs:
    """What may be claimed, what may not, what contradicts it, what remains, and what to do."""

    schema: str
    study_id: str
    hypothesis_sha256: str
    bundle_sha256: str
    revision: int
    assessment: ClaimLadderAssessment
    claimable: Tuple[str, ...]
    entitlement: str
    not_claimable: Tuple[UnreachableRung, ...]
    contradicting: Tuple[ContraryEntry, ...]
    alternatives: Tuple[Alternative, ...]
    next_observation: Optional[NextObservation]

    @property
    def rung(self) -> str:
        return self.assessment.rung

    @property
    def blocked(self) -> bool:
        return bool(self.assessment.blocking_entries)

    def permits(self, claim: str) -> bool:
        """Delegates to the ladder, so causal claim kinds are refused here too (R7)."""
        return self.assessment.permits(claim)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "study_id": self.study_id,
            "hypothesis_sha256": self.hypothesis_sha256, "bundle_sha256": self.bundle_sha256,
            "revision": self.revision, "assessment": self.assessment.to_mapping(),
            "claimable": list(self.claimable), "entitlement": self.entitlement,
            "not_claimable": [rung.to_mapping() for rung in self.not_claimable],
            "contradicting": [entry.to_mapping() for entry in self.contradicting],
            "alternatives": [alternative.to_mapping() for alternative in self.alternatives],
            "next_observation": (None if self.next_observation is None
                                 else self.next_observation.to_mapping()),
        }

    @property
    def summary_sha256(self) -> str:
        """A digest binding these five outputs to the exact bundle revision they came from."""
        return hashlib.sha256(_canonical(self.to_mapping())).hexdigest()

    def render(self) -> str:
        """A deterministic plain-text statement of the five outputs, in fixed order."""
        lines = ["study %s, revision %d, bundle %s" % (self.study_id, self.revision,
                                                       self.bundle_sha256),
                 "",
                 "1. What can be claimed: %s" % self.rung,
                 "   %s" % self.entitlement,
                 "2. What cannot be claimed:"]
        if self.not_claimable:
            for rung in self.not_claimable:
                lines.append("   %s - blocked by %s" % (rung.rung, ", ".join(rung.blocked_by)))
        else:
            lines.append("   no rung above %s is defined" % self.rung)
        lines.append("   causal claims are outside the ladder at every rung (R7)")
        lines.append("3. Evidence against:")
        if self.contradicting:
            for entry in self.contradicting:
                lines.append("   [%d] %s %s: %s" % (entry.sequence, entry.category, entry.status,
                                                    entry.label))
        else:
            lines.append("   none recorded; that is not the same as none existing")
        lines.append("4. Alternative explanations remaining:")
        if self.alternatives:
            for alternative in self.alternatives:
                lines.append("   %s (%s): %s" % (alternative.identifier, alternative.origin,
                                                 alternative.description))
        else:
            lines.append("   none that this bundle records or that its gates can name")
        lines.append("5. Most efficient next observation:")
        if self.next_observation is None:
            lines.append("   none identified; that is not the same as none existing")
        else:
            lines.append("   %s: %s" % (self.next_observation.kind,
                                        self.next_observation.requirement))
        return "\n".join(lines)


def _contradicting(bundle: EvidenceBundle,
                   blocking: Tuple[int, ...]) -> Tuple[ContraryEntry, ...]:
    """Every entry that argues against the hypothesis, in append order."""
    against = []
    for entry in bundle.entries:
        adverse = entry.status in BLOCKING_STATUSES
        contrary_field = entry.category in CONTRARY_FIELDS and entry.status != "NOT_APPLICABLE"
        if not adverse and not contrary_field:
            continue
        against.append(ContraryEntry(
            sequence=entry.sequence, category=entry.category, status=entry.status,
            label=entry.label, summary=entry.summary, entry_sha256=entry.entry_sha256,
            caps_at_observation=entry.sequence in blocking))
    return tuple(against)


def _alternatives(bundle: EvidenceBundle,
                  assessment: ClaimLadderAssessment) -> Tuple[Alternative, ...]:
    """Structural alternatives their gate has not closed, then alternatives someone recorded."""
    remaining = [Alternative(identifier=STRUCTURAL_ALTERNATIVES[gate.name][0],
                             origin="structural",
                             description=STRUCTURAL_ALTERNATIVES[gate.name][1],
                             closes_when=gate.requirement, sequence=None)
                 for gate in assessment.gates
                 if not gate.satisfied and gate.name in STRUCTURAL_ALTERNATIVES]
    for entry in bundle.entries:
        if entry.category == "confounders" and entry.status not in ("PASS", "NOT_APPLICABLE"):
            standing = "the confounder is addressed and recorded as passing"
        elif entry.category == "contradictory_evidence" and entry.status != "NOT_APPLICABLE":
            standing = "the contradiction is resolved rather than recorded as standing"
        elif entry.category == "null_results" and entry.status == "PASS":
            standing = "the null result is reconciled with the claimed relationship"
        else:
            continue
        remaining.append(Alternative(
            identifier="entry:%d" % entry.sequence, origin="recorded",
            description=entry.label, closes_when=standing, sequence=entry.sequence))
    return tuple(remaining)


def _next_observation(assessment: ClaimLadderAssessment,
                      alternatives: Tuple[Alternative, ...]) -> Optional[NextObservation]:
    """A fixed precedence, not an expected-information calculation: unblock, then climb, then
    resolve what was recorded.  The bundle carries no likelihoods, so none are pretended."""
    if assessment.blocking_entries:
        sequence = min(assessment.blocking_entries)
        return NextObservation(
            kind="resolve_blocking_entry", target="entry:%d" % sequence,
            requirement="Resolve the recorded failure at entry %d. While it stands, no rung "
                        "above %s is reachable by any further evidence."
                        % (sequence, CLAIM_RUNGS[0]),
            addresses=tuple(sorted("entry:%d" % number
                                   for number in assessment.blocking_entries)),
            sequence=sequence)
    for gate in assessment.gates:
        if gate.satisfied or gate.name not in STRUCTURAL_ALTERNATIVES:
            continue
        identifier, _, requirement = STRUCTURAL_ALTERNATIVES[gate.name]
        return NextObservation(kind="close_structural_gate", target=gate.name,
                               requirement=requirement, addresses=(identifier,), sequence=None)
    recorded = [alternative for alternative in alternatives if alternative.origin == "recorded"]
    if recorded:
        first = min(recorded, key=lambda alternative: alternative.sequence)
        return NextObservation(kind="resolve_recorded_alternative", target=first.identifier,
                               requirement="Distinguish the hypothesis from the alternative "
                                           "recorded at entry %d: %s"
                                           % (first.sequence, first.closes_when),
                               addresses=(first.identifier,), sequence=first.sequence)
    return None


def summarise_evidence(bundle: EvidenceBundle) -> FiveOutputs:
    """State the five outputs.  A pure function of the bundle: no clock, no I/O, no other input."""
    if not isinstance(bundle, EvidenceBundle):
        raise InvalidParameterError("bundle", type(bundle).__name__, "an EvidenceBundle")
    assessment = assess_claim_ladder(bundle)
    reached = assessment.rung_index

    not_claimable = []
    for index, rung in enumerate(CLAIM_RUNGS[reached + 1:], reached + 1):
        # A rung may satisfy every gate it declares and still be unreachable, because the ladder
        # is climbed in order: a failure on the floor, or an unmet gate on any rung between,
        # stands in the way.  Name the nearest such rung rather than leaving the explanation
        # empty, so no unreachable rung is ever reported without a reason.
        standing: Tuple[GateResult, ...] = ()
        for lower in range(index, -1, -1):
            standing = tuple(gate for gate in assessment.gates
                             if gate.rung == CLAIM_RUNGS[lower] and not gate.satisfied)
            if standing:
                break
        not_claimable.append(UnreachableRung(
            rung=rung, entitlement=RUNG_ENTITLEMENTS[rung],
            blocked_by=tuple(gate.name for gate in standing),
            requirements=tuple(gate.requirement for gate in standing)))

    alternatives = _alternatives(bundle, assessment)
    return FiveOutputs(
        schema=FIVE_OUTPUTS_SCHEMA, study_id=bundle.study_id,
        hypothesis_sha256=bundle.hypothesis.digest(), bundle_sha256=bundle.bundle_sha256,
        revision=bundle.revision, assessment=assessment,
        claimable=CLAIM_RUNGS[:reached + 1], entitlement=RUNG_ENTITLEMENTS[assessment.rung],
        not_claimable=tuple(not_claimable),
        contradicting=_contradicting(bundle, assessment.blocking_entries),
        alternatives=alternatives,
        next_observation=_next_observation(assessment, alternatives))


__all__ = [
    "FIVE_OUTPUTS_SCHEMA", "RUNG_ENTITLEMENTS", "CONTRARY_FIELDS", "STRUCTURAL_ALTERNATIVES",
    "OBSERVATION_KINDS", "ContraryEntry", "Alternative", "UnreachableRung", "NextObservation",
    "FiveOutputs", "summarise_evidence",
]
