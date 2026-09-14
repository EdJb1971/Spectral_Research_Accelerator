"""Follow-up experiment proposals: a proposal is a test, or it is a suggestion (`T4F.8`).

The platform already proposes follow-ups. `PatternDiscoveryEngine._propose_numerical_followup`
sees a positive correlation between a parameter and a metric and proposes a sweep of larger
values of that parameter; `_propose_categorical_followup` fixes the best-performing category and
re-runs. Both are useful and neither is a test. **They are optimisers**: their proposed run is
chosen so as to improve the metric, and no outcome of it would retract the finding that
prompted it. A procedure that only ever produces confirmations is not closing a loop, it is
closing a circle.

This module proposes the other kind. Seven things here exist because of the difference.

**1. A proposal that cannot come back negative is refused.** Every re-test carries a
`Refutation`: a named statistic, a threshold, and the direction that would retract the finding.
The threshold is checked against the range its statistic can actually attain, and a condition
no possible outcome satisfies is not a refutation -- it is a form of words. The commonest way
that happens is a base rate of zero, where no upper bound on a confidence can ever fall at or
below it, and the proposal is refused rather than dressed up.

**2. The prediction is registered before the record is read, and digested.** A predicted number
that can be edited after the result is known is a description of the result. `Registration`
digests exactly the pre-declared design -- the rule, the ground, the prediction, the refutation,
the required occurrences and the whole statistical declaration -- and excludes `status`, the
signatory and the date, so that registering an unchanged design is traceable to the design that
was reviewed. As in T4F.6, the code will not sign: `register(registered_by=...)` needs a name.

**3. A follow-up may not be proposed on the ground the finding was made on.** Re-running the
same test on the same region is not a re-test, and a target region whose pattern identities were
fitted there is R6's leak with a map in place of a calendar (T4F.7). The target must be a
declared held-out region of the partition, and declared leakage refuses the proposal.

**4. A rule that did not clear its own null is not a discovery to follow up.** Proposing a
"confirmation" for a rule the record could not distinguish from its surrogates manufactures a
finding out of a negative result, and it is the single easiest way for a proposal engine to
look productive. Only a `PRECURSOR_SIGNATURE` gets a re-test.

**5. A negative gets a proposal too, and it is a different kind.** A tool that proposes
follow-ups only for the things that worked has publication bias built into it. So a rule that
was *not* distinguished from its null can be proposed for a power increase -- but a power
proposal predicts nothing about the alignment and therefore carries no prediction and no
refutation, and it says so. It cannot confirm anything; it can only turn an absence that means
nothing into an absence that means something. And it is refused when the study that produced the
negative already had the occurrences needed to detect the effect: that is a real negative, and
asking for more data until it goes away is chasing.

**6. Power is computed from what can be observed without performing the test.** The quantity
under test is the *alignment* between antecedent and consequent. The design quantities are not:
how many of the antecedent's occurrences have a wholly-observed window, and how often the
consequent falls in a window dropped anywhere on the lattice, are both properties of the record
that can be read without ever counting the pair. So `required_occurrences` -- the smallest
number of eligible antecedent occurrences at which the predicted effect could be separated from
the base rate at the declared alpha, in both directions -- is computed and compared against what
the target actually supplies. A proposal the target cannot support comes back `UNDERPOWERED`
with the shortfall named, rather than being emitted as if it could answer.

**7. A proposal states its cost so it can be declined.** Frames at the record's cadence, the
lead in the record's own hours through T4F.6's bridge or the refusal naming which clock the two
sides are on, and the observable the target must carry. A proposal a maintainer cannot cost is
one they cannot say no to.

The predicted figures in a proposal are predictions. They are named `predicted_*` throughout and
`PROPOSAL_CLAIM_BOUNDARY` says it again, because the one failure mode this object invites is
that its numbers are read as measurements of something.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.analysis_engine.spectral_events import EventSeries
from src.analysis_engine.spectral_precursors import (
    INTERVAL_ASSUMPTION, INTERVAL_METHOD, PRECURSOR_CLAIM_BOUNDARY, PrecursorRule,
    RULE_PRECURSOR, eligible_anchors, window_base_rate, wilson_interval,
)
from src.analysis_engine.spectral_reference import Measurement, PhysicalBridge, lead_for
from src.analysis_engine.spectral_regions import (
    IdentityLeakage, ROLE_HELD_OUT, RegionPartition,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.errors import InvalidParameterError
from src.statistics.multiple_comparisons import PROCEDURES, check_power

PROPOSAL_SCHEMA = "spectral-followup-proposal/v1"
REGISTRATION_SCHEMA = "spectral-prediction-registration/v1"

#: What a proposal is for. The two are not two flavours of the same thing: the first carries a
#: prediction and can be refuted, the second carries neither and cannot confirm anything.
KIND_RE_TEST = "RE_TEST_ON_UNSEEN_GROUND"
KIND_POWER = "MAKE_AN_ABSENCE_INTERPRETABLE"
KINDS = (KIND_RE_TEST, KIND_POWER)

#: The design is sound and can be carried out. It means slightly different things to the two
#: kinds, and the difference is worth saying out loud: for a re-test it means the target supplies
#: the occurrences the design needs, so the experiment could confirm *or* retract; for a power
#: proposal it means some attainable number of occurrences would make the absence interpretable
#: and this proposal names it -- the record falling short of that number is the reason the
#: proposal exists, not a fault in it.
PROPOSAL_TESTABLE = "TESTABLE"
#: The design is sound but the target cannot supply the occurrences it needs. This is not a
#: refusal: the shortfall is named, and it is exactly what an acquisition would have to close.
PROPOSAL_UNDERPOWERED = "UNDERPOWERED_AS_PROPOSED"
#: There is no honest experiment here. The reasons say which of the seven walls it hit.
PROPOSAL_REFUSED = "REFUSED"

REGISTRATION_DRAFT = "draft"
REGISTRATION_REGISTERED = "registered"

#: The statistic a re-test is refuted on, and the direction. Named rather than implied because
#: "it did not replicate" is not a result and "the upper bound on the re-test's confidence fell
#: at or below the base rate that re-test measured" is.
REFUTATION_STATISTIC = "the Wilson upper bound on the re-test's confidence at the declared alpha"
REFUTATION_CONDITION = "at_or_below"
#: A confidence is a proportion, so its bounds live here and nowhere else. A threshold outside
#: this interval, or one no attainable bound could reach, makes the refutation unreachable.
CONFIDENCE_RANGE = (0.0, 1.0)

#: The largest number of trials the required-occurrence search will consider. A predicted
#: confidence and a base rate close enough to need more than this are not separable by any
#: record this programme could acquire, and saying so is the honest answer.
MAXIMUM_SEARCHED_OCCURRENCES = 100000

PROPOSAL_CLAIM_BOUNDARY = (
    "A proposal is a design, not a result. Every figure in it marked `predicted_` is what the "
    "discovery would imply if it holds on the new ground, and none of them has been measured "
    "there; quoting one as a measurement would report an experiment that has not been run. A "
    "proposal licenses no claim of cause, driver, mechanism, trigger, forecast or intervention "
    "(R7), and registering one makes it traceable, not true. A `TESTABLE` status says the "
    "target can supply the occurrences the design needs and that a stated outcome would retract "
    "the finding -- it does not say the finding is right, and it is not a prediction that the "
    "re-test will succeed. A power proposal carries no prediction at all: it can convert an "
    "uninformative absence into an informative one and it can never confirm anything.")

OPTIMISER_NOTE = (
    "The hypothesis engine's `_propose_numerical_followup` and `_propose_categorical_followup` "
    "are optimisers: they propose the parameter range or the category that made the metric "
    "better, so no outcome of the proposed run would retract the finding that prompted it. "
    "They are not tests and must not be reported as replication.")


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def _proportion(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidParameterError(name, value, "a proportion between 0 and 1")
    if not 0.0 <= number <= 1.0:
        raise InvalidParameterError(name, value, "a proportion between 0 and 1")
    return number


# --------------------------------------------------------------------------------------------
# what a re-test predicts, and what would retract it
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """What the discovery implies the new ground should show, stated before it is read."""

    predicted_confidence: float
    predicted_lift: Optional[float]
    discovery_base_rate: Optional[float]
    discovery_support: int
    discovery_eligible: int
    basis: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "predicted_confidence": self.predicted_confidence,
            "predicted_lift": self.predicted_lift,
            "measured_on_the_discovery_ground": {
                "base_rate": self.discovery_base_rate,
                "support": self.discovery_support,
                "eligible_antecedents": self.discovery_eligible,
            },
            "basis": self.basis,
            "these_are_predictions": (
                "Nothing here has been measured on the target ground. These are the figures the "
                "discovery implies, and the experiment exists to find out whether they hold."),
        }


@dataclass(frozen=True)
class Refutation:
    """The named outcome that retracts the finding, and the check that it can happen."""

    statistic: str
    condition: str
    threshold: float
    attainable_low: float
    attainable_high: float
    reachable: bool
    note: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "condition": self.condition,
            "threshold": self.threshold,
            "statistic_attainable_range": [self.attainable_low, self.attainable_high],
            "reachable": self.reachable,
            "note": self.note,
        }


def _refutation(base_rate: Optional[float]) -> Refutation:
    """Construct the retraction condition for a re-test, and say whether anything can meet it.

    The finding is retracted when the re-test's own confidence, bounded above at the declared
    alpha, fails to clear the base rate that same re-test measures. That is a statement about a
    proportion, so its attainable range is fixed at [0, 1] and a threshold outside it, or one
    the bound cannot reach, is not a refutation. A base rate of zero is the case worth naming:
    a Wilson upper bound is strictly positive for any number of trials, so "at or below zero" is
    a condition no experiment can ever satisfy and the proposal has no way to fail.
    """
    low, high = CONFIDENCE_RANGE
    if base_rate is None:
        return Refutation(
            statistic=REFUTATION_STATISTIC, condition=REFUTATION_CONDITION, threshold=float("nan"),
            attainable_low=low, attainable_high=high, reachable=False,
            note=("The ground carries no base rate, so there is no number for a re-test's "
                  "confidence to fail to clear and no outcome that would retract anything."))
    threshold = float(base_rate)
    reachable = low < threshold <= high
    if threshold <= low:
        note = ("A base rate of %g leaves this condition unreachable: a Wilson upper bound is "
                "strictly above zero for any number of trials, so no outcome of the proposed "
                "experiment could ever satisfy it and the design has no way to fail."
                % threshold)
    elif threshold > high:
        note = ("A threshold of %g is outside the range a confidence can take, so the condition "
                "is met by every outcome and distinguishes nothing." % threshold)
    else:
        note = ("The finding is retracted if the re-test's confidence, bounded above at the "
                "declared alpha, falls at or below the base rate the re-test itself measures -- "
                "that is, if the antecedent did no better than dropping the same window "
                "anywhere. %s" % INTERVAL_ASSUMPTION)
    return Refutation(statistic=REFUTATION_STATISTIC, condition=REFUTATION_CONDITION,
                      threshold=threshold, attainable_low=low, attainable_high=high,
                      reachable=reachable, note=note)


# --------------------------------------------------------------------------------------------
# how many occurrences it would take
# --------------------------------------------------------------------------------------------


def separable_at(n: int, predicted_confidence: float, base_rate: float, alpha: float) -> Dict[str, bool]:
    """Whether `n` trials could separate the predicted effect from the base rate, both ways.

    Two conditions, and both are needed. **Detectable**: if the effect is as predicted, the
    interval around it excludes the base rate, so the experiment could confirm. **Refutable**:
    if there is no effect at all, the interval around the base rate excludes the predicted
    confidence, so the experiment could retract. A design that satisfies only the first can
    report a success and cannot report a failure, which is the whole thing this module exists
    to prevent.

    The expected successes are carried as `p * n` rather than as a whole number of occurrences.
    Rounding makes both conditions non-monotone in `n` -- 336 trials can fail a separation that
    335 passes -- and a search over a jagged set returns an arbitrary member of it rather than
    the smallest sufficient design.
    """
    level = 1.0 - float(alpha)
    effect = wilson_interval(float(predicted_confidence) * n, n, level)
    null = wilson_interval(float(base_rate) * n, n, level)
    if effect is None or null is None:
        return {"detectable": False, "refutable": False}
    return {"detectable": effect[0] > float(base_rate),
            "refutable": null[1] < float(predicted_confidence)}


def required_occurrences(predicted_confidence: float, base_rate: float, alpha: float,
                         *, ceiling: int = MAXIMUM_SEARCHED_OCCURRENCES) -> Optional[int]:
    """The smallest number of eligible antecedent occurrences that separates the two, or `None`.

    `None` means no attainable record separates them at this alpha, which is a fact about how
    close the predicted confidence is to the base rate and not a fact about the target. It is
    returned rather than a very large number because a number would invite an acquisition that
    could not succeed.
    """
    predicted = _proportion("predicted_confidence", predicted_confidence)
    rate = _proportion("base_rate", base_rate)
    if predicted <= rate:
        # An early return rather than a semantic guard: the search below would reach the ceiling
        # and return `None` anyway, since no number of trials separates a prediction from a base
        # rate it does not exceed. It is kept because it says so in one line instead of after
        # seventeen doublings, and mutation testing correctly reports it as equivalent.
        return None
    low, high = 1, 1
    while high <= ceiling:
        checks = separable_at(high, predicted, rate, alpha)
        if checks["detectable"] and checks["refutable"]:
            break
        low = high + 1
        high *= 2
    if high > ceiling:
        checks = separable_at(ceiling, predicted, rate, alpha)
        if not (checks["detectable"] and checks["refutable"]):
            return None
        high = ceiling
    while low < high:
        middle = (low + high) // 2
        checks = separable_at(middle, predicted, rate, alpha)
        if checks["detectable"] and checks["refutable"]:
            high = middle
        else:
            low = middle + 1
    return int(low)


@dataclass(frozen=True)
class PowerRequirement:
    """What the design needs, what the ground supplies, and the gap between them."""

    alpha: float
    predicted_confidence: float
    base_rate: Optional[float]
    base_rate_source: str
    required: Optional[int]
    available: Optional[int]
    surrogate_power: Mapping[str, Any]
    note: str

    @property
    def met(self) -> bool:
        return (self.required is not None and self.available is not None
                and self.available >= self.required
                and bool(self.surrogate_power.get("can_reject_after_correction")))

    @property
    def shortfall(self) -> Optional[int]:
        if self.required is None or self.available is None:
            return None
        return max(0, self.required - self.available)

    def as_record(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "predicted_confidence": self.predicted_confidence,
            "base_rate": self.base_rate,
            "base_rate_source": self.base_rate_source,
            "required_eligible_antecedent_occurrences": self.required,
            "available_eligible_antecedent_occurrences": self.available,
            "shortfall": self.shortfall,
            "interval_method": INTERVAL_METHOD,
            "interval_assumption": INTERVAL_ASSUMPTION,
            "surrogate_power": dict(self.surrogate_power),
            "note": self.note,
            "observable_without_running_the_test": (
                "Both design quantities -- how many of the antecedent's occurrences have a "
                "wholly observed window, and how often the consequent falls in a window dropped "
                "anywhere on the lattice -- are properties of the record. Neither counts the "
                "pair, so reading them does not touch the alignment the experiment is for."),
        }


# --------------------------------------------------------------------------------------------
# the proposal
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One follow-up experiment: what it would test, on what ground, and how it could fail."""

    kind: str
    status: str
    antecedent: int
    consequent: int
    window: TransitionWindow
    target_region: Optional[str]
    partition_digest: Optional[str]
    prediction: Optional[Prediction]
    refutation: Optional[Refutation]
    power: Optional[PowerRequirement]
    lead: Optional[Measurement]
    observable: Optional[str]
    reasons: Tuple[str, ...]

    @property
    def testable(self) -> bool:
        return self.status == PROPOSAL_TESTABLE

    def design(self) -> Dict[str, Any]:
        """Exactly the pre-declared content, which is what a registration digests.

        `status`, the signatory and the date are not here on purpose: the digest has to identify
        the design that was reviewed, so that signing an unchanged design is traceable to it.
        """
        return {
            "schema": PROPOSAL_SCHEMA,
            "kind": self.kind,
            "rule": {"antecedent": self.antecedent, "consequent": self.consequent,
                     "window": [float(self.window.minimum_lag), float(self.window.maximum_lag),
                                self.window.time_units]},
            "target_region": self.target_region,
            "partition_digest": self.partition_digest,
            "prediction": None if self.prediction is None else self.prediction.as_record(),
            "refutation": None if self.refutation is None else self.refutation.as_record(),
            "power": None if self.power is None else self.power.as_record(),
            "observable_the_target_must_carry": self.observable,
        }

    def as_record(self) -> Dict[str, Any]:
        record = self.design()
        record.update({
            "status": self.status,
            "lead": None if self.lead is None else self.lead.as_record(),
            "reasons": list(self.reasons),
            "claim_boundary": PROPOSAL_CLAIM_BOUNDARY,
            "inherited_claim_boundary": PRECURSOR_CLAIM_BOUNDARY,
            "optimiser_note": OPTIMISER_NOTE,
        })
        return record


@dataclass(frozen=True)
class Registration:
    """A proposal fixed before the record is read, with a digest over the design alone."""

    proposal: Proposal
    status: str
    registered_by: Optional[str]
    registered_on: Optional[str]
    alpha: float
    correction: str
    null: str
    n_surrogates: int

    @property
    def digest(self) -> str:
        return _digest({"design": self.proposal.design(), "alpha": self.alpha,
                        "correction": self.correction, "null": self.null,
                        "n_surrogates": self.n_surrogates})

    def register(self, *, registered_by: str, registered_on: str) -> "Registration":
        """Sign it. The code will not: a pre-registration with no author is not one."""
        for name, value in (("registered_by", registered_by),
                            ("registered_on", registered_on)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    name, value,
                    "a non-empty declaration. A pre-registration is a statement somebody made "
                    "on a date, and code cannot make it on their behalf")
        return Registration(proposal=self.proposal, status=REGISTRATION_REGISTERED,
                            registered_by=registered_by.strip(),
                            registered_on=registered_on.strip(), alpha=self.alpha,
                            correction=self.correction, null=self.null,
                            n_surrogates=self.n_surrogates)

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema": REGISTRATION_SCHEMA,
            "status": self.status,
            "digest": self.digest,
            "registered_by": self.registered_by,
            "registered_on": self.registered_on,
            "declared": {"alpha": self.alpha, "correction": self.correction, "null": self.null,
                         "n_surrogates": self.n_surrogates},
            "digest_excludes": (
                "the proposal's status, the signatory and the date, so that registering an "
                "unchanged design is traceable to the design that was reviewed"),
            "proposal": self.proposal.as_record(),
        }


def register(proposal: Proposal, *, alpha: float, correction: str, null: str,
             n_surrogates: int) -> Registration:
    """Draft a registration for a testable proposal. A refused design cannot be registered."""
    if not isinstance(proposal, Proposal):
        raise InvalidParameterError("proposal", type(proposal).__name__, "a Proposal")
    if proposal.status == PROPOSAL_REFUSED:
        raise InvalidParameterError(
            "proposal.status", proposal.status,
            "a proposal this module did not refuse. Registering a refused design would put a "
            "digest on an experiment that cannot answer: %s" % "; ".join(proposal.reasons))
    if correction not in PROCEDURES:
        raise InvalidParameterError("correction", correction,
                                    "one of %s" % sorted(PROCEDURES))
    return Registration(proposal=proposal, status=REGISTRATION_DRAFT, registered_by=None,
                        registered_on=None, alpha=float(alpha), correction=str(correction),
                        null=str(null), n_surrogates=int(n_surrogates))


# --------------------------------------------------------------------------------------------
# the two proposals
# --------------------------------------------------------------------------------------------


def _check_target(target: str, partition: RegionPartition,
                  leakage: Optional[IdentityLeakage]) -> List[str]:
    """The ground checks, as a list of reasons to refuse. An empty list is a clean target."""
    reasons: List[str] = []
    names = [item.name for item in partition]
    if target not in names:
        return ["region %r is not in this partition, which declares %s"
                % (target, ", ".join(repr(item) for item in names))]
    region = partition.region(target)
    if region.role != ROLE_HELD_OUT:
        reasons.append(
            "region %r is the discovery region, so re-running the test there is not a re-test: "
            "the rule was found on that ground and would be confirmed by it whatever is true "
            "elsewhere" % target)
    if leakage is not None and not leakage.clean:
        reasons.append(
            "the pattern identities were fitted using %d member(s) drawn from held-out ground, "
            "so the thing being re-tested was partly defined by the data it would be re-tested "
            "on (R6)" % leakage.n_leaked)
    return reasons


def _supply(series: Optional[EventSeries], antecedent: int, consequent: int,
            window: TransitionWindow) -> Tuple[bool, Optional[int], Optional[float]]:
    """What the target supplies, read without counting the pair: trials, and the base rate.

    The first element says whether a record was offered at all, and it is separate from whether
    a base rate could be read out of one. Those are different situations and conflating them is
    how the discovery's own base rate gets quietly substituted for a target that was handed over
    and turned out to be unreadable -- a borrowed measurement standing in for a refused one.
    """
    if series is None:
        return False, None, None
    anchors = eligible_anchors(series, int(antecedent), window)
    rate, _positions, _hits = window_base_rate(series, int(consequent), window)
    return True, len(anchors), rate


def _lead(bridge: Optional[PhysicalBridge], series: Optional[EventSeries],
          window: TransitionWindow) -> Optional[Measurement]:
    if bridge is None:
        return None
    if series is None:
        return bridge.lead(window)
    return lead_for(bridge, series, window)


def propose_re_test(rule: PrecursorRule, *, target: str, partition: RegionPartition,
                    supply: Optional[EventSeries] = None,
                    leakage: Optional[IdentityLeakage] = None,
                    bridge: Optional[PhysicalBridge] = None,
                    alpha: float = 0.05,
                    correction: str = "benjamini_yekutieli",
                    n_surrogates: int = 999,
                    family_size: int = 1) -> Proposal:
    """Propose re-testing one precursor signature on ground it was not found on.

    Refused when the rule is not a signature, when the target is the discovery region or is not
    declared at all, when the identities leaked into held-out ground, or when the refutation
    condition the figures produce is one no outcome could satisfy. `UNDERPOWERED` when the
    design is sound but the target cannot supply the occurrences it needs -- with the shortfall
    named, because that is the number an acquisition would have to close.
    """
    if not isinstance(rule, PrecursorRule):
        raise InvalidParameterError("rule", type(rule).__name__, "a PrecursorRule")
    if not isinstance(partition, RegionPartition):
        raise InvalidParameterError("partition", type(partition).__name__, "a RegionPartition")
    if correction not in PROCEDURES:
        raise InvalidParameterError("correction", correction, "one of %s" % sorted(PROCEDURES))

    reasons: List[str] = []
    if rule.status != RULE_PRECURSOR:
        reasons.append(
            "this rule is %r, not %r, so there is no finding to re-test. Proposing a "
            "confirmation for a rule the record could not distinguish from its own surrogates "
            "would manufacture a discovery out of a negative result"
            % (rule.status, RULE_PRECURSOR))
    reasons.extend(_check_target(target, partition, leakage))

    offered, available, target_rate = _supply(supply, rule.antecedent, rule.consequent,
                                              rule.window)
    if target_rate is not None:
        base_rate, source = target_rate, "the target region's own record"
    elif not offered:
        base_rate = rule.base_rate
        source = ("the discovery region's, assumed for the target because the target supplies "
                  "no record to read one from")
    else:
        # A record was handed over and no base rate could be read from it. Falling back to the
        # discovery's here would put a number on this ground that this ground did not produce.
        base_rate = None
        source = ("unreadable: a record was supplied for this target and no base rate could be "
                  "taken from it, so there is none to assume and none to borrow")
        reasons.append(
            "a record was supplied for region %r and no base rate could be read from it -- its "
            "grid declares no cadence, or no window on it was wholly observed. Using the "
            "discovery region's base rate for ground that refused to give one would put a "
            "figure on this target that this target did not produce" % target)

    predicted = rule.confidence
    refutation = _refutation(base_rate)
    if not refutation.reachable:
        reasons.append("the refutation condition is unreachable: %s" % refutation.note)

    power: Optional[PowerRequirement] = None
    if predicted is not None and base_rate is not None:
        # `required_occurrences` already returns `None` when the prediction is no better than
        # the base rate; a second copy of that comparison here would be a second thing to keep
        # in step with it.
        required = required_occurrences(predicted, base_rate, alpha)
        surrogate_power = check_power(int(n_surrogates), int(family_size), alpha=float(alpha),
                                      method=correction)
        if required is None:
            note = ("A predicted confidence of %g does not exceed a base rate of %g, so there "
                    "is no effect for any number of occurrences to separate."
                    % (predicted, base_rate)) if predicted <= base_rate else (
                    "A predicted confidence of %g and a base rate of %g are too close to "
                    "separate at alpha=%g with any record of at most %d occurrences."
                    % (predicted, base_rate, alpha, MAXIMUM_SEARCHED_OCCURRENCES))
        else:
            note = ("%d eligible antecedent occurrences would let the predicted confidence of "
                    "%g be separated from a base rate of %g at alpha=%g, in both directions: "
                    "the interval around the effect excludes the base rate, and the interval "
                    "around the base rate excludes the effect."
                    % (required, predicted, base_rate, alpha))
        power = PowerRequirement(
            alpha=float(alpha), predicted_confidence=float(predicted), base_rate=base_rate,
            base_rate_source=source, required=required, available=available,
            surrogate_power=surrogate_power, note=note)
    else:
        reasons.append(
            "this rule has no measured confidence or no base rate, so there is no number to "
            "predict and nothing for an experiment to check it against")

    prediction = None if predicted is None else Prediction(
        predicted_confidence=float(predicted),
        predicted_lift=None if rule.lift is None else float(rule.lift),
        discovery_base_rate=rule.base_rate, discovery_support=int(rule.support),
        discovery_eligible=int(rule.eligible_antecedents),
        basis=("the confidence and lift this rule measured on the discovery ground, carried "
               "across unchanged. The base rate is a property of the target's own record and is "
               "not carried across"))

    if reasons:
        status = PROPOSAL_REFUSED
    elif power is not None and power.met:
        status = PROPOSAL_TESTABLE
    else:
        status = PROPOSAL_UNDERPOWERED
        if power is not None and power.required is None:
            # The note already distinguishes "the prediction is not above the base rate" from
            # "they are too close to separate", and a second wording here could disagree.
            reasons.append(power.note)
        elif power is not None and power.available is None:
            reasons.append("the target supplies no record yet, so an acquisition carrying at "
                           "least %d eligible antecedent occurrences is what this proposal is "
                           "asking for" % power.required)
        elif power is not None and power.shortfall:
            reasons.append("the target supplies %d eligible antecedent occurrences and the "
                           "design needs %d, a shortfall of %d"
                           % (power.available, power.required, power.shortfall))
        elif power is not None:
            reasons.append("the surrogate ensemble cannot reject after correction: %s"
                           % power.surrogate_power["warning"])

    return Proposal(
        kind=KIND_RE_TEST, status=status, antecedent=int(rule.antecedent),
        consequent=int(rule.consequent), window=rule.window, target_region=str(target),
        partition_digest=partition.digest, prediction=prediction, refutation=refutation,
        power=power, lead=_lead(bridge, supply, rule.window),
        observable=None if bridge is None else bridge.observable,
        reasons=tuple(reasons))


def propose_power_increase(rule: PrecursorRule, *, supply: Optional[EventSeries] = None,
                           bridge: Optional[PhysicalBridge] = None,
                           target_confidence: Optional[float] = None,
                           alpha: float = 0.05,
                           correction: str = "benjamini_yekutieli",
                           n_surrogates: int = 999,
                           family_size: int = 1) -> Proposal:
    """Propose what it would take to make a negative result mean something.

    This carries **no prediction and no refutation**, and both are `None` rather than filled in
    with something plausible: nothing about the alignment is being asserted, so there is nothing
    an outcome could retract. What it states is how many occurrences would have been needed to
    detect an effect of the declared size, so that a reader can tell an absence that means
    "there is nothing there" from one that means "this record could never have said".

    Refused when the rule already cleared its null -- there is no absence to interpret -- and
    refused when the study that produced the negative already had the occurrences required. That
    second refusal is the important one: a negative from an adequately powered study is a
    result, and proposing more data until it changes is chasing it.
    """
    if not isinstance(rule, PrecursorRule):
        raise InvalidParameterError("rule", type(rule).__name__, "a PrecursorRule")
    if correction not in PROCEDURES:
        raise InvalidParameterError("correction", correction, "one of %s" % sorted(PROCEDURES))

    reasons: List[str] = []
    if rule.status == RULE_PRECURSOR:
        reasons.append(
            "this rule cleared its null, so there is no absence to interpret. A power increase "
            "for a result that already rejected would be asking for data to strengthen a "
            "finding rather than to make an uninformative one informative")

    offered, available, target_rate = _supply(supply, rule.antecedent, rule.consequent,
                                              rule.window)
    if available is None:
        available = int(rule.eligible_antecedents)
    if target_rate is not None:
        base_rate, source = target_rate, "the record supplied with this proposal"
    elif not offered:
        base_rate, source = rule.base_rate, "the record the negative was measured on"
    else:
        base_rate = None
        source = ("unreadable: a record was supplied and no base rate could be taken from it, "
                  "so there is none to assume and none to borrow")

    if target_confidence is None and base_rate is not None:
        # The default effect size is the smallest one worth an experiment: a doubling of the
        # base rate. Declared rather than derived from the observation, because sizing a study
        # on the effect that record happened to show is sizing it on noise.
        declared = min(1.0, 2.0 * float(base_rate))
    else:
        declared = None if target_confidence is None else _proportion("target_confidence",
                                                                      target_confidence)

    power: Optional[PowerRequirement] = None
    if declared is not None and base_rate is not None:
        required = required_occurrences(declared, base_rate, alpha)
        surrogate_power = check_power(int(n_surrogates), int(family_size), alpha=float(alpha),
                                      method=correction)
        note = ("%s occurrences would be needed to separate a confidence of %g from a base rate "
                "of %g at alpha=%g. The record carried %d."
                % ("No attainable number of" if required is None else str(required),
                   declared, base_rate, alpha, available))
        power = PowerRequirement(
            alpha=float(alpha), predicted_confidence=float(declared), base_rate=base_rate,
            base_rate_source=source, required=required, available=int(available),
            surrogate_power=surrogate_power, note=note)
        if reasons:
            # There is no negative here to be adequately powered *for*, and running the guard
            # anyway would describe a rule that cleared its null as "this negative".
            pass
        elif required is not None and available >= required \
                and surrogate_power["can_reject_after_correction"]:
            reasons.append(
                "the study that produced this negative already carried %d eligible antecedent "
                "occurrences against the %d an effect of this size needs, so the absence is a "
                "result rather than a gap. Asking for more data until it changes is chasing it"
                % (available, required))
    else:
        reasons.append(
            "there is no base rate on this ground, so there is no effect size to power against "
            "and no number of occurrences that would make the absence mean anything")

    status = PROPOSAL_REFUSED if reasons else (
        PROPOSAL_TESTABLE if power is not None and power.required is not None
        and power.surrogate_power["can_reject_after_correction"] else PROPOSAL_UNDERPOWERED)
    if status == PROPOSAL_UNDERPOWERED and power is not None:
        if power.required is None:
            reasons.append("no attainable record separates an effect of this size from the base "
                           "rate at this alpha, so no acquisition would make the absence "
                           "interpretable")
        else:
            reasons.append("the surrogate ensemble cannot reject after correction whatever the "
                           "record supplies: %s" % power.surrogate_power["warning"])

    return Proposal(
        kind=KIND_POWER, status=status, antecedent=int(rule.antecedent),
        consequent=int(rule.consequent), window=rule.window, target_region=None,
        partition_digest=None, prediction=None, refutation=None, power=power,
        lead=_lead(bridge, supply, rule.window),
        observable=None if bridge is None else bridge.observable,
        reasons=tuple(reasons))


# --------------------------------------------------------------------------------------------
# closing the loop: the config the experiment engine already runs
# --------------------------------------------------------------------------------------------


def followup_experiment_config(proposal: Proposal, parent: Mapping[str, Any], *,
                               parent_id: Optional[str] = None,
                               parent_name: str = "the discovery run") -> Dict[str, Any]:
    """The proposal as the `parameter_matrix` config the experiment engine already accepts.

    This is the point of the task: the same shape `_propose_numerical_followup` returns, so the
    platform can run its own follow-up, but carrying the prediction, the refutation and the
    power requirement with it. A refused proposal does not get one -- emitting a runnable config
    for a design this module refused would hand the runner an experiment that cannot answer.
    """
    if not isinstance(proposal, Proposal):
        raise InvalidParameterError("proposal", type(proposal).__name__, "a Proposal")
    if proposal.status == PROPOSAL_REFUSED:
        raise InvalidParameterError(
            "proposal.status", proposal.status,
            "a proposal this module did not refuse (%s)" % "; ".join(proposal.reasons))
    if not isinstance(parent, Mapping):
        raise InvalidParameterError("parent", type(parent).__name__,
                                    "the parent experiment's config mapping")

    matrix = dict(parent.get("parameter_matrix", {}))
    matrix["precursor_antecedent"] = [int(proposal.antecedent)]
    matrix["precursor_consequent"] = [int(proposal.consequent)]
    matrix["transition_window"] = [[float(proposal.window.minimum_lag),
                                    float(proposal.window.maximum_lag)]]
    if proposal.target_region is not None:
        matrix["region"] = [proposal.target_region]

    if proposal.kind == KIND_RE_TEST:
        description = (
            "Re-test of the rule (%d -> %d) on region %r, which is held out of the ground it "
            "was found on. The finding is retracted if %s falls %s %g."
            % (proposal.antecedent, proposal.consequent, proposal.target_region,
               REFUTATION_STATISTIC, REFUTATION_CONDITION.replace("_", " "),
               proposal.refutation.threshold))
    else:
        description = (
            "Power increase for the rule (%d -> %d), which was not distinguished from its null. "
            "This proposes no prediction and can confirm nothing; it exists so that an absence "
            "on this pair can be told apart from a record that could never have spoken."
            % (proposal.antecedent, proposal.consequent))

    return {
        "name": "Follow-up: %s (%s)" % (parent_name, proposal.kind),
        "description": description,
        "parameter_matrix": matrix,
        "pipeline": list(parent.get("pipeline", [])),
        "metadata": {
            "code_revision": parent.get("metadata", {}).get("code_revision", "unknown"),
            "dataset_version": parent.get("metadata", {}).get("dataset_version", "unknown"),
            "parent_experiment_id": parent_id,
            "proposal": proposal.as_record(),
        },
    }


def describe_proposal(proposal: Proposal) -> Dict[str, Any]:
    """The whole proposal as a record, with its boundary and its status attached."""
    if not isinstance(proposal, Proposal):
        raise InvalidParameterError("proposal", type(proposal).__name__, "a Proposal")
    return proposal.as_record()


__all__ = [
    "PROPOSAL_SCHEMA", "REGISTRATION_SCHEMA", "PROPOSAL_CLAIM_BOUNDARY", "OPTIMISER_NOTE",
    "KIND_RE_TEST", "KIND_POWER", "KINDS",
    "PROPOSAL_TESTABLE", "PROPOSAL_UNDERPOWERED", "PROPOSAL_REFUSED",
    "REGISTRATION_DRAFT", "REGISTRATION_REGISTERED",
    "REFUTATION_STATISTIC", "REFUTATION_CONDITION", "CONFIDENCE_RANGE",
    "MAXIMUM_SEARCHED_OCCURRENCES",
    "Prediction", "Refutation", "PowerRequirement", "Proposal", "Registration",
    "separable_at", "required_occurrences", "register",
    "propose_re_test", "propose_power_increase", "followup_experiment_config",
    "describe_proposal",
]
