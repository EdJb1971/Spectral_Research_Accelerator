"""The whole multi-domain search, priced before anything is acquired (TG17.5, rule R18).

**What this module refuses to let happen.** A four-domain study is a search over domain
combinations, windows, channels, structural scales, relationships, lags, representations and
motifs. Each of those is a knob a researcher can turn while looking at results, and every turn
is a test. TG3.1 already prices a declared family exactly and refuses one the declared ensemble
cannot resolve; what was missing was the step that turns *a manifest* into that declaration.
Until this slice `preflight_manifest` computed its family size from a product written inline —
`pairs x channels x scales x windows x relationships` — which is a second formula beside
`SearchSpecification`, and the family module's own docstring says why that is a mistake: a
second formula is a second opportunity to be wrong by a factor nobody notices. That product is
now built here as declared axes, so the number and the labels it claims to describe are checked
against each other.

**Screening does not shrink the family.** Pairwise screens are useful and are not forbidden.
What is forbidden is pricing the confirmation at the number of candidates that survived them:
the survivors were chosen by looking at the data, so correcting over them is correcting a family
chosen after the fact. `ScreenedSearch` therefore reports the complete declared search as the
correction unit, and `correct_over_candidates` refuses the candidate count *unless* a held-out
partition is named — which is TG3.2's generate/confirm split, where the confirmatory family is
legitimately small because it was frozen before the held-out data was opened.

**Two counts that are not the same question.** `family_size` is what was declared and is the
correction unit. `precedence_availability` reports how many declared members a domain's absent
lag policy leaves untestable. It deliberately does not reduce `family_size`, for exactly the
reason `audit_admissibility` was built: a family narrowed to what survived is a family chosen
after looking. A domain with no justified precedence policy can still take part in structural
association; it makes the precedence members of the family unavailable, and both facts appear on
the receipt.

**What "in human terms" means here.** `family_expansion` renders the multiplication as a
sentence, names what each axis contributes, and prices the same declaration with one more
domain, one more duration, one more scale and one more channel. A researcher deciding whether to
add a fifth domain should be able to see what it costs *before* the freeze, rather than
discovering after acquisition that the study became arithmetically incapable of rejecting
anything. That is the whole of R18 expressed as a decision someone can actually make.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.family import (FamilyAccount, FamilyUnaffordableError, MAX_ENUMERATED,
                             SearchAxis, SearchSpecification, SearchTerm,
                             max_affordable_family)
from src.statistics.multiple_comparisons import check_power, required_surrogates


SCHEMA = "experiment-family/v1"

#: The component order of every member this module enumerates. Fixed and exported, because a
#: predicate over members addresses them by position and a silent reordering would move a
#: precedence audit onto the wrong component.
COMPONENT_ORDER: Tuple[str, ...] = (
    "domain_set", "window", "channel", "scale", "relationship", "lag_seconds",
    "representation", "motif",
)

#: Relationships that read one record as leading another. Separated from `co_occurrence`
#: because rule R21 governs these and not that: a domain with no justified lag policy may
#: still be tested for structural association in the same interval.
PRECEDENCE_BEARING: Tuple[str, ...] = ("precedence", "lead_lag", "causality")

#: How a domain set is written in a member label. One separator, chosen once, so a label can be
#: split back into its domains by a reader and by a test.
DOMAIN_SET_SEPARATOR = "+"


class CandidateCountCorrectionError(InvalidParameterError):
    """Raised when a confirmation is priced at the number of candidates a screen produced."""

    def __init__(self, n_candidates: int, complete_size: int) -> None:
        super().__init__(
            "correction family size", n_candidates,
            "the complete declared search that produced these candidates, which has %d "
            "members, not the %d that survived the screen. The survivors were chosen by "
            "looking at the data: correcting over them prices a family selected after the "
            "fact, and the resulting p-values are the ones a screen was run to obtain. Rule "
            "R18 admits one way to correct over a smaller set — TG3.2's generate/confirm "
            "split, where the confirmatory family is frozen under a content hash *before* a "
            "held-out partition is opened. Name that partition and this call is allowed."
            % (complete_size, n_candidates),
            complete_family_size=complete_size, candidates=n_candidates)


# --------------------------------------------------------------------------- domain sets


def domain_set_labels(domains: Sequence[str], arities: Sequence[int]) -> Tuple[str, ...]:
    """Every declared combination of domains, as sorted labels.

    Arities are a set rather than a single number because a study that tests pairs *and*
    triples has tested both, and a family priced at the pairs alone is short by every triple.
    The combinations of each declared size are unioned, not multiplied: a member is one
    combination, and unioning is what makes the count `sum C(n, k)` rather than a product.
    """
    names = sorted(dict.fromkeys(str(domain) for domain in domains))
    if len(names) < 2:
        raise InvalidParameterError(
            "domains", list(domains),
            "at least two distinct domains. A cross-domain family over one domain has no "
            "member, and an empty family reports nothing for a reason that is not the data")
    sizes = sorted(dict.fromkeys(int(arity) for arity in arities))
    if not sizes:
        raise InvalidParameterError("domain_arities", list(arities),
                                    "at least one declared combination size")
    for size in sizes:
        if size < 2 or size > len(names):
            raise InvalidParameterError(
                "domain_arities", size,
                "a combination size between 2 and the %d declared domains. A size of 1 is a "
                "single-domain question and a size above %d cannot be formed"
                % (len(names), len(names)))
    labels: List[str] = []
    for size in sizes:
        labels.extend(DOMAIN_SET_SEPARATOR.join(combination)
                      for combination in itertools.combinations(names, size))
    return tuple(labels)


def domains_in(label: str) -> Tuple[str, ...]:
    return tuple(label.split(DOMAIN_SET_SEPARATOR))


# ------------------------------------------------------------------------- specification


def _axis_values(spec: Any) -> Dict[str, Tuple[Any, ...]]:
    family = spec.family
    return {
        "domain_set": domain_set_labels(
            [observation.domain for observation in spec.observations],
            getattr(family, "domain_arities", None) or [2]),
        "window": tuple(window.name for window in spec.windows),
        "channel": tuple(family.channels),
        "scale": tuple(float(scale) for scale in family.scales),
        "relationship": tuple(family.relationships),
        "lag_seconds": tuple(float(lag) for lag in (getattr(family, "lags_seconds", None) or [0.0])),
        "representation": tuple(getattr(family, "representations", None) or ["canonical"]),
        "motif": tuple(getattr(family, "motifs", None) or ["none"]),
    }


def declared_ensemble(spec: Any) -> int:
    """The ensemble the declared nulls actually give, which is the smallest of them.

    A study declaring one null at 2,000 replications and another at 200 cannot resolve past
    what the 200 supports for the members that use it, and reporting the larger number would
    describe a resolution the pass does not have for part of its own family.
    """
    replications = [int(null.replications) for null in spec.nulls]
    if not replications:  # pragma: no cover - the manifest requires at least one null
        raise InvalidParameterError("nulls", [], "at least one declared null")
    return min(replications)


def experiment_search_specification(spec: Any, *,
                                    n_surrogates: Optional[int] = None) -> SearchSpecification:
    """The manifest's complete declared search, as one priced specification."""
    values = _axis_values(spec)
    axes = tuple(SearchAxis(name, values[name]) for name in COMPONENT_ORDER)
    return SearchSpecification(
        terms=(SearchTerm("product", axes),),
        n_surrogates=int(n_surrogates if n_surrogates is not None else declared_ensemble(spec)),
        alpha=float(spec.alpha), correction=spec.correction,
        label_format="|".join("{%d}" % index for index in range(len(COMPONENT_ORDER))),
        study_id=spec.study_id,
        notes={"schema": SCHEMA, "mode": spec.mode,
               "manifest_windows": [window.name for window in spec.windows]})


def screening_specification(spec: Any, *,
                            n_surrogates: Optional[int] = None) -> SearchSpecification:
    """The pairwise subset a screen would enumerate. Never the correction unit."""
    values = dict(_axis_values(spec))
    pairs = tuple(label for label in values["domain_set"] if len(domains_in(label)) == 2)
    if not pairs:
        raise InvalidParameterError(
            "domain_arities", list(getattr(spec.family, "domain_arities", []) or []),
            "a declaration that includes pairs, for a pairwise screen to enumerate")
    values["domain_set"] = pairs
    axes = tuple(SearchAxis(name, values[name]) for name in COMPONENT_ORDER)
    return SearchSpecification(
        terms=(SearchTerm("product", axes),),
        n_surrogates=int(n_surrogates if n_surrogates is not None else declared_ensemble(spec)),
        alpha=float(spec.alpha), correction=spec.correction,
        label_format="|".join("{%d}" % index for index in range(len(COMPONENT_ORDER))),
        study_id=spec.study_id,
        notes={"schema": SCHEMA, "mode": spec.mode, "stage": "screen"})


# ------------------------------------------------------------------- screen and confirm


@dataclass(frozen=True)
class ScreenedSearch:
    """A pairwise screen and the complete search it lives inside.

    Holding both is the point. A screen reported on its own looks like a small, affordable
    study; the same screen reported beside the family it was drawn from shows what the pass
    actually searched, which is the number the confirmation has to answer to.
    """

    screening: SearchSpecification
    complete: SearchSpecification

    @property
    def correction_unit(self) -> int:
        return self.complete.family_size

    def account(self) -> FamilyAccount:
        return self.complete.account()

    def correct_over_candidates(self, n_candidates: int, *,
                                held_out: Optional[str] = None) -> int:
        """The size a confirmation may correct over, or a refusal naming why not."""
        if not isinstance(n_candidates, int) or isinstance(n_candidates, bool) \
                or n_candidates < 0:
            raise InvalidParameterError("n_candidates", n_candidates,
                                        "a non-negative count of surviving candidates")
        if held_out is None:
            if n_candidates < self.correction_unit:
                raise CandidateCountCorrectionError(n_candidates, self.correction_unit)
            return self.correction_unit
        if not str(held_out).strip():
            raise InvalidParameterError(
                "held_out", held_out,
                "a named partition. An unnamed held-out set cannot be shown to have been "
                "closed while the candidates were chosen, which is the only thing it claims")
        return int(n_candidates)

    def describe(self) -> Dict[str, Any]:
        return {
            "screen_family_size": self.screening.family_size,
            "complete_family_size": self.complete.family_size,
            "correction_unit": self.correction_unit,
            "screen_sha256": self.screening.fingerprint(),
            "complete_sha256": self.complete.fingerprint(),
            "rule": ("A pairwise screen may generate candidates. The confirmation corrects "
                     "over the complete search that produced them, unless a held-out "
                     "partition frozen before the screen ran is named (TG3.2)."),
        }


def screened_search(spec: Any, *, n_surrogates: Optional[int] = None) -> ScreenedSearch:
    return ScreenedSearch(screening=screening_specification(spec, n_surrogates=n_surrogates),
                          complete=experiment_search_specification(spec,
                                                                   n_surrogates=n_surrogates))


# ----------------------------------------------------------------------------- precedence


def precedence_availability(spec: Any, *,
                            admissible: Optional[Mapping[str, bool]] = None) -> Dict[str, Any]:
    """Which declared members a domain's absent lag policy leaves untestable.

    Reported beside the family rather than subtracted from it. A study whose precedence members
    are unavailable has a smaller set of answerable questions and exactly the same number of
    declared tests, and conflating those two is how a family shrinks to fit its result.
    """
    if admissible is None:
        from src.adapters import register_all_adapters
        from src.core.experiment_adapter import adapter_for_domain

        register_all_adapters()
        admissible = {}
        for observation in spec.observations:
            try:
                adapter = adapter_for_domain(observation.domain)
            except Exception:  # noqa: BLE001 - an unregistered domain refuses in preflight
                admissible[observation.domain] = False
                continue
            admissible[observation.domain] = bool(
                adapter.declaration.precedence_admissible)
    refusing = sorted(domain for domain, allowed in admissible.items() if not allowed)
    search = experiment_search_specification(spec)
    relationships = tuple(spec.family.relationships)
    precedence_declared = tuple(name for name in relationships if name in PRECEDENCE_BEARING)

    def testable(member: Tuple[Any, ...]) -> bool:
        if member[COMPONENT_ORDER.index("relationship")] not in PRECEDENCE_BEARING:
            return True
        return not any(domain in refusing
                       for domain in domains_in(member[COMPONENT_ORDER.index("domain_set")]))

    basis = ("rule R21: a precedence reading needs a justified lag policy, and %s declare none"
             % (", ".join(refusing) or "no domains"))
    audit: Dict[str, Any] = {}
    entirely_unavailable = False
    if precedence_declared and search.family_size <= MAX_ENUMERATED:
        try:
            audit = search.audit_admissibility(testable, basis=basis)
        except InvalidParameterError:
            # Every declared member is a precedence reading over a domain that cannot carry
            # one. The audit refuses to report a family with nothing left in it, and so does
            # this: the study has no answerable question, which is a fact about the
            # declaration and not a result about the data.
            entirely_unavailable = True
    unavailable = (search.family_size if entirely_unavailable
                   else int(audit.get("n_inadmissible", 0)))
    return {
        "schema": SCHEMA,
        "declared_family_size": search.family_size,
        "precedence_relationships_declared": list(precedence_declared),
        "domains_without_precedence_policy": refusing,
        "unavailable_precedence_members": unavailable,
        "association_members_unaffected": search.family_size - unavailable,
        "family_size_unchanged": True,
        "entire_family_unavailable": entirely_unavailable,
        "basis": basis,
        "note": ("A domain with no justified precedence policy still takes part in structural "
                 "association. It makes the precedence members of the family unavailable, and "
                 "the declared family size stays the correction unit either way."),
    }


# ------------------------------------------------------------------------ correction unit


def correction_plan(spec: Any, *, n_surrogates: Optional[int] = None) -> Dict[str, Any]:
    """What R18 is applied to, which is not always the size of the declared search.

    A `confirmatory_only` study corrects over everything it declared. A study taking TG3.2's
    generate/confirm split corrects over the confirmatory family it named, and its generate
    stage produces candidates rather than claims — an uncorrected pass whose selection used the
    data, which is a description this report states rather than leaves to the reader. The split
    is not a discount: the complete search is still enumerated, still costs what it costs, and
    is still the number reported beside the confirmatory one.
    """
    ensemble = int(n_surrogates if n_surrogates is not None else declared_ensemble(spec))
    search = experiment_search_specification(spec, n_surrogates=ensemble)
    policy = getattr(spec, "confirmation", None)
    stage = getattr(policy, "stage", "confirmatory_only")
    declared = search.family_size
    if stage == "generate_then_confirm":
        unit = int(policy.confirmatory_members or 0)
        if unit > declared:
            raise InvalidParameterError(
                "confirmatory_members", unit,
                "a confirmatory family inside the declared search of %d members. Confirming a "
                "member that was never generated is a fresh search on held-out data under the "
                "name of a confirmation" % declared)
    else:
        unit = declared
    power = check_power(ensemble, unit, alpha=float(spec.alpha), method=spec.correction)
    return {
        "stage": stage,
        "declared_search_members": declared,
        "correction_unit_members": unit,
        "held_out_partition": getattr(policy, "held_out_partition", None),
        "n_surrogates": ensemble,
        "surrogates_required": int(power["surrogates_required"]),
        "p_value_floor": float(power["p_value_floor"]),
        "affordable": bool(power["can_reject_after_correction"]),
        "warning": power["warning"],
        "generate_stage_makes_claims": False if stage == "generate_then_confirm" else None,
        "note": ("The generate stage enumerates and tests all %d declared members on the "
                 "training partition. Its p-values are uncorrected and its selection used the "
                 "data, so its output is a list of candidates and not a result. Only the %d "
                 "confirmatory members, frozen before %s was opened, are corrected and "
                 "claimed." % (declared, unit, policy.held_out_partition)
                 if stage == "generate_then_confirm" else
                 "Every declared member is corrected over, because every one of them is "
                 "claimed."),
    }


# ------------------------------------------------------------------------------ expansion


def _sentence(values: Mapping[str, Tuple[Any, ...]], total: int) -> str:
    parts = []
    for name in COMPONENT_ORDER:
        count = len(values[name])
        if count == 1 and name in ("lag_seconds", "representation", "motif"):
            continue
        parts.append("%d %s%s" % (count, name.replace("_", " "), "" if count == 1 else "s"))
    return "%s = %d declared tests." % (" x ".join(parts), total)


def _delta(spec: Any, values: Mapping[str, Tuple[Any, ...]], axis: str,
           n_surrogates: int, alpha: float, correction: str) -> Dict[str, Any]:
    """What one more value on this axis would cost, priced now rather than after the freeze."""
    counts = {name: len(items) for name, items in values.items()}
    before = math.prod(counts.values())
    if axis == "domain_set":
        domains = sorted({observation.domain for observation in spec.observations})
        arities = getattr(spec.family, "domain_arities", None) or [2]
        after_sets = len(domain_set_labels(domains + ["one_more_domain"], arities))
        counts["domain_set"] = after_sets
    else:
        counts[axis] = counts[axis] + 1
    after = math.prod(counts.values())
    return {
        "axis": axis,
        "adds": ("a fifth domain" if axis == "domain_set"
                 else "one more %s" % axis.replace("_", " ")),
        "family_size_before": before,
        "family_size_after": after,
        "members_added": after - before,
        "surrogates_required_before": required_surrogates(before, alpha, correction),
        "surrogates_required_after": required_surrogates(after, alpha, correction),
        "declared_surrogates": n_surrogates,
        "affordable_after": required_surrogates(after, alpha, correction) <= n_surrogates,
    }


def family_expansion(spec: Any, *, n_surrogates: Optional[int] = None) -> Dict[str, Any]:
    """The declared multi-domain family, priced and written out in human terms.

    Everything here is computed from the manifest before acquisition, which is the point: the
    resolution and power check that decides whether this study can reject anything must happen
    while the answer is still "declare something else", not after four archives have been read.
    """
    ensemble = int(n_surrogates if n_surrogates is not None else declared_ensemble(spec))
    values = _axis_values(spec)
    search = experiment_search_specification(spec, n_surrogates=ensemble)
    account = search.account()
    ceiling = max_affordable_family(ensemble, float(spec.alpha), spec.correction)
    axes = [{"axis": name, "declared_values": len(values[name]),
             "examples": [str(value) for value in values[name][:4]],
             "contributes": "x%d" % len(values[name])}
            for name in COMPONENT_ORDER]
    return {
        "schema": SCHEMA,
        "study_id": spec.study_id,
        "mode": spec.mode,
        "axes": axes,
        "in_human_terms": _sentence(values, search.family_size),
        "family_size": search.family_size,
        "specification_sha256": search.fingerprint(),
        "declared_surrogates": ensemble,
        "declared_surrogates_basis": ("the smallest declared null replication count, which is "
                                      "the resolution the whole family actually has"),
        "account": account.describe(),
        "correction": correction_plan(spec, n_surrogates=ensemble),
        "largest_affordable_family": ceiling,
        "resource_requirement": {
            "surrogate_evaluations_declared": search.family_size * ensemble,
            "surrogate_evaluations_required": (search.family_size
                                               * account.surrogates_required),
            "note": ("One evaluation is one member measured against one surrogate. This is "
                     "the arithmetic cost of the declaration, not a runtime estimate."),
        },
        "expansion_cost": [
            _delta(spec, values, axis, ensemble, float(spec.alpha), spec.correction)
            for axis in ("domain_set", "window", "scale", "channel")],
        "screen_and_confirm": screened_search(spec, n_surrogates=ensemble).describe(),
        "precedence": precedence_availability(spec),
        "claim_boundary": ("This is an accounting statement about a declared search, computed "
                           "before acquisition. An affordable family is not evidence, and a "
                           "family that is affordable is not thereby a good question."),
    }


def assert_family_affordable(spec: Any, *, n_surrogates: Optional[int] = None) -> Dict[str, Any]:
    """The R18 gate over a whole manifest: refuse the search before anything is acquired."""
    plan = correction_plan(spec, n_surrogates=n_surrogates)
    if not plan["affordable"]:
        if plan["stage"] == "confirmatory_only":
            # `declare()` raises `FamilyUnaffordableError` with both R18 remedies priced.
            experiment_search_specification(spec, n_surrogates=plan["n_surrogates"]).declare()
        raise InvalidParameterError(
            "confirmatory_members", plan["correction_unit_members"],
            "a confirmatory family this ensemble can resolve. %s" % plan["warning"])
    return plan


__all__ = ["SCHEMA", "COMPONENT_ORDER", "PRECEDENCE_BEARING", "DOMAIN_SET_SEPARATOR",
           "correction_plan",
           "CandidateCountCorrectionError", "ScreenedSearch", "assert_family_affordable",
           "declared_ensemble", "domain_set_labels", "domains_in", "experiment_search_specification",
           "family_expansion", "precedence_availability", "screened_search",
           "screening_specification", "FamilyUnaffordableError"]
