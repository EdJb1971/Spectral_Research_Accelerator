"""Exact family accounting for a declared search (roadmap TG3.1, rule R18).

**What this module refuses to let happen.** A mining pass enumerates candidate relationships,
tests each one against a surrogate null, and reports what survives. Every enumerated candidate
is a test, and a family of `m` tests corrected under Benjamini-Yekutieli needs a raw p-value
near `alpha / (m * H_m)` before anything can be rejected. A surrogate p-value cannot go below
`1 / (1 + n)`, so for every family size there is an ensemble below which the pass is
*arithmetically incapable* of reporting anything - and it will duly report nothing, which is
indistinguishable from a clean negative result. D8 is the precedent: nine "discoveries" from a
nine-run sweep, including a strong correlation between grid size and reconstruction error.

Rule R18 therefore fixes the family *before* execution. This module is where that happens:

*   `SearchSpecification` declares the search as terms over named, enumerated axes.
*   `SearchSpecification.account()` prices it - family size, the surrogates
    `required_surrogates` demands, and whether the declared ensemble meets them.
*   `SearchSpecification.declare()` is the gate. It raises `FamilyUnaffordableError` when
    `check_power` fails, and the refusal names which R18 remedy applies, with numbers.

**Why the family is enumerated rather than computed from a formula.** `GateProtocol` already
carries `family_size = n_scales * (n_scales - 1) * len(lags)`, which is correct and which
describes exactly one search shape. A constellation sweep over scales x orientations x lags x
representations x motif configurations is a different shape, and a second formula written
beside the first is a second opportunity to be wrong by a factor nobody notices. Here each
combinator both *counts* and *enumerates*, and a test asserts the two agree - so a family size
can be checked against the labels it claims to describe.

**Two counts, and only one of them is the correction unit.** `family_size` is what was
declared. `audit_admissibility` reports how many members a pre-data rule - a lag floor, a
geometry, a support requirement - leaves testable. It deliberately does **not** reduce
`family_size`, because re-pricing a family after screening it is the move R18 exists to
forbid: a family narrowed to what survived is a family chosen after looking.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.registry import Registry
from src.statistics.multiple_comparisons import (
    ASSUMPTIONS, check_power, required_surrogates, resolution_limit,
)

#: The largest family this module will materialise as labels in one call. Counting is cheap
#: and exact at any size; listing a million labels in order to price them is not, and a
#: caller that wants them can ask term by term.
MAX_ENUMERATED = 100_000


# ----------------------------------------------------------------------------------- axes


@dataclass(frozen=True)
class SearchAxis:
    """One declared dimension of a search, with its values written down.

    The values are enumerated rather than described by a count, because a family whose
    members cannot be named cannot be checked against the labels a sweep actually produces -
    and "36" agreeing with "36" for two different reasons is exactly the failure this slice
    exists to prevent.
    """

    name: str
    values: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidParameterError("axis name", self.name, "a non-empty name")
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise InvalidParameterError(
                "axis %r values" % self.name, [],
                "at least one declared value. An empty axis empties the whole family, and "
                "an empty family reports nothing for a reason that is not the data")
        labels = [str(value) for value in self.values]
        if len(set(labels)) != len(labels):
            raise InvalidParameterError(
                "axis %r values" % self.name, labels,
                "distinct values. Members are keyed by label, so a duplicate would collapse "
                "two tests into one and the family would be priced short")

    def __len__(self) -> int:
        return len(self.values)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "n_values": len(self.values),
                "values": [str(value) for value in self.values]}


# --------------------------------------------------------------------------- combinators


@dataclass(frozen=True)
class Combinator:
    """How one term turns its declared axes into members.

    `count` prices a term without building it, which is what lets an unaffordable family be
    refused rather than materialised first. `enumerate` builds the members. The two must
    agree, and `test_family_accounting.py` asserts that they do for every registered entry
    rather than trusting the pair of implementations to stay in step.
    """

    count: Callable[[Sequence[SearchAxis]], int]
    enumerate: Callable[[Sequence[SearchAxis]], Iterable[Tuple[Any, ...]]]
    n_axes: Optional[int]
    components: Callable[[Sequence[SearchAxis]], int]


#: Registered rather than branched (standard E1). A family size is not always a product: the
#: T4C.6 family is *ordered distinct pairs* of three scales crossed with six lags, and no
#: Cartesian rule expresses either the "ordered" or the "distinct" part.
FAMILY_COMBINATORS: Registry[Combinator] = Registry("search family combinator")


def _one_axis(axes: Sequence[SearchAxis], name: str) -> SearchAxis:
    if len(axes) != 1:
        raise InvalidParameterError(
            "%s axes" % name, [axis.name for axis in axes],
            "exactly one axis. This combinator draws both members of each pair from the "
            "same declared set, and two axes would be a different question")
    return axes[0]


FAMILY_COMBINATORS.add(
    "product",
    Combinator(
        count=lambda axes: math.prod(len(axis) for axis in axes),
        enumerate=lambda axes: itertools.product(*[axis.values for axis in axes]),
        n_axes=None,
        components=lambda axes: len(axes),
    ),
    description="The Cartesian product of every declared axis.",
    capabilities={"ordered": False, "distinct": False, "axes": "one or more"},
)

FAMILY_COMBINATORS.add(
    "ordered_pairs",
    Combinator(
        count=lambda axes: len(_one_axis(axes, "ordered_pairs")) * (
            len(_one_axis(axes, "ordered_pairs")) - 1),
        enumerate=lambda axes: (
            (a, b) for a in _one_axis(axes, "ordered_pairs").values
            for b in _one_axis(axes, "ordered_pairs").values if str(a) != str(b)),
        n_axes=1,
        components=lambda axes: 2,
    ),
    description=("Every ordered pair of distinct values from one axis - source and target, "
                 "which are different questions and so are two members."),
    capabilities={"ordered": True, "distinct": True, "axes": 1},
)

FAMILY_COMBINATORS.add(
    "unordered_pairs",
    Combinator(
        count=lambda axes: len(_one_axis(axes, "unordered_pairs")) * (
            len(_one_axis(axes, "unordered_pairs")) - 1) // 2,
        enumerate=lambda axes: itertools.combinations(
            _one_axis(axes, "unordered_pairs").values, 2),
        n_axes=1,
        components=lambda axes: 2,
    ),
    description=("Every unordered pair of distinct values from one axis, for a symmetric "
                 "relation where a-b and b-a are one test rather than two."),
    capabilities={"ordered": False, "distinct": True, "axes": 1},
)


# ---------------------------------------------------------------------------------- terms


@dataclass(frozen=True)
class SearchTerm:
    """One combinator applied to its declared axes."""

    combinator: str
    axes: Tuple[SearchAxis, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        rule = FAMILY_COMBINATORS.entry(self.combinator)
        if not self.axes:
            raise InvalidParameterError(
                "term %r axes" % self.combinator, [], "at least one declared axis")
        expected = rule.value.n_axes
        if expected is not None and len(self.axes) != expected:
            raise InvalidParameterError(
                "term %r axes" % self.combinator, [axis.name for axis in self.axes],
                "exactly %d axis/axes for this combinator" % expected)
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise InvalidParameterError(
                "term %r axes" % self.combinator, names, "distinct axis names within a term")
        # Prices the term now, so a malformed axis combination fails at declaration rather
        # than during the enumeration of a family already believed to be affordable.
        if self.size < 1:
            raise InvalidParameterError(
                "term %r" % self.combinator, [axis.describe() for axis in self.axes],
                "axes that yield at least one member. `ordered_pairs` over a single value "
                "yields none, and a term of size zero empties the whole family: the pass "
                "then reports nothing because there was nothing to test, which is not a "
                "result about the data and must not be reachable as a narrowing remedy")

    @property
    def _rule(self) -> Combinator:
        return FAMILY_COMBINATORS.get(self.combinator)

    @property
    def size(self) -> int:
        return int(self._rule.count(self.axes))

    @property
    def n_components(self) -> int:
        """How many slots this term contributes to each member's component tuple."""
        return int(self._rule.components(self.axes))

    def members(self) -> Tuple[Tuple[Any, ...], ...]:
        return tuple(tuple(member) for member in self._rule.enumerate(self.axes))

    def describe(self) -> Dict[str, Any]:
        return {"combinator": self.combinator, "size": self.size,
                "n_components": self.n_components,
                "axes": [axis.describe() for axis in self.axes]}


# -------------------------------------------------------------------------------- refusal


def max_affordable_family(n_surrogates: int, alpha: float = 0.05,
                          method: str = "benjamini_yekutieli") -> int:
    """The largest family `n_surrogates` can still reject one member of after correction.

    `required_surrogates` is non-decreasing in the family size, so this is a bisection rather
    than a formula, and `test_family_accounting.py` asserts the monotonicity the bisection
    assumes rather than trusting it.
    """
    if required_surrogates(1, alpha, method) > n_surrogates:
        return 0
    low, high = 1, 2
    while required_surrogates(high, alpha, method) <= n_surrogates:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if required_surrogates(middle, alpha, method) <= n_surrogates:
            low = middle
        else:
            high = middle
    return low


class FamilyUnaffordableError(InvalidParameterError):
    """Raised when a declared family cannot be corrected at the declared ensemble size.

    The refusal names both remedies R18 admits, with the arithmetic for each, because
    "reduce the number of tests" without saying by how much is an invitation to reduce it
    until the answer appears.
    """

    def __init__(self, account: "FamilyAccount", remedies: Mapping[str, Any]) -> None:
        narrowing = remedies["preregistered_narrowing"]
        lines = [
            "This search cannot report anything. %d surrogates give a p-value floor of %.4g, "
            "and rejecting one member of a declared family of %d under %s at alpha=%.3g needs "
            "about %d. A pass run in this configuration returns an empty result for an "
            "arithmetic reason rather than a scientific one, and is indistinguishable from a "
            "clean negative."
            % (account.n_surrogates, account.p_value_floor, account.family_size,
               account.correction, account.alpha, account.surrogates_required),
            "Rule R18 admits two remedies, and neither of them is testing fewer members than "
            "were declared.",
            "  (1) preregistered narrowing: at %d surrogates the largest affordable family is "
            "%d, so %d members must leave the declaration *before* the pass runs. %s"
            % (account.n_surrogates, narrowing["max_affordable_family"],
               narrowing["members_to_remove"], narrowing["by_axis_summary"]),
            "  (2) generate/confirm split (TG3.2): mine on train, freeze the declaration under "
            "a content hash, and test the frozen members once on held-out data. This does not "
            "make the family cheaper - it moves the correction onto a smaller confirmatory "
            "family that was fixed before the held-out partition was opened.",
            "Raising the ensemble to %d surrogates also resolves it, at %.1fx the surrogate "
            "cost of the declared %d."
            % (account.surrogates_required,
               account.surrogates_required / float(account.n_surrogates),
               account.n_surrogates),
        ]
        super().__init__(
            "n_surrogates", account.n_surrogates,
            "an ensemble that can resolve the corrected level of the declared family.\n"
            + "\n".join(lines),
            family_size=account.family_size,
            surrogates_required=account.surrogates_required,
            p_value_floor=account.p_value_floor,
            correction=account.correction,
            alpha=account.alpha,
            remedies=dict(remedies),
            specification_sha256=account.specification_sha256)


# -------------------------------------------------------------------------------- account


@dataclass(frozen=True)
class FamilyAccount:
    """What one declared search costs, computed before it runs."""

    family_size: int
    alpha: float
    correction: str
    dependence_assumption: str
    n_surrogates: int
    surrogates_required: int
    p_value_floor: float
    affordable: bool
    power: Mapping[str, Any]
    terms: Tuple[Mapping[str, Any], ...]
    specification_sha256: str
    remedies: Mapping[str, Any] = dc_field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "family_size": self.family_size,
            "alpha": self.alpha,
            "correction": self.correction,
            "dependence_assumption": self.dependence_assumption,
            "n_surrogates": self.n_surrogates,
            "surrogates_required": self.surrogates_required,
            "p_value_floor": self.p_value_floor,
            "affordable": self.affordable,
            "power": dict(self.power),
            "terms": [dict(term) for term in self.terms],
            "specification_sha256": self.specification_sha256,
            "remedies": dict(self.remedies),
            "claim_boundary": (
                "This is an accounting statement about the declared family, not a result. It "
                "says the pass is capable of rejecting a member; it says nothing about "
                "whether any member is true, and affordability is not evidence."),
        }


# -------------------------------------------------------------------------- specification


@dataclass(frozen=True)
class SearchSpecification:
    """A declared search, priced before it is allowed to enumerate anything (R18).

    `label_format` renders one member from its components in declaration order, so a
    specification can be checked against the labels the sweep it describes actually emits.
    Without it the family is a number, and a number cannot be wrong in a way anyone notices.
    """

    terms: Tuple[SearchTerm, ...]
    n_surrogates: int
    alpha: float = 0.05
    correction: str = "benjamini_yekutieli"
    label_format: Optional[str] = None
    study_id: str = ""
    notes: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "notes", dict(self.notes))
        if not self.terms:
            raise InvalidParameterError(
                "terms", [], "at least one term. A search with no declared axes has no "
                             "family, and an undeclared family is what R18 refuses")
        names = [axis.name for term in self.terms for axis in term.axes]
        if len(set(names)) != len(names):
            raise InvalidParameterError(
                "axis names", names,
                "distinct axis names across the whole specification. Two terms over one name "
                "read as one axis in the receipt, and the reader cannot then tell which term "
                "produced which component")
        if self.correction not in ASSUMPTIONS:
            raise InvalidParameterError("correction", self.correction,
                                        "one of %s" % sorted(ASSUMPTIONS))
        if isinstance(self.n_surrogates, bool) or int(self.n_surrogates) != self.n_surrogates \
                or self.n_surrogates < 1:
            raise InvalidParameterError("n_surrogates", self.n_surrogates,
                                        "a positive integer ensemble size")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise InvalidParameterError("alpha", self.alpha, "a probability in (0, 1)")
        if self.label_format is not None:
            try:
                self.label_format.format(*range(self.n_components))
            except (IndexError, KeyError) as exc:
                raise InvalidParameterError(
                    "label_format", self.label_format,
                    "a positional format string over the %d components this specification "
                    "declares" % self.n_components) from exc

    # -------------------------------------------------------------------- enumeration

    @property
    def n_components(self) -> int:
        return sum(term.n_components for term in self.terms)

    @property
    def family_size(self) -> int:
        """The declared family size, counted exactly and without materialising it."""
        return math.prod(term.size for term in self.terms)

    def enumerate_family(self) -> Tuple[Tuple[Any, ...], ...]:
        """Every declared member, as a flat component tuple in declaration order."""
        if self.family_size > MAX_ENUMERATED:
            raise InvalidParameterError(
                "family_size", self.family_size,
                "a family of at most %d members to list in one call. The family is priced "
                "exactly at any size by `family_size`; materialising %d labels in order to "
                "read them is a separate request" % (MAX_ENUMERATED, self.family_size))
        return tuple(
            tuple(itertools.chain.from_iterable(parts))
            for parts in itertools.product(*[term.members() for term in self.terms]))

    def labels(self) -> Tuple[str, ...]:
        """Every declared member's label, for comparison against a sweep's own labels."""
        if self.label_format is None:
            return tuple("|".join(str(part) for part in member)
                         for member in self.enumerate_family())
        return tuple(self.label_format.format(*member)
                     for member in self.enumerate_family())

    # ----------------------------------------------------------------------- identity

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema": "search-specification/v1",
            "study_id": self.study_id,
            "terms": [term.describe() for term in self.terms],
            "n_surrogates": int(self.n_surrogates),
            "alpha": float(self.alpha),
            "correction": self.correction,
            "label_format": self.label_format,
            "notes": dict(self.notes),
        }

    def fingerprint(self) -> str:
        """Content hash of the declaration, so TG3.2 can freeze exactly this search."""
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # --------------------------------------------------------------------- accounting

    def _remedies(self) -> Dict[str, Any]:
        """What R18 permits, with the arithmetic for each - computed, not suggested."""
        ceiling = max_affordable_family(self.n_surrogates, self.alpha, self.correction)
        by_axis: List[Dict[str, Any]] = []
        for term in self.terms:
            others = self.family_size // max(1, term.size)
            for axis in term.axes:
                kept = None
                for candidate in range(len(axis) - 1, 0, -1):
                    try:
                        trimmed = SearchTerm(
                            term.combinator,
                            tuple(SearchAxis(a.name, a.values[:candidate]) if a is axis else a
                                  for a in term.axes))
                    except InvalidParameterError:
                        # The trim empties the term. Reaching the ceiling by emptying the
                        # search is not a remedy, so this axis cannot get there alone.
                        continue
                    if 0 < trimmed.size * others <= ceiling:
                        kept = candidate
                        break
                by_axis.append({
                    "axis": axis.name,
                    "declared_values": len(axis),
                    "largest_affordable_values": kept,
                    "achievable_alone": kept is not None,
                })
        achievable = [record for record in by_axis if record["achievable_alone"]]
        if ceiling == 0:
            summary = ("No family at all is affordable at this ensemble size: even a single "
                       "test needs %d surrogates under %s at alpha=%.3g."
                       % (required_surrogates(1, self.alpha, self.correction),
                          self.correction, self.alpha))
        elif achievable:
            summary = "Narrowing one axis alone is enough: " + "; ".join(
                "%s from %d to %d values" % (record["axis"], record["declared_values"],
                                             record["largest_affordable_values"])
                for record in achievable) + "."
        else:
            summary = ("No single axis can be narrowed far enough on its own; the declaration "
                       "must lose members across more than one axis, or take remedy (2).")
        return {
            "preregistered_narrowing": {
                "max_affordable_family": ceiling,
                "members_to_remove": max(0, self.family_size - ceiling),
                "by_axis": by_axis,
                "by_axis_summary": summary,
            },
            "generate_confirm_split": {
                "available": True,
                "slice": "TG3.2",
                "effect": ("moves the correction onto a confirmatory family frozen under a "
                           "content hash before the held-out partition is opened; it does not "
                           "reduce the family mined on train"),
            },
        }

    def account(self) -> FamilyAccount:
        """Price the declared family. Never raises on affordability - reports it."""
        size = self.family_size
        power = check_power(self.n_surrogates, size, alpha=self.alpha, method=self.correction)
        affordable = bool(power["can_reject_after_correction"])
        return FamilyAccount(
            family_size=size,
            alpha=float(self.alpha),
            correction=self.correction,
            dependence_assumption=ASSUMPTIONS[self.correction],
            n_surrogates=int(self.n_surrogates),
            surrogates_required=int(power["surrogates_required"]),
            p_value_floor=float(resolution_limit(self.n_surrogates)),
            affordable=affordable,
            power=power,
            terms=tuple(term.describe() for term in self.terms),
            specification_sha256=self.fingerprint(),
            remedies={} if affordable else self._remedies(),
        )

    def declare(self) -> FamilyAccount:
        """The R18 gate. Returns the account, or refuses the specification.

        Mining code calls this and nothing else: a pass that cannot reject anything is
        refused before it runs, rather than run and read as a negative result afterwards.
        """
        account = self.account()
        if not account.affordable:
            raise FamilyUnaffordableError(account, self._remedies())
        return account

    # ------------------------------------------------------------------ admissibility

    def audit_admissibility(
        self, predicate: Callable[[Tuple[Any, ...]], bool], *, basis: str,
    ) -> Dict[str, Any]:
        """Which declared members a pre-data rule leaves testable, reported separately.

        A lag below its support floor, a scale with no valid interior, a relation the domain
        has no mechanism for: these are members that were never testable, and knowing how
        many there are before acquisition is worth having. What this must **not** do is
        change `family_size`. Correcting only the members that survived a screen is
        correcting a family chosen after looking, which is the specific move R18 forbids - so
        the declared size stays the correction unit and the shortfall is a fact on the
        receipt beside it.
        """
        if not isinstance(basis, str) or not basis.strip():
            raise InvalidParameterError(
                "basis", basis,
                "a stated basis for the rule. An exclusion whose justification is not "
                "recorded is indistinguishable from one chosen to improve the answer")
        members = self.enumerate_family()
        labels = self.labels()
        admissible: List[str] = []
        inadmissible: List[str] = []
        for member, label in zip(members, labels):
            (admissible if predicate(member) else inadmissible).append(label)
        if not admissible:
            raise InvalidParameterError(
                "admissible members", 0,
                "at least one testable member. This rule (%s) excludes the entire declared "
                "family of %d, so the pass has no possible outcome and its empty result "
                "would say nothing about the data" % (basis, self.family_size))
        return {
            "basis": basis,
            "declared_family_size": self.family_size,
            "n_admissible": len(admissible),
            "n_inadmissible": len(inadmissible),
            "inadmissible_labels": inadmissible,
            "correction_unit": self.family_size,
            "note": ("The declared family remains the correction unit. These members were "
                     "never testable under the stated rule; they are not survivors of a "
                     "screen, and re-pricing the family at %d would be the post-hoc "
                     "narrowing R18 refuses." % len(admissible)),
        }
