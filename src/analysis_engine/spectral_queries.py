"""Bidirectional queries: one engine, two directions, and no second test (`T4F.4`).

T4F.3 tested a declared family and paid the correction on it. This module asks the two
questions the roadmap words as top-down and bottom-up -- *given a coarse structure, what finer
configurations preceded it?* and *given a fine configuration, what coarser structure follows?* --
and it asks them **of the report that already exists**, not of the record.

That is the whole design, and it is a scientific choice rather than an implementation
convenience. Four things follow from it, and each is a way this step lies if it is not done.

**1. A query is a view, and a view is not a test.** The tempting shortcut is to re-run the
inference restricted to the pattern being asked about: one target, a handful of counterparts, a
family of six instead of ninety, and every q-value falls. That is choosing the family after
seeing the data, and it is the failure the per-lag family exists to avoid one level down --
`precursor_report` reports every member of the declared family precisely so that no selection
precedes the correction. So nothing here recomputes a p-value, a q-value or a status. The rows
are read out of the report exactly as the report wrote them, the family the correction was paid
on is published beside every answer, and a caller who genuinely wants a narrower family must
declare it *before* the record is read, by passing `pairs=` to `precursor_report`.

**2. Coarse and fine have to be measured, and the record is allowed to refuse them.** The
direction words are meaningless without an ordering on scale, and the ordering is not the
programme's to assume. `scale_ordering` builds it from the catalogue's own member scales in the
catalogue's own units, and it is an **interval order**: a pattern occupies the range of scales
its members actually span, and one pattern is finer than another only when those ranges are
disjoint. Two patterns whose scale ranges overlap are not orderable, and saying which of them is
coarser would be an arrow taken from a sort rather than from the record -- the same fabrication
`spectral_events` refuses when it publishes simultaneous events unordered. The ordering is
therefore partial by construction, and the number of pairs it cannot order is published rather
than hidden by a tie-break.

Under T4E.2's scale-invariant mode there is no ordering at all, and the refusal is exact rather
than cautious: `SignaturePoint.from_signature` divides that mode's scales by their own geometric
mean, so every signature's geometric-mean scale is exactly 1.0. An ordering built on it would
order floating-point residue. That mode buys cross-domain comparability by discarding absolute
scale, and a top-down query is a question about absolute scale; the two cannot both be had, and
the module says which one was asked for.

**Cardinality is not scale.** A constellation of five nodes is not coarser than one of three; it
is bigger. The ordering reads scales and nothing else, and the suite holds it to that on a
record where the two disagree.

**3. Both directions are the same engine, and neither is the other reversed.** Top-down fixes
the target as the consequent and requires the counterpart to be finer; bottom-up fixes it as the
antecedent and requires the counterpart to be coarser. That is the only difference, and it is
two arguments to one selection rather than two code paths -- so a fix to one direction cannot
fail to reach the other. What is *not* true is that one direction's answer can be read off the
other's: confidence has a different denominator each way, so `A -> B` and `B -> A` are two
distinct hypotheses with two distinct lifts, both of them in the declared family and both of
them corrected.

**4. The ranking is a scientific choice, and the obvious one is the wrong one.** The roadmap
words the top-down question as *what most commonly preceded it*, and answering that literally --
ranking by how often the counterpart occurred -- returns the record's commonest pattern
whatever it precedes. That is the same error as dividing a window confidence by a per-frame base
rate, one level up. Every ranking key is therefore available, every entry carries its rank under
*every* key, each key that can mislead carries the sentence saying how, and the result publishes
whether the keys disagree about what comes first. On the acceptance record they do: ranked by
frequency the first answer is a pattern the null did not distinguish.

**Every considered row is returned, including the ones that found nothing.** An answer of three
signatures out of four candidates and an answer of three out of forty are different answers, and
a query that returned only the rejections would make them look identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from src.analysis_engine.spectral_clustering import PatternCatalogue
from src.analysis_engine.spectral_precursors import (
    PRECURSOR_CLAIM_BOUNDARY, PrecursorReport, RULE_BASE_RATE_ZERO,
    RULE_NO_ELIGIBLE_ANTECEDENT, RULE_PRECURSOR, RULE_UNDECIDABLE, SelectedLagRule,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.errors import InvalidParameterError


QUERY_SCHEMA = "spectral-bidirectional-query/v1"
SCALE_ORDER_SCHEMA = "spectral-scale-ordering/v1"

#: Given a coarse structure, which finer configurations preceded it.
DIRECTION_TOP_DOWN = "top_down"
#: Given a fine configuration, which coarser structure followed it.
DIRECTION_BOTTOM_UP = "bottom_up"
DIRECTIONS = (DIRECTION_TOP_DOWN, DIRECTION_BOTTOM_UP)

#: The role the queried pattern plays in the rule, per direction. This mapping *is* the
#: difference between the two directions; there is no second selection path.
DIRECTION_TARGET_ROLE: Mapping[str, str] = {
    DIRECTION_TOP_DOWN: "consequent",
    DIRECTION_BOTTOM_UP: "antecedent",
}

#: Where the counterpart must sit in the scale ordering, per direction.
DIRECTION_COUNTERPART_SIDE: Mapping[str, str] = {
    DIRECTION_TOP_DOWN: "finer",
    DIRECTION_BOTTOM_UP: "coarser",
}

DIRECTION_QUESTION: Mapping[str, str] = {
    DIRECTION_TOP_DOWN: (
        "given this structure, which strictly finer configurations preceded it within a "
        "declared lag window?"),
    DIRECTION_BOTTOM_UP: (
        "given this configuration, which strictly coarser structures followed it within a "
        "declared lag window?"),
}

#: One hypothesis per (pair, lag): the full six figures, but a counterpart appears once per lag.
FAMILY_PER_LAG = "per_lag"
#: One row per pair, already priced for the lag search: no support and no interval on it.
FAMILY_SELECTED_LAG = "selected_lag"
FAMILIES = (FAMILY_PER_LAG, FAMILY_SELECTED_LAG)

RANK_Q_VALUE = "q_value"
RANK_LIFT = "lift"
RANK_CONFIDENCE = "confidence"
RANK_SUPPORT = "support"
RANK_KEYS = (RANK_Q_VALUE, RANK_LIFT, RANK_CONFIDENCE, RANK_SUPPORT)

#: True where a smaller number is the stronger answer.
RANK_ASCENDING: Mapping[str, bool] = {
    RANK_Q_VALUE: True, RANK_LIFT: False, RANK_CONFIDENCE: False, RANK_SUPPORT: False,
}

RANK_CAUTION: Mapping[str, Optional[str]] = {
    RANK_Q_VALUE: None,
    RANK_LIFT: (
        "Lift rewards a rare consequent, so its leading entry can rest on a handful of "
        "eligible trials. Read it with lift_interval and with support, not on its own."),
    RANK_CONFIDENCE: (
        "Confidence is not referenced to anything. A consequent that is common in this record "
        "scores high behind every antecedent, including antecedents the null did not "
        "distinguish from chance."),
    RANK_SUPPORT: (
        "This is the roadmap's phrase 'most commonly preceded it' taken literally, and it "
        "ranks the record's commonest pattern first whatever it precedes. It is published "
        "because it is the obvious question, and it is not a measure of association: read "
        "which of these entries the null actually distinguished before reading the order."),
}

#: At least one orderable counterpart was distinguished from the null.
ANSWER_SIGNATURES = "PRECURSOR_SIGNATURES_FOUND"
#: Counterparts were tested on the right side of the ordering and none was distinguished.
ANSWER_NONE_DISTINGUISHED = "NO_COUNTERPART_DISTINGUISHED_FROM_NULL"
#: Rules named the target, but no counterpart's scale range sat wholly on the required side.
ANSWER_NONE_ORDERABLE = "NO_COUNTERPART_ORDERABLE_ON_THE_REQUIRED_SIDE"
#: Counterparts were on the right side and not one of them could be measured at all.
ANSWER_NOT_MEASURABLE = "NOTHING_ON_THE_REQUIRED_SIDE_WAS_MEASURABLE"

#: Row statuses that carry no measurement, so no ranking key has a value on them.
UNMEASURED_STATUSES = (RULE_NO_ELIGIBLE_ANTECEDENT, RULE_BASE_RATE_ZERO, RULE_UNDECIDABLE)

SCALE_STATISTIC = "geometric mean of every member node's scale, in the catalogue's own units"

SCALE_ORDER_BASIS = (
    "A pattern occupies the range of scales its members actually span, and one pattern is finer "
    "than another only when those ranges are disjoint. Patterns whose ranges overlap are not "
    "orderable and are reported as such: naming one of them the coarser would be an ordering "
    "taken from a sort rather than from the record. The relation is therefore partial, and "
    "n_unorderable_pairs says how partial.")

QUERY_CLAIM_BOUNDARY = (
    "A query is a view of a report that already exists and it is not a second test. No p-value, "
    "q-value or status here was recomputed: every one of them was corrected against the family "
    "named in correction_family, which is the family declared before the record was read, and "
    "narrowing the question to one target afterwards does not narrow that family or raise any "
    "answer's significance. A returned entry is a precursor signature under the report's own "
    "null and nothing above it: not a cause, a driver, a mechanism, a trigger or a forecast, "
    "and no intervention is implied (R7). The direction words are about scale and not about "
    "influence -- 'top-down' names a query whose target is the coarser of the two patterns, "
    "and asserts nothing about which of them acts on the other. The ordering is measured in "
    "this record's own scale units and stops at the domain boundary, and the ranking is a "
    "choice: entries carry their rank under every key because the keys disagree.")


# --------------------------------------------------------------------------------------------
# the ordering that gives 'coarse' and 'fine' a meaning
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleOrdering:
    """A partial order on patterns, read off the scales their members actually occupied."""

    scale_units: str
    mode: str
    statistic: Mapping[int, float]
    low: Mapping[int, float]
    high: Mapping[int, float]

    def __post_init__(self) -> None:
        if set(self.statistic) != set(self.low) or set(self.low) != set(self.high):
            raise InvalidParameterError(
                "ScaleOrdering", sorted(set(self.statistic) ^ set(self.low)),
                "one statistic, one low and one high per pattern")

    @property
    def patterns(self) -> Tuple[int, ...]:
        return tuple(sorted(self.statistic))

    def __contains__(self, pattern_id: Any) -> bool:
        return int(pattern_id) in self.statistic

    def _known(self, pattern_id: int) -> int:
        value = int(pattern_id)
        if value not in self.statistic:
            raise InvalidParameterError(
                "pattern_id", value,
                "a pattern this ordering carries a scale range for (%s)" % (self.patterns,))
        return value

    def finer_than(self, pattern_id: int, other_id: int) -> bool:
        """Strictly finer: the whole of one range sits below the whole of the other."""
        left = self._known(pattern_id)
        right = self._known(other_id)
        return self.high[left] < self.low[right]

    def coarser_than(self, pattern_id: int, other_id: int) -> bool:
        return self.finer_than(other_id, pattern_id)

    def orderable(self, pattern_id: int, other_id: int) -> bool:
        return (self.finer_than(pattern_id, other_id)
                or self.finer_than(other_id, pattern_id))

    def separation(self, pattern_id: int, other_id: int) -> Optional[float]:
        """The gap between two disjoint ranges, or `None` where they overlap."""
        left = self._known(pattern_id)
        right = self._known(other_id)
        if self.finer_than(left, right):
            return float(self.low[right] - self.high[left])
        if self.finer_than(right, left):
            return float(self.low[left] - self.high[right])
        return None

    @property
    def unorderable_pairs(self) -> Tuple[Tuple[int, int], ...]:
        ids = self.patterns
        return tuple((left, right) for index, left in enumerate(ids)
                     for right in ids[index + 1:] if not self.orderable(left, right))

    def describe(self) -> Dict[str, Any]:
        ids = self.patterns
        return {
            "schema": SCALE_ORDER_SCHEMA,
            "mode": self.mode,
            "scale_units": self.scale_units,
            "statistic": SCALE_STATISTIC,
            "basis": SCALE_ORDER_BASIS,
            "n_patterns": len(ids),
            "n_pairs": len(ids) * (len(ids) - 1) // 2,
            "n_unorderable_pairs": len(self.unorderable_pairs),
            "unorderable_pairs": [list(pair) for pair in self.unorderable_pairs],
            "patterns": [{"pattern_id": pattern_id,
                          "scale": self.statistic[pattern_id],
                          "scale_low": self.low[pattern_id],
                          "scale_high": self.high[pattern_id]} for pattern_id in ids],
        }


def scale_ordering(catalogue: PatternCatalogue) -> ScaleOrdering:
    """Read coarse and fine off a catalogue's own scales, or refuse to invent them."""
    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a T4E.3 PatternCatalogue")
    if not catalogue.patterns:
        raise InvalidParameterError(
            "catalogue.patterns", 0,
            "at least one pattern. An empty catalogue is an unrun clustering, and an empty "
            "ordering would answer every direction query with a silent nothing")

    modes = {member.mode for pattern in catalogue for member in pattern.members}
    if "scale_invariant" in modes:
        raise InvalidParameterError(
            "catalogue.patterns.members.mode", sorted(modes),
            "the scale-specific mode. Under scale invariance every signature's scales are "
            "divided by their own geometric mean, so that statistic is exactly 1.0 for every "
            "pattern in the catalogue and an ordering built on it would order floating-point "
            "residue. That mode buys cross-domain comparability by discarding absolute scale, "
            "and top-down and bottom-up are questions about absolute scale: run the query on a "
            "scale-specific catalogue, or ask a question that does not name a direction")
    if len(modes) > 1:
        raise InvalidParameterError(
            "catalogue.patterns.members.mode", sorted(modes),
            "one scale mode across the catalogue")

    units = {member.scale_units for pattern in catalogue for member in pattern.members}
    if None in units:
        raise InvalidParameterError(
            "catalogue.patterns.members.scale_units", None,
            "a declared scale unit on every member. 'Coarser' is a comparison of magnitudes, "
            "and magnitudes without a unit cannot be compared")
    if len(units) > 1:
        raise InvalidParameterError(
            "catalogue.patterns.members.scale_units", sorted(units),
            "one scale unit across the catalogue. Two units are two orderings, and ranking "
            "them together would be arithmetic across a conversion nobody performed")

    statistic: Dict[int, float] = {}
    low: Dict[int, float] = {}
    high: Dict[int, float] = {}
    for pattern in catalogue:
        values: List[float] = []
        for member in pattern.members:
            for scale in member.scales:
                value = float(scale)
                if not math.isfinite(value) or value <= 0.0:
                    raise InvalidParameterError(
                        "catalogue.patterns.members.scales", scale,
                        "a positive finite scale. A geometric mean is not defined otherwise, "
                        "and a non-positive scale is not a magnitude")
                values.append(value)
        if not values:
            raise InvalidParameterError(
                "catalogue.patterns.members.scales", 0,
                "at least one scale value in pattern %d" % pattern.pattern_id)
        key = int(pattern.pattern_id)
        statistic[key] = math.exp(sum(math.log(v) for v in values) / len(values))
        low[key] = min(values)
        high[key] = max(values)

    return ScaleOrdering(scale_units=str(sorted(units)[0]), mode=sorted(modes)[0],
                         statistic=statistic, low=low, high=high)


# --------------------------------------------------------------------------------------------
# one row shape, so that both directions and both families are one engine
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryEntry:
    """One row of the report, seen from the target's side, with its rank under every key."""

    antecedent: int
    consequent: int
    counterpart: int
    window: TransitionWindow
    status: str

    support: Optional[int]
    confidence: Optional[float]
    lift: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float]

    counterpart_scale: float
    counterpart_scale_range: Tuple[float, float]
    scale_separation: float
    ranks: Mapping[str, Optional[int]]
    #: The report's own object, unaltered, so nothing this view drops is lost.
    row: Any

    @property
    def rejected(self) -> bool:
        return self.status == RULE_PRECURSOR

    @property
    def measured(self) -> bool:
        return self.status not in UNMEASURED_STATUSES

    def describe(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "counterpart": self.counterpart,
            "window": self.window.describe(),
            "status": self.status,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "counterpart_scale": self.counterpart_scale,
            "counterpart_scale_range": list(self.counterpart_scale_range),
            "scale_separation": self.scale_separation,
            "ranks": dict(self.ranks),
        }


@dataclass(frozen=True)
class QueryResult:
    """What the report already says about one pattern, in one direction, under one ranking."""

    direction: str
    target: int
    family: str
    rank_by: str
    status: str
    entries: Tuple[QueryEntry, ...]

    target_scale: float
    target_scale_range: Tuple[float, float]
    scale_units: str

    rows_in_family: int
    rows_naming_target: int
    withheld_overlapping_scale: int
    withheld_wrong_side: int
    unmeasured_entries: int

    correction_family: Mapping[str, Any]
    top_by: Mapping[str, Optional[Tuple[int, float, float]]]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[QueryEntry]:
        return iter(self.entries)

    @property
    def signatures(self) -> Tuple[QueryEntry, ...]:
        return tuple(entry for entry in self.entries if entry.rejected)

    @property
    def rankings_disagree(self) -> bool:
        """Do the ranking keys disagree about which answer comes first?"""
        leaders = {value for value in self.top_by.values() if value is not None}
        return len(leaders) > 1

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": QUERY_SCHEMA,
            "direction": self.direction,
            "question": DIRECTION_QUESTION[self.direction],
            "target": self.target,
            "target_role": DIRECTION_TARGET_ROLE[self.direction],
            "counterpart_side": DIRECTION_COUNTERPART_SIDE[self.direction],
            "target_scale": self.target_scale,
            "target_scale_range": list(self.target_scale_range),
            "scale_units": self.scale_units,
            "family": self.family,
            "rank_by": self.rank_by,
            "ranking_caution": RANK_CAUTION[self.rank_by],
            "ranking_cautions": {key: RANK_CAUTION[key] for key in RANK_KEYS},
            "rankings_disagree": self.rankings_disagree,
            "top_by": {key: (None if value is None else list(value))
                       for key, value in self.top_by.items()},
            "status": self.status,
            "n_entries": len(self.entries),
            "n_signatures": len(self.signatures),
            "rows_in_family": self.rows_in_family,
            "rows_naming_target": self.rows_naming_target,
            "withheld_overlapping_scale": self.withheld_overlapping_scale,
            "withheld_wrong_side": self.withheld_wrong_side,
            "unmeasured_entries": self.unmeasured_entries,
            "correction_family": dict(self.correction_family),
            "entries": [entry.describe() for entry in self.entries],
            "claim_boundary": QUERY_CLAIM_BOUNDARY,
            "report_claim_boundary": PRECURSOR_CLAIM_BOUNDARY,
        }


def _rows(report: PrecursorReport, family: str) -> Tuple[Any, ...]:
    return tuple(report.rules) if family == FAMILY_PER_LAG else tuple(report.selected)


def _figure(row: Any, key: str) -> Optional[float]:
    """The one place a row's ranking values are read, so both families read alike.

    A row whose status says nothing could be measured has no value under any key, including
    support: its support is zero because there was nothing to count, and ranking that zero
    against a measured one would put a censored antecedent in a league table of associations.
    """
    if row.status in UNMEASURED_STATUSES:
        return None
    if key == RANK_SUPPORT:
        return None if isinstance(row, SelectedLagRule) else float(row.support)
    return getattr(row, key)


def _identity(item: Tuple[int, TransitionWindow, Any]) -> Tuple[int, float, float]:
    counterpart, window, _row = item
    return (int(counterpart), float(window.minimum_lag), float(window.maximum_lag))


def bidirectional_query(report: PrecursorReport, *, direction: str, target: int,
                        ordering: ScaleOrdering, family: str = FAMILY_PER_LAG,
                        rank_by: str = RANK_Q_VALUE) -> QueryResult:
    """Read one pattern's side of a completed report, in one declared direction.

    Nothing is recomputed. The rows, their p-values, their q-values and their statuses are the
    report's own, corrected against the family the report declared; this function selects,
    orders and publishes them, and refuses the questions the report cannot answer.
    """
    if not isinstance(report, PrecursorReport):
        raise InvalidParameterError(
            "report", type(report).__name__,
            "a completed T4F.3 PrecursorReport. A query is a view of a test that has already "
            "been corrected, and re-running the inference for one target would choose the "
            "family after seeing the data")
    if direction not in DIRECTIONS:
        raise InvalidParameterError(
            "direction", direction,
            "one of %s, declared. There is no default direction: 'what preceded this' and "
            "'what follows this' are two different hypotheses about the same pair, and a "
            "silently chosen one would answer a question nobody asked" % (list(DIRECTIONS),))
    if family not in FAMILIES:
        raise InvalidParameterError("family", family, "one of %s" % (list(FAMILIES),))
    if rank_by not in RANK_KEYS:
        raise InvalidParameterError("rank_by", rank_by, "one of %s" % (list(RANK_KEYS),))
    if not isinstance(ordering, ScaleOrdering):
        raise InvalidParameterError(
            "ordering", type(ordering).__name__,
            "a measured ScaleOrdering. Coarse and fine are properties of this record's "
            "catalogue, not of the pattern identifiers")
    if family == FAMILY_SELECTED_LAG and rank_by == RANK_SUPPORT:
        raise InvalidParameterError(
            "rank_by", rank_by,
            "a key the selected-lag family carries. A selected-lag row is one pair's strongest "
            "declared lag and it does not carry a support count, because the count belongs to "
            "the lag that won rather than to the pair. Rank the per-lag family by support, or "
            "rank this one by q_value")

    rows = _rows(report, family)
    if not rows:
        raise InvalidParameterError(
            "report.%s" % ("rules" if family == FAMILY_PER_LAG else "selected"), 0,
            "a report containing at least one row of the requested family")

    tested = {int(row.antecedent) for row in rows} | {int(row.consequent) for row in rows}
    missing = sorted(pattern for pattern in tested if pattern not in ordering)
    if missing:
        raise InvalidParameterError(
            "ordering", missing,
            "a scale range for every pattern the report tested. An ordering built from a "
            "different catalogue would drop rules silently, and a query that answers from a "
            "subset it never mentions is a query that lies about its denominator")

    target = int(target)
    if target not in ordering:
        raise InvalidParameterError(
            "target", target, "a pattern this ordering carries a scale range for")

    role = DIRECTION_TARGET_ROLE[direction]
    naming = tuple(row for row in rows if int(getattr(row, role)) == target)
    if not naming:
        raise InvalidParameterError(
            "target", target,
            "a pattern the report tested as the %s of at least one rule. Returning an empty "
            "answer here would read as 'nothing preceded it' when the truth is that the "
            "declared family never asked" % role)

    other_role = "antecedent" if role == "consequent" else "consequent"
    want_finer = DIRECTION_COUNTERPART_SIDE[direction] == "finer"

    selected: List[Tuple[int, TransitionWindow, Any]] = []
    overlapping = 0
    wrong_side = 0
    for row in naming:
        counterpart = int(getattr(row, other_role))
        if not ordering.orderable(counterpart, target):
            overlapping += 1
            continue
        on_side = (ordering.finer_than(counterpart, target) if want_finer
                   else ordering.coarser_than(counterpart, target))
        if not on_side:
            wrong_side += 1
            continue
        selected.append((counterpart, row.window, row))

    ranks: Dict[str, Dict[Tuple[int, float, float], int]] = {}
    top_by: Dict[str, Optional[Tuple[int, float, float]]] = {}
    for key in RANK_KEYS:
        scored = [(item, _figure(item[2], key)) for item in selected]
        scored = [(item, value) for item, value in scored if value is not None]
        scored.sort(key=lambda pair: ((pair[1] if RANK_ASCENDING[key] else -pair[1]),
                                      _identity(pair[0])))
        ranks[key] = {_identity(item): index + 1 for index, (item, _v) in enumerate(scored)}
        top_by[key] = _identity(scored[0][0]) if scored else None

    entries: List[QueryEntry] = []
    for item in selected:
        counterpart, window, row = item
        identity = _identity(item)
        entries.append(QueryEntry(
            antecedent=int(row.antecedent), consequent=int(row.consequent),
            counterpart=counterpart, window=window, status=row.status,
            support=(None if isinstance(row, SelectedLagRule) else int(row.support)),
            confidence=row.confidence, lift=row.lift,
            p_value=row.p_value, q_value=row.q_value,
            counterpart_scale=float(ordering.statistic[counterpart]),
            counterpart_scale_range=(float(ordering.low[counterpart]),
                                     float(ordering.high[counterpart])),
            scale_separation=float(ordering.separation(counterpart, target)),
            ranks={key: ranks[key].get(identity) for key in RANK_KEYS},
            row=row))

    entries.sort(key=lambda entry: (
        entry.ranks[rank_by] is None,
        entry.ranks[rank_by] if entry.ranks[rank_by] is not None else 0,
        entry.counterpart, float(entry.window.minimum_lag), float(entry.window.maximum_lag)))

    unmeasured = sum(1 for entry in entries if not entry.measured)
    if any(entry.rejected for entry in entries):
        status = ANSWER_SIGNATURES
    elif not entries:
        status = ANSWER_NONE_ORDERABLE
    elif unmeasured == len(entries):
        status = ANSWER_NOT_MEASURABLE
    else:
        status = ANSWER_NONE_DISTINGUISHED

    correction = (report.per_lag_correction if family == FAMILY_PER_LAG
                  else report.selected_lag_correction)
    return QueryResult(
        direction=direction, target=target, family=family, rank_by=rank_by, status=status,
        entries=tuple(entries),
        target_scale=float(ordering.statistic[target]),
        target_scale_range=(float(ordering.low[target]), float(ordering.high[target])),
        scale_units=ordering.scale_units,
        rows_in_family=len(rows), rows_naming_target=len(naming),
        withheld_overlapping_scale=overlapping, withheld_wrong_side=wrong_side,
        unmeasured_entries=unmeasured,
        correction_family={
            "family": family,
            "declared_before_the_record_was_read": True,
            "alpha": report.alpha,
            "correction": report.correction,
            "null": report.null_method,
            "n_surrogates": report.n_surrogates,
            "seed": report.seed,
            "size": dict(correction).get("n_tests"),
            "note": (
                "these q-values were corrected against the whole declared family and are "
                "reproduced here unchanged; restricting the question to one target after the "
                "fact does not restrict the family it was corrected against"),
        },
        top_by=top_by)


def directional_pair(report: PrecursorReport, earlier: int, later: int,
                     window: TransitionWindow) -> Dict[str, Any]:
    """Both orderings of one pair at one lag, side by side, because they are two hypotheses.

    Confidence has a different denominator in each direction, so `A -> B` cannot be read off
    `B -> A`. Both are members of the declared family and both were corrected in it; this puts
    them next to each other rather than deriving one from the other.
    """
    if not isinstance(report, PrecursorReport):
        raise InvalidParameterError("report", type(report).__name__, "a T4F.3 PrecursorReport")
    forward = report.rule(int(earlier), int(later), window)
    reverse = report.rule(int(later), int(earlier), window)
    return {
        "window": window.describe(),
        "forward": forward.describe(),
        "reverse": reverse.describe(),
        "confidence_differs": forward.confidence != reverse.confidence,
        "lift_differs": forward.lift != reverse.lift,
        "basis": (
            "confidence is the fraction of one pattern's eligible occurrences followed by the "
            "other, so the two directions divide by different denominators; neither is the "
            "other's reverse and both were tested in the declared family"),
        "claim_boundary": QUERY_CLAIM_BOUNDARY,
    }


__all__ = [
    "QUERY_SCHEMA", "SCALE_ORDER_SCHEMA",
    "DIRECTION_TOP_DOWN", "DIRECTION_BOTTOM_UP", "DIRECTIONS",
    "DIRECTION_TARGET_ROLE", "DIRECTION_COUNTERPART_SIDE", "DIRECTION_QUESTION",
    "FAMILY_PER_LAG", "FAMILY_SELECTED_LAG", "FAMILIES",
    "RANK_Q_VALUE", "RANK_LIFT", "RANK_CONFIDENCE", "RANK_SUPPORT", "RANK_KEYS",
    "RANK_ASCENDING", "RANK_CAUTION",
    "ANSWER_SIGNATURES", "ANSWER_NONE_DISTINGUISHED", "ANSWER_NONE_ORDERABLE",
    "ANSWER_NOT_MEASURABLE", "UNMEASURED_STATUSES",
    "SCALE_STATISTIC", "SCALE_ORDER_BASIS", "QUERY_CLAIM_BOUNDARY",
    "ScaleOrdering", "scale_ordering", "QueryEntry", "QueryResult",
    "bidirectional_query", "directional_pair",
]
