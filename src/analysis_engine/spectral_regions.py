"""Cross-region generalisation: where a rule holds, and where it does not (`T4F.7`, R14).

T4F.3 asked whether a rule beat a null on one record. R14 asks the question that decides what
kind of claim that was: **orography, coastlines and land-sea contrast produce behaviour that is
real and local**, so a rule measured in one place is a fact about that place until it has been
re-tested somewhere else. "We found a thing about the Alps" and "we found a thing about the
atmosphere" are both valuable and they are not the same claim.

Re-testing sounds like re-running the counter on another crop. It is not, and six things here
exist because it is not.

**1. A region is held out only if the pattern's identity was not fitted on it.** T4E.3 clusters
signatures into patterns and a centroid is fitted to whatever it was shown. If the clustering saw
the held-out regions, then the thing being re-tested was partly defined by the data it is being
re-tested on, which is R6's leak wearing a map instead of a calendar. So a catalogue is built on
the discovery region and held-out constellations are **matched into it** at the tolerance T4E.3
already calibrated -- never re-clustered, never re-calibrated. `identity_leakage` counts any
member of the supplied catalogue that came from a held-out region, and a catalogue with leakage
cannot return `general`.

**2. A constellation belongs to a region only if its whole footprint does.** T4F.5 measured that
a coefficient maximum sits about one analysing width from what excited it and that its honest
extent is the filter's own support, so assigning a configuration to a region by the point its
maximum landed on would put structures in the wrong region by up to half a filter width at a
boundary. Membership is therefore decided on footprints, and a configuration that is not wholly
inside exactly one declared region belongs to **none** of them. There are three ways that
happens and they say different things about the declaration, so they are counted apart rather
than lumped together: straddling two declared boxes says the boxes were drawn closer together
than this transform's own footprints reach, reaching out of the one box it touches says the
configuration extends into ground nobody declared, and being outside every box says it is
somewhere else altogether.

**3. Adjacent regions are not independent, and this module will not pretend otherwise.** One
weather system spans several hundred kilometres, so two neighbouring boxes see the same events
and a "held-out" test on the box next door is close to no test at all. The partition therefore
publishes the gap between every pair of regions in cells and in kilometres, and a caller may
declare a decorrelation length; regions closer than it are named, and `general` is refused while
any held-out region is that close to the discovery region. Without a declared decorrelation
length the independence of the regions is **unestablished** -- which is reported, not assumed.

**4. A region where the pattern never occurred did not fail the test; it never took it.** This
is the distinction the whole task turns on. "The rule did not hold in region C" and "region C
never carried the antecedent" are different findings, and a report that renders the second as
the first turns an absence of data into evidence of locality. Every region comes back `HOLDS`,
`DOES_NOT_HOLD` or `NOT_ASSESSABLE`, and the third is not a failure.

**5. Testing one rule in K regions is K tests.** The declared held-out regions are a family and
are corrected as one, over the number of regions **declared** rather than the number that
returned a number, because a region that produced no p-value was still part of the design.

**6. Physiography is declared, not derived.** A crop of one variable at one level carries no
coastline and no orography, so a `Physiography` is something a maintainer states with a source.
R14 asks for re-testing on similar *and* dissimilar physiography, and that requirement can only
be checked where the declaration exists: on a record where no region declares one, `general` is
refused with that as the reason rather than granted by default.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_clustering import (
    ConstellationPattern, PatternCatalogue, SignaturePoint,
)
from src.analysis_engine.spectral_constellation import ConstellationSet
from src.analysis_engine.spectral_events import EventSeries, ObservationGrid, events_from_catalogue
from src.analysis_engine.spectral_precursors import (
    NULL_CIRCULAR_SHIFT, PRECURSOR_CLAIM_BOUNDARY, PrecursorReport, RULE_PRECURSOR,
)
from src.analysis_engine.spectral_projection import Geography, project_occurrence
from src.analysis_engine.spectral_reference import PhysicalBridge
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.errors import InvalidParameterError
from src.statistics.multiple_comparisons import adjust

REGION_SCHEMA = "spectral-region-partition/v1"
GENERALISATION_SCHEMA = "spectral-cross-region-generalisation/v1"

#: What a region is for. A partition needs exactly one discovery region -- the place the rule
#: was found -- and at least one region it was not found in.
ROLE_DISCOVERY = "discovery"
ROLE_HELD_OUT = "held_out"
ROLES = (ROLE_DISCOVERY, ROLE_HELD_OUT)

#: Where one configuration sits relative to the declared regions, decided on footprints. Only
#: the first is placed; the other three are three different reasons for not placing it, and they
#: are kept apart because they say different things about the declaration. `STRADDLES` means the
#: boxes were drawn too close together for this transform's own footprints; `PARTLY_OUTSIDE`
#: means the configuration reaches into ground nobody declared; `OUTSIDE` means it is somewhere
#: else entirely.
MEMBERSHIP_INSIDE = "WHOLLY_INSIDE_ONE_REGION"
MEMBERSHIP_STRADDLES = "STRADDLES_MORE_THAN_ONE_REGION"
MEMBERSHIP_PARTLY_OUTSIDE = "REACHES_OUTSIDE_THE_ONE_REGION_IT_TOUCHES"
MEMBERSHIP_OUTSIDE = "OUTSIDE_EVERY_DECLARED_REGION"

#: How the rule came out in one region. The third is not a failure and must never be counted
#: as one: it says the region never took the test.
REGION_HOLDS = "HOLDS"
REGION_FAILS = "DOES_NOT_HOLD"
REGION_UNASSESSABLE = "NOT_ASSESSABLE_IN_THIS_REGION"

#: R14's two labels, and the third that keeps them honest.
VERDICT_GENERAL = "general"
VERDICT_REGIONAL = "regional"
VERDICT_UNASSESSABLE = "unassessable"

#: A rule re-tested in one held-out region has been re-tested once. Two is the floor for a
#: label that claims generality, and the number is here so a reader can argue with it.
MINIMUM_HELD_OUT_ASSESSED = 2

#: The sentence that stops a label being read as a mechanism.
GENERALISATION_CLAIM_BOUNDARY = (
    "A `general` label means this rule was measured again in at least two held-out regions, "
    "cleared the null in all of the ones that could be assessed, and did so across regions of "
    "declared and differing physiography. It is a statement about where the association has "
    "been observed to recur and about nothing else: not a cause, a driver, a mechanism, a "
    "trigger, a forecast or an intervention, and not a claim that the rule holds anywhere it "
    "has not been tested. A `regional` label says the association did not recur in at least one "
    "region that could carry it, which is a fact about those regions and not an explanation of "
    "them. `unassessable` means the question could not be put on this record and licenses "
    "nothing at all.")


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ the declaration


@dataclass(frozen=True)
class Physiography:
    """What a maintainer says the ground under a region is like, and where they got it.

    Declared rather than derived: a decomposition of one variable at one level carries no
    coastline, no orography and no land-sea mask, so anything this module said about the surface
    would be invention. `source` is required for the same reason a citation is required in
    T4F.6 -- a class assigned without one makes "similar" and "dissimilar" mean whatever the
    person assigning them wanted.
    """

    label: str
    description: str
    source: str

    def __post_init__(self) -> None:
        for name, value in (("label", self.label), ("description", self.description),
                            ("source", self.source)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "Physiography.%s" % name, value,
                    "a non-empty %s. R14 asks whether a rule was re-tested on similar and on "
                    "dissimilar ground, and a class with no stated basis cannot answer that: it "
                    "would make two regions similar because someone typed the same word" % name)

    def as_record(self) -> Dict[str, Any]:
        return {"label": self.label, "description": self.description, "source": self.source}


@dataclass(frozen=True)
class Region:
    """One box of the parent grid, its role in the design, and what is declared about it.

    `rows` and `columns` are inclusive cell ranges on the parent grid, which is the frame every
    footprint is already drawn on (T4D.1 removes each level's analysis delay before a position
    is reported), so no coordinate transform happens here and none can be wrong.
    """

    name: str
    rows: Tuple[int, int]
    columns: Tuple[int, int]
    role: str
    physiography: Optional[Physiography] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidParameterError("Region.name", self.name, "a non-empty name")
        if self.role not in ROLES:
            raise InvalidParameterError(
                "Region.role", self.role,
                "one of %s. A region with no declared role cannot be told apart from the one "
                "the rule was found in, and re-testing a rule on the region it was found in is "
                "not a re-test" % list(ROLES))
        for name, pair in (("rows", self.rows), ("columns", self.columns)):
            low, high = int(pair[0]), int(pair[1])
            if low > high:
                raise InvalidParameterError(
                    "Region.%s" % name, pair,
                    "an inclusive range whose low bound is at or below its high bound. A "
                    "reversed range is an empty region, and an empty region tests nothing while "
                    "looking like it tested something")
            if low < 0:
                raise InvalidParameterError(
                    "Region.%s" % name, pair, "cell indices at or above zero")

    def contains(self, low_row: int, high_row: int,
                 low_col: int, high_col: int) -> bool:
        """Whether a whole cell box lies inside this region."""
        return (int(self.rows[0]) <= low_row and high_row <= int(self.rows[1])
                and int(self.columns[0]) <= low_col and high_col <= int(self.columns[1]))

    def overlaps(self, other: "Region") -> bool:
        return (int(self.rows[0]) <= int(other.rows[1])
                and int(other.rows[0]) <= int(self.rows[1])
                and int(self.columns[0]) <= int(other.columns[1])
                and int(other.columns[0]) <= int(self.columns[1]))

    def gap_cells(self, other: "Region") -> float:
        """The shortest distance in cells between this region's box and another's."""
        row_gap = max(0, int(other.rows[0]) - int(self.rows[1]),
                      int(self.rows[0]) - int(other.rows[1]))
        col_gap = max(0, int(other.columns[0]) - int(self.columns[1]),
                      int(self.columns[0]) - int(other.columns[1]))
        return float(math.hypot(row_gap, col_gap))

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": self.name, "role": self.role,
            "rows": [int(self.rows[0]), int(self.rows[1])],
            "columns": [int(self.columns[0]), int(self.columns[1])],
        }
        if self.physiography is not None:
            record["physiography"] = self.physiography.as_record()
        return record


@dataclass(frozen=True)
class RegionPartition:
    """The declared regions, hashed, with the one the rule was found in named among them."""

    regions: Tuple[Region, ...]
    declared_by: str
    declared_on: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.regions:
            raise InvalidParameterError(
                "RegionPartition.regions", 0,
                "at least a discovery region and one held-out region. R14 is a question about "
                "somewhere else, and there is no somewhere else here")
        names = [item.name for item in self.regions]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise InvalidParameterError(
                "RegionPartition.regions", duplicated,
                "distinct region names, so that an outcome can be traced back to the box it was "
                "measured in")
        discovery = [item for item in self.regions if item.role == ROLE_DISCOVERY]
        if len(discovery) != 1:
            raise InvalidParameterError(
                "RegionPartition.regions", len(discovery),
                "exactly one discovery region. With none there is nothing to generalise from, "
                "and with two the rule was found in a union that no held-out region is held out "
                "of")
        if not [item for item in self.regions if item.role == ROLE_HELD_OUT]:
            raise InvalidParameterError(
                "RegionPartition.regions", 0,
                "at least one held-out region. Re-testing a rule on the region it was found in "
                "is not a re-test, it is the same measurement reported twice")
        for index, left in enumerate(self.regions):
            for right in self.regions[index + 1:]:
                if left.overlaps(right):
                    raise InvalidParameterError(
                        "RegionPartition.regions", (left.name, right.name),
                        "regions that do not overlap. A configuration inside the overlap would "
                        "be counted in both, so a rule could be 'confirmed' in a held-out region "
                        "by the very occurrences it was discovered from")
        for name, value in (("declared_by", self.declared_by),
                            ("declared_on", self.declared_on)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "RegionPartition.%s" % name, value, "a non-empty %s" % name)

    def __iter__(self):
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def digest(self) -> str:
        """A hash of the declared boxes, roles and physiography.

        T4F.6 learned this the hard way round: the set of regions a rule is re-tested on is a
        design choice, and a set chosen after the outcomes are known is a different design from
        the one that was declared. The digest is what makes those two tellable apart.
        """
        return _digest({
            "schema": REGION_SCHEMA,
            "regions": [item.as_record() for item in self.regions],
            "declared_by": self.declared_by,
            "declared_on": self.declared_on,
            "note": self.note,
        })

    @property
    def discovery(self) -> Region:
        return [item for item in self.regions if item.role == ROLE_DISCOVERY][0]

    @property
    def held_out(self) -> Tuple[Region, ...]:
        return tuple(item for item in self.regions if item.role == ROLE_HELD_OUT)

    def region(self, name: str) -> Region:
        for item in self.regions:
            if item.name == name:
                return item
        raise InvalidParameterError(
            "name", name, "a declared region (%s)" % sorted(item.name for item in self.regions))

    def gaps(self, bridge: Optional[PhysicalBridge] = None) -> Tuple[Dict[str, Any], ...]:
        """The shortest gap between every pair of declared regions, in cells and in kilometres.

        Published whether or not a decorrelation length was declared, because a reader deciding
        whether to believe a held-out result needs to know how far away the held-out box was.
        """
        rows: List[Dict[str, Any]] = []
        for index, left in enumerate(self.regions):
            for right in self.regions[index + 1:]:
                cells = left.gap_cells(right)
                record: Dict[str, Any] = {"regions": [left.name, right.name],
                                          "gap_cells": cells}
                if bridge is not None and bridge.has_length:
                    record["gap_km"] = {"low": cells * bridge.cell_km_low,
                                        "high": cells * bridge.cell_km_high}
                elif bridge is not None:
                    record["gap_km_refusal"] = bridge.cell_refusal
                rows.append(record)
        return tuple(rows)


# ------------------------------------------------------------------------------ where things are


@dataclass(frozen=True)
class Membership:
    """Which region one configuration is in, decided on the footprints it was drawn with."""

    key: Tuple[Any, ...]
    time: float
    status: str
    region: Optional[str]
    touched: Tuple[str, ...]
    rows: Tuple[int, int]
    columns: Tuple[int, int]

    def as_record(self) -> Dict[str, Any]:
        return {"key": list(self.key), "time": self.time, "status": self.status,
                "region": self.region, "regions_touched": list(self.touched),
                "footprint_rows": [self.rows[0], self.rows[1]],
                "footprint_columns": [self.columns[0], self.columns[1]]}


@dataclass(frozen=True)
class RegionAssignment:
    """Every configuration of a pass, placed or explicitly not placed."""

    memberships: Mapping[Tuple[Any, ...], Membership]
    inside: Mapping[str, int]
    n_straddling: int
    n_partly_outside: int
    n_outside: int

    def of(self, key: Sequence[Any]) -> Optional[Membership]:
        return self.memberships.get(tuple(key))

    def keys_in(self, region: str) -> Tuple[Tuple[Any, ...], ...]:
        return tuple(sorted(
            (key for key, item in self.memberships.items()
             if item.status == MEMBERSHIP_INSIDE and item.region == region),
            key=lambda key: tuple(str(part) for part in key)))

    def as_record(self) -> Dict[str, Any]:
        return {
            "configurations": len(self.memberships),
            "inside_each_region": dict(self.inside),
            "straddling_two_or_more_regions": self.n_straddling,
            "reaching_outside_the_one_region_they_touch": self.n_partly_outside,
            "outside_every_region": self.n_outside,
            "note": ("membership is decided on the whole footprint of every member, not on the "
                     "cell its maximum landed in, because a coefficient maximum sits about one "
                     "analysing width from what excited it (T4F.5). A configuration that is "
                     "not wholly inside exactly one declared region belongs to none of them, "
                     "and the three ways that happens are counted apart: straddling two "
                     "declared regions says the boxes were drawn closer together than this "
                     "transform's own footprints reach, reaching outside the one region it "
                     "touches says it extends into ground nobody declared, and neither is the "
                     "same as being somewhere else altogether"),
        }


def assign_regions(constellations: ConstellationSet, partition: RegionPartition,
                   geography: Geography) -> RegionAssignment:
    """Place every configuration in a region, or record that it straddles or falls outside."""
    if not isinstance(constellations, ConstellationSet):
        raise InvalidParameterError(
            "constellations", type(constellations).__name__, "a T4E.1 ConstellationSet")
    if not isinstance(partition, RegionPartition):
        raise InvalidParameterError(
            "partition", type(partition).__name__, "a declared RegionPartition")
    if not isinstance(geography, Geography):
        raise InvalidParameterError(
            "geography", type(geography).__name__,
            "a T4F.5 Geography, so that a configuration is placed by the footprints it is "
            "actually drawn with rather than by the cell its maximum landed in")

    memberships: Dict[Tuple[Any, ...], Membership] = {}
    inside: Dict[str, int] = {item.name: 0 for item in partition}
    straddling = 0
    partly = 0
    outside = 0
    for constellation in constellations:
        projection = project_occurrence(constellation, geography=geography, pattern_id=0)
        low_row = min(item.rows[0] for item in projection.footprints)
        high_row = max(item.rows[1] for item in projection.footprints)
        columns = [segment for item in projection.footprints for segment in item.columns]
        low_col = min(segment[0] for segment in columns)
        high_col = max(segment[1] for segment in columns)
        touched = tuple(sorted(
            region.name for region in partition
            if not (high_row < int(region.rows[0]) or low_row > int(region.rows[1])
                    or high_col < int(region.columns[0])
                    or low_col > int(region.columns[1]))))
        home = [region for region in partition
                if region.contains(low_row, high_row, low_col, high_col)]
        if len(home) == 1 and len(touched) == 1:
            status, region_name = MEMBERSHIP_INSIDE, home[0].name
            inside[region_name] += 1
        elif len(touched) >= 2:
            status, region_name = MEMBERSHIP_STRADDLES, None
            straddling += 1
        elif touched:
            status, region_name = MEMBERSHIP_PARTLY_OUTSIDE, None
            partly += 1
        else:
            status, region_name = MEMBERSHIP_OUTSIDE, None
            outside += 1
        key = tuple(constellation.key())
        memberships[key] = Membership(
            key=key, time=float(constellation.time), status=status, region=region_name,
            touched=touched, rows=(low_row, high_row), columns=(low_col, high_col))
    return RegionAssignment(memberships=memberships, inside=inside,
                            n_straddling=straddling, n_partly_outside=partly,
                            n_outside=outside)


# ------------------------------------------------------------------------------ identity, held


@dataclass(frozen=True)
class MatchedCatalogue:
    """A catalogue whose identities were fitted in one place and matched in another.

    `fitted` is the set of member keys the centroids were actually clustered from. Everything
    else in `catalogue` was matched into an identity that already existed, which is the only
    way a held-out region can supply occurrences of a pattern without also helping to define it.
    """

    catalogue: PatternCatalogue
    fitted: Tuple[Tuple[Any, ...], ...]
    n_matched: int
    n_unmatched: int

    def as_record(self) -> Dict[str, Any]:
        return {
            "patterns": len(self.catalogue.patterns),
            "fitted_members": len(self.fitted),
            "matched_into_an_existing_identity": self.n_matched,
            "matched_nothing": self.n_unmatched,
            "note": ("a configuration is matched into a pattern only if it lies inside the "
                     "tolerance radius T4E.3 calibrated for that pattern. Nothing is "
                     "re-clustered and no tolerance is re-calibrated, because a held-out region "
                     "that got to redefine the pattern would not be held out. A configuration "
                     "that matches nothing is not an occurrence of anything and is counted here "
                     "rather than assigned to its nearest centroid"),
        }


def match_into_catalogue(catalogue: PatternCatalogue,
                         signatures: Sequence[Any]) -> MatchedCatalogue:
    """Attach further occurrences to patterns whose identity was fitted somewhere else.

    This is the step that makes a held-out region testable at all. The centroids, the metric and
    the tolerance radius all come from the supplied catalogue and none of them moves; a signature
    joins the nearest pattern whose calibrated radius contains it, and joins nothing otherwise.
    """
    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a T4E.3 PatternCatalogue")
    if not catalogue.patterns:
        raise InvalidParameterError(
            "catalogue.patterns", 0,
            "at least one pattern to match into. An empty catalogue has no identity to hold "
            "constant, so matching into it would be clustering under another name")

    fitted = tuple(tuple(member.key) for pattern in catalogue for member in pattern.members)
    known = set(fitted)
    extra: Dict[int, List[Any]] = {}
    matched = 0
    unmatched = 0
    for signature in signatures:
        if tuple(signature.key) in known:
            continue
        point = SignaturePoint.from_signature(signature)
        best_id = None
        best_distance = None
        for pattern in catalogue:
            if pattern.centroid.family != point.family:
                continue
            distance = catalogue.metric.distance(point, pattern.centroid)
            if distance > float(pattern.tolerance_radius):
                continue
            if best_distance is None or distance < best_distance:
                best_id, best_distance = int(pattern.pattern_id), distance
        if best_id is None:
            unmatched += 1
            continue
        extra.setdefault(best_id, []).append(signature)
        known.add(tuple(signature.key))
        matched += 1

    patterns: List[ConstellationPattern] = []
    for pattern in catalogue:
        members = tuple(sorted(tuple(pattern.members) + tuple(extra.get(int(pattern.pattern_id),
                                                                       ())),
                               key=lambda item: (float(item.time), tuple(item.key))))
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=pattern.pattern_id, members=members, centroid=pattern.centroid,
            tolerance_radius=pattern.tolerance_radius,
            observed_radius=max(catalogue.metric.distance(point, pattern.centroid)
                                for point in points)))
    extended = PatternCatalogue(patterns=tuple(patterns), metric=catalogue.metric,
                                tolerance=catalogue.tolerance,
                                n_signatures=sum(len(item.members) for item in patterns))
    return MatchedCatalogue(catalogue=extended, fitted=fitted, n_matched=matched,
                            n_unmatched=unmatched)


@dataclass(frozen=True)
class IdentityLeakage:
    """How much of the pattern identities being re-tested was fitted on the held-out regions."""

    n_members: int
    n_from_held_out: int
    n_unplaced: int
    per_pattern: Mapping[int, int]

    @property
    def clean(self) -> bool:
        return self.n_from_held_out == 0

    def as_record(self) -> Dict[str, Any]:
        return {
            "catalogue_members": self.n_members,
            "members_from_held_out_regions": self.n_from_held_out,
            "members_not_placed_in_any_region": self.n_unplaced,
            "per_pattern": {str(key): value for key, value in sorted(self.per_pattern.items())},
            "clean": self.clean,
            "note": ("a pattern's centroid is fitted to the members it was clustered from. If "
                     "any of those came from a held-out region then the thing being re-tested "
                     "was partly defined by the data it is being re-tested on, which is R6's "
                     "leak with a map in place of a calendar"),
        }


def identity_leakage(catalogue: PatternCatalogue, assignment: RegionAssignment,
                     partition: RegionPartition, *,
                     fitted: Optional[Sequence[Sequence[Any]]] = None) -> IdentityLeakage:
    """Count the members the pattern identities were *fitted* on that came from held-out ground.

    `fitted` names the members the centroids were clustered from; a `MatchedCatalogue` supplies
    it. Omitting it is the conservative reading and the right default: a catalogue that cannot
    say which of its members defined it must be assumed to have been defined by all of them.
    """
    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a T4E.3 PatternCatalogue")
    held_out = {item.name for item in partition.held_out}
    chosen = None if fitted is None else {tuple(key) for key in fitted}
    total = 0
    leaked = 0
    unplaced = 0
    per_pattern: Dict[int, int] = {}
    for pattern in catalogue:
        for member in pattern.members:
            if chosen is not None and tuple(member.key) not in chosen:
                continue
            total += 1
            placed = assignment.of(member.key)
            if placed is None or placed.status != MEMBERSHIP_INSIDE:
                unplaced += 1
                continue
            if placed.region in held_out:
                leaked += 1
                per_pattern[int(pattern.pattern_id)] = (
                    per_pattern.get(int(pattern.pattern_id), 0) + 1)
    return IdentityLeakage(n_members=total, n_from_held_out=leaked, n_unplaced=unplaced,
                           per_pattern=per_pattern)


def restrict_catalogue(catalogue: PatternCatalogue, assignment: RegionAssignment,
                       region: str) -> PatternCatalogue:
    """The same patterns, keeping only the occurrences that lie wholly inside one region.

    The centroids, the metric and the calibrated tolerance are carried through untouched: this
    is a restriction of *where* the occurrences were, not a re-clustering, and re-fitting a
    centroid per region would give every region its own definition of the pattern and make the
    comparison between them meaningless.
    """
    patterns: List[ConstellationPattern] = []
    for pattern in catalogue:
        members = tuple(member for member in pattern.members
                        if (assignment.of(member.key) is not None
                            and assignment.of(member.key).status == MEMBERSHIP_INSIDE
                            and assignment.of(member.key).region == region))
        if not members:
            continue
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=pattern.pattern_id, members=members, centroid=pattern.centroid,
            tolerance_radius=pattern.tolerance_radius,
            observed_radius=max(catalogue.metric.distance(point, pattern.centroid)
                                for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=catalogue.metric,
                            tolerance=catalogue.tolerance,
                            n_signatures=sum(len(item.members) for item in patterns))


def region_series(catalogue: PatternCatalogue, assignment: RegionAssignment, region: str, *,
                  grid: ObservationGrid) -> Optional[EventSeries]:
    """This region's own event series, or `None` when it carries no occurrence at all.

    `None` is the honest answer rather than an empty series: an `EventSeries` with no events
    would be counted as a region that was searched and found nothing, and a region whose
    restricted catalogue is empty has not been counted at all.
    """
    restricted = restrict_catalogue(catalogue, assignment, region)
    if not restricted.patterns:
        return None
    return events_from_catalogue(restricted, grid=grid)


# ------------------------------------------------------------------------------ one region


@dataclass(frozen=True)
class RegionOutcome:
    """How the rule came out in one region, with everything it was decided from."""

    region: str
    role: str
    status: str
    physiography: Optional[str]
    n_occurrences_of_antecedent: int
    n_occurrences_of_consequent: int
    support: Optional[int]
    eligible_antecedents: Optional[int]
    confidence: Optional[float]
    base_rate: Optional[float]
    lift: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float]
    rule_status: Optional[str]
    reason: str

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "region": self.region, "role": self.role, "outcome": self.status,
            "physiography": self.physiography,
            "occurrences": {"antecedent": self.n_occurrences_of_antecedent,
                            "consequent": self.n_occurrences_of_consequent},
            "reason": self.reason,
        }
        for name in ("support", "eligible_antecedents", "confidence", "base_rate", "lift",
                     "p_value", "q_value", "rule_status"):
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        return record


# ------------------------------------------------------------------------------ the verdict


@dataclass(frozen=True)
class Generalisation:
    """R14's label for one rule, and the whole design it was reached under."""

    antecedent: int
    consequent: int
    window: TransitionWindow
    verdict: str
    outcomes: Tuple[RegionOutcome, ...]
    partition_digest: str
    correction: Mapping[str, Any]
    leakage: IdentityLeakage
    assignment: Mapping[str, Any]
    independence: Mapping[str, Any]
    reasons: Tuple[str, ...]

    @property
    def held_out_outcomes(self) -> Tuple[RegionOutcome, ...]:
        return tuple(item for item in self.outcomes if item.role == ROLE_HELD_OUT)

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema": GENERALISATION_SCHEMA,
            "rule": {"antecedent": self.antecedent, "consequent": self.consequent,
                     "window": [float(self.window.minimum_lag), float(self.window.maximum_lag),
                                self.window.time_units]},
            "verdict": self.verdict,
            "partition_digest": self.partition_digest,
            "regions": [item.as_record() for item in self.outcomes],
            "correction": dict(self.correction),
            "identity_leakage": self.leakage.as_record(),
            "assignment": dict(self.assignment),
            "independence": dict(self.independence),
            "reasons": list(self.reasons),
            "claim_boundary": GENERALISATION_CLAIM_BOUNDARY,
            "inherited_claim_boundary": PRECURSOR_CLAIM_BOUNDARY,
        }


def _independence(partition: RegionPartition, bridge: Optional[PhysicalBridge],
                  decorrelation_km: Optional[float]) -> Dict[str, Any]:
    """How far apart the regions are, and whether that has been declared to be far enough."""
    gaps = partition.gaps(bridge)
    record: Dict[str, Any] = {"gaps": list(gaps),
                              "decorrelation_km": decorrelation_km}
    if decorrelation_km is None:
        record["status"] = "UNESTABLISHED"
        record["note"] = (
            "no decorrelation length was declared, so nothing here establishes that the "
            "held-out regions are independent of the discovery region. One weather system "
            "spans several hundred kilometres and two boxes inside one see the same events, so "
            "a held-out result from a neighbouring box may be the discovery repeated rather "
            "than reproduced")
        return record
    if bridge is None or not bridge.has_length:
        record["status"] = "UNESTABLISHED"
        record["note"] = (
            "a decorrelation length in kilometres was declared but this grid declares no "
            "length, so the gaps between regions cannot be compared with it")
        return record
    discovery = partition.discovery.name
    too_close = []
    for row in gaps:
        if discovery not in row["regions"]:
            continue
        if row["gap_km"]["high"] < float(decorrelation_km):
            too_close.append([name for name in row["regions"] if name != discovery][0])
    record["status"] = "ESTABLISHED" if not too_close else "NOT_ESTABLISHED"
    record["closer_to_the_discovery_region_than_the_declared_decorrelation_length"] = too_close
    record["note"] = (
        "every held-out region is further from the discovery region than the declared "
        "decorrelation length, at the widest kilometre reading this grid supports"
        if not too_close else
        "these held-out regions sit closer to the discovery region than the declared "
        "decorrelation length, so a result from them may be the discovery seen again rather "
        "than reproduced; `general` is refused while that is so")
    return record


def cross_region_generalisation(
        catalogue: PatternCatalogue, constellations: ConstellationSet, *,
        partition: RegionPartition, geography: Geography, grid: ObservationGrid,
        antecedent: int, consequent: int, window: TransitionWindow,
        n_surrogates: int, seed: int, alpha: float = 0.05,
        correction: str = "benjamini_yekutieli",
        null: str = NULL_CIRCULAR_SHIFT,
        bridge: Optional[PhysicalBridge] = None,
        decorrelation_km: Optional[float] = None,
        fitted: Optional[Sequence[Sequence[Any]]] = None) -> Generalisation:
    """Re-test one rule in every declared region and label it `general` or `regional`.

    Each region's own `precursor_report` is what decides it -- nothing statistical is
    re-implemented here -- and the held-out regions are corrected as one family, over the number
    of regions **declared** rather than the number that returned a number.
    """
    if not isinstance(partition, RegionPartition):
        raise InvalidParameterError(
            "partition", type(partition).__name__, "a declared RegionPartition")
    if not isinstance(grid, ObservationGrid):
        raise InvalidParameterError(
            "grid", type(grid).__name__,
            "the ObservationGrid of the pass. The frames a region was searched over are the "
            "frames the pass searched, and they are not recoverable from a catalogue")
    if int(antecedent) == int(consequent):
        raise InvalidParameterError(
            "consequent", consequent,
            "a pattern other than the antecedent. A pattern preceding itself is a recurrence "
            "and T4F.2 counts it under that name")

    assignment = assign_regions(constellations, partition, geography)
    leakage = identity_leakage(catalogue, assignment, partition, fitted=fitted)
    independence = _independence(partition, bridge, decorrelation_km)

    raw: Dict[str, Optional[float]] = {}
    provisional: List[Dict[str, Any]] = []
    for region in partition:
        series = region_series(catalogue, assignment, region.name, grid=grid)
        counts = {int(antecedent): 0, int(consequent): 0}
        if series is not None:
            for event in series.events:
                if int(event.pattern_id) in counts:
                    counts[int(event.pattern_id)] += 1
        entry: Dict[str, Any] = {
            "region": region, "antecedent_occurrences": counts[int(antecedent)],
            "consequent_occurrences": counts[int(consequent)], "rule": None, "reason": "",
        }
        if series is None or counts[int(antecedent)] == 0:
            entry["reason"] = (
                "this region carries no occurrence of pattern %d at all, so the rule was never "
                "put to the test here. That is not the rule failing in this region"
                % int(antecedent))
        elif counts[int(consequent)] == 0:
            entry["reason"] = (
                "pattern %d occurs here but pattern %d never does, so the base rate this lift "
                "would be measured against is zero and no lift exists. The configuration was "
                "absent, which is not the same finding as the association being absent"
                % (int(antecedent), int(consequent)))
        else:
            try:
                report = precursor_in_region(
                    series, antecedent=antecedent, consequent=consequent, window=window,
                    n_surrogates=n_surrogates, seed=seed, alpha=alpha, correction=correction, null=null)
            except InvalidParameterError as exc:
                entry["reason"] = (
                    "this region's own design was refused before anything was counted: %s"
                    % str(exc))
            else:
                rule = report.rule(int(antecedent), int(consequent), window)
                entry["rule"] = rule
                raw[region.name] = rule.p_value
        provisional.append(entry)

    held_out_names = [item.name for item in partition.held_out]
    measured = [(name, raw[name]) for name in held_out_names
                if raw.get(name) is not None]
    correction_record: Dict[str, Any] = {
        "method": correction, "alpha": alpha,
        "regions_declared_held_out": len(held_out_names),
        "regions_returning_a_p_value": len(measured),
        "note": ("the family is the number of held-out regions *declared*, not the number that "
                 "returned a number: a region that produced no p-value was still part of the "
                 "design, and correcting only over the ones that reported would price a search "
                 "as though it had not happened"),
    }
    corrected: Dict[str, float] = {}
    rejected: Dict[str, bool] = {}
    if measured:
        result = adjust([value for _name, value in measured], method=correction, alpha=alpha,
                        n_tests=len(held_out_names),
                        labels=[name for name, _value in measured])
        for (name, _value), q_value, ok in zip(measured, result["adjusted"],
                                               result["rejected"]):
            corrected[name] = float(q_value)
            rejected[name] = bool(ok)
        correction_record["dependence"] = result.get("dependence")

    outcomes: List[RegionOutcome] = []
    for entry in provisional:
        region: Region = entry["region"]
        rule = entry["rule"]
        if rule is None:
            status, reason = REGION_UNASSESSABLE, entry["reason"]
            figures: Dict[str, Any] = {}
        elif region.role == ROLE_DISCOVERY:
            status = REGION_HOLDS if rule.status == RULE_PRECURSOR else REGION_FAILS
            reason = ("the discovery region, reported for reference and deliberately outside "
                      "the corrected family: it is where the rule was found, so counting it "
                      "again is not a re-test")
            figures = {"q_value": rule.q_value}
        else:
            held = rejected.get(region.name, False)
            status = REGION_HOLDS if held else REGION_FAILS
            reason = ("cleared the null in this region and survived the correction over the %d "
                      "declared held-out regions" % len(held_out_names) if held else
                      "did not clear the null in this region after the correction over the %d "
                      "declared held-out regions, on a region that did carry both patterns"
                      % len(held_out_names))
            figures = {"q_value": corrected.get(region.name)}
        outcomes.append(RegionOutcome(
            region=region.name, role=region.role, status=status,
            physiography=None if region.physiography is None else region.physiography.label,
            n_occurrences_of_antecedent=entry["antecedent_occurrences"],
            n_occurrences_of_consequent=entry["consequent_occurrences"],
            support=None if rule is None else rule.support,
            eligible_antecedents=None if rule is None else rule.eligible_antecedents,
            confidence=None if rule is None else rule.confidence,
            base_rate=None if rule is None else rule.base_rate,
            lift=None if rule is None else rule.lift,
            p_value=None if rule is None else rule.p_value,
            q_value=figures.get("q_value"),
            rule_status=None if rule is None else rule.status,
            reason=reason))

    verdict, reasons = _verdict(outcomes, partition, leakage, independence)
    return Generalisation(
        antecedent=int(antecedent), consequent=int(consequent), window=window, verdict=verdict,
        outcomes=tuple(outcomes), partition_digest=partition.digest,
        correction=correction_record, leakage=leakage,
        assignment=assignment.as_record(), independence=independence, reasons=tuple(reasons))


def precursor_in_region(series: EventSeries, *, antecedent: int, consequent: int,
                        window: TransitionWindow, n_surrogates: int, seed: int,
                        alpha: float, correction: str, null: str) -> PrecursorReport:
    """One region's own T4F.3 report, on that region's own occurrences.

    A thin wrapper so that the import sits in one place and so that the one thing this module
    must never do -- re-implement the counting or the null -- is visibly not done.
    """
    from src.analysis_engine.spectral_precursors import LagFamily, precursor_report
    return precursor_report(series, lags=LagFamily((window,)), n_surrogates=n_surrogates,
                            seed=seed, alpha=alpha, correction=correction,
                            null=null,
                            pairs=[(int(antecedent), int(consequent))])


def _verdict(outcomes: Sequence[RegionOutcome], partition: RegionPartition,
             leakage: IdentityLeakage,
             independence: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """R14's label, and every reason it is not `general`."""
    held_out = [item for item in outcomes if item.role == ROLE_HELD_OUT]
    holds = [item for item in held_out if item.status == REGION_HOLDS]
    fails = [item for item in held_out if item.status == REGION_FAILS]
    assessed = holds + fails
    reasons: List[str] = []

    if not assessed:
        reasons.append(
            "no held-out region could be assessed: none of the %d declared carried both "
            "patterns, so the rule has not been re-tested anywhere" % len(held_out))
        return VERDICT_UNASSESSABLE, reasons
    if len(assessed) < MINIMUM_HELD_OUT_ASSESSED:
        reasons.append(
            "only %d held-out region could be assessed and %d is the floor for a label about "
            "generality: one re-test is a second measurement, not a demonstration that a rule "
            "travels" % (len(assessed), MINIMUM_HELD_OUT_ASSESSED))
        return VERDICT_UNASSESSABLE, reasons

    if fails:
        reasons.append(
            "the rule held in %d of the %d held-out regions that could be assessed and did not "
            "hold in %s" % (len(holds), len(assessed),
                            ", ".join(sorted(item.region for item in fails))))
        return VERDICT_REGIONAL, reasons

    reasons.append("the rule held in all %d held-out regions that could be assessed (%s)"
                   % (len(assessed), ", ".join(sorted(item.region for item in holds))))
    blocked = False
    if not leakage.clean:
        reasons.append(
            "but %d of the catalogue's members came from held-out regions, so the pattern "
            "identities being re-tested were partly fitted on the data they were re-tested on. "
            "`general` is refused: cluster on the discovery region and match the held-out "
            "configurations into those patterns" % leakage.n_from_held_out)
        blocked = True
    if independence.get("status") != "ESTABLISHED":
        reasons.append("but the independence of the held-out regions is %s: %s"
                       % (str(independence.get("status", "UNESTABLISHED")).lower(),
                          independence.get("note", "")))
        blocked = True
    declared = [item for item in assessed if item.physiography is not None]
    classes = {item.physiography for item in declared}
    if len(declared) != len(assessed):
        reasons.append(
            "but %d of the %d assessed held-out regions declare no physiography, so R14's "
            "requirement that a rule be re-tested on similar *and* dissimilar ground cannot be "
            "checked" % (len(assessed) - len(declared), len(assessed)))
        blocked = True
    elif len(classes) < 2:
        reasons.append(
            "but every assessed held-out region is of one declared physiography (%s), so the "
            "rule has been shown to recur on similar ground and not on dissimilar ground, "
            "which is the comparison R14 asks for"
            % ", ".join(sorted(str(item) for item in classes)))
        blocked = True
    return (VERDICT_REGIONAL if blocked else VERDICT_GENERAL), reasons


def describe_generalisation(result: Generalisation) -> Dict[str, Any]:
    """The verdict as a plain record, for a receipt or a route."""
    if not isinstance(result, Generalisation):
        raise InvalidParameterError("result", type(result).__name__, "a Generalisation")
    return result.as_record()
