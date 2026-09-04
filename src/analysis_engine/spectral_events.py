"""T4F.1: the event substrate transition mining runs over, and the clock it refuses to assume.

Phase 4F asks which small configurations at `t` precede which structures at `t + delta`. The
existing engine mines scalar run metrics out of flattened `results` JSON and structurally cannot
express `A4 -> A8 -> B8 -> C16`, because nothing in that record carries an order. This module
supplies the ordered substrate instead: every clustered pattern occurrence as a timed event, on a
declared observation grid, with the spans between occurrences classified rather than assumed.

It counts nothing. Support, confidence, recurrence intervals and lift begin at T4F.2, and a span
here is an elapsed distance between two occurrences in this record -- not a period, not evidence
of periodicity, and not a precursor relation.

**Three things a `PatternCatalogue` cannot answer on its own, and this module refuses rather than
guesses.**

*   *In what units is the clock?* T4E.2 signed each constellation with a bare float `time` and
    dropped the `time_units` its `FrameConstellation` carried (D93, fixed at that seam in this
    slice). An interval in unnamed units is not an interval, so the grid must declare the unit
    and every member that carries one must agree with it.
*   *What was looked at?* A catalogue holds occurrences, and an occurrence is evidence only
    against the frames that were searched. An event at a time the pass never examined is refused
    by name; it asserts a sighting where nothing looked.
*   *Is a gap a gap, or is it an absence?* Two occurrences twelve hours apart mean one thing when
    every intervening frame was searched and found empty, and another when six of them were never
    read. The searched frames alone cannot tell those apart -- the grid is by construction
    complete in itself -- so the answer needs a declared cadence. Without one the span is marked
    `COVERAGE_UNDECLARED` rather than silently treated as continuous observation, for the reason
    T4E.1 reports a refused relation by name instead of dropping it.

**Simultaneity is not order.** Two events on the same frame are co-occurring, and the substrate
says so. Sorting them into a list would hand T4F.2 a succession that came from tuple comparison
rather than from the record, which is exactly the fabricated arrow `A4 -> A8` exists to avoid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_clustering import PatternCatalogue
from src.core.errors import InvalidParameterError


EVENT_SCHEMA = "spectral-transition-events/v1"

#: A frame lands on the declared cadence when its lattice index is this close to an integer.
CADENCE_TOLERANCE = 1e-6

COVERAGE_UNDECLARED = "COVERAGE_UNDECLARED"
COVERAGE_MEASURED = "MEASURED"
COVERAGE_INCOMPLETE = "SPANS_UNOBSERVED_TIME"

CLAIM_BOUNDARY = (
    "an event is one clustered occurrence at one searched frame, and a span is the elapsed "
    "distance between two of them in this record. Neither is a recurrence, a period, a "
    "precursor, a rate, a support count or a discovery; those begin at T4F.2")


def _identity(key: Sequence[Any]) -> str:
    """The same occurrence identity T4E.4 counts support in, spelled the same way."""
    return json.dumps(list(key), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ObservationGrid:
    """The frames a pass searched, the unit they are in, and optionally the cadence they sample.

    `frames` is what was *looked at*, not what was found. The distinction is the whole point of
    the object: a catalogue can say when a pattern occurred and can never say when it did not.
    """

    frames: Tuple[float, ...]
    time_units: str
    cadence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_units, str) or not self.time_units.strip():
            raise InvalidParameterError(
                "ObservationGrid.time_units", self.time_units,
                "a non-empty unit name. T4F.2 asks whether the gap between occurrences recurs, "
                "and a number without a unit cannot answer it")
        if not self.frames:
            raise InvalidParameterError(
                "ObservationGrid.frames", 0,
                "at least one searched frame. An empty grid is an unrun pass, not a pass that "
                "observed nothing")
        previous: Optional[float] = None
        for frame in self.frames:
            value = float(frame)
            if not math.isfinite(value):
                raise InvalidParameterError(
                    "ObservationGrid.frames", frame, "finite frame times")
            if previous is not None and value <= previous:
                raise InvalidParameterError(
                    "ObservationGrid.frames", value,
                    "strictly increasing frame times (%s follows %s). A repeated or reordered "
                    "frame would make one instant two observations" % (value, previous))
            previous = value
        if self.cadence is not None:
            cadence = float(self.cadence)
            if not math.isfinite(cadence) or cadence <= 0.0:
                raise InvalidParameterError(
                    "ObservationGrid.cadence", self.cadence,
                    "a positive finite sampling interval, or none at all")
            origin = float(self.frames[0])
            for frame in self.frames:
                steps = (float(frame) - origin) / cadence
                if abs(steps - round(steps)) > CADENCE_TOLERANCE:
                    raise InvalidParameterError(
                        "ObservationGrid.frames", frame,
                        "a frame on the declared cadence %s %s from %s. This one sits %s steps "
                        "along, so the cadence does not describe the grid and the missing "
                        "frames it would imply are an artefact of the wrong lattice"
                        % (cadence, self.time_units, origin, steps))

    def __len__(self) -> int:
        return len(self.frames)

    def __contains__(self, time: Any) -> bool:
        return float(time) in set(self.frames)

    @property
    def span(self) -> float:
        return float(self.frames[-1]) - float(self.frames[0])

    def coverage_between(self, earlier: float, later: float) -> Tuple[str, int, Optional[int]]:
        """Classify the span between two searched frames: status, observed, missing."""
        observed = sum(1 for frame in self.frames if earlier < float(frame) < later)
        if self.cadence is None:
            return COVERAGE_UNDECLARED, observed, None
        expected = int(round((later - earlier) / float(self.cadence))) - 1
        missing = max(expected - observed, 0)
        status = COVERAGE_MEASURED if missing == 0 else COVERAGE_INCOMPLETE
        return status, observed, missing

    def describe(self) -> Dict[str, Any]:
        return {
            "n_frames_searched": len(self.frames),
            "first_frame": float(self.frames[0]),
            "last_frame": float(self.frames[-1]),
            "span": self.span,
            "time_units": self.time_units,
            "cadence": None if self.cadence is None else float(self.cadence),
            "coverage_decidable": self.cadence is not None,
            "coverage_basis": (
                "spans are checked against the declared cadence, so an unsearched instant is "
                "counted rather than assumed away" if self.cadence is not None else
                "no cadence was declared, so this grid cannot distinguish a searched frame "
                "that found nothing from an instant that was never read"),
        }


@dataclass(frozen=True)
class PatternEvent:
    """One clustered occurrence, at one frame, under one scale mode."""

    pattern_id: int
    time: float
    identity: str
    cardinality: int
    mode: str

    def describe(self) -> Dict[str, Any]:
        return {"pattern_id": self.pattern_id, "time": self.time, "identity": self.identity,
                "cardinality": self.cardinality, "mode": self.mode}


@dataclass(frozen=True)
class EventSpan:
    """The distance between two consecutive occurrences of one pattern, and what was watched."""

    pattern_id: int
    earlier: float
    later: float
    duration: float
    time_units: str
    frames_observed_between: int
    frames_unobserved_between: Optional[int]
    coverage: str

    def describe(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id, "earlier": self.earlier, "later": self.later,
            "duration": self.duration, "time_units": self.time_units,
            "frames_observed_between": self.frames_observed_between,
            "frames_unobserved_between": self.frames_unobserved_between,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class EventSeries:
    """Every occurrence in one catalogue, ordered, on one grid, in one declared unit."""

    events: Tuple[PatternEvent, ...]
    grid: ObservationGrid
    mode: str
    units_confirmed_by_members: int
    units_uncarried_by_members: int

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[PatternEvent]:
        return iter(self.events)

    @property
    def pattern_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({event.pattern_id for event in self.events}))

    def at(self, time: float) -> Tuple[PatternEvent, ...]:
        return tuple(event for event in self.events if event.time == float(time))

    def for_pattern(self, pattern_id: int) -> Tuple[PatternEvent, ...]:
        return tuple(event for event in self.events if event.pattern_id == int(pattern_id))

    def co_occurrences(self) -> Mapping[float, Tuple[int, ...]]:
        """Frames where more than one pattern fired. These are unordered, deliberately."""
        by_frame: Dict[float, set] = {}
        for event in self.events:
            by_frame.setdefault(event.time, set()).add(event.pattern_id)
        return {time: tuple(sorted(ids)) for time, ids in sorted(by_frame.items())
                if len(ids) > 1}

    def spans(self, pattern_id: int) -> Tuple[EventSpan, ...]:
        occurrences = self.for_pattern(pattern_id)
        result = []
        for earlier, later in zip(occurrences, occurrences[1:]):
            coverage, observed, missing = self.grid.coverage_between(earlier.time, later.time)
            result.append(EventSpan(
                pattern_id=int(pattern_id), earlier=earlier.time, later=later.time,
                duration=later.time - earlier.time, time_units=self.grid.time_units,
                frames_observed_between=observed, frames_unobserved_between=missing,
                coverage=coverage))
        return tuple(result)

    def describe(self) -> Dict[str, Any]:
        co_occurring = self.co_occurrences()
        return {
            "schema": EVENT_SCHEMA,
            "n_events": len(self.events),
            "n_patterns": len(self.pattern_ids),
            "mode": self.mode,
            "grid": self.grid.describe(),
            "time_units_confirmed_by_members": self.units_confirmed_by_members,
            "time_units_uncarried_by_members": self.units_uncarried_by_members,
            "frames_with_co_occurrence": len(co_occurring),
            "co_occurrences": {str(time): list(ids) for time, ids in co_occurring.items()},
            "co_occurrence_basis": (
                "events sharing a frame are simultaneous and carry no order between them; a "
                "succession read off their list position would be an artefact of sorting"),
            "events": [event.describe() for event in self.events],
            "spans": [span.describe() for pattern_id in self.pattern_ids
                      for span in self.spans(pattern_id)],
            "claim_boundary": CLAIM_BOUNDARY,
        }


def events_from_catalogue(catalogue: PatternCatalogue, *,
                          grid: ObservationGrid) -> EventSeries:
    """Read a T4E.3 catalogue as a timed event series, or refuse to invent its clock."""
    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a complete T4E.3 PatternCatalogue")
    if not isinstance(grid, ObservationGrid):
        raise InvalidParameterError(
            "grid", type(grid).__name__,
            "an explicit ObservationGrid. The frames that were searched are not recoverable "
            "from a catalogue of what was found")
    if not catalogue.patterns:
        raise InvalidParameterError(
            "catalogue.patterns", 0,
            "at least one pattern. An empty catalogue is an unrun clustering, and reporting it "
            "as a series with no events would present a missing input as a scientific absence")

    searched = set(float(frame) for frame in grid.frames)
    modes = {member.mode for pattern in catalogue for member in pattern.members}
    if not modes:
        raise InvalidParameterError(
            "catalogue.patterns.members", 0,
            "at least one member occurrence across the catalogue")
    if len(modes) > 1:
        raise InvalidParameterError(
            "catalogue.patterns.members.mode", sorted(modes),
            "one scale mode across the whole catalogue. The modes answer different questions -- "
            "whether a configuration recurs at any scale, or at this one -- and ordering them "
            "into one series would sequence two questions as though they were one")
    mode = modes.pop()

    seen: Dict[str, int] = {}
    confirmed = 0
    uncarried = 0
    events = []
    for pattern in catalogue:
        for member in pattern.members:
            identity = _identity(member.key)
            if identity in seen:
                raise InvalidParameterError(
                    "catalogue.patterns.members.key", identity,
                    "one occurrence of each constellation identity. The same key appears in "
                    "patterns %d and %d; sequencing it twice would manufacture a transition"
                    % (seen[identity], pattern.pattern_id))
            seen[identity] = pattern.pattern_id

            time = float(member.time)
            if not math.isfinite(time):
                raise InvalidParameterError(
                    "catalogue.patterns.members.time", member.time,
                    "a finite occurrence time")
            if time not in searched:
                raise InvalidParameterError(
                    "catalogue.patterns.members.time", time,
                    "an occurrence at a searched frame. This one is at a time the grid never "
                    "examined, so the event asserts a sighting where nothing looked")

            carried = member.time_units
            if carried is None:
                uncarried += 1
            elif str(carried) != grid.time_units:
                raise InvalidParameterError(
                    "catalogue.patterns.members.time_units", carried,
                    "the unit the grid declares (%r). A member measured in one unit and a grid "
                    "declared in another would make every span a mixture of the two"
                    % grid.time_units)
            else:
                confirmed += 1

            events.append(PatternEvent(
                pattern_id=int(pattern.pattern_id), time=time, identity=identity,
                cardinality=int(member.cardinality), mode=member.mode))

    events.sort(key=lambda event: (event.time, event.pattern_id, event.identity))
    return EventSeries(events=tuple(events), grid=grid, mode=mode,
                       units_confirmed_by_members=confirmed,
                       units_uncarried_by_members=uncarried)


__all__ = [
    "EVENT_SCHEMA", "CADENCE_TOLERANCE", "CLAIM_BOUNDARY",
    "COVERAGE_UNDECLARED", "COVERAGE_MEASURED", "COVERAGE_INCOMPLETE",
    "ObservationGrid", "PatternEvent", "EventSpan", "EventSeries", "events_from_catalogue",
]
