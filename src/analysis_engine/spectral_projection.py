"""Evidence projection: a rule put back on the map it was measured on (`T4F.5`).

T4F.3 tested a declared family and T4F.4 made the corrected table askable from either end. Both
answer in pattern identities -- integers -- and an integer is not evidence. This module takes a
pattern or a rule and says **where on the parent grid, in which frames, and at what value of the
field** the coefficients that constitute it actually were, so that "why does the platform believe
this?" is answered by looking at the record rather than at the catalogue.

The mapping is direct rather than inferred, and that is T3.5.7's doing: the bank is undecimated,
every scale already lives on the parent grid, and T4D.1 subtracts each level's own analysis delay
(D88) before a position is ever reported. There is no interpolation step here to be wrong about.
What there is, is a set of claims that would each be a lie if made carelessly.

**1. A coefficient maximum is not a pixel, and projecting it onto one would overstate what the
transform knows.** A coefficient is the response of a filter whose support covers many parent
cells; the maximum names where that filter sat, not where the structure's edge is. So a
projection here is a **footprint** -- the parent cells inside the transform's own filter support
at that scale, taken from the same `filter_support` the R13 margin is cut with -- and never a
point. The extent is the measured support, not the dyadic octave label the scale ratios use: at
haar level 4 those are 16 cells and 8 cells and they are different numbers about different
things.

**2. A detail coefficient peaks at a flank, so a footprint is where the coefficient is and not
where the weather is.** A detail wavelet is derivative-like: a symmetric blob has *two* maxima,
about one analysing width out along the axis the band high-passes, and none at its centre. On
the acceptance record the planted vortex sits at (32, 32) and its `LH` maximum is at
(31.9, 38.1) -- exact across the axis the band resolves, six cells out along the one it does
not. Reporting that cell as the structure's location would be wrong by six cells while looking
precise to two decimals, and the footprint is what makes the projection true instead: the
planted centre is inside it.

**And it is inside it only while the level that found the structure is wide enough to reach
back to it**, which is a boundary this module measures rather than assumes. Over the twenty-four
frames of the acceptance record the offset from the planted centre to the maximum is `1.08`
times the vortex's own width at level 4 and `1.26` times it at level 5, while the reach of a
footprint is half the filter support -- 8 cells and 16 cells. So the level-4 projection holds
the planted cell out to frame 9, while the vortex is still narrower than that reach, and loses
it in all fifteen frames after; the level-5 projection holds it in all fifteen frames it detects
anything. A coefficient at a level far finer than the structure points away from that
structure by more than its own reach, and the projection is then honestly empty of it. Evidence
should be read at the level that resolves the thing.

**3. Coordinates are the grid's to give, and most grids do not give degrees.** A `latlon` grid
yields latitudes and longitudes; a `cartesian` grid yields offsets in metres from the crop's own
origin, which is a distance and not a place; a `pixel` grid yields cells and nothing else. Each
is reported under its own name, and a grid that cannot say where it is on the Earth is never
made to guess. The same holds vertically -- a level with no declared axis is a number, not
hectopascals -- and in time: a frame index is reported as a frame unless the record carries a
calendar.

**4. The field's values, and its anomalies, are inputs and not derivations.** The values under a
footprint are read from the record the coefficients were taken from, which must be supplied; an
anomaly is read from a supplied anomaly record. Neither is invented from the other. In
particular a raw value is never called an anomaly and the record's own time mean is never
quietly subtracted to make one: which baseline was removed is a scientific decision belonging to
whoever removed it (R11), and this module reports what it was handed or reports that it was
handed nothing.

**5. The instances are the ones the rule was counted on, decided by the code that counted it.**
A projected instance list that disagreed with the support a correction was paid on would be a
second study wearing the first one's q-value. So the per-anchor decision lives once, in
`spectral_sequences.anchor_verdicts`, `count_sequence` tallies it and this module lists it; and
before publishing, every enumerated total is reconciled against the figures the rule already
published. A disagreement refuses rather than reports, because it means the series in hand is
not the series the rule was measured on. Anchors that did *not* support the rule are listed too,
under their own verdicts -- eligible and unfollowed, truncated by the record's end, or spanning
time nobody observed -- because a reader shown only the successes cannot see the denominator.

**What is not claimed.** A footprint is where a coefficient of this transform was, at this
scale, in this frame. It is not a detection of any named phenomenon, not an attribution, and the
sequence of two footprints is not a mechanism. The scale mode does not matter here and that is
worth saying, because it mattered in T4F.4: a projection reads the record's own positions rather
than the signature's, so a scale-invariant catalogue projects exactly as a scale-specific one
does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.scale_signature import _interior_halfwidth
from src.analysis_engine.spectral_clustering import ConstellationPattern, PatternCatalogue
from src.analysis_engine.spectral_constellation import ConstellationSet, FrameConstellation
from src.analysis_engine.spectral_events import EventSeries
from src.analysis_engine.spectral_precursors import (
    PRECURSOR_CLAIM_BOUNDARY, PrecursorRule,
)
from src.analysis_engine.spectral_sequences import (
    TransitionWindow, WINDOW_INCOMPLETE, WINDOW_MEASURED, WINDOW_TRUNCATED, anchor_verdicts,
    times_by_pattern, window_completions,
)
from src.analysis_engine.spectral_tracking import spans_globe
from src.core.errors import InvalidParameterError
from src.core.level_axis import units_of
from src.physical_core.sequence import FieldSequence

PROJECTION_SCHEMA = "spectral-evidence-projection/v1"
INSTANCE_SCHEMA = "spectral-rule-instances/v1"

#: What kind of horizontal coordinate the grid was able to give. Three grids, three different
#: true statements, and none of them is "unknown, so assume degrees".
PLACE_GEOGRAPHIC = "GEOGRAPHIC_DEGREES"
PLACE_OFFSET = "PROJECTED_OFFSET_FROM_CROP_ORIGIN"
PLACE_CELLS_ONLY = "GRID_CELLS_ONLY"

#: What the vertical coordinate is. A bare number set by hand is a real state of the tree, and
#: the honest report of it is that nobody declared what it is a number of.
LEVEL_DECLARED = "DECLARED_VERTICAL_COORDINATE"
LEVEL_UNDECLARED_AXIS = "VALUE_WITHOUT_A_DECLARED_AXIS"
LEVEL_NONE = "NO_VERTICAL_COORDINATE"

#: What a frame's time is. A number on a clock whose epoch nobody declared is a frame, and
#: rendering it as a date would invent an epoch.
TIME_CALENDAR = "CALENDAR_TIMESTAMP"
TIME_FRAME = "FRAME_ON_THE_RECORD_CLOCK"

#: How one antecedent occurrence turned out. The last three are why a support of six is not the
#: same finding as a confidence of six sixths.
INSTANCE_SUPPORTING = "SUPPORTING"
INSTANCE_UNFOLLOWED = "ELIGIBLE_AND_NOT_FOLLOWED"
INSTANCE_TRUNCATED = "INELIGIBLE_TRUNCATED_BY_RECORD"
INSTANCE_UNOBSERVED = "INELIGIBLE_SPANS_UNOBSERVED_TIME"

_VERDICT_OF_WINDOW = {
    (WINDOW_MEASURED, True): INSTANCE_SUPPORTING,
    (WINDOW_MEASURED, False): INSTANCE_UNFOLLOWED,
    (WINDOW_TRUNCATED, True): INSTANCE_TRUNCATED,
    (WINDOW_TRUNCATED, False): INSTANCE_TRUNCATED,
    (WINDOW_INCOMPLETE, True): INSTANCE_UNOBSERVED,
    (WINDOW_INCOMPLETE, False): INSTANCE_UNOBSERVED,
}

#: The sentence that stops a footprint being read as a sighting of something.
PROJECTION_CLAIM_BOUNDARY = (
    "A footprint is the set of parent-grid cells within this transform's own filter support of "
    "a coefficient maximum, in one frame, at one scale and orientation. It is where a "
    "coefficient was, not where a structure is: a detail band peaks at a flank about one "
    "analysing width from a symmetric structure's centre, so a member's footprint is offset "
    "from the thing that excited it by an amount that grows with scale. It is not a detection "
    "of any named phenomenon, it is not an attribution, and two footprints in succession are "
    "not a cause, a driver, a mechanism, a trigger, a forecast or an intervention. The field "
    "values reported under a footprint are the record's own values at those cells and are not "
    "evidence that the coefficient measured them.")

#: Said wherever a share of coefficient energy is published, because the arithmetic invites
#: exactly the reading it does not support.
REDUNDANCY_NOTE = (
    "the bank is undecimated and therefore redundant, so per-scale coefficient energies do not "
    "partition the field's variance and these shares are of this pattern's own members only. A "
    "structure whose width straddles two levels excites both, and appears in both shares.")


# ------------------------------------------------------------------------------ where and when


@dataclass(frozen=True)
class Place:
    """One position on the parent grid, named as far as the grid can name it and no further."""

    row: float
    col: float
    row_cell: int
    col_cell: int
    basis: str
    coordinates: Mapping[str, float] = dc_field(default_factory=dict)
    units: Optional[str] = None
    refusal: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", dict(self.coordinates))

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "row": self.row, "col": self.col,
            "row_cell": self.row_cell, "col_cell": self.col_cell,
            "cell_units": "parent-grid cells, y down and x across",
            "basis": self.basis,
            "coordinates": dict(self.coordinates),
            "coordinate_units": self.units,
        }
        if self.refusal is not None:
            record["coordinates_refused"] = self.refusal
        return record


@dataclass(frozen=True)
class Moment:
    """One frame's time, as a frame, and as a date only where the record carries one."""

    time: float
    time_units: Optional[str]
    frame_index: int
    basis: str
    timestamp: Optional[str] = None
    refusal: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "time": self.time, "time_units": self.time_units,
            "frame_index": self.frame_index, "basis": self.basis,
            "timestamp": self.timestamp,
        }
        if self.refusal is not None:
            record["timestamp_refused"] = self.refusal
        return record


@dataclass(frozen=True)
class Vertical:
    """The one vertical coordinate a `CoefficientField` carries, or the absence of one."""

    status: str
    value: Optional[float]
    axis: Optional[str]
    units: Optional[str]
    note: str

    def describe(self) -> Dict[str, Any]:
        return {"status": self.status, "value": self.value, "axis": self.axis,
                "units": self.units, "note": self.note}


@dataclass(frozen=True)
class FieldSample:
    """The record's own values over one footprint, with the count they were taken over."""

    quantity: str
    at_centre: Optional[float]
    mean: float
    minimum: float
    maximum: float
    extreme_magnitude: float
    n_cells: int
    units: Optional[str]

    def describe(self) -> Dict[str, Any]:
        return {"quantity": self.quantity, "at_centre_cell": self.at_centre, "mean": self.mean,
                "minimum": self.minimum, "maximum": self.maximum,
                "extreme_magnitude": self.extreme_magnitude, "n_cells": self.n_cells,
                "units": self.units}


# ------------------------------------------------------------------------------ the footprint


@dataclass(frozen=True)
class Footprint:
    """One member's coefficient, put back on the cells the transform gathered it from.

    `columns` is a tuple of inclusive `(low, high)` segments rather than one range because a
    grid whose columns close on a circle of longitude has a seam, and a footprint straddling it
    covers two segments at opposite ends of the array. Reporting the single range that spans
    them would claim a footprint reaching most of the way round the world.
    """

    track_id: int
    band: str
    scale_label: str
    moment: Moment
    centre: Place
    rows: Tuple[int, int]
    columns: Tuple[Tuple[int, int], ...]
    support_cells: int
    radius_cells: float
    support_basis: str
    octave_cells: float
    strength: float
    strength_units: Optional[str]
    strength_sigma: Optional[float]
    phase: Optional[float]
    wraps_seam: bool
    clipped_by_crop: bool
    values: Optional[FieldSample] = None
    anomalies: Optional[FieldSample] = None
    refusals: Mapping[str, str] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "refusals", dict(self.refusals))

    @property
    def n_cells(self) -> int:
        rows = self.rows[1] - self.rows[0] + 1
        return rows * sum(high - low + 1 for low, high in self.columns)

    def contains(self, row: float, col: float) -> bool:
        """Is this parent-grid position inside the footprint? Cell-wise, not sub-pixel."""
        r, c = int(round(float(row))), int(round(float(col)))
        if not self.rows[0] <= r <= self.rows[1]:
            return False
        return any(low <= c <= high for low, high in self.columns)

    def cells(self) -> Tuple[Tuple[int, int], ...]:
        """Every parent cell inside the footprint, row-major. Bounded by the filter support."""
        return tuple((r, c) for r in range(self.rows[0], self.rows[1] + 1)
                     for low, high in self.columns for c in range(low, high + 1))

    def describe(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "track_id": self.track_id,
            "band": self.band,
            "scale_label": self.scale_label,
            "moment": self.moment.describe(),
            "centre": self.centre.describe(),
            "rows": list(self.rows),
            "columns": [list(item) for item in self.columns],
            "n_cells": self.n_cells,
            "support_cells": self.support_cells,
            "radius_cells": self.radius_cells,
            "support_basis": self.support_basis,
            "octave_cells": self.octave_cells,
            "extent_basis": (
                "the transform's own filter support at this level, in parent cells, the same "
                "number R13 cuts the contaminated margin with. It is NOT the dyadic octave "
                "label the scale ratios use, which is %g cells here" % self.octave_cells),
            "strength": self.strength,
            "strength_units": self.strength_units,
            "strength_sigma": self.strength_sigma,
            "phase": self.phase,
            "wraps_seam": self.wraps_seam,
            "clipped_by_crop": self.clipped_by_crop,
            "values": None if self.values is None else self.values.describe(),
            "anomalies": None if self.anomalies is None else self.anomalies.describe(),
        }
        if self.refusals:
            record["refused"] = dict(self.refusals)
        return record


@dataclass(frozen=True)
class OccurrenceProjection:
    """One occurrence of one pattern: every member's footprint, and what they share."""

    pattern_id: int
    key: Tuple[Any, ...]
    moment: Moment
    footprints: Tuple[Footprint, ...]
    common_rows: Optional[Tuple[int, int]]
    common_columns: Tuple[Tuple[int, int], ...]
    vertical: Vertical

    @property
    def cardinality(self) -> int:
        return len(self.footprints)

    @property
    def orientations(self) -> Tuple[str, ...]:
        """Each member's band orientation, which is what decides whether the flanks agree."""
        return tuple(str(item.band).split("/")[-1] for item in self.footprints)

    @property
    def common_cells(self) -> Tuple[Tuple[int, int], ...]:
        """The cells inside *every* member's footprint. Empty is a real and frequent answer."""
        if self.common_rows is None or not self.common_columns:
            return ()
        return tuple((r, c) for r in range(self.common_rows[0], self.common_rows[1] + 1)
                     for low, high in self.common_columns for c in range(low, high + 1))

    def describe(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "key": list(self.key),
            "moment": self.moment.describe(),
            "cardinality": self.cardinality,
            "vertical": self.vertical.describe(),
            "footprints": [item.describe() for item in self.footprints],
            "common_rows": None if self.common_rows is None else list(self.common_rows),
            "common_columns": [list(item) for item in self.common_columns],
            "n_common_cells": len(self.common_cells),
            "orientations": list(self.orientations),
            "common_basis": (
                "the cells inside every member's footprint. It is not a location of anything: "
                "two members whose bands resolve different axes have flanks pointing in "
                "different directions and their intersection straddles what excited them, "
                "while two members of the same orientation at different levels have flanks "
                "pointing the same way and intersect in a region beside it. On the acceptance "
                "record the LH/HL pair of one level intersects on the planted cell in 11 of 11 "
                "occurrences and the LH/LH pair of two levels excludes it in 11 of 11. An "
                "empty intersection is also a real answer and is not evidence of anything"),
            "claim_boundary": PROJECTION_CLAIM_BOUNDARY,
            **({"common_caution": (
                "every member of this occurrence came from the same band orientation, so their "
                "flanks point the same way and this intersection sits beside whatever excited "
                "them rather than over it")}
               if len(set(self.orientations)) == 1 and self.common_cells else {}),
        }


@dataclass(frozen=True)
class PatternProjection:
    """Every occurrence of one clustered pattern, on the map, with its per-scale shares."""

    pattern_id: int
    occurrences: Tuple[OccurrenceProjection, ...]
    scale_shares: Tuple[Mapping[str, Any], ...]
    band_shares: Tuple[Mapping[str, Any], ...]
    n_members: int

    def __len__(self) -> int:
        return len(self.occurrences)

    def __iter__(self) -> Iterator[OccurrenceProjection]:
        return iter(self.occurrences)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": PROJECTION_SCHEMA,
            "pattern_id": self.pattern_id,
            "n_members": self.n_members,
            "n_occurrences_projected": len(self.occurrences),
            "scale_shares": [dict(item) for item in self.scale_shares],
            "band_shares": [dict(item) for item in self.band_shares],
            "share_basis": (
                "each scale's share of the summed squared coefficient magnitude of this "
                "pattern's members: %s" % REDUNDANCY_NOTE),
            "occurrences": [item.describe() for item in self.occurrences],
            "claim_boundary": PROJECTION_CLAIM_BOUNDARY,
        }


# ------------------------------------------------------------------------------ the instances


@dataclass(frozen=True)
class RuleInstance:
    """One antecedent occurrence and how it turned out, projected at both ends."""

    antecedent: int
    consequent: int
    anchor: float
    verdict: str
    completions: Tuple[float, ...]
    lags: Tuple[float, ...]
    antecedents: Tuple[OccurrenceProjection, ...]
    consequents: Tuple[OccurrenceProjection, ...]

    @property
    def supporting(self) -> bool:
        return self.verdict == INSTANCE_SUPPORTING

    def describe(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "anchor": self.anchor,
            "verdict": self.verdict,
            "completions": list(self.completions),
            "lags": list(self.lags),
            "antecedent_occurrences": [item.describe() for item in self.antecedents],
            "consequent_occurrences": [item.describe() for item in self.consequents],
        }


@dataclass(frozen=True)
class RuleProjection:
    """A rule's own figures, echoed, beside the occurrences they were counted over."""

    antecedent: int
    consequent: int
    window: TransitionWindow
    status: str
    figures: Mapping[str, Any]
    instances: Tuple[RuleInstance, ...]
    reconciliation: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "figures", dict(self.figures))
        object.__setattr__(self, "reconciliation", dict(self.reconciliation))

    @property
    def supporting(self) -> Tuple[RuleInstance, ...]:
        return tuple(item for item in self.instances if item.supporting)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": INSTANCE_SCHEMA,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "window": self.window.describe(),
            "status": self.status,
            "figures": dict(self.figures),
            "figures_basis": (
                "read from the rule exactly as T4F.3 wrote them. Nothing here is recomputed: a "
                "projection is a view of a test that has already been corrected"),
            "n_instances": len(self.instances),
            "n_supporting": len(self.supporting),
            "instances": [item.describe() for item in self.instances],
            "reconciliation": dict(self.reconciliation),
            "reconciliation_basis": (
                "every enumerated total is checked against the figure the rule published, "
                "because an instance list that disagreed with the support a correction was "
                "paid on would be a second study wearing the first one's q-value"),
            "claim_boundary": PROJECTION_CLAIM_BOUNDARY,
            "inference_claim_boundary": PRECURSOR_CLAIM_BOUNDARY,
        }


# ------------------------------------------------------------------------------ the geography


class Geography:
    """The grid, the clock and the record a projection reads, bound together and checked once.

    Built by `geography_of`. Holding it is what makes a projection cheap to repeat: the filter
    supports, the seam and the frame index are all properties of the decomposition rather than
    of any one pattern, and re-deriving them per footprint is how two footprints of one record
    end up on two different maps.
    """

    def __init__(self, field: Any, *, time_units: Optional[str],
                 values: Optional[FieldSequence], anomalies: Optional[FieldSequence],
                 supports: Mapping[str, Mapping[str, Any]]) -> None:
        self.field = field
        self.grid = field.grid
        self.time_units = time_units
        self.values = values
        self.anomalies = anomalies
        self.supports = {str(key): dict(item) for key, item in supports.items()}
        self.wraps = spans_globe(field.grid)
        self.height, self.width = int(field.grid.shape[0]), int(field.grid.shape[1])
        self._times = np.asarray(field.times, dtype=np.float64)
        self._index = {float(value): position
                       for position, value in enumerate(self._times)}
        # A decomposition's clock is float seconds by construction -- `decompose_sequence`
        # hands it `FieldSequence.times_seconds` -- so the field itself cannot say whether its
        # numbers are dates. The record can: a sequence built from datetime64 keeps the
        # timestamps it was given. So a frame is dated when, and only when, the record whose
        # values are being read is dated, and otherwise it is a frame.
        self._raw = None
        if values is not None and getattr(values, "time_kind", None) == "datetime64":
            self._raw = np.asarray(values.raw_times, dtype="datetime64[ns]")
        self._calendar = self._raw is not None

    # ------------------------------------------------------------------ the clock, in public

    @property
    def has_calendar(self) -> bool:
        """Whether a frame of this record can be dated at all."""
        return self._calendar

    def calendar_times(self) -> Optional[np.ndarray]:
        """The record's own timestamps, or `None` when it has none.

        Public because T4F.6 must convert a lag in frames into a lag in hours before it can
        compare one with a phenomenon's declared lead, and the cadence it needs is the one this
        record actually carries. Deriving it anywhere else would let a gate measure its lead on
        a clock the projection would refuse to print.
        """
        return None if self._raw is None else np.asarray(self._raw)

    # ------------------------------------------------------------------ horizontal

    def place(self, row: float, col: float) -> Place:
        """Name a parent-grid position as far as this grid's geometry can name it."""
        row, col = float(row), float(col)
        cell_row = int(np.clip(round(row), 0, self.height - 1))
        cell_col = int(round(col))
        cell_col = (cell_col % self.width if self.wraps
                    else int(np.clip(cell_col, 0, self.width - 1)))
        grid = self.grid
        if grid.capability("has_latitude", False):
            latitude = float(grid.lat0) + float(grid.dy) * row
            longitude = float(grid.lon0) + float(grid.dx) * col
            if self.wraps:
                longitude = float(grid.lon0) + (longitude - float(grid.lon0)) % 360.0
            return Place(row=row, col=col, row_cell=cell_row, col_cell=cell_col,
                         basis=PLACE_GEOGRAPHIC,
                         coordinates={"latitude": latitude, "longitude": longitude},
                         units="degrees north / degrees east, in this grid's own convention"
                               " (row 0 at latitude %g, spacing %g)"
                               % (float(grid.lat0), float(grid.dy)))
        if grid.is_physical:
            return Place(row=row, col=col, row_cell=cell_row, col_cell=cell_col,
                         basis=PLACE_OFFSET,
                         coordinates={"northing": float(grid.dy) * row,
                                      "easting": float(grid.dx) * col},
                         units=grid.length_units,
                         refusal=("this grid declares a metric but no latitude, so these are "
                                  "distances from the crop's own cell (0, 0) and not a "
                                  "position on the Earth"))
        return Place(row=row, col=col, row_cell=cell_row, col_cell=cell_col,
                     basis=PLACE_CELLS_ONLY,
                     refusal=("a %s grid declares no length and no latitude, so this position "
                              "is an array index and nothing else. Reporting degrees for it "
                              "would invent a geography the record has not got" % grid.kind))

    # ------------------------------------------------------------------ vertical

    def vertical(self) -> Vertical:
        """The one level the decomposition carries, under its declared axis or under none."""
        level = getattr(self.field, "level", None)
        axis = getattr(self.field, "level_axis", None)
        if level is None:
            return Vertical(
                status=LEVEL_NONE, value=None, axis=None, units=None,
                note=("this decomposition declares no vertical coordinate. Every footprint "
                      "here is at whatever level the source field was, and this record cannot "
                      "say which"))
        if axis is None:
            return Vertical(
                status=LEVEL_UNDECLARED_AXIS, value=float(level), axis=None, units=None,
                note=("a level was set without the coordinate it is a value of, so it is a "
                      "number. Calling it hectopascals is the assumption TG1.5 exists to "
                      "refuse"))
        return Vertical(status=LEVEL_DECLARED, value=float(level), axis=str(axis),
                        units=units_of(str(axis)),
                        note="the vertical coordinate this decomposition declares")

    # ------------------------------------------------------------------ time

    def when(self, time: float) -> Moment:
        """Locate a frame on the record's clock, and date it only if the record is dated."""
        key = float(time)
        if key not in self._index:
            raise InvalidParameterError(
                "time", time,
                "a frame of this decomposition's own time axis. A constellation whose frame is "
                "not one of the frames that were decomposed came from a different record, and "
                "projecting it would place one record's pattern on another record's map")
        index = self._index[key]
        if self._calendar:
            return Moment(time=key, time_units=self.time_units, frame_index=index,
                          basis=TIME_CALENDAR, timestamp=str(self._raw[index]))
        return Moment(
            time=key, time_units=self.time_units, frame_index=index, basis=TIME_FRAME,
            refusal=("the record supplied carries numbers on a clock with no declared epoch "
                     "or calendar -- or no record was supplied at all -- so this frame has an "
                     "index and a value and not a date"))

    # ------------------------------------------------------------------ extent

    def extent(self, row: float, col: float, radius: float
               ) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...], bool, bool]:
        """The cells within `radius` of a position: rows, column segments, wrap, clip."""
        low_row = int(math.ceil(float(row) - radius))
        high_row = int(math.floor(float(row) + radius))
        clipped = low_row < 0 or high_row > self.height - 1
        low_row = max(0, low_row)
        high_row = min(self.height - 1, high_row)
        if high_row < low_row:                       # a radius smaller than the gap to a cell
            low_row = high_row = int(np.clip(round(float(row)), 0, self.height - 1))

        low_col = int(math.ceil(float(col) - radius))
        high_col = int(math.floor(float(col) + radius))
        if high_col < low_col:
            low_col = high_col = int(round(float(col)))
        wraps = False
        if self.wraps:
            if high_col - low_col + 1 >= self.width:
                # A filter reaching the whole circle covers every column once rather than
                # several times, and it still crosses the seam: saying otherwise would report a
                # footprint that goes all the way round as one that stops at the edge.
                wraps = True
                segments = ((0, self.width - 1),)
            else:
                low = low_col % self.width
                high = high_col % self.width
                if low <= high:
                    segments = ((low, high),)
                else:
                    wraps = True
                    segments = ((0, high), (low, self.width - 1))
        else:
            clipped = clipped or low_col < 0 or high_col > self.width - 1
            low = max(0, low_col)
            high = min(self.width - 1, high_col)
            segments = ((low, high),) if high >= low else ()
        return (low_row, high_row), segments, wraps, clipped

    # ------------------------------------------------------------------ the record's values

    def sample(self, sequence: Optional[FieldSequence], quantity: str, index: int,
               rows: Tuple[int, int], columns: Sequence[Tuple[int, int]],
               centre: Place) -> Optional[FieldSample]:
        """The supplied record's values over one footprint, or nothing at all."""
        if sequence is None or not columns:
            return None
        values = np.asarray(sequence.at(index).data.detach().cpu().numpy(), dtype=np.float64)
        blocks = [values[rows[0]:rows[1] + 1, low:high + 1] for low, high in columns]
        gathered = np.concatenate([block.reshape(-1) for block in blocks])
        return FieldSample(
            quantity=quantity,
            at_centre=float(values[centre.row_cell, centre.col_cell]),
            mean=float(gathered.mean()), minimum=float(gathered.min()),
            maximum=float(gathered.max()),
            extreme_magnitude=float(np.abs(gathered).max()),
            n_cells=int(gathered.size),
            units=getattr(sequence, "units", None))

    def describe(self) -> Dict[str, Any]:
        sample = self.place(0.0, 0.0)
        return {
            "schema": PROJECTION_SCHEMA,
            "grid": self.grid.describe(),
            "grid_kind": self.grid.kind,
            "shape": [self.height, self.width],
            "coordinate_basis": sample.basis,
            "coordinate_refusal": sample.refusal,
            "columns_close_on_a_circle": self.wraps,
            "n_frames": int(len(self._times)),
            "time_units": self.time_units,
            "time_basis": TIME_CALENDAR if self._calendar else TIME_FRAME,
            "vertical": self.vertical().describe(),
            "supports": {key: dict(item) for key, item in self.supports.items()},
            "values_supplied": self.values is not None,
            "anomalies_supplied": self.anomalies is not None,
            "wavelet_family": self.field.wavelet_family,
            "claim_boundary": PROJECTION_CLAIM_BOUNDARY,
        }


def scale_supports(field: Any) -> Dict[str, Dict[str, Any]]:
    """Each scale's filter support in parent cells, from the transform's own filters.

    The same number `detect_features` cuts the R13 margin with, and read from the same place, so
    a footprint and the margin that excluded its neighbours cannot disagree about how far this
    filter reaches.
    """
    supports: Dict[str, Dict[str, Any]] = {}
    for position, scale in enumerate(field.scales):
        record = _interior_halfwidth(field, scale, position + 1)
        support = int(record["support_parent_px"])
        supports[str(scale)] = {
            "support_cells": support,
            "radius_cells": support / 2.0,
            "level": int(record["level"]),
            "basis": record["basis"],
        }
    return supports


def geography_of(field: Any, *, time_units: Optional[str] = None,
                 values: Optional[FieldSequence] = None,
                 anomalies: Optional[FieldSequence] = None) -> Geography:
    """Bind a decomposition to the record it came from, refusing anything that disagrees.

    `values` is the physical record the coefficients were taken from and `anomalies` a record of
    departures from a baseline somebody else declared and removed (R11). Both are optional and
    neither is derived from the other: a projection with no field supplied reports footprints
    and says that no values were supplied, which is a smaller answer than the wrong one.
    """
    for name in ("grid", "times", "scales", "wavelet_family"):
        if not hasattr(field, name):
            raise InvalidParameterError(
                "field", type(field).__name__,
                "a CoefficientField. A projection needs the grid, the clock and the filters "
                "that produced these coefficients, and a bare array carries none of them")
    for name, sequence in (("values", values), ("anomalies", anomalies)):
        if sequence is None:
            continue
        if not isinstance(sequence, FieldSequence):
            raise InvalidParameterError(
                "geography_of.%s" % name, type(sequence).__name__,
                "a FieldSequence of the record these coefficients were taken from")
        if len(sequence) != int(field.n_times):
            raise InvalidParameterError(
                "geography_of.%s" % name, len(sequence),
                "one frame per decomposed frame (%d). A record of a different length cannot be "
                "the record these coefficients came from" % int(field.n_times))
        if tuple(sequence.grid.shape) != tuple(field.grid.shape):
            raise InvalidParameterError(
                "geography_of.%s.grid" % name, tuple(sequence.grid.shape),
                "the parent grid these coefficients live on %s. Sampling one grid at another "
                "grid's indices reports the wrong cells and looks entirely plausible doing it"
                % (tuple(field.grid.shape),))
        if not np.array_equal(np.asarray(sequence.times_seconds), np.asarray(field.times)):
            raise InvalidParameterError(
                "geography_of.%s.times" % name, "a different time axis",
                "the same frames the decomposition ran on, in the same order")
    return Geography(field, time_units=time_units, values=values, anomalies=anomalies,
                     supports=scale_supports(field))


# ------------------------------------------------------------------------------ projecting


def _member_footprint(node: Any, feature: Any, geography: Geography) -> Footprint:
    """One node, with the feature it was built from, put on the map."""
    provenance = dict(getattr(feature, "provenance", {}) or {})
    if provenance.get("alignment_applied") is False:
        raise InvalidParameterError(
            "constellation.features.provenance.alignment_applied", False,
            "a detection whose positions were aligned to the parent grid (D88). Without the "
            "analysis delay removed, a coefficient's index is at the filter's anchor rather "
            "than at the place it responded to -- half a cell at level 1 and tens of cells at "
            "coarse levels -- so projecting it onto the map would draw the footprint in the "
            "wrong place by an amount that grows with scale")
    scale_label = str(provenance.get("scale_label", node.band.split("/")[0].lstrip("L")))
    if scale_label not in geography.supports:
        raise InvalidParameterError(
            "constellation.features.provenance.scale_label", scale_label,
            "a scale this decomposition carries (%s). A member measured at a scale the field "
            "does not have came from a different decomposition"
            % sorted(geography.supports))
    support = geography.supports[scale_label]
    radius = float(support["radius_cells"])
    row, col = float(node.coords["row"]), float(node.coords["col"])
    moment = geography.when(node.time)
    centre = geography.place(row, col)
    rows, columns, wraps, clipped = geography.extent(row, col, radius)

    refusals: Dict[str, str] = {}
    if node.strength_sigma is None:
        refusals["strength_sigma"] = (
            "the detection recorded no per-band threshold, so this magnitude cannot be put on "
            "a common footing with another band's")
    if geography.values is None:
        refusals["values"] = (
            "no physical record was supplied, so the field's values under this footprint are "
            "unknown. They are not zero and they are not the coefficient")
    if geography.anomalies is None:
        refusals["anomalies"] = (
            "no anomaly record was supplied. A raw value is not an anomaly, and subtracting "
            "this record's own time mean to make one would be a baseline nobody declared (R11)")

    return Footprint(
        track_id=int(node.track_id), band=str(node.band), scale_label=scale_label,
        moment=moment, centre=centre, rows=rows, columns=columns,
        support_cells=int(support["support_cells"]), radius_cells=radius,
        support_basis=str(support["basis"]), octave_cells=float(node.scale),
        strength=float(node.strength), strength_units=node.strength_units,
        strength_sigma=node.strength_sigma, phase=node.phase,
        wraps_seam=wraps, clipped_by_crop=clipped,
        values=geography.sample(geography.values, "field value", moment.frame_index,
                                rows, columns, centre),
        anomalies=geography.sample(geography.anomalies, "anomaly", moment.frame_index,
                                   rows, columns, centre),
        refusals=refusals)


def common_extent(footprints: Sequence[Footprint]
                  ) -> Tuple[Optional[Tuple[int, int]], Tuple[Tuple[int, int], ...]]:
    """The rows and column segments common to every footprint, as boxes rather than sets.

    Public because the intersection is a claim a caller may want to make about footprints this
    module did not group -- two frames of one track, say -- and because it is where the
    arithmetic that decides an empty answer lives.
    """
    low = max(item.rows[0] for item in footprints)
    high = min(item.rows[1] for item in footprints)
    rows = (low, high) if high >= low else None
    segments: List[Tuple[int, int]] = list(footprints[0].columns)
    for item in footprints[1:]:
        found: List[Tuple[int, int]] = []
        for left in segments:
            for right in item.columns:
                lo, hi = max(left[0], right[0]), min(left[1], right[1])
                if hi >= lo:
                    found.append((lo, hi))
        segments = sorted(found)
    if rows is None:
        segments = []
    return rows, tuple(segments)


def project_occurrence(constellation: FrameConstellation, *, geography: Geography,
                       pattern_id: int = -1) -> OccurrenceProjection:
    """Put one frame constellation on the map, member by member."""
    if not isinstance(constellation, FrameConstellation):
        raise InvalidParameterError(
            "constellation", type(constellation).__name__,
            "a T4E.1 FrameConstellation. A projection reads the record's own positions rather "
            "than a signature's, and a signature does not carry them")
    if not constellation.features:
        raise InvalidParameterError(
            "constellation.features", 0,
            "a constellation carrying the features its nodes were built from. Without them "
            "nothing can say which scale's filter support a node's footprint has, nor whether "
            "its position was ever registered to the parent grid (D88)")
    footprints = tuple(_member_footprint(node, feature, geography)
                       for node, feature in zip(constellation.nodes, constellation.features))
    rows, columns = common_extent(footprints)
    return OccurrenceProjection(
        pattern_id=int(pattern_id), key=tuple(constellation.key()),
        moment=footprints[0].moment, footprints=footprints,
        common_rows=rows, common_columns=columns, vertical=geography.vertical())


class EvidenceIndex:
    """A catalogue, the constellations behind it and the map they are all on, bound once.

    The binding is the point. A pattern's members are `ConstellationSignature`s, and a
    signature deliberately carries no positions -- it is the comparable half, and comparability
    is what dropping the positions buys. Recovering them means going back to the constellation
    the signature was signed from, by its key, and doing that by hand at a call site is how a
    member ends up projected through another member's coefficient.
    """

    def __init__(self, catalogue: PatternCatalogue, constellations: ConstellationSet,
                 geography: Geography) -> None:
        if not isinstance(catalogue, PatternCatalogue):
            raise InvalidParameterError(
                "catalogue", type(catalogue).__name__, "a T4E.3 PatternCatalogue")
        if not isinstance(constellations, ConstellationSet):
            raise InvalidParameterError(
                "constellations", type(constellations).__name__,
                "the T4E.1 ConstellationSet these patterns were clustered from. The catalogue "
                "carries signatures, and a signature carries no position by design")
        if not isinstance(geography, Geography):
            raise InvalidParameterError(
                "geography", type(geography).__name__,
                "a Geography from geography_of, so every footprint of this record is drawn on "
                "one map")
        self.catalogue = catalogue
        self.constellations = constellations
        self.geography = geography
        self._by_key: Dict[Tuple[Any, ...], FrameConstellation] = {
            tuple(item.key()): item for item in constellations}
        self._by_pattern: Dict[int, ConstellationPattern] = {
            int(pattern.pattern_id): pattern for pattern in catalogue}

        missing = [tuple(member.key) for pattern in catalogue for member in pattern.members
                   if tuple(member.key) not in self._by_key]
        if missing:
            raise InvalidParameterError(
                "constellations", len(missing),
                "the extraction these patterns were signed from. %d of the catalogue's members "
                "have no constellation in this set, and a projection that skipped them would "
                "show fewer occurrences than the pattern was clustered on -- understating the "
                "evidence while looking complete. First missing key: %s"
                % (len(missing), missing[0]))

    def pattern(self, pattern_id: int) -> ConstellationPattern:
        try:
            return self._by_pattern[int(pattern_id)]
        except KeyError:
            raise InvalidParameterError(
                "pattern_id", pattern_id,
                "a pattern this catalogue contains (%s)" % sorted(self._by_pattern)) from None

    def occurrences_at(self, pattern_id: int, time: float) -> Tuple[OccurrenceProjection, ...]:
        """Every occurrence of this pattern in one frame, projected. Usually one; never zero
        for a time the event series reported, which is what makes an empty answer a defect."""
        pattern = self.pattern(pattern_id)
        found = [member for member in pattern.members if float(member.time) == float(time)]
        return tuple(project_occurrence(self._by_key[tuple(member.key)],
                                        geography=self.geography, pattern_id=int(pattern_id))
                     for member in found)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": PROJECTION_SCHEMA,
            "n_patterns": len(self._by_pattern),
            "n_constellations": len(self._by_key),
            "geography": self.geography.describe(),
        }


def evidence_index(catalogue: PatternCatalogue, constellations: ConstellationSet,
                   geography: Geography) -> EvidenceIndex:
    """Bind a catalogue to the constellations behind it and the map they are on."""
    return EvidenceIndex(catalogue, constellations, geography)


def _shares(members: Sequence[Any], index: EvidenceIndex,
            attribute: str) -> Tuple[Mapping[str, Any], ...]:
    """Each scale's or band's share of this pattern's own summed squared magnitude."""
    energy: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for member in members:
        constellation = index._by_key[tuple(member.key)]
        for node in constellation.nodes:
            key = (str(node.band) if attribute == "band"
                   else str(node.band).split("/")[0])
            energy[key] = energy.get(key, 0.0) + float(node.strength) ** 2
            counts[key] = counts.get(key, 0) + 1
    total = sum(energy.values())
    return tuple({
        "name": key,
        "energy": energy[key],
        "share_of_energy": (energy[key] / total) if total > 0 else None,
        "n_nodes": counts[key],
        "share_of_nodes": counts[key] / float(sum(counts.values())),
    } for key in sorted(energy, key=lambda name: (-energy[name], name)))


def project_pattern(pattern_id: int, index: EvidenceIndex) -> PatternProjection:
    """Every occurrence of one clustered pattern, on the map it was measured on."""
    if not isinstance(index, EvidenceIndex):
        raise InvalidParameterError(
            "index", type(index).__name__,
            "an EvidenceIndex binding the catalogue, its constellations and one map")
    pattern = index.pattern(pattern_id)
    occurrences = tuple(
        project_occurrence(index._by_key[tuple(member.key)], geography=index.geography,
                           pattern_id=int(pattern_id))
        for member in sorted(pattern.members, key=lambda item: (float(item.time),
                                                                tuple(item.key))))
    return PatternProjection(
        pattern_id=int(pattern_id), occurrences=occurrences,
        scale_shares=_shares(pattern.members, index, "scale"),
        band_shares=_shares(pattern.members, index, "band"),
        n_members=len(pattern.members))


# ------------------------------------------------------------------------------ rules


def project_rule(rule: PrecursorRule, series: EventSeries,
                 index: Optional[EvidenceIndex] = None) -> RuleProjection:
    """List the occurrences one rule was counted over, reconciled against its own figures.

    `index` is optional: without it the instances carry their times and verdicts but no
    footprints, which is the right answer when the caller has the event record and not the
    decomposition. With it, both ends of every instance are projected.
    """
    if not isinstance(rule, PrecursorRule):
        raise InvalidParameterError(
            "rule", type(rule).__name__,
            "a T4F.3 PrecursorRule at one declared lag. A selected-lag row is a pair's "
            "strongest lag rather than a count, and the instances belong to the lag that won")
    if not isinstance(series, EventSeries):
        raise InvalidParameterError(
            "series", type(series).__name__,
            "the T4F.1 EventSeries the rule was measured on")
    if index is not None and not isinstance(index, EvidenceIndex):
        raise InvalidParameterError(
            "index", type(index).__name__, "an EvidenceIndex, or nothing at all")

    pair = (int(rule.antecedent), int(rule.consequent))
    times = times_by_pattern(series)
    verdicts = anchor_verdicts(pair, series, times, rule.window)

    instances: List[RuleInstance] = []
    tally = {INSTANCE_SUPPORTING: 0, INSTANCE_UNFOLLOWED: 0,
             INSTANCE_TRUNCATED: 0, INSTANCE_UNOBSERVED: 0}
    for anchor, status, completed in verdicts:
        verdict = _VERDICT_OF_WINDOW[(status, completed)]
        tally[verdict] += 1
        found = tuple(window_completions(times, pair[1], anchor, rule.window))
        antecedents: Tuple[OccurrenceProjection, ...] = ()
        consequents: Tuple[OccurrenceProjection, ...] = ()
        if index is not None:
            antecedents = index.occurrences_at(pair[0], anchor)
            consequents = tuple(item for time in found
                                for item in index.occurrences_at(pair[1], time))
        instances.append(RuleInstance(
            antecedent=pair[0], consequent=pair[1], anchor=float(anchor), verdict=verdict,
            completions=found, lags=tuple(float(time) - float(anchor) for time in found),
            antecedents=antecedents, consequents=consequents))

    enumerated = {
        "antecedent_occurrences": len(instances),
        "eligible_antecedents": tally[INSTANCE_SUPPORTING] + tally[INSTANCE_UNFOLLOWED],
        "support": tally[INSTANCE_SUPPORTING],
        "ineligible_truncated": tally[INSTANCE_TRUNCATED],
        "ineligible_unobserved": tally[INSTANCE_UNOBSERVED],
    }
    published = {
        "antecedent_occurrences": int(rule.antecedent_occurrences),
        "eligible_antecedents": int(rule.eligible_antecedents),
        "support": int(rule.support),
        "ineligible_truncated": int(rule.ineligible_truncated),
        "ineligible_unobserved": int(rule.ineligible_unobserved),
    }
    disagreed = sorted(name for name in published if published[name] != enumerated[name])
    if disagreed:
        raise InvalidParameterError(
            "series", {name: (published[name], enumerated[name]) for name in disagreed},
            "the event series this rule was measured on. Enumerating its anchors here gives a "
            "different %s than the rule published, which means these occurrences are not the "
            "ones the correction was paid on. Projecting them beside that q-value would put a "
            "second study's evidence under the first study's inference"
            % " and a different ".join(disagreed))

    return RuleProjection(
        antecedent=pair[0], consequent=pair[1], window=rule.window, status=rule.status,
        figures={
            "support": rule.support,
            "eligible_antecedents": rule.eligible_antecedents,
            "antecedent_occurrences": rule.antecedent_occurrences,
            "confidence": rule.confidence,
            "base_rate": rule.base_rate,
            "lift": rule.lift,
            "p_value": rule.p_value,
            "q_value": rule.q_value,
            "n_surrogates": rule.n_surrogates,
        },
        instances=tuple(instances),
        reconciliation={
            "published": published, "enumerated": enumerated, "agree": True,
            "decided_by": ("spectral_sequences.anchor_verdicts, the same per-anchor decision "
                           "count_sequence tallies, so the instances shown and the support "
                           "corrected cannot come from two different rules"),
        })


def describe_projection(projection: Any) -> Dict[str, Any]:
    """A receipt for whichever projection was handed in, with its claim boundary attached."""
    if isinstance(projection, (PatternProjection, RuleProjection, OccurrenceProjection,
                               EvidenceIndex, Geography)):
        return projection.describe()
    raise InvalidParameterError(
        "projection", type(projection).__name__,
        "a projection this module produced")
