"""Known-phenomenon cross-reference, and the physical gate it exists to run (`T4F.6`).

T4F.3 asked whether a rule beat a null, T4F.4 made the corrected table askable from either end,
and T4F.5 put a rule back on the map it was measured on. None of them asks the question R10
asks, which is not statistical at all: **do the things this pipeline finds look like the things
that are already known to be there?** If the answer is no, the prior is that the instrument is
broken rather than that the atmosphere is surprising, and Phase 4G does not start.

That makes this module a gate and not a feature, and a gate is only worth as much as its
capacity to fail. Five things here exist to protect that capacity.

**1. A catalogue written after the answer is not a gate.** Every entry carries a citation and
the catalogue carries a digest over its own canonical content, so a gate receipt names the
catalogue it was adjudicated under and an entry edited afterwards changes the digest. A
catalogue is `draft` until a maintainer freezes it, and `physical_gate` refuses to adjudicate
under a draft: choosing which phenomena count as recognisable, and in what envelopes, is a
scientific declaration and this module will not make it on anybody's behalf.

**2. `recognised` means consistent with a declared envelope. It never means "is".** A pattern
whose measured scale and geometry fall inside an entry's declared ranges is consistent with that
entry, and so, quite possibly, are several others; every consistent entry is listed rather than
the best one chosen, because picking the closest would turn an overlap into an identification.
The catalogue's own overlapping pairs are published beside the labels for the same reason.

**3. The record's geometry decides how much the cross-reference can discriminate, and it is
measured before anything is adjudicated.** A scale in cells becomes a scale in kilometres
through the grid's own metric, and on a lat/lon crop that metric is not one number: on this
programme's own ERA5 record, latitudes -60 to -20 at 0.25 degrees, the meridional spacing is
27.80 km everywhere and the zonal spacing runs from 13.90 km to 26.12 km, a factor of 1.879 and
a 61.1% variation that `GridSpec.anisotropy` already warns about. So a footprint of *n* cells is
between `13.90n` and `27.80n` km wide, an interval spanning a factor of two before any
measurement error at all, and an envelope narrower than that cannot exclude anything. The gate
therefore counts how many exclusions the catalogue actually made on this record, and a catalogue
that excluded nothing returns `INVALID` rather than `PASS`. Recognition by imprecision is the
failure mode this task invites, and it is refused by measurement rather than by hope.

**4. Three outcomes, not two, because "we could not check" is not "we checked and it was not
there".** An entry needing an observable the record does not carry -- a 500 hPa field on a
record of 850 hPa temperature, say -- is `unassessable` for every pattern, and a pattern nothing
could be assessed against is `unassessable` rather than `unrecognised`. Labelling it
`unrecognised` would record a limit of the instrument as a fact about the atmosphere, which is
exactly the FAIL/INVALID boundary T4C.5i drew for the statistical gate, arriving here.

**5. What the gate licenses.** `PASS` means a pattern consistent with a declared known
phenomenon appeared in the top results of a ranking declared in advance, and it licenses the
start of Phase 4G and nothing else. `FAIL` means the cross-reference was capable of recognising
something and did not, and the standing conclusion is then that the pipeline is broken. `INVALID`
means the question could not be asked on this record, licenses nothing, and is not a `FAIL` --
an under-powered gate recorded as a negative finding is how an instrument's blindness becomes a
result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.spectral_precursors import PrecursorReport, PrecursorRule
from src.analysis_engine.spectral_projection import Geography, PatternProjection
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.errors import InvalidParameterError

REFERENCE_SCHEMA = "spectral-phenomenon-reference/v1"
DECLARATION_SCHEMA = "spectral-physical-gate-declaration/v1"
GATE_SCHEMA = "spectral-physical-gate/v1"

#: How one declared criterion came out on one pattern. `UNASSESSABLE` is the load-bearing one:
#: it is the difference between a record that disagrees with an envelope and a record that
#: cannot carry the quantity the envelope is about.
CRITERION_INSIDE = "INSIDE_THE_DECLARED_ENVELOPE"
CRITERION_OUTSIDE = "OUTSIDE_THE_DECLARED_ENVELOPE"
CRITERION_UNASSESSABLE = "NOT_MEASURABLE_ON_THIS_RECORD"

#: How one pattern came out against one catalogue entry.
MATCH_CONSISTENT = "CONSISTENT_WITH_THE_DECLARED_ENVELOPE"
MATCH_EXCLUDED = "EXCLUDED_BY_A_DECLARED_CRITERION"
MATCH_UNASSESSABLE = "NOT_ASSESSABLE_ON_THIS_RECORD"

#: How one pattern came out against the whole catalogue. R10's two labels, and the third that
#: keeps them honest.
LABEL_RECOGNISED = "recognised"
LABEL_UNRECOGNISED = "unrecognised"
LABEL_UNASSESSABLE = "unassessable"

#: The gate's three verdicts. `INVALID` licenses nothing and is not a `FAIL`.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INVALID = "INVALID"

#: A catalogue is a scientific declaration, and this module will not sign one.
CATALOGUE_DRAFT = "draft"
CATALOGUE_FROZEN = "frozen"

#: Which of a coefficient's two sizes an entry's horizontal envelope is declared against.
#: T4F.5 measured these as different numbers about different things -- at haar level 4 the
#: filter support is 16 cells and the dyadic octave label is 8 -- so a catalogue declared in one
#: convention and read in the other would be wrong by exactly a factor of two, silently.
SCALE_FILTER_SUPPORT = "filter_support_cells"
SCALE_OCTAVE_LABEL = "dyadic_octave_cells"
SCALE_CONVENTIONS = (SCALE_FILTER_SUPPORT, SCALE_OCTAVE_LABEL)

#: A pattern consistent with an entry on one criterion alone has not been recognised; it has
#: failed to be excluded. Two independently measured criteria is the floor for a label that
#: gates a phase, and the number is here rather than inline so a reader can argue with it.
MINIMUM_ASSESSED_CRITERIA = 2

#: The spellings of "metre" a physical grid may declare. Both shipped physical geometries
#: declare `m`, so nothing else can reach the refusal below today; the set exists so that a
#: geometry registered later declaring feet or nautical miles is refused rather than divided by
#: a thousand and called kilometres. Only metres are listed, and a grid declaring kilometres
#: would be refused too, because a conversion nobody has exercised is not one to ship.
_METRE_SPELLINGS = frozenset(("m", "metres", "meters"))

#: Frames whose spacing differs by more than this fraction of the median are not one cadence,
#: and a record without one cadence cannot turn a lag in frames into a lag in hours.
CADENCE_TOLERANCE = 1e-6

#: The sentence that stops `recognised` being read as a detection.
RECOGNITION_NOTE = (
    "'recognised' means this pattern's measured scale, geometry and lead all fall inside the "
    "ranges a maintainer declared for a known phenomenon, on a record whose grid could express "
    "them. It is a statement of consistency and not of identity: many structures are consistent "
    "with a range, the ranges of different phenomena overlap, and the entries this pattern is "
    "consistent with are listed together rather than resolved into one. It is not a detection "
    "of the named phenomenon, and 'unrecognised' is not a discovery.")

#: The sentence that stops the gate being read as a finding.
GATE_CLAIM_BOUNDARY = (
    "This gate asks one question: did a pattern consistent with a declared known phenomenon "
    "appear in the top results of a ranking declared before it was run. PASS licenses the start "
    "of Phase 4G and claims nothing about the atmosphere -- not a cause, a driver, a mechanism, "
    "a trigger, a forecast or an intervention, and not that any named phenomenon was detected. "
    "FAIL is a statement about this pipeline, whose standing interpretation is that it is broken "
    "rather than that physics is. INVALID means the question could not be asked on this record "
    "and licenses nothing at all; it must never be read or reported as a FAIL.")


# ------------------------------------------------------------------------------ declared ranges


@dataclass(frozen=True)
class Envelope:
    """An inclusive range a maintainer declared, with the unit it is a range of.

    Inclusive at both ends because these are read off the literature as "roughly 1000 to 2000
    km", and an exclusive boundary would make a pattern sitting exactly on a round number that
    somebody typed fall outside it.
    """

    low: float
    high: float
    units: str

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("high", self.high)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise InvalidParameterError(
                    "Envelope.%s" % name, value,
                    "a finite bound. An envelope with an infinite end excludes nothing and "
                    "would make every pattern consistent with it")
        if float(self.low) > float(self.high):
            raise InvalidParameterError(
                "Envelope.low", self.low,
                "a low bound at or below the high bound (%s). A reversed range is empty, and an "
                "empty range excludes every pattern including the one it was written for"
                % self.high)
        if not isinstance(self.units, str) or not self.units.strip():
            raise InvalidParameterError(
                "Envelope.units", self.units,
                "a non-empty unit. A range of numbers is not a range of kilometres")

    def overlaps(self, low: float, high: float) -> bool:
        """Whether a measured interval meets this envelope anywhere."""
        return float(high) >= float(self.low) and float(low) <= float(self.high)

    def fraction_of(self, low: float, high: float) -> float:
        """How much of a measured interval lies inside this envelope, 0.0 to 1.0.

        A bare overlap and a containment both count as `inside`, and this is what tells them
        apart on the receipt. A degenerate measurement -- an isotropic grid gives one number,
        not a range -- is either wholly in or wholly out.
        """
        low, high = float(low), float(high)
        inner = min(high, float(self.high)) - max(low, float(self.low))
        if high <= low:
            return 1.0 if self.overlaps(low, high) else 0.0
        return max(0.0, inner) / (high - low)

    def as_record(self) -> Dict[str, Any]:
        return {"low": float(self.low), "high": float(self.high), "units": self.units}


@dataclass(frozen=True)
class Phenomenon:
    """One known phenomenon, as a maintainer declared it, with the source they declared it from.

    `observables` names what a record must carry for this entry to be checkable at all. It is
    matched against what the record says it holds, and an entry needing something absent is
    `unassessable` -- never `unrecognised`, which would blame the atmosphere for the crop.
    """

    name: str
    citation: str
    observables: Tuple[str, ...]
    horizontal_scale: Envelope
    lead: Envelope
    cardinality: Optional[int] = None
    separation: Optional[Envelope] = None
    scale_convention: str = SCALE_FILTER_SUPPORT
    note: str = ""

    def __post_init__(self) -> None:
        for name, value in (("name", self.name), ("citation", self.citation)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "Phenomenon.%s" % name, value,
                    "a non-empty %s. An entry with no source is an assertion about the "
                    "atmosphere with nothing behind it, and a catalogue of those cannot gate "
                    "anything: it would recognise whatever it was written to recognise" % name)
        if not self.observables:
            raise InvalidParameterError(
                "Phenomenon.observables", (),
                "at least one observable this entry needs. An entry that names nothing is "
                "checkable against every record, including records that cannot see it")
        if self.horizontal_scale.units != "km":
            raise InvalidParameterError(
                "Phenomenon.horizontal_scale.units", self.horizontal_scale.units,
                "km. The bridge from cells to a physical length produces kilometres, and a "
                "comparison across two length units nobody converted is arithmetic on names")
        if self.lead.units != "hours":
            raise InvalidParameterError(
                "Phenomenon.lead.units", self.lead.units,
                "hours. A lead is compared against a transition window converted through the "
                "record's own cadence, and that conversion produces hours")
        if float(self.lead.low) < 0.0:
            raise InvalidParameterError(
                "Phenomenon.lead.low", self.lead.low,
                "a non-negative lead. T4F.2's transition window has a strictly positive minimum "
                "lag so that simultaneity can never become a step, and a negative declared lead "
                "would be asking this gate to recognise the consequent preceding its antecedent")
        if self.separation is not None and self.separation.units != "km":
            raise InvalidParameterError(
                "Phenomenon.separation.units", self.separation.units, "km")
        if self.cardinality is not None and int(self.cardinality) < 2:
            raise InvalidParameterError(
                "Phenomenon.cardinality", self.cardinality,
                "at least two members, or none at all. A constellation of one has no geometry, "
                "and T4E.1 never builds one")
        if self.scale_convention not in SCALE_CONVENTIONS:
            raise InvalidParameterError(
                "Phenomenon.scale_convention", self.scale_convention,
                "one of %s. A haar level's filter support and its dyadic octave label differ by "
                "a factor of two, so an envelope declared against one and read against the "
                "other is wrong by that factor and looks entirely reasonable"
                % list(SCALE_CONVENTIONS))

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": self.name,
            "citation": self.citation,
            "observables": list(self.observables),
            "horizontal_scale": self.horizontal_scale.as_record(),
            "lead": self.lead.as_record(),
            "scale_convention": self.scale_convention,
        }
        if self.cardinality is not None:
            record["cardinality"] = int(self.cardinality)
        if self.separation is not None:
            record["separation"] = self.separation.as_record()
        if self.note:
            record["note"] = self.note
        return record


def _digest(payload: Any) -> str:
    """A sha256 over the canonical JSON of a declaration, so an edit is visible."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReferenceCatalogue:
    """The declared reference set, its provenance, and whether a maintainer has signed it."""

    phenomena: Tuple[Phenomenon, ...]
    declared_by: str
    declared_on: str
    status: str = CATALOGUE_DRAFT
    note: str = ""
    #: Who signed this catalogue off, and deliberately outside `digest`: see that property.
    signed_by: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.phenomena:
            raise InvalidParameterError(
                "ReferenceCatalogue.phenomena", 0,
                "at least one declared phenomenon. An empty catalogue labels every pattern "
                "unassessable and would let a gate that can recognise nothing return anything "
                "other than INVALID")
        names = [item.name for item in self.phenomena]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise InvalidParameterError(
                "ReferenceCatalogue.phenomena", duplicated,
                "distinct entry names. Two entries under one name make a receipt that says a "
                "pattern is consistent with 'cyclogenesis' unreadable, because it cannot be "
                "traced back to which envelope admitted it")
        for name, value in (("declared_by", self.declared_by),
                            ("declared_on", self.declared_on)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "ReferenceCatalogue.%s" % name, value,
                    "a non-empty %s. A gate receipt names who declared the catalogue it was "
                    "adjudicated under and when" % name)
        if self.status not in (CATALOGUE_DRAFT, CATALOGUE_FROZEN):
            raise InvalidParameterError(
                "ReferenceCatalogue.status", self.status,
                "either %r or %r" % (CATALOGUE_DRAFT, CATALOGUE_FROZEN))

    def __iter__(self):
        return iter(self.phenomena)

    def __len__(self) -> int:
        return len(self.phenomena)

    @property
    def digest(self) -> str:
        """A hash of the entries and their provenance, but not of the status or the signature.

        Freezing must not change the digest, and this is checked: the point of the hash is that
        the *envelopes* a gate was adjudicated under are the ones that were declared, and a
        maintainer's signature on an unchanged catalogue does not change what it says. If
        signing moved the digest, a receipt could not be traced back to the draft that was
        reviewed, which is the one thing the digest exists to make possible.
        """
        return _digest({
            "schema": REFERENCE_SCHEMA,
            "phenomena": [item.as_record() for item in self.phenomena],
            "declared_by": self.declared_by,
            "declared_on": self.declared_on,
            "note": self.note,
        })

    def freeze(self, *, signed_by: str) -> "ReferenceCatalogue":
        """A maintainer's signature on this catalogue, as a new object.

        Deliberately explicit, and deliberately not something the gate can do for itself: which
        phenomena count as recognisable, and in what ranges, is the scientific content of the
        whole task.
        """
        if not isinstance(signed_by, str) or not signed_by.strip():
            raise InvalidParameterError(
                "signed_by", signed_by,
                "the name of whoever is signing. A frozen catalogue with no signatory is a "
                "draft that has stopped saying it is one")
        # The signature goes in its own field rather than into `note`, which is hashed. A
        # maintainer signing an unchanged catalogue must not change what the digest says was
        # declared, or a gate receipt could not be traced back to the draft that was reviewed.
        return ReferenceCatalogue(
            phenomena=self.phenomena, declared_by=self.declared_by,
            declared_on=self.declared_on, status=CATALOGUE_FROZEN, note=self.note,
            signed_by=signed_by.strip())

    def overlapping_pairs(self) -> Tuple[Dict[str, Any], ...]:
        """Entry pairs whose declared envelopes cannot separate a pattern, published as found.

        Two entries overlapping is not an error; it is what the literature looks like. It is
        reported because a pattern consistent with both of them is *not* evidence for either
        one, and a reader who cannot see the overlap will read it as evidence for the first.
        """
        pairs: List[Dict[str, Any]] = []
        for i, left in enumerate(self.phenomena):
            for right in self.phenomena[i + 1:]:
                scale = left.horizontal_scale.overlaps(
                    right.horizontal_scale.low, right.horizontal_scale.high)
                lead = left.lead.overlaps(right.lead.low, right.lead.high)
                if scale and lead:
                    pairs.append({
                        "entries": [left.name, right.name],
                        "overlapping": ["horizontal_scale", "lead"],
                        "note": ("a pattern inside both envelopes is consistent with both "
                                 "entries and is evidence for neither over the other"),
                    })
        return tuple(pairs)


#: The observable the programme's own ERA5 record carries, spelled as `PhysicalBridge` spells
#: it. One variable at one level: whatever else this crop is good for, it cannot show a
#: structure at one level above a structure at another.
SOUTHERN_OCEAN_OBSERVABLE = "t@850 pressure_hpa"


def draft_southern_ocean_reference() -> ReferenceCatalogue:
    """A first reference set for the T4C.5k record, as a **draft** nobody has signed.

    The record it is written for is 850 hPa temperature, 0.25 degrees, six-hourly, latitudes
    -60 to -20 and longitudes 140 to 180: the Tasman Sea and the Southern Ocean south of it,
    which is one of the better-studied cyclogenesis regions there is.

    **Every envelope here is the implementer's reading and not a number quoted from a paper.**
    The citations name where each phenomenon's standard description comes from; the ranges are a
    proposal for a maintainer to correct. That is why the catalogue returns `draft` and why
    `physical_gate` refuses to adjudicate under it: an envelope is the scientific content of
    this gate, and a gate whose envelopes were chosen by whoever wrote the code that runs it is
    a gate that recognises what it was built to recognise.

    The fourth entry is included precisely because this record cannot check it. Classical
    cyclogenesis is described as an upper-level disturbance overtaking a low-level baroclinic
    zone, and a single-level record of temperature has no upper level in it. That entry will
    come back `unassessable` on this crop, for every pattern, forever -- which is the true
    answer and is not the same answer as `unrecognised`.
    """
    return ReferenceCatalogue(
        declared_by="implementer's draft, T4F.6",
        declared_on="2026-09-06",
        status=CATALOGUE_DRAFT,
        note=("Envelopes are a first reading proposed from the cited literature's standard "
              "descriptions of each phenomenon; none of them is a value quoted from a paper. "
              "They are to be reviewed, corrected and frozen by the maintainer before any gate "
              "is adjudicated under them, and the digest of the reviewed catalogue is what a "
              "gate receipt should name."),
        phenomena=(
            Phenomenon(
                name="extratropical-cyclone-thermal-couplet",
                citation=("Sinclair (1995), 'A climatology of cyclogenesis for the Southern "
                          "Hemisphere', Mon. Wea. Rev. 123, 1601-1619; Sanders and Gyakum "
                          "(1980), 'Synoptic-dynamic climatology of the bomb', Mon. Wea. Rev. "
                          "108, 1589-1606"),
                observables=(SOUTHERN_OCEAN_OBSERVABLE,),
                horizontal_scale=Envelope(500.0, 2000.0, "km"),
                lead=Envelope(6.0, 48.0, "hours"),
                cardinality=2,
                separation=Envelope(300.0, 1500.0, "km"),
                note=("the warm and cold thermal anomalies either side of a developing surface "
                      "low's frontal zone, which is the part of cyclogenesis an 850 hPa "
                      "temperature field can carry at all")),
            Phenomenon(
                name="frontal-wave",
                citation=("Sinclair (1995), Mon. Wea. Rev. 123, 1601-1619, for the Southern "
                          "Hemisphere frequency and scale of these features"),
                observables=(SOUTHERN_OCEAN_OBSERVABLE,),
                horizontal_scale=Envelope(200.0, 800.0, "km"),
                lead=Envelope(6.0, 24.0, "hours"),
                cardinality=2,
                separation=Envelope(100.0, 600.0, "km"),
                note="a smaller and faster relative of the entry above, and it overlaps it"),
            Phenomenon(
                name="blocking-onset",
                citation=("Rex (1950), 'Blocking action in the middle troposphere and its "
                          "effect upon regional climate', Tellus 2, 196-211"),
                observables=(SOUTHERN_OCEAN_OBSERVABLE,),
                horizontal_scale=Envelope(2000.0, 6000.0, "km"),
                lead=Envelope(48.0, 168.0, "hours"),
                cardinality=2,
                note=("a planetary-scale entry on a 40-degree crop: much of this envelope is "
                      "wider than the domain, so it is here as much to be excluded as to be "
                      "matched")),
            Phenomenon(
                name="upper-level-pv-precursor",
                citation=("Hoskins, McIntyre and Robertson (1985), 'On the use and "
                          "significance of isentropic potential vorticity maps', Q. J. R. "
                          "Meteorol. Soc. 111, 877-946"),
                observables=("pv@315 potential_temperature_k", "z@500 pressure_hpa"),
                horizontal_scale=Envelope(800.0, 3000.0, "km"),
                lead=Envelope(12.0, 72.0, "hours"),
                cardinality=2,
                note=("the entry this record cannot check. It needs a field at a second level, "
                      "and a single-level record has none, so it is unassessable here rather "
                      "than absent")),
        ))


# ------------------------------------------------------------------------------ the record's own

@dataclass(frozen=True)
class Measurement:
    """One quantity read off the record, as the interval the record can actually support."""

    name: str
    low: Optional[float]
    high: Optional[float]
    units: Optional[str]
    basis: str
    refusal: Optional[str] = None

    @property
    def measured(self) -> bool:
        return self.refusal is None and self.low is not None and self.high is not None

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {"name": self.name, "basis": self.basis}
        if self.measured:
            record.update({"low": float(self.low), "high": float(self.high),
                           "units": self.units})
        if self.refusal:
            record["refusal"] = self.refusal
        return record


class PhysicalBridge:
    """Cells to kilometres and frames to hours for one record, or a named refusal per axis.

    Built by `physical_bridge`. Both conversions are intervals rather than numbers, and for
    different reasons. Horizontally, a lat/lon crop's zonal spacing varies with latitude and its
    cells are anisotropic, so `n` cells is a genuine range of lengths and not an uncertain one.
    Temporally, a record either carries a calendar and therefore a cadence, or does not and
    therefore has no hours at all.
    """

    def __init__(self, *, grid: Any, variable: str, level: Optional[float],
                 level_axis: Optional[str], cell_km: Tuple[Optional[float], Optional[float]],
                 cell_refusal: Optional[str], cadence_hours: Optional[float],
                 cadence_refusal: Optional[str], anisotropy: Mapping[str, Any],
                 record_times: Sequence[float]) -> None:
        self.grid = grid
        #: The frames of the record these coefficients were taken from. A lag is converted
        #: through this record's cadence, so a series counted on some other clock must be
        #: refused rather than converted; `lead_for` is where that is decided.
        self.record_times = tuple(float(item) for item in record_times)
        self.variable = variable
        self.level = level
        self.level_axis = level_axis
        self.cell_km_low, self.cell_km_high = cell_km
        self.cell_refusal = cell_refusal
        self.cadence_hours = cadence_hours
        self.cadence_refusal = cadence_refusal
        self.anisotropy = dict(anisotropy)

    # ------------------------------------------------------------------ what the record holds

    @property
    def observable(self) -> str:
        """The one observable this record carries, spelled as an entry must name it.

        A level with no declared axis is a number and is spelled as one (TG1.5), so an entry
        asking for 850 hPa does not match a record that merely has `850` set on it.
        """
        if self.level is None:
            return "%s@no-declared-level" % self.variable
        if self.level_axis is None:
            return "%s@level-%g-with-no-declared-axis" % (self.variable, float(self.level))
        return "%s@%g %s" % (self.variable, float(self.level), self.level_axis)

    @property
    def has_length(self) -> bool:
        return self.cell_refusal is None

    @property
    def has_clock(self) -> bool:
        return self.cadence_refusal is None

    # ------------------------------------------------------------------ the two conversions

    def length(self, name: str, cells_low: float, cells_high: float) -> Measurement:
        """A span in cells as the span in kilometres this grid can support."""
        if not self.has_length:
            return Measurement(name=name, low=None, high=None, units=None,
                               basis="this grid declares no length", refusal=self.cell_refusal)
        return Measurement(
            name=name, low=float(cells_low) * self.cell_km_low,
            high=float(cells_high) * self.cell_km_high, units="km",
            basis=("%g to %g cells through this grid's own metric, whose cell side runs %.2f to "
                   "%.2f km across the crop (%s)"
                   % (float(cells_low), float(cells_high), self.cell_km_low, self.cell_km_high,
                      self.grid.kind)))

    def lead(self, window: TransitionWindow) -> Measurement:
        """A transition window in frames as the lead in hours this record can support."""
        if not self.has_clock:
            return Measurement(name="lead", low=None, high=None, units=None,
                               basis="this record carries no calendar",
                               refusal=self.cadence_refusal)
        return Measurement(
            name="lead", low=float(window.minimum_lag) * self.cadence_hours,
            high=float(window.maximum_lag) * self.cadence_hours, units="hours",
            basis=("the declared window of %g to %g %s at this record's own measured cadence of "
                   "%g hours per frame"
                   % (float(window.minimum_lag), float(window.maximum_lag), window.time_units,
                      self.cadence_hours)))

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "grid": self.grid.describe() if hasattr(self.grid, "describe") else str(self.grid),
            "observable": self.observable,
            "anisotropy": {key: value for key, value in self.anisotropy.items()
                           if key != "warnings"},
            "grid_warnings": list(self.anisotropy.get("warnings", ())),
        }
        if self.has_length:
            record["cell_km"] = {"low": self.cell_km_low, "high": self.cell_km_high,
                                 "ratio": self.cell_km_high / self.cell_km_low}
        else:
            record["cell_km_refusal"] = self.cell_refusal
        if self.has_clock:
            record["cadence_hours"] = self.cadence_hours
        else:
            record["cadence_refusal"] = self.cadence_refusal
        return record


def physical_bridge(geography: Geography, *, variable: str) -> PhysicalBridge:
    """Measure this record's cells-to-kilometres and frames-to-hours, or refuse each by name.

    `variable` is what the tracking pass declared it was tracking; it is asked for rather than
    guessed because a decomposition carries filters and a clock and does not carry the name of
    the field it was taken from.
    """
    if not isinstance(geography, Geography):
        raise InvalidParameterError(
            "geography", type(geography).__name__,
            "a T4F.5 Geography. The bridge needs the grid, the clock and the record bound "
            "together and already checked against each other")
    if not isinstance(variable, str) or not variable.strip():
        raise InvalidParameterError(
            "variable", variable,
            "the name of the field these coefficients were taken from. An entry declares which "
            "observable it needs, and a record that cannot say what it holds matches everything")

    grid = geography.grid
    cell_low: Optional[float] = None
    cell_high: Optional[float] = None
    cell_refusal: Optional[str] = None
    if grid.capability("has_latitude", False):
        dx = np.asarray(grid.dx_metres().detach().cpu().numpy(), dtype=np.float64) / 1000.0
        dy = np.asarray(grid.dy_metres().detach().cpu().numpy(), dtype=np.float64) / 1000.0
        cell_low = float(min(dx.min(), dy.min()))
        cell_high = float(max(dx.max(), dy.max()))
    elif grid.is_physical:
        # Both physical geometries this programme ships declare `m`, so the refusal here is
        # unreachable today and no test can reach it without registering a geometry. It is a
        # refusal rather than a fallback because the only fallback available is "assume the
        # number is already in the unit we want", and a geometry registered later that declared
        # feet would then be silently converted as though it were metres -- the exact
        # assumption this module refuses everywhere else.
        unit = str(grid.length_units).strip()
        if unit not in _METRE_SPELLINGS:
            raise InvalidParameterError(
                "geography.grid.length_units", unit,
                "a grid whose length is declared in metres (%s). An envelope is declared in "
                "kilometres, and converting a length in an unexercised unit would be "
                "arithmetic across a conversion nobody performed"
                % sorted(_METRE_SPELLINGS))
        cell_low = float(min(abs(grid.dy), abs(grid.dx))) / 1000.0
        cell_high = float(max(abs(grid.dy), abs(grid.dx))) / 1000.0
    else:
        cell_refusal = (
            "a %s grid declares no length, so a footprint of n cells has no size in kilometres "
            "and no declared envelope in kilometres can be compared with it. Assuming a spacing "
            "here would let the gate recognise a phenomenon by a number nobody measured"
            % grid.kind)
    if cell_low is not None and cell_low <= 0.0:
        cell_low, cell_high = None, None
        cell_refusal = ("this grid's smallest cell side is not positive, so cells and kilometres "
                        "are not in proportion on it")

    cadence: Optional[float] = None
    cadence_refusal: Optional[str] = None
    stamps = geography.calendar_times()
    if stamps is None:
        cadence_refusal = (
            "this record carries no calendar, so a frame index is a frame and not an hour. A "
            "declared lead in hours cannot be compared with a window in frames without a "
            "cadence, and inventing one would make the lead criterion a restatement of the "
            "window")
    else:
        seconds = np.asarray(stamps, dtype="datetime64[ns]").astype("int64") / 1e9
        if seconds.size < 2:
            cadence_refusal = ("this record carries one frame, and one frame has no cadence")
        else:
            gaps = np.diff(seconds)
            median = float(np.median(gaps))
            if median <= 0.0:
                cadence_refusal = "this record's timestamps do not increase"
            elif float(np.max(np.abs(gaps - median))) > CADENCE_TOLERANCE * abs(median):
                cadence_refusal = (
                    "this record's frames are not evenly spaced (gaps from %g to %g seconds "
                    "against a median of %g), so it has no one cadence and a lag in frames is "
                    "not a fixed number of hours"
                    % (float(gaps.min()), float(gaps.max()), median))
            else:
                cadence = median / 3600.0

    return PhysicalBridge(
        grid=grid, variable=variable.strip(),
        level=getattr(geography.field, "level", None),
        level_axis=getattr(geography.field, "level_axis", None),
        cell_km=(cell_low, cell_high), cell_refusal=cell_refusal,
        cadence_hours=cadence, cadence_refusal=cadence_refusal,
        anisotropy=grid.anisotropy() if hasattr(grid, "anisotropy") else {},
        record_times=np.asarray(geography.field.times, dtype=np.float64).tolist())


# ------------------------------------------------------------------------------ measuring one

def measure_pattern(projection: PatternProjection, bridge: PhysicalBridge) -> Dict[str, Any]:
    """What this pattern's own footprints say about its scale, its geometry and its size.

    Everything here is read from T4F.5's projection rather than from the signature, so the
    numbers a gate adjudicates are the numbers a reader can see on the map. The horizontal scale
    is the filter support the footprint was drawn with, and the octave label is carried beside
    it under its own name because the two differ by a factor of two and an entry declares which
    of them its envelope is about.
    """
    if not isinstance(projection, PatternProjection):
        raise InvalidParameterError(
            "projection", type(projection).__name__,
            "a T4F.5 PatternProjection. The cross-reference measures footprints, because a "
            "footprint is the only extent this transform actually justifies")
    if not isinstance(bridge, PhysicalBridge):
        raise InvalidParameterError("bridge", type(bridge).__name__, "a PhysicalBridge")

    supports: List[float] = []
    octaves: List[float] = []
    cardinalities = set()
    separations: List[float] = []
    for occurrence in projection.occurrences:
        cardinalities.add(len(occurrence.footprints))
        for footprint in occurrence.footprints:
            supports.append(float(footprint.support_cells))
            octaves.append(float(footprint.octave_cells))
        prints = occurrence.footprints
        for i, left in enumerate(prints):
            for right in prints[i + 1:]:
                separations.append(math.hypot(left.centre.row - right.centre.row,
                                              left.centre.col - right.centre.col))
    if not supports:
        raise InvalidParameterError(
            "projection.occurrences", 0,
            "at least one projected occurrence carrying at least one footprint. A pattern with "
            "nothing on the map has no measured scale, and labelling it against a declared "
            "envelope would be labelling an empty set")

    measured: Dict[str, Any] = {
        SCALE_FILTER_SUPPORT: bridge.length("horizontal_scale", min(supports), max(supports)),
        SCALE_OCTAVE_LABEL: bridge.length("horizontal_scale", min(octaves), max(octaves)),
        "separation": bridge.length("separation", min(separations), max(separations))
                      if separations else
                      Measurement(name="separation", low=None, high=None, units=None,
                                  basis="single-member occurrences",
                                  refusal="a constellation of one member has no separation"),
        "cardinality": sorted(cardinalities),
        "support_cells": (min(supports), max(supports)),
        "octave_cells": (min(octaves), max(octaves)),
        "n_occurrences": len(projection.occurrences),
    }
    return measured


# ------------------------------------------------------------------------------ one verdict

@dataclass(frozen=True)
class CriterionVerdict:
    """How one declared criterion came out, with what was measured and what was expected."""

    name: str
    status: str
    measured: Optional[Measurement]
    expected: Optional[Envelope]
    fraction_inside: Optional[float]
    note: str = ""

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {"criterion": self.name, "status": self.status}
        if self.measured is not None:
            record["measured"] = self.measured.as_record()
        if self.expected is not None:
            record["declared"] = self.expected.as_record()
        if self.fraction_inside is not None:
            record["fraction_of_the_measured_interval_inside"] = self.fraction_inside
        if self.note:
            record["note"] = self.note
        return record


def _envelope_verdict(name: str, measurement: Measurement,
                      envelope: Envelope) -> CriterionVerdict:
    if not measurement.measured:
        return CriterionVerdict(name=name, status=CRITERION_UNASSESSABLE, measured=measurement,
                                expected=envelope, fraction_inside=None,
                                note=measurement.refusal or "not measurable on this record")
    inside = envelope.overlaps(measurement.low, measurement.high)
    return CriterionVerdict(
        name=name, status=CRITERION_INSIDE if inside else CRITERION_OUTSIDE,
        measured=measurement, expected=envelope,
        fraction_inside=envelope.fraction_of(measurement.low, measurement.high),
        note=("the measured interval meets the declared envelope" if inside else
              "the measured interval lies wholly outside the declared envelope"))


@dataclass(frozen=True)
class PhenomenonMatch:
    """One pattern against one catalogue entry, criterion by criterion."""

    phenomenon: str
    citation: str
    status: str
    criteria: Tuple[CriterionVerdict, ...]

    @property
    def n_assessed(self) -> int:
        """How many *substantive* criteria this record could decide, either way.

        The `observables` check is excluded, exactly as the status rule below excludes it: it
        says the record could be asked, not that the answer was yes. Counting it here would put
        a two on a receipt beside a declared floor of two whenever a single envelope had been
        checked, which is the reading this module least wants a reader to make.
        """
        return sum(1 for item in self.criteria
                   if item.name != "observables" and item.status != CRITERION_UNASSESSABLE)

    def as_record(self) -> Dict[str, Any]:
        return {"phenomenon": self.phenomenon, "citation": self.citation, "status": self.status,
                "substantive_criteria_decided": self.n_assessed,
                "minimum_for_consistency": MINIMUM_ASSESSED_CRITERIA,
                "criteria": [item.as_record() for item in self.criteria]}


def match_phenomenon(measured: Mapping[str, Any], phenomenon: Phenomenon,
                     bridge: PhysicalBridge, *,
                     lead: Optional[Measurement] = None) -> PhenomenonMatch:
    """Check one pattern against one entry, and say for each criterion which of three it is.

    The observable comes first: an entry needing something this record does not hold is
    unassessable outright, and running its scale criterion anyway would produce a verdict about
    a quantity of the wrong field.
    """
    if bridge.observable not in phenomenon.observables:
        return PhenomenonMatch(
            phenomenon=phenomenon.name, citation=phenomenon.citation,
            status=MATCH_UNASSESSABLE,
            criteria=(CriterionVerdict(
                name="observables", status=CRITERION_UNASSESSABLE, measured=None, expected=None,
                fraction_inside=None,
                note=("this entry needs one of %s and this record carries %r. Nothing about "
                      "this phenomenon can be checked here, and calling the pattern "
                      "unrecognised would record the crop's contents as a fact about the "
                      "atmosphere" % (list(phenomenon.observables), bridge.observable))),))

    criteria: List[CriterionVerdict] = [
        CriterionVerdict(name="observables", status=CRITERION_INSIDE, measured=None,
                         expected=None, fraction_inside=None,
                         note="this record carries %r" % bridge.observable),
        _envelope_verdict("horizontal_scale", measured[phenomenon.scale_convention],
                          phenomenon.horizontal_scale),
    ]
    if lead is not None:
        criteria.append(_envelope_verdict("lead", lead, phenomenon.lead))
    else:
        criteria.append(CriterionVerdict(
            name="lead", status=CRITERION_UNASSESSABLE, measured=None,
            expected=phenomenon.lead, fraction_inside=None,
            note=("a lead is a property of a rule and not of a pattern; this pattern was "
                  "labelled outside any rule, so nothing here has a lag to compare")))
    if phenomenon.cardinality is not None:
        observed = list(measured["cardinality"])
        inside = int(phenomenon.cardinality) in observed
        criteria.append(CriterionVerdict(
            name="cardinality", status=CRITERION_INSIDE if inside else CRITERION_OUTSIDE,
            measured=Measurement(name="cardinality", low=float(min(observed)),
                                 high=float(max(observed)), units="members",
                                 basis="the members of this pattern's projected occurrences"),
            expected=Envelope(float(phenomenon.cardinality), float(phenomenon.cardinality),
                              "members"),
            fraction_inside=1.0 if inside else 0.0,
            note=("this pattern has %s members" % observed)))
    if phenomenon.separation is not None:
        criteria.append(_envelope_verdict("separation", measured["separation"],
                                          phenomenon.separation))

    # `observables` is not counted as evidence of anything: it says the record could be asked,
    # not that the answer was yes. Consistency established without two substantive criteria is
    # a failure to exclude wearing the word 'recognised'.
    substantive = tuple(item for item in criteria if item.name != "observables")
    if any(item.status == CRITERION_OUTSIDE for item in substantive):
        status = MATCH_EXCLUDED
    elif sum(1 for item in substantive
             if item.status == CRITERION_INSIDE) >= MINIMUM_ASSESSED_CRITERIA:
        status = MATCH_CONSISTENT
    else:
        status = MATCH_UNASSESSABLE
    return PhenomenonMatch(phenomenon=phenomenon.name, citation=phenomenon.citation,
                           status=status, criteria=tuple(criteria))


# ------------------------------------------------------------------------------ one pattern

@dataclass(frozen=True)
class PatternLabel:
    """R10's label for one pattern, with every entry it was checked against."""

    pattern_id: int
    label: str
    matches: Tuple[PhenomenonMatch, ...]
    consistent_with: Tuple[str, ...]
    excluded_by: Tuple[str, ...]
    unassessable_for: Tuple[str, ...]
    measured: Mapping[str, Any] = dc_field(default_factory=dict)

    @property
    def non_discriminating(self) -> bool:
        """Consistent with every entry that could be checked, and excluded by none of them.

        Entries the record cannot assess are not counted against this. They contribute no
        discrimination in either direction, so a pattern consistent with every *checkable*
        entry has been separated from nothing whether or not some fourth entry needed a field
        the crop does not carry.
        """
        return bool(self.consistent_with) and not self.excluded_by

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "pattern_id": self.pattern_id,
            "label": self.label,
            "consistent_with": list(self.consistent_with),
            "excluded_by": list(self.excluded_by),
            "unassessable_for": list(self.unassessable_for),
            "matches": [item.as_record() for item in self.matches],
            "recognition_note": RECOGNITION_NOTE,
        }
        if self.non_discriminating:
            record["caution"] = (
                "this pattern is consistent with every entry this record could check it "
                "against and excluded by none of them, so its label separates it from nothing "
                "and is not evidence for any one of them")
        return record


def label_pattern(pattern_id: int, measured: Mapping[str, Any],
                  catalogue: ReferenceCatalogue, bridge: PhysicalBridge, *,
                  lead: Optional[Measurement] = None) -> PatternLabel:
    """Label one pattern `recognised`, `unrecognised` or `unassessable` against the catalogue."""
    matches = tuple(match_phenomenon(measured, item, bridge, lead=lead) for item in catalogue)
    consistent = tuple(item.phenomenon for item in matches
                       if item.status == MATCH_CONSISTENT)
    excluded = tuple(item.phenomenon for item in matches if item.status == MATCH_EXCLUDED)
    unassessable = tuple(item.phenomenon for item in matches
                         if item.status == MATCH_UNASSESSABLE)
    if consistent:
        label = LABEL_RECOGNISED
    elif excluded:
        label = LABEL_UNRECOGNISED
    else:
        label = LABEL_UNASSESSABLE
    return PatternLabel(pattern_id=int(pattern_id), label=label, matches=matches,
                        consistent_with=consistent, excluded_by=excluded,
                        unassessable_for=unassessable, measured=dict(measured))


def cross_reference(catalogue: ReferenceCatalogue,
                    projections: Sequence[PatternProjection],
                    bridge: PhysicalBridge) -> Tuple[PatternLabel, ...]:
    """Label every mined pattern against the catalogue, in pattern order.

    No lead is supplied here, because a pattern on its own has no lag: the lead criterion is
    reported unassessable and the label rests on scale and geometry. `physical_gate` labels the
    ranked rules' antecedents again with the lead the rule actually declared.
    """
    if not isinstance(catalogue, ReferenceCatalogue):
        raise InvalidParameterError(
            "catalogue", type(catalogue).__name__, "a ReferenceCatalogue")
    labels = [label_pattern(item.pattern_id, measure_pattern(item, bridge), catalogue, bridge)
              for item in projections]
    return tuple(sorted(labels, key=lambda item: item.pattern_id))


# ------------------------------------------------------------------------------ can it fail?

@dataclass(frozen=True)
class Discrimination:
    """How much the catalogue could tell apart on this record, measured before adjudication.

    A cross-reference that excluded nothing has not recognised anything either; it has merely
    failed to reject, and a gate built on it cannot fail. So this is computed first and the gate
    reads it, rather than being a note appended to a verdict already reached.
    """

    n_phenomena: int
    n_patterns: int
    #: Pattern-entry pairs the catalogue excluded, not criteria: one pattern excluded by an
    #: entry on three separate criteria is one exclusion here.
    n_exclusions: int
    patterns_excluding_something: int
    patterns_recognised: int
    patterns_unassessable: int
    non_discriminating_patterns: Tuple[int, ...]
    overlapping_pairs: Tuple[Dict[str, Any], ...]
    cell_km: Optional[Tuple[float, float]]
    cadence_hours: Optional[float]
    verdict: str
    reason: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "n_phenomena": self.n_phenomena,
            "n_patterns": self.n_patterns,
            "pattern_entry_exclusions": self.n_exclusions,
            "patterns_excluding_at_least_one_entry": self.patterns_excluding_something,
            "patterns_recognised": self.patterns_recognised,
            "patterns_unassessable": self.patterns_unassessable,
            "patterns_consistent_with_every_entry": list(self.non_discriminating_patterns),
            "catalogue_overlaps": list(self.overlapping_pairs),
            "cell_km": (None if self.cell_km is None else
                        {"low": self.cell_km[0], "high": self.cell_km[1],
                         "ratio": self.cell_km[1] / self.cell_km[0],
                         "note": ("a scale in cells is this wide a range of kilometres on this "
                                  "crop before any measurement error, so an envelope narrower "
                                  "than the range cannot exclude anything")}),
            "cadence_hours": self.cadence_hours,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def discrimination(catalogue: ReferenceCatalogue, labels: Sequence[PatternLabel],
                   bridge: PhysicalBridge) -> Discrimination:
    """Measure whether this catalogue could have failed anything on this record."""
    exclusions = sum(len(item.excluded_by) for item in labels)
    excluding = sum(1 for item in labels if item.excluded_by)
    recognised = sum(1 for item in labels if item.label == LABEL_RECOGNISED)
    unassessable = sum(1 for item in labels if item.label == LABEL_UNASSESSABLE)
    blunt = tuple(item.pattern_id for item in labels if item.non_discriminating)

    if not labels:
        verdict, reason = GATE_INVALID, (
            "no pattern was cross-referenced, so nothing was recognised and nothing was "
            "excluded")
    elif unassessable == len(labels):
        # Two quite different things end here and a receipt that conflated them would send a
        # reader to fix the wrong one: an entry asking for an observable this record does not
        # hold, and an entry the record could be asked about but whose envelopes left fewer
        # than `MINIMUM_ASSESSED_CRITERIA` criteria actually decided.
        wrong_observable = sum(
            1 for item in labels
            if all(any(verdict.name == "observables"
                       and verdict.status == CRITERION_UNASSESSABLE
                       for verdict in match.criteria) for match in item.matches))
        verdict, reason = GATE_INVALID, (
            "every one of the %d patterns was unassessable. %d of them because no entry asks "
            "for %r, which is what this record carries; the rest because fewer than %d of each "
            "entry's criteria could be decided on this record, and a consistency resting on one "
            "criterion is a failure to exclude rather than a recognition"
            % (len(labels), wrong_observable, bridge.observable, MINIMUM_ASSESSED_CRITERIA))
    elif exclusions == 0:
        verdict, reason = GATE_INVALID, (
            "the catalogue excluded no pattern on any criterion, so it could not have failed "
            "anything and a PASS from it would record the width of its envelopes rather than "
            "the content of the record")
    else:
        verdict, reason = "DISCRIMINATING", (
            "%d of %d patterns were excluded by at least one entry, so the catalogue was "
            "capable of refusing recognition on this record" % (excluding, len(labels)))

    return Discrimination(
        n_phenomena=len(catalogue), n_patterns=len(labels), n_exclusions=exclusions,
        patterns_excluding_something=excluding, patterns_recognised=recognised,
        patterns_unassessable=unassessable, non_discriminating_patterns=blunt,
        overlapping_pairs=catalogue.overlapping_pairs(),
        cell_km=((bridge.cell_km_low, bridge.cell_km_high) if bridge.has_length else None),
        cadence_hours=bridge.cadence_hours, verdict=verdict, reason=reason)


# ------------------------------------------------------------------------------ the gate

#: The ranking keys a gate may be declared against, and the direction each is read in. Both are
#: figures the report already published; nothing here is recomputed, which is the discipline
#: T4F.4 priced when it showed a q-value moving from 0.0417 to 0.0050 under a narrowed family.
RANK_KEYS = {"q_value": "ascending", "p_value": "ascending", "lift": "descending",
             "support": "descending"}


@dataclass(frozen=True)
class GateDeclaration:
    """What was declared before the gate was run, hashed so that it cannot be declared after.

    `period_justification` is the documented event this record was chosen to contain, and its
    source. It is required and it is not verified here -- this module cannot read a journal --
    but it is hashed into the declaration, so a period justified after the answer was seen is a
    different declaration from the one the receipt names.
    """

    catalogue: ReferenceCatalogue
    rank_by: str
    top_n: int
    period_justification: str
    declared_by: str
    declared_on: str

    def __post_init__(self) -> None:
        if self.rank_by not in RANK_KEYS:
            raise InvalidParameterError(
                "GateDeclaration.rank_by", self.rank_by,
                "one of %s. T4F.4 showed that the ranking key decides the answer -- the same "
                "table's leading rule is a pattern the null did not distinguish under one key "
                "and the planted precursor under another -- so a gate must name its key before "
                "it is run rather than after" % sorted(RANK_KEYS))
        if not isinstance(self.top_n, int) or isinstance(self.top_n, bool) or self.top_n < 1:
            raise InvalidParameterError(
                "GateDeclaration.top_n", self.top_n,
                "a positive integer count of 'top results'. R10's gate is about what the mining "
                "pass surfaces first, and a top-N chosen after the ranking is seen is not a gate")
        for name, value in (("period_justification", self.period_justification),
                            ("declared_by", self.declared_by),
                            ("declared_on", self.declared_on)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "GateDeclaration.%s" % name, value,
                    "a non-empty %s. The acceptance is that a recognisable pattern is recovered "
                    "on a period containing a documented event; without naming the event and "
                    "its source, a PASS says only that some period contained something" % name)

    @property
    def digest(self) -> str:
        return _digest({
            "schema": DECLARATION_SCHEMA,
            "catalogue_digest": self.catalogue.digest,
            "rank_by": self.rank_by,
            "top_n": int(self.top_n),
            "period_justification": self.period_justification,
            "declared_by": self.declared_by,
            "declared_on": self.declared_on,
        })


@dataclass(frozen=True)
class RankedRule:
    """One rule in the declared ranking, with the label its antecedent earned."""

    rank: int
    antecedent: int
    consequent: int
    window: TransitionWindow
    figure: Optional[float]
    status: str
    label: Optional[PatternLabel]
    lead: Optional[Measurement]

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "rank": self.rank, "antecedent": self.antecedent, "consequent": self.consequent,
            "window": [float(self.window.minimum_lag), float(self.window.maximum_lag),
                       self.window.time_units],
            "figure": self.figure, "rule_status": self.status,
            "antecedent_label": None if self.label is None else self.label.label,
        }
        if self.lead is not None:
            record["lead"] = self.lead.as_record()
        if self.label is not None:
            record["consistent_with"] = list(self.label.consistent_with)
        return record


@dataclass(frozen=True)
class PhysicalGate:
    """R10's verdict: PASS, FAIL or INVALID, and everything it was reached from."""

    status: str
    declaration_digest: str
    catalogue_digest: str
    ranked: Tuple[RankedRule, ...]
    labels: Tuple[PatternLabel, ...]
    discrimination: Discrimination
    bridge: Mapping[str, Any]
    unmeasured_rules: int
    reasons: Tuple[str, ...]

    @property
    def phase_4g_may_start(self) -> bool:
        """The one thing this gate licenses, stated as the boolean the roadmap asks for."""
        return self.status == GATE_PASS

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "status": self.status,
            "phase_4g_may_start": self.phase_4g_may_start,
            "declaration_digest": self.declaration_digest,
            "catalogue_digest": self.catalogue_digest,
            "record": dict(self.bridge),
            "discrimination": self.discrimination.as_record(),
            "ranking": [item.as_record() for item in self.ranked],
            "rules_without_the_ranking_figure": self.unmeasured_rules,
            "labels": [item.as_record() for item in self.labels],
            "reasons": list(self.reasons),
            "recognition_note": RECOGNITION_NOTE,
            "claim_boundary": GATE_CLAIM_BOUNDARY,
        }


def _figure(rule: PrecursorRule, key: str) -> Optional[float]:
    value = getattr(rule, key, None)
    return None if value is None else float(value)


def lead_for(bridge: PhysicalBridge, series: Any, window: TransitionWindow) -> Measurement:
    """The rule's window in hours, or a refusal naming which clock the two sides are on.

    A cadence is a property of the record the coefficients came from, and a window is a lag on
    the event series' own observation grid. Converting one with the other is only sound when
    those are the same clock, and nothing in either object asserts that they are. What can be
    checked is the necessary condition: every frame the series searched must be a frame of the
    record. A series counted on some other lattice is refused rather than converted, because a
    lead in hours obtained by multiplying somebody else's frames by this record's cadence would
    look like a physical quantity and be an arithmetic coincidence.
    """
    if not bridge.has_clock:
        return bridge.lead(window)
    grid = getattr(series, "grid", None)
    frames = tuple(float(item) for item in getattr(grid, "frames", ()))
    if not frames:
        return Measurement(
            name="lead", low=None, high=None, units=None,
            basis="no observation grid was supplied with this series",
            refusal=("this series names no searched frames, so there is nothing to check its "
                     "clock against the record's"))
    known = set(bridge.record_times)
    missing = sorted(item for item in frames if item not in known)
    if missing:
        return Measurement(
            name="lead", low=None, high=None, units=None,
            basis="the series and the record are on different lattices",
            refusal=("%d of this series' %d searched frames are not frames of the record these "
                     "coefficients came from (%s and %d more), so its lags are counted on a "
                     "different clock. The record's cadence of %g hours per frame does not "
                     "convert them, and using it anyway would put a number of hours on a "
                     "quantity that has none"
                     % (len(missing), len(frames),
                        ", ".join("%g" % item for item in missing[:3]),
                        max(0, len(missing) - 3), bridge.cadence_hours)))
    return bridge.lead(window)


def physical_gate(report: PrecursorReport, declaration: GateDeclaration,
                  projections: Sequence[PatternProjection],
                  bridge: PhysicalBridge, series: Any) -> PhysicalGate:
    """Run R10's gate: was a recognisable pattern among the top results of a declared ranking.

    The ranking is a view of figures the report already published, sorted by the key the
    declaration named. Rules whose figure the report did not measure are counted and left out of
    the ranking rather than sorted as though a missing q-value were a good one. `series` is the
    event series the report was counted on, and it is here so that a lead in hours is produced
    only when its clock is the record's own.
    """
    if not isinstance(report, PrecursorReport):
        raise InvalidParameterError(
            "report", type(report).__name__, "a T4F.3 PrecursorReport")
    if not isinstance(declaration, GateDeclaration):
        raise InvalidParameterError(
            "declaration", type(declaration).__name__, "a GateDeclaration")
    catalogue = declaration.catalogue
    if catalogue.status != CATALOGUE_FROZEN:
        raise InvalidParameterError(
            "declaration.catalogue.status", catalogue.status,
            "a frozen catalogue. Which phenomena count as recognisable, in which envelopes and "
            "on whose authority, is the scientific content of this gate, and a draft is a "
            "proposal nobody has signed. Call `ReferenceCatalogue.freeze(signed_by=...)` when "
            "the entries and their citations have actually been reviewed")

    labels = cross_reference(catalogue, projections, bridge)
    by_pattern = {item.pattern_id: item for item in labels}
    power = discrimination(catalogue, labels, bridge)

    # Read back from the labels rather than measure a second time. The cross-reference above
    # has already measured every pattern off its footprints, and one measurement with several
    # readers is the shape this programme keeps arriving at: T4F.5 moved the per-anchor verdict
    # into one place for the same reason, and T4F.4 refuses to recompute a figure a report has
    # already published.
    measured_by_pattern = {item.pattern_id: dict(item.measured) for item in labels}
    key = declaration.rank_by
    scored = [(rule, _figure(rule, key)) for rule in report.rules]
    unmeasured = sum(1 for _rule, value in scored if value is None)
    usable = [(rule, value) for rule, value in scored if value is not None]
    reverse = RANK_KEYS[key] == "descending"
    usable.sort(key=lambda item: (item[1], item[0].antecedent, item[0].consequent),
                reverse=reverse)

    ranked: List[RankedRule] = []
    for position, (rule, value) in enumerate(usable[:int(declaration.top_n)], start=1):
        lead = lead_for(bridge, series, rule.window)
        label = None
        if rule.antecedent in measured_by_pattern:
            label = label_pattern(rule.antecedent, measured_by_pattern[rule.antecedent],
                                  catalogue, bridge, lead=lead)
        ranked.append(RankedRule(rank=position, antecedent=int(rule.antecedent),
                                 consequent=int(rule.consequent), window=rule.window,
                                 figure=value, status=rule.status, label=label, lead=lead))

    reasons: List[str] = [power.reason]
    recognised = [item for item in ranked
                  if item.label is not None and item.label.label == LABEL_RECOGNISED]
    if power.verdict == GATE_INVALID:
        status = GATE_INVALID
        reasons.append(
            "the cross-reference could not have failed, so it cannot have passed either. This "
            "is not a FAIL: nothing has been learned about the pipeline")
    elif not usable:
        status = GATE_INVALID
        reasons.append(
            "no rule in this report carries a %s, so there is no ranking to take the top %d of"
            % (key, declaration.top_n))
    elif recognised:
        status = GATE_PASS
        reasons.append(
            "rank %d of the %s ranking is rule %d -> %d, whose antecedent is consistent with %s"
            % (recognised[0].rank, key, recognised[0].antecedent, recognised[0].consequent,
               ", ".join(recognised[0].label.consistent_with)))
    else:
        status = GATE_FAIL
        unlabelled = sum(1 for item in ranked if item.label is None)
        reasons.append(
            "no antecedent among the top %d by %s is consistent with any declared phenomenon, "
            "on a catalogue that excluded %d pattern-entry pairs and so was capable of "
            "recognising one. %d of the ranked rules named a pattern that is not in this "
            "projection and so could not be labelled at all. The standing interpretation of a "
            "FAIL is that the pipeline is broken (R10), and Phase 4G does not start"
            % (declaration.top_n, key, power.n_exclusions, unlabelled))
    if by_pattern and all(item.non_discriminating for item in by_pattern.values()):
        reasons.append(
            "every labelled pattern is consistent with every entry, so these labels separate "
            "nothing")

    return PhysicalGate(
        status=status, declaration_digest=declaration.digest,
        catalogue_digest=catalogue.digest, ranked=tuple(ranked), labels=labels,
        discrimination=power, bridge=bridge.as_record(), unmeasured_rules=unmeasured,
        reasons=tuple(reasons))


def describe_gate(gate: PhysicalGate) -> Dict[str, Any]:
    """The gate as a plain record, for a receipt or a route."""
    if not isinstance(gate, PhysicalGate):
        raise InvalidParameterError("gate", type(gate).__name__, "a PhysicalGate")
    return gate.as_record()
