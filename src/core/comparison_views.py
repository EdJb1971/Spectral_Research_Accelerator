"""TG17.8 Scientific comparison views: the abstraction made inspectable rather than magical.

**The failure this exists to prevent.** Every earlier G17 slice refuses a bad *declaration*. This
one refuses a bad *picture*, which is a different and harder problem, because a picture is
persuasive before it is read. Put a reanalysis temperature series and an order-book depth series
on one y-axis and the eye performs a comparison the manifest never authorised: it sees one curve
above another and concludes *larger*. Draw them left to right and it concludes *first, therefore
before*. Nothing in the arithmetic said either thing. The chart said both, and a reader who
believes the chart has been given a result the study is structurally incapable of producing.

So the axis is not a rendering detail. `Axis` is a declaration of what a coordinate *means*, and
a `native_magnitude` axis carrying two domains raises `MagnitudeEquivalenceError` rather than
drawing. There is no plotting call in this module to check: the refusal happens when the view is
*constructed*, so a view that would have laundered two units into one number cannot reach a
browser, an export or a screenshot. Likewise `assert_reading_admissible` refuses `precedence` on
the scale-shape coordinate, `magnitude_equivalence` in every mode, and `causality` in every mode -
the first two because the coordinate they would be computed on carries no clock and no unit, and
the third because the design that would license it has no field in the manifest to be declared in.

Four properties follow, and each is the answer to a specific way a chart lies:

1.  **Absence is not zero.** A coverage cell is `COVERED`, `SPARSE`, `ABSENT` or `REFUSED` - a
    named state, never a float. A gap rendered as 0.0 is the single most expensive graphical
    mistake available here: it turns "we could not look" into "we looked and found nothing",
    which is the difference between an unasked question and a null result.

2.  **Every mark carries its artefact, or says why it has none.** `Mark` requires exactly one of
    `artifact_sha256` and `no_artifact_reason`. A point whose provenance is optional is a point
    that will eventually be drawn from a variable someone computed in a notebook.

3.  **The role is carried in three channels.** Colour alone cannot distinguish a generated
    candidate from a held-out confirmation for a reader who cannot see the colour, and the
    distinction between those two is the entire scientific content of the generate/confirm split.
    Each `Encoding` therefore carries a colour, a marker shape *and* a word, and registration
    refuses a role that duplicates any of the three.

4.  **Every visual has the table under it.** `render_view` returns `table` beside `body`, from the
    same values, so the numbers behind a picture are never reconstructed by eye from it.

**What this module does not do.** It opens no archive, runs no statistic and reports no measured
value. Stage workers do not exist yet, so the result and null views render the *declared* cells
with `NOT_YET_MEASURED` and the reason - which is the same discipline the composer already
applies to an empty panel. The correction denominator, the p-value floor and the family size are
shown now because they are arithmetic about the declaration and are computable before a byte
exists; that is precisely why they belong in front of the researcher before the run, not after.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.experiment_adapter import adapter_for_domain
from src.core.experiment_manifest import CrossDomainExperimentSpec, manifest_sha256
from src.core.registry import Registry
from src.core.structural_alignment import (CALENDAR_RELATIONSHIPS, MODE_FORBIDS,
                                           MODE_RELATIONSHIPS, SCALE_SHAPE_RELATIONSHIPS)


SCHEMA = "comparison-views/v1"

#: What a coordinate can mean. The list is closed because an axis kind nobody named is an axis
#: whose sharing rule nobody decided.
AXIS_KINDS: Tuple[str, ...] = (
    "utc_instant",
    "normalized_structural_scale",
    "native_magnitude",
    "dimensionless_statistic",
    "rank",
    "categorical",
)

#: Axis kinds more than one domain may occupy at once. `native_magnitude` is deliberately absent:
#: a degree Celsius and a unit of book depth have no common ruler, and an axis that carried both
#: would be inviting exactly the comparison every claim boundary in this repository forbids.
SHARED_AXIS_KINDS: Tuple[str, ...] = (
    "utc_instant", "normalized_structural_scale", "dimensionless_statistic", "rank", "categorical")

#: A coverage cell's state. Named values rather than a fraction, so that "no data here" and
#: "a measured value of zero here" cannot be the same pixel.
COVERAGE_CELLS: Tuple[str, ...] = ("COVERED", "SPARSE", "ABSENT", "REFUSED")

#: Readings no alignment in this framework licenses, in any mode. They are named here rather
#: than merely absent so that asking for one produces a refusal with a reason instead of an
#: unrecognised string, which is the difference between a boundary and a gap in a lookup table.
NEVER_ADMISSIBLE_READINGS: Tuple[str, ...] = ("magnitude_equivalence", "semantic_equivalence")

#: Readings the *manifest* layer may legitimately declare and test, and the *view* layer may
#: still not draw.
#:
#: `causality` is a declarable relationship in `calendar_aligned` mode, and that is correct: a
#: study with an external intervention design can test it, and `MODE_RELATIONSHIPS` is the list of
#: what may be declared. Rendering is a different question. Every alignment this framework
#: computes is observational, and no field of `CrossDomainExperimentSpec` can carry the design
#: that would license the arrow - so a view drawing one would be asserting from a picture exactly
#: what the manifest cannot state. The split matters: refusing the declaration would forbid a
#: legitimate study, and permitting the drawing would let any co-occurrence be read as a cause.
REQUIRES_EXTERNAL_DESIGN: Tuple[str, ...] = ("causality",)

READINGS: Tuple[str, ...] = tuple(sorted(
    set(CALENDAR_RELATIONSHIPS) | set(SCALE_SHAPE_RELATIONSHIPS) | set(NEVER_ADMISSIBLE_READINGS)))


class MagnitudeEquivalenceError(InvalidParameterError):
    """Two domains were asked to share one axis of native magnitude."""

    def __init__(self, axis: str, domains: Sequence[str], units: Optional[str]) -> None:
        super().__init__(
            "Axis(%s).domains" % axis, list(domains),
            "one domain on an axis of native magnitude. %s have no common ruler, so a shared "
            "axis would render a difference in units as a difference in size. Give each domain "
            "its own axis, or move to a dimensionless statistic that says what it is comparing"
            % " and ".join(domains),
            axis=axis, units=units)
        self.axis = axis
        self.domains = tuple(domains)


class ForbiddenReadingError(InvalidParameterError):
    """A view was asked to express a reading its comparison mode cannot support."""

    def __init__(self, mode: str, reading: str, detail: str) -> None:
        super().__init__("reading", reading, detail, mode=mode)
        self.mode = mode
        self.reading = reading


def assert_reading_admissible(mode: str, reading: str) -> None:
    """Refuse a reading the mode cannot support, by name and with the mode's own boundary.

    This is the check that makes the TG17.0 semantic traps unrenderable rather than merely
    discouraged. `magnitude_equivalence` and `semantic_equivalence` are refused in *both* modes:
    no alignment in this framework licenses them, so there is no mode in which asking is a
    legitimate question with an inconvenient answer. `causality` is refused separately and for a
    different reason - see `REQUIRES_EXTERNAL_DESIGN`.
    """
    if mode not in MODE_RELATIONSHIPS:
        raise UnknownNameError("comparison mode", mode, sorted(MODE_RELATIONSHIPS))
    if reading not in READINGS:
        raise UnknownNameError("reading", reading, list(READINGS), mode=mode)
    if reading in NEVER_ADMISSIBLE_READINGS:
        raise ForbiddenReadingError(
            mode, reading,
            "a reading some alignment in this framework can support. Neither shared UTC support "
            "nor a shared normalized scale coordinate makes two native quantities the same "
            "quantity, so this reading has no mode to be admissible in")
    if reading in REQUIRES_EXTERNAL_DESIGN:
        raise ForbiddenReadingError(
            mode, reading,
            "a reading a view can draw. Every alignment computed here is observational, and the "
            "external design that would license this reading has no field in the manifest to be "
            "declared in. A study holding such a design may still declare and test %r; what it "
            "may not do is read it off one of these pictures" % reading)
    if reading not in MODE_RELATIONSHIPS[mode]:
        raise ForbiddenReadingError(
            mode, reading,
            "a relationship this mode admits (%s). %s"
            % (", ".join(MODE_RELATIONSHIPS[mode]), MODE_FORBIDS[mode]))


# ------------------------------------------------------------------------------------- axes


@dataclass(frozen=True)
class Axis:
    """One coordinate, and which domains are permitted to occupy it."""

    name: str
    kind: str
    domains: Tuple[str, ...]
    units: Optional[str] = None
    why: str = ""

    def __post_init__(self) -> None:
        if self.kind not in AXIS_KINDS:
            raise UnknownNameError("axis kind", self.kind, list(AXIS_KINDS), axis=self.name)
        if not self.domains:
            raise InvalidParameterError(
                "Axis(%s).domains" % self.name, list(self.domains),
                "at least one domain. An axis belonging to nothing cannot be checked for the "
                "one property axes exist to be checked for")
        if len(self.domains) > 1 and self.kind not in SHARED_AXIS_KINDS:
            raise MagnitudeEquivalenceError(self.name, self.domains, self.units)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "domains": list(self.domains),
                "units": self.units, "shared": len(self.domains) > 1, "why": self.why}


# -------------------------------------------------------------------------------- encodings


@dataclass(frozen=True)
class Encoding:
    """How one kind of mark is told apart, in three channels rather than one."""

    role: str
    word: str
    colour: str
    marker: str
    ordinal: int
    definition: str
    admits_claim: bool

    def describe(self) -> Dict[str, Any]:
        return {"role": self.role, "word": self.word, "colour": self.colour,
                "marker": self.marker, "ordinal": self.ordinal,
                "definition": self.definition, "admits_claim": self.admits_claim}


ENCODINGS: Registry[Encoding] = Registry("comparison view encoding")


def register_encoding(encoding: Encoding, *, replace: bool = False) -> Encoding:
    """Register a mark role, refusing one that is not distinguishable in all three channels."""
    for existing in (entry.value for entry in ENCODINGS.entries()):
        if existing.role == encoding.role and replace:
            continue
        for channel in ("word", "colour", "marker"):
            if getattr(existing, channel) == getattr(encoding, channel):
                raise InvalidParameterError(
                    "Encoding(%s).%s" % (encoding.role, channel), getattr(encoding, channel),
                    "a value no other role uses. %r already uses it, and a role distinguished in "
                    "only two of three channels is one a reader loses whenever the third is "
                    "unavailable to them" % existing.role)
    ENCODINGS.register(encoding.role, description=encoding.definition,
                       capabilities=encoding.describe(), replace=replace)(encoding)
    return encoding


for _encoding in (
    Encoding("generated_candidate", word="candidate", colour="#f59e0b", marker="hollow_circle",
             ordinal=1, admits_claim=False,
             definition=("Produced by a pass whose selection used the data it was selected on. "
                         "Uncorrected, and not a result at any p-value.")),
    Encoding("held_out_confirmation", word="confirmed", colour="#22c55e", marker="filled_square",
             ordinal=2, admits_claim=True,
             definition=("Tested on a partition frozen before it was opened, against the "
                         "confirmatory family named in the manifest.")),
    Encoding("external_transfer", word="transferred", colour="#6366f1", marker="triangle",
             ordinal=3, admits_claim=True,
             definition=("A frozen motif applied to a target the motif's author never saw. The "
                         "target is spent once and cannot be reused.")),
    Encoding("null_draw", word="surrogate", colour="#64748b", marker="thin_line", ordinal=4,
             admits_claim=False,
             definition=("One draw from a declared null family. The reference a statistic is "
                         "read against, never a finding of its own.")),
    Encoding("refusal", word="refused", colour="#ef4444", marker="cross", ordinal=5,
             admits_claim=False,
             definition=("The framework declined to produce this cell, with a reason. A refusal "
                         "occupies its cell rather than leaving it blank.")),
):
    register_encoding(_encoding)


def encoding_for(role: str) -> Encoding:
    return ENCODINGS.get(role)


def ordered_encodings() -> List[Encoding]:
    """Roles in their declared order, which is the order a legend must use."""
    return sorted((entry.value for entry in ENCODINGS.entries()), key=lambda item: item.ordinal)


# ------------------------------------------------------------------------------------ marks


@dataclass(frozen=True)
class Mark:
    """One thing a view draws, with the artefact it came from or the reason it has none."""

    label: str
    role: str
    domain: Optional[str] = None
    value: Optional[float] = None
    display: str = ""
    artifact_sha256: Optional[str] = None
    no_artifact_reason: str = ""

    def __post_init__(self) -> None:
        encoding_for(self.role)
        has_artifact = bool(self.artifact_sha256)
        has_reason = bool(self.no_artifact_reason.strip())
        if has_artifact == has_reason:
            raise InvalidParameterError(
                "Mark(%s)" % self.label, self.artifact_sha256 or self.no_artifact_reason,
                "exactly one of an artefact digest and a reason there is none. A mark with "
                "neither cannot be traced to anything immutable; a mark with both is claiming "
                "two different provenances for one point")

    def describe(self) -> Dict[str, Any]:
        encoding = encoding_for(self.role)
        return {"label": self.label, "role": self.role, "word": encoding.word,
                "colour": encoding.colour, "marker": encoding.marker, "domain": self.domain,
                "value": self.value, "display": self.display or self.label,
                "artifact_sha256": self.artifact_sha256,
                "no_artifact_reason": self.no_artifact_reason,
                "admits_claim": encoding.admits_claim}


# ---------------------------------------------------------------------------------- context


@dataclass(frozen=True)
class ViewContext:
    """Everything a view may read. Deliberately small, and none of it is a measurement."""

    spec: CrossDomainExperimentSpec
    preflight: Mapping[str, Any]
    receipt: Optional[Mapping[str, Any]] = None

    @property
    def mode(self) -> str:
        return self.spec.mode

    @property
    def domains(self) -> Tuple[str, ...]:
        return tuple(row.domain for row in self.spec.observations)

    def coverage_row(self, domain: str) -> Mapping[str, Any]:
        for row in self.preflight.get("coverage", []):
            if row.get("domain") == domain:
                return row
        return {}

    @property
    def run_state(self) -> str:
        return str((self.receipt or {}).get("state") or "NO_RUN")

    @property
    def results_exist(self) -> bool:
        """Whether any measured value could legitimately be shown.

        Two conditions, not one. The run must have completed *and* the stage that produces
        measurements must have published an artefact for it. A COMPLETE state alone is what a
        view would have to trust if this returned `state == "COMPLETE"`, and TG17.6 permits a
        run to complete under `partial_permitted` with mining components missing by name.
        """
        if self.run_state != "COMPLETE":
            return False
        artefacts = dict((self.receipt or {}).get("artefacts") or {})
        return any(key.startswith("MINING/") for key in artefacts)


NOT_YET_MEASURED = "NOT_YET_MEASURED"


def _not_yet(context: ViewContext) -> str:
    """Why a declared cell has no number in it, in the researcher's terms."""
    if context.run_state == "NO_RUN":
        return ("no run has been opened at this manifest's address, so this cell is an unasked "
                "question rather than an empty result")
    if not context.results_exist:
        return ("the run at this manifest's address is %s and has published no mining artefact, "
                "so this cell has not been measured" % context.run_state)
    return ("this build ships the comparison views over a declared plan; the stage workers that "
            "produce measured values arrive with the slices that own them")


# ----------------------------------------------------------------------------------- views


@dataclass(frozen=True)
class ComparisonView:
    """One linked view: what it asks, what it may conclude, and what it draws it on."""

    view_id: str
    ordinal: int
    title: str
    question: str
    axes: Tuple[Axis, ...]
    roles: Tuple[str, ...]
    may_conclude: str
    may_not_conclude: Tuple[str, ...]
    build: Callable[["ViewContext"], Dict[str, Any]]
    selectable: bool = False

    def __post_init__(self) -> None:
        for role in self.roles:
            encoding_for(role)

    def describe(self) -> Dict[str, Any]:
        return {"view_id": self.view_id, "ordinal": self.ordinal, "title": self.title,
                "question": self.question,
                "axes": [axis.describe() for axis in self.axes],
                "roles": list(self.roles),
                "may_conclude": self.may_conclude,
                "may_not_conclude": list(self.may_not_conclude),
                "selectable": self.selectable}


COMPARISON_VIEWS: Registry[ComparisonView] = Registry("comparison view")


def register_view(view: ComparisonView, *, replace: bool = False) -> ComparisonView:
    COMPARISON_VIEWS.register(view.view_id, description=view.question,
                              capabilities=view.describe(), replace=replace)(view)
    return view


def ordered_views() -> List[ComparisonView]:
    """Views in their declared order.

    By ordinal, not by name: `Registry.entries()` sorts alphabetically, and the reading order of
    a workbench is a scientific statement about what is looked at before what.
    """
    return sorted((entry.value for entry in COMPARISON_VIEWS.entries()),
                  key=lambda view: view.ordinal)


# --------------------------------------------------------------------- 1. coverage timeline


def _cell_state(row: Mapping[str, Any], window: Mapping[str, Any]) -> Tuple[str, str]:
    if not row:
        return "REFUSED", "this domain has no coverage row; preflight refused it by name"
    if row.get("status") == "REFUSED":
        return "REFUSED", str(row.get("reason") or "preflight refused this domain")
    if not window:
        return "ABSENT", "the domain's plan addresses no support inside this window"
    if window.get("coverage_exact"):
        return "COVERED", "metadata addresses the requested UTC extent exactly"
    return "SPARSE", ("native support is sparse or sector-bounded; the covered fraction is "
                      "established only after acquisition")


def _build_coverage_timeline(context: ViewContext) -> Dict[str, Any]:
    windows = [{"name": window.name,
                "start_utc": window.start_utc.isoformat(),
                "end_utc": window.end_utc.isoformat()} for window in context.spec.windows]
    rows: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for observation in context.spec.observations:
        row = context.coverage_row(observation.domain)
        planned = {entry.get("name"): entry for entry in row.get("windows", [])}
        cells = []
        for window in windows:
            entry = planned.get(window["name"], {})
            state, reason = _cell_state(row, entry)
            mark = Mark(label="%s/%s" % (observation.domain, window["name"]),
                        role="refusal" if state == "REFUSED" else "generated_candidate",
                        domain=observation.domain, display=state,
                        no_artifact_reason=("coverage is planned from archive metadata; no "
                                            "artefact exists until acquisition runs"))
            cells.append({"window": window["name"], "state": state, "reason": reason,
                          "estimated_bytes": int(entry.get("estimated_bytes", 0)),
                          "is_measured_zero": False,
                          "mark": mark.describe()})
            table.append({"domain": observation.domain, "window": window["name"],
                          "state": state, "estimated_bytes": int(entry.get("estimated_bytes", 0)),
                          "support_kind": row.get("support_kind"),
                          "native_cadence_seconds": row.get("native_cadence_seconds")})
        rows.append({"domain": observation.domain, "label": observation.label,
                     "support_kind": row.get("support_kind"),
                     "native_cadence_seconds": row.get("native_cadence_seconds"),
                     "cells": cells})
    return {
        "windows": windows,
        "rows": rows,
        "table": {"columns": ["domain", "window", "state", "estimated_bytes", "support_kind",
                              "native_cadence_seconds"], "rows": table},
        "zero_distinction": ("ABSENT and SPARSE are states, not the number zero. Nothing in this "
                             "view has been measured, so no cell here can be a measured zero."),
    }


register_view(ComparisonView(
    view_id="coverage_timeline", ordinal=1,
    title="Cross-domain coverage timeline",
    question="Which domains have addressable support inside each declared window?",
    axes=(Axis("utc", "utc_instant", domains=("*",), units="UTC",
               why="Every domain's support is placed on one clock because UTC is the one "
                   "coordinate they genuinely share."),
          Axis("domain", "categorical", domains=("*",),
               why="Rows are domains. Their vertical order carries no magnitude.")),
    roles=("generated_candidate", "refusal"),
    may_conclude="Which windows a domain can be planned for from archive metadata alone.",
    may_not_conclude=("that a covered window contains a signal",
                      "that an absent cell is a measured zero",
                      "that two covered rows are measuring comparable quantities"),
    build=_build_coverage_timeline, selectable=True))


# ---------------------------------------------------------------------- 2. native preview


def _build_native_preview(context: ViewContext) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for observation in context.spec.observations:
        coverage = context.coverage_row(observation.domain)
        adapter = adapter_for_domain(observation.domain)
        declaration = adapter.declaration
        rows.append({
            "domain": observation.domain,
            "native": {
                "measure": observation.measure,
                "semantics": observation.semantics,
                "units": observation.units,
                "support_kind": coverage.get("support_kind"),
                "native_cadence_seconds": coverage.get("native_cadence_seconds"),
                "axis": Axis("%s native" % observation.domain, "native_magnitude",
                             domains=(observation.domain,), units=observation.units,
                             why="One domain only. This quantity has no shared ruler.").describe(),
            },
            "canonical": {
                "role": observation.role,
                "channels": list(context.spec.family.channels),
                "violations": list(declaration.violations),
                "lag_policy": declaration.lag_policy,
                "axis": Axis("%s structural" % observation.domain, "dimensionless_statistic",
                             domains=(observation.domain,),
                             why="The canonical trajectory is dimensionless by construction; "
                                 "that is what makes it comparable at all.").describe(),
            },
            "comparable_across_domains": False,
            "why_not": ("The native panel is in %s. Nothing converts %s into another domain's "
                        "units, so the two native panels are read side by side and never "
                        "subtracted." % (observation.units, observation.units)),
        })
        table.append({"domain": observation.domain, "measure": observation.measure,
                      "units": observation.units, "semantics": observation.semantics,
                      "role": observation.role, "support_kind": coverage.get("support_kind")})
    return {
        "rows": rows,
        "table": {"columns": ["domain", "measure", "units", "semantics", "role", "support_kind"],
                  "rows": table},
        "shared_magnitude_axis": None,
        "why_no_shared_axis": ("Each domain keeps its own magnitude axis. Constructing a shared "
                               "one raises MagnitudeEquivalenceError rather than rescaling."),
    }


register_view(ComparisonView(
    view_id="native_preview", ordinal=2,
    title="Native record beside canonical trajectory",
    question="What did the archive actually contain, and what did the adapter make of it?",
    axes=(Axis("record index", "rank", domains=("*",),
               why="Position within a record. Shared because it is an ordinal, not a quantity."),),
    roles=("generated_candidate",),
    may_conclude="What each adapter received and what it produced, per domain.",
    may_not_conclude=("that two native panels share a magnitude scale",
                      "that a taller native curve is a larger effect",
                      "that the canonical trajectory preserves the native units"),
    build=_build_native_preview))


# ------------------------------------------------------------------- 3. scale mapping


def _build_scale_mapping(context: ViewContext) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for scale in context.spec.family.scales:
        entries = []
        for observation in context.spec.observations:
            coverage = context.coverage_row(observation.domain)
            cadence = coverage.get("native_cadence_seconds")
            native = None if cadence in (None, 0) else float(scale) * float(cadence)
            entries.append({"domain": observation.domain,
                            "native_cadence_seconds": cadence,
                            "native_duration_seconds": native,
                            "native_units": "s",
                            "resolvable": native is not None,
                            "why_not": ("" if native is not None else
                                        "this domain declares no regular native cadence, so a "
                                        "structural scale has no single native duration here")})
            table.append({"coordinate": float(scale), "domain": observation.domain,
                          "native_cadence_seconds": cadence,
                          "native_duration_seconds": native})
        rows.append({"coordinate": float(scale), "domains": entries})
    return {
        "rows": rows,
        "table": {"columns": ["coordinate", "domain", "native_cadence_seconds",
                              "native_duration_seconds"], "rows": table},
        "mapping_note": ("A coordinate is shared; the native durations under it are not. The same "
                         "coordinate can be 1.8 hours in one domain and 46 days in another, and "
                         "both numbers stay attached to it."),
        "carries_a_clock": False,
    }


register_view(ComparisonView(
    view_id="scale_mapping", ordinal=3,
    title="Native-to-structural scale mapping",
    question="What native duration does each declared structural scale mean in each domain?",
    axes=(Axis("structural scale", "normalized_structural_scale", domains=("*",),
               why="Dimensionless by construction, which is the only reason it can be shared."),
          Axis("domain", "categorical", domains=("*",), why="Rows are domains.")),
    roles=("generated_candidate",),
    may_conclude="Which native duration a structural coordinate corresponds to, per domain.",
    may_not_conclude=("simultaneity", "precedence", "that equal coordinates are equal durations"),
    build=_build_scale_mapping))


# --------------------------------------------------------------------- 4. result matrix


def _combinations(context: ViewContext) -> List[Tuple[str, ...]]:
    domains = context.domains
    out: List[Tuple[str, ...]] = []
    for arity in sorted(set(int(value) for value in context.spec.family.domain_arities)):
        if 2 <= arity <= len(domains):
            out.extend(itertools.combinations(domains, arity))
    return out


def _build_result_matrix(context: ViewContext) -> Dict[str, Any]:
    correction = dict(dict(context.preflight.get("family", {})).get("correction", {}))
    denominator = int(correction.get("correction_unit_members", 0))
    reason = _not_yet(context)
    cells: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for combination in _combinations(context):
        for relationship in context.spec.family.relationships:
            admissible = True
            refusal = ""
            try:
                assert_reading_admissible(context.mode, relationship)
            except (ForbiddenReadingError, UnknownNameError) as error:
                admissible, refusal = False, str(error)
            role = "generated_candidate" if admissible else "refusal"
            mark = Mark(label="%s:%s" % ("+".join(combination), relationship), role=role,
                        display=NOT_YET_MEASURED if admissible else "REFUSED",
                        no_artifact_reason=refusal or reason)
            cells.append({"domains": list(combination), "arity": len(combination),
                          "relationship": relationship,
                          "status": NOT_YET_MEASURED if admissible else "REFUSED",
                          "admissible": admissible,
                          "reason": refusal or reason,
                          "corrected_over": denominator,
                          "mark": mark.describe()})
            table.append({"domains": "+".join(combination), "relationship": relationship,
                          "status": NOT_YET_MEASURED if admissible else "REFUSED",
                          "corrected_over": denominator,
                          "alpha": float(context.spec.alpha),
                          "correction": context.spec.correction})
    return {
        "cells": cells,
        "arities": sorted(set(len(cell["domains"]) for cell in cells)),
        "correction": {"method": context.spec.correction, "alpha": float(context.spec.alpha),
                       "denominator": denominator,
                       "declared_search_members": int(correction.get("declared_search_members",
                                                                     0)),
                       "stage": correction.get("stage"),
                       "why_visible": ("A corrected value without its denominator is a number "
                                       "whose meaning is held somewhere else.")},
        "table": {"columns": ["domains", "relationship", "status", "corrected_over", "alpha",
                              "correction"], "rows": table},
    }


register_view(ComparisonView(
    view_id="result_matrix", ordinal=4,
    title="Pair, triple and quartet result matrix",
    question="Which declared domain combinations were tested for which relationship?",
    axes=(Axis("domain combination", "categorical", domains=("*",),
               why="A combination is a label, not a position on a scale."),
          Axis("corrected statistic", "dimensionless_statistic", domains=("*",),
               why="Shared only because it is dimensionless and states its denominator.")),
    roles=("generated_candidate", "held_out_confirmation", "refusal"),
    may_conclude="Which combinations the declared family covers, and what each cell is "
                 "corrected over.",
    may_not_conclude=("causality", "that an untested cell is a negative result",
                      "that a larger statistic in one cell is a larger effect in native units"),
    build=_build_result_matrix, selectable=True))


# ------------------------------------------------------------- 5. motif correspondence


def _build_motif_correspondence(context: ViewContext) -> Dict[str, Any]:
    motifs = [name for name in context.spec.family.motifs if name != "none"]
    reason = _not_yet(context)
    rows: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    if not motifs:
        return {
            "rows": [], "table": {"columns": ["motif", "target_domain", "status"], "rows": []},
            "unavailable_reason": ("this manifest declares no motif, so there is no frozen shape "
                                   "to look for. Declaring one is a family axis and it is "
                                   "priced: it multiplies the declared search."),
            "declared_motifs": [],
        }
    for motif in motifs:
        for observation in context.spec.observations:
            mark = Mark(label="%s->%s" % (motif, observation.domain), role="external_transfer",
                        domain=observation.domain, display=NOT_YET_MEASURED,
                        no_artifact_reason=reason)
            rows.append({"motif": motif, "target_domain": observation.domain,
                         "status": NOT_YET_MEASURED, "reason": reason,
                         "target_spent": False,
                         "mark": mark.describe()})
            table.append({"motif": motif, "target_domain": observation.domain,
                          "status": NOT_YET_MEASURED})
    return {"rows": rows,
            "table": {"columns": ["motif", "target_domain", "status"], "rows": table},
            "unavailable_reason": "",
            "declared_motifs": motifs,
            "transfer_note": ("A transfer target is spent by its first opening. A second transfer "
                              "onto the same target is a second look at data already used.")}


register_view(ComparisonView(
    view_id="motif_correspondence", ordinal=5,
    title="Motif correspondence and transfer",
    question="Where does a frozen shape recur, and onto which unseen target was it transferred?",
    axes=(Axis("structural scale", "normalized_structural_scale", domains=("*",),
               why="A motif is a shape on the normalized coordinate."),
          Axis("target domain", "categorical", domains=("*",), why="Targets are labels.")),
    roles=("generated_candidate", "external_transfer", "refusal"),
    may_conclude="That a frozen shape recurred in a target the author had not seen.",
    may_not_conclude=("that recurrence is a shared mechanism", "precedence", "causality"),
    build=_build_motif_correspondence))


# ------------------------------------------------------- 6. nulls, correction and power


def _build_null_and_correction(context: ViewContext) -> Dict[str, Any]:
    from src.core.structural_nulls import NULL_FAMILIES

    family = dict(context.preflight.get("family", {}))
    correction = dict(family.get("correction", {}))
    nulls: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for declaration in context.spec.nulls:
        registered = NULL_FAMILIES.get(declaration.method)
        nulls.append({
            "name": declaration.name,
            "method": declaration.method,
            "replications": int(declaration.replications),
            "preserves": list(registered.preserves),
            "destroys": list(registered.destroys),
            "operates_on": registered.operates_on,
            "distribution": {"status": NOT_YET_MEASURED, "reason": _not_yet(context),
                             "mark": Mark(label=declaration.name, role="null_draw",
                                          display=NOT_YET_MEASURED,
                                          no_artifact_reason=_not_yet(context)).describe()},
        })
        table.append({"null": declaration.name, "method": declaration.method,
                      "replications": int(declaration.replications),
                      "preserves": ", ".join(registered.preserves)})
    resolution = {
        "declared_surrogates": int(family.get("declared_surrogates", 0)),
        "surrogates_required": int(correction.get("surrogates_required", 0)),
        "p_value_floor": float(correction.get("p_value_floor", 0.0)),
        "affordable": bool(correction.get("affordable", False)),
        "warning": correction.get("warning", ""),
        "note": ("The floor is the smallest p-value this ensemble can produce. A family whose "
                 "corrected threshold is below its own floor will run, cost the full amount and "
                 "be arithmetically incapable of rejecting anything."),
    }
    return {
        "nulls": nulls,
        "correction": {"method": context.spec.correction, "alpha": float(context.spec.alpha),
                       "denominator": int(correction.get("correction_unit_members", 0)),
                       "declared_search_members": int(correction.get("declared_search_members",
                                                                     0)),
                       "stage": correction.get("stage"),
                       "held_out_partition": correction.get("held_out_partition")},
        "resolution": resolution,
        "table": {"columns": ["null", "method", "replications", "preserves"], "rows": table},
    }


register_view(ComparisonView(
    view_id="null_and_correction", ordinal=6,
    title="Null distributions, corrected values and resolution",
    question="What is each statistic read against, and can this ensemble resolve it at all?",
    axes=(Axis("statistic", "dimensionless_statistic", domains=("*",),
               why="A surrogate and its observation are the same dimensionless quantity."),
          Axis("density", "dimensionless_statistic", domains=("*",),
               why="A null distribution's density is dimensionless.")),
    roles=("null_draw", "generated_candidate", "held_out_confirmation"),
    may_conclude="What the declared ensemble can and cannot resolve, before it is run.",
    may_not_conclude=("that an unresolvable family produced a negative result",
                      "that an uncorrected candidate p-value is a finding"),
    build=_build_null_and_correction))


# ------------------------------------------------------------------ 7. provenance drill-down


def _build_provenance(context: ViewContext) -> Dict[str, Any]:
    artefacts = dict((context.receipt or {}).get("artefacts") or {})
    rows: List[Dict[str, Any]] = []
    table: List[Dict[str, Any]] = []
    for observation in context.spec.observations:
        coverage = context.coverage_row(observation.domain)
        adapter = adapter_for_domain(observation.domain)
        digest = artefacts.get("ACQUIRING/%s" % observation.domain)
        rows.append({
            "domain": observation.domain,
            "source": {"source_id": observation.acquisition.source_id,
                       "source_version": observation.acquisition.source_version,
                       "identity": dict(observation.acquisition.identity),
                       "licence": adapter.declaration.licence,
                       "access": coverage.get("access"),
                       "access_means": coverage.get("access_means")},
            "adapter": {"adapter_id": adapter.adapter_id,
                        "adapter_version": adapter.adapter_version,
                        "definition_sha256": adapter.definition_sha256,
                        "parameters": dict(observation.adapter.parameters)},
            "support": {"support_kind": coverage.get("support_kind"),
                        "native_cadence_seconds": coverage.get("native_cadence_seconds"),
                        "opens_measurement_values": coverage.get("opens_measurement_values")},
            "artefact": {"stage": "ACQUIRING", "sha256": digest,
                         "reason": ("" if digest else
                                    "no acquisition artefact exists at this manifest's address")},
        })
        table.append({"domain": observation.domain,
                      "source_id": observation.acquisition.source_id,
                      "adapter_id": adapter.adapter_id,
                      "adapter_version": adapter.adapter_version,
                      "definition_sha256": adapter.definition_sha256,
                      "artefact_sha256": digest or ""})
    return {"rows": rows,
            "manifest_sha256": manifest_sha256(context.spec),
            "run": {"run_id": (context.receipt or {}).get("run_id"),
                    "state": context.run_state},
            "table": {"columns": ["domain", "source_id", "adapter_id", "adapter_version",
                                  "definition_sha256", "artefact_sha256"], "rows": table}}


register_view(ComparisonView(
    view_id="provenance_drilldown", ordinal=7,
    title="Provenance drill-down",
    question="Which source, adapter, parameters and support produced this?",
    axes=(Axis("provenance field", "categorical", domains=("*",),
               why="Provenance is text and digests; it has no quantitative axis."),),
    roles=("generated_candidate", "refusal"),
    may_conclude="Exactly which immutable identities a cell rests on.",
    may_not_conclude=("that a recorded provenance makes the result correct",
                      "that an adapter version change preserves a result"),
    build=_build_provenance))


# ------------------------------------------------------------------------------ rendering


def render_view(view_id: str, context: ViewContext) -> Dict[str, Any]:
    """One view, with its axes, its legend, its numbers and its boundary in one payload."""
    view = COMPARISON_VIEWS.get(view_id)
    body = view.build(context)
    table = body.pop("table", {"columns": [], "rows": []})
    return {
        "schema": SCHEMA,
        "view_id": view.view_id,
        "ordinal": view.ordinal,
        "title": view.title,
        "question": view.question,
        "mode": context.mode,
        "manifest_sha256": manifest_sha256(context.spec),
        "axes": [axis.describe() for axis in view.axes],
        "legend": [encoding_for(role).describe() for role in view.roles],
        "body": body,
        "table": table,
        "results_exist": context.results_exist,
        "run_state": context.run_state,
        "may_conclude": view.may_conclude,
        "may_not_conclude": list(view.may_not_conclude),
        "mode_forbids": MODE_FORBIDS[context.mode],
        "claim_boundary": ("A view is a reading aid. It shows what was declared and what was "
                           "produced; it admits nothing as evidence."),
    }


def render_all(context: ViewContext) -> Dict[str, Any]:
    return {"schema": SCHEMA, "mode": context.mode,
            "manifest_sha256": manifest_sha256(context.spec),
            "views": [render_view(view.view_id, context) for view in ordered_views()]}


# ------------------------------------------------------------------------ linked selection


def linked_selection(context: ViewContext, *, window: str,
                     domains: Sequence[str] = ()) -> Dict[str, Any]:
    """What one selected match contributes, in every domain's own native terms.

    The selection is a window, and the answer is per-domain native intervals - never a single
    merged interval. A merged one would be the shared-axis mistake in temporal clothing: it would
    show four domains agreeing about an extent that only one of them actually addresses.
    """
    names = [entry.name for entry in context.spec.windows]
    if window not in names:
        raise UnknownNameError("declared window", window, names,
                               study_id=context.spec.study_id)
    chosen = tuple(domains) or context.domains
    unknown = [name for name in chosen if name not in context.domains]
    if unknown:
        raise UnknownNameError("declared domain", unknown[0], list(context.domains),
                               window=window)
    declared = next(entry for entry in context.spec.windows if entry.name == window)
    contributions: List[Dict[str, Any]] = []
    for domain in chosen:
        row = context.coverage_row(domain)
        planned = next((entry for entry in row.get("windows", [])
                        if entry.get("name") == window), {})
        state, reason = _cell_state(row, planned)
        contributions.append({
            "domain": domain,
            "state": state,
            "reason": reason,
            "native_interval": {"start_utc": declared.start_utc.isoformat(),
                                "end_utc": declared.end_utc.isoformat(),
                                "support_kind": row.get("support_kind"),
                                "native_cadence_seconds": row.get("native_cadence_seconds"),
                                "coverage_exact": bool(planned.get("coverage_exact", False))},
            "contributes": state in ("COVERED", "SPARSE"),
        })
    return {
        "schema": SCHEMA,
        "window": window,
        "manifest_sha256": manifest_sha256(context.spec),
        "contributions": contributions,
        "merged_interval": None,
        "why_not_merged": ("Each domain keeps its own native interval. One merged extent would "
                           "show agreement about coverage only one domain addresses."),
        "claim_boundary": ("Highlighted support is where a comparison could be computed, not "
                           "where one was found."),
    }


# ------------------------------------------------------------------------------- contract


def describe_views() -> Dict[str, Any]:
    """The served contract, generated from the registries rather than maintained beside them."""
    return {
        "schema": SCHEMA,
        "views": [view.describe() for view in ordered_views()],
        "encodings": [encoding.describe() for encoding in ordered_encodings()],
        "axis_kinds": list(AXIS_KINDS),
        "shared_axis_kinds": list(SHARED_AXIS_KINDS),
        "coverage_cells": list(COVERAGE_CELLS),
        "readings": {"declarable_by_mode": {mode: list(names)
                                            for mode, names in MODE_RELATIONSHIPS.items()},
                     "renderable_by_mode": {
                         mode: [name for name in names
                                if name not in REQUIRES_EXTERNAL_DESIGN]
                         for mode, names in MODE_RELATIONSHIPS.items()},
                     "never_admissible": list(NEVER_ADMISSIBLE_READINGS),
                     "requires_external_design": list(REQUIRES_EXTERNAL_DESIGN),
                     "why_two_lists": ("What a manifest may declare and test is not what a "
                                       "picture may assert. Causality is declarable by a study "
                                       "holding an external design and is drawable by none of "
                                       "these views.")},
        "mode_forbids": dict(MODE_FORBIDS),
        "refusals": {
            "shared_native_magnitude_axis": (
                "An axis of native magnitude carrying more than one domain raises "
                "MagnitudeEquivalenceError when the view is constructed, so it cannot be drawn."),
            "absent_is_not_zero": (
                "A coverage cell is a named state. There is no code path that renders absent "
                "support as the number zero."),
            "mark_without_provenance": (
                "A mark requires an artefact digest or an explicit reason it has none."),
        },
        "claim_boundary": ("These views make a declared plan and an executed run inspectable. "
                           "None of them admits evidence or promotes a finding."),
    }


__all__ = ["AXIS_KINDS", "Axis", "COMPARISON_VIEWS", "COVERAGE_CELLS", "ComparisonView",
           "ENCODINGS", "Encoding", "ForbiddenReadingError", "MagnitudeEquivalenceError",
           "Mark", "NEVER_ADMISSIBLE_READINGS", "NOT_YET_MEASURED", "READINGS", "SCHEMA",
           "REQUIRES_EXTERNAL_DESIGN", "SHARED_AXIS_KINDS", "ViewContext", "assert_reading_admissible", "describe_views",
           "encoding_for", "linked_selection", "ordered_encodings", "ordered_views",
           "register_encoding", "register_view", "render_all", "render_view"]
