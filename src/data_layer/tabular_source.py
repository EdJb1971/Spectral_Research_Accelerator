"""Tabular channel records: the first non-atmospheric adapter (TG0.2).

**What this is.** A delimited text file with one clock column and one column per channel,
read into a `ChannelSeries` plus the `DomainDeclaration` that says what the source breaks.
It is the cheapest possible second domain: no grid, no transform, no physics, no network — and
therefore the sharpest available test of whether the accepted falsification layer still works
when nothing atmospheric is left.

Most public archives that are not gridded model output arrive in roughly this shape: station
records, instrument logs, index series, counts per interval. So while this adapter is small,
it is not a toy — TG8.1's onboarding contract is essentially this file plus a licence record.

**Boundary, stated plainly.** No public dataset has been ingested. This is offline-accepted
infrastructure, in the same sense that `cds_source` was accepted before any live CDS transfer:
the adapter, its refusals and its declaration are tested against local fixtures, and pointing
it at a real archive is a separate, deliberate act that must also record that archive's
licence. Nothing here reaches the network, and nothing here should be read as evidence about
any real-world domain.

**Two conventions worth knowing.**

*   **The clock is seconds, and it must be regular unless the domain says otherwise.** A lag is
    expressed in frames, and a frame is only a duration if the sampling is regular. An irregular
    clock is not silently resampled — resampling invents values, and inventing values upstream
    of a dependence estimator is how a sampling artefact becomes a finding. It is refused
    unless the domain declares ``irregular_sampling``, and even then cadence-derived quantities
    are unavailable.
*   **Every channel declares its parent-axis footprint.** For an instantaneous reading that is
    genuinely 1. For an aggregate it is the window length, and the domain should also declare
    the ``aggregated_values`` violation. It is never guessed: `support_parent_px` is what the
    lag floor is computed from, and a fabricated one would set every floor in every result.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.channel_series import ChannelSeries, assert_presence_contract
from src.core.domain import KNOWN_VIOLATIONS, AxisSpec, DomainDeclaration, declaration_for
from src.core.errors import InvalidParameterError, MissingParameterError
from src.core.onboarding import DOMAIN_ONBOARDINGS, is_onboarded

#: The measure name a tabular channel carries. Deliberately not one of `ScaleSignature`'s
#: measure names: a raw column is a value, not an energy fraction or a participation ratio,
#: and borrowing those names would imply a normalisation that has not happened.
VALUE_MEASURE = "value"

#: Size ceilings for a single read (TG8.4). A file above either is **refused by name**, never
#: downsampled to fit: this module's whole position is that resampling upstream of a dependence
#: estimator manufactures short-lag structure, and thinning a record to make it fit a response
#: body is the same act performed for a worse reason.
MAX_ROWS = 200_000
MAX_CELLS = 2_000_000

#: The axis roles a flat channel table can actually supply. A domain declaring anything else has
#: declared a record richer than the file in front of it (standard E14).
CHANNEL_TABLE_ROLES = ("time", "category")


def _content_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tabular_channels(
    path: str,
    *,
    domain_name: str,
    description: str,
    licence: str,
    violations: Sequence[str],
    time_column: str,
    time_units: str = "s",
    channel_columns: Optional[Sequence[str]] = None,
    support_parent_px: Optional[Mapping[str, float]] = None,
    lag_policy: str = "none",
    declared_floor_frames: Optional[int] = None,
    declared_floor_basis: Optional[str] = None,
    delimiter: str = ",",
    expected_cadence_seconds: Optional[float] = None,
) -> Tuple[ChannelSeries, DomainDeclaration]:
    """Read a local delimited file into a `ChannelSeries` and its `DomainDeclaration`.

    `violations` and `licence` are required rather than inferred. An adapter that guessed them
    would produce a domain that appears to have been thought about and has not, which rule R17
    exists to prevent.
    """
    if not os.path.isfile(path):
        raise InvalidParameterError(
            "path", path, "an existing local file. This adapter never fetches: pointing a "
                          "domain at a remote archive is a separate, deliberate act that must "
                          "also record that archive's licence and terms")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()

    table = _parse_delimited(text, source_name=os.path.basename(path),
                             time_column=time_column, time_units=time_units,
                             channel_columns=channel_columns, delimiter=delimiter,
                             content_sha256=_content_sha256(path))

    cadence = _validate_cadence(table.times, violations, expected_cadence_seconds,
                                table.source_name)
    supports = _resolve_supports(table.channel_columns, support_parent_px)

    declaration = DomainDeclaration(
        name=domain_name,
        description=description,
        axes=(AxisSpec(name=time_column, role="time", units="s"),
              AxisSpec(name="channel", role="category", ordered=False)),
        licence=licence,
        violations=tuple(violations),
        lag_policy=lag_policy,
        declared_floor_frames=declared_floor_frames,
        declared_floor_basis=declared_floor_basis,
        provenance=dict(table.provenance("tabular_source.read_tabular_channels"),
                        cadence_seconds=cadence),
    )

    series = ChannelSeries(
        channels=list(table.channel_columns),
        times_seconds=table.times,
        measures={VALUE_MEASURE: table.values},
        support_parent_px=[supports[name] for name in table.channel_columns],
        provenance=dict(declaration.provenance),
    )
    assert_presence_contract(series, declaration.violations, declaration.name)
    return series, declaration


# ------------------------------------------------------------------------------ the parser
#
# One parser, two entry points (TG8.4). `read_tabular_channels` builds a declaration from its
# arguments; `read_channels_for_domain` reads against one that was already onboarded. They must
# agree byte for byte about what the file says, so they share this rather than each having their
# own copy of the numeric handling that D48 taught us to be careful about.


@dataclass(frozen=True)
class _ParsedTable:
    """A delimited file, parsed and checked for everything a declaration cannot excuse."""

    source_name: str
    content_sha256: str
    fieldnames: List[str]
    time_column: str
    time_units: str
    channel_columns: List[str]
    times: np.ndarray
    values: np.ndarray

    def provenance(self, adapter: str) -> Dict[str, Any]:
        return {
            "adapter": adapter,
            "path_basename": self.source_name,
            "content_sha256": self.content_sha256,
            "n_rows": int(self.times.size),
            "columns": list(self.fieldnames),
            "time_column": self.time_column,
            "time_units": self.time_units,
            "network_access": "none: this adapter reads a local file and never fetches",
        }


def _parse_delimited(text: str, *, source_name: str, time_column: str, time_units: str,
                     channel_columns: Optional[Sequence[str]], delimiter: str,
                     content_sha256: str) -> _ParsedTable:
    """Columns, clock and values, with every refusal that no declaration can waive.

    Deliberately *not* here: cadence regularity and aggregate support. Those are refused against
    what a domain declares, so they belong with the domain rather than with the file.
    """
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

    if time_column not in fieldnames:
        raise MissingParameterError(
            time_column, "the clock column. The file declares columns %s" % (fieldnames,))
    if channel_columns is None:
        channel_columns = [name for name in fieldnames if name != time_column]
    missing = [name for name in channel_columns if name not in fieldnames]
    if missing:
        raise MissingParameterError(
            ", ".join(missing), "channel columns present in the file, which declares %s"
                                % (fieldnames,))
    if len(rows) < 2:
        raise InvalidParameterError(
            "path", source_name, "a file with at least two rows. A single sample has no clock "
                                 "and no lag can be taken along it")
    _assert_within_caps(source_name, len(rows), len(list(channel_columns)))

    scale = _time_scale(time_units)
    try:
        times = np.asarray([float(row[time_column]) for row in rows],
                           dtype=np.float64) * scale
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            time_column, "non-numeric entries",
            "a numeric clock in %r. Timestamp parsing is deliberately not attempted here: a "
            "guessed format that silently reorders a record would change every lag in every "
            "result" % time_units) from exc

    values = np.empty((len(rows), len(list(channel_columns))), dtype=np.float64)
    for column, name in enumerate(channel_columns):
        try:
            values[:, column] = [float(row[name]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(
                name, "non-numeric entries",
                "numeric channel values. Missing entries are not filled here, because an "
                "imputed value upstream of a dependence estimator becomes a finding") from exc
    if not np.all(np.isfinite(values)):
        bad = [name for column, name in enumerate(channel_columns)
               if not np.all(np.isfinite(values[:, column]))]
        raise InvalidParameterError(
            ", ".join(bad), "non-finite entries",
            "finite values in every channel. A NaN propagates through the joint histogram "
            "and silently shrinks the effective sample size of every test that uses it")

    return _ParsedTable(source_name=source_name, content_sha256=content_sha256,
                        fieldnames=fieldnames, time_column=time_column,
                        time_units=time_units, channel_columns=list(channel_columns),
                        times=times, values=values)


def _assert_within_caps(source_name: str, n_rows: int, n_channels: int) -> None:
    cells = n_rows * n_channels
    if n_rows > MAX_ROWS or cells > MAX_CELLS:
        raise InvalidParameterError(
            "path", source_name,
            "a record of at most %d rows and %d values (%d rows x %d channels = %d here). It "
            "is refused rather than thinned: a record silently reduced to fit is "
            "indistinguishable from one that was always that size, and every lag reported "
            "from it would describe a cadence nobody chose"
            % (MAX_ROWS, MAX_CELLS, n_rows, n_channels, cells))


def clock_facts(times: np.ndarray) -> Dict[str, Any]:
    """What the clock *is*, stated without deciding what any domain may do about it.

    This is the observation half of TG8.4's rule — **detection may create a required
    declaration; it may never satisfy one.** Nothing here consults a domain, and nothing here
    refuses: an irregular clock is reported as irregular, and what that obliges a domain to have
    declared is decided elsewhere, by `required_violations`.
    """
    intervals = np.diff(np.asarray(times, dtype=np.float64))
    increasing = bool(intervals.size > 0 and np.all(intervals > 0))
    regular = bool(increasing and np.allclose(
        intervals, intervals[0], rtol=0.0, atol=1e-9 * max(abs(float(intervals[0])), 1.0)))
    return {
        "strictly_increasing": increasing,
        "regular": regular,
        "interval_seconds_min": float(intervals.min()) if intervals.size else None,
        "interval_seconds_max": float(intervals.max()) if intervals.size else None,
        "cadence_seconds": float(intervals[0]) if regular else None,
    }


def required_violations(facts: Mapping[str, Any]) -> List[str]:
    """The violations a domain must already declare before it may read a file like this.

    An obligation, not an answer. The caller does not get a filled-in declaration; it gets a
    list of things whichever domain it chooses must already have said about itself.
    """
    required: List[str] = []
    if facts.get("strictly_increasing") and not facts.get("regular"):
        required.append("irregular_sampling")
    return required


# ------------------------------------------------------------------------------ helpers

_TIME_SCALES: Dict[str, float] = {"s": 1.0, "seconds": 1.0, "min": 60.0, "minutes": 60.0,
                                  "h": 3600.0, "hours": 3600.0, "d": 86400.0,
                                  "days": 86400.0}


def _time_scale(time_units: str) -> float:
    if time_units not in _TIME_SCALES:
        raise InvalidParameterError(
            "time_units", time_units,
            "one of %s. The clock is converted to seconds so that a lag in frames can be "
            "reported as a duration" % (sorted(_TIME_SCALES),))
    return _TIME_SCALES[time_units]


def _validate_cadence(times: np.ndarray, violations: Sequence[str],
                      expected: Optional[float], path: str) -> Optional[float]:
    """Regular sampling, or an explicit declaration that it is not."""
    if np.any(np.diff(times) <= 0):
        raise InvalidParameterError(
            "path", os.path.basename(path),
            "a strictly increasing clock. Rows are not sorted here: a file whose order was "
            "silently corrected is one whose provenance no longer describes it")
    intervals = np.diff(times)
    regular = bool(np.allclose(intervals, intervals[0], rtol=0.0,
                               atol=1e-9 * max(abs(float(intervals[0])), 1.0)))
    if not regular:
        if "irregular_sampling" not in violations:
            raise InvalidParameterError(
                "path", os.path.basename(path),
                "a regularly sampled clock, or the declared violation 'irregular_sampling'. "
                "Intervals range over [%g, %g] s. The record is not resampled to fix this, "
                "because interpolation upstream of a dependence estimator manufactures "
                "exactly the short-lag structure rule R4 exists to exclude"
                % (float(intervals.min()), float(intervals.max())))
        return None
    cadence = float(intervals[0])
    if expected is not None and not np.isclose(cadence, float(expected), rtol=1e-9,
                                               atol=0.0):
        raise InvalidParameterError(
            "expected_cadence_seconds", expected,
            "the cadence actually present in the file, which is %g s. A record read at a "
            "different cadence than the caller believes turns every lag into a different "
            "duration" % cadence)
    return cadence


def _resolve_supports(channel_columns: Sequence[str],
                      support_parent_px: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """Per-channel parent-axis footprint. Defaults to 1 sample, which is a fact, not a guess.

    An instantaneous reading depends on exactly one sample of the clock. That is the honest
    default and the only one available without knowing the instrument. An aggregate must say
    so, because its footprint is the window and using 1 would understate the shared support
    that rule R4 is about — the same understatement that produced D48.
    """
    supports: Dict[str, float] = {name: 1.0 for name in channel_columns}
    for name, value in dict(support_parent_px or {}).items():
        if name not in supports:
            raise InvalidParameterError(
                "support_parent_px", name,
                "a declared channel. Known channels are %s" % (list(channel_columns),))
        if not isinstance(value, (int, float)) or value <= 0:
            raise InvalidParameterError(
                "support_parent_px[%r]" % name, value,
                "a positive number of parent-axis samples")
        supports[name] = float(value)
    return supports


# --------------------------------------------------------------------- reading for a domain
#
# TG8.4. `read_tabular_channels` above builds a `DomainDeclaration` out of its own arguments,
# which predates TG8.1 and is now the wrong shape: it produces a domain that was never onboarded,
# never checked against a geometry and never registered, so nothing downstream can ask what it
# refuses. Everything below reads a file **against a domain that already exists**.


def unsatisfiable_axes(declaration: DomainDeclaration) -> List[str]:
    """Declared axes a flat channel table cannot supply (standard E14).

    A CSV carries a clock and some columns. A domain declaring latitude, longitude or a pressure
    level has declared a record richer than the file in front of it, and reading the file under
    that domain would quietly drop axes the domain says its results are indexed by. The roles are
    compared rather than the names, because E14's whole position is that a role is declared and
    never inferred from an axis being called `lat`.
    """
    return ["%s (role %s)" % (axis.name, axis.role) for axis in declaration.axes
            if axis.role not in CHANNEL_TABLE_ROLES]


def assert_domain_admits_channel_table(declaration: DomainDeclaration) -> None:
    unsatisfiable = unsatisfiable_axes(declaration)
    if unsatisfiable:
        raise InvalidParameterError(
            "domain", declaration.name,
            "a domain whose declared axes a channel table can supply. Domain %r declares %s, "
            "and a delimited file carries a clock and columns only. Reading it here would drop "
            "declared axes that this domain's results are indexed by, which is a silent change "
            "of what a result means rather than a missing column (standard E14)"
            % (declaration.name, ", ".join(unsatisfiable)))


def _assert_supports_declared(declaration: DomainDeclaration,
                              supports: Mapping[str, float]) -> None:
    """An aggregate channel obliges the domain to declare `aggregated_values` (E15, R17).

    `_resolve_supports` documents this and until TG8.4 nothing enforced it. A `support_parent_px`
    above one says a value depends on more than one sample of the parent axis, which is the
    definition of the violation; declaring the support without declaring the violation gives the
    lag floor the right number while leaving every downstream refusal switched off.
    """
    aggregates = sorted(name for name, value in supports.items() if float(value) > 1.0)
    if aggregates and "aggregated_values" not in tuple(declaration.violations):
        raise InvalidParameterError(
            "domain", declaration.name,
            "the declared violation 'aggregated_values', because %s %s a parent-axis footprint "
            "above one sample. %s Declaring the footprint without declaring the violation sets "
            "the lag floor correctly and leaves every other refusal that depends on it switched "
            "off (standard E15, rule R17)"
            % (", ".join(aggregates), "carries" if len(aggregates) == 1 else "carry",
               KNOWN_VIOLATIONS["aggregated_values"].capitalize() + "."))


def _resolve_declaration(domain: str) -> DomainDeclaration:
    if not is_onboarded(domain):
        known = ", ".join(DOMAIN_ONBOARDINGS.names()) or "none"
        raise InvalidParameterError(
            "domain", domain,
            "a domain onboarded through the contract. Onboarded domains: %s. A domain assembled "
            "from separate registrations was never checked as a whole, so what it refuses is "
            "unknown; see GET /api/v1/findings/onboarding" % known)
    return declaration_for(domain)


def inspect_delimited(text: str, *, source_name: str, delimiter: str = ",",
                      time_column: Optional[str] = None,
                      time_units: str = "s") -> Dict[str, Any]:
    """Describe a delimited file and the obligations its shape creates, choosing nothing.

    The two-call shape is `/import/inspect` then `/import/field` (`src/api/main.py`), and it
    exists for the same reason: *picking on the researcher's behalf* imports something they did
    not choose while everything downstream describes that arbitrary choice. Here the choice is
    the clock column and the domain, and neither is made here.

    Every registered domain is reported with whether it admits this file **and why not** when it
    does not, so the refusal a researcher would hit is visible before they hit it.
    """
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    if not fieldnames:
        raise InvalidParameterError(
            "file", source_name,
            "a delimited file with a header row naming the clock column and one column per "
            "channel")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    candidates = _numeric_increasing_columns(fieldnames, rows)
    chosen = time_column or fieldnames[0]

    report: Dict[str, Any] = {
        "source_name": source_name,
        "content_sha256": digest,
        "delimiter": delimiter,
        "columns": fieldnames,
        "n_rows": len(rows),
        "candidate_time_columns": candidates,
        "time_column": chosen,
        "time_units": time_units,
        "row_cap": MAX_ROWS,
        "cell_cap": MAX_CELLS,
        "aggregate_note": (
            "A channel whose value is a window aggregate must be given a parent-axis footprint "
            "above one when it is read, and the domain must then already declare "
            "'aggregated_values'. Nothing in the file says which channels those are, so this "
            "is stated rather than detected."),
    }

    # No clock is chosen on the caller's behalf. `write_tabular_channels` puts the clock first
    # and that is the convention this format follows, so column zero is *proposed* — but a
    # first column that cannot serve as a clock stops the inspection rather than quietly
    # promoting whichever other column happens to increase. Silently reading `bid` as the clock
    # because `t` ran backwards is precisely the substitution the two-call shape exists to
    # prevent, and it would be invisible in every result downstream.
    if time_column is None and fieldnames[0] not in candidates:
        report.update({
            "readable": False,
            "refused_because": (
                "the file's first column %r cannot serve as a clock: it is non-numeric, or it "
                "is not strictly increasing. No column is substituted for it. Name a clock "
                "column explicitly; the columns that could serve as one are %s"
                % (fieldnames[0], candidates or "none")),
            "channel_columns": [], "clock": None, "required_violations": [], "domains": []})
        return report

    try:
        table = _parse_delimited(text, source_name=source_name, time_column=chosen,
                                 time_units=time_units, channel_columns=None,
                                 delimiter=delimiter, content_sha256=digest)
    except (InvalidParameterError, MissingParameterError) as exc:
        report.update({"readable": False, "refused_because": str(exc),
                       "channel_columns": [], "clock": None, "required_violations": [],
                       "domains": []})
        return report

    facts = clock_facts(table.times)
    required = required_violations(facts)
    report.update({
        "readable": bool(facts["strictly_increasing"]),
        "refused_because": None if facts["strictly_increasing"] else (
            "the clock is not strictly increasing. Rows are not sorted here: a file whose order "
            "was silently corrected is one whose provenance no longer describes it. No domain "
            "declaration can waive this"),
        "channel_columns": list(table.channel_columns),
        "clock": facts,
        "required_violations": required,
        "domains": [_domain_admission(name, facts, required)
                    for name in DOMAIN_ONBOARDINGS.names()],
    })
    return report


def _numeric_increasing_columns(fieldnames: Sequence[str],
                                rows: Sequence[Mapping[str, str]]) -> List[str]:
    """Columns that could serve as a clock: numeric throughout and strictly increasing.

    Offered as candidates the researcher picks from, never as a choice made for them. A file with
    two such columns is a file only its author can disambiguate.
    """
    candidates: List[str] = []
    for name in fieldnames:
        try:
            column = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if column.size >= 2 and np.all(np.isfinite(column)) and np.all(np.diff(column) > 0):
            candidates.append(name)
    return candidates


def _domain_admission(name: str, facts: Mapping[str, Any],
                      required: Sequence[str]) -> Dict[str, Any]:
    """Whether one onboarded domain admits a file with these facts, and why not if it does not."""
    declaration = declaration_for(name)
    refusals: List[str] = []

    unsatisfiable = unsatisfiable_axes(declaration)
    if unsatisfiable:
        refusals.append(
            "declares %s, which a channel table cannot supply (E14)" % ", ".join(unsatisfiable))
    for violation in required:
        if violation not in tuple(declaration.violations):
            refusals.append("does not declare %r, which this file's clock requires: %s"
                            % (violation, KNOWN_VIOLATIONS[violation]))
    if not facts.get("strictly_increasing"):
        refusals.append("the clock is not strictly increasing, which no declaration waives")

    return {"name": name, "admits": not refusals, "refusals": refusals,
            "violations": list(declaration.violations),
            "precedence_admissible": declaration.precedence_admissible}


def read_channels_for_domain(
    text: str,
    *,
    source_name: str,
    domain: str,
    time_column: str,
    time_units: str = "s",
    channel_columns: Optional[Sequence[str]] = None,
    support_parent_px: Optional[Mapping[str, float]] = None,
    delimiter: str = ",",
    expected_cadence_seconds: Optional[float] = None,
) -> Tuple[ChannelSeries, DomainDeclaration]:
    """Read a delimited record **against an onboarded domain**, or refuse by name.

    The declaration is fetched, never built. That is the whole difference from
    `read_tabular_channels`: the domain this record is read under is one that passed TG8.1's
    contract — its geometry and its violations were checked against each other, and what it
    refuses is already published at `GET /api/v1/findings/domains`.

    **Reading a file under a domain does not establish that the file came from it.** It
    establishes that the domain's declaration admits the file's shape. The attribution gap
    recorded for findings applies here unchanged.
    """
    declaration = _resolve_declaration(domain)
    assert_domain_admits_channel_table(declaration)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    table = _parse_delimited(text, source_name=source_name, time_column=time_column,
                             time_units=time_units, channel_columns=channel_columns,
                             delimiter=delimiter, content_sha256=digest)

    cadence = _validate_cadence(table.times, tuple(declaration.violations),
                                expected_cadence_seconds, table.source_name)
    supports = _resolve_supports(table.channel_columns, support_parent_px)
    _assert_supports_declared(declaration, supports)

    provenance = dict(table.provenance("tabular_source.read_channels_for_domain"),
                      cadence_seconds=cadence,
                      domain=declaration.name,
                      onboarding_sha256=DOMAIN_ONBOARDINGS.get(domain).onboarding_sha256,
                      domain_attribution=(
                          "This record was read under domain %r because a reader chose that "
                          "domain. Nothing here establishes that it came from it." % domain))

    series = ChannelSeries(
        channels=list(table.channel_columns),
        times_seconds=table.times,
        measures={VALUE_MEASURE: table.values},
        support_parent_px=[supports[name] for name in table.channel_columns],
        provenance=provenance,
    )
    assert_presence_contract(series, declaration.violations, declaration.name)
    return series, declaration


def write_tabular_channels(path: str, times: Sequence[float],
                           columns: Mapping[str, Sequence[float]],
                           *, time_column: str = "t", delimiter: str = ",") -> str:
    """Write a channel table. Present so fixtures and round-trips share one format."""
    names: List[str] = list(columns)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow([time_column] + names)
        for index, moment in enumerate(times):
            writer.writerow([repr(float(moment))]
                            + [repr(float(columns[name][index])) for name in names])
    return path
