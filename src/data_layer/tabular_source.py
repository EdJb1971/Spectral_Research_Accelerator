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
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.channel_series import ChannelSeries
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, MissingParameterError

#: The measure name a tabular channel carries. Deliberately not one of `ScaleSignature`'s
#: measure names: a raw column is a value, not an energy fraction or a participation ratio,
#: and borrowing those names would imply a normalisation that has not happened.
VALUE_MEASURE = "value"


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
        reader = csv.DictReader(handle, delimiter=delimiter)
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
            "path", path, "a file with at least two rows. A single sample has no clock and "
                          "no lag can be taken along it")

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

    values = np.empty((len(rows), len(channel_columns)), dtype=np.float64)
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

    cadence = _validate_cadence(times, violations, expected_cadence_seconds, path)

    supports = _resolve_supports(channel_columns, support_parent_px)

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
        provenance={
            "adapter": "tabular_source.read_tabular_channels",
            "path_basename": os.path.basename(path),
            "content_sha256": _content_sha256(path),
            "n_rows": len(rows),
            "columns": list(fieldnames),
            "time_column": time_column,
            "time_units": time_units,
            "cadence_seconds": cadence,
            "network_access": "none: this adapter reads a local file and never fetches",
        },
    )

    series = ChannelSeries(
        channels=list(channel_columns),
        times_seconds=times,
        measures={VALUE_MEASURE: values},
        support_parent_px=[supports[name] for name in channel_columns],
        provenance=dict(declaration.provenance),
    )
    return series, declaration


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
