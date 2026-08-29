"""Cross-domain temporal precedence on an exact physical clock (TG4.3).

The common structural vocabulary makes relationships *testable* across domains; it does not
make their quantities interchangeable.  This module therefore keeps the two source records
separate until it has established an exact shared clock, namespaces every channel by domain,
and carries the original semantics, units, licence, cadence and lag basis into the aligned
record and every confirmatory receipt.

No interpolation is performed.  Interpolation would manufacture values at times one source
did not observe and then let those manufactured values vote in a lagged relationship.  The
alignment is the exact timestamp intersection, and its lineage records how many native
observations were retained and discarded.  Lags are declared in seconds and only converted
to frames after the common cadence is known.

The statistic remains TG4.1's dimensionless correlation, with its circular-shift null,
generate/confirm split and held-out ledger.  A cross-domain result is therefore a temporal
association: it says that one domain's structural series carries information about another's
later structural series.  It never compares their raw magnitudes and it is not a causal
claim (rule R19).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError, ShapeMismatchError
from src.core.precedence import (
    MIN_FRAMES,
    PrecedenceCandidate,
    Record,
    ScaleSeries,
    SweepResult,
    admissible_lags,
    confirm_precedence,
    sweep,
)
from src.core.preregistration import HeldOutLedger, PartitionIdentity, Seal


CROSS_DOMAIN_SCHEMA = "cross-domain-record/v1"
CHANNEL_SEPARATOR = "::"


class ExactClockRequiredError(InvalidParameterError):
    """Raised when two native clocks cannot support a non-invented shared record."""


class PhysicalLagRequiredError(InvalidParameterError):
    """Raised when a frame lag cannot be justified in physical time by both domains."""


@dataclass(frozen=True)
class DomainChannel:
    """One scalar structural series, in its source domain's own vocabulary."""

    label: str
    semantics: str
    units: str
    values: Tuple[float, ...]
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "semantics", str(self.semantics))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not self.label.strip():
            raise InvalidParameterError(
                "DomainChannel.label", self.label,
                "a non-empty channel name that can identify an operand in a frozen member")
        reserved = [token for token in (CHANNEL_SEPARATOR, ">", "@") if token in self.label]
        if reserved:
            raise InvalidParameterError(
                "DomainChannel.label", self.label,
                "a label without the reserved relationship delimiters %s. The frozen label "
                "must be reversible into exactly two operands and one lag" % reserved)
        if not self.semantics.strip():
            raise InvalidParameterError(
                "DomainChannel.semantics", self.semantics,
                "a statement of what the values mean. A canonical series does not erase "
                "the source quantity's meaning (rule R19)")
        if not self.units.strip():
            raise InvalidParameterError(
                "DomainChannel.units", self.units,
                "the original source units. Cross-domain structure is comparable; raw "
                "magnitudes are not, so the units may never be dropped (rule R19)")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise InvalidParameterError(
                "DomainChannel.values", "non-finite or non-vector",
                "one finite scalar value per observation. Missingness is a fact to resolve "
                "before alignment, not a value to average away inside it")

    def describe(self) -> Dict[str, Any]:
        return {"label": self.label, "semantics": self.semantics, "units": self.units,
                "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class DomainTimeSeries:
    """Several structural channels observed on one native domain clock."""

    declaration: DomainDeclaration
    dataset_id: str
    times_seconds: Tuple[float, ...]
    channels: Tuple[DomainChannel, ...]
    aggregation_window_seconds: Optional[float] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "times_seconds", tuple(float(v) for v in self.times_seconds))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not self.dataset_id.strip():
            raise InvalidParameterError(
                "DomainTimeSeries.dataset_id", self.dataset_id,
                "a stable source-record identifier. A domain name identifies a kind of "
                "source, not the record whose observations were used")
        times = np.asarray(self.times_seconds, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise InvalidParameterError(
                "DomainTimeSeries.times_seconds", "invalid clock",
                "a finite one-dimensional clock with at least two observations")
        if np.any(np.diff(times) <= 0):
            raise InvalidParameterError(
                "DomainTimeSeries.times_seconds", "non-increasing clock",
                "strictly increasing timestamps. Sorting here would detach values from the "
                "order supplied by the source adapter")
        if not self.channels:
            raise InvalidParameterError(
                "DomainTimeSeries.channels", [],
                "at least one structural channel. The second operand comes from the other "
                "domain, so a one-channel native record is valid but an empty one is not")
        labels = [channel.label for channel in self.channels]
        if len(set(labels)) != len(labels):
            raise InvalidParameterError(
                "DomainTimeSeries.channels", labels,
                "distinct channel labels within a domain")
        for channel in self.channels:
            if len(channel.values) != times.size:
                raise ShapeMismatchError(
                    "channel %r" % channel.label, (len(channel.values),),
                    "the native clock", (times.size,),
                    fix="Every value must name an actual observation time before two "
                        "domains can be aligned.")

        aggregated = "aggregated_values" in self.declaration.violations
        window = self.aggregation_window_seconds
        if aggregated and (window is None or not math.isfinite(float(window))
                           or float(window) <= 0):
            raise InvalidParameterError(
                "aggregation_window_seconds", window,
                "a positive physical window when the domain declares 'aggregated_values'. "
                "Without it, two overlapping windows can be reported as precedence")
        if not aggregated and window is not None:
            raise InvalidParameterError(
                "aggregation_window_seconds", window,
                "None unless the domain declares 'aggregated_values'. Carrying an averaging "
                "window while claiming instantaneous observations hides a changed sample")

    @property
    def cadence_seconds(self) -> float:
        """The native regular cadence, or an explanatory refusal."""
        if "irregular_sampling" in self.declaration.violations:
            raise ExactClockRequiredError(
                "domain.violations", "irregular_sampling",
                "two regular native clocks for TG4.3 precedence. A frame lag cannot be "
                "converted into physical time for domain %r, so association may be measured "
                "but precedence is refused" % self.declaration.name,
                domain=self.declaration.name)
        differences = np.diff(np.asarray(self.times_seconds, dtype=np.float64))
        cadence = float(differences[0])
        tolerance = max(abs(cadence) * 1e-9, 1e-9)
        if not np.allclose(differences, cadence, rtol=1e-9, atol=tolerance):
            raise ExactClockRequiredError(
                "times_seconds", differences[:8].tolist(),
                "a regular native clock for domain %r, or an association-only analysis. "
                "A lag in frames is not a duration on this clock" % self.declaration.name,
                domain=self.declaration.name)
        return cadence

    def minimum_lag_seconds(self) -> float:
        """The domain's R21 floor, converted before clocks are combined."""
        self.declaration.assert_precedence_admissible()
        frames = self.declaration.minimum_admissible_lag()
        if frames is None:
            raise PhysicalLagRequiredError(
                "domain.lag_policy", self.declaration.lag_policy,
                "a policy whose floor is fixed by the declaration for this cross-domain "
                "series. Domain %r requires geometry or runtime parameters that this scalar "
                "record does not carry; compute and record that physical floor before "
                "asking TG4.3 for precedence" % self.declaration.name,
                domain=self.declaration.name)
        floor = float(frames) * self.cadence_seconds
        if self.aggregation_window_seconds is not None:
            floor = max(floor, float(self.aggregation_window_seconds))
        return floor


def _clock_digest(times: np.ndarray) -> str:
    payload = json.dumps([float(v) for v in times], separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _domain_block(series: DomainTimeSeries) -> Dict[str, Any]:
    return {
        "dataset_id": series.dataset_id,
        "declaration": series.declaration.describe(),
        "native_cadence_seconds": float(series.cadence_seconds),
        "minimum_lag_seconds": float(series.minimum_lag_seconds()),
        "aggregation_window_seconds": (None if series.aggregation_window_seconds is None
                                        else float(series.aggregation_window_seconds)),
        "n_native_observations": len(series.times_seconds),
        "provenance": dict(series.provenance),
    }


def align_exact(first: DomainTimeSeries, second: DomainTimeSeries, *, name: str) -> Record:
    """Align two domains by exact timestamp intersection, never interpolation."""
    if first.declaration.name == second.declaration.name:
        raise InvalidParameterError(
            "domains", [first.declaration.name, second.declaration.name],
            "two distinct domain declarations. Two records from one domain are replication "
            "or multivariate analysis, not evidence that the abstraction crosses a boundary")
    for declaration in (first.declaration, second.declaration):
        reserved = [token for token in (CHANNEL_SEPARATOR, ">", "@")
                    if token in declaration.name]
        if reserved:
            raise InvalidParameterError(
                "DomainDeclaration.name", declaration.name,
                "a cross-domain name without the reserved relationship delimiters %s. "
                "Domain and channel are frozen together as one reversible operand label"
                % reserved)
    if not str(name).strip():
        raise InvalidParameterError("name", name, "a non-empty cross-domain study name")

    # These calls validate both R21 floors before any values are combined.
    first_block, second_block = _domain_block(first), _domain_block(second)
    first_times = np.asarray(first.times_seconds, dtype=np.float64)
    second_times = np.asarray(second.times_seconds, dtype=np.float64)
    common, first_index, second_index = np.intersect1d(
        first_times, second_times, assume_unique=True, return_indices=True)
    if common.size < MIN_FRAMES:
        raise ExactClockRequiredError(
            "shared timestamps", int(common.size),
            "at least %d exact observations shared by domains %r and %r. Interpolation is "
            "not offered because it would let invented values vote in the relationship"
            % (MIN_FRAMES, first.declaration.name, second.declaration.name),
            first_domain=first.declaration.name, second_domain=second.declaration.name)
    common_differences = np.diff(common)
    common_cadence = float(common_differences[0])
    tolerance = max(abs(common_cadence) * 1e-9, 1e-9)
    if not np.allclose(common_differences, common_cadence, rtol=1e-9, atol=tolerance):
        raise ExactClockRequiredError(
            "shared timestamps", common_differences[:8].tolist(),
            "a regular exact intersection. The two native clocks overlap irregularly, so a "
            "frame lag on their intersection is not one physical duration")

    values = []
    metadata: Dict[str, Dict[str, Any]] = {}
    domain_channels: Dict[str, list] = {
        first.declaration.name: [], second.declaration.name: []}
    for native, indices in ((first, first_index), (second, second_index)):
        for channel in native.channels:
            label = "%s%s%s" % (native.declaration.name, CHANNEL_SEPARATOR, channel.label)
            selected = np.asarray(channel.values, dtype=np.float64)[indices]
            values.append(ScaleSeries(label, tuple(float(v) for v in selected)))
            metadata[label] = {
                "domain": native.declaration.name,
                "dataset_id": native.dataset_id,
                "native_label": channel.label,
                "semantics": channel.semantics,
                "units": channel.units,
                "provenance": dict(channel.provenance),
            }
            domain_channels[native.declaration.name].append(label)

    floor = max(float(first_block["minimum_lag_seconds"]),
                float(second_block["minimum_lag_seconds"]))
    provenance = {
        "cross_domain": {
            "schema": CROSS_DOMAIN_SCHEMA,
            "alignment": "exact_timestamp_intersection",
            "interpolation": "none",
            "clock_sha256": _clock_digest(common),
            "clock_start_seconds": float(common[0]),
            "clock_stop_seconds": float(common[-1]),
            "common_cadence_seconds": common_cadence,
            "n_common_observations": int(common.size),
            "physical_lag_floor_seconds": floor,
            "domains": {
                first.declaration.name: first_block,
                second.declaration.name: second_block,
            },
            "channels": metadata,
            "domain_channels": domain_channels,
            "retained_native_observations": {
                first.declaration.name: int(first_index.size),
                second.declaration.name: int(second_index.size),
            },
            "discarded_native_observations": {
                first.declaration.name: int(first_times.size - first_index.size),
                second.declaration.name: int(second_times.size - second_index.size),
            },
            "statistic_units": "dimensionless",
            "claim_boundary": (
                "Cross-domain structural temporal association only. Original magnitudes "
                "retain different semantics and units and are never compared; precedence "
                "does not identify a causal mechanism."),
        }
    }
    return Record(name=str(name), series=tuple(values), cadence_seconds=common_cadence,
                  provenance=provenance)


def cross_domain_metadata(record: Record) -> Dict[str, Any]:
    """The validated cross-domain lineage carried by an aligned record or partition."""
    block = record.provenance.get("cross_domain")
    if not isinstance(block, Mapping) or block.get("schema") != CROSS_DOMAIN_SCHEMA:
        raise InvalidParameterError(
            "record.provenance.cross_domain", block,
            "an aligned record produced by `align_exact`. A plain `Record` does not retain "
            "the domains, units and semantics needed to interpret its operands")
    channels = block.get("channels")
    domains = block.get("domains")
    if not isinstance(channels, Mapping) or not isinstance(domains, Mapping):
        raise InvalidParameterError(
            "record.provenance.cross_domain", block,
            "domain and channel metadata for every operand")
    if set(record.levels) != set(str(label) for label in channels):
        raise InvalidParameterError(
            "record.levels", list(record.levels),
            "exactly the channels named by the cross-domain provenance. A channel added or "
            "removed after alignment changes the family")
    return dict(block)


def cross_domain_pairs(record: Record) -> Tuple[Tuple[str, str], ...]:
    """Every ordered pair crossing the declared boundary, and no within-domain pair."""
    block = cross_domain_metadata(record)
    channels = block["channels"]
    pairs = tuple(
        (driver, driven)
        for driver in record.levels
        for driven in record.levels
        if channels[driver]["domain"] != channels[driven]["domain"])
    if not pairs:
        raise InvalidParameterError(
            "cross_domain_pairs", [],
            "at least one ordered pair whose operands belong to different domains")
    return pairs


def physical_lags(record: Record, lag_seconds: Sequence[float]) -> Tuple[int, ...]:
    """Validate a physical lag family and express it on the exact common clock."""
    block = cross_domain_metadata(record)
    durations = tuple(float(value) for value in lag_seconds)
    if not durations:
        raise PhysicalLagRequiredError(
            "lag_seconds", [],
            "at least one physical duration. Native frame counts are incomparable across "
            "domains with different cadences")
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise PhysicalLagRequiredError(
            "lag_seconds", list(durations), "finite positive physical durations")
    if len(set(durations)) != len(durations):
        raise PhysicalLagRequiredError(
            "lag_seconds", list(durations),
            "distinct durations. A duplicated physical lag is a duplicated family member")
    floor = float(block["physical_lag_floor_seconds"])
    below = [duration for duration in durations if duration < floor]
    if below:
        raise PhysicalLagRequiredError(
            "lag_seconds", below,
            "durations at or above both domains' conservative physical floor of %.9g "
            "seconds. They are refused rather than silently removed from the declared family"
            % floor)
    cadence = float(block["common_cadence_seconds"])
    frames = []
    for duration in durations:
        quotient = duration / cadence
        nearest = int(round(quotient))
        if nearest < 1 or not math.isclose(quotient, nearest, rel_tol=1e-9, abs_tol=1e-9):
            raise PhysicalLagRequiredError(
                "lag_seconds", duration,
                "an integer multiple of the exact common cadence %.9g seconds. No "
                "interpolation is used to manufacture an in-between lag" % cadence)
        frames.append(nearest)
    admissible = admissible_lags(record, max_lag=max(frames), minimum_lag=min(frames))
    unavailable = [frame for frame in frames if frame not in admissible]
    if unavailable:
        raise PhysicalLagRequiredError(
            "lag_seconds", [frame * cadence for frame in unavailable],
            "durations leaving enough effective pairs on this record")
    return tuple(frames)


@dataclass(frozen=True)
class CrossDomainSweep:
    """A TG4.1 sweep whose family is expressed in domain and physical-time terms."""

    result: SweepResult
    lag_seconds: Tuple[float, ...]
    lag_frames: Tuple[int, ...]
    pairs: Tuple[Tuple[str, str], ...]
    metadata: Mapping[str, Any]

    def describe(self) -> Dict[str, Any]:
        return {
            "sweep": self.result.describe(),
            "lag_seconds": list(self.lag_seconds),
            "lag_frames": list(self.lag_frames),
            "cross_domain_pairs": ["%s>%s" % pair for pair in self.pairs],
            "domains": sorted(self.metadata["domains"]),
            "alignment": self.metadata["alignment"],
            "interpolation": self.metadata["interpolation"],
            "physical_lag_floor_seconds": self.metadata["physical_lag_floor_seconds"],
            "claim_boundary": self.metadata["claim_boundary"],
        }


def sweep_cross_domain(record: Record, *, lag_seconds: Sequence[float], n_surrogates: int,
                       alpha: float = 0.05,
                       correction: str = "benjamini_yekutieli",
                       study_id: str = "") -> CrossDomainSweep:
    """Declare and measure every cross-domain direction at every physical lag."""
    block = cross_domain_metadata(record)
    for name, domain in block["domains"].items():
        violations = domain["declaration"].get("violations", {})
        if "no_natural_cycle" not in violations:
            raise InvalidParameterError(
                "domain.violations", sorted(violations),
                "domain %r to declare 'no_natural_cycle', or a native-clock calendar "
                "removal before TG4.3. Fitting one calendar after alignment would confuse "
                "two different sampling processes" % name,
                domain=name)
    frames = physical_lags(record, lag_seconds)
    pairs = cross_domain_pairs(record)
    result = sweep(record, lags=frames, pairs=pairs, n_surrogates=n_surrogates,
                   alpha=alpha, correction=correction, study_id=study_id)
    return CrossDomainSweep(
        result=result, lag_seconds=tuple(float(v) for v in lag_seconds),
        lag_frames=frames, pairs=pairs, metadata=block)


def confirm_cross_domain(seal: Seal, chosen: Sequence[PrecedenceCandidate], *,
                         record: Record, held_out: PartitionIdentity,
                         ledger: HeldOutLedger, opened_at: str, seed: int,
                         n_surrogates: Optional[int] = None) -> Dict[str, Any]:
    """Confirm a frozen cross-domain family and restore its semantic boundary to the receipt."""
    block = cross_domain_metadata(record)
    allowed = set(cross_domain_pairs(record))
    outside = [candidate.label for candidate in chosen
               if (candidate.driver, candidate.driven) not in allowed]
    if outside:
        raise InvalidParameterError(
            "chosen", outside,
            "members whose operands cross the two declared domains. A within-domain member "
            "cannot be laundered through a cross-domain seal")
    receipt = confirm_precedence(
        seal, chosen, record=record, held_out=held_out, ledger=ledger,
        opened_at=opened_at, seed=seed, n_surrogates=n_surrogates)
    channels = block["channels"]
    by_label = {candidate.label: candidate for candidate in chosen}
    for evidence in receipt["relationships"]:
        candidate = by_label[evidence["label"]]
        evidence["lag_seconds"] = float(candidate.lag * record.cadence_seconds)
        evidence["statistic_units"] = "dimensionless"
        evidence["driver_operand"] = dict(channels[candidate.driver])
        evidence["driven_operand"] = dict(channels[candidate.driven])
        evidence["claim_boundary"] = (
            "A held-out, surrogate-referenced temporal association between structural "
            "series. It compares no raw magnitude and establishes no causal mechanism.")
    receipt["cross_domain"] = {
        "schema": block["schema"],
        "domains": block["domains"],
        "alignment": block["alignment"],
        "interpolation": block["interpolation"],
        "clock_sha256": block["clock_sha256"],
        "common_cadence_seconds": block["common_cadence_seconds"],
        "physical_lag_floor_seconds": block["physical_lag_floor_seconds"],
        "statistic_units": "dimensionless",
        "claim_boundary": block["claim_boundary"],
    }
    receipt["claim_boundary"] = block["claim_boundary"]
    return receipt


__all__ = [
    "CROSS_DOMAIN_SCHEMA", "CHANNEL_SEPARATOR",
    "ExactClockRequiredError", "PhysicalLagRequiredError",
    "DomainChannel", "DomainTimeSeries", "CrossDomainSweep",
    "align_exact", "cross_domain_metadata", "cross_domain_pairs", "physical_lags",
    "sweep_cross_domain", "confirm_cross_domain",
]
