"""The channel-series contract: what the inference layer actually consumes (TG0.1).

**Why this module exists.** `cross_scale_dependency` and `support_floor` are the accepted
falsification layer — surrogate-referenced, BY-corrected, train/test-replicated. They were
written against `ScaleSignature`, so it has always *looked* as though answering the cross-scale
question required a wavelet decomposition of a 2D atmospheric field.

Reading the code says otherwise. Between them those two functions touch exactly four things:

*   ``to_matrix(measure)`` -> a ``(time, channel)`` float matrix,
*   ``channels`` -> the channel labels,
*   ``channel_records`` -> per-channel validity and representation support, and
*   ``provenance`` -> a dict which may carry a physical ``grid``.

Nothing else. Not the field, not the transform, not the grid, not the variable, not the
pressure level. So the real interface between *structure* and *inference* is:

    a set of labelled scalar time series, a per-channel validity mask, and a per-channel
    minimum-lag basis.

That contract is domain-independent, and this module states it explicitly instead of leaving
it implicit in one class's attribute names. `ScaleSignature` satisfies it; so does anything
else that can produce labelled series on a shared clock.

**This module changes no behaviour.** It names an interface that already existed and adds a
concrete implementation for callers that are not wavelet decompositions. The atmospheric path
computes exactly what it computed before, byte for byte, and a test asserts that a
`ChannelSeries` carrying a signature's own numbers reproduces the signature's own result.

**Two deliberate limits, recorded rather than hidden.**

*   **Record keys are unchanged.** A channel record still uses the keys ``scale``, ``usable``
    and ``support_parent_px``, which are wavelet vocabulary. Renaming them would change the
    ``valid_interiors`` block published in every gate receipt, and the TG0.1 acceptance
    criterion is a bit-identical receipt. Renaming belongs to TG1.5, together with the
    `level_hpa` and `LevelBank` work. The *attribute* names generalise here; the *key* names
    wait for a task that is allowed to change receipts.
*   **A channel with no representation support cannot get a geometric lag floor.**
    `support_floor` requires a positive ``support_parent_px`` and refuses without one. For a
    domain whose representation has no spatial footprint that refusal is *correct* under rule
    R21 — no propagation mechanism, no geometric floor, no precedence claim — but the message
    it currently produces names a wavelet field and will not read well to someone onboarding a
    new domain. Improving it is TG1.3, which introduces lag-admissibility policies. It is left
    alone here so that TG0.1 changes no behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.core.errors import InvalidParameterError, ShapeMismatchError


@runtime_checkable
class ChannelGeometry(Protocol):
    """What a **lag floor** requires: labels, per-channel records, provenance. No data.

    Split out from `ChannelSeriesLike` because `support_floor` genuinely needs less than
    `cross_scale_dependency` does, and that difference is load-bearing rather than cosmetic:
    the pre-acquisition audit in `gate_campaign` checks whether a proposed lag family clears
    the physical floor **before any data exists to measure**. It had been expressing that with
    an ad-hoc `SimpleNamespace`, which is duck-typing standing in for an interface nobody had
    written down. Now it declares the interface it actually satisfies.

    Kept a `Protocol` rather than a base class, for the same reason `DataSource` is one: a
    producer should not have to inherit from us to be analysable.
    """

    @property
    def channels(self) -> Sequence[Any]:
        """Channel labels, in the column order of `to_matrix` where one exists."""

    @property
    def channel_records(self) -> Sequence[Mapping[str, Any]]:
        """Per-channel validity and lag basis, in the same order as `channels`.

        Each record carries at least ``scale`` (the label as a string) and ``usable``. A
        record may carry ``support_parent_px``, the representation's footprint in samples of
        the parent axis, which is what an advective lag floor is computed from. A channel
        without one can still be analysed; it simply cannot be given a geometric floor, and
        rule R21 then forbids a precedence claim rather than inventing a floor.
        """

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Everything needed to interpret the series, including an optional ``grid``."""


@runtime_checkable
class ChannelSeriesLike(ChannelGeometry, Protocol):
    """`ChannelGeometry` plus the data itself: what the inference layer requires."""

    def to_matrix(self, measure: str) -> np.ndarray:
        """The ``(time, channel)`` matrix for one named measure."""


@dataclass(frozen=True)
class ChannelGeometrySpec:
    """Geometry without data, for a lag-admissibility check made before acquisition.

    This exists so a pre-transfer audit can ask "would the proposed lag family clear the
    physical floor?" without pretending to hold a record it has not downloaded yet.
    """

    channels: Sequence[Any]
    support_parent_px: Sequence[Optional[float]]
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.support_parent_px):
            raise ShapeMismatchError(
                "support_parent_px", (len(self.support_parent_px),),
                "one entry per channel", (len(self.channels),),
                fix="A per-channel support must line up with `channels`, or the floor is "
                    "computed from another channel's filter footprint.")

    @property
    def channel_records(self) -> List[Dict[str, Any]]:
        return [{"scale": str(label), "usable": True,
                 **({} if support is None else {"support_parent_px": support})}
                for label, support in zip(self.channels, self.support_parent_px)]


@dataclass(frozen=True)
class ChannelSeries:
    """A concrete, domain-neutral `ChannelSeriesLike` for producers that are not transforms.

    This is what a non-atmospheric adapter builds (TG0.2). It carries one or more named
    measures over the same channels and clock, so a domain that has several natural scalar
    reductions per channel can offer them all and let the analysis choose, exactly as a
    signature offers energy density, energy fraction, participation ratio and Gini.

    `support_parent_px` is **optional and has no default**, for the same reason
    `advection_speed_m_s` has none: a fabricated support would silently set every lag floor in
    every result from a number the reader never chose.
    """

    channels: Sequence[Any]
    times_seconds: np.ndarray
    measures: Mapping[str, np.ndarray]
    usable: Optional[Sequence[bool]] = None
    support_parent_px: Optional[Sequence[Optional[float]]] = None
    unusable_reason: Optional[Mapping[Any, str]] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        n_channels = len(self.channels)
        if n_channels < 2:
            raise InvalidParameterError(
                "channels", list(self.channels),
                "at least two channels. A cross-channel dependency needs an ordered pair, "
                "and a one-channel series has none")
        if len(set(str(label) for label in self.channels)) != n_channels:
            raise InvalidParameterError(
                "channels", list(self.channels),
                "distinct channel labels. Results are keyed by label, so duplicates would "
                "silently overwrite one another in the reported family")

        times = np.asarray(self.times_seconds, dtype=np.float64)
        if times.ndim != 1 or times.size < 2:
            raise InvalidParameterError(
                "times_seconds", getattr(self.times_seconds, "shape", self.times_seconds),
                "a 1-D clock with at least two samples")
        if not np.all(np.isfinite(times)):
            raise InvalidParameterError(
                "times_seconds", "non-finite entries",
                "a finite clock. A lag in frames is only meaningful against a real clock")
        if np.any(np.diff(times) <= 0):
            raise InvalidParameterError(
                "times_seconds", "non-increasing",
                "a strictly increasing clock. Unordered samples make a lag meaningless, and "
                "sorting them here would hide a broken adapter")

        if not self.measures:
            raise InvalidParameterError(
                "measures", {},
                "at least one named measure. A channel series with no values cannot be "
                "analysed and an empty result would look like a clean negative")
        for name, values in self.measures.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (times.size, n_channels):
                raise ShapeMismatchError(
                    "measure %r" % name, array.shape,
                    "the (time, channel) grid", (times.size, n_channels),
                    fix="Every measure must be sampled on the same clock and the same "
                        "channels; otherwise a cross-channel lag is comparing two "
                        "different records.")

        for name, supplied, expected in (("usable", self.usable, n_channels),
                                         ("support_parent_px", self.support_parent_px,
                                          n_channels)):
            if supplied is not None and len(supplied) != expected:
                raise ShapeMismatchError(
                    name, (len(supplied),), "one entry per channel", (expected,),
                    fix="A per-channel mask must line up with `channels`, or the mask "
                        "applies to the wrong channel.")

    # ------------------------------------------------------------------ the contract

    @property
    def channel_records(self) -> List[Dict[str, Any]]:
        """Per-channel validity and lag basis, built from the declared masks.

        A channel with no declared support carries no ``support_parent_px`` key at all,
        rather than a zero or a `None`. `support_floor` then refuses it by name, which is
        rule R21's intended behaviour: a domain with no propagation mechanism gets a refusal,
        not a fabricated floor.
        """
        records: List[Dict[str, Any]] = []
        reasons = self.unusable_reason or {}
        for position, label in enumerate(self.channels):
            usable = True if self.usable is None else bool(self.usable[position])
            record: Dict[str, Any] = {"scale": str(label), "usable": usable}
            if not usable:
                record["reason"] = reasons.get(
                    label, reasons.get(str(label),
                                       "declared unusable by the producing adapter"))
            if self.support_parent_px is not None:
                support = self.support_parent_px[position]
                if support is not None:
                    record["support_parent_px"] = support
            records.append(record)
        return records

    def to_matrix(self, measure: str) -> np.ndarray:
        if measure not in self.measures:
            raise InvalidParameterError(
                "measure", measure,
                "one of the declared measures %s" % (sorted(self.measures),))
        return np.asarray(self.measures[measure], dtype=np.float64)

    @property
    def n_times(self) -> int:
        return int(np.asarray(self.times_seconds).size)

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def from_channel_series(series: ChannelSeriesLike) -> Dict[str, Any]:
    """Summarise any `ChannelSeriesLike` for a lineage record, without holding its arrays."""
    records = list(series.channel_records)
    return {
        "n_channels": len(list(series.channels)),
        "channels": [str(label) for label in series.channels],
        "usable": [str(record.get("scale")) for record in records
                   if record.get("usable")],
        "with_declared_support": [str(record.get("scale")) for record in records
                                  if record.get("support_parent_px")],
        "provenance_keys": sorted(str(key) for key in dict(series.provenance)),
    }
