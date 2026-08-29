"""`ScaleSignature`: what each scale is doing, per frame (roadmap T4C.1, rule R3).

A `CoefficientField` is `(time, scale, orientation, y, x)` and far too large to reason about,
store, or test against a surrogate ensemble two hundred times over. The signature is its small
reduction: for every `(time, scale)`, four numbers saying **how much** structure that scale
carries and **how concentrated** it is. Everything downstream in Phase 4C - the cross-scale
lagged dependency, the scale-population exponent, the gate itself - runs on this matrix rather
than on the coefficients, which is what makes the whole phase affordable on a laptop.

**Rule R3 is the reason three of the four measures exist.** It names two traps:

*   **Threshold sensitivity.** "How many significant coefficients are there at scale 3" has no
    answer independent of the threshold, because each scale has its own coefficient variance.
    `threshold_fraction` is computed and reported - it is legible, and a reader will ask for
    it - but it is never primary, and `test_scale_signature.py` measures how far it moves when
    the threshold changes while the other three do not move at all.
*   **Geometric bias.** In a decimated pyramid the *number of available coefficients* falls as
    `s**-2`, so any raw per-scale total is reporting the pyramid's geometry. Every measure here
    is either a per-coefficient density, a fraction, or a population-normalised concentration.

The three threshold-free measures answer different questions and are kept separate because
they disagree in informative ways:

*   `energy_fraction` - how the field's energy divides between scales. Says nothing about
    whether that energy sits in one structure or is spread evenly.
*   `participation_ratio` - `(sum w)**2 / (n * sum w**2)` on the coefficient energies `w`.
    One dominant coefficient gives `1/n`; a perfectly even scale gives `1`. This is the
    inverse participation ratio of localisation physics, normalised by the population so that
    scales with different coefficient counts are comparable.
*   `gini` - the inequality of the same energy distribution, `0` even and `1` maximally
    concentrated. It responds to the whole distribution's shape rather than to its second
    moment, so a scale with a long tail of medium coefficients separates from one with a
    single spike, which the participation ratio alone would not distinguish.

**Everything is computed on native coefficients, inside the valid interior.** Two corrections
that are easy to get wrong and silent when wrong:

1.  A resampled (DTCWT) band on the parent grid repeats every native coefficient `4**j` times.
    Energy *fractions* survive that unharmed - the factor cancels in the ratio - but the
    participation ratio does not: replicating every coefficient `r` times multiplies it by
    exactly `r`, so a signature taken from the aligned view would report the coarse scales as
    far more evenly spread than they are. Both facts are asserted as tests.
2.  Coefficients within one filter support of the domain edge are contaminated by the padding
    (rule R13), and they look like strong, localised, oriented features - precisely what a
    concentration measure is built to notice. The interior mask is per-scale and computed from
    the transform's own filter lengths, and a scale with no valid interior left returns NaN
    with a named reason rather than a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from src.core.errors import InvalidParameterError
from src.core.level_axis import require_pressure, units_of
from src.analysis_engine.power_law import compare_exponent_to_null, loglog_fit

MEASURES = ("energy_density", "energy_fraction", "participation_ratio", "gini",
            "threshold_fraction")

THRESHOLD_FREE = ("energy_density", "energy_fraction", "participation_ratio", "gini")

DEFAULT_THRESHOLD_SIGMA = 3.0

# Below this many surviving coefficients a scale is reported as *thin*: the measures are
# computed, because they are well defined, but a participation ratio over a handful of
# samples is dominated by sampling noise. The figure is the point at which the standard
# error of a variance estimate falls under about 20 percent, not a round number chosen for
# its looks (rule R16).
MIN_INTERIOR_COEFFICIENTS = 64


# ---------------------------------------------------------------- concentration measures

def participation_ratio(weights: Sequence[float]) -> float:
    """`(sum w)**2 / (n * sum w**2)`, in `(0, 1]`, for non-negative `w`.

    Normalised by the population `n` deliberately (rule R3): the unnormalised ratio is an
    *effective number of participating coefficients* and so scales with how many there are,
    which would make a fine scale look more spread out than a coarse one purely because a
    decimated pyramid gives it sixteen times as many coefficients.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return float("nan")
    if np.any(w < 0):
        raise InvalidParameterError(
            "weights", "negative values",
            "non-negative weights. The participation ratio is defined on an energy-like "
            "quantity; passing signed coefficients would let cancellation produce a ratio "
            "above one, which has no interpretation")
    total = float(w.sum())
    total_sq = float((w ** 2).sum())
    if total <= 0 or total_sq <= 0:
        return float("nan")
    return (total ** 2) / (w.size * total_sq)


def gini(weights: Sequence[float]) -> float:
    """Gini coefficient of a non-negative distribution: `0` even, `1` maximally concentrated.

    Computed from the sorted values as `2 * sum(i * w_i) / (n * sum w) - (n + 1) / n`, which
    is the exact discrete definition rather than an approximation from a Lorenz curve.

    Invariant to duplicating every value, which is why it is safe to compare between a
    decimated and an undecimated family, and invariant to a global rescale, which is why it
    survives a change of units.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return float("nan")
    if np.any(w < 0):
        raise InvalidParameterError(
            "weights", "negative values",
            "non-negative weights; the Gini coefficient of a signed quantity is not defined")
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    ordered = np.sort(w)
    n = ordered.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * float((index * ordered).sum())) / (n * total) - (n + 1.0) / n)


# ---------------------------------------------------------------- the signature itself

@dataclass(frozen=True)
class ScaleSignature:
    """Per `(time, scale)` measures, plus everything needed to interpret them.

    Small by construction: a 200-frame five-scale signature is 4,000 floats, so it fits in a
    lineage row and can be recomputed on two hundred surrogates without a second thought.
    """

    times_seconds: np.ndarray
    scales: List[Any]
    wavelet_family: str
    energy_density: np.ndarray          # (T, S) energy per available coefficient
    energy_fraction: np.ndarray         # (T, S) normalised across scales, sums to 1
    participation_ratio: np.ndarray     # (T, S) in (0, 1]
    gini: np.ndarray                    # (T, S) in [0, 1)
    threshold_fraction: np.ndarray      # (T, S) fraction of coefficients above threshold
    threshold_sigma: float
    threshold_values: np.ndarray        # (S,) the absolute thresholds actually applied
    available: np.ndarray               # (S,) coefficients used, after the interior mask
    interior: List[Dict[str, Any]]
    source_variable: Optional[str] = None
    # TG1.5: the vertical level this signature was computed at, and the registered
    # coordinate it is a value of. This was `level_hpa` - a unit baked into an
    # attribute name on an otherwise generic record. Not to be confused with the
    # `level` key inside an `interior` record, which is a wavelet decomposition level;
    # that one is transform vocabulary and correctly stays where it is.
    level: Optional[float] = None
    level_axis: Optional[str] = None
    warnings: List[str] = dataclass_field(default_factory=list)
    provenance: Dict[str, Any] = dataclass_field(default_factory=dict)

    # -- the domain-neutral channel-series contract (TG0.1) ----------------------------
    #
    # `ScaleSignature` satisfies `src.core.channel_series.ChannelSeriesLike` without being
    # restructured to do so. These two aliases exist so the inference layer can speak the
    # general vocabulary - a scale is one kind of channel, a valid interior is one kind of
    # validity record - while this class keeps the wavelet names that are correct for it.
    #
    # The record *keys* inside `interior` are deliberately not renamed here: they are
    # published in the `valid_interiors` block of every gate receipt, and TG0.1's acceptance
    # criterion is a bit-identical receipt. That rename is TG1.5.

    @property
    def level_units(self) -> Optional[str]:
        """Units of `level`, or `None` when no vertical coordinate was declared."""
        return units_of(self.level_axis)

    @property
    def level_hpa(self) -> Optional[float]:
        """The level, read as hectopascals - and a refusal when it is not one.

        Kept because "the 850-hPa signature" is how the atmospheric line talks, and removing
        the spelling would make every such reader carry the axis check itself. It refuses a
        non-pressure axis rather than returning `None`, because a `None` here would read as
        "this signature has no level" about a signature that has one.
        """
        if self.level is None:
            return None
        require_pressure(self.level_axis, "ScaleSignature.level_hpa")
        return self.level

    @property
    def channels(self):
        """The channel labels. For a signature these are the scales."""
        return self.scales

    @property
    def channel_records(self):
        """Per-channel validity and lag basis. For a signature these are the interiors."""
        return self.interior

    @property
    def n_times(self) -> int:
        return int(self.times_seconds.size)

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    def to_matrix(self, measure: str = "energy_fraction") -> np.ndarray:
        """The `A_t(s)` matrix rule R4's cross-scale analysis consumes."""
        if measure not in MEASURES:
            raise InvalidParameterError(
                "measure", measure,
                "one of %s. `threshold_fraction` is available but is never the primary "
                "measure: it moves with the threshold, which is the trap rule R3 names."
                % (list(MEASURES),))
        return np.asarray(getattr(self, measure), dtype=np.float64)

    def dominant_scale(self) -> List[Any]:
        """The scale carrying the largest energy fraction, per frame.

        `None` for a frame whose measures are all NaN - a frame with no energy has no
        dominant scale, and returning the first label would invent one.
        """
        out: List[Any] = []
        for row in self.energy_fraction:
            if not np.any(np.isfinite(row)):
                out.append(None)
            else:
                out.append(self.scales[int(np.nanargmax(row))])
        return out

    def usable_scales(self) -> List[Any]:
        """Scales with a valid interior; the others carry NaN and a recorded reason."""
        return [self.scales[i] for i, record in enumerate(self.interior)
                if record["usable"]]

    def summary(self) -> Dict[str, Any]:
        """Lineage-safe: the measures' time means, the labels, and the caveats."""

        def mean_row(values: np.ndarray) -> List[float]:
            with np.errstate(invalid="ignore"):
                return [float(v) for v in np.nanmean(values, axis=0)]

        return {
            "type": "ScaleSignature",
            "wavelet_family": self.wavelet_family,
            "source_variable": self.source_variable,
            "level": self.level,
            "level_axis": self.level_axis,
            "level_units": self.level_units,
            "n_times": self.n_times,
            "scales": [str(s) for s in self.scales],
            "measures": list(MEASURES),
            "primary_measures": list(THRESHOLD_FREE),
            "mean_energy_fraction": mean_row(self.energy_fraction),
            "mean_participation_ratio": mean_row(self.participation_ratio),
            "mean_gini": mean_row(self.gini),
            "mean_threshold_fraction": mean_row(self.threshold_fraction),
            "threshold_sigma": self.threshold_sigma,
            "threshold_values": [float(v) for v in self.threshold_values],
            "threshold_caveat": (
                "threshold_fraction is reported, never primary: the count of 'significant' "
                "coefficients at a scale is dominated by the threshold, because each scale "
                "has its own coefficient variance (rule R3). The threshold applied is "
                "recorded above so the number can be reproduced and compared."),
            "available_coefficients": [int(v) for v in self.available],
            "valid_interior": list(self.interior),
            "dominant_scale_per_frame": [None if s is None else str(s)
                                         for s in self.dominant_scale()],
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }

    def as_records(self) -> List[Dict[str, Any]]:
        """One flat row per `(time, scale)`, for the `scale_signatures` table."""
        rows: List[Dict[str, Any]] = []
        for t_index, time_value in enumerate(self.times_seconds):
            for s_index, scale in enumerate(self.scales):
                rows.append({
                    "time_seconds": float(time_value),
                    "scale": str(scale),
                    "energy_density": float(self.energy_density[t_index, s_index]),
                    "energy_fraction": float(self.energy_fraction[t_index, s_index]),
                    "participation_ratio": float(
                        self.participation_ratio[t_index, s_index]),
                    "gini": float(self.gini[t_index, s_index]),
                    "threshold_fraction": float(
                        self.threshold_fraction[t_index, s_index]),
                    "threshold_value": float(self.threshold_values[s_index]),
                    "threshold_sigma": self.threshold_sigma,
                    "available_coefficients": int(self.available[s_index]),
                })
        return rows


# ---------------------------------------------------------------- construction

def _level_of(scale: Any, position: int) -> int:
    try:
        return int(scale)
    except (TypeError, ValueError):
        return position


def _interior_halfwidth(field, scale: Any, position: int) -> Dict[str, Any]:
    """Contaminated margin per side, in **native** samples, from the transform's own filters."""
    level = _level_of(scale, position)
    if field.wavelet_family == "swt":
        from src.transform_engine import stationary as swt_mod
        wavelet = str(field.config.get("wavelet", "haar"))
        halfwidth = swt_mod.valid_interior_halfwidth(wavelet, level)
        support_parent = swt_mod.filter_support(wavelet, level)
        basis = "swt %s, undecimated: halfwidth in parent pixels equals halfwidth in " \
                "native samples" % wavelet
    else:
        from src.transform_engine import dtcwt as dtcwt_mod
        level1 = str(field.config.get("level1", "near_sym_b"))
        qshift = str(field.config.get("qshift", "qshift_b"))
        halfwidth = dtcwt_mod.native_halfwidth(level, level1, qshift)
        support_parent = dtcwt_mod.filter_support(level, level1, qshift)
        basis = ("dtcwt %s / %s, decimated by 2**%d: parent halfwidth %d px"
                 % (level1, qshift, level,
                    dtcwt_mod.valid_interior_halfwidth(level, level1, qshift)))
    return {"scale": str(scale), "level": level, "halfwidth_native": int(halfwidth),
            "support_parent_px": int(support_parent),
            "basis": basis}


def scale_signature(
    field,
    *,
    interior: bool = True,
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
    threshold_values: Optional[Sequence[float]] = None,
) -> ScaleSignature:
    """Reduce a `CoefficientField` to its per-`(time, scale)` signature.

    `interior=True` (the default) excludes the boundary-contaminated margin per scale, per
    rule R13. It can be switched off - a synthetic field on a periodic domain has no
    artificial edges - but the choice is recorded in the provenance either way, because a
    concentration measure computed over a contaminated margin is not comparable with one that
    was not.
    """
    if threshold_sigma <= 0:
        raise InvalidParameterError(
            "threshold_sigma", threshold_sigma,
            "a positive multiple of the per-scale RMS. A non-positive threshold would count "
            "every coefficient and the measure would carry no information")

    n_times, n_scales = field.n_times, field.n_scales
    shape = (n_times, n_scales)
    energy_density = np.full(shape, np.nan)
    energy_total = np.full(shape, np.nan)
    pr = np.full(shape, np.nan)
    gn = np.full(shape, np.nan)
    thresh_fraction = np.full(shape, np.nan)
    if threshold_values is None:
        thresholds = np.full(n_scales, np.nan)
        threshold_source = "fitted from this complete coefficient record"
    else:
        thresholds = np.asarray(threshold_values, dtype=np.float64)
        if thresholds.shape != (n_scales,) or np.any(~np.isfinite(thresholds)) \
                or np.any(thresholds < 0):
            raise InvalidParameterError(
                "threshold_values", threshold_values,
                "%d finite non-negative per-scale thresholds" % n_scales)
        thresholds = thresholds.copy()
        threshold_source = "supplied frozen per-scale thresholds"
    available = np.zeros(n_scales, dtype=np.int64)
    interior_records: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Pass one: gather the interior magnitudes per (time, scale), and the per-scale RMS the
    # threshold is expressed in. The threshold is set from the whole record rather than per
    # frame, so a quiet frame and a stormy one are measured against the same yardstick.
    gathered: List[List[Optional[np.ndarray]]] = []
    for s_index, scale in enumerate(field.scales):
        record = _interior_halfwidth(field, scale, s_index + 1)
        native_h, native_w = _native_shape(field, scale)
        margin = record["halfwidth_native"] if interior else 0
        record["native_shape"] = [int(native_h), int(native_w)]
        record["interior_shape"] = [int(native_h - 2 * margin), int(native_w - 2 * margin)]
        record["applied"] = bool(interior)
        record["usable"] = bool(min(record["interior_shape"]) > 0)
        record["n_interior"] = (int(record["interior_shape"][0] * record["interior_shape"][1])
                                if record["usable"] else 0)
        record["thin"] = bool(0 < record["n_interior"] < MIN_INTERIOR_COEFFICIENTS)
        interior_records.append(record)
        if record["thin"]:
            warnings.append(
                "scale %s is statistically thin: %d native coefficients survive the "
                "interior mask (%dx%d of %dx%d). The measures are computed because they are "
                "defined, but a concentration measure over that few samples is mostly "
                "sampling noise. Rule R13's practical minimum is stated for a 14-tap "
                "undecimated filter; this transform's filters are longer, so this crop is "
                "too small for this scale."
                % (scale, record["n_interior"], record["interior_shape"][0],
                   record["interior_shape"][1], native_h, native_w))
        if not record["usable"]:
            warnings.append(
                "scale %s has no valid interior: a %dx%d native band loses %d samples per "
                "side to boundary contamination, leaving nothing. Rule R13 - the crop is too "
                "small for this scale, and every measure at it is NaN rather than a number "
                "computed on contaminated coefficients."
                % (scale, native_h, native_w, margin))

    for s_index, scale in enumerate(field.scales):
        record = interior_records[s_index]
        per_time: List[Optional[np.ndarray]] = []
        for t_index in range(n_times):
            if not record["usable"]:
                per_time.append(None)
                continue
            margin = record["halfwidth_native"] if interior else 0
            parts = []
            for orientation in field.orientations:
                band = field.native_band(t_index, scale, orientation)
                if margin:
                    band = band[margin:band.shape[0] - margin,
                                margin:band.shape[1] - margin]
                magnitude = torch.abs(band) if torch.is_complex(band) else band
                parts.append(magnitude.to(torch.float64).reshape(-1).numpy())
            per_time.append(np.concatenate(parts))
        gathered.append(per_time)
        usable = [values for values in per_time if values is not None]
        if usable and threshold_values is None:
            available[s_index] = int(usable[0].size)
            stacked = np.concatenate(usable)
            rms = float(np.sqrt(np.mean(stacked ** 2)))
            thresholds[s_index] = threshold_sigma * rms
        elif usable:
            available[s_index] = int(usable[0].size)

    # Pass two: the measures.
    for s_index, scale in enumerate(field.scales):
        for t_index in range(n_times):
            values = gathered[s_index][t_index]
            if values is None or values.size == 0:
                continue
            weights = values ** 2
            total = float(weights.sum())
            energy_total[t_index, s_index] = total
            energy_density[t_index, s_index] = total / values.size
            if total <= 0:
                continue
            pr[t_index, s_index] = participation_ratio(weights)
            gn[t_index, s_index] = gini(weights)
            if np.isfinite(thresholds[s_index]):
                thresh_fraction[t_index, s_index] = float(
                    np.count_nonzero(values > thresholds[s_index])) / values.size

    with np.errstate(invalid="ignore"):
        row_totals = np.nansum(energy_density, axis=1, keepdims=True)
    fractions = np.full(shape, np.nan)
    nonzero = (row_totals[:, 0] > 0) & np.isfinite(row_totals[:, 0])
    fractions[nonzero] = energy_density[nonzero] / row_totals[nonzero]
    if not np.all(nonzero):
        warnings.append(
            "%d frame(s) carry no energy at any usable scale, so they have no signature and "
            "their measures are NaN. A zero frame given an energy fraction of zero would be "
            "indistinguishable downstream from a frame whose energy sits elsewhere."
            % int(np.count_nonzero(~nonzero)))

    return ScaleSignature(
        times_seconds=np.asarray(field.times, dtype=np.float64),
        scales=list(field.scales),
        wavelet_family=field.wavelet_family,
        energy_density=energy_density,
        energy_fraction=fractions,
        participation_ratio=pr,
        gini=gn,
        threshold_fraction=thresh_fraction,
        threshold_sigma=float(threshold_sigma),
        threshold_values=thresholds,
        available=available,
        interior=interior_records,
        source_variable=field.source_variable,
        level=field.level,
        level_axis=field.level_axis,
        warnings=warnings,
        provenance={
            "computed_from": "native coefficients",
            "threshold_source": threshold_source,
            "interior_mask": bool(interior),
            "interior_rule": "R13",
            "resampled_to_parent": bool(field.resampled_to_parent),
            "config": dict(field.config),
            "grid": field.grid.to_provenance(),
            "native_rationale": (
                "measures are taken on the coefficients the transform computed, not on the "
                "parent-grid aligned view. Replication multiplies the participation ratio by "
                "the replication factor, so an aligned-view signature would report a "
                "resampled family as far more evenly spread than it is."),
        },
    )


def stream_scale_signature(
    frame_reader: Callable[[int], Any],
    times: Sequence[Any],
    *,
    family: str = "swt",
    config: Optional[Mapping[str, Any]] = None,
    interior: bool = True,
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
    threshold_values: Optional[Sequence[float]] = None,
    source_provenance: Optional[Mapping[str, Any]] = None,
) -> ScaleSignature:
    """Build the exact record-level signature while retaining one frame at a time.

    The ordinary path first materialises ``(T,S,O,H,W)`` coefficients.  That is appropriate
    for small interactive records and impossible for the multi-year 512x512 T4C.6 gate.  This
    path makes two deterministic passes through a random-access frame reader: the first emits
    every threshold-free statistic and fits the record-level RMS thresholds; the second emits
    only the fractions above those frozen thresholds.  At no point is more than one source
    frame and one frame's coefficient pyramid retained.

    Both passes hash the exact field bytes and must agree.  A mutable cache or reader therefore
    fails rather than combining statistics from two different records.
    """
    from src.physical_core.field import PhysicalField
    from src.transform_engine.coefficient_field import decompose_field

    raw_times = np.asarray(times)
    if raw_times.ndim != 1 or raw_times.size == 0:
        raise InvalidParameterError("times", raw_times.shape,
                                    "a non-empty one-dimensional time coordinate")
    if not callable(frame_reader):
        raise InvalidParameterError("frame_reader", type(frame_reader).__name__,
                                    "a callable accepting one integer frame index")
    config_dict = dict(config or {})

    first_signature: Optional[ScaleSignature] = None
    first_grid = first_coords = first_variable = first_level = first_units = None
    first_level_axis = None
    threshold_free: Dict[str, List[np.ndarray]] = {
        name: [] for name in THRESHOLD_FREE
    }
    input_hashes: List[str] = []
    warnings: List[str] = []
    peak_source_bytes = peak_coefficient_bytes = 0

    def pass_digest(pass_index: int, collect: bool,
                    frozen_thresholds: Optional[np.ndarray] = None) -> List[np.ndarray]:
        nonlocal first_signature, first_grid, first_coords, first_variable, first_level
        nonlocal first_units, first_level_axis, peak_source_bytes, peak_coefficient_bytes
        digest = hashlib.sha256()
        digest.update(np.asarray(raw_times).astype("datetime64[ns]").astype("int64").tobytes()
                      if raw_times.dtype.kind in "MOUS" else
                      np.asarray(raw_times, dtype=np.float64).tobytes())
        fractions: List[np.ndarray] = []
        for index in range(int(raw_times.size)):
            frame = frame_reader(index)
            if not isinstance(frame, PhysicalField):
                raise InvalidParameterError(
                    "frame_reader(%d)" % index, type(frame).__name__, "a PhysicalField")
            if not bool(torch.isfinite(frame.data).all()):
                raise InvalidParameterError(
                    "frame_reader(%d)" % index, "non-finite values",
                    "a complete finite physical field; gate data are never imputed")
            grid_record = frame.grid.to_provenance()
            coord_record = {
                name: np.asarray(value.detach().cpu(), dtype=np.float64).tolist()
                for name, value in sorted(frame.coords.items())
            }
            variable = frame.metadata.get("variable")
            level = frame.metadata.get("level")
            # TG1.5: the coordinate is part of the frame's identity, not decoration.
            # 500 on a pressure axis and 500 on a height axis are different frames, and
            # a record that changed axis partway would pass every other check here.
            level_axis = frame.metadata.get("level_axis")
            units = frame.units
            if first_grid is None:
                first_grid, first_coords = grid_record, coord_record
                first_variable, first_level, first_units = variable, level, units
                first_level_axis = level_axis
            elif (grid_record != first_grid or coord_record != first_coords
                  or variable != first_variable or level != first_level
                  or level_axis != first_level_axis or units != first_units):
                raise InvalidParameterError(
                    "frame_reader(%d)" % index, "field identity drift",
                    "the exact grid, coordinates, variable, level, vertical coordinate "
                    "and units of frame 0")

            values = np.ascontiguousarray(frame.data.detach().cpu().numpy())
            digest.update(str(tuple(values.shape)).encode("utf-8"))
            digest.update(str(values.dtype).encode("utf-8"))
            digest.update(values.tobytes())
            peak_source_bytes = max(peak_source_bytes, int(values.nbytes))

            coefficients = decompose_field(
                frame, family=family, config=config_dict, keep_native=True)
            peak_coefficient_bytes = max(
                peak_coefficient_bytes,
                int(coefficients.data.numel() * coefficients.data.element_size()))
            signature = scale_signature(
                coefficients, interior=interior, threshold_sigma=threshold_sigma,
                threshold_values=frozen_thresholds)
            if first_signature is None:
                first_signature = signature
            elif (signature.scales != first_signature.scales
                  or signature.interior != first_signature.interior
                  or not np.array_equal(signature.available, first_signature.available)):
                raise InvalidParameterError(
                    "frame_reader(%d)" % index, "coefficient geometry drift",
                    "the same scales, valid interiors and coefficient populations as frame 0")
            if collect:
                for name in THRESHOLD_FREE:
                    threshold_free[name].append(np.asarray(getattr(signature, name)[0]))
                warnings.extend(signature.warnings)
            else:
                fractions.append(np.asarray(signature.threshold_fraction[0]))
        input_hashes.append(digest.hexdigest())
        return fractions

    pass_digest(1, collect=True)
    assert first_signature is not None
    energy_density = np.stack(threshold_free["energy_density"], axis=0)
    if threshold_values is None:
        with np.errstate(invalid="ignore"):
            frozen_thresholds = threshold_sigma * np.sqrt(np.nanmean(energy_density, axis=0))
        threshold_fit = "fitted from this streamed record"
    else:
        frozen_thresholds = np.asarray(threshold_values, dtype=np.float64)
        if frozen_thresholds.shape != (first_signature.n_scales,) \
                or np.any(~np.isfinite(frozen_thresholds)) or np.any(frozen_thresholds < 0):
            raise InvalidParameterError(
                "threshold_values", threshold_values,
                "%d finite non-negative train-fitted thresholds"
                % first_signature.n_scales)
        threshold_fit = "supplied from the independent training split"
    if np.any(~np.isfinite(frozen_thresholds)):
        raise InvalidParameterError(
            "thresholds", frozen_thresholds.tolist(),
            "at least one finite-energy valid interior at every declared scale")
    threshold_fraction = np.stack(
        pass_digest(2, collect=False, frozen_thresholds=frozen_thresholds), axis=0)
    if input_hashes[0] != input_hashes[1]:
        raise InvalidParameterError(
            "frame_reader", input_hashes,
            "identical bytes in both streaming passes; the source changed during analysis")

    provenance = dict(first_signature.provenance)
    provenance.update({
        "execution": "two-pass bounded-memory streaming",
        "source_frames_resident": 1,
        "coefficient_frames_resident": 1,
        "peak_source_frame_bytes": peak_source_bytes,
        "peak_aligned_coefficient_frame_bytes": peak_coefficient_bytes,
        "input_stream_sha256": input_hashes[0],
        "input_verified_unchanged_between_passes": True,
        "threshold_fit": threshold_fit,
        "source": dict(source_provenance or {}),
        "source_provenance_sha256": hashlib.sha256(json.dumps(
            dict(source_provenance or {}), sort_keys=True, separators=(",", ":"),
            allow_nan=False, default=str).encode("utf-8")).hexdigest(),
    })
    return ScaleSignature(
        times_seconds=np.asarray(raw_times).astype("datetime64[ns]").astype("int64") / 1e9
        if raw_times.dtype.kind in "MOUS" else np.asarray(raw_times, dtype=np.float64),
        scales=list(first_signature.scales),
        wavelet_family=first_signature.wavelet_family,
        energy_density=energy_density,
        energy_fraction=np.stack(threshold_free["energy_fraction"], axis=0),
        participation_ratio=np.stack(threshold_free["participation_ratio"], axis=0),
        gini=np.stack(threshold_free["gini"], axis=0),
        threshold_fraction=threshold_fraction,
        threshold_sigma=float(threshold_sigma),
        threshold_values=frozen_thresholds,
        available=np.asarray(first_signature.available),
        interior=list(first_signature.interior),
        source_variable=first_signature.source_variable,
        level=first_signature.level,
        level_axis=first_signature.level_axis,
        warnings=sorted(set(warnings)),
        provenance=provenance,
    )


def _native_shape(field, scale: Any):
    if field.resampled_to_parent:
        native = field.native_shapes.get(scale)
        if native is None:
            raise InvalidParameterError(
                "native_shapes", None,
                "a native shape for every scale of a resampled field; without it neither "
                "the available coefficient count nor the interior margin is knowable")
        return int(native[0]), int(native[1])
    height, width = field.shape[-2:]
    return int(height), int(width)


# ---------------------------------------------------------------- scale-population exponent

def scale_energy_exponent(
    signature: ScaleSignature,
    null_exponents: Optional[Sequence[float]] = None,
    measure: str = "energy_density",
) -> Dict[str, Any]:
    """Fit `A(s) ~ s**-alpha` across scales, and refuse to report it without a null (R2).

    `s` is the dyadic scale in samples (`2**level`), and the fitted quantity defaults to the
    per-available-coefficient energy density, which is rule R3's normalisation: a raw
    population count in a decimated pyramid already falls as `s**-2` from geometry alone, so
    an unnormalised fit would recover `alpha = 2 + physics` and attribute the two to the same
    cause.

    The exponent comes back with `reportable: False` unless `null_exponents` is supplied.
    This is R2 as a return value rather than a warning in prose: a fractional Brownian field
    produces a beautiful high-`r_squared` power law and contains no organisation whatsoever,
    so the number that matters is the *difference* from the surrogate ensemble.
    """
    values = getattr(signature, measure, None)
    if values is None:
        raise InvalidParameterError("measure", measure,
                                    "an attribute of ScaleSignature holding a (T, S) array")
    with np.errstate(invalid="ignore"):
        mean_per_scale = np.nanmean(np.asarray(values, dtype=np.float64), axis=0)

    scales_in_samples = np.asarray(
        [float(2 ** _level_of(scale, position))
         for position, scale in enumerate(signature.scales, start=1)],
        dtype=np.float64)

    fit = loglog_fit(scales_in_samples, mean_per_scale)
    alpha = -fit["slope"] if fit["fitted"] else float("nan")
    comparison = compare_exponent_to_null(alpha, null_exponents or [],
                                          label="scale_%s_exponent" % measure)

    return {
        "alpha": float(alpha),
        "alpha_standard_error": fit["slope_standard_error"],
        "r_squared": fit["r_squared"],
        "n_scales_fitted": fit["n_points"],
        "fitted": fit["fitted"],
        "fit_reason": fit["reason"],
        "measure": measure,
        "scales_in_samples": [float(v) for v in scales_in_samples],
        "values": [float(v) for v in mean_per_scale],
        "normalisation": (
            "per available coefficient (rule R3). An unnormalised population count in a "
            "decimated pyramid falls as s**-2 from geometry alone."),
        "versus_null": comparison,
        "reportable": bool(comparison["reportable"]),
    }
