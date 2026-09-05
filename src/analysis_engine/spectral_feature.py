"""Local coefficient maxima as located, labelled features (T4D.1).

**What a feature is here.** A `SpectralFeature` is one local maximum of coefficient magnitude
in one `(time, scale, orientation)` band, above a threshold calibrated on the whole record,
located to sub-pixel precision on the parent grid. It is deliberately *not* a physical object:
it is a place where this transform, at this scale and orientation, found a locally concentrated
amount of energy. Naming it a "feature" is a claim about the coefficient field and nothing more,
which is why every record carries the band it came from and the threshold it cleared.

**Three decisions that are not free, and are made once here.**

*Positions are on the parent grid, not the native band.* Tracking (T4D.2) has to associate
detections across scales, and two detections at different scales are only comparable if they are
expressed in the same frame. The native bands of a decimated family are not that frame.

*And the parent grid has to be earned, not assumed (D88).* Having the parent grid's **shape** is
not the same as being registered to it. An analysis filter anchored at index 0 displaces its
response by half its accumulated support -- half a pixel at level 1, 22.5 pixels at db2 level 4 --
so two levels of one decomposition are displaced by tens of pixels relative to each other. Every
detection here is therefore reported at its index minus the shift the field itself declares, and
when the field cannot declare one the output says so rather than letting the positions pass as
comparable. Nothing that collapses a band to a scalar was ever affected by this, which is why it
survived until something read a coefficient's index as a place.

*The threshold is fitted on the whole record and can be frozen and re-supplied.* A threshold set
per frame would make a quiet frame and a stormy one report the same number of features by
construction, and any subsequent count would be a statement about the normalisation rather than
about the atmosphere. This mirrors `scale_signature`, which fits its thresholds on train and
hands them to test, and it exists so that the same discipline is available to detection.

*Detections inside the boundary-contaminated margin are refused, not reported and flagged.* Rule
R13: a coefficient within the filter's support of the edge is a statement about the crop, and a
tracker fed those detections would report births and deaths that are artefacts of where the crop
was cut. `scale_signature` masks that margin for its statistics; the same margin is masked here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.analysis_engine.scale_signature import (
    DEFAULT_THRESHOLD_SIGMA,
    _interior_halfwidth,
)
from src.core.errors import InvalidParameterError

DETECTION_SCHEMA = "spectral-feature-detection/v1"

# A band of a few hundred pixels a side can in principle contain thousands of local maxima, and
# an unbounded list would make the cost of T4E's pairwise mining quadratic in a number nobody
# declared. The cap is per band and per frame, the strongest are kept, and truncation is always
# recorded -- a silently truncated detection set would make a count of features meaningless.
DEFAULT_MAX_FEATURES_PER_BAND = 256


@dataclass(frozen=True)
class SpectralFeature:
    """One located maximum. Coordinates are parent-grid pixels, `y` down and `x` across."""

    feature_id: str
    time_index: int
    scale: str
    orientation: str
    y: float
    x: float
    strength: float
    threshold: float
    # How many equal-valued pixels the maximum occupies. 1 is a single peak located by
    # curvature; more means a flat top located by its centroid, and therefore located less
    # sharply. A consumer that treats the two alike is overstating the second.
    plateau_pixels: int
    phase: Optional[float]

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


def _as_magnitude(band: torch.Tensor) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Magnitude, and phase where the family actually has one.

    A real band has no phase, and reporting zero for it would be inventing a measurement. The
    caller gets `None` and the record says so.
    """
    if torch.is_complex(band):
        values = band.detach().cpu().numpy()
        return np.abs(values).astype(np.float64), np.angle(values).astype(np.float64)
    return band.detach().cpu().numpy().astype(np.float64), None


def _parent_margin(field, scale: Any, position: int) -> int:
    """Contaminated margin per side in **parent** pixels, from the transform's own filters."""
    record = _interior_halfwidth(field, scale, position)
    return int(record["support_parent_px"]) // 2


def _sub_pixel_offset(low: float, middle: float, high: float) -> float:
    """Vertex of the parabola through three samples, clipped to the sample it belongs to.

    The clip matters: with a flat or non-concave triple the vertex can land arbitrarily far
    away, and a position outside the pixel that produced it is not a refinement of anything.
    """
    denominator = low - 2.0 * middle + high
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    offset = 0.5 * (low - high) / denominator
    if not np.isfinite(offset):
        return 0.0
    return float(np.clip(offset, -0.5, 0.5))


def _local_maxima(magnitude: np.ndarray, threshold: float,
                  margin: int) -> List[Dict[str, Any]]:
    """Regional maxima above `threshold`, excluding the R13 margin.

    *Regional*, not strict, and the difference is not pedantry. A feature centred exactly
    between two samples produces two exactly equal ones, and a strict test rejects both -- so a
    perfectly symmetric feature at a half-pixel position would be invisible, which is precisely
    the position a smoothly advecting feature passes through twice per pixel of travel. A
    tracker built on that detector would report a feature blinking out and back as it drifts.

    So a connected group of equal-valued pixels is one candidate, accepted when every pixel
    adjoining the group is strictly lower. Its position is the group's centroid, which for the
    two-sample tie is exactly the midpoint the feature actually sits at, and for a genuinely
    flat-topped region is the centre of that region rather than an arbitrary corner of it. The
    group's size travels with the detection, because a maximum spread over many pixels is
    located less precisely than one spread over two and a consumer is entitled to know.
    """
    from scipy import ndimage

    height, width = magnitude.shape
    if height - 2 * margin < 3 or width - 2 * margin < 3:
        return []
    # One extra pixel inside the margin, because a maximum is defined by its neighbours and the
    # neighbours of a pixel on the margin's edge lie inside the contaminated region.
    lo_y, hi_y = margin + 1, height - margin - 1
    lo_x, hi_x = margin + 1, width - margin - 1
    if hi_y <= lo_y or hi_x <= lo_x:
        return []

    window = np.ones((3, 3), dtype=bool)
    dilated = ndimage.maximum_filter(magnitude, footprint=window, mode="nearest")
    candidate = (magnitude >= dilated) & (magnitude > threshold)
    # Restrict to the usable interior before labelling, so a group straddling the margin is not
    # accepted on the strength of contaminated pixels.
    usable = np.zeros_like(candidate)
    usable[lo_y:hi_y, lo_x:hi_x] = True
    candidate &= usable
    if not candidate.any():
        return []

    labels, count = ndimage.label(candidate, structure=window)
    if count == 0:
        return []
    records: List[Dict[str, Any]] = []
    objects = ndimage.find_objects(labels)
    for index, window_slice in enumerate(objects, start=1):
        if window_slice is None:
            continue
        member = labels[window_slice] == index
        values = magnitude[window_slice][member]
        value = float(values.max())
        # A plateau on a slope survives the dilation test in its interior but adjoins something
        # higher; the group is a maximum only if nothing touching it is greater.
        grown = ndimage.binary_dilation(labels == index, structure=window)
        halo = grown & ~(labels == index)
        if halo.any() and float(magnitude[halo].max()) >= value:
            continue
        ys, xs = np.nonzero(member)
        offset_y, offset_x = window_slice[0].start, window_slice[1].start
        records.append({
            "y": float(ys.mean()) + offset_y,
            "x": float(xs.mean()) + offset_x,
            "peak_y": int(ys[int(np.argmax(values))]) + offset_y,
            "peak_x": int(xs[int(np.argmax(values))]) + offset_x,
            "strength": value,
            "plateau_pixels": int(member.sum()),
        })
    return records


def _fit_thresholds(field, threshold_sigma: float, margins: Sequence[int]) -> np.ndarray:
    """One threshold per scale: `sigma` times the interior RMS over the whole record.

    Pooled across time and orientation, because a threshold that varied with either would make
    detections from different frames or orientations incomparable -- which is precisely what a
    count of features across a record has to be able to do.
    """
    thresholds = np.zeros(field.n_scales, dtype=np.float64)
    for s_index, scale in enumerate(field.scales):
        margin = margins[s_index]
        gathered: List[np.ndarray] = []
        for t_index in range(field.n_times):
            for orientation in field.orientations:
                magnitude, _ = _as_magnitude(field.band(t_index, scale, orientation))
                height, width = magnitude.shape
                if height - 2 * margin <= 0 or width - 2 * margin <= 0:
                    continue
                interior = (magnitude[margin:height - margin, margin:width - margin]
                            if margin else magnitude)
                gathered.append(interior.reshape(-1))
        if not gathered:
            thresholds[s_index] = np.inf
            continue
        pooled = np.concatenate(gathered)
        thresholds[s_index] = float(threshold_sigma * np.sqrt(np.mean(pooled ** 2)))
    return thresholds


def detect_features(
    field,
    *,
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
    threshold_values: Optional[Sequence[float]] = None,
    interior: bool = True,
    align: bool = True,
    max_features_per_band: int = DEFAULT_MAX_FEATURES_PER_BAND,
) -> Dict[str, Any]:
    """Detect located maxima in every band of a `CoefficientField`.

    `threshold_values` freezes the per-scale thresholds instead of fitting them, so a detection
    pass over held-out frames can inherit the yardstick fitted on the frames it is compared
    with. Supplying them is the leakage-safe path and fitting them is not, which is why the
    provenance records which one happened.

    `align` subtracts the transform's own analysis delay so a reported position is on the parent
    grid rather than at the filter's anchor (D88). Turning it off is for inspecting the raw
    coefficient indices; the result then says so, and cross-scale comparison of those positions
    is meaningless.
    """
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise InvalidParameterError(
            "threshold_sigma", threshold_sigma,
            "a positive multiple of the interior RMS. A non-positive threshold would accept "
            "every pixel and a detection would carry no information")
    if isinstance(max_features_per_band, bool) or int(max_features_per_band) != max_features_per_band \
            or max_features_per_band < 1:
        raise InvalidParameterError(
            "max_features_per_band", max_features_per_band, "a positive integer cap per band")

    margins = [(_parent_margin(field, scale, index + 1) if interior else 0)
               for index, scale in enumerate(field.scales)]
    alignments = [field.parent_alignment(scale) for scale in field.scales]
    shifts = [(float(record["shift_px"])
               if align and record["shift_px"] is not None else 0.0)
              for record in alignments]

    if threshold_values is None:
        thresholds = _fit_thresholds(field, float(threshold_sigma), margins)
        threshold_source = "fitted from this complete coefficient record"
    else:
        thresholds = np.asarray(threshold_values, dtype=np.float64)
        if thresholds.shape != (field.n_scales,) or np.any(~np.isfinite(thresholds)) \
                or np.any(thresholds < 0):
            raise InvalidParameterError(
                "threshold_values", threshold_values,
                "%d finite non-negative per-scale thresholds" % field.n_scales)
        thresholds = thresholds.copy()
        threshold_source = "supplied frozen per-scale thresholds"

    features: List[SpectralFeature] = []
    truncated: List[Dict[str, Any]] = []
    unusable: List[Dict[str, Any]] = []
    for s_index, scale in enumerate(field.scales):
        margin = margins[s_index]
        threshold = float(thresholds[s_index])
        for t_index in range(field.n_times):
            for orientation in field.orientations:
                magnitude, phase = _as_magnitude(field.band(t_index, scale, orientation))
                found = _local_maxima(magnitude, threshold, margin)
                if not found:
                    height, width = magnitude.shape
                    if (height - 2 * margin < 3 or width - 2 * margin < 3) and t_index == 0 \
                            and orientation == field.orientations[0]:
                        unusable.append({
                            "scale": str(scale), "margin_parent_px": margin,
                            "shape": [int(height), int(width)],
                            "reason": ("no interior survives the R13 margin with a neighbourhood "
                                       "around it, so no maximum at this scale is defined on "
                                       "uncontaminated coefficients"),
                        })
                    continue
                found.sort(key=lambda record: -record["strength"])
                if len(found) > max_features_per_band:
                    truncated.append({
                        "time_index": int(t_index), "scale": str(scale),
                        "orientation": str(orientation),
                        "found": int(len(found)), "kept": int(max_features_per_band),
                    })
                    found = found[:max_features_per_band]
                for record in found:
                    y_index, x_index = record["peak_y"], record["peak_x"]
                    if record["plateau_pixels"] == 1:
                        # A single peak carries curvature on both sides, so the parabola is
                        # informative. A plateau does not: its samples are equal by definition
                        # and the centroid is already the best statement about where it sits.
                        y = float(y_index) + _sub_pixel_offset(
                            float(magnitude[y_index - 1, x_index]),
                            float(magnitude[y_index, x_index]),
                            float(magnitude[y_index + 1, x_index]))
                        x = float(x_index) + _sub_pixel_offset(
                            float(magnitude[y_index, x_index - 1]),
                            float(magnitude[y_index, x_index]),
                            float(magnitude[y_index, x_index + 1]))
                    else:
                        y, x = record["y"], record["x"]
                    y -= shifts[s_index]
                    x -= shifts[s_index]
                    features.append(SpectralFeature(
                        feature_id=_feature_id(t_index, scale, orientation, y_index, x_index),
                        time_index=int(t_index), scale=str(scale),
                        orientation=str(orientation),
                        y=y, x=x,
                        strength=float(record["strength"]),
                        threshold=threshold,
                        plateau_pixels=int(record["plateau_pixels"]),
                        phase=(float(phase[y_index, x_index]) if phase is not None else None),
                    ))

    return {
        "schema": DETECTION_SCHEMA,
        "features": [record.to_mapping() for record in features],
        "n_features": len(features),
        "thresholds": {str(scale): float(thresholds[index])
                       for index, scale in enumerate(field.scales)},
        "threshold_sigma": float(threshold_sigma),
        "threshold_source": threshold_source,
        "interior": {
            "applied": bool(interior),
            "margin_parent_px": {str(scale): int(margins[index])
                                 for index, scale in enumerate(field.scales)},
            "basis": ("R13: half the transform's own parent-grid filter support per side. A "
                      "maximum needs its neighbours, so one further pixel is excluded."),
            "unusable_scales": unusable,
        },
        "truncated_bands": truncated,
        "max_features_per_band": int(max_features_per_band),
        "alignment": {
            "applied": bool(align),
            "shift_parent_px": {str(scale): float(shifts[index])
                                for index, scale in enumerate(field.scales)},
            "declarable": {str(scale): alignments[index]["shift_px"] is not None
                           for index, scale in enumerate(field.scales)},
            "exact": {str(scale): bool(alignments[index]["exact"])
                      for index, scale in enumerate(field.scales)},
            "basis": {str(scale): alignments[index]["basis"]
                      for index, scale in enumerate(field.scales)},
            "cross_scale_comparable": bool(
                align and all(record["exact"] for record in alignments)),
            "meaning": ("D88: the analysis delay subtracted from each coefficient index to put "
                        "it on the parent grid. Positions from two scales may only be compared "
                        "when it was applied and is exact at both; otherwise the levels are "
                        "displaced relative to each other by an amount that grows with level"),
        },
        "coordinate_frame": (
            "parent-grid pixels, y down and x across, sub-pixel by parabolic vertex clipped to "
            "the sample that produced it"
            + (", with each scale's declared analysis delay subtracted (D88)" if align
               else ", at the raw coefficient index with no analysis delay removed, so two "
                    "scales' positions are NOT comparable")),
        "claim_boundary": (
            "A feature is a local maximum of this transform's coefficient magnitude above a "
            "threshold fitted on this record. It is not a physical object, not a detection of "
            "any named phenomenon, and its count is comparable across frames only because the "
            "threshold is not refitted per frame."),
        "provenance": {
            "wavelet_family": field.wavelet_family,
            "scales": [str(scale) for scale in field.scales],
            "orientations": [str(orientation) for orientation in field.orientations],
            "n_times": int(field.n_times),
            "complex_family": bool(field.is_complex),
        },
    }


def _feature_id(time_index: int, scale: Any, orientation: Any, y: int, x: int) -> str:
    digest = hashlib.sha256(
        json.dumps([int(time_index), str(scale), str(orientation), int(y), int(x)],
                   sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]
