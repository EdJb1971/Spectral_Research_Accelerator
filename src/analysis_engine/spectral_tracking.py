"""T4D.2: the wavelet detections of T4D.1, spoken in the language the tracker already reads.

**Why this is a bridge and not a tracker.** Frame-to-frame association already exists, in
`src.core.tracking`, and it is not a sketch: the association radius is *derived* from the same
alpha the search was calibrated at and the frame's own density rather than chosen; the gates on
scale and orientation are declared rates that are refused outright when the features cannot
measure them; an associator's answer is checked against the gates that admitted it; and the
clock is every frame that was searched, so a frame in which nothing was found ends a track
instead of being silently bridged. Writing a second tracker here would mean writing a second set
of those refusals, and the second set is the one that would be weaker.

What was genuinely missing is the translation. `detect_features` speaks in bands: a maximum in
one `(time, scale, orientation)` slice of a `CoefficientField`, positioned in parent-grid pixels.
`track` speaks in `src.core.feature.SpectralFeature`: a structure with a magnitude, a location on
declared axes, a clock with units, an optional physical scale and an optional orientation that
knows how it wraps. This module is that translation, and every lossy step in it is a decision:

*Bands are pooled into one frame, and one structure is still several tracks.* Pooling is what
lets a scale gate mean anything: the gate is on the ratio between an observation's scale and its
track's, and there is no ratio to gate inside a single band. But pooling does not turn a growing
structure into one track that migrates upward, and measurement says why. The bank is redundant,
so a structure whose width doubles does not leave one level for the next -- it excites both at
once, and the coarse level's detections appear *alongside* the fine level's rather than instead
of them. A growing vortex therefore reads as parallel tracks, one per excited band, and its
scale evolution is in the population of tracks rather than in any one track's `scale_velocity`.
Reporting it the other way would require claiming a merge, and this tracker does not claim one.

*A detail coefficient peaks at a structure's flank, not at its centre.* A real detail wavelet is
a derivative-like filter: a symmetric blob has zero detail response exactly at its middle and
two maxima on either side, about one analysing width out along the axis the band high-passes.
So a track from an `LH` band follows a point offset from the vortex by its own scale in `x`, and
locates it precisely in `y`. That is not an error to be corrected -- it is what the coefficient
measures -- but it does mean a track's position is a statement about a flank, and the distance
from flank to centre grows as the structure does. Only the transverse coordinate, and any
*difference* of positions, are statements about the structure itself.

*Scale is the dyadic octave, not the filter support.* Consecutive levels of a dyadic transform
stand in a ratio of exactly two by construction, and the ratio is the only thing the scale gate
or `scale_velocity` ever uses. The measured filter support is reported alongside it in
provenance, because it is the honest absolute number and it is *not* exactly dyadic at low
levels -- edge support inflates it -- so using it as the scale would put a transform artefact
into every doubling time.

*A separable band label is not an angle.* SWT's LH/HL/HH are the transform's own admission that
it cannot tell the two diagonal signs apart; assigning them degrees would manufacture an
orientation, and an orientation gate would then appear in the receipt while testing nothing.
Those features carry `orientation=None` and the tracker refuses a turn gate on them by itself.
A DTCWT band does carry a feature angle, under a declared convention, and it is passed through.

*Significance is not established here.* A maximum clearing `sigma` times the record's RMS is not
a test against a null, and calling it one would let a threshold crossing be counted as a finding.
The threshold and its provenance travel with every feature; the significance field says nothing,
which is the true statement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.analysis_engine.spectral_feature import detect_features
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.feature import (
    FeatureLocation, FeatureSet, Orientation, Quantity, SpectralFeature,
)
from src.core.tracking import NO_BOUNDS, MotionBounds, SearchVolume, track

TRACKING_SCHEMA = "spectral-feature-tracking/v1"

#: The axes a parent-grid detection lives on. Named as the benchmarks and `src.core.tracking`
#: name them, in cells, because a distance is only comparable to a gate in the same units.
ROW = "row"
COL = "col"


def _axes(periodic: bool) -> Tuple[AxisSpec, AxisSpec]:
    return (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
            AxisSpec(COL, "space", units="cells", periodic=bool(periodic), ordinal=1))


def _spans_globe(grid) -> bool:
    """True when this grid's columns cover a whole circle of longitude.

    Latitude never wraps -- the pole is not the other pole -- so only the column axis is ever
    periodic, and only when the columns actually close.
    """
    if getattr(grid, "kind", None) != "latlon":
        return False
    span = abs(float(grid.dx)) * int(grid.shape[1])
    return abs(span - 360.0) < 1e-6


def _scale_quantity(scale: Any) -> Quantity:
    """The dyadic octave of a level, as a length whose *ratios* are the transform's own.

    Level `j` is given `2 ** (j - 1)` parent pixels. The constant is a labelling convention and
    nothing reads it; what everything reads is that consecutive levels differ by a factor of
    two, which is true of the transform by construction rather than true of this line.

    The unit is `cells`, the same name `_axes` gives the parent grid, and it says `cells`
    because it originally said `parent-grid px` (D90). Those are two spellings of one unit --
    the bank is undecimated and the alignment step puts every level on the parent grid, so a
    parent-grid pixel *is* a cell -- but TG3.3's `distance` relation divides a separation by a
    scale and refuses the quotient when the two unit *names* differ. The effect was that the
    one relation which makes a configuration recognisable refused on every feature this
    module produces, for a spelling. The precision that name carried is not lost: `scale_basis`
    below still records that the octave is counted in cells of the parent grid.
    """
    try:
        level = int(scale)
    except (TypeError, ValueError):
        level = None
    if level is None or level < 1 or str(scale) != str(level):
        raise InvalidParameterError(
            "CoefficientField.scales", scale,
            "a positive integer dyadic level. A scale label that is not a level has no "
            "declared octave, and inventing one would put a made-up number into every scale "
            "ratio and every doubling time computed from these tracks")
    return Quantity(float(2 ** (level - 1)), "cells")


def _orientation(value: Any, convention_note: Optional[str]) -> Optional[Orientation]:
    """An angle when the field says its orientations are angles, otherwise nothing.

    The field's own convention string is the authority. A separable band label is refused an
    angle rather than given a plausible one: LH/HL/HH are three filters, and HH responds to
    both diagonal signs, so no angle describes it.
    """
    if convention_note is None or "degrees" not in str(convention_note):
        return None
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(degrees):
        return None
    return Orientation(degrees, "axis_180")


def _position_uncertainty(plateau_pixels: int) -> float:
    """Half the linear size of the tied group the maximum occupies.

    A maximum spread over sixteen equal pixels could sit anywhere inside them, and its centroid
    is a statement about the group rather than about the peak. A single refined peak keeps half
    a pixel, which is the sampling bound its parabola started from and is not claimed to be
    beaten.
    """
    return 0.5 * math.sqrt(max(1, int(plateau_pixels)))


def spectral_feature_set(
    field,
    detection: Optional[Mapping[str, Any]] = None,
    *,
    domain: str,
    dataset: str,
    variable: Optional[str] = None,
    time_units: Optional[str] = "s",
    periodic: Optional[bool] = None,
    **detect_kwargs: Any,
) -> Optional[FeatureSet]:
    """Translate a T4D.1 detection over `field` into trackable features.

    Returns `None` when nothing was detected anywhere: a `FeatureSet` refuses to be empty, and
    "the search found nothing" is a fact about the search rather than a set with no members.

    `domain` and `dataset` are required and are not guessed. They are what rule R19 refuses
    comparisons by name on, and a default would put a name nobody chose onto every feature.
    """
    for name, value in (("domain", domain), ("dataset", dataset)):
        if not str(value or "").strip():
            raise InvalidParameterError(
                "spectral_feature_set.%s" % name, value,
                "a non-empty %s. It is what R19 refuses a cross-domain comparison by, and a "
                "feature that cannot name it cannot be refused by name either" % name)
    resolved_variable = variable or getattr(field, "source_variable", None)
    if not str(resolved_variable or "").strip():
        raise InvalidParameterError(
            "spectral_feature_set.variable", variable,
            "the variable these coefficients came from. The field did not record one, so it "
            "has to be declared here rather than left blank")

    wraps = _spans_globe(field.grid) if periodic is None else bool(periodic)
    if periodic is False and _spans_globe(field.grid):
        raise InvalidParameterError(
            "spectral_feature_set.periodic", periodic,
            "True for a grid whose columns span the full 360 degrees of longitude. Declaring "
            "it flat would make every crossing of the date line two tracks instead of one, "
            "and the count of objects is the answer this produces")
    axes = _axes(wraps)

    if detection is None:
        detection = detect_features(field, **detect_kwargs)
    elif detect_kwargs:
        raise InvalidParameterError(
            "spectral_feature_set.detect_kwargs", sorted(detect_kwargs),
            "no detection settings when a finished detection is supplied. Two sets of "
            "settings, one of which did not run, is a receipt that describes the wrong pass")

    alignment = detection["alignment"]
    if len(field.scales) > 1 and not alignment["cross_scale_comparable"]:
        offenders = sorted(str(scale) for scale in alignment["exact"]
                           if not alignment["exact"][scale])
        raise InvalidParameterError(
            "spectral_feature_set.field", offenders,
            "a decomposition whose scales are registered to one another (D88). Linking pools "
            "every band into one frame, so two scales displaced relative to each other would "
            "be associated on a separation that is mostly the transform's analysis delay -- "
            "tens of pixels at coarse levels, and growing with level, so the error is not even "
            "constant. Either decompose with a linear-phase filter, whose delay is one number "
            "and comes out exactly, or select a single scale and track within it. Bases: %s"
            % "; ".join(sorted({alignment["basis"][scale] for scale in offenders})))

    units = getattr(field.grid, "variable_units", None)
    representation = _representation(field)
    shared = {
        "detection_schema": detection["schema"],
        "threshold_source": detection["threshold_source"],
        "threshold_sigma": detection["threshold_sigma"],
        "interior_applied": detection["interior"]["applied"],
        "wavelet_family": field.wavelet_family,
        "wavelet_config": dict(getattr(field, "config", {}) or {}),
        "orientation_convention": getattr(field, "orientation_convention", None),
        "alignment_applied": alignment["applied"],
        "alignment_cross_scale_comparable": alignment["cross_scale_comparable"],
        "significance": ("not established. A maximum clearing a multiple of the record's RMS "
                         "has not been tested against a null, and the threshold it cleared is "
                         "recorded here instead of being reported as a p-value"),
    }

    features = []
    for record in detection["features"]:
        time_index = int(record["time_index"])
        uncertainty = _position_uncertainty(record["plateau_pixels"])
        features.append(SpectralFeature(
            domain=str(domain), dataset=str(dataset), variable=str(resolved_variable),
            magnitude=Quantity(float(record["strength"]), units),
            location=FeatureLocation(
                {ROW: float(record["y"]), COL: float(record["x"])}, axes,
                uncertainty={ROW: uncertainty, COL: uncertainty}),
            time=float(field.times[time_index]),
            time_units=time_units,
            representation=representation,
            spatial_scale=_scale_quantity(record["scale"]),
            orientation=_orientation(record["orientation"], shared["orientation_convention"]),
            significance=None,
            provenance={
                **shared,
                "feature_id": record["feature_id"],
                "time_index": time_index,
                "scale_label": record["scale"],
                "orientation_label": record["orientation"],
                "threshold": record["threshold"],
                "plateau_pixels": record["plateau_pixels"],
                "phase": record["phase"],
                "scale_basis": ("dyadic octave 2 ** (level - 1) cells of the parent grid; "
                                "only ratios are used and consecutive levels differ by "
                                "exactly two"),
                "alignment_shift_parent_px": alignment["shift_parent_px"][record["scale"]],
            }))
    if not features:
        return None
    return FeatureSet(features)


def _representation(field) -> str:
    """What produced these coefficients, at the granularity rule R8's audit has to ask at.

    The family alone is not enough: a Haar bank and a db4 bank of the same family manufacture
    different edges, and TG2.4 asks which one a feature came from.
    """
    config = dict(getattr(field, "config", {}) or {})
    parts = [str(field.wavelet_family)]
    for key in ("wavelet", "level1", "qshift", "levels", "mode"):
        if key in config:
            parts.append("%s=%s" % (key, config[key]))
    return "wavelet[%s]" % ",".join(parts)


#: The public name for the representation string, read by T4D.3's narratives. One
#: definition, so a receipt and a sentence cannot disagree about which bank ran.
representation_of = _representation


def track_spectral_features(
    field,
    *,
    domain: str,
    dataset: str,
    variable: Optional[str] = None,
    time_units: Optional[str] = "s",
    periodic: Optional[bool] = None,
    detection: Optional[Mapping[str, Any]] = None,
    alpha: float = 0.05,
    associator: str = "hungarian",
    bounds: MotionBounds = NO_BOUNDS,
    **detect_kwargs: Any,
):
    """Detect in every band of `field`, then link the detections frame to frame.

    Returns a `src.core.tracking.TrackingResult`, or `None` when no band of no frame produced a
    detection. The clock handed to the tracker is *every* frame of the field, not every frame a
    feature was found in, so a structure that leaves the crop closes its track at the frame it
    left rather than at the end of its own evidence.
    """
    features = spectral_feature_set(
        field, detection, domain=domain, dataset=dataset, variable=variable,
        time_units=time_units, periodic=periodic, **detect_kwargs)
    if features is None:
        return None
    height, width = int(field.grid.shape[0]), int(field.grid.shape[1])
    volume = SearchVolume({ROW: float(height), COL: float(width)}, units="cells")
    return track(features, volume=volume, times=[float(t) for t in field.times],
                 associator=associator, alpha=alpha, bounds=bounds)


def describe_tracking(result, *, field=None) -> Dict[str, Any]:
    """A receipt for a tracking pass, with the boundary of the claim attached to it.

    `TrackingResult.describe` already reports the gates, the radii and what each step refused.
    What it cannot know is that these features are coefficient maxima: that a track is a chain
    of local maxima of one transform and not a trajectory of an object, and that a scale
    velocity is a movement between dyadic levels of that transform. Saying so here is what
    stops a track count being read as an object count.
    """
    body = dict(result.describe())
    body["schema"] = TRACKING_SCHEMA
    body["claim_boundary"] = (
        "A track links local maxima of this transform's coefficient magnitude across frames "
        "under a derived coincidence radius and the declared gates. It is not a trajectory of "
        "a physical object, it does not claim a merge or a split, and its scale velocity is "
        "movement between dyadic levels of this transform rather than growth of anything "
        "named. Bands are pooled before linking, so one structure straddling two levels can "
        "present two detections in a frame and only one of them may continue a track.")
    if field is not None:
        body["representation"] = _representation(field)
        body["bands"] = {
            "scales": [str(scale) for scale in field.scales],
            "orientations": [str(orientation) for orientation in field.orientations],
            "orientation_convention": getattr(field, "orientation_convention", None),
        }
    return body


def track_lengths(result) -> Sequence[int]:
    """Frames per track, longest first -- the cheapest honest summary of a tracking pass."""
    return sorted((len(item) for item in result), reverse=True)
