"""T4D.3: what a track from T4D.2 may be said to be, in a sentence a person can read.

A narrative is the most dangerous artefact in this phase, and it is worth saying why before
saying what it does. Every other output of Phase 4D is a number with units attached, and a
number that is misread is usually misread visibly. A sentence is different: it is *believed*.
"Travelled southeast over six frames while its dominant scale doubled" reads as a description
of a storm, and nothing in the grammar admits that each of its three clauses is a claim about
a bank of filters rather than about the weather. So this module is mostly a set of refusals
about wording, and the numbers in it are the easy half.

**Four things this refuses to say, and the measurement behind each.**

*It never says the structure travelled.* A detail coefficient peaks at a structure's flank, one
analysing width out along the axis its band high-passes (T4D.2), and that offset grows as the
structure grows. What moved between two frames is the maximum, and every sentence here names
the maximum as its subject. The distinction is not pedantry: along the high-passed axis, the
track's speed is advection *plus* the structure's own growth, and the two are not separable
from one track.

*It never says a track's dominant scale doubled.* The bank is redundant, so a structure whose
width doubles excites the coarse level *beside* the fine one rather than instead of it. A track
that holds one level therefore has a scale velocity of exactly zero, and that zero is a true
statement about the track and a false one about the structure. Scale evolution is reported once,
over the population -- which bands were excited and when each was first excited -- and it is
labelled as co-occurrence of two bands over one record, which is what was measured. A single
structure's ascent through the bank would require claiming a merge, and the tracker claims none.

*It never says north without a grid that knows where north is.* A compass bearing needs two
things a pixel grid does not have: the sign that relates row order to latitude, and the
cosine that stops a degree of longitude being counted as long as a degree of latitude. With a
`latlon` grid carrying `lat0` both are available and the bearing is computed in metres. Without
them the sentence says "toward increasing row and decreasing column" -- uglier, and true.

*It never says energy when it measured magnitude.* The roadmap's example sentence ends "and
coefficient energy rose 43%". A coefficient's energy is the square of its magnitude, so a 43%
rise in one is a 104% rise in the other, and the two names are not interchangeable at any
precision. Both are reported, each under its own name.

**The causal guard, and why it scans one half of the output.** `OUTSIDE_THE_LADDER` is imported
from `claim_ladder` rather than restated, so there is one list in the programme of what it will
not say, and `assert_no_causal_language` runs over every rendered sentence before a narrative is
returned -- it catches an edit to a template here, not a bad input. What it does *not* scan is
the entitlement, for the same reason `src.core.translation` does not: the entitlement's job is
to name the boundary, and it names "mechanism" precisely in order to refuse it. Scanning the
sentence that holds the line would refuse the line (R7).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.analysis_engine.spectral_tracking import COL, ROW, representation_of
from src.core.claim_ladder import OUTSIDE_THE_LADDER
from src.core.errors import InvalidParameterError
from src.core.feature import SemanticComparisonError

NARRATIVE_SCHEMA = "spectral-track-narrative/v1"

#: Eight points, because a track of a dozen frames does not support sixteen. The bearing is
#: measured clockwise from north, and each name owns the 45 degrees centred on it.
COMPASS_POINTS = ("north", "north-east", "east", "south-east",
                  "south", "south-west", "west", "north-west")

#: The one sentence every narrative carries, and the only place in this module where a word
#: from `OUTSIDE_THE_LADDER` is allowed to appear -- it appears in order to be refused.
ENTITLEMENT = (
    "This describes a chain of local maxima of one transform's coefficient magnitude, linked "
    "under a derived coincidence radius and the declared gates. It is not the path of a "
    "physical object. The position followed is a flank of the structure, offset along the "
    "band's high-passed axis by roughly the structure's own width, so the along-axis speed is "
    "advection together with growth and the two are not separable from one track. No "
    "significance was established: the detections cleared a multiple of the record's RMS, "
    "which is not a test against a null. Nothing here is a statement about mechanism or cause, "
    "and this programme provides no framework that could license one (R7).")


# ------------------------------------------------------------------------------ the guard


def assert_no_causal_language(sentences: Sequence[str], *, where: str = "narrative") -> None:
    """Refuse rendered narrative text carrying a term the ladder places outside itself (R7).

    A hit means one of two things, and both are worth stopping before a reader sees them: a
    template in this module was edited, or a caller named their domain, dataset or variable
    with a causal word and it flowed into a sentence. The second is not a false positive --
    the sentence is what gets believed, and a dataset called `co2_causes_warming` puts the
    claim in front of the reader regardless of who typed it.

    Punctuation is flattened to spaces before the word boundaries are applied, and that is
    what makes the second case work at all: `causes` does not match inside
    `co2_causes_warming`, an underscore being a word character, so a plain word-boundary scan
    passes exactly the identifiers a dataset is most likely to be named with. What is still
    not caught is a word run together with another without a separator -- `co2causeswarming`
    -- and that limit is stated rather than papered over with a substring match, which would
    refuse "becausewhat" and every other innocent word containing one of these.
    """
    for index, sentence in enumerate(sentences):
        lowered = re.sub(r"[^a-z0-9]+", " ", str(sentence).lower())
        for word in OUTSIDE_THE_LADDER:
            if re.search(r"\b%s\b" % re.escape(word), lowered):
                raise InvalidParameterError(
                    "%s sentence %d" % (where, index), word,
                    "wording free of causal vocabulary. The claim ladder places %r outside "
                    "itself at every rung, and a track of coefficient maxima is at the very "
                    "bottom of it; rendering the word for a reader is how the boundary is "
                    "lost (R7)" % word)


# ------------------------------------------------------------------------------ direction


def _bearing(track, grid) -> Optional[Dict[str, Any]]:
    """A compass bearing for this track's displacement, or `None` with the reason recorded.

    Two facts have to come from the grid, and neither can be assumed. The sign of `dy` says
    whether row order runs north to south (ERA5) or south to north, and `lat0` gives the
    latitude at which a degree of longitude is measured -- without the cosine, a track at 60
    degrees north is reported as having gone twice as far east as it did, which rotates the
    bearing rather than merely mis-scaling it.

    There is deliberately no guard here for a `latlon` grid that records no `lat0`: `GridSpec`
    already refuses to construct one, on the grounds that the zonal metric depends on latitude
    and a lat/lon grid without a latitude origin has no metric at all. A second check would be
    a branch no input can reach, and an unreachable refusal reads like a covered case.
    """
    if grid is None:
        return None
    if getattr(grid, "kind", None) != "latlon":
        return None
    displacement = track.displacement()
    if ROW not in displacement or COL not in displacement:
        return None
    rows = [item.location.coords[ROW] for item in track]
    mean_lat = float(grid.lat0) + (sum(rows) / len(rows)) * float(grid.dy)
    if abs(mean_lat) >= 90.0:
        return None
    radians = math.pi / 180.0
    radius = float(grid.radius_m)
    # lat(row) = lat0 + row * dy, so a positive row displacement moves north exactly when dy
    # is positive -- which is the case ERA5 does not satisfy, and the sign is read from the
    # grid rather than assumed for that reason.
    north_m = displacement[ROW] * float(grid.dy) * radians * radius
    east_m = (displacement[COL] * float(grid.dx) * radians * radius
              * math.cos(mean_lat * radians))
    if north_m == 0.0 and east_m == 0.0:
        return None
    degrees = math.degrees(math.atan2(east_m, north_m)) % 360.0
    index = int((degrees + 22.5) // 45.0) % 8
    return {"point": COMPASS_POINTS[index], "bearing_deg": degrees,
            "north_m": north_m, "east_m": east_m,
            "basis": ("bearing clockwise from north, from displacement converted to metres "
                      "with the grid's own dy sign and a cosine at the track's mean latitude "
                      "of %.3f degrees" % mean_lat)}


def _axis_direction(displacement: Mapping[str, float]) -> str:
    """Axis-relative wording, for a grid that does not know where north is."""
    parts = []
    for name in (ROW, COL):
        delta = displacement.get(name, 0.0)
        if delta > 0:
            parts.append("increasing %s" % name)
        elif delta < 0:
            parts.append("decreasing %s" % name)
        else:
            parts.append("no net %s" % name)
    return "toward %s and %s" % (parts[0], parts[1])


def _no_compass_reason(grid) -> str:
    if grid is None:
        return ("no grid was supplied, so nothing relates row order to latitude")
    kind = getattr(grid, "kind", None)
    if kind != "latlon":
        return ("a %r grid declares no orientation: its rows are an index, not a latitude, "
                "and calling one end of them north would be an invention" % kind)
    return ("this latlon grid could take a bearing, but this track's net displacement is "
            "exactly zero or its mean latitude is at or beyond a pole, and neither has a "
            "direction")


# -------------------------------------------------------------------------------- pieces


def _band(track) -> Dict[str, Any]:
    first = track[0]
    labels = {(item.provenance.get("scale_label"),
               item.provenance.get("orientation_label")) for item in track}
    return {
        "scale_label": first.provenance.get("scale_label"),
        "orientation_label": first.provenance.get("orientation_label"),
        "single_band": len(labels) == 1,
        "bands_visited": sorted("%s/%s" % pair for pair in labels),
    }


def _magnitude_change(track) -> Optional[Dict[str, Any]]:
    """First-to-last change in peak coefficient magnitude, and the same change in energy.

    The two percentages are omitted, and the reason recorded in their place, when the first
    observation's magnitude is zero: a change from zero is not large, it is undefined, and
    rendering "inf%" is worse than rendering nothing. The absolute values are still reported,
    since those were measured.
    """
    first = float(track[0].magnitude.value)
    last = float(track[-1].magnitude.value)
    values = [float(item.magnitude.value) for item in track]
    record = {"first": first, "last": last, "peak": max(values),
              "units": track[0].magnitude.units,
              "meaning": ("magnitude of the wavelet detail coefficient at the tracked "
                          "maximum; energy is its square")}
    if first == 0.0:
        record["magnitude_change_pct"] = None
        record["energy_change_pct"] = None
        record["refused"] = ("a percentage change from a first magnitude of zero is "
                             "undefined, not infinite")
        return record
    record["magnitude_change_pct"] = (last - first) / abs(first) * 100.0
    record["energy_change_pct"] = ((last * last) - (first * first)) / (first * first) * 100.0
    return record


def _scale_story(track, band) -> Dict[str, Any]:
    """What this one track supports about scale, which is usually less than a reader expects."""
    if band["single_band"] and len(track) > 1:
        return {
            "held_level": band["scale_label"],
            "scale_velocity_doublings_per_time": 0.0,
            "doubling_time": None,
            "note": ("this track held one dyadic level for its whole life, so its scale "
                     "velocity is exactly zero. The bank is redundant: a structure whose "
                     "width doubles excites the coarser level alongside this one rather than "
                     "vacating this one, so a growing structure appears as parallel tracks "
                     "and its scale evolution is in the population, not here"),
        }
    story: Dict[str, Any] = {"held_level": None, "doubling_time": None}
    if len(track) < 2:
        story["scale_velocity_doublings_per_time"] = None
        story["note"] = "a single sighting supports no growth rate"
        return story
    try:
        rate = track.scale_velocity()
    except (InvalidParameterError, SemanticComparisonError) as exc:
        story["scale_velocity_doublings_per_time"] = None
        story["refused"] = str(exc)
        return story
    story["scale_velocity_doublings_per_time"] = rate
    if rate > 0.0:
        story["doubling_time"] = track.doubling_time().describe()
    story["note"] = ("this track was linked across more than one dyadic level, so the rate is "
                     "a fit over the levels the association admitted and not the growth of a "
                     "named object")
    return story


def _fmt(value: float, places: int = 2) -> str:
    return ("%%.%df" % places) % value


# ------------------------------------------------------------------------------ per track


def narrate_track(track, *, grid=None, frames_searched: Optional[int] = None) -> Dict[str, Any]:
    """One track, as a sentence plus every number the sentence was built from.

    The numbers are returned beside the prose deliberately. A narrative that cannot be checked
    against its own inputs is a claim, and the point of this module is that it produces a
    description instead.
    """
    band = _band(track)
    clock = track.time_units or "unrecorded units"
    where = "level %s, %s" % (band["scale_label"], band["orientation_label"])
    # The dataset and variable are named in the prose, not only in the receipt. A sentence
    # about "the coefficient maximum" with no subject is one a reader will supply a subject
    # for, and it also puts the caller's own words where `assert_no_causal_language` can see
    # them: a dataset named for a causal claim is refused here rather than rendered.
    subject = "%s, %s; %s" % (track.observations.dataset, track.observations.variable, where)

    if len(track) < 2:
        sentence = (
            "Track %d (%s): a maximum was seen once, at time %s %s, and not linked to any "
            "other frame. A single sighting supports no direction, no speed and no growth."
            % (track.track_id, subject, _fmt(track.birth_time), clock))
        body = {"schema": NARRATIVE_SCHEMA, "track_id": track.track_id, "band": band,
                "observations": len(track), "sentence": sentence,
                "structural_signature": "L%s/%s|n=1" % (band["scale_label"],
                                                          band["orientation_label"]),
                "entitlement": ENTITLEMENT}
        assert_no_causal_language([sentence], where="track %d" % track.track_id)
        return body

    displacement = track.displacement()
    speed = track.speed()
    bearing = _bearing(track, grid)
    distance = math.sqrt(sum(d * d for d in displacement.values()))
    magnitude = _magnitude_change(track)
    scale = _scale_story(track, band)

    if bearing is not None:
        direction_clause = "%s (bearing %s degrees)" % (bearing["point"],
                                                        _fmt(bearing["bearing_deg"], 1))
    else:
        direction_clause = _axis_direction(displacement)

    span = "%s to %s %s" % (_fmt(track.birth_time), _fmt(track.last_time), clock)
    frames = ("%d consecutive searched frames" % len(track) if frames_searched is None
              else "%d of %d searched frames, consecutively" % (len(track), frames_searched))

    clauses = [
        "Track %d (%s): the coefficient maximum was followed across %s, %s."
        % (track.track_id, subject, frames, span),
        "It moved %s cells %s, at a mean %s %s."
        % (_fmt(distance, 1), direction_clause, _fmt(speed.value), speed.units or "cells/step"),
    ]
    if scale.get("held_level") is not None:
        clauses.append("It stayed at level %s throughout, so this track reports no change of "
                       "scale; which bands were excited, and when, is a statement about the "
                       "set of tracks rather than about this one."
                       % scale["held_level"])
    elif scale.get("scale_velocity_doublings_per_time"):
        clauses.append("Its linked level changed at %s doublings per %s."
                       % (_fmt(scale["scale_velocity_doublings_per_time"], 3), clock))
    if magnitude and magnitude.get("magnitude_change_pct") is not None:
        clauses.append(
            "Its peak coefficient magnitude went from %s to %s, a change of %s%%, which is a "
            "change of %s%% in coefficient energy -- energy being the square, so the two "
            "figures are not interchangeable."
            % (_fmt(magnitude["first"], 4), _fmt(magnitude["last"], 4),
               _fmt(magnitude["magnitude_change_pct"], 1),
               _fmt(magnitude["energy_change_pct"], 1)))

    sentence = " ".join(clauses)
    assert_no_causal_language([sentence], where="track %d" % track.track_id)
    return {
        "schema": NARRATIVE_SCHEMA,
        "track_id": track.track_id,
        "band": band,
        "observations": len(track),
        "sentence": sentence,
        "span": {"from": track.birth_time, "to": track.last_time, "units": track.time_units},
        "displacement_cells": dict(displacement),
        "distance_cells": distance,
        "speed": speed.describe(),
        "direction": (bearing if bearing is not None
                      else {"point": None, "axis_relative": _axis_direction(displacement),
                            "refused": _no_compass_reason(grid)}),
        "magnitude": magnitude,
        "scale": scale,
        "structural_signature": _structural_signature(track, band, distance, speed, magnitude),
        "entitlement": ENTITLEMENT,
    }


def _structural_signature(track, band, distance, speed, magnitude) -> str:
    """A compact label naming structure and nothing that is domain-specific.

    R19's boundary in one string: no variable, no dataset, no units of the measured quantity.
    A percentage change of a coefficient and a count of frames survive a domain change; a
    temperature does not.
    """
    change = ("na" if not magnitude or magnitude.get("magnitude_change_pct") is None
              else "%+.0f%%" % magnitude["magnitude_change_pct"])
    return "%s|n=%d|d=%s|v=%s|dmag=%s" % (
        "L%s/%s" % (band["scale_label"], band["orientation_label"]),
        len(track), _fmt(distance, 1), _fmt(speed.value), change)


# ----------------------------------------------------------------------------- population


def narrate_population(result) -> Dict[str, Any]:
    """Where scale evolution actually lives: which bands were excited, and when each began.

    This is the honest home of the roadmap's "dominant scale doubled". Two bands excited nine
    frames apart is an ordering of two measurements over one record. It is offered as a
    *candidate precursor relationship* -- the wording R7 permits -- and it is neither a merge,
    nor a causal statement, nor a tested one: no null was consulted, and one record is one
    observation of the ordering, not evidence that it recurs.
    """
    per_band: Dict[str, Dict[str, Any]] = {}
    for item in result:
        band = _band(item)
        key = "L%s/%s" % (band["scale_label"], band["orientation_label"])
        record = per_band.setdefault(key, {"band": key, "tracks": 0, "first_seen": None,
                                           "longest": 0})
        record["tracks"] += 1
        record["longest"] = max(record["longest"], len(item))
        if record["first_seen"] is None or item.birth_time < record["first_seen"]:
            record["first_seen"] = item.birth_time
    order = sorted(per_band.values(), key=lambda r: (r["first_seen"], r["band"]))

    if not order:
        sentence = "No band produced a track, so there is no ordering of bands to report."
    elif len(order) == 1:
        sentence = ("One band produced tracks (%s, first at %s). One band supports no "
                    "ordering and no candidate precursor relationship."
                    % (order[0]["band"], _fmt(order[0]["first_seen"])))
    else:
        earliest, latest = order[0], order[-1]
        sentence = (
            "%d bands produced tracks. The earliest was %s at %s and the latest %s at %s, a "
            "separation of %s in the clock's units. That ordering is a candidate precursor "
            "relationship between two bands of one record: it is co-occurrence with a "
            "recorded sign of the time offset, it was not tested against a null, and it is "
            "not a structure moving up the bank -- the transform is redundant, so both bands "
            "respond at once and no merge is claimed."
            % (len(order), earliest["band"], _fmt(earliest["first_seen"]),
               latest["band"], _fmt(latest["first_seen"]),
               _fmt(latest["first_seen"] - earliest["first_seen"])))
    assert_no_causal_language([sentence], where="population")
    return {"schema": NARRATIVE_SCHEMA, "bands": order, "sentence": sentence,
            "entitlement": ENTITLEMENT}


# -------------------------------------------------------------------------------- the pass


def narrate_tracking(result, *, field=None, grid=None) -> Dict[str, Any]:
    """Every track in a pass, plus the population statement and the pass's own accounting.

    `result` may not be `None`. `track_spectral_features` returns `None` when no band of no
    frame produced a detection, and a narrative of nothing is not an empty list of sentences:
    it is the statement that a search ran and found nothing, which the caller has to make
    itself rather than have inferred from a missing argument.
    """
    if result is None:
        raise InvalidParameterError(
            "narrate_tracking.result", None,
            "a TrackingResult. `None` means the search ran and no band of no frame produced "
            "a detection, which is a fact about the search that a narrative cannot state on "
            "its behalf; say so at the call site instead of narrating an absence")
    if grid is None and field is not None:
        grid = getattr(field, "grid", None)
    frames = len(result.frame_times)
    narratives: List[Dict[str, Any]] = [
        narrate_track(item, grid=grid, frames_searched=frames) for item in result]
    population = narrate_population(result)
    body = {
        "schema": NARRATIVE_SCHEMA,
        "tracks": narratives,
        "population": population,
        "frames_searched": frames,
        "empty_frames": list(result.empty_frames),
        "track_count": len(result),
        "compass": ({"available": True, "basis": narratives[0]["direction"].get("basis")}
                    if narratives and narratives[0]["direction"].get("point") is not None
                    else {"available": False, "reason": _no_compass_reason(grid)}),
        "entitlement": ENTITLEMENT,
    }
    if field is not None:
        body["representation"] = representation_of(field)
    return body


def render_text(narrative: Mapping[str, Any]) -> str:
    """The whole pass as plain prose, entitlement last and exactly once."""
    lines = [item["sentence"] for item in narrative["tracks"]]
    lines.append(narrative["population"]["sentence"])
    lines.append(narrative["entitlement"])
    return "\n\n".join(lines)
