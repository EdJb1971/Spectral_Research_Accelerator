"""T4E.26: subtracting what was found, so the null is less a null for itself.

T4E.25 measured the problem this module addresses. A cut calibrated on a scene containing signal
sits **3.53x** above the cut on that scene's own background, because the signal's power enters
every surrogate of the field it is meant to be a null for. Relaxing the declared error level does
not reach it: a tenfold change in alpha moves the threshold about 20 per cent.

The procedure here is the only repair that could run on a real record, where the signal-free
background does not exist. Extract, subtract what was found, recalibrate on the residual,
re-extract from the original scene against the lowered cut. One round.

**Two things travel with every number this module produces.**

*The reported significance changes meaning.* `NullCalibration` states its hypothesis as a field
with **the same power spectrum** as the frame. A peeled ensemble has the residual's spectrum, so
a p-value taken through a peeled cut answers a different question. That is not a caveat to be
placed beneath a coverage figure; it is a finding that belongs beside one.

*The subtraction is systematically wrong where the features are asymmetric.*
`local_maximum_extractor` declares `shape_model: isotropic_gaussian_on_a_flat_baseline`. Removing
an isotropic fit from a stretched feature leaves lobes along the stretch axis -- and those lobes
are then measured against a threshold the same subtraction has just lowered. The failure mode is
manufacturing the features it goes on to report, which is why `split_contamination` exists and
why a single combined contamination figure does not satisfy this task's acceptance.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: A spurious feature this close to a peeled one, in units of that peeled feature's own fitted
#: width, is counted as a residual artefact rather than as background contamination. Two widths
#: is where an isotropic fit to a 2.5x-stretched feature leaves its lobes; the multiple is
#: declared rather than tuned, and it is not moved after the counts exist.
ARTEFACT_RADIUS_IN_FITTED_WIDTHS: float = 2.0


def fitted_gaussian(shape: Tuple[int, int], centre: Tuple[float, float],
                    amplitude: float, width: float) -> np.ndarray:
    """The isotropic Gaussian the extractor itself fitted, rebuilt from what it published.

    Every parameter is read from the feature's provenance -- `magnitude` less `baseline`,
    `scale_cells`, `index_position` -- and none is refitted here. A refit would make the
    subtraction a second, unaudited estimator, and the point of peeling is to remove exactly what
    the pipeline claimed to find, not something better.
    """
    n0, n1 = int(shape[0]), int(shape[1])
    rows, cols = np.mgrid[0:n0, 0:n1].astype(np.float64)
    dy, dx = rows - float(centre[0]), cols - float(centre[1])
    sigma = float(width)
    if not (sigma > 0.0) or not math.isfinite(sigma):
        return np.zeros((n0, n1), dtype=np.float64)
    return float(amplitude) * np.exp(-0.5 * (dy * dy + dx * dx) / (sigma * sigma))


def peel(values: np.ndarray, features: Sequence) -> Tuple[np.ndarray, Tuple[Dict[str, float], ...]]:
    """Subtract every reported feature's own fitted shape, and say what was subtracted.

    A feature whose provenance lacks a usable width is left in place and reported rather than
    guessed at: a subtraction with an invented width removes power the pipeline never claimed.
    """
    residual = np.array(values, dtype=np.float64, copy=True)
    subtracted: List[Dict[str, float]] = []
    for feature in features:
        provenance = dict(feature.provenance)
        width = provenance.get("scale_cells")
        centre = provenance.get("index_position")
        baseline = provenance.get("baseline")
        if width is None or centre is None or baseline is None:
            continue
        amplitude = float(feature.magnitude.value) - float(baseline)
        residual -= fitted_gaussian(residual.shape, (centre[0], centre[1]), amplitude, width)
        subtracted.append({"row": float(centre[0]), "col": float(centre[1]),
                           "width": float(width), "amplitude": float(amplitude)})
    return residual, tuple(subtracted)


def gap_closed(round_zero: float, peeled: float, oracle: float) -> Optional[float]:
    """How much of the distance to the unavailable oracle one round of peeling covered.

    `None` when round zero already sits at or below the oracle, because a fraction of a
    non-existent gap is not zero and reporting it as zero would invent a failure.
    """
    span = float(round_zero) - float(oracle)
    if span <= 0.0:
        return None
    return (float(round_zero) - float(peeled)) / span


def split_contamination(spurious: Sequence[Tuple[float, float]],
                        subtracted: Sequence[Dict[str, float]],
                        *, radius_in_widths: float = ARTEFACT_RADIUS_IN_FITTED_WIDTHS
                        ) -> Dict[str, object]:
    """Separate features the subtraction may have manufactured from ordinary contamination.

    Reporting one total would let a residual lobe -- a feature the procedure created and then
    found -- hide inside a background rate, and the whole risk of this procedure is that it does
    exactly that.
    """
    artefacts, elsewhere = [], []
    for position in spurious:
        row, col = float(position[0]), float(position[1])
        near = False
        for peeled in subtracted:
            reach = float(radius_in_widths) * float(peeled["width"])
            if math.hypot(row - peeled["row"], col - peeled["col"]) <= reach:
                near = True
                break
        (artefacts if near else elsewhere).append((row, col))
    total = len(artefacts) + len(elsewhere)
    return {
        "residual_artefacts": len(artefacts),
        "elsewhere": len(elsewhere),
        "artefact_fraction_of_spurious": (len(artefacts) / total) if total else None,
        "artefact_positions": tuple(artefacts),
        "radius_in_fitted_widths": float(radius_in_widths),
    }


def nearest_planting_stretch(position: Tuple[float, float],
                             plantings: Sequence) -> Optional[float]:
    """The stretch of the planting nearest a manufactured feature.

    Known by construction here and unavailable on a real record, which is why it is reported as
    a description of the mechanism and adjudicates nothing.
    """
    if not len(plantings):
        return None
    row, col = float(position[0]), float(position[1])
    best = min(plantings, key=lambda f: math.hypot(row - f.row, col - f.col))
    return float(best.stretch)
