"""Descriptive T4E.8 identity errors against tracked-constellation proxy labels.

No null, significance, independent-sample claim or automatic calibration acceptance lives here.
The caller supplies a predeclared radius, or explicitly requests a calibration quantile.
Storage is O(P + U), runtime O(U log U + P log U), for P repeat and U unrelated distances.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.core.errors import InvalidParameterError


def _distances(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise InvalidParameterError("identity_audit.distances", "invalid",
                                    "a one-dimensional sequence of finite nonnegative distances")
    return result


def recall_radius(repeats: Sequence[float], recall: float) -> Optional[float]:
    """Smallest observed radius reaching the declared empirical recall; None if unmeasured."""
    if not math.isfinite(recall) or not 0 < recall <= 1:
        raise InvalidParameterError("identity_audit.recall", recall, "a recall in (0, 1]")
    values = _distances(repeats)
    if not len(values):
        return None
    return float(np.sort(values)[math.ceil(recall * len(values)) - 1])


def labelled_errors(repeats: Sequence[float], unrelated: Sequence[float],
                    radius: Optional[float]) -> Dict[str, Any]:
    """Both absolute empirical error rates, with ties included inside the radius.

    AUC is P(repeat distance < unrelated distance) + half the tie probability. Empty
    populations remain unmeasured, never zero errors. Labels describe track continuity only.
    """
    same, different = _distances(repeats), _distances(unrelated)
    if radius is not None and (not math.isfinite(radius) or radius < 0):
        raise InvalidParameterError("identity_audit.radius", radius,
                                    "a finite nonnegative radius or None when not calibrated")
    auc = None
    if len(same) and len(different):
        sorted_different = np.sort(different)
        below = np.searchsorted(sorted_different, same, side="left")
        through = np.searchsorted(sorted_different, same, side="right")
        auc = float(np.mean((len(different) - through + (through - below) / 2)
                            / len(different)))
    split = int(np.count_nonzero(same > radius)) if radius is not None and len(same) else None
    admitted = (int(np.count_nonzero(different <= radius))
                if radius is not None and len(different) else None)
    return {
        "radius": radius, "repeat_pairs": len(same), "unrelated_pairs": len(different),
        "false_split_count": split, "false_admission_count": admitted,
        "false_split_rate": None if split is None else split / len(same),
        "false_admission_rate": None if admitted is None else admitted / len(different),
        "auc": auc,
        "label_boundary": "Tracked-key proxy labels, not independent measurements or physical ground truth",
    }


def radius_feasibility(repeats: Sequence[float], unrelated: Sequence[float], *,
                       max_split: float, max_admission: float) -> Dict[str, Any]:
    """Whether any radius meets both empirical bounds on these supplied proxy populations.

    Admission is monotone in radius. The smallest radius satisfying the split bound
    therefore minimises admission among all split-feasible radii. This is a diagnostic of
    this population only, never an approved operating point or a generalisation bound.
    """
    for name, value in (("max_split", max_split), ("max_admission", max_admission)):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise InvalidParameterError("identity_audit." + name, value, "a bound in [0, 1)")
    same = _distances(repeats)
    # Count admissible rejections directly. Subtracting max_split from 1 first can
    # round 0.3 upward and ceil(0.30000000000000004 * 10) would then require four.
    radius = (float(np.sort(same)[len(same) - math.floor(max_split * len(same)) - 1])
              if len(same) else None)
    report = labelled_errors(repeats, unrelated, radius)
    admission = report["false_admission_rate"]
    return {"smallest_split_feasible_radius": radius, "errors": report,
            "any_radius_meets_both_empirical_bounds": (None if radius is None or admission is None
                                                      else admission <= max_admission),
            "boundary": "Descriptive feasibility on supplied proxy labels; not a mining radius"}
