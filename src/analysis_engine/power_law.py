"""The log-log least-squares core, with no turbulence in it (roadmap T4C.4, rules R2/R3).

`spectra.fit_power_law` grew up inside the radial-spectrum code and carries its vocabulary:
annuli, isotropy, `beta_energy_1d`, Charney and Kolmogorov. All of that is correct *for a
power spectrum* and meaningless for the two other power laws Phase 4C needs to fit - energy
against scale, and feature population against scale. Copying the fit would have produced a
second least-squares implementation to keep in step with the first; wrapping the spectral one
would have attached a turbulence interpretation to a quantity that has no turbulence in it.

So the arithmetic lives here, once, knowing nothing about what `x` and `y` are.
`spectra.fit_power_law` calls it and adds the spectral interpretation on top; its behaviour
is unchanged, which `test_spectra.py` continues to assert.

**A fitted exponent is not a finding (rule R2).** A fractional Brownian field yields a clean,
high-`r_squared` power law and contains nothing. Every exponent this module produces is
therefore returned with `reportable: False` until it has been compared against a surrogate
ensemble - see `compare_exponent_to_null`. That is a refusal expressed in the return value
rather than in a docstring, because a docstring cannot be checked by a test.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

MIN_POINTS = 4


class PowerLawError(ValueError):
    """Raised when a fit is impossible rather than merely uncertain."""


def loglog_fit(
    x: Sequence[float],
    y: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Weighted least squares of `ln y` on `ln x`, returning the slope with its uncertainty.

    Returns `fitted: False` and NaN estimates rather than raising when fewer than
    `MIN_POINTS` usable points survive the range and positivity masks: too few points is a
    property of the data, and a caller sweeping many scales needs the run to continue and
    the gap to be visible in the result. A *singular* design - every retained point at the
    same `x` - does raise, because that is a mistake in the call rather than a thin dataset.

    A slope without a standard error is not a measurement, so both are always present.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise PowerLawError("x and y must be the same length, got %d and %d"
                            % (x_arr.size, y_arr.size))
    w_arr = (np.ones_like(x_arr) if weights is None
             else np.asarray(weights, dtype=np.float64))
    if w_arr.shape != x_arr.shape:
        raise PowerLawError("weights must match x, got %d and %d"
                            % (w_arr.size, x_arr.size))

    positive = x_arr > 0
    lo = x_min if x_min is not None else (float(x_arr[positive].min())
                                          if np.any(positive) else 0.0)
    hi = x_max if x_max is not None else (float(x_arr.max()) if x_arr.size else 0.0)

    mask = ((x_arr >= lo) & (x_arr <= hi) & positive & (y_arr > 0)
            & np.isfinite(y_arr) & np.isfinite(x_arr) & (w_arr > 0))
    n = int(mask.sum())

    base: Dict[str, Any] = {"n_points": n, "x_min": float(lo), "x_max": float(hi),
                            "n_supplied": int(x_arr.size)}
    if n < MIN_POINTS:
        base.update({
            "fitted": False,
            "slope": float("nan"),
            "slope_standard_error": float("nan"),
            "intercept_ln_c": float("nan"),
            "r_squared": float("nan"),
            "reason": ("only %d usable points in [%.6g, %.6g]; a log-log slope needs at "
                       "least %d to have leverage, and a standard error needs the residual "
                       "degrees of freedom" % (n, lo, hi, MIN_POINTS)),
        })
        return base

    ln_x = np.log(x_arr[mask])
    ln_y = np.log(y_arr[mask])
    w = w_arr[mask]

    sw = float(np.sum(w))
    swx = float(np.sum(w * ln_x))
    swy = float(np.sum(w * ln_y))
    swxx = float(np.sum(w * ln_x * ln_x))
    swxy = float(np.sum(w * ln_x * ln_y))
    denom = sw * swxx - swx ** 2
    if denom <= 0:
        raise PowerLawError(
            "degenerate power-law fit: all %d retained points share the same x, so the "
            "design matrix is singular and no slope exists." % n)

    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw

    resid = ln_y - (slope * ln_x + intercept)
    ss_res = float(np.sum(w * resid ** 2))
    ybar = swy / sw
    ss_tot = float(np.sum(w * (ln_y - ybar) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma2 = ss_res / (n - 2)
    slope_se = math.sqrt(max(sigma2, 0.0) * sw / denom)

    base.update({
        "fitted": True,
        "slope": float(slope),
        "slope_standard_error": float(slope_se),
        "intercept_ln_c": float(intercept),
        "r_squared": float(r_squared),
        "reason": None,
    })
    return base


def compare_exponent_to_null(observed: float, null_exponents: Sequence[float],
                             label: str = "exponent") -> Dict[str, Any]:
    """Rule R2 made mechanical: an exponent becomes reportable only beside its null.

    Returns the observed exponent, the surrogate ensemble's mean and spread, a standardised
    effect size, and a two-sided empirical p-value using the `(1 + k) / (1 + n)` convention -
    the added one keeps the test exact in finite samples and stops a `p = 0` that no finite
    ensemble can support.

    `reportable` is `True` only when a non-empty ensemble was supplied. It is deliberately
    *not* conditioned on significance: a null result is reportable and often the point.
    """
    nulls = np.asarray([v for v in null_exponents if np.isfinite(v)], dtype=np.float64)
    if nulls.size == 0:
        return {
            "label": label,
            "observed": float(observed),
            "reportable": False,
            "n_surrogates": 0,
            "warnings": [
                "R2: an exponent on its own is not a finding. A fractional Brownian field "
                "gives a clean high-r-squared power law and contains no organisation, so "
                "alpha must be reported against alpha_surrogate with the effect size "
                "between them. No surrogate ensemble was supplied, so this exponent is not "
                "reportable."],
        }

    mean = float(nulls.mean())
    std = float(nulls.std(ddof=1)) if nulls.size > 1 else 0.0
    excursion = abs(float(observed) - mean)
    k = int(np.sum(np.abs(nulls - mean) >= excursion))
    p_value = (1.0 + k) / (1.0 + nulls.size)
    return {
        "label": label,
        "observed": float(observed),
        "surrogate_mean": mean,
        "surrogate_std": std,
        "effect_size": (float(observed) - mean) / std if std > 0 else float("nan"),
        "difference": float(observed) - mean,
        "p_value": float(p_value),
        "p_value_floor": 1.0 / (1.0 + nulls.size),
        "n_surrogates": int(nulls.size),
        "alternative": "two_sided",
        "reportable": True,
        "warnings": ([] if std > 0 else
                     ["the surrogate ensemble has zero spread, so no effect size is "
                      "defined; this usually means the statistic is degenerate on the null"]),
    }
