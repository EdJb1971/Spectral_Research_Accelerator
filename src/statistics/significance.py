"""Significance testing that is defendable: surrogate nulls, ESS, and FDR together.

Roadmap R1, R5, R12; defect D8. This module is deliberately the only place a p-value is
produced, because the three ways to get one wrong are independent and a result needs all
three handled at once:

1.  **No null model** - comparing a statistic to a threshold rather than to what the statistic
    does on structured-but-unrelated data (R1).
2.  **Treating dependent samples as independent** - autocorrelated frames inflate the
    effective sample size and therefore the significance (R12). Measured on the benchmark
    suite: AR(1) with phi = 0.85 rejects a true null **40%** of the time at alpha = 0.05.
3.  **No multiplicity control** - scanning many pairs and reporting the extremes (R5).

Every function here returns the assumptions it made along with the number, so a p-value cannot
be quoted without them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.statistics import surrogates as surrogate_module
from src.statistics.multiple_comparisons import adjust, check_power


def effective_sample_size(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """Bartlett effective sample size for correlating two autocorrelated series.

    ``n_eff = n (1 - r1 r2) / (1 + r1 r2)`` from the lag-1 autocorrelations. Exact for AR(1),
    first-order approximate otherwise - stated because presenting an approximate correction as
    exact would be the same sin R12 is about.
    """
    def lag1(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=np.float64)
        v = v - v.mean()
        denom = float(np.sum(v * v))
        if denom <= 0:
            return 0.0
        return float(np.sum(v[1:] * v[:-1]) / denom)

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    r1 = lag1(x)
    r2 = lag1(y) if y is not None else r1
    product = max(min(r1 * r2, 0.999), -0.999)
    return max(3.0, n * (1.0 - product) / (1.0 + product))


def correlation_test(x: Sequence[float], y: Sequence[float],
                     account_for_autocorrelation: bool = True) -> Dict[str, Any]:
    """Pearson correlation with a p-value that accounts for serial dependence.

    Returns both the naive and the ESS-corrected p-value. Both, deliberately: the ratio is how
    much serial dependence was inflating the result, and hiding it would remove the reader's
    ability to judge whether the correction mattered.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("x and y must be the same length; got %d and %d" % (x.size, y.size))
    if x.size < 4:
        return {"r": float("nan"), "p_value": float("nan"), "n": int(x.size),
                "note": "fewer than 4 paired samples; no correlation is estimable"}
    if np.std(x) == 0 or np.std(y) == 0:
        return {"r": float("nan"), "p_value": float("nan"), "n": int(x.size),
                "note": "one series is constant, so correlation is undefined"}

    r = float(np.corrcoef(x, y)[0, 1])
    n = float(x.size)
    n_eff = effective_sample_size(x, y) if account_for_autocorrelation else n

    def p_from(nn: float) -> float:
        if nn <= 2 or abs(r) >= 1.0:
            return 0.0
        t = r * math.sqrt((nn - 2.0) / (1.0 - r * r))
        return float(2.0 * stats.t.sf(abs(t), df=nn - 2.0))

    return {
        "r": r,
        "n": int(n),
        "n_effective": float(n_eff),
        "p_value": p_from(n_eff),
        "p_value_naive": p_from(n),
        "autocorrelation_inflation": (p_from(n_eff) / p_from(n)) if p_from(n) > 0 else 1.0,
        "assumptions": [
            "Pearson correlation assumes a linear, monotonic relationship",
            ("effective sample size %.1f of %d used (Bartlett lag-1 correction); exact for "
             "AR(1), approximate otherwise" % (n_eff, int(n)))
            if account_for_autocorrelation else
            "samples assumed independent - NOT valid for a time series (rule R12)",
        ],
    }


# --------------------------------------------------------------------- stationarity

# Lag-1 autocorrelation above which an FT surrogate test stops being trustworthy.
#
# **Calibrated against measured false-positive rate, not chosen for neatness.** 200 trials per
# phi on AR(1) series, phase-randomised surrogates, 99 draws, two-sided, alpha = 0.05:
#
#     phi    mean rho1   FPR     inflation
#     0.00   +0.005      0.050   1.0x
#     0.50   +0.492      0.045   0.9x
#     0.70   +0.687      0.060   1.2x
#     0.80   +0.785      0.110   2.2x   <- crosses 2x nominal
#     0.85   +0.834      0.155   3.1x
#     0.90   +0.883      0.225   4.5x
#     0.95   +0.931      0.390   7.8x
#     0.99   +0.967      0.665  13.3x
#
# The test is well calibrated up to rho1 ~ 0.69 and unusable beyond ~0.79, so the boundary is
# placed between them. A first draft used 0.90, which would have passed AR(1) phi=0.85 -
# a series whose true false-positive rate is 3.1x nominal - as trustworthy.
UNIT_ROOT_RHO = 0.75
#: Split-half variance ratio above which the variance is treated as non-stationary.
VARIANCE_RATIO_LIMIT = 3.0


def stationarity_check(x: Sequence[float]) -> Dict[str, Any]:
    """Screen a series for the stationarity that FT surrogate tests require.

    **Why this gate exists, measured.** Fourier surrogates assume the series is stationary:
    they preserve a *global* power spectrum, so if the local structure changes over the record
    the surrogates do not resemble the data and the test rejects for the wrong reason. The
    false-positive rate at alpha = 0.05 was measured on 200 trials per case:

    ==========================================  ==================  ===========
    series                                      FPR (nominal 0.05)  inflation
    ==========================================  ==================  ===========
    white noise (stationary)                    0.050               calibrated
    AR(1), phi = 0.70                           0.060               1.2x
    AR(1), phi = 0.80                           0.110               2.2x
    AR(1), phi = 0.85                           0.155               3.1x
    AR(1), phi = 0.95                           0.390               7.8x
    random walk (non-stationary)                0.775               15x
    ==========================================  ==================  ===========

    IAAFT reduces but does not remove the inflation (0.615 for the random walk, 0.135 for
    AR(1) 0.85), so choosing the better surrogate is not a fix. **A significant result on a
    near-unit-root series is not evidence of anything**, and since atmospheric series are
    routinely strongly red, this is not a corner case for this platform - it is the default
    situation.

    Three cheap diagnostics, no statsmodels dependency: lag-1 autocorrelation (proximity to a
    unit root), split-half variance ratio (variance non-stationarity), and a t-test on a linear
    trend (mean non-stationarity).
    """
    from scipy import stats

    arr = np.asarray(list(x), dtype=np.float64)
    n = arr.size
    if n < 16:
        return {"n": int(n), "is_stationary": False, "checks": {},
                "warnings": ["series too short (%d points) to assess stationarity" % n],
                "recommendation": "use at least 32 points, or a block bootstrap"}

    centred = arr - arr.mean()
    denom = float(np.sum(centred * centred))
    rho1 = float(np.sum(centred[1:] * centred[:-1]) / denom) if denom > 0 else 0.0

    half = n // 2
    v1, v2 = float(np.var(arr[:half])), float(np.var(arr[half:]))
    ratio = (max(v1, v2) / min(v1, v2)) if min(v1, v2) > 0 else float("inf")

    # The trend test is corrected for serial dependence. OLS assumes independent residuals,
    # so on autocorrelated data it reports a "significant trend" that is really the spurious
    # regression effect - it flagged AR(1) phi=0.85 at p = 1.7e-14 with no trend present. The
    # series would still (correctly) fail the unit-root check, but the *reason* given would
    # have been wrong, and a diagnostic that misdiagnoses is worse than one that abstains.
    t_index = np.arange(n, dtype=np.float64)
    slope_res = stats.linregress(t_index, arr)
    n_eff = effective_sample_size(arr)
    if abs(slope_res.rvalue) >= 1.0 or n_eff <= 2:
        trend_p = 0.0
    else:
        t_stat = slope_res.rvalue * math.sqrt((n_eff - 2.0) /
                                              (1.0 - slope_res.rvalue ** 2))
        trend_p = float(2.0 * stats.t.sf(abs(t_stat), df=n_eff - 2.0))

    checks = {
        "lag1_autocorrelation": rho1,
        "near_unit_root": bool(rho1 > UNIT_ROOT_RHO),
        "split_half_variance_ratio": ratio,
        "variance_nonstationary": bool(ratio > VARIANCE_RATIO_LIMIT),
        "trend_p_value": trend_p,
        "trend_significant": bool(trend_p < 0.01),
    }
    problems = [k for k in ("near_unit_root", "variance_nonstationary",
                            "trend_significant") if checks[k]]
    warnings: List[str] = []
    if problems:
        warnings.append(
            "This series fails %s. Fourier surrogate tests assume stationarity and are "
            "badly anti-conservative without it - measured false-positive rates of 0.20 "
            "(AR(1) phi=0.85), 0.60 (phi=0.98) and 0.78 (random walk) against a nominal "
            "0.05. Treat any 'significant' result here as uninterpretable."
            % ", ".join(problems))
    return {
        "n": int(n),
        "is_stationary": not problems,
        "checks": checks,
        "failed": problems,
        "warnings": warnings,
        "recommendation": None if not problems else (
            "difference the series (np.diff) to remove a trend or unit root, or use "
            "method='block_bootstrap' / 'circular_shift', which do not assume a global "
            "spectrum. Re-check stationarity afterwards rather than assuming."),
    }


def surrogate_p_value(observed: float, null_distribution: Sequence[float],
                      alternative: str = "greater") -> Dict[str, Any]:
    """Empirical p-value from a surrogate ensemble, as ``(1 + k) / (1 + n)``.

    The added one is not a fudge: it makes the test exact in finite samples (the observed
    statistic is itself one draw under the null) and it prevents ``p = 0``, which would claim
    infinite evidence from a finite ensemble.
    """
    if alternative not in ("greater", "less", "two_sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two_sided'")
    null = np.asarray(list(null_distribution), dtype=np.float64)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0:
        raise ValueError("the surrogate ensemble is empty; no p-value can be computed")

    if alternative == "greater":
        k = int(np.sum(null >= observed))
    elif alternative == "less":
        k = int(np.sum(null <= observed))
    else:
        centre = float(np.median(null))
        k = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))

    p = (1.0 + k) / (1.0 + n)
    std = float(np.std(null))
    return {
        "observed": float(observed),
        "p_value": float(p),
        "n_surrogates": n,
        "n_exceeding": k,
        "p_value_floor": 1.0 / (1.0 + n),
        "null_mean": float(np.mean(null)),
        "null_std": std,
        "z_score": float((observed - np.mean(null)) / std) if std > 0 else float("nan"),
        "alternative": alternative,
    }


def surrogate_test(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    method: str = "iaaft",
    n_surrogates: int = 999,
    seed: int = 20260819,
    alternative: str = "greater",
    **surrogate_kwargs: Any,
) -> Dict[str, Any]:
    """Test a statistic against a surrogate ensemble, carrying the null's identity along.

    The result records *which* null was used and what it preserves, because "p = 0.004" means
    something different against a phase-randomised null than against a circular-shift null,
    and a reader cannot interpret one without the other.
    """
    arr = np.asarray(data)
    observed = float(statistic(arr))

    # Gate the test on the assumption it depends on. Reported, not raised: a caller may have
    # good reason to proceed (an exploratory look, or a statistic insensitive to the trend),
    # but the result must not be quotable without the caveat attached to it.
    stationarity = None
    if arr.ndim == 1 and method in ("phase_randomise", "aaft", "iaaft"):
        stationarity = stationarity_check(arr)

    ensemble = surrogate_module.generate(data, method=method, n=n_surrogates, seed=seed,
                                         **surrogate_kwargs)
    null = [float(statistic(member)) for member in ensemble["members"]]
    result = surrogate_p_value(observed, null, alternative=alternative)
    result.update({
        "surrogate_method": method,
        "null_preserves": ensemble["preserves"],
        "null_hypothesis": (
            "the observed statistic is no more extreme than expected for data preserving: %s"
            % ", ".join(ensemble["preserves"])),
        "root_seed": ensemble["root_seed"],
        "stationarity": stationarity,
        "valid": True if stationarity is None else stationarity["is_stationary"],
        "warnings": [] if stationarity is None else list(stationarity["warnings"]),
    })
    return result


def screen(
    tests: Sequence[Dict[str, Any]],
    alpha: float = 0.05,
    method: str = "benjamini_yekutieli",
    n_tests: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply multiplicity control to a family of tests and report what survives.

    ``benjamini_yekutieli`` is the default rather than plain BH. Spectral and cross-scale
    statistics have an unknown - and not necessarily positive - dependence structure, and BY is
    the procedure that remains valid without assuming otherwise. It costs power; the honest
    trade is to say so rather than to assume PRDS because it is more convenient.

    Each input dict must carry ``p_value`` and ``label``. The output is the same list with
    ``q_value`` and ``significant`` attached, plus a power check: if the ensemble was too small
    to reject anything after correction, that is reported instead of being mistaken for a clean
    negative.
    """
    if not tests:
        return {"n_tests": 0, "n_significant": 0, "results": [], "correction": None,
                "warnings": ["no tests were supplied; nothing was screened"]}

    usable = [t for t in tests if t.get("p_value") is not None
              and np.isfinite(t.get("p_value", np.nan))]
    skipped = len(tests) - len(usable)
    if not usable:
        return {"n_tests": len(tests), "n_significant": 0, "results": [],
                "correction": None,
                "warnings": ["every test had a non-finite p-value; nothing was screened"]}

    family_size = n_tests if n_tests is not None else len(tests)
    correction = adjust([t["p_value"] for t in usable], method=method, alpha=alpha,
                        n_tests=family_size,
                        labels=[str(t.get("label", i)) for i, t in enumerate(usable)])

    results = []
    for test, q, rejected in zip(usable, correction["adjusted"], correction["rejected"]):
        enriched = dict(test)
        enriched["q_value"] = q
        enriched["significant"] = bool(rejected)
        results.append(enriched)

    warnings = list(correction["warnings"])
    if skipped:
        warnings.append(
            "%d of %d tests had a non-finite p-value and were excluded from the correction "
            "but counted in the family size, which is the conservative choice."
            % (skipped, len(tests)))

    smallest_ensemble = min((t.get("n_surrogates", 0) for t in usable
                             if t.get("n_surrogates")), default=0)
    power = None
    if smallest_ensemble:
        power = check_power(smallest_ensemble, family_size, alpha=alpha, method=method)
        if power["warning"]:
            warnings.append(power["warning"])

    return {
        "n_tests": family_size,
        "n_screened": len(usable),
        "n_significant": correction["n_rejected"],
        "alpha": alpha,
        "correction": {k: correction[k] for k in
                       ("method", "assumption", "n_tests", "min_adjusted")},
        "power_check": power,
        "results": results,
        "warnings": warnings,
    }
