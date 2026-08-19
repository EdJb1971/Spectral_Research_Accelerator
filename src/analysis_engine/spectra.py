"""Radial power spectra in physical wavenumber, and power-law fitting.

Roadmap T3.5.13 (defect D13, standard E3) and the vectorised binning half of T3.5.20
(defect D17). Two distinct defects are fixed here and both changed reported numbers:

**D13 - pixel wavenumbers.** The previous ``compute_radial_psd`` binned on integer *pixel*
radius. A circle in pixel space is an ellipse in physical space whenever ``dx != dy``, and a
32x32 ERA5 patch at 60 degrees north has an aspect ratio of 1.79.

The consequence was measured rather than assumed, and the measurement corrected the
expectation: for a *pure power law* pixel binning preserves the slope **exactly** (the
angular anisotropy factor is scale-free, so only the intercept moves - slope error at
aspect 4 was +0.001). What it destroys is the physical meaning of ``k`` and the *shape* of
the spectrum: a Gaussian spectral peak on an aspect-3 grid was smeared from fractional
half-width 0.020 to 1.935. Since a `ScaleSignature` (T4C.1) is built from spectral features
and is compared across latitudes and times, that is the more damaging failure. See
``test_pixel_binning_preserves_slope_but_destroys_spectral_features``.

**D26 - the spectrum convention was off by one exponent.** The annulus mean of ``|F(k)|^2``
estimates the *2D spectral density* ``S(k)``. The regime targets the platform compared it
against - 5/3 for Kolmogorov, 3 for Charney/Kraichnan - are defined for the *1D
isotropic energy spectrum* ``E(k)``, and in two dimensions

.. math:: E(k) = 2\\pi k\\, S(k)

so ``E ~ k^-5/3`` means ``S ~ k^-8/3``. Measured consequence: a synthetic field built with a
textbook Kolmogorov spectrum was reported by the old code as
*"Charney/Kraichnan 2D Enstrophy Cascade"* - the opposite physical regime. Every regime
label the platform emitted before this module was wrong by one exponent. Both conventions
are now computed, each is labelled, and interpretation always happens on ``E``.

**Estimator choices, declared (standard E8).** A periodogram annulus average is a scaled
chi-squared variate, so ``log`` of it is heteroscedastic: bins at low ``k`` contain few
modes and are far noisier. The fit is therefore weighted by mode count, and the
multiplicative log bias ``log(m) - psi(m)`` is available as an explicit, measured option
rather than applied invisibly - see :func:`fit_power_law`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.physical_core.grid import GridSpec, GridError

#: Reference turbulence regimes, expressed in the 1D energy-spectrum convention E(k).
REGIMES = (
    (5.0 / 3.0, 0.30, "Kolmogorov 3D / mesoscale kinetic energy cascade (E ~ k^-5/3)"),
    (3.0, 0.40, "Charney / Kraichnan 2D enstrophy cascade (E ~ k^-3)"),
)


def _as_tensor_and_grid(field, grid: Optional[GridSpec]) -> Tuple[torch.Tensor, GridSpec]:
    if isinstance(field, torch.Tensor):
        data = field
        gspec = grid if grid is not None else GridSpec.pixel(tuple(data.shape))
    else:
        data = field.data
        gspec = grid if grid is not None else getattr(field, "grid", None)
        if gspec is None:
            gspec = GridSpec.pixel(tuple(data.shape))
    if data.dim() != 2:
        raise GridError("radial spectra need a 2D field, got shape %r" % (tuple(data.shape),))
    if tuple(data.shape) != tuple(gspec.shape):
        raise GridError(
            "field shape %r does not match grid shape %r (%s)"
            % (tuple(data.shape), tuple(gspec.shape), gspec.describe())
        )
    return data, gspec


def _window_2d(shape: Tuple[int, int], name: str, device, dtype) -> Tuple[torch.Tensor, float]:
    """Separable analysis window plus its power-normalisation factor.

    A regional crop is not periodic, so an unwindowed FFT has a discontinuity at the wrap
    that leaks broadband power and *flattens* a steep spectrum. The correction factor is
    ``mean(w^2)``, which restores total variance; it does not restore the resolution lost
    to the window's main-lobe width, and that is why ``k_min`` for a fit should stay a few
    bins above the fundamental.
    """
    H, W = shape
    if name in (None, "none"):
        w = torch.ones(shape, device=device, dtype=dtype)
    elif name == "hann":
        wy = torch.hann_window(H, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(W, periodic=False, device=device, dtype=dtype)
        w = torch.outer(wy, wx)
    elif name == "hamming":
        wy = torch.hamming_window(H, periodic=False, device=device, dtype=dtype)
        wx = torch.hamming_window(W, periodic=False, device=device, dtype=dtype)
        w = torch.outer(wy, wx)
    else:
        raise GridError("unknown window %r; expected 'none', 'hann' or 'hamming'" % (name,))
    return w, float(torch.mean(w ** 2))


def radial_power_spectrum(
    field,
    grid: Optional[GridSpec] = None,
    units: str = "rad_per_m",
    nbins: Optional[int] = None,
    convention: str = "energy_1d",
    detrend: str = "mean",
    window: str = "none",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Isotropically-averaged power spectrum on a physical wavenumber axis.

    Parameters
    ----------
    units
        Wavenumber units for the returned axis: ``rad_per_m`` (default), ``cycles_per_m``,
        ``cycles_per_km``, ``rad_per_km``, or ``legacy_pixel``. ``legacy_pixel`` reproduces
        the pre-D13 integer-radius binning *exactly*, kept so the old numbers stay
        reproducible and so the bias it introduces can be measured rather than asserted.
    convention
        ``energy_1d`` for ``E(k)`` with ``integral E dk = variance`` (the convention in which
        Kolmogorov is 5/3), or ``density_2d`` for the 2D density ``S(k)``. ``E = 2 pi k S``.
    detrend
        ``mean`` removes the area-weighted mean (a non-zero mean otherwise dumps power into
        the DC bin and leaks through the window), ``none`` leaves the field alone.
    window
        ``none``, ``hann`` or ``hamming``. Use a window for a non-periodic regional crop.

    Returns a dict whose ``k`` and ``power`` are plain lists, alongside the mode counts, the
    units, the isotropy report, and the honest fit ceiling ``k_max_isotropic``.
    """
    if convention not in ("energy_1d", "density_2d"):
        raise GridError(
            "convention must be 'energy_1d' (E(k), the 5/3 convention) or 'density_2d' "
            "(S(k)); got %r" % (convention,)
        )
    data, gspec = _as_tensor_and_grid(field, grid)
    work = data.to(dtype)
    H, W = work.shape
    warnings: List[str] = []

    if detrend == "mean":
        w_area = gspec.area_weights(device=work.device, dtype=dtype)
        work = work - torch.sum(work * w_area)
    elif detrend != "none":
        raise GridError("detrend must be 'mean' or 'none', got %r" % (detrend,))

    win, win_power = _window_2d((H, W), window, work.device, dtype)
    work = work * win

    legacy = (units == "legacy_pixel")

    fft = torch.fft.fftshift(torch.fft.fft2(work))
    modulus_sq = (fft.real ** 2 + fft.imag ** 2)

    if legacy:
        # Exact pre-D13 behaviour, preserved for comparison. Integer pixel radius,
        # |F|^2 / (H*W), annulus mean, bins 0..min(H,W)//2.
        power_field = modulus_sq / (H * W)
        y = torch.arange(H, dtype=dtype, device=work.device) - H // 2
        x = torch.arange(W, dtype=dtype, device=work.device) - W // 2
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        kmag = torch.sqrt(gy ** 2 + gx ** 2)
        idx = torch.round(kmag).to(torch.long)
        n_bins = int(min(H, W) // 2) + 1
        valid = idx < n_bins
        k_units_label = "cycles per domain (pixel index)"
        k_max_iso = float(n_bins - 1)
    else:
        # Physical spectral density S(k) normalised so that the discrete integral
        # sum S * dky * dkx equals the mean square of the (detrended, windowed) field.
        dy_m = gspec.representative_dy_metres()
        dx_m = gspec.representative_dx_metres()
        two_pi_sq = (2.0 * math.pi) ** 2
        power_field = modulus_sq * (dy_m * dx_m) / (H * W * two_pi_sq)
        power_field = power_field / win_power

        kmag = gspec.wavenumber_magnitude(units="rad_per_m", device=work.device,
                                          dtype=dtype, shifted=True)
        k_max_iso_rad = gspec.isotropic_k_max("rad_per_m")
        n_bins = int(nbins) if nbins else int(min(H, W) // 2)
        if n_bins < 2:
            raise GridError(
                "a radial spectrum needs at least 2 bins; this %dx%d field gives %d. "
                "Use a field at least 8x8, or pass nbins explicitly." % (H, W, n_bins)
            )
        dk = k_max_iso_rad / n_bins
        idx = torch.floor(kmag / dk).to(torch.long)
        valid = (idx < n_bins) & (kmag > 0)
        k_units_label = gspec.wavelength_units(units)
        k_max_iso = k_max_iso_rad

    # --- vectorised annulus reduction (D17): three bincounts, no Python loop over bins ---
    flat_idx = idx[valid].reshape(-1)
    counts = torch.bincount(flat_idx, minlength=n_bins).to(dtype)
    p_sum = torch.bincount(flat_idx, weights=power_field[valid].reshape(-1), minlength=n_bins)
    k_sum = torch.bincount(flat_idx, weights=kmag[valid].reshape(-1), minlength=n_bins)

    occupied = counts > 0
    p_mean = torch.zeros_like(counts)
    k_mean = torch.zeros_like(counts)
    p_mean[occupied] = p_sum[occupied] / counts[occupied]
    # Bin centre is the *mean |k| of the modes in the bin*, not the nominal bin midpoint.
    # At low k a bin holds few modes whose mean can sit well away from the midpoint, and a
    # log-log fit is sensitive to exactly that.
    k_mean[occupied] = k_sum[occupied] / counts[occupied]

    k_out = k_mean[occupied]
    p_out = p_mean[occupied]
    c_out = counts[occupied]

    if legacy:
        # Legacy path never applied a 2 pi k factor, so it is a density-like quantity in
        # arbitrary units. Refuse to relabel it as an energy spectrum.
        if convention == "energy_1d":
            warnings.append(
                "units='legacy_pixel' reproduces the pre-D13 annulus mean of |F|^2, which "
                "is a 2D density S(k) in pixel units. It cannot be converted to the 1D "
                "energy convention E(k) because that needs a physical k, so convention "
                "has been reported as 'density_2d'. Any comparison against the 5/3 or 3 "
                "targets in this mode is off by one exponent - that was defect D26."
            )
        effective_convention = "density_2d"
    else:
        effective_convention = convention
        if convention == "energy_1d":
            p_out = 2.0 * math.pi * k_out * p_out
        # Convert the k axis (and the density with it, so the integral is preserved).
        if units != "rad_per_m":
            scale = (gspec.wavenumber_axes(units=units, shifted=False)[1].abs().max()
                     / gspec.wavenumber_axes(units="rad_per_m", shifted=False)[1].abs().max())
            scale = float(scale)
            k_out = k_out * scale
            p_out = p_out / scale
            k_max_iso = k_max_iso * scale

    iso = gspec.anisotropy()
    warnings.extend(iso["warnings"])
    if legacy and not iso["is_isotropic"]:
        warnings.append(
            "This grid is anisotropic (dy/dx = %.3f) *and* legacy_pixel binning was "
            "requested. Those two together are the D13 defect exactly: the fitted slope "
            "will be biased. Use a physical wavenumber unit." % iso["aspect_ratio"]
        )

    # Both the k label and the power label come from the same call, so they cannot drift
    # apart (one saying metres while the other says pixels).
    var_units = gspec.variable_units or "value"
    if legacy:
        power_units = "%s^2 (arbitrary, pixel convention)" % var_units
    elif effective_convention == "energy_1d":
        power_units = "%s^2 / (%s)" % (var_units, k_units_label)
    else:
        power_units = "%s^2 / (%s)^2" % (var_units, k_units_label)

    # How much of the field's variance the returned bins actually account for. Reported
    # because it is otherwise invisible and easily misread: numerically integrating the
    # returned E(k) recovered only 82% of the variance of a red-noise field, which looks
    # like a normalisation bug and is not one. The 2D density identity is exact
    # (sum S dky dkx / var = 1.000000000000, test_density_satisfies_parseval); the shortfall
    # is quadrature - trapezoid over linear bins under-resolves a steep E(k) near the
    # first bin, and modes in the k-square corners beyond k_max_isotropic are excluded by
    # design. This field states the exact captured fraction so neither is mistaken for the
    # other.
    if legacy:
        captured = float("nan")
        variance_total = float("nan")
    else:
        ky_ax, kx_ax = gspec.wavenumber_axes("rad_per_m", device=work.device, dtype=dtype,
                                             shifted=False)
        dk2 = float(ky_ax[1] - ky_ax[0]) * float(kx_ax[1] - kx_ax[0])             if (H > 1 and W > 1) else 1.0
        variance_total = float(torch.sum(power_field)) * dk2
        binned = torch.zeros_like(power_field)
        binned[valid] = power_field[valid]
        in_bins = valid & (idx < n_bins)
        captured_abs = float(torch.sum(power_field[in_bins])) * dk2
        captured = captured_abs / variance_total if variance_total > 0 else float("nan")

    return {
        "k": [float(v) for v in k_out],
        "power": [float(v) for v in p_out],
        "counts": [int(v) for v in c_out],
        "variance_captured_fraction": captured,
        "variance_total": variance_total,
        "k_units": k_units_label,
        "power_units": power_units,
        "convention": effective_convention,
        "convention_note": (
            "E(k) = 2*pi*k*S(k) in 2D; Kolmogorov is E ~ k^-5/3, i.e. S ~ k^-8/3"
        ),
        "k_max_isotropic": k_max_iso,
        "n_bins_requested": n_bins,
        "detrend": detrend,
        "window": window,
        "window_power_correction": win_power,
        "isotropy": iso,
        "is_physical": gspec.is_physical and not legacy,
        "grid": gspec.to_provenance(),
        "warnings": warnings,
    }


def _digamma(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import digamma  # type: ignore
        return digamma(x)
    except Exception:  # pragma: no cover - scipy is a declared dependency
        # Asymptotic series, adequate for m >= 2 which is where it is used.
        return np.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x ** 2)


def fit_power_law(
    k: Sequence[float],
    power: Sequence[float],
    counts: Optional[Sequence[float]] = None,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    convention: str = "energy_1d",
    weighting: str = "counts",
    log_bias_correction: bool = False,
) -> Dict[str, Any]:
    """Fit ``P(k) = C k^-beta`` by least squares in log-log space.

    Returns ``slope_beta`` with a **standard error**, ``r_squared``, the exponent expressed
    in *both* conventions, and the list of assumptions the number depends on. A slope
    without an uncertainty is not a measurement, and rule R2 is explicit that a power-law
    fit is not by itself a finding.

    ``weighting='counts'`` performs weighted least squares with weights proportional to the
    number of Fourier modes per bin. Justification: an annulus average of ``m`` modes is
    approximately ``chi^2_{2m}/2m`` distributed, whose log has variance ``psi'(m) ~ 1/m``, so
    inverse-variance weighting is weight ``~ m``. Unweighted OLS lets the handful of noisy
    low-``k`` bins dominate a fit that spans two decades.

    ``log_bias_correction`` adds ``log(m) - psi(m)`` to each log-power before fitting,
    removing the downward bias of the log of a chi-squared average. It is **off by default**
    because it is a measured trade-off rather than a free improvement: it is exact for the
    intercept but, since ``m`` grows with ``k``, it introduces a weak ``1/k`` term that can
    perturb the slope. ``test_log_bias_correction_is_a_measured_tradeoff`` records the size
    of both effects.
    """
    if convention not in ("energy_1d", "density_2d"):
        raise GridError("convention must be 'energy_1d' or 'density_2d', got %r" % (convention,))
    if weighting not in ("counts", "none"):
        raise GridError("weighting must be 'counts' or 'none', got %r" % (weighting,))

    k_arr = np.asarray(k, dtype=np.float64)
    p_arr = np.asarray(power, dtype=np.float64)
    c_arr = (np.asarray(counts, dtype=np.float64) if counts is not None
             else np.ones_like(k_arr))

    if k_arr.shape != p_arr.shape:
        raise GridError("k and power must be the same length, got %d and %d"
                        % (k_arr.size, p_arr.size))

    lo = k_min if k_min is not None else float(k_arr[k_arr > 0].min()) if np.any(k_arr > 0) else 0.0
    hi = k_max if k_max is not None else float(k_arr.max())

    mask = (k_arr >= lo) & (k_arr <= hi) & (k_arr > 0) & (p_arr > 0) & np.isfinite(p_arr)
    n = int(mask.sum())

    assumptions = [
        "isotropy: power is averaged over annuli in |k|, so an anisotropic field is "
        "summarised by its angular mean and any directional structure is discarded",
        "a single power law holds across [k_min, k_max]; no break point is fitted or tested",
        "convention: exponent reported for %s" % convention,
    ]

    if n < 4:
        return {
            "slope_beta": float("nan"),
            "slope_standard_error": float("nan"),
            "intercept_ln_c": float("nan"),
            "r_squared": float("nan"),
            "n_points": n,
            "k_min": lo,
            "k_max": hi,
            "convention": convention,
            "beta_energy_1d": float("nan"),
            "beta_density_2d": float("nan"),
            "regime_interpretation": (
                "Not fitted: only %d usable spectral bins in [%.4g, %.4g]. A log-log slope "
                "needs at least 4 to have any leverage, and a standard error needs the "
                "residual degrees of freedom. Widen the band, increase the field size, or "
                "reduce nbins." % (n, lo, hi)
            ),
            "assumptions": assumptions,
            "weighting": weighting,
            "log_bias_corrected": False,
        }

    x = np.log(k_arr[mask])
    y = np.log(p_arr[mask])
    m = np.maximum(c_arr[mask] / 2.0, 1.0)   # complex modes per bin, floored

    corrected = False
    if log_bias_correction:
        y = y + (np.log(m) - _digamma(m))
        corrected = True

    w = m if weighting == "counts" else np.ones_like(x)

    sw = np.sum(w)
    swx = np.sum(w * x)
    swy = np.sum(w * y)
    swxx = np.sum(w * x * x)
    swxy = np.sum(w * x * y)
    denom = sw * swxx - swx ** 2
    if denom <= 0:
        raise GridError(
            "degenerate power-law fit: all %d retained bins share the same wavenumber, so "
            "the design matrix is singular." % n
        )
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw

    resid = y - (slope * x + intercept)
    ss_res_w = float(np.sum(w * resid ** 2))
    ybar_w = swy / sw
    ss_tot_w = float(np.sum(w * (y - ybar_w) ** 2))
    r_squared = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float("nan")

    sigma2 = ss_res_w / (n - 2)
    slope_se = math.sqrt(sigma2 * sw / denom)

    beta = -slope
    beta_energy = beta if convention == "energy_1d" else beta - 1.0
    beta_density = beta + 1.0 if convention == "energy_1d" else beta

    interpretation = _interpret(beta_energy, slope_se, r_squared)

    return {
        "slope_beta": float(beta),
        "slope_standard_error": float(slope_se),
        "intercept_ln_c": float(intercept),
        "r_squared": float(r_squared),
        "n_points": n,
        "k_min": float(lo),
        "k_max": float(hi),
        "convention": convention,
        "beta_energy_1d": float(beta_energy),
        "beta_density_2d": float(beta_density),
        "regime_interpretation": interpretation,
        "assumptions": assumptions,
        "weighting": weighting,
        "log_bias_corrected": corrected,
    }


def _interpret(beta_energy: float, slope_se: float, r_squared: float) -> str:
    """Name a turbulence regime only when the data actually distinguish it.

    Interpretation is on the 1D energy exponent (defect D26). A regime is named only when
    the fit is tight *and* the reference exponent lies within roughly two standard errors,
    so a noisy slope that happens to land near 5/3 is reported as inconclusive rather than
    as a cascade.
    """
    if not math.isfinite(beta_energy):
        return "Not fitted"
    if math.isfinite(r_squared) and r_squared < 0.8:
        return ("No single power law: E(k) exponent %.2f +/- %.2f but R^2 = %.2f, so the "
                "spectrum is not straight in log-log over this band. Do not read a cascade "
                "regime off this fit." % (beta_energy, slope_se, r_squared))

    tol_se = 2.0 * slope_se if math.isfinite(slope_se) else 0.0
    for reference, tol_fixed, label in REGIMES:
        if abs(beta_energy - reference) <= max(tol_fixed, tol_se):
            return ("%s; measured E(k) exponent %.2f +/- %.2f (R^2 = %.3f)"
                    % (label, beta_energy, slope_se, r_squared))
    if beta_energy < 0.7:
        return ("Flat spectrum: E(k) exponent %.2f +/- %.2f. Consistent with white noise or "
                "a numerical artefact rather than a cascade."
                % (beta_energy, slope_se))
    return ("Intermediate: E(k) exponent %.2f +/- %.2f (R^2 = %.3f), between the 5/3 and 3 "
            "reference regimes and not within tolerance of either."
            % (beta_energy, slope_se, r_squared))


def spectral_slope(
    field,
    grid: Optional[GridSpec] = None,
    units: str = "rad_per_m",
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    **spectrum_kwargs: Any,
) -> Dict[str, Any]:
    """Convenience: radial spectrum then power-law fit, carrying the metadata through.

    ``k_max`` defaults to ``k_max_isotropic``, the largest wavenumber resolved on *both*
    axes, rather than to the longest axis's Nyquist - beyond that limit an annulus is only
    partly sampled and its average is biased by the missing sectors.
    """
    spec = radial_power_spectrum(field, grid=grid, units=units, **spectrum_kwargs)
    ks = spec["k"]
    if not ks:
        raise GridError("radial spectrum produced no occupied bins for this field")
    lo = k_min if k_min is not None else ks[min(2, len(ks) - 1)]
    hi = k_max if k_max is not None else spec["k_max_isotropic"]
    fit = fit_power_law(ks, spec["power"], counts=spec["counts"], k_min=lo, k_max=hi,
                        convention=spec["convention"])
    fit["spectrum"] = spec
    fit["k_units"] = spec["k_units"]
    fit["warnings"] = spec["warnings"]
    return fit
