"""Undecimated (stationary / a trous) 2D wavelet transform.

Task T3.5.7. This is the transform the Phase 4 spectral-feature layer is built on, and it
exists because the decimated DWT in `transforms.py` is unsuitable for two specific reasons:

1.  **Every scale keeps the parent grid shape.** A decimated pyramid puts scale *n* on a
    different grid from the field it came from, so `CoefficientField` (T4B.1) would need
    per-scale coordinate bookkeeping and cross-scale spatial reasoning would be an
    interpolation guess. Here `coeffs[level][band].shape == field.data.shape` for every
    level, so coefficients inherit the parent field's coordinates directly. That is also
    what makes T4F.5 evidence projection — mapping a coefficient structure back to the
    grid cells that produced it — exact rather than approximate.

2.  **Shift invariance.** Because nothing is subsampled, translating the input translates
    the coefficients. The decimated Haar transform fails this badly: measured on a sharp
    front translated 0-7 px, its subband energy varies by ~153% (see
    `test_dwt_shift_variance_is_measurable`). Feature tracking in Phase 4D cannot be built
    on a transform that reports births and deaths caused by grid parity.

The price is redundancy: one 2D level produces four full-size bands, so the representation
is 4x the input per level rather than 1x. That is the cost that buys tracking, and it is
why the `ArtifactStore` (T4A.3) exists.

Boundary handling
-----------------
`mode="periodic"` (default) uses circular convolution, which is **exactly invertible** and
is the correct choice for a globally periodic longitude axis. `mode="reflect"` avoids
wrap-around contamination on a regional crop but is **not** exactly invertible; it is
offered for analysis only and `inverse_swt2d` will refuse it. Note that either way, R13
requires excluding a scale-dependent margin before drawing conclusions near an edge — see
`valid_interior_halfwidth`.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.physical_core.field import PhysicalField

# Orthonormal analysis filter pairs. Kept deliberately small and explicit; the genuine
# Kingsbury q-shift filters for the DTCWT are a separate matter (T3.5.6 / defect D1).
_FILTERS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    # Full double precision, transcribed from PyWavelets' dec_lo / dec_hi
    # (pywt.Wavelet(name)) so that orthonormality holds to machine epsilon rather than to
    # the ~1e-11 that truncated literals gave. PyWavelets is a TEST-only dependency; these
    # values are vendored deliberately so the runtime has no dependency on it.
    # Verified: sum(h**2) - 1 == 0.0 exactly for db2 and db3, 2.2e-16 for haar.
    "haar": ((0.7071067811865476, 0.7071067811865476),
             (-0.7071067811865476, 0.7071067811865476)),
    "db2": ((-0.12940952255126037, 0.2241438680420134,
             0.8365163037378079, 0.48296291314453416),
            (-0.48296291314453416, 0.8365163037378079,
             -0.2241438680420134, -0.12940952255126037)),
    "db3": ((0.03522629188570953, -0.08544127388202666, -0.13501102001025458,
             0.45987750211849154, 0.8068915093110925, 0.33267055295008263),
            (-0.33267055295008263, 0.8068915093110925, -0.45987750211849154,
             -0.13501102001025458, 0.08544127388202666, 0.03522629188570953)),
}

AVAILABLE_WAVELETS = tuple(_FILTERS.keys())

# Per-2D-level tight-frame constant. With orthonormal h, g the 1D undecimated operator
# satisfies H*H + G*G = 2I, so the separable 2D operator gives 4I. Verified numerically by
# test_swt_frame_constant.
_FRAME_CONSTANT_2D = 4.0


def get_filters(
    wavelet: str, device: torch.device = None, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analysis low-pass and high-pass filters for `wavelet`.

    dtype is explicit: numpy scalars promote to float64 and silently break conv against
    float32 fields (that was defect D23 in the decimated transform).
    """
    key = wavelet.lower()
    if key not in _FILTERS:
        raise ValueError(
            "Unknown wavelet %r. Available: %s" % (wavelet, ", ".join(AVAILABLE_WAVELETS))
        )
    h, g = _FILTERS[key]
    return (
        torch.tensor(h, device=device, dtype=dtype),
        torch.tensor(g, device=device, dtype=dtype),
    )


def dilate_filter(f: torch.Tensor, dilation: int) -> torch.Tensor:
    """Insert `dilation - 1` zeros between taps (the 'a trous' step).

    Level *j* uses dilation 2**(j-1), which is what moves the filter's passband down an
    octave without discarding samples.
    """
    if dilation <= 1:
        return f
    out = torch.zeros((f.numel() - 1) * dilation + 1, device=f.device, dtype=f.dtype)
    out[::dilation] = f
    return out


def filter_support(wavelet: str, level: int) -> int:
    """Full parent-grid support of the *recursive* coefficient at ``level``.

    Level ``j`` is not produced by one dilated filter acting on the original field: it acts on
    the low-pass output of all preceding levels. Supports therefore accumulate as
    ``1 + (L-1) * sum(2**m, m=0..j-1)``. The former single-stage formula silently discarded
    that inherited support (D44).
    """
    if level < 1:
        raise ValueError("level must be >= 1, got %r" % level)
    h, _ = _FILTERS[wavelet.lower()]
    return 1 + (len(h) - 1) * (2 ** level - 1)


def valid_interior_halfwidth(wavelet: str, level: int) -> int:
    """Margin (per side) contaminated by the boundary at `level`, per rule R13.

    Feature detection and any statistic must exclude this margin, and a crop must be
    large enough to leave a usable interior: `interior = N - 2 * halfwidth`. Coarse levels
    exclude far more than fine ones, which is why a 64x64 crop cannot support level-4
    analysis at all.
    """
    return filter_support(wavelet, level) // 2


def circular_filter_1d(x: torch.Tensor, f: torch.Tensor, dim: int,
                       adjoint: bool = False) -> torch.Tensor:
    """Circular convolution (or its adjoint) along one axis, via FFT.

    Done in the frequency domain deliberately: circular convolution is exact there and the
    adjoint is unambiguous (multiply by the conjugate). An earlier spatial-domain attempt
    got the analysis right but the synthesis anchoring wrong, which is exactly the class of
    off-by-one that a reconstruction test catches and a shape test does not.
    """
    n = x.shape[dim]
    if f.numel() > n:
        raise ValueError(
            "Filter support (%d) exceeds axis length (%d); reduce `levels` or use a larger field."
            % (f.numel(), n)
        )
    padded = torch.nn.functional.pad(f, (0, n - f.numel()))
    F = torch.fft.fft(padded, n=n)
    if adjoint:
        F = torch.conj(F)
    shape = [1] * x.ndim
    shape[dim] = n
    return torch.fft.ifft(torch.fft.fft(x, dim=dim) * F.view(shape), dim=dim).real


# Backwards-compatible private alias. The training representation imports the public function so
# the analytical and batched paths cannot acquire different phase conventions.
_circular_filter_1d = circular_filter_1d


def _reflect_filter_1d(x: torch.Tensor, f: torch.Tensor, dim: int) -> torch.Tensor:
    """Dilated correlation along one axis with reflect padding (analysis only)."""
    k = f.numel()
    pad_lo = k // 2
    pad_hi = k - 1 - pad_lo
    xm = x.movedim(dim, -1)
    flat = xm.reshape(-1, 1, xm.shape[-1])
    flat = torch.nn.functional.pad(flat, (pad_lo, pad_hi), mode="reflect")
    y = torch.nn.functional.conv1d(flat, f.flip(0).view(1, 1, -1))
    return y.reshape(*xm.shape).movedim(-1, dim)


def _analyse_1d(x: torch.Tensor, f: torch.Tensor, dim: int, mode: str) -> torch.Tensor:
    if mode == "periodic":
        return _circular_filter_1d(x, f, dim, adjoint=False)
    if mode == "reflect":
        return _reflect_filter_1d(x, f, dim)
    raise ValueError("mode must be 'periodic' or 'reflect', got %r" % mode)


def apply_swt2d(
    field: PhysicalField,
    levels: int = 1,
    wavelet: str = "haar",
    mode: str = "periodic",
) -> Dict[str, Any]:
    """Undecimated 2D wavelet analysis.

    Returns ``{"LL": tensor, "level_1": {"LH","HL","HH"}, ..., "meta": {...}}`` where every
    tensor has exactly the shape of ``field.data``.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1, got %d" % levels)

    data = field.data
    h, g = get_filters(wavelet, data.device, data.dtype)

    # Guard rather than silently producing meaningless coarse scales.
    max_support = filter_support(wavelet, levels)
    if max_support > min(data.shape):
        raise ValueError(
            "Level %d with wavelet '%s' has support %d px, which exceeds the smallest field "
            "dimension %d. Use fewer levels or a larger field (see rule R13)."
            % (levels, wavelet, max_support, min(data.shape))
        )

    coeffs: Dict[str, Any] = {}
    current = data
    for level in range(1, levels + 1):
        d = 2 ** (level - 1)
        hd, gd = dilate_filter(h, d), dilate_filter(g, d)

        low_rows = _analyse_1d(current, hd, 0, mode)
        high_rows = _analyse_1d(current, gd, 0, mode)

        coeffs["level_%d" % level] = {
            "LH": _analyse_1d(low_rows, gd, 1, mode),
            "HL": _analyse_1d(high_rows, hd, 1, mode),
            "HH": _analyse_1d(high_rows, gd, 1, mode),
        }
        current = _analyse_1d(low_rows, hd, 1, mode)

    coeffs["LL"] = current
    coeffs["meta"] = {
        "wavelet": wavelet,
        "levels": levels,
        "mode": mode,
        "shape": tuple(data.shape),
        "invertible": mode == "periodic",
        # R13: report the contaminated margin per level rather than making callers guess.
        "valid_interior_halfwidth": {
            level: valid_interior_halfwidth(wavelet, level) for level in range(1, levels + 1)
        },
    }
    return coeffs


def inverse_swt2d(
    coeffs: Dict[str, Any],
    levels: Optional[int] = None,
    wavelet: Optional[str] = None,
) -> PhysicalField:
    """Exact undecimated 2D synthesis. Requires coefficients produced with mode='periodic'."""
    meta = coeffs.get("meta", {})
    levels = levels if levels is not None else meta.get("levels")
    wavelet = wavelet or meta.get("wavelet", "haar")
    mode = meta.get("mode", "periodic")

    if levels is None:
        raise ValueError("levels must be given when coefficients carry no meta block.")
    if mode != "periodic":
        raise ValueError(
            "inverse_swt2d requires mode='periodic'. Coefficients were produced with "
            "mode=%r, which is not exactly invertible; re-run the analysis with "
            "mode='periodic' if you need reconstruction." % mode
        )

    current = coeffs["LL"]
    h, g = get_filters(wavelet, current.device, current.dtype)

    for level in range(levels, 0, -1):
        d = 2 ** (level - 1)
        hd, gd = dilate_filter(h, d), dilate_filter(g, d)
        band = coeffs["level_%d" % level]

        def adj(x, f_row, f_col):
            return _circular_filter_1d(
                _circular_filter_1d(x, f_col, 1, adjoint=True), f_row, 0, adjoint=True
            )

        current = (
            adj(current, hd, hd)
            + adj(band["LH"], hd, gd)
            + adj(band["HL"], gd, hd)
            + adj(band["HH"], gd, gd)
        ) / _FRAME_CONSTANT_2D

    return PhysicalField(current)


def level_gain(level: int) -> float:
    """Amplitude gain the a trous scheme applies at `level`, relative to the input.

    Each 2D level's dilated orthonormal filters have DC gain sqrt(2) per axis, so
    amplitudes grow by 2**level and energies by 4**level as the level index rises. This is
    a property of the undecimated construction, not of the data.

    Verified against PyWavelets: our coefficients equal ``pywt.swt2(..., norm=True)``
    multiplied by exactly ``2**level`` (agreement to 7 significant figures, see
    ``test_swt_matches_pywavelets_oracle``). PyWavelets divides this factor out;
    we keep it so that ``inverse_swt2d`` is an exact inverse, and divide it out in the
    summary functions below instead — where the choice is visible.
    """
    return float(2 ** level)


def swt_scale_energies(coeffs: Dict[str, Any], normalize: str = "redundancy") -> Dict[str, float]:
    """Per-level detail energy plus the approximation energy.

    Direct precursor of `ScaleSignature` (T4C.1).

    normalize="redundancy" (default): divide each level's energy by ``4**level``, removing
        the construction's own gain so that levels are comparable to one another. This is
        the PyWavelets ``norm=True`` convention.
    normalize="none": raw coefficient energies.

    **Rule R3 in practice, and worth reading before trusting any scale plot.** On one
    32x32 noise field with db2 the raw per-level detail energies are 3003 / 3036 / 3716 for
    levels 1-3 — which reads as a flat, almost rising distribution. Normalised they are
    751 / 190 / 58, a clean octave-by-octave decay. Same data, same transform; the entire
    apparent shape of the scale distribution comes from the normalisation choice. A
    `ScaleSignature` built on raw energies would be measuring the transform, not the
    atmosphere.
    """
    if normalize not in ("redundancy", "none"):
        raise ValueError("normalize must be 'redundancy' or 'none', got %r" % normalize)

    out: Dict[str, float] = {}
    total = 0.0
    levels = coeffs["meta"]["levels"] if "meta" in coeffs else None
    if levels is None:
        levels = sum(1 for k in coeffs if k.startswith("level_"))

    for level in range(1, levels + 1):
        band = coeffs["level_%d" % level]
        e = sum(torch.sum(band[b] ** 2).item() for b in ("LH", "HL", "HH"))
        if normalize == "redundancy":
            e /= level_gain(level) ** 2
        out["level_%d" % level] = e
        total += e

    ll = torch.sum(coeffs["LL"] ** 2).item()
    if normalize == "redundancy":
        ll /= level_gain(levels) ** 2
    out["LL"] = ll
    out["total"] = total + ll
    out["normalize"] = normalize  # recorded so a plot can never hide which convention it used
    return out


def swt_energy_fractions(coeffs: Dict[str, Any]) -> Dict[str, float]:
    """Redundancy-normalised energy *fraction* per level - the R3-compliant summary.

    Fractions sum to 1 and are invariant to a global rescale of the input field, which is
    what makes them comparable across fields, times and wavelet families.
    """
    e = swt_scale_energies(coeffs, normalize="redundancy")
    total = e["total"] or 1.0
    return {k: v / total for k, v in e.items() if k not in ("total", "normalize")}
