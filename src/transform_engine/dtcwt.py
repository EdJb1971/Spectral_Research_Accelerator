"""A real dual-tree complex wavelet transform (Kingsbury q-shift).

Roadmap T3.5.6, defect **D1** - the oldest and highest-risk entry in the ledger. The previous
`apply_dtcwt2d` ran four filter-bank trees whose tree-B filters were degenerate: at
`transforms.py:148-151`, `h_b = [cos(pi/4), sin(pi/4)]` is *identical* to the Haar low-pass
`h_a`, and `g_b = -g_a`. All four trees were therefore the same filter bank up to a sign, so
there was no Hilbert pair, no analytic response, no shift invariance and no orientation -
while perfect reconstruction still held, because four copies of the same transform average
back to the input. The round-trip test passed and hid the defect completely. That is the
canonical argument for rule R8: an inverse test validates a transform against itself.

**What a genuine DTCWT provides that the undecimated SWT (T3.5.7) does not: orientation.**
The SWT already gives exact shift invariance and keeps every scale on the parent grid, which
is why D1 stopped being a Phase 4 blocker once it landed. What the SWT cannot give is
*directional selectivity*: its separable subbands mix +45 and -45 degree features into a
single HH band and cannot distinguish them. The DTCWT's six oriented complex subbands
(+/-15, +/-45, +/-75 degrees) are what Phase 4E needs to describe a front's orientation, and
its complex magnitude is smooth under translation, which is what makes a feature's amplitude
a stable attribute rather than a function of grid parity.

**Construction.** Level 1 uses odd-length near-symmetric biorthogonal filters applied without
decimation; the two trees are then the even and odd sample positions, separated by `_q2c`,
which reads each 2x2 quad as two complex numbers. Levels >= 2 use q-shift filters, where
tree B's coefficients are the exact time-reverse of tree A's - that reversal is what produces
the half-sample (quarter-cycle) delay difference and hence the approximate Hilbert-pair
relationship between the trees.

Coefficients are vendored in `kingsbury_coeffs.py` so the runtime depends on neither `dtcwt`
nor `pytorch_wavelets`. Both remain **test-only oracles**: two independent validators are
worth far more than one implementation, and importing the filters from an oracle would make
the agreement partly circular.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.transform_engine.kingsbury_coeffs import (
    COEFFICIENT_PROVENANCE,
    LEVEL1_FILTERS,
    QSHIFT_FILTERS,
)

# Orientation of each of the six complex subbands, **measured rather than assumed**.
#
# A first draft of this module wrote the canonical set in ascending order,
# ``(15, 45, 75, 105, 135, 165)``. That is the right set of angles in the wrong index order,
# and it would have mislabelled every orientation Phase 4E ever reported while looking
# entirely plausible. It was corrected by measuring each subband's passband centre directly:
# feed an impulse into one subband, invert, and take the energy-weighted axial mean angle of
# the FFT. Measured centres at levels 2-4 (q-shift stages):
#
#     subband   0      1      2      3      4      5
#     degrees   74.7   45.0   15.3  164.6  135.0  105.4
#
# Level 1 deviates by up to 7 degrees (82.1, 45.0, 7.9, 161.2, 135.0, 108.8) because the
# odd-length biorthogonal filters used there have a less precise directional response than
# the q-shift filters. That is a property of the construction, not an error, and it is why
# orientation estimates should be taken from level >= 2.
#
#: Direction of the subband's spatial *wavevector* (the direction of variation), degrees.
SUBBAND_WAVEVECTOR_DEG: Tuple[float, ...] = (75.0, 45.0, 15.0, 165.0, 135.0, 105.0)

#: Orientation of the *feature* (edge, front, filament) the subband responds to, which is
#: the wavevector direction rotated by 90 degrees. This is the convention a meteorologist
#: means by "the front is oriented at 45 degrees", so it is the one reported by
#: :func:`subband_energies` - and the two are kept as separate named constants precisely
#: because silently mixing them is a 90-degree error that looks like a sign convention.
SUBBAND_FEATURE_DEG: Tuple[float, ...] = tuple((w + 90.0) % 180.0 for w in SUBBAND_WAVEVECTOR_DEG)

#: Backwards-compatible alias; prefer the two explicit names above.
SUBBAND_ORIENTATIONS_DEG: Tuple[float, ...] = SUBBAND_FEATURE_DEG


class DTCWTError(ValueError):
    """Raised when a DTCWT cannot be computed as requested."""


# --------------------------------------------------------------------- primitives

def _reflect(x: torch.Tensor, minx: float, maxx: float) -> torch.Tensor:
    """Reflect indices into ``[minx, maxx]`` about half-sample boundaries.

    Symmetric ("whole-sample-off") extension: reflection is about -0.5 and r-0.5 so no
    sample is duplicated at the boundary. Getting this wrong shifts every output by half a
    sample, which no shape test can detect.
    """
    rng = maxx - minx
    rng2 = 2 * rng
    mod = torch.remainder(x - minx, rng2)
    normed = torch.where(mod < 0, mod + rng2, mod)
    out = torch.where(normed >= rng, rng2 - normed, normed) + minx
    return out.round().to(torch.long)


def _col_convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Valid-mode convolution down dim 0 of a ``(R, C)`` tensor.

    ``conv1d`` is a *correlation*, so the kernel is flipped to make this a convolution -
    matching the reference implementation's `numpy.convolve` semantics. Reversing this by
    accident produces a filter that is the mirror of the intended one: still a valid
    filter, still perfectly reconstructing, and wrong.
    """
    r, c = x.shape
    m = h.numel()
    if r < m:
        raise DTCWTError(
            "cannot convolve %d rows with a %d-tap filter; the field is too small for this "
            "level. Reduce `levels` or use a larger crop." % (r, m))
    inp = x.transpose(0, 1).reshape(c, 1, r)
    ker = h.flip(0).reshape(1, 1, m).to(x.dtype)
    out = F.conv1d(inp, ker)
    return out.reshape(c, -1).transpose(0, 1)


def _colfilter(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Undecimated filtering down the columns, symmetric extension, odd-length ``h``."""
    r = x.shape[0]
    m = h.numel()
    m2 = m // 2
    idx = _reflect(torch.arange(-m2, r + m2, device=x.device, dtype=torch.float64),
                   -0.5, r - 0.5)
    return _col_convolve(x.index_select(0, idx.to(x.device)), h)


def _coldfilt(x: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
    """Filter and decimate by 2, applying ``ha``/``hb`` to alternate output samples.

    This is the q-shift stage: the two trees' filters are interleaved in the output, which
    is what makes the pair analytic.
    """
    r, c = x.shape
    if r % 4 != 0:
        raise DTCWTError(
            "coldfilt needs a row count divisible by 4, got %d. The transform pads LoLo to "
            "a multiple of 4 between levels; reaching here means that padding was skipped."
            % r)
    if ha.numel() != hb.numel():
        raise DTCWTError("ha and hb must be the same length")
    m = ha.numel()
    if m % 2 != 0:
        raise DTCWTError("q-shift filters must have even length, got %d" % m)

    idx = _reflect(torch.arange(-m, r + m, device=x.device, dtype=torch.float64),
                   -0.5, r - 0.5)
    hao, hae = ha[0:m:2], ha[1:m:2]
    hbo, hbe = hb[0:m:2], hb[1:m:2]
    t = torch.arange(5, r + 2 * m - 2, 4, device=x.device)
    r2 = r // 2

    out = torch.zeros((r2, c), dtype=x.dtype, device=x.device)
    # The branch on sign(sum(ha*hb)) decides which tree lands on even output samples. It is
    # a property of the filter pair, not a convention we may pick.
    if float(torch.sum(ha * hb)) > 0:
        s1, s2 = slice(0, r2, 2), slice(1, r2, 2)
    else:
        s2, s1 = slice(0, r2, 2), slice(1, r2, 2)

    out[s1, :] = (_col_convolve(x.index_select(0, idx[t - 1]), hao)
                  + _col_convolve(x.index_select(0, idx[t - 3]), hae))
    out[s2, :] = (_col_convolve(x.index_select(0, idx[t]), hbo)
                  + _col_convolve(x.index_select(0, idx[t - 2]), hbe))
    return out


def _colifilt(x: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
    """Upsample by 2 and filter - the adjoint of :func:`_coldfilt`."""
    r, c = x.shape
    if r % 2 != 0:
        raise DTCWTError("colifilt needs an even row count, got %d" % r)
    m = ha.numel()
    m2 = m // 2
    idx = _reflect(torch.arange(-m2, r + m2, device=x.device, dtype=torch.float64),
                   -0.5, r - 0.5)
    hao, hae = ha[0:m:2], ha[1:m:2]
    hbo, hbe = hb[0:m:2], hb[1:m:2]
    out = torch.zeros((r * 2, c), dtype=x.dtype, device=x.device)
    positive = float(torch.sum(ha * hb)) > 0

    if m2 % 2 == 0:
        t = torch.arange(3, r + m, 2, device=x.device)
        ta, tb = (t, t - 1) if positive else (t - 1, t)
        s = torch.arange(0, r * 2, 4, device=x.device)
        out[s, :] = _col_convolve(x.index_select(0, idx[tb - 2]), hae)
        out[s + 1, :] = _col_convolve(x.index_select(0, idx[ta - 2]), hbe)
        out[s + 2, :] = _col_convolve(x.index_select(0, idx[tb]), hao)
        out[s + 3, :] = _col_convolve(x.index_select(0, idx[ta]), hbo)
    else:
        t = torch.arange(2, r + m - 1, 2, device=x.device)
        ta, tb = (t, t - 1) if positive else (t - 1, t)
        s = torch.arange(0, r * 2, 4, device=x.device)
        out[s, :] = _col_convolve(x.index_select(0, idx[tb]), hao)
        out[s + 1, :] = _col_convolve(x.index_select(0, idx[ta]), hbo)
        out[s + 2, :] = _col_convolve(x.index_select(0, idx[tb]), hae)
        out[s + 3, :] = _col_convolve(x.index_select(0, idx[ta]), hbe)
    return out


def _q2c(y: torch.Tensor) -> torch.Tensor:
    """Read 2x2 quads as two complex subbands. Returns ``(H/2, W/2, 2)`` complex.

    The quad
        a b
        c d
    becomes ``p = (a + jb)/sqrt(2)`` and ``q = (d - jc)/sqrt(2)``, and the two subbands are
    ``p - q`` and ``p + q``. This is where the two trees are separated at level 1.
    """
    s = math.sqrt(0.5)
    a, b = y[0::2, 0::2], y[0::2, 1::2]
    c, d = y[1::2, 0::2], y[1::2, 1::2]
    p = torch.complex(a * s, b * s)
    q = torch.complex(d * s, -c * s)
    return torch.stack((p - q, p + q), dim=-1)


def _c2q(w: torch.Tensor, gain: Tuple[float, float]) -> torch.Tensor:
    """Inverse of :func:`_q2c`."""
    s = math.sqrt(0.5)
    p = w[..., 0] * (s * gain[0]) + w[..., 1] * (s * gain[1])
    q = w[..., 0] * (s * gain[0]) - w[..., 1] * (s * gain[1])
    h, wid = p.shape
    out = torch.zeros((h * 2, wid * 2), dtype=p.real.dtype, device=p.device)
    out[0::2, 0::2] = p.real
    out[0::2, 1::2] = p.imag
    out[1::2, 0::2] = q.imag
    out[1::2, 1::2] = -q.real
    return out


def _get(name: str, table: Dict[str, Dict[str, Tuple[float, ...]]], key: str,
         device, dtype) -> torch.Tensor:
    return torch.tensor(table[name][key], dtype=dtype, device=device)


def available_filters() -> Dict[str, List[str]]:
    return {"level1": sorted(LEVEL1_FILTERS), "qshift": sorted(QSHIFT_FILTERS)}


def _pad_to_multiple_of_four(x: torch.Tensor) -> torch.Tensor:
    """Edge-replicate rows/columns so the q-shift stage's /4 requirement is met."""
    r, c = x.shape
    if r % 4 != 0:
        x = torch.cat((x[:1, :], x, x[-1:, :]), dim=0)
    if c % 4 != 0:
        x = torch.cat((x[:, :1], x, x[:, -1:]), dim=1)
    return x


# --------------------------------------------------------------------- transform

def apply_dtcwt2d(
    field: PhysicalField,
    levels: int = 3,
    level1: str = "near_sym_b",
    qshift: str = "qshift_b",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Forward dual-tree complex wavelet transform.

    Returns a dict with ``lowpass`` (real, coarsest approximation), ``highpass`` (a list of
    complex tensors, one per level, each ``(H_l, W_l, 6)``), plus the orientation table and
    enough metadata for :func:`inverse_dtcwt2d` to invert it exactly.
    """
    if levels < 1:
        raise DTCWTError("levels must be >= 1, got %r" % (levels,))
    if level1 not in LEVEL1_FILTERS:
        raise DTCWTError("unknown level-1 filter %r; available: %s"
                         % (level1, sorted(LEVEL1_FILTERS)))
    if qshift not in QSHIFT_FILTERS:
        raise DTCWTError("unknown q-shift filter %r; available: %s"
                         % (qshift, sorted(QSHIFT_FILTERS)))

    data = field.data.to(dtype)
    if data.dim() != 2:
        raise DTCWTError("DTCWT expects a 2D field, got shape %r" % (tuple(data.shape),))

    original_shape = tuple(data.shape)
    # Odd dimensions are extended by one row/column, as in the reference. Recorded so the
    # inverse can crop back rather than silently returning a differently-shaped field.
    row_extended = data.shape[0] % 2 == 1
    col_extended = data.shape[1] % 2 == 1
    if row_extended:
        data = torch.cat((data, data[-1:, :]), dim=0)
    if col_extended:
        data = torch.cat((data, data[:, -1:]), dim=1)

    device = data.device
    h0o = _get(level1, LEVEL1_FILTERS, "h0o", device, dtype)
    h1o = _get(level1, LEVEL1_FILTERS, "h1o", device, dtype)
    h0a = _get(qshift, QSHIFT_FILTERS, "h0a", device, dtype)
    h0b = _get(qshift, QSHIFT_FILTERS, "h0b", device, dtype)
    h1a = _get(qshift, QSHIFT_FILTERS, "h1a", device, dtype)
    h1b = _get(qshift, QSHIFT_FILTERS, "h1b", device, dtype)

    min_dim = min(data.shape)
    if min_dim < 2 ** levels:
        raise DTCWTError(
            "a %d-level DTCWT needs at least %d samples on the shorter axis; this field is "
            "%dx%d. Reduce `levels` to %d or use a larger crop."
            % (levels, 2 ** levels, original_shape[0], original_shape[1],
               max(1, int(math.log2(min_dim)))))

    highpass: List[torch.Tensor] = []
    shapes: List[Tuple[int, int]] = []

    # ---- level 1: odd-length filters, no decimation; trees separated by _q2c ----
    lo = _colfilter(data, h0o).transpose(0, 1)
    hi = _colfilter(data, h1o).transpose(0, 1)
    lolo = _colfilter(lo, h0o).transpose(0, 1)

    band = torch.zeros((lolo.shape[0] // 2, lolo.shape[1] // 2, 6),
                       dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
                       device=device)
    horiz = _q2c(_colfilter(hi, h0o).transpose(0, 1))
    vert = _q2c(_colfilter(lo, h1o).transpose(0, 1))
    diag = _q2c(_colfilter(hi, h1o).transpose(0, 1))
    band[:, :, 0], band[:, :, 5] = horiz[..., 0], horiz[..., 1]
    band[:, :, 2], band[:, :, 3] = vert[..., 0], vert[..., 1]
    band[:, :, 1], band[:, :, 4] = diag[..., 0], diag[..., 1]
    highpass.append(band)
    shapes.append(tuple(lolo.shape))

    # ---- levels >= 2: q-shift filters ----
    for _ in range(1, levels):
        lolo = _pad_to_multiple_of_four(lolo)
        shapes.append(tuple(lolo.shape))
        lo = _coldfilt(lolo, h0b, h0a).transpose(0, 1)
        hi = _coldfilt(lolo, h1b, h1a).transpose(0, 1)
        lolo = _coldfilt(lo, h0b, h0a).transpose(0, 1)

        band = torch.zeros(
            (lolo.shape[0] // 2, lolo.shape[1] // 2, 6),
            dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
            device=device)
        horiz = _q2c(_coldfilt(hi, h0b, h0a).transpose(0, 1))
        vert = _q2c(_coldfilt(lo, h1b, h1a).transpose(0, 1))
        diag = _q2c(_coldfilt(hi, h1b, h1a).transpose(0, 1))
        band[:, :, 0], band[:, :, 5] = horiz[..., 0], horiz[..., 1]
        band[:, :, 2], band[:, :, 3] = vert[..., 0], vert[..., 1]
        band[:, :, 1], band[:, :, 4] = diag[..., 0], diag[..., 1]
        highpass.append(band)

    return {
        "lowpass": lolo,
        "highpass": highpass,
        "levels": levels,
        "level1": level1,
        "qshift": qshift,
        "feature_orientations_deg": SUBBAND_FEATURE_DEG,
        "wavevector_orientations_deg": SUBBAND_WAVEVECTOR_DEG,
        "original_shape": original_shape,
        "row_extended": row_extended,
        "col_extended": col_extended,
        "intermediate_shapes": shapes,
        "grid": field.grid.to_provenance(),
        "coefficient_provenance": dict(COEFFICIENT_PROVENANCE),
    }


def inverse_dtcwt2d(coeffs: Dict[str, Any], dtype: torch.dtype = torch.float64) -> PhysicalField:
    """Reconstruct a field from :func:`apply_dtcwt2d` output."""
    level1 = coeffs["level1"]
    qshift = coeffs["qshift"]
    lowpass = coeffs["lowpass"].to(dtype)
    highpass = coeffs["highpass"]
    device = lowpass.device

    g0o = _get(level1, LEVEL1_FILTERS, "g0o", device, dtype)
    g1o = _get(level1, LEVEL1_FILTERS, "g1o", device, dtype)
    g0a = _get(qshift, QSHIFT_FILTERS, "g0a", device, dtype)
    g0b = _get(qshift, QSHIFT_FILTERS, "g0b", device, dtype)
    g1a = _get(qshift, QSHIFT_FILTERS, "g1a", device, dtype)
    g1b = _get(qshift, QSHIFT_FILTERS, "g1b", device, dtype)

    z = lowpass
    for level in range(len(highpass) - 1, 0, -1):
        band = highpass[level]
        lh = _c2q(torch.stack((band[:, :, 0], band[:, :, 5]), dim=-1), (1.0, 1.0))
        hl = _c2q(torch.stack((band[:, :, 2], band[:, :, 3]), dim=-1), (1.0, 1.0))
        hh = _c2q(torch.stack((band[:, :, 1], band[:, :, 4]), dim=-1), (1.0, 1.0))

        target = (lh.shape[0], lh.shape[1])
        if tuple(z.shape) != target:
            z = z[:target[0], :target[1]]

        # `lh` pairs with the *column* synthesis low-pass and `hl` with the row one. An
        # earlier version had these two swapped: the forward transform still matched the
        # reference oracle to 1e-15 and every shape was correct, but reconstruction MSE was
        # ~0.5 on unit-variance input. Only a round-trip test catches it - the mirror of
        # the D1 lesson, where only a *property* test caught what a round trip missed.
        y1 = _colifilt(z, g0b, g0a) + _colifilt(lh, g1b, g1a)
        y2 = _colifilt(hl, g0b, g0a) + _colifilt(hh, g1b, g1a)
        z = (_colifilt(y1.transpose(0, 1), g0b, g0a)
             + _colifilt(y2.transpose(0, 1), g1b, g1a)).transpose(0, 1)

        # Crop back to twice the next finer level's band size, undoing any multiple-of-four
        # padding the forward pass applied.
        finer = highpass[level - 1]
        expected = (2 * finer.shape[0], 2 * finer.shape[1])
        if z.shape[0] != expected[0]:
            z = z[1:-1, :]
        if z.shape[1] != expected[1]:
            z = z[:, 1:-1]
        if tuple(z.shape) != expected:
            raise DTCWTError(
                "synthesis shape %r does not match the expected %r at level %d; the "
                "highpass bands are inconsistent with each other."
                % (tuple(z.shape), expected, level))

    band = highpass[0]
    lh = _c2q(torch.stack((band[:, :, 0], band[:, :, 5]), dim=-1), (1.0, 1.0))
    hl = _c2q(torch.stack((band[:, :, 2], band[:, :, 3]), dim=-1), (1.0, 1.0))
    hh = _c2q(torch.stack((band[:, :, 1], band[:, :, 4]), dim=-1), (1.0, 1.0))

    if tuple(z.shape) != (lh.shape[0], lh.shape[1]):
        z = z[:lh.shape[0], :lh.shape[1]]

    y1 = _colfilter(z, g0o) + _colfilter(lh, g1o)
    y2 = _colfilter(hl, g0o) + _colfilter(hh, g1o)
    out = (_colfilter(y1.transpose(0, 1), g0o)
           + _colfilter(y2.transpose(0, 1), g1o)).transpose(0, 1)

    h, w = coeffs["original_shape"]
    out = out[:h, :w]
    grid = GridSpec.from_provenance(coeffs["grid"])
    if tuple(grid.shape) != tuple(out.shape):
        grid = grid.subset(0, out.shape[0], 0, out.shape[1])
    return PhysicalField(out, grid=grid)


# --------------------------------------------------------------------- summaries

def subband_energies(coeffs: Dict[str, Any]) -> Dict[str, Any]:
    """Per-level, per-orientation complex-magnitude energy.

    This is the quantity Phase 4E wants: an orientation histogram per scale. Reported as
    fractions per level as well as raw, following rule R3 - a raw energy comparison across
    levels measures the transform's own gain as much as the data.
    """
    out: Dict[str, Any] = {
        "feature_orientations_deg": list(SUBBAND_FEATURE_DEG),
        "wavevector_orientations_deg": list(SUBBAND_WAVEVECTOR_DEG),
        "orientation_convention": (
            "dominant_orientation_deg is the FEATURE (edge/front) orientation; the "
            "wavevector direction is 90 degrees from it"),
        "levels": {},
    }
    for i, band in enumerate(coeffs["highpass"], start=1):
        energies = [float(torch.sum(torch.abs(band[:, :, k]) ** 2)) for k in range(6)]
        total = sum(energies) or 1.0
        out["levels"]["level_%d" % i] = {
            "energy": energies,
            "fraction": [e / total for e in energies],
            "dominant_subband": max(range(6), key=lambda k: energies[k]),
            "dominant_orientation_deg": SUBBAND_FEATURE_DEG[
                max(range(6), key=lambda k: energies[k])],
            "dominant_wavevector_deg": SUBBAND_WAVEVECTOR_DEG[
                max(range(6), key=lambda k: energies[k])],
        }
    out["lowpass_energy"] = float(torch.sum(coeffs["lowpass"] ** 2))
    return out
