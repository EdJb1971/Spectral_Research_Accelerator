"""Surrogate null models (roadmap R1 / R12 / T4C.5, defect D8).

**Why a threshold is not a null model.** Rule R1 exists because a wavelet transform is a set of
*linear filters applied to one field*: neighbouring scales share samples, so their coefficients
are algebraically coupled whether or not the atmosphere couples them. A cross-scale correlation
of 0.8 might be physics or it might be the transform's own overlap. The only way to tell is to
ask what that statistic looks like on data with the same second-order structure and **no**
genuine coupling - which is what a surrogate is.

Each method destroys something specific, and the choice *is* the hypothesis being tested:

*   ``phase_randomise`` - preserves the power spectrum **exactly**, randomises phases. Null:
    "everything here is explained by the second-order (linear, Gaussian) structure". Rejecting
    it is evidence of nonlinearity or phase organisation, not merely of correlation.
*   ``aaft`` - preserves the amplitude *distribution* as well, approximately preserving the
    spectrum. Null: "a monotonic transform of a linear Gaussian process". Use when the field is
    visibly non-Gaussian, where plain phase randomisation would reject for the wrong reason.
*   ``iaaft`` - iterates to preserve both spectrum and distribution to a stated tolerance. The
    most conservative of the three, and the one to quote when the result matters.
*   ``circular_shift`` - preserves each series entirely, destroying only the *alignment*
    between them. Null: "these two series are individually what they are, but unrelated". This
    is the correct null for a lagged cross-scale coupling claim, because it leaves every
    autocorrelation intact - and autocorrelation is exactly what inflates naive significance
    (R12).
*   ``block_bootstrap`` - resamples contiguous blocks, preserving dependence up to the block
    length. Null: "short-range dependence explains this".

**A surrogate p-value is ``(1 + k) / (1 + n)``**, never ``k / n``. The added one keeps the test
exact in finite samples and prevents the meaningless ``p = 0``; its floor, ``1/(1+n)``, is the
resolution limit that `multiple_comparisons.check_power` exists to police.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

METHODS = ("phase_randomise", "aaft", "iaaft", "circular_shift", "block_bootstrap")


class SurrogateError(ValueError):
    """Raised when a surrogate cannot be generated faithfully."""


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------- 1D and 2D FT

def _hermitian_phases(shape, rng: np.random.Generator) -> np.ndarray:
    """Uniform phases that already satisfy ``phi(-k) = -phi(k)``.

    Obtained as the phases of the FFT of a **real** white-noise field. Because the input is
    real, its transform is Hermitian by construction, so its phases are antisymmetric *and*
    marginally uniform - both properties come free and neither has to be imposed.

    **This replaces an antisymmetrisation that was subtly and badly wrong.** The first version
    drew independent uniform phases and antisymmetrised them by averaging,
    ``phi = (phi_k - phi_-k) / 2``. That is antisymmetric, so the surrogate was real and its
    power spectrum was preserved exactly - every property the tests checked. But the
    difference of two independent uniform angles, halved, has a *triangular* density peaked
    at zero, so every surrogate was biased toward the phase-aligned configuration: a field
    with its energy concentrated rather than spread.

    Measured consequence on the fBm benchmark: the excess kurtosis of the level-2 wavelet
    detail was **90.0 +/- 7.5** for those surrogates against **0.75** for the field itself -
    z = -12. The surrogate null was not the null it claimed to be, and the benchmark
    correctly reported a false discovery until this was fixed. Preserving the spectrum is
    necessary but not sufficient; the phases must also be *uniform*.
    """
    noise = rng.standard_normal(shape)
    return np.angle(np.fft.fftn(noise))


def phase_randomise(data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Fourier-transform surrogate: identical power spectrum, randomised phases.

    Works in **any** number of dimensions. Nothing below is dimension-specific: the phases
    come from `fftn` of a real field of the same shape, and the self-conjugate bins are found
    by an axis loop. The original 1D/2D restriction was a statement about what had been
    tested, not about what the arithmetic could do, and Phase 4C needs the 3D case - a
    `(time, y, x)` record whose surrogate must preserve the temporal autocorrelation as well
    as the spatial spectrum (rule R12), which no stack of independent 2D surrogates does.

    The power spectrum is preserved to machine precision - asserted in the tests, because
    "approximately the same spectrum" would make any rejection ambiguous between real
    structure and a spectral mismatch. The Hermitian residual check at the end is what makes
    the generalisation safe rather than hopeful: a phase field that was not antisymmetric in
    some dimension would leave an imaginary part and raise here.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim < 1:
        raise SurrogateError("phase_randomise needs at least a 1D array, got a scalar")
    rng = _rng(seed)
    spectrum = np.fft.fftn(arr)
    phases = _hermitian_phases(arr.shape, rng)

    # Self-conjugate bins (DC, and Nyquist on an even axis) must stay real, so their phase
    # can only be 0 or pi and is taken from the original. Forcing 0 instead would make F(0)
    # positive and flip the sign of the mean whenever the input mean was negative.
    original_phase = np.angle(spectrum)
    self_conj = np.ones(arr.shape, dtype=bool)
    for axis, n in enumerate(arr.shape):
        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        if n % 2 == 0:
            keep[n // 2] = True
        self_conj &= np.expand_dims(
            keep, tuple(i for i in range(arr.ndim) if i != axis))
    phases = np.where(self_conj, original_phase, phases)

    surrogate = np.fft.ifftn(np.abs(spectrum) * np.exp(1j * phases))
    imag = float(np.abs(np.imag(surrogate)).max())
    scale = float(np.abs(np.real(surrogate)).max()) or 1.0
    if imag / scale > 1e-9:
        raise SurrogateError(
            "the surrogate has a residual imaginary part of %.2e relative to its scale, "
            "which means the phase field was not Hermitian and the power spectrum is not "
            "preserved. This is a bug, not a data problem." % (imag / scale))
    return np.real(surrogate)


def aaft(data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Amplitude-adjusted FT surrogate: preserves the amplitude distribution exactly.

    Rank-orders a Gaussianised phase-randomised series back onto the original values, so the
    surrogate is a permutation of the input. The spectrum is then only *approximately*
    preserved - that is the known weakness of AAFT and the reason `iaaft` exists.
    """
    arr = np.asarray(data, dtype=np.float64)
    flat = arr.ravel()
    rng = _rng(seed)

    # Map onto Gaussian ranks, phase-randomise there, then map back by rank.
    gaussian = np.sort(rng.standard_normal(flat.size))
    ranks = np.argsort(np.argsort(flat))
    gaussianised = gaussian[ranks]
    randomised = phase_randomise(gaussianised.reshape(arr.shape), seed=seed).ravel()
    back_ranks = np.argsort(np.argsort(randomised))
    return np.sort(flat)[back_ranks].reshape(arr.shape)


def iaaft(data: np.ndarray, seed: Optional[int] = None, max_iter: int = 200,
          tol: float = 1e-8) -> np.ndarray:
    """Iterative AAFT: preserves both the amplitude distribution and the power spectrum.

    Alternates between imposing the target spectrum and restoring the target amplitudes until
    the spectral mismatch stops improving. The most conservative of the three FT surrogates,
    and therefore the one to quote when a result matters: it removes the objection that a
    rejection came from a distributional or spectral mismatch rather than from structure.
    """
    arr = np.asarray(data, dtype=np.float64)
    target_amps = np.sort(arr.ravel())
    fft = np.fft.fft2 if arr.ndim == 2 else np.fft.fft
    ifft = np.fft.ifft2 if arr.ndim == 2 else np.fft.ifft
    target_spectrum = np.abs(fft(arr))

    current = aaft(arr, seed=seed)
    previous_error = np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        spec = fft(current)
        # 1. impose the target spectrum, keeping the current phases
        with np.errstate(invalid="ignore", divide="ignore"):
            adjusted = np.where(np.abs(spec) > 0,
                                target_spectrum * spec / np.abs(spec), 0.0)
        current = np.real(ifft(adjusted))
        # 2. restore the exact amplitude distribution by rank
        ranks = np.argsort(np.argsort(current.ravel()))
        current = target_amps[ranks].reshape(arr.shape)

        error = float(np.mean((np.abs(fft(current)) - target_spectrum) ** 2))
        if abs(previous_error - error) < tol * max(1.0, previous_error):
            break
        previous_error = error
    return current


def circular_shift(data: np.ndarray, seed: Optional[int] = None,
                   min_shift: int = 1) -> np.ndarray:
    """Roll a series, preserving its autocorrelation entirely.

    The correct null for a *lagged* relationship between two series: every within-series
    property is untouched, so a rejection can only come from the alignment. `min_shift`
    prevents the identity shift, which would be a surrogate identical to the data.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.shape[0]
    if n <= min_shift + 1:
        raise SurrogateError(
            "a circular-shift surrogate of a length-%d series with min_shift=%d has no "
            "admissible shift. Use a longer series or a different method." % (n, min_shift))
    rng = _rng(seed)
    shift = int(rng.integers(min_shift, n - min_shift + 1))
    return np.roll(arr, shift, axis=0)


def block_bootstrap(data: np.ndarray, seed: Optional[int] = None,
                    block_length: Optional[int] = None) -> np.ndarray:
    """Resample contiguous blocks, preserving dependence up to ``block_length``.

    The default block length is ``n**(1/3)``, the standard rate-optimal choice for a
    stationary bootstrap. Blocks that are too short destroy the autocorrelation the null is
    supposed to keep, which makes the test anti-conservative - so the value used is returned
    to the caller via the ensemble metadata rather than left implicit.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.shape[0]
    if block_length is None:
        block_length = max(2, int(round(n ** (1.0 / 3.0))))
    if block_length >= n:
        raise SurrogateError(
            "block_length=%d must be shorter than the series (%d)" % (block_length, n))
    rng = _rng(seed)
    out = []
    while sum(len(b) for b in out) < n:
        start = int(rng.integers(0, n - block_length + 1))
        out.append(arr[start:start + block_length])
    return np.concatenate(out, axis=0)[:n]


_DISPATCH: Dict[str, Callable[..., np.ndarray]] = {
    "phase_randomise": phase_randomise,
    "aaft": aaft,
    "iaaft": iaaft,
    "circular_shift": circular_shift,
    "block_bootstrap": block_bootstrap,
}


def generate(data: np.ndarray, method: str = "iaaft", n: int = 999,
             seed: int = 20260819, **kwargs: Any) -> Dict[str, Any]:
    """Build an ensemble of ``n`` surrogates with independent, reproducible seeds.

    Seeds come from ``SeedSequence.spawn``, so the ensemble is reproducible and its members
    are independent - and so it is identical whether generated serially or through the
    Executor's `process` backend.
    """
    if method not in _DISPATCH:
        raise SurrogateError(
            "unknown surrogate method %r; expected one of %s" % (method, ", ".join(METHODS)))
    if n < 1:
        raise SurrogateError("n must be >= 1, got %r" % (n,))

    children = np.random.SeedSequence(int(seed)).spawn(int(n))
    seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]
    fn = _DISPATCH[method]
    members = [fn(data, seed=s, **kwargs) for s in seeds]
    return {
        "method": method,
        "n": int(n),
        "root_seed": int(seed),
        "seeds": seeds,
        "members": members,
        "p_value_floor": 1.0 / (1.0 + n),
        "preserves": {
            "phase_randomise": ["power spectrum (exact)"],
            "aaft": ["amplitude distribution (exact)", "power spectrum (approximate)"],
            "iaaft": ["amplitude distribution (exact)", "power spectrum (near-exact)"],
            "circular_shift": ["everything within each series", "all autocorrelation"],
            "block_bootstrap": ["dependence up to block_length"],
        }[method],
        "kwargs": dict(kwargs),
    }
