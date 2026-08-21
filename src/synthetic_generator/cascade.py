"""A sequence with a **known** cross-scale coupling at a **known** lag (roadmap T4C.3).

Phase 4C's central claim is that fine-scale activity at `t` precedes coarse-scale activity at
`t + lag`. A statistic that cannot recover that relationship when it is present by
construction is not worth running on the atmosphere, and one that finds it in a field where
it was never put is worse than useless. This module builds both cases:

*   `build_cascade(...)` - fine-band amplitude follows a random modulation `m_t`; coarse-band
    amplitude follows the **same** modulation delayed by `lag`. Nothing else links them.
*   the same sequence phase-randomised in 3D, which preserves every power spectrum and every
    autocorrelation and destroys the alignment. The dependency must vanish there.

**Why the modulation is red rather than white.** A white modulation makes the test far too
easy: the coarse band would then be predictable from the fine band and from nothing else,
including its own past, so even a badly-conditioned transfer entropy would find it. Real
atmospheric amplitude series are strongly autocorrelated, and an autocorrelated modulation is
precisely the case where lagged mutual information reports a coupling that transfer entropy
correctly declines to - because the target's own past already explains it. Building the easy
case would have hidden that distinction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import torch

from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence

DEFAULT_CADENCE_S = 3600.0


def _band(shape, wavelength_px: float, rng: np.random.Generator) -> np.ndarray:
    """A random-phase field whose energy sits in one octave around `wavelength_px`.

    Built in the Fourier domain and masked to the octave, so the band's scale is exact rather
    than approximately right, and so the two bands cannot leak into each other's levels.
    """
    height, width = shape
    noise = rng.standard_normal(shape)
    spectrum = np.fft.fft2(noise)
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.fftfreq(width)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    centre = 1.0 / wavelength_px
    mask = (k >= centre / math.sqrt(2.0)) & (k <= centre * math.sqrt(2.0))
    band = np.real(np.fft.ifft2(spectrum * mask))
    norm = float(np.sqrt((band ** 2).mean()))
    return band / norm if norm > 0 else band


def build_cascade(
    n: int = 128,
    n_frames: int = 96,
    fine_wavelength_px: float = 6.0,
    coarse_wavelength_px: float = 24.0,
    lag: int = 3,
    coupling: float = 1.0,
    modulation_phi: float = 0.5,
    noise_level: float = 0.25,
    spacing_m: float = 25000.0,
    seed: int = 20260821,
) -> FieldSequence:
    """A record in which the fine band at `t` sets the coarse band at `t + lag`, and nothing else.

    `coupling` in `[0, 1]` mixes the delayed modulation with an independent one, so `1.0` is
    a perfect cascade and `0.0` is a null with identical spectra and identical marginal
    behaviour at every scale - the control that shows the recovery is of the coupling and not
    of the construction.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must lie in [0, 1]; got %r" % (coupling,))
    if lag < 1 or lag >= n_frames:
        raise ValueError("lag must be at least 1 and shorter than the record; got %r" % (lag,))

    rng = np.random.default_rng(seed)

    def red(length: int) -> np.ndarray:
        out = np.empty(length)
        out[0] = rng.standard_normal()
        innovation = math.sqrt(max(1.0 - modulation_phi ** 2, 1e-12))
        for i in range(1, length):
            out[i] = modulation_phi * out[i - 1] + innovation * rng.standard_normal()
        return out

    driver = red(n_frames)
    independent = red(n_frames)

    # Amplitudes must be positive: they scale a band's energy. A softplus keeps the ordering
    # of the modulation intact without ever producing a negative amplitude, which would flip
    # the band's sign and leave the energy - the quantity the signature measures - unchanged.
    def amplitude(series: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(series)) + 0.1

    fine_amplitude = amplitude(driver)
    delayed = np.roll(driver, lag)
    coarse_amplitude = amplitude(coupling * delayed
                                 + math.sqrt(max(1.0 - coupling ** 2, 0.0)) * independent)

    grid = GridSpec.cartesian((n, n), dy_m=spacing_m, dx_m=spacing_m)
    fields: List[PhysicalField] = []
    for index in range(n_frames):
        frame = (fine_amplitude[index] * _band((n, n), fine_wavelength_px, rng)
                 + coarse_amplitude[index] * _band((n, n), coarse_wavelength_px, rng)
                 + noise_level * rng.standard_normal((n, n)))
        fields.append(PhysicalField(torch.as_tensor(frame, dtype=torch.float64), grid=grid,
                                    units="dimensionless"))

    times = np.arange(n_frames, dtype=np.float64) * DEFAULT_CADENCE_S
    return FieldSequence(fields, times, metadata={
        "synthetic": "cross_scale_cascade",
        "injected_lag_frames": int(lag),
        "injected_lag_seconds": float(lag * DEFAULT_CADENCE_S),
        "fine_wavelength_px": float(fine_wavelength_px),
        "coarse_wavelength_px": float(coarse_wavelength_px),
        "coupling": float(coupling),
        "modulation_phi": float(modulation_phi),
        "direction": "fine at t precedes coarse at t + lag",
    })


def cascade_truth(lag: int = 3, fine_wavelength_px: float = 6.0,
                  coarse_wavelength_px: float = 24.0, **_: Any) -> Dict[str, Any]:
    """What a correct analysis must find, and what it must not."""
    return {
        "injected_lag_frames": int(lag),
        "fine_level": int(math.floor(math.log2(fine_wavelength_px))),
        "coarse_level": int(math.floor(math.log2(coarse_wavelength_px))),
        "direction": "fine -> coarse",
        "reverse_direction_expected": (
            "coarse -> fine at the same lag must not be reported: nothing in the "
            "construction runs that way, and a symmetric estimator that reports it is "
            "measuring shared modulation rather than precedence"),
        "phase_randomised_expected": (
            "nothing. A 3D phase randomisation preserves every spectrum and every "
            "autocorrelation and destroys only the alignment, so a dependency that survives "
            "it was never about the alignment."),
    }
