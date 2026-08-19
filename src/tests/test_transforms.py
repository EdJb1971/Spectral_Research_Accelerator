"""Tests for the spectral transform engine.

This file did not exist before T3.5.0. The transform engine is the mathematical core of
the platform, and it had no test coverage at all — which is why defects D20 (2D reflect
pad), D23 (float64 filter promotion), D25 (stack on the wrong dim) and D2 (hybrid inverse)
all survived undetected. Every inverse transform was dead code.

Note the R8 principle throughout: reconstruction tests are necessary but not sufficient.
The fake DTCWT reconstructs perfectly (see test_dtcwt_trees_are_degenerate), so only a
*property* test can detect it.
"""
import pytest
import torch

from src.physical_core.field import PhysicalField
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.transform_engine.transforms import SpectralTransformEngine as Engine

SHAPES = [(32, 32), (31, 33), (64, 48)]
LEVELS = [1, 2, 3]


def _field(shape, seed=0):
    torch.manual_seed(seed)
    return PhysicalField(torch.randn(*shape))


# --------------------------------------------------------------------------- FFT / DCT

@pytest.mark.parametrize("shape", SHAPES)
def test_fft_roundtrip_is_exact(shape):
    f = _field(shape)
    mag, phase = Engine.apply_fft2d(f)
    recon = Engine.inverse_fft2d(mag, phase, shape)
    assert torch.allclose(f.data, recon.data, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_dct_roundtrip_is_exact(shape):
    f = _field(shape)
    recon = Engine.inverse_dct2d(Engine.apply_dct2d(f))
    assert torch.allclose(f.data, recon.data, atol=1e-4)


def test_dct_is_orthonormal():
    """DCT-II with the implemented normalisation must preserve energy."""
    f = _field((32, 32))
    coeffs = Engine.apply_dct2d(f)
    assert torch.sum(coeffs ** 2).item() == pytest.approx(torch.sum(f.data ** 2).item(), rel=1e-4)


# --------------------------------------------------------------------------- DWT

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("levels", LEVELS)
def test_dwt_roundtrip_is_exact(shape, levels):
    """Regression for D20 (pad), D23 (dtype) and D25 (stack dim) simultaneously."""
    f = _field(shape)
    coeffs = Engine.apply_dwt2d(f, levels=levels)
    recon = Engine.inverse_dwt2d(coeffs, levels=levels, target_shape=shape)
    assert recon.data.shape == shape
    assert torch.mean((f.data - recon.data) ** 2).item() < 1e-9


def test_dwt_preserves_dtype_of_input():
    """D23: filters must adopt the field dtype, not numpy's float64."""
    f = PhysicalField(torch.randn(16, 16))
    coeffs = Engine.apply_dwt2d(f, levels=1)
    assert coeffs["LL"].dtype == f.data.dtype


def test_dwt_energy_is_conserved():
    f = _field((32, 32))
    c = Engine.apply_dwt2d(f, levels=1)
    total = torch.sum(c["LL"] ** 2) + sum(
        torch.sum(c["level_1"][k] ** 2) for k in ("LH", "HL", "HH")
    )
    assert total.item() == pytest.approx(torch.sum(f.data ** 2).item(), rel=1e-4)


def test_dwt_of_constant_field_has_no_detail():
    """A flat field must place all energy in the approximation band."""
    f = PhysicalField(torch.full((32, 32), 3.5))
    c = Engine.apply_dwt2d(f, levels=1)
    for k in ("LH", "HL", "HH"):
        assert torch.max(torch.abs(c["level_1"][k])).item() < 1e-5


# --------------------------------------------------------------------------- DTCWT

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("levels", LEVELS)
def test_dtcwt_roundtrip_is_exact(shape, levels):
    f = _field(shape)
    coeffs = Engine.apply_dtcwt2d(f, levels=levels)
    recon = Engine.inverse_dtcwt2d(coeffs, levels=levels, target_shape=shape)
    assert recon.data.shape == shape
    assert torch.mean((f.data - recon.data) ** 2).item() < 1e-9


def test_dtcwt_trees_are_degenerate():
    """Documents defect D1 as it currently stands, with hard numbers.

    Tree B's filters are ``[cos(pi/4), sin(pi/4)]`` and ``[-sin(pi/4), cos(pi/4)]``,
    which equal the Haar filters up to a sign. Consequently tree BB is the exact
    negation of tree AA, so the ``_real``/``_imag`` pair carries no phase information
    and the transform has no analytic (Hilbert) structure.

    T3.5.6 replaces these with genuine Kingsbury q-shift filters. When it lands, this
    test SHOULD start failing and must be deleted.
    """
    f = _field((32, 32))
    c = Engine.apply_dtcwt2d(f, levels=1)["level_1"]

    # BB is the exact sign-flip of AA -> not an independent tree.
    assert torch.max(torch.abs(c["LH_AA"] + c["LH_BB"])).item() == pytest.approx(0.0, abs=1e-6)
    # And the nominal "imaginary" part is therefore just -real.
    assert torch.max(torch.abs(c["LH_real"] + c["LH_imag"])).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="D1 is FIXED in transform_engine/dtcwt.py (T3.5.6). This xfail now guards the "
           "retained *artefact* in transforms.py, which must stay broken so the "
           "head-to-head regression tests in test_dtcwt.py keep their comparison arm. "
           "If this ever XPASSes, the artefact has been altered and those tests are no "
           "longer proving anything.",
)
def test_dtcwt_is_shift_invariant():
    """The property test a real DTCWT must satisfy (roadmap T3.5.6 acceptance 1).

    Uses a SHARP front deliberately. An earlier version of this test used a smooth
    vortex and total subband energy, and it passed against the known-broken transform —
    a useful reminder that a property test is only as good as the structure it probes.
    Measured against the current implementation, a sharp front's subband energy swings
    by ~153% under translations of 1-7 px, and the magnitude envelope decorrelates
    completely (corr < 0). A genuinely shift-invariant transform keeps both stable.

    This is exactly why Phase 4D feature tracking cannot be built on the current
    transform: coefficients flip with grid parity, so a tracker would report births and
    deaths that are pure aliasing.
    """
    base = SyntheticFieldGenerator.generate_front(
        64, 64, angle=30.0, offset=0.0, width_param=0.01, amplitude=1.0
    )

    energies = []
    envelopes = []
    for shift_px in range(8):
        shifted = PhysicalField(torch.roll(base.data, shifts=shift_px, dims=1))
        c = Engine.apply_dtcwt2d(shifted, levels=2)["level_2"]
        envelope = torch.sqrt(c["LH_real"] ** 2 + c["HH_real"] ** 2)
        energies.append(torch.sum(envelope ** 2).item())
        envelopes.append(envelope.flatten())

    e = torch.tensor(energies)
    spread = ((e.max() - e.min()) / e.mean()).item()
    assert spread < 0.05, "subband energy varies by %.1f%% across 0-7 px shifts" % (spread * 100)

    ref = envelopes[0]
    worst = min(
        torch.corrcoef(torch.stack([ref, env]))[0, 1].item() for env in envelopes[1:]
    )
    assert worst > 0.9, "magnitude envelope decorrelates under shift (worst corr %.3f)" % worst


def test_dwt_shift_variance_is_measurable():
    """Positive control: quantifies how shift-variant the decimated transform actually is.

    This test asserts the CURRENT (broken) behaviour so that the number is recorded and
    a regression in the opposite direction is also caught. It documents the baseline
    that T3.5.6 has to beat.
    """
    base = SyntheticFieldGenerator.generate_front(
        64, 64, angle=30.0, offset=0.0, width_param=0.01, amplitude=1.0
    )
    energies = []
    for shift_px in range(8):
        shifted = PhysicalField(torch.roll(base.data, shifts=shift_px, dims=1))
        c = Engine.apply_dwt2d(shifted, levels=2)["level_2"]
        energies.append(torch.sum(c["LH"] ** 2 + c["HH"] ** 2).item())

    e = torch.tensor(energies)
    spread = ((e.max() - e.min()) / e.mean()).item()
    # Decimated Haar is grossly shift-variant for a sharp edge: measured ~1.5 (150%).
    assert spread > 0.5, "expected strong shift variance from decimated Haar, got %.3f" % spread


# --------------------------------------------------------------------------- Hybrid

@pytest.mark.parametrize("crossover", [0.1, 0.3, 0.5, 0.9])
def test_hybrid_roundtrip_is_exact(crossover):
    """Regression for D2: the forward pass splits low + residual, so the inverse sums them."""
    f = _field((32, 32))
    coeffs = Engine.apply_hybrid(f, crossover_freq=crossover, mixing_weight=0.5)
    recon = Engine.inverse_hybrid(coeffs, (32, 32))
    assert torch.mean((f.data - recon.data) ** 2).item() < 1e-9


def test_hybrid_lossy_blend_is_opt_in_and_labelled_lossy():
    """exact=False is a deliberate blend, not a reconstruction; it must differ."""
    f = _field((32, 32))
    coeffs = Engine.apply_hybrid(f, crossover_freq=0.3, mixing_weight=0.9)
    exact = Engine.inverse_hybrid(coeffs, (32, 32), exact=True)
    lossy = Engine.inverse_hybrid(coeffs, (32, 32), exact=False)
    assert torch.mean((f.data - exact.data) ** 2).item() < 1e-9
    assert torch.mean((exact.data - lossy.data) ** 2).item() > 1e-6
