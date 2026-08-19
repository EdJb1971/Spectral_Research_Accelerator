"""Tests for the undecimated (stationary / a trous) 2D wavelet transform, T3.5.7.

The two properties Phase 4 depends on are shift invariance and every scale sharing the
parent grid. Both are asserted here directly, and the shift-invariance test is written so
that the decimated DWT would fail it — see `test_swt_beats_dwt_shift_variance`.
"""
import numpy as np
import pytest
import torch

from src.physical_core.field import PhysicalField
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.transform_engine.stationary import (
    AVAILABLE_WAVELETS,
    apply_swt2d,
    dilate_filter,
    filter_support,
    get_filters,
    inverse_swt2d,
    level_gain,
    swt_energy_fractions,
    swt_scale_energies,
    valid_interior_halfwidth,
)
from src.transform_engine.transforms import SpectralTransformEngine as Decimated

SHAPES = [(32, 32), (31, 33), (64, 48)]
LEVELS = [1, 2, 3]


def _field(shape, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return PhysicalField(torch.randn(*shape).to(dtype))


# ----------------------------------------------------------------- filters

@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_filters_are_orthonormal(wavelet):
    h, g = get_filters(wavelet, dtype=torch.float64)
    # Coefficients are vendored at full double precision from PyWavelets, so these hold
    # to machine epsilon. Truncated literals previously gave only ~1e-11, which was enough
    # to fail the db3 oracle cross-check at 1e-11 -- a useful reminder that filter
    # precision propagates straight into every downstream energy statistic.
    assert torch.sum(h ** 2).item() == pytest.approx(1.0, abs=1e-15)
    assert torch.sum(g ** 2).item() == pytest.approx(1.0, abs=1e-15)
    assert torch.dot(h, g).item() == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_filters_adopt_requested_dtype(wavelet):
    """The D23 lesson: never let numpy promote a filter to float64 behind our backs."""
    h, g = get_filters(wavelet, dtype=torch.float32)
    assert h.dtype == torch.float32 and g.dtype == torch.float32


def test_dilate_filter_inserts_zeros():
    f = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(dilate_filter(f, 1), f)
    assert torch.equal(dilate_filter(f, 2), torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0]))
    assert dilate_filter(f, 4).numel() == 9


def test_support_and_interior_grow_with_level():
    """R13 arithmetic: coarse levels exclude far more of the domain than fine ones."""
    supports = [filter_support("db2", j) for j in range(1, 6)]
    assert supports == sorted(supports) and len(set(supports)) == 5
    # A 64px crop still leaves a (small) interior for db2 at level 5 ...
    assert 64 - 2 * valid_interior_halfwidth("db2", 5) == 16
    # ... but db3 at level 5 exhausts it completely: support 81 px, halfwidth 40.
    assert 64 - 2 * valid_interior_halfwidth("db3", 5) < 0


# ----------------------------------------------------------------- core properties

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("levels", LEVELS)
@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_every_scale_stays_on_the_parent_grid(shape, levels, wavelet):
    """The property that makes CoefficientField coordinate-clean (T4B.1)."""
    f = _field(shape)
    c = apply_swt2d(f, levels=levels, wavelet=wavelet)
    assert c["LL"].shape == shape
    for level in range(1, levels + 1):
        for band in ("LH", "HL", "HH"):
            assert c["level_%d" % level][band].shape == shape


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("levels", LEVELS)
@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_perfect_reconstruction(shape, levels, wavelet):
    f = _field(shape)
    c = apply_swt2d(f, levels=levels, wavelet=wavelet, mode="periodic")
    r = inverse_swt2d(c)
    assert r.data.shape == shape
    assert torch.mean((f.data - r.data) ** 2).item() < 1e-10


@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_frame_constant_is_four_per_2d_level(wavelet):
    """One undecimated 2D level is a tight frame with constant 4 (2 per axis).

    This test is why defect D16 got fixed: while `PhysicalField` force-cast to float32,
    the ratio came out as 3.999998860393 for *every* wavelet - identical across filters,
    i.e. float32 epsilon rather than any property of the transform. With float64 preserved
    it holds to 1e-11.
    """
    f = _field((32, 32), dtype=torch.float64)
    assert f.data.dtype == torch.float64, "D16 regression: PhysicalField downcast the input"
    c = apply_swt2d(f, levels=1, wavelet=wavelet)
    energy = c["LL"] ** 2
    for band in ("LH", "HL", "HH"):
        energy = energy + c["level_1"][band] ** 2
    ratio = (torch.sum(energy) / torch.sum(f.data ** 2)).item()
    assert ratio == pytest.approx(4.0, rel=1e-11)


def test_reflect_mode_preserves_shape_but_refuses_inversion():
    f = _field((32, 32))
    c = apply_swt2d(f, levels=2, wavelet="db2", mode="reflect")
    assert c["level_2"]["HH"].shape == (32, 32)
    assert c["meta"]["invertible"] is False
    with pytest.raises(ValueError, match="periodic"):
        inverse_swt2d(c)


def test_too_many_levels_raises_rather_than_returning_garbage():
    """R13: refuse an analysis the field is too small to support."""
    f = _field((16, 16))
    with pytest.raises(ValueError, match="support"):
        apply_swt2d(f, levels=6, wavelet="db3")


def test_unknown_wavelet_names_available_ones():
    with pytest.raises(ValueError, match="Available"):
        apply_swt2d(_field((16, 16)), wavelet="not_a_wavelet")


# ----------------------------------------------------------------- shift invariance

@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
def test_swt_is_shift_invariant(wavelet):
    """The property D1 was blocking, delivered here instead.

    With circular convolution, translating the input translates the coefficients exactly,
    so after undoing the shift the magnitude envelope is identical and total subband
    energy is constant. This is what makes Phase 4D feature tracking possible.
    """
    base = SyntheticFieldGenerator.generate_front(
        64, 64, angle=30.0, offset=0.0, width_param=0.01, amplitude=1.0
    )
    energies, envelopes = [], []
    for shift_px in range(8):
        shifted = PhysicalField(torch.roll(base.data, shifts=shift_px, dims=1))
        c = apply_swt2d(shifted, levels=2, wavelet=wavelet)["level_2"]
        env = torch.sqrt(c["LH"] ** 2 + c["HH"] ** 2)
        energies.append(torch.sum(env ** 2).item())
        envelopes.append(torch.roll(env, shifts=-shift_px, dims=1).flatten())

    e = torch.tensor(energies)
    spread = ((e.max() - e.min()) / e.mean()).item()
    assert spread < 1e-4, "energy varied by %.4f%% across shifts" % (spread * 100)

    ref = envelopes[0]
    worst = min(
        torch.corrcoef(torch.stack([ref, v]))[0, 1].item() for v in envelopes[1:]
    )
    assert worst > 0.9999, "envelope not translation-equivariant (worst corr %.6f)" % worst


def test_swt_beats_dwt_shift_variance():
    """Head-to-head on the same field, so the improvement is a measurement not a claim."""
    base = SyntheticFieldGenerator.generate_front(
        64, 64, angle=30.0, offset=0.0, width_param=0.01, amplitude=1.0
    )

    def spread_of(fn):
        vals = []
        for shift_px in range(8):
            shifted = PhysicalField(torch.roll(base.data, shifts=shift_px, dims=1))
            vals.append(fn(shifted))
        t = torch.tensor(vals)
        return ((t.max() - t.min()) / t.mean()).item()

    def dwt_energy(f):
        c = Decimated.apply_dwt2d(f, levels=2)["level_2"]
        return torch.sum(c["LH"] ** 2 + c["HH"] ** 2).item()

    def swt_energy(f):
        c = apply_swt2d(f, levels=2, wavelet="haar")["level_2"]
        return torch.sum(c["LH"] ** 2 + c["HH"] ** 2).item()

    dwt_spread = spread_of(dwt_energy)
    swt_spread = spread_of(swt_energy)
    assert dwt_spread > 0.5, "expected the decimated transform to be badly shift-variant"
    assert swt_spread < 1e-4
    assert swt_spread < dwt_spread / 1000.0


# ----------------------------------------------------------------- oracle cross-check

@pytest.mark.parametrize("wavelet", AVAILABLE_WAVELETS)
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_swt_matches_pywavelets_oracle(wavelet, levels):
    """Independent validation against PyWavelets (R8: don't trust our own round-trip alone).

    PyWavelets' ``norm=True`` divides out the a trous gain; we keep it so that
    `inverse_swt2d` is exact. The two therefore agree after multiplying the reference by
    ``level_gain(level)`` — which is itself asserted here, so the convention cannot drift
    silently.
    """
    pywt = pytest.importorskip("pywt")

    torch.manual_seed(0)
    x = torch.randn(32, 32, dtype=torch.float64)
    ours = apply_swt2d(PhysicalField(x), levels=levels, wavelet=wavelet, mode="periodic")
    ref = pywt.swt2(x.numpy(), wavelet, level=levels, norm=True, trim_approx=False)

    # pywt returns coarsest level first.
    for level, entry in zip(range(levels, 0, -1), ref):
        _, details = entry
        ours_energies = sorted(
            float(torch.sum(ours["level_%d" % level][b] ** 2)) for b in ("LH", "HL", "HH")
        )
        ref_energies = sorted(float(np.sum(d ** 2)) * level_gain(level) ** 2 for d in details)
        for a, b in zip(ours_energies, ref_energies):
            assert a == pytest.approx(b, rel=1e-11)


# ----------------------------------------------------------------- scale summaries (R3)

def test_energy_fractions_sum_to_one_and_are_scale_invariant():
    """R3: fractions must be invariant to a global amplitude rescale."""
    f = _field((32, 32))
    big = PhysicalField(f.data * 1000.0)

    fr_a = swt_energy_fractions(apply_swt2d(f, levels=3, wavelet="db2"))
    fr_b = swt_energy_fractions(apply_swt2d(big, levels=3, wavelet="db2"))

    assert sum(fr_a.values()) == pytest.approx(1.0, abs=1e-9)
    for k in fr_a:
        assert fr_a[k] == pytest.approx(fr_b[k], rel=1e-5)


def test_normalisation_changes_the_apparent_scale_distribution():
    """Records the R3 trap as an executable fact rather than a warning in a docstring.

    Raw per-level energies rise with level purely because of the construction's 4**level
    gain; normalised they decay. Same data, opposite conclusion.
    """
    f = _field((32, 32))
    c = apply_swt2d(f, levels=3, wavelet="db2")

    raw = swt_scale_energies(c, normalize="none")
    norm = swt_scale_energies(c, normalize="redundancy")

    raw_levels = [raw["level_%d" % j] for j in (1, 2, 3)]
    norm_levels = [norm["level_%d" % j] for j in (1, 2, 3)]

    # Normalised energies decay monotonically octave by octave...
    assert norm_levels == sorted(norm_levels, reverse=True)
    # ...while the raw ones do not show that decay at all.
    assert raw_levels[2] > norm_levels[2] * 10
    assert raw["normalize"] == "none" and norm["normalize"] == "redundancy"


def test_sinusoid_energy_concentrates_at_the_matching_scale():
    """Physical sanity: a single-scale input should not spread across all levels."""
    fine = SyntheticFieldGenerator.generate_sinusoid(
        64, 64, frequencies=[(16.0, 16.0)], amplitudes=[1.0]
    )
    coarse = SyntheticFieldGenerator.generate_sinusoid(
        64, 64, frequencies=[(2.0, 2.0)], amplitudes=[1.0]
    )
    fr_fine = swt_energy_fractions(apply_swt2d(fine, levels=3, wavelet="db2"))
    fr_coarse = swt_energy_fractions(apply_swt2d(coarse, levels=3, wavelet="db2"))

    # The high-frequency field puts more of its energy in level 1 than the low-frequency one.
    assert fr_fine["level_1"] > fr_coarse["level_1"]
    # And the low-frequency field retains far more in the approximation band.
    assert fr_coarse["LL"] > fr_fine["LL"]


def test_constant_field_has_no_detail_energy():
    f = PhysicalField(torch.full((32, 32), 4.25))
    fr = swt_energy_fractions(apply_swt2d(f, levels=2, wavelet="db2"))
    assert fr["level_1"] < 1e-9
    assert fr["level_2"] < 1e-9
    assert fr["LL"] == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- integration

def test_swt_available_through_the_api(client):
    """A transform nobody can call is only half delivered."""
    import math

    n = 32
    field = [[math.sin(2 * math.pi * 3 * x / n) * math.cos(2 * math.pi * 2 * y / n)
              for x in range(n)] for y in range(n)]
    resp = client.post(
        "/api/v1/transforms/apply",
        json={"field_data": field, "transform_type": "swt",
              "config": {"levels": 2, "wavelet": "db2"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["metrics"]["mean_squared_error"] < 1e-8
    coeffs = body["coefficients"]
    # Every band is returned on the parent grid.
    assert len(coeffs["LL"]) == n and len(coeffs["LL"][0]) == n
    for lvl in ("level_1", "level_2"):
        for band in ("LH", "HL", "HH"):
            assert len(coeffs[lvl][band]) == n
    # R3-compliant summary travels with the coefficients.
    fractions = coeffs["energy_fractions"]
    assert abs(sum(fractions.values()) - 1.0) < 1e-6
    assert coeffs["meta"]["invertible"] is True


def test_swt_runs_in_a_declarative_sweep(client):
    """End-to-end: the wavelet family is a sweepable parameter, as T4B.2 will need."""
    payload = {
        "name": "swt sweep",
        "parameter_matrix": {"wname": ["haar", "db2", "db3"]},
        "pipeline": [
            {"name": "gen", "action": "generate_synthetic",
             "args": {"type": "front", "height": 32, "width": 32,
                      "params": {"angle": 30.0, "width_param": 0.05}}},
            {"name": "xf", "action": "apply_transform",
             "args": {"field_data": "{gen.field_data}", "transform_type": "swt",
                      "config": {"levels": 2, "wavelet": "{wname}"}}},
        ],
        "metadata": {"code_revision": "t357"},
    }
    created = client.post("/api/v1/experiments", json=payload)
    assert created.status_code == 200, created.text
    detail = client.get("/api/v1/experiments/%s" % created.json()["id"]).json()
    assert detail["status"] == "COMPLETED", detail
    assert len(detail["runs"]) == 3
    for run in detail["runs"]:
        assert run["status"] == "COMPLETED", run["error_message"]
        assert run["results"]["xf_metrics"]["mean_squared_error"] < 1e-8
