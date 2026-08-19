"""Tests for the Kingsbury q-shift dual-tree complex wavelet transform (T3.5.6, defect D1).

D1 is the oldest entry in the ledger and the one the roadmap called the highest-risk item in
Phase 3.5. The defect survived for so long because the *only* test covering the old
implementation was a round trip, and a round trip cannot see it: four copies of the same
filter bank average back to the input perfectly. Several tests here are therefore deliberate
head-to-heads against the old transform, so the fix cannot silently regress.

Two independent oracles are used - the reference `dtcwt` package and `pytorch_wavelets` -
and both are **test-only**. The filter coefficients are vendored, so agreement with an oracle
is a genuine cross-check rather than a tautology.
"""

import math
import warnings

import numpy as np
import pytest
import torch

from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.transform_engine import dtcwt as D
from src.transform_engine import stationary as SW
from src.transform_engine.kingsbury_coeffs import (
    COEFFICIENT_PROVENANCE,
    LEVEL1_FILTERS,
    QSHIFT_FILTERS,
)
from src.transform_engine.transforms import SpectralTransformEngine

N = 128
_IDX = torch.arange(N, dtype=torch.float64)
_YY, _XX = torch.meshgrid(_IDX, _IDX, indexing="ij")


def periodic_front(shift: float, width: float = 0.8, angle_deg: float = 0.0) -> PhysicalField:
    """A sharp front that is genuinely periodic, so a shift is a true torus translation.

    The periodicity matters. Using `torch.roll` on a non-periodic ramp puts a large step at
    the wrap point; that step then dominates the measurement and the DTCWT scored 97.97%
    instead of 4.99%. The probe was wrong, not the transform.
    """
    th = math.radians(angle_deg)
    proj = _XX * math.cos(th) + _YY * math.sin(th)
    return PhysicalField(torch.tanh((((proj - (40 + shift)) % N) - N / 2) / width))


def energy_spread_pct(values) -> float:
    a = np.asarray(values, dtype=float)
    return 100.0 * (a.max() - a.min()) / a.mean()


# ============================================================== vendored coefficients

def test_qshift_tree_b_is_the_time_reverse_of_tree_a():
    """The defining property of a q-shift pair, and the source of the Hilbert relationship."""
    for name, f in QSHIFT_FILTERS.items():
        for lo, hi in (("h0a", "h0b"), ("h1a", "h1b")):
            a = torch.tensor(f[lo], dtype=torch.float64)
            b = torch.tensor(f[hi], dtype=torch.float64)
            assert torch.equal(b, a.flip(0)), "%s: %s is not the reverse of %s" % (name, hi, lo)


def test_qshift_filters_are_orthonormal_to_machine_precision():
    """Truncated literals cost real accuracy - db3 in T3.5.7 held only 1e-11."""
    for name, f in QSHIFT_FILTERS.items():
        for key in ("h0a", "h0b", "h1a", "h1b"):
            h = torch.tensor(f[key], dtype=torch.float64)
            assert abs(float(torch.sum(h * h)) - 1.0) < 1e-12, "%s/%s" % (name, key)


def test_coefficients_record_their_provenance():
    assert COEFFICIENT_PROVENANCE["author"].endswith("Kingsbury")
    assert COEFFICIENT_PROVENANCE["source_package"] == "dtcwt"
    assert "float64" in COEFFICIENT_PROVENANCE["precision"]


def test_runtime_does_not_import_the_oracle_packages():
    """The whole point of vendoring: `pytorch_wavelets` imports the deprecated
    `pkg_resources`, and a platform meant to outlast its dependencies must not need it."""
    import ast
    import inspect

    # Parsed, not grepped: the module docstring legitimately *names* these packages while
    # explaining why they are not imported, so a substring search flags its own explanation.
    for module in (D, __import__("src.transform_engine.kingsbury_coeffs", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("dtcwt", "pytorch_wavelets", "pkg_resources"):
            assert forbidden not in imported, (
                "%s imports %r at runtime; it must stay test-only"
                % (module.__name__, forbidden))


# ============================================================== primitives vs reference

@pytest.fixture(scope="module")
def numpy_lowlevel():
    warnings.filterwarnings("ignore")
    return pytest.importorskip("dtcwt.numpy.lowlevel")


@pytest.fixture(scope="module")
def numpy_transform2d():
    warnings.filterwarnings("ignore")
    return pytest.importorskip("dtcwt.numpy.transform2d")


def test_primitives_match_the_reference_implementation(numpy_lowlevel, numpy_transform2d):
    """Validated one primitive at a time - an off-by-one here is invisible downstream."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 24))
    xt = torch.tensor(x, dtype=torch.float64)

    h0o = np.array(LEVEL1_FILTERS["near_sym_b"]["h0o"])
    h0a = np.array(QSHIFT_FILTERS["qshift_b"]["h0a"])
    h0b = np.array(QSHIFT_FILTERS["qshift_b"]["h0b"])

    assert np.abs(numpy_lowlevel.colfilter(x, h0o)
                  - D._colfilter(xt, torch.tensor(h0o)).numpy()).max() < 1e-14
    assert np.abs(numpy_lowlevel.coldfilt(x, h0b, h0a)
                  - D._coldfilt(xt, torch.tensor(h0b), torch.tensor(h0a)).numpy()).max() < 1e-14
    assert np.abs(numpy_lowlevel.colifilt(x, h0b, h0a)
                  - D._colifilt(xt, torch.tensor(h0b), torch.tensor(h0a)).numpy()).max() < 1e-14

    ref_q = numpy_transform2d.q2c(x)
    assert np.abs(ref_q - D._q2c(xt).numpy()).max() == 0.0
    ref_c = numpy_transform2d.c2q(ref_q, np.array([1.0, 1.0]))
    assert np.abs(ref_c - D._c2q(torch.tensor(ref_q), (1.0, 1.0)).numpy()).max() == 0.0


@pytest.mark.parametrize("shape", [(64, 64), (32, 48), (31, 33)])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_forward_matches_the_dtcwt_oracle(shape, levels):
    """Acceptance criterion 4: coefficients agree with a reference oracle."""
    warnings.filterwarnings("ignore")
    dtcwt_pkg = pytest.importorskip("dtcwt")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape)

    ref = dtcwt_pkg.Transform2d(biort="near_sym_b", qshift="qshift_b").forward(x, nlevels=levels)
    ours = D.apply_dtcwt2d(PhysicalField(torch.tensor(x, dtype=torch.float64)), levels=levels)

    assert np.abs(ref.lowpass - ours["lowpass"].numpy()).max() < 1e-13
    for i in range(levels):
        assert np.abs(ref.highpasses[i] - ours["highpass"][i].numpy()).max() < 1e-13


def test_forward_matches_a_second_independent_oracle():
    """`pytorch_wavelets` is a separate implementation; agreeing with both is stronger."""
    warnings.filterwarnings("ignore")
    pw = pytest.importorskip("pytorch_wavelets")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 64))

    xfm = pw.DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b")
    yl, yh = xfm(torch.tensor(x, dtype=torch.float32).reshape(1, 1, 64, 64))
    ours = D.apply_dtcwt2d(PhysicalField(torch.tensor(x, dtype=torch.float64)), levels=3)

    # pytorch_wavelets stores (batch, channel, orientation, h, w, real/imag) in float32.
    for lvl in range(3):
        ref = yh[lvl][0, 0].detach().numpy()
        ref_c = ref[..., 0] + 1j * ref[..., 1]
        got = ours["highpass"][lvl].numpy()
        got_t = np.transpose(got, (2, 0, 1))
        assert np.abs(np.abs(ref_c) - np.abs(got_t)).max() < 1e-4, (
            "level %d magnitudes disagree with pytorch_wavelets" % (lvl + 1))


# ============================================================== reconstruction

@pytest.mark.parametrize("shape", [(64, 64), (32, 48), (128, 128), (31, 33), (96, 64)])
@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_perfect_reconstruction(shape, levels):
    """Acceptance criterion 3. Note this passed for the *broken* transform too - which is
    exactly why it is not sufficient on its own (rule R8)."""
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.standard_normal(shape), dtype=torch.float64)
    coeffs = D.apply_dtcwt2d(PhysicalField(x), levels=levels)
    rec = D.inverse_dtcwt2d(coeffs).data
    assert rec.shape == x.shape
    assert float(torch.mean((rec - x) ** 2)) < 1e-24


def test_reconstruction_preserves_the_grid():
    grid = GridSpec.latlon((64, 64), 50.0, -0.25, 0.0, 0.25, variable_units="K")
    field = PhysicalField(torch.randn(64, 64, dtype=torch.float64), grid=grid)
    out = D.inverse_dtcwt2d(D.apply_dtcwt2d(field, levels=2))
    assert out.grid.kind == "latlon"
    assert out.data.shape == field.data.shape


# ============================================================== shift invariance

def test_shift_invariance_beats_the_old_transform_and_the_dwt():
    """Acceptance criterion 1, as a three-way head-to-head.

    The old `apply_dtcwt2d` scores **identically to a plain Haar DWT** (236.11% both), which
    is the cleanest possible demonstration that its four trees were one filter bank.
    """
    old, new, dwt = [], [], []
    for s in range(9):
        f = periodic_front(s)
        o = SpectralTransformEngine.apply_dtcwt2d(f, levels=3)["level_2"]
        old.append(sum(float(torch.sum(o[k] ** 2)) for k in o))
        new.append(float(torch.sum(torch.abs(D.apply_dtcwt2d(f, levels=3)["highpass"][1]) ** 2)))
        w = SpectralTransformEngine.apply_dwt2d(f, levels=3)["level_2"]
        dwt.append(sum(float(torch.sum(w[b] ** 2)) for b in ("LH", "HL", "HH")))

    new_spread = energy_spread_pct(new)
    old_spread = energy_spread_pct(old)
    dwt_spread = energy_spread_pct(dwt)

    assert new_spread < 10.0, "measured %.2f%%" % new_spread
    assert old_spread > 100.0, "the old transform should be badly shift-variant"
    assert new_spread < dwt_spread / 20.0
    # The old transform is not merely worse - it is the DWT.
    assert abs(old_spread - dwt_spread) < 1.0


def test_the_specified_vortex_probe_cannot_tell_the_transforms_apart():
    """The roadmap's stated acceptance probe is too weak, and this records that.

    T3.5.6's acceptance says "translate a synthetic vortex 0-8 px". On a smooth, nearly
    radially symmetric vortex, *total subband energy is flat for every transform tested* -
    including the known-broken one and a plain Haar DWT, both at 0.00%. A test using it
    would pass against a transform with no shift invariance at all.

    This is the same trap that appeared in T3.5.7, where an early shift-invariance test
    XPASSED against the defective transform for exactly this reason. Recording it as an
    executable test means the weak probe cannot quietly come back.
    """
    from src.synthetic_generator.generator import SyntheticFieldGenerator

    base = SyntheticFieldGenerator.generate_vortex(N, N).data.to(torch.float64)
    new, dwt = [], []
    for s in range(9):
        f = PhysicalField(torch.roll(base, shifts=s, dims=1))
        new.append(float(torch.sum(torch.abs(D.apply_dtcwt2d(f, levels=3)["highpass"][1]) ** 2)))
        w = SpectralTransformEngine.apply_dwt2d(f, levels=3)["level_2"]
        dwt.append(sum(float(torch.sum(w[b] ** 2)) for b in ("LH", "HL", "HH")))

    assert energy_spread_pct(new) < 0.5
    assert energy_spread_pct(dwt) < 0.5, (
        "if the DWT ever fails here the vortex probe has become discriminating and this "
        "test's premise should be revisited")


def test_the_undecimated_swt_remains_the_exactly_invariant_transform():
    """Division of labour, asserted: SWT for position, DTCWT for orientation.

    The DTCWT is *near* shift invariant (~5% here) by construction, not exact. The SWT is
    exact. Phase 4D tracking should therefore keep using the SWT, and this test states that
    numerically so the choice is not re-litigated from memory.
    """
    swt, dtc = [], []
    for s in range(9):
        f = periodic_front(s)
        c = SW.apply_swt2d(f, levels=3, wavelet="db2")["level_2"]
        swt.append(sum(float(torch.sum(c[b] ** 2)) for b in ("LH", "HL", "HH")))
        dtc.append(float(torch.sum(torch.abs(D.apply_dtcwt2d(f, levels=3)["highpass"][1]) ** 2)))
    assert energy_spread_pct(swt) < 1e-6
    assert energy_spread_pct(dtc) > 1e-3, "DTCWT is approximate, not exact - do not overclaim"


# ============================================================== orientation

def _impulse_passband_angle(subband: int, level: int) -> float:
    """Energy-weighted axial mean angle of a subband's impulse response in the k-plane."""
    base = D.apply_dtcwt2d(PhysicalField(torch.zeros(N, N, dtype=torch.float64)), levels=4)
    coeffs = dict(base)
    coeffs["highpass"] = [h.clone().zero_() for h in base["highpass"]]
    coeffs["lowpass"] = base["lowpass"].clone().zero_()
    b = coeffs["highpass"][level - 1]
    b[b.shape[0] // 2, b.shape[1] // 2, subband] = 1.0 + 0j
    imp = D.inverse_dtcwt2d(coeffs).data
    spec = torch.abs(torch.fft.fftshift(torch.fft.fft2(imp))) ** 2
    ky = torch.fft.fftshift(torch.fft.fftfreq(N, dtype=torch.float64)).unsqueeze(1).expand(N, N)
    kx = torch.fft.fftshift(torch.fft.fftfreq(N, dtype=torch.float64)).unsqueeze(0).expand(N, N)
    ang = torch.atan2(ky, kx)
    s = float((spec * torch.sin(2 * ang)).sum())
    c = float((spec * torch.cos(2 * ang)).sum())
    return math.degrees(0.5 * math.atan2(s, c)) % 180.0


@pytest.mark.parametrize("subband", range(6))
def test_subband_passband_centre_matches_the_declared_orientation(subband):
    """Acceptance criterion 2, measured at the source rather than inferred from a stimulus.

    A first draft declared the orientations in ascending order `(15, 45, 75, 105, 135, 165)`.
    That is the correct set of angles in the *wrong index order* - it would have mislabelled
    every orientation Phase 4E reported, while looking entirely reasonable. Measuring each
    subband's passband centre directly is what caught it.
    """
    measured = _impulse_passband_angle(subband, level=3)
    declared = D.SUBBAND_WAVEVECTOR_DEG[subband]
    err = min(abs(measured - declared), 180 - abs(measured - declared))
    assert err < 3.0, "subband %d: measured %.1f deg, declared %.1f deg" % (
        subband, measured, declared)


def test_the_six_subbands_are_distinct_and_evenly_spaced():
    angles = sorted(D.SUBBAND_WAVEVECTOR_DEG)
    gaps = [angles[i + 1] - angles[i] for i in range(5)] + [angles[0] + 180 - angles[-1]]
    assert all(abs(g - 30.0) < 1e-9 for g in gaps), gaps
    assert len(set(D.SUBBAND_FEATURE_DEG)) == 6


def test_feature_orientation_is_ninety_degrees_from_the_wavevector():
    """These two conventions differ by 90 degrees; conflating them is a silent error."""
    for w, f in zip(D.SUBBAND_WAVEVECTOR_DEG, D.SUBBAND_FEATURE_DEG):
        assert abs(((f - w) % 180) - 90.0) < 1e-9


def test_an_oriented_front_concentrates_in_the_matching_subband():
    """Stimulus-level confirmation: a front at a subband's angle lands in that subband."""
    for subband in range(6):
        wavevector = D.SUBBAND_WAVEVECTOR_DEG[subband]
        th = math.radians(wavevector)
        proj = _XX * math.cos(th) + _YY * math.sin(th)
        grating = torch.sin(2 * math.pi * proj / 6.0)
        coeffs = D.apply_dtcwt2d(PhysicalField(grating), levels=4)
        best_val, best = -1.0, None
        for lvl in range(4):
            band = coeffs["highpass"][lvl]
            for k in range(6):
                v = float(torch.sum(torch.abs(band[:, :, k]) ** 2))
                if v > best_val:
                    best_val, best = v, k
        assert best == subband, (
            "a grating at %.0f deg should peak in subband %d, got %d"
            % (wavevector, subband, best))


def test_dtcwt_separates_plus_and_minus_45_where_the_swt_cannot():
    """The reason this module exists at all.

    A separable real wavelet has one diagonal (HH) subband, so a +45 and a -45 degree
    feature produce the *same* HH energy and are indistinguishable. The DTCWT splits them
    into two subbands. Without this, Phase 4E cannot describe a front's orientation.
    """
    def grating(deg):
        th = math.radians(deg)
        return PhysicalField(torch.sin(
            2 * math.pi * (_XX * math.cos(th) + _YY * math.sin(th)) / 6.0))

    plus, minus = grating(45.0), grating(135.0)

    swt_p = SW.apply_swt2d(plus, levels=2, wavelet="db2")["level_2"]["HH"]
    swt_m = SW.apply_swt2d(minus, levels=2, wavelet="db2")["level_2"]["HH"]
    e_p, e_m = float(torch.sum(swt_p ** 2)), float(torch.sum(swt_m ** 2))
    assert abs(e_p - e_m) / max(e_p, e_m) < 0.05, (
        "the SWT's single HH band should not distinguish +45 from -45")

    dp = D.apply_dtcwt2d(plus, levels=3)["highpass"][2]
    dm = D.apply_dtcwt2d(minus, levels=3)["highpass"][2]
    ep = [float(torch.sum(torch.abs(dp[:, :, k]) ** 2)) for k in range(6)]
    em = [float(torch.sum(torch.abs(dm[:, :, k]) ** 2)) for k in range(6)]
    assert int(np.argmax(ep)) == 1 and int(np.argmax(em)) == 4, (
        "DTCWT must place +45 and -45 in different subbands; got %d and %d"
        % (int(np.argmax(ep)), int(np.argmax(em))))


def test_subband_energies_reports_both_conventions():
    coeffs = D.apply_dtcwt2d(periodic_front(0, angle_deg=45.0), levels=3)
    out = D.subband_energies(coeffs)
    assert len(out["feature_orientations_deg"]) == 6
    assert len(out["wavevector_orientations_deg"]) == 6
    assert "FEATURE" in out["orientation_convention"]
    lvl = out["levels"]["level_2"]
    assert abs(sum(lvl["fraction"]) - 1.0) < 1e-12
    assert lvl["dominant_orientation_deg"] in D.SUBBAND_FEATURE_DEG
    assert lvl["dominant_wavevector_deg"] in D.SUBBAND_WAVEVECTOR_DEG


# ============================================================== errors and edges

def test_unknown_filter_names_list_the_available_ones():
    f = PhysicalField(torch.randn(32, 32, dtype=torch.float64))
    with pytest.raises(D.DTCWTError, match="near_sym_b"):
        D.apply_dtcwt2d(f, level1="not_a_filter")
    with pytest.raises(D.DTCWTError, match="qshift_b"):
        D.apply_dtcwt2d(f, qshift="not_a_filter")


def test_too_many_levels_names_the_limit():
    f = PhysicalField(torch.randn(16, 16, dtype=torch.float64))
    with pytest.raises(D.DTCWTError, match="Reduce `levels`"):
        D.apply_dtcwt2d(f, levels=8)


def test_non_2d_input_is_refused():
    field = PhysicalField(torch.randn(8, 8, dtype=torch.float64))
    field.data = torch.randn(2, 8, 8, dtype=torch.float64)
    with pytest.raises(D.DTCWTError, match="2D"):
        D.apply_dtcwt2d(field)


def test_zero_or_negative_levels_are_refused():
    f = PhysicalField(torch.randn(32, 32, dtype=torch.float64))
    with pytest.raises(D.DTCWTError, match="levels must be"):
        D.apply_dtcwt2d(f, levels=0)


def test_available_filters_lists_what_is_vendored():
    avail = D.available_filters()
    assert "near_sym_b" in avail["level1"]
    assert "qshift_b" in avail["qshift"]
    assert set(avail["qshift"]) == set(QSHIFT_FILTERS)


@pytest.mark.parametrize("qshift", ["qshift_a", "qshift_b", "qshift_c", "qshift_d"])
def test_every_vendored_qshift_set_reconstructs(qshift):
    x = torch.tensor(np.random.default_rng(5).standard_normal((64, 64)), dtype=torch.float64)
    coeffs = D.apply_dtcwt2d(PhysicalField(x), levels=3, qshift=qshift)
    assert float(torch.mean((D.inverse_dtcwt2d(coeffs).data - x) ** 2)) < 1e-24


@pytest.mark.parametrize("level1", ["near_sym_a", "near_sym_b", "legall"])
def test_every_vendored_level1_set_reconstructs(level1):
    x = torch.tensor(np.random.default_rng(6).standard_normal((64, 64)), dtype=torch.float64)
    coeffs = D.apply_dtcwt2d(PhysicalField(x), levels=2, level1=level1)
    assert float(torch.mean((D.inverse_dtcwt2d(coeffs).data - x) ** 2)) < 1e-20


def test_subband_shapes_halve_per_level():
    coeffs = D.apply_dtcwt2d(PhysicalField(torch.randn(128, 128, dtype=torch.float64)),
                             levels=4)
    # Level 1 filters without decimation and `_q2c` then halves, so a 128-px field gives
    # 64 -> 32 -> 16 -> 8, not 32 -> 16 -> ... . The first expectation written here was
    # wrong for exactly that reason.
    expected = 64
    for lvl, band in enumerate(coeffs["highpass"]):
        assert band.shape == (expected, expected, 6), "level %d" % (lvl + 1)
        expected //= 2


def test_coefficients_are_complex_not_a_real_pair():
    """The old implementation stored `_real`/`_imag` tensors holding identical data."""
    coeffs = D.apply_dtcwt2d(PhysicalField(torch.randn(64, 64, dtype=torch.float64)), levels=2)
    band = coeffs["highpass"][0]
    assert band.is_complex()
    assert float(torch.abs(band.imag).max()) > 0
    # Real and imaginary parts must be genuinely different - the D1 signature was that they
    # were equal or sign-flipped copies.
    assert float(torch.abs(band.real - band.imag).max()) > 1e-3
    assert float(torch.abs(band.real + band.imag).max()) > 1e-3


def test_no_runtime_code_uses_the_degenerate_dtcwt():
    """The artefact must stay reachable for regression tests and unreachable in production.

    Leaving a broken function callable is a trap, so this asserts that nothing under `src/`
    outside the test suite calls `SpectralTransformEngine.apply_dtcwt2d` or its inverse.

    Parsed with `ast`, not grepped. A substring version flagged the *comment* in
    `engine.py` that explains why the artefact is no longer used - the same failure the
    documentation guard hit, where a search matches the prose describing the thing it is
    looking for.
    """
    import ast
    import glob
    import os

    offenders = []
    for path in glob.glob(os.path.join("src", "**", "*.py"), recursive=True):
        rel = path.replace(os.sep, "/")
        if "/tests/" in rel or "__pycache__" in rel:
            continue
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("apply_dtcwt2d", "inverse_dtcwt2d")
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "SpectralTransformEngine"):
                offenders.append("%s:%d %s" % (rel, node.lineno, node.attr))
    assert not offenders, (
        "runtime code still calls the degenerate DTCWT (defect D1): "
        + "; ".join(offenders))
