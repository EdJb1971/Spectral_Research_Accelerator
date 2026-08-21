"""Phase 4C, T4C.1 and T4C.4: the scale signature and the generic power-law core.

The acceptance criteria in the roadmap are three sentences - a sinusoid concentrates at its
scale, white noise is flat, both survive a rescale - and they are all here. Most of the file
is the other thing: the measures have *analytic* values on white noise, and asserting those
rather than "roughly flat" is what turns the suite into a check on the arithmetic instead of
a check that it ran.
"""

import math

import numpy as np
import pytest
import torch

from src.analysis_engine.power_law import (PowerLawError, compare_exponent_to_null,
                                           loglog_fit)
from src.analysis_engine.scale_signature import (MEASURES, ScaleSignature, gini,
                                                 participation_ratio, scale_energy_exponent,
                                                 scale_signature)
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import decompose_sequence

GRID = GridSpec.cartesian((256, 256), dy_m=25000.0, dx_m=25000.0)
GRID_512 = GridSpec.cartesian((512, 512), dy_m=25000.0, dx_m=25000.0)


def _sequence(frames, grid=GRID):
    fields = [PhysicalField(torch.as_tensor(f, dtype=torch.float64), grid=grid)
              for f in frames]
    return FieldSequence(fields, np.arange(len(frames), dtype=np.float64) * 3600.0)


def _sinusoid(n, wavelength):
    x = torch.arange(n, dtype=torch.float64)
    return torch.sin(2 * math.pi * x / wavelength).repeat(n, 1)


def _noise(n, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, n, dtype=torch.float64, generator=generator)


def _decompose(sequence, family, levels=4):
    config = ({"levels": levels, "wavelet": "db2"} if family == "swt"
              else {"levels": levels})
    return decompose_sequence(sequence, family, config, keep_native=True)


# ============================================================ the stated acceptance criteria

@pytest.mark.parametrize("family", ["swt", "dtcwt"])
@pytest.mark.parametrize("wavelength,expected_level", [(8.0, 3), (16.0, 4)])
def test_a_pure_sinusoid_concentrates_at_its_own_scale(family, wavelength, expected_level):
    """A dyadic level `j` covers wavelengths `2**j` to `2**(j+1)` samples."""
    signature = scale_signature(_decompose(_sequence([_sinusoid(256, wavelength)]), family))
    dominant = signature.dominant_scale()[0]
    assert int(dominant) == expected_level
    assert signature.energy_fraction[0, expected_level - 1] > 0.5


@pytest.mark.parametrize("family", ["swt", "dtcwt"])
def test_white_noise_is_flat_in_energy_fraction(family):
    """Flat is the whole point: white noise has no preferred scale to find.

    Level 4 is excluded for DTCWT because a 256x256 crop leaves it a 2x2 valid interior -
    the signature says so itself, and `test_a_thin_scale_is_named` asserts that it does.
    """
    signature = scale_signature(_decompose(_sequence([_noise(256, 11)]), family))
    fractions = signature.energy_fraction[0]
    thick = [i for i, record in enumerate(signature.interior) if not record["thin"]]
    values = fractions[thick]
    assert np.allclose(values, values.mean(), rtol=0.15)


@pytest.mark.parametrize("family", ["swt", "dtcwt"])
def test_every_measure_survives_a_global_amplitude_rescale(family):
    """A change of units is not a change of structure.

    The threshold measure is included deliberately: its threshold is a multiple of the
    per-scale RMS, so it rescales with the data. What it is *not* invariant to is the choice
    of multiple - which is the next test, and the reason it is never primary.
    """
    frame = _noise(256, 12)
    base = scale_signature(_decompose(_sequence([frame]), family))
    scaled = scale_signature(_decompose(_sequence([frame * 1000.0]), family))

    for measure in MEASURES:
        if measure == "energy_density":
            continue  # a density has units; it scales by 1e6, as it should
        assert np.allclose(base.to_matrix(measure), scaled.to_matrix(measure),
                           equal_nan=True), measure
    assert np.allclose(base.energy_density * 1e6, scaled.energy_density, rtol=1e-9,
                       equal_nan=True)


def test_the_threshold_measure_moves_with_the_threshold_and_the_others_do_not():
    """Rule R3's first trap, measured rather than described."""
    sequence = _sequence([_noise(256, 13)])
    field = _decompose(sequence, "swt")
    low = scale_signature(field, threshold_sigma=2.0)
    high = scale_signature(field, threshold_sigma=4.0)

    ratio = (np.nanmean(low.threshold_fraction) / np.nanmean(high.threshold_fraction))
    assert ratio > 10.0, "a threshold count that barely moves would not be worth warning about"

    for measure in ("energy_fraction", "participation_ratio", "gini"):
        assert np.array_equal(low.to_matrix(measure), high.to_matrix(measure),
                              equal_nan=True), measure


# ============================================================ analytic values on white noise

def test_the_participation_ratio_of_white_noise_matches_its_analytic_value():
    """For real Gaussian coefficients `w = c**2` is chi-squared with one degree of freedom,
    so `E[w] = s**2`, `E[w**2] = 3 s**4` and the normalised participation ratio tends to 1/3.
    For a *circular* complex band `|c|**2` is exponential, `E[w**2] = 2 E[w]**2`, and the
    limit is 1/2.

    Both derived rather than observed, which is what makes this a test of the estimator and
    not a snapshot of it.
    """
    real = scale_signature(_decompose(_sequence([_noise(256, 14)]), "swt"))
    complex_field = scale_signature(_decompose(_sequence([_noise(512, 14)], GRID_512), "dtcwt"))

    assert real.participation_ratio[0, 0] == pytest.approx(1.0 / 3.0, abs=0.02)
    # Level 2 and above; level 1 is the exception and the next test explains it.
    assert complex_field.participation_ratio[0, 1] == pytest.approx(0.5, abs=0.02)


def test_the_gini_of_white_noise_matches_its_analytic_value():
    """Gini of a chi-squared-with-one-degree-of-freedom population is `2/pi`; of an
    exponential population it is exactly `1/2`."""
    real = scale_signature(_decompose(_sequence([_noise(256, 15)]), "swt"))
    complex_field = scale_signature(_decompose(_sequence([_noise(512, 15)], GRID_512), "dtcwt"))

    assert real.gini[0, 0] == pytest.approx(2.0 / math.pi, abs=0.02)
    assert complex_field.gini[0, 1] == pytest.approx(0.5, abs=0.02)


def test_dtcwt_level_one_is_not_a_hilbert_pair_and_the_measures_show_it():
    """A measured property of the transform, not of this module - worth pinning because it
    changes what a level-1 concentration measure means.

    The q-shift filters that make the two trees a Hilbert pair only start at level 2; level 1
    uses the odd-length near-symmetric pair, and its subbands are correspondingly not
    analytic. On white noise the real and imaginary variances of a level-1 subband differ by
    a factor of about 2.2, while at levels 2 to 4 they agree to within a few percent. The
    consequence for the signature is that level 1 sits **between** the real-Gaussian value
    1/3 and the circular-complex value 1/2, and it is there for a reason rather than by
    accident.

    The assertion is a genuine prediction: the participation ratio computed from the whole
    sample must equal `E[w]**2 / E[w**2]` predicted from the two variances alone.
    """
    field = _decompose(_sequence([_noise(512, 25)], GRID_512), "dtcwt")
    ratios = []
    for scale in field.scales:
        band = field.native_band(0, scale, field.orientations[0]).numpy().ravel()
        ratios.append(band.real.var() / band.imag.var())
    assert ratios[0] > 1.8, "level 1 is expected to be markedly non-circular"
    assert all(0.9 < r < 1.15 for r in ratios[1:]), ratios

    band = field.native_band(0, field.scales[0], field.orientations[0]).numpy().ravel()
    v1, v2 = band.real.var(), band.imag.var()
    predicted = (v1 + v2) ** 2 / (3 * v1 ** 2 + 3 * v2 ** 2 + 2 * v1 * v2)
    assert 1.0 / 3.0 < predicted < 0.5
    assert participation_ratio(np.abs(band) ** 2) == pytest.approx(predicted, abs=0.02)


# ============================================================ the measures themselves

def test_participation_ratio_spans_its_range():
    assert participation_ratio(np.ones(64)) == pytest.approx(1.0)
    spike = np.zeros(64)
    spike[0] = 1.0
    assert participation_ratio(spike) == pytest.approx(1.0 / 64.0)


def test_gini_spans_its_range():
    assert gini(np.ones(64)) == pytest.approx(0.0, abs=1e-12)
    spike = np.zeros(64)
    spike[0] = 1.0
    assert gini(spike) == pytest.approx(1.0 - 1.0 / 64.0)


def test_gini_is_invariant_to_duplicating_every_value():
    """Which is why it can be compared between a decimated and an undecimated family."""
    values = np.array([1.0, 2.0, 7.0, 0.5])
    assert gini(np.repeat(values, 16)) == pytest.approx(gini(values))


def test_the_participation_ratio_is_not_invariant_to_duplication():
    """The trap this module exists to avoid, asserted as a fact rather than a fear.

    Replicating every coefficient `r` times multiplies `(sum w)**2 / (n sum w**2)` by exactly
    `r` before the `1/n` normalisation, and by exactly `1` after it - which is why the
    normalisation is not optional and why the measures are taken on native coefficients.
    """
    values = np.array([1.0, 2.0, 7.0, 0.5])
    assert participation_ratio(np.repeat(values, 16)) == pytest.approx(
        participation_ratio(values))
    raw = lambda w: (w.sum() ** 2) / (w ** 2).sum()
    assert raw(np.repeat(values, 16)) == pytest.approx(16.0 * raw(values))


@pytest.mark.parametrize("measure", [participation_ratio, gini])
def test_a_signed_input_is_refused(measure):
    with pytest.raises(InvalidParameterError):
        measure(np.array([1.0, -2.0]))


# ============================================================ the interior mask (R13)

def test_a_scale_with_no_valid_interior_is_nan_and_says_why():
    """A 64x64 crop cannot support DTCWT level 4 - rule R13, met in practice.

    The undecimated transform refuses this case itself, at the transform, because its filter
    support exceeds the field. The decimated one does not: its native band at level 4 is a
    tidy 4x4 array that looks perfectly usable and is entirely boundary. That asymmetry is
    why the interior mask lives here as well as in `stationary.apply_swt2d`.
    """
    small = GridSpec.cartesian((64, 64), dy_m=25000.0, dx_m=25000.0)
    signature = scale_signature(_decompose(_sequence([_noise(64, 16)], small), "dtcwt",
                                           levels=4))
    unusable = [record for record in signature.interior if not record["usable"]]
    # Levels 3 and 4 both: a 64x64 crop supports two DTCWT levels, not four.
    assert [record["scale"] for record in unusable] == ["3", "4"]
    assert np.all(np.isnan(signature.energy_fraction[0, 2:]))
    assert any("no valid interior" in w for w in signature.warnings)


def test_a_thin_scale_is_named():
    """DTCWT's filters are longer than the 14-tap figure rule R13's table is written for, so
    a 256x256 crop - the roadmap's stated minimum for four levels - leaves level 4 a 2x2
    interior. Reported, not silently averaged over."""
    signature = scale_signature(_decompose(_sequence([_noise(256, 17)]), "dtcwt"))
    thin = [record for record in signature.interior if record["thin"]]
    assert [record["scale"] for record in thin] == ["4"]
    assert thin[0]["interior_shape"] == [2, 2]
    assert any("statistically thin" in w for w in signature.warnings)


def test_turning_the_interior_mask_off_is_recorded():
    field = _decompose(_sequence([_noise(256, 18)]), "swt")
    assert scale_signature(field, interior=False).provenance["interior_mask"] is False
    assert scale_signature(field).provenance["interior_mask"] is True


# ============================================================ resampled families

def test_the_signature_is_the_same_computed_from_native_or_restored_coefficients():
    """A field that has lost its native coefficients still yields the true signature, because
    the aligned view is a replication and can be subsampled back exactly."""
    sequence = _sequence([_noise(256, 19)])
    live = _decompose(sequence, "dtcwt")
    restored = live.drop_native()

    for measure in ("energy_fraction", "participation_ratio", "gini"):
        assert np.allclose(scale_signature(live).to_matrix(measure),
                           scale_signature(restored).to_matrix(measure), equal_nan=True)


def test_a_zero_frame_has_no_dominant_scale():
    signature = scale_signature(_decompose(
        _sequence([torch.zeros(256, 256, dtype=torch.float64)]), "swt"))
    assert signature.dominant_scale() == [None]
    assert any("no energy" in w for w in signature.warnings)


# ============================================================ reporting

def test_the_summary_is_small_enough_for_a_lineage_row():
    import json
    signature = scale_signature(_decompose(_sequence([_noise(256, 20) for _ in range(4)]),
                                           "swt"))
    payload = json.dumps(signature.summary())
    assert len(payload) < 4096
    assert "threshold_caveat" in payload
    assert signature.summary()["primary_measures"][-1] == "gini"


def test_records_carry_the_threshold_that_produced_them():
    signature = scale_signature(_decompose(_sequence([_noise(256, 21)]), "swt"),
                                threshold_sigma=2.5)
    rows = signature.as_records()
    assert len(rows) == signature.n_times * signature.n_scales
    assert all(row["threshold_sigma"] == 2.5 for row in rows)
    assert all(row["threshold_value"] > 0 for row in rows)


def test_an_unknown_measure_is_refused_by_name():
    signature = scale_signature(_decompose(_sequence([_noise(256, 22)]), "swt"))
    with pytest.raises(InvalidParameterError) as excinfo:
        signature.to_matrix("kurtosis")
    assert "threshold_fraction" in str(excinfo.value)


def test_a_non_positive_threshold_is_refused():
    field = _decompose(_sequence([_noise(256, 23)]), "swt")
    with pytest.raises(InvalidParameterError):
        scale_signature(field, threshold_sigma=0.0)


# ============================================================ T4C.4: the power-law core

def test_loglog_fit_recovers_a_known_exponent():
    x = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    y = 3.0 * x ** -1.7
    fit = loglog_fit(x, y)
    assert fit["fitted"] is True
    assert fit["slope"] == pytest.approx(-1.7)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["intercept_ln_c"] == pytest.approx(math.log(3.0))


def test_too_few_points_is_reported_rather_than_raised():
    fit = loglog_fit([1.0, 2.0, 3.0], [1.0, 0.5, 0.25])
    assert fit["fitted"] is False
    assert math.isnan(fit["slope"])
    assert "at least 4" in fit["reason"]


def test_a_singular_design_raises():
    with pytest.raises(PowerLawError):
        loglog_fit([2.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_the_spectral_fit_still_delegates_to_the_same_arithmetic():
    """T4C.4 requires existing behaviour to be unchanged; this pins the two together."""
    from src.analysis_engine.spectra import fit_power_law
    k = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    power = 5.0 * k ** -2.4
    spectral = fit_power_law(k, power)
    generic = loglog_fit(k, power)
    assert spectral["slope_beta"] == pytest.approx(-generic["slope"])
    assert spectral["r_squared"] == pytest.approx(generic["r_squared"])


def test_an_exponent_without_a_null_is_not_reportable():
    """Rule R2 as a return value. A fractional Brownian field gives a clean power law and
    contains nothing, so alpha alone cannot be a finding."""
    result = compare_exponent_to_null(2.3, [])
    assert result["reportable"] is False
    assert any("not a finding" in w for w in result["warnings"])

    against = compare_exponent_to_null(2.3, [1.0, 1.1, 0.9, 1.05])
    assert against["reportable"] is True
    assert against["effect_size"] > 5.0
    assert against["p_value"] == pytest.approx(1.0 / 5.0)


def test_the_scale_exponent_carries_its_null_through():
    signature = scale_signature(_decompose(_sequence([_noise(256, 24)]), "swt"))
    bare = scale_energy_exponent(signature)
    assert bare["reportable"] is False
    assert "s**-2" in bare["normalisation"]

    with_null = scale_energy_exponent(signature, null_exponents=[0.0, 0.01, -0.02, 0.005])
    assert with_null["reportable"] is True
    assert with_null["versus_null"]["n_surrogates"] == 4


def test_the_scale_exponent_recovers_an_injected_slope():
    """Built by hand so the answer is known exactly: energy density falling as `s**-1.5`."""
    scales = [1, 2, 3, 4, 5]
    density = np.array([[float(2 ** level) ** -1.5 for level in scales]])
    signature = ScaleSignature(
        times_seconds=np.array([0.0]), scales=scales, wavelet_family="swt",
        energy_density=density, energy_fraction=density / density.sum(),
        participation_ratio=np.full((1, 5), 0.5), gini=np.full((1, 5), 0.5),
        threshold_fraction=np.zeros((1, 5)), threshold_sigma=3.0,
        threshold_values=np.ones(5), available=np.full(5, 1024),
        interior=[{"scale": str(s), "usable": True, "thin": False} for s in scales])
    result = scale_energy_exponent(signature)
    assert result["alpha"] == pytest.approx(1.5)
    assert result["r_squared"] == pytest.approx(1.0)
