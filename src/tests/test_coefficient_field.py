"""`CoefficientField` and the level bank (roadmap T4B.1, T4B.4).

T4B.1's stated acceptance criteria are asserted directly: coordinate alignment across all
scales, perfect reconstruction per family, and a lineage-safe `.summary()`.

Beyond those, the properties tested here are the ones whose failure is *silent*:

*   **The DTCWT alignment is upsampled, and reconstruction must not use it.** A level-3
    subband is 8x8 for a 64x64 field. Inverting the upsampled view would return something a
    few percent wrong that looks exactly like a reconstruction.
*   **Complex energy.** `c ** 2` for a complex coefficient is complex, and its sum is not an
    energy. It also does not raise.
*   **Level labels that are not levels.** Two pressure levels selected by nearest-neighbour
    from a dataset carrying only one return identical arrays, and a cross-level correlation
    computed from them is a field correlated with itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.core.errors import InvalidParameterError, ShapeMismatchError, UnknownNameError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import (BANK_FAMILIES, CoefficientField, LevelBank,
                                                    decompose_field, decompose_levels,
                                                    decompose_sequence)


def make_sequence(n_frames: int = 4, size: int = 64, seed: int = 0,
                  grid: GridSpec = None) -> FieldSequence:
    generator = torch.Generator().manual_seed(seed)
    fields = [PhysicalField(torch.randn(size, size, generator=generator, dtype=torch.float64),
                            grid=grid)
              for _ in range(n_frames)]
    return FieldSequence(fields, np.arange(n_frames) * 21600.0)


# ======================================================== T4B.1: coordinate alignment

@pytest.mark.parametrize("family", BANK_FAMILIES)
def test_every_scale_shares_the_parent_grid(family):
    """The stated acceptance criterion, for both families.

    SWT satisfies it by construction. DTCWT does not - its level-j subband is H/2**j - so this
    passing for `dtcwt` is a statement about the upsampling this module performs, which is why
    the next test checks that the upsampling is *declared*.
    """
    sequence = make_sequence()
    field = decompose_sequence(sequence, family, {"levels": 3})

    assert field.shape[3:] == (64, 64)
    assert tuple(field.grid.shape) == (64, 64)
    for scale in field.scales:
        for orientation in field.orientations:
            assert tuple(field.band(0, scale, orientation).shape) == (64, 64)


def test_a_decimated_family_declares_that_it_was_resampled():
    """The alignment above is real for SWT and constructed for DTCWT. Both say which."""
    sequence = make_sequence()

    swt = decompose_sequence(sequence, "swt", {"levels": 3})
    assert swt.resampled_to_parent is False
    assert swt.native_shapes == {}

    dtcwt = decompose_sequence(sequence, "dtcwt", {"levels": 3})
    assert dtcwt.resampled_to_parent is True
    assert dtcwt.native_shapes == {1: (32, 32), 2: (16, 16), 3: (8, 8)}, (
        "the native shapes are the honest statement of effective resolution; without them a "
        "reader would take the 64x64 aligned array at face value")
    assert "No information is added" in dtcwt.summary()["resampling"]


def test_upsampling_replicates_rather_than_interpolating():
    """Nearest neighbour: every parent pixel carries a coefficient that actually exists.

    Bilinear would manufacture intermediate complex values, blending phases that belong to
    different spatial positions - values no filter ever produced.
    """
    sequence = make_sequence(n_frames=1)
    field = decompose_sequence(sequence, "dtcwt", {"levels": 3})
    band = field.band(0, 3, field.orientations[0])
    # A level-3 band is 8x8 replicated to 64x64: each native coefficient covers an 8x8 block.
    block = band[0:8, 0:8]
    assert torch.all(block == block[0, 0]), "a replicated block must be constant"
    # `torch.unique` has no complex implementation, so count distinct values through the
    # real/imaginary pair - which is what "the same coefficient" means for a complex band.
    pairs = {(float(v.real), float(v.imag)) for v in band.flatten()}
    assert len(pairs) <= 64, (
        "an 8x8 native band cannot produce more than 64 distinct values once replicated; "
        "more would mean values were invented")


# ======================================================== T4B.1: perfect reconstruction

@pytest.mark.parametrize("family", BANK_FAMILIES)
def test_perfect_reconstruction_per_family(family):
    """The second stated acceptance criterion. float64 throughout, so 'perfect' means ~1e-15."""
    sequence = make_sequence()
    field = decompose_sequence(sequence, family, {"levels": 3}, keep_native=True)
    recovered = field.reconstruct()

    assert isinstance(recovered, FieldSequence)
    assert np.array_equal(recovered.times_seconds, sequence.times_seconds), (
        "reconstruction must return the time axis it was given, not frame indices")
    error = float((recovered.to_tensor() - sequence.to_tensor()).abs().max())
    assert error < 1e-12, "%s reconstruction error %.3g" % (family, error)


def test_reconstruction_uses_the_native_coefficients_not_the_aligned_view():
    """The failure this prevents is silent and small - the worst combination available.

    Inverting the upsampled DTCWT array would return a field a few percent wrong that is
    indistinguishable from a real reconstruction by inspection. So a field without its native
    coefficients refuses, by name, instead of approximating.
    """
    sequence = make_sequence()
    field = decompose_sequence(sequence, "dtcwt", {"levels": 3}, keep_native=False)

    assert field.has_native() is False
    assert field.summary()["reconstructable"] is False
    with pytest.raises(InvalidParameterError) as excinfo:
        field.reconstruct()
    assert "keep_native=True" in str(excinfo.value)


def test_drop_native_forfeits_reconstruction_explicitly():
    sequence = make_sequence()
    field = decompose_sequence(sequence, "swt", {"levels": 2}, keep_native=True)
    assert field.reconstruct() is not None
    field.drop_native()
    with pytest.raises(InvalidParameterError):
        field.reconstruct()


def test_a_subset_cannot_be_inverted():
    """`select()` narrows the view; a bank missing orientations is not invertible."""
    sequence = make_sequence()
    field = decompose_sequence(sequence, "swt", {"levels": 2}, keep_native=True)
    subset = field.select(orientations=["LH"])
    assert subset.has_native() is False
    with pytest.raises(InvalidParameterError):
        subset.reconstruct()


# ======================================================== T4B.1: the summary

def test_the_summary_is_lineage_safe():
    """The third stated acceptance criterion: a dict small enough for a database row."""
    sequence = make_sequence(n_frames=10)
    field = decompose_sequence(sequence, "dtcwt", {"levels": 4})
    record = field.summary()

    encoded = json.dumps(record, default=str)
    assert len(encoded.encode()) < 4096, "summary is %d bytes" % len(encoded.encode())
    # For contrast, the payload it describes:
    assert field.data.numel() * 16 > 2_000_000
    assert record["axes"] == ["time", "scale", "orientation", "y", "x"]
    assert record["is_complex"] is True
    assert record["orientation_convention"].startswith("feature orientation")


def test_the_summary_carries_no_coefficients():
    sequence = make_sequence(n_frames=2, size=32)
    field = decompose_sequence(sequence, "swt", {"levels": 2})
    encoded = json.dumps(field.summary(), default=str)
    for value in field.data.flatten()[:20].tolist():
        assert repr(value) not in encoded


def test_swt_band_labels_do_not_claim_angles():
    """`HH` responds to both diagonal signs; labelling it '45 deg' would assert selectivity
    the transform has not got, and that claim would then propagate into every figure."""
    sequence = make_sequence(n_frames=1, size=32)
    field = decompose_sequence(sequence, "swt", {"levels": 2})
    assert field.orientations == ["LH", "HL", "HH"]
    assert "cannot separate" in field.summary()["band_meaning"]["HH"]
    assert "deg" not in field.orientation_convention.replace("degrees", "")


# ======================================================== energy

def test_energy_uses_magnitude_for_a_complex_family():
    """`c ** 2` is complex and its sum is not an energy - and it does not raise."""
    sequence = make_sequence(n_frames=3)
    field = decompose_sequence(sequence, "dtcwt", {"levels": 2})
    energy = field.energy()

    assert energy.dtype == torch.float64
    assert energy.shape == (3, 2, 6)
    assert torch.all(energy >= 0), "an energy cannot be negative"
    manual = float((field.band(0, 1, field.orientations[0]).abs() ** 2).sum())
    assert abs(float(energy[0, 0, 0]) - manual) < 1e-9


def test_energy_fractions_are_invariant_to_a_global_rescale():
    """The property that makes two signatures comparable at all."""
    sequence = make_sequence(n_frames=3, size=32)
    scaled = FieldSequence([PhysicalField(f.data * 1000.0) for f in sequence],
                           sequence.times_seconds)

    a = decompose_sequence(sequence, "swt", {"levels": 2}).energy_fractions()
    b = decompose_sequence(scaled, "swt", {"levels": 2}).energy_fractions()
    assert torch.allclose(a, b, atol=1e-12)
    assert torch.allclose(a.sum(dim=(1, 2)), torch.ones(3, dtype=torch.float64))


def test_a_zero_field_reports_zero_rather_than_nan():
    """A frame with no energy has none to apportion; NaN would poison every later mean."""
    zeros = FieldSequence([PhysicalField(torch.zeros(32, 32, dtype=torch.float64))
                           for _ in range(2)], [0.0, 21600.0])
    fractions = decompose_sequence(zeros, "swt", {"levels": 2}).energy_fractions()
    assert torch.all(torch.isfinite(fractions))
    assert float(fractions.sum()) == 0.0


def test_a_sinusoid_concentrates_its_energy_at_one_scale():
    """A physical sanity check: the decomposition must respond to scale, not just run."""
    y, x = torch.meshgrid(torch.arange(64, dtype=torch.float64),
                          torch.arange(64, dtype=torch.float64), indexing="ij")
    fine = PhysicalField(torch.sin(2 * np.pi * x / 4.0))
    coarse = PhysicalField(torch.sin(2 * np.pi * x / 32.0))

    fine_fraction = decompose_field(fine, "swt", {"levels": 4}).energy_fractions()[0].sum(1)
    coarse_fraction = decompose_field(coarse, "swt", {"levels": 4}).energy_fractions()[0].sum(1)

    assert int(torch.argmax(fine_fraction)) < int(torch.argmax(coarse_fraction)), (
        "a 4-pixel wave must peak at a finer level than a 32-pixel wave")


# ======================================================== construction and refusals

def test_a_four_dimensional_array_is_refused():
    grid = GridSpec.pixel((8, 8))
    with pytest.raises(ShapeMismatchError) as excinfo:
        CoefficientField(torch.zeros(2, 3, 8, 8), wavelet_family="swt", scales=[1, 2, 3],
                         orientations=["LH"], times=[0.0, 1.0], grid=grid)
    assert "five-dimensional" in str(excinfo.value)


def test_unlabelled_axes_are_refused():
    grid = GridSpec.pixel((8, 8))
    with pytest.raises(ShapeMismatchError):
        CoefficientField(torch.zeros(2, 3, 1, 8, 8), wavelet_family="swt", scales=[1, 2],
                         orientations=["LH"], times=[0.0, 1.0], grid=grid)


def test_a_grid_that_disagrees_with_the_array_is_refused():
    """A grid of the wrong shape means the coordinates label different pixels than the data."""
    with pytest.raises(ShapeMismatchError) as excinfo:
        CoefficientField(torch.zeros(1, 1, 1, 8, 8), wavelet_family="swt", scales=[1],
                         orientations=["LH"], times=[0.0], grid=GridSpec.pixel((16, 16)))
    assert "parent grid" in str(excinfo.value)


def test_an_unknown_scale_is_refused_with_the_ones_that_exist():
    field = decompose_sequence(make_sequence(n_frames=1, size=32), "swt", {"levels": 2})
    with pytest.raises(UnknownNameError) as excinfo:
        field.band(0, 9, "LH")
    assert "1" in str(excinfo.value) and "2" in str(excinfo.value)


def test_a_non_bank_transform_is_refused_by_name():
    """`fft` is a registered transform with no (scale, orientation) factorisation."""
    with pytest.raises(UnknownNameError) as excinfo:
        decompose_sequence(make_sequence(n_frames=1, size=32), "fft", {})
    assert "swt" in str(excinfo.value)


def test_too_many_levels_for_the_field_is_refused():
    """Rule R13: a filter wider than the field measures the boundary treatment."""
    from src.core.errors import FieldTooSmallError

    with pytest.raises(FieldTooSmallError):
        decompose_sequence(make_sequence(n_frames=1, size=8), "dtcwt", {"levels": 6})


def test_select_returns_a_labelled_field_not_a_tensor():
    field = decompose_sequence(make_sequence(n_frames=4, size=32), "swt", {"levels": 3})
    subset = field.select(times=[0, 1], scales=[2], orientations=["LH", "HH"])
    assert isinstance(subset, CoefficientField)
    assert subset.shape == (2, 1, 2, 32, 32)
    assert subset.scales == [2] and subset.orientations == ["LH", "HH"]


def test_magnitude_field_is_a_physical_field_carrying_the_grid():
    field = decompose_sequence(make_sequence(n_frames=1, size=32), "dtcwt", {"levels": 2})
    band = field.magnitude_field(0, 1, field.orientations[0])
    assert isinstance(band, PhysicalField)
    assert tuple(band.data.shape) == (32, 32)
    assert band.data.dtype == torch.float64
    assert band.metadata["quantity"] == "coefficient magnitude"


# ======================================================== physical scale reporting

def test_wavelength_bands_are_reported_as_bands_on_a_physical_grid():
    """A wavelet level covers a band of wavelengths; quoting one number implies selectivity
    the filter has not got."""
    grid = GridSpec.latlon((64, 64), lat0=40.0, dlat=0.25, lon0=0.0, dlon=0.25)
    sequence = make_sequence(n_frames=1, size=64, grid=grid)
    bands = decompose_sequence(sequence, "swt", {"levels": 3}).scale_wavelength_bands()

    assert bands is not None and len(bands) == 3
    for band in bands:
        assert band["max_wavelength_x_m"] == pytest.approx(2 * band["min_wavelength_x_m"])
    assert bands[0]["min_wavelength_x_m"] < bands[2]["min_wavelength_x_m"]


def test_a_pixel_grid_reports_no_wavelength_rather_than_a_fabricated_one():
    """Metres derived from a grid with no physical spacing would be invented."""
    field = decompose_sequence(make_sequence(n_frames=1, size=32), "swt", {"levels": 2})
    assert field.scale_wavelength_bands() is None
    assert "scale_wavelength_bands" not in field.summary()


# ======================================================== T4B.4: the level bank

def make_level_banks(levels=(850.0, 500.0), n_frames=3, size=32):
    banks = {}
    for index, level in enumerate(levels):
        sequence = make_sequence(n_frames=n_frames, size=size, seed=index)
        banks[level] = decompose_sequence(sequence, "swt", {"levels": 2}, keep_native=False)
    return banks


def test_a_level_bank_orders_levels_by_pressure():
    bank = LevelBank(make_level_banks())
    assert bank.levels == [500.0, 850.0]
    assert len(bank) == 2
    assert bank.to_tensor().shape[0] == 2


def test_levels_sampled_at_different_times_are_refused():
    """A vertical lead-lag across mismatched time axes measures the sampling, not the sky."""
    banks = make_level_banks()
    other = decompose_sequence(
        FieldSequence(list(make_sequence(n_frames=3, size=32)), [0.0, 999.0, 5000.0]),
        "swt", {"levels": 2}, keep_native=False)
    banks[300.0] = other
    with pytest.raises(InvalidParameterError) as excinfo:
        LevelBank(banks)
    assert "sampling" in str(excinfo.value)


def test_levels_decomposed_differently_are_refused():
    banks = make_level_banks()
    banks[300.0] = decompose_sequence(make_sequence(n_frames=3, size=32), "swt",
                                      {"levels": 3}, keep_native=False)
    with pytest.raises(ShapeMismatchError):
        LevelBank(banks)


def test_mixed_families_in_one_level_bank_are_refused():
    banks = make_level_banks()
    banks[300.0] = decompose_sequence(make_sequence(n_frames=3, size=32), "dtcwt",
                                      {"levels": 2}, keep_native=False)
    with pytest.raises((InvalidParameterError, ShapeMismatchError)):
        LevelBank(banks)


def test_vertical_offsets_are_signed_and_name_their_direction():
    """`500 - 850 = -350` is *upward*: pressure decreases with height, and a reader who got
    that backwards would invert every precursor relationship the bank is built to find.

    The numbers are unchanged by TG1.5; the keys generalised and the rule that produced
    `"upward"` moved into the declared coordinate.
    """
    bank = LevelBank(make_level_banks())
    offsets = {(p["from_level"], p["to_level"]): p for p in bank.vertical_offsets()}

    upward = offsets[(850.0, 500.0)]
    assert upward["offset"] == -350.0
    assert upward["direction"] == "upward"
    assert upward["level_units"] == "hPa"
    assert offsets[(500.0, 850.0)]["direction"] == "downward"


def test_a_level_bank_states_that_it_is_not_a_3d_transform():
    """Scope discipline from the roadmap, asserted so a later reader cannot mistake it."""
    record = LevelBank(make_level_banks()).summary()
    assert "not a 3D wavelet transform" in record["scope"]
    assert record["levels"] == [500.0, 850.0]
    assert record["level_axis"] == "pressure_hpa"
    assert record["level_units"] == "hPa"


def test_an_empty_level_bank_is_refused():
    with pytest.raises(InvalidParameterError):
        LevelBank({})


def test_decompose_levels_builds_a_bank_from_sequences():
    sequences = {850.0: make_sequence(n_frames=3, size=32, seed=1),
                 500.0: make_sequence(n_frames=3, size=32, seed=2)}
    bank = decompose_levels(sequences, "swt", {"levels": 2})
    assert bank.levels == [500.0, 850.0]
    assert bank.at_level(850).wavelet_family == "swt"
    with pytest.raises(UnknownNameError):
        bank.at_level(700)


# ======================================================== T4B.4: adapter level slicing

def test_slice_level_sequences_returns_one_sequence_per_level():
    from src.data_layer.adapters import MeteorologicalDataAdapter

    sequences = MeteorologicalDataAdapter.slice_level_sequences(
        "era5_reanalysis", "z", [850, 500], time_range=("2023-01-01", "2023-01-03"))
    assert sorted(sequences) == [500.0, 850.0]
    assert sequences[850.0].metadata["level"] == 850.0
    assert np.array_equal(sequences[850.0].times_seconds, sequences[500.0].times_seconds), (
        "the shared time axis is the whole point of slicing the levels together")


def test_duplicate_levels_are_refused_rather_than_deduplicated():
    from src.data_layer.adapters import MeteorologicalDataAdapter

    with pytest.raises(ValueError) as excinfo:
        MeteorologicalDataAdapter.slice_level_sequences("era5_reanalysis", "z", [850, 850])
    assert "duplicate" in str(excinfo.value)


def test_levels_that_collapse_onto_one_stored_level_are_refused():
    """The guard that earned its place immediately.

    `t2m` has no vertical axis, so nearest-level selection returns the same field for every
    requested level. A cross-level correlation computed from that is a field correlated with
    itself: a coefficient of 1.0 that means nothing and looks like a discovery.
    """
    from src.data_layer.adapters import MeteorologicalDataAdapter

    with pytest.raises(ValueError) as excinfo:
        MeteorologicalDataAdapter.slice_level_sequences("era5_reanalysis", "t2m", [850, 500])
    message = str(excinfo.value)
    assert "identical data" in message
    assert "correlating a field with itself" in message


# ==================================================== native coefficients (added for T4C.1)

def test_the_aligned_view_subsamples_back_to_the_native_coefficients_exactly():
    """Nearest-neighbour alignment is replication, so the stride recovers the original.

    This is what lets a scale signature be computed from an artifact-restored field: the
    coefficients are still there, spread out, and the spreading is exactly invertible.
    """
    sequence = make_sequence(2)
    live = decompose_sequence(sequence, "dtcwt", {"levels": 3}, keep_native=True)
    restored = live.drop_native()

    for scale in live.scales:
        for orientation in live.orientations:
            assert torch.equal(live.native_band(0, scale, orientation),
                               restored.native_band(0, scale, orientation))
    assert tuple(live.native_band(0, 3, live.orientations[0]).shape) == (8, 8)


def test_the_replication_factor_is_exactly_four_to_the_level():
    field = decompose_sequence(make_sequence(1), "dtcwt", {"levels": 3}, keep_native=False)
    assert [field.replication_factor(s) for s in field.scales] == [4.0, 16.0, 64.0]
    swt = decompose_sequence(make_sequence(1), "swt", {"levels": 3}, keep_native=False)
    assert [swt.replication_factor(s) for s in swt.scales] == [1.0, 1.0, 1.0]


def test_aligned_energy_is_inflated_but_the_fraction_is_not():
    """The correction that turned out not to be needed, pinned so it stays that way.

    The aligned band at level `j` repeats every native coefficient `4**j` times, so its
    summed energy is inflated by exactly that factor - and the obvious conclusion, that the
    energy fractions are therefore biased towards coarse scales, is **wrong**: the parent
    grid has the same number of cells at every scale, so the factor cancels in the ratio.
    Asserting both halves keeps a future 'fix' from breaking a quantity that is already right.
    """
    field = decompose_sequence(make_sequence(2), "dtcwt", {"levels": 3}, keep_native=True)
    aligned = field.energy()
    native = field.native_energy()

    for index, scale in enumerate(field.scales):
        assert torch.allclose(aligned[:, index, :],
                              native[:, index, :] * field.replication_factor(scale))

    aligned_fraction = aligned.sum(dim=2) / aligned.sum(dim=(1, 2), keepdim=True)[:, :, 0]
    density = field.energy_density().sum(dim=2)
    density_fraction = density / density.sum(dim=1, keepdim=True)
    assert torch.allclose(aligned_fraction, density_fraction)


def test_available_coefficients_counts_what_the_transform_computed():
    field = decompose_sequence(make_sequence(1), "dtcwt", {"levels": 3}, keep_native=False)
    assert field.available_coefficients() == [32 * 32 * 6, 16 * 16 * 6, 8 * 8 * 6]
    swt = decompose_sequence(make_sequence(1), "swt", {"levels": 3}, keep_native=False)
    assert swt.available_coefficients() == [64 * 64 * 3] * 3


def test_the_dtcwt_filter_geometry_grows_with_level():
    """Rule R13's margin, computed from the dual tree's own cascade rather than assumed.

    The R13 table is written for a single 14-tap filter repeated at every level. The dual
    tree is not that: level 1 uses the 19-tap near-symmetric highpass and levels above use
    the q-shift pair, so the parent-grid margin is larger than the table suggests - which is
    why a 256x256 crop leaves DTCWT level 4 almost nothing.
    """
    from src.transform_engine import dtcwt as dtcwt_mod

    parent = [dtcwt_mod.valid_interior_halfwidth(j) for j in range(1, 5)]
    assert parent == sorted(parent) and parent[0] < parent[-1]
    assert parent[3] > 52, "the 14-tap table's level-4 margin understates the dual tree's"

    native = [dtcwt_mod.native_halfwidth(j) for j in range(1, 5)]
    assert max(native) - min(native) <= 3, (
        "in native samples the margin is roughly constant across levels, which is the "
        "standard property of a decimated pyramid")
