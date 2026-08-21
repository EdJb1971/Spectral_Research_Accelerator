"""Acceptance tests for the training-native representation seam (T5.1a-d)."""

from dataclasses import replace

import pytest
import pywt
import torch

from src.transform_engine.training import (
    DCTRepresentation,
    DTCWTRepresentation,
    DB2Representation,
    EncodedRepresentation,
    FFTRepresentation,
    HaarRepresentation,
    RawRepresentation,
    RepresentationError,
    SWTRepresentation,
    REPRESENTATIONS,
    make_representation,
    training_representation_catalogue,
)
from src.physical_core.field import PhysicalField
from src.transform_engine.stationary import apply_swt2d
from src.transform_engine.dtcwt import apply_dtcwt2d


@pytest.mark.parametrize("shape", [(2, 3, 8, 10), (1, 2, 7, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_first_slice_representations_reconstruct(shape, dtype):
    torch.manual_seed(5)
    inputs = torch.randn(shape, dtype=dtype)
    modules = (
        RawRepresentation(),
        FFTRepresentation(),
        DCTRepresentation(shape[-2:], dtype=dtype),
    )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    for module in modules:
        reconstructed = module.inverse(module(inputs))
        assert reconstructed.shape == inputs.shape
        assert reconstructed.dtype == inputs.dtype
        assert torch.allclose(reconstructed, inputs, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("module", [RawRepresentation(), FFTRepresentation(norm="ortho")])
def test_stateless_modules_equal_itemwise_encoding(module):
    inputs = torch.randn(3, 2, 9, 10)
    batch = module(inputs).values
    itemwise = torch.cat([module(inputs[i:i + 1]).values for i in range(3)], dim=0)
    assert torch.allclose(batch, itemwise)


def test_dct_batch_encoding_equals_itemwise_encoding():
    module = DCTRepresentation((9, 10))
    inputs = torch.randn(3, 2, 9, 10)
    batch = module(inputs).values
    itemwise = torch.cat([module(inputs[i:i + 1]).values for i in range(3)], dim=0)
    assert torch.allclose(batch, itemwise)


def test_fft_uses_explicit_real_imaginary_channel_packing():
    inputs = torch.randn(2, 3, 7, 10)
    encoded = FFTRepresentation()(inputs)
    assert encoded.values.shape == (2, 6, 7, 6)
    assert encoded.layout == "b_2c_h_rfftwidth__real_then_imag"
    assert encoded.metadata["complex_packing"] == "real_channels_then_imag_channels"
    assert not encoded.values.is_complex()


@pytest.mark.parametrize(
    "module",
    [RawRepresentation(), FFTRepresentation(norm="ortho"),
     DCTRepresentation((4, 5), dtype=torch.float64)],
)
def test_forward_inverse_passes_gradcheck(module):
    module = module.double()
    inputs = torch.randn(1, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: module.inverse(module(value)), (inputs,))


def test_model_output_can_reuse_immutable_synthesis_context():
    module = FFTRepresentation()
    inputs = torch.randn(2, 2, 8, 9)
    encoded = module(inputs)
    prediction = encoded.values * 0.5
    predicted_context = encoded.with_values(prediction)
    assert predicted_context is not encoded
    assert predicted_context.original_shape == encoded.original_shape
    assert torch.allclose(module.inverse(predicted_context), inputs * 0.5, atol=1e-6)


def test_dct_matrices_are_registered_cached_buffers_reused_across_calls():
    module = DCTRepresentation((8, 10))
    pointers = (module.matrix_y.data_ptr(), module.matrix_x.data_ptr())
    state = module.state_dict()
    assert set(state) == {"matrix_y", "matrix_x"}
    module(torch.randn(2, 3, 8, 10))
    module(torch.randn(1, 1, 8, 10))
    assert pointers == (module.matrix_y.data_ptr(), module.matrix_x.data_ptr())


def test_dct_buffers_follow_module_dtype_and_preserve_reconstruction():
    module = DCTRepresentation((7, 9)).double()
    inputs = torch.randn(2, 2, 7, 9, dtype=torch.float64)
    assert module.matrix_y.dtype == torch.float64
    assert module.matrix_x.dtype == torch.float64
    assert torch.allclose(module.inverse(module(inputs)), inputs, atol=1e-11, rtol=1e-11)


def test_float32_dct_basis_is_accurate_on_the_target_regional_shape():
    torch.manual_seed(12)
    module = DCTRepresentation((120, 80))
    inputs = torch.randn(2, 3, 120, 80)
    error = torch.max(torch.abs(module.inverse(module(inputs)) - inputs))
    assert float(error) < 5e-6


def _available_accelerator_devices():
    """PyTorch device names, not vendor names; ROCm deliberately uses ``cuda`` here."""
    devices = []
    if torch.cuda.is_available():
        devices.extend("cuda:%d" % index for index in range(torch.cuda.device_count()))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize(
    "device_name",
    _available_accelerator_devices() or [pytest.param(
        None, marks=pytest.mark.skip(reason="no supported PyTorch accelerator is available"))],
)
def test_accelerator_reconstruction_gradients_and_cpu_parity(device_name):
    torch.manual_seed(22)
    device = torch.device(device_name)
    cpu_inputs = torch.randn(2, 2, 8, 10)
    constructors = (
        RawRepresentation,
        FFTRepresentation,
        lambda: DCTRepresentation((8, 10)),
        lambda: HaarRepresentation(levels=2),
        lambda: DB2Representation(levels=2),
        lambda: SWTRepresentation(wavelet="db2", levels=1),
    )
    for constructor in constructors:
        cpu_module = constructor()
        accelerator_module = constructor().to(device)
        cpu_encoded = cpu_module(cpu_inputs)
        accelerator_inputs = cpu_inputs.to(device).requires_grad_(True)
        accelerator_encoded = accelerator_module(accelerator_inputs)
        accelerator_reconstructed = accelerator_module.inverse(accelerator_encoded)

        assert torch.allclose(
            accelerator_encoded.values.detach().cpu(), cpu_encoded.values,
            atol=3e-5, rtol=3e-5,
        )
        assert torch.allclose(
            accelerator_reconstructed.detach().cpu(), cpu_inputs,
            atol=3e-5, rtol=3e-5,
        )
        accelerator_reconstructed.square().mean().backward()
        assert accelerator_inputs.grad is not None
        assert bool(torch.isfinite(accelerator_inputs.grad).all())


@pytest.mark.parametrize(
    "bad_input,message",
    [
        (torch.randn(2, 8, 8), "(B,C,H,W)"),
        (torch.ones(1, 1, 4, 4, dtype=torch.int64), "float32 or float64"),
        (torch.empty(1, 0, 4, 4), "empty axis"),
    ],
)
def test_input_contract_rejects_bad_batches(bad_input, message):
    with pytest.raises(RepresentationError, match=message):
        RawRepresentation()(bad_input)


def test_dct_rejects_shape_or_dtype_drift_instead_of_rebuilding_cache():
    module = DCTRepresentation((8, 10))
    with pytest.raises(RepresentationError, match="configured for spatial shape"):
        module(torch.randn(1, 1, 9, 10))
    with pytest.raises(RepresentationError, match="move the module"):
        module(torch.randn(1, 1, 8, 10, dtype=torch.float64))


def test_inverse_rejects_wrong_representation_or_coefficient_shape():
    fft = FFTRepresentation()
    encoded = fft(torch.randn(1, 2, 8, 9))
    with pytest.raises(RepresentationError, match="cannot invert"):
        RawRepresentation().inverse(encoded)
    malformed = replace(encoded, values=encoded.values[..., :-1])
    with pytest.raises(RepresentationError, match="must have shape"):
        fft.inverse(malformed)


def test_factory_has_stable_names_and_clear_unknown_error():
    assert isinstance(make_representation(" RAW "), RawRepresentation)
    assert isinstance(make_representation("fft", norm="ortho"), FFTRepresentation)
    assert isinstance(make_representation("dct", spatial_shape=(8, 10)), DCTRepresentation)
    assert isinstance(make_representation("haar", levels=2), HaarRepresentation)
    assert isinstance(make_representation("db2", levels=2), DB2Representation)
    assert isinstance(make_representation("swt", levels=2), SWTRepresentation)
    assert isinstance(make_representation("dtcwt", levels=1), DTCWTRepresentation)
    with pytest.raises(
        RepresentationError, match="available: db2, dct, dtcwt, fft, haar, raw, swt"
    ):
        make_representation("curvelet")


def test_encoded_context_requires_tensor_replacement():
    encoded = EncodedRepresentation(
        values=torch.zeros(1, 1, 2, 2), representation="raw",
        original_shape=(1, 1, 2, 2), layout="bchw")
    with pytest.raises(RepresentationError, match="torch.Tensor"):
        encoded.with_values([[1.0]])


def test_encoded_metadata_is_actually_immutable():
    encoded = FFTRepresentation()(torch.randn(1, 1, 4, 4))
    with pytest.raises(TypeError):
        encoded.metadata["norm"] = "forward"


@pytest.mark.parametrize("module_type", [HaarRepresentation, DB2Representation])
@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decimated_wavelets_reconstruct_batched_odd_shapes(module_type, levels, dtype):
    torch.manual_seed(31)
    inputs = torch.randn(2, 3, 15, 10, dtype=dtype)
    module = module_type(levels=levels, dtype=dtype)
    reconstructed = module.inverse(module(inputs))
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    assert reconstructed.shape == inputs.shape
    assert torch.allclose(reconstructed, inputs, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "module_type,wavelet", [(HaarRepresentation, "haar"), (DB2Representation, "db2")]
)
def test_level_one_packed_bands_match_the_pywavelets_oracle(module_type, wavelet):
    torch.manual_seed(32)
    inputs = torch.randn(1, 1, 8, 10, dtype=torch.float64)
    packed = module_type(levels=1, dtype=torch.float64)(inputs).values[0, 0]
    half_h, half_w = 4, 5
    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(
        inputs[0, 0].numpy(), wavelet, mode="periodization"
    )
    assert torch.allclose(packed[:half_h, :half_w], torch.from_numpy(approximation), atol=1e-12)
    assert torch.allclose(packed[:half_h, half_w:], torch.from_numpy(vertical), atol=1e-12)
    assert torch.allclose(packed[half_h:, :half_w], torch.from_numpy(horizontal), atol=1e-12)
    assert torch.allclose(packed[half_h:, half_w:], torch.from_numpy(diagonal), atol=1e-12)


@pytest.mark.parametrize(
    "module_type,wavelet", [(HaarRepresentation, "haar"), (DB2Representation, "db2")]
)
def test_multilevel_packed_energy_matches_pywavelets(module_type, wavelet):
    torch.manual_seed(33)
    inputs = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    packed = module_type(levels=3, dtype=torch.float64)(inputs).values
    oracle = pywt.wavedec2(inputs[0, 0].numpy(), wavelet, mode="periodization", level=3)
    oracle_energy = float((oracle[0] ** 2).sum())
    for horizontal, vertical, diagonal in oracle[1:]:
        oracle_energy += float((horizontal ** 2).sum())
        oracle_energy += float((vertical ** 2).sum())
        oracle_energy += float((diagonal ** 2).sum())
    assert float(torch.sum(packed ** 2)) == pytest.approx(oracle_energy, abs=1e-10)


def test_db2_metadata_exposes_padding_redundancy_and_corrected_support():
    encoded = DB2Representation(levels=3)(torch.randn(2, 3, 15, 10))
    assert encoded.values.shape == (2, 3, 16, 16)
    assert encoded.layout == "mallat_quadrants_LL_LH_HL_HH"
    assert encoded.metadata["padding_bottom_right"] == (1, 6)
    assert encoded.metadata["valid_interior_halfwidth_by_level"] == (2, 5, 11)
    assert encoded.metadata["coefficient_count_ratio"] == pytest.approx(256 / 150)
    assert encoded.metadata["decimation_phase"] == \
        "pywavelets_periodization_even_length_anchor"
    assert encoded.metadata["implementation"] == "conv2d"


def test_wavelet_can_refuse_implicit_dyadic_padding():
    module = DB2Representation(levels=3, pad_to_multiple=False)
    with pytest.raises(RepresentationError, match="not divisible"):
        module(torch.randn(1, 1, 15, 10))


def test_decimated_wavelet_batch_encoding_equals_itemwise_encoding():
    module = DB2Representation(levels=2)
    inputs = torch.randn(3, 2, 12, 20)
    batch = module(inputs).values
    itemwise = torch.cat([module(inputs[i:i + 1]).values for i in range(3)], dim=0)
    assert torch.allclose(batch, itemwise)


@pytest.mark.parametrize("module_type", [HaarRepresentation, DB2Representation])
@pytest.mark.parametrize("implementation", ["conv2d", "reference"])
def test_decimated_wavelet_forward_inverse_passes_gradcheck(module_type, implementation):
    module = module_type(
        levels=2, dtype=torch.float64, implementation=implementation
    )
    inputs = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: module.inverse(module(value)), (inputs,))


@pytest.mark.parametrize("module_type", [HaarRepresentation, DB2Representation])
def test_fused_wavelet_matches_reference_coefficients_inverse_and_gradient(module_type):
    torch.manual_seed(34)
    fused = module_type(levels=3, dtype=torch.float64, implementation="conv2d")
    reference = module_type(levels=3, dtype=torch.float64, implementation="reference")
    fused_input = torch.randn(2, 3, 16, 24, dtype=torch.float64, requires_grad=True)
    reference_input = fused_input.detach().clone().requires_grad_(True)

    fused_encoded = fused(fused_input)
    reference_encoded = reference(reference_input)
    assert torch.allclose(
        fused_encoded.values, reference_encoded.values, atol=2e-12, rtol=2e-12
    )
    assert torch.allclose(
        fused.inverse(fused_encoded), reference.inverse(reference_encoded),
        atol=2e-12, rtol=2e-12,
    )

    weights = torch.randn_like(fused_encoded.values)
    torch.sum(fused_encoded.values * weights).backward()
    torch.sum(reference_encoded.values * weights).backward()
    assert torch.allclose(fused_input.grad, reference_input.grad, atol=2e-12, rtol=2e-12)


def test_fused_wavelet_bank_is_a_registered_migration_aware_cache():
    module = DB2Representation(levels=2)
    pointer = module.analysis_bank.data_ptr()
    assert module.analysis_bank.shape == (4, 1, 4, 4)
    assert "analysis_bank" in module.state_dict()
    module(torch.randn(2, 3, 8, 12))
    module(torch.randn(1, 1, 8, 12))
    assert module.analysis_bank.data_ptr() == pointer

    migrated = module.double()
    assert migrated.analysis_bank.dtype == torch.float64
    expected = migrated._make_analysis_bank(migrated.lowpass, migrated.highpass)
    assert torch.equal(migrated.analysis_bank, expected)


def test_wavelet_implementation_is_explicit_and_cannot_drift_at_inverse():
    with pytest.raises(RepresentationError, match="implementation must be"):
        DB2Representation(implementation="magic")
    encoded = DB2Representation(implementation="reference")(torch.randn(1, 1, 8, 8))
    with pytest.raises(RepresentationError, match="implementation metadata"):
        DB2Representation(implementation="conv2d").inverse(encoded)


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3"])
@pytest.mark.parametrize("implementation", ["conv2d", "fft_reference"])
def test_swt_reconstructs_batched_tensors_and_preserves_parent_grid(wavelet, implementation):
    torch.manual_seed(40)
    inputs = torch.randn(2, 3, 48, 40, dtype=torch.float64)
    module = SWTRepresentation(
        wavelet=wavelet, levels=3, dtype=torch.float64, implementation=implementation
    )
    encoded = module(inputs)
    assert encoded.values.shape == (2, 30, 48, 40)
    assert torch.allclose(module.inverse(encoded), inputs, atol=1e-11, rtol=1e-11)


def test_swt_packed_coefficients_match_the_coordinate_aware_analytical_path():
    torch.manual_seed(41)
    inputs = torch.randn(2, 2, 32, 40, dtype=torch.float64)
    encoded = SWTRepresentation(
        wavelet="db2", levels=3, dtype=torch.float64
    )(inputs)
    unpacked = encoded.values.reshape(2, 2, 10, 32, 40)
    for batch in range(2):
        for channel in range(2):
            oracle = apply_swt2d(
                PhysicalField(inputs[batch, channel]), levels=3,
                wavelet="db2", mode="periodic",
            )
            assert torch.allclose(unpacked[batch, channel, 0], oracle["LL"], atol=1e-11)
            for level in range(1, 4):
                offset = 1 + 3 * (level - 1)
                for band_offset, band in enumerate(("LH", "HL", "HH")):
                    assert torch.allclose(
                        unpacked[batch, channel, offset + band_offset],
                        oracle["level_%d" % level][band], atol=1e-11,
                    )


def test_swt_conv_and_fft_oracles_match_coefficients_inverse_and_gradient():
    torch.manual_seed(42)
    fused = SWTRepresentation("db2", levels=3, dtype=torch.float64, implementation="conv2d")
    oracle = SWTRepresentation(
        "db2", levels=3, dtype=torch.float64, implementation="fft_reference"
    )
    fused_input = torch.randn(1, 2, 32, 40, dtype=torch.float64, requires_grad=True)
    oracle_input = fused_input.detach().clone().requires_grad_(True)
    fused_encoded, oracle_encoded = fused(fused_input), oracle(oracle_input)
    assert torch.allclose(fused_encoded.values, oracle_encoded.values, atol=2e-11, rtol=2e-11)
    assert torch.allclose(
        fused.inverse(fused_encoded), oracle.inverse(oracle_encoded), atol=2e-11, rtol=2e-11
    )
    weights = torch.randn_like(fused_encoded.values)
    torch.sum(fused_encoded.values * weights).backward()
    torch.sum(oracle_encoded.values * weights).backward()
    assert torch.allclose(fused_input.grad, oracle_input.grad, atol=2e-11, rtol=2e-11)


def test_swt_is_exactly_translation_equivariant_under_declared_periodic_boundary():
    torch.manual_seed(43)
    module = SWTRepresentation("db2", levels=3, dtype=torch.float64)
    inputs = torch.randn(1, 2, 32, 40, dtype=torch.float64)
    shifted = torch.roll(inputs, shifts=(5, -7), dims=(-2, -1))
    expected = torch.roll(module(inputs).values, shifts=(5, -7), dims=(-2, -1))
    assert torch.allclose(module(shifted).values, expected, atol=2e-11, rtol=2e-11)


def test_swt_metadata_makes_redundancy_boundaries_and_band_meaning_explicit():
    encoded = SWTRepresentation("db2", levels=3)(torch.randn(2, 5, 32, 40))
    assert encoded.layout == "per_input_channel__LL_then_levelwise_LH_HL_HH"
    assert encoded.metadata["bands_per_input_channel"] == 10
    assert encoded.metadata["coefficient_count_ratio"] == 10.0
    assert encoded.metadata["accumulated_support_by_level"] == (4, 10, 22)
    assert encoded.metadata["valid_interior_halfwidth_by_level"] == (2, 5, 11)
    assert encoded.metadata["valid_interior_shape_by_level"] == ((28, 36), (22, 30), (10, 18))
    assert encoded.metadata["level_amplitude_gain_by_level"] == (2, 4, 8)
    assert encoded.metadata["implementation_policy"] == "auto"
    assert encoded.metadata["implementation"] == "fft_reference"
    assert "not six signed orientations" in encoded.metadata["directionality"]
    assert "exclude" in encoded.metadata["regional_boundary_warning"]


def test_swt_gradcheck_cache_migration_and_support_refusal():
    module = SWTRepresentation("db2", levels=1, dtype=torch.float64)
    inputs = torch.randn(1, 1, 6, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: module.inverse(module(value)), (inputs,))
    pointers = tuple(module._bank(level).data_ptr() for level in range(1, 2))
    module(torch.randn(1, 1, 6, 7, dtype=torch.float64))
    assert pointers == tuple(module._bank(level).data_ptr() for level in range(1, 2))
    assert SWTRepresentation("db2", levels=2).double()._bank(2).dtype == torch.float64
    with pytest.raises(RepresentationError, match="accumulated support"):
        SWTRepresentation("db3", levels=3)(torch.randn(1, 1, 32, 40))


def test_swt_rejects_unknown_implementation_and_mismatched_inverse_context():
    with pytest.raises(RepresentationError, match="implementation must be"):
        SWTRepresentation(implementation="magic")
    encoded = SWTRepresentation(implementation="fft_reference")(
        torch.randn(1, 1, 8, 8)
    )
    with pytest.raises(RepresentationError, match="implementation metadata"):
        SWTRepresentation(implementation="conv2d").inverse(encoded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA precision-policy regression")
def test_swt_is_not_changed_by_ambient_deterministic_tf32_cudnn_policy():
    torch.manual_seed(22)
    inputs = torch.randn(2, 2, 8, 10)
    module = SWTRepresentation("db2", levels=1).cuda()
    with torch.backends.cudnn.flags(deterministic=True, benchmark=False, allow_tf32=True):
        assert torch.backends.cudnn.allow_tf32
        reconstructed = module.inverse(module(inputs.cuda())).cpu()
        # The module's local full-precision scope must not mutate its caller's global setting.
        assert torch.backends.cudnn.allow_tf32
    assert torch.allclose(reconstructed, inputs, atol=3e-6, rtol=3e-6)


def test_training_catalogue_distinguishes_accepted_modules_from_analysis_only_candidates():
    catalogue = training_representation_catalogue(4, "db2", (64, 64))
    by_name = {entry["name"]: entry for entry in catalogue["representations"]}
    assert {name for name, entry in by_name.items() if entry["status"] == "accepted"} == \
        set(REPRESENTATIONS)
    assert by_name["dtcwt"]["status"] == "accepted"
    assert by_name["dtcwt"]["exact_inverse"]
    assert by_name["dtcwt"]["autograd"]

    selected = by_name["swt"]["selected_configuration"]
    assert selected["coefficient_channels_per_input_channel"] == 13
    assert selected["accumulated_support_by_level"] == (4, 10, 22, 46)
    assert selected["valid_interior_halfwidth_by_level"] == (2, 5, 11, 23)
    assert selected["valid_interior_shape_by_level"] == (
        (60, 60), (54, 54), (42, 42), (18, 18)
    )
    assert selected["coarsest_scale_has_valid_interior"]
    assert "AMD ROCm hardware" in by_name["swt"]["not_run"]

    selected_dtcwt = by_name["dtcwt"]["selected_configuration"]
    assert selected_dtcwt["atlas_planes_per_input_channel"] == 4
    assert selected_dtcwt["valid_interior_halfwidth_parent_px_by_level"] == (9, 19, 45, 97)
    assert not selected_dtcwt["coarsest_scale_has_valid_interior"]


def test_training_catalogue_marks_an_exhausted_selected_swt_interior():
    catalogue = training_representation_catalogue(4, "db2", (32, 40))
    swt = next(entry for entry in catalogue["representations"] if entry["name"] == "swt")
    assert not swt["selected_configuration"]["coarsest_scale_has_valid_interior"]
    assert swt["selected_configuration"]["valid_interior_shape_by_level"][-1] == (0, 0)


@pytest.mark.parametrize("levels,shape", [(1, (32, 40)), (2, (64, 72)), (3, (128, 112))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtcwt_reconstructs_batched_coefficients_without_discarding_complex_parts(
    levels, shape, dtype
):
    torch.manual_seed(50)
    module = DTCWTRepresentation(levels=levels, dtype=dtype)
    inputs = torch.randn(2, 2, *shape, dtype=dtype)
    encoded = module(inputs)
    reconstructed = module.inverse(encoded)
    tolerance = 4e-5 if dtype == torch.float32 else 2e-11
    assert encoded.values.shape == (2, 8, *shape)
    assert not encoded.values.is_complex()
    assert torch.allclose(reconstructed, inputs, atol=tolerance, rtol=tolerance)


def test_dtcwt_atlas_unpacks_to_the_coordinate_aware_complex_oracle():
    torch.manual_seed(51)
    module = DTCWTRepresentation(levels=2, dtype=torch.float64)
    inputs = torch.randn(2, 2, 64, 72, dtype=torch.float64)
    encoded = module(inputs)
    atlas = encoded.values.reshape(4, 4, 64, 72)
    lowpass, bands = module._unpack(atlas)
    for item, field in enumerate(inputs.reshape(4, 64, 72)):
        oracle = apply_dtcwt2d(PhysicalField(field), levels=2)
        assert torch.allclose(lowpass[item], oracle["lowpass"], atol=2e-11, rtol=2e-11)
        for actual, expected in zip(bands, oracle["highpass"]):
            assert torch.allclose(actual[item], expected, atol=2e-11, rtol=2e-11)


def test_dtcwt_atlas_is_an_exact_bijection_for_arbitrary_model_coefficients():
    module = DTCWTRepresentation(levels=3, dtype=torch.float64)
    inputs = torch.randn(1, 1, 128, 112, dtype=torch.float64)
    encoded = module(inputs)
    atlas = encoded.values.reshape(1, 4, 128, 112)
    lowpass, bands = module._unpack(atlas)
    assert torch.equal(module._pack(lowpass, bands), atlas)


def test_dtcwt_random_coefficient_loss_and_roundtrip_are_autograd_clean():
    torch.manual_seed(52)
    module = DTCWTRepresentation(levels=1, dtype=torch.float64)
    inputs = torch.randn(1, 1, 24, 24, dtype=torch.float64, requires_grad=True)
    encoded = module(inputs)
    torch.sum(encoded.values * torch.randn_like(encoded.values)).backward()
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    fresh = inputs.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda value: module.inverse(module(value)), (fresh,), fast_mode=True
    )


def test_dtcwt_metadata_prevents_visual_and_boundary_misinterpretation():
    encoded = DTCWTRepresentation(levels=2)(torch.randn(1, 2, 64, 72))
    metadata = encoded.metadata
    assert encoded.layout == "per_input_channel__four_plane_recursive_complex_atlas"
    assert metadata["coefficient_count_ratio"] == 4.0
    assert metadata["component_order"] == "Re0,Im0,Re1,Im1,...,Re5,Im5"
    assert metadata["valid_interior_halfwidth_parent_px_by_level"] == (9, 19)
    assert metadata["valid_interior_halfwidth_native_px_by_level"] == (5, 5)
    assert metadata["valid_interior_native_shape_by_level"] == ((22, 26), (6, 8))
    assert "not physical spatial adjacency" in metadata["atlas_warning"]
    assert "never interpolate" in metadata["display_contract"]
    assert "7.1 degrees" in metadata["level1_orientation_warning"]


def test_dtcwt_refuses_a_configuration_with_no_valid_interior():
    with pytest.raises(RepresentationError, match="leaving no valid interior"):
        DTCWTRepresentation(levels=3)(torch.randn(1, 1, 120, 80))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA DTCWT acceptance")
def test_dtcwt_cuda_coefficients_inverse_gradients_and_cpu_parity():
    torch.manual_seed(53)
    inputs = torch.randn(1, 1, 64, 72)
    cpu = DTCWTRepresentation(levels=2)
    gpu = DTCWTRepresentation(levels=2).cuda()
    expected = cpu(inputs)
    gpu_inputs = inputs.cuda().requires_grad_(True)
    actual = gpu(gpu_inputs)
    reconstructed = gpu.inverse(actual)
    assert torch.allclose(actual.values.detach().cpu(), expected.values, atol=4e-5, rtol=4e-5)
    assert torch.allclose(reconstructed.detach().cpu(), inputs, atol=4e-5, rtol=4e-5)
    reconstructed.square().mean().backward()
    assert gpu_inputs.grad is not None and bool(torch.isfinite(gpu_inputs.grad).all())
