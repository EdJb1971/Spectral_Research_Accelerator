"""Acceptance tests for the training-native representation seam (T5.1 first slice)."""

from dataclasses import replace

import pytest
import pywt
import torch

from src.transform_engine.training import (
    DCTRepresentation,
    DB2Representation,
    EncodedRepresentation,
    FFTRepresentation,
    HaarRepresentation,
    RawRepresentation,
    RepresentationError,
    make_representation,
)


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
    with pytest.raises(RepresentationError, match="available: db2, dct, fft, haar, raw"):
        make_representation("dtcwt")


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
def test_decimated_wavelet_forward_inverse_passes_gradcheck(module_type):
    module = module_type(levels=2, dtype=torch.float64)
    inputs = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: module.inverse(module(value)), (inputs,))
