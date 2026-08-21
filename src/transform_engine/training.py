"""Training-native spectral representations.

The analytical transform registry operates on :class:`PhysicalField`, intentionally a single
coordinate-aware 2D field.  A forecasting model instead needs a conventional PyTorch module
over ``(batch, channel, height, width)`` tensors.  This module is that boundary: it shares the
canonical transform arithmetic without weakening the scientific data model into an arbitrary
batch container.

Only representations that meet the training acceptance tests belong in ``REPRESENTATIONS``.
The accepted numerical paths are identity, real FFT, DCT-II, and decimated Haar/db2 with a
structured Mallat layout. SWT and DTCWT join only after their batched synthesis paths meet the
same contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from src.transform_engine.stationary import (
    get_filters as get_wavelet_filters,
    valid_interior_halfwidth,
)
from src.transform_engine.transforms import get_dct_matrix


class RepresentationError(ValueError):
    """A tensor or encoded batch violates a representation's declared contract."""


@dataclass(frozen=True)
class EncodedRepresentation:
    """A model-ready tensor together with the information needed for exact synthesis.

    ``values`` remains an ordinary autograd-connected tensor.  A forecaster consumes it and
    returns another tensor with the same representation layout; ``with_values`` transfers the
    immutable synthesis context to that prediction without storing per-call state on the
    module (which would be unsafe under concurrent data-parallel execution).
    """

    values: torch.Tensor
    representation: str
    original_shape: Tuple[int, int, int, int]
    layout: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def with_values(self, values: torch.Tensor) -> "EncodedRepresentation":
        """Return the same synthesis context carrying model-produced coefficients."""
        if not isinstance(values, torch.Tensor):
            raise RepresentationError("encoded values must be a torch.Tensor")
        return replace(self, values=values)


class RepresentationModule(nn.Module, ABC):
    """Base contract for differentiable ``(B,C,H,W)`` representations."""

    name: str
    supported_dtypes = (torch.float32, torch.float64)

    def forward(self, inputs: torch.Tensor) -> EncodedRepresentation:
        self._validate_inputs(inputs)
        return self.encode(inputs)

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        """Encode a validated batch without detaching its autograd graph."""

    @abstractmethod
    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        """Reconstruct a ``(B,C,H,W)`` tensor from an encoded batch."""

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise RepresentationError("representation input must be a torch.Tensor")
        if inputs.ndim != 4:
            raise RepresentationError(
                "representation input must have shape (B,C,H,W), got %r"
                % (tuple(inputs.shape),)
            )
        if any(size < 1 for size in inputs.shape):
            raise RepresentationError(
                "representation input cannot contain an empty axis, got %r"
                % (tuple(inputs.shape),)
            )
        if inputs.dtype not in self.supported_dtypes:
            raise RepresentationError(
                "representation input dtype must be float32 or float64, got %s"
                % inputs.dtype
            )

    def _validate_encoded(self, encoded: EncodedRepresentation) -> None:
        if not isinstance(encoded, EncodedRepresentation):
            raise RepresentationError("inverse expects an EncodedRepresentation")
        if encoded.representation != self.name:
            raise RepresentationError(
                "%s cannot invert coefficients produced by %s"
                % (self.name, encoded.representation)
            )
        if encoded.values.dtype not in self.supported_dtypes:
            raise RepresentationError(
                "encoded values must be float32 or float64, got %s"
                % encoded.values.dtype
            )


class RawRepresentation(RepresentationModule):
    """Identity control arm with the same encode/inverse interface as spectral modules."""

    name = "raw"

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        return EncodedRepresentation(
            values=inputs,
            representation=self.name,
            original_shape=tuple(inputs.shape),
            layout="bchw",
            metadata={"invertible": True},
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        if tuple(encoded.values.shape) != encoded.original_shape:
            raise RepresentationError(
                "raw coefficients must retain shape %r, got %r"
                % (encoded.original_shape, tuple(encoded.values.shape))
            )
        return encoded.values


class FFTRepresentation(RepresentationModule):
    """Real 2D FFT with real and imaginary parts packed along the channel axis.

    Magnitude/phase is deliberately not the training layout: phase has a branch cut and is
    undefined at zero magnitude.  Real/imaginary packing is continuous, differentiable and
    lets ordinary real-valued convolutional models consume the coefficients.
    """

    name = "fft"

    def __init__(self, norm: str = "backward") -> None:
        super().__init__()
        if norm not in ("backward", "forward", "ortho"):
            raise RepresentationError(
                "FFT norm must be 'backward', 'forward' or 'ortho', got %r" % norm
            )
        self.norm = norm

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        coefficients = torch.fft.rfft2(inputs, dim=(-2, -1), norm=self.norm)
        values = torch.cat((coefficients.real, coefficients.imag), dim=1)
        return EncodedRepresentation(
            values=values,
            representation=self.name,
            original_shape=tuple(inputs.shape),
            layout="b_2c_h_rfftwidth__real_then_imag",
            metadata={
                "channel_count": inputs.shape[1],
                "complex_packing": "real_channels_then_imag_channels",
                "norm": self.norm,
                "invertible": True,
            },
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        batch, channels, height, width = encoded.original_shape
        expected = (batch, 2 * channels, height, width // 2 + 1)
        if tuple(encoded.values.shape) != expected:
            raise RepresentationError(
                "FFT coefficients must have shape %r for original shape %r, got %r"
                % (expected, encoded.original_shape, tuple(encoded.values.shape))
            )
        if encoded.metadata.get("norm") != self.norm:
            raise RepresentationError(
                "FFT norm mismatch: module uses %r but coefficients record %r"
                % (self.norm, encoded.metadata.get("norm"))
            )
        real, imag = torch.split(encoded.values, channels, dim=1)
        coefficients = torch.complex(real, imag)
        return torch.fft.irfft2(
            coefficients, s=(height, width), dim=(-2, -1), norm=self.norm
        )

    def extra_repr(self) -> str:
        return "norm=%r" % self.norm


class DCTRepresentation(RepresentationModule):
    """Orthonormal batched DCT-II with cached DCT-III synthesis matrices.

    Spatial shape is fixed at construction because regional forecasting models normally have a
    fixed grid and because registered buffers then follow normal ``module.to(...)`` device and
    dtype semantics.  No cosine matrix is rebuilt during ``forward`` or ``inverse``.
    """

    name = "dct"

    def __init__(self, spatial_shape: Tuple[int, int], dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if len(spatial_shape) != 2 or any(int(size) < 1 for size in spatial_shape):
            raise RepresentationError(
                "DCT spatial_shape must be two positive integers, got %r"
                % (spatial_shape,)
            )
        if dtype not in self.supported_dtypes:
            raise RepresentationError("DCT matrix dtype must be float32 or float64")
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.register_buffer(
            "matrix_y", get_dct_matrix(self.spatial_shape[0], torch.device("cpu"), dtype)
        )
        self.register_buffer(
            "matrix_x", get_dct_matrix(self.spatial_shape[1], torch.device("cpu"), dtype)
        )

    def _apply(self, fn):
        """Move buffers, then regenerate them at the destination precision.

        PyTorch's default ``module.double()`` would only cast float32 cosine values to float64;
        the rounded digits would already be lost.  Rebuilding here makes dtype/device migration
        an explicit one-off cache event while keeping every training call allocation-free.
        """
        super()._apply(fn)
        device, dtype = self.matrix_y.device, self.matrix_y.dtype
        self.matrix_y = get_dct_matrix(self.spatial_shape[0], device, dtype)
        self.matrix_x = get_dct_matrix(self.spatial_shape[1], device, dtype)
        return self

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        super()._validate_inputs(inputs)
        if tuple(inputs.shape[-2:]) != self.spatial_shape:
            raise RepresentationError(
                "DCT module is configured for spatial shape %r, got %r"
                % (self.spatial_shape, tuple(inputs.shape[-2:]))
            )
        if inputs.device != self.matrix_y.device or inputs.dtype != self.matrix_y.dtype:
            raise RepresentationError(
                "DCT input is on %s/%s but cached matrices are on %s/%s; move the module "
                "with module.to(device=input.device, dtype=input.dtype)"
                % (inputs.device, inputs.dtype, self.matrix_y.device, self.matrix_y.dtype)
            )

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        values = torch.matmul(torch.matmul(self.matrix_y, inputs), self.matrix_x.T)
        return EncodedRepresentation(
            values=values,
            representation=self.name,
            original_shape=tuple(inputs.shape),
            layout="bchw_dct2_ortho",
            metadata={
                "spatial_shape": self.spatial_shape,
                "normalisation": "ortho",
                "invertible": True,
            },
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        if tuple(encoded.values.shape) != encoded.original_shape:
            raise RepresentationError(
                "DCT coefficients must retain shape %r, got %r"
                % (encoded.original_shape, tuple(encoded.values.shape))
            )
        if tuple(encoded.original_shape[-2:]) != self.spatial_shape:
            raise RepresentationError(
                "DCT coefficients record spatial shape %r but module is configured for %r"
                % (tuple(encoded.original_shape[-2:]), self.spatial_shape)
            )
        if (encoded.values.device != self.matrix_y.device
                or encoded.values.dtype != self.matrix_y.dtype):
            raise RepresentationError(
                "DCT coefficients and cached matrices must share device and dtype"
            )
        return torch.matmul(
            torch.matmul(self.matrix_y.T, encoded.values), self.matrix_x
        )

    def extra_repr(self) -> str:
        return "spatial_shape=%r, dtype=%s" % (self.spatial_shape, self.matrix_y.dtype)


def _periodic_analyse_1d(
    values: torch.Tensor, lowpass: torch.Tensor, highpass: torch.Tensor, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyWavelets-compatible periodisation then even-phase decimation."""
    anchor = lowpass.numel() // 2
    low = sum(
        lowpass[tap] * torch.roll(values, shifts=tap - anchor, dims=dim)
        for tap in range(lowpass.numel())
    )
    high = sum(
        highpass[tap] * torch.roll(values, shifts=tap - anchor, dims=dim)
        for tap in range(highpass.numel())
    )
    selector = [slice(None)] * values.ndim
    selector[dim] = slice(0, None, 2)
    return low[tuple(selector)], high[tuple(selector)]


def _periodic_synthesise_1d(
    low: torch.Tensor,
    high: torch.Tensor,
    lowpass: torch.Tensor,
    highpass: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Adjoint of :func:`_periodic_analyse_1d`; exact for an orthonormal filter pair."""
    shape = list(low.shape)
    shape[dim] *= 2
    low_up = torch.zeros(shape, device=low.device, dtype=low.dtype)
    high_up = torch.zeros(shape, device=high.device, dtype=high.dtype)
    selector = [slice(None)] * low.ndim
    selector[dim] = slice(0, None, 2)
    low_up[tuple(selector)] = low
    high_up[tuple(selector)] = high
    anchor = lowpass.numel() // 2
    return sum(
        lowpass[tap] * torch.roll(low_up, shifts=anchor - tap, dims=dim)
        + highpass[tap] * torch.roll(high_up, shifts=anchor - tap, dims=dim)
        for tap in range(lowpass.numel())
    )


class DecimatedWaveletRepresentation(RepresentationModule):
    """Multilevel periodic orthonormal DWT packed into a Mallat coefficient plane.

    Every level replaces its top-left approximation quadrant recursively, so the packed tensor
    remains model-friendly ``(B,C,H_pad,W_pad)`` and contains each coefficient exactly once.
    The phase convention is explicit and the inverse is the exact adjoint. Inputs that are not
    divisible by ``2**levels`` are periodically extended only when ``pad_to_multiple=True``;
    the extension and coefficient-count change travel in metadata.
    """

    def __init__(
        self,
        wavelet: str,
        levels: int = 1,
        pad_to_multiple: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        wavelet = wavelet.strip().lower()
        if wavelet not in ("haar", "db2"):
            raise RepresentationError("training DWT wavelet must be 'haar' or 'db2'")
        if int(levels) < 1:
            raise RepresentationError("DWT levels must be >= 1")
        if dtype not in self.supported_dtypes:
            raise RepresentationError("DWT filter dtype must be float32 or float64")
        self.wavelet = wavelet
        self.name = wavelet
        self.levels = int(levels)
        self.pad_to_multiple = bool(pad_to_multiple)
        lowpass, highpass = get_wavelet_filters(wavelet, torch.device("cpu"), dtype)
        self.register_buffer("lowpass", lowpass)
        self.register_buffer("highpass", highpass)

    def _apply(self, fn):
        super()._apply(fn)
        device, dtype = self.lowpass.device, self.lowpass.dtype
        self.lowpass, self.highpass = get_wavelet_filters(self.wavelet, device, dtype)
        return self

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        super()._validate_inputs(inputs)
        if inputs.device != self.lowpass.device or inputs.dtype != self.lowpass.dtype:
            raise RepresentationError(
                "DWT input is on %s/%s but cached filters are on %s/%s; move the module "
                "with module.to(device=input.device, dtype=input.dtype)"
                % (inputs.device, inputs.dtype, self.lowpass.device, self.lowpass.dtype)
            )
        factor = 2 ** self.levels
        if min(inputs.shape[-2:]) < factor:
            raise RepresentationError(
                "%d DWT levels require each spatial axis to contain at least %d pixels, got %r"
                % (self.levels, factor, tuple(inputs.shape[-2:]))
            )

    def _analyse_level(self, values: torch.Tensor):
        low_x, high_x = _periodic_analyse_1d(
            values, self.lowpass, self.highpass, dim=-1
        )
        ll, hl = _periodic_analyse_1d(low_x, self.lowpass, self.highpass, dim=-2)
        lh, hh = _periodic_analyse_1d(high_x, self.lowpass, self.highpass, dim=-2)
        return ll, lh, hl, hh

    def _synthesise_level(self, ll, lh, hl, hh):
        low_x = _periodic_synthesise_1d(
            ll, hl, self.lowpass, self.highpass, dim=-2
        )
        high_x = _periodic_synthesise_1d(
            lh, hh, self.lowpass, self.highpass, dim=-2
        )
        return _periodic_synthesise_1d(
            low_x, high_x, self.lowpass, self.highpass, dim=-1
        )

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        factor = 2 ** self.levels
        height, width = inputs.shape[-2:]
        pad_h, pad_w = (-height) % factor, (-width) % factor
        if (pad_h or pad_w) and not self.pad_to_multiple:
            raise RepresentationError(
                "spatial shape %r is not divisible by 2**levels=%d; enable explicit "
                "periodic extension or choose fewer levels"
                % ((height, width), factor)
            )
        padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="circular") \
            if (pad_h or pad_w) else inputs

        details = []
        current = padded
        for _level in range(1, self.levels + 1):
            current, lh, hl, hh = self._analyse_level(current)
            details.append((lh, hl, hh))

        packed = current
        for lh, hl, hh in reversed(details):
            packed = torch.cat(
                (torch.cat((packed, lh), dim=-1), torch.cat((hl, hh), dim=-1)), dim=-2
            )

        margins = tuple(
            valid_interior_halfwidth(self.wavelet, level)
            for level in range(1, self.levels + 1)
        )
        return EncodedRepresentation(
            values=packed,
            representation=self.name,
            original_shape=tuple(inputs.shape),
            layout="mallat_quadrants_LL_LH_HL_HH",
            metadata={
                "wavelet": self.wavelet,
                "levels": self.levels,
                "boundary": "periodic",
                "decimation_phase": "pywavelets_periodization_even_length_anchor",
                "filter_length": int(self.lowpass.numel()),
                "padded_spatial_shape": tuple(padded.shape[-2:]),
                "padding_bottom_right": (pad_h, pad_w),
                "valid_interior_halfwidth_by_level": margins,
                "coefficient_count_ratio": float(packed.numel() / inputs.numel()),
                "invertible": True,
            },
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        if encoded.metadata.get("wavelet") != self.wavelet:
            raise RepresentationError("DWT wavelet metadata does not match this module")
        if encoded.metadata.get("levels") != self.levels:
            raise RepresentationError("DWT level metadata does not match this module")
        padded_shape = tuple(encoded.metadata.get("padded_spatial_shape", ()))
        expected = encoded.original_shape[:2] + padded_shape
        if tuple(encoded.values.shape) != expected:
            raise RepresentationError(
                "packed DWT coefficients must have shape %r, got %r"
                % (expected, tuple(encoded.values.shape))
            )
        if (encoded.values.device != self.lowpass.device
                or encoded.values.dtype != self.lowpass.dtype):
            raise RepresentationError("DWT coefficients and cached filters must share device/dtype")

        current_region = encoded.values
        details = []
        for _level in range(1, self.levels + 1):
            half_h, half_w = current_region.shape[-2] // 2, current_region.shape[-1] // 2
            details.append((
                current_region[..., :half_h, half_w:],
                current_region[..., half_h:, :half_w],
                current_region[..., half_h:, half_w:],
            ))
            current_region = current_region[..., :half_h, :half_w]

        current = current_region
        for lh, hl, hh in reversed(details):
            current = self._synthesise_level(current, lh, hl, hh)
        height, width = encoded.original_shape[-2:]
        return current[..., :height, :width]

    def extra_repr(self) -> str:
        return "wavelet=%r, levels=%d, pad_to_multiple=%r, dtype=%s" % (
            self.wavelet, self.levels, self.pad_to_multiple, self.lowpass.dtype
        )


class HaarRepresentation(DecimatedWaveletRepresentation):
    name = "haar"

    def __init__(self, levels: int = 1, pad_to_multiple: bool = True,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__("haar", levels, pad_to_multiple, dtype)


class DB2Representation(DecimatedWaveletRepresentation):
    name = "db2"

    def __init__(self, levels: int = 1, pad_to_multiple: bool = True,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__("db2", levels, pad_to_multiple, dtype)


REPRESENTATIONS: Dict[str, Type[RepresentationModule]] = {
    "raw": RawRepresentation,
    "fft": FFTRepresentation,
    "dct": DCTRepresentation,
    "haar": HaarRepresentation,
    "db2": DB2Representation,
}


def make_representation(name: str, **kwargs: Any) -> RepresentationModule:
    """Construct an accepted training representation by stable public name."""
    key = name.strip().lower()
    try:
        representation = REPRESENTATIONS[key]
    except KeyError as exc:
        raise RepresentationError(
            "unknown training representation %r; available: %s"
            % (name, ", ".join(sorted(REPRESENTATIONS)))
        ) from exc
    return representation(**kwargs)
