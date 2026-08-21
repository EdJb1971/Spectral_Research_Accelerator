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
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from src.transform_engine.stationary import (
    circular_filter_1d,
    dilate_filter,
    filter_support,
    get_filters as get_wavelet_filters,
    valid_interior_halfwidth,
)
from src.transform_engine.transforms import get_dct_matrix
from src.transform_engine import dtcwt as dtcwt_engine
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec


class RepresentationError(ValueError):
    """A tensor or encoded batch violates a representation's declared contract."""


def _full_precision_convolution(device: torch.device):
    """Local cuDNN policy: never let ambient TF32 change a transform's definition.

    `enable_determinism` selects a different cuDNN algorithm on the accepted RTX. With ambient
    TF32 enabled that algorithm moved an SWT round trip from 2.4e-7 to 4.5e-4. The transform
    therefore scopes full float32 convolution precision around its own kernels and restores the
    caller's global policy afterwards.
    """
    if device.type == "cuda" and torch.backends.cudnn.is_available():
        return torch.backends.cudnn.flags(allow_tf32=False)
    return nullcontext()


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
        implementation: str = "conv2d",
    ) -> None:
        super().__init__()
        wavelet = wavelet.strip().lower()
        if wavelet not in ("haar", "db2"):
            raise RepresentationError("training DWT wavelet must be 'haar' or 'db2'")
        if int(levels) < 1:
            raise RepresentationError("DWT levels must be >= 1")
        if dtype not in self.supported_dtypes:
            raise RepresentationError("DWT filter dtype must be float32 or float64")
        implementation = implementation.strip().lower()
        if implementation not in ("conv2d", "reference"):
            raise RepresentationError(
                "DWT implementation must be 'conv2d' or 'reference', got %r"
                % implementation
            )
        self.wavelet = wavelet
        self.name = wavelet
        self.levels = int(levels)
        self.pad_to_multiple = bool(pad_to_multiple)
        self.implementation = implementation
        lowpass, highpass = get_wavelet_filters(wavelet, torch.device("cpu"), dtype)
        self.register_buffer("lowpass", lowpass)
        self.register_buffer("highpass", highpass)
        self.register_buffer("analysis_bank", self._make_analysis_bank(lowpass, highpass))

    @staticmethod
    def _make_analysis_bank(
        lowpass: torch.Tensor, highpass: torch.Tensor
    ) -> torch.Tensor:
        """Four cross-correlation kernels in packed LL/LH/HL/HH order."""
        kernels = torch.stack((
            torch.outer(lowpass, lowpass),
            torch.outer(lowpass, highpass),
            torch.outer(highpass, lowpass),
            torch.outer(highpass, highpass),
        ))
        # The reference equations are convolutions; torch conv2d is cross-correlation.
        return kernels.flip((-2, -1)).unsqueeze(1)

    def _apply(self, fn):
        super()._apply(fn)
        device, dtype = self.lowpass.device, self.lowpass.dtype
        self.lowpass, self.highpass = get_wavelet_filters(self.wavelet, device, dtype)
        self.analysis_bank = self._make_analysis_bank(self.lowpass, self.highpass)
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

    def _analyse_level_reference(self, values: torch.Tensor):
        low_x, high_x = _periodic_analyse_1d(
            values, self.lowpass, self.highpass, dim=-1
        )
        ll, hl = _periodic_analyse_1d(low_x, self.lowpass, self.highpass, dim=-2)
        lh, hh = _periodic_analyse_1d(high_x, self.lowpass, self.highpass, dim=-2)
        return ll, lh, hl, hh

    def _synthesise_level_reference(self, ll, lh, hl, hh):
        low_x = _periodic_synthesise_1d(
            ll, hl, self.lowpass, self.highpass, dim=-2
        )
        high_x = _periodic_synthesise_1d(
            lh, hh, self.lowpass, self.highpass, dim=-2
        )
        return _periodic_synthesise_1d(
            low_x, high_x, self.lowpass, self.highpass, dim=-1
        )

    @property
    def _periodic_pad(self) -> int:
        # For every accepted even-length orthonormal filter, PyWavelets periodization is a
        # symmetric circular pad of L/2-1 followed by correlation with the reversed kernel.
        return int(self.lowpass.numel() // 2 - 1)

    def _analyse_level_conv2d(self, values: torch.Tensor):
        """Evaluate all four separable bands in one strided convolution call."""
        batch, channels, height, width = values.shape
        pad = self._periodic_pad
        padded = F.pad(values, (pad, pad, pad, pad), mode="circular") if pad else values
        with _full_precision_convolution(values.device):
            bands = F.conv2d(
                padded.reshape(batch * channels, 1, *padded.shape[-2:]),
                self.analysis_bank,
                stride=2,
            ).reshape(batch, channels, 4, height // 2, width // 2)
        return tuple(bands[:, :, band] for band in range(4))

    @staticmethod
    def _fold_periodic_adjoint(
        extended: torch.Tensor, pad: int, height: int, width: int
    ) -> torch.Tensor:
        """Adjoint of symmetric circular padding, including corner contributions."""
        if not pad:
            return extended
        folded_width = (
            extended[..., :, pad:pad + width]
            + F.pad(extended[..., :, :pad], (width - pad, 0))
            + F.pad(extended[..., :, pad + width:], (0, width - pad))
        )
        return (
            folded_width[..., pad:pad + height, :]
            + F.pad(folded_width[..., :pad, :], (0, 0, height - pad, 0))
            + F.pad(folded_width[..., pad + height:, :], (0, 0, 0, height - pad))
        )

    def _synthesise_level_conv2d(self, ll, lh, hl, hh):
        """Exact adjoint synthesis using one transposed convolution and periodic fold."""
        batch, channels, height, width = ll.shape
        bands = torch.stack((ll, lh, hl, hh), dim=2)
        with _full_precision_convolution(ll.device):
            extended = F.conv_transpose2d(
                bands.reshape(batch * channels, 4, height, width),
                self.analysis_bank,
                stride=2,
            )
        reconstructed = self._fold_periodic_adjoint(
            extended, self._periodic_pad, 2 * height, 2 * width
        )
        return reconstructed.reshape(batch, channels, 2 * height, 2 * width)

    def _analyse_level(self, values: torch.Tensor):
        if self.implementation == "conv2d":
            return self._analyse_level_conv2d(values)
        return self._analyse_level_reference(values)

    def _synthesise_level(self, ll, lh, hl, hh):
        if self.implementation == "conv2d":
            return self._synthesise_level_conv2d(ll, lh, hl, hh)
        return self._synthesise_level_reference(ll, lh, hl, hh)

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
                "implementation": self.implementation,
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
        if encoded.metadata.get("implementation") != self.implementation:
            raise RepresentationError("DWT implementation metadata does not match this module")
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
        return "wavelet=%r, levels=%d, pad_to_multiple=%r, implementation=%r, dtype=%s" % (
            self.wavelet, self.levels, self.pad_to_multiple, self.implementation,
            self.lowpass.dtype
        )


class HaarRepresentation(DecimatedWaveletRepresentation):
    name = "haar"

    def __init__(self, levels: int = 1, pad_to_multiple: bool = True,
                 dtype: torch.dtype = torch.float32,
                 implementation: str = "conv2d") -> None:
        super().__init__("haar", levels, pad_to_multiple, dtype, implementation)


class DB2Representation(DecimatedWaveletRepresentation):
    name = "db2"

    def __init__(self, levels: int = 1, pad_to_multiple: bool = True,
                 dtype: torch.dtype = torch.float32,
                 implementation: str = "conv2d") -> None:
        super().__init__("db2", levels, pad_to_multiple, dtype, implementation)


class SWTRepresentation(RepresentationModule):
    """Undecimated, exactly shift-equivariant periodic wavelet representation.

    The packed channel order for each input channel is final LL followed by LH/HL/HH for
    levels 1..L. Every band retains the parent ``(H,W)`` grid, so the coefficient expansion is
    exactly ``1 + 3*levels``. This is useful for tracking but materially more expensive than a
    decimated pyramid; the expansion and boundary-valid interior travel with every batch.
    """

    name = "swt"

    def __init__(
        self,
        wavelet: str = "db2",
        levels: int = 1,
        dtype: torch.dtype = torch.float32,
        implementation: str = "auto",
    ) -> None:
        super().__init__()
        wavelet = wavelet.strip().lower()
        if wavelet not in ("haar", "db2", "db3"):
            raise RepresentationError("training SWT wavelet must be haar, db2 or db3")
        if int(levels) < 1:
            raise RepresentationError("SWT levels must be >= 1")
        if dtype not in self.supported_dtypes:
            raise RepresentationError("SWT filter dtype must be float32 or float64")
        implementation = implementation.strip().lower()
        if implementation not in ("auto", "conv2d", "fft_reference"):
            raise RepresentationError(
                "SWT implementation must be 'auto', 'conv2d' or 'fft_reference', got %r"
                % implementation
            )
        self.wavelet = wavelet
        self.levels = int(levels)
        self.implementation = implementation
        lowpass, highpass = get_wavelet_filters(wavelet, torch.device("cpu"), dtype)
        self.register_buffer("lowpass", lowpass)
        self.register_buffer("highpass", highpass)
        for level in range(1, self.levels + 1):
            self.register_buffer(
                "analysis_bank_%d" % level,
                self._make_level_bank(lowpass, highpass, level),
            )

    @staticmethod
    def _make_level_bank(
        lowpass: torch.Tensor, highpass: torch.Tensor, level: int
    ) -> torch.Tensor:
        dilation = 2 ** (level - 1)
        low = dilate_filter(lowpass, dilation)
        high = dilate_filter(highpass, dilation)
        kernels = torch.stack((
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        ))
        return kernels.flip((-2, -1)).unsqueeze(1)

    def _bank(self, level: int) -> torch.Tensor:
        return getattr(self, "analysis_bank_%d" % level)

    def _apply(self, fn):
        super()._apply(fn)
        device, dtype = self.lowpass.device, self.lowpass.dtype
        self.lowpass, self.highpass = get_wavelet_filters(self.wavelet, device, dtype)
        for level in range(1, self.levels + 1):
            setattr(
                self,
                "analysis_bank_%d" % level,
                self._make_level_bank(self.lowpass, self.highpass, level),
            )
        return self

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        super()._validate_inputs(inputs)
        if inputs.device != self.lowpass.device or inputs.dtype != self.lowpass.dtype:
            raise RepresentationError(
                "SWT input is on %s/%s but cached filters are on %s/%s; move the module "
                "with module.to(device=input.device, dtype=input.dtype)"
                % (inputs.device, inputs.dtype, self.lowpass.device, self.lowpass.dtype)
            )
        support = filter_support(self.wavelet, self.levels)
        if support > min(inputs.shape[-2:]):
            raise RepresentationError(
                "level %d %s SWT has accumulated support %d px, exceeding spatial shape %r; "
                "use fewer levels or a larger regional field (R13)"
                % (self.levels, self.wavelet, support, tuple(inputs.shape[-2:]))
            )

    @staticmethod
    def _fold_left_top_periodic_adjoint(
        extended: torch.Tensor, pad: int, height: int, width: int
    ) -> torch.Tensor:
        if not pad:
            return extended
        folded_width = (
            extended[..., :, pad:pad + width]
            + F.pad(extended[..., :, :pad], (width - pad, 0))
        )
        return (
            folded_width[..., pad:pad + height, :]
            + F.pad(folded_width[..., :pad, :], (0, 0, height - pad, 0))
        )

    def _analyse_level_conv2d(self, values: torch.Tensor, level: int):
        batch, channels, height, width = values.shape
        bank = self._bank(level)
        pad = bank.shape[-1] - 1
        padded = F.pad(values, (pad, 0, pad, 0), mode="circular") if pad else values
        with _full_precision_convolution(values.device):
            bands = F.conv2d(
                padded.reshape(batch * channels, 1, *padded.shape[-2:]), bank
            ).reshape(batch, channels, 4, height, width)
        return tuple(bands[:, :, band] for band in range(4))

    def _synthesise_level_conv2d(self, ll, lh, hl, hh, level: int):
        batch, channels, height, width = ll.shape
        bank = self._bank(level)
        with _full_precision_convolution(ll.device):
            extended = F.conv_transpose2d(
                torch.stack((ll, lh, hl, hh), dim=2).reshape(
                    batch * channels, 4, height, width
                ),
                bank,
            )
        reconstructed = self._fold_left_top_periodic_adjoint(
            extended, bank.shape[-1] - 1, height, width
        ) / 4.0
        return reconstructed.reshape(batch, channels, height, width)

    def _analyse_level_fft(self, values: torch.Tensor, level: int):
        dilation = 2 ** (level - 1)
        low = dilate_filter(self.lowpass, dilation)
        high = dilate_filter(self.highpass, dilation)
        low_rows = circular_filter_1d(values, low, -2)
        high_rows = circular_filter_1d(values, high, -2)
        return (
            circular_filter_1d(low_rows, low, -1),
            circular_filter_1d(low_rows, high, -1),
            circular_filter_1d(high_rows, low, -1),
            circular_filter_1d(high_rows, high, -1),
        )

    def _synthesise_level_fft(self, ll, lh, hl, hh, level: int):
        dilation = 2 ** (level - 1)
        low = dilate_filter(self.lowpass, dilation)
        high = dilate_filter(self.highpass, dilation)

        def adjoint(values, row_filter, column_filter):
            return circular_filter_1d(
                circular_filter_1d(values, column_filter, -1, adjoint=True),
                row_filter, -2, adjoint=True,
            )

        return (
            adjoint(ll, low, low)
            + adjoint(lh, low, high)
            + adjoint(hl, high, low)
            + adjoint(hh, high, high)
        ) / 4.0

    def _analyse_level(self, values: torch.Tensor, level: int):
        if self._resolved_implementation(values.device) == "conv2d":
            return self._analyse_level_conv2d(values, level)
        return self._analyse_level_fft(values, level)

    def _synthesise_level(self, ll, lh, hl, hh, level: int):
        if self._resolved_implementation(ll.device) == "conv2d":
            return self._synthesise_level_conv2d(ll, lh, hl, hh, level)
        return self._synthesise_level_fft(ll, lh, hl, hh, level)

    def _resolved_implementation(self, device: torch.device) -> str:
        if self.implementation != "auto":
            return self.implementation
        # Measured on the accepted workload: FFT is 3.5x faster on CPU, while convolution is
        # 2.1x faster and uses half the peak allocation on CUDA. ROCm also presents as cuda.
        return "conv2d" if device.type in ("cuda", "mps") else "fft_reference"

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        current = inputs
        details = []
        for level in range(1, self.levels + 1):
            current, lh, hl, hh = self._analyse_level(current, level)
            details.extend((lh, hl, hh))
        packed = torch.stack((current, *details), dim=2).reshape(
            inputs.shape[0], inputs.shape[1] * (1 + 3 * self.levels), *inputs.shape[-2:]
        )
        margins = tuple(
            valid_interior_halfwidth(self.wavelet, level)
            for level in range(1, self.levels + 1)
        )
        interiors = tuple(
            (max(0, inputs.shape[-2] - 2 * margin),
             max(0, inputs.shape[-1] - 2 * margin))
            for margin in margins
        )
        labels = ("LL_final",) + tuple(
            "%s_level_%d" % (band, level)
            for level in range(1, self.levels + 1)
            for band in ("LH", "HL", "HH")
        )
        return EncodedRepresentation(
            values=packed,
            representation=self.name,
            original_shape=tuple(inputs.shape),
            layout="per_input_channel__LL_then_levelwise_LH_HL_HH",
            metadata={
                "wavelet": self.wavelet,
                "levels": self.levels,
                "implementation": self._resolved_implementation(inputs.device),
                "implementation_policy": self.implementation,
                "boundary": "periodic",
                "channel_labels_per_input_channel": labels,
                "bands_per_input_channel": 1 + 3 * self.levels,
                "coefficient_count_ratio": float(1 + 3 * self.levels),
                "accumulated_support_by_level": tuple(
                    filter_support(self.wavelet, level)
                    for level in range(1, self.levels + 1)
                ),
                "valid_interior_halfwidth_by_level": margins,
                "valid_interior_shape_by_level": interiors,
                "level_amplitude_gain_by_level": tuple(
                    2 ** level for level in range(1, self.levels + 1)
                ),
                "energy_comparison":
                    "divide each level's raw detail energy by 4**level before comparing scales",
                "shift_behavior": "exact translation equivariance under periodic boundaries",
                "directionality":
                    "three separable LH/HL/HH bands; not six signed orientations",
                "regional_boundary_warning":
                    "periodic wrap is artificial on a crop; exclude each level's reported margin",
                "invertible": True,
            },
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        for key, expected in (
            ("wavelet", self.wavelet),
            ("levels", self.levels),
            ("implementation", self._resolved_implementation(encoded.values.device)),
            ("implementation_policy", self.implementation),
            ("boundary", "periodic"),
        ):
            if encoded.metadata.get(key) != expected:
                raise RepresentationError("SWT %s metadata does not match this module" % key)
        batch, channels, height, width = encoded.original_shape
        expected_shape = (batch, channels * (1 + 3 * self.levels), height, width)
        if tuple(encoded.values.shape) != expected_shape:
            raise RepresentationError(
                "packed SWT coefficients must have shape %r, got %r"
                % (expected_shape, tuple(encoded.values.shape))
            )
        if (encoded.values.device != self.lowpass.device
                or encoded.values.dtype != self.lowpass.dtype):
            raise RepresentationError("SWT coefficients and cached filters must share device/dtype")
        unpacked = encoded.values.reshape(
            batch, channels, 1 + 3 * self.levels, height, width
        )
        current = unpacked[:, :, 0]
        for level in range(self.levels, 0, -1):
            offset = 1 + 3 * (level - 1)
            current = self._synthesise_level(
                current,
                unpacked[:, :, offset],
                unpacked[:, :, offset + 1],
                unpacked[:, :, offset + 2],
                level,
            )
        return current

    def extra_repr(self) -> str:
        return "wavelet=%r, levels=%d, implementation=%r, dtype=%s" % (
            self.wavelet, self.levels, self.implementation, self.lowpass.dtype
        )


class DTCWTRepresentation(RepresentationModule):
    """Batched differentiable Kingsbury DTCWT with an exact real-valued atlas.

    Each input channel becomes four real atlas planes. Plane 0 recursively nests the
    low-pass and levels >=2; planes 1-3 contain the twelve real/imaginary components of
    level 1. No coefficient is interpolated or discarded, so arbitrary model-produced
    atlases can be inverted as well as transform-produced ones.
    """

    name = "dtcwt"
    _FILTER_KEYS = (
        "h0o", "h1o", "g0o", "g1o",
        "h0a", "h0b", "h1a", "h1b", "g0a", "g0b", "g1a", "g1b",
    )

    def __init__(self, levels: int = 3, level1: str = "near_sym_b",
                 qshift: str = "qshift_b", pad_to_multiple: bool = True,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if int(levels) < 1:
            raise RepresentationError("DTCWT levels must be >= 1")
        if level1 not in dtcwt_engine.LEVEL1_FILTERS:
            raise RepresentationError("unknown DTCWT level-1 filter %r" % level1)
        if qshift not in dtcwt_engine.QSHIFT_FILTERS:
            raise RepresentationError("unknown DTCWT q-shift filter %r" % qshift)
        if dtype not in self.supported_dtypes:
            raise RepresentationError("DTCWT filter dtype must be float32 or float64")
        self.levels = int(levels)
        self.level1 = level1
        self.qshift = qshift
        self.pad_to_multiple = bool(pad_to_multiple)
        for key in self._FILTER_KEYS:
            table = (dtcwt_engine.LEVEL1_FILTERS if key.endswith("o")
                     else dtcwt_engine.QSHIFT_FILTERS)
            family = level1 if key.endswith("o") else qshift
            self.register_buffer(key, torch.tensor(table[family][key], dtype=dtype))

    def _filters(self) -> Dict[str, torch.Tensor]:
        return {key: getattr(self, key) for key in self._FILTER_KEYS}

    def _apply(self, fn):
        super()._apply(fn)
        # Recreate from the vendored decimal coefficients after dtype migration; converting
        # a float32 buffer to float64 cannot recover the precision that was rounded away.
        device, dtype = self.h0o.device, self.h0o.dtype
        for key in self._FILTER_KEYS:
            table = (dtcwt_engine.LEVEL1_FILTERS if key.endswith("o")
                     else dtcwt_engine.QSHIFT_FILTERS)
            family = self.level1 if key.endswith("o") else self.qshift
            setattr(self, key, torch.tensor(table[family][key], device=device, dtype=dtype))
        return self

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        super()._validate_inputs(inputs)
        if inputs.device != self.h0o.device or inputs.dtype != self.h0o.dtype:
            raise RepresentationError(
                "DTCWT input and cached filters must share device/dtype; move the module "
                "with module.to(device=input.device, dtype=input.dtype)"
            )
        margin = dtcwt_engine.valid_interior_halfwidth(
            self.levels, self.level1, self.qshift
        )
        if 2 * margin >= min(inputs.shape[-2:]):
            raise RepresentationError(
                "level %d DTCWT has a %d px edge margin per side, leaving no valid "
                "interior on spatial shape %r; use fewer levels or a larger crop"
                % (self.levels, margin, tuple(inputs.shape[-2:]))
            )

    @staticmethod
    def _tile_four(values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] != 4:
            raise RepresentationError("an atlas tile requires exactly four components")
        return torch.cat((torch.cat((values[:, 0], values[:, 1]), -1),
                          torch.cat((values[:, 2], values[:, 3]), -1)), -2)

    @staticmethod
    def _untile_four(tile: torch.Tensor) -> torch.Tensor:
        h, w = tile.shape[-2:]
        if h % 2 or w % 2:
            raise RepresentationError("DTCWT atlas quadrant has an odd spatial dimension")
        hh, hw = h // 2, w // 2
        return torch.stack((tile[:, :hh, :hw], tile[:, :hh, hw:],
                            tile[:, hh:, :hw], tile[:, hh:, hw:]), 1)

    @staticmethod
    def _components(band: torch.Tensor) -> torch.Tensor:
        # orientation-major order: Re0, Im0, Re1, Im1, ... Re5, Im5.
        return torch.view_as_real(band).permute(0, 3, 4, 1, 2).reshape(
            band.shape[0], 12, band.shape[1], band.shape[2]
        )

    @staticmethod
    def _complex_band(components: torch.Tensor) -> torch.Tensor:
        n, _, h, w = components.shape
        pairs = components.reshape(n, 6, 2, h, w).permute(0, 3, 4, 1, 2).contiguous()
        return torch.view_as_complex(pairs)

    def _pack(self, lowpass: torch.Tensor, bands: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        first = self._components(bands[0])
        level1_planes = tuple(self._tile_four(first[:, i:i + 4]) for i in (0, 4, 8))
        region = lowpass
        for band in reversed(bands[1:]):
            components = self._components(band)
            tiles = tuple(self._tile_four(components[:, i:i + 4]) for i in (0, 4, 8))
            region = torch.cat((torch.cat((region, tiles[0]), -1),
                                torch.cat((tiles[1], tiles[2]), -1)), -2)
        return torch.stack((region, *level1_planes), 1)

    def _unpack(self, atlas: torch.Tensor):
        first = torch.cat(tuple(self._untile_four(atlas[:, plane])
                                for plane in (1, 2, 3)), 1)
        bands = [self._complex_band(first)]
        region = atlas[:, 0]
        for _level in range(2, self.levels + 1):
            h, w = region.shape[-2:]
            if h % 2 or w % 2:
                raise RepresentationError("DTCWT recursive atlas geometry is inconsistent")
            hh, hw = h // 2, w // 2
            tiles = (region[:, :hh, hw:], region[:, hh:, :hw], region[:, hh:, hw:])
            components = torch.cat(tuple(self._untile_four(tile) for tile in tiles), 1)
            bands.append(self._complex_band(components))
            region = region[:, :hh, :hw]
        return region, tuple(bands)

    def encode(self, inputs: torch.Tensor) -> EncodedRepresentation:
        batch, channels, height, width = inputs.shape
        factor = 2 ** self.levels
        padded_h = ((height + factor - 1) // factor) * factor
        padded_w = ((width + factor - 1) // factor) * factor
        if (padded_h, padded_w) != (height, width) and not self.pad_to_multiple:
            raise RepresentationError(
                "DTCWT spatial shape must be divisible by %d when padding is disabled" % factor
            )
        padded = F.pad(inputs, (0, padded_w - width, 0, padded_h - height), mode="replicate")
        flat = padded.reshape(batch * channels, padded_h, padded_w)
        filters = self._filters()

        def analyse(value):
            coeffs = dtcwt_engine.apply_dtcwt2d(
                PhysicalField(value), levels=self.levels, level1=self.level1,
                qshift=self.qshift, dtype=value.dtype, _filters=filters,
            )
            return (coeffs["lowpass"], *tuple(coeffs["highpass"]))

        transformed = torch.vmap(analyse)(flat)
        atlas = self._pack(transformed[0], tuple(transformed[1:])).reshape(
            batch, channels * 4, padded_h, padded_w
        )
        margins = tuple(dtcwt_engine.valid_interior_halfwidth(
            level, self.level1, self.qshift) for level in range(1, self.levels + 1))
        native_margins = tuple(dtcwt_engine.native_halfwidth(
            level, self.level1, self.qshift) for level in range(1, self.levels + 1))
        native_shapes = tuple((padded_h // (2 ** level), padded_w // (2 ** level))
                              for level in range(1, self.levels + 1))
        native_interiors = tuple(
            (max(0, shape[0] - 2 * margin), max(0, shape[1] - 2 * margin))
            for shape, margin in zip(native_shapes, native_margins)
        )
        return EncodedRepresentation(
            values=atlas, representation=self.name, original_shape=tuple(inputs.shape),
            layout="per_input_channel__four_plane_recursive_complex_atlas",
            metadata={
                "levels": self.levels, "level1": self.level1, "qshift": self.qshift,
                "padding_bottom_right": (padded_h - height, padded_w - width),
                "padded_spatial_shape": (padded_h, padded_w),
                "atlas_planes_per_input_channel": 4,
                "component_order": "Re0,Im0,Re1,Im1,...,Re5,Im5",
                "coefficient_count_ratio": 4.0 * padded_h * padded_w / (height * width),
                "wavevector_orientations_deg": dtcwt_engine.SUBBAND_WAVEVECTOR_DEG,
                "feature_orientations_deg": dtcwt_engine.SUBBAND_FEATURE_DEG,
                "measured_level1_wavevector_deg": dtcwt_engine.MEASURED_LEVEL1_WAVEVECTOR_DEG,
                "measured_qshift_wavevector_deg": dtcwt_engine.MEASURED_QSHIFT_WAVEVECTOR_DEG,
                "native_shape_by_level": native_shapes,
                "valid_interior_halfwidth_parent_px_by_level": margins,
                "valid_interior_halfwidth_native_px_by_level": native_margins,
                "valid_interior_native_shape_by_level": native_interiors,
                "coarsest_scale_has_valid_interior": all(v > 0 for v in native_interiors[-1]),
                "orientation_convention": "feature angle = wavevector angle + 90 degrees (axial modulo 180)",
                "level1_orientation_warning": "level-1 measured centres deviate by up to 7.1 degrees; prefer level >=2 for orientation estimates",
                "atlas_warning": "atlas tile adjacency is packing geometry, not physical spatial adjacency",
                "display_contract": "show native complex magnitude with invalid edge inset marked; never interpolate levels into one heatmap",
                "shift_behavior": "near shift invariant, not exact",
                "boundary": "symmetric filter extension; optional bottom/right replicate dyadic padding",
                "invertible": True,
            },
        )

    def inverse(self, encoded: EncodedRepresentation) -> torch.Tensor:
        self._validate_encoded(encoded)
        for key, expected in (("levels", self.levels), ("level1", self.level1),
                              ("qshift", self.qshift)):
            if encoded.metadata.get(key) != expected:
                raise RepresentationError("DTCWT %s metadata does not match this module" % key)
        batch, channels, height, width = encoded.original_shape
        padded_h, padded_w = encoded.metadata["padded_spatial_shape"]
        expected = (batch, channels * 4, padded_h, padded_w)
        if tuple(encoded.values.shape) != expected:
            raise RepresentationError("packed DTCWT coefficients must have shape %r" % (expected,))
        atlas = encoded.values.reshape(batch * channels, 4, padded_h, padded_w)
        lowpass, bands = self._unpack(atlas)
        filters = self._filters()
        grid = GridSpec.pixel((padded_h, padded_w)).to_provenance()

        def synthesise(low, *high):
            coeffs = {
                "lowpass": low, "highpass": list(high), "levels": self.levels,
                "level1": self.level1, "qshift": self.qshift,
                "original_shape": (padded_h, padded_w), "grid": grid,
            }
            return dtcwt_engine.inverse_dtcwt2d(
                coeffs, dtype=low.dtype, _filters=filters
            ).data

        reconstructed = torch.vmap(synthesise)(lowpass, *bands).reshape(
            batch, channels, padded_h, padded_w
        )
        return reconstructed[:, :, :height, :width]

    def extra_repr(self) -> str:
        return "levels=%d, level1=%r, qshift=%r, dtype=%s" % (
            self.levels, self.level1, self.qshift, self.h0o.dtype
        )


REPRESENTATIONS: Dict[str, Type[RepresentationModule]] = {
    "raw": RawRepresentation,
    "fft": FFTRepresentation,
    "dct": DCTRepresentation,
    "haar": HaarRepresentation,
    "db2": DB2Representation,
    "swt": SWTRepresentation,
    "dtcwt": DTCWTRepresentation,
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


def training_representation_catalogue(
    levels: int = 3,
    wavelet: str = "db2",
    spatial_shape: Tuple[int, int] = (120, 80),
) -> Dict[str, Any]:
    """UI-facing capability and caveat record derived from the accepted training seam.

    This is deliberately separate from the analytical transform registry. A transform being
    available for one 2D field does not mean it is batched, differentiable, or cheap enough for
    a model's inner loop. The catalogue makes that distinction machine-readable.
    """
    if levels < 1:
        raise RepresentationError("catalogue levels must be >= 1")
    if wavelet not in ("haar", "db2", "db3"):
        raise RepresentationError("catalogue SWT wavelet must be haar, db2 or db3")
    height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
    if height < 1 or width < 1:
        raise RepresentationError("catalogue spatial shape must be positive")

    verified = ("CPU float32/float64", "NVIDIA CUDA float32 (RTX 5050 Laptop GPU)")
    not_run = ("NVIDIA CUDA float64 acceptance", "AMD ROCm hardware", "Apple MPS hardware",
               "mixed precision", "torch.compile")
    common = {
        "status": "accepted",
        "batched": True,
        "autograd": True,
        "exact_inverse": True,
        "verified": verified,
        "not_run": not_run,
    }
    entries = [
        {**common, "name": "raw", "label": "Raw control", "coefficient_ratio": "1x",
         "shift_behavior": "equivariant by identity", "directionality": "none",
         "boundary": "none", "scientific_role": "required no-transform baseline",
         "limitations": ("does not separate scale or orientation",)},
        {**common, "name": "fft", "label": "Real FFT", "coefficient_ratio": "~1x real-packed",
         "shift_behavior": "translation changes phase", "directionality": "global wavevector",
         "boundary": "periodic domain assumed",
         "scientific_role": "global spectral baseline",
         "limitations": ("not spatially localised", "regional edge discontinuities contaminate modes")},
        {**common, "name": "dct", "label": "DCT-II", "coefficient_ratio": "1x",
         "shift_behavior": "not shift invariant", "directionality": "separable axes",
         "boundary": "even reflection implied",
         "scientific_role": "real global-frequency baseline without periodic wrap",
         "limitations": ("not spatially localised", "fixed spatial shape per cached module")},
        {**common, "name": "haar", "label": "Decimated Haar", "coefficient_ratio": "1x",
         "shift_behavior": "shift variant after decimation", "directionality": "three separable bands",
         "boundary": "periodization; explicit dyadic extension when needed",
         "scientific_role": "short-support localisation control",
         "limitations": ("grid-parity sensitive", "not six signed orientations")},
        {**common, "name": "db2", "label": "Decimated db2", "coefficient_ratio": "1x",
         "shift_behavior": "shift variant after decimation", "directionality": "three separable bands",
         "boundary": "periodization; explicit dyadic extension when needed",
         "scientific_role": "longer-support localisation comparison",
         "limitations": ("larger contaminated margin than Haar", "not six signed orientations")},
    ]

    swt_margins = tuple(valid_interior_halfwidth(wavelet, level)
                        for level in range(1, levels + 1))
    swt_interiors = tuple((max(0, height - 2 * margin), max(0, width - 2 * margin))
                          for margin in swt_margins)
    swt_supports = tuple(filter_support(wavelet, level) for level in range(1, levels + 1))
    swt_usable = swt_supports[-1] <= min(height, width) and all(
        y > 0 and x > 0 for y, x in swt_interiors
    )
    entries.append({
        **common,
        "name": "swt",
        "label": "%s stationary wavelet" % wavelet,
        "coefficient_ratio": "%dx" % (1 + 3 * levels),
        "shift_behavior": "exact translation equivariance under periodic boundaries",
        "directionality": "three separable LH/HL/HH bands; not six signed orientations",
        "boundary": "periodic; regional wrap is artificial",
        "scientific_role": "shift-invariant multiscale tracking/comparison arm",
        "limitations": (
            "exclude the reported scale-dependent edge margin on regional crops",
            "divide raw level energy by 4**level before comparing scales",
            "memory and coefficient traffic grow as 1+3*levels",
        ),
        "selected_configuration": {
            "wavelet": wavelet,
            "levels": levels,
            "spatial_shape": (height, width),
            "coefficient_channels_per_input_channel": 1 + 3 * levels,
            "accumulated_support_by_level": swt_supports,
            "valid_interior_halfwidth_by_level": swt_margins,
            "valid_interior_shape_by_level": swt_interiors,
            "coarsest_scale_has_valid_interior": swt_usable,
            "implementation_policy": "FFT on CPU; convolution on CUDA/ROCm/MPS",
        },
    })
    dtcwt_margins = tuple(dtcwt_engine.valid_interior_halfwidth(
        level, "near_sym_b", "qshift_b") for level in range(1, levels + 1))
    dtcwt_native_margins = tuple(dtcwt_engine.native_halfwidth(
        level, "near_sym_b", "qshift_b") for level in range(1, levels + 1))
    dtcwt_factor = 2 ** levels
    dtcwt_padded_height = ((height + dtcwt_factor - 1) // dtcwt_factor) * dtcwt_factor
    dtcwt_padded_width = ((width + dtcwt_factor - 1) // dtcwt_factor) * dtcwt_factor
    dtcwt_native_shapes = tuple(
        (dtcwt_padded_height // (2 ** level), dtcwt_padded_width // (2 ** level))
        for level in range(1, levels + 1)
    )
    dtcwt_native_interiors = tuple(
        (max(0, shape[0] - 2 * margin), max(0, shape[1] - 2 * margin))
        for shape, margin in zip(dtcwt_native_shapes, dtcwt_native_margins)
    )
    entries.append({
        **common,
        "name": "dtcwt",
        "label": "Dual-tree complex wavelet",
        "coefficient_ratio": "4x before dyadic padding",
        "shift_behavior": "near shift invariant, not exact",
        "directionality": "six oriented complex bands",
        "boundary": "symmetric extension; scale/filter-dependent valid interior",
        "scientific_role": "orientation-aware, near-shift-invariant comparison arm",
        "limitations": (
            "level-1 direction centres deviate by up to 7.1 degrees; prefer level >=2",
            "exclude the reported scale-dependent edge margin on regional crops",
            "atlas tile adjacency is storage geometry, not spatial adjacency",
        ),
        "selected_configuration": {
            "level1": "near_sym_b", "qshift": "qshift_b", "levels": levels,
            "spatial_shape": (height, width),
            "atlas_planes_per_input_channel": 4,
            "valid_interior_halfwidth_parent_px_by_level": dtcwt_margins,
            "valid_interior_halfwidth_native_px_by_level": dtcwt_native_margins,
            "native_shape_by_level": dtcwt_native_shapes,
            "valid_interior_native_shape_by_level": dtcwt_native_interiors,
            "coarsest_scale_has_valid_interior": bool(dtcwt_native_interiors) and all(
                value > 0 for value in dtcwt_native_interiors[-1]
            ),
            "display_contract": "native magnitude maps with invalid edge inset marked",
        },
    })
    return {
        "contract": "(B,C,H,W) -> encoded tensor -> inverse -> loss.backward()",
        "selected_levels": levels,
        "selected_wavelet": wavelet,
        "selected_spatial_shape": (height, width),
        "representations": entries,
    }
