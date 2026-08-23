"""`CoefficientField` - the (time, scale, orientation, y, x) container (roadmap T4B.1/T4B.4).

**What this is for.** Phase 4A gave the platform a time axis. Everything above 4B needs a
second and third: *scale* and *orientation*. The research question - which small configurations
at t precede which large structures at t+delta - is a statement about a five-dimensional array,
and until now that array had no representation. `apply_transform` returned a per-frame dict
whose keys were `"level_1"`, `"level_2"`, ...; a caller wanting "the energy at scale 3 over
time" had to reassemble it from strings, which is exactly how the time axis went missing before
`FieldSequence` existed.

**The parent-grid claim, stated honestly.** The roadmap says "backed by SWT/DTCWT so all scales
share the parent grid". That is true of SWT by construction and *not* true of DTCWT:

*   **SWT is undecimated.** Every band at every level already has the shape of the source
    field. Nothing is resampled, and `resampled_to_parent` is `False`.
*   **DTCWT is decimated.** A level-`j` subband is roughly `H/2**j x W/2**j`. To place it on the
    parent grid it must be **upsampled**, and this module does that with nearest-neighbour
    replication, records `resampled_to_parent=True`, and keeps every native shape in the
    metadata. Nearest neighbour rather than interpolation is deliberate: the aligned view is a
    *labelling* of parent pixels by the coefficient covering them, not a smooth field to be
    interpolated, and bilinear blending of complex coefficients mixes phases belonging to
    different spatial positions. **Upsampling invents no information** - the effective
    resolution of scale `j` is still `2**j` pixels, and treating an aligned DTCWT band as if it
    resolved parent-grid detail would be a false claim about the data.

Because of that, **reconstruction never uses the aligned array**. The native per-frame
coefficients are retained and `reconstruct()` inverts those, so "perfect reconstruction per
family" means what it says. A field whose native coefficients have been dropped refuses to
reconstruct rather than returning an approximation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.core.errors import (FieldTooSmallError, InvalidParameterError, ShapeMismatchError,
                             UnknownNameError)
from src.core.level_axis import PRESSURE_HPA, coordinate_for, units_of
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence

#: Families this module knows how to arrange as a (scale, orientation) bank. `fft`, `dct` and
#: `hybrid` are registered transforms but are not banks: they have no scale/orientation
#: factorisation, and admitting them would mean inventing axes that do not exist.
BANK_FAMILIES: Tuple[str, ...] = ("swt", "dtcwt")

#: What the three separable SWT bands actually are. Deliberately *not* labelled in degrees:
#: `HH` responds to both +45 and -45 degree features and a separable real wavelet cannot
#: distinguish them. That ambiguity is the classical motivation for the dual tree, and writing
#: "45 deg" on an SWT band would assert a directional selectivity the transform has not got.
SWT_BAND_MEANING: Dict[str, str] = {
    "LH": "low-pass along y, high-pass along x: varies across columns, responds to "
          "vertically-elongated features",
    "HL": "high-pass along y, low-pass along x: varies down rows, responds to "
          "horizontally-elongated features",
    "HH": "high-pass on both axes: responds to diagonal features of BOTH signs, which a "
          "separable real wavelet cannot separate (this is why DTCWT exists)",
}


class CoefficientField:
    """Wavelet coefficients on axes `(time, scale, orientation, y, x)`.

    Mirrors `PhysicalField`'s shape - data plus a grid plus metadata - with the invariants
    checked once here so nothing above has to re-derive them.
    """

    def __init__(
        self,
        data: torch.Tensor,
        *,
        wavelet_family: str,
        scales: Sequence[Any],
        orientations: Sequence[Any],
        times: Any,
        grid: GridSpec,
        source_variable: Optional[str] = None,
        orientation_convention: str = "unspecified",
        resampled_to_parent: bool = False,
        native_shapes: Optional[Dict[Any, Tuple[int, int]]] = None,
        native: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        level: Optional[float] = None,
        # TG1.5: which vertical coordinate `level` is a value of. `None` means nobody
        # declared one, which is reported as such rather than assumed to be pressure.
        level_axis: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if data.dim() != 5:
            raise ShapeMismatchError(
                "coefficients", tuple(data.shape), "expected rank",
                "(time, scale, orientation, y, x)",
                fix="A CoefficientField is five-dimensional by definition. Collapsing an "
                    "axis before construction is what makes 'the energy at scale 3 over "
                    "time' unanswerable, which is the failure this class exists to prevent.")

        n_time, n_scale, n_orient, height, width = (int(s) for s in data.shape)
        scales = list(scales)
        orientations = list(orientations)

        if len(scales) != n_scale:
            raise ShapeMismatchError("scales", (len(scales),), "data scale axis", (n_scale,),
                                     fix="Every scale must be labelled; an unlabelled scale "
                                         "axis cannot be reported in physical units.")
        if len(orientations) != n_orient:
            raise ShapeMismatchError("orientations", (len(orientations),),
                                     "data orientation axis", (n_orient,),
                                     fix="Every orientation must be labelled.")

        times_array = np.asarray(times)
        if times_array.ndim != 1 or len(times_array) != n_time:
            raise ShapeMismatchError("times", tuple(times_array.shape), "data time axis",
                                     (n_time,),
                                     fix="Supply exactly one timestamp per frame, as "
                                         "FieldSequence does.")

        if tuple(grid.shape) != (height, width):
            raise ShapeMismatchError(
                "grid", tuple(grid.shape), "coefficient spatial axes", (height, width),
                fix="Every scale of a CoefficientField lives on the parent grid. A grid that "
                    "disagrees with the array means the coordinates attached to these "
                    "coefficients describe different pixels than the ones they label.")

        self.data = data
        self.wavelet_family = wavelet_family
        self.scales = scales
        self.orientations = orientations
        self.times = times_array
        self.grid = grid
        self.source_variable = source_variable
        self.orientation_convention = orientation_convention
        self.resampled_to_parent = bool(resampled_to_parent)
        self.native_shapes = dict(native_shapes or {})
        self.level = level
        if level_axis is not None:
            coordinate_for(level_axis)  # refuse an unregistered vertical coordinate
        self.level_axis = None if level_axis is None else str(level_axis)
        self.config = dict(config or {})
        self.metadata = dict(metadata or {})
        self._native = native

    # ------------------------------------------------------------------ shape and access

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def n_times(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_scales(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_orientations(self) -> int:
        return int(self.data.shape[2])

    @property
    def is_complex(self) -> bool:
        return bool(torch.is_complex(self.data))

    def scale_index(self, scale: Any) -> int:
        """Index of a scale by its label, or the label refused with the ones that exist."""
        if scale in self.scales:
            return self.scales.index(scale)
        raise UnknownNameError("scale", scale, [str(s) for s in self.scales])

    def orientation_index(self, orientation: Any) -> int:
        if orientation in self.orientations:
            return self.orientations.index(orientation)
        raise UnknownNameError("orientation", orientation,
                               [str(o) for o in self.orientations])

    def band(self, t: int, scale: Any, orientation: Any) -> torch.Tensor:
        """One `(y, x)` coefficient band, complex where the family is complex."""
        return self.data[t, self.scale_index(scale), self.orientation_index(orientation)]

    def magnitude_field(self, t: int, scale: Any, orientation: Any) -> PhysicalField:
        """One band as a real `PhysicalField`, so it can flow through existing analysis.

        Magnitude, not the raw coefficient: a complex band has no real-valued rendering that
        is not a choice, and taking the real part would silently discard the phase that makes
        DTCWT approximately shift-invariant in the first place.
        """
        band = self.band(t, scale, orientation)
        values = torch.abs(band) if torch.is_complex(band) else band
        return PhysicalField(
            values.to(torch.float64), grid=self.grid,
            metadata={**self.metadata, "wavelet_family": self.wavelet_family,
                      "scale": scale, "orientation": orientation,
                      "quantity": ("coefficient magnitude" if torch.is_complex(band)
                                   else "coefficient"),
                      "source_variable": self.source_variable,
                      "resampled_to_parent": self.resampled_to_parent})

    def select(self, *, times: Optional[Sequence[int]] = None,
               scales: Optional[Sequence[Any]] = None,
               orientations: Optional[Sequence[Any]] = None) -> "CoefficientField":
        """A sub-bank, still a `CoefficientField` - never a bare tensor.

        Returning a tensor here would drop the labels, and an unlabelled `(T, S, O, Y, X)`
        array is the exact failure mode this class was built to end.
        """
        t_idx = list(range(self.n_times)) if times is None else [int(t) for t in times]
        s_idx = ([self.scale_index(s) for s in scales] if scales is not None
                 else list(range(self.n_scales)))
        o_idx = ([self.orientation_index(o) for o in orientations] if orientations is not None
                 else list(range(self.n_orientations)))
        data = self.data[t_idx][:, s_idx][:, :, o_idx]
        return CoefficientField(
            data, wavelet_family=self.wavelet_family,
            scales=[self.scales[i] for i in s_idx],
            orientations=[self.orientations[i] for i in o_idx],
            times=self.times[t_idx], grid=self.grid,
            source_variable=self.source_variable,
            orientation_convention=self.orientation_convention,
            resampled_to_parent=self.resampled_to_parent,
            native_shapes={self.scales[i]: self.native_shapes[self.scales[i]]
                           for i in s_idx if self.scales[i] in self.native_shapes},
            native=None,  # a subset cannot be inverted; see reconstruct()
            config=self.config, level=self.level, level_axis=self.level_axis,
            metadata={**self.metadata, "subset_of": "CoefficientField"})

    # ------------------------------------------------------------------ energy

    def energy(self) -> torch.Tensor:
        """`(time, scale, orientation)` summed squared magnitude.

        `abs()**2` rather than `**2`, because for a complex band the square of the complex
        value is complex and its sum is not an energy.
        """
        magnitude = torch.abs(self.data) if self.is_complex else self.data
        return (magnitude.to(torch.float64) ** 2).sum(dim=(-2, -1))

    def energy_fractions(self) -> torch.Tensor:
        """Energy normalised within each frame, so it is invariant to a global rescale."""
        energy = self.energy()
        total = energy.sum(dim=(1, 2), keepdim=True)
        # A frame of exact zeros has no energy to apportion. Returning zeros rather than NaN
        # keeps the array usable, and the zero total stays visible in `energy()` itself.
        safe = torch.where(total > 0, total, torch.ones_like(total))
        return energy / safe

    # ------------------------------------------------------------------ native geometry

    def replication_factor(self, scale: Any) -> float:
        """Parent-grid cells occupied by one native coefficient of `scale`.

        `1.0` for an undecimated family. For DTCWT it is `4**j`, and it is the reason the
        next two methods exist.
        """
        if not self.resampled_to_parent:
            return 1.0
        native = self.native_shapes.get(scale)
        if native is None:
            raise InvalidParameterError(
                "scale", scale,
                "a scale whose native shape was recorded. This field declares "
                "resampled_to_parent=True but carries no native shape for %r, so the "
                "replication factor cannot be recovered and any per-scale energy taken from "
                "the aligned view would be wrong by an unknown factor" % (scale,))
        height, width = self.shape[-2:]
        return (height * width) / float(native[0] * native[1])

    def available_coefficients(self) -> List[int]:
        """Coefficients the transform actually computed per scale, over all orientations.

        Rule R3's second trap in one number. In a decimated pyramid the *number of available
        coefficients* falls as `s**-2`, so any per-scale quantity that is not divided by this
        is measuring the pyramid's geometry as much as the field.
        """
        n_orientations = self.n_orientations
        counts = []
        for scale in self.scales:
            if self.resampled_to_parent:
                native = self.native_shapes.get(scale)
                if native is None:
                    raise InvalidParameterError(
                        "native_shapes", None,
                        "a native shape for every scale; without it the available "
                        "coefficient count per scale is unknown")
                counts.append(int(native[0] * native[1]) * n_orientations)
            else:
                height, width = self.shape[-2:]
                counts.append(int(height * width) * n_orientations)
        return counts

    def native_band(self, t: int, scale: Any, orientation: Any) -> torch.Tensor:
        """One band as the transform computed it, before any alignment.

        Recovered from the retained native coefficients where they exist, and otherwise by
        **subsampling the aligned view on the replication stride** - which is exact, not an
        approximation, because the alignment is nearest-neighbour replication and
        `interpolate(..., mode='nearest')` maps output index `i` to input index `i // r`.
        A field restored from an artifact therefore still yields its true coefficients, which
        is what lets a scale signature be computed from stored lineage rather than only from
        a live decomposition.

        Refused rather than guessed when the parent shape is not an exact multiple of the
        native shape: the stride would then be fractional and the subsample would silently
        pick up duplicated rows.
        """
        aligned = self.band(t, scale, orientation)
        if not self.resampled_to_parent:
            return aligned

        s_index = self.scale_index(scale)
        o_index = self.orientation_index(orientation)
        if self._native is not None and self.wavelet_family == "dtcwt":
            return self._native[t]["highpass"][s_index][:, :, o_index]

        native = self.native_shapes.get(scale)
        if native is None:
            raise InvalidParameterError(
                "scale", scale, "a scale whose native shape was recorded")
        height, width = self.shape[-2:]
        if height % native[0] or width % native[1]:
            raise ShapeMismatchError(
                "aligned band", (height, width), "an exact multiple of the native shape",
                tuple(native),
                fix="The replication stride is not an integer, so the aligned view cannot be "
                    "subsampled back to the native coefficients. Re-run the decomposition "
                    "with keep_native=True, or use a parent grid whose size is a multiple of "
                    "2**levels.")
        return aligned[::height // native[0], ::width // native[1]]

    def native_energy(self) -> torch.Tensor:
        """`(time, scale, orientation)` energy of the coefficients **as computed**.

        Identical to `energy()` for an undecimated family. For a resampled one it is the
        aligned energy divided by the replication factor - the energy the transform actually
        produced, which is the one Parseval relates to the reconstructed band.
        """
        out = torch.zeros((self.n_times, self.n_scales, self.n_orientations),
                          dtype=torch.float64)
        for t in range(self.n_times):
            for s_i, scale in enumerate(self.scales):
                for o_i, orientation in enumerate(self.orientations):
                    band = self.native_band(t, scale, orientation)
                    magnitude = torch.abs(band) if torch.is_complex(band) else band
                    out[t, s_i, o_i] = (magnitude.to(torch.float64) ** 2).sum()
        return out

    def energy_density(self) -> torch.Tensor:
        """`(time, scale, orientation)` energy **per available coefficient** (rule R3).

        `energy()` reports the energy of the array as stored, which on a resampled family is
        inflated at level `j` by the replication factor `4**j`: the aligned band repeats each
        native coefficient that many times. The inflation is worth stating plainly because
        the obvious conclusion from it is wrong, and it was checked rather than assumed:

        *   **Energy *fractions* are unaffected.** `density(j) = aligned(j) / (H * W)`, and
            `H * W` does not depend on the scale, so the replication factor cancels in the
            ratio exactly. `energy_fractions()` is therefore already the R3-normalised
            quantity, not a biased one. A test asserts the two agree to float64.
        *   **Absolute per-coefficient energy is not**, and it is what a power-law fit against
            scale consumes, so that fit uses this method and not `energy()`.

        The same distinction decides where the concentration measures are computed: a
        participation ratio taken on the aligned view is `r` times too large, because
        replicating every coefficient `r` times multiplies it by `r`. The Gini coefficient
        survives replication unchanged. Both facts are asserted in `test_scale_signature.py`
        rather than left as reasoning.
        """
        per_orientation = torch.tensor(
            [count / self.n_orientations for count in self.available_coefficients()],
            dtype=torch.float64).view(1, -1, 1)
        return self.native_energy() / per_orientation

    # ------------------------------------------------------------------ reconstruction

    def has_native(self) -> bool:
        return self._native is not None

    def drop_native(self) -> "CoefficientField":
        """Release the native coefficients. Saves memory; forfeits reconstruction."""
        self._native = None
        return self

    def reconstruct(self) -> FieldSequence:
        """Invert to a `FieldSequence` from the retained *native* coefficients.

        Not from `self.data`. For DTCWT the aligned array is an upsampled view, and inverting
        it would produce something that looked like a reconstruction and was not one - the
        worst available outcome, because it fails silently and only by a few percent.
        """
        if self._native is None:
            raise InvalidParameterError(
                "native_coefficients", None,
                "coefficients retained at construction. This CoefficientField was built with "
                "keep_native=False or has had drop_native() called, so the only coefficients "
                "left are the parent-grid aligned view. Inverting that view would return an "
                "approximation indistinguishable from a real reconstruction; re-run the "
                "decomposition with keep_native=True instead")

        from src.transform_engine import dtcwt as dtcwt_mod
        from src.transform_engine import stationary as swt_mod

        fields: List[PhysicalField] = []
        for coeffs in self._native:
            if self.wavelet_family == "swt":
                recovered = swt_mod.inverse_swt2d(coeffs)
            else:
                recovered = dtcwt_mod.inverse_dtcwt2d(coeffs)
            fields.append(PhysicalField(recovered.data, grid=self.grid,
                                        metadata={"reconstructed_from": self.wavelet_family}))
        return FieldSequence(fields, self.times,
                             metadata={"reconstructed_from": self.wavelet_family,
                                       "source_variable": self.source_variable})

    # ------------------------------------------------------------------ provenance

    def scale_wavelength_bands(self) -> Optional[List[Dict[str, Any]]]:
        """Physical wavelength band per scale, or `None` on a non-physical grid.

        A dyadic level `j` detail band covers wavelengths of roughly `2**j` to `2**(j+1)`
        samples. Reported as a *band* rather than a single number because that is what a
        wavelet level is; quoting one wavelength would imply a selectivity the filter has not
        got. `None` rather than a guess when the grid carries no physical spacing - a
        wavelength in metres derived from a pixel grid would be fabricated.
        """
        if not self.grid.is_physical:
            return None
        dx = self.grid.representative_dx_metres()
        dy = self.grid.representative_dy_metres()
        bands = []
        for position, label in enumerate(self.scales, start=1):
            level = int(label) if isinstance(label, (int, float)) else position
            bands.append({
                "scale": label,
                "min_wavelength_x_m": float(2 ** level * dx),
                "max_wavelength_x_m": float(2 ** (level + 1) * dx),
                "min_wavelength_y_m": float(2 ** level * dy),
                "max_wavelength_y_m": float(2 ** (level + 1) * dy),
                "effective_resolution_px": 2 ** level,
            })
        return bands

    def summary(self) -> Dict[str, Any]:
        """A lineage-safe description: labels, statistics and provenance, never the payload."""
        energy = self.energy()
        fractions = self.energy_fractions()
        magnitude = torch.abs(self.data) if self.is_complex else self.data
        record: Dict[str, Any] = {
            "type": "CoefficientField",
            "wavelet_family": self.wavelet_family,
            "shape": list(self.shape),
            "axes": ["time", "scale", "orientation", "y", "x"],
            "n_times": self.n_times,
            "scales": [str(s) for s in self.scales],
            "orientations": [str(o) for o in self.orientations],
            "orientation_convention": self.orientation_convention,
            "is_complex": self.is_complex,
            "dtype": str(self.data.dtype),
            "source_variable": self.source_variable,
            "level": self.level,
            "level_axis": self.level_axis,
            "level_units": units_of(self.level_axis),
            "grid": self.grid.to_provenance(),
            "resampled_to_parent": self.resampled_to_parent,
            "config": dict(self.config),
            "reconstructable": self.has_native(),
            # Mean over time of the per-frame energy fraction: the compact form of "which
            # scale carries the structure", small enough for a lineage row.
            "mean_energy_fraction": [[float(v) for v in row]
                                     for row in fractions.mean(dim=0)],
            "total_energy_per_scale": [float(v) for v in energy.sum(dim=(0, 2))],
            "magnitude_min": float(magnitude.min()),
            "magnitude_max": float(magnitude.max()),
            "magnitude_mean": float(magnitude.mean()),
            "n_nonfinite": int(torch.count_nonzero(~torch.isfinite(magnitude))),
            "metadata": dict(self.metadata),
        }
        if self.resampled_to_parent:
            record["native_shapes"] = {str(k): list(v)
                                       for k, v in self.native_shapes.items()}
            record["resampling"] = (
                "nearest-neighbour upsampling to the parent grid. No information is added: "
                "the effective resolution of scale j remains 2**j pixels, and reconstruction "
                "uses the retained native coefficients, never this view.")
        bands = self.scale_wavelength_bands()
        if bands is not None:
            record["scale_wavelength_bands"] = bands
        if self.wavelet_family == "swt":
            record["band_meaning"] = dict(SWT_BAND_MEANING)
        return record

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return ("CoefficientField(%s, shape=%s, scales=%s, orientations=%s)"
                % (self.wavelet_family, self.shape, self.scales, self.orientations))


# --------------------------------------------------------------------------- decomposition

def _upsample_to(band: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour replication of a decimated band onto the parent grid.

    Complex tensors are handled by upsampling the real and imaginary parts separately;
    `interpolate` does not accept complex input, and under nearest neighbour that is identical
    to replicating the complex value, so no phase is disturbed.
    """
    target_h, target_w = shape
    if tuple(band.shape) == (target_h, target_w):
        return band
    if torch.is_complex(band):
        return torch.complex(_upsample_to(band.real, shape), _upsample_to(band.imag, shape))
    resized = torch.nn.functional.interpolate(
        band[None, None].to(torch.float64), size=(target_h, target_w), mode="nearest")
    return resized[0, 0]


def _swt_bank(field: PhysicalField, config: Dict[str, Any]):
    from src.transform_engine import stationary as swt_mod

    levels = int(config.get("levels", 3))
    coeffs = swt_mod.apply_swt2d(
        field, levels=levels, wavelet=config.get("wavelet", "haar"),
        mode=config.get("mode", "periodic"))
    bands = []
    for level in range(1, levels + 1):
        block = coeffs["level_%d" % level]
        bands.append(torch.stack([block[name] for name in ("LH", "HL", "HH")], dim=0))
    return torch.stack(bands, dim=0), coeffs


def _dtcwt_bank(field: PhysicalField, config: Dict[str, Any], parent_shape: Tuple[int, int]):
    from src.transform_engine import dtcwt as dtcwt_mod

    levels = int(config.get("levels", 3))
    coeffs = dtcwt_mod.apply_dtcwt2d(
        field, levels=levels, level1=config.get("level1", "near_sym_b"),
        qshift=config.get("qshift", "qshift_b"))
    bands = []
    native_shapes: Dict[Any, Tuple[int, int]] = {}
    for level, highpass in enumerate(coeffs["highpass"], start=1):
        native_shapes[level] = (int(highpass.shape[0]), int(highpass.shape[1]))
        oriented = [_upsample_to(highpass[:, :, o], parent_shape) for o in range(6)]
        bands.append(torch.stack(oriented, dim=0))
    return torch.stack(bands, dim=0), coeffs, native_shapes


def decompose_sequence(
    sequence: FieldSequence,
    family: str = "swt",
    config: Optional[Dict[str, Any]] = None,
    keep_native: bool = True,
) -> CoefficientField:
    """Decompose every frame of a `FieldSequence` into one `CoefficientField`.

    The whole point of doing it here rather than frame by frame in a caller: the time axis
    survives into the result, so "the energy at scale 3 over time" is an array slice instead
    of a reassembly job.
    """
    from src.transform_engine import dtcwt as dtcwt_mod

    config = dict(config or {})
    if family not in BANK_FAMILIES:
        raise UnknownNameError(
            "wavelet bank family", family, list(BANK_FAMILIES),
            hint="fft, dct and hybrid are registered transforms but are not banks: they have "
                 "no (scale, orientation) factorisation, and giving them one would invent "
                 "axes the transform does not have.")
    if not isinstance(sequence, FieldSequence):
        raise InvalidParameterError("sequence", type(sequence).__name__, "a FieldSequence")

    parent_shape = (sequence.shape[1], sequence.shape[2])
    levels = int(config.get("levels", 3))

    if family == "dtcwt" and min(parent_shape) < 2 ** levels:
        raise FieldTooSmallError(
            "dtcwt bank at %d levels" % levels, parent_shape, (2 ** levels, 2 ** levels),
            fix="Reduce `levels` or use a larger crop; a level whose filter is wider than "
                "the field measures the boundary treatment, not the atmosphere (rule R13).")

    stacked: List[torch.Tensor] = []
    native: List[Dict[str, Any]] = []
    native_shapes: Dict[Any, Tuple[int, int]] = {}
    resampled = family == "dtcwt"

    for field in sequence:
        if family == "swt":
            bank, coeffs = _swt_bank(field, config)
        else:
            bank, coeffs, native_shapes = _dtcwt_bank(field, config, parent_shape)
        stacked.append(bank)
        if keep_native:
            native.append(coeffs)

    data = torch.stack(stacked, dim=0)
    scales = list(range(1, levels + 1))

    if family == "swt":
        orientations: List[Any] = ["LH", "HL", "HH"]
        convention = ("separable band labels, not angles: HH responds to both diagonal signs "
                      "and cannot distinguish them")
    else:
        orientations = [float(d) for d in dtcwt_mod.SUBBAND_FEATURE_DEG]
        convention = ("feature orientation in degrees (the direction of the elongated "
                      "feature, 90 degrees from the wavevector); see "
                      "dtcwt.SUBBAND_WAVEVECTOR_DEG for the other convention")

    first = sequence.at(0)
    return CoefficientField(
        data,
        wavelet_family=family,
        scales=scales,
        orientations=orientations,
        times=sequence.times_seconds,
        grid=sequence.grid,
        source_variable=first.metadata.get("variable") or sequence.metadata.get("variable"),
        orientation_convention=convention,
        resampled_to_parent=resampled,
        native_shapes=native_shapes,
        native=native if keep_native else None,
        config={"family": family, **config},
        level=sequence.metadata.get("level"),
        # TG1.5: the coordinate travels with the number, from the reader that knew it. Read
        # from the same place as `level` and by the same rule, so a sequence that carries one
        # and not the other is reported as exactly that rather than assumed to be pressure.
        level_axis=sequence.metadata.get("level_axis"),
        metadata={"n_source_frames": len(sequence),
                  "source_units": sequence.units,
                  "is_simulated": sequence.metadata.get("is_simulated"),
                  "time_kind": sequence.time_kind})


def decompose_field(field: PhysicalField, family: str = "swt",
                    config: Optional[Dict[str, Any]] = None,
                    keep_native: bool = True) -> CoefficientField:
    """One field as a one-frame `CoefficientField`, so a snapshot and a record share a type."""
    sequence = FieldSequence([field], [0.0], metadata=dict(field.metadata))
    return decompose_sequence(sequence, family=family, config=config, keep_native=keep_native)


# --------------------------------------------------------------------------- level banks

class LevelBank:
    """`CoefficientField`s at several levels of one declared vertical axis (T4B.4, TG1.5).

    **Why level is a bank dimension and not a filter.** The most famous precursor relationship
    in synoptic meteorology is vertical: an upper-level trough preceding surface cyclogenesis.
    While level was only a selector - "analyse 500 hPa" - that relationship could not even be
    expressed, because two levels were two unrelated runs with nothing tying their time axes
    together.

    **Scope, stated plainly: this is 2D-per-level, not a 3D wavelet transform.** Nothing here
    resolves vertical structure *within* a decomposition; each level is decomposed
    independently, and the vertical relationship is carried as an edge attribute by the
    constellation work in 4E. A 3D transform would be a different and much more expensive
    object, and calling this one 3D would misdescribe what it computes.

    **The vertical coordinate is declared (TG1.5).** `level_axis` names a registered
    `LevelCoordinate`, which supplies the units and - the part that is not cosmetic - which
    direction along the axis is up. It defaults to `pressure_hpa` because every caller in the
    tree is atmospheric and the default is recorded in `summary()`, but the direction of a
    vertical offset is now read from the coordinate rather than assumed. A bank of heights
    declares `height_m` and gets its offsets labelled the right way round.
    """

    def __init__(self, banks: Dict[float, CoefficientField],
                 metadata: Optional[Dict[str, Any]] = None,
                 level_axis: str = PRESSURE_HPA) -> None:
        if not banks:
            raise InvalidParameterError(
                "banks", {}, "at least one level. An empty bank has no grid and no "
                             "time axis to validate anything else against")

        self.level_axis = str(level_axis)
        self.coordinate = coordinate_for(self.level_axis)
        self.levels: List[float] = list(self.coordinate.ordered(banks))
        self.banks: Dict[float, CoefficientField] = {float(k): v for k, v in banks.items()}

        reference = self.banks[self.levels[0]]
        for level in self.levels[1:]:
            other = self.banks[level]
            if other.shape != reference.shape:
                raise ShapeMismatchError(
                    "level %g" % self.levels[0], reference.shape,
                    "level %g" % level, other.shape,
                    fix="Levels of one bank must be decomposed identically; otherwise a "
                        "cross-level comparison is comparing different transforms.")
            if not np.array_equal(other.times, reference.times):
                raise InvalidParameterError(
                    "banks[%g].times" % level, "different time axis",
                    "the same timestamps as level %g. A vertical lead-lag measured across "
                    "levels sampled at different times is measuring the sampling, not the "
                    "atmosphere" % self.levels[0])
            if other.wavelet_family != reference.wavelet_family:
                raise InvalidParameterError(
                    "banks[%g].wavelet_family" % level, other.wavelet_family,
                    "the same family as level %g (%s)"
                    % (self.levels[0], reference.wavelet_family))
        self.metadata = dict(metadata or {})

    @property
    def level_units(self) -> str:
        return self.coordinate.units

    def axis_spec(self, name: str = "level"):
        """This bank's vertical axis as a declared `level`-role `AxisSpec` (TG1.1)."""
        return self.coordinate.axis_spec(name)

    def __len__(self) -> int:
        return len(self.levels)

    def at_level(self, level: float) -> CoefficientField:
        value = float(level)
        if value not in self.banks:
            raise UnknownNameError("level (%s)" % self.coordinate.units, level,
                                   ["%g" % lev for lev in self.levels])
        return self.banks[value]

    @property
    def reference(self) -> CoefficientField:
        return self.banks[self.levels[0]]

    def vertical_offsets(self) -> List[Dict[str, Any]]:
        """Every ordered level pair with its signed offset - the 4E edge attribute.

        Signed and in the axis's own units: "500 hPa relative to 850 hPa" is `-350`, i.e.
        *above*, and the sign is what distinguishes an upper-level precursor from a surface
        one. **Which sign means up is the coordinate's business, not this method's.** Before
        TG1.5 the rule was written here as `"upward" if upper < lower`, which is right for
        pressure and for depth and backwards for height - a hardcoded atmospheric convention
        sitting at the one place that reports the direction of a vertical relationship.
        """
        pairs = []
        for lower in self.levels:
            for upper in self.levels:
                if lower == upper:
                    continue
                pairs.append({
                    "from_level": lower,
                    "to_level": upper,
                    "offset": self.coordinate.offset(lower, upper),
                    "level_units": self.coordinate.units,
                    "direction": self.coordinate.direction(lower, upper),
                })
        return pairs

    def to_tensor(self) -> torch.Tensor:
        """`(level, time, scale, orientation, y, x)`, levels ascending in the axis's units."""
        return torch.stack([self.banks[lev].data for lev in self.levels], dim=0)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": "LevelBank",
            "levels": list(self.levels),
            "n_levels": len(self.levels),
            "shape_per_level": list(self.reference.shape),
            "wavelet_family": self.reference.wavelet_family,
            "scope": ("2D decomposition per level, not a 3D wavelet transform: no vertical "
                      "structure is resolved within a decomposition, and the vertical "
                      "relationship is carried as a cross-level edge attribute"),
            "vertical_offsets": sorted({p["offset"] for p in self.vertical_offsets()}),
            "per_level": {"%g" % lev: self.banks[lev].summary() for lev in self.levels},
            "metadata": dict(self.metadata),
            **self.coordinate.describe(self.level_axis),
        }


def decompose_levels(sequences: Dict[float, FieldSequence], family: str = "swt",
                     config: Optional[Dict[str, Any]] = None,
                     keep_native: bool = False,
                     level_axis: str = PRESSURE_HPA) -> LevelBank:
    """Decompose one sequence per level into a `LevelBank` (T4B.4).

    `keep_native` defaults to `False` here and `True` for a single sequence: a bank over levels
    multiplies the native payload by the number of levels, and the reason to build one is
    cross-level analysis rather than reconstruction.

    Each decomposed field is stamped with the level *and* the axis it belongs to, so a
    `CoefficientField` pulled out of a bank still knows what its own number means.
    """
    coordinate_for(level_axis)
    banks = {}
    for level, sequence in sequences.items():
        field = decompose_sequence(sequence, family=family, config=config,
                                   keep_native=keep_native)
        field.level = float(level)
        field.level_axis = level_axis
        banks[float(level)] = field
    return LevelBank(banks, metadata={"family": family, "config": dict(config or {})},
                     level_axis=level_axis)
