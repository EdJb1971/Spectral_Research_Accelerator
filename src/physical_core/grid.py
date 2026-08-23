"""Grid geometry: the physical metric that turns pixel indices into metres.

Defect D13 (roadmap T3.5.13, standard E3). Before this module the platform computed
gradients with `torch.gradient` at unit spacing, binned power spectra on integer *pixel*
radius, and averaged errors with an unweighted mean. Each of those is a statement about
the array, not about the atmosphere, and the three fail in different ways:

*   **Gradients.** On a lat/lon grid the zonal metric is ``R cos(lat) dlon`` and the
    meridional metric is ``R dlat``. Unit spacing is therefore wrong by a latitude-dependent
    factor - at 60 degrees the zonal error is 2x, at 80 degrees it is ~5.8x. The error is not
    a constant rescale, so it distorts gradient *direction* as well as magnitude, and no
    amount of downstream normalisation can undo it.
*   **Radial spectra.** A circle in pixel space is an ellipse in physical space whenever
    ``dx != dy``, so annulus averaging mixes genuinely different physical wavenumbers.

    What that does is narrower than it first appears, and the measurement corrected an
    expectation held while writing this module. For a *pure power law* the angular factor
    ``<(cos^2/a^2 + sin^2)^(-beta/2)>`` is independent of radius, so pixel binning
    rescales the intercept and leaves the **slope exact** - measured slope error at aspect
    ratio 4 was +0.001, i.e. no bias at all. The damage is elsewhere, and it is severe:
    pixel ``k`` has no physical meaning (you cannot say "the peak is at 500 km"), spectral
    *features* are smeared across a factor-``aspect`` range of physical wavenumber (a
    Gaussian spectral peak on an aspect-3 grid went from fractional half-width 0.020 to
    1.935, ~97x broader), and two patches at different latitudes have different
    pixel-to-physical maps so their spectra are not comparable. Since a `ScaleSignature` is
    made of exactly those features, this matters more than a slope bias would have. See
    ``test_pixel_binning_preserves_slope_but_destroys_spectral_features``.
*   **Domain statistics.** Cells on a lat/lon grid have areas proportional to
    ``cos(lat)``; an unweighted mean silently over-weights the poleward rows.

A `GridSpec` is attached to every `PhysicalField`. The default is ``GridSpec.pixel(...)``,
which is deliberately *not* None: "we are working in pixel units" is then a recorded fact
that travels with the data and gets printed in the units field of every derived quantity,
rather than an unexamined assumption.

**TG1.2 (standard E13).** `kind` used to be a closed set of three strings and this module
branched on it in eleven places. It is now a key into `src.physical_core.geometry.GEOMETRIES`,
and every one of those branches is a method on the registered `Geometry`. `GridSpec` still
owns the *shape* arithmetic that is true of any geometry - endpoint-preserving resampling,
crop bounds, wavenumber axes - and delegates everything that depends on what the surface
actually is. Callers ask `spec.capability(...)`, never `spec.kind == ...`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from src.core.errors import UnknownNameError
from src.physical_core.geometry import (
    EARTH_RADIUS_M,
    IFS_EARTH_RADIUS_M,
    capability,
    geometry_for,
    recognisers,
)

# EARTH_RADIUS_M / IFS_EARTH_RADIUS_M are defined in `geometry.py` (the spherical geometry is
# what needs them) and re-exported here, because every existing caller imports them from this
# module and a constant moving house is not a reason to touch thirty call sites.
#
# IUGG mean Earth radius (R_1, arithmetic mean of the WGS84 semi-axes), metres.
# ERA5 / IFS use 6371229.0; the difference is 3.5e-5 relative and is recorded rather
# than hidden, because a spectral slope fitted over two decades of k is not sensitive to
# it but an absolute gradient in K/m is quoted to more digits than that.

#: A metric below this many metres is treated as degenerate rather than merely small.
#: cos(90 degrees) evaluates to 6.1e-17 in float64, not 0, so a polar row yields a zonal
#: spacing of ~1e-12 m: finite, enormously wrong, and silently usable. Any threshold that
#: tests `> 0` therefore fails to catch it.
DEGENERATE_METRIC_M = 1e-6

#: Anisotropy beyond this (as |aspect - 1|) makes pixel-space radial binning unsafe.
ANISOTROPY_WARN_THRESHOLD = 0.05

#: Variation of the zonal metric across a lat/lon patch beyond which a single
#: representative dx is a poor approximation and the patch should be regridded.
LATLON_DX_VARIATION_WARN_PCT = 5.0

_WAVENUMBER_UNITS = {
    # name: (factor applied to cycles-per-metre, label)
    "rad_per_m": (2.0 * math.pi, "rad m^-1"),
    "cycles_per_m": (1.0, "cycles m^-1"),
    "cycles_per_km": (1000.0, "cycles km^-1"),
    "rad_per_km": (2.0 * math.pi * 1000.0, "rad km^-1"),
}


class GridError(ValueError):
    """Raised when a grid specification is impossible or an operation needs a metric it lacks."""


@dataclass(frozen=True)
class GridSpec:
    """Physical geometry of a 2D field indexed ``[row, col]`` = ``[y, x]``.

    Three geometries ship as builtins; `kind` is a registry key, so there may be more:

    ``pixel``     - dimensionless index space. Spacing 1, area weights uniform. Honest
                    default; every derived quantity is then labelled ``per_pixel``.
    ``cartesian`` - uniform spacing in metres on a plane. Anisotropy allowed (dy != dx).
    ``latlon``    - regular longitude/latitude grid on a sphere of radius ``radius_m``.
                    ``dlat`` may be negative (ERA5 stores north-to-south); the sign is
                    preserved for coordinate reconstruction and taken as absolute for
                    metrics.

    `lat0`, `lon0` and `radius_m` are ``latlon``'s parameters with named fields, kept for
    compatibility; a geometry registered later puts its own parameters in `params`.
    """

    kind: str
    shape: Tuple[int, int]
    dy: float = 1.0          # cartesian: metres. latlon: degrees latitude (may be negative).
    dx: float = 1.0          # cartesian: metres. latlon: degrees longitude.
    lat0: Optional[float] = None   # latitude of row 0, degrees (latlon only)
    lon0: Optional[float] = None   # longitude of col 0, degrees (latlon only)
    radius_m: float = EARTH_RADIUS_M
    variable_units: Optional[str] = None   # units of the *values*, e.g. "K", "m s^-1"
    notes: Tuple[str, ...] = dc_field(default_factory=tuple)
    #: Geometry-specific scalars, for a geometry the builtin fields do not describe (TG1.2).
    #: `lat0`, `lon0` and `radius_m` are *latlon's* parameters that happen to have named
    #: fields for historical reasons; a registered geometry with a different parameterisation
    #: puts its own here instead of being unable to exist. Stored as a sorted tuple of pairs
    #: rather than a dict so that the dataclass stays frozen, hashable and comparable, which
    #: `notes` already established as the house style for this class.
    params: Tuple[Tuple[str, float], ...] = dc_field(default_factory=tuple)

    # ---------------------------------------------------------------- construction

    def __post_init__(self) -> None:
        # TG1.2: the kind is a registry key, so the valid set is whatever is registered -
        # including a geometry a plugin added a moment ago. `UnknownNameError` is translated
        # rather than propagated because every existing caller catches `GridError` here, and
        # its "did you mean" survives in the message.
        try:
            geometry = geometry_for(self.kind)
        except UnknownNameError as exc:
            raise GridError(
                "%s Use GridSpec.pixel(shape) if the field genuinely has no physical "
                "metric, or register a geometry in src.physical_core.geometry.GEOMETRIES."
                % (exc,)
            ) from None
        if len(self.shape) != 2:
            raise GridError("GridSpec.shape must be (H, W), got %r" % (self.shape,))
        if self.shape[0] < 1 or self.shape[1] < 1:
            raise GridError("GridSpec.shape must be positive, got %r" % (self.shape,))

        # Accept a mapping for convenience and normalise to the canonical sorted tuple, so
        # two grids built from equal parameters compare equal regardless of insertion order.
        if isinstance(self.params, Mapping):
            object.__setattr__(self, "params",
                               tuple(sorted((str(k), float(v))
                                            for k, v in self.params.items())))
        else:
            object.__setattr__(self, "params",
                               tuple(sorted((str(k), float(v)) for k, v in self.params)))

        geometry.validate(self)

    def param(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """One geometry-specific scalar, or ``default`` (TG1.2).

        A geometry that *requires* a parameter should raise from its `validate`, not read a
        default here: a grid missing the number that gives it a metric has no metric, and
        discovering that at the first gradient rather than at construction is precisely the
        failure D13 exists to prevent.
        """
        for key, value in self.params:
            if key == name:
                return value
        return default

    # ---------------------------------------------------------------- geometry

    @property
    def geometry(self) -> Any:
        """The registered `Geometry` for this grid's `kind` (TG1.2)."""
        return geometry_for(self.kind)

    def capability(self, key: str, default: Any = None) -> Any:
        """One declared capability of this grid's geometry.

        Standard E13: ask what a geometry *can do*, never which one it is. `is_physical`,
        `length_units` and every branch in `operators.py` go through here.
        """
        return capability(self.kind, key, default)

    def assert_uniform_metric(self, operation: str) -> None:
        """Refuse an operation that assumes one spacing for the whole grid.

        The pre-TG1.2 code had no equivalent: `laplacian`'s ``else`` branch applied the
        five-point Cartesian stencil to *anything* that was not ``latlon``, so a geometry
        whose metric varies across the grid would have received a plausible number labelled
        ``value per m^2``. Now the else-branch states its precondition.
        """
        if not self.capability("uniform_metric", True):
            raise GridError(
                "%s assumes a single spacing for the whole grid, but geometry %r declares "
                "uniform_metric=False - its metric varies from row to row, so one dy/dx is "
                "not a description of it. Either give %r a specialised implementation, or "
                "regrid to a geometry with a uniform metric first."
                % (operation, self.kind, self.kind)
            )

    @classmethod
    def pixel(cls, shape: Tuple[int, int], variable_units: Optional[str] = None) -> "GridSpec":
        """Dimensionless index space - the explicit default."""
        return cls(kind="pixel", shape=tuple(shape), dy=1.0, dx=1.0,
                   variable_units=variable_units)

    @classmethod
    def cartesian(
        cls,
        shape: Tuple[int, int],
        dy_m: float,
        dx_m: Optional[float] = None,
        variable_units: Optional[str] = None,
    ) -> "GridSpec":
        """Uniform planar grid with spacing in metres. ``dx_m`` defaults to ``dy_m``."""
        return cls(kind="cartesian", shape=tuple(shape), dy=float(dy_m),
                   dx=float(dy_m if dx_m is None else dx_m), variable_units=variable_units)

    @classmethod
    def latlon(
        cls,
        shape: Tuple[int, int],
        lat0: float,
        dlat: float,
        lon0: float,
        dlon: float,
        radius_m: float = EARTH_RADIUS_M,
        variable_units: Optional[str] = None,
    ) -> "GridSpec":
        """Regular lon/lat grid on a sphere. ``dlat`` may be negative (north-to-south)."""
        return cls(kind="latlon", shape=tuple(shape), dy=float(dlat), dx=float(dlon),
                   lat0=float(lat0), lon0=float(lon0), radius_m=float(radius_m),
                   variable_units=variable_units)

    @classmethod
    def from_coords(
        cls,
        coords: Dict[str, torch.Tensor],
        shape: Tuple[int, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GridSpec":
        """Infer a grid from coordinate vectors, falling back to ``pixel``.

        Each registered geometry is asked, in ascending `coord_priority`, whether it
        recognises these coordinates; ``pixel`` sits last and always says yes. Anything no
        geometry recognises therefore yields a pixel grid, because guessing a metric is worse
        than recording that we do not have one.

        Coordinate *names* come from TG1.1's registry rather than from literals here, so
        ``latitude``/``longitude`` spelled in full - which is how CF and ERA5 spell them - is
        now recognised as a sphere. Before TG1.2 it fell through to a pixel grid: a field
        with a perfectly good spherical metric, silently analysed in array indices.
        """
        metadata = metadata or {}
        shape = tuple(shape)
        var_units = metadata.get("units")
        for geometry in recognisers():
            fields = geometry.from_coords(coords, shape, metadata)
            if fields is not None:
                return cls(variable_units=var_units, **fields)
        return cls.pixel(shape, variable_units=var_units)

    # ---------------------------------------------------------------- derivation

    def resampled(self, new_shape: Tuple[int, int]) -> "GridSpec":
        """The grid of this field after interpolation to ``new_shape``.

        Assumes ``align_corners=True`` interpolation, which preserves the coordinate
        *endpoints*; the spacing therefore scales as ``(n_old - 1) / (n_new - 1)``, not as
        ``n_old / n_new``. Getting that wrong puts a half-cell offset into every physical
        length after a resample, which is precisely the sort of error that survives all the
        way to a published number, so it lives here once rather than at each call site.
        """
        new_shape = tuple(new_shape)
        if len(new_shape) != 2 or new_shape[0] < 1 or new_shape[1] < 1:
            raise GridError("resampled() needs a positive (H, W), got %r" % (new_shape,))

        def scale(old_n: int, new_n: int, spacing: float) -> float:
            if new_n == 1 or old_n == 1:
                return spacing
            return spacing * (old_n - 1) / (new_n - 1)

        dy = scale(self.height, new_shape[0], self.dy)
        dx = scale(self.width, new_shape[1], self.dx)
        return dc_replace(self, **self.geometry.resampled(self, new_shape, dy, dx))

    def subset(self, row_start: int = 0, row_stop: Optional[int] = None,
               col_start: int = 0, col_stop: Optional[int] = None) -> "GridSpec":
        """The grid of a contiguous crop ``[row_start:row_stop, col_start:col_stop]``.

        Spacing is unchanged; the *origin* moves. For a lat/lon grid that changes the mean
        latitude and therefore the representative zonal metric, which is why a crop must
        carry a new GridSpec rather than reuse its parent's.
        """
        row_stop = self.height if row_stop is None else row_stop
        col_stop = self.width if col_stop is None else col_stop
        h = row_stop - row_start
        w = col_stop - col_start
        if h < 1 or w < 1:
            raise GridError(
                "subset is empty: rows [%d:%d] cols [%d:%d] of a %dx%d grid"
                % (row_start, row_stop, col_start, col_stop, self.height, self.width)
            )
        return dc_replace(self, **self.geometry.subset(self, row_start, col_start, (h, w)))

    # ---------------------------------------------------------------- coordinates

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def is_physical(self) -> bool:
        """True when lengths are in a physical unit, as the geometry declares."""
        return bool(self.capability("physical_metric", False))

    @property
    def length_units(self) -> str:
        return str(self.capability("length_units", "pixel"))

    def latitudes(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Row latitudes in degrees. Raises for other grids rather than inventing them."""
        if not self.capability("has_latitude", False):
            raise GridError(
                "latitudes() is only defined for a grid whose geometry declares "
                "has_latitude; this grid is %r. A %s grid has no latitude, so cos(lat) area "
                "weighting does not apply (and uniform weights are already correct for it)."
                % (self.kind, self.kind)
            )
        idx = torch.arange(self.height, device=device, dtype=dtype)
        return self.lat0 + self.dy * idx

    def longitudes(self, device=None, dtype=torch.float64) -> torch.Tensor:
        if not self.capability("has_latitude", False):
            raise GridError("longitudes() is only defined for a grid whose geometry declares "
                            "has_latitude; this grid is %r." % (self.kind,))
        idx = torch.arange(self.width, device=device, dtype=dtype)
        return self.lon0 + self.dx * idx

    # ---------------------------------------------------------------- metrics

    def dy_metres(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Meridional spacing per row, shape ``(H,)``. Constant for all three builtins."""
        return self.geometry.dy_metres(self, device, dtype)

    def dx_metres(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Zonal spacing per row, shape ``(H,)``.

        This is the whole point of the module: on a lat/lon grid it varies as ``cos(lat)``,
        so it is returned per row and never collapsed to a scalar without saying so.
        """
        return self.geometry.dx_metres(self, device, dtype)

    def representative_dx_metres(self) -> float:
        """A single zonal spacing for FFT use, taken at the patch's mean latitude.

        A lat/lon patch is not conformal, so any Fourier analysis of it uses one
        representative metric. ``anisotropy()`` reports how much the true metric varies
        across the patch, so the size of that approximation is always visible rather than
        assumed small.
        """
        return float(self.geometry.representative_dx(self))

    def representative_dy_metres(self) -> float:
        return float(self.geometry.representative_dy(self))

    def cell_area(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Area of each cell, shape ``(H, W)``. m^2 for physical grids, 1.0 for pixel.

        For lat/lon the exact spherical cell area is used,
        ``R^2 * dlon_rad * (sin(lat_north) - sin(lat_south))``, not the ``cos(lat)``
        small-angle approximation - they differ negligibly at 0.25 degrees but by far more
        for coarse grids, and the exact form also stays correct in a polar row where
        cos(lat) -> 0.
        """
        return self.geometry.cell_area(self, device, dtype)

    def area_weights(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Cell areas normalised to sum to 1 - the weights for any domain statistic."""
        area = self.cell_area(device=device, dtype=dtype)
        total = torch.sum(area)
        if float(total) <= 0.0:
            raise GridError("total cell area is zero; grid=%r" % (self,))
        return area / total

    # ---------------------------------------------------------------- spectral

    def wavenumber_axes(
        self,
        units: str = "rad_per_m",
        device=None,
        dtype=torch.float64,
        shifted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Physical wavenumber axes ``(ky, kx)`` for a 2D FFT of this grid.

        ``units`` is one of ``rad_per_m``, ``cycles_per_m``, ``cycles_per_km``,
        ``rad_per_km``, or ``pixel`` (integer index, i.e. the pre-D13 convention, kept so
        that the old behaviour stays reproducible and testable rather than merely gone).
        """
        # dtype is passed *into* fftfreq, not applied afterwards. torch.fft.fftfreq
        # defaults to float32, and a trailing .to(float64) preserves an already-rounded
        # value: it capped the Parseval identity at a systematic 5.8e-8 (half of float32
        # epsilon) rather than machine precision. Same class of defect as D16.
        if units == "pixel":
            fy = torch.fft.fftfreq(self.height, d=1.0 / self.height, device=device, dtype=dtype)
            fx = torch.fft.fftfreq(self.width, d=1.0 / self.width, device=device, dtype=dtype)
        else:
            if units not in _WAVENUMBER_UNITS:
                raise GridError(
                    "unknown wavenumber units %r; expected one of %s or 'pixel'"
                    % (units, sorted(_WAVENUMBER_UNITS))
                )
            factor = _WAVENUMBER_UNITS[units][0]
            dy_m = self.representative_dy_metres()
            dx_m = self.representative_dx_metres()
            fy = torch.fft.fftfreq(self.height, d=dy_m, device=device, dtype=dtype) * factor
            fx = torch.fft.fftfreq(self.width, d=dx_m, device=device, dtype=dtype) * factor
        if shifted:
            fy = torch.fft.fftshift(fy)
            fx = torch.fft.fftshift(fx)
        return fy, fx

    def wavenumber_magnitude(
        self,
        units: str = "rad_per_m",
        device=None,
        dtype=torch.float64,
        shifted: bool = True,
    ) -> torch.Tensor:
        """``|k|`` on the 2D FFT grid, shape ``(H, W)``, in the requested units."""
        ky, kx = self.wavenumber_axes(units=units, device=device, dtype=dtype, shifted=shifted)
        gy, gx = torch.meshgrid(ky, kx, indexing="ij")
        return torch.sqrt(gy ** 2 + gx ** 2)

    def nyquist(self, units: str = "rad_per_m") -> Tuple[float, float]:
        """Nyquist wavenumber on each axis, in the requested units."""
        ky, kx = self.wavenumber_axes(units=units, shifted=False)
        return float(torch.max(torch.abs(ky))), float(torch.max(torch.abs(kx)))

    def isotropic_k_max(self, units: str = "rad_per_m") -> float:
        """Largest ``|k|`` fully resolved on *both* axes - the honest limit for a radial fit.

        Beyond ``min(nyquist_y, nyquist_x)`` a radial shell is only partly sampled, so an
        annulus average there is biased by the missing sectors regardless of binning.
        """
        ny, nx = self.nyquist(units=units)
        return min(ny, nx)

    def wavelength_units(self, units: str = "rad_per_m") -> str:
        """Label for the wavenumber units, for axis titles and provenance records.

        On a **pixel** grid the length unit is the pixel, so ``rad_per_m`` is reported as
        ``rad pixel^-1`` rather than ``rad m^-1``. Caught end-to-end: the API was returning
        ``k_units = "rad m^-1"`` for a field with no physical metric at all, which is
        exactly the mislabelling defect D13 is about - a plot axis in metres over data that
        never had metres. Kilometre variants have no meaning on a pixel grid and are
        refused rather than silently relabelled.
        """
        if units == "pixel":
            return "cycles per domain (pixel index)"
        label = _WAVENUMBER_UNITS[units][1]
        if not self.is_physical:
            if "km" in label:
                raise GridError(
                    "wavenumber units %r are kilometre-based but this is a pixel grid, "
                    "which has no physical length. Attach a GridSpec.cartesian or "
                    "GridSpec.latlon to the field, or request 'rad_per_m'/'cycles_per_m' "
                    "which are reported per pixel." % (units,)
                )
            return label.replace(" m^-1", " %s^-1" % self.length_units)
        return label

    # ---------------------------------------------------------------- diagnostics

    def anisotropy(self) -> Dict[str, Any]:
        """Report grid anisotropy and, for lat/lon, the spread of the zonal metric.

        Returned rather than raised: an anisotropic grid is perfectly analysable, it simply
        must not be radially binned in *pixel* space. Callers attach this to any spectral
        result, so the isotropy assumption always travels next to the number that depends
        on it.
        """
        dy_m = self.representative_dy_metres()
        dx_m = self.representative_dx_metres()
        aspect = dy_m / dx_m if dx_m else float("inf")

        info: Dict[str, Any] = {
            "kind": self.kind,
            "dy_metres": dy_m,
            "dx_metres": dx_m,
            "aspect_ratio": aspect,
            "is_isotropic": abs(aspect - 1.0) <= ANISOTROPY_WARN_THRESHOLD,
            "dx_variation_pct": 0.0,
            "warnings": [],
        }

        if not info["is_isotropic"]:
            info["warnings"].append(
                "Grid cells are anisotropic (dy/dx = %.4f). Radial binning in pixel space "
                "averages over ellipses in physical space and biases a fitted spectral "
                "slope; bin on physical |k| instead (units != 'pixel')." % aspect
            )

        self.geometry.anisotropy(self, info)
        return info

    def describe(self) -> str:
        """One-line human-readable summary, for provenance records and error messages."""
        return self.geometry.describe(self)

    def to_provenance(self) -> Dict[str, Any]:
        """JSON-serialisable record, so a grid round-trips through a lineage node (E5)."""
        record = {
            "kind": self.kind,
            "shape": list(self.shape),
            "dy": self.dy,
            "dx": self.dx,
            "length_units": self.length_units,
            "variable_units": self.variable_units,
            "description": self.describe(),
        }
        if self.params:
            record["params"] = {k: v for k, v in self.params}
        record.update(self.geometry.provenance(self))
        return record

    @classmethod
    def from_provenance(cls, record: Dict[str, Any]) -> "GridSpec":
        """Inverse of :meth:`to_provenance`."""
        kind = record["kind"]
        fields = dict(kind=kind, shape=tuple(record["shape"]),
                      dy=record["dy"], dx=record["dx"],
                      variable_units=record.get("variable_units"))
        if record.get("params"):
            fields["params"] = record["params"]
        fields.update(geometry_for(kind).rebuild(record))
        return cls(**fields)
