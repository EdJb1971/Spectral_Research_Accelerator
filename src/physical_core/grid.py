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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Optional, Tuple

import torch

# IUGG mean Earth radius (R_1, arithmetic mean of the WGS84 semi-axes), metres.
# ERA5 / IFS use 6371229.0; the difference is 3.5e-5 relative and is recorded rather
# than hidden, because a spectral slope fitted over two decades of k is not sensitive to
# it but an absolute gradient in K/m is quoted to more digits than that.
EARTH_RADIUS_M = 6371008.7714
IFS_EARTH_RADIUS_M = 6371229.0

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

    Three kinds:

    ``pixel``     - dimensionless index space. Spacing 1, area weights uniform. Honest
                    default; every derived quantity is then labelled ``per_pixel``.
    ``cartesian`` - uniform spacing in metres on a plane. Anisotropy allowed (dy != dx).
    ``latlon``    - regular longitude/latitude grid on a sphere of radius ``radius_m``.
                    ``dlat`` may be negative (ERA5 stores north-to-south); the sign is
                    preserved for coordinate reconstruction and taken as absolute for
                    metrics.
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

    # ---------------------------------------------------------------- construction

    def __post_init__(self) -> None:
        if self.kind not in ("pixel", "cartesian", "latlon"):
            raise GridError(
                "GridSpec.kind must be 'pixel', 'cartesian' or 'latlon', got %r. "
                "Use GridSpec.pixel(shape) if the field genuinely has no physical metric."
                % (self.kind,)
            )
        if len(self.shape) != 2:
            raise GridError("GridSpec.shape must be (H, W), got %r" % (self.shape,))
        if self.shape[0] < 1 or self.shape[1] < 1:
            raise GridError("GridSpec.shape must be positive, got %r" % (self.shape,))
        if self.kind == "cartesian" and (self.dy <= 0 or self.dx <= 0):
            raise GridError(
                "cartesian spacing must be positive metres, got dy=%r dx=%r" % (self.dy, self.dx)
            )
        if self.kind == "latlon":
            if self.dy == 0 or self.dx <= 0:
                raise GridError(
                    "latlon grid needs non-zero dlat and positive dlon in degrees, "
                    "got dlat=%r dlon=%r" % (self.dy, self.dx)
                )
            if self.lat0 is None or self.lon0 is None:
                raise GridError(
                    "latlon grid requires lat0 and lon0 (degrees) - the zonal metric "
                    "R*cos(lat)*dlon depends on latitude, so a lat/lon grid without a "
                    "latitude origin has no metric at all."
                )
            lat_end = self.lat0 + self.dy * (self.shape[0] - 1)
            for name, value in (("lat0", self.lat0), ("last latitude", lat_end)):
                if not (-90.0 - 1e-9 <= value <= 90.0 + 1e-9):
                    raise GridError(
                        "latlon grid runs off the sphere: %s = %.4f degrees, outside "
                        "[-90, 90]. Check the sign of dlat (ERA5 latitudes descend, so "
                        "dlat is negative)." % (name, value)
                    )
            if self.radius_m <= 0:
                raise GridError("radius_m must be positive, got %r" % (self.radius_m,))

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

        Recognises ``lat``/``lon`` (degrees, assumed regular) and ``y``/``x`` when
        ``metadata['grid_units']`` is a length unit. Anything else yields a pixel grid,
        because guessing a metric is worse than recording that we do not have one.
        """
        metadata = metadata or {}
        units = str(metadata.get("grid_units", "")).lower()
        var_units = metadata.get("units")

        if "lat" in coords and "lon" in coords:
            lat = coords["lat"].flatten().to(torch.float64)
            lon = coords["lon"].flatten().to(torch.float64)
            if lat.numel() >= 2 and lon.numel() >= 2:
                dlat = float((lat[-1] - lat[0]) / (lat.numel() - 1))
                dlon = float((lon[-1] - lon[0]) / (lon.numel() - 1))
                if dlat != 0.0 and dlon > 0.0:
                    return cls.latlon(shape, float(lat[0]), dlat, float(lon[0]), dlon,
                                      variable_units=var_units)

        if units in ("m", "metre", "metres", "meter", "meters",
                     "km", "kilometre", "kilometres", "kilometer", "kilometers"):
            scale = 1000.0 if units.startswith("k") else 1.0
            y = coords.get("y")
            x = coords.get("x")
            if y is not None and x is not None and y.numel() >= 2 and x.numel() >= 2:
                yy = y.flatten().to(torch.float64)
                xx = x.flatten().to(torch.float64)
                dy = abs(float((yy[-1] - yy[0]) / (yy.numel() - 1))) * scale
                dx = abs(float((xx[-1] - xx[0]) / (xx.numel() - 1))) * scale
                if dy > 0 and dx > 0:
                    return cls.cartesian(shape, dy, dx, variable_units=var_units)

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
        if self.kind == "pixel":
            return GridSpec.pixel(new_shape, variable_units=self.variable_units)

        def scale(old_n: int, new_n: int, spacing: float) -> float:
            if new_n == 1 or old_n == 1:
                return spacing
            return spacing * (old_n - 1) / (new_n - 1)

        dy = scale(self.height, new_shape[0], self.dy)
        dx = scale(self.width, new_shape[1], self.dx)
        if self.kind == "cartesian":
            return GridSpec.cartesian(new_shape, dy, dx, variable_units=self.variable_units)
        return GridSpec.latlon(new_shape, self.lat0, dy, self.lon0, dx,
                               radius_m=self.radius_m, variable_units=self.variable_units)

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
        if self.kind == "pixel":
            return GridSpec.pixel((h, w), variable_units=self.variable_units)
        if self.kind == "cartesian":
            return GridSpec.cartesian((h, w), self.dy, self.dx,
                                      variable_units=self.variable_units)
        return GridSpec.latlon((h, w), self.lat0 + self.dy * row_start, self.dy,
                               self.lon0 + self.dx * col_start, self.dx,
                               radius_m=self.radius_m, variable_units=self.variable_units)

    # ---------------------------------------------------------------- coordinates

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def is_physical(self) -> bool:
        """True when lengths are in metres, i.e. anything other than a pixel grid."""
        return self.kind != "pixel"

    @property
    def length_units(self) -> str:
        return "pixel" if self.kind == "pixel" else "m"

    def latitudes(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Row latitudes in degrees. Raises for other grids rather than inventing them."""
        if self.kind != "latlon":
            raise GridError(
                "latitudes() is only defined for a latlon grid; this grid is %r. "
                "A %s grid has no latitude, so cos(lat) area weighting does not apply "
                "(and uniform weights are already correct for it)." % (self.kind, self.kind)
            )
        idx = torch.arange(self.height, device=device, dtype=dtype)
        return self.lat0 + self.dy * idx

    def longitudes(self, device=None, dtype=torch.float64) -> torch.Tensor:
        if self.kind != "latlon":
            raise GridError("longitudes() is only defined for a latlon grid; this grid is %r."
                            % (self.kind,))
        idx = torch.arange(self.width, device=device, dtype=dtype)
        return self.lon0 + self.dx * idx

    # ---------------------------------------------------------------- metrics

    def dy_metres(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Meridional spacing per row, shape ``(H,)``. Constant for all three kinds."""
        if self.kind == "latlon":
            value = self.radius_m * math.radians(abs(self.dy))
        else:
            value = abs(self.dy)
        return torch.full((self.height,), float(value), device=device, dtype=dtype)

    def dx_metres(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Zonal spacing per row, shape ``(H,)``.

        This is the whole point of the module: on a lat/lon grid it varies as ``cos(lat)``,
        so it is returned per row and never collapsed to a scalar without saying so.
        """
        if self.kind == "latlon":
            lat_rad = torch.deg2rad(self.latitudes(device=device, dtype=dtype))
            return self.radius_m * torch.cos(lat_rad) * math.radians(self.dx)
        return torch.full((self.height,), float(abs(self.dx)), device=device, dtype=dtype)

    def representative_dx_metres(self) -> float:
        """A single zonal spacing for FFT use, taken at the patch's mean latitude.

        A lat/lon patch is not conformal, so any Fourier analysis of it uses one
        representative metric. ``anisotropy()`` reports how much the true metric varies
        across the patch, so the size of that approximation is always visible rather than
        assumed small.
        """
        if self.kind == "latlon":
            lats = self.latitudes()
            mean_lat = float(lats.mean())
            return self.radius_m * math.cos(math.radians(mean_lat)) * math.radians(self.dx)
        return float(abs(self.dx))

    def representative_dy_metres(self) -> float:
        if self.kind == "latlon":
            return self.radius_m * math.radians(abs(self.dy))
        return float(abs(self.dy))

    def cell_area(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Area of each cell, shape ``(H, W)``. m^2 for physical grids, 1.0 for pixel.

        For lat/lon the exact spherical cell area is used,
        ``R^2 * dlon_rad * (sin(lat_north) - sin(lat_south))``, not the ``cos(lat)``
        small-angle approximation - they differ negligibly at 0.25 degrees but by far more
        for coarse grids, and the exact form also stays correct in a polar row where
        cos(lat) -> 0.
        """
        if self.kind != "latlon":
            cell = self.representative_dy_metres() * self.representative_dx_metres()
            return torch.full(self.shape, float(cell), device=device, dtype=dtype)

        lats = self.latitudes(device=device, dtype=dtype)
        half = abs(self.dy) / 2.0
        north = torch.clamp(lats + half, max=90.0)
        south = torch.clamp(lats - half, min=-90.0)
        band = (self.radius_m ** 2) * math.radians(self.dx) * (
            torch.sin(torch.deg2rad(north)) - torch.sin(torch.deg2rad(south))
        )
        band = torch.abs(band)
        return band.unsqueeze(1).expand(self.height, self.width).contiguous()

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
        if self.kind == "pixel":
            if "km" in label:
                raise GridError(
                    "wavenumber units %r are kilometre-based but this is a pixel grid, "
                    "which has no physical length. Attach a GridSpec.cartesian or "
                    "GridSpec.latlon to the field, or request 'rad_per_m'/'cycles_per_m' "
                    "which are reported per pixel." % (units,)
                )
            return label.replace(" m^-1", " pixel^-1")
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

        if self.kind == "latlon":
            lats = self.latitudes()
            dxs = self.radius_m * torch.cos(torch.deg2rad(lats)) * math.radians(self.dx)
            lo, hi = float(torch.min(dxs)), float(torch.max(dxs))
            mid = 0.5 * (lo + hi)
            variation = 100.0 * (hi - lo) / mid if mid > 0 else float("inf")
            info["dx_variation_pct"] = variation
            info["dx_metres_min"] = lo
            info["dx_metres_max"] = hi
            info["lat_range_deg"] = (float(torch.min(lats)), float(torch.max(lats)))
            if variation > LATLON_DX_VARIATION_WARN_PCT:
                info["warnings"].append(
                    "The zonal metric varies by %.1f%% across this lat/lon patch "
                    "(%.0f m at one edge, %.0f m at the other). Fourier analysis uses a "
                    "single representative dx at the mean latitude, so spectral results "
                    "carry an error of that order. Regrid to an equal-area or conformal "
                    "projection, or use a narrower latitude band, before quoting a slope."
                    % (variation, lo, hi)
                )
        return info

    def describe(self) -> str:
        """One-line human-readable summary, for provenance records and error messages."""
        if self.kind == "pixel":
            return "pixel grid %dx%d (dimensionless; lengths are array indices)" % self.shape
        if self.kind == "cartesian":
            return "cartesian grid %dx%d, dy=%.1f m, dx=%.1f m" % (
                self.shape[0], self.shape[1], self.dy, self.dx)
        lats = self.latitudes()
        return (
            "latlon grid %dx%d, lat %.3f..%.3f deg (dlat=%.4f), lon %.3f..%.3f deg "
            "(dlon=%.4f), R=%.1f m, dy=%.0f m, dx=%.0f m at mean latitude"
            % (self.shape[0], self.shape[1], float(lats[0]), float(lats[-1]), self.dy,
               self.lon0, self.lon0 + self.dx * (self.width - 1), self.dx,
               self.radius_m, self.representative_dy_metres(),
               self.representative_dx_metres())
        )

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
        if self.kind == "latlon":
            record.update({"lat0": self.lat0, "lon0": self.lon0, "radius_m": self.radius_m})
        return record

    @classmethod
    def from_provenance(cls, record: Dict[str, Any]) -> "GridSpec":
        """Inverse of :meth:`to_provenance`."""
        kind = record["kind"]
        shape = tuple(record["shape"])
        if kind == "pixel":
            return cls.pixel(shape, variable_units=record.get("variable_units"))
        if kind == "cartesian":
            return cls.cartesian(shape, record["dy"], record["dx"],
                                 variable_units=record.get("variable_units"))
        return cls.latlon(shape, record["lat0"], record["dy"], record["lon0"], record["dx"],
                          radius_m=record.get("radius_m", EARTH_RADIUS_M),
                          variable_units=record.get("variable_units"))
