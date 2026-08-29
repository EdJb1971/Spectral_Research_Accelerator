"""Grid geometries as a registry, not an enum (TG1.2, standards E1/E13).

**What was wrong.** `GridSpec.kind` was a closed set of three strings — ``pixel``,
``cartesian``, ``latlon`` — and eleven methods across `grid.py` plus five call sites in
`operators.py` branched on it. That is exactly the shape standard E1 exists to forbid, and it
had the two failure modes E1 predicts:

*   **A fourth geometry cannot be added from outside.** An ocean model on a tripolar grid, a
    microscope stage in micrometres, a spectrogram whose axes are seconds and hertz — each
    needs `GridSpec.__post_init__` edited, then every branch found. There is no list of the
    branches; you find them by grepping for ``kind ==`` and hoping.
*   **The `else` branch is a silent default.** `laplacian` reads ``if kind == "latlon": ...
    else: cartesian_5point``. A geometry whose metric varies across the grid — which is the
    interesting case, and the reason ``latlon`` needed its own branch — would land in the
    Cartesian branch and produce a number. Not an error: a number, labelled
    ``value per m^2``. See `assert_uniform_metric`.

**What a geometry declares.** Registration carries capabilities, and operators ask about
those rather than about the name:

``physical_metric``   lengths are metres, so anything may be reported per metre.
``uniform_metric``    spacing is the same at every row. False for ``latlon``, where the zonal
                      metric goes as ``cos(lat)``.
``spherical``         the surface is a sphere, so the Laplacian is Laplace–Beltrami and not
                      the five-point stencil.
``has_latitude``      rows carry a latitude, so ``cos(lat)`` area weighting applies.
``length_units``      the label printed next to every derived quantity.

The three original kinds register here as the first three entries and behave exactly as
before — every branch was moved, none was rewritten. `test_geometry_registry.py` holds a
fourth geometry registered from the test module, which is the acceptance criterion, and it is
a deliberately awkward one (a non-uniform metric that is not a sphere) because a fourth
geometry that is merely ``cartesian`` with a different name proves nothing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from src.core.axes import find_coordinate
from src.core.registry import Registry

#: IUGG mean Earth radius (R_1), metres. Imported by `grid.py`, which re-exports it; it lives
#: here because the spherical geometry is the thing that needs it.
EARTH_RADIUS_M = 6371008.7714
IFS_EARTH_RADIUS_M = 6371229.0


class Geometry:
    """One grid geometry. Subclass, then register.

    Every method takes the `GridSpec` as an argument rather than being a method *on* it: a
    geometry is a strategy, and keeping it stateless means a registered geometry is a
    singleton with no per-grid state to get out of step.

    The derivation methods (`resampled`, `subset`, `from_coords`, `rebuild`) return a **dict
    of GridSpec fields**, not a `GridSpec`. That is deliberate: it keeps this module free of
    any import of `grid.py`, so the dependency runs one way only and a plugin geometry cannot
    create an import cycle by existing.
    """

    #: Registry name. Set by the subclass; must match the `name` passed to `register`.
    name: str = ""

    # ------------------------------------------------------------------ validation

    def validate(self, spec: Any) -> None:
        """Raise `GridError` if this spec is impossible for this geometry."""

    # ------------------------------------------------------------------ metrics

    def representative_dy(self, spec: Any) -> float:
        """A single meridional spacing, in this geometry's length units."""
        return float(abs(spec.dy))

    def representative_dx(self, spec: Any) -> float:
        """A single zonal spacing, in this geometry's length units.

        For a geometry with ``uniform_metric=False`` this is an *approximation* whose size
        must be reported by `anisotropy`, not assumed small.
        """
        return float(abs(spec.dx))

    def dy_metres(self, spec: Any, device: Any, dtype: Any) -> torch.Tensor:
        """Meridional spacing per row, shape ``(H,)``."""
        return torch.full((spec.height,), self.representative_dy(spec),
                          device=device, dtype=dtype)

    def dx_metres(self, spec: Any, device: Any, dtype: Any) -> torch.Tensor:
        """Zonal spacing per row, shape ``(H,)``. Per row, never collapsed silently."""
        return torch.full((spec.height,), self.representative_dx(spec),
                          device=device, dtype=dtype)

    def cell_area(self, spec: Any, device: Any, dtype: Any) -> torch.Tensor:
        """Area of every cell, shape ``(H, W)``."""
        cell = self.representative_dy(spec) * self.representative_dx(spec)
        return torch.full(spec.shape, float(cell), device=device, dtype=dtype)

    def north_sign(self, spec: Any) -> float:
        """+1 if increasing row index means increasing north, -1 if it means decreasing.

        The default states the plane convention explicitly rather than leaving it implied:
        rows increase downward in array order and are treated as +y.
        """
        return 1.0

    # ------------------------------------------------------------------ derivation

    def resampled(self, spec: Any, new_shape: Tuple[int, int],
                  dy: float, dx: float) -> Dict[str, Any]:
        """Field changes after interpolation to ``new_shape``.

        ``dy``/``dx`` are the endpoint-preserving rescaled spacings, computed once in
        `GridSpec.resampled` because that arithmetic is a property of ``align_corners=True``
        interpolation rather than of any geometry.
        """
        return {"shape": new_shape, "dy": dy, "dx": dx}

    def subset(self, spec: Any, row_start: int, col_start: int,
               shape: Tuple[int, int]) -> Dict[str, Any]:
        """Field changes after a contiguous crop. Spacing holds; an origin may move."""
        return {"shape": shape}

    # ------------------------------------------------------------------ description

    def describe(self, spec: Any) -> str:
        return "%s grid %dx%d, dy=%s, dx=%s" % (self.name, spec.shape[0], spec.shape[1],
                                                spec.dy, spec.dx)

    def provenance(self, spec: Any) -> Dict[str, Any]:
        """Extra keys this geometry needs in order to round-trip. Merged into the record."""
        return {}

    def rebuild(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Constructor fields from a provenance record. Inverse of `provenance`."""
        return {}

    def anisotropy(self, spec: Any, info: Dict[str, Any]) -> None:
        """Augment the anisotropy report in place with anything specific to this geometry."""

    # ------------------------------------------------------------------ recognition

    def from_coords(self, coords: Dict[str, torch.Tensor], shape: Tuple[int, int],
                    metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Constructor fields if this geometry recognises the coordinates, else ``None``.

        Consulted in ascending `coord_priority`. Returning ``None`` is the normal answer;
        guessing a metric is worse than recording that we do not have one.
        """
        return None

    #: Order in which `GridSpec.from_coords` consults geometries. Lower goes first; ``pixel``
    #: sits at the end as the honest fallback.
    coord_priority: int = 100


#: The registry. Capabilities live on the `Entry`, so `GEOMETRIES.with_capability("spherical")`
#: answers "which geometries curve?" without anyone maintaining a list.
GEOMETRIES: Registry[Geometry] = Registry("grid geometry")


def geometry_for(kind: str) -> Geometry:
    """The registered geometry, or `UnknownNameError` listing what is available."""
    return GEOMETRIES.get(kind)


def capability(kind: str, key: str, default: Any = None) -> Any:
    """One declared capability of a registered geometry."""
    return GEOMETRIES.entry(kind).capabilities.get(key, default)


def recognisers() -> Tuple[Geometry, ...]:
    """Registered geometries in the order `from_coords` should consult them."""
    return tuple(sorted((e.value for e in GEOMETRIES.entries()
                         if e.capabilities.get("grid_compatible", True)),
                        key=lambda g: (g.coord_priority, g.name)))


# ===================================================================== pixel

class PixelGeometry(Geometry):
    """Dimensionless index space. Spacing 1, uniform weights, honest default."""

    name = "pixel"
    coord_priority = 1000

    def resampled(self, spec, new_shape, dy, dx):
        # Not the rescaled spacing: a pixel grid's spacing is 1 by definition, and carrying a
        # fractional "pixel spacing" through a resample would make `length_units == "pixel"`
        # a lie about a quantity that is no longer an index.
        return {"shape": new_shape, "dy": 1.0, "dx": 1.0}

    def subset(self, spec, row_start, col_start, shape):
        return {"shape": shape, "dy": 1.0, "dx": 1.0}

    def describe(self, spec):
        return "pixel grid %dx%d (dimensionless; lengths are array indices)" % spec.shape

    def from_coords(self, coords, shape, metadata):
        return {"kind": "pixel", "shape": shape, "dy": 1.0, "dx": 1.0}


GEOMETRIES.add(
    "pixel", PixelGeometry(),
    description="Dimensionless array index space; lengths are pixels.",
    capabilities={"physical_metric": False, "uniform_metric": True, "spherical": False,
                  "has_latitude": False, "length_units": "pixel"},
    tags=["builtin"])


# ===================================================================== cartesian

class CartesianGeometry(Geometry):
    """Uniform spacing in metres on a plane. Anisotropy allowed (dy != dx)."""

    name = "cartesian"
    coord_priority = 20

    #: Length units `from_coords` will accept from ``metadata['grid_units']``, and the factor
    #: that converts each to metres.
    UNIT_SCALE: Dict[str, float] = {
        "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
        "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
        "kilometer": 1000.0, "kilometers": 1000.0,
    }

    def validate(self, spec):
        from src.physical_core.grid import GridError
        if spec.dy <= 0 or spec.dx <= 0:
            raise GridError(
                "cartesian spacing must be positive metres, got dy=%r dx=%r"
                % (spec.dy, spec.dx))

    def describe(self, spec):
        return "cartesian grid %dx%d, dy=%.1f m, dx=%.1f m" % (
            spec.shape[0], spec.shape[1], spec.dy, spec.dx)

    def from_coords(self, coords, shape, metadata):
        units = str(metadata.get("grid_units", "")).lower()
        if units not in self.UNIT_SCALE:
            return None
        scale = self.UNIT_SCALE[units]
        y_name = find_coordinate(coords, "space", 0)
        x_name = find_coordinate(coords, "space", 1)
        y = coords.get(y_name) if y_name else None
        x = coords.get(x_name) if x_name else None
        if y is None or x is None or y.numel() < 2 or x.numel() < 2:
            return None
        yy = y.flatten().to(torch.float64)
        xx = x.flatten().to(torch.float64)
        dy = abs(float((yy[-1] - yy[0]) / (yy.numel() - 1))) * scale
        dx = abs(float((xx[-1] - xx[0]) / (xx.numel() - 1))) * scale
        if dy <= 0 or dx <= 0:
            return None
        return {"kind": "cartesian", "shape": shape, "dy": dy, "dx": dx}


GEOMETRIES.add(
    "cartesian", CartesianGeometry(),
    description="Uniform planar grid with spacing in metres.",
    capabilities={"physical_metric": True, "uniform_metric": True, "spherical": False,
                  "has_latitude": False, "length_units": "m"},
    tags=["builtin"])


# ===================================================================== latlon

class LatLonGeometry(Geometry):
    """Regular lon/lat grid on a sphere.

    ``dy`` is dlat in degrees and may be negative (ERA5 stores north-to-south); the sign is
    preserved for coordinate reconstruction and taken as absolute for metrics. This is the
    only builtin with ``uniform_metric=False``, and everything awkward about the module
    traces to that one fact.
    """

    name = "latlon"
    coord_priority = 10

    def validate(self, spec):
        from src.physical_core.grid import GridError
        if spec.dy == 0 or spec.dx <= 0:
            raise GridError(
                "latlon grid needs non-zero dlat and positive dlon in degrees, "
                "got dlat=%r dlon=%r" % (spec.dy, spec.dx))
        if spec.lat0 is None or spec.lon0 is None:
            raise GridError(
                "latlon grid requires lat0 and lon0 (degrees) - the zonal metric "
                "R*cos(lat)*dlon depends on latitude, so a lat/lon grid without a "
                "latitude origin has no metric at all.")
        lat_end = spec.lat0 + spec.dy * (spec.shape[0] - 1)
        for label, value in (("lat0", spec.lat0), ("last latitude", lat_end)):
            if not (-90.0 - 1e-9 <= value <= 90.0 + 1e-9):
                raise GridError(
                    "latlon grid runs off the sphere: %s = %.4f degrees, outside "
                    "[-90, 90]. Check the sign of dlat (ERA5 latitudes descend, so "
                    "dlat is negative)." % (label, value))
        if spec.radius_m <= 0:
            raise GridError("radius_m must be positive, got %r" % (spec.radius_m,))

    # ------------------------------------------------------------------ metrics

    def representative_dy(self, spec):
        return spec.radius_m * math.radians(abs(spec.dy))

    def representative_dx(self, spec):
        lats = spec.latitudes()
        mean_lat = float(lats.mean())
        return spec.radius_m * math.cos(math.radians(mean_lat)) * math.radians(spec.dx)

    def dx_metres(self, spec, device, dtype):
        lat_rad = torch.deg2rad(spec.latitudes(device=device, dtype=dtype))
        return spec.radius_m * torch.cos(lat_rad) * math.radians(spec.dx)

    def cell_area(self, spec, device, dtype):
        # Exact spherical band area, R^2 dlon (sin(lat_n) - sin(lat_s)), not the cos(lat)
        # small-angle form: they differ negligibly at 0.25 degrees but by far more on a
        # coarse grid, and the exact form stays correct in a polar row where cos(lat) -> 0.
        lats = spec.latitudes(device=device, dtype=dtype)
        half = abs(spec.dy) / 2.0
        north = torch.clamp(lats + half, max=90.0)
        south = torch.clamp(lats - half, min=-90.0)
        band = (spec.radius_m ** 2) * math.radians(spec.dx) * (
            torch.sin(torch.deg2rad(north)) - torch.sin(torch.deg2rad(south)))
        band = torch.abs(band)
        return band.unsqueeze(1).expand(spec.height, spec.width).contiguous()

    def north_sign(self, spec):
        return 1.0 if spec.dy > 0 else -1.0

    # ------------------------------------------------------------------ derivation

    def subset(self, spec, row_start, col_start, shape):
        # The origin moves, which changes the mean latitude and therefore the representative
        # zonal metric. That is why a crop must carry a new GridSpec rather than its parent's.
        return {"shape": shape,
                "lat0": spec.lat0 + spec.dy * row_start,
                "lon0": spec.lon0 + spec.dx * col_start}

    # ------------------------------------------------------------------ description

    def describe(self, spec):
        lats = spec.latitudes()
        return (
            "latlon grid %dx%d, lat %.3f..%.3f deg (dlat=%.4f), lon %.3f..%.3f deg "
            "(dlon=%.4f), R=%.1f m, dy=%.0f m, dx=%.0f m at mean latitude"
            % (spec.shape[0], spec.shape[1], float(lats[0]), float(lats[-1]), spec.dy,
               spec.lon0, spec.lon0 + spec.dx * (spec.width - 1), spec.dx,
               spec.radius_m, self.representative_dy(spec), self.representative_dx(spec)))

    def provenance(self, spec):
        return {"lat0": spec.lat0, "lon0": spec.lon0, "radius_m": spec.radius_m}

    def rebuild(self, record):
        return {"lat0": record["lat0"], "lon0": record["lon0"],
                "radius_m": record.get("radius_m", EARTH_RADIUS_M)}

    def anisotropy(self, spec, info):
        from src.physical_core.grid import LATLON_DX_VARIATION_WARN_PCT
        lats = spec.latitudes()
        dxs = spec.radius_m * torch.cos(torch.deg2rad(lats)) * math.radians(spec.dx)
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
                % (variation, lo, hi))

    # ------------------------------------------------------------------ recognition

    def from_coords(self, coords, shape, metadata):
        lat_name = find_coordinate(coords, "space", 0)
        lon_name = find_coordinate(coords, "space", 1)
        # A spatial pair resolved by *position* is not evidence of a sphere: every 2D field
        # has a row axis and a column axis. Only a recognised or declared latitude counts.
        if lat_name not in ("lat", "latitude", "nlat") \
                or lon_name not in ("lon", "longitude", "nlon"):
            return None
        lat = coords[lat_name].flatten().to(torch.float64)
        lon = coords[lon_name].flatten().to(torch.float64)
        if lat.numel() < 2 or lon.numel() < 2:
            return None
        dlat = float((lat[-1] - lat[0]) / (lat.numel() - 1))
        dlon = float((lon[-1] - lon[0]) / (lon.numel() - 1))
        if dlat == 0.0 or dlon <= 0.0:
            return None
        return {"kind": "latlon", "shape": shape, "dy": dlat, "dx": dlon,
                "lat0": float(lat[0]), "lon0": float(lon[0])}


GEOMETRIES.add(
    "latlon", LatLonGeometry(),
    description="Regular longitude/latitude grid on a sphere of radius radius_m.",
    capabilities={"physical_metric": True, "uniform_metric": False, "spherical": True,
                  "has_latitude": True, "length_units": "m"},
    tags=["builtin"])
