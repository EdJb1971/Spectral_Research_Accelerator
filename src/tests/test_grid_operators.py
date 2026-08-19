"""Tests for grid geometry, metric-aware operators and physical-wavenumber spectra.

Covers roadmap T3.5.13 (defect D13), the newly-found defect D26 (spectrum convention off
by one exponent), and the vectorised-binning half of T3.5.20 (defect D17).

Written to rule R8: property tests and analytic ground truth, not inverse-of-itself checks.
Several tests are deliberately *head-to-head* - they assert both that the new code is right
and that the old behaviour is wrong on the same input, so a regression cannot pass by
quietly reverting the fix.
"""

import math

import numpy as np
import pytest
import torch

from src.physical_core.grid import (
    GridSpec,
    GridError,
    EARTH_RADIUS_M,
)
from src.physical_core import operators
from src.physical_core.field import PhysicalField
from src.analysis_engine import spectra
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine


# ============================================================== helpers

def latlon_field(shape, lat0, dlat, lon0, dlon, fn):
    """Build a field from an analytic function of (lat_rad, lon_rad)."""
    g = GridSpec.latlon(shape, lat0, dlat, lon0, dlon)
    lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape).contiguous()
    lon = torch.deg2rad(g.longitudes()).unsqueeze(0).expand(shape).contiguous()
    return fn(lat, lon), g


def power_law_field(n, beta_energy, seed=0, dy=31000.0, dx=31000.0):
    """Synthetic field whose 1D energy spectrum is exactly E(k) ~ k^-beta_energy.

    Built in the Fourier domain on the *physical* wavenumber grid, so the known answer is
    known in physical units and does not depend on the grid being square.
    """
    g = GridSpec.cartesian((n, n), dy, dx)
    k = g.wavenumber_magnitude("rad_per_m", shifted=False, dtype=torch.float64).clone()
    k[0, 0] = 1.0
    amp = k ** (-(beta_energy + 1.0) / 2.0)   # S = E / (2 pi k)  =>  S ~ k^-(beta+1)
    amp[0, 0] = 0.0
    gen = torch.Generator().manual_seed(seed)
    phase = torch.rand(n, n, generator=gen, dtype=torch.float64) * 2 * math.pi
    field = torch.fft.ifft2(amp * torch.exp(1j * phase)).real
    return field, g


# ============================================================== GridSpec geometry

def test_global_cell_areas_sum_to_the_sphere():
    """The single strongest check on the area metric: it must reproduce 4 pi R^2."""
    g = GridSpec.latlon((721, 1440), 90.0, -0.25, 0.0, 0.25)
    total = float(g.cell_area().sum())
    exact = 4.0 * math.pi * EARTH_RADIUS_M ** 2
    assert abs(total - exact) / exact < 1e-12


def test_area_weights_sum_to_one():
    for g in (GridSpec.latlon((181, 360), 90.0, -1.0, 0.0, 1.0),
              GridSpec.cartesian((32, 48), 25000.0, 31000.0),
              GridSpec.pixel((17, 23))):
        assert abs(float(g.area_weights().sum()) - 1.0) < 1e-12


def test_exact_spherical_area_differs_from_the_cos_lat_approximation():
    """Justifies using sin(lat) differences rather than cos(lat) * dlat.

    At a coarse resolution the small-angle approximation is measurably wrong, and it is
    wrong worst exactly where cell areas are smallest and the weighting matters most.
    """
    g = GridSpec.latlon((19, 36), 90.0, -10.0, 0.0, 10.0)
    exact = g.cell_area()[:, 0]
    lat = g.latitudes()
    approx = (EARTH_RADIUS_M ** 2) * math.radians(10.0) * math.radians(10.0) * torch.cos(
        torch.deg2rad(lat))
    rel = torch.abs(exact - approx) / exact.clamp(min=1e-30)
    assert float(rel.max()) > 0.1, "approximation should be visibly wrong at 10 degrees"
    # ... and the exact form stays finite at the pole, where cos(lat) -> 0.
    assert float(exact[0]) > 0.0
    assert torch.isfinite(exact).all()


def test_zonal_metric_follows_cos_lat():
    g = GridSpec.latlon((41, 8), 80.0, -2.0, 0.0, 0.25)
    dx = g.dx_metres()
    lat = g.latitudes()
    predicted = EARTH_RADIUS_M * torch.cos(torch.deg2rad(lat)) * math.radians(0.25)
    assert torch.allclose(dx, predicted, rtol=1e-12)
    # The spread across this band is the whole point of D13.
    assert float(dx.max() / dx.min()) > 5.0


def test_meridional_metric_is_constant_and_correct():
    g = GridSpec.latlon((10, 10), 45.0, -0.25, 0.0, 0.25)
    dy = g.dy_metres()
    assert float(dy.max() - dy.min()) == 0.0
    assert abs(float(dy[0]) - EARTH_RADIUS_M * math.radians(0.25)) < 1e-9


def test_era5_patch_at_60n_is_strongly_anisotropic():
    """The concrete case that motivates the whole module."""
    g = GridSpec.latlon((32, 32), 60.0, -0.25, 0.0, 0.25)
    iso = g.anisotropy()
    assert not iso["is_isotropic"]
    assert 1.7 < iso["aspect_ratio"] < 1.9
    assert iso["dx_variation_pct"] > 5.0
    assert any("anisotropic" in w for w in iso["warnings"])
    assert any("zonal metric varies" in w for w in iso["warnings"])


def test_isotropic_grid_reports_no_warnings():
    """Positive control: the anisotropy detector must not fire on everything."""
    g = GridSpec.cartesian((64, 64), 31000.0, 31000.0)
    iso = g.anisotropy()
    assert iso["is_isotropic"]
    assert iso["warnings"] == []


def test_wavenumber_axes_are_double_precision():
    """Regression guard: torch.fft.fftfreq defaults to float32.

    A trailing `.to(float64)` preserves an already-rounded value and capped the Parseval
    identity at a systematic 5.8e-8 - half of float32 epsilon - rather than machine
    precision. Same class of defect as D16.
    """
    g = GridSpec.cartesian((256, 256), 31000.0)
    ky, kx = g.wavenumber_axes("rad_per_m")
    assert ky.dtype == torch.float64 and kx.dtype == torch.float64

    # Read the spacing off the *unshifted* axis, where kx[1] is the spacing itself.
    # Differencing two shifted entries near -1e-4 cancels ~2 significant digits and would
    # make this a test of cancellation rather than of precision.
    _, kx_u = g.wavenumber_axes("rad_per_m", shifted=False)
    expected = 2.0 * math.pi / (256 * 31000.0)
    assert abs(float(kx_u[1]) - expected) / expected < 1e-15

    # And the identity that actually depends on it: Parseval to machine precision.
    gen = torch.Generator().manual_seed(99)
    f = torch.randn(256, 256, generator=gen, dtype=torch.float64)
    fft = torch.fft.fft2(f - f.mean())
    S = (fft.real ** 2 + fft.imag ** 2) * (g.dy * g.dx) / (65536 * (2 * math.pi) ** 2)
    dk2 = float(kx_u[1]) * float(g.wavenumber_axes("rad_per_m", shifted=False)[0][1])
    ratio = float(S.sum()) * dk2 / float(torch.mean((f - f.mean()) ** 2))
    assert abs(ratio - 1.0) < 1e-12, (
        "float32 fftfreq previously capped this at 5.8e-8; got %.3e" % abs(ratio - 1.0))


def test_resample_preserves_the_coordinate_span():
    """align_corners=True keeps endpoints, so spacing scales by (n-1)/(m-1), not n/m."""
    g = GridSpec.latlon((32, 64), 60.0, -0.25, 0.0, 0.25)
    span_lat = g.dy * (g.height - 1)
    span_lon = g.dx * (g.width - 1)
    r = g.resampled((16, 32))
    assert abs(r.dy * (r.height - 1) - span_lat) < 1e-12
    assert abs(r.dx * (r.width - 1) - span_lon) < 1e-12
    # The naive n/m rule would give a different answer; make sure we are not doing that.
    assert abs(r.dy - g.dy * 32 / 16) > 1e-6


def test_subset_moves_the_origin_and_changes_the_zonal_metric():
    g = GridSpec.latlon((64, 64), 80.0, -0.25, 0.0, 0.25)
    south = g.subset(row_start=48)
    assert south.height == 16
    assert abs(south.lat0 - (80.0 - 0.25 * 48)) < 1e-12
    # A crop at a different latitude has a genuinely different metric - which is why a
    # crop must carry its own GridSpec rather than reuse the parent's.
    assert south.representative_dx_metres() > g.representative_dx_metres()


def test_subset_rejects_an_empty_crop():
    g = GridSpec.pixel((8, 8))
    with pytest.raises(GridError, match="subset is empty"):
        g.subset(row_start=5, row_stop=5)


def test_grid_provenance_round_trips():
    for g in (GridSpec.latlon((32, 32), 60.0, -0.25, 10.0, 0.25, variable_units="K"),
              GridSpec.cartesian((8, 16), 1000.0, 2000.0),
              GridSpec.pixel((4, 4))):
        assert GridSpec.from_provenance(g.to_provenance()) == g


def test_latlon_errors_name_the_fix():
    """Standard E6: an error must say what to do, not merely that something is wrong."""
    with pytest.raises(GridError, match="dlat is negative"):
        GridSpec.latlon((100, 10), 89.0, 0.25, 0.0, 0.25)      # runs off the north pole
    with pytest.raises(GridError, match="latitude origin"):
        GridSpec(kind="latlon", shape=(4, 4), dy=-0.25, dx=0.25)
    with pytest.raises(GridError, match="GridSpec.pixel"):
        GridSpec(kind="mercator", shape=(4, 4))


def test_latitudes_refuses_a_non_spherical_grid_with_an_explanation():
    g = GridSpec.cartesian((4, 4), 1000.0)
    with pytest.raises(GridError, match="uniform weights are already correct"):
        g.latitudes()


def test_from_coords_infers_or_falls_back_honestly():
    lat = torch.linspace(60.0, 52.25, 32)
    lon = torch.linspace(0.0, 15.75, 64)
    assert GridSpec.from_coords({"lat": lat, "lon": lon}, (32, 64)).kind == "latlon"

    y = torch.linspace(0.0, 310.0, 32)
    x = torch.linspace(0.0, 630.0, 64)
    cart = GridSpec.from_coords({"y": y, "x": x}, (32, 64), {"grid_units": "km"})
    assert cart.kind == "cartesian"
    assert abs(cart.dy - 10000.0) < 1e-6

    # Unknown coordinates must fall back to pixel rather than guess a metric.
    assert GridSpec.from_coords({"x": x, "y": y}, (32, 64)).kind == "pixel"


# ============================================================== gradient

@pytest.mark.parametrize("lat0", [4.0, 34.0, 64.0, 84.0])
def test_zonal_gradient_matches_the_analytic_answer_at_every_latitude(lat0):
    """f = sin(lon)  =>  df/d(east) = cos(lon) / (R cos(lat)), exactly.

    This is the D13 acceptance criterion: correct in physical units at multiple latitudes.
    """
    shape = (48, 48)
    f, g = latlon_field(shape, lat0, -0.25, 0.0, 0.25, lambda la, lo: torch.sin(lo))
    lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape)
    lon = torch.deg2rad(g.longitudes()).unsqueeze(0).expand(shape)
    exact = torch.cos(lon) / (EARTH_RADIUS_M * torch.cos(lat))

    out = operators.gradient(f, g)
    m = out["valid_mask"]
    rel = torch.abs(out["d_deast"] - exact)[m] / torch.abs(exact)[m]
    assert float(rel.max()) < 1e-5
    # A purely zonal field has no meridional gradient.
    assert float(torch.abs(out["d_dnorth"])[m].max()) < 1e-12
    assert out["units"] == "value per m"
    assert out["is_physical"] is True


def test_unit_spacing_fails_the_same_test_and_fails_differently_by_latitude():
    """Head-to-head with the pre-D13 behaviour, so the fix cannot silently regress.

    The important part is not that unit spacing is wrong by a large factor - that alone
    could be absorbed by any downstream normalisation. It is that the error *ratio between
    two latitudes* is not 1, so no single rescaling can repair it.
    """
    shape = (48, 48)
    errors = {}
    for lat0 in (4.0, 64.0):
        f, g = latlon_field(shape, lat0, -0.25, 0.0, 0.25, lambda la, lo: torch.sin(lo))
        lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape)
        lon = torch.deg2rad(g.longitudes()).unsqueeze(0).expand(shape)
        exact = torch.cos(lon) / (EARTH_RADIUS_M * torch.cos(lat))

        _, naive_dx = torch.gradient(f.to(torch.float64), edge_order=1)
        m = operators._interior_mask(shape)
        errors[lat0] = float((torch.abs(naive_dx - exact)[m] / torch.abs(exact)[m]).max())

    assert errors[4.0] > 1e3 and errors[64.0] > 1e3
    ratio = errors[4.0] / errors[64.0]
    assert ratio > 1.4, (
        "unit-spacing error must vary with latitude, otherwise it would be a mere "
        "rescale; measured ratio %.3f" % ratio
    )


def test_meridional_gradient_matches_the_analytic_answer():
    """f = sin(lat)  =>  df/d(north) = cos(lat) / R."""
    shape = (48, 48)
    f, g = latlon_field(shape, 34.0, -0.25, 0.0, 0.25, lambda la, lo: torch.sin(la))
    lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape)
    exact = torch.cos(lat) / EARTH_RADIUS_M
    out = operators.gradient(f, g)
    m = out["valid_mask"]
    rel = torch.abs(out["d_dnorth"] - exact)[m] / torch.abs(exact)[m]
    assert float(rel.max()) < 1e-5


def test_gradient_error_is_second_order_in_the_grid_spacing():
    """Stronger than any fixed tolerance: the residual must be *truncation*, not a bug.

    The remaining error of the zonal gradient is 3.17e-6 at dlat = 0.25 degrees, which is
    exactly h^2/6 for the central difference of a sinusoid. A wrong metric would leave a
    residual that does not shrink with h at all, so asserting the convergence *order*
    distinguishes "correct scheme, finite resolution" from "subtly wrong scheme" in a way
    that choosing a looser tolerance never could.
    """
    errors = []
    for refine in (1, 2, 4):
        n = 48 * refine
        step = 0.5 / refine
        shape = (n, n)
        f, g = latlon_field(shape, 34.0, -step, 0.0, step, lambda la, lo: torch.sin(lo))
        lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape)
        lon = torch.deg2rad(g.longitudes()).unsqueeze(0).expand(shape)
        exact = torch.cos(lon) / (EARTH_RADIUS_M * torch.cos(lat))
        out = operators.gradient(f, g)
        m = out["valid_mask"]
        errors.append(float((torch.abs(out["d_deast"] - exact)[m] / torch.abs(exact)[m]).max()))

    for coarse, fine in zip(errors, errors[1:]):
        ratio = coarse / fine
        assert 3.5 < ratio < 4.5, (
            "halving the spacing must cut the error ~4x for a second-order scheme; "
            "errors %r gave ratio %.2f" % (errors, ratio))


def test_north_sign_follows_the_sign_of_dlat():
    """ERA5 latitudes descend with row index. Getting this wrong flips every flow direction."""
    shape = (32, 32)
    f_down, g_down = latlon_field(shape, 40.0, -0.25, 0.0, 0.25, lambda la, lo: torch.sin(la))
    f_up, g_up = latlon_field(shape, 32.25, +0.25, 0.0, 0.25, lambda la, lo: torch.sin(la))
    d = operators.gradient(f_down, g_down)
    u = operators.gradient(f_up, g_up)
    # Same physical field, opposite storage order: d_dnorth must agree in sign.
    assert float(d["d_dnorth"][d["valid_mask"]].mean()) > 0
    assert float(u["d_dnorth"][u["valid_mask"]].mean()) > 0
    # ...while the raw row derivative does not.
    assert float(d["d_drow"][d["valid_mask"]].mean()) < 0
    assert float(u["d_drow"][u["valid_mask"]].mean()) > 0


def test_edge_ring_is_first_order_and_labelled_as_such():
    """The edge ring is returned, not silently dropped - so its error is measured here."""
    shape = (48, 48)
    f, g = latlon_field(shape, 34.0, -0.25, 0.0, 0.25, lambda la, lo: torch.sin(lo))
    lat = torch.deg2rad(g.latitudes()).unsqueeze(1).expand(shape)
    lon = torch.deg2rad(g.longitudes()).unsqueeze(0).expand(shape)
    exact = torch.cos(lon) / (EARTH_RADIUS_M * torch.cos(lat))
    out = operators.gradient(f, g)
    m = out["valid_mask"]
    rel = torch.abs(out["d_deast"] - exact) / torch.abs(exact)
    interior_err = float(rel[m].max())
    edge_err = float(rel[~m].max())
    assert edge_err > 20 * interior_err, (
        "edge ring should be visibly worse (first order); interior %.2e edge %.2e"
        % (interior_err, edge_err))
    assert m[0, 0].item() is False and m[24, 24].item() is True


def test_pixel_grid_labels_its_output_as_per_pixel():
    out = operators.gradient(torch.randn(16, 16))
    assert out["units"] == "value per pixel"
    assert out["is_physical"] is False


def test_polar_row_gives_nan_not_a_silent_huge_number():
    """cos(lat) -> 0 means the zonal derivative is genuinely undefined at the pole."""
    g = GridSpec.latlon((4, 8), 90.0, -0.25, 0.0, 0.25)
    out = operators.gradient(torch.randn(4, 8, dtype=torch.float64), g)
    assert torch.isnan(out["d_dcol"][0]).all()
    assert torch.isfinite(out["d_dcol"][2]).all()


def test_operator_rejects_a_grid_that_does_not_match_the_field():
    g = GridSpec.latlon((32, 32), 60.0, -0.25, 0.0, 0.25)
    with pytest.raises(GridError, match="build a new GridSpec"):
        operators.gradient(torch.randn(16, 16), g)


# ============================================================== Laplacian

@pytest.mark.parametrize("lat0", [34.0, 64.0])
def test_spherical_laplacian_matches_harmonic_eigenvalue(lat0):
    """Laplace-Beltrami eigenvalue is exactly -l(l+1)/R^2. Here l = 1, f = sin(lat)."""
    shape = (64, 64)
    f, g = latlon_field(shape, lat0, -0.125, 0.0, 0.125, lambda la, lo: torch.sin(la))
    exact = -2.0 / EARTH_RADIUS_M ** 2 * f
    out = operators.laplacian(f, g)
    m = out["valid_mask"]
    rel = torch.abs(out["laplacian"] - exact)[m] / torch.abs(exact)[m]
    assert float(rel.max()) < 1e-5
    assert out["operator"] == "laplace_beltrami_sphere"


def test_spherical_laplacian_on_a_sectoral_harmonic_exercises_both_terms():
    """Y_2^2 ~ cos^2(lat) cos(2 lon): eigenvalue -6/R^2, and neither term is zero."""
    shape = (64, 64)
    f, g = latlon_field(shape, 34.0, -0.125, 0.0, 0.125,
                        lambda la, lo: torch.cos(la) ** 2 * torch.cos(2 * lo))
    exact = -6.0 / EARTH_RADIUS_M ** 2 * f
    out = operators.laplacian(f, g)
    m = out["valid_mask"]
    scale = float(torch.abs(exact).max())
    err = float((torch.abs(out["laplacian"] - exact)[m]).max()) / scale
    assert err < 1e-5


def test_staggered_flux_form_beats_nested_gradients_at_the_second_row():
    """Regression guard for a real bug found while writing this module.

    Computing the meridional term as gradient-of-gradient widens the stencil to +/-2 rows,
    so the first call's first-order edge row contaminates the *second* row of the result.
    Measured 25% relative error there while the true interior held 4e-6. The staggered
    half-point form is compact and does not have this failure.
    """
    shape = (64, 64)
    f, g = latlon_field(shape, 34.0, -0.125, 0.0, 0.125, lambda la, lo: torch.sin(la))
    exact = -2.0 / EARTH_RADIUS_M ** 2 * f

    good = operators.laplacian(f, g)["laplacian"]

    # The naive nested implementation, reproduced here so the comparison is explicit.
    dphi = math.radians(g.dy)
    cos_phi = torch.cos(torch.deg2rad(g.latitudes())).unsqueeze(1)
    d1, _ = torch.gradient(f.to(torch.float64), edge_order=1)
    flux = cos_phi * (d1 / dphi)
    d2, _ = torch.gradient(flux, edge_order=1)
    naive = (d2 / dphi) / (EARTH_RADIUS_M ** 2 * cos_phi)

    row = 1
    good_err = float((torch.abs(good[row] - exact[row]) / torch.abs(exact[row])).max())
    naive_err = float((torch.abs(naive[row] - exact[row]) / torch.abs(exact[row])).max())
    assert naive_err > 0.1, "nested form should be badly wrong at row 1"
    assert good_err < 1e-4
    assert naive_err > 1000 * good_err


def test_cartesian_laplacian_matches_an_analytic_sinusoid():
    n, d = 64, 1000.0
    g = GridSpec.cartesian((n, n), d, d)
    x = torch.arange(n, dtype=torch.float64) * d
    kx = 2 * math.pi / (16 * d)
    f = torch.sin(kx * x).unsqueeze(0).expand(n, n).contiguous()
    exact = -(kx ** 2) * f
    out = operators.laplacian(f, g)
    m = out["valid_mask"]
    scale = float(torch.abs(exact).max())
    rel = float(torch.abs(out["laplacian"] - exact)[m].max()) / scale

    # The 3-point second difference has a known truncation error of (k h)^2 / 12 relative,
    # so the test asserts the measured residual *equals the prediction* rather than merely
    # being under some threshold. That way an error of the right size for the wrong reason
    # still fails.
    predicted = (kx * d) ** 2 / 12.0
    assert abs(rel - predicted) / predicted < 0.1, (
        "measured %.4e vs predicted truncation %.4e" % (rel, predicted))
    assert out["operator"] == "cartesian_5point"


def test_second_difference_needs_three_points():
    g = GridSpec.cartesian((2, 8), 1000.0)
    with pytest.raises(GridError, match="at least 3 points"):
        operators.laplacian(torch.randn(2, 8, dtype=torch.float64), g)


# ============================================================== area weighting

def test_area_weighted_mean_differs_from_the_unweighted_mean_at_high_latitude():
    """The size of the effect, measured rather than asserted."""
    shape = (64, 32)
    f, g = latlon_field(shape, 84.0, -0.5, 0.0, 0.5, lambda la, lo: torch.cos(la))
    weighted = operators.area_weighted_mean(f, g)
    unweighted = float(f.mean())
    assert abs(weighted - unweighted) / abs(unweighted) > 0.01
    # The weighted mean must lean toward the equatorward (larger-area, larger-cos) rows.
    assert weighted > unweighted


def test_area_weighted_mean_of_a_constant_is_that_constant():
    g = GridSpec.latlon((32, 32), 80.0, -0.5, 0.0, 0.5)
    f = torch.full((32, 32), 3.5, dtype=torch.float64)
    assert abs(operators.area_weighted_mean(f, g) - 3.5) < 1e-12
    assert operators.area_weighted_variance(f, g) < 1e-20


def test_area_weighted_statistics_agree_with_plain_ones_on_a_uniform_grid():
    """Positive control: weighting must be a no-op where cells are equal."""
    g = GridSpec.cartesian((32, 32), 1000.0, 1000.0)
    gen = torch.Generator().manual_seed(4)
    f = torch.randn(32, 32, generator=gen, dtype=torch.float64)
    assert abs(operators.area_weighted_mean(f, g) - float(f.mean())) < 1e-12
    assert abs(operators.area_weighted_std(f, g) - float(f.std(unbiased=False))) < 1e-12


def test_error_metrics_report_both_weighted_and_unweighted():
    shape = (48, 24)
    g = GridSpec.latlon(shape, 84.0, -0.5, 0.0, 0.5, variable_units="K")
    gen = torch.Generator().manual_seed(5)
    truth = torch.randn(*shape, generator=gen, dtype=torch.float64)
    lat_w = torch.linspace(0.0, 2.0, shape[0], dtype=torch.float64).unsqueeze(1)
    forecast = truth + lat_w                     # error grows toward one pole
    m = operators.area_weighted_error_metrics(forecast, truth, grid=g)
    assert m["rmse"] != m["rmse_unweighted"]
    assert m["weighting_effect_ratio"] != 1.0
    assert m["variable_units"] == "K"
    assert m["grid"]["kind"] == "latlon"


def test_scoring_refuses_mismatched_grid_kinds():
    a = PhysicalField(torch.randn(8, 8, dtype=torch.float64),
                      grid=GridSpec.cartesian((8, 8), 1000.0))
    b = PhysicalField(torch.randn(8, 8, dtype=torch.float64),
                      grid=GridSpec.pixel((8, 8)))
    with pytest.raises(GridError, match="Regrid one"):
        operators.area_weighted_error_metrics(a, b)


def test_empty_mask_raises_rather_than_dividing_by_zero():
    g = GridSpec.pixel((8, 8))
    mask = torch.zeros(8, 8, dtype=torch.bool)
    with pytest.raises(GridError, match="empty mask"):
        operators.area_weighted_mean(torch.randn(8, 8), g, mask=mask)


# ============================================================== spectra

def test_density_satisfies_parseval_exactly():
    """The normalisation identity: sum S(k) dky dkx == variance, to machine precision."""
    g = GridSpec.cartesian((128, 128), 31000.0, 31000.0)
    gen = torch.Generator().manual_seed(6)
    f = torch.randn(128, 128, generator=gen, dtype=torch.float64)
    fft = torch.fft.fft2(f - f.mean())
    S = (fft.real ** 2 + fft.imag ** 2) * (g.dy * g.dx) / (128 * 128 * (2 * math.pi) ** 2)
    ky, kx = g.wavenumber_axes("rad_per_m", shifted=False)
    dk2 = float(ky[1] - ky[0]) * float(kx[1] - kx[0])
    ratio = float(S.sum()) * dk2 / float(torch.mean((f - f.mean()) ** 2))
    assert abs(ratio - 1.0) < 1e-12


@pytest.mark.parametrize("beta", [5.0 / 3.0, 2.0, 3.0])
def test_recovers_an_injected_energy_slope(beta):
    """Ground truth: a field built with a known E(k) exponent must give it back."""
    f, g = power_law_field(256, beta, seed=1)
    fit = spectra.spectral_slope(f, g)
    assert abs(fit["beta_energy_1d"] - beta) < 0.05
    assert fit["r_squared"] > 0.98
    assert 0.0 < fit["slope_standard_error"] < 0.05
    assert fit["n_points"] > 50


def test_d26_a_kolmogorov_field_is_no_longer_called_charney():
    """Defect D26, head-to-head. This is the exact mislabelling the old code produced.

    E(k) ~ k^-5/3 means S(k) ~ k^-8/3 = k^-2.67. Interpreting that 2.67 as if it were an
    energy exponent lands it next to the Charney/Kraichnan value of 3 - the opposite
    physical regime from the one the field was built with.
    """
    f, g = power_law_field(256, 5.0 / 3.0, seed=2)

    correct = spectra.spectral_slope(f, g)
    assert "Kolmogorov" in correct["regime_interpretation"]
    assert abs(correct["beta_energy_1d"] - 5.0 / 3.0) < 0.1

    legacy = spectra.radial_power_spectrum(f, g, units="legacy_pixel")
    old_style = spectra.fit_power_law(legacy["k"], legacy["power"],
                                      counts=legacy["counts"], k_min=4.0, k_max=60.0,
                                      convention="energy_1d")   # the old, wrong reading
    assert "Charney" in old_style["regime_interpretation"]

    # Reading the same legacy numbers in the correct convention recovers the truth,
    # which shows the defect was the interpretation and not the binning.
    fixed = spectra.fit_power_law(legacy["k"], legacy["power"], counts=legacy["counts"],
                                  k_min=4.0, k_max=60.0, convention="density_2d")
    assert "Kolmogorov" in fixed["regime_interpretation"]


def test_the_two_conventions_differ_by_exactly_one_exponent():
    f, g = power_law_field(128, 2.5, seed=3)
    e = spectra.spectral_slope(f, g, convention="energy_1d")
    s = spectra.spectral_slope(f, g, convention="density_2d")
    assert abs((s["slope_beta"] - e["slope_beta"]) - 1.0) < 1e-6
    assert abs(e["beta_energy_1d"] - s["beta_energy_1d"]) < 1e-6


def test_energy_equals_two_pi_k_times_density():
    f, g = power_law_field(64, 2.0, seed=7)
    e = spectra.radial_power_spectrum(f, g, convention="energy_1d")
    s = spectra.radial_power_spectrum(f, g, convention="density_2d")
    k = np.array(e["k"])
    assert np.allclose(np.array(e["power"]), 2 * math.pi * k * np.array(s["power"]),
                       rtol=1e-12)


def test_pixel_binning_preserves_slope_but_destroys_spectral_features():
    """The corrected claim, both halves asserted.

    An earlier draft of this module asserted that pixel binning biases the fitted slope on
    an anisotropic grid. Measurement refuted that: the angular anisotropy factor is
    independent of radius, so for a pure power law it rescales the intercept and leaves the
    slope alone. What it genuinely destroys is *spectral structure*, which is what a
    ScaleSignature is made of - so this test asserts the true behaviour in both directions.
    """
    # (a) a pure power law: pixel binning gets the slope right even at aspect 4
    f, g = power_law_field(256, 3.0, seed=8, dy=31000.0 * 4, dx=31000.0)
    legacy = spectra.radial_power_spectrum(f, g, units="legacy_pixel")
    lg_fit = spectra.fit_power_law(legacy["k"], legacy["power"], counts=legacy["counts"],
                                   k_min=4.0, k_max=60.0, convention="density_2d")
    assert abs(lg_fit["beta_energy_1d"] - 3.0) < 0.1

    # (b) a peaked spectrum: pixel binning smears it, physical binning does not
    n = 256
    for aspect, expect_smeared in ((1.0, False), (3.0, True)):
        gg = GridSpec.cartesian((n, n), 31000.0 * aspect, 31000.0)
        kk = gg.wavenumber_magnitude("rad_per_m", shifted=False, dtype=torch.float64)
        k0 = 0.35 * gg.isotropic_k_max("rad_per_m")
        amp = torch.exp(-((kk - k0) ** 2) / (2 * (0.03 * k0) ** 2))
        gen = torch.Generator().manual_seed(9)
        phase = torch.rand(n, n, generator=gen, dtype=torch.float64) * 2 * math.pi
        field = torch.fft.ifft2(amp * torch.exp(1j * phase)).real

        def relative_spread(record):
            """Power-weighted std of k over its mean: a stable measure of smearing.

            A half-maximum width was tried first and rejected: only 2-3 bins clear half
            maximum, so the measure swung between 0.07 and 1.94 across random seeds purely
            on whether one outlying bin happened to cross the threshold. The weighted
            second moment uses every bin and is reproducible to ~5% across seeds.
            """
            kv = np.array(record["k"])
            pv = np.clip(np.array(record["power"]), 0.0, None)
            mu = (kv * pv).sum() / pv.sum()
            sd = math.sqrt(((kv - mu) ** 2 * pv).sum() / pv.sum())
            return sd / mu

        phys = spectra.radial_power_spectrum(field, gg, convention="density_2d")
        pix = spectra.radial_power_spectrum(field, gg, units="legacy_pixel")

        # physical binning locates the peak in real units, to within a few percent
        assert abs(np.array(phys["k"])[np.array(phys["power"]).argmax()] / k0 - 1) < 0.05
        assert relative_spread(phys) < 0.05

        if expect_smeared:
            assert relative_spread(pix) > 10 * relative_spread(phys), (
                "pixel binning must smear a spectral peak on an anisotropic grid: "
                "physical %.4f vs pixel %.4f"
                % (relative_spread(phys), relative_spread(pix)))
        else:
            # Positive control: with square cells the two agree, so the test above is
            # detecting anisotropy and not merely a difference of binning scheme.
            assert abs(relative_spread(pix) - relative_spread(phys)) < 0.01


def test_a_sinusoid_is_located_at_its_true_physical_wavelength():
    """The property a regional user actually wants: "the peak is at N km"."""
    n, d = 128, 31000.0
    g = GridSpec.cartesian((n, n), d, d)
    wavelength_m = 16 * d
    x = torch.arange(n, dtype=torch.float64) * d
    f = torch.sin(2 * math.pi * x / wavelength_m).unsqueeze(0).expand(n, n).contiguous()

    rec = spectra.radial_power_spectrum(f, g, convention="density_2d")
    k = np.array(rec["k"])
    peak_k = k[np.array(rec["power"]).argmax()]
    recovered = 2 * math.pi / peak_k
    assert abs(recovered - wavelength_m) / wavelength_m < 0.1, (
        "recovered %.1f km vs true %.1f km" % (recovered / 1000, wavelength_m / 1000))


def test_variance_captured_fraction_is_reported_and_sensible():
    """Turns a confusing quadrature shortfall into a stated number.

    For a *flat* spectrum the returned bins cover the disc inscribed in the k-square, so
    the captured fraction should sit near pi/4. That the number is not 1 is a property of
    radial averaging, not a normalisation bug - and reporting it is what keeps the two
    distinguishable.
    """
    g = GridSpec.cartesian((256, 256), 31000.0, 31000.0)
    gen = torch.Generator().manual_seed(10)
    white = torch.randn(256, 256, generator=gen, dtype=torch.float64)
    rec = spectra.radial_power_spectrum(white, g)
    assert abs(rec["variance_captured_fraction"] - math.pi / 4) < 0.02

    # A red field concentrates power at low k, well inside the disc, so it captures more.
    red, gr = power_law_field(256, 3.0, seed=11)
    assert spectra.radial_power_spectrum(red, gr)["variance_captured_fraction"] > 0.95


def test_legacy_pixel_reproduces_the_old_implementation_exactly():
    """The pre-D13 path is preserved bit-for-bit so the change is auditable."""
    gen = torch.Generator().manual_seed(12)
    data = torch.randn(64, 48, generator=gen, dtype=torch.float64)

    H, W = data.shape
    fft = torch.fft.fftshift(torch.fft.fft2(data - data.mean()))
    power = torch.abs(fft) ** 2 / (H * W)
    y = torch.arange(H, dtype=torch.float64) - H // 2
    x = torch.arange(W, dtype=torch.float64) - W // 2
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    r = torch.round(torch.sqrt(gy ** 2 + gx ** 2)).to(torch.long)
    ref_k, ref_p = [], []
    for kk in range(0, int(min(H, W) // 2) + 1):
        mask = (r == kk)
        if torch.any(mask):
            ref_p.append(float(power[mask].mean()))
            ref_k.append(float(kk))

    rec = spectra.radial_power_spectrum(data, units="legacy_pixel")
    assert len(rec["k"]) == len(ref_k)
    assert np.allclose(np.array(rec["power"]), np.array(ref_p), rtol=1e-12)


def test_bincount_binning_equals_a_reference_python_loop():
    """Defect D17: the vectorised reduction must be numerically identical to the loop."""
    g = GridSpec.cartesian((96, 96), 31000.0, 31000.0)
    gen = torch.Generator().manual_seed(13)
    data = torch.randn(96, 96, generator=gen, dtype=torch.float64)
    rec = spectra.radial_power_spectrum(data, g, convention="density_2d")

    H, W = data.shape
    fft = torch.fft.fftshift(torch.fft.fft2(data - data.mean()))
    S = (fft.real ** 2 + fft.imag ** 2) * (g.dy * g.dx) / (H * W * (2 * math.pi) ** 2)
    kmag = g.wavenumber_magnitude("rad_per_m", shifted=True)
    nb = min(H, W) // 2
    dk = g.isotropic_k_max("rad_per_m") / nb
    idx = torch.floor(kmag / dk).to(torch.long)
    ref_p = []
    for b in range(nb):
        sel = (idx == b) & (kmag > 0)
        if torch.any(sel):
            ref_p.append(float(S[sel].mean()))
    assert np.allclose(np.array(rec["power"]), np.array(ref_p), rtol=1e-12)


def test_legacy_pixel_with_energy_convention_warns_about_d26():
    rec = spectra.radial_power_spectrum(torch.randn(32, 32), units="legacy_pixel",
                                        convention="energy_1d")
    assert rec["convention"] == "density_2d"
    assert any("D26" in w for w in rec["warnings"])


def test_anisotropy_warnings_travel_with_the_spectrum():
    g = GridSpec.latlon((32, 32), 60.0, -0.25, 0.0, 0.25)
    rec = spectra.radial_power_spectrum(torch.randn(32, 32, dtype=torch.float64), g)
    assert not rec["isotropy"]["is_isotropic"]
    assert rec["warnings"], "an anisotropic grid must not produce a silent spectrum"


def test_a_slope_is_reported_with_a_standard_error():
    """Rule R2: a power-law exponent without an uncertainty is not a measurement."""
    f, g = power_law_field(128, 2.0, seed=14)
    fit = spectra.spectral_slope(f, g)
    assert math.isfinite(fit["slope_standard_error"])
    assert fit["slope_standard_error"] > 0
    assert any("isotropy" in a for a in fit["assumptions"])
    assert any("single power law" in a for a in fit["assumptions"])


def test_a_non_power_law_is_not_given_a_regime_label():
    """The false-positive guard: a spectrum that is not straight must say so."""
    n = 128
    g = GridSpec.cartesian((n, n), 31000.0, 31000.0)
    kk = g.wavenumber_magnitude("rad_per_m", shifted=False, dtype=torch.float64)
    k0 = 0.3 * g.isotropic_k_max("rad_per_m")
    amp = torch.exp(-((kk - k0) ** 2) / (2 * (0.05 * k0) ** 2))
    gen = torch.Generator().manual_seed(15)
    phase = torch.rand(n, n, generator=gen, dtype=torch.float64) * 2 * math.pi
    field = torch.fft.ifft2(amp * torch.exp(1j * phase)).real
    fit = spectra.spectral_slope(field, g)
    assert "No single power law" in fit["regime_interpretation"]
    assert "Kolmogorov" not in fit["regime_interpretation"]
    assert "Charney" not in fit["regime_interpretation"]


def test_too_few_bins_explains_itself_instead_of_returning_a_number():
    fit = spectra.fit_power_law([1.0, 2.0, 3.0], [1.0, 0.5, 0.2])
    assert math.isnan(fit["slope_beta"])
    assert "at least 4" in fit["regime_interpretation"]
    assert "Widen the band" in fit["regime_interpretation"]


def test_count_weighting_changes_the_fit_and_is_declared():
    f, g = power_law_field(128, 2.0, seed=16)
    rec = spectra.radial_power_spectrum(f, g)
    w = spectra.fit_power_law(rec["k"], rec["power"], counts=rec["counts"],
                              weighting="counts")
    u = spectra.fit_power_law(rec["k"], rec["power"], counts=rec["counts"],
                              weighting="none")
    assert w["weighting"] == "counts" and u["weighting"] == "none"
    assert w["slope_beta"] != u["slope_beta"]
    # Weighting by mode count should not make the answer worse than unweighted.
    assert abs(w["beta_energy_1d"] - 2.0) <= abs(u["beta_energy_1d"] - 2.0) + 0.05


def test_log_bias_correction_is_off_by_default_and_flagged_when_on():
    f, g = power_law_field(128, 2.0, seed=17)
    rec = spectra.radial_power_spectrum(f, g)
    off = spectra.fit_power_law(rec["k"], rec["power"], counts=rec["counts"])
    on = spectra.fit_power_law(rec["k"], rec["power"], counts=rec["counts"],
                               log_bias_correction=True)
    assert off["log_bias_corrected"] is False
    assert on["log_bias_corrected"] is True
    # It mainly moves the intercept; the slope should shift only slightly.
    assert abs(on["intercept_ln_c"] - off["intercept_ln_c"]) > 1e-6
    assert abs(on["slope_beta"] - off["slope_beta"]) < 0.2


def test_hann_window_preserves_variance_for_a_stationary_field():
    """The mean(w^2) correction is exact only under stationarity, so it is tested there.

    On white noise the correction restores total power to within a few percent. On a
    strongly red field it does not, and that is a property of the estimator rather than a
    bug: a red field's variance is not uniformly distributed across the window's support,
    so scaling by mean(w^2) over- or under-corrects. Measured at 0.55 for E(k) ~ k^-2.
    That caveat is asserted here too, so it stays a known limitation rather than a
    surprise in Phase 4C.
    """
    g = GridSpec.cartesian((128, 128), 31000.0, 31000.0)
    gen = torch.Generator().manual_seed(18)
    white = torch.randn(128, 128, generator=gen, dtype=torch.float64)
    plain = spectra.radial_power_spectrum(white, g, window="none")
    hann = spectra.radial_power_spectrum(white, g, window="hann")
    ratio = hann["variance_total"] / plain["variance_total"]
    assert abs(ratio - 1.0) < 0.05, "window power correction is off; ratio %.3f" % ratio

    red, gr = power_law_field(128, 2.0, seed=18)
    red_ratio = (spectra.radial_power_spectrum(red, gr, window="hann")["variance_total"]
                 / spectra.radial_power_spectrum(red, gr, window="none")["variance_total"])
    assert red_ratio < 0.8, (
        "the known non-stationary caveat should still be visible; got %.3f" % red_ratio)


def test_unknown_units_and_conventions_are_rejected_with_the_valid_options():
    with pytest.raises(GridError, match="cycles_per_km"):
        GridSpec.pixel((8, 8)).wavenumber_axes(units="furlongs")
    with pytest.raises(GridError, match="density_2d"):
        spectra.radial_power_spectrum(torch.randn(8, 8), convention="nonsense")
    with pytest.raises(GridError, match="hann"):
        spectra.radial_power_spectrum(torch.randn(8, 8), window="bartlett")


# ============================================================== integration

def test_physical_field_carries_and_propagates_its_grid():
    g = GridSpec.latlon((32, 64), 60.0, -0.25, 0.0, 0.25, variable_units="K")
    f = PhysicalField(torch.randn(32, 64, dtype=torch.float64), grid=g)
    assert f.units == "K"
    assert f.grid.kind == "latlon"

    small = f.scale_resolution((16, 32))
    assert small.grid.shape == (16, 32)
    assert abs(small.grid.dy * 15 - g.dy * 31) < 1e-12

    parts = f.split_field()
    assert parts["train"].grid.shape[1] == parts["train"].data.shape[1]
    assert parts["test"].grid.lon0 > parts["train"].grid.lon0


def test_physical_field_defaults_to_an_explicit_pixel_grid():
    f = PhysicalField(torch.randn(8, 8))
    assert f.grid.kind == "pixel"
    assert f.grid.is_physical is False


def test_physical_field_rejects_a_grid_of_the_wrong_shape():
    with pytest.raises(ValueError, match="grid.subset"):
        PhysicalField(torch.randn(8, 8), grid=GridSpec.pixel((4, 4)))


def test_diagnostics_carry_units_grid_and_convention():
    g = GridSpec.latlon((32, 32), 50.0, -0.25, 0.0, 0.25, variable_units="K")
    gen = torch.Generator().manual_seed(19)
    truth = PhysicalField(torch.randn(32, 32, generator=gen, dtype=torch.float64), grid=g)
    fcst = PhysicalField(truth.data + 0.1 * torch.randn(32, 32, generator=gen,
                                                        dtype=torch.float64), grid=g)
    d = SpectralSpatialAnalysisEngine.compute_diagnostics(fcst, truth)

    assert d["grid"]["kind"] == "latlon"
    assert d["spatial_metrics"]["area_weighted"] is True
    assert "root_mean_squared_error_unweighted" in d["spatial_metrics"]
    assert d["gradient_errors"]["units"] == "value per m"
    assert d["gradient_errors"]["edge_ring_excluded"] is True
    assert d["spectral_diagnostics"]["k_units"] == "rad m^-1"
    assert d["spectral_diagnostics"]["convention"] == "energy_1d"
    assert d["spectral_diagnostics"]["warnings"], "anisotropic patch must warn"
    assert math.isfinite(
        d["spectral_diagnostics"]["forecast_slope_analysis"]["slope_standard_error"])


def test_diagnostics_still_work_on_a_bare_pixel_field():
    """Backward compatibility: no grid supplied means pixel units, clearly labelled."""
    gen = torch.Generator().manual_seed(20)
    a = PhysicalField(torch.randn(32, 32, generator=gen))
    b = PhysicalField(torch.randn(32, 32, generator=gen))
    d = SpectralSpatialAnalysisEngine.compute_diagnostics(a, b)
    assert d["grid"]["kind"] == "pixel"
    assert d["gradient_errors"]["units"] == "value per pixel"
    assert d["gradient_errors"]["gradient_magnitude_mae"] > 0.0


def test_coherence_of_a_field_with_itself_is_one():
    """Positive control for the vectorised coherence reduction."""
    g = GridSpec.cartesian((64, 64), 31000.0)
    gen = torch.Generator().manual_seed(21)
    f = torch.randn(64, 64, generator=gen, dtype=torch.float64)
    _, coh = SpectralSpatialAnalysisEngine.compute_spectral_coherence(f, f, grid=g)
    assert np.allclose(np.array(coh), 1.0, atol=1e-9)


def test_coherence_of_independent_fields_is_low_at_high_wavenumber():
    g = GridSpec.cartesian((128, 128), 31000.0)
    gen = torch.Generator().manual_seed(22)
    a = torch.randn(128, 128, generator=gen, dtype=torch.float64)
    b = torch.randn(128, 128, generator=gen, dtype=torch.float64)
    k, coh = SpectralSpatialAnalysisEngine.compute_spectral_coherence(a, b, grid=g)
    # High-k annuli hold many modes, so the estimate is not inflated by small samples.
    tail = np.array(coh)[len(coh) // 2:]
    assert tail.mean() < 0.1

def test_unit_labels_follow_the_grid_not_the_requested_name():
    """Caught end-to-end: the API reported k in "rad m^-1" for a field with no metric.

    Requesting `rad_per_m` on a pixel grid is legitimate - it means radians per pixel - but
    labelling it in metres is the exact failure D13 exists to prevent: a plot axis in
    metres over data that never had metres.
    """
    assert GridSpec.pixel((8, 8)).wavelength_units("rad_per_m") == "rad pixel^-1"
    assert GridSpec.pixel((8, 8)).wavelength_units("cycles_per_m") == "cycles pixel^-1"
    assert GridSpec.cartesian((8, 8), 1000.0).wavelength_units("rad_per_m") == "rad m^-1"
    assert GridSpec.latlon((8, 8), 50.0, -0.25, 0.0, 0.25).wavelength_units(
        "rad_per_m") == "rad m^-1"

    # A kilometre-based unit has no meaning without a physical metric, so it is refused
    # rather than silently relabelled.
    with pytest.raises(GridError, match="Attach a GridSpec"):
        GridSpec.pixel((8, 8)).wavelength_units("cycles_per_km")


def test_spectrum_labels_agree_with_the_grid_on_a_pixel_field():
    rec = spectra.radial_power_spectrum(torch.randn(32, 32, dtype=torch.float64))
    assert rec["k_units"] == "rad pixel^-1"
    assert "pixel" in rec["power_units"]
    assert rec["is_physical"] is False


def test_boundary_analysis_declares_its_frame():
    """The boundary lab keeps pixel units on purpose; it must say so in the payload."""
    from src.boundary_lab.boundary import BoundaryConditionLab
    out = BoundaryConditionLab.analyze_boundary_artefacts(
        field=PhysicalField(torch.randn(32, 32)), treatment="zero", pad_width=4)
    assert out["gradient_units"] == "value per pixel"
    assert out["distance_units"] == "pixel"
    assert "padding" in out["gradient_frame_note"]
    assert out["grid"]["kind"] == "pixel"
