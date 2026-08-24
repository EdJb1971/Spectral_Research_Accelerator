"""Single-field benchmarks with analytically known answers (roadmap T3.5.17).

Every generator here is built in a way that makes its answer derivable *independently of the
analysis code*, which is the whole point: a field synthesised in the Fourier domain from
``S(k) ~ k^-beta`` has that exponent by construction, so recovering it tests the estimator
rather than testing the synthesiser against itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.analysis_engine import spectra
from src.benchmarks.core import (
    Benchmark,
    CheckResult,
    Outcome,
    register_benchmark,
    stage_check,
    not_yet_runnable,
)
from src.benchmarks.seeding import SeedBundle
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.transform_engine import stationary

DEFAULT_SPACING_M = 31000.0     # ~ERA5 0.25 degree at mid-latitude, in round numbers


def _grid(n: int, spacing_m: float = DEFAULT_SPACING_M) -> GridSpec:
    return GridSpec.cartesian((n, n), spacing_m, spacing_m)


def _spectral_field(n: int, beta_density: float, bundle: SeedBundle,
                    spacing_m: float = DEFAULT_SPACING_M) -> Tuple[torch.Tensor, GridSpec]:
    """Field with 2D spectral density S(k) ~ k^-beta_density, random phases.

    Amplitudes are deterministic and only the phases are random, so the *expected* spectrum
    is exact rather than merely asymptotic - which is why the recovered slope has a spread
    of ~0.015 across seeds instead of the ~0.1 a fully random-amplitude construction gives.
    """
    g = _grid(n, spacing_m)
    k = g.wavenumber_magnitude("rad_per_m", shifted=False, dtype=torch.float64).clone()
    k[0, 0] = 1.0
    amp = k ** (-beta_density / 2.0)
    amp[0, 0] = 0.0

    # Phases are taken from the FFT of a real white-noise field, so they are Hermitian by
    # construction and the inverse transform is exactly real. The first version drew
    # independent uniform phases and took `.real` of a non-Hermitian inverse, which is *not*
    # a Gaussian random field: `.real` averages each mode with its conjugate mirror, so the
    # realised amplitude at k is modulated by the random phase difference between +k and -k
    # and the field's true spectrum is no longer `amp`. That left a residual phase structure
    # detectable at z = 2.1 against a correct phase-randomised null - a benchmark that did
    # not quite contain what it claimed, which is fatal for a null benchmark specifically.
    noise = torch.randn(n, n, generator=bundle.torch_generator(), dtype=torch.float64)
    phase = torch.angle(torch.fft.fft2(noise))
    field = torch.fft.ifft2(amp * torch.exp(1j * phase))
    residual = float(torch.abs(field.imag).max())
    scale = float(torch.abs(field.real).max()) or 1.0
    if residual / scale > 1e-9:
        raise ValueError(
            "synthetic field has a residual imaginary part of %.2e; the phase field was not "
            "Hermitian and the realised spectrum is not the requested one" % (residual / scale))
    return field.real, g


# ------------------------------------------------------------------ 1. pure sinusoid

def build_sinusoid(bundle: SeedBundle, n: int = 256, wavelength_cells: int = 16,
                   orientation_deg: float = 0.0,
                   spacing_m: float = DEFAULT_SPACING_M) -> PhysicalField:
    """A single spatial frequency. No randomness at all - the seed is unused by design."""
    g = _grid(n, spacing_m)
    idx = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    theta = math.radians(orientation_deg)
    projected = xx * math.cos(theta) + yy * math.sin(theta)
    data = torch.sin(2 * math.pi * projected / wavelength_cells)
    return PhysicalField(data, grid=g, units="dimensionless")


def truth_sinusoid(n: int = 256, wavelength_cells: int = 16,
                   orientation_deg: float = 0.0,
                   spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    wavelength_m = wavelength_cells * spacing_m
    return {
        "wavelength_m": wavelength_m,
        "wavenumber_rad_per_m": 2 * math.pi / wavelength_m,
        "orientation_deg": orientation_deg,
        # A wavelength of L cells sits in SWT detail level j where 2^j ~ L/2.
        "dominant_swt_level": max(1, int(round(math.log2(wavelength_cells / 2.0)))),
        "all_energy_at_one_scale": True,
    }


@stage_check("4C.scale_signature")
def _check_sinusoid_peak(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    rec = spectra.radial_power_spectrum(field, convention="density_2d")
    k = np.array(rec["k"])
    peak = float(k[np.array(rec["power"]).argmax()])
    target = truth["wavenumber_rad_per_m"]
    rel = abs(peak - target) / target
    return CheckResult(
        "4C.scale_signature",
        Outcome.PASS if rel < 0.08 else Outcome.FAIL,
        "spectral peak at %.4e rad/m vs true %.4e (%.1f%% off); wavelength %.0f km vs %.0f km"
        % (peak, target, 100 * rel, 2 * math.pi / peak / 1000,
           truth["wavelength_m"] / 1000),
        {"peak_k": peak, "true_k": target, "relative_error": rel})


@stage_check("4C.scale_signature.wavelet")
def _check_sinusoid_swt_level(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    levels = 5
    coeffs = stationary.apply_swt2d(field, levels=levels, wavelet="db2")
    fractions = stationary.swt_energy_fractions(coeffs)
    detail = {int(kk.split("_")[1]): v for kk, v in fractions.items()
              if kk.startswith("level_")}
    dominant = max(detail, key=detail.get)
    expected = truth["dominant_swt_level"]
    return CheckResult(
        "4C.scale_signature.wavelet",
        Outcome.PASS if abs(dominant - expected) <= 1 else Outcome.FAIL,
        "dominant SWT detail level %d (expected %d); energy fractions %s"
        % (dominant, expected, {k: round(v, 4) for k, v in sorted(detail.items())}),
        {"dominant_level": dominant, "expected_level": expected, "fractions": detail})


register_benchmark(Benchmark(
    name="pure_sinusoid",
    kind="field",
    description="A single spatial frequency; all energy belongs at one scale.",
    gates=("4C.scale_signature", "4C.scale_signature.wavelet"),
    build=build_sinusoid,
    known_answer=truth_sinusoid,
    checks=(_check_sinusoid_peak, _check_sinusoid_swt_level),
    params={"n": 256, "wavelength_cells": 16},
))


# ------------------------------------------------------------------ 2. fBm, known H

def build_fbm(bundle: SeedBundle, n: int = 256, hurst: float = 0.7,
              spacing_m: float = DEFAULT_SPACING_M) -> PhysicalField:
    """Fractional Brownian surface with Hurst exponent ``hurst``.

    For a 2D fBm, ``S(k) ~ k^-(2H + 2)``; the extra ``+d`` is the dimension term, so the
    1D energy spectrum is ``E(k) = 2 pi k S(k) ~ k^-(2H + 1)``. Verified empirically across
    H = 0.3, 0.5, 0.7, 0.9 to within 1e-3.
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError("Hurst exponent must lie strictly in (0, 1); got %r" % (hurst,))
    data, g = _spectral_field(n, 2.0 * hurst + 2.0, bundle, spacing_m)
    return PhysicalField(data, grid=g, units="dimensionless")


def truth_fbm(n: int = 256, hurst: float = 0.7,
              spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    return {
        "hurst": hurst,
        "beta_energy_1d": 2.0 * hurst + 1.0,
        "beta_density_2d": 2.0 * hurst + 2.0,
        "has_organisation": False,
        "expected_surrogate_finding_count": 0,
        "note": ("Self-similar by construction: scale-free, with no localised structure and "
                 "no preferred scale. A stage that reports coherent features here is "
                 "reporting its own noise."),
    }


@stage_check("4C.alpha")
def _check_fbm_slope(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    fit = spectra.spectral_slope(field)
    got = fit["beta_energy_1d"]
    want = truth["beta_energy_1d"]
    se = fit["slope_standard_error"]
    # Tolerance from the fit's own uncertainty plus a small systematic allowance, rather
    # than a hand-picked constant (rule R16).
    tol = max(4.0 * se, 0.06)
    return CheckResult(
        "4C.alpha",
        Outcome.PASS if abs(got - want) <= tol else Outcome.FAIL,
        "recovered E(k) exponent %.4f +/- %.4f, true %.4f (H = %.2f), tolerance %.3f"
        % (got, se, want, truth["hurst"], tol),
        {"recovered": got, "true": want, "standard_error": se, "r_squared": fit["r_squared"]})


@stage_check("4C.surrogate_null")
def _check_fbm_no_organisation(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    """**The false-positive floor for surrogate testing** (defect D8, rule R1).

    fBm is scale-free by construction: no preferred scale, no localised structure, nothing to
    find. A surrogate test that reports organisation here is reporting its own noise.

    The statistic is deliberately one that *would* detect real organisation - the excess
    kurtosis of the level-2 wavelet detail coefficients, which rises when energy concentrates
    into localised structures rather than spreading evenly. Against a phase-randomised null,
    which preserves the power spectrum exactly, a rejection could only come from phase
    organisation. There is none, so the test must not reject.
    """
    from src.statistics.significance import surrogate_test
    from src.transform_engine import stationary as swt

    def concentration(data) -> float:
        f = PhysicalField(torch.as_tensor(np.asarray(data), dtype=torch.float64))
        coeffs = swt.apply_swt2d(f, levels=2, wavelet="db2")
        detail = torch.cat([coeffs["level_2"][b].flatten() for b in ("LH", "HL", "HH")])
        centred = detail - detail.mean()
        variance = float(torch.mean(centred ** 2))
        if variance <= 0:
            return 0.0
        return float(torch.mean(centred ** 4)) / (variance ** 2) - 3.0

    result = surrogate_test(field.data.numpy(), concentration,
                            method="phase_randomise", n_surrogates=199,
                            seed=4242, alternative="two_sided")
    rejected = result["p_value"] <= 0.05
    return CheckResult(
        "4C.surrogate_null",
        Outcome.FAIL if rejected else Outcome.PASS,
        ("REPORTED ORGANISATION IN SCALE-FREE fBm: p = %.4f, z = %.2f" if rejected else
         "no organisation claimed on scale-free fBm: p = %.4f (z = %.2f), floor %.4f, "
         "%d phase-randomised surrogates")
        % ((result["p_value"], result["z_score"]) if rejected else
           (result["p_value"], result["z_score"], result["p_value_floor"],
            result["n_surrogates"])),
        {"p_value": result["p_value"], "z_score": result["z_score"],
         "n_surrogates": result["n_surrogates"], "null_preserves": result["null_preserves"]})


register_benchmark(Benchmark(
    name="fractional_brownian",
    kind="field",
    description="Scale-free fBm with a known Hurst exponent; exact alpha, NO organisation.",
    gates=("4C.alpha", "4C.surrogate_null"),
    build=build_fbm,
    known_answer=truth_fbm,
    checks=(_check_fbm_slope, _check_fbm_no_organisation),
    params={"n": 256, "hurst": 0.7},
    is_null=True,
))


# ------------------------------------------------------------------ 3. white noise field

def build_white_noise(bundle: SeedBundle, n: int = 256,
                      spacing_m: float = DEFAULT_SPACING_M) -> PhysicalField:
    g = _grid(n, spacing_m)
    data = torch.randn(n, n, generator=bundle.torch_generator(), dtype=torch.float64)
    return PhysicalField(data, grid=g, units="dimensionless")


def truth_white_noise(n: int = 256, spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    return {
        # White noise has flat S(k), so E(k) = 2 pi k S(k) rises linearly: beta_E = -1.
        # Stated explicitly because "flat spectrum" means beta = 0 in one convention and
        # beta = -1 in the other, and conflating them is defect D26 in miniature (R15).
        "beta_energy_1d": -1.0,
        "beta_density_2d": 0.0,
        "has_organisation": False,
        "expected_regime_label": "Flat spectrum",
        "expected_surrogate_finding_count": 0,
    }


@stage_check("4C.alpha")
def _check_white_noise_flat(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    fit = spectra.spectral_slope(field)
    got = fit["beta_energy_1d"]
    ok = abs(got - truth["beta_energy_1d"]) < 0.15
    return CheckResult(
        "4C.alpha",
        Outcome.PASS if ok else Outcome.FAIL,
        "E(k) exponent %.4f (expected %.1f for white noise); label: %s"
        % (got, truth["beta_energy_1d"], fit["regime_interpretation"][:60]),
        {"recovered": got, "label": fit["regime_interpretation"]})


@stage_check("4C.false_positive")
def _check_white_noise_names_no_cascade(field: PhysicalField,
                                        truth: Dict[str, Any]) -> CheckResult:
    """The false-positive floor: white noise must not be called a turbulent cascade."""
    label = spectra.spectral_slope(field)["regime_interpretation"]
    bad = ("Kolmogorov" in label) or ("Charney" in label)
    return CheckResult(
        "4C.false_positive",
        Outcome.FAIL if bad else Outcome.PASS,
        ("REPORTED A CASCADE ON WHITE NOISE: %s" % label) if bad
        else "no cascade regime claimed; reported as %r" % label[:60],
        {"label": label})


register_benchmark(Benchmark(
    name="white_noise_field",
    kind="field",
    description="Spatially uncorrelated noise; the single-field false-positive floor.",
    gates=("4C.alpha", "4C.false_positive"),
    build=build_white_noise,
    known_answer=truth_white_noise,
    checks=(_check_white_noise_flat, _check_white_noise_names_no_cascade),
    params={"n": 256},
    is_null=True,
))


# ------------------------------------------------- 5. structureless field, every lens

#: A family-wise level over forty-five planes cannot be resolved by fewer. The audit refuses
#: rather than quietly reporting at a level it cannot reach, so this number is load-bearing:
#: 499 surrogates is a refusal, not a coarser answer.
_AUDIT_SURROGATES = 999
_AUDIT_SEED = 20260825

#: 128 cells, not 256. Large enough that the dual-tree level-3 subbands still have a valid
#: interior once their filters have taken six samples off every side, and small enough that
#: a thousand decompositions of five representations run in a test suite.
_AUDIT_N = 128


def build_representation_null(bundle: SeedBundle, n: int = _AUDIT_N, hurst: float = 0.7,
                              spacing_m: float = DEFAULT_SPACING_M) -> PhysicalField:
    """The same scale-free fBm as benchmark 2, sized so every registered lens can be audited.

    fBm rather than white noise on purpose. White noise has no structure *and* no
    correlation, so a lens has very little to work with; fBm is smooth, correlated and
    edge-bearing, which is exactly the material a boundary rule or a decimation phase can
    turn into a localised artefact. It is the harder null, and the one rule R8 is about.
    """
    return build_fbm(bundle, n=n, hurst=hurst, spacing_m=spacing_m)


def truth_representation_null(n: int = _AUDIT_N, hurst: float = 0.7,
                              spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    return {
        "hurst": hurst,
        "has_organisation": False,
        "expected_feature_count": 0,
        "family_wise_alpha": 0.05,
        "expected_uncovered_transforms": 0,
        "minimum_planes_audited": 40,
        "note": ("Scale-free by construction: no localised structure at any scale, so a "
                 "feature reported in any plane of any registered representation was made "
                 "by the representation and not found in the data (rule R8). The audit "
                 "corrects for its own family, because forty-five planes tested at a "
                 "nominal 0.05 each report something on roughly five structureless fields "
                 "out of six."),
    }


@stage_check("4E.representation_audit")
def _check_no_representation_manufactures_a_feature(field: PhysicalField,
                                                    truth: Dict[str, Any]) -> CheckResult:
    """**The false-positive floor for representations** (rule R8, roadmap TG2.4).

    TG2.2 measured this floor on the raw array. This measures it on every plane of every
    registered lens: the null is propagated *through* each representation, so an artefact
    present in every realisation raises the cut instead of being reported, and the family of
    planes is counted and paid for before any of them is read.

    Three ways to pass without having looked, all of them failures here: a registered
    transform with no plane builder (`uncovered`), a plane whose null nothing could exceed
    (`vacuous`), and a plane whose valid interior is empty (refused, and struck from the
    family). The report carries all three counts and the number of cells actually searched.
    """
    from src.core.domain import AxisSpec
    from src.core.extraction import ExtractionField
    from src.core.representation import audit_all

    values = np.asarray(field.data.numpy(), dtype=np.float64)
    axes = (AxisSpec("row", "space", units="cells", ordinal=0),
            AxisSpec("col", "space", units="cells", ordinal=1))
    frame = ExtractionField(
        values=values, axes=axes, domain="synthetic", dataset="representation_null",
        variable="amplitude", units=None, time=0.0, time_units="frames",
        representation="identity")
    report = audit_all(frame, alpha=float(truth["family_wise_alpha"]),
                       n_surrogates=_AUDIT_SURROGATES, seed=_AUDIT_SEED)
    summary = report.describe()
    level = report.level
    measured = {
        "found": report.found,
        "findings": summary["findings"],
        "representations": summary["representations"],
        "planes_audited": summary["planes_audited"],
        "planes_refused": summary["planes_refused"],
        "searchable_cells": report.searched,
        "uncovered_transforms": summary["uncovered_transforms"],
        "unauditable": summary["unauditable"],
        "vacuous": summary["vacuous"],
        "family_size": level.family_size,
        "alpha_per_plane": level.alpha_per_plane,
        "measured_fwer": level.measured_fwer,
        "uncorrected_fwer": level.notes.get("uncorrected_fwer"),
        "n_surrogates": _AUDIT_SURROGATES,
    }
    problems = []
    if report.found:
        problems.append("MANUFACTURED %d FEATURE(S) IN SCALE-FREE fBm: %s"
                        % (report.found, summary["findings"]))
    if summary["uncovered_transforms"]:
        problems.append("registered transforms with no plane builder, so they were never "
                        "audited: %s" % ", ".join(summary["uncovered_transforms"]))
    if summary["planes_audited"] < int(truth["minimum_planes_audited"]):
        problems.append("only %d planes could be audited, below the %d this benchmark "
                        "expects; a clean result from too few planes is a pass earned by "
                        "not looking"
                        % (summary["planes_audited"], int(truth["minimum_planes_audited"])))
    vacuous_audited = [p.plane.name for a in report.audits for p in a.audited_planes
                       if p.vacuous]
    if vacuous_audited:
        problems.append("planes whose null nothing could exceed were counted as clean: %s"
                        % ", ".join(vacuous_audited))
    return CheckResult(
        "4E.representation_audit",
        Outcome.FAIL if problems else Outcome.PASS,
        ("; ".join(problems)) if problems else
        ("no feature in %d planes of %d representations, %d cells searched; family of %d "
         "corrected to alpha %.4g per plane (measured FWER %.3f, %.3f uncorrected); %d "
         "planes refused by name, %d transforms uncovered"
         % (summary["planes_audited"], len(summary["representations"]), report.searched,
            level.family_size, level.alpha_per_plane, level.measured_fwer,
            level.notes.get("uncorrected_fwer", float("nan")),
            summary["planes_refused"], len(summary["uncovered_transforms"]))),
        measured)


register_benchmark(Benchmark(
    name="representation_null_field",
    kind="field",
    description=("Scale-free fBm audited through every registered representation; the "
                 "false-positive floor rule R8 asks for."),
    gates=("4E.representation_audit",),
    build=build_representation_null,
    known_answer=truth_representation_null,
    checks=(_check_no_representation_manufactures_a_feature,),
    params={"n": _AUDIT_N, "hurst": 0.7},
    is_null=True,
))


# ------------------------------------------------------------------ 4. planted configuration

def _gaussian_blob(n: int, cy: float, cx: float, sigma: float,
                   amplitude: float = 1.0) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    return amplitude * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def build_planted_configuration(
    bundle: SeedBundle,
    n: int = 256,
    triangle_side: float = 40.0,
    centre: Optional[Tuple[float, float]] = None,
    feature_sigma: float = 6.0,
    rotation_deg: float = 0.0,
    scale_factor: float = 1.0,
    translation: Tuple[float, float] = (0.0, 0.0),
    noise_amplitude: float = 0.05,
    spacing_m: float = DEFAULT_SPACING_M,
) -> PhysicalField:
    """Three small features in an equilateral triangle - a known "constellation".

    ``rotation_deg``, ``scale_factor`` and ``translation`` exist so the *same* configuration
    can be presented transformed. A 4E matcher that is genuinely translation-, rotation- and
    scale-invariant must return the same pattern for all of them; one that has merely
    memorised pixel positions will not.
    """
    g = _grid(n, spacing_m)
    cy, cx = centre if centre is not None else (n / 2.0, n / 2.0)
    cy += translation[0]
    cx += translation[1]
    side = triangle_side * scale_factor
    radius = side / math.sqrt(3.0)
    sigma = feature_sigma * scale_factor

    data = torch.zeros(n, n, dtype=torch.float64)
    for i in range(3):
        angle = math.radians(rotation_deg + 120.0 * i - 90.0)
        data = data + _gaussian_blob(n, cy + radius * math.sin(angle),
                                     cx + radius * math.cos(angle), sigma)
    if noise_amplitude > 0:
        data = data + noise_amplitude * torch.randn(
            n, n, generator=bundle.torch_generator(), dtype=torch.float64)
    return PhysicalField(data, grid=g, units="dimensionless")


def truth_planted_configuration(
    n: int = 256,
    triangle_side: float = 40.0,
    centre: Optional[Tuple[float, float]] = None,
    feature_sigma: float = 6.0,
    rotation_deg: float = 0.0,
    scale_factor: float = 1.0,
    translation: Tuple[float, float] = (0.0, 0.0),
    noise_amplitude: float = 0.05,
    spacing_m: float = DEFAULT_SPACING_M,
) -> Dict[str, Any]:
    cy, cx = centre if centre is not None else (n / 2.0, n / 2.0)
    cy += translation[0]
    cx += translation[1]
    side = triangle_side * scale_factor
    radius = side / math.sqrt(3.0)
    positions = []
    for i in range(3):
        angle = math.radians(rotation_deg + 120.0 * i - 90.0)
        positions.append((cy + radius * math.sin(angle), cx + radius * math.cos(angle)))
    return {
        "feature_count": 3,
        "positions_rowcol": positions,
        "pairwise_distance_cells": side,
        "pairwise_distance_m": side * spacing_m,
        "feature_sigma_cells": feature_sigma * scale_factor,
        # Relative geometry is what a translation-invariant matcher should key on, and it
        # is invariant to translation and rotation but NOT to scale - so the scale ratio is
        # reported separately, which is exactly the toggle T4E needs.
        "invariant_signature": "equilateral triangle, 3 features",
        "scale_ratio_vs_reference": scale_factor,
        # What the 4E.invariance check needs to rebuild this configuration under transforms
        # and to measure its own noise floor. The minima are stated here, in the known
        # answer, rather than inside the check: a gate that decides how hard to look at the
        # moment it looks can always decide to look less hard.
        "noise_amplitude": noise_amplitude,
        "minimum_invariance_replicates": 10,
        "minimum_scale_ratios_recovered": 3,
        "invariance_transforms": ("rotation", "translation", "rescaling"),
    }


@stage_check("4E.feature_detection")
def _check_planted_features_present(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    """Validates the *generator*, which is a prerequisite for trusting the 4E gate.

    The mining stage does not exist yet, but a benchmark that does not contain what it
    claims to contain would silently invalidate that future gate, so the geometry is
    verified here and now.
    """
    data = field.data
    peak = float(data.max())
    found = []
    for (ry, rx) in truth["positions_rowcol"]:
        iy, ix = int(round(ry)), int(round(rx))
        window = data[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        found.append(float(window.max()) > 0.6 * peak)
    ok = all(found)
    return CheckResult(
        "4E.feature_detection",
        Outcome.PASS if ok else Outcome.FAIL,
        "all 3 planted features present at their stated positions: %s (peak %.3f)"
        % (found, peak),
        {"found": found, "positions": truth["positions_rowcol"]})


_INVARIANCE_REPLICATES = 10
_INVARIANCE_ROTATIONS = (37.0, 71.0, 211.0, 259.0)
_INVARIANCE_TRANSLATIONS = ((25.0, -30.0), (-40.0, 15.0), (18.0, 22.0))
_INVARIANCE_SCALES = (0.5, 0.75, 1.5, 2.0, 3.0)
_INVARIANCE_SURROGATES = 99
_INVARIANCE_SEED = 1234


def _invariance_features(values):
    """Extract the configuration from one array, declared the way TG2.2 requires."""
    from src.core.domain import AxisSpec
    from src.core.extraction import ExtractionField, extract

    frame = ExtractionField(
        values=np.asarray(values, dtype=np.float64),
        axes=(AxisSpec("row", "space", units="cells", ordinal=0),
              AxisSpec("col", "space", units="cells", ordinal=1)),
        domain="synthetic", dataset="planted_configuration", variable="amplitude",
        units=None, time=0.0, time_units="frames", representation="identity")
    return list(extract(frame, n_surrogates=_INVARIANCE_SURROGATES, seed=_INVARIANCE_SEED))


@stage_check("4E.invariance")
def _check_planted_invariance(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    """**The 4E gate** (roadmap TG3.4): is the same configuration recognised as the same?

    The check does not ask whether some matcher happens to match. It asks whether every
    registered matcher's *declared* invariance is the invariance it has, which makes the
    position-memorising control fail by the general rule rather than by a special case, and
    makes a future matcher that overclaims fail the same way without an edit here.

    Three things it refuses to accept as a pass. A noise floor measured from too few
    replicates, because that floor is a maximum and a maximum over few samples is biased
    low, in the direction that rejects a matcher which is invariant. A test whose smallest
    possible p-value sits above its own alpha, which is TG2.4's vacuous rule - it would
    report invariance because it could not have reported anything else. And a scale ratio
    nobody recovered: a matcher blind to scale and a matcher that measures scale and states
    it both match a rescaled triangle, and only one of them has said how much bigger it was.
    """
    from src.benchmarks.seeding import derive
    from src.core.invariance import (
        MATCHERS, TRANSFORMS, Presentation, audit_declared_invariance,
    )

    n = int(field.data.shape[0])
    base = dict(n=n, triangle_side=float(truth["pairwise_distance_cells"]),
                feature_sigma=float(truth["feature_sigma_cells"]),
                noise_amplitude=float(truth["noise_amplitude"]))

    def built(label, **overrides):
        params = dict(base)
        params.update(overrides)
        bundle = derive("planted_configuration/%s" % label)
        return _invariance_features(
            build_planted_configuration(bundle, **params).data.numpy())

    reference = _invariance_features(field.data.numpy())
    replicates = [built("replicate/%d" % i) for i in range(_INVARIANCE_REPLICATES)]

    presentations = []
    for angle in _INVARIANCE_ROTATIONS:
        name = "rotation/%g" % angle
        presentations.append(
            Presentation(name, ("rotation",), built(name, rotation_deg=angle)))
    for shift in _INVARIANCE_TRANSLATIONS:
        name = "translation/%g,%g" % shift
        presentations.append(
            Presentation(name, ("translation",), built(name, translation=shift)))
    for factor in _INVARIANCE_SCALES:
        name = "rescaling/%g" % factor
        presentations.append(
            Presentation(name, ("rescaling",), built(name, scale_factor=factor),
                         expected_scale_ratio=factor))
    combined = "combined/17deg+shift+1.5x"
    presentations.append(Presentation(
        combined, TRANSFORMS,
        built(combined, rotation_deg=17.0, translation=(-20.0, 40.0), scale_factor=1.5),
        expected_scale_ratio=1.5))

    reports = audit_declared_invariance(reference, presentations, replicates)

    problems = []
    minimum = int(truth["minimum_invariance_replicates"])
    if len(replicates) < minimum:
        problems.append(
            "the noise floor was measured from %d replicates, below the %d this benchmark "
            "requires; that floor is a maximum, and a maximum over too few samples is "
            "biased low in the direction that rejects a matcher which is invariant"
            % (len(replicates), minimum))
    for name, report in sorted(reports.items()):
        if report.vacuous:
            problems.append(
                "matcher %r was passed on %s by a test that could not have failed"
                % (name, ", ".join(report.vacuous)))
        if report.overclaimed:
            problems.append("matcher %r declares invariance to %s and does not have it"
                            % (name, ", ".join(report.overclaimed)))
        if report.understated:
            problems.append("matcher %r survived %s without declaring it"
                            % (name, ", ".join(report.understated)))

    fully = sorted(name for name, report in reports.items()
                   if tuple(report.measured) == TRANSFORMS)
    if not fully:
        problems.append("no registered matcher survived all of %s, which is the gate itself"
                        % ", ".join(TRANSFORMS))
    controls = sorted(name for name in reports
                      if not MATCHERS.entry(name).capabilities.get("invariant_to", ()))
    if not controls:
        problems.append(
            "no matcher declaring no invariance was audited, so nothing showed that this "
            "configuration can be got wrong; a suite in which everything passes is a suite "
            "that has not been shown able to fail")
    for name in controls:
        if reports[name].measured:
            problems.append(
                "the position-memorising control %r survived %s, so the presentations do "
                "not move the configuration far enough to tell a matcher from a memory"
                % (name, ", ".join(reports[name].measured)))

    needed = int(truth["minimum_scale_ratios_recovered"])
    recovered = 0
    for name in fully:
        recovery = reports[name].scale_recovery
        if recovery is None:
            problems.append(
                "matcher %r was never asked how much bigger anything was; a matcher blind "
                "to scale and one that measures scale and states it both match a rescaled "
                "triangle, and only one of them produces the number" % name)
            continue
        recovered = max(recovered, len(recovery.errors))
        if recovery.vacuous:
            problems.append(
                "matcher %r had its scale recovery passed by a test that could not have "
                "failed" % name)
        elif not recovery.accurate:
            worst = max(range(len(recovery.errors)), key=lambda i: recovery.errors[i])
            problems.append(
                "matcher %r recovered scale ratios worse than its own noise floor "
                "(p=%.4g at alpha %.4g); worst was %s, built %g times bigger and recovered "
                "as %.4f"
                % (name, recovery.p_value, recovery.alpha, recovery.presentations[worst],
                   recovery.expected[worst], recovery.recovered[worst]))
        if len(recovery.errors) < needed:
            problems.append(
                "only %d rescaled presentation(s) were put to matcher %r, below the %d "
                "this benchmark requires" % (len(recovery.errors), name, needed))

    summary = "; ".join(
        "%s declared %s, survived %s"
        % (name, ",".join(report.declared) or "nothing",
           ",".join(report.measured) or "nothing")
        for name, report in sorted(reports.items()))
    return CheckResult(
        "4E.invariance",
        Outcome.FAIL if problems else Outcome.PASS,
        ("; ".join(problems)) if problems else
        ("%d matchers audited over %d presentations against %d replicates: %s; %d scale "
         "ratio(s) recovered"
         % (len(reports), len(presentations), len(replicates), summary, recovered)),
        {"reports": {name: report.describe() for name, report in sorted(reports.items())},
         "n_presentations": len(presentations), "n_replicates": len(replicates),
         "scale_ratios_recovered": recovered})


register_benchmark(Benchmark(
    name="planted_configuration",
    kind="field",
    description="Three features in a known equilateral triangle; the 4E invariance target.",
    gates=("4E.feature_detection", "4E.invariance"),
    build=build_planted_configuration,
    known_answer=truth_planted_configuration,
    checks=(_check_planted_features_present, _check_planted_invariance),
    params={"n": 256, "triangle_side": 40.0},
))
