"""Time-sequence benchmarks with analytically known answers (roadmap T3.5.17).

Three of the five entries here are **null benchmarks** - their correct answer is "there is
nothing to find". They exist because the failure mode that would destroy this project is not
missing a real signal, it is confidently reporting one that is not there, and real
atmospheric data can never tell you which of those has happened.

Each null targets a *specific* way a mining pipeline manufactures findings:

*   ``pure_noise_sequence`` - the baseline. Nothing whatsoever.
*   ``seasonal_diurnal_sequence`` - deterministic cycles and no weather at all. This is the
    single most likely false discovery on real ERA5: every field correlates with every other
    field because they all follow the sun. Rule R11 says mine anomalies; this benchmark is
    what proves the anomaly step actually happened.
*   ``red_noise_sequence`` - temporally autocorrelated but independent. Frames are not
    independent samples (rule R12), so naive significance testing over-rejects. The check
    here *measures* that over-rejection rather than asserting it in a comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.benchmarks.core import (
    Benchmark,
    CheckResult,
    Outcome,
    register_benchmark,
    stage_check,
)
from src.benchmarks.fields import DEFAULT_SPACING_M, _gaussian_blob, _grid
from src.analysis_engine.climatology import remove_climatology
from src.benchmarks.seeding import SeedBundle
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.transform_engine import stationary


@dataclass
class FieldSequence:
    """A time-ordered stack of fields sharing one grid."""

    fields: List[PhysicalField]
    times_hours: np.ndarray
    grid: GridSpec

    def __len__(self) -> int:
        return len(self.fields)

    def stack(self) -> torch.Tensor:
        return torch.stack([f.data for f in self.fields], dim=0)

    def scale_energy_series(self, levels: int = 4, wavelet: str = "db2") -> Dict[int, np.ndarray]:
        """Per-level SWT detail energy as a function of time - the 'activity by scale' vector.

        Redundancy-normalised (rule R3), so levels are comparable to one another rather than
        reflecting the transform's own 4^level gain.
        """
        series: Dict[int, List[float]] = {j: [] for j in range(1, levels + 1)}
        for f in self.fields:
            coeffs = stationary.apply_swt2d(f, levels=levels, wavelet=wavelet)
            energies = stationary.swt_scale_energies(coeffs, normalize="redundancy")
            for j in range(1, levels + 1):
                series[j].append(energies["level_%d" % j])
        return {j: np.asarray(v) for j, v in series.items()}


# ------------------------------------------------------------------ helpers

def effective_sample_size(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """Bartlett effective sample size for correlating two autocorrelated series.

    ``n_eff = n * (1 - r1 r2) / (1 + r1 r2)`` using lag-1 autocorrelations. This is the
    standard first-order correction; it is exact for AR(1) and approximate otherwise, which
    is stated because rule R12 is about not pretending dependent samples are independent -
    and an approximate correction quietly presented as exact would be the same sin.
    """
    def lag1(v: np.ndarray) -> float:
        v = v - v.mean()
        denom = float(np.sum(v * v))
        if denom <= 0:
            return 0.0
        return float(np.sum(v[1:] * v[:-1]) / denom)

    n = len(x)
    r1 = lag1(x)
    r2 = lag1(y) if y is not None else r1
    prod = max(min(r1 * r2, 0.999), -0.999)
    return max(3.0, n * (1.0 - prod) / (1.0 + prod))


def correlation_p_value(x: np.ndarray, y: np.ndarray, n_eff: Optional[float] = None) -> Tuple[float, float]:
    """Pearson r and a two-sided p-value, optionally against an effective sample size."""
    from scipy import stats  # declared dependency

    r = float(np.corrcoef(x, y)[0, 1])
    n = float(len(x) if n_eff is None else n_eff)
    if n <= 2 or abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2.0) / (1.0 - r * r))
    return r, float(2.0 * stats.t.sf(abs(t), df=n - 2.0))


# ------------------------------------------------------------------ 1. advected vortex

def _torus_gaussian_blob(n: int, cy: float, cx: float, sigma: float,
                         amplitude: float = 1.0) -> torch.Tensor:
    """A Gaussian on a doubly-periodic grid - the same blob, on the declared topology.

    Distances are minimum-image, so the blob that leaves one edge arrives at the other with
    its mass intact. `_gaussian_blob` is the Euclidean version and clips instead; which of
    the two is drawn is the benchmark's `periodic` declaration and never an inference from
    where the trajectory happens to go (defect D59, standard E14).
    """
    idx = torch.arange(n, dtype=torch.float64)
    dy = (idx - cy + n / 2.0) % n - n / 2.0
    dx = (idx - cx + n / 2.0) % n - n / 2.0
    yy, xx = torch.meshgrid(dy, dx, indexing="ij")
    return amplitude * torch.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma ** 2))


def _vortex_centres(n: int, steps: int, start: Tuple[float, float],
                    velocity: Tuple[float, float],
                    periodic: bool) -> List[Tuple[float, float]]:
    """The trajectory, read off the declared topology rather than assumed.

    On a torus the position is taken modulo the axis length; on a plane it is not, and a
    vortex advected past the edge has a position outside the frame - which is the honest
    statement that it left, not a coordinate to be wrapped into a place where there is
    nothing to see.
    """
    wrap = (lambda v: v % n) if periodic else (lambda v: v)
    return [(wrap(start[0] + velocity[0] * t), wrap(start[1] + velocity[1] * t))
            for t in range(steps)]


def _mass_inside_frame(n: int, cy: float, cx: float, sigma: float,
                       periodic: bool) -> float:
    """Fraction of the blob's integral that lies inside the frame - analytic, no detector.

    A statement about the field alone, so that "the feature is measurable in this frame" is
    a property of the benchmark rather than a property of whichever extractor is asking.
    """
    if periodic:
        return 1.0
    root2 = math.sqrt(2.0)
    axis = 1.0
    for centre in (cy, cx):
        axis *= 0.5 * (math.erf(centre / (sigma * root2))
                       + math.erf((n - centre) / (sigma * root2)))
    return float(axis)


def build_advected_vortex(
    bundle: SeedBundle,
    n: int = 128,
    steps: int = 24,
    start: Tuple[float, float] = (32.0, 32.0),
    velocity_cells_per_step: Tuple[float, float] = (1.5, 2.5),
    sigma0: float = 5.0,
    doubling_steps: float = 16.0,
    noise_amplitude: float = 0.02,
    spacing_m: float = DEFAULT_SPACING_M,
    periodic: bool = False,
) -> FieldSequence:
    """A Gaussian vortex on a known straight trajectory, growing at a known rate.

    `periodic` declares the topology of the *field*, and both the drawing here and the
    recorded answer in `truth_advected_vortex` read it. Before defect D59 was fixed they
    disagreed: the answer wrapped and the field did not, which cost nothing at these
    parameters - where the vortex never reaches an edge - and several cells anywhere near
    one.
    """
    g = _grid(n, spacing_m)
    gen = bundle.torch_generator()
    blob = _torus_gaussian_blob if periodic else _gaussian_blob
    centres = _vortex_centres(n, steps, start, velocity_cells_per_step, periodic)
    fields = []
    for t, (cy, cx) in enumerate(centres):
        sigma = sigma0 * (2.0 ** (t / doubling_steps))
        data = blob(n, cy, cx, sigma)
        if noise_amplitude > 0:
            data = data + noise_amplitude * torch.randn(n, n, generator=gen,
                                                        dtype=torch.float64)
        fields.append(PhysicalField(data, grid=g, units="dimensionless"))
    return FieldSequence(fields, np.arange(steps, dtype=float), g)


def truth_advected_vortex(
    n: int = 128, steps: int = 24, start: Tuple[float, float] = (32.0, 32.0),
    velocity_cells_per_step: Tuple[float, float] = (1.5, 2.5), sigma0: float = 5.0,
    doubling_steps: float = 16.0, noise_amplitude: float = 0.02,
    spacing_m: float = DEFAULT_SPACING_M,
    periodic: bool = False,
    measurable_mass_fraction: float = 0.99,
) -> Dict[str, Any]:
    """The recorded answer, derived from the same declaration the builder draws from.

    `track_count`, `births` and `deaths` are **derived** rather than asserted. They used to
    be the constants 1, 1 and 0, which is the right answer only while the vortex stays
    inside the frame; a sequence that advects it out is one where a tracker reporting a
    death is correct and the recorded answer saying otherwise is the defect.

    `measurable_mass_fraction` is the benchmark's own declaration of when a clipped blob has
    stopped being the feature it was - the fraction of its integral that must remain inside
    the frame. It is stated here, in the asset, rather than inferred by whatever detector is
    being graded against it.
    """
    centres = _vortex_centres(n, steps, start, velocity_cells_per_step, periodic)
    sigmas = [sigma0 * (2.0 ** (t / doubling_steps)) for t in range(steps)]
    mass = [_mass_inside_frame(n, cy, cx, s, periodic)
            for (cy, cx), s in zip(centres, sigmas)]
    measurable = [m >= measurable_mass_fraction for m in mass]
    runs = _contiguous_runs(measurable)
    return {
        "positions_rowcol": centres,
        "velocity_cells_per_step": velocity_cells_per_step,
        "speed_m_per_step": math.hypot(*velocity_cells_per_step) * spacing_m,
        "sigma_cells": sigmas,
        "scale_doubling_steps": doubling_steps,
        "periodic": periodic,
        "measurable_mass_fraction": measurable_mass_fraction,
        "mass_inside_frame": mass,
        "measurable_frames": [t for t, ok in enumerate(measurable) if ok],
        "track_count": len(runs),
        "births": len(runs),
        "deaths": sum(1 for _, end in runs if end < steps - 1),
        "note": ("One track for as long as the vortex is inside the frame, with no births "
                 "or deaths while it stays there. A shift-variant transform makes a tracker "
                 "report spurious births and deaths on this sequence - which is what defect "
                 "D1 caused and what T3.5.7's undecimated SWT fixes. The topology is "
                 "declared by `periodic` and read by both the builder and this answer "
                 "(defect D59)."),
    }


def _contiguous_runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of each maximal run of True."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


@stage_check("4D.position")
def _check_vortex_position(seq: FieldSequence, truth: Dict[str, Any]) -> CheckResult:
    """Centroid tracking of the known trajectory - runnable without the 4D tracker.

    Two things here read the benchmark's declared topology and one deliberately does not.
    The **distance** does - a wrap distance on a torus and an ordinary one on a plane - and
    so does the **set of frames**, which is now the frames where the recorded answer says
    99% of the blob is still inside the frame rather than all of them.

    The **centroid estimator** does not, and that is a measurement rather than an oversight.
    Replacing the circular mean with an ordinary one on the Euclidean sequence made the
    worst error eight times larger, 0.830 cells against 0.052: the diffuse noise floor
    survives the `clamp(d - mean, 0)**2` weighting everywhere in the frame, and an ordinary
    mean drags toward the centre of the array while a circular one lets it cancel. The
    circular estimator's failure mode is a blob straddling the seam of a *non*-periodic
    frame, and the measurable-frames filter above is exactly what excludes those.
    """
    periodic = bool(truth.get("periodic", False))
    measurable = set(truth.get("measurable_frames", range(len(seq.fields))))
    errors = []
    for t, (f, (ty, tx)) in enumerate(zip(seq.fields, truth["positions_rowcol"])):
        if t not in measurable:
            continue
        d = f.data
        n = d.shape[0]
        w = torch.clamp(d - d.mean(), min=0.0) ** 2
        idx = torch.arange(n, dtype=torch.float64)

        # circular mean: robust to the diffuse noise floor on either topology
        def circ(weight_1d):
            ang = 2 * math.pi * idx / n
            s = float((weight_1d * torch.sin(ang)).sum())
            c = float((weight_1d * torch.cos(ang)).sum())
            return (math.atan2(s, c) % (2 * math.pi)) * n / (2 * math.pi)

        cy = circ(w.sum(dim=1))
        cx = circ(w.sum(dim=0))
        dy, dx = abs(cy - ty), abs(cx - tx)
        if periodic:
            dy, dx = min(dy, n - dy), min(dx, n - dx)
        errors.append(math.hypot(dy, dx))
    worst = max(errors)
    return CheckResult(
        "4D.position",
        Outcome.PASS if worst < 1.0 else Outcome.FAIL,
        "worst centroid error %.3f cells over %d frames (mean %.3f); target < 1 px"
        % (worst, len(errors), float(np.mean(errors))),
        {"worst_error_cells": worst, "mean_error_cells": float(np.mean(errors))})


@stage_check("4D.scale_evolution")
def _check_vortex_scale_growth(seq: FieldSequence, truth: Dict[str, Any]) -> CheckResult:
    """The vortex doubles in width on a known schedule, so its scale centroid must rise.

    Measured as the energy-weighted mean level **including the LL approximation band as the
    coarsest bin**. A first version used the coarse fraction of *detail* energy only and
    reported a spurious decrease: as the vortex grew past the level-4 band its energy left
    the detail bands entirely for LL, which rose from 4.9e1 to 5.6e2 while the detail total
    fell. Dropping LL from a scale signature loses exactly the energy that "growing to a
    larger scale" consists of, which is a trap T4C.1 will meet again.
    """
    levels = 4
    weighted_mean = []
    for f in seq.fields:
        coeffs = stationary.apply_swt2d(f, levels=levels, wavelet="db2")
        energies = stationary.swt_scale_energies(coeffs, normalize="redundancy")
        bins = {j: energies["level_%d" % j] for j in range(1, levels + 1)}
        bins[levels + 1] = energies["LL"]          # LL is the coarsest bin, not a leftover
        total = sum(bins.values())
        weighted_mean.append(sum(j * e for j, e in bins.items()) / max(total, 1e-30))
    early = float(np.mean(weighted_mean[:4]))
    late = float(np.mean(weighted_mean[-4:]))
    monotone = float(np.corrcoef(np.arange(len(weighted_mean)), weighted_mean)[0, 1])
    ok = late > early and monotone > 0.9
    return CheckResult(
        "4D.scale_evolution",
        Outcome.PASS if ok else Outcome.FAIL,
        "scale centroid rose from level %.3f to %.3f (rank correlation with time %.3f) "
        "as sigma doubled every %.0f steps"
        % (early, late, monotone, truth["scale_doubling_steps"]),
        {"early_centroid": early, "late_centroid": late, "monotonicity": monotone})


#: Enough surrogates to resolve alpha = 0.05 on the (1 + k) / (1 + n) convention, and few
#: enough that the gate runs in a test suite. The threshold is calibrated on the first frame
#: and reused, so a birth cannot be the cut moving between frames.
_TRACKING_SURROGATES = 99
_TRACKING_SEED = 20260824


def _tracked(seq: FieldSequence, truth: Dict[str, Any]):
    """Extract every frame under one calibration and link them (TG2.2 then TG2.3).

    The axes are declared from the benchmark's own `periodic` flag, so the extractor's
    valid-interior rule and the tracker's wrap distance are both reading the same
    declaration the field was drawn from.
    """
    from src.core.domain import AxisSpec
    from src.core.extraction import ExtractionField, calibrate, extract
    from src.core.tracking import track_extractions

    periodic = bool(truth.get("periodic", False))
    axes = (AxisSpec("row", "space", units="cells", periodic=periodic, ordinal=0),
            AxisSpec("col", "space", units="cells", periodic=periodic, ordinal=1))
    frames = [np.asarray(f.data.numpy(), dtype=np.float64) for f in seq.fields]
    calibration = calibrate(frames[0], alpha=0.05, n_surrogates=_TRACKING_SURROGATES,
                            seed=_TRACKING_SEED)
    results = []
    for t, values in enumerate(frames):
        field = ExtractionField(
            values=values, axes=axes, domain="synthetic",
            dataset="advected_vortex_sequence", variable="amplitude", units=None,
            time=float(t), time_units="frames", representation="identity")
        results.append(extract(field, calibration=calibration))
    return track_extractions(results), results


@stage_check("4D.tracking")
def _check_vortex_tracking(seq: FieldSequence, truth: Dict[str, Any]) -> CheckResult:
    """One object, one birth, no deaths, and a growth rate nobody told the tracker.

    The recorded answer this is graded against - the trajectory, the doubling time, the
    track count - was written before any of the code it grades. What the tracker is allowed
    to know is the field, the declared axes and the same alpha the extraction was calibrated
    at; the velocity and the doubling time are outputs, never inputs.
    """
    tracking, results = _tracked(seq, truth)
    if tracking is None:
        return CheckResult("4D.tracking", Outcome.FAIL,
                           "the extractor reported no features in any frame, so there is "
                           "nothing to associate", {"found": 0})
    measurable = list(truth.get("measurable_frames", range(len(seq.fields))))
    longest = tracking.longest
    errors = [
        math.hypot(*_wrapped_delta(item.location.coords, truth["positions_rowcol"][int(item.time)],
                                   n=seq.fields[0].data.shape[0],
                                   periodic=bool(truth.get("periodic", False))))
        for item in longest]
    worst = max(errors)
    doubling = longest.doubling_time().value
    known = float(truth["scale_doubling_steps"])
    detail = {
        "track_count": len(tracking), "births": tracking.births,
        "deaths": tracking.deaths, "longest_track_frames": len(longest),
        "worst_error_cells": worst,
        "coincidence_radius_cells": tracking.steps[1].coincidence_radius,
        "measured_doubling_steps": doubling,
        "features_per_frame": sorted({r.found for r in results}),
    }
    ok = (len(tracking) == truth["track_count"]
          and tracking.births == truth["births"]
          and tracking.deaths == truth["deaths"]
          and len(longest) == len(measurable)
          and worst < 1.0
          and abs(doubling / known - 1.0) < 0.10)
    return CheckResult(
        "4D.tracking",
        Outcome.PASS if ok else Outcome.FAIL,
        "%d track(s), %d birth(s), %d death(s) over %d measurable frames; worst position "
        "error %.3f cells (target < 1); scale doubles every %.2f steps against a recorded "
        "%.0f; gate radius %.1f cells, derived from alpha and the frame's own density"
        % (len(tracking), tracking.births, tracking.deaths, len(measurable), worst,
           doubling, known, tracking.steps[1].coincidence_radius),
        detail)


def _wrapped_delta(coords: Dict[str, float], recorded: Tuple[float, float], *,
                   n: int, periodic: bool) -> Tuple[float, float]:
    dy = abs(coords["row"] - recorded[0])
    dx = abs(coords["col"] - recorded[1])
    if periodic:
        dy, dx = min(dy, n - dy), min(dx, n - dx)
    return dy, dx


register_benchmark(Benchmark(
    name="advected_vortex_sequence",
    kind="sequence",
    description="One vortex on a known trajectory with a known scale-doubling time.",
    gates=("4D.position", "4D.scale_evolution", "4D.tracking"),
    build=build_advected_vortex,
    known_answer=truth_advected_vortex,
    checks=(_check_vortex_position, _check_vortex_scale_growth, _check_vortex_tracking),
    params={"n": 128, "steps": 24},
))


register_benchmark(Benchmark(
    name="advected_vortex_periodic_sequence",
    kind="sequence",
    description="The same vortex on a declared torus, started so that it crosses both "
                "seams - the case defect D59 made unanswerable.",
    gates=("4D.position", "4D.tracking"),
    build=build_advected_vortex,
    known_answer=truth_advected_vortex,
    checks=(_check_vortex_position, _check_vortex_tracking),
    params={"n": 128, "steps": 24, "start": (120.0, 120.0), "periodic": True},
))


# ------------------------------------------------------------------ 2. cross-scale cascade

def build_coupled_cascade(
    bundle: SeedBundle,
    n: int = 128,
    steps: int = 96,
    lag: int = 6,
    coupling_strength: float = 0.9,
    fine_level: int = 1,
    coarse_level: int = 4,
    noise_amplitude: float = 0.25,
    spacing_m: float = DEFAULT_SPACING_M,
) -> FieldSequence:
    """Fine-scale activity at time t drives coarse-scale amplitude at t + lag.

    Built by *modulating band-limited noise*: an independent driver series sets the
    fine-scale amplitude at each step, and the same series - delayed by ``lag`` - sets the
    coarse-scale amplitude. The coupling is therefore in the data by construction, with a
    known lag and known scales, and is not an artefact of the analysis.
    """
    g = _grid(n, spacing_m)
    gen = bundle.torch_generator()
    driver = torch.rand(steps + lag, generator=gen, dtype=torch.float64) * 0.9 + 0.1

    kmag = g.wavenumber_magnitude("rad_per_m", shifted=False, dtype=torch.float64)
    k_nyq = g.isotropic_k_max("rad_per_m")

    def band(level: int) -> torch.Tensor:
        """Ring mask around the wavenumber of SWT detail level ``level``."""
        centre = k_nyq / (2.0 ** level)
        return torch.exp(-((kmag - centre) ** 2) / (2.0 * (0.35 * centre) ** 2))

    fine_mask = band(fine_level)
    coarse_mask = band(coarse_level)

    fields = []
    for t in range(steps):
        phase_f = torch.rand(n, n, generator=gen, dtype=torch.float64) * 2 * math.pi
        phase_c = torch.rand(n, n, generator=gen, dtype=torch.float64) * 2 * math.pi
        fine_amp = float(driver[t + lag])
        coarse_amp = float(driver[t]) * coupling_strength + (1 - coupling_strength) * 0.5
        fine = torch.fft.ifft2(fine_mask * fine_amp * torch.exp(1j * phase_f)).real
        coarse = torch.fft.ifft2(coarse_mask * coarse_amp * torch.exp(1j * phase_c)).real
        data = fine / (fine.std() + 1e-12) * fine_amp + coarse / (coarse.std() + 1e-12) * coarse_amp
        data = data + noise_amplitude * torch.randn(n, n, generator=gen, dtype=torch.float64)
        fields.append(PhysicalField(data, grid=g, units="dimensionless"))
    return FieldSequence(fields, np.arange(steps, dtype=float), g)


def truth_coupled_cascade(
    n: int = 128, steps: int = 96, lag: int = 6, coupling_strength: float = 0.9,
    fine_level: int = 1, coarse_level: int = 4, noise_amplitude: float = 0.25,
    spacing_m: float = DEFAULT_SPACING_M,
) -> Dict[str, Any]:
    return {
        "coupling_lag_steps": lag,
        "driver_level": fine_level,
        "driven_level": coarse_level,
        "direction": "fine leads coarse",
        "coupling_strength": coupling_strength,
        "has_organisation": True,
        "note": ("The fine-scale amplitude at t is set by the same driver that sets the "
                 "coarse-scale amplitude at t + lag, so the correct answer is a single "
                 "cross-scale link at exactly this lag and no other."),
    }


@stage_check("4C.cross_scale")
def _check_cascade_lag(seq: FieldSequence, truth: Dict[str, Any]) -> CheckResult:
    """Recover the injected lag from the cross-correlation of the two scale-energy series."""
    series = seq.scale_energy_series(levels=max(truth["driven_level"], 4))
    fine = series[truth["driver_level"]]
    coarse = series[truth["driven_level"]]
    fine = (fine - fine.mean()) / (fine.std() + 1e-30)
    coarse = (coarse - coarse.mean()) / (coarse.std() + 1e-30)

    max_lag = min(20, len(fine) // 3)
    lags = np.arange(0, max_lag + 1)
    # positive lag means fine leads coarse
    corr = np.array([float(np.mean(fine[:len(fine) - l] * coarse[l:])) for l in lags])
    best = int(lags[int(np.argmax(corr))])
    want = truth["coupling_lag_steps"]
    return CheckResult(
        "4C.cross_scale",
        Outcome.PASS if abs(best - want) <= 1 else Outcome.FAIL,
        "peak cross-correlation at lag %d (true %d), r = %.3f; level %d leads level %d"
        % (best, want, float(corr.max()), truth["driver_level"], truth["driven_level"]),
        {"recovered_lag": best, "true_lag": want, "peak_correlation": float(corr.max())})


register_benchmark(Benchmark(
    name="coupled_cascade_sequence",
    kind="sequence",
    description="Fine-scale activity drives coarse-scale amplitude at a known lag.",
    gates=("4C.cross_scale",),
    build=build_coupled_cascade,
    known_answer=truth_coupled_cascade,
    checks=(_check_cascade_lag,),
    params={"n": 128, "steps": 96, "lag": 6},
))


# ------------------------------------------------------------------ 3. pure noise (NULL)

def build_pure_noise_sequence(bundle: SeedBundle, n: int = 128, steps: int = 96,
                              spacing_m: float = DEFAULT_SPACING_M) -> FieldSequence:
    g = _grid(n, spacing_m)
    gen = bundle.torch_generator()
    fields = [PhysicalField(torch.randn(n, n, generator=gen, dtype=torch.float64),
                            grid=g, units="dimensionless") for _ in range(steps)]
    return FieldSequence(fields, np.arange(steps, dtype=float), g)


def truth_pure_noise_sequence(n: int = 128, steps: int = 96,
                              spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    return {
        "has_organisation": False,
        "expected_cross_scale_links": 0,
        "expected_tracks": 0,
        "expected_patterns": 0,
        "expected_findings_at_alpha_0_05": "at most the nominal false-positive rate",
        "note": "Every stage must return nothing. This is the false-positive floor.",
    }


@stage_check("4C.cross_scale.null")
def _check_noise_has_no_cross_scale_link(seq: FieldSequence,
                                         truth: Dict[str, Any]) -> CheckResult:
    """Independent frames must give no lagged cross-scale coupling beyond chance."""
    series = seq.scale_energy_series(levels=4)
    fine = series[1]
    coarse = series[4]
    n_eff = effective_sample_size(fine, coarse)
    best_p = 1.0
    best_lag = 0
    for lag in range(1, 11):
        r, p = correlation_p_value(fine[:len(fine) - lag], coarse[lag:],
                                   n_eff=n_eff * (len(fine) - lag) / len(fine))
        if p < best_p:
            best_p, best_lag = p, lag
    # Ten lags tested, so a Bonferroni-style floor of 0.05/10 is the honest threshold.
    threshold = 0.05 / 10
    return CheckResult(
        "4C.cross_scale.null",
        Outcome.PASS if best_p > threshold else Outcome.FAIL,
        "best of 10 lags: p = %.4f at lag %d (threshold %.4f after correcting for 10 "
        "tests); no cross-scale coupling claimed" % (best_p, best_lag, threshold),
        {"best_p": best_p, "best_lag": best_lag, "n_eff": n_eff})


register_benchmark(Benchmark(
    name="pure_noise_sequence",
    kind="sequence",
    description="Independent white-noise frames. NOTHING is present; every stage must agree.",
    gates=("4C.cross_scale.null", "4D.tracking.null", "4F.mining.null"),
    build=build_pure_noise_sequence,
    known_answer=truth_pure_noise_sequence,
    checks=(_check_noise_has_no_cross_scale_link,),
    params={"n": 128, "steps": 96},
    is_null=True,
))


# ------------------------------------------------------------------ 4. seasonal + diurnal (NULL)

def build_seasonal_diurnal_sequence(
    bundle: SeedBundle,
    n: int = 64,
    days: int = 40,
    samples_per_day: int = 4,
    seasonal_amplitude: float = 10.0,
    diurnal_amplitude: float = 4.0,
    noise_amplitude: float = 0.05,
    spacing_m: float = DEFAULT_SPACING_M,
) -> FieldSequence:
    """Deterministic cycles and a fixed spatial pattern. **No weather whatsoever.**

    This is what real ERA5 looks like before the climatology is removed, and it is the most
    dangerous input a mining pipeline can be handed: every variable correlates with every
    other variable at every scale, because they all follow the sun. Any "discovery" here is
    a rediscovery of the calendar.
    """
    g = _grid(n, spacing_m)
    gen = bundle.torch_generator()
    idx = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    spatial = torch.cos(2 * math.pi * yy / n) + 0.5 * torch.sin(4 * math.pi * xx / n)

    steps = days * samples_per_day
    fields = []
    times = []
    for t in range(steps):
        hours = 24.0 * t / samples_per_day
        seasonal = seasonal_amplitude * math.sin(2 * math.pi * hours / (24.0 * 365.25))
        diurnal = diurnal_amplitude * math.sin(2 * math.pi * hours / 24.0)
        data = spatial * (seasonal + diurnal)
        data = data + noise_amplitude * torch.randn(n, n, generator=gen, dtype=torch.float64)
        fields.append(PhysicalField(data, grid=g, units="K"))
        times.append(hours)
    return FieldSequence(fields, np.asarray(times), g)


def truth_seasonal_diurnal_sequence(
    n: int = 64, days: int = 40, samples_per_day: int = 4,
    seasonal_amplitude: float = 10.0, diurnal_amplitude: float = 4.0,
    noise_amplitude: float = 0.05, spacing_m: float = DEFAULT_SPACING_M,
) -> Dict[str, Any]:
    return {
        "has_weather": False,
        "cycles_present": ("diurnal", "seasonal"),
        "expected_patterns_after_anomaly_step": 0,
        "residual_variance_ratio_target": 0.05,
        "note": ("The cycles must be REMOVED by the R11 anomaly step, not 'discovered'. "
                 "Mining the raw fields will find strong structure; that structure is the "
                 "calendar."),
    }


@stage_check("4C.r11_anomaly")
def _check_cycles_are_removable(seq: FieldSequence, truth: Dict[str, Any]) -> CheckResult:
    """Removing the diurnal climatology must collapse the variance to the noise floor.

    Implemented here rather than assumed: subtract the mean field for each time-of-day
    across all days, and check what is left. If this benchmark ever fails, the anomaly step
    is not doing its job and every downstream 'finding' is suspect.
    """
    stack = seq.stack()
    hours = seq.times_hours

    # Bin climatology by time of day - the obvious approach, kept as the comparison.
    tod = np.round(hours % 24.0).astype(int)
    raw_var = float(torch.var(stack, unbiased=False))
    binned = stack.clone()
    for slot in np.unique(tod):
        idx = torch.as_tensor(np.flatnonzero(tod == slot), dtype=torch.long)
        binned[idx] = stack[idx] - stack[idx].mean(dim=0, keepdim=True)
    binned_ratio = float(torch.var(binned, unbiased=False)) / raw_var

    # Harmonic regression on both periods - what R11 actually requires.
    out = remove_climatology(stack, hours, n_harmonics=2)
    ratio = out["residual_variance_ratio"]
    target = truth["residual_variance_ratio_target"]
    return CheckResult(
        "4C.r11_anomaly",
        Outcome.PASS if ratio < target else Outcome.FAIL,
        "residual variance after harmonic declimatology: %.6f of raw (target < %.2f). "
        "A time-of-day BIN climatology leaves %.4f, because a %.0f-day record cannot form "
        "a day-of-year climatology and the annual cycle passes straight through."
        % (ratio, target, binned_ratio, (hours[-1] - hours[0]) / 24.0),
        {"harmonic_ratio": ratio, "binned_ratio": binned_ratio,
         "variance_explained": out["variance_explained_by_climatology"]})


@stage_check("4C.r11_split_aware")
def _check_climatology_can_be_fitted_train_only(seq: FieldSequence,
                                                truth: Dict[str, Any]) -> CheckResult:
    """Rule R6: a climatology fitted on all frames leaks the test period into training.

    Fitting on the first 60% and applying everywhere must still remove the cycles, which is
    what makes the split-aware path usable rather than merely correct.
    """
    stack = seq.stack()
    hours = seq.times_hours
    mask = np.zeros(len(seq), dtype=bool)
    mask[: int(0.6 * len(seq))] = True
    out = remove_climatology(stack, hours, n_harmonics=2, fit_mask=mask)
    held_out = out["anomalies"][~torch.as_tensor(mask)]
    ratio = float(torch.var(held_out, unbiased=False)) / out["raw_variance"]
    return CheckResult(
        "4C.r11_split_aware",
        Outcome.PASS if ratio < 0.1 else Outcome.FAIL,
        "climatology fitted on %d of %d frames still removes the cycles on the held-out "
        "frames: residual %.6f of raw variance"
        % (out["fitted_on_frames"], len(seq), ratio),
        {"held_out_ratio": ratio, "fitted_on_all": out["fitted_on_all_frames"]})


@stage_check("4C.r11_raw_is_deceptive")
def _check_raw_cycles_look_like_a_finding(seq: FieldSequence,
                                          truth: Dict[str, Any]) -> CheckResult:
    """Positive control for the trap: the RAW sequence must look strongly organised.

    If the raw fields did not produce a compelling spurious signal, this benchmark would
    not be testing anything. Measuring the trap is what makes the anomaly check meaningful.
    """
    series = seq.scale_energy_series(levels=3)
    fine, coarse = series[1], series[3]
    r, p = correlation_p_value(fine, coarse)
    return CheckResult(
        "4C.r11_raw_is_deceptive",
        Outcome.PASS if abs(r) > 0.5 else Outcome.FAIL,
        "raw (un-anomalised) scale energies correlate at r = %.3f, p = %.2e - a strong "
        "'finding' that is purely the calendar. This is the trap R11 exists for." % (r, p),
        {"raw_correlation": r, "raw_p": p})


register_benchmark(Benchmark(
    name="seasonal_diurnal_sequence",
    kind="sequence",
    description="Deterministic cycles, no weather. The most likely false discovery on ERA5.",
    gates=("4C.r11_anomaly", "4C.r11_split_aware", "4C.r11_raw_is_deceptive",
           "4F.mining.null"),
    build=build_seasonal_diurnal_sequence,
    known_answer=truth_seasonal_diurnal_sequence,
    checks=(_check_cycles_are_removable, _check_climatology_can_be_fitted_train_only,
            _check_raw_cycles_look_like_a_finding),
    params={"n": 64, "days": 40},
    is_null=True,
))


# ------------------------------------------------------------------ 5. red noise (NULL)

def build_red_noise_sequence(
    bundle: SeedBundle,
    n: int = 64,
    steps: int = 200,
    phi: float = 0.85,
    spacing_m: float = DEFAULT_SPACING_M,
) -> FieldSequence:
    """AR(1) in time, independent in space: ``x_t = phi x_{t-1} + sqrt(1-phi^2) eps_t``.

    The innovation is scaled so the marginal variance is 1 regardless of ``phi``, which
    keeps the only difference between this and white noise the *dependence*, not the
    amplitude - otherwise a failed test could be blamed on scale.
    """
    if not -1.0 < phi < 1.0:
        raise ValueError("AR(1) coefficient must satisfy |phi| < 1 for stationarity; got %r"
                         % (phi,))
    g = _grid(n, spacing_m)
    gen = bundle.torch_generator()
    scale = math.sqrt(1.0 - phi * phi)
    state = torch.randn(n, n, generator=gen, dtype=torch.float64)
    fields = []
    for _ in range(steps):
        state = phi * state + scale * torch.randn(n, n, generator=gen, dtype=torch.float64)
        fields.append(PhysicalField(state.clone(), grid=g, units="dimensionless"))
    return FieldSequence(fields, np.arange(steps, dtype=float), g)


def truth_red_noise_sequence(n: int = 64, steps: int = 200, phi: float = 0.85,
                             spacing_m: float = DEFAULT_SPACING_M) -> Dict[str, Any]:
    return {
        "ar1_phi": phi,
        "has_organisation": False,
        "effective_sample_size": steps * (1.0 - phi * phi) / (1.0 + phi * phi),
        "naive_sample_size": steps,
        "expected_findings": 0,
        "note": ("Frames are not independent (rule R12). Naive significance testing treats "
                 "them as if they were and therefore over-rejects; the ESS correction is "
                 "what makes the null hold."),
    }


@stage_check("4C.r12_effective_sample_size")
def _check_r12_overrejection_is_corrected(seq: FieldSequence,
                                          truth: Dict[str, Any]) -> CheckResult:
    """Measure the R12 trap and its fix, rather than asserting either.

    Correlates many pairs of *independent* AR(1) series with the same phi as the benchmark.
    The null is true by construction, so a correct test rejects ~5% of the time. Naive
    testing rejects far more; ESS-corrected testing comes back to nominal.
    """
    phi = truth["ar1_phi"]
    steps = len(seq)
    rng = np.random.default_rng(12345)
    scale = math.sqrt(1.0 - phi * phi)

    def ar1(m):
        out = np.empty(m)
        out[0] = rng.standard_normal()
        for i in range(1, m):
            out[i] = phi * out[i - 1] + scale * rng.standard_normal()
        return out

    trials = 300
    naive_rejects = 0
    ess_rejects = 0
    for _ in range(trials):
        a, b = ar1(steps), ar1(steps)
        _, p_naive = correlation_p_value(a, b)
        _, p_ess = correlation_p_value(a, b, n_eff=effective_sample_size(a, b))
        naive_rejects += p_naive < 0.05
        ess_rejects += p_ess < 0.05

    naive_rate = naive_rejects / trials
    ess_rate = ess_rejects / trials
    ok = naive_rate > 0.15 and ess_rate < 0.12
    return CheckResult(
        "4C.r12_effective_sample_size",
        Outcome.PASS if ok else Outcome.FAIL,
        "phi = %.2f, n = %d, ESS = %.1f. False-positive rate at alpha = 0.05: naive "
        "%.1f%% (should be badly inflated), ESS-corrected %.1f%% (should be near 5%%)"
        % (phi, steps, truth["effective_sample_size"], 100 * naive_rate, 100 * ess_rate),
        {"naive_false_positive_rate": naive_rate, "ess_false_positive_rate": ess_rate,
         "effective_sample_size": truth["effective_sample_size"]})


register_benchmark(Benchmark(
    name="red_noise_sequence",
    kind="sequence",
    description="Temporally autocorrelated but independent; catches R12 violations.",
    gates=("4C.r12_effective_sample_size", "4C.cross_scale.null"),
    build=build_red_noise_sequence,
    known_answer=truth_red_noise_sequence,
    checks=(_check_r12_overrejection_is_corrected,),
    params={"n": 64, "steps": 200, "phi": 0.85},
    is_null=True,
))
