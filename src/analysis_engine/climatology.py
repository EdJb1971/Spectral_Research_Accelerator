"""Climatology removal: turning fields into anomalies (rule R11, with rule R6's split).

Mining raw meteorological fields discovers the calendar. Every variable correlates with
every other variable at every scale because they all follow the sun, and a pattern miner
handed raw ERA5 will report those correlations with overwhelming significance. They are
real, they are already known, and they are not what anyone is looking for.

**Why harmonic regression rather than a bin climatology.** The obvious approach - average all
frames sharing a time of day (or a day of year) and subtract - fails whenever the record is
shorter than the longest cycle. Measured on the `seasonal_diurnal_sequence` benchmark, which
contains *only* deterministic cycles and 0.05-amplitude noise: removing the time-of-day
climatology left **15.5% of the original variance**, because a 40-day record cannot form a
day-of-year climatology and the annual cycle passed straight through. That residual would
have been mined as weather. Harmonic regression fits the cycles as continuous functions of
time and so removes a partial annual cycle from a partial year.

**The solve is deterministic by construction, not by luck.** A harmonic basis containing a
period longer than the record is close to rank-deficient - for a 40-day record with two
annual harmonics the condition number is 8.4e13 and the smallest singular value is 2.5e-13.
``torch.linalg.lstsq``'s default driver makes its own rank decision there, and that decision
was observed to **flip depending on what had run before it in the same process**: the same
data, the same seed, gave a residual of 0.000172 in one ordering and 0.157639 in another.
A climatology whose result depends on the order of unrelated work is not reproducible, so
the rank decision is made here explicitly - columns are normalised, the solve goes through
an SVD pseudo-inverse with a stated tolerance, and the effective rank and condition number
are reported. When the basis *is* rank-deficient the caller is told which components are not
identifiable from the record rather than being handed a silently pseudo-inverted answer.

**Fitting is split-aware (R6).** A climatology fitted on all the data and then subtracted
from all the data leaks test-period information into the training anomalies. ``fit_mask``
restricts estimation to the training frames; the fitted coefficients are then applied
everywhere, and which frames were used is recorded in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

#: Mean tropical year in hours (365.2422 days), and the solar day.
HOURS_PER_YEAR = 365.2422 * 24.0
HOURS_PER_DAY = 24.0

DEFAULT_PERIODS_HOURS: Tuple[float, ...] = (HOURS_PER_DAY, HOURS_PER_YEAR)


class ClimatologyError(ValueError):
    """Raised when a climatology cannot be estimated honestly from what was supplied."""


@dataclass
class StreamingHarmonicClimatology:
    """Train-fitted harmonic coefficients with a bounded per-frame anomaly reader."""

    frame_reader: Callable[[int], Any]
    design_normalised: torch.Tensor
    solution: torch.Tensor
    reference: Dict[str, Any]
    provenance: Dict[str, Any]

    def anomaly(self, index: int):
        """Read one source frame and subtract the train-fitted climatology exactly."""
        from src.physical_core.field import PhysicalField

        if isinstance(index, bool) or int(index) != index \
                or not 0 <= int(index) < int(self.design_normalised.shape[0]):
            raise ClimatologyError(
                "frame index %r is outside 0..%d"
                % (index, int(self.design_normalised.shape[0]) - 1))
        frame = self.frame_reader(int(index))
        _validate_stream_frame(frame, self.reference, int(index))
        row = self.design_normalised[int(index)].to(self.solution.device)
        climatology = torch.einsum("p,phw->hw", row, self.solution)
        anomaly = frame.data.to(self.solution.dtype) - climatology
        return PhysicalField(
            anomaly,
            coords=frame.coords,
            metadata={
                **frame.metadata,
                "anomaly": "train-fitted harmonic climatology removed",
                "climatology_fit_sha256": self.provenance["fit_sha256"],
            },
            split=frame.split,
            grid=frame.grid,
            units=frame.units,
        )

    def summary(self) -> Dict[str, Any]:
        return dict(self.provenance)


def _stream_frame_identity(frame: Any) -> Dict[str, Any]:
    from src.physical_core.field import PhysicalField

    if not isinstance(frame, PhysicalField):
        raise ClimatologyError("frame reader returned %s, expected PhysicalField"
                               % type(frame).__name__)
    coords = {
        name: np.asarray(value.detach().cpu(), dtype=np.float64).tolist()
        for name, value in sorted(frame.coords.items())
    }
    return {
        "shape": list(frame.data.shape),
        "dtype": str(frame.data.dtype),
        "grid": frame.grid.to_provenance(),
        "coords": coords,
        "variable": frame.metadata.get("variable"),
        "level": frame.metadata.get("level"),
        "units": frame.units,
    }


def _validate_stream_frame(frame: Any, reference: Dict[str, Any], index: int) -> None:
    observed = _stream_frame_identity(frame)
    if observed != reference:
        raise ClimatologyError(
            "frame %d identity differs from the fitted source: expected %s, observed %s"
            % (index, reference, observed))
    if not bool(torch.isfinite(frame.data).all()):
        raise ClimatologyError(
            "frame %d contains non-finite values; climatology fitting never imputes" % index)


def harmonic_design_matrix(
    times_hours: np.ndarray,
    periods_hours: Sequence[float] = DEFAULT_PERIODS_HOURS,
    n_harmonics: int = 2,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, list]:
    """Design matrix ``[1, sin/cos(2 pi k t / P), ...]`` for each period and harmonic.

    Returns the matrix and a list of column labels, because a regression whose columns are
    anonymous cannot be audited.
    """
    if n_harmonics < 1:
        raise ClimatologyError("n_harmonics must be >= 1, got %r" % (n_harmonics,))
    t = np.asarray(times_hours, dtype=np.float64)
    columns = [np.ones_like(t)]
    labels = ["intercept"]
    for period in periods_hours:
        if period <= 0:
            raise ClimatologyError("periods must be positive hours, got %r" % (period,))
        for k in range(1, n_harmonics + 1):
            omega = 2.0 * math.pi * k / period
            columns.append(np.sin(omega * t))
            columns.append(np.cos(omega * t))
            labels.append("sin(%d x %.4gh)" % (k, period))
            labels.append("cos(%d x %.4gh)" % (k, period))
    return torch.as_tensor(np.stack(columns, axis=1), dtype=dtype), labels


def fit_harmonic_climatology_stream(
    frame_reader: Callable[[int], Any],
    times_hours: Sequence[float],
    *,
    fit_mask: Sequence[bool],
    periods_hours: Sequence[float] = DEFAULT_PERIODS_HOURS,
    n_harmonics: int = 2,
    rank_rtol: float = 1e-10,
    dtype: torch.dtype = torch.float64,
) -> StreamingHarmonicClimatology:
    """Fit on declared training frames while retaining one physical field at a time.

    The small harmonic design matrix and ``(parameters,H,W)`` coefficient tensor remain in
    memory; the ``(time,H,W)`` atmospheric record never does.  The arithmetic uses the same
    explicitly truncated SVD pseudo-inverse as :func:`remove_climatology`, accumulated one
    selected frame at a time, so bounded execution does not create a second scientific
    definition of the climatology.
    """
    if not callable(frame_reader):
        raise ClimatologyError("frame_reader must be callable")
    times = np.asarray(times_hours, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)):
        raise ClimatologyError("times_hours must be a non-empty finite 1D coordinate")
    selected = np.asarray(fit_mask, dtype=bool)
    if selected.shape != times.shape:
        raise ClimatologyError(
            "fit_mask has shape %r but times_hours has shape %r"
            % (selected.shape, times.shape))
    fit_idx = np.flatnonzero(selected)
    if fit_idx.size == 0:
        raise ClimatologyError("fit_mask selects no frames; nothing to fit on")

    design, labels = harmonic_design_matrix(times, periods_hours, n_harmonics, dtype)
    n_params = int(design.shape[1])
    if fit_idx.size < 3 * n_params:
        raise ClimatologyError(
            "cannot fit a %d-parameter climatology from %d frames. Fitting needs at least "
            "3x the parameter count to avoid absorbing the signal you are trying to keep"
            % (n_params, fit_idx.size))
    if not math.isfinite(rank_rtol) or rank_rtol <= 0:
        raise ClimatologyError("rank_rtol must be finite and positive")

    col_norm = torch.linalg.vector_norm(design, dim=0, keepdim=True)
    col_norm = torch.where(col_norm > 0, col_norm, torch.ones_like(col_norm))
    design_n = design / col_norm
    fit_design = design_n[fit_idx]
    u, sv, vh = torch.linalg.svd(fit_design, full_matrices=False)
    cutoff = float(sv[0]) * rank_rtol if sv.numel() and float(sv[0]) > 0 else 0.0
    keep = sv > cutoff
    rank = int(keep.sum())
    condition = float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf")
    sv_inv = torch.zeros_like(sv)
    sv_inv[keep] = 1.0 / sv[keep]
    pinv = vh.transpose(-2, -1) @ torch.diag(sv_inv) @ u.transpose(-2, -1)

    first = frame_reader(int(fit_idx[0]))
    reference = _stream_frame_identity(first)
    _validate_stream_frame(first, reference, int(fit_idx[0]))
    height, width = (int(value) for value in first.data.shape)
    solution = torch.zeros((n_params, height, width), dtype=dtype,
                           device=first.data.device)
    source_digest = hashlib.sha256()
    source_digest.update(np.ascontiguousarray(times).tobytes())
    source_digest.update(np.ascontiguousarray(fit_idx).tobytes())

    for position, frame_index in enumerate(fit_idx):
        frame = first if position == 0 else frame_reader(int(frame_index))
        _validate_stream_frame(frame, reference, int(frame_index))
        values = frame.data.to(dtype)
        source_digest.update(np.ascontiguousarray(
            values.detach().cpu().numpy()).tobytes())
        solution += pinv[:, position].to(values.device)[:, None, None] * values[None, :, :]

    coefficient_bytes = np.ascontiguousarray(solution.detach().cpu().numpy()).tobytes()
    coefficient_sha256 = hashlib.sha256(coefficient_bytes).hexdigest()
    warnings = []
    span = float(times.max() - times.min())
    for period in periods_hours:
        if span < 0.5 * period:
            warnings.append(
                "the record spans %.1f h, less than half of the %.4g h period; that cycle "
                "is only partially observed and its removal is an extrapolation"
                % (span, period))
    if rank < n_params:
        warnings.append(
            "the climatology basis is rank-deficient: %d of %d columns are dropped at "
            "tolerance %.1e (condition number %.2e)"
            % (n_params - rank, n_params, rank_rtol, condition))

    identity = {
        "schema": "streaming-harmonic-climatology/v1",
        "times_hours_sha256": hashlib.sha256(
            np.ascontiguousarray(times).tobytes()).hexdigest(),
        "fit_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(fit_idx).tobytes()).hexdigest(),
        "fit_source_stream_sha256": source_digest.hexdigest(),
        "coefficient_sha256": coefficient_sha256,
        "periods_hours": [float(value) for value in periods_hours],
        "n_harmonics": int(n_harmonics),
        "rank_rtol": float(rank_rtol),
        "dtype": str(dtype),
        "reference": reference,
    }
    fit_sha256 = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=str).encode("utf-8")).hexdigest()
    provenance = {
        **identity,
        "fit_sha256": fit_sha256,
        "design_columns": labels,
        "n_parameters": n_params,
        "effective_rank": rank,
        "condition_number": condition,
        "fitted_on_frames": int(fit_idx.size),
        "fitted_on_all_frames": bool(fit_idx.size == times.size),
        "total_frames": int(times.size),
        "source_frames_resident": 1,
        "coefficient_shape": list(solution.shape),
        "coefficient_bytes": len(coefficient_bytes),
        "warnings": warnings,
        "claim_boundary": "climatology fit only; no cross-scale relationship or skill",
    }
    return StreamingHarmonicClimatology(
        frame_reader=frame_reader,
        design_normalised=design_n,
        solution=solution,
        reference=reference,
        provenance=provenance,
    )


def remove_climatology(
    stack: torch.Tensor,
    times_hours: np.ndarray,
    periods_hours: Sequence[float] = DEFAULT_PERIODS_HOURS,
    n_harmonics: int = 2,
    fit_mask: Optional[np.ndarray] = None,
    rank_rtol: float = 1e-10,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Fit and subtract a harmonic climatology from a ``(T, H, W)`` stack.

    Parameters
    ----------
    fit_mask
        Boolean array over time. When given, the climatology is estimated **only** from
        these frames (rule R6: no test-period information in the training anomalies) and
        then applied to all of them.
    rank_rtol
        Singular values below ``rank_rtol * max(sv)`` are treated as null directions and
        dropped. Making this explicit is what makes the fit reproducible; see the module
        docstring.

    Returns a dict with ``anomalies``, ``climatology``, the fraction of variance the
    climatology explained, the column labels, and the frames it was fitted on.
    """
    if stack.dim() != 3:
        raise ClimatologyError(
            "expected a (time, height, width) stack, got shape %r" % (tuple(stack.shape),))
    T, H, W = stack.shape
    t = np.asarray(times_hours, dtype=np.float64)
    if t.shape[0] != T:
        raise ClimatologyError(
            "times_hours has %d entries but the stack has %d frames" % (t.shape[0], T))

    design, labels = harmonic_design_matrix(t, periods_hours, n_harmonics, dtype)
    n_params = design.shape[1]

    if fit_mask is None:
        fit_idx = np.arange(T)
    else:
        fit_idx = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
        if fit_idx.size == 0:
            raise ClimatologyError("fit_mask selects no frames; nothing to fit on")

    if fit_idx.size < 3 * n_params:
        raise ClimatologyError(
            "cannot fit a %d-parameter climatology from %d frames. Fitting needs at least "
            "3x the parameter count to avoid absorbing the signal you are trying to keep; "
            "reduce n_harmonics (currently %d), drop a period from %r, or supply a longer "
            "record." % (n_params, fit_idx.size, n_harmonics, tuple(periods_hours)))

    # A period far longer than the record is not estimable: sin and cos over a small phase
    # arc are nearly collinear with the intercept, so the fit will happily absorb a trend
    # that is not actually the cycle. Warn rather than refuse - a partial annual cycle is
    # still worth removing - but say so.
    span = float(t.max() - t.min())
    warnings = []
    for period in periods_hours:
        if span < 0.5 * period:
            warnings.append(
                "the record spans %.1f h, less than half of the %.4g h period; that cycle "
                "is only partially observed and its removal is an extrapolation. The "
                "anomalies are still usable, but a trend in the data can be absorbed into "
                "this term." % (span, period))

    work = stack.to(dtype).reshape(T, H * W)

    # Normalise columns before solving. Harmonic columns have wildly different norms when a
    # period exceeds the record, and rescaling them is mathematically free (the coefficients
    # absorb it) while improving the conditioning by orders of magnitude.
    col_norm = torch.linalg.vector_norm(design, dim=0, keepdim=True)
    col_norm = torch.where(col_norm > 0, col_norm, torch.ones_like(col_norm))
    design_n = design / col_norm

    fit_design = design_n[fit_idx]
    u, sv, vh = torch.linalg.svd(fit_design, full_matrices=False)
    cutoff = float(sv[0]) * rank_rtol if sv.numel() and float(sv[0]) > 0 else 0.0
    keep = sv > cutoff
    rank = int(keep.sum())
    condition = float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf")

    if rank < n_params:
        warnings.append(
            "the climatology basis is rank-deficient: %d of %d columns are not identifiable "
            "from this record at tolerance %.1e (condition number %.2e). The unidentifiable "
            "directions are dropped rather than pseudo-inverted, which keeps the fit "
            "reproducible; the usual cause is asking for a harmonic of a period the record "
            "does not cover. Reduce n_harmonics or shorten the period list."
            % (n_params - rank, n_params, rank_rtol, condition))

    sv_inv = torch.zeros_like(sv)
    sv_inv[keep] = 1.0 / sv[keep]
    pinv = vh.transpose(-2, -1) @ torch.diag(sv_inv) @ u.transpose(-2, -1)
    solution = pinv @ work[fit_idx]

    climatology = (design_n @ solution).reshape(T, H, W)
    anomalies = stack.to(dtype) - climatology

    raw_var = float(torch.var(stack.to(dtype), unbiased=False))
    resid_var = float(torch.var(anomalies, unbiased=False))
    explained = 1.0 - resid_var / raw_var if raw_var > 0 else 0.0

    return {
        "anomalies": anomalies,
        "climatology": climatology,
        "raw_variance": raw_var,
        "residual_variance": resid_var,
        "residual_variance_ratio": resid_var / raw_var if raw_var > 0 else 0.0,
        "variance_explained_by_climatology": explained,
        "design_columns": labels,
        "n_parameters": n_params,
        "effective_rank": rank,
        "condition_number": condition,
        "rank_rtol": rank_rtol,
        "fitted_on_frames": fit_idx.size,
        "fitted_on_all_frames": fit_mask is None,
        "periods_hours": tuple(float(p) for p in periods_hours),
        "n_harmonics": n_harmonics,
        "warnings": warnings,
    }
