"""Surrogate nulls for *sequences* of fields (roadmap T4C.2, rules R1 and R12).

`src/statistics/surrogates.py` already generates surrogates of an array, and this module does
not reimplement any of that - it calls it. What it adds is the thing Phase 4C actually needs:
a null for a **`FieldSequence`**, where the statistic under test is computed from the whole
record rather than from one frame, and where getting the null's temporal structure wrong is
the difference between a discovery and an artefact.

**Why this is not `analysis_engine/surrogates.py`.** The roadmap names that path. Two modules
called `surrogates` in one codebase is a footgun - `from ... import surrogates` would resolve
to whichever package the reader happened to be in - and it invites the phase-randomisation
core to be forked and drift. The generators stay in one place; this file is the sequence-level
layer above them and is named for what it is.

**The choice of null is the hypothesis, and here it is a choice about time.**

*   `spatiotemporal_phase` (**the default**) randomises the phases of the full 3D transform.
    It preserves the 3D power spectrum exactly, and therefore preserves *both* the spatial
    power spectrum of the record and its **temporal autocorrelation**. This is the null rule
    R12 demands: hourly frames are massively autocorrelated, and a null without that
    autocorrelation is easier to beat than reality.
*   `per_frame_phase` randomises each frame independently. Every frame keeps its own PSD -
    which is what "preserving the per-frame PSD" literally asks for - but the record's
    temporal structure is destroyed along the way. Against this null a red-noise sequence with
    no organisation whatsoever tests as significant, because the *autocorrelation itself*
    beats the null. It is provided because it is the obvious thing to reach for and because
    the failure is worth being able to demonstrate; `test_surrogate_null.py` measures the
    false-positive rate of both on the same data.
*   `circular_time_shift` shifts the record in time, preserving every frame and the full
    spatial structure, and destroying only the alignment between one part of the record and
    another. The right null for a *lagged* claim.

Every result carries the null's identity and what it preserves, because "p = 0.004" means
different things against different nulls and cannot be interpreted without knowing which.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.sequence import FieldSequence
from src.statistics import surrogates as surrogate_module

SEQUENCE_METHODS = ("spatiotemporal_phase", "per_frame_phase", "circular_time_shift")

DEFAULT_ENSEMBLE = 199

PRESERVES: Dict[str, List[str]] = {
    "spatiotemporal_phase": [
        "the full 3D power spectrum, hence the spatial power spectrum of every scale",
        "the temporal autocorrelation of the record (rule R12)",
        "the mean and variance of the record",
    ],
    "per_frame_phase": [
        "the power spectrum of each frame taken alone",
        "the mean and variance of each frame",
    ],
    "circular_time_shift": [
        "every frame exactly, and so the entire spatial structure",
        "the autocorrelation of the record up to the wrap point",
    ],
}

DESTROYS: Dict[str, List[str]] = {
    "spatiotemporal_phase": [
        "phase organisation: localised structures, cross-scale alignment, coherent features",
    ],
    "per_frame_phase": [
        "phase organisation within each frame",
        "**the temporal autocorrelation of the record** - which is why this null is "
        "anti-conservative for any statistic that depends on time",
    ],
    "circular_time_shift": [
        "the alignment between the record and itself at a lag",
    ],
}


class SurrogateNullError(ValueError):
    """Raised when a sequence-level surrogate cannot be generated faithfully."""


# ---------------------------------------------------------------- generation

def _stack(sequence: FieldSequence) -> np.ndarray:
    return sequence.to_tensor().to(torch.float64).numpy()


def _rebuild(sequence: FieldSequence, data: np.ndarray, label: str) -> FieldSequence:
    fields = [PhysicalField(torch.as_tensor(frame, dtype=torch.float64),
                            grid=sequence.grid,
                            metadata={"surrogate": label})
              for frame in data]
    return FieldSequence(fields, sequence.times_seconds,
                         metadata={"surrogate": label,
                                   "surrogate_of": getattr(sequence, "metadata", {}) or {}})


def sequence_surrogate(sequence: FieldSequence, method: str = "spatiotemporal_phase",
                       seed: Optional[int] = None) -> FieldSequence:
    """One surrogate of a whole record, as a `FieldSequence` with the same time axis.

    The time axis is carried through unchanged and deliberately: a surrogate is a null for
    the *values*, not for when they were observed, and a statistic computed at a lag has to
    see the same cadence on the null as on the data or the comparison is not like for like.
    """
    if method not in SEQUENCE_METHODS:
        raise InvalidParameterError(
            "method", method,
            "one of %s. Frame-level array methods such as 'aaft' live in "
            "src.statistics.surrogates and apply to a single array" % (list(SEQUENCE_METHODS),))

    data = _stack(sequence)
    if method == "spatiotemporal_phase":
        return _rebuild(sequence, surrogate_module.phase_randomise(data, seed=seed), method)

    if method == "per_frame_phase":
        rng = np.random.default_rng(seed)
        frames = [surrogate_module.phase_randomise(frame, seed=int(rng.integers(0, 2 ** 31 - 1)))
                  for frame in data]
        return _rebuild(sequence, np.stack(frames, axis=0), method)

    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if n < 3:
        raise SurrogateNullError(
            "a circular time shift needs at least 3 frames to produce a shift that is "
            "neither zero nor a reflection; this record has %d" % n)
    shift = int(rng.integers(1, n))
    return _rebuild(sequence, np.roll(data, shift, axis=0), method)


def verify_surrogate(source: FieldSequence, surrogate: FieldSequence) -> Dict[str, Any]:
    """The acceptance check of T4C.2, as a function rather than only as a test.

    Reports the relative error between the source and surrogate power spectra - which a phase
    randomisation preserves to floating-point - and the correlation between them frame by
    frame, which measures whether the phase organisation was actually destroyed. A surrogate
    that matches the spectrum *and* still correlates with the source has not randomised
    anything, and would make every test that used it trivially conservative.
    """
    a = _stack(source)
    b = _stack(surrogate)
    if a.shape != b.shape:
        raise SurrogateNullError("surrogate shape %r does not match source %r"
                                 % (b.shape, a.shape))

    spec_a = np.abs(np.fft.fftn(a)) ** 2
    spec_b = np.abs(np.fft.fftn(b)) ** 2
    denom = float(spec_a.sum())
    psd_error = float(np.abs(spec_a - spec_b).sum() / denom) if denom > 0 else float("nan")

    per_frame_error = []
    correlations = []
    for frame_a, frame_b in zip(a, b):
        fa = np.abs(np.fft.fftn(frame_a)) ** 2
        fb = np.abs(np.fft.fftn(frame_b)) ** 2
        total = float(fa.sum())
        per_frame_error.append(float(np.abs(fa - fb).sum() / total) if total > 0
                               else float("nan"))
        x = frame_a.ravel() - frame_a.mean()
        y = frame_b.ravel() - frame_b.mean()
        norm = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
        correlations.append(float((x * y).sum() / norm) if norm > 0 else float("nan"))

    return {
        "psd_relative_error": psd_error,
        "per_frame_psd_relative_error": per_frame_error,
        "max_per_frame_psd_relative_error": float(np.nanmax(per_frame_error)),
        "frame_correlation": correlations,
        "max_abs_frame_correlation": float(np.nanmax(np.abs(correlations))),
        "note": ("psd_relative_error is the L1 error of the 3D spectrum; a phase "
                 "randomisation preserves it to floating point. per_frame values are the "
                 "spectrum of each frame taken alone, which only per_frame_phase preserves "
                 "exactly."),
    }


# ---------------------------------------------------------------- the ensemble

class SurrogateNull:
    """An ensemble of surrogate sequences, and the empirical null it defines.

    Built once and reused, because generating two hundred surrogates of a record is the
    expensive part and several statistics are usually tested against the same null.
    """

    def __init__(self, sequence: FieldSequence, method: str = "spatiotemporal_phase",
                 n: int = DEFAULT_ENSEMBLE, seed: int = 20260821) -> None:
        if n < 1:
            raise InvalidParameterError(
                "n", n, "at least one surrogate. An empirical p-value floors at 1/(1+n), so "
                        "an ensemble of n cannot resolve anything finer than that")
        if method not in SEQUENCE_METHODS:
            raise InvalidParameterError("method", method, "one of %s" % (list(SEQUENCE_METHODS),))
        self.sequence = sequence
        self.method = method
        self.n = int(n)
        self.seed = int(seed)
        self._members: Optional[List[FieldSequence]] = None

    @property
    def members(self) -> List[FieldSequence]:
        if self._members is None:
            rng = np.random.default_rng(self.seed)
            self._members = [
                sequence_surrogate(self.sequence, self.method,
                                   seed=int(rng.integers(0, 2 ** 31 - 1)))
                for _ in range(self.n)]
        return self._members

    def p_value_floor(self) -> float:
        return 1.0 / (1.0 + self.n)

    def test(self, statistic: Callable[[FieldSequence], float],
             alternative: str = "greater", label: str = "statistic") -> Dict[str, Any]:
        """Compare `statistic(observed)` against the ensemble, reporting the null's identity.

        The p-value is `(1 + k) / (1 + n)`, never `k / n`: the added one keeps the test exact
        in finite samples and stops a `p = 0` that no finite ensemble can support.
        """
        if alternative not in ("greater", "less", "two_sided"):
            raise InvalidParameterError("alternative", alternative,
                                        "'greater', 'less' or 'two_sided'")
        observed = float(statistic(self.sequence))
        null = np.asarray([float(statistic(member)) for member in self.members],
                          dtype=np.float64)
        finite = null[np.isfinite(null)]
        if finite.size == 0 or not np.isfinite(observed):
            return {
                "label": label, "observed": observed, "p_value": float("nan"),
                "surrogate_method": self.method, "n_surrogates": int(null.size),
                "valid": False,
                "warnings": ["the statistic was not finite on the data or on any surrogate, "
                             "so no comparison was possible"],
            }

        if alternative == "greater":
            k = int(np.sum(finite >= observed))
        elif alternative == "less":
            k = int(np.sum(finite <= observed))
        else:
            centre = float(finite.mean())
            k = int(np.sum(np.abs(finite - centre) >= abs(observed - centre)))
        p_value = (1.0 + k) / (1.0 + finite.size)
        std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0

        return {
            "label": label,
            "observed": observed,
            "p_value": float(p_value),
            "p_value_floor": 1.0 / (1.0 + finite.size),
            "surrogate_mean": float(finite.mean()),
            "surrogate_std": std,
            "effect_size": ((observed - float(finite.mean())) / std if std > 0
                            else float("nan")),
            "n_surrogates": int(finite.size),
            "n_nonfinite_surrogates": int(null.size - finite.size),
            "alternative": alternative,
            "surrogate_method": self.method,
            "null_preserves": list(PRESERVES[self.method]),
            "null_destroys": list(DESTROYS[self.method]),
            "null_hypothesis": (
                "the observed statistic is no more extreme than expected for a record "
                "preserving: %s" % "; ".join(PRESERVES[self.method])),
            "root_seed": self.seed,
            "valid": True,
            "warnings": ([] if self.method != "per_frame_phase" else [
                "per_frame_phase destroys the record's temporal autocorrelation, so any "
                "statistic that depends on time is being compared against a null that is "
                "easier to beat than reality (rule R12). Use spatiotemporal_phase unless the "
                "statistic is computed frame by frame."]),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "type": "SurrogateNull",
            "method": self.method,
            "n_surrogates": self.n,
            "p_value_floor": self.p_value_floor(),
            "preserves": list(PRESERVES[self.method]),
            "destroys": list(DESTROYS[self.method]),
            "root_seed": self.seed,
            "n_frames": len(self.sequence),
        }
