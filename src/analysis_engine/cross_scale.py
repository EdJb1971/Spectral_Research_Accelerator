"""Cross-scale lagged dependency over `A_t(s)` (roadmap T4C.3, rules R4, R1, R5, R12).

The research question - *do small structures at `t` precede large ones at `t + dt`* - becomes,
once a `ScaleSignature` exists, a question about one small matrix: for every ordered pair of
scales and every admissible lag, does scale `s` at `t` carry information about scale `s'` at
`t + lag` beyond what `s'` already tells about itself?

Two estimators, because they answer different questions:

*   **Lagged mutual information** `I(A_t(s) ; A_{t+lag}(s'))` - any dependence at all,
    including dependence that is entirely explained by each series' own autocorrelation. It
    is the more sensitive and the more misleading of the two.
*   **Transfer entropy** `I(A_{t+lag}(s') ; A_t(s) | A_t(s'))` - dependence *beyond* the
    target's own past. This is the one that speaks to precedence, and it is why the target's
    present is conditioned on rather than ignored.

Both are estimated from equiprobable bins with the Miller-Madow correction. Both are still
biased upward on short records - that is a property of every plug-in estimator, not a
shortcoming of this one - so **neither is reported as a raw value**. Every number here is
reported as an excess over a surrogate ensemble that has the same length, the same bin count
and the same marginal distributions, so the estimator's bias is present on both sides and
cancels.

**Three ways this can produce a confident wrong answer, and what is done about each.**

1.  **The transform's own support (rule R4).** A coarse coefficient and the fine coefficients
    beneath it are computed from the same pixels, so they are coupled by construction. The
    spatial transform used here has *no temporal support at all* - it is applied frame by
    frame - so the honest floor is not a filter width in seconds but the time the field needs
    to advect across that filter width. Below it, "fine precedes coarse" is the same air seen
    twice through overlapping filters. `support_floor` computes it from the grid spacing and a
    declared advection speed, and refuses to invent the speed.
2.  **Circularity (measured, not assumed).** Fourier-transform surrogates are circularly
    stationary; a record is not. A lagged statistic computed on the record as a *line* is
    therefore systematically larger than the same statistic on the surrogates, and rejects
    almost always. Measured on 20 AR(1) records against a phase-randomised null: **20 of 20
    false rejections with a linear lag, 0 of 20 with a circular one** (median p 0.010 against
    0.485). Lags therefore wrap by default, and the fraction of pairs that come from the wrap
    is reported so the dilution is visible.
3.  **Multiplicity (rule R5).** Every scale pair at every lag is a test, and a five-scale
    signature over eight lags is 160 of them. The family is corrected together, by
    Benjamini-Yekutieli, whose dependence assumption is the one that holds here.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.core.errors import InvalidParameterError
from src.statistics.multiple_comparisons import check_power
from src.statistics.significance import screen

DEFAULT_BINS = 6

DEFAULT_SHIFT_SURROGATES = 199

MIN_SAMPLES_PER_CELL = 5.0


@dataclass(frozen=True)
class GateProtocol:
    """Frozen design for the T4C.6 train/test replication gate.

    The protocol is deliberately independent of a particular cloud store. Data acquisition
    may change when a better chunk layout becomes available; the hypothesis family, power,
    temporal split and decision rule must not change after results are seen.
    """

    study_id: str
    n_scales: int
    lags: tuple
    expected_frames: int
    cadence_seconds: float = 21600.0
    train_ratio: float = 0.6
    embargo_frames: int = 8
    estimator: str = "transfer_entropy"
    measure: str = "energy_density"
    bins: int = DEFAULT_BINS
    n_surrogates: int = 4999
    alpha: float = 0.05
    correction: str = "benjamini_yekutieli"
    require_advection_floor: bool = True
    seed: int = 20260821

    @property
    def family_size(self) -> int:
        return self.n_scales * (self.n_scales - 1) * len(self.lags)

    @property
    def samples_per_joint_cell_required(self) -> int:
        dimensions = 3 if self.estimator == "transfer_entropy" else 2
        return int((self.bins ** dimensions) * MIN_SAMPLES_PER_CELL)

    def validate(self) -> Dict[str, Any]:
        """Refuse a design incapable of producing an interpretable PASS or FAIL."""
        if not self.study_id.strip():
            raise InvalidParameterError("study_id", self.study_id, "a non-empty identifier")
        if isinstance(self.n_scales, bool) or int(self.n_scales) != self.n_scales \
                or self.n_scales < 2:
            raise InvalidParameterError("n_scales", self.n_scales, "at least two scales")
        if (not self.lags or any(isinstance(v, bool) or int(v) != v or int(v) < 1
                                 for v in self.lags)
                or tuple(sorted(set(int(v) for v in self.lags))) != tuple(self.lags)):
            raise InvalidParameterError(
                "lags", self.lags, "strictly increasing unique positive integer frame lags")
        if self.embargo_frames < max(int(v) for v in self.lags):
            raise InvalidParameterError(
                "embargo_frames", self.embargo_frames,
                "at least the longest tested lag (%d frames), so a target cannot cross a "
                "temporal split boundary" % max(int(v) for v in self.lags))
        if not math.isfinite(self.cadence_seconds) or self.cadence_seconds <= 0:
            raise InvalidParameterError("cadence_seconds", self.cadence_seconds,
                                        "a finite positive physical cadence")
        if not math.isfinite(self.train_ratio) or not 0.0 < self.train_ratio < 1.0:
            raise InvalidParameterError("train_ratio", self.train_ratio, "a fraction in (0, 1)")
        if isinstance(self.expected_frames, bool) or int(self.expected_frames) != self.expected_frames \
                or self.expected_frames < 1:
            raise InvalidParameterError("expected_frames", self.expected_frames,
                                        "a positive frame count")
        if self.estimator not in ("transfer_entropy", "mutual_information"):
            raise InvalidParameterError("estimator", self.estimator,
                                        "'transfer_entropy' or 'mutual_information'")
        if self.measure not in (
                "energy_density", "energy_fraction", "participation_ratio", "gini"):
            raise InvalidParameterError(
                "measure", self.measure,
                "a threshold-free scale measure; threshold_fraction is never a gate primary")
        if isinstance(self.bins, bool) or int(self.bins) != self.bins or self.bins < 2:
            raise InvalidParameterError("bins", self.bins, "an integer >= 2")
        if isinstance(self.n_surrogates, bool) or int(self.n_surrogates) != self.n_surrogates \
                or self.n_surrogates < 1:
            raise InvalidParameterError("n_surrogates", self.n_surrogates,
                                        "a positive integer ensemble size")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise InvalidParameterError("alpha", self.alpha, "a finite probability in (0, 1)")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise InvalidParameterError("seed", self.seed, "a non-negative integer")

        train_frames = int(self.expected_frames * self.train_ratio)
        test_frames = self.expected_frames - train_frames - self.embargo_frames
        required = self.samples_per_joint_cell_required
        if min(train_frames, test_frames) < required:
            raise InvalidParameterError(
                "expected_frames", self.expected_frames,
                "enough frames that both independent partitions have at least %d samples "
                "(%d bins across the estimator's joint cells at %.0f samples/cell). This "
                "design gives train=%d and test=%d after a %d-frame embargo"
                % (required, self.bins, MIN_SAMPLES_PER_CELL, train_frames, test_frames,
                   self.embargo_frames))

        power = check_power(self.n_surrogates, self.family_size, alpha=self.alpha,
                            method=self.correction)
        if not power["can_reject_after_correction"]:
            raise InvalidParameterError(
                "n_surrogates", self.n_surrogates,
                power["warning"] or "enough surrogates to reject after correction")
        return {
            "family_size": self.family_size,
            "train_frames": train_frames,
            "test_frames": test_frames,
            "embargo_frames": self.embargo_frames,
            "minimum_frames_per_partition": required,
            "power": power,
            "fingerprint": self.fingerprint(),
        }

    def to_mapping(self) -> Dict[str, Any]:
        mapping = asdict(self)
        mapping["lags"] = list(self.lags)
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "GateProtocol":
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        if missing or extra:
            raise InvalidParameterError(
                "gate protocol keys", sorted(mapping),
                "exact keys; missing=%s unknown=%s" % (missing, extra))
        values = dict(mapping)
        values["lags"] = tuple(values["lags"])
        protocol = cls(**values)
        protocol.validate()
        return protocol

    def fingerprint(self) -> str:
        """Stable identity for the exact protocol placed beside every gate result."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                             default=list).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def evaluate_replication_gate(train: Dict[str, Any], test: Dict[str, Any],
                              protocol: GateProtocol) -> Dict[str, Any]:
    """PASS only relationships that replicate under the frozen design.

    FAIL is a scientifically valid null result. INVALID means the run cannot adjudicate the
    question (configuration drift, no geometric support floor, or inadequate surrogate
    power) and must never be displayed as a negative finding.
    """
    design = protocol.validate()
    problems: List[str] = []
    for split_name, result in (("train", train), ("test", test)):
        if result.get("protocol_fingerprint") != design["fingerprint"]:
            problems.append("%s result is not bound to the frozen protocol" % split_name)
        if result.get("estimator") != protocol.estimator:
            problems.append("%s estimator is %r, expected %r" %
                            (split_name, result.get("estimator"), protocol.estimator))
        if result.get("measure") != protocol.measure:
            problems.append("%s measure is %r, expected %r" %
                            (split_name, result.get("measure"), protocol.measure))
        if [int(v) for v in result.get("lags_frames", [])] != [int(v) for v in protocol.lags]:
            problems.append("%s lag family differs from the frozen protocol" % split_name)
        if int(result.get("n_tests", -1)) != protocol.family_size:
            problems.append("%s tested %r hypotheses, expected the declared family of %d" %
                            (split_name, result.get("n_tests"), protocol.family_size))
        if int(result.get("bins", -1)) != protocol.bins:
            problems.append("%s bin count differs from the frozen protocol" % split_name)
        correction = result.get("correction") or {}
        if correction.get("method") != protocol.correction:
            problems.append("%s correction differs from the frozen protocol" % split_name)
        if not math.isclose(float(result.get("alpha", float("nan"))), protocol.alpha,
                            rel_tol=0.0, abs_tol=0.0):
            problems.append("%s alpha differs from the frozen protocol" % split_name)
        if int(result.get("n_surrogates_requested", -1)) != protocol.n_surrogates:
            problems.append("%s surrogate count differs from the frozen protocol" % split_name)
        if not (result.get("power") or {}).get("can_reject_after_correction", False):
            problems.append("%s partition was underpowered after correction" % split_name)
        if protocol.require_advection_floor and not (
                result.get("support_floor") or {}).get("enforced", False):
            problems.append("%s partition did not enforce the declared advection floor" %
                            split_name)

    if problems:
        return {"verdict": "INVALID", "problems": problems,
                "protocol_fingerprint": design["fingerprint"], "replicated": []}

    def positive(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            str(row["label"]): row for row in rows
            if bool(row.get("significant"))
            and float(row.get("q_value", 1.0)) <= protocol.alpha
            and float(row.get("excess_nats", 0.0)) > 0.0
        }

    train_positive = positive(train.get("results", []))
    test_positive = positive(test.get("results", []))
    labels = sorted(set(train_positive) & set(test_positive))
    replicated = [{"label": label,
                   "train": train_positive[label],
                   "test": test_positive[label]}
                  for label in labels]
    return {
        "verdict": "PASS" if replicated else "FAIL",
        "problems": [],
        "protocol_fingerprint": design["fingerprint"],
        "replicated": replicated,
        "decision_rule": ("PASS requires the same positive, q<=alpha relationship in the "
                          "independent train and test partitions; an adequately powered "
                          "absence is FAIL, never INVALID"),
    }


class CrossScaleError(ValueError):
    """Raised when a dependency estimate would not mean what it appears to mean."""


# ---------------------------------------------------------------- information estimators

def _quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    """Equiprobable binning: each bin holds about the same number of samples.

    Equal-width bins on a heavy-tailed quantity - which coefficient energy always is - put
    almost every sample in the first bin and leave the rest nearly empty, so the estimate
    collapses towards zero regardless of the dependence. Equiprobable bins keep the marginal
    entropy at its maximum, which is also what makes the surrogate comparison clean: a
    circular shift changes the alignment without changing the marginals at all.
    """
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # Ties can collapse edges; np.digitize then silently merges bins. Made visible by
    # returning fewer distinct codes rather than by pretending there were `bins` of them.
    return np.digitize(x, edges[1:-1], right=False)


def _entropy(counts: np.ndarray, n: int) -> float:
    """Plug-in entropy in nats with the Miller-Madow bias correction.

    The correction adds `(m - 1) / (2n)` for `m` **occupied** cells, which is the leading
    term of the plug-in estimator's downward bias. It is small, cheap and well understood; it
    does not make the estimate unbiased, which is why every value here is still reported
    against a surrogate ensemble rather than on its own.
    """
    occupied = counts[counts > 0]
    if occupied.size == 0 or n <= 0:
        return 0.0
    p = occupied / n
    return float(-(p * np.log(p)).sum() + (occupied.size - 1) / (2.0 * n))


def _joint_entropy(*codes: np.ndarray) -> float:
    stacked = np.stack(codes, axis=0)
    n = stacked.shape[1]
    # A single integer key per joint cell, so the histogram is exact rather than binned again.
    key = np.zeros(n, dtype=np.int64)
    for row in stacked:
        key = key * (int(row.max()) + 1) + row
    counts = np.bincount(key)
    return _entropy(counts, n)


def mutual_information(x: Sequence[float], y: Sequence[float],
                       bins: int = DEFAULT_BINS) -> float:
    """`I(X ; Y)` in nats, from equiprobable bins with a Miller-Madow correction.

    Non-negative by construction in the population; the estimate is clipped at zero, because
    a negative mutual information is an artefact of the correction and reporting it would
    invite it to be read as "anti-dependence", which does not exist.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise CrossScaleError("x and y must be the same length, got %d and %d"
                              % (xa.size, ya.size))
    finite = np.isfinite(xa) & np.isfinite(ya)
    if int(finite.sum()) < 4:
        return float("nan")
    cx = _quantile_bins(xa[finite], bins)
    cy = _quantile_bins(ya[finite], bins)
    value = _joint_entropy(cx) + _joint_entropy(cy) - _joint_entropy(cx, cy)
    return float(max(value, 0.0))


def transfer_entropy(source: Sequence[float], target: Sequence[float], lag: int,
                     bins: int = DEFAULT_BINS, wrap: bool = True) -> float:
    """`I(target_{t+lag} ; source_t | target_t)` in nats.

    Conditioning on `target_t` is the whole point: without it, a target with a long
    autocorrelation shows dependence on *anything* that shares its timescale, including
    itself. With it, the quantity is the information the source adds beyond the target's own
    present - which is as close to precedence as a statistic of this kind is entitled to get,
    and it is still not causation (rule R7).

    A three-variable joint entropy over `T` samples needs `bins**3` cells; the caller is
    warned about that arithmetic by `cross_scale_dependency`, which checks the samples per
    cell before it reports anything.
    """
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if s.shape != t.shape:
        raise CrossScaleError("source and target must be the same length")
    n = s.size
    if lag < 1:
        raise InvalidParameterError("lag", lag, "a positive number of frames")
    if not wrap and lag >= n - 3:
        return float("nan")

    if wrap:
        future = np.roll(t, -lag)
        present_t, present_s = t, s
    else:
        future, present_t, present_s = t[lag:], t[:-lag], s[:-lag]

    finite = np.isfinite(future) & np.isfinite(present_t) & np.isfinite(present_s)
    if int(finite.sum()) < 6:
        return float("nan")
    f = _quantile_bins(future[finite], bins)
    pt = _quantile_bins(present_t[finite], bins)
    ps = _quantile_bins(present_s[finite], bins)

    value = (_joint_entropy(f, pt) + _joint_entropy(ps, pt)
             - _joint_entropy(pt) - _joint_entropy(f, pt, ps))
    return float(max(value, 0.0))


def lagged_mutual_information(source: Sequence[float], target: Sequence[float], lag: int,
                              bins: int = DEFAULT_BINS, wrap: bool = True) -> float:
    """`I(source_t ; target_{t+lag})`, wrapping by default - see the module docstring."""
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if lag < 1:
        raise InvalidParameterError("lag", lag, "a positive number of frames")
    if wrap:
        return mutual_information(s, np.roll(t, -lag), bins=bins)
    if lag >= s.size - 3:
        return float("nan")
    return mutual_information(s[:-lag], t[lag:], bins=bins)


# ---------------------------------------------------------------- the support floor (R4)

def support_floor(signature, cadence_seconds: float,
                  advection_speed_m_s: Optional[float] = None) -> Dict[str, Any]:
    """The minimum admissible lag per scale, and an honest account of where it comes from.

    Rule R4 says a lag shorter than the transform's own support measures filter geometry
    rather than weather. For a **spatial** transform applied frame by frame - which is what
    Phase 4 uses, deliberately, since 3D wavelets are out of scope - the temporal support is
    exactly zero, and quoting one would be an invention. What is real is that a structure
    must cross the filter's spatial support before a change at that scale can be anything
    other than the same air seen twice: `t_cross(s) = support(s) * dx / U`.

    `advection_speed_m_s` is therefore required for a geometric floor and is **not given a
    default**. A plausible-looking 10 m/s would silently set every floor in every result, and
    a reader would have no way to know a number they never supplied was doing the work.
    Without it the only floor applied is one frame, and the result says so in as many words.
    """
    if cadence_seconds <= 0:
        raise InvalidParameterError("cadence_seconds", cadence_seconds,
                                    "a positive sampling interval")
    grid = signature.provenance.get("grid") or {}
    dx = grid.get("dx") or grid.get("representative_dx_metres") or grid.get("dx_metres")
    physical = isinstance(dx, (int, float)) and dx and math.isfinite(dx) and dx > 0

    floors: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for position, scale in enumerate(signature.scales, start=1):
        try:
            level = int(scale)
        except (TypeError, ValueError):
            level = position
        interior_record = (signature.interior[position - 1]
                           if position - 1 < len(signature.interior) else {})
        support_px = interior_record.get("support_parent_px")
        if not isinstance(support_px, (int, float)) or support_px <= 0:
            raise InvalidParameterError(
                "signature.interior[%d].support_parent_px" % (position - 1), support_px,
                "the transform's measured positive parent-grid filter support. Using 2**level "
                "would understate the shared spatial footprint for longer filters")
        record: Dict[str, Any] = {
            "scale": str(scale),
            "level": level,
            "spatial_support_px": support_px,
            "floor_frames": 1,
            "basis": ("sampling cadence only: consecutive frames are the finest lag the "
                      "record can express; spatial support is the transform's exact %d-pixel "
                      "filter cascade" % support_px),
        }
        if physical and advection_speed_m_s:
            support_m = support_px * float(dx)
            crossing = support_m / float(advection_speed_m_s)
            frames = max(1, int(math.ceil(crossing / cadence_seconds)))
            record.update({
                "spatial_support_m": support_m,
                "crossing_time_s": crossing,
                "floor_frames": frames,
                "basis": ("advective crossing of the filter support: %.0f m at %.1f m/s is "
                          "%.0f s, which is %d frame(s) at this cadence"
                          % (support_m, advection_speed_m_s, crossing, frames)),
            })
        floors.append(record)

    if not physical:
        warnings.append(
            "the signature's grid carries no physical spacing, so no advective floor could "
            "be computed and the only floor applied is one frame. A lag floor in metres "
            "derived from a pixel grid would be fabricated.")
    if advection_speed_m_s is None:
        warnings.append(
            "no advection speed was supplied, so the geometric floor of rule R4 is not "
            "enforced. This is reported rather than defaulted: a default speed would set "
            "every floor in every result from a number the reader never chose.")

    return {
        "cadence_seconds": float(cadence_seconds),
        "advection_speed_m_s": (None if advection_speed_m_s is None
                                else float(advection_speed_m_s)),
        "floors": floors,
        "floor_by_scale": {record["scale"]: record["floor_frames"] for record in floors},
        "enforced": bool(physical and advection_speed_m_s),
        "temporal_support_note": (
            "the transform is spatial and is applied frame by frame, so its temporal support "
            "is zero. The floor below is advective, not filter-geometric, and it is the "
            "honest version of rule R4 for a 2D-per-frame decomposition."),
        "warnings": warnings,
    }


# ---------------------------------------------------------------- the dependency sweep

def decorrelation_frames(x: Sequence[float]) -> int:
    """First lag at which the autocorrelation falls below `1/e`, floored at 1.

    Used as a Theiler window, not as a claim about the physics: it is the distance beyond
    which two samples of this series are approximately independent, which is what a shift has
    to exceed before it counts as a shuffle.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 4:
        return 1
    centred = a - a.mean()
    variance = float((centred ** 2).sum())
    if variance <= 0:
        return 1
    threshold = 1.0 / math.e
    for lag in range(1, a.size // 2):
        acf = float((centred[:-lag] * centred[lag:]).sum()) / variance
        if abs(acf) < threshold:
            return lag
    return max(1, a.size // 2)


def admissible_shifts(n: int, lag: int, theiler: int) -> np.ndarray:
    """Circular shifts that leave **no** real alignment between the two series intact.

    Rolling the source by `s` measures the pair at an effective lag of `lag + s`, so the
    ensemble has to exclude every shift that lands near an alignment the data actually has:

    *   `s` near `0` - the tested alignment itself, which is the alternative hypothesis.
    *   `s` near `-lag` - the **simultaneous** alignment, effective lag zero. This one is easy
        to forget and it is not hypothetical: anything that varies frame by frame and touches
        every scale at once - a passing burst of broadband activity, a change in the field's
        overall amplitude - couples the scales instantaneously. On the synthetic cascade,
        which carries exactly such a term by construction, the single shift at `s = -lag`
        produced a transfer entropy of **0.412 nats against an observed 0.211**: the largest
        value in the whole "null" ensemble came from a real relationship in the data.

    Both windows are `theiler` frames wide, measured circularly. Leaving either one in caps
    the achievable p-value at the fraction of shifts that fall inside it, no matter how many
    surrogates are drawn - significance limited by a null that was wrong rather than by
    evidence that was weak.
    """
    window = max(1, int(theiler))
    shifts = np.arange(1, n)
    def circular_distance(values: np.ndarray, centre: int) -> np.ndarray:
        raw = np.abs(values - centre) % n
        return np.minimum(raw, n - raw)
    keep = ((circular_distance(shifts, 0) >= window)
            & (circular_distance(shifts, (-lag) % n) >= window))
    admissible = shifts[keep]
    if admissible.size < 2:
        raise CrossScaleError(
            "no circular shift of a %d-frame record avoids both the tested alignment and the "
            "simultaneous one within a %d-frame decorrelation window. Every available "
            "surrogate would still carry an alignment the data really has. A longer record is "
            "the only fix." % (n, window))
    return admissible


def _shift_null(source: np.ndarray, target: np.ndarray, lag: int, bins: int, wrap: bool,
                statistic: Callable[..., float], n_surrogates: int, seed: int,
                theiler: int = 1) -> np.ndarray:
    """Circular-shift surrogates of the source series, outside the admissible-shift windows.

    The right null for a lagged coupling claim (rule R1's fourth bullet): each series keeps
    its own distribution and its own autocorrelation entirely, and only the *alignment*
    between them is destroyed. Autocorrelation is exactly what inflates naive significance
    (rule R12), so leaving it intact is what makes the comparison mean anything - and
    `admissible_shifts` is what stops the ensemble quietly keeping an alignment as well.
    """
    rng = np.random.default_rng(seed)
    shifts = admissible_shifts(source.size, lag, theiler)
    out = np.empty(n_surrogates, dtype=np.float64)
    for index in range(n_surrogates):
        out[index] = statistic(np.roll(source, int(rng.choice(shifts))), target, lag,
                               bins, wrap)
    return out


def cross_scale_dependency(
    signature,
    *,
    lags: Sequence[int],
    cadence_seconds: float,
    measure: str = "energy_fraction",
    estimator: str = "transfer_entropy",
    bins: int = DEFAULT_BINS,
    wrap: bool = True,
    advection_speed_m_s: Optional[float] = None,
    n_surrogates: int = DEFAULT_SHIFT_SURROGATES,
    alpha: float = 0.05,
    correction: str = "benjamini_yekutieli",
    seed: int = 20260821,
) -> Dict[str, Any]:
    """Every ordered scale pair at every admissible lag, surrogate-referenced and FDR-corrected.

    Returns the whole family - including the tests that were excluded by the support floor
    and why - because a sweep that silently drops what it cannot test looks identical to a
    sweep that had nothing to drop.
    """
    if estimator not in ("transfer_entropy", "mutual_information"):
        raise InvalidParameterError("estimator", estimator,
                                    "'transfer_entropy' or 'mutual_information'")
    statistic = (transfer_entropy if estimator == "transfer_entropy"
                 else lagged_mutual_information)

    matrix = signature.to_matrix(measure)
    n_times, n_scales = matrix.shape
    floors = support_floor(signature, cadence_seconds, advection_speed_m_s)
    warnings: List[str] = list(floors["warnings"])

    cells = bins ** (3 if estimator == "transfer_entropy" else 2)
    per_cell = n_times / cells
    if per_cell < MIN_SAMPLES_PER_CELL:
        warnings.append(
            "%d frames spread over %d joint cells is %.2f samples per cell; below about %.0f "
            "the estimate is dominated by its own bias. It is still reported, because the "
            "surrogate ensemble carries the same bias and the excess remains interpretable, "
            "but the raw value is not."
            % (n_times, cells, per_cell, MIN_SAMPLES_PER_CELL))

    usable = {record["scale"] for record in signature.interior if record["usable"]}
    tests: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    for source_index, source_scale in enumerate(signature.scales):
        for target_index, target_scale in enumerate(signature.scales):
            if source_index == target_index:
                continue
            if (str(source_scale) not in usable) or (str(target_scale) not in usable):
                excluded.append({
                    "source_scale": str(source_scale), "target_scale": str(target_scale),
                    "reason": "one of the scales has no valid interior (rule R13), so its "
                              "signature is NaN and no dependency can be estimated"})
                continue
            source = matrix[:, source_index]
            target = matrix[:, target_index]
            floor = max(floors["floor_by_scale"][str(source_scale)],
                        floors["floor_by_scale"][str(target_scale)])
            for lag in lags:
                if lag < floor:
                    excluded.append({
                        "source_scale": str(source_scale),
                        "target_scale": str(target_scale), "lag_frames": int(lag),
                        "reason": ("below the rule R4 support floor of %d frame(s); at this "
                                   "lag the two scales are the same air seen twice through "
                                   "overlapping filters" % floor)})
                    continue
                observed = statistic(source, target, lag, bins, wrap)
                theiler = max(decorrelation_frames(source), decorrelation_frames(target))
                null = _shift_null(source, target, lag, bins, wrap, statistic,
                                   n_surrogates, seed + 7919 * lag + source_index,
                                   theiler=theiler)
                finite = null[np.isfinite(null)]
                if not np.isfinite(observed) or finite.size == 0:
                    excluded.append({
                        "source_scale": str(source_scale),
                        "target_scale": str(target_scale), "lag_frames": int(lag),
                        "reason": "the estimator was not finite on the data or on any "
                                  "surrogate"})
                    continue
                k = int(np.sum(finite >= observed))
                std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                tests.append({
                    "label": "%s->%s@%d" % (source_scale, target_scale, lag),
                    "source_scale": str(source_scale),
                    "target_scale": str(target_scale),
                    "lag_frames": int(lag),
                    "lag_seconds": float(lag * cadence_seconds),
                    "support_floor_frames": int(floor),
                    "estimator": estimator,
                    "observed_nats": float(observed),
                    "surrogate_mean_nats": float(finite.mean()),
                    "excess_nats": float(observed - finite.mean()),
                    "effect_size": ((observed - float(finite.mean())) / std if std > 0
                                    else float("nan")),
                    "p_value": float((1.0 + k) / (1.0 + finite.size)),
                    "n_surrogates": int(finite.size),
                    "wrap": bool(wrap),
                    "theiler_window_frames": int(theiler),
                })

    # Whether this configuration *could* have reported anything, checked before the result is
    # read rather than after. A surrogate p-value floors at 1/(1+n), and a family of this size
    # under this correction needs a raw p far below that floor; without this block a sweep
    # that was arithmetically incapable of rejecting anything is indistinguishable from a
    # clean negative (rule R5, and the reason T4C.5 built check_power).
    power = check_power(n_surrogates, max(len(tests), 1), alpha=alpha, method=correction)
    if tests and not power["can_reject_after_correction"]:
        warnings.append(power["warning"])

    screened = screen(tests, alpha=alpha, method=correction) if tests else {
        "n_tests": 0, "n_significant": 0, "results": [], "correction": None,
        "warnings": ["no test survived the support floor and the interior mask, so nothing "
                     "was screened"]}
    warnings.extend(screened.get("warnings", []))

    wrap_fraction = (float(max(lags)) / n_times) if (wrap and lags and n_times) else 0.0
    if wrap and wrap_fraction > 0.1:
        warnings.append(
            "the longest lag wraps %.0f%% of the record onto itself. Lags wrap so that the "
            "statistic and the circularly-stationary surrogate null are computed the same "
            "way - the alternative rejects almost always - but at this fraction the wrapped "
            "pairs are a large share of the sample and the estimate is diluted. Use a longer "
            "record rather than a shorter lag." % (100.0 * wrap_fraction))

    analysis_config = {
        "estimator": estimator,
        "measure": measure,
        "bins": int(bins),
        "wrap": bool(wrap),
        "n_scales": int(n_scales),
        "lags_frames": [int(value) for value in lags],
        "cadence_seconds": float(cadence_seconds),
        "advection_speed_m_s": (float(advection_speed_m_s)
                                 if advection_speed_m_s is not None else None),
        "n_surrogates": int(n_surrogates),
        "alpha": float(alpha),
        "correction": correction,
        "seed": int(seed),
    }
    analysis_config_sha256 = hashlib.sha256(json.dumps(
        analysis_config, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return {
        "estimator": estimator,
        "measure": measure,
        "bins": int(bins),
        "wrap": bool(wrap),
        "n_frames": int(n_times),
        "n_scales": int(n_scales),
        "lags_frames": [int(v) for v in lags],
        "cadence_seconds": float(cadence_seconds),
        "support_floor": floors,
        "n_tests": len(tests),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "results": screened.get("results", []),
        "n_significant": screened.get("n_significant", 0),
        "power": power,
        "alpha": float(alpha),
        "n_surrogates_requested": int(n_surrogates),
        "seed": int(seed),
        "analysis_config": analysis_config,
        "analysis_config_sha256": analysis_config_sha256,
        "correction": screened.get("correction"),
        "null_model": ("circular shift of the source series: both series keep their own "
                       "distribution and autocorrelation, only the alignment between them "
                       "is destroyed"),
        "causality_caveat": (
            "rule R7: this is a precursor relationship, not a causal one. Two scales may "
            "both respond to an unobserved process, and transfer entropy cannot tell that "
            "apart from one driving the other."),
        "warnings": warnings,
    }
