"""T4E.24: how often the extractor loses a feature that is genuinely there.

T4E.13 fixed ``ABSENCES_TOLERATED = 1`` as the minimal relaxation of candidate 2 - the only `k`
below `S` nameable without choosing a free fraction - and said in its own declaration that the
choice was **not** calibrated against any absence mechanism, because none had been measured.
T4E.21 then found one from an unexpected direction: at a feature density *below* the record's
own, 80 of 247 planted features were never recovered at all.

That matters because the criterion consumes recovered features. If the extractor drops a third
of what is present, then *absent from this scene* and *present but suppressed* are the same
observation, and the tolerance was set without knowing how often that happens.

**What this module measures, and what it deliberately does not.** It measures a property of
`local_maximum_extractor` on synthetic evidence with known truth. It chooses no tolerance,
proposes no proportion of `S`, ranks no candidate against another, and modifies nothing in the
extractor. Choosing a tolerance against a measured absence rate is its own declaration with its
own blindness claim (R20).

**The quantity is not the one T4E.21 measured.** T4E.21 planted independently in every frame, so
it measured a marginal recovery rate over unrelated features and could not ask whether absences
fall on the *same* features repeatedly. The criterion does not consume marginal rates. It
consumes the number of scenes a *particular* feature is seen in, and that count is what is
measured here.

**Geometry is held identical across the `S` scenes on purpose.** It is the most favourable case
for recovery that exists: real recurrence carries jitter, drift and evolution, and every one of
those can only make recovery harder. So the recall measured here is an **upper** bound and the
false absence rate a **lower** bound - both erring in the direction that weakens the criterion's
prospects rather than flattering them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: T4E.21's planting envelope, reused unchanged so this result speaks to the same regime.
#: Drawn from the real record under identical raw extraction: 3 to 10 features per scene,
#: scales with median 3.67 cells over 1.73 to 11.32, peak-to-background ratios 8 to 32.
DECLARED_COUNT_RANGE: Tuple[int, int] = (3, 10)
DECLARED_SCALE_RANGE: Tuple[float, float] = (1.73, 11.32)
DECLARED_SCALE_MEDIAN: float = 3.67
DECLARED_RATIO_RANGE: Tuple[float, float] = (8.0, 32.0)
DECLARED_STRETCHES: Tuple[float, ...] = (1.0, 1.5, 2.5)

#: T4E.21's separation rule, and the reason it is a declaration rather than a tuning knob: a
#: planting closer than this to another could not be distinguished from a merge, and one closer
#: than the edge margin would fire the R13 off-frame refusal instead of being recovered.
DECLARED_MINIMUM_SEPARATION_CELLS: float = 8.0
DECLARED_EDGE_MARGIN_CELLS: float = 12.0

#: T4E.21's pairing radius, equal to the minimum separation so a planting can never be paired
#: with its neighbour's feature.
DECLARED_PAIRING_RADIUS_CELLS: float = 8.0

#: The partition size candidate 3 is measured at. The result speaks to the criterion as it
#: stands rather than to a size it has never run at.
DECLARED_SCENES_PER_CONFIGURATION: int = 6


@dataclass(frozen=True)
class PlantedFeature:
    """One vortex, with everything about it known by construction."""

    row: float
    col: float
    sigma: float
    ratio: float
    stretch: float
    angle: float

    def describe(self) -> Dict[str, float]:
        return {"row": self.row, "col": self.col, "sigma": self.sigma,
                "ratio": self.ratio, "stretch": self.stretch, "angle": self.angle}


def draw_configuration(rng: np.random.Generator, shape: Tuple[int, int],
                       *, count: Optional[int] = None) -> Tuple[PlantedFeature, ...]:
    """Draw one configuration of features, all parameters fixed once and then reused.

    Every parameter is drawn *here* and never again, which is what makes the `S` scenes of a
    configuration differ in their background and in nothing else.
    """
    n0, n1 = int(shape[0]), int(shape[1])
    low, high = DECLARED_COUNT_RANGE
    n = int(rng.integers(low, high + 1)) if count is None else int(count)
    margin = DECLARED_EDGE_MARGIN_CELLS
    if n0 - 2 * margin <= 0 or n1 - 2 * margin <= 0:
        raise ValueError("the frame is smaller than twice the declared edge margin")

    features: List[PlantedFeature] = []
    separation_sq = DECLARED_MINIMUM_SEPARATION_CELLS ** 2
    for _ in range(n):
        for _attempt in range(1000):
            row = float(rng.uniform(margin, n0 - margin))
            col = float(rng.uniform(margin, n1 - margin))
            if all((row - f.row) ** 2 + (col - f.col) ** 2 >= separation_sq for f in features):
                break
        else:
            # Refused rather than crowded: a configuration that cannot be laid out at the
            # declared separation is dropped short, and the count it actually carries is the
            # count reported. Silently placing the last feature too close would put a merge
            # into the ground truth and then score it as an absence.
            break
        # Log-normal about the record's median, clipped to the record's observed range. The
        # spread reproduces a median near 3.67 inside those bounds; it is not fitted to any
        # recovery outcome, and no recovery number was looked at when it was set.
        sigma = float(np.clip(rng.lognormal(math.log(DECLARED_SCALE_MEDIAN), 0.5),
                              DECLARED_SCALE_RANGE[0], DECLARED_SCALE_RANGE[1]))
        features.append(PlantedFeature(
            row=row, col=col, sigma=sigma,
            ratio=float(rng.uniform(DECLARED_RATIO_RANGE[0], DECLARED_RATIO_RANGE[1])),
            stretch=float(rng.choice(DECLARED_STRETCHES)),
            angle=float(rng.uniform(0.0, math.pi))))
    return tuple(features)


def background_scale(values: np.ndarray) -> float:
    """The robust width a peak-to-background ratio is quoted against.

    A median absolute deviation rather than a standard deviation, because the planted field's
    own peaks would inflate the latter - and this quantity is read from the *background*,
    before anything is planted, for exactly that reason.
    """
    arr = np.asarray(values, dtype=np.float64)
    return 1.4826 * float(np.median(np.abs(arr - np.median(arr))))


def plant(background: np.ndarray, features: Sequence[PlantedFeature],
          *, scale: Optional[float] = None) -> np.ndarray:
    """Add the configuration to one background, leaving the background otherwise untouched."""
    values = np.array(background, dtype=np.float64, copy=True)
    if not len(features):
        return values
    width = background_scale(background) if scale is None else float(scale)
    n0, n1 = values.shape
    rows, cols = np.mgrid[0:n0, 0:n1].astype(np.float64)
    for feature in features:
        cos_a, sin_a = math.cos(feature.angle), math.sin(feature.angle)
        dy, dx = rows - feature.row, cols - feature.col
        along = (dy * cos_a + dx * sin_a) / (feature.sigma * feature.stretch)
        across = (-dy * sin_a + dx * cos_a) / feature.sigma
        values += feature.ratio * width * np.exp(-0.5 * (along * along + across * across))
    return values


def pair_plantings(features: Sequence[PlantedFeature],
                   extracted: Sequence[Tuple[float, float]],
                   *, radius: float = DECLARED_PAIRING_RADIUS_CELLS
                   ) -> Tuple[Tuple[Optional[int], ...], Tuple[float, ...]]:
    """T4E.21's rule: nearest unclaimed extracted feature within `radius` cells.

    Returns the index paired to each planting (`None` for an absence) and the offset in cells
    (`nan` for an absence). Claiming is what stops one strong extraction answering for two
    plantings; the radius equals the declared minimum separation, so a planting can never be
    paired with its neighbour's feature.

    Assignment is greedy over the closest available pair rather than in planting order, because
    planting order is arbitrary and a first-come rule would let it decide which of two plantings
    is recorded as absent.
    """
    n_features, n_extracted = len(features), len(extracted)
    paired: List[Optional[int]] = [None] * n_features
    offsets: List[float] = [float("nan")] * n_features
    if not n_features or not n_extracted:
        return tuple(paired), tuple(offsets)

    candidates: List[Tuple[float, int, int]] = []
    for i, feature in enumerate(features):
        for j, position in enumerate(extracted):
            distance = math.hypot(float(position[0]) - feature.row,
                                  float(position[1]) - feature.col)
            if distance <= float(radius):
                candidates.append((distance, i, j))
    candidates.sort()

    claimed_features: set = set()
    claimed_extracted: set = set()
    for distance, i, j in candidates:
        if i in claimed_features or j in claimed_extracted:
            continue
        claimed_features.add(i)
        claimed_extracted.add(j)
        paired[i] = j
        offsets[i] = distance
    return tuple(paired), tuple(offsets)


def presence_distribution(counts: Sequence[int], scenes: int) -> Dict[str, int]:
    """The full distribution over 0..S, never only its mean (acceptance condition 2)."""
    histogram = {str(k): 0 for k in range(int(scenes) + 1)}
    for value in counts:
        histogram[str(int(value))] += 1
    return histogram


def admission_rates(counts: Sequence[int], scenes: int,
                    absences: Sequence[int] = (0, 1, 2, 3)) -> Dict[str, Dict[str, object]]:
    """The fraction of truly-recurrent features whose presence count reaches `k = S - a`.

    A count over the observed features, not a derivation from any model. This is the number the
    criterion's behaviour actually depends on, and it is the strong half of this slice.
    """
    total = len(counts)
    out: Dict[str, Dict[str, object]] = {}
    for a in absences:
        k = int(scenes) - int(a)
        admitted = sum(1 for value in counts if int(value) >= k)
        out["a=%d" % int(a)] = {
            "k": k,
            "admitted": admitted,
            "of": total,
            "rate": (admitted / total) if total else float("nan")}
    return out


def dispersion(counts: Sequence[int], scenes: int, *, draws: int = 999,
               seed: int = 20260911) -> Dict[str, object]:
    """Observed spread of presence counts against what per-scene independence predicts.

    The null is parametric: `Binomial(S, p_hat)` at the observed recovery rate and the observed
    number of features. If absences fell independently across scenes the observed variance would
    sit inside the band; concentration on particular features puts it above.

    The band is a bootstrap interval, not a test with a stated level, and it is reported as a
    description of the mechanism. The design fixes the very thing that would cause concentration
    - identical neighbours, scales and amplitudes across the `S` scenes - so a concentrated
    result is close to built in. That caveat travels with this number wherever it goes.
    """
    values = np.asarray([int(v) for v in counts], dtype=np.float64)
    n_features, s = values.size, int(scenes)
    if not n_features:
        return {"observed_variance": None, "reason": "no features were planted"}
    p_hat = float(values.mean() / s) if s else float("nan")
    observed = float(values.var(ddof=1)) if n_features > 1 else 0.0

    rng = np.random.default_rng(int(seed))
    simulated = np.empty(int(draws), dtype=np.float64)
    for d in range(int(draws)):
        sample = rng.binomial(s, p_hat, size=n_features).astype(np.float64)
        simulated[d] = float(sample.var(ddof=1)) if n_features > 1 else 0.0
    low = float(np.quantile(simulated, 0.025))
    high = float(np.quantile(simulated, 0.975))
    return {
        "observed_variance": observed,
        "binomial_variance": float(s * p_hat * (1.0 - p_hat)),
        "p_hat_recovery": p_hat,
        "bootstrap_draws": int(draws),
        "bootstrap_band_95": [low, high],
        "outside_the_band": bool(observed < low or observed > high),
        "direction": ("over-dispersed" if observed > high else
                      "under-dispersed" if observed < low else "inside the band"),
        "what_this_is_not": (
            "a test with a stated level, and not evidence that the mechanism generalises. "
            "The design holds neighbours, scales and amplitudes fixed across the S scenes, "
            "which is what would cause concentration, so a concentrated result is close to "
            "built in and is declared as the weak half of this slice."),
    }


def survival_by_cardinality(seen_by_configuration: Sequence[Sequence[int]], scenes: int,
                            cardinalities: Sequence[int] = (2, 3)) -> Dict[str, object]:
    """How many of a configuration's pairs and triples survive, which is what the pipeline uses.

    `ALLOWED_CARDINALITIES` in `spectral_constellation.py` is `(2, 3)`: constellations are pairs
    and triples, and a fourth node is refused by name. So asking whether a whole configuration
    survives measures something nothing downstream consumes, and this reports the unit that is
    actually consumed instead.

    A group is **intact** when every member was recovered in all `scenes` scenes, and
    **assemblable** when every member was recovered at least once -- a group containing a feature
    the extractor never recovered anywhere cannot be matched by any criterion, whatever tolerance
    it applies over scenes, because the object it would have to match was never built.
    """
    out: Dict[str, object] = {}
    for k in cardinalities:
        k = int(k)
        total = intact = assemblable = 0
        for counts in seen_by_configuration:
            values = [int(v) for v in counts]
            n = len(values)
            total += math.comb(n, k) if n >= k else 0
            always = sum(1 for v in values if v >= int(scenes))
            ever = sum(1 for v in values if v > 0)
            intact += math.comb(always, k) if always >= k else 0
            assemblable += math.comb(ever, k) if ever >= k else 0
        out["k=%d" % k] = {
            "groups": total,
            "intact_in_every_scene": intact,
            "assemblable_in_at_least_one": assemblable,
            "intact_rate": (intact / total) if total else None,
            "assemblable_rate": (assemblable / total) if total else None,
        }
    out["why_these_sizes"] = (
        "spectral_constellation.ALLOWED_CARDINALITIES is (2, 3); a fourth node is refused by "
        "name as the beginning of frequent-subgraph mining. These are the units a criterion "
        "actually consumes.")
    return out
