"""T4E.9: the T4E identity path against an answer that holds by construction (closes D101).

This repository carries two identity layers over one shared geometric core. `src/core/motif.py`
is exercised by `planted_motif` and `motif_null` and has been since TG3.5. The T4E path --
`spectral_constellation.extract_constellations`, `spectral_invariance.sign_constellations`,
`spectral_clustering.SignatureMetric` -- is the one T4E.8 measures, the one T4F.6 would
adjudicate, and the one D96, D97, D99 and D100 all describe, and until this module it appeared
nowhere in `src/benchmarks/`. Its only ground-truth-like evidence was T4E.7's synthetic check,
which plants replicates at the *signature* level and so tests the matcher while skipping
detection, tracking and constellation extraction entirely.

**What makes the answer certified rather than merely planted.** The motif generator places a
scalene triangle of declared arm ratios at a uniformly random centre and a uniformly random
rotation in each scene, among distractors drawn under the same minimum separation the surrogate
null uses. So the three motif features are known by construction in every scene, and the
transformation between any two scenes' motifs is a translation and a rotation -- exactly the
invariance `spatial_geometry` declares. A cross-scene pair of motif configurations is a true
positive *by construction*, not by proxy label. Every other cross-scene pair is a true negative
by construction, because distractors are drawn independently per scene.

This is the two-sided measurement T4E.6 requires, taken against construction rather than
against tracked keys. It says nothing about the atmosphere: a real record has no planted
motif, and passing here does not discharge T4E.8's acquired-record acceptance or approve any
mining radius. What it settles is whether the T4E identity definition can recover a known
answer at all.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, SignaturePoint,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.operating_point import ESTIMATORS, estimate, required_support
from src.analysis_engine.spectral_identity_audit import (
    labelled_errors, radius_feasibility, recall_radius)
from src.analysis_engine.spectral_invariance import sign_constellations
from src.benchmarks.core import Benchmark, CheckResult, Outcome, register_benchmark, stage_check
from src.benchmarks.seeding import SeedBundle, derive
from src.benchmarks import fields as F
from src.core.feature import FeatureSet
from src.core.tracking import MotionBounds, SearchVolume, Track, TrackingResult
from src.physical_core.field import PhysicalField


#: Declared before the measurement. Changing any of these changes what is being certified.
CARDINALITY = 3
MODE = "spatial_geometry"
SCENES_PER_PARTITION = 6
#: Disjoint seed blocks. Calibration fixes the radius; the two evaluation blocks never inform it.
CALIBRATION_SEEDS = tuple(range(100, 100 + SCENES_PER_PARTITION))
EVALUATION_SEEDS = tuple(range(200, 200 + SCENES_PER_PARTITION))
NULL_SEEDS = tuple(range(500, 500 + SCENES_PER_PARTITION))
#: The radius admits this fraction of calibration motif pairs, then is frozen.
CALIBRATION_RECALL = 0.9
#: Acceptance. Both must hold on the evaluation partitions for a PASS.
MAX_FALSE_SPLIT = 0.10
MAX_FALSE_ADMISSION = 0.10
#: A planted position must claim exactly one feature this close, in cells. The generator
#: guarantees at least 30 cells between features, so half of that cannot be ambiguous.
LABEL_TOLERANCE_CELLS = 15.0
WEIGHTS = AttributeWeights(geometry=1.0, bearings=1.0, strengths=1.0, scales=1.0)
SCOPE = "t4e9/planted-motif/256-cell-grid"

CLAIM_BOUNDARY = (
    "Certified on synthetic scenes whose motif is known by construction. It measures the T4E "
    "identity path, not the atmosphere: no mining radius is approved and T4E.8's "
    "acquired-record acceptance is not discharged by any outcome here.")


#: T4E.11 reserves these blocks for a confirmatory evaluation and they have never been built.
#: Declared in `data/identity_calibration/t4e11-normalised-distance-declaration.json` before
#: they existed. Sequential preregistered attempts still accumulate multiplicity -- three
#: candidates tried and the first success reported is three tests reported as one -- and
#: untouched blocks are what converts that into a single confirmatory test. This constant is
#: the mechanism: intention alone has already been shown insufficient elsewhere in this
#: programme, which is why `audit_spatial_identity.py` refuses the forecast period in code.
RESERVED_CONFIRMATORY_SEEDS = frozenset(
    seed for start in (700, 710, 720, 730) for seed in range(start, start + 6))


class ReservedSceneOpened(RuntimeError):
    """A confirmatory block was reached by development code, refused by name.

    Looking is the whole cost. A block inspected during development is development data
    afterwards however it is later described, so the refusal has to come before the scene is
    built rather than before it is reported.
    """


def _refuse_reserved(seeds, *, confirmatory: bool) -> None:
    if confirmatory:
        return
    reserved = sorted(set(int(seed) for seed in seeds) & RESERVED_CONFIRMATORY_SEEDS)
    if reserved:
        raise ReservedSceneOpened(
            "scene seeds %s are reserved for T4E.11's confirmatory evaluation and have never "
            "been generated. Building them here would make them development data. Pass "
            "confirmatory=True only from a run whose declaration has been amended to adopt a "
            "criterion." % reserved)


class UnlabelledScene(RuntimeError):
    """A scene whose planted features could not be identified in the extraction, by name.

    Guessing here would be worse than failing: a mislabelled motif turns a true positive into
    a true negative silently, and the error rates would then measure the labelling.
    """


def _scene(seed: int, *, plant: bool) -> Tuple[List[Any], List[Tuple[float, float]]]:
    """One scene's extracted features, and the positions the generator planted the motif at."""
    bundle = derive("t4e9/scene/%s/%d" % ("planted" if plant else "null", seed))
    field = F.build_planted_motif(bundle, scene_seed=seed, plant=plant)
    features = F._invariance_features(field.data.numpy(), dataset="planted_motif")
    if len(features) != F._MOTIF_FEATURES_PER_SCENE:
        raise UnlabelledScene(
            "scene %d yielded %d features where the generator planted %d"
            % (seed, len(features), F._MOTIF_FEATURES_PER_SCENE))
    positions = F._motif_scene_positions(seed, plant=plant)
    motif = positions[:F._MOTIF_CONFIGURATION_SIZE] if plant else []
    return features, motif


def _motif_track_ids(features: Sequence[Any],
                     planted: Sequence[Tuple[float, float]]) -> Tuple[int, ...]:
    """Which extracted features are the planted motif, by nearest position. Refuses ambiguity."""
    claimed: Dict[int, Tuple[float, float]] = {}
    for position in planted:
        distances = [math.hypot(float(f.location.coords["row"]) - position[0],
                                float(f.location.coords["col"]) - position[1])
                     for f in features]
        index = int(np.argmin(distances))
        if distances[index] > LABEL_TOLERANCE_CELLS:
            raise UnlabelledScene(
                "planted motif position (%.1f, %.1f) has no extracted feature within %.1f "
                "cells; nearest is %.1f away" % (position[0], position[1],
                                                 LABEL_TOLERANCE_CELLS, distances[index]))
        if index in claimed:
            raise UnlabelledScene(
                "two planted positions both claim extracted feature %d; the labelling is "
                "ambiguous and a guess would be measured as an error rate" % index)
        claimed[index] = position
    return tuple(sorted(claimed))


def _signed_scene(seed: int, *, plant: bool):
    """Constellations of one scene, signed through the T4E path, with the motif identified."""
    features, planted = _scene(seed, plant=plant)
    motif_ids = _motif_track_ids(features, planted) if plant else ()
    tracks = tuple(Track(index, FeatureSet([feature]), {})
                   for index, feature in enumerate(features))
    tracking = TrackingResult(
        features=FeatureSet(list(features)), associator="hungarian", alpha=0.05,
        bounds=MotionBounds(max_doublings=0.5),
        volume=SearchVolume({"row": 1024.0, "col": 1024.0}, units="cells"),
        times=(0.0,), tracks=tracks)
    constellations = extract_constellations(tracking, cardinalities=(CARDINALITY,))
    signed = sign_constellations(constellations, mode=MODE, comparison_scope=SCOPE)
    if len(signed) != len(constellations):
        raise UnlabelledScene(
            "scene %d signed %d of %d constellations; a partial population would make the "
            "error rates measure the refusals" % (seed, len(signed), len(constellations)))
    points, is_motif = [], []
    for constellation, signature in zip(constellations, signed):
        points.append(SignaturePoint.from_signature(signature))
        is_motif.append(bool(plant) and tuple(sorted(constellation.track_ids)) == motif_ids)
    if plant and sum(is_motif) != 1:
        raise UnlabelledScene(
            "scene %d holds %d motif configurations where exactly one was planted"
            % (seed, sum(is_motif)))
    return points, is_motif


def _partition(seeds: Sequence[int], *, plant: bool, confirmatory: bool = False):
    _refuse_reserved(seeds, confirmatory=confirmatory)
    return [_signed_scene(seed, plant=plant) for seed in seeds]


def _cross_scene_distances(partition, metric: SignatureMetric):
    """Distances split by construction: motif-to-motif, and everything else.

    Only cross-scene pairs are formed. Two configurations inside one scene share features and
    are never independent observations of anything.
    """
    positive, negative = [], []
    for (left_points, left_motif), (right_points, right_motif) in itertools.combinations(
            partition, 2):
        for i, left in enumerate(left_points):
            for j, right in enumerate(right_points):
                distance = metric.distance(left, right)
                if left_motif[i] and right_motif[j]:
                    positive.append(distance)
                else:
                    negative.append(distance)
    return positive, negative


def certify(*, calibration_seeds: Sequence[int] = CALIBRATION_SEEDS,
            evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
            null_seeds: Sequence[int] = NULL_SEEDS) -> Dict[str, Any]:
    """Calibrate a radius on one partition, then apply it unchanged to two it never saw."""
    metric = SignatureMetric(WEIGHTS)

    calibration = _partition(calibration_seeds, plant=True)
    calibration_positive, _ = _cross_scene_distances(calibration, metric)
    radius = recall_radius(calibration_positive, CALIBRATION_RECALL)
    if radius is None:
        return {
            "outcome": "INVALID", "radius": None,
            "reason": ("no calibration motif pairs were formed, so no radius exists; an "
                       "unmeasured population is not a permissive one"),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    evaluation = _partition(evaluation_seeds, plant=True)
    positive, negative = _cross_scene_distances(evaluation, metric)
    null = _partition(null_seeds, plant=False)
    _, null_negative = _cross_scene_distances(null, metric)

    planted_errors = labelled_errors(
        positive, negative, radius,
        "Construction labels: the motif is planted, not inferred, and distractors are drawn "
        "independently per scene")
    null_errors = labelled_errors(
        [], null_negative, radius,
        "Construction labels: nothing recurs in these scenes, so every admission is a false "
        "positive")

    split = planted_errors["false_split_rate"]
    admission = planted_errors["false_admission_rate"]
    null_admission = null_errors["false_admission_rate"]
    if split is None or admission is None or null_admission is None:
        outcome, reason = "INVALID", "an evaluation population was empty and is unmeasured"
    elif (split <= MAX_FALSE_SPLIT and admission <= MAX_FALSE_ADMISSION
            and null_admission <= MAX_FALSE_ADMISSION):
        outcome, reason = "PASS", "both error rates are within their declared bounds"
    else:
        outcome, reason = "FAIL", (
            "declared bounds are split <= %.2f and admission <= %.2f; measured split %.4f, "
            "admission %.4f, null admission %.4f"
            % (MAX_FALSE_SPLIT, MAX_FALSE_ADMISSION, split, admission, null_admission))

    # Descriptive amendment, added after the first run was read and changing no window,
    # threshold, weight or acceptance criterion: it reports whether ANY radius meets both
    # bounds on this evaluation population. A frozen radius that fails while a feasible one
    # exists is a statement about the calibration procedure, not about the definition.
    feasibility = radius_feasibility(
        positive, negative, max_split=MAX_FALSE_SPLIT, max_admission=MAX_FALSE_ADMISSION,
        label_boundary="Construction labels; descriptive only, and not an approved radius")

    return {
        "outcome": outcome, "reason": reason, "radius": radius,
        "empirical_radius_feasibility": feasibility,
        "mode": MODE, "cardinality": CARDINALITY,
        "calibration": {"seeds": list(calibration_seeds),
                        "motif_pairs": len(calibration_positive),
                        "recall": CALIBRATION_RECALL},
        "planted_evaluation": {"seeds": list(evaluation_seeds), **planted_errors},
        "null_evaluation": {"seeds": list(null_seeds), **null_errors},
        "acceptance": {"max_false_split": MAX_FALSE_SPLIT,
                       "max_false_admission": MAX_FALSE_ADMISSION},
        "claim_boundary": CLAIM_BOUNDARY,
    }


#: T4E.11 candidate B, adopted 2026-09-09. Declared in
#: `data/identity_calibration/t4e11-normalised-distance-declaration.json` (sha256 ee9715ea...)
#: before any of it was measured, and adopted separately so the declared artifact keeps its hash.
NORMALISER_NAME = "median_nearest_cross_scene_neighbour"


def partition_normaliser(partition, metric: SignatureMetric) -> Optional[float]:
    """Candidate B's scale for one partition, computed without any label.

    For every configuration, the distance to its nearest neighbour among configurations from
    *other* scenes in this partition; the normaliser is the median of those values. Label-free
    by construction, so it is computable on a real record and cannot smuggle the answer into
    the scale it divides by. A median of minima rather than a mean or an extremum, so one
    badly extracted scene moves it a little rather than a lot. And it is taken from the
    close-pair regime the radius operates in: a scale read off the bulk of the distance
    distribution would be set by unrelated pairs and would track a different quantity.
    """
    if len(partition) < 2:
        return None
    nearest = []
    for index, (points, _) in enumerate(partition):
        others = [point for other, (candidates, _) in enumerate(partition) if other != index
                  for point in candidates]
        for point in points:
            nearest.append(min(metric.distance(point, other) for other in others))
    return float(np.median(nearest)) if nearest else None


def measure_normalised_blocks(blocks: Sequence[Sequence[int]] = None) -> Dict[str, Any]:
    """Candidate B's development evaluation: does normalising collapse the 1.88x spread?

    The declaration states the assumption this rests on -- that the label-free close-pair scale
    moves with the same-configuration scale between partitions -- and states that it is an
    assumption rather than a measurement. This is the measurement. It runs on the four blocks
    that informed the declaration, so it is development evidence and is not confirmation of
    anything.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    metric = SignatureMetric(WEIGHTS)
    rows = []
    for seeds in blocks:
        partition = _partition(seeds, plant=True)
        raw, _ = _cross_scene_distances(partition, metric)
        scale = partition_normaliser(partition, metric)
        array = np.asarray(raw, dtype=np.float64)
        normalised = array / scale
        rows.append({
            "seeds": list(seeds), "pairs": int(array.size), "normaliser": scale,
            "raw_mean": float(array.mean()), "raw_median": float(np.median(array)),
            "normalised_mean": float(normalised.mean()),
            "normalised_median": float(np.median(normalised)),
            "normalised_max": float(normalised.max()),
        })
    raw_means = np.asarray([row["raw_mean"] for row in rows])
    norm_means = np.asarray([row["normalised_mean"] for row in rows])
    return {
        "candidate": "B_comparably_scaled_distance",
        "normaliser": NORMALISER_NAME,
        "blocks": rows,
        "raw_mean_ratio": float(raw_means.max() / raw_means.min()),
        "normalised_mean_ratio": float(norm_means.max() / norm_means.min()),
        "evidence_class": "development",
        "boundary": ("Measured on the four blocks that informed the declaration. Development "
                     "evidence only; the reserved confirmatory blocks are untouched and this "
                     "is not confirmation of candidate B."),
    }


#: T4E.10. Independent scene blocks of the declared size, used to measure whether blocks drawn
#: from one generator are exchangeable at all. If they are not, no radius frozen on one of them
#: transfers to another, whatever estimator produced it.
EXCHANGEABILITY_BLOCKS = (tuple(range(100, 106)), tuple(range(200, 206)),
                          tuple(range(300, 306)), tuple(range(400, 406)))


def measure_block_exchangeability(blocks: Sequence[Sequence[int]] = EXCHANGEABILITY_BLOCKS
                                  ) -> Dict[str, Any]:
    """Same-configuration distance summaries per block, and the spread between blocks.

    A tolerance bound is distribution-free but not assumption-free: it covers the population
    the calibration sample was drawn from. If disjoint blocks of scenes built by one generator
    have visibly different distance distributions, the calibration sample and the evaluation
    population are not one population, and no bound calibrated on the first carries a
    guarantee about the second at any support.
    """
    metric = SignatureMetric(WEIGHTS)
    summaries = []
    for seeds in blocks:
        distances, _ = _cross_scene_distances(_partition(seeds, plant=True), metric)
        array = np.asarray(distances, dtype=np.float64)
        summaries.append({
            "seeds": list(seeds), "pairs": int(array.size),
            "mean": float(array.mean()), "median": float(np.median(array)),
            "max": float(array.max()),
        })
    means = np.asarray([item["mean"] for item in summaries])
    return {
        "blocks": summaries,
        "block_mean_min": float(means.min()), "block_mean_max": float(means.max()),
        "block_mean_ratio": float(means.max() / means.min()),
        "block_mean_sd": float(means.std(ddof=1)) if len(means) > 1 else None,
        "boundary": ("Every block is built by the same generator with the same parameters. A "
                     "ratio above one is variation between blocks, not between designs."),
    }


#: T4E.10. Two calibration supports, chosen before measuring: six scenes give 15 cross-scene
#: motif pairs, below the 22 a 90/90 tolerance bound needs, and eight give 28, above it.
ESTIMATOR_SUPPORTS = {"below_requirement": tuple(range(100, 106)),
                      "above_requirement": tuple(range(100, 108))}
COVERAGE = CALIBRATION_RECALL
CONFIDENCE = 0.9


def certify_estimators(*, evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
                       null_seeds: Sequence[int] = NULL_SEEDS) -> Dict[str, Any]:
    """T4E.10: every declared operating-point estimator, calibrated then frozen, at two supports.

    The evaluation partitions are the same ones T4E.9 used and neither informs any radius. A
    refusal is recorded as an outcome in its own right: an estimator that declines to name a
    radius on support that cannot carry its guarantee has behaved correctly, and scoring that
    as a failure would reward the estimator that answers anyway.
    """
    metric = SignatureMetric(WEIGHTS)
    evaluation = _partition(evaluation_seeds, plant=True)
    positive, negative = _cross_scene_distances(evaluation, metric)
    null = _partition(null_seeds, plant=False)
    _, null_negative = _cross_scene_distances(null, metric)

    results: Dict[str, Any] = {}
    for support_name, seeds in ESTIMATOR_SUPPORTS.items():
        calibration = _partition(seeds, plant=True)
        calibration_positive, _ = _cross_scene_distances(calibration, metric)
        per_estimator = {}
        for name in ESTIMATORS.names():
            point = estimate(name, calibration_positive, coverage=COVERAGE,
                             confidence=CONFIDENCE)
            if point.refused:
                per_estimator[name] = {"outcome": "REFUSED", **point.describe()}
                continue
            planted = labelled_errors(positive, negative, point.radius,
                                      "Construction labels, not proxy")
            nulled = labelled_errors([], null_negative, point.radius,
                                     "Construction labels: nothing recurs here")
            split, admission = planted["false_split_rate"], planted["false_admission_rate"]
            null_admission = nulled["false_admission_rate"]
            holds = (split is not None and admission is not None
                     and null_admission is not None
                     and split <= MAX_FALSE_SPLIT and admission <= MAX_FALSE_ADMISSION
                     and null_admission <= MAX_FALSE_ADMISSION)
            per_estimator[name] = {
                "outcome": "HOLDS" if holds else "DOES_NOT_HOLD",
                **point.describe(),
                "planted_evaluation": planted, "null_evaluation": nulled,
            }
        results[support_name] = {
            "calibration_seeds": list(seeds),
            "calibration_pairs": len(calibration_positive),
            "required_support": required_support(COVERAGE, CONFIDENCE),
            "estimators": per_estimator,
        }
    return {
        "block_exchangeability": measure_block_exchangeability(),
        "coverage": COVERAGE, "confidence": CONFIDENCE,
        "acceptance": {"max_false_split": MAX_FALSE_SPLIT,
                       "max_false_admission": MAX_FALSE_ADMISSION},
        "evaluation_seeds": list(evaluation_seeds), "null_seeds": list(null_seeds),
        "supports": results,
        "claim_boundary": CLAIM_BOUNDARY,
        "refusal_boundary": ("A REFUSED outcome is correct behaviour, not a failure: the "
                             "support could not carry the declared guarantee and the "
                             "estimator declined to issue a radius that would carry none."),
    }


@stage_check("4E.identity_certified")
def _check_identity_certified(field: PhysicalField, truth: Dict[str, Any]) -> CheckResult:
    """**The T4E.9 gate**: can the T4E identity path recover an answer known by construction?

    The radius is calibrated on scenes the evaluation never sees and then frozen, because a
    radius chosen after the evaluation was read is not a measurement of anything. Both error
    rates are reported against construction, and the null partition -- same feature count,
    same family, nothing recurring -- supplies the false-positive rate where the correct
    answer is that there is nothing to find.
    """
    try:
        result = certify()
    except UnlabelledScene as refusal:
        return CheckResult("4E.identity_certified", Outcome.FAIL,
                           "scene labelling refused: %s" % refusal, measured=None)
    if result["outcome"] == "INVALID":
        return CheckResult("4E.identity_certified", Outcome.NOT_YET_RUNNABLE,
                           result["reason"], measured=result)
    if result["outcome"] == "FAIL":
        return CheckResult("4E.identity_certified", Outcome.FAIL, result["reason"],
                           measured=result)
    return CheckResult(
        "4E.identity_certified", Outcome.PASS,
        "radius %.6f frozen on %d calibration motif pairs; planted split %.4f, admission "
        "%.4f; null admission %.4f"
        % (result["radius"], result["calibration"]["motif_pairs"],
           result["planted_evaluation"]["false_split_rate"],
           result["planted_evaluation"]["false_admission_rate"],
           result["null_evaluation"]["false_admission_rate"]),
        measured=result)


def truth_identity_certified(n: int = 256, **_: Any) -> Dict[str, Any]:
    return {
        "path_under_test": "spectral_constellation -> spectral_invariance -> "
                           "spectral_clustering (the T4E identity path)",
        "mode": MODE,
        "cardinality": CARDINALITY,
        "scenes_per_partition": SCENES_PER_PARTITION,
        "configurations_per_scene": math.comb(F._MOTIF_FEATURES_PER_SCENE, CARDINALITY),
        "motif_configurations_per_planted_scene": 1,
        "expected_motif_configurations_per_null_scene": 0,
        "calibration_recall": CALIBRATION_RECALL,
        "max_false_split": MAX_FALSE_SPLIT,
        "max_false_admission": MAX_FALSE_ADMISSION,
        "labels": "construction, not proxy: the motif is planted and the distractors are not",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_identity_certification(bundle: SeedBundle, n: int = 256,
                                 **_: Any) -> PhysicalField:
    """The first calibration scene, so the field shown is one the result was computed from."""
    return F.build_planted_motif(bundle, n=n, scene_seed=CALIBRATION_SEEDS[0], plant=True)


register_benchmark(Benchmark(
    name="t4e_identity_certified",
    kind="field",
    description=("The T4E identity path against a motif known by construction: a radius frozen "
                 "on one partition, both error rates measured on two it never saw, and a null "
                 "partition where the right answer is nothing. Closes D101."),
    gates=("4E.identity_certified",),
    build=build_identity_certification,
    known_answer=truth_identity_certified,
    checks=(_check_identity_certified,),
))
