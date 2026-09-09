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

import collections
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import genpareto

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

#: Opened 2026-09-09 by `data/identity_calibration/t4e12-confirmatory-amendment.json`, for a
#: confirmatory evaluation of T4E.12 candidate 2 and nothing else. Once built these are
#: inspected data permanently; renaming them later does not change that.
OPENED_CONFIRMATORY_SEEDS = frozenset(
    seed for start in (700, 710) for seed in range(start, start + 6))

#: Still reserved, and refused EVEN WHEN `confirmatory=True`. The maintainer split the reserved
#: evidence rather than spending it: candidate 2's `k` was chosen after a feasibility probe on
#: the development blocks, so a confirmatory run of it cannot test a structural choice made
#: blind. These two partitions are kept for a criterion whose structural choices ARE fixed
#: blind, which is what the reserved evidence was for. The split is enforced here rather than
#: intended, because intention has already been shown insufficient in this programme.
STILL_RESERVED_CONFIRMATORY_SEEDS = RESERVED_CONFIRMATORY_SEEDS - OPENED_CONFIRMATORY_SEEDS


class ReservedSceneOpened(RuntimeError):
    """A confirmatory block was reached by development code, refused by name.

    Looking is the whole cost. A block inspected during development is development data
    afterwards however it is later described, so the refusal has to come before the scene is
    built rather than before it is reported.
    """


def _refuse_reserved(seeds, *, confirmatory: bool) -> None:
    wanted = set(int(seed) for seed in seeds)
    held = sorted(wanted & STILL_RESERVED_CONFIRMATORY_SEEDS)
    if held:
        raise ReservedSceneOpened(
            "scene seeds %s are still reserved and are refused even under confirmatory=True. "
            "The 2026-09-09 amendment opened 700-715 for T4E.12 candidate 2 and deliberately "
            "kept these for a criterion whose structural choices are fixed without seeing "
            "them. Opening them needs a further amendment and the maintainer's decision."
            % held)
    if confirmatory:
        return
    reserved = sorted(wanted & RESERVED_CONFIRMATORY_SEEDS)
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


def _scene(seed: int, *, plant: bool,
           richness: Optional[int] = None) -> Tuple[List[Any], List[Tuple[float, float]]]:
    """One scene's extracted features, and the positions the generator planted the motif at.

    `richness` is the feature count and defaults to the frozen six. It is a parameter for
    T4E.12, which asks what a signature must carry as scenes get richer; every T4E.9 and
    T4E.11 caller leaves it alone and builds exactly the scenes it built before.
    """
    wanted = F._MOTIF_FEATURES_PER_SCENE if richness is None else int(richness)
    bundle = derive("t4e9/scene/%s/%d" % ("planted" if plant else "null", seed))
    field = F.build_planted_motif(bundle, scene_seed=seed, plant=plant, features=wanted)
    features = F._invariance_features(field.data.numpy(), dataset="planted_motif")
    if len(features) != wanted:
        raise UnlabelledScene(
            "scene %d yielded %d features where the generator planted %d"
            % (seed, len(features), wanted))
    positions = F._motif_scene_positions(seed, plant=plant, features=wanted)
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


#: T4E.11 candidate C, adopted 2026-09-09. Declared in
#: `t4e11-mutual-nearest-neighbour-declaration.json` (sha256 6b7105cc...) before it was
#: measured, and adopted separately so the declared artifact keeps its hash.
CRITERION_C_NAME = "mutual_nearest_neighbour_per_scene_pair"


def mutual_nearest_matches(left, right, metric: SignatureMetric):
    """Candidate C for one ordered scene pair: the mutual nearest-neighbour partial matching.

    `a` and `b` match when `b` is `a`'s nearest neighbour in the right scene AND `a` is `b`'s
    nearest neighbour in the left. No radius, no threshold and no normaliser, so there is no
    magnitude to carry between partitions -- which is the whole point, after a frozen radius
    (T4E.10) and a normalised one (candidate B) both failed to transfer.

    This is a matching rule and not a single-winner rule: it returns as many pairs as the two
    scenes support. It assumes one-to-one correspondence at the instance level, not that a
    physical kind occurs once per scene.
    """
    left_points, right_points = left[0], right[0]
    if not left_points or not right_points:
        return []
    forward = [min(range(len(right_points)),
                   key=lambda j: metric.distance(a, right_points[j]))
               for a in left_points]
    backward = [min(range(len(left_points)),
                    key=lambda i: metric.distance(left_points[i], b))
                for b in right_points]
    return [(i, j) for i, j in enumerate(forward) if backward[j] == i]


def measure_criterion_c(blocks: Sequence[Sequence[int]] = None,
                        null_blocks: Sequence[Sequence[int]] = None) -> Dict[str, Any]:
    """Candidate C's development evaluation: does it recover the motif, stably, and refuse noise?

    Transfer means something different here. B had a radius to carry between blocks; C has
    nothing to carry, so what must hold is that the error rates are similar across blocks whose
    distance magnitudes differ by 1.876x. A good average across unstable blocks is a failure.

    The null blocks do more work than they did for a radius. A rank-based rule always returns
    some nearest neighbour, so its false-admission rate where nothing recurs is the number that
    decides whether it means anything at all.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)

    def evaluate(seeds, *, plant):
        partition = _partition(seeds, plant=plant)
        if len(partition) < 2:
            return {"seeds": list(seeds), "refused":
                    "a partition of fewer than two scenes has no cross-scene pair"}
        matched = motif_matched = motif_total = 0
        for left, right in itertools.combinations(partition, 2):
            pairs = mutual_nearest_matches(left, right, metric)
            matched += len(pairs)
            left_motif = left[1].index(True) if any(left[1]) else None
            right_motif = right[1].index(True) if any(right[1]) else None
            if left_motif is not None and right_motif is not None:
                motif_total += 1
                if (left_motif, right_motif) in pairs:
                    motif_matched += 1
        return {
            "seeds": list(seeds), "scene_pairs": len(partition) * (len(partition) - 1) // 2,
            "matches_returned": matched, "motif_pairs": motif_total,
            "motif_pairs_matched": motif_matched,
            "false_split_rate": (None if not motif_total
                                 else (motif_total - motif_matched) / motif_total),
            "false_admission_rate": (None if not matched
                                     else (matched - motif_matched) / matched),
        }

    planted = [evaluate(seeds, plant=True) for seeds in blocks]
    nulls = [evaluate(seeds, plant=False) for seeds in null_blocks]
    splits = [row["false_split_rate"] for row in planted if row.get("false_split_rate") is not None]
    admissions = [row["false_admission_rate"] for row in planted
                  if row.get("false_admission_rate") is not None]
    return {
        "candidate": "C_mutual_nearest_neighbour_no_absolute_scale",
        "criterion": CRITERION_C_NAME,
        "planted_blocks": planted, "null_blocks": nulls,
        "false_split_range": [min(splits), max(splits)] if splits else None,
        "false_admission_range": [min(admissions), max(admissions)] if admissions else None,
        "stability_boundary": ("Transfer here is stability of the error rates across blocks, "
                               "not portability of a number. A good average across unstable "
                               "blocks is a failure."),
        "evidence_class": "development",
        "boundary": ("Measured on the blocks that informed the declaration. Development "
                     "evidence only; the reserved confirmatory blocks are untouched."),
    }


#: T4E.11 candidate D, declared in `t4e11-ratio-margin-declaration.json` (sha256 b2ba2b4e...).
#: Pinned before measurement and not answerable to this record: it is the canonical
#: nearest/second-nearest ratio from Lowe (2004), IJCV 60(2), taken for its external provenance
#: rather than because it is expected to be optimal here. A tau read off these already-inspected
#: blocks would be fitted and could not be reported as a criterion at all.
MARGIN_TAU = 0.8
CRITERION_D_NAME = "mutual_nearest_neighbour_with_ratio_margin"


class UndefinedMargin(Exception):
    """A scene with one configuration has no second nearest neighbour, so the margin is undefined.

    Declared as a refusal rather than a default. Admitting the pair would mean the rule is
    strongest exactly where it has the least evidence, and counting it as a non-match would
    charge a false split to a test that was never applied.
    """


def _pair_distances(left, right, metric: SignatureMetric):
    """The full cross-scene distance matrix for one ordered pair, computed once."""
    return [[metric.distance(a, b) for b in right[0]] for a in left[0]]


def ratio_margin_matches(left, right, metric: SignatureMetric, *, tau: float = MARGIN_TAU):
    """Candidate D: candidate C, narrowed by a margin that is a ratio and not a distance.

    A mutual nearest-neighbour pair is kept only when its distance is at most `tau` times the
    distance to the second nearest, in BOTH directions. Both quantities come from the same two
    scenes, so the test is dimensionless: rescaling the metric within a partition leaves it
    unchanged, and there is no magnitude to carry between blocks. That is the property T4E.10
    showed an absolute radius cannot have, and it is what distinguishes this from candidate B,
    which divided by an *estimated* partition scale and failed at the estimate.

    This is a strict narrowing of candidate C: condition (ii) can only remove pairs, never add
    one. So the false split can only rise from C's 0.0000 and the match count can only fall.
    The question the declaration asks is whether that trade is close to free.
    """
    if not left[0] or not right[0]:
        return []
    if len(left[0]) < 2 or len(right[0]) < 2:
        raise UndefinedMargin(
            "a scene with fewer than two configurations has no second nearest neighbour, so the "
            "margin test is undefined and the scene pair is refused rather than decided")
    distances = _pair_distances(left, right, metric)
    kept = []
    for i, j in mutual_nearest_matches(left, right, metric):
        near = distances[i][j]
        forward_second = min(d for k, d in enumerate(distances[i]) if k != j)
        backward_second = min(distances[k][j] for k in range(len(distances)) if k != i)
        if near <= tau * forward_second and near <= tau * backward_second:
            kept.append((i, j))
    return kept


def margin_ratios(left, right, metric: SignatureMetric):
    """The declared diagnostic: each mutual pair's nearest/second-nearest ratio, motif or not.

    A DIAGNOSTIC and not a criterion. It says, if candidate D fails, whether the motif and
    non-motif ratio distributions overlap or merely sit either side of a badly placed tau. A tau
    read off it would be fitted to already-inspected blocks; the declaration forbids reporting
    such a tau as an operating point, and doing so would need a candidate E evaluated on data
    these blocks did not select.
    """
    if len(left[0]) < 2 or len(right[0]) < 2:
        raise UndefinedMargin("no second nearest neighbour, so no ratio exists to report")
    distances = _pair_distances(left, right, metric)
    left_motif = left[1].index(True) if any(left[1]) else None
    right_motif = right[1].index(True) if any(right[1]) else None
    motif, other = [], []
    for i, j in mutual_nearest_matches(left, right, metric):
        near = distances[i][j]
        second = min(min(d for k, d in enumerate(distances[i]) if k != j),
                     min(distances[k][j] for k in range(len(distances)) if k != i))
        ratio = None if second == 0 else near / second
        if ratio is None:
            continue
        (motif if (i == left_motif and j == right_motif) else other).append(ratio)
    return {"motif": motif, "other": other}


def measure_criterion_d(blocks: Sequence[Sequence[int]] = None,
                        null_blocks: Sequence[Sequence[int]] = None,
                        *, tau: float = MARGIN_TAU) -> Dict[str, Any]:
    """Candidate D's development evaluation, against the statistics its declaration named.

    Two of the three acceptance conditions are the familiar error rates. The third is
    RETENTION, defined in the declaration before it was computed: on the null partition every
    match is false by construction, so the false-admission rate there is 1.0 whenever anything
    is returned and carries no information by itself. What discriminates is how much of
    candidate C's null output the margin discards. C is therefore measured alongside D on the
    same null blocks, as the denominator of that fraction and for no other purpose -- this is
    not a comparison to pick a winner, which R20 forbids and which would be meaningless anyway
    since C is already falsified.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)

    def evaluate(seeds, *, plant):
        partition = _partition(seeds, plant=plant)
        if len(partition) < 2:
            return {"seeds": list(seeds), "refused":
                    "a partition of fewer than two scenes has no cross-scene pair"}
        matched = motif_matched = motif_total = baseline = refused = 0
        ratios = {"motif": [], "other": []}
        for left, right in itertools.combinations(partition, 2):
            try:
                pairs = ratio_margin_matches(left, right, metric, tau=tau)
            except UndefinedMargin:
                refused += 1
                continue
            baseline += len(mutual_nearest_matches(left, right, metric))
            for key, values in margin_ratios(left, right, metric).items():
                ratios[key].extend(values)
            matched += len(pairs)
            left_motif = left[1].index(True) if any(left[1]) else None
            right_motif = right[1].index(True) if any(right[1]) else None
            if left_motif is not None and right_motif is not None:
                motif_total += 1
                if (left_motif, right_motif) in pairs:
                    motif_matched += 1
        return {
            "seeds": list(seeds), "scene_pairs": len(partition) * (len(partition) - 1) // 2,
            "scene_pairs_refused_for_undefined_margin": refused,
            "matches_returned": matched,
            "matches_returned_by_candidate_C": baseline,
            "retention_of_candidate_C": (None if not baseline else matched / baseline),
            "motif_pairs": motif_total, "motif_pairs_matched": motif_matched,
            "false_split_rate": (None if not motif_total
                                 else (motif_total - motif_matched) / motif_total),
            "false_admission_rate": (None if not matched
                                     else (matched - motif_matched) / matched),
            "margin_ratio_diagnostic": {
                "is_a_diagnostic_not_a_criterion": (
                    "Recorded to say whether the distributions overlap. A tau read off it is "
                    "fitted to already-inspected blocks and may not be reported as an "
                    "operating point."),
                "motif_pairs": len(ratios["motif"]),
                "motif_ratio_range": ([min(ratios["motif"]), max(ratios["motif"])]
                                      if ratios["motif"] else None),
                "other_pairs": len(ratios["other"]),
                "other_ratio_range": ([min(ratios["other"]), max(ratios["other"])]
                                      if ratios["other"] else None),
            },
        }

    planted = [evaluate(seeds, plant=True) for seeds in blocks]
    nulls = [evaluate(seeds, plant=False) for seeds in null_blocks]
    splits = [row["false_split_rate"] for row in planted
              if row.get("false_split_rate") is not None]
    admissions = [row["false_admission_rate"] for row in planted
                  if row.get("false_admission_rate") is not None]
    retentions = [row["retention_of_candidate_C"] for row in nulls
                  if row.get("retention_of_candidate_C") is not None]
    return {
        "candidate": "D_mutual_nearest_neighbour_with_a_dimensionless_margin",
        "criterion": CRITERION_D_NAME, "tau": tau,
        "planted_blocks": planted, "null_blocks": nulls,
        "false_split_range": [min(splits), max(splits)] if splits else None,
        "false_admission_range": [min(admissions), max(admissions)] if admissions else None,
        "null_retention_range": [min(retentions), max(retentions)] if retentions else None,
        "acceptance_boundary": (
            "False split at most 0.10 on every block, null retention at most 0.10, and both "
            "materially stable between blocks. A good average across unstable blocks is a fail."),
        "evidence_class": "development",
        "boundary": ("Measured on the blocks that informed the declaration. Development "
                     "evidence only; the reserved confirmatory blocks are untouched."),
    }


# ---------------------------------------------------------------------------------------------
# A diagnostic on the ground truth itself, run after four candidates were falsified against it.
#
# It adopts nothing, thresholds nothing and reports no operating point. It asks one arithmetic
# question that none of the four declarations thought to ask: WHAT ARE the pairs the error rates
# have been calling false admissions?
#
# Each scene holds six features -- three motif, three distractors -- and the signature is
# cardinality three, so each scene yields C(6,3) = 20 configurations, of which exactly one is
# the motif, NINE share two motif features, nine share one, and one shares none. The nine that
# share two carry the same two physical features, replanted under translation and rotation, in
# every scene of every block: one motif edge at fixed length and bearing, and two members with
# identical strengths and scales. That is recurring structure by construction, and the labelling
# scores a criterion that finds it as wrong.
#
# Whether that matters is not a matter of opinion, and this measures it.


def _motif_member_ids(features: Sequence[Any],
                      planted: Sequence[Tuple[float, float]]) -> Tuple[int, ...]:
    """The motif's feature indices IN PLANTED ORDER, so a vertex is identifiable across scenes.

    `_motif_track_ids` sorts, which is right for asking whether a configuration is the motif and
    wrong for asking WHICH motif vertices it holds. Vertex k is the same physical feature in
    every scene, so the sets must be comparable between scenes and a sorted tuple destroys that.
    Ambiguity is refused there and stays refused here; this reuses that labelling rather than
    repeating it.
    """
    claimed: Dict[int, Tuple[float, float]] = {}
    order: List[int] = []
    for position in planted:
        distances = [math.hypot(float(f.location.coords["row"]) - position[0],
                                float(f.location.coords["col"]) - position[1])
                     for f in features]
        index = int(np.argmin(distances))
        if distances[index] > LABEL_TOLERANCE_CELLS or index in claimed:
            raise UnlabelledScene(
                "the motif labelling is ambiguous for this scene; refused rather than guessed")
        claimed[index] = position
        order.append(index)
    return tuple(order)


def _scene_with_motif_membership(seed: int, *, plant: bool, richness: Optional[int] = None,
                                confirmatory: bool = False):
    """One signed scene, plus which motif vertices each configuration holds.

    Returns `(points, is_motif, membership)` where `membership[i]` is the frozenset of motif
    vertex numbers configuration `i` contains. The first two elements are exactly what
    `_signed_scene` returns, so the criteria can be run against this unchanged.
    """
    _refuse_reserved([seed], confirmatory=confirmatory)
    features, planted = _scene(seed, plant=plant, richness=richness)
    members = _motif_member_ids(features, planted) if plant else ()
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
            "decomposition measure the refusals" % (seed, len(signed), len(constellations)))
    points, is_motif, membership = [], [], []
    for constellation, signature in zip(constellations, signed):
        points.append(SignaturePoint.from_signature(signature))
        held = frozenset(members.index(track) for track in constellation.track_ids
                         if track in members)
        membership.append(held)
        is_motif.append(bool(plant) and len(held) == CARDINALITY)
    if plant and sum(is_motif) != 1:
        raise UnlabelledScene(
            "scene %d holds %d motif configurations where exactly one was planted"
            % (seed, sum(is_motif)))
    return points, is_motif, membership


def _pair_class(left_members: frozenset, right_members: frozenset) -> str:
    """What a matched pair actually is, in the language of the construction.

    `shared_2` and `shared_1` are pairs whose two configurations hold the SAME motif vertices --
    the same physical features, replanted -- so they share real recurring structure that the
    ground truth nonetheless scores as a false admission. `crossed` holds motif features that
    are not the same ones, and `unrelated` holds none in common. Only the last of these is
    unambiguously a coincidence.
    """
    shared = left_members & right_members
    if len(shared) == CARDINALITY:
        return "motif"
    if left_members == right_members and shared:
        return "shared_%d" % len(shared)
    if shared:
        return "crossed_%d" % len(shared)
    return "unrelated"


def decompose_matched_pairs(blocks: Sequence[Sequence[int]] = None,
                            null_blocks: Sequence[Sequence[int]] = None,
                            *, tau: float = MARGIN_TAU) -> Dict[str, Any]:
    """What the false admissions ARE, decomposed by the construction rather than by the label.

    A DIAGNOSTIC. It adopts nothing, chooses no threshold and reports no operating point. It
    exists because four candidates were falsified against a ground truth that no declaration had
    examined, and because the arithmetic of the scene population -- nine of twenty configurations
    sharing two motif features -- says that ground truth cannot be scoring only confusion.

    Both criteria are decomposed. Candidate C's matches say what the ordering finds; candidate
    D's say what survives the margin. That is not the comparison R20 forbids: both are already
    falsified and neither is being selected over the other.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)

    def evaluate(seeds, *, plant):
        scenes = [_scene_with_motif_membership(seed, plant=plant) for seed in seeds]
        counts = {"C": collections.Counter(), "D": collections.Counter()}
        available = collections.Counter()
        for left, right in itertools.combinations(scenes, 2):
            pair = ((left[0], left[1]), (right[0], right[1]))
            for name, matches in (("C", mutual_nearest_matches(*pair, metric)),
                                  ("D", ratio_margin_matches(*pair, metric, tau=tau))):
                for i, j in matches:
                    counts[name][_pair_class(left[2][i], right[2][j])] += 1
            for i in range(len(left[0])):
                for j in range(len(right[0])):
                    available[_pair_class(left[2][i], right[2][j])] += 1
        return {
            "seeds": list(seeds),
            "configurations_per_scene": len(scenes[0][0]),
            "cross_scene_pairs_available": dict(available),
            "matched_by_candidate_C": dict(counts["C"]),
            "matched_by_candidate_D": dict(counts["D"]),
        }

    planted = [evaluate(seeds, plant=True) for seeds in blocks]
    nulls = [evaluate(seeds, plant=False) for seeds in null_blocks]

    def totalled(rows, key):
        total = collections.Counter()
        for row in rows:
            total.update(row[key])
        return dict(total)

    return {
        "diagnostic": "what the false admissions are, by construction",
        "is_a_diagnostic_not_a_criterion": (
            "Nothing is adopted, no threshold is chosen and no operating point is reported. A "
            "change to the ground truth suggested by this is a scientific choice needing its own "
            "declaration, and any criterion evaluated against a changed ground truth is a new "
            "measurement, not a re-reading of an old one."),
        "tau": tau,
        "classes": {
            "motif": "both configurations are the full planted motif; the only true positive.",
            "shared_2": "both hold the SAME two motif vertices -- the same two physical "
                        "features, replanted -- so they share one motif edge at fixed length "
                        "and bearing plus two members with identical strengths and scales. "
                        "Scored as a false admission by the current ground truth.",
            "shared_1": "both hold the same single motif vertex.",
            "crossed_2": "both hold two motif vertices but not the same two.",
            "crossed_1": "both hold one motif vertex but not the same one.",
            "unrelated": "no motif vertex in common; unambiguously a coincidence.",
        },
        "planted_blocks": planted, "null_blocks": nulls,
        "planted_totals": {
            "available": totalled(planted, "cross_scene_pairs_available"),
            "candidate_C": totalled(planted, "matched_by_candidate_C"),
            "candidate_D": totalled(planted, "matched_by_candidate_D"),
        },
        "null_totals": {
            "available": totalled(nulls, "cross_scene_pairs_available"),
            "candidate_C": totalled(nulls, "matched_by_candidate_C"),
            "candidate_D": totalled(nulls, "matched_by_candidate_D"),
        },
        "evidence_class": "development",
        "boundary": ("Computed on the blocks that informed four declarations. Development "
                     "evidence; the reserved confirmatory blocks are untouched."),
    }


# ---------------------------------------------------------------------------------------------
# T4E.12 candidate 2: identity as consistency across the partition, not a verdict on one pair.
#
# Declared in `data/identity_calibration/t4e12-consistency-declaration.json` and NOT MEASURED
# until that declaration is adopted.
#
# Every candidate before this one asked the same question -- given ONE pair of configurations,
# is it a match? -- and five failed in three different ways. None used what the partition
# actually offers: the motif is in every scene, so its matches form a complete graph, while a
# coincidental match between two scenes has no reason to extend to the rest.
#
# The feasibility probe that preceded this declaration is disclosed in it and is the reason it
# was written: across six-scene partitions the motif reached a clique of 6 in every scene of
# every block at richness 6 and 9, and no non-motif configuration ever exceeded 5.

#: Declared clique size: a configuration must be consistently matched across this many scenes.
#: Set to the partition size, which is the strictest possible value and is a statement about
#: what recurrence means rather than a number tuned to these blocks. See the declaration's
#: limitation section: on an acquired record recurrence is partial and this will not transfer.
CONSISTENCY_K = None  # None means "every scene in the partition"
CRITERION_CONSISTENCY_NAME = "mutual_nearest_neighbour_consistency_across_the_partition"


def _partner_map(partition, metric: SignatureMetric):
    """For each configuration, its mutual nearest neighbour in each other scene, if any.

    Mutual nearest-neighbour matching returns a one-to-one partial matching, so a configuration
    has AT MOST ONE partner in any other scene. That is what makes the consistency test exact
    and cheap rather than a general clique search: the candidate group of a node is determined
    by the node, and only its internal agreement has to be checked.
    """
    partners: Dict[Tuple[int, int], Dict[int, int]] = collections.defaultdict(dict)
    for a, b in itertools.combinations(range(len(partition)), 2):
        for i, j in mutual_nearest_matches(partition[a], partition[b], metric):
            partners[(a, i)][b] = j
            partners[(b, j)][a] = i
    return partners


def consistent_group(node, partners) -> Tuple[Tuple[int, int], ...]:
    """The largest set of mutually agreeing partners containing `node`, computed exactly.

    The candidate members are `node` plus its unique partner in each other scene, so there are
    at most as many as there are scenes and the maximum clique among them can be enumerated
    outright. No greedy approximation: a greedy walk can miss the largest agreeing set, and a
    criterion that sometimes misses one would make its own error rates unreproducible.
    """
    candidates = [node] + [(scene, index) for scene, index in sorted(partners[node].items())]
    best: Tuple[Tuple[int, int], ...] = (node,)
    for size in range(len(candidates), 1, -1):
        if size <= len(best):
            break
        for subset in itertools.combinations(candidates, size):
            if node not in subset:
                continue
            if len({scene for scene, _ in subset}) != len(subset):
                continue          # one configuration per scene, or it is not a correspondence
            agreeing = all(partners[left].get(right[0]) == right[1]
                           for left, right in itertools.permutations(subset, 2))
            if agreeing:
                best = subset
                break
        if len(best) == size:
            break
    return best


def consistency_admitted_pairs(partition, metric: SignatureMetric, *, k: Optional[int] = None):
    """T4E.12 candidate 2: admit the pairs inside groups that agree across at least `k` scenes.

    `k` defaults to the whole partition, which is the declared setting. Returns the admitted
    cross-scene pairs as `(scene_a, index_a, scene_b, index_b)` with `scene_a < scene_b`, so the
    error rates are counted over exactly the population candidates C and D were counted over and
    the three are directly comparable.

    There is no radius, no ratio, no threshold and no fitted model -- so nothing to carry
    between partitions, nothing to estimate wrongly, and nothing whose support can run out. What
    it uses instead is the partition's own structure.
    """
    wanted = len(partition) if k is None else int(k)
    if wanted < 2:
        raise ValueError("a consistency group of fewer than two scenes is not a correspondence")
    partners = _partner_map(partition, metric)
    admitted = set()
    for node in list(partners):
        group = consistent_group(node, partners)
        if len(group) < wanted:
            continue
        for (a, i), (b, j) in itertools.combinations(sorted(group), 2):
            admitted.add((a, i, b, j))
    return sorted(admitted)


def measure_consistency_criterion(
        richness_levels: Sequence[int] = (6, 9, 12),
        blocks: Sequence[Sequence[int]] = None,
        null_blocks: Sequence[Sequence[int]] = None,
        *, k: Optional[int] = None, confirmatory: bool = False) -> Dict[str, Any]:
    """Candidate 2's development evaluation, against T4E.12's four declared conditions.

    Unlike candidate 1 this is evaluable at every declared richness, because there is no tail
    model whose support can run out -- which was the first of candidate 1's two feasibility
    failures and is fixed by construction rather than by argument.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)

    def evaluate(seeds, richness, *, plant):
        scenes = [_scene_with_motif_membership(seed, plant=plant, richness=richness,
                                              confirmatory=confirmatory)
                  for seed in seeds]
        incomparable = 0
        trimmed = []
        for scene in scenes:
            keep, dropped = comparable_subset(scene)
            incomparable += len(dropped)
            trimmed.append(([scene[0][i] for i in keep], [scene[1][i] for i in keep]))
        admitted = consistency_admitted_pairs(trimmed, metric, k=k)
        scene_pairs = len(trimmed) * (len(trimmed) - 1) // 2
        configurations = len(trimmed[0][0])
        candidates = sum(len(trimmed[a][0]) * len(trimmed[b][0])
                         for a, b in itertools.combinations(range(len(trimmed)), 2))
        motif_index = [scene[1].index(True) if any(scene[1]) else None for scene in trimmed]
        motif_admitted = motif_total = 0
        if plant:
            for a, b in itertools.combinations(range(len(trimmed)), 2):
                motif_total += 1
                if (a, motif_index[a], b, motif_index[b]) in set(admitted):
                    motif_admitted += 1
        proposed = sum(len(mutual_nearest_matches(trimmed[a], trimmed[b], metric))
                       for a, b in itertools.combinations(range(len(trimmed)), 2))
        row = {
            "seeds": list(seeds), "richness": int(richness),
            "clique_size_required": len(trimmed) if k is None else int(k),
            "configurations_per_scene": configurations,
            "configurations_refused_as_incomparable": incomparable,
            "scene_pairs": scene_pairs, "candidate_pairs": candidates,
            "pairs_proposed_by_candidate_C": proposed,
            "admitted": len(admitted),
            "matched_fraction": (len(admitted) / scene_pairs) / configurations,
        }
        if plant:
            false_admissions = len(admitted) - motif_admitted
            required = required_pair_rate(
                configurations_as_features(configurations, allow_trimmed=True))
            row.update({
                "motif_pairs": motif_total, "motif_pairs_admitted": motif_admitted,
                "false_split_rate": (motif_total - motif_admitted) / motif_total,
                "false_admission_rate": (None if not admitted
                                         else false_admissions / len(admitted)),
                "false_pair_rate": false_admissions / (candidates - motif_total),
                "required_pair_rate": required["required_pair_rate"],
                "shortfall_factor": ((false_admissions / (candidates - motif_total))
                                     / required["required_pair_rate"]),
            })
        return row

    planted, nulls = [], []
    for richness in richness_levels:
        for seeds in blocks:
            planted.append(evaluate(seeds, richness, plant=True))
        for seeds in null_blocks:
            nulls.append(evaluate(seeds, richness, plant=False))
    return {
        "candidate": "2_consistency_across_the_partition",
        "criterion": CRITERION_CONSISTENCY_NAME,
        "confirmatory": bool(confirmatory),
        "richness_levels": [int(value) for value in richness_levels],
        "planted_blocks": planted, "null_blocks": nulls,
        "acceptance_boundary": (
            "Condition 1: worst false split at most 0.10 at every richness. Condition 2: the "
            "matched fraction must FALL as richness grows. Condition 3: the shortfall must be "
            "non-increasing and the 0.10 admission bound met at the richest level. Condition 4: "
            "refusals reported by name."),
        "what_the_null_measures": (
            "Nothing recurs there, so every admitted pair is false by construction and the "
            "count is the criterion's false-positive rate where the answer is none. This is "
            "what candidates C and D failed at, at 116 and 62 matches respectively."),
        "evidence_class": "confirmatory" if confirmatory else "development",
        "boundary": (
            ("Measured on partitions 700-715, opened once by the 2026-09-09 amendment and "
             "spent permanently. This is unseen-data evidence for both error rates, and it "
             "does NOT establish that a blind structural choice would have produced the same "
             "result: k was chosen after a probe on the development blocks. Partitions "
             "720-735 remain reserved and are refused in code. No confirmatory null was "
             "declared, so null evidence remains development only.")
            if confirmatory else
            ("Measured on blocks that informed the declaration and the feasibility probe "
             "disclosed in it. Development evidence only; the reserved confirmatory blocks "
             "are untouched.")),
    }


# ---------------------------------------------------------------------------------------------
# T4E.15 candidate 4: closure under the matching.
#
# Declared in `data/identity_calibration/t4e15-closure-declaration.json` and NOT MEASURED until
# that declaration is adopted.
#
# T4E.14 built evidence where the motif is present in only j of S scenes, so a criterion can at
# last be wrong in both directions on the same data. Candidate 2's k = S is arithmetically dead
# there and candidate 3's k = S - 1 is falsified, so neither may be borrowed.
#
# This candidate drops the clique size entirely. What candidate 2 actually exploited was never
# the number 6 -- it was a conjunction with no loose ends: every motif configuration's partners
# were ONLY other motif configurations. Closure keeps that and discards the size.
#
# A design was discarded for a derivable reason and is recorded rather than forgotten. Ranking
# groups by span and admitting the widest -- the most direct reading of "score how much of a
# window a configuration spans rather than thresholding it" -- is dead on arrival: candidate 3
# showed coincidental groups reach S - 1 = 5, so at j = 3 the widest group in the partition is a
# coincidence and the motif is rejected outright, false split 1.0000 before the rule runs.
#
# What closure reads that a span threshold cannot is the ABSENCE of outside partners. A
# coincidental group whose members also match configurations elsewhere is not a coherent
# identity however far it reaches; a true group matching nothing outside itself is one however
# short it is.

CRITERION_CLOSURE_NAME = "closed_consistent_sets_under_the_mutual_nearest_neighbour_matching"


def is_closed(group: Sequence[Tuple[int, int]], partners) -> bool:
    """True when no member of `group` has a mutual nearest neighbour outside it.

    Closure is tested on the MAXIMAL agreeing set, never on a subset: a subset would
    automatically fail on the members just removed, so testing subsets would admit nothing and
    would not be a different rule but a broken one.
    """
    members = set(group)
    for node in group:
        for scene, index in partners[node].items():
            if (scene, index) not in members:
                return False
    return True


def closure_admitted_pairs(partition, metric: SignatureMetric):
    """T4E.15 candidate 4: admit the pairs inside sets that are consistent AND closed.

    No k, no span threshold, no radius, no ratio, no fitted model, no estimated normaliser. A
    group of three is admitted on the same terms as a group of six, which is what makes this a
    criterion FOR partial recurrence rather than a criterion with its tolerance widened.

    Returns the admitted cross-scene pairs as `(scene_a, index_a, scene_b, index_b)` with
    `scene_a < scene_b`, so the error rates are counted over exactly the population candidates
    C, D, 2 and 3 were counted over.
    """
    partners = _partner_map(partition, metric)
    admitted = set()
    for node in list(partners):
        group = consistent_group(node, partners)
        if len(group) < 2 or not is_closed(group, partners):
            continue
        for (a, i), (b, j) in itertools.combinations(sorted(group), 2):
            admitted.add((a, i, b, j))
    return sorted(admitted)


def _closure_row(scenes, presence, metric, *, richness):
    """One partition's counts, with the population that actually applies to it.

    The pairs a criterion must recover are C(j,2) over the PRESENT scenes, not C(S,2). An error
    rate divided by the wrong population is the quietest way to report a wrong number, so the
    population is derived from the presence labels rather than assumed.
    """
    incomparable = 0
    trimmed = []
    for scene in scenes:
        keep, dropped = comparable_subset(scene)
        incomparable += len(dropped)
        trimmed.append(([scene[0][i] for i in keep], [scene[1][i] for i in keep]))
    admitted = closure_admitted_pairs(trimmed, metric)
    admitted_set = set(admitted)
    present = [index for index, holds in enumerate(presence) if holds]
    motif_index = {index: (trimmed[index][1].index(True) if any(trimmed[index][1]) else None)
                   for index in range(len(trimmed))}
    true_pairs, recovered = 0, 0
    for a, b in itertools.combinations(present, 2):
        if motif_index[a] is None or motif_index[b] is None:
            continue
        true_pairs += 1
        if (a, motif_index[a], b, motif_index[b]) in admitted_set:
            recovered += 1
    absent = {index for index, holds in enumerate(presence) if not holds}
    hallucinated = sum(1 for a, _, b, _ in admitted if a in absent or b in absent)
    candidates = sum(len(trimmed[a][0]) * len(trimmed[b][0])
                     for a, b in itertools.combinations(range(len(trimmed)), 2))
    row = {
        "richness": int(richness),
        "scenes": len(trimmed),
        "scenes_holding_the_motif": len(present),
        "configurations_per_scene": len(trimmed[0][0]),
        "configurations_refused_as_incomparable": incomparable,
        "candidate_pairs": candidates,
        "pairs_proposed_by_candidate_C": sum(
            len(mutual_nearest_matches(trimmed[a], trimmed[b], metric))
            for a, b in itertools.combinations(range(len(trimmed)), 2)),
        "admitted": len(admitted),
        "true_pairs": true_pairs,
        "true_pairs_recovered": recovered,
        "pairs_touching_an_absent_scene": hallucinated,
    }
    row["false_split_rate"] = (None if not true_pairs
                               else (true_pairs - recovered) / true_pairs)
    row["false_admission_rate"] = (None if not admitted
                                   else (len(admitted) - recovered) / len(admitted))
    row["hallucinated_presence_rate"] = (None if not admitted
                                         else hallucinated / len(admitted))
    return row


def measure_closure_criterion(presence_levels: Sequence[int] = None,
                              blocks: Sequence[Sequence[int]] = None,
                              null_blocks: Sequence[Sequence[int]] = None,
                              richness_levels: Sequence[int] = (6, 9, 12)) -> Dict[str, Any]:
    """Candidate 4 on the T4E.14 partial-presence evidence, against its six declared conditions.

    This is the primary measurement: the first in the sequence where BOTH error rates are
    genuinely at risk on the same data. Candidate 4's recall here is not arithmetic -- a motif
    configuration in a present scene may be the mutual nearest neighbour of something unrelated
    in an absent scene, and closure would then reject the motif. Nothing measured so far bears
    on how often that happens.
    """
    presence_levels = (tuple(presence_levels) if presence_levels is not None
                       else PRESENCE_LEVELS)
    blocks = tuple(blocks) if blocks is not None else PARTIAL_PRESENCE_BLOCKS
    null_blocks = (tuple(null_blocks) if null_blocks is not None
                   else PARTIAL_PRESENCE_NULL_BLOCKS)
    metric = SignatureMetric(WEIGHTS)
    planted, nulls, refusals = [], [], []

    for richness in richness_levels:
        for present_count in presence_levels:
            for block in blocks:
                try:
                    scenes, presence = build_partial_presence_partition(
                        block, present_count, richness=richness)
                except UnlabelledScene as error:
                    refusals.append({"block": list(block), "richness": int(richness),
                                     "present_count": int(present_count),
                                     "refused_by_name": str(error)})
                    continue
                row = _closure_row(scenes, presence, metric, richness=richness)
                row.update({"block": list(block), "present_count": int(present_count)})
                planted.append(row)
        for block in null_blocks:
            scenes, presence = build_partial_presence_partition(block, 0, richness=richness)
            row = _closure_row(scenes, presence, metric, richness=richness)
            row.update({"block": list(block), "present_count": 0})
            nulls.append(row)

    return {
        "candidate": "4_closure_under_the_matching",
        "criterion": CRITERION_CLOSURE_NAME,
        "evidence": "t4e14_partial_presence",
        "evidence_class": "development",
        "presence_levels": [int(value) for value in presence_levels],
        "richness_levels": [int(value) for value in richness_levels],
        "planted_blocks": planted, "null_blocks": nulls, "refusals": refusals,
        "acceptance_boundary": (
            "Condition 1: false split at most 0.10 at every j and richness, counted against "
            "C(j,2) and NOT C(S,2) -- and NOT arithmetic this time. Condition 2: false "
            "admission at most 0.10. Condition 3: pairs touching an absent scene at most 0.10 "
            "of admissions, reported separately. Condition 4: both nulls admit ZERO. Condition "
            "5: the total-recurrence cross-check. Condition 6: refusals by name."),
        "why_recall_is_the_informative_half_here": (
            "On total-recurrence evidence every candidate's recall was near-guaranteed by "
            "construction, and candidate 3's was arithmetic outright. Here a motif "
            "configuration in a present scene may be the mutual nearest neighbour of something "
            "unrelated in an absent scene, and closure would then reject the motif. That risk "
            "is the substance of this candidate and it was stated before the run."),
        "boundary": (
            "Development evidence. The T4E.14 partitions were built and audited on 2026-09-10 "
            "and are inspected data permanently; a blind declaration does not make them fresh. "
            "Partial-presence blocks 880-895 have never been built and are refused "
            "unconditionally. Nothing here concerns recurrence across genuinely separated "
            "epochs, which this generator cannot produce."),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def measure_closure_cross_check(blocks: Sequence[Sequence[int]] = None,
                                null_blocks: Sequence[Sequence[int]] = None,
                                richness_levels: Sequence[int] = (6, 9, 12)) -> Dict[str, Any]:
    """Candidate 4 on the TOTAL-recurrence blocks: does closure keep what candidate 2 kept?

    A declared cross-check, not a horse race. Candidate 2 is already adjudicated and candidate 4
    is judged only against its own conditions; this asks whether closure discards coherent
    identities where they are unambiguous, which would be a failure of the rule rather than of
    the evidence.

    That it admits AT LEAST the motif here is derivable and is not a finding: a group spanning
    all S scenes uses a partner in every other scene, and the matching gives at most one partner
    per scene, so those are all of each member's partners and the motif group is closed by
    construction.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)
    planted, nulls = [], []

    for richness in richness_levels:
        for seeds in blocks:
            scenes = [_scene_with_motif_membership(seed, plant=True, richness=richness)
                      for seed in seeds]
            row = _closure_row(scenes, [True] * len(scenes), metric, richness=richness)
            row["block"] = list(seeds)
            planted.append(row)
        for seeds in null_blocks:
            scenes = [_scene_with_motif_membership(seed, plant=False, richness=richness)
                      for seed in seeds]
            row = _closure_row(scenes, [False] * len(scenes), metric, richness=richness)
            row["block"] = list(seeds)
            nulls.append(row)

    return {
        "cross_check": "candidate_4_on_total_recurrence_evidence",
        "criterion": CRITERION_CLOSURE_NAME,
        "evidence_class": "development",
        "planted_blocks": planted, "null_blocks": nulls,
        "this_is_not_a_horse_race": (
            "R20 forbids it and the declaration forbids it by name. Candidate 2 is already "
            "adjudicated. This asks only whether closure discards coherent identities where "
            "they are unambiguous."),
        "what_is_derivable_here_and_is_not_a_finding": (
            "Candidate 4 admits at least the motif's full clique on these blocks, because a "
            "group spanning every scene uses a partner in every other scene and the matching "
            "gives at most one per scene, so the motif group is closed by construction."),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------------------------
# T4E.14: evidence in which recurrence is PARTIAL.
#
# Declared in `data/identity_calibration/t4e14-partial-presence-design.json` and NOT BUILT until
# that declaration is adopted.
#
# Everything measured so far -- candidates B, C, D, 1, 2 and 3 -- ran on partitions where the
# motif is planted in EVERY scene. Recurrence there is total, so no criterion has ever been
# shown a scene in which a true configuration is genuinely absent. Candidate 3's false-split
# column was 0.0000 partly for that reason: there was no true absence for it to mishandle, and
# it was therefore measured on one side only. Its falsification stands; what it could not do is
# fail for the other reason.
#
# This section builds partitions where the motif is present in j of S scenes. It is a test bed,
# not a rule: no criterion is declared, adopted or measured here.
#
# Two things about the existing generator are worth stating because they bound what this can
# repair. Scenes are independent draws with the same motif shape at a random centre and
# rotation, and the blocks are disjoint seed ranges over them -- so scenes within a block differ
# from one another exactly as scenes across blocks do, and "across partitions" is not a distinct
# phenomenon here. This design repairs partial presence. It does NOT supply epoch-to-epoch
# recurrence, which the acquired-record path still owes.

#: The partial-presence development partitions. New seeds, because blocks 100-405 are
#: total-recurrence partitions and rebuilding them at partial presence would put a different
#: artefact behind a familiar name.
PARTIAL_PRESENCE_BLOCKS = tuple(
    tuple(range(start, start + SCENES_PER_PARTITION)) for start in (800, 810, 820, 830))

#: The partial-presence null. Nothing is planted anywhere, so the right answer is nothing --
#: which is a different situation from an ABSENT SCENE inside a partition that does hold the
#: motif elsewhere, and both are needed.
PARTIAL_PRESENCE_NULL_BLOCKS = (tuple(range(850, 850 + SCENES_PER_PARTITION)),)

#: How many of the S scenes hold the motif. j = 6 is total recurrence and is the existing
#: planted partitions, already measured; j < 3 leaves at most one true cross-scene pair, which
#: is no population to discriminate on.
PRESENCE_LEVELS = (3, 4, 5)

#: Reserved partial-presence partitions, declared before a single partial-presence scene
#: existed. Blocks 720-735 are reserved TOTAL-recurrence partitions and cannot serve as
#: confirmatory evidence for a partial-recurrence criterion, because the property under test is
#: absent from them. There is exactly one honest moment to reserve these and it is before any
#: of them exists.
RESERVED_PARTIAL_PRESENCE_SEEDS = frozenset(
    seed for start in (880, 890) for seed in range(start, start + SCENES_PER_PARTITION))


class ReservedPartialPresenceScene(RuntimeError):
    """A reserved partial-presence block was reached, refused by name before it is built.

    There is no `confirmatory=True` escape on this one. No amendment opening these exists, so
    a flag that could open them would be a door left ajar for the convenience of whoever
    arrives next.
    """


def _refuse_reserved_partial_presence(seeds) -> None:
    held = sorted(set(int(seed) for seed in seeds) & RESERVED_PARTIAL_PRESENCE_SEEDS)
    if held:
        raise ReservedPartialPresenceScene(
            "partial-presence seeds %s are reserved and are refused unconditionally. They were "
            "reserved in the T4E.14 design declaration before any partial-presence scene "
            "existed, and spending them needs a criterion that first meets its acceptance on "
            "the development blocks, plus its own amendment." % held)


def presence_pattern(block: Sequence[int], present_count: int,
                     *, size: Optional[int] = None) -> Tuple[int, ...]:
    """Which scenes of a partition hold the motif: a uniformly random subset of size j.

    Deterministic in the block's first seed and j, so the partition is reproducible and the
    pattern is disclosed rather than incidental.

    NOT contiguous, deliberately. These scenes carry no ordering, so planting the motif in a
    contiguous run would introduce a temporal structure the generator does not have and a
    criterion could then exploit -- it would score well by discovering the planting rule.
    """
    scenes = len(tuple(block)) if size is None else int(size)
    wanted = int(present_count)
    if not 0 <= wanted <= scenes:
        raise ValueError(
            "cannot plant the motif in %d of %d scenes" % (wanted, scenes))
    rng = np.random.default_rng(int(tuple(block)[0]) * 1000 + wanted)
    return tuple(sorted(int(index) for index in
                        rng.choice(scenes, size=wanted, replace=False)))


def build_partial_presence_partition(block: Sequence[int], present_count: int,
                                     *, richness: Optional[int] = None):
    """A partition in which the motif is planted in `present_count` of its scenes.

    Returns `(scenes, presence)` where `scenes` are exactly what the criteria already consume
    and `presence[i]` says whether scene `i` holds the motif. The absent scenes are ordinary
    distractor scenes -- same feature count, same minimum-separation rule -- so absence is the
    absence of the motif and not a different kind of scene, and presence cannot be inferred
    from how many features a scene holds.
    """
    seeds = tuple(int(seed) for seed in block)
    _refuse_reserved_partial_presence(seeds)
    present = set(presence_pattern(seeds, present_count))
    scenes, presence = [], []
    for index, seed in enumerate(seeds):
        holds = index in present
        scenes.append(_scene_with_motif_membership(seed, plant=holds, richness=richness))
        presence.append(holds)
    return scenes, presence


def true_motif_pairs(presence: Sequence[bool]) -> int:
    """The cross-scene pairs a criterion is required to recover: C(j, 2), and no others.

    Stated as a function because the population changes with j, and an error rate divided by
    the wrong population is the quietest way to report the wrong number.
    """
    return sum(1 for _ in itertools.combinations(
        [index for index, holds in enumerate(presence) if holds], 2))


def audit_partial_presence(presence_levels: Sequence[int] = PRESENCE_LEVELS,
                           blocks: Sequence[Sequence[int]] = None,
                           null_blocks: Sequence[Sequence[int]] = None,
                           richness_levels: Sequence[int] = (6, 9, 12)) -> Dict[str, Any]:
    """T4E.14's audit: does the constructed evidence have the properties claimed for it?

    This measures the TEST BED, not any rule. It succeeds if the partitions are what the design
    declaration says they are and fails if they are not. No criterion appears anywhere in it.
    """
    blocks = tuple(blocks) if blocks is not None else PARTIAL_PRESENCE_BLOCKS
    null_blocks = (tuple(null_blocks) if null_blocks is not None
                   else PARTIAL_PRESENCE_NULL_BLOCKS)
    rows, refusals = [], []

    def inspect(block, present_count, richness):
        try:
            scenes, presence = build_partial_presence_partition(
                block, present_count, richness=richness)
        except UnlabelledScene as error:
            refusals.append({
                "block": list(block), "present_count": int(present_count),
                "richness": int(richness), "refused_by_name": str(error)})
            return None
        holds = [any(scene[1]) for scene in scenes]
        motif_configurations = [sum(1 for flag in scene[1] if flag) for scene in scenes]
        return {
            "block": list(block), "present_count": int(present_count),
            "richness": int(richness),
            "presence_pattern": list(presence_pattern(block, present_count)),
            "scenes_holding_the_motif": sum(1 for flag in holds if flag),
            "scenes_holding_none": sum(1 for flag in holds if not flag),
            "motif_configurations_per_scene": motif_configurations,
            "configurations_per_scene": [len(scene[0]) for scene in scenes],
            "true_motif_pairs": true_motif_pairs(presence),
        }

    for richness in richness_levels:
        for present_count in presence_levels:
            for block in blocks:
                row = inspect(block, present_count, richness)
                if row is not None:
                    rows.append(row)
        for block in null_blocks:
            row = inspect(block, 0, richness)
            if row is not None:
                row["is_null_partition"] = True
                rows.append(row)

    planted = [row for row in rows if not row.get("is_null_partition")]
    nulls = [row for row in rows if row.get("is_null_partition")]
    return {
        "audit": "t4e14_partial_presence",
        "presence_levels": [int(value) for value in presence_levels],
        "richness_levels": [int(value) for value in richness_levels],
        "partitions": rows,
        "refusals": refusals,
        "condition_1_presence_counts": all(
            row["scenes_holding_the_motif"] == row["present_count"]
            and row["scenes_holding_none"] == SCENES_PER_PARTITION - row["present_count"]
            for row in planted),
        "condition_2_no_leakage_by_configuration_count": all(
            len(set(row["configurations_per_scene"])) == 1 for row in rows),
        "condition_3_labels_refused_rather_than_guessed": {
            "refused": len(refusals),
            "note": ("Scenes whose motif labelling is ambiguous are refused by name by "
                     "_motif_member_ids and counted here. A silent mislabel would put a wrong "
                     "ground truth under every error rate measured on this evidence."),
        },
        "condition_4_absent_scenes_are_ordinary": all(
            sum(1 for count in row["motif_configurations_per_scene"] if count > 0)
            == row["present_count"] for row in planted),
        "condition_5_reproducible": all(
            row["presence_pattern"] == list(presence_pattern(row["block"],
                                                             row["present_count"]))
            for row in planted),
        "the_null_holds_nothing_anywhere": all(
            row["scenes_holding_the_motif"] == 0 for row in nulls),
        "what_is_derivable_and_is_not_a_finding": (
            "Candidate 2 requires a group spanning every scene, so at j < S no such group "
            "containing the motif exists and its false split is 1.0000 here BEFORE it is run. "
            "Candidate 3 recovers the motif only at j = 5. Both are arithmetic, both were "
            "stated in the design declaration, and neither is a discovery about a falsified "
            "or a passing candidate."),
        "what_this_evidence_does_not_supply": (
            "Epoch-to-epoch recurrence. The blocks are disjoint seed ranges over independent "
            "draws, so scenes within a block differ from one another exactly as scenes across "
            "blocks do. Partial presence is repaired here; recurrence across genuinely "
            "separated epochs is not, and the acquired-record path still owes it."),
        "reserved": (
            "Partial-presence blocks 880-895 are refused unconditionally by "
            "ReservedPartialPresenceScene and have never been built."),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------------------------
# T4E.13 candidate 3: partial recurrence, at a clique size below the partition.
#
# Declared in `data/identity_calibration/t4e13-partial-recurrence-declaration.json` and NOT
# MEASURED until that declaration is adopted.
#
# Candidate 2 passed at k = S and its own declaration says why that is not enough: a real
# pattern need not appear in every window, and k = S rejects a configuration absent from a
# single scene outright. This candidate takes the minimal step off that setting -- tolerate
# exactly one absence -- and it is the first criterion in the sequence whose k was fixed before
# ANY measurement at ANY k below S. That is the whole point of it; the relaxation itself is
# small and its transfer value is small with it.
#
# Two things about this candidate are arithmetic rather than evidence, and are derived here
# because candidate 1 failed on properties that were derivable before adoption and were not
# derived:
#
#   * admitted(k) is non-decreasing as k falls, since a group spanning S scenes also spans
#     S - 1. So candidate 3 admits a SUPERSET of candidate 2 on every partition.
#   * candidate 2's false split is 0.0000 everywhere, so candidate 3's is 0.0000 everywhere
#     BEFORE it is run. Acceptance condition 1 is passed by arithmetic and carries no
#     evidential weight. The entire empirical content is on the admission side.
#
# The same monotonicity says the criterion cannot be inert the way candidate 1 was.

#: How many scenes a consistent group may be absent from. Declared as ONE, by the rule "tolerate
#: exactly one absence" -- the only k below S nameable without choosing a free fraction. Not to
#: be moved after measurement: a different tolerance is a different candidate needing its own
#: declaration, and it will not be able to make this one's blindness claim.
ABSENCES_TOLERATED = 1
CRITERION_PARTIAL_NAME = "mutual_nearest_neighbour_consistency_tolerating_one_absence"

#: The clique sizes swept for the declared k profile. Characterisation of how the criterion
#: degrades as the tolerance widens, which is what the mining path needs. It adjudicates
#: NOTHING: the criterion under evaluation is k = S - 1 and R20 forbids picking a winner from
#: a sweep. See `the_k_profile` in the declaration.
PROFILE_ABSENCES = (0, 1, 2, 3)


def consistency_k_for(partition_size: int, *, absences: int = ABSENCES_TOLERATED) -> int:
    """The declared clique size as a FUNCTION of the partition size, not a constant.

    The criterion is "at most `absences` scenes missing", so k tracks S. A criterion whose k
    were the literal number 5 would silently become stricter or laxer when S changed and its
    error rates would not be comparable across partitions of different sizes.
    """
    partition_size, absences = int(partition_size), int(absences)
    if absences < 0:
        raise ValueError("a negative number of tolerated absences is not a relaxation")
    k = partition_size - absences
    if k < 2:
        raise ValueError(
            "tolerating %d absences in a %d-scene partition leaves k = %d, and a correspondence "
            "needs at least two scenes" % (absences, partition_size, k))
    return k


def _partition_size(blocks: Sequence[Sequence[int]]) -> int:
    """One S for the whole measurement, or a refusal by name.

    k is a function of S, so blocks of differing sizes would be measured under different
    criteria and their error rates pooled as though they were one. That is refused rather than
    averaged.
    """
    sizes = {len(seeds) for seeds in blocks}
    if len(sizes) != 1:
        raise ValueError(
            "blocks of sizes %s cannot share one criterion, because k is a function of the "
            "partition size" % (sorted(sizes),))
    return sizes.pop()


def measure_partial_recurrence_criterion(
        richness_levels: Sequence[int] = (6, 9, 12),
        blocks: Sequence[Sequence[int]] = None,
        null_blocks: Sequence[Sequence[int]] = None,
        *, absences: int = ABSENCES_TOLERATED,
        confirmatory: bool = False) -> Dict[str, Any]:
    """Candidate 3 at k = S - `absences`, against T4E.13's five declared conditions.

    Delegates to candidate 2's measurement with the derived k, so the population, the proposer
    and both error-rate definitions are identical and candidates C, D, 2 and 3 stay directly
    comparable. That identity is a declared property of this candidate, not a convenience.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    scenes = _partition_size(tuple(blocks) + tuple(null_blocks))
    k = consistency_k_for(scenes, absences=absences)
    report = measure_consistency_criterion(
        richness_levels=richness_levels, blocks=blocks, null_blocks=null_blocks,
        k=k, confirmatory=confirmatory)
    report.update({
        "candidate": "3_partial_recurrence_at_k_below_the_partition_size",
        "criterion": CRITERION_PARTIAL_NAME,
        "partition_size": scenes,
        "absences_tolerated": int(absences),
        "clique_size_required": k,
        "acceptance_boundary": (
            "Condition 1: worst false split at most 0.10 -- DECLARED IN ADVANCE TO BE PASSED BY "
            "ARITHMETIC, by monotonicity from candidate 2, and carrying no evidential weight. "
            "Condition 2: the matched fraction must FALL as richness grows. Condition 3: false "
            "admission at most 0.10 at every richness and a non-increasing shortfall. Condition "
            "4: the null must admit ZERO, not near zero. Condition 5: refusals reported by "
            "name, verdict to include INVALID."),
        "what_is_arithmetic_rather_than_evidence": (
            "admitted(k) is non-decreasing as k falls, so candidate 3 admits a superset of "
            "candidate 2 and inherits its 0.0000 false split before being run. The recall half "
            "of this measurement is not a finding. The entire empirical content is how many "
            "coincidental groups span exactly %d scenes, and whether that grows with richness."
            % (k,)),
        "blindness": (
            "k was fixed by the rule 'tolerate exactly one absence' in the T4E.13 declaration, "
            "before any measurement at any k below the partition size. What was already known "
            "and is disclosed there: candidate 2's exact run shows no coincidental group "
            "reaches S, and a greedy probe found some reaching S - 1. That points away from "
            "tuning -- k = S - 1 sits AT the observed coincidental ceiling and is the value "
            "most likely to fail."),
    })
    return report


def measure_k_profile(richness_levels: Sequence[int] = (6, 9, 12),
                      blocks: Sequence[Sequence[int]] = None,
                      null_blocks: Sequence[Sequence[int]] = None,
                      *, absences: Sequence[int] = PROFILE_ABSENCES,
                      confirmatory: bool = False) -> Dict[str, Any]:
    """How the criterion degrades as the tolerance widens. Characterisation, not adjudication.

    The mining path needs to know the shape of this curve and no measurement so far supplies
    it. It decides nothing: the criterion under evaluation is k = S - 1, judged alone against
    its own conditions. Any other k made attractive by this sweep is a further candidate that
    needs its own declaration -- and one that will NOT be able to claim blindness, because this
    profile will have been seen. That consequence is declared in advance rather than discovered.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    scenes = _partition_size(tuple(blocks) + tuple(null_blocks))
    rungs = []
    for tolerated in absences:
        report = measure_partial_recurrence_criterion(
            richness_levels=richness_levels, blocks=blocks, null_blocks=null_blocks,
            absences=tolerated, confirmatory=confirmatory)
        rungs.append({
            "absences_tolerated": int(tolerated),
            "clique_size_required": consistency_k_for(scenes, absences=tolerated),
            "planted_blocks": report["planted_blocks"], "null_blocks": report["null_blocks"],
        })
    return {
        "profile": "t4e13_k_profile",
        "partition_size": scenes,
        "declared_operating_point": consistency_k_for(scenes),
        "rungs": rungs,
        "evidence_class": "confirmatory" if confirmatory else "development",
        "this_adjudicates_nothing": (
            "R20 forbids the horse race and the T4E.13 declaration forbids it by name. The "
            "criterion under evaluation is k = S - 1. A k selected from this sweep is a new "
            "candidate needing its own declaration and its own evidence, and it cannot claim "
            "its structural choice was blind."),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------------------------
# T4E.12 candidate 1: a rule that counts its own comparisons.
#
# Declared in `data/identity_calibration/t4e12-multiplicity-declaration.json` (sha256
# f0c45d92...) and NOT MEASURED until that declaration is adopted.
#
# Candidates B, C and D each applied the same decision to every candidate pair in ignorance of
# how many pairs there were. At six features a scene pair offers 400; at twelve it offers
# 726,000. So their admissions grew with the population while the truth stayed at one pair per
# scene pair, and the shortfall widened x33, x184, x423. This candidate makes the admission
# threshold alpha / m^2, which tightens by construction as scenes get richer.
#
# The obvious candidate -- recalibrating the attribute weights, which are all 1.0 and never were
# calibrated -- is NOT this one, and the reason is derivable rather than measured: a
# nearest-neighbour matching returns at most one pair per configuration whatever the weights
# are, so reweighting changes which pairs match and never how many.

#: The declared family-wise error rate: expected false admissions per ordered scene pair if the
#: tail model holds. Declared before measurement and not to be moved.
ADMISSION_ALPHA = 0.05
#: The tail model's threshold, as a percentile of the scene pair's own distance population.
TAIL_PERCENTILE = 1.0
#: Below this many exceedances the fit is refused rather than trusted.
MIN_EXCEEDANCES = 50
CRITERION_MULTIPLICITY_NAME = "multiplicity_aware_tail_admission"


class TailModelRefused(Exception):
    """The tail model could not be fitted, so no admission decision is available, by name.

    Admitting on a failed fit would make the rule most permissive exactly where its model is
    least supported. A refusal contributes to neither error rate.
    """


def fit_lower_tail(distances: Sequence[float], *, percentile: float = TAIL_PERCENTILE,
                   min_exceedances: int = MIN_EXCEEDANCES):
    """A generalised Pareto fitted to how small the small distances get. Peaks over threshold.

    Extreme-value theory is what licenses stating a tail probability of 1e-7 from a few hundred
    thousand samples: the exceedances below a high threshold converge to this family whatever
    the parent distribution is. The distances are negated so the LOWER tail becomes an upper
    one, which is the orientation `genpareto` is written for.

    Refuses rather than extrapolates when the support is thin or the fit does not converge.
    """
    values = np.asarray(list(distances), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_exceedances:
        raise TailModelRefused(
            "%d finite distances is below the %d needed to fit a tail"
            % (values.size, min_exceedances))
    threshold = float(np.percentile(values, percentile))
    exceedances = -(values[values <= threshold]) + threshold
    exceedances = exceedances[exceedances > 0.0]
    if exceedances.size < min_exceedances:
        raise TailModelRefused(
            "%d exceedances below the %.1fth percentile is below the %d needed"
            % (exceedances.size, percentile, min_exceedances))
    try:
        shape, location, scale = genpareto.fit(exceedances, floc=0.0)
    except Exception as error:                                   # pragma: no cover - scipy path
        raise TailModelRefused("the generalised Pareto fit did not converge: %s" % (error,))
    if not (math.isfinite(shape) and math.isfinite(scale)) or scale <= 0.0:
        raise TailModelRefused(
            "the fit returned shape %r and scale %r, which describe no distribution"
            % (shape, scale))
    rate = float(exceedances.size) / float(values.size)
    return {"threshold": threshold, "shape": float(shape), "scale": float(scale),
            "exceedance_rate": rate, "exceedances": int(exceedances.size),
            "population": int(values.size)}


def tail_probability(distance: float, model: Dict[str, Any]) -> float:
    """How surprising this distance is under the fitted lower tail. Small means surprising."""
    if distance > model["threshold"]:
        # Above the threshold the model says nothing sharper than the empirical rate.
        return 1.0
    excess = model["threshold"] - float(distance)
    survival = float(genpareto.sf(excess, model["shape"], loc=0.0, scale=model["scale"]))
    probability = model["exceedance_rate"] * survival
    if not math.isfinite(probability):
        raise TailModelRefused("the tail probability is not finite")
    return probability


def multiplicity_aware_matches(left, right, metric: SignatureMetric,
                               *, alpha: float = ADMISSION_ALPHA):
    """T4E.12 candidate 1. Candidate C proposes the pairs; this decides which survive.

    A matched pair is admitted only when a distance as small as its own would arise with
    probability at most `alpha / m^2` under a tail model fitted to that same scene pair's own
    cross-scene distances. The threshold therefore tightens automatically as scenes get richer
    -- 1.25e-4 at six features, 6.9e-8 at twelve -- which is the 1/m^2 scaling the acceptance
    demands, obtained by construction rather than by calibration.

    The null is the scene pair's own distances, so the decision is invariant to rescaling the
    metric within a partition, as candidates C and D were. Returns `(pairs, model)`; raises
    `TailModelRefused` rather than deciding without a model.
    """
    if not left[0] or not right[0]:
        return [], None
    distances = _pair_distances(left, right, metric)
    flat = [value for row in distances for value in row]
    model = fit_lower_tail(flat)
    candidates = len(left[0]) * len(right[0])
    bound = alpha / float(candidates)
    model = dict(model, candidate_pairs=candidates, admission_bound=bound, alpha=alpha)
    kept = [(i, j) for i, j in mutual_nearest_matches(left, right, metric)
            if tail_probability(distances[i][j], model) <= bound]
    return kept, model


def comparable_subset(scene):
    """The configurations of one scene that share its dominant signature family, and the rest.

    Found while measuring T4E.12 candidate 1, and named here rather than worked around. At
    twelve features a small number of configurations are so nearly isotropic that
    `sign_constellations` refuses their principal axis -- correctly, because
    `AXIS_ISOTROPY_FLOOR` says an elongation that small is indistinguishable from noise -- so
    they carry no bearing block. `SignatureMetric` then refuses to compare them with anything
    that does, because distances between different measured quantities are not numbers.

    Measured extent: none at six or nine features in any block, and 1, 1 and 2 configurations
    out of 1,320 in three of the four blocks at twelve. The planted motif carries a bearing
    block in every scene of every block at every richness measured, so excluding these cannot
    manufacture a recall the criterion did not earn -- but the counts are reported in every row
    rather than asserted, because that is the reader's judgement and not this function's.

    The excluded configurations are a REFUSAL and not a rejection: they are not counted as
    splits, admissions or candidate pairs. A configuration the metric cannot compare has not
    been judged.
    """
    points, is_motif = scene[0], scene[1]
    if not points:
        return [], []
    families = collections.Counter(point.family.has_bearings for point in points)
    dominant = families.most_common(1)[0][0]
    keep = [index for index, point in enumerate(points)
            if point.family.has_bearings == dominant]
    dropped = [index for index in range(len(points)) if index not in set(keep)]
    if any(is_motif[index] for index in dropped):
        raise UnlabelledScene(
            "the planted motif is in the minority signature family, so excluding that family "
            "would remove the answer; refused rather than measured around")
    return keep, dropped


def measure_multiplicity_criterion(
        richness_levels: Sequence[int] = (6, 9, 12),
        blocks: Sequence[Sequence[int]] = None,
        null_blocks: Sequence[Sequence[int]] = None,
        *, alpha: float = ADMISSION_ALPHA) -> Dict[str, Any]:
    """T4E.12 candidate 1's development evaluation, against the four conditions declared for it.

    Adopted in `t4e12-multiplicity-adoption.json` at the hash the declaration was committed
    under. The conditions are richness-wise rather than block-wise, because the scaling
    measurement showed an admission fraction is a property of the signature and the scene
    together: recall at every richness, a matched fraction that FALLS as richness grows, a
    shortfall that does not widen with the bound met at the richest level, and refusals by name.

    Candidate D's matched fraction is the arithmetic reference for condition 2 and candidate C
    proposes the pairs. Neither is adjudication; both are already falsified.
    """
    blocks = tuple(blocks) if blocks is not None else EXCHANGEABILITY_BLOCKS
    null_blocks = tuple(null_blocks) if null_blocks is not None else (NULL_SEEDS,)
    metric = SignatureMetric(WEIGHTS)

    def evaluate(seeds, richness, *, plant):
        partition = [_scene_with_motif_membership(seed, plant=plant, richness=richness)
                     for seed in seeds]
        admitted = motif_admitted = motif_total = candidates = refused = 0
        proposed = margin_kept = incomparable = 0
        shapes = []
        trimmed = []
        for scene in partition:
            keep, dropped = comparable_subset(scene)
            incomparable += len(dropped)
            trimmed.append(([scene[0][i] for i in keep], [scene[1][i] for i in keep]))
        for left, right in itertools.combinations(trimmed, 2):
            pair = (left, right)
            candidates += len(left[0]) * len(right[0])
            proposed += len(mutual_nearest_matches(*pair, metric))
            try:
                margin_kept += len(ratio_margin_matches(*pair, metric))
            except UndefinedMargin:
                pass
            try:
                pairs, model = multiplicity_aware_matches(*pair, metric, alpha=alpha)
            except TailModelRefused:
                refused += 1
                continue
            if model is not None:
                shapes.append(model["shape"])
            admitted += len(pairs)
            if plant:
                motif_total += 1
                if (left[1].index(True), right[1].index(True)) in pairs:
                    motif_admitted += 1
        scene_pairs = len(partition) * (len(partition) - 1) // 2
        configurations = len(trimmed[0][0])
        row = {
            "seeds": list(seeds), "richness": int(richness),
            "configurations_per_scene": configurations,
            "configurations_refused_as_incomparable": incomparable,
            "scene_pairs": scene_pairs, "scene_pairs_refused": refused,
            "candidate_pairs": candidates,
            "pairs_proposed_by_candidate_C": proposed,
            "pairs_kept_by_candidate_D": margin_kept,
            "admitted": admitted,
            "admitted_per_scene_pair": (None if scene_pairs == refused
                                        else admitted / (scene_pairs - refused)),
            "matched_fraction": (None if scene_pairs == refused else
                                 (admitted / (scene_pairs - refused)) / configurations),
            "candidate_D_matched_fraction": (
                None if not scene_pairs else (margin_kept / scene_pairs) / configurations),
            "fitted_shape_range": ([min(shapes), max(shapes)] if shapes else None),
        }
        if plant:
            false_admissions = admitted - motif_admitted
            required = required_pair_rate(
                configurations_as_features(configurations, allow_trimmed=True))
            row.update({
                "motif_pairs": motif_total, "motif_pairs_admitted": motif_admitted,
                "false_split_rate": (None if not motif_total
                                     else (motif_total - motif_admitted) / motif_total),
                "false_admission_rate": (None if not admitted
                                         else false_admissions / admitted),
                "false_pair_rate": false_admissions / (candidates - motif_total),
                "required_pair_rate": required["required_pair_rate"],
                "shortfall_factor": ((false_admissions / (candidates - motif_total))
                                     / required["required_pair_rate"]),
            })
        return row

    planted, nulls = [], []
    for richness in richness_levels:
        for seeds in blocks:
            planted.append(evaluate(seeds, richness, plant=True))
        for seeds in null_blocks:
            nulls.append(evaluate(seeds, richness, plant=False))

    def pooled(richness):
        rows = [row for row in planted if row["richness"] == richness]
        splits = [row["false_split_rate"] for row in rows
                  if row.get("false_split_rate") is not None]
        fractions = [row["matched_fraction"] for row in rows
                     if row.get("matched_fraction") is not None]
        shortfalls = [row["shortfall_factor"] for row in rows
                      if row.get("shortfall_factor") is not None]
        admissions = [row["false_admission_rate"] for row in rows
                      if row.get("false_admission_rate") is not None]
        return {
            "richness": int(richness),
            "worst_false_split": max(splits) if splits else None,
            "matched_fraction_range": [min(fractions), max(fractions)] if fractions else None,
            "false_admission_range": [min(admissions), max(admissions)] if admissions else None,
            "worst_shortfall_factor": max(shortfalls) if shortfalls else None,
        }

    return {
        "candidate": "1_multiplicity_aware_admission_via_a_declared_tail_model",
        "criterion": CRITERION_MULTIPLICITY_NAME, "alpha": alpha,
        "richness_levels": [int(value) for value in richness_levels],
        "planted_blocks": planted, "null_blocks": nulls,
        "by_richness": [pooled(value) for value in richness_levels],
        "acceptance_boundary": (
            "Condition 1: worst false split at most 0.10 at every richness. Condition 2: the "
            "matched fraction must FALL as richness grows rather than holding near candidate "
            "D's 0.23. Condition 3: the shortfall must be non-increasing across the three "
            "levels and the 0.10 admission bound met at the richest. Condition 4: refusals "
            "reported by name."),
        "the_null_tests_the_tail_model_not_the_signature": (
            "If the tail model is correct, admissions per scene pair equal alpha by "
            "construction whatever the signature is like. So a null admission count near alpha "
            "means the model holds and any remaining failure is the signature's; far above it "
            "means the fit is wrong here and the signature has not been tested at all."),
        "evidence_class": "development",
        "boundary": ("Measured on the blocks that informed the declaration and its acceptance. "
                     "Development evidence only; the reserved confirmatory blocks are "
                     "untouched."),
    }


def configurations_as_features(configurations: int, cardinality: int = CARDINALITY,
                              *, allow_trimmed: bool = False) -> int:
    """Invert C(f, k) for the small feature counts this task uses. Exact, and refuses otherwise.

    `allow_trimmed` accepts a count slightly below an exact C(f, k), which is what a scene looks
    like after `comparable_subset` has removed configurations the metric cannot compare. It
    returns the smallest feature count whose configuration total is at least the one given, so
    the required per-pair rate is computed against the population the scene would have had --
    the stricter of the two readings.
    """
    for features in range(cardinality, 200):
        total = math.comb(features, cardinality)
        if total == configurations or (allow_trimmed and total >= configurations):
            return features
    raise ValueError("%d is not C(f, %d) for any feature count under 200"
                     % (configurations, cardinality))


# ---------------------------------------------------------------------------------------------
# T4E.12: what scene richness does to the identity problem.
#
# The T4E.11 diagnostic left a number: candidate D's per-pair false rate is about 1%, against
# prior odds of 1 true correspondence in 400 candidate pairs. Six features at cardinality three
# give C(6,3) = 20 configurations, and a scene pair therefore offers 20 x 20 = 400 pairs while
# holding exactly one true correspondence.
#
# Both halves of that scale with the feature count, and they scale differently. Configurations
# grow as C(f,3), candidate pairs as its square, and the number of true correspondences does not
# grow at all. So the admission FRACTION is not a property of the signature: it is a property of
# the signature and the scene richness together, and a bound met on one record need not hold on
# a richer one. That is T4E.10's transfer problem again, in a place no declaration had looked.
#
# `required_pair_rate` derives the consequence, and `measure_richness_scaling` measures whether
# the per-pair rate itself stays put as richness grows -- which arithmetic cannot settle.


def configuration_count(features: int, cardinality: int = CARDINALITY) -> int:
    """How many configurations a scene of `features` features yields. C(f, k)."""
    return math.comb(int(features), int(cardinality))


def required_pair_rate(features: int, *, admission_bound: float = MAX_FALSE_ADMISSION,
                       true_pairs_per_scene_pair: int = 1,
                       cardinality: int = CARDINALITY) -> Dict[str, Any]:
    """The per-candidate-pair false rate an admission bound demands, at a given richness.

    Arithmetic, not measurement. With `m = C(f, k)` configurations per scene, an ordered scene
    pair offers `m^2` candidate pairs and holds `t` true correspondences, so an admission
    fraction of `a` permits `t * a / (1 - a)` false admissions and therefore a per-pair rate of
    that over `m^2 - t`. The rate falls roughly as the fourth power of the feature count while
    the signature stays the same, which is the whole point of computing it.
    """
    if not 0.0 < admission_bound < 1.0:
        raise ValueError("an admission bound outside (0, 1) does not name an operating point")
    configurations = configuration_count(features, cardinality)
    candidates = configurations * configurations
    permitted = true_pairs_per_scene_pair * admission_bound / (1.0 - admission_bound)
    return {
        "features": int(features),
        "configurations_per_scene": configurations,
        "candidate_pairs_per_scene_pair": candidates,
        "true_pairs_per_scene_pair": int(true_pairs_per_scene_pair),
        "false_admissions_permitted": permitted,
        "required_pair_rate": permitted / (candidates - true_pairs_per_scene_pair),
        "boundary": ("Arithmetic. It says what a bound demands, not what any signature "
                     "delivers, and it approves nothing."),
    }


def measure_richness_scaling(feature_counts: Sequence[int] = (6, 9, 12),
                             seeds: Sequence[int] = CALIBRATION_SEEDS,
                             *, tau: float = MARGIN_TAU) -> Dict[str, Any]:
    """Does the per-pair false rate hold as scenes get richer? Arithmetic cannot say; this can.

    A DIAGNOSTIC. Candidate D is used as a probe of the signature, not as a candidate being
    selected: it is already falsified, and nothing here adopts it or anything else. What is
    being measured is the signature's behaviour as the candidate set grows, which is the
    question T4E.12 exists to ask.

    If the per-pair rate is roughly constant in richness, then the admission fraction degrades
    as the fourth power of the feature count and no fixed bound transfers between records of
    different density. If instead the rate falls as richness grows, the picture is better than
    the arithmetic alone suggests. Either answer is a result.
    """
    metric = SignatureMetric(WEIGHTS)
    rows = []
    for count in feature_counts:
        partition = [_scene_with_motif_membership(seed, plant=True, richness=count)
                     for seed in seeds]
        matched = motif_matched = motif_total = candidates = 0
        for left, right in itertools.combinations(partition, 2):
            pair = ((left[0], left[1]), (right[0], right[1]))
            pairs = ratio_margin_matches(*pair, metric, tau=tau)
            matched += len(pairs)
            candidates += len(left[0]) * len(right[0])
            left_motif = left[1].index(True)
            right_motif = right[1].index(True)
            motif_total += 1
            if (left_motif, right_motif) in pairs:
                motif_matched += 1
        false_admissions = matched - motif_matched
        required = required_pair_rate(count)
        rows.append({
            "features": int(count),
            "configurations_per_scene": len(partition[0][0]),
            "candidate_pairs": candidates,
            "matches_returned": matched,
            "motif_pairs": motif_total, "motif_pairs_matched": motif_matched,
            "false_split_rate": (motif_total - motif_matched) / motif_total,
            "false_admission_rate": (None if not matched else false_admissions / matched),
            "false_pair_rate": false_admissions / (candidates - motif_total),
            "required_pair_rate": required["required_pair_rate"],
            "shortfall_factor": (false_admissions / (candidates - motif_total))
                                / required["required_pair_rate"],
        })
    return {
        "diagnostic": "how the identity problem scales with scene richness",
        "is_a_diagnostic_not_a_criterion": (
            "Candidate D is a probe here, not a candidate under selection; it is already "
            "falsified. Nothing is adopted, no threshold is chosen and no operating point is "
            "reported."),
        "tau": tau, "seeds": list(seeds), "rows": rows,
        "evidence_class": "development",
        "boundary": ("Synthetic scenes at several densities settle a property of the "
                     "signature, not of the atmosphere. The reserved confirmatory blocks are "
                     "untouched and no defect is closed."),
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
