"""Phase G3, TG3.5: mining for a configuration that recurs, and paying for the search.

The acceptance is the pair of benchmark gates. `4E.motif_recovery` mines six training
scenes, freezes what it found before the held-out scenes exist, and confirms one motif on
a partition it was not mined from; `4E.motif_null` runs the identical pass over scenes with
nothing planted in them and confirms nothing. Per this tree's own Definition of Done the
second is the load-bearing one, because the failure mode the whole phase is built against
is a mining pass that finds a motif in anything.

Four findings are load-bearing, and each is measured here rather than asserted.

*   **The family is the number of configurations looked at, and it is unaffordable.** Six
    scenes of six features at `k = 3` is 120 members, which TG3.1 prices at 12,885
    surrogates before one of them could be rejected. The generate/confirm split is not an optimisation in this slice,
    it is the only shape that can be paid for, and the benchmark refuses a pass if its own
    generate family ever becomes affordable in one stage - because then it would have
    stopped testing the thing it exists to test.
*   **Matching under a tolerance is not transitive.** A matches B and B matches C without A
    matching C, so a candidate is a star around one exemplar rather than a cluster, and
    `count_intransitive` measures how often the difference would have mattered.
*   **Ranking by raw count prefers the promiscuous.** Support saturates at the number of
    scenes, and the ties at the cap are broken by how many configurations an exemplar
    matched - which has to prefer *fewer*, or the top of the ranking fills with shapes that
    match everything.
*   **Invariance is not sufficiency.** `always_matches` is exactly invariant to rotation,
    translation and rescaling, and this file registers it into TG3.4's `MATCHERS` and shows
    that the `4E.invariance` audit calls it honest. What separates it from a matcher that
    measures something is the null, not the invariance.

Most of the file runs on exact synthetic geometry, where a configuration can be built to
one decimal and the statistics can be handed data they are supposed to fail. The two
benchmarks are run once, at the end, because that is the only place the whole pipeline -
extractor, matcher, family, seal and ledger - is under test at once.
"""

import math

import numpy as np
import pytest

from src.benchmarks.core import Outcome, get_benchmark
from src.core.constellation import RelationContext
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.family import FAMILY_COMBINATORS, SearchAxis, SearchTerm
from src.core.feature import (
    FeatureLocation,
    Quantity,
    Significance,
    SpectralFeature,
)
from src.core.invariance import (
    MATCHERS,
    Presentation,
    audit_declared_invariance,
    build_signature,
    calibrate_match_tolerance,
)
from src.core.motif import (
    CONFIGURATION_COMBINATORS,
    DEFAULT_MATCHER,
    MAX_PLACEMENT_ATTEMPTS,
    MotifCandidate,
    MotifEvidence,
    MotifSizeUnavailableError,
    Occurrence,
    Scene,
    SurrogateInfeasibleError,
    affordable_candidate_count,
    always_matches,
    candidates_from,
    choose_candidates,
    confirm_motifs,
    confirmatory_specification,
    count_intransitive,
    deduplicate_candidates,
    enumerate_occurrences,
    freeze_motifs,
    mine,
    motif_p_value,
    motif_search_specification,
    observed_minimum_separation,
    report_motif_generation,
    signatures_match,
    support_of,
    surrogate_scene,
)
from src.core.preregistration import HeldOutLedger, PartitionIdentity
from src.core.registry import restore, snapshot

#: The benchmark's own geometry, so nothing here is priced off an invented number.
ARM_RATIOS = (0.62, 1.0, 1.45)
SIDE_CELLS = 60.0
SIGMA_CELLS = 6.0
MIN_SEPARATION = 30.0
GRID_CELLS = 256.0

AXES = (AxisSpec("row", "space", "cells", ordinal=0),
        AxisSpec("col", "space", "cells", ordinal=1))


def _feature(row, col, sigma=SIGMA_CELLS, dataset="motif"):
    return SpectralFeature(
        domain="synthetic", dataset=dataset, variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({"row": row, "col": col}, AXES),
        time=0.0, representation="identity",
        spatial_scale=Quantity(sigma, "cells"),
        significance=Significance(0.01, "surrogate_quantile"))


def _motif(centre=(128.0, 128.0), rotation_deg=0.0, arms=ARM_RATIOS, side=SIDE_CELLS,
           jitter=0.0, seed=0):
    """The planted shape, exactly, wherever and however turned - optionally jittered.

    The jitter stands in for the extractor's localisation error without paying for an
    extraction, and it displaces row and column independently: a wobble that scaled with
    the configuration would make a larger motif no better measured than a small one, which
    is not how a localisation error behaves.
    """
    rng = np.random.default_rng(seed)
    radius = side / math.sqrt(3.0)
    out = []
    for index in range(3):
        angle = math.radians(rotation_deg + 120.0 * index - 90.0)
        arm = radius * arms[index]
        wobble = (0.0, 0.0) if not jitter else tuple(jitter * rng.normal(size=2))
        out.append(_feature(centre[0] + arm * math.sin(angle) + wobble[0],
                            centre[1] + arm * math.cos(angle) + wobble[1]))
    return out


def _scene(name, seed, *, plant=True, n_distractors=3, jitter=0.15):
    """One scene: the motif somewhere, plus distractors that are not it."""
    rng = np.random.default_rng(seed)
    features = []
    if plant:
        features += _motif(centre=(float(rng.uniform(90.0, 166.0)),
                                   float(rng.uniform(90.0, 166.0))),
                           rotation_deg=float(rng.uniform(0.0, 360.0)),
                           jitter=jitter, seed=seed)
    guard = 0
    while len(features) < 3 + n_distractors:
        guard += 1
        assert guard < 5000
        candidate = _feature(float(rng.uniform(40.0, 216.0)), float(rng.uniform(40.0, 216.0)))
        if all(float(candidate.separation_to(other).value) >= MIN_SEPARATION
               for other in features):
            features.append(candidate)
    return Scene(name, tuple(features))


def _partition(name, count, split, start=0):
    return PartitionIdentity(name, count, 1, ("amplitude",), (start, start + count),
                             {"split": split})


TOLERANCE = 0.02


# ------------------------------------------------------- 1. what the search costs to run


def test_the_family_is_every_subset_of_every_scene():
    scenes = [_scene("s%d" % i, 10 + i) for i in range(6)]
    spec = motif_search_specification(scenes, size=3, n_surrogates=199)
    assert spec.family_size == 6 * math.comb(6, 3) == 120
    assert len(spec.labels()) == 120
    assert spec.labels()[0] == "s0|0-1-2"


def test_the_priced_family_and_the_pass_are_the_same_search_member_for_member():
    """A count check would pass on two different searches that happen to be the same size.

    TG3.1 built `labels()` for exactly this comparison, and `mine` makes it rather than
    trusting that an enumeration written twice stays written twice.
    """
    scenes = [_scene("s%d" % i, 20 + i) for i in range(3)]
    spec = motif_search_specification(scenes, size=3, n_surrogates=199)
    occurrences = enumerate_occurrences(scenes, size=3)
    assert tuple(o.label for o in occurrences) == spec.labels()


def test_mining_refuses_when_the_enumeration_drifts_from_the_declaration(monkeypatch):
    scenes = [_scene("s%d" % i, 30 + i) for i in range(3)]
    real = enumerate_occurrences

    def short(*args, **kwargs):
        return real(*args, **kwargs)[:-1]

    monkeypatch.setattr("src.core.motif.enumerate_occurrences", short)
    with pytest.raises(InvalidParameterError, match="member for member"):
        mine(scenes, size=3, tolerance=TOLERANCE, n_surrogates=199)


def test_the_generate_family_cannot_be_paid_for_in_one_stage():
    """The phase's boundary, as a number rather than an argument.

    120 members needs more surrogates than any ensemble this benchmark runs, which is why
    the split exists. If this ever became affordable the benchmark would have stopped
    testing the thing it exists to test, and the gate refuses a pass on that ground.
    """
    scenes = [_scene("s%d" % i, 40 + i) for i in range(6)]
    result = mine(scenes, size=3, tolerance=TOLERANCE, n_surrogates=199)
    assert result.specification.family_size == 120
    assert result.affordable_here is False
    assert result.specification.account().surrogates_required > 199


def test_a_size_with_no_registered_combinator_is_refused_rather_than_computed():
    scenes = [_scene("s%d" % i, 50 + i, n_distractors=5) for i in range(2)]
    with pytest.raises(MotifSizeUnavailableError, match="registered family combinator"):
        motif_search_specification(scenes, size=5, n_surrogates=199)


def test_the_subset_combinators_count_what_they_enumerate():
    """TG3.1's rule, applied to the entries this module registered.

    A combinator whose count and enumeration disagree makes a refusal and a receipt
    describe different searches, and nothing else in the tree would notice.
    """
    for size, name in sorted(CONFIGURATION_COMBINATORS.items()):
        term = SearchTerm(name, (SearchAxis("feature", tuple(range(7))),))
        assert term.size == math.comb(7, size) == len(term.members())
        assert term.n_components == size


def test_the_registration_came_from_here_and_not_from_an_edit_to_family():
    for name in ("unordered_triples", "unordered_quadruples"):
        assert name in FAMILY_COMBINATORS
        assert FAMILY_COMBINATORS.entry(name).capabilities["distinct"] is True


def test_scenes_of_different_sizes_have_no_single_family_size():
    scenes = [_scene("s0", 60), _scene("s1", 61, n_distractors=4)]
    with pytest.raises(InvalidParameterError, match="same number of features"):
        motif_search_specification(scenes, size=3, n_surrogates=199)


def test_one_scene_cannot_demonstrate_recurrence():
    with pytest.raises(InvalidParameterError, match="more than one scene"):
        motif_search_specification([_scene("s0", 70)], size=3, n_surrogates=199)


def test_two_scenes_with_one_name_would_be_counted_once():
    with pytest.raises(InvalidParameterError, match="distinct scene names"):
        motif_search_specification([_scene("s", 71), _scene("s", 72)], size=3,
                                   n_surrogates=199)


# --------------------------------------------- 2. a motif is an exemplar, not a cluster


def test_matching_under_a_tolerance_is_not_transitive():
    """The measurement behind the design decision, not an argument for it.

    Three shapes a little apart in a chain: the ends match the middle and not each other.
    A single-linkage cluster would call all three one motif with a support of three.
    """
    left = build_signature(DEFAULT_MATCHER, _motif(arms=(0.62, 1.0, 1.45)))
    middle = build_signature(DEFAULT_MATCHER, _motif(arms=(0.62, 1.0, 1.50)))
    right = build_signature(DEFAULT_MATCHER, _motif(arms=(0.62, 1.0, 1.55)))
    assert signatures_match(left, middle, 0.02)
    assert signatures_match(middle, right, 0.02)
    assert not signatures_match(left, right, 0.02)


def test_a_candidate_is_the_star_around_its_exemplar():
    scenes = [Scene("a", tuple(_motif(arms=(0.62, 1.0, 1.45)) + _motif(
        centre=(60.0, 60.0), arms=(0.62, 1.0, 1.50)))),
        Scene("b", tuple(_motif(centre=(70.0, 180.0), arms=(0.62, 1.0, 1.55)) + _motif(
            centre=(190.0, 70.0), arms=(0.62, 1.0, 1.45))))]
    occurrences = enumerate_occurrences(scenes, size=3)
    candidates = candidates_from(occurrences, tolerance=0.02)
    by_label = {c.label: c for c in candidates}
    for candidate in candidates:
        for member in candidate.occurrences:
            assert signatures_match(candidate.exemplar.graph, member.graph, 0.02)
    assert any(len(by_label[c.label].occurrences) < len(occurrences)
               for c in candidates)


def test_the_intransitive_pairs_are_counted_rather_than_hidden():
    scenes = [Scene("a", tuple(_motif(arms=(0.62, 1.0, 1.45)))),
              Scene("b", tuple(_motif(arms=(0.62, 1.0, 1.50)))),
              Scene("c", tuple(_motif(arms=(0.62, 1.0, 1.55))))]
    result = mine(scenes, size=3, tolerance=0.02, n_surrogates=199)
    assert result.intransitive_pairs == 1


def test_a_configuration_that_does_not_match_itself_is_not_a_motif():
    occurrence = Occurrence("a", (0, 1, 2), build_signature(DEFAULT_MATCHER, _motif()))
    other = Occurrence("b", (0, 1, 2), build_signature(DEFAULT_MATCHER, _motif()))
    with pytest.raises(InvalidParameterError, match="does not match itself"):
        MotifCandidate(exemplar=occurrence, occurrences=(other,), n_examined=2)


def test_one_subset_has_one_label():
    with pytest.raises(InvalidParameterError, match="ascending order"):
        Occurrence("a", (2, 0, 1), build_signature(DEFAULT_MATCHER, _motif()))


# --------------------------------------------------------------- 3. what a support counts


def test_support_counts_scenes_and_not_occurrences():
    """Two overlapping triangles in one scene are one piece of evidence.

    A support that counted occurrences would grow fastest where the configurations share
    features, which is exactly where they carry the least independent information.
    """
    shape = _motif()
    crowded = Scene("a", tuple(shape + _motif(centre=(60.0, 60.0))))
    other = Scene("b", tuple(_motif(centre=(180.0, 70.0), rotation_deg=41.0)
                             + _motif(centre=(70.0, 190.0), rotation_deg=41.0)))
    result = mine([crowded, other], size=3, tolerance=TOLERANCE, n_surrogates=199)
    top = result.ranked[0]
    assert top.support == 2
    assert len(top.occurrences) > 2


def test_the_ranking_prefers_the_shape_that_matched_fewer_things():
    """The key that is easy to write backwards, and what happens when it is.

    Support saturates at the number of scenes, so the ties at the cap decide the ranking.
    A shape matching three configurations per scene is a looser shape than one matching
    exactly one - not a stronger finding.
    """
    scenes = [_scene("s%d" % i, 80 + i) for i in range(4)]
    result = mine(scenes, size=3, tolerance=0.05, n_surrogates=199)
    ranked = result.ranked
    tied = [c for c in ranked if c.support == ranked[0].support]
    assert len(tied) > 1
    counts = [len(c.occurrences) for c in tied]
    assert counts == sorted(counts)


def test_specificity_is_reported_rather_than_thresholded():
    scenes = [_scene("s%d" % i, 90 + i) for i in range(3)]
    result = mine(scenes, size=3, tolerance=TOLERANCE, n_surrogates=199)
    for candidate in result.candidates:
        assert 0.0 <= candidate.specificity < 1.0
        assert candidate.describe()["specificity"] == candidate.specificity


def test_the_same_shape_enters_the_ranking_once_per_scene_and_is_frozen_once():
    """Six names for one hypothesis would be a correction unit of six paid for one test."""
    scenes = [Scene("s%d" % i, tuple(_motif(centre=(100.0 + 8 * i, 100.0),
                                            rotation_deg=17.0 * i)))
              for i in range(4)]
    result = mine(scenes, size=3, tolerance=TOLERANCE, n_surrogates=199)
    assert len(result.candidates) == 4
    assert len(deduplicate_candidates(result.ranked, tolerance=TOLERANCE)) == 1


def test_the_frozen_family_never_exceeds_what_the_ensemble_can_reject_one_of():
    scenes = [_scene("s%d" % i, 110 + i) for i in range(5)]
    result = mine(scenes, size=3, tolerance=0.06, n_surrogates=199)
    assert affordable_candidate_count(199) == 4
    assert len(choose_candidates(result)) <= 4
    with pytest.raises(InvalidParameterError, match="cannot pay for"):
        choose_candidates(result, n_candidates=5)


# ------------------------------------------------------------------ 4. what a null draws


def test_a_surrogate_moves_the_features_and_changes_nothing_else():
    scene = _scene("a", 120)
    drawn = surrogate_scene(scene, rng=np.random.default_rng(0),
                            bounds={"row": (40.0, 216.0), "col": (40.0, 216.0)},
                            minimum_separation=MIN_SEPARATION)
    assert len(drawn) == len(scene)
    for before, after in zip(scene.features, drawn.features):
        assert after.magnitude == before.magnitude
        assert after.spatial_scale == before.spatial_scale
        assert after.provenance == before.provenance
    assert any(after.location.coords != before.location.coords
               for before, after in zip(scene.features, drawn.features))


def test_the_surrogate_respects_the_separation_the_data_demonstrates():
    scene = _scene("a", 121)
    minimum = observed_minimum_separation([scene])
    drawn = surrogate_scene(scene, rng=np.random.default_rng(1),
                            bounds={"row": (40.0, 216.0), "col": (40.0, 216.0)},
                            minimum_separation=minimum)
    assert observed_minimum_separation([drawn]) >= minimum - 1e-9


def test_an_impossible_constraint_is_refused_rather_than_relaxed():
    scene = _scene("a", 122)
    with pytest.raises(SurrogateInfeasibleError, match="not a parameter to relax"):
        surrogate_scene(scene, rng=np.random.default_rng(2),
                        bounds={"row": (100.0, 110.0), "col": (100.0, 110.0)},
                        minimum_separation=MIN_SEPARATION)
    assert MAX_PLACEMENT_ATTEMPTS > 1


def test_an_unconstrained_null_is_easier_than_the_data_and_shrinks_every_p_value():
    """The measurement behind the constraint, rather than an argument for it.

    Without the minimum separation a surrogate can place two features closer than the
    extractor could have resolved them, and the triangles it then draws are near-degenerate
    slivers the pipeline could not have returned. Such a null supports the motif *less*
    often than a real arrangement does, so the observed support looks more surprising than
    it is: over five seeds the p-value roughly halves, from 0.20-0.25 to 0.07-0.14. A
    p-value made smaller by the null's construction is exactly what R18 is about.
    """
    scenes = [_scene("s%d" % i, 130 + i) for i in range(4)]
    exemplar = build_signature(DEFAULT_MATCHER, list(scenes[0].features[:3]))
    honest = motif_p_value(exemplar, scenes, size=3, tolerance=0.05, n_surrogates=99,
                           seed=5)
    loose = motif_p_value(exemplar, scenes, size=3, tolerance=0.05, n_surrogates=99,
                          seed=5, minimum_separation=1e-6)
    assert loose.p_value < honest.p_value
    assert loose.surrogate_support != honest.surrogate_support


def test_the_p_value_floor_is_never_zero():
    evidence = MotifEvidence(label="m", support=4, n_scenes=4, n_surrogates=99,
                             p_value=0.01)
    assert evidence.power_floor == pytest.approx(0.01)
    assert evidence.vacuous(0.005) is True
    assert evidence.vacuous(0.05) is False
    with pytest.raises(InvalidParameterError, match="never zero"):
        MotifEvidence(label="m", support=4, n_scenes=4, n_surrogates=99, p_value=0.0)


def test_support_stops_at_the_first_occurrence_in_a_scene():
    shape = _motif()
    crowded = Scene("a", tuple(shape + _motif(centre=(60.0, 60.0))))
    exemplar = build_signature(DEFAULT_MATCHER, shape)
    assert support_of(exemplar, [crowded, Scene("b", tuple(_motif(centre=(190.0, 60.0))))],
                      size=3, tolerance=TOLERANCE) == 2


# ------------------------------------------------- 5. invariance is not sufficiency


@pytest.fixture
def constant_matcher():
    """`always_matches`, registered only for the duration of one test.

    Registered here rather than in `src/`, so nothing outside these tests can be graded by
    a matcher that measures nothing - and the `4E.invariance` gate is never asked to hold
    an opinion about it.
    """
    state = snapshot(MATCHERS)
    try:
        MATCHERS.add("always_matches", always_matches,
                     description="A constant. Measures nothing, matches everything.",
                     capabilities={"invariant_to": ("rotation", "translation", "rescaling"),
                                   "min_nodes": 3, "crosses_domains": True,
                                   "keys_on": "constant"})
        yield "always_matches"
    finally:
        restore(MATCHERS, state)


def test_the_constant_matcher_is_genuinely_invariant_to_everything(constant_matcher):
    """And the invariance audit calls it honest, which is the whole point.

    It declares invariance to all three transforms and it has all three: its signature does
    not change under any of them, because its signature does not change under anything.
    TG3.4's audit is working correctly when it passes this - invariance is a true property
    of the matcher, and being true is not the same as being useful.
    """
    reference = _motif(jitter=0.15, seed=999)
    replicates = [_motif(jitter=0.15, seed=i) for i in range(12)]
    presentations = []
    for index, angle in enumerate((37.0, 71.0, 211.0)):
        presentations.append(Presentation(
            "rotation/%g" % angle, ("rotation",),
            _motif(rotation_deg=angle, jitter=0.15, seed=10 + index)))
    for index, centre in enumerate(((150.0, 100.0), (90.0, 170.0), (140.0, 150.0))):
        presentations.append(Presentation(
            "translation/%d" % index, ("translation",),
            _motif(centre=centre, jitter=0.15, seed=20 + index)))
    for index, factor in enumerate((0.5, 1.5, 2.0)):
        presentations.append(Presentation(
            "rescaling/%g" % factor, ("rescaling",),
            _motif(side=SIDE_CELLS * factor, jitter=0.15, seed=30 + index)))
    reports = audit_declared_invariance(reference, presentations, replicates)
    constant = reports[constant_matcher]
    assert constant.honest
    assert set(constant.measured) == {"rotation", "translation", "rescaling"}
    assert not constant.vacuous


def test_the_constant_matcher_measures_no_size_and_says_so(constant_matcher):
    """A matcher can recognise a shape without ever having measured how big it was.

    TG3.4 recovered a scale ratio from the separations the matcher carried. This one
    carries none, and the refusal is by name rather than a `KeyError` out of a dictionary.
    """
    reference = _motif(jitter=0.15, seed=999)
    replicates = [_motif(jitter=0.15, seed=i) for i in range(12)]
    presentations = [Presentation("rescaling/2", ("rescaling",),
                                  _motif(side=SIDE_CELLS * 2, jitter=0.15, seed=41),
                                  expected_scale_ratio=2.0)]
    for index, angle in enumerate((37.0, 71.0)):
        presentations.append(Presentation(
            "rotation/%g" % angle, ("rotation",),
            _motif(rotation_deg=angle, jitter=0.15, seed=50 + index)))
    report = audit_declared_invariance(reference, presentations,
                                       replicates)[constant_matcher]
    assert report.scale_recovery is None
    ratio = report.matches["rescaling/2"].scale_ratio
    assert not ratio
    assert "measured no length" in ratio.unavailable


def test_what_separates_the_constant_matcher_from_a_real_one_is_the_null(constant_matcher):
    """It supports every motif in every surrogate scene, so its p-value is the maximum."""
    scenes = [_scene("s%d" % i, 140 + i) for i in range(4)]
    exemplar = always_matches(list(scenes[0].features[:3]), RelationContext())
    evidence = motif_p_value(exemplar, scenes, size=3, tolerance=TOLERANCE,
                             n_surrogates=49, seed=11, matcher=constant_matcher)
    assert evidence.support == len(scenes)
    assert evidence.p_value == 1.0


def test_the_constant_matcher_is_not_registered_in_src():
    assert "always_matches" not in MATCHERS


# ----------------------------------------------------------- 6. generate, freeze, confirm


@pytest.fixture(scope="module")
def mined():
    scenes = [_scene("train%d" % i, 200 + i) for i in range(5)]
    return scenes, mine(scenes, size=3, tolerance=TOLERANCE, n_surrogates=199,
                        study_id="tests")


def test_the_generate_stage_produces_candidates_and_says_so(mined):
    _, result = mined
    report = report_motif_generation(result, train=_partition("train", 5, "train"))
    assert report["stage"] == "generate"
    assert "not findings" in report["claim_boundary"]
    assert report["generate_family_size"] == result.specification.family_size
    assert set(report["candidates"]) <= set(result.specification.labels())


def test_mining_the_held_out_partition_is_refused_by_name(mined):
    _, result = mined
    with pytest.raises(InvalidParameterError, match="already used the data"):
        report_motif_generation(result, train=_partition("test", 5, "test"))


def test_the_confirmatory_family_lies_inside_the_generated_one(mined):
    _, result = mined
    chosen = choose_candidates(result)
    confirm = confirmatory_specification(chosen, n_surrogates=199)
    assert set(confirm.labels()) <= set(result.specification.labels())
    assert confirm.family_size == len(chosen)


def test_a_motif_frozen_twice_would_inflate_the_correction_unit(mined):
    _, result = mined
    chosen = choose_candidates(result)
    with pytest.raises(InvalidParameterError, match="distinct exemplars"):
        confirmatory_specification(tuple(chosen) + tuple(chosen), n_surrogates=199)


def test_freezing_prices_the_confirmatory_family_at_its_own_size(mined):
    _, result = mined
    seal, frozen = freeze_motifs(result, held_out=_partition("test", 5, "test", 5),
                                 sealed_at="2026-08-25T00:00:00Z", n_surrogates=199,
                                 study_id="tests")
    assert seal.confirm_family_size == len(frozen)
    assert seal.confirm_labels == tuple(c.label for c in frozen)
    assert seal.generate_family_size == result.specification.family_size
    assert seal.verify()["seal_sha256"] == seal.seal_sha256


def test_a_seal_cannot_be_frozen_against_the_partition_it_was_mined_on(mined):
    _, result = mined
    with pytest.raises(InvalidParameterError, match="not mined"):
        freeze_motifs(result, held_out=_partition("train", 5, "train"),
                      sealed_at="2026-08-25T00:00:00Z", n_surrogates=199)


def test_a_motif_tested_under_a_frozen_label_must_be_the_signature_that_was_frozen(mined):
    """R20: redefinition is refused by machinery, not discouraged by a comment."""
    _, result = mined
    held = [_scene("test%d" % i, 300 + i) for i in range(4)]
    seal, frozen = freeze_motifs(result, held_out=_partition("test", 4, "test", 5),
                                 sealed_at="2026-08-25T00:00:00Z", n_surrogates=49)
    elsewhere = Occurrence("elsewhere", (0, 1, 2),
                           build_signature(DEFAULT_MATCHER, _motif()))
    swapped = (MotifCandidate(exemplar=elsewhere, occurrences=(elsewhere,),
                              n_examined=1),) * len(frozen)
    with pytest.raises(InvalidParameterError, match="redefinition"):
        confirm_motifs(seal, swapped, scenes=held, held_out=_partition("test", 4, "test", 5),
                       ledger=HeldOutLedger(), opened_at="2026-08-25T01:00:00Z",
                       size=3, tolerance=TOLERANCE, seed=1, n_surrogates=19)


def test_the_held_out_partition_is_spent_once(mined):
    _, result = mined
    held = [_scene("test%d" % i, 400 + i) for i in range(4)]
    identity = _partition("test-once", 4, "test", 5)
    ledger = HeldOutLedger()
    seal, frozen = freeze_motifs(result, held_out=identity,
                                 sealed_at="2026-08-25T00:00:00Z", n_surrogates=49,
                                 ledger=ledger)
    kwargs = dict(scenes=held, held_out=identity, ledger=ledger,
                  opened_at="2026-08-25T01:00:00Z", size=3, tolerance=TOLERANCE,
                  seed=2, n_surrogates=19)
    first = confirm_motifs(seal, frozen, **kwargs)
    assert first["correction_unit"] == len(frozen)
    assert len(first["motifs"]) == len(frozen)
    with pytest.raises(Exception, match="(?i)opened"):
        confirm_motifs(seal, frozen, **kwargs)


def test_the_receipt_carries_the_supports_beside_the_p_values(mined):
    _, result = mined
    held = [_scene("test%d" % i, 500 + i) for i in range(4)]
    identity = _partition("test-receipt", 4, "test", 5)
    ledger = HeldOutLedger()
    seal, frozen = freeze_motifs(result, held_out=identity,
                                 sealed_at="2026-08-25T00:00:00Z", n_surrogates=49,
                                 ledger=ledger)
    receipt = confirm_motifs(seal, frozen, scenes=held, held_out=identity, ledger=ledger,
                             opened_at="2026-08-25T01:00:00Z", size=3,
                             tolerance=TOLERANCE, seed=3, n_surrogates=49)
    assert len(receipt["supports"]) == len(frozen)
    assert receipt["minimum_separation"] > 0.0
    assert all(0.0 < p <= 1.0 for p in receipt["p_values"])
    assert "opened once" in receipt["claim_boundary"]


# ------------------------------------------------------------------- 7. the gates


@pytest.fixture(scope="module")
def motif_results():
    bench = get_benchmark("planted_motif")
    return bench, bench.run()


@pytest.fixture(scope="module")
def null_results():
    bench = get_benchmark("motif_null")
    return bench, bench.run()


def test_the_planted_motif_is_recovered_and_confirmed(motif_results):
    _, results = motif_results
    recovery = [r for r in results if r.stage == "4E.motif_recovery"]
    assert len(recovery) == 1
    assert recovery[0].outcome is Outcome.PASS, recovery[0].detail
    assert recovery[0].measured["receipt"]["n_rejected"] == 1


def test_the_confirmed_motif_was_corrected_over_the_family_that_was_frozen(motif_results):
    _, results = motif_results
    receipt = [r for r in results if r.stage == "4E.motif_recovery"][0].measured["receipt"]
    assert receipt["correction_unit"] == len(receipt["labels"])
    assert min(receipt["adjusted"]) < 0.05


def test_nothing_is_confirmed_where_nothing_was_planted(null_results):
    """The load-bearing result of this slice."""
    _, results = null_results
    null = [r for r in results if r.stage == "4E.motif_null"]
    assert len(null) == 1
    assert null[0].outcome is Outcome.PASS, null[0].detail
    assert null[0].measured["receipt"]["n_rejected"] == 0


def test_the_null_benchmark_really_looked(null_results):
    """A null that proposed nothing is a pass obtained by not looking.

    The null run examines the same family, freezes real candidates - the shapes that most
    recurred in its own training scenes - and puts every one of them to the same ensemble.
    """
    _, results = null_results
    evidence = [r for r in results if r.stage == "4E.motif_null"][0].measured
    assert evidence["mining"]["n_examined"] == evidence["mining"]["generate_family_size"]
    assert evidence["receipt"]["correction_unit"] >= 1
    assert evidence["mining"]["affordable_in_one_stage"] is False


def test_both_benchmarks_examined_the_same_search(motif_results, null_results):
    """The only difference between them is whether anything was planted."""
    planted = [r for r in motif_results[1] if r.stage == "4E.motif_recovery"][0].measured
    null = [r for r in null_results[1] if r.stage == "4E.motif_null"][0].measured
    assert planted["mining"]["generate_family_size"] == null["mining"]["generate_family_size"]
    assert planted["mining"]["size"] == null["mining"]["size"]
    assert planted["mining"]["matcher"] == null["mining"]["matcher"]


def test_the_null_benchmark_is_declared_a_null():
    assert get_benchmark("motif_null").is_null is True
