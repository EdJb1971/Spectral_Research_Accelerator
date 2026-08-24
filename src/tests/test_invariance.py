"""Phase G3, TG3.4: the same configuration, moved, turned and resized.

The acceptance is the benchmark gate itself. `4E.invariance` was the last stage in the
suite still reporting `NOT_YET_RUNNABLE`, and it now reports **PASS**: a matcher declaring
invariance to rotation, translation and rescaling is measured to have all three, and the
position-memorising control registered beside it is measured to have none - failing not by
a special case written for it but by the general rule applied to every registered matcher.

Three findings are load-bearing here, and each is measured rather than asserted.

*   **A dimensionless relation is only as invariant as what it divided by.** TG3.3's
    `distance` divides a separation by the features' *estimated* spatial scale, and the
    estimate drifts with the scale being estimated. On exact synthetic geometry `distance`
    is perfectly scale-invariant; on the benchmark it moves by 4.8% at `scale_factor=3`
    against a 3.3% noise floor. The relation is not at fault and neither is the extractor -
    what is measured here is that the two compose badly, and that a ratio of one separation
    to another avoids the composition entirely.
*   **The extractor does not return its features in a stable order.** Comparing replicate
    node 0 with replicate node 0 measures which blob happened to be brightest. Minimising
    over correspondences instead cut the position matcher's noise floor from 0.27 to 0.026 -
    and at 0.27 that matcher called a 37-degree rotation the same picture.
*   **A noise floor is an operating point, not a test.** Thresholding thirteen
    presentations at the largest of sixty-six noise samples rejects a genuinely invariant
    matcher about one time in five. That is R18's subject, and the remedy is R18's: test the
    presentations against the whole null with a rank test, correct alpha across the family,
    and refuse a pass from a test that could not have failed.

Most of the file runs on exact synthetic geometry, where the truth is known to the last
decimal and the statistics can be given data they are supposed to fail. The benchmark is
run once, at the end, because that is the only place the whole pipeline is under test.
"""

import math

import numpy as np
import pytest

from src.benchmarks.core import Outcome, get_benchmark
from src.core.constellation import RelationContext, constellation
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.feature import (
    FeatureLocation,
    Quantity,
    Significance,
    SpectralFeature,
)
from src.core.invariance import (
    DEFAULT_ALPHA,
    MATCHERS,
    RAW_SEPARATION,
    SHAPE_RATIO,
    TRANSFORMS,
    InvarianceTest,
    MatchTolerance,
    Presentation,
    ScaleRatio,
    ScaleRatioUnavailableError,
    audit_declared_invariance,
    best_deviation,
    build_signature,
    calibrate_match_tolerance,
    declared_invariance,
    match,
    matcher_for,
    measure_invariance,
    null_deviations,
    recover_scale_ratio,
    relative_geometry,
    scale_normalised,
    scale_recovery_null,
)
from src.core.registry import restore, snapshot

#: The benchmark's own geometry, so nothing here is priced off an invented number.
TRIANGLE_SIDE_CELLS = 40.0
FEATURE_SIGMA_CELLS = 6.0
GRID_CELLS = 256

#: Enough replicates that the rank test has power to spare: C(12, 2) = 66 null values, so
#: the smallest p-value a five-presentation comparison can return is 7.7e-08.
REPLICATES = 12

AXES = (AxisSpec("row", "space", "cells", ordinal=0),
        AxisSpec("col", "space", "cells", ordinal=1))


def _triangle(*, rotation_deg=0.0, translation=(0.0, 0.0), scale=1.0, jitter=0.0,
              seed=0, units="cells", spacing=1.0, domain="synthetic",
              dataset="planted_configuration", variable="amplitude",
              magnitude_units=None, sides=(1.0, 1.0, 1.0)):
    """The planted triangle, exactly, under a transform - and optionally jittered.

    `jitter` is a per-feature localisation error in cells, drawn from `seed`, standing in
    for the extractor's without paying for an extraction. It is what makes a synthetic
    replicate a *replicate* rather than a copy: a null measured from copies is zero, and
    zero is not a measurement of noise. It displaces row and column independently and does
    not scale with the configuration - which is how a localisation error actually behaves,
    and the reason a larger configuration is measured proportionally more precisely.

    `sides` scales each vertex's radius independently, which breaks the equilateral
    symmetry - useful where a test needs a configuration only one correspondence recovers.
    """
    axes = (AxisSpec("row", "space", units, ordinal=0),
            AxisSpec("col", "space", units, ordinal=1))
    radius = (TRIANGLE_SIDE_CELLS * scale * spacing) / math.sqrt(3.0)
    centre = (GRID_CELLS / 2.0) * spacing
    rng = np.random.default_rng(seed)
    features = []
    for index in range(3):
        angle = math.radians(rotation_deg + 120.0 * index - 90.0)
        arm = radius * sides[index]
        wobble = (0.0, 0.0) if not jitter else tuple(jitter * rng.normal(size=2))
        features.append(SpectralFeature(
            domain=domain, dataset=dataset, variable=variable,
            magnitude=Quantity(1.0, magnitude_units),
            location=FeatureLocation(
                {"row": centre + (translation[0] + wobble[0]) * spacing
                        + arm * math.sin(angle),
                 "col": centre + (translation[1] + wobble[1]) * spacing
                        + arm * math.cos(angle)}, axes),
            time=0.0, representation="identity",
            spatial_scale=Quantity(FEATURE_SIGMA_CELLS * scale * spacing, units),
            significance=Significance(0.01, "surrogate_quantile")))
    return features


def _reference(**kwargs):
    """The configuration everything is compared against - itself a measurement.

    Jittered like every replicate. An exact reference would make each presentation's
    deviation a single localisation error while the null carried the difference of two,
    which is a null wider than what it is judging by a factor of root two.
    """
    return _triangle(jitter=0.4, seed=999, **kwargs)


def _replicates(n=REPLICATES, **kwargs):
    """`n` measurements of one configuration, differing only in localisation error."""
    return [_triangle(jitter=0.4, seed=index, **kwargs) for index in range(n)]


def _presentations(**kwargs):
    """One presentation per transform, three each, plus a combined one."""
    out = []
    for index, angle in enumerate((37.0, 71.0, 211.0)):
        out.append(Presentation("rotation/%g" % angle, ("rotation",),
                                _triangle(rotation_deg=angle, jitter=0.4,
                                          seed=100 + index, **kwargs)))
    for index, shift in enumerate(((25.0, -30.0), (-40.0, 15.0), (18.0, 22.0))):
        out.append(Presentation("translation/%d" % index, ("translation",),
                                _triangle(translation=shift, jitter=0.4,
                                          seed=200 + index, **kwargs)))
    for index, factor in enumerate((0.5, 1.5, 3.0)):
        out.append(Presentation("rescaling/%g" % factor, ("rescaling",),
                                _triangle(scale=factor, jitter=0.4,
                                          seed=300 + index, **kwargs),
                                expected_scale_ratio=factor))
    out.append(Presentation(
        "combined", TRANSFORMS,
        _triangle(rotation_deg=17.0, translation=(-20.0, 40.0), scale=1.5, jitter=0.4,
                  seed=400, **kwargs),
        expected_scale_ratio=1.5))
    return out


@pytest.fixture
def matchers_registry():
    """Register a matcher for one test without leaking it into the next."""
    state = snapshot(MATCHERS)
    try:
        yield MATCHERS
    finally:
        restore(MATCHERS, state)


@pytest.fixture(scope="module")
def synthetic_audit():
    """One audit over exact synthetic geometry - the statistics with truth known."""
    return audit_declared_invariance(_reference(), _presentations(), _replicates())


# ============================================ 1. what survives a transform, and what does not


def test_the_shape_is_identical_under_exact_rotation_translation_and_rescaling():
    """No tolerance, no noise: the quantity itself does not move.

    Everything downstream is a question about measurement error. This is the question about
    the quantity, and it is asked first because a statistic defending an invariant that is
    not one would be defending nothing.
    """
    reference = build_signature("relative_geometry", _triangle())
    for overrides in ({"rotation_deg": 37.0}, {"translation": (25.0, -30.0)},
                      {"scale": 3.0},
                      {"rotation_deg": 17.0, "translation": (-20.0, 40.0), "scale": 1.5}):
        moved = build_signature("relative_geometry", _triangle(**overrides))
        deviation, _ = best_deviation(reference, moved)
        assert deviation < 1e-12, overrides


def test_the_position_matcher_moves_under_every_one_of_them():
    """The control, on the same exact data - it is not failing because of noise."""
    reference = build_signature("absolute_position", _triangle())
    for overrides in ({"rotation_deg": 37.0}, {"translation": (25.0, -30.0)},
                      {"scale": 3.0}):
        moved = build_signature("absolute_position", _triangle(**overrides))
        deviation, _ = best_deviation(reference, moved)
        assert deviation > 0.05, overrides


def test_the_scale_normalised_relation_is_exactly_invariant_when_the_scale_is_exact():
    """The finding that decided the slice, stated from the other side.

    TG3.3's `distance` divides by the features' spatial scale, and on geometry where that
    scale is known exactly the relation does not move at all under a rescaling. Its failure
    on the benchmark is therefore not a property of the relation. It is what happens when a
    relation divides by an *estimate* whose bias is a function of the thing being estimated.
    """
    reference = scale_normalised(_triangle(), RelationContext())
    moved = scale_normalised(_triangle(scale=3.0), RelationContext())
    deviation, _ = best_deviation(reference, moved)
    assert deviation < 1e-12


def test_a_configuration_of_a_different_shape_does_not_match():
    """Scale-invariance is not permission to match anything.

    An equilateral triangle and a scalene one are different configurations at every size,
    and a matcher that divided the scale out and then reported them the same would have
    reported that all three-feature configurations are one configuration.
    """
    reference = build_signature("relative_geometry", _triangle())
    scalene = build_signature("relative_geometry", _triangle(sides=(1.0, 1.6, 0.7)))
    deviation, _ = best_deviation(reference, scalene)
    assert deviation > 0.1


def test_the_shape_ratios_of_an_equilateral_triangle_are_all_one():
    graph = build_signature("relative_geometry", _triangle())
    for edge in graph.edges.values():
        assert abs(edge[SHAPE_RATIO].value - 1.0) < 1e-12


def test_the_shape_is_the_same_number_in_two_domains_that_share_no_units():
    """TG3.3's cross-domain acceptance, carried into the invariant this slice keys on.

    Cells over cells and metres over metres are the same number, and unlike TG3.3's
    `distance` this one divided by a measurement rather than by a model - so it is
    dimensionless without borrowing anything's accuracy.
    """
    cells = build_signature("relative_geometry", _triangle(sides=(1.0, 1.6, 0.7)))
    metres = build_signature("relative_geometry", _triangle(
        sides=(1.0, 1.6, 0.7), units="m", spacing=30.0, domain="ocean",
        dataset="sst_l4", variable="temperature", magnitude_units="K"))
    assert cells.attributes == metres.attributes
    for pair, edge in cells.edges.items():
        assert abs(edge[SHAPE_RATIO].value
                   - metres.edges[pair][SHAPE_RATIO].value) < 1e-12


def test_the_two_domains_disagree_about_the_raw_separation_they_agree_about_the_shape():
    """The negative control: the same pair of configurations, keyed on the undivided number."""
    cells = build_signature("absolute_position", _triangle())
    metres = build_signature("absolute_position", _triangle(
        units="m", spacing=30.0, domain="ocean", dataset="sst_l4",
        variable="temperature", magnitude_units="K"))
    left = cells.edge(0, 1)[RAW_SEPARATION].value
    right = metres.edge(0, 1)[RAW_SEPARATION].value
    assert abs(right / left - 30.0) < 1e-9


# ============================================================== 2. two features have no shape


def test_a_pair_of_features_is_refused_by_the_shape_matcher():
    """A single edge normalised by itself is 1.0 for every configuration in the world.

    A matcher that returns the same signature for everything reports a discovery on every
    pair it is shown, so the refusal is not a size limit - it is the difference between a
    matcher and a constant.
    """
    with pytest.raises(InvalidParameterError) as excinfo:
        build_signature("relative_geometry", _triangle()[:2])
    assert "at least 3 features" in str(excinfo.value)
    assert "ratio" in str(excinfo.value)


def test_the_scale_normalised_comparison_is_the_one_that_works_on_a_pair():
    """Which is why TG3.3 divided by the modelled scale, and was right to."""
    graph = scale_normalised(_triangle()[:2], RelationContext())
    assert graph.n_nodes == 2
    assert graph.relations == ("distance",)


def test_the_minimum_is_read_from_the_registry_not_hard_coded():
    assert MATCHERS.entry("relative_geometry").capabilities["min_nodes"] == 3
    assert MATCHERS.entry("absolute_position").capabilities["min_nodes"] == 2


# ================================================= 3. the noise floor, and the shuffle in it


def test_the_deviation_is_minimised_over_correspondences_not_taken_node_for_node():
    """The defect that inflated the position matcher's floor to 0.27.

    A deterministic extractor does not return its features in a stable order between noise
    realisations - the order follows which blob happened to be brightest. Comparing node 0
    with node 0 measures that shuffle, and a floor built from it is wide enough to call a
    rotated configuration the same picture.
    """
    reference = build_signature("absolute_position", _triangle(sides=(1.0, 1.6, 0.7)))
    shuffled = _triangle(sides=(1.0, 1.6, 0.7))
    shuffled = [shuffled[2], shuffled[0], shuffled[1]]
    permuted = build_signature("absolute_position", shuffled)
    deviation, _ = best_deviation(reference, permuted)
    assert deviation < 1e-12


def test_the_null_is_every_pair_of_replicates():
    null = null_deviations(_replicates(), matcher="relative_geometry")
    assert len(null) == REPLICATES * (REPLICATES - 1) // 2
    assert all(value > 0.0 for value in null)


def test_the_tolerance_is_the_largest_deviation_the_noise_produced():
    replicates = _replicates()
    tolerance = calibrate_match_tolerance(replicates, matcher="relative_geometry")
    null = null_deviations(replicates, matcher="relative_geometry")
    assert tolerance.value == max(null)
    assert tolerance.n_replicates == REPLICATES


def test_the_tolerance_states_the_false_rejection_rate_it_comes_with():
    """It is not zero, and a tolerance that did not say so would be claiming it was."""
    tolerance = calibrate_match_tolerance(_replicates(), matcher="relative_geometry")
    assert "false-rejection rate" in tolerance.basis
    assert "%.3g" % (1.0 / 67.0) in tolerance.basis


def test_one_replicate_is_refused_because_it_measures_nothing():
    with pytest.raises(InvalidParameterError) as excinfo:
        calibrate_match_tolerance([_triangle()], matcher="relative_geometry")
    assert "zero is not a measurement of noise" in str(excinfo.value)


def test_replicates_that_disagree_about_how_many_features_there_are_are_refused():
    """A detection failure must not be laundered into a measurement error."""
    replicates = _replicates(n=3)
    replicates[1] = replicates[1][:2] + [replicates[1][2], _triangle(seed=9)[0]]
    with pytest.raises(InvalidParameterError) as excinfo:
        calibrate_match_tolerance(replicates, matcher="relative_geometry")
    assert "bury a detection failure" in str(excinfo.value)


def test_a_tolerance_cannot_be_built_from_a_single_realisation():
    with pytest.raises(InvalidParameterError):
        MatchTolerance(0.01, 1, 12, "edge0-1.shape_ratio", "measured once")


def test_a_tolerance_with_no_stated_basis_is_refused():
    """A number whose derivation is not written down cannot be told from a tuned one."""
    with pytest.raises(InvalidParameterError) as excinfo:
        MatchTolerance(0.01, 12, 12, "edge0-1.shape_ratio", "   ")
    assert "indistinguishable from one that was tuned" in str(excinfo.value)


def test_a_negative_tolerance_is_refused():
    with pytest.raises(InvalidParameterError):
        MatchTolerance(-0.01, 12, 12, "edge0-1.shape_ratio", "measured")


def test_match_refuses_a_tolerance_that_is_merely_a_number():
    """The whole argument of the module is that the two cannot be told apart afterwards."""
    with pytest.raises(InvalidParameterError) as excinfo:
        match(_triangle(), _triangle(rotation_deg=37.0),
              matcher="relative_geometry", tolerance=0.05)
    assert "cannot be distinguished from one that was tuned" in str(excinfo.value)


# ==================================================== 4. invariance is a test, not a threshold


def test_the_invariant_matcher_survives_every_transform(synthetic_audit):
    report = synthetic_audit["relative_geometry"]
    assert report.measured == TRANSFORMS
    assert report.honest
    assert not report.overclaimed and not report.understated


def test_the_position_memorising_matcher_survives_none_of_them(synthetic_audit):
    """What the roadmap asked to see, and it fails by the general rule, not a special case."""
    report = synthetic_audit["absolute_position"]
    assert report.measured == ()
    assert report.honest
    for transform in TRANSFORMS:
        assert report.tests[transform].p_value < report.tests[transform].alpha


def test_every_registered_matcher_declared_exactly_what_it_survived(synthetic_audit):
    for name, report in synthetic_audit.items():
        assert report.honest, (name, report.overclaimed, report.understated,
                               report.vacuous)


def test_alpha_is_divided_across_the_tests_that_were_actually_run(synthetic_audit):
    """R18 at its smallest scale: four chances to call a matcher broken is a family."""
    report = synthetic_audit["relative_geometry"]
    # two matchers, and within each three transforms plus the scale recovery
    assert report.tests["rotation"].alpha == pytest.approx(DEFAULT_ALPHA / 2.0 / 4.0)
    assert report.scale_recovery.alpha == pytest.approx(DEFAULT_ALPHA / 2.0 / 4.0)


def test_a_matcher_that_overclaims_is_caught(matchers_registry):
    """The audit is not a test of the two matchers that ship; it is a rule."""
    matchers_registry.add(
        "boastful", lambda features, context: build_signature(
            "absolute_position", features, context=context),
        description="Keys on position, claims the world.",
        capabilities={"invariant_to": TRANSFORMS, "min_nodes": 2})
    report = measure_invariance("boastful", _reference(), _presentations(), _replicates())
    assert report.overclaimed == TRANSFORMS
    assert not report.honest


def test_a_matcher_that_understates_is_caught_too(matchers_registry):
    """Conservatism is not honesty - a caller reaches for a heavier matcher it did not need."""
    matchers_registry.add(
        "modest", lambda features, context: relative_geometry(features, context),
        description="Invariant to everything, declares nothing.",
        capabilities={"invariant_to": (), "min_nodes": 3})
    report = measure_invariance("modest", _reference(), _presentations(), _replicates())
    assert report.understated == TRANSFORMS
    assert not report.honest


def test_a_test_that_could_not_have_failed_does_not_confer_invariance():
    """TG2.4's vacuous rule, carried into a rank test.

    Two replicates give one null value, and one presentation against one null value cannot
    return a p-value below 0.5 however wrong it is. Reporting invariance from that would be
    reporting the shape of the arithmetic, not a property of the matcher.
    """
    presentations = [Presentation("rotation/37", ("rotation",),
                                  _triangle(rotation_deg=37.0, jitter=0.4, seed=1))]
    report = measure_invariance("relative_geometry", _reference(), presentations,
                                _replicates(n=2))
    assert report.tests["rotation"].vacuous
    assert not report.tests["rotation"].invariant
    assert report.measured == ()
    assert not report.honest


def test_the_power_floor_is_the_smallest_p_value_the_comparison_could_return():
    test = InvarianceTest("rotation", ("a",), (0.1,), 1, 0.5, 0.05,
                          1.0 / math.comb(2, 1))
    assert test.power_floor == 0.5
    assert test.vacuous
    assert not test.invariant


def test_a_combined_presentation_confers_nothing_on_its_own():
    """Two errors that cancel is not a property, so it may not be recorded as one."""
    presentations = [p for p in _presentations() if not p.is_single]
    assert presentations and all(len(p.transforms) == 3 for p in presentations)
    with pytest.raises(InvalidParameterError) as excinfo:
        measure_invariance("relative_geometry", _reference(), presentations, _replicates())
    assert "cannot say which transform is wrong" in str(excinfo.value)


def test_a_presentation_naming_an_unknown_transform_is_refused():
    with pytest.raises(InvalidParameterError):
        Presentation("shear", ("shear",), _triangle())


# ================================================================ 5. how much bigger was it?


def test_the_scale_ratio_is_recovered_from_the_separations():
    """The scale-aware half: invariant to size, and it says what the size was."""
    tolerance = calibrate_match_tolerance(_replicates(), matcher="relative_geometry")
    found = match(_triangle(), _triangle(scale=2.5), matcher="relative_geometry",
                  tolerance=tolerance)
    assert found
    assert found.scale_ratio
    assert found.scale_ratio.value == pytest.approx(2.5, rel=1e-9)
    assert found.scale_ratio.units == "cells"


def test_the_recovered_ratio_did_not_decide_the_match():
    """R19's structure, not its documentation.

    The separation scale lives in `carried`, nothing in `AttributedGraph.matches` can read
    it, and rescaling one side changes the ratio without changing the decision.
    """
    tolerance = calibrate_match_tolerance(_replicates(), matcher="relative_geometry")
    ratios = []
    for factor in (0.5, 1.0, 2.0, 4.0):
        found = match(_triangle(), _triangle(scale=factor),
                      matcher="relative_geometry", tolerance=tolerance)
        assert found.report.matched
        ratios.append(found.scale_ratio.value)
    assert ratios == pytest.approx([0.5, 1.0, 2.0, 4.0], rel=1e-9)


def test_recovering_a_ratio_before_a_decision_is_structurally_impossible():
    """It takes the `MatchReport`, so it cannot run without one having been made."""
    left = build_signature("relative_geometry", _triangle())
    right = build_signature("relative_geometry", _triangle(sides=(1.0, 1.6, 0.7)))
    report = left.matches(right, tolerance=1e-9)
    assert not report.matched
    with pytest.raises(ScaleRatioUnavailableError) as excinfo:
        recover_scale_ratio(left, right, report)
    assert "no pairing of their nodes" in str(excinfo.value)


def test_a_ratio_across_a_unit_boundary_is_refused_by_name():
    """The match crossed that boundary precisely because it had divided the units out."""
    tolerance = calibrate_match_tolerance(_replicates(), matcher="relative_geometry")
    metres = _triangle(units="m", spacing=30.0, domain="ocean", dataset="sst_l4",
                       variable="temperature", magnitude_units="K")
    left = build_signature("relative_geometry", _triangle())
    right = build_signature("relative_geometry", metres)
    report = left.matches(right, tolerance=tolerance.value)
    assert report.matched
    ratio = recover_scale_ratio(left, right, report)
    assert not ratio
    assert ratio.value is None
    assert "'cells'" in ratio.unavailable and "'m'" in ratio.unavailable


def test_a_scale_ratio_must_be_a_value_or_a_reason_and_never_both():
    with pytest.raises(InvalidParameterError):
        ScaleRatio(1.5, "cells", "a ratio", unavailable="also unavailable")
    with pytest.raises(InvalidParameterError):
        ScaleRatio(None, "cells", "a ratio")


def test_a_scale_ratio_of_zero_or_infinity_is_not_a_ratio():
    for value in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(InvalidParameterError):
            ScaleRatio(value, "cells", "a ratio")


def test_the_recovery_is_judged_against_its_own_noise_floor_not_the_shapes(synthetic_audit):
    """The defect the first version of the gate had.

    The shape's floor and the size's floor are the noise of two different numbers, and
    checking one against the other is a category error that failed a correct matcher.
    """
    report = synthetic_audit["relative_geometry"]
    shape_null = null_deviations(_replicates(), matcher="relative_geometry")
    size_null = scale_recovery_null(_replicates(), matcher="relative_geometry")
    assert max(shape_null) != max(size_null)
    assert report.scale_recovery is not None
    assert report.scale_recovery.n_null == len(size_null)
    assert report.scale_recovery.accurate


def test_the_recovery_error_is_symmetric_in_the_direction_of_the_mistake():
    """Reporting 1.5 where the answer is 3 is the same mistake as 3 where it is 1.5.

    Measured against an exact reference, so the recovered ratio is exactly 2 and the two
    errors are the same mistake in opposite directions rather than two different ones.
    """
    presentations = [
        Presentation("double", ("rescaling",), _triangle(scale=2.0),
                     expected_scale_ratio=4.0),
        Presentation("half", ("rescaling",), _triangle(scale=2.0),
                     expected_scale_ratio=1.0),
    ]
    report = measure_invariance("relative_geometry", _triangle(), presentations,
                                _replicates())
    assert report.scale_recovery.errors[0] == pytest.approx(
        report.scale_recovery.errors[1], rel=1e-9)


def test_a_matcher_that_recovers_the_wrong_size_fails_the_recovery_test():
    presentations = [
        Presentation("rescaling/%g" % factor, ("rescaling",),
                     _triangle(scale=factor, jitter=0.4, seed=500 + index),
                     expected_scale_ratio=factor * 1.5)
        for index, factor in enumerate((0.5, 0.75, 1.5, 2.0, 3.0))]
    report = measure_invariance("relative_geometry", _reference(), presentations,
                                _replicates())
    assert not report.scale_recovery.accurate
    assert report.scale_recovery.p_value < report.scale_recovery.alpha


def test_a_presentation_with_no_stated_size_asks_for_no_recovery():
    presentations = [p for p in _presentations()
                     if p.is_single and p.expected_scale_ratio is None]
    report = measure_invariance("relative_geometry", _reference(), presentations,
                                _replicates())
    assert report.scale_recovery is None


def test_a_presentation_cannot_claim_to_have_been_resized_by_nothing():
    for bad in (0.0, -2.0, math.inf):
        with pytest.raises(InvalidParameterError):
            Presentation("rescaling", ("rescaling",), _triangle(),
                         expected_scale_ratio=bad)


# ================================================================== 6. matching is a registry


def test_a_matcher_registered_from_this_module_is_audited_without_editing_src(
        matchers_registry):
    """The registry is the extension point, and the audit reaches whatever is in it."""
    def centroid_distances(features, context):
        return relative_geometry(features, context)

    matchers_registry.add(
        "from_the_test_module", centroid_distances,
        description="Registered here, audited by the same rule as the rest.",
        capabilities={"invariant_to": TRANSFORMS, "min_nodes": 3})
    reports = audit_declared_invariance(_reference(), _presentations(), _replicates())
    assert "from_the_test_module" in reports
    assert reports["from_the_test_module"].honest


def test_an_unknown_matcher_is_refused_with_the_names_that_exist():
    with pytest.raises(UnknownNameError) as excinfo:
        build_signature("relatve_geometry", _triangle())
    assert "relative_geometry" in str(excinfo.value)


def test_the_declared_invariance_is_read_from_the_registry():
    assert declared_invariance("relative_geometry") == TRANSFORMS
    assert declared_invariance("absolute_position") == ()


def test_matcher_for_returns_the_registered_callable():
    assert matcher_for("relative_geometry") is relative_geometry


def test_the_scale_normalised_comparison_is_deliberately_not_registered():
    """A registry entry carries a declaration, and this one has none it can demonstrate.

    Measured directly it is not rescaling-invariant; tested the way `measure_invariance`
    tests, over seven rescalings, the evidence comes to p = 0.016 and does not survive the
    family correction. True and unproven at once is not a state a declaration can hold, so
    the function stays public and the registry stays honest.
    """
    assert "scale_normalised" not in MATCHERS.names()
    graph = scale_normalised(_triangle(), RelationContext())
    assert graph.relations == ("distance",)
    assert graph.edges == constellation(_triangle(), relations=["distance"]).edges


def test_a_matcher_too_large_for_the_configuration_is_skipped_not_failed(matchers_registry):
    """A matcher that was never run has not overclaimed anything."""
    matchers_registry.add(
        "needs_five", lambda features, context: relative_geometry(features, context),
        description="Wants five features.",
        capabilities={"invariant_to": TRANSFORMS, "min_nodes": 5})
    reports = audit_declared_invariance(_reference(), _presentations(), _replicates())
    assert "needs_five" not in reports
    assert "relative_geometry" in reports


def test_an_audit_with_nothing_runnable_is_refused_rather_than_reported_clean():
    with pytest.raises(InvalidParameterError):
        audit_declared_invariance(_reference()[:1], _presentations(), _replicates(),
                                  names=["relative_geometry"])


def test_each_matcher_gets_its_own_noise_floor(synthetic_audit):
    """Cells and dimensionless ratios are not the same quantity, so not the same floor."""
    floors = {name: report.tolerance.value for name, report in synthetic_audit.items()}
    assert floors["absolute_position"] != floors["relative_geometry"]


def test_the_report_describes_itself_completely(synthetic_audit):
    described = synthetic_audit["relative_geometry"].describe()
    assert set(described) >= {"matcher", "declared", "measured", "overclaimed",
                              "understated", "honest", "vacuous", "tolerance", "tests",
                              "matches", "scale_recovery"}
    assert described["tests"]["rotation"]["n_null"] == REPLICATES * (REPLICATES - 1) // 2


# ================================================================ 7. the gate, on the real thing


@pytest.fixture(scope="module")
def planted_results():
    """The benchmark, run once. Everything below reads the same run."""
    return get_benchmark("planted_configuration").run()


def test_the_4e_invariance_gate_is_no_longer_pending(planted_results):
    """TG3.4's acceptance: the last `NOT_YET_RUNNABLE` stage in the suite is now enforced."""
    gate = [c for c in planted_results if c.stage == "4E.invariance"]
    assert gate, "the planted configuration must still define the 4E.invariance gate"
    assert gate[0].outcome is Outcome.PASS, gate[0].detail


def test_the_gate_audited_the_position_memorising_control_and_it_survived_nothing(
        planted_results):
    gate = [c for c in planted_results if c.stage == "4E.invariance"][0]
    reports = gate.measured["reports"]
    assert reports["absolute_position"]["measured"] == []
    assert reports["absolute_position"]["honest"] is True


def test_the_gate_measured_full_invariance_on_the_real_extractor(planted_results):
    gate = [c for c in planted_results if c.stage == "4E.invariance"][0]
    report = gate.measured["reports"]["relative_geometry"]
    assert report["measured"] == list(TRANSFORMS)
    assert report["honest"] is True
    assert report["scale_recovery"]["accurate"] is True


def test_the_gate_recovered_the_planted_scale_factors(planted_results):
    """Invariant to size and able to say what the size was - the scale-aware reading."""
    gate = [c for c in planted_results if c.stage == "4E.invariance"][0]
    recovery = gate.measured["reports"]["relative_geometry"]["scale_recovery"]
    assert len(recovery["recovered"]) >= 3
    for expected, found in zip(recovery["expected"], recovery["recovered"]):
        assert abs(math.log(found / expected)) < 0.02


def test_the_gate_states_how_hard_it_looked(planted_results):
    """A pass from too few replicates is a pass earned by not measuring the noise."""
    gate = [c for c in planted_results if c.stage == "4E.invariance"][0]
    truth = get_benchmark("planted_configuration").truth()
    assert gate.measured["n_replicates"] >= truth["minimum_invariance_replicates"]
    assert gate.measured["scale_ratios_recovered"] >= truth[
        "minimum_scale_ratios_recovered"]
    assert gate.measured["n_presentations"] >= len(TRANSFORMS)


def test_the_scale_normalised_relation_drifts_on_the_real_extractor():
    """The measurement that ruled `distance` out of the registry, on the real pipeline.

    It is not noise. The extractor's spatial scale estimate runs about +1.4% high at a
    6-cell sigma and about -2.6% low at 18 cells, and a separation divided by that estimate
    carries the whole of the difference.
    """
    import numpy as np

    from src.core.extraction import ExtractionField, extract

    bench = get_benchmark("planted_configuration")
    measured = {}
    for factor in (1.0, 3.0):
        data = np.asarray(bench.make(scale_factor=factor).data.numpy(), dtype=np.float64)
        frame = ExtractionField(
            values=data, axes=AXES, domain="synthetic",
            dataset="planted_configuration", variable="amplitude", units=None,
            time=0.0, time_units="frames", representation="identity")
        features = list(extract(frame, n_surrogates=99, seed=1234))
        assert len(features) == 3
        expected = bench.truth(scale_factor=factor)["feature_sigma_cells"]
        measured[factor] = sum(f.spatial_scale.value for f in features) / 3.0 / expected
    assert measured[1.0] > 1.0
    assert measured[3.0] < 1.0
    assert measured[1.0] / measured[3.0] - 1.0 > 0.03
