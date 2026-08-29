"""Phase G3, TG3.3: the same configuration, described from two domains that share nothing.

The acceptance criterion is a measurement rather than a property: the planted equilateral
triangle is built twice - once as a synthetic amplitude field on a grid in **cells**, once as
a sea-surface temperature field on a grid in **metres**, with a different domain, dataset,
variable, units and spacing - and the two attributed graphs are **equal**. Around it sits the
negative control that makes the equality mean something: a relation registered deliberately
*without* the dimensionless division, reporting the raw separation, makes the same two graphs
disagree. So the division is measured to be doing the work, rather than asserted to be.

The rest of the file is the machinery that has to hold for the acceptance to be trustworthy:

*   every registered relation declares what it requires, and one asked for a quantity the
    features do not carry **refuses by name** rather than treating absence as agreement -
    which is TG2.3's rule about gates, carried into relations;
*   `direction` and `convergence` therefore refuse on the only extractor TG2.2 registers,
    which declares `reports_orientation: False`, and that refusal is asserted against the
    registry's own capability rather than against a comment;
*   `measurable_relations` and `relation_axis` keep a TG3.1 declaration from being priced
    over relations that will refuse; and
*   matching is exhaustive over node correspondences or **refused**, because a heuristic
    assignment that returns True is a claim.
"""

import math

import pytest

from src.core.constellation import (
    DEFAULT_TOLERANCE,
    MAX_MATCH_NODES,
    RELATIONS,
    AttributedGraph,
    GraphTooLargeError,
    Relation,
    RelationContext,
    RelationUnmeasurableError,
    RelationValue,
    constellation,
    constellations_from_set,
    measurable_relations,
    relation_axis,
    relation_for,
)
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.extraction import EXTRACTORS
from src.core.family import SearchAxis, SearchSpecification, SearchTerm
from src.core.feature import (
    FeatureLocation,
    FeatureSet,
    Orientation,
    Quantity,
    SemanticComparisonError,
    Significance,
    SpectralFeature,
)
from src.core.registry import restore, snapshot

#: The benchmark's own geometry, so the acceptance is priced off `planted_configuration`
#: rather than off numbers invented here. `build_planted_configuration` defaults to a
#: 40-cell side and a 6-cell feature sigma on a 256-cell grid.
TRIANGLE_SIDE_CELLS = 40.0
FEATURE_SIGMA_CELLS = 6.0
GRID_CELLS = 256


def _planted_triangle(*, domain, dataset, variable, magnitude_units, spacing,
                      length_units, representation="morlet", rotation_deg=0.0,
                      significance=0.01, temporal_scale=None, time_units=None,
                      times=(0.0, 0.0, 0.0)):
    """The `planted_configuration` triangle, expressed in one domain's own units.

    `spacing` is how many of `length_units` one cell of the benchmark is, so the same
    configuration can be handed to two domains whose numbers have nothing in common.
    """
    axes = (AxisSpec("row", "space", length_units, ordinal=0),
            AxisSpec("col", "space", length_units, ordinal=1))
    radius = (TRIANGLE_SIDE_CELLS * spacing) / math.sqrt(3.0)
    centre = (GRID_CELLS / 2.0) * spacing
    features = []
    for index in range(3):
        angle = math.radians(rotation_deg + 120.0 * index - 90.0)
        features.append(SpectralFeature(
            domain=domain, dataset=dataset, variable=variable,
            magnitude=Quantity(1.0, magnitude_units),
            location=FeatureLocation(
                {"row": centre + radius * math.sin(angle),
                 "col": centre + radius * math.cos(angle)}, axes),
            time=float(times[index]), representation=representation,
            time_units=time_units,
            spatial_scale=Quantity(FEATURE_SIGMA_CELLS * spacing, length_units),
            temporal_scale=temporal_scale,
            significance=Significance(significance, "surrogate_quantile")))
    return features


def _cells_domain(**kwargs):
    """The synthetic benchmark as it is: a dimensionless amplitude on a grid in cells."""
    return _planted_triangle(domain="synthetic", dataset="planted_configuration",
                             variable="amplitude", magnitude_units=None, spacing=1.0,
                             length_units="cells", **kwargs)


def _metres_domain(**kwargs):
    """A different domain entirely: kelvin on a 30-metre grid, sharing only the geometry."""
    return _planted_triangle(domain="ocean", dataset="sst_l4", variable="temperature",
                             magnitude_units="K", spacing=30.0, length_units="m",
                             **kwargs)


def _scalene():
    """Three features at mutually different separations - a configuration with no symmetry,
    so exactly one node correspondence can recover it."""
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    positions = ((0.0, 0.0), (0.0, 30.0), (40.0, 0.0))
    return [_feature(location=FeatureLocation({"row": r, "col": c}, axes),
                     spatial_scale=Quantity(2.0, "cells"))
            for r, c in positions]


def _feature(**overrides):
    """One feature, with every optional quantity absent unless a test asks for it."""
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    fields = dict(
        domain="synthetic", dataset="planted_configuration", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({"row": 10.0, "col": 10.0}, axes),
        time=0.0, representation="morlet")
    fields.update(overrides)
    return SpectralFeature(**fields)


def _pair(*, left_at=(10.0, 10.0), right_at=(10.0, 22.0), units="cells", **shared):
    """Two features on the same axes, at declared positions, sharing declared quantities."""
    axes = (AxisSpec("row", "space", units, ordinal=0),
            AxisSpec("col", "space", units, ordinal=1))
    out = []
    for coords in (left_at, right_at):
        out.append(_feature(
            location=FeatureLocation({"row": coords[0], "col": coords[1]}, axes),
            **shared))
    return out


@pytest.fixture
def relations_registry():
    """Register a relation for one test without leaking it into the next."""
    state = snapshot(RELATIONS)
    try:
        yield RELATIONS
    finally:
        restore(RELATIONS, state)


# =========================================================================== the acceptance


def test_the_same_configuration_in_two_domains_is_the_same_attributed_graph():
    """TG3.3's acceptance, measured.

    Nothing numeric is shared between the two sides: 40 cells against 1,200 metres, a
    dimensionless amplitude against kelvin, `synthetic/planted_configuration/amplitude`
    against `ocean/sst_l4/temperature`. The graphs are equal because every relation was
    divided by something the features carry themselves.
    """
    cells = _cells_domain()
    metres = _metres_domain()

    assert cells[0].location.coords != metres[0].location.coords
    assert cells[0].spatial_scale.units == "cells"
    assert metres[0].spatial_scale.units == "m"

    left = constellation(cells)
    right = constellation(metres)
    report = left.matches(right)

    assert report, report.reason
    assert report.correspondence is not None
    assert "distance" in report.relations_compared


def test_the_acceptance_compares_relations_that_are_actually_dimensionless():
    """The graphs above agree on a *number*, not merely on which edges exist."""
    cells = constellation(_cells_domain())
    metres = constellation(_metres_domain())
    # 40 cells of side over a 6-cell scale, and 1200 m over 180 m: the same 6.67.
    expected = TRIANGLE_SIDE_CELLS / FEATURE_SIGMA_CELLS
    for graph in (cells, metres):
        assert graph.value(0, 1, "distance").value == pytest.approx(expected, rel=1e-12)


def test_a_relation_that_skips_the_division_makes_the_two_domains_disagree(
        relations_registry):
    """The negative control: without the division there is no cross-domain match.

    Registered from the test module, which is also the check that the registry is the
    extension point it claims to be. `raw_separation` is a perfectly reasonable-looking
    relation and it reports 40 in one domain and 1,200 in the other.
    """
    def _raw(left, right, context):
        separation = left.separation_to(right, periods=context.periods)
        return RelationValue("raw_separation", separation.value, None,
                             "the separation in %s, undivided" % separation.units)

    relations_registry.add(
        "raw_separation",
        Relation("raw_separation", "continuous", (), False, _raw,
                 "The separation in the originating units - deliberately not transferable."))

    cells = constellation(_cells_domain(), relations=("raw_separation",))
    metres = constellation(_metres_domain(), relations=("raw_separation",))

    assert cells.value(0, 1, "raw_separation").value == pytest.approx(
        TRIANGLE_SIDE_CELLS, rel=1e-12)
    assert metres.value(0, 1, "raw_separation").value == pytest.approx(
        TRIANGLE_SIDE_CELLS * 30.0, rel=1e-12)
    report = cells.matches(metres)
    assert not report
    assert "no correspondence" in report.reason


def test_the_match_report_states_the_two_records_it_refused_to_compare():
    """R19: the graphs match while remaining a temperature field and an amplitude field."""
    report = constellation(_cells_domain()).matches(constellation(_metres_domain()))
    assert report
    assert {c["variable"] for c in report.left_carried} == {"amplitude"}
    assert {c["variable"] for c in report.right_carried} == {"temperature"}
    assert {c["units"] for c in report.right_carried} == {"K"}
    described = report.describe()
    assert described["left_carried"][0]["domain"] == "synthetic"
    assert described["right_carried"][0]["domain"] == "ocean"


def test_no_carried_field_reaches_the_comparable_view():
    """Built rather than filtered: a carried value cannot be in `attributes` by accident."""
    graph = constellation(_metres_domain())
    carried_values = {"ocean", "sst_l4", "temperature", "K", "morlet", "m"}
    for node in graph.attributes:
        assert not (set(str(v) for v in node.values()) & carried_values)
    assert "representation" not in graph.attributes[0]
    assert graph.carried[0]["representation"] == "morlet"


def test_two_domains_match_even_when_their_representations_differ_and_it_is_reported():
    """R8 is stated, not silently enforced: a representation cannot be compared numerically,
    but a reader must be able to see that two graphs came from different ones."""
    report = constellation(_cells_domain()).matches(
        constellation(_metres_domain(representation="dtcwt")))
    assert report
    assert report.representations == (("morlet",), ("dtcwt",))


def test_a_rotated_triangle_still_matches_on_the_relations_this_slice_measures():
    """Not TG3.4's acceptance, and deliberately not claimed as one.

    Distance and relative scale are rotation-invariant on their own, so a rotated
    configuration matches here. `direction` and `convergence` are the relations that would
    fail, they need an orientation, and no registered extractor reports one - which is why
    `4E.invariance` remains TG3.4's to move.
    """
    report = constellation(_cells_domain()).matches(
        constellation(_cells_domain(rotation_deg=37.0)))
    assert report
    assert "direction" not in report.relations_compared


# ==================================================================== the relation registry


def test_all_eight_named_relations_are_registered():
    assert set(RELATIONS.names()) == {
        "distance", "relative_scale", "temporal_lag", "direction", "convergence",
        "containment", "succession", "co_occurrence"}


def test_every_relation_declares_a_kind_a_requirement_list_and_a_symmetry():
    for entry in RELATIONS.entries():
        rule = entry.value
        assert rule.kind in Relation.KINDS
        assert isinstance(rule.symmetric, bool)
        assert rule.name == entry.name
        assert entry.capabilities["requires"] == list(rule.requires)
        assert entry.capabilities["dimensionless_by"]


def test_a_relation_cannot_declare_a_requirement_that_is_not_a_feature_field():
    with pytest.raises(UnknownNameError):
        Relation("bogus", "continuous", ("colour",), True, lambda a, b, c: None)


def test_an_unregistered_relation_is_refused_by_name_with_the_valid_options():
    with pytest.raises(UnknownNameError) as excinfo:
        relation_for("proximity")
    assert "distance" in str(excinfo.value)


def test_only_succession_needs_no_characteristic_scale():
    """The finding: sorting the eight by what each needs does not give eight of a kind."""
    needs_nothing = [e.name for e in RELATIONS.entries() if not e.value.requires]
    assert needs_nothing == ["succession"]
    assert relation_for("co_occurrence").requires == ("temporal_scale",)


# ======================================================================= the eight measured


def test_distance_is_the_separation_in_units_of_the_features_own_scale():
    left, right = _pair(spatial_scale=Quantity(3.0, "cells"))
    value = relation_for("distance").measure(left, right, RelationContext())
    assert value.value == pytest.approx(12.0 / 3.0)
    assert "geometric mean" in value.basis


def test_distance_divides_by_the_geometric_mean_so_it_does_not_depend_on_the_larger():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    left = _feature(location=FeatureLocation({"row": 0.0, "col": 0.0}, axes),
                    spatial_scale=Quantity(1.0, "cells"))
    right = _feature(location=FeatureLocation({"row": 0.0, "col": 10.0}, axes),
                     spatial_scale=Quantity(100.0, "cells"))
    value = relation_for("distance").measure(left, right, RelationContext())
    assert value.value == pytest.approx(10.0 / 10.0)      # sqrt(1 * 100), not (1 + 100)/2


def test_relative_scale_is_the_ratio_and_it_is_not_symmetric():
    left, right = _pair(spatial_scale=Quantity(4.0, "cells"))
    left = _feature(location=left.location, spatial_scale=Quantity(8.0, "cells"))
    assert relation_for("relative_scale").measure(
        left, right, RelationContext()).value == pytest.approx(2.0)
    assert relation_for("relative_scale").measure(
        right, left, RelationContext()).value == pytest.approx(0.5)
    assert relation_for("relative_scale").symmetric is False


def test_temporal_lag_is_signed_and_measured_in_characteristic_times():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    def at(time):
        return _feature(location=FeatureLocation({"row": 0.0}, axes), time=time,
                        time_units="hours", temporal_scale=Quantity(2.0, "hours"))
    value = relation_for("temporal_lag").measure(at(0.0), at(6.0), RelationContext())
    assert value.value == pytest.approx(3.0)
    back = relation_for("temporal_lag").measure(at(6.0), at(0.0), RelationContext())
    assert back.value == pytest.approx(-3.0)


def test_temporal_lag_refuses_a_clock_and_a_scale_in_different_units():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    def at(time):
        return _feature(location=FeatureLocation({"row": 0.0}, axes), time=time,
                        time_units="hours", temporal_scale=Quantity(2.0, "frames"))
    with pytest.raises(SemanticComparisonError):
        relation_for("temporal_lag").measure(at(0.0), at(6.0), RelationContext())


def test_direction_is_the_bearing_measured_from_the_features_own_orientation():
    """The grid's north cancels: rotating both the configuration and the orientations
    leaves the relation unchanged, which is the whole reason a bare bearing is not one."""
    def rotated(by):
        axes = (AxisSpec("row", "space", "cells", ordinal=0),
                AxisSpec("col", "space", "cells", ordinal=1))
        offset = (10.0 * math.cos(math.radians(by)), 10.0 * math.sin(math.radians(by)))
        left = _feature(location=FeatureLocation({"row": 0.0, "col": 0.0}, axes),
                        orientation=Orientation(by, "direction_360"))
        right = _feature(location=FeatureLocation(
            {"row": offset[0], "col": offset[1]}, axes),
            orientation=Orientation(by, "direction_360"))
        return relation_for("direction").measure(left, right, RelationContext())

    assert rotated(0.0).value == pytest.approx(rotated(53.0).value, abs=1e-9)
    assert rotated(0.0).value == pytest.approx(0.0, abs=1e-9)


def test_convergence_is_refused_on_an_undirected_axis_because_an_axis_does_not_point():
    left, right = _pair(orientation=Orientation(45.0, "axis_180"))
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("convergence").measure(left, right, RelationContext())
    assert "does not point" in str(excinfo.value)


def test_convergence_is_positive_when_both_point_inward_and_negative_when_both_point_out():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    def facing(left_deg, right_deg):
        left = _feature(location=FeatureLocation({"row": 0.0, "col": 0.0}, axes),
                        orientation=Orientation(left_deg, "direction_360"))
        right = _feature(location=FeatureLocation({"row": 0.0, "col": 10.0}, axes),
                         orientation=Orientation(right_deg, "direction_360"))
        return relation_for("convergence").measure(left, right, RelationContext())

    inward = facing(90.0, 270.0)          # left points +col, right points -col
    outward = facing(270.0, 90.0)
    assert inward.value == pytest.approx(2.0)
    assert inward.holds is True
    assert outward.value == pytest.approx(-2.0)
    assert outward.holds is False


def test_containment_holds_only_when_the_smaller_lies_wholly_inside_the_larger():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    big = _feature(location=FeatureLocation({"row": 0.0, "col": 0.0}, axes),
                   extent=Quantity(10.0, "cells"))
    inside = _feature(location=FeatureLocation({"row": 0.0, "col": 4.0}, axes),
                      extent=Quantity(2.0, "cells"))
    outside = _feature(location=FeatureLocation({"row": 0.0, "col": 9.0}, axes),
                       extent=Quantity(2.0, "cells"))
    assert relation_for("containment").measure(big, inside, RelationContext()).holds
    assert not relation_for("containment").measure(big, outside, RelationContext()).holds
    assert not relation_for("containment").measure(inside, big, RelationContext()).holds


def test_succession_is_an_ordering_and_needs_no_scale_to_survive_a_change_of_units():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    def at(time, units):
        return _feature(location=FeatureLocation({"row": 0.0}, axes), time=time,
                        time_units=units)
    in_hours = relation_for("succession").measure(
        at(0.0, "hours"), at(6.0, "hours"), RelationContext())
    in_frames = relation_for("succession").measure(
        at(0.0, "frames"), at(2.0, "frames"), RelationContext())
    assert in_hours.holds is in_frames.holds is True
    assert in_hours.value is None
    assert relation_for("succession").measure(
        at(6.0, "hours"), at(0.0, "hours"), RelationContext()).holds is False


def test_co_occurrence_is_a_tolerance_and_therefore_needs_one():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    def at(time):
        return _feature(location=FeatureLocation({"row": 0.0}, axes), time=time,
                        time_units="hours", temporal_scale=Quantity(4.0, "hours"))
    together = relation_for("co_occurrence").measure(at(0.0), at(3.0), RelationContext())
    apart = relation_for("co_occurrence").measure(at(0.0), at(9.0), RelationContext())
    assert together.holds is True
    assert apart.holds is False
    assert apart.value == pytest.approx(9.0 / 4.0)


# ================================================================================ refusals


def test_a_relation_refuses_by_name_when_a_feature_does_not_carry_what_it_needs():
    left, right = _pair()                      # no spatial scale on either
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("distance").measure(left, right, RelationContext())
    assert excinfo.value.relation == "distance"
    assert excinfo.value.requirement == "spatial_scale"
    assert "no evidence against" in str(excinfo.value)


def test_the_refusal_names_which_side_lacked_the_quantity():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    left = _feature(location=FeatureLocation({"row": 0.0}, axes),
                    spatial_scale=Quantity(2.0, "cells"))
    right = _feature(location=FeatureLocation({"row": 5.0}, axes))
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("distance").measure(left, right, RelationContext())
    assert "right feature" in str(excinfo.value)


def test_direction_and_convergence_refuse_on_the_only_extractor_tg22_registers():
    """The ruling for this slice, tied to the registry rather than to a comment."""
    assert EXTRACTORS.entry("local_maximum").capabilities["reports_orientation"] is False
    without_orientation = _cells_domain()
    assert all(f.orientation is None for f in without_orientation)
    for name in ("direction", "convergence"):
        with pytest.raises(RelationUnmeasurableError):
            relation_for(name).measure(without_orientation[0], without_orientation[1],
                                       RelationContext())


def test_distance_refuses_a_location_in_cells_beside_a_scale_in_metres():
    left, right = _pair(units="cells", spatial_scale=Quantity(3.0, "m"))
    with pytest.raises(SemanticComparisonError) as excinfo:
        relation_for("distance").measure(left, right, RelationContext())
    assert "not a number of scale lengths" in str(excinfo.value)


def test_a_direction_from_a_point_to_itself_is_undefined_rather_than_zero():
    left, right = _pair(left_at=(4.0, 4.0), right_at=(4.0, 4.0),
                        orientation=Orientation(0.0, "direction_360"))
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("direction").measure(left, right, RelationContext())
    assert "undefined" in str(excinfo.value)


def test_a_direction_across_a_periodic_axis_is_refused_rather_than_guessed():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("lon", "space", "cells", periodic=True, ordinal=1))
    def at(col):
        return _feature(location=FeatureLocation({"row": 0.0, "lon": col}, axes),
                        orientation=Orientation(0.0, "direction_360"))
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("direction").measure(at(1.0), at(300.0),
                                          RelationContext({"lon": 360.0}))
    assert "opposite ways" in str(excinfo.value)


def test_a_bearing_needs_exactly_two_spatial_axes():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),)
    def at(row):
        return _feature(location=FeatureLocation({"row": row}, axes),
                        orientation=Orientation(0.0, "direction_360"))
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_for("direction").measure(at(0.0), at(5.0), RelationContext())
    assert "exactly two" in str(excinfo.value)


def test_a_relation_value_measuring_neither_a_number_nor_a_predicate_is_refused():
    with pytest.raises(InvalidParameterError):
        RelationValue("distance", None, None, "nothing at all")


def test_a_relation_value_carrying_a_nan_is_refused_because_it_matches_nothing():
    with pytest.raises(InvalidParameterError):
        RelationValue("distance", float("nan"), None, "an undefined ratio")


def test_a_relation_value_must_say_what_it_divided_by():
    with pytest.raises(InvalidParameterError):
        RelationValue("distance", 1.0, None, "   ")


def test_a_separation_across_a_periodic_axis_still_needs_its_declared_length():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("lon", "space", "cells", periodic=True, ordinal=1))
    def at(col):
        return _feature(location=FeatureLocation({"row": 0.0, "lon": col}, axes),
                        spatial_scale=Quantity(2.0, "cells"))
    with pytest.raises(InvalidParameterError):
        relation_for("distance").measure(at(1.0), at(300.0), RelationContext())
    wrapped = relation_for("distance").measure(at(1.0), at(300.0),
                                               RelationContext({"lon": 360.0}))
    assert wrapped.value == pytest.approx(61.0 / 2.0)


# ============================================================ what a set can actually support


def test_measurable_relations_reports_only_what_every_feature_carries():
    assert measurable_relations(_cells_domain()) == (
        "distance", "relative_scale", "succession")


def test_a_feature_carrying_more_supports_more():
    features = _cells_domain(temporal_scale=Quantity(1.0, "hours"), time_units="hours",
                             times=(0.0, 1.0, 2.0))
    assert set(measurable_relations(features)) == {
        "distance", "relative_scale", "succession", "temporal_lag", "co_occurrence"}


def test_measurable_relations_is_vacuously_true_of_no_features_so_it_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        measurable_relations([])
    assert "vacuously" in str(excinfo.value)


def test_the_relation_axis_declares_exactly_the_measurable_relations():
    axis = relation_axis(_cells_domain())
    assert axis.name == "relation"
    assert axis.values == ("distance", "relative_scale", "succession")


def test_a_family_priced_over_all_eight_relations_declares_tests_that_never_run():
    """Why `relation_axis` exists rather than `RELATIONS.names()`.

    Eight relations against three measurable ones is not a conservative rounding; it is a
    receipt naming five tests that could not have happened.
    """
    features = _cells_domain()

    def _declare(axis):
        return SearchSpecification(
            terms=(SearchTerm("unordered_pairs", (axis,)),),
            n_surrogates=999, alpha=0.05, correction="benjamini_yekutieli",
            label_format="{0}~{1}", study_id="tg33_relations")

    honest = _declare(relation_axis(features))
    inflated = _declare(SearchAxis("relation", tuple(RELATIONS.names())))
    assert len(RELATIONS) == 8
    assert honest.family_size == 3            # C(3, 2), the measurable relations
    assert inflated.family_size == 28         # C(8, 2), five of them unmeasurable
    assert inflated.family_size > honest.family_size


def test_asking_for_relations_none_of_which_are_measurable_is_refused():
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        relation_axis(_cells_domain(), ("direction", "convergence"))
    assert "no axis to declare" in str(excinfo.value)


# ================================================================== building the graph


def test_a_constellation_of_one_feature_has_no_relations_and_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        constellation(_cells_domain()[:1])
    assert "at least two" in str(excinfo.value)


def test_the_same_feature_twice_is_refused_rather_than_given_a_zero_edge():
    feature = _cells_domain()[0]
    with pytest.raises(InvalidParameterError) as excinfo:
        constellation([feature, feature])
    assert "distinct features" in str(excinfo.value)


def test_a_constellation_spanning_two_datasets_is_refused():
    mixed = [_cells_domain()[0], _metres_domain()[1]]
    with pytest.raises(SemanticComparisonError) as excinfo:
        constellation(mixed)
    assert "what crosses a domain boundary is the graph" in str(excinfo.value)


def test_every_ordered_pair_gets_an_edge():
    graph = constellation(_cells_domain())
    assert graph.n_nodes == 3
    assert set(graph.edges) == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}


def test_a_symmetric_relation_is_measured_once_and_an_asymmetric_one_twice():
    graph = constellation(_cells_domain())
    assert graph.value(0, 1, "distance").value == pytest.approx(
        graph.value(1, 0, "distance").value)
    scale_up = graph.value(0, 1, "relative_scale").value
    scale_down = graph.value(1, 0, "relative_scale").value
    assert scale_up * scale_down == pytest.approx(1.0)


def test_strict_construction_propagates_a_refusal_rather_than_dropping_the_edge():
    with pytest.raises(RelationUnmeasurableError):
        constellation(_cells_domain(), relations=("direction",))


def test_non_strict_construction_records_the_refusal_where_a_receipt_can_read_it():
    graph = constellation(_cells_domain(), relations=("distance", "direction"),
                          strict=False)
    assert graph.value(0, 1, "distance").value > 0.0
    assert "direction" in graph.refusals[(0, 1)]
    assert "orientation" in graph.describe()["refusals"]["0->1"]["direction"]


def test_an_unknown_relation_is_refused_before_anything_is_measured():
    with pytest.raises(UnknownNameError):
        constellation(_cells_domain(), relations=("proximity",))


def test_a_graph_describes_itself_including_what_it_refused_to_measure():
    described = constellation(_cells_domain(), relations=("distance", "convergence"),
                              strict=False).describe()
    assert described["n_nodes"] == 3
    assert described["relations"] == ["distance", "convergence"]
    assert described["edges"]["0->1"]["distance"]["value"] > 0.0
    assert set(described["refusals"]) == {"0->1", "0->2", "1->0", "1->2", "2->0", "2->1"}


def test_asking_for_an_edge_or_relation_that_is_not_there_names_the_alternatives():
    graph = constellation(_cells_domain())
    with pytest.raises(UnknownNameError):
        graph.edge(0, 7)
    with pytest.raises(UnknownNameError):
        graph.value(0, 1, "convergence")


# ================================================================================ matching


def test_a_relabelled_configuration_matches_under_the_permutation_that_recovers_it():
    """On a scalene configuration there is exactly one correspondence, and it is found."""
    features = _scalene()
    graph = constellation(features, relations=("distance",))
    shuffled = constellation([features[2], features[0], features[1]],
                             relations=("distance",))
    report = graph.matches(shuffled)
    assert report
    assert report.correspondence == (1, 2, 0)


def test_an_equilateral_configuration_matches_under_its_own_symmetry():
    """The planted triangle is symmetric, so the identity is a correspondence too - which
    is why the permutation test above needs a configuration that is not."""
    features = _cells_domain()
    graph = constellation(features)
    report = graph.matches(constellation([features[2], features[0], features[1]]))
    assert report
    assert report.correspondence == (0, 1, 2)


def test_graphs_of_different_sizes_do_not_match_and_say_so_without_permuting():
    three = constellation(_cells_domain())
    two = constellation(_cells_domain()[:2])
    report = three.matches(two)
    assert not report
    assert "3 and 2 nodes" in report.reason


def test_a_configuration_that_is_not_the_same_shape_does_not_match():
    features = _cells_domain()
    axes = features[0].location.axes
    moved = SpectralFeature(
        domain="synthetic", dataset="planted_configuration", variable="amplitude",
        magnitude=Quantity(1.0, None),
        location=FeatureLocation({"row": 200.0, "col": 40.0}, axes),
        time=0.0, representation="morlet",
        spatial_scale=Quantity(FEATURE_SIGMA_CELLS, "cells"),
        significance=Significance(0.01, "surrogate_quantile"))
    stretched = constellation([features[0], features[1], moved])
    assert not constellation(features).matches(stretched)


def test_the_tolerance_is_relative_and_is_the_reason_a_near_match_is_a_match():
    features = _cells_domain()
    nudged = _cells_domain()
    nudged[0] = SpectralFeature(
        domain="synthetic", dataset="planted_configuration", variable="amplitude",
        magnitude=Quantity(1.0, None), location=features[0].location,
        time=0.0, representation="morlet",
        spatial_scale=Quantity(FEATURE_SIGMA_CELLS * (1.0 + 1e-7), "cells"),
        significance=Significance(0.01, "surrogate_quantile"))
    left, right = constellation(features), constellation(nudged)
    assert not left.matches(right, tolerance=DEFAULT_TOLERANCE)
    assert left.matches(right, tolerance=1e-5)


def test_a_negative_tolerance_is_refused():
    graph = constellation(_cells_domain())
    with pytest.raises(InvalidParameterError):
        graph.matches(graph, tolerance=-1.0)


def test_matching_on_a_relation_neither_graph_measured_is_refused_by_name():
    graph = constellation(_cells_domain(), relations=("distance",))
    with pytest.raises(UnknownNameError):
        graph.matches(graph, relations=("containment",))


def test_a_graph_with_an_unmeasured_relation_refuses_the_comparison_it_cannot_answer():
    """A defect this file found: without the refusal, such a graph did not match *itself*,
    and the stated reason blamed the configuration for a hole in the record."""
    graph = constellation(_cells_domain(), relations=("distance", "direction"),
                          strict=False)
    with pytest.raises(RelationUnmeasurableError) as excinfo:
        graph.matches(graph)
    assert "measurement nobody made" in str(excinfo.value)


def test_narrowing_to_what_was_measured_is_what_the_refusal_asks_for():
    graph = constellation(_cells_domain(), relations=("distance", "direction"),
                          strict=False)
    assert graph.matches(graph, relations=("distance",))


def test_matching_refuses_to_approximate_above_the_declared_node_limit():
    """`k!` correspondences, or a refusal - never a heuristic that returns True."""
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    many = [_feature(location=FeatureLocation({"row": 0.0, "col": float(i)}, axes),
                     spatial_scale=Quantity(2.0, "cells"))
            for i in range(MAX_MATCH_NODES + 1)]
    graph = constellation(many, relations=("distance",))
    assert graph.n_nodes == MAX_MATCH_NODES + 1
    with pytest.raises(GraphTooLargeError) as excinfo:
        graph.matches(graph)
    assert "approximate" in str(excinfo.value)


def test_a_graph_at_the_node_limit_still_matches_itself():
    axes = (AxisSpec("row", "space", "cells", ordinal=0),
            AxisSpec("col", "space", "cells", ordinal=1))
    many = [_feature(location=FeatureLocation({"row": 0.0, "col": float(i * i)}, axes),
                     spatial_scale=Quantity(2.0, "cells"))
            for i in range(MAX_MATCH_NODES)]
    graph = constellation(many, relations=("distance",))
    assert graph.matches(graph)


def test_a_graph_needs_at_least_two_nodes():
    with pytest.raises(InvalidParameterError):
        AttributedGraph(attributes=({},), carried=({},), edges={}, relations=())


# ================================================================ enumeration over a set


def test_every_unordered_subset_of_a_set_is_enumerated_in_the_shape_tg31_prices():
    features = _cells_domain() + _cells_domain(rotation_deg=90.0)
    graphs = list(constellations_from_set(FeatureSet(features), 3,
                                          relations=("distance",)))
    assert len(graphs) == 20                    # C(6, 3)
    assert all(graph.n_nodes == 3 for graph in graphs)


def test_a_subset_larger_than_the_set_is_refused_rather_than_silently_truncated():
    with pytest.raises(InvalidParameterError):
        list(constellations_from_set(FeatureSet(_cells_domain()), 4))


def test_a_subset_of_one_is_refused_because_a_configuration_is_about_relations():
    with pytest.raises(InvalidParameterError):
        list(constellations_from_set(FeatureSet(_cells_domain()), 1))
