"""T4F.4: two directions over one table, an ordering the record is allowed to refuse, and a
ranking that decides the answer."""

from __future__ import annotations

import math

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, ConstellationPattern, PatternCatalogue, SignatureMetric, SignaturePoint,
    calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_events import ObservationGrid, events_from_catalogue
from src.analysis_engine.spectral_invariance import AxisAdmission, ConstellationSignature
from src.analysis_engine.spectral_precursors import (
    LagFamily, PRECURSOR_CLAIM_BOUNDARY, RULE_NOT_DISTINGUISHED, RULE_NO_ELIGIBLE_ANTECEDENT,
    RULE_PRECURSOR, precursor_report,
)
from src.analysis_engine.spectral_queries import (
    ANSWER_NONE_DISTINGUISHED, ANSWER_NONE_ORDERABLE, ANSWER_NOT_MEASURABLE, ANSWER_SIGNATURES,
    DIRECTION_BOTTOM_UP, DIRECTION_COUNTERPART_SIDE, DIRECTION_QUESTION, DIRECTION_TARGET_ROLE,
    DIRECTION_TOP_DOWN, FAMILY_PER_LAG, FAMILY_SELECTED_LAG, QUERY_CLAIM_BOUNDARY, QUERY_SCHEMA,
    RANK_CAUTION, RANK_CONFIDENCE, RANK_KEYS, RANK_LIFT, RANK_Q_VALUE, RANK_SUPPORT,
    SCALE_ORDER_SCHEMA, ScaleOrdering, bidirectional_query, directional_pair, scale_ordering,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.claim_ladder import OUTSIDE_THE_LADDER
from src.core.errors import InvalidParameterError


METRIC = SignatureMetric(AttributeWeights(
    geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))

#: Distinct geometry per letter so the letters are distinct patterns, and scales chosen so the
#: ordering has all three cases in it: strictly finer, strictly coarser, and not orderable.
FACTORS = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
SCALES = {
    "A": (2.0, 2.0, 4.0),        # fine: the planted precursor
    "B": (16.0, 16.0, 32.0),     # the coarse target every top-down query asks about
    "C": (3.0, 3.0, 5.0),        # fine, common, and unaligned: the decoy
    "D": (10.0, 20.0, 40.0),     # spans B's range, so B and D are not orderable at all
    "E": (64.0, 64.0, 128.0),    # coarser than B, so it is on the wrong side of a top-down ask
    "F": (1.0, 1.0, 2.0),        # fine, but every occurrence is censored by the record's end
}
BASE_ID = {letter: 1000 * (index + 1) for index, letter in enumerate(sorted(FACTORS))}

FRAMES = 220
WINDOW = TransitionWindow(1.0, 3.0, "frames")
SECOND_WINDOW = TransitionWindow(4.0, 6.0, "frames")
LAGS = LagFamily((WINDOW,))

#: The target's occurrences are deliberately irregular. A regularly spaced antecedent against a
#: regularly spaced consequent can be re-aligned by a rotation of the right phase, so a lattice
#: fixture would measure the lattice rather than the planted relation.
B_TIMES = [8, 15, 31, 36, 41, 72, 77, 92, 99, 121, 133, 140, 151, 159, 164, 175, 193, 209]
#: Two frames in front of every third target occurrence: perfect confidence on six trials.
A_TIMES = [time - 2 for time in B_TIMES[::3]]
#: Common, evenly spread, and related to nothing. It collects more hits than the real precursor
#: purely by occurring more often, which is the whole point of it.
C_TIMES = list(range(0, FRAMES, 7))
D_TIMES = [20, 60, 100, 180]
E_TIMES = [25, 65, 105, 185]
#: Late enough that its declared window runs past the last searched frame.
F_TIMES = [217, 218]

PLAN = ([("A", time) for time in A_TIMES] + [("B", time) for time in B_TIMES]
        + [("C", time) for time in C_TIMES] + [("D", time) for time in D_TIMES]
        + [("E", time) for time in E_TIMES] + [("F", time) for time in F_TIMES])


def _signature(letter, identity, time, *, scales=None, cardinality=3, mode="scale_specific",
               units="cells", jitter=0.0):
    edges = cardinality * (cardinality - 1) // 2
    return ConstellationSignature(
        key=(float(identity),) + tuple(range(cardinality)), time=float(time),
        cardinality=cardinality, mode=mode, order=tuple(range(cardinality)),
        geometry=tuple(FACTORS[letter] * (1.0 + 0.3 * index) * (1.0 + jitter)
                       for index in range(edges)),
        geometry_relation="distance",
        bearings=tuple(10.0 + 5.0 * index for index in range(edges)),
        strengths=tuple(0.75 + 0.1 * index for index in range(cardinality)),
        scales=tuple((scales.get(letter, SCALES[letter]) if isinstance(scales, dict) else scales)
                     if scales is not None else SCALES[letter]),
        scale_units=units, axis=AxisAdmission(True, 3.0, 1.0421, 25.0),
        track_ids=tuple(range(cardinality)),
        bands=tuple("L%d" % (index + 1) for index in range(cardinality)),
        time_units="frames")


def _tolerance(**overrides):
    calibration = [_signature("A", 900, 0.0, **overrides),
                   _signature("A", 901, 1.0, jitter=0.05, **overrides)]
    return calibrate_signature_tolerance(calibration, metric=METRIC)


def _catalogue(plan, **overrides):
    """Assemble the catalogue T4E.3 would produce, without re-running its agglomeration.

    The clustering itself is T4E.3's and has its own suite; what this one needs is a catalogue
    whose patterns have known scales. One test below does run `cluster_signatures` end to end,
    so the ordering is held to a genuinely clustered catalogue as well as to this one.
    """
    tolerance = _tolerance(**overrides)
    groups = {}
    counts = {}
    for letter, time in plan:
        index = counts.get(letter, 0)
        counts[letter] = index + 1
        groups.setdefault(letter, []).append(
            _signature(letter, BASE_ID[letter] + index, time,
                       jitter=0.003 * (index % 4), **overrides))
    patterns = []
    for index, letter in enumerate(sorted(groups)):
        members = tuple(groups[letter])
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=index + 1, members=members, centroid=points[0],
            tolerance_radius=tolerance.value,
            observed_radius=max(METRIC.distance(point, points[0]) for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=METRIC, tolerance=tolerance,
                            n_signatures=sum(len(group) for group in groups.values()))


def _series(catalogue, *, frames=FRAMES):
    grid = ObservationGrid(frames=tuple(float(index) for index in range(frames)),
                           time_units="frames", cadence=1.0)
    return events_from_catalogue(catalogue, grid=grid)


def _ids(catalogue):
    mapping = {}
    for pattern in catalogue:
        letters = {letter for letter in BASE_ID for member in pattern.members
                   if BASE_ID[letter] <= member.key[0] < BASE_ID[letter] + 900}
        assert len(letters) == 1
        mapping[letters.pop()] = pattern.pattern_id
    return mapping


@pytest.fixture(scope="module")
def record():
    catalogue = _catalogue(PLAN)
    return catalogue, _series(catalogue), _ids(catalogue)


@pytest.fixture(scope="module")
def ordering(record):
    return scale_ordering(record[0])


def _report(record, letters, *, lags=LAGS, n_surrogates=199, seed=20260905):
    _catalogue_obj, series, ids = record
    pairs = [(ids[a], ids[b]) for a, b in letters]
    return precursor_report(series, lags=lags, n_surrogates=n_surrogates, seed=seed,
                            pairs=pairs)


@pytest.fixture(scope="module")
def report(record):
    """The declared family: four antecedents asked about one target, at one lag."""
    return _report(record, [("A", "B"), ("C", "B"), ("D", "B"), ("E", "B")])


# ---------------------------------------------------------------------------------------------
# the ordering that gives coarse and fine a meaning
# ---------------------------------------------------------------------------------------------


def test_the_ordering_publishes_the_range_each_pattern_actually_occupied(ordering, record):
    ids = record[2]
    assert ordering.scale_units == "cells"
    assert ordering.mode == "scale_specific"
    assert (ordering.low[ids["A"]], ordering.high[ids["A"]]) == (2.0, 4.0)
    assert (ordering.low[ids["B"]], ordering.high[ids["B"]]) == (16.0, 32.0)
    assert (ordering.low[ids["D"]], ordering.high[ids["D"]]) == (10.0, 40.0)


def test_the_statistic_is_the_geometric_mean_of_every_member_node_scale(ordering, record):
    ids = record[2]
    expected = math.exp(sum(math.log(value) for value in SCALES["A"]) / 3.0)
    assert ordering.statistic[ids["A"]] == pytest.approx(expected)
    assert ordering.statistic[ids["B"]] == pytest.approx(
        math.exp(sum(math.log(value) for value in SCALES["B"]) / 3.0))


def test_finer_than_is_strict_and_asymmetric(ordering, record):
    ids = record[2]
    assert ordering.finer_than(ids["A"], ids["B"]) is True
    assert ordering.finer_than(ids["B"], ids["A"]) is False
    assert ordering.coarser_than(ids["B"], ids["A"]) is True


def test_overlapping_ranges_are_not_orderable_in_either_direction(ordering, record):
    ids = record[2]
    assert ordering.finer_than(ids["D"], ids["B"]) is False
    assert ordering.finer_than(ids["B"], ids["D"]) is False
    assert ordering.orderable(ids["B"], ids["D"]) is False
    assert tuple(sorted((ids["B"], ids["D"]))) in ordering.unorderable_pairs


def test_the_ordering_is_partial_and_says_how_partial(ordering, record):
    ids = record[2]
    # A and C both live at the fine end and their ranges touch, so the record does not order
    # them; naming one of them the coarser would be a sort speaking for the measurement.
    assert ordering.orderable(ids["A"], ids["C"]) is False
    assert len(ordering.unorderable_pairs) == ordering.describe()["n_unorderable_pairs"]
    assert 0 < len(ordering.unorderable_pairs) < ordering.describe()["n_pairs"]


def test_ranges_that_touch_at_one_scale_are_not_separated_by_the_record():
    """Sharing a scale is sharing a scale; the boundary is strict, not touching."""
    touching = scale_ordering(_catalogue(
        [("A", 0.0), ("B", 5.0)],
        scales={"A": (2.0, 2.0, 4.0), "B": (4.0, 8.0, 16.0)}))
    fine, coarse = sorted(touching.patterns, key=lambda item: touching.statistic[item])
    assert touching.high[fine] == touching.low[coarse] == 4.0
    assert touching.orderable(fine, coarse) is False
    assert touching.separation(fine, coarse) is None
    apart = scale_ordering(_catalogue(
        [("A", 0.0), ("B", 5.0)],
        scales={"A": (2.0, 2.0, 4.0), "B": (4.5, 8.0, 16.0)}))
    fine, coarse = sorted(apart.patterns, key=lambda item: apart.statistic[item])
    assert apart.finer_than(fine, coarse) is True


def test_the_relation_is_transitive_where_it_holds(ordering, record):
    ids = record[2]
    assert ordering.finer_than(ids["A"], ids["B"])
    assert ordering.finer_than(ids["B"], ids["E"])
    assert ordering.finer_than(ids["A"], ids["E"])


def test_separation_is_the_measured_gap_and_none_where_the_ranges_overlap(ordering, record):
    ids = record[2]
    assert ordering.separation(ids["A"], ids["B"]) == pytest.approx(12.0)
    assert ordering.separation(ids["B"], ids["A"]) == pytest.approx(12.0)
    assert ordering.separation(ids["B"], ids["D"]) is None


def test_scale_invariance_leaves_nothing_to_order_and_the_query_says_so():
    plan = [("A", 0.0), ("B", 5.0)]
    with pytest.raises(InvalidParameterError) as error:
        scale_ordering(_catalogue(plan, mode="scale_invariant", units=None))
    message = str(error.value).lower()
    assert "geometric mean" in message and "1.0" in message
    assert "absolute scale" in message


def test_the_refusal_rests_on_an_exact_fact_about_that_mode():
    """Every scale-invariant signature's geometric-mean scale is exactly one, by construction."""
    for scales in ((2.0, 2.0, 4.0), (16.0, 16.0, 32.0), (1.5, 90.0, 7.25)):
        point = SignaturePoint.from_signature(
            _signature("A", 1, 0.0, scales=scales, mode="scale_invariant", units=None))
        geometric = math.exp(sum(math.log(value) for value in point.scales)
                             / len(point.scales))
        assert geometric == pytest.approx(1.0, abs=1e-12)


def test_a_scale_without_a_unit_is_refused_rather_than_ordered():
    with pytest.raises(InvalidParameterError) as error:
        scale_ordering(_catalogue([("A", 0.0), ("B", 5.0)], units=None))
    assert "unit" in str(error.value).lower()


def test_two_scale_units_in_one_catalogue_are_two_orderings():
    tolerance = _tolerance()
    members = {
        "A": (_signature("A", 1, 0.0, units="cells"),),
        "B": (_signature("B", 2, 5.0, units="km"),),
    }
    patterns = tuple(
        ConstellationPattern(
            pattern_id=index + 1, members=group,
            centroid=SignaturePoint.from_signature(group[0]),
            tolerance_radius=tolerance.value, observed_radius=0.0)
        for index, (_letter, group) in enumerate(sorted(members.items())))
    catalogue = PatternCatalogue(patterns=patterns, metric=METRIC, tolerance=tolerance,
                                 n_signatures=2)
    with pytest.raises(InvalidParameterError) as error:
        scale_ordering(catalogue)
    assert "conversion nobody performed" in str(error.value)


def test_a_non_positive_scale_is_not_a_magnitude():
    """`ConstellationSignature` does not check this, so the ordering has to."""
    tolerance = _tolerance()
    good = _signature("A", 1, 0.0)
    bad = _signature("B", 2, 5.0, scales=(0.0, 1.0, 2.0))
    patterns = (
        ConstellationPattern(pattern_id=1, members=(good,),
                             centroid=SignaturePoint.from_signature(good),
                             tolerance_radius=tolerance.value, observed_radius=0.0),
        ConstellationPattern(pattern_id=2, members=(bad,),
                             centroid=SignaturePoint.from_signature(good),
                             tolerance_radius=tolerance.value, observed_radius=0.0))
    catalogue = PatternCatalogue(patterns=patterns, metric=METRIC, tolerance=tolerance,
                                 n_signatures=2)
    with pytest.raises(InvalidParameterError) as error:
        scale_ordering(catalogue)
    assert "positive finite scale" in str(error.value)


def test_an_empty_or_wrongly_typed_catalogue_is_refused():
    with pytest.raises(InvalidParameterError):
        scale_ordering("not a catalogue")
    tolerance = _tolerance()
    empty = PatternCatalogue(patterns=(), metric=METRIC, tolerance=tolerance, n_signatures=0)
    with pytest.raises(InvalidParameterError) as error:
        scale_ordering(empty)
    assert "silent nothing" in str(error.value)


def test_cardinality_is_not_scale():
    """A bigger constellation is not a coarser one, and the ordering reads scales only."""
    three = scale_ordering(_catalogue([("A", 0.0), ("B", 5.0)],
                                      scales=(2.0, 4.0, 8.0), cardinality=3))
    four = scale_ordering(_catalogue([("A", 0.0), ("B", 5.0)],
                                     scales=(2.0, 4.0, 8.0, 4.0), cardinality=4))
    assert three.statistic == pytest.approx(four.statistic)
    assert three.low == four.low and three.high == four.high


def test_a_pattern_range_spans_every_member_and_not_merely_the_first():
    """A cluster's members are not identical, and the range has to cover all of them."""
    tolerance = _tolerance()
    members = (_signature("A", 1, 0.0, scales=(2.0, 2.0, 4.0)),
               _signature("A", 2, 1.0, scales=(6.0, 6.0, 9.0)),
               _signature("B", 3, 2.0, scales=(30.0, 30.0, 40.0)))
    patterns = (
        ConstellationPattern(pattern_id=1, members=members[:2],
                             centroid=SignaturePoint.from_signature(members[0]),
                             tolerance_radius=tolerance.value, observed_radius=0.0),
        ConstellationPattern(pattern_id=2, members=members[2:],
                             centroid=SignaturePoint.from_signature(members[2]),
                             tolerance_radius=tolerance.value, observed_radius=0.0))
    ordering = scale_ordering(PatternCatalogue(
        patterns=patterns, metric=METRIC, tolerance=tolerance, n_signatures=3))
    assert (ordering.low[1], ordering.high[1]) == (2.0, 9.0)
    assert ordering.statistic[1] == pytest.approx(
        math.exp(sum(math.log(v) for v in (2.0, 2.0, 4.0, 6.0, 6.0, 9.0)) / 6.0))
    assert ordering.finer_than(1, 2)


def test_the_ordering_reads_a_genuinely_clustered_catalogue_too():
    tolerance = _tolerance()
    signatures = [_signature(letter, BASE_ID[letter] + index, float(index * 3 + offset),
                             jitter=0.003 * index)
                  for offset, letter in enumerate(("A", "B"))
                  for index in range(4)]
    catalogue = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
    ordering = scale_ordering(catalogue)
    assert len(ordering.patterns) == 2
    fine, coarse = sorted(ordering.patterns, key=lambda item: ordering.statistic[item])
    assert ordering.finer_than(fine, coarse)
    assert (ordering.low[fine], ordering.high[fine]) == (2.0, 4.0)


def test_the_ordering_receipt_publishes_its_basis_and_every_range(ordering):
    described = ordering.describe()
    assert described["schema"] == SCALE_ORDER_SCHEMA
    assert "disjoint" in described["basis"]
    assert described["n_patterns"] == len(ordering.patterns)
    assert {row["pattern_id"] for row in described["patterns"]} == set(ordering.patterns)
    for row in described["patterns"]:
        assert row["scale_low"] <= row["scale"] <= row["scale_high"]


def test_an_ordering_missing_a_bound_is_refused():
    with pytest.raises(InvalidParameterError):
        ScaleOrdering(scale_units="cells", mode="scale_specific",
                      statistic={1: 2.0, 2: 4.0}, low={1: 1.0}, high={1: 3.0})


def test_an_unknown_pattern_is_refused_rather_than_treated_as_unorderable(ordering):
    with pytest.raises(InvalidParameterError) as error:
        ordering.finer_than(9999, ordering.patterns[0])
    assert "scale range" in str(error.value)


# ---------------------------------------------------------------------------------------------
# one engine, two directions
# ---------------------------------------------------------------------------------------------


def test_top_down_returns_the_finer_side_and_counts_what_it_withheld(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.rows_in_family == 4
    assert result.rows_naming_target == 4
    assert result.withheld_overlapping_scale == 1     # D's range spans B's
    assert result.withheld_wrong_side == 1            # E is coarser than B, not finer
    assert [entry.counterpart for entry in result] == [ids["A"], ids["C"]]
    assert result.status == ANSWER_SIGNATURES


def test_the_rows_that_named_the_target_are_counted_apart_from_the_whole_family(
        record, ordering):
    ids = record[2]
    wider = _report(record, [("A", "B"), ("A", "C")], n_surrogates=99)
    result = bidirectional_query(wider, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.rows_in_family == 2
    assert result.rows_naming_target == 1
    assert len(result) == 1


def test_bottom_up_returns_the_coarser_side(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_BOTTOM_UP, target=ids["A"],
                                 ordering=ordering)
    assert [entry.counterpart for entry in result] == [ids["B"]]
    assert result.entries[0].status == RULE_PRECURSOR
    assert result.entries[0].scale_separation == pytest.approx(12.0)
    assert result.status == ANSWER_SIGNATURES


def test_the_two_directions_read_the_very_same_row(report, ordering, record):
    ids = record[2]
    down = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                               ordering=ordering)
    up = bidirectional_query(report, direction=DIRECTION_BOTTOM_UP, target=ids["A"],
                             ordering=ordering)
    theirs = [entry for entry in down if entry.counterpart == ids["A"]][0]
    assert theirs.row is up.entries[0].row
    assert (theirs.p_value, theirs.q_value, theirs.status) == (
        up.entries[0].p_value, up.entries[0].q_value, up.entries[0].status)


def test_the_direction_is_two_arguments_to_one_selection_and_nothing_else():
    assert set(DIRECTION_TARGET_ROLE) == set(DIRECTION_COUNTERPART_SIDE)
    assert DIRECTION_TARGET_ROLE[DIRECTION_TOP_DOWN] == "consequent"
    assert DIRECTION_TARGET_ROLE[DIRECTION_BOTTOM_UP] == "antecedent"
    assert DIRECTION_COUNTERPART_SIDE[DIRECTION_TOP_DOWN] == "finer"
    assert DIRECTION_COUNTERPART_SIDE[DIRECTION_BOTTOM_UP] == "coarser"


def test_the_question_each_direction_publishes_is_the_side_it_actually_selects():
    for direction in (DIRECTION_TOP_DOWN, DIRECTION_BOTTOM_UP):
        side = DIRECTION_COUNTERPART_SIDE[direction]
        question = DIRECTION_QUESTION[direction]
        other = "coarser" if side == "finer" else "finer"
        assert side in question and other not in question


def test_there_is_no_default_direction(report, ordering, record):
    ids = record[2]
    with pytest.raises(TypeError):
        bidirectional_query(report, target=ids["B"], ordering=ordering)
    with pytest.raises(InvalidParameterError) as error:
        bidirectional_query(report, direction="downwards", target=ids["B"], ordering=ordering)
    assert "answer a question nobody asked" in str(error.value)


def test_every_considered_row_comes_back_including_the_ones_that_found_nothing(
        report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert len(result) == 2
    assert len(result.signatures) == 1
    assert {entry.status for entry in result} == {RULE_PRECURSOR, RULE_NOT_DISTINGUISHED}


def test_a_target_the_family_never_asked_about_in_that_role_is_refused_by_name(
        report, ordering, record):
    ids = record[2]
    with pytest.raises(InvalidParameterError) as error:
        bidirectional_query(report, direction=DIRECTION_BOTTOM_UP, target=ids["B"],
                            ordering=ordering)
    assert "never asked" in str(error.value)
    with pytest.raises(InvalidParameterError):
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["A"],
                            ordering=ordering)


def test_an_ordering_that_does_not_cover_the_report_is_refused(report, record):
    partial = scale_ordering(_catalogue([(letter, float(index))
                                         for index, letter in enumerate("ABCD")]))
    with pytest.raises(InvalidParameterError) as error:
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=record[2]["B"],
                            ordering=partial)
    assert "lies about its denominator" in str(error.value)


def test_a_query_needs_a_completed_report_and_a_measured_ordering(report, ordering, record):
    ids = record[2]
    with pytest.raises(InvalidParameterError) as error:
        bidirectional_query({"rules": []}, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                            ordering=ordering)
    assert "after seeing the data" in str(error.value)
    with pytest.raises(InvalidParameterError):
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                            ordering={"A": 1.0})


def test_a_target_outside_the_ordering_is_refused(report, ordering):
    with pytest.raises(InvalidParameterError):
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=9999,
                            ordering=ordering)


# ---------------------------------------------------------------------------------------------
# a query is a view, and a view is not a test
# ---------------------------------------------------------------------------------------------


def test_nothing_is_recomputed_and_every_figure_is_the_report_own(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    for entry in result:
        row = report.rule(entry.antecedent, entry.consequent, entry.window)
        assert (entry.p_value, entry.q_value, entry.status) == (
            row.p_value, row.q_value, row.status)
        assert (entry.support, entry.confidence, entry.lift) == (
            row.support, row.confidence, row.lift)


def test_the_family_published_is_the_declared_one_not_the_answer(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.correction_family["size"] == 4
    assert len(result) == 2
    assert "does not restrict the family" in result.correction_family["note"]


def test_asking_a_narrower_question_afterwards_does_not_raise_any_answer(record, ordering):
    """The shortcut this module exists to refuse, measured rather than merely warned about."""
    ids = record[2]
    declared = _report(record, [("A", "B"), ("C", "B"), ("D", "B"), ("E", "B")])
    narrowed = _report(record, [("A", "B")])
    wide = declared.rule(ids["A"], ids["B"], WINDOW)
    narrow = narrowed.rule(ids["A"], ids["B"], WINDOW)
    assert wide.p_value == narrow.p_value          # the same record, the same null, one seed
    assert narrow.q_value < wide.q_value           # only the family changed
    result = bidirectional_query(declared, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.signatures[0].q_value == wide.q_value


def test_two_different_questions_of_one_report_return_identical_figures(
        report, ordering, record):
    ids = record[2]
    down = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                               ordering=ordering)
    up = bidirectional_query(report, direction=DIRECTION_BOTTOM_UP, target=ids["A"],
                             ordering=ordering)
    assert down.signatures[0].q_value == up.signatures[0].q_value


# ---------------------------------------------------------------------------------------------
# the ranking is a scientific choice
# ---------------------------------------------------------------------------------------------


def test_the_obvious_ranking_puts_an_undistinguished_pattern_first(report, ordering, record):
    """'What most commonly preceded it', taken literally, answers with the commonest pattern."""
    ids = record[2]
    by_support = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                     ordering=ordering, rank_by=RANK_SUPPORT)
    by_q = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                               ordering=ordering, rank_by=RANK_Q_VALUE)
    leader = by_support.entries[0]
    assert leader.counterpart == ids["C"]
    assert leader.status == RULE_NOT_DISTINGUISHED
    assert leader.support > by_q.entries[0].support
    assert by_q.entries[0].counterpart == ids["A"]
    assert by_q.entries[0].status == RULE_PRECURSOR


def test_the_result_says_the_rankings_disagree(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.rankings_disagree is True
    assert result.top_by[RANK_SUPPORT] != result.top_by[RANK_Q_VALUE]
    assert result.top_by[RANK_LIFT] == result.top_by[RANK_CONFIDENCE] == result.top_by[
        RANK_Q_VALUE]
    # top_by names the entry that came first, not merely one of them.
    for key in RANK_KEYS:
        leader = [entry for entry in result if entry.ranks[key] == 1][0]
        assert result.top_by[key] == (leader.counterpart,
                                      float(leader.window.minimum_lag),
                                      float(leader.window.maximum_lag))


def test_one_answer_cannot_disagree_with_itself(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_BOTTOM_UP, target=ids["A"],
                                 ordering=ordering)
    assert len(result) == 1
    assert result.rankings_disagree is False


def test_every_entry_carries_its_rank_under_every_key(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    for entry in result:
        assert set(entry.ranks) == set(RANK_KEYS)
        assert all(rank in (1, 2) for rank in entry.ranks.values())


def test_the_misleading_keys_carry_the_sentence_saying_how(report, ordering, record):
    assert RANK_CAUTION[RANK_Q_VALUE] is None
    assert "most commonly preceded it" in RANK_CAUTION[RANK_SUPPORT]
    assert "not referenced to anything" in RANK_CAUTION[RANK_CONFIDENCE]
    assert "handful" in RANK_CAUTION[RANK_LIFT]
    ids = record[2]
    described = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                    ordering=ordering, rank_by=RANK_SUPPORT).describe()
    assert described["ranking_caution"] == RANK_CAUTION[RANK_SUPPORT]
    assert set(described["ranking_cautions"]) == set(RANK_KEYS)


def test_a_smaller_q_ranks_first_while_a_larger_lift_does(report, ordering, record):
    ids = record[2]
    result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    by_rank = {entry.ranks[RANK_Q_VALUE]: entry for entry in result}
    assert by_rank[1].q_value < by_rank[2].q_value
    assert by_rank[1].lift > by_rank[2].lift
    assert by_rank[1].ranks[RANK_LIFT] == 1


def test_the_entries_come_back_in_the_requested_order(report, ordering, record):
    ids = record[2]
    for key in RANK_KEYS:
        result = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                     ordering=ordering, rank_by=key)
        assert [entry.ranks[key] for entry in result] == [1, 2]


def test_the_order_is_deterministic_across_repeated_queries(report, ordering, record):
    ids = record[2]
    runs = [tuple(entry.counterpart for entry in
                  bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                      ordering=ordering, rank_by=RANK_SUPPORT))
            for _ in range(3)]
    assert len(set(runs)) == 1


def test_an_unknown_ranking_key_is_refused(report, ordering, record):
    with pytest.raises(InvalidParameterError):
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=record[2]["B"],
                            ordering=ordering, rank_by="interestingness")


# ---------------------------------------------------------------------------------------------
# the two families, and what each of them can be asked
# ---------------------------------------------------------------------------------------------


def test_the_selected_lag_family_is_one_row_per_pair(record, ordering):
    ids = record[2]
    two = _report(record, [("A", "B"), ("C", "B")],
                  lags=LagFamily((WINDOW, SECOND_WINDOW)))
    per_lag = bidirectional_query(two, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                  ordering=ordering, family=FAMILY_PER_LAG)
    selected = bidirectional_query(two, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                   ordering=ordering, family=FAMILY_SELECTED_LAG)
    assert len(per_lag) == 4 and len(selected) == 2
    assert all(entry.support is None for entry in selected)
    assert selected.correction_family["family"] == FAMILY_SELECTED_LAG
    # Each family was corrected against its own size, and each answer says which it was.
    assert per_lag.correction_family["size"] == 4
    assert selected.correction_family["size"] == 2
    assert selected.entries[0].q_value == two.selected_rule(
        selected.entries[0].antecedent, selected.entries[0].consequent).q_value


def test_ranking_the_selected_family_by_support_is_refused_by_name(record, ordering):
    ids = record[2]
    two = _report(record, [("A", "B"), ("C", "B")],
                  lags=LagFamily((WINDOW, SECOND_WINDOW)))
    with pytest.raises(InvalidParameterError) as error:
        bidirectional_query(two, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                            ordering=ordering, family=FAMILY_SELECTED_LAG,
                            rank_by=RANK_SUPPORT)
    assert "belongs to the lag that won" in str(error.value)


def test_a_counterpart_appears_once_per_declared_lag_in_the_per_lag_family(record, ordering):
    ids = record[2]
    two = _report(record, [("A", "B"), ("C", "B")],
                  lags=LagFamily((WINDOW, SECOND_WINDOW)))
    result = bidirectional_query(two, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    windows = {(entry.counterpart, entry.window) for entry in result}
    assert len(windows) == 4
    assert sorted(entry.ranks[RANK_Q_VALUE] for entry in result) == [1, 2, 3, 4]
    # Support is the count that succeeded, not the count that was eligible: the decoy has five
    # times the eligible trials of the planted rule and still ranks below it at the second lag.
    by_support = bidirectional_query(two, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                     ordering=ordering, rank_by=RANK_SUPPORT)
    assert [entry.support for entry in by_support] == [9, 6, 5, 0]
    assert [entry.row.eligible_antecedents for entry in by_support] == [31, 6, 31, 6]


def test_an_unknown_family_is_refused(report, ordering, record):
    with pytest.raises(InvalidParameterError):
        bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=record[2]["B"],
                            ordering=ordering, family="everything")


# ---------------------------------------------------------------------------------------------
# the four things an answer can be
# ---------------------------------------------------------------------------------------------


def test_an_answer_of_nothing_distinguished_is_not_an_answer_of_nothing(record, ordering):
    ids = record[2]
    only_decoy = _report(record, [("C", "B")], n_surrogates=99)
    result = bidirectional_query(only_decoy, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.status == ANSWER_NONE_DISTINGUISHED
    assert len(result) == 1 and not result.signatures


def test_nothing_orderable_is_reported_as_such_rather_than_as_nothing_found(record, ordering):
    ids = record[2]
    only_overlap = _report(record, [("D", "B")], n_surrogates=99)
    result = bidirectional_query(only_overlap, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.status == ANSWER_NONE_ORDERABLE
    assert result.rows_naming_target == 1
    assert result.withheld_overlapping_scale == 1
    assert len(result) == 0


def test_an_unmeasurable_counterpart_is_not_a_measured_absence(record, ordering):
    ids = record[2]
    censored = _report(record, [("F", "B")], n_surrogates=99)
    assert censored.rules[0].status == RULE_NO_ELIGIBLE_ANTECEDENT
    result = bidirectional_query(censored, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering)
    assert result.status == ANSWER_NOT_MEASURABLE
    assert result.unmeasured_entries == 1
    assert result.entries[0].measured is False
    assert all(rank is None for rank in result.entries[0].ranks.values())
    assert result.rankings_disagree is False


def test_what_could_not_be_measured_is_returned_but_never_ranked_above_what_was(
        record, ordering):
    ids = record[2]
    mixed = _report(record, [("A", "B"), ("F", "B")], n_surrogates=99)
    result = bidirectional_query(mixed, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                 ordering=ordering, rank_by=RANK_SUPPORT)
    assert result.status == ANSWER_SIGNATURES
    assert [entry.counterpart for entry in result] == [ids["A"], ids["F"]]
    assert result.entries[0].ranks[RANK_SUPPORT] == 1
    assert result.entries[1].ranks[RANK_SUPPORT] is None
    assert result.unmeasured_entries == 1


def test_no_status_this_module_emits_names_a_term_the_ladder_places_outside_itself():
    statuses = (ANSWER_SIGNATURES, ANSWER_NONE_DISTINGUISHED, ANSWER_NONE_ORDERABLE,
                ANSWER_NOT_MEASURABLE)
    for status in statuses:
        words = set(status.lower().split("_"))
        assert not words & {term.lower() for term in OUTSIDE_THE_LADDER}


# ---------------------------------------------------------------------------------------------
# what a direction word does and does not claim
# ---------------------------------------------------------------------------------------------


def test_the_claim_boundary_stops_below_causation_and_below_a_second_test():
    lowered = QUERY_CLAIM_BOUNDARY.lower()
    for forbidden in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert forbidden in lowered      # named in order to be refused
    assert "is not a second test" in lowered
    assert "asserts nothing about which of them acts on the other" in lowered


def test_the_receipt_carries_the_question_it_answered(report, ordering, record):
    ids = record[2]
    described = bidirectional_query(report, direction=DIRECTION_TOP_DOWN, target=ids["B"],
                                    ordering=ordering).describe()
    assert described["schema"] == QUERY_SCHEMA
    assert described["question"] == DIRECTION_QUESTION[DIRECTION_TOP_DOWN]
    assert described["target_role"] == "consequent"
    assert described["counterpart_side"] == "finer"
    assert described["target_scale_range"] == [ordering.low[ids["B"]],
                                               ordering.high[ids["B"]]]
    assert described["target_scale_range"][0] < described["target_scale_range"][1]
    assert described["scale_units"] == "cells"
    assert described["correction_family"]["declared_before_the_record_was_read"] is True
    assert described["claim_boundary"] == QUERY_CLAIM_BOUNDARY
    assert described["report_claim_boundary"] == PRECURSOR_CLAIM_BOUNDARY
    assert len(described["entries"]) == 2


def test_neither_direction_of_a_pair_can_be_read_off_the_other(record):
    ids = record[2]
    both = _report(record, [("A", "B"), ("B", "A")], n_surrogates=99)
    pair = directional_pair(both, ids["A"], ids["B"], WINDOW)
    assert pair["forward"]["confidence"] != pair["reverse"]["confidence"]
    assert pair["lift_differs"] is True
    assert "different denominators" in pair["basis"]
    assert pair["forward"]["support"] != pair["reverse"]["support"]


def test_directional_pair_needs_a_report(record):
    ids = record[2]
    with pytest.raises(InvalidParameterError):
        directional_pair(None, ids["A"], ids["B"], WINDOW)
