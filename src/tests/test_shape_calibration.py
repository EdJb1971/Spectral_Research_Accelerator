"""TG17.11 slice 2: the scale/shape statistic, and what it is allowed to claim.

The invariance this mode rests on is asserted here against a *measurement* rather than against a
threshold someone chose, and the two things that could be mistaken for it — row density and a
mis-declared native duration — are pinned separately so that neither can be quietly counted as
invariance.
"""

import numpy as np
import pytest

from src.benchmarks.family_calibration import case_trajectories
from src.benchmarks.shape_calibration import (MINIMUM_ROWS_PER_CYCLE, NATIVE_FACTORS,
                                              ShapeComparisonRefusal, ShapeRecord,
                                              measure_cadence_dependence,
                                              measure_normalisation_sensitivity,
                                              measure_phase_window_tradeoff,
                                              measure_scale_invariance, represent_at,
                                              shape_record_from_trajectory, shape_recurrence)


def _shape(phase):
    """A profile with a second harmonic, so a comparison has more to agree about than a sine."""
    return np.sin(2.0 * np.pi * phase) + 0.4 * np.sin(4.0 * np.pi * phase + 0.7)


def _synthetic(label, *, native_seconds, rows, span_cycles, profile=_shape, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, span_cycles * native_seconds, rows + 1)
    starts, ends = edges[:-1], edges[1:]
    phase = ((starts + ends) / 2.0) / native_seconds
    return ShapeRecord(label=label, starts=starts, ends=ends,
                       values=profile(phase) + noise * rng.standard_normal(rows),
                       valid=np.ones(rows, dtype=bool), native_seconds=native_seconds)


@pytest.fixture(scope="module")
def recurring_pair():
    """One shape presented at two native durations 2,222 times apart, at unrelated cadences."""
    return (_synthetic("light_curve", native_seconds=1800.0, rows=600, span_cycles=6,
                       noise=0.02, seed=1),
            _synthetic("float", native_seconds=4.0e6, rows=97, span_cycles=6,
                       noise=0.02, seed=2))


def test_the_statistic_separates_a_recurring_shape_from_an_unrelated_one(recurring_pair):
    left, right = recurring_pair
    unrelated = _synthetic("unrelated", native_seconds=4.0e6, rows=97, span_cycles=6,
                           profile=lambda phase: np.sin(2.0 * np.pi * (phase * 1.37 + 0.3)),
                           noise=0.02, seed=3)
    assert shape_recurrence(left, right) > 0.9
    assert shape_recurrence(left, unrelated) < 0.2


def test_scale_invariance_is_exact_because_the_denominator_is_declared_not_estimated(recurring_pair):
    """The mode's whole content, measured rather than asserted.

    `invariance.py` reports 0.37% reproduction for `relative_geometry` and records why
    `scale_normalised` is left out of `MATCHERS`: its denominator is an *estimated* scale that
    drifts +3.6% to -2.6% across a sixfold range, and the drift lands in the quotient. This
    denominator is a declared native duration, so there is no estimator to drift and the movement
    is at machine precision — which is a stronger claim and therefore needs the stricter test.
    """
    measured = measure_scale_invariance(*recurring_pair)
    assert measured["native_range"] == (min(NATIVE_FACTORS), max(NATIVE_FACTORS))
    assert measured["reference_statistic"] > 0.9
    assert measured["max_relative_deviation"] < 1e-12, measured["worst_presentation"]


def test_rewriting_a_record_at_a_different_row_density_does_not_move_the_statistic(recurring_pair):
    """Support-duration weighting in phase, so the comparison is over overlap and not over rows."""
    left, right = recurring_pair
    reference = shape_recurrence(left, right)
    for cadence_factor in (0.25, 0.5):
        finer = represent_at(right, cadence_factor=cadence_factor)
        assert len(finer.starts) != len(right.starts)
        assert abs(shape_recurrence(left, finer) - reference) / reference < 1e-9


def test_coarsening_past_the_resolution_floor_is_refused_rather_than_scored(recurring_pair):
    """Movement under coarsening is real information loss, and below the floor it is refused.

    A statistic that reported unchanged agreement at two rows per cycle would be inventing detail
    the rows no longer carry, so this is not treated as a failure of invariance. It is treated as
    a limit on what may be compared, and the limit is read off the measurement rather than picked.
    """
    measured = measure_cadence_dependence(*recurring_pair)
    scored = [row for row in measured["rows"] if not row["refused"]]
    refused = [row for row in measured["rows"] if row["refused"]]
    assert scored and refused, "the sweep must cross the floor to measure where it is"
    assert all(row["rows_per_cycle"] >= MINIMUM_ROWS_PER_CYCLE for row in scored)
    assert all(row["rows_per_cycle"] < MINIMUM_ROWS_PER_CYCLE for row in refused)
    fine = [row for row in scored if row["rows_per_cycle"] >= 16.0]
    assert fine and all(row["relative_deviation"] < 1e-9 for row in fine), (
        "row density must stop mattering well before the floor, or the floor is in the wrong place")


def test_a_mis_declared_native_duration_costs_more_the_more_cycles_are_compared(recurring_pair):
    """The trade-off the mode cannot escape, stated rather than discovered later.

    A phase comparison is phase-locked, so a wrong native duration drifts the records apart once
    per cycle. Comparing more cycles buys support and spends tolerance, and a study that picked a
    phase window without this table would be picking its own sensitivity by accident.
    """
    table = measure_phase_window_tradeoff(*recurring_pair)["table"]
    windows = [row["cycles"] for row in table]
    assert windows == sorted(windows) and len(windows) >= 3

    for row in table:
        exact = [cost for cost in row["costs"] if cost["declared_error"] == 0.0]
        assert exact and exact[0]["relative_deviation"] == 0.0
        positive = [cost for cost in row["costs"] if cost["declared_error"] > 0.0]
        deviations = [cost["relative_deviation"] for cost in
                      sorted(positive, key=lambda cost: cost["declared_error"])]
        assert deviations == sorted(deviations), "a larger error must not cost less"

    def cost_at(row, error):
        return next(c["relative_deviation"] for c in row["costs"] if c["declared_error"] == error)

    assert cost_at(table[-1], 0.10) > 4.0 * cost_at(table[0], 0.10), (
        "the cost of a mis-declaration must be shown to compound with the phase window")


def test_the_sensitivity_report_is_zero_where_the_declaration_is_right(recurring_pair):
    measured = measure_normalisation_sensitivity(*recurring_pair)
    exact = [row for row in measured["rows"] if row["declared_error"] == 0.0]
    assert exact and exact[0]["relative_deviation"] == 0.0
    assert measured["max_relative_deviation"] > 0.0, (
        "a sweep that found no cost would mean the phase axis was not being used")


def test_a_native_duration_in_an_unrecognised_unit_is_refused_not_converted():
    with pytest.raises(ShapeComparisonRefusal, match="native_units"):
        ShapeRecord(label="x", starts=[0.0, 1.0], ends=[1.0, 2.0], values=[0.0, 1.0],
                    valid=[True, True], native_seconds=1.0, native_units="days")


def test_no_flagship_fixture_declares_a_native_scale_that_can_carry_a_shape():
    """A finding about the existing fixtures, not about the statistic.

    Every G17 flagship record declares its structural scale as its own row cadence, so each
    resolves exactly one row per native cycle. A shape needs more than one sample per cycle to
    exist, so none of the calendar fixtures can be reused as a scale/shape fixture at its declared
    scale. Pinned here because it is what forces slice 3 to build its own records rather than
    borrow TG17.0's, and a later slice must not quietly borrow them anyway.
    """
    trajectories = case_trajectories("shared_calendar_event", focus="planted")
    records = {domain: shape_record_from_trajectory(trajectory)
               for domain, trajectory in trajectories.items()}
    assert records, "the flagship fixtures must still be readable as shape records"
    for domain, record in records.items():
        assert abs(record.rows_per_cycle - 1.0) < 0.01, domain
    domains = sorted(records)
    with pytest.raises(ShapeComparisonRefusal, match="rows per declared native cycle"):
        shape_recurrence(records[domains[0]], records[domains[1]])
