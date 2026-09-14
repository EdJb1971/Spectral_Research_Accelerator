"""T4E.27: the tolerance, its components, and the refusals that are the point of it.

The failure this module exists to prevent is a bar that looks authoritative and cannot be
argued with. So most of what is pinned here is that the parts stay visible, that a missing
input is refused rather than defaulted, and that "we could not say" never collapses into "no".
"""
from __future__ import annotations

import math

import pytest

from src.analysis_engine.position_tolerance import (
    DECLARED_GRID_KM_PER_CELL,
    DECLARED_LOCALISATION_CELLS,
    acceptance_curve,
    acceptance_over,
    localisation_km,
    tolerance_for,
)


# ---------------- the components, and their provenance


def test_the_estimator_component_is_the_number_t4e21_measured():
    """Measured on synthetic ground truth for a different purpose, and used as measured."""
    assert DECLARED_LOCALISATION_CELLS == 0.295
    assert DECLARED_GRID_KM_PER_CELL == 27.75
    assert localisation_km() == pytest.approx(8.186, abs=0.01)


def test_the_tolerance_is_the_quadrature_sum_of_its_two_parts():
    tolerance = tolerance_for("X", 11.12)
    assert tolerance.total_km == pytest.approx(math.hypot(11.12, localisation_km()))


def test_every_component_carries_where_it_came_from():
    """A bar a reader cannot trace is one they can only accept or reject."""
    described = tolerance_for("X", 20.0).describe()
    names = {c["name"] for c in described["components"]}

    assert names == {"catalogue_uncertainty", "estimator_localisation"}
    for component in described["components"]:
        assert component["source"].strip(), component


def test_the_excluded_component_is_named_in_the_description():
    """What is missing from a bar matters as much as what is in it."""
    described = tolerance_for("X", 20.0).describe()
    assert "surface centre" in described["what_is_not_included"]
    assert "fit the bar to the result" in described["what_is_not_included"]


# ---------------- a missing uncertainty is refused, not defaulted


def test_a_zero_radius_is_refused_by_name_rather_than_used():
    """The exact case that made one of eighteen storms unpassable by construction."""
    tolerance = tolerance_for("LINDA", 0.0)

    assert tolerance.refused is True
    assert tolerance.total_km is None
    assert "missing report" in tolerance.refusal
    assert "exactly zero" in tolerance.refusal


def test_an_absent_radius_is_refused_too():
    assert tolerance_for("X", None).refused is True


def test_a_non_finite_radius_is_refused_rather_than_propagated():
    """An infinite bar admits every separation, which is worse than having no bar at all."""
    assert tolerance_for("X", float("nan")).refused is True
    assert tolerance_for("X", float("inf")).refused is True
    assert "admits everything" in tolerance_for("X", float("inf")).refusal


def test_a_refused_tolerance_answers_none_and_never_false():
    """`we could not say` and `no` are different answers; conflating them counts a missing
    agency report as a failed detection, which is the error this task exists to correct."""
    tolerance = tolerance_for("LINDA", 0.0)

    assert tolerance.admits(1.0) is None
    assert tolerance.admits(10_000.0) is None
    assert tolerance.unexplained_residual(50.0) is None


# ---------------- admitting, and the residual that measures what is missing


def test_a_separation_inside_the_bar_is_admitted_and_one_outside_is_not():
    tolerance = tolerance_for("X", 20.0)
    assert tolerance.admits(5.0) is True
    assert tolerance.admits(500.0) is False


def test_the_boundary_is_inclusive_and_stated():
    tolerance = tolerance_for("X", 20.0)
    assert tolerance.admits(tolerance.total_km) is True


def test_the_residual_is_what_the_justified_components_do_not_explain():
    tolerance = tolerance_for("X", 20.0)
    assert tolerance.unexplained_residual(100.0) == pytest.approx(100.0 - tolerance.total_km)


def test_a_comfortable_separation_gives_a_negative_residual_rather_than_zero():
    """Clipping to zero would discard the information that a join closed with room to spare."""
    tolerance = tolerance_for("X", 60.0)
    assert tolerance.unexplained_residual(10.0) < 0.0


# ---------------- counting, with the refused out of the denominator


def test_refused_observations_leave_the_denominator():
    tolerances = [tolerance_for("A", 50.0), tolerance_for("B", 0.0), tolerance_for("C", 50.0)]
    result = acceptance_over([1.0, 1.0, 9999.0], tolerances)

    assert result["judged"] == 2
    assert result["refused"] == 1
    assert result["refused_observations"] == ("B",)
    assert result["admitted"] == 1
    assert result["rate"] == pytest.approx(0.5)


def test_a_population_that_is_entirely_refused_reports_no_rate():
    result = acceptance_over([1.0], [tolerance_for("B", 0.0)])
    assert result["judged"] == 0
    assert result["rate"] is None


def test_mismatched_inputs_are_refused_rather_than_zipped_short():
    """Silently truncating would drop observations off the end of the population."""
    with pytest.raises(ValueError):
        acceptance_over([1.0, 2.0], [tolerance_for("A", 10.0)])


# ---------------- the curve is for inspection


def test_the_curve_is_monotone_in_the_bar():
    tolerances = [tolerance_for(str(i), 10.0) for i in range(5)]
    curve = acceptance_curve([10.0, 50.0, 100.0, 200.0, 400.0], tolerances,
                             [5.0, 25.0, 75.0, 150.0, 500.0])
    admitted = [point["admitted"] for point in curve]

    assert admitted == sorted(admitted)
    assert admitted[-1] == 5


def test_the_curve_excludes_refused_observations_from_its_denominator_too():
    tolerances = [tolerance_for("A", 10.0), tolerance_for("B", 0.0)]
    curve = acceptance_curve([1.0, 1.0], tolerances, [100.0])

    assert curve[0]["judged"] == 1
    assert curve[0]["admitted"] == 1
