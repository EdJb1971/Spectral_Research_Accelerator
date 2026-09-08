"""T4E.10: estimators for an identity radius, and the support each one needs.

T4E.9 found that a radius calibrated at 90% recall on 15 pairs and frozen split 46.7% of motif
pairs in a partition it had never seen, with construction labels and AUC 1.0. These tests pin
the estimators built in response, and above all the refusal: an estimator that cannot support
its declared coverage must decline to name a radius rather than name one carrying no guarantee.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import beta

from src.analysis_engine.operating_point import (
    ESTIMATORS, bootstrap_upper, empirical_quantile, estimate,
    nonparametric_tolerance_bound, required_support,
)
from src.core.errors import InvalidParameterError, UnknownNameError


def _sample(n, seed=0):
    return np.random.default_rng(seed).gamma(2.0, 0.003, size=n)


# ------------------------------------------------------------------ the support requirement

@pytest.mark.parametrize("coverage,confidence,expected", [
    (0.9, 0.9, 22), (0.95, 0.95, 59), (0.9, 0.5, 7), (0.99, 0.9, 230),
])
def test_the_required_support_is_the_closed_form(coverage, confidence, expected):
    """From 1 - coverage**n >= confidence, the maximum's guarantee and so the weakest one."""
    assert required_support(coverage, confidence) == expected


def test_the_required_support_agrees_with_the_order_statistic_it_is_derived_from():
    """At exactly the required n the maximum works; one below, no order statistic does."""
    coverage = confidence = 0.9
    n = required_support(coverage, confidence)
    assert 1.0 - beta.cdf(coverage, n, 1) >= confidence
    assert all(1.0 - beta.cdf(coverage, k, n - 1 - k + 1) < confidence
               for k in range(1, n))


# ---------------------------------------------------------------------------- the refusal

def test_the_tolerance_bound_refuses_thin_support_and_names_what_would_carry_it():
    """The finding T4E.9 could not reach: the support was insufficient before any radius."""
    point = nonparametric_tolerance_bound(_sample(15), coverage=0.9, confidence=0.9)
    assert point.refused is True
    assert point.radius is None
    assert "22 observations are required" in point.refusal
    assert "carry no guarantee at all" in point.refusal
    assert point.describe()["required_support"] == 22


def test_the_tolerance_bound_names_a_radius_once_the_support_carries_it():
    point = nonparametric_tolerance_bound(_sample(28), coverage=0.9, confidence=0.9)
    assert point.refused is False
    assert point.radius is not None
    assert point.describe()["achieved_confidence"] >= 0.9


def test_the_tolerance_bound_is_never_narrower_than_the_empirical_quantile():
    """A bound that must hold on unseen data cannot be tighter than the sample's own quantile."""
    for n in (22, 30, 60, 200):
        sample = _sample(n, seed=n)
        bound = nonparametric_tolerance_bound(sample, coverage=0.9, confidence=0.9)
        quantile = empirical_quantile(sample, coverage=0.9)
        assert bound.radius >= quantile.radius


def test_an_empty_population_refuses_rather_than_returning_zero():
    for estimator in (empirical_quantile, nonparametric_tolerance_bound, bootstrap_upper):
        point = estimator([], coverage=0.9, confidence=0.9)
        assert point.refused is True and point.radius is None


# --------------------------------------------------------------- what each estimator claims

def test_the_empirical_quantile_states_that_it_guarantees_nothing():
    point = empirical_quantile(_sample(40), coverage=0.9)
    assert "None." in point.guarantee
    assert "below the population quantile about half the time" in point.guarantee


def test_the_empirical_quantile_ignores_confidence_and_says_so():
    """Accepting a parameter it cannot honour, silently, is how an unearned number is issued."""
    low = empirical_quantile(_sample(40), coverage=0.9, confidence=0.5)
    high = empirical_quantile(_sample(40), coverage=0.9, confidence=0.999)
    assert low.radius == high.radius
    assert low.describe()["confidence_requested_and_ignored"] == 0.5


def test_the_bootstrap_widens_rather_than_refusing_and_names_its_weakness():
    point = bootstrap_upper(_sample(15), coverage=0.9, confidence=0.9)
    assert point.refused is False
    assert "not exact at small support" in point.guarantee


def test_the_bootstrap_is_deterministic_for_one_seed():
    first = bootstrap_upper(_sample(30), coverage=0.9, confidence=0.9, seed=7)
    second = bootstrap_upper(_sample(30), coverage=0.9, confidence=0.9, seed=7)
    assert first.radius == second.radius


def test_every_registered_estimator_publishes_a_guarantee():
    for name in ESTIMATORS.names():
        specification = ESTIMATORS.get(name)
        assert specification.guarantees
        point = estimate(name, _sample(40), coverage=0.9, confidence=0.9)
        assert point.guarantee == specification.guarantees
        assert point.estimator == name


def test_only_the_tolerance_bound_declares_that_it_refuses_on_thin_support():
    refusing = {name for name in ESTIMATORS.names()
                if ESTIMATORS.get(name).refuses_on_insufficient_support}
    assert refusing == {"nonparametric_tolerance_bound"}


# ---------------------------------------------------------------------------------- refusals

def test_an_unknown_estimator_is_refused_with_its_correction():
    with pytest.raises(UnknownNameError) as raised:
        estimate("nonparametric_tolerance_bounds", _sample(30))
    assert "nonparametric_tolerance_bound" in str(raised.value)


@pytest.mark.parametrize("coverage,confidence", [(0.0, 0.9), (1.0, 0.9), (0.9, 0.0), (0.9, 1.0)])
def test_a_coverage_or_confidence_outside_the_open_unit_interval_is_refused(coverage, confidence):
    with pytest.raises(InvalidParameterError):
        nonparametric_tolerance_bound(_sample(30), coverage=coverage, confidence=confidence)


def test_a_negative_or_non_finite_distance_is_refused():
    for bad in ([0.1, -0.2], [0.1, math.nan], [0.1, math.inf]):
        with pytest.raises(InvalidParameterError):
            nonparametric_tolerance_bound(bad, coverage=0.9, confidence=0.9)
