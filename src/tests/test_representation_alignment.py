"""Mutual k-NN alignment, and the two references the Platonic paper never supplies.

The measurement under test is the one arXiv:2405.07987 reports as 0.16 out of 1 and then
asks, in its own limitations section, whether 0.16 is a lot. These tests pin the answer's
machinery: the closed-form chance floor, the permutation null that holds each kernel's
geometry fixed, and the refusals that stop either from being read as more than it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis_engine.representation_alignment import (
    ALIGNMENT_BOUNDARY, chance_alignment, describe_measurement, knn_sets,
    mutual_knn_alignment, permutation_null,
)
from src.core.errors import InvalidParameterError


def _kernel(points: np.ndarray) -> np.ndarray:
    """Inner-product kernel, as the paper defines it: K(i,j) = <f(i), f(j)>."""
    return points @ points.T


def _paired_views(n: int, dim: int, *, noise: float, seed: int):
    """Two representations of ONE point set: a rotation of each other, plus noise.

    Structurally aligned by construction, so the metric should be high and the permuted
    pairing should not reproduce it.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, dim))
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    left = latent
    right = latent @ rotation + noise * rng.normal(size=(n, dim))
    return _kernel(left), _kernel(right)


# ------------------------------------------------------------------ the metric itself

def test_a_kernel_is_perfectly_aligned_with_itself():
    kernel, _ = _paired_views(60, 8, noise=0.0, seed=1)
    assert mutual_knn_alignment(kernel, kernel, 5) == 1.0


def test_alignment_is_invariant_to_rotation_because_the_kernel_is():
    """An inner-product kernel is rotation invariant, so a rotated view is the same kernel."""
    left, right = _paired_views(60, 8, noise=0.0, seed=2)
    assert mutual_knn_alignment(left, right, 5) == pytest.approx(1.0)


def test_alignment_falls_as_the_two_views_are_driven_apart():
    scores = [mutual_knn_alignment(*_paired_views(120, 10, noise=noise, seed=3), 10)
              for noise in (0.0, 0.5, 1.5, 4.0)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_neighbour_sets_break_ties_deterministically_by_index():
    kernel = np.ones((6, 6))
    first, second = knn_sets(kernel, 3), knn_sets(kernel, 3)
    assert np.array_equal(first, second)
    # Every off-diagonal similarity is equal, so point 0's neighbours are 1, 2, 3 in order.
    assert list(first[0]) == [1, 2, 3]


# ------------------------------------------------------------------ the chance floor

@pytest.mark.parametrize("n,k", [(1000, 10), (500, 10), (100, 10), (1000, 50)])
def test_the_analytic_chance_floor_matches_random_neighbour_sets(n, k):
    """Two independent uniform k-subsets of n-1 points intersect in k^2/(n-1) on average."""
    rng = np.random.default_rng(11)
    draws = np.array([len(set(rng.choice(n - 1, k, replace=False).tolist())
                          & set(rng.choice(n - 1, k, replace=False).tolist())) / k
                      for _ in range(4000)])
    standard_error = draws.std(ddof=1) / np.sqrt(len(draws))
    assert abs(draws.mean() - chance_alignment(n, k)) < 4 * standard_error + 1e-12


def test_the_chance_floor_rises_with_k_and_falls_with_n():
    assert chance_alignment(1000, 10) < chance_alignment(1000, 50)
    assert chance_alignment(1000, 10) < chance_alignment(100, 10)
    assert chance_alignment(101, 100) == 1.0


# ------------------------------------------------------------------ the permutation null

def test_a_genuinely_paired_view_clears_its_permuted_pairing():
    left, right = _paired_views(150, 10, noise=0.4, seed=5)
    report = permutation_null(left, right, k=10, permutations=100, seed=0)
    assert report["measured_alignment"] > report["null_quantile_95"]
    assert report["permutation_exceedance"] == pytest.approx(1 / 101)
    reading = describe_measurement(report, label="rotated view")
    assert reading["outcome"] == "alignment_exceeds_permuted_pairing"
    assert "not about convergence" in reading["reading"]


def test_two_unrelated_representations_do_not_clear_their_null():
    """The case the paper never runs. An unpaired measurement must come back unresolved."""
    rng = np.random.default_rng(9)
    left, right = _kernel(rng.normal(size=(150, 10))), _kernel(rng.normal(size=(150, 10)))
    report = permutation_null(left, right, k=10, permutations=100, seed=0)
    assert report["measured_alignment"] <= report["null_quantile_95"]
    assert describe_measurement(report)["outcome"] == "unresolved"


def test_the_null_sits_near_the_analytic_floor_for_unstructured_kernels():
    rng = np.random.default_rng(13)
    left, right = _kernel(rng.normal(size=(200, 12))), _kernel(rng.normal(size=(200, 12)))
    report = permutation_null(left, right, k=10, permutations=120, seed=1)
    assert report["null_mean"] == pytest.approx(report["analytic_chance_alignment"], abs=0.02)


@pytest.mark.parametrize("structure", ["clustered", "duplicated"])
def test_the_closed_form_survives_strong_kernel_structure(structure):
    """The closed form estimates the null mean even when the kernels are badly non-uniform.

    A first version of this test asserted the opposite -- that clustering would lift the
    permutation null well above k/(n-1). It does not, and the measurement is recorded here
    rather than quietly dropped. Under a uniform permutation the right-hand neighbour sets
    land uniformly whatever structure they carry, so the null mean returns to the floor.
    Strong clusters and nine-fold duplicated points both agree with it to under one percent.
    """
    rng = np.random.default_rng(17)
    if structure == "clustered":
        n, per = 180, 30
        blocks = np.repeat(np.arange(n // per), per)
        points = np.eye(n // per)[blocks] * 6.0 + rng.normal(size=(n, n // per)) * 0.3
    else:
        points = np.repeat(rng.normal(size=(20, 6)), 9, axis=0)
        points = points + rng.normal(size=points.shape) * 0.01
    left = _kernel(points)
    shuffled = rng.permutation(len(points))
    right = left[np.ix_(shuffled, shuffled)]
    report = permutation_null(left, right, k=10, permutations=120, seed=2)
    floor = report["analytic_chance_alignment"]
    assert abs(report["null_mean"] - floor) < 0.05 * floor
    assert describe_measurement(report)["outcome"] == "unresolved"


def test_the_null_supplies_the_spread_the_closed_form_cannot():
    """Which is why the permutation null is still the one to run: a floor is not a threshold."""
    rng = np.random.default_rng(23)
    left, right = _kernel(rng.normal(size=(180, 12))), _kernel(rng.normal(size=(180, 12)))
    report = permutation_null(left, right, k=10, permutations=150, seed=5)
    assert report["null_std"] > 0.0
    assert report["null_quantile_95"] > report["analytic_chance_alignment"]


def test_an_exceedance_is_never_reported_as_exactly_zero():
    left, right = _paired_views(80, 8, noise=0.0, seed=21)
    report = permutation_null(left, right, k=5, permutations=20, seed=0)
    assert report["permutation_exceedance"] > 0.0


# ------------------------------------------------------------------ refusals

def test_a_non_square_kernel_is_refused():
    with pytest.raises(InvalidParameterError):
        mutual_knn_alignment(np.zeros((4, 5)), np.zeros((4, 5)), 2)


def test_two_kernels_over_different_point_sets_are_refused():
    left, _ = _paired_views(40, 6, noise=0.0, seed=31)
    other, _ = _paired_views(50, 6, noise=0.0, seed=32)
    with pytest.raises(InvalidParameterError) as raised:
        mutual_knn_alignment(left, other, 5)
    assert "one ordered point set" in str(raised.value)


def test_a_non_finite_similarity_is_refused_rather_than_treated_as_distant():
    kernel, _ = _paired_views(20, 5, noise=0.0, seed=41)
    broken = kernel.copy()
    broken[3, 7] = np.nan
    with pytest.raises(InvalidParameterError) as raised:
        mutual_knn_alignment(broken, kernel, 4)
    assert "not a distant one" in str(raised.value)


@pytest.mark.parametrize("k", [0, -1, 40])
def test_a_neighbour_count_outside_one_to_n_minus_one_is_refused(k):
    kernel, _ = _paired_views(40, 6, noise=0.0, seed=51)
    with pytest.raises(InvalidParameterError):
        mutual_knn_alignment(kernel, kernel, k)


def test_a_null_with_no_permutations_is_refused():
    kernel, other = _paired_views(30, 6, noise=0.5, seed=61)
    with pytest.raises(InvalidParameterError) as raised:
        permutation_null(kernel, other, k=4, permutations=0)
    assert "an unmeasured null is not a null" in str(raised.value)


def test_too_few_points_to_have_neighbours_is_refused():
    with pytest.raises(InvalidParameterError):
        mutual_knn_alignment(np.eye(2), np.eye(2), 1)


def test_every_report_carries_what_the_measurement_does_not_license():
    left, right = _paired_views(60, 8, noise=0.3, seed=71)
    report = permutation_null(left, right, k=5, permutations=20, seed=0)
    assert report["boundary"] == ALIGNMENT_BOUNDARY
    assert "licenses no claim about" in ALIGNMENT_BOUNDARY
    assert describe_measurement(report)["boundary"] == ALIGNMENT_BOUNDARY
