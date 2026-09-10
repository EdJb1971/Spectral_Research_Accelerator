"""T4E.25: the coverage/contamination sweep, and the properties its comparison depends on.

The sweep's whole claim is that its four alphas are four cuts through one measurement rather
than four runs. These tests pin the things that would silently break that: a swept list that
drifted from the declaration, a configuration-coverage count that admitted a configuration it
should not, and the monotonicity of recovery in alpha, which is what makes "newly recovered"
a set difference rather than an estimate.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.extraction import calibrate, calibration_from_maxima
from src.statistics import surrogates as surrogate_module
from tools.measure_coverage_contamination import DECLARED_ALPHAS, configuration_coverage


# ---------------- the swept list is closed, and closed in the declaration


def test_the_swept_alphas_are_the_ones_the_declaration_fixed():
    """Adding a fifth alpha once the curve is visible would fit the sweep to its own answer."""
    assert DECLARED_ALPHAS == (0.05, 0.10, 0.25, 0.50)


def test_the_sweep_starts_at_the_operating_point_it_must_reproduce():
    """Acceptance condition 2 compares against T4E.24, which ran at 0.05."""
    assert DECLARED_ALPHAS[0] == 0.05


def test_the_swept_alphas_are_strictly_increasing():
    """The step differences are only meaningful against an ordered sweep."""
    assert list(DECLARED_ALPHAS) == sorted(DECLARED_ALPHAS)
    assert len(set(DECLARED_ALPHAS)) == len(DECLARED_ALPHAS)


# ---------------- one ensemble, four cuts: the pairing the design rests on


def test_a_shared_ensemble_reproduces_calibrate_exactly_at_the_same_alpha():
    """If this ever drifts, the alpha = 0.05 baseline stops being T4E.24's and the sweep lies."""
    values = np.random.default_rng(3).normal(size=(48, 48))
    direct = calibrate(values, alpha=0.05, n_surrogates=99, seed=17)

    ensemble = surrogate_module.generate(values, method="phase_randomise", n=99, seed=17)
    maxima = [float(np.max(np.asarray(m))) for m in ensemble["members"]]
    shared = calibration_from_maxima(maxima, alpha=0.05, method="phase_randomise", seed=17)

    assert shared.threshold == direct.threshold
    assert shared.null_maxima == direct.null_maxima


def test_a_higher_alpha_never_raises_the_cut():
    """Recovery is monotone in alpha because the threshold is, and the sweep assumes it."""
    values = np.random.default_rng(4).normal(size=(48, 48))
    ensemble = surrogate_module.generate(values, method="phase_randomise", n=199, seed=21)
    maxima = [float(np.max(np.asarray(m))) for m in ensemble["members"]]

    thresholds = [calibration_from_maxima(maxima, alpha=a, method="phase_randomise",
                                          seed=21).threshold
                  for a in DECLARED_ALPHAS]
    assert thresholds == sorted(thresholds, reverse=True)


# ---------------- configuration coverage: the quantity the addendum showed to be binding


def test_a_configuration_with_one_never_seen_feature_is_not_recoverable():
    """The object a criterion would match was never assembled in any scene."""
    result = configuration_coverage([[6, 6, 0]], scenes=6)

    assert result["every_feature_seen_at_least_once"] == 0
    assert result["recoverable_in_principle"] == 0.0
    assert result["intact_across_all_scenes"] == 0.0


def test_a_configuration_seen_partially_everywhere_is_recoverable_but_not_intact():
    result = configuration_coverage([[6, 5, 1]], scenes=6)

    assert result["recoverable_in_principle"] == 1.0
    assert result["intact_across_all_scenes"] == 0.0


def test_a_wholly_present_configuration_counts_as_both():
    result = configuration_coverage([[6, 6, 6]], scenes=6)

    assert result["recoverable_in_principle"] == 1.0
    assert result["intact_across_all_scenes"] == 1.0
    assert result["mean_fraction_of_a_configuration_always_present"] == 1.0


def test_the_two_coverage_measures_are_ordered_by_construction():
    """Intact implies recoverable, so the second can never exceed the first."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        configurations = [list(rng.integers(0, 7, size=int(rng.integers(3, 11))))
                          for _ in range(12)]
        result = configuration_coverage(configurations, scenes=6)
        assert result["intact_across_all_scenes"] <= result["recoverable_in_principle"]


def test_the_addendum_figures_are_reproduced_by_this_function():
    """T4E.24's committed receipt, read through the code that will report it at every alpha."""
    import json
    from pathlib import Path
    import collections

    receipt = json.loads(
        Path("measurements/t4e24_false_absence.json").read_text(encoding="utf-8"))
    grouped = collections.defaultdict(list)
    for row in receipt["rows"]:
        grouped[row["configuration"]].append(int(row["seen_in"]))
    result = configuration_coverage(list(grouped.values()),
                                    scenes=receipt["scenes_per_configuration"])

    assert result["configurations"] == 60
    assert result["every_feature_seen_at_least_once"] == 10
    assert result["every_feature_seen_in_all_scenes"] == 5


def test_no_configurations_returns_not_a_number_rather_than_zero():
    """An empty sweep has no coverage; reporting 0.0 would claim it measured one."""
    import math

    result = configuration_coverage([], scenes=6)
    assert result["configurations"] == 0
    assert math.isnan(result["recoverable_in_principle"])
    assert math.isnan(result["intact_across_all_scenes"])


def test_an_alpha_below_the_ensembles_resolution_is_refused_by_name():
    """The declaration says the refusal is never reached at 999; this pins that it exists."""
    from src.core.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        calibration_from_maxima([1.0, 2.0, 3.0], alpha=0.001)
