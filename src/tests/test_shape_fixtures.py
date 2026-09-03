"""TG17.11 slice 3: the frozen scale/shape fixtures, and the inference they are scored under.

The four cases carry their expected answers in `shape_fixtures.FIXTURE_EXPECTATIONS`, stated from
what each case *is* rather than from what it was measured to do. What is checked here is that each
fixture really has the property its name claims — otherwise a case could pass by not testing
anything — and that the inference applied to them is the exact partner test rather than the
Monte Carlo template, whose failure on this null is measured rather than argued.
"""

import numpy as np
import pytest

from src.benchmarks.shape_calibration import MINIMUM_ROWS_PER_CYCLE
from src.benchmarks.shape_fixtures import (ALPHA, FAMILY_SIZE, FIXTURE_BUILDERS,
                                           FIXTURE_EXPECTATIONS, ROWS_PER_CYCLE, SPAN_CYCLES,
                                           corrected_family, exact_partner_p_values,
                                           minimum_resolvable_family,
                                           monte_carlo_partner_p_values, statistic_grid)
from src.core.structural_nulls import NullRefusal
from src.statistics.multiple_comparisons import adjust


SEED = 20260903


@pytest.fixture(scope="module")
def families():
    """Each scoreable case run once as one corrected family, which is the level it is stated at."""
    return {entry["case"]: corrected_family(FIXTURE_BUILDERS[entry["case"]](SEED))
            for entry in FIXTURE_EXPECTATIONS if entry["outcome"] != "refuses"}


def test_the_family_size_is_solved_from_the_null_rather_than_chosen():
    """A member has `k - 1` alternative partners, so its p-value cannot fall below `1/k`.

    That floor is a property of the null, not of the budget, and Benjamini-Yekutieli then decides
    what family size it permits. 105 is the answer at alpha 0.05 and is asserted against the real
    correction: a family of 104 cannot reject even when every member beats every alternative, which
    is the most favourable result the null can produce.
    """
    assert FAMILY_SIZE == minimum_resolvable_family() == 105
    for size, expected in ((FAMILY_SIZE - 1, False), (FAMILY_SIZE, True)):
        corrected = adjust([1.0 / size] * size, method="benjamini_yekutieli", alpha=ALPHA,
                           n_tests=size, labels=[str(index) for index in range(size)])
        assert all(corrected["rejected"]) is expected, size


def test_every_fixture_record_resolves_a_shape_rather_than_its_own_cadence():
    """The failure that disqualified the TG17.0 records must not be reintroduced here."""
    for case, build in FIXTURE_BUILDERS.items():
        for left, right in build(SEED, 4):
            for record in (left, right):
                assert record.rows_per_cycle == pytest.approx(ROWS_PER_CYCLE, rel=0.05), case
                assert record.rows_per_cycle > 2 * MINIMUM_ROWS_PER_CYCLE, case
                assert record.phase_span == pytest.approx(SPAN_CYCLES, rel=0.05), case


def test_the_planted_case_carries_a_shared_shape_and_the_safeguards_do_not():
    """Each fixture is checked to have the property its name claims, before it is scored.

    A safeguard that happened to contain no structure at all would pass by being empty rather than
    by being absorbed, and a planted case whose shape did not actually recur would fail for the
    wrong reason. The separation is between the declared pairing and its alternatives, because
    that is exactly what the null compares.
    """
    def own_and_alternatives(case):
        grid = statistic_grid(FIXTURE_BUILDERS[case](SEED, 20))
        off_diagonal = grid[~np.eye(len(grid), dtype=bool)]
        return float(np.mean(np.diag(grid))), float(np.mean(off_diagonal))

    own, alternatives = own_and_alternatives("planted_shape_recurrence")
    assert own > 0.95 and own - alternatives > 0.4

    own, alternatives = own_and_alternatives("same_normalisation_unrelated")
    assert abs(own - alternatives) < 0.1, "a declared pairing must be no better than any other"

    own, alternatives = own_and_alternatives("native_scale_alias")
    assert own > 0.6 and alternatives > 0.6, (
        "the alias fixture must correlate strongly, or it is not testing aliasing at all")
    assert own - alternatives < 0.1, "the aliasing must be shared by the alternatives too"


@pytest.mark.parametrize("entry", [e for e in FIXTURE_EXPECTATIONS if e["outcome"] == "rejects"],
                         ids=lambda entry: entry["case"])
def test_the_planted_case_is_recovered_at_the_declared_alpha(entry, families):
    result = families[entry["case"]]
    assert result["family_size"] == FAMILY_SIZE
    assert result["n_rejected_after_correction"] == FAMILY_SIZE, entry["expectation"]


@pytest.mark.parametrize("entry",
                         [e for e in FIXTURE_EXPECTATIONS if e["outcome"] == "does_not_reject"],
                         ids=lambda entry: entry["case"])
def test_each_safeguard_rejects_nothing(entry, families):
    result = families[entry["case"]]
    assert result["n_rejected_after_correction"] == 0, entry["expectation"]


def test_the_degenerate_inventory_refuses_rather_than_reporting_a_safeguard_passing():
    """D91's case as a fixture. A p-value of 1.0 here would read as the safeguard working."""
    pairings = FIXTURE_BUILDERS["degenerate_inventory"](SEED, 6)
    with pytest.raises(NullRefusal, match="admissible alternative partner"):
        exact_partner_p_values(pairings)


def test_the_p_value_floor_is_the_inventory_size_and_is_reported_as_such(families):
    """The resolution the null actually has, carried in the result rather than left implicit."""
    for result in families.values():
        assert result["p_value_floor"] == pytest.approx(1.0 / FAMILY_SIZE)
        assert min(result["p_values"]) >= result["p_value_floor"] - 1e-12
        assert set(result["reference_sizes"]) == {FAMILY_SIZE}


def test_the_monte_carlo_template_is_registered_and_refused_by_name():
    """The trap is an answered question rather than one nobody thought about.

    Calendar mode's method is correct there and wrong here, and the difference is a property of
    the null: a clock shift's support far exceeds the replication count, while partner
    reassignment offers `k - 1` values per member however many replications are paid for.
    """
    with pytest.raises(NullRefusal, match="exact partner test"):
        monte_carlo_partner_p_values()


def test_resampling_a_finite_null_would_understate_the_p_value_by_the_inventory_size():
    """The measurement behind the refusal, run rather than quoted.

    A member that beats all its alternatives is reported by the Monte Carlo form at the
    replication floor, when its exact tail probability is `1/k`. The ratio is the whole error, and
    it runs anti-conservatively: the wrong number is the small one.
    """
    size, replications = 6, 999
    pairings = FIXTURE_BUILDERS["same_normalisation_unrelated"](SEED, size)
    grid = statistic_grid(pairings)
    exact = exact_partner_p_values(pairings)

    rng = np.random.default_rng(SEED)
    monte_carlo = []
    for index in range(size):
        alternatives = np.asarray([grid[index][other] for other in range(size) if other != index])
        draws = alternatives[rng.integers(len(alternatives), size=replications)]
        monte_carlo.append((1 + int(np.sum(draws >= grid[index][index]))) / (1.0 + replications))

    assert min(exact["p_values"]) == pytest.approx(1.0 / size)
    assert min(monte_carlo) == pytest.approx(1.0 / (1 + replications))
    assert min(exact["p_values"]) > 100 * min(monte_carlo), (
        "the two forms must be shown to disagree by the order of magnitude claimed")

    for name, p_values, expected in (("monte carlo", monte_carlo, True), ("exact", exact["p_values"], False)):
        corrected = adjust(list(p_values), method="benjamini_yekutieli", alpha=ALPHA,
                           n_tests=size, labels=[str(index) for index in range(size)])
        assert (sum(corrected["rejected"]) > 0) is expected, (
            "%s reported %d rejections on an inventory with no shared shape"
            % (name, sum(corrected["rejected"])))


def test_the_calendar_null_is_not_affected_because_its_support_is_not_finite_in_practice():
    """The refusal is a property of this null, not a blanket rule, so the contrast is measured.

    `family_calibration.py` draws 999 replications and is right to: an independent clock shift over
    two records of 56 and 1,344 rows has a surrogate space of tens of thousands, so each draw is a
    genuinely new surrogate and the Monte Carlo denominator claims no resolution the null lacks.
    Partner reassignment offers `k - 1` values per member. Without this comparison the refusal
    above would read as a general suspicion of resampling rather than as the specific finding it is.
    """
    from src.benchmarks.family_calibration import case_trajectories, support_weighted_correlation
    from src.core.structural_nulls import bind_null

    trajectories = case_trajectories("same_window_unrelated", focus="safeguards")
    bound = bind_null("independent_native_clock_shift", {}, mode="calendar_aligned")
    left, right = trajectories["argo_float"], trajectories["tess_lightcurve"]
    draws = 200
    distinct = {round(support_weighted_correlation(bound.apply(left, seed),
                                                   bound.apply(right, 10000 + seed)), 12)
                for seed in range(draws)}
    assert len(distinct) > 0.95 * draws, (
        "the calendar null must be shown to supply a new surrogate per draw, or the scale/shape "
        "refusal would have to apply to it too")
