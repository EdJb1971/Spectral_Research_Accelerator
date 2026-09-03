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
                                           minimum_family_for_detected_fraction,
                                           FIXTURE_EXPECTATIONS, ROWS_PER_CYCLE, SPAN_CYCLES,
                                           corrected_family, exact_partner_p_values,
                                           minimum_resolvable_family,
                                           monte_carlo_partner_p_values, statistic_grid)
from src.core.errors import InvalidParameterError
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


def test_the_monte_carlo_error_is_worst_at_the_inventory_sizes_a_study_would_try():
    """Why the wrong method is dangerous rather than merely wrong, measured across sizes.

    The anti-conservative error is a factor of `k`, so it shrinks as the inventory grows while the
    correction stringency grows with it. The two cross: at the resolvable family size the
    correction absorbs the error entirely, and at the handful-of-domains sizes anyone would
    actually reach for it inflates the family-wise error by more than an order of magnitude. The
    wrong method is therefore safe only where the right method already works, which is the shape of
    a trap rather than of an approximation, and is the reason it is refused by name rather than
    documented as an acceptable shortcut.
    """
    replications = 999
    rates = {}
    for size, realisations in ((6, 30), (30, 20)):
        firing = 0
        for index in range(realisations):
            grid = statistic_grid(FIXTURE_BUILDERS["same_normalisation_unrelated"](5000 + index,
                                                                                   size))
            rng = np.random.default_rng(index)
            p_values = []
            for member in range(size):
                alternatives = np.asarray([grid[member][other] for other in range(size)
                                           if other != member])
                draws = alternatives[rng.integers(len(alternatives), size=replications)]
                p_values.append((1 + int(np.sum(draws >= grid[member][member])))
                                / (1.0 + replications))
            corrected = adjust(p_values, method="benjamini_yekutieli", alpha=ALPHA, n_tests=size,
                               labels=[str(index) for index in range(size)])
            firing += sum(corrected["rejected"]) > 0
        rates[size] = firing / float(realisations)

    assert rates[6] > 0.5, (
        "the Monte Carlo form must be shown to fail badly on a small inventory, or the refusal "
        "has no measurement behind it")
    assert rates[6] > 4 * ALPHA and rates[30] < rates[6], (
        "the error must be shown to shrink with inventory size, which is what makes it a trap "
        "rather than a uniform bias")


# --------------------------------------------------- TG17.11 slice 4: the calibration itself


@pytest.fixture(scope="module")
def calibration():
    """The declared configuration, run once. A test that ran a cheaper one would not be the gate."""
    from src.benchmarks.shape_fixtures import calibrate_shape_family

    return calibrate_shape_family()


def test_every_case_meets_the_expectation_frozen_in_the_module(calibration):
    """The expectations live in `FIXTURE_EXPECTATIONS`, so this asserts them rather than restating."""
    by_case = {row["case"]: row for row in calibration["cases"]}
    assert set(by_case) == {entry["case"] for entry in FIXTURE_EXPECTATIONS}
    for entry in FIXTURE_EXPECTATIONS:
        assert by_case[entry["case"]]["met"], entry["expectation"]
    assert calibration["all_met"]


def test_the_planted_case_is_fully_recovered_and_the_safeguards_never_fire(calibration):
    """Power and false-positive rate, which are the two numbers the gate is about."""
    by_case = {row["case"]: row for row in calibration["cases"]}
    planted = by_case["planted_shape_recurrence"]
    assert planted["full_recovery_rate"] == 1.0
    assert planted["mean_rejections"] == float(calibration["family_size"])

    for case in ("same_normalisation_unrelated", "native_scale_alias"):
        safeguard = by_case[case]
        assert safeguard["family_wise_rejection_rate"] == 0.0
        assert safeguard["false_rejections"] == 0
        assert safeguard["member_tests"] >= 20 * calibration["family_size"]

    assert by_case["degenerate_inventory"]["refused"] is True


def test_the_calibration_records_that_it_resampled_nothing(calibration):
    """A guard against the trap being reintroduced by a later slice reaching for replications.

    The artefact states its inference and states that it used no replications. If a future change
    resamples this null, either this field stops being true or the change had to edit the sentence
    that says it is — and editing that sentence is a decision somebody has to make deliberately.
    """
    assert calibration["inference"] == "exact_partner_p_values"
    assert "none:" in calibration["replications"]
    assert calibration["p_value_floor"] == pytest.approx(1.0 / calibration["family_size"])
    assert "not chosen" in calibration["family_size_is"]
    assert "not evidence" in calibration["claim_boundary"]


def test_the_correction_holds_its_alpha_on_this_null_s_own_p_value_lattice(calibration):
    """The false-positive rate measured where draws are nearly free, beside the fixture rate.

    Twenty fixture realisations bound a zero count at 14% by the rule of three, which is weaker
    than the alpha being claimed. The lattice draws resolve the same quantity to a much tighter
    bound, and reporting both is what keeps the weaker one from being read as the stronger.
    """
    lattice = calibration["null_lattice"]
    assert lattice["draws"] >= 10000
    assert lattice["family_wise_false_positive_rate"] <= ALPHA
    assert lattice["mean_false_rejections"] <= ALPHA
    if lattice["family_wise_false_positive_rate"] == 0.0:
        assert lattice["one_sided_95_upper_bound"] < ALPHA


def test_the_detection_profile_is_a_step_and_names_the_inventory_each_fraction_needs(calibration):
    """The operating characteristic, which is the finding a study most needs and least expects.

    Every genuinely recurring member sits at exactly the same p-value floor, so there is no region
    of partial power: at a given inventory size a family either rejects or does not. The declared
    family is the smallest that can reject at all, so it is on the knife edge — it recovers a
    wholly recurring inventory and nothing sparser. Detecting a sparser recurrence is not a matter
    of more computation; it is a larger inventory, and the sizes are reported.
    """
    profile = calibration["detection"]
    row = next(entry for entry in profile["rows"]
               if entry["family_size"] == calibration["family_size"])
    outcomes = {entry["fraction"]: entry["rejects"] for entry in row["fractions"]}
    assert outcomes[1.0] is True
    assert all(outcomes[fraction] is False for fraction in outcomes if fraction < 1.0), (
        "the declared family is the smallest that can reject, so it must be knife-edge")

    minimums = profile["minimum_family_by_fraction"]
    assert minimums[0]["minimum_family_size"] == calibration["family_size"]
    sizes = [entry["minimum_family_size"] for entry in minimums]
    fractions = [entry["fraction"] for entry in minimums]
    assert fractions == sorted(fractions, reverse=True)
    assert sizes == sorted(sizes), "a sparser recurrence must require a larger inventory, not a smaller"
    assert sizes[-1] > 10 * sizes[0], (
        "the cost of detecting sparse recurrence must be shown, not implied")


def test_a_recurrence_too_sparse_to_detect_is_refused_with_the_reason_rather_than_answered():
    """`minimum_family_for_detected_fraction` must not return a number it did not find."""
    with pytest.raises(InvalidParameterError, match="not detectable under this null"):
        minimum_family_for_detected_fraction(0.01, limit=200)
    with pytest.raises(InvalidParameterError, match="fraction of the family"):
        minimum_family_for_detected_fraction(0.0)
