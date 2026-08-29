"""TG0.3: the negative control, and the FAIL/INVALID distinction that makes it mean something.

Phase G0 closes here. TG0.1 showed the inference layer *could* take channels from anywhere;
TG0.2 sent a non-atmospheric record through it. Neither established that a **null** record is
reported as a null rather than as an inconclusive run — and that distinction is the difference
between an instrument and a machine for generating plausible findings.

Three verdicts, and the middle one is the point:

*   **PASS** — the same corrected relationship replicates independently in train and test.
*   **FAIL** — an adequately powered absence. A scientific result.
*   **INVALID** — the run could not adjudicate. Never displayable as a negative finding.

A system that returns "nothing found" when it was arithmetically incapable of finding anything
is worse than one that returns nothing at all, because the first looks like evidence. So the
load-bearing assertions here are `test_a_null_record_returns_fail_not_invalid_and_not_pass` and
`test_a_family_that_shrank_is_invalid_not_fail`.

The false-positive calibration is deliberately a measurement rather than an assertion of
correctness: 60 independent AR(1) records through the full gate produced 60 FAIL and 0 PASS,
which bounds the false-positive rate below 4.87% at 95% confidence (the bound a run of 60
can support, not a claim that the rate is zero). The test below re-runs a smaller set, because a
calibration that takes two minutes on every commit stops being run.
"""

import numpy as np
import pytest

from src.analysis_engine import domain_analysis as da
from src.analysis_engine.cross_scale import GateProtocol
from src.core.channel_series import (ChannelSeries, GATE_MEASURES, require_gate_measure,
                                     split_channel_series)
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, UnknownNameError

N_FRAMES = 900
PLANTED_LAG = 3
CADENCE = 60.0
CHANNELS = ("alpha", "beta", "gamma")


def _declaration(**overrides):
    kwargs = dict(
        name="synthetic_instrument_log",
        description="Three unitless channels on a regular clock. No physics.",
        licence="CC-BY-4.0 (fixture; no real archive was accessed)",
        axes=(AxisSpec("minute", "time", units="s"),
              AxisSpec("channel", "category", ordered=False)),
        violations=("no_physical_metric", "no_propagation_speed", "no_natural_cycle"),
        lag_policy="declared", declared_floor_frames=2,
        declared_floor_basis="two reporting intervals")
    kwargs.update(overrides)
    return DomainDeclaration(**kwargs)


def _protocol(**overrides):
    kwargs = dict(study_id="tg0.3-acceptance", n_scales=3, lags=(3,),
                  expected_frames=N_FRAMES, cadence_seconds=CADENCE, train_ratio=0.6,
                  embargo_frames=3, measure="value", bins=4, n_surrogates=499,
                  alpha=0.05, require_advection_floor=False, seed=20260823)
    kwargs.update(overrides)
    return GateProtocol(**kwargs)


def _ar1(rng, rho=0.6):
    x = np.zeros(N_FRAMES)
    noise = rng.normal(size=N_FRAMES)
    for t in range(1, N_FRAMES):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def _record(seed, coupling, **overrides):
    """`alpha` at t sets `gamma` at t+3 when coupled; `beta` is never involved."""
    rng = np.random.default_rng(seed)
    alpha, beta, gamma_noise = _ar1(rng), _ar1(rng), _ar1(rng)
    gamma = np.empty(N_FRAMES)
    gamma[:PLANTED_LAG] = gamma_noise[:PLANTED_LAG]
    gamma[PLANTED_LAG:] = (coupling * alpha[:-PLANTED_LAG]
                           + (1.0 - coupling) * gamma_noise[PLANTED_LAG:])
    kwargs = dict(channels=CHANNELS,
                  times_seconds=np.arange(N_FRAMES, dtype=float) * CADENCE,
                  measures={"value": np.column_stack([alpha, beta, gamma])},
                  support_parent_px=(1, 1, 1))
    kwargs.update(overrides)
    return ChannelSeries(**kwargs)


# --------------------------------------------------------------- the three verdicts

def test_a_planted_relationship_replicates_across_the_split_and_passes():
    result = da.run_domain_gate(_record(7, 0.9), _declaration(), _protocol())
    assert result["verdict"] == "PASS"
    assert [row["label"] for row in result["gate"]["replicated"]] == ["alpha->gamma@3"]
    assert result["split"]["embargo_at_least_longest_lag"] is True
    assert result["split"]["train_frames"] == 540
    assert result["split"]["test_frames"] == 357


@pytest.mark.parametrize("seed", [1001, 1002, 1003, 1004, 1005])
def test_a_null_record_returns_fail_not_invalid_and_not_pass(seed):
    """An adequately powered absence is a result. This is the load-bearing test of G0."""
    result = da.run_domain_gate(_record(seed, 0.0), _declaration(), _protocol())
    assert result["verdict"] == "FAIL"
    assert result["gate"]["problems"] == []
    assert result["gate"]["replicated"] == []
    for partition in ("train", "test"):
        assert result[partition]["power"]["can_reject_after_correction"] is True, (
            "a FAIL is only meaningful if the partition could have rejected")


def test_a_family_that_shrank_is_invalid_not_fail():
    """A channel dropping out reduces the family. That is inconclusive, not negative.

    Reported as INVALID because the frozen design declared six tests and six were corrected
    for; two tests corrected as though they were six is a different experiment, and calling
    its empty result FAIL would present a design error as evidence of absence.
    """
    result = da.run_domain_gate(_record(7, 0.9, usable=(True, True, False)),
                                _declaration(), _protocol())
    assert result["verdict"] == "INVALID"
    assert any("hypotheses" in problem for problem in result["gate"]["problems"])
    assert result["gate"]["replicated"] == []


def test_the_calibration_finds_no_relationship_in_independent_noise():
    """A smaller re-run of the recorded 60-trial calibration (60 FAIL, 0 PASS).

    Kept short on purpose: a calibration slow enough to be skipped is not a calibration.
    """
    verdicts = [da.run_domain_gate(_record(2000 + i, 0.0), _declaration(),
                                   _protocol())["verdict"]
                for i in range(8)]
    assert set(verdicts) == {"FAIL"}, verdicts


# ------------------------------------------------------------------ design refusals

def test_an_underpowered_design_is_refused_before_it_runs():
    """Refused at validation, so it can never reach the gate and be explained afterwards."""
    with pytest.raises(InvalidParameterError):
        _protocol(n_surrogates=20).validate()


def test_an_embargo_shorter_than_the_longest_lag_is_refused():
    with pytest.raises(InvalidParameterError, match="at least the longest tested lag"):
        _protocol(lags=(3, 6), embargo_frames=3).validate()


def test_a_protocol_that_miscounts_the_channels_is_refused():
    """Declaring two channels for a three-channel record corrects the wrong family size.

    Deliberately understated rather than overstated: `n_scales=4` is refused earlier still,
    by the power check, because a larger declared family needs more surrogates. Both orders
    are correct; this one reaches the channel-count guard.
    """
    with pytest.raises(InvalidParameterError, match="number of channels actually present"):
        da.run_domain_gate(_record(7, 0.9), _declaration(), _protocol(n_scales=2))


def test_an_inflated_family_is_refused_by_the_power_check_first():
    with pytest.raises(InvalidParameterError, match="cannot produce a significant result"):
        da.run_domain_gate(_record(7, 0.9), _declaration(), _protocol(n_scales=4))


def test_a_record_shorter_than_the_design_is_refused():
    with pytest.raises(InvalidParameterError, match="frames actually present"):
        da.run_domain_gate(_record(7, 0.9), _declaration(),
                           _protocol(expected_frames=N_FRAMES + 1))


def test_demanding_an_advective_floor_from_a_domain_without_one_is_refused():
    """Otherwise the gate returns INVALID for what is really a mis-declared design."""
    with pytest.raises(InvalidParameterError, match="mis-declared design"):
        da.run_domain_gate(_record(7, 0.9), _declaration(),
                           _protocol(require_advection_floor=True))


# ------------------------------------------------- the measure registry (found by TG0.3)

def test_a_generic_measure_is_admissible_and_the_threshold_measure_is_not():
    """R3 forbids a measure that moves with a threshold, not one that lacks a wavelet name."""
    assert require_gate_measure("value").threshold_free is True
    with pytest.raises(InvalidParameterError, match="factor of ten"):
        require_gate_measure("threshold_fraction")
    with pytest.raises(InvalidParameterError, match="factor of ten"):
        _protocol(measure="threshold_fraction").validate()


def test_an_unknown_measure_lists_the_registered_ones():
    with pytest.raises(UnknownNameError) as caught:
        _protocol(measure="engery_density").validate()      # deliberate typo
    assert "energy_density" in str(caught.value)


def test_the_four_wavelet_measures_are_still_admissible():
    """The registry widened what is accepted; it must not have changed the atmospheric set."""
    for measure in ("energy_density", "energy_fraction", "participation_ratio", "gini"):
        assert require_gate_measure(measure).threshold_free is True
    assert "threshold_fraction" in GATE_MEASURES


# ---------------------------------------------------------------------- the split (R6)

def test_the_embargo_is_returned_rather_than_silently_dropped():
    parts = split_channel_series(_record(7, 0.9), train_ratio=0.6, embargo_frames=8)
    assert parts["train"].n_times == 540
    assert parts["embargo"].n_times == 8
    assert parts["test"].n_times == N_FRAMES - 540 - 8
    # No frame appears in two partitions: the gap is real, not a boundary.
    assert parts["train"].times_seconds[-1] < parts["embargo"].times_seconds[0]
    assert parts["embargo"].times_seconds[-1] < parts["test"].times_seconds[0]


def test_the_split_records_its_own_provenance():
    parts = split_channel_series(_record(7, 0.9), train_ratio=0.6, embargo_frames=8)
    provenance = parts["test"].provenance
    assert provenance["split"] == "test"
    assert provenance["split_frames"] == [548, N_FRAMES]
    assert provenance["split_embargo_frames"] == 8


def test_a_split_that_starves_a_partition_is_refused():
    with pytest.raises(InvalidParameterError, match="at least two frames"):
        split_channel_series(_record(7, 0.9), train_ratio=0.999, embargo_frames=8)


# ------------------------------------------------------------------- the claim boundary

def test_the_verdict_carries_its_own_claim_boundary_and_domain():
    result = da.run_domain_gate(_record(7, 0.9), _declaration(), _protocol())
    assert result["protocol_fingerprint"] == _protocol().fingerprint()
    assert result["domain"]["name"] == "synthetic_instrument_log"
    boundary = result["claim_boundary"]
    assert "not causality" in boundary and "not universality" in boundary
    assert "R20" in boundary          # a PASS here licenses no cross-domain transfer


def test_an_association_only_domain_still_reaches_a_verdict_but_a_weaker_one():
    """No lag floor means no precedence claim — the gate still adjudicates the association."""
    declaration = _declaration(lag_policy="none", declared_floor_frames=None,
                               declared_floor_basis=None)
    result = da.run_domain_gate(_record(7, 0.9), declaration, _protocol())
    assert result["verdict"] == "PASS"
    assert result["train"]["claim_boundary"] == da.ASSOCIATION_CLAIM
    assert "declared claim level 'association'" in result["claim_boundary"]
    assert any("must not be described as precedence" in w
               for w in result["train"]["warnings"])
