"""Phase 4C, T4C.3: cross-scale lagged dependency.

The acceptance criterion is a pair: recover an injected cross-scale coupling in a synthetic
cascade, and report **null** on a phase-randomised version of that same field. Both are here,
run through the whole path - decompose, signature, sweep, surrogate, FDR - because it is the
path that has to be right, not the estimator alone.

The rest of the file is the machinery that had to be got right before the pair would work:
the shift null must exclude the simultaneous alignment as well as the tested one, the support
floor must refuse to invent an advection speed, and a sweep that cannot possibly reject
anything must say so rather than returning a clean-looking zero.
"""

import math

import numpy as np
import pytest

from src.analysis_engine import cross_scale as cs
from src.analysis_engine.scale_signature import scale_signature
from src.analysis_engine.surrogate_null import sequence_surrogate
from src.core.errors import InvalidParameterError
from src.synthetic_generator.cascade import build_cascade, cascade_truth
from src.transform_engine.coefficient_field import decompose_sequence

CADENCE = 3600.0


def _signature(sequence, levels=3):
    return scale_signature(decompose_sequence(
        sequence, "swt", {"levels": levels, "wavelet": "db2"}, keep_native=False))


def _gate_result(protocol, *, significant=True, powered=True, floor=True):
    rows = []
    for source in range(1, protocol.n_scales + 1):
        for target in range(1, protocol.n_scales + 1):
            if source == target:
                continue
            for lag in protocol.lags:
                rows.append({
                    "label": "%d->%d@%d" % (source, target, lag),
                    "significant": bool(significant and source == 1 and target == 2
                                        and lag == protocol.lags[0]),
                    "q_value": 0.01 if significant else 1.0,
                    "excess_nats": 0.2 if significant else 0.0,
                })
    return {"estimator": protocol.estimator, "measure": protocol.measure,
            "lags_frames": list(protocol.lags), "n_tests": protocol.family_size,
            "bins": protocol.bins, "alpha": protocol.alpha,
            "n_surrogates_requested": protocol.n_surrogates,
            "correction": {"method": protocol.correction},
            "protocol_fingerprint": protocol.fingerprint(),
            "power": {"can_reject_after_correction": powered},
            "support_floor": {"enforced": floor}, "results": rows}


def test_gate_protocol_refuses_an_underpowered_or_leaky_design():
    with pytest.raises(InvalidParameterError, match="cannot produce a significant result"):
        cs.GateProtocol("underpowered", 3, (1, 2), 4000,
                        embargo_frames=2, n_surrogates=99).validate()
    with pytest.raises(InvalidParameterError, match="longest tested lag"):
        cs.GateProtocol("leaky", 3, (1, 2, 3), 4000,
                        embargo_frames=2, n_surrogates=4999).validate()


def test_gate_protocol_refuses_partitions_too_short_for_the_estimator():
    with pytest.raises(InvalidParameterError, match="both independent partitions"):
        cs.GateProtocol("short", 3, (1, 2), 2000,
                        embargo_frames=2, n_surrogates=4999).validate()


def test_gate_protocol_fingerprint_is_stable_and_sensitive():
    protocol = cs.GateProtocol("nz", 3, (1, 2), 4000,
                               embargo_frames=2, n_surrogates=4999)
    assert protocol.fingerprint() == protocol.fingerprint()
    changed = cs.GateProtocol("nz", 3, (1, 2), 4000,
                              embargo_frames=2, n_surrogates=5000)
    assert protocol.fingerprint() != changed.fingerprint()
    assert cs.GateProtocol.from_mapping(protocol.to_mapping()) == protocol


def test_replication_gate_distinguishes_pass_fail_and_invalid():
    protocol = cs.GateProtocol("nz", 3, (1, 2), 4000,
                               embargo_frames=2, n_surrogates=4999)
    positive = _gate_result(protocol)
    null = _gate_result(protocol, significant=False)
    assert cs.evaluate_replication_gate(positive, positive, protocol)["verdict"] == "PASS"
    assert cs.evaluate_replication_gate(positive, null, protocol)["verdict"] == "FAIL"
    invalid = _gate_result(protocol, powered=False)
    assert cs.evaluate_replication_gate(positive, invalid, protocol)["verdict"] == "INVALID"


@pytest.fixture(scope="module")
def cascade():
    return build_cascade(n=96, n_frames=192, lag=3, seed=5)


@pytest.fixture(scope="module")
def cascade_signature(cascade):
    return _signature(cascade)


@pytest.fixture(scope="module")
def randomised_signature(cascade):
    return _signature(sequence_surrogate(cascade, "spatiotemporal_phase", seed=9))


# ============================================================ the estimators

def test_mutual_information_is_zero_for_independent_series():
    rng = np.random.default_rng(1)
    value = cs.mutual_information(rng.standard_normal(2000), rng.standard_normal(2000))
    assert value < 0.02


def test_mutual_information_saturates_for_identical_series():
    """With `b` equiprobable bins, `I(X ; X)` is the marginal entropy, `log b`."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    assert cs.mutual_information(x, x, bins=6) == pytest.approx(math.log(6), rel=0.02)


def test_transfer_entropy_is_directional_where_mutual_information_is_not():
    """A driven pair: `y_t` is `x_{t-2}` plus noise, and nothing runs the other way."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3000)
    y = np.roll(x, 2) + 0.3 * rng.standard_normal(3000)

    forward = cs.transfer_entropy(x, y, lag=2, bins=5)
    backward = cs.transfer_entropy(y, x, lag=2, bins=5)
    assert forward > 4 * backward

    # Lagged mutual information sees the relationship in both directions, which is why it is
    # the more sensitive and the more misleading of the two.
    assert cs.lagged_mutual_information(x, y, lag=2, bins=5) > 0.3


def test_a_zero_lag_is_refused():
    for estimator in (cs.transfer_entropy, cs.lagged_mutual_information):
        with pytest.raises(InvalidParameterError):
            estimator(np.zeros(50), np.zeros(50), lag=0)


def test_decorrelation_time_of_white_noise_is_one_frame():
    rng = np.random.default_rng(5)
    assert cs.decorrelation_frames(rng.standard_normal(1000)) == 1


# ============================================================ the shift null

def test_admissible_shifts_exclude_both_the_tested_and_the_simultaneous_alignment():
    """The simultaneous alignment is the one that is easy to forget.

    Rolling the source by `-lag` puts the two series at effective lag zero - a real alignment
    of the data, not a shuffle. On the synthetic cascade that single shift produced the
    largest transfer entropy in the entire ensemble, above the observed value, and capped the
    achievable p-value at about 0.005 no matter how many surrogates were drawn.
    """
    shifts = set(cs.admissible_shifts(100, lag=3, theiler=2).tolist())
    for excluded in (1, 99, 97, 98, 96):  # near 0, and near -3 == 97
        assert excluded not in shifts
    assert 50 in shifts


def test_a_record_too_short_for_any_admissible_shift_is_refused():
    with pytest.raises(cs.CrossScaleError) as excinfo:
        cs.admissible_shifts(8, lag=2, theiler=4)
    assert "longer record" in str(excinfo.value)


# ============================================================ the support floor (R4)

def test_the_support_floor_refuses_to_invent_an_advection_speed(cascade_signature):
    floors = cs.support_floor(cascade_signature, CADENCE)
    assert floors["enforced"] is False
    assert all(record["floor_frames"] == 1 for record in floors["floors"])
    assert any("no advection speed" in w for w in floors["warnings"])
    assert "temporal support is zero" in floors["temporal_support_note"]


def test_a_declared_advection_speed_produces_a_scale_dependent_floor(cascade_signature):
    """The floor uses db2's complete cascade support, not the dyadic scale label."""
    floors = cs.support_floor(cascade_signature, CADENCE, advection_speed_m_s=10.0)
    assert floors["enforced"] is True
    frames = [record["floor_frames"] for record in floors["floors"]]
    assert frames == sorted(frames), "a coarser scale cannot have a shorter floor"
    assert frames[-1] > frames[0]
    assert floors["floors"][-1]["spatial_support_px"] == 22
    assert floors["floors"][-1]["crossing_time_s"] == pytest.approx(22 * 25000.0 / 10.0)


def test_latlon_support_floor_converts_degrees_and_uses_conservative_physical_axis():
    """D53: GridSpec.dx=0.25 is degrees, not the 0.25 metres the old path assumed."""
    from src.core.channel_series import ChannelGeometrySpec
    from src.physical_core.grid import GridSpec

    grid = GridSpec.latlon(
        (161, 161), lat0=-20.0, dlat=-0.25, lon0=140.0, dlon=0.25)
    signature = ChannelGeometrySpec(
        channels=[1, 2, 3], support_parent_px=[4, 10, 22],
        provenance={"grid": grid.to_provenance()})
    floors = cs.support_floor(signature, 21600.0, advection_speed_m_s=10.0)
    coarse = floors["floors"][-1]
    assert grid.to_provenance()["dx"] == 0.25
    assert coarse["physical_spacing_m_per_parent_px"] > 27_000
    assert coarse["floor_frames"] == 3
    assert "converted to metres" in coarse["physical_spacing_basis"]


def test_lags_below_the_floor_are_excluded_by_name(cascade_signature):
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[1, 2], cadence_seconds=CADENCE,
        advection_speed_m_s=5.0, n_surrogates=19)
    assert result["n_tests"] == 0
    assert result["excluded"]
    assert any("support floor" in record["reason"] for record in result["excluded"])
    assert any("same air seen twice" in record["reason"] for record in result["excluded"])


# ============================================================ the acceptance criterion

def test_the_injected_cascade_is_recovered_at_the_right_lag_and_direction(cascade_signature):
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[2, 3, 4], cadence_seconds=CADENCE,
        measure="energy_density", bins=4, n_surrogates=1999)

    assert result["power"]["can_reject_after_correction"] is True
    significant = [row for row in result["results"] if row["significant"]]
    assert significant, "the coupling is present by construction and must be found"

    truth = cascade_truth(lag=3)
    for row in significant:
        assert row["lag_frames"] == truth["injected_lag_frames"]
        assert int(row["source_scale"]) < int(row["target_scale"]), "fine precedes coarse"
        assert row["excess_nats"] > 0.1
        assert row["q_value"] < 0.05


def test_the_phase_randomised_cascade_reports_nothing(randomised_signature):
    """Same spectra, same autocorrelations, no alignment - so there must be nothing left."""
    result = cs.cross_scale_dependency(
        randomised_signature, lags=[2, 3, 4], cadence_seconds=CADENCE,
        measure="energy_density", bins=4, n_surrogates=1999)
    assert result["power"]["can_reject_after_correction"] is True
    assert result["n_significant"] == 0


# ============================================================ honesty of the sweep

def test_an_underpowered_sweep_says_so_instead_of_reporting_a_clean_negative(
        cascade_signature):
    """99 surrogates floor the p-value at 0.01, and 18 tests under Benjamini-Yekutieli need
    a raw p far below that. Without this check the result is indistinguishable from a real
    null."""
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[2, 3, 4], cadence_seconds=CADENCE, n_surrogates=99)
    assert result["n_significant"] == 0
    assert result["power"]["can_reject_after_correction"] is False
    assert result["power"]["surrogates_required"] > 99
    assert any("cannot produce a significant result" in w for w in result["warnings"])


def test_every_test_carries_its_lag_floor_and_theiler_window(cascade_signature):
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[3], cadence_seconds=CADENCE, n_surrogates=19)
    assert result["results"]
    for row in result["results"]:
        assert row["lag_seconds"] == 3 * CADENCE
        assert row["support_floor_frames"] >= 1
        assert row["theiler_window_frames"] >= 1
        assert row["wrap"] is True


def test_the_result_states_that_it_is_not_causal(cascade_signature):
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[3], cadence_seconds=CADENCE, n_surrogates=19)
    assert "precursor relationship, not a causal one" in result["causality_caveat"]
    assert "circular shift" in result["null_model"]


def test_an_unknown_estimator_is_refused(cascade_signature):
    with pytest.raises(InvalidParameterError):
        cs.cross_scale_dependency(cascade_signature, lags=[3], cadence_seconds=CADENCE,
                                  estimator="granger")


def test_a_scale_with_no_interior_is_excluded_rather_than_estimated():
    """A NaN column would otherwise propagate into every pair it touches."""
    from src.physical_core.grid import GridSpec
    from src.physical_core.field import PhysicalField
    from src.physical_core.sequence import FieldSequence
    import torch

    grid = GridSpec.cartesian((64, 64), dy_m=25000.0, dx_m=25000.0)
    generator = torch.Generator().manual_seed(9)
    fields = [PhysicalField(torch.randn(64, 64, dtype=torch.float64, generator=generator),
                            grid=grid) for _ in range(40)]
    sequence = FieldSequence(fields, np.arange(40, dtype=np.float64) * CADENCE)
    signature = scale_signature(decompose_sequence(sequence, "dtcwt", {"levels": 4},
                                                   keep_native=False))

    result = cs.cross_scale_dependency(signature, lags=[2], cadence_seconds=CADENCE,
                                       n_surrogates=19)
    assert any("no valid interior" in record["reason"] for record in result["excluded"])
    assert all("3" not in (row["source_scale"], row["target_scale"])
               for row in result["results"])


def test_a_long_lag_relative_to_the_record_warns_about_dilution(cascade_signature):
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[40], cadence_seconds=CADENCE, n_surrogates=19)
    assert any("wraps" in w for w in result["warnings"])


# ============================================================ the pipeline action

@pytest.fixture
def store(tmp_path, monkeypatch):
    from src.artifact_store import store as artifact_store
    made = artifact_store.ArtifactStore(str(tmp_path / "artifacts"))
    monkeypatch.setattr(artifact_store, "_DEFAULT", made)
    return made


def _stored_field(store, cascade):
    field = decompose_sequence(cascade, "swt", {"levels": 3, "wavelet": "db2"},
                               keep_native=False)
    return store.put(field, name="cf").ref


def test_the_action_runs_end_to_end_and_summarises_small(store, cascade):
    import json
    import torch
    from src.experiment_engine import actions

    reference = _stored_field(store, cascade)
    result = actions.execute("cross_scale_dependency", {
        "coefficients": reference, "lags": [2, 3], "cadence_seconds": CADENCE,
        "n_surrogates": 99}, torch.device("cpu"))

    assert result["n_tests"] == 12
    assert "precursor relationship" in result["causality_caveat"]

    summary = actions.summarise("cross_scale_dependency", result, {})
    assert len(json.dumps(summary)) < 4096
    assert summary["can_reject_after_correction"] is False
    assert summary["support_floor_enforced"] is False


def test_the_action_refuses_a_dereferenced_payload(store):
    import numpy as np
    import torch
    from src.core.errors import InvalidParameterError
    from src.experiment_engine import actions

    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("cross_scale_dependency",
                        {"coefficients": np.zeros((2, 3, 3, 8, 8))}, torch.device("cpu"))
    assert "{step.coefficients_ref}" in str(excinfo.value)


def test_the_action_refuses_a_zero_lag(store, cascade):
    import torch
    from src.core.errors import InvalidParameterError
    from src.experiment_engine import actions

    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("cross_scale_dependency",
                        {"coefficients": _stored_field(store, cascade), "lags": [0]},
                        torch.device("cpu"))
    assert "simultaneous association" in str(excinfo.value)


def test_a_sweeps_surrogate_ensemble_can_be_recovered_from_its_seed(cascade_signature):
    """T4C.5i step 7: the minimum detectable effect is read off the ensemble the sweep used.

    `cross_scale_dependency` keeps only the ensemble's summary, because carrying 4,999 numbers
    per test through every receipt would multiply its size by two orders of magnitude for a
    quantity nothing read. The power audit does read it -- the detection threshold is an order
    statistic, which no summary recovers -- so it must be able to reconstruct the *identical*
    ensemble rather than draw a fresh one and call the result the study's decision boundary.
    """
    result = cs.cross_scale_dependency(
        cascade_signature, lags=[3], cadence_seconds=CADENCE, measure="energy_density",
        bins=4, n_surrogates=99, seed=515)
    row = next(r for r in result["results"] if r["label"] == "1->2@3")
    channels = [str(value) for value in cascade_signature.channels]
    matrix = cascade_signature.to_matrix("energy_density")
    source_index = channels.index("1")

    recovered = cs.shift_null_ensemble(
        matrix[:, source_index], matrix[:, channels.index("2")], 3, bins=4, n_surrogates=99,
        seed=cs.surrogate_seed(515, 3, source_index),
        theiler=int(row["theiler_window_frames"]))
    assert recovered.size == row["n_surrogates"]
    assert float(recovered.mean()) == pytest.approx(row["surrogate_mean_nats"], rel=1e-12)

    # A different test's seed is a different null, and must not be mistaken for this one's.
    other = cs.shift_null_ensemble(
        matrix[:, source_index], matrix[:, channels.index("2")], 3, bins=4, n_surrogates=99,
        seed=cs.surrogate_seed(515, 3, source_index + 1),
        theiler=int(row["theiler_window_frames"]))
    assert float(other.mean()) != float(recovered.mean())


def test_the_surrogate_seed_separates_lags_from_source_indices():
    """A prime multiplier, so a neighbouring lag cannot collide with a neighbouring channel."""
    assert cs.surrogate_seed(0, 1, 0) != cs.surrogate_seed(0, 0, 1)
    assert len({cs.surrogate_seed(20260821, lag, index)
                for lag in range(3, 9) for index in range(3)}) == 18
