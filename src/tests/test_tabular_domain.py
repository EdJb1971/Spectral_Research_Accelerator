"""TG0.2: a non-atmospheric domain through the unmodified falsification layer.

Phase G0's question is whether the accepted cross-channel machinery — surrogate ensemble, BY
correction, power check, excluded-test accounting — still works when every atmospheric thing is
removed: no grid, no metric, no transform, no advection, no annual cycle, no physics at all.

The two tests that answer it are
`test_the_unmodified_sweep_recovers_a_planted_coupling_in_a_non_physical_domain` and
`test_independent_autocorrelated_channels_produce_no_significant_relationship`. They are a
pair, and the second is the more important of the two: a detector that finds a planted signal
but also finds signal in autocorrelated noise has found nothing. The record is deliberately
AR(1), because the failure this platform has already measured once (T4C.3: a linear lag against
a circularly stationary null falsely rejected 20 of 20 AR(1) records) is exactly what a naive
lag test does to autocorrelated data.

Everything else here is the refusal surface, which for a cross-domain system is most of the
value: rule R21 must stop a precedence claim from a domain that cannot justify a lag floor,
and rule R17 must stop an adapter author from declaring a domain that breaks nothing.
"""

import numpy as np
import pytest

from src.analysis_engine import domain_analysis as da
from src.core.domain import (AxisSpec, DomainDeclaration, KNOWN_VIOLATIONS,
                             PrecedenceNotAdmissibleError)
from src.core.errors import InvalidParameterError, MissingParameterError, UnknownNameError
from src.data_layer.tabular_source import (VALUE_MEASURE, read_tabular_channels,
                                           write_tabular_channels)

CADENCE = 60.0
N_FRAMES = 800
PLANTED_LAG = 3
LICENCE = "CC-BY-4.0 (fixture; no real archive was accessed)"
VIOLATIONS = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle")


def _ar1(rng, n=N_FRAMES, rho=0.6):
    """Autocorrelated noise: the thing a naive lag test mistakes for a relationship."""
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def _write(tmp_path, columns, name="record.csv"):
    path = str(tmp_path / name)
    write_tabular_channels(path, np.arange(N_FRAMES, dtype=float) * (CADENCE / 60.0),
                           columns, time_column="minute")
    return path


def _planted(tmp_path, seed=7, coupling=0.9):
    """`alpha` at t sets `gamma` at t+3. `beta` is an innocent bystander."""
    rng = np.random.default_rng(seed)
    alpha, beta, gamma_noise = _ar1(rng), _ar1(rng), _ar1(rng)
    gamma = np.empty(N_FRAMES)
    gamma[:PLANTED_LAG] = gamma_noise[:PLANTED_LAG]
    gamma[PLANTED_LAG:] = (coupling * alpha[:-PLANTED_LAG]
                           + (1.0 - coupling) * gamma_noise[PLANTED_LAG:])
    return _write(tmp_path, {"alpha": alpha, "beta": beta, "gamma": gamma})


def _read(path, **overrides):
    kwargs = dict(domain_name="synthetic_instrument_log",
                  description="Three unitless channels on a regular clock. No physics.",
                  licence=LICENCE, violations=VIOLATIONS, time_column="minute",
                  time_units="min", lag_policy="declared", declared_floor_frames=2,
                  declared_floor_basis="two reporting intervals: the logger writes a value "
                                       "every interval and a change cannot be observed "
                                       "before the following one has been written")
    kwargs.update(overrides)
    return read_tabular_channels(path, **kwargs)


# ------------------------------------------------------------------ the adapter reads

def test_a_tabular_record_round_trips_with_provenance_and_cadence(tmp_path):
    path = _planted(tmp_path)
    series, declaration = _read(path)

    assert list(series.channels) == ["alpha", "beta", "gamma"]
    assert series.to_matrix(VALUE_MEASURE).shape == (N_FRAMES, 3)
    assert series.times_seconds[1] - series.times_seconds[0] == CADENCE
    assert declaration.provenance["cadence_seconds"] == CADENCE
    assert len(declaration.provenance["content_sha256"]) == 64
    assert declaration.provenance["network_access"].startswith("none")
    # An instantaneous reading depends on exactly one sample. That is a fact about the
    # instrument, not a placeholder, and it is what the lag floor is computed from.
    assert all(record["support_parent_px"] == 1.0 for record in series.channel_records)


def test_an_aggregated_channel_declares_its_window(tmp_path):
    path = _planted(tmp_path)
    series, _ = _read(path, support_parent_px={"beta": 10},
                      violations=VIOLATIONS + ("aggregated_values",))
    supports = {record["scale"]: record["support_parent_px"]
                for record in series.channel_records}
    assert supports == {"alpha": 1.0, "beta": 10.0, "gamma": 1.0}


# --------------------------------------------------- the Phase G0 acceptance criterion

def test_the_unmodified_sweep_recovers_a_planted_coupling_in_a_non_physical_domain(tmp_path):
    """No grid, no transform, no advection, no cycle — and the accepted gate still works."""
    series, declaration = _read(_planted(tmp_path))
    result = da.analyse_precedence(series, declaration, lags=(2, 3),
                                   cadence_seconds=CADENCE, measure=VALUE_MEASURE,
                                   n_surrogates=999, bins=4, seed=20260823)

    assert result["power"]["can_reject_after_correction"] is True
    assert result["n_tests"] == 12                      # 6 ordered pairs x 2 lags
    assert result["claim_boundary"] == da.PRECEDENCE_CLAIM
    assert result["applied_lag_floor"] == {
        "policy": "declared", "frames": 2,
        "basis": declaration.declared_floor_basis}

    significant = {row["label"]: row for row in result["results"] if row["significant"]}
    assert "alpha->gamma@3" in significant, "the planted relationship was not recovered"
    assert not any(label.startswith("beta->") or label.endswith("->beta")
                   for label in significant), "the bystander channel was implicated"

    # The planted lag must dominate. Lag 2 can also register, because alpha is autocorrelated
    # and therefore partly carries its own value at t-3 - reporting that as an equal finding
    # would be the artefact, so the discriminating check is the excess, not the flag.
    assert (significant["alpha->gamma@3"]["excess_nats"]
            > 4.0 * significant.get("alpha->gamma@2", {"excess_nats": 0.0})["excess_nats"])


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_independent_autocorrelated_channels_produce_no_significant_relationship(tmp_path,
                                                                                 seed):
    """The null returns null. Without this the test above is not evidence of anything."""
    rng = np.random.default_rng(seed)
    path = _write(tmp_path, {name: _ar1(rng) for name in ("alpha", "beta", "gamma")},
                  name="null_%d.csv" % seed)
    series, declaration = _read(path)
    result = da.analyse_precedence(series, declaration, lags=(2, 3),
                                   cadence_seconds=CADENCE, measure=VALUE_MEASURE,
                                   n_surrogates=999, bins=4, seed=20260823)
    assert result["power"]["can_reject_after_correction"] is True
    assert result["n_significant"] == 0


# --------------------------------------------------------------- rule R21 at the boundary

def test_a_domain_with_no_lag_floor_is_refused_a_precedence_claim(tmp_path):
    series, declaration = _read(_planted(tmp_path), lag_policy="none",
                                declared_floor_frames=None, declared_floor_basis=None)
    assert declaration.precedence_admissible is False
    with pytest.raises(PrecedenceNotAdmissibleError) as caught:
        da.analyse_precedence(series, declaration, lags=(2, 3), cadence_seconds=CADENCE,
                              measure=VALUE_MEASURE)
    message = str(caught.value)
    assert "synthetic_instrument_log" in message
    assert "no_propagation_speed" in message
    assert "lag_policy='declared'" in message      # the refusal names the way forward


def test_association_is_available_where_precedence_is_not_and_says_so(tmp_path):
    series, declaration = _read(_planted(tmp_path), lag_policy="none",
                                declared_floor_frames=None, declared_floor_basis=None)
    result = da.association_only(series, declaration, lags=(3,), cadence_seconds=CADENCE,
                                 measure=VALUE_MEASURE, n_surrogates=499, bins=4,
                                 seed=20260823)
    assert result["claim_boundary"] == da.ASSOCIATION_CLAIM
    assert result["applied_lag_floor"]["frames"] is None
    assert any("must not be described as precedence" in w for w in result["warnings"])


def test_association_only_is_refused_when_the_domain_can_justify_a_floor(tmp_path):
    """Downgrading the claim while skipping the floor is weaker on both counts."""
    series, declaration = _read(_planted(tmp_path))
    with pytest.raises(InvalidParameterError, match="analyse_precedence"):
        da.association_only(series, declaration, lags=(3,), cadence_seconds=CADENCE,
                            measure=VALUE_MEASURE)


def test_lags_below_the_declared_floor_are_refused_rather_than_dropped(tmp_path):
    series, declaration = _read(_planted(tmp_path))
    with pytest.raises(InvalidParameterError) as caught:
        da.analyse_precedence(series, declaration, lags=(1, 2, 3), cadence_seconds=CADENCE,
                              measure=VALUE_MEASURE)
    assert "reporting intervals" in str(caught.value)      # the basis, not just the number


def test_two_floors_from_two_bases_are_refused(tmp_path):
    series, declaration = _read(_planted(tmp_path))
    with pytest.raises(InvalidParameterError, match="Two floors"):
        da.analyse_precedence(series, declaration, lags=(3,), cadence_seconds=CADENCE,
                              measure=VALUE_MEASURE, advection_speed_m_s=10.0)


def test_an_advective_domain_still_requires_a_declared_speed(tmp_path):
    series, declaration = _read(_planted(tmp_path), lag_policy="advective",
                                declared_floor_frames=None, declared_floor_basis=None,
                                violations=("no_natural_cycle",))
    with pytest.raises(InvalidParameterError, match="no default"):
        da.analyse_precedence(series, declaration, lags=(3,), cadence_seconds=CADENCE,
                              measure=VALUE_MEASURE)


def _without_footprints(series):
    """The same record with every channel's parent-axis footprint removed."""
    from src.core.channel_series import ChannelSeries

    return ChannelSeries(channels=list(series.channels),
                         times_seconds=series.times_seconds,
                         measures={VALUE_MEASURE: series.to_matrix(VALUE_MEASURE)})


def test_a_missing_channel_footprint_is_refused_in_domain_language(tmp_path):
    """The wavelet-flavoured message must not reach a non-wavelet adapter.

    Asked of an *advective* domain since TG1.3, because that is the policy which reads the
    footprint. The refusal is unchanged; what changed is which domains are asked for it.
    """
    series, declaration = _read(_planted(tmp_path), lag_policy="advective",
                                declared_floor_frames=None, declared_floor_basis=None,
                                violations=("no_natural_cycle",))
    with pytest.raises(InvalidParameterError) as caught:
        da.analyse_precedence(_without_footprints(series), declaration, lags=(3,),
                              cadence_seconds=CADENCE, measure=VALUE_MEASURE,
                              advection_speed_m_s=10.0)
    message = str(caught.value)
    assert "synthetic_instrument_log" in message
    assert "instantaneous reading" in message
    assert "parent-grid filter support" not in message


def test_a_domain_whose_floor_cannot_read_a_footprint_is_no_longer_asked_for_one(tmp_path):
    """The finding TG1.3 turned into behaviour.

    A logger that reports every two minutes has a floor of two frames because of the logger,
    not because of a filter. `support_parent_px` cannot enter that floor by any route - and
    until TG1.3 the sweep refused to run without it, because `cross_scale_dependency` called
    `support_floor` unconditionally whatever the domain had declared. The number a
    non-wavelet adapter had to supply was one it could only have invented, and inventing it
    changed nothing.
    """
    series, declaration = _read(_planted(tmp_path))
    result = da.analyse_precedence(_without_footprints(series), declaration, lags=(3,),
                                   cadence_seconds=CADENCE, measure=VALUE_MEASURE)
    assert result["applied_lag_floor"] == {
        "policy": "declared", "frames": 2,
        "basis": declaration.declared_floor_basis}
    # And the floor the sweep applied is the domain's, not the advective one it used to
    # report: every test record now carries 2, where it used to carry 1 and cite rule R4.
    assert {row["support_floor_frames"] for row in result["results"]} == {2}
    assert result["support_floor"]["policy"] == "declared"


# ------------------------------------------------------------- rule R17 and declarations

def test_a_domain_that_breaks_nothing_is_refused(tmp_path):
    with pytest.raises(InvalidParameterError, match="second variable, not a second domain"):
        _read(_planted(tmp_path), violations=(), lag_policy="none",
              declared_floor_frames=None, declared_floor_basis=None)


def test_an_unknown_violation_name_lists_the_vocabulary(tmp_path):
    with pytest.raises(UnknownNameError) as caught:
        _read(_planted(tmp_path), violations=("no_seasons",))
    assert "no_natural_cycle" in str(caught.value)


def test_a_declared_floor_without_a_basis_is_refused(tmp_path):
    with pytest.raises(InvalidParameterError, match="stated basis"):
        _read(_planted(tmp_path), declared_floor_basis="  ")


def test_an_advective_policy_contradicting_its_own_violations_is_refused():
    with pytest.raises(InvalidParameterError, match="consistent with the declared"):
        DomainDeclaration(name="d", description="", licence=LICENCE,
                          axes=(AxisSpec("t", "time"),),
                          violations=("no_propagation_speed",), lag_policy="advective")


def test_exactly_one_time_axis_is_required():
    with pytest.raises(InvalidParameterError, match="exactly one axis with role 'time'"):
        DomainDeclaration(name="d", description="", licence=LICENCE,
                          axes=(AxisSpec("t", "time"), AxisSpec("u", "time")),
                          violations=("no_physical_metric",))


def test_a_licence_is_not_optional():
    with pytest.raises(InvalidParameterError, match="licence or terms of use"):
        DomainDeclaration(name="d", description="", licence="",
                          axes=(AxisSpec("t", "time"),), violations=("no_physical_metric",))


def test_an_unknown_axis_role_is_refused():
    with pytest.raises(UnknownNameError):
        AxisSpec("station", "place")


def test_the_declaration_travels_with_every_result(tmp_path):
    series, declaration = _read(_planted(tmp_path), lag_policy="none",
                                declared_floor_frames=None, declared_floor_basis=None)
    result = da.association_only(series, declaration, lags=(3,), cadence_seconds=CADENCE,
                                 measure=VALUE_MEASURE, n_surrogates=499, bins=4,
                                 seed=20260823)
    described = result["domain"]
    assert described["licence"] == LICENCE
    assert set(described["violations"]) == set(VIOLATIONS)
    assert all(described["violations"][name] == KNOWN_VIOLATIONS[name]
               for name in VIOLATIONS)
    assert described["precedence_admissible"] is False


# ---------------------------------------------------------------- adapter refusal surface

def test_an_irregular_clock_is_refused_unless_declared(tmp_path):
    path = str(tmp_path / "irregular.csv")
    times = np.arange(20, dtype=float)
    times[10:] += 5.0
    write_tabular_channels(path, times, {"a": np.arange(20.0), "b": np.arange(20.0)},
                           time_column="minute")
    with pytest.raises(InvalidParameterError, match="irregular_sampling"):
        read_tabular_channels(path, domain_name="d", description="", licence=LICENCE,
                              violations=VIOLATIONS, time_column="minute", time_units="min")
    series, _ = read_tabular_channels(
        path, domain_name="d", description="", licence=LICENCE,
        violations=VIOLATIONS + ("irregular_sampling",), time_column="minute",
        time_units="min")
    assert series.n_times == 20


def test_a_non_increasing_clock_is_refused_rather_than_sorted(tmp_path):
    path = str(tmp_path / "unsorted.csv")
    times = np.arange(20, dtype=float)
    times[5], times[6] = times[6], times[5]
    write_tabular_channels(path, times, {"a": np.arange(20.0), "b": np.arange(20.0)},
                           time_column="minute")
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        read_tabular_channels(path, domain_name="d", description="", licence=LICENCE,
                              violations=VIOLATIONS, time_column="minute", time_units="min")


def test_non_finite_values_are_refused_rather_than_imputed(tmp_path):
    path = str(tmp_path / "gappy.csv")
    a = np.arange(20.0)
    a[7] = np.nan
    write_tabular_channels(path, np.arange(20.0), {"a": a, "b": np.arange(20.0)},
                           time_column="minute")
    with pytest.raises(InvalidParameterError, match="finite values"):
        read_tabular_channels(path, domain_name="d", description="", licence=LICENCE,
                              violations=VIOLATIONS, time_column="minute", time_units="min")


def test_a_cadence_that_contradicts_the_caller_is_refused(tmp_path):
    with pytest.raises(InvalidParameterError, match="cadence actually present"):
        _read(_planted(tmp_path), expected_cadence_seconds=3600.0)


def test_a_missing_clock_column_names_the_columns_present(tmp_path):
    with pytest.raises(MissingParameterError) as caught:
        _read(_planted(tmp_path), time_column="timestamp")
    assert "alpha" in str(caught.value)


def test_unknown_time_units_are_refused(tmp_path):
    with pytest.raises(InvalidParameterError, match="time_units"):
        _read(_planted(tmp_path), time_units="fortnights")


def test_the_adapter_never_fetches(tmp_path):
    with pytest.raises(InvalidParameterError, match="never fetches"):
        _read(str(tmp_path / "absent.csv"))
