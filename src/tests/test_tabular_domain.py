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


# ---------------------------------------------------------------------------------- TG8.4
#
# Reading a record **against a domain that already exists**, rather than against one the reader
# invents from its own arguments. Everything below concerns the two new entry points and the
# three checks that had no enforcement before the ingestion seam existed.

from src.core.domain import DOMAIN_DECLARATIONS                            # noqa: E402
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain         # noqa: E402
from src.core.registry import restore, snapshot                            # noqa: E402
from src.core.translation import DOMAIN_GLOSSARIES                         # noqa: E402
from src.data_layer.tabular_source import (                                # noqa: E402
    MAX_CELLS, MAX_ROWS, _assert_within_caps, assert_domain_admits_channel_table, clock_facts,
    inspect_delimited, read_channels_for_domain, required_violations, unsatisfiable_axes)

REGULAR_CSV = "t,bid,ask\n0,1,2\n1,1.5,2.5\n2,2,3\n3,2.5,3.5\n"
IRREGULAR_CSV = "t,bid,ask\n0,1,2\n1,1.5,2.5\n5,2,3\n9,2.5,3.5\n"
BACKWARDS_CSV = "t,bid,ask\n0,1,2\n5,1.5,2.5\n3,2,3\n"


@pytest.fixture()
def domains():
    """Snapshot the three domain registries, so a test may onboard and leave no trace."""
    from src.core.builtin_domains import register_builtin_domains

    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    try:
        yield
    finally:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


def _channel_domain(name, violations, **kwargs):
    """An onboarded domain shaped like a channel table: one clock, one unordered category."""
    from src.core.builtin_glossaries import ORDER_BOOK_PHRASES

    declaration = DomainDeclaration(
        name=name,
        description="A channel-table domain declared by the tabular tests.",
        axes=(AxisSpec(name="t", role="time", units="s"),
              AxisSpec(name="channel", role="category", ordered=False)),
        licence=LICENCE, violations=violations, lag_policy="none", **kwargs)
    return onboard_domain(declaration, ORDER_BOOK_PHRASES, geometry=None,
                          onboarded_by="src.tests.test_tabular_domain")


# ------------------------------------------------------------------ observation, not decision


def test_clock_facts_describe_the_clock_without_consulting_any_domain():
    regular = clock_facts(np.array([0.0, 1.0, 2.0, 3.0]))
    assert regular["strictly_increasing"] and regular["regular"]
    assert regular["cadence_seconds"] == 1.0

    ragged = clock_facts(np.array([0.0, 1.0, 5.0, 9.0]))
    assert ragged["strictly_increasing"] and not ragged["regular"]
    assert ragged["cadence_seconds"] is None
    assert ragged["interval_seconds_min"] == 1.0 and ragged["interval_seconds_max"] == 4.0

    assert clock_facts(np.array([0.0, 5.0, 3.0]))["strictly_increasing"] is False


def test_an_irregular_clock_creates_an_obligation_rather_than_satisfying_one():
    """TG8.4's rule: detection may create a required declaration, never satisfy one."""
    facts = clock_facts(np.array([0.0, 1.0, 5.0, 9.0]))
    assert required_violations(facts) == ["irregular_sampling"]
    assert required_violations(clock_facts(np.array([0.0, 1.0, 2.0]))) == []


def test_inspect_reports_candidates_and_chooses_no_clock_column():
    report = inspect_delimited("t,u,v\n0,3,9\n1,5,8\n2,7,7\n", source_name="two.csv")
    # `t` and `u` both increase strictly; `v` decreases. A file with two candidates is one only
    # its author can disambiguate, so both are offered.
    assert report["candidate_time_columns"] == ["t", "u"]
    assert report["n_rows"] == 3


def test_inspect_reports_a_non_monotonic_clock_as_unfixable_by_any_declaration():
    report = inspect_delimited(BACKWARDS_CSV, source_name="back.csv")
    assert report["readable"] is False
    assert "not strictly increasing" in report["refused_because"]
    assert all(not row["admits"] for row in report["domains"])


def test_inspect_names_the_aggregate_obligation_it_cannot_detect():
    """No column says it is a window aggregate, so this is stated rather than inferred."""
    report = inspect_delimited(REGULAR_CSV, source_name="reg.csv")
    assert "aggregated_values" in report["aggregate_note"]
    assert "aggregated_values" not in report["required_violations"]


def test_inspect_digest_matches_the_read_digest(domains):
    report = inspect_delimited(REGULAR_CSV, source_name="reg.csv")
    series, _ = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                         domain="order_book", time_column="t")
    assert report["content_sha256"] == series.provenance["content_sha256"]


# ------------------------------------------------------------------ declared axes (E14)


def test_a_domain_declaring_space_or_level_cannot_read_a_channel_table(domains):
    from src.core.domain import declaration_for

    assert unsatisfiable_axes(declaration_for("order_book")) == []
    unsatisfiable = unsatisfiable_axes(declaration_for("reanalysis"))
    assert unsatisfiable and all("space" in item for item in unsatisfiable)

    with pytest.raises(InvalidParameterError) as excinfo:
        read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                 domain="reanalysis", time_column="t")
    assert "channel table" in str(excinfo.value)
    assert "latitude" in str(excinfo.value)


def test_the_axis_check_is_callable_on_its_own(domains):
    from src.core.domain import declaration_for

    assert_domain_admits_channel_table(declaration_for("order_book"))
    with pytest.raises(InvalidParameterError):
        assert_domain_admits_channel_table(declaration_for("reanalysis"))


def test_inspect_predicts_exactly_the_refusal_the_read_enforces(domains):
    """Acceptance (d): the advertised refusal and the enforced one are the same refusal."""
    report = inspect_delimited(IRREGULAR_CSV, source_name="irr.csv")
    assert any(row["admits"] for row in report["domains"]), "the case must not pass vacuously"
    for row in report["domains"]:
        if row["admits"]:
            read_channels_for_domain(IRREGULAR_CSV, source_name="irr.csv",
                                     domain=row["name"], time_column="t")
        else:
            with pytest.raises((InvalidParameterError, MissingParameterError)):
                read_channels_for_domain(IRREGULAR_CSV, source_name="irr.csv",
                                         domain=row["name"], time_column="t")


# ------------------------------------------------------------------ irregular sampling (E15)


def test_an_irregular_record_is_refused_by_a_domain_that_did_not_declare_it(domains):
    _channel_domain("tidy_clock", violations=("no_physical_metric", "no_propagation_speed"))
    with pytest.raises(InvalidParameterError) as excinfo:
        read_channels_for_domain(IRREGULAR_CSV, source_name="irr.csv",
                                 domain="tidy_clock", time_column="t")
    assert "irregular_sampling" in str(excinfo.value)


def test_an_irregular_record_loads_under_a_domain_that_declared_it(domains):
    series, declaration = read_channels_for_domain(
        IRREGULAR_CSV, source_name="irr.csv", domain="order_book", time_column="t")
    assert "irregular_sampling" in declaration.violations
    # Reported as absent rather than as a number: an irregular record has no cadence, and a
    # fabricated one would turn every lag in frames into a duration nobody measured.
    assert series.provenance["cadence_seconds"] is None


# ------------------------------------------------------------------ aggregate support (E15)


def test_an_aggregate_channel_requires_the_domain_to_declare_aggregated_values(domains):
    _channel_domain("instantaneous_only",
                    violations=("no_physical_metric", "no_propagation_speed"))
    with pytest.raises(InvalidParameterError) as excinfo:
        read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                 domain="instantaneous_only", time_column="t",
                                 support_parent_px={"bid": 60.0})
    message = str(excinfo.value)
    assert "aggregated_values" in message and "bid" in message
    # The consequence text is drawn from `KNOWN_VIOLATIONS`, not restated, so what a reader is
    # told here and what the analysis layer enforces cannot drift apart (R17).
    assert "depends on more than one sample" in message


def test_an_aggregate_channel_loads_where_the_domain_declared_it(domains):
    series, _ = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                         domain="order_book", time_column="t",
                                         support_parent_px={"bid": 60.0})
    assert list(series.support_parent_px) == [60.0, 1.0]


def test_a_support_of_exactly_one_is_not_an_aggregate(domains):
    """The boundary matters: one sample of the parent axis is an instantaneous reading."""
    _channel_domain("strictly_instantaneous",
                    violations=("no_physical_metric", "no_propagation_speed"))
    series, _ = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                         domain="strictly_instantaneous", time_column="t",
                                         support_parent_px={"bid": 1.0})
    assert list(series.support_parent_px) == [1.0, 1.0]


# ------------------------------------------------------------------ the caps


def test_a_record_above_the_cap_is_refused_rather_than_thinned():
    """Checked on the counts rather than by materialising a 200,000-row string.

    A test that spends a second writing CSV to prove an arithmetic guard is a test that gets
    skipped, and the guard it protects is the one that keeps a record from being silently
    thinned to fit a response body.
    """
    with pytest.raises(InvalidParameterError) as excinfo:
        _assert_within_caps("big.csv", MAX_ROWS + 1, 4)
    message = str(excinfo.value)
    assert "refused rather than thinned" in message
    assert str(MAX_ROWS) in message


def test_the_cell_cap_bites_independently_of_the_row_cap():
    _assert_within_caps("ok.csv", 1000, 10)
    with pytest.raises(InvalidParameterError):
        _assert_within_caps("wide.csv", 1000, MAX_CELLS // 1000 + 1)


def test_a_record_within_the_caps_passes_the_same_gate(domains):
    series, _ = read_channels_for_domain(REGULAR_CSV, source_name="small.csv",
                                         domain="order_book", time_column="t")
    assert len(series.channels) == 2


# ------------------------------------------------------------------ the domain must exist


def test_a_domain_that_was_never_onboarded_is_refused_and_points_at_the_contract(domains):
    DOMAIN_DECLARATIONS.add(
        "piecemeal_domain",
        DomainDeclaration(name="piecemeal_domain",
                          description="Registered without the contract.",
                          axes=(AxisSpec(name="t", role="time", units="s"),
                                AxisSpec(name="channel", role="category", ordered=False)),
                          licence=LICENCE, violations=("no_physical_metric",),
                          lag_policy="none"))
    with pytest.raises(InvalidParameterError) as excinfo:
        read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                 domain="piecemeal_domain", time_column="t")
    assert "onboarded through the contract" in str(excinfo.value)
    assert "findings/onboarding" in str(excinfo.value)


def test_the_provenance_names_the_domain_and_refuses_to_imply_attribution(domains):
    series, _ = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                         domain="order_book", time_column="t")
    assert series.provenance["domain"] == "order_book"
    assert len(series.provenance["onboarding_sha256"]) == 64
    assert "Nothing here establishes that it came from it" in \
        series.provenance["domain_attribution"]
    assert series.provenance["adapter"].endswith("read_channels_for_domain")


def test_the_declaration_returned_is_the_onboarded_one_not_a_freshly_built_copy(domains):
    from src.core.domain import declaration_for

    _series, declaration = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                                    domain="order_book", time_column="t")
    assert declaration is declaration_for("order_book")


def test_reading_the_same_record_twice_yields_one_digest(domains):
    first, _ = read_channels_for_domain(REGULAR_CSV, source_name="reg.csv",
                                        domain="order_book", time_column="t")
    second, _ = read_channels_for_domain(REGULAR_CSV, source_name="other_name.csv",
                                         domain="order_book", time_column="t")
    assert first.provenance["content_sha256"] == second.provenance["content_sha256"]
    assert first.provenance["path_basename"] != second.provenance["path_basename"]
