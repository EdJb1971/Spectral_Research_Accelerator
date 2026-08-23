"""TG1.3: the lag floor is a registered policy, and the sweep applies the declared one.

Three property groups, in the order they were written.

**The seam is real.** A fourth policy registers from this module, with its own parameter, its
own basis and a floor that *varies per channel*, and it runs a complete sweep. It was chosen
to be awkward on the axis the builtins are not: `advective` computes a per-channel floor but
only from a physical grid, `declared` needs no data at all but gives every channel the same
number. `instrument_response` needs neither a grid nor a uniform floor, which is what a lab
archive of sensors with different settling times actually looks like. A fourth policy that is
`declared` with a different name would have proved nothing.

**Nothing atmospheric moved.** `support_floor` is the same function in a different module, and
the advective block, the exclusion wording and the analysis fingerprint are asserted against
values recomputed here from the pre-TG1.3 arithmetic rather than against themselves.

**The defect the closed set was hiding.** `cross_scale_dependency` called `support_floor`
unconditionally, so a domain declaring ``lag_policy='declared'`` had its floor enforced at the
boundary and then reported the *advective* floor of one frame in every test record, with an
exclusion reason citing rule R4. The right tests ran; the receipt described a different study.
"""

import math

import numpy as np
import pytest

from src.analysis_engine import domain_analysis as da
from src.analysis_engine.cross_scale import (GateProtocol, cross_scale_dependency,
                                             evaluate_replication_gate, support_floor)
from src.core import lag_policy as lp
from src.core.channel_series import ChannelGeometrySpec, ChannelSeries
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.registry import restore, snapshot
from src.physical_core.grid import GridSpec

N_FRAMES = 600
CADENCE = 60.0
CHANNELS = ("fast", "medium", "slow")
LICENCE = "CC-BY-4.0 (fixture; no real archive was accessed)"


# ------------------------------------------------------------------ the fourth policy


class InstrumentResponsePolicy(lp.LagPolicy):
    """A floor per channel, from each instrument's settling time.

    Awkward on purpose. The floor is neither uniform across channels (as `declared`'s is) nor
    derived from a physical grid (as `advective`'s is): it comes from a parameter the caller
    supplies, keyed by channel. If the seam only supported floors shaped like the two that
    already existed, this would not fit through it.
    """

    name = "instrument_response"

    def validate_declaration(self, declaration):
        super().validate_declaration(declaration)
        if "irregular_sampling" in tuple(declaration.violations or ()):
            raise InvalidParameterError(
                "DomainDeclaration.lag_policy", self.name,
                "a regular clock: a settling time in seconds is only a lag in frames when "
                "the frames are evenly spaced")

    def check_lags(self, lags, declaration, params):
        if not params.get("response_seconds"):
            raise InvalidParameterError(
                "response_seconds", None,
                "a settling time per channel. It has no default for the same reason "
                "`advection_speed_m_s` has none")

    def _frames(self, declaration, params, cadence_seconds, scale):
        seconds = float(params["response_seconds"][str(scale)])
        return max(1, int(math.ceil(seconds / cadence_seconds)))

    def floors(self, signature, cadence_seconds, declaration, params):
        records = []
        for position, scale in enumerate(signature.channels, start=1):
            frames = self._frames(declaration, params, cadence_seconds, scale)
            records.append({
                "scale": str(scale), "level": position, "floor_frames": frames,
                "response_seconds": float(params["response_seconds"][str(scale)]),
                "basis": "instrument settling time for channel %s" % scale})
        return {
            "policy": self.name,
            "cadence_seconds": float(cadence_seconds),
            "floors": records,
            "floor_by_scale": {record["scale"]: record["floor_frames"]
                               for record in records},
            "enforced": True,
            "basis": "per-instrument settling times",
            "warnings": [],
        }

    def exclusion_reason(self, floor_frames):
        return ("below the %d-frame settling time of one of the two instruments"
                % floor_frames)

    def config_entries(self, declaration, params):
        return {"response_seconds": {str(key): float(value) for key, value
                                     in sorted(params["response_seconds"].items())}}


@pytest.fixture
def fourth_policy():
    """Registered for the duration of one test, exactly as a plugin would register it."""
    state = snapshot(lp.LAG_POLICIES)
    lp.LAG_POLICIES.add(
        "instrument_response", InstrumentResponsePolicy(),
        description="A floor per channel, from each instrument's settling time.",
        capabilities={"precedence_admissible": True, "floor_from_declaration": False,
                      "floor_from_geometry": False, "requires_channel_support": False,
                      "parameters": ("response_seconds",)})
    try:
        yield "instrument_response"
    finally:
        restore(lp.LAG_POLICIES, state)


# ------------------------------------------------------------------ fixtures


def _declaration(**overrides):
    kwargs = dict(
        name="synthetic_instrument_log",
        description="Three unitless channels on a regular clock. No physics.",
        licence=LICENCE,
        axes=(AxisSpec("minute", "time", units="s"),
              AxisSpec("channel", "category", ordered=False)),
        violations=("no_physical_metric", "no_propagation_speed"),
        lag_policy="declared", declared_floor_frames=2,
        declared_floor_basis="two reporting intervals")
    kwargs.update(overrides)
    return DomainDeclaration(**kwargs)


def _ar1(rng, rho=0.6, n=N_FRAMES):
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def _series(seed=11, **overrides):
    rng = np.random.default_rng(seed)
    kwargs = dict(channels=CHANNELS,
                  times_seconds=np.arange(N_FRAMES, dtype=float) * CADENCE,
                  measures={"value": np.column_stack([_ar1(rng) for _ in CHANNELS])},
                  support_parent_px=(1, 1, 1))
    kwargs.update(overrides)
    return ChannelSeries(**kwargs)


def _atmospheric_signature(levels=3, supports=(7, 19, 43)):
    """The shape `support_floor` was written for: labelled scales on a lat/lon crop."""
    grid = GridSpec.latlon((64, 96), lat0=-30.0, dlat=-0.25, lon0=165.0, dlon=0.25)
    return ChannelGeometrySpec(channels=list(range(1, levels + 1)),
                               support_parent_px=list(supports[:levels]),
                               provenance={"grid": grid.to_provenance()})


# ------------------------------------------------------------------ the seam is real


def test_the_three_original_policies_declare_what_they_license():
    """The capability table, asserted rather than described in a docstring."""
    assert lp.policy_names() == ("advective", "declared", "none")
    table = {name: {key: lp.capability(name, key)
                    for key in ("precedence_admissible", "floor_from_declaration",
                                "floor_from_geometry", "requires_channel_support",
                                "parameters")}
             for name in lp.policy_names()}
    assert table == {
        "advective": {"precedence_admissible": True, "floor_from_declaration": False,
                      "floor_from_geometry": True, "requires_channel_support": True,
                      "parameters": ("advection_speed_m_s",)},
        "declared": {"precedence_admissible": True, "floor_from_declaration": True,
                     "floor_from_geometry": False, "requires_channel_support": False,
                     "parameters": ()},
        "none": {"precedence_admissible": False, "floor_from_declaration": False,
                 "floor_from_geometry": False, "requires_channel_support": False,
                 "parameters": ()},
    }


def test_an_unknown_policy_lists_the_ones_that_exist():
    with pytest.raises(UnknownNameError) as caught:
        _declaration(lag_policy="advection")
    message = str(caught.value)
    assert "advective" in message and "declared" in message and "none" in message


def test_a_fourth_policy_registers_without_editing_src(fourth_policy):
    """The acceptance criterion for this seam.

    Declaration, validation, the pre-flight lag check, the floor the sweep applies, the
    exclusion wording and the analysis fingerprint all come from a class defined in this
    file. Nothing in `src/` names it.
    """
    declaration = _declaration(lag_policy=fourth_policy, declared_floor_frames=None,
                               declared_floor_basis=None)
    assert declaration.precedence_admissible is True
    assert declaration.minimum_admissible_lag() is None

    response = {"fast": 30.0, "medium": 100.0, "slow": 200.0}
    result = da.analyse_precedence(
        _series(), declaration, lags=(2, 3, 4), cadence_seconds=CADENCE, measure="value",
        lag_policy_params={"response_seconds": response}, n_surrogates=99, bins=4)

    # ceil(30/60)=1, ceil(100/60)=2, ceil(200/60)=4: a floor that varies per channel, which
    # neither builtin policy can express.
    assert result["support_floor"]["floor_by_scale"] == {"fast": 1, "medium": 2, "slow": 4}
    assert result["applied_lag_floor"] == {
        "policy": "instrument_response", "frames": 4,
        "basis": "per-instrument settling times"}
    assert result["analysis_config"]["response_seconds"] == response
    assert "advection_speed_m_s" not in result["analysis_config"]


def test_the_fourth_policys_floor_excludes_only_the_pairs_that_are_below_it(fourth_policy):
    """A per-channel floor has to reach the sweep, not merely the receipt."""
    declaration = _declaration(lag_policy=fourth_policy, declared_floor_frames=None,
                               declared_floor_basis=None)
    result = da.analyse_precedence(
        _series(), declaration, lags=(2, 3, 4), cadence_seconds=CADENCE, measure="value",
        lag_policy_params={"response_seconds": {"fast": 30.0, "medium": 100.0,
                                                "slow": 200.0}},
        n_surrogates=99, bins=4)

    tested = {(row["source_scale"], row["target_scale"], row["lag_frames"])
              for row in result["results"]}
    # `slow` is involved in every pair with a floor of 4, so only lag 4 survives for those.
    assert ("fast", "slow", 2) not in tested and ("fast", "slow", 4) in tested
    # `fast`->`medium` clears its floor of 2 at every requested lag.
    assert {2, 3, 4} <= {lag for source, target, lag in tested
                         if (source, target) == ("fast", "medium")}
    reasons = {row["reason"] for row in result["excluded"]}
    assert any("settling time" in reason for reason in reasons)
    assert not any("rule R4" in reason for reason in reasons), (
        "a domain with no advection must not be told its lag was refused by rule R4")


def test_a_parameter_the_policy_does_not_consume_is_refused_not_ignored():
    """The old `advection_speed_m_s`-under-`declared` refusal, now generic.

    Every policy gets it, including a fourth nobody has written yet, because the rule is
    stated once against the declared parameter list.
    """
    with pytest.raises(InvalidParameterError, match="Two floors"):
        da.analyse_precedence(_series(), _declaration(), lags=(3,),
                              cadence_seconds=CADENCE, measure="value",
                              advection_speed_m_s=10.0)
    with pytest.raises(InvalidParameterError, match="Two floors"):
        da.analyse_precedence(_series(), _declaration(), lags=(3,),
                              cadence_seconds=CADENCE, measure="value",
                              lag_policy_params={"response_seconds": {"fast": 1.0}})


def test_a_speed_given_twice_is_refused_rather_than_reconciled(fourth_policy):
    declaration = _declaration(lag_policy="advective", declared_floor_frames=None,
                               declared_floor_basis=None,
                               violations=("no_physical_metric",))
    with pytest.raises(InvalidParameterError, match="supplied once"):
        da.analyse_precedence(_series(), declaration, lags=(3,), cadence_seconds=CADENCE,
                              measure="value", advection_speed_m_s=10.0,
                              lag_policy_params={"advection_speed_m_s": 12.0})


def test_precedence_is_licensed_by_what_the_policy_declares_not_by_its_name(fourth_policy):
    """`precedence_admissible` used to read `lag_policy != "none"`.

    A fourth policy that justified no floor would have licensed precedence by default, which
    is the failure R21 exists to prevent: the permissive answer must be the declared one.
    """
    state = snapshot(lp.LAG_POLICIES)
    try:
        lp.LAG_POLICIES.add("unfloored", lp.NoFloorPolicy(),
                            capabilities={"precedence_admissible": False,
                                          "requires_channel_support": False,
                                          "parameters": ()})
        declaration = _declaration(lag_policy="unfloored", declared_floor_frames=None,
                                   declared_floor_basis=None)
        assert declaration.precedence_admissible is False
        with pytest.raises(InvalidParameterError, match="rule R21 forbids"):
            declaration.assert_precedence_admissible()
    finally:
        restore(lp.LAG_POLICIES, state)


# ------------------------------------------------------------------ nothing atmospheric moved


def test_the_advective_floor_is_the_pre_registry_arithmetic():
    """Recomputed here from `support * dx / U`, not compared against itself."""
    signature = _atmospheric_signature()
    speed = 12.0
    block = support_floor(signature, CADENCE, speed)

    grid = GridSpec.latlon((64, 96), lat0=-30.0, dlat=-0.25, lon0=165.0, dlon=0.25)
    spacing = max(float(grid.dx_metres().max()), float(grid.dy_metres().max()))
    for record, support in zip(block["floors"], (7, 19, 43)):
        crossing = support * spacing / speed
        assert record["floor_frames"] == max(1, int(math.ceil(crossing / CADENCE)))
        assert record["spatial_support_m"] == pytest.approx(support * spacing)
        assert record["crossing_time_s"] == pytest.approx(crossing)
    assert block["enforced"] is True
    assert block["policy"] == "advective"
    assert block["warnings"] == []


def test_without_a_speed_the_advective_floor_is_one_frame_and_says_why():
    block = support_floor(_atmospheric_signature(), CADENCE)
    assert {record["floor_frames"] for record in block["floors"]} == {1}
    assert block["enforced"] is False
    assert any("never chose" in warning for warning in block["warnings"])


def test_the_default_sweep_still_binds_the_advective_policy():
    """No caller outside the domain layer passes a policy, and none had to start."""
    signature = _atmospheric_signature()
    series = ChannelSeries(channels=list(signature.channels),
                           times_seconds=np.arange(N_FRAMES, dtype=float) * CADENCE,
                           measures={"value": np.column_stack(
                               [_ar1(np.random.default_rng(3)) for _ in range(3)])},
                           support_parent_px=(7, 19, 43),
                           provenance=dict(signature.provenance))
    result = cross_scale_dependency(series, lags=(6,), cadence_seconds=CADENCE,
                                    measure="value", n_surrogates=99, bins=4)
    assert result["support_floor"]["policy"] == "advective"
    assert result["analysis_config"]["advection_speed_m_s"] is None
    assert set(result["analysis_config"]) == {
        "estimator", "measure", "bins", "wrap", "n_scales", "lags_frames",
        "cadence_seconds", "advection_speed_m_s", "n_surrogates", "alpha", "correction",
        "seed"}, "the atmospheric fingerprint must carry exactly the keys it always carried"


def test_support_floor_is_still_importable_from_its_old_home():
    """`gate_run` and `gate_campaign` import it from `cross_scale`, and still may."""
    from src.analysis_engine import cross_scale

    assert cross_scale.support_floor is lp.support_floor


def test_an_explicit_policy_and_a_bare_speed_together_are_refused():
    bound = lp.bind("advective", None, advection_speed_m_s=12.0)
    with pytest.raises(InvalidParameterError, match="not say which"):
        cross_scale_dependency(_series(), lags=(3,), cadence_seconds=CADENCE,
                               measure="value", advection_speed_m_s=10.0,
                               lag_floor=bound, n_surrogates=99, bins=4)


# ------------------------------------------------------------------ the defects


def test_a_declared_floor_now_reaches_the_sweep_and_not_only_the_boundary():
    """The receipt defect.

    Before TG1.3 the domain's floor of 2 was enforced by `analyse_precedence`, and then every
    test record reported ``support_floor_frames: 1`` because the sweep computed its own
    advective floor. Nothing was mis-tested; the receipt named the wrong rule and the wrong
    number, which is what a receipt exists to prevent.
    """
    result = da.analyse_precedence(_series(), _declaration(), lags=(3, 4),
                                   cadence_seconds=CADENCE, measure="value",
                                   n_surrogates=99, bins=4)
    assert {row["support_floor_frames"] for row in result["results"]} == {2}
    assert result["support_floor"]["policy"] == "declared"
    assert result["support_floor"]["declared_floor_frames"] == 2
    assert result["analysis_config"]["declared_floor_frames"] == 2
    assert "advection_speed_m_s" not in result["analysis_config"], (
        "a speed nobody supplied and no floor used should not enter the fingerprint")


def test_a_lag_family_below_the_declared_floor_is_still_refused_whole():
    with pytest.raises(InvalidParameterError, match="refused rather than dropped"):
        da.analyse_precedence(_series(), _declaration(), lags=(1, 3),
                              cadence_seconds=CADENCE, measure="value")


def test_the_replication_gate_refuses_a_floor_from_the_wrong_policy():
    """`require_advection_floor` checked `enforced` alone, which a declared floor satisfies.

    A declared floor is a floor, but it is not the one the protocol froze, and rule R18 fixes
    the design before the run rather than accepting a substitute during it.
    """
    protocol = GateProtocol(study_id="tg1.3", n_scales=3, lags=(3,),
                            expected_frames=N_FRAMES, cadence_seconds=CADENCE,
                            train_ratio=0.6, embargo_frames=3, measure="value", bins=3,
                            n_surrogates=499, alpha=0.05, require_advection_floor=True,
                            seed=1)
    declared = da.analyse_precedence(_series(), _declaration(), lags=(3,),
                                     cadence_seconds=CADENCE, measure="value",
                                     n_surrogates=499, bins=3, alpha=0.05,
                                     estimator=protocol.estimator,
                                     correction=protocol.correction)
    verdict = evaluate_replication_gate(declared, declared, protocol)
    assert verdict["verdict"] == "INVALID"
    assert any("did not enforce the declared advection floor" in problem
               for problem in verdict["problems"])


def test_the_domain_gate_can_now_run_an_advective_domain():
    """Defect D58.

    `run_domain_gate` refused `require_advection_floor` for any policy but `advective`, and
    then called `analyse_precedence` with no way to pass a speed - so the advective path
    through the domain gate could not be executed at all. The declaration and the runner each
    looked correct; the pair was unreachable.
    """
    # Hourly frames, because an advective floor over a 0.25-degree crop at 12 m/s is
    # 28 frames: the atmospheric path's own scale, not a fixture's.
    frames, cadence, speed = 900, 3600.0, 12.0
    signature = _atmospheric_signature()
    rng = np.random.default_rng(5)
    series = ChannelSeries(channels=list(signature.channels),
                           times_seconds=np.arange(frames, dtype=float) * cadence,
                           measures={"value": np.column_stack(
                               [_ar1(rng, n=frames) for _ in range(3)])},
                           support_parent_px=(7, 19, 43),
                           provenance=dict(signature.provenance))
    declaration = _declaration(
        lag_policy="advective", declared_floor_frames=None, declared_floor_basis=None,
        violations=("no_natural_cycle",))
    protocol = GateProtocol(study_id="tg1.3-advective", n_scales=3, lags=(30,),
                            expected_frames=frames, cadence_seconds=cadence,
                            train_ratio=0.6, embargo_frames=30, measure="value", bins=3,
                            n_surrogates=499, alpha=0.05, require_advection_floor=True,
                            seed=2)
    result = da.run_domain_gate(series, declaration, protocol,
                                lag_policy_params={"advection_speed_m_s": speed})
    assert result["verdict"] in ("PASS", "FAIL")
    assert result["train"]["support_floor"]["enforced"] is True
    assert result["train"]["applied_lag_floor"]["policy"] == "advective"


def test_a_declared_floor_carried_by_a_policy_that_ignores_it_is_refused():
    """Stated once against `floor_from_declaration` rather than once per policy."""
    with pytest.raises(InvalidParameterError, match="never applied to a single test"):
        _declaration(lag_policy="none", declared_floor_frames=4,
                     declared_floor_basis="a number nothing would read")


def test_rule_r17_still_refuses_a_domain_that_breaks_nothing():
    """Moved into the `none` policy, where it belongs, and unchanged in effect."""
    with pytest.raises(InvalidParameterError,
                       match="second variable, not a second domain"):
        _declaration(lag_policy="none", violations=(), declared_floor_frames=None,
                     declared_floor_basis=None)


def test_an_advective_domain_that_declares_no_propagation_speed_is_refused():
    with pytest.raises(InvalidParameterError, match="no_propagation_speed"):
        _declaration(lag_policy="advective", declared_floor_frames=None,
                     declared_floor_basis=None,
                     violations=("no_physical_metric", "no_propagation_speed"))


def test_the_fourth_policy_may_refuse_a_declaration_the_builtins_accept(fourth_policy):
    """A policy's own consistency rules travel with it, as `advective`'s now do."""
    with pytest.raises(InvalidParameterError, match="regular clock"):
        _declaration(lag_policy=fourth_policy, declared_floor_frames=None,
                     declared_floor_basis=None,
                     violations=("no_physical_metric", "irregular_sampling"))


def test_a_domain_with_no_floor_says_so_in_the_block_and_not_only_in_the_warning():
    declaration = _declaration(lag_policy="none", declared_floor_frames=None,
                               declared_floor_basis=None)
    result = da.association_only(_series(), declaration, lags=(3,),
                                 cadence_seconds=CADENCE, measure="value",
                                 n_surrogates=99, bins=4)
    assert result["claim_boundary"] == "association"
    assert result["support_floor"]["policy"] == "none"
    assert result["support_floor"]["enforced"] is False
    assert result["applied_lag_floor"] == {
        "policy": "none", "frames": None,
        "basis": "no floor exists for this domain (rule R21)"}
    assert any("does not have" in warning
               for warning in result["support_floor"]["warnings"])
