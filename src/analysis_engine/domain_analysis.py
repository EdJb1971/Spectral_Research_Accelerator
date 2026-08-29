"""Running the accepted cross-channel sweep on a declared domain (TG0.2, rule R21).

This module is deliberately thin. It does not reimplement anything: `cross_scale_dependency`
runs **unmodified**, with the same surrogate ensemble, the same BY correction, the same power
check and the same excluded-test accounting it applies to ERA5. That is the point of Phase G0 —
if a second domain needed its own inference path, the abstraction would already have failed.

What this adds is the gate that the atmospheric path got for free and a generic domain does
not: rule R21's question of whether a lag is admissible *at all*, asked before any work.

Since TG1.3 the policies are a registry (`src.core.lag_policy`) and this module no longer
knows their names. It binds the domain's declared policy to whatever parameters the caller
supplied, hands the bound policy to `cross_scale_dependency`, and asks it three questions:
may this domain claim precedence, is this lag family admissible, and which floor decided. The
three original policies - ``advective``, ``declared`` and ``none`` - answer them exactly as
the branches here used to, and a fourth answers them without this file being edited.

**What the seam fixed.** The floor the *sweep* applied was always the advective one, because
`cross_scale_dependency` called `support_floor` unconditionally. Under ``declared`` the
domain's floor was enforced here, at the boundary, and then every test record in the receipt
reported ``support_floor_frames: 1`` and an exclusion reason citing rule R4. The tests that
ran were the right ones; the receipt described a different study.

**Why a refusal can still live here.** A refusal is better at the domain boundary than inside
the wavelet vocabulary: it can name the domain, the violation it declared, and the way
forward. What changed is that the *condition* is now asked of the policy - only a policy that
measures a filter crossing needs `support_parent_px` - so a sensor archive is no longer asked
for a wavelet number that could not have reached any floor it applies.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.analysis_engine.cross_scale import (GateProtocol, cross_scale_dependency,
                                             evaluate_replication_gate)
from src.core.channel_series import (ChannelSeries, ChannelSeriesLike,
                                     assert_presence_contract, from_channel_series,
                                     split_channel_series)
from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.lag_policy import BoundLagPolicy, bind as bind_lag_policy

#: What a result is permitted to be called, given the domain's lag policy. Deliberately not
#: the G6 claim ladder: this is one rung of provenance, not an adjudication.
PRECEDENCE_CLAIM = "precedence"
ASSOCIATION_CLAIM = "association"


def _bind(declaration: DomainDeclaration, advection_speed_m_s: Optional[float],
          lag_policy_params: Optional[Mapping[str, Any]]) -> BoundLagPolicy:
    """Resolve the domain's policy and check its parameters before anything is computed.

    `advection_speed_m_s` keeps its own keyword because every atmospheric caller passes it by
    name, but it is merely one policy's parameter now: a policy that does not consume it
    refuses it here by the generic rule, rather than by a branch that names it.
    """
    params: Dict[str, Any] = dict(lag_policy_params or {})
    if advection_speed_m_s is not None:
        if "advection_speed_m_s" in params:
            raise InvalidParameterError(
                "advection_speed_m_s", advection_speed_m_s,
                "the speed supplied once. It arrived both directly and in "
                "lag_policy_params, and the two need not agree")
        params["advection_speed_m_s"] = float(advection_speed_m_s)
    return bind_lag_policy(declaration.lag_policy, declaration, **params)


def _require_declared_support(series: ChannelSeriesLike, declaration: DomainDeclaration,
                              bound: BoundLagPolicy) -> None:
    """Fail at the domain boundary rather than inside the wavelet vocabulary.

    Only for a policy that declares `requires_channel_support`. An advective floor needs each
    channel's footprint on the parent axis - how many source samples one value depends on -
    and that generalises honestly beyond wavelets: a raw sensor reading depends on exactly
    one sample, a ten-minute mean of one-minute data on ten. What it must never be is
    invented, so it is required where it is read and not required where it is not. Before
    TG1.3 every domain paid it, including ones whose floor could not have read it.
    """
    if not bound.capability("requires_channel_support", False):
        return
    missing = [str(record.get("scale")) for record in series.channel_records
               if not isinstance(record.get("support_parent_px"), (int, float))
               or record.get("support_parent_px", 0) <= 0]
    if missing:
        raise InvalidParameterError(
            "channel_records.support_parent_px", missing,
            "a declared parent-axis footprint for every channel of domain %r: the number of "
            "source samples each value depends on. Declare 1 for an instantaneous reading, "
            "or the window length for an aggregate (and add the 'aggregated_values' "
            "violation). It has no default because a fabricated footprint would set the "
            "lag floor of every result from a number the reader never chose"
            % declaration.name,
            domain=declaration.name)


def _run(series: ChannelSeriesLike, declaration: DomainDeclaration, *, claim: str,
         lags: Sequence[int], cadence_seconds: float, measure: str,
         bound: BoundLagPolicy, **kwargs: Any) -> Dict[str, Any]:
    assert_presence_contract(series, declaration.violations, declaration.name)
    _require_declared_support(series, declaration, bound)
    result = cross_scale_dependency(series, lags=lags, cadence_seconds=cadence_seconds,
                                    measure=measure, lag_floor=bound, **kwargs)
    result["domain"] = declaration.describe()
    result["channel_series"] = from_channel_series(series)
    result["claim_boundary"] = claim
    # Which floor actually decided admissibility, said by the policy that decided it. The
    # block predates TG1.3 and existed because the `support_floor` block underneath it was
    # always the advective one whatever the domain declared; it is kept because a reader
    # should not have to know which policy writes which keys to find the number that set the
    # boundary of the study.
    result["applied_lag_floor"] = bound.applied_floor(result["support_floor"])
    present = getattr(series, "present", None)
    if present is not None and bool(np.all(np.asarray(present, dtype=bool))):
        result["warnings"] = list(result.get("warnings", [])) + [
            "domain %r declares 'non_stationary_support' and supplied the required mask; "
            "this particular record is fully present. The declaration describes the source, "
            "not a promise that every query contains a gap." % declaration.name]
    if claim == ASSOCIATION_CLAIM:
        result["warnings"] = list(result.get("warnings", [])) + [
            "domain %r declares no admissible lag floor (rule R21), so this result is an "
            "association at the tested lags and must not be described as precedence, lead-lag "
            "or a precursor relationship. The estimator is unchanged; only what it licenses "
            "is." % declaration.name]
    return result


def analyse_precedence(series: ChannelSeriesLike, declaration: DomainDeclaration, *,
                       lags: Sequence[int], cadence_seconds: float,
                       measure: str, advection_speed_m_s: Optional[float] = None,
                       lag_policy_params: Optional[Mapping[str, Any]] = None,
                       **kwargs: Any) -> Dict[str, Any]:
    """Cross-channel precedence, refused unless the domain can justify a lag floor (R21).

    Every policy-specific step - which parameters are admissible, whether the requested lag
    family clears the floor, what the floor is and what it rests on - is asked of the
    registered policy. This function names none of them.
    """
    declaration.assert_precedence_admissible()
    lags = [int(lag) for lag in lags]
    if not lags:
        raise InvalidParameterError("lags", lags, "at least one lag to test")

    bound = _bind(declaration, advection_speed_m_s, lag_policy_params)
    bound.check_lags(lags)
    return _run(series, declaration, claim=PRECEDENCE_CLAIM, lags=lags, bound=bound,
                cadence_seconds=cadence_seconds, measure=measure, **kwargs)


def association_only(series: ChannelSeriesLike, declaration: DomainDeclaration, *,
                     lags: Sequence[int], cadence_seconds: float, measure: str,
                     lag_policy_params: Optional[Mapping[str, Any]] = None,
                     **kwargs: Any) -> Dict[str, Any]:
    """Cross-channel association for a domain with no admissible lag floor.

    The estimator, surrogate null, correction and power check are identical to the precedence
    path. What differs is the label the result carries and the warning it emits, because the
    difference between association and precedence here is a matter of what the domain can
    justify, not of what was computed.
    """
    if declaration.precedence_admissible:
        raise InvalidParameterError(
            "domain.lag_policy", declaration.lag_policy,
            "'none' for the association-only path. Domain %r can justify a lag floor, so use "
            "`analyse_precedence`, which applies it. Downgrading the claim while skipping the "
            "floor would report a weaker conclusion from a weaker test and look conservative "
            "while being neither" % declaration.name)
    lags = [int(value) for value in lags]
    bound = _bind(declaration, None, lag_policy_params)
    bound.check_lags(lags)
    return _run(series, declaration, claim=ASSOCIATION_CLAIM, lags=lags, bound=bound,
                cadence_seconds=cadence_seconds, measure=measure, **kwargs)


# ------------------------------------------------------------------- the replication gate

def run_domain_gate(series: ChannelSeries, declaration: DomainDeclaration,
                    protocol: GateProtocol, *,
                    lag_policy_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Split, sweep both partitions under the frozen protocol, and adjudicate (R6, TG0.3).

    The verdict comes from `evaluate_replication_gate` unchanged, so it means here exactly
    what it means for ERA5:

    *   **PASS** — the same positive, corrected relationship appears independently in train and
        test.
    *   **FAIL** — an adequately powered absence. A result, not a failure of the run.
    *   **INVALID** — the run cannot adjudicate: design drift, or a partition too weak to
        reject anything. Must never be displayed as a negative finding.

    The distinction between FAIL and INVALID is the whole point of the exercise. A system that
    reports "nothing found" when it was arithmetically incapable of finding anything is worse
    than one that reports nothing at all, because the first looks like evidence.

    The protocol is validated **before** the split, so an underpowered or leaky design is
    refused rather than run and then explained.
    """
    design = protocol.validate()

    channels = list(series.channels)
    if len(channels) != protocol.n_scales:
        raise InvalidParameterError(
            "protocol.n_scales", protocol.n_scales,
            "the number of channels actually present (%d: %s). The frozen family size is "
            "computed from this, and a mismatch would correct the wrong number of tests"
            % (len(channels), channels))
    if series.n_times != protocol.expected_frames:
        raise InvalidParameterError(
            "protocol.expected_frames", protocol.expected_frames,
            "the number of frames actually present (%d). The design's per-partition power was "
            "validated against the declared count, so a shorter record silently invalidates it"
            % series.n_times)
    # As in `evaluate_replication_gate`: `require_advection_floor` names one policy
    # deliberately. This is the protocol asserting which design it froze, not the analysis
    # layer choosing a code path from a string.
    if protocol.require_advection_floor and declaration.lag_policy != "advective":
        raise InvalidParameterError(
            "protocol.require_advection_floor", True,
            "False for domain %r, which declares lag_policy=%r. The gate would demand an "
            "enforced advective floor that this domain has already said does not exist, and "
            "return INVALID for a reason that is really a mis-declared design"
            % (declaration.name, declaration.lag_policy))

    partitions = split_channel_series(series, train_ratio=protocol.train_ratio,
                                      embargo_frames=protocol.embargo_frames)

    def sweep(part: ChannelSeries, seed: int) -> Dict[str, Any]:
        runner = (analyse_precedence if declaration.precedence_admissible
                  else association_only)
        result = runner(part, declaration, lags=protocol.lags,
                        lag_policy_params=lag_policy_params,
                        cadence_seconds=protocol.cadence_seconds,
                        measure=protocol.measure, estimator=protocol.estimator,
                        bins=protocol.bins, n_surrogates=protocol.n_surrogates,
                        alpha=protocol.alpha, correction=protocol.correction, seed=seed)
        result["protocol_fingerprint"] = protocol.fingerprint()
        return result

    train = sweep(partitions["train"], protocol.seed)
    test = sweep(partitions["test"], protocol.seed + 1)
    gate = evaluate_replication_gate(train, test, protocol)

    return {
        "protocol": protocol.to_mapping(),
        "protocol_fingerprint": protocol.fingerprint(),
        "design": design,
        "domain": declaration.describe(),
        "split": {
            "train_frames": partitions["train"].n_times,
            "embargo_frames": protocol.embargo_frames,
            "test_frames": partitions["test"].n_times,
            "embargo_at_least_longest_lag": (
                protocol.embargo_frames >= max(int(v) for v in protocol.lags)),
        },
        "train": train,
        "test": test,
        "gate": gate,
        "verdict": gate["verdict"],
        "claim_boundary": (
            "PASS adjudicates only the frozen relationship family on this exact record from "
            "domain %r, at the declared claim level %r. It is not causality, not "
            "universality, and not transfer to any other domain (rules R7, R14, R20)."
            % (declaration.name, train["claim_boundary"])),
    }
