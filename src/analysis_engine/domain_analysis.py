"""Running the accepted cross-channel sweep on a declared domain (TG0.2, rule R21).

This module is deliberately thin. It does not reimplement anything: `cross_scale_dependency`
runs **unmodified**, with the same surrogate ensemble, the same BY correction, the same power
check and the same excluded-test accounting it applies to ERA5. That is the point of Phase G0 —
if a second domain needed its own inference path, the abstraction would already have failed.

What this adds is the gate that the atmospheric path got for free and a generic domain does
not: rule R21's question of whether a lag is admissible *at all*, asked before any work.

Three policies, from the domain's own declaration:

*   ``advective`` — the atmospheric path. Delegates to `support_floor`, which derives the floor
    from the transform's measured filter support and a declared speed.
*   ``declared`` — the domain supplies a floor in frames with a recorded basis. Every requested
    lag is checked against it here, and lags below it are refused rather than dropped: a sweep
    that silently discards what it cannot test looks identical to one that had nothing to
    discard.
*   ``none`` — no floor exists, so no precedence claim is admissible. `association_only` is
    then the honest entry point, and it labels its own output accordingly.

**Why the refusal lives here rather than in `support_floor`.** `support_floor` refuses a
missing `support_parent_px` with a message about parent-grid filter support, which is exactly
right for a wavelet bank and useless to someone onboarding a sensor archive. Catching the
condition at the domain boundary means the refusal can name the domain, the violation it
declared, and the two ways forward. Generalising `support_floor` itself into a policy registry
is TG1.3; this is the boundary that makes TG0.2 usable before that lands.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.analysis_engine.cross_scale import (GateProtocol, cross_scale_dependency,
                                             evaluate_replication_gate)
from src.core.channel_series import (ChannelSeries, ChannelSeriesLike, from_channel_series,
                                     split_channel_series)
from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError

#: What a result is permitted to be called, given the domain's lag policy. Deliberately not
#: the G6 claim ladder: this is one rung of provenance, not an adjudication.
PRECEDENCE_CLAIM = "precedence"
ASSOCIATION_CLAIM = "association"


def _require_declared_support(series: ChannelSeriesLike,
                              declaration: DomainDeclaration) -> None:
    """Fail at the domain boundary rather than inside the wavelet vocabulary.

    `support_floor` needs each channel's footprint on the parent axis — how many source
    samples one value depends on. That generalises honestly beyond wavelets: a raw sensor
    reading depends on exactly one sample, a ten-minute mean of one-minute data on ten. What
    it must never be is invented. Without the declaration this refusal names the domain and
    the fix; with it, the wavelet-flavoured message never reaches a non-wavelet adapter.
    """
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
         **kwargs: Any) -> Dict[str, Any]:
    _require_declared_support(series, declaration)
    result = cross_scale_dependency(series, lags=lags, cadence_seconds=cadence_seconds,
                                    measure=measure, **kwargs)
    result["domain"] = declaration.describe()
    result["channel_series"] = from_channel_series(series)
    result["claim_boundary"] = claim
    # Which floor actually decided admissibility. Without this the receipt carries only
    # `support_floor`'s internal block, which under a declared or absent policy reports one
    # frame and reads as though no floor was applied at all.
    result["applied_lag_floor"] = {
        "policy": declaration.lag_policy,
        "frames": (declaration.minimum_admissible_lag()
                   if declaration.lag_policy == "declared"
                   else (None if declaration.lag_policy == "none"
                         else max(record["floor_frames"]
                                  for record in result["support_floor"]["floors"]))),
        "basis": (declaration.declared_floor_basis if declaration.lag_policy == "declared"
                  else ("no floor exists for this domain (rule R21)"
                        if declaration.lag_policy == "none"
                        else "advective crossing of the declared representation support")),
    }
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
                       **kwargs: Any) -> Dict[str, Any]:
    """Cross-channel precedence, refused unless the domain can justify a lag floor (R21)."""
    declaration.assert_precedence_admissible()
    lags = [int(lag) for lag in lags]
    if not lags:
        raise InvalidParameterError("lags", lags, "at least one lag to test")

    if declaration.lag_policy == "declared":
        floor = declaration.minimum_admissible_lag()
        below = [lag for lag in lags if lag < floor]
        if below:
            raise InvalidParameterError(
                "lags", below,
                "lags at or above domain %r's declared floor of %d frame(s) (%s). They are "
                "refused rather than dropped, because a family silently reduced to the "
                "testable lags is indistinguishable from one that had nothing to drop, and "
                "rule R18 fixes the family before the sweep runs"
                % (declaration.name, floor, declaration.declared_floor_basis))
        # The domain's floor is the whole justification, so an advection speed would be a
        # second, unreconciled one. Refused rather than ignored.
        if advection_speed_m_s is not None:
            raise InvalidParameterError(
                "advection_speed_m_s", advection_speed_m_s,
                "None under lag_policy='declared'. Two floors from two different bases would "
                "silently take the larger, and the result would not say which applied")
        return _run(series, declaration, claim=PRECEDENCE_CLAIM, lags=lags,
                    cadence_seconds=cadence_seconds, measure=measure, **kwargs)

    if advection_speed_m_s is None:
        raise InvalidParameterError(
            "advection_speed_m_s", None,
            "a declared transport speed under lag_policy='advective'. It has no default here "
            "for the same reason it has none in `support_floor`: a plausible-looking value "
            "would set every floor in every result from a number the reader never chose")
    return _run(series, declaration, claim=PRECEDENCE_CLAIM, lags=lags,
                cadence_seconds=cadence_seconds, measure=measure,
                advection_speed_m_s=advection_speed_m_s, **kwargs)


def association_only(series: ChannelSeriesLike, declaration: DomainDeclaration, *,
                     lags: Sequence[int], cadence_seconds: float,
                     measure: str, **kwargs: Any) -> Dict[str, Any]:
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
    return _run(series, declaration, claim=ASSOCIATION_CLAIM, lags=[int(v) for v in lags],
                cadence_seconds=cadence_seconds, measure=measure, **kwargs)


# ------------------------------------------------------------------- the replication gate

def run_domain_gate(series: ChannelSeries, declaration: DomainDeclaration,
                    protocol: GateProtocol) -> Dict[str, Any]:
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
