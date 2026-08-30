"""TG17.0 four-domain flagship contract and known-answer fixtures.

These fixtures precede the G17 experiment manifest, adapters and orchestrator.  They freeze the
hard cases those later operations must pass without pretending that a configurable multi-domain
experiment already exists.  Construction checks use small elementary oracles that are independent
of the future structural-translation and mining implementation.

The fourth domain is deliberately the existing order-book declaration, not another convenient
grid.  Its licensed user-record source and absent lag policy exercise irregular sampling,
aggregation, no physical metric and a mandatory precedence refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.benchmarks.core import Benchmark, CheckResult, Outcome, register_benchmark, stage_check
from src.benchmarks.seeding import SeedBundle, derive


DOMAIN_NAMES: Tuple[str, ...] = ("reanalysis", "argo", "tess", "order_book")
PLANTED_CASES: Tuple[str, ...] = ("shared_calendar_event", "scale_shifted_motif")
SAFEGUARD_CASES: Tuple[str, ...] = (
    "same_window_unrelated",
    "gap_alias",
    "inadmissible_precedence",
    "family_budget_refusal",
)
ALL_CASES = PLANTED_CASES + SAFEGUARD_CASES

EXPERIMENT_CONTRACT: Dict[str, Any] = {
    "contract_version": "tg17.0-v1",
    "domains": {
        "reanalysis": {
            "source": "ERA5 calendar crop through the existing CDS acquisition contract",
            "shape": "regular gridded field sequence",
        },
        "argo": {
            "source": "official Argo GDAC profiles through the Ifremer ERDDAP view",
            "shape": "sparse irregular profile collection",
        },
        "tess": {
            "source": "MAST light-curve products resolved from sector coverage to UTC interval",
            "shape": "regular light curve with observational gaps",
        },
        "order_book": {
            "source": "licensed user-supplied content-addressed channel record",
            "shape": "irregular aggregated channel record",
        },
    },
    "comparison_modes": {
        "calendar_aligned": {
            "question": "Which declared structures occurred within the same UTC interval?",
            "forbids": ("semantic equivalence", "causality without an external design"),
        },
        "scale_shape_aligned": {
            "question": "Where does a frozen structural shape recur across native scales?",
            "forbids": ("simultaneity", "precedence", "causality"),
        },
    },
    "duration_presets": ("week", "three_months", "six_months"),
    "acceptance_policy": {
        "replications": 200,
        "alpha": 0.05,
        "maximum_null_rejection_rate": 0.075,
        "minimum_planted_detection_rate": 0.80,
        "maximum_family_members": 10_000,
        "maximum_planned_bytes": 4_294_967_296,
        "maximum_planned_runtime_seconds": 3_600,
        "rule": (
            "Every later G17 operation must calibrate its mode-specific nulls and demonstrate "
            "power on the complete declared four-domain family before capability exposure."
        ),
    },
    "claim_boundary": (
        "Shared calendar support permits co-occurrence only; shared structural shape permits "
        "structural comparison only. Neither makes native magnitudes or meanings comparable."
    ),
}


@dataclass(frozen=True)
class NativeDomainFixture:
    domain: str
    semantics: str
    units: str
    sample_times_seconds: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    native_scale_seconds: float
    lag_policy: str
    violations: Tuple[str, ...]
    construction_stream: Tuple[int, ...]


@dataclass(frozen=True)
class FlagshipCase:
    name: str
    comparison_mode: str
    domains: Mapping[str, NativeDomainFixture]
    expected_outcome: str
    oracle: Mapping[str, Any]


@dataclass(frozen=True)
class MultiDomainFlagshipData:
    cases: Mapping[str, FlagshipCase]
    focus: str
    provenance: Mapping[str, Any]


def _clock(domain: str, duration_seconds: float, rng: np.random.Generator) -> np.ndarray:
    cadence = {"reanalysis": 10_800.0, "argo": 43_200.0,
               "tess": 1_800.0, "order_book": 7_200.0}[domain]
    times = np.arange(0.0, duration_seconds, cadence, dtype=np.float64)
    if domain == "order_book":
        jitter = rng.uniform(-0.28 * cadence, 0.28 * cadence, len(times))
        jitter[0] = 0.0
        times = np.maximum.accumulate(times + jitter)
    return times


def _validity(domain: str, times: np.ndarray, duration_seconds: float,
              *, shared_gap: bool = False) -> np.ndarray:
    phase = times / duration_seconds
    valid = np.ones(len(times), dtype=bool)
    if domain == "argo":
        valid[np.arange(len(times)) % 5 == 0] = False
    elif domain == "tess":
        valid[(phase > 0.43) & (phase < 0.49)] = False
    elif domain == "order_book":
        valid[(phase < 0.08) | (phase > 0.92)] = False
    if shared_gap:
        valid[(phase > 0.33) & (phase < 0.48)] = False
        valid[(phase > 0.70) & (phase < 0.79)] = False
    return valid


def _domain_fixture(bundle: SeedBundle, case_name: str, domain: str,
                    duration_seconds: float, signal) -> NativeDomainFixture:
    stream = derive("%s:%s:%s" % (bundle.label, case_name, domain), bundle.root_seed)
    rng = stream.numpy_generator()
    times = _clock(domain, duration_seconds, rng)
    phase = times / duration_seconds
    values = np.asarray(signal(phase, rng), dtype=np.float64)
    shared_gap = case_name == "gap_alias"
    valid = _validity(domain, times, duration_seconds, shared_gap=shared_gap)
    metadata = {
        "reanalysis": ("air temperature anomaly", "K", 10_800.0, "advective", ()),
        "argo": ("practical salinity profile structure", "1e-3", 43_200.0, "none",
                 ("irregular_sampling", "non_stationary_support")),
        "tess": ("relative stellar flux", "dimensionless", 1_800.0, "none",
                 ("no_physical_metric", "no_propagation_speed")),
        "order_book": ("aggregated traded volume", "shares", 7_200.0, "none",
                       ("no_physical_metric", "no_propagation_speed", "aggregated_values",
                        "irregular_sampling", "unordered_channels")),
    }[domain]
    return NativeDomainFixture(
        domain=domain, semantics=metadata[0], units=metadata[1],
        sample_times_seconds=times, values=values, valid_mask=valid,
        native_scale_seconds=metadata[2], lag_policy=metadata[3], violations=metadata[4],
        construction_stream=stream.numpy_seed_entropy)


def _case(bundle: SeedBundle, name: str) -> FlagshipCase:
    duration = 28.0 * 86_400.0

    if name == "shared_calendar_event":
        center, width = 0.61, 0.045

        def signal(phase, rng):
            return np.exp(-0.5 * ((phase - center) / width) ** 2) + 0.035 * rng.standard_normal(len(phase))

        return FlagshipCase(
            name, "calendar_aligned",
            {domain: _domain_fixture(bundle, name, domain, duration, signal)
             for domain in DOMAIN_NAMES},
            "one shared structural event; no semantic, magnitude or causal claim",
            {"event_phase": center, "maximum_peak_separation_seconds": 43_200.0,
             "minimum_peak_contrast": 0.75})

    if name == "scale_shifted_motif":
        widths = {"reanalysis": 0.018, "argo": 0.055, "tess": 0.008,
                  "order_book": 0.095}
        centers = {"reanalysis": 0.22, "argo": 0.47, "tess": 0.71, "order_book": 0.84}
        domains = {}
        for domain in DOMAIN_NAMES:
            width, center = widths[domain], centers[domain]

            def signal(phase, rng, width=width, center=center):
                z = (phase - center) / width
                return (1.0 - 0.24 * z) * np.exp(-0.5 * z * z) + 0.02 * rng.standard_normal(len(phase))

            domains[domain] = _domain_fixture(bundle, name, domain, duration, signal)
        return FlagshipCase(
            name, "scale_shape_aligned", domains,
            "one normalized shape at different native durations; no timing claim",
            {"centers": centers, "widths": widths, "minimum_width_ratio": 5.0})

    if name == "same_window_unrelated":
        def signal(phase, rng):
            return rng.standard_normal(len(phase))

        return FlagshipCase(
            name, "calendar_aligned",
            {domain: _domain_fixture(bundle, name, domain, duration, signal)
             for domain in DOMAIN_NAMES},
            "no cross-domain structural claim despite identical outer interval",
            {"independent_construction_streams": True})

    if name == "gap_alias":
        def signal(phase, rng):
            return rng.standard_normal(len(phase))

        return FlagshipCase(
            name, "calendar_aligned",
            {domain: _domain_fixture(bundle, name, domain, duration, signal)
             for domain in DOMAIN_NAMES},
            "refuse similarity induced by shared observation gaps",
            {"minimum_coverage_correlation": 0.70,
             "independent_value_construction_streams": True})

    if name == "inadmissible_precedence":
        centers = {"reanalysis": 0.35, "argo": 0.42, "tess": 0.49, "order_book": 0.56}
        domains = {}
        for domain in DOMAIN_NAMES:
            center = centers[domain]

            def signal(phase, rng, center=center):
                return np.exp(-0.5 * ((phase - center) / 0.025) ** 2) + 0.02 * rng.standard_normal(len(phase))

            domains[domain] = _domain_fixture(bundle, name, domain, duration, signal)
        return FlagshipCase(
            name, "calendar_aligned", domains,
            "structural association may be tested but four-domain precedence is unavailable",
            {"event_order": tuple(centers), "refusing_domain": "order_book",
             "required_lag_policy": "none"})

    if name == "family_budget_refusal":
        def signal(phase, rng):
            return rng.standard_normal(len(phase))

        return FlagshipCase(
            name, "scale_shape_aligned",
            {domain: _domain_fixture(bundle, name, domain, duration, signal)
             for domain in DOMAIN_NAMES},
            "refuse before acquisition because the declared family exceeds its frozen cap",
            {"declared_family_members": 24_576,
             "maximum_family_members": EXPERIMENT_CONTRACT["acceptance_policy"]["maximum_family_members"]})

    raise KeyError(name)


def build_multidomain_flagship(bundle: SeedBundle, *, focus: str) -> MultiDomainFlagshipData:
    if focus not in ("planted", "safeguards"):
        raise ValueError("focus must be 'planted' or 'safeguards'")
    names = PLANTED_CASES if focus == "planted" else SAFEGUARD_CASES
    return MultiDomainFlagshipData(
        cases={name: _case(bundle, name) for name in names}, focus=focus,
        provenance={**bundle.to_provenance(), "focus": focus,
                    "contract_version": EXPERIMENT_CONTRACT["contract_version"]})


def multidomain_flagship_truth(*, focus: str) -> Dict[str, Any]:
    names = PLANTED_CASES if focus == "planted" else SAFEGUARD_CASES
    return {"focus": focus, "cases": list(names), "complete_required_family": list(ALL_CASES),
            "experiment_contract": EXPERIMENT_CONTRACT,
            "claim_boundary": (
                "Known-answer construction and future acceptance thresholds only; no G17 "
                "acquisition, translation, mining or cross-domain result is produced.")}


def _peak_time(fixture: NativeDomainFixture) -> float:
    values = np.where(fixture.valid_mask, fixture.values, -np.inf)
    return float(fixture.sample_times_seconds[int(np.argmax(values))])


def _binned(fixture: NativeDomainFixture, *, bins: int = 32,
            coverage: bool = False) -> np.ndarray:
    duration = max(float(fixture.sample_times_seconds[-1]), 1.0)
    indices = np.minimum((fixture.sample_times_seconds / duration * bins).astype(int), bins - 1)
    result = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        selected = indices == index
        if coverage:
            result[index] = float(np.mean(fixture.valid_mask[selected])) if np.any(selected) else 0.0
        else:
            admitted = selected & fixture.valid_mask
            if np.any(admitted):
                result[index] = float(np.mean(fixture.values[admitted]))
    return result


def _max_pair_correlation(domains: Mapping[str, NativeDomainFixture], *, coverage: bool) -> float:
    maximum = 0.0
    names = tuple(domains)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            left = _binned(domains[first], coverage=coverage)
            right = _binned(domains[second], coverage=coverage)
            valid = np.isfinite(left) & np.isfinite(right)
            if int(np.sum(valid)) >= 4 and np.std(left[valid]) > 0 and np.std(right[valid]) > 0:
                maximum = max(maximum, abs(float(np.corrcoef(left[valid], right[valid])[0, 1])))
    return maximum


def _common_contract_problems(data: MultiDomainFlagshipData, truth: Dict[str, Any]) -> list[str]:
    problems = []
    if tuple(truth["complete_required_family"]) != ALL_CASES:
        problems.append("the complete required case family drifted")
    contract = truth["experiment_contract"]
    if tuple(contract["domains"]) != DOMAIN_NAMES:
        problems.append("the frozen four-domain order or membership drifted")
    if set(contract["comparison_modes"]) != {"calendar_aligned", "scale_shape_aligned"}:
        problems.append("the two comparison modes are not separately frozen")
    if tuple(contract["duration_presets"]) != ("week", "three_months", "six_months"):
        problems.append("the flagship duration family drifted")
    for case in data.cases.values():
        if tuple(case.domains) != DOMAIN_NAMES:
            problems.append("case %s does not contain the frozen four domains" % case.name)
            continue
        semantics = {fixture.semantics for fixture in case.domains.values()}
        units = {fixture.units for fixture in case.domains.values()}
        streams = {fixture.construction_stream for fixture in case.domains.values()}
        if len(semantics) != 4 or len(units) != 4:
            problems.append("case %s no longer crosses four semantic/unit boundaries" % case.name)
        if len(streams) != 4:
            problems.append("case %s reuses a random construction stream across domains" % case.name)
        for fixture in case.domains.values():
            if not (len(fixture.sample_times_seconds) == len(fixture.values) == len(fixture.valid_mask)):
                problems.append("case %s domain %s has inconsistent record lengths"
                                % (case.name, fixture.domain))
            if not np.all(np.diff(fixture.sample_times_seconds) > 0):
                problems.append("case %s domain %s clock is not strictly increasing"
                                % (case.name, fixture.domain))
    return problems


@stage_check("G17.0.flagship_contract")
def _check_planted(data: MultiDomainFlagshipData, truth: Dict[str, Any]) -> CheckResult:
    problems = _common_contract_problems(data, truth)
    shared = data.cases["shared_calendar_event"]
    peaks = {name: _peak_time(fixture) for name, fixture in shared.domains.items()}
    peak_separation = max(peaks.values()) - min(peaks.values())
    if peak_separation > shared.oracle["maximum_peak_separation_seconds"]:
        problems.append("the planted calendar event peaks are not within one Argo interval")
    contrasts = {}
    for name, fixture in shared.domains.items():
        admitted = fixture.values[fixture.valid_mask]
        contrasts[name] = float(np.max(admitted) - np.median(admitted))
    if min(contrasts.values()) < shared.oracle["minimum_peak_contrast"]:
        problems.append("the shared event is not independently visible in every native record")

    shifted = data.cases["scale_shifted_motif"]
    widths = shifted.oracle["widths"]
    width_ratio = max(widths.values()) / min(widths.values())
    if width_ratio < shifted.oracle["minimum_width_ratio"]:
        problems.append("the scale-shifted motif does not span the frozen native-scale range")
    centers = shifted.oracle["centers"]
    if len({round(float(value), 2) for value in centers.values()}) != 4:
        problems.append("the scale-shifted case accidentally plants calendar alignment")

    measured = {"shared_peak_times_seconds": peaks,
                "shared_peak_separation_seconds": peak_separation,
                "shared_peak_contrasts": contrasts, "native_width_ratio": width_ratio,
                "acceptance_policy": truth["experiment_contract"]["acceptance_policy"]}
    return CheckResult(
        "G17.0.flagship_contract", Outcome.FAIL if problems else Outcome.PASS,
        "; ".join(problems) if problems else
        "two planted cases cross four clocks, units and meanings: one calendar-coincident "
        "event and one common shape deliberately separated in calendar time and native scale",
        measured)


@stage_check("G17.0.flagship_contract")
def _check_safeguards(data: MultiDomainFlagshipData, truth: Dict[str, Any]) -> CheckResult:
    problems = _common_contract_problems(data, truth)
    unrelated = data.cases["same_window_unrelated"]
    unrelated_corr = _max_pair_correlation(unrelated.domains, coverage=False)

    gaps = data.cases["gap_alias"]
    coverage_corr = _max_pair_correlation(gaps.domains, coverage=True)
    gap_value_corr = _max_pair_correlation(gaps.domains, coverage=False)
    if coverage_corr < gaps.oracle["minimum_coverage_correlation"]:
        problems.append("the gap-alias trap lacks shared coverage structure")

    precedence = data.cases["inadmissible_precedence"]
    peak_order = tuple(name for name, _ in sorted(
        ((name, _peak_time(fixture)) for name, fixture in precedence.domains.items()),
        key=lambda item: item[1]))
    refusing = precedence.domains[precedence.oracle["refusing_domain"]]
    if peak_order != precedence.oracle["event_order"]:
        problems.append("the precedence trap no longer has the planted event order")
    if refusing.lag_policy != precedence.oracle["required_lag_policy"]:
        problems.append("the precedence trap's refusing domain acquired a lag policy")

    budget = data.cases["family_budget_refusal"].oracle
    if budget["declared_family_members"] <= budget["maximum_family_members"]:
        problems.append("the resource-refusal case no longer exceeds the frozen family cap")

    measured = {"unrelated_max_binned_correlation": unrelated_corr,
                "gap_coverage_correlation": coverage_corr,
                "gap_value_correlation": gap_value_corr,
                "planted_peak_order": peak_order,
                "precedence_refusing_domain": refusing.domain,
                "declared_family_members": budget["declared_family_members"],
                "maximum_family_members": budget["maximum_family_members"],
                "acceptance_policy": truth["experiment_contract"]["acceptance_policy"]}
    return CheckResult(
        "G17.0.flagship_contract", Outcome.FAIL if problems else Outcome.PASS,
        "; ".join(problems) if problems else
        "four safeguards freeze the same-window null, shared-gap alias, mandatory precedence "
        "refusal and pre-acquisition family-budget refusal",
        measured)


register_benchmark(Benchmark(
    name="multidomain_flagship_planted", kind="cross_domain",
    description="Four-domain planted calendar and normalized-scale structural events across "
                "reanalysis, Argo, TESS and an irregular order-book record.",
    gates=("G17.0.flagship_contract",),
    build=build_multidomain_flagship, known_answer=multidomain_flagship_truth,
    checks=(_check_planted,), params={"focus": "planted"},
))

register_benchmark(Benchmark(
    name="multidomain_flagship_safeguards", kind="cross_domain",
    description="Four-domain null and refusal traps for shared windows, gaps, inadmissible "
                "precedence and unaffordable complete families.",
    gates=("G17.0.flagship_contract",),
    build=build_multidomain_flagship, known_answer=multidomain_flagship_truth,
    checks=(_check_safeguards,), params={"focus": "safeguards"}, is_null=True,
))


__all__ = [
    "ALL_CASES", "DOMAIN_NAMES", "EXPERIMENT_CONTRACT", "PLANTED_CASES",
    "SAFEGUARD_CASES", "FlagshipCase", "MultiDomainFlagshipData", "NativeDomainFixture",
    "build_multidomain_flagship", "multidomain_flagship_truth",
]
