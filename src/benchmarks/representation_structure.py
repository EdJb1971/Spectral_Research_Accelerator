"""TG16.0 paired known-answer sample-table benchmarks.

These fixtures precede the G16 estimators.  They define the cases every later redundancy,
conditional-information or stable-subspace recipe must pass before it can enter the capability
registry.  The checks here verify the planted construction against elementary independent
oracles; they do not pretend that a G16 scientific operation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.analysis_engine.representation_structure import audit_pair_structure
from src.benchmarks.core import Benchmark, CheckResult, Outcome, register_benchmark, stage_check
from src.benchmarks.seeding import SeedBundle, derive


N_SAMPLES = 4096
CALIBRATION_N_SAMPLES = 320
CALIBRATION_BINS = 3
CALIBRATION_PERMUTATIONS = 109
ACCEPTANCE_POLICY: Dict[str, Any] = {
    "replications": 200,
    "alpha": 0.05,
    "maximum_null_rejection_rate": 0.075,
    "minimum_planted_detection_rate": 0.80,
    "rule": (
        "Each future G16 operation must report empirical null calibration and planted-effect "
        "power over the declared replication family before registry exposure."
    ),
}


@dataclass(frozen=True)
class StructureCase:
    name: str
    features: Mapping[str, np.ndarray]
    target: np.ndarray
    nuisance: np.ndarray | None
    interpretation: str


@dataclass(frozen=True)
class RepresentationStructureData:
    cases: Mapping[str, StructureCase]
    focus: str
    provenance: Mapping[str, Any]


def _rng(bundle: SeedBundle, label: str) -> np.random.Generator:
    return derive("%s:%s" % (bundle.label, label), bundle.root_seed).numpy_generator()


def _case(bundle: SeedBundle, name: str, n: int) -> StructureCase:
    rng = _rng(bundle, name)
    if name == "exact_duplicate":
        latent = rng.standard_normal(n)
        return StructureCase(name, {"first": latent, "duplicate": latent.copy()}, latent.copy(),
                             None, "The two candidates are exactly the same random variable.")
    if name == "independent_features":
        return StructureCase(name,
                             {"first": rng.standard_normal(n),
                              "second": rng.standard_normal(n)},
                             rng.standard_normal(n), None,
                             "Candidates and target are mutually independent.")
    if name == "redundant_noisy_copies":
        latent = rng.standard_normal(n)
        first = latent + 0.20 * rng.standard_normal(n)
        second = latent + 0.20 * rng.standard_normal(n)
        return StructureCase(name, {"first": first, "second": second}, latent, None,
                             "Two non-identical candidates carry strongly overlapping signal.")
    if name == "complementary_information":
        first, second = rng.standard_normal((2, n))
        target = first + second + 0.20 * rng.standard_normal(n)
        return StructureCase(name, {"first": first, "second": second}, target, None,
                             "Independent candidates each contribute distinct target signal.")
    if name == "synergistic_pair":
        first = rng.integers(0, 2, n).astype(np.float64)
        second = rng.integers(0, 2, n).astype(np.float64)
        target = np.logical_xor(first.astype(bool), second.astype(bool)).astype(np.float64)
        return StructureCase(name, {"first": first, "second": second}, target, None,
                             "XOR is informative jointly while each candidate is weak alone.")
    if name == "nuisance_only_association":
        nuisance = rng.standard_normal(n)
        first = nuisance + 0.25 * rng.standard_normal(n)
        target = nuisance + 0.25 * rng.standard_normal(n)
        return StructureCase(name, {"candidate": first}, target, nuisance,
                             "Marginal association is entirely carried by the declared nuisance.")
    if name == "signal_survives_conditioning":
        nuisance = rng.standard_normal(n)
        first = 0.40 * nuisance + rng.standard_normal(n)
        target = 1.20 * first + 1.50 * nuisance + 0.30 * rng.standard_normal(n)
        return StructureCase(name, {"candidate": first}, target, nuisance,
                             "Candidate signal remains after conditioning on the nuisance.")
    if name == "conditional_null":
        nuisance = rng.standard_normal(n)
        first = rng.standard_normal(n)
        target = nuisance + 0.25 * rng.standard_normal(n)
        return StructureCase(name, {"candidate": first}, target, nuisance,
                             "The candidate has no marginal or conditional target signal.")
    if name == "collider_counterexample":
        first = rng.standard_normal(n)
        target = rng.standard_normal(n)
        nuisance = first + target + 0.20 * rng.standard_normal(n)
        return StructureCase(name, {"candidate": first}, target, nuisance,
                             "Conditioning on the declared collider creates association.")
    raise KeyError(name)


PLANTED_CASES: Tuple[str, ...] = (
    "exact_duplicate", "redundant_noisy_copies", "complementary_information",
    "synergistic_pair", "signal_survives_conditioning",
)
SAFEGUARD_CASES: Tuple[str, ...] = (
    "independent_features", "nuisance_only_association", "conditional_null",
    "collider_counterexample",
)
ALL_CASES = PLANTED_CASES + SAFEGUARD_CASES


def build_representation_structure(bundle: SeedBundle, *, focus: str,
                                   n: int = N_SAMPLES) -> RepresentationStructureData:
    if focus not in ("planted", "safeguards"):
        raise ValueError("focus must be 'planted' or 'safeguards'")
    names = PLANTED_CASES if focus == "planted" else SAFEGUARD_CASES
    return RepresentationStructureData(
        cases={name: _case(bundle, name, int(n)) for name in names}, focus=focus,
        provenance={**bundle.to_provenance(), "n_samples": int(n), "focus": focus})


def representation_structure_truth(*, focus: str, n: int = N_SAMPLES) -> Dict[str, Any]:
    names = PLANTED_CASES if focus == "planted" else SAFEGUARD_CASES
    return {
        "focus": focus, "n_samples": int(n), "cases": list(names),
        "complete_required_family": list(ALL_CASES),
        "acceptance_policy": dict(ACCEPTANCE_POLICY),
        "claim_boundary": (
            "Known-answer construction and future acceptance thresholds only; no redundancy, "
            "conditional-information or subspace result is produced."
        ),
    }


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1])


def _residual(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(nuisance)), nuisance))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _partial_corr(case: StructureCase) -> float:
    assert case.nuisance is not None
    candidate = next(iter(case.features.values()))
    return _corr(_residual(candidate, case.nuisance),
                 _residual(case.target, case.nuisance))


def _discrete_mi(first: np.ndarray, second: np.ndarray) -> float:
    values_first, inverse_first = np.unique(first, return_inverse=True)
    values_second, inverse_second = np.unique(second, return_inverse=True)
    counts = np.zeros((len(values_first), len(values_second)), dtype=np.float64)
    np.add.at(counts, (inverse_first, inverse_second), 1.0)
    joint = counts / counts.sum()
    product = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / product[mask])))


@stage_check("G16.0.benchmark_contract")
def _check_planted(data: RepresentationStructureData, truth: Dict[str, Any]) -> CheckResult:
    problems = []
    duplicate = data.cases["exact_duplicate"]
    if not np.array_equal(duplicate.features["first"], duplicate.features["duplicate"]):
        problems.append("the exact duplicate differs")
    noisy = data.cases["redundant_noisy_copies"]
    noisy_corr = _corr(noisy.features["first"], noisy.features["second"])
    if np.array_equal(noisy.features["first"], noisy.features["second"]) or noisy_corr < 0.90:
        problems.append("the noisy copies are not distinct and strongly redundant")
    complementary = data.cases["complementary_information"]
    complementary_corrs = [_corr(value, complementary.target)
                           for value in complementary.features.values()]
    if min(complementary_corrs) < 0.60:
        problems.append("both complementary candidates do not carry planted signal")
    synergy = data.cases["synergistic_pair"]
    individual_mi = [_discrete_mi(value, synergy.target) for value in synergy.features.values()]
    joint = 2.0 * synergy.features["first"] + synergy.features["second"]
    joint_mi = _discrete_mi(joint, synergy.target)
    if max(individual_mi) > 0.01 or joint_mi < 0.65:
        problems.append("the XOR pair is not weak individually and strong jointly")
    surviving = _partial_corr(data.cases["signal_survives_conditioning"])
    if surviving < 0.85:
        problems.append("the planted signal does not survive conditioning")
    measured = {"noisy_copy_correlation": noisy_corr,
                "complementary_correlations": complementary_corrs,
                "synergy_individual_mi_nats": individual_mi,
                "synergy_joint_mi_nats": joint_mi,
                "surviving_partial_correlation": surviving,
                "acceptance_policy": truth["acceptance_policy"]}
    return CheckResult("G16.0.benchmark_contract",
                       Outcome.FAIL if problems else Outcome.PASS,
                       "; ".join(problems) if problems else
                       "five planted structures match independent construction oracles; future "
                       "G16 recipes remain unregistered until calibrated and powered",
                       measured)


@stage_check("G16.0.benchmark_contract")
def _check_safeguards(data: RepresentationStructureData,
                      truth: Dict[str, Any]) -> CheckResult:
    problems = []
    independent = data.cases["independent_features"]
    independent_corrs = [_corr(value, independent.target)
                         for value in independent.features.values()]
    if max(abs(value) for value in independent_corrs) > 0.08:
        problems.append("the independent controls have material sample correlation")
    nuisance_only = data.cases["nuisance_only_association"]
    nuisance_marginal = _corr(nuisance_only.features["candidate"], nuisance_only.target)
    nuisance_partial = _partial_corr(nuisance_only)
    if nuisance_marginal < 0.85 or abs(nuisance_partial) > 0.08:
        problems.append("the nuisance-only association is not marginal-only")
    conditional_null = _partial_corr(data.cases["conditional_null"])
    if abs(conditional_null) > 0.08:
        problems.append("the conditional null carries residual association")
    collider = data.cases["collider_counterexample"]
    collider_marginal = _corr(collider.features["candidate"], collider.target)
    collider_partial = _partial_corr(collider)
    if abs(collider_marginal) > 0.08 or collider_partial > -0.85:
        problems.append("the collider does not create the planted conditional association")
    measured = {"independent_correlations": independent_corrs,
                "nuisance_only_marginal_correlation": nuisance_marginal,
                "nuisance_only_partial_correlation": nuisance_partial,
                "conditional_null_partial_correlation": conditional_null,
                "collider_marginal_correlation": collider_marginal,
                "collider_partial_correlation": collider_partial,
                "acceptance_policy": truth["acceptance_policy"]}
    return CheckResult("G16.0.benchmark_contract",
                       Outcome.FAIL if problems else Outcome.PASS,
                       "; ".join(problems) if problems else
                       "four null/conditioning safeguards match independent construction "
                       "oracles, including the collider counterexample",
                       measured)


def _structure_calibration(names: Tuple[str, ...], expected: Mapping[str, str],
                           *, root_seed: int) -> Dict[str, Any]:
    replications = int(ACCEPTANCE_POLICY["replications"])
    counts = {name: 0 for name in names}
    false_claims = {name: 0 for name in names}
    for replication in range(replications):
        bundle = derive("G16.1:%d" % replication, root_seed)
        for case_index, name in enumerate(names):
            case = _case(bundle, name, CALIBRATION_N_SAMPLES)
            if len(case.features) < 2:
                outcome = "unresolved"
            else:
                measured = audit_pair_structure(
                    case.features, case.target, bins=CALIBRATION_BINS,
                    permutations=CALIBRATION_PERMUTATIONS,
                    seed=root_seed + 1000003 * (replication + 1) + 1009 * case_index,
                    alpha=float(ACCEPTANCE_POLICY["alpha"]))
                outcome = measured["pairs"][0]["outcome"]
            counts[name] += int(outcome == expected[name])
            false_claims[name] += int(expected[name] == "unresolved" and outcome != "unresolved")
    return {
        "replications": replications, "n_samples_per_replication": CALIBRATION_N_SAMPLES,
        "bins": CALIBRATION_BINS, "permutations": CALIBRATION_PERMUTATIONS,
        "detection_rates": {name: counts[name] / replications for name in names},
        "false_claim_rates": {name: false_claims[name] / replications for name in names},
    }


@stage_check("G16.1.redundancy_structure")
def _check_redundancy_power(data: RepresentationStructureData,
                            truth: Dict[str, Any]) -> CheckResult:
    expected = {
        "exact_duplicate": "supported_redundancy",
        "redundant_noisy_copies": "supported_redundancy",
        "complementary_information": "supported_complementarity",
        "synergistic_pair": "supported_complementarity",
        "signal_survives_conditioning": "unresolved",
    }
    measured = _structure_calibration(tuple(expected), expected, root_seed=16101)
    minimum = float(ACCEPTANCE_POLICY["minimum_planted_detection_rate"])
    powered = {name: rate for name, rate in measured["detection_rates"].items()
               if name != "signal_survives_conditioning"}
    problems = ["%s power %.3f is below %.3f" % (name, rate, minimum)
                for name, rate in powered.items() if rate < minimum]
    if measured["false_claim_rates"]["signal_survives_conditioning"] > 0:
        problems.append("a single-candidate case produced a pair-structure claim")
    return CheckResult(
        "G16.1.redundancy_structure", Outcome.FAIL if problems else Outcome.PASS,
        "; ".join(problems) if problems else
        "duplicate/noisy-copy redundancy and complementary/XOR structure meet the frozen "
        "power floor; the single-candidate case produces no pair claim",
        measured)


@stage_check("G16.1.redundancy_structure")
def _check_redundancy_nulls(data: RepresentationStructureData,
                            truth: Dict[str, Any]) -> CheckResult:
    expected = {name: "unresolved" for name in SAFEGUARD_CASES}
    measured = _structure_calibration(tuple(expected), expected, root_seed=26101)
    ceiling = float(ACCEPTANCE_POLICY["maximum_null_rejection_rate"])
    problems = ["%s false-claim rate %.3f exceeds %.3f" % (name, rate, ceiling)
                for name, rate in measured["false_claim_rates"].items() if rate > ceiling]
    return CheckResult(
        "G16.1.redundancy_structure", Outcome.FAIL if problems else Outcome.PASS,
        "; ".join(problems) if problems else
        "the independent null stays calibrated and every one-candidate safeguard produces "
        "no pair-structure claim",
        measured)


register_benchmark(Benchmark(
    name="representation_structure_planted", kind="sample_table",
    description="Five planted independent-sample structures spanning duplication, overlap, "
                "complementarity, synergy and conditional survival.",
    gates=("G16.0.benchmark_contract", "G16.1.redundancy_structure"),
    build=build_representation_structure, known_answer=representation_structure_truth,
    checks=(_check_planted, _check_redundancy_power),
    params={"focus": "planted", "n": N_SAMPLES},
))

register_benchmark(Benchmark(
    name="representation_structure_safeguards", kind="sample_table",
    description="Four safeguards spanning independence, nuisance-only association, a "
                "conditional null and a collider counterexample.",
    gates=("G16.0.benchmark_contract", "G16.1.redundancy_structure"),
    build=build_representation_structure, known_answer=representation_structure_truth,
    checks=(_check_safeguards, _check_redundancy_nulls),
    params={"focus": "safeguards", "n": N_SAMPLES},
))


__all__ = [
    "ACCEPTANCE_POLICY", "ALL_CASES", "CALIBRATION_BINS", "CALIBRATION_N_SAMPLES",
    "CALIBRATION_PERMUTATIONS", "N_SAMPLES", "PLANTED_CASES", "SAFEGUARD_CASES",
    "RepresentationStructureData", "StructureCase", "build_representation_structure",
    "representation_structure_truth",
]
