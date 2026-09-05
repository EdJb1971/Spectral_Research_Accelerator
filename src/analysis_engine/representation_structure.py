"""Bounded pairwise information-structure estimator (TG16.1).

The estimator reports candidate structure, not a partial-information decomposition.  In
particular, interaction information is not a count of independent information pieces: it is
used only as a signed, permutation-calibrated discriminator inside a frozen pair family.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.statistics.multiple_comparisons import adjust


STRUCTURE_OUTCOMES = (
    "supported_redundancy", "supported_complementarity", "unresolved")


def candidate_pairs(labels: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """Return the complete, deterministic two-candidate group family."""
    return tuple(combinations(sorted(str(label) for label in labels), 2))


def _codes(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def _entropy(*codes: np.ndarray) -> float:
    key = np.zeros(codes[0].size, dtype=np.int64)
    for values in codes:
        key = key * (int(values.max()) + 1) + values
    counts = np.bincount(key)
    occupied = counts[counts > 0]
    probabilities = occupied.astype(np.float64) / occupied.sum()
    # The occupied-cell Miller-Madow correction matches the existing MI estimator.
    return float(-(probabilities * np.log(probabilities)).sum()
                 + (occupied.size - 1) / (2.0 * occupied.sum()))


def _information(first: np.ndarray, target: np.ndarray) -> float:
    return _entropy(first) + _entropy(target) - _entropy(first, target)


def _joint_information(first: np.ndarray, second: np.ndarray,
                       target: np.ndarray) -> float:
    return (_entropy(first, second) + _entropy(target)
            - _entropy(first, second, target))


def _shuffle_within(values: np.ndarray, strata: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for level in np.unique(strata):
        indices = np.flatnonzero(strata == level)
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def audit_pair_structure(features: Mapping[str, Sequence[float]], target: Sequence[float],
                         *, bins: int, permutations: int, seed: int, alpha: float,
                         correction: str = "benjamini_yekutieli") -> Dict[str, Any]:
    """Measure every frozen candidate pair using one jointly corrected test family.

    Three hypotheses are paid for per pair: positive interaction information, and the
    conditional increment contributed by each member beyond the other.  The redundancy null
    shuffles one candidate within target bins, preserving both candidate/target marginals;
    each increment null shuffles the target within bins of the other candidate.
    """
    labels = sorted(features)
    arrays = {label: np.asarray(features[label], dtype=np.float64) for label in labels}
    target_array = np.asarray(target, dtype=np.float64)
    if any(value.shape != target_array.shape for value in arrays.values()):
        raise ValueError("every candidate and the target must have the same shape")
    if target_array.ndim != 1 or not np.all(np.isfinite(target_array)) \
            or any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("candidate structure requires finite one-dimensional arrays")

    target_codes = _codes(target_array, bins)
    feature_codes = {label: _codes(value, bins) for label, value in arrays.items()}
    rows = []
    raw_p_values = []
    test_labels = []
    for pair_index, (first_name, second_name) in enumerate(candidate_pairs(labels)):
        first, second = feature_codes[first_name], feature_codes[second_name]
        first_mi = _information(first, target_codes)
        second_mi = _information(second, target_codes)
        joint_mi = _joint_information(first, second, target_codes)
        interaction = first_mi + second_mi - joint_mi
        first_increment = joint_mi - second_mi
        second_increment = joint_mi - first_mi
        exact_duplicate = bool(np.array_equal(arrays[first_name], arrays[second_name]))

        rng = np.random.default_rng(int(seed) + 104729 * (pair_index + 1))
        redundant_exceed = first_exceed = second_exceed = 0
        for _ in range(int(permutations)):
            permuted_second = _shuffle_within(second, target_codes, rng)
            null_second_mi = _information(permuted_second, target_codes)
            null_joint = _joint_information(first, permuted_second, target_codes)
            redundant_exceed += int(first_mi + null_second_mi - null_joint >= interaction)

            target_within_second = _shuffle_within(target_codes, second, rng)
            null_first_increment = (
                _joint_information(first, second, target_within_second)
                - _information(second, target_within_second))
            first_exceed += int(null_first_increment >= first_increment)

            target_within_first = _shuffle_within(target_codes, first, rng)
            null_second_increment = (
                _joint_information(first, second, target_within_first)
                - _information(first, target_within_first))
            second_exceed += int(null_second_increment >= second_increment)

        denominator = int(permutations) + 1
        p_values = [(redundant_exceed + 1) / denominator,
                    (first_exceed + 1) / denominator,
                    (second_exceed + 1) / denominator]
        start = len(raw_p_values)
        raw_p_values.extend(p_values)
        test_labels.extend((f"{first_name}|{second_name}:redundancy",
                            f"{first_name}|{second_name}:increment:{first_name}",
                            f"{first_name}|{second_name}:increment:{second_name}"))
        rows.append({
            "candidates": [first_name, second_name], "group_size": 2,
            "exact_duplicate": exact_duplicate,
            "singleton_mi_nats": {first_name: first_mi, second_name: second_mi},
            "joint_mi_nats": joint_mi, "interaction_information_nats": interaction,
            "conditional_increment_nats": {
                first_name: first_increment, second_name: second_increment},
            "p_values": {"redundancy": p_values[0],
                         f"increment:{first_name}": p_values[1],
                         f"increment:{second_name}": p_values[2]},
            "_test_slice": [start, start + 3],
        })

    corrected = adjust(raw_p_values, method=correction, alpha=alpha,
                       n_tests=3 * len(candidate_pairs(labels)), labels=test_labels)
    for row in rows:
        start, stop = row.pop("_test_slice")
        q_values = corrected["adjusted"][start:stop]
        rejected = corrected["rejected"][start:stop]
        first_name, second_name = row["candidates"]
        row["q_values"] = {"redundancy": q_values[0],
                           f"increment:{first_name}": q_values[1],
                           f"increment:{second_name}": q_values[2]}
        if row["exact_duplicate"] or (rejected[0]
                                        and row["interaction_information_nats"] > 0):
            outcome = "supported_redundancy"
            basis = ("exact candidate identity" if row["exact_duplicate"] else
                     "positive interaction information against the frozen conditional null")
        elif rejected[1] and rejected[2]:
            outcome = "supported_complementarity"
            basis = "both candidates add conditional target information beyond the other"
        else:
            outcome = "unresolved"
            basis = "the frozen family did not support redundancy or two-sided complementarity"
        row["outcome"] = outcome
        row["basis"] = basis

    return {
        "pairs": rows,
        "correction": corrected,
        "outcome_vocabulary": list(STRUCTURE_OUTCOMES),
        "claim_boundary": (
            "This is a candidate redundancy structure. Interaction information is not a "
            "partial-information decomposition or a count of independent information pieces; "
            "no outcome instructs removal or selection of a feature."),
    }


__all__ = ["STRUCTURE_OUTCOMES", "audit_pair_structure", "candidate_pairs"]
