"""Bounded conditional-association estimator (TG16.2).

The declared nuisance is a statistical conditioning role.  It is not inferred to be a
confounder, and a supported result is not a causal or "nuisance-free" claim.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.statistics.multiple_comparisons import adjust


CONDITIONAL_OUTCOMES = ("supported_conditional_association", "unresolved")


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
    return float(-(probabilities * np.log(probabilities)).sum()
                 + (occupied.size - 1) / (2.0 * occupied.sum()))


def _conditional_mi(candidate: np.ndarray, target: np.ndarray,
                    nuisance: np.ndarray) -> float:
    return float(_entropy(candidate, nuisance) + _entropy(target, nuisance)
                 - _entropy(nuisance) - _entropy(candidate, target, nuisance))


def conditional_support(features: Mapping[str, Sequence[float]], target: Sequence[float],
                        nuisance: Sequence[float], *, bins: int,
                        minimum_rows_per_cell: int = 5) -> Dict[str, Any]:
    """Evaluate the frozen overlap/effective-support admission rule."""
    labels = sorted(features)
    arrays = {label: np.asarray(features[label], dtype=np.float64) for label in labels}
    target_array = np.asarray(target, dtype=np.float64)
    nuisance_array = np.asarray(nuisance, dtype=np.float64)
    if target_array.ndim != 1 or nuisance_array.shape != target_array.shape \
            or any(value.shape != target_array.shape for value in arrays.values()):
        raise ValueError("candidates, target and nuisance must be same-length vectors")
    if not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(nuisance_array)) \
            or any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("conditional information requires finite complete vectors")

    target_codes = _codes(target_array, bins)
    nuisance_codes = _codes(nuisance_array, bins)
    feature_codes = {label: _codes(value, bins) for label, value in arrays.items()}
    strata = []
    for level in range(int(bins)):
        selected = nuisance_codes == level
        strata.append({
            "stratum": level, "rows": int(np.sum(selected)),
            "target_levels": int(np.unique(target_codes[selected]).size),
            "candidate_levels": {
                label: int(np.unique(values[selected]).size)
                for label, values in feature_codes.items()},
        })
    candidates = []
    for label, values in feature_codes.items():
        occupied = int(np.unique(np.column_stack(
            (values, target_codes, nuisance_codes)), axis=0).shape[0])
        effective = float(len(values) / occupied) if occupied else 0.0
        candidates.append({
            "candidate": label, "occupied_joint_cells": occupied,
            "effective_rows_per_occupied_joint_cell": effective,
            "admitted": bool(effective >= minimum_rows_per_cell and all(
                row["rows"] >= minimum_rows_per_cell * bins * bins
                and row["target_levels"] >= 2
                and row["candidate_levels"][label] >= 2 for row in strata)),
        })
    design = np.column_stack((np.ones(nuisance_array.size), nuisance_array))
    residual = target_array - design @ np.linalg.lstsq(
        design, target_array, rcond=None)[0]
    centred = nuisance_array - np.mean(nuisance_array)
    quadratic = centred * centred
    quadratic_correlation = (0.0 if np.std(quadratic) == 0 or np.std(residual) == 0
                             else float(np.corrcoef(residual, quadratic)[0, 1]))
    residual_variances = [float(np.var(residual[nuisance_codes == level], ddof=1))
                          for level in range(int(bins))]
    positive_variances = [value for value in residual_variances if value > 0]
    variance_ratio = (float(max(positive_variances) / min(positive_variances))
                      if len(positive_variances) == int(bins) else float("inf"))
    model_admitted = bool(abs(quadratic_correlation) <= 0.20 and variance_ratio <= 4.0)
    admitted = bool(candidates and all(row["admitted"] for row in candidates)
                    and model_admitted)
    return {
        "admitted": admitted, "bins": int(bins),
        "minimum_rows_per_cell": int(minimum_rows_per_cell),
        "minimum_rows_per_nuisance_stratum": int(minimum_rows_per_cell * bins * bins),
        "rule": (
            "Every nuisance stratum has at least five rows per possible candidate/target "
            "cell and at least two occupied target and candidate levels; every candidate "
            "also averages at least five rows per occupied candidate/target/nuisance cell. "
            "The sealed linear target-given-nuisance model additionally refuses absolute "
            "quadratic residual correlation above 0.20 or nuisance-stratum residual variance "
            "ratios above 4."),
        "conditional_model": {
            "form": "target = intercept + slope * declared_nuisance + exchangeable residual",
            "absolute_quadratic_residual_correlation": abs(quadratic_correlation),
            "maximum_absolute_quadratic_residual_correlation": 0.20,
            "residual_variance_by_nuisance_stratum": residual_variances,
            "residual_variance_ratio": variance_ratio,
            "maximum_residual_variance_ratio": 4.0,
            "admitted": model_admitted,
        },
        "strata": strata, "candidates": candidates,
    }


def audit_conditional_information(
        features: Mapping[str, Sequence[float]], target: Sequence[float],
        nuisance: Sequence[float], *, bins: int, permutations: int, seed: int,
        alpha: float, correction: str = "benjamini_yekutieli") -> Dict[str, Any]:
    """Test every frozen candidate with a sealed linear conditional-randomisation null."""
    support = conditional_support(features, target, nuisance, bins=bins)
    if not support["admitted"]:
        raise ValueError("conditional-information overlap/effective support is inadequate")
    labels = sorted(features)
    target_array = np.asarray(target, dtype=np.float64)
    nuisance_array = np.asarray(nuisance, dtype=np.float64)
    target_codes = _codes(target_array, bins)
    nuisance_codes = _codes(nuisance_array, bins)
    design = np.column_stack((np.ones(nuisance_array.size), nuisance_array))
    fitted = design @ np.linalg.lstsq(design, target_array, rcond=None)[0]
    residual = target_array - fitted
    feature_codes = {
        label: _codes(np.asarray(features[label], dtype=np.float64), bins)
        for label in labels}
    rows = []
    p_values = []
    for index, label in enumerate(labels):
        observed = _conditional_mi(feature_codes[label], target_codes, nuisance_codes)
        rng = np.random.default_rng(int(seed) + 104729 * (index + 1))
        exceed = 0
        for _ in range(int(permutations)):
            # The linear model is the sealed conditional randomisation model.  Permuting its
            # residuals preserves the fitted target/nuisance relationship, unlike a global
            # target permutation, before each synthetic target is discretised anew.
            permuted_target = _codes(fitted + rng.permutation(residual), bins)
            null_value = _conditional_mi(
                feature_codes[label], permuted_target, nuisance_codes)
            exceed += int(null_value >= observed)
        p_value = (exceed + 1) / (int(permutations) + 1)
        p_values.append(p_value)
        rows.append({"candidate": label, "conditional_mi_nats": observed,
                     "p_value": p_value})

    corrected = adjust(
        p_values, method=correction, alpha=alpha, n_tests=len(labels), labels=labels)
    for index, row in enumerate(rows):
        supported = bool(corrected["rejected"][index]
                         and row["conditional_mi_nats"] > 0)
        row["q_value"] = corrected["adjusted"][index]
        row["outcome"] = ("supported_conditional_association"
                          if supported else "unresolved")
        row["basis"] = (
            "positive conditional mutual information against the sealed linear conditional-"
            "randomisation null" if supported else
            "the frozen family did not support conditional association")

    return {
        "candidates": rows, "support": support, "correction": corrected,
        "outcome_vocabulary": list(CONDITIONAL_OUTCOMES),
        "claim_boundary": (
            "This estimates conditional association given a researcher-declared nuisance. "
            "The nuisance role is not proof of confounding; the result does not mean "
            "'confounding removed' or 'nuisance-free' and is not causal. Collider and "
            "post-treatment interpretations cannot be decided by this computation."),
    }


__all__ = ["CONDITIONAL_OUTCOMES", "audit_conditional_information",
           "conditional_support"]
