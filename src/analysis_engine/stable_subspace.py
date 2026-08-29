"""Bounded generate-only linear stable-subspace search (TG16.3)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.statistics.multiple_comparisons import adjust, required_surrogates


SUBSPACE_OUTCOMES = ("candidate_compact_stable_subspace", "unresolved")
CONFIRMATION_OUTCOMES = ("internally_replicated_candidate", "not_replicated",
                         "not_a_generate_candidate")


def projector(basis: np.ndarray) -> np.ndarray:
    """Return the Euclidean projector identifying a span, independent of its basis."""
    array = np.asarray(basis, dtype=np.float64)
    if array.ndim != 2 or not array.size or np.any(~np.isfinite(array)):
        raise ValueError("basis must be a finite non-empty matrix")
    q, _r = np.linalg.qr(array)
    return q @ q.T


def projector_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Chordal distance between equal-dimensional spans, normalised to [0, 1]."""
    first_p, second_p = projector(first), projector(second)
    if first_p.shape != second_p.shape:
        raise ValueError("subspaces must occupy the same ambient feature space")
    rank = int(round(float(np.trace(first_p))))
    other_rank = int(round(float(np.trace(second_p))))
    if rank != other_rank:
        raise ValueError("subspaces must have the same dimension")
    return float(np.linalg.norm(first_p - second_p, ord="fro") / np.sqrt(2.0 * rank))


def _standardize(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("every feature must vary on the generate partition")
    return (features - mean) / scale, mean, scale


def _regions(nuisance: np.ndarray | None, n: int) -> Tuple[np.ndarray, ...]:
    if nuisance is None:
        return (np.arange(n),)
    cuts = np.quantile(nuisance, (1.0 / 3.0, 2.0 / 3.0))
    labels = np.digitize(nuisance, cuts, right=True)
    return tuple(np.flatnonzero(labels == index) for index in range(3))


def _objective_matrix(x: np.ndarray, y: np.ndarray, nuisance: np.ndarray | None,
                      *, ridge: float, nuisance_penalty: float,
                      variance_weight: float) -> np.ndarray:
    n, width = x.shape
    covariance = x.T @ x / n
    relevance = x.T @ y / n
    instability = np.zeros((width, width), dtype=np.float64)
    regions = _regions(nuisance, n)
    if nuisance is not None:
        for indices in regions:
            regional = x[indices].T @ y[indices] / indices.size
            delta = regional - relevance
            instability += (indices.size / n) * np.outer(delta, delta)
    signal = np.outer(relevance, relevance) + variance_weight * covariance \
        - nuisance_penalty * instability
    metric = covariance + ridge * np.eye(width)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    inverse_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    whitened = inverse_root @ signal @ inverse_root
    # A scalar shift preserves eigenvectors and makes block power iteration well behaved.
    floor = float(np.linalg.eigvalsh(whitened).min())
    if floor <= 0:
        whitened = whitened + (abs(floor) + 1e-9) * np.eye(width)
    return whitened


def _fit_basis(x: np.ndarray, y: np.ndarray, nuisance: np.ndarray | None, *, dimension: int,
               ridge: float, nuisance_penalty: float, variance_weight: float, restarts: int,
               iterations: int, seed: int) -> Tuple[np.ndarray, float]:
    matrix = _objective_matrix(x, y, nuisance, ridge=ridge,
                               nuisance_penalty=nuisance_penalty,
                               variance_weight=variance_weight)
    rng = np.random.default_rng(seed)
    best_basis = None
    best_objective = -np.inf
    for _restart in range(restarts):
        q, _r = np.linalg.qr(rng.standard_normal((x.shape[1], dimension)))
        for _iteration in range(iterations):
            q, _r = np.linalg.qr(matrix @ q)
        value = float(np.trace(q.T @ matrix @ q))
        if value > best_objective:
            best_basis, best_objective = q, value
    assert best_basis is not None
    return best_basis, best_objective


def _explained_fraction(x: np.ndarray, y: np.ndarray, basis: np.ndarray) -> float:
    design = np.column_stack((np.ones(x.shape[0]), x @ basis))
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 1e-15:
        raise ValueError("target must vary on the generate partition")
    return float(max(0.0, 1.0 - np.sum((y - fitted) ** 2) / denominator))


def confirm_stable_subspaces(
        features: Mapping[str, np.ndarray], target: np.ndarray, *,
        preprocessing: Mapping[str, Sequence[float]],
        frozen_subspaces: Sequence[Mapping[str, Any]],
        family_members: Sequence[str], nuisance: np.ndarray | None = None,
        nuisance_region_cuts: Sequence[float] | None = None,
        permutations: int = 4999, seed: int = 16401, alpha: float = 0.05,
        correction: str = "benjamini_yekutieli",
        nuisance_stability_threshold: float = 0.35) -> Dict[str, Any]:
    """Apply frozen generate spans once to an untouched confirmation partition."""
    names = tuple(sorted(features))
    arrays = [np.asarray(features[name], dtype=np.float64) for name in names]
    y = np.asarray(target, dtype=np.float64)
    z = None if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    if not names or y.ndim != 1 or any(value.ndim != 1 or value.size != y.size
                                       for value in arrays) \
            or (z is not None and (z.ndim != 1 or z.size != y.size)):
        raise ValueError("features, target and nuisance must be same-length vectors")
    if y.size < 40 or any(np.any(~np.isfinite(value)) for value in arrays + [y]) \
            or (z is not None and np.any(~np.isfinite(z))):
        raise ValueError("at least 40 finite confirmation rows are required")
    mean = np.asarray(preprocessing.get("mean"), dtype=np.float64)
    scale = np.asarray(preprocessing.get("scale"), dtype=np.float64)
    if mean.shape != (len(names),) or scale.shape != (len(names),) \
            or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) \
            or np.any(scale <= 1e-12):
        raise ValueError("frozen preprocessing must provide one finite mean and scale per feature")
    x = (np.column_stack(arrays) - mean) / scale
    if float(np.std(y, ddof=1)) <= 1e-12:
        raise ValueError("target must vary on the confirmation partition")
    labels = tuple(str(value) for value in family_members)
    rows_by_label = {str(row.get("label")): row for row in frozen_subspaces}
    if not labels or len(set(labels)) != len(labels) or set(rows_by_label) != set(labels) \
            or len(rows_by_label) != len(frozen_subspaces):
        raise ValueError("frozen subspaces must cover the complete searched family exactly once")
    if not isinstance(permutations, (int, np.integer)) or isinstance(permutations, (bool, np.bool_)):
        raise ValueError("permutations must be an integer")
    minimum_permutations = required_surrogates(len(labels), alpha, correction)
    if not minimum_permutations <= permutations <= 9999:
        raise ValueError("permutation count cannot resolve the corrected confirmation family")
    if z is None:
        if nuisance_region_cuts not in (None, (), []):
            raise ValueError("nuisance cuts require a declared nuisance")
        region_indices = (np.arange(y.size),)
        cuts = None
    else:
        cuts_array = np.asarray(nuisance_region_cuts, dtype=np.float64)
        if cuts_array.shape != (2,) or np.any(~np.isfinite(cuts_array)) \
                or not cuts_array[0] < cuts_array[1]:
            raise ValueError("two ordered generate-derived nuisance cuts are required")
        cuts = cuts_array.tolist()
        region_labels = np.digitize(z, cuts_array, right=True)
        region_indices = tuple(np.flatnonzero(region_labels == index) for index in range(3))
        largest_dimension = max(int(rows_by_label[label]["dimension"]) for label in labels)
        minimum = max(10, 2 * (largest_dimension + 1))
        sizes = [int(indices.size) for indices in region_indices]
        if min(sizes) < minimum:
            raise ValueError("every frozen nuisance region needs at least %d confirmation rows" % minimum)
    observed = []
    bases = []
    for label in labels:
        row = rows_by_label[label]
        basis = np.asarray(row.get("basis_for_application"), dtype=np.float64)
        dimension = int(row.get("dimension", 0))
        if basis.shape != (len(names), dimension) or not 1 <= dimension < len(names) \
                or np.any(~np.isfinite(basis)):
            raise ValueError("every frozen basis must match its feature width and dimension")
        recorded_projector = np.asarray(row.get("projector"), dtype=np.float64)
        if recorded_projector.shape != (len(names), len(names)) \
                or not np.allclose(projector(basis), recorded_projector, atol=1e-10):
            raise ValueError("each frozen projector must identify the span of its application basis")
        bases.append(basis)
        observed.append(_explained_fraction(x, y, basis))
    exceed = np.zeros(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _permutation in range(permutations):
        null_target = rng.permutation(y)
        for index, basis in enumerate(bases):
            exceed[index] += int(_explained_fraction(x, null_target, basis) >= observed[index])
    p_values = ((exceed + 1) / (permutations + 1)).tolist()
    corrected = adjust(p_values, method=correction, alpha=alpha,
                       n_tests=len(labels), labels=list(labels))
    results = []
    for index, label in enumerate(labels):
        frozen = rows_by_label[label]
        regional = [_explained_fraction(x[indices], y[indices], bases[index])
                    for indices in region_indices]
        regional_range = (float(max(regional) - min(regional)) if z is not None else None)
        region_stable = bool(z is None or regional_range <= nuisance_stability_threshold)
        generated = frozen.get("outcome") == "candidate_compact_stable_subspace"
        replicated = bool(generated and corrected["rejected"][index] and region_stable)
        outcome = ("internally_replicated_candidate" if replicated else
                   ("not_replicated" if generated else "not_a_generate_candidate"))
        results.append({
            "label": label, "dimension": int(frozen["dimension"]),
            "regularization": float(frozen["regularization"]),
            "generate_outcome": frozen["outcome"],
            "confirmation_explained_fraction": observed[index],
            "p_value": p_values[index], "q_value": corrected["adjusted"][index],
            "nuisance_region_explained_fractions": regional,
            "nuisance_region_range": regional_range,
            "nuisance_region_stable": region_stable, "outcome": outcome,
        })
    return {
        "features": list(names), "family_members": list(labels),
        "correction_unit": len(labels), "correction": correction, "alpha": alpha,
        "permutations": permutations,
        "nuisance_regions": {
            "definition": ("generate-derived tertile cut points applied unchanged"
                           if z is not None else "not applicable: no nuisance declared"),
            "cuts": cuts, "confirmation_sizes": [int(value.size) for value in region_indices],
            "explained_fraction_range_threshold": nuisance_stability_threshold,
        },
        "subspaces": results,
        "internally_replicated_candidates": [
            row for row in results if row["outcome"] == "internally_replicated_candidate"],
        "outcome_vocabulary": list(CONFIRMATION_OUTCOMES),
        "claim_boundary": (
            "Internal replication on one reserved partition of this dataset. It is not an "
            "external replication certificate, an optimal or causal representation, or an "
            "instruction to use or remove features."),
    }


def _region_diagnostics(x: np.ndarray, y: np.ndarray, nuisance: np.ndarray | None,
                        basis: np.ndarray) -> Dict[str, Any]:
    regions = _regions(nuisance, x.shape[0])
    minimum = max(20, 4 * (basis.shape[1] + 1))
    sizes = [int(indices.size) for indices in regions]
    if min(sizes) < minimum:
        raise ValueError("every declared-nuisance region needs at least %d generate rows" % minimum)
    scores = [_explained_fraction(x[indices], y[indices], basis) for indices in regions]
    score_range = float(max(scores) - min(scores))
    return {
        "criterion": ("generate-tertile explained-fraction range <= 0.35"
                      if nuisance is not None else "not applicable: no nuisance declared"),
        "region_sizes": sizes, "explained_fractions": scores,
        "range": score_range if nuisance is not None else None,
        "admitted": bool(nuisance is None or score_range <= 0.35),
        "claim_boundary": (
            "Generate-only descriptive stability across researcher-declared nuisance regions; "
            "it is not conditional information and is not evidence that nuisance was removed."),
    }


def generate_stable_subspaces(
        features: Mapping[str, np.ndarray], target: np.ndarray,
        nuisance: np.ndarray | None = None, *, dimensions: Sequence[int] = (1,),
        regularizations: Sequence[float] = (0.01, 0.1, 1.0), permutations: int = 499,
        restarts: int = 6, iterations: int = 48, perturbations: int = 6,
        perturbation_scale: float = 0.10, stability_threshold: float = 0.10,
        nuisance_penalty: float = 1.0, variance_weight: float = 0.05,
        seed: int = 16301, alpha: float = 0.05,
        correction: str = "benjamini_yekutieli") -> Dict[str, Any]:
    """Enumerate a frozen linear family and return generate-only candidate spans."""
    names = tuple(sorted(features))
    if len(names) < 2 or len(names) > 6:
        raise ValueError("stable-subspace generation needs two to six features")
    arrays = [np.asarray(features[name], dtype=np.float64) for name in names]
    y = np.asarray(target, dtype=np.float64)
    z = None if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    if y.ndim != 1 or any(value.ndim != 1 or value.size != y.size for value in arrays) \
            or (z is not None and (z.ndim != 1 or z.size != y.size)):
        raise ValueError("features, target and nuisance must be same-length vectors")
    if y.size < 80 or any(np.any(~np.isfinite(value)) for value in arrays + [y]) \
            or (z is not None and np.any(~np.isfinite(z))):
        raise ValueError("at least 80 finite generate rows are required")
    try:
        dims = tuple(sorted(int(value) for value in dimensions))
        regs = tuple(sorted(float(value) for value in regularizations))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dimensions and regularizations must be finite numeric sequences") from exc
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))
           or float(value) != int(value) for value in dimensions):
        raise ValueError("dimensions must contain integers")
    if not dims or len(set(dims)) != len(dims) or min(dims) < 1 or max(dims) >= len(names):
        raise ValueError("dimensions must be distinct and lie from 1 through feature_count - 1")
    if not regs or len(regs) > 4 or len(set(regs)) != len(regs) \
            or np.any(~np.isfinite(regs)) or min(regs) <= 0 or max(regs) > 10:
        raise ValueError("regularizations must be distinct positive values")
    integer_values = (permutations, restarts, iterations, perturbations, seed)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in integer_values):
        raise ValueError("permutations, optimizer counts and seed must be integers")
    if not 2 <= restarts <= 12 or not 16 <= iterations <= 128 \
            or not 3 <= perturbations <= 20:
        raise ValueError("optimizer and perturbation counts are outside the bounded recipe")
    numeric_values = (perturbation_scale, stability_threshold, nuisance_penalty,
                      variance_weight, alpha)
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))
           for value in numeric_values):
        raise ValueError("objective, stability and alpha values must be finite numbers")
    if not 0 < perturbation_scale <= 0.1 or not 0 < stability_threshold <= 0.5 \
            or not 0 <= nuisance_penalty <= 10 or not 0 <= variance_weight <= 1 \
            or not 0 < alpha < 1:
        raise ValueError("objective, stability or alpha values are outside the bounded recipe")
    minimum_permutations = required_surrogates(len(dims) * len(regs), alpha, correction)
    if not minimum_permutations <= permutations <= 9999:
        raise ValueError("permutation count cannot resolve the corrected frozen family")
    x, mean, scale = _standardize(np.column_stack(arrays))
    target_scale = float(y.std(ddof=1))
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        raise ValueError("target must vary on the generate partition")
    y = (y - y.mean()) / target_scale
    if z is not None:
        if np.std(z, ddof=1) <= 1e-12:
            raise ValueError("declared nuisance must vary on the generate partition")
        z = (z - z.mean()) / z.std(ddof=1)
        minimum_region = max(20, 4 * (max(dims) + 1))
        region_sizes = [indices.size for indices in _regions(z, z.size)]
        if min(region_sizes) < minimum_region:
            raise ValueError("every nuisance region needs at least %d generate rows" %
                             minimum_region)
    members = [(dimension, ridge) for dimension in dims for ridge in regs]
    observed = []
    for index, (dimension, ridge) in enumerate(members):
        basis, objective = _fit_basis(
            x, y, z, dimension=dimension, ridge=ridge,
            nuisance_penalty=nuisance_penalty, variance_weight=variance_weight,
            restarts=restarts, iterations=iterations, seed=seed + 1009 * index)
        observed.append((basis, objective, _explained_fraction(x, y, basis)))
    exceed = np.zeros(len(members), dtype=np.int64)
    rng = np.random.default_rng(seed + 7919)
    for permutation in range(permutations):
        null_target = rng.permutation(y)
        for index, (dimension, ridge) in enumerate(members):
            null_basis, _objective = _fit_basis(
                x, null_target, z, dimension=dimension, ridge=ridge,
                nuisance_penalty=nuisance_penalty, variance_weight=variance_weight,
                restarts=restarts, iterations=iterations,
                seed=seed + 104729 * (permutation + 1) + 1009 * index)
            exceed[index] += int(_explained_fraction(x, null_target, null_basis)
                                 >= observed[index][2])
    p_values = ((exceed + 1) / (permutations + 1)).tolist()
    labels = ["dimension=%d;ridge=%g" % member for member in members]
    corrected = adjust(p_values, method=correction, alpha=alpha,
                       n_tests=len(members), labels=labels)
    rows = []
    for index, ((dimension, ridge), (basis, objective, explained)) in enumerate(
            zip(members, observed)):
        distances = []
        perturbation_rows = []
        perturbation_stream_seed = seed + 1000003 + 1009 * index
        perturb_rng = np.random.default_rng(perturbation_stream_seed)
        for perturbation in range(perturbations):
            perturbed_x = x + perturbation_scale * perturb_rng.standard_normal(x.shape)
            perturbed_y = y + perturbation_scale * perturb_rng.standard_normal(y.shape)
            optimizer_seed = seed + 2000003 + 65537 * perturbation + 1009 * index
            perturbed_basis, _value = _fit_basis(
                perturbed_x, perturbed_y, z, dimension=dimension, ridge=ridge,
                nuisance_penalty=nuisance_penalty, variance_weight=variance_weight,
                restarts=restarts, iterations=iterations,
                seed=optimizer_seed)
            distance = projector_distance(basis, perturbed_basis)
            distances.append(distance)
            perturbation_rows.append({
                "draw_index": perturbation,
                "perturbation_stream_seed": perturbation_stream_seed,
                "optimizer_seed": optimizer_seed,
                "projector_distance": distance,
            })
        region = _region_diagnostics(x, y, z, basis)
        stable = bool(max(distances) <= stability_threshold and region["admitted"])
        supported = bool(corrected["rejected"][index] and stable)
        rows.append({
            "label": labels[index], "dimension": dimension, "regularization": ridge,
            "projector": projector(basis).tolist(),
            "basis_for_application": basis.tolist(),
            "objective": objective, "generate_explained_fraction": explained,
            "p_value": p_values[index], "q_value": corrected["adjusted"][index],
            "perturbation_projector_distances": distances,
            "multi_seed_perturbations": perturbation_rows,
            "maximum_projector_distance": float(max(distances)),
            "nuisance_stability": region,
            "outcome": ("candidate_compact_stable_subspace" if supported else "unresolved"),
        })
    candidates = [row for row in rows
                  if row["outcome"] == "candidate_compact_stable_subspace"]
    candidates.sort(key=lambda row: (row["dimension"], row["q_value"],
                                     row["regularization"]))
    return {
        "features": list(names),
        "preprocessing": {"method": "generate-only standardization",
                          "mean": mean.tolist(), "scale": scale.tolist()},
        "family": {"dimensions": list(dims), "regularizations": list(regs),
                   "members": labels, "n_tests": len(members),
                   "correction": correction, "permutations": permutations},
        "search": {"objective": "regularized supervised covariance with nuisance-region penalty",
                   "nuisance_penalty": nuisance_penalty,
                   "variance_weight": variance_weight,
                   "optimizer": "seeded block power iteration", "restarts": restarts,
                   "iterations": iterations, "seed": seed,
                   "seed_derivation": (
                       "fixed member, permutation, perturbation-stream and optimizer offsets")},
        "stability": {"perturbations": perturbations,
                      "perturbation_scale": perturbation_scale,
                      "projector_distance_threshold": stability_threshold},
        "subspaces": rows, "compact_candidates": candidates,
        "outcome_vocabulary": list(SUBSPACE_OUTCOMES),
        "claim_boundary": (
            "These are generate-partition candidate compact stable subspaces, identified by "
            "their projectors. They are not optimal representations, confirmed findings, "
            "causal structures or instructions to use or remove features."),
    }


__all__ = ["CONFIRMATION_OUTCOMES", "SUBSPACE_OUTCOMES", "confirm_stable_subspaces",
           "generate_stable_subspaces", "projector", "projector_distance"]
