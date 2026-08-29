"""Generic file probing and explicit unordered sample declarations (G14.1).

The probe reports storage facts only. It may say that a column is numeric or monotone; it never
turns a name such as ``target`` or ``time`` into scientific meaning. Roles are a separate,
content-bound declaration supplied by the researcher (E14).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis_engine.cross_scale import mutual_information
from src.analysis_engine.conditional_information import (
    audit_conditional_information, conditional_support)
from src.analysis_engine.representation_structure import (audit_pair_structure,
                                                           candidate_pairs)
from src.analysis_engine.stable_subspace import generate_stable_subspaces
from src.core.errors import InvalidParameterError
from src.core.dataset_capabilities import build_profile
from src.statistics.multiple_comparisons import adjust, required_surrogates

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 256
MISSING = {"", "na", "n/a", "null", "none", "nan"}
SAMPLE_ROLES = ("sample_id", "target", "nuisance", "feature", "group", "ordering", "ignore")
SAMPLE_RELATIONSHIPS = ("independent", "grouped", "ordered")
AUDIT_REPRESENTATIONS = ("identity", "pca")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _delimiter(value: str) -> str:
    if value == "\\t":
        value = "\t"
    if value not in (",", "\t", ";", "|"):
        raise InvalidParameterError("delimiter", value, "one of comma, tab, semicolon or pipe")
    return value


def _rows(payload: bytes, delimiter: str) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InvalidParameterError("upload bytes", len(payload),
                                    "at most %d bytes" % MAX_UPLOAD_BYTES)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidParameterError("file encoding", "not UTF-8",
                                    "UTF-8 CSV/TSV for this ingress adapter") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=_delimiter(delimiter))
    try:
        header = tuple(value.strip() for value in next(reader))
    except StopIteration as exc:
        raise InvalidParameterError("file", "empty", "a header and at least one data row") from exc
    if not header or len(header) > MAX_COLUMNS or any(not value for value in header) \
            or len(set(header)) != len(header):
        raise InvalidParameterError("header", list(header),
                                    "1-256 distinct non-empty column names")
    rows = []
    for number, row in enumerate(reader, start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise InvalidParameterError("row %d" % number, len(row),
                                        "%d columns, matching the header" % len(header))
        rows.append(tuple(value.strip() for value in row))
        if len(rows) > MAX_ROWS:
            raise InvalidParameterError("row count", len(rows), "at most %d" % MAX_ROWS)
    if not rows:
        raise InvalidParameterError("file", "header only", "at least one data row")
    return header, tuple(rows)


def _column_facts(values: Sequence[str]) -> Dict[str, Any]:
    present = [value for value in values if value.lower() not in MISSING]
    numeric = []
    for value in present:
        try:
            parsed = float(value)
        except ValueError:
            numeric = []
            break
        if not math.isfinite(parsed):
            numeric = []
            break
        numeric.append(parsed)
    if len(numeric) == len(present) and present:
        array = np.asarray(numeric, dtype=np.float64)
        storage_type = "numeric"
        monotone = bool(array.size > 1 and np.all(np.diff(array) > 0))
        minimum, maximum = float(array.min()), float(array.max())
    else:
        lowered = {value.lower() for value in present}
        storage_type = "boolean" if present and lowered <= {"true", "false"} else "text"
        monotone, minimum, maximum = False, None, None
    return {"storage_type": storage_type, "missing_count": len(values) - len(present),
            "distinct_count": len(set(present)), "strictly_increasing": monotone,
            "minimum": minimum, "maximum": maximum}


def probe_delimited(payload: bytes, *, filename: str, delimiter: str = ",") -> Dict[str, Any]:
    header, rows = _rows(payload, delimiter)
    columns = []
    for index, name in enumerate(header):
        columns.append({"name": name, **_column_facts([row[index] for row in rows])})
    return {"schema": "spectral.file-probe.v1", "filename": filename,
            "content_sha256": _sha(payload), "format": "delimited_text",
            "delimiter": _delimiter(delimiter), "n_rows": len(rows),
            "n_columns": len(header), "columns": columns,
            "semantic_inference": False,
            "obligations": [
                "Declare every column role; names are not interpreted (E14).",
                "Declare whether samples are independent, grouped or ordered.",
                "A target/nuisance search family must be frozen before analysis (R18)."],
            "claim_boundary": "Storage facts only; no scientific role or analysis was inferred."}


@dataclass(frozen=True)
class SampleTableDeclaration:
    roles: Mapping[str, str]
    sample_relationship: str
    units: Mapping[str, str]

    def validate(self, probe: Mapping[str, Any]) -> None:
        columns = [value["name"] for value in probe["columns"]]
        if set(self.roles) != set(columns):
            raise InvalidParameterError("declared columns", sorted(self.roles),
                                        "exactly the probed columns %s" % columns)
        unknown = sorted(set(self.roles.values()) - set(SAMPLE_ROLES))
        if unknown:
            raise InvalidParameterError("column roles", unknown,
                                        "roles from %s" % (SAMPLE_ROLES,))
        if self.sample_relationship not in SAMPLE_RELATIONSHIPS:
            raise InvalidParameterError("sample_relationship", self.sample_relationship,
                                        "one of %s" % (SAMPLE_RELATIONSHIPS,))
        by_role = {role: [name for name, value in self.roles.items() if value == role]
                   for role in SAMPLE_ROLES}
        if len(by_role["target"]) != 1 or not by_role["feature"]:
            raise InvalidParameterError("roles", by_role,
                                        "exactly one target and at least one feature")
        if len(by_role["sample_id"]) > 1 or len(by_role["group"]) > 1 \
                or len(by_role["ordering"]) > 1:
            raise InvalidParameterError("roles", by_role,
                                        "at most one sample_id, group and ordering column")
        expected_ordering = 1 if self.sample_relationship == "ordered" else 0
        if len(by_role["ordering"]) != expected_ordering:
            raise InvalidParameterError("ordering role", by_role["ordering"],
                                        ("exactly one ordering column" if expected_ordering
                                         else "no ordering column for independent/grouped samples"))
        if self.sample_relationship == "grouped" and len(by_role["group"]) != 1:
            raise InvalidParameterError("group role", by_role["group"],
                                        "exactly one group column for grouped samples")
        numeric = {row["name"] for row in probe["columns"]
                   if row["storage_type"] == "numeric"}
        required_numeric = set(by_role["target"] + by_role["feature"] + by_role["nuisance"])
        if not required_numeric <= numeric:
            raise InvalidParameterError("numeric analysis columns",
                                        sorted(required_numeric - numeric),
                                        "numeric target, feature and nuisance columns")
        missing_units = sorted(name for name in required_numeric
                               if not str(self.units.get(name, "")).strip())
        if missing_units:
            raise InvalidParameterError("units", missing_units,
                                        "an explicit unit or 'dimensionless' for every numeric role")

    def canonical(self) -> Dict[str, Any]:
        return {"roles": dict(sorted(self.roles.items())),
                "sample_relationship": self.sample_relationship,
                "units": dict(sorted(self.units.items()))}


def sample_table_capability_profile(payload: bytes, *, filename: str, delimiter: str,
                                    declaration: SampleTableDeclaration) -> Dict[str, Any]:
    """Route a declared sample table without pretending grouped/ordered rows are independent."""
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    relationship = declaration.sample_relationship
    return build_profile(
        kind="sample_table", phase="declared", identity=probe["content_sha256"],
        facts={
            "sample_table": True, "channel_series": False, "profile_collection": False,
            "spatial_grid_2d": False, "physical_metric": False,
            "ordered_time_axis": relationship == "ordered", "regular_cadence": None,
            "irregular_support": None, "transform_compatible": False,
            "precedence_admissible": False, "independent_samples": relationship == "independent",
            "declared_nuisance": sum(
                role == "nuisance" for role in declaration.roles.values()) == 1,
        },
        basis={"filename": filename, "declaration": declaration.canonical(),
               "semantic_inference": False, "n_rows": probe["n_rows"]})


def _numeric_table(payload: bytes, delimiter: str, declaration: SampleTableDeclaration
                   ) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...]]:
    header, rows = _rows(payload, delimiter)
    values: Dict[str, np.ndarray] = {}
    for index, name in enumerate(header):
        if declaration.roles[name] not in ("target", "feature", "nuisance", "ordering"):
            continue
        parsed = []
        for row in rows:
            raw = row[index]
            parsed.append(float("nan") if raw.lower() in MISSING else float(raw))
        values[name] = np.asarray(parsed, dtype=np.float64)
    return values, header


def require_independent_samples(declaration: SampleTableDeclaration, *, recipe: str) -> None:
    """Authoritative admission rule for row-random independent-sample recipes.

    G16 starts beside the existing sample-table spine.  Its first recipes must reuse this
    refusal rather than each inventing a subtly different grouped/ordered fallback.
    """
    relationship = declaration.sample_relationship
    if relationship not in SAMPLE_RELATIONSHIPS:
        raise InvalidParameterError("sample_relationship", relationship,
                                    "one of %s" % (SAMPLE_RELATIONSHIPS,))
    if relationship == "independent":
        return
    needed = ("group-held-out confirmation with benchmarked nulls"
              if relationship == "grouped"
              else "blocked and embargoed confirmation with benchmarked nulls")
    raise InvalidParameterError(
        "sample_relationship", relationship,
        "'independent' for %s. %s data require %s; treating dependent rows as exchangeable "
        "would invalidate the null and leak samples across partitions." %
        (recipe, relationship.capitalize(), needed))


def plan_redundancy_structure_audit(payload: bytes, *, filename: str, delimiter: str,
                                    declaration: SampleTableDeclaration, bins: int = 4,
                                    permutations: int = 4999, seed: int = 16101,
                                    alpha: float = 0.05) -> Dict[str, Any]:
    """Seal the complete raw-feature pair family before measuring any relationship."""
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="the redundancy structure audit")
    features = sorted(name for name, role in declaration.roles.items() if role == "feature")
    if not 2 <= len(features) <= 6:
        raise InvalidParameterError(
            "feature family", len(features),
            "2-6 predeclared raw features; this bounded recipe enumerates every pair")
    if isinstance(bins, bool) or not 2 <= int(bins) <= 4:
        raise InvalidParameterError("bins", bins, "an integer from 2 through 4")
    if not 0 < float(alpha) < 1:
        raise InvalidParameterError("alpha", alpha, "a value in (0, 1)")
    pairs = [[first, second] for first, second in candidate_pairs(features)]
    n_tests = 3 * len(pairs)
    minimum = required_surrogates(n_tests, float(alpha), "benjamini_yekutieli")
    if not minimum <= int(permutations) <= 9999:
        raise InvalidParameterError(
            "permutations", permutations,
            "%d-9999 fixed conditional permutations for this %d-test family; the smallest "
            "attainable p-value must survive Benjamini-Yekutieli correction"
            % (minimum, n_tests))
    minimum_samples = 5 * int(bins) ** 3
    if int(probe["n_rows"]) < minimum_samples:
        raise InvalidParameterError(
            "sample count", probe["n_rows"],
            "at least %d rows (five per possible %d-bin joint cell)" %
            (minimum_samples, int(bins)))
    body = {
        "schema": "spectral.redundancy-structure-plan.v1",
        "content_sha256": probe["content_sha256"],
        "declaration": declaration.canonical(),
        "candidates": features, "candidate_groups": pairs,
        "group_policy": "all unordered raw-feature pairs; group size exactly 2",
        "estimator": "equiprobable-bin Miller-Madow mutual information",
        "bins": int(bins), "neighbourhood_policy": None,
        "null": ("candidate-within-target-bin permutation for positive interaction "
                 "information; target-within-other-candidate-bin permutation for each "
                 "conditional increment"),
        "permutations": int(permutations), "seed": int(seed),
        "alpha": float(alpha), "correction": "benjamini_yekutieli",
        "tests_per_group": 3, "n_tests": n_tests,
    }
    body["plan_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        **body, "probe": probe,
        "claim_boundary": (
            "This freezes every pair, estimator, discretisation, conditional null, seed, "
            "alpha and correction before enumeration. It computes no structure result."),
    }


def run_redundancy_structure_audit(payload: bytes, *, filename: str, delimiter: str,
                                   plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the exact sealed TG16.1 pair family without selecting or deleting features."""
    expected = dict(plan)
    supplied_digest = expected.pop("plan_sha256", "")
    expected.pop("probe", None)
    expected.pop("claim_boundary", None)
    actual_digest = hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied_digest != actual_digest:
        raise InvalidParameterError("plan_sha256", supplied_digest,
                                    "the digest of the complete frozen structure plan")
    if expected.get("schema") != "spectral.redundancy-structure-plan.v1":
        raise InvalidParameterError("schema", expected.get("schema"),
                                    "spectral.redundancy-structure-plan.v1")
    if _sha(payload) != expected["content_sha256"]:
        raise InvalidParameterError("content_sha256", _sha(payload),
                                    "the exact file bytes the structure plan sealed")
    declaration = SampleTableDeclaration(**expected["declaration"])
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="the redundancy structure audit")
    columns, _header = _numeric_table(payload, delimiter, declaration)
    target_name = next(name for name, role in declaration.roles.items() if role == "target")
    required = [target_name] + list(expected["candidates"])
    missing = {name: int(np.sum(~np.isfinite(columns[name]))) for name in required}
    if any(missing.values()):
        raise InvalidParameterError("missing analysis values", missing,
                                    "complete target and feature columns for this recipe")
    actual_candidates = sorted(name for name, role in declaration.roles.items()
                               if role == "feature")
    actual_pairs = [[first, second] for first, second in candidate_pairs(actual_candidates)]
    if actual_candidates != expected["candidates"] or actual_pairs != expected["candidate_groups"]:
        raise InvalidParameterError("candidate family", actual_pairs,
                                    "the exact complete pair family sealed by the plan")
    measured = audit_pair_structure(
        {name: columns[name] for name in actual_candidates}, columns[target_name],
        bins=int(expected["bins"]), permutations=int(expected["permutations"]),
        seed=int(expected["seed"]), alpha=float(expected["alpha"]),
        correction=str(expected["correction"]))
    return {
        "schema": "spectral.redundancy-structure-audit.v1",
        "plan_sha256": supplied_digest, "content_sha256": expected["content_sha256"],
        "target": target_name, "sample_relationship": declaration.sample_relationship,
        "family": {"candidates": actual_candidates, "candidate_groups": actual_pairs,
                   "group_policy": expected["group_policy"],
                   "tests_per_group": expected["tests_per_group"],
                   "n_tests": expected["n_tests"], "correction": expected["correction"],
                   "alpha": expected["alpha"], "permutations": expected["permutations"]},
        "method": {key: expected[key] for key in
                   ("estimator", "bins", "neighbourhood_policy", "null", "seed")},
        "structure_map": measured["pairs"],
        "outcome_vocabulary": measured["outcome_vocabulary"],
        "stored": False, "rung_moved": False,
        "claim_boundary": measured["claim_boundary"],
    }


def plan_conditional_information_audit(
        payload: bytes, *, filename: str, delimiter: str,
        declaration: SampleTableDeclaration, bins: int = 3,
        permutations: int = 4999, seed: int = 16201,
        alpha: float = 0.05) -> Dict[str, Any]:
    """Seal the complete raw-feature conditional-information family and support rule."""
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="the conditional-information audit")
    features = sorted(name for name, role in declaration.roles.items() if role == "feature")
    nuisances = sorted(name for name, role in declaration.roles.items() if role == "nuisance")
    if not 1 <= len(features) <= 6:
        raise InvalidParameterError("feature family", len(features),
                                    "1-6 predeclared raw features")
    if len(nuisances) != 1:
        raise InvalidParameterError(
            "nuisance role", nuisances,
            "exactly one researcher-declared nuisance in this bounded recipe")
    if isinstance(bins, bool) or not 2 <= int(bins) <= 4:
        raise InvalidParameterError("bins", bins, "an integer from 2 through 4")
    if not 0 < float(alpha) < 1:
        raise InvalidParameterError("alpha", alpha, "a value in (0, 1)")
    minimum = required_surrogates(
        len(features), float(alpha), "benjamini_yekutieli")
    if not minimum <= int(permutations) <= 9999:
        raise InvalidParameterError(
            "permutations", permutations,
            "%d-9999 fixed conditional-randomisation draws for this %d-test family; "
            "the smallest attainable p-value must survive Benjamini-Yekutieli correction"
            % (minimum, len(features)))
    minimum_samples = 5 * int(bins) ** 3
    if int(probe["n_rows"]) < minimum_samples:
        raise InvalidParameterError(
            "sample count", probe["n_rows"],
            "at least %d rows under the frozen five-per-cell support rule"
            % minimum_samples)
    columns, _header = _numeric_table(payload, delimiter, declaration)
    target = next(name for name, role in declaration.roles.items() if role == "target")
    required = [target, nuisances[0], *features]
    missing = {name: int(np.sum(~np.isfinite(columns[name]))) for name in required}
    if any(missing.values()):
        raise InvalidParameterError(
            "missing analysis values", missing,
            "complete target, nuisance and feature columns for this recipe")
    support = conditional_support(
        {name: columns[name] for name in features}, columns[target], columns[nuisances[0]],
        bins=int(bins))
    if not support["admitted"]:
        raise InvalidParameterError(
            "conditional overlap/effective support", support,
            support["rule"])
    body = {
        "schema": "spectral.conditional-information-plan.v1",
        "content_sha256": probe["content_sha256"],
        "declaration": declaration.canonical(), "target": target,
        "nuisance": nuisances[0], "candidates": features,
        "family_policy": "every declared raw feature conditioned on the one declared nuisance",
        "quantity": "I(candidate; target | declared nuisance)",
        "estimator": "equiprobable-bin Miller-Madow conditional mutual information",
        "bins": int(bins), "support_admission": support,
        "null": ("linear conditional-randomisation model: fit target on the declared "
                 "nuisance, permute residuals, reconstruct and rediscretise the target; "
                 "preserves the fitted target/nuisance relationship"),
        "permutations": int(permutations), "seed": int(seed),
        "alpha": float(alpha), "correction": "benjamini_yekutieli",
        "n_tests": len(features),
    }
    body["plan_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        **body, "probe": probe,
        "claim_boundary": (
            "This freezes the complete candidate family, declared nuisance, estimator, "
            "support rule, conditional null, seed, alpha and correction. It computes no "
            "conditional-association result."),
    }


def run_conditional_information_audit(
        payload: bytes, *, filename: str, delimiter: str,
        plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the exact sealed TG16.2 family without causal interpretation."""
    expected = dict(plan)
    supplied_digest = expected.pop("plan_sha256", "")
    expected.pop("probe", None)
    expected.pop("claim_boundary", None)
    actual_digest = hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied_digest != actual_digest:
        raise InvalidParameterError(
            "plan_sha256", supplied_digest,
            "the digest of the complete frozen conditional-information plan")
    if expected.get("schema") != "spectral.conditional-information-plan.v1":
        raise InvalidParameterError(
            "schema", expected.get("schema"),
            "spectral.conditional-information-plan.v1")
    if _sha(payload) != expected["content_sha256"]:
        raise InvalidParameterError(
            "content_sha256", _sha(payload),
            "the exact file bytes the conditional-information plan sealed")
    declaration = SampleTableDeclaration(**expected["declaration"])
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="the conditional-information audit")
    columns, _header = _numeric_table(payload, delimiter, declaration)
    actual_target = next(name for name, role in declaration.roles.items() if role == "target")
    actual_nuisances = sorted(name for name, role in declaration.roles.items()
                              if role == "nuisance")
    actual_candidates = sorted(name for name, role in declaration.roles.items()
                               if role == "feature")
    if (actual_target != expected["target"] or actual_nuisances != [expected["nuisance"]]
            or actual_candidates != expected["candidates"]):
        raise InvalidParameterError(
            "conditional family",
            {"target": actual_target, "nuisance": actual_nuisances,
             "candidates": actual_candidates},
            "the exact target, nuisance and candidate family sealed by the plan")
    required = [actual_target, expected["nuisance"], *actual_candidates]
    missing = {name: int(np.sum(~np.isfinite(columns[name]))) for name in required}
    if any(missing.values()):
        raise InvalidParameterError(
            "missing analysis values", missing,
            "complete target, nuisance and feature columns for this recipe")
    support = conditional_support(
        {name: columns[name] for name in actual_candidates}, columns[actual_target],
        columns[expected["nuisance"]], bins=int(expected["bins"]))
    if support != expected["support_admission"] or not support["admitted"]:
        raise InvalidParameterError(
            "conditional overlap/effective support", support,
            "the exact admitted support assessment sealed by the plan")
    measured = audit_conditional_information(
        {name: columns[name] for name in actual_candidates}, columns[actual_target],
        columns[expected["nuisance"]], bins=int(expected["bins"]),
        permutations=int(expected["permutations"]), seed=int(expected["seed"]),
        alpha=float(expected["alpha"]), correction=str(expected["correction"]))
    return {
        "schema": "spectral.conditional-information-audit.v1",
        "plan_sha256": supplied_digest, "content_sha256": expected["content_sha256"],
        "target": actual_target, "declared_nuisance": expected["nuisance"],
        "sample_relationship": declaration.sample_relationship,
        "family": {"candidates": actual_candidates, "n_tests": expected["n_tests"],
                   "family_policy": expected["family_policy"],
                   "correction": expected["correction"], "alpha": expected["alpha"],
                   "permutations": expected["permutations"]},
        "method": {key: expected[key] for key in
                   ("quantity", "estimator", "bins", "null", "seed")},
        "support_admission": measured["support"],
        "conditional_associations": measured["candidates"],
        "outcome_vocabulary": measured["outcome_vocabulary"],
        "stored": False, "rung_moved": False,
        "claim_boundary": measured["claim_boundary"],
    }


def plan_stable_subspace_generation(
        payload: bytes, *, filename: str, delimiter: str,
        declaration: SampleTableDeclaration, dimensions: Sequence[int] = (1,),
        regularizations: Sequence[float] = (0.01, 0.1, 1.0), permutations: int = 4999,
        generate_fraction: float = 0.7, restarts: int = 6, iterations: int = 48,
        perturbations: int = 6, perturbation_scale: float = 0.10,
        stability_threshold: float = 0.10, nuisance_penalty: float = 1.0,
        variance_weight: float = 0.05, seed: int = 16301,
        alpha: float = 0.05) -> Dict[str, Any]:
    """Freeze the complete bounded family without opening a subspace result."""
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="stable-subspace generation")
    features = sorted(name for name, role in declaration.roles.items() if role == "feature")
    nuisances = sorted(name for name, role in declaration.roles.items() if role == "nuisance")
    if not 2 <= len(features) <= 6:
        raise InvalidParameterError("feature family", len(features),
                                    "two to six predeclared raw features")
    if len(nuisances) > 1:
        raise InvalidParameterError("nuisance family", nuisances,
                                    "zero or one researcher-declared nuisance")
    try:
        dims = tuple(sorted(int(value) for value in dimensions))
        regs = tuple(sorted(float(value) for value in regularizations))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParameterError("subspace family", [dimensions, regularizations],
                                    "finite numeric dimension and ridge sequences") from exc
    if any(isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value))
           or float(value) != int(value) for value in dimensions):
        raise InvalidParameterError("dimensions", list(dimensions),
                                    "integer dimensions only")
    if not dims or len(set(dims)) != len(dims) or min(dims) < 1 \
            or max(dims) >= len(features):
        raise InvalidParameterError("dimensions", list(dimensions),
                                    "distinct dimensions from 1 through feature_count - 1")
    if not regs or len(regs) > 4 or len(set(regs)) != len(regs) \
            or any(not math.isfinite(value) for value in regs) or min(regs) <= 0 \
            or max(regs) > 10:
        raise InvalidParameterError("regularizations", list(regularizations),
                                    "one to four distinct positive values no larger than 10")
    if not 0.5 <= float(generate_fraction) <= 0.8 or not 0 < float(alpha) < 1:
        raise InvalidParameterError("split/alpha", [generate_fraction, alpha],
                                    "generate_fraction in [0.5, 0.8] and alpha in (0, 1)")
    if not 2 <= int(restarts) <= 12 or not 16 <= int(iterations) <= 128 \
            or not 3 <= int(perturbations) <= 20:
        raise InvalidParameterError("optimizer/stability counts",
                                    [restarts, iterations, perturbations],
                                    "2-12 restarts, 16-128 iterations and 3-20 perturbations")
    if not 0 < float(perturbation_scale) <= 0.1 \
            or not 0 < float(stability_threshold) <= 0.5 \
            or not 0 <= float(nuisance_penalty) <= 10 \
            or not 0 <= float(variance_weight) <= 1:
        raise InvalidParameterError(
            "objective/stability values",
            [perturbation_scale, stability_threshold, nuisance_penalty, variance_weight],
            "perturbation scale (0,0.1], span threshold (0,0.5], nuisance penalty [0,10] "
            "and variance weight [0,1]")
    n_tests = len(dims) * len(regs)
    minimum_permutations = required_surrogates(
        n_tests, float(alpha), "benjamini_yekutieli")
    if not minimum_permutations <= int(permutations) <= 9999:
        raise InvalidParameterError(
            "permutations", permutations,
            "at least %d and at most 9999 for family size %d: the smallest attainable "
            "p-value must survive the frozen Benjamini-Yekutieli correction"
            % (minimum_permutations, n_tests))
    columns, _header = _numeric_table(payload, delimiter, declaration)
    required = features + nuisances + [
        next(name for name, role in declaration.roles.items() if role == "target")]
    missing = {name: int(np.sum(~np.isfinite(columns[name]))) for name in required}
    if any(missing.values()):
        raise InvalidParameterError("missing analysis values", missing,
                                    "complete target, feature and nuisance columns")
    n = probe["n_rows"]
    order = np.random.default_rng(int(seed)).permutation(n)
    stop = int(math.floor(n * float(generate_fraction)))
    generate, confirmation = order[:stop], order[stop:]
    if generate.size < 80 or confirmation.size < 40:
        raise InvalidParameterError("partition sizes", [generate.size, confirmation.size],
                                    "at least 80 generate and 40 reserved confirmation rows")
    generate_scales = {name: float(np.std(columns[name][generate], ddof=1))
                       for name in features}
    if any(not math.isfinite(value) or value <= 1e-12
           for value in generate_scales.values()):
        raise InvalidParameterError("generate feature variation", generate_scales,
                                    "every feature varying on the generate partition")
    target_scale = float(np.std(columns[required[-1]][generate], ddof=1))
    if not math.isfinite(target_scale) or target_scale <= 1e-12:
        raise InvalidParameterError("generate target variation", target_scale,
                                    "a varying target on the generate partition")
    if nuisances:
        nuisance_values = columns[nuisances[0]][generate]
        nuisance_scale = float(np.std(nuisance_values, ddof=1))
        region_sizes = np.bincount(np.digitize(
            nuisance_values, np.quantile(nuisance_values, (1 / 3, 2 / 3)), right=True),
                                   minlength=3)
        if nuisance_scale <= 1e-12 or int(region_sizes.min()) < 20:
            raise InvalidParameterError(
                "generate nuisance support",
                {"scale": nuisance_scale, "region_sizes": region_sizes.tolist()},
                "a varying nuisance with at least 20 generate rows in each frozen tertile")
    body = {
        "schema": "spectral.stable-subspace-generation-plan.v1",
        "content_sha256": probe["content_sha256"],
        "declaration": declaration.canonical(), "features": features,
        "target": required[-1], "nuisance": nuisances[0] if nuisances else None,
        "dimensions": list(dims), "regularizations": list(regs),
        "preprocessing": "generate-only z-score standardization",
        "objective": "regularized supervised covariance with nuisance-region penalty",
        "nuisance_stability_criterion": (
            "generate-tertile explained-fraction range <= 0.35 when nuisance is declared; "
            "otherwise no nuisance-region claim"),
        "optimizer": "seeded block power iteration", "restarts": int(restarts),
        "iterations": int(iterations), "perturbations": int(perturbations),
        "seed_derivation": (
            "fixed member, permutation, perturbation-stream and optimizer offsets"),
        "perturbation_scale": float(perturbation_scale),
        "projector_distance_threshold": float(stability_threshold),
        "nuisance_penalty": float(nuisance_penalty),
        "variance_weight": float(variance_weight), "permutations": int(permutations),
        "generate_fraction": float(generate_fraction), "seed": int(seed),
        "alpha": float(alpha), "correction": "benjamini_yekutieli",
        "family_members": ["dimension=%d;ridge=%g" % (dimension, ridge)
                           for dimension in dims for ridge in regs],
        "n_tests": n_tests,
        "partitions": {
            "generate_n": int(generate.size), "confirmation_n": int(confirmation.size),
            "generate_indices_sha256": _sha(generate.astype("<i8").tobytes()),
            "confirmation_indices_sha256": _sha(confirmation.astype("<i8").tobytes()),
            "confirmation_opened": False,
        },
    }
    body["plan_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**body, "probe": probe,
            "claim_boundary": (
                "This seals the complete generate-only linear search. It computes no subspace "
                "and does not open the reserved confirmation values.")}


def run_stable_subspace_generation(payload: bytes, *, filename: str, delimiter: str,
                                   plan: Mapping[str, Any]) -> Dict[str, Any]:
    expected = dict(plan)
    supplied_digest = expected.pop("plan_sha256", "")
    expected.pop("probe", None)
    expected.pop("claim_boundary", None)
    actual_digest = hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied_digest != actual_digest:
        raise InvalidParameterError("plan_sha256", supplied_digest,
                                    "the digest of the complete frozen subspace plan")
    if expected.get("schema") != "spectral.stable-subspace-generation-plan.v1":
        raise InvalidParameterError("plan schema", expected.get("schema"),
                                    "spectral.stable-subspace-generation-plan.v1")
    if _sha(payload) != expected["content_sha256"]:
        raise InvalidParameterError("content_sha256", _sha(payload),
                                    "the exact file bytes the plan sealed")
    declaration = SampleTableDeclaration(**expected["declaration"])
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="stable-subspace generation")
    actual_features = sorted(name for name, role in declaration.roles.items()
                             if role == "feature")
    actual_target = next(name for name, role in declaration.roles.items()
                         if role == "target")
    actual_nuisances = sorted(name for name, role in declaration.roles.items()
                              if role == "nuisance")
    if actual_features != expected["features"] or actual_target != expected["target"] \
            or (actual_nuisances[0] if actual_nuisances else None) != expected["nuisance"]:
        raise InvalidParameterError("analysis family", [actual_features, actual_target,
                                                         actual_nuisances],
                                    "the complete frozen feature/target/nuisance family")
    columns, _header = _numeric_table(payload, delimiter, declaration)
    n = probe["n_rows"]
    order = np.random.default_rng(int(expected["seed"])).permutation(n)
    stop = int(math.floor(n * float(expected["generate_fraction"])))
    generate, confirmation = order[:stop], order[stop:]
    partition = expected["partitions"]
    actual_partition = {
        "generate_n": int(generate.size), "confirmation_n": int(confirmation.size),
        "generate_indices_sha256": _sha(generate.astype("<i8").tobytes()),
        "confirmation_indices_sha256": _sha(confirmation.astype("<i8").tobytes()),
        "confirmation_opened": False,
    }
    if partition != actual_partition:
        raise InvalidParameterError("partitions", actual_partition,
                                    "the exact frozen generate/confirmation split")
    measured = generate_stable_subspaces(
        {name: columns[name][generate] for name in actual_features},
        columns[actual_target][generate],
        (columns[actual_nuisances[0]][generate] if actual_nuisances else None),
        dimensions=expected["dimensions"],
        regularizations=expected["regularizations"],
        permutations=int(expected["permutations"]), restarts=int(expected["restarts"]),
        iterations=int(expected["iterations"]), perturbations=int(expected["perturbations"]),
        perturbation_scale=float(expected["perturbation_scale"]),
        stability_threshold=float(expected["projector_distance_threshold"]),
        nuisance_penalty=float(expected["nuisance_penalty"]),
        variance_weight=float(expected["variance_weight"]), seed=int(expected["seed"]),
        alpha=float(expected["alpha"]), correction=expected["correction"])
    return {
        "schema": "spectral.stable-subspace-generation.v1",
        "plan_sha256": supplied_digest, "content_sha256": expected["content_sha256"],
        "target": actual_target, "declared_nuisance": expected["nuisance"],
        "sample_relationship": declaration.sample_relationship,
        "partitions": actual_partition,
        "preprocessing": measured["preprocessing"], "family": measured["family"],
        "search": measured["search"], "stability": measured["stability"],
        "subspaces": measured["subspaces"],
        "candidate_compact_stable_subspaces": measured["compact_candidates"],
        "outcome_vocabulary": measured["outcome_vocabulary"],
        "stored": False, "rung_moved": False,
        "claim_boundary": measured["claim_boundary"] +
            " The confirmation partition remains unopened in this generation operation.",
    }


def plan_representation_audit(payload: bytes, *, filename: str, delimiter: str,
                              declaration: SampleTableDeclaration,
                              representations: Sequence[str] = ("identity", "pca"),
                              pca_components: int = 3, bins: int = 4,
                              permutations: int = 4999, generate_fraction: float = 0.7,
                              seed: int = 1729, alpha: float = 0.05) -> Dict[str, Any]:
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    require_independent_samples(declaration, recipe="this first representation recipe")
    representations = tuple(dict.fromkeys(str(value) for value in representations))
    unknown = sorted(set(representations) - set(AUDIT_REPRESENTATIONS))
    if not representations or unknown:
        raise InvalidParameterError("representations", list(representations),
                                    "a non-empty subset of %s" % (AUDIT_REPRESENTATIONS,))
    features = sorted(name for name, role in declaration.roles.items() if role == "feature")
    if len(features) > 32:
        raise InvalidParameterError("feature family", len(features),
                                    "at most 32 predeclared features in this bounded recipe")
    if isinstance(pca_components, bool) or not 1 <= int(pca_components) <= len(features):
        raise InvalidParameterError("pca_components", pca_components,
                                    "an integer from 1 through the feature count")
    if not 2 <= int(bins) <= 8:
        raise InvalidParameterError("bins", bins, "an integer from 2 through 8")
    if not 99 <= int(permutations) <= 9999:
        raise InvalidParameterError("permutations", permutations,
                                    "99-9999 fixed target permutations")
    if not 0.5 <= float(generate_fraction) <= 0.8 or not 0 < float(alpha) < 1:
        raise InvalidParameterError("split/alpha", [generate_fraction, alpha],
                                    "generate_fraction in [0.5, 0.8] and alpha in (0, 1)")
    candidates = []
    if "identity" in representations:
        candidates.extend("identity:%s" % name for name in features)
    if "pca" in representations:
        candidates.extend("pca:component_%d" % (index + 1)
                          for index in range(int(pca_components)))
    body = {"schema": "spectral.representation-audit-plan.v1",
            "content_sha256": probe["content_sha256"], "declaration": declaration.canonical(),
            "representations": list(representations), "pca_components": int(pca_components),
            "bins": int(bins), "permutations": int(permutations),
            "generate_fraction": float(generate_fraction), "seed": int(seed),
            "alpha": float(alpha), "correction": "benjamini_yekutieli",
            "candidates": candidates, "n_tests_per_partition": len(candidates)}
    by_penalty = sum(1.0 / value for value in range(1, len(candidates) + 1))
    minimum_permutations = math.ceil(len(candidates) * by_penalty / float(alpha)) - 1
    if int(permutations) < minimum_permutations:
        raise InvalidParameterError(
            "permutations", permutations,
            "at least %d for family size %d: the smallest attainable p-value must be able "
            "to survive the declared Benjamini-Yekutieli correction at alpha=%g"
            % (minimum_permutations, len(candidates), alpha))
    body["plan_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**body, "probe": probe,
            "admissibility": {"structure": True, "representation_audit": True,
                              "temporal_dependencies":
                                  declaration.sample_relationship == "ordered"},
            "claim_boundary": (
                "This freezes the complete candidate family and split before enumeration. "
                "It computes no association and opens no held-out values.")}


def _permutation_p(x: np.ndarray, target: np.ndarray, *, bins: int,
                   permutations: int, seed: int) -> Tuple[float, float]:
    observed = mutual_information(x, target, bins=bins)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        null = mutual_information(x, rng.permutation(target), bins=bins)
        exceed += int(null >= observed)
    return float(observed), float((exceed + 1) / (permutations + 1))


def run_representation_audit(payload: bytes, *, filename: str, delimiter: str,
                             plan: Mapping[str, Any]) -> Dict[str, Any]:
    expected = dict(plan)
    supplied_digest = expected.pop("plan_sha256", "")
    # API render-only keys are not part of the sealed body.
    expected.pop("probe", None)
    expected.pop("admissibility", None)
    expected.pop("claim_boundary", None)
    actual_digest = hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied_digest != actual_digest:
        raise InvalidParameterError("plan_sha256", supplied_digest,
                                    "the digest of the complete frozen plan")
    if _sha(payload) != expected["content_sha256"]:
        raise InvalidParameterError("content_sha256", _sha(payload),
                                    "the exact file bytes the plan sealed")
    declaration = SampleTableDeclaration(**expected["declaration"])
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
    columns, _header = _numeric_table(payload, delimiter, declaration)
    target_name = next(name for name, role in declaration.roles.items() if role == "target")
    feature_names = sorted(name for name, role in declaration.roles.items() if role == "feature")
    required = [target_name] + feature_names + [name for name, role in declaration.roles.items()
                                                if role == "nuisance"]
    missing = {name: int(np.sum(~np.isfinite(columns[name]))) for name in required}
    if any(missing.values()):
        raise InvalidParameterError("missing analysis values", missing,
                                    "complete target, feature and nuisance columns for this first recipe")
    n = columns[target_name].size
    if n < 40:
        raise InvalidParameterError("sample count", n,
                                    "at least 40 complete samples for a generate/confirm split")
    rng = np.random.default_rng(int(expected["seed"]))
    order = rng.permutation(n)
    stop = int(math.floor(n * float(expected["generate_fraction"])))
    generate, confirm = order[:stop], order[stop:]
    if min(generate.size, confirm.size) < 16:
        raise InvalidParameterError("partition sizes", [generate.size, confirm.size],
                                    "at least 16 samples in both generate and confirm")
    features = np.column_stack([columns[name] for name in feature_names])
    candidate_values: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if "identity" in expected["representations"]:
        for index, name in enumerate(feature_names):
            candidate_values["identity:%s" % name] = (features[generate, index],
                                                       features[confirm, index])
    singular_values = []
    if "pca" in expected["representations"]:
        mean = features[generate].mean(axis=0)
        scale = features[generate].std(axis=0, ddof=1)
        if np.any(scale <= 0):
            constants = [feature_names[index] for index in np.flatnonzero(scale <= 0)]
            raise InvalidParameterError("PCA features", constants,
                                        "non-constant features in the generate partition")
        z_generate = (features[generate] - mean) / scale
        z_confirm = (features[confirm] - mean) / scale
        _u, singular, vt = np.linalg.svd(z_generate, full_matrices=False)
        # Canonical signs make identical bytes reproduce identical component coordinates.
        for row in vt:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1
        singular_values = [float(value) for value in singular]
        for index in range(int(expected["pca_components"])):
            candidate_values["pca:component_%d" % (index + 1)] = (
                z_generate @ vt[index], z_confirm @ vt[index])
    if list(candidate_values) != list(expected["candidates"]):
        raise InvalidParameterError("candidate family", list(candidate_values),
                                    "the exact ordered family sealed by the plan")
    target = columns[target_name]
    rows = []
    nuisance_names = sorted(name for name, role in declaration.roles.items()
                            if role == "nuisance")
    for index, (label, (x_generate, x_confirm)) in enumerate(candidate_values.items()):
        g_mi, g_p = _permutation_p(
            x_generate, target[generate], bins=int(expected["bins"]),
            permutations=int(expected["permutations"]),
            seed=int(expected["seed"]) + 1009 * (index + 1))
        c_mi, c_p = _permutation_p(
            x_confirm, target[confirm], bins=int(expected["bins"]),
            permutations=int(expected["permutations"]),
            seed=int(expected["seed"]) + 1000003 + 1009 * (index + 1))
        stability = []
        for nuisance_name in nuisance_names:
            nuisance = columns[nuisance_name][confirm]
            cut = float(np.median(columns[nuisance_name][generate]))
            strata = [("at_or_below_generate_median", nuisance <= cut),
                      ("above_generate_median", nuisance > cut)]
            measured = []
            for stratum, mask in strata:
                measured.append({"stratum": stratum, "n": int(mask.sum()),
                                 "mi_nats": (mutual_information(
                                     x_confirm[mask], target[confirm][mask], bins=int(expected["bins"]))
                                     if int(mask.sum()) >= 8 else None)})
            finite_mi = [value["mi_nats"] for value in measured
                         if value["mi_nats"] is not None and math.isfinite(value["mi_nats"])]
            stability.append({"nuisance": nuisance_name,
                              "split_value_from_generate": cut, "confirm_strata": measured,
                              "mi_range_nats": (max(finite_mi) - min(finite_mi)
                                                if len(finite_mi) == 2 else None),
                              "interpretation": (
                                  "descriptive stratified robustness only; not conditional "
                                  "mutual information or proof that nuisance was removed")})
        rows.append({"candidate": label, "generate_mi_nats": g_mi,
                     "generate_p_value": g_p, "confirm_mi_nats": c_mi,
                     "confirm_p_value": c_p, "nuisance_stability": stability})
    labels = [row["candidate"] for row in rows]
    generated = adjust([row["generate_p_value"] for row in rows],
                       method=expected["correction"], alpha=float(expected["alpha"]),
                       n_tests=int(expected["n_tests_per_partition"]), labels=labels)
    confirmed = adjust([row["confirm_p_value"] for row in rows],
                       method=expected["correction"], alpha=float(expected["alpha"]),
                       n_tests=int(expected["n_tests_per_partition"]), labels=labels)
    for index, row in enumerate(rows):
        row["generate_q_value"] = generated["adjusted"][index]
        row["confirm_q_value"] = confirmed["adjusted"][index]
        row["survives_both"] = bool(generated["rejected"][index]
                                     and confirmed["rejected"][index])
    covariance = np.atleast_2d(np.corrcoef(features, rowvar=False))
    eigenvalues = np.linalg.eigvalsh(covariance)
    effective_dimension = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))
    return {"schema": "spectral.representation-audit.v1",
            "plan_sha256": supplied_digest, "content_sha256": expected["content_sha256"],
            "target": target_name, "sample_relationship": declaration.sample_relationship,
            "declared_nuisances": nuisance_names,
            "partitions": {"generate_n": int(generate.size), "confirm_n": int(confirm.size),
                           "split_seed": int(expected["seed"]),
                           "held_out_opened_once": True},
            "family": {"candidates": labels, "n_tests_per_partition": len(labels),
                       "correction": expected["correction"],
                       "permutations": int(expected["permutations"]),
                       "minimum_p_value": 1 / (int(expected["permutations"]) + 1)},
            "structure": {"feature_count": len(feature_names),
                          "effective_dimension": effective_dimension,
                          "absolute_correlation": np.abs(covariance).tolist(),
                          "pca_singular_values_generate": singular_values},
            "candidates": sorted(rows, key=lambda value: (
                not value["survives_both"], value["confirm_q_value"],
                -value["confirm_mi_nats"], value["candidate"])),
            "stored": False, "rung_moved": False,
            "claim_boundary": (
                "Candidates are associations surviving one fixed generate/confirm programme. "
                "They are not certified invariants, causal features or a recommendation to use; "
                "task-specific external validation remains required.")}


__all__ = ["AUDIT_REPRESENTATIONS", "MAX_UPLOAD_BYTES", "SAMPLE_RELATIONSHIPS",
           "SAMPLE_ROLES", "SampleTableDeclaration", "plan_conditional_information_audit",
           "plan_redundancy_structure_audit", "plan_representation_audit",
           "plan_stable_subspace_generation",
           "probe_delimited", "require_independent_samples",
           "run_conditional_information_audit", "run_redundancy_structure_audit",
           "run_representation_audit", "run_stable_subspace_generation",
           "sample_table_capability_profile"]
