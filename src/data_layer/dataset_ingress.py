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
from src.core.errors import InvalidParameterError
from src.statistics.multiple_comparisons import adjust

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


def plan_representation_audit(payload: bytes, *, filename: str, delimiter: str,
                              declaration: SampleTableDeclaration,
                              representations: Sequence[str] = ("identity", "pca"),
                              pca_components: int = 3, bins: int = 4,
                              permutations: int = 4999, generate_fraction: float = 0.7,
                              seed: int = 1729, alpha: float = 0.05) -> Dict[str, Any]:
    probe = probe_delimited(payload, filename=filename, delimiter=delimiter)
    declaration.validate(probe)
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
           "SAMPLE_ROLES", "SampleTableDeclaration", "plan_representation_audit",
           "probe_delimited", "run_representation_audit"]
