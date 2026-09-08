"""Run the local-only T4E.8 spatial identity design and write its exploratory receipt.

Run from the repository root: python -m tools.audit_spatial_identity --output PATH
The design is read and hashed before source values. No downloads or test-period reads occur.
Peak coefficient storage is O(window_frames * bands * grid_cells); decomposition is chunked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import xarray as xr

from src.analysis_engine.climatology import StreamingHarmonicClimatology
from src.analysis_engine.spectral_clustering import AttributeWeights, SignatureMetric, SignaturePoint
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_identity_audit import labelled_errors, recall_radius, radius_feasibility
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_tracking import track_spectral_features
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import CoefficientField, decompose_sequence


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _window(ds, model, design, start, stop):
    raw = np.asarray(ds.t.isel(time=slice(start, stop), level=0).values, dtype=np.float64)
    timestamps = ds.time.values[start:stop].astype("datetime64[s]").astype(np.float64)
    source_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    reference = model["reference"]
    grid = GridSpec.latlon(raw.shape[1:], lat0=float(ds.latitude.values[0]),
                          lon0=float(ds.longitude.values[0]),
                          dlat=float(np.diff(ds.latitude.values)[0]),
                          dlon=float(np.diff(ds.longitude.values)[0]))

    def read_frame(index):
        return PhysicalField(torch.from_numpy(raw[index - start]), grid=grid,
                             coords={k: torch.tensor(v, dtype=torch.float64)
                                     for k, v in reference["coords"].items()},
                             metadata={"variable": reference["variable"], "level": reference["level"]},
                             units=reference["units"])

    fit = StreamingHarmonicClimatology(read_frame, model["design_normalised"], model["solution"],
                                      reference, model["provenance"])
    chunks, first = [], None
    chunk_size = design["execution"]["decomposition_chunk_frames"]
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        seq = FieldSequence([fit.anomaly(i) for i in range(chunk_start, chunk_stop)],
                            timestamps[chunk_start - start:chunk_stop - start])
        part = decompose_sequence(seq, design["transform"]["family"],
                                  {"levels": design["transform"]["levels"],
                                   "wavelet": design["transform"]["wavelet"]}).select(
                                       scales=design["transform"]["levels_read"])
        chunks.append(part.data.to(torch.float32))
        if first is None:
            first = part
    field = CoefficientField(torch.cat(chunks), wavelet_family=first.wavelet_family,
                             scales=first.scales, orientations=first.orientations,
                             times=timestamps, grid=first.grid, source_variable=first.source_variable,
                             orientation_convention=first.orientation_convention,
                             resampled_to_parent=first.resampled_to_parent,
                             native_shapes=first.native_shapes, config=first.config,
                             level=first.level, level_axis=first.level_axis, metadata=first.metadata)
    del chunks, first, part
    tracked = track_spectral_features(field, domain="atmosphere", dataset=design["record_path"],
                                      variable="t", time_units="s",
                                      threshold_sigma=design["detection"]["threshold_sigma"])
    del field, raw
    if tracked is None:
        raise ValueError("NO_FEATURES: this design produced no trackable observations")
    configurations = extract_constellations(
        tracked, cardinalities=tuple(design["cardinalities"]),
        max_nodes_per_frame=design["detection"]["max_nodes_per_frame"])
    return configurations, source_hash, float(timestamps[1] - timestamps[0])


def _labels(configurations, cadence, count, seed):
    keys = [tuple(sorted(c.track_ids)) for c in configurations]
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    same = []
    masks = {name: [] for name in ("stationary", "legacy_clean", "stationary_band_changed")}
    changed, contributing = 0, set()
    for key, observations in groups.items():
        for left, right in itertools.combinations(observations, 2):
            a, b = configurations[left], configurations[right]
            aa, bb = ({n.track_id: n for n in c.nodes} for c in (a, b))
            band_changed = any(aa[t].band != bb[t].band for t in key)
            displacement = max(math.sqrt(sum((aa[t].coords[axis] - bb[t].coords[axis]) ** 2
                                            for axis in aa[t].coords)) for t in key)
            stationary = b.time - a.time == cadence and displacement <= 1.0
            same.append((left, right))
            masks["stationary"].append(stationary)
            masks["legacy_clean"].append(stationary and not band_changed)
            masks["stationary_band_changed"].append(stationary and band_changed)
            changed += int(band_changed)
            contributing.add(key)
    if len(groups) < 2:
        raise ValueError("NO_UNRELATED_KEYS: proxy negatives cannot be sampled")
    rng = np.random.default_rng(seed)
    unrelated = []
    while len(unrelated) < count:
        left, right = map(int, rng.integers(0, len(keys), size=2))
        if keys[left] != keys[right]:
            unrelated.append((left, right))
    metadata = {
        "signatures": len(keys), "distinct_track_keys": len(groups), "repeat_pairs": len(same),
        "contributing_repeat_keys": len(contributing), "band_changed_pairs": changed,
        "band_changed_fraction": changed / len(same) if same else None,
        "band_change_denominator": "all repeated-key observation pairs; not only adjacent frames",
        "unrelated_unique_unordered_pairs": len({tuple(sorted(pair)) for pair in unrelated}),
        "strata": {name: {"pairs": sum(mask),
                           "contributing_keys": len({keys[same[i][0]] for i, yes in enumerate(mask) if yes}),
                           "upper_decile_observations": math.ceil(sum(mask) * 0.1)}
                   for name, mask in masks.items()},
    }
    return same, unrelated, masks, metadata


def run(design_path, output):
    started = time.perf_counter()
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    design_hash = digest_file(design_path)
    source_files = ["tools/audit_spatial_identity.py"] + sorted(
        p.as_posix() for p in Path("src").rglob("*.py") if "tests" not in p.parts)
    source_hashes = {p: digest_file(p) for p in source_files}
    print("Design sha256: " + design_hash, flush=True)
    for field in ("record_path", "climatology_path", "source_declaration"):
        if not Path(design[field]).exists():
            raise ValueError("LOCAL_INPUT_MISSING: " + design[field])
    if any(w["start"] < 0 or w["stop"] > 5844 or w["stop"] <= w["start"]
           for w in design["windows"]):
        raise ValueError("WINDOW_OUTSIDE_TRAINING: refuses to open the held-out forecast period")
    torch.set_num_threads(design["execution"]["torch_threads"])
    model = torch.load(design["climatology_path"], map_location="cpu", weights_only=False)
    ds = xr.open_zarr(design["record_path"], consolidated=True)
    grid_body = {name: ds[name].values.tolist() for name in ("latitude", "longitude", "level")}
    grid_body["record"] = design["record_path"]
    scope = hashlib.sha256(json.dumps(grid_body, sort_keys=True).encode()).hexdigest()
    metric = SignatureMetric(AttributeWeights(**design["weights"]))
    results, frozen_radius = [], None
    try:
        for number, window in enumerate(design["windows"]):
            tick = time.perf_counter()
            start, stop = window["start"], window["stop"]
            print("Window %d:%d: extracting" % (start, stop), flush=True)
            configurations, source_hash, cadence = _window(ds, model, design, start, stop)
            same, unrelated, masks, census = _labels(configurations, cadence,
                                                     design["unrelated_pairs"],
                                                     design["sampling_seed"] + number)
            print("Window %d:%d: %d configurations, %d repeat pairs" %
                  (start, stop, len(configurations), len(same)), flush=True)
            reports = {}
            for mode in ("scale_specific", design["signature_mode"]):
                signed = sign_constellations(configurations, mode=mode, comparison_scope=scope)
                if len(signed) != len(configurations):
                    raise ValueError("SIGNING_REFUSED: cannot compare a silently changed label population: "
                                     + json.dumps(signed.describe()))
                points = [SignaturePoint.from_signature(s) for s in signed]
                distances = np.array([metric.distance(points[a], points[b]) for a, b in same])
                negatives = [metric.distance(points[a], points[b]) for a, b in unrelated]
                if number == 0 and mode == design["signature_mode"]:
                    frozen_radius = recall_radius(distances[np.asarray(masks["stationary"], dtype=bool)], .9)
                diagnostic_radius = recall_radius(
                    distances[np.asarray(masks["stationary"], dtype=bool)], .9)
                reports[mode] = {
                    "diagnostic_stationary_r90_not_an_approved_radius": diagnostic_radius,
                    "all_pairs": labelled_errors(distances, negatives,
                                                  frozen_radius if mode == design["signature_mode"] else diagnostic_radius),
                    "strata": {name: labelled_errors(distances[np.asarray(mask, dtype=bool)], negatives,
                                                       frozen_radius if mode == design["signature_mode"] else diagnostic_radius)
                               for name, mask in masks.items()},
                }
                if design.get("report_empirical_radius_feasibility", False):
                    reports[mode]["empirical_radius_feasibility"] = radius_feasibility(
                        distances, negatives, max_split=design["acceptance_max_false_split"],
                        max_admission=design["acceptance_max_false_admission"])
                del signed, points
            result = {**window, "source_values_float64_sha256": source_hash,
                      "census": census, "modes": reports, "elapsed_seconds": time.perf_counter() - tick}
            results.append(result)
            print(json.dumps({"window": [start, stop], "candidate": reports[design["signature_mode"]],
                              "census": census}, sort_keys=True), flush=True)
            del configurations
            gc.collect()
    finally:
        ds.close()
    evaluations = [r["modes"][design["signature_mode"]]["all_pairs"] for r in results[1:]]
    passed = bool(evaluations) and all(
        e["false_split_rate"] is not None and e["false_admission_rate"] is not None
        and e["false_split_rate"] <= design["acceptance_max_false_split"]
        and e["false_admission_rate"] <= design["acceptance_max_false_admission"] for e in evaluations)
    if source_hashes != {p: digest_file(p) for p in source_files} or digest_file(design_path) != design_hash:
        raise ValueError("MEASUREMENT_SOURCE_CHANGED: refusing to publish a mixed-source receipt")
    receipt = {
        "schema": "spectral-spatial-identity-audit/v1", "design_sha256": design_hash,
        "design": design, "comparison_scope": scope, "metric": metric.describe(),
        "metric_digest": metric.digest, "source_hashes": source_hashes,
        "climatology_file_sha256": digest_file(design["climatology_path"]),
        "climatology_provenance": model["provenance"],
        "source_declaration_sha256": digest_file(design["source_declaration"]),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "frozen_radius": frozen_radius, "windows": results,
        "status": "DIAGNOSTIC_CRITERIA_MET" if passed else "DISCRIMINATION_CRITERIA_NOT_MET",
        "approved_mining_radius": None,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": design["claim_boundary"],
    }
    Path(output).write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("Receipt: %s; %s; %.2f s" % (output, receipt["status"], receipt["elapsed_seconds"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="data/identity_calibration/t4e8-spatial-design.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output exists; use a new path to preserve the previous measurement")
    run(args.design, args.output)
