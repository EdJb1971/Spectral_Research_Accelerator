"""Run the T4E.24 false-absence measurement and write its receipt.

Run from the repository root:

    python -m tools.measure_false_absence --output measurements/t4e24_false_absence.json

The adopted declaration is read and hashed before anything is measured, and the hash is written
into the receipt, so a receipt can be checked against the declaration it claims to answer. No
download occurs and no frame of the 2022-2023 forecast-test period is opened: the record is read
from `data/cds_downloads/t4e18_vorticity`, which stops at 2021-12-31 by acquisition.

T4E.20 and T4E.21 produced their receipts from scripts that were never committed. That is a
reproducibility gap this slice does not repeat -- the code that produced the numbers is here.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr

from src.benchmarks.false_absence import (
    DECLARED_PAIRING_RADIUS_CELLS,
    DECLARED_SCENES_PER_CONFIGURATION,
    admission_rates,
    background_scale,
    dispersion,
    draw_configuration,
    pair_plantings,
    plant,
    presence_distribution,
)
from src.benchmarks.synthetic_backgrounds import feature_density_gate
from src.core.domain import AxisSpec
from src.core.extraction import ExtractionField, calibrate, extract
from src.statistics import surrogates as surrogate_module

DECLARATION = "data/identity_calibration/t4e24-false-absence-declaration.json"
ADOPTION = "data/identity_calibration/t4e24-false-absence-adoption.json"
RECORD = "data/cds_downloads/t4e18_vorticity"

#: Chosen at implementation time and recorded here rather than left implicit. It was fixed
#: before any scene was built and no result was looked at before it was set; had a different
#: seed been tried and this one kept, that would be a fitted choice and this comment would be
#: a lie. It is not drawn from any reserved block: 720-735 and 880-895 are untouched by this
#: slice, which draws its own scenes and needs none of them.
ROOT_SEED = 20260911

#: The declaration's setting: 60 configurations, each planted into S independent backgrounds.
CONFIGURATIONS = 60

#: Index space. The whole measurement is in cells -- the pairing radius, the separation and the
#: offsets -- so no coordinate vector is supplied and `spacing` is 1.0 by construction.
AXES = (AxisSpec(name="row", role="space", units=None, periodic=False),
        AxisSpec(name="col", role="space", units=None, periodic=False))


def digest_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision() -> Dict[str, object]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:                                        # pragma: no cover - git absent
        return {"revision": None, "code_dirty": None}
    return {"revision": head, "code_dirty": dirty}


def record_frames() -> List[Tuple[str, int]]:
    """Every (shard, time index) in the acquired record, in a deterministic order."""
    shards = sorted(glob.glob(os.path.join(RECORD, "*.nc")))
    if not shards:
        raise SystemExit("no record shards under %s; this slice reads the record and "
                         "downloads nothing" % RECORD)
    frames: List[Tuple[str, int]] = []
    for shard in shards:
        with xr.open_dataset(shard) as ds:
            frames.extend((shard, i) for i in range(int(ds.sizes["valid_time"])))
    return frames


def read_frame(shard: str, index: int) -> np.ndarray:
    """One frame, negated -- the sign convention T4E.18 declared before the record existed.

    In the southern hemisphere cyclonic rotation is *negative* relative vorticity, so a
    maximum-finder on the raw field would locate anticyclones. The negation is applied here for
    consistency with the record's declared convention even though it is arithmetically inert
    once the frame is phase-randomised: `|FFT|` is unchanged by a sign flip. Stating that it is
    inert is better than implying it did work it did not do.
    """
    with xr.open_dataset(shard) as ds:
        values = ds["vo"].isel(valid_time=int(index), pressure_level=0).values
    return -np.asarray(values, dtype=np.float64)


def field_for(values: np.ndarray, time: float) -> ExtractionField:
    return ExtractionField(
        values=values, axes=AXES, domain="atmosphere", dataset="t4e24_synthetic_scene",
        variable="negated_relative_vorticity", units="s**-1", time=float(time),
        representation="raw")


def extracted_positions(result) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(f.location.coords["row"]), float(f.location.coords["col"]))
                 for f in result.features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--configurations", type=int, default=CONFIGURATIONS)
    parser.add_argument("--scenes", type=int, default=DECLARED_SCENES_PER_CONFIGURATION)
    parser.add_argument("--surrogates", type=int, default=999)
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    parser.add_argument("--calibrate-on", choices=("background", "scene"), default="scene")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit("%s exists; a receipt is written once and never overwritten" % output)

    started = time.time()
    before = revision()
    declaration_hash = digest_file(DECLARATION)
    adoption_hash = digest_file(ADOPTION)

    frames = record_frames()
    n_scenes_total = int(args.configurations) * int(args.scenes)
    if n_scenes_total > len(frames):
        raise SystemExit("the record holds %d frames and %d distinct backgrounds were asked "
                         "for" % (len(frames), n_scenes_total))

    rng = np.random.default_rng(int(args.seed))
    # Distinct source frames, drawn without replacement, so no two backgrounds in the run share
    # a spectrum and "independent" means independent in the source as well as in the phases.
    chosen = rng.choice(len(frames), size=n_scenes_total, replace=False)

    per_scene_counts: List[int] = []
    rows: List[Dict[str, object]] = []
    presence_counts: List[int] = []
    trials = 0
    absences = 0
    spurious_total = 0
    scene_cursor = 0

    for c in range(int(args.configurations)):
        shard, index = frames[int(chosen[scene_cursor])]
        first = read_frame(shard, index)
        features = draw_configuration(rng, first.shape)
        seen = [0] * len(features)
        per_feature_offsets: List[List[float]] = [[] for _ in features]

        for s in range(int(args.scenes)):
            shard, index = frames[int(chosen[scene_cursor])]
            scene_cursor += 1
            source = read_frame(shard, index)
            seed = int(rng.integers(0, 2 ** 31 - 1))
            background = np.asarray(surrogate_module.generate(
                source, method="phase_randomise", n=1, seed=seed)["members"][0],
                dtype=np.float64)
            width = background_scale(background)
            values = plant(background, features, scale=width)

            # WHERE THE CUT COMES FROM, and why it is the harsher of the two options.
            #
            # The declaration does not fix this, so it is recorded as a choice rather than
            # slipped in. Calibrating on the BACKGROUND gives a cleaner null -- the plantings
            # do not enter the surrogates that set their own threshold -- and it was measured:
            # recovery 0.940 against 0.774, but 63 extracted features matching no planting
            # across 48 scenes, against 0. A cut that admits more than one noise peak per
            # scene is not controlling the family-wise error its own receipt claims.
            #
            # Calibrating on the SCENE is what the pipeline does on the real record, and it
            # reproduces T4E.21 -- the slice whose 32 per cent loss is the reason this one
            # exists -- at gate median 4 and near-zero spurious features. The criterion is
            # handed features from that pipeline, not from an idealised one, so this is the
            # faithful choice and it is also the less flattering one. Both were measured
            # before either was chosen, and both are reported.
            cut_from = background if args.calibrate_on == "background" else values
            calibration = calibrate(cut_from, n_surrogates=int(args.surrogates), seed=seed)
            result = extract(field_for(values, time=float(s)), calibration=calibration)
            positions = extracted_positions(result)
            per_scene_counts.append(len(positions))

            paired, offsets = pair_plantings(
                features, positions, radius=DECLARED_PAIRING_RADIUS_CELLS)
            spurious_total += len(positions) - sum(1 for p in paired if p is not None)
            for i, (p, offset) in enumerate(zip(paired, offsets)):
                trials += 1
                if p is None:
                    absences += 1
                else:
                    seen[i] += 1
                    per_feature_offsets[i].append(float(offset))

        for i, feature in enumerate(features):
            presence_counts.append(seen[i])
            found = per_feature_offsets[i]
            rows.append({
                "configuration": c,
                "feature": i,
                "seen_in": seen[i],
                "of": int(args.scenes),
                "sigma": feature.sigma,
                "ratio": feature.ratio,
                "stretch": feature.stretch,
                "nearest_neighbour_cells": min(
                    [math.hypot(feature.row - g.row, feature.col - g.col)
                     for j, g in enumerate(features) if j != i] or [float("nan")]),
                "median_offset_cells": (float(np.median(found)) if found else None),
            })

    gate = feature_density_gate(per_scene_counts)
    marginal = absences / trials if trials else float("nan")

    receipt: Dict[str, object] = {
        "measurement": "t4e24_false_absence",
        "declared_in": DECLARATION,
        "declaration_sha256": declaration_hash,
        "adopted_in": ADOPTION,
        "adoption_sha256": adoption_hash,
        "measured_on": time.strftime("%Y-%m-%d"),
        "root_seed": int(args.seed),
        "configurations": int(args.configurations),
        "scenes_per_configuration": int(args.scenes),
        "surrogates_per_calibration": int(args.surrogates),
        "calibrated_on": args.calibrate_on,
        "gate": gate,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    receipt.update({"source_" + k: v for k, v in before.items()})

    if not gate["passed"]:
        # Acceptance condition 1. The background is not tuned until it passes.
        receipt["VERDICT"] = "EVIDENCE_NOT_REPRESENTATIVE"
        receipt["what_this_licenses"] = (
            "Nothing. The synthetic evidence did not carry features at the density the record "
            "carries, so no false absence rate measured on it describes the extractor's "
            "behaviour on the record. The declaration required the slice to stop here rather "
            "than adjust the background, and it stopped.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"gate": gate, "VERDICT": receipt["VERDICT"]}, indent=2))
        return 1

    distribution = presence_distribution(presence_counts, int(args.scenes))
    admissions = admission_rates(presence_counts, int(args.scenes))
    spread = dispersion(presence_counts, int(args.scenes), seed=int(args.seed))

    recovered_rows = [r for r in rows if r["median_offset_cells"] is not None]

    def by(key: str, edges: Sequence[Tuple[float, float]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for low, high in edges:
            band = [r for r in rows if low <= float(r[key]) < high]
            if not band:
                continue
            out.append({"from": low, "to": high, "features": len(band),
                        "mean_seen_in": float(np.mean([r["seen_in"] for r in band])),
                        "recovery": float(np.mean([r["seen_in"] for r in band]))
                        / float(args.scenes)})
        return out

    receipt.update({
        "VERDICT": "MEASURED",
        "trials": trials,
        "features": len(rows),
        "marginal_false_absence_rate": marginal,
        "extracted_matching_no_planting": spurious_total,
        "presence_count_distribution": distribution,
        "admission_rates": admissions,
        "dispersion": spread,
        "what_predicts_an_absence": {
            "by_scale_cells": by("sigma", [(0, 2.5), (2.5, 4), (4, 6), (6, 12)]),
            "by_amplitude_ratio": by("ratio", [(8, 14), (14, 20), (20, 26), (26, 32)]),
            "by_nearest_neighbour_cells": by(
                "nearest_neighbour_cells", [(8, 15), (15, 25), (25, 40), (40, 1e9)]),
            "adjudicates": "nothing; this is a description of the mechanism",
        },
        "median_offset_cells_over_recovered": (
            float(np.median([r["median_offset_cells"] for r in recovered_rows]))
            if recovered_rows else None),
        "rows": rows,
        "the_two_claims_are_separate": {
            "the_weak_half_mechanism": (
                "Whether absences concentrate on particular features. The design holds "
                "neighbours, scales and amplitudes fixed across the S scenes, which is what "
                "would cause concentration, so a concentrated result is close to built in. "
                "Declared before the measurement, not conceded after it."),
            "the_strong_half_consequence": (
                "The admission rate at each a, as a count over the evidence the criterion "
                "would actually be handed. This is what the slice may be cited for."),
        },
        "claim_boundary": (
            "This measures a property of the extractor on synthetic evidence with known "
            "truth. It settles nothing about the atmosphere, nothing about any real cyclone, "
            "and nothing about whether any identity criterion is correct. It bears on the "
            "partial-recurrence criterion only by bounding what fraction of genuinely "
            "recurrent features can reach it at all. The number belongs to 850 hPa relative "
            "vorticity over this crop under this planting and is not quoted for any other "
            "domain."),
        "what_this_does_not_do": [
            "It chooses no tolerance and proposes no proportion of S (R20).",
            "It modifies nothing in the extractor and proposes no replacement.",
            "It re-measures no identity criterion and does not supersede T4E.13.",
            "No reserved seed block (720-735, 880-895) and no frame of the 2022-2023 "
            "forecast-test period was read.",
        ],
    })
    receipt.update({"final_" + k: v for k, v in revision().items()})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    summary = {k: receipt[k] for k in (
        "gate", "trials", "features", "marginal_false_absence_rate",
        "presence_count_distribution", "admission_rates", "dispersion",
        "extracted_matching_no_planting", "elapsed_seconds")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
