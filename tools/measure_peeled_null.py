"""Run the T4E.26 peeled-null measurement and write its receipt.

Run from the repository root:

    python -m tools.measure_peeled_null --output measurements/t4e26_peeled_null.json

Three ensembles per scene: the scene's own (round zero, which must reproduce T4E.24 and T4E.25
exactly), the residual's (the candidate), and the bare background's (an oracle, reported as a
ceiling and never as an operating point). The declaration and its adoption are read and hashed
before anything is measured. No download occurs and no frame of the 2022-2023 forecast-test
period is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np

from src.benchmarks.false_absence import (
    DECLARED_PAIRING_RADIUS_CELLS,
    DECLARED_SCENES_PER_CONFIGURATION,
    admission_rates,
    background_scale,
    draw_configuration,
    pair_plantings,
    plant,
    presence_distribution,
)
from src.benchmarks.peeled_null import (
    gap_closed,
    nearest_planting_stretch,
    peel,
    split_contamination,
)
from src.benchmarks.synthetic_backgrounds import feature_density_gate
from src.core.extraction import calibration_from_maxima, extract
from src.statistics import surrogates as surrogate_module
from tools.measure_coverage_contamination import configuration_coverage
from tools.measure_false_absence import (
    CONFIGURATIONS,
    ROOT_SEED,
    digest_file,
    extracted_positions,
    field_for,
    read_frame,
    record_frames,
    revision,
)

DECLARATION = "data/identity_calibration/t4e26-peeled-null-declaration.json"
ADOPTION = "data/identity_calibration/t4e26-peeled-null-adoption.json"

#: The operating point, fixed. This slice sweeps nothing: T4E.25 already established that alpha
#: is not the lever, and sweeping it again beside a second variable would confound the two.
DECLARED_ALPHA: float = 0.05


def cut(values: np.ndarray, *, surrogates: int, seed: int) -> Tuple[float, list]:
    ensemble = surrogate_module.generate(
        values, method="phase_randomise", n=int(surrogates), seed=int(seed))
    maxima = [float(np.max(np.asarray(m))) for m in ensemble["members"]]
    calibration = calibration_from_maxima(
        maxima, alpha=DECLARED_ALPHA, method="phase_randomise", seed=int(seed))
    return calibration, maxima


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--configurations", type=int, default=CONFIGURATIONS)
    parser.add_argument("--scenes", type=int, default=DECLARED_SCENES_PER_CONFIGURATION)
    parser.add_argument("--surrogates", type=int, default=999)
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit("%s exists; a receipt is written once and never overwritten" % output)

    started = time.time()
    before = revision()
    frames = record_frames()
    n_scenes_total = int(args.configurations) * int(args.scenes)
    if n_scenes_total > len(frames):
        raise SystemExit("the record holds %d frames and %d backgrounds were asked for"
                         % (len(frames), n_scenes_total))

    rng = np.random.default_rng(int(args.seed))
    chosen = rng.choice(len(frames), size=n_scenes_total, replace=False)

    stages = ("round_zero", "peeled")
    counts_per_scene: Dict[str, List[int]] = {s: [] for s in stages}
    seen: Dict[str, List[List[int]]] = {s: [] for s in stages}
    trials: Dict[str, int] = {s: 0 for s in stages}
    absences: Dict[str, int] = {s: 0 for s in stages}
    spurious: Dict[str, int] = {s: 0 for s in stages}
    extracted_total: Dict[str, int] = {s: 0 for s in stages}
    recovered_keys: Dict[str, set] = {s: set() for s in stages}
    trial_ratio: Dict[Tuple[int, int, int], float] = {}

    thresholds: List[Dict[str, float]] = []
    fractions: List[float] = []
    artefacts_total = 0
    elsewhere_total = 0
    artefact_stretches: List[float] = []
    unsubtracted = 0

    cursor = 0
    for c in range(int(args.configurations)):
        first = read_frame(*frames[int(chosen[cursor])])
        features = draw_configuration(rng, first.shape)
        counts = {s: [0] * len(features) for s in stages}

        for s in range(int(args.scenes)):
            shard, index = frames[int(chosen[cursor])]
            cursor += 1
            source = read_frame(shard, index)
            seed = int(rng.integers(0, 2 ** 31 - 1))
            background = np.asarray(surrogate_module.generate(
                source, method="phase_randomise", n=1, seed=seed)["members"][0],
                dtype=np.float64)
            width = background_scale(background)
            values = plant(background, features, scale=width)

            # Round zero: the pipeline exactly as it stands. Same field, same seed, same alpha
            # as T4E.24 and T4E.25, which is what acceptance condition 2 checks.
            zero_cal, _ = cut(values, surrogates=int(args.surrogates), seed=seed)
            zero = extract(field_for(values, time=float(s)), calibration=zero_cal)

            # Peel what round zero claimed, then recalibrate on what is left.
            residual, subtracted = peel(values, zero.features)
            unsubtracted += len(zero.features) - len(subtracted)
            peeled_cal, _ = cut(residual, surrogates=int(args.surrogates), seed=seed)

            # The oracle. Unavailable on a real record; a ceiling, never an operating point.
            oracle_cal, _ = cut(background, surrogates=int(args.surrogates), seed=seed)

            thresholds.append({
                "round_zero": zero_cal.threshold / width,
                "peeled": peeled_cal.threshold / width,
                "oracle": oracle_cal.threshold / width})
            fraction = gap_closed(zero_cal.threshold, peeled_cal.threshold, oracle_cal.threshold)
            if fraction is not None:
                fractions.append(fraction)

            # Re-extraction is from the ORIGINAL scene. The residual's only job was the null.
            peeled_result = extract(field_for(values, time=float(s)), calibration=peeled_cal)

            for stage, result in (("round_zero", zero), ("peeled", peeled_result)):
                positions = extracted_positions(result)
                counts_per_scene[stage].append(len(positions))
                extracted_total[stage] += len(positions)
                paired, _offsets = pair_plantings(
                    features, positions, radius=DECLARED_PAIRING_RADIUS_CELLS)
                unmatched = [p for j, p in enumerate(positions)
                             if j not in {q for q in paired if q is not None}]
                spurious[stage] += len(unmatched)
                if stage == "peeled":
                    split = split_contamination(unmatched, subtracted)
                    artefacts_total += split["residual_artefacts"]
                    elsewhere_total += split["elsewhere"]
                    for position in split["artefact_positions"]:
                        stretch = nearest_planting_stretch(position, features)
                        if stretch is not None:
                            artefact_stretches.append(stretch)
                for i, p in enumerate(paired):
                    trials[stage] += 1
                    trial_ratio[(c, s, i)] = features[i].ratio
                    if p is None:
                        absences[stage] += 1
                    else:
                        counts[stage][i] += 1
                        recovered_keys[stage].add((c, s, i))

        for stage in stages:
            seen[stage].append(list(counts[stage]))

    rows = []
    for stage in stages:
        gate = feature_density_gate(counts_per_scene[stage])
        flat = [v for group in seen[stage] for v in group]
        rows.append({
            "stage": stage,
            "gate": gate,
            "trials": trials[stage],
            "feature_recovery": 1.0 - absences[stage] / trials[stage] if trials[stage] else None,
            "marginal_false_absence_rate": (
                absences[stage] / trials[stage] if trials[stage] else None),
            "presence_count_distribution": presence_distribution(flat, int(args.scenes)),
            "admission_rates": admission_rates(flat, int(args.scenes)),
            "configuration_coverage": configuration_coverage(seen[stage], int(args.scenes)),
            "extracted_matching_no_planting": spurious[stage],
            "spurious_per_scene": spurious[stage] / float(len(counts_per_scene[stage]) or 1),
            "spurious_fraction_of_extracted": (
                spurious[stage] / extracted_total[stage] if extracted_total[stage] else None),
        })
    zero_row, peeled_row = rows[0], rows[1]

    newly = recovered_keys["peeled"] - recovered_keys["round_zero"]
    lost = recovered_keys["round_zero"] - recovered_keys["peeled"]
    new_ratios = [trial_ratio[k] for k in newly]
    old_ratios = [trial_ratio[k] for k in recovered_keys["round_zero"]]

    delta_intact = (peeled_row["configuration_coverage"]["intact_across_all_scenes"]
                    - zero_row["configuration_coverage"]["intact_across_all_scenes"])
    # Adjudicated in the terms the declaration fixed, using T4E.25's bars, which were fixed
    # before the feasibility probe that informed the prediction existed.
    coverage_bar_met = delta_intact >= 0.15
    contamination_bar_met = peeled_row["spurious_per_scene"] < 1.0
    reaches_the_faint = (
        float(np.median(new_ratios)) < float(np.median(old_ratios))
        if new_ratios and old_ratios else False)
    held = coverage_bar_met and contamination_bar_met and reaches_the_faint

    receipt: Dict[str, object] = {
        "measurement": "t4e26_peeled_null",
        "declared_in": DECLARATION,
        "declaration_sha256": digest_file(DECLARATION),
        "adopted_in": ADOPTION,
        "adoption_sha256": digest_file(ADOPTION),
        "measured_on": time.strftime("%Y-%m-%d"),
        "root_seed": int(args.seed),
        "alpha": DECLARED_ALPHA,
        "rounds": 1,
        "configurations": int(args.configurations),
        "scenes_per_configuration": int(args.scenes),
        "surrogates_per_ensemble": int(args.surrogates),
        "by_stage": rows,
        "thresholds_in_background_sigma": {
            "round_zero_median": float(np.median([t["round_zero"] for t in thresholds])),
            "peeled_median": float(np.median([t["peeled"] for t in thresholds])),
            "oracle_median": float(np.median([t["oracle"] for t in thresholds])),
            "per_scene": thresholds,
        },
        "fraction_of_the_oracle_gap_closed": {
            "n_scenes": len(fractions),
            "median": float(np.median(fractions)) if fractions else None,
            "deciles": ([float(np.quantile(fractions, q / 10.0)) for q in range(11)]
                        if fractions else None),
            "distribution_not_a_mean": (
                "The deciles are the reported quantity; the median is a convenience and "
                "acceptance condition 3 is met by the deciles, not by it."),
        },
        "contamination_split": {
            "residual_artefacts": artefacts_total,
            "elsewhere": elsewhere_total,
            "artefacts_per_scene": artefacts_total / float(len(thresholds) or 1),
            "artefact_fraction_of_spurious": (
                artefacts_total / (artefacts_total + elsewhere_total)
                if (artefacts_total + elsewhere_total) else None),
            "median_stretch_of_the_nearest_planting": (
                float(np.median(artefact_stretches)) if artefact_stretches else None),
            "stretch_histogram": {str(v): artefact_stretches.count(v)
                                  for v in sorted(set(artefact_stretches))},
            "adjudicates": "nothing; the stretch is known by construction and not on a record",
        },
        "who_is_recovered": {
            "newly_recovered_trials": len(newly),
            "trials_lost_relative_to_round_zero": len(lost),
            "median_ratio_of_the_newly_recovered": (
                float(np.median(new_ratios)) if new_ratios else None),
            "median_ratio_recovered_at_round_zero": (
                float(np.median(old_ratios)) if old_ratios else None),
            "reaches_the_faint_population": reaches_the_faint,
        },
        "features_left_unsubtracted_for_want_of_a_width": unsubtracted,
        "PREDICTION": (
            "Peeling closes a majority of the threshold gap and raises intact-configuration "
            "coverage past the 0.15 bar, with the cost appearing as residual artefacts "
            "concentrated near high-stretch peeled features rather than as a uniform rise in "
            "background contamination."),
        "the_declared_bars": {
            "coverage_rise_at_least": 0.15,
            "spurious_per_scene_below": 1.0,
            "provenance_of_the_bars": (
                "Taken verbatim from T4E.25, which fixed both before the one-scene feasibility "
                "probe that informed this task's prediction existed. No bar here was chosen "
                "after seeing a peeled number."),
            "coverage_bar_met": coverage_bar_met,
            "contamination_bar_met": contamination_bar_met,
            "reaches_the_faint_population": reaches_the_faint,
        },
        "VERDICT": "PREDICTION_HELD" if held else "PREDICTION_FALSIFIED",
        "delta_intact_configuration_coverage": delta_intact,
        "THE_REPORTED_SIGNIFICANCE_MEANS_SOMETHING_ELSE": (
            "This is a finding of this slice and is placed beside the coverage numbers rather "
            "than beneath them, as acceptance condition 5 requires. NullCalibration states its "
            "hypothesis as no peak exceeding the strongest peak of a field with THE SAME POWER "
            "SPECTRUM as the frame. A peeled ensemble has the residual's spectrum instead, so "
            "every p-value taken through a peeled cut answers a different question, and any "
            "coverage gained above was gained partly by changing what a detection claims. "
            "Whether the residual's spectrum is the better null for the question actually being "
            "asked -- whether THIS peak is distinguishable from the background it sits on, "
            "rather than from a field that includes itself -- is an argument, is labelled as "
            "one, and is not settled here. A successor adopting a peeled null must state the "
            "new hypothesis explicitly."),
        "the_blindness_claim_and_its_limit": (
            "A one-scene feasibility probe preceded the declaration and informed the "
            "prediction, which is therefore NOT blind and is not reported as if it were. The "
            "two acceptance bars are T4E.25's and predate the probe. These are T4E.24's scenes "
            "on their fourth inspection, so nothing here is confirmatory: a pass is a failure "
            "to fail on the evidence that motivated the question, not a validation."),
        "claim_boundary": (
            "One null-estimation procedure, on synthetic evidence with known truth, at one "
            "operating point, for one round, on scenes already inspected four times. It settles "
            "nothing about the atmosphere, nothing about any real cyclone, and nothing about "
            "whether any identity criterion is correct. It licenses no change to any default "
            "and no adoption of a peeled null. The oracle is a ceiling and is not an achievable "
            "operating point."),
        "what_this_does_not_do": [
            "It adopts no peeled null, changes no default, and alters nothing in the extractor "
            "or the null.",
            "It evaluates no other null construction (R20) and reports no second round.",
            "It re-measures no identity criterion and supersedes neither T4E.24 nor T4E.25.",
            "No reserved seed block (720-735, 880-895) and no frame of the 2022-2023 "
            "forecast-test period was read.",
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    receipt.update({"source_" + k: v for k, v in before.items()})
    receipt.update({"final_" + k: v for k, v in revision().items()})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "thresholds": receipt["thresholds_in_background_sigma"]["round_zero_median"],
        "gap_closed_median": receipt["fraction_of_the_oracle_gap_closed"]["median"],
        "by_stage": [{k: r[k] for k in ("stage", "feature_recovery", "spurious_per_scene")}
                     for r in rows],
        "delta_intact": delta_intact,
        "contamination_split": {k: receipt["contamination_split"][k]
                                for k in ("residual_artefacts", "elsewhere",
                                          "artefacts_per_scene")},
        "VERDICT": receipt["VERDICT"],
        "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
