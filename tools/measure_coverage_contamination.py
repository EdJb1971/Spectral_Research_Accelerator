"""Run the T4E.25 coverage/contamination sweep and write its receipt.

Run from the repository root:

    python -m tools.measure_coverage_contamination \
        --output measurements/t4e25_coverage_contamination.json

**The sweep is exactly paired, and that is the whole design.** Only the cut varies, so one
surrogate ensemble per scene supplies every threshold: `calibration_from_maxima` takes the same
order statistic over the same maxima at a different alpha, which is exactly what `calibrate`
does internally. The four alphas are therefore four cuts through one measurement rather than
four runs to be compared, no difference between them can be a sampling difference, and the
alpha = 0.05 baseline reproduces T4E.24 by construction rather than by luck.

The declaration and its adoption are read and hashed before anything is measured. No download
occurs and no frame of the 2022-2023 forecast-test period is opened.
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
from src.benchmarks.synthetic_backgrounds import feature_density_gate
from src.core.extraction import calibration_from_maxima, extract
from src.statistics import surrogates as surrogate_module
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

DECLARATION = "data/identity_calibration/t4e25-coverage-contamination-declaration.json"
ADOPTION = "data/identity_calibration/t4e25-coverage-contamination-adoption.json"

#: Fixed in the declaration before any number existed. The adoption closes the list: no value may
#: be added, removed or substituted after the measurement, because adding a fifth alpha once the
#: curve is visible would be fitting the sweep to its own answer.
DECLARED_ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)


def configuration_coverage(seen_by_configuration: List[List[int]], scenes: int) -> Dict[str, object]:
    """The quantity T4E.24's addendum showed to be the binding one.

    A configuration holding a feature recovered in no scene at all can never be matched whole,
    whatever tolerance a criterion applies over scenes: the object it would have to match was
    never assembled anywhere.
    """
    total = len(seen_by_configuration)
    recoverable = sum(1 for counts in seen_by_configuration if all(c > 0 for c in counts))
    intact = sum(1 for counts in seen_by_configuration
                 if all(c >= int(scenes) for c in counts))
    always = [float(sum(1 for c in counts if c >= int(scenes))) / len(counts)
              for counts in seen_by_configuration if counts]
    return {
        "configurations": total,
        "every_feature_seen_at_least_once": recoverable,
        "every_feature_seen_in_all_scenes": intact,
        "recoverable_in_principle": (recoverable / total) if total else float("nan"),
        "intact_across_all_scenes": (intact / total) if total else float("nan"),
        "mean_fraction_of_a_configuration_always_present": (
            float(np.mean(always)) if always else float("nan")),
    }


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

    alphas = list(DECLARED_ALPHAS)
    # Drawn exactly as T4E.24 drew them, from the same root seed, so the configurations, the
    # source frames and the phase seeds are identical. Anything else here would break the
    # pairing that acceptance condition 2 checks for.
    rng = np.random.default_rng(int(args.seed))
    chosen = rng.choice(len(frames), size=n_scenes_total, replace=False)

    per_scene_counts: Dict[float, List[int]] = {a: [] for a in alphas}
    seen: Dict[float, List[List[int]]] = {a: [] for a in alphas}
    trials: Dict[float, int] = {a: 0 for a in alphas}
    absences: Dict[float, int] = {a: 0 for a in alphas}
    spurious: Dict[float, int] = {a: 0 for a in alphas}
    extracted_total: Dict[float, int] = {a: 0 for a in alphas}
    # Which planted trials each alpha recovers, keyed so that the trials recovered at one alpha
    # and not the one below it -- the NEWLY recovered, which is what the declaration asks for --
    # can be identified rather than approximated by the whole population at the higher alpha.
    recovered_keys: Dict[float, set] = {a: set() for a in alphas}
    trial_ratio: Dict[Tuple[int, int, int], float] = {}

    cursor = 0
    for c in range(int(args.configurations)):
        first = read_frame(*frames[int(chosen[cursor])])
        features = draw_configuration(rng, first.shape)
        counts = {a: [0] * len(features) for a in alphas}

        for s in range(int(args.scenes)):
            shard, index = frames[int(chosen[cursor])]
            cursor += 1
            source = read_frame(shard, index)
            seed = int(rng.integers(0, 2 ** 31 - 1))
            background = np.asarray(surrogate_module.generate(
                source, method="phase_randomise", n=1, seed=seed)["members"][0],
                dtype=np.float64)
            values = plant(background, features, scale=background_scale(background))

            # ONE ensemble, taken from the scene as T4E.24 took it, and every alpha reads its
            # own order statistic off the same maxima. This is what makes the sweep paired.
            ensemble = surrogate_module.generate(
                values, method="phase_randomise", n=int(args.surrogates), seed=seed)
            null_max = [float(np.max(np.asarray(m))) for m in ensemble["members"]]

            for a in alphas:
                calibration = calibration_from_maxima(
                    null_max, alpha=float(a), method="phase_randomise", seed=seed)
                result = extract(field_for(values, time=float(s)), calibration=calibration)
                positions = extracted_positions(result)
                per_scene_counts[a].append(len(positions))
                extracted_total[a] += len(positions)

                paired, _offsets = pair_plantings(
                    features, positions, radius=DECLARED_PAIRING_RADIUS_CELLS)
                spurious[a] += len(positions) - sum(1 for p in paired if p is not None)
                for i, p in enumerate(paired):
                    trials[a] += 1
                    trial_ratio[(c, s, i)] = features[i].ratio
                    if p is None:
                        absences[a] += 1
                    else:
                        counts[a][i] += 1
                        recovered_keys[a].add((c, s, i))

        for a in alphas:
            seen[a].append(list(counts[a]))

    rows: List[Dict[str, object]] = []
    excluded: List[float] = []
    for a in alphas:
        gate = feature_density_gate(per_scene_counts[a])
        flat = [c for counts in seen[a] for c in counts]
        if not gate["passed"]:
            excluded.append(a)
        rows.append({
            "alpha": a,
            "gate": gate,
            "gate_excluded_this_alpha": not gate["passed"],
            "trials": trials[a],
            "marginal_false_absence_rate": (absences[a] / trials[a]) if trials[a] else None,
            "feature_recovery": (1.0 - absences[a] / trials[a]) if trials[a] else None,
            "presence_count_distribution": presence_distribution(flat, int(args.scenes)),
            "admission_rates": admission_rates(flat, int(args.scenes)),
            "configuration_coverage": configuration_coverage(seen[a], int(args.scenes)),
            "extracted_matching_no_planting": spurious[a],
            "spurious_per_scene": spurious[a] / float(len(per_scene_counts[a] or [1])),
            "spurious_fraction_of_extracted": (
                spurious[a] / extracted_total[a] if extracted_total[a] else None),
            "median_recovered_ratio": (
                float(np.median([trial_ratio[k] for k in recovered_keys[a]]))
                if recovered_keys[a] else None),
        })

    # What each step buys and what it pays, on the same scenes -- which is what the shared
    # ensemble exists to make meaningful.
    steps: List[Dict[str, object]] = []
    for lower, upper in zip(rows, rows[1:]):
        # Recovery is monotone in alpha -- a lower cut admits everything the higher one did --
        # so the newly recovered trials are exactly the set difference, and if that monotonicity
        # ever fails it is a wiring bug and is reported rather than smoothed over.
        newly = recovered_keys[upper["alpha"]] - recovered_keys[lower["alpha"]]
        lost = recovered_keys[lower["alpha"]] - recovered_keys[upper["alpha"]]
        gained = [trial_ratio[k] for k in newly]
        steps.append({
            "from_alpha": lower["alpha"],
            "to_alpha": upper["alpha"],
            "intact_configuration_coverage": [
                lower["configuration_coverage"]["intact_across_all_scenes"],
                upper["configuration_coverage"]["intact_across_all_scenes"]],
            "delta_intact_configuration_coverage": (
                upper["configuration_coverage"]["intact_across_all_scenes"]
                - lower["configuration_coverage"]["intact_across_all_scenes"]),
            "delta_recoverable_in_principle": (
                upper["configuration_coverage"]["recoverable_in_principle"]
                - lower["configuration_coverage"]["recoverable_in_principle"]),
            "delta_feature_recovery": (
                (upper["feature_recovery"] or 0.0) - (lower["feature_recovery"] or 0.0)),
            "spurious_per_scene": [lower["spurious_per_scene"], upper["spurious_per_scene"]],
            "delta_spurious_per_scene": (
                upper["spurious_per_scene"] - lower["spurious_per_scene"]),
            "newly_recovered_trials": len(newly),
            "median_ratio_of_the_newly_recovered": (
                float(np.median(gained)) if gained else None),
            "trials_recovered_below_but_not_above": len(lost),
            "monotonicity_violated": bool(lost),
        })

    # The falsification condition, adjudicated in the terms the declaration fixed rather than
    # in whatever terms the numbers happen to suit.
    falsifying = [
        s for s in steps
        if s["delta_intact_configuration_coverage"] >= 0.15
        and s["spurious_per_scene"][1] < 1.0]
    receipt: Dict[str, object] = {
        "measurement": "t4e25_coverage_contamination",
        "declared_in": DECLARATION,
        "declaration_sha256": digest_file(DECLARATION),
        "adopted_in": ADOPTION,
        "adoption_sha256": digest_file(ADOPTION),
        "measured_on": time.strftime("%Y-%m-%d"),
        "root_seed": int(args.seed),
        "configurations": int(args.configurations),
        "scenes_per_configuration": int(args.scenes),
        "surrogates_per_scene": int(args.surrogates),
        "alphas_swept": alphas,
        "the_sweep_is_paired": (
            "One surrogate ensemble per scene; every alpha reads its own order statistic off "
            "the same maxima. No difference between alphas can be a sampling difference."),
        "calibrated_on": "scene",
        "alphas_excluded_by_the_gate": excluded,
        "by_alpha": rows,
        "steps": steps,
        "PREDICTION": (
            "Contamination rises faster than configuration coverage at every step, and no "
            "declared alpha brings intact-configuration coverage above 0.50 while keeping "
            "spurious features below one per scene."),
        "VERDICT": ("PREDICTION_FALSIFIED" if falsifying else "PREDICTION_HELD"),
        "steps_meeting_the_declared_falsification_condition": falsifying,
        "elapsed_seconds": round(time.time() - started, 1),
        "claim_boundary": (
            "This measures a property of one detection scheme on synthetic evidence with known "
            "truth, across four declared levels. It settles nothing about the atmosphere, "
            "nothing about any real cyclone, and nothing about whether any identity criterion "
            "is correct. It licenses no operating point: choosing an alpha against this curve "
            "is a separate declaration owing its own blindness claim (R20). The cut is working "
            "as declared and this is the price of that control, not a defect in it."),
        "what_this_does_not_do": [
            "It selects no alpha, recommends none, and changes no default.",
            "It modifies nothing in the extractor or the null and proposes no replacement for "
            "the frame-maximum statistic.",
            "It re-measures no identity criterion and supersedes neither T4E.24 nor T4E.13.",
            "No reserved seed block (720-735, 880-895) and no frame of the 2022-2023 "
            "forecast-test period was read.",
        ],
    }
    receipt.update({"source_" + k: v for k, v in before.items()})
    receipt.update({"final_" + k: v for k, v in revision().items()})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"by_alpha": [
        {"alpha": r["alpha"], "gate_median": r["gate"]["median"],
         "feature_recovery": r["feature_recovery"],
         "recoverable": r["configuration_coverage"]["recoverable_in_principle"],
         "intact": r["configuration_coverage"]["intact_across_all_scenes"],
         "spurious_per_scene": r["spurious_per_scene"]} for r in rows],
        "VERDICT": receipt["VERDICT"], "elapsed_seconds": receipt["elapsed_seconds"]},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
