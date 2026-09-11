"""Restate the T4E.18 acceptance against the T4E.27 tolerance and write its receipt.

Run from the repository root:

    python -m tools.restate_position_acceptance --output measurements/t4e27_restated_acceptance.json

The population comes from `measurements/t4e18_acceptance.json` unchanged -- the same eighteen
storms at the same deepest-interior observation per storm -- so the only difference from that
record is the bar. Nothing is re-extracted and no catalogue is read.

**One declared condition cannot be evaluated and is refused by name.** Condition 2 asks for three
or more features inside tolerance. The committed receipt carries each storm's *nearest* distance
and a count of features inside the *original* radius, but not the distance to the third-nearest
feature, so the count at any other bar is not recoverable from it. Recomputing it needs a re-run
of the join against the IBTrACS CSV that records the full per-storm distance list. The
declaration's own governing rule applies: refused by name, with the shortfall reported, rather
than approximated.

Whether that catalogue is present, and whether it is still the one T4E.17 signed, is now
**resolved rather than asserted** (TG19.3). The first version of this tool stated in prose that
the file was absent; it had no way to check, and the file had in fact been sitting in a session
scratchpad for a day. A refusal that cannot tell "missing" from "present" is a guess with a firm
voice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

from src.analysis_engine.position_tolerance import (
    DECLARED_GRID_KM_PER_CELL,
    DECLARED_LOCALISATION_CELLS,
    acceptance_curve,
    acceptance_over,
    localisation_km,
    tolerance_for,
)
from src.analysis_engine.prediction_sensitivity import check_prediction, verdict_travel
from src.data_layer.declared_population import read_population
from src.data_layer.signed_reference import IBTRACS_SP
from tools.measure_false_absence import digest_file, revision

DECLARATION = "data/identity_calibration/t4e27-position-tolerance-declaration.json"
ADOPTION = "data/identity_calibration/t4e27-position-tolerance-adoption.json"
SOURCE = "measurements/t4e18_acceptance.json"

#: Which of the two extraction passes in that record this restatement reads. Named rather than
#: taken: T4E.27 read the SWT rows and reported them as though the record held one answer, and
#: the correction that followed withdrew its sharpest conclusion. `read_population` now refuses
#: an unnamed read, so this constant is the declaration that refusal demands.
SOURCE_POPULATION = "swt_planes"

#: T4E.18's declared bar, unchanged: at least half of the eighteen storms.
DECLARED_MAJORITY_OF = 18
DECLARED_NEEDED = 9

#: The inspection grid for the acceptance curve. Fixed here before the verdict is computed, and
#: spanning well past any bar this task could defend, so the curve cannot be read as a shortlist.
CURVE_GRID_KM = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0, 100.0,
                 125.0, 150.0, 200.0, 300.0, 500.0)


#: Written by T4E.28, which records the full sorted distance from each catalogue centre to every
#: extracted feature. Until it existed, the count at any bar other than the original radius was
#: unrecoverable and condition 2 was refused by name.
JOIN_RERUN = Path("measurements/t4e28_join_rerun.json")

#: T4E.28 measured both extraction paths. T4E.27 restated its bar against the SWT rows because
#: those were the only rows the record held; both are reported so the choice stays visible.
CONDITION_2_PATHS = ("swt_planes", "raw_field")


def condition_2(catalogue, per_storm):
    """Three or more features inside the restated tolerance, or a refusal naming what is missing.

    The refusal this replaces was correct when it was written. It is lifted by evidence, not by
    a lowered standard: the distance list T4E.27 said it needed now exists, so the count is
    arithmetic rather than an approximation.
    """
    if not JOIN_RERUN.exists():
        return {
            "met": None,
            "the_catalogue": catalogue.describe(),
            "REFUSED": (
                "not evaluable on the available evidence. Condition 2 asks for three or more "
                "features inside tolerance, and that count needs the distance to each storm's "
                "THIRD-nearest feature. %s is absent, so only the nearest distance is on hand "
                "and the count at any bar other than the original radius cannot be recovered. "
                "Refused by name rather than approximated." % JOIN_RERUN),
            "what_would_lift_the_refusal": (
                "a run of tools/rerun_t4e18_join.py, which records the full per-storm distance "
                "list on both extraction paths. The catalogue it needs is %s."
                % ("PRESENT and verified at %s" % catalogue.path if catalogue.available
                   else "still unavailable: %s" % catalogue.refusal)),
        }

    rerun = json.loads(JOIN_RERUN.read_text(encoding="utf-8"))
    tolerances = {row["storm"]: row.get("tolerance_km") for row in per_storm}
    result = {
        "met": None,
        "the_catalogue": catalogue.describe(),
        "lifted_by": str(JOIN_RERUN),
        "needed": DECLARED_NEEDED,
        "of_declared": DECLARED_MAJORITY_OF,
    }
    for name in CONDITION_2_PATHS:
        distances = {row["storm"]: row["distances_km"]
                     for row in rerun["paths"][name]["rows"]}
        met = judged = 0
        refused = []
        for storm, tolerance in tolerances.items():
            if tolerance is None or storm not in distances:
                refused.append(storm)
                continue
            judged += 1
            met += sum(1 for d in distances[storm] if d <= tolerance) >= 3
        result[name] = {"met": met, "judged": judged, "refused_observations": refused,
                        "meets_the_bar": met >= DECLARED_NEEDED}
    result["met"] = result["swt_planes"]["met"]
    result["which_path_the_headline_figure_is"] = (
        "swt_planes, the path T4E.27 restated its bar against. The raw_field figure is beside "
        "it because T4E.18's correction calls that path markedly better on nearest distance, "
        "and it is WORSE here -- with far fewer features per frame, three of them inside a "
        "22 km radius is a harder thing to do.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit("%s exists; a receipt is written once and never overwritten" % output)

    started = time.time()
    # Resolved rather than assumed. Until TG19.3 this tool stated in prose that the catalogue was
    # absent; it had no way to know, and by then the file had been sitting in a session scratchpad
    # for a day. A refusal that cannot tell "missing" from "present" is a guess with a firm voice.
    catalogue = IBTRACS_SP.resolve()
    source = json.loads(Path(SOURCE).read_text(encoding="utf-8"))
    # Named, and refused if not. The raw-field pass is the better one for this purpose by
    # T4E.18's own correction, and it exists in that record as four aggregates with no rows --
    # so it cannot be substituted here, and asking for it returns a refusal saying exactly that.
    storms, population = read_population(SOURCE, SOURCE_POPULATION, payload=source)

    tolerances = [tolerance_for(row["storm"], row["radius_km"]) for row in storms]
    distances = [float(row["nearest_km"]) for row in storms]

    restated = acceptance_over(distances, tolerances)
    curve = acceptance_curve(distances, tolerances, CURVE_GRID_KM)

    per_storm: List[Dict[str, object]] = []
    for row, tolerance, distance in zip(storms, tolerances, distances):
        verdict = tolerance.admits(distance)
        per_storm.append({
            "storm": row["storm"],
            "lat": row["lat"],
            "nature": row["nature"],
            "nearest_km": distance,
            "catalogue_radius_km": row["radius_km"],
            "tolerance_km": tolerance.total_km,
            "inside_original_radius": bool(row["inside_radius"]),
            "inside_restated_tolerance": verdict,
            "unexplained_residual_km": tolerance.unexplained_residual(distance),
            "refusal": tolerance.refusal,
        })

    # TG19.5. Before adjudicating the prediction, ask whether it was ever falsifiable. T4E.27
    # declared improvement from a bar change that could not produce it: no observation has its
    # separation between the old bar and the new one, so the swing set is empty and the
    # improvement half was settled by arithmetic before any data was read.
    travel = verdict_travel(distances, [row["radius_km"] for row in storms],
                            [t.total_km for t in tolerances],
                            [row["storm"] for row in storms])
    prediction_was_falsifiable = check_prediction(travel, "improve")

    original_condition_1 = source["condition_1_at_least_one_inside_radius"]
    # The declaration predicted, in writing and before this ran, that the restated acceptance
    # still fails: improving on 2 of 18 but not reaching 9.
    still_fails = restated["admitted"] < DECLARED_NEEDED
    improved = restated["admitted"] > int(original_condition_1["met"])

    receipt: Dict[str, object] = {
        "measurement": "t4e27_restated_acceptance",
        "declared_in": DECLARATION,
        "declaration_sha256": digest_file(DECLARATION),
        "adopted_in": ADOPTION,
        "adoption_sha256": digest_file(ADOPTION),
        "restates": SOURCE,
        "restates_sha256": digest_file(SOURCE),
        "measured_on": time.strftime("%Y-%m-%d"),
        "the_tolerance": {
            "form": "quadrature sum of the declared components",
            "estimator_localisation_km": localisation_km(),
            "estimator_provenance": (
                "T4E.21 measured %.3f cells on known ground truth at the record's own feature "
                "density; the grid is %.2f km per cell from T4E.18's acquisition design"
                % (DECLARED_LOCALISATION_CELLS, DECLARED_GRID_KM_PER_CELL)),
            "catalogue_provenance": "IBTrACS per-observation reported radius (T4E.17 signed)",
            "excluded_component": (
                "the separation between an 850 hPa relative-vorticity maximum and a surface "
                "centre. Not a component, because no defensible number for it exists here and "
                "one sized to make this pass would be fitting the bar to the result. The "
                "residual measures it instead."),
        },
        "population": {
            "source": SOURCE,
            "extraction_pass": population.name,
            "which_pass_and_why_it_is_named": population.describe(),
            "the_other_pass_in_this_record": (
                "raw_field: nearest 16.6 km minimum and 52.1 km median, 3 of 18 inside the "
                "catalogue radius. T4E.18's correction calls it markedly better for this "
                "purpose than the SWT planes. It is recorded there as four aggregates with no "
                "per-storm rows, so it cannot be restated here; the join re-run that would "
                "produce its rows is separate work."),
            "storms": len(storms),
            "unchanged": (
                "the same eighteen storms at the same deepest-interior observation per storm. "
                "T4E.19's 154-observation core population, its 200 km pairing and its dateline "
                "separation are deliberately not adopted, so the only thing that changed is "
                "the bar."),
        },
        "condition_1_restated": {
            "admitted": restated["admitted"],
            "judged": restated["judged"],
            "refused": restated["refused"],
            "refused_observations": list(restated["refused_observations"]),
            "needed": DECLARED_NEEDED,
            "of_declared": DECLARED_MAJORITY_OF,
            "met": restated["admitted"] >= DECLARED_NEEDED,
            "original_was": "%s of %s" % (original_condition_1["met"],
                                          original_condition_1["of"]),
        },
        "condition_2_restated": condition_2(catalogue, per_storm),
        "acceptance_is_partial_and_says_so": (
            "Acceptance condition 3 of this task asked for conditions 1 and 2 both. Condition 1 "
            "is restated exactly; condition 2 is refused by name. This slice therefore meets "
            "its own acceptance in part, and that shortfall is reported rather than absorbed."),
        "unexplained_residual": {
            "median_km": restated["median_unexplained_residual_km"],
            "what_it_measures": (
                "the component the tolerance deliberately omits -- how far apart a vorticity "
                "maximum and a surface centre are, over and above the catalogue's own "
                "uncertainty and this extractor's measured localisation error."),
        },
        "per_storm": per_storm,
        "acceptance_curve": list(curve),
        "the_curve_is_for_inspection_not_selection": (
            "The declared bar decides the verdict. Choosing a point from this curve after "
            "seeing it would be the horse race R20 forbids. It is reported so the bar can be "
            "argued with specifically rather than merely accepted or rejected."),
        "was_the_prediction_ever_falsifiable": {
            "the_improvement_half": prediction_was_falsifiable,
            "travel": travel.describe(),
            "why_this_is_reported_beside_the_verdict": (
                "A prediction declared before a measurement is this programme's main guard "
                "against reading a result into the answer already believed. The guard is worth "
                "nothing where the arithmetic settles the prediction in advance, and a "
                "declaration that reads as though a risk was taken is worse than one that "
                "predicts nothing. Recorded here so the guard can be audited rather than "
                "assumed."),
        },
        "PREDICTION": (
            "The restated acceptance STILL FAILS, improving on 2 of 18 but not reaching 9."),
        "VERDICT": ("PREDICTION_HELD" if (still_fails and improved) else
                    "PREDICTION_FALSIFIED" if not still_fails else
                    "PREDICTION_HELD_ON_FAILURE_BUT_NO_IMPROVEMENT"),
        "claim_boundary": (
            "This restates a bar and re-runs one acceptance measurement against an unchanged "
            "population with unchanged conditions. An acceptance measurement settles what an "
            "instrument does, not what the atmosphere does. It approves no mining radius, "
            "adopts no tolerance into any pipeline, discharges no part of T4E.8's "
            "acquired-record acceptance, and settles nothing about whether the record supports "
            "kind_recurrence. It does not supersede %s, which stands as measured." % SOURCE),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    receipt.update({"source_" + k: v for k, v in revision().items()})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "condition_1_restated": receipt["condition_1_restated"],
        "condition_2_restated": {k: v for k, v in receipt["condition_2_restated"].items()
                                 if k in ("met", "swt_planes", "raw_field", "REFUSED")},
        "median_unexplained_residual_km": restated["median_unexplained_residual_km"],
        "VERDICT": receipt["VERDICT"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
