"""T4E.30: put the T4E.28 join re-run in front of a review panel, or refuse by name.

    .venv/Scripts/python.exe -m tools.bundle_join_rerun --output data/studies/t4e28.json

**Every entry below is a scientific judgement and is written here rather than inferred.**
`measurement_evidence.py` deliberately cannot read a record and decide that some number is a null
result or a contradiction; a module that did would be authoring the evidence it claims to
transport, and the panel would end up reviewing its opinion of the work instead of the work. So
the mapping from the T4E.28 record to evidence entries is stated in full, in one file, where it
can be disagreed with.

The hypothesis is not asserted to predate the evidence -- it is proved from git. T4E.28's
declaration landed in `70640b4`, which carries no measurement, and the run landed in `8290c8b`
seventeen minutes later. That ordering is the whole reason this study can be bundled at all.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.getcwd())

from src.core.claim_ladder import assess_claim_ladder                       # noqa: E402
from src.core.evidence import save_evidence_bundle                          # noqa: E402
from src.core.five_outputs import summarise_evidence                        # noqa: E402
from src.core.measurement_evidence import (                                 # noqa: E402
    EvidenceClaim, RegistrationNotEstablished, bundle_from_measurement,
)

DECLARATION = Path("data/identity_calibration/t4e28-join-rerun-declaration.json")
MEASUREMENT = Path("measurements/t4e28_join_rerun.json")

STUDY_ID = "t4e28-join-rerun"

STATEMENT = (
    "The figures T4E.18's acceptance record asserts about the catalogue join can be regenerated "
    "from this repository, and the extraction parameters that produced them -- recovered from a "
    "session scratchpad under %TEMP% -- are the parameters that produced them.")

PREDICTION = (
    "Declared before the run in 70640b4, which carries no measurement: the raw-field pass "
    "reproduces nearest min 16.6, median 52.1, max 3685.3 km, 3 of 18 inside the catalogue "
    "radius and 0 to 13 features per frame; the SWT pass reproduces min 29.9, median 127.1, max "
    "2055.9, 2 of 18 and 73 to 153 features; and the recovered SWT parameters regenerate all 18 "
    "rows T4E.18 recorded, exactly. Tolerance 0.1 km, the precision the record states. NO "
    "prediction was offered for condition 2 on the raw path, which had never been computed.")


def claims(record: Dict[str, Any]) -> List[EvidenceClaim]:
    """What this measurement is asserted to show, entry by entry, at the author's risk."""
    raw = record["paths"]["raw_field"]
    swt = record["paths"]["swt_planes"]
    raw_totals = raw["aggregates"]
    swt_totals = swt["aggregates"]

    return [
        EvidenceClaim(
            category="replication_results",
            label="both declared gates reproduced",
            status="PASS",
            summary=(
                "Both extraction passes regenerate every aggregate T4E.18's correction states, "
                "within the declared 0.1 km. The gate was committed before the run that it "
                "judges."),
            payload={
                "raw_field": raw["gate"]["verdict"],
                "swt_planes": swt["gate"]["verdict"],
                "tolerance_km": raw["gate"]["tolerance_km"],
                "raw_nearest_km": raw_totals["nearest_km"],
                "swt_nearest_km": swt_totals["nearest_km"],
            }),
        EvidenceClaim(
            category="provenance",
            label="the recovered extraction parameters are the ones that produced the record",
            status="PASS",
            summary=(
                "The SWT wavelet, level and plane selection existed only in a session scratchpad "
                "under %TEMP%, which the operating system clears without notice. Re-running with "
                "them regenerates all 18 rows T4E.18 recorded -- feature count, nearest distance "
                "and inside-radius count -- so the reconstruction is the original rather than a "
                "guess that lands near the same median."),
            payload={
                "verdict": swt["parameter_recovery"]["verdict"],
                "rows_recorded": swt["parameter_recovery"]["detail"]["rows_recorded"],
                "rows_observed": swt["parameter_recovery"]["detail"]["rows_observed"],
                "disagreements": swt["parameter_recovery"]["detail"]["disagreements"],
                "parameters": record["extraction"],
            }),
        EvidenceClaim(
            category="provenance",
            label="the catalogue is the signed one, resolved by digest",
            status="PASS",
            summary=(
                "IBTrACS v04r01 South Pacific, resolved through the signed reference rather than "
                "assumed present. The scratchpad copy that fed the original runs is "
                "byte-identical to the committed one, so a difference in result could not have "
                "been blamed on a revised catalogue."),
            payload={
                "sha256": record["catalogue"]["expected_sha256"],
                "bytes": record["catalogue"]["expected_bytes"],
                "signature_verified": record["catalogue"]["signature_verified"],
                "citation_required": record["catalogue"]["citation_required"],
                "record": record["record"],
            }),
        EvidenceClaim(
            category="contradictory_evidence",
            label="T4E.18's published non-dateline range is false",
            status="PASS",
            summary=(
                "That correction states the nearest feature away from the dateline is '16.6 to "
                "99.3 km', a factor of two to three and not an order of magnitude. The median is "
                "right; the range is not. SETH at longitude 155.9 sits 315.1 km out and HOLA at "
                "175.8 sits 247.7, neither near a boundary and both an order of magnitude above "
                "the 22.1 km radius median. GRETEL yields no feature at all. RUBY, at longitude "
                "179.0 inside the band named as refused by the off-frame rule, is fine at 67.6 "
                "km, so dateline longitude is not sufficient for the failure. The claim stood "
                "because the record held one number per storm and the aggregate was taken over a "
                "subset nobody named."),
            payload={
                "published_range_km": [16.6, 99.3],
                "observed_range_km": [16.6, 315.1],
                "observed_median_km": 35.9,
                "breaks_the_published_range": [
                    {"storm": "SETH", "lon": 155.9, "nearest_km": 315.1},
                    {"storm": "HOLA", "lon": 175.8, "nearest_km": 247.7},
                ],
                "at_a_dateline_longitude_and_unaffected": [
                    {"storm": "RUBY", "lon": 179.0, "nearest_km": 67.6}],
                "yields_no_feature": ["GRETEL"],
            }),
        EvidenceClaim(
            category="null_results",
            label="condition 2 on the raw path, measured for the first time",
            status="FAIL",
            summary=(
                "Three or more features inside the catalogue radius: 0 of 18 on the raw pass "
                "against the SWT pass's 1 of 18, needing 9. No prediction was offered for this "
                "and none is claimed retrospectively -- it had never been computed, and "
                "declaring a guess as a prediction is the failure TG19.5 exists to catch. It "
                "qualifies 'markedly better': the raw pass wins on nearest distance and loses on "
                "the count inside the radius."),
            payload={
                "raw_field": raw_totals["condition_2_three_inside_radius"],
                "swt_planes": swt_totals["condition_2_three_inside_radius"],
                "raw_features_per_frame": raw_totals["features_per_frame"],
                "swt_features_per_frame": swt_totals["features_per_frame"],
            }),
        EvidenceClaim(
            category="failure_states",
            label="the acceptance this reproduces still fails",
            status="PASS",
            summary=(
                "T4E.18 FAILED and a reproduction of a failing measurement is still a failing "
                "measurement. Condition 1 is met by 3 of 18 on the raw pass against a declared "
                "bar of 9. One storm carries a catalogue radius of exactly 0.00 and could never "
                "have passed however good the extraction was; one frame yields no feature at "
                "all. Both stay in every denominator."),
            payload={
                "raw_condition_1": raw_totals["condition_1_at_least_one_inside_radius"],
                "swt_condition_1": swt_totals["condition_1_at_least_one_inside_radius"],
                "storms_with_no_feature": raw_totals["storms_with_no_feature"],
                "zero_radius_observation": "LINDA",
                "verdict": record["VERDICT"],
            }),
        EvidenceClaim(
            category="uncertainty",
            label="what the catalogue's own positional uncertainty is",
            status="PASS",
            summary=(
                "Each storm's radius is the furthest of its agency fixes from the reported "
                "centre, so a feature inside it is one the catalogue could not distinguish from "
                "the centre. An observation with fewer than two independent fixes has no radius "
                "that means anything and is refused rather than given zero."),
            payload={
                "radius_definition": record["selection"]["radius_definition"],
                "minimum_agency_fixes": record["selection"]["minimum_agency_fixes"],
                "population": record["selection"]["dates"],
                "census": record["census"],
            }),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="where to publish the bundle; never overwritten")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report without publishing")
    arguments = parser.parse_args()

    record = json.loads(MEASUREMENT.read_text(encoding="utf-8"))
    try:
        bundle, registration = bundle_from_measurement(
            study_id=STUDY_ID, hypothesis_id="T4E.28", statement=STATEMENT,
            prediction=PREDICTION, declaration=DECLARATION, measurement=MEASUREMENT,
            claims=claims(record))
    except RegistrationNotEstablished as refusal:
        print("REFUSED: %s" % refusal)
        return 2

    print("registration established from git")
    print("  declaration %s  %s" % (registration.declaration.commit[:7],
                                    registration.declaration.committed_at))
    print("  measurement %s  %s" % (registration.measurement.commit[:7],
                                    registration.measurement.committed_at))
    print("\nentries: %d" % bundle.revision)
    for entry in bundle.entries:
        print("  %-24s %-6s %s" % (entry.category, entry.status, entry.label))

    assessment = assess_claim_ladder(bundle)
    print("\nclaim ladder: %s" % assessment.rung)
    for gate in assessment.gates:
        if not gate.satisfied:
            print("  unmet  %-34s %s" % (gate.name, gate.requirement))

    outputs = summarise_evidence(bundle)
    print("\nthe five outputs are computable over this bundle: %s"
          % ", ".join(sorted(outputs.to_mapping())))

    if arguments.dry_run:
        print("\ndry run; nothing published")
        return 0

    target = Path(arguments.output)
    if target.exists():
        print("\nREFUSED: %s exists. A bundle is immutable and is never overwritten." % target)
        return 2
    digest = save_evidence_bundle(target, bundle)
    print("\npublished %s\nbundle_sha256 %s" % (target, digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
