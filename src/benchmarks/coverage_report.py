"""A coverage check any extractor, on any field, can be put through.

T4E.24 to T4E.27 measured what *this* extractor loses on *this* record: 37% of features present
are never extracted, 85% of triples never survive six scenes, the detection cut cannot be relaxed
to fix it, and a catalogue join fails by 93 km. Those numbers are now readable. They were not
runnable -- a researcher arriving with their own field and their own extractor could read what
this programme measured and could not ask the same question of their own instrument.

This module is that question, asked generically. It plants a configuration of known features into
independent backgrounds, runs a **registered** extractor over them, and reports what came back:
the recall, the full presence-count distribution, the pair and triple survival the constellation
layer actually consumes, and the contamination. Nothing here is atmospheric. The caller supplies
the backgrounds and declares what the field is; `src/benchmarks/fields.py` can supply synthetic
ones with known spectra for a caller who has no record of their own.

**What it reports is a property of an instrument, never of a world.** A coverage report says what
fraction of structure that is present by construction survives extraction. It says nothing about
whether such structure exists in any real field, and a report from this module is not evidence
about any domain's data.

**What it refuses.** If the planted scenes do not carry features at a density comparable to the
field being stood in for, the report says its evidence is unrepresentative and returns that
instead of a number -- the gate T4E.20 declared in prose and did not implement, and T4E.21 put
into code. The caller may state the band their own field warrants; the default is the one the
atmospheric work declared and is almost certainly wrong for anyone else.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    survival_by_cardinality,
)
from src.benchmarks.synthetic_backgrounds import (
    DECLARED_FRAME_CEILING,
    DECLARED_MEDIAN_BAND,
    feature_density_gate,
)
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError
from src.core.extraction import EXTRACTORS, ExtractionField, calibrate, extract

#: Index space. A coverage report is in cells throughout -- the pairing radius, the separation
#: and the offsets -- so no coordinate vector is supplied and `spacing` is 1.0 by construction.
#: A caller wanting kilometres converts afterwards, with their own grid, which they know and this
#: module does not.
CELL_AXES = (AxisSpec(name="row", role="space", units=None, periodic=False),
             AxisSpec(name="col", role="space", units=None, periodic=False))


def _field(values: np.ndarray, *, domain: str, dataset: str, variable: str,
           units: Optional[str], time_index: float) -> ExtractionField:
    return ExtractionField(
        values=values, axes=CELL_AXES, domain=domain, dataset=dataset, variable=variable,
        units=units, time=float(time_index), representation="raw")


def coverage_report(
    background_for: Callable[[int], np.ndarray],
    *,
    domain: str,
    dataset: str,
    variable: str,
    units: Optional[str] = None,
    extractor: str = "local_maximum",
    configurations: int = 20,
    scenes: int = DECLARED_SCENES_PER_CONFIGURATION,
    surrogates: int = 999,
    alpha: float = 0.05,
    seed: int = 0,
    calibrate_on: str = "scene",
    median_band: Tuple[int, int] = DECLARED_MEDIAN_BAND,
    frame_ceiling: int = DECLARED_FRAME_CEILING,
    cardinalities: Sequence[int] = (2, 3),
) -> Dict[str, object]:
    """Plant known structure, extract it, and report what survived.

    `background_for(i)` returns the i-th background as a 2-D array; it is called
    `configurations * scenes` times and must return an independent field each time, because the
    backgrounds are the only thing allowed to differ between the scenes of one configuration.

    `calibrate_on` is `"scene"` -- what a pipeline does on real data, where the signal is in the
    field whose null is being built -- or `"background"`, which is cleaner but unavailable outside
    a benchmark. T4E.25 measured the difference at a factor of 3.53 in the threshold, so the
    choice is reported rather than assumed.
    """
    if extractor not in EXTRACTORS.names():
        raise InvalidParameterError(
            "coverage_report.extractor", extractor,
            "a registered extractor; known: %s" % ", ".join(EXTRACTORS.names()))
    if calibrate_on not in ("scene", "background"):
        raise InvalidParameterError(
            "coverage_report.calibrate_on", calibrate_on,
            "'scene' (what a pipeline does on real data) or 'background' (a benchmark oracle)")
    if int(configurations) < 1 or int(scenes) < 1:
        raise InvalidParameterError(
            "coverage_report.configurations", (configurations, scenes),
            "at least one configuration and one scene")

    started = time.time()
    rng = np.random.default_rng(int(seed))
    per_scene_counts: List[int] = []
    seen_by_configuration: List[List[int]] = []
    rows: List[Dict[str, object]] = []
    trials = absences = spurious = extracted_total = 0
    cursor = 0

    for c in range(int(configurations)):
        first = np.asarray(background_for(cursor), dtype=np.float64)
        if first.ndim != 2:
            raise InvalidParameterError(
                "coverage_report.background_for", first.ndim,
                "a 2-dimensional background; this report plants into a plane")
        features = draw_configuration(rng, first.shape)
        counts = [0] * len(features)

        for s in range(int(scenes)):
            background = np.asarray(background_for(cursor), dtype=np.float64)
            cursor += 1
            width = background_scale(background)
            if not (width > 0.0) or not math.isfinite(width):
                raise InvalidParameterError(
                    "coverage_report.background_for", width,
                    "a background with non-zero spread. A constant field has no scale for a "
                    "peak-to-background ratio to be quoted against")
            values = plant(background, features, scale=width)
            cut_from = values if calibrate_on == "scene" else background
            calibration = calibrate(cut_from, alpha=float(alpha),
                                    n_surrogates=int(surrogates),
                                    seed=int(rng.integers(0, 2 ** 31 - 1)))
            result = extract(_field(values, domain=domain, dataset=dataset, variable=variable,
                                    units=units, time_index=float(s)),
                             extractor=extractor, calibration=calibration)
            positions = tuple((float(f.location.coords["row"]), float(f.location.coords["col"]))
                              for f in result.features)
            per_scene_counts.append(len(positions))
            extracted_total += len(positions)

            paired, _offsets = pair_plantings(
                features, positions, radius=DECLARED_PAIRING_RADIUS_CELLS)
            spurious += len(positions) - sum(1 for p in paired if p is not None)
            for i, p in enumerate(paired):
                trials += 1
                if p is None:
                    absences += 1
                else:
                    counts[i] += 1

        seen_by_configuration.append(list(counts))
        for i, feature in enumerate(features):
            rows.append({"configuration": c, "feature": i, "seen_in": counts[i],
                         "of": int(scenes), "sigma": feature.sigma, "ratio": feature.ratio,
                         "stretch": feature.stretch})

    gate = feature_density_gate(per_scene_counts, median_band=median_band,
                               frame_ceiling=frame_ceiling)
    report: Dict[str, object] = {
        "report": "coverage",
        "instrument": {
            "extractor": extractor,
            "capabilities": dict(EXTRACTORS.entry(extractor).capabilities),
            "alpha": float(alpha),
            "surrogates": int(surrogates),
            "calibrated_on": calibrate_on,
            "why_that_matters": (
                "T4E.25 measured a scene-calibrated cut sitting 3.53x above the same scene's "
                "background-calibrated one, because the signal's power enters every surrogate "
                "of the field whose null is being built. 'scene' is what a pipeline does on "
                "real data; 'background' is an oracle a benchmark has and a record does not."),
        },
        "field": {"domain": domain, "dataset": dataset, "variable": variable, "units": units},
        "design": {"configurations": int(configurations), "scenes": int(scenes),
                   "seed": int(seed),
                   "geometry_is_identical_across_scenes": (
                       "The configuration is drawn once and reused, so the backgrounds are the "
                       "only thing that differ. That is the most favourable case for recovery: "
                       "real recurrence carries jitter, drift and evolution, each of which can "
                       "only reduce it. Recall here is an UPPER bound.")},
        "gate": gate,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if not gate["passed"]:
        report["VERDICT"] = "EVIDENCE_NOT_REPRESENTATIVE"
        report["what_this_licenses"] = (
            "Nothing. The planted scenes did not carry features at a density comparable to the "
            "band declared for this field, so no coverage measured on them describes what this "
            "extractor does on it. Declare a band this field warrants, or supply backgrounds "
            "that carry one -- but do not tune the background until the number improves.")
        return report

    flat = [v for group in seen_by_configuration for v in group]
    report.update({
        "VERDICT": "MEASURED",
        "trials": trials,
        "features": len(rows),
        "recall": (1.0 - absences / trials) if trials else None,
        "false_absence_rate": (absences / trials) if trials else None,
        "presence_count_distribution": presence_distribution(flat, int(scenes)),
        "admission_rates": admission_rates(flat, int(scenes)),
        "survival": survival_by_cardinality(seen_by_configuration, int(scenes),
                                            cardinalities=cardinalities),
        "dispersion": dispersion(flat, int(scenes), seed=int(seed)),
        "extracted_matching_no_planting": spurious,
        "spurious_per_scene": spurious / float(len(per_scene_counts) or 1),
        "spurious_fraction_of_extracted": (
            spurious / extracted_total if extracted_total else None),
        "rows": rows,
        "claim_boundary": (
            "This is a property of an instrument, not of a world. It reports what fraction of "
            "structure present BY CONSTRUCTION survives this extractor at this operating point. "
            "It says nothing about whether such structure exists in any real field, licenses no "
            "operating point, and is not evidence about any domain's data. The number belongs to "
            "the field and planting it was measured on and is not quoted for another."),
    })
    return report
