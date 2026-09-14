"""T4E.28: re-run the T4E.18 join from this repository, against a gate declared before it runs.

    .venv/Scripts/python.exe -u tools/rerun_t4e18_join.py

Writes `measurements/t4e28_join_rerun.json`. Runs both extraction paths over the same declared
population, records the full distance list per storm on each, and adjudicates the figures the
T4E.18 record asserts against what this repository regenerates.

**The extraction parameters below are the substance of this file.** They were recovered on
2026-09-11 from scripts in a session scratchpad under `%TEMP%` -- the only place they had ever
been written down, for figures carried by an adopted acceptance record. Whether the recovery is
correct is not assumed: the SWT path is checked row by row against the 18 rows the record already
holds, and a reconstruction that regenerates a distribution it was not fitted to is the
reconstruction rather than a guess.

Nothing here reaches a network. If the signed catalogue or the acquired record is absent, the run
refuses by name and records no numbers.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.getcwd())

import numpy as np                                                          # noqa: E402
import xarray as xr                                                         # noqa: E402

from src.benchmarks.catalogue_join import (                                 # noqa: E402
    Selection, aggregates, check_gate, compare_rows, deepest_per_storm, join_row,
    read_observations,
)
from src.core.domain import AxisSpec                                        # noqa: E402
from src.core.extraction import ExtractionField, extract                    # noqa: E402
from src.core.representation import plane_field, representation_planes      # noqa: E402
from src.data_layer.signed_reference import (                               # noqa: E402
    SignedReferenceUnavailable, resolve,
)

# ---------------------------------------------------------------- declared before the run

RECORD_GLOB = "data/cds_downloads/t4e18_vorticity/*.nc"
RECORDED = Path("measurements/t4e18_acceptance.json")
DECLARATION = Path("data/identity_calibration/t4e28-join-rerun-declaration.json")
OUTPUT = Path("measurements/t4e28_join_rerun.json")

#: Read from the declaration rather than restated here, so the boundary the run is reported under
#: cannot drift from the boundary it was declared under.
BOUNDARY = json.loads(DECLARATION.read_text(encoding="utf-8"))["the_claim_boundary"] \
    if DECLARATION.exists() else None

SELECTION = Selection(start_date="2018-01-01", end_date="2021-12-31",
                      south=-58.0, north=-18.0, west=140.0, east=180.0)

#: The T4E.18 sign convention. `local_maximum_extractor` finds maxima; southern-hemisphere
#: cyclonic rotation is NEGATIVE relative vorticity, so a maximum-finder on the raw field would
#: locate anticyclones and miss every storm in the catalogue.
NEGATE_THE_FIELD = True

#: Identical on both paths, so that any difference between them is the representation and not the
#: calibration's randomness.
SURROGATES = 99
SEED = 1234

#: Recovered from the scratchpad script that produced the record's per-storm rows. Prediction 2
#: of the declaration is what tests this.
SWT_CONFIG = {"wavelet": "db2", "level": 3}

#: The figures the T4E.18 record asserts, as declared in the T4E.28 gate. Not edited after the
#: run; the git history of this file is the check on that.
DECLARED_RAW = {
    "nearest_km_min": 16.6,
    "nearest_km_median": 52.1,
    "nearest_km_max": 3685.3,
    "condition_1_inside_radius": "3 of 18",
    "features_per_frame": "min 0, median 7, max 13",
}
DECLARED_SWT = {
    "nearest_km_min": 29.9,
    "nearest_km_median": 127.1,
    "nearest_km_max": 2055.9,
    "condition_1_inside_radius": "2 of 18",
    "features_per_frame": "min 73, median 123, max 153",
}


def _observed(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The aggregates, shaped to the keys the gate was declared in."""
    nearest = summary["nearest_km"]
    counts = summary["features_per_frame"]
    condition = summary["condition_1_at_least_one_inside_radius"]
    return {
        "nearest_km_min": round(nearest["min"], 1),
        "nearest_km_median": round(nearest["median"], 1),
        "nearest_km_max": round(nearest["max"], 1),
        "condition_1_inside_radius": "%d of %d" % (condition["met"], condition["of"]),
        "features_per_frame": "min %d, median %d, max %d"
                              % (counts["min"], counts["median"], counts["max"]),
    }


def _positions(features, latitudes, longitudes) -> List[Tuple[float, float]]:
    """Feature cell coordinates carried back to latitude and longitude by linear interpolation."""
    rows = np.arange(len(latitudes))
    columns = np.arange(len(longitudes))
    out = []
    for feature in features:
        r = float(feature.location.coords["row"])
        c = float(feature.location.coords["col"])
        out.append((float(np.interp(r, rows, latitudes)),
                    float(np.interp(c, columns, longitudes))))
    return out


def _field(values, axes) -> ExtractionField:
    return ExtractionField(values=values, axes=axes, domain="reanalysis",
                           dataset="era5_sp_850hPa_vo", variable="relative_vorticity_negated",
                           units="s**-1", time=0.0, time_units="s", representation="identity")


def _raw_path(field: ExtractionField, latitudes, longitudes):
    return _positions(extract(field, n_surrogates=SURROGATES, seed=SEED), latitudes, longitudes)


def _swt_path(field: ExtractionField, latitudes, longitudes):
    found: List[Tuple[float, float]] = []
    for plane in representation_planes("swt", field, SWT_CONFIG):
        if not plane.extractable:
            continue
        features = extract(plane_field(field, "swt", plane, SWT_CONFIG),
                           n_surrogates=SURROGATES, seed=SEED)
        found.extend(_positions(features, latitudes, longitudes))
    return found


PATHS = (("raw_field", _raw_path, DECLARED_RAW),
         ("swt_planes", _swt_path, DECLARED_SWT))


class _RowView:
    """A recorded row dict presented with the attributes `compare_rows` reads."""

    def __init__(self, described: Dict[str, Any]) -> None:
        self.storm = described["storm"]
        self.features = described["features"]
        self.nearest_km = described["nearest_km"]
        self.inside_radius = described["inside_radius"]


def main() -> int:
    if not DECLARATION.exists():
        print("REFUSED: %s is absent. The gate has to exist before the run that is measured "
              "against it." % DECLARATION)
        return 2
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    print("declaration %s  status %s" % (DECLARATION.name, declaration["status"]), flush=True)

    try:
        catalogue = resolve("ibtracs_sp_v04r01")
        catalogue_path = catalogue.require()
    except SignedReferenceUnavailable as refusal:
        print("REFUSED: %s" % refusal)
        return 2
    print("catalogue resolved: %s  signature_verified=%s"
          % (catalogue_path, catalogue.signature_verified), flush=True)

    shards = sorted(glob.glob(RECORD_GLOB))
    if not shards:
        print("REFUSED: no record shards at %s. The acquired vorticity record is absent, and "
              "nothing here downloads it." % RECORD_GLOB)
        return 2
    print("record: %d shards" % len(shards), flush=True)

    observations, census = read_observations(catalogue_path, SELECTION)
    sample = deepest_per_storm(observations, SELECTION)
    print("catalogue rows %d | in the declared population %d | storms sampled %d"
          % (census["rows"], len(observations), len(sample)), flush=True)

    dataset = xr.open_mfdataset(shards, combine="by_coords")
    latitudes = dataset.latitude.values
    longitudes = dataset.longitude.values
    axes = (AxisSpec("row", "space", units="cells", ordinal=0),
            AxisSpec("col", "space", units="cells", ordinal=1))

    results: Dict[str, Any] = {}
    for name, run_path, declared in PATHS:
        print("\n---- %s" % name, flush=True)
        print("%-9s %7s %7s %7s %9s %7s %7s"
              % ("storm", "lon", "radius", "feats", "nearest", "in_rad", ">=3"), flush=True)
        rows = []
        for observation in sample:
            stamp = observation.time.replace(" ", "T")
            frame = dataset.vo.sel(valid_time=np.datetime64(stamp)).squeeze()
            values = np.asarray(frame.values, dtype=np.float64)
            if NEGATE_THE_FIELD:
                values = -values
            row = join_row(observation, run_path(_field(values, axes), latitudes, longitudes))
            rows.append(row)
            print("%-9s %7.1f %7.1f %7d %9s %7d %7s"
                  % (row.storm[:9], row.lon, row.radius_km, row.features,
                     "none" if row.nearest_km is None else "%.1f" % row.nearest_km,
                     row.inside_radius, "yes" if row.inside_radius >= 3 else "no"), flush=True)

        summary = aggregates(rows)
        observed = _observed(summary)
        gate = check_gate(observed, declared)
        results[name] = {"rows": [row.describe() for row in rows],
                         "aggregates": summary, "gate": gate}
        print("\n%s: nearest min %.1f  median %.1f  max %.1f | condition 1: %d of %d | "
              "condition 2: %d of %d"
              % (name, summary["nearest_km"]["min"], summary["nearest_km"]["median"],
                 summary["nearest_km"]["max"],
                 summary["condition_1_at_least_one_inside_radius"]["met"], summary["storms"],
                 summary["condition_2_three_inside_radius"]["met"], summary["storms"]), flush=True)
        print("GATE %s" % gate["verdict"], flush=True)
        for comparison in gate["comparisons"]:
            if not comparison["agrees"]:
                print("   %s: declared %r observed %r"
                      % (comparison["quantity"], comparison["declared"],
                         comparison["observed"]), flush=True)

    # Prediction 2: are the recovered SWT parameters the ones that produced the record?
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))["per_storm"]
    live = [_RowView(r) for r in results["swt_planes"]["rows"]]
    recovery = compare_rows(live, recorded)
    results["swt_planes"]["parameter_recovery"] = {
        "claim": "the wavelet, level and plane selection recovered from the scratchpad are the "
                 "ones that produced measurements/t4e18_acceptance.json",
        "verdict": "CONFIRMED" if recovery["agrees"] else "NOT CONFIRMED",
        "detail": recovery,
    }
    print("\nrecovered SWT parameters against the 18 recorded rows: %s"
          % results["swt_planes"]["parameter_recovery"]["verdict"], flush=True)
    for disagreement in recovery["disagreements"][:10]:
        print("   %s" % disagreement, flush=True)

    gates = [results[name]["gate"]["verdict"] for name, _, _ in PATHS]
    recovered = results["swt_planes"]["parameter_recovery"]["verdict"]
    verdict = ("%s. raw_field %s, swt_planes %s; the recovered SWT extraction parameters are %s "
               "against all 18 recorded rows. The T4E.18 acceptance is untouched and still "
               "FAILS -- a reproduction of a failing measurement is still a failing measurement."
               % ("REPRODUCED" if set(gates) == {"REPRODUCED"} else "NOT REPRODUCED",
                  gates[0], gates[1], recovered))
    OUTPUT.write_text(json.dumps({
        "measurement": "t4e28_join_rerun",
        "declaration": str(DECLARATION),
        "declaration_status": declaration["status"],
        "catalogue": catalogue.describe(),
        "record": "%s (%d shards)" % (RECORD_GLOB, len(shards)),
        "selection": SELECTION.describe(),
        "census": census,
        "extraction": {"negated": NEGATE_THE_FIELD, "n_surrogates": SURROGATES, "seed": SEED,
                       "raw_field": "representation 'identity'",
                       "swt_planes": SWT_CONFIG},
        "paths": results,
        # T4E.22's rule, which this tool broke on its first run: no view states a number
        # without what it may not be used for. The panel showed a null verdict and an empty
        # boundary list for this record, which is the shape a reader is entitled to read as
        # "nothing was concluded" -- on a record whose whole content is a conclusion.
        "VERDICT": verdict,
        "claim_boundary": BOUNDARY,
    }, indent=2), encoding="utf-8")
    print("\nwrote %s" % OUTPUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
