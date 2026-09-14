"""Run a coverage report against any registered extractor, on a field you supply.

    # a synthetic field with a known spectrum -- no data of your own required
    python -m tools.coverage_report --output report.json --source fbm --hurst 0.7

    # your own 2-D variable, one report per named variable
    python -m tools.coverage_report --output report.json --source netcdf \
        --path data/cds_downloads/t4e18_vorticity --variable vo --domain atmosphere

    python -m tools.coverage_report --list-extractors

**What this answers.** How much of what is genuinely there does this extractor give back? T4E.24
to T4E.27 answered that for one extractor on one record; the numbers were readable and the
question was not runnable. This makes it runnable on somebody else's instrument and somebody
else's field.

**What it does not answer.** Whether the structure exists in the real field. A coverage report
plants what it then looks for, so it measures the instrument and never the world.

Backgrounds are phase-randomised from the source, so they carry the source's spectrum and none
of its structure -- a planted feature then faces a background as rough as the real one without
competing with real features whose true positions nobody knows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from src.benchmarks.coverage_report import coverage_report
from src.benchmarks.synthetic_backgrounds import DECLARED_FRAME_CEILING, DECLARED_MEDIAN_BAND
from src.core.extraction import EXTRACTORS
from src.statistics import surrogates as surrogate_module


def _fbm_source(n: int, hurst: float, seed: int) -> Callable[[int], np.ndarray]:
    """A synthetic field with a declared Hurst exponent, for a caller with no record."""
    from src.benchmarks.fields import build_fbm
    from src.benchmarks.seeding import derive

    def background_for(index: int) -> np.ndarray:
        # One bundle per background, labelled by its index, so the scenes of a configuration
        # differ in their background and the whole run reproduces from --seed alone.
        bundle = derive("coverage_report.background.%d" % int(index), root_seed=int(seed))
        field = build_fbm(bundle, n=int(n), hurst=float(hurst))
        return np.asarray(field.data.detach().cpu().numpy(), dtype=np.float64)

    return background_for


def _netcdf_source(path: str, variable: str, seed: int,
                   negate: bool) -> Callable[[int], np.ndarray]:
    """Frames from a user's own NetCDF, phase-randomised so only the spectrum survives."""
    import xarray as xr

    shards = sorted(glob.glob(os.path.join(path, "*.nc"))) if os.path.isdir(path) else [path]
    if not shards:
        raise SystemExit("no NetCDF files at %s" % path)

    frames = []
    for shard in shards:
        with xr.open_dataset(shard) as ds:
            if variable not in ds:
                raise SystemExit("%s has no variable %r; it has %s"
                                 % (shard, variable, ", ".join(map(str, ds.data_vars))))
            axis = ds[variable].dims[0]
            frames.extend((shard, i) for i in range(int(ds.sizes[axis])))

    def background_for(index: int) -> np.ndarray:
        shard, i = frames[int(index) % len(frames)]
        with xr.open_dataset(shard) as ds:
            array = ds[variable].isel({ds[variable].dims[0]: i}).values
        array = np.asarray(np.squeeze(array), dtype=np.float64)
        if array.ndim != 2:
            raise SystemExit("variable %r gives a %d-dimensional frame; this report plants "
                             "into a plane" % (variable, array.ndim))
        if negate:
            array = -array
        member = surrogate_module.generate(
            array, method="phase_randomise", n=1, seed=int(seed) + int(index))["members"][0]
        return np.asarray(member, dtype=np.float64)

    return background_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-extractors", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--source", choices=("fbm", "netcdf"), default="fbm")
    parser.add_argument("--path", help="NetCDF file or directory, for --source netcdf")
    parser.add_argument("--variable", default="vo")
    parser.add_argument("--negate", action="store_true",
                        help="negate before extraction, for a field whose features are minima")
    parser.add_argument("--n", type=int, default=161, help="grid size, for --source fbm")
    parser.add_argument("--hurst", type=float, default=0.7)
    parser.add_argument("--extractor", default="local_maximum")
    parser.add_argument("--domain", default="synthetic")
    parser.add_argument("--dataset", default="coverage_report")
    parser.add_argument("--units", default=None)
    parser.add_argument("--configurations", type=int, default=20)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--surrogates", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate-on", choices=("scene", "background"), default="scene")
    parser.add_argument("--median-band", type=int, nargs=2, default=list(DECLARED_MEDIAN_BAND),
                        help="the feature-count band this field warrants. The default is the "
                             "one the atmospheric work declared and is probably wrong for you")
    parser.add_argument("--frame-ceiling", type=int, default=DECLARED_FRAME_CEILING)
    args = parser.parse_args()

    if args.list_extractors:
        for entry in EXTRACTORS.describe():
            print("%-16s %s" % (entry["name"], entry.get("description", "")))
        return 0

    if not args.output:
        parser.error("--output is required unless --list-extractors is given")
    output = Path(args.output)
    if output.exists():
        raise SystemExit("%s exists; a report is written once and never overwritten" % output)

    if args.source == "fbm":
        background_for = _fbm_source(args.n, args.hurst, args.seed)
        variable = "fbm_h%.2f" % args.hurst
        dataset = "%s:fbm" % args.dataset
    else:
        if not args.path:
            parser.error("--path is required for --source netcdf")
        background_for = _netcdf_source(args.path, args.variable, args.seed, args.negate)
        variable = args.variable
        dataset = "%s:%s" % (args.dataset, os.path.basename(str(args.path).rstrip("/\\")))

    report = coverage_report(
        background_for, domain=args.domain, dataset=dataset, variable=variable,
        units=args.units, extractor=args.extractor, configurations=args.configurations,
        scenes=args.scenes, surrogates=args.surrogates, alpha=args.alpha, seed=args.seed,
        calibrate_on=args.calibrate_on,
        median_band=(int(args.median_band[0]), int(args.median_band[1])),
        frame_ceiling=int(args.frame_ceiling))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["VERDICT"] != "MEASURED":
        print(json.dumps({"VERDICT": report["VERDICT"], "gate": report["gate"],
                          "what_this_licenses": report["what_this_licenses"]},
                         indent=2, default=str))
        return 1
    print(json.dumps({
        "VERDICT": report["VERDICT"],
        "extractor": report["instrument"]["extractor"],
        "recall": report["recall"],
        "false_absence_rate": report["false_absence_rate"],
        "survival": {k: v for k, v in report["survival"].items() if k.startswith("k=")},
        "spurious_per_scene": report["spurious_per_scene"],
        "gate": report["gate"],
        "elapsed_seconds": report["elapsed_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
