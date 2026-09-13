"""Derive what any size-scaled identity criterion must survive, and publish it.

Arithmetic over scene counts bound to results already in the record. It declares no candidate,
opens no reserve, touches no signature or partition, and reaches no network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.core.errors import SpectralEarthError  # noqa: E402
from src.core.publication import publish_new_bytes  # noqa: E402
from src.core.size_scaled_null import (  # noqa: E402
    reassembly_rate, simulate_reassembly, size_scaled_constraints)


DEFAULT_OUTPUT = Path("measurements/t4e44_size_scaled_constraints.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the constraints on a size-scaled identity criterion")
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive", help="print the constraint set")
    derive.add_argument("--json", action="store_true")
    publish = commands.add_parser("publish", help="write the constraint set once")
    publish.add_argument("--output", default=str(DEFAULT_OUTPUT))
    check = commands.add_parser(
        "check", help="check the closed form against a simulation of the same quantity")
    check.add_argument("--trials", type=int, default=40000)
    check.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            exact = reassembly_rate()["per_replicate"]
            simulated = simulate_reassembly(trials=args.trials, seed=args.seed)["per_replicate"]
            print("closed form  : %.6f" % exact)
            print("simulated    : %.6f  (%d trials, seed %d)"
                  % (simulated, args.trials, args.seed))
            print("difference   : %.6f" % abs(exact - simulated))
            return 0

        result = size_scaled_constraints()
        if args.command == "derive":
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            print("STATUS                   : %s" % result["status"])
            print("addresses                : %s" % result["addresses"])
            frozen = result["frozen_partition"]
            print("redistribution surrogate : %.6f per replicate, %.4f over %d, at S = %d"
                  % (frozen["per_replicate"], frozen["over_replicates"],
                     frozen["replicates"], frozen["scenes"]))
            print("scene-count sensitivity  :")
            for row in result["scene_count_sensitivity"]:
                print("    S = %-3d %.6f per replicate, %.4f over %d"
                      % (row["scenes"], row["per_replicate"], row["over_replicates"],
                         row["replicates"]))
            print("constraints              :")
            for item in result["constraints"]:
                print("    %-3s %s" % (item["id"], item["constraint"]))
                print("        %s" % item["forbids"])
            print("receipt sha256           : %s" % result["receipt_sha256"])
            return 0

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            publish_new_bytes(
                output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                "T4E.44 size-scaled constraints")
        except FileExistsError as exc:
            raise SpectralEarthError(str(exc))
        print(json.dumps({"written": str(output).replace("\\", "/"),
                          "receipt_sha256": result["receipt_sha256"],
                          "status": result["status"]}, indent=2))
        return 0
    except SpectralEarthError as error:
        print("REFUSED: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
