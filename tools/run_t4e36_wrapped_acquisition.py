"""Plan, acquire, materialise, or evaluate the guarded T4E.36 experiment.

Planning is always offline. Acquisition needs both a current maintainer adoption and the
experiment-specific ``--authorise-network`` flag; the CDS layer independently requires its
normal network opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.benchmarks.wrapped_join import evaluate_wrapped_join  # noqa: E402
from src.core.errors import SpectralEarthError  # noqa: E402
from src.core.local_environment import load_local_environment  # noqa: E402
from src.data_layer.cds_source import estimate_cds_storage, plan_monthly_shards  # noqa: E402
from src.data_layer.wrapped_record import (  # noqa: E402
    DEFAULT_COMPLEMENT_DIR,
    DEFAULT_DECLARATION,
    DEFAULT_PARENT_DIR,
    DEFAULT_RECORD_DIR,
    acquire_wrapped_complement,
    load_wrapped_declaration,
    materialise_wrapped_record,
    request_from_declaration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the guarded T4E.36 wrapped acquisition")
    parser.add_argument("--declaration", default=str(DEFAULT_DECLARATION))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="validate and print the frozen offline shard plan")

    acquire = commands.add_parser("acquire", help="download/resume the complementary shards")
    acquire.add_argument("--complement-dir", default=str(DEFAULT_COMPLEMENT_DIR))
    acquire.add_argument(
        "--authorise-network", action="store_true",
        help="separate consent for this adopted experiment; the general network gate also applies")

    materialise = commands.add_parser(
        "materialise", help="offline seam-check and publish the combined wrapped Zarr")
    materialise.add_argument("--parent-dir", default=str(DEFAULT_PARENT_DIR))
    materialise.add_argument("--complement-dir", default=str(DEFAULT_COMPLEMENT_DIR))
    materialise.add_argument("--record-dir", default=str(DEFAULT_RECORD_DIR))
    materialise.add_argument("--time-chunk", type=int, default=32)

    evaluate = commands.add_parser(
        "evaluate", help="run the frozen 18-row extraction and primary raw-field gate")
    evaluate.add_argument("--record-receipt", required=True)
    evaluate.add_argument("--baseline", default="measurements/t4e28_join_rerun.json")
    evaluate.add_argument("--output", default="measurements/t4e36_wrapped_join.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    declaration_path = Path(args.declaration)
    if args.command == "plan":
        declaration = load_wrapped_declaration(declaration_path)
        spec = request_from_declaration(declaration)
        result = {
            "declaration": str(declaration_path).replace("\\", "/"),
            "request": spec.to_provenance(),
            "monthly_shards": [s.to_provenance() for s in plan_monthly_shards(spec)],
            "storage_estimate": estimate_cds_storage(spec),
            "network_used": False,
            "status": declaration["status"],
        }
    elif args.command == "acquire":
        load_local_environment()
        result = acquire_wrapped_complement(
            declaration_path, args.complement_dir,
            explicit_network_authorisation=args.authorise_network)
    elif args.command == "materialise":
        result = materialise_wrapped_record(
            declaration_path, parent_dir=args.parent_dir,
            complement_dir=args.complement_dir, record_dir=args.record_dir,
            time_chunk=args.time_chunk)
    else:
        result = evaluate_wrapped_join(
            declaration_path, args.record_receipt,
            baseline_path=args.baseline, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpectralEarthError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
