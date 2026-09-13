"""Plan T4E.39 or, after adoption, freeze its catalogue-only holdout census."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.benchmarks.reference_holdout import (  # noqa: E402
    DEFAULT_CENSUS,
    DEFAULT_DECLARATION,
    build_reference_holdout_census,
    plan_reference_holdout,
)
from src.benchmarks.reference_holdout_fields import (  # noqa: E402
    DEFAULT_DOWNLOAD_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_RECORD_ROOT,
    acquire_holdout_fields,
    evaluate_holdout,
    materialise_holdout_record,
    plan_holdout_fields,
)
from src.core.errors import SpectralEarthError  # noqa: E402
from src.core.local_environment import load_local_environment  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the guarded T4E.39 holdout boundary")
    parser.add_argument("--declaration", default=str(DEFAULT_DECLARATION))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="validate sources without parsing holdout rows")
    census = commands.add_parser(
        "census", help="after adoption, freeze catalogue identities without opening ERA5")
    census.add_argument("--output", default=str(DEFAULT_CENSUS))
    field_plan = commands.add_parser(
        "field-plan", help="plan the census-bound four exact field requests")
    field_plan.add_argument("--census", default=str(DEFAULT_CENSUS))
    acquire = commands.add_parser(
        "acquire", help="download or resume all census-bound exact field shards")
    acquire.add_argument("--census", default=str(DEFAULT_CENSUS))
    acquire.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    acquire.add_argument("--authorise-network", action="store_true")
    materialise = commands.add_parser(
        "materialise", help="join the four exact field streams into an immutable record")
    materialise.add_argument("--census", default=str(DEFAULT_CENSUS))
    materialise.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    materialise.add_argument("--record-root", default=str(DEFAULT_RECORD_ROOT))
    evaluate = commands.add_parser(
        "evaluate", help="run the frozen holdout basin comparison")
    evaluate.add_argument("--census", default=str(DEFAULT_CENSUS))
    evaluate.add_argument("--record-receipt", required=True)
    evaluate.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    declaration = Path(args.declaration)
    if args.command == "plan":
        result = plan_reference_holdout(declaration)
    elif args.command == "census":
        result = build_reference_holdout_census(
            declaration, output_path=Path(args.output))
    elif args.command == "field-plan":
        result = plan_holdout_fields(declaration, args.census)
    elif args.command == "acquire":
        load_local_environment()
        result = acquire_holdout_fields(
            declaration, args.census, download_root=args.download_root,
            explicit_network_authorisation=args.authorise_network)
    elif args.command == "materialise":
        result = materialise_holdout_record(
            declaration, args.census, download_root=args.download_root,
            record_root=args.record_root)
    else:
        result = evaluate_holdout(
            args.record_receipt, declaration, args.census, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpectralEarthError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
