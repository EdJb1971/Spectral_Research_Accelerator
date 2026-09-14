"""Plan or acquire the guarded T4E.38 exact-time MSLP record."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.benchmarks.reference_alignment import (  # noqa: E402
    DEFAULT_DECLARATION,
    DEFAULT_DOWNLOAD_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_RECORD_ROOT,
    acquire_reference_alignment,
    evaluate_reference_alignment,
    materialise_reference_record,
    plan_reference_alignment,
)
from src.core.errors import SpectralEarthError  # noqa: E402
from src.core.local_environment import load_local_environment  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the guarded T4E.38 MSLP acquisition")
    parser.add_argument("--declaration", default=str(DEFAULT_DECLARATION))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="validate the exact offline request")
    acquire = commands.add_parser("acquire", help="download or resume all 36 exact-time shards")
    acquire.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    acquire.add_argument("--authorise-network", action="store_true")
    materialise = commands.add_parser(
        "materialise", help="join the 18 acquired segment pairs into an immutable MSLP record")
    materialise.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    materialise.add_argument("--record-root", default=str(DEFAULT_RECORD_ROOT))
    evaluate = commands.add_parser(
        "evaluate", help="run the frozen 18-row MSLP/vorticity basin comparison")
    evaluate.add_argument("--record-receipt", required=True)
    evaluate.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    declaration = Path(args.declaration)
    if args.command == "plan":
        result = plan_reference_alignment(declaration)
    elif args.command == "acquire":
        load_local_environment()
        result = acquire_reference_alignment(
            declaration, download_root=args.download_root,
            explicit_network_authorisation=args.authorise_network)
    elif args.command == "materialise":
        result = materialise_reference_record(
            declaration, download_root=args.download_root, record_root=args.record_root)
    else:
        result = evaluate_reference_alignment(
            args.record_receipt, declaration, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpectralEarthError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
