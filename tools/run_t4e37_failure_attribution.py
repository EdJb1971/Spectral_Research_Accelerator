"""Validate or execute the adopted T4E.37 failure-attribution diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.benchmarks.failure_attribution import (  # noqa: E402
    DEFAULT_DECLARATION,
    DEFAULT_OUTPUT,
    evaluate_failure_attribution,
    load_failure_declaration,
)
from src.core.adoption import adoption_state  # noqa: E402
from src.core.errors import SpectralEarthError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the guarded T4E.37 diagnostic")
    parser.add_argument("--declaration", default=str(DEFAULT_DECLARATION))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="validate the frozen declaration without opening the record")
    evaluate = commands.add_parser("evaluate", help="evaluate after a current human adoption")
    evaluate.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    declaration_path = Path(args.declaration)
    if args.command == "plan":
        declaration = load_failure_declaration(declaration_path)
        result = {
            "task": declaration["task"],
            "status": declaration["status"],
            "population_rows": declaration["frozen_population"]["expected_rows"],
            "adoption": adoption_state(declaration_path.parent, declaration_path.name),
            "record_opened": False,
            "network_used": False,
        }
    else:
        result = evaluate_failure_attribution(
            declaration_path, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpectralEarthError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
