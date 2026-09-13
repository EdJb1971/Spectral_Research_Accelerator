"""Read T4E.8's acceptance, or register one measurement against one of its conditions.

Neither subcommand can accept T4E.8. `state` computes and prints; `register` writes an
unadopted claim that a named person must then sign through the ordinary adoption surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from src.core.errors import SpectralEarthError  # noqa: E402
from src.core.identity_acceptance import (  # noqa: E402
    CONDITION_IDS, DEFAULT_DECLARATION, DEFAULT_REGISTER_DIR, REGISTER_SCHEMA,
    acceptance_state, load_acceptance_declaration)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read or register against T4E.8's acceptance")
    parser.add_argument("--declaration", default=str(DEFAULT_DECLARATION))
    parser.add_argument("--register-dir", default=str(DEFAULT_REGISTER_DIR))
    commands = parser.add_subparsers(dest="command", required=True)

    state = commands.add_parser("state", help="compute the per-condition acceptance state")
    state.add_argument("--json", action="store_true", help="print the whole receipt")

    register = commands.add_parser(
        "register", help="claim that one measurement addresses one condition")
    register.add_argument("--condition", required=True, choices=list(CONDITION_IDS))
    register.add_argument("--measurement", required=True)
    register.add_argument("--addresses", required=True,
                          help="how this measurement meets the condition, in your words")
    register.add_argument("--outcome", choices=["MET", "NOT_MET"], default="MET")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _print_state(state: dict, whole: bool) -> None:
    if whole:
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    print("VERDICT                  : %s" % state["VERDICT"])
    print("declaration status       : %s" % state["declaration_status"])
    print("acceptance adopted       : %s" % state["acceptance_adopted"])
    bound = state["bound_evidence"]
    print("bound evidence           : %d checked, all verified %s"
          % (bound["checked"], bound["all_verified"]))
    for item in bound["drifted"] + bound["absent"]:
        print("  !! %s" % item)
    print("conditions met           : %d of %d"
          % (state["conditions_met"], state["conditions_total"]))
    for condition in state["conditions"]:
        print("  %-3s %-32s %s" % (condition["id"], condition["status"], condition["name"]))
        if condition["why"]:
            print("      %s" % condition["why"])
    print("code may emit accepted   : %s" % state["code_may_emit_accepted"])
    print("receipt sha256           : %s" % state["receipt_sha256"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    declaration = Path(args.declaration)
    register_dir = Path(args.register_dir)
    try:
        if args.command == "state":
            _print_state(acceptance_state(declaration, register_dir), args.json)
            return 0

        body = load_acceptance_declaration(declaration)
        measurement = Path(args.measurement)
        if not measurement.exists():
            raise SpectralEarthError("%s does not exist" % measurement)
        entry = {
            "schema": REGISTER_SCHEMA,
            "task": "T4E.41",
            "condition": args.condition,
            "acceptance_declaration": str(declaration).replace("\\", "/"),
            "acceptance_declaration_sha256": _file_sha256(declaration),
            "measurement": str(measurement).replace("\\", "/"),
            "measurement_sha256": _file_sha256(measurement),
            "addresses": args.addresses,
            "outcome": args.outcome,
            "status": "CLAIMED_NOT_ADOPTED",
            "claim_boundary": (
                "This entry claims that one measurement addresses one condition. It is not a "
                "finding, not an adoption and not an acceptance. The condition stays unmet "
                "until a named person adopts this entry."),
        }
        register_dir.mkdir(parents=True, exist_ok=True)
        path = register_dir / ("%s-%s.json" % (args.condition.lower(), measurement.stem))
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        except FileExistsError:
            raise SpectralEarthError("%s already exists" % path)
        print(json.dumps({"written": str(path).replace("\\", "/"),
                          "status": "CLAIMED_NOT_ADOPTED",
                          "condition": args.condition,
                          "next": "a named person must adopt this entry before %s counts"
                                  % args.condition,
                          "acceptance_status": body.get("status")}, indent=2))
        return 0
    except SpectralEarthError as error:
        print("REFUSED: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
