"""Build one source-bound G17 partner profile from a local delimited record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.real_pool_ingress import RecordProfileDeclaration, build_source_bound_profile
from src.core.real_pool_method_review import load_adopted_method_review


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("declaration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--method-review", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.output.name.endswith(".partner-profile.json"):
        parser.error("output must end with .partner-profile.json")
    try:
        raw_declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        if not isinstance(raw_declaration, dict):
            raise ValueError("expected a JSON object")
        declaration = RecordProfileDeclaration(**raw_declaration)
        method_review = load_adopted_method_review(args.method_review)
        result = build_source_bound_profile(
            args.source.read_bytes(), filename=args.source.name,
            delimiter=args.delimiter, declaration=declaration, method_review=method_review)
    except (OSError, TypeError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; profiles are immutable")
    print("%s: %s" % (result["profile"]["record_id"], result["source"]["content_sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())