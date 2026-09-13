"""Prepare an exact G17 review request without deciding or signing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.real_pool_curation import build_curation_review_request


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile-root", action="append", type=Path)
    args = parser.parse_args(argv)
    options = {"roots": tuple(args.profile_root)} if args.profile_root else {}
    try:
        result = build_curation_review_request(packet_path=args.packet, **options)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; review requests are immutable")
    except (OSError, TypeError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d records awaiting human review" % len(result["record_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())