"""Verify a human-adopted G17 curation review against the live profile inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.real_pool_curation import load_adopted_curation_review


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--profile-root", action="append", type=Path)
    args = parser.parse_args(argv)
    options = {"roots": tuple(args.profile_root)} if args.profile_root else {}
    try:
        adopted = load_adopted_curation_review(
            args.review, packet_path=args.packet, **options)
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print(json.dumps({
        "review": json.loads(adopted.review.json(by_alias=True)),
        "binding": adopted.binding,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())