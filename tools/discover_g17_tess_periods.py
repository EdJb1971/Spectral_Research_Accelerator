"""Record bounded TOI candidates with externally catalogued native periods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.data_layer.exoplanet_period_source import discover_periodic_tess_candidates


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--sector-days", type=float, default=27.0)
    parser.add_argument("--minimum-cycles", type=float, default=8.0)
    args = parser.parse_args(argv)
    try:
        result = discover_periodic_tess_candidates(
            candidate_limit=args.candidates, nominal_sector_days=args.sector_days,
            minimum_cycles=args.minimum_cycles)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; discovery records are immutable")
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d periodic candidates" % result["candidate_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())