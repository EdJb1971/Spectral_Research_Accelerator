"""Measure candidate G17 marginals from a period-qualified TESS acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.tess_pool_assessment import assess_tess_pool_acquisition


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/tess_g17_periodic_raw"))
    args = parser.parse_args(argv)
    try:
        result = assess_tess_pool_acquisition(args.acquisition, args.raw_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; assessments are immutable")
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d candidates; %d records reach 48 alternatives" % (
        result["candidate_count"], result["records_reaching_minimum"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())