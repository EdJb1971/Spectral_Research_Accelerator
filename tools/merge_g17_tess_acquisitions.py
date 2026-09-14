"""Merge immutable period-qualified TESS acquisition receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.tess_pool_assessment import merge_tess_acquisitions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("receipts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        result = merge_tess_acquisitions(args.receipts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; merged receipts are immutable")
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d unique targets" % result["target_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())