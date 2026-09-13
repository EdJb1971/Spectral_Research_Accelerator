"""Join frozen TOI periods to bounded exact MAST SPOC product metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.data_layer.tess_source import discover_periodic_tess_products


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("period_catalogue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exclude-acquisition", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    try:
        catalogue = json.loads(args.period_catalogue.read_text(encoding="utf-8"))
        if not isinstance(catalogue, dict):
            raise ValueError("period catalogue must contain a JSON object")
        excluded = set()
        for path in args.exclude_acquisition:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict) or not isinstance(receipt.get("targets"), list):
                raise ValueError("excluded acquisition must contain a targets list")
            excluded.update(str(row["tic_id"]) for row in receipt["targets"])
        result = discover_periodic_tess_products(
            catalogue, target_limit=args.targets, excluded_target_ids=excluded,
            workers=args.workers)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; discovery records are immutable")
    except (OSError, TypeError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d targets, %d bytes predicted" % (
        result["selected_targets"], result["predicted_download_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())