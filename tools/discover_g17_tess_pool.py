"""Record a bounded metadata-only TESS candidate discovery for G17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.data_layer.tess_source import discover_tess_targets


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sector", required=True, type=int)
    parser.add_argument("--targets", default=48, type=int)
    args = parser.parse_args(argv)
    try:
        result = discover_tess_targets(sector=args.sector, target_limit=args.targets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; discovery records are immutable")
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d targets, %d bytes predicted" % (
        result["selected_targets"], result["predicted_download_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())