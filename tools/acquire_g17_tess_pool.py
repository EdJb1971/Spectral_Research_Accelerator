"""Acquire the exact TESS products named by a frozen G17 discovery record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.data_layer.tess_source import acquire_discovered_tess


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/tess_g17_raw"))
    parser.add_argument("--maximum-total-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
        if not isinstance(discovery, dict):
            raise ValueError("discovery must contain a JSON object")
        result = acquire_discovered_tess(
            discovery, root=args.raw_root,
            maximum_total_bytes=args.maximum_total_bytes, workers=args.workers)
        result["discovery_file"] = args.discovery.as_posix()
        result["discovery_file_sha256"] = hashlib.sha256(
            args.discovery.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; acquisition receipts are immutable")
    except (OSError, TypeError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d targets, %d bytes acquired" % (
        result["target_count"], result["downloaded_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())