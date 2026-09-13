"""Rebuild a TESS acquisition from frozen discovery and verified cached products."""

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
    parser.add_argument("superseded_receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/tess_g17_periodic_raw"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        discovery_raw = args.discovery.read_bytes()
        discovery = json.loads(discovery_raw)
        superseded_raw = args.superseded_receipt.read_bytes()
        superseded = json.loads(superseded_raw)
        cached = {
            product["data_uri"]: product
            for target in superseded["targets"]
            for product in target["products"]
        }

        def read_cached(data_uri: str, maximum: int) -> bytes:
            product = cached[data_uri]
            payload = (args.raw_root / product["cache_file"]).read_bytes()
            if len(payload) != maximum:
                raise ValueError("cached product size differs from frozen discovery")
            if hashlib.sha256(payload).hexdigest() != product["content_sha256"]:
                raise ValueError("cached product differs from superseded receipt digest")
            return payload

        result = acquire_discovered_tess(
            discovery, root=args.raw_root,
            maximum_total_bytes=discovery["predicted_download_bytes"],
            workers=args.workers, download=read_cached)
        result["discovery_file"] = args.discovery.as_posix()
        result["discovery_file_sha256"] = hashlib.sha256(discovery_raw).hexdigest()
        result["supersedes"] = {
            "path": args.superseded_receipt.as_posix(),
            "sha256": hashlib.sha256(superseded_raw).hexdigest(),
            "defect": (
                "All period metadata was copied from the final discovery target because the "
                "acquisition emitter referenced a stale loop variable."),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; corrected acquisition receipts are immutable")
    except (KeyError, OSError, TypeError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d targets rebuilt from verified cache" % result["target_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())