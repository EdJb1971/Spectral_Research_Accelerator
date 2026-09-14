"""T4E.35: describe crop-edge support in the committed T4E.28 join receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.benchmarks.edge_support import audit_edge_support


DEFAULT_SOURCE = Path("measurements/t4e28_join_rerun.json")
DEFAULT_OUTPUT = Path("measurements/t4e35_edge_support_audit.json")


def build_receipt(source: Path) -> Dict[str, Any]:
    raw = source.read_bytes()
    parent = json.loads(raw.decode("utf-8"))
    selection = parent["selection"]
    south, north = (float(value) for value in selection["latitude"].split(".."))
    west, east = (float(value) for value in selection["longitude"].split(".."))
    margin = float(selection["interior_margin_degrees"])
    paths = {
        name: audit_edge_support(
            parent["paths"][name]["rows"], south=south, north=north,
            west=west, east=east, margin_degrees=margin)
        for name in ("raw_field", "swt_planes")
    }
    return {
        "schema": "t4e35-edge-support-audit/v1",
        "measurement": "t4e35_edge_support_audit",
        "status": "POST_HOC_DIAGNOSTIC",
        "source": {
            "path": source.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "measurement": parent.get("measurement"),
        },
        "question": (
            "Do catalogue-join errors cluster near any edge of the acquired crop strongly enough "
            "to justify testing a wider or wrapped-longitude acquisition?"),
        "paths": paths,
        "claim_boundary": (
            "This reuses previously inspected T4E.28 rows and is therefore descriptive and "
            "post-hoc. It can identify an acquisition experiment worth declaring; it cannot "
            "establish crop truncation as a cause or alter the failed T4E.18 acceptance."),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    receipt = build_receipt(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": args.output.as_posix(),
        "status": receipt["status"],
        "paths": {name: body["strata"] for name, body in receipt["paths"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())