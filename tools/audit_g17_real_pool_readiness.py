"""Write the repository-local G17 real-pool readiness assessment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.getcwd())

from src.core.real_pool_readiness import MEASUREMENT_FILE, audit_real_pool_readiness
from src.core.scale_shape_successor import successor_scale_shape_manifest


DEFAULT_OUTPUT = MEASUREMENT_FILE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = audit_real_pool_readiness(successor_scale_shape_manifest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("%s: %d profiles, shortfall %d" % (
        result["status"], result["profile_count"], result["shortfall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())