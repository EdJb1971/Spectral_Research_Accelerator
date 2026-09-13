"""Export source-bound G17 profiles from a period-qualified TESS acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.errors import SpectralEarthError
from src.core.tess_profile_export import export_tess_partner_profiles


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/tess_g17_periodic_raw"))
    parser.add_argument("--record-root", type=Path, default=Path("data/tess_g17_records"))
    parser.add_argument("--profile-root", type=Path,
                        default=Path("data/profile_collections/g17_tess"))
    args = parser.parse_args(argv)
    try:
        result = export_tess_partner_profiles(
            args.acquisition, raw_root=args.raw_root, record_root=args.record_root,
            profile_root=args.profile_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FileExistsError:
        parser.error("output already exists; export receipts are immutable")
    except (OSError, ValueError, SpectralEarthError) as error:
        parser.error(str(error))
    print("%d profiles exported" % result["profile_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())