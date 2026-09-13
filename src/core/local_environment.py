"""Load repository-local backend configuration without a launcher dependency.

`.env.local` is machine-local and may contain secrets.  Values are placed only in the process
environment, are never returned or logged, and never replace an explicitly supplied variable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, MutableMapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"
LOAD_ENV_VAR = "SPECTRALEARTH_LOAD_LOCAL_ENV"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_local_environment(
        path: Optional[Path] = None,
        *, environ: Optional[MutableMapping[str, str]] = None) -> dict[str, Any]:
    """Load a simple KEY=VALUE file with process values taking precedence."""
    target = Path(path) if path is not None else DEFAULT_ENV_FILE
    destination = os.environ if environ is None else environ
    enabled = str(destination.get(LOAD_ENV_VAR, "1")).strip().lower() not in {
        "0", "false", "no", "off",
    }
    report: dict[str, Any] = {
        "status": "DISABLED" if not enabled else "NOT_FOUND",
        "path": str(target), "loaded_names": [], "preserved_names": [],
        "ignored_lines": [],
    }
    if not enabled or not target.is_file():
        return report

    for number, raw in enumerate(target.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            report["ignored_lines"].append(number)
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if not _NAME.fullmatch(name):
            report["ignored_lines"].append(number)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if name in destination:
            report["preserved_names"].append(name)
        else:
            destination[name] = value
            report["loaded_names"].append(name)
    report["status"] = "LOADED"
    return report


__all__ = ["DEFAULT_ENV_FILE", "LOAD_ENV_VAR", "REPO_ROOT", "load_local_environment"]
