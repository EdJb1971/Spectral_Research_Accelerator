"""One-command consistency audit of the documentation against the code.

`src/tests/test_documentation.py` enforces these facts in CI. This script reports them, which
is a different job: it answers "is the documentation currently honest, and where is it not"
without requiring a test run, and it prints the numbers a slice write-up needs.

    python tools/audit_docs.py            # human-readable
    python tools/audit_docs.py --json     # machine-readable

Exit code is 0 only if every check passes, so it can be used as a pre-commit gate. It exists
because this audit was previously an ad-hoc script retyped from memory each time - and a
check that has to be remembered is a check that will be skipped exactly when it matters.
"""

from __future__ import annotations

import argparse
import ast
import glob
import io
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name: str) -> str:
    return io.open(os.path.join(REPO_ROOT, name), encoding="utf-8").read()


def _source_modules():
    out = []
    for path in glob.glob(os.path.join(REPO_ROOT, "src", "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
        if "__pycache__" in rel or "/tests/" in rel:
            continue
        out.append(rel)
    return sorted(out)


def audit() -> dict:
    architecture = _read("architecture.md")
    roadmap = _read("roadmap.md")

    undocumented_modules = []
    for rel in _source_modules():
        short = rel[len("src/"):]
        base = os.path.basename(rel)
        if base in ("__init__.py", "__main__.py"):
            pkg = os.path.dirname(short)
            if pkg and pkg + "/" in architecture:
                continue
        if short in architecture or base in architecture:
            continue
        undocumented_modules.append(rel)

    routes = re.findall(r'@app\.(?:get|post|put|delete)\("([^"]+)"', _read("src/api/main.py"))
    undocumented_routes = [r for r in routes if r not in architecture]

    rows = re.findall(r"^\| (D\d+) \|(.*)$", architecture, re.M)
    fixed = [d for d, body in rows if "**FIXED**" in body]
    partial = [d for d, body in rows if "PARTIAL" in body]
    open_ids = [d for d, body in rows if "**FIXED**" not in body and "PARTIAL" not in body]

    functions = {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "src", "tests", "test_*.py"))):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        functions[os.path.basename(path)] = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    documented_counts = dict(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"\|\s*`(test_\w+\.py)`\s*\|\s*(\d+)\s*\|", architecture))
    stale_counts = {k: {"documented": documented_counts.get(k), "actual": v}
                    for k, v in functions.items() if documented_counts.get(k) != v}

    def claimed(doc):
        m = re.search(r"Backend test suite \| \*\*(\d+) passed, (\d+) xfailed", doc)
        return (int(m.group(1)), int(m.group(2))) if m else None

    result = {
        "undocumented_modules": undocumented_modules,
        "undocumented_routes": undocumented_routes,
        "defects": {"defined": len(rows), "fixed": len(fixed),
                    "partial": partial, "open": open_ids},
        "test_functions": sum(functions.values()),
        "stale_test_counts": stale_counts,
        "claimed_suite_architecture": claimed(architecture),
        "claimed_suite_roadmap": claimed(roadmap),
    }
    result["ok"] = (
        not undocumented_modules
        and not undocumented_routes
        and not stale_counts
        and result["claimed_suite_architecture"] == result["claimed_suite_roadmap"]
        and result["claimed_suite_architecture"] is not None
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    result = audit()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("undocumented modules : %s" % (result["undocumented_modules"] or "none"))
        print("undocumented routes  : %s" % (result["undocumented_routes"] or "none"))
        d = result["defects"]
        print("defects              : %d defined, %d fixed, partial %s, open %s"
              % (d["defined"], d["fixed"], d["partial"] or "none", d["open"] or "none"))
        print("test functions       : %d" % result["test_functions"])
        print("stale inventory rows : %s" % (result["stale_test_counts"] or "none"))
        print("claimed suite totals : architecture %s / roadmap %s"
              % (result["claimed_suite_architecture"], result["claimed_suite_roadmap"]))
        print("RESULT               : %s" % ("ok" if result["ok"] else "INCONSISTENT"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
