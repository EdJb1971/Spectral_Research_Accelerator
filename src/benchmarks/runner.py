"""Run the benchmark suite and report it in a form a researcher can act on.

The report separates three outcomes on purpose. A summary that collapses
``NOT_YET_RUNNABLE`` into ``PASS`` would let "all green" mean "we never looked", which is
the failure this whole module exists to prevent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.benchmarks.core import (
    Benchmark,
    CheckResult,
    Outcome,
    all_benchmarks,
    null_benchmarks,
)
from src.benchmarks.seeding import DEFAULT_ROOT_SEED


def run_all(root_seed: int = DEFAULT_ROOT_SEED,
            names: Optional[List[str]] = None,
            **overrides: Any) -> Dict[str, List[CheckResult]]:
    """Run every registered benchmark (or a named subset) and collect results."""
    selected = [b for b in all_benchmarks() if names is None or b.name in names]
    return {b.name: b.run(root_seed, **overrides) for b in selected}


def summarise(results: Dict[str, List[CheckResult]]) -> Dict[str, int]:
    counts = {o.value: 0 for o in Outcome}
    for checks in results.values():
        for c in checks:
            counts[c.outcome.value] += 1
    return counts


def failures(results: Dict[str, List[CheckResult]]) -> List[Tuple[str, CheckResult]]:
    return [(name, c) for name, checks in results.items() for c in checks
            if c.outcome is Outcome.FAIL]


def null_failures(results: Dict[str, List[CheckResult]]) -> List[Tuple[str, CheckResult]]:
    """Failures on null benchmarks - the ones that mean the platform invents findings."""
    nulls = {b.name for b in null_benchmarks()}
    return [(n, c) for n, c in failures(results) if n in nulls]


def format_report(results: Dict[str, List[CheckResult]]) -> str:
    lines: List[str] = []
    nulls = {b.name for b in null_benchmarks()}
    by_name = {b.name: b for b in all_benchmarks()}

    lines.append("Ground-Truth Benchmark Suite")
    lines.append("=" * 78)
    for name in sorted(results):
        bench: Benchmark = by_name[name]
        tag = "  [NULL - correct answer is 'nothing']" if name in nulls else ""
        lines.append("")
        lines.append("%s%s" % (name, tag))
        lines.append("  %s" % bench.description)
        for c in results[name]:
            lines.append("    %-17s %-32s %s" % (c.outcome.value, c.stage, c.detail))

    counts = summarise(results)
    lines.append("")
    lines.append("-" * 78)
    lines.append("PASS %d   FAIL %d   NOT_YET_RUNNABLE %d"
                 % (counts["PASS"], counts["FAIL"], counts["NOT_YET_RUNNABLE"]))

    nf = null_failures(results)
    if nf:
        lines.append("")
        lines.append("*** %d FAILURE(S) ON NULL BENCHMARKS ***" % len(nf))
        lines.append("A stage is reporting structure in data that contains none. Every "
                     "finding it has ever produced is suspect until this is fixed.")
        for n, c in nf:
            lines.append("    %s / %s: %s" % (n, c.stage, c.detail))

    pending = [(n, c) for n, checks in results.items() for c in checks
               if c.outcome is Outcome.NOT_YET_RUNNABLE]
    if pending:
        lines.append("")
        lines.append("Gates defined but not yet enforceable (the stage does not exist yet):")
        for n, c in pending:
            lines.append("    %s / %s" % (n, c.stage))
    return "\n".join(lines)
