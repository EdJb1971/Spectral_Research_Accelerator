"""Run the Ground-Truth Benchmark Suite from the command line.

    python -m src.benchmarks              # run everything
    python -m src.benchmarks --seed 99    # a different root seed
    python -m src.benchmarks --name fractional_brownian

Exit code is 1 if any check FAILED, so this is usable directly as a CI gate. Pending gates
(NOT_YET_RUNNABLE) do not fail the run, but they are always printed.
"""

import argparse
import sys

from src.benchmarks import format_report, run_all
from src.benchmarks.runner import failures, null_failures
from src.benchmarks.seeding import DEFAULT_ROOT_SEED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.benchmarks")
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED,
                        help="root seed (default %(default)s)")
    parser.add_argument("--name", action="append", dest="names",
                        help="run only this benchmark (repeatable)")
    args = parser.parse_args(argv)

    results = run_all(root_seed=args.seed, names=args.names)
    print(format_report(results))

    if null_failures(results):
        print("\nFAILED: a null benchmark reported structure in data that has none.")
        return 1
    if failures(results):
        print("\nFAILED: one or more benchmarks did not reproduce their known answer.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
