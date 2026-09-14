"""Ground-Truth Benchmark Suite (roadmap T3.5.17, standard E7).

Importing this package registers every benchmark. See `core.py` for the design rationale;
the short version is that these are synthetic datasets whose correct answer is known
*before* the analysis runs, and that the most important of them have the answer "nothing".
"""

from src.benchmarks.core import (  # noqa: F401
    Benchmark,
    CheckResult,
    Outcome,
    all_benchmarks,
    benchmarks_gating,
    get_benchmark,
    null_benchmarks,
    register_benchmark,
)
from src.benchmarks.seeding import DEFAULT_ROOT_SEED, SeedBundle, derive  # noqa: F401

# Importing for the registration side effect. Ordered field-then-sequence because
# sequences.py reuses helpers from fields.py.
from src.benchmarks import fields  # noqa: F401,E402
from src.benchmarks import sequences  # noqa: F401,E402
from src.benchmarks import cross_domain  # noqa: F401,E402
from src.benchmarks import representation_structure  # noqa: F401,E402
from src.benchmarks import multidomain_flagship  # noqa: F401,E402
from src.benchmarks import identity_certification  # noqa: F401,E402
from src.benchmarks.runner import run_all, format_report  # noqa: F401,E402

__all__ = [
    "Benchmark", "CheckResult", "Outcome", "all_benchmarks", "benchmarks_gating",
    "get_benchmark", "null_benchmarks", "register_benchmark",
    "DEFAULT_ROOT_SEED", "SeedBundle", "derive", "run_all", "format_report",
]
