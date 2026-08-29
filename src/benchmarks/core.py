"""Benchmark registry: synthetic data whose correct answer is known before it is analysed.

Roadmap T3.5.17, standard E7. This is the scientific test harness every later phase must
pass before it is allowed near real data.

**Why a benchmark differs from the existing synthetic generators.** `synthetic_generator/`
exists to keep the UI alive offline; it produces plausible-looking fields with no declared
truth. A benchmark carries an **analytic answer** computed independently of the code under
test, so a stage can be *wrong* against it rather than merely *different*.

**The two most important entries produce nothing.** The pure-noise and fBm benchmarks have
the known answer "there is no organisation here". Any stage that reports a finding on them
is broken, and that is far more informative than a stage that reports something on ERA5 -
where nobody can say whether the finding is real. A discovery tool without a false-positive
floor is a machine for generating confident nonsense.

**Runnability is declared, not faked.** Several benchmarks gate stages that do not exist yet
(4D tracking, 4E constellation mining). Their checks report ``NOT_YET_RUNNABLE`` naming the
missing stage, rather than being silently skipped or - far worse - quietly passing. The
runner's summary distinguishes the three outcomes, so "all green" can never mean "we did not
look".
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.benchmarks.seeding import DEFAULT_ROOT_SEED, SeedBundle, derive


class Outcome(str, Enum):
    """Three outcomes, never two. Collapsing NOT_YET_RUNNABLE into PASS hides the gap."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_YET_RUNNABLE = "NOT_YET_RUNNABLE"


@dataclass
class CheckResult:
    stage: str
    outcome: Outcome
    detail: str
    measured: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        """True unless the check actually failed.

        NOT_YET_RUNNABLE counts as not-failed so CI stays green while stages are still
        being built, but the runner reports it in its own column - it must never be
        summarised as a pass.
        """
        return self.outcome is not Outcome.FAIL


@dataclass
class Benchmark:
    """A synthetic dataset with a known answer and the stages it gates."""

    name: str
    kind: str                      # "field", "sequence", "cross_domain", or "sample_table"
    description: str
    gates: Tuple[str, ...]         # e.g. ("4C.alpha", "4C.surrogate_null")
    build: Callable[..., Any]      # (bundle, **params) -> field or sequence
    known_answer: Callable[..., Dict[str, Any]]
    checks: Tuple[Callable[[Any, Dict[str, Any]], CheckResult], ...] = ()
    params: Dict[str, Any] = dc_field(default_factory=dict)
    is_null: bool = False          # True when the correct answer is "nothing here"
    tier_shapes: Dict[str, int] = dc_field(
        default_factory=lambda: {"smoke": 64, "laptop": 256, "workstation": 512})

    def bundle(self, root_seed: int = DEFAULT_ROOT_SEED) -> SeedBundle:
        return derive(self.name, root_seed)

    def make(self, root_seed: int = DEFAULT_ROOT_SEED, **overrides: Any):
        """Build the data. Deterministic given ``root_seed`` and parameters."""
        params = dict(self.params)
        params.update(overrides)
        return self.build(self.bundle(root_seed), **params)

    def truth(self, **overrides: Any) -> Dict[str, Any]:
        params = dict(self.params)
        params.update(overrides)
        return self.known_answer(**params)

    def run(self, root_seed: int = DEFAULT_ROOT_SEED, **overrides: Any) -> List[CheckResult]:
        data = self.make(root_seed, **overrides)
        truth = self.truth(**overrides)
        results = []
        for check in self.checks:
            try:
                results.append(check(data, truth))
            except NotImplementedError as exc:
                # Take the stage from the check's own tag, not from the message: the
                # report groups by stage, and using the message as the key made every
                # pending gate its own unique row.
                results.append(CheckResult(
                    getattr(check, "stage", check.__name__),
                    Outcome.NOT_YET_RUNNABLE, str(exc)))
            except Exception as exc:   # a check that crashes is a failure, not an error
                results.append(CheckResult(
                    getattr(check, "stage", check.__name__), Outcome.FAIL,
                    "check raised %s: %s" % (type(exc).__name__, exc)))
        return results


_REGISTRY: Dict[str, Benchmark] = {}


def register_benchmark(bench: Benchmark) -> Benchmark:
    """Register a benchmark by name. Duplicate names are an error, not a silent overwrite."""
    if bench.name in _REGISTRY:
        raise ValueError(
            "benchmark %r is already registered (from %s). Benchmark names appear in CI "
            "output and in the roadmap's acceptance criteria, so a silent overwrite would "
            "make a gate disappear without anyone noticing."
            % (bench.name, _REGISTRY[bench.name].description[:60]))
    _REGISTRY[bench.name] = bench
    return bench


def get_benchmark(name: str) -> Benchmark:
    if name not in _REGISTRY:
        raise KeyError(
            "no benchmark named %r; registered: %s" % (name, sorted(_REGISTRY)))
    return _REGISTRY[name]


def all_benchmarks() -> List[Benchmark]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def benchmarks_gating(stage_prefix: str) -> List[Benchmark]:
    """Every benchmark that gates a stage, e.g. ``benchmarks_gating("4C")``."""
    return [b for b in all_benchmarks()
            if any(g.startswith(stage_prefix) for g in b.gates)]


def null_benchmarks() -> List[Benchmark]:
    """The false-positive floor: benchmarks whose correct answer is "nothing"."""
    return [b for b in all_benchmarks() if b.is_null]


def stage_check(stage: str):
    """Decorator tagging a check function with the stage it exercises."""
    def wrap(fn):
        fn.stage = stage
        return fn
    return wrap


def not_yet_runnable(stage: str, missing: str) -> CheckResult:
    """Declare that a gate exists but the stage it gates does not.

    Deliberately *not* a skip: a skipped test is invisible in a summary line, whereas this
    appears in the report as an outstanding gate with the reason attached.
    """
    return CheckResult(
        stage, Outcome.NOT_YET_RUNNABLE,
        "gate is defined and its known answer is recorded, but %s is not implemented yet"
        % missing)
