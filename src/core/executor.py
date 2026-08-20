"""The Executor seam: where work runs, decided in one place (T3.5.19, standard E11).

Science code must not know whether it is running serially, on threads, in processes, or on a
cluster. It calls `executor.map(fn, items)`; the backend is configuration. That is what makes
"runs on a laptop, accelerates on HPC" a deployment choice rather than a rewrite, and it is
why Phase 6's Celery backend is meant to be additive.

**Three traps this seam exists to handle**, all of which produce wrong science rather than
merely slow science:

1.  **Result order.** `as_completed` yields results in finishing order, which varies run to
    run. A sweep that aggregates in that order produces different output on every execution
    even with identical inputs. Every backend here returns results in **submission order**,
    always, and that is what makes serial and parallel byte-identical.

2.  **Seeding.** Workers that share a seed produce identical "random" draws; workers seeded
    from the clock or the PID produce irreproducible ones. Seeds are derived up front with
    `SeedSequence.spawn`, one substream per item, so a run's randomness depends on its index
    and the root seed and on nothing else - not on which worker picked it up.

3.  **Thread oversubscription.** `n_workers` processes each defaulting to one BLAS thread per
    core means `n_workers x n_cores` threads on `n_cores` cores. Throughput *falls*. Each
    worker's thread budget is set on entry (see `core.device.thread_budget`).

A fourth, specific to this codebase: **database sessions are not shareable across processes**.
The executor deliberately carries no database handle. Callers compute in workers and persist
in the parent, or give each worker its own session factory.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from src.core.errors import InvalidParameterError, UserInputError


class ExecutorStartupError(UserInputError):
    """A backend could not start. Almost always a caller-side requirement."""


#: Backends that can be requested by name.
BACKENDS = ("serial", "thread", "process")


@dataclass
class TaskResult:
    """One unit of work: its result or its failure, never silently dropped."""

    index: int
    ok: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    seed: Optional[int] = None
    elapsed_s: float = 0.0

    def raise_if_failed(self) -> Any:
        if not self.ok:
            raise RuntimeError("task %d failed: %s: %s" % (self.index, self.error_type,
                                                           self.error))
        return self.value


class Executor(Protocol):
    """The seam. Backends differ only in how `map` is realised."""

    name: str

    def map(self, fn: Callable[..., Any], items: Sequence[Any],
            seeds: Optional[Sequence[int]] = None) -> List[TaskResult]:
        """Apply ``fn`` to each item, returning results in **submission order**."""


def derive_seeds(root_seed: int, n: int) -> List[int]:
    """One independent substream seed per item, from a single root.

    `SeedSequence.spawn` rather than `root + i`: sequential seeds are not independent seeds,
    and the whole point is that run 7 draws the same numbers whether it ran seventh on one
    core or first on eight.
    """
    children = np.random.SeedSequence(int(root_seed)).spawn(int(n))
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def _run_one(fn, index, item, seed, threads):
    """Worker entry point. Must stay importable at module level for the process backend."""
    from src.core.device import configure_threads

    if threads:
        configure_threads(threads)
    started = time.time()
    try:
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
        value = fn(item, seed) if _takes_seed(fn) else fn(item)
        return TaskResult(index=index, ok=True, value=value, seed=seed,
                          elapsed_s=time.time() - started)
    except Exception as exc:
        # A failure is returned, never raised out of the pool: one bad run in a 20-run sweep
        # must not discard the other 19, and the reason has to survive the process boundary.
        return TaskResult(index=index, ok=False, error=str(exc),
                          error_type=type(exc).__name__,
                          traceback=traceback.format_exc(), seed=seed,
                          elapsed_s=time.time() - started)


def _takes_seed(fn) -> bool:
    try:
        import inspect
        return len(inspect.signature(fn).parameters) >= 2
    except Exception:
        return False


@dataclass
class SerialExecutor:
    """One at a time, in order. The reference against which parallel backends are compared."""

    name: str = "serial"
    n_workers: int = 1
    threads_per_worker: Optional[int] = None

    def map(self, fn, items, seeds=None) -> List[TaskResult]:
        items = list(items)
        seeds = list(seeds) if seeds is not None else [None] * len(items)
        return [_run_one(fn, i, item, seed, self.threads_per_worker)
                for i, (item, seed) in enumerate(zip(items, seeds))]


@dataclass
class ThreadExecutor:
    """Threads. Useful for I/O-bound work; limited for compute by the GIL, though PyTorch
    releases it inside kernels so numeric work does parallelise to a degree."""

    n_workers: int = 4
    threads_per_worker: Optional[int] = None
    name: str = "thread"

    def map(self, fn, items, seeds=None) -> List[TaskResult]:
        items = list(items)
        seeds = list(seeds) if seeds is not None else [None] * len(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = [pool.submit(_run_one, fn, i, item, seed, self.threads_per_worker)
                       for i, (item, seed) in enumerate(zip(items, seeds))]
            # Indexed, not `as_completed`: submission order is the contract.
            return [f.result() for f in futures]


@dataclass
class ProcessExecutor:
    """Separate processes: true parallelism for compute-bound work.

    Everything crossing the boundary must pickle. `fn` therefore has to be a module-level
    callable, not a lambda or a closure - a restriction stated here because the failure
    otherwise appears as an opaque pickling error far from its cause.
    """

    n_workers: int = 4
    threads_per_worker: Optional[int] = None
    name: str = "process"

    def map(self, fn, items, seeds=None) -> List[TaskResult]:
        items = list(items)
        seeds = list(seeds) if seeds is not None else [None] * len(items)
        if not items:
            return []
        threads = self.threads_per_worker
        if threads is None:
            from src.core.device import thread_budget
            threads = thread_budget(self.n_workers)
        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = [pool.submit(_run_one, fn, i, item, seed, threads)
                           for i, (item, seed) in enumerate(zip(items, seeds))]
                return [f.result() for f in futures]
        except (BrokenProcessPool, RuntimeError) as exc:
            # On Windows and macOS the default start method is `spawn`, which re-imports
            # the caller's `__main__`. Without an `if __name__ == "__main__":` guard that
            # re-import runs the whole script again in every worker, and the failure
            # surfaces as an opaque `BrokenProcessPool` far from its cause. Translating it
            # here is the difference between a 20-minute mystery and a one-line fix.
            message = str(exc)
            if ("bootstrapping phase" in message or "freeze_support" in message
                    or isinstance(exc, BrokenProcessPool)):
                raise ExecutorStartupError(
                    "the 'process' backend could not start its workers (%s: %s). "
                    "On this platform multiprocessing uses 'spawn', which re-imports the "
                    "calling module in every worker. Two requirements follow: "
                    "(1) the entry point must be guarded by `if __name__ == \"__main__\":`; "
                    "(2) the mapped function must be importable at module level - not a "
                    "lambda, a closure, or a function defined inside a test. "
                    "Use backend='thread' if neither is practical; it needs no pickling."
                    % (type(exc).__name__, message.strip().splitlines()[-1][:120]),
                    backend="process", n_workers=self.n_workers) from exc
            raise


def get_executor(backend: str = "serial", n_workers: int = 1,
                 threads_per_worker: Optional[int] = None) -> Executor:
    """Build an executor by name. Unknown names are refused with the valid options."""
    backend = (backend or "serial").strip().lower()
    if backend not in BACKENDS:
        raise InvalidParameterError("backend", backend,
                                    "one of %s" % ", ".join(BACKENDS))
    if n_workers < 1:
        raise InvalidParameterError("n_workers", n_workers, "an integer >= 1")
    if backend == "serial":
        return SerialExecutor(threads_per_worker=threads_per_worker)
    if backend == "thread":
        return ThreadExecutor(n_workers=n_workers, threads_per_worker=threads_per_worker)
    return ProcessExecutor(n_workers=n_workers, threads_per_worker=threads_per_worker)


def describe(executor: Executor) -> Dict[str, Any]:
    """Provenance record for how work was distributed."""
    from src.core.device import thread_budget

    n_workers = getattr(executor, "n_workers", 1)
    tpw = getattr(executor, "threads_per_worker", None)
    return {
        "backend": executor.name,
        "n_workers": n_workers,
        "threads_per_worker": tpw if tpw is not None else thread_budget(n_workers),
        "cpu_count": os.cpu_count(),
        "result_order": "submission",
    }
