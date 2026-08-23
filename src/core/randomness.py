"""Per-task random streams, held in context rather than in process globals (D55, standard E4).

**The defect.** `executor._run_one` used to open a task by calling ``torch.manual_seed(seed)``
and ``np.random.seed(seed)``. Both of those mutate a generator that belongs to the *process*.
Under `SerialExecutor` and `ProcessExecutor` that is harmless, because exactly one task is
between its seeding and its draws at any moment. Under `ThreadExecutor` with more than one
worker it is not: every thread shares the one generator, so worker B's seed lands inside
worker A's seed-to-draw window and A draws from B's stream.

The window is normally microseconds, which is why the property test passed. Held open by 10 ms
of real work - which is what a sweep payload is - thread(8) disagreed with serial in **10 of
10** trials. A result that depends on the scheduler is not a result, and this silently
falsified `roadmap.md`'s claim that a sweep is byte-identical across executor backends.

**The fix, and the option that was rejected.** The cheap option was to refuse ``n_workers > 1``
when seeds are supplied, which would preserve every existing value at the cost of the thread
backend's seeded use. It was rejected deliberately: it removes the symptom and leaves the
design defect - process-global randomness inside a unit of work that the platform is
explicitly allowed to run concurrently - in place for whatever runs next to trip over.

So the streams are per task. A `TaskStreams` is bound to a `contextvars.ContextVar` for the
duration of one task, which gives exactly the isolation required and nothing more:

*   a `ContextVar` set inside a thread is visible only to that thread, so two workers cannot
    see each other's streams no matter how the pool interleaves them;
*   it costs nothing in the serial and process backends, where it is simply one binding;
*   and code deep inside a payload can reach the stream without every intervening signature
    growing a `generator=` parameter. That mattered: the alternative was a mechanical rewrite
    of the whole action layer, and a mechanical rewrite is where defects hide.

**A stream is stateful, and that is the point.** `TaskStreams.torch_generator()` returns the
*same* generator every time within one task, so successive draws advance one stream. A helper
that rebuilt the generator per call - which is what `benchmarks.seeding.SeedBundle` does, and
correctly, for a different purpose - would hand out the same numbers twice.

**On value continuity.** ``torch.Generator().manual_seed(s)`` yields the same sequence as the
global generator after ``torch.manual_seed(s)``, so torch draws are unchanged. NumPy's are
not: the old code seeded the *legacy* `RandomState` via ``np.random.seed``, and this module
uses `default_rng`, which is a different bit generator. Nothing in `src/` outside
`data_layer.adapters`' synthetic fixtures drew from the NumPy global inside a task - every
statistical path already takes an explicit seed and builds its own `default_rng` - so the
change is confined to that fixture, which is documented at its call site.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from src.core.errors import InvalidParameterError


@dataclass
class TaskStreams:
    """The random streams belonging to one unit of work.

    Not frozen and not shared: exactly one task holds one of these, and its generators carry
    that task's position in its own stream. Passing one between threads would reintroduce the
    defect this module exists to remove, so don't.
    """

    seed: int
    _torch: Dict[str, torch.Generator] = dc_field(default_factory=dict, repr=False)
    _numpy: Optional[np.random.Generator] = dc_field(default=None, repr=False)

    def torch_generator(self, device: Any = None) -> torch.Generator:
        """This task's torch stream for ``device``, created once and then advanced.

        Generators are device-bound in torch, so a task that draws on both CPU and CUDA gets
        one per device. They are seeded from the same task seed rather than from a spawn,
        which keeps a CPU-only run's numbers identical to the pre-D55 global-seeded ones.
        """
        key = "cpu" if device is None else str(torch.device(device))
        generator = self._torch.get(key)
        if generator is None:
            generator = (torch.Generator() if device is None
                         else torch.Generator(device=torch.device(device)))
            generator.manual_seed(int(self.seed))
            self._torch[key] = generator
        return generator

    def numpy_generator(self) -> np.random.Generator:
        """This task's NumPy stream, created once and then advanced."""
        if self._numpy is None:
            self._numpy = np.random.default_rng(int(self.seed))
        return self._numpy

    def describe(self) -> Dict[str, Any]:
        return {"seed": int(self.seed),
                "torch_devices": sorted(self._torch),
                "numpy_bit_generator": "PCG64 (numpy.random.default_rng)",
                "scope": "task"}


#: Bound for the duration of one task and unbound afterwards. A `ContextVar` set inside a
#: worker thread is invisible to every other thread, which is the isolation D55 needed.
#: `None` outside a task is a meaningful answer, not a missing one - see `current`.
_CURRENT: contextvars.ContextVar[Optional[TaskStreams]] = contextvars.ContextVar(
    "spectral_earth_task_streams", default=None)


def current() -> Optional[TaskStreams]:
    """This task's streams, or ``None`` when not running inside one.

    ``None`` is deliberately not an error. A function called from a notebook, a test, or an
    API request is not in a seeded task and should behave as it always did - unseeded, and
    *labelled* unseeded. Silently inventing a stream would make an unreproducible result
    indistinguishable from a reproducible one, which is defect D12 all over again.
    """
    return _CURRENT.get()


def require(context: str) -> TaskStreams:
    """This task's streams, or refuse. For code that must not run unseeded."""
    streams = _CURRENT.get()
    if streams is None:
        raise InvalidParameterError(
            "randomness.current()", None,
            "a task random stream. %s draws randomly and must be reproducible, so it has to "
            "run inside `randomness.task_randomness(seed)` - which every executor backend "
            "opens for each task - or be given an explicit seed or generator." % context)
    return streams


def torch_generator(device: Any = None) -> Optional[torch.Generator]:
    """This task's torch stream for ``device``, or ``None`` outside a task."""
    streams = _CURRENT.get()
    return None if streams is None else streams.torch_generator(device)


def numpy_generator() -> Optional[np.random.Generator]:
    """This task's NumPy stream, or ``None`` outside a task."""
    streams = _CURRENT.get()
    return None if streams is None else streams.numpy_generator()


@contextmanager
def task_randomness(seed: Optional[int]) -> Iterator[Optional[TaskStreams]]:
    """Bind fresh streams for one task, and unbind them afterwards.

    ``seed=None`` binds nothing and yields ``None``: an unseeded task stays unseeded rather
    than acquiring a stream derived from something arbitrary.

    The unbind is not optional housekeeping. `ThreadPoolExecutor` reuses its threads, so a
    binding left in place would be inherited by the *next* task on that thread - a subtler
    version of the same defect, and one that only shows up when a later task forgets to seed.
    """
    if seed is None:
        yield None
        return
    streams = TaskStreams(seed=int(seed))
    token = _CURRENT.set(streams)
    try:
        yield streams
    finally:
        _CURRENT.reset(token)
