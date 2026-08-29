"""Defect D55: the streams belong to the task, and the values did not move.

`test_executor.py` holds the end-to-end proof - a real sweep with a random draw in it, now
byte-identical across `serial`, `thread(2)`, `thread(4)`, `process(2)` and `process(4)`. This
file covers the seam itself, and one claim that is easy to assert here and awkward to assert
there: **the numbers are the same as before**. ``torch.Generator().manual_seed(s)`` yields the
same sequence as the global generator after ``torch.manual_seed(s)``, so a run seeded before
D55 and a run seeded after it draw identically. That mattered enough to choose the fix that
had it - the alternative, refusing threaded seeding, would have preserved every value too but
left process-global randomness sitting inside a unit of work the platform runs concurrently.
"""

import threading

import numpy as np
import pytest
import torch

from src.core import randomness
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.synthetic_generator.perturbation import PerturbationEngine


# ------------------------------------------------------------------ value continuity

def test_a_task_stream_draws_exactly_what_the_global_generator_used_to():
    """The continuity claim, stated as an assertion rather than as a note in a docstring."""
    torch.manual_seed(1234)
    before = torch.randn(8, dtype=torch.float64)

    with randomness.task_randomness(1234) as streams:
        after = torch.randn(8, generator=streams.torch_generator(), dtype=torch.float64)

    assert torch.equal(before, after)


def test_the_stream_advances_rather_than_restarting():
    """A generator rebuilt per call would hand out the same numbers twice.

    `benchmarks.seeding.SeedBundle` does rebuild per call, correctly, because it is deriving
    *independent labelled* streams rather than carrying one task's position through a
    sequence of draws. Getting the two confused would make every draw in a task identical.
    """
    with randomness.task_randomness(99) as streams:
        first = torch.randn(4, generator=streams.torch_generator(), dtype=torch.float64)
        second = torch.randn(4, generator=streams.torch_generator(), dtype=torch.float64)
    assert not torch.equal(first, second)

    torch.manual_seed(99)
    assert torch.equal(first, torch.randn(4, dtype=torch.float64))
    assert torch.equal(second, torch.randn(4, dtype=torch.float64))


def test_the_numpy_stream_is_a_modern_generator_and_says_so():
    with randomness.task_randomness(7) as streams:
        drawn = streams.numpy_generator().standard_normal(5)
        assert streams.describe()["numpy_bit_generator"].startswith("PCG64")
    assert np.allclose(drawn, np.random.default_rng(7).standard_normal(5))


def test_the_same_generator_object_comes_back_for_the_same_device():
    with randomness.task_randomness(3) as streams:
        assert streams.torch_generator() is streams.torch_generator()
        assert streams.torch_generator() is streams.torch_generator("cpu")


# ------------------------------------------------------------------ binding discipline

def test_outside_a_task_there_is_no_stream_and_that_is_an_answer():
    """``None`` is not a missing value here.

    A function called from a notebook, a test or an API request is not in a seeded task. It
    should behave as it always did - unseeded, and *labelled* unseeded. Inventing a stream
    would make an unreproducible result indistinguishable from a reproducible one, which is
    defect D12 restated.
    """
    assert randomness.current() is None
    assert randomness.torch_generator() is None
    assert randomness.numpy_generator() is None


def test_a_none_seed_binds_nothing():
    with randomness.task_randomness(None) as streams:
        assert streams is None
        assert randomness.current() is None


def test_the_binding_is_released_even_when_the_task_raises():
    with pytest.raises(ValueError):
        with randomness.task_randomness(5):
            assert randomness.current() is not None
            raise ValueError("the task failed")
    assert randomness.current() is None


def test_nested_bindings_restore_the_outer_one():
    with randomness.task_randomness(1):
        with randomness.task_randomness(2):
            assert randomness.current().seed == 2
        assert randomness.current().seed == 1
    assert randomness.current() is None


def test_a_binding_in_one_thread_is_invisible_to_another():
    """The property the whole fix rests on.

    A `ContextVar` set inside a thread belongs to that thread. This is asserted directly
    rather than trusted, because if it were false every other test here would still pass and
    D55 would still be live.
    """
    seen = {}
    started = threading.Event()
    release = threading.Event()

    def hold():
        with randomness.task_randomness(111):
            started.set()
            release.wait(timeout=5)
            seen["inner"] = randomness.current().seed

    worker = threading.Thread(target=hold)
    worker.start()
    started.wait(timeout=5)
    try:
        with randomness.task_randomness(222):
            seen["outer"] = randomness.current().seed
    finally:
        release.set()
        worker.join(timeout=5)

    assert seen == {"inner": 111, "outer": 222}


def test_require_refuses_rather_than_drawing_unseeded():
    with pytest.raises(InvalidParameterError, match="task_randomness"):
        randomness.require("the surrogate generator")
    with randomness.task_randomness(8) as streams:
        assert randomness.require("the surrogate generator") is streams


# ------------------------------------------------------------------ the call sites

def test_add_noise_uses_the_task_stream_when_given_neither_seed_nor_generator():
    """The old code reached the same numbers by accident: `randn_like` read the global
    generator that the executor had just seeded. That accident is what D55 removed, so the
    fallback is now explicit and recorded."""
    field = PhysicalField(torch.arange(64, dtype=torch.float64).reshape(8, 8))

    with randomness.task_randomness(31415):
        a = PerturbationEngine.add_noise(field, level=0.5)
    with randomness.task_randomness(31415):
        b = PerturbationEngine.add_noise(field, level=0.5)

    assert torch.equal(a.data, b.data)
    assert a.metadata["seeded"] is True
    assert a.metadata["rng_source"] == "task"


def test_add_noise_outside_a_task_is_still_labelled_unreproducible():
    field = PhysicalField(torch.arange(64, dtype=torch.float64).reshape(8, 8))
    out = PerturbationEngine.add_noise(field, level=0.5)
    assert out.metadata["seeded"] is False
    assert out.metadata["rng_source"] == "none"


def test_an_explicit_seed_still_wins_over_the_ambient_stream():
    """Otherwise a caller's declared seed would be quietly overridden by its surroundings."""
    field = PhysicalField(torch.arange(64, dtype=torch.float64).reshape(8, 8))
    with randomness.task_randomness(1):
        explicit = PerturbationEngine.add_noise(field, level=0.5, seed=777)
    standalone = PerturbationEngine.add_noise(field, level=0.5, seed=777)
    assert torch.equal(explicit.data, standalone.data)
    assert explicit.metadata["rng_source"] == "seed"


def test_the_perturb_action_no_longer_drops_the_seed_it_advertises():
    """Defect D57.

    `perturb_field` declared ``params={'params': '... noise accepts `seed` for
    reproducibility'}`` and then called `add_noise` without it. The request looked honoured
    and was not - the worst shape a reproducibility defect can take, because nothing fails.
    """
    from src.experiment_engine import actions

    args = {"field_data": torch.arange(64, dtype=torch.float64).reshape(8, 8).tolist(),
            "perturbations": [{"type": "noise", "noise_type": "gaussian",
                               "level": 0.5, "seed": 4242}]}
    first = actions.execute("perturb_field", args, torch.device("cpu"))
    second = actions.execute("perturb_field", args, torch.device("cpu"))
    assert first["metrics"] == second["metrics"]

    other = dict(args, perturbations=[dict(args["perturbations"][0], seed=99)])
    assert actions.execute("perturb_field", other,
                           torch.device("cpu"))["metrics"] != first["metrics"]


def test_enable_determinism_records_the_scope_of_the_seed_it_was_given():
    """It used to seed the process. It now reports which task owns the seed instead."""
    from src.core import device as device_policy

    unbound = device_policy.enable_determinism(1234)
    assert unbound["seed"] == 1234
    assert unbound["seed_scope"] == "unbound"

    with randomness.task_randomness(1234):
        bound = device_policy.enable_determinism(1234)
    assert bound["seed_scope"] == "task"

    with randomness.task_randomness(4321):
        mismatched = device_policy.enable_determinism(1234)
    assert mismatched["seed_scope"] == "task_mismatch"
