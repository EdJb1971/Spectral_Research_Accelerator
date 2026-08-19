"""Seed discipline: reproducible, independent random streams (roadmap T3.5.12, standard E4).

A benchmark whose output is not bit-reproducible is not a benchmark - it is an anecdote. And
a benchmark suite that reuses one global RNG across generators is worse than one that does
not seed at all, because adding a generator silently changes every result computed after it.

Three specific traps this module exists to avoid.

**1. Python's `hash()` on a string is randomised per process.** Deriving a stream as
``SeedSequence([seed, hash(label)])`` gives different numbers on every interpreter run
unless ``PYTHONHASHSEED`` happens to be fixed. That is a reproducibility bug that passes
every test inside a single session and fails across sessions - the worst kind. A stable
``zlib.crc32`` of the UTF-8 bytes is used instead.

**2. Sequential seeds are not independent seeds.** ``manual_seed(1)``, ``manual_seed(2)``
… produce streams with detectable correlation for some generators. ``SeedSequence.spawn``
exists precisely to turn one root seed into many statistically independent children, and it
is what makes a `process`-backend sweep (T3.5.19) give the same answer as a serial one.

**3. torch and numpy are separate streams.** Both are used across the platform, so both are
derived here from the same root and returned together. Drawing from one and not the other
still leaves the other reproducible.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

#: The root seed used when a caller does not supply one. Fixed rather than random, because
#: a benchmark suite must default to reproducible, not to "probably fine".
DEFAULT_ROOT_SEED = 20260819


def stable_label_hash(label: str) -> int:
    """Deterministic 32-bit hash of a label, stable across processes and Python versions.

    ``hash()`` is deliberately not used - see the module docstring.
    """
    return zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF


@dataclass(frozen=True)
class SeedBundle:
    """A reproducible pair of RNG streams plus the provenance needed to rebuild them."""

    root_seed: int
    label: str
    torch_seed: int
    numpy_seed_entropy: Tuple[int, ...]

    def torch_generator(self, device: Optional[str] = None) -> torch.Generator:
        gen = torch.Generator(device=device) if device else torch.Generator()
        gen.manual_seed(self.torch_seed)
        return gen

    def numpy_generator(self) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(list(self.numpy_seed_entropy)))

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "label": self.label,
            "torch_seed": self.torch_seed,
            "numpy_seed_entropy": list(self.numpy_seed_entropy),
            "derivation": "SeedSequence([root_seed, crc32(label)]) -> spawn",
        }


def derive(label: str, root_seed: int = DEFAULT_ROOT_SEED) -> SeedBundle:
    """Derive an independent, reproducible stream pair for a named component.

    The same ``(label, root_seed)`` always yields the same streams; different labels yield
    independent ones. Both properties are asserted in ``test_benchmarks.py`` - the second
    matters because a test that only checks "same seed gives same answer" passes trivially
    for a generator that ignores its seed entirely.
    """
    entropy = (int(root_seed), stable_label_hash(label))
    seq = np.random.SeedSequence(list(entropy))
    child_torch, child_numpy = seq.spawn(2)
    torch_seed = int(child_torch.generate_state(2, dtype=np.uint32).astype(np.uint64)
                     @ np.array([1, 2 ** 32], dtype=np.uint64))
    # torch.Generator.manual_seed accepts a signed 64-bit value.
    torch_seed = int(torch_seed % (2 ** 63 - 1))
    return SeedBundle(
        root_seed=int(root_seed),
        label=label,
        torch_seed=torch_seed,
        numpy_seed_entropy=tuple(int(v) for v in child_numpy.entropy) + tuple(
            int(v) for v in child_numpy.spawn_key),
    )


def randn(shape, bundle: SeedBundle, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Standard normal draw from a bundle's torch stream."""
    return torch.randn(*shape, generator=bundle.torch_generator(), dtype=dtype)


def rand(shape, bundle: SeedBundle, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Uniform [0, 1) draw from a bundle's torch stream."""
    return torch.rand(*shape, generator=bundle.torch_generator(), dtype=dtype)
