"""Domain-legitimate nulls over the canonical record (TG17.3, `ed-dev`).

**Why these live in the framework.** Every adapter needs a null, and a null is precisely the
place where an adapter author's convenience quietly becomes a scientific error. Shuffling the
values of an irregular record destroys the irregularity that made it a second domain; resampling
a gapped record across its gaps manufactures the coverage the gaps deny; permuting rows treats a
clock as an index. Four adapters writing four nulls would produce four different mistakes, and
each would be invisible inside the adapter that made it.

So the null is supplied here and *named in the adapter's declaration* rather than implemented
there. `legitimate_null_family` on a `StructuralAdapterDeclaration` says which of these an
adapter's domain admits; the conformance kit then checks the null preserves the support and
validity it claims to.

**What the clock shift does and does not preserve.** `circular_clock_shift` rolls the canonical
channel values by a seeded offset and leaves the `[start, end)` support and the validity mask
exactly where they were. That is the TG17.0 ``independent_native_clock_shift`` with
``preserve_gaps=True``: the record keeps its own cadence, its own gaps and its own marginal
distribution, and loses only its alignment with any other record. It therefore tests the
question a cross-domain experiment actually asks — *is this co-occurrence more than these two
records' own structure could produce?* — rather than the weaker question a value shuffle asks.

It does **not** preserve the pairing between a value and the gap that surrounded it. A record
whose values depend on their distance from a gap is not well served by this null, which is why
the family is declared per adapter rather than assumed.

**The surrogate is labelled as one.** A null trajectory carries a ``null:`` trajectory id, its
lineage names the shift and its parameters, and its channel digests are recomputed. It would
otherwise be a record that reconstructs to different values than its lineage claims — a forged
provenance chain, and exactly the failure the TG17.2 conformance pass exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, Mapping

import numpy as np

from src.core.structural_trajectory import ChannelLineage, StructuralTrajectory


NULL_FAMILIES = {
    "independent_native_clock_shift": (
        "roll the canonical values by a seeded offset, preserving the native clock, interval "
        "support, gaps and marginal distribution, and destroying only cross-record alignment"),
}


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def circular_clock_shift(trajectory: StructuralTrajectory, seed: int) -> StructuralTrajectory:
    """One domain-preserving surrogate: same clock, same gaps, broken alignment."""
    length = len(trajectory.support_start_seconds)
    if length < 2:
        raise ValueError("a clock-shift null needs at least two supports to shift between")
    # A zero shift would return the record itself and be indistinguishable from no null at all,
    # so the offset is drawn from [1, length - 1] rather than [0, length).
    offset = int(np.random.default_rng(seed).integers(1, length))
    channels: Dict[str, np.ndarray] = {}
    lineage: Dict[str, ChannelLineage] = {}
    for name, values in trajectory.channels.items():
        rolled = np.roll(np.asarray(values), offset)
        rolled.setflags(write=False)
        channels[name] = rolled
        lineage[name] = ChannelLineage(
            channel=name, source_variable=trajectory.variable,
            source_indices=(np.arange(length) - offset) % length,
            operation="circular_clock_shift(canonical_channel, offset)",
            parameters={"offset": float(offset), "seed": float(seed)},
            output_sha256=_array_digest(rolled))
    return replace(trajectory,
                   trajectory_id="null:%s:%d" % (trajectory.trajectory_id, offset),
                   channels=channels, lineage=lineage)


def describe_null(name: str) -> Mapping[str, Any]:
    if name not in NULL_FAMILIES:
        raise ValueError("unregistered null family %r; registered: %s"
                         % (name, sorted(NULL_FAMILIES)))
    return {"name": name, "definition": NULL_FAMILIES[name],
            "preserves": ["native clock", "interval support", "validity mask",
                          "marginal distribution"],
            "destroys": ["alignment with any other record"]}


__all__ = ["NULL_FAMILIES", "circular_clock_shift", "describe_null"]
