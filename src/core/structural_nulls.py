"""Domain-legitimate, mode-specific nulls over the canonical record (TG17.3, TG17.5, `ed-dev`).

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

**A null belongs to a comparison mode, not to a study.** Calendar mode asks whether two records
carry structure in the same UTC interval, so its null must break alignment and keep everything
else. Scale/shape mode asks whether a frozen shape recurs across native scales, where there is
no shared clock to break: shifting a record's clock there answers nothing, because the question
never referred to the clock. `NULL_FAMILIES` therefore carries the mode each family answers for,
and `assert_null_admits_mode` refuses a manifest that names the other mode's null — at the
manifest, where the question is declared, rather than in the wording of a result.

**What each family preserves is written down, not assumed.** A null is legitimate exactly to the
extent that it preserves the features which would otherwise manufacture the structure being
tested: autocorrelation, seasonal or cyclic phase, irregular gaps, profile support, the
observation window and any declared grouping. Each registered family declares its own
`preserves` and `destroys`, both are carried into the receipt, and `test_experiment_family.py`
asserts the declaration against the surrogate the family actually produces rather than trusting
the description.

**`global_value_shuffle` is registered and refused.** Every domain can execute it, which is the
whole reason it needs to be refusable *by name*: an operation that every adapter can perform
looks like the natural common denominator right up to the point where it destroys the
autocorrelation and the gap structure that made the comparison worth testing. Leaving it out of
the registry would make it a null nobody had thought about; leaving it in, marked inadmissible
with its reason, makes it one the framework has already answered.

**The surrogate is labelled as one.** A null trajectory carries a ``null:`` trajectory id, its
lineage names the shift and its parameters, and its channel digests are recomputed. It would
otherwise be a record that reconstructs to different values than its lineage claims — a forged
provenance chain, and exactly the failure the TG17.2 conformance pass exists to catch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.core.registry import Registry
from src.core.structural_trajectory import ChannelLineage, StructuralTrajectory


#: The features a null may preserve. Named rather than free text, because "preserves the
#: seasonality" and "preserves seasonal phase" would be two spellings of one guarantee and a
#: reader comparing two families could not tell whether they meant the same thing.
NULL_FEATURES: Tuple[str, ...] = (
    "autocorrelation",
    "seasonal_phase",
    "irregular_gaps",
    "profile_support",
    "observation_window",
    "declared_grouping",
    "marginal_distribution",
    "native_scale",
    "cross_record_alignment",
)

#: The comparison modes a null can answer for. Mirrors `structural_alignment.MODE_RELATIONSHIPS`
#: deliberately: a null is the mechanical counterpart of the question a mode is allowed to ask.
NULL_MODES: Tuple[str, ...] = ("calendar_aligned", "scale_shape_aligned")


class NullRefusal(InvalidParameterError):
    """A declared null this framework will not run, named with what it would have destroyed."""

    def __init__(self, requirement: str, subject: Any, detail: str) -> None:
        super().__init__(requirement, subject, detail)


# Above this size the valid reassignments are no longer enumerated exhaustively, and a
# uniform draw would have to be argued for rather than demonstrated. No declared family
# approaches it: the largest currently declared inventory names six pairings.
MAX_REASSIGNABLE_PAIRINGS = 8


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------------------- families


@dataclass(frozen=True)
class NullFamily:
    """One declared surrogate construction, with the mode it answers for.

    `parameters` are required and have no defaults, for the reason TG17.4 gave for alignment
    kernels: a cycle length the framework picked is a scientific choice nobody made, and it
    would sit inside every p-value the study reports.
    """

    name: str
    modes: Tuple[str, ...]
    operates_on: str
    preserves: Tuple[str, ...]
    destroys: Tuple[str, ...]
    apply: Optional[Callable[..., Any]] = None
    parameters: Tuple[str, ...] = ()
    admissible: bool = True
    inadmissible_reason: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        for mode in self.modes:
            if mode not in NULL_MODES:
                raise InvalidParameterError("null %r mode" % self.name, mode,
                                            "one of %s" % list(NULL_MODES))
        for feature in tuple(self.preserves) + tuple(self.destroys):
            if feature not in NULL_FEATURES:
                raise InvalidParameterError(
                    "null %r feature" % self.name, feature,
                    "one of the named features %s. A guarantee spelled freehand cannot be "
                    "compared against another family's" % list(NULL_FEATURES))
        both = set(self.preserves) & set(self.destroys)
        if both:
            raise InvalidParameterError(
                "null %r" % self.name, sorted(both),
                "features that are either preserved or destroyed, not both")
        if self.admissible and self.apply is None:
            raise InvalidParameterError(
                "null %r" % self.name, None,
                "an implementation. An admissible family with nothing behind it is a null a "
                "manifest can declare and no pass can run")
        if not self.admissible and not self.inadmissible_reason.strip():
            raise InvalidParameterError(
                "null %r" % self.name, self.inadmissible_reason,
                "the reason this family is refused. A null refused without a stated reason "
                "reads as an omission, and the next author adds it back")

    def resolve(self, parameters: Mapping[str, Any]) -> Dict[str, float]:
        """The declared parameters, refusing both the unknown and the missing."""
        supplied = dict(parameters or {})
        unknown = sorted(set(supplied) - set(self.parameters))
        if unknown:
            raise NullRefusal(
                "null %r parameters" % self.name, unknown,
                "only the parameters this family declares (%s). An ignored parameter is a "
                "setting a researcher believes is in force and is not"
                % (list(self.parameters) or "none"))
        missing = [name for name in self.parameters if name not in supplied]
        if missing:
            raise NullRefusal(
                "null %r parameters" % self.name, missing,
                "a declared value for every parameter. A cycle length or block length with a "
                "framework default would be a scientific choice nobody made, sitting inside "
                "every p-value the study reports")
        return {name: float(supplied[name]) for name in self.parameters}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modes": list(self.modes),
            "operates_on": self.operates_on,
            "preserves": list(self.preserves),
            "destroys": list(self.destroys),
            "parameters": list(self.parameters),
            "admissible": self.admissible,
            "inadmissible_reason": self.inadmissible_reason or None,
            "definition": self.description,
            "claim_boundary": ("A null describes what a surrogate keeps and what it breaks. "
                               "Calibrating against it is not evidence that the null is the "
                               "right one for a question."),
        }


#: Registered rather than branched (standard E1). A new domain that needs a null its support
#: can carry adds a family here, where its mode and its preserved features are declared once,
#: instead of a `build_null` inside an adapter where nobody else can see what it broke.
NULL_FAMILIES: Registry[NullFamily] = Registry("structural null family")


def register_null_family(family: NullFamily) -> NullFamily:
    NULL_FAMILIES.add(family.name, family, description=family.description,
                      capabilities={"modes": list(family.modes),
                                    "admissible": family.admissible,
                                    "operates_on": family.operates_on})
    return family


# ------------------------------------------------------------------------ implementations


def _shift_channels(trajectory: StructuralTrajectory, offsets: np.ndarray,
                    operation: str, parameters: Mapping[str, float]) -> StructuralTrajectory:
    """Rewrite every channel by a declared row permutation, keeping support where it was.

    The support arrays and the validity mask are untouched by construction rather than by
    intention: this function never receives them. A null that moved support would be answering
    a different question — whether the record could have been observed elsewhere — and the
    gaps are the thing a calendar comparison must not be allowed to invent.
    """
    channels: Dict[str, np.ndarray] = {}
    lineage: Dict[str, ChannelLineage] = {}
    for name, values in trajectory.channels.items():
        moved = np.asarray(values)[offsets]
        moved = np.array(moved, copy=True)
        moved.setflags(write=False)
        channels[name] = moved
        lineage[name] = ChannelLineage(
            channel=name, source_variable=trajectory.variable,
            source_indices=offsets,
            operation=operation,
            parameters=dict(parameters),
            output_sha256=_array_digest(moved))
    tag = "-".join("%s=%g" % item for item in sorted(parameters.items()))
    return replace(trajectory,
                   trajectory_id="null:%s:%s" % (trajectory.trajectory_id, tag),
                   channels=channels, lineage=lineage)


def circular_clock_shift(trajectory: StructuralTrajectory, seed: int) -> StructuralTrajectory:
    """One domain-preserving surrogate: same clock, same gaps, broken alignment."""
    length = len(trajectory.support_start_seconds)
    if length < 2:
        raise ValueError("a clock-shift null needs at least two supports to shift between")
    # A zero shift would return the record itself and be indistinguishable from no null at all,
    # so the offset is drawn from [1, length - 1] rather than [0, length).
    offset = int(np.random.default_rng(seed).integers(1, length))
    indices = (np.arange(length) - offset) % length
    return _shift_channels(trajectory, indices,
                           "circular_clock_shift(canonical_channel, offset)",
                           {"offset": float(offset), "seed": float(seed)})


def whole_cycle_clock_shift(trajectory: StructuralTrajectory, seed: int, *,
                            cycle_seconds: float) -> StructuralTrajectory:
    """Shift by a whole number of declared cycles, so seasonal phase survives the null.

    The plain clock shift breaks cross-record alignment and the annual or diurnal phase along
    with it. For a record whose structure is partly seasonal that overstates the null: the
    surrogate is now unlike the original in a way the alternative hypothesis never claimed, and
    the test is calibrated against a record the domain does not produce. Shifting by whole
    cycles keeps every observation at the same point of its cycle and moves only which cycle it
    is in — so a seasonal record stays seasonal and only its alignment with another record is
    destroyed.
    """
    length = len(trajectory.support_start_seconds)
    if length < 2:
        raise ValueError("a clock-shift null needs at least two supports to shift between")
    if not np.isfinite(cycle_seconds) or cycle_seconds <= 0.0:
        raise InvalidParameterError("cycle_seconds", cycle_seconds,
                                    "a positive declared cycle length in seconds")
    spans = np.diff(np.asarray(trajectory.support_start_seconds, dtype=np.float64))
    step = float(np.median(spans[spans > 0.0])) if np.any(spans > 0.0) else 0.0
    rows_per_cycle = int(round(cycle_seconds / step)) if step > 0.0 else 0
    if rows_per_cycle < 1 or rows_per_cycle >= length:
        raise NullRefusal(
            "cycle_seconds", cycle_seconds,
            "a cycle this record spans more than once. %d rows at a %.6g s median step carry "
            "%.3g cycles, and a shift by a whole cycle would be a shift by the whole record "
            "or by nothing" % (length, step, (length * step) / cycle_seconds
                               if cycle_seconds else float("inf")))
    cycles = length // rows_per_cycle
    draw = int(np.random.default_rng(seed).integers(1, cycles)) if cycles > 1 else 1
    offset = draw * rows_per_cycle
    indices = (np.arange(length) - offset) % length
    return _shift_channels(
        trajectory, indices,
        "whole_cycle_clock_shift(canonical_channel, whole_cycles)",
        {"offset": float(offset), "seed": float(seed), "cycle_seconds": float(cycle_seconds),
         "rows_per_cycle": float(rows_per_cycle)})


def within_group_clock_shift(trajectory: StructuralTrajectory, seed: int, *,
                             group_seconds: float) -> StructuralTrajectory:
    """Shift inside each declared group, so a grouped record stays grouped.

    A record whose rows belong to declared groups — one Argo float's profiles, one TESS sector,
    one trading session — carries structure at the group boundary that a whole-record shift
    moves values across. The surrogate then differs from the original in group composition as
    well as alignment, and a test calibrated against it is testing the grouping.
    """
    starts = np.asarray(trajectory.support_start_seconds, dtype=np.float64)
    length = len(starts)
    if length < 2:
        raise ValueError("a clock-shift null needs at least two supports to shift between")
    if not np.isfinite(group_seconds) or group_seconds <= 0.0:
        raise InvalidParameterError("group_seconds", group_seconds,
                                    "a positive declared group length in seconds")
    groups = np.floor((starts - starts[0]) / group_seconds).astype(np.int64)
    if len(np.unique(groups)) < 2:
        raise NullRefusal(
            "group_seconds", group_seconds,
            "a group length that divides this record into more than one group. One group is "
            "the whole record, and a within-group shift over it is the unrestricted shift "
            "under another name")
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    for group in np.unique(groups):
        members = indices[groups == group]
        if len(members) < 2:
            continue
        offset = int(rng.integers(1, len(members)))
        indices[members] = members[(np.arange(len(members)) - offset) % len(members)]
    return _shift_channels(
        trajectory, indices,
        "within_group_clock_shift(canonical_channel, group)",
        {"seed": float(seed), "group_seconds": float(group_seconds),
         "groups": float(len(np.unique(groups)))})


def _is_same_member(left: Any, right: Any) -> bool:
    """Whether two inventory members are the same record, not merely similar ones."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:  # pragma: no cover - members that decline equality are distinct by identity
        return False


def admissible_partners(pairs: Sequence[Tuple[Any, Any]]) -> Tuple[Tuple[int, ...], ...]:
    """For each pairing, which of the inventory's right members could legitimately replace its own.

    A candidate is admissible only where it is neither the partner the pairing already had — whose
    surrogate would be the observation — nor the pairing's own left member, whose similarity to
    itself is maximal by construction. Exposed rather than kept private because it is also the
    reference set an exact partner test draws its p-value denominator from, and the two must be
    the same set or the test would be priced against a null it is not running.
    """
    lefts = [left for left, _ in pairs]
    rights = [right for _, right in pairs]
    member_of: list = []
    for position, right in enumerate(rights):
        same = [member_of[earlier] for earlier in range(position)
                if _is_same_member(right, rights[earlier])]
        member_of.append(same[0] if same else max(member_of, default=-1) + 1)
    return tuple(
        tuple(candidate for candidate in range(len(pairs))
              if member_of[candidate] != member_of[position]
              and not _is_same_member(rights[candidate], lefts[position]))
        for position in range(len(pairs)))


def _valid_reassignments(pairs: Sequence[Tuple[Any, Any]]) -> Tuple[Tuple[int, ...], ...]:
    """Every *distinguishable* reassignment of the inventory's right members.

    An assignment is valid only where, for every pairing, the substituted partner is neither the
    partner it already had nor the pairing's own left member. The first would return the
    observation as its own surrogate; the second would compare a record against itself, whose
    similarity is maximal by construction. Both would be counted as evidence that the null could
    not be rejected.

    Assignments are then deduplicated by the surrogate they *produce*, not by the index
    permutation that produced it. Where an inventory names the same record in several pairings,
    many permutations yield one surrogate; counting them separately would both overstate how much
    the null explores and bias a uniform draw towards whichever surrogate has the most spellings.
    """
    lefts = [left for left, _ in pairs]
    rights = [right for _, right in pairs]
    member_of: list = []
    for position, right in enumerate(rights):
        same = [member_of[earlier] for earlier in range(position)
                if _is_same_member(right, rights[earlier])]
        member_of.append(same[0] if same else max(member_of, default=-1) + 1)
    allowed = admissible_partners(pairs)
    if any(not candidates for candidates in allowed):
        return ()
    distinct: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    order: list = []
    taken = set()

    def extend(position: int) -> None:
        if position == len(pairs):
            distinct.setdefault(tuple(member_of[index] for index in order), tuple(order))
            return
        for candidate in allowed[position]:
            if candidate in taken:
                continue
            taken.add(candidate)
            order.append(candidate)
            extend(position + 1)
            order.pop()
            taken.discard(candidate)

    extend(0)
    return tuple(distinct.values())


def reassign_scale_partners(pairings: Sequence[Tuple[Any, Any]], seed: int)         -> Tuple[Tuple[Any, Any], ...]:
    """Break which native scales were compared, and change nothing about either record.

    Scale/shape mode has no shared clock to shift, so its null cannot be a clock shift. What the
    mode asserts is that *this* shape at *this* native duration resembles *that* shape at *that*
    one; the corresponding null is the same inventory of shapes paired differently. Every
    record keeps its values, its support, its gaps and its native scale — the surrogate differs
    from the observation only in the correspondence being tested, which is the only thing the
    claim was about.

    **The reassignment is of pairings, not of positions (D91).** Deranging the positions of the
    right members guarantees only that each moved somewhere else in the list. Where an inventory
    names the same record as the right member of more than one pairing — which every all-pairs
    inventory does — a member can move position while the pairing it produces is identical to the
    one it replaced. That surrogate equals its observation, satisfies the ``>=`` of a one-sided
    surrogate p-value, and so raises that member's p-value floor towards 1 no matter how strong
    the effect. The bias is conservative, which is why it has to be excluded here rather than
    noticed downstream: it presents as a well-behaved safeguard result rather than as a fault.

    **An inventory that admits no valid reassignment is refused, not approximated.** Three
    records compared all-against-all is such an inventory: one of its three pairings has no
    substitute partner that is neither its own nor itself. That is a real property of the
    declared family and not an error to route around — returning the observation, or relaxing
    the constraint to get an answer, would be manufacturing a null the inventory cannot support.
    """
    pairs = [(left, right) for left, right in pairings]
    if len(pairs) < 2:
        raise NullRefusal(
            "pairings", len(pairs),
            "at least two declared correspondences to reassign between. One pairing has no "
            "alternative partner, so its null is the observation itself")
    if len(pairs) > MAX_REASSIGNABLE_PAIRINGS:
        raise NullRefusal(
            "pairings", len(pairs),
            f"at most {MAX_REASSIGNABLE_PAIRINGS} declared correspondences, the size at which "
            "the valid reassignments can still be enumerated exactly and drawn from uniformly. "
            "A larger inventory needs a sampler whose uniformity has been demonstrated, not an "
            "approximation adopted silently")
    assignments = _valid_reassignments(pairs)
    if not assignments:
        raise NullRefusal(
            "pairings", len(pairs),
            "an inventory admitting a reassignment in which every pairing changes and no record "
            "is paired with itself. This one admits none, so it has no surrogate distinguishable "
            "from its observation and cannot be tested under this null")
    if len(assignments) < 2:
        raise NullRefusal(
            "pairings", len(pairs),
            "an inventory admitting more than one distinguishable reassignment. This one admits "
            "exactly one, so every replication returns the same surrogate: a null that is a "
            "constant rather than a distribution, whose p-value can take only two values "
            "regardless of how many surrogates are paid for")
    rng = np.random.default_rng(seed)
    order = assignments[int(rng.integers(len(assignments)))]
    rights = [right for _, right in pairs]
    return tuple((pairs[index][0], rights[order[index]]) for index in range(len(pairs)))


def global_value_shuffle(*args: Any, **kwargs: Any) -> Any:
    """Registered so it can be refused by name; never called."""
    raise NullRefusal(
        "null", "global_value_shuffle",
        "a null that preserves what the comparison is testing. This one is refused: see the "
        "registered family's stated reason")


# ------------------------------------------------------------------------- registrations


register_null_family(NullFamily(
    name="independent_native_clock_shift",
    modes=("calendar_aligned",),
    operates_on="trajectory",
    preserves=("autocorrelation", "irregular_gaps", "profile_support", "observation_window",
               "marginal_distribution", "native_scale"),
    destroys=("cross_record_alignment", "seasonal_phase", "declared_grouping"),
    apply=circular_clock_shift,
    description=("Roll the canonical values by a seeded offset, preserving the native clock, "
                 "interval support, gaps and marginal distribution, and destroying only "
                 "cross-record alignment."),
))

register_null_family(NullFamily(
    name="whole_cycle_clock_shift",
    modes=("calendar_aligned",),
    operates_on="trajectory",
    preserves=("autocorrelation", "seasonal_phase", "irregular_gaps", "profile_support",
               "observation_window", "marginal_distribution", "native_scale"),
    destroys=("cross_record_alignment", "declared_grouping"),
    apply=whole_cycle_clock_shift,
    parameters=("cycle_seconds",),
    description=("Roll by a whole number of declared cycles, so every observation keeps its "
                 "seasonal or diurnal phase and only which cycle it fell in is destroyed."),
))

register_null_family(NullFamily(
    name="within_group_clock_shift",
    modes=("calendar_aligned",),
    operates_on="trajectory",
    preserves=("autocorrelation", "irregular_gaps", "profile_support", "observation_window",
               "marginal_distribution", "native_scale", "declared_grouping"),
    destroys=("cross_record_alignment", "seasonal_phase"),
    apply=within_group_clock_shift,
    parameters=("group_seconds",),
    description=("Roll inside each declared group, so group composition survives the null and "
                 "only alignment across records is destroyed."),
))

register_null_family(NullFamily(
    name="scale_partner_reassignment",
    modes=("scale_shape_aligned",),
    operates_on="correspondence",
    preserves=("autocorrelation", "seasonal_phase", "irregular_gaps", "profile_support",
               "observation_window", "marginal_distribution", "native_scale",
               "declared_grouping"),
    destroys=("cross_record_alignment",),
    apply=reassign_scale_partners,
    description=("Pair the same inventory of frozen shapes with different native scales. No "
                 "record is altered; only the correspondence under test is broken."),
))

register_null_family(NullFamily(
    name="global_value_shuffle",
    modes=NULL_MODES,
    operates_on="trajectory",
    preserves=("marginal_distribution", "observation_window"),
    destroys=("autocorrelation", "seasonal_phase", "irregular_gaps", "profile_support",
              "declared_grouping", "cross_record_alignment"),
    apply=global_value_shuffle,
    admissible=False,
    inadmissible_reason=(
        "Every domain can execute a global shuffle, and that is the only argument for it. It "
        "destroys the autocorrelation, the cyclic phase, the gap structure and the profile "
        "support that would otherwise produce apparent cross-domain structure, so a "
        "co-occurrence tested against it is compared to a record no domain here emits. The "
        "surrogate p-values are correspondingly optimistic and there is no way to tell from "
        "the result that they are."),
    description="A global permutation of values. Registered so that it can be refused by name.",
))


# ------------------------------------------------------------------------------- binding


@dataclass(frozen=True)
class BoundNull:
    """A null family, frozen with the parameters and mode the manifest declared."""

    name: str
    mode: str
    parameters: Mapping[str, float] = dc_field(default_factory=dict)

    @property
    def family(self) -> NullFamily:
        return NULL_FAMILIES.get(self.name)

    def apply(self, subject: Any, seed: int) -> Any:
        family = self.family
        if family.apply is None:  # pragma: no cover - guarded at registration
            raise NullRefusal("null", self.name, "an implemented family")
        return family.apply(subject, seed, **dict(self.parameters))

    def describe(self) -> Dict[str, Any]:
        return {**self.family.describe(), "mode": self.mode,
                "resolved_parameters": dict(self.parameters)}


def assert_null_admits_mode(name: str, mode: str) -> NullFamily:
    """Refuse the other mode's null where the question is declared, not where it is worded."""
    if mode not in NULL_MODES:
        raise InvalidParameterError("mode", mode, "one of %s" % list(NULL_MODES))
    family = NULL_FAMILIES.get(name)
    if mode not in family.modes:
        raise NullRefusal(
            "null %r" % name, mode,
            "a null declared for this comparison mode. %r answers for %s. A clock shift is no "
            "null for a question that never referred to a clock, and a partner reassignment is "
            "no null for a question about a shared calendar interval"
            % (name, list(family.modes)))
    return family


def bind_null(name: str, parameters: Mapping[str, Any], *, mode: str,
              admissible_by_adapter: Optional[Mapping[str, Sequence[str]]] = None) -> BoundNull:
    """Freeze one declared null, or refuse it by name.

    The three refusals are separate on purpose. A family can be inadmissible in itself
    (`global_value_shuffle`), admissible but belonging to the other mode, or fine everywhere
    except in a domain whose support cannot carry it — and a study that hits the third needs to
    be told which domain, because that is the one whose declaration it must argue with.
    """
    if name not in NULL_FAMILIES:
        raise NullRefusal("null", name,
                          "a registered null family; registered: %s" % NULL_FAMILIES.names())
    family = NULL_FAMILIES.get(name)
    if not family.admissible:
        raise NullRefusal("null", name, family.inadmissible_reason)
    assert_null_admits_mode(name, mode)
    resolved = family.resolve(parameters)
    for adapter_id in sorted(admissible_by_adapter or {}):
        admitted = tuple(admissible_by_adapter[adapter_id])
        if name not in admitted:
            raise NullRefusal(
                "null %r" % name, adapter_id,
                "a null every participating adapter admits. %s admits %s. A surrogate its "
                "support cannot carry is a calibration of the surrogate rather than of the "
                "record" % (adapter_id, list(admitted)))
    return BoundNull(name=name, mode=mode, parameters=resolved)


def describe_null(name: str) -> Mapping[str, Any]:
    return NULL_FAMILIES.get(name).describe()


def nulls_for_mode(mode: str) -> Tuple[str, ...]:
    """Every admissible family this mode can be calibrated with."""
    if mode not in NULL_MODES:
        raise InvalidParameterError("mode", mode, "one of %s" % list(NULL_MODES))
    return tuple(entry.name for entry in NULL_FAMILIES.entries()
                 if entry.value.admissible and mode in entry.value.modes)


__all__ = ["NULL_FAMILIES", "NULL_FEATURES", "NULL_MODES", "NullFamily", "NullRefusal",
           "BoundNull", "register_null_family", "circular_clock_shift",
           "whole_cycle_clock_shift", "within_group_clock_shift", "reassign_scale_partners",
           "admissible_partners",
           "assert_null_admits_mode", "bind_null", "describe_null", "nulls_for_mode"]
