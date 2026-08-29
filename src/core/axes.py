"""Axis roles are declared; coordinate names are labels (TG1.1, standard E14).

**The assumption being localised.** Until this module, the platform decided what an axis
*meant* by reading what it was *called*. `importers._spatial_dims` held two tuples of names -
``latitude, lat, y, nlat`` and ``longitude, lon, x, nlon`` - and where those failed it took the
last two dimensions and hoped. Both halves are atmospheric conventions wearing the clothes of
general code: the name list is CF convention, and the trailing-axes rule is true of gridded
model output and of very little else.

Neither half is wrong for ERA5, and neither is removed here. What changes is that a guess is
now **recorded as a guess**, and a caller who knows the answer can say so and be believed.

**Three bases, and the whole point is telling them apart.**

*   ``declared`` - the caller supplied the role. Authoritative; never overridden by a name.
*   ``name`` - matched a registered naming convention. A strong hint, still a hint.
*   ``position`` - the trailing-axes fallback. Correct for gridded output, unjustified
    elsewhere, and the one an onboarding adapter should usually refuse.

`AxisResolution.require_declared` exists so a domain-general path can refuse ``name`` and
``position`` outright. That refusal is the deliverable: a sensor archive whose columns happen
to be called ``x`` and ``y`` must not silently acquire a spatial geometry, because a geometry
is what licenses per-metre reporting and radial binning, and this domain has neither.

**What this does not do.** It does not decide whether two space axes form a *metric* - that is
geometry, and geometry is TG1.2's registry. Here an axis has a role and a position within that
role, and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.registry import Registry

#: Axis roles the analysis layer understands. An axis declares one; nothing infers it from a
#: coordinate's name. `category` exists so a non-ordered axis (station, instrument, cohort)
#: can be carried without being mistaken for something a lag can be taken along.
AXIS_ROLES: Tuple[str, ...] = ("time", "space", "level", "member", "category")

#: How a role was arrived at, in decreasing order of trust. Recorded per axis and carried into
#: provenance, so a reader can see which axes the platform was told about and which it worked
#: out for itself.
AXIS_BASES: Tuple[str, ...] = ("declared", "name", "position")

#: Roles for which a positional fallback exists at all. Deliberately only `space`: the
#: trailing-two-axes rule describes gridded output, and there is no comparable convention that
#: would let a time or level axis be identified by where it sits.
POSITIONAL_ROLES: Tuple[str, ...] = ("space",)


class AxisRoleNotDeclaredError(InvalidParameterError):
    """Raised when a caller requires declared roles and got a guess instead (E14)."""

    def __init__(self, resolution: "AxisResolution", context: str) -> None:
        guessed = ", ".join("%s (by %s)" % (dim, resolution.basis[dim])
                            for dim in resolution.guessed_axes())
        super().__init__(
            "axis_roles", guessed or "(nothing resolved)",
            "declared roles for every axis, because %s. These were worked out from the "
            "coordinate names or their position, which is a convention of gridded "
            "atmospheric output rather than a fact about this data. Pass the roles "
            "explicitly - an axis called 'x' is not a spatial axis until somebody says it is, "
            "and treating it as one licenses per-metre reporting the domain cannot support."
            % context,
            axes=list(resolution.axes))
        self.resolution = resolution


@dataclass(frozen=True)
class NameHint:
    """A naming convention: which axis names have historically meant which role.

    `ordinal` is the position within the role - 0 before 1 - which is what makes a
    ``(lon, lat)`` file come back as ``(lat, lon)`` rather than transposed. A transposed field
    still looks like a field, so every orientation and anisotropy statistic computed from it
    would be wrong in a way nothing downstream can detect.
    """

    role: str
    ordinal: int
    names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in AXIS_ROLES:
            raise UnknownNameError("axis role", self.role, AXIS_ROLES)
        if self.ordinal < 0:
            raise InvalidParameterError("NameHint.ordinal", self.ordinal,
                                        "a non-negative position within the role")


#: Naming conventions, as a registry rather than two module-level tuples (standard E1). A new
#: archive's vocabulary is added by registering, including from outside `src/`; nothing here is
#: privileged, and every entry produces basis ``name`` - a hint, never a declaration.
AXIS_NAME_HINTS: Registry[NameHint] = Registry("axis name hint")

AXIS_NAME_HINTS.add(
    "space_y", NameHint("space", 0, ("latitude", "lat", "y", "nlat")),
    description="Row axis of a gridded field, by CF-style convention.")
AXIS_NAME_HINTS.add(
    "space_x", NameHint("space", 1, ("longitude", "lon", "x", "nlon")),
    description="Column axis of a gridded field, by CF-style convention.")
AXIS_NAME_HINTS.add(
    "time", NameHint("time", 0, ("time", "valid_time", "step")),
    description="Clock axis. A lag is taken along this one and no other.")
AXIS_NAME_HINTS.add(
    "level", NameHint("level", 0, ("level", "plev", "pressure_level", "isobaricinhpa",
                                   "height", "depth")),
    description="Vertical axis. Pressure is one instance of it, not its definition.")
AXIS_NAME_HINTS.add(
    "member", NameHint("member", 0, ("member", "number", "ensemble", "realization")),
    description="Ensemble axis. Ordered by label only; the order carries no metric.")


def _hint_table() -> Dict[str, NameHint]:
    """Flatten the registry to ``lowercase axis name -> hint``, refusing collisions.

    Iteration is over the registry's sorted order, so a collision is reported deterministically
    rather than depending on which module imported first.
    """
    table: Dict[str, NameHint] = {}
    owner: Dict[str, str] = {}
    for entry in AXIS_NAME_HINTS:
        for name in entry.value.names:
            key = name.lower()
            if key in table:
                raise InvalidParameterError(
                    "axis name hint", name,
                    "a name claimed by exactly one hint. %r claims it and so does %r, so the "
                    "role of an axis with that name would depend on registration order"
                    % (owner[key], entry.name))
            table[key] = entry.value
            owner[key] = entry.name
    return table


@dataclass(frozen=True)
class AxisResolution:
    """What each axis is taken to mean, and on whose authority.

    Immutable and self-describing, because this record travels into provenance: an imported
    field whose spatial axes were chosen by position should say so in the file that cites it.
    """

    axes: Tuple[str, ...]
    roles: Mapping[str, str]
    ordinals: Mapping[str, int]
    basis: Mapping[str, str]

    # ----------------------------------------------------------------- interrogation

    def dims_with_role(self, role: str) -> Tuple[str, ...]:
        """Axes carrying `role`, ordered by their position within it."""
        if role not in AXIS_ROLES:
            raise UnknownNameError("axis role", role, AXIS_ROLES)
        matched = [dim for dim in self.axes if self.roles.get(dim) == role]
        return tuple(sorted(matched,
                            key=lambda dim: (self.ordinals[dim], self.axes.index(dim))))

    def spatial_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """The two spatial axes in ``(row, column)`` order, or ``None`` where unresolved."""
        space = self.dims_with_role("space")
        if len(space) >= 2:
            return space[0], space[1]
        if len(space) == 1:
            return space[0], None
        return None, None

    def guessed_axes(self) -> Tuple[str, ...]:
        """Axes whose role was not declared. The ones a general adapter should worry about."""
        return tuple(dim for dim in self.axes
                     if dim in self.roles and self.basis[dim] != "declared")

    @property
    def fully_declared(self) -> bool:
        """Every axis has a role and every role was declared. An unresolved axis is not
        "declared by omission": it is an axis nobody has said anything about."""
        return (bool(self.axes) and len(self.roles) == len(self.axes)
                and not self.guessed_axes())

    def require_declared(self, context: str) -> "AxisResolution":
        """Refuse a guess. Returns self, so it can be used inline."""
        if not self.fully_declared:
            raise AxisRoleNotDeclaredError(self, context)
        return self

    def describe(self) -> Dict[str, Any]:
        return {
            "axes": list(self.axes),
            "roles": {dim: self.roles[dim] for dim in self.axes if dim in self.roles},
            "basis": {dim: self.basis[dim] for dim in self.axes if dim in self.basis},
            "unresolved": [dim for dim in self.axes if dim not in self.roles],
            "fully_declared": self.fully_declared,
        }


def _declared_role(dim: str, value: Any) -> Tuple[str, Optional[int]]:
    """Accept a bare role name or anything carrying `.role` (an `AxisSpec`, typically)."""
    role = getattr(value, "role", value)
    ordinal = getattr(value, "ordinal", None)
    if not isinstance(role, str) or role not in AXIS_ROLES:
        raise UnknownNameError("axis role", role, AXIS_ROLES, axis=dim)
    if ordinal is not None and (not isinstance(ordinal, int) or ordinal < 0):
        raise InvalidParameterError("axis ordinal for %r" % dim, ordinal,
                                    "a non-negative position within the role")
    return role, ordinal


def resolve_axis_roles(
    dims: Sequence[Any],
    declared: Optional[Mapping[str, Any]] = None,
    *,
    allow_name_inference: bool = True,
    allow_positional_inference: bool = True,
) -> AxisResolution:
    """Assign a role to each axis, recording how each assignment was reached.

    The order of authority is fixed: declaration, then registered naming convention, then the
    trailing-axes fallback. A later stage never overwrites an earlier one, which is what makes
    "declare it and be believed" a property rather than a convention.

    Both inference stages can be switched off. ``allow_name_inference=False`` is the setting a
    domain-general adapter wants: it leaves an axis unresolved rather than plausible, and an
    unresolved axis fails loudly at the first operator that needs the role.
    """
    axes = tuple(str(dim) for dim in dims)
    if len(set(axes)) != len(axes):
        raise InvalidParameterError(
            "dims", list(axes),
            "distinct axis names. Two axes sharing a name cannot be given separate roles, and "
            "resolving either would silently pick one of them")

    roles: Dict[str, str] = {}
    ordinals: Dict[str, int] = {}
    basis: Dict[str, str] = {}

    def assign(dim: str, role: str, ordinal: int, how: str) -> None:
        roles[dim] = role
        ordinals[dim] = ordinal
        basis[dim] = how

    def unassign(dim: str) -> None:
        roles.pop(dim, None)
        ordinals.pop(dim, None)
        basis.pop(dim, None)

    # --- 1. declaration -----------------------------------------------------------------
    declared = dict(declared or {})
    unknown = [name for name in declared if name not in axes]
    if unknown:
        raise InvalidParameterError(
            "declared", unknown,
            "roles only for axes that are present, which are %s. A declaration naming an axis "
            "the data does not have describes different data, and that mismatch is far more "
            "likely to be why a result looks wrong than anything downstream of it"
            % (list(axes),))

    stated = {dim: _declared_role(dim, declared[dim]) for dim in axes if dim in declared}
    for role in AXIS_ROLES:
        same_role = [dim for dim in axes if dim in stated and stated[dim][0] == role]
        explicit = {dim: stated[dim][1] for dim in same_role if stated[dim][1] is not None}
        if len(set(explicit.values())) != len(explicit):
            raise InvalidParameterError(
                "declared", sorted(explicit),
                "distinct ordinals within role %r. Two axes at the same position in one role "
                "leaves their order decided by chance, and for a spatial pair that is the "
                "difference between a field and its transpose" % role)
        for index, dim in enumerate(same_role):
            assign(dim, role, explicit.get(dim, index), "declared")

    # --- 2. registered naming conventions -----------------------------------------------
    if allow_name_inference:
        table = _hint_table()
        for dim in axes:
            if dim in roles:
                continue
            hint = table.get(dim.lower())
            if hint is None:
                continue
            # First axis in dimension order wins a given (role, ordinal); a second candidate
            # is left unresolved rather than displacing it. This reproduces the behaviour of
            # the name lists it replaces, which also took the first match.
            taken = any(roles.get(other) == hint.role
                        and ordinals.get(other) == hint.ordinal for other in axes)
            if taken:
                continue
            assign(dim, hint.role, hint.ordinal, "name")

    # --- 3. the trailing-axes fallback, for space only ----------------------------------
    if allow_positional_inference and "space" in POSITIONAL_ROLES:
        space = [dim for dim in axes if roles.get(dim) == "space"]
        if not any(basis.get(dim) == "declared" for dim in space) and len(space) < 2 \
                and len(axes) >= 2:
            # The name pass found at most one spatial axis, which is not a usable pair. It is
            # discarded rather than half-trusted: a file with dims (lat, time, cols) is not a
            # file whose columns are longitude, and pairing them would invent a grid.
            for dim in space:
                unassign(dim)
            trailing = axes[-2:]
            if all(dim not in roles for dim in trailing):
                for ordinal, dim in enumerate(trailing):
                    assign(dim, "space", ordinal, "position")

    return AxisResolution(axes=axes, roles=dict(roles), ordinals=dict(ordinals),
                          basis=dict(basis))


def find_coordinate(available: Iterable[Any], role: str, ordinal: int = 0) -> Optional[str]:
    """First registered name for ``role``/``ordinal`` that is present in `available`.

    For adapters that already know the data is gridded and only need to know whether this
    particular archive spells it ``lat`` or ``latitude``. That is a spelling question, not a
    role question - which is why it is a separate, smaller function, and why it does not
    produce an `AxisResolution` that could be mistaken for one.
    """
    if role not in AXIS_ROLES:
        raise UnknownNameError("axis role", role, AXIS_ROLES)
    present = {str(name) for name in available}
    for entry in AXIS_NAME_HINTS:
        hint = entry.value
        if hint.role != role or hint.ordinal != ordinal:
            continue
        for name in hint.names:
            if name in present:
                return name
    return None
