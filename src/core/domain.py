"""Domain declarations: what a data source is, and what it breaks (TG0.2, standards E14/E15).

**The problem this solves.** The analysis layer inherited its safety from the atmosphere. It
knows that a lag shorter than a filter's advective crossing time measures geometry rather than
weather (rule R4), because someone wrote that rule while looking at ERA5. Hand it a domain with
no advection and the rule silently does not apply — and *silently* is the failure. A refusal is
a scientific result; a number produced under an assumption nobody checked is not.

So a domain does not merely supply data here. It supplies:

*   **declared axis roles** (E14) — what `time`, `space`, `level`, `member` and `category` mean
    for this source, so no code has to infer meaning from an axis being called ``lat``;
*   **declared violations** (E15) — which of the analysis layer's inherited assumptions this
    domain breaks, drawn from a fixed vocabulary rather than free text, so the refusals can be
    automatic rather than remembered; and
*   **a declared lag policy** (R21) — the justification for calling any lag admissible at all.

Rule R17 makes the first of those the entry price: *a domain that violates nothing is not a
second domain*, it is a second variable, and it is no evidence that the abstraction generalises.
`DomainDeclaration` therefore refuses an empty violation set for a domain that also declares no
geometry, because that combination is almost always an adapter author who has not thought about
it yet.

**On `lag_policy="none"`.** This is the honest position for most non-physical domains, and it
is not a failure state: it says the domain has no propagation mechanism, so no geometric floor
exists, so a precedence claim is inadmissible. Association may still be measured. What is
refused is the *interpretation* — and it is refused loudly, at the entry point, rather than
being caught downstream by a message about wavelet filter support that will mean nothing to
someone onboarding a sensor archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UnknownNameError

#: Axis roles the analysis layer understands. An axis declares one; nothing infers it from a
#: coordinate's name. `category` exists so a non-ordered axis (station, instrument, cohort)
#: can be carried without being mistaken for something a lag can be taken along.
AXIS_ROLES: Tuple[str, ...] = ("time", "space", "level", "member", "category")

#: The assumptions the analysis layer inherited from the atmosphere, and what breaking each
#: one costs. A domain names the ones it breaks; the analysis layer then refuses what those
#: violations forbid, rather than relying on the reader to remember.
KNOWN_VIOLATIONS: Dict[str, str] = {
    "no_physical_metric":
        "axes carry no length units, so nothing may be reported per metre and radial "
        "binning in physical wavenumber is unavailable",
    "no_propagation_speed":
        "no mechanism transports structure across the domain, so rule R4's advective lag "
        "floor does not exist and must not be simulated by a plausible-looking speed",
    "no_natural_cycle":
        "there is no diurnal or annual forcing, so the R11 harmonic climatology has nothing "
        "to remove and its default periods would fit noise",
    "irregular_sampling":
        "the clock is not regularly spaced, so a lag in frames is not a lag in time and "
        "cadence-derived quantities are unavailable",
    "non_stationary_support":
        "channels start and stop during the record, so the effective sample size differs "
        "per channel and per pair",
    "unordered_channels":
        "channel labels have no ordering, so any result that reads them as a scale hierarchy "
        "is meaningless",
    "aggregated_values":
        "values are aggregates over a window rather than instantaneous samples, so each "
        "value depends on more than one sample of the parent axis",
}

#: How a domain justifies calling a lag admissible (rule R21).
#:
#: ``advective``  the atmospheric path: filter support crossed at a declared speed.
#: ``declared``   an explicit floor in frames with a recorded, domain-specific basis.
#: ``none``       no floor exists. Association may be measured; precedence may not be claimed.
LAG_POLICIES: Tuple[str, ...] = ("advective", "declared", "none")


class PrecedenceNotAdmissibleError(InvalidParameterError):
    """Raised when a domain with no lag floor is asked for a precedence claim (R21)."""

    def __init__(self, declaration: "DomainDeclaration") -> None:
        super().__init__(
            "domain.lag_policy", declaration.lag_policy,
            "a domain that can justify a minimum admissible lag. Domain %r declares "
            "lag_policy='none'%s, so rule R21 forbids a precedence claim: with no floor, a "
            "lag short enough to sit inside the data's own dependence structure is "
            "indistinguishable from one that carries information. Measure association "
            "instead, or declare a floor with lag_policy='declared' and a recorded basis "
            "(for example an instrument response time, a reporting interval, or a "
            "known minimum transit time)."
            % (declaration.name,
               (" because it also declares %r" % "no_propagation_speed")
               if "no_propagation_speed" in declaration.violations else ""),
            domain=declaration.name, violations=list(declaration.violations))
        self.declaration = declaration


@dataclass(frozen=True)
class AxisSpec:
    """One declared axis. `role` is chosen, never inferred from `name` (standard E14)."""

    name: str
    role: str
    units: Optional[str] = None
    periodic: bool = False
    ordered: bool = True

    def __post_init__(self) -> None:
        if self.role not in AXIS_ROLES:
            raise UnknownNameError("axis role", self.role, AXIS_ROLES, axis=self.name)
        if self.role == "time" and not self.ordered:
            raise InvalidParameterError(
                "AxisSpec.ordered", self.ordered,
                "True for a time axis. An unordered clock makes a lag meaningless, and "
                "declaring one would let a precedence result be computed from it anyway")

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "role": self.role, "units": self.units,
                "periodic": self.periodic, "ordered": self.ordered}


@dataclass(frozen=True)
class DomainDeclaration:
    """What a source is, what it breaks, and what may therefore be claimed from it.

    `licence` is required rather than optional. TG8.2 makes export refuse a derived product
    whose source licence forbids it, and a licence field that can be left empty is one that
    will be — at which point the provenance chain has a hole exactly where a public archive's
    terms should be.
    """

    name: str
    description: str
    axes: Sequence[AxisSpec]
    licence: str
    violations: Sequence[str] = dc_field(default_factory=tuple)
    lag_policy: str = "none"
    declared_floor_frames: Optional[int] = None
    declared_floor_basis: Optional[str] = None
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise InvalidParameterError(
                "DomainDeclaration.name", self.name,
                "a non-empty domain name. Results are attributed to a domain and an "
                "anonymous one cannot be cited or refused by name")
        if not str(self.licence).strip():
            raise InvalidParameterError(
                "DomainDeclaration.licence", self.licence,
                "the source's licence or terms of use. A public archive's terms decide "
                "what may be redistributed, and an empty field would put that decision "
                "beyond the provenance chain")

        for violation in self.violations:
            if violation not in KNOWN_VIOLATIONS:
                raise UnknownNameError("assumption violation", violation,
                                       sorted(KNOWN_VIOLATIONS), domain=self.name)
        if len(set(self.violations)) != len(list(self.violations)):
            raise InvalidParameterError(
                "DomainDeclaration.violations", list(self.violations),
                "distinct violation names")

        time_axes = [axis for axis in self.axes if axis.role == "time"]
        if len(time_axes) != 1:
            raise InvalidParameterError(
                "DomainDeclaration.axes", [axis.name for axis in self.axes],
                "exactly one axis with role 'time'. A channel series is measured against one "
                "clock, and two would make a lag ambiguous")

        if self.lag_policy not in LAG_POLICIES:
            raise UnknownNameError("lag policy", self.lag_policy, LAG_POLICIES,
                                   domain=self.name)
        if self.lag_policy == "declared":
            if not isinstance(self.declared_floor_frames, int) \
                    or self.declared_floor_frames < 1:
                raise InvalidParameterError(
                    "DomainDeclaration.declared_floor_frames", self.declared_floor_frames,
                    "a positive integer number of frames. lag_policy='declared' means the "
                    "domain supplies the floor rule R21 requires; omitting the number makes "
                    "the declaration an assertion rather than a rule")
            if not (self.declared_floor_basis or "").strip():
                raise InvalidParameterError(
                    "DomainDeclaration.declared_floor_basis", self.declared_floor_basis,
                    "a stated basis for the declared floor. A floor without a reason is a "
                    "number the reader cannot check, which is exactly what "
                    "`advection_speed_m_s` is denied a default to prevent")
        elif self.declared_floor_frames is not None:
            raise InvalidParameterError(
                "DomainDeclaration.declared_floor_frames", self.declared_floor_frames,
                "None unless lag_policy='declared'. A floor carried by a policy that does "
                "not use it would be reported and never applied")

        if self.lag_policy == "advective" and "no_propagation_speed" in self.violations:
            raise InvalidParameterError(
                "DomainDeclaration.lag_policy", self.lag_policy,
                "a policy consistent with the declared violations: this domain declares "
                "'no_propagation_speed' and then asks for an advective floor")

        # Rule R17, made enforceable. A domain with no physical metric that also claims to
        # break nothing has not been thought about: at minimum it has broken the metric
        # assumption the analysis layer inherited.
        if not self.violations and self.lag_policy == "none":
            raise InvalidParameterError(
                "DomainDeclaration.violations", list(self.violations),
                "at least one declared violation for a domain with no lag floor (rule R17). "
                "A domain that breaks nothing is a second variable, not a second domain, and "
                "provides no evidence that the abstraction generalises. If that is genuinely "
                "true here, declare the assumptions it does break — most non-physical "
                "sources break at least 'no_physical_metric' and 'no_propagation_speed'")

    # ------------------------------------------------------------------ admissibility

    @property
    def precedence_admissible(self) -> bool:
        """Whether rule R21 permits a precedence (lead-lag) claim from this domain."""
        return self.lag_policy != "none"

    def assert_precedence_admissible(self) -> None:
        if not self.precedence_admissible:
            raise PrecedenceNotAdmissibleError(self)

    def minimum_admissible_lag(self) -> Optional[int]:
        """The declared floor in frames, or None when the policy computes it elsewhere."""
        return self.declared_floor_frames if self.lag_policy == "declared" else None

    def describe(self) -> Dict[str, Any]:
        """The record that travels with every result derived from this domain."""
        return {
            "name": self.name,
            "description": self.description,
            "licence": self.licence,
            "axes": [axis.describe() for axis in self.axes],
            "violations": {name: KNOWN_VIOLATIONS[name] for name in self.violations},
            "lag_policy": self.lag_policy,
            "declared_floor_frames": self.declared_floor_frames,
            "declared_floor_basis": self.declared_floor_basis,
            "precedence_admissible": self.precedence_admissible,
            "provenance": dict(self.provenance),
        }
