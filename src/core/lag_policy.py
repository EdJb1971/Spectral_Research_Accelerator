"""Lag admissibility as a registry, not three branches (TG1.3, standards E1/E16, rule R21).

**What was wrong.** Rule R21 asks one question before any sweep runs: *is a lag admissible at
all?* Three answers existed — an advective crossing time, a domain-declared floor, or no floor
— and they were spread across four files as branches on the string `DomainDeclaration
.lag_policy`. `domain.py` validated the policy-specific fields, `domain_analysis.py` chose the
pre-flight check and assembled the `applied_lag_floor` block, and `cross_scale.py` did not
branch at all: it called `support_floor` **unconditionally**, so the floor that actually
excluded tests was always the atmospheric one.

That last part is the interesting failure, and it is the same shape TG1.2 found in
`laplacian`. Under ``lag_policy='declared'`` the domain's floor was enforced at the entry
point — `analyse_precedence` refused lags below it — and then the sweep applied its own
advective floor of one frame and wrote ``support_floor_frames: 1`` into every test record. No
test was wrong; every *receipt* was, and it named the wrong rule. A silent default, not a
fallback.

**What a policy declares.** Registration carries capabilities, and callers ask about those
rather than about the name:

``precedence_admissible``    rule R21 permits a lead-lag reading of a positive result.
``floor_from_declaration``   the floor is known from the declaration alone, before any data.
``floor_from_geometry``      the floor is computed from the record's physical grid and cadence.
``requires_channel_support`` the policy needs each channel's parent-axis footprint. **False**
                             for every policy that does not measure a filter crossing, which
                             is why a sensor archive no longer has to invent a wavelet number.
``parameters``               the extra arguments the policy consumes. A parameter that no
                             policy in force accepts is refused rather than ignored, which is
                             how "two floors from two different bases" is now caught for any
                             policy instead of for `advective` alone.

**Layering.** This module never imports `DomainDeclaration`; it accepts anything carrying the
handful of attributes it reads, exactly as `physical_core.geometry` accepts a `GridSpec`
without importing `grid.py`. `support_floor` moved here unchanged — including the D48/D53
physical-grid reconstruction — and `analysis_engine.cross_scale` re-exports it, so every
existing caller is untouched and the atmospheric receipt is byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.channel_series import ChannelGeometry
from src.core.errors import InvalidParameterError
from src.core.registry import Registry


class LagPolicy:
    """One answer to rule R21. Subclass, then register.

    Stateless, like `physical_core.geometry.Geometry`: the declaration and the policy's
    parameters arrive as arguments, so a registered policy is a singleton with nothing to get
    out of step with the domain it is serving.
    """

    #: Registry name. Set by `register`, and used in the receipts.
    name: str = ""

    # ------------------------------------------------------------- declaration time

    def validate_declaration(self, declaration: Any) -> None:
        """Refuse a declaration this policy cannot act on.

        The base implementation enforces the one rule that is true of every policy that does
        not read a declared floor: carrying the number and never applying it is worse than
        not carrying it, because it will be read from the receipt as though it applied.
        """
        if declaration.declared_floor_frames is not None:
            raise InvalidParameterError(
                "DomainDeclaration.declared_floor_frames", declaration.declared_floor_frames,
                "None unless the lag policy reads it. Policy %r declares "
                "floor_from_declaration=False, so this floor would be reported in the "
                "receipt and never applied to a single test" % self.name,
                domain=getattr(declaration, "name", None))

    def declared_floor_frames(self, declaration: Any) -> Optional[int]:
        """The floor in frames if the declaration alone fixes it, else None."""
        return None

    # ------------------------------------------------------------- sweep time

    def validate_params(self, params: Mapping[str, Any]) -> None:
        """Refuse a parameter this policy does not consume.

        Ignoring it would be the "two floors from two different bases" failure: the sweep
        would take one of them, and the result would not say which.
        """
        accepted = tuple(capability(self.name, "parameters", ()) or ())
        unexpected = sorted(key for key in params if key not in accepted)
        if unexpected:
            raise InvalidParameterError(
                "lag_policy_params", unexpected,
                "only the parameters lag policy %r consumes (%s). Two floors from two "
                "different bases would silently take one of them, and the result would not "
                "say which applied"
                % (self.name, ", ".join(accepted) if accepted else "none"))

    def check_lags(self, lags: Sequence[int], declaration: Any,
                   params: Mapping[str, Any]) -> None:
        """Refuse a requested lag family this policy cannot admit, before the sweep runs."""

    def floors(self, signature: ChannelGeometry, cadence_seconds: float,
               declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        """The per-channel floor block that decides which tests are admissible."""
        raise NotImplementedError

    def exclusion_reason(self, floor_frames: int) -> str:
        """Why a test below the floor was excluded, in this policy's own vocabulary."""
        return "below the lag floor of %d frame(s) applied by policy %r" % (
            floor_frames, self.name)

    def config_entries(self, declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        """This policy's contribution to the analysis fingerprint.

        The policy owns its own entries so the hash records what actually set the floor. The
        advective policy contributes exactly the key the pre-TG1.3 code hard-coded, which is
        why every atmospheric `analysis_config_sha256` is unchanged.
        """
        return {}

    def applied_floor(self, declaration: Any, floors: Mapping[str, Any]) -> Dict[str, Any]:
        """What the receipt says about which floor decided admissibility, and why."""
        frames = [int(record["floor_frames"]) for record in floors.get("floors", ())]
        return {"policy": self.name,
                "frames": max(frames) if frames else None,
                "basis": str(floors.get("basis") or "")}


#: Every lag policy in force. A fourth registers from outside `src/` (see
#: `test_lag_policy_registry.py`), which is the acceptance criterion for this seam.
LAG_POLICIES: Registry[LagPolicy] = Registry("lag policy")


def lag_policy_for(name: str) -> LagPolicy:
    """The registered policy, or `UnknownNameError` naming the ones that exist."""
    return LAG_POLICIES.get(str(name))


def capability(name: str, key: str, default: Any = None) -> Any:
    """One declared capability of a registered policy."""
    return LAG_POLICIES.entry(str(name)).capabilities.get(key, default)


def policy_names() -> Tuple[str, ...]:
    """The registered policy names, sorted. Replaces the old closed `LAG_POLICIES` tuple."""
    return tuple(sorted(LAG_POLICIES.names()))


@dataclass(frozen=True)
class BoundLagPolicy:
    """A policy bound to the declaration and parameters that make it concrete.

    The sweep is handed one of these rather than a policy name plus a bag of keyword
    arguments, so there is exactly one place where "which floor applies here" is decided.
    """

    policy: LagPolicy
    declaration: Any = None
    params: Mapping[str, Any] = dc_field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.policy.name

    def capability(self, key: str, default: Any = None) -> Any:
        return capability(self.policy.name, key, default)

    def check_lags(self, lags: Sequence[int]) -> None:
        self.policy.check_lags(lags, self.declaration, self.params)

    def floors(self, signature: ChannelGeometry, cadence_seconds: float) -> Dict[str, Any]:
        return self.policy.floors(signature, cadence_seconds, self.declaration, self.params)

    def exclusion_reason(self, floor_frames: int) -> str:
        return self.policy.exclusion_reason(int(floor_frames))

    def config_entries(self) -> Dict[str, Any]:
        return self.policy.config_entries(self.declaration, self.params)

    def applied_floor(self, floors: Mapping[str, Any]) -> Dict[str, Any]:
        return self.policy.applied_floor(self.declaration, floors)


def bind(name: str, declaration: Any = None, **params: Any) -> BoundLagPolicy:
    """Resolve a policy and check its parameters before anything is computed."""
    policy = lag_policy_for(name)
    supplied = {key: value for key, value in params.items() if value is not None}
    policy.validate_params(supplied)
    return BoundLagPolicy(policy=policy, declaration=declaration, params=supplied)


# ------------------------------------------------------------------ shared floor assembly

def _uniform_floors(signature: ChannelGeometry, cadence_seconds: float, frames: int,
                    basis: str) -> List[Dict[str, Any]]:
    """One floor per channel, all the same, carrying the footprint only if it was declared.

    A policy that does not measure a filter crossing has no use for `support_parent_px`, so
    it neither requires nor discards it: an adapter that has the number keeps it in the
    receipt, and one that does not is not asked to invent it.
    """
    records = list(signature.channel_records)
    out: List[Dict[str, Any]] = []
    for position, scale in enumerate(signature.channels, start=1):
        try:
            level = int(scale)
        except (TypeError, ValueError):
            level = position
        declared = (records[position - 1].get("support_parent_px")
                    if position - 1 < len(records) else None)
        record: Dict[str, Any] = {"scale": str(scale), "level": level,
                                  "floor_frames": int(frames), "basis": basis}
        if isinstance(declared, (int, float)) and declared > 0:
            record["spatial_support_px"] = declared
        out.append(record)
    return out


def _require_positive_cadence(cadence_seconds: float) -> None:
    if cadence_seconds <= 0:
        raise InvalidParameterError("cadence_seconds", cadence_seconds,
                                    "a positive sampling interval")


# ------------------------------------------------------------------ the advective floor (R4)

def support_floor(signature: ChannelGeometry, cadence_seconds: float,
                  advection_speed_m_s: Optional[float] = None) -> Dict[str, Any]:
    """The minimum admissible lag per scale, and an honest account of where it comes from.

    Rule R4 says a lag shorter than the transform's own support measures filter geometry
    rather than weather. For a **spatial** transform applied frame by frame - which is what
    Phase 4 uses, deliberately, since 3D wavelets are out of scope - the temporal support is
    exactly zero, and quoting one would be an invention. What is real is that a structure
    must cross the filter's spatial support before a change at that scale can be anything
    other than the same air seen twice: `t_cross(s) = support(s) * dx / U`.

    `advection_speed_m_s` is therefore required for a geometric floor and is **not given a
    default**. A plausible-looking 10 m/s would silently set every floor in every result, and
    a reader would have no way to know a number they never supplied was doing the work.
    Without it the only floor applied is one frame, and the result says so in as many words.

    Moved into this module by TG1.3, unchanged, as the `advective` policy's implementation.
    `analysis_engine.cross_scale` re-exports it for the gate modules that call it directly.
    """
    _require_positive_cadence(cadence_seconds)
    grid = signature.provenance.get("grid") or {}
    spacing_m = None
    spacing_basis = None
    kind = grid.get("kind")
    if kind in ("latlon", "cartesian"):
        try:
            from src.physical_core.grid import GridSpec
            physical_grid = GridSpec.from_provenance(dict(grid))
            # Support is expressed in parent-grid pixels. With no frozen flow direction,
            # use the larger physical cell axis so an anisotropic/lat-lon grid cannot make
            # the crossing floor anti-conservative. In particular, GridSpec.dx is degrees
            # for lat/lon and must never be interpreted as metres (D53).
            dx_values = physical_grid.dx_metres()
            dy_values = physical_grid.dy_metres()
            spacing_m = max(float(dx_values.max()), float(dy_values.max()))
            spacing_basis = (
                "maximum physical cell-axis spacing over the crop, reconstructed from "
                "GridSpec; angular dx/dy are converted to metres"
                if kind == "latlon" else
                "maximum declared Cartesian cell-axis spacing in metres")
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidParameterError(
                "signature.provenance.grid", grid,
                "a complete physical GridSpec for an advective support floor: %s" % exc) from exc
    else:
        # Compatibility with older explicitly metric provenance. Deliberately do not read
        # bare `dx`: its unit depends on grid kind and caused D53.
        candidate = grid.get("representative_dx_metres") or grid.get("dx_metres")
        if isinstance(candidate, (int, float)) and candidate \
                and math.isfinite(candidate) and candidate > 0:
            spacing_m = float(candidate)
            spacing_basis = "legacy provenance field explicitly labelled in metres"
    physical = (isinstance(spacing_m, (int, float)) and math.isfinite(spacing_m)
                and spacing_m > 0)

    floors: List[Dict[str, Any]] = []
    warnings: List[str] = []
    # Hoisted: `channel_records` is a property, and on a producer that builds its records on
    # demand rather than storing them, reading it per channel is quadratic for no reason.
    records = list(signature.channel_records)
    for position, scale in enumerate(signature.channels, start=1):
        try:
            level = int(scale)
        except (TypeError, ValueError):
            level = position
        interior_record = records[position - 1] if position - 1 < len(records) else {}
        support_px = interior_record.get("support_parent_px")
        if not isinstance(support_px, (int, float)) or support_px <= 0:
            raise InvalidParameterError(
                "channel_records[%d].support_parent_px" % (position - 1), support_px,
                "the transform's measured positive parent-grid filter support. Using 2**level "
                "would understate the shared spatial footprint for longer filters")
        record: Dict[str, Any] = {
            "scale": str(scale),
            "level": level,
            "spatial_support_px": support_px,
            "floor_frames": 1,
            "basis": ("sampling cadence only: consecutive frames are the finest lag the "
                      "record can express; spatial support is the transform's exact %d-pixel "
                      "filter cascade" % support_px),
        }
        if physical and advection_speed_m_s:
            support_m = support_px * float(spacing_m)
            crossing = support_m / float(advection_speed_m_s)
            frames = max(1, int(math.ceil(crossing / cadence_seconds)))
            record.update({
                "spatial_support_m": support_m,
                "physical_spacing_m_per_parent_px": float(spacing_m),
                "physical_spacing_basis": spacing_basis,
                "crossing_time_s": crossing,
                "floor_frames": frames,
                "basis": ("advective crossing of the filter support: %.0f m at %.1f m/s is "
                          "%.0f s, which is %d frame(s) at this cadence"
                          % (support_m, advection_speed_m_s, crossing, frames)),
            })
        floors.append(record)

    if not physical:
        warnings.append(
            "the signature's grid carries no physical spacing, so no advective floor could "
            "be computed and the only floor applied is one frame. A lag floor in metres "
            "derived from a pixel grid would be fabricated.")
    if advection_speed_m_s is None:
        warnings.append(
            "no advection speed was supplied, so the geometric floor of rule R4 is not "
            "enforced. This is reported rather than defaulted: a default speed would set "
            "every floor in every result from a number the reader never chose.")

    return {
        "policy": "advective",
        "cadence_seconds": float(cadence_seconds),
        "advection_speed_m_s": (None if advection_speed_m_s is None
                                else float(advection_speed_m_s)),
        "floors": floors,
        "floor_by_scale": {record["scale"]: record["floor_frames"] for record in floors},
        "enforced": bool(physical and advection_speed_m_s),
        "temporal_support_note": (
            "the transform is spatial and is applied frame by frame, so its temporal support "
            "is zero. The floor below is advective, not filter-geometric, and it is the "
            "honest version of rule R4 for a 2D-per-frame decomposition."),
        "warnings": warnings,
    }


# ------------------------------------------------------------------ the three policies


class AdvectivePolicy(LagPolicy):
    """The atmospheric path, and the only policy that reads the record's own geometry."""

    name = "advective"

    def validate_declaration(self, declaration: Any) -> None:
        super().validate_declaration(declaration)
        if "no_propagation_speed" in tuple(getattr(declaration, "violations", ()) or ()):
            raise InvalidParameterError(
                "DomainDeclaration.lag_policy", self.name,
                "a policy consistent with the declared violations: this domain declares "
                "'no_propagation_speed' and then asks for an advective floor")

    def check_lags(self, lags: Sequence[int], declaration: Any,
                   params: Mapping[str, Any]) -> None:
        if params.get("advection_speed_m_s") is None:
            raise InvalidParameterError(
                "advection_speed_m_s", None,
                "a declared transport speed under lag_policy='advective'. It has no default "
                "here for the same reason it has none in `support_floor`: a plausible-looking "
                "value would set every floor in every result from a number the reader never "
                "chose")

    def floors(self, signature: ChannelGeometry, cadence_seconds: float,
               declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        return support_floor(signature, cadence_seconds,
                             params.get("advection_speed_m_s"))

    def exclusion_reason(self, floor_frames: int) -> str:
        return ("below the rule R4 support floor of %d frame(s); at this lag the two scales "
                "are the same air seen twice through overlapping filters" % floor_frames)

    def config_entries(self, declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        speed = params.get("advection_speed_m_s")
        return {"advection_speed_m_s": None if speed is None else float(speed)}

    def applied_floor(self, declaration: Any, floors: Mapping[str, Any]) -> Dict[str, Any]:
        return {"policy": self.name,
                "frames": max(int(record["floor_frames"])
                              for record in floors["floors"]),
                "basis": "advective crossing of the declared representation support"}


LAG_POLICIES.add(
    "advective", AdvectivePolicy(),
    description="Filter support crossed at a declared transport speed (rule R4).",
    params={"advection_speed_m_s": "transport speed in m/s. No default, deliberately."},
    capabilities={"precedence_admissible": True, "floor_from_declaration": False,
                  "floor_from_geometry": True, "requires_channel_support": True,
                  "parameters": ("advection_speed_m_s",)},
    tags=["physical"])


class DeclaredPolicy(LagPolicy):
    """The domain supplies the floor, and must supply the reason with it."""

    name = "declared"

    def validate_declaration(self, declaration: Any) -> None:
        frames = declaration.declared_floor_frames
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
            raise InvalidParameterError(
                "DomainDeclaration.declared_floor_frames", frames,
                "a positive integer number of frames. lag_policy='declared' means the "
                "domain supplies the floor rule R21 requires; omitting the number makes "
                "the declaration an assertion rather than a rule")
        if not (getattr(declaration, "declared_floor_basis", None) or "").strip():
            raise InvalidParameterError(
                "DomainDeclaration.declared_floor_basis",
                getattr(declaration, "declared_floor_basis", None),
                "a stated basis for the declared floor. A floor without a reason is a "
                "number the reader cannot check, which is exactly what "
                "`advection_speed_m_s` is denied a default to prevent")

    def declared_floor_frames(self, declaration: Any) -> Optional[int]:
        return int(declaration.declared_floor_frames)

    def check_lags(self, lags: Sequence[int], declaration: Any,
                   params: Mapping[str, Any]) -> None:
        floor = int(declaration.declared_floor_frames)
        below = [int(lag) for lag in lags if int(lag) < floor]
        if below:
            raise InvalidParameterError(
                "lags", below,
                "lags at or above domain %r's declared floor of %d frame(s) (%s). They are "
                "refused rather than dropped, because a family silently reduced to the "
                "testable lags is indistinguishable from one that had nothing to drop, and "
                "rule R18 fixes the family before the sweep runs"
                % (declaration.name, floor, declaration.declared_floor_basis))

    def floors(self, signature: ChannelGeometry, cadence_seconds: float,
               declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        _require_positive_cadence(cadence_seconds)
        frames = int(declaration.declared_floor_frames)
        basis = "declared by domain %r: %s" % (declaration.name,
                                               declaration.declared_floor_basis)
        records = _uniform_floors(signature, cadence_seconds, frames, basis)
        return {
            "policy": self.name,
            "cadence_seconds": float(cadence_seconds),
            "declared_floor_frames": frames,
            "declared_floor_basis": str(declaration.declared_floor_basis),
            "floors": records,
            "floor_by_scale": {record["scale"]: frames for record in records},
            "enforced": True,
            "basis": basis,
            "temporal_support_note": (
                "the floor is the domain's, not the transform's: it applies equally to every "
                "channel because it describes the record rather than the representation."),
            "warnings": [],
        }

    def exclusion_reason(self, floor_frames: int) -> str:
        return ("below the domain's declared admissible lag floor of %d frame(s) (rule R21)"
                % floor_frames)

    def config_entries(self, declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {"declared_floor_frames": int(declaration.declared_floor_frames),
                "declared_floor_basis": str(declaration.declared_floor_basis)}

    def applied_floor(self, declaration: Any, floors: Mapping[str, Any]) -> Dict[str, Any]:
        return {"policy": self.name,
                "frames": int(declaration.declared_floor_frames),
                "basis": str(declaration.declared_floor_basis)}


LAG_POLICIES.add(
    "declared", DeclaredPolicy(),
    description="An explicit floor in frames, with a recorded domain-specific basis.",
    capabilities={"precedence_admissible": True, "floor_from_declaration": True,
                  "floor_from_geometry": False, "requires_channel_support": False,
                  "parameters": ()},
    tags=["declared"])


class NoFloorPolicy(LagPolicy):
    """The honest position for most non-physical domains, and not a failure state."""

    name = "none"

    def validate_declaration(self, declaration: Any) -> None:
        super().validate_declaration(declaration)
        # Rule R17, made enforceable. A domain with no physical metric that also claims to
        # break nothing has not been thought about: at minimum it has broken the metric
        # assumption the analysis layer inherited.
        if not tuple(getattr(declaration, "violations", ()) or ()):
            raise InvalidParameterError(
                "DomainDeclaration.violations", [],
                "at least one declared violation for a domain with no lag floor (rule R17). "
                "A domain that breaks nothing is a second variable, not a second domain, and "
                "provides no evidence that the abstraction generalises. If that is genuinely "
                "true here, declare the assumptions it does break - most non-physical "
                "sources break at least 'no_physical_metric' and 'no_propagation_speed'")

    def floors(self, signature: ChannelGeometry, cadence_seconds: float,
               declaration: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        _require_positive_cadence(cadence_seconds)
        basis = ("sampling cadence only: consecutive frames are the finest lag the record "
                 "can express, and no mechanism justifies a longer floor")
        records = _uniform_floors(signature, cadence_seconds, 1, basis)
        return {
            "policy": self.name,
            "cadence_seconds": float(cadence_seconds),
            "floors": records,
            "floor_by_scale": {record["scale"]: 1 for record in records},
            "enforced": False,
            "basis": basis,
            "temporal_support_note": (
                "no floor exists for this domain (rule R21), so nothing here licenses a "
                "precedence reading of a positive result at any lag."),
            "warnings": [
                "domain %r declares no admissible lag floor, so the only floor applied is "
                "one frame. That is a statement about the domain, not a defaulted number: a "
                "floor invented from a mechanism the domain has said it does not have would "
                "look like the atmospheric one and mean nothing."
                % getattr(declaration, "name", "?")],
        }

    def exclusion_reason(self, floor_frames: int) -> str:
        return ("below one frame, which is the finest lag the record can express"
                if floor_frames <= 1 else super().exclusion_reason(floor_frames))

    def applied_floor(self, declaration: Any, floors: Mapping[str, Any]) -> Dict[str, Any]:
        return {"policy": self.name, "frames": None,
                "basis": "no floor exists for this domain (rule R21)"}


LAG_POLICIES.add(
    "none", NoFloorPolicy(),
    description="No floor exists. Association may be measured; precedence may not be claimed.",
    capabilities={"precedence_admissible": False, "floor_from_declaration": False,
                  "floor_from_geometry": False, "requires_channel_support": False,
                  "parameters": ()},
    tags=["declared"])
