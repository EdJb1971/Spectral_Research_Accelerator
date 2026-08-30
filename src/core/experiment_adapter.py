"""The registered domain adapter contract (TG17.3, `ed-dev`).

**The switch statement this replaces.** Until this module, adding a domain to a cross-domain
experiment meant editing three unrelated places. `src/api/acquisitions.py` assembled its
catalogue from four hand-written shape branches — `_grid_acquisitions`, `_profile_acquisitions`,
`_lightcurve_acquisitions`, `_channel_acquisition` — and a route that called each by name.
`preflight_manifest` carried a literal ``source_plans`` table mapping four source ids to a
five-tuple, plus a ``source == "channel_table:local"`` special case. The benchmark translators
carried a ``FIXTURE_TO_MANIFEST`` lookup. None of those is an extension point: each is a place
a fifth domain has to be remembered, and a place it can be forgotten.

A `DomainExperimentAdapter` is the whole of what a domain must supply, in one registration:

===========================  ==============================================================
``declaration``              what the domain is and what it breaks (E14, E15, R17, R21)
``controls``                 the typed schema the Composer renders — never a hardcoded form
``plan_acquisition``         native addressing, support kind, access and cost, from metadata
``materialize``              the content-addressed native record for one explicit window
``structural_declaration``   the executable translation contract (TG17.2)
``translate``                native record to canonical `StructuralTrajectory`
``derive_capabilities``      what this record does and does not support, as facts
``build_null``               the domain-legitimate null this record admits
``render_provenance``        how a canonical value is shown to have arisen
===========================  ==============================================================

**Window arithmetic belongs to the framework, not to the adapter.** `plan_acquisition` returns
what only the domain knows — native cadence, support kind, whether a stated extent establishes
exact coverage, what access it needs, what it costs per day. `plan_windows` then does the
interval arithmetic for every declared window, identically for every domain. This is the line
that decides whether onboarding cost is domain mathematics or framework glue: an adapter that
had to compute its own expected-sample counts would be re-implementing the framework, and four
such adapters would drift into four subtly different definitions of "expected".

**Detection may create a required declaration; it may never satisfy one.** This module holds
the same rule TG8.4 established in `src/data_layer/tabular_source.py`. Nothing here infers a
domain's violations, lag policy or axis roles from the data it is handed. An adapter supplies a
`DomainDeclaration` that was onboarded through the contract, and the conformance kit checks the
adapter against that declaration rather than the other way round.

**What a registration deliberately cannot do.** It cannot widen the canonical channel
vocabulary — channels come from the `StructuralAdapterDeclaration`, which requires a benchmark
identity per channel. It cannot claim precedence its declaration does not admit. It cannot
report exact coverage from an intersection. And it cannot register a domain that has not passed
`onboard_domain`, because a domain assembled from separate registrations was never checked as a
whole and what it refuses is therefore unknown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.onboarding import is_onboarded
from src.core.registry import Registry


SCHEMA = "domain-experiment-adapter/v1"
CONTROL_SCHEMA = "domain-adapter-controls/v1"
PLAN_SCHEMA = "domain-acquisition-plan/v1"

#: What a control may be. Deliberately small and closed: the Composer renders these and nothing
#: else, so an adapter cannot smuggle in a bespoke widget that only its own domain understands.
#: `content_record` is a selector over content-addressed local records; `utc_instant` carries an
#: offset-bearing instant, never a naive one.
CONTROL_KINDS: Tuple[str, ...] = ("text", "integer", "number", "boolean", "enum",
                                  "utc_instant", "content_record")

#: How a domain's native support answers "what does one observation cover?". A plan declaring
#: `coverage_exact=False` may never be reported as exact merely because its extent intersects a
#: requested window — the TESS sector case that TG17.1 froze.
SUPPORT_KINDS: Tuple[str, ...] = ("regular_grid_extent", "sparse_point_support",
                                  "intersecting_observational_sectors", "irregular_local_record")


class AdapterConformanceError(InvalidParameterError):
    """A registered adapter violated the contract it is registered under."""

    def __init__(self, adapter_id: str, requirement: str, detail: str) -> None:
        super().__init__("adapter.%s" % requirement, adapter_id, detail, adapter_id=adapter_id)
        self.adapter_id = adapter_id
        self.requirement = requirement


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _callable_identity(function: Optional[Callable[..., Any]]) -> Optional[str]:
    """Identify the code path without pretending to have hashed its behaviour."""
    if function is None:
        return None
    return "%s:%s" % (getattr(function, "__module__", "?"),
                      getattr(function, "__qualname__", repr(function)))


# --------------------------------------------------------------------------- typed controls


@dataclass(frozen=True)
class ControlField:
    """One control the Composer renders for this domain, declared rather than hardcoded."""

    name: str
    label: str
    kind: str
    help: str
    required: bool = True
    default: Any = None
    choices: Tuple[Any, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    units: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in CONTROL_KINDS:
            raise InvalidParameterError(
                "ControlField.kind", self.kind,
                "one of %s. The Composer renders a closed set of controls so that a domain "
                "cannot require a widget only it understands" % (list(CONTROL_KINDS),))
        if not str(self.name).strip() or not str(self.label).strip():
            raise InvalidParameterError("ControlField.name", self.name,
                                        "a non-empty machine name and human label")
        if not str(self.help).strip():
            raise InvalidParameterError(
                "ControlField.help", self.help,
                "an explanation of what this control decides. A control whose meaning lives "
                "only in the adapter author's head cannot be operated by a researcher")
        if self.kind == "enum" and not self.choices:
            raise InvalidParameterError("ControlField.choices", self.choices,
                                        "at least one choice for an enum control")
        if self.kind != "enum" and self.choices:
            raise InvalidParameterError("ControlField.choices", self.choices,
                                        "choices only on an enum control")

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label, "kind": self.kind, "help": self.help,
                "required": self.required, "default": self.default,
                "choices": list(self.choices), "minimum": self.minimum,
                "maximum": self.maximum, "units": self.units}

    def coerce(self, value: Any) -> Any:
        """Normalise one supplied value, or refuse it in the researcher's terms."""
        if self.kind == "boolean":
            if not isinstance(value, bool):
                raise InvalidParameterError(self.name, value, "true or false")
            return value
        if self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidParameterError(self.name, value, "a whole number")
            return int(self._bounded(int(value)))
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidParameterError(self.name, value, "a number")
            return float(self._bounded(float(value)))
        if self.kind == "enum":
            if value not in self.choices:
                raise InvalidParameterError(self.name, value,
                                            "one of %s" % (list(self.choices),))
            return value
        if self.kind == "utc_instant":
            return _utc_instant(self.name, value)
        if self.kind == "content_record":
            text = str(value)
            if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
                raise InvalidParameterError(
                    self.name, value,
                    "a 64-character lowercase sha256 of a record already published to this "
                    "server. A path or a filename is not an identity: two files with the same "
                    "name are not the same record, and a result must name the bytes it read")
            return text
        text = str(value)
        if not text.strip():
            raise InvalidParameterError(self.name, value, "a non-empty value")
        return text

    def _bounded(self, value: float) -> float:
        if self.minimum is not None and value < self.minimum:
            raise InvalidParameterError(self.name, value, "at least %s" % self.minimum)
        if self.maximum is not None and value > self.maximum:
            raise InvalidParameterError(self.name, value, "at most %s" % self.maximum)
        return value


def _utc_instant(name: str, value: Any) -> datetime:
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise InvalidParameterError(
            name, value,
            "an instant carrying a UTC offset. A naive timestamp acquires whichever zone the "
            "reader assumes, and a cross-domain window compared under two assumed zones is "
            "not a shared window")
    return moment.astimezone(timezone.utc)


@dataclass(frozen=True)
class ControlSchema:
    """Every control one domain contributes, and the only way it may contribute any."""

    fields: Tuple[ControlField, ...] = ()

    def __post_init__(self) -> None:
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise InvalidParameterError("ControlSchema.fields", names, "distinct control names")

    def describe(self) -> Dict[str, Any]:
        return {"schema": CONTROL_SCHEMA, "fields": [item.describe() for item in self.fields]}

    def validate(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        """Normalised parameters, refusing anything the schema does not declare.

        Unknown keys are refused rather than ignored. A silently dropped parameter is a
        researcher who believes they configured something they did not, and the manifest would
        record the setting while the run ignored it.
        """
        supplied = dict(parameters)
        declared = {item.name: item for item in self.fields}
        unknown = sorted(set(supplied) - set(declared))
        if unknown:
            raise InvalidParameterError(
                "parameters", unknown,
                "only controls this domain declares: %s" % (sorted(declared) or "none",))
        resolved: Dict[str, Any] = {}
        for name, control in declared.items():
            if name in supplied and supplied[name] is not None:
                resolved[name] = control.coerce(supplied[name])
            elif control.default is not None:
                resolved[name] = control.coerce(control.default)
            elif control.required:
                raise InvalidParameterError(name, None, "a value: %s" % control.help)
        return resolved


# ------------------------------------------------------------------------ acquisition plans


@dataclass(frozen=True)
class AcquisitionPlan:
    """What only the domain knows about reaching its archive, from metadata alone.

    It answers questions about *addressing*, never about values. `opens_measurement_values` is
    a field rather than an assumption so that a later slice which does open values has to say
    so in the plan a researcher can read.
    """

    source_id: str
    source_version: str
    support_kind: str
    access: str
    access_means: str
    coverage_exact: bool
    native_cadence_seconds: Optional[float] = None
    estimated_bytes_per_day: int = 0
    opens_measurement_values: bool = False
    network_used: bool = False
    identity: Mapping[str, Any] = dc_field(default_factory=dict)
    refusal: Optional[str] = None

    def __post_init__(self) -> None:
        if self.support_kind not in SUPPORT_KINDS:
            raise InvalidParameterError("AcquisitionPlan.support_kind", self.support_kind,
                                        "one of %s" % (list(SUPPORT_KINDS),))
        if self.coverage_exact and self.native_cadence_seconds is None:
            raise InvalidParameterError(
                "AcquisitionPlan.native_cadence_seconds", None,
                "a cadence, because this plan claims exact coverage. Exactness without a "
                "cadence cannot be turned into an expected sample count, so the claim could "
                "not be checked against what is later acquired")
        if self.native_cadence_seconds is not None and self.native_cadence_seconds <= 0:
            raise InvalidParameterError("AcquisitionPlan.native_cadence_seconds",
                                        self.native_cadence_seconds, "a positive duration")
        if self.estimated_bytes_per_day < 0:
            raise InvalidParameterError("AcquisitionPlan.estimated_bytes_per_day",
                                        self.estimated_bytes_per_day, "a non-negative size")
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))

    def describe(self) -> Dict[str, Any]:
        return {"schema": PLAN_SCHEMA, "source_id": self.source_id,
                "source_version": self.source_version, "support_kind": self.support_kind,
                "access": self.access, "access_means": self.access_means,
                "coverage_exact": self.coverage_exact,
                "native_cadence_seconds": self.native_cadence_seconds,
                "estimated_bytes_per_day": self.estimated_bytes_per_day,
                "opens_measurement_values": self.opens_measurement_values,
                "network_used": self.network_used, "identity": dict(self.identity),
                "refusal": self.refusal}


def plan_windows(plan: AcquisitionPlan, windows: Sequence[Any]) -> List[Dict[str, Any]]:
    """Interval arithmetic for every declared window, done once for every domain.

    Deliberately not an adapter responsibility. Four adapters computing their own expected
    sample counts would become four definitions of "expected", and the disagreement would only
    surface as a coverage matrix whose columns could not be compared.
    """
    rows: List[Dict[str, Any]] = []
    for window in windows:
        seconds = (window.end_utc - window.start_utc).total_seconds()
        cadence = plan.native_cadence_seconds
        rows.append({
            "name": window.name,
            "start_utc": window.start_utc.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"),
            "end_utc": window.end_utc.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"),
            "expected_samples": int(seconds // cadence) if cadence else None,
            "expected_samples_basis": ("nominal product cadence if archive coverage exists"
                                       if cadence
                                       else "unknown for native sparse/irregular support"),
            "estimated_bytes": int(seconds / 86400.0 * plan.estimated_bytes_per_day),
            "coverage_exact": plan.coverage_exact,
            "gaps": ("none implied by extent metadata" if plan.coverage_exact
                     else "not established until archive metadata is returned"),
        })
    return rows


# ------------------------------------------------------------------------------- the adapter


@dataclass(frozen=True)
class DomainExperimentAdapter:
    """One domain's complete contribution to a cross-domain experiment."""

    adapter_id: str
    adapter_version: str
    declaration: DomainDeclaration
    controls: ControlSchema
    plan_acquisition: Callable[..., AcquisitionPlan]
    structural_declaration: Callable[..., Any]
    translate: Callable[..., Any]
    #: The exact translator configuration this adapter runs under, whose digest the canonical
    #: record carries. It is part of the contract rather than an assumption of the conformance
    #: pass: TG17.2's pass compared every trajectory against `{"ddof": 0}`, which silently made
    #: one channel's configuration the definition of a correct configuration for all of them.
    translator_config: Optional[Callable[..., Mapping[str, Any]]] = None
    materialize: Optional[Callable[..., Any]] = None
    derive_capabilities: Optional[Callable[..., Mapping[str, Any]]] = None
    build_null: Optional[Callable[..., Any]] = None
    render_provenance: Optional[Callable[..., Mapping[str, Any]]] = None
    onboarding_cost: Mapping[str, Any] = dc_field(default_factory=dict)
    definition_sha256: str = dc_field(init=False, default="")

    def __post_init__(self) -> None:
        if not str(self.adapter_id).strip() or not str(self.adapter_version).strip():
            raise InvalidParameterError("DomainExperimentAdapter.adapter_id", self.adapter_id,
                                        "a non-empty adapter identity and version")
        if not is_onboarded(self.declaration.name):
            raise AdapterConformanceError(
                self.adapter_id, "onboarded",
                "domain %r has not passed `onboard_domain`. A domain assembled from separate "
                "registrations was never checked as a whole, so what it refuses is unknown and "
                "an adapter over it would inherit refusals nobody has verified exist"
                % self.declaration.name)
        object.__setattr__(self, "onboarding_cost",
                           MappingProxyType(dict(self.onboarding_cost)))
        object.__setattr__(self, "definition_sha256", _digest({
            "schema": SCHEMA, "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "declaration": self.declaration.describe(),
            "controls": self.controls.describe(),
            "code": {name: _callable_identity(getattr(self, name))
                     for name in ("plan_acquisition", "structural_declaration", "translate",
                                  "translator_config", "materialize", "derive_capabilities",
                                  "build_null", "render_provenance")},
        }))

    @property
    def domain(self) -> str:
        return self.declaration.name

    def resolve(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        return self.controls.validate(parameters)

    def config(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        """The declared translator configuration, or an empty one when none is declared."""
        return dict(self.translator_config(parameters)) if self.translator_config else {}

    def plan(self, parameters: Mapping[str, Any],
             identity: Optional[Mapping[str, Any]] = None) -> AcquisitionPlan:
        """Resolve the controls, then ask the domain what reaching its archive would mean.

        `identity` is the manifest's acquisition identity — which bytes, under which licence
        scope — and is deliberately separate from the controls. A control is a choice a
        researcher makes; an identity is a fact about the record that choice addresses, and
        conflating them would let a re-run silently address different bytes.
        """
        return self.plan_acquisition(self.resolve(parameters), dict(identity or {}))

    def describe(self) -> Dict[str, Any]:
        """The inventory row: generated, never hand-maintained (standard E1)."""
        return {
            "schema": SCHEMA, "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version, "domain": self.domain,
            "definition_sha256": self.definition_sha256,
            "controls": self.controls.describe(),
            "declaration": self.declaration.describe(),
            "implements": sorted(name for name in
                                 ("translator_config", "materialize", "derive_capabilities",
                                  "build_null", "render_provenance")
                                 if getattr(self, name) is not None),
            "onboarding_cost": dict(self.onboarding_cost),
        }


#: Every registered domain adapter (standard E1). The acquisition catalogue, the Composer's
#: controls and the metadata preflight all read this registry, so a fifth domain reaches all
#: three by registering once rather than by being remembered in three places.
EXPERIMENT_ADAPTERS: Registry[DomainExperimentAdapter] = Registry("domain experiment adapter")


def register_experiment_adapter(adapter: DomainExperimentAdapter,
                                *, replace: bool = False) -> DomainExperimentAdapter:
    EXPERIMENT_ADAPTERS.add(
        adapter.adapter_id, adapter, description=adapter.declaration.description,
        params=adapter.controls.describe(),
        capabilities={"domain": adapter.domain,
                      "precedence_admissible": adapter.declaration.precedence_admissible,
                      "definition_sha256": adapter.definition_sha256},
        tags=[adapter.domain], replace=replace)
    return adapter


def adapter_for(adapter_id: str) -> DomainExperimentAdapter:
    return EXPERIMENT_ADAPTERS.get(adapter_id)


def adapter_for_domain(domain: str) -> DomainExperimentAdapter:
    """The single adapter registered for a domain, or a refusal naming the ambiguity."""
    matches = [entry.value for entry in EXPERIMENT_ADAPTERS.entries()
               if entry.value.domain == domain]
    if not matches:
        raise InvalidParameterError(
            "domain", domain,
            "a domain with a registered experiment adapter. Registered: %s"
            % (sorted({entry.value.domain for entry in EXPERIMENT_ADAPTERS.entries()})
               or "none",))
    if len(matches) > 1:
        raise InvalidParameterError(
            "domain", domain,
            "an unambiguous domain. %d adapters claim it (%s), so which one produced a result "
            "could not be recorded" % (len(matches), ", ".join(sorted(
                item.adapter_id for item in matches))))
    return matches[0]


def registered_domains() -> Tuple[str, ...]:
    return tuple(sorted({entry.value.domain for entry in EXPERIMENT_ADAPTERS.entries()}))


__all__ = ["AcquisitionPlan", "AdapterConformanceError", "CONTROL_KINDS", "CONTROL_SCHEMA",
           "ControlField", "ControlSchema", "DomainExperimentAdapter", "EXPERIMENT_ADAPTERS",
           "PLAN_SCHEMA", "SCHEMA", "SUPPORT_KINDS", "adapter_for", "adapter_for_domain",
           "plan_windows", "register_experiment_adapter", "registered_domains"]
