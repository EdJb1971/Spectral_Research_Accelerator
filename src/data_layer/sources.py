"""Data source registry with an explicit, recorded fallback chain (T3.5.15, standard E2).

Before this module, `adapters.py` hard-coded three dataset ids in three separate places -
`_get_simulated_fallback`, `list_datasets`, and the loader - so adding a source meant editing
all three and hoping none was missed. Worse, the fallback from real data to simulated data
happened inside a `try/except` and was recorded only as a string on the adapter.

**A fallback is a scientific fact, not an implementation detail.** If a run was silently
served synthetic data because a NetCDF file was missing, every number it produced means
something different, and that must be visible in the lineage record rather than inferable
from a log line. `resolve()` therefore returns the whole attempt chain - what was tried, what
each one said, and which one won - and that record is what the adapter attaches to the
dataset.

Sources declare **capabilities** so callers can ask what a source can do rather than
hard-coding which sources exist: `SOURCES.with_capability("streaming")` is how the Zarr
adapter (T3.5.18) will be selected without anyone editing a dispatch chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from src.core.errors import AllSourcesFailedError, UnknownNameError
from src.core.registry import Registry

SOURCES: Registry["DataSourceSpec"] = Registry("data source")


@runtime_checkable
class DataSource(Protocol):
    """What a data source must provide.

    Kept a `Protocol` rather than a base class so a source can be any callable object,
    including one built by a factory or supplied by a plugin, without inheriting from us.
    """

    def can_serve(self, dataset_id: str) -> bool:
        """Cheap check - must not perform network or disk I/O."""

    def fetch(self, dataset_id: str, **kwargs: Any) -> Any:
        """Return the dataset, or raise with an explanatory message."""

    def why_not(self, dataset_id: str) -> str:
        """Optional: why `can_serve` returned False, for the provenance record.

        A source that merely declines leaves no failed attempt behind, so without this the
        record would say only "simulated data was used" - true but not actionable. With it,
        the researcher is told *"no NetCDF file at data/era5_reanalysis.nc"*, which names the
        fix. Sources that do not implement it get a generic reason.
        """


@dataclass
class DataSourceSpec:
    """A registered source plus the metadata needed to order and describe it."""

    can_serve: Callable[[str], bool]
    fetch: Callable[..., Any]
    why_not: Optional[Callable[[str], str]] = None
    #: Lower runs first. Real observational data should outrank simulated fallbacks.
    priority: int = 100
    #: True when this source fabricates data rather than reading it.
    is_simulated: bool = False
    kind: str = "unknown"


@dataclass
class Resolution:
    """The outcome of a fallback chain: the data, and the full record of how it was chosen."""

    dataset: Any
    source_name: str
    kind: str
    is_simulated: bool
    attempts: List[Dict[str, Any]] = dc_field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def used_fallback(self) -> bool:
        return len(self.attempts) > 1

    @property
    def fallback_reason(self) -> Optional[str]:
        """Why a non-preferred source served this, or None if the preferred one did.

        Populated in *two* cases, which is the point. A real source can fail (it appears in
        `attempts` with a reason), but it can also simply decline - `can_serve` returns False
        because no file is on disk - leaving no failed attempt behind at all. The researcher
        still received fabricated data either way, so both produce a reason.
        """
        failed = [a for a in self.attempts if not a.get("ok")]
        if failed:
            return failed[-1]["reason"]
        if self.is_simulated:
            return ("no real source could serve this dataset (none declared it available), "
                    "so the simulated source was used")
        return None

    def to_provenance(self) -> Dict[str, Any]:
        """The record that goes into lineage (standard E5)."""
        return {
            "source": self.source_name,
            "kind": self.kind,
            "is_simulated": self.is_simulated,
            "used_fallback": self.used_fallback,
            "attempts": self.attempts,
            "elapsed_s": self.elapsed_s,
            # Fires whenever simulated data was served, not only after a *failure*. A
            # real source that merely declines (`can_serve` returns False, e.g. no file on
            # disk) leaves no failed attempt behind, yet the researcher still received
            # fabricated data - which is the fact that matters.
            "warning": (
                "This result was served by a SIMULATED source and is NOT observational "
                "data. Sources tried: %s."
                % ", ".join(a["source"] for a in self.attempts)
                if self.is_simulated else None),
        }


def register_source(name: str, description: str = "", priority: int = 100,
                    is_simulated: bool = False, kind: str = "unknown",
                    capabilities: Optional[Dict] = None, tags=None):
    """Register a source. The decorated object must satisfy the `DataSource` protocol."""
    caps = dict(capabilities or {})
    caps.setdefault("simulated", is_simulated)
    caps.setdefault("kind", kind)

    def decorator(obj):
        spec = DataSourceSpec(
            can_serve=obj.can_serve, fetch=obj.fetch,
            why_not=getattr(obj, "why_not", None), priority=priority,
            is_simulated=is_simulated, kind=kind)
        SOURCES.register(name, description=description or
                         (getattr(obj, "__doc__", "") or "").strip().split("\n")[0],
                         capabilities=caps, tags=tags)(spec)
        return obj

    return decorator


def candidates_for(dataset_id: str) -> List[str]:
    """Names of every source that claims it can serve ``dataset_id``, in priority order."""
    usable = []
    for entry in SOURCES.entries():
        try:
            if entry.value.can_serve(dataset_id):
                usable.append(entry)
        except Exception:
            # `can_serve` is documented as cheap and side-effect free; a source that throws
            # there is excluded rather than allowed to break the whole chain.
            continue
    usable.sort(key=lambda e: (e.value.priority, e.name))
    return [e.name for e in usable]


def resolve(dataset_id: str, **kwargs: Any) -> Resolution:
    """Try each capable source in priority order; return the first success with its record.

    Raises :class:`AllSourcesFailedError` when every source fails, reporting what each one
    said - a single "could not load data" would discard exactly the information needed to
    fix any of them.
    """
    started = time.time()
    attempts: List[Dict[str, Any]] = []
    names = candidates_for(dataset_id)

    # Record the sources that *declined*, with their reason. A decline is as much a part of
    # the provenance as a failure: it is why a higher-priority source did not serve this.
    for entry in SOURCES.entries():
        if entry.name in names:
            continue
        try:
            if entry.value.can_serve(dataset_id):
                continue
        except Exception:
            pass
        reason = "declined"
        if entry.value.why_not is not None:
            try:
                reason = "declined: %s" % entry.value.why_not(dataset_id)
            except Exception:
                pass
        # A source that never claimed this dataset has not "declined" it, and recording it
        # as a decline would fill the provenance of every request with irrelevant reasons.
        # Any capability key ending in `dataset_ids` counts as a claim: the Zarr source
        # (T3.5.18) declares `crop_dataset_ids` rather than `dataset_ids`, because a crop
        # family is not a listable dataset - and with only the literal key checked here, it
        # was recorded as declining `era5_reanalysis`, a dataset it has never heard of.
        claimed = [
            value
            for key, values in entry.capabilities.items()
            if key.endswith("dataset_ids") and isinstance(values, (list, tuple))
            for value in values
        ]
        if claimed and dataset_id not in claimed:
            continue
        attempts.append({"source": entry.name, "kind": entry.value.kind, "ok": False,
                         "declined": True, "reason": reason})

    if not names:
        raise UnknownNameError(
            "dataset", dataset_id,
            sorted({n for e in SOURCES.entries()
                    for n in e.capabilities.get("dataset_ids", [])}) or SOURCES.names())

    for name in names:
        spec = SOURCES.get(name)
        try:
            data = spec.fetch(dataset_id, **kwargs)
        except Exception as exc:
            attempts.append({"source": name, "kind": spec.kind, "ok": False,
                             "reason": "%s: %s" % (type(exc).__name__, exc)})
            continue
        attempts.append({"source": name, "kind": spec.kind, "ok": True, "reason": "served"})
        return Resolution(dataset=data, source_name=name, kind=spec.kind,
                          is_simulated=spec.is_simulated, attempts=attempts,
                          elapsed_s=time.time() - started)

    raise AllSourcesFailedError(dataset_id, attempts)


def describe_sources() -> List[Dict[str, Any]]:
    out = []
    for entry in SOURCES.entries():
        d = entry.to_dict()
        d["priority"] = entry.value.priority
        d["is_simulated"] = entry.value.is_simulated
        d["kind"] = entry.value.kind
        out.append(d)
    return sorted(out, key=lambda d: (d["priority"], d["name"]))
