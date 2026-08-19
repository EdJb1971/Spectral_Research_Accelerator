"""Decorator-based registries (roadmap T3.5.15, defect D15, standard E1).

Everything pluggable is a registry, never an `if/elif` chain. Before this module, adding a
transform meant editing `engine.py` *and* `main.py`; adding a data source meant editing
`adapters.py` in three places. Those are not extension points, and the cost was visible:
`swt` (T3.5.7) and `dtcwt` (T3.5.6) each added two more branches, both landing with a
`TODO(T3.5.15)` comment because there was nowhere better to put them.

A registry entry is more than a function pointer. It carries a description, a parameter
schema and arbitrary capability metadata, which is what lets `GET /api/v1/actions` and
`GET /api/v1/transforms` be **generated** rather than hand-maintained - a hand-maintained
list is a document that goes stale, which is the same failure the documentation guard exists
to catch.

Two deliberate design choices:

*   **Duplicate names raise.** A silent overwrite makes behaviour depend on module import
    order, which is close to undiagnosable. Overriding requires the explicit `replace=True`.
*   **Unknown names raise `UnknownNameError`**, which lists the valid options and offers a
    `did you mean`. A registry's dominant failure mode is a typo, so that path deserves the
    best error in the codebase rather than a `KeyError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

from src.core.errors import DuplicateRegistrationError, UnknownNameError

T = TypeVar("T")


@dataclass
class Entry(Generic[T]):
    """One registered implementation plus everything needed to describe it in an API."""

    name: str
    value: T
    description: str = ""
    params: Dict[str, Any] = dc_field(default_factory=dict)
    capabilities: Dict[str, Any] = dc_field(default_factory=dict)
    tags: List[str] = dc_field(default_factory=list)
    defined_in: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "capabilities": self.capabilities,
            "tags": list(self.tags),
            "defined_in": self.defined_in,
        }


class Registry(Generic[T]):
    """A named collection of pluggable implementations."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Entry[T]] = {}

    # ---------------------------------------------------------------- registration

    def register(
        self,
        name: str,
        description: str = "",
        params: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        replace: bool = False,
    ) -> Callable[[T], T]:
        """Decorator registering the decorated object under ``name``."""

        def decorator(value: T) -> T:
            if name in self._entries and not replace:
                raise DuplicateRegistrationError(
                    self.kind, name, self._entries[name].defined_in or "unknown")
            module = getattr(value, "__module__", "")
            self._entries[name] = Entry(
                name=name, value=value, description=description or
                (getattr(value, "__doc__", "") or "").strip().split("\n")[0],
                params=dict(params or {}), capabilities=dict(capabilities or {}),
                tags=list(tags or []), defined_in=module)
            return value

        return decorator

    def add(self, name: str, value: T, **kwargs: Any) -> T:
        """Non-decorator form, for registering something you did not define."""
        return self.register(name, **kwargs)(value)

    # ---------------------------------------------------------------- lookup

    def get(self, name: str) -> T:
        entry = self.entry(name)
        return entry.value

    def entry(self, name: str) -> Entry[T]:
        if name not in self._entries:
            raise UnknownNameError(self.kind, name, self.names())
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry[T]]:
        return iter(self._entries[k] for k in sorted(self._entries))

    def names(self) -> List[str]:
        return sorted(self._entries)

    def entries(self) -> List[Entry[T]]:
        return [self._entries[k] for k in sorted(self._entries)]

    def describe(self) -> List[Dict[str, Any]]:
        """JSON-serialisable inventory - the source for the discovery endpoints."""
        return [e.to_dict() for e in self.entries()]

    def with_capability(self, key: str, value: Any = True) -> List[Entry[T]]:
        """Entries declaring a capability, e.g. every source that can stream Zarr.

        Standard E2: a caller should ask *what a source can do* rather than hard-coding
        which sources exist.
        """
        return [e for e in self.entries() if e.capabilities.get(key) == value]

    def unregister(self, name: str) -> None:
        """Remove an entry. Exists for tests; production code should not need it."""
        self._entries.pop(name, None)


def snapshot(registry: "Registry[Any]") -> Dict[str, Entry[Any]]:
    """Copy a registry's contents, so a test can register temporarily and restore."""
    return dict(registry._entries)


def restore(registry: "Registry[Any]", state: Dict[str, Entry[Any]]) -> None:
    registry._entries.clear()
    registry._entries.update(state)
