"""`WaveletBank` - a declarative sweep over families, scales and levels (roadmap T4B.2).

The roadmap's claim about this task is that a `wavelet_bank` block needs **no engine changes**,
because its entries are ordinary parameter-matrix entries. That is true, and this module keeps
it true by producing a plain `Dict[str, List]` that `expand_parameter_matrix` consumes
unmodified - including its 1,000-combination guard, which is checked here as well so the
refusal names the bank rather than arriving from deep inside the engine.

**One asymmetry, stated rather than papered over.** `families` and `scales` are genuine sweep
axes: each combination is a separate decomposition producing a different array. `orientations`
is **not**. A wavelet transform computes all of its orientations in a single pass, so sweeping
them would run the identical decomposition once per orientation and throw most of each result
away. Orientations therefore travel as a *selector* applied within each run. The roadmap lists
all three together; treating them identically would have quietly multiplied every sweep's cost
by six while producing the same coefficients each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UnknownNameError
from src.transform_engine.coefficient_field import BANK_FAMILIES

#: Mirrors the engine's own ceiling. Duplicated deliberately rather than imported: importing
#: `engine` here would close a cycle through `actions`, and the number is asserted equal to the
#: engine's in the tests, so the two cannot drift apart unnoticed.
MAX_COMBINATIONS = 1000


def bank_families() -> List[str]:
    """Registered transforms tagged `wavelet_bank`, intersected with what this module builds.

    The registry is the source of the list, so a new family joins a bank by being registered
    rather than by anyone editing a sweep definition - the claim `registry.py` makes in its own
    docstring. The intersection is the honest part: a transform can be tagged before
    `coefficient_field` knows how to arrange it into (scale, orientation) axes, and offering it
    in a bank at that point would fail at run time instead of at configuration time.
    """
    from src.transform_engine.registry import TRANSFORMS

    tagged = [entry.name for entry in TRANSFORMS.entries()
              if "wavelet_bank" in (entry.tags or [])]
    return [name for name in tagged if name in BANK_FAMILIES]


@dataclass(frozen=True)
class WaveletBank:
    """A declarative `{families, scales, orientations, levels_hpa}` block."""

    families: Tuple[str, ...] = ("swt",)
    scales: Tuple[int, ...] = (3,)
    orientations: Optional[Tuple[Any, ...]] = None
    levels_hpa: Optional[Tuple[float, ...]] = None
    config: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.families:
            raise InvalidParameterError(
                "wavelet_bank.families", [], "at least one family; available: %s"
                % ", ".join(bank_families()))
        available = bank_families()
        for family in self.families:
            if family not in available:
                raise UnknownNameError(
                    "wavelet bank family", family, available,
                    hint="fft, dct and hybrid are registered transforms but have no "
                         "(scale, orientation) factorisation, so they cannot form a bank.")
        if not self.scales:
            raise InvalidParameterError("wavelet_bank.scales", [], "at least one scale depth")
        for scale in self.scales:
            if not isinstance(scale, int) or scale < 1:
                raise InvalidParameterError(
                    "wavelet_bank.scales", scale,
                    "integers >= 1. A scale entry is the number of dyadic levels to compute, "
                    "not a physical wavelength; CoefficientField.scale_wavelength_bands() "
                    "converts levels to metres once a physical grid is attached")
        if self.levels_hpa is not None:
            if not self.levels_hpa:
                raise InvalidParameterError(
                    "wavelet_bank.levels_hpa", [],
                    "at least one pressure level, or omit the key entirely")
            for level in self.levels_hpa:
                if not (0 < float(level) <= 1100):
                    raise InvalidParameterError(
                        "wavelet_bank.levels_hpa", level,
                        "a pressure in hPa between 0 and 1100. Values outside that range are "
                        "almost always metres or Pa supplied by mistake")
        self.validate_size()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WaveletBank":
        """Build from the `wavelet_bank:` block of an experiment config."""
        if not isinstance(config, dict):
            raise InvalidParameterError("wavelet_bank", type(config).__name__,
                                        "a mapping with families/scales/orientations keys")
        known = {"families", "scales", "orientations", "levels_hpa", "config"}
        unknown = sorted(set(config) - known)
        if unknown:
            raise InvalidParameterError(
                "wavelet_bank", ", ".join(unknown),
                "only %s. An unrecognised key is silently ignored by a permissive parser, "
                "which is how a sweep ends up not sweeping what its author wrote"
                % ", ".join(sorted(known)))

        def tup(value: Any) -> Optional[Tuple[Any, ...]]:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return (value,)

        def as_level(value: Any) -> int:
            """Refuse a fractional scale rather than truncating it.

            `int(2.5)` is 2, so a config asking for 2.5 levels would silently run 2 and
            report 2 - a difference nobody would ever look for. There is no such thing as
            half a dyadic level.
            """
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise InvalidParameterError("wavelet_bank.scales", value, "an integer >= 1")
            try:
                number = float(value)
            except ValueError:
                raise InvalidParameterError("wavelet_bank.scales", value, "an integer >= 1")
            if number != int(number):
                raise InvalidParameterError(
                    "wavelet_bank.scales", value,
                    "a whole number of dyadic levels. There is no half level, and int() "
                    "would have truncated this to %d and reported it as what was asked for"
                    % int(number))
            return int(number)

        return cls(
            families=tuple(tup(config.get("families", ("swt",))) or ()),
            scales=tuple(as_level(s) for s in (tup(config.get("scales", (3,))) or ())),
            orientations=tup(config.get("orientations")),
            levels_hpa=(tuple(float(v) for v in tup(config["levels_hpa"]))
                        if config.get("levels_hpa") is not None else None),
            config=dict(config.get("config") or {}),
        )

    # ------------------------------------------------------------------ expansion

    def n_combinations(self) -> int:
        return (len(self.families) * len(self.scales)
                * (len(self.levels_hpa) if self.levels_hpa else 1))

    def validate_size(self, limit: int = MAX_COMBINATIONS) -> int:
        total = self.n_combinations()
        if total > limit:
            raise InvalidParameterError(
                "wavelet_bank", total,
                "at most %d combinations (%d families x %d scales x %d levels). Each "
                "combination is a full decomposition of every frame, so this is a compute "
                "budget, not a formality: raise it only after measuring one run"
                % (limit, len(self.families), len(self.scales),
                   len(self.levels_hpa) if self.levels_hpa else 1))
        return total

    def to_parameter_matrix(self) -> Dict[str, List[Any]]:
        """The plain matrix `expand_parameter_matrix` already understands.

        No engine change is needed because there is nothing here the engine has not handled
        since it was written: list-valued keys, expanded as a Cartesian product.
        """
        matrix: Dict[str, List[Any]] = {
            "wavelet_family": list(self.families),
            "levels": list(self.scales),
        }
        if self.levels_hpa:
            matrix["level_hpa"] = list(self.levels_hpa)
        return matrix

    def combinations(self) -> List[Dict[str, Any]]:
        """Every run this bank describes, expanded by the *engine's* own function.

        Calling `expand_parameter_matrix` rather than reimplementing the product is the point:
        a bank cannot expand differently from an ordinary parameter matrix, because it is one.
        """
        from src.experiment_engine.engine import DeclarativeExperimentEngine

        expanded = DeclarativeExperimentEngine.expand_parameter_matrix(
            self.to_parameter_matrix())
        for run in expanded:
            if self.orientations is not None:
                run["orientations"] = list(self.orientations)
            if self.config:
                run.update(self.config)
        return expanded

    # ------------------------------------------------------------------ provenance

    def summary(self) -> Dict[str, Any]:
        return {
            "type": "WaveletBank",
            "families": list(self.families),
            "scales": list(self.scales),
            "orientations": list(self.orientations) if self.orientations else None,
            "orientation_role": (
                "selector applied within each run, not a sweep axis: a wavelet transform "
                "computes every orientation in one pass, so sweeping them would repeat the "
                "identical decomposition once per orientation"),
            "levels_hpa": list(self.levels_hpa) if self.levels_hpa else None,
            "n_combinations": self.n_combinations(),
            "max_combinations": MAX_COMBINATIONS,
            "config": dict(self.config),
        }


def select_orientations(coefficient_field, orientations: Optional[Sequence[Any]]):
    """Apply a bank's orientation selector to a decomposed field.

    Kept as a function rather than a `decompose_sequence` argument so the full decomposition
    is always what gets computed and cached: the selector narrows the *view*, and a later run
    asking for a different subset does not have to redo the transform.
    """
    if orientations is None:
        return coefficient_field
    return coefficient_field.select(orientations=list(orientations))
