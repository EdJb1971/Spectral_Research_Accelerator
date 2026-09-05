"""Registered domain experiment adapters (TG17.3, `ed-dev`).

`register_builtin_adapters` is eager and idempotent for the reason `register_builtin_domains` is
(defect D35): a registry whose contents depend on which handler happened to run first is a
registry whose refusals can be missed, and an adapter that appears only after a researcher
visits the right tab is an adapter whose declaration nobody was held to.

Only reanalysis and the bespoke record family are built in here. Argo and TESS register from
`extensions/`, beside the domain declarations they already own — the same seam a third party
would use, exercised by this programme's own adapters rather than demonstrated separately.
"""

from __future__ import annotations

from typing import Tuple

from src.core.experiment_adapter import EXPERIMENT_ADAPTERS, register_experiment_adapter


def register_builtin_adapters() -> Tuple[str, ...]:
    """Register every adapter that ships inside `src/`, once, and return their ids."""
    from src.adapters import bespoke_record, reanalysis

    registered = []
    for module in (reanalysis, bespoke_record):
        adapter = module.ADAPTER
        if adapter.adapter_id not in EXPERIMENT_ADAPTERS:
            register_experiment_adapter(adapter)
        registered.append(adapter.adapter_id)
    return tuple(registered)


def register_extension_adapters() -> Tuple[str, ...]:
    """Register the adapters declared outside `src/`, through the supported seam."""
    from extensions import argo_float, tess_lightcurve

    registered = []
    for module in (argo_float, tess_lightcurve):
        adapter = module.register_adapter()
        registered.append(adapter.adapter_id)
    return tuple(registered)


def register_all_adapters() -> Tuple[str, ...]:
    return register_builtin_adapters() + register_extension_adapters()


__all__ = ["register_all_adapters", "register_builtin_adapters", "register_extension_adapters"]
