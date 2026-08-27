"""Domain-first acquisition catalogue (TG10.2).

This router does not invent another source registry.  It projects the registered domain,
gridded-store and channel-table contracts into the order a researcher needs them: choose a
domain, then see the acquisition shapes that domain can actually use.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT, refuse_bare_confidence
from src.core.builtin_domains import register_builtin_domains
from src.core.domain import DOMAIN_DECLARATIONS, declaration_for, refusals_for
from src.core.onboarding import audit_onboarding, is_onboarded
from src.data_layer.stores import (ACCESS_REQUIREMENTS, register_builtin_stores,
                                   stores_for_domain)
from src.data_layer.tabular_source import unsatisfiable_axes


router = APIRouter(prefix="/api/v1/acquisitions", tags=["acquisitions"])

ACQUISITION_SHAPES = {
    "grid_crop": "A bounded time, horizontal and optional vertical selection from a regular grid.",
    "profile_query": "A region, time window and vertical range returning irregular profiles.",
    "channel_table": "A local delimited file with one clock column and named value channels.",
}

# Eager registration keeps the response independent of route visitation order (D35).
REGISTERED_DOMAINS = register_builtin_domains()
REGISTERED_STORES = register_builtin_stores()


def _limits(name: str) -> Dict[str, Any]:
    declaration = declaration_for(name)
    return {
        "declaration": declaration.describe(),
        "refuses": [dict(item) for item in refusals_for(declaration)],
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
    }


def _grid_acquisitions(domain: str, limits: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for store in stores_for_domain(domain):
        rows.append({
            "id": "grid_crop:%s" % store.name,
            "name": store.name,
            "shape": "grid_crop",
            "available": True,
            "access": store.access,
            "access_means": ACCESS_REQUIREMENTS[store.access],
            "store": store.to_dict(),
            "domain_limits": limits,
        })
    return rows


def _channel_acquisition(domain: str, limits: Dict[str, Any]) -> Dict[str, Any]:
    declaration = declaration_for(domain)
    missing_axes = unsatisfiable_axes(declaration)
    onboarded = is_onboarded(domain)
    available = onboarded and not missing_axes
    if not onboarded:
        reason = ("This domain was assembled from separate registrations and has not passed "
                  "the atomic onboarding contract.")
    elif missing_axes:
        reason = ("A channel table cannot supply the declared axes: %s."
                  % ", ".join(missing_axes))
    else:
        reason = None
    return {
        "id": "channel_table:%s" % domain,
        "name": "Local channel table",
        "shape": "channel_table",
        "available": available,
        "access": "local",
        "access_means": "A local UTF-8 CSV, TSV or text file; no network access.",
        "unavailable_reason": reason,
        "domain_limits": limits,
    }


@router.get("")
async def list_acquisitions() -> Dict[str, Any]:
    """Every declared domain and the acquisitions it admits, generated from registries.

    Unavailable channel-table entries are retained with the exact structural reason.  Hiding
    them would make an absent acquisition look like an incomplete catalogue rather than an E14
    refusal.  A newly registered domain or store changes rows here; it never adds a UI tab.
    """
    domains: List[Dict[str, Any]] = []
    for name in DOMAIN_DECLARATIONS.names():
        declaration = declaration_for(name)
        limits = _limits(name)
        acquisitions = _grid_acquisitions(name, limits)
        acquisitions.append(_channel_acquisition(name, limits))
        domains.append({
            "name": name,
            "description": declaration.description,
            "licence": declaration.licence,
            "onboarding": audit_onboarding(name),
            "domain_limits": limits,
            "acquisitions": acquisitions,
        })
    return refuse_bare_confidence({
        "domains": domains,
        "shapes": ACQUISITION_SHAPES,
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
        "note": ("Choose a domain first. A catalogue entry is an available acquisition path, "
                 "not evidence that data was fetched or that a record came from that domain."),
    }, where="acquisitions")


__all__ = ["ACQUISITION_SHAPES", "router"]
