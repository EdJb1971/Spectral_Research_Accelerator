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
from src.core.domain import (DOMAIN_DECLARATIONS, KNOWN_VIOLATIONS, declaration_for,
                             refusals_for)
from src.core.onboarding import audit_onboarding, is_onboarded
from src.data_layer.stores import (ACCESS_REQUIREMENTS, register_builtin_stores,
                                   stores_for_domain)
from src.data_layer.tabular_source import unsatisfiable_axes
from src.data_layer.argo_source import register_argo_source
from src.data_layer.profiles import PROFILE_SOURCES
from src.data_layer.tess_source import register_tess_source
from src.data_layer.lightcurves import LIGHTCURVE_SOURCES


router = APIRouter(prefix="/api/v1/acquisitions", tags=["acquisitions"])

ACQUISITION_SHAPES = {
    "grid_crop": "A bounded time, horizontal and optional vertical selection from a regular grid.",
    "profile_query": "A region, time window and vertical range returning irregular profiles.",
    "lightcurve_query": "A per-target sequence of fluxes over one or more observational sectors.",
    "channel_table": "A local delimited file with one clock column and named value channels.",
}

# Eager registration keeps the response independent of route visitation order (D35).
REGISTERED_PROFILE_SOURCE = register_argo_source()
REGISTERED_LIGHTCURVE_SOURCE = register_tess_source()
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
            "label": store.display_name or store.name,
            "provider": store.provider or None,
            "product_family": store.product_family or None,
            "shape": "grid_crop",
            "available": True,
            "access": store.access,
            "access_means": ACCESS_REQUIREMENTS[store.access],
            "store": store.to_dict(),
            "domain_limits": limits,
        })
    return rows


def _profile_acquisitions(domain: str, limits: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in PROFILE_SOURCES:
        source = entry.value
        if getattr(source, "domain", None) != domain:
            continue
        rows.append({
            "id": "profile_query:%s" % getattr(source, "name", "unknown"),
            "name": getattr(source, "name", "unknown"),
            "shape": "profile_query", "available": True,
            "access": getattr(source, "access", "anonymous"),
            "access_means": ("Public network access through the official Argo GDAC view; "
                             "explicit server opt-in is required."),
            "profile_source": source.describe() if hasattr(source, "describe") else {}, "domain_limits": limits,
        })
    return rows


def _lightcurve_acquisitions(domain: str, limits: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in LIGHTCURVE_SOURCES:
        source = entry.value
        if source.domain != domain:
            continue
        rows.append({
            "id": "lightcurve_query:%s" % source.name,
            "name": source.name,
            "shape": "lightcurve_query", "available": True,
            "access": source.access,
            "access_means": ("Public network access through MAST/AWS; "
                             "explicit server opt-in is required."),
            "lightcurve_source": source.describe(), "domain_limits": limits,
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
        acquisitions.extend(_profile_acquisitions(name, limits))
        acquisitions.extend(_lightcurve_acquisitions(name, limits))
        acquisitions.append(_channel_acquisition(name, limits))
        domains.append({
            "name": name,
            "description": declaration.description,
            "licence": declaration.licence,
            "onboarding": audit_onboarding(name),
            "domain_limits": limits,
            "acquisitions": acquisitions,
        })
    coverage: Dict[str, List[Dict[str, str]]] = {name: [] for name in KNOWN_VIOLATIONS}
    for domain in domains:
        declared = declaration_for(domain["name"])
        paths = [item for item in domain["acquisitions"] if item["available"]]
        for violation in declared.violations:
            coverage[violation].extend({"domain": domain["name"], "shape": item["shape"],
                                        "path": item["id"]} for item in paths)
    return refuse_bare_confidence({
        "domains": domains,
        "shapes": ACQUISITION_SHAPES,
        "operational_routes": [{
            "id": "era5_cds_regional",
            "domain": "reanalysis",
            "label": "ERA5 regional request · Copernicus CDS",
            "provider": "Copernicus Climate Data Store",
            "product_family": "ERA5 atmospheric reanalysis",
            "ui_status": "PLANNER_AVAILABLE",
            "execution": "browser plan; bounded resumable CLI acquisition",
            "configuration": ["variables", "date range", "UTC hours", "latitude/longitude",
                              "pressure levels", "grid spacing", "analysis depth",
                              "download/cache directories", "time chunk"],
            "reason": ("The browser validates, hashes, shards and prices the exact request "
                       "without network use. Durable browser execution and progress are not "
                       "mounted yet; acquisition remains available through the resumable CLI."),
        }],
        "violation_coverage": coverage,
        "attribution_caveat": DOMAIN_ATTRIBUTION_CAVEAT,
        "note": ("Choose a domain first. A catalogue entry is an available acquisition path, "
                 "not evidence that data was fetched or that a record came from that domain."),
    }, where="acquisitions")


__all__ = ["ACQUISITION_SHAPES", "router"]
