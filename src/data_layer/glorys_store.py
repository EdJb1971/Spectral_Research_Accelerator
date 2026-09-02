"""GLORYS gridded-ocean store registration (roadmap TG12.1).

This module is the extension seam made concrete: it records no acquisition mathematics and
imports no Copernicus client.  It loads the checked-in metadata probes, cites the friendly ARCO
layout by digest, and registers one ordinary :class:`GriddedStore`.  Importing it performs no
network access.

The Copernicus service labels and physical layouts are easy to confuse.  The selected asset is
literally ``geoChunked.zarr`` and has time-deep ``(2081, 1, 16, 16)`` chunks; the rejected
``timeChunked.zarr`` asset has spatially huge ``(1, 1, 512, 2048)`` chunks.  Both observations are
preserved in ``data/store_probes`` so the friendly choice does not erase its alternative.
"""

from __future__ import annotations

from pathlib import Path

from src.data_layer.store_probe import load_probes
from src.data_layer.stores import (GRIDDED_STORES, ChunkFacts, GriddedStore,
                                   register_store)


STORE_NAME = "glorys_phy_my_0p083deg_p1d"
STORE_URI = (
    "https://s3.waw3-1.cloudferro.com/mdl-arco-geo-025/arco/"
    "GLOBAL_MULTIYEAR_PHY_001_030/"
    "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/geoChunked.zarr"
)
PROBE_DIGEST = "ccdb0625e7e8fb1d"

_PROBE_DIR = Path(__file__).resolve().parents[2] / "data" / "store_probes"
load_probes(str(_PROBE_DIR))

GLORYS_STORE = GriddedStore(
    name=STORE_NAME,
    uri=STORE_URI,
    domain="reanalysis",
    access="anonymous",
    display_name="GLORYS12V1 · 0.083° · daily · 50 ocean levels",
    provider="Copernicus Marine Service",
    product_family="GLORYS global ocean reanalysis",
    vertical_dim="elevation",
    grid=(2041, 4320),
    resolution_deg=1.0 / 12.0,
    cadence_hours=24.0,
    levels=50,
    variables_note=(
        "The live TG12.1 probe measured thetao. The product also publishes salinity, currents, "
        "sea level, mixed-layer, bottom-temperature and sea-ice fields; those variables were "
        "not used to price the registered crop."
    ),
    note=(
        "Daily global GLORYS12V1 physics reanalysis, 1993 onward. The selected time-deep ARCO "
        "layout measured 2.02x uncompressed amplification for the sealed 1993-1995, 20-degree "
        "New Zealand-scale, one-variable, one-elevation crop. It breaks no new analysis "
        "assumption and is therefore a source in the existing reanalysis domain, not evidence "
        "of a second domain. A crop crossing a 2081-frame boundary can cost more and must be "
        "inspected before acquisition."
    ),
    probe_digest=PROBE_DIGEST,
    chunks=ChunkFacts(
        megabytes_per_chunk=4.262,
        method="live inspection",
        measured_on="2026-08-28",
        shape=(2081, 1, 16, 16),
        dims=("time", "elevation", "latitude", "longitude"),
        regional_amplification=2.02,
        note=(
            "Cited probe ccdb0625e7e8fb1d. Its paired rejected layout is retained as probe "
            "f9a45764fcf52c2a at 72.52x amplification. Counts are exact chunk-grid arithmetic "
            "over coordinate positions; no ocean values were transferred."
        ),
    ),
    extra={
        "product_id": "GLOBAL_MULTIYEAR_PHY_001_030",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311",
        "asset": "geoChunked.zarr",
        "licence": "Copernicus Marine Service product terms; attribution required",
        "probe_scope": "metadata only",
        "rejected_probe_digest": "f9a45764fcf52c2a",
        "acquisition_defaults": {
            "variables": ["thetao"],
            "time_start": "1993-01-01",
            "time_end": "1995-12-31",
            "lat_min": -50.0,
            "lat_max": -30.0,
            "lon_min": 160.0,
            "lon_max": 180.0,
            "levels": [-0.49402499198913574],
            "n_levels_analysis": 4,
        },
    },
)

if STORE_NAME not in GRIDDED_STORES:
    register_store(GLORYS_STORE, tags=["ocean", "GLORYS", "TG12.1"])


__all__ = ["GLORYS_STORE", "PROBE_DIGEST", "STORE_NAME", "STORE_URI"]
