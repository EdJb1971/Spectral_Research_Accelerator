"""A fifth gridded store, added in this file and nowhere else (roadmap TG10.1).

This exists to make TG10.1's acceptance criterion executable rather than aspirational:

    "a fifth store registers from a file no core module knows about, declares a vertical
    axis that is not ERA5's, and reaches the catalogue route - with zero edits to
    `zarr_source.py` or `main.py`."

Nothing in `src/` imports this module. `test_stores.py` imports it and then checks that the
store is live in the registry *and* in the HTTP catalogue, and that the two core files are
byte-identical afterwards. It follows the method `test_registries.py` already uses for the
data-source and pipeline-action seams, because a claim about not editing files is checkable
and should be checked the same way each time.

It doubles as the worked example for anyone adding a store. Note what the entry is obliged
to say and could not say before TG10.1:

*   **which domain it belongs to.** This one says `reanalysis`, not a domain of its own. An
    ocean product on the same grid shape breaks none of the inherited assumptions, and rule
    R17 is explicit that a domain violating nothing is not a second domain. Registering it
    under a freshly invented domain name would be the exact false confidence R17 refuses -
    so the honest entry is a *source* for an existing domain.
*   **that its vertical axis is `depth`, not `level`.** Declared, never inferred (E14). This
    is the whole reason `CropSpec.vertical_dim` exists.
*   **that nobody has measured its chunking.** `method="not measured"` is an admission, and
    it is the honest one here: this store is a fixture URI that has never been opened.
    Defect D43 is what happens when an unmeasured store is treated as a known quantity, and
    TG10.3 is the slice that makes a probe a precondition rather than a courtesy.
"""

from __future__ import annotations

from src.data_layer.stores import ChunkFacts, GriddedStore, register_store

STORE_NAME = "example_ocean_depth"

EXAMPLE_STORE = GriddedStore(
    name=STORE_NAME,
    uri="gs://example-bucket/ocean/example-depth-levels.zarr",
    domain="reanalysis",
    access="anonymous",
    vertical_dim="depth",
    grid=(180, 360),
    resolution_deg=1.0,
    cadence_hours=24,
    levels=5,
    note=("A worked example of a store on a `depth` axis rather than a pressure axis. It is "
          "not a real archive and has never been opened; it exists so the plugin seam is "
          "exercised by something whose vertical axis ERA5's code path cannot assume."),
    chunks=ChunkFacts(
        megabytes_per_chunk=None,
        method="not measured",
        note=("Nothing has probed this URI. Saying so is the point: an entry that quoted a "
              "chunk size nobody measured would read exactly like one that was."),
    ),
    variables_note="Declares no variables; it is never opened.",
)

register_store(EXAMPLE_STORE)
