"""The reanalysis experiment adapter (TG17.3, `ed-dev`).

The domain the analysis layer's assumptions were written against, and therefore the adapter
whose registration should be shortest: it breaks nothing, floors its lag advectively, and reads
a regular grid whose extent metadata does establish exact coverage. Everything below is either
its archive's addressing or its declaration; no structural mathematics is written here, because
none of it is specific to this domain.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.adapters.standardized_level_adapter import (BENCHMARK_BINDING,
                                                     build_standardized_level_adapter)
from src.core.builtin_domains import REANALYSIS, register_builtin_domains

from src.core.experiment_adapter import AcquisitionPlan, ControlField

# Eager and idempotent for defect D35's reason: an adapter whose domain is onboarded
# only if some other module happened to import first is an adapter whose refusals
# depend on import order.
register_builtin_domains()

ADAPTER_ID = "reanalysis.standardized-level"

#: ERA5's nominal hourly cadence and the per-day byte estimate TG17.1 priced the flagship at.
#: Held as named constants rather than inline literals so that the number a researcher is shown
#: and the number the cap is checked against are the same one.
ERA5_CADENCE_SECONDS = 3600.0
ERA5_BYTES_PER_DAY = 2_000_000

CONTROLS = (
    ControlField(name="variable", label="Variable", kind="enum",
                 help="Which ERA5 variable this observation reads. The native unit follows "
                      "from the variable and is not a separate choice.",
                 choices=("temperature",), default="temperature"),
    ControlField(name="level_hpa", label="Pressure level", kind="integer", units="hPa",
                 help="The pressure level to read. A level is part of what a result is indexed "
                      "by, so it is declared rather than defaulted silently.",
                 default=850, minimum=1, maximum=1000),
)


def plan(parameters: Mapping[str, Any],
         identity: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    return AcquisitionPlan(
        source_id="grid_crop:era5_0p25_1h_full37", source_version="CDS-era5",
        support_kind="regular_grid_extent", access="credentials_required",
        access_means="A Copernicus Climate Data Store account and an accepted licence.",
        coverage_exact=True, native_cadence_seconds=ERA5_CADENCE_SECONDS,
        estimated_bytes_per_day=ERA5_BYTES_PER_DAY,
        identity={"licence_scope": "source-declared",
                  "variable": parameters.get("variable"),
                  "level_hpa": parameters.get("level_hpa")})


def _fixture(parameters: Mapping[str, Any]):
    from src.benchmarks.structural_trajectory import known_answer_native
    return known_answer_native("reanalysis")


ADAPTER = build_standardized_level_adapter(
    declaration=REANALYSIS, adapter_id=ADAPTER_ID,
    accepted_semantics="air temperature anomaly", accepted_units="K",
    controls=CONTROLS, plan=plan, fixture_record=_fixture,
    live_refusal=("a binding this slice can materialise. The ERA5 acquisition path exists "
                  "(see the Acquire tab) but is not yet wired to the canonical translator; "
                  "that is TG17.6's orchestrator, not a missing credential."),
    domain_mathematics=("none beyond the shared standardized-level channel: this domain "
                        "breaks no inherited assumption, which is why it is the cheapest "
                        "adapter and why it proves least about the seam",))

__all__ = ["ADAPTER", "ADAPTER_ID", "CONTROLS", "ERA5_BYTES_PER_DAY", "ERA5_CADENCE_SECONDS",
           "plan"]
