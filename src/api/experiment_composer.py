"""TG17.1 HTTP surface for composing and preflighting one scientific manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from src.adapters import register_all_adapters
from src.benchmarks.structural_trajectory import known_answer_native, known_answer_preview
from src.core.adapter_conformance import ConformanceCase, run_conformance
from src.core.errors import SpectralEarthError
from src.core.experiment_adapter import EXPERIMENT_ADAPTERS, adapter_for, adapter_for_domain
from src.core.experiment_family import family_expansion
from src.core.structural_alignment import (ALIGNMENT_KERNELS, MODE_FORBIDS, MODE_RELATIONSHIPS,
                                           alignment_report, bind_kernel, support_profile)
from src.core.structural_nulls import NULL_FAMILIES, NULL_MODES
from src.core.composer_path import (compose_state, describe_path, domain_menu,
                                    export_envelope, import_envelope,
                                    preregistration_summary, window_presets)
from src.core.experiment_manifest import (CrossDomainExperimentSpec, ManifestStore,
                                          flagship_recipe, manifest_envelope,
                                          manifest_sha256, preflight_manifest)
from src.core.experiment_run import RunStore, run_identity


router = APIRouter(prefix="/api/v1/experiment-composer", tags=["experiment-composer"])

#: Eager and idempotent (defect D35): the domain selector must not depend on which tab a
#: researcher opened first. A domain that appears only after some other route has run is a
#: domain whose declared refusals can be missed.
REGISTERED_ADAPTERS = register_all_adapters()


def _run_store(request: Request) -> RunStore:
    """The run journals, resolved exactly as `src.api.experiment_runs` resolves them.

    The composer never asks the caller for a run id. A run identity *is* the content address of
    its manifest, so "does this plan already have a run" is a question the plan answers itself -
    and a composer that took a run id could be pointed at a run executing a different plan.
    """
    root = getattr(request.app.state, "experiment_run_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_RUN_DIR", "data/experiment_runs")
    return RunStore(Path(root))


def _store(request: Request) -> ManifestStore:
    root = getattr(request.app.state, "experiment_manifest_dir", None)
    if root is None:
        root = os.getenv("EXPERIMENT_MANIFEST_DIR", "data/experiment_manifests")
    return ManifestStore(Path(root))


@router.get("")
async def composer_contract() -> Dict[str, Any]:
    return {
        "schema": "experiment-composer/v1",
        "modes": {
            "calendar_aligned": "Shared UTC support permits co-occurrence analysis only.",
            "scale_shape_aligned": "Normalized structural recurrence; no simultaneity or precedence.",
        },
        "mode_forbids": dict(MODE_FORBIDS),
        "mode_relationships": {name: list(group)
                               for name, group in sorted(MODE_RELATIONSHIPS.items())},
        "duration_presets": ["week", "three_months", "six_months"],
        "domains": [{"domain": entry.value.domain, "adapter_id": entry.value.adapter_id,
                     "label": entry.value.declaration.description}
                    for entry in EXPERIMENT_ADAPTERS.entries()],
        # TG17.7: the workflow is no longer a list of words maintained here. It is the
        # registered path, so this contract and the guided UI cannot describe different studies.
        "workflow": [step["step_id"] for step in describe_path()["steps"]],
        "path_route": "/api/v1/experiment-composer/path",
        "available_now": ["manifest", "saved draft", "metadata-only preflight",
                          "StructuralTrajectory contract and known-answer preview",
                          "registered domain adapters, their controls and conformance",
                          "declared alignment kernels and the support they would share",
                          "the complete declared family, priced before acquisition",
                          "declared null families and which domains admit each one",
                          "the guided path, its single next legitimate action and the ladder "
                          "separating acquired material, an executed run, a finding and evidence",
                          "server-resolved duration presets, a plain-language preregistration "
                          "summary, and machine-readable manifest export and import"],
        "not_yet_available": ["live adapter translation", "acquire quartet",
                              "live acquisition and translation workers"],
        "claim_boundary": "A ready preflight is not acquisition, analysis, evidence or a result.",
    }


@router.get("/recipes")
async def recipes() -> Dict[str, Any]:
    spec = flagship_recipe()
    envelope = manifest_envelope(spec)
    return {"recipes": [{"recipe_id": "g17-flagship-calendar", "title": spec.title,
                          "mode": spec.mode, "domains": [item.domain for item in spec.observations],
                          "manifest_sha256": envelope["manifest_sha256"]}],
            "note": "Recipes are complete manifests, not code generators or hidden defaults."}


@router.get("/recipes/g17-flagship-calendar")
async def flagship() -> Dict[str, Any]:
    return {"recipe_id": "g17-flagship-calendar", **manifest_envelope(flagship_recipe())}


@router.get("/adapters")
async def adapters() -> Dict[str, Any]:
    """Every registered domain adapter and its typed controls, generated from the registry.

    The Composer renders this rather than a hardcoded form per domain. A fifth adapter reaches
    the selector by registering, and this route does not learn its name.
    """
    return {"schema": "experiment-composer-adapters/v1",
            "adapters": [entry.value.describe() for entry in EXPERIMENT_ADAPTERS.entries()],
            "note": ("Each row carries the whole adapter description, including its domain "
                     "declaration and refusals. There is deliberately no per-adapter route: an "
                     "endpoint nothing reaches is an untested contract."),
            "claim_boundary": ("A registered adapter is a declared translation contract. It is "
                               "not evidence that its archive was read or that its declaration "
                               "describes that archive correctly.")}


@router.post("/adapters/{adapter_id}/conformance")
async def adapter_conformance(adapter_id: str,
                              parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the conformance kit against this adapter's deterministic known-answer record.

    Visible rather than a build-time check: an adapter author, and a reader of a result, can both
    see which declared invariances were executed and which are merely declared.
    """
    try:
        item = adapter_for(adapter_id)
    except SpectralEarthError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        record = known_answer_native(item.domain)
    except KeyError as error:
        raise HTTPException(status_code=422, detail=(
            "no deterministic known-answer record is registered for domain %r, so this adapter "
            "cannot be conformance-tested without acquiring data" % item.domain)) from error
    from datetime import datetime, timezone

    class _Window:
        name = "conformance_week"
        start_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_utc = datetime(2026, 1, 8, tzinfo=timezone.utc)

    report = run_conformance(item, ConformanceCase(
        parameters=dict(parameters or {}), windows=[_Window()],
        maximum_planned_bytes=4 * 1024 ** 3, native_record=record))
    return {**report.describe(), "record_kind": "deterministic_known_answer_not_acquired_data"}


@router.get("/alignment-kernels")
async def alignment_kernels() -> Dict[str, Any]:
    """Every declared alignment kernel, and which adapters admit each one (TG17.4).

    Generated from the kernel registry and the adapter registry rather than listed, so a kernel
    no adapter admits is visibly unusable instead of appearing as an available option.
    """
    admitted = {}
    for entry in EXPERIMENT_ADAPTERS.entries():
        for name in entry.value.admissible_kernels:
            admitted.setdefault(name, []).append(entry.value.adapter_id)
    rows = []
    for entry in ALIGNMENT_KERNELS.entries():
        rows.append({**entry.value.describe(),
                     "admitted_by": sorted(admitted.get(entry.name, [])),
                     "usable_across_all_registered_domains":
                         len(admitted.get(entry.name, [])) == len(EXPERIMENT_ADAPTERS)})
    return {"schema": "experiment-composer-alignment-kernels/v1", "kernels": rows,
            "default": "exact_support_overlap",
            "note": ("Only `exact_support_overlap` runs without being named in the manifest. "
                     "Every other kernel widens, snaps or carries support, must be admitted by "
                     "every participating adapter, and reports in seconds how much of the "
                     "resulting overlap it created rather than observed."),
            "claim_boundary": ("A kernel is a declared operation on support. Admitting one is "
                               "not evidence that applying it is appropriate for a question.")}


@router.get("/null-families")
async def null_families() -> Dict[str, Any]:
    """Every declared null family, its mode, what it preserves, and who admits it (TG17.5).

    Generated from the null registry and the adapter registry. A family no adapter admits is
    visibly unusable rather than absent, and `global_value_shuffle` appears with its refusal
    stated: an operation every domain can execute is exactly the one that needs to be refusable
    by name rather than quietly missing.
    """
    admitted: Dict[str, Any] = {}
    for entry in EXPERIMENT_ADAPTERS.entries():
        for name in entry.value.admissible_nulls:
            admitted.setdefault(name, []).append(entry.value.adapter_id)
    rows = []
    for entry in NULL_FAMILIES.entries():
        rows.append({**entry.value.describe(),
                     "admitted_by": sorted(admitted.get(entry.name, [])),
                     "usable_across_all_registered_domains":
                         len(admitted.get(entry.name, [])) == len(EXPERIMENT_ADAPTERS)})
    return {"schema": "experiment-composer-null-families/v1", "families": rows,
            "modes": list(NULL_MODES),
            "note": ("A null belongs to a comparison mode. A clock shift is no null for a "
                     "question that never referred to a clock, and a partner reassignment is "
                     "no null for a question about a shared calendar interval."),
            "claim_boundary": ("A null describes what a surrogate keeps and what it breaks. "
                               "Calibrating against one is not evidence that it is the right "
                               "null for a question.")}


@router.post("/manifests/family")
async def family(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """The complete declared search, priced in human terms before anything is acquired.

    Separate from `/manifests/preflight` on purpose. Preflight answers "can these archives be
    read"; this answers "can this study reject anything, and what does one more domain cost" —
    a question whose only useful answer arrives while the reply is still "declare something
    else".
    """
    try:
        return family_expansion(spec)
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/manifests/alignment")
async def alignment(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """The support this manifest's domains actually share, over the known-answer records.

    Deliberately not the metadata plan: `POST /manifests/preflight` already reports what the
    catalogue implies, and for three of the four flagship domains that is honestly "bounded by
    the window". This route opens the deterministic benchmark records instead, so the coverage a
    researcher sees before the freeze is measured support rather than a product description. The
    binding is stated in every response and travels with anything derived from it.
    """
    profiles, extent = [], 0.0
    for observation in spec.observations:
        try:
            item = adapter_for_domain(observation.domain)
            record = known_answer_native(item.domain)
        except (SpectralEarthError, KeyError) as error:
            raise HTTPException(status_code=422, detail=(
                "domain %r has no registered adapter with a deterministic known-answer record, "
                "so its support cannot be shown without acquiring data: %s"
                % (observation.domain, error))) from error
        extent = max(extent, float(record.sample_times_seconds[-1])
                     + float(record.native_scale_seconds))
        profiles.append((observation.domain, record))
    window = (0.0, extent)
    try:
        kernel = bind_kernel(
            spec.alignment.kernel, spec.alignment.parameters,
            declarations=[adapter_for_domain(domain).declaration for domain, _ in profiles],
            admissible_by_adapter={adapter_for_domain(domain).adapter_id:
                                   adapter_for_domain(domain).admissible_kernels
                                   for domain, _ in profiles})
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    built = [support_profile(
        label=domain, domain=domain, starts=record.sample_times_seconds,
        ends=record.sample_times_seconds + record.native_scale_seconds, window=window,
        native_scale_seconds=float(record.native_scale_seconds),
        valid=record.valid_mask) for domain, record in profiles]
    report = alignment_report(built, mode=spec.mode, kernel=kernel,
                              minimum_overlap_seconds=spec.alignment.minimum_overlap_seconds,
                              minimum_effective_samples=spec.alignment.minimum_effective_samples,
                              relationships=list(spec.family.relationships))
    return {**report, "manifest_sha256": manifest_sha256(spec),
            "record_binding": "benchmark_known_answer",
            "record_kind": "deterministic_known_answer_not_acquired_data",
            "window_seconds": extent}


@router.post("/manifests/validate")
async def validate_manifest(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    return manifest_envelope(spec)


@router.post("/manifests/preflight")
async def preflight(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    return preflight_manifest(spec)


@router.post("/manifests/representation-preview")
async def representation_preview(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """Inspect TG17.2 on frozen fixtures; never imply that manifest sources were acquired."""
    required = {"reanalysis", "argo_float", "tess_lightcurve"}
    observations = {item.domain: item for item in spec.observations}
    if not required.issubset(observations):
        raise HTTPException(status_code=422, detail=(
            "known-answer preview requires declared reanalysis, Argo and TESS observations"))
    preview = known_answer_preview(spec.seeds.get("family", 20260830))
    preview["manifest_sha256"] = manifest_sha256(spec)
    preview["selected_observations"] = sorted(required)
    return preview


@router.put("/drafts/{draft_id}")
async def save_draft(draft_id: str, spec: CrossDomainExperimentSpec,
                     request: Request) -> Dict[str, Any]:
    try:
        return _store(request).save(draft_id, spec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/drafts/{draft_id}")
async def load_draft(draft_id: str, request: Request) -> Dict[str, Any]:
    try:
        spec = _store(request).load_draft(draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail="saved experiment draft not found")
    return {"draft_id": draft_id, **manifest_envelope(spec)}


@router.get("/manifests/{manifest_sha256}")
async def load_manifest(manifest_sha256: str, request: Request) -> Dict[str, Any]:
    try:
        spec = _store(request).load_manifest(manifest_sha256)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="immutable experiment manifest not found")
    return manifest_envelope(spec)


# ------------------------------------------------------ TG17.7 the guided path


@router.get("/path")
async def composer_path() -> Dict[str, Any]:
    """The seven steps, their questions and their claim boundaries, as served data.

    The browser renders this. It does not hold a copy of the workflow, because two copies of an
    order of operations is two orders of operations, and the one a researcher follows would not
    be the one the receipt records.
    """
    return describe_path()


@router.post("/path/state")
async def composer_path_state(spec: CrossDomainExperimentSpec,
                              request: Request) -> Dict[str, Any]:
    """Where this manifest stands on the path, and the single next legitimate action.

    Metadata only: this prices the family and reads the catalogue, opens no archive and reads no
    measurement value. If a run already exists at this manifest's content address its receipt is
    folded in, which is how a refreshed browser comes back to the same place in the same study.
    """
    run_receipt = None
    saved = False
    # Look, never open. `RunStore.open` publishes a frozen manifest and creates the run
    # directory, and opening a run is a deliberate act a researcher takes at step six. A read
    # of where the plan stands must not be the thing that starts it.
    store = _run_store(request)
    identity = run_identity(spec)
    if (store.root / "runs" / identity["run_id"] / "manifest.json").exists():
        try:
            run_receipt = store.load(identity["run_id"]).receipt()
        except (SpectralEarthError, OSError, ValueError):
            run_receipt = None
    try:
        _store(request).load_manifest(manifest_sha256(spec))
        saved = True
    except (FileNotFoundError, ValueError, OSError):
        saved = False
    try:
        return compose_state(spec, saved=saved, run=run_receipt)
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/window-presets")
async def composer_window_presets(anchor_utc: str,
                                  stride_seconds: int = 3600) -> Dict[str, Any]:
    """Resolve `week`, `three_months` and `six_months` against one anchor, on the server."""
    try:
        return window_presets(anchor_utc, stride_seconds=stride_seconds)
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/domain-menu")
async def composer_domain_menu(selected: str = "") -> Dict[str, Any]:
    """Every registered domain with the assumptions it breaks, none of them filtered out."""
    chosen = [name for name in selected.split(",") if name]
    return domain_menu(selected=chosen)


@router.post("/preregistration-summary")
async def composer_preregistration_summary(
        spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """The plan in sentences, generated from the bytes that are hashed."""
    try:
        return preregistration_summary(spec)
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/manifests/export")
async def composer_export(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """A machine-readable envelope of the canonical manifest and its digest."""
    return export_envelope(spec)


@router.post("/manifests/import")
async def composer_import(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept an exported envelope, refusing one whose body disagrees with its digest."""
    try:
        return manifest_envelope(import_envelope(payload))
    except SpectralEarthError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=(
            "this is not an exported experiment manifest envelope: %s" % error)) from error


__all__ = ["router"]
