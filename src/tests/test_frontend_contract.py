"""The frontend/backend contract, checked mechanically (roadmap T3.5.22).

**Why this exists, and what it is not.** `npm run build` runs `tsc`, so the frontend's *internal*
types are checked. Nothing checks them against the **backend**: every API response is typed by
hand in `frontend/src/types/api.ts`, and the fields that matter most are typed
`Record<string, any>` precisely because their shape is nested and open-ended. A renamed route or a
restructured payload therefore compiles cleanly and fails in the browser.

That is not hypothetical. While wiring the hypothesis card, the UI was written to render
``statistics.correction`` as a string; it is an **object** (`{method, assumption, n_tests,
min_adjusted}`), so React would have thrown *"Objects are not valid as a React child"* at runtime.
`tsc` passed, `vite build` passed, and the test suite was silent. These tests close that gap:

*   every path `api.ts` fetches must be a route the app actually serves;
*   every nested key the UI reads out of an untyped payload must exist in the real response.

**What they still do not prove:** that anything *renders*. No browser is available in this
environment, so visual verification of the nine tabs remains outstanding and is recorded as
outstanding rather than implied by a green build.
"""

from __future__ import annotations

import io
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND = os.path.join(REPO_ROOT, "frontend", "src")


def _read(*parts: str) -> str:
    return io.open(os.path.join(FRONTEND, *parts), encoding="utf-8").read()


@pytest.fixture(scope="module")
def api_service() -> str:
    return _read("services", "api.ts")


@pytest.fixture(scope="module")
def app_source() -> str:
    return _read("App.tsx")


@pytest.fixture(scope="module")
def all_sources() -> str:
    """Every .tsx/.ts under frontend/src, concatenated.

    Reachability is a property of the *app*, not of one file: the export controls live in
    `components/ExportBar.tsx`, and a check that only read `App.tsx` would call them unreachable
    while a researcher was clicking them.
    """
    import glob

    chunks = []
    for path in sorted(glob.glob(os.path.join(FRONTEND, "**", "*.ts*"), recursive=True)):
        chunks.append(io.open(path, encoding="utf-8").read())
    return chr(10).join(chunks)


def _served_routes():
    from src.api.main import app

    return {r.path for r in app.routes if getattr(r, "path", "").startswith("/api")}


def _fetched_paths(api_service: str):
    """Every `${BASE_URL}/...` template the service fetches, as a concrete route path.

    Two kinds of interpolation appear and they mean different things. A whole segment,
    ``/experiments/${id}``, is a **path parameter** and matches FastAPI's ``{experiment_id}``.
    An interpolation glued onto a literal, ``/proposals${params ? '?' + params : ''}``, is a
    **query string** and is not part of the route at all. Treating the second as a path
    parameter is what made the first version of this parser report a route that does not
    exist - the failure was in the test, not in the frontend.
    """
    marker = "<INTERPOLATION>"
    out = set()
    for literal in re.findall(r"`([^`]*)`", api_service):
        if "${BASE_URL}" not in literal:
            continue
        raw = literal.split("${BASE_URL}", 1)[1]
        marked = re.sub(r"\$\{[^}]*\}", marker, raw).split("?")[0]
        segments = []
        for segment in marked.strip("/").split("/"):
            if not segment:
                continue
            if segment == marker:
                segments.append("*")                        # a path parameter
            elif marker in segment:
                literal_part = segment.replace(marker, "")  # a query string on a literal
                segments.append(literal_part or "*")
            else:
                segments.append(segment)
        out.add("/api/v1/" + "/".join(segments))
    return out


def _wildcard(route: str) -> str:
    return re.sub(r"\{[^}]*\}", "*", route)


# ======================================================== routes

def test_every_fetched_path_is_a_served_route(api_service):
    """A call to a route that does not exist compiles perfectly and 404s in the browser."""
    served = {_wildcard(r) for r in _served_routes()}
    fetched = _fetched_paths(api_service)
    assert fetched, "no BASE_URL fetches were found - the parser is broken, not the frontend"
    missing = sorted(p for p in fetched if p not in served)
    assert not missing, (
        "the frontend fetches these paths, which the API does not serve:\n  %s\nserved:\n  %s"
        % ("\n  ".join(missing), "\n  ".join(sorted(served))))


def test_the_new_endpoints_are_actually_consumed(api_service):
    """T3.5.22's whole point: these existed for several slices with no consumer."""
    for path in ("/health", "/benchmarks", "/data/sources", "/data/zarr/catalogue",
                 "/data/zarr/cached", "/data/zarr/inspect"):
        assert path in api_service, "%s is served but the frontend never calls it" % path
    assert "/acquisitions" in api_service


def test_acquisition_navigation_is_domain_driven_and_does_not_grow_tabs(app_source):
    view = _read("components", "AcquisitionView.tsx")
    nav_ids = re.findall(r"\{ id: '([^']+)', name:", app_source)
    assert nav_ids.count("acquire") == 1
    assert "channels" not in nav_ids and "era5" not in nav_ids
    assert "catalogue.domains.map" in view, "domains must come from the API, not a UI list"
    assert "domain.acquisitions.map" in view, "acquisitions must come from the domain row"
    assert "domainName={domain.name}" in view
    assert "acquisition?.shape === 'grid_crop'" in view


def test_navigation_follows_the_scientific_workflow_and_labels_the_grid_line(app_source):
    """TG11.0 replaces a numbered feature list with the workflow it supports."""
    assert "WORKFLOW_NAV" in app_source
    for section in ("Acquire", "Analyse", "Evidence", "Review", "Read", "Platform"):
        assert "section: '%s'" % section in app_source
    assert "Gridded field line" in app_source
    assert "Review surface arrives in TG11.5" in app_source
    assert 'aria-label="Scientific workflow"' in app_source
    assert re.search(r"name: '\d+\.", app_source) is None


def test_record_and_study_are_shell_owned_persistent_context(app_source):
    """A panel switch must not discard the record or study the researcher already chose."""
    acquisition = _read("components", "AcquisitionView.tsx")
    channels = _read("components", "ChannelRecords.tsx")
    findings = _read("components", "FindingsView.tsx")
    assert "selectedRecord" in app_source and "setSelectedRecord" in app_source
    assert "selectedStudyId" in app_source and "setSelectedStudyId" in app_source
    assert 'aria-label="Current research context"' in app_source
    assert "selectedRecord={selectedRecord}" in app_source
    assert "selectedStudyId={selectedStudyId}" in app_source
    assert "onSelectRecord={onSelectRecord}" in acquisition
    assert "file, timeColumn, supportParentPx: supports" in channels
    assert "setRecord(selectedRecord?.record ?? null)" in channels, (
        "clearing shell context must clear the panel")
    assert "interface ChannelRecordSelection" in _read("types", "api.ts")
    assert "onSelectStudy?.(row.study_id as string)" in findings


def test_every_era5_control_survived_the_consolidation():
    view = _read("components", "AcquisitionView.tsx")
    for call in ("zarrCatalogue", "zarrCached", "zarrProbes", "zarrProbe", "zarrInspect"):
        assert "apiService.%s" % call in view
    for field in ("variables", "time_start", "time_end", "lat_min", "lat_max",
                  "lon_min", "lon_max", "levels", "n_levels_analysis"):
        assert field in view
    assert "inspection.cli" in view, "materialisation guidance must remain reachable"


def test_every_api_method_is_reachable_from_the_ui(api_service, all_sources):
    """A service method nothing calls is a route that is still invisible to a researcher."""
    methods = set(re.findall(r"^  async (\w+)\(", api_service, re.M))
    assert methods, "no service methods parsed"
    unused = sorted(m for m in methods if ("apiService.%s" % m) not in all_sources)
    assert not unused, (
        "these api service methods are defined but never called from App.tsx, so the "
        "endpoints behind them remain unreachable in the UI: %s" % unused)


# ======================================================== payload shapes

def test_health_payload_has_the_fields_the_status_tab_reads(client):
    """Names read out of `Record<string, any>` are unchecked by tsc; they are checked here."""
    body = client.get("/api/v1/health").json()
    assert "torch_device" in body
    assert "database_url_scheme" in body
    for key in ("cpu_count", "torch_num_threads", "backends", "default_backend",
                "default_backend_rationale"):
        assert key in body["execution"], "execution.%s is read by the UI" % key
    for key in ("revision", "head", "pending", "up_to_date", "error"):
        assert key in body["schema_state"], "schema_state.%s is read by the UI" % key


def test_benchmark_payload_has_the_fields_the_table_reads(client):
    rows = client.get("/api/v1/benchmarks").json()
    assert rows
    for key in ("name", "kind", "gates", "is_null", "known_answer"):
        assert key in rows[0], "benchmark.%s is read by the UI" % key
    assert any(r["is_null"] for r in rows), (
        "the status tab reports how many benchmarks are nulls; if none are, that claim is "
        "vacuous and the false-positive floor has gone missing")


def test_data_source_payload_carries_the_observational_flag(client):
    """The UI labels a source SIMULATED from this flag. Getting it from `kind` was defect D-.

    A string comparison on `kind` reported any simulated source with a different label - a
    demo source, a future model-output source - as real observational data. The flag is
    declared by the source itself, so the UI must read the flag.
    """
    sources = client.get("/api/v1/data/sources").json()
    assert sources
    for source in sources:
        assert "capabilities" in source
        assert "observational" in source["capabilities"], (
            "%s does not declare `observational`, so the UI cannot tell whether its data is "
            "real" % source["name"])


def test_zarr_catalogue_payload_has_the_fields_the_form_reads(client):
    body = client.get("/api/v1/data/zarr/catalogue").json()
    for key in ("stores", "network_enabled", "network_env_var", "r13_minimum_crop"):
        assert key in body
    first = next(iter(body["stores"].values()))
    assert "note" in first, "the store picker shows the note; it must be present"
    assert body["r13_minimum_crop"]["4"] == 512


def test_training_readiness_payload_carries_every_scientific_caveat_the_ui_reads(client):
    body = client.get(
        "/api/v1/training/representations",
        params={"levels": 3, "wavelet": "db2", "height": 120, "width": 80},
    ).json()
    for key in ("contract", "selected_levels", "selected_wavelet",
                "selected_spatial_shape", "representations"):
        assert key in body
    by_name = {entry["name"]: entry for entry in body["representations"]}
    assert by_name["swt"]["status"] == "accepted"
    assert by_name["dtcwt"]["status"] == "accepted"
    for entry in body["representations"]:
        for key in ("label", "status", "batched", "autograd", "exact_inverse",
                    "coefficient_ratio", "shift_behavior", "directionality", "boundary",
                    "scientific_role", "limitations", "verified", "not_run"):
            assert key in entry, "%s.%s is rendered by TrainingReadiness" % (entry["name"], key)
    selected = by_name["swt"]["selected_configuration"]
    for key in ("coefficient_channels_per_input_channel", "valid_interior_halfwidth_by_level",
                "valid_interior_shape_by_level", "coarsest_scale_has_valid_interior",
                "implementation_policy"):
        assert key in selected, "swt.selected_configuration.%s is rendered" % key
    selected_dtcwt = by_name["dtcwt"]["selected_configuration"]
    for key in ("atlas_planes_per_input_channel",
                "valid_interior_halfwidth_parent_px_by_level",
                "valid_interior_native_shape_by_level", "coarsest_scale_has_valid_interior",
                "display_contract"):
        assert key in selected_dtcwt, "dtcwt.selected_configuration.%s is rendered" % key


def test_zarr_inspect_payload_has_the_nested_keys_the_ui_reads(client, tmp_path):
    """The report the crop tab renders, field by field, including the nested ones."""
    zs = pytest.importorskip("src.data_layer.zarr_source")
    xr = pytest.importorskip("xarray")
    np = pytest.importorskip("numpy")

    path = str(tmp_path / "tiny.zarr")
    dataset = xr.Dataset(
        {"temperature": (("time", "level", "latitude", "longitude"),
                         np.zeros((4, 2, 32, 32), dtype="float32"))},
        coords={"time": np.arange("2020-01-01", "2020-01-02",
                                  np.timedelta64(6, "h"), dtype="datetime64[ns]")[:4],
                "level": [850, 500], "latitude": np.linspace(20, -20, 32),
                "longitude": np.linspace(0, 40, 32)},
    )
    dataset.chunk({"time": 1, "level": 2, "latitude": 32, "longitude": 32}).to_zarr(
        path, mode="w", consolidated=True)

    body = client.post("/api/v1/data/zarr/inspect", json={
        "store": path, "variables": ["temperature"],
        "time_start": "2020-01-01", "time_end": "2020-01-01",
        "lat_min": -10.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 20.0,
        "levels": [850, 500], "n_levels_analysis": 4,
    }).json()

    for key in ("spec", "cached", "structure", "assessment", "geometry", "cli"):
        assert key in body
    for key in ("amplification", "chunk_hostile", "bytes_wanted",
                "bytes_fetched_estimate", "warning", "advice", "byte_basis", "selection"):
        assert key in body["assessment"], "assessment.%s is read by the UI" % key
    variable = body["structure"]["variables"]["temperature"]
    for key in ("shape", "chunks", "chunk_megabytes"):
        assert key in variable, "structure.variables[].%s is read by the UI" % key
    # A crop below the R13 floor reports `ok: False` with an `error` string, and the UI
    # renders exactly those two keys.
    assert body["geometry"]["ok"] is False
    assert isinstance(body["geometry"]["error"], str)
    assert isinstance(body["assessment"]["advice"], list)


def test_hypothesis_statistics_correction_is_an_object_not_a_string():
    """The runtime bug tsc could not see, pinned as a contract.

    `statistics.correction` is `{method, assumption, n_tests, min_adjusted}`. The hypothesis
    card renders `correction.method` and `correction.assumption`; rendering `correction`
    itself throws "Objects are not valid as a React child" in the browser while passing every
    build step. If this payload is ever flattened to a string, this test fails and the UI is
    updated with it.
    """
    from src.statistics import significance

    test = significance.correlation_test(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.1, 2.2, 2.9, 4.2, 5.1, 5.8, 7.3, 8.1])
    screening = significance.screen(
        [{"label": "a", "p_value": test["p_value"]}], method="benjamini_yekutieli")
    correction = screening["correction"]
    assert isinstance(correction, dict), (
        "the UI reads correction.method and correction.assumption; a string here would be "
        "rendered as an object and crash the card")
    assert "method" in correction and "assumption" in correction


def test_app_does_not_render_the_correction_object_directly(app_source):
    """A regression guard on the exact mistake, in the source the browser runs."""
    assert "{h.statistics.correction}" not in app_source, (
        "`statistics.correction` is an object; rendering it directly throws in React")
    assert "h.statistics.correction.method" in app_source


def test_hypothesis_card_shows_a_warning_when_uncorrected(app_source):
    """An effect size with no q-value must not be presented as though it were tested.

    This is defect D8 as a *presentation* problem: the card previously showed only
    "Confidence: 96.0%", which is what made nine noise correlations from a 9-run sweep read as
    nine discoveries.
    """
    assert "Effect size:" in app_source, (
        "the card must not label a bare |r| as 'Confidence'")
    assert "No multiplicity correction was reported" in app_source
    assert "h.q_value != null" in app_source


def test_every_tab_in_the_nav_has_a_body(app_source):
    """A nav entry with no matching panel is a button that does nothing."""
    ids = re.findall(r"\{ id: '(\w+)', name: '[^']*', icon: \w+", app_source)
    assert len(ids) >= 9, "expected at least nine research modules, found %s" % ids
    for tab_id in ids:
        assert "activeTab === '%s'" % tab_id in app_source, (
            "tab %r appears in the navigation but has no panel" % tab_id)


def test_browser_rendering_evidence_is_described_accurately():
    """The honest counterpart to every contract test above, updated as the facts changed.

    These tests check contracts, not pixels. For most of this project no browser was available,
    so the guard asserted that `roadmap.md` kept saying the UI had never been seen. On
    2026-08-20 the platform was started and the user confirmed the nine tabs render and work -
    so that sentence would now be false, and a guard that forced a false statement to stay would
    be worse than no guard.

    What replaced it is the distinction that still matters: **a user's report is not a captured
    artefact.** T3.5.0 asked for a screenshot per tab; none exists in this repository, so nobody
    can re-examine the rendering claim the way they can re-examine every other number in
    `VERIFICATION.md`. This test asserts the documents keep saying exactly that, and no more.
    """
    roadmap = io.open(os.path.join(REPO_ROOT, "roadmap.md"), encoding="utf-8").read()
    assert "confirmed working by the user" in roadmap or "confirmed the nine tabs" in roadmap, (
        "roadmap.md must record who confirmed the UI renders and when")
    assert "No screenshot per tab has been captured" in roadmap or (
        "no screenshot per tab exists" in roadmap.lower()), (
        "roadmap.md must keep stating that the rendering is attested rather than evidenced, "
        "for as long as no screenshots exist in the repository")

    screenshots = [
        name for name in os.listdir(REPO_ROOT)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    docs_dir = os.path.join(REPO_ROOT, "docs")
    if os.path.isdir(docs_dir):
        screenshots += [n for n in os.listdir(docs_dir)
                        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    assert not screenshots, (
        "screenshots exist (%s) - T3.5.0's evidence clause is now satisfiable, so update the "
        "documents to reference them and tighten this test accordingly" % screenshots)


# ======================================================== scientific integrity of the UI

def test_no_fabricated_results_remain(app_source):
    """The excision, asserted. Slice 13 deleted every browser-side fabrication path.

    Before this, an unreachable backend made the UI invent fields, transforms ("approximate
    reconstruction with tiny errors"), diagnostics, boundary analyses and experiment IDs. One
    badge in the header said "Offline Sandbox Mock Mode"; the individual results said nothing,
    so a spectral slope computed from `Math.random()` looked exactly like one computed from
    ERA5. That is the same class of defect the backend's `is_simulated` provenance exists to
    prevent, committed one layer up.
    """
    for banned in ("getMockDatasets", "runMockFieldGenerator", "applyMockPerturbation",
                   "Mock Generator", "Mock Diagnostics", "Mock transform",
                   "Offline Sandbox Mock Mode"):
        assert banned not in app_source, (
            "%r is still in App.tsx - the UI can still fabricate a result" % banned)
    assert "if (backendConnected)" not in app_source, (
        "a `backendConnected` branch means there is still a path that computes without the "
        "backend")


def test_offline_state_says_nothing_is_fabricated(app_source):
    assert "Backend unreachable" in app_source
    assert "No results are fabricated in its absence" in app_source


def test_no_unqualified_validation_claims_in_the_ui(app_source):
    """The transform tab showed two green ticks whatever the measured error was.

    "Mathematically rigorous floating point calculations" and "Verified perfect reconstruct
    limits" were unconditional - an unqualified validation claim of exactly the kind the
    project's own rules forbid in its documents, displayed to the researcher instead. The
    measurement is now judged against a stated tolerance, and on its first live run that
    change found defect D36: float32 at the HTTP boundary, 1.9e-07 where float64 gives 2.8e-16.
    """
    for banned in ("Mathematically rigorous floating point calculations",
                   "Verified perfect reconstruct limits"):
        assert banned not in app_source, "unconditional validation claim: %r" % banned
    assert "perfect reconstruction to double precision" in app_source
    assert "1e-9" in app_source, "the tolerance the claim is judged against must be stated"


def test_the_synthetic_forecast_is_seeded(app_source):
    """It used to be built with `Math.random()` - unseeded, and uniform despite the control
    being labelled StDev. Every diagnostic computed from it was irreproducible."""
    assert "Math.random() - 0.5" not in app_source
    assert "forecastSeed" in app_source
    assert "apiService.perturbField" in app_source


def test_dataset_simulated_flag_is_shown_where_the_data_is_used(app_source):
    """`is_simulated` was fetched and displayed nowhere on the tab that uses the data."""
    assert "SIMULATED DATA - not an observation" in app_source
    assert "chosen.fallback_reason" in app_source


def test_regional_forecast_ui_refuses_unverified_physical_time_claims(all_sources):
    assert "cadence: NOT VERIFIED" in all_sources or "cadence_verified" in all_sources
    assert "physical lead labels: NOT AVAILABLE" in all_sources
    assert "ratios (dates not frozen)" in all_sources


def test_units_and_spectral_convention_are_displayed(app_source):
    """R15: the same field has different exponents under E(k) and S(k)."""
    for key in ("k_units", "power_units", "convention", "convention_note"):
        assert "spectral_diagnostics.%s" % key in app_source, (
            "%s is returned by the backend and not displayed" % key)


def test_spectral_slope_is_shown_with_its_uncertainty(app_source):
    """A slope with no standard error cannot be compared against -5/3 or -3."""
    assert "slope_standard_error" in app_source


def test_no_literal_latex_is_rendered(app_source):
    """`$R^2$` in JSX renders as dollar signs, not mathematics."""
    import re as _re

    literals = _re.findall(r"\$[^$\n{]{1,20}\$", app_source)
    assert not literals, "literal LaTeX would render as raw text: %s" % literals[:5]


def test_every_export_format_is_reachable_from_the_ui():
    """All four data formats plus PNG and SVG have a control."""
    bar = _read("components", "ExportBar.tsx")
    for fmt in ("csv", "json", "netcdf", "zarr"):
        assert "'%s'" % fmt in bar, "%s has no export control" % fmt
    figure = _read("components", "FigureExport.tsx")
    assert "'png'" in figure and "'svg'" in figure


def test_exports_carry_provenance_from_the_ui(app_source):
    """The UI must pass a provenance record, or the backend has nothing to embed."""
    assert "fieldProvenance" in app_source
    assert "datasetProvenance" in app_source
    assert "is_simulated" in app_source


# ======================================================== reachability and import

def test_no_served_route_is_unreachable_from_the_ui(api_service):
    """Every endpoint must be usable by a researcher, not only by a script.

    Four routes were served and uncallable from the UI after T3.5.22: the benchmark *runner*
    (the gates could be seen but not run), both registries (capability discovery), and the
    stored-proposals listing. An endpoint nobody can reach is a capability the platform does
    not really have.
    """
    served = {_wildcard(r) for r in _served_routes()}
    fetched = _fetched_paths(api_service)
    # Routes exempt by design, with the reason recorded so an exemption cannot be silent.
    exempt = {
        "/api/v1/hypothesis/proposals":
            "the discovery call returns the same records; a separate listing adds no capability",
    }
    unreachable = sorted(r for r in served if r not in fetched and r not in exempt)
    assert not unreachable, (
        "these routes are served but unreachable from the UI: %s" % unreachable)


def test_import_is_wired_into_the_ui(all_sources):
    """Real data could previously arrive only as a file placed in data/ by hand."""
    assert "importInspect" in all_sources and "importField" in all_sources
    assert 'type="file"' in all_sources, "there must be an actual file input"
    assert "Load into field buffer" in all_sources


def test_import_pins_extra_dimensions_rather_than_guessing(all_sources):
    """An ERA5 file is (time, level, lat, lon); `[0, 0]` chosen silently is a wrong answer."""
    assert "needs_selection" in all_sources
    assert "extra_dims" in all_sources
    assert "recorded in the provenance" in all_sources


def test_imported_origin_is_reported_as_unknown_not_real(all_sources):
    """`is_simulated: null` must display as "unknown", never as observational."""
    assert "Origin unknown" in all_sources
    assert "makes no claim about whether this is real data" in all_sources


def test_regional_dataset_readiness_keeps_three_claims_separate(all_sources):
    """A suitable manifest is not evidence that preparation or ERA5 agreement ran."""
    assert "T5.2 structure eligible" in all_sources
    assert "Prepared dataset: NO" in all_sources
    assert "train-only normalisation verified: NO" in all_sources
    assert "independent ERA5 cross-check: NOT RUN" in all_sources
    assert "claim_boundary" in all_sources


def test_benchmarks_can_be_run_from_the_ui(all_sources):
    assert "runBenchmarks" in all_sources
    assert "Run the suite" in all_sources
    # The three outcomes must stay separate on screen too.
    assert "NOT YET RUNNABLE" in all_sources
    assert "null_failures" in all_sources


# ======================================================== the findings view (TG9.2)
#
# Phase G9's rule is that the client computes and formats no scientific number. These assert it
# over the source, because a rule enforced only by whoever writes the JSX is not enforced.

FINDINGS_COMPONENTS = ("components/FindingsView.tsx",)


def _findings_sources() -> str:
    return "\n".join(_read(name) for name in FINDINGS_COMPONENTS)


def _without_comments(source: str) -> str:
    """Strip block and line comments, so prose *about* the rule is not read as breaking it.

    The component's own docstring names `toFixed` in order to say it does not use one. A check
    that could not tell those apart would forbid documenting the constraint, which is the
    opposite of what is wanted.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def test_the_findings_view_formats_no_scientific_number():
    """TG9.2's acceptance criterion: R9 made mechanical on the frontend.

    R9 calls a bare confidence percentage a hard constraint on the frontend, not only on the
    mining code. The backend already refuses to serve one; this refuses to let a view build one.
    Together they mean a bare confidence is not withheld by discipline, it is unobtainable.
    """
    code = _without_comments(_findings_sources())
    assert "toFixed" not in code, "a findings component formatted a number itself"
    assert "toPrecision" not in code
    assert not re.search(r"\d\s*%", code), "a percent literal reached a findings component"


def test_the_findings_view_reads_no_claim_bearing_field_directly():
    """The stronger form: the six figures are never touched, only the assembled string is.

    Reading `figures.confidence` is how a well-meaning author reintroduces the failure - the
    number is right there and printing it looks harmless. So no member of `AssociationFigures`
    may be read at all; `figures_text` is the only permitted route to the association strength.
    """
    code = _without_comments(_findings_sources())
    for field in ("confidence", "base_rate", "lift", "surrogate_corrected_lift", "support"):
        assert ".%s" % field not in code, (
            "a findings component read figures.%s directly; only figures_text may be shown "
            "(R9)" % field)
    assert "figures_text" in code, "the assembled figure string must actually be rendered"


def test_the_findings_view_never_shows_a_claim_without_its_bound():
    """`rendered` and `licences` are one unit; showing the first alone is promotion by layout."""
    code = _findings_sources()
    assert "unit.rendered" in code and "unit.licences" in code
    rendered_at = code.index("unit.rendered")
    licences_at = code.index("unit.licences")
    assert abs(licences_at - rendered_at) < 400, (
        "the entitlement must be rendered beside the claim, not in a distant branch")


def test_every_empty_findings_section_says_absence_is_not_evidence():
    """Silence reads as reassurance, so an empty section states what it does not mean."""
    code = _findings_sources()
    assert code.count("that is not the same as none existing") >= 3


def test_the_findings_view_is_operable_without_a_mouse(all_sources):
    """TG9.4's condition for the new surface only.

    Accessibility across `frontend/src` is zero, measured, and `roadmap.md` records that as
    deliberate rather than overlooked. This does not change that measurement for the legacy
    workbench. It requires that the surface added by Phase G9 does not add to the debt.
    """
    code = _findings_sources()
    assert "role=\"tablist\"" in code and "role=\"tab\"" in code
    assert "aria-selected" in code
    assert "aria-label" in code
    assert "aria-pressed" in code
    assert "htmlFor" in code, "every control needs a label bound to it"
    assert "focus:ring" in code, "keyboard focus must be visible"
    assert "aria-hidden" in code, "decorative icons must be hidden from a screen reader"
    assert "FindingsView" in all_sources, "the view must be mounted in the app"


def test_commentary_never_shares_a_container_with_claim_text():
    """TG9.3: recorded argument is fenced off from what the record actually permits.

    R23 commentary is the output of a non-deterministic process that moved nothing. Rendering it
    in the same container as a claim would let a reader take a challenger's fluent assertion for
    part of the finding - which is the failure the whole review layer was fenced to prevent.

    Asserted structurally: `commentary` is rendered inside its own `<section>` carrying its own
    aria-label, and no `TranslationUnit` field is rendered inside that section.
    """
    code = _findings_sources()
    assert 'aria-label="Recorded commentary"' in code
    start = code.index('aria-label="Recorded commentary"')
    end = code.index("</section>", start)
    block = code[start:end]
    assert "finding.commentary" in block
    for field in ("unit.rendered", "unit.licences", "claimable", "not_claimable"):
        assert field not in block, (
            "claim text (%s) was rendered inside the commentary container" % field)
    assert "moved nothing above" in block, "the fence must be labelled, not merely present"


def test_the_findings_view_shows_what_the_selected_domain_refuses():
    """TG9.3's first acceptance criterion, on the frontend."""
    code = _findings_sources()
    assert "domain_limits" in code
    assert "refuses" in code
    assert "precedence_admissible" in code
    assert "unadmitted_reading" in code


def test_domain_limits_are_never_shown_without_the_attribution_caveat():
    """A bundle does not record its domain, so limits must never read as a check on the study.

    The caveat is rendered from the payload rather than written into the component, so the
    sentence a reader sees is the one the API vouched for.
    """
    code = _findings_sources()
    assert code.count("attribution_caveat") >= 2, (
        "every place a domain limit is shown must render the caveat beside it")
    assert "does not record which domain" not in code, (
        "the caveat must come from the payload, not be restated in the view where it could "
        "drift from what the API actually guarantees")


def test_an_unregistered_domain_declaration_is_reported_not_rendered_as_no_limits():
    code = _findings_sources()
    assert "That is not the same as it refusing nothing." in code


# ======================================================== domain records (TG8.4)


def _channel_view() -> str:
    return _read("components", "ChannelRecords.tsx")


def test_the_domain_records_tab_is_wired(app_source):
    """TG10.2 consolidates records into Acquire instead of adding an archive tab."""
    acquisition = _read("components", "AcquisitionView.tsx")
    assert "name: 'Acquire data'" in app_source
    assert "activeTab === 'acquire'" in app_source
    assert "12. Domain Records" not in app_source
    assert "<ChannelRecords" in acquisition
    assert "domainName={domain.name}" in acquisition


def test_the_domain_records_view_renders_the_caveat_the_api_vouched_for(all_sources):
    """Restating it in the component would let the two drift; the sentence must be rendered."""
    view = _without_comments(_channel_view())
    assert "domain_limits.attribution_caveat" in view
    assert "does not record which domain produced it" not in view, (
        "the caveat must be rendered from the payload, not restated in the component")


def test_the_domain_records_view_renders_refusals_rather_than_deciding_them(all_sources):
    view = _without_comments(_channel_view())
    # Every refusal shown comes from the payload: the per-domain admission reasons, the
    # inspection's own refusal, and the domain's declared limits.
    assert "row.refusals" in view
    assert "refused_because" in view
    assert "domain_limits.refuses" in view
    assert "refusal.consequence" in view


def test_the_domain_records_view_does_not_decide_a_clock_or_a_domain(all_sources):
    """Both choices are the researcher's, and both are rendered as controls."""
    view = _without_comments(_channel_view())
    assert "candidate_time_columns.map" in view, "the clock column must be chosen from candidates"
    assert 'name="channel-domain"' in view, "the domain must be chosen, not inferred"
    assert "disabled={!row.admits}" in view, "a refusing domain must not be selectable"


def test_the_domain_records_view_says_a_plot_is_not_an_analysis(all_sources):
    view = _without_comments(_channel_view())
    assert "record.preview_note" in view
    assert "not an analysis" not in view, (
        "the boundary must be the sentence the API served, not one written here")


def test_the_domain_records_view_reports_an_irregular_clock_as_declared(all_sources):
    """`cadence_seconds` is null for an irregular record and must not be rendered as a number."""
    view = _without_comments(_channel_view())
    assert "irregular (declared)" in view
    assert "cadence_seconds !== null" in view, (
        "a cadence must be shown only where the backend reported one")


def test_the_domain_records_view_reports_withheld_rows_rather_than_hiding_them(all_sources):
    view = _without_comments(_channel_view())
    assert "rows_withheld" in view
    assert "nothing was thinned" in view


def test_the_domain_records_view_formats_no_scientific_quantity(all_sources):
    """Phase G9's rule, applied to a tab G9 did not write.

    Row counts and raw data values are formatted here and neither is a claim. What must not
    appear is a computed statistic or a percentage — those come from the backend or not at all.
    """
    view = _without_comments(_channel_view())
    assert "toFixed" not in view
    assert "%" not in view.replace("100%", ""), "no percentage may be composed in this view"
    for field in ("confidence", "base_rate", "lift", "surrogate_corrected"):
        assert field not in view, "the records view must read no claim-bearing field: %s" % field
