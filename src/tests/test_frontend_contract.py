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

**What they still do not prove:** that anything *renders*. TG11.6 adds a separate rendered
keyboard inspection; these source checks remain useful because they fail on semantic regressions
without requiring a browser runtime.
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


def _strip_comments(source: str) -> str:
    """Source without its comments, for checks about code rather than about prose.

    A doc comment that explains *why there is no domain branch here* must name a domain to say
    so. Refusing the explanation would push the reasoning out of the file it belongs in, which
    costs more than the check gains.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"^\s*//.*$", "", without_blocks, flags=re.M)


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
    assert "name: 'Recorded review'" in app_source
    assert "Recorded argument; never claim permission" in app_source
    assert 'aria-label="Scientific workflow"' in app_source
    assert 'id="scientific-workflow-nav"' in app_source
    assert 'aria-controls="scientific-workflow-nav"' in app_source
    assert "aria-expanded={mobileNavOpen}" in app_source
    assert "setMobileNavOpen(false)" in app_source
    assert "workflow-nav-scrim" in app_source
    assert "event.key === 'Escape'" in app_source
    assert "event.key !== 'Tab'" in app_source
    assert "event.preventDefault(); last.focus()" in app_source
    assert "document.body.style.overflow = 'hidden'" in app_source
    assert "mobileNavTriggerRef.current?.focus()" in app_source
    assert re.search(r"name: '\d+\.", app_source) is None


def test_the_workflow_has_a_skip_link_and_moves_focus_when_the_workspace_changes(app_source):
    """TG11.6: a route change must be announced where the new work begins, not leave keyboard
    focus behind on a navigation control whose visible context has changed."""
    assert 'href="#workspace-main"' in app_source
    assert 'id="workspace-main"' in app_source
    assert 'aria-labelledby="workspace-heading"' in app_source
    assert "workspaceHeadingRef.current?.focus()" in app_source
    assert 'tabIndex={-1}' in app_source


def test_every_legacy_shell_label_is_programmatically_bound(app_source):
    """The old gridded panels used adjacent labels, which look labelled but have no accessible
    name. Their controls now use explicit id/htmlFor pairs; wrapper labels remain valid in the
    newer components and are checked by their own contracts."""
    labels = re.findall(r"<label\b([^>]*)>", app_source)
    assert labels and all("htmlFor=" in attrs for attrs in labels)
    ids = set(re.findall(r'\bid="([^"]+)"', app_source))
    targets = re.findall(r'htmlFor="([^"]+)"', app_source)
    assert not sorted(set(targets) - ids)


def test_global_keyboard_focus_and_reduced_motion_are_not_panel_options():
    css = _read("index.css")
    assert ":focus-visible" in css and "outline: 3px solid" in css
    assert ".skip-link:focus" in css
    assert "prefers-reduced-motion: reduce" in css
    # The canvas rule must exclude `sr-only` children.  Giving the workspace heading and the
    # live-status paragraph a real width and `margin-inline: auto` overrides the 1px clipped box
    # that makes them screen-reader-only; being absolutely positioned, they then escape the
    # workspace's clipping and widen the document by 14px at every viewport (TG18.1).
    assert ".workspace-main > *:not(.sr-only)" in css and "--workspace-max" in css
    assert ".workspace-main > * {" not in css
    # The metadata floor covers every legacy sub-11px step, not only 10 and 11.
    assert '.workspace-main [class~="text-[9px]"]' in css
    assert '.workspace-main [class~="text-[10px]"]' in css
    # Collapsing tracks without releasing `col-span-*` leaves an implicit column behind.
    assert "grid-column: auto" in css
    assert ".workspace-main .grid.grid-cols-2" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css
    assert ".workflow-nav-scrim" in css
    assert "position: sticky" in css and ".research-context" in css
    assert "min-height: 2.5rem" in css


def test_retry_and_lineage_nodes_are_keyboard_operable(app_source):
    """The only legacy click target that was not a native control was the connection retry;
    SVG provenance nodes require an explicit keyboard equivalent because SVG has no button."""
    lineage = _read("components", "LineageGraph.tsx")
    assert "Backend unreachable - no computation available; retry" in app_source
    assert re.search(r"<span[^>]+onClick=", app_source) is None
    for token in ('role="button"', "tabIndex={0}", "onKeyDown", "event.key === 'Enter'",
                  "event.key === ' '", "aria-pressed"):
        assert token in lineage


def test_visualisations_have_text_equivalents():
    heatmap = _read("components", "Heatmap2D.tsx")
    line = _read("components", "LineChart.tsx")
    for source in (heatmap, line):
        assert "<figure" in source and "aria-labelledby" in source
        assert "<figcaption" in source and 'className="sr-only"' in source
    assert "rows and" in heatmap and "Value units" in heatmap
    assert "series.map(item => item.name).join" in line
    # A caption describing the shape of the data is not an equivalent; both families must reach
    # the shared exact-value panel (TG18.2).
    for source in (heatmap, line):
        assert "FigureDataDisclosure" in source and "FigureContract" in source


def test_figure_data_panels_transcribe_without_authoring_a_statistic():
    """The text equivalent may restate what a figure encodes and name what it could not encode.
    It may not derive a new number: G18 is presentation only, and a statistic authored by a view
    is indistinguishable on screen from one the analysis layer stands behind."""
    figure_data = _read("components", "FigureData.tsx")
    heatmap = _read("components", "Heatmap2D.tsx")
    line = _read("components", "LineChart.tsx")
    export = _read("components", "FigureExport.tsx")

    # `<details>` groups carry an explicit accessible name; an unnamed one is announced only as
    # a disclosure triangle (the TG17.8 defect that recurred as D82).
    assert "aria-label={name}" in figure_data
    # Exact values are keyboard-addressable rather than hover-only, and the readout is live.
    assert 'aria-live="polite"' in figure_data and "<output" in figure_data
    assert 'type="number"' in figure_data
    assert "htmlFor={rowId}" in figure_data and "htmlFor={columnId}" in figure_data
    assert "CellInspector" in heatmap

    # Missingness and normalization provenance are stated, not inferred by the reader.
    assert "Missing samples" in heatmap and "not finite" in heatmap
    assert "supplied, shared across panels" in heatmap
    assert "derived from this panel alone" in heatmap
    # A log axis silently discards non-positive samples; the count is part of the contract.
    assert "non-positive" in line and "Points drawn" in line
    # Truncation is stated rather than silent.
    assert "MAX_TABULATED_POINTS" in line and "Showing the first" in line

    # The refusal, in both the prose and the absence of the statistics themselves.
    for source in (heatmap, line):
        assert "derives no summary statistic" in source
        assert "<FigureExport" in source, "every figure family must expose publication export"
    body = _strip_comments(figure_data + heatmap + line)
    for forbidden in ("'Mean'", '"Mean"', "'Median'", "'Std", "'Correlation'"):
        assert forbidden not in body

    # The publication form is a self-contained vector sheet with the same stated facts and
    # producer qualifications. It snapshots the live plot and never mutates or recomputes it.
    assert "Publication HTML" in export and "Plotly as any).toImage" in export
    assert "Figure reading contract" in export
    assert "Producer statements and qualifications" in export
    assert "performs no scientific analysis" in export
    assert "Plotly.relayout" not in export and "Plotly.react" not in export
    assert "facts: () => factsFor(scanValues())" in heatmap
    assert "facts: () => factsFor(scanOmissions())" in line
    assert "decision.statement.qualifiers" in line


def test_side_by_side_fields_carry_a_stated_comparison_contract(app_source):
    """Two heat maps side by side are an invitation to compare them, and Plotly autoscales each
    panel to its own extremes unless told otherwise -- so an inverse reconstruction that lost most
    of its amplitude rendered as a near-identical picture beside its original, with the difference
    surviving only in two small colour-bar ranges. `zRange` was supplied at exactly one call site
    in the whole frontend before TG18.2."""
    comparison = _read("components", "FigureComparison.tsx")

    # Comparability is decided, not assumed, and every branch carries a reason written for the
    # page rather than for a log.
    assert "buildComparisonContract" in comparison
    for refusal in ("different units", "different quantities",
                    "does not declare which quantity", "no finite sample"):
        assert refusal in comparison, refusal
    # Scale comparability and cell correspondence are separate claims: a pair can honestly have
    # one without the other.
    assert "linked" in comparison and "differ in shape" in comparison

    contracts = app_source.count("buildComparisonContract(")
    assert contracts, "no gridded pair declares a comparison contract"
    # Every contract that is built is also stated on the page and offered a shared address...
    assert app_source.count("<FigureComparisonNotice") == contracts
    assert app_source.count("useLinkedAddress(") == contracts
    # ...and reaches at least two panels, since a contract governing one panel governs nothing.
    assert app_source.count("zRange={") >= 2 * contracts


def test_figures_state_the_domain_a_claim_was_fitted_over(app_source):
    """A power spectrum is drawn across every wavenumber bin; the exponent quoted for it is not.

    The backend fits over ``[k_min, k_max]`` with ``n_points`` of those bins, and returns a
    standard error, an R-squared, a weighting scheme and an explicit ``assumptions`` list. Before
    TG18.2 the figure showed the whole curve, the exponent sat in a card below it, and the
    assumptions were returned by the API, typed in ``api.ts`` and rendered nowhere at all -- so
    the only available reading was that the exponent described the curve on screen.
    """
    validity = _read("components", "FigureValidity.tsx")
    line = _read("components", "LineChart.tsx")

    # Whether a declared band may honestly be drawn is decided, and every branch carries a reason
    # written for the page.
    assert "buildValidityContract" in validity
    for refusal in ("not both finite numbers", "encloses nothing", "lies entirely",
                    "was not produced"):
        assert refusal in validity, refusal
    # The sentence the whole overlay exists for, present on both marked branches: a band that
    # runs off the figure must not lose it, and that is the branch the platform's own spectra
    # take, since the fit reaches Nyquist and the plotted bins stop short.
    assert validity.count("drawn but were not used") == 2
    # A value quoted without an uncertainty is flagged, not silently rendered as exact.
    assert "no uncertainty supplied" in validity
    assert "cannot be compared against a reference value" in validity
    # Plotly reads shape coordinates on a log axis as log10 of the value, and a fitted spectrum
    # is read on log-log axes.
    assert "Math.log10" in validity
    # The band is clamped to what is drawn, so the mark on screen is the one the prose describes.
    assert "markedFrom" in validity and "markedTo" in validity

    # The claim language the analysis layer authored is carried, not paraphrased (R22, R23), and
    # its interpretation is deliberately not repeated beside the figure.
    assert "assumptions" in validity and "qualifiers" in validity
    body = _strip_comments(validity)
    assert "regime_interpretation" not in body

    # The refusal that keeps this an overlay rather than a second analysis: the view states the
    # fit, it does not draw it.
    assert "no uncertainty envelope is drawn around it" in validity
    for forbidden in ("Math.exp(", "Math.pow(", "** slope", "intercept_ln_c"):
        assert forbidden not in body, forbidden

    # The chart states the decision outside the disclosure, draws only the bands it accepted, and
    # gives the shading a text equivalent in the tabulated points.
    assert "FigureValidityNotice" in line
    assert "validityShapes(validityContract" in line
    assert "insideMarkedDomains" in line and "in declared domain" in line

    # Every figure that declares a domain also states it and offers the uncertainty beside it.
    declared = app_source.count("validity={")
    assert declared, "no figure declares a validity domain"
    assert app_source.count("uncertainties={") == declared


def test_declared_figure_pairs_share_a_bounded_accessible_resizer(app_source):
    """Pane width is presentation state, not a fourth comparison decision.

    Both figures must remain mounted, and the interaction must be operable without a pointer.
    Narrow-screen stacking is a reading-order requirement rather than a squeezed resizer.
    """
    resizer = _read("components", "ResizableFigurePair.tsx")
    css = _read("index.css")

    assert 'role="separator"' in resizer
    assert 'aria-orientation="vertical"' in resizer
    assert "aria-valuemin={25}" in resizer and "aria-valuemax={75}" in resizer
    for key in ("ArrowLeft", "ArrowRight", "Home", "End", "Enter"):
        assert key in resizer, key
    assert "setPointerCapture" in resizer
    assert "Pane width changes presentation only" in resizer
    assert "children[0]" in resizer and "children[1]" in resizer

    declared_pairs = app_source.count("<FigureComparisonNotice")
    assert app_source.count("<ResizableFigurePair") == declared_pairs
    assert "@media (max-width: 767px)" in css
    assert ".resizable-figure-pair__separator" in css and "display: none" in css


def test_async_workflow_surfaces_expose_busy_state(all_sources):
    for name in ("AcquisitionView", "DomainAnalysisView", "PreregistrationView", "EvidenceView",
                 "StructureMiningView", "CrossDomainRecordView", "FindingsView",
                 "GateRecordView"):
        source = _read("components", f"{name}.tsx")
        assert "aria-busy=" in source, name
    assert 'role="alert"' in all_sources and 'role="status"' in all_sources


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


def test_g17_composer_is_visible_manifest_driven_and_honest_about_the_runner():
    app = _read("App.tsx")
    view = _read("components", "ExperimentComposer.tsx")
    service = _read("services", "api.ts")
    assert "Experiment Composer" in app
    assert "<ExperimentComposer onSelectStudy={setSelectedStudyId}" in app
    assert "getFlagshipRecipe" in view and "preflightExperimentManifest(manifest)" in view
    assert "saveExperimentDraft(manifest.study_id, manifest)" in view
    assert "loadExperimentManifest(result.manifest_sha256)" in view
    assert "localStorage.getItem(SAVED_DRAFT_KEY)" in view
    assert "No network used and no measurement values opened." in view
    assert "/experiment-composer/manifests/preflight" in service


def test_g17_composer_exposes_honest_structural_contract_preview():
    view = _read("components", "ExperimentComposer.tsx")
    service = _read("services", "api.ts")
    types = _read("types", "api.ts")
    assert "Inspect structural contract" in view
    assert "Deterministic known-answer records, not acquired observations" in view
    assert "native_record.content_sha256" in view
    assert "adapter.definition_sha256" in view
    assert "/experiment-composer/manifests/representation-preview" in service
    assert "StructuralTrajectoryPreview" in types


def test_domain_analysis_uses_the_persistent_full_record_and_cannot_write(app_source, api_service):
    """TG11.1 must never substitute TG8.4's capped preview for the retained source file."""
    view = _read("components", "DomainAnalysisView.tsx")
    assert "name: 'Cross-domain analysis'" in app_source
    assert "activeTab === 'domainWorkbench'" in app_source
    assert "selectedRecord={selectedRecord}" in app_source
    assert "form.append('file', selection.file)" in api_service
    assert "/analysis/run" in api_service and "/analysis" in api_service
    assert "runDomainAnalysis" in view and "getDomainAnalysisCapabilities" in view
    assert "read_only" not in view, "the component must not fabricate the server's receipt"
    assert "response.claim_boundary" in view


def test_domain_analysis_exposes_r21_and_all_three_engine_operations():
    view = _read("components", "DomainAnalysisView.tsx")
    for operation in ("association", "precedence", "domain_gate"):
        assert operation in view
    assert "precedenceAdmissible" in view
    assert "disabled={refused}" in view
    assert "R21" in view
    assert "PASS" in view and "FAIL" in view and "INVALID" in view


def test_domain_analysis_checks_the_result_is_about_the_record_on_screen():
    """The re-read is the design; an unchecked re-read is an assumption that it read the same thing."""
    view = _read("components", "DomainAnalysisView.tsx")
    assert "response.source.content_sha256 !== selectedRecord.record.content_sha256" in view
    assert "response.source.frames !== selectedRecord.record.n_rows" in view


def test_preregistration_declares_a_family_and_never_a_digest_or_a_p_value(app_source,
                                                                            api_service):
    """TG11.2. The client declares; the server decides what that hashes to and what it means.

    A caller-supplied sealing time could be written after the partition was opened, and a
    caller-supplied p-value is a scientific number this line does not let a client compute.
    Both would make the seal a formality.
    """
    view = _read("components", "PreregistrationView.tsx")
    assert "name: 'Preregistration'" in app_source
    assert "activeTab === 'preregistration'" in app_source
    assert "sealFamily" in view and "describePartition" in view and "confirmSeal" in view
    assert "/preregistration/seal" in api_service
    assert "/preregistration/partition" in api_service
    assert "/confirm" in api_service
    assert "sealed_at" not in api_service.split("preregistration (TG11.2)")[1].split(
        "the findings surface")[0], "the client must not send a sealing time"
    assert "p_value" not in view.replace("p_values[index]", "")


def test_preregistration_confirmation_sends_the_record_and_nothing_else(api_service):
    """Every setting of a confirmatory sweep was frozen; a knob left turnable is a choice made
    after the declaration."""
    block = api_service.split("async confirmSeal")[1].split("},")[0]
    for tunable in ("lags", "n_surrogates", "alpha", "correction", "estimator", "bins",
                    "seed", "train_ratio", "domain", "time_column"):
        assert tunable not in block, "%r must come from the seal, not the caller" % tunable
    assert "form.append('file', file)" in block


def test_preregistration_says_a_seal_is_not_evidence_about_itself():
    """A local copy hashes to itself by construction; the published digest is the real check."""
    view = _read("components", "PreregistrationView.tsx")
    assert "publishedSha" in view
    assert "cannot rewrite" in view
    assert "checked_against_publication" in view
    assert "self-consistency only" in view


def test_preregistration_shows_a_spent_partition_as_gone_rather_than_as_a_failure():
    """Spent is not failed. A partition that has been tested is simply no longer available."""
    view = _read("components", "PreregistrationView.tsx")
    assert "row.spent" in view
    assert "will not be tested again" in view
    assert "already_opened" in view


def test_preregistration_does_not_present_a_receipt_as_a_claim():
    view = _read("components", "PreregistrationView.tsx")
    assert "records no evidence and moves no rung" in view
    assert "are not corrected for here and are" in view
    assert "read_only" not in view, "the component must not fabricate the server's receipt"


def test_the_evidence_panel_reaches_the_write_path_and_nothing_else(app_source, api_service):
    """TG11.3. The only writing panel in the application, and the only one that could break R22."""
    view = _read("components", "EvidenceView.tsx")
    assert "name: 'Evidence record'" in app_source
    assert "activeTab === 'evidence'" in app_source
    for call in ("openStudy", "getEvidenceHead", "appendEvidence", "appendPrecedence",
                 "getEvidenceCapabilities"):
        assert "apiService.%s" % call in view
    assert "/evidence/studies" in api_service


def test_no_request_this_client_can_send_carries_a_rung(api_service):
    """R22, checked on the wire shapes rather than trusted to the panel that fills them in."""
    section = api_service.split("the evidence write path (TG11.3)")[1].split(
        "the findings surface")[0]
    # Prose about the rule is not a violation of it, so the comments come out first: what is
    # being checked is what these methods put on the wire.
    block = "\n".join(line for line in section.splitlines()
                      if not line.strip().startswith("//"))
    for asserted in ("rung", "claim_level", "temporal_precedence", "confidence", "blocked"):
        assert asserted not in block, "%r must be computed by the server, not sent" % asserted
    types_source = _read("types", "api.ts")
    append = types_source.split("export interface EvidenceAppend")[1].split("}")[0]
    for asserted in ("rung", "recorded_at", "temporal_precedence"):
        assert asserted not in append


def test_the_panel_presents_the_rung_as_computed_rather_than_as_recorded():
    view = _read("components", "EvidenceView.tsx")
    assert "recomputed by the claim ladder" in view
    assert "state?.rung_source" in view
    # The gates are rendered from the response; a hard-coded rung name would be this file
    # deciding what a chain is worth.
    assert "ladder.gates.map" in view
    for rung in ("'robust_association'", "'candidate_precursor'",
                 "'demonstrated_predictive_utility'"):
        assert rung not in view


def test_the_append_form_shows_the_revision_it_is_extending():
    """A compare-and-swap refusal is only comprehensible if the head was on screen."""
    view = _read("components", "EvidenceView.tsx")
    assert "expected_head_sha256: state.head_sha256" in view
    assert "Extending revision" in view


def test_the_mining_panel_reaches_every_route_the_surface_serves(app_source, api_service):
    """TG11.4. Eleven routes, and a panel that would be a catalogue if it called only some."""
    view = _read("components", "StructureMiningView.tsx")
    assert "name: 'Structure mining'" in app_source
    assert "activeTab === 'mining'" in app_source
    for call in ("getMiningCapabilities", "listMiningRecords", "admitRecord", "priceFamily",
                 "calibrateTolerance", "generateMotifs", "freezeMotifs", "confirmMotifs",
                 "publishMotif", "transferMotif", "auditInvariance"):
        assert "apiService.%s" % call in view, call


def test_no_request_this_client_can_send_carries_a_feature(api_service):
    """The R22-shaped rule of TG11.4, checked on the wire shapes rather than on the panel.

    A motif is a configuration of features. A client that could send one could draw the shape
    it wanted the programme to confirm, and every number computed afterwards would be correct
    and meaningless. The comments come out first: prose about the rule is not a breach of it.
    """
    section = api_service.split("structure mining (TG11.4)")[1].split(
        "the findings surface")[0]
    block = "\n".join(line for line in section.splitlines()
                      if not line.strip().startswith("//"))
    # `n_features` is how many the server found, which the pricing call needs and which
    # cannot carry a shape. A coordinate can, so the distinction is a count against a value.
    block = block.replace("n_features:", "n_found:")
    for asserted in ("coordinate", "coords", "features:", "graph", "occurrence",
                     "p_value", "rung", "support:"):
        assert asserted not in block, "%r is measured by the server, not sent" % asserted


def test_the_tolerance_travels_as_a_digest_and_never_as_a_number(api_service):
    """It decides which configurations count as repeats; a typed one is where a wrong answer
    enters looking like a measurement."""
    types_source = _read("types", "api.ts")
    request = types_source.split("export interface MiningRunRequest")[1].split("}")[0]
    assert "tolerance_sha256" in request
    assert "tolerance:" not in request, (
        "a run must name a calibration, not carry a width")
    view = _without_comments(_read("components", "StructureMiningView.tsx"))
    assert "tolerance?.tolerance_sha256" in view


def test_the_mining_panel_prices_the_search_before_it_runs_one():
    """R18 made visible: the refusal is arithmetic, and it is shown before anything is mined."""
    view = _without_comments(_read("components", "StructureMiningView.tsx"))
    assert "split_is_not_optional" in view
    assert "surrogates_required" in view
    assert "priceIt" in view


def test_the_mining_panel_presents_candidates_as_selection_not_as_findings():
    view = _read("components", "StructureMiningView.tsx")
    assert "generated.claim_boundary" in view
    assert "confirmation.receipt.claim_boundary" in view
    # A verdict composed here would be this file deciding what a mining pass was worth.
    for invented in ("confirmed motif", "significant", "discovery"):
        assert invented not in _without_comments(view).lower()


def test_the_cross_domain_panel_reaches_every_route_the_surface_serves(app_source,
                                                                      api_service):
    """TG11.4b. Seven routes; a panel that called only some would be a catalogue of them."""
    view = _read("components", "CrossDomainRecordView.tsx")
    assert "name: 'Cross-domain record'" in app_source
    assert "activeTab === 'crossDomainRecord'" in app_source
    for call in ("getCrossDomainCapabilities", "alignDomains", "priceCrossDomainLags",
                 "describeCrossDomainPartition", "generateCrossDomain", "sealCrossDomain",
                 "confirmCrossDomain"):
        assert "apiService.%s" % call in view, call


def test_no_cross_domain_request_carries_a_lag_in_frames(api_service):
    """The module exists because two clocks have two frame sizes.

    A lag entered in frames here would be a duration on one of the two clocks and a different
    duration on the other, and the family would mean something different to each domain. The
    comments come out first: prose about the rule is not a breach of it.
    """
    section = api_service.split("the cross-domain record (TG11.4b)")[1].split(
        "the findings surface")[0]
    block = chr(10).join(line for line in section.splitlines()
                         if not line.strip().startswith("//"))
    assert "lag_seconds" in block
    for asserted in ("lag_frames", "'lags'", "max_lag", "cadence_seconds"):
        assert asserted not in block, "%r is the server's conversion, not the client's" % asserted


def test_the_cross_domain_panel_declares_units_rather_than_defaulting_them(api_service):
    """R19: a channel table carries names and numbers, and neither says what they mean."""
    types_source = _read("types", "api.ts")
    source_type = types_source.split("export interface CrossDomainSource")[1].split("}")[0]
    assert "channels" in source_type
    assert "?" not in source_type.split("channels")[1].split(";")[0], (
        "the per-channel declaration must be required, not optional")
    view = _without_comments(_read("components", "CrossDomainRecordView.tsx"))
    assert "semantics" in view and "units" in view
    for invented in ("'unknown'", '"unknown"', "arbitrary units"):
        assert invented not in view


def test_the_cross_domain_panel_offers_no_way_to_resample(api_service):
    """The tempting repair for two clocks that do not line up, and it is not on the panel."""
    view = _without_comments(_read("components", "CrossDomainRecordView.tsx"))
    for offered in ("resample", "interpolate", "nearest", "reindex", "ffill"):
        assert offered not in view.lower(), "%r must not be a control here" % offered
    section = api_service.split("the cross-domain record (TG11.4b)")[1].split(
        "the findings surface")[0]
    assert "resample" not in section.lower()


def test_the_cross_domain_confirmation_sends_the_records_and_nothing_else(api_service):
    """A knob still turnable after the seal is a family member chosen after the declaration."""
    call = api_service.split("async confirmCrossDomain")[1].split("},")[0]
    for appended in re.findall(r"form\.append\('([^']+)'", call):
        assert appended in ("first", "second", "published_sha256"), appended


def test_the_cross_domain_panel_shows_what_the_alignment_discarded():
    """A join that quietly kept a third of one record is a different study from the declared one."""
    view = _read("components", "CrossDomainRecordView.tsx")
    assert "discarded_native_observations" in view
    assert "retained_native_observations" in view
    assert "n_common_observations" in view


def test_the_cross_domain_panel_presents_the_receipt_rather_than_a_verdict():
    view = _read("components", "CrossDomainRecordView.tsx")
    assert "confirmation.claim_boundary" in view
    assert "confirmation.receipt.vacuous" in view
    stripped = _without_comments(view).lower()
    for invented in ("causes", "causal link", "significant"):
        assert invented not in stripped


def test_the_mining_panel_shows_a_vacuous_confirmation_as_not_one():
    """R5 on the wire: an ensemble that could not have rejected anything did not check it."""
    view = _read("components", "StructureMiningView.tsx")
    assert "confirmation.vacuous" in view
    assert "not confirmed by anything" in view


def test_the_mining_panel_says_whose_claim_the_declared_transforms_are():
    view = _read("components", "StructureMiningView.tsx")
    assert "audit.declared_transforms_are_the_callers" in view
    assert "report.overclaimed" in view


def test_the_mining_panel_carries_the_origin_licence_into_a_transfer():
    """A definition that crossed domains having forgotten where it came from is the erasure
    R17 exists to prevent."""
    view = _read("components", "StructureMiningView.tsx")
    assert "published.origin_licence" in view
    assert "seal.publication_note" in view


def test_negative_evidence_is_presented_as_load_bearing():
    view = _read("components", "EvidenceView.tsx")
    assert "blocking_entries" in view
    assert "no quantity of favourable" in view
    assert "unblocked_rung" in view


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


def test_dataset_capabilities_gate_navigation_with_visible_backend_reasons(app_source):
    """Disabled scientific paths must teach rather than disappear or silently grey out."""
    for operation in ("cross_domain_analysis", "boundary_lab", "dtcwt_spatial",
                      "gridded_diagnostics", "structure_mining", "forecast_evaluation"):
        assert "operation: '%s'" % operation in app_source
    assert "selectedCapability.operations[tab.operation]" in app_source
    assert "disabled={unavailable}" in app_source
    assert "Unavailable" in app_source and "decision.reason" in app_source
    assert "aria-describedby" in app_source


def test_capability_profile_shows_yes_no_unknown_and_operation_refusals():
    view = _read("components", "DatasetCapabilityProfile.tsx")
    assert "profile.capabilities.map" in view
    assert "profile.operations" in view
    assert "not established" in view
    assert "decision.reason" in view
    assert "profile.profile_sha256" in view


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
    for key in ("stores", "network_enabled", "network_env_var", "r13_minimum_crop",
                "r13_dyadic_operational_crop", "analysis_transforms", "r13_legacy_note"):
        assert key in body
    first = next(iter(body["stores"].values()))
    assert "note" in first, "the store picker shows the note; it must be present"
    # T4C.5i step 6: the table is the requirement; the dyadic rounding is reported separately.
    assert body["r13_minimum_crop"]["4"] == 324
    assert body["r13_dyadic_operational_crop"]["4"] == 512
    assert "not a derived power criterion" in body["r13_legacy_note"]


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
        "levels": [850, 500], "n_levels_analysis": 3,
        "analysis": {"transform_family": "swt", "levels": 3,
                     "wavelet": "db3", "boundary_mode": "reflect",
                     "dtcwt_level1": "near_sym_b", "dtcwt_qshift": "qshift_b"},
    }).json()

    for key in ("spec", "cached", "structure", "assessment", "geometry",
                "acquisition_plan", "cli"):
        assert key in body
    for key in ("amplification", "chunk_hostile", "bytes_wanted",
                "bytes_fetched_estimate", "warning", "advice", "byte_basis", "selection"):
        assert key in body["assessment"], "assessment.%s is read by the UI" % key
    variable = body["structure"]["variables"]["temperature"]
    for key in ("shape", "chunks", "chunk_megabytes"):
        assert key in variable, "structure.variables[].%s is read by the UI" % key
    # The UI renders both thresholds, the implementation-derived per-level geometry and the
    # coordinate expansion/cost result rather than one generic error string.
    geometry = body["geometry"]
    for key in ("current_shape", "verdict", "meets_absolute_minimum",
                "meets_recommended_minimum", "absolute_minimum", "recommended_minimum",
                "levels", "analysis_sha256"):
        assert key in geometry
    plan = body["acquisition_plan"]
    assert set(plan["suggestions"]) == {"absolute", "recommended"}
    assert len(plan["plan_sha256"]) == 64
    assert geometry["analysis"]["transform_family"] == "swt"
    assert geometry["analysis"]["levels"] == 3
    assert geometry["analysis"]["config"] == {"wavelet": "db3", "mode": "reflect"}
    assert "--analysis-levels 3 --analysis-transform swt" in body["cli"]
    assert "--wavelet db3 --boundary-mode reflect" in body["cli"]
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


# ================================================= recorded review (TG11.5)


def _review_view() -> str:
    return _read("components", "ReviewView.tsx")


def test_the_review_workspace_is_routed_and_kept_separate_from_findings(app_source):
    assert "name: 'Recorded review'" in app_source
    assert "activeTab === 'review'" in app_source
    assert "<ReviewView" in app_source
    review_start = app_source.index("activeTab === 'review'")
    review_end = app_source.index(")}", review_start)
    assert "FindingsView" not in app_source[review_start:review_end]


def test_the_review_client_can_read_but_cannot_run_or_write_a_review():
    service = _read("services", "api.ts")
    start = service.index("async getStudyReview")
    block = service[start:service.index("\n  }", start)]
    assert "method: 'GET'" in block
    assert "method: 'POST'" not in block
    assert "fetch(" in block
    assert "run" not in block.lower()


def test_recorded_argument_is_visibly_fenced_from_any_claim_permission():
    view = _review_view()
    assert 'aria-label="Recorded-not-reproducible boundary"' in view
    assert "surface.declaration" in view
    assert "surface.claim_boundary" in view
    assert "argument, not evidence and not permission to make a claim" in view
    assert "Recorded, not reproducible" in view


def test_the_complete_record_outcome_dissent_and_cost_audit_are_rendered():
    view = _review_view()
    assert "review.rendered" in view
    assert "entry.rendered" in view
    assert "review.cost_receipts" in view
    for field in ("call_count", "input_tokens", "cached_input_tokens", "output_tokens",
                  "total_tokens", "cache_hit_fraction", "receipt_sha256"):
        assert "receipt.%s" % field in view
    assert "no price or review-quality claim" in view


def test_missing_review_artifacts_never_render_as_reassurance():
    view = _review_view()
    assert "surface.absence_note" in view
    assert "not the same as no review existing" in view
    assert "not evidence that no exchange occurred" in view
    assert "not the same as the review costing nothing" in view


def test_the_review_surface_preserves_the_workflow_accessibility_contract():
    view = _review_view()
    assert "aria-busy={busy}" in view
    assert 'role="status"' in view
    assert 'role="alert"' in view
    assert 'htmlFor="review-study-id"' in view
    assert 'id="review-study-id"' in view
    assert view.count('aria-hidden="true"') >= 3


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

# ---------------------------------------------------------- TG17.3 schema-driven controls


def test_adapter_controls_contain_no_domain_branch():
    """The review rule TG17.3 states: a hardcoded form in the generic composer fails.

    Checked mechanically because it is the kind of rule that erodes one convenient special case
    at a time. The panel may switch on a control's declared `kind`; it may not know that
    reanalysis has a pressure level or that TESS has sectors.
    """
    panel = _strip_comments(_read("components", "AdapterControls.tsx"))
    for domain in ("reanalysis", "argo", "tess", "order_book", "era5"):
        assert domain not in panel.lower(), (
            "AdapterControls.tsx names the domain %r; a fifth adapter would then need this "
            "file edited, which is exactly what the registry exists to prevent" % domain)
    composer = _strip_comments(_read("components", "ExperimentComposer.tsx"))
    assert "AdapterControlPanel" in composer
    for domain in ("era5", "argo_float", "tess_lightcurve", "order_book"):
        assert domain not in composer, (
            "ExperimentComposer.tsx names the domain %r rather than rendering the registered "
            "schema" % domain)


def test_adapter_controls_render_every_declared_control_kind():
    """A control kind the backend can register and the UI cannot render is an unusable control."""
    from src.core.experiment_adapter import CONTROL_KINDS

    panel = _read("components", "AdapterControls.tsx")
    for kind in CONTROL_KINDS:
        if kind == "text":
            continue  # the default branch
        assert "'%s'" % kind in panel, (
            "control kind %r can be registered but has no renderer, so an adapter declaring it "
            "would produce a control a researcher cannot operate" % kind)


def test_adapter_payload_has_the_fields_the_control_panel_reads(client):
    body = client.get("/api/v1/experiment-composer/adapters").json()
    assert body["adapters"], "the domain selector reads this list"
    for row in body["adapters"]:
        assert {"adapter_id", "domain", "definition_sha256", "controls",
                "declaration", "implements", "onboarding_cost"} <= set(row)
        for field in row["controls"]["fields"]:
            assert {"name", "label", "kind", "help", "required", "default", "choices",
                    "minimum", "maximum", "units"} <= set(field)
            assert field["help"].strip(), "every control renders its declared help text"


def test_conformance_payload_has_the_fields_the_report_panel_reads(client):
    body = client.post(
        "/api/v1/experiment-composer/adapters/reanalysis.standardized-level/conformance",
        json={}).json()
    assert {"adapter_id", "domain", "conformant", "counts", "checks",
            "claim_boundary", "record_kind"} <= set(body)
    assert body["record_kind"] == "deterministic_known_answer_not_acquired_data"
    for check in body["checks"]:
        assert {"check", "status", "detail", "evidence"} <= set(check)
        assert check["status"] in {"PASS", "FAIL", "NOT_APPLICABLE", "NOT_PROBED"}


# ------------------------------------------------- TG17.4 clock, support and coverage


def test_the_coverage_view_draws_support_by_time_and_not_by_index():
    """A sparse record must not be able to look dense because it has as many rows.

    Positions come from the support bounds and the window; nothing here indexes into an array
    to decide where a bar starts. This is the visual half of the invariant the backend holds.
    """
    view = _strip_comments(_read("components", "CoverageTimeline.tsx"))
    assert "window_end_seconds - row.window_start_seconds" in view
    assert "row.intervals.map" in view
    assert "raw_row_count" in view, "the row count is shown as the number that is not evidence"
    assert "rows_are_not_evidence" in view


def test_the_coverage_view_shows_manufactured_overlap_separately():
    """Support a kernel created is never mixed into support the records contain."""
    view = _read("components", "CoverageTimeline.tsx")
    assert "manufactured_overlap_seconds" in view
    assert "created by the declared kernel" in view
    assert "effective_sample_size" in view and "governing_scale_seconds" in view


def test_the_kernel_picker_offers_no_default_for_a_declared_parameter():
    """A tolerance the form pre-filled is a scientific choice nobody made."""
    picker = _read("components", "CoverageTimeline.tsx")
    assert "no default — this is a scientific choice" in picker
    assert "manufactures_simultaneity" in picker
    assert "no registered adapter admits this" in picker


def test_the_coverage_view_contains_no_domain_branch():
    view = _strip_comments(_read("components", "CoverageTimeline.tsx"))
    for domain in ("reanalysis", "argo", "tess", "order_book", "era5"):
        assert domain not in view.lower(), (
            "CoverageTimeline.tsx names the domain %r; coverage is drawn from declared support "
            "and a fifth domain must reach it without this file being edited" % domain)


def test_alignment_kernel_payload_has_the_fields_the_picker_reads(client):
    body = client.get("/api/v1/experiment-composer/alignment-kernels").json()
    assert body["kernels"], "the kernel selector reads this list"
    for row in body["kernels"]:
        assert {"name", "summary", "required_parameters", "manufactures_simultaneity",
                "invents_values", "refused_over_violations", "admitted_by",
                "usable_across_all_registered_domains"} <= set(row)
    assert body["default"] == "exact_support_overlap"


def test_alignment_payload_has_the_fields_the_coverage_view_reads(client):
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    body = client.post("/api/v1/experiment-composer/manifests/alignment",
                       json=recipe["canonical_manifest"]).json()
    assert body["record_binding"] == "benchmark_known_answer"
    assert body["row_indices_were_not_compared"] is True
    for row in body["coverage"]:
        assert {"label", "window_start_seconds", "window_end_seconds", "intervals", "gaps",
                "gap_count", "covered_fraction", "raw_row_count", "valid_row_count",
                "rows_are_not_evidence", "support_is_stationary",
                "native_scale_seconds"} <= set(row)
    for pair in body["pairs"]:
        assert {"left", "right", "overlap_seconds", "manufactured_overlap_seconds",
                "governing_scale_seconds", "effective_sample_size",
                "effective_sample_size_basis", "row_counts_not_used", "status"} <= set(pair)


def test_the_preflight_alignment_block_is_shaped_as_the_type_declares(client):
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    body = client.post("/api/v1/experiment-composer/manifests/preflight",
                       json=recipe["canonical_manifest"]).json()
    alignment = body["alignment"]
    assert alignment["measurement_values_opened"] is False
    for window in alignment["windows"]:
        assert {"name", "window_seconds", "clock", "clock_note", "pairs"} <= set(window)
        assert {"elapsed_seconds", "nominal_seconds", "discrepancy_seconds",
                "clock_is_uniform"} <= set(window["clock"])


def test_the_runner_the_composer_now_offers_still_acquires_nothing():
    """The disabled "not available yet" button is gone; what replaced it must not overclaim.

    Until TG17.6 this test asserted the composer refused to offer a runner at all. It now offers
    one, so the check moved to the thing that was actually being protected: the runner executes
    the declared state machine against registered suites that open no archive, and the browser
    says so beside the button rather than leaving a researcher to infer it from a run that
    completed.
    """
    composer = _read("components", "ExperimentComposer.tsx")
    monitor = _read("components", "RunMonitor.tsx")
    assert "Run experiment — not available yet" not in composer
    assert "runContract.not_yet_available" in composer
    assert "A completed run is an executed plan, not evidence." in composer
    assert "this is a rehearsal, not data" in monitor


# ------------------------------------------ TG17.5 family accounting and declared nulls


def test_the_browser_no_longer_computes_its_own_family_size():
    """The fourth copy of the family formula, removed.

    A product in the browser could disagree with the receipt — and it did: it multiplied pairs
    by channels, scales, windows and relationships, and knew nothing of the arities, lags,
    representations or motifs the manifest declares. The Composer now shows the number the
    server priced, so a researcher cannot read the size of their own search two ways.
    """
    composer = _strip_comments(_read("components", "ExperimentComposer.tsx"))
    assert "n * (n - 1) / 2" not in composer
    assert "declaredFamily" in composer
    assert "experimentFamily" in composer


def test_the_family_panel_shows_the_multiplication_and_what_one_more_axis_costs():
    view = _read("components", "FamilyPlan.tsx")
    assert "in_human_terms" in view
    assert "expansion_cost" in view
    assert "what one more of each would cost, before the freeze" in view
    assert "surrogates_required_after" in view


def test_the_family_panel_keeps_the_correction_unit_beside_its_held_out_partition():
    """A smaller correction unit is legitimate only because a partition was closed."""
    view = _read("components", "FamilyPlan.tsx")
    assert "correction_unit_members" in view
    assert "held_out_partition" in view
    assert "generate_then_confirm" in view


def test_the_family_panel_reports_precedence_availability_without_shrinking_the_family():
    view = _read("components", "FamilyPlan.tsx")
    assert "unavailable_precedence_members" in view
    assert "domains_without_precedence_policy" in view
    assert "family_size" in view


def test_a_refused_null_is_shown_disabled_with_its_reason_rather_than_hidden():
    view = _read("components", "FamilyPlan.tsx")
    assert "inadmissible_reason" in view
    assert "disabled" in view
    assert "preserves" in view and "destroys" in view


def test_a_declared_null_parameter_is_offered_with_no_default():
    view = _read("components", "FamilyPlan.tsx")
    assert "no default — this is a scientific choice" in view


def test_the_family_panel_has_no_domain_branch():
    view = _strip_comments(_read("components", "FamilyPlan.tsx"))
    for domain in ("reanalysis", "argo", "tess", "order_book"):
        assert domain not in view


def test_the_family_payload_is_shaped_as_the_type_declares(client):
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    body = client.post("/api/v1/experiment-composer/manifests/family",
                       json=recipe["canonical_manifest"]).json()
    assert {"schema", "axes", "in_human_terms", "family_size", "declared_surrogates",
            "account", "correction", "largest_affordable_family", "resource_requirement",
            "expansion_cost", "screen_and_confirm", "precedence"} <= set(body)
    assert {"stage", "declared_search_members", "correction_unit_members",
            "held_out_partition", "surrogates_required", "affordable"} <= set(body["correction"])
    assert {"axis", "declared_values", "examples", "contributes"} <= set(body["axes"][0])


def test_the_null_family_payload_is_shaped_as_the_type_declares(client):
    body = client.get("/api/v1/experiment-composer/null-families").json()
    assert {"schema", "families", "modes", "note", "claim_boundary"} <= set(body)
    family = body["families"][0]
    assert {"name", "modes", "operates_on", "preserves", "destroys", "parameters",
            "admissible", "inadmissible_reason", "admitted_by",
            "usable_across_all_registered_domains"} <= set(family)


# ------------------------------------------ TG17.6 the orchestrated, resumable run


def test_the_browser_has_no_create_run_operation_only_open_or_resume():
    """A "new run" button would be a second run of a plan that already has one.

    The identity is the content address of the manifest, so posting it again resumes. The label
    the researcher reads has to say that, because the whole guarantee is invisible otherwise.
    """
    composer = _read("components", "ExperimentComposer.tsx")
    assert "Open or resume the run" in composer
    assert "openExperimentRun" in composer
    assert "Run experiment — not available yet" not in composer


def test_the_runner_draws_the_state_machine_the_backend_enforces():
    view = _read("components", "RunMonitor.tsx")
    assert "machine.work_stages" in view
    assert "machine.terminal_states" in view
    assert "RunStateTrail" in view


def test_a_retry_and_an_editable_copy_are_offered_for_different_failures():
    """Asking the archive again is not the remedy for the archive saying no."""
    view = _strip_comments(_read("components", "RunMonitor.tsx"))
    assert "progress.retryable" in view
    assert "progress.state === 'REFUSED'" in view
    assert "Only an operational failure can be retried." in view
    assert "The refused run is left exactly as it is." in view


def test_the_progress_view_reports_replay_rather_than_re_request():
    view = _read("components", "RunMonitor.tsx")
    assert "replayed, not re-requested" in view
    assert "component.reused" in view


def test_the_progress_bar_is_bounded_by_the_declared_plan():
    view = _read("components", "RunMonitor.tsx")
    assert "bounded_work" in view
    assert "declared steps" in view
    assert 'aria-valuemax={work.total_steps}' in view


def test_the_runner_names_every_component_the_run_did_not_produce():
    view = _read("components", "RunMonitor.tsx")
    assert "missing_components" in view
    assert "Components this run did not produce" in view
    assert "coverage_policy" in view


def test_a_rehearsal_suite_is_labelled_as_acquiring_nothing():
    view = _read("components", "RunMonitor.tsx")
    assert "capabilities.acquires === false" in view
    assert "this is a rehearsal, not data" in view


def test_the_runner_has_no_domain_branch():
    view = _strip_comments(_read("components", "RunMonitor.tsx"))
    for domain in ("reanalysis", "argo", "tess", "order_book"):
        assert domain not in view


def test_the_run_contract_payload_is_shaped_as_the_type_declares(client):
    body = client.get("/api/v1/experiment-runs").json()
    assert {"schema", "state_machine", "worker_suites", "runs", "available_now",
            "not_yet_available", "claim_boundary"} <= set(body)
    assert {"states", "work_stages", "terminal_states", "transitions", "component_statuses",
            "retryable_statuses"} <= set(body["state_machine"])
    assert {"name", "description", "capabilities"} <= set(body["worker_suites"][0])


def test_the_run_progress_and_receipt_payloads_are_shaped_as_the_types_declare(client):
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    opened = client.post("/api/v1/experiment-runs", json=recipe["canonical_manifest"]).json()
    assert {"run_id", "run_sha256", "manifest_sha256", "resumed", "progress",
            "receipt"} <= set(opened)
    progress = client.get("/api/v1/experiment-runs/%s/progress" % opened["run_id"]).json()
    assert {"run_id", "state", "stages", "bounded_work", "retryable", "results_visible",
            "claim_boundary"} <= set(progress)
    assert {"stage", "components"} <= set(progress["stages"][0])
    assert {"component", "status", "artifact_sha256", "remediation",
            "reused"} <= set(progress["stages"][0]["components"][0])
    receipt = client.get("/api/v1/experiment-runs/%s" % opened["run_id"]).json()
    assert {"state", "history", "artefacts", "missing_components", "stage_decisions",
            "bounded_work", "coverage_policy", "confirmation", "claim_boundary"} <= set(receipt)


# ----------------------------------------------- TG17.7 the guided path in the browser


def _composer() -> str:
    return _read("components", "ExperimentComposer.tsx")


def _path_view() -> str:
    return _read("components", "ComposerPath.tsx")


def test_the_composer_holds_no_copy_of_the_order_of_operations():
    """The order of operations *is* the scientific discipline.

    A family priced after acquisition is priced knowing what the data looked like. A component
    that decided for itself which step comes next would be a second opinion about that, and the
    one a researcher followed would not be the one the receipt records.
    """
    view = _strip_comments(_composer())
    assert "composerPathState" in view
    assert "pathState.steps" in view
    for step_id in ("question", "domains", "observation", "preflight", "analysis",
                    "freeze_and_run", "interpret"):
        assert "panel('%s'" % step_id in view, step_id
    assert "ordinal <" not in view and "ordinal >" not in view, (
        "step order is the server's; comparing ordinals here would re-derive it")


def test_exactly_one_next_action_is_rendered_and_it_comes_from_the_server():
    view = _strip_comments(_path_view())
    assert "state.next_action" in view
    assert "Next legitimate action" in view
    assert "Blocked before" in view
    assert view.count("action.label") == 1, "one action means one label"


def test_the_path_is_operable_without_a_mouse():
    """TG9.4's condition, applied to the surface a whole experiment is declared through."""
    view = _path_view()
    assert 'role="tablist"' in view and 'role="tab"' in view
    assert "aria-selected" in view and "aria-controls" in view
    assert "ArrowRight" in view and "ArrowLeft" in view and "Home" in view and "End" in view
    assert "tabIndex" in view, "only the active tab is in the tab order"
    assert "focus:ring" in view, "keyboard focus must be visible"
    assert 'aria-hidden="true"' in view, "decorative icons must be hidden from a screen reader"


def test_every_step_panel_is_a_labelled_tabpanel():
    view = _composer()
    assert 'role="tabpanel"' in view
    assert "aria-labelledby={`composer-tab-${stepId}`}" in view
    assert "aria-busy={!!busy}" in view


def test_a_blocked_step_stays_reachable_rather_than_disappearing():
    """Hiding a blocked step hides the reason it is blocked, which is the useful part."""
    view = _strip_comments(_path_view())
    assert "steps.map(" in view
    assert "disabled" not in view.split("role=\"tablist\"")[1].split("</div>")[0], (
        "no step tab may be disabled; a step nobody can open is a refusal nobody can read")


def test_the_researcher_keeps_their_place_across_navigation_and_refresh():
    view = _composer()
    assert "ACTIVE_STEP_KEY" in view
    assert "localStorage.getItem(ACTIVE_STEP_KEY)" in view
    assert "localStorage.setItem(ACTIVE_STEP_KEY" in view


def test_an_unavailable_domain_is_shown_disabled_with_the_backend_s_reason():
    """A control that vanishes when it becomes inadmissible is indistinguishable from one that
    was never offered."""
    view = _strip_comments(_path_view())
    assert "row.selectable" in view
    assert "row.unavailable_reason" in view
    assert "Unavailable:" in view
    assert ".filter(" not in view.split("menu.domains.map(")[0].split("<ul")[-1], (
        "the menu is rendered whole; filtering it here would hide a refusal"
    )


def test_the_domain_menu_shows_what_each_domain_breaks_before_it_is_chosen():
    view = _path_view()
    assert "row.breaks" in view
    assert "Breaks:" in view
    assert "no declared assumption" in view


def test_the_browser_never_invents_an_observation_for_a_domain():
    """An adapter says how a domain is translated; it does not say what is measured, in which
    units, in which role, or from which record."""
    view = _strip_comments(_composer())
    assert "option?.observation" in view
    assert "option.observation as any" in view
    assert "domainMenu.minimum_domains" in view


def test_a_preset_is_applied_as_the_instants_the_server_resolved_it_to():
    view = _strip_comments(_composer())
    assert "composerWindowPresets" in view
    assert "preset.start_utc" in view and "preset.end_utc" in view
    assert "setDate" not in view and "getMonth" not in view, (
        "a boundary computed twice is two boundaries"
    )


def test_the_preregistration_summary_is_the_server_s_sentences_not_the_ui_s():
    view = _strip_comments(_path_view())
    assert "summary.sentences" in view
    assert "summary.manifest_sha256" in view
    composer = _strip_comments(_composer())
    assert "composerPreregistrationSummary" in composer


def test_the_ladder_separates_acquisition_a_run_a_finding_and_evidence():
    view = _path_view()
    assert "StageLadder" in view
    assert "rung.is_not" in view and "rung.gate" in view and "rung.why_not" in view
    assert "Not:" in view


def test_a_destructive_action_asks_twice_and_says_what_is_lost():
    view = _path_view()
    assert "ConfirmButton" in view
    assert 'role="alertdialog"' in view
    assert "confirmLabel" in view
    composer = _composer()
    assert "ConfirmButton" in composer


def test_the_advanced_manifest_inspector_is_disclosed_and_never_required():
    view = _path_view()
    assert "ManifestInspector" in view
    assert "<details" in view
    assert "never requires editing this" in view
    assert "composerExportManifest" in _composer()
    assert "composerImportManifest" in _composer()


def test_every_empty_panel_says_an_unasked_question_is_not_a_clean_result():
    """An empty panel that reads as reassurance is the failure this whole programme is about."""
    view = _composer()
    assert "an unasked question, not a clean" in view
    assert "not a null result" in view
    assert "an unknown cost, not a small one" in view


def test_the_path_view_has_no_domain_branch():
    view = _strip_comments(_path_view())
    for domain in ("reanalysis", "argo", "tess", "order_book"):
        assert domain not in view


def test_the_path_payloads_are_shaped_as_the_types_declare(client):
    contract = client.get("/api/v1/experiment-composer/path").json()
    assert {"schema", "steps", "step_statuses", "duration_presets", "ladder", "note",
            "claim_boundary"} <= set(contract)
    assert {"step_id", "ordinal", "title", "question", "settles", "controls", "action_label",
            "action_route", "claim_boundary"} <= set(contract["steps"][0])
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    state = client.post("/api/v1/experiment-composer/path/state",
                        json=recipe["canonical_manifest"]).json()
    assert {"schema", "manifest_sha256", "study_id", "steps", "satisfied", "next_action",
            "ladder", "run_state", "claim_boundary"} <= set(state)
    assert {"status", "reason", "detail"} <= set(state["steps"][0])
    assert {"step_id", "label", "route", "status", "why", "blocked"} <= set(state["next_action"])
    assert {"rung", "title", "is", "is_not", "gate", "reached", "why_not"} <= set(state["ladder"][0])


# --------------------------------------------- TG17.8 the comparison views in the browser


def _views() -> str:
    return _read("components", "ComparisonViews.tsx")


def test_the_view_component_holds_no_claim_boundary_of_its_own():
    """What may be concluded is a scientific fact, so it is served rather than written here.

    A boundary a component composed would be a second boundary, and the sentence a reader saw
    under a chart would not be the sentence a receipt could show them afterwards.
    """
    view = _views()
    assert "may_not_conclude" in view and "may_conclude" in view
    assert "mode_forbids" in view
    for invented in ("co-occurrence only", "carries no clock", "no common ruler"):
        assert invented not in view, (
            "the boundary text belongs to the server; %r here is a second copy" % invented)


def test_the_coverage_cell_is_drawn_from_a_named_state_and_never_from_a_number():
    """The graphical failure this slice exists for.

    A cell whose width came from a fraction would draw absent support as a zero-width bar, and a
    reader would see "we looked and found nothing" where the truth is "we could not look".
    """
    view = _strip_comments(_views())
    assert "CELL_STYLE" in view
    for state in ("COVERED", "SPARSE", "ABSENT", "REFUSED"):
        assert state in view, state
    assert "width:" not in view.replace(" ", "")
    assert "cell.value" not in view
    assert "is_measured_zero" not in view, (
        "the component must not need the flag; it never has a number to mistake for a state")


def test_every_role_is_drawn_in_its_colour_its_marker_and_its_word():
    view = _views()
    assert "item.colour" in view
    assert "item.marker" in view
    assert "item.word" in view


def test_the_numbers_behind_every_view_are_reachable_from_the_view():
    view = _views()
    assert "AccessibleTable" in view
    assert "payload.table" in view or "view.table" in view
    assert "Show the numbers behind each view" in view
    assert "<caption" in view


def test_an_unregistered_view_renders_rather_than_breaking_the_page():
    """A view the server registers and this build has no drawing for still shows its table,
    its legend and its boundary. Falling through to a crash would make a new view look like a
    broken one, which is the wrong signal in both directions."""
    view = _views()
    assert "default: return null;" in view


def test_the_selection_is_cleared_when_the_plan_changes():
    view = _views()
    assert "setSelection(null)" in view
    assert "why_not_merged" in view


def test_the_views_have_no_domain_branch():
    view = _strip_comments(_views())
    for domain in ("reanalysis", "argo", "tess", "order_book"):
        assert domain not in view


def test_the_composer_shows_the_views_on_the_interpret_step():
    composer = _composer()
    assert "ComparisonViews" in composer
    assert "Comparison views" in composer


def test_the_view_payloads_are_shaped_as_the_types_declare(client):
    contract = client.get("/api/v1/comparison-views").json()
    assert {"schema", "views", "encodings", "axis_kinds", "shared_axis_kinds", "coverage_cells",
            "readings", "mode_forbids", "refusals", "routes", "not_yet_available",
            "claim_boundary"} <= set(contract)
    assert {"view_id", "ordinal", "title", "question", "axes", "roles", "may_conclude",
            "may_not_conclude", "selectable"} <= set(contract["views"][0])
    assert {"role", "word", "colour", "marker", "ordinal", "definition",
            "admits_claim"} <= set(contract["encodings"][0])
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar").json()
    manifest = recipe["canonical_manifest"]
    rendered = client.post("/api/v1/comparison-views/render/coverage_timeline",
                           json=manifest).json()
    assert {"schema", "view_id", "ordinal", "title", "question", "mode", "manifest_sha256",
            "axes", "legend", "body", "table", "results_exist", "run_state", "may_conclude",
            "may_not_conclude", "mode_forbids", "claim_boundary"} <= set(rendered)
    assert {"name", "kind", "domains", "units", "shared", "why"} <= set(rendered["axes"][0])
    assert {"columns", "rows"} <= set(rendered["table"])
    selection = client.post("/api/v1/comparison-views/linked-selection",
                            json={"manifest": manifest, "window": "week"}).json()
    assert {"schema", "window", "manifest_sha256", "contributions", "merged_interval",
            "why_not_merged", "claim_boundary"} <= set(selection)
    assert {"domain", "state", "reason", "native_interval",
            "contributes"} <= set(selection["contributions"][0])


def test_every_served_comparison_view_has_a_drawing_or_a_declared_fallback():
    """The guard that catches a view registered on the server and forgotten in the browser."""
    from src.core.comparison_views import ordered_views

    view = _views()
    for served in ordered_views():
        assert "case '%s':" % served.view_id in view, served.view_id


def test_tg17_receipt_trust_surface_is_reachable_in_composer_and_platform():
    receipt = _read("components", "ExperimentReceipt.tsx")
    composer = _read("components", "ExperimentComposer.tsx")
    app = _read("App.tsx")
    assert "<ExperimentReceiptPanel runId={receipt?.run_id}" in composer
    assert "<ExperimentReceiptPanel trustOnly" in app
    assert "setActiveTab('evidence')" in app
    for key in ("operations", "adapters", "refusals", "receipt_fields", "lineage",
                "claim_boundary"):
        assert "capabilities.%s" % key in receipt or key == "adapters", key


def test_tg17_receipt_import_is_read_only_and_shows_every_evidence_absence():
    receipt = _read("components", "ExperimentReceipt.tsx")
    assert "replayExperimentReceipt" in receipt
    assert "openEvidence" not in receipt
    assert "automatic_actions" not in receipt  # there is deliberately no renderer that executes them
    assert "evidence_handoff.categories.map" in receipt
    assert "Open a separate evidence-study draft" in receipt


def test_tg17_receipt_wire_shapes_cover_export_replay_and_the_generated_contract(client):
    capabilities = client.get("/api/v1/experiment-receipts").json()
    assert {"schema", "software_version", "operations", "adapters", "refusals",
            "receipt_fields", "lineage", "claim_boundary"} <= set(capabilities)
    assert capabilities["operations"] and capabilities["adapters"]
    assert all({"name", "label", "meaning"} <= set(row)
               for row in capabilities["receipt_fields"])
    typescript = _read("types", "api.ts")
    for shape in ("ExperimentReceiptCapabilities", "ExperimentReplayBundle",
                  "ExperimentReceiptExport", "ExperimentReceiptReplay"):
        assert "interface %s" % shape in typescript


def test_tg17_qualification_gate_is_reachable_and_never_hides_blockers():
    view = _read("components", "ExperimentQualification.tsx")
    app = _read("App.tsx")
    assert "<ExperimentQualificationPanel" in app
    assert "record.gates.map" in view
    assert "record.matrix.map" in view
    assert "record.verdict" in view
    assert "Run offline qualification" in view
    assert "live-source acceptance are different gates" in view


def test_mode_switch_changes_relationship_null_and_language_as_one_revision():
    composer = _read("components", "ExperimentComposer.tsx")
    assert "const switchMode" in composer
    assert "mode_relationships?.[mode]" in composer
    assert "row.modes.includes(mode)" in composer
    assert "selectedAdapters.every" in composer
    assert "family: { ...manifest.family, relationships: [relationships[0]] }" in composer
    assert "method: family.name, parameters: {}" in composer
    assert "onChange={() => switchMode(mode)}" in composer


def test_qualification_wire_shapes_match_the_served_plan(client):
    plan = client.get("/api/v1/experiment-qualification").json()
    assert {"schema", "qualification_sha256", "verdict", "record_kind", "matrix", "gates",
            "scientist_actions", "claim_boundary"} <= set(plan)
    assert len(plan["matrix"]) == 6
    assert {"cell_id", "duration", "mode", "start_utc", "end_utc", "manifest_sha256",
            "family_correction", "record_kind", "status"} <= set(plan["matrix"][0])
    assert {"gate_id", "title", "status", "blocking", "detail"} <= set(plan["gates"][0])
    typescript = _read("types", "api.ts")
    for shape in ("ExperimentQualificationGate", "ExperimentQualificationCell",
                  "ExperimentQualificationRecord"):
        assert "interface %s" % shape in typescript


def test_qualification_client_uses_only_the_two_public_routes():
    service = _read("services", "api.ts")
    assert "experimentQualificationPlan" in service
    assert "rehearseExperimentQualification" in service
    assert "`${BASE_URL}/experiment-qualification`" in service
    assert "`${BASE_URL}/experiment-qualification/rehearse`" in service


# ======================================================== T4C.5j: the atmospheric gate record

def _gate_client_section(api_service: str) -> str:
    """The gate methods only. Everything after the marker comment is this surface."""
    marker = "T4C.5j: the atmospheric gate record"
    assert marker in api_service, "the gate client section must be identifiable"
    return api_service.split(marker, 1)[1]


def test_the_gate_panel_reaches_every_route_the_surface_serves(app_source, api_service):
    """A served route with no consumer is the gap this whole slice was opened to close."""
    for path in ("/gate`", "/gate/campaigns`", "/gate/campaigns/$", "/gate/supersessions`",
                 "/gate/supersessions/$", "/gate/receipts`", "/gate/receipts/$"):
        assert path in api_service, path
    assert "GateRecordView" in app_source
    assert "activeTab === 'gate'" in app_source
    assert "id: 'gate', name: 'Atmospheric gate record'" in app_source


def test_no_gate_request_this_client_can_send_changes_anything(api_service):
    """Read-only must be a property of the client too, not only of the routing table.

    A panel that can only read is what makes the missing acquire button a refusal rather than an
    omission a later slice might casually fill in.
    """
    section = _gate_client_section(api_service)
    for verb in ("'POST'", "'PUT'", "'PATCH'", "'DELETE'", "FormData"):
        assert verb not in section, verb


def test_the_gate_panel_shows_a_retirement_before_the_design_it_retires():
    """A reader who has reached the calendar split is already reading the design as live."""
    view = _read("components", "GateRecordView.tsx")
    assert view.index("openCampaign.retired_by &&") < view.index("Decision rule"), (
        "the retirement banner must precede the design body, not follow it")
    assert "acquisition {openCampaign.retired_by.acquisition}" in view
    assert "surface.refusals.map" in view, (
        "the refusals must be rendered from the server's list, not implied by absent buttons")


def test_the_gate_panel_shows_a_retired_design_rather_than_hiding_it():
    """Hiding the retired design would erase the record of what was actually preregistered."""
    view = _read("components", "GateRecordView.tsx")
    assert "Read design" in view
    assert "row.resolvable ? 'yes' : 'no'" in view, (
        "the defect that retired a design must be visible on the row that names it")
    assert "openSupersession.deferred_to_run.map" in view, (
        "a retirement must show what it did not settle as prominently as its reasons")


def test_the_gate_panel_names_an_empty_receipt_list_as_an_absence_of_runs():
    """Rendered bare, an empty list reads as an absence of findings - the opposite claim."""
    view = _read("components", "GateRecordView.tsx")
    assert "NOT_YET_MEASURED" in view
    assert "receipts.statement" in view


def test_the_gate_panel_never_shows_one_verdict_without_the_other():
    """The FAIL/INVALID boundary is only legible if both verdicts and the rule are on screen."""
    view = _read("components", "GateRecordView.tsx")
    assert "openReceipt.gate_verdict" in view and "openReceipt.scientific_verdict" in view
    assert "Replication rule returned" in view
    assert "openReceipt.power_adjudication.reason" in view
    assert "power_applied" in view


def test_the_gate_panel_shows_an_undeclared_agreement_rule_as_a_refusal():
    """D86. A campaign that never preregistered the rule authorising its own acquisition must
    say so on the row. An empty cell would read as a formatting gap rather than as the reason
    that design may not be acquired at all."""
    view = _read("components", "GateRecordView.tsx")
    assert "Agreement rule" in view
    assert "row.overlap_criterion === null" in view
    assert "not preregistered" in view
    # And the rule that is declared is shown in the unit it is actually applied in, because a
    # bound in Kelvin is what D86 was.
    assert "encoding step" in view


# ============================================================= TG18: research instrument shell

def test_research_archive_keeps_record_classes_distinct_and_reachable(app_source):
    archive = _read("components", "ResearchArchive.tsx")

    assert "Research archive" in app_source
    assert "activeTab === 'researchArchive'" in app_source
    for classification in ("SCIENTIFIC EVIDENCE", "EXPERIMENT RUN", "GATE RECEIPT",
                           "EVALUATION RECEIPT", "ACQUISITION RECORD",
                           "VALIDATION FIXTURE"):
        assert classification in archive
    for method in ("listStudies", "experimentRunContract", "listGateReceipts",
                   "listEvaluationReports", "zarrProbes", "listCDSJobs", "listBenchmarks"):
        assert "apiService.%s(" % method in archive
    assert "job.state === 'COMPLETE' && job.acquisition_record" in archive
    assert "a passing fixture is not a published study" in archive


def test_acquire_surfaces_noninteractive_routes_and_human_source_identity():
    source = _read("components", "AcquisitionView.tsx")
    planner = _read("components", "CDSPlanner.tsx")
    service = _read("services", "api.ts")

    assert "catalogue.operational_routes" in source
    assert "route.ui_status" in source
    assert "option.label || option.name" in source
    assert "option.provider || option.product_family" in source
    assert "Source routes" in source
    assert "<CDSPlanner" in source
    assert "apiService.cdsCapabilities(" in planner
    assert "apiService.planCDS(" in planner
    assert "apiService.submitCDSJob(" in planner
    assert "apiService.cancelCDSJob(" in planner
    assert "apiService.resumeCDSJob(" in planner
    assert "apiService.getCDSAcquisitionRecord(" in planner
    assert "Validate plan — no network" in planner
    assert "Planning uses no network" in planner
    assert "plan.network_used" in planner
    assert "A validated plan is not a job" in planner
    assert "Submit durable acquisition job" in planner
    assert "Server managed" in planner
    assert "/data/cds/plan" in service


def test_findings_refuses_to_treat_a_run_or_receipt_label_as_a_published_study():
    findings = _read("components", "FindingsView.tsx")
    archive = _read("components", "ResearchArchive.tsx")

    assert "publishedStudyId" in findings
    assert "studies.some(" in findings
    assert "apiService.getTranslation(publishedStudyId, glossaryName)" in findings
    assert "Current context is not published" in findings
    assert "No LLM action is required" in findings
    gate_mapping = archive.split("...gates.receipts.map", 1)[1].split(
        "...evaluations.map", 1)[0]
    assert "studyId:" not in gate_mapping


# ------------------------------------------------ TG18.3 the global guided research journey


def test_the_global_journey_is_navigation_and_not_a_second_scientific_judge(app_source):
    journey = _read("components", "ResearchJourney.tsx")
    composer = _read("components", "ExperimentComposer.tsx")
    for stage in ("Acquire", "Inspect", "Design", "Run", "Compare", "Admit", "Report"):
        assert "label: '%s'" % stage in journey
    assert 'aria-label="Guided research journey"' in journey
    assert "Navigation only" in journey
    assert "does not advance or replace the claim ladder" in journey
    assert "requestedStep" in composer
    assert "composerPathState" in composer
    assert "setPathState" in composer
    # The shell sends a panel name, never a verdict about the panel.
    for scientific_status in ("SATISFIED", "ACTION_REQUIRED", "INCONCLUSIVE", "PASS", "FAIL"):
        assert scientific_status not in journey
    assert "ResearchJourney" in app_source


def test_every_global_blocker_names_one_remediation_and_legacy_tools_stay_distinct(app_source):
    journey = _read("components", "ResearchJourney.tsx")
    assert "Blocked:" in journey
    assert "Next legitimate action:" in journey
    assert "no record is selected for inspection" in journey
    assert "no study is selected for evidence admission" in journey
    assert "data-workflow-line={'context' in tab ? 'legacy-gridded' : 'evidence'}" in app_source
    assert "legacy-workspace-entry" in app_source
    assert "Legacy · {tab.context}" in app_source


# --------------------------------------- TG18.4 rendered accessibility acceptance


def test_assistive_acceptance_complements_the_source_contract_without_claiming_certification(
        app_source):
    acceptance = io.open(os.path.join(REPO_ROOT, "frontend", "e2e",
                                      "assistive-acceptance.spec.ts"), encoding="utf-8").read()
    css = _read("index.css")

    assert "not WCAG certification" in acceptance
    assert "not a screen-reader" in acceptance
    for concern in ("keyboard route", "focus indicator", "reflow", "contrast",
                    "reduced-motion preference", "text cues"):
        assert concern in acceptance
    for layout in ("desktop", "laptop", "narrow"):
        assert f"name: '{layout}'" in acceptance
    assert "prefers-reduced-motion: reduce" in css
    assert "window.requestAnimationFrame(() => workspaceHeadingRef.current?.focus())" in app_source
    assert "previousActiveTabRef.current !== activeTab" in app_source


def test_rendered_non_colour_acceptance_keeps_status_words_and_semantics():
    acceptance = io.open(os.path.join(REPO_ROOT, "frontend", "e2e",
                                      "assistive-acceptance.spec.ts"), encoding="utf-8").read()

    assert "journey location and blockers remain named when colour is removed" in acceptance
    assert "Blocked: no record is selected for inspection." in acceptance
    assert "Blocked: no study is selected for evidence admission." in acceptance
    assert "aria-current" in acceptance
    assert "accessibleNames" in acceptance
