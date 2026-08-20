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


def test_every_api_method_is_reachable_from_the_ui(api_service, app_source):
    """A service method nothing calls is a route that is still invisible to a researcher."""
    methods = set(re.findall(r"^  async (\w+)\(", api_service, re.M))
    assert methods, "no service methods parsed"
    unused = sorted(m for m in methods if ("apiService.%s" % m) not in app_source)
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
    assert body["r13_minimum_crop"]["4"] == 256


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
    ids = re.findall(r"\{ id: '(\w+)', name: '[^']*', icon: \w+ \}", app_source)
    assert len(ids) >= 9, "expected at least nine research modules, found %s" % ids
    for tab_id in ids:
        assert "activeTab === '%s'" % tab_id in app_source, (
            "tab %r appears in the navigation but has no panel" % tab_id)


def test_browser_rendering_is_still_recorded_as_unverified():
    """The honest counterpart to every test above.

    These tests check contracts, not pixels. No browser is available here, so the nine tabs
    have never been *seen*. That must stay written down in `roadmap.md`, because a green suite
    plus a green build is exactly the combination that makes people assume otherwise.
    """
    roadmap = io.open(os.path.join(REPO_ROOT, "roadmap.md"), encoding="utf-8").read()
    assert "Rendered appearance in a browser still unverified" in roadmap, (
        "roadmap.md must keep stating that the UI has not been visually verified, for as "
        "long as that is true")
