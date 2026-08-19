"""Tests for API infrastructure added in Phase 3.5: health, experiment listing, CORS,
and the data-source visibility that E2 requires.
"""
import os

import pytest

from src.data_layer.adapters import MeteorologicalDataAdapter


# --------------------------------------------------------------------------- health (T3.5.9)

def test_health_reports_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["database_url_scheme"] == "sqlite"
    assert body["datasets_available"] >= 3
    assert body["torch_device"]  # cpu or cuda:N


# --------------------------------------------------------------------- experiment list (T3.5.9)

def _make_experiment(client, name):
    payload = {
        "name": name,
        "description": "listing test",
        "parameter_matrix": {},
        "pipeline": [
            {
                "name": "gen",
                "action": "generate_synthetic",
                "args": {"type": "sinusoid", "height": 8, "width": 8},
            }
        ],
        "metadata": {"code_revision": "test"},
    }
    r = client.post("/api/v1/experiments", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_experiment_list_is_empty_initially(client):
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["experiments"] == []
    assert body["limit"] == 25 and body["offset"] == 0


def test_experiment_list_returns_created_experiments(client):
    ids = {_make_experiment(client, "list-%d" % i) for i in range(3)}
    body = client.get("/api/v1/experiments").json()
    assert body["total"] == 3
    assert {e["id"] for e in body["experiments"]} == ids


def test_experiment_list_pagination_and_bounds(client):
    for i in range(5):
        _make_experiment(client, "page-%d" % i)

    page = client.get("/api/v1/experiments?limit=2&offset=0").json()
    assert page["total"] == 5 and len(page["experiments"]) == 2

    page2 = client.get("/api/v1/experiments?limit=2&offset=4").json()
    assert len(page2["experiments"]) == 1

    # limit is clamped rather than trusted
    clamped = client.get("/api/v1/experiments?limit=9999&offset=-5").json()
    assert clamped["limit"] <= 200 and clamped["offset"] == 0


def test_experiment_list_status_filter(client):
    _make_experiment(client, "filtered")
    none_pending = client.get("/api/v1/experiments?status=NO_SUCH_STATUS").json()
    assert none_pending["total"] == 0


# --------------------------------------------------------------------------- CORS (D5/T3.5.2)

def test_cors_headers_present_for_allowed_origin(client):
    resp = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_preflight_is_answered(client):
    resp = client.options(
        "/api/v1/transforms/apply",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unlisted_origin(client):
    resp = client.get("/api/v1/health", headers={"Origin": "http://evil.example.com"})
    # The request still succeeds server-side; the browser is what enforces the absence
    # of the header. What matters is that we do NOT echo an unlisted origin.
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ------------------------------------------------------ data source visibility (E2 / D9)

def test_dataset_listing_declares_its_source(client):
    resp = client.get("/api/v1/data/datasets")
    assert resp.status_code == 200
    for ds in resp.json():
        assert ds["source_kind"] in ("netcdf", "simulated")
        # With no .nc files present these must be honestly labelled as simulated,
        # never silently passed off as reanalysis.
        if ds["source_kind"] == "simulated":
            assert ds["is_simulated"] is True
            assert ds["fallback_reason"]


def test_source_info_reports_simulated_fallback_reason():
    MeteorologicalDataAdapter.invalidate_cache()
    info = MeteorologicalDataAdapter.get_source_info("era5_reanalysis")
    assert info["kind"] == "simulated"
    assert "no file at" in info["fallback_reason"]
    assert info["path"] is None


def test_cache_invalidation_reresolves(tmp_path, monkeypatch):
    """D9: a newly added .nc file must be picked up without an API restart."""
    import numpy as np
    import xarray as xr

    MeteorologicalDataAdapter.invalidate_cache()
    monkeypatch.setattr(MeteorologicalDataAdapter, "DATA_DIR", str(tmp_path))

    # Initially simulated.
    assert MeteorologicalDataAdapter.get_source_info("toy_climate_model")["kind"] == "simulated"

    # Write a real file where the adapter looks.
    lats = np.linspace(-10.0, 10.0, 8)
    lons = np.linspace(-10.0, 10.0, 8)
    ds = xr.Dataset(
        {"sst": (["time", "lat", "lon"], np.ones((1, 8, 8), dtype=np.float32))},
        coords={"time": [np.datetime64("2020-01-01")], "lat": lats, "lon": lons},
        attrs={"description": "unit-test file"},
    )
    target = os.path.join(str(tmp_path), "toy_climate_model.nc")
    ds.to_netcdf(target)

    # Same process, no restart: the adapter must now resolve to the file.
    info = MeteorologicalDataAdapter.get_source_info("toy_climate_model")
    assert info["kind"] == "netcdf"
    assert info["path"] == target

    MeteorologicalDataAdapter.invalidate_cache()


def test_benchmark_listing_exposes_known_answers(client):
    """Standard E7: what the platform is proved to get right must be visible in the API."""
    resp = client.get("/api/v1/benchmarks")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 8
    by_name = {b["name"]: b for b in items}
    fbm = by_name["fractional_brownian"]
    assert fbm["is_null"] is True
    assert fbm["known_answer"]["beta_energy_1d"] == 2.4
    assert "4C.alpha" in fbm["gates"]
    assert any(b["is_null"] for b in items)


def test_benchmark_run_separates_the_three_outcomes(client):
    resp = client.post("/api/v1/benchmarks/run?name=white_noise_field")
    assert resp.status_code == 200
    body = resp.json()
    assert body["failed"] == 0
    assert body["passed"] >= 2
    assert body["null_failures"] == []
    assert body["root_seed"] > 0
    outcomes = {c["outcome"] for b in body["benchmarks"] for c in b["checks"]}
    assert outcomes <= {"PASS", "FAIL", "NOT_YET_RUNNABLE"}


def test_benchmark_run_reports_pending_gates(client):
    resp = client.post("/api/v1/benchmarks/run?name=fractional_brownian")
    body = resp.json()
    assert body["not_yet_runnable"] >= 1
    stages = [c["stage"] for b in body["benchmarks"] for c in b["checks"]
              if c["outcome"] == "NOT_YET_RUNNABLE"]
    assert "4C.surrogate_null" in stages


def test_unknown_benchmark_name_is_a_404_not_a_silent_empty_pass(client):
    """A gate that a typo can make vanish is not a gate."""
    resp = client.post("/api/v1/benchmarks/run?name=not_a_benchmark")
    assert resp.status_code == 404
    assert "Available:" in resp.json()["detail"]
