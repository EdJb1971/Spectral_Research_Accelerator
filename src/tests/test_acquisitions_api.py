"""TG10.2 domain-first acquisition catalogue and TG18.1 durable CDS jobs."""

import json
import time

from src.api.main import app
from src.core.cds_job import CDSJobStore
from src.data_layer.cds_source import CDSAcquisitionCancelled, plan_monthly_shards

from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT
from src.core.domain import KNOWN_VIOLATIONS


def _by_name(client):
    response = client.get("/api/v1/acquisitions")
    assert response.status_code == 200
    return {row["name"]: row for row in response.json()["domains"]}


def test_domains_come_before_their_acquisitions(client):
    body = client.get("/api/v1/acquisitions").json()
    assert set(body["shapes"]) == {
        "grid_crop", "profile_query", "lightcurve_query", "channel_table"}
    assert {row["name"] for row in body["domains"]} >= {"reanalysis", "order_book"}
    assert all("acquisitions" in row and "domain_limits" in row for row in body["domains"])


def test_registered_lightcurve_source_is_reachable_only_under_tess(client):
    domains = _by_name(client)
    tess = [item for item in domains["tess_lightcurve"]["acquisitions"]
            if item["shape"] == "lightcurve_query"]
    assert len(tess) == 1
    assert tess[0]["name"] == "mast_tess_spoc"
    assert tess[0]["lightcurve_source"]["metadata_preflight"] is True
    assert all(not any(item["shape"] == "lightcurve_query"
                       for item in domain["acquisitions"])
               for name, domain in domains.items() if name != "tess_lightcurve")


def test_all_registered_grid_sources_are_reachable_under_reanalysis(client):
    domain = _by_name(client)["reanalysis"]
    grids = [item for item in domain["acquisitions"] if item["shape"] == "grid_crop"]
    assert {item["name"] for item in grids} == {
        "era5_0p25_6h", "era5_0p25_1h_full37", "era5_1p5_6h", "era5_0p7_6h",
        "glorys_phy_my_0p083deg_p1d",
    }
    assert all(item["available"] and item["store"]["domain"] == "reanalysis"
               for item in grids)


def test_channel_tables_follow_the_existing_axis_refusal(client):
    domains = _by_name(client)
    order_book = next(item for item in domains["order_book"]["acquisitions"]
                      if item["shape"] == "channel_table")
    reanalysis = next(item for item in domains["reanalysis"]["acquisitions"]
                      if item["shape"] == "channel_table")
    assert order_book["available"] is True
    assert reanalysis["available"] is False
    assert "latitude" in reanalysis["unavailable_reason"]


def test_every_acquisition_carries_limits_and_the_attribution_caveat(client):
    for domain in _by_name(client).values():
        for acquisition in domain["acquisitions"]:
            assert acquisition["domain_limits"] == domain["domain_limits"]
            assert acquisition["domain_limits"]["attribution_caveat"] == \
                DOMAIN_ATTRIBUTION_CAVEAT


def test_every_known_violation_has_a_registered_data_path_not_only_a_declaration(client):
    body = client.get("/api/v1/acquisitions").json()
    assert set(body["violation_coverage"]) == set(KNOWN_VIOLATIONS)
    assert all(body["violation_coverage"][name] for name in KNOWN_VIOLATIONS)
    assert {row["shape"] for row in body["violation_coverage"]["no_natural_cycle"]} == {
        "lightcurve_query"}
    assert {row["shape"] for row in body["violation_coverage"]["non_stationary_support"]} >= {
        "profile_query", "lightcurve_query"}


def test_catalogue_exposes_separate_cds_planning_and_durable_job_execution(client):
    body = client.get("/api/v1/acquisitions").json()
    cds = next(row for row in body["operational_routes"]
               if row["id"] == "era5_cds_regional")

    assert cds["domain"] == "reanalysis"
    assert cds["ui_status"] == "DURABLE_JOB_AVAILABLE"
    assert "confirmed durable browser job" in cds["execution"]
    assert {"variables", "date range", "pressure levels", "time chunk"}.issubset(
        cds["configuration"])
    assert "without network use" in cds["reason"]
    assert "completion-only acquisition record" in cds["reason"]


def test_cds_browser_planner_reproduces_the_accepted_six_year_request_without_network(client):
    capabilities = client.get("/api/v1/data/cds")
    assert capabilities.status_code == 200
    assert capabilities.json()["planner_network_used"] is False
    assert capabilities.json()["execution_status"] == "NETWORK_DISABLED"

    response = client.post("/api/v1/data/cds/plan", json=capabilities.json()["defaults"])
    assert response.status_code == 200
    body = response.json()
    assert body["network_used"] is False
    assert body["execution_status"] == "NETWORK_DISABLED"
    assert len(body["monthly_shards"]) == 72
    assert body["storage_estimate"]["frames"] == 8764
    assert body["request_sha256"] == \
        "297204dd6d576828dece605bb4a94ce96c35b6cce0a8152f85b1174e853eb5aa"
    assert body["submission_confirmation"]["confirm_request_sha256"] == body["request_sha256"]


def _completed_acquisition(spec, _directory, **kwargs):
    shards = plan_monthly_shards(spec)
    completed = {}
    for index, shard in enumerate(shards, 1):
        kwargs["progress_callback"]({"completed_shards": index,
                                     "total_shards": len(shards), "current_shard": None})
        completed[shard.filename] = {
            "sha256": "%064x" % index, "bytes": 100 + index,
            "request_sha256": shard.request_sha256,
        }
    return {"completed_shards": completed,
            "run": {"downloaded_shards": len(shards), "resumed_shards": 0,
                    "total_bytes": sum(item["bytes"] for item in completed.values()),
                    "complete": True}}


def _wait_for_job(client, job_id, state):
    for _ in range(200):
        body = client.get("/api/v1/data/cds/jobs/%s" % job_id).json()
        if body["state"] == state:
            return body
        time.sleep(0.01)
    raise AssertionError("CDS job did not reach %s; last state was %s" % (state, body["state"]))


def test_cds_job_requires_exact_confirmation_network_opt_in_and_server_storage(
        client, tmp_path, monkeypatch):
    app.state.cds_job_dir = tmp_path / "cds-jobs"
    app.state.cds_job_runner = _completed_acquisition
    plan = client.post("/api/v1/data/cds/plan",
                       json=client.get("/api/v1/data/cds").json()["defaults"]).json()
    payload = {"request": plan["request"], "confirm_request_sha256": "0" * 64,
               "confirm_network_access": True}
    assert client.post("/api/v1/data/cds/jobs", json=payload).status_code == 409
    payload["confirm_request_sha256"] = plan["request_sha256"]
    payload["confirm_network_access"] = False
    assert client.post("/api/v1/data/cds/jobs", json=payload).status_code == 409
    payload["confirm_network_access"] = "true"
    assert client.post("/api/v1/data/cds/jobs", json=payload).status_code == 422
    payload["confirm_network_access"] = True
    assert client.post("/api/v1/data/cds/jobs", json=payload).status_code == 409

    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    submitted = client.post("/api/v1/data/cds/jobs", json=payload)
    assert submitted.status_code == 202
    job = _wait_for_job(client, submitted.json()["job_id"], "COMPLETE")
    assert job["storage"]["ownership"] == "SERVER_MANAGED"
    assert job["storage"]["client_path_accepted"] is False
    assert "downloads" not in str(job)
    assert job["storage_preflight"]["status"] == "READY"
    assert job["progress"]["fraction"] == 1.0

    record = client.get("/api/v1/data/cds/jobs/%s/record" % job["job_id"])
    assert record.status_code == 200
    assert record.json()["record_sha256"]
    assert record.json()["completed_shards"] == 72
    journal_path = tmp_path / "cds-jobs" / job["job_id"] / "job.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["acquisition_record"]["total_bytes"] += 1
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    assert client.get("/api/v1/data/cds/jobs/%s/record" % job["job_id"]).status_code == 502
    app.state.cds_job_runner = None
    app.state.cds_job_dir = None


def test_cds_job_cancellation_is_cooperative_and_verified_work_is_resumable(
        client, tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    app.state.cds_job_dir = tmp_path / "cds-jobs"

    def waits_for_cancel(spec, _directory, **kwargs):
        kwargs["progress_callback"]({"completed_shards": 1, "total_shards": 72,
                                     "current_shard": "in-flight.nc"})
        for _ in range(500):
            if kwargs["cancellation_requested"]():
                raise CDSAcquisitionCancelled("cancelled in test")
            time.sleep(0.002)
        raise AssertionError("test job was not cancelled")

    app.state.cds_job_runner = waits_for_cancel
    defaults = client.get("/api/v1/data/cds").json()["defaults"]
    plan = client.post("/api/v1/data/cds/plan", json=defaults).json()
    response = client.post("/api/v1/data/cds/jobs", json={
        "request": defaults, "confirm_request_sha256": plan["request_sha256"],
        "confirm_network_access": True})
    running = _wait_for_job(client, response.json()["job_id"], "RUNNING")
    cancelled = client.post("/api/v1/data/cds/jobs/%s/cancel" % running["job_id"],
                            json={"reason": "operator requested stop"})
    assert cancelled.status_code == 200
    stopped = _wait_for_job(client, running["job_id"], "CANCELLED")
    assert stopped["resumable"] is True
    assert stopped["progress"]["completed_shards"] == 1
    assert client.get("/api/v1/data/cds/jobs/%s/record" % running["job_id"]).status_code == 409
    app.state.cds_job_runner = None
    app.state.cds_job_dir = None


def test_a_server_restart_marks_active_cds_work_interrupted_for_explicit_resume(
        client, tmp_path, monkeypatch):
    store = CDSJobStore(tmp_path / "jobs")
    from src.api.cds import CDSPlanRequest, _spec
    job, _ = store.submit(_spec(CDSPlanRequest()))
    assert job["state"] == "QUEUED"
    journal_path = tmp_path / "jobs" / job["job_id"] / "job.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "RUNNING"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    app.state.cds_job_dir = tmp_path / "jobs"
    app.state.cds_job_runner = _completed_acquisition
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    resumed = client.post("/api/v1/data/cds/jobs/%s/resume" % job["job_id"])
    assert resumed.status_code == 202
    complete = _wait_for_job(client, job["job_id"], "COMPLETE")
    assert complete["attempts"] == 1
    assert complete["acquisition_record"] is not None
    app.state.cds_job_runner = None
    app.state.cds_job_dir = None


def test_cds_browser_planner_refuses_bounds_that_would_be_server_snapped(client):
    request = client.get("/api/v1/data/cds").json()["defaults"]
    request["lat_min"] = -59.9
    response = client.post("/api/v1/data/cds/plan", json=request)

    assert response.status_code == 400
    assert "bounds/grid" in response.json()["detail"]
    assert "server-side snapping would change the frozen crop" in response.json()["detail"]


def test_gridded_acquisitions_publish_researcher_facing_source_identity(client):
    body = client.get("/api/v1/acquisitions").json()
    reanalysis = next(row for row in body["domains"] if row["name"] == "reanalysis")
    grids = [row for row in reanalysis["acquisitions"] if row["shape"] == "grid_crop"]

    assert grids
    assert all(row["label"] and row["provider"] and row["product_family"] for row in grids)
    assert any("WeatherBench 2" in row["provider"] for row in grids)
    assert any("Copernicus Marine" in row["provider"] for row in grids)
