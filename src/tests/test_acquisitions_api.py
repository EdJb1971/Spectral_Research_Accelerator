"""TG10.2 domain-first acquisition catalogue."""

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


def test_catalogue_exposes_the_non_http_cds_route_instead_of_hiding_it(client):
    body = client.get("/api/v1/acquisitions").json()
    cds = next(row for row in body["operational_routes"]
               if row["id"] == "era5_cds_regional")

    assert cds["domain"] == "reanalysis"
    assert cds["ui_status"] == "PLANNER_NOT_EXPOSED"
    assert cds["execution"] == "bounded resumable CLI acquisition"
    assert {"variables", "date range", "pressure levels", "time chunk"}.issubset(
        cds["configuration"])
    assert "no HTTP route" in cds["reason"]


def test_gridded_acquisitions_publish_researcher_facing_source_identity(client):
    body = client.get("/api/v1/acquisitions").json()
    reanalysis = next(row for row in body["domains"] if row["name"] == "reanalysis")
    grids = [row for row in reanalysis["acquisitions"] if row["shape"] == "grid_crop"]

    assert grids
    assert all(row["label"] and row["provider"] and row["product_family"] for row in grids)
    assert any("WeatherBench 2" in row["provider"] for row in grids)
    assert any("Copernicus Marine" in row["provider"] for row in grids)
