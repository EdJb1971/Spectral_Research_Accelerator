"""TG10.2 domain-first acquisition catalogue."""

from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT


def _by_name(client):
    response = client.get("/api/v1/acquisitions")
    assert response.status_code == 200
    return {row["name"]: row for row in response.json()["domains"]}


def test_domains_come_before_their_acquisitions(client):
    body = client.get("/api/v1/acquisitions").json()
    assert set(body["shapes"]) == {"grid_crop", "profile_query", "channel_table"}
    assert {row["name"] for row in body["domains"]} >= {"reanalysis", "order_book"}
    assert all("acquisitions" in row and "domain_limits" in row for row in body["domains"])


def test_all_existing_era5_stores_are_reachable_under_reanalysis(client):
    domain = _by_name(client)["reanalysis"]
    grids = [item for item in domain["acquisitions"] if item["shape"] == "grid_crop"]
    assert {item["name"] for item in grids} == {
        "era5_0p25_6h", "era5_0p25_1h_full37", "era5_1p5_6h", "era5_0p7_6h"
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
