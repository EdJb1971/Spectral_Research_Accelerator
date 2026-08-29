"""TG12.2b-d: immutable profiles, declared reductions and the bounded Argo seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from extensions.argo_float import ARGO
from src.api.main import app
from src.core.errors import InvalidParameterError, UnknownNameError
from src.data_layer.argo_source import (acquire_profiles, inspect_profile_query,
                                        load_collection, persist_collection)
from src.data_layer.profile_reductions import PROFILE_REDUCTIONS, reduce_profiles
from src.data_layer.profiles import ProfileSpec
from src.data_layer.tabular_source import assert_domain_admits_channel_table


FIELDS = ("fileNumber", "platform_number", "cycle_number", "direction", "time",
          "latitude", "longitude")
VALUE_FIELDS = FIELDS + (
    "pres", "pres_qc", "pres_adjusted", "pres_adjusted_qc",
    "temp", "temp_qc", "temp_adjusted", "temp_adjusted_qc",
    "psal", "psal_qc", "psal_adjusted", "psal_adjusted_qc")


def spec(**changes):
    values = dict(time_start="2026-01-01", time_end="2026-02-01",
                  lat_min=-46, lat_max=-34, lon_min=165, lon_max=180,
                  pressure_min_dbar=0, pressure_max_dbar=200,
                  variables=("temperature",), max_profiles=20)
    values.update(changes)
    return ProfileSpec(**values)


def _fixture():
    profiles = [
        ("5900001", 1, "2026-01-01T00:00:00Z", -40.0, 170.0),
        ("5900002", 1, "2026-01-02T03:00:00Z", -41.0, 171.0),
        ("5900001", 2, "2026-01-11T01:00:00Z", -40.2, 170.3),
        ("5900002", 2, "2026-01-12T06:00:00Z", -41.1, 171.4),
    ]
    header = []
    values = []
    for platform, cycle, when, lat, lon in profiles:
        head = [platform, platform, str(cycle), "A", when, str(lat), str(lon)]
        header.append(",".join(head))
        for pressure, temperature in ((0, 20 + cycle), (100, 10 + cycle), (200, 5 + cycle)):
            values.append(",".join(head + [
                str(pressure), "1", str(pressure), "1",
                str(temperature), "1", str(temperature), "1",
                "35", "1", "35", "1",
            ]))
    return ("\n".join(header) + "\n").encode(), ("\n".join(values) + "\n").encode()


def fetcher(url: str, _limit: int) -> bytes:
    header, values = _fixture()
    return values if "pres%" in url or ",pres," in url else header


def test_profile_spec_identity_is_machine_independent_and_axes_remain_a_scatter():
    first = spec()
    second = spec(time_start="2026-01-01T00:00:00Z", time_end="2026-02-01T00:00:00+00:00")
    assert first.request_sha256() == second.request_sha256()
    collection = acquire_profiles(first, fetch=fetcher)
    assert collection.n_profiles == 4
    assert len(collection.pressure_dbar) == 4
    assert all(vector.shape == (3,) for vector in collection.pressure_dbar)
    assert collection.describe()["n_platforms"] == 2


def test_query_preflight_counts_profiles_before_values_and_refuses_the_cap():
    calls = []

    def observed(url, limit):
        calls.append((url, limit))
        return fetcher(url, limit)

    plan = inspect_profile_query(spec(), fetch=observed)
    assert plan["candidate_profiles"] == 4
    assert plan["candidate_platforms"] == 2
    assert len(calls) == 1 and "distinct()" in calls[0][0]
    with pytest.raises(InvalidParameterError, match="max_profiles=2"):
        acquire_profiles(spec(max_profiles=2), fetch=observed)
    assert len(calls) == 2, "the refused acquisition must not issue the level-value request"


def test_qc_adjusted_preference_and_observed_invalid_are_distinct_from_absence():
    header, values = _fixture()
    rows = values.decode().splitlines()
    parts = rows[1].split(",")
    parts[14], parts[15], parts[12], parts[13] = "999", "4", "123", "4"
    rows[1] = ",".join(parts)

    def changed(url, _limit):
        return ("\n".join(rows) + "\n").encode() if ",pres," in url else header

    collection = acquire_profiles(spec(), fetch=changed)
    assert np.isnan(collection.measures["temperature"][0][1])
    reduced = reduce_profiles(collection, ARGO, "per_float_at_pressure", {
        "variable": "temperature", "pressure_dbar": 0,
        "max_interpolation_gap_dbar": 25})
    assert reduced.series.present.dtype == np.dtype(bool)
    assert np.any(~reduced.series.present)


def test_per_float_reduction_preserves_both_violations_and_records_axis_consumption():
    collection = acquire_profiles(spec(), fetch=fetcher)
    result = reduce_profiles(collection, ARGO, "per_float_at_pressure", {
        "variable": "temperature", "pressure_dbar": 100,
        "max_interpolation_gap_dbar": 25})
    assert set(result.declaration.violations) == {"irregular_sampling",
                                                  "non_stationary_support"}
    assert "aggregated_values" not in result.declaration.violations
    assert result.series.channels == ("5900001", "5900002") or \
        list(result.series.channels) == ["5900001", "5900002"]
    assert result.reduction["consumes_axes"] == ["pressure", "latitude", "longitude"]
    assert result.series.channel_records[0]["present_count"] == 2


def test_depth_bin_reduction_declares_aggregation_and_changes_content_identity():
    collection = acquire_profiles(spec(), fetch=fetcher)
    per_float = reduce_profiles(collection, ARGO, "per_float_at_pressure", {
        "variable": "temperature", "pressure_dbar": 100,
        "max_interpolation_gap_dbar": 25})
    bins = reduce_profiles(collection, ARGO, "depth_bin_mean", {
        "variable": "temperature", "bin_edges_dbar": [0, 50, 150, 200]})
    assert "irregular_sampling" in bins.declaration.violations
    assert "aggregated_values" in bins.declaration.violations
    assert "non_stationary_support" not in bins.declaration.violations
    assert bins.reduction["aggregation_window"] == {
        "axis": "pressure", "units": "dbar", "edges": [0.0, 50.0, 150.0, 200.0]}
    assert per_float.reduction["reduction_sha256"] != bins.reduction["reduction_sha256"]


def test_profile_collection_publication_round_trips_canonical_bytes(tmp_path: Path):
    collection = acquire_profiles(spec(), fetch=fetcher)
    receipt = persist_collection(collection, tmp_path)
    loaded = load_collection(receipt["collection_sha256"], tmp_path)
    assert loaded.collection_sha256() == collection.collection_sha256()
    assert persist_collection(collection, tmp_path)["publication"] == "already-present-identical"


def test_argo_parent_still_refuses_a_flat_channel_table_by_all_unsatisfied_axes():
    with pytest.raises(InvalidParameterError) as caught:
        assert_domain_admits_channel_table(ARGO)
    message = str(caught.value)
    assert all(name in message for name in ("latitude", "longitude", "pressure"))


def test_profile_reduction_registry_is_discoverable_and_misspellings_are_refused():
    assert PROFILE_REDUCTIONS.names() == ["depth_bin_mean", "per_float_at_pressure"]
    with pytest.raises(UnknownNameError):
        reduce_profiles(acquire_profiles(spec(), fetch=fetcher), ARGO, "float_mean", {})


def test_profile_api_contract_and_real_refusal_are_visible(monkeypatch, tmp_path):
    import src.api.profiles as api

    collection = acquire_profiles(spec(), fetch=fetcher)
    monkeypatch.setattr(api, "inspect_profile_query",
                        lambda requested: {"candidate_profiles": 4,
                                           "request_sha256": requested.request_sha256()})
    monkeypatch.setattr(api, "acquire_profiles", lambda requested: collection)
    monkeypatch.setattr(api, "persist_collection",
                        lambda value: persist_collection(value, tmp_path))
    client = TestClient(app)
    capabilities = client.get("/api/v1/profiles")
    assert capabilities.status_code == 200
    assert {row["name"] for row in capabilities.json()["reductions"]} == \
        {"depth_bin_mean", "per_float_at_pressure"}
    request = {
        "spec": spec().canonical(),
        "reduction": {"name": "per_float_at_pressure", "configuration": {
            "variable": "temperature", "pressure_dbar": 100,
            "max_interpolation_gap_dbar": 25}},
    }
    # Convert canonical axis-pair representation back to the HTTP shape.
    request["spec"] = {
        "source": "argo_gdac_erddap", "time_start": "2026-01-01",
        "time_end": "2026-02-01", "lat_min": -46, "lat_max": -34,
        "lon_min": 165, "lon_max": 180, "pressure_min_dbar": 0,
        "pressure_max_dbar": 200, "variables": ["temperature"], "max_profiles": 20}
    response = client.post("/api/v1/profiles/acquire", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_readiness"]["decision"] == "refuse_frame_lags"
    assert set(body["analysis_readiness"]["basis"]) == {
        "irregular_sampling", "non_stationary_support"}
    assert body["preview"]["channels"][0]["present_count"] == 2


@pytest.mark.live
def test_live_argo_gdac_small_query_when_network_is_explicitly_enabled():
    import os
    if os.getenv("SPECTRALEARTH_RUN_LIVE_ARGO") != "1":
        pytest.skip("set SPECTRALEARTH_RUN_LIVE_ARGO=1 for the bounded live acceptance")
    live = ProfileSpec(time_start="2026-07-01", time_end="2026-08-01",
                       lat_min=-46, lat_max=-34, lon_min=165, lon_max=180,
                       pressure_min_dbar=0, pressure_max_dbar=200,
                       variables=("temperature",), max_profiles=100)
    plan = inspect_profile_query(live)
    assert 2 <= plan["candidate_profiles"] <= 100
    collection = acquire_profiles(live)
    result = reduce_profiles(collection, ARGO, "per_float_at_pressure", {
        "variable": "temperature", "pressure_dbar": 100,
        "max_interpolation_gap_dbar": 25})
    counts = [record["present_count"] for record in result.series.channel_records]
    assert result.series.n_channels >= 2
    assert len(set(counts)) > 1, "real floats must contribute different sample counts"
    assert np.any(~result.series.present)
    assert set(result.declaration.violations) == {
        "irregular_sampling", "non_stationary_support"}
    assert collection.response_sha256 != hashlib.sha256(b"").hexdigest()
