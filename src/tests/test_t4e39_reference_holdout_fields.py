import hashlib

import numpy as np
import pytest

from src.benchmarks.reference_holdout import DEFAULT_DECLARATION
from src.benchmarks.reference_holdout_fields import (
    acquire_holdout_fields,
    holdout_decision,
    join_field_segments,
    plan_holdout_fields,
)
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_pressure_level_exact import (
    CDSPressureLevelExactRequest,
    plan_pressure_level_exact_shards,
)

xr = pytest.importorskip("xarray")


def test_field_plan_binds_census_to_48_exact_offline_shards():
    plan = plan_holdout_fields()
    assert plan["selected_storms"] == 12
    assert plan["total_shards"] == 48
    assert [(item["field"], item["segment"], len(item["shards"]))
            for item in plan["requests"]] == [
                ("msl", "parent", 12), ("msl", "complement", 12),
                ("vo", "parent", 12), ("vo", "complement", 12)]
    assert plan["declaration_sha256"] == hashlib.sha256(
        DEFAULT_DECLARATION.read_bytes()).hexdigest()
    assert plan["record_opened"] is False and plan["network_used"] is False


def test_pressure_shards_request_one_exact_850_hpa_time():
    spec = CDSPressureLevelExactRequest(
        variables=("vo",), pressure_levels=(850,),
        timestamps=("2023-01-01 06:00:00",),
        lat_min=-2, lat_max=-1, lon_min=179, lon_max=180, grid_degrees=0.5)
    shard = plan_pressure_level_exact_shards(spec)[0]
    assert shard.request["variable"] == ["vorticity"]
    assert shard.request["pressure_level"] == ["850"]
    assert shard.request["time"] == ["06:00"]
    with pytest.raises(InvalidParameterError, match="pressure levels"):
        CDSPressureLevelExactRequest(
            variables=("vo",), pressure_levels=(999,), timestamps=spec.timestamps,
            lat_min=-2, lat_max=-1, lon_min=179, lon_max=180)


def test_experiment_and_general_network_gates_precede_client_construction(monkeypatch):
    with pytest.raises(DataSourceError, match="experiment-specific"):
        acquire_holdout_fields(explicit_network_authorisation=False)
    monkeypatch.delenv("SPECTRALEARTH_ALLOW_NETWORK", raising=False)
    with pytest.raises(DataSourceError, match="network-disabled"):
        acquire_holdout_fields(explicit_network_authorisation=True)


def _dataset(field, spec, timestamp, values):
    dims = ("valid_time", "latitude", "longitude")
    coords = {
        "valid_time": [np.datetime64(timestamp.replace(" ", "T"))],
        "latitude": np.linspace(spec.lat_max, spec.lat_min, values.shape[0]),
        "longitude": np.linspace(spec.lon_min, spec.lon_max, values.shape[1]),
    }
    if field == "vo":
        dims = ("valid_time", "pressure_level", "latitude", "longitude")
        coords["pressure_level"] = [850]
        values = values[None, :, :]
    return xr.Dataset({field: (dims, values[None, :, :])}, coords=coords)


@pytest.mark.parametrize("field", ["msl", "vo"])
def test_each_field_seam_is_measured_and_deduplicated(field):
    plan = plan_holdout_fields()
    items = [item for item in plan["requests"] if item["field"] == field]
    from src.benchmarks.reference_holdout_fields import _spec_from_plan
    specs = [_spec_from_plan(item) for item in items]
    timestamp = plan["timestamps"][0]
    left = (np.arange(161 * 161).reshape(161, 161) + 1000).astype(np.float32)
    right = (np.arange(161 * 161).reshape(161, 161) + 2000).astype(np.float32)
    right[:, 0] = left[:, -1]
    joined, seam = join_field_segments(
        _dataset(field, specs[0], timestamp, left),
        _dataset(field, specs[1], timestamp, right),
        field, specs[0], specs[1], timestamp)
    assert joined.shape == (1, 161, 321)
    assert seam["passed"] is True and seam["maximum_absolute_difference"] == 0


def test_holdout_decision_keeps_ties_and_refusals_in_strict_majority_denominator():
    vertical = [{"classification": "VERTICAL_QUANTITY_SEPARATION_CANDIDATE"}] * 7
    other = [{"classification": "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE"}] * 3
    neither = [{"classification": "EXACT_TIE"}, {"classification": "REFUSED"}]
    decision = holdout_decision(vertical + other + neither)
    assert decision["population_denominator"] == 12
    assert decision["strict_majority_needed"] == 7
    assert decision["outcome"] == "HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE"
    assert decision["is_acceptance_verdict"] is False
