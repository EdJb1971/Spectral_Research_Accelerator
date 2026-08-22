"""T4C.6 gate-job acceptance without representing synthetic data as ERA5 evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis_engine.cross_scale import GateProtocol
from src.analysis_engine.gate_run import (
    GateStudyPlan,
    load_gate_plan,
    load_gate_receipt,
    preflight_cached_gate,
    run_cached_gate,
    save_gate_plan,
)
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.zarr_source import CropSpec, materialise

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


def _cache(tmp_path):
    source = tmp_path / "source.zarr"
    rng = np.random.default_rng(4406)
    times = np.arange("2020-01-01", "2020-01-31", np.timedelta64(6, "h"),
                      dtype="datetime64[ns]")[:120]
    latitude = np.linspace(-32.0, -47.75, 64)
    longitude = np.linspace(160.0, 175.75, 64)
    values = rng.standard_normal((120, 1, 64, 64)).astype(np.float32)
    # A deterministic seasonal component makes the train-fitted climatology non-trivial.
    values += np.sin(np.arange(120, dtype=np.float32)[:, None, None, None]
                     * np.float32(2.0 * np.pi / 4.0))
    dataset = xr.Dataset(
        {"t": (("time", "level", "latitude", "longitude"), values,
               {"units": "K"})},
        coords={"time": times, "level": [850],
                "latitude": latitude, "longitude": longitude})
    dataset.chunk({"time": 4, "level": 1, "latitude": 64,
                   "longitude": 64}).to_zarr(source, mode="w", consolidated=True)
    crop = CropSpec(
        store=str(source), variables=("t",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-30T18:00:00",
        lat_min=-47.75, lat_max=-32.0, lon_min=160.0, lon_max=175.75,
        levels=(850,), n_levels_analysis=2)
    cache = tmp_path / "cache"
    materialise(crop, cache_dir=str(cache), time_chunk=4, check_size=False)
    return crop, cache


def _plan(crop):
    protocol = GateProtocol(
        study_id="synthetic-gate-acceptance", n_scales=2, lags=(1,),
        expected_frames=120, cadence_seconds=21600.0, train_ratio=0.6,
        embargo_frames=1, estimator="transfer_entropy", measure="energy_density",
        bins=2, n_surrogates=59, alpha=0.05,
        correction="benjamini_yekutieli", seed=4406)
    return GateStudyPlan(
        study_id=protocol.study_id, evidence_role="synthetic_acceptance",
        crop=crop, protocol=protocol, variable="t", level_hpa=850,
        transform_family="swt", wavelet="db2", advection_speed_m_s=50.0,
        climatology_harmonics=1)


def test_gate_job_is_frozen_bounded_atomic_and_synthetically_honest(tmp_path):
    crop, cache = _cache(tmp_path)
    plan = _plan(crop)
    plan_path = tmp_path / "gate-plan.json"
    assert save_gate_plan(plan_path, plan) == plan.fingerprint()
    assert load_gate_plan(plan_path) == plan
    with pytest.raises(FileExistsError):
        save_gate_plan(plan_path, plan)

    preflight = preflight_cached_gate(plan, cache_dir=str(cache))
    assert preflight["status"] == "READY"
    assert preflight["network_used"] is False
    assert preflight["split"]["train_frames"] == 72
    assert preflight["split"]["test_frames"] == 47
    assert preflight["support_floor"]["enforced"] is True
    assert preflight["minimum_valid_parent_pixels"] == 128
    assert preflight["valid_parent_interiors"][-1]["valid_parent_shape"] == [54, 54]

    receipt_path = tmp_path / "gate-receipt.json"
    receipt = run_cached_gate(
        plan, cache_dir=str(cache), receipt_path=receipt_path)
    assert receipt["scientific_verdict"] == "NOT_ESTABLISHED"
    assert receipt["gate"]["verdict"] in ("PASS", "FAIL")
    assert receipt["plan_sha256"] == plan.fingerprint()
    assert receipt["signatures"]["test_thresholds_fitted_on"] == "train"
    assert receipt["climatology"]["fitted_on_all_frames"] is False
    assert receipt["climatology"]["source_frames_resident"] == 1
    assert receipt["preflight"]["source"]["network_used"] is False
    assert load_gate_receipt(receipt_path)["receipt_sha256"] == receipt["receipt_sha256"]
    with pytest.raises(FileExistsError):
        run_cached_gate(plan, cache_dir=str(cache), receipt_path=receipt_path)

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["scientific_verdict"] = "PASS"
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DataSourceError, match="does not authenticate"):
        load_gate_receipt(receipt_path)

    with pytest.raises(InvalidParameterError, match="cannot be relabelled"):
        GateStudyPlan(
            **{**plan.__dict__, "evidence_role": "real_era5_gate"})
