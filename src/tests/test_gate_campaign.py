"""Offline acceptance for the frozen two-stage T4C.6 acquisition campaign."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.analysis_engine.cross_scale import GateProtocol
from src.analysis_engine.gate_campaign import (
    GateCampaign,
    load_gate_campaign,
    main,
    preflight_gate_campaign,
    review_gate_campaign,
    save_gate_campaign,
)
from src.analysis_engine.gate_run import GateStudyPlan
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_source import CDSRegionalRequest
from src.data_layer.zarr_source import CropSpec


def _request(**overrides):
    values = dict(
        variables=("t",), date_start="2020-01-01", date_end="2020-01-30",
        hours_utc=(0, 6, 12, 18), lat_min=-60.0, lat_max=-20.0,
        lon_min=140.0, lon_max=180.0, pressure_levels=(850,),
        grid_degrees=0.25, n_levels_analysis=2)
    values.update(overrides)
    return CDSRegionalRequest(**values)


def _campaign(**overrides):
    full = overrides.pop("full_acquisition", _request())
    canary = overrides.pop("canary_acquisition", _request(date_end="2020-01-02"))
    protocol = GateProtocol(
        study_id="nz-era5-cross-scale-v1", n_scales=2, lags=(1,),
        expected_frames=120, cadence_seconds=21600.0, train_ratio=0.6,
        embargo_frames=1, estimator="transfer_entropy", measure="energy_density",
        bins=2, n_surrogates=59, alpha=0.05,
        correction="benjamini_yekutieli", seed=4406)
    plan = overrides.pop("gate_plan", GateStudyPlan(
        study_id=protocol.study_id, evidence_role="real_era5_gate",
        crop=full.to_crop_spec(), protocol=protocol, variable="t", level_hpa=850,
        transform_family="swt", wavelet="db2", advection_speed_m_s=50.0,
        climatology_harmonics=1))
    overlap = overrides.pop("weatherbench_overlap", CropSpec(
        store="era5_0p25_6h", variables=("temperature",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-02T18:00:00",
        lat_min=-60.0, lat_max=-20.0, lon_min=140.0, lon_max=180.0,
        levels=(850,), n_levels_analysis=2))
    expected_overlap_frames = overrides.pop("expected_overlap_frames", 8)
    return GateCampaign(
        campaign_id="nz-era5-cross-scale-campaign-v1", full_acquisition=full,
        canary_acquisition=canary, weatherbench_overlap=overlap,
        expected_overlap_frames=expected_overlap_frames, gate_plan=plan, **overrides)


def test_campaign_is_exact_hashable_atomic_and_tamper_detecting(tmp_path):
    campaign = _campaign()
    unknown = campaign.to_mapping()
    unknown["full_acquisition"]["typo"] = "ignored science"
    with pytest.raises(InvalidParameterError, match="unknown"):
        GateCampaign.from_mapping(unknown)
    path = tmp_path / "campaign.json"
    assert save_gate_campaign(path, campaign) == campaign.fingerprint()
    assert load_gate_campaign(path) == campaign
    with pytest.raises(FileExistsError):
        save_gate_campaign(path, campaign)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["campaign"]["expected_overlap_frames"] = 7
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises((DataSourceError, InvalidParameterError)):
        load_gate_campaign(path)


def test_checked_in_real_campaign_is_an_authenticated_preregistration(capsys):
    path = (Path(__file__).parents[2] / "campaigns"
            / "t4c6_nz_era5_temperature_850_v1.json")
    campaign = load_gate_campaign(path)
    assert campaign.fingerprint() == (
        "84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975")
    review = review_gate_campaign(campaign)
    design = review["scientific_design"]
    assert review["network_used"] is False
    assert design["primary_analysis"] == {
        "variable": "t", "level_hpa": 850.0, "transform_family": "swt",
        "transform_config": {"levels": 3, "wavelet": "db2", "mode": "periodic"},
        "measure": "energy_density", "estimator": "transfer_entropy",
        "lags_frames": [3, 4, 5, 6, 7, 8],
        "lags_hours": [18.0, 24.0, 30.0, 36.0, 42.0, 48.0],
        "multiple_comparison_correction": "benjamini_yekutieli", "alpha": 0.05,
    }
    assert design["full_frames"] == 7304
    assert design["calendar_split"] == {
        "train_start": "2018-01-01T00:00:00.000000000",
        "train_end": "2020-12-31T06:00:00.000000000",
        "embargo_start": "2020-12-31T12:00:00.000000000",
        "embargo_end": "2021-01-02T06:00:00.000000000",
        "test_start": "2021-01-02T12:00:00.000000000",
        "test_end": "2022-12-31T18:00:00.000000000",
    }
    assert design["hypothesis_family_size"] == 36
    assert design["power"]["surrogates_required"] == 3005
    assert design["power"]["can_reject_after_correction"] is True
    assert min(design["primary_analysis"]["lags_frames"]) == max(
        item["floor_frames"] for item in
        design["physical_support_floor"]["floors"])

    assert main(["review", "--campaign", str(path)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["campaign_sha256"] == campaign.fingerprint()
    assert emitted["scientific_design"] == design


@pytest.mark.parametrize("change,match", [
    ({"canary_acquisition": _request(date_start="2019-12-31", date_end="2020-01-02")},
     "canary timestamps"),
    ({"weatherbench_overlap": CropSpec(
        store="era5_1p5_6h", variables=("temperature",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-02T18:00:00",
        lat_min=-60.0, lat_max=-20.0, lon_min=140.0, lon_max=180.0,
        levels=(850,), n_levels_analysis=2)}, "exact CDS grid"),
    ({"expected_overlap_frames": 7}, "expected_overlap_frames"),
])
def test_campaign_refuses_canary_overlap_or_frame_drift(change, match):
    with pytest.raises(InvalidParameterError, match=match):
        _campaign(**change)


def test_campaign_refuses_bad_geometry_or_physical_lags_before_transfer():
    small = _request(lat_min=-46.0, lat_max=-45.0, lon_min=170.0, lon_max=171.0)
    small_canary = _request(
        date_end="2020-01-02", lat_min=-46.0, lat_max=-45.0,
        lon_min=170.0, lon_max=171.0)
    small_overlap = CropSpec(
        store="era5_0p25_6h", variables=("temperature",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-02T18:00:00",
        lat_min=-46.0, lat_max=-45.0, lon_min=170.0, lon_max=171.0,
        levels=(850,), n_levels_analysis=2)
    with pytest.raises(InvalidParameterError, match="valid parent-grid pixels"):
        _campaign(full_acquisition=small, canary_acquisition=small_canary,
                  weatherbench_overlap=small_overlap)

    full = _request()
    protocol = GateProtocol(
        study_id="nz-era5-cross-scale-v1", n_scales=2, lags=(1,),
        expected_frames=120, cadence_seconds=21600.0, train_ratio=0.6,
        embargo_frames=1, estimator="transfer_entropy", measure="energy_density",
        bins=2, n_surrogates=59, alpha=0.05,
        correction="benjamini_yekutieli", seed=4406)
    slow_plan = GateStudyPlan(
        study_id=protocol.study_id, evidence_role="real_era5_gate",
        crop=full.to_crop_spec(), protocol=protocol, variable="t", level_hpa=850,
        transform_family="swt", wavelet="db2", advection_speed_m_s=10.0,
        climatology_harmonics=1)
    with pytest.raises(InvalidParameterError, match="physical support floor"):
        _campaign(gate_plan=slow_plan)


def test_preflight_aggregates_all_storage_and_never_constructs_a_client(
        tmp_path, monkeypatch):
    campaign = _campaign()
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.shutil.disk_usage",
        lambda path: SimpleNamespace(total=10**12, used=0, free=10**12))
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.importlib.util.find_spec",
        lambda name: object())
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign._credential_configuration",
        lambda: {"configuration_present": True, "configuration_source": "test sentinel",
                 "secret_values_inspected": False,
                 "remote_validity_or_licence_acceptance_proven": False})
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    report = preflight_gate_campaign(
        campaign, full_download_dir=tmp_path / "full-download",
        canary_download_dir=tmp_path / "canary-download", cache_dir=tmp_path / "cache",
        independent_cache_dir=tmp_path / "independent", minimum_free_reserve_bytes=0)
    assert report["status"] == "READY_FOR_CANARY"
    assert report["network_used"] is False and report["client_constructed"] is False
    assert len(report["storage"]["volumes"]) == 1
    assert set(report["storage"]["volumes"][0]["roles"]) == {
        "full_download", "full_cache", "canary_download", "canary_cache",
        "weatherbench_overlap_cache"}
    assert report["mandatory_order"][2] == "require_canary_overlap_PASS_or_stop"
    assert report["scientific_design"]["grid_shape"] == [161, 161]
    assert report["scientific_design"]["valid_parent_interiors"][-1][
        "valid_parent_shape"] == [151, 151]
    assert report["scientific_design"]["physical_support_floor"]["enforced"] is True


def test_cli_freeze_and_preflight_are_machine_readable_and_zero_network(
        tmp_path, monkeypatch, capsys):
    campaign = _campaign()
    design, frozen = tmp_path / "design.json", tmp_path / "campaign.json"
    design.write_text(json.dumps(campaign.to_mapping()), encoding="utf-8")
    assert main(["freeze", "--design", str(design), "--out", str(frozen)]) == 0
    assert json.loads(capsys.readouterr().out)["campaign_sha256"] == campaign.fingerprint()
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.shutil.disk_usage",
        lambda path: SimpleNamespace(total=10**12, used=0, free=10**12))
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.importlib.util.find_spec",
        lambda name: None)
    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign._credential_configuration",
        lambda: {"configuration_present": False, "configuration_source": None,
                 "secret_values_inspected": False,
                 "remote_validity_or_licence_acceptance_proven": False})
    monkeypatch.delenv("SPECTRALEARTH_ALLOW_NETWORK", raising=False)
    assert main([
        "preflight", "--campaign", str(frozen),
        "--full-download-dir", str(tmp_path / "full-download"),
        "--canary-download-dir", str(tmp_path / "canary-download"),
        "--cache-dir", str(tmp_path / "cache"),
        "--independent-cache-dir", str(tmp_path / "independent"),
    ]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED" and report["network_used"] is False
    assert len(report["blockers"]) == 3
