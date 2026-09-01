"""Offline acceptance for the frozen two-stage T4C.6 acquisition campaign."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.analysis_engine.cross_scale import GateProtocol
from src.analysis_engine.gate_campaign import (
    CampaignSupersession,
    GateCampaign,
    load_gate_campaign,
    load_gate_campaign_supersession,
    main,
    preflight_gate_campaign,
    review_gate_campaign,
    review_gate_campaign_supersession,
    save_gate_campaign,
    save_gate_campaign_supersession,
)
from src.analysis_engine.gate_run import GateStudyPlan
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_source import CDSRegionalRequest
from src.data_layer.zarr_source import CropSpec


def _request(**overrides):
    values = dict(
        variables=("t",), date_start="2020-01-01", date_end="2020-02-19",
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
        expected_frames=200, cadence_seconds=21600.0, train_ratio=0.6,
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

    # Defect D85, pinned here so the frozen campaign carries its own refutation. The two
    # adjacent numbers count different things: `power` counts the 4,999 surrogates *requested*
    # and is satisfied, while the exact test's reference set is the distinct admissible circular
    # shifts the record contains. The confirmatory partition holds 2,912 of the 3,005 needed, so
    # no draw from it can reach the corrected level. The campaign is left frozen and unedited;
    # re-freezing it is a recorded supersession, not a repair.
    resolution = design["surrogate_resolution"]
    assert design["resolvable"] is False
    assert resolution["train"]["resolves_corrected_level"] is True
    assert resolution["test"]["resolves_corrected_level"] is False
    assert resolution["test"]["distinct_admissible_shifts"] == 2912
    assert resolution["test"]["distinct_shifts_required"] == 3005
    assert resolution["test"]["frames_required"] == 3007
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
        expected_frames=200, cadence_seconds=21600.0, train_ratio=0.6,
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


def test_acquisition_refuses_the_unresolvable_frozen_campaign(tmp_path):
    """D85 blocks acquisition, not review: the campaign stays readable, the 2.5 GB is not spent."""
    path = (Path(__file__).parents[2] / "campaigns"
            / "t4c6_nz_era5_temperature_850_v1.json")
    campaign = load_gate_campaign(path)
    with pytest.raises(InvalidParameterError, match="resolve the declared family"):
        preflight_gate_campaign(
            campaign, full_download_dir=tmp_path / "full",
            canary_download_dir=tmp_path / "canary", cache_dir=tmp_path / "cache",
            independent_cache_dir=tmp_path / "independent")


def test_a_resolvable_campaign_is_not_blocked_by_the_new_audit(tmp_path, monkeypatch):
    """The audit must refuse the short record and nothing else."""
    campaign = _campaign()
    design = review_gate_campaign(campaign)["scientific_design"]
    assert design["resolvable"] is True
    assert design["surrogate_resolution"]["adequate"] is True

# --------------------------------------------------------------------------------------
# T4C.5i step 8 -- superseding a frozen campaign
# --------------------------------------------------------------------------------------

_CAMPAIGNS = Path(__file__).parents[2] / "campaigns"
_V1 = _CAMPAIGNS / "t4c6_nz_era5_temperature_850_v1.json"
_V2 = _CAMPAIGNS / "t4c6_nz_era5_temperature_850_v2.json"
_SUPERSESSION = _CAMPAIGNS / "t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json"


def _relength(campaign, *, date_end, expected_frames, suffix):
    """The same frozen design over a different record length."""
    full = dataclasses.replace(campaign.full_acquisition, date_end=date_end)
    protocol = dataclasses.replace(
        campaign.gate_plan.protocol,
        study_id="t4c6-nz-era5-temperature-850-%s" % suffix,
        expected_frames=expected_frames)
    plan = dataclasses.replace(
        campaign.gate_plan, study_id=protocol.study_id,
        crop=full.to_crop_spec(), protocol=protocol)
    return GateCampaign(
        campaign_id="t4c6-nz-era5-temperature-850-campaign-%s" % suffix,
        full_acquisition=full, canary_acquisition=campaign.canary_acquisition,
        weatherbench_overlap=campaign.weatherbench_overlap,
        expected_overlap_frames=campaign.expected_overlap_frames, gate_plan=plan)


def _reason(**overrides):
    record = {
        "defect": "D85", "check": "surrogate_resolution", "parameters": {},
        "statement": "the confirmatory partition cannot resolve the declared family",
    }
    record.update(overrides)
    return record


def test_the_checked_in_supersession_retires_v1_for_a_resolvable_v2():
    """The three artifacts are one record: what was frozen, what replaces it, and why."""
    superseded, successor = load_gate_campaign(_V1), load_gate_campaign(_V2)
    supersession = load_gate_campaign_supersession(_SUPERSESSION)

    # The retired campaign is not edited. Its own review still carries its defect, which is
    # the entire reason a supersession is a third file rather than a change to the first.
    assert superseded.fingerprint() == (
        "84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975")
    assert review_gate_campaign(superseded)["scientific_design"]["resolvable"] is False

    assert successor.fingerprint() == (
        "c66284d619d7439638ec5e1886671894df4d12708ab3cdced245d7e5f80fa23c")
    design = review_gate_campaign(successor)["scientific_design"]
    assert design["full_frames"] == 8764
    assert (design["train_frames"], design["test_frames"]) == (5258, 3498)
    assert design["resolvable"] is True
    assert design["surrogate_resolution"]["test"]["distinct_admissible_shifts"] == 3496
    assert design["surrogate_resolution"]["test"]["distinct_shifts_required"] == 3005
    # Everything the defect was not about is unchanged, including the crop D84 is open on.
    assert successor.gate_plan.protocol.family_size == 36
    assert successor.gate_plan.crop.lat_min == superseded.gate_plan.crop.lat_min
    assert successor.gate_plan.protocol.lags == superseded.gate_plan.protocol.lags
    assert successor.canary_acquisition == superseded.canary_acquisition

    review = review_gate_campaign_supersession(supersession)
    assert review["network_used"] is False
    assert review["superseded_campaign_sha256"] == superseded.fingerprint()
    assert review["successor_campaign_sha256"] == successor.fingerprint()
    assert [finding["check"] for finding in review["reasons"]] == [
        "surrogate_resolution", "resolution_margin"]
    for finding in review["reasons"]:
        assert finding["defect"] == "D85"
        assert finding["superseded"]["passes"] is False
        assert finding["successor"]["passes"] is True
    for entry in review["preserved"]:
        assert entry["superseded"]["passes"] is True
        assert entry["successor"]["passes"] is True
    # D84 is named as unresolved rather than quietly carried along by the re-freeze.
    assert {entry["defect"] for entry in review["deferred_to_run"]} == {"D84", "D85", "D43"}
    assert "does not establish" in [
        entry["statement"] for entry in review["deferred_to_run"]
        if entry["defect"] == "D84"][0]


def test_a_reason_must_be_a_defect_the_retired_design_actually_has():
    """A supersession cannot be written for a problem the superseded campaign does not have."""
    successor = load_gate_campaign(_V2)
    later = _relength(successor, date_end="2024-12-31", expected_frames=8764 + 1464,
                      suffix="v3")
    with pytest.raises(InvalidParameterError, match="does not describe a defect"):
        CampaignSupersession(
            supersession_id="v2-to-v3", superseded=successor, successor=later,
            reasons=(_reason(),))


def test_a_supersession_cannot_claim_a_repair_that_did_not_happen():
    """The minimal repair clears the audit and fails the margin, so it cannot state the margin.

    2023-02-27 gives exactly the 3,007 confirmatory frames `frames_required` asked for, and
    resolves only at the most favourable Theiler window of one frame. The reason that says the
    successor carries a margin is refused against it, which is what stopped the re-freeze from
    being a design that would have failed again after the transfer.
    """
    superseded = load_gate_campaign(_V1)
    minimal = _relength(superseded, date_end="2023-02-27", expected_frames=7536,
                        suffix="minimal")
    assert review_gate_campaign(minimal)["scientific_design"]["resolvable"] is True

    # The audit-level reason is admissible, because the minimal record does repair that.
    CampaignSupersession(
        supersession_id="v1-to-minimal", superseded=superseded, successor=minimal,
        reasons=(_reason(),))
    with pytest.raises(InvalidParameterError, match="does not repair what it claims"):
        CampaignSupersession(
            supersession_id="v1-to-minimal", superseded=superseded, successor=minimal,
            reasons=(_reason(check="resolution_margin",
                             parameters={"minimum_theiler_frames": 16}),))


def test_a_preserved_property_the_repair_drops_is_not_preserved():
    """The minimal repair spans five years and 58 days, and cannot claim whole annual cycles."""
    superseded = load_gate_campaign(_V1)
    minimal = _relength(superseded, date_end="2023-02-27", expected_frames=7536,
                        suffix="minimal")
    with pytest.raises(InvalidParameterError, match="a change rather than a preserved"):
        CampaignSupersession(
            supersession_id="v1-to-minimal", superseded=superseded, successor=minimal,
            reasons=(_reason(),),
            preserved=({"check": "whole_annual_cycles", "parameters": {},
                        "statement": "five whole calendar years"},))


def test_an_unverifiable_reason_cannot_retire_a_frozen_design():
    superseded, successor = load_gate_campaign(_V1), load_gate_campaign(_V2)
    with pytest.raises(InvalidParameterError, match="cannot be checked"):
        CampaignSupersession(
            supersession_id="v1-to-v2", superseded=superseded, successor=successor,
            reasons=(_reason(check="the design felt wrong"),))
    with pytest.raises(InvalidParameterError, match="not retired by assertion"):
        CampaignSupersession(
            supersession_id="v1-to-v2", superseded=superseded, successor=successor,
            reasons=())


def test_a_supersession_is_not_an_edit_wearing_a_new_name():
    superseded = load_gate_campaign(_V1)
    successor = load_gate_campaign(_V2)
    renamed = dataclasses.replace(successor, campaign_id=superseded.campaign_id)
    with pytest.raises(InvalidParameterError, match="an identifier distinct"):
        CampaignSupersession(
            supersession_id="v1-to-v1", superseded=superseded, successor=renamed,
            reasons=(_reason(),))
    with pytest.raises(InvalidParameterError, match="rename rather than a supersession"):
        CampaignSupersession(
            supersession_id="v1-to-v1", superseded=superseded,
            successor=dataclasses.replace(superseded, campaign_id="other"),
            reasons=(_reason(),))


def test_a_supersession_is_atomic_hashable_and_refuses_a_swapped_campaign(tmp_path):
    supersession = load_gate_campaign_supersession(_SUPERSESSION)
    target = tmp_path / "supersession.json"
    assert save_gate_campaign_supersession(target, supersession) == supersession.fingerprint()
    with pytest.raises(FileExistsError):
        save_gate_campaign_supersession(target, supersession)
    assert load_gate_campaign_supersession(target).fingerprint() == supersession.fingerprint()

    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["superseded_campaign"] = envelope["successor_campaign"]
    swapped = tmp_path / "swapped.json"
    swapped.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(DataSourceError, match="not the one it names"):
        load_gate_campaign_supersession(swapped)

    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["supersession"]["reasons"][0]["statement"] = "a different story"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(DataSourceError, match="does not authenticate"):
        load_gate_campaign_supersession(edited)


def test_acquisition_refuses_a_retired_campaign_and_admits_its_successor(tmp_path, monkeypatch):
    """The supersession bites where the money is spent, not where the record is read."""
    supersession = load_gate_campaign_supersession(_SUPERSESSION)
    directories = dict(
        full_download_dir=tmp_path / "full", canary_download_dir=tmp_path / "canary",
        cache_dir=tmp_path / "cache", independent_cache_dir=tmp_path / "independent")

    # Reading the retired campaign is still allowed, and still reports its own defect.
    assert review_gate_campaign(supersession.superseded)["scientific_design"][
        "resolvable"] is False
    with pytest.raises(InvalidParameterError, match="no supplied supersession has retired"):
        preflight_gate_campaign(
            supersession.superseded, supersessions=[supersession], **directories)

    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.shutil.disk_usage",
        lambda path: SimpleNamespace(total=1 << 50, used=0, free=1 << 50))
    report = preflight_gate_campaign(
        supersession.successor, supersessions=[supersession], **directories)
    assert report["network_used"] is False

def test_the_supersession_is_reviewable_and_blocks_acquisition_from_the_command_line(
        tmp_path, capsys, monkeypatch):
    supersession = load_gate_campaign_supersession(_SUPERSESSION)
    assert main(["review-supersession", "--supersession", str(_SUPERSESSION)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["supersession_sha256"] == supersession.fingerprint()
    assert emitted["successor_review"]["scientific_design"]["resolvable"] is True

    monkeypatch.setattr(
        "src.analysis_engine.gate_campaign.shutil.disk_usage",
        lambda path: SimpleNamespace(total=1 << 50, used=0, free=1 << 50))
    arguments = [
        "preflight", "--full-download-dir", str(tmp_path / "full"),
        "--canary-download-dir", str(tmp_path / "canary"),
        "--cache-dir", str(tmp_path / "cache"),
        "--independent-cache-dir", str(tmp_path / "independent"),
        "--supersession", str(_SUPERSESSION)]
    with pytest.raises(InvalidParameterError, match="no supplied supersession has retired"):
        main(arguments + ["--campaign", str(_V1)])
    # The successor is not retired, so preflight proceeds to the readiness checks and reports
    # whatever this machine actually is. On a machine without the CDS client or a recorded
    # consent that is BLOCKED, and the point of the assertion is that it is blocked on
    # readiness rather than on the supersession.
    assert main(arguments + ["--campaign", str(_V2)]) in (0, 2)
    report = json.loads(capsys.readouterr().out)
    assert report["network_used"] is False and report["client_constructed"] is False
    assert report["campaign_sha256"] == supersession.successor.fingerprint()
    assert report["status"] in ("READY_FOR_CANARY", "BLOCKED")
    assert not any("supersede" in str(blocker) for blocker in report["blockers"])
