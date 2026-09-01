"""T4C.6 gate-job acceptance without representing synthetic data as ERA5 evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis_engine.cross_scale import GateProtocol
from src.analysis_engine.gate_run import (
    GateStudyPlan,
    _audit_limitations,
    _audited_test,
    _power_adjudication,
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
    assert receipt["power_adjudication"]["power_applied"] is False
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


# ================================================ the derived power record (T4C.5i step 7)


def _real_plan():
    """A plan in the real gate role, built without a store because nothing here reads one.

    `_power_adjudication` decides a verdict from a role and two records; giving it a
    materialised cache would be several seconds spent proving nothing about the rule.
    """
    crop = CropSpec(
        store="cds:reanalysis-era5-pressure-levels", variables=("t",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-30T18:00:00",
        lat_min=-47.75, lat_max=-32.0, lon_min=160.0, lon_max=175.75,
        levels=(850,), n_levels_analysis=2)
    protocol = GateProtocol(
        study_id="power-boundary", n_scales=2, lags=(1,), expected_frames=120,
        cadence_seconds=21600.0, train_ratio=0.6, embargo_frames=1,
        estimator="transfer_entropy", measure="energy_density", bins=2, n_surrogates=59,
        alpha=0.05, correction="benjamini_yekutieli", seed=4406)
    return GateStudyPlan(
        study_id=protocol.study_id, evidence_role="real_era5_gate", crop=crop,
        protocol=protocol, variable="t", level_hpa=850, transform_family="swt",
        wavelet="db2", advection_speed_m_s=50.0, climatology_harmonics=1)


def _power(verdict="ADEQUATE", *, status="MEASURED", reproduces=True, deficit=None):
    return {
        "status": status,
        "verdict": verdict,
        "surrogate_ensemble": {"reproduces_sweep_summary": reproduces},
        "refusal": {"verdict": verdict, "deficit": deficit, "reason": "measured",
                    "remedies": []},
        "reason": "the audit did not run",
    }


def test_the_receipt_publishes_the_derivation_rather_than_a_judgement(tmp_path):
    """Every quantity the removed constant was standing in for, in the receipt.

    Step 6 demoted `MIN_VALID_INTERIOR` to a labelled heuristic; that only helps a reviewer if
    the derived quantities replacing it are *published*. So the receipt has to carry the
    decorrelation lengths, the effective sample sizes, the measured attenuation curve, the
    minimum detectable effect and the rule that turns them into a FAIL/INVALID boundary --
    otherwise the study has swapped one number a reader must trust for four of them.
    """
    crop, cache = _cache(tmp_path)
    receipt = run_cached_gate(_plan(crop), cache_dir=str(cache))
    power = receipt["spatial_power"]
    assert power["status"] == "MEASURED"
    assert power["partition"] == "train", "the held-out partition is not an audit sample"

    # Steps 1 and 2: a decorrelation length and an effective sample count per scale.
    assert len(power["interiors"]) == 2
    for record in power["interiors"]:
        assert record["measured"] is True
        assert len(record["median_decorrelation_px_per_axis"]) == 2
        assert record["median_effective_samples"] > 0
        assert 0.0 <= record["saturated_fraction"] <= 1.0

    # Step 3: attenuation measured over concentric sub-crops, only precision varying.
    curve = power["attenuation"]
    sizes = [row["interior_px"] for row in curve["curve"]]
    assert sizes == sorted(set(sizes))
    assert sizes[-1] == power["matched_interior_px"]
    assert curve["lag"] == power["audited_test"]["lag_frames"]
    assert curve["bins"] == 2

    # Step 4: the threshold comes from the sweep's own ensemble, not a fresh one.
    assert power["surrogate_ensemble"]["reproduces_sweep_summary"] is True
    assert power["surrogate_ensemble"]["n_surrogates"] == 59
    assert "minimum_detectable_effect" in power["refusal"]

    # The two scales lose different margins, so the curve's largest row is not the sweep's own
    # estimate. Both are published and neither is adjusted into the other.
    assert power["interior_sides_px"] == {"source": 60, "target": 54}
    assert power["matched_interior_px"] == 54
    assert "reconciled" in power["matched_interior_note"]
    assert "sweep_observed_nats" in power

    # The caveats travel with the claim rather than sitting in a docstring. SWT is undecimated,
    # so the decimation caveat must be absent here and present for a family that decimates.
    assert any("within-field trend" in text for text in power["limitations"])
    assert not any("decimates" in text for text in power["limitations"])


def test_the_audit_takes_the_familys_best_case_and_takes_it_from_train():
    """The binding case for an absence is the test that came closest to surviving."""
    rows = [
        {"label": "1->2@1", "p_value": 0.4, "excess_nats": 0.1, "observed_nats": 0.5},
        {"label": "2->1@1", "p_value": 0.1, "excess_nats": 0.2, "observed_nats": 0.6},
        {"label": "1->3@1", "p_value": 0.1, "excess_nats": 0.9, "observed_nats": 0.7},
        {"label": "3->1@1", "p_value": 0.05, "excess_nats": 0.3,
         "observed_nats": float("nan")},
    ]
    # The smallest p-value belongs to a row the estimator could not evaluate, so it is not
    # auditable; among the rest the tie on p is broken by the larger excess.
    assert _audited_test(rows)["label"] == "1->3@1"
    assert _audited_test([]) is None


def test_an_absence_is_a_finding_only_when_the_power_record_says_so():
    """T4C.5i step 5's deferred half, which is the point of the whole task.

    A FAIL asserts that the atmosphere does not carry the relationship. That assertion is only
    available to a run whose instrument could have seen it, and until this step nothing in the
    system derived the spatial-precision term that decides. Every branch is pinned, because a
    boundary that is right in one direction and silent in the other is not a boundary.
    """
    real = _real_plan()

    adequate = _power_adjudication(real, {"verdict": "FAIL"}, _power("ADEQUATE"))
    assert adequate["scientific_verdict"] == "FAIL"
    assert adequate["power_applied"] is True

    attenuated = _power_adjudication(
        real, {"verdict": "FAIL"}, _power("INVALID", deficit="attenuation"))
    assert attenuated["scientific_verdict"] == "INVALID"
    assert attenuated["deficit"] == "attenuation"
    assert "not a negative finding" in attenuated["reason"]

    # An absence whose power was never characterised is not a negative finding either.
    unaudited = _power_adjudication(
        real, {"verdict": "FAIL"}, _power(status="NOT_AUDITED"))
    assert unaudited["scientific_verdict"] == "INVALID"
    assert "the audit did not run" in unaudited["reason"]

    # Nor is one whose threshold came from an ensemble that is not this study's.
    drifted = _power_adjudication(
        real, {"verdict": "FAIL"}, _power("ADEQUATE", reproduces=False))
    assert drifted["scientific_verdict"] == "INVALID"
    assert "decision boundary" in drifted["reason"]

    # A PASS is never downgraded: attenuation biases toward the null, so an undersized crop
    # cannot manufacture one. Recording why is the point -- silence would look like oversight.
    passed = _power_adjudication(
        real, {"verdict": "PASS"}, _power("INVALID", deficit="attenuation"))
    assert passed["scientific_verdict"] == "PASS"
    assert passed["power_applied"] is False
    assert "biases toward the null" in passed["reason"]

    # An INVALID gate stays INVALID for its own reasons and is not relabelled by this rule.
    broken = _power_adjudication(real, {"verdict": "INVALID"}, _power("ADEQUATE"))
    assert broken["scientific_verdict"] == "INVALID"
    assert broken["power_applied"] is False


def test_synthetic_acceptance_publishes_the_record_and_is_adjudicated_by_nothing(tmp_path):
    """Orchestration evidence must not acquire a verdict it did not earn."""
    crop = CropSpec(
        store="local.zarr", variables=("t",),
        time_start="2020-01-01T00:00:00", time_end="2020-01-30T18:00:00",
        lat_min=-47.75, lat_max=-32.0, lon_min=160.0, lon_max=175.75,
        levels=(850,), n_levels_analysis=2)
    ruling = _power_adjudication(_plan(crop), {"verdict": "FAIL"}, _power("ADEQUATE"))
    assert ruling["scientific_verdict"] == "NOT_ESTABLISHED"
    assert ruling["power_applied"] is False


def test_a_decimated_family_says_its_matched_window_is_not_a_shared_area():
    """The concentric window matches coefficient counts; only SWT makes that a shared patch.

    DTCWT halves the grid at every level, so the same number of native pixels at level 1 and
    level 3 covers different amounts of atmosphere. The curve is still readable as a precision
    trend, but calling it a matched-area comparison would be a claim nothing supports -- so the
    record says so, in the receipt, next to the number.
    """
    plan = _real_plan()
    swt = _audit_limitations(plan)
    assert not any("decimates" in text for text in swt)
    dtcwt = _audit_limitations(GateStudyPlan(**{**plan.__dict__,
                                                "transform_family": "dtcwt"}))
    caveat = [text for text in dtcwt if "decimates" in text]
    assert len(caveat) == 1
    assert "not affected" in caveat[0] or "is not\naffected" in caveat[0] \
        or "is not affected" in caveat[0]
