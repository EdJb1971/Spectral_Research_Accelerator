"""Transform-derived, metadata-only acquisition planning (R13)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.errors import FieldTooSmallError, InvalidParameterError
from src.core.registry import restore, snapshot
from src.data_layer import crop_planner as cp
from src.data_layer import zarr_source as zs
from src.transform_engine import dtcwt, stationary
from src.transform_engine.registry import TRANSFORMS, TransformSpec

xr = pytest.importorskip("xarray")


def _dataset(ny=600, nx=700):
    values = np.zeros((2, ny, nx), dtype=np.float32)
    dataset = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values)},
        coords={"time": np.arange(2), "latitude": np.arange(ny, dtype=float),
                "longitude": np.arange(nx, dtype=float)})
    dataset["temperature"].encoding["chunks"] = (1, 100, 100)
    return dataset


def _crop(**overrides):
    values = dict(
        store="fixture", variables=("temperature",), time_start="0", time_end="1",
        lat_min=100.0, lat_max=483.0, lon_min=200.0, lon_max=519.0,
        levels=(), n_levels_analysis=4)
    values.update(overrides)
    return zs.CropSpec(**values)


def test_support_and_thresholds_come_from_the_registered_transform_implementations():
    swt = cp.assess_shape(384, 320, cp.TransformSupportRequest(
        transform_family="swt", levels=4, wavelet="db2"))
    assert swt["levels"][-1]["support_parent_px"] == stationary.filter_support("db2", 4)
    assert swt["levels"][-1]["margin_parent_px"] == \
        stationary.valid_interior_halfwidth("db2", 4)
    assert swt["absolute_minimum"]["shape"] == [47, 47]
    assert swt["recommended_minimum"]["shape"] == [174, 174]
    assert swt["recommended_minimum"]["dyadic_operational_shape"] == [256, 256]
    assert swt["meets_recommended_minimum"] is True

    dual = cp.assess_shape(384, 320, cp.TransformSupportRequest(
        transform_family="dtcwt", levels=4, dtcwt_level1="near_sym_b",
        dtcwt_qshift="qshift_b"))
    assert dual["levels"][-1]["support_parent_px"] == dtcwt.filter_support(4)
    assert dual["levels"][-1]["margin_native_px"] == dtcwt.native_halfwidth(4)
    assert dual["absolute_minimum"]["shape"] == [240, 240]
    assert dual["recommended_minimum"]["shape"] == [352, 352]
    assert dual["recommended_minimum"]["dyadic_operational_shape"] == [512, 512]
    assert dual["verdict"] == "technical_only"


def test_an_external_transform_can_supply_support_without_editing_the_planner():
    state = snapshot(TRANSFORMS)
    try:
        TRANSFORMS.add(
            "plugin_multiscale", TransformSpec(
                apply=lambda *_: {}, inverse=lambda _c, field: field,
                summarise=lambda _c: {},
                support=lambda levels, _config: {
                    "transform_family": "plugin_multiscale", "config": {},
                    "alignment_cells": 1,
                    "levels": [{"level": level, "support_parent_px": 2 * level + 1,
                                "margin_parent_px": level, "sampling_factor": 1,
                                "margin_native_px": level}
                               for level in range(1, levels + 1)]}),
            description="fixture")
        result = cp.assess_shape(
            256, 256, cp.TransformSupportRequest(
                transform_family="plugin_multiscale", levels=3))
        assert result["levels"][-1]["margin_parent_px"] == 3
        assert result["recommended_minimum"]["shape"] == [134, 134]
    finally:
        restore(TRANSFORMS, state)


def test_a_transform_without_a_support_contract_is_refused_before_source_access():
    with pytest.raises(InvalidParameterError, match="does not declare boundary support"):
        cp.TransformSupportRequest(transform_family="fft", levels=4)


def test_plan_expands_symmetrically_on_native_coordinates_and_reprices_the_result():
    dataset = _dataset()
    analysis = cp.TransformSupportRequest(transform_family="dtcwt", levels=4)
    plan = cp.plan_acquisition(dataset, _crop(), analysis)
    assert plan["geometry"]["current_shape"] == [384, 320]
    suggestion = plan["suggestions"]["recommended"]
    assert suggestion["feasible"] is True
    assert suggestion["actual_shape"] == [384, 352]
    assert suggestion["bounds"] == {
        "lat_min": 100.0, "lat_max": 483.0,
        "lon_min": 184.0, "lon_max": 535.0}
    # T4C.5i step 6: the expansion is priced against the requirement, not the round number.
    assert plan["geometry"]["recommended_minimum"]["shape"] == [352, 352]
    assert suggestion["cost"]["bytes_fetched_estimate"] > 0
    assert suggestion["cost"]["bytes_wanted"] > 0
    assert len(plan["plan_sha256"]) == 64
    assert len(plan["source_observation_sha256"]) == 64


def test_plan_reports_when_the_source_cannot_supply_the_recommendation():
    dataset = _dataset(ny=300, nx=340)
    crop = _crop(lat_max=299.0, lon_max=339.0)
    plan = cp.plan_acquisition(
        dataset, crop, cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    suggestion = plan["suggestions"]["recommended"]
    assert suggestion["feasible"] is False
    assert "store itself has only 300x340" in suggestion["reason"]


def test_plan_identity_moves_with_transform_configuration_but_not_field_values():
    first = _dataset()
    second = _dataset()
    second["temperature"].values[:] = 99.0
    changed_coordinates = _dataset()
    changed_coordinates["latitude"].values[0] = -0.25
    crop = _crop()
    dtcwt_plan = cp.plan_acquisition(
        first, crop, cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    same_geometry = cp.plan_acquisition(
        second, crop, cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    swt_plan = cp.plan_acquisition(
        first, crop, cp.TransformSupportRequest(
            transform_family="swt", levels=4, wavelet="db2"))
    coordinate_plan = cp.plan_acquisition(
        changed_coordinates, crop,
        cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    assert dtcwt_plan["plan_sha256"] == same_geometry["plan_sha256"]
    assert dtcwt_plan["plan_sha256"] != swt_plan["plan_sha256"]
    assert dtcwt_plan["source_observation_sha256"] != \
        coordinate_plan["source_observation_sha256"]
    assert dtcwt_plan["plan_sha256"] != coordinate_plan["plan_sha256"]


def test_materialisation_refuses_from_metadata_before_constructing_a_data_selection(
        monkeypatch, tmp_path):
    dataset = _dataset(ny=300, nx=300)
    monkeypatch.setattr(zs, "is_cached", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(zs, "open_dataset", lambda *_args, **_kwargs: (dataset, type(
        "Counter", (), {"bytes_read": 0, "keys_read": 0})()))
    monkeypatch.setattr(zs, "select", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("data selection must not be constructed before the geometry refusal")))
    with pytest.raises(FieldTooSmallError, match="R13 heuristic interior") as refusal:
        zs.materialise(
            _crop(lat_min=0, lat_max=299, lon_min=0, lon_max=299),
            cache_dir=str(tmp_path),
            analysis=cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    assert "absolute minimum" in str(refusal.value)


def test_invalid_filter_configuration_is_refused_as_a_client_parameter():
    with pytest.raises(InvalidParameterError, match="filter settings accepted"):
        cp.TransformSupportRequest(
            transform_family="swt", levels=4, wavelet="invented")


# ============================================ demoting the constants (T4C.5i step 6)

def test_the_recommended_threshold_is_the_requirement_not_the_round_number():
    """The dyadic size is reported beside the threshold, and is never the threshold.

    T4C.5i step 6. A DTCWT level-4 crop needs 352 aligned native cells to leave the heuristic
    128 uncontaminated parent cells. 512 is the next power of two. A 384 px crop clears the
    requirement and misses the round number, and refusing it would be refusing an inconvenient
    pixel count rather than an inadequate one.
    """
    analysis = cp.TransformSupportRequest(
        transform_family="dtcwt", levels=4, dtcwt_level1="near_sym_b",
        dtcwt_qshift="qshift_b")
    awkward = cp.assess_shape(384, 384, analysis)
    assert awkward["recommended_minimum"]["shape"] == [352, 352]
    assert awkward["recommended_minimum"]["dyadic_operational_shape"] == [512, 512]
    assert awkward["meets_recommended_minimum"] is True
    assert awkward["verdict"] == "recommended"


def test_the_threshold_declares_itself_a_heuristic_and_says_what_replaces_it():
    """A reported recommendation must not read as a derivation (T4C.5i step 6)."""
    geometry = cp.assess_shape(384, 320, cp.TransformSupportRequest(
        transform_family="swt", levels=4, wavelet="db2"))
    recommended = geometry["recommended_minimum"]
    assert recommended["heuristic"] is True
    assert "heuristic" in recommended["basis"]
    assert "not a derived power criterion" in recommended["limitation"]
    assert "spatial_power" in recommended["limitation"]
    # And it must not claim the authority it was demoted out of.
    assert "statistically recommended" not in recommended["basis"]
    assert "requires" not in recommended["basis"]


def test_the_dyadic_convention_never_moves_a_verdict():
    """Every verdict is reproducible from the aligned requirement alone."""
    analysis = cp.TransformSupportRequest(
        transform_family="dtcwt", levels=4, dtcwt_level1="near_sym_b",
        dtcwt_qshift="qshift_b")
    for side in (240, 300, 352, 384, 512):
        geometry = cp.assess_shape(side, side, analysis)
        required = geometry["recommended_minimum"]["shape"][0]
        assert geometry["meets_recommended_minimum"] is (side >= required)
        assert (geometry["verdict"] == "recommended") is (side >= required)


def test_the_frozen_campaign_crop_no_longer_falls_between_two_gates():
    """D84's own case: the two geometry gates now agree about the same crop.

    The T4C.6 preregistration is a 161 px crop analysed with db2 SWT at level 3. The accumulated
    support contaminates 11 px per side, leaving a 139 px valid interior, so `gate_campaign`
    admitted it against the 128 px heuristic. The planner took the same 128, reached a raw
    requirement of 150 -- which 161 clears -- and then rounded to 256 and refused. The crop was
    admitted by one gate and refused by the other purely because of the rounding, and the
    refusal described 256 to the caller as *statistically recommended*.

    With the rounding removed from the refusal path (T4C.5i step 6) the threshold is 150 and
    both gates admit the crop. This closes the contradiction, not the defect: whether 139 px of
    interior is *enough* is a power question, answered by `analysis_engine/spatial_power.py`,
    and the FAIL/INVALID adjudication that consumes the answer is still unbuilt.
    """
    geometry = cp.assess_shape(161, 161, cp.TransformSupportRequest(
        transform_family="swt", levels=3, wavelet="db2"))
    assert geometry["levels"][-1]["valid_parent_shape"] == [139, 139]
    assert geometry["recommended_minimum"]["shape"] == [150, 150]
    assert geometry["recommended_minimum"]["dyadic_operational_shape"] == [256, 256]
    assert geometry["meets_recommended_minimum"] is True
    assert geometry["verdict"] == "recommended"
