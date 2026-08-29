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
    assert swt["recommended_minimum"]["shape"] == [256, 256]
    assert swt["meets_recommended_minimum"] is True

    dual = cp.assess_shape(384, 320, cp.TransformSupportRequest(
        transform_family="dtcwt", levels=4, dtcwt_level1="near_sym_b",
        dtcwt_qshift="qshift_b"))
    assert dual["levels"][-1]["support_parent_px"] == dtcwt.filter_support(4)
    assert dual["levels"][-1]["margin_native_px"] == dtcwt.native_halfwidth(4)
    assert dual["absolute_minimum"]["shape"] == [240, 240]
    assert dual["recommended_minimum"]["shape"] == [512, 512]
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
        assert result["recommended_minimum"]["shape"] == [256, 256]
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
    assert suggestion["actual_shape"] == [512, 512]
    assert suggestion["bounds"] == {
        "lat_min": 36.0, "lat_max": 547.0,
        "lon_min": 104.0, "lon_max": 615.0}
    assert suggestion["cost"]["bytes_fetched_estimate"] > 0
    assert suggestion["cost"]["bytes_wanted"] > 0
    assert len(plan["plan_sha256"]) == 64
    assert len(plan["source_observation_sha256"]) == 64


def test_plan_reports_when_the_source_cannot_supply_the_recommendation():
    dataset = _dataset(ny=400, nx=450)
    crop = _crop(lat_max=399.0, lon_max=449.0)
    plan = cp.plan_acquisition(
        dataset, crop, cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    suggestion = plan["suggestions"]["recommended"]
    assert suggestion["feasible"] is False
    assert "store itself has only 400x450" in suggestion["reason"]


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
    with pytest.raises(FieldTooSmallError, match="statistically recommended") as refusal:
        zs.materialise(
            _crop(lat_min=0, lat_max=299, lon_min=0, lon_max=299),
            cache_dir=str(tmp_path),
            analysis=cp.TransformSupportRequest(transform_family="dtcwt", levels=4))
    assert "absolute minimum" in str(refusal.value)


def test_invalid_filter_configuration_is_refused_as_a_client_parameter():
    with pytest.raises(InvalidParameterError, match="filter settings accepted"):
        cp.TransformSupportRequest(
            transform_family="swt", levels=4, wavelet="invented")
