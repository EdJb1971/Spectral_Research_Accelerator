"""TG13.1: bounded, content-bound TESS photometry and its refusals."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from extensions.tess_lightcurve import TESS, register as register_tess_domain
from src.api.main import app
from src.core.domain import DOMAIN_DECLARATIONS, PrecedenceNotAdmissibleError
from src.core.errors import InvalidParameterError, UserInputError
from src.core.tess_pool_assessment import (assess_tess_pool_acquisition,
                                           merge_tess_acquisitions,
                                           qualify_tess_acquisition)
from src.core.tess_profile_export import export_tess_partner_profiles
from src.data_layer.lightcurves import (LIGHTCURVE_SOURCES, LightCurveCollection,
                                        LightCurveProduct, LightCurveSpec, load_collection,
                                        persist_collection)
from src.data_layer.tess_source import (acquire_discovered_tess, acquire_tess,
                                        discover_periodic_tess_products, discover_tess_targets,
                                        inspect_tess_query, parse_spoc_fits,
                                        register_tess_source)

TIC = "261136679"
RA, DEC = 84.291188, -80.46912


def spec(**changes) -> LightCurveSpec:
    values = dict(target_id=TIC, sectors=(1,), max_products=2,
                  max_download_bytes=8 * 1024 * 1024)
    values.update(changes)
    return LightCurveSpec(**values)


def fits_bytes(flux_offset: float = 0.0, times=None, quality=None, tic_id: str = TIC) -> bytes:
    """A SPOC-shaped product. `times` may carry NaN, as every real one does."""
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = int(tic_id)
    primary.header["RA_OBJ"] = RA
    primary.header["DEC_OBJ"] = DEC
    time_array = np.array([1.0, 1.01, 1.03] if times is None else times, dtype=np.float64)
    n = time_array.size
    def ramp(start, step):
        return np.array([start + step * index for index in range(n)], dtype=np.float32)
    flux = ramp(100.0, 1.0)
    if n == 3 and times is None:
        flux = np.array([100.0, 101.0 + flux_offset, 99.5], dtype=np.float32)
    quality_array = np.array(
        ([0, 32, 0] if n == 3 else [0] * n) if quality is None else quality, dtype=np.int32)
    columns = [
        fits.Column(name="TIME", format="D", array=time_array),
        fits.Column(name="PDCSAP_FLUX", format="E", array=flux),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=ramp(0.1, 0.0)),
        fits.Column(name="SAP_FLUX", format="E", array=ramp(110.0, 1.0)),
        fits.Column(name="SAP_FLUX_ERR", format="E", array=ramp(0.2, 0.0)),
        fits.Column(name="QUALITY", format="J", array=quality_array),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["BJDREFI"] = 2457000
    table.header["BJDREFF"] = 0.0
    table.header["TIMESYS"] = "TDB"
    output = io.BytesIO()
    fits.HDUList([primary, table]).writeto(output, checksum=True)
    return output.getvalue()


def fake_download(payload: bytes):
    return lambda _uri, _maximum: payload


def fake_query(payload_size: int):
    filename = "tess-s0001-%s-0001-s_lc.fits" % TIC.zfill(16)

    def query(request):
        service = request["service"]
        if service == "Mast.Catalogs.Filtered.Tic":
            return {"status": "COMPLETE", "data": [{"ID": int(TIC), "ra": RA, "dec": DEC}]}
        if service == "Mast.Caom.Filtered.Position":
            return {"status": "COMPLETE", "data": [{
                "obsid": "123", "obs_id": "tess-s0001-%s" % TIC,
                "target_name": "TIC %s" % TIC, "sequence_number": 1,
                "provenance_name": "SPOC"}]}
        if service == "Mast.Caom.Products":
            return {"status": "COMPLETE", "data": [{
                "obs_id": "tess-s0001-%s" % TIC, "productType": "SCIENCE",
                "productSubGroupDescription": "LC", "productFilename": filename,
                "dataURI": "mast:TESS/product/" + filename, "size": payload_size}]}
        raise AssertionError("unexpected service %s" % service)

    return query


def test_domain_is_atomically_onboarded_and_refuses_precedence():
    record = register_tess_domain()
    assert record.name == "tess_lightcurve"
    declaration = DOMAIN_DECLARATIONS.get("tess_lightcurve")
    assert set(declaration.violations) == {
        "no_natural_cycle", "irregular_sampling", "non_stationary_support"}
    assert declaration.provenance["time_scale"] == "BJD_TDB"
    with pytest.raises(PrecedenceNotAdmissibleError):
        declaration.assert_precedence_admissible()


def test_spec_identity_is_canonical_and_bounded():
    assert spec(target_id=" TIC 0261136679 ").request_sha256() == spec().request_sha256()
    with pytest.raises(InvalidParameterError, match="one to eight"):
        spec(sectors=())
    with pytest.raises(InvalidParameterError, match="536870912"):
        spec(max_download_bytes=1024 ** 3)


def test_metadata_preflight_names_exact_products_without_downloading_values():
    payload = fits_bytes()
    plan = inspect_tess_query(spec(), query=fake_query(len(payload)))
    assert plan["metadata_only"] is True
    assert plan["candidate_products"] == 1
    assert plan["predicted_download_bytes"] == len(payload)
    assert plan["products"][0]["filename"].endswith("s_lc.fits")


def test_sector_discovery_selects_exact_tic_products_without_downloading_values():
    calls = []

    def query(request):
        calls.append(request["service"])
        if request["service"] == "Mast.Caom.Filtered":
            return {"status": "COMPLETE", "data": [
                {"obsid": "3", "obs_id": "tess-s0001-0000000000000003",
                 "target_name": "TIC 3", "sequence_number": 1,
                 "provenance_name": "SPOC"},
                {"obsid": "1", "obs_id": "tess-s0001-0000000000000001",
                 "target_name": "TIC 1", "sequence_number": 1,
                 "provenance_name": "SPOC"},
                {"obsid": "2", "obs_id": "tess-s0001-0000000000000002",
                 "target_name": "TIC 2", "sequence_number": 1,
                 "provenance_name": "SPOC"},
            ]}
        obsid = str(request["params"]["obsid"])
        filename = "tess-s0001-%s-0001-s_lc.fits" % obsid.zfill(16)
        return {"status": "COMPLETE", "data": [{
            "obs_id": "tess-s0001-%s" % obsid,
            "productType": "SCIENCE", "productSubGroupDescription": "LC",
            "productFilename": filename, "dataURI": "mast:TESS/product/" + filename,
            "size": 1000 + int(obsid),
        }]}

    result = discover_tess_targets(sector=1, target_limit=2, query=query)

    assert result["metadata_only"] is True
    assert [row["tic_id"] for row in result["targets"]] == ["1", "2"]
    assert result["selected_targets"] == 2
    assert result["predicted_download_bytes"] == 2003
    assert calls == ["Mast.Caom.Filtered", "Mast.Caom.Products", "Mast.Caom.Products"]
    assert "exchangeability" in result["claim_boundary"]


def test_sector_discovery_is_bounded_and_reports_productless_candidates():
    def query(request):
        if request["service"] == "Mast.Caom.Filtered":
            return {"status": "COMPLETE", "data": [
                {"obsid": "7", "obs_id": "tess-s0001-0000000000000007",
                 "target_name": "TIC 7", "sequence_number": 1,
                 "provenance_name": "SPOC"},
            ]}
        return {"status": "COMPLETE", "data": []}

    result = discover_tess_targets(sector=1, target_limit=48, query=query)
    assert result["selected_targets"] == 0
    assert result["shortfall"] == 48
    assert result["refused_targets"][0]["tic_id"] == "7"

    with pytest.raises(InvalidParameterError, match="from 1 through"):
        discover_tess_targets(sector=1, target_limit=201, query=query)


def test_period_catalogue_is_joined_to_earliest_exact_spoc_product():
    catalogue = {
        "schema": "spectral.tess-period-candidates/v1", "response_sha256": "a" * 64,
        "candidates": [{"tic_id": "123", "toi": "1.01", "disposition": "KP",
                        "orbital_period_days": 2.0, "orbital_period_seconds": 172800.0,
                        "period_error_upper_days": 0.1, "period_error_lower_days": -0.1}
                       for _ in range(48)],
    }
    for index, row in enumerate(catalogue["candidates"]):
        row["tic_id"] = str(123 + index)

    def query(request):
        if request["service"] == "Mast.Caom.Filtered":
            target = request["params"]["filters"][-1]["values"][0]
            return {"status": "COMPLETE", "data": [{
                "obsid": target, "obs_id": "tess-s0002-%s" % target.zfill(16),
                "target_name": target, "sequence_number": 2, "provenance_name": "SPOC"}]}
        target = str(request["params"]["obsid"])
        filename = "tess-s0002-%s-test_lc.fits" % target.zfill(16)
        return {"status": "COMPLETE", "data": [{
            "obs_id": "obs-" + target, "productType": "SCIENCE",
            "productSubGroupDescription": "LC", "productFilename": filename,
            "dataURI": "mast:TESS/product/" + filename, "size": 2000}]}

    result = discover_periodic_tess_products(catalogue, target_limit=48, query=query)
    assert result["selected_targets"] == 48
    assert result["shortfall_from_g17_minimum"] == 0
    assert result["targets"][0]["orbital_period_seconds"] == 172800.0
    assert result["targets"][0]["sector"] == 2
    assert result["predicted_download_bytes"] == 96000


def test_periodic_product_expansion_excludes_an_existing_inventory():
    catalogue = {
        "schema": "spectral.tess-period-candidates/v1", "response_sha256": "a" * 64,
        "candidates": [{"tic_id": str(100 + index), "toi": "%d.01" % index,
                        "disposition": "KP", "orbital_period_days": 2.0,
                        "orbital_period_seconds": 172800.0,
                        "period_error_upper_days": 0.1, "period_error_lower_days": -0.1}
                       for index in range(49)],
    }

    def query(request):
        if request["service"] == "Mast.Caom.Filtered":
            target = request["params"]["filters"][-1]["values"][0]
            return {"status": "COMPLETE", "data": [{
                "obsid": target, "obs_id": "tess-s0001-%s" % target.zfill(16),
                "target_name": target, "sequence_number": 1, "provenance_name": "SPOC"}]}
        target = str(request["params"]["obsid"])
        filename = "tess-s0001-%s-test_lc.fits" % target.zfill(16)
        return {"status": "COMPLETE", "data": [{
            "obs_id": "obs-" + target, "productType": "SCIENCE",
            "productSubGroupDescription": "LC", "productFilename": filename,
            "dataURI": "mast:TESS/product/" + filename, "size": 2000}]}

    result = discover_periodic_tess_products(
        catalogue, target_limit=48, excluded_target_ids={"100"}, query=query)
    assert result["selected_targets"] == 48
    assert result["excluded_target_count"] == 1
    assert result["targets"][0]["tic_id"] == "101"


def test_discovery_bound_acquisition_validates_and_publishes_exact_fits_bytes(tmp_path):
    payload = fits_bytes()
    product = LightCurveProduct(
        observation_id="obs-1", data_uri="mast:TESS/product/test.fits",
        filename="tess-s0001-%s-test_lc.fits" % TIC.zfill(16), sector=1,
        size_bytes=len(payload))
    discovery = {
        "schema": "spectral.tess-pool-discovery.v1",
        "sector": 1,
        "selected_targets": 1,
        "predicted_download_bytes": len(payload),
        "targets": [{"tic_id": TIC, "observation_ids": ["obs-1"],
                     "products": [product.describe()],
                     "predicted_download_bytes": len(payload)}],
    }

    receipt = acquire_discovered_tess(
        discovery, root=tmp_path / "raw", maximum_total_bytes=len(payload),
        download=lambda _uri, _maximum: payload)

    assert receipt["target_count"] == 1
    assert receipt["downloaded_bytes"] == len(payload)
    assert receipt["targets"][0]["tic_id"] == TIC
    assert receipt["targets"][0]["timestamped_samples"] == 3
    assert receipt["targets"][0]["finite_quality_zero_flux"] == 2
    assert all(row["checksum_valid"] is True
               for row in receipt["targets"][0]["products"][0]["fits_checksum"])
    assert (tmp_path / "raw" / (hashlib.sha256(payload).hexdigest() + ".fits")).read_bytes() \
        == payload


def test_discovery_bound_acquisition_refuses_before_download_when_total_exceeds_cap(tmp_path):
    discovery = {
        "schema": "spectral.tess-pool-discovery.v1", "sector": 1,
        "selected_targets": 1, "predicted_download_bytes": 100,
        "targets": [{"tic_id": TIC, "observation_ids": ["obs-1"], "products": [],
                     "predicted_download_bytes": 100}],
    }
    calls = []
    with pytest.raises(InvalidParameterError, match="before any FITS product"):
        acquire_discovered_tess(
            discovery, root=tmp_path / "raw", maximum_total_bytes=99,
            download=lambda uri, maximum: calls.append((uri, maximum)))
    assert calls == []


def test_period_qualified_acquisition_retains_external_native_period(tmp_path):
    payload = fits_bytes()
    product = LightCurveProduct(
        observation_id="obs-1", data_uri="mast:TESS/product/test.fits",
        filename="tess-s0001-%s-test_lc.fits" % TIC.zfill(16), sector=1,
        size_bytes=len(payload))
    discovery = {
        "schema": "spectral.tess-period-product-discovery/v1",
        "selected_targets": 1, "predicted_download_bytes": len(payload),
        "targets": [{
            "tic_id": TIC, "toi": "100.01", "disposition": "KP",
            "orbital_period_seconds": 172800.0,
            "period_error_upper_days": 0.01, "period_error_lower_days": -0.01,
            "product": product.describe(), "predicted_download_bytes": len(payload),
        }],
    }

    receipt = acquire_discovered_tess(
        discovery, root=tmp_path / "raw", maximum_total_bytes=len(payload),
        download=lambda _uri, _maximum: payload)

    assert receipt["discovery_schema"] == discovery["schema"]
    assert receipt["targets"][0]["orbital_period_seconds"] == 172800.0


def test_period_qualified_acquisition_retains_each_targets_period_metadata(tmp_path):
    target_ids = (TIC, str(int(TIC) + 1))
    payloads = {target_id: fits_bytes(tic_id=target_id) for target_id in target_ids}
    targets = []
    for index, target_id in enumerate(target_ids):
        payload = payloads[target_id]
        product = LightCurveProduct(
            observation_id="obs-%d" % index,
            data_uri="mast:TESS/product/test-%s.fits" % target_id,
            filename="tess-s0001-%s-test_lc.fits" % target_id.zfill(16), sector=1,
            size_bytes=len(payload))
        targets.append({
            "tic_id": target_id, "toi": "%d.01" % (100 + index), "disposition": "KP",
            "orbital_period_seconds": 172800.0 + index,
            "period_error_upper_days": 0.01 + index,
            "period_error_lower_days": -0.01 - index,
            "product": product.describe(), "predicted_download_bytes": len(payload),
        })
    discovery = {
        "schema": "spectral.tess-period-product-discovery/v1",
        "selected_targets": 2,
        "predicted_download_bytes": sum(len(payload) for payload in payloads.values()),
        "targets": targets,
    }

    receipt = acquire_discovered_tess(
        discovery, root=tmp_path / "raw",
        maximum_total_bytes=discovery["predicted_download_bytes"],
        download=lambda uri, _maximum: payloads[uri.rsplit("-", 1)[-1][:-5]])

    assert [(row["tic_id"], row["toi"], row["orbital_period_seconds"])
            for row in receipt["targets"]] == [
                (target_ids[0], "100.01", 172800.0),
                (target_ids[1], "101.01", 172801.0),
            ]


def test_period_qualified_assessment_binds_the_adopted_methods(tmp_path):
    payload = fits_bytes()
    product = LightCurveProduct(
        observation_id="obs-1", data_uri="mast:TESS/product/test.fits",
        filename="tess-s0001-%s-test_lc.fits" % TIC.zfill(16), sector=1,
        size_bytes=len(payload))
    discovery = {
        "schema": "spectral.tess-period-product-discovery/v1",
        "selected_targets": 1, "predicted_download_bytes": len(payload),
        "targets": [{
            "tic_id": TIC, "toi": "100.01", "disposition": "KP",
            "orbital_period_seconds": 100.0,
            "period_error_upper_days": 0.01, "period_error_lower_days": -0.01,
            "product": product.describe(), "predicted_download_bytes": len(payload),
        }],
    }
    receipt = acquire_discovered_tess(
        discovery, root=tmp_path / "raw", maximum_total_bytes=len(payload),
        download=lambda _uri, _maximum: payload)
    receipt_path = tmp_path / "acquisition.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assessment = assess_tess_pool_acquisition(receipt_path, tmp_path / "raw")

    assert assessment["candidate_count"] == 1
    assert assessment["profile_method_status"] == "ADOPTED"
    assert assessment["method_review"]["adopted_by"] == "Ed Bentley"
    assert assessment["records_reaching_minimum"] == 0
    assert "no g17 profile is emitted" in assessment["claim_boundary"].lower()


def test_acquisition_receipts_merge_without_duplicate_targets(tmp_path):
    first = {"schema": "spectral.tess-pool-acquisition/v1",
             "discovery_schema": "spectral.tess-period-product-discovery/v1",
             "target_count": 1, "downloaded_bytes": 10,
             "targets": [{"tic_id": "1"}], "claim_boundary": "raw only"}
    second = {**first, "downloaded_bytes": 20, "targets": [{"tic_id": "2"}]}
    paths = []
    for index, body in enumerate((first, second)):
        path = tmp_path / ("receipt-%d.json" % index)
        path.write_text(json.dumps(body), encoding="utf-8")
        paths.append(path)

    merged = merge_tess_acquisitions(paths)
    assert merged["target_count"] == 2
    assert merged["downloaded_bytes"] == 30
    assert [row["tic_id"] for row in merged["targets"]] == ["1", "2"]

    second["targets"] = [{"tic_id": "1"}]
    paths[1].write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(UserInputError, match="unique"):
        merge_tess_acquisitions(paths)


def test_qualification_records_targets_that_fail_corrected_cycle_coverage(
        tmp_path, monkeypatch):
    receipt = {
        "schema": "spectral.tess-pool-acquisition-collection/v1",
        "targets": [{"tic_id": "1"}, {"tic_id": "2"}],
    }
    path = tmp_path / "merged.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    def profile(target, _raw_root):
        if target["tic_id"] == "2":
            raise UserInputError("TESS candidate spans fewer than eight declared orbital cycles")

    monkeypatch.setattr("src.core.tess_pool_assessment._profile", profile)
    result = qualify_tess_acquisition(path, tmp_path)

    assert result["target_count"] == 1
    assert result["targets"] == [{"tic_id": "1"}]
    assert result["refused_target_count"] == 1
    assert result["refused_targets"][0]["tic_id"] == "2"


def test_adopted_methods_export_canonical_source_bound_profile(tmp_path):
    payload = fits_bytes(times=np.linspace(1.0, 1.02, 20), quality=[0] * 20)
    product = LightCurveProduct(
        observation_id="obs-1", data_uri="mast:TESS/product/test.fits",
        filename="tess-s0001-%s-test_lc.fits" % TIC.zfill(16), sector=1,
        size_bytes=len(payload))
    discovery = {
        "schema": "spectral.tess-period-product-discovery/v1",
        "selected_targets": 1, "predicted_download_bytes": len(payload),
        "targets": [{
            "tic_id": TIC, "toi": "100.01", "disposition": "KP",
            "orbital_period_seconds": 100.0,
            "period_error_upper_days": 0.01, "period_error_lower_days": -0.01,
            "product": product.describe(), "predicted_download_bytes": len(payload),
        }],
    }
    receipt = acquire_discovered_tess(
        discovery, root=tmp_path / "raw", maximum_total_bytes=len(payload),
        download=lambda _uri, _maximum: payload)
    receipt_path = tmp_path / "acquisition.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = export_tess_partner_profiles(
        receipt_path, raw_root=tmp_path / "raw", record_root=tmp_path / "records",
        profile_root=tmp_path / "profiles")

    assert result["profile_count"] == 1
    profile_path = tmp_path / "profiles" / ("tic-%s.partner-profile.json" % TIC)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    record = tmp_path / "records" / ("tic-%s.csv" % TIC)
    assert profile["source"]["content_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert profile["method_review"]["adopted_by"] == "Ed Bentley"
    assert profile["profile"]["provenance_key"].endswith(
        receipt["targets"][0]["products"][0]["content_sha256"])

    with pytest.raises(UserInputError, match="immutable"):
        export_tess_partner_profiles(
            receipt_path, raw_root=tmp_path / "raw", record_root=tmp_path / "records",
            profile_root=tmp_path / "profiles")


def test_acquisition_parses_archive_time_flags_and_binds_every_value():
    first_bytes = fits_bytes()
    second_bytes = fits_bytes(flux_offset=7.0)
    first = acquire_tess(spec(), query=fake_query(len(first_bytes)),
                         download=lambda _uri, _maximum: first_bytes)
    second = acquire_tess(spec(), query=fake_query(len(second_bytes)),
                          download=lambda _uri, _maximum: second_bytes)
    assert np.allclose(first.times_bjd_tdb, [2457001.0, 2457001.01, 2457001.03])
    assert first.quality.tolist() == [0, 32, 0]
    assert first.source_sha256 == (hashlib.sha256(first_bytes).hexdigest(),)
    assert first.collection_sha256() != second.collection_sha256(), \
        "equal-length products with different flux values must never collide"
    parsed = parse_spoc_fits(first_bytes, spec(), LightCurveProduct(
        observation_id="obs", data_uri="mast:TESS/product/test.fits",
        filename="test.fits", sector=1, size_bytes=len(first_bytes)))
    assert all(row["checksum_valid"] is True for row in parsed["fits_checksum"])


def test_cadences_the_archive_never_timestamped_are_dropped_and_counted():
    """D95, found by the first real MAST acquisition in TG17.14.

    SPOC emits one row per cadence in the window, including the ones no photometry was
    produced for, and those carry a non-finite TIME. The real sector-1 product for TIC
    261136679 has 815 of them in 20,076 rows. Every synthetic fixture in this suite had a
    clean clock, so `acquire_tess` had never met one and handed all of them to a collection
    whose stated invariant is a finite strictly increasing clock. The gate refused the whole
    domain rather than record a light curve, which is the invariant working.

    A row with no timestamp cannot be placed on a clock, so it is dropped -- but the count
    is a property of the record and travels with it rather than vanishing.
    """
    payload = fits_bytes(times=[1.0, float("nan"), 1.01, float("nan"), 1.03],
                         quality=[0, 128, 0, 128, 0])
    collection = acquire_tess(spec(), query=fake_query(len(payload)),
                              download=fake_download(payload))
    assert collection.n_samples == 3
    assert collection.unclocked_samples_dropped == 2
    assert np.all(np.isfinite(collection.times_bjd_tdb))
    assert np.all(np.diff(collection.times_bjd_tdb) > 0)
    assert collection.describe()["unclocked_samples_dropped"] == 2
    assert collection.canonical()["unclocked_samples_dropped"] == 2
    # The count survives a round trip, or a receipt could lose what was discarded.
    assert LightCurveCollection.from_canonical(
        collection.canonical()).unclocked_samples_dropped == 2


def test_dropping_untimestamped_rows_does_not_weaken_the_duplicate_clock_check():
    """The check that D95's fix must not cost.

    Worth stating what this is *not*, because the first reading of D95 got it wrong: the
    duplicate check was never blind to untimestamped rows. numpy sorts NaN to the end, so
    the finite prefix was always compared correctly, and `[1.0, nan, 1.0]` refused before
    this slice exactly as it does after. What the drop must not do is remove a duplicate
    from view by removing the rows around it, so the duplicate is placed beside the gap.
    """
    payload = fits_bytes(times=[1.0, float("nan"), 1.0, 1.03], quality=[0, 128, 0, 0])
    with pytest.raises(InvalidParameterError, match="duplicate or reversed"):
        acquire_tess(spec(), query=fake_query(len(payload)), download=fake_download(payload))


def test_a_product_with_no_timestamped_cadence_at_all_is_refused_by_name():
    payload = fits_bytes(times=[float("nan")] * 3, quality=[128, 128, 128])
    with pytest.raises(InvalidParameterError, match="actually timestamped"):
        acquire_tess(spec(), query=fake_query(len(payload)), download=fake_download(payload))


def test_download_caps_refuse_before_a_product_is_opened():
    payload = fits_bytes()
    downloads = []
    with pytest.raises(InvalidParameterError, match="before values are fetched"):
        acquire_tess(spec(max_download_bytes=len(payload) - 1),
                     query=fake_query(len(payload)),
                     download=lambda uri, maximum: downloads.append((uri, maximum)))
    assert downloads == []


def test_collection_rejects_duplicate_clock_and_round_trips_canonical_bytes(tmp_path: Path):
    payload = fits_bytes()
    collection = acquire_tess(spec(), query=fake_query(len(payload)),
                              download=lambda _uri, _maximum: payload)
    receipt = persist_collection(collection, tmp_path)
    loaded = load_collection(receipt["collection_sha256"], tmp_path)
    assert loaded.collection_sha256() == collection.collection_sha256()
    assert persist_collection(collection, tmp_path)["publication"] == "already-present-identical"
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        LightCurveCollection(
            spec=collection.spec, products=collection.products,
            times_bjd_tdb=[1.0, 1.0], flux=[1.0, 2.0], flux_error=[0.1, 0.1],
            quality=[0, 0], sectors=[1, 1], source_sha256=collection.source_sha256,
            target_ra_deg=RA, target_dec_deg=DEC)


def test_source_is_registered_with_bounded_capabilities():
    source = register_tess_source()
    assert LIGHTCURVE_SOURCES.get("mast_tess_spoc") is source
    assert source.describe()["metadata_preflight"] is True


def test_mast_metadata_query_retries_one_transient_transport_timeout(monkeypatch):
    import src.data_layer.tess_source as source

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self, _limit):
            return json.dumps({"status": "COMPLETE", "data": []}).encode()

    calls = []
    def opened(_request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("transient MAST timeout")
        return Response()

    monkeypatch.setenv(source.NETWORK_ENV_VAR, "1")
    monkeypatch.setattr(source, "urlopen", opened)
    monkeypatch.setattr(source.time, "sleep", lambda _seconds: None)
    result = source._mast_query({"service": "Mast.Caom.Filtered.Position", "params": {}})
    assert result["status"] == "COMPLETE"
    assert calls == [source.MAST_TIMEOUT_SECONDS, source.MAST_TIMEOUT_SECONDS]


def test_api_exposes_refusals_and_never_claims_acquisition_is_a_finding(monkeypatch, tmp_path):
    import src.api.lightcurves as api

    payload = fits_bytes()
    collection = acquire_tess(spec(), query=fake_query(len(payload)),
                              download=lambda _uri, _maximum: payload)
    monkeypatch.setattr(api, "acquire_tess", lambda _spec: collection)
    monkeypatch.setattr(api, "persist_collection",
                        lambda value: persist_collection(value, tmp_path))
    client = TestClient(app)
    capabilities = client.get("/api/v1/lightcurves")
    assert capabilities.status_code == 200
    assert {row["basis"] for row in capabilities.json()["refusals"]} >= {
        "violation:no_natural_cycle", "violation:irregular_sampling",
        "violation:non_stationary_support", "lag_policy:none"}
    response = client.post("/api/v1/lightcurves/acquire", json={
        "target_id": TIC, "sectors": [1], "max_products": 2,
        "max_download_bytes": 8 * 1024 * 1024})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_readiness"]["calendar_climatology"].startswith("refused")
    assert "no variability" in body["claim_boundary"]


@pytest.mark.live
def test_live_mast_tess_product_when_network_is_explicitly_enabled():
    if os.getenv("SPECTRALEARTH_RUN_LIVE_TESS") != "1":
        pytest.skip("set SPECTRALEARTH_RUN_LIVE_TESS=1 for bounded live MAST acceptance")
    live = LightCurveSpec(target_id=TIC, sectors=(1,), max_products=2,
                          max_download_bytes=64 * 1024 * 1024)
    plan = inspect_tess_query(live)
    assert 1 <= plan["candidate_products"] <= 2
    assert plan["within_byte_cap"] is True
    collection = acquire_tess(live)
    assert collection.n_samples > 100
    assert collection.source_sha256
    assert np.any(collection.quality != 0)
