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
from src.core.errors import InvalidParameterError
from src.data_layer.lightcurves import (LIGHTCURVE_SOURCES, LightCurveCollection,
                                        LightCurveProduct, LightCurveSpec, load_collection,
                                        persist_collection)
from src.data_layer.tess_source import (acquire_tess, inspect_tess_query,
                                        register_tess_source)

TIC = "261136679"
RA, DEC = 84.291188, -80.46912


def spec(**changes) -> LightCurveSpec:
    values = dict(target_id=TIC, sectors=(1,), max_products=2,
                  max_download_bytes=8 * 1024 * 1024)
    values.update(changes)
    return LightCurveSpec(**values)


def fits_bytes(flux_offset: float = 0.0) -> bytes:
    primary = fits.PrimaryHDU()
    primary.header["TICID"] = int(TIC)
    primary.header["RA_OBJ"] = RA
    primary.header["DEC_OBJ"] = DEC
    columns = [
        fits.Column(name="TIME", format="D", array=np.array([1.0, 1.01, 1.03])),
        fits.Column(name="PDCSAP_FLUX", format="E",
                    array=np.array([100.0, 101.0 + flux_offset, 99.5], dtype=np.float32)),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E",
                    array=np.array([0.1, 0.1, 0.2], dtype=np.float32)),
        fits.Column(name="SAP_FLUX", format="E",
                    array=np.array([110.0, 111.0, 109.5], dtype=np.float32)),
        fits.Column(name="SAP_FLUX_ERR", format="E",
                    array=np.array([0.2, 0.2, 0.3], dtype=np.float32)),
        fits.Column(name="QUALITY", format="J", array=np.array([0, 32, 0], dtype=np.int32)),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["BJDREFI"] = 2457000
    table.header["BJDREFF"] = 0.0
    table.header["TIMESYS"] = "TDB"
    output = io.BytesIO()
    fits.HDUList([primary, table]).writeto(output, checksum=True)
    return output.getvalue()


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
