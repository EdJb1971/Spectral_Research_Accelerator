import pytest

from src.core.errors import InvalidParameterError
from src.data_layer.exoplanet_period_source import discover_periodic_tess_candidates


def test_period_candidates_are_external_to_flux_and_resolve_eight_cycles():
    payload = (
        "tid,toi,tfopwg_disp,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbperlim\n"
        "123,100.01,KP,2.0,0.01,-0.01,0\n"
        "456,101.01,PC,3.0,,,0\n").encode()
    seen = []

    result = discover_periodic_tess_candidates(
        candidate_limit=48, nominal_sector_days=27.0, minimum_cycles=8.0,
        fetch=lambda url, maximum: seen.append((url, maximum)) or payload)

    assert [row["tic_id"] for row in result["candidates"]] == ["123", "456"]
    assert result["candidates"][0]["orbital_period_seconds"] == 172800.0
    assert result["maximum_period_days"] == 3.375
    assert "not estimated" in result["native_seconds_basis"]
    assert "tfopwg_disp%3C%3E%27FP%27" in seen[0][0]


def test_period_candidate_query_refuses_underresolved_or_malformed_inputs():
    with pytest.raises(InvalidParameterError, match="calibrated"):
        discover_periodic_tess_candidates(minimum_cycles=7.0, fetch=lambda *_: b"")
    with pytest.raises(InvalidParameterError, match="declared CSV"):
        discover_periodic_tess_candidates(fetch=lambda *_: b"wrong,columns\n1,2\n")


def test_multi_period_host_is_refused_instead_of_duplicated_or_chosen():
    payload = (
        "tid,toi,tfopwg_disp,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbperlim\n"
        "123,100.01,KP,2.0,,,0\n"
        "123,100.02,KP,3.0,,,0\n"
        "456,101.01,PC,1.0,,,0\n").encode()

    result = discover_periodic_tess_candidates(fetch=lambda *_: payload)

    assert [row["tic_id"] for row in result["candidates"]] == ["456"]
    assert result["ambiguous_host_count"] == 1
    assert result["refused_ambiguous_hosts"][0]["tic_id"] == "123"
    assert len(result["refused_ambiguous_hosts"][0]["toi_periods"]) == 2