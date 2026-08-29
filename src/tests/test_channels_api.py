"""TG8.4: the ingestion seam over HTTP (`ed-dev`).

`test_tabular_domain.py` covers the adapter. This covers the wire: that the two-call shape does
not choose for the caller, that a refusal reaches the researcher in the adapter's own words
rather than as "invalid file", and that a successful read carries the domain's limits and the
attribution caveat with it.

The property worth stating separately, because it is the one a reader has to trust: **the
refusal `/inspect` advertises and the refusal `/read` enforces are the same refusal.** An
inspection that promised a domain would work and then a read that refused it would be worse than
no inspection at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.channels import PREVIEW_ROWS
from src.api.main import app
from src.core.builtin_domains import register_builtin_domains
from src.core.domain import DOMAIN_DECLARATIONS
from src.core.onboarding import DOMAIN_ONBOARDINGS
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES

REGULAR = "t,bid,ask\n0,1,2\n1,1.5,2.5\n2,2,3\n3,2.5,3.5\n"
IRREGULAR = "t,bid,ask\n0,1,2\n1,1.5,2.5\n5,2,3\n9,2.5,3.5\n"
BACKWARDS = "t,bid,ask\n0,1,2\n5,1.5,2.5\n3,2,3\n"
NON_NUMERIC = "t,bid,ask\n0,1,2\n1,oops,2.5\n2,2,3\n"


@pytest.fixture()
def client():
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    try:
        yield TestClient(app)
    finally:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


def _inspect(client, text, name="record.csv", **form):
    return client.post("/api/v1/channels/inspect",
                       files={"file": (name, text, "text/csv")}, data=form)


def _read(client, text, name="record.csv", **form):
    return client.post("/api/v1/channels/read",
                       files={"file": (name, text, "text/csv")}, data=form)


# ------------------------------------------------------------------ inspect chooses nothing


def test_inspect_reports_the_file_without_choosing_a_domain(client):
    body = _inspect(client, REGULAR).json()
    assert body["columns"] == ["t", "bid", "ask"]
    assert body["n_rows"] == 4
    assert body["candidate_time_columns"] == ["t", "bid", "ask"]
    assert body["clock"]["regular"] is True
    assert body["clock"]["cadence_seconds"] == 1.0
    assert body["required_violations"] == []
    # Every onboarded domain is reported. None is selected.
    assert {row["name"] for row in body["domains"]} >= {"order_book", "reanalysis"}
    assert "selected_domain" not in body


def test_inspect_turns_an_irregular_clock_into_an_obligation(client):
    body = _inspect(client, IRREGULAR).json()
    assert body["required_violations"] == ["irregular_sampling"]
    assert body["clock"]["cadence_seconds"] is None
    by_name = {row["name"]: row for row in body["domains"]}
    assert by_name["order_book"]["admits"] is True
    assert by_name["reanalysis"]["admits"] is False


def test_inspect_refuses_to_substitute_a_clock_column(client):
    """`bid` increases strictly while `t` runs backwards; promoting it would be invisible."""
    body = _inspect(client, BACKWARDS).json()
    assert body["readable"] is False
    assert "cannot serve as a clock" in body["refused_because"]
    assert "bid" in body["refused_because"], "the usable candidates must still be named"
    assert body["domains"] == []


def test_inspect_honours_a_clock_column_the_caller_names(client):
    """Naming one explicitly is the supported path when the file is not in the usual order."""
    # `seq` is numeric but not increasing, so it is no candidate; `t` is the clock and sits in
    # the second column, which is exactly the case naming one explicitly exists for.
    body = _inspect(client, "seq,t,v\n5,0,1\n3,1,2\n4,2,3\n", time_column="t").json()
    assert body["readable"] is True
    assert body["time_column"] == "t"
    assert body["channel_columns"] == ["seq", "v"]
    assert body["candidate_time_columns"] == ["t", "v"]


def test_inspect_states_the_aggregate_obligation_it_cannot_detect(client):
    body = _inspect(client, REGULAR).json()
    assert "aggregated_values" in body["aggregate_note"]
    assert "aggregated_values" not in body["required_violations"]


def test_inspect_carries_the_attribution_caveat(client):
    from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT

    assert _inspect(client, REGULAR).json()["attribution_caveat"] == DOMAIN_ATTRIBUTION_CAVEAT


# ------------------------------------------------------------------ inspect predicts the read


def test_the_advertised_refusal_is_the_enforced_refusal(client):
    """The property the whole two-call shape rests on.

    A third domain is onboarded first, and that is not decoration. With only the two built-ins
    present, every domain that admits an irregular record also declares `irregular_sampling` and
    every domain that refuses one refuses it on its axes, so a check that stopped reporting the
    clock obligation would still agree with itself. `tidy_venue` is the case where inspect and
    read can disagree, which is the only case this test is really about.
    """
    _onboard_tidy_venue()
    for text in (REGULAR, IRREGULAR):
        body = _inspect(client, text).json()
        assert any(row["admits"] for row in body["domains"]), "must not pass vacuously"
        for row in body["domains"]:
            response = _read(client, text, domain=row["name"], time_column="t")
            assert response.status_code == (200 if row["admits"] else 400), (
                "%s: inspect said admits=%s" % (row["name"], row["admits"]))


# ------------------------------------------------------------------ reading


def test_a_record_loads_under_a_domain_that_admits_it(client):
    body = _read(client, REGULAR, domain="order_book", time_column="t").json()
    assert body["domain"] == "order_book"
    assert body["n_rows"] == 4
    assert [channel["name"] for channel in body["channels"]] == ["bid", "ask"]
    assert body["times_seconds"] == [0.0, 1.0, 2.0, 3.0]
    assert len(body["content_sha256"]) == 64
    assert len(body["onboarding_sha256"]) == 64


def test_a_loaded_record_carries_what_its_domain_refuses(client):
    body = _read(client, REGULAR, domain="order_book", time_column="t").json()
    limits = body["domain_limits"]
    assert limits["precedence_admissible"] is False
    bases = {item["basis"] for item in limits["refuses"]}
    assert "violation:no_physical_metric" in bases
    assert "lag_policy:none" in bases


def test_a_loaded_record_says_the_plot_is_not_an_analysis(client):
    body = _read(client, REGULAR, domain="order_book", time_column="t").json()
    assert "not an analysis" in body["preview_note"]
    assert "R22" in body["preview_note"]


def test_a_loaded_record_refuses_to_imply_the_domain_produced_it(client):
    from src.api.findings import DOMAIN_ATTRIBUTION_CAVEAT

    body = _read(client, REGULAR, domain="order_book", time_column="t").json()
    assert body["domain_limits"]["attribution_caveat"] == DOMAIN_ATTRIBUTION_CAVEAT
    assert "Nothing here establishes that it came from it" in \
        body["provenance"]["domain_attribution"]


def test_an_aggregate_channel_is_marked_and_requires_the_declaration(client):
    ok = _read(client, REGULAR, domain="order_book", time_column="t",
               support_parent_px='{"bid": 60}').json()
    by_name = {channel["name"]: channel for channel in ok["channels"]}
    assert by_name["bid"]["is_aggregate"] is True
    assert by_name["bid"]["support_parent_px"] == 60.0
    assert by_name["ask"]["is_aggregate"] is False


def test_an_irregular_record_reports_no_cadence_rather_than_a_number(client):
    body = _read(client, IRREGULAR, domain="order_book", time_column="t").json()
    assert body["clock"]["regular"] is False
    assert body["clock"]["cadence_seconds"] is None
    assert body["provenance"]["cadence_seconds"] is None


def test_the_preview_is_a_head_and_says_how_much_it_withheld(client):
    body = _read(client, REGULAR, domain="order_book", time_column="t").json()
    assert body["preview_rows"] == 4
    assert body["rows_withheld"] == 0
    assert PREVIEW_ROWS > 0
    for channel in body["channels"]:
        assert len(channel["values"]) == body["preview_rows"]


# ------------------------------------------------------------------ refusals reach the reader


def test_a_domain_that_cannot_describe_a_channel_table_is_refused_in_its_own_words(client):
    response = _read(client, REGULAR, domain="reanalysis", time_column="t")
    assert response.status_code == 400
    detail = str(response.json()["detail"])
    assert "channel table" in detail and "latitude" in detail
    assert "E14" in detail


def _onboard_tidy_venue():
    """A channel-table domain that does *not* declare an irregular clock.

    Needed because both built-ins agree about irregular records — `order_book` declares the
    violation and `reanalysis` is refused on its axes long before the clock is reached — so
    neither can show the difference between a domain that admits a ragged clock and one that
    does not.
    """
    from src.core.builtin_glossaries import ORDER_BOOK_PHRASES
    from src.core.domain import AxisSpec, DomainDeclaration
    from src.core.onboarding import onboard_domain

    return onboard_domain(
        DomainDeclaration(
            name="tidy_venue", description="A venue with a metronomic clock, for the test.",
            axes=(AxisSpec(name="t", role="time", units="s"),
                  AxisSpec(name="channel", role="category", ordered=False)),
            licence="Fixture licence.", violations=("no_physical_metric",), lag_policy="none"),
        ORDER_BOOK_PHRASES, geometry=None, onboarded_by=__name__)


def test_an_irregular_record_is_refused_where_the_domain_did_not_declare_it(client):
    _onboard_tidy_venue()

    response = _read(client, IRREGULAR, domain="tidy_venue", time_column="t")
    assert response.status_code == 400
    detail = str(response.json()["detail"])
    assert "irregular_sampling" in detail
    # The adapter's reasoning survives to the researcher rather than becoming "invalid file".
    assert "interpolation upstream of a dependence estimator" in detail


def test_a_non_numeric_channel_is_refused_with_the_reason_imputation_is_not_offered(client):
    response = _read(client, NON_NUMERIC, domain="order_book", time_column="t")
    assert response.status_code == 400
    assert "imputed value" in str(response.json()["detail"])


def test_an_unonboarded_domain_is_refused_and_points_at_the_contract(client):
    response = _read(client, REGULAR, domain="no_such_domain", time_column="t")
    assert response.status_code == 400
    assert "findings/onboarding" in str(response.json()["detail"])


def test_a_missing_clock_column_names_the_columns_the_file_declares(client):
    response = _read(client, REGULAR, domain="order_book", time_column="timestamp")
    assert response.status_code == 400
    assert "bid" in str(response.json()["detail"])


def test_a_binary_upload_is_sent_to_the_route_that_reads_binary(client):
    response = client.post("/api/v1/channels/read",
                           files={"file": ("field.nc", b"\x89HDF\r\n\x1a\n\x00\xff",
                                           "application/octet-stream")},
                           data={"domain": "order_book", "time_column": "t"})
    assert response.status_code == 400
    assert "/api/v1/import/inspect" in str(response.json()["detail"])


def test_malformed_json_form_fields_are_refused_with_an_example(client):
    response = _read(client, REGULAR, domain="order_book", time_column="t",
                     support_parent_px="{not json}")
    assert response.status_code == 400
    assert "trade_count" in str(response.json()["detail"])


def test_a_support_map_that_is_not_an_object_is_refused(client):
    response = _read(client, REGULAR, domain="order_book", time_column="t",
                     support_parent_px="[1, 2]")
    assert response.status_code == 400
    assert "JSON object" in str(response.json()["detail"])


# ------------------------------------------------------------------ the wire guard


def test_a_channel_named_confidence_is_not_mistaken_for_a_claim(client):
    """R9's wire guard screens mapping keys, and a column heading is not a claim.

    Channels are served as a list of named entries rather than a mapping keyed by channel name
    precisely so an honestly-named `confidence` column cannot be refused as a bare confidence
    figure. This is the test that keeps that shape from being "simplified" back into a dict.
    """
    text = "t,confidence,ask\n0,1,2\n1,1.5,2.5\n2,2,3\n"
    response = _read(client, text, domain="order_book", time_column="t")
    assert response.status_code == 200
    assert [c["name"] for c in response.json()["channels"]] == ["confidence", "ask"]


def test_reading_is_read_only(client):
    """A GET does not move a rung, and neither does this POST: nothing is stored (R22)."""
    from src.api.findings import study_root

    before = sorted(p.name for p in study_root().glob("*")) if study_root().exists() else []
    _read(client, REGULAR, domain="order_book", time_column="t")
    after = sorted(p.name for p in study_root().glob("*")) if study_root().exists() else []
    assert before == after
