"""TG11.1: the domain-analysis engine through HTTP, without a write path."""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from src.benchmarks import all_benchmarks
from src.core.builtin_domains import register_builtin_domains
from src.core.builtin_glossaries import ORDER_BOOK_PHRASES
from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES


@pytest.fixture(autouse=True)
def isolated_domains():
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    try:
        yield
    finally:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


def _declared_domain():
    return onboard_domain(
        DomainDeclaration(
            name="instrument_log", description="Regular synthetic instrument channels.",
            axes=(AxisSpec("t", "time", units="s"),
                  AxisSpec("channel", "category", ordered=False)),
            licence="Fixture licence; no real archive was accessed.",
            violations=("no_physical_metric", "no_propagation_speed", "no_natural_cycle"),
            lag_policy="declared", declared_floor_frames=1,
            declared_floor_basis="one instrument reporting interval",
            provenance={"declared_by": __name__}),
        ORDER_BOOK_PHRASES, geometry=None, onboarded_by=__name__)


def _record(n=100, planted=False):
    rng = np.random.default_rng(20260827)
    alpha = rng.normal(size=n)
    beta = rng.normal(size=n)
    if planted:
        beta[2:] = alpha[:-2] + 0.05 * beta[2:]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["t", "alpha", "beta"])
    for i in range(n):
        writer.writerow([i, alpha[i], beta[i]])
    return output.getvalue()


def _run(client, text, *, operation, domain, configuration):
    return client.post(
        "/api/v1/analysis/run",
        files={"file": ("record.csv", text, "text/csv")},
        data={"operation": operation, "domain": domain, "time_column": "t",
              "configuration": __import__("json").dumps(configuration)})


def test_capabilities_state_the_read_only_claim_boundary(client):
    body = client.get("/api/v1/analysis").json()
    assert set(body["operations"]) == {"association", "precedence", "domain_gate"}
    assert body["read_only"] is True and body["stores"] is False
    assert body["moves_rung"] is False
    assert "TG11.2" in body["claim_boundary"] and "R22" in body["claim_boundary"]


def test_association_only_runs_over_the_full_admitted_record(client):
    response = _run(client, _record(), operation="association", domain="order_book",
                    configuration={"lags": [1, 2], "bins": 2, "n_surrogates": 99,
                                   "seed": 7})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"]["frames"] == 100
    assert body["result"]["claim_boundary"] == "association"
    assert body["result"]["n_tests"] == 4
    assert any("must not be described as precedence" in item
               for item in body["result"]["warnings"])
    assert body["read_only"] is True and body["stored"] is False
    assert body["rung_moved"] is False


def test_precedence_runs_only_when_the_declared_floor_admits_it(client):
    _declared_domain()
    response = _run(client, _record(planted=True), operation="precedence",
                    domain="instrument_log",
                    configuration={"lags": [2], "bins": 2, "n_surrogates": 99,
                                   "seed": 8})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["claim_boundary"] == "precedence"
    assert result["applied_lag_floor"]["policy"] == "declared"
    assert result["n_tests"] == 2


def test_r21_refusal_reaches_the_http_caller_before_precedence(client):
    response = _run(client, _record(), operation="precedence", domain="order_book",
                    configuration={"lags": [1], "bins": 2, "n_surrogates": 99})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "rule R21 forbids" in detail
    assert "Measure association instead" in detail


def test_domain_gate_derives_record_facts_and_returns_three_valued_verdict(client):
    response = _run(client, _record(), operation="domain_gate", domain="order_book",
                    configuration={"study_id": "http-null", "lags": [1], "bins": 2,
                                   "n_surrogates": 99, "embargo_frames": 1, "seed": 9})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["verdict"] in {"PASS", "FAIL", "INVALID"}
    assert body["result"]["protocol"]["n_scales"] == 2
    assert body["result"]["protocol"]["expected_frames"] == 100
    assert body["result"]["protocol"]["cadence_seconds"] == 1.0
    assert body["result"]["train"]["claim_boundary"] == "association"


def test_unknown_configuration_is_refused_not_silently_ignored(client):
    response = _run(client, _record(), operation="association", domain="order_book",
                    configuration={"lags": [1], "lagz": [9]})
    assert response.status_code == 400
    assert "Unknown settings are refused rather than ignored" in response.json()["detail"]


def test_every_sequence_and_cross_domain_benchmark_passes_through_http(client):
    """TG11.1 acceptance: the scientific ground truth crosses the actual HTTP boundary.

    Defect **D78**: this read `assert len(names) == 13`, a literal written at TG11.1. TG17.0
    registered `multidomain_flagship_planted` and `multidomain_flagship_safeguards`, the count
    became 15, and the assertion aborted the test *before the HTTP call* - so the two benchmarks
    that carry the four-domain flagship's planted and safeguard answers were never once exercised
    across the API boundary this test exists to exercise. A guard whose coverage shrinks when new
    work arrives is worse than no guard, because it reports a failure that looks like a stale
    number and hides a gap that is not.

    The count is now derived from the registry, with a floor so coverage cannot silently shrink,
    and the flagship benchmarks are named because they are the ones the omission cost.
    """
    names = [b.name for b in all_benchmarks() if b.kind in ("sequence", "cross_domain")]
    assert len(names) >= 13, "sequence and cross-domain benchmark coverage has shrunk"
    assert {"multidomain_flagship_planted", "multidomain_flagship_safeguards"} <= set(names)
    response = client.post("/api/v1/benchmarks/run",
                           params=[("name", name) for name in names])
    assert response.status_code == 200, response.text
    body = response.json()
    assert {row["name"] for row in body["benchmarks"]} == set(names)
    assert body["failed"] == 0
    assert body["not_yet_runnable"] == 0
    assert body["null_failures"] == []
    nulls = [row for row in body["benchmarks"] if row["is_null"]]
    assert nulls and all(check["outcome"] == "PASS" for row in nulls
                         for check in row["checks"])
