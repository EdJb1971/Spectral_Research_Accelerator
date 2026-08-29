"""TG9.1: the read-only claim surface.

The two acceptance criteria of the slice are
``test_no_route_can_serve_a_confidence_without_the_figures_that_give_it_meaning`` and
``test_a_new_domain_reaches_the_api_without_editing_src_api``. Everything else supports them.

The first matters because R9 is usually described as a frontend constraint, which puts it in the
one place it cannot be enforced. Here it is enforced on the wire, structurally, over every
response body a route can produce - so a handler written later by someone who has not read R9
still cannot leak a bare confidence.

The second is TG8.1's onboarding condition carried onto HTTP: a domain that registers a glossary
appears in `GET /domains` with no edit to `src/api/`. It is asserted by actually registering one
from this test module, which is outside `src/core` and outside `src/api` both.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.findings import (
    R9_FIGURES,
    STUDY_ROOT_ENV,
    StudyStore,
    refuse_bare_confidence,
)
from src.api.main import app
from src.core.builtin_glossaries import register_builtin_glossaries
from src.core.errors import InvalidParameterError
from src.core.evidence import save_evidence_bundle
from src.core.five_outputs import summarise_evidence
from src.core.registry import restore, snapshot
from src.core.translation import (
    DOMAIN_GLOSSARIES,
    STRUCTURAL_VOCABULARY,
    DomainGlossary,
)
from src.tests.test_five_outputs import _climbed
from src.tests.test_translation import _blocked, _contradicted, _figures


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A study directory of this test's own, so no researcher's real studies are ever read."""
    monkeypatch.setenv(STUDY_ROOT_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture()
def client(store):
    register_builtin_glossaries()
    return TestClient(app)


def _publish(root, bundle):
    save_evidence_bundle(root / ("%s.json" % bundle.study_id), bundle)
    return bundle


def _with_figures(bundle):
    """A bundle whose passing effect-size entry carries all six of R9's figures."""
    return bundle.append(
        "effect_sizes", label="lift against base rate", status="PASS",
        summary="the association, with the context that makes it readable",
        recorded_at="2026-04-01T00:00:00+00:00",
        payload=_figures().to_mapping(), source_sha256s=("a" * 64,))


# ------------------------------------------------------------ 1. the wire guard (R9)


def test_the_wire_guard_refuses_a_bare_confidence_at_any_depth():
    """Structural, not textual, so a payload shape nobody anticipated is still caught."""
    for payload in ({"confidence": 0.82},
                    {"result": {"confidence": 0.82}},
                    {"rows": [{"ok": True}, {"confidence": 0.82}]},
                    [[{"nested": {"confidence": 0.82}}]]):
        with pytest.raises(InvalidParameterError) as excinfo:
            refuse_bare_confidence(payload)
        assert "base_rate" in str(excinfo.value)


def test_the_wire_guard_passes_a_confidence_that_travels_with_all_six_figures():
    whole = _figures().to_mapping()
    assert set(R9_FIGURES) <= set(whole)
    assert refuse_bare_confidence({"figures": whole}) is not None


def test_the_wire_guard_restates_r9s_figures_rather_than_importing_them():
    """A guard that imported its expectations from the thing it guards agrees with any change.

    `R9_FIGURES` is deliberately a second statement of the same list. This asserts the two agree
    today, so a change to either is a failing test rather than a silent divergence.
    """
    assert set(R9_FIGURES) == set(_figures().to_mapping())


def test_no_route_can_serve_a_confidence_without_the_figures_that_give_it_meaning(client, store):
    """First acceptance criterion: R9 enforced on the wire, over every route.

    The study deliberately carries a full figure set, so the routes genuinely do serve a
    confidence - a test that passed because nothing anywhere reported one would be worthless.
    """
    _publish(store, _with_figures(_climbed("candidate_precursor")))
    served_a_confidence = False
    paths = ["/api/v1/findings/domains",
             "/api/v1/findings/glossaries/reanalysis",
             "/api/v1/findings/studies",
             "/api/v1/findings/studies/planted_precursor_v1",
             "/api/v1/findings/studies/planted_precursor_v1/outputs",
             "/api/v1/findings/studies/planted_precursor_v1/translation?glossary=reanalysis"]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.json()
        refuse_bare_confidence(body, where=path)
        if "confidence" in json.dumps(body):
            served_a_confidence = True
    assert served_a_confidence, "the corpus must actually contain a confidence to be a test"


# ------------------------------------------------------------ 2. the registry condition


def test_a_new_domain_reaches_the_api_without_editing_src_api(client):
    """Second acceptance criterion: TG8.1's condition, carried onto the HTTP layer."""
    state = snapshot(DOMAIN_GLOSSARIES)
    try:
        DOMAIN_GLOSSARIES.add("estuary", DomainGlossary(
            domain="estuary",
            phrases={term: "the %s reading in the estuary survey"
                     % term.split(".", 1)[1].replace("_", " ").replace(".", " of ")
                     for term in STRUCTURAL_VOCABULARY}),
            description="Registered from the test module: outside src/core and src/api both.")
        names = [row["name"] for row in client.get("/api/v1/findings/domains").json()]
        assert "estuary" in names
        served = client.get("/api/v1/findings/glossaries/estuary")
        assert served.status_code == 200
        assert served.json()["domain"] == "estuary"
    finally:
        restore(DOMAIN_GLOSSARIES, state)


def test_the_builtin_glossaries_are_registered_eagerly_not_on_first_request():
    """The D35 defence.

    D35 was a fallback chain that depended on browsing order: registration was an import side
    effect of a module imported lazily inside its own handler, so `GET /data/sources` returned
    different answers before and after a researcher visited a particular tab. Importing the
    router must be sufficient; no request may be needed to make a domain appear.
    """
    import importlib
    module = importlib.import_module("src.api.findings")
    assert set(module.REGISTERED_GLOSSARIES) == {"reanalysis", "order_book"}
    for name in module.REGISTERED_GLOSSARIES:
        assert name in DOMAIN_GLOSSARIES


def test_registering_the_builtins_twice_is_not_an_error():
    assert register_builtin_glossaries() == register_builtin_glossaries()


# ------------------------------------------------------------ 3. the store is read-only


def test_the_surface_never_writes_and_a_get_moves_no_claim(client, store):
    """A read of a claim cannot change it (R22), asserted over the bytes on disk."""
    bundle = _publish(store, _with_figures(_climbed("robust_association")))
    path = store / ("%s.json" % bundle.study_id)
    before = path.read_bytes()
    before_rung = summarise_evidence(bundle).rung
    for suffix in ("", "/outputs", "/translation?glossary=order_book"):
        assert client.get("/api/v1/findings/studies/%s%s"
                          % (bundle.study_id, suffix)).status_code == 200
    assert path.read_bytes() == before
    assert summarise_evidence(bundle).rung == before_rung


def test_an_unreadable_bundle_is_reported_rather_than_silently_skipped(client, store):
    """Omitting it would let a corrupted study look like one nobody ever ran."""
    _publish(store, _climbed("association"))
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    rows = client.get("/api/v1/findings/studies").json()
    unreadable = [row for row in rows if not row["readable"]]
    assert len(unreadable) == 1
    assert unreadable[0]["file"] == "broken.json"
    assert unreadable[0]["refused_because"]


def test_an_absent_study_root_is_an_empty_list_not_an_error(client, monkeypatch, tmp_path):
    monkeypatch.setenv(STUDY_ROOT_ENV, str(tmp_path / "nothing here"))
    assert client.get("/api/v1/findings/studies").json() == []


def test_an_unknown_study_and_an_unknown_glossary_are_both_404(client, store):
    _publish(store, _climbed("association"))
    assert client.get("/api/v1/findings/studies/absent").status_code == 404
    assert client.get("/api/v1/findings/glossaries/absent").status_code == 404
    assert client.get("/api/v1/findings/studies/planted_precursor_v1/translation"
                      "?glossary=absent").status_code == 404


# ------------------------------------------------------------ 4. what the surface serves


def test_the_study_list_reports_the_rung_and_whether_anything_caps_it(client, store):
    _publish(store, _climbed("candidate_precursor"))
    row = client.get("/api/v1/findings/studies").json()[0]
    assert row["rung"] == "candidate_precursor"
    assert row["blocked"] is False
    assert row["readable"] is True
    assert row["bundle_sha256"] and row["summary_sha256"]


def test_a_blocked_study_is_reported_as_blocked(client, store):
    _publish(store, _blocked())
    row = client.get("/api/v1/findings/studies").json()[0]
    assert row["blocked"] is True
    assert row["rung"] == "observation"


def test_a_translation_is_served_rendered_rather_than_as_parts(client, store):
    """Phase G9's principle on the wire: a client displays a string, it does not build one."""
    _publish(store, _with_figures(_climbed("candidate_precursor")))
    body = client.get("/api/v1/findings/studies/planted_precursor_v1/translation"
                      "?glossary=reanalysis").json()
    assert body["rendered_text"].strip()
    assert "What can be said:" in body["rendered_text"]
    assert body["structural_keys"]
    assert body["translation_sha256"]
    # The entitlement travels welded to the claim, so a client cannot drop it.
    assert any("may not be described as" in unit["licences"] or unit["licences"]
               for unit in body["claimable"])


def test_the_same_study_in_two_vocabularies_states_identical_facts(client, store):
    """TG7.4's acceptance, now visible through HTTP - which is what the UI will show."""
    _publish(store, _with_figures(_climbed("candidate_precursor")))
    base = "/api/v1/findings/studies/planted_precursor_v1/translation?glossary=%s"
    first = client.get(base % "reanalysis").json()
    second = client.get(base % "order_book").json()
    assert first["structural_keys"] == second["structural_keys"]
    assert first["rendered_text"] != second["rendered_text"]
    assert first["summary_sha256"] == second["summary_sha256"]
    # No wording is shared between the two vocabularies, which is what makes it a real test.
    assert "reanalysis" in first["rendered_text"]
    assert "order book" in second["rendered_text"]


def test_the_outputs_route_serves_the_untranslated_claim_state_for_checking(client, store):
    """A reader must be able to see both forms to check the wording changed no fact."""
    bundle = _publish(store, _climbed("robust_association"))
    body = client.get("/api/v1/findings/studies/%s/outputs" % bundle.study_id).json()
    assert body["assessment"]["rung"] == "robust_association"
    assert body["summary_sha256"] == summarise_evidence(bundle).summary_sha256


def test_a_contradicted_study_serves_the_evidence_against_it(client, store):
    bundle = _publish(store, _contradicted())
    body = client.get("/api/v1/findings/studies/%s/translation?glossary=order_book"
                      % bundle.study_id).json()
    assert body["contradicting"]
    assert "What argues against it:" in body["rendered_text"]


def test_a_partial_figure_set_is_served_as_no_figures_rather_than_as_a_subset(client, store):
    """Four of six figures is not four-sixths of a finding, so it is not rendered at all."""
    bundle = _climbed("association").append(
        "effect_sizes", label="an incomplete record", status="PASS",
        summary="a confidence with no base rate beside it",
        recorded_at="2026-04-01T00:00:00+00:00",
        payload={"confidence": 0.82, "support": 40}, source_sha256s=("b" * 64,))
    _publish(store, bundle)
    body = client.get("/api/v1/findings/studies/%s/translation?glossary=reanalysis"
                      % bundle.study_id).json()
    assert body["figures"] is None
    assert "%" not in body["rendered_text"]


def test_the_domains_route_is_generated_from_the_registry_not_hand_maintained(client):
    rows = client.get("/api/v1/findings/domains").json()
    names = {row["name"] for row in rows}
    assert {"reanalysis", "order_book"} <= names
    for row in rows:
        assert row["term_count"] == len(STRUCTURAL_VOCABULARY)
        assert row["glossary_sha256"]


def test_the_store_reads_the_directory_it_is_pointed_at(store):
    bundle = _publish(store, _climbed("association"))
    loaded, path = StudyStore(store).load(bundle.study_id)
    assert loaded.bundle_sha256 == bundle.bundle_sha256
    assert path.name == "%s.json" % bundle.study_id
    with pytest.raises(KeyError):
        StudyStore(store).load("absent")


# ------------------------------------------------------------ 5. the refusal surface (TG9.3)


def test_a_domain_serves_what_it_refuses_beside_how_it_speaks(client):
    """The half of TG9.1 that was declared and not built, delivered here.

    `/domains` listed glossaries - wording - and nothing about limits. A reader could be told a
    finding in fluent domain words with no way to learn that the domain forbids the reading.
    """
    rows = {row["name"]: row for row in client.get("/api/v1/findings/domains").json()}
    order_book = rows["order_book"]["declaration"]
    assert order_book is not None
    assert order_book["precedence_admissible"] is False
    bases = {item["basis"] for item in order_book["refuses"]}
    assert "lag_policy:none" in bases
    assert "violation:no_propagation_speed" in bases
    for item in order_book["refuses"]:
        assert item["consequence"].strip(), "a refusal with no stated reason is folklore"


def test_a_domain_that_breaks_nothing_still_says_so_rather_than_showing_a_blank(client):
    rows = {row["name"]: row for row in client.get("/api/v1/findings/domains").json()}
    reanalysis = rows["reanalysis"]["declaration"]
    assert reanalysis["precedence_admissible"] is True
    assert reanalysis["refuses"] == []
    assert reanalysis["declared"]["violations"] == {}


def test_the_refusal_reason_is_the_one_the_analysis_layer_enforces(client):
    """Drawn from `KNOWN_VIOLATIONS`, not restated, so displayed and enforced cannot drift."""
    from src.core.domain import KNOWN_VIOLATIONS

    rows = {row["name"]: row for row in client.get("/api/v1/findings/domains").json()}
    for item in rows["order_book"]["declaration"]["refuses"]:
        if item["basis"].startswith("violation:"):
            name = item["basis"].split(":", 1)[1]
            assert item["consequence"] == KNOWN_VIOLATIONS[name]


def test_every_served_domain_limit_carries_the_attribution_caveat(client, store):
    """A bundle does not record its domain, so presenting limits as a check would fabricate one.

    This is the constraint that shapes the whole slice: nothing here verifies that a study came
    from the domain whose words it is being read in, and the caveat must travel with the claim
    rather than sitting in documentation nobody opens.
    """
    _publish(store, _climbed("candidate_precursor"))
    rows = client.get("/api/v1/findings/domains").json()
    for row in rows:
        if row["declaration"]:
            assert "does not record which domain produced it" in \
                row["declaration"]["attribution_caveat"]
    body = client.get("/api/v1/findings/studies/planted_precursor_v1/translation"
                      "?glossary=order_book").json()
    assert "does not record which domain produced it" in \
        body["domain_limits"]["attribution_caveat"]
    assert "does not record which domain produced it" in \
        body["unadmitted_reading"]["attribution_caveat"]


def test_a_precedence_rung_read_in_a_domain_that_forbids_one_is_reported(client, store):
    """The tension worth surfacing: a rung asserting order, in words that cannot carry it."""
    _publish(store, _climbed("candidate_precursor"))
    base = "/api/v1/findings/studies/planted_precursor_v1/translation?glossary=%s"
    forbidden = client.get(base % "order_book").json()["unadmitted_reading"]
    assert forbidden is not None
    assert forbidden["lag_policy"] == "none"
    assert "R21" in forbidden["note"]
    assert client.get(base % "reanalysis").json()["unadmitted_reading"] is None


def test_a_rung_below_precedence_reports_no_tension_in_either_domain(client, store):
    """Below the rung that asserts ordering, a domain refusing precedence refuses nothing said."""
    bundle = _publish(store, _climbed("robust_association"))
    for name in ("reanalysis", "order_book"):
        body = client.get("/api/v1/findings/studies/%s/translation?glossary=%s"
                          % (bundle.study_id, name)).json()
        assert body["unadmitted_reading"] is None


def test_reporting_an_unadmitted_reading_moves_no_claim(client, store):
    """R22 holds through the refusal surface: it reports a tension, it does not resolve one."""
    bundle = _publish(store, _climbed("candidate_precursor"))
    before = summarise_evidence(bundle).summary_sha256
    body = client.get("/api/v1/findings/studies/%s/translation?glossary=order_book"
                      % bundle.study_id).json()
    assert body["unadmitted_reading"] is not None
    assert body["summary_sha256"] == before
    assert summarise_evidence(bundle).rung == "candidate_precursor"


def test_a_vocabulary_without_a_declaration_reports_the_absence(client):
    """Unknown limits must not render as a domain that happens to forbid nothing."""
    state = snapshot(DOMAIN_GLOSSARIES)
    try:
        DOMAIN_GLOSSARIES.add("wordsonly", DomainGlossary(
            domain="wordsonly",
            phrases={term: "the %s note" % term.split(".", 1)[1].replace("_", " ")
                     for term in STRUCTURAL_VOCABULARY}))
        rows = {row["name"]: row for row in client.get("/api/v1/findings/domains").json()}
        assert rows["wordsonly"]["declaration"] is None
    finally:
        restore(DOMAIN_GLOSSARIES, state)


def test_the_builtin_domains_are_registered_eagerly(client):
    import importlib
    module = importlib.import_module("src.api.findings")
    assert set(module.REGISTERED_DOMAINS) == {"reanalysis", "order_book"}
