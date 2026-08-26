"""TG8.1: the onboarding contract, and its acceptance criterion (`ed-dev`).

The criterion the roadmap sets is *a domain is onboarded without editing `src/`*, matching the
existing third-party transform plugin test. `test_registries.py` proves that for sources and
actions by hashing the three files a plugin would otherwise have had to touch and checking they
are byte-identical after the plugin is live. The same method is used here, against the files a
domain would otherwise have had to touch.

The rest of the module tests the thing the contract adds over two `Registry.add` calls:
atomicity, and the geometry/violation biconditional that no single object can check because the
two facts live in different registries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.onboarding import (DOMAIN_ONBOARDINGS, ONBOARDING_SCHEMA, OnboardingRefused,
                                 REQUIRED_DECLARATIONS, REQUIRED_NAMES, assert_geometry_agrees,
                                 audit_onboarding, geometry_offers_metric, is_onboarded,
                                 onboard_domain, onboarded_names, onboarding_for)
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The files a domain would have to be added to if the registries were not a seam. TG8.1's
#: acceptance is that importing `domain_plugin_example` changes none of them.
UNTOUCHED_FILES = (
    "src/core/domain.py",
    "src/core/onboarding.py",
    "src/core/translation.py",
    "src/core/builtin_domains.py",
    "src/core/builtin_glossaries.py",
    "src/api/findings.py",
)


def _hashes():
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in UNTOUCHED_FILES}


@pytest.fixture()
def registries():
    """Snapshot all three registries, so a test may register freely and leave no trace."""
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    try:
        yield
    finally:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


def _phrases(source: str = "reanalysis"):
    """A complete, screen-passing phrase map, borrowed from a built-in.

    Borrowed rather than written afresh: these tests are about the contract, and a bespoke
    thirty-eight-term glossary in each of them would test TG7.4's screens over and over while
    making the contract harder to see.
    """
    from src.core.builtin_glossaries import ORDER_BOOK_PHRASES, REANALYSIS_PHRASES

    return dict(REANALYSIS_PHRASES if source == "reanalysis" else ORDER_BOOK_PHRASES)


def _declaration(name: str, *, violations=(), lag_policy="advective", **kwargs):
    return DomainDeclaration(
        name=name,
        description="A declaration written by the onboarding tests.",
        axes=(AxisSpec(name="time", role="time", units="s"),
              AxisSpec(name="row", role="space", units="m", ordinal=0),
              AxisSpec(name="column", role="space", units="m", ordinal=1)),
        licence="Test licence; not a real archive.",
        violations=violations,
        lag_policy=lag_policy,
        provenance={"declared_by": __name__},
        **kwargs)


# ------------------------------------------------------------------ the recipe


def test_the_recipe_names_the_seven_declarations_the_roadmap_lists():
    assert REQUIRED_NAMES == ("axes", "geometry", "lag_policy", "violations", "licence",
                              "provenance", "glossary")
    assert len(set(REQUIRED_NAMES)) == len(REQUIRED_NAMES)
    for _name, why in REQUIRED_DECLARATIONS:
        assert why.strip(), "every requirement states why it is required"


def test_a_checklist_is_generated_from_the_recipe_rather_than_written_out(registries):
    record = onboard_domain(_declaration("checklist_demo"), _phrases(), geometry="latlon")
    rendered = record.checklist()
    assert [item["requirement"] for item in rendered] == list(REQUIRED_NAMES)
    by_name = {item["requirement"]: item["declared"] for item in rendered}
    assert by_name["geometry"] == "latlon"
    assert by_name["lag_policy"] == "advective"
    assert by_name["glossary"] == record.glossary.glossary_sha256


# ------------------------------------------------------------------ the acceptance criterion


def test_a_domain_is_onboarded_without_editing_src():
    """TG8.1's acceptance, in the form `test_registries.py` uses for sources and actions."""
    before = _hashes()
    import src.tests.domain_plugin_example as plugin       # noqa: F401  (import is the test)
    after = _hashes()

    assert before == after, "importing a domain plugin edited a file inside src/"
    assert is_onboarded(plugin.DOMAIN_NAME)
    assert plugin.DOMAIN_NAME in DOMAIN_GLOSSARIES
    assert plugin.DOMAIN_NAME in DOMAIN_DECLARATIONS


def test_the_plugin_domain_is_attributed_to_the_file_that_onboarded_it():
    """`Entry.defined_in` would say `src.core.translation`, which is the class's home."""
    import src.tests.domain_plugin_example as plugin

    record = onboarding_for(plugin.DOMAIN_NAME)
    assert record.onboarded_by == "src.tests.domain_plugin_example"
    assert not record.onboarded_by.startswith("src.core")


def test_the_plugin_domain_breaks_assumptions_no_registered_domain_had_broken():
    """R17's reasoning, asserted: a domain that breaks nothing new proves nothing new."""
    import src.tests.domain_plugin_example as plugin

    from src.core.builtin_domains import BUILTIN_DECLARATIONS

    already = set()
    for declaration in BUILTIN_DECLARATIONS:
        already.update(declaration.violations)
    new = set(plugin.ARGO.violations) - already
    assert new == {"irregular_sampling", "non_stationary_support"}


def test_the_plugin_domain_is_the_first_with_an_irregular_clock_and_admissible_precedence():
    import src.tests.domain_plugin_example as plugin

    assert plugin.ARGO.precedence_admissible
    assert "irregular_sampling" in plugin.ARGO.violations
    assert plugin.ARGO.minimum_admissible_lag() == 1
    assert plugin.ARGO.declared_floor_basis.strip()


def test_the_plugin_domain_is_served_by_the_findings_api():
    """Without which the dropdown is a list a researcher cannot reach."""
    from fastapi.testclient import TestClient

    import src.tests.domain_plugin_example as plugin
    from src.api.main import app

    rows = TestClient(app).get("/api/v1/findings/domains").json()
    row = next(r for r in rows if r["name"] == plugin.DOMAIN_NAME)
    assert row["onboarding"]["complete"] is True
    assert row["declaration"]["precedence_admissible"] is True
    assert row["term_count"] == 38


# ------------------------------------------------------------------ the biconditional


def test_a_geometry_with_a_metric_may_not_accompany_no_physical_metric(registries):
    declaration = _declaration("contradiction_a", violations=("no_physical_metric",),
                               lag_policy="none")
    with pytest.raises(OnboardingRefused) as excinfo:
        onboard_domain(declaration, _phrases(), geometry="cartesian")
    assert "physical_metric=True" in str(excinfo.value)


def test_supplying_no_geometry_requires_renouncing_the_metric(registries):
    declaration = _declaration("contradiction_b", violations=("irregular_sampling",),
                               lag_policy="declared", declared_floor_frames=2,
                               declared_floor_basis="a reporting interval, for the test")
    with pytest.raises(OnboardingRefused) as excinfo:
        onboard_domain(declaration, _phrases(), geometry=None)
    assert "no_physical_metric" in str(excinfo.value)


def test_both_consistent_pairings_are_accepted(registries):
    metric = onboard_domain(_declaration("consistent_metric"), _phrases(), geometry="latlon")
    assert metric.geometry == "latlon"

    renounced = onboard_domain(
        _declaration("consistent_none", violations=("no_physical_metric",), lag_policy="none"),
        _phrases("order_book"), geometry=None)
    assert renounced.geometry is None


def test_a_geometry_without_a_metric_counts_as_renouncing_one(registries):
    """`pixel` declares `physical_metric: False`, so it sits on the same side as `None`."""
    assert geometry_offers_metric("pixel") is False
    record = onboard_domain(
        _declaration("pixel_domain", violations=("no_physical_metric",), lag_policy="none"),
        _phrases(), geometry="pixel")
    assert record.geometry == "pixel"


def test_an_unregistered_geometry_is_refused_by_the_registry_that_owns_the_vocabulary(
        registries):
    with pytest.raises(UnknownNameError):
        onboard_domain(_declaration("unknown_geometry"), _phrases(), geometry="tripolar")


def test_the_biconditional_is_callable_on_its_own():
    """It is public because a probe (deferred) will want to ask before building anything."""
    assert_geometry_agrees(_declaration("standalone"), "latlon")
    with pytest.raises(OnboardingRefused):
        assert_geometry_agrees(_declaration("standalone"), None)


# ------------------------------------------------------------------ atomicity


def test_a_refused_glossary_leaves_every_registry_untouched(registries):
    before = (DOMAIN_GLOSSARIES.names(), DOMAIN_DECLARATIONS.names(),
              DOMAIN_ONBOARDINGS.names())
    broken = _phrases()
    broken["rung.association"] = "a link seen in 8 of the frames examined"   # digit screen

    with pytest.raises(InvalidParameterError):
        onboard_domain(_declaration("partial_a"), broken, geometry="latlon")

    assert (DOMAIN_GLOSSARIES.names(), DOMAIN_DECLARATIONS.names(),
            DOMAIN_ONBOARDINGS.names()) == before
    assert "partial_a" not in DOMAIN_GLOSSARIES


def test_a_refused_geometry_leaves_every_registry_untouched(registries):
    before = (DOMAIN_GLOSSARIES.names(), DOMAIN_DECLARATIONS.names(),
              DOMAIN_ONBOARDINGS.names())
    with pytest.raises(OnboardingRefused):
        onboard_domain(_declaration("partial_b"), _phrases(), geometry=None)
    assert (DOMAIN_GLOSSARIES.names(), DOMAIN_DECLARATIONS.names(),
            DOMAIN_ONBOARDINGS.names()) == before


def test_a_failure_during_the_writes_rolls_back_the_registries_already_written(
        registries, monkeypatch):
    """The rollback path itself, exercised where the ordinary refusals never reach it.

    Every screen runs before the first `Registry.add`, so a refusal cannot leave a partial
    write. This forces the case the ordering is designed against anyway — a failure *between*
    the writes — because a rollback that is never executed is a rollback nobody has checked.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(DOMAIN_ONBOARDINGS, "add", explode)

    with pytest.raises(RuntimeError):
        onboard_domain(_declaration("rolled_back"), _phrases(), geometry="latlon")

    assert "rolled_back" not in DOMAIN_GLOSSARIES
    assert "rolled_back" not in DOMAIN_DECLARATIONS
    assert "rolled_back" not in DOMAIN_ONBOARDINGS


def test_re_onboarding_an_existing_name_is_refused_and_names_what_holds_it(registries):
    onboard_domain(_declaration("taken"), _phrases(), geometry="latlon")
    with pytest.raises(OnboardingRefused) as excinfo:
        onboard_domain(_declaration("taken"), _phrases(), geometry="latlon")
    message = str(excinfo.value)
    assert "glossary" in message and "declaration" in message and "onboarding" in message
    assert "replace=True" in message


def test_replace_is_available_when_it_is_asked_for_explicitly(registries):
    first = onboard_domain(_declaration("replaceable"), _phrases(), geometry="latlon")
    second = onboard_domain(_declaration("replaceable"), _phrases("order_book"),
                            geometry="latlon", replace=True)
    assert first.onboarding_sha256 != second.onboarding_sha256
    assert DOMAIN_GLOSSARIES.get("replaceable").glossary_sha256 == \
        second.glossary.glossary_sha256


def test_a_name_the_glossary_would_normalise_differently_is_refused(registries):
    """Two registries keyed differently is the half-onboarded state in a subtler form."""
    with pytest.raises(OnboardingRefused) as excinfo:
        onboard_domain(_declaration(" padded "), _phrases(), geometry="latlon")
    assert "normalis" in str(excinfo.value)


def test_a_mapping_is_not_accepted_in_place_of_a_declaration(registries):
    with pytest.raises(InvalidParameterError):
        onboard_domain({"name": "not_a_declaration"}, _phrases(), geometry="latlon")


# ------------------------------------------------------------------ the audit


def test_a_piecemeal_domain_is_reported_incomplete_rather_than_passing_for_onboarded(
        registries):
    from src.core.translation import DomainGlossary

    DOMAIN_GLOSSARIES.add("piecemeal", DomainGlossary(domain="piecemeal",
                                                      phrases=_phrases()))
    report = audit_onboarding("piecemeal")
    assert report["complete"] is False
    assert report["missing"] == ["declaration", "onboarding"]
    assert "piecemeal" in DOMAIN_GLOSSARIES
    assert "never checked against each other" in report["note"]


def test_an_unknown_domain_is_reported_as_missing_everything():
    report = audit_onboarding("no_such_domain")
    assert report["complete"] is False
    assert report["missing"] == ["declaration", "glossary", "onboarding"]
    assert "note" not in report


def test_a_complete_domain_reports_its_digest_and_its_author(registries):
    onboard_domain(_declaration("audited"), _phrases(), geometry="latlon",
                   onboarded_by="a.test.module")
    report = audit_onboarding("audited")
    assert report["complete"] is True
    assert report["onboarded_by"] == "a.test.module"
    assert len(report["onboarding_sha256"]) == 64


# ------------------------------------------------------------------ the digest


def test_the_digest_is_stable_across_equal_declarations(registries):
    first = onboard_domain(_declaration("digest_a"), _phrases(), geometry="latlon",
                           onboarded_by="fixed")
    restore(DOMAIN_GLOSSARIES, {n: DOMAIN_GLOSSARIES.entry(n)
                                for n in DOMAIN_GLOSSARIES.names() if n != "digest_a"})
    restore(DOMAIN_DECLARATIONS, {n: DOMAIN_DECLARATIONS.entry(n)
                                  for n in DOMAIN_DECLARATIONS.names() if n != "digest_a"})
    restore(DOMAIN_ONBOARDINGS, {n: DOMAIN_ONBOARDINGS.entry(n)
                                 for n in DOMAIN_ONBOARDINGS.names() if n != "digest_a"})
    again = onboard_domain(_declaration("digest_a"), _phrases(), geometry="latlon",
                           onboarded_by="fixed")
    assert first.onboarding_sha256 == again.onboarding_sha256


def test_changing_one_phrase_changes_the_digest(registries):
    base = onboard_domain(_declaration("digest_b"), _phrases(), geometry="latlon",
                          onboarded_by="fixed")
    altered = dict(_phrases())
    altered["status.PASS"] = "held, against the record"
    changed = onboard_domain(_declaration("digest_b"), altered, geometry="latlon",
                             onboarded_by="fixed", replace=True)
    assert base.onboarding_sha256 != changed.onboarding_sha256


def test_the_describe_record_names_its_schema(registries):
    record = onboard_domain(_declaration("schema_demo"), _phrases(), geometry="latlon")
    described = record.describe()
    assert described["schema"] == ONBOARDING_SCHEMA
    assert described["geometry_offers_physical_metric"] is True
    assert described["glossary_term_count"] == 38


# ------------------------------------------------------------------ the built-ins


def test_the_builtins_are_held_to_the_contract_they_document():
    from src.core.builtin_domains import register_builtin_domains

    register_builtin_domains()
    for name in ("reanalysis", "order_book"):
        report = audit_onboarding(name)
        assert report["complete"] is True, name
        assert report["onboarded_by"] == "src.core.builtin_domains"


def test_either_builtin_entry_point_registers_the_whole_pair():
    """A way in that quietly produced wording-without-limits would reopen TG9.1's hole."""
    from src.core.builtin_domains import register_builtin_domains
    from src.core.builtin_glossaries import register_builtin_glossaries

    assert register_builtin_glossaries() == register_builtin_domains()
    for name in register_builtin_glossaries():
        assert name in DOMAIN_GLOSSARIES and name in DOMAIN_DECLARATIONS
        assert name in DOMAIN_ONBOARDINGS


def test_a_builtin_found_half_registered_is_repaired_rather_than_skipped(registries):
    from src.core.builtin_domains import register_builtin_domains

    register_builtin_domains()
    DOMAIN_GLOSSARIES.unregister("order_book")
    assert "order_book" not in DOMAIN_GLOSSARIES

    register_builtin_domains()
    assert "order_book" in DOMAIN_GLOSSARIES
    assert audit_onboarding("order_book")["complete"] is True


def test_the_builtin_geometries_are_the_two_sides_of_the_biconditional():
    from src.core.builtin_domains import register_builtin_domains

    register_builtin_domains()
    assert onboarding_for("reanalysis").geometry == "latlon"
    assert onboarding_for("order_book").geometry is None
    assert "no_physical_metric" in onboarding_for("order_book").declaration.violations
    assert "no_physical_metric" not in onboarding_for("reanalysis").declaration.violations


# ------------------------------------------------------------------ the served contract


def test_the_contract_route_serves_the_recipe_it_enforces():
    from fastapi.testclient import TestClient

    from src.api.main import app

    payload = TestClient(app).get("/api/v1/findings/onboarding").json()
    assert payload["schema"] == ONBOARDING_SCHEMA
    assert [item["requirement"] for item in payload["required"]] == list(REQUIRED_NAMES)
    assert {row["name"] for row in payload["onboarded"]} >= {"reanalysis", "order_book"}
    assert all(row["complete"] for row in payload["onboarded"])


def test_a_declaration_without_wording_still_appears_in_the_domain_listing(registries):
    """The TG9.1 omission with its halves swapped: limits registered, and invisible."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    DOMAIN_DECLARATIONS.add("silent_domain", _declaration("silent_domain"),
                            description="Limits with no wording behind them.")
    rows = TestClient(app).get("/api/v1/findings/domains").json()
    row = next(r for r in rows if r["name"] == "silent_domain")
    assert row["glossary_sha256"] is None
    assert row["term_count"] == 0
    assert row["declaration"] is not None
    assert row["onboarding"]["complete"] is False


def test_onboarded_names_is_sorted_and_matches_the_registry():
    names = onboarded_names()
    assert list(names) == sorted(names)
    assert set(names) == set(DOMAIN_ONBOARDINGS.names())
