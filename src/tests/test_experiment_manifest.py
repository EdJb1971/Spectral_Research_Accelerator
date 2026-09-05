"""TG17.1: one immutable scientific configuration from recipe through preflight."""

import json

import pytest
from pydantic import ValidationError

from src.core.experiment_manifest import (CrossDomainExperimentSpec, ManifestStore,
                                          canonical_bytes, flagship_recipe,
                                          manifest_sha256, preflight_manifest)


def _bound_flagship():
    body = json.loads(canonical_bytes(flagship_recipe()))
    order_book = next(row for row in body["observations"] if row["domain"] == "order_book")
    order_book["acquisition"]["identity"]["content_sha256"] = "a" * 64
    return CrossDomainExperimentSpec.parse_obj(body)


def test_flagship_recipe_is_the_frozen_quartet_with_explicit_boundaries():
    spec = flagship_recipe()
    assert [row.domain for row in spec.observations] == [
        "reanalysis", "argo_float", "tess_lightcurve", "order_book"]
    assert [row.name for row in spec.windows] == ["week", "three_months", "six_months"]
    assert spec.windows[1].start_utc.isoformat() == "2026-01-01T00:00:00+00:00"
    assert spec.windows[1].end_utc.isoformat() == "2026-04-01T00:00:00+00:00"


def test_canonical_round_trip_is_byte_stable_and_is_the_run_identity():
    first = flagship_recipe()
    second = CrossDomainExperimentSpec.parse_raw(canonical_bytes(first))
    assert canonical_bytes(second) == canonical_bytes(first)
    assert manifest_sha256(second) == manifest_sha256(first)


def test_multiple_durations_are_one_declared_family():
    """Three durations are three times the tests, and TG17.5 shows the multiplication."""
    report = preflight_manifest(flagship_recipe())
    family = report["family"]
    assert family["declared_members"] == 288
    assert family["maximum_members"] == 10000
    assert family["windows_are_one_family"] is True
    assert family["in_human_terms"] == ("6 domain sets x 3 windows x 4 channels x 4 scales "
                                        "x 1 relationship = 288 declared tests.")
    window_cost = next(row for row in family["expansion_cost"] if row["axis"] == "window")
    assert window_cost["members_added"] == 96


def test_metadata_preflight_never_uses_network_or_opens_values():
    report = preflight_manifest(flagship_recipe())
    assert report["metadata_only"] is True
    assert report["network_used"] is False
    assert report["measurement_values_opened"] is False
    assert all(row.get("opens_measurement_values") is False for row in report["coverage"])


def test_native_support_is_not_laundered_into_exact_common_coverage():
    report = preflight_manifest(flagship_recipe())
    rows = {row["domain"]: row for row in report["coverage"]}
    assert rows["reanalysis"]["support_kind"] == "regular_grid_extent"
    assert rows["argo_float"]["support_kind"] == "sparse_point_support"
    assert rows["tess_lightcurve"]["support_kind"] == "intersecting_observational_sectors"
    assert all(not window["coverage_exact"] for window in rows["tess_lightcurve"]["windows"])


def test_missing_local_record_is_a_stable_visible_refusal():
    """TG17.3 generalised this refusal out of being order-book-specific.

    The wording changed deliberately: order book is now one declaration in the bespoke record
    family, so a refusal naming "order-book record" would have been the family's one example
    written into the framework. What stays stable is the refusal's *identity* — same domain,
    same remedy, reachable before any acquisition — not a sentence about markets.
    """
    report = preflight_manifest(flagship_recipe())
    assert report["status"] == "REFUSED"
    assert [row["domain"] for row in report["refusals"]] == ["order_book"]
    reason = report["refusals"][0]["reason"]
    assert reason.startswith("select a content-addressed local record")
    assert "order-book" not in reason and "order book" not in reason


def test_bound_record_permits_partial_coverage_only_under_the_frozen_policy():
    report = preflight_manifest(_bound_flagship())
    assert report["status"] == "PARTIAL"
    body = json.loads(canonical_bytes(_bound_flagship()))
    body["coverage_policy"] = {"requirement": "complete_required", "minimum_fraction": 1.0}
    refused = preflight_manifest(CrossDomainExperimentSpec.parse_obj(body))
    assert refused["status"] == "REFUSED"
    assert "complete coverage is required" in refused["refusals"][0]["reason"]


def test_scale_shape_mode_cannot_omit_its_normalization():
    body = json.loads(canonical_bytes(flagship_recipe()))
    body["mode"] = "scale_shape_aligned"
    with pytest.raises(ValidationError, match="scale_normalization"):
        CrossDomainExperimentSpec.parse_obj(body)


def test_duplicate_domains_and_over_cap_family_are_refused_before_acquisition():
    body = json.loads(canonical_bytes(flagship_recipe()))
    body["observations"][1]["domain"] = body["observations"][0]["domain"]
    with pytest.raises(ValidationError, match="each domain only once"):
        CrossDomainExperimentSpec.parse_obj(body)
    body = json.loads(canonical_bytes(flagship_recipe()))
    body["family"]["maximum_members"] = 10001
    with pytest.raises(ValidationError, match="family maximum exceeds"):
        CrossDomainExperimentSpec.parse_obj(body)


def test_saved_draft_moves_its_pointer_without_rewriting_old_manifest(tmp_path):
    store = ManifestStore(tmp_path)
    first = flagship_recipe()
    saved_first = store.save("flagship", first)
    body = json.loads(canonical_bytes(first))
    body["title"] = "A new immutable revision"
    second = CrossDomainExperimentSpec.parse_obj(body)
    saved_second = store.save("flagship", second)
    assert saved_first["manifest_sha256"] != saved_second["manifest_sha256"]
    assert store.load_manifest(saved_first["manifest_sha256"]).title == first.title
    assert store.load_draft("flagship").title == second.title


def test_composer_api_recipe_save_reload_and_preflight_share_one_identity(client, tmp_path,
                                                                           monkeypatch):
    from src.api.main import app
    monkeypatch.setattr(app.state, "experiment_manifest_dir", tmp_path, raising=False)
    recipe = client.get("/api/v1/experiment-composer/recipes/g17-flagship-calendar")
    assert recipe.status_code == 200
    envelope = recipe.json()
    manifest = envelope["canonical_manifest"]
    validated = client.post("/api/v1/experiment-composer/manifests/validate", json=manifest)
    assert validated.status_code == 200
    assert validated.json()["manifest_sha256"] == envelope["manifest_sha256"]
    saved = client.put("/api/v1/experiment-composer/drafts/api-flagship", json=manifest)
    assert saved.status_code == 200
    loaded = client.get("/api/v1/experiment-composer/drafts/api-flagship")
    assert loaded.json()["canonical_manifest"] == manifest
    preflight = client.post("/api/v1/experiment-composer/manifests/preflight", json=manifest)
    assert preflight.status_code == 200
    assert preflight.json()["manifest_sha256"] == envelope["manifest_sha256"]
    assert preflight.json()["status"] == "REFUSED"


def test_the_composer_serves_no_run_route_and_is_honest_about_what_is_not_yet_real(client):
    """Defect **D80**: this asserted `"run experiment" in not_yet_available`.

    That was true when TG17.1 wrote it and false from TG17.6, which shipped the orchestrator and
    put an *Open or resume the run* button in this very view. The assertion did not fail - it
    **held the stale claim in place**, so for a whole slice the composer contract told every
    client that running an experiment was unavailable while the run contract on the next router
    described the state machine that ran it. Two served documents disagreeing about what the
    system can do is the failure this field exists to prevent, and pinning the wrong one is worse
    than not checking at all.

    What is genuinely still true is the routing boundary: composing and running are different
    surfaces, and the composer router serves no run route. That is asserted below, and the
    capability claim is now asserted against the run contract rather than against a literal.
    """
    contract = client.get("/api/v1/experiment-composer").json()
    assert contract["claim_boundary"].startswith("A ready preflight is not")
    assert "run experiment" not in contract["not_yet_available"], (
        "runs have existed since TG17.6; a contract still calling them unavailable is lying")

    # The two contracts must agree about what has not shipped, rather than each keeping a list.
    run_contract = client.get("/api/v1/experiment-runs").json()
    assert set(contract["not_yet_available"]) & set(run_contract["not_yet_available"]), (
        "the composer and the runner must name the same missing capability, or a client can be "
        "told two different things about the same system")

    routes = {(route.path, tuple(sorted(route.methods or [])))
              for route in client.app.routes if route.path.startswith("/api/v1/experiment-composer")}
    assert all("run" not in path for path, _ in routes), (
        "composing and running are different surfaces; the run routes live on their own router")
