"""TG17.8: the comparison views, and the pictures they refuse to draw."""

import json

import pytest

from src.core.comparison_views import (AXIS_KINDS, Axis, COMPARISON_VIEWS, COVERAGE_CELLS,
                                       ENCODINGS, Encoding, ForbiddenReadingError,
                                       MagnitudeEquivalenceError, Mark, NOT_YET_MEASURED,
                                       NEVER_ADMISSIBLE_READINGS, REQUIRES_EXTERNAL_DESIGN,
                                       SHARED_AXIS_KINDS, ViewContext,
                                       assert_reading_admissible, describe_views, encoding_for,
                                       linked_selection, ordered_encodings, ordered_views,
                                       register_encoding, render_all, render_view)
from src.core.errors import UnknownNameError
from src.core.experiment_manifest import (CrossDomainExperimentSpec, canonical_bytes,
                                          flagship_recipe, preflight_manifest)


@pytest.fixture(scope="module")
def spec():
    return flagship_recipe()


@pytest.fixture(scope="module")
def context(spec):
    return ViewContext(spec=spec, preflight=preflight_manifest(spec))


def _respec(**changes):
    body = json.loads(canonical_bytes(flagship_recipe()))
    body.update(changes)
    return CrossDomainExperimentSpec.parse_obj(body)


def _context_for(spec):
    return ViewContext(spec=spec, preflight=preflight_manifest(spec))


# --------------------------------------------------------------------------------- the axes


def test_a_native_magnitude_axis_carrying_two_domains_refuses_to_be_constructed():
    """The single most important line in this slice.

    Not "is discouraged", not "is warned about": the object cannot exist. There is no rendering
    call to police, because a view holding this axis never finishes being built.
    """
    with pytest.raises(MagnitudeEquivalenceError) as error:
        Axis("temperature and depth", "native_magnitude",
             domains=("reanalysis", "order_book"), units="K")
    message = str(error.value)
    assert "reanalysis and order_book" in message
    assert "no common ruler" in message
    assert "difference in units as a difference in size" in message


def test_a_native_magnitude_axis_for_one_domain_is_fine():
    axis = Axis("sst", "native_magnitude", domains=("reanalysis",), units="K")
    assert axis.describe()["shared"] is False


@pytest.mark.parametrize("kind", [kind for kind in AXIS_KINDS if kind in SHARED_AXIS_KINDS])
def test_every_shareable_axis_kind_may_carry_several_domains(kind):
    assert Axis("shared", kind, domains=("a", "b", "c")).describe()["shared"] is True


def test_native_magnitude_is_the_only_unshareable_kind():
    """Stated as an assertion rather than left implicit in two lists that could drift apart."""
    assert set(AXIS_KINDS) - set(SHARED_AXIS_KINDS) == {"native_magnitude"}


def test_an_axis_belonging_to_no_domain_is_refused():
    with pytest.raises(Exception, match="at least one domain"):
        Axis("nothing", "categorical", domains=())


def test_an_unknown_axis_kind_names_the_registered_ones():
    with pytest.raises(UnknownNameError) as error:
        Axis("mystery", "log_scale", domains=("a",))
    assert "native_magnitude" in str(error.value)


# ------------------------------------------------------------------------------ the readings


@pytest.mark.parametrize("reading", NEVER_ADMISSIBLE_READINGS)
@pytest.mark.parametrize("mode", ["calendar_aligned", "scale_shape_aligned"])
def test_equivalence_readings_are_refused_in_every_mode(mode, reading):
    with pytest.raises(ForbiddenReadingError) as error:
        assert_reading_admissible(mode, reading)
    assert "no mode to be admissible in" in str(error.value)


@pytest.mark.parametrize("mode", ["calendar_aligned", "scale_shape_aligned"])
def test_causality_is_declarable_by_a_manifest_and_drawable_by_no_view(mode):
    """The split TG17.8 turns on.

    `calendar_aligned` admits `causality` as a declarable relationship, and that stays true: a
    study holding an external intervention design may test it. What no view may do is draw it,
    because the design that licenses the arrow has no field in the manifest, so the picture would
    be asserting something the plan cannot state.
    """
    from src.core.structural_alignment import MODE_RELATIONSHIPS

    assert "causality" in MODE_RELATIONSHIPS["calendar_aligned"]
    with pytest.raises(ForbiddenReadingError) as error:
        assert_reading_admissible(mode, "causality")
    assert "external design" in str(error.value)
    assert "may still declare and test" in str(error.value)


def test_precedence_is_a_calendar_reading_and_not_a_scale_shape_one():
    assert_reading_admissible("calendar_aligned", "precedence")
    with pytest.raises(ForbiddenReadingError) as error:
        assert_reading_admissible("scale_shape_aligned", "precedence")
    assert "carries no clock" in str(error.value)


def test_an_unknown_reading_lists_the_readings_rather_than_failing_a_lookup():
    with pytest.raises(UnknownNameError):
        assert_reading_admissible("calendar_aligned", "vibes")


def test_an_unknown_mode_is_refused_before_the_reading_is_considered():
    with pytest.raises(UnknownNameError):
        assert_reading_admissible("eyeball_aligned", "magnitude_equivalence")


# ----------------------------------------------------------------------------- the encodings


def test_every_role_is_distinguishable_in_three_channels():
    encodings = ordered_encodings()
    for channel in ("word", "colour", "marker"):
        values = [getattr(item, channel) for item in encodings]
        assert len(values) == len(set(values)), channel


def test_a_role_duplicating_any_single_channel_is_refused():
    existing = encoding_for("null_draw")
    with pytest.raises(Exception, match="only two of three channels"):
        register_encoding(Encoding("shadow", word="shadow", colour=existing.colour,
                                   marker="star", ordinal=9, admits_claim=False,
                                   definition="a role reusing one colour"))
    assert "shadow" not in ENCODINGS.names()


def test_only_confirmations_and_transfers_admit_a_claim():
    """A legend that let a candidate admit a claim would undo the generate/confirm split."""
    admitting = {item.role for item in ordered_encodings() if item.admits_claim}
    assert admitting == {"held_out_confirmation", "external_transfer"}


def test_the_candidate_definition_says_its_selection_used_the_data():
    assert "selection used the data" in encoding_for("generated_candidate").definition


# --------------------------------------------------------------------------------- the marks


def test_a_mark_must_carry_an_artefact_or_say_why_it_has_none():
    with pytest.raises(Exception, match="exactly one of an artefact digest"):
        Mark(label="floating", role="null_draw")
    with pytest.raises(Exception, match="exactly one of an artefact digest"):
        Mark(label="both", role="null_draw", artifact_sha256="a" * 64,
             no_artifact_reason="and also this")
    assert Mark(label="fine", role="null_draw", artifact_sha256="a" * 64).describe()["role"]


def test_a_mark_carries_its_role_word_and_marker_into_the_payload():
    described = Mark(label="one", role="held_out_confirmation",
                     no_artifact_reason="not run yet").describe()
    assert described["word"] == "confirmed"
    assert described["marker"] == "filled_square"
    assert described["admits_claim"] is True


def test_a_mark_with_an_unregistered_role_is_refused():
    with pytest.raises(UnknownNameError):
        Mark(label="one", role="vibe", no_artifact_reason="none")


# ---------------------------------------------------------------------------------- the path


def test_seven_views_are_registered_and_ordered_by_ordinal():
    views = ordered_views()
    assert [view.view_id for view in views] == [
        "coverage_timeline", "native_preview", "scale_mapping", "result_matrix",
        "motif_correspondence", "null_and_correction", "provenance_drilldown"]
    assert [view.ordinal for view in views] == list(range(1, 8))


def test_the_order_is_the_declared_ordinal_and_not_how_names_happen_to_sort():
    """`Registry.entries()` sorts alphabetically, and `coverage_timeline` sorts before
    `native_preview` only by luck. Reading order is a scientific statement - coverage before
    results, results before provenance - so it is declared."""
    alphabetical = [entry.value.view_id for entry in COMPARISON_VIEWS.entries()]
    assert alphabetical != [view.view_id for view in ordered_views()]


def test_every_view_states_what_it_may_not_conclude(context):
    for view in ordered_views():
        assert view.may_conclude.strip()
        assert view.may_not_conclude, view.view_id


def test_every_rendered_view_carries_axes_a_legend_and_a_table(context):
    for view in ordered_views():
        payload = render_view(view.view_id, context)
        assert payload["axes"], view.view_id
        assert payload["legend"], view.view_id
        assert payload["table"]["columns"], view.view_id
        assert payload["may_not_conclude"], view.view_id
        assert payload["mode_forbids"]


def test_no_rendered_axis_shares_a_native_magnitude(context):
    for view in ordered_views():
        for axis in render_view(view.view_id, context)["axes"]:
            assert not (axis["kind"] == "native_magnitude" and axis["shared"]), view.view_id


def test_render_all_returns_every_registered_view_once(context):
    payload = render_all(context)
    ids = [view["view_id"] for view in payload["views"]]
    assert ids == [view.view_id for view in ordered_views()]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------------ coverage timeline


def test_absent_coverage_is_a_named_state_and_never_the_number_zero(context):
    body = render_view("coverage_timeline", context)["body"]
    states = {cell["state"] for row in body["rows"] for cell in row["cells"]}
    assert states <= set(COVERAGE_CELLS)
    for row in body["rows"]:
        for cell in row["cells"]:
            assert cell["is_measured_zero"] is False
            assert isinstance(cell["state"], str)
    assert "not the number zero" in body["zero_distinction"]


def test_the_bespoke_domain_keeps_a_refused_row_rather_than_disappearing(context):
    body = render_view("coverage_timeline", context)["body"]
    order_book = next(row for row in body["rows"] if row["domain"] == "order_book")
    assert {cell["state"] for cell in order_book["cells"]} == {"REFUSED"}
    assert all(cell["reason"].strip() for cell in order_book["cells"])
    assert order_book["cells"][0]["mark"]["word"] == "refused"


def test_a_sector_bounded_domain_reads_as_sparse_not_as_covered(context):
    body = render_view("coverage_timeline", context)["body"]
    tess = next(row for row in body["rows"] if row["domain"] == "tess_lightcurve")
    assert {cell["state"] for cell in tess["cells"]} == {"SPARSE"}
    assert "established only after acquisition" in tess["cells"][0]["reason"]


def test_the_timeline_table_carries_every_cell_that_the_picture_does(context):
    body = render_view("coverage_timeline", context)["body"]
    drawn = sum(len(row["cells"]) for row in body["rows"])
    payload = render_view("coverage_timeline", context)
    assert len(payload["table"]["rows"]) == drawn


# --------------------------------------------------------------------------- native preview


def test_each_domain_keeps_its_own_native_magnitude_axis(context):
    body = render_view("native_preview", context)["body"]
    assert body["shared_magnitude_axis"] is None
    for row in body["rows"]:
        axis = row["native"]["axis"]
        assert axis["kind"] == "native_magnitude"
        assert axis["domains"] == [row["domain"]]
        assert row["comparable_across_domains"] is False
        assert row["native"]["units"] in row["why_not"]


def test_the_canonical_side_is_dimensionless_and_says_so(context):
    body = render_view("native_preview", context)["body"]
    for row in body["rows"]:
        assert row["canonical"]["axis"]["kind"] == "dimensionless_statistic"


def test_the_native_preview_reports_what_each_domain_breaks(context):
    body = render_view("native_preview", context)["body"]
    violations = {row["domain"]: row["canonical"]["violations"] for row in body["rows"]}
    assert any(violations.values()), "at least one flagship domain breaks an assumption by name"


# ----------------------------------------------------------------------------- scale mapping


def test_a_shared_coordinate_keeps_each_domains_native_duration(context):
    body = render_view("scale_mapping", context)["body"]
    assert body["carries_a_clock"] is False
    first = body["rows"][0]
    durations = {entry["domain"]: entry["native_duration_seconds"] for entry in first["domains"]}
    resolvable = [value for value in durations.values() if value is not None]
    assert len(set(resolvable)) > 1, (
        "the same coordinate must mean different native durations in different domains, or the "
        "mapping is not carrying the information it exists to carry")


def test_a_domain_with_no_regular_cadence_says_so_rather_than_guessing(context):
    body = render_view("scale_mapping", context)["body"]
    unresolvable = [entry for row in body["rows"] for entry in row["domains"]
                    if not entry["resolvable"]]
    for entry in unresolvable:
        assert "no regular native cadence" in entry["why_not"]


def test_the_scale_view_forbids_simultaneity_and_precedence():
    view = COMPARISON_VIEWS.get("scale_mapping")
    assert "simultaneity" in view.may_not_conclude
    assert "precedence" in view.may_not_conclude


# ----------------------------------------------------------------------------- result matrix


def test_the_matrix_enumerates_the_declared_arities_and_no_others(context):
    body = render_view("result_matrix", context)["body"]
    arities = {cell["arity"] for cell in body["cells"]}
    assert arities <= set(int(value) for value in context.spec.family.domain_arities)
    assert 2 in arities


def test_every_matrix_cell_shows_the_denominator_it_would_be_corrected_over(context):
    body = render_view("result_matrix", context)["body"]
    denominator = body["correction"]["denominator"]
    assert denominator > 0
    assert all(cell["corrected_over"] == denominator for cell in body["cells"])
    assert body["correction"]["declared_search_members"] >= denominator


def test_an_unmeasured_cell_is_not_yet_measured_and_says_why(context):
    body = render_view("result_matrix", context)["body"]
    admissible = [cell for cell in body["cells"] if cell["admissible"]]
    assert admissible
    for cell in admissible:
        assert cell["status"] == NOT_YET_MEASURED
        assert "unasked question" in cell["reason"] or "not been measured" in cell["reason"]


def test_a_matrix_declaring_causality_renders_it_as_a_refusal_not_as_a_result():
    """The trap in its most tempting form: the relationship *is* declared, so the cell exists.

    It is drawn - a blank would be indistinguishable from a cell nobody thought about - and what
    is drawn in it is the refusal and its reason.
    """
    body = json.loads(canonical_bytes(flagship_recipe()))
    body["family"]["relationships"] = ["co_occurrence", "causality"]
    context = _context_for(CrossDomainExperimentSpec.parse_obj(body))
    cells = render_view("result_matrix", context)["body"]["cells"]
    causal = [cell for cell in cells if cell["relationship"] == "causality"]
    assert causal, "the declared relationship must still occupy its cells"
    for cell in causal:
        assert cell["status"] == "REFUSED"
        assert cell["admissible"] is False
        assert "external design" in cell["reason"]
        assert cell["mark"]["word"] == "refused"
        assert cell["mark"]["admits_claim"] is False


def test_the_matrix_may_not_conclude_causality():
    assert "causality" in COMPARISON_VIEWS.get("result_matrix").may_not_conclude


# ---------------------------------------------------------------------- motif correspondence


def test_a_manifest_declaring_no_motif_says_so_rather_than_showing_an_empty_grid(context):
    body = render_view("motif_correspondence", context)["body"]
    assert body["declared_motifs"] == []
    assert body["rows"] == []
    assert "no frozen shape to look for" in body["unavailable_reason"]
    assert "priced" in body["unavailable_reason"]


def test_a_declared_motif_produces_transfer_rows_that_name_the_spent_target():
    body = json.loads(canonical_bytes(flagship_recipe()))
    body["family"]["motifs"] = ["winter_ramp"]
    context = _context_for(CrossDomainExperimentSpec.parse_obj(body))
    rendered = render_view("motif_correspondence", context)["body"]
    assert rendered["declared_motifs"] == ["winter_ramp"]
    assert {row["target_domain"] for row in rendered["rows"]} == set(context.domains)
    assert all(row["target_spent"] is False for row in rendered["rows"])
    assert "spent by its first opening" in rendered["transfer_note"]


# ------------------------------------------------------------------- nulls and correction


def test_every_declared_null_reports_what_it_preserves_and_destroys(context):
    body = render_view("null_and_correction", context)["body"]
    assert body["nulls"]
    for null in body["nulls"]:
        assert null["preserves"], null["name"]
        assert set(null["preserves"]).isdisjoint(null["destroys"])
        assert null["distribution"]["status"] == NOT_YET_MEASURED


def test_the_resolution_panel_shows_the_floor_beside_the_threshold(context):
    body = render_view("null_and_correction", context)["body"]
    resolution = body["resolution"]
    assert resolution["p_value_floor"] > 0.0
    assert resolution["surrogates_required"] > 0
    assert "incapable of rejecting anything" in resolution["note"]


def test_the_correction_panel_names_its_unit_and_its_held_out_partition(context):
    body = render_view("null_and_correction", context)["body"]
    correction = body["correction"]
    assert correction["method"] == context.spec.correction
    assert correction["denominator"] > 0
    assert correction["stage"] in ("confirmatory_only", "generate_then_confirm")


def test_a_null_distribution_is_marked_as_a_surrogate_and_admits_no_claim(context):
    body = render_view("null_and_correction", context)["body"]
    mark = body["nulls"][0]["distribution"]["mark"]
    assert mark["word"] == "surrogate"
    assert mark["admits_claim"] is False


# --------------------------------------------------------------------------- provenance


def test_provenance_names_source_adapter_parameters_and_support(context):
    body = render_view("provenance_drilldown", context)["body"]
    for row in body["rows"]:
        assert row["source"]["source_id"]
        assert row["source"]["licence"].strip()
        assert len(row["adapter"]["definition_sha256"]) == 64
        assert "support_kind" in row["support"]


def test_provenance_says_no_artefact_exists_rather_than_showing_a_blank(context):
    body = render_view("provenance_drilldown", context)["body"]
    for row in body["rows"]:
        assert row["artefact"]["sha256"] is None
        assert "no acquisition artefact exists" in row["artefact"]["reason"]
    assert body["run"]["state"] == "NO_RUN"


def test_provenance_carries_the_manifest_address_the_view_was_built_from(context):
    from src.core.experiment_manifest import manifest_sha256

    body = render_view("provenance_drilldown", context)["body"]
    assert body["manifest_sha256"] == manifest_sha256(context.spec)


# --------------------------------------------------------------------- linked selection


def test_selecting_a_window_highlights_each_domains_own_native_interval(context):
    linked = linked_selection(context, window="week")
    assert [row["domain"] for row in linked["contributions"]] == list(context.domains)
    assert linked["merged_interval"] is None
    assert "only one domain addresses" in linked["why_not_merged"]


def test_a_refused_domain_contributes_nothing_and_says_which_it_is(context):
    linked = linked_selection(context, window="week")
    order_book = next(row for row in linked["contributions"] if row["domain"] == "order_book")
    assert order_book["state"] == "REFUSED"
    assert order_book["contributes"] is False


def test_a_selection_may_be_narrowed_to_named_domains(context):
    linked = linked_selection(context, window="three_months",
                              domains=["reanalysis", "argo_float"])
    assert [row["domain"] for row in linked["contributions"]] == ["reanalysis", "argo_float"]


def test_an_undeclared_window_or_domain_is_refused_by_name(context):
    with pytest.raises(UnknownNameError):
        linked_selection(context, window="a_fortnight")
    with pytest.raises(UnknownNameError):
        linked_selection(context, window="week", domains=["seismic"])


# ------------------------------------------------------------------------ nothing measured


def test_no_view_reports_a_measured_value(context):
    """The property that makes this whole module safe to render before any worker exists.

    Recursive over keys, because the failure would arrive as a nested field somebody added
    without noticing what it implied. `p_value_floor` and `surrogates_required` are deliberately
    permitted: both are arithmetic about the declared ensemble and are computable before a byte
    exists, which is exactly why they belong in front of a researcher *before* the run.
    """
    payload = render_all(context)
    found = set()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                found.add(key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    for forbidden in ("observed_p_value", "measurement", "effect_size", "correlation",
                      "test_statistic", "rejected", "significant", "observed_value"):
        assert forbidden not in found, forbidden
    assert "p_value_floor" in found
    assert payload["views"][0]["results_exist"] is False


def test_results_exist_requires_a_mining_artefact_and_not_merely_a_complete_state(spec):
    """A COMPLETE run under `partial_permitted` can be missing mining components by name.

    If this returned `state == "COMPLETE"`, a run that completed without mining anything would
    render its result matrix as though the cells had been measured and come back empty.
    """
    complete_without_mining = ViewContext(
        spec=spec, preflight=preflight_manifest(spec),
        receipt={"state": "COMPLETE", "artefacts": {"ACQUIRING/reanalysis": "a" * 64}})
    assert complete_without_mining.results_exist is False
    with_mining = ViewContext(
        spec=spec, preflight=preflight_manifest(spec),
        receipt={"state": "COMPLETE", "artefacts": {"MINING/pair": "b" * 64}})
    assert with_mining.results_exist is True


def test_a_run_that_has_not_completed_explains_the_empty_cell_by_its_state(spec):
    context = ViewContext(spec=spec, preflight=preflight_manifest(spec),
                          receipt={"state": "ACQUIRING", "artefacts": {}})
    cell = render_view("result_matrix", context)["body"]["cells"][0]
    assert "ACQUIRING" in cell["reason"]
    assert "has not been measured" in cell["reason"]


# -------------------------------------------------------------------------------- contract


def test_the_contract_is_generated_from_the_registries():
    contract = describe_views()
    assert [view["view_id"] for view in contract["views"]] == [
        view.view_id for view in ordered_views()]
    assert [item["role"] for item in contract["encodings"]] == [
        item.role for item in ordered_encodings()]


def test_the_contract_separates_what_may_be_declared_from_what_may_be_drawn():
    readings = describe_views()["readings"]
    assert "causality" in readings["declarable_by_mode"]["calendar_aligned"]
    assert "causality" not in readings["renderable_by_mode"]["calendar_aligned"]
    assert readings["requires_external_design"] == list(REQUIRES_EXTERNAL_DESIGN)
    assert "picture may assert" in readings["why_two_lists"]


def test_the_contract_names_the_three_refusals_this_slice_exists_for():
    refusals = describe_views()["refusals"]
    assert "MagnitudeEquivalenceError" in refusals["shared_native_magnitude_axis"]
    assert "no code path" in refusals["absent_is_not_zero"]
    assert "artefact digest" in refusals["mark_without_provenance"]


# ------------------------------------------------------------------------------------- api


def test_the_api_serves_the_contract_and_the_legend(client):
    contract = client.get("/api/v1/comparison-views")
    assert contract.status_code == 200
    assert len(contract.json()["views"]) == 7
    assert contract.json()["not_yet_available"] == ["measured values from executed stage workers"]
    legend = client.get("/api/v1/comparison-views/encodings")
    assert legend.status_code == 200
    assert "still has two" in legend.json()["why_three_channels"]
    assert len(legend.json()["encodings"]) == len(ordered_encodings())


def test_the_api_renders_every_view_from_one_manifest(client):
    manifest = json.loads(canonical_bytes(flagship_recipe()))
    response = client.post("/api/v1/comparison-views/render", json=manifest)
    assert response.status_code == 200
    assert len(response.json()["views"]) == 7


def test_the_api_renders_one_view_and_refuses_an_unregistered_one(client):
    manifest = json.loads(canonical_bytes(flagship_recipe()))
    ok = client.post("/api/v1/comparison-views/render/coverage_timeline", json=manifest)
    assert ok.status_code == 200
    assert ok.json()["view_id"] == "coverage_timeline"
    missing = client.post("/api/v1/comparison-views/render/scatter_everything", json=manifest)
    assert missing.status_code == 404
    assert "registered" in missing.json()["detail"]


def test_the_api_answers_a_linked_selection_per_domain(client):
    manifest = json.loads(canonical_bytes(flagship_recipe()))
    response = client.post("/api/v1/comparison-views/linked-selection",
                           json={"manifest": manifest, "window": "week"})
    assert response.status_code == 200
    assert response.json()["merged_interval"] is None
    bad = client.post("/api/v1/comparison-views/linked-selection",
                      json={"manifest": manifest, "window": "yesterday"})
    assert bad.status_code == 422


def test_the_api_says_why_a_reading_is_refused_rather_than_only_that_it_is(client):
    refused = client.post("/api/v1/comparison-views/readings/check",
                          json={"mode": "calendar_aligned", "reading": "causality"})
    assert refused.status_code == 200
    assert refused.json()["renderable"] is False
    assert "external design" in refused.json()["reason"]
    allowed = client.post("/api/v1/comparison-views/readings/check",
                          json={"mode": "calendar_aligned", "reading": "co_occurrence"})
    assert allowed.json()["renderable"] is True


def test_rendering_a_view_does_not_open_a_run(client, tmp_path, monkeypatch):
    """A read must not be the act that freezes a plan.

    `RunStore.open` publishes a frozen manifest and creates the run directory. If these routes
    opened rather than looked, merely visiting the Interpret tab would start the run whose
    provenance it was displaying.
    """
    from src.api.main import app

    monkeypatch.setattr(app.state, "experiment_run_dir", tmp_path, raising=False)
    manifest = json.loads(canonical_bytes(flagship_recipe()))
    response = client.post("/api/v1/comparison-views/render/provenance_drilldown",
                           json=manifest)
    assert response.status_code == 200
    assert response.json()["run_state"] == "NO_RUN"
    assert not (tmp_path / "runs").exists()


def test_every_comparison_view_route_lives_under_its_own_prefix(client):
    paths = {route.path for route in client.app.routes
             if route.path.startswith("/api/v1/comparison-views")}
    assert len(paths) == 6
    assert all("run" not in path for path in paths), (
        "reading a view and running an experiment are different surfaces")
