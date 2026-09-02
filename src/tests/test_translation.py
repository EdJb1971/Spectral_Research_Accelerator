"""TG7.4: translation, bounded.

What these tests are for.  The four rules this module has to hold are held three different ways,
and the tests are organised around that distinction rather than around the module's functions:

*   R9 and R19 are held **by construction** — ``AssociationFigures`` cannot exist without all six
    figures, and a cross-domain sentence has no access to a unit-bearing field.  Those are tested
    by trying to break the construction.
*   R7 and R22 are held by **screening at registration** and by the shape of ``translate``'s
    signature.  Those are tested with a hostile glossary that attempts every promotion in turn.
*   The guards are **backstops** that catch an edit to the module rather than a bad input.  Those
    are tested by making the edits, which is what the mutation section does.

The acceptance test of the phase is
``test_one_bundle_in_three_glossaries_states_identical_facts_in_different_words``: the same
corpus rendered into an atmospheric and a financial vocabulary must give two documents that read
completely differently and assert exactly the same set of structural facts.  If translation could
move a claim, that is the test that would fail.
"""

from __future__ import annotations

import itertools
import json
import random
from types import SimpleNamespace

import pytest

from src.core.claim_ladder import CLAIM_RUNGS, OUTSIDE_THE_LADDER, _GATES
from src.core.errors import InvalidParameterError
from src.core.evidence import EVIDENCE_FIELDS
from src.core.domain import AxisSpec
from src.core.feature import FeatureLocation, Quantity, SpectralFeature
from src.core.five_outputs import STRUCTURAL_ALTERNATIVES, summarise_evidence
from src.core.registry import restore, snapshot
from src.core.translation import (
    ALTERNATIVE_NAMES,
    DOMAIN_GLOSSARIES,
    GATE_NAMES,
    STRUCTURAL_VOCABULARY,
    AssociationFigures,
    DomainGlossary,
    TranslatedFinding,
    TranslationUnit,
    assert_no_bare_confidence,
    assert_no_causal_language,
    assert_no_semantic_comparison,
    assert_structural_only,
    glossary_for,
    load_translation,
    render_comparison,
    save_translation,
    translate,
    verify_translation_independence,
)
from src.tests.test_five_outputs import (
    ALL_REQUIREMENTS,
    RUNG_REQUIREMENTS,
    _bundle,
    _climbed,
    _moment,
    _with,
)

ROW = AxisSpec("row", "space", units="cells", ordinal=0)
COL = AxisSpec("col", "space", units="cells", ordinal=1)


# ------------------------------------------------------------ the three glossaries


def _phrases(wording):
    """Build a total glossary from a per-kind wording function."""
    return {term: wording(*term.split(".", 1)) for term in STRUCTURAL_VOCABULARY}


#: An atmospheric vocabulary.  Deliberately fluent: the point of translation is that a reader
#: who does not know what `robust_association.confounders_addressed` means can still act.
ATMOSPHERIC = _phrases(lambda kind, rest: {
    "rung": "a %s standing in the reanalysis" % rest.replace("_", " "),
    "gate": "the %s check over the reanalysis" % rest.rsplit(".", 1)[-1].replace("_", " "),
    "alternative": "atmospheric %s" % rest.replace("_", " "),
    "status": "%s against the archive" % rest.lower(),
    "category": "%s in the reanalysis record" % rest.replace("_", " "),
}[kind])

#: A vocabulary from a domain sharing nothing with the first.  This is what actually tests R19:
#: two glossaries over one bundle must not be able to say anything about each other.
FINANCIAL = _phrases(lambda kind, rest: {
    "rung": "a %s in the order book" % rest.replace("_", " "),
    "gate": "the %s control on the venue feed" % rest.rsplit(".", 1)[-1].replace("_", " "),
    "alternative": "market %s" % rest.replace("_", " "),
    "status": "%s at settlement" % rest.lower(),
    "category": "%s in the trade record" % rest.replace("_", " "),
}[kind])


def _atmospheric():
    return DomainGlossary(domain="reanalysis", phrases=ATMOSPHERIC,
                          description="ERA5-flavoured wording for the structural vocabulary.")


def _financial():
    return DomainGlossary(domain="exchange", phrases=FINANCIAL,
                          description="Order-book wording for the structural vocabulary.")


#: Every way a glossary can try to promote a finding in wording alone, each paired with the term
#: it attacks.  These are not hypothetical: each is a sentence somebody would write in good
#: faith, and each would move what a reader believes without moving a single gate.
HOSTILE_PHRASES = (
    ("rung.association", "an early warning signal in the feed",
     "borrows candidate_precursor's wording for a rung two below it"),
    ("rung.observation", "a robust replicated relationship",
     "borrows two rungs at once"),
    ("rung.robust_association", "a predictive indicator",
     "borrows the top rung"),
    ("gate.association.uncertainty_quantified", "the check on what causes the spread",
     "smuggles causal vocabulary through a gate's wording"),
    ("category.effect_sizes", "the size of the effect, some 82 percent of the time",
     "puts a magnitude in the vocabulary rather than in the record"),
    ("alternative.chance", "an explanation stronger than the hypothesis",
     "asserts a relation of size the structure does not support"),
)


# ------------------------------------------------------------ 1. the vocabulary is closed


def test_the_restated_gate_names_still_match_the_ladders_own():
    """The one line of duplication in the module, guarded.

    ``claim_ladder._GATES`` is private and TG7.4 may not modify the module that owns it (R22),
    so the gate names are restated.  A restatement that drifts is worse than none: a new gate
    would get no domain wording and would reach a reader as raw programme jargon.  This makes
    that drift a failing test rather than a bad sentence.
    """
    assert GATE_NAMES == tuple(gate.name for gate in _GATES)


def test_the_structural_vocabulary_is_total_over_every_term_a_reader_can_meet():
    assert set(ALTERNATIVE_NAMES) == {identifier for identifier, _d, _c
                                      in STRUCTURAL_ALTERNATIVES.values()}
    for rung in CLAIM_RUNGS:
        assert "rung.%s" % rung in STRUCTURAL_VOCABULARY
    for gate in GATE_NAMES:
        assert "gate.%s" % gate in STRUCTURAL_VOCABULARY
    for category in EVIDENCE_FIELDS:
        assert "category.%s" % category in STRUCTURAL_VOCABULARY
    assert len(set(STRUCTURAL_VOCABULARY)) == len(STRUCTURAL_VOCABULARY)


def test_a_glossary_missing_one_term_is_refused_rather_than_falling_back_to_jargon():
    partial = dict(ATMOSPHERIC)
    del partial["rung.candidate_precursor"]
    with pytest.raises(InvalidParameterError) as excinfo:
        DomainGlossary(domain="reanalysis", phrases=partial)
    assert "rung.candidate_precursor" in str(excinfo.value)


def test_a_glossary_may_reword_the_structure_but_not_add_to_it():
    with pytest.raises(InvalidParameterError) as excinfo:
        DomainGlossary(domain="reanalysis",
                       phrases=dict(ATMOSPHERIC, **{"rung.certainty": "certain"}))
    assert "structural vocabulary" in str(excinfo.value)


# ------------------------------------------------------------ 2. R9 has a structure now


def _figures(**overrides):
    values = dict(support=40, confidence=0.82, base_rate=0.40, lift=0.82 / 0.40,
                  lift_interval=(1.6, 2.5), surrogate_corrected_lift=1.9)
    values.update(overrides)
    return AssociationFigures(**values)


def test_the_roadmaps_own_cautionary_example_renders_with_the_context_that_defuses_it():
    """*"82% of the time"* is the sentence R9 was written to prevent.

    It is not banned here; it is made honest.  The base rate that decides whether 82% is a
    finding or noise is in the same string, so there is no rendering of the one without the
    other.
    """
    rendered = _figures().render()
    assert "82.0%" in rendered
    assert "base rate of 40.0%" in rendered
    assert "lift 2.05" in rendered
    assert "1.90 after surrogate correction" in rendered


def test_a_confidence_without_a_base_rate_is_not_a_weaker_finding_but_no_finding():
    with pytest.raises(InvalidParameterError) as excinfo:
        AssociationFigures.from_payload({"confidence": 0.82})
    message = str(excinfo.value)
    assert "all six" in message
    for missing in ("support", "base_rate", "lift", "surrogate_corrected_lift"):
        assert missing in message


@pytest.mark.parametrize("dropped", ["support", "confidence", "base_rate", "lift",
                                     "lift_interval", "surrogate_corrected_lift"])
def test_every_one_of_r9s_six_figures_is_load_bearing(dropped):
    payload = dict(_figures().to_mapping())
    del payload[dropped]
    with pytest.raises(InvalidParameterError) as excinfo:
        AssociationFigures.from_payload(payload)
    assert dropped in str(excinfo.value)


def test_a_lift_that_is_not_the_ratio_it_claims_to_be_is_refused():
    """The one figure a reader cannot check without doing the division themselves."""
    with pytest.raises(InvalidParameterError) as excinfo:
        _figures(lift=9.9)
    assert "confidence over base rate" in str(excinfo.value)


def test_a_base_rate_of_zero_is_refused_because_lift_against_it_is_not_a_number():
    with pytest.raises(InvalidParameterError):
        _figures(base_rate=0.0, lift=1.0)


def test_a_figure_set_round_trips_through_a_payload_unchanged():
    figures = _figures()
    assert AssociationFigures.from_payload(figures.to_mapping()) == figures


def test_an_interval_spanning_one_reports_that_it_has_not_beaten_the_base_rate():
    assert _figures().beats_base_rate is True
    assert _figures(lift_interval=(0.8, 2.5)).beats_base_rate is False
    assert _figures(surrogate_corrected_lift=0.9).survives_surrogate is False


# ------------------------------------------------------------ 3. the hostile glossary


@pytest.mark.parametrize("term,phrase,why", HOSTILE_PHRASES,
                         ids=[item[0] for item in HOSTILE_PHRASES])
def test_a_glossary_that_promotes_a_finding_in_wording_is_refused_at_registration(term, phrase,
                                                                                 why):
    """Each of these moves what a reader believes without moving a gate.  That is the attack.

    Refusing at registration rather than at render matters: the error names the phrase to the
    person who wrote the glossary, once, instead of surfacing to a reader who cannot fix it.
    """
    with pytest.raises(InvalidParameterError) as excinfo:
        DomainGlossary(domain="hostile", phrases=dict(ATMOSPHERIC, **{term: phrase}))
    assert term in str(excinfo.value), why


def test_the_hostile_glossary_cannot_be_registered_at_all():
    hostile = dict(ATMOSPHERIC)
    for term, phrase, _why in HOSTILE_PHRASES:
        hostile[term] = phrase
    with pytest.raises(InvalidParameterError):
        DomainGlossary(domain="hostile", phrases=hostile)


@pytest.mark.parametrize("word", ["causes", "mechanism", "explains", "efficacy"])
def test_causal_vocabulary_cannot_enter_through_a_glossary(word):
    with pytest.raises(InvalidParameterError) as excinfo:
        DomainGlossary(domain="reanalysis",
                       phrases=dict(ATMOSPHERIC,
                                    **{"status.PASS": "the reading that %s it" % word}))
    assert word in str(excinfo.value)


@pytest.mark.parametrize("word", ["causes", "mechanism", "explains", "efficacy"])
def test_a_causal_word_inside_an_identifier_is_caught_too(word):
    """D89, found by T4D.3's narratives, which run the same guard over their own prose.

    An underscore is a word character, so a plain `\bcauses\b` scan passes straight over
    `co2_causes_warming` -- and an identifier is exactly the shape of the author-supplied text
    that reaches `rendered` through an entry's label or an alternative's wording. Punctuation
    is now flattened to spaces before the boundaries are applied. What is still not caught,
    and is asserted here so the limit is recorded rather than assumed away, is a word run
    together with another with no separator at all; a substring match would catch that and
    would also refuse "causeway".
    """
    unit = TranslationUnit(structural_key="rung.observation", source_sha256="0" * 64,
                           rendered="the co2_%s_warming record was read" % word,
                           licences="this is not a claim about mechanism")
    with pytest.raises(InvalidParameterError) as excinfo:
        assert_no_causal_language(SimpleNamespace(units=(unit,)))
    assert word in str(excinfo.value) and "R7" in str(excinfo.value)

    innocent = TranslationUnit(structural_key="rung.observation", source_sha256="0" * 64,
                               rendered="the causeway record was read",
                               licences="this is not a claim about mechanism")
    assert_no_causal_language(SimpleNamespace(units=(innocent,)))


def test_a_digit_cannot_enter_through_a_glossary():
    """Every number a reader sees must come from the record, not from the vocabulary."""
    with pytest.raises(InvalidParameterError) as excinfo:
        DomainGlossary(domain="reanalysis",
                       phrases=dict(ATMOSPHERIC, **{"status.PASS": "passed 3 checks"}))
    assert "without digits" in str(excinfo.value)


# ------------------------------------------------------------ 4. R19 by construction


def _feature(domain, variable, units, **overrides):
    kwargs = dict(domain=domain, dataset="%s_v1" % domain, variable=variable,
                  magnitude=Quantity(3.0, units),
                  location=FeatureLocation({"row": 1.0, "col": 2.0}, (ROW, COL)),
                  time=0.0, time_units="frames", representation="identity",
                  spatial_scale=Quantity(4.0, "cells"))
    kwargs.update(overrides)
    return SpectralFeature(**kwargs)


def test_two_features_of_one_variable_may_be_described_with_their_units():
    left = _feature("era5", "temperature", "K")
    right = _feature("era5", "temperature", "K", magnitude=Quantity(5.0, "K"))
    rendered = render_comparison(left, right)
    assert "temperature" in rendered
    assert "K" in rendered


def test_a_cross_domain_rendering_carries_no_unit_bearing_field():
    """The acceptance criterion for R19, asserted as an absence.

    What makes this a real check rather than a spot inspection is that it asserts what is *not*
    in the sentence: no variable, no dataset, no units. A renderer that reached past the
    structural signature would put one of them there.
    """
    weather = _feature("era5", "temperature", "K")
    market = _feature("exchange", "volume", "shares")
    rendered = render_comparison(weather, market)
    for forbidden in ("temperature", "volume", "era5_v1", "exchange_v1", " K", "shares"):
        assert forbidden not in rendered
    assert "identity representation" in rendered
    assert_structural_only(rendered, [weather, market])


def test_the_cross_domain_sentence_cannot_acquire_a_magnitude_even_deliberately():
    weather = _feature("era5", "temperature", "K", magnitude=Quantity(291.5, "K"))
    market = _feature("exchange", "volume", "shares", magnitude=Quantity(1e6, "shares"))
    rendered = render_comparison(weather, market)
    assert "291.5" not in rendered
    assert "1000000" not in rendered


def test_the_structural_only_guard_catches_a_leak_a_widened_template_would_cause():
    """The backstop, tested by handing it what a broken renderer would have produced."""
    weather = _feature("era5", "temperature", "K")
    market = _feature("exchange", "volume", "shares")
    leaked = "the temperature structure resembles the other"
    with pytest.raises(InvalidParameterError) as excinfo:
        assert_structural_only(leaked, [weather, market])
    assert "variable" in str(excinfo.value)


# ------------------------------------------------------------ 5. translation states facts


def _outputs(rung="candidate_precursor"):
    return summarise_evidence(_climbed(rung))


def test_a_translation_binds_the_revision_and_the_glossary_that_worded_it():
    outputs = _outputs()
    document = translate(outputs, _atmospheric())
    assert document.summary_sha256 == outputs.summary_sha256
    assert document.glossary_sha256 == _atmospheric().glossary_sha256
    assert document.revision == outputs.revision


def test_one_bundle_in_three_glossaries_states_identical_facts_in_different_words():
    """The acceptance test of the phase.

    Two fluent, mutually unintelligible vocabularies over one corpus standing on all five rungs
    plus a blocked bundle and a contradicted one.  Every document must assert exactly the same
    structural facts, and none of the wording may be shared.  The third glossary is the hostile
    one, and it never gets this far — it is refused at registration, which is asserted above.
    """
    corpus = [_climbed(rung) for rung in CLAIM_RUNGS] + [_blocked(), _contradicted()]
    for bundle in corpus:
        outputs = summarise_evidence(bundle)
        atmospheric = translate(outputs, _atmospheric())
        financial = translate(outputs, _financial())

        assert atmospheric.structural_keys == financial.structural_keys
        assert atmospheric.claim_text() != financial.claim_text()
        assert atmospheric.translation_sha256 != financial.translation_sha256
        # The claim state itself is untouched by either wording.
        assert atmospheric.summary_sha256 == outputs.summary_sha256
        assert financial.summary_sha256 == outputs.summary_sha256


def test_every_claimable_rung_is_rendered_welded_to_what_it_does_not_license():
    """A UI cannot show the claim and drop the bound: they are one string, not two."""
    document = translate(_outputs("candidate_precursor"), _atmospheric())
    keys = [unit.structural_key for unit in document.claimable]
    assert keys == ["rung.%s" % rung for rung in
                    CLAIM_RUNGS[:CLAIM_RUNGS.index("candidate_precursor") + 1]]
    for unit in document.claimable:
        assert unit.licences
        assert unit.licences in unit.text
        assert unit.rendered in unit.text
    top = document.claimable[-1]
    assert "Predictive utility is not shown" in top.text


def test_a_unit_cannot_be_built_without_the_bound_that_makes_it_readable():
    with pytest.raises(InvalidParameterError):
        TranslationUnit(structural_key="rung.association", source_sha256="a" * 64,
                        rendered="an association in the order book", licences="")


def test_the_refusal_of_a_causal_reading_is_stated_at_every_rung_not_only_the_top():
    for rung in CLAIM_RUNGS:
        document = translate(_outputs(rung), _atmospheric())
        assert "outside_the_ladder" in document.structural_keys
        assert "bringing about" in document.render()


def test_a_contradicted_bundle_says_what_argues_against_it_in_domain_words():
    outputs = summarise_evidence(_contradicted())
    document = translate(outputs, _financial())
    assert document.contradicting
    for unit in document.contradicting:
        assert "trade record" in unit.rendered or "settlement" in unit.rendered


def test_an_empty_section_says_so_rather_than_leaving_silence_to_reassure():
    rendered = translate(_outputs("observation"), _atmospheric()).render()
    assert "that is not the same as none existing" in rendered


# ------------------------------------------------------------ 6. R22 at the reader's edge


def test_translate_has_no_parameter_through_which_a_review_could_arrive():
    """R22 enforced by the signature rather than by a check inside it."""
    import inspect
    parameters = inspect.signature(translate).parameters
    assert "outputs" in parameters
    assert not any("review" in name or "bundle" in name for name in parameters)


def test_commentary_is_carried_beside_the_finding_and_never_inside_it():
    outputs = _outputs()
    dissent = "The confounder treatment was disputed and never resolved by the panel."
    document = translate(outputs, _atmospheric(), commentary=[dissent])
    assert dissent not in document.claim_text()
    assert dissent in document.render()
    assert "moved nothing above" in document.render()


def test_commentary_reproduced_inside_the_finding_is_refused_by_name():
    """The one way argument could cross the fence: a phrase copied into the claim text."""
    outputs = _outputs()
    document = translate(outputs, _atmospheric(),
                         commentary=["The confounder treatment was disputed at length."])
    # Assembled from the mapping rather than through `translate`, because `translate` has no
    # path that could produce this: the smuggling it models is a person copying an answer
    # across, which is exactly the failure TG7.1's own check was written for.
    rebuilt = document.to_mapping()
    rebuilt["claimable"][0]["rendered"] = "The confounder treatment was disputed at length"
    with pytest.raises(InvalidParameterError) as excinfo:
        verify_translation_independence(TranslatedFinding.from_mapping(rebuilt),
                                        _climbed("candidate_precursor"))
    assert "no LLM output may ever be" in str(excinfo.value)


def test_a_translation_of_a_superseded_revision_is_refused_as_stale():
    bundle = _climbed("candidate_precursor")
    document = translate(summarise_evidence(bundle), _atmospheric())
    moved = _with(bundle, (("observations", {"n": 999}),), start=40)
    with pytest.raises(InvalidParameterError) as excinfo:
        verify_translation_independence(document, moved)
    assert "no longer produces" in str(excinfo.value)


def test_a_translation_moves_no_claim_level_for_any_bundle_in_the_corpus():
    """R22's executable form for this layer: translating changes nothing about the claim."""
    for bundle in [_climbed(rung) for rung in CLAIM_RUNGS] + [_blocked(), _contradicted()]:
        before = summarise_evidence(bundle)
        document = translate(before, _financial(),
                             commentary=["Every seat argued this deserves promotion."])
        after = summarise_evidence(bundle)
        assert after.summary_sha256 == before.summary_sha256
        assert after.rung == before.rung
        assert verify_translation_independence(document, bundle) == before.summary_sha256


# ------------------------------------------------------------ 7. persistence


def test_a_translation_round_trips_through_disk_byte_identically(tmp_path):
    document = translate(_outputs(), _atmospheric(), figures=_figures(),
                         commentary=["A dissent that outlived the argument."])
    path = tmp_path / "translation.json"
    published = save_translation(path, document)
    loaded = load_translation(path, published_sha256=published)
    assert loaded.to_mapping() == document.to_mapping()
    assert loaded.render() == document.render()
    assert loaded.translation_sha256 == document.translation_sha256


def test_a_translation_is_never_written_over_the_bundle_it_describes(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
    document = translate(_outputs(), _atmospheric())
    with pytest.raises(InvalidParameterError) as excinfo:
        save_translation(bundle_path, document, bundle_path=bundle_path)
    assert "never" in str(excinfo.value)


def test_a_translation_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "translation.json"
    save_translation(path, translate(_outputs(), _atmospheric()))
    with pytest.raises(FileExistsError):
        save_translation(path, translate(_outputs(), _financial()))


def test_a_tampered_translation_is_refused_on_load(tmp_path):
    path = tmp_path / "translation.json"
    published = save_translation(path, translate(_outputs(), _atmospheric()))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["claimable"][0]["rendered"] = "a much stronger reading of the same record"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    with pytest.raises(InvalidParameterError):
        load_translation(path, published_sha256=published)


# ------------------------------------------------------------ 8. the registry (TG8.1)


def test_a_domain_registers_its_own_glossary_without_editing_src():
    """The TG8.1 onboarding condition, met here rather than promised."""
    state = snapshot(DOMAIN_GLOSSARIES)
    try:
        DOMAIN_GLOSSARIES.add("seismic", DomainGlossary(
            domain="seismic",
            phrases=_phrases(lambda kind, rest: "the %s reading of %s"
                             % (kind, rest.replace("_", " ")))),
            description="Registered from the test module, outside src/core.")
        assert "seismic" in DOMAIN_GLOSSARIES.names()
        document = translate(_outputs(), glossary_for("seismic"))
        assert document.domain == "seismic"
        assert "reading of" in document.claim_text()
    finally:
        restore(DOMAIN_GLOSSARIES, state)


# ------------------------------------------------------------ 9. randomised sweeps


def _blocked():
    return _with(_with(_bundle(), ALL_REQUIREMENTS),
                 (("observations", {"n": 4}),), status="FAIL", start=30,
                 label="a failed check", summary="this caps the bundle")


def _contradicted():
    return _with(_with(_bundle(), ALL_REQUIREMENTS),
                 (("contradictory_evidence", {"note": 1}),), start=32,
                 label="a standing contradiction", summary="this argues against")


def _random_bundle(rng):
    """A bundle reaching a random rung, sometimes blocked, sometimes contradicted."""
    reached = rng.randrange(len(CLAIM_RUNGS))
    bundle = _with(_bundle(), tuple(itertools.chain.from_iterable(
        RUNG_REQUIREMENTS[name] for name in CLAIM_RUNGS[1:reached + 1])))
    if rng.random() < 0.25:
        bundle = _with(bundle, (("observations", {"n": 2}),), status="FAIL", start=30,
                       label="a failed check", summary="capped")
    if rng.random() < 0.25:
        bundle = _with(bundle, (("null_results", {"n": 3}),), start=34,
                       label="a recorded null", summary="contrary but not capping")
    return bundle


def test_translation_states_the_same_facts_in_either_vocabulary_over_a_randomised_sweep():
    """Sweep one: 600 documents, checked against an independently written rule.

    The rule restated here is deliberately not the one the module applies — it is derived from
    the ``FiveOutputs`` directly, so agreement means two implementations agree rather than that
    one implementation is self-consistent.
    """
    rng = random.Random(20260826)
    rungs = set()
    checked = 0
    for _ in range(300):
        bundle = _random_bundle(rng)
        outputs = summarise_evidence(bundle)
        rungs.add(outputs.rung)
        expected = (
            tuple("rung.%s" % rung for rung in outputs.claimable)
            + tuple("rung.%s" % item.rung for item in outputs.not_claimable)
            + ("outside_the_ladder",)
            + tuple("entry.%d" % entry.sequence for entry in outputs.contradicting)
            + tuple("alternative.%s" % item.identifier for item in outputs.alternatives)
            + (() if outputs.next_observation is None
               else ("next_observation.%s" % outputs.next_observation.kind,)))
        for glossary in (_atmospheric(), _financial()):
            document = translate(outputs, glossary)
            assert document.structural_keys == expected
            checked += 1
    assert checked == 600
    assert len(rungs) >= 4, "the sweep must reach most rungs to be worth running"


def test_no_rendering_ever_carries_a_number_the_record_does_not_contain():
    """Sweep two: numeral containment, over both vocabularies and with figures attached."""
    rng = random.Random(31415926)
    with_figures = 0
    for _ in range(200):
        bundle = _random_bundle(rng)
        outputs = summarise_evidence(bundle)
        figures = _figures(support=rng.randrange(5, 500)) if rng.random() < 0.5 else None
        with_figures += figures is not None
        for glossary in (_atmospheric(), _financial()):
            document = translate(outputs, glossary, figures=figures)
            assert_no_semantic_comparison(document, outputs)
            assert_no_bare_confidence(document)
            assert_no_causal_language(document)
    assert 50 < with_figures < 150, "both branches must occur for the sweep to mean anything"


def test_the_template_partition_holds_over_a_randomised_sweep_of_feature_pairs():
    """Sweep three: half the pairs share a semantic key, half do not.  None may leak units."""
    rng = random.Random(2718281)
    domains = [("era5", "temperature", "K"), ("exchange", "volume", "shares"),
               ("seismic", "displacement", "m"), ("census", "headcount", "people")]
    shared = crossed = 0
    for _ in range(300):
        if rng.random() < 0.5:
            spec = rng.choice(domains)
            left, right = _feature(*spec), _feature(*spec, magnitude=Quantity(9.0, spec[2]))
            rendered = render_comparison(left, right)
            assert spec[1] in rendered, "a within-domain sentence may name its variable"
            shared += 1
        else:
            first, second = rng.sample(domains, 2)
            left, right = _feature(*first), _feature(*second)
            rendered = render_comparison(left, right)
            assert_structural_only(rendered, [left, right])
            crossed += 1
    assert shared > 100 and crossed > 100


# ------------------------------------------------------------ 10. the mutations
#
# Each of these is an edit somebody could plausibly make to `translation.py`, applied here to
# the data the module works on rather than to the module, and each must be caught. A guard that
# no mutation reaches is a guard that is not doing anything.


def test_mutation_a_cross_domain_sentence_that_gained_a_magnitude_is_caught():
    """Mutation 1: `CROSS_DOMAIN_FIELDS` widened to admit a unit-bearing field."""
    weather = _feature("era5", "temperature", "K", magnitude=Quantity(291.5, "K"))
    market = _feature("exchange", "volume", "shares")
    widened = ("the first is described in the identity representation with a magnitude of "
               "291.5 K; the second is described in the identity representation")
    with pytest.raises(InvalidParameterError):
        assert_structural_only(widened, [weather, market])


def test_mutation_a_unit_stripped_of_its_bound_is_caught():
    """Mutation 2: the entitlement dropped from a `TranslationUnit`."""
    with pytest.raises(InvalidParameterError):
        TranslationUnit(structural_key="rung.candidate_precursor", source_sha256="b" * 64,
                        rendered="a candidate precursor in the reanalysis", licences="   ")


def test_mutation_a_glossary_carrying_a_magnitude_is_caught():
    """Mutation 3: the digit screen relaxed at registration."""
    with pytest.raises(InvalidParameterError):
        DomainGlossary(domain="reanalysis",
                       phrases=dict(ATMOSPHERIC,
                                    **{"rung.association": "a link seen 82 percent of the time"}))


def test_mutation_a_figure_set_built_without_a_base_rate_is_caught():
    """Mutation 4: `AssociationFigures` allowed to construct partially."""
    with pytest.raises(TypeError):
        AssociationFigures(support=40, confidence=0.82, lift=2.05,
                           lift_interval=(1.6, 2.5), surrogate_corrected_lift=1.9)


def test_mutation_a_rendered_numeral_absent_from_the_record_is_caught():
    """Mutation 5: a renderer that computed a number instead of carrying one."""
    outputs = _outputs()
    document = translate(outputs, _atmospheric())
    rebuilt = document.to_mapping()
    rebuilt["claimable"][0]["rendered"] += " over 4173 frames"
    with pytest.raises(InvalidParameterError) as excinfo:
        assert_no_semantic_comparison(TranslatedFinding.from_mapping(rebuilt), outputs)
    assert "4173" in str(excinfo.value)


def test_mutation_a_bare_percentage_reaching_a_unit_is_caught():
    """Mutation 6: a template that formatted a confidence directly."""
    outputs = _outputs()
    document = translate(outputs, _atmospheric())
    rebuilt = document.to_mapping()
    rebuilt["claimable"][0]["rendered"] += ", holding 82% of the time"
    with pytest.raises(InvalidParameterError) as excinfo:
        assert_no_bare_confidence(TranslatedFinding.from_mapping(rebuilt))
    assert "R9" in str(excinfo.value)


def test_mutation_a_causal_verb_reaching_a_rendering_is_caught():
    """Mutation 7: a causal word entering through an evidence label rather than a glossary."""
    outputs = _outputs()
    document = translate(outputs, _atmospheric())
    rebuilt = document.to_mapping()
    rebuilt["claimable"][0]["rendered"] += " which explains the outcome"
    with pytest.raises(InvalidParameterError) as excinfo:
        assert_no_causal_language(TranslatedFinding.from_mapping(rebuilt))
    assert "explains" in str(excinfo.value)


def test_the_causal_guard_does_not_refuse_the_programmes_own_refusals():
    """The counterpart to the mutation above, and the reason the two halves differ.

    Several entitlements name a causal term precisely in order to deny it — *"never about
    mechanism"*. A guard that scanned them would refuse the sentence whose whole job is to hold
    the line, so this asserts the distinction is real rather than accidental.
    """
    document = translate(_outputs("demonstrated_predictive_utility"), _atmospheric())
    assert "mechanism" in document.claim_text()
    assert any(word in unit.licences for unit in document.units for word in OUTSIDE_THE_LADDER)
    assert_no_causal_language(document)
