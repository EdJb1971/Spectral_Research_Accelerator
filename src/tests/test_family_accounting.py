"""Phase G3, TG3.1: the declared family is priced before anything is allowed to enumerate.

The acceptance criterion is that the enumerator reproduces T4C.6's known family - **36 tests
and 3,005 surrogates** - from the frozen campaign JSON on disk, not from a literal typed into
a test. Around it sit the properties that make the number trustworthy rather than merely
correct once:

*   every registered combinator's cheap count agrees with its own enumeration, because the
    count is what prices an unaffordable family and the enumeration is what a sweep is checked
    against, and two implementations that disagree would produce a family size nobody could
    falsify;
*   the label set is identical to the labels a **real sweep** emits, so 36 agreeing with 36 is
    not two different families agreeing by arithmetic accident;
*   an unaffordable specification is **refused**, and the refusal computes both R18 remedies
    rather than naming them; and
*   the admissibility audit never moves the correction unit, which is option A of the design
    decision this slice turned on.
"""

import json
import os

import numpy as np
import pytest

from src.analysis_engine import cross_scale as cs
from src.core.channel_series import ChannelSeries
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.family import (
    FAMILY_COMBINATORS,
    MAX_ENUMERATED,
    FamilyUnaffordableError,
    SearchAxis,
    SearchSpecification,
    SearchTerm,
    max_affordable_family,
)
from src.core.lag_policy import bind as bind_lag_policy
from src.core.registry import restore, snapshot
from src.statistics.multiple_comparisons import required_surrogates

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO_ROOT, "campaigns",
                        "t4c6_nz_era5_temperature_850_v1.json")


def _t4c6_protocol():
    """The frozen T4C.6 protocol, read off disk rather than retyped here."""
    with open(CAMPAIGN, encoding="utf-8") as handle:
        document = json.load(handle)
    mapping = document["campaign"]["gate_plan"]["protocol"]
    return cs.GateProtocol.from_mapping(mapping)


def _spec(scales=3, lags=(3, 4, 5, 6, 7, 8), n_surrogates=4999,
          correction="benjamini_yekutieli", alpha=0.05):
    return SearchSpecification(
        terms=(SearchTerm("ordered_pairs",
                          (SearchAxis("scale", tuple(range(1, scales + 1))),)),
               SearchTerm("product", (SearchAxis("lag_frames", tuple(lags)),))),
        n_surrogates=n_surrogates, alpha=alpha, correction=correction,
        label_format="{0}->{1}@{2}", study_id="test")


# ------------------------------------------------------------------------- the acceptance


def test_the_enumerator_reproduces_t4c6s_known_family_from_the_frozen_campaign_json():
    """36 tests and 3,005 surrogates, from the campaign file rather than from a literal."""
    protocol = _t4c6_protocol()
    account = protocol.search_specification().declare()

    assert account.family_size == 36
    assert account.surrogates_required == 3005
    assert account.affordable is True
    assert account.n_surrogates == 4999
    assert account.correction == "benjamini_yekutieli"


def test_the_specification_agrees_with_the_formula_it_is_replacing():
    """A second piece of arithmetic is only useful if it can disagree - here it must not."""
    protocol = _t4c6_protocol()
    assert protocol.search_specification().family_size == protocol.family_size


def test_the_required_surrogate_count_is_the_one_the_protocol_already_enforced():
    protocol = _t4c6_protocol()
    design = protocol.validate()
    account = protocol.search_specification().account()
    assert account.surrogates_required == design["power"]["surrogates_required"]
    assert account.family_size == design["family_size"]


def test_the_declared_labels_are_the_labels_a_real_sweep_emits():
    """The strong form of the acceptance: not 36 == 36, but the same 36 members.

    A count can be right for the wrong reason. Here the specification's labels are compared
    against the labels `cross_scale_dependency` actually produces on a three-channel record
    over the frozen lag family, in order - so an enumerator that got the count right by
    double-counting one pair and dropping another would fail.
    """
    lags = (3, 4, 5, 6, 7, 8)
    rng = np.random.default_rng(20260824)
    n_frames = 400
    series = ChannelSeries(
        channels=[1, 2, 3],
        times_seconds=np.arange(n_frames, dtype=np.float64) * 3600.0,
        measures={"energy_density": rng.standard_normal((n_frames, 3))},
        support_parent_px=[1.0, 1.0, 1.0],
        provenance={"origin": "TG3.1 label-agreement fixture"},
    )
    result = cs.cross_scale_dependency(
        series, lags=lags, cadence_seconds=3600.0, measure="energy_density",
        # `none` so that no lag is excluded and the sweep enumerates its whole family: the
        # floor is not what is under test here, the label set is.
        n_surrogates=9, lag_floor=bind_lag_policy("none"),
    )
    assert result["n_tests"] == 36
    swept = [test["label"] for test in result["results"]]

    declared = _spec(scales=3, lags=lags).labels()
    assert sorted(declared) == sorted(swept)
    assert list(declared) == swept


# --------------------------------------------------------------- count vs. enumeration


@pytest.mark.parametrize("combinator", ["product", "ordered_pairs", "unordered_pairs"])
@pytest.mark.parametrize("n_values", [2, 3, 5, 8])
def test_every_combinator_counts_exactly_what_it_enumerates(combinator, n_values):
    """The property the whole module rests on, asserted rather than assumed.

    `count` prices a family that may be far too large to build; `enumerate` builds the one
    that is not. If they can drift apart then a refusal and a receipt describe different
    searches, and nothing downstream could detect it.
    """
    term = SearchTerm(combinator, (SearchAxis("a", tuple(range(n_values))),))
    members = term.members()
    assert term.size == len(members)
    assert len(set(members)) == len(members)
    assert all(len(member) == term.n_components for member in members)


def test_ordered_and_unordered_pairs_differ_by_exactly_the_factor_they_should():
    ordered = SearchTerm("ordered_pairs", (SearchAxis("a", (1, 2, 3, 4)),))
    unordered = SearchTerm("unordered_pairs", (SearchAxis("a", (1, 2, 3, 4)),))
    assert ordered.size == 12 and unordered.size == 6
    assert not any(a == b for a, b in ordered.members())


def test_the_product_of_several_axes_is_the_product_of_their_lengths():
    term = SearchTerm("product", (SearchAxis("scale", (1, 2, 3)),
                                  SearchAxis("orientation", (15.0, 45.0, 75.0, 105.0)),
                                  SearchAxis("lag", (1, 2))))
    assert term.size == 24 == len(term.members())
    assert term.n_components == 3


def test_a_term_that_yields_no_members_is_refused():
    """A found defect, not a hypothetical: emptying the search was scoring as a remedy.

    `ordered_pairs` over a single value has no members, so the family size is zero and the
    narrowing search reported "scale from 5 to 1 values" as sufficient - a suggestion that
    would have produced a pass with nothing in it and an empty result to read.
    """
    with pytest.raises(InvalidParameterError):
        SearchTerm("ordered_pairs", (SearchAxis("scale", (1,)),))
    with pytest.raises(InvalidParameterError):
        SearchTerm("unordered_pairs", (SearchAxis("scale", (1,)),))


def test_a_pair_combinator_refuses_two_axes_rather_than_guessing_which_one():
    with pytest.raises(InvalidParameterError):
        SearchTerm("ordered_pairs", (SearchAxis("a", (1, 2)), SearchAxis("b", (3, 4))))


def test_an_unregistered_combinator_names_the_registered_ones():
    with pytest.raises(UnknownNameError):
        SearchTerm("cartesian", (SearchAxis("a", (1, 2)),))


def test_a_new_combinator_registers_without_editing_src():
    """Standard E1: the next search shape is a registration, not an edit to this module.

    TG3.3 will need constellations of `k` features, which is neither a product nor a pair.
    Registering that shape from outside `src/` is the acceptance that the registry is real.
    """
    import itertools

    from src.core.family import Combinator

    state = snapshot(FAMILY_COMBINATORS)
    try:
        FAMILY_COMBINATORS.add(
            "triples",
            Combinator(
                count=lambda axes: (len(axes[0]) * (len(axes[0]) - 1) * (len(axes[0]) - 2)
                                    // 6),
                enumerate=lambda axes: itertools.combinations(axes[0].values, 3),
                n_axes=1,
                components=lambda axes: 3,
            ),
            description="Unordered triples, the shape a three-feature constellation has.",
        )
        term = SearchTerm("triples", (SearchAxis("feature", tuple(range(6))),))
        assert term.size == 20 == len(term.members())
        spec = SearchSpecification(terms=(term,), n_surrogates=4999)
        assert spec.declare().family_size == 20
    finally:
        restore(FAMILY_COMBINATORS, state)


# ------------------------------------------------------------------------------ refusal


def test_an_unaffordable_family_is_refused_rather_than_run():
    """R18's whole point: the pass is refused before it runs, not read afterwards."""
    spec = _spec(scales=5, lags=tuple(range(1, 9)), n_surrogates=499)
    account = spec.account()
    assert account.family_size == 160
    assert account.affordable is False

    with pytest.raises(FamilyUnaffordableError) as exc:
        spec.declare()
    assert exc.value.context["family_size"] == 160
    assert exc.value.context["surrogates_required"] == required_surrogates(
        160, 0.05, "benjamini_yekutieli")


def test_the_refusal_computes_both_r18_remedies_instead_of_naming_them():
    """Reduce the tests *by how much* - a remedy without a number is an invitation."""
    spec = _spec(scales=5, lags=tuple(range(1, 9)), n_surrogates=4999)
    with pytest.raises(FamilyUnaffordableError) as exc:
        spec.declare()
    remedies = exc.value.context["remedies"]
    narrowing = remedies["preregistered_narrowing"]

    ceiling = max_affordable_family(4999, 0.05, "benjamini_yekutieli")
    assert narrowing["max_affordable_family"] == ceiling
    assert narrowing["members_to_remove"] == 160 - ceiling
    assert required_surrogates(ceiling, 0.05, "benjamini_yekutieli") <= 4999
    assert required_surrogates(ceiling + 1, 0.05, "benjamini_yekutieli") > 4999

    # Every declared axis is priced, and the summary is arithmetic rather than advice.
    by_axis = {record["axis"]: record for record in narrowing["by_axis"]}
    assert set(by_axis) == {"scale", "lag_frames"}
    for record in by_axis.values():
        if record["achievable_alone"]:
            assert record["largest_affordable_values"] < record["declared_values"]
    assert remedies["generate_confirm_split"]["slice"] == "TG3.2"
    assert "TG3.2" in str(exc.value)


def test_a_narrowing_the_refusal_says_is_enough_actually_is():
    """The remedy is checked by taking it, not by reading it.

    It is also checked to be *tight*: one more value than the refusal permits must still be
    refused. A remedy that under-reports is a slow way of narrowing until the answer appears.
    """
    lags = tuple(range(1, 9))
    spec = _spec(scales=5, lags=lags, n_surrogates=4999)
    with pytest.raises(FamilyUnaffordableError) as exc:
        spec.declare()
    by_axis = {record["axis"]: record
               for record in exc.value.context["remedies"][
                   "preregistered_narrowing"]["by_axis"]}
    lag_record = by_axis["lag_frames"]
    assert lag_record["achievable_alone"] is True
    kept = lag_record["largest_affordable_values"]

    assert _spec(scales=5, lags=lags[:kept], n_surrogates=4999).declare().affordable is True
    with pytest.raises(FamilyUnaffordableError):
        _spec(scales=5, lags=lags[:kept + 1], n_surrogates=4999).declare()


def test_when_no_single_axis_can_be_narrowed_far_enough_the_refusal_says_so():
    """The honest answer is sometimes that this search shape is not available at all.

    At 499 surrogates the largest affordable family is 8 members. A five-scale sweep has 20
    ordered pairs before a single lag is chosen, so no reduction of the lag axis alone can
    reach it - and a remedy that claimed otherwise would send the reader round a loop.
    """
    spec = _spec(scales=5, lags=tuple(range(1, 9)), n_surrogates=499)
    with pytest.raises(FamilyUnaffordableError) as exc:
        spec.declare()
    narrowing = exc.value.context["remedies"]["preregistered_narrowing"]
    assert narrowing["max_affordable_family"] == 8
    assert not any(record["achievable_alone"] for record in narrowing["by_axis"])
    assert "more than one axis" in narrowing["by_axis_summary"]


def test_max_affordable_family_is_the_boundary_it_claims_to_be():
    for n_surrogates in (99, 199, 499, 999, 4999):
        for method in ("benjamini_yekutieli", "benjamini_hochberg", "bonferroni"):
            ceiling = max_affordable_family(n_surrogates, 0.05, method)
            assert required_surrogates(ceiling, 0.05, method) <= n_surrogates
            assert required_surrogates(ceiling + 1, 0.05, method) > n_surrogates


def test_required_surrogates_is_non_decreasing_in_family_size():
    """The monotonicity `max_affordable_family` bisects on, asserted rather than trusted."""
    for method in ("benjamini_yekutieli", "benjamini_hochberg", "bonferroni", "holm"):
        values = [required_surrogates(m, 0.05, method) for m in range(1, 400)]
        assert all(b >= a for a, b in zip(values, values[1:]))


def test_the_dependence_assumption_travels_with_the_price():
    """A q-value cannot be quoted without its assumption, and neither can its cost."""
    by = _spec().account()
    bh = _spec(correction="benjamini_hochberg").account()
    assert by.dependence_assumption != bh.dependence_assumption
    assert bh.surrogates_required < by.surrogates_required


def test_the_constellation_sweep_r18_warns_about_is_priced_not_asserted():
    """The programme's boundary is a measurement this module makes, not a claim in prose.

    Roadmap Phase G3 says that if TG3.1 proves every scientifically interesting family
    unaffordable, that is the real boundary and is written up as such. These are the numbers
    that statement has to be made from.
    """
    spec = SearchSpecification(
        terms=(SearchTerm("ordered_pairs", (SearchAxis("scale", tuple(range(1, 6))),)),
               SearchTerm("product", (SearchAxis("orientation", (15.0, 45.0, 75.0, 105.0)),
                                      SearchAxis("lag", tuple(range(1, 9))),
                                      SearchAxis("representation",
                                                 ("raw", "swt", "dtcwt", "dwt", "fft",
                                                  "dct", "hybrid")),))),
        n_surrogates=4999)
    account = spec.account()
    assert account.family_size == 20 * 4 * 8 * 7 == 4480
    assert account.affordable is False
    assert account.surrogates_required > 800_000


# ------------------------------------------------------------------------- admissibility


def test_the_admissibility_audit_never_moves_the_correction_unit():
    """Option A, asserted: a screened family is a family chosen after looking.

    The lag floor genuinely makes some members untestable before any data exists. Reporting
    how many is useful; re-pricing the family at the survivors is the exact move R18 forbids,
    so `family_size` and `correction_unit` must both stay at the declared 36.
    """
    spec = _spec()
    audit = spec.audit_admissibility(
        lambda member: int(member[2]) >= 5,
        basis="advective support floor of 5 frames at the frozen grid and cadence")
    assert audit["declared_family_size"] == 36
    assert audit["correction_unit"] == 36
    assert audit["n_admissible"] == 24
    assert audit["n_inadmissible"] == 12
    assert spec.account().family_size == 36
    assert spec.account().surrogates_required == 3005
    assert all(label.endswith("@3") or label.endswith("@4")
               for label in audit["inadmissible_labels"])


def test_an_audit_that_empties_the_family_is_refused():
    """A pass with no possible outcome must not run and report an empty result."""
    with pytest.raises(InvalidParameterError):
        _spec().audit_admissibility(lambda member: False,
                                    basis="a floor above every declared lag")


def test_an_exclusion_with_no_stated_basis_is_refused():
    with pytest.raises(InvalidParameterError):
        _spec().audit_admissibility(lambda member: True, basis="   ")


# -------------------------------------------------------------------- declaration hygiene


def test_the_fingerprint_moves_with_every_scientific_input():
    base = _spec()
    assert base.fingerprint() == _spec().fingerprint()
    assert base.fingerprint() != _spec(lags=(3, 4, 5, 6, 7)).fingerprint()
    assert base.fingerprint() != _spec(scales=4).fingerprint()
    assert base.fingerprint() != _spec(n_surrogates=3005).fingerprint()
    assert base.fingerprint() != _spec(correction="benjamini_hochberg").fingerprint()
    assert base.fingerprint() != _spec(alpha=0.01).fingerprint()
    assert base.account().specification_sha256 == base.fingerprint()


def test_a_duplicate_axis_value_is_refused_rather_than_collapsed():
    with pytest.raises(InvalidParameterError):
        SearchAxis("lag", (3, 4, 4, 5))


def test_an_empty_axis_is_refused():
    with pytest.raises(InvalidParameterError):
        SearchAxis("lag", ())


def test_two_terms_cannot_share_an_axis_name():
    with pytest.raises(InvalidParameterError):
        SearchSpecification(
            terms=(SearchTerm("product", (SearchAxis("lag", (1, 2)),)),
                   SearchTerm("product", (SearchAxis("lag", (3, 4)),))),
            n_surrogates=999)


def test_a_label_format_that_does_not_fit_the_components_is_refused():
    with pytest.raises(InvalidParameterError):
        SearchSpecification(
            terms=(SearchTerm("product", (SearchAxis("lag", (1, 2)),)),),
            n_surrogates=999, label_format="{0}->{1}@{2}")


def test_an_unknown_correction_is_refused_at_declaration():
    with pytest.raises(InvalidParameterError):
        SearchSpecification(terms=(SearchTerm("product", (SearchAxis("lag", (1, 2)),)),),
                            n_surrogates=999, correction="fdr_by")


def test_a_family_too_large_to_list_is_still_priced_exactly():
    """Counting is exact at any size; materialising is the part that has a limit."""
    spec = SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("a", tuple(range(1000))),
                                      SearchAxis("b", tuple(range(1000))),)),),
        n_surrogates=4999)
    assert spec.family_size == 1_000_000 > MAX_ENUMERATED
    assert spec.account().family_size == 1_000_000
    with pytest.raises(InvalidParameterError):
        spec.enumerate_family()


def test_the_account_says_what_it_is_not_evidence_of():
    described = _spec().account().describe()
    assert "not a result" in described["claim_boundary"]
    assert described["family_size"] == 36
    assert described["surrogates_required"] == 3005
    json.dumps(described)
