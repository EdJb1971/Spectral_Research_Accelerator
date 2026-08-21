"""`WaveletBank` and the two Phase 4B pipeline actions (roadmap T4B.2, T4B.3).

T4B.2's claim is that a bank needs **no engine changes**, because its entries are ordinary
parameter-matrix entries. That claim is tested by expanding a bank through the engine's *own*
`expand_parameter_matrix` rather than through a reimplementation - if the two could differ, the
claim would be false and nothing else here would notice.

The rest of this file covers the failures that would otherwise be silent:

*   **A level dimension with one sequence.** Decomposing the same frames once per pressure
    level and labelling the results 850 and 500 hPa produces identical coefficients under
    different labels. Every cross-level statistic downstream would then measure that fiction.
*   **A dereferenced payload arriving where a reference was meant.** `resolve_value` turns a
    literal `artifact://...` into a bare array, which has no time axis and no grid.
*   **Lineage size.** A bank of eight combinations is eight coefficient fields; if their
    payloads reached the lineage row, the store would have been pointless.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.artifact_store import store as artifact_store
from src.core.errors import InvalidParameterError, UnknownNameError
from src.experiment_engine import actions
from src.experiment_engine.engine import DeclarativeExperimentEngine
from src.physical_core.field import PhysicalField
from src.physical_core.sequence import FieldSequence
from src.transform_engine.bank import (MAX_COMBINATIONS, WaveletBank, bank_families,
                                       select_orientations)
from src.transform_engine.coefficient_field import decompose_sequence

CPU = torch.device("cpu")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    made = artifact_store.ArtifactStore(str(tmp_path / "artifacts"))
    monkeypatch.setattr(artifact_store, "_DEFAULT", made)
    return made


def make_sequence(n_frames: int = 4, size: int = 32, seed: int = 0) -> FieldSequence:
    generator = torch.Generator().manual_seed(seed)
    fields = [PhysicalField(torch.randn(size, size, generator=generator, dtype=torch.float64))
              for _ in range(n_frames)]
    return FieldSequence(fields, np.arange(n_frames) * 21600.0)


# ======================================================== T4B.2: the bank is a matrix

def test_a_bank_expands_through_the_engines_own_function():
    """The whole T4B.2 claim in one assertion.

    `combinations()` calls `expand_parameter_matrix`, so a bank cannot expand differently from
    an ordinary parameter matrix - because it *is* one. Reimplementing the product here would
    have tested the reimplementation instead.
    """
    bank = WaveletBank.from_config({"families": ["swt", "dtcwt"], "scales": [2, 3]})
    through_bank = bank.combinations()
    through_engine = DeclarativeExperimentEngine.expand_parameter_matrix(
        bank.to_parameter_matrix())

    assert through_bank == through_engine
    assert len(through_bank) == 4
    assert {c["wavelet_family"] for c in through_bank} == {"swt", "dtcwt"}
    assert {c["levels"] for c in through_bank} == {2, 3}


def test_the_parameter_matrix_is_a_plain_dict_of_lists():
    """Nothing here is a new type the engine would have to learn."""
    matrix = WaveletBank.from_config(
        {"families": ["swt"], "scales": [1, 2], "levels_hpa": [850, 500]}
    ).to_parameter_matrix()

    assert matrix == {"wavelet_family": ["swt"], "levels": [1, 2],
                      "level_hpa": [850.0, 500.0]}
    for value in matrix.values():
        assert isinstance(value, list)


def test_the_thousand_combination_guard_is_kept():
    """Each combination is a full decomposition of every frame: a compute budget, not a
    formality."""
    with pytest.raises(InvalidParameterError) as excinfo:
        WaveletBank(families=("swt",), scales=tuple(range(1, MAX_COMBINATIONS + 2)))
    assert "combinations" in str(excinfo.value)


def test_the_banks_ceiling_matches_the_engines():
    """Two numbers that must agree. The bank duplicates the engine's limit to keep the import
    graph acyclic, so the duplication is checked here rather than trusted."""
    matrix = {"a": list(range(MAX_COMBINATIONS + 1))}
    with pytest.raises(ValueError) as excinfo:
        DeclarativeExperimentEngine.expand_parameter_matrix(matrix)
    assert str(MAX_COMBINATIONS) in str(excinfo.value)


def test_orientations_are_a_selector_not_a_sweep_axis():
    """A wavelet transform computes every orientation in one pass.

    Sweeping them would run the identical decomposition six times and discard five sixths of
    each result - the same coefficients, six times the cost.
    """
    bank = WaveletBank.from_config(
        {"families": ["swt"], "scales": [2], "orientations": ["LH", "HH"]})

    assert "orientations" not in bank.to_parameter_matrix()
    assert len(bank.combinations()) == 1
    assert bank.combinations()[0]["orientations"] == ["LH", "HH"]
    assert "selector applied within each run" in bank.summary()["orientation_role"]


def test_selecting_orientations_narrows_the_view_after_the_transform():
    field = decompose_sequence(make_sequence(n_frames=2), "swt", {"levels": 2})
    narrowed = select_orientations(field, ["LH"])
    assert narrowed.orientations == ["LH"]
    assert narrowed.shape[2] == 1
    assert select_orientations(field, None) is field


# ======================================================== T4B.2: refusals

def test_families_come_from_the_registry():
    """A new family joins a bank by being registered, not by anyone editing a sweep."""
    assert set(bank_families()) == {"swt", "dtcwt"}
    from src.transform_engine.registry import TRANSFORMS

    for name in bank_families():
        assert "wavelet_bank" in (TRANSFORMS.entry(name).tags or [])


def test_a_non_bank_transform_is_refused_at_configuration_time():
    """`fft` has no (scale, orientation) factorisation. Refusing at config time rather than
    at run time means a 40-minute sweep does not die on its last combination."""
    with pytest.raises(UnknownNameError) as excinfo:
        WaveletBank.from_config({"families": ["fft"], "scales": [2]})
    assert "swt" in str(excinfo.value)


def test_an_unrecognised_key_is_refused_rather_than_ignored():
    """A permissive parser is how a sweep ends up not sweeping what its author wrote."""
    with pytest.raises(InvalidParameterError) as excinfo:
        WaveletBank.from_config({"families": ["swt"], "scale": [2]})
    assert "scale" in str(excinfo.value)


def test_a_scale_that_is_not_a_level_count_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        WaveletBank.from_config({"families": ["swt"], "scales": [2.5]})
    assert "no half level" in str(excinfo.value)
    # The bug this replaced: int(2.5) is 2, so the bank ran two levels and reported two.
    assert WaveletBank.from_config({"families": ["swt"], "scales": [2.0]}).scales == (2,)


def test_a_pressure_in_the_wrong_units_is_refused():
    """5000 is not a pressure in hPa; it is almost always metres or Pa by mistake."""
    with pytest.raises(InvalidParameterError) as excinfo:
        WaveletBank.from_config({"families": ["swt"], "scales": [2], "levels_hpa": [50000]})
    assert "hPa" in str(excinfo.value)


def test_an_empty_family_list_is_refused():
    with pytest.raises(InvalidParameterError):
        WaveletBank.from_config({"families": [], "scales": [2]})


def test_a_scalar_is_accepted_as_a_one_element_list():
    """Configs are written by hand; `families: swt` is the obvious thing to type."""
    bank = WaveletBank.from_config({"families": "swt", "scales": 3})
    assert bank.families == ("swt",) and bank.scales == (3,)


# ======================================================== T4B.3: decompose_bank

def test_decompose_bank_stores_references_not_payloads(store):
    sequence = make_sequence(n_frames=5)
    handle = store.put(sequence, name="seq")
    args = {"sequence": handle.ref,
            "wavelet_bank": {"families": ["swt", "dtcwt"], "scales": [2, 3]}}

    result = actions.execute("decompose_bank", args, CPU)

    assert result["n_combinations"] == 4
    assert len(result["refs"]) == 4
    for ref in result["refs"]:
        assert artifact_store.is_ref(ref)
    assert store.stats()["n_artifacts"] >= 5  # the sequence plus four banks


def test_the_bank_lineage_row_stays_small(store):
    """Four coefficient fields, one database row."""
    sequence = make_sequence(n_frames=5, size=64)
    handle = store.put(sequence, name="seq")
    args = {"sequence": handle.ref,
            "wavelet_bank": {"families": ["swt", "dtcwt"], "scales": [2, 3]}}

    result = actions.execute("decompose_bank", args, CPU)
    node = actions.summarise("decompose_bank", result, args)
    encoded = len(json.dumps(node, default=str).encode())

    assert encoded < 4096, "lineage row is %d bytes" % encoded
    # The payload this row replaces: 5 frames x 64x64 x 45 (scale, orientation) bands,
    # complex for the two DTCWT combinations. The exact count is asserted rather than a round
    # number, so a change to what the bank computes shows up here instead of passing quietly.
    total_elements = sum(int(np.prod(r["summary"]["shape"])) for r in result["results"])
    assert total_elements == 921_600
    assert total_elements * 8 > 7_000_000, "at least 7 MB of coefficients, in a 4 KB row"


def test_a_stored_bank_reloads_with_its_labels(store):
    """An artifact carries its own axes: a `(T, S, O, Y, X)` array is not a CoefficientField."""
    sequence = make_sequence(n_frames=3)
    handle = store.put(sequence, name="seq")
    result = actions.execute(
        "decompose_bank",
        {"sequence": handle.ref, "wavelet_bank": {"families": ["dtcwt"], "scales": [2]}}, CPU)

    restored = store.load_coefficient_field(result["refs"][0])
    assert restored.wavelet_family == "dtcwt"
    assert restored.scales == [1, 2]
    assert len(restored.orientations) == 6
    assert restored.is_complex
    assert restored.has_native() is False, (
        "native coefficients are not stored; the restored field must say it cannot invert "
        "rather than inverting the aligned view")


def test_a_dereferenced_array_is_refused_with_the_fix(store):
    """`resolve_value` turns a literal ref into an array, and an array is not a sequence."""
    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("decompose_bank",
                        {"sequence": np.zeros((3, 8, 8)),
                         "wavelet_bank": {"families": ["swt"], "scales": [1]}}, CPU)
    message = str(excinfo.value)
    assert "no time coordinate" in message
    assert "{step.sequence_ref}" in message


# ======================================================== T4B.4: levels in a bank

def test_a_vertical_bank_requires_one_sequence_per_level(store):
    """The silent failure this prevents.

    Without it, `levels_hpa: [850, 500]` with a single sequence decomposes the same frames
    twice and labels the results with two pressures. The arrays are identical, so every
    cross-level statistic computed afterwards measures a vertical structure that was invented
    by the labelling.
    """
    sequence = make_sequence(n_frames=3)
    handle = store.put(sequence, name="seq")

    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("decompose_bank", {
            "sequence": handle.ref,
            "wavelet_bank": {"families": ["swt"], "scales": [2], "levels_hpa": [850, 500]},
        }, CPU)
    message = str(excinfo.value)
    assert "identical coefficients with different pressures" in message
    assert "sequences" in message


def test_a_vertical_bank_decomposes_each_level_separately(store):
    sequence_850 = make_sequence(n_frames=3, seed=1)
    sequence_500 = make_sequence(n_frames=3, seed=2)
    refs = {"850": store.put(sequence_850, name="l850").ref,
            "500": store.put(sequence_500, name="l500").ref}

    result = actions.execute("decompose_bank", {
        "sequences": refs,
        "wavelet_bank": {"families": ["swt"], "scales": [2], "levels_hpa": [850, 500]},
    }, CPU)

    assert result["n_combinations"] == 2
    assert {r["level_hpa"] for r in result["results"]} == {850.0, 500.0}
    assert len(set(result["refs"])) == 2, (
        "two levels that produced the same content-addressed ref would mean the same data "
        "was decomposed twice under different labels")
    for record in result["results"]:
        assert record["summary"]["level"] == record["level_hpa"]


def test_a_missing_level_is_named(store):
    refs = {"850": store.put(make_sequence(n_frames=3, seed=1), name="l850").ref}
    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("decompose_bank", {
            "sequences": refs,
            "wavelet_bank": {"families": ["swt"], "scales": [2], "levels_hpa": [850, 500]},
        }, CPU)
    assert "500" in str(excinfo.value)


# ======================================================== T4B.3: extract_scale_signature

def test_extract_scale_signature_reports_energy_per_time_and_scale(store):
    sequence = make_sequence(n_frames=5)
    field = decompose_sequence(sequence, "swt", {"levels": 3})
    handle = store.put(field, name="cf")

    result = actions.execute("extract_scale_signature", {"coefficients": handle.ref}, CPU)

    assert len(result["energy_fraction_per_scale"]) == 5
    assert len(result["energy_fraction_per_scale"][0]) == 3
    for row in result["energy_fraction_per_scale"]:
        assert sum(row) == pytest.approx(1.0)
    assert len(result["dominant_scale_per_time"]) == 5


def test_the_signature_states_its_own_scope(store):
    """In Phase 4B this action carried the energy half of rule R3 and said so; T4C.1 finished
    it. The assertion moved with the code rather than being deleted, so the result still has
    to state what it covers at the point a reader actually looks.
    """
    field = decompose_sequence(make_sequence(n_frames=2), "swt", {"levels": 2})
    result = actions.execute(
        "extract_scale_signature", {"coefficients": store.put(field, name="cf").ref}, CPU)
    assert "rule R3 in full" in result["scope"]
    assert "never primary" in result["scope"], "the threshold count must carry its caveat"
    for measure in ("participation_ratio", "gini", "threshold_fraction", "threshold_values"):
        assert measure in result


def test_the_signature_accepts_a_field_directly(store):
    field = decompose_sequence(make_sequence(n_frames=2), "swt", {"levels": 2})
    result = actions.execute("extract_scale_signature", {"coefficients": field}, CPU)
    assert result["wavelet_family"] == "swt"


def test_the_signature_refuses_an_unlabelled_array(store):
    with pytest.raises(InvalidParameterError) as excinfo:
        actions.execute("extract_scale_signature",
                        {"coefficients": np.zeros((2, 2, 3, 8, 8))}, CPU)
    assert "lost its scale and orientation labels" in str(excinfo.value)


def test_the_signature_lineage_row_is_small(store):
    field = decompose_sequence(make_sequence(n_frames=10, size=64), "dtcwt", {"levels": 3})
    args = {"coefficients": store.put(field, name="cf").ref}
    result = actions.execute("extract_scale_signature", args, CPU)
    node = actions.summarise("extract_scale_signature", result, args)
    assert len(json.dumps(node, default=str).encode()) < 4096


# ======================================================== registration

def test_both_actions_are_registered_with_node_types():
    assert actions.node_type_for("decompose_bank") == "coefficients"
    assert actions.node_type_for("extract_scale_signature") == "metrics"


def test_both_actions_have_summarisers():
    """`ActionSpec` is frozen, so summarisers are attached by re-registering. A missed
    re-registration leaves the default empty summariser and the lineage row silently loses
    everything but the action name."""
    for name in ("decompose_bank", "extract_scale_signature"):
        spec = actions.ACTIONS.get(name)
        assert spec.summarise({}, {}) != {}, "%s has no summariser attached" % name
