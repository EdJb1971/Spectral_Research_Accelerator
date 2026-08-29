"""TG1.5: the vertical coordinate is declared, and pressure is one instance.

The last Phase G1 task. `LevelBank` was keyed on pressure in hectopascals, `ScaleSignature`
carried a field called `level_hpa`, and `CoefficientField.level` carried a bare number with no
units anywhere. The unit lived in an attribute name in one place, a docstring in another, and
nowhere at all in the third.

The part that was not cosmetic is `vertical_offsets`. It decided direction with

    "direction": "upward" if upper < lower else "downward"

which is right for pressure, right for depth, and exactly backwards for height - at the one
place that reports the direction of a vertical precursor relationship, which is the sign that
distinguishes an upper-level trough from a surface one. Nothing shipped wrong, because no
height axis exists in the tree; the point is that nothing *could* have gone right either, and
the test below is the one that could not have been written before.

Three groups: the coordinate registry, the declaration travelling with the number, and the
atmospheric arithmetic unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis_engine.scale_signature import scale_signature
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.level_axis import (LEVEL_COORDINATES, LevelCoordinate, PRESSURE_HPA,
                                 coordinate_for, coordinate_names, require_pressure, units_of)
from src.core.registry import restore, snapshot
from src.core.sample import StructuredSample
from src.physical_core.field import PhysicalField
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import (CoefficientField, LevelBank,
                                                    decompose_levels, decompose_sequence)


def make_sequence(n_frames: int = 3, size: int = 32, seed: int = 0) -> FieldSequence:
    generator = torch.Generator().manual_seed(seed)
    fields = [PhysicalField(torch.randn(size, size, generator=generator, dtype=torch.float64))
              for _ in range(n_frames)]
    return FieldSequence(fields, np.arange(n_frames) * 21600.0)


def make_banks(levels=(850.0, 500.0)):
    return {level: decompose_sequence(make_sequence(), "swt", {"levels": 2},
                                      keep_native=False)
            for level in levels}


# ==================================================== 1. the coordinate is registered


def test_the_three_registered_coordinates_declare_their_own_sense_of_up():
    assert set(coordinate_names()) == {"pressure_hpa", "height_m", "depth_m"}
    assert coordinate_for(PRESSURE_HPA).units == "hPa"
    assert not coordinate_for(PRESSURE_HPA).increases_upward
    assert coordinate_for("height_m").increases_upward
    # Depth shares pressure's sense for an unrelated reason, which is exactly why the rule
    # belongs to the coordinate and not to a hardcoded comparison.
    assert not coordinate_for("depth_m").increases_upward


def test_the_direction_of_an_offset_is_read_from_the_coordinate_not_the_sign():
    """The defect the hardcoded rule would have produced, stated as a difference.

    Going from 850 to 500 is *upward* in pressure and the identical arithmetic is *downward*
    in height. `"upward" if upper < lower` gets the second one backwards, and there is nothing
    downstream that could detect it.
    """
    pressure, height = coordinate_for(PRESSURE_HPA), coordinate_for("height_m")

    assert pressure.direction(850.0, 500.0) == "upward"
    assert height.direction(850.0, 500.0) == "downward"
    assert pressure.direction(500.0, 850.0) == "downward"
    assert height.direction(500.0, 850.0) == "upward"
    assert pressure.direction(500.0, 500.0) == "same"


def test_an_ascending_stack_runs_a_direction_the_coordinate_names():
    """Ordering and direction are separate, and both are reported."""
    assert coordinate_for(PRESSURE_HPA).ascending_runs == "downward"
    assert coordinate_for("height_m").ascending_runs == "upward"
    assert coordinate_for(PRESSURE_HPA).ordered([850.0, 500.0, 700.0]) == (500.0, 700.0, 850.0)


def test_a_coordinate_without_units_is_refused():
    with pytest.raises(InvalidParameterError, match="non-empty units"):
        LevelCoordinate("  ", increases_upward=True)


def test_an_unregistered_coordinate_names_the_alternatives():
    with pytest.raises(UnknownNameError) as excinfo:
        coordinate_for("sigma_level")
    assert "pressure_hpa" in str(excinfo.value)


@pytest.fixture
def registered_sigma():
    """A fourth coordinate, registered from outside `src/` (standard E1)."""
    state = snapshot(LEVEL_COORDINATES)
    try:
        LEVEL_COORDINATES.add(
            "sigma", LevelCoordinate("1", increases_upward=False,
                                     description="Terrain-following sigma: 1 at the surface, "
                                                 "0 at the model top."),
            capabilities={"units": "1", "increases_upward": False, "metric": False},
            description="Terrain-following model coordinate.")
        yield
    finally:
        restore(LEVEL_COORDINATES, state)


def test_a_fourth_coordinate_drives_a_whole_bank_without_editing_src(registered_sigma):
    bank = LevelBank(make_banks(levels=(0.85, 0.5)), level_axis="sigma")

    assert bank.level_units == "1"
    assert bank.levels == [0.5, 0.85]
    record = bank.summary()
    assert record["level_axis"] == "sigma"
    assert record["ascending_order_runs"] == "downward"
    upward = [p for p in bank.vertical_offsets()
              if (p["from_level"], p["to_level"]) == (0.85, 0.5)]
    assert upward[0]["direction"] == "upward"


def test_a_height_bank_labels_its_offsets_the_other_way_round():
    """The test that could not be written before TG1.5."""
    bank = LevelBank(make_banks(levels=(200.0, 1500.0)), level_axis="height_m")

    offsets = {(p["from_level"], p["to_level"]): p for p in bank.vertical_offsets()}
    assert offsets[(200.0, 1500.0)]["direction"] == "upward"
    assert offsets[(200.0, 1500.0)]["offset"] == 1300.0
    assert offsets[(1500.0, 200.0)]["direction"] == "downward"
    assert bank.summary()["level_units"] == "m"


def test_an_unregistered_bank_axis_is_refused_before_anything_is_stacked():
    with pytest.raises(UnknownNameError):
        LevelBank(make_banks(), level_axis="furlongs")


# ============================================ 2. the declaration travels with the number


def test_a_decomposed_bank_stamps_each_field_with_its_axis():
    """A `CoefficientField` taken out of a bank still knows what its own number means."""
    bank = decompose_levels({850.0: make_sequence(), 500.0: make_sequence()},
                            family="swt", config={"levels": 2})

    field = bank.at_level(500.0)
    assert field.level == 500.0
    assert field.level_axis == PRESSURE_HPA
    assert field.summary()["level_units"] == "hPa"


def test_a_level_with_no_declared_axis_reports_no_units_rather_than_pressure():
    """The honest report of a bare number, which is a real state of the tree."""
    field = decompose_sequence(make_sequence(), "swt", {"levels": 2})
    field.level = 850.0

    assert field.level_axis is None
    assert field.summary()["level_units"] is None
    assert units_of(None) is None


def test_an_unregistered_axis_on_a_coefficient_field_is_refused_at_construction():
    """Refused where the field is built, not where somebody later asks for its units."""
    reference = decompose_sequence(make_sequence(), "swt", {"levels": 2})

    with pytest.raises(UnknownNameError):
        CoefficientField(reference.data, wavelet_family=reference.wavelet_family,
                         scales=reference.scales, orientations=reference.orientations,
                         times=reference.times, grid=reference.grid,
                         level=1.0, level_axis="furlongs")


def test_the_signature_carries_the_axis_it_was_computed_at():
    bank = decompose_levels({850.0: make_sequence()}, family="swt", config={"levels": 2})
    signature = scale_signature(bank.at_level(850.0), interior=False)

    assert signature.level == 850.0
    assert signature.level_axis == PRESSURE_HPA
    assert signature.level_units == "hPa"
    record = signature.summary()
    assert record["level"] == 850.0
    assert record["level_axis"] == PRESSURE_HPA
    assert record["level_units"] == "hPa"


def test_level_hpa_still_reads_a_pressure_level_and_refuses_anything_else():
    """The atmospheric spelling is kept, and it is now a claim that can be wrong.

    Returning `None` for a height level would read as "this signature has no level" about a
    signature that has one, so it refuses instead.
    """
    bank = decompose_levels({850.0: make_sequence()}, family="swt", config={"levels": 2})
    signature = scale_signature(bank.at_level(850.0), interior=False)
    assert signature.level_hpa == 850.0

    heights = decompose_levels({1500.0: make_sequence()}, family="swt",
                               config={"levels": 2}, level_axis="height_m")
    in_metres = scale_signature(heights.at_level(1500.0), interior=False)
    assert in_metres.level == 1500.0
    with pytest.raises(InvalidParameterError, match="pressure_hpa"):
        in_metres.level_hpa


def test_require_pressure_names_what_the_record_actually_declares():
    with pytest.raises(InvalidParameterError, match="height_m"):
        require_pressure("height_m", "a test")
    with pytest.raises(InvalidParameterError, match="no vertical coordinate at all"):
        require_pressure(None, "a test")
    require_pressure(PRESSURE_HPA, "a test")  # returns, does not raise


def test_a_level_coordinate_is_a_declared_level_role_axis():
    """The literal form of what the roadmap asked for, and it plugs into TG1.4's spine."""
    spec = coordinate_for(PRESSURE_HPA).axis_spec("plev")

    assert isinstance(spec, AxisSpec)
    assert (spec.role, spec.units, spec.ordered) == ("level", "hPa", True)

    sample = StructuredSample(np.zeros((2, 3)),
                              [spec, AxisSpec("station", "category", ordinal=0)],
                              coords={"plev": np.array([500.0, 850.0])})
    assert sample.axes_with_role("level") == ("plev",)


def test_a_streamed_record_that_changes_vertical_coordinate_is_refused():
    """500 on a pressure axis and 500 on a height axis are different frames.

    The streaming signature already refused drift in grid, coordinates, variable, level and
    units. The coordinate joins that list in TG1.5: a record that switched axis partway would
    have passed every other check, because the *number* never moved.
    """
    from src.analysis_engine.scale_signature import stream_scale_signature

    sequence = make_sequence(n_frames=2)
    axes = [PRESSURE_HPA, "height_m"]

    def reader(index):
        frame = sequence.at(index)
        return PhysicalField(frame.data, coords=frame.coords,
                             metadata={"variable": "t", "level": 500.0,
                                       "level_axis": axes[index]})

    with pytest.raises(InvalidParameterError, match="identity drift"):
        stream_scale_signature(reader, np.arange(2) * 21600.0, family="swt",
                                  config={"levels": 2}, interior=False)


# ================================================ 3. the atmospheric arithmetic is unchanged


def test_the_pressure_offsets_are_the_numbers_they_always_were():
    """Bit-identity for the only axis the atmospheric line uses.

    Recomputed inline from the definition rather than copied from the implementation: the
    signed offset is `to - from` in hPa, and 500 above 850 is negative.
    """
    bank = LevelBank(make_banks())

    assert bank.levels == [500.0, 850.0]
    for pair in bank.vertical_offsets():
        assert pair["offset"] == pair["to_level"] - pair["from_level"]
        assert pair["direction"] == ("upward" if pair["to_level"] < pair["from_level"]
                                     else "downward")
    assert sorted({p["offset"] for p in bank.vertical_offsets()}) == [-350.0, 350.0]


def test_the_stacked_tensor_keeps_its_ascending_order():
    bank = LevelBank(make_banks())
    stacked = bank.to_tensor()

    assert stacked.shape[0] == 2
    assert torch.equal(stacked[0], bank.at_level(500.0).data)
    assert torch.equal(stacked[1], bank.at_level(850.0).data)
    assert bank.reference is bank.at_level(500.0)


def test_an_absent_level_is_refused_in_the_axis_units():
    bank = LevelBank(make_banks())
    with pytest.raises(UnknownNameError, match="hPa"):
        bank.at_level(700.0)
