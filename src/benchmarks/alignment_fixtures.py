"""Adversarial clock, support and coverage fixtures (TG17.4, `ed-dev`).

Each fixture is a case where a row-index comparison, a nominal-day denominator or a closed
interval would report structure that is not there. They exist to be aligned as declared or
refused by name — never quietly repaired.

Nothing here is acquired and nothing here is a measurement. A fixture carries support and a
known answer about that support; it carries no values, because every failure mode in this file
is reachable without opening a single one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - the project targets 3.9+
    ZoneInfo = None  # type: ignore

from src.core.structural_alignment import Interval, SupportProfile, support_profile


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
WEEK_SECONDS = 7 * 86400.0
HOUR = 3600.0

#: The transition this project cares about: the UK moves to summer time on 2026-03-29, so the
#: local calendar day that begins at 2026-03-29T00:00 local is twenty-three hours long.
DST_ZONE = "Europe/London"
DST_LOCAL_DAY = (2026, 3, 29)


@dataclass(frozen=True)
class AlignmentFixture:
    """One adversarial case, its support, and the answer it is known to have."""

    name: str
    question: str
    window: Interval
    profiles: Mapping[str, SupportProfile]
    expected: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "question": self.question,
                "window_start_seconds": self.window[0], "window_end_seconds": self.window[1],
                "profiles": {name: profile.describe()
                             for name, profile in sorted(self.profiles.items())},
                "expected": dict(self.expected)}


def _tiled(label: str, domain: str, window: Interval, cadence: float, *,
           offset: float = 0.0, duty: float = 1.0) -> SupportProfile:
    """A regularly sampled record whose supports tile the window at one cadence."""
    starts, ends = [], []
    cursor = window[0] + offset
    while cursor < window[1]:
        stop = min(cursor + cadence * duty, window[1])
        if stop > cursor:
            starts.append(cursor)
            ends.append(stop)
        cursor += cadence
    return support_profile(label=label, domain=domain, starts=starts, ends=ends,
                           window=window, native_scale_seconds=cadence)


def _week() -> Interval:
    return (EPOCH.timestamp(), EPOCH.timestamp() + WEEK_SECONDS)


# ------------------------------------------------------------------------------ the six cases


def _unequal_cadence() -> AlignmentFixture:
    window = _week()
    hourly = _tiled("hourly", "reanalysis", window, HOUR)
    six_hourly = _tiled("six_hourly", "reanalysis_coarse", window, 6 * HOUR)
    return AlignmentFixture(
        name="unequal_cadence",
        question=("Two records covering the same week at cadences differing by six. Does the "
                  "finer record lend the coarser one resolution it does not have?"),
        window=window, profiles={"hourly": hourly, "six_hourly": six_hourly},
        expected={
            "overlap_seconds": WEEK_SECONDS,
            "governing_scale_seconds": 6 * HOUR,
            "effective_sample_size": WEEK_SECONDS / (6 * HOUR),
            "row_counts": {"hourly": 168, "six_hourly": 28},
            "answer": ("28 effective observations, not 168. The coarser scale governs, so a "
                       "record cannot borrow resolution from the record it is compared with."),
        })


def _abutting_boundary() -> AlignmentFixture:
    window = _week()
    midpoint = window[0] + WEEK_SECONDS / 2
    before = support_profile(label="before", domain="reanalysis", starts=[window[0]],
                             ends=[midpoint], window=window, native_scale_seconds=HOUR)
    after = support_profile(label="after", domain="reanalysis", starts=[midpoint],
                            ends=[window[1]], window=window, native_scale_seconds=HOUR)
    return AlignmentFixture(
        name="abutting_boundary",
        question=("Supports that touch at exactly one instant. Under a closed convention every "
                  "boundary in every regular record is a coincidence. Is it one here?"),
        window=window, profiles={"before": before, "after": after},
        expected={
            "overlap_seconds": 0.0, "effective_sample_size": 0.0,
            "answer": ("Nothing. `[a, b)` and `[b, c)` abut and do not overlap, so the pair "
                       "refuses rather than reporting an instant of shared support."),
        })


def _daylight_saving_day() -> AlignmentFixture:
    if ZoneInfo is None:  # pragma: no cover
        raise RuntimeError("this fixture needs zoneinfo")
    zone = ZoneInfo(DST_ZONE)
    local_start = datetime(*DST_LOCAL_DAY, tzinfo=zone)
    local_end = (local_start + timedelta(days=1)).astimezone(zone)
    start, end = local_start.timestamp(), local_end.timestamp()
    window = (start, end)
    elapsed = end - start
    covered = _tiled("hourly", "reanalysis", window, HOUR)
    return AlignmentFixture(
        name="daylight_saving_day",
        question=("One local calendar day across a spring transition. Is the coverage "
                  "denominator 86400 seconds?"),
        window=window, profiles={"hourly": covered},
        expected={
            "elapsed_seconds": elapsed, "nominal_seconds": 86400.0,
            "discrepancy_seconds": elapsed - 86400.0,
            "covered_fraction": 1.0,
            "answer": ("It is %d seconds. A nominal-day denominator would report %.1f%% "
                       "coverage for a record that covers the window completely."
                       % (int(elapsed), 100.0 * elapsed / 86400.0)),
        })


def _sparse_profile() -> AlignmentFixture:
    """Argo-shaped: a ten-day cycle with a short ascent, against a continuous grid."""
    window = (EPOCH.timestamp(), EPOCH.timestamp() + 90 * 86400.0)
    ascent = 4 * HOUR
    starts = [window[0] + index * 10 * 86400.0 for index in range(9)]
    profile = support_profile(label="argo", domain="argo_float", starts=starts,
                              ends=[start + ascent for start in starts], window=window,
                              native_scale_seconds=ascent)
    grid = _tiled("era5", "reanalysis", window, HOUR)
    return AlignmentFixture(
        name="sparse_profile",
        question=("Nine four-hour ascents inside ninety days of continuous grid. How much of "
                  "the window is actually shared?"),
        window=window, profiles={"argo": profile, "era5": grid},
        expected={
            "overlap_seconds": 9 * ascent,
            "argo_covered_fraction": 9 * ascent / (90 * 86400.0),
            "governing_scale_seconds": ascent,
            "effective_sample_size": 9.0,
            "answer": ("Thirty-six hours out of ninety days: 1.67% of the window. Nine "
                       "effective observations, which is the number of ascents."),
        })


def _interrupted_light_curve() -> AlignmentFixture:
    """TESS-shaped: two sector halves either side of a mid-sector downlink gap."""
    window = (EPOCH.timestamp(), EPOCH.timestamp() + 27 * 86400.0)
    gap_start = window[0] + 13 * 86400.0
    gap_end = gap_start + 2 * 86400.0
    curve = support_profile(
        label="tess", domain="tess_lightcurve",
        starts=[window[0], gap_end], ends=[gap_start, window[1]], window=window,
        native_scale_seconds=120.0)
    grid = _tiled("era5", "reanalysis", window, HOUR)
    return AlignmentFixture(
        name="interrupted_light_curve",
        question=("A sector with a two-day downlink gap in the middle. Does the gap survive "
                  "into the shared support, or does a continuous partner paper over it?"),
        window=window, profiles={"tess": curve, "era5": grid},
        expected={
            "gap_count": 1, "largest_gap_seconds": 2 * 86400.0,
            "overlap_seconds": 25 * 86400.0,
            "covered_fraction": 25.0 / 27.0,
            "answer": ("The gap survives: shared support is 25 of 27 days and the overlap is "
                       "reported as two intervals, not one."),
        })


def _non_stationary_support() -> AlignmentFixture:
    """Support that lengthens through the window, as a drifting or degrading sensor does."""
    window = _week()
    starts, ends, cursor, width = [], [], window[0], HOUR
    while cursor < window[1]:
        stop = min(cursor + width, window[1])
        starts.append(cursor)
        ends.append(stop)
        cursor = stop
        width *= 1.25
    drifting = support_profile(label="drifting", domain="bespoke_record", starts=starts,
                               ends=ends, window=window, native_scale_seconds=HOUR)
    steady = _tiled("steady", "reanalysis", window, HOUR)
    return AlignmentFixture(
        name="non_stationary_support",
        question=("Support that grows from one hour to over a day inside one week. Is a single "
                  "governing scale still an honest denominator?"),
        window=window, profiles={"drifting": drifting, "steady": steady},
        expected={
            "support_is_stationary": False,
            "overlap_seconds": WEEK_SECONDS,
            "answer": ("No, and the report says so: the effective sample size is labelled an "
                       "upper bound because one record's support duration is not constant."),
        })


_BUILDERS = {
    "unequal_cadence": _unequal_cadence,
    "abutting_boundary": _abutting_boundary,
    "daylight_saving_day": _daylight_saving_day,
    "sparse_profile": _sparse_profile,
    "interrupted_light_curve": _interrupted_light_curve,
    "non_stationary_support": _non_stationary_support,
}

ALIGNMENT_FIXTURES: Tuple[str, ...] = tuple(_BUILDERS)


def alignment_fixture(name: str) -> AlignmentFixture:
    if name not in _BUILDERS:
        raise KeyError("unknown alignment fixture %r; known: %s"
                       % (name, ", ".join(ALIGNMENT_FIXTURES)))
    return _BUILDERS[name]()


def build_alignment_fixtures() -> Mapping[str, AlignmentFixture]:
    return MappingProxyType({name: builder() for name, builder in _BUILDERS.items()})


def densify(profile: SupportProfile, factor: int) -> SupportProfile:
    """The same support, written as `factor` times as many rows.

    The acceptance instrument for the invariant this slice exists to hold: every derived
    quantity must be unchanged, because nothing about the record changed except how many lines
    it took to write it down.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    starts, ends = [], []
    for start, end in profile.intervals:
        step = (end - start) / factor
        for index in range(factor):
            starts.append(start + index * step)
            ends.append(start + (index + 1) * step)
    return support_profile(label=profile.label, domain=profile.domain, starts=starts,
                           ends=ends, window=profile.window,
                           native_scale_seconds=profile.native_scale_seconds)


__all__ = ["ALIGNMENT_FIXTURES", "AlignmentFixture", "DST_ZONE", "alignment_fixture",
           "build_alignment_fixtures", "densify"]
