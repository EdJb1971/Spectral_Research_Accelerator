"""Can the change you are about to make move the answer at all?

**The failure this exists to prevent, which already happened.** T4E.27 declared, before measuring,
that restating a bar would leave the acceptance failing *"improving on 2 of 18 but not reaching
9"*. The failure half held. The improvement half was not merely wrong -- it was **impossible**, and
the impossibility was computable from three columns that already existed.

The restated bar was `sqrt(radius^2 + 8.19^2)` against radii of 11 to 145 km. It moves a bar of
11.12 to 13.81 and one of 89.38 to 89.76. Set that against separations of 30 to 2056 km and **not
one observation** has its separation between the old bar and the new one. The swing set is empty:
no verdict could change, whatever the atmosphere did.

**What this module computes.** Given the values being judged, the old bar and the new one, it
reports which observations *could* change verdict -- those whose value lies between the two bars --
and refuses a predicted direction the arithmetic cannot produce. It is not a forecast and not a
substitute for measuring: it says only whether the measurement has any room to surprise you.

**Why that is worth a module.** A prediction declared before a measurement is this programme's
main guard against reading a result into the answer already believed. That guard is worth less
when the prediction was unfalsifiable in a direction nobody checked. Declaring a prediction the
arithmetic forbids does not make a slice rigorous; it makes it look rigorous, which is worse.

**What it does not do.** It sees one kind of error: a threshold test whose threshold moves. It
says nothing about whether the right quantity is being thresholded, whether the population is the
right one, or whether the bar is defensible -- the three things that actually decided T4E.27. A
clean report here is not a sound design, and a module that implied otherwise would be selling the
same false comfort it was written to remove.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError

#: What a bar change can do to one observation.
OUTCOMES = ("gained", "lost", "admitted_either_way", "rejected_either_way", "not_judged")

#: The directions a declaration can predict, and what each needs to be possible.
DIRECTIONS = ("improve", "worsen", "unchanged")


@dataclass(frozen=True)
class Travel:
    """How much room a bar change has to move the answer, and where."""

    outcomes: Tuple[Tuple[str, str], ...]
    gained: Tuple[str, ...]
    lost: Tuple[str, ...]
    not_judged: Tuple[str, ...]
    admitted_before: int
    admitted_after: int
    judged: int

    @property
    def swing(self) -> Tuple[str, ...]:
        """Everything whose verdict the change actually moves."""
        return tuple(list(self.gained) + list(self.lost))

    @property
    def inert(self) -> bool:
        """No verdict can change, so no prediction of a changed verdict can hold."""
        return not self.swing

    def describe(self) -> Dict[str, object]:
        return {
            "judged": self.judged,
            "not_judged": list(self.not_judged),
            "admitted_before": self.admitted_before,
            "admitted_after": self.admitted_after,
            "gained": list(self.gained),
            "lost": list(self.lost),
            "swing_set_size": len(self.swing),
            "inert": self.inert,
            "what_inert_means": (
                "No observation has its value between the old bar and the new one, so no "
                "verdict can change however the bar is justified. A prediction of improvement "
                "or of worsening is not risky here -- it is arithmetically impossible, and "
                "declaring one would look like a guard without being one."),
            "what_this_does_not_check": (
                "Whether the right quantity is thresholded, whether the population is the right "
                "one, and whether the bar is defensible. A clean report here is not a sound "
                "design."),
        }


def verdict_travel(values: Sequence[float],
                   old_bars: Sequence[Optional[float]],
                   new_bars: Sequence[Optional[float]],
                   labels: Optional[Sequence[str]] = None) -> Travel:
    """Which observations could change verdict when the bar moves from old to new.

    A bar of `None` is a refusal -- the observation is not judged under that bar and cannot
    change, which keeps a missing input out of the swing set rather than counting it as movement.
    Bars are inclusive: a value equal to its bar is admitted, matching `PositionTolerance.admits`.
    """
    n = len(values)
    if not (len(old_bars) == len(new_bars) == n):
        raise InvalidParameterError(
            "verdict_travel.lengths", (n, len(old_bars), len(new_bars)),
            "one old bar and one new bar per value; a short zip would silently drop "
            "observations off the end of the population")
    names = [str(x) for x in labels] if labels is not None else [str(i) for i in range(n)]
    if len(names) != n:
        raise InvalidParameterError(
            "verdict_travel.labels", len(names), "one label per value")

    outcomes: List[Tuple[str, str]] = []
    gained: List[str] = []
    lost: List[str] = []
    not_judged: List[str] = []
    before = after = judged = 0

    for name, value, old, new in zip(names, values, old_bars, new_bars):
        if old is None or new is None or not math.isfinite(float(value)):
            outcomes.append((name, "not_judged"))
            not_judged.append(name)
            continue
        judged += 1
        was = float(value) <= float(old)
        now = float(value) <= float(new)
        before += 1 if was else 0
        after += 1 if now else 0
        if was and now:
            outcomes.append((name, "admitted_either_way"))
        elif not was and not now:
            outcomes.append((name, "rejected_either_way"))
        elif now:
            outcomes.append((name, "gained"))
            gained.append(name)
        else:
            outcomes.append((name, "lost"))
            lost.append(name)

    return Travel(outcomes=tuple(outcomes), gained=tuple(gained), lost=tuple(lost),
                  not_judged=tuple(not_judged), admitted_before=before,
                  admitted_after=after, judged=judged)


def check_prediction(travel: Travel, predicted: str) -> Dict[str, object]:
    """Is a predicted direction possible at all, given where the bar can move?

    Returns a verdict on the *prediction*, not on the measurement. `POSSIBLE` means the
    arithmetic permits it and the measurement is what decides; `IMPOSSIBLE` means it is settled
    before any data is touched, and a declaration carrying it would be claiming a guard it does
    not have.
    """
    if predicted not in DIRECTIONS:
        raise InvalidParameterError(
            "check_prediction.predicted", predicted,
            "one of %s" % ", ".join(DIRECTIONS))

    possible = {
        "improve": bool(travel.gained),
        "worsen": bool(travel.lost),
        "unchanged": True,
    }[predicted]

    if predicted == "unchanged" and travel.inert:
        return {
            "predicted": predicted,
            "verdict": "TRIVIALLY_TRUE",
            "reason": (
                "Nothing can change, so predicting no change is true before the measurement "
                "runs and carries no evidential weight. Say so in the declaration rather than "
                "letting it read as a risk that was taken."),
            "swing_set_size": 0,
        }
    return {
        "predicted": predicted,
        "verdict": "POSSIBLE" if possible else "IMPOSSIBLE",
        "reason": (
            ("%d observation(s) could move that way, so the measurement decides it"
             % len(travel.gained if predicted == "improve" else travel.lost))
            if possible else
            ("No observation can move that way: none has its value between the old bar and the "
             "new one in that direction. The prediction is settled by arithmetic before any "
             "data is read, so declaring it does not make the slice falsifiable in this "
             "respect -- and a declaration that reads as though it did is worse than one that "
             "predicts nothing.")),
        "swing_set_size": len(travel.swing),
    }
