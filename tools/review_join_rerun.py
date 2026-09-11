"""T4E.31: run the adversarial round-robin over a published evidence bundle.

    # see exactly what would be sent, and send nothing
    .venv/Scripts/python.exe -m tools.review_join_rerun --bundle data/studies/t4e28-join-rerun.r7.json --dry-run

    # actually run it: eight paid calls to an external service
    GEMINI_API_KEY=... .venv/Scripts/python.exe -m tools.review_join_rerun \
        --bundle data/studies/t4e28-join-rerun.r7.json --output-dir data/reviews \
        --send-to-the-network

**Every part of this was already built and none of it had ever been run.** The protocol, the eight
seats, the dissent register, the R22 independence check and a Gemini batch transport are all
tested machinery. What was missing was a bundle to review -- T4E.30 supplied the first one -- and
a runner. This is the runner and nothing more.

**Nothing reaches the network unless the maintainer says so, in the command.** `--send-to-the-network`
is required, a key must be present, and without both the run refuses by name and sends nothing.
A round-robin is eight calls to an external service: it costs money, it puts the bundle's contents
in front of a third party, and it is not reversible. That decision belongs to a person.

**There is deliberately no stub transport here.** Tests fabricate responses to exercise the
protocol, which is correct in a test and would be a forgery in this directory: a review record is
a claim that a panel said something, and a fabricated one sitting in `data/reviews/` beside real
ones is the worst artefact this programme could produce. If the network is refused, the run stops
with nothing written.

**A turn that comes back malformed is kept.** `advance` records the call before validating it, so
a disappointing answer that was paid for is never silently discarded (R23). This tool reports the
refusal, writes the partial review record, and stops.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

sys.path.insert(0, os.getcwd())

from src.core.errors import SpectralEarthError                              # noqa: E402
from src.core.evidence import load_evidence_bundle                          # noqa: E402
from src.core.recorded_call import (                                        # noqa: E402
    REVIEW_ROLES, save_review_record, verify_claim_independence,
)
from src.core.review_cost import GeminiBatchTransport                       # noqa: E402
from src.core.round_robin import (                                          # noqa: E402
    SCHEMA_FOR_ROLE, PanelSeat, RecordedTurnRefused, ReviewPanel, RoundRobin, advance,
    close_round_robin, open_round_robin,
)

#: Environment variables a key may arrive in. Never a command-line argument: an argument lands in
#: the shell history and in the process table, where a key does not belong.
KEY_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: One model in several seats is a weaker exchange than several and is RECORDED rather than
#: refused -- an independent reassessment by the model that wrote the candidate synthesis is not
#: independent in the sense the word usually carries. `ReviewPanel` computes the overlap and
#: carries it in the panel digest, so a reader sees what kind of panel answered.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_EFFORT = "high"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def panel_from(model: str, effort: str) -> ReviewPanel:
    return ReviewPanel(seats={role: PanelSeat(role=role, model_id=model, effort=effort)
                              for role in REVIEW_ROLES})


def resolve_key() -> Optional[str]:
    for name in KEY_VARIABLES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def describe_turn(exchange: RoundRobin) -> Dict[str, Any]:
    """What the next call would carry, without making it."""
    turn = exchange.next_turn
    if turn is None:
        return {}
    seat = exchange.panel.seat(turn.role)
    described = turn.describe()
    return {"role": turn.role, "model_id": seat.model_id, "effort": seat.effort,
            "expects": sorted(turn.schema.fields), "turn": described}


def run(bundle_path: Path, *, panel: ReviewPanel, output_dir: Optional[Path],
        transport: Optional[Any]) -> int:
    bundle = load_evidence_bundle(bundle_path)
    exchange = open_round_robin(bundle, panel)

    print("bundle      %s" % bundle_path)
    print("study       %s  revision %d" % (bundle.study_id, bundle.revision))
    print("bundle_sha  %s" % bundle.bundle_sha256)
    print("panel       %s" % ", ".join(
        "%s=%s" % (role, panel.seat(role).model_id) for role in REVIEW_ROLES))
    print("claim state is independent of this review: %s"
          % verify_claim_independence(exchange.reviewed)[:16])

    if transport is None:
        print("\n-- dry run: the eight turns that would be taken, in order --")
        # Reported from the protocol's own role order and schema table. The later turns cannot
        # be walked without answers, and pretending the exchange advanced would be showing a
        # plan that had not been checked against anything.
        for index, role in enumerate(REVIEW_ROLES, 1):
            print("%d. %-30s %-16s expects %s"
                  % (index, role, panel.seat(role).model_id,
                     ", ".join(sorted(SCHEMA_FOR_ROLE[role].fields))))
        print("\nNOTHING WAS SENT. Re-run with --send-to-the-network and a key in %s to "
              "actually convene the panel." % " or ".join(KEY_VARIABLES))
        return 0

    for index in range(len(REVIEW_ROLES)):
        turn = exchange.next_turn
        if turn is None:
            break
        print("\n[%d/%d] %s" % (index + 1, len(REVIEW_ROLES), turn.role), flush=True)
        try:
            exchange = advance(exchange, transport=transport, requested_at=_now())
        except RecordedTurnRefused as refusal:
            print("  REFUSED: %s" % refusal)
            if output_dir is not None:
                partial = output_dir / ("%s.review.partial.json" % bundle.study_id)
                save_review_record(partial, refusal.reviewed.review, bundle_path=bundle_path)
                print("  the call was made and is kept (R23): %s" % partial)
            return 3
        answer = exchange.answer(turn.role) or {}
        for key in sorted(answer):
            value = answer[key]
            text = value if isinstance(value, str) else json.dumps(value)
            print("  %-18s %s" % (key, text[:160]))

    outcome = close_round_robin(exchange)
    print("\n---- outcome")
    print("finding        %s" % outcome.finding[:400])
    print("bounded by     %s" % "; ".join(outcome.bounded_by)[:400])
    print("dissent kept   %d" % len(outcome.retained_dissent))
    for dissent in outcome.retained_dissent:
        print("   %s" % json.dumps(dissent.to_mapping())[:200])
    print("rung           %s   (copied from the bundle, never set here)" % outcome.rung)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        review_path = output_dir / ("%s.review.json" % bundle.study_id)
        outcome_path = output_dir / ("%s.round-robin.json" % bundle.study_id)
        save_review_record(review_path, exchange.reviewed.review, bundle_path=bundle_path)
        outcome_path.write_text(json.dumps(outcome.to_mapping(), indent=2), encoding="utf-8")
        print("\nwrote %s\nwrote %s" % (review_path, outcome_path))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convene the round-robin over a bundle.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the turns that would be taken and send nothing")
    parser.add_argument("--send-to-the-network", action="store_true",
                        help="required to make any call; eight paid calls to an external service")
    arguments = parser.parse_args()

    panel = panel_from(arguments.model, arguments.effort)
    bundle_path = Path(arguments.bundle)
    if not bundle_path.is_file():
        print("REFUSED: no bundle at %s. Build one with tools/bundle_join_rerun.py."
              % bundle_path)
        return 2

    transport = None
    if not arguments.dry_run:
        if not arguments.send_to_the_network:
            print("REFUSED: this would make %d calls to an external service, which costs money "
                  "and puts the bundle in front of a third party. Pass --send-to-the-network to "
                  "authorise it, or --dry-run to see exactly what would be sent."
                  % len(REVIEW_ROLES))
            return 2
        key = resolve_key()
        if key is None:
            print("REFUSED: --send-to-the-network was given but no key is present in %s. "
                  "Nothing was sent. A key is read from the environment and never from an "
                  "argument, which would land it in the shell history and the process table."
                  % " or ".join(KEY_VARIABLES))
            return 2
        transport = GeminiBatchTransport(key)

    try:
        return run(bundle_path, panel=panel,
                   output_dir=Path(arguments.output_dir) if arguments.output_dir else None,
                   transport=transport)
    except SpectralEarthError as error:
        print("REFUSED: %s" % error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
