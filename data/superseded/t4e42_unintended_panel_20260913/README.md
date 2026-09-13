# An eight-seat review panel nobody convened (2026-09-13)

These two files are a complete, paid, recorded round-robin review of `t4e28-join-rerun` at bundle
revision 7. They were produced at 18:22 on 2026-09-13 by a **browser test**, not by a maintainer,
and they are archived here rather than kept in `data/reviews/` for that reason alone. Nothing
about them is known to be technically wrong.

## What happened

`frontend/e2e/adoption.spec.ts` contained a test named *"convening without a key on the server
refuses and says nothing was sent"*. It ticked the authorisation control and clicked convene,
expecting the refusal that the API returns when no review API key is present. That expectation
was an assumption about the machine, not a fact the test established: `.env.local` on this
machine supplies `GEMINI_API_KEY`, so the click did not refuse. It convened the panel — eight
seats, `gemini-3.5-flash`, up to eleven paid calls on the maintainer's own account, with the
review bundle sent to a third-party service.

It went unnoticed for several runs because of a second defect. `convene_round_robin` was declared
`async def` while its transport is blocking, so the call sat on the event loop and the entire API
stopped answering — including `/health` — until the process was killed. Every panel in the
browser suite then failed to load, which read as a broken frontend rather than as a convening in
progress.

Both defects are fixed under T4E.42: the handler is a synchronous route that FastAPI runs in a
threadpool, the browser suite's backend has the key variables cleared, and the test now asserts
`key_present === false` against the server before it clicks anything.

## Why these files are kept

The calls were made and paid for, and they cannot be regenerated: R23 says nothing paid for is
discarded. Deleting them would also delete the evidence of how they came to exist.

## Why they are not in `data/reviews/`

The record's provenance is a test harness, not a person. The study's own history states that two
paid attempts were preserved and **no valid panel had completed**; letting an unintended run
silently become that study's completed panel would put a finding into the record that no
maintainer asked for, reviewed, or adopted. If this panel is ever to count, that is a maintainer's
decision, made deliberately, and it can be moved back.

## What is in them

- `t4e28-join-rerun.round-robin.json` — the outcome: `bundle_sha256 b1e809d6…`, revision 7,
  `panel_sha256 9fcef030…`, `record_sha256 e285e117…`, rung `observation`, no retained dissent.
- `t4e28-join-rerun.review.json` — the full recorded exchange, 171 KB.

Their content is untouched: the bytes are exactly as written at 18:22.
