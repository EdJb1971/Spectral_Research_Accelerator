# Researcher-supplied order-book records (TG17.14)

**The CSV this directory is for is deliberately not committed**, and its absence is the point
rather than an oversight.

`order_book` is the one domain in the TG17.14 live-source contract whose
`network_required_for_pass` is `false`: it must demonstrate a **content-addressed local record**
supplied by the researcher before execution, with no network use at all. The gate binds that
record by `sha256` and records its provenance and licence; it never stores the record itself. The
domain's own declaration in `src/core/builtin_domains.py` reads *"Venue market-data terms;
redistribution of raw depth is generally prohibited"*, so committing the file would contradict the
licence the platform states while reading it.

`data/market_records/*.csv` is therefore in `.gitignore`. What follows is everything needed to
rebuild the record byte-for-byte and check the hash the gate recorded.

## The record used for the recorded run

| | |
|---|---|
| File | `btcusdt_bookdepth_2026-09-01.csv` |
| `sha256` | `b2fd0c2cfe426cf8c3c142ca2195da9564a6e65517c3cf8287afa4968edfe94f` |
| Source | `https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2026-09-01.zip` |
| Retrieved | 2026-09-04 UTC |
| Rows | 2,880 snapshots x 12 channels = 34,560 observed values |
| Clock | epoch seconds UTC, 1788220804 to 1788307171, strictly increasing |
| Spacing | **irregular**, 28-33 s |

The irregular spacing is worth noting rather than smoothing away. `order_book` is the only
builtin domain that declares `irregular_sampling`, and it acquired that declaration in TG8.4 as
defect **D61** — the description had said "irregular trading clock" from the first commit while
the violation tuple did not. This record is the first *real* data to be read under that
declaration, and it is irregular in exactly the way the declaration claims. A domain that did not
declare it would refuse this file by name.

## Rebuilding it

The venue publishes long format — `timestamp,percentage,depth,notional`, twelve rows per
snapshot. The channel table is a pivot of that and nothing else:

* the twelve percentage levels (-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5) each become one
  column, named `depth_bid_*` below the mid and `depth_ask_*` above it, with `0.2` written `0p2`;
* the UTC `timestamp` becomes `t` in epoch seconds, formatted `%.6f`;
* `depth` is retained formatted `%.8f`; `notional` is discarded;
* rows are ordered by `t` and **no row is added, removed, reordered or resampled**.

Every snapshot in this day carries all twelve levels, so the table has no missing cells. That is a
property of this file, not a guarantee about the feed: a snapshot missing a level would leave an
empty cell, and `read_channels_for_domain` would carry it as absent rather than interpolate it.

## What the record is not

Reading this file under `order_book` establishes nothing about where it came from. The reader says
so itself, in the provenance it returns: *"This record was read under domain 'order_book' because
a reader chose that domain. Nothing here establishes that it came from it."* The live-source gate
qualifies bounded acquisition and record binding. It is not a market result, not a scientific
claim, and not evidence about anything that happens in a market.

The fabricated demonstration records in `data/channels/` remain fabricated and are explicitly
**ineligible** here: `measure_live_sources` hashes the candidate against every committed channel
CSV and refuses a match, so a fixture cannot satisfy a live-source gate.
