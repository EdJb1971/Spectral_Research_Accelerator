# Demonstration channel records (TG8.4)

**These are fabricated. No market data, and no data of any kind from any real venue or archive,
has been ingested by this project.** They exist so the ingestion seam in *12. Domain Records* can
be exercised without a network, in the same sense that `synthetic_generator/` keeps the transform
workbench alive offline and declares no truth. Nothing derived from them is evidence about
anything.

They were written by a seeded generator (`numpy` default_rng, seed 20260827), so they are
reproducible and identical on every machine.

| File | Rows | What it demonstrates |
|---|---|---|
| `order_book_regular.csv` | 600 | A regularly sampled record, one row per minute. Loads under **order_book**; the tab reports a cadence of 60 s. |
| `order_book_irregular.csv` | 600 | The clock a venue actually gives you — rows arrive when trades do. Loads under **order_book** only because that domain declares `irregular_sampling`, and the tab reports *irregular (declared)* rather than inventing a cadence. |
| `clock_runs_backwards.csv` | 120 | A clock that jumps backwards partway through. **No declaration waives this**, so inspection stops and says so. The rows are not sorted for you: a file whose order was silently corrected is one whose provenance no longer describes it. |

Every one of them is refused by **reanalysis**, which declares latitude and longitude — axes a
flat channel table cannot supply (standard E14). That refusal is the point rather than an
inconvenience: a gridded domain does not describe a channel table, and reading one under the
other would quietly drop axes that domain's results are indexed by.

## Things worth trying

* Load `order_book_regular.csv` under `order_book`, then open *Declare a channel as a window
  aggregate* and give `trade_count` a footprint of 60 samples. It loads, because `order_book`
  declares `aggregated_values`. A domain that did not would refuse it by name.
* Load `order_book_irregular.csv` and read the per-domain verdicts before choosing: the
  inspection states that whichever domain you pick must already declare `irregular_sampling`,
  and names which ones do.
* Load `clock_runs_backwards.csv` and note that no column is substituted for the broken clock,
  even though the price columns happen to increase.
