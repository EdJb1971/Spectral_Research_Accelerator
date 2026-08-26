# Store probe records (roadmap TG10.3)

One JSON file per recorded probe, named by its 16-character digest. A probe is one look at one
store: what its dimensions and chunks are, what a stated crop would cost, and — just as
readily — that the store was unreachable, wanted credentials, or was never attempted because
network access is switched off here. **Those are results about a store, not failures**, which
is why they are written down rather than raised.

Records are content-addressed and written atomically, and an existing file is never rewritten:
a file already at a digest holds that same observation.

**Nothing in this directory has been produced by a live probe of a public archive.** The four
ERA5 entries in the catalogue cite *transcriptions* of inspections run on 2026-08-20 and
2026-08-21, before this module existed; they live in `src/data_layer/stores.py` as
`BUILTIN_PROBES` and are counted by `store_probe.transcribed_probes()` so the number can only
fall where anyone can see it. Reaching a real archive stays opt-in via
`SPECTRALEARTH_ALLOW_NETWORK`.
