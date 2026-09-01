# Store probe records (roadmap TG10.3)

One JSON file per recorded probe, named by its 16-character digest. A probe is one look at one
store: what its dimensions and chunks are, what a stated crop would cost, and — just as
readily — that the store was unreachable, wanted credentials, or was never attempted because
network access is switched off here. **Those are results about a store, not failures**, which
is why they are written down rather than raised.

Records are content-addressed and written atomically, and an existing file is never rewritten:
a file already at a digest holds that same observation.

The two checked-in JSON records are live anonymous metadata probes of the paired GLORYS ARCO
layouts, run for TG12.1 on 2026-08-28. `ccdb0625e7e8fb1d` is the selected time-deep
`geoChunked.zarr` layout; `f9a45764fcf52c2a` preserves the rejected spatially huge
`timeChunked.zarr` layout. They transferred metadata only, not ocean field values.

`1a28d5980c97a38e` is the 0.25-degree WeatherBench 2 ERA5 store, probed on 2026-09-01
immediately before the T4C.6 v2 eight-frame overlap window was materialised from it. It is
checked in because the D43 ledger row and `VERIFICATION.md` both cite its figures: chunks
of `(1, 13, 721, 1440)` at 53.99 MB, and 520.7x amplification for that crop. Metadata only;
no field values were transferred by the probe itself.

The four ERA5 entries in the catalogue still cite *transcriptions* of inspections run on
2026-08-20 and 2026-08-21, before this module existed; they live in
`src/data_layer/stores.py` as `BUILTIN_PROBES` and are counted by
`store_probe.transcribed_probes()`. Reaching a real archive remains opt-in via
`SPECTRALEARTH_ALLOW_NETWORK`.
