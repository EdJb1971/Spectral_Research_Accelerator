# Mining-pass declarations

Analysis designs fixed **before** a record is read, and the artefacts fitted under them.

These are not gate receipts and they must not live in `data/gate_receipts/`. That directory is
served by the read-only T4C gate record, which reads every file in it and reports anything whose
schema it does not recognise as **unreadable** — correctly, since a receipt store that quietly
skipped files it could not parse would be a store you could not trust. Putting a declaration
there made the store report itself unreadable, which is how this directory came to exist.

* `t4f6-tasman-mining-declaration.json` — the declared design for the T4F.6 mining pass over the
  acquired 8,764-frame ERA5 record: detection threshold, co-presence cap, train/test split,
  climatology model and transform depth, with the amendment that superseded the initial
  four-level transform recorded in the open rather than replaced.
* `t4f6-tasman-climatology.pt` — the harmonic climatology fitted under it on the 5,844 training
  frames of 2018–2021 alone (R11), with its own coefficient and fit digests.
