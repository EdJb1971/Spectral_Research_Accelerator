# Identity-calibration receipts

**T4E.8 slice 2:** `t4e8-spatial-design.json` and `t4e8-spatial-audit.json` preserve the
first exploratory spatial-identity audit. `t4e8-spatial-design-v2.json` openly amends it to
add a monotone empirical feasibility diagnostic; `t4e8-spatial-audit-v2.json` is the
source-bound rerun, taken from an uncommitted tree and so carrying `code_dirty: true`.
`t4e8-spatial-audit-v2-clean.json` re-runs that same design against committed revision
`b902b87`; it is the citable receipt, and it reproduces every figure of the dirty one
exactly, differing only in wall-clock timings. The amendment changes no window, threshold, weight or radius rule.
**T4E.8 slice 3:** `t4e8-spatial-design-v3.json` amends v2 by naming the identity target
(`spatial_persistence`) and evidence class (`record_derived_proxy`) it was already measuring,
and changes nothing else. `t4e8-spatial-audit-v3.json` is its receipt, bound to clean revision
`3fc491a`, and reproduces `t4e8-spatial-audit-v2-clean.json` bit-for-bit on every figure. The
audit now refuses to run at all without those two fields, and refuses `kind_recurrence` against
record-derived labels as circular.
These designs are engineering declarations, not signed scientific preregistrations.
The candidate improves discrimination but fails its fixed-radius acceptance. No mining radius
is approved. All windows are within the existing training period; repeated track keys are
proxy labels and pairs are dependent. Reproduce without network access using:

```
.venv/Scripts/python.exe -m tools.audit_spatial_identity --design data/identity_calibration/t4e8-spatial-design-v2.json --output NEW_RECEIPT_PATH
```

The command refuses an existing output path and checks that its source and design did not
change during measurement. Historical receipts are retained even when their source bindings
are superseded. Their captured versions are evidence of what ran, not claims that the current
source remains identical.

What T4E.7's null calibration returned on the acquired ERA5 record, and the sweep it returned it
from. These are **not** gate receipts and must not live in `data/gate_receipts/`: that directory
is served by the read-only T4C gate record, which reports any file whose schema it does not
recognise as unreadable — correctly, and putting a foreign schema there once made the store
report itself unreadable. See `data/mining_declarations/README.md`, which exists for the same
reason.

* `t4e7-acquired-record-6000.json` — 6,000 pairs sampled from each population.
* `t4e7-acquired-record-40000.json` — 40,000 pairs, on the same signatures and the same null.

* `t4e8-replicate-census.json` -- T4E.8 slice 1, and the only receipt here that involves no null
  at all. The record labels its own strictest reading of identity: a signature carries the
  `track_ids` it was signed from, so two signatures sharing them are one tracked constellation
  observed twice. There are 6,838 such pairs. Both rates in this receipt are absolute counts
  against those labels, which is why it carries no p-value, no band and no mixture fraction --
  there is nothing in it to estimate.

**Both return no radius**, with status `RECURRENCE_FRACTION_UNSTABLE`, and they agree: the
simultaneous band moves from 0.0387 to 0.0398 across a 6.7× increase in sampled pairs and
nothing else changes. That is what makes the refusal a fact about the record rather than about
the sample, so both are kept rather than only the larger one.

Each receipt carries the whole sweep it was read from — every radius with its observed and null
close-pair fractions, its excess, its contamination estimate and whether it cleared the band —
because a receipt that states a radius without the sweep behind it asks to be believed rather
than checked, and here the claim *is* about the shape of the sweep: the excess is negative
throughout the close-pair tail and only clears the band at `r ≈ 0.37`, where the null already
admits a quarter of its own pairs.

Each also carries the `ensemble.warnings` entry recording that the record produced 69,580
signatures against a median of 16,090 for the null members — a factor of 4.3. That is **D98**,
and it is why no excess measured against this null can be read as recurrence.

**What is not kept here.** The sampled distance arrays. They are 7 MB of derived binary,
reproducible from the record and the declared seeds, and the receipts are self-contained without
them. Regenerating them means re-running the pipeline over the record and nineteen
`spatiotemporal_phase` surrogates of it, which takes about fourteen minutes on the machine these
were measured on.
