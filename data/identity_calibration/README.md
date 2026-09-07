# Identity-calibration receipts

What T4E.7's null calibration returned on the acquired ERA5 record, and the sweep it returned it
from. These are **not** gate receipts and must not live in `data/gate_receipts/`: that directory
is served by the read-only T4C gate record, which reports any file whose schema it does not
recognise as unreadable — correctly, and putting a foreign schema there once made the store
report itself unreadable. See `data/mining_declarations/README.md`, which exists for the same
reason.

* `t4e7-acquired-record-6000.json` — 6,000 pairs sampled from each population.
* `t4e7-acquired-record-40000.json` — 40,000 pairs, on the same signatures and the same null.

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
