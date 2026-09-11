# External catalogues

Reference data this programme did not produce, admitted as `external_reference` evidence by a
signed declaration and bound by content hash rather than committed. The files are **deliberately
absent from git** (`data/catalogues/*.csv` in `.gitignore`), for the same reason the CDS shards
and the market record are: third-party data of this size does not belong in a source repository,
and a digest is a stronger binding than a copy.

## `ibtracs.SP.list.v04r01.csv`

International Best Track Archive for Climate Stewardship (IBTrACS), Project Version 4.01, **South
Pacific basin subset**. NOAA National Centers for Environmental Information.

| | |
|---|---|
| DOI | `10.25921/82ty-9e16` |
| Source | `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.SP.list.v04r01.csv` |
| Retrieved | 2026-09-10 |
| sha256 | `631f76b95c77a6a4e409233466a0d501bb4848324e421fc58e228efea2086c44` |
| Bytes | 35,482,417 |
| Signed by | T4E.17, `t4e17-external-catalogue-signature.json`, on the amended domain |

**The digest is the binding, not the filename.** T4E.17's signature admits *that content* on terms
naming a specific population -- 210 observations, 20 storms, 20,241 cross-storm pairs, base rate
0.391 on `NATURE`, in the crop latitude -58 to -18 and longitude 140 to 180. The design states it
plainly: *"Any re-download must reproduce that digest or the evaluation is invalid and must say
so."*

**A fresh download will probably not match.** IBTrACS v04r01 is a living file: NOAA appends
observations as storms occur and revises earlier best-track entries in later passes. A copy
fetched today is a different catalogue, the signature does not extend to it, and the population
figures above would have to be re-derived rather than assumed. Replacing this file is therefore a
declaration, not a file copy.

**Verify before use:**

```
sha256sum data/catalogues/ibtracs.SP.list.v04r01.csv
```

### The citation is a binding obligation, not a courtesy

NCEI states full and open access for scientific research; WMO Resolution 40 governs commercial
use. This programme is a personal, non-commercial experiment on public data, so the open-access
terms apply and **the citation requirement is what this repository owes in return**. Any work
using this file cites both:

> Gahtan, Jennifer; Knapp, Kenneth R.; Schreck, Carl J.; Diamond, Howard J.; Kossin, James P.;
> Kruk, Michael C. (2024). International Best Track Archive for Climate Stewardship (IBTrACS)
> Project, Version 4.01. NOAA National Centers for Environmental Information.
> https://doi.org/10.25921/82ty-9e16

> Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann (2010): The
> International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone
> best track data. *Bulletin of the American Meteorological Society*, 91, 363-376.

### Why this directory exists at all

The file spent a day in a Claude session's scratchpad under `%TEMP%`, where it had survived by
luck and could have been cleaned up at any time. Losing it would have voided the signature,
because a re-download almost certainly would not reproduce the digest. The signed design records
the URL, the hash and the byte count and records **no expected local path**, so nothing in the
repository knew where to look and nothing failed loudly when it was missing --
`tools/restate_position_acceptance.py` could only refuse T4E.27's condition 2 by name and say the
file was absent.

That is the same lesson T4E.27 drew about a catalogue radius of `0.00`, applied to a file instead
of a field: **a missing input should be refused by name, not discovered later.** A loader that
checks this path and this digest, and refuses with both when they do not match, is the repair and
has not been written.

### What it unblocks

* **T4E.27 condition 2** -- three or more features inside tolerance, currently refused because the
  third-nearest distance is not in the committed receipt and recomputing it needs this file.
* **`kind_recurrence`** -- the primary identity target, whose only admissible evidence class is
  `external_reference`, of which this is the sole instance in the programme.
