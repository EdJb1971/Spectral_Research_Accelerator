# SpectralEarth: Verification Record

Per `roadmap.md` Section 10, no claim of "validated", "verified" or "zero-error" may appear
anywhere in this repository without corresponding recorded command output in this file.

Every entry below is actual captured output, not a summary of intent.

---

## T3.5.0 — Establish a verified baseline — **COMPLETE**

**Date:** 2026-08-19
**Platform:** Windows 11 (10.0.26200), Python 3.10.11, Node v24.18.1, npm 11.16.0

### Result summary

| Check | Before | After |
|---|---|---|
| Backend test suite | **8 failed, 11 passed** | **54 passed, 1 xfailed** |
| Test files | 4 (none for the transform engine) | 5 (+36 transform cases) |
| Frontend build | exit 0, **1 module, no JS/CSS emitted** | 1,378 modules, JS + CSS emitted |
| Frontend rendering | never mounted (no entry point) | mounts |
| DWT / DTCWT / hybrid transforms | all raised, every inverse dead | exact round-trip, MSE < 1e-9 |
| Boundary-Condition Lab (Tukey) | `TypeError`, never worked | works |
| End-to-end sweep | untested | 9/9 runs, 28 lineage nodes, 54 edges |
| Defects known | 18 (static reading) | 25 known, **11 fixed** |

### Installed dependency versions

```
$ .venv/Scripts/python.exe -m pip list
fastapi           0.110.3      pydantic          1.10.26
httpx             0.28.1       pytest            9.1.1
numpy             2.2.6        SQLAlchemy        2.0.52
pandas            2.3.3        starlette         0.37.2
torch             2.13.0       uvicorn           0.52.4
xarray            2025.6.1
```

`requirements.txt` pins only lower bounds, so this resolved to **numpy 2.x** and **torch 2.13**.
Two latent defects (D20, D23) are direct consequences — they would not have appeared on the
versions the code was originally written against. Pinning upper bounds is follow-up work.

Repository was **not** under version control; `git init` was run to enable the automatic
provenance capture E5 requires. 41 files staged, **no commit made** (awaiting your go-ahead).

---

### Backend: first test run (baseline, before any fix)

```
$ .venv/Scripts/python.exe -m pytest src/tests -q
FAILED src/tests/test_analysis_data.py::test_analysis_engine_diagnostics
FAILED src/tests/test_analysis_data.py::test_api_endpoints_analysis_data
FAILED src/tests/test_boundary_synthetic.py::test_synthetic_vortex_generation
FAILED src/tests/test_boundary_synthetic.py::test_boundary_condition_lab
FAILED src/tests/test_boundary_synthetic.py::test_api_endpoints
FAILED src/tests/test_experiments.py::test_experiment_engine_execution
FAILED src/tests/test_experiments.py::test_experiment_api_endpoints
FAILED src/tests/test_hypothesis.py::test_hypothesis_api_endpoints
8 failed, 11 passed, 2 warnings in 8.59s
```

I had predicted exactly one failure (D3). There were eight, from five distinct root causes.
Four were new defects; one "failure" turned out to be a wrong test.

**Minimal repros, run to confirm each cause rather than infer it:**

```
1) 2D reflect pad 4-tuple:   FAIL: Padding size 4 is not supported for 2D input tensor.   -> D20
2) torch.cos(float):         FAIL: TypeError cos(): argument 'input' must be Tensor       -> D21
3) sqlite :memory: threads:  FAIL: (sqlite3.OperationalError) no such table: t            -> D22
4) torch.tensor([1.0/np.sqrt(2.0), ...]).dtype = torch.float64  (expected float32)        -> D23
5) stack(dim=1) on 2D -> (1, 16, 4, 16)   WRONG; stack(dim=0) -> (1, 4, 16, 16) correct   -> D25
```

### The vortex "failure" was a bad test, not a bad generator

```
E   assert 1.9770091772079468 == 2.0 ± 0.02
```

1.977 is **correct**. At N=32 on [0, 1] the grid spacing is 1/31, so the requested centre
(0.5, 0.5) falls *between* nodes; the nearest node is at 15/31 = 0.4839, and
`2·exp(−2·0.0161²/(2·0.15²)) = 1.977`. The generator is right; the assertion assumed the peak
landed on a node. The test now asserts 1.9770 at N=32 **and** exactly 2.0 at N=33, where the
centre does land on a node.

### Backend: final test run

```
$ .venv/Scripts/pytest -q          # bare pytest, from the repo root
54 passed, 1 xfailed, 2 warnings in 4.43s
```

Per-file: `test_transforms.py` 36, `test_boundary_synthetic.py` 7, `test_analysis_data.py` 6,
`test_experiments.py` 3, `test_hypothesis.py` 3.

Bare `pytest` (not `python -m pytest`) resolves `src.*` imports via the new
`src/tests/conftest.py`, which puts the repo root on `sys.path` — D4's fix. Note the baseline
run above used `python -m pytest`, which inserts the cwd itself, so **I never observed D4
failing in this environment**; the conftest addresses it, but I am not claiming a before/after
measurement I did not take.

The single `xfailed` is `test_dtcwt_is_shift_invariant`, marked `strict=True` — see below.

---

### The transform engine had no tests at all

This is the root cause behind D2, D20, D23 and D25 surviving: `src/tests/` contained no file
covering `transform_engine`, the mathematical core. Every inverse transform was dead code.
`src/tests/test_transforms.py` now covers round-trip exactness across shapes
`{(32,32), (31,33), (64,48)}` and levels `{1,2,3}`, energy conservation, DCT orthonormality,
dtype preservation, the constant-field detail check, and shift behaviour.

```
$ transform round-trip after fixes
shape (32, 32):  dwt L1 MSE=2.68e-14 OK | dtcwt L1 MSE=2.68e-14 OK
                 dwt L2 MSE=4.48e-14 OK | dtcwt L2 MSE=4.48e-14 OK
                 dwt L3 MSE=5.14e-14 OK | dtcwt L3 MSE=5.14e-14 OK
shape (31, 33):  dwt L1 MSE=2.59e-14 OK | dtcwt L1 MSE=2.59e-14 OK
shape (64, 48):  dwt L1 MSE=2.38e-14 OK | dtcwt L1 MSE=2.38e-14 OK
```

### D2 (hybrid inverse) — fixed, T3.5.5

Before (any `mixing_weight`, never a reconstruction):

```
mixing_weight=0.0 -> recon MSE 2.8836e-01
mixing_weight=0.5 -> recon MSE 2.6417e-01
mixing_weight=1.0 -> recon MSE 7.6833e-01
```

After (`exact=True` default, `low + residual`):

```
crossover=0.1 -> MSE 2.851e-14      crossover=0.5 -> MSE 6.710e-15
crossover=0.3 -> MSE 2.195e-14      crossover=0.9 -> MSE 0.000e+00
```

### D1 (fake DTCWT) — confirmed empirically, **NOT fixed** (awaits T3.5.6)

```
LH_real vs LH_imag identical: False
max|LH_AA - LH_AB| = 7.6842942237854
max|LH_AA + LH_BB| = 0.0            <- tree BB is the EXACT negation of tree AA
```

Note also that `dwt` and `dtcwt` reconstruction MSE are **bit-identical** at every shape and
level above. The four "trees" collapse to one filter bank up to sign.

Shift-variance, measured on a sharp front (`width_param=0.01`) translated 0–7 px, level 2:

| Probe structure | Subband energy spread | Worst envelope correlation |
|---|---|---|
| smooth vortex r=0.12 | 0.0% | 0.629 |
| sharp vortex r=0.03 | **126.0%** | 0.140 |
| sharp front w=0.01 | **153.5%** | **−0.109** |
| white noise | 2.2% | −0.135 |
| impulse | 0.0% | −0.004 |

**A methodological note worth keeping.** My first version of the shift-invariance test used a
smooth vortex and total subband energy — and it **XPASSED against the known-broken transform**,
i.e. my "test with teeth" had none. The table above is why: a smooth structure barely changes
under translation (0.0% spread), so the probe structure determines whether the test can detect
anything at all. The committed test uses a sharp front, where the current transform fails by
153% against a 5% threshold. This is R8 in practice, and it cost one wasted iteration to learn.

`test_dtcwt_is_shift_invariant` is committed as `xfail(strict=True)`, so it will **fail the build
the moment T3.5.6 makes it pass**, forcing removal of the marker rather than letting a stale
exemption sit there. A positive control (`test_dwt_shift_variance_is_measurable`) records the
current 150%-ish baseline that T3.5.6 must beat.

---

### Frontend: first build (baseline) — revealed D19

```
$ npm run build
> tsc && vite build
vite v5.4.21 building for production...
✓ 1 modules transformed.
dist/index.html  0.49 kB │ gzip: 0.35 kB
✓ built in 208ms
[exit=0]
```

**Exit code 0, and a completely non-functional application.** One module transformed, no JS or
CSS emitted.

> **D19 — `frontend/index.html` had no `<script type="module" src="/src/main.tsx">` tag.**
> Vite had no entry point, `src/main.tsx` was orphaned, and `App.tsx` was never mounted. The UI
> had never rendered; `npm run dev` would have served a blank page with an empty `<div id="root">`.
> This is more severe than D6 — with Tailwind fixed but no entry point, the page is still blank.
> It also explains why the committed `frontend/dist/` contained only `index.html`.

This is precisely how `architecture.md` came to claim "fully compiled and validated … producing
optimized production bundles cleanly": the build command really does exit 0.

`tsc` passed clean both before and after, so the **TypeScript** half of that claim was accurate.
The build wiring was not.

### Frontend: after fixes

Changes: entry-point `<script>` tag (D19); `tailwind.config.js` and `postcss.config.js` created (D6).

```
$ rm -rf dist && npm run build
✓ 1378 modules transformed.
dist/index.html                  0.65 kB │ gzip:     0.42 kB
dist/assets/index-D3o2cxn4.css  16.38 kB │ gzip:     3.84 kB
dist/assets/index-ca00OUVk.js  5,105.41 kB │ gzip: 1,545.24 kB
✓ built in 1m 4s
[exit=0]
```

The 16.38 kB of CSS confirms Tailwind is compiling — the `@tailwind` directives in
`src/index.css` were previously passed through unprocessed.

**Open follow-up (not blocking):** 5.1 MB uncompressed JS (1.5 MB gzipped), dominated by
`plotly.js-dist-min`; Vite emits a chunk-size warning. Worth `plotly.js-basic-dist` or a dynamic
import before this is production-grade.

---

### Live backend: endpoint smoke test

Server started on port 8123, readiness-polled against `/openapi.json`.

```
GET  /api/v1/data/datasets   -> 200  ['era5_reanalysis', 'gfs_forecast', 'toy_climate_model']
POST /api/v1/transforms/apply
       fft    -> 200  mse=1.78e-15  maxerr=5.96e-08
       dct    -> 200  mse=1.47e-13  maxerr=1.19e-06
       dwt    -> 200  mse=1.48e-14  maxerr=2.38e-07     (previously 500)
       dtcwt  -> 200  mse=1.48e-14  maxerr=2.38e-07     (previously 500)
       hybrid -> 200  mse=6.11e-30  maxerr=7.11e-15     (previously 500 / lossy)
POST /api/v1/boundary/analyze  (treatment=reflect, window=tukey, alpha=0.3) -> 200
```

### Live backend: end-to-end experiment sweep

A 9-run sweep over `gsize ∈ {16,24,32} × tname ∈ {fft,dwt,hybrid}`, pipeline
`generate_synthetic → apply_transform → compute_diagnostics`:

```
create experiment:   HTTP 200  status=PENDING
experiment status:   COMPLETED   runs=9
run statuses:        {'COMPLETED': 9}
sample run {'gsize': 16, 'tname': 'fft'} recon mse=5.67e-15
lineage:             HTTP 200  nodes=28  edges=54
                     types=['code_revision', 'coefficients', 'field', 'metrics']
hypotheses:          HTTP 200  discovered=9
```

Cross-step reference resolution (`{gen.field_data}`, `{gsize}`), parameter expansion, lineage
node/edge emission and the hypothesis engine all work end to end.

### Live confirmation of D8 — no multiple-comparison control

That same 9-run sweep produced **9 hypotheses**, including *"strong positive correlation
(r = 0.96) between parameter 'gsize' and metric …"*. These are artifacts of testing every
parameter against every metric, with no FDR control, on nine samples — correlations between grid
size and floating-point round-off.

It is a live preview of exactly what R5 and T4C.5 exist to prevent, and concrete evidence that
the hypothesis engine must not be pointed at real ERA5 data before they land.

---

## Fixes applied under T3.5.0

| Defect | Fix | Task |
|---|---|---|
| D19 | entry-point `<script>` tag in `index.html` | T3.5.3 |
| D6 | `tailwind.config.js`, `postcss.config.js` | T3.5.3 |
| D20 | lift to NCHW before 4-element `pad` (2 sites) | T3.5.0 |
| D21 | vectorised Tukey taper, no `torch.cos(float)` | T3.5.0 |
| D23 | `_get_haar_filters(device, dtype)`, callers pass field dtype | T3.5.0 |
| D25 | `stack(dim=0)` + crop LL to detail-band shape (2 sites) | T3.5.0 |
| D2 | `inverse_hybrid` returns `low + residual`; lossy blend now opt-in | T3.5.5 |
| D3 | `execute_experiment(..., session_factory=None)` injectable | T3.5.1 |
| D4 | `src/tests/conftest.py` puts repo root on `sys.path` | T3.5.1 |
| D22 | `StaticPool` for the in-memory test engine | T3.5.1 |
| D24 | API background task resolves `app.state.session_factory` | T3.5.0 |
| — | `.gitignore`, `.gitattributes`, `git init` | T3.5.0 |
| — | `src/tests/test_transforms.py` (36 cases, new) | T3.5.0 |

---

## Slice 2 — UI verification + infrastructure blockers

**Date:** 2026-08-19. Suite after this slice: **65 passed, 1 xfailed**.

### T3.5.0 closure: the UI actually serves and mounts

With both servers running (`uvicorn :8000`, `vite :3000`):

```
$ curl -s http://localhost:3000/ | grep -E "script|root"
    <script type="module" src="/@vite/client"></script>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>     <- D19 fix present

$ curl -s http://localhost:3000/src/main.tsx | head -6
    import App from "/src/App.tsx";                          <- module graph resolves
    ReactDOM.createRoot(document.getElementById("root")).render(

$ curl -s http://localhost:3000/src/App.tsx | grep -oE "[0-9]\. [A-Za-z-]+"
    1. Synthetic Generator      5. Diagnostic
    2. Meteorological Data      6. Experiment Engine
    3. Boundary-Condition Lab   7. Automated Hypotheses
    4. Spectral Transforms

$ curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/v1/data/datasets
    200                                                      <- Vite proxy -> FastAPI works
```

All seven tabs are present in the served module, the entry point resolves, and the proxy
path `services/api.ts` relies on returns 200.

**Honest limitation:** this verifies the app is *served and wired*, not that it *renders
correctly*. No browser automation is installed (no Playwright or Puppeteer), so T3.5.0's
"screenshot per tab" criterion is **not** met. Given D19 — where a build exited 0 while
producing a blank page — "serves" and "renders" are different claims and I am not
asserting the latter. Installing Playwright would close this permanently and make it a CI
check; that is a dependency decision for you.

### T3.5.2 — CORS (D5)

Origins come from `CORS_ALLOW_ORIGINS` (comma-separated), defaulting to ports 3000 and
5173 on both `localhost` and `127.0.0.1`.

```
$ curl -D - -H "Origin: http://localhost:3000" .../api/v1/health | grep -i access-control
access-control-allow-credentials: true
access-control-allow-origin: http://localhost:3000

$ curl -D - -H "Origin: http://evil.example.com" .../api/v1/health | grep -ci access-control-allow-origin
0                                          <- unlisted origin is not echoed
```

### T3.5.10 — health and experiment-collection endpoints

```
$ curl -s .../api/v1/health
{"status":"ok","api_version":"1.0.0","database":"ok","database_url_scheme":"sqlite",
 "datasets_available":3,"torch_device":"cpu"}

$ curl -s ".../api/v1/experiments?limit=2"        # after creating 3
list: total=3 returned=2 (page 1)
   0d1fd1fc  COMPLETED    slice2-2
   e2589ce5  COMPLETED    slice2-1
page 2: ['slice2-0']
filter COMPLETED: 3
```

`limit` is clamped to 200 and a negative `offset` to 0, rather than trusted.

### T3.5.9 / E2 — dataset cache invalidation and source transparency (D9)

The adapter now records `{kind, path, mtime}` per dataset and re-resolves whenever the file
appears, disappears or changes, so a newly dropped `.nc` is picked up without an API
restart. More importantly, **the fallback is now a visible provenance fact**:

```
$ curl -s .../api/v1/data/datasets
  era5_reanalysis      kind=simulated  simulated=True  reason=no file at data/era5_reanalysis.nc
  gfs_forecast         kind=simulated  simulated=True  reason=no file at data/gfs_forecast.nc
  toy_climate_model    kind=simulated  simulated=True  reason=no file at data/toy_climate_model.nc
```

Previously nothing in the API or the UI distinguished simulated data from reanalysis — a run
could be satisfied by simulation while the researcher believed they were using ERA5.
`test_cache_invalidation_reresolves` writes a real NetCDF file mid-process and asserts the
adapter switches from `simulated` to `netcdf` with no restart.

Two bugs in my own additions, caught by these tests rather than by inspection:
`ExperimentListResponse` referenced `ExperimentResponse` before its definition (`NameError`
at import), and `get_source_info` short-circuited on the cached `_sources` entry, defeating
the very cache check it was meant to trigger.

### T3.5.4 — launcher and debug config (D7, D11)

D7 reproduced and then fixed, measured directly:

```
--- OLD (no Set-Location), reproduces D7 ---
  ModuleNotFoundError: No module named 'src'
--- NEW (Set-Location $using:ProjectRoot) ---
  import OK; cwd=C:/Users/edjbe/Downloads/Spectral_Research_Accelerator
```

`start_platform.ps1` was rewritten to pin the job's working directory, prefer `.venv` over
system Python, **poll `/api/v1/health` until the backend is genuinely ready**, and abort with
the captured backend output rather than launching the frontend against a dead API. The
duplicate `"name"` key in `.vscode/launch.json` (D11) was removed and the file now parses as
valid JSON.

### T3.5.11 — dependency truth-up (D10)

Added `alembic`, `scipy`, `ipywidgets`, `plotly`, `matplotlib`, and `PyWavelets` as a
test-only reference oracle. `numpy` and `torch` now carry **upper bounds** (`<3.0.0`): D20
and D23 were both caused by unpinned major versions, so this is a direct lesson learned.
Also fixed the pandas `6H` deprecation warning that appeared in every test run and obscured
real warnings.

---

---

## Slice 3 — T3.5.7 undecimated SWT, and T3.5.16 (D16)

**Date:** 2026-08-19. Suite after this slice: **152 passed, 1 xfailed** (up from 65).
New module `src/transform_engine/stationary.py`; 87 new tests in `src/tests/test_stationary.py`.

### Scope decision, stated up front

T3.5.6 (real Kingsbury DTCWT) was deferred to its own slice and **T3.5.7 done first**,
because T3.5.7 delivers the property Phase 4 actually blocks on. DTCWT's distinct benefit
over SWT is *orientation*, which only 4E needs; shift invariance and scales-on-parent-grid
are needed by 4B, 4C and 4D, and SWT gives both. D1 therefore drops from "blocking Phase 4"
to "needed for orientation in 4E".

### The headline result: shift invariance

Same probe as the D1 measurement — a sharp front (`width_param=0.01`) translated 0-7 px,
level 2, envelope realigned before comparison:

| Transform | Subband energy spread | Worst aligned envelope correlation |
|---|---|---|
| decimated DWT (haar) | **153.50%** | −0.1088 |
| undecimated SWT (haar) | **0.00%** | **+1.0000** |
| undecimated SWT (db2) | **0.00%** | **+1.0000** |
| undecimated SWT (db3) | **0.00%** | **+1.0000** |

Exact translation equivariance, not merely "better": circular convolution commutes with a
circular shift, so translating the input translates the coefficients precisely.
`test_swt_beats_dwt_shift_variance` asserts the head-to-head on one field, so the
improvement stays a measurement rather than a claim.

This is what makes Phase 4D feature tracking viable. On the decimated transform a tracker
would report births and deaths caused by grid parity.

### Correctness

All 27 combinations of shape × level × wavelet:

```
(32,32) / (31,33) / (64,48)  x  levels 1..3  x  haar / db2 / db3
  every band shape == parent field shape       PASS (27/27)
  perfect reconstruction MSE                   1e-14 .. 2e-13  (27/27)
```

Tight-frame constant measured at **4.0 to 1e-11** for every wavelet (2 per axis, 4 for the
separable 2D operator), which is the relation the `/4`-per-level synthesis relies on.

`mode="reflect"` is offered for regional crops but is **not** exactly invertible, and
`inverse_swt2d` raises rather than returning a plausible-looking wrong answer.
`apply_swt2d` also refuses a level whose filter support exceeds the field, instead of
silently producing meaningless coarse scales (R13).

### Independent validation against PyWavelets

Our coefficients equal `pywt.swt2(..., norm=True)` multiplied by exactly `2**level`, to
7 significant figures initially and to **1e-11** after the precision fix below. The factor
is asserted in the test, so the convention cannot drift silently. PyWavelets remains a
**test-only** dependency; the filter coefficients are vendored so the runtime does not
depend on it.

### R3 caught in the act — the normalisation trap, measured

Raw versus redundancy-normalised per-level detail energy, same field, same transform
(32×32 noise, db2, 3 levels):

```
raw        : level_1=3302.2  level_2=3191.5  level_3=2739.5   -> reads as flat
normalised : level_1= 825.5  level_2= 199.5  level_3=  42.8   -> clean octave decay
fractions  : level_1=0.7629  level_2=0.1843  level_3=0.0396  LL=0.0132   (sums to 1.0)
```

The a trous construction applies a `2**level` amplitude gain (`4**level` in energy). Raw
energies therefore look almost flat because that gain roughly cancels the physical decay —
**same data, opposite conclusion, purely from the normalisation choice.** A `ScaleSignature`
built on raw energies would be measuring the transform, not the atmosphere.

`swt_scale_energies(normalize=...)` records which convention was used in its own output, and
`swt_energy_fractions()` provides the R3-compliant summary (sums to 1, invariant to a global
rescale — both asserted). This is exactly the trap R3 was written about, now an executable
test (`test_normalisation_changes_the_apparent_scale_distribution`) rather than a warning.

### D16 fixed (T3.5.16), because it was capping validation precision

The tight-frame test first came out as **3.999998860393 for every wavelet** — identical
across three different filters, which is the signature of float32 epsilon rather than any
property of a filter. Cause: `PhysicalField.__init__` called `data.float()` unconditionally,
silently downcasting float64 input.

`PhysicalField(data, dtype=None)` now honours an explicit dtype, preserves a floating input
dtype, and promotes only non-floating input to float32. With float64 preserved, the frame
constant holds to 1e-11 and the PyWavelets cross-check to 1e-11. The full suite passes
unchanged, so nothing depended on the silent downcast.

### Two of my own errors, both caught by tests

*   **Spatial-domain synthesis anchoring.** My first implementation got analysis right
    (energy ratio exactly 4.0, matching the FFT result) but the adjoint wrong — reconstruction
    MSE 7.94. Diagnosed rather than guessed: moving to FFT-domain circular convolution made
    the adjoint unambiguous and PR exact at 4.9e-14. Recorded in the module docstring,
    because it is the class of off-by-one a shape test cannot catch and a reconstruction
    test catches immediately.
*   **Truncated db3 filter literals.** Orthonormality held only to 1e-11, enough to fail the
    db3 oracle cross-check while haar and db2 passed. Fixed by vendoring full double
    precision from PyWavelets: `sum(h**2) - 1 == 0.0` exactly for db2 and db3, 2.2e-16 for
    haar. Filter precision propagates straight into every downstream energy statistic.

Also two test-arithmetic errors of mine: db2 at level 5 leaves an interior of 16 px on a
64 px crop (not 0 — db3 is the case that exhausts it), and the orthonormality tolerance was
initially set tighter than truncated literals could satisfy.

### Reachable, not shelved

`swt` is wired into both `/api/v1/transforms/apply` and the declarative pipeline's
`apply_transform` action, with `levels`, `wavelet` and `mode` as config. Verified end to end:

```
POST /api/v1/transforms/apply  transform_type=swt, levels=2, wavelet=db2
  -> 200, recon MSE < 1e-8, all bands 32x32, energy_fractions sum to 1

declarative sweep over wavelet family {haar, db2, db3}
  -> COMPLETED, 3/3 runs, each recon MSE < 1e-8
```

That last one matters beyond this slice: the wavelet family already works as an ordinary
parameter-matrix entry, which is the mechanism T4B.2's wavelet bank depends on — confirmed
now rather than assumed later.

**Debt incurred, recorded honestly:** adding `swt` meant two more branches on the if/elif
chains that defect D15 is about (one in the engine, one in the API). Both carry a comment
pointing at T3.5.15. This is the last transform that should be added that way.

### Phase 3.5 progress after slice 3

**Fixed: 17 of 25 defects** — D2, D3, D4, D5, D6, D7, D9, D10, D11, **D16**, D19, D20, D21,
D22, D23, D24, D25.

**Tasks complete:** T3.5.0, T3.5.1, T3.5.2, T3.5.3, T3.5.4, T3.5.5, **T3.5.7**, T3.5.9,
T3.5.10, T3.5.11, **T3.5.16**.

**Still outstanding:** D1 (T3.5.6, real DTCWT — now for orientation rather than as a
blocker), D8 (T4C.5, FDR), D12 (seeds), D13 (metric-aware gradients), D14 (error taxonomy),
D15 (registries), D17 (per-bin loops), D18 (device policy); plus T3.5.8 (Alembic),
T3.5.17 (benchmark suite), T3.5.18 (Zarr/ERA5), T3.5.19 (Executor seam), browser-based UI
verification, and wiring health/list into the frontend.

---

## Slice 4 - T3.5.13 metric-aware operators and physical wavenumbers (D13), and D26

**Date:** 2026-08-20. Suite after this slice: **222 passed, 1 xfailed** (up from 152).
New modules `src/physical_core/grid.py`, `src/physical_core/operators.py`,
`src/analysis_engine/spectra.py`; 70 new tests in `src/tests/test_grid_operators.py`.

### What was wrong

Three call sites computed physics in pixel units and said nothing about it:
`torch.gradient` with no `spacing` (diagnostics and boundary lab), radial PSD binned on
integer pixel radius, and every RMSE an unweighted `torch.mean`. On a lat/lon grid the
zonal metric is `R cos(lat) dlon`, so unit spacing is wrong by a latitude-dependent factor -
which distorts gradient *direction* as well as magnitude, and cannot be undone downstream.

### Captured output

```
### metric ###
global cell area sum      : 5.100659e+14   4 pi R^2 = 5.100659e+14   rel err 1.23e-16
32x32 patch at 60N        : aspect 1.7941, dx varies 20.2% (13899 m .. 17019 m)

### gradient: analytic vs unit spacing ###
lat      metric-aware rel err   unit-spacing rel err  
4.0      3.173e-06              2.780e+04             
34.0     3.173e-06              2.568e+04             
64.0     3.173e-06              1.692e+04             
84.0     3.173e-06              8.358e+03             

### gradient convergence order (must quarter when h halves) ###
dlat 0.500 -> 1.269e-05 | 0.250 -> 3.173e-06 | 0.125 -> 7.933e-07   ratios 4.00, 4.00

### spherical Laplacian vs harmonic eigenvalues ###
  l=1  sin(lat)          rel err 9.884e-07
  l=2  3sin^2-1          rel err 1.598e-06
  l=2  cos^2 cos(2lon)   rel err 1.589e-06

### D26: regime classification ###
field built with E(k) ~ k^-5/3 (Kolmogorov)
  OLD (pixel k, S read as E): beta=2.670 -> Charney / Kraichnan 2D enstrophy cascade (E ~ k^-3); mea
  NEW (physical k, E form)  : beta_E=1.666 +/- 0.009 -> Kolmogorov 3D / mesoscale kinetic energy cascade (E ~ k^

### slope recovery ###
  injected 1.6667 -> mean err -0.0024  max |err| 0.0159
  injected 2.0000 -> mean err -0.0022  max |err| 0.0157
  injected 3.0000 -> mean err -0.0014  max |err| 0.0151
```

### D26 - the finding this slice did not set out to make

The task's acceptance criterion required re-validating regime classification. Doing so
revealed that the classifier had been comparing the **wrong quantity** against its reference
values for the platform's entire history.

The annulus mean of `|F(k)|^2` estimates the 2D spectral density `S(k)`. The reference
exponents 5/3 (Kolmogorov) and 3 (Charney/Kraichnan) are defined for the 1D isotropic
**energy** spectrum `E(k)`, and in two dimensions `E(k) = 2 pi k S(k)`. So `E ~ k^-5/3`
means `S ~ k^-8/3 = k^-2.67`, and reading that 2.67 as an energy exponent lands it beside
the Charney value of 3.

Demonstrated by construction above: a field built with a textbook Kolmogorov spectrum was
labelled *"Charney / Kraichnan 2D enstrophy cascade"* - the opposite physical regime. The
new path reports `beta_E = 1.666 +/- 0.009` and names Kolmogorov.

Both conventions are now computed and labelled, interpretation always happens on `E`, and
this is now standing rule **R15**.

### Two of my own claims that measurement refuted

Both had been written into module docstrings as fact before being checked. Recorded in
`architecture.md` 7.2e; the corrected versions are more useful than the originals.

**"Pixel binning biases the fitted slope on an anisotropic grid."** False. For a pure power
law the angular anisotropy factor is independent of radius, so it moves the intercept and
leaves the slope exact - measured slope error at aspect ratio 4 was **+0.001**. What pixel
binning actually destroys is the physical meaning of `k` and the *shape* of the spectrum:

```
Gaussian spectral peak, power-weighted relative spread (3 seeds)
  aspect 1 : physical 0.0224 / 0.0225 / 0.0220   pixel 0.0224 / 0.0222 / 0.0221   (identical)
  aspect 3 : physical 0.0211 / 0.0219 / 0.0225   pixel 0.3649 / 0.3854 / 0.3472   (~16x broader)
```

Since a `ScaleSignature` (T4C.1) is built from spectral features and compared across
latitudes, that is the more damaging failure - and it would have gone unnoticed had the
original claim been accepted.

*Method note on the metric itself:* a half-maximum width was tried first and rejected. Only
2-3 bins clear half maximum, so it swung between 0.07 and 1.94 across random seeds purely on
whether one outlying bin crossed the threshold - it would have made a flaky test that
happened to pass on the first seed tried. The power-weighted second moment uses every bin
and reproduces to ~5%.

**"The Laplacian's `valid_mask` marks the accurate region."** False as first written.
Computing the meridional flux term as gradient-of-gradient widens the stencil to +/-2 rows,
so the first call's first-order edge row contaminated the *second* row: **25% relative error
at row 1**, inside a mask claiming that row was valid, while the true interior held 4e-6.
Replaced with a staggered half-point flux form - compact, one honestly-marked invalid ring,
and conservative. Asserted head-to-head in
`test_staggered_flux_form_beats_nested_gradients_at_the_second_row`.

### Three further defects, all found by tolerance discipline (now rule R16)

*   **D27** - `torch.fft.fftfreq` returns **float32** by default, and a trailing
    `.to(float64)` preserves an already-rounded value. This capped the Parseval identity at
    a systematic `1 - 5.78e-8` for *every* field - a constant, not noise, which is what gave
    it away. Half of float32 epsilon. Same class as D16. After the fix the identity is
    `1.000000000000000`. **Invisible to any tolerance looser than 1e-7.**
*   **D28** - the polar-degeneracy guard tested `dx_metres > 0`. `cos(90 degrees)` is
    6.1e-17 in float64, not 0, so a polar row had a zonal spacing of ~1e-12 m: finite,
    passing the guard, and yielding gradients of order **1e11 per metre** with no NaN
    anywhere to flag them. Now guarded against a physical length threshold.
*   **D29** - found by exercising the API rather than the library: it reported
    `k_units = "rad m^-1"` for a field carrying no physical metric at all. A plot axis in
    metres over data that never had metres is D13's own failure mode reappearing one layer
    out. Unit labels now derive from the grid kind (`rad pixel^-1`), and kilometre-based
    units on a pixel grid are refused with an explanation.

### Tolerance discipline, stated as method

Four tests assert what the mathematics predicts rather than what the code can pass:

*   Parseval to **machine epsilon**, because the identity is exact in exact arithmetic.
    This is what exposed D27.
*   The Laplacian residual must **equal** the predicted truncation `(k h)^2 / 12` to within
    10%, so an error of the right size for the wrong reason still fails.
*   The gradient is validated by **convergence order** - halving the spacing quartered the
    error, measured ratios `4.00, 4.00`. That is what allows the residual 3.2e-6 to be
    attributed confidently to truncation rather than to a metric error.
*   Global cell areas must reproduce `4 pi R^2`, an external constant the code cannot
    influence.

Four of my own test tolerances were wrong on first run and were corrected *upward toward
theory*, not loosened to fit: the meridional gradient tolerance was tighter than
second-order truncation allows, the dtype test differenced two shifted axis entries and so
measured catastrophic cancellation rather than precision, the Cartesian Laplacian threshold
sat just below its own predicted truncation, and the Hann-window check was applied to a red
field where the `mean(w^2)` correction's stationarity assumption does not hold (that caveat
is now asserted explicitly rather than papered over).

### Also delivered

*   **Partial T3.5.20 (D17).** The radial-PSD and coherence annulus reductions are now
    `torch.bincount` single-pass reductions, asserted numerically identical (rtol 1e-12) to
    a reference Python loop written inside the test.
*   **Grid provenance (E5).** Every `PhysicalField` carries a `GridSpec`, which propagates
    correctly through `scale_resolution` (endpoint-preserving, so spacing scales as
    `(n-1)/(m-1)` and not `n/m`) and `split_field` (origin moves, so a crop at a different
    latitude gets its own metric). It round-trips to and from a JSON record, and a grid that
    does not match its field's shape is rejected with a message naming `grid.subset` /
    `grid.resampled` as the fix.
*   **Uncertainty on every exponent (R2).** `fit_power_law` returns a standard error from
    count-weighted least squares, and `_interpret` names a turbulence regime only when the
    fit is tight *and* the reference exponent is within tolerance - a peaked, non-power-law
    spectrum is reported as *"No single power law"* rather than assigned a cascade.
*   **Declared estimator choices (E8).** Count weighting is justified from the chi-squared
    variance of an annulus average; the `log(m) - psi(m)` bias correction is available but
    **off by default** as a measured trade-off rather than a silent improvement.

### Suite after slice 4

```
222 passed, 1 xfailed, 1 warning

  test_stationary.py        87
  test_grid_operators.py    70   (new)
  test_transforms.py        36
  test_api_infrastructure.py 11
  test_boundary_synthetic.py  7
  test_analysis_data.py       6
  test_experiments.py         3
  test_hypothesis.py          3
```

**Fixed: 21 of 29 defects** - D2, D3, D4, D5, D6, D7, D9, D10, D11, **D13**, D16, D19, D20,
D21, D22, D23, D24, D25, **D26**, **D27**, **D28**, **D29** (D26-D29 were found *during*
this slice).

**Still outstanding:** D1 (T3.5.6, real DTCWT), D8 (FDR), D12 (seeds), D14 (error taxonomy),
D15 (registries), D17 (remaining per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (device policy); plus T3.5.8 (Alembic), T3.5.17
(benchmark suite), T3.5.18 (Zarr/ERA5), T3.5.19 (Executor seam), browser-based UI
verification, and wiring health/list into the frontend.

---

## Slice 5 - T3.5.17 Ground-Truth Benchmark Suite, and T3.5.12 seed discipline

**Date:** 2026-08-20. Suite after this slice: **271 passed, 1 xfailed** (up from 222).
New package `src/benchmarks/` (registry, seeding, 9 datasets, runner, CLI) and
`src/analysis_engine/climatology.py`; 46 new tests in `src/tests/test_benchmarks.py`.

### What this is for

A discovery tool without a false-positive floor is a machine for generating confident
nonsense. Real atmospheric data can never tell you whether a reported pattern is real, so
the platform is now measured against data whose answer is known *before* the analysis runs -
and five of the nine entries have the answer **"there is nothing here"**.

### Captured output

```
Ground-Truth Benchmark Suite
==============================================================================

advected_vortex_sequence
  One vortex on a known trajectory with a known scale-doubling time.
    PASS              4D.position                      worst centroid error 0.052 cells over 24 frames (mean 0.021); target < 1 px
    PASS              4D.scale_evolution               scale centroid rose from level 4.366 to 4.915 (rank correlation with time 0.971) as sigma doubled every 16 steps
    NOT_YET_RUNNABLE  4D.tracking                      4D.tracking: SpectralFeatureTrack linking (T4D) is not implemented; the known answer (1 track, 1 birth, 0 deaths, exact positions) is recorded

coupled_cascade_sequence
  Fine-scale activity drives coarse-scale amplitude at a known lag.
    PASS              4C.cross_scale                   peak cross-correlation at lag 6 (true 6), r = 1.004; level 1 leads level 4

fractional_brownian  [NULL - correct answer is 'nothing']
  Scale-free fBm with a known Hurst exponent; exact alpha, NO organisation.
    PASS              4C.alpha                         recovered E(k) exponent 2.3831 +/- 0.0093, true 2.4000 (H = 0.70), tolerance 0.060
    NOT_YET_RUNNABLE  4C.surrogate_null                4C.surrogate_null: phase-randomised surrogate testing (T4C.5) is not implemented; the known answer 'zero findings on fBm' is recorded and will be enforced then

planted_configuration
  Three features in a known equilateral triangle; the 4E invariance target.
    PASS              4E.feature_detection             all 3 planted features present at their stated positions: [True, True, True] (peak 1.082)
    NOT_YET_RUNNABLE  4E.invariance                    4E.invariance: constellation matching with relative geometry (T4E) is not implemented; the transformed variants and their expected match are recorded

pure_noise_sequence  [NULL - correct answer is 'nothing']
  Independent white-noise frames. NOTHING is present; every stage must agree.
    PASS              4C.cross_scale.null              best of 10 lags: p = 0.0847 at lag 10 (threshold 0.0050 after correcting for 10 tests); no cross-scale coupling claimed

pure_sinusoid
  A single spatial frequency; all energy belongs at one scale.
    PASS              4C.scale_signature               spectral peak at 1.2954e-05 rad/m vs true 1.2668e-05 (2.3% off); wavelength 485 km vs 496 km
    PASS              4C.scale_signature.wavelet       dominant SWT detail level 3 (expected 3); energy fractions {1: 0.0042, 2: 0.0578, 3: 0.469, 4: 0.469, 5: 0.0}

red_noise_sequence  [NULL - correct answer is 'nothing']
  Temporally autocorrelated but independent; catches R12 violations.
    PASS              4C.r12_effective_sample_size     phi = 0.85, n = 200, ESS = 32.2. False-positive rate at alpha = 0.05: naive 40.0% (should be badly inflated), ESS-corrected 2.3% (should be near 5%)

seasonal_diurnal_sequence  [NULL - correct answer is 'nothing']
  Deterministic cycles, no weather. The most likely false discovery on ERA5.
    PASS              4C.r11_anomaly                   residual variance after harmonic declimatology: 0.000171 of raw (target < 0.05). A time-of-day BIN climatology leaves 0.1552, because a 40-day record cannot form a day-of-year climatology and the annual cycle passes straight through.
    PASS              4C.r11_split_aware               climatology fitted on 96 of 160 frames still removes the cycles on the held-out frames: residual 0.037306 of raw variance
    PASS              4C.r11_raw_is_deceptive          raw (un-anomalised) scale energies correlate at r = 0.999, p = 2.02e-219 - a strong 'finding' that is purely the calendar. This is the trap R11 exists for.

white_noise_field  [NULL - correct answer is 'nothing']
  Spatially uncorrelated noise; the single-field false-positive floor.
    PASS              4C.alpha                         E(k) exponent -0.9805 (expected -1.0 for white noise); label: Flat spectrum: E(k) exponent -0.98 +/- 0.01. Consistent with
    PASS              4C.false_positive                no cascade regime claimed; reported as 'Flat spectrum: E(k) exponent -0.98 +/- 0.01. Consistent with'

------------------------------------------------------------------------------
PASS 14   FAIL 0   NOT_YET_RUNNABLE 3

Gates defined but not yet enforceable (the stage does not exist yet):
    advected_vortex_sequence / 4D.tracking
    fractional_brownian / 4C.surrogate_null
    planted_configuration / 4E.invariance
```

The three pending entries gate stages that do not exist yet. They report
`NOT_YET_RUNNABLE` naming the missing stage rather than being skipped, because a skipped
test is invisible in a summary line and would let "all green" mean "we never looked".

### R12 measured: frames are not samples

On 200-frame AR(1) sequences with `phi = 0.85` (effective sample size **32.2**, from 200
frames), correlating *independent* series:

| test | false-positive rate at alpha = 0.05 |
|---|---|
| naive, treating frames as independent | **40.0%** |
| effective-sample-size corrected | **2.3%** |
| nominal | 5% |

An eight-fold inflation. Any 4C-4F result computed without this correction would be mostly
false positives, and the benchmark measures the effect rather than asserting it.

### R11 measured, and it forced a real module to be written

The seasonal+diurnal benchmark contains *only* deterministic cycles plus 0.05-amplitude
noise. The obvious anomaly step - average all frames sharing a time of day, subtract - left
**15.5% of the original variance**, because a 40-day record cannot form a day-of-year
climatology and the annual cycle passed straight through. That 15.5% is exactly what a
pattern miner would report as weather.

This was a benchmark failure that could not be fixed by adjusting the benchmark, so
`analysis_engine/climatology.py` was written: harmonic regression on the diurnal and annual
periods, which fits the cycles as continuous functions of time and therefore removes a
*partial* annual cycle from a *partial* year.

```
residual variance as a fraction of raw:
  time-of-day bin climatology : 0.155193
  harmonic regression         : 0.000171     (a factor of ~900)
```

The suite also asserts the **trap itself**: the raw sequence's scale energies correlate at
`r = 0.999, p = 2e-219`. Without that positive control the anomaly check would prove
nothing, because a benchmark that never produced a spurious finding could not demonstrate
one being removed.

Climatology fitting is **split-aware** (R6): fitted on the first 60% of frames it still
removes the cycles on the held-out 40% (residual 0.037 of raw), so anomalies can be produced
without leaking test-period information into training.

### D30 - a result that changed depending on what ran before it

The most serious defect found this slice, and it surfaced only because the tests ran in a
different order than the ad-hoc script had.

`test_harmonic_declimatology...` passed alone and failed in the full module. Same data, same
seed, no randomness involved. Bisecting showed that running *any other benchmark first*
changed the answer:

```
residual variance ratio, identical inputs:
  run alone                                      0.000172
  after test_benchmarks_are_deterministic...     0.157639
  after test_fbm_slope_follows_two_h_plus_one    0.157639
```

Cause: the harmonic design matrix for a short record is near rank-deficient - condition
number **8.44e13**, smallest singular value **2.5e-13** - because the second annual harmonic
over 40 days is numerically indistinguishable from a combination of the other columns.
`torch.linalg.lstsq`'s default driver makes its own rank decision there, and that decision
flipped with prior BLAS state.

A climatology whose result depends on the order of unrelated work is not reproducible, and
this would have been essentially undiagnosable had it first appeared in a Phase 4 result.
Fixed by making the rank decision ours: columns normalised (condition number **8.44e13 ->
1.85e4** from that alone), solved through an SVD pseudo-inverse with an explicit
`rank_rtol`, with effective rank and condition number reported and a warning naming
unidentifiable components. The regression test deliberately performs unrelated `svd` and
`lstsq` work first.

**Method note.** This is the second time this slice that ordering or context, not the
mathematics, produced a wrong answer - and both were caught by an *incidental* difference in
how the code was exercised. Reproducibility here means bit-identical under reordering, not
merely "seeded".

### D31 - a gate a typo could delete

`POST /api/v1/benchmarks/run?name=<typo>` filtered the suite to nothing and returned HTTP
200 with zero failures. Read as a summary, that is "everything passed". Now a 404 listing
the available benchmarks.

### T3.5.12 seed discipline

`benchmarks/seeding.py` derives independent `torch` and `numpy` streams from a root seed via
`SeedSequence.spawn`. Three specific traps are avoided and documented:

*   **Python's `hash()` on a string is salted per process.** A label-derived seed using it
    reproduces within one session and silently changes between sessions - the worst kind of
    reproducibility bug, since every test passes. `zlib.crc32` is used instead, and the test
    asserts a *literal* hash value so a regression fails on the second machine rather than
    producing different data.
*   **Sequential seeds are not independent seeds.** `SeedSequence.spawn` is what will make a
    `process`-backend sweep (T3.5.19) agree with a serial one.
*   **torch and numpy are separate streams**, so both are derived together.

`PerturbationEngine.add_noise` previously used `torch.randn_like`, drawing from global state:
a perturbation experiment could not be re-run to the same numbers from its lineage record,
and sweep results would have depended on executor interleaving. It now takes `seed=` or a
threaded `generator=`, refuses both at once, preserves the field's `GridSpec`, and labels an
unseeded run `seeded: False` so it is visibly distinguishable from a reproducible one.

Both halves of the acceptance criterion are asserted: same seed reproduces bit-for-bit, and
*different* seeds actually differ - the second because the first passes trivially for a
generator that ignores its seed entirely.

### Suite after slice 5

```
271 passed, 1 xfailed, 1 warning

  test_stationary.py         87
  test_grid_operators.py     70
  test_benchmarks.py         46   (new)
  test_transforms.py         36
  test_api_infrastructure.py 15
  test_boundary_synthetic.py  7
  test_analysis_data.py       6
  test_experiments.py         3
  test_hypothesis.py          3
```

**Fixed: 24 of 31 defects.** New this slice: **D30**, **D31**; **D12** closed for generation
and perturbation.

**Still outstanding:** D1 (T3.5.6, real DTCWT), D8 (FDR), D14 (error taxonomy), D15
(registries), D17 (remaining per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (device policy); plus T3.5.8 (Alembic), T3.5.18
(Zarr/ERA5), T3.5.19 (Executor seam, which also completes D12's lineage half), browser-based
UI verification, and surfacing the benchmark suite in the frontend.

---

## Slice 5b - Documentation audit: the docs had drifted, and now they cannot

**Date:** 2026-08-20. Suite: **286 passed, 1 xfailed** (up from 271).
New file `src/tests/test_documentation.py` (15 tests).

### The question, and the honest answer

Asked directly whether `architecture.md` and `roadmap.md` were fully up to date with what
had actually been implemented, I audited them mechanically instead of answering from memory.
They were **not**. Six concrete failures:

| Document | Stale claim | Reality |
|---|---|---|
| `architecture.md` §7.1 | "**152 passed**, 1 xfailed" | 271 - stale by two slices |
| `architecture.md` §6 | "the API declares no CORS middleware" | Added in T3.5.2 (D5). It also **contradicted a sentence two lines below it** |
| `architecture.md` §6 | frontend "has never been installed or built" | Built in T3.5.0/T3.5.3 |
| `architecture.md` §3 | module assessment 3.1-3.7 | **Five modules absent** (~1,800 lines): `grid.py`, `operators.py`, `spectra.py`, `climatology.py`, `benchmarks/` |
| `architecture.md` | no endpoint inventory | **10 of 17 routes** unmentioned |
| `roadmap.md` §1 | "Honest Technical Status" | The **pre-Phase-3.5 baseline snapshot** presented as current: "No CORS, no collection endpoint, no health endpoint", "Never installed or built", "19 tests written; at least one fails", "Anything ever executed here: **Nothing**", "ledger (D1-D11)" |

The pattern is clear and worth naming: **the new material was accurate every time, and the
old material was never re-audited.** Appending a correct section to a document does not make
the document correct. Every slice added an honest new subsection while leaving contradicting
older paragraphs in place, and nothing detected it because nothing tested the documents.
Prose keeps "passing" no matter how wrong it gets.

### What was fixed

*   `architecture.md` gains §3.8 (grid + operators), §3.9 (spectra), §3.10 (climatology),
    §3.11 (benchmarks) and **§3.12, a full 17-route API inventory**.
*   §6's CORS and never-built claims corrected; the genuinely unverified part - the
    frontend's *rendered appearance in a browser* - is now stated precisely rather than
    being overstated in one direction and understated in the other.
*   §7.1 execution status brought current, with the full trajectory (19 -> 65 -> 152 -> 222
    -> 271) rather than a single number that goes stale on the next commit.
*   §7.4 test inventory added, counted from the AST.
*   `roadmap.md` §1 rewritten as current status, with the original baseline audit explicitly
    relabelled as history rather than deleted.

### The actual fix: the documents now fail like code

`src/tests/test_documentation.py` derives facts from the source and asserts the docs match:

*   every module under `src/` appears in `architecture.md`;
*   every `@app.route` appears, and the stated route count matches the served count;
*   the per-file test inventory matches an **AST count** of `def test_` (so it cannot be
    fudged), including the total;
*   every referenced defect ID is defined, IDs are contiguous (a gap means an entry was
    deleted rather than resolved), and every `FIXED` entry names the task that fixed it;
*   `T3.5.x` sections are in numeric order with no duplicates, and R-rules are contiguous -
    both drifted repeatedly in earlier slices and needed a repair script each time;
*   every task marked `DONE` carries an evidence block;
*   the benchmark PASS/FAIL/PENDING triple in the docs matches a **live run**;
*   the two specific contradictions found above cannot return.

**Verified to have teeth**, by injecting each defect and confirming the failure:

```
inject a stale count   -> AssertionError: inventory is stale (documented, actual):
                          {'test_benchmarks.py': (999, 45)}
remove a module        -> AssertionError: these modules exist but are not mentioned
                          in architecture.md: src/analysis_engine/spectra.py
restore the CORS claim -> 1 failed
```

A guard that has never been seen to fail is not known to work - the same reasoning that
made `test_swt_beats_dwt_shift_variance` a head-to-head measurement rather than an assertion.

### One test of mine was wrong on its first run

`test_no_unqualified_validated_claims` checked line by line and flagged the sentence
*"Earlier revisions ... claimed the platform was 'validated' and 'zero-error'. It was not"* -
a sentence explicitly disowning the claim - because the disqualifying context sat on the
preceding line. Fixed to check per paragraph. Recorded because a documentation guard that
cries wolf gets disabled, which would be worse than not having it.

### Suite after slice 5b

```
286 passed, 1 xfailed
190 test functions across 10 files (pytest reports more cases; several are parametrised)
```

---

## Slice 6 - T3.5.6 real Kingsbury q-shift DTCWT (defect D1 closed)

**Date:** 2026-08-20. Suite after this slice: **351 passed, 1 xfailed** (up from 286).
New modules `src/transform_engine/dtcwt.py` and `kingsbury_coeffs.py`, generator
`tools/gen_kingsbury_coeffs.py`; 65 tests in `src/tests/test_dtcwt.py`.

**D1 is the oldest entry in the ledger and the roadmap's stated highest-risk item.** It
survived six slices because the only test covering it was a round trip - and a round trip
provably cannot see it.

### Captured output

```
### shift invariance, level 2, sharp periodic front, 0-8 px ###
  undecimated SWT (exact)      :     0.00%
  NEW DTCWT (Kingsbury q-shift):     4.99%
  OLD 'DTCWT' (D1 artefact)    :   236.11%
  plain Haar DWT               :   236.11%
  -> the old transform is numerically the DWT (difference 0.00%)

### orientation: measured passband centres (impulse response) ###
  subband  declared  measured(L3)  error
     0        75.0        74.7     0.35
     1        45.0        45.0     0.00
     2        15.0        15.3     0.35
     3       165.0       164.7     0.28
     4       135.0       135.0     0.00
     5       105.0       105.3     0.28

### +45 vs -45 discrimination (why this module exists) ###
  SWT single HH band: +45 energy 2.1722e+04, -45 energy 2.1722e+04  (differ by 0.00%)
  DTCWT: +45 peaks in subband 1, -45 peaks in subband 4  -> separated

### oracle agreement and reconstruction ###
  worst elementwise disagreement vs dtcwt oracle : 2.19e-15  (20 shape/level combos)
  worst reconstruction MSE                       : 1.20e-31
```

### The defect, finally quantified

The old `apply_dtcwt2d` scores **236.11%**, and a plain Haar DWT on the identical field
scores **236.11%** - a difference of 0.00%. The "dual-tree complex wavelet transform" was
numerically indistinguishable from a single real Haar DWT. Every pair of its four "trees" is
identical up to sign (`max|A - B| = 0` or `max|A + B| = 0` to 1e-16).

It reconstructed perfectly the whole time, because four copies of one transform average back
to the input. That is the cleanest illustration of rule **R8** the project is likely to
produce: *an inverse test validates a transform against itself and can be satisfied by a
transform that does nothing it claims to do.*

### The adopt-vs-build decision, made explicitly

Deferred two slices ago, now resolved as **build, vendor the coefficients, keep both
packages as test-only oracles**:

*   `pytorch_wavelets` imports `pkg_resources`, an API already past its announced removal
    date. Taking a *runtime* dependency on a deprecated loader - for what are ultimately
    ~80 constants - is poor stewardship for a platform meant to outlast its dependencies.
*   Two independent oracles are worth far more as **validators** than one is as an
    implementation. Vendoring is what keeps that check honest: importing the filters from an
    oracle would make agreement partly circular.

Coefficients are generated once by `tools/gen_kingsbury_coeffs.py` at full `repr()`
precision, carry their provenance in the module, and a test **parses the imports** to assert
neither oracle is reachable from the runtime.

### Method: incremental validation against the reference

The transform was built primitive by primitive, each checked against the reference before
the next was written - `colfilter`, `coldfilt`, `colifilt`, `q2c`, `c2q`, all matching to
1e-14 or exactly. Then the full forward transform matched **elementwise to 2.19e-15 across
20 shape/level combinations**.

That discipline paid off immediately: with the forward transform provably exact, the first
reconstruction attempt still gave MSE ~0.5. Because the forward was already validated, the
fault was necessarily in synthesis, and it was a **swapped `lh`/`hl` pairing**. Perfect
reconstruction to **1.20e-31**. Note what would have happened otherwise: every shape was
correct and the forward matched an oracle, so only a round trip could catch it - the exact
mirror of D1, where only a *property* test could catch what the round trip missed. Neither
test class subsumes the other.

### Two of my own errors, both caught by measurement

**The orientation table was in the wrong order.** The first draft declared the canonical
angles ascending, `(15, 45, 75, 105, 135, 165)`. Those are the right six angles assigned to
the wrong six subbands - it would have mislabelled every orientation Phase 4E ever reported
while looking entirely reasonable. Caught by measuring each subband's passband centre
directly (impulse response -> FFT -> energy-weighted axial mean); the true wavevector order
is **(75, 45, 15, 165, 135, 105)**, now verified to within 0.35 degrees. Feature orientation
and wavevector orientation are kept as two separately named constants, because silently
mixing them is a 90-degree error that reads like a sign convention.

**The acceptance criterion's own probe is too weak.** T3.5.6 specifies "translate a synthetic
vortex 0-8 px". On a smooth vortex, total subband energy is flat at **0.00% for every
transform tested** - including the broken one and a plain Haar DWT. A test built to that
criterion would have passed against a transform with no shift invariance whatsoever. This is
the *same trap* that appeared in T3.5.7, where an early test XPASSed against the defective
transform for exactly this reason. A periodic sharp front is used instead, and the weak probe
is now itself an executable test so it cannot quietly return.

Also worth recording: a first "sharp front" probe used `torch.roll` on a non-periodic ramp,
which puts a large step at the wrap point and gave 97.97%. The probe was wrong, not the
transform.

### Scope, stated honestly

The DTCWT is **near** shift invariant (4.99%), not exact. The undecimated SWT is exact
(0.00%) and remains the right tool for Phase 4D tracking. The DTCWT's distinct contribution
is **orientation**, which a separable transform cannot supply: the SWT's single diagonal band
gives *identical* energy for +45 and -45 degree gratings (differing by 0.00%), while the
DTCWT places them in different subbands. That is what Phase 4E needs to describe a front's
orientation, and it is why D1 was correctly downgraded from a Phase 4 blocker to a 4E
requirement once the SWT landed - rather than being quietly forgotten.

### The artefact is retained, and fenced

The broken implementation stays in `transforms.py` so the head-to-head regression tests keep
their comparison arm - a fix that cannot be demonstrated against the thing it fixed is a
weaker fix. It carries an unmissable docstring, no runtime path calls it, and
`test_no_runtime_code_uses_the_degenerate_dtcwt` **parses the AST** of every module under
`src/` to enforce that. The strict `xfail` in `test_transforms.py` now guards the artefact:
if it ever XPASSes, the artefact has been altered and the regression tests have stopped
proving anything.

That guard found a real leftover on its first run. Its first version used a substring search
and flagged the *comment in `engine.py` explaining why the artefact is no longer used* - the
same "the search matches its own explanation" failure the documentation guard hit in slice
5b, and the second time in two slices that grepping prose had to become parsing code.

### Reachable

`dtcwt` is wired into the declarative engine and `POST /api/v1/transforms/apply`, with
`level1` and `qshift` read from `config` so the filter family is an ordinary
parameter-matrix entry - the mechanism T4B.2's wavelet bank needs. All four q-shift sets and
all three level-1 sets reconstruct through the API. The stale serialiser that emitted
`_real`/`_imag` arrays holding identical data was removed; the payload now carries the
oriented energy summary, both orientation conventions and the coefficient provenance.

### Suite after slice 6

```
351 passed, 1 xfailed
218 test functions across 12 files
```

**Fixed: 25 of 31 defects**, including **D1**.

**Still outstanding:** D8 (FDR), D14 (error taxonomy), D15 (registries - now carrying a
`dtcwt` branch as well), D17 (remaining per-bin loops), D18 (device policy); plus T3.5.8
(Alembic), T3.5.18 (Zarr/ERA5), T3.5.19 (Executor seam), browser-based UI verification, and
surfacing the benchmark suite and orientation output in the frontend.

---

## Slice 7 - T3.5.15 registries and T3.5.14 error taxonomy (D15, D14)

**Date:** 2026-08-20. Suite after this slice: **379 passed, 1 xfailed** (up from 351).
New modules `src/core/registry.py`, `src/core/errors.py`,
`src/transform_engine/registry.py`, `src/experiment_engine/actions.py`,
`src/data_layer/sources.py`, `src/data_layer/builtin_sources.py`, plus the worked example
`src/tests/plugin_example.py`; 28 tests in `src/tests/test_registries.py`.

### Captured output

```
### dispatch chains removed ###
  src/api/main.py                    elif transform_type: 0   elif action ==: 0   lines: 1030
  src/experiment_engine/engine.py    elif transform_type: 0   elif action ==: 1   lines: 295

### registries ###
  transforms: ['dct', 'dtcwt', 'dwt', 'fft', 'hybrid', 'swt']
  actions   : ['analyze_boundary', 'apply_transform', 'compute_diagnostics', 'decompose_errors', 'generate_synthetic', 'perturb_field', 'slice_dataset']
  sources   : [('netcdf_local', 10, False), ('simulated', 900, True)]
  by capability -> shift_invariant: ['swt'] | oriented: ['dtcwt']

### fallback chain provenance ###
  netcdf_local  ok=False declined: no file at data\era5_reanalysis.nc - drop a NetCDF file 
  simulated     ok=True  served
  fallback_reason: declined: no file at data\era5_reanalysis.nc - drop a NetCDF file there to

### error taxonomy through the API ###
  unknown transform -> 404  Unknown transform 'dwt2'. Did you mean 'dwt' or 'dtcwt'? Available: dct, dtcwt
  GET /api/v1/actions    -> 7 entries
  GET /api/v1/transforms -> 6 entries
  GET /api/v1/data/sources -> 2 entries
  fft     200  MSE 2.73e-14
  dct     200  MSE 5.33e-12
  dwt     200  MSE 4.05e-14
  swt     200  MSE 1.40e-13
  dtcwt   200  MSE 1.04e-31
  hybrid  200  MSE 7.17e-15
```

### The acceptance criterion, executed literally

T3.5.15's acceptance is a claim about *files*: "a new data source and a new pipeline action
are each added in a **new file only**, with zero edits to `engine.py`, `adapters.py` or
`main.py`". A claim about not editing files is checkable, so `test_acceptance_new_plugin_
needs_no_core_edits` **hashes all three before and after** importing the plugin, rather than
asserting it in prose. Both halves pass.

**And the acceptance test immediately found a real latent bug.** `list_datasets` computed
`is_simulated` as `kind == "simulated"` - a literal string comparison. The demo source
declares `kind="demo", is_simulated=True`, and was reported to the researcher as **real
observational data**. Any future simulated source not labelled with that exact string would
have been mislabelled the same way. It now reads the flag the source declared. This is
precisely the failure mode D15 is about: behaviour keyed on hard-coded literals rather than
on a declaration, invisible until a third case exists.

### What was removed

```
                          before          after
engine.py                 562 lines       295 lines
  if/elif on action       3 chains        0
  if/elif on transform    1 chain         0
main.py
  if/elif on transform    1 chain         0
```

The two transform chains mattered because they were *duplicates*: `engine.py` and `main.py`
each had their own six-branch dispatch with independently written defaults, so the same
config could mean two different things depending on which entry point ran it. There is now
one definition, and a test asserts the API contains no such chain.

Three separate chains listed the seven action names - dispatch, node type, and lineage
summary - and all three had to be edited together by hand. Each action now declares all
three facts in one registration.

### A fallback is a scientific fact

`resolve()` returns the whole attempt chain, and sources that **decline** are recorded with a
reason, not only those that fail:

```
netcdf_local  ok=False  declined: no file at data/era5_reanalysis.nc - drop a NetCDF file there...
simulated     ok=True   served
```

That distinction was found by a failing test. A source that merely declines - `can_serve`
returns False because no file is on disk - leaves *no failed attempt behind*, so the first
implementation recorded `fallback_reason: None` while still serving fabricated data. The
`why_not()` hook lets a source explain its decline, which turns "simulated data was used"
into "no file at `data/era5_reanalysis.nc` - drop a NetCDF file there". One states a fact;
the other names the fix.

### Errors: the kind decides the status

Previously every failure became `500 An internal error occurred`, which tells a researcher
nothing and reports their typo as our fault. Now:

```
POST /transforms/apply {"transform_type": "dwt2"}
  -> 404  Unknown transform 'dwt2'. Did you mean 'dwt' or 'dtcwt'?
          Available: dct, dtcwt, dwt, fft, hybrid, swt.
```

while a genuine fault stays opaque - asserted, including that a 500 raised from a path like
`/secret/path/to/model.ckpt` does not echo it. `PipelineStepError` **inherits its cause's
status code**, so a bad parameter inside a step is still a 4xx rather than being promoted to
a platform fault by the wrapping.

### Capability queries, which is the point for Phase 4

Transforms declare properties, not just names:

```
shift_invariant -> ['swt']        oriented -> ['dtcwt']        tag wavelet_bank -> both
```

T4B.2's wavelet bank can therefore ask for "every shift-invariant multiscale transform"
instead of hard-coding a list, and a transform joins the bank by being registered. The same
mechanism selects data sources: `SOURCES.with_capability("streaming")` is how T3.5.18's Zarr
adapter will slot in, at priority ~20 in the deliberately wide gap between `netcdf_local`
(10) and `simulated` (900).

### Method note

The seven action bodies were moved by a script that split `engine.py` on the branch
boundaries and dedented, rather than by hand - 267 lines of mechanical edit is exactly where
hand-editing introduces a silent error. The dedent was wrong on the first run (8 spaces
instead of 4) and failed loudly at parse time, which is the desired failure. The real check
is that **all 350 pre-existing tests passed unchanged afterwards**: this slice changed
dispatch, not behaviour, and that is what verifies it.

Also, for the third time in three slices, a guard written with a substring search matched its
own explanatory comment - here the `elif action ==` count found the phrase inside the
docstring describing its removal. Grepping prose keeps having to become parsing code.

### Suite after slice 7

```
379 passed, 1 xfailed
246 test functions across 13 files
```

**Fixed: 27 of 31 defects**, adding **D14** and **D15**.

**Still outstanding:** D8 (FDR), D17 (remaining per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (device policy); plus T3.5.8 (Alembic), T3.5.18
(Zarr/ERA5), T3.5.19 (Executor seam - which also completes D12's lineage half and T3.5.14's
sweep-level failure reporting), browser-based UI verification, and surfacing the benchmark
suite, orientation output and discovery endpoints in the frontend.

---

## Slice 8 - T3.5.19 Executor seam, T3.5.21 device policy (D12 completed, D18 partial)

**Date:** 2026-08-20. Suite after this slice: **407 passed, 1 xfailed** (up from 379).
New modules `src/core/executor.py`, `src/core/device.py`; 28 tests in
`src/tests/test_executor.py`.

### Captured output

```
### machine ###
  cpus: 12   torch: 2.13.0+cpu   device: cpu   sqlite: wal

### T3.5.19 acceptance: 8-run sweep, byte-identical across backends ###
  serial   w=1   1.60s   8 runs, all COMPLETED=True   (reference)
  thread   w=2   0.24s   identical to serial: True
  thread   w=4   0.22s   identical to serial: True
  process  w=2   5.55s   identical to serial: True
  process  w=4   8.05s   identical to serial: True
  process  w=8  10.47s   identical to serial: True

### why serial is the default: intra-op threading already fills the cores ###
  torch threads   serial    thread(4)      process(4)
  1               2.06s     1.47s  1.40x    6.11s  0.34x
  12              0.78s     1.39s  0.56x    5.77s  0.14x

### fixed cost of the process backend on this platform (spawn) ###
  8 trivial tasks: serial 0.009s   process(4) 6.32s   -> ~6.3s worker startup
```

### The acceptance criterion, met

Byte-identical results across **six** backend configurations, compared on the persisted
parameters, status, seed and results of **every** run - not on a summary statistic, which
could agree while individual runs differed. Plus the concurrency stress: 8 threads x 25
commits = 200 rows, **zero** `database is locked`, with SQLite in WAL mode and a 30 s
`busy_timeout` applied per connection.

What makes the identity hold is a structural split, not a tolerance. `execute_run_payload`
computes a run with **no database access at all** - which is what lets it cross a process
boundary, since a SQLAlchemy session cannot - and returns a payload the parent persists in
submission order. Workers do arithmetic; ordering, identity and persistence stay in one place.

### The honest answer on speed-up: parallelism does not help this workload

The roadmap asked for a measured speed-up per tier. The measurement says the naive
expectation is wrong, and that is more useful than a favourable number would have been:

```
torch intra-op threads    serial     thread(4)        process(4)
1                         2.06s      1.47s  1.40x     6.11s  0.34x
12                        0.78s      1.39s  0.56x     5.77s  0.14x
```

**PyTorch already parallelises FFT and BLAS across cores.** Serial with 12 intra-op threads
(0.78 s) beats every parallel configuration. Run-level workers do not add parallelism to this
workload - they compete for cores PyTorch is already using. Constrain intra-op threading to 1
and run-level parallelism starts working (1.40x), which confirms the mechanism rather than
just observing the outcome.

The `process` backend additionally costs **~6.3 s of fixed startup** on this platform: `spawn`
re-imports torch in every worker. 8 trivial tasks take 0.009 s serially and 6.32 s in
processes.

So `serial` is the **measured** default, and `GET /api/v1/health` returns that rationale
alongside the available backends. The seam still matters - it is what makes a cluster backend
additive in Phase 6, and `thread`/`process` are the right choice for the small-and-numerous or
genuinely GIL-bound work Phase 4's surrogate ensembles will involve - but claiming a speed-up
here would have been false.

### Three defects found in my own new code, all by tests

*   **Failure attributed to the wrong step.** `execute_run_payload` reported
    `steps[-1]` - the last *appended* step - so a failure in step 2 was blamed on step 1. A
    misattributed error is worse than a vague one: it sends the reader somewhere confidently
    wrong. Now tracked in a variable set before each step executes.
*   **An unresolvable `{step.key}` reference stayed a literal string.** It then failed far
    downstream as `torch.tensor("{gen.field}") -> must be real number, not NoneType`.
    `ReferenceResolutionError` existed in the taxonomy from T3.5.14 but had never been wired;
    it now fires at the reference and suggests the right key
    (`Did you mean 'gen.field_data'?`).
*   **A fourth transform if/elif chain survived the registry slice.** Moving the action
    bodies out of `engine.py` mechanically carried the six-branch chain into `actions.py`,
    where the previous slice's check did not look. `apply_transform` now dispatches through
    the registry like the other two call sites. Worth recording as the cost of a mechanical
    refactor: the move was correct, but "the chain is gone" was verified only where it was
    expected to be.

### The documentation guard caught me overclaiming

I marked **T3.5.21 as DONE** while its own evidence block said *"Partially met"*.
`test_tasks_marked_done_record_their_evidence` failed on the mismatch. It is correct: this
machine is CPU-only (`torch 2.13.0+cpu`), so the acceptance criterion's
*"passes identically on CPU, CUDA and MPS"* **cannot be executed here and is not claimed**.
T3.5.21 is now labelled **PARTIAL** and D18 likewise. The selection chain, the `SPECTRAL_DEVICE`
override, the refusal path and the thread budget are implemented and tested; cross-device
agreement is outstanding.

That refusal path is worth its own note: requesting an unavailable device **raises** rather
than falling back to CPU. A run that claims to have used a GPU must have used one, and a
silent fallback would make that claim untrue in exactly the situation where nobody checks.

### D12 completed

Every `ExperimentRun` now records its derived `seed` and an `execution` provenance block -
backend, worker count, thread budget, device, torch version, root seed and the
`result_order: submission` contract. Combined with `SeedSequence.spawn` derivation, a run
reproduces to the same numbers regardless of which worker executed it or in what order. That
was the outstanding half of D12 noted in slice 5.

### T3.5.14 completed

The sweep-level acceptance from the error-taxonomy task also lands here, because run-level
failure handling belongs in the run loop. A partially-failing sweep completes its good runs:
a test with `wname` in `{haar, db2, not_a_wavelet, db3}` asserts 3 COMPLETED and 1 FAILED,
with the failure naming its bad parameter.

### Suite after slice 8

```
407 passed, 1 xfailed
272 test functions across 14 files
```

**Fixed: 28 of 31 defects, with D18 partial.** D12 closed this slice.

**Still outstanding:** D8 (FDR), D17 (remaining per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (cross-device verification only); plus T3.5.8 (Alembic),
T3.5.18 (Zarr/ERA5 - the last thing standing between the platform and real data),
browser-based UI verification, and surfacing the benchmark suite, orientation output,
discovery endpoints and execution controls in the frontend.

---

## Slice 9 - T4C.5 statistical validity (defect D8 closed)

**Date:** 2026-08-20. Suite after this slice: **449 passed, 1 xfailed** (up from 407).
New package `src/statistics/` (`multiple_comparisons.py`, `surrogates.py`,
`significance.py`); 41 tests in `src/tests/test_statistics.py`. Benchmark suite:
**15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE** (was 14/0/3).

**This is the slice that decides whether any reported finding is defendable.** Before it, the
hypothesis engine reported every parameter-metric pair whose `abs(r)` exceeded a threshold,
with no p-value at all. On the 9-run smoke sweep that produced 9 "discoveries", including
*"strong positive correlation (r = 0.96) between grid size and floating-point reconstruction
error"*.

### Captured output

```
### 1. calibration of the surrogate test on a TRUE null (white noise) ###
  FPR@0.05 = 0.045   FPR@0.10 = 0.095   (nominal 0.05 / 0.10)
  before the phase-uniformity fix this was 0.765

### 2. the stationarity gate, against measured surrogate FPR ###
  phi    true FPR   gate flags   verdict
  0.00   0.050        0.0%       pass
  0.50   0.045        6.5%       pass
  0.70   0.060       15.5%       pass
  0.80   0.110       86.0%       flag
  0.85   0.155       99.5%       flag
  0.95   0.390      100.0%       flag

### 3. the D8 scenario: 20 tests on 9-sample noise (the smoke sweep's shape) ###
  |r| >= 0.5  (the pre-D8 rule)     :  3 of 20 'discoveries'
  raw p <= 0.05                     :  1 of 20
  after Benjamini-Yekutieli FDR     :  0 of 20   <- correct answer is 0

### 4. power is retained: 1 real effect hidden among 19 nulls, n=40 ###
  recovered: ['real']   (n_significant=1 of 20)

### 5. BH/BY validated against scipy ###
  n=  5  BH max|diff| 0.00e+00   BY max|diff| 0.00e+00
  n= 20  BH max|diff| 1.11e-16   BY max|diff| 0.00e+00
  n=100  BH max|diff| 1.11e-16   BY max|diff| 0.00e+00

### 6. the power trap ###
  99 surrogates, 500 tests -> p floor 0.0100, need 67928, capable: False
```

### The defect, closed and measured both ways

An automated discovery engine's purpose is to test many hypotheses, so it will *always* find
some at a fixed effect-size threshold - on any data, including noise. The fix has to remove
the false positives **without** removing the true ones, and both halves are asserted:

| | reported |
|---|---|
| `abs(r) >= 0.5`, the pre-D8 rule, on 9-sample noise | 3 of 20 |
| raw `p <= 0.05` | 1 of 20 |
| **after Benjamini-Yekutieli FDR** | **0 of 20** |
| one real effect among 19 nulls, n = 40 | **1 of 20, and it is the real one** |

The engine is now silent when underpowered (12 points, weak effect: nothing; best q = 0.58)
and speaks when the evidence is there (40 points, same effect: found, q = 1.9e-11). A
correction that rejected everything would have "closed" D8 by making the platform mute, which
is why the power case is asserted alongside the null case.

**BY rather than BH by default.** BH controls FDR under independence or positive regression
dependence; BY controls it under *arbitrary* dependence at the cost of a `sum(1/i)` penalty.
Parameter-metric pairs drawn from the same runs are dependent in ways not guaranteed to be
positive, so assuming PRDS because it is more convenient would trade validity for power. Both
are validated against `scipy.stats.false_discovery_control` to **1e-16**.

### A surrogate null that was not the null it claimed - the most instructive defect yet

Phase-randomised surrogates were built by drawing independent uniform phases and
antisymmetrising them by averaging, `phi = (phi_k - phi_-k) / 2`. That *is* antisymmetric, so
the surrogate was real-valued and its power spectrum was preserved **exactly to 1e-16** - the
one property the method is defined by, and the only one being tested.

But the difference of two independent uniform angles, halved, has a **triangular density peaked
at zero**. Every surrogate was biased toward the phase-aligned configuration: energy
concentrated rather than spread.

```
excess kurtosis of level-2 wavelet detail
  the fBm field itself     0.75
  its "surrogates"        90.0 +/- 7.5      -> z = -12
false-positive rate on a true null
  before                   0.765            (nominal 0.05)
  after                    0.045
```

**Nothing in the codebase would have caught this.** The tests checked what the method is
*specified* to preserve, and it preserved it perfectly. It was found by making the fBm
benchmark's `4C.surrogate_null` gate runnable: the null benchmark reported a false discovery
within seconds of first execution, which is exactly the job those benchmarks exist to do.
Fixed by taking phases from the FFT of a real white-noise field - Hermitian *and* marginally
uniform by construction rather than by imposition.

The **fBm generator had the same flaw**: it too took `.real` of a non-Hermitian inverse, which
averages each mode with its conjugate mirror and modulates the realised amplitude by the random
phase difference between +k and -k. The field's true spectrum was therefore not the requested
one, leaving residual phase structure at z = 2.1. A null benchmark that does not quite contain
what it claims is worse than no benchmark, so the generator now builds a genuine Gaussian
random field.

**The generalisable lesson: preserving a spectrum is necessary but not sufficient.** A
surrogate must match the null's *distribution*, not only the summary statistic the method is
named after.

### The stationarity gate, calibrated rather than chosen

FT surrogates assume stationarity. Without that, the test is not conservative - it is broken:

```
series                     FPR@0.05   inflation
white noise                0.050      1.0x
AR(1) phi = 0.70           0.060      1.2x
AR(1) phi = 0.80           0.110      2.2x
AR(1) phi = 0.85           0.155      3.1x
AR(1) phi = 0.95           0.390      7.8x
random walk                0.775      15.5x
```

IAAFT reduces but does not remove the inflation, so choosing a better surrogate is not a fix.
**Atmospheric series are routinely strongly red, so this is the default situation, not a corner
case.**

The gate's threshold is set from these measurements: the test is well behaved to
`rho1 ~ 0.69` and unusable beyond `~0.79`, so the boundary sits at **0.75**. A first draft used
0.90, which would have passed AR(1) phi = 0.85 - true FPR 3.1x nominal - as trustworthy.
Measured gate behaviour: flags 0-15% of reliable series and 86-100% of unreliable ones.

Its trend sub-test is itself ESS-corrected. Plain OLS reported a "significant trend" at
p = 1.7e-14 on an AR(1) series with no trend - the spurious-regression effect. The series would
still have been flagged, but for the wrong reason, and **a diagnostic that misdiagnoses is worse
than one that abstains.**

### The power trap

A surrogate p-value floors at `1/(1+n)`. So 99 surrogates cannot reject one of 500 tests after
correction - it would need **67,928**. Such a study reports nothing while looking exactly like
a clean negative result, and is indistinguishable from one. `check_power` now says so
explicitly, and `screen` attaches the warning to any family whose ensemble is too small.

### What a finding now carries

Every `Hypothesis` row and API response carries `p_value`, `q_value`, `n_tests`, and a
`statistics` block naming the test, the correction procedure, **its dependence assumption**, and
the rule-R7 caveat that a designed-grid association is not causal. A finding that travels
without those cannot be judged by the person receiving it.

### Suite after slice 9

```
449 passed, 1 xfailed
309 test functions across 16 files
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE
```

**Fixed: 29 of 31 defects, with D18 partial.** D8 closed this slice.

**Still outstanding:** D17 (per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (cross-device verification only, needs GPU hardware); plus
T3.5.8 (Alembic - now genuinely needed, since this slice and the last added five columns that
only exist via `create_all`), T3.5.18 (Zarr/ERA5 - the last thing between the platform and real
data), browser-based UI verification, and surfacing the statistics, benchmarks, orientation and
execution controls in the frontend.

---

## Slice 10 - T3.5.8: schema migrations (defect D32)

**Captured 2026-08-20.** Every figure below is output from a command in this section.

### The defect, reproduced on the repository's own database before fixing it

`Base.metadata.create_all` adds missing *tables*. It does not add missing *columns*, and it
returns successfully either way. T3.5.12 added `experiment_runs.seed` and `execution`; T4C.5
added `p_value`, `q_value`, `n_tests` and `statistics` to `hypotheses`. The checked-in
`spectral_earth.db` predates both:

```
LEGACY before : OperationalError (sqlite3.OperationalError) no such column: experiment_runs.seed
LEGACY detect : 0001
LEGACY ensure : {"action": "upgrade", "from": "0001", "to": "0002", "applied": ["0002"],
                 "adopted_as": "0001", "case": "adopted_pre_alembic_database"}
LEGACY drift  : []
```

Startup had reported success every time. Nothing logged, nothing failed, until a query touched
one of the five columns. Recorded as **D32**.

### The trap in adopting a pre-Alembic database

The conventional move is `alembic stamp head`, and here it is wrong. Stamping head asserts the
five columns exist; for a pre-slice-8 file they do not, `0002` is then skipped permanently, and
the result is a database Alembic believes is current and the ORM cannot query - the original
defect with a version number attached. So revision `0001` deliberately describes the schema
*without* those columns, and `detect_legacy_revision` **probes for `experiment_runs.seed`** and
stamps 0001 or 0002 by observation. The measured `detect -> 0001, applied 0002, drift []` above
is that path working on the real file.

### SQLite: the acceptance criterion, and more than it asked for

```
FRESH ensure  : {"action": "upgrade", "from": null, "to": "0002", "applied": ["0001", "0002"], "case": "empty_database"}
FRESH drift   : []
DOWN base     : {"action": "downgrade", "from": "0002", "to": null}
  tables left : ['alembic_version']
RE-UP         : {"action": "upgrade", "from": null, "to": "0002", "applied": ["0001", "0002"]}
  drift again : []
```

And through the CLI the criterion actually names:

```
$ DATABASE_URL=sqlite:///.../cli.db python -m alembic upgrade head
exit=0
$ python -m src.database.migrate status
{"revision": "0002", "head": "0002", "pending": [], "up_to_date": true,
 "auto_migrate": true, "error": null}
$ python -m alembic current
0002 (head)
```

`drift: []` is Alembic's own `compare_metadata` with `compare_type` and
`compare_server_default` enabled, asked whether the migrated schema and `models.py` disagree
about any table, column, type, nullability or index. That check - not `upgrade head` - is what
keeps the history honest, because `upgrade head` passes perfectly well for a migration history
that has silently drifted from the ORM, which is the exact failure this layer exists to
prevent.

### PostgreSQL: rendered, not run

The acceptance criterion names both backends. Only one was run. The migrations are asserted to
render valid DDL for the `postgresql` dialect in Alembic's offline mode, which proves they
*compile* there - no SQLite-only construct, no unrenderable type - and does **not** prove they
execute, because no PostgreSQL server was available on this machine. Half of this criterion is
demonstrated and half is asserted; it is written down that way rather than rounded up.

### What the tests actually guard

25 tests in `src/tests/test_migrations.py`. The ones carrying weight:

*   `test_no_drift_against_orm` - adding a column to a model without writing a migration now
    fails a test instead of surfacing as `no such column` in production.
*   `test_create_all_ignores_missing_columns` - asserts the defect itself, so the reason this
    layer exists cannot be lost to a later cleanup.
*   `test_each_revision_round_trips` - every revision up and down individually. A history is
    only reversible if each step is, and the broken step would otherwise be found exactly when
    a rollback was needed.
*   `test_downgrade_0002_preserves_rows` - batch-mode `DROP COLUMN` rebuilds the table on
    SQLite; a mistake there drops data silently rather than erroring.
*   `test_unknown_stamped_revision_is_an_error` - a database migrated by a newer checkout must
    stop us. An older ORM against a newer schema reads and writes the wrong columns without
    complaint.

### One design decision worth stating

All five columns are **nullable**, and that is a scientific choice rather than a convenience.
A run recorded before seed capture genuinely has no seed; `NULL` says exactly that, whereas a
default of `0` would claim a reproducibility that does not exist. Verified: the adopted legacy
run reads back as `('r1', None)`.

`SPECTRALEARTH_AUTO_MIGRATE=0` makes `ensure_schema` refuse and name the outstanding revisions
instead of applying them - the right behaviour for a shared deployment. Refusing to start beats
serving requests against a stale schema and failing on the first query that touches a new
column.

### A staleness audit that the guards had not covered

While marking T3.5.8 done I checked Section 1 of `roadmap.md` against the code. It was wrong in
six places, some of them badly:

| Claim in roadmap.md Section 1 | Reality |
|---|---|
| "271 passed, 1 xfailed" | 478 |
| "14 PASS, 0 FAIL, 3 NOT_YET_RUNNABLE" | 15 / 0 / 2 |
| "DTCWT **still not implemented as advertised** (D1)" | closed in T3.5.6, three slices earlier |
| "Registries / extension seams: **still if/elif chains** (D15)" | closed in T3.5.15 |
| "Hypothesis engine ... **still no multiple-comparison control** (D8)" | closed in T4C.5, the previous slice |
| "HPC / executor seam: **not started**" | landed in T3.5.19 |

`architecture.md` was separately claiming 446 passing tests. None of this failed anything,
because the documentation guards checked the *module inventory*, the *route list*, the *test
inventory table* and the *defect ledger* - but nothing checked the prose status tables, which
are the part a reader looks at first. That is the same class of gap the guards were written to
close, one level up.

Fixed, and then made unable to recur: three new guards in `test_documentation.py` assert that
the two documents agree with each other on the suite total, that the claimed total is at least
the statically countable number of test functions, and that the roadmap's summary of the defect
ledger (`D1-D32, of which 30 fixed, 1 partial (D18), 1 open (D17)`) matches the ledger itself.
The audit script became a committed tool, `tools/audit_docs.py`, exiting non-zero on any
inconsistency - it had been retyped from memory once per slice, and a check that must be
remembered is one that gets skipped exactly when it matters.

```
$ python tools/audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 32 defined, 30 fixed, partial ['D18'], open ['D17']
test functions       : 337
stale inventory rows : none
claimed suite totals : architecture (478, 1) / roadmap (478, 1)
RESULT               : ok
audit exit: 0
```

### Suite after slice 10

```
478 passed, 1 xfailed
337 test functions across 16 files
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
```

**Fixed: 30 of 32 defects, with D18 partial.** D32 was found and closed within this slice.

**Still outstanding:** D17 (per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (cross-device CPU/CUDA/MPS agreement - needs GPU hardware,
not claimed); PostgreSQL migration *execution* (rendered only); plus **T3.5.18 (Zarr/ERA5 - now
the last substantive thing between the platform and real observations)**, browser-based UI
verification, and surfacing the statistics, benchmarks, orientation and execution controls in
the frontend.

---

## Slice 11 - T3.5.18: real ERA5 over cloud Zarr (defects D33 and the unrechunked cache)

**Captured 2026-08-20.** The platform reads real observational data for the first time. Every
figure below came from a command run against the **live** WeatherBench 2 archive, not a mock.

### The archive is reachable, anonymously

```
$ fsspec.filesystem("gs", token="anon").ls("weatherbench2/datasets/era5")
24 stores, 1959-2023, resolutions from 64x32 to 1440x721
```

No account, no credentials, no download portal. `token="anon"` matters: without it `gcsfs` looks
for default credentials and fails with an authentication error, which reads as "you need a Google
account" when in fact nothing is needed at all.

### The chunk-hostility trap, predicted then measured

The roadmap predicted this qualitatively — "reading a decade for one region could move terabytes
to analyse megabytes". The real numbers:

| store | `temperature` chunks | MB/chunk |
|---|---|---|
| `full_37-1h-0p25deg-chunk-1` | `(1, 37, 721, 1440)` | 153.6 |
| `wb13-6h-1440x721` | `(1, 13, 721, 1440)` | 54.0 |
| `6h-240x121` | `(8, 13, 240, 121)` | 12.1 |
| `6h-64x32` | `(100, 13, 64, 32)` | 10.6 |

One timestep, every pressure level, the entire planet, per chunk. `assess_access_pattern` reads
that metadata and predicts the amplification **before transferring anything**:

```
$ python -m src.data_layer.zarr_source inspect --start 2020-01-01 --end 2020-12-31 \
      --lat -4 60 --lon 0 64 --levels 850,700,500,300
bytes_wanted            1,547,131,776   (1.55 GB)
bytes_fetched_estimate 79,039,134,720   (79.04 GB)
amplification                    51.09
chunk_hostile                     true
advice: "temperature: the latitude, level, longitude dimension(s) cannot be narrowed by
         selection because the chunk spans more than the request - subsetting them saves
         nothing over the network, only memory."
```

An independent hand-timed read of the same crop shape moved 108 MB to deliver 2.11 MB — **51.1x
measured against 51.09x predicted**. The predictor is calibrated against the thing it predicts.

That `advice` line is the part a researcher could not have known: **asking for 4 of 13 levels
saves nothing**, because level lives inside the chunk. Only the variable and time axes narrow the
wire.

### Real data, materialised end to end

```
MATERIALISE  content_key 7a2ce3b34e550a69   content_hash 692e7f4613ee5aaf...
             shape {time: 32, level: 4, latitude: 257, longitude: 257}
             predicted 1727.63 MB (uncompressed)  measured 951.34 MB wire  186.3 s
             R13 interior by level {1: 245, 2: 231, 3: 205, 4: 153}
CACHE_READ   8 chunk reads, 19.0 MB, 0.05 s   remote_bytes 0
DATA         (32, 4, 257, 257)  chunks (32, 4, 257, 257)  all finite
             mean 268.03 K   min 219.82 K   max 310.32 K
REPEAT       cache_hit True   bytes 0   0.006 s
REPLAY       key_match True   hash_match True
```

Real ERA5 temperature over 8 days, 4 pressure levels, a 64-degree box: 219.8–310.3 K, mean
268.0 K. Physically sensible values, not a simulation.

**951 MB across the network becomes 19 MB on disk, read in 0.05 s instead of 186 s.** That ratio
is the entire argument for stage 2 existing.

### The criterion that measurement refused

The acceptance criterion asks for a 256x256 region **over one year** within the `laptop` tier
budget ("minutes"). It cannot be done, and the reason is arithmetic rather than implementation:

```
1,464 frames x 54.0 MB/chunk = 79.0 GB to deliver 1.55 GB
sustained throughput, 16 concurrent chunk fetches: 11.6 MB/s
                                        => about 2 hours
```

Sequential reads gave 6.5 MB/s and 16-way concurrency only 11.6 MB/s, so this is bandwidth, not
serialisation. **R13 already resolves the conflict**: the `laptop` tier is constrained on frames,
bank breadth and surrogate count, *never* by shrinking the grid below its valid-interior floor.
So the honest laptop-tier ERA5 crop at 0.25 degree is 256x256 x 4 levels x **days**, and 8 days
took 3.1 minutes. The criterion is recorded as met except for that clause, with the clause
quantified — rather than rewritten to match what happened to work.

### A cache that was not rechunked, caught by its own test

`test_cache_is_rechunked_time_contiguous` failed, and it was right to. `to_zarr` prefers each
variable's inherited `encoding["chunks"]` — copied from the **remote** store — so `.chunk()` set
the dask graph and the write ignored it. The cache came out with **one timestep per chunk: the
exact layout it exists to escape.**

Nothing errored. The manifest still recorded the *requested* chunking, so the provenance record
asserted a property the data did not have. The only visible symptom was a cache read costing 32
chunk fetches instead of 1 — indistinguishable from normal unless measured. Fixed by clearing the
inherited encoding and passing target chunks explicitly. Measured on the real crop, before and
after:

| | chunk reads | MB | seconds |
|---|---|---|---|
| inherited encoding (broken) | 39 | 26.55 | 1.27 |
| explicit encoding (fixed) | **8** | **19.03** | **0.05** |

25x faster and 28% smaller — larger chunks compress better. The content hash was **identical
before and after**, which is the property it was designed for: it describes the data, not the
layout.

**Generalisable lesson: a manifest that records intent is not evidence. It has to record
measurement.** The chunking field said what was asked for; only the byte counter said what
happened.

### D33: a documented feature that could not work

```
$ xr.open_dataset("era5.nc")   # HDF5 magic bytes
ValueError: found the following matches with the input file in xarray's IO backends:
['netcdf4', 'h5netcdf']. But their dependencies may not be installed
$ sorted(xr.backends.list_engines())
['scipy', 'store', 'zarr']
```

`requirements.txt` declared `xarray` and **no NetCDF engine**. `data/README.md` invites the
researcher to drop an ERA5 `.nc` file into `data/`, and `LocalNetCDFSource` is the
highest-priority source in the fallback chain — but a modern ERA5 download is NetCDF4/HDF5, and
`scipy` reads only NetCDF3 classic. The platform's primary documented real-data path had never
been able to work. `h5netcdf` + `h5py` are now declared, and a test writes and reopens an
HDF5-format file so dependency drift cannot silently undo it.

### A capability key that mattered more than it looked

The Zarr source first declared `capabilities={"dataset_ids": [...]}`, and T3.5.15's plugin
acceptance test failed: it asserts that **every id declared under `dataset_ids` appears in
`GET /api/v1/data/datasets`** with concrete variables, a time range, a bounding box and a
resolution. A crop *family* has none of those — the archive is 64 years of the whole planet — so
the invariant was right and the declaration was wrong. It now declares `crop_dataset_ids`;
materialised crops, which do have concrete extents, are listed by `GET /api/v1/data/zarr/cached`.

That change exposed a second, smaller wrong: `sources.py` checked the literal key `dataset_ids`
when deciding whether a source had "declined" a dataset, so the Zarr source was recorded as
declining `era5_reanalysis` — a dataset it has never heard of — in every provenance record. Now
any capability key ending in `dataset_ids` counts as a claim.

And a promise in `sources.py` became true: its docstring said
`SOURCES.with_capability("streaming")` is how this adapter "will be selected without anyone
editing a dispatch chain". Until this slice that query returned an empty list. There is now a
test asserting it returns `["era5_zarr"]`.

### Network access is opt-in, and that is a design position

`SPECTRALEARTH_ALLOW_NETWORK` defaults to off. Reaching the internet must never be a side effect
of running a test or a sweep: it makes results depend on connectivity, and a mistyped bounding
box against a 0.25 degree store moves tens of gigabytes. All 57 non-live tests run offline
against synthetic Zarr stores built with the real archive's pathological layout; the single live
check is opt-in and asserts the documented chunk shape still holds, so if WeatherBench 2 rechunks
its archive the test says so rather than this document quietly becoming false.

### Suite after slice 11

```
535 passed, 1 skipped (the opt-in live check), 1 xfailed
395 test functions across 16 files
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
tools/audit_docs.py: RESULT ok (exit 0)
```

**Fixed: 31 of 33 defects, with D18 partial.** D33 found and closed within this slice.

**Still outstanding:** D17 (per-bin loops in `decompose_by_boundary` and
`analyze_boundary_artefacts`), D18 (cross-device CPU/CUDA/MPS agreement - needs GPU hardware, not
claimed); PostgreSQL migration *execution* (rendered only); a one-year 0.25 degree crop
(quantified as infeasible at laptop tier rather than unimplemented); **browser-based UI
verification and surfacing statistics, benchmarks, orientation, schema state and the Zarr crop
tools in the frontend — now the largest gap between what the backend can do and what a researcher
can reach.**

---

## Slice 12 - T3.5.22: surfacing the backend in the UI

**Captured 2026-08-20.** The backend had grown capabilities faster than the workbench could
reach them. Health, the benchmark suite, the data-source chain, the schema revision, the ERA5
crop tools and every statistical field on a hypothesis were all served and all invisible.

### What a researcher can now see

Two new modules — **8. Platform & Evidence** and **9. Real ERA5 (Zarr)** — and a statistics
block on every hypothesis card.

| Previously invisible | Now shown |
|---|---|
| `torch_device`, cores, threads, executor backends | Platform tab, with the measured rationale for the serial default |
| SQLite journal mode and busy timeout | Platform tab |
| Alembic revision, head, pending migrations | Platform tab, with an explicit warning when the schema is behind the code |
| The nine-dataset benchmark suite and its declared known answers | Platform tab, null benchmarks marked |
| `is_simulated` / `observational` per data source | Platform tab, read from the source's own declared flag |
| ERA5 store catalogue, chunk structure, amplification, R13 floor | ERA5 tab, metadata only |
| `p_value`, `q_value`, `n_tests`, correction assumption, R7 caveat | Every hypothesis card |

### D8's presentation half, closed

The card showed `Confidence: 96.0%` and nothing else. That is an **effect size** wearing the word
"confidence", and it is the presentation half of the defect that let a 9-run sweep read as nine
discoveries — the statistics were computed and corrected in slice 9, and then displayed nowhere.

The card now labels it *effect size* and shows the q-value, the raw p-value, the family size, the
correction procedure **with its dependence assumption**, and the non-causality caveat. A pattern
with no correction is shown with an explicit warning, because a card that looks identical either
way is exactly what made the original defect invisible.

### A runtime bug that three green checks missed

`npm run build` runs `tsc`. It passed. `vite build` emitted 1,378 modules. It passed. 547 backend
tests passed. And the hypothesis card would have crashed on the first mined hypothesis:

```
statistics.correction  is  {method, assumption, n_tests, min_adjusted}
```

The UI rendered `{h.statistics.correction}` directly — an object — which throws *"Objects are not
valid as a React child"* in React. Nothing could catch it: the field is typed `Record<string, any>`
precisely because its shape is nested and open-ended, so TypeScript had nothing to check.

**Generalisable lesson: a type annotation the developer wrote is not a contract with the server.**
`Record<string, any>` is an honest admission that the shape is unknown to the compiler, and every
key read out of one is unverified until something compares it against a real response.

### The contract tests

`src/tests/test_frontend_contract.py`, 13 tests, which:

*   parse `api.ts` for every path it fetches and assert each is a route the app serves —
    distinguishing `/experiments/${id}` (a path parameter) from
    `/proposals${params ? '?' + params : ''}` (a query string, not part of the route). **The first
    version of the parser got that wrong and reported a false positive; the failure was in the
    test, not in the frontend, and it is documented in the parser's docstring so the next reader
    does not re-derive it.**
*   assert every service method is actually called from `App.tsx` — a method nothing calls leaves
    its endpoint just as unreachable as before.
*   assert every nested key the UI reads exists in a real response from the running app:
    `execution.default_backend_rationale`, `schema_state.pending`, `assessment.amplification`,
    `structure.variables[].chunk_megabytes`, `geometry.error`, and the rest.
*   assert every nav entry has a matching panel — a button that does nothing is worse than a
    missing button.
*   assert `roadmap.md` still says the UI has not been visually verified, for as long as that is
    true.

### What is still not verified, stated plainly

**The nine tabs have never been seen.** No browser is available in this environment, so
T3.5.0's screenshot-per-tab criterion remains open. What is now verified is that the UI compiles
under `tsc`, calls routes that exist, and reads fields that are present. What is not verified is
that any of it renders, lays out, or is usable. That distinction is kept explicit — and asserted by
a test — because a green suite plus a green build is the exact combination that makes people
assume otherwise.

### Suite after slice 12

```
548 passed, 1 skipped (opt-in live GCS), 1 xfailed
408 test functions across 17 files
frontend: tsc clean; vite build 1,378 modules, 5.13 MB JS / 17.7 kB CSS
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
tools/audit_docs.py: RESULT ok (exit 0)
```

**Fixed: 31 of 33 defects, with D18 partial.** No new defect ID was raised this slice: the React
child bug was introduced and fixed within it, and never existed outside this working tree.

**Still outstanding:** browser-based visual verification of the nine tabs (T3.5.0); D17 (per-bin
loops in `decompose_by_boundary` and `analyze_boundary_artefacts`); D18 (cross-device CPU/CUDA/MPS
agreement — needs GPU hardware); PostgreSQL migration *execution* (rendered only); and a one-year
0.25 degree ERA5 crop (quantified as infeasible at laptop tier rather than unimplemented).

---

## Slice 13 - T3.5.23: the UI stops fabricating, and starts exporting

**Captured 2026-08-20.** Prompted by a direct question about whether the UI is actually a
world-class scientific tool. It was not, and the audit is recorded here with the fixes.

### The audit, before any changes

| Question | Answer, with evidence |
|---|---|
| Export or import? | **None.** `grep -niE "download\|export\|Blob\|createObjectURL\|csv"` over `frontend/src` matched only the word `export` in `export default function App`. Nothing could leave the browser. |
| Does it fabricate? | **Yes, in ~14 places.** Mock generator, perturbation, transform ("approximate reconstruction with tiny errors"), diagnostics, boundary, slicer, experiment IDs, lineage, hypotheses. |
| Is fabrication labelled? | **Only in the header.** Individual results carried nothing. |
| Is `is_simulated` shown? | **No.** The dataset selector rendered `d.name` only, though the payload carried `is_simulated`, `fallback_reason`, `source_kind` and `source_path`. |
| Are units shown? | **No.** `k_units`, `power_units`, `convention`, `convention_note`, `grid` were returned and displayed nowhere - and absent from the frontend's own types, so it *could not* have shown them. |
| Slope uncertainty? | **No.** `slope_standard_error` computed, returned, never displayed. |
| Unqualified claims? | **Two**, as unconditional green ticks in the transform tab. |
| Literal LaTeX? | **16 instances** rendering as raw dollar signs. |
| Accessibility? | **0** `aria-*`/`role` attributes, **0** keyboard handlers. |

### What changed

**Every fabrication path deleted.** A brace-matching transform collapsed 8
`if (backendConnected) { real } else { fabricated }` guards to the real branch and removed the
three mock helpers; `App.tsx` lost 11,839 characters of code whose only purpose was to invent
results. With no backend there is now no data, an explicit "Backend unreachable - no computation
available" state, and the sentence *"No results are fabricated in its absence."*

**Export, in six formats.** Four server-side, each asserted by reopening the file with the
library a researcher would use:

```
csv     -> numpy.loadtxt(..., comments="#")   values match, header preserved
json    -> nested provenance preserved, not flattened
netcdf  -> magic bytes 89 48 44 46 (HDF5)  ->  xarray: values, coords, units, attrs
zarr    -> zip -> xr.open_zarr             ->  values, coords, attrs
```

PNG and SVG render client-side from the live Plotly figure, deliberately: a server-side
re-render would be a *different* picture from the one on screen, and a figure that does not match
what the researcher saw is worse than no figure.

**NetCDF4, not NetCDF3.** `to_netcdf()` with no path supports only the scipy engine, which writes
NetCDF3 classic - no groups, no compression, 32-bit offsets. The exporter writes via `h5netcdf`
to a temporary file and returns the bytes, and a test asserts the HDF5 magic number. "It produced
a file" is not the same claim as "it produced the file the researcher meant".

**Provenance inside the file.** NetCDF attributes cannot hold a nested dict, so nesting is
flattened to dotted keys rather than dropped - dropping it would silently lose the crop spec, the
correction and the seed. The two warnings are **derived** from the metadata rather than passed
in, so no export path can omit them by forgetting:

```
# is_simulated: True
# warnings.0: THIS DATA IS SIMULATED. ... is NOT an observation.
# warnings.1: THIS DATA IS NOT REPRODUCIBLE. ...
```

### D34: reproducibility that existed only in Python

`PerturbationEngine.add_noise` has taken a seed since T3.5.12. The endpoint never passed one, so
every perturbation requested over HTTP came from the global RNG - and the engine dutifully
recorded `seeded: False` in metadata nothing surfaced. The frontend compounded it, building its
synthetic "forecast" in the browser with `Math.random()`: unseeded, and **uniform** despite the
control being labelled "StDev".

```
seeded (42) twice   -> identical fields
unseeded twice      -> different fields, reproducible: false
```

The seed is now threaded through, returned in `provenance`, and shown on screen next to the
diagnostic it produced.

### A defect I introduced and the compiler caught

The LaTeX cleanup replaced `$...$` literals across `App.tsx` with a blanket substitution. Two
template literals, `` `${k}=${v}` ``, match that pattern exactly and became `` `k ={v}` ``.
`tsc` flagged it as an unused destructure; nothing else would have, and the rendered text would
have been quietly wrong.

**Generalisable lesson: a blanket text substitution over source is a refactor, not a formatting
fix.** It should be reviewed as one.

### Suite after slice 13

```
593 passed, 1 skipped (opt-in live GCS), 1 xfailed
450 test functions across 19 files
frontend: tsc clean; vite build 1,378 modules
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
tools/audit_docs.py: RESULT ok (exit 0)
```

**Fixed: 32 of 34 defects, with D18 partial.** D34 found and closed within this slice.

### What is honestly still not world class

*   **The UI has still never been rendered.** No browser here. Every export control is proved to
    compile, to call a route that exists, and to produce a file that reopens correctly - and has
    never been clicked.
*   **No import.** NetCDF upload is not implemented; real files still have to be placed in
    `data/` by hand or streamed via the Zarr adapter.
*   **Accessibility is zero.** No ARIA roles, no keyboard handling, no focus management.
*   **`App.tsx` is ~2,500 lines with 5 components.** Every tab is inline, so no panel can be
    unit-tested in isolation.
*   **Four endpoints remain unreachable from the UI:** `/benchmarks/run` (the gates can be *seen*
    but not *run*), `/transforms` and `/actions` (capability discovery), `/hypothesis/proposals`.
*   **No CSV/NetCDF ingest into the analysis path**, no session save/restore, no figure captions
    burned into the image beyond the short provenance line.

---

## Slice 14 - T3.5.24: data in, evidence on demand, capabilities visible

**Captured 2026-08-20.** The three things a researcher could not do: bring their own data, make
the platform prove its claims, or see what it can do.

### Import, closing the loop with export

Every format the platform writes now reads back, and every test round-trips through
`exporters` rather than a hand-written fixture - a fixture can encode the same misunderstanding
twice.

```
csv     -> values, coords, units, provenance   round trip exact
json    -> values, coords, units, provenance   round trip exact
netcdf  -> values, coords, units, attrs        round trip exact
zarr    -> values, coords, units, attrs        round trip exact
```

### The refusal that matters

An ERA5 file is `(time, level, lat, lon)`. There is no single field in it.

```
$ read_field(era5.nc, variable="t")
InvalidParameterError: expected an index for every non-spatial dimension
  (time: 0..1, level: 0..2). 't' is 4D; taking index 0 without being asked would import
  a field you did not choose, and every statistic computed from it would describe that
  arbitrary slice.

$ read_field(era5.nc, variable="t", selection={"time": 1, "level": 2})
4 x 5, units K, selection recorded in provenance, values match xarray isel exactly
```

**Generalisable lesson: the dangerous defaults are the ones that produce a plausible answer.**
A crash from `[0, 0]` would have been harmless. A field is not.

### Two failure modes that look like success

*   **Transposition.** Axes are matched by *name* before position. A file with dims
    `(lat, lon, time)` read positionally comes back transposed - and a transposed field still
    looks like a field. Every orientation and anisotropy statistic computed from it would be
    wrong in a way nothing downstream can detect.
*   **Laundering.** Export a simulated field, import it back: it must still say
    `is_simulated: true`. Asserted for all four formats. Without it the platform would offer a
    one-step path from fabricated data to apparently observational data, which is worse than
    never having labelled it. A file of unknown origin reports **null**, not false - claiming
    `False` would be the platform asserting something nobody told it, and the UI renders that
    third state as "origin unknown".

A zipped-Zarr upload is checked for `..` and absolute paths before extraction. This is the one
place the platform accepts arbitrary bytes from outside itself.

### Evidence a researcher can generate

The Ground-Truth Benchmark Suite was **listable and not runnable** - you could see what the
platform claims to get right and could not make it prove it. The Platform tab now runs it at a
chosen root seed, shows each gate's outcome, and raises a distinct alarm when a **null**
benchmark reports a discovery, because that is a false positive in the platform itself rather
than a result. The three outcomes stay separate on screen for the same reason the runner keeps
them separate: folding NOT_YET_RUNNABLE into PASS would let "all green" mean "we never looked".
Verified over HTTP: same seed, same counts; an unknown benchmark name is a 404, not an empty
pass (defect D31 staying closed).

### No route left unreachable

```
served but unreachable from the UI: []
```

A test now asserts this, and any exemption must name its reason inside the test. Exactly one
does: `/hypothesis/proposals`, because the discovery call returns the same records.

### A test of mine that was wrong

`test_a_1d_csv_is_refused` failed: `np.loadtxt(..., ndmin=2)` reads a single row as a genuine
1xN field, and nothing raised. The test asserted a rule I had assumed rather than written. The
right resolution was not to add a second size rule to the reader - the platform already has one
where it belongs, `FieldTooSmallError` at transform time, which names the minimum for the
requested number of levels. Two size rules in two places would diverge. The test now records the
actual behaviour and points at the gate that does the work.

### Suite after slice 14

```
642 passed, 1 skipped (opt-in live GCS), 1 xfailed
490 test functions across 21 files
27 API routes, 0 unreachable from the UI
frontend: tsc clean; vite build 1,378 modules
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
tools/audit_docs.py: RESULT ok (exit 0)
```

**Fixed: 32 of 34 defects, with D18 partial.** No new defect ID this slice.

### Still outstanding, unchanged and stated

*   **The UI has never been rendered.** No browser here. The import panel, the benchmark runner
    and the registry tables compile, call routes that exist, and exchange payloads whose every
    field is present - and have never been seen.
*   **Accessibility is zero.** No ARIA roles, no keyboard handling, no focus management.
*   **`App.tsx` is ~2,900 lines** with 7 components; no tab panel can be unit-tested alone.
*   D17 (per-bin loops), D18 (needs GPU hardware), PostgreSQL migration execution (rendered
    only), and a one-year 0.25 degree ERA5 crop (measured as infeasible at laptop tier).

---

## Slice 15 - T3.5.25: starting the platform, and what that alone found

**Captured 2026-08-20.** Backend on `127.0.0.1:8000`, Vite on `localhost:3000`. Three defects
found in the first ten minutes of it actually running, none of which 642 passing tests had
caught.

### It starts

```
$ python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
backend ready in ~1s
status ok | database ok (sqlite) | schema 0002 (head 0002) up_to_date True
device cpu | cores 12 | torch threads 6 | datasets 3
sqlite: journal_mode=wal busy_timeout=30000 synchronous=1

$ npm run dev
VITE v5.4.21 ready in 453 ms -> http://localhost:3000/
```

The **D32 adoption path ran for real** on the checked-in database, unprompted: it had no
`alembic_version` and was missing the five slice-8/9 columns, and startup stamped it 0001,
applied 0002 and left it queryable. Verified directly against the file afterwards.

Every module transforms through Vite (`/src/App.tsx` -> HTTP 200, 655 kB), and the proxy path
the browser actually uses works end to end.

### D35: a fallback chain that depended on browsing order

```
GET /data/sources  (fresh process)          -> ['netcdf_local', 'simulated']
GET /data/zarr/catalogue                    -> (registers era5_zarr as a side effect)
GET /data/sources  (same process)           -> ['netcdf_local', 'era5_zarr', 'simulated']
```

Source registration is an import side effect, and `zarr_source` was imported only lazily inside
its own handlers. So **which sources `resolve()` searched depended on which endpoint the
researcher happened to call first** - two identical machines could serve the same request from
different sources. Now imported at the composition root. The guard is a **subprocess** test,
because asserting it in-process would pass on whatever the other tests in that module left
behind - the exact order-dependence being tested for.

### D36: the double-precision core was discarded at the HTTP boundary

The transform tab's unconditional green ticks were replaced last slice by a measured round-trip
error judged against a stated tolerance. On its first live run it went amber:

```
fft    max_abs_err 1.937e-07  -> amber (>1e-9, not expected for fft/dct/swt)
swt    max_abs_err 4.172e-07  -> amber
dtcwt  max_abs_err 1.221e-15  -> PASS
```

`dtcwt` at machine epsilon while `fft` sat at 1.9e-07 is the signature of float32. Confirmed on
the identical field:

```
fft  float32  1.937e-07
fft  float64  2.776e-16
```

**Ten sites in `api/main.py` cast every incoming field to `torch.float32`.** The platform's
double-precision discipline - Parseval verified to machine epsilon, tight-frame constants to
1e-11, benchmarks in float64 - was thrown away the moment a request arrived, so every number a
researcher saw through the UI carried nine orders of magnitude more error than the same
computation in a notebook. D27 had been fixed *precisely because* a float32 `fftfreq` capped
Parseval at 5.8e-8; casting the data itself gave all of it back.

After the fix, through the same HTTP path:

```
fft   2.776e-16     dct   3.553e-15     swt   5.872e-16
dwt   3.331e-16     dtcwt 1.221e-15        all PASS <= 1e-9
```

Promoting the boundary to float64 immediately raised
`RuntimeError: expected scalar type Double but found Float` in `apply_affine`, whose rotation
matrix was a hard-coded float32 literal. Its dtype now follows the data.

**Generalisable lesson: a check is only worth what it is judged against.** The green ticks that
said "verified perfect reconstruct limits" had been there all along, above a number that was
seven orders of magnitude off. Replacing an unconditional claim with a measurement against a
stated tolerance found a defect the same afternoon.

### D37: CSV export was lossy, found by a round trip and nothing else

```
export -> import, over HTTP:
  csv      59281 bytes   reimported identical = False
  json    117090 bytes   reimported identical = True
  netcdf   44544 bytes   reimported identical = True
  zarr     12293 bytes   reimported identical = True
```

`%.10g`. IEEE-754 double needs **17** significant digits to round-trip; seven were being
discarded silently. The same precision D36 had just been fixed to stop throwing away, thrown
away again one layer out in the file format. Now `%.17g`:

```
  csv      87833 bytes   identical = True
  json    117075 bytes   identical = True
  netcdf   41984 bytes   identical = True
  zarr     12283 bytes   identical = True
```

About 50% more bytes, which is the cheapest correctness anywhere in the module. A bit-exactness
test now covers all four formats using values chosen to expose it (`0.1234567890123456`, `1e300`,
`-0.0`).

### The rest of the live exercise

```
benchmark suite via the UI's own path:  PASS 15  FAIL 0  NOT_YET_RUNNABLE 2
null-benchmark false positives:         none
seeded perturbation, repeated:          identical fields, reproducible: true, seed 20260820
diagnostics:  k rad pixel^-1 | E(k) energy_1d convention | beta -0.886 +/- 0.098 (R2 0.754)
registries:   6 transforms, 7 actions
data sources: netcdf_local (observational) -> era5_zarr (observational, streaming) -> simulated
datasets:     all three SIMULATED, each naming the missing file
ERA5:         4 stores, network gate off, R13 floor 256 / 512 px
```

The diagnostics line is worth reading: it reports the wavenumber units, **states the E(k)
convention**, gives the slope **with its uncertainty**, and declines to name a regime because
R² = 0.75 means there is no single power law - all four of which were invisible two slices ago.

### Suite after slice 15

```
647 passed, 1 skipped (opt-in live GCS), 1 xfailed
492 test functions across 21 files
benchmark suite: 15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE (exit 0)
tools/audit_docs.py: RESULT ok (exit 0)
```

**Fixed: 35 of 37 defects, with D18 partial.** Three defects found and closed in this slice, all
three by *running the thing* rather than by testing it.

### Still outstanding

Browser rendering remains unverified **by me** - the servers are up and the pages have never
been looked at by a human. That is now a matter of opening `http://localhost:3000`, not of
missing capability. Accessibility is still zero, `App.tsx` is still monolithic, D17 and D18
unchanged.

---

## Slice 15 addendum - the UI, confirmed working by the user

**2026-08-20.** The user started the platform, exercised the workbench and reported the nine
tabs working. Both servers were then stopped cleanly (ports 8000 and 3000 confirmed closed).

**What that establishes, and what it does not.** T3.5.0's final acceptance criterion asks for a
`VERIFICATION.md` recording actual command output *and a screenshot per tab*. Every other clause
is met and captured. The rendering clause is now **attested** - a person ran it and said it
works - but not **evidenced**: no screenshot exists in this repository, so nobody can
re-examine that claim the way they can re-examine the round-trip errors, the benchmark counts or
the transfer measurements recorded above.

That distinction is worth keeping sharp precisely because it is the easy one to let slide. "It
worked when I ran it" is how the original "validated / zero-error" claims in this repository came
to exist, and the audit that opened this whole effort found them to be aspirational. So the
documents now say the rendering was confirmed by the user, name the date, and state plainly that
no screenshot was captured.

`test_browser_rendering_is_still_recorded_as_unverified` had asserted that `roadmap.md` kept
saying the UI had never been seen. That sentence is now false, and a guard that forces a false
statement to remain is worse than no guard - so it was replaced by
`test_browser_rendering_evidence_is_described_accurately`, which asserts the documents record
*who* confirmed it and *that no screenshot exists*, and which **fails if screenshots are ever
added** so the claim gets tightened rather than left stale.

### Servers stopped

```
stopped port 8000 (pid 25476)
stopped port 3000 (pid 18164)
port 8000 down
port 3000 down
```

### And one I broke while writing that down

Replacing `test_browser_rendering_is_still_recorded_as_unverified` was done by slicing the test
file from that function to EOF and substituting. The function was **not** the last one - fifteen
tests written in slices 13 and 14 sat after it, and all fifteen were deleted. The suite went
647 -> 631 and the documentation guard caught it within a minute:

```
architecture.md test inventory is stale (documented, actual): {'test_frontend_contract.py': (28, 13)}
```

Restored, verified back at 28. **Second instance this week of the same mistake**: after the
blanket LaTeX substitution that corrupted two template literals, this is another whole-file text
operation applied on an assumption about structure that was never checked. The lesson is the
same one, and it earned repeating: **editing source by text position is a refactor and needs the
same care as one.** The reason both were caught in under a minute is that the counts are
asserted rather than remembered - which is exactly what `tools/audit_docs.py` and the inventory
test exist for.

---

## Slice 16 - Phase 4A: the time axis and the artifact store

**Captured 2026-08-21.** The first slice of the scientific core. Everything before this was the
instrument; this is the substrate the discovery machinery stands on.

### The gap it closes

`PhysicalField` is strictly 2D and raises on anything else. Every transform, statistic and
benchmark in fifteen slices of work operated on **one snapshot**. The question the whole project
exists to answer - *which small configurations at t precede which large structures at t+delta* -
could not be **stated**, because nothing represented "t". `decompose_by_lead_time` only appeared
to handle time because the caller assembled the list itself, so the axis lived in a local
variable and vanished on return.

### The validation that produces numbers instead of errors

Three consistency rules, each guarding a failure that would otherwise be silent:

```
different shapes  -> refused   (a ragged sequence cannot be stacked or transformed as a block)
different grids   -> refused   (frames on different grids are not one region; a mean across
                                them averages different PLACES together and nothing downstream
                                can detect it)
unsorted times    -> refused   (NOT sorted: a sequence silently reordered is worse than one
                                that errors)
duplicate times   -> refused   ("the next frame" becomes ambiguous and every lag is wrong by
                                an unknown amount)
```

Grid equality is on the **metric, not object identity**. Two frames cropped from the same archive
are separate `GridSpec` instances describing the same region; refusing those would make the class
unusable on real data, and accepting genuinely different ones is the defect above. Both cases are
tested.

Irregular cadence is *reported*, not refused - a concatenation of two crops is legitimate - but
`lag_to_seconds` refuses on it, because "3 frames" is not a fixed duration across a gap.

### R6, and the half that looks clean and leaks

```
overlapping windows                      -> ValueError "they overlap ... measuring memorisation"
disjoint windows, embargo 1, lag 4       -> ValueError "clean-looking and leaks"
gap EXACTLY equal to the lag             -> ValueError (last training target lands on the
                                            first test frame)
embargo 5, lag 3                         -> passes
```

The second row is the one that matters. The windows are disjoint, so an overlap check passes -
and a training example near the boundary still has its *target* inside the test window. That is
the failure R6 exists for, and it has to be checked explicitly rather than assumed away by "they
do not overlap".

The embargo frames are **returned**, not dropped, so a lineage record shows what was excluded and
a reader can verify the gap was real. Refusing an over-long embargo says the wrong fix out loud:
*"do NOT shrink the embargo below the longest lag under test."*

### The artifact store, against its stated criteria

| T4A.3 acceptance criterion | Result |
|---|---|
| 100-frame 64x64 sequence round-trips | **exact**, 3.15 MB on disk |
| Every lineage row under 4 KB | **1,338 bytes** for the `slice_sequence` node |
| `analyze_boundary` no longer embeds a padded field | **already true** via the summariser - now asserted so it cannot regress |

The third is worth stating plainly rather than claiming as new work: it had been fixed earlier by
the summariser layer, and the test exists so a later change cannot quietly reintroduce the
payload.

Beyond the criteria: identical content deduplicates to one file; a tampered artifact **raises**
rather than returning a subtly wrong array; a handle that smuggled a payload into its summary is
refused by its own size budget; arbitrary objects are refused rather than pickled, because an
artifact only the writing code can read is not reproducible data.

### Two bugs, both found by the tests written for them

**An unreachable branch.** `resolve_value` gained artifact dereferencing as an `elif` *after* the
string branch - and a bare `artifact://...` is a string, so the generic `{placeholder}` handling
returned it unchanged and the new branch could never run. The smoke test caught it on the first
call. Moved to the top of the function.

**A tamper test that proved nothing.** `test_a_tampered_artifact_fails_its_checksum` overwrote
the last four bytes of the artifact with zeros. The last bytes of a zip are the
end-of-central-directory comment length, which is **already zero** - so the file was rewritten
byte-identically, the checksum matched, and the test failed for the right reason but would have
*passed* for entirely the wrong one had the assertion been inverted. Now it flips a bit in the
middle and asserts the bytes actually changed first.

**Generalisable lesson: a test of a corruption check has to corrupt something.** Verifying that
the tamper took effect is part of the test, not a nicety.

### Suite after slice 16

```
709 passed, 1 skipped (opt-in live GCS), 1 xfailed
554 test functions across 23 files
tools/audit_docs.py: RESULT ok (exit 0)
```

Phase 4A complete: **4 of 4 tasks**, all four acceptance criteria met and measured.

### Next

**Phase 4B** - `CoefficientField` (time x scale x orientation x y x x), the wavelet bank as a
sweepable parameter, and pressure level as a bank dimension. The store built here is what makes
it possible: a bank over 10 frames x 4 scales x 6 orientations is ~10M floats, which is exactly
the payload the lineage seam could not carry an hour ago.

---

## Slice 17 - Phase 4B: `CoefficientField`, the wavelet bank, and the vertical axis

T4B.1, T4B.2, T4B.3, T4B.4. Four of four.

### T4B.1 acceptance criteria, measured

```
swt    shape=(5, 3, 3, 64, 64) complex=False resampled=False recon_max_err=2e-15
dtcwt  shape=(5, 3, 6, 64, 64) complex=True  resampled=True  recon_max_err=2.22e-15
native shapes    : {1: (32, 32), 2: (16, 16), 3: (8, 8)}
summary bytes    : 2008 for 983040 complex coefficients
```

All three stated criteria met: every scale of both families returns a band on the parent grid;
reconstruction is exact to float64; the summary fits a database row against a payload of nearly
a million complex numbers.

### The claim in T4B.1 that was not true as written

"Backed by SWT/DTCWT so all scales share the parent grid." True of SWT, which is undecimated.
**False of DTCWT**, whose level-*j* subband is about `H/2**j` - the `native shapes` line above is
that fact measured. The alignment is nearest-neighbour **upsampling**, and the field says so:
`resampled_to_parent=True`, every native shape recorded, and the summary stating in words that
the effective resolution of scale *j* remains `2**j` pixels.

Nearest neighbour rather than interpolation, because the aligned view is a *labelling* of parent
pixels by the coefficient covering them - not a smooth field. Bilinear would blend phases from
different spatial positions into values no filter ever produced. Asserted directly: an 8x8
native band spread over 64x64 yields **at most 64 distinct complex values**.

**Reconstruction never uses the aligned array.** The native coefficients are retained and
inverted; a field without them refuses rather than approximating. This is the important one -
inverting the upsampled view returns a field a few percent wrong that is indistinguishable from
a real reconstruction by inspection. Silent, small, and therefore the worst available failure.

### What SWT bands are not allowed to claim

`LH`, `HL`, `HH` - and no angles. A separable real wavelet's `HH` responds to **both** diagonal
signs and cannot distinguish them. Labelling it "45 deg" would assert selectivity the transform
has not got, and that claim would then propagate into every figure drawn from it. The ambiguity
is the classical motivation for the dual tree, and it is recorded as such rather than smoothed
over for the sake of a tidier axis label.

### T4B.2 - the "no engine changes" claim, tested rather than asserted

`combinations()` expands by calling the engine's own `expand_parameter_matrix`, and a test
requires the two to produce identical output. `engine.py` was not touched in this slice.

**An asymmetry in the task's own wording, corrected.** `families` and `scales` are sweep axes;
**`orientations` is not.** A wavelet transform computes every orientation in one pass, so
sweeping them would run the identical decomposition six times and discard five sixths of each
result - the same coefficients at six times the cost. They travel as a selector applied within
each run.

Closed on the way: `int(2.5) == 2`, so a config asking for 2.5 levels ran two and reported two.
There is no half dyadic level; fractional scales are now refused. Found by a test I wrote
expecting a refusal that was not there.

### T4B.3 - the payload stays out of the database

```
bank             : 4 combinations, 921600 coefficients, lineage row 2415 bytes
signature row    : 473 bytes; dominant scale per frame [2, 2, 2, 2, 1]
```

Both actions refuse a **dereferenced payload** by name. `resolve_value` turns a literal
`artifact://...` into a bare array before an action sees it, and an array has no time axis, no
grid and no scale labels - accepting one would mean inventing a cadence. The refusal names
`{step.sequence_ref}` as the fix.

`extract_scale_signature` states its own scope in its result payload: this is the energy half of
R3, and participation ratio, Gini and threshold counts are T4C.1. Written where a reader
actually looks rather than only in the roadmap.

### T4B.4 - two silent failures, both found by building it

**A vertical bank with one sequence.** `decompose_bank` originally ignored `levels_hpa`
entirely. It would have decomposed the same frames once per level and labelled the results
850 hPa and 500 hPa - **identical coefficient arrays under different labels**. Every cross-level
statistic downstream would then have measured a vertical structure created by the labelling. A
bank declaring `levels_hpa` now requires one sequence per level and refuses to guess.

**Levels that collapse onto one stored level.** `sel(method="nearest")` maps a level a dataset
does not carry onto its neighbour, so two requests can return the same data. The guard compares
the arrays and refuses:

```
IDENTICAL: levels 850.0 and 500.0 hPa returned identical data. The dataset does not carry
both, and nearest-level selection has mapped them onto one stored level; a cross-level
statistic computed from this would be correlating a field with itself.
```

It fired immediately on `t2m`, which has no vertical axis at all. A correlation of 1.0 that
means nothing and looks like a discovery is exactly the failure mode this project cannot afford.

`vertical_offsets()` is signed and names its direction: 500 relative to 850 is `-350`, *upward*,
because pressure decreases with height. A reader who got that backwards would invert every
precursor relationship the bank exists to find.

**Scope discipline is asserted, not just written.** `LevelBank.summary()` states that this is 2D
per level and not a 3D wavelet transform, and a test requires that sentence to be present.

### Two defects closed in the artifact store on the way

**D38 - a complex summary that silently described only the real part.**

```
D38 complex : {'n_elements': 2, ..., 'is_complex': True, 'statistic_of': 'magnitude',
               'min': 2.23606797749979, 'max': 3.1622776601683795,
               'mean_real': 2.0, 'mean_imag': 0.5}
```

`summarise` called `float(values.min())` unconditionally. For a complex array that does **not**
raise - numpy casts to real, discards the imaginary part, and warns where nobody reads it.
`[1+2j, 3-1j]` reported `min = 1.0`. Every DTCWT coefficient field is complex, so the lineage
rows of an entire phase would have carried real-part statistics labelled as statistics of the
array.

**D39 - the store dropped the time axis.**

```
sequence reload : times preserved True | values exact True
```

`put(sequence)` stored the `(T, H, W)` tensor and the grid but not the timestamps. `load`
returned an array that was not a sequence, and any caller rebuilding one would have assumed a
regular cadence - silently mis-dating every frame of an irregular record. The irony is exact:
Phase 4A existed to give the platform a time axis, and the store built in the same slice threw
it away. Artifacts now carry their own axes inside the `.npz`, `allow_pickle=False` throughout,
and the 4 KB handle budget is unaffected because the axes went into the archive rather than the
database row. Verified on a deliberately irregular record.

The store's docstring also claimed a `.pt` format that was never implemented. It now states the
single format and the reason: a torch checkpoint is a pickle readable only by the version that
wrote it, which is not a property reproducible data should have.

### Two of my own mistakes, caught by the tests I wrote

*   `assert record["min"] == abs(3 - 1j)` - I had the magnitudes the wrong way round.
    `|1+2j| = sqrt(5)` is the smaller. The code was right; the assertion was not.
*   `assert total_floats > 1_000_000` - the real figure is 921,600. The test now asserts the
    exact count, so a change in what the bank computes shows up here instead of passing.

### Suite after slice 17

```
781 passed, 1 skipped (opt-in live GCS), 1 xfailed
624 test functions across 25 files
tools/audit_docs.py: RESULT ok (exit 0)
```

The documentation guard failed first on both new modules and both new test files, which is what
it is for.

Phase 4B complete: **4 of 4 tasks**, every stated acceptance criterion met and measured.

### Next

**Phase 4C - the gate.** `ScaleSignature` (R3 in full: participation ratio, Gini, threshold
counts with the threshold recorded), `SurrogateNull` integration, and cross-scale lagged
dependency. It is self-contained - it needs nothing from 4D-4G - and it is where the project
finds out whether the central idea is real. Everything built so far exists to make that
question askable honestly; 4C is where it gets asked.

---

## Slice 18 - Phase 4C: the gate's instruments

T4C.1, T4C.2, T4C.3, T4C.4. Four of four. T4C.5 was already done; **T4C.6, the gate review
itself, is not run** - see the last section.

### T4C.1 acceptance criteria, measured

```
sinusoid wl=8   -> dominant level 3 (swt and dtcwt), fraction > 0.5
sinusoid wl=16  -> dominant level 4 (swt and dtcwt)
white noise     -> energy fraction flat to within 15% across usable scales
rescale x1000   -> every threshold-free measure bit-identical
```

The measures have **analytic** values on white noise, and the tests assert those rather than
"roughly flat". Real transform: coefficient energy is chi-squared with one degree of freedom,
so the participation ratio is exactly `1/3` and the Gini exactly `2/pi`. Circular complex
band: exponential, so `1/2` and `1/2`.

```
swt   level 1 : PR 0.334  gini 0.636   (analytic 0.3333, 0.6366)
dtcwt level 2 : PR 0.499  gini 0.500   (analytic 0.5,    0.5)
```

### Rule R3's first trap, measured rather than described

Moving the threshold from 2 to 4 sigma changes the reported threshold count by **more than a
factor of ten**, while `energy_fraction`, `participation_ratio` and `gini` are *bit-identical*.
That is why the threshold count is reported with its threshold and is never primary.

### Two things that would have been silently wrong in the signature

**Replication.** A resampled DTCWT band repeats every native coefficient `4**j` times. The
obvious conclusion - that energy fractions are biased towards coarse scales - is **wrong**, and
was checked rather than assumed: the parent grid has the same cell count at every scale, so the
factor cancels in the ratio exactly. The participation ratio does *not* survive it; replicating
`r` times multiplies it by exactly `r`. Both halves are now asserted, so a future "fix" cannot
break a quantity that is already right. Measures are taken on native coefficients, recovered
from the aligned view by exact stride subsampling when the native arrays are gone.

**Rule R13's crop table understates the dual tree by nearly two (D40).** The table is derived
for one 14-tap filter repeated at every level. DTCWT uses a 19-tap near-symmetric highpass at
level 1 and the q-shift pair above:

```
dtcwt parent halfwidth by level : 9, 19, 45, 97     (R13 table for 14 taps: 6, 13, 26, 52)
dtcwt native halfwidth by level : 5,  5,  6,  7     (roughly constant, as a decimated
                                                     pyramid should be)
256x256 dtcwt interiors : L1 118x118  L2 54x54  L3 20x20  L4 2x2
512x512 dtcwt interiors : L1 246x246  L2 118x118 L3 52x52  L4 18x18
```

A 256x256 crop - this project's stated practical minimum for four dyadic levels - leaves DTCWT
level 4 **four coefficients per orientation**. The signature names that scale *thin* rather
than averaging over it.

### A measured property of the transform, worth recording

Level 1 of a DTCWT is not an analytic signal: the q-shift filters that make the two trees a
Hilbert pair start at level 2.

```
level 1 : var(re) 0.3412  var(im) 0.1555  ratio 2.19   E[w^2]/E[w]^2 = 2.128
level 2 : var(re) 0.2466  var(im) 0.2472  ratio 1.00   E[w^2]/E[w]^2 = 2.002
level 3 : ratio 1.03                                   1.987
level 4 : ratio 1.06                                   1.999
```

The level-1 participation ratio therefore sits at 0.467, between the real-Gaussian `1/3` and
the circular-complex `1/2`, and is predictable from those two variances alone to within 0.02.
The test asserts that prediction, which makes it a check on the arithmetic rather than a
snapshot.

### T4C.2 - and one deviation from the task, stated

The module is `analysis_engine/surrogate_null.py`, **not** `analysis_engine/surrogates.py` as
the task names it. Two modules called `surrogates` in one codebase resolve differently
depending on which package the reader is in, and it invites the phase-randomisation core to be
forked and drift. The generators in `src/statistics/surrogates.py` are called, not
reimplemented; `phase_randomise` was generalised from 2D to any dimension, which its own
arithmetic already supported - the restriction was a statement about what had been tested.

```
spatiotemporal_phase : psd3d err 3.44e-16  max frame psd err 7.15e-01  max |corr| 0.068
per_frame_phase      : psd3d err 1.43e+00  max frame psd err 3.63e-16  max |corr| 0.097
circular_time_shift  : psd3d err 0.00e+00  max frame psd err 8.33e-01  max |corr| 0.591
```

**The choice of null is a choice about time.** On an AR(1) record with no organisation at all:

| null | null lag-one autocorrelation | record's |
|---|---|---|
| `spatiotemporal_phase` (default) | **0.845** | 0.900 |
| `per_frame_phase` | **0.013** | 0.900 |

Preserving the *per-frame* PSD is what the task literally asks for, and it destroys the
record's temporal structure - so the autocorrelation itself beats the null (rule R12). It is
kept, with a warning attached to every result that uses it, because being able to demonstrate
the failure is worth more than removing it.

### T4C.3 - the acceptance criterion, end to end

```
cascade            : 2 significant - 1->3 @ lag 3 (excess 0.170 nats, q 0.0157)
                                     2->3 @ lag 3 (excess 0.155 nats, q 0.0157)
                     nothing in the reverse direction
phase-randomised   : 0 significant
both               : power ok (1,999 surrogates against 18 tests under BY)
```

Correct lag, correct direction. The phase-randomised twin preserves every spectrum and every
autocorrelation and destroys only the alignment.

### The two calibrations that decide whether any of that means anything

**1. A linear lag against a circular null rejects every time.** FT surrogates are circularly
stationary; a record is not. Twenty independent AR(1) records, null true by construction:

```
linear lag statistic   : 20 of 20 false rejections, median p 0.010
circular lag statistic :  0 of 20 false rejections, median p 0.485
```

Lags therefore wrap by default and the wrap fraction is reported. Without this the gate would
have passed on pure red noise.

**2. The shift null must exclude the simultaneous alignment as well as the tested one.**
Rolling the source by `s` measures effective lag `lag + s`, so `s = -lag` puts the two series
at effective lag zero - a real alignment of the data, not a shuffle. On the cascade:

```
observed TE (lag 3)          : 0.211 nats
largest value in the "null"  : 0.412 nats, at exactly s = -lag
p-value ceiling this imposed : ~0.005, regardless of ensemble size
after excluding both windows : p reaches the floor, q = 0.0157
```

Significance limited by a null that was wrong rather than by evidence that was weak. Anything
that varies frame by frame and touches every scale at once couples the scales instantaneously,
so this is the ordinary case and not a corner one.

### Rule R4's support floor, stated honestly

The transform is spatial and applied frame by frame, so its **temporal support is zero**.
Quoting one would be an invention. What is real is the advective crossing time of the filter
support, `support * dx / U`, and `support_floor` **refuses to default the wind speed**: a
plausible-looking 10 m/s would set every floor in every result from a number the reader never
chose. Without it the only floor applied is one frame, and the result says so in as many words.

### A sweep that cannot reject anything says so

```
120 tests, 99 surrogates, Benjamini-Yekutieli -> surrogates required: 12,885
"This study cannot produce a significant result... A null result from this
 configuration says nothing about the data."
```

`check_power`, built in T4C.5, is consulted before the result is read. The first full sweep of
this slice returned zero significant results and was *correct* to - it was arithmetically
incapable of returning anything else, which is a completely different statement from a clean
negative.

### T4C.4

The least-squares core is now `analysis_engine/power_law.loglog_fit`, which knows nothing about
turbulence; `spectra.fit_power_law` calls it and adds the Charney/Kolmogorov interpretation on
top. Existing behaviour unchanged - the 118 spectral and benchmark tests pass untouched - with a
new test pinning the two together. Rule R2 is enforced in the return value: an exponent comes
back `reportable: False` until it sits beside a surrogate ensemble, and `reportable` is
deliberately *not* conditioned on significance, because a null result is reportable and is
often the point.

### Defects

*   **D40 - R13's crop table understates the dual tree.** Fixed: `dtcwt.filter_support`
    accumulates the real cascade, and thin scales are named.
*   **D41 - two implementations of R13 that disagree by one pixel per side.** `OPEN`, with the
    fix specified. `zarr_source.valid_interior` floors a half-integer halfwidth where
    `stationary.valid_interior_halfwidth` rounds it up; flooring is the anti-conservative
    direction, declaring one contaminated pixel per side valid, in the module that sizes crops.
    Bounded and known - two pixels of interior width - and nothing currently depends on the
    difference, but the two must not disagree. Left open rather than fixed late in a long slice
    because the fix changes a published table quoted in several places.

### Suite after slice 18

```
855 passed, 1 skipped (opt-in live GCS), 1 xfailed
692 test functions across 28 files
tools/audit_docs.py: RESULT ok (exit 0)
```

Phase 4C: **4 of 4 implementable tasks complete**, every stated acceptance criterion met and
measured.

### Next

**T4C.6, the gate review, on real ERA5 data.** It is deliberately not written from synthetic
evidence. Everything it needs now exists and is calibrated, and on synthetic data the
instrument gives the answers a working instrument should: it finds an injected cascade at the
right lag and direction, and finds nothing in that same record once the alignment is destroyed.
The verdict belongs to a run on the atmosphere.

Two smaller items this slice deliberately did not take on: registering the cascade as a
ground-truth benchmark (the generator and its truth function exist and are tested; only the
`register_benchmark` call and the suite's documented counts are missing), and closing D41.

### Slice 18 addendum - a tracking audit, and three gaps it found

Prompted by a direct question: are the documents sufficient for another agent to pick this up
cold, and are they honest about what is implemented? Checked rather than answered.

**What was already right.** `tools/audit_docs.py` reports no undocumented modules, no
undocumented routes, no stale inventory rows, a consistent defect ledger and matching suite
totals. Every roadmap task without a **DONE** label is genuinely unstarted (T4C.6 and all of
4D-4G). The known shortfalls were already recorded where they belong: no screenshot per tab,
PostgreSQL migrations rendered but never executed, `App.tsx` still monolithic, D17 open, D18
blocked on hardware. `VERIFICATION.md` carries an unbroken record from slice 2 to slice 18.
**No project memory files exist**, so nothing about this project is held anywhere except the
repository.

**Three gaps, now closed:**

*   **`roadmap.md` Section 1 was dated 2026-08-20**, a day stale, on the section whose whole
    purpose is to be current.
*   **No row anywhere stated phase progress.** A reader could reconstruct it by scanning
    thirty task headings for **DONE** labels, which is exactly the kind of reconstruction a
    status table exists to prevent. Added: what is complete, what is not, and the explicit
    statement that an unlabelled task is unstarted.
*   **Accessibility was measured at zero in `VERIFICATION.md` and named as unmet in
    `roadmap.md`, but appeared nowhere in `architecture.md`** - the document designated as
    the record of what exists. A gap recorded in two places out of three is a gap that will
    eventually be read as absent rather than as unbuilt.

**One imprecision corrected.** The T4C.4 evidence said "all 118 spectral and benchmark tests
pass untouched". The figure is right but the description was loose: it is the 118 tests in
`test_grid_operators.py`, `test_benchmarks.py` and `test_hypothesis.py`, which are the files
that exercise the fit. Now stated that way in both documents.

---

## T4C.5a - one conservative edge convention (D41)

Closed on 2026-08-21 before designing the real-ERA5 gate. The crop-sizing path used
`floor((L-1) * 2**(j-1) / 2)` while the SWT analysis path used half of the full effective
support. For an even-length filter at level 1 the radius is a half-integer, so the former
declared one contaminated pixel per side valid.

The implementation now computes the effective support first and uses `support // 2`, matching
`stationary.valid_interior_halfwidth`. The published 14-tap R13 margins are therefore
7/13/26/52 rather than 6/13/26/52 pixels per side; the N=64 level-1 interior is 50 rather than
52 pixels. Coarser margins and the 256/512 minimum crop sizes are unchanged. The existing R13
geometry test now compares the Zarr sizing implementation directly with the SWT implementation
for db2, preventing the two definitions from drifting independently again.

Evidence command:

```text
python -m pytest -q src/tests/test_zarr_source.py -k
"edge_exclusion or valid_interior or minimum_crop or crop_too_small or crop_at_the_floor"
```

The result is recorded in the next suite block after execution. D41 is now **FIXED**; D17 is
the sole fully open defect and D18 remains partial because cross-device hardware is absent.

---

## T4C.5b - real ERA5 reaches the Phase 4 pipeline (D42)

Closed on 2026-08-21 while preparing the real-data gate. Static review found that the Zarr
source required a crop block but the registered `slice_sequence` action could pass only a
dataset id. The dedicated ERA5 tab and the Phase 4 analysis spine were therefore both working
and disconnected. WeatherBench coordinates also use `latitude`/`longitude`, while the adapter
indexed only `lat`/`lon`.

The adapter now accepts parameterised source options without putting them in its legacy
dataset-id cache, the action passes the crop and cache directory through, and both coordinate
naming conventions map onto the physical `lat`/`lon` spine. Resolution provenance and the
source request are stored in `FieldSequence.metadata`.

Executed against a local Zarr store with WeatherBench's dimensions and hostile remote chunk
layout. The crop was materialised, the network disabled, then the cached data ran through the
adapter and registered action into the artifact store. Assertions require multiple timestamps,
a lat/lon physical grid, `source_kind == "zarr"`, `is_simulated is False`, and an
`artifact://` sequence handle.

```text
python -m pytest -q \
  src/tests/test_zarr_source.py::test_source_serves_a_cached_crop_with_no_network

1 passed
```

This proves the transport and provenance seam on a structurally faithful local store. It does
not claim that T4C.6 has run on the atmosphere; no real crop is currently present under
`data/zarr_cache`, and the gate protocol still has to be frozen before a network transfer.

---

## T4C.5c - frozen replication protocol and the honest data blocker (D43)

`GateProtocol` and `evaluate_replication_gate` now make the T4C.6 decision rule executable.
Four tests assert rejection of an underpowered surrogate family, rejection of an embargo
shorter than the longest lag, rejection of estimator-starved temporal partitions, stable and
sensitive protocol fingerprints, and the distinction between PASS, FAIL and INVALID.

The live 0.7-degree WeatherBench store was inspected without materialising data:

```text
store: era5_0p7_6h
variable: temperature
window: 2018-01-01 to 2020-12-31
selection: 4384 times x 1 level x 255 lon x 255 lat
chunks: 8 x 13 x 512 x 256
wanted: 1,140,278,400 bytes
estimated fetched: 29,880,221,696 bytes
amplification: 26.2x
```

That is a metadata-derived uncompressed upper bound, not a measured wire transfer, and is
labelled accordingly. It is sufficient to reject the current layout for a laptop-tier gate:
the layout transfers every level and global spatial chunk regardless of the regional request.
D43 records the missing spatially tiled temporal source. The gate remains **not run**.

### Suite after T4C.5a-c

```text
859 passed, 1 skipped (opt-in live GCS), 1 xfailed
696 test functions across 28 files
tools/audit_docs.py: RESULT ok (exit 0)
```

The full suite completed in 226.68 s. The xfail remains the deliberately retained regression
arm for the original degenerate transform; the skip remains the explicitly opt-in live-GCS
transport check. Neither is a T4C.6 atmospheric verdict.

---

## Phase 5 readiness audit - planning evidence, not implementation acceptance

The immediate regional-training proposal was checked against the current implementation before
being added to `architecture.md` and `roadmap.md`. This was a read-only smoke test, not a new
feature or an acceptance test.

For the implemented `near_sym_b` / `qshift_b` DTCWT, `valid_interior_halfwidth` returned:

```text
levels 1..4: [9, 19, 45, 97] parent-grid pixels per side
```

For the current SWT filters it returned:

```text
Haar levels 1..4: [1, 1, 2, 4]
db2  levels 1..4: [2, 3, 6, 12]
```

On fresh float64 32x32 tensors with `requires_grad=True`, forward/inverse reconstruction for
DTCWT (two levels), db2 SWT (two levels), DCT and FFT produced finite input gradients and
maximum reconstruction errors between `1.1e-15` and `2.6e-14`. This establishes only that the
present CPU operations can participate in those simple autograd graphs.

The same audit confirmed the blocking interface fact:

```text
PhysicalField(torch.randn(2, 5, 32, 32))
ValueError: PhysicalField data must be 2D. Got shape torch.Size([2, 5, 32, 32])
CUDA available: False
```

Therefore no batched, CUDA, mixed-precision, compiled, cached-filter or training-throughput
claim is accepted. Those remain explicit T5.1 acceptance work. No inference about the external
poster follows from these margins until its exact domain, filters, levels and boundary mode are
obtained and frozen under T5.0.

---

## T5.1a - training-native raw, FFT and DCT representations

`src/transform_engine/training.py` adds the first accepted part of `RepresentationModule`.
It accepts `(B,C,H,W)` float32/float64 tensors and keeps synthesis context in an immutable
`EncodedRepresentation`, rather than mutable module state. Accepted representations are:

* raw identity control;
* real FFT with real channels followed by imaginary channels; and
* orthonormal DCT-II/III with fixed-shape registered matrix buffers.

The DCT buffers are reused during every forward/inverse call. Device/dtype migration rebuilds
the deterministic matrices once at the destination precision; it does not promote previously
rounded float32 values. The canonical float32 basis is evaluated in float64 before casting.

Focused acceptance command:

```text
python -m pytest -q src/tests/test_training_representations.py src/tests/test_transforms.py
57 passed, 1 skipped, 1 xfailed, 1 warning in 2.70s
```

At the time of this CPU audit, the skip was the explicit accelerator parity test because the
venv contained a CPU-only PyTorch wheel. That historical result is superseded by the real-GPU
verification below. The xfail is the pre-existing degenerate DTCWT regression arm.

A single-machine microbenchmark measured combined encode+inverse after five warmups and over
30 samples. It is descriptive, not a portable performance guarantee:

```text
PyTorch 2.13.0+cpu; 6 threads; CPU; float32; shape (4,5,120,80)
representation  median ms  p90 ms  encoded bytes  max reconstruction error
raw                0.0043    0.0046        768000  0
fft                0.5728    0.7136        787200  1.1920929e-06
dct                0.6779    0.9684        768000  1.90734863e-06
```

The tests cover batch/item equivalence, odd and even widths, explicit complex packing,
float32/float64 reconstruction, forward/inverse `gradcheck`, immutable model-output context,
cache pointer reuse, module dtype migration, malformed inputs and factory errors. Haar, db2,
SWT and DTCWT are not present in the training factory; T5.1 remains **PARTIAL**.

### Full suite after T5.1a

```text
882 passed, 2 skipped, 1 xfailed, 6 warnings in 125.16s
712 test functions across 29 files
```

The second skip is the CUDA test described above; the pre-existing live-GCS test remains the
first. The full suite and documentation integrity checks are green. This result accepts the
CPU raw/FFT/DCT slice only; it does not close T5.1 or provide forecast evidence.

---

## T5.1a accelerator verification - RTX CUDA, vendor-neutral test seam

The workstation was not CPU-only. `nvidia-smi` reported an NVIDIA GeForce RTX 5050 Laptop GPU
with 8,151 MiB, but the venv held `torch 2.13.0+cpu`. The official CUDA 13.0 wheel was installed:

```text
torch: 2.13.0+cu130
compiled CUDA runtime: 13.0
driver-advertised CUDA runtime: 13.3
device: NVIDIA GeForce RTX 5050 Laptop GPU
compute capability: 12.0
device memory: 7.96 GiB
```

The former CUDA-specific test is now a vendor-neutral accelerator test. It enumerates available
PyTorch devices, moves each representation through ordinary `.to(device)` semantics, compares
raw/FFT/DCT coefficients and reconstructions with CPU, and performs backward propagation with
finite-gradient assertions. PyTorch ROCm deliberately exposes AMD devices through its `cuda`
API; `core/device.available_devices` now records `accelerator_runtime = rocm|cuda|mps` plus the
compiled CUDA/HIP versions so AMD execution cannot be mislabeled as NVIDIA.

Focused result on the installed RTX:

```text
python -m pytest -q src/tests/test_training_representations.py src/tests/test_executor.py
52 passed, 1 warning in 23.35s
```

No accelerator test was skipped. A CUDA-event timing of 200 encode+inverse iterations after 20
warmups on float32 `(4,5,120,80)` measured:

```text
representation  mean ms  encoded bytes  max reconstruction error
raw               0.0052        768000  0
fft               0.1733        787200  1.1920929e-06
dct               0.1465        768000  1.9073486e-06
```

This verifies NVIDIA CUDA for T5.1a. The code path contains no NVIDIA-specific tensor calls and
is compatible by construction with a supported PyTorch ROCm build, but AMD ROCm, Apple MPS and
Windows DirectML hardware have not been run and are not claimed as verified. T5.1 remains
partial because the training wavelet representations are still outstanding.

### Full suite on the accelerator-enabled build

```text
PyTorch 2.13.0+cu130
883 passed, 1 skipped, 1 xfailed, 6 warnings in 129.27s
712 test functions across 29 files
```

The sole skip is the explicitly opt-in live-GCS transport check. The accelerator test executed
and passed; it is no longer part of the skip count.

---

## Portable laptop/local-GPU/HPC execution profiles

The execution policy now exposes `auto`, `cpu`, `accelerator` and `hpc` profiles. Tests assert
CPU forcing, conflict refusal, accelerator-without-device refusal, Slurm/PBS detection,
scheduler local-rank placement, login-node refusal, profile environment handling and a real
doctor smoke report. The obsolete `experiment_engine.get_execution_device` implementation now
delegates to the same policy used by API health and experiments.

Focused result:

```text
python -m pytest -q src/tests/test_executor.py src/tests/test_training_representations.py
59 passed, 1 warning in 23.26s
```

Actual human-readable doctor output on this workstation:

```text
SpectralEarth execution doctor
  platform: Windows AMD64
  PyTorch:  2.13.0+cu130
  profile:  auto -> cuda:0
  runtime:  cuda
  smoke cpu:     PASS
  smoke cuda:0:  PASS
```

The doctor performs no network operation and has no scheduler client. `hpc` recognises only an
active Slurm, PBS or LSF allocation and refuses to run on a login node. This makes the same
codebase easy to run locally or within an allocated job without making local functionality
depend on cluster availability. Remote submission and artifact transfer are not implemented.

### Full suite after portable profiles and doctor

```text
890 passed, 1 skipped, 1 xfailed, 6 warnings in 128.00s
719 test functions across 29 files
```

The suite ran under the accelerator-enabled build with the default `auto` profile selecting the
RTX. The only skip remains live GCS. A simulated Slurm allocation
(`SLURM_JOB_ID=doctor-simulation`, `SLURM_LOCALID=0`) resolved `hpc -> cuda:0` and passed CPU and
CUDA smoke plus `--require-accelerator`; this verifies allocation parsing and placement logic,
not a real scheduler submission or cluster performance result.

---

## T5.1b - batched decimated Haar/db2 and the corrected cascade budget (D44)

`HaarRepresentation` and `DB2Representation` now accept `(B,C,H,W)` float32/float64 tensors,
use the canonical filter definitions, and return one recursively packed Mallat plane. The
boundary convention is explicit PyWavelets-compatible `periodization`; bottom/right dyadic
padding and the inverse crop are immutable synthesis metadata, and callers may refuse implicit
padding. No mutable per-call state is held on the module.

The numerical acceptance is independent where it matters:

* level-one LL/LH/HL/HH bands match `pywt.dwt2(..., mode="periodization")` in float64;
* multilevel packed energy matches a separate PyWavelets decomposition;
* odd and even shapes reconstruct, batch execution equals item execution, and float64
  `gradcheck` passes through encode and inverse;
* the vendor-neutral accelerator test checks CPU/RTX coefficients, reconstruction and backward
  gradients for Haar and db2 as well as raw/FFT/DCT.

The focused regression result after the implementation and R13 correction was:

```text
python -m pytest -q src/tests/test_training_representations.py src/tests/test_stationary.py src/tests/test_zarr_source.py
189 passed, 1 skipped, 5 warnings in 18.94s
```

The skip is the opt-in live-GCS transport test, not accelerator coverage.

### D44: the earlier generic edge budget was wrong

The previous SWT/Zarr formula counted only the filter newly applied at level `j`:
`(L-1)2^(j-1)+1`. A recursive coefficient has already passed through every earlier low-pass
stage. Directly composing the dilated filters gives the complete support
`1+(L-1)(2^j-1)`. A test now performs that convolution independently and compares its measured
length with `filter_support`.

Consequences recorded rather than softened:

```text
wavelet/filter   corrected margins per side
Haar, levels 1-4       1, 2, 4, 8
db2, levels 1-4        2, 5, 11, 23
14 taps, levels 1-4    7, 20, 46, 98
```

With the declared minimum of 128 valid pixels, the generic four/five-level crop floors are now
512/1024, not 256/512. The historical 257x257 WeatherBench transfer remains valid evidence of
transport amplification, but it is only geometrically adequate for three generic levels under
R13. Earlier D40/D41 verification text is retained as history and superseded by this correction.

### Honest performance status

Float32 encode+inverse on `(4,5,120,80)`, three levels, measured after warmup:

```text
representation   CPU mean ms   RTX mean ms   encoded bytes   max error
haar                  5.6843        4.8761          768000    9.54e-7
db2                   9.2046       13.7383          768000    9.54e-7
```

The clear implementation issues multiple `torch.roll` kernels per tap. It is accepted as the
numerical reference path, not yet as the final cheap training-loop path: db2 is measurably
slower on this RTX than on CPU. Fused convolution/compilation and mixed-precision acceptance
remain explicit T5.1 work rather than being hidden behind the fact that CUDA executes.

### Full suite after T5.1b and D44

```text
911 passed, 1 skipped, 1 xfailed, 6 warnings in 130.10s
726 test functions across 29 files
```

The only skip remains the opt-in live-GCS transport check. The xfail remains the declared
degenerate DTCWT regression arm. On the first full attempt, the Windows process-pool stress
test had one transient worker memory-allocation failure and the frontend contract exposed its
stale expected crop floor. The contract was corrected; both checks then passed together, and
the complete clean result above is from a fresh full-suite rerun.

---

## T5.1c - convolution wavelet training kernel, reference retained

The default Haar/db2 implementation now evaluates all four bands at each level with one
strided `conv2d`. Synthesis uses the same registered four-kernel bank in
`conv_transpose2d`, followed by the exact adjoint of the symmetric circular pad. The earlier
per-tap `torch.roll` implementation remains selectable as `implementation="reference"`; it is
an executable oracle, not deleted optimization history.

Acceptance compares more than reconstruction through an identity:

* fused and reference packed coefficients agree in float64;
* their independently executed inverses agree;
* gradients of a random weighted coefficient loss with respect to the input agree, so a
  coefficient-order or adjoint error cannot hide behind perfect reconstruction;
* both implementations pass float64 `gradcheck`;
* the cached `(4,1,L,L)` bank is a registered buffer, reused across calls and regenerated from
  canonical full-precision filters after dtype/device migration;
* implementation identity travels in immutable synthesis metadata, and inverse refuses a
  context from the other implementation.

Focused result, including the available RTX accelerator arm:

```text
python -m pytest -q src/tests/test_training_representations.py
51 passed, 1 warning in 5.02s
```

### Runtime and training-step measurement

Float32 `(4,5,120,80)`, three levels, PyTorch `2.13.0+cu130`; every timed region was warmed up
and device-synchronised. `train step` is encode -> inverse -> mean-square loss -> backward with
a fresh leaf tensor, not forward timing mislabeled as training throughput.

```text
device  wavelet  implementation  round trip ms  train step ms  max error
CPU     Haar     reference              4.4808        10.3181    9.54e-7
CPU     Haar     conv2d                 2.7521         7.3525    9.54e-7
CPU     db2      reference              7.3852        17.8177    1.19e-6
CPU     db2      conv2d                 3.9173        14.0860    1.19e-6
RTX     Haar     reference              4.8243        11.4812    9.54e-7
RTX     Haar     conv2d                 0.7041         3.9465    1.19e-6
RTX     db2      reference              7.5844        18.2640    9.54e-7
RTX     db2      conv2d                 1.4504         8.6618    1.19e-6
```

This reverses T5.1b's important performance failure: db2 now benefits materially from the RTX.
It does not imply that the whole future model is GPU-bound or that AMD has been measured.

CUDA peak memory was reset after allocating the module and input, then measured across one full
training step. Encoded storage is 0.732 MiB in every arm:

```text
wavelet  implementation  incremental peak MiB
Haar     reference                    6.042
Haar     conv2d                       3.343
db2      reference                    5.310
db2      conv2d                       5.866
```

The db2 speedup costs 0.556 MiB of incremental peak allocation on this batch; that tradeoff is
small on an 8 GiB device but is recorded rather than omitted. Mixed precision, `torch.compile`,
ROCm and MPS remain NOT RUN.

### Full suite after T5.1c

```text
917 passed, 1 skipped, 1 xfailed, 6 warnings in 131.01s
729 test functions across 29 files
```

The only skip remains the opt-in live-GCS transport check; the xfail remains the declared
degenerate DTCWT regression arm. The available RTX accelerator test executed and passed.

---

## T5.1d - batched autograd SWT and truthful readiness UI

`SWTRepresentation` is now an accepted `(B,C,H,W)` training representation for Haar, db2 and
db3. Its model tensor packs final LL then levelwise LH/HL/HH for each input channel. Every band
keeps `(H,W)`, giving the explicit ratio `1+3*levels`; db2 at three levels therefore turns five
input variables into 50 model channels and expands float32 `(4,5,120,80)` from 0.732 MiB to
7.324 MiB.

Numerical acceptance does not rely on self-reconstruction alone:

* packed coefficients agree channel-by-channel with the existing coordinate-aware analytical
  SWT;
* convolution and an independent FFT circular-filter path agree in packed coefficients,
  inverse and gradients of a random weighted coefficient loss;
* both implementations reconstruct batched Haar/db2/db3 through three levels;
* float64 `gradcheck` passes and the vendor-neutral accelerator arm executes encode, inverse and
  backward on the RTX;
* rolling the input by `(5,-7)` rolls every coefficient by exactly `(5,-7)` under the declared
  periodic boundary;
* a support-exhausted request is refused, and metadata carries support, per-side exclusion,
  valid interior HxW, level amplitude gain, energy-normalisation rule and band meaning.

Focused training result after the precision regression below:

```text
python -m pytest -q src/tests/test_training_representations.py
66 passed, 1 warning in 4.80s
```

### Measured execution policy and cost

Float32 `(4,5,120,80)`, db2, three levels; timed regions were warmed and device-synchronised.
The training step is encode -> inverse -> mean-square loss -> backward with a fresh leaf.

```text
device  implementation  round trip ms  training step ms  encoded MiB  max error
CPU     FFT reference          19.8094           47.2215        7.324   2.15e-6
CPU     convolution            69.4380          150.2459        7.324   1.91e-6
RTX     FFT reference          13.5782           19.5791        7.324   2.62e-6
RTX     convolution             6.4003           14.5261        7.324   1.43e-6
```

Incremental RTX peak allocation was 52.485 MiB for FFT and 25.464 MiB for convolution. The
default `auto` policy therefore selects FFT on CPU and convolution on CUDA/ROCm/MPS. That is a
measured CPU/RTX decision and an execution policy for the other PyTorch devices, **not** ROCm or
MPS verification.

### D45 - ambient cuDNN state changed the transform

The first full-suite run failed the accelerator reconstruction test after `enable_determinism`
had run. On the identical seeded L1 db2 input, deterministic cuDNN with ambient TF32 moved the
maximum error from 2.38e-7 to 4.48e-4. A focused fresh-process run had hidden the state
dependence. Training convolution operations now locally scope `allow_tf32=False` and restore
the caller's policy. A dedicated test enters deterministic+TF32 state, verifies reconstruction
within 3e-6, and verifies that TF32 is still enabled for the caller afterwards. D45 is fixed,
not papered over with a wider tolerance.

### Readiness API and Spectral Transforms UI

`GET /api/v1/training/representations` is separate from the analytical transform registry. It
reports which representations are genuinely batched/autograd accepted and, for the selected
grid/wavelet/levels, derives SWT coefficient channels, accumulated support, edge exclusion and
valid interior. The React panel renders coefficient expansion, shift behaviour, directional
meaning, boundary convention, scientific role, limitations, verified environments and NOT RUN
environments. DTCWT is present as **analysis only**, preventing availability for one 2D field
from being misread as training readiness.

```text
python -m pytest -q src/tests/test_training_representations.py src/tests/test_frontend_contract.py
94 passed, 5 warnings in 10.69s

npm run build
TypeScript PASS; Vite PASS; 1384 modules transformed
bundle: 10,009.37 kB JS (3,016.88 kB gzip)
```

The bundle size warning is real and code-splitting remains outstanding. Browser-skill setup was
attempted, but the session reported no available controllable browser, so rendered visual
inspection is **NOT RUN**. No screenshot or visual acceptance is claimed.

### Full suite after T5.1d and D45

```text
933 passed, 1 skipped, 1 xfailed, 6 warnings in 137.81s
740 test functions across 29 files
```

The skip remains the opt-in live-GCS transport test. The xfail remains the declared degenerate
DTCWT regression arm. The RTX accelerator test executed and passed in the clean rerun.

## T5.1e - batched autograd DTCWT and native scientific visualisation

`DTCWTRepresentation` now accepts float32/float64 `(B,C,H,W)`, vmaps the canonical Kingsbury
analysis/synthesis arithmetic, and registers all twelve level-1/q-shift analysis and synthesis
filters as dtype/device-migration-aware buffers. Its four-real-plane recursive atlas is a tested
bijection: unpacking and repacking arbitrary coefficients is exact, and every one of the six
complex orientations at every native scale is retained without interpolation.

Acceptance covers batch reconstruction, analytical `PhysicalField` coefficient agreement,
random coefficient-loss gradients, fast float64 `gradcheck`, cache migration, refusal when the
declared edge margin leaves no strict two-dimensional interior, and CPU/RTX coefficient parity,
inverse and backward flow. The CUDA test ran on the installed RTX 5050; ROCm/MPS remain NOT RUN.

For float32 `(2,5,120,80)`, `near_sym_b`/`qshift_b`, L2, ten timed iterations after warm-up:

```text
device  round trip ms  training step ms  encoded MiB  max error
CPU          46.23          138.34          1.465       1.20e-6
RTX          35.87           65.91          1.465       1.20e-6
```

The analytical DTCWT summary now returns six complex-magnitude maps at each level's native
resolution, measured and nominal direction centres, and native/parent validity margins. The UI
uses one colour range across the six panels of a level, draws the valid-interior inset, and says
explicitly that panel/atlas adjacency is not physical, levels are not resampled together, and
phase is retained for synthesis rather than painted as a scalar physical field. The production
TypeScript/Vite build passed (1,385 modules; 10,014.34 kB JS / 3,018.35 kB gzip); the existing
large-bundle warning remains.

Focused acceptance:

```text
143 passed, 1 warning in 10.30s
137 passed, 5 warnings in 19.78s  # training + registry + frontend contracts
```

The first full run found only a stale documentation count, which was corrected before the clean
rerun. Final suite:

```text
946 passed, 1 skipped, 1 xfailed, 6 warnings in 156.12s
748 test functions across 29 files
```

The skip is still the opt-in live-GCS transport check and the xfail is still the declared old
degenerate-DTCWT comparison arm. The backend and frontend were started and answered health/HTTP
checks, but browser discovery returned an empty list, so rendered inspection is **NOT RUN** and
no screenshot or visual acceptance is claimed. Both exact temporary server processes were then
stopped.

## T5.2a - leakage-safe regional forecast dataset bridge

`src/data_layer/regional_forecast.py` now constructs ordinary PyTorch datasets from a
materialised xarray/ERA5 crop. Canonical `t/q/u/v/z` resolve against either short names or the
WeatherBench long names, 850 hPa is selected explicitly, every variable must share exact time,
latitude and longitude coordinates, and non-finite cells are refused rather than imputed by an
unstated policy.

The accepted R6 `split_temporal` implementation runs before sample indexing. The configuration
refuses an embargo shorter than the longest lead; tests then enumerate every input and target
frame from every sample and prove each remains inside its assigned split and outside the two
returned embargo sequences. Per-variable population means and standard deviations are computed
in float64 over training frames/grid cells only, stored in a hashed `NormalisationArtifact`, and
reused by identity for validation and test. A held-out million-unit offset does not alter the
training means, making the leakage test causal rather than a metadata assertion.

The default PyTorch collator produces `(B,history,C,H,W)` inputs and `(B,lead,C,H,W)` targets,
with Unix-nanosecond timestamps and original frame indices. Bundle provenance carries the source
manifest/content hash, crop/chunking, canonical and resolved variables, pressure level, complete
time/grid fingerprints, split extents and normalisation artifact. A local test writes a
chunk-hostile five-variable Zarr source, materialises/rechunks it into an arbitrary directory,
and prepares the dataset through that cache with measured `remote_bytes == 0`; changing that
directory to an HPC shared path does not change the Python interface or require a GPU.

`cross_check_era5_overlap` is implemented as the real acceptance instrument: it requires exact
coordinates, permits no interpolation, and reports per-variable maximum/mean absolute error
under declared tolerances. It passes against an independent local construction and detects both
a changed value and shifted coordinates. It has **NOT RUN against an actual second ERA5
acquisition route**, so T5.2 remains partial and D43 remains open. At T5.2a acceptance the cache
loader still eagerly materialised the prepared crop in host memory; that specific limitation
was subsequently removed by T5.2b below.

The ERA5 cache panel now reports manifest-only structural eligibility while explicitly showing
`Prepared dataset: NO`, `train-only normalisation verified: NO` and `independent ERA5
cross-check: NOT RUN`. A green manifest therefore cannot be mistaken for completed value-level
acceptance.

Focused acceptance:

```text
python -m pytest -q src/tests/test_regional_forecast.py \
  src/tests/test_sequence.py src/tests/test_zarr_source.py src/tests/test_frontend_contract.py
132 passed, 1 skipped, 5 warnings in 18.06s

npm run build
TypeScript PASS; Vite PASS; 1,385 modules transformed
bundle: 10,015.27 kB JS (3,018.56 kB gzip)
```

The existing large-bundle warning remains. Both local servers answered HTTP 200. The mandatory
browser connection was attempted, troubleshooting was followed and browser discovery returned
an empty list, so rendered inspection of the T5.2 addition is **NOT RUN** and no screenshot is
claimed.

Clean full suite after the documentation inventory was reconciled:

```text
955 passed, 1 skipped, 1 xfailed, 6 warnings in 149.29s
757 test functions across 30 files
```

The skip remains the opt-in live-GCS transport test; the xfail remains the declared historical
degenerate-DTCWT comparison arm.

## T5.2b - worker-safe lazy Zarr and streaming train statistics

`open_cached_lazy` preserves the cache-only/no-network refusal but omits `.load()`. Cached
forecast preparation reads only coordinates and metadata eagerly, splits/embargoes the
timeline, then computes per-variable train-only population moments in bounded configurable
frame blocks. Blocks are converted to float64 and combined with Chan's stable parallel-
variance formula; normalisation provenance records the method and block size.

The constructor also verifies the manifest's physical Zarr time chunk is no larger than the
declared statistics block. A missing or oversized chunk is refused with an exact
`time_chunk=<N>` rematerialisation instruction. This matters because selecting seven frames
from a lazy array can still load an all-time physical chunk; bounded memory is therefore backed
by both the algorithm and the store layout.

`_LazyZarrValues` stores the cache path and canonical layout, not a live xarray object. Its
pickle state strips all handles, and its PID guard closes/reopens a handle inherited through
fork. Each DataLoader process therefore owns its local store handle. Sample reads combine the
history and targets into one unique index selection, check fetched values are finite and return
the same CPU tensor dictionary as T5.2a. `RegionalForecastBundle.close()` and context-manager
support release parent handles explicitly.

Acceptance tests demonstrate lazy/eager mean and standard-deviation agreement to `1e-12`,
tensor agreement to `1e-6`, a seven-frame upper bound during statistics fitting, and a forced-
spawn two-worker DataLoader after the parent has already read a sample. The broader focused
suite result is:

```text
134 passed, 1 skipped, 5 warnings in 34.57s
```

Clean full regression after adding explicit reader/worker teardown and the physical-chunk guard:

```text
957 passed, 1 skipped, 1 xfailed, 6 warnings in 164.23s
759 test functions across 30 files
```

The scientific boundary remains: training values are exhaustively checked while their moments
stream; validation/test values are checked on access, not by a second full-crop scan. No actual
independent ERA5 route or viable multi-year NZ crop has been acquired, so T5.2 and D43 remain
partial/open respectively.

## T5.3a - forecaster seam, persistence baseline and deterministic tiny backward pass

`src/forecasting/adapter.py` adds a physical-space `Forecaster` contract over the exact tensor
shapes emitted by `RegionalForecastDataset`. `PersistenceForecaster` is a zero-parameter exact
baseline: every requested lead is the last input state. It does not pass through a transform,
so transform reconstruction error cannot contaminate the baseline definition.

`ForecasterAdapter` wraps a one-step coefficient model between any accepted
`RepresentationModule` and its inverse. It requires the model output to preserve encoded shape,
dtype and device and refuses non-finite coefficients. Multi-lead output is an autoregressive
physical-state rollout. T5.3a uses the final history frame only and records that limitation,
the representation, rollout policy and model parameter counts in provenance.

The importable `run_tiny_deterministic_step` consumes a real default-collated dataset batch and
executes dataset -> Haar representation -> residual 1x1 coefficient model -> inverse -> MSE ->
`backward()`. Evidence includes the seed, device/dtype, all tensor shapes, parameter count, loss,
finite non-zero gradient norm and prediction/gradient hashes. Two fixed CPU runs are byte-
identical. The acceptance test also compares prediction and parameter gradients between CPU and
the available RTX through PyTorch's vendor-neutral `cuda` API; it passed. ROCm hardware itself
is still NOT RUN.

Focused acceptance:

```text
python -m pytest src/tests/test_forecasting_adapter.py -q
5 passed, 1 warning in 6.45s

python -m pytest src/tests/test_forecasting_adapter.py \
  src/tests/test_regional_forecast.py src/tests/test_training_representations.py -q
93 passed, 1 warning in 81.47s
```

Clean full regression from the moved external-drive workspace:

```text
962 passed, 1 skipped, 1 xfailed, 6 warnings in 227.51s
764 test functions across 31 files
```

The skip remains the opt-in live-GCS check and the xfail remains the declared historical
degenerate-DTCWT comparison. This run proves the integration seam, deterministic fixture and
autograd path. The tiny residual model is not any external model, is not trained to skill and supplies
no evidence that one representation forecasts better than another. Actual laboratory-model
configuration/checkpoint lineage and its train/evaluate protocol remain T5.3 outstanding work.

## T5.3b - verified model artifact and persistence-relative evaluation contract

`src/forecasting/artifact.py` records a laboratory model as a tensor-only PyTorch `state_dict`
and canonical JSON manifest. The record includes caller-declared model/representation config,
training provenance, fully qualified model class, parameter counts, state-schema hash and full
checkpoint SHA-256. Loading verifies config and file identity before `torch.load`, uses
`weights_only=True`, requires a string-to-tensor mapping and performs strict state-dict loading.
The caller constructs the model; no manifest value is imported or executed. The verified
artifact is bound into `ForecasterAdapter` provenance.

`src/forecasting/evaluation.py` evaluates a declared non-training split in bounded memory and
compares every prediction with physical-space persistence on exactly the same target. It reports
standardized RMSE, MAE, bias and MSE skill score per lead/variable, plus standardized-only
aggregation. Optional physical errors require explicit train-normalisation standard deviations
and never combine unlike units. A zero persistence MSE yields undefined (`None`) skill. Counts,
split/dataset/model provenance and prediction/target stream hashes are retained. The claim
boundary says this is a single-checkpoint result without multi-seed uncertainty or significance.

Focused and forecasting-path acceptance:

```text
python -m pytest src/tests/test_forecasting_artifact_evaluation.py \
  src/tests/test_forecasting_adapter.py -q
12 passed, 1 warning in 3.64s

python -m pytest src/tests/test_forecasting_artifact_evaluation.py \
  src/tests/test_forecasting_adapter.py src/tests/test_regional_forecast.py \
  src/tests/test_training_representations.py -q
100 passed, 1 warning in 25.60s
```

The accelerator evaluation test executed on the available RTX through PyTorch's vendor-neutral
`cuda` device API and passed. That same API is used by ROCm, but AMD hardware remains NOT RUN.
Seven T5.3b tests cover exact restore, checkpoint tamper refusal, configuration/manifest drift,
overwrite refusal, artifact-bound provenance, physical/standardized metric semantics,
perfect-persistence undefined skill, bad split/empty/misaligned refusal and CPU/accelerator parity.

The first full-suite run after adding the modules produced **2 documentation-integrity failures
and 967 passes**: the new source modules and test file were not yet in `architecture.md`. This was
an honest ordered gate failure, not a numerical failure. The inventory and claimed totals were
then updated before the clean rerun recorded below.

The actual external/laboratory architecture, weights, history semantics and declared training
recipe have still not been supplied or executed. This slice makes a real model verifiable and
evaluable when supplied; it provides no present evidence of forecast or representation skill.

Clean full regression after documentation reconciliation:

```text
969 passed, 1 skipped, 1 xfailed, 6 warnings in 197.56s
771 test functions across 30 `test_*.py` files
```

The skip remains the opt-in live-GCS check and the xfail remains the declared historical
degenerate-DTCWT comparison. `tools/audit_docs.py` reports no undocumented modules/routes, no
stale inventory rows, 45 defects (42 fixed, D18 partial, D17/D43 open), matching 969/1 suite
claims and `RESULT: ok`.

## T5.2c / D43a - resumable regional CDS acquisition, offline acceptance

`src/data_layer/cds_source.py` adds the direct regional acquisition route proposed by D43 while
keeping CDS outside the normal fallback chain. `CDSRegionalRequest` records canonical variables,
inclusive dates, explicit UTC hours, pressure levels, grid spacing and north/west/south/east
bounds. Calendar-month shards carry exact request hashes. Planning is network-free; acquisition
requires the existing explicit network gate and the optional `requirements-cds.txt` dependency.
Credentials remain exclusively in standard CDS client configuration and never enter a record.

Downloads use `.part` files, validate as NetCDF, receive full file hashes and are atomically
renamed before acquisition state advances. Resume re-hashes completed shards and refuses changed
or untracked files. Conversion normalises accepted CDS name/coordinate variants, requires the
exact requested timestamp/level/grid contract, refuses unresolved member/`expver` dimensions and
non-finite values, then writes the same bounded-time-chunk content-addressed Zarr cache consumed
lazily by `RegionalForecastDataset`. `rematerialise_cds_from_provenance` preserves the acquisition
route; the ordinary Zarr rematerialiser is not allowed to mistake CDS for a Zarr URI.

The command below was executed without network access and printed two exact monthly request
payloads plus their hashes:

```text
python -m src.data_layer.cds_source plan --date-start 2020-01-30 \
  --date-end 2020-02-02 --hours 0,6,12,18 --lat -46 -45 \
  --lon 170 171 --analysis-levels 1
network_used: false; monthly shards: 2
```

Offline focused acceptance:

```text
python -m pytest src/tests/test_cds_source.py -q
12 passed, 1 warning in 5.10s

python -m pytest src/tests/test_cds_source.py src/tests/test_zarr_source.py \
  src/tests/test_regional_forecast.py -q
78 passed, 1 skipped, 5 warnings in 31.18s
```

Clean full regression after documentation reconciliation:

```text
981 passed, 1 skipped, 1 xfailed, 6 warnings in 187.50s
779 test functions across 31 `test_*.py` files
```

The focused tests use a deterministic fake CDS client that writes real HDF5 NetCDF shards. They
prove our request, resume, validation, conversion and Dataset contracts; they do not prove the
current Copernicus queue, credentials, wire transfer or ERA5 values. No live CDS request,
multi-year NZ crop or WeatherBench overlap has run. D43 therefore remains open. T5.2d calendar
splits/physical lead durations and T5.3c protocol-selected model semantics are now explicit
planned roadmap tasks rather than implicit assumptions.

## T5.2d - exact calendar splits and physical forecast time

`RegionalForecastConfig` now supports exact `(validation_start, test_start)` calendar boundaries
without removing ratio mode. Each declared boundary must be an exact returned timestamp; the
configured embargo is excluded after it before samples are constructed. `expected_cadence_hours`
is checked against every interval when declared. Dataset schema v2 records measured cadence and
split mode, and every item derives `lead_durations_ns` from target timestamps relative to the
final input timestamp while carrying the cadence measured over the complete source axis.

Evaluation schema v2 requires input times, target times and derived durations. It recomputes the
durations, verifies one regular cadence maps every declared frame lead to hours, requires that
mapping to match the complete axis and remain constant across samples/batches, and publishes
`lead_durations_hours`. Refusal
tests cover missing timing, irregular samples, inconsistent duration/timestamp provenance,
cadence mismatch and non-exact calendar boundaries.

The metadata-only crop UI does not promote manifest eligibility into timing evidence: it displays
ratio dates as unfrozen, cadence as `NOT VERIFIED`, and physical lead labels as unavailable.

```text
python -m pytest -q src/tests/test_regional_forecast.py \
  src/tests/test_forecasting_artifact_evaluation.py \
  src/tests/test_forecasting_adapter.py src/tests/test_frontend_contract.py
56 passed, 5 warnings in 12.12s

cd frontend && npm run build
tsc passed; 1,385 modules transformed; built in 1m 27s
```

Rendered browser inspection is **NOT RUN**: the browser-control skill was followed, but no
controllable browser tool was attached to this session. This is stated explicitly rather than
substituting a build for visual evidence.

Clean full regression and documentation audit:

```text
985 passed, 1 skipped, 1 xfailed, 6 warnings in 165.37s
783 test functions across 31 `test_*.py` files
```

D43 remains open: no live CDS request, multi-year NZ crop or cross-route ERA5 comparison ran in
this slice. T5.2d proves the temporal contract, not the data acquisition or forecast skill.

## T5.0a - versioned, hashable motivating-experiment protocol contract

`src/forecasting/protocol.py` implements schema `motivating-forecast-protocol/v1`. A record must
name and hash its source laboratory evidence and provide exact sections for domain and coordinate
identity, dataset/cadence, histories and frame/physical leads, transform semantics, train-only
normalisation, calendar splits/embargo, optimiser/scheduler/training budget, rollout semantics and
model parameter counts. Exact-key parsing refuses absent or unknown fields. Placeholder values,
unsupported versions, invalid evidence/artifact hashes, non-training normalisation, inconsistent
physical leads, insufficient embargo and contradictory rollout feedback are refused.

Model and optimiser configuration JSON is recursively immutable after construction. Canonical
serialization produces a full SHA-256 and persisted envelopes refuse overwrite and verify that
hash on load. Focused acceptance, including all parametrised missing-section cases and adjacent
artifact/evaluator regression:

```text
python -m pytest -q src/tests/test_forecasting_protocol.py \
  src/tests/test_forecasting_artifact_evaluation.py \
  src/tests/test_forecasting_adapter.py
28 passed, 1 warning in 2.77s
```

The tests use a clearly fictional laboratory path, hashes and protocol. They prove contract
behaviour only. No external repository, configuration, coordinate hash or normalisation
artifact was supplied, so the actual motivating experiment is **NOT FROZEN**, T5.0 remains
partial and this slice makes no scientific-skill claim.

Clean full regression and documentation audit:

```text
1000 passed, 1 skipped, 1 xfailed, 6 warnings in 173.43s
790 test functions across 32 test_*.py files
tools/audit_docs.py: RESULT ok
```

## T5.0b - frozen protocol to runtime binding

`src/forecasting/binding.py` now prevents a complete protocol from becoming detached from the
objects actually executed. `bind_dataset_to_protocol` checks source/version, variables and level,
regular cadence, history/leads, embargo, exact UTC split instants, grid geometry/order/convention,
and train-only normalisation. It independently recomputes the full coordinate SHA-256 from the
recorded arrays and the statistics SHA-256 from the recorded normalisation values. Dataset grid,
timestamp and normalisation identities were upgraded from truncated 32-hex digests to truthful
64-hex SHA-256 values; source-cache content hashes retain their existing explicitly separate
contract.

`protocol_training_provenance` supplies the exact protocol/dataset/model-version/optimiser/rollout
identity that a training job must preserve. `bind_artifact_to_protocol` accepts a checkpoint only
when those identities plus model class/config/counts and complete transform semantics agree.
Bound evaluation uses schema `forecast-evaluation/v3`, carries the resulting combined binding,
and refuses another dataset, checkpoint, variable/lead family, or samples outside the declared
held-out split. Unbound generic evaluation remains schema v2 and cannot be mistaken for a bound
motivating-experiment run.

Focused protocol, binding, dataset and adjacent artefact/evaluation acceptance:

```text
python -m pytest src/tests/test_forecasting_protocol_binding.py \
  src/tests/test_forecasting_protocol.py src/tests/test_regional_forecast.py \
  src/tests/test_forecasting_artifact_evaluation.py -q
43 passed, 1 warning in 9.99s
```

The eight new executed cases include exact binding plus deliberate source, grid, cadence,
normalisation, split, unbound-training and cross-run substitution failures. All identities and
evidence are synthetic fixtures. No real external configuration, data crop, statistics,
checkpoint or scientific-skill evidence was created, so T5.0 remains partial and the UI remains
truthfully unpromoted.

Clean full regression and documentation audit:

```text
1008 passed, 1 skipped, 1 xfailed, 6 warnings in 174.11s
794 test functions across 33 test_*.py files
tools/audit_docs.py: RESULT ok
```

## T5.6a - offline FourCastNet 3 request/result boundary

`src/forecasting/external_fcn3.py` implements a dependency-free boundary between the portable
workbench and a future isolated Earth2Studio worker. Schema `external-forecast-run/fcn3-v1`
requires reviewed model/package/checkpoint/model-card/licence identities; pinned Earth2Studio,
PyTorch, torch-harmonics and environment-lock versions/hashes; the complete canonical 72-channel
input order; global 721x1440 coordinates and hash; source and normalisation identities;
timezone-bearing initialization times; six-hour rollout; one unique seed per intrinsic
stochastic member; output variables/format/reference; global-then-crop semantics; precision; and
an explicit prepare-only or accepted Linux/NVIDIA-worker placement.

Regional inputs, reordered/missing channels, external perturbation, naive timestamps, seed-count
drift, placeholders, extra fields and unverified Windows/AMD worker claims are refused. Planning,
canonical hashing and race-safe persistence import no Earth2Studio, Makani, torch-harmonics or
network client.

Schema `external-forecast-result/fcn3-v1` seals only an external-worker request. It binds the
request to selected outputs, global geometry and time contract, measured OS/device/hardware,
precision, wall time, peak RAM/VRAM, worker-log SHA-256 and a deterministic content hash. NetCDF
must be a file; Zarr must be a non-empty tree whose digest includes every relative name, file
size and byte stream without following symlinks. Result manifests have a derived hash, refuse
overwrite, verify against their originating request, and detect artifact changes. The result's
claim boundary says explicitly that no array schema, units, values, calibration, spectral
fidelity or forecast skill have been verified.

Focused acceptance:

```text
python -m pytest src/tests/test_external_fcn3.py -q
19 passed, 1 warning in 2.13s

python -m pytest src/tests/test_external_fcn3.py \
  src/tests/test_forecasting_protocol.py \
  src/tests/test_forecasting_protocol_binding.py \
  src/tests/test_forecasting_artifact_evaluation.py -q
50 passed, 1 warning in 3.79s
```

All model/data/checkpoint/environment hashes and artifacts in these tests are explicitly
synthetic. No NGC access, download, FCN3 dependency, global ERA5 state, worker, inference,
forecast-array validation, NZ crop or ensemble evaluation ran. RTX 5050, AMD, CPU and HPC FCN3
execution therefore remain `NOT RUN`.

Documentation audit after reconciliation:

```text
18 passed, 1 warning in 8.67s
803 test functions across 34 test_*.py files
tools/audit_docs.py: RESULT ok
```

Clean full regression for the delivered tree:

```text
1027 passed, 1 skipped, 1 xfailed, 6 warnings in 164.09s
```

## T5.6b - lazy canonical forecast cube and global-to-regional import

`src/forecasting/external_cube.py` authenticates a sealed T5.6a NetCDF4 file or Zarr tree before
xarray opens it. Schema `canonical-forecast-cube/fcn3-v1` requires the exact global dimensions,
UTC initialization axis, member seeds, six-hour physical lead durations, 721x1440 coordinate
values and coordinate hash, requested variable set, floating data and explicit canonical SI
units. NetCDF CF durations and Zarr timedeltas are normalized to the same nanosecond axis.

The finite-value pass is lazy but complete: it checks storage chunk sizes before computation,
refuses a decompressed chunk larger than the declared memory budget and then visits every value
one chunk at a time. An exact `GeographicBounds` subset remains Dask-backed and records the
request, result, artifact and validation hashes, inclusive endpoints and regional grid hash.
There is no implicit rounding, interpolation, longitude conversion, antimeridian wrap or default
NZ box. Tampered artifacts are refused before the xarray opener is called.

Focused acceptance over compressed synthetic global NetCDF4 and Zarr fixtures:

```text
python -m pytest src/tests/test_external_forecast_cube.py -q
15 passed, 1 warning in 8.09s
```

Adjacent forecasting/data acceptance:

```text
python -m pytest src/tests/test_external_forecast_cube.py \
  src/tests/test_external_fcn3.py \
  src/tests/test_forecasting_protocol_binding.py \
  src/tests/test_regional_forecast.py -q
54 passed, 1 warning in 14.51s
```

Documentation reconciliation:

```text
python -m pytest src/tests/test_documentation.py -q
18 passed, 1 warning in 8.12s
python tools/audit_docs.py
811 test functions across 35 test_*.py files
tools/audit_docs.py: RESULT ok
```

Clean full regression for the delivered tree:

```text
1042 passed, 1 skipped, 1 xfailed, 6 warnings in 163.16s
```

These tests contain constant synthetic fields, not FCN3 forecasts. The validation receipt proves
identity, canonical structure, units and finite values only. It does not prove meteorological
correctness, calibration, spectral fidelity or forecast skill. No Earth2Studio dependency,
checkpoint, worker, initial condition or real external forecast ran.

## T5.6c - matched-truth regional ensemble evaluation

`src/forecasting/ensemble_evaluation.py` evaluates a validated lazy regional ensemble against
verifying truth and the observed field at forecast initialization on exact shared timestamps,
lead durations, coordinates, variables and SI units. Both observation cubes require completed
writes, source SHA-256 identities and an explicitly held-out, matching split. The regional
forecast's request/result/artifact/validation and coordinate identities are rechecked. Training
truth, altered coordinates, reordered variable axes, changed units, missing variables and any
non-finite value are refused; there is no interpolation or complete-case deletion.

For each variable and lead, analytic acceptance covers member and ensemble-mean RMSE/MAE/bias,
the physical persistence baseline, `1 - ensemble_mean_MSE / persistence_MSE`, exact empirical
ensemble CRPS, RMS population ensemble spread, the explicitly uncorrected spread/RMSE ratio and
rank bins. Spatial scores use measured cosine-latitude weights. Exact ties are distributed
uniformly and fractionally over every admissible truth rank. Zero denominators are represented
as `None`, not infinity. Variables in different units are not aggregated. Input reads remain
Dask-backed and are materialised as bounded regional source tiles; the receipt records observed
and allowed source-tile bytes and SHA-256 identities of the evaluated value streams.

Focused analytic acceptance:

```text
python -m pytest src/tests/test_external_ensemble_evaluation.py -q
14 passed, 1 warning in 3.40s
```

Adjacent FCN contract/cube/evaluator acceptance:

```text
python -m pytest src/tests/test_external_ensemble_evaluation.py \
  src/tests/test_external_forecast_cube.py \
  src/tests/test_external_fcn3.py -q
48 passed, 1 warning in 9.32s
```

Clean full regression for the delivered tree:

```text
1056 passed, 1 skipped, 1 xfailed, 6 warnings in 190.57s
```

The fourteen cases use small analytic synthetic arrays with known answers; they are not weather
forecasts. This slice validates metric definitions and refusal behavior only. Rank histograms
are diagnostics, not independent draws or hypothesis tests, and no uncertainty interval is yet
reported. No FCN3 checkpoint, Earth2Studio worker, real forecast, ERA5 truth match, calibration,
spectral-fidelity test or scientific skill claim ran. UI skill/calibration results therefore
remain unavailable rather than fabricated.

## T5.6d - lazy matched ERA5/CDS truth builder

`src/forecasting/matched_truth.py` constructs the two exact observation cubes required by T5.6c
from a canonical regional pressure-level xarray dataset and its source manifest. It computes
each valid time as initialization plus the forecast's nanosecond lead and selects analyses by
labelled integer index. Synthetic acceptance confirms the outputs remain Dask-backed and enter
the evaluator directly. The source axis must be datetime64, unique and strictly increasing;
every initialization and valid time must exist exactly. Grid coordinates, the single 850-hPa
level, source alias, floating dtype and canonical SI units must match. Tests refuse nearest-time
substitution, regridding, approximate levels, unit conversion and ambiguous aliases.

The split contract applies to targets as well as initializations. Every required timestamp must
remain in the declared non-training interval. `fresh_post_2019_holdout` refuses any overlap with
FCN3's published 1980-2015 train, 2016-2017 test and 2018-2019 evaluation partitions;
`published_partition_diagnostic` is required when those dates are intentionally inspected and
is itself refused for a wholly post-2019 sample. Pre-1980 dates are outside the v1 declaration
and refused. The deterministic receipt binds the complete source-manifest SHA-256, its existing
materialized-value digest, forecast/grid identity, variables/aliases, level, split, period role,
initializations, leads, valid-time digest and exact-selection semantics. T5.6c preserves and
compares that builder lineage between truth and initialization.

Focused synthetic acceptance:

```text
python -m pytest src/tests/test_matched_truth.py -q
14 passed, 1 warning in 3.39s
```

Adjacent T5.6 request/cube/truth/evaluator acceptance:

```text
python -m pytest src/tests/test_matched_truth.py \
  src/tests/test_external_ensemble_evaluation.py \
  src/tests/test_external_forecast_cube.py \
  src/tests/test_external_fcn3.py -q
62 passed, 1 warning in 10.74s
```

Clean full regression for the delivered tree:

```text
1070 passed, 1 skipped, 1 xfailed, 6 warnings in 181.88s
```

All field values and forecasts in these tests are synthetic. No CDS request, ERA5 independent
overlap check, FCN3 worker, model checkpoint, global initial condition, real forecast or real
verification score ran. The receipt proves exact matching and declared period lineage, not
source independence, meteorological correctness, calibration, spectral fidelity or skill. The
UI therefore still has no FCN3 scientific result to display.

## T5.6e - reproducible external-evaluation run orchestration

`src/forecasting/evaluation_run.py` composes the accepted T5.6b-d seams without adding network
or model-worker behavior. `EvaluationRunConfig` is versioned and hashable and requires explicit
grid-aligned bounds, non-training split dates, FCN3-period role, the 850-hPa bridge and positive
forecast-chunk/evaluation-tile byte limits. `run_external_evaluation` authenticates and scans the
sealed global artifact, crops it exactly, opens an existing content-addressed ERA5 cache through
the local-only lazy path, builds exact truth, runs bounded ensemble/persistence evaluation and
closes both stores on every exit path. Existing output is refused before either store opens.

Successful execution publishes canonical JSON from a flushed same-directory temporary file via
an atomic no-overwrite filesystem operation (rename on Windows; hard link on POSIX). The receipt includes run/config identity, forecast validation,
regional lineage, complete ERA5 manifest, matched-truth selection, all metrics and evaluated
value-stream hashes. Reload verifies the outer SHA-256, nested run/config/evaluation hashes and
cross-section forecast, validation, ERA5-source and truth-builder identities. These hashes detect
corruption and substitution; they are not cryptographic signatures.

Focused synthetic acceptance:

```text
python -m pytest src/tests/test_evaluation_run.py -q
10 passed, 1 warning in 5.61s
```

Adjacent T5.6b-e acceptance:

```text
python -m pytest src/tests/test_evaluation_run.py \
  src/tests/test_external_forecast_cube.py \
  src/tests/test_external_ensemble_evaluation.py \
  src/tests/test_matched_truth.py -q
53 passed, 1 warning in 12.88s
```

Clean full regression for the delivered tree:

```text
1080 passed, 1 skipped, 1 xfailed, 6 warnings in 190.66s
```

The end-to-end acceptance run uses synthetic Zarr forecast and ERA5 stores with analytically
inspectable values. No CDS network request, independent ERA5 overlap check, FCN3 worker,
checkpoint, global initial condition or real forecast ran. The receipt proves pipeline execution
and identity for supplied inputs, not independence, calibration, uncertainty, significance,
generalisation, spectral fidelity or meteorological skill. No FCN3 result is exposed in the UI.

## T5.6f - portable laptop/HPC evaluation job runner

`src/forecasting/evaluation_job.py` separates one machine-independent scientific job from its
machine-local filesystem paths. `EvaluationJob` strictly embeds and hashes the validated FCN3
request/result, ERA5 `CropSpec` and T5.6e evaluation config. `EvaluationPathBindings` has its own
hash, contains only forecast/cache/receipt paths and must name the exact job hash. Relative paths
resolve from the bindings file, so moving a job between Windows and Linux/shared storage does not
change scientific identity.

The `create`, `bind`, `preflight` and `run` CLI commands execute without Earth2Studio, CUDA or a
network client. Preflight authenticates the complete forecast artifact, opens the existing
local-only ERA5 cache, verifies crop identity, refuses an existing output and writes nothing.
Run invokes T5.6e, reloads its atomic receipt and proves that config/request/result/artifact/crop
identity matches the portable job. Jobs and bindings also use atomic no-overwrite publication.
No scheduler is contacted or simulated.

Focused synthetic acceptance:

```text
python -m pytest src/tests/test_evaluation_job.py -q
9 passed, 1 warning in 29.08s
```

Adjacent T5.6a-f acceptance:

```text
python -m pytest src/tests/test_evaluation_job.py \
  src/tests/test_evaluation_run.py src/tests/test_matched_truth.py \
  src/tests/test_external_ensemble_evaluation.py \
  src/tests/test_external_forecast_cube.py src/tests/test_external_fcn3.py -q
81 passed, 1 warning in 14.79s
```

Clean full regression for the delivered tree:

```text
1089 passed, 1 skipped, 1 xfailed, 6 warnings in 177.06s
```

The focused end-to-end cases use synthetic Zarr forecast and ERA5 stores. They prove portable
contract identity, relocation, authenticated local preflight, refusal behavior, CLI operation
and deterministic receipt binding. They do not show that any real laptop or HPC allocation was
used, that FCN3 or CDS ran, or that any real forecast has accuracy, calibration, spectral
fidelity or skill. No evaluation result was added to the UI.

## T5.6g - receipt-backed API/UI reporting

`src/forecasting/evaluation_report.py` sends no receipt directly to the browser. It first uses
the T5.6e verifier for outer, nested and cross-lineage hashes, requires an exact accepted ERA5
catalogue URI with no synthetic/fixture declaration, flattens metrics without combining units,
and removes machine-local paths. Admitted JSON is stored under the receipt SHA-256 and reverified
on every list/get. Deliberately corrupted files are omitted rather than partially rendered.

The API accepts JSON uploads only and provides content-addressed list/get routes. The tenth
Forecast Evaluation tab renders no metrics or plots in the empty state. A verified report shows
ensemble-mean and persistence errors, persistence-relative MSE skill, CRPS, spread/skill,
diagnostic rank frequencies and their tie policy, exact dates/domain/level/split/member/grid
scope, model/checkpoint/content identities and both claim boundaries. It always reports
`scientific_skill: NOT_ESTABLISHED`; a point estimate is not uncertainty or generalisation.
The ERA5 label is explicitly a checked source declaration, not a digital signature over values.

Focused receipt/run/job acceptance:

```text
python -m pytest src/tests/test_evaluation_report.py \
  src/tests/test_evaluation_run.py src/tests/test_evaluation_job.py -q
24 passed, 2 warnings in 11.19s
```

Frontend static acceptance:

```text
npx tsc --noEmit
0 errors

npm run build
1386 modules transformed; built in 1m 25s
```

The five T5.6g cases use synthetic forecast/truth arrays. One test temporarily installs an
explicit controlled catalogue declaration solely to exercise the admission/display path; the
ordinary fixture is refused and leaves the store empty. Therefore no real ERA5/FCN3 receipt was
created, imported or displayed, and the new tab has not been visually inspected. The tests prove
the reporting boundary, not source authenticity, meteorological skill or calibration.

Clean full regression for the delivered T5.6g tree:

```text
1094 passed, 1 skipped, 1 xfailed, 6 warnings in 215.71s
```

## Proprietary ownership and named researcher licence

`LICENSE.md` now records Edward Jonathan Bentley as owner and provides for a designated Named Licensee with a
perpetual, worldwide, royalty-free licence to use and modify SpectralEarth for lawful personal,
academic, research and commercial work, including institutional/cloud/HPC execution. The core
cannot be publicly redistributed, sold as a platform or sublicensed without Edward's separate
written permission. Narrow collaborator access, independent extensions, future upstream
contributions, third-party materials, scientific responsibility, warranty, breach and New
Zealand governing law are addressed separately rather than compressed into “do what he likes.”
The repository describes this accurately as proprietary and not open source.

A documentation test pins the owner/licensee names and contacts plus the substantive grant and
restriction boundary:

```text
python -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 9.18s
```

Clean full regression after adding that guard:

```text
1095 passed, 1 skipped, 1 xfailed, 6 warnings in 196.84s
```

This evidence proves repository consistency only. The bespoke licence has not been reviewed by
a New Zealand intellectual-property lawyer and is not represented as professional legal advice.

## T4C.5d - bounded, content-bound gate readiness and D17 closure

The remaining D17 boundary-ring reductions now use single-pass grouped reductions and match
independent loop oracles. On float64 512x512 input with 256 rings, the old
`decompose_by_boundary` path measured 0.148814 s mean and the vectorised path 0.007700 s mean,
a 19.3x speed-up.

The direct CDS conversion no longer retains all monthly shards in memory: it validates and
appends declared time blocks to a temporary Zarr store, verifies the complete time axis,
streams the chunk-independent logical content hash and publishes atomically. Train-only
harmonic climatology and exact two-pass scale signatures retain one source/coefficient frame.
The frozen `GateStudyPlan` binds source, crop, transform, climatology and statistical protocol;
local preflight uses the chosen filters' exact parent-grid support for both the R13 interior and
advection floor. Receipts are no-overwrite and hash authenticated. Synthetic acceptance remains
explicitly `scientific_verdict: NOT_ESTABLISHED`.

Focused evidence:

```text
D17 analysis/boundary acceptance                         85 passed
CDS and Zarr bounded materialisation/storage             73 passed, 1 skipped
streamed scale signatures                                35 passed
streamed climatology/benchmark acceptance                46 passed
cross-scale protocol and exact support floor             60 passed
synthetic cached gate plan -> authenticated receipt       1 passed
documentation consistency                                19 passed
```

Clean full regression for the delivered tree:

```text
1104 passed, 1 skipped, 1 xfailed, 6 warnings in 178.68s
```

No `.cdsapirc` was present and the optional `cdsapi` dependency was not installed during this
verification. Therefore no network request, multi-year NZ crop, independent-route overlap or
real T4C.6 gate was run. D43 remains open, and Phase 4D-4H remains gated on that atmospheric
PASS/FAIL rather than on synthetic plumbing evidence.

D50 storage acceptance adds two refusal tests. The estimator assumes no compression credit,
groups download and cache requirements by physical volume, and the insufficient-space case
proves that the fake CDS client's call list remains empty. At verification time D: had
1,180.1 GiB free; this observation is operational context only, because the live command will
recheck capacity against the final frozen request.

## T4C.5e - independent ERA5 evidence is executable, bounded and content-bound

Static review found D51: `cross_check_era5_overlap` produced a useful in-memory report, but the
real gate accepted only the mutable manifest string `independent_overlap_check: PASS`. There
was no path from comparison to durable evidence, and no proof that PASS belonged to the CDS
bytes, request, WeatherBench bytes, variable or level presented to the gate.

`src/data_layer/era5_overlap.py` now opens existing caches only, selects exact coordinates
without interpolation, checks compatible declared units, and compares each variable in bounded
frame blocks using frozen relative and variable-specific absolute tolerances. It publishes one
atomic no-overwrite receipt binding both content hashes, the CDS request, coordinate hash,
level, variables, tolerances, error metrics and memory bound. The CDS manifest embeds that
receipt and its SHA; real gate preflight recomputes and validates them. A one-cell perturbation
is durably recorded as FAIL and is explicitly refused as gate authority; receipt tampering and
comparison-design drift are also refused.

Offline acceptance:

```text
CDS acquisition + gate job focused suite       18 passed
full repository suite                          1106 passed, 1 skipped, 1 xfailed
```

The WeatherBench fixture in these tests is derived locally from deterministic fake CDS output.
It proves the mechanism and refusal boundary only. No Copernicus request, real WeatherBench
value comparison, multi-year crop or T4C.6 atmospheric verdict ran; D43 remains open.

## T4C.5f - frozen canary-first acquisition campaign

D52 was an ordering and identity defect: the full acquisition, WeatherBench overlap and gate
plan could be operated separately, so the multi-year transfer was effectively the first live
integration test of CDS decoding. A mismatch discovered afterwards would waste the expensive
transfer, while manually coordinated JSON could drift in dates, grid, variables or cadence.

`GateCampaign` now binds the full request, an exact contained CDS canary, a catalogued
WeatherBench overlap and the real `GateStudyPlan`. Reconstruction is strict down through nested
provenance: unknown fields and stale derived hashes are refused. The preflight aggregates
worst-case storage for both NetCDF downloads, both CDS Zarr caches and the WeatherBench cache by
physical volume without compression credit. It checks dependency, standard configuration-file
or environment presence and explicit consent, but never reads a secret, constructs a client or
uses the network.

The frozen order requires a canary overlap PASS before full acquisition and a second overlap
receipt bound to the full cache before T4C.6. Six tests cover immutable round-trip/tamper
refusal, canary containment, WeatherBench grid/cadence matching, frame-count drift, aggregate
storage/readiness and the machine-readable freeze/preflight CLI.

```text
campaign acceptance                              6 passed
campaign + gate + CDS focused acceptance        24 passed
full repository suite                         1112 passed, 1 skipped, 1 xfailed
```

These tests use synthetic request identities and mocked readiness signals. No scientific
campaign design was selected or saved, no dependency or credential was installed, and no live
network request or atmospheric verdict occurred.

## T4C.5g - the preflight's distance unit was wrong (D53-D54)

The world-class design audit found D53 in the decisive R4 guard. `GridSpec` correctly stores
lat/lon `dx` in degrees, but `support_floor` selected bare `dx` before any metric field and
treated ERA5's `0.25` as metres. A db2 level-3 support of 22 pixels therefore appeared 5.5 m
wide instead of hundreds of kilometres, generally collapsing the declared advection floor to
one frame. The receipt would have looked precise while admitting the exact short lags the rule
exists to exclude.

Physical grid provenance is now reconstructed through `GridSpec`; angular axes are converted
using the spherical metric and the largest physical cell axis across the crop is used when flow
direction is not frozen. A 161x161, 0.25° test spanning 20-60°S now measures more than 27 km per
parent pixel and gives a three-frame level-3 floor at 10 m/s and six-hour cadence. The test
explicitly asserts that the provenance still contains `dx=0.25`, proving the fix is unit
conversion rather than a changed fixture.

D54 was the matching timing defect. R13 geometry and physical lags were checked only when the
full cache already existed. `GateCampaign` now computes actual SWT/DTCWT supports, valid parent
interiors and the metric support floor before acquisition, and refuses undersized crops or lag
families below it. Its report also records split sizes, annual-cycle coverage, family size,
power and an upper bound on surrogate statistic evaluations. CDS requests require integer grid
intervals; ingestion separately refuses shifted endpoints even when their spacing is correct.

```text
cross-scale + CDS + campaign + gate focused acceptance    52 passed
full repository suite                                   1116 passed, 1 skipped, 1 xfailed
```

No atmospheric protocol was selected in T4C.5g and no live data was accessed. The corrected
floor means the eventual lag family may be longer than earlier planning implied; that is a
scientific correction, not a parameter to relax for convenience. T4C.5h below records the
subsequent preregistration.

## T4C.5h - primary scientific campaign preregistration

The primary T4C.6 campaign is now checked in as
`campaigns/t4c6_nz_era5_temperature_850_v1.json` and authenticated by campaign SHA-256
`84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975`. It freezes one
field and one family before atmospheric values are acquired: 850-hPa temperature, 2018--2022,
six-hour cadence, the 161x161 20--60 S / 140--180 E grid, three db2 SWT levels, energy-density
transfer entropy, lags 3--8 frames, six bins, 4,999 surrogates, BY at 0.05, eight embargo frames
and seed 20260821.

The local review reports 7,304 total frames; 4,382 train, 8 embargo and 2,914 test; 139x139
deepest valid parent interior; a deepest physical support floor of three frames; 36 hypotheses;
and a BY minimum of 3,005 surrogates. Its calendar split is explicit down to the timestamp.
The acceptance test pins the complete campaign hash and review output, so changing any nested
request or statistic is visible. The review command constructs no client and uses no network:

```text
campaign preregistration acceptance                     8 passed
campaign review network_used                             false
full repository suite                                  1117 passed, 1 skipped, 1 xfailed
```

The real frozen preflight was then run against the repository's intended D: locations. It
estimated 4.75 GiB working bytes across the full/canary NetCDF and Zarr artifacts plus the
WeatherBench overlap, required another 5 GiB reserve, observed 1,180.06 GiB free and passed the
storage gate. Status was correctly `BLOCKED` on `cdsapi` not installed, standard CDS credential
configuration absent and explicit network consent disabled. `client_constructed` and
`network_used` were both false; no secret value was inspected.

This is protocol evidence only. No CDS credential was validated, no WeatherBench or CDS byte
was transferred, no overlap receipt passed and no T4C.6 verdict exists. D43 remains open and
4D--4H remain gated.

## TG5.1 - durable motif freezing (`ed-dev`)

`src/core/motif_freeze.py` closes the process boundary TG3.5 explicitly left open. A
`FrozenMotif` serialises the exact dimensionless exemplar graph, matcher, configuration size
and measured tolerance under `definition_sha256`. `motif_sha256` additionally binds the
originating `DomainDeclaration`, every node's carried semantic record, the training partition
and its digest, the generate-family identity and selection record, study identity and a
timezone-bearing freeze time. Structure remains separate from carried domain meaning.

The in-memory record recursively freezes nested mappings. Persistence emits canonical JSON,
uses exclusive creation, flushes and `fsync`s, and refuses overwrite. Reload rejects schema
drift, incomplete origin meaning, non-canonical bytes, an invalid reconstructed graph and any
partition/definition/outer hash mismatch. A published-digest check distinguishes local
self-consistency from evidence that the definition existed before a later operation. TG5.2,
not this slice, must record that digest before opening the target domain.

Focused acceptance after the final constructor hardening:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_motif_freezing.py src/tests/test_motif.py src/tests/test_preregistration.py src/tests/test_cross_domain.py -q
153 passed, 1 warning in 48.37s
```

The 14 TG5.1 tests cover structural/origin separation and dual hashes, exact graph round-trip,
deterministic canonical bytes, nested immutability, no-overwrite publication, structural and
semantic tamper detection, whole-artifact rewrite versus a published digest, held-out-source
refusal, label/signature redefinition, origin laundering, explicit timezone, strict fields and
non-canonical serialization.

The final complete repository run, after architecture and roadmap updates:

```text
> .\.venv\Scripts\python.exe -m pytest -q
1986 passed, 1 skipped, 1 xfailed, 6 warnings in 831.86s (0:13:51)
```

Standalone scientific gates and documentation audit:

```text
> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1684
stale inventory rows : none
claimed suite totals : architecture (1986, 1) / roadmap (1986, 1)
RESULT               : ok
```

This is serialization, integrity and provenance evidence over synthetic motif fixtures. It is
not a blind transfer: no target domain was opened, no transfer search was run and no real
cross-domain scientific result exists. Those were deferred to TG5.2 at this checkpoint.

## TG5.2 - bind, open, then search without redefinition (`ed-dev`)

`src/core/motif_transfer.py` closes the chronological boundary left by TG5.1. The only target
data entry is a callback. Before invoking it, `TransferLedger` verifies the frozen artifact
against its externally published `motif_sha256`, requires strict timezone-bearing freeze/bind/
open order, and durably writes the motif and definition identities plus the exact target
partition and domain declarations. It treats the target as spent at that write even if target
loading or validation later fails. Reload verifies canonical bytes, exact schemas, record,
partition and declaration hashes, state and chronology.

The subsequent exhaustive search exposes no size, matcher, relation or tolerance parameters;
all come from the verified frozen definition. It refuses a matcher that has not declared
cross-domain capability, a source domain presented as a target, mislabelled target features and
a post-open subset of the frame count bound before access. Match reports retain both domains'
carried semantics, and a content-addressed receipt records all examined/matching configurations,
including a complete zero-match result.

Focused and integrated acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_motif_transfer.py -q
18 passed, 1 warning in 0.77s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_motif_transfer.py src/tests/test_motif_freezing.py src/tests/test_motif.py src/tests/test_preregistration.py src/tests/test_cross_domain.py -q
171 passed, 1 warning in 47.19s
```

The 16 TG5.2 test functions (18 parametrized cases) cover durable pre-opener binding, strict
chronology, absence of a definition-override surface, semantic separation, wrong published
digest and non-cross-domain matcher refusal, one-use target identity across processes, failure
spending, target-domain laundering, post-open subset refusal, complete null and exhaustive
search receipts, canonical reload, nested ledger tampering and receipt identity.

Final repository, documentation and scientific-gate verification:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2004 passed, 1 skipped, 1 xfailed, 6 warnings in 820.53s (0:13:40)

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 102.61s (0:01:42)

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1700
stale inventory rows : none
claimed suite totals : architecture (2004, 1) / roadmap (2004, 1)
RESULT               : ok

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

This establishes binding-before-opening for access performed through the API, not proof that an
archive was never inspected by a person or another program; external access control must supply
that fact. The acceptance target is synthetic and no real archive was opened. TG5.2 reports a
descriptive motif transfer search only; corrected relationship transfer is supplied by TG5.3.

## TG5.3 - corrected relationship transfer on target train and test (`ed-dev`)

`src/core/motif_relationship.py` adds the corrected inferential layer deliberately excluded from
TG5.2. `RelationshipPlan` binds the published frozen motif, target domain, ordered non-overlapping
train/test identities, outcome x positive-lag `SearchSpecification`, one-sided mean-difference
statistic, circular-shift null, ensemble, alpha, correction and seed. R21 precedence admission,
the declared lag floor and G3 affordability are checked before the plan exists. Canonical
no-overwrite persistence and a separately published plan digest protect the declaration across
processes.

Execution exposes no analysis override parameters. `RelationshipLedger` verifies both published
digests and durably spends both partitions before either opener, globally refusing reuse of one
partition even with a new counterpart. Every frozen family member is evaluated in both splits;
non-estimable members remain at p=1. The full family is corrected independently in train and test,
and only the same positive rejected label in both yields `PASS`. An adequately powered empty
intersection yields the complete `FAIL` result required by the roadmap.

Focused acceptance after final ledger and reload hardening:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_motif_relationship.py -q
16 passed, 1 warning in 2.46s
```

G3/G5 integration:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_motif_relationship.py src/tests/test_motif_transfer.py src/tests/test_motif_freezing.py src/tests/test_motif.py src/tests/test_family_accounting.py src/tests/test_preregistration.py src/tests/test_cross_domain.py -q
228 passed, 1 warning in 46.55s
```

The 16 TG5.3 tests cover the complete content-addressed family and split declaration, absence of
an execution-time redefinition surface, both-opening precommit, same-label corrected replication,
planted PASS and null FAIL, no-motif p=1 accounting, wrong published digest before access,
failure spending, target subset/outcome laundering, split chronology, R21 and lag-floor refusal,
family affordability, canonical plan round-trip, plan/ledger tampering, global partition reuse and
receipt identity.

Direct `RelationshipPlan` construction, factory construction and reload all execute the same
domain reconstruction, R21/floor, split, family-identity and affordability checks; the public
dataclass is not a bypass around its factory.

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 101.28s (0:01:41)

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1716
stale inventory rows : none
claimed suite totals : architecture (2020, 1) / roadmap (2020, 1)
RESULT               : ok
```

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2020 passed, 1 skipped, 1 xfailed, 6 warnings in 822.43s (0:13:42)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

This is synthetic instrument evidence, not a real cross-domain relationship result. Circular
shift protects the observed marginal series and destroys their alignment, but it does not rule
out a shared driver. A `PASS` therefore supports replicated precedence/association only; it does
not establish mechanism, causality or predictive utility. As in TG5.2, the ledger proves API
ordering and external access control must establish that the archive was not inspected earlier.

## TG6.1 - the hashed, append-only evidence bundle (`ed-dev`)

`src/core/evidence.py` opens Phase G6 with the structure the claim ladder will read. An
`EvidenceBundle` binds one `Hypothesis` - identifier, statement, prediction, timezone-bearing
registration time and provenance - to an ordered chain of `EvidenceEntry` records under ten
first-class scientific fields: `observations`, `effect_sizes`, `uncertainty`, `null_results`,
`replication_results`, `holdout_performance`, `provenance`, `confounders`,
`contradictory_evidence` and `failure_states`.

`append()` returns the next immutable snapshot and leaves the receiver unchanged byte for byte,
so a prior revision stays a citable object. Each entry hashes its own body including the previous
entry's digest, anchored on a digest over schema, study id, creation time and hypothesis. Sequence
numbers are gap-free, append chronology is non-decreasing, payloads are deep-frozen and required
to be non-empty finite JSON, and every entry must name one of the ten fields. There is no
`commentary`, `notes` or `interpretation` route, so free text cannot enter the structure the TG6.2
gates read; TG7 may record adversarial commentary beside the bundle, never inside it (R22).

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_evidence_bundle.py -q
24 passed, 1 warning in 2.30s
```

The 24 tests cover revision zero complete before any evidence; all ten fields routed and queryable;
the receiver unchanged byte for byte across an append; the previous-digest chain; entries and
nested payloads read-only; `contradictory_evidence` and `failure_states` carried on the same chain
and undeletable by later appends; commentary, prose-only and empty payloads, non-finite and
unserialisable values, invalid statuses, malformed or duplicated source digests and offset-less or
backwards timestamps all refused; a hypothesis refused after the bundle it anchors and refused when
swapped under an existing chain; edited, dropped, reordered, substituted and never-linked entries
and a mismatched bundle digest all detected; canonical exclusive publication, no-overwrite,
published-digest verification, reload refusing dropped, softened, relocated, unknown-field and
reskinned files; process-independent reload continuing the same chain; and a digest sensitive to
append order, not only content.

One of these deserves naming. Dropping, reordering and duplicating entries are all caught by the
sequence check alone, which means a test built only from those cases never exercises the hash link
at all. `test_substituting_an_earlier_entry_breaks_the_link_the_later_entry_committed_to` supplies
a well-formed replacement for entry 1 with the correct sequence, category and chronology but
different content, and a successor whose sequence is right but which was never linked to its
predecessor. Only the bound previous digest rejects these, so the chain property is tested as
itself rather than as a side effect of numbering.

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2044 passed, 1 skipped, 1 xfailed, 6 warnings in 815.33s (0:13:35)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 99.32s (0:01:39)

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1740
stale inventory rows : none
claimed suite totals : architecture (2044, 1) / roadmap (2044, 1)
RESULT               : ok
```

**An audit gap found while verifying this slice.** The first confirming full run returned
`1 failed, 2043 passed`: the inventory's `| **total** |` row still said 1716 against an actual
1740. `tools/audit_docs.py` had already reported `RESULT: ok` on that same tree. It checks the
per-file inventory rows and the two claimed suite totals but never the inventory total row, so it
can pass while `test_documentation.py::test_documented_test_counts_match_the_source` fails.
Architecture 7.4 presents the tool as reporting the same facts outside a test run; on this row it
does not. The row is corrected and the suite is green, but the tool remains the weaker of the two
checks and should not be treated as sufficient on its own. Not fixed here - it is outside TG6.1
and belongs in the defect register.

**Claim boundary.** This is a tamper-evident container and a routing discipline, not a judgement.
It does not decide whether the evidence inside supports anything: TG6.2's ladder and TG6.3's five
outputs own that, and no rung logic exists yet. The hashes detect edits to a published bundle;
they do not authenticate an author, and they cannot show that relevant evidence was gathered and
simply never appended. Only what is appended can be weighed.

## TG6.2 - the claim ladder (`ed-dev`)

`src/core/claim_ladder.py` assigns `observation -> association -> robust association ->
candidate precursor -> demonstrated predictive utility` from a TG6.1 bundle and nothing else.
`assess_claim_ladder(bundle)` takes no second argument and reads no clock, filesystem,
environment or random source, so the verdict is recomputable by anyone holding the published
snapshot. Ten declarative gates decide the climb; only `PASS` advances one.

Two gates sit on the floor rung and dominate the rest. Any `FAIL` or `INVALID` entry anywhere,
and any `contradictory_evidence` or `failure_states` entry recorded as `PASS` - the bundle
asserting that the contradiction or failure stands - caps the bundle at `observation`. Because
TG6.1 entries are immutable, a failure cannot be appended away. Causal claim kinds are refused by
`permits()` rather than answered `False`, naming R7.

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_claim_ladder.py -q
30 passed, 1 warning in 2.46s
```

**How the purity requirement was actually tested.** Determinism to the digest; two separately
built bundles with the same evidence agreeing; independence from append order over forty
shuffles; a bundle whose labels read `DEFINITIVE PROOF OF CAUSATION` and whose payloads carry
`interpretation: causal` and `rung: demonstrated_predictive_utility` receiving byte-identical
gates to a plain one; and an identical rung after a round trip through `save_evidence_bundle` and
`load_evidence_bundle`. The strongest of them restates the ladder rule independently, in the test
file, and compares the two implementations over a randomised sweep of eight hundred bundles.

**How the blocking requirement was actually tested.** Exhaustively over every first-class field
at every status appended to a fully evidenced bundle - fifty cases, each asserted to block or not
block exactly as the rule says - and over eight hundred randomised whole bundles. A blocked
bundle stays on the floor as eleven further rounds of complete favourable evidence are piled on
it, and a later `PASS` provenance entry claiming to supersede the failure does not lift it.

**A weak property test found and replaced.** The first randomised sweep drew categories and
statuses uniformly. Instrumenting it showed the result: all six hundred bundles landed on
`observation`, four hundred and eighty-five of them blocked and the rest simply short of
evidence. That sweep would have passed unchanged against a `assess_claim_ladder` that returned
`"observation"` unconditionally, so it was testing almost nothing. The generator now seeds a
random subset of the full evidence set before adding noise, and the tests assert their own
coverage: all five rungs must occur in the agreement sweep, and both the blocked and the clear
branch must occur more than a hundred times each in the blocking sweep.

**Mutation check.** Four deliberate defects were introduced into the gate table one at a time and
the focused suite re-run against each: ignoring standing contradictions (7 failures), removing the
blocking cap so `rung` follows `unblocked_rung` (9), accepting a truthy precedence value instead
of exactly `true` (2), and dropping the uncertainty gate (3). Each was caught, and the file was
restored from a backup between runs.

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2074 passed, 1 skipped, 1 xfailed, 6 warnings in 801.61s (0:13:21)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1770
stale inventory rows : none
claimed suite totals : architecture (2074, 1) / roadmap (2074, 1)
RESULT               : ok
```

**Claim boundary.** The ladder grades the evidence that was appended and cannot know what was
never gathered. It is a floor on rigour, not a certificate: reaching the top rung means a holdout
result was recorded and passed, not that the design was sound, that the holdout was honestly held
out, or that the effect transfers. Gate thresholds are deliberately structural - one passing entry
of the right kind - and say nothing about the statistical adequacy of what that entry contains;
TG5's instruments own that upstream. No rung of this ladder, including its top, licenses a causal
reading.

## TG6.3 - the five outputs of an evidence bundle (`ed-dev`)

`src/core/five_outputs.py` closes Phase G6's deterministic layer. For any TG6.1 bundle,
`summarise_evidence(bundle)` states what can be claimed, what cannot, what evidence contradicts
it, which alternative explanations remain, and which single observation would most efficiently
distinguish between them. It takes the bundle and nothing else - no clock, no filesystem, no
environment, no randomness - so the report is recomputable by anyone holding the snapshot, and
`summary_sha256` binds it to the exact revision it came from.

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_five_outputs.py -q
38 passed, 1 warning in 3.27s
```

**How the invariants were actually tested.** Two randomised sweeps carry the phase, each over 600
bundles drawn to land anywhere on the ladder. The first asserts that the claimable rungs are
exactly those at or below the ladder's verdict; that claimable and not-claimable partition the
ladder with every unreachable rung carrying a non-empty explanation; that the evidence against is
exactly the adverse and contrary entries in append order and is a superset of the ladder's
blocking set; and that the open structural alternatives mirror the unsatisfied climbing gates
exactly. The second asserts the next observation follows its stated precedence - unblock, then
climb, then resolve, then nominate nothing. Both sweeps assert their own coverage: the first
requires all five rungs to occur, the second requires every branch of the precedence including the
empty one, so neither can pass vacuously on a degenerate sample.

**A defect the sweep found.** The first run of the partition sweep failed on
`all(item.blocked_by for item in outputs.not_claimable)`. Because the ladder is climbed in order,
a bundle can satisfy every gate a rung declares and still not reach it - evidence recorded out of
order, so `robust_association`'s own gates pass while `association`'s do not. The original code
named the floor gates in that case, which are satisfied here, and so reported an unreachable rung
with an empty explanation. It now walks down from the rung to the floor and names the nearest rung
that actually blocks the climb. `test_a_rung_whose_own_gates_all_pass_is_still_explained_by_the_gap_beneath_it`
tests the case directly rather than leaving it to the sweep.

**Mutation check.** Five deliberate defects were introduced one at a time and the focused suite
re-run; the file was restored between runs.

```text
drop the null-result widening of output three               -> 2 failed, 36 passed
climb before unblocking in output five                      -> 3 failed, 35 passed
report structural alternatives their gate already closed    -> 4 failed, 34 passed
claim one rung more than the ladder allows                  -> 4 failed, 34 passed
nominate the last recorded alternative rather than the first -> 1 failed, 37 passed
```

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2112 passed, 1 skipped, 1 xfailed, 6 warnings in 833.93s (0:13:53)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1808
stale inventory rows : none
claimed suite totals : architecture (2112, 1) / roadmap (2112, 1)
RESULT               : ok
```

**Claim boundary.** The fifth output is a precedence rule, not an experiment design. No expected
information gain is computed, because the bundle carries no likelihoods to compute one from;
"most efficient" means "the cheapest thing standing in the way", and a reader who wants a genuine
design of experiments will not find one here. The fourth output can only name the eight
alternatives its gates correspond to and the ones a person recorded, so a domain-specific rival
explanation nobody wrote down is invisible to it - its absence from the report is not evidence of
its absence in fact. The same holds of the third output, and the rendered text says "none
recorded; that is not the same as none existing" rather than letting silence be read as
reassurance. Nothing here grades the statistical adequacy of any entry, and no output, at any
rung, licenses a causal reading (R7).

## TG7.1 - the recorded-call boundary (`ed-dev`)

`src/core/recorded_call.py` opens Phase G7. It is the only door through which the adversarial
review layer may speak, and it is built so the layer cannot reach the G6 gates through it. Every
call captures the verbatim request, the verbatim response bytes, the exact model id, the effort
setting, the API request id and both timestamps, because an LLM output cannot be regenerated
(R23); every response is constrained to a declared schema sent as `output_config.format`; and no
recorded output is an input to any claim level (R22).

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_recorded_call.py -q
55 passed, 1 warning in 1.80s
```

**The acceptance test of the phase.** `test_deleting_every_llm_output_from_a_corpus_of_bundles_changes_no_claim_level`
builds a corpus of seven bundles - one standing on each of the five rungs, one blocked by a
failed replication, one carrying a standing contradiction - and reviews each with all eight
TG7.2 roles. The commentary is deliberately assertive: it returns `supported`, asserts the
relationship is causal, and demands promotion to demonstrated predictive utility by name. Every
claim level, every claimable set and every summary digest is identical after deleting the whole
review, and the corpus asserts its own coverage, so it cannot pass by standing on one rung.

`verify_claim_independence` makes the same guarantee executable outside the test. It compares the
claim digest with the review present and deleted, re-derives it from the bundle's own canonical
bytes, and refuses if any recorded phrase of at least `SMUGGLING_FLOOR` characters is found
inside them. That last check is the one that could actually fail, and
`test_recorded_commentary_found_inside_the_evidence_chain_is_refused` shows it firing on a bundle
where a reviewer's argument has been copied into a confounders summary.

**How the invariants were actually tested.** Two randomised sweeps carry the rest. The first
draws 300 bundles landing anywhere on the ladder, attaches between zero and four calls of random
role, effort and verdict, and asserts the claim digest, rung and blocking set are untouched and
that the five outputs are byte-identical after stripping the review; it requires all five rungs
to occur and at least one bundle to have been reviewed at all, so it cannot pass vacuously. The
second draws 200 records of one to five calls and asserts the recorded chain stays self-checking
and reloadable through canonical JSON, requiring every chain length to occur.

**A masked check the mutation run found.** The fifth mutation - letting a call bind a bundle
other than the one under review - passed all 54 tests at first. The test that should have caught
it appended a call from another bundle, but that call's `previous_sha256` did not link up either,
so the chain check refused it first and the binding check was never exercised.
`test_a_call_bound_to_another_bundle_is_refused_even_when_the_chain_would_accept_it` now
constructs a stray call whose digest does link up and whose subject does not, and the mutation is
caught.

**Mutation check.** Five deliberate defects were introduced one at a time and the focused suite
re-run; the file was restored between runs.

```text
do not check the response against its declared schema        -> 4 failed, 50 passed
accept an answer from a model that was not asked             -> 1 failed, 53 passed
allow sampling parameters to be pinned                       -> 5 failed, 49 passed
skip the smuggling check in verify_claim_independence        -> 1 failed, 53 passed
let a call bind a bundle other than the one under review     -> 1 failed, 54 passed
```

The first four were run against the 54-test suite; the fifth against the 55 cases that include
the isolating test described above.

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2167 passed, 1 skipped, 1 xfailed, 6 warnings in 1241.16s (0:20:41)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1858
stale inventory rows : none
claimed suite totals : architecture (2167, 1) / roadmap (2167, 1)
RESULT               : ok
```

**Claim boundary.** This slice records calls and fences them off; it does not conduct a review.
No test here judges whether a challenge was any good, and a well-formed response saying something
false is recorded exactly as faithfully as a true one. The transport is injected and every test
uses a recorded one, so nothing here demonstrates that a real API client behaves as the boundary
expects - that is TG7.3's problem, along with the `cache_read_input_tokens` assertion the usage
block is being kept for. The smuggling check finds recorded wording reproduced in a bundle; it
cannot detect a person who reads commentary, is persuaded by it, and records a genuine-looking
measurement in their own words. No structural check can, and the defence against that is the
provenance the gates already require, not this function. Determinism is not claimed anywhere in
the layer: an identical request may return a different answer tomorrow, which is exactly why the
answer is stored rather than recomputed.

## TG7.2 - the adversarial round-robin (`ed-dev`)

`src/core/round_robin.py` runs the eight seats in their fixed order over one frozen bundle, every
turn taken through the TG7.1 boundary and recorded verbatim beside it. The order is replayed
rather than trusted, the panel is pinned seat by seat, dissent is retired by argument and never by
arithmetic, and the final synthesis must carry every unresolved dissent by name.

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_round_robin.py -q
50 passed, 1 warning in 7.17s
```

**The acceptance of the slice.** Two tests carry it.
`test_a_complete_round_robin_over_a_corpus_of_bundles_changes_no_claim_level` runs a full
eight-role exchange over seven bundles - one on each of the five rungs, one blocked by a failed
replication, one carrying a standing contradiction - with the candidate and all four challengers
naming the current rung and arguing it should be promoted to demonstrated predictive utility.
Every claim level, claimable set and summary digest is identical afterwards, the corpus asserts
its own coverage of the ladder, and `close_round_robin` re-runs `verify_claim_independence` on the
way out. `test_three_agreeing_challengers_do_not_retire_the_fourths_objection` carries the other
half: one challenger objects, is answered `unresolved`, and its argument and unexcluded
alternatives survive into the outcome verbatim.

**Not majority voting, made checkable.**
`test_what_the_agreeing_challengers_said_makes_no_difference_at_all` runs the same exchange twice,
changing the other three challengers' verdicts between runs, and asserts the retained dissent is
byte-identical. If any count were happening anywhere in the module, that test would catch it.
Three more tests refuse a final synthesis that drops an unresolved dissent, invents one nobody
raised, or reports `dissent_remains: false` while one stands.

**The closure rule.** A dissent is retired by the candidate conceding it, or by a rebuttal that
the independent reassessment declines to reopen; until the reassessment has spoken a rebuttal is
provisional, because the candidate does not get to mark its own homework. Four tests cover the
four paths through that rule, and a fifth refuses a reassessment that tries to originate a dissent
at a point where nothing downstream would answer it.

**How the invariants were actually tested.** Two randomised sweeps. The first draws 120 exchanges
with random verdicts, dissent flags, response outcomes and reopenings, computes the expected
retained dissent from an independently written rule in the test, and asserts the claim digest,
rung and five outputs are untouched; it requires all four rungs and at least three distinct
dissent counts to occur, so it cannot pass on a degenerate sample. The second draws 80 exchanges
over randomly seated panels and re-replays each finished record against the protocol, requiring
every possible exchange length from seven to eleven turns to occur.

**A missing guard the mutation run found by hanging.** The sixth mutation removed the check that a
response answers the dissent currently on the floor. The suite did not fail: it hung. With that
check gone, an answer naming the wrong challenger leaves the right one permanently unanswered, the
plan keeps demanding the same turn, and the exchange never terminates. The ordering check had been
carrying a termination guarantee that nothing stated. Each challenge is now refused a second
answer, which bounds the exchange independently of the ordering check, and
`test_a_dissent_answered_twice_is_refused_so_the_exchange_has_to_terminate` isolates it - the
ordering check would otherwise mask it, exactly as the chain check masked a binding check in
TG7.1. With the guard in place the sixth mutation fails cleanly.

**Mutation check.** Six deliberate defects were introduced one at a time and the focused suite
re-run; the file was restored between runs.

```text
let a dissent be retired by the candidate's own rebuttal alone -> 2 failed, 48 passed
let the final synthesis drop an unresolved dissent            -> 2 failed, 48 passed
stop pinning the panel to the seat that answered              -> 2 failed, 48 passed
let a reassessment originate a dissent nothing will answer    -> 1 failed, 49 passed
let a challenge find the claim unsupported without dissenting -> 2 failed, 48 passed
answer dissents in any order rather than the order raised     -> 1 failed, 49 passed
```

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2217 passed, 1 skipped, 1 xfailed, 6 warnings in 934.21s (0:15:34)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

Documentation and inventory:

```text
> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 59 defined, 57 fixed, partial ['D18'], open ['D43']
test functions       : 1907
stale inventory rows : none
claimed suite totals : architecture (2217, 1) / roadmap (2217, 1)
RESULT               : ok
```

**Provider independence.** The module names no vendor and opens no socket. `model_id` is any
string, `effort` is an abstract three-valued knob a later adapter translates into whatever the
provider takes, and the closed response schema is enforced locally by `ResponseSchema.validate` on
the parse - so an undeclared field is refused whether or not the provider honoured
`additionalProperties: false`. The tests seat two model ids and mix them across the panel, which
exercises the pinning but proves nothing about any real client.

**Claim boundary.** This slice conducts the exchange; it does not judge it. Nothing here measures
whether a challenge was any good, whether a concession was warranted, or whether a rebuttal was
honest - a fluent, false objection is retained as faithfully as a sound one, and a lazy panel that
raises no dissent produces a clean outcome that means nothing at all. Independence is checked at
the level of the model id and no deeper: `reassessment_is_independent` calls two different ids
independent, but two sizes of one family share training data, tokenizer and failure modes, and it
is the correlated blind spot that an adversarial exchange exists to catch. A tiered panel drawn
from one provider - a lighter model challenging, a larger one synthesising - buys cost control and
a capability gradient, not independence in the sense the word carries. Seating genuinely unrelated
reviewers is a configuration decision the panel records and does not make, which is also why
reviewer overlap is recorded rather than refused. Every test uses a recorded transport, so
nothing here shows that a real client behaves as the protocol expects. And retention is not
resolution: an outcome carrying four unresolved dissents is an honest record of an argument
nobody won, not a finding.

## TG7.3 - cost-controlled review transport (`ed-dev`)

`src/core/review_cost.py` adds a provider-neutral route/usage audit and the first concrete API
transport. `GeminiBatchTransport` maps TG7.1 calls onto Gemini 3.5 Flash's asynchronous inlined
Batch GenerateContent endpoint, polls the named operation, preserves the raw structured response
and `usageMetadata`, and returns exactly the transport mapping `record_call` already accepts.
The API key is header-only, has no serialisation path, and is redacted from `repr` and provider
error messages.

Official provider material reviewed 2026-08-26:

* `https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5` - GA model id
  `gemini-3.5-flash`, structured output, Batch/caching support, and low/medium/high thinking;
* `https://ai.google.dev/gemini-api/docs/caching` - implicit caching on Gemini 2.5+, the
  4,096-token floor for Gemini 3.5 Flash, and measured cached-token usage;
* `https://ai.google.dev/api/batch-api` - inlined requests/responses, `batches/{id}`, polling and
  terminal states; and
* `https://ai.google.dev/gemini-api/docs/pricing` - Batch at 50% of standard token price.

The cost policy fixes one real model id and every effort before the review: low for the four
challengers, medium for the response, high for candidate synthesis, independent reassessment and
final synthesis. `audit_review_cost` independently reconciles normalized input/output/cached/total
counts against the raw provider object, requires the Batch route and at least one measured cache
hit, and binds the aggregate to both the review-record and policy digests. The receipt persists
atomically without overwrite and rechecks its content digest on load. It records tokens and the
dated pricing source, not a dollar figure that would become false when a price changes.

Focused acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_review_cost.py -q
19 passed, 1 warning in 1.67s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_review_cost.py src/tests/test_recorded_call.py src/tests/test_round_robin.py -q
124 passed, 1 warning in 8.67s
```

The 15 test functions (19 cases) cover the exact Batch payload, schema and thinking-level
translation; submission and polling; both the reference and first live completed-operation
shapes; structured response and raw usage preservation; visible-plus-thinking output accounting;
header-only key handling; safe HTTP and failed-batch errors; an eight-call 90%-cache-hit receipt;
and refusals for configured-but-zero cache, standard service, effort drift, raw/normalized
disagreement, batch-id splicing, duplicate identities, invalid arithmetic, overwrite and tampering.

Full acceptance:

```text
> .\.venv\Scripts\python.exe -m pytest -q
2236 passed, 1 skipped, 1 xfailed, 6 warnings in 857.91s (0:14:17)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 60 defined, 58 fixed, partial ['D18'], open ['D43']
test functions       : 1922
stale inventory rows : none
claimed suite totals : architecture (2236, 1) / roadmap (2236, 1)
RESULT               : ok
```

**Live smoke, and D60.** After `GEMINI_API_KEY` was supplied through the ignored `.env.local`,
three tiny calls were attempted. The first authenticated and completed but exposed that the
operation's `metadata` and `response` are siblings rather than the direct resource shape used by
the offline fixture. After that parser fix, the second completed and exposed a worse issue: the
legacy structured-output fields were accepted but ignored, returning valid JSON outside the
declared schema; it also showed that billed output is visible candidates plus thinking tokens.
The current `responseFormat` field initially returned a safe 400 because Batch requires protobuf
enum `APPLICATION_JSON`, not the synchronous REST example's `application/json`. With that dialect
pinned and local response validation added, the final batch succeeded:

```text
batch        : batches/nd6n27...mb26
model        : gemini-3.5-flash
response     : {"note": "Structured Batch transport is operational for test TG7.3a live smoke.", "status": "ok"}
usage        : 52 input, 32 candidate, 92 thinking, 176 total
normalised   : 52 input, 124 output, 0 cached
```

The key was read from `.env.local`, which `git check-ignore` confirms is ignored, and was never
printed or written into an artifact. Authentication, submission, polling, structured output and
usage are therefore live-accepted. The smoke prompt is below Gemini 3.5 Flash's 4,096-token cache
floor and correctly hit no cache, so a real cache hit, full eight-role review and live cost receipt
remain **NOT RUN**. The offline non-zero cache fixture tests the auditor, not Google's cache.

**Claim boundary.** Cost routing and token accounting say nothing about whether a challenge is
good. The receipt cannot reach the EvidenceBundle or any G6 gate, and caching does not make an LLM
answer reproducible. R22 and R23 are unchanged.

## TG7.4 - translation, bounded (`ed-dev`)

`src/core/translation.py` renders a finding in domain language. It is the first point in the
programme where text is produced for a human to act on, and therefore the point where four rules
break at once if nothing stops them: R19's forbidden semantic comparison, R9's bare confidence
figure, R7's causal verbs entering through prose rather than through claim kinds, and R22's
promotion carried out entirely in wording.

**No model is called.** Domain wording is declared, content-hashed data screened at registration;
the renderer emits only from closed template sets bound to structural facts.

**A gap closed on the way.** R9 names six figures - support, confidence, base rate, lift with an
interval, and surrogate-corrected lift. None of them had a structured home anywhere in `src/`
before this slice; `claim_ladder` asks only whether an `effect_sizes` entry passes and never reads
what is inside one, so the figures lived unvalidated in a payload mapping. `AssociationFigures`
keeps all six together, refuses a partial set by name, and checks lift against
`confidence / base_rate` rather than trusting it. Its `render` is the only method in the module
that can format a percentage:

```text
observed in 40 occurrences: 82.0% of the time, against a base rate of 40.0%
(lift 2.05, interval 1.60 to 2.50, 1.90 after surrogate correction)
```

That is the roadmap's own cautionary example, made honest rather than banned.

Three rules are structural rather than checked. **R19:** two disjoint template sets selected by
`semantic_key` equality, the cross-domain set reaching only `structural_signature()`, so a
cross-domain magnitude sentence cannot be constructed. **R9:** `AssociationFigures` cannot exist
partially, and nothing else formats a percentage. **R22:** `translate` takes a `FiveOutputs`,
never a `ReviewedBundle`, so commentary has no parameter through which to arrive.

**Acceptance.** A corpus standing on all five rungs plus a blocked bundle and a contradicted one,
translated into an atmospheric and a financial vocabulary, gives documents that read completely
differently and assert an identical set of structural facts, with every claim digest unchanged. A
hostile glossary attempting six promotions in wording alone is refused at registration, by name,
for each one.

**Evidence.** `src/tests/test_translation.py`, 47 tests (60 cases). Three randomised sweeps: 600
documents over both vocabularies against an independently written restatement of the fact set; 400
checking numeral containment, bare-confidence and causal-vocabulary refusal with and without
figures; 300 feature pairs, half sharing a `semantic_key`, asserting no unit-bearing field ever
leaves a cross-domain rendering.

**Two mutations initially proved worthless, and that is recorded rather than tidied away.** Nine
deliberate mutations were applied to the module itself. Seven were caught on the first attempt.
The eighth - removing the causal guard - appeared to be caught, but the edit pattern never matched,
so the run had been against unmutated source; re-applied correctly, it is caught. The ninth showed
the two halves of `assert_no_bare_confidence` are **fully redundant**: removing either leaves the
other catching every case. The per-unit loop is kept for its diagnostics and is now documented as
redundant in the module, so a later reader does not mistake belt for braces.

**Claim boundary.** A translation is faithful to the record, not to the world. A glossary mapping a
structural term to a misleading-but-non-causal domain word is accepted, because no structural check
knows what "anomaly" means to an oceanographer; the defence is that the glossary is declared,
hashed and reviewable, not that it is correct. Refusing causal *vocabulary* is not refusing causal
*implication*: a reader who reads "precursor" as "cause" is caught by nothing here. The R19 guard
stops the system emitting a cross-domain comparison; it cannot stop a reader setting two
within-domain renderings side by side and drawing one themselves, and layout is out of scope.
Nothing judges whether a finding was worth translating.

## TG9.1 - the read-only claim surface (`ed-dev`)

Before this slice the cross-domain line had **no HTTP surface at all**. All twelve G-line modules
- `domain`, `feature`, `motif`, `evidence`, `claim_ladder`, `five_outputs`, `recorded_call`,
`round_robin`, `translation`, `cross_domain`, `constellation`, `family` - had zero references in
`src/api/`. Every existing route belonged to the atmospheric/transform line. Nothing a browser
could reach knew a claim ladder existed, which is why Phase G9 is ordered API first.

`src/api/findings.py` mounts six read-only routes under `/api/v1/findings`. Nothing appends
evidence, records a call or moves a rung: a GET cannot change what may be claimed (R22), asserted
over the bundle bytes on disk and the rung before and after.

**R9 on the wire.** `refuse_bare_confidence` walks every response body at any depth and refuses a
`confidence` key without all six of R9's figures. R9 is usually described as a frontend
constraint, which puts it in the one place it cannot be enforced. The guard restates the six
field names rather than importing them from `AssociationFigures`, because a guard that imported
its expectations from the thing it guards would agree with any change made to it; a test asserts
the two statements still agree. A partial figure set is served as no figures at all.

**The D35 defence.** Registration is eager at module import, not an import side effect of a
lazily imported handler module. D35 was exactly that: `GET /data/sources` returned
`['netcdf_local', 'simulated']` on a fresh process and a longer list after the researcher visited
the ERA5 tab, so which sources the fallback chain considered depended on browsing order.
`DOMAIN_GLOSSARIES` has that shape. `register_builtin_glossaries()` is idempotent and returns the
full built-in set so a caller can assert it rather than hope.

**Two glossaries, written against the screens.** `src/core/builtin_glossaries.py` supplies
reanalysis and order-book wording for all 38 structural terms each. Both passed TG7.4's four
registration screens on the first attempt - no digit, nothing from `OUTSIDE_THE_LADDER`, no
comparative asserting a relation of size, and no rung phrase borrowing wording reserved above it.

**Acceptance, both criteria met.**

1. *No route can serve a bare confidence.* Asserted over a corpus that genuinely does report one -
   the test fails if no confidence is present anywhere, so it cannot pass vacuously.
2. *A new domain reaches the API without editing `src/api/`.* A glossary registered from the test
   module, outside `src/core` and `src/api` both, appears in `GET /domains` and is served whole by
   `GET /glossaries/{name}`. TG8.1's condition carried onto the HTTP layer.

One study served through both vocabularies reads completely differently and returns byte-identical
`structural_keys`:

```text
reanalysis : "a link whose earlier and later parts are ordered in time and whose working
              can be traced back to the archive it came from"
order_book : "a link whose earlier and later parts are ordered in trading time and whose
              working can be traced back to the venue feed"
keys       : identical in both
```

**Evidence.** `src/tests/test_findings_api.py`, 20 tests.

**Claim boundary.** A surface that cannot serve a bare confidence does not make the science behind
it good; it removes one way of misreading it. A fluent rendering of a weak result is more
persuasive than a jargon-laden rendering of the same result - a risk this surface creates rather
than removes. An empty "evidence against" section means nothing was recorded, not that nothing
exists. The store reads a directory; it does not establish that anything in it was worth
publishing.

**Delivered short of the declaration, recorded rather than glossed.** The TG9.1 declaration said
`GET /domains` would carry each domain's declared violations (E15) and lag policy (R21), "so a
client can show what a domain refuses as readily as what it permits". It does not. The route lists
registered **glossaries** - wording - not `DomainDeclaration`s: `src/api/findings.py` contains zero
references to `violations`, `lag_policy`, `precedence_admissible` or `DomainDeclaration`. The two
acceptance criteria above are genuinely met and the route is honestly named for what it serves, but
the refusal half of the declared scope was not built, and is moved explicitly to TG9.3. Until then
**a domain's refusals are unreachable from the API**: nothing a client can call reveals that a
domain forbids a precedence claim.

**This surface is read-only in the strong sense.** It has no write path at all - zero POST, PUT or
DELETE routes - so nothing in the discovery pipeline is reachable through it: not mining, family
declaration, preregistration, motif freezing, blind transfer, nor adversarial review. Twenty-two
G-line modules remain without any HTTP surface; only `evidence`, `five_outputs` and `translation`
are exposed. This renders claims that some other process already produced and published to disk.

## TG9.2 / TG9.4 - the findings view (`ed-dev`)

`frontend/src/components/FindingsView.tsx` is an eleventh tab rendering a `TranslatedFinding`:
five outputs as five sections, each claim welded to its bound, with panels for the untranslated
claim state, the glossary that worded it, and the evidence bundle. A domain selector renders one
study through any registered vocabulary. Nothing in the existing workbench was refactored.

**The acceptance criterion is asserted over the source.** R9 calls a bare confidence percentage a
hard constraint on the frontend, not only on the mining code - a rule of that shape cannot be
enforced by whoever writes the JSX remembering it. Two tests enforce it mechanically:

* `test_the_findings_view_formats_no_scientific_number` - no `toFixed`, no `toPrecision`, no
  percent literal.
* `test_the_findings_view_reads_no_claim_bearing_field_directly` - no read of `confidence`,
  `base_rate`, `lift`, `support` or `surrogate_corrected_lift`. `figures_text` is the only route
  to the association strength.

Comments are stripped before both checks, so the component can document the constraint without
appearing to breach it.

**A gap found by writing the view, recorded rather than quietly fixed.** The first draft
interpolated `figures.support` into its own panel. No formatting, no arithmetic - and still wrong,
because a view that builds that line from parts puts R9 back in the hands of the JSX author. The
API now serves `figures_text`, the line the backend assembled, so the view has nothing to build.
The test was tightened from "formats no number" to "reads no claim-bearing field" because of that
mistake, not in anticipation of it.

**Three deliberate mutations of the component, each caught:** a `toFixed` on a rendered value, a
read of `figures.confidence`, and a bare percent literal.

**A guard that had stopped covering new code.** `test_route_count_claim_matches_reality` scanned
`@app.*` decorators in `main.py` alone. TG9.1 mounted the findings surface as an `APIRouter` in its
own module, so six real endpoints were invisible to the guard whose entire job is refusing an
undocumented endpoint - it passed while covering less than it claimed. `_routes()` now scans every
route-defining module with its mount prefix, and the documented count moved from 31 to 37.

**TG9.4 applies to this surface only.** The findings views carry `role="tablist"`/`role="tab"`,
`aria-selected`, `aria-pressed`, `aria-label`, `htmlFor`-bound labels, visible focus rings and
`aria-hidden` on decorative icons, asserted by test. `roadmap.md` §1's platform-wide measurement is
unchanged: the legacy workbench was not touched. This stops the new surface adding to the debt; it
does not repay it.

**Evidence.** Five new tests in `src/tests/test_frontend_contract.py` (36 total). `npx tsc
--noEmit` clean. `npm run build` succeeds.

**Claim boundary. Rendered appearance is NOT RUN.** No browser has displayed this tab and no
screenshot exists in this repository, exactly as for the tenth tab before it. The contract tests
prove the tab compiles, calls routes that exist and reads fields that are present; they do not
prove it renders, is legible, or is usable. A view that cannot render a bare confidence does not
make the finding it displays worth reading.

## TG9.3 - the refusal surface (`ed-dev`)

Delivers what the instrument will not do, and carries the half of TG9.1 that was declared and not
built. `DOMAIN_DECLARATIONS` is a registry in `src/core/domain.py`; `src/core/builtin_domains.py`
supplies the two declarations behind the built-in vocabularies.

The pair is chosen to make the tension visible rather than to look tidy:

```text
reanalysis : violations = ()                      lag_policy = advective  precedence = True
order_book : no_physical_metric, no_propagation_speed,
             unordered_channels, aggregated_values lag_policy = none       precedence = False
```

`refusals_for` draws each consequence from `KNOWN_VIOLATIONS` rather than restating it, so what a
reader is told and what the analysis layer enforces cannot drift (R17). A test asserts the two are
the same string.

**The constraint that shapes the slice.** An `EvidenceBundle` does not record which domain
produced it - its fields are the hypothesis, the ten evidence categories and their digests, and
nothing else. So nothing here checks a study against a domain. `DOMAIN_ATTRIBUTION_CAVEAT` travels
with every served limit, and a contract test refuses a view that restates the caveat in its own
words instead of rendering the sentence the API vouched for.

**`unadmitted_reading`.** When a record stands at `candidate_precursor` or above and the selected
vocabulary belongs to a domain declaring no lag floor, the surface reports it:

```text
glossary=reanalysis  -> unadmitted_reading: null
glossary=order_book  -> "This record stands at a rung that asserts temporal ordering, and the
                         order_book domain declares no admissible lag floor, so R21 does not
                         permit a lead-lag reading from it. Nothing here changes the rung..."
```

Verified that this moves no claim: the summary digest and the rung are identical with and without
the report. A rung below `candidate_precursor` reports no tension in either domain, so the check
is not firing indiscriminately.

**Acceptance met.** A blocked bundle and a contradicted one render their refusals. Commentary is
rendered in its own `<section>`, and a structural test refuses any `TranslationUnit` field inside
that container - recorded argument cannot be mistaken for what the record permits.

**Evidence.** 9 new tests in `src/tests/test_findings_api.py` (29 total), 4 in
`test_frontend_contract.py` (40 total). Two deliberate mutations of the view were each caught:
claim text moved inside the commentary container, and the caveat restated in the component instead
of rendered from the payload. `npx tsc --noEmit` clean; `npm run build` succeeds.

**Claim boundary.** Showing what a domain refuses does not enforce it: R17's refusals live in the
analysis layer and this displays the same facts rather than adding a check. A domain declaring no
violations is not unconstrained - reanalysis breaks nothing only because the inherited assumptions
were written against it. **The attribution gap is real and unclosed:** until a bundle records its
domain, `unadmitted_reading` is a statement about a vocabulary a reader chose, not about a study.
Closing it means putting domain provenance into a G6 structure, which belongs to no declared slice.
Rendered browser inspection remains **NOT RUN**.

## TG8.1 - the onboarding contract (`ed-dev`)

Makes a domain's declaration and its wording one indivisible act. `src/core/onboarding.py`
registers glossary, declaration and contract record atomically through `onboard_domain`; every
screen runs before the first `Registry.add`, and any failure restores all three registries.

**The hole.** Wording and limits registered through two registries that knew nothing about each
other, so either could exist alone. Wording without a declaration is a domain that speaks fluently
and refuses nothing - not hypothetical, since TG9.1 shipped exactly that and served it for a slice
before TG9.3 caught it. `register_builtin_domains` now goes through the contract and
`register_builtin_glossaries` delegates to it, so no entry point can quietly reproduce that state;
a built-in found half-registered is repaired rather than skipped.

**The recipe is a tuple.** `REQUIRED_DECLARATIONS` carries the seven requirements with the reason
for each. `OnboardedDomain.checklist()` generates from it, `GET /api/v1/findings/onboarding`
serves it, the tests assert against it:

```text
axes | geometry | lag_policy | violations | licence | provenance | glossary
```

**The one check nothing else could make.** The geometry/violation biconditional, which no single
object can see because the two facts live in different registries:

```text
geometry           physical_metric   violations                     verdict
latlon             True              ()                             accepted   (reanalysis)
None               -                 no_physical_metric, ...        accepted   (order_book)
latlon             True              irregular_sampling, ...        accepted   (argo_float)
pixel              False             no_physical_metric             accepted
cartesian          True              no_physical_metric             REFUSED - metric supplied then renounced
None               -                 irregular_sampling             REFUSED - lengths claimed, none available
tripolar           unregistered      -                              REFUSED - UnknownNameError, by the registry that owns the vocabulary
```

Asked of the geometry's declared `physical_metric` capability rather than its name (E2), so a
fourth geometry registered from outside `src/` answers for itself.

**`/domains` now lists the union of both registries.** A domain that declared its limits and never
declared its wording was invisible - the TG9.1 omission with its halves swapped. It now appears
with `glossary_sha256: null`, `term_count: 0` and `onboarding.complete: false`. `audit_onboarding`
reports a piecemeal domain as incomplete with what is missing and why, rather than letting it pass
for one that was checked whole.

**Acceptance met.** `src/tests/domain_plugin_example.py` onboards a third domain in one file
outside `src/`. The six files a domain would otherwise have had to touch are hashed before and
after the import and asserted byte-identical - the method `test_registries.py` uses for sources
and actions:

```text
src/core/domain.py  src/core/onboarding.py  src/core/translation.py
src/core/builtin_domains.py  src/core/builtin_glossaries.py  src/api/findings.py
```

The domain is **Argo profiling floats**, chosen by R17's reasoning rather than by sector. Of the
seven entries in `KNOWN_VIOLATIONS` the two built-ins between them break four;
`irregular_sampling` and `non_stationary_support` had never been broken by any registered domain,
so no refusal depending on them had ever fired against a declared source. Argo breaks exactly
those two, is the first declared domain where precedence is admissible while the clock is
irregular, is the only one exercising `lag_policy="declared"` (floor: one park-and-profile cycle,
with the basis recorded), and keeps a physical metric - the side of the biconditional neither
built-in occupies. Its thirty-eight-term glossary passed TG7.4's four registration screens first
time, written against them rather than fixed up afterwards.

`onboarded_by` is captured from the calling frame, because `Entry.defined_in` records
`value.__module__`, which for a `DomainGlossary` is always `src.core.translation` - the class's
home, never the adapter's. The test asserts the plugin domain is attributed to
`src.tests.domain_plugin_example` and not to anything under `src.core`.

**Evidence.** 33 new tests in `src/tests/test_domain_onboarding.py`. Five deliberate mutations,
each caught:

```text
M1  drop the metric-renunciation half of the biconditional   -> 3 tests failed
M2  remove the rollback from onboard_domain                  -> 1 test failed
M3  infer completeness from the declaration registry         -> 1 test failed
M4  skip a half-registered built-in instead of repairing it   -> 1 test failed
M5  list only the glossary registry in /domains              -> 1 test failed
```

M3 was caught by a different test than predicted: the piecemeal fixture registers only a glossary,
so a declaration-derived `complete` is still false there, and the failure surfaced in the
declaration-only listing test instead. Recorded rather than tidied, because a mutation caught by
an unexpected test is evidence about the tests, not only about the code.

The rollback test injects a failure between the writes with `monkeypatch`, since every ordinary
refusal runs before the first write and would never reach that path. A rollback nobody executes is
a rollback nobody has checked.

`GET /api/v1/findings/onboarding` is a new route, so the frontend reachability guard fired
(`test_no_served_route_is_unreachable_from_the_ui`). It was honoured rather than exempted: the
findings view gains a sixth panel, *How this domain was declared*, rendering the recipe and each
domain's audit, and the domain selector marks an incomplete declaration in the option text. The
panel formats no number and reads no claim-bearing field. `npx tsc --noEmit` clean.

**Claim boundary.** The contract checks a declaration for completeness and internal agreement. It
reads no data file, so it cannot know whether a domain's declarations describe the source it
names, nor whether the wording chosen means to a practitioner what it appears to mean. It
establishes nothing about which domain produced any given `EvidenceBundle`, because a bundle still
does not record one - `DOMAIN_ATTRIBUTION_CAVEAT` continues to travel with every served limit, and
the attribution gap recorded under TG9.3 is unchanged. R17's refusals remain enforced in the
analysis layer; this makes the declarations they read from complete, not self-enforcing. No Argo
data was fetched: the declaration is a declaration, and no adapter reads the archive.

## TG8.4 - the ingestion seam (`ed-dev`)

TG8.1 made a domain declarable from outside `src/`. This makes one readable. `src/api/channels.py`
mounts two routes over `src/data_layer/tabular_source.py`, which had read delimited channel
records since TG0.2 and was referenced **zero times** from `src/api/` and `frontend/src/`.

**The rule the slice is built on.** *Detection may create a required declaration; it may never
satisfy one.* Observation and decision are separate functions: `clock_facts` consults no domain
and refuses nothing, `required_violations` turns what it found into obligations, and only
`read_channels_for_domain` decides anything.

**The two calls, on the three shipped fixtures:**

```text
                          readable  required                  admitted by
order_book_regular.csv    yes       []                        order_book
order_book_irregular.csv  yes       ['irregular_sampling']    order_book
clock_runs_backwards.csv  no        -                         (none)
```

`reanalysis` refuses all three, and says why: it declares latitude and longitude, which a channel
table cannot supply (E14). That refusal is the slice working, not a gap.

**Acceptance met.**

*   **(a)** `order_book_regular.csv` loads under `order_book` and is refused under `reanalysis`
    by name, citing the declared axes.
*   **(b)** `order_book_irregular.csv` is refused under `tidy_venue` (onboarded by the test,
    lacking `irregular_sampling`) and loads under `order_book`, reporting
    `cadence_seconds: null` rather than a fabricated number.
*   **(c)** A channel given a footprint of 60 samples is refused unless the domain declares
    `aggregated_values`, and marked `is_aggregate` where it is allowed.
*   **(d)** For every fixture, every domain `inspect` reported as admitting returned 200 from
    `read` and every domain it reported as refusing returned 400. Checked against a **third**
    domain onboarded for the purpose: both built-ins agree about ragged clocks, so neither can
    show the case where inspection and reading could diverge.
*   **(e)** The same bytes yield the same `content_sha256` under two different filenames, and the
    provenance names the file, the clock column, the domain and the onboarding digest.
*   **(f)** Both routes are reachable from tab 12; `test_no_served_route_is_unreachable_from_the_ui`
    passes with no new exemption.

**Two things found while building, recorded rather than smoothed over.**

*A price column was promoted to be the clock.* Given `clock_runs_backwards.csv`, whose `t` jumps
backwards but whose `bid` increases monotonically, the first implementation picked `bid` as the
clock and reported the file as readable. Every lag downstream would have described that
substitution and nothing would have said so. Inspection now stops when the first column cannot
serve, names the candidates, and requires the caller to choose. The test that caught it was
written expecting a refusal and got `readable: True` — the code was changed, not the test.

*D61: a declaration that contradicted its own prose.* `order_book` has described "an irregular
trading clock" since its first commit while omitting `irregular_sampling` from its violation
tuple. Four slices passed without it being noticed because nothing had yet tried to **read data**
under a declaration. The knock-on is recorded rather than hidden: half of what TG8.1 credited to
Argo was really this gap, so `domain_plugin_example.py` and the coverage test now claim
`non_stationary_support` alone, and both say why.

**Evidence.** 24 new tests in `src/tests/test_channels_api.py`, 21 added to
`test_tabular_domain.py` (27 -> 48), 8 added to `test_frontend_contract.py` (40 -> 48). Five
deliberate mutations, each caught:

```text
M1  drop the declared-axes check            -> 3 tests failed
M2  drop the aggregate-declaration check    -> 1 test failed
M3  stop reporting the clock obligation     -> 3 tests failed (2 before the cross-check
                                               was strengthened; see below)
M4  remove the size cap                     -> 2 tests failed
M5  rebuild the declaration instead of
    returning the onboarded one             -> 1 test failed
```

M3 initially failed only the two tests that read `required_violations` directly, and **not** the
acceptance-(d) cross-check that exists to catch exactly that class of change. The reason was
vacuity: with only `reanalysis` and `order_book` registered, every domain that admits an irregular
record declares `irregular_sampling` and every domain that refuses one refuses it earlier on its
axes, so the check agreed with itself no matter what. `tidy_venue` was added to the test and M3
was re-run; it then failed the cross-check as intended. Recorded because a mutation that a guard
*should* have caught and did not is evidence about the guard.

**Claim boundary.** Reading a file under a domain establishes that the domain's declaration admits
the file's shape - not that the file came from that domain, and no check here could establish
that. `DOMAIN_ATTRIBUTION_CAVEAT` is served with every inspection and every read, and the view
renders that sentence rather than restating it. **No public dataset has been ingested**: the
adapter reads local files and never fetches, and the three fixtures are seeded fabrications with a
README saying so. The preview plot is a preview - nothing is mined, no claim exists, no rung moves
(R22). Refusing a file is not validating the data in it: finite, monotonic and regularly sampled
says nothing about whether the values are right. Rendered browser inspection of tab 12 is
**NOT RUN**, as it is for tabs 10 and 11.

---

## TG10.1 - The store catalogue becomes a registry (2026-08-27, `ed-dev`)

**What was verified.** `GRIDDED_STORES` is a `Registry[GriddedStore]`; the four ERA5 stores
register eagerly and idempotently; every registered store names a domain something has actually
declared; and `zarr_source.CATALOGUE` survives as a read-only mapping view that keeps every key
the old dictionary carried. `CropSpec` gained one declared field, `vertical_dim`, and `select()`
applies the vertical selection to that name instead of a hard-coded `"level"`.

**Acceptance criterion, executed literally.** `src/tests/store_plugin_example.py` registers a
fifth store on a `depth` axis in a file no core module imports. The test asserts it reaches
`GET /api/v1/data/zarr/catalogue`, and hashes `src/data_layer/zarr_source.py` and
`src/api/main.py` before and after to assert both are byte-identical - the method
`test_registries.py` uses for the data-source seam. The example file sits in `src/tests/` rather
than literally outside `src/`, alongside `plugin_example.py` and `domain_plugin_example.py`; what
is checked is the substance of the criterion, that no core file was edited.

**Results.**

```
Full suite            2459 passed, 1 skipped, 1 xfailed
test_stores.py        39 passed (29 test functions)
adjacent suites       test_zarr_source, test_registries, test_evaluation_run,
                      test_cds_source, test_evaluation_report - 122 passed, 1 skipped
Documentation audit   19 passed
```

**Six deliberate mutations, each caught.**

```
M1  remove the domain check from register_store   -> 1 test failed
M2  accept an undated live inspection             -> 1 test failed
M3  let an unmeasured store quote a chunk size    -> 1 test failed
M4  hard-code the level axis back into select()   -> 2 tests failed
M5  put vertical_dim in the content key always    -> 2 tests failed
M6  make the catalogue view writable              -> 1 test failed
```

**A silent failure found while building.** A store on a `depth` axis, read through the ERA5 path,
selected **no vertical subset at all** and reported nothing wrong: `"level" in subset.coords` was
simply false, so the vertical selection was skipped. The full depth axis flowed into the cache
while the manifest recorded the request rather than what arrived. That is now
`test_a_depth_store_read_as_though_it_were_era5_selects_no_vertical_subset`, which asserts the old
behaviour explicitly so the reason for the change stays legible.

**The content key was pinned, not recomputed.** `09d0e1b7cacc25b0` was obtained by running
`git show HEAD:src/data_layer/zarr_source.py` against the spec and reading its answer, not by
writing down what the new code produced - a pin copied from the code it guards guards nothing. It
is unchanged because `vertical_dim` enters the canonical form only when it is not `level`, which
is a compatibility decision taken deliberately: adding it unconditionally would orphan every
materialised crop in the local cache and make every recorded provenance record name a key that no
longer resolves.

**Claim boundary.** **No store was opened and no live fetch was run.** Registering a store is a
declaration; nothing in this slice reaches the network or verifies that a recorded chunk figure is
still true, and TG10.3 is the slice that makes probing a recorded act and a precondition of
registration. The recorded 51.1x and 26.2x amplifications are transcriptions of earlier live
inspections (2026-08-20 and 2026-08-21), now carrying their method and date as fields rather than
as prose; they were not re-measured here. The generalisation reaches the **selection** path only -
the cached-crop reader still speaks in pressure levels and `level_hpa`, which is honest for the
four ERA5 stores that exist and is the remaining half of the job when a real depth-axis store
arrives. The fifth store in the acceptance test is a fixture URI that has never been opened, and
says so through `method="not measured"`. Rendered browser inspection of the catalogue tab after
this change is **NOT RUN**.

---

## TG10.3 - Store probing as a recorded act (2026-08-27, `ed-dev`)

Taken **before** TG10.2 rather than after, because the acquisition surface renders what the
probe produces and building it first would have meant rendering transcribed prose and revising
it a week later.

**Acceptance criterion, in two halves.**

*   **"A deliberately hostile store is characterised as hostile before anyone crops it"** -
    **MET, offline.** The load-bearing word is *before*, so the test asserts it: a hostile
    fixture store is characterised as hostile with **nothing materialised**, no cache entry
    created, and an amplification agreeing to within 1e-9 with what `assess_access_pattern`
    predicts from chunk metadata alone.
*   **"The four ERA5 stores' recorded notes are reproduced by the probe"** - **NOT MET, and
    recorded as such.** The four notes are held as **transcriptions** (`evidence="prior
    recorded inspection"`), not as probe runs. They are counted by `transcribed_probes()` and
    the count is served by `GET /api/v1/data/zarr/probes`, so the debt is published rather
    than hidden and the number can only fall in the open. The opt-in test that would check a
    transcription against the live WeatherBench store is written and **NOT RUN**.

**Results.**

```
Full suite            2503 passed, 2 skipped, 1 xfailed
test_store_probe.py   43 passed, 1 skipped (34 test functions; the skip is the live probe)
test_stores.py        31 passed
adjacent suites       test_frontend_contract, test_zarr_source - 189 passed, 2 skipped
Documentation audit   19 passed
tsc --noEmit          clean
npm run build         succeeds
```

The two skips are the two opt-in live checks: the WeatherBench GCS read from T3.5.18, and this
slice's live probe. Both require `SPECTRALEARTH_ALLOW_NETWORK=1` and neither has been run.

**Eight deliberate mutations. Six caught on the first pass, two survived.**

```
M1  drop the probe requirement from registration  -> SURVIVED, then 1 test failed
M2  let a probe of any URI license any store      -> 1 test failed
M3  let an entry quote a figure the probe denies  -> 1 test failed
M4  raise on network-off instead of recording it  -> 2 tests failed
M5  fold unknown amplification into "not hostile" -> 2 tests failed
M6  drop the reason from an open failure          -> 6 tests failed
M7  report the mean chunk rather than the worst   -> SURVIVED, then 1 test failed
M8  let save_probe overwrite an existing record   -> 1 test failed
```

**Both survivors were weak tests, not weak guards, and the tests were fixed rather than the
code.** M7 survived because the two-variable fixture it used carries two `float32` variables of
identical shape, so the maximum and the mean are the same number and the assertion was
decorative; it now probes a store whose variables differ in dtype, and asserts the two figures
differ before comparing. M1 survived because the *next* check also raised - deleting "you must
cite a probe" left "that digest is not in the ledger", which is true, unhelpful, and would send
an author looking in the wrong place; the test now asserts the message names the claim being
made, the remedy, and the honest alternative.

**Two defects found and fixed, both introduced by TG10.1.**

*   **D62** - `era5_0p7_6h` carried a per-chunk size of **8.0 MB that no inspection produced**,
    in the registry built to refuse figures nobody measured. The 2026-08-21 note records an
    amplification and an estimated total and no per-chunk size, and 8.0 does not even follow
    from the chunk shape the note describes, which works out at 54.5 MB. It passed because
    `ChunkFacts` *demanded* a positive figure for any method other than `not measured`, so
    filling the field was the only way to record a real inspection - a validation rule that
    made the dishonest entry the easy one. **Not recorded** is now a third state distinct from
    **not measured**.
*   **D63** - adding `vertical_dim` to `CropSpec` changed `to_provenance`, which is embedded in
    **authenticated** artefacts: a gate campaign's preregistration is fingerprinted over it, so
    a checked-in signed record written before the field existed stopped loading.
    `to_provenance` now omits the field when it is `level`, as `canonical()` does, so an ERA5
    record is byte-identical to its pre-TG10.1 form.

**How D63 reached a commit, recorded because the process failure matters more than the bug.**
TG10.1 was reported complete on the strength of targeted suites while the full run was still in
progress, and the full run is what found it. The targeted suites were the wrong evidence for
the claim that was made, and the honest statement at that point would have been "targeted
suites pass, full suite still running".

**Claim boundary.** **No live probe of a public archive has been run**, and
`data/store_probes/` holds nothing produced by one. Every figure in the catalogue is still a
transcription of an inspection this code did not run. A probe records structure and *predicted*
cost from chunk metadata; it transfers no data, and it does not validate the data in a store. A
store characterised as friendly is friendly **for the crop that was stated** - the same fixture
amplifies 1x for a request that lines up with its chunks and 30x for one that straddles them,
which is why an amplification is refused unless the crop travels with it. The registration gate
proves that an entry cites a look; it does not prove the look was recent, or that the archive
has not rechunked since. Rendered browser inspection of the probe panel is **NOT RUN**.

## TG10.2 - Domain-first acquisition surface (2026-08-27, `ed-dev`)

**Implemented.** `src/api/acquisitions.py` serves `GET /api/v1/acquisitions` as a projection of
domain declarations, the gridded-store registry and the existing channel-table admission rule.
Every acquisition carries its domain limits and attribution caveat. The React navigation now
has one Acquire tab: it maps returned domains and acquisitions, embeds `ChannelRecords` for an
admitted table domain, and retains ERA5 catalogue, crop, probe, inspection, cached-readiness
and materialisation-command capability for grid crops. The standalone Domain Records tab was
removed, so registering a domain cannot grow the tab list.

**Evidence.** The acquisition, store, channel and frontend contract suites pass, and
`npm run build` completes with 1,389 modules transformed and real JS/CSS assets. The complete
suite for the TG10.2 implementation reports:

```text
2509 passed, 2 skipped, 1 xfailed, 6 warnings in 1357.52s (0:22:37)
```

The two skips remain the opt-in live GCS read and live store probe. The expected failure is the
repository's declared xfail, not a TG10.2 regression. After recording that receipt, the final
documentation-count correction and domain-switch remount guard pass their focused tests and
`tsc --noEmit` remains clean.

**Claim boundary.** No public data was fetched and no live archive probe was run. The endpoint
publishes capability and refusals; it does not establish source attribution or executed
acquisition. Rendered browser inspection is **NOT RUN**.

## TG11.0 - Workflow information architecture (2026-08-27, `ed-dev`)

**Implemented.** The eleven flat numbered destinations are grouped under Acquire, Analyse,
Evidence, Review, Read and Platform. Spatial-only destinations carry the **Gridded field line**
marker; Automated Hypotheses does not. Review is a labelled TG11.5 waypoint rather than a link
to an invented surface. Acquire is the default workflow entry.

The shell now owns the selected channel record and study. The channel selection retains the
original browser `File`, clock choice and aggregate supports beside the API's bounded preview,
so a later analysis panel can submit the admitted full input without a second chooser or using
truncated preview values. Both contexts remain visible across panels, restore their respective
Acquire/Findings state, and can be explicitly cleared.

**Evidence.** The exact final tree reports:

```text
frontend + documentation: 71 passed, 5 warnings in 117.43s (0:01:57)
frontend production build: 1,389 modules transformed; JS/CSS assets emitted
complete suite: 2511 passed, 2 skipped, 1 xfailed, 6 warnings in 1292.03s (0:21:32)
```

The skips remain the opt-in live GCS read and live store probe; the xfail is the repository's
declared expected failure. Vite's existing large-chunk warning remains non-fatal. Rendered
browser inspection is **NOT RUN**.

**Claim boundary.** This is workflow state and labelling, not scientific validation. It does
not make gridded tools domain-general, persist an evidence bundle, run analysis, write evidence,
or move a claim rung. A retained browser `File` disappears when the application session ends.

## TG11.1 - The analysis surface (2026-08-27, `ed-dev`)

**Implemented.** `src/api/analysis.py` mounts the cross-domain analysis engine on
`/api/v1/analysis`, which had no HTTP boundary at all: `association_only`,
`analyse_precedence` and `run_domain_gate` were unreachable from a browser. Two endpoints -
capabilities and one stateless `POST /run` operation selector - and no new science; every
estimator, correction, lag floor and verdict stays in `analysis_engine/domain_analysis.py`.

The route re-reads the original uploaded record rather than `/channels/read`'s bounded preview,
so the analysis is of the admitted file rather than of a truncation. Cadence, frames, channels,
measure and source identity are derived server-side; the client supplies only the hypothesis
family and estimator settings. Four refusals fire before any computation: an unknown
configuration key, an irregular clock, a non-UTF-8 upload, and R21 propagated from the engine
word for word. Every success carries `read_only: true`, `stored: false` and `rung_moved: false`.

The workbench panel checks that the returned `content_sha256` and frame count are the selected
record's before reading the result as being about it - an unchecked re-read is only a belief
that the server read the same file. `INVALID` is named on screen as a design that did not hold
rather than styled as a leftover beside `FAIL`.

**Evidence.** The exact final tree reports:

```text
analysis + frontend contract + documentation: test_analysis_api.py 7 passed,
  test_frontend_contract.py 55 passed, test_documentation.py 19 passed
frontend production build: 1,390 modules transformed; JS/CSS assets emitted
complete suite: 2521 passed, 2 skipped, 1 xfailed, 6 warnings in 1046.84s (0:17:26)
```

The acceptance is `test_all_thirteen_sequence_and_cross_domain_benchmarks_pass_through_http`:
all thirteen `sequence` and `cross_domain` benchmarks run through the live FastAPI test client
with `failed == 0`, `not_yet_runnable == 0` and `null_failures == []`, eight of the thirteen
being nulls. `tsc --noEmit` is clean. Rendered browser inspection is **NOT RUN**.

**D64, found by mounting the router.** The documentation guard's route scan matched
`@router.<verb>("([^"]+)"`, so a route mounted at its own router's prefix - `@router.get("")` -
was invisible to it. `GET /api/v1/acquisitions` had been served and unseen since TG10.2, and the
documented count read 42 against a served 43 while the check passed. The quantifier is now
`[^"]*` and the served count is 45.

**Claim boundary.** Reachability is not correctness. The thirteen benchmarks argue for
correctness on the thirteen cases they cover and on nothing else. This slice records no
evidence, derives no rung, preregisters nothing and spends no held-out partition.

## TG11.2 - Preregistration first (2026-08-27, `ed-dev`)

**Implemented.** `src/api/preregistration.py` mounts `core/preregistration.py` - 574 lines that
nothing outside the tests could reach - on `/api/v1/preregistration`. Six endpoints: capabilities,
a partition description built from geometry and lineage without reading a measure value, sealing,
a seal listing, a seal read with both digest layers recomputed, and the one-shot confirmation. No
estimator, digest, correction or refusal is implemented here.

The ordering R18 requires is enforced by the server rather than by the panel. A confirmation
against a seal that does not exist is refused before the record is read; against a partition the
seal did not name, by `PartitionMismatchError`; against a spent partition, by the ledger. TG11.1's
`domain_gate` splits internally and returns a verdict on its own test partition, so it now reads
the ledger before running and is refused `409` on data already spent.

`/confirm` accepts the record and an optional published digest, and nothing else: lags, ensemble
size, alpha, correction, estimator, bins, seed, split, domain, clock and delimiter are all read
back out of the seal, where they were frozen as the confirmatory specification's `notes`. The
sealing time is the server's, because a caller-supplied one could be written after the partition
was opened.

**Evidence.** The exact final tree reports:

```text
preregistration + frontend contract + documentation: test_preregistration_api.py 17 passed,
  test_frontend_contract.py 60 passed, test_documentation.py 19 passed
frontend production build: 1,391 modules transformed; JS/CSS assets emitted
complete suite: 2543 passed, 2 skipped, 1 xfailed, 6 warnings in 981.29s (0:16:21)
```

The acceptance is the double-spend refusal in two forms. In the weaker form the second seal is
written after the first confirmation; in the stronger,
`test_two_seals_written_before_any_opening_still_buy_only_one_look` freezes both seals before
anything is opened, so neither is post-hoc in any sense - no edit, no backdating, no narrowing
chosen after a look - and the second confirmation is still refused, because two declarations over
one held-out partition are two tests of it whose corrections were each computed as though it were
the only one. `test_a_refused_confirmation_does_not_spend_the_partition` asserts the other half:
a design error costs nothing and the partition is still spendable afterwards. `tsc --noEmit` is
clean. Rendered browser inspection is **NOT RUN**.

**D65, found while writing the double-spend acceptance.** `PartitionIdentity.from_series` hashes
the series' provenance wholesale, and a record read from an upload carries `path_basename` there.
The same held-out bytes re-uploaded as `data2.csv` hashed to a different partition, so
`HeldOutLedger` - keyed on the partition precisely so a second honest seal cannot buy a second
look - did not fire. The domain the file was read under had the same problem for the same reason.
The acceptance passed against a same-name re-upload and would have passed for the wrong reason.
The identity is now built at the boundary from `content_sha256`, the clock, the columns and the
split window, and a test renames the file between two identical requests and asserts the digest
does not move.

**Claim boundary.** A seal is a promise about ordering, not a result: it says a family was fixed
before a partition was opened and nothing about whether the family is any good. A confirmation
receipt records no evidence and moves no rung (R22); writing one into a bundle is TG11.3. The
generated family is corrected for nowhere here and its members are not claims. `report_generation`
remains without a route and is named as such rather than covered by a thin one. Seals and the
ledger are programme state under `data/preregistrations`, not a cache - deleting them destroys the
record of what has been spent - and the JSON ledger takes no lock, so "once" is once per server
and a multi-worker deployment needs a real store.

## TG11.3 - The evidence write path (2026-08-27, `ed-dev`)

**Implemented.** `src/api/evidence.py` mounts `core/evidence.py` and `core/claim_ladder.py` on
`/api/v1/evidence`. Five endpoints: capabilities, opening a study at revision zero, reading the
head, appending one entry, and appending a provenance entry whose precedence verdict is computed
by the server. No digest, chain check or ladder gate is implemented here; this is the wire
boundary for two modules the interface previously could only read through TG9.1.

This is the first surface in the programme that writes anything bearing on a claim, so it is the
first that could break R22. Three things enforce the rule structurally rather than by review.
No request model has a field for a rung, a claim level or a confidence, and unknown fields are
forbidden, so a body carrying one is refused rather than ignored. The rung in every response is
`assess_claim_ladder` recomputed over the chain that was just written, and it is stored nowhere.
And `temporal_precedence` - the single payload key the ladder reads, and the gate for
`candidate_precursor` - is refused at any depth of a hand-written payload; the route that writes
it runs `analyse_precedence` here and records what it returns, `false` included.

Appends are compare-and-swap. Each states the `head_sha256` it extends, and because a bundle is
immutable and `save_evidence_bundle` refuses to overwrite, each revision is published as its own
file created exclusively - which makes the exclusive create the concurrency control. Two writers
racing from one head produce one append and one `409` rather than a lost entry, which is
deliberately stronger than TG11.2's unlocked ledger.

**Evidence.** The exact final tree reports:

```text
evidence + frontend contract + documentation: test_evidence_api.py 22 passed,
  test_frontend_contract.py 65 passed, test_documentation.py 19 passed
frontend production build: 1,392 modules transformed; JS/CSS assets emitted
complete suite: 2570 passed, 2 skipped, 1 xfailed, 6 warnings in 915.05s (0:15:15)
```

The acceptance is that the rung cannot be reached by typing. Two tests approach it from opposite
sides: `test_a_request_carrying_a_rung_is_refused_rather_than_ignored` and
`test_a_payload_asserting_temporal_precedence_is_refused_and_names_the_route`, the second of which
is the one that mattered, because that key needs no arithmetic and no estimator would have noticed
it. `test_the_rung_moves_because_the_evidence_moved_it` shows the ladder climbing to `association`
through three appends that named nothing, and
`test_one_failed_entry_caps_the_chain_at_observation_through_the_wire` shows a single `FAIL`
pulling it back down over favourable evidence already recorded. `tsc --noEmit` is clean. Rendered
browser inspection is **NOT RUN**.

**D66, found by asserting through the read surface what the write surface had just returned.**
`StudyStore.load` resolved a study by taking the first parseable file whose `study_id` matched,
and `summaries` listed one row per file. That was correct while nothing wrote bundles: TG9.1 read
a store a researcher populated by hand, one file per study. Revision-per-file makes it wrong -
sorted first is `r00000`, so the read surface would have served revision zero for ever while the
write path reported the revision it had just appended, and one study worked on five times would
have listed as five studies. Resolution is now by chain rather than by name. The check that found
it is deliberately cross-surface: two consistent halves of one store can agree with each other and
both be wrong.

**Claim boundary.** Recording evidence is not establishing a finding. The ladder grades what is in
the chain, and a chain of one favourable observation earns `observation`. An underpowered sweep is
recorded `INCONCLUSIVE` and opens no gate, because a family that could not have rejected anything
did not check anything (R5). A bundle carries no domain, so nothing written here records which
instrument the evidence came from. This surface does not consult the held-out ledger: running a
precedence analysis over data preregistered as held out spends it outside the record, which the
ledger cannot see and the capabilities note cannot prevent. Bundles are programme state under
`data/studies`, not a cache.

## TG11.4 - Structure mining (2026-08-27, `ed-dev`)

**Implemented.** `src/api/mining.py` mounts `core/motif.py`, `core/constellation.py`,
`core/family.py`, `core/invariance.py`, `core/motif_freeze.py` and `core/motif_transfer.py` on
`/api/v1/mining`. Eleven endpoints: capabilities, admitting a field, listing what is admitted,
pricing a family before mining it, calibrating a match tolerance, mining the training frames,
freezing the confirmatory family, opening the held-out frames once, publishing a durable motif
definition, transferring it into a second domain, and auditing every registered matcher against
its own declared invariance. No matcher, null, correction, tolerance or p-value is implemented at
the boundary; this is the wire boundary for about four thousand lines that nothing outside the
test suite could previously call.

Two things cannot be typed on this surface, and both are structural rather than reviewed. A motif
is a configuration of extracted features, so no request model has a field for a feature, a
coordinate or a graph: a caller admits a `.npy` stack of frames and the server extracts, under
settings that become part of the record's digest. And the match tolerance - which decides what
counts as the same shape - travels as the digest of a calibration the server performed, because
one built the tempting way came out at 0.41 against a correct 0.0083 in this tree's own
benchmark, wide enough that every triangle matched every other.

The confirmatory run is driven from the seal. `/confirm` takes a seal digest and an optional
published one; the record, the split, the size, the matcher, the tolerance, the ensemble, the
correction and the seed are sealed as notes on the confirmatory specification, and the training
candidates are re-derived by re-running the deterministic mining pass and checked label for label
against what the seal froze. One additive core change made that possible:
`motif.confirmatory_specification` and `motif.freeze_motifs` take an optional `notes` mapping,
merged beside the notes they already write (E12). Mining seals are stored in TG11.2's seal store
and spend TG11.2's held-out ledger.

**Evidence.** The exact final tree reports:

```text
mining + frontend contract + documentation: test_mining_api.py 41 passed,
  test_frontend_contract.py 73 passed, test_documentation.py 19 passed
frontend production build: 1,393 modules transformed; JS/CSS assets emitted
complete suite: 2619 passed, 2 skipped, 1 xfailed, 7 warnings in 4090.28s (1:08:10)
```

The acceptance is the pair of gates the mining phase was built against.
`test_a_null_record_confirms_nothing` runs the whole chain - admit, calibrate, price, mine,
freeze, open - over frames with nothing planted in them, using the same generator, the same
feature count, the same family and the same ensemble as the planted record, and confirms nothing.
`test_the_planted_motif_is_confirmed_on_frames_it_was_not_mined_from` shows the same pass finding
what is there, at a corrected q below alpha on frames it was not mined from. Beside them,
`test_no_route_on_this_surface_accepts_a_feature` and `test_the_tolerance_cannot_be_typed` are the
structural pair, and
`test_frames_that_disagree_about_how_many_features_they_hold_are_refused_not_trimmed` asserts that
the refusal names the repair it is declining. `tsc --noEmit` is clean. Rendered browser inspection
is **NOT RUN**.

**One ordering mistake, found by asserting what a refusal costs.** `/confirm` first verified the
published digest *after* running the confirmation, so a seal that disagreed with its published
digest was refused - having already spent the held-out partition on a result nobody was then
allowed to use. The check now runs before anything is opened, and the test asserts the
consequence rather than the status code: after a refused confirmation, a correct one still
succeeds. A refusal that costs the data it refused is worse than no refusal, because it looks
like a guard.

**One implementation note worth recording, because it looks like a fudge and is not.** A transfer
record is refused unless `frozen_at < bound_at < opened_at`, and that ordering is the entire
content of the record. A Windows clock ticks about every 15 ms, so two events that really did
happen in that order can be issued one timestamp, and a correctly ordered transfer is then refused
for a reason about the clock rather than about the science. `_now` advances an instant that would
repeat or go backwards by one microsecond. Nothing waits and nothing is back-dated: the ordering
reported is the ordering that happened, at a resolution the clock does not have.

**Claim boundary.** Mining produces candidates. Support on the training frames is selection and
not evidence: every exemplar is one of the occurrences it is counted among, so its support starts
at one by construction, and it was ranked highly for having been counted often. A confirmed motif
is a configuration that recurred on frames it was not mined from more often than the surrogate
null placed it there - not a mechanism, not a cause, and not a claim until something records it,
which is TG11.3's write path (R22). A motif reported `vacuous` was confirmed by an ensemble that
could not have rejected it (R5). A transfer match count is descriptive rather than a corrected
transfer result, and the transfer ledger establishes ordering inside this API only - it cannot
show that nobody looked at the target before it was admitted. `core/cross_domain.py` is **not**
routed by this slice and is carried as TG11.4b: its input is two channel tables on a common
clock, not scenes, and it belongs beside the analysis surface. Admitted fields, calibrated
tolerances, published definitions and the transfer ledger are programme state under `data/mining`,
not a cache.

## TG11.4b - The cross-domain record (2026-08-27, `ed-dev`)

**Implemented.** `src/api/cross_domain.py` mounts `core/cross_domain.py` on
`/api/v1/cross-domain`. Seven endpoints: capabilities, aligning two native clocks by exact
timestamp intersection, pricing a lag family declared in seconds, splitting the aligned record
with an embargo, sweeping every crossing direction on the training partition, freezing the
confirmatory family, and opening the held-out partition once. This is the wire boundary for the
505 lines that made a lag a duration rather than a frame count, and only the benchmarks could
previously reach them.

Three refusals are structural rather than reviewed. Nothing is resampled: two clocks are
intersected exactly, and a pair that shares too few observations is refused with the refusal
naming interpolation as what it declines - every response reports what each domain retained and
discarded. Nothing is defaulted: each column must declare its `semantics` and `units` before the
record can be read (R19), and both are carried into every confirmed relationship in the receipt
alongside the lead in seconds. And nothing can be tuned at confirmation: `/confirm` takes the two
records and, optionally, a published digest, and reads everything else back out of the seal.

One additive core change made the last of those possible: `precedence.confirmatory_specification`
and `precedence.freeze_precedence` take an optional `notes` mapping, merged beside the notes they
already write (E12), so the run settings sit inside the specification's fingerprint rather than
in a file beside the seal. Seals are stored in TG11.2's seal store and spend TG11.2's held-out
ledger.

**Evidence.** The exact final tree reports:

```text
cross-domain + frontend contract + documentation: test_cross_domain_api.py 35 passed,
  test_frontend_contract.py 80 passed, test_documentation.py 19 passed
frontend production build: 1,394 modules transformed; JS/CSS assets emitted
complete suite: 2661 passed, 2 skipped, 1 xfailed, 6 warnings in 908.44s (0:15:08)
```

The acceptance is the planted/null pair the cross-domain phase was built against.
`test_the_planted_relationship_is_confirmed_on_data_it_was_not_selected_from` runs the whole chain
over an hourly domain and a three-hourly one carrying one delayed relationship across the
boundary, and confirms exactly that relationship at exactly the planted duration on frames it was
not selected from. `test_the_same_pipeline_over_an_uncoupled_pair_confirms_nothing` is the same
builder with the coupling knob at zero and confirms nothing. Beside them,
`test_a_column_whose_meaning_was_not_declared_is_refused_not_defaulted`,
`test_clocks_that_share_no_observation_are_refused_rather_than_resampled` and
`test_no_route_on_this_surface_accepts_a_lag_in_frames` are the structural three.
`tsc --noEmit` is clean. Rendered browser inspection is **NOT RUN**.

**One note on what a test could not reach, recorded because the alternative is overstating it.**
`_candidates_from` refuses a frozen member the training partition does not select. No caller can
reach that refusal: an edited seal fails its own digest at load, and a different pair of records
produces a different held-out partition identity, which `confirm_on_held_out` refuses first. Both
of those happen before the ledger is written, so a wrong upload costs nothing either way - which
is what the tests assert, rather than asserting a refusal message that only the module's own
future drift could produce.

**Claim boundary.** A cross-domain result is a temporal association between structural series.
The two records' raw magnitudes keep different semantics and units and are never compared, the
statistic is dimensionless, and precedence identifies no causal mechanism (R19, R21). A
confirmation receipt records no evidence and moves no rung (R22); it is an input to TG11.3's
write path. The held-out ledger identifies a partition by the data and its split, so the same two
files aligned under two different registered domains are two partitions to it and this surface
cannot detect that they hold the same rows (D65). Seals and the ledger are programme state under
`data/preregistrations`, not a cache. After the complete run `data/` holds only `README.md`,
`channels` and `store_probes`: this surface creates no directory as a side effect of being read.

## TG11.6 - Workflow accessibility contract (2026-08-28, `ed-dev`)

**Implemented.** Accessibility is no longer confined to the findings and channel-record islands.
`App.tsx` now gives the workflow a skip link, a named main landmark and focus transfer to a
programmatic workspace heading after navigation. The disconnected-backend retry is a native
button rather than a clickable `span`; every visible label in the legacy gridded panels is bound
to its control; shell errors are alerts; and shell plus acquire/analyse/evidence/read surfaces
publish asynchronous busy state.

`index.css` supplies a three-pixel high-contrast `:focus-visible` outline after Tailwind, so the
older `focus:outline-none` classes cannot make keyboard focus disappear. Reduced-motion preference
collapses animation and transition. Heat maps and line charts are labelled figures with text
summaries of their carried shape, units, axes, series, scale and valid inset. SVG provenance nodes
are named pressed-state controls with Enter and Space activation. Field import, evaluation receipt
selection and both sides of the cross-domain record have programmatically associated names; the
two cross-domain operands are fieldsets rather than one label visually covering eight controls.

**Evidence.** The exact final tree reports:

```text
frontend contract: 86 passed, 5 warnings in 50.69s (six TG11.6 tests added)
frontend production build: 1,394 modules transformed; JS/CSS assets emitted; tsc clean
documentation + frontend contract: 105 passed, 5 warnings in 152.16s
complete suite: 2667 passed, 2 skipped, 1 xfailed, 6 warnings in 1233.21s (0:20:33)
```

The six structural tests guard skip and route-focus behaviour, explicit legacy label bindings,
global focus visibility and reduced motion, keyboard retry/lineage operation, figure text
equivalents, and busy/alert/status semantics across every workflow surface. `git diff --check`
passes. No API route, request shape, estimator, evidence category or claim-ladder input changed.

**The rendered check was attempted and is not silently promoted.** The configured in-app browser
runtime was initialised against `http://127.0.0.1:3000/`; discovery returned no available browser
backend, including after the documented recovery check. No unrelated automation surface was
substituted. Rendered keyboard traversal and screen-reader inspection therefore remain **NOT
RUN**, even though TypeScript, Vite and the source contracts pass.

**Claim boundary.** This slice establishes source semantics, keyboard activation paths and build
integrity. It does not establish a WCAG conformance level, screen-reader quality, contrast in the
rendered Plotly output or visual focus placement. A figure summary restates metadata carried by
the application and does not interpret the scientific result.

## TG11.5 - Recorded review surface (2026-08-28, `ed-dev`)

**Implemented.** `src/api/reviews.py` adds one read-only endpoint,
`GET /api/v1/reviews/studies/{study_id}`. It resolves the latest published evidence bundle, reads
the dedicated review artifact directory, classifies JSON by declared schema, and reconstructs
`ReviewRecord`, `RoundRobinOutcome` and `ReviewCostReceipt` through their core types so their
content digests are rechecked. A record attaches only to the exact study, bundle digest and bundle
revision it reviewed; outcomes and receipts additionally bind the review-record digest. Malformed,
tampered and unknown artifacts are reported rather than skipped.

`ReviewView.tsx` fills the Review waypoint introduced by TG11.0. It is separate from Findings,
leads with the server's recorded-not-reproducible declaration and claim boundary, renders the
core's complete call and round-robin text including retained dissent, and shows cost receipts as
route/token audit without a price or quality claim. It can read a selected study and do nothing
else: no control and no client method can run a panel, record a call, append evidence or move a
rung. Empty record, outcome and receipt states each say what absence does not establish.

**Focused evidence.** The new API tests and expanded frontend contracts report:

```text
test_reviews_api.py: 8 passed, 2 warnings in 4.42s
test_frontend_contract.py + test_reviews_api.py: 100 passed, 5 warnings in 7.54s
frontend production build: 1,395 modules transformed; JS/CSS assets emitted; tsc clean
```

The eight API tests cover explicit absence, the complete verified record/outcome/receipt payload,
exact latest-revision binding, record-digest attachment, corrupt-artifact reporting, unknown-study
404, GET-only routing and a read that leaves the evidence bundle byte-identical. Six frontend
tests cover the separate workspace, GET-only service method, visible R23 fence, complete argument
and cost display, non-reassuring empty states and the workflow accessibility contract.

**Rendered inspection is NOT RUN.** The configured in-app browser runtime was initialised against
`http://127.0.0.1:3000/`; discovery returned no available browser backend after the documented
recovery check. No unrelated browser-control surface was substituted. TypeScript, Vite and source
contracts are not evidence of rendered layout or keyboard traversal.

**Claim boundary.** Every displayed call and outcome is R23 recorded-not-reproducible argument
beside the evidence. It is not evidence, consensus, reproducible computation or permission to
claim, and deleting it changes no claim level (R22, R23). The cost receipt establishes only the
recorded route and token accounting; it says nothing about whether an argument is sound.

---

## TG11.5 addendum - rendered inspection, now RUN (2026-08-28, `ed-dev`)

The TG11.5 entry above records rendered inspection as **NOT RUN**, because no browser backend was
available at the time. It has since been run. That claim is superseded here rather than edited, so
both the original limit and its lifting stay legible.

**Method.** Playwright/Chromium driving a Vite dev server on port 3002 against a backend on 8001,
with `SPECTRAL_STUDY_ROOT` and `SPECTRAL_REVIEW_ROOT` pointed at a scratch store seeded from the
`_recorded_review()` fixture that `test_reviews_api.py` uses. Study `tg7_3_cost`. The developer's
own 3000/8000 pair was left untouched, and nothing was written into the repository's `data/`.

**Observed in the rendered page:**

*   Empty state renders *"No study selected. That is not the same as no review existing."*
*   The amber **Recorded, not reproducible** banner carries the R23 declaration and the claim
    boundary together, bound to `tg7_3_cost`, bundle revision 0, with the full bundle digest.
*   All eight recorded calls render with seat, model, effort, request id, timestamp, dissent flag
    and finding.
*   The round-robin outcome renders both honesty notes verbatim: that one model answered all eight
    seats, and that the reassessment was made by the model that wrote the candidate synthesis.
*   Route and token receipts render calls, input/cached/output/total tokens and cache-hit fraction
    under *"Token and route audit only; no price or review-quality claim."* No price string appears
    anywhere in the rendered output, matching the assertion in `test_reviews_api.py`.

**Zero console errors, zero page errors, no failing API responses** across the session.

**Claim boundary.** This establishes that the panel renders what the API vouched for, and nothing
about whether the recorded argument is sound. Keyboard traversal was **not** measured here; the
TG11.6 accessibility contract remains asserted by test rather than by rendered inspection.

---

## Launcher `.env.local` loading (2026-08-28, `ed-dev`)

Shipped in commit `b621f8d` with no recorded verification. Recorded here after the fact.

**The defect.** `SPECTRALEARTH_ALLOW_NETWORK` is read by `os.getenv` at
`zarr_source.py:network_enabled`. Nothing loaded `.env.local`: `grep -rn "dotenv|load_dotenv" src/`
returns no hits, and `python-dotenv` is neither in `requirements.txt` nor installed. Vite does not
load it either - it reads env files from `frontend/` and exposes only `VITE_`-prefixed names to
browser code. A flag set in `.env.local` was therefore inert, and the Acquire tab's "Network is off"
banner was the visible symptom.

**The fix.** `start_platform.ps1` promotes `.env.local` into its own process environment before
launching, which the uvicorn `Start-Job` and `npm run dev` inherit as child processes. No new
dependency, and the backend's config does not become cwd-dependent - which is the shape of defect
D7 that this script already exists to fix.

**Verified against a deliberately awkward fixture, in a clean `-NoProfile` shell:**

```
[*] Loading local environment from .env.local ...
    Ignoring unparseable line: junkline-with-no-equals
    Ignoring unparseable line: =leading-equals-is-junk
    Loaded: GEMINI_API_KEY, SPECTRALEARTH_ALLOW_NETWORK, QUOTED_DOUBLE, QUOTED_SINGLE, SPACED_KEY
    Already set in this shell, file ignored for: ALREADY_SET
--- resulting values ---
QUOTED_DOUBLE = [quoted value]      QUOTED_SINGLE = [single value]
SPACED_KEY    = [spaced value]      ALREADY_SET   = [from-shell-should-win]
--- child process inheritance (Start-Job) ---
child sees NETWORK=[1] KEY_SET=[True]
```

End to end against the real file: `zarr network_enabled() -> True`, where it was `False` before.
`Parser::ParseFile` reports no syntax errors. **Only variable names are printed, never values.**

**Claim boundary.** Enabling network access is now a property of a gitignored file rather than of
the command typed. That is what was asked for and it is visible on every launch, but it does mean
the refusal message that guarded the opt-in no longer appears on this machine.

---

## TG12.1 - GLORYS gridded ocean source (2026-08-28, `ed-dev`) - **COMPLETE**

Discovery, implementation and verification. One GLORYS layout is registered from an extension
module, both live metadata looks are persisted, and the cost estimator now completes on the real
12227 x 50 x 2041 x 4320 archive without constructing a dask data graph.

**Candidate triage, by probe rather than by reputation, as TG12.1 requires:**

| Candidate | What it actually is | Probe outcome |
|---|---|---|
| NOAA OISST | bucket reachable; `data/v2.1/...` is **per-day netCDF** | `probe_store` opens Zarr; cannot open it |
| ECCO | netCDF granules behind Earthdata | same, plus credentials |
| **GLORYS** | **Zarr** - Copernicus Marine ARCO on CloudFerro | `described`, **anonymously** |

**A correction to an earlier claim made in session:** GLORYS does **not** require credentials. An
initial `403` came from a guessed bucket path, not a real one. The real ARCO stores return
`HTTP 200` and probe anonymously. Copernicus credentials were needed only to *resolve* the URIs
through the toolbox, so the store registers as `access="anonymous"`.

**The two ARCO layouts, on an identical 1993-01-01 through 1995-12-31, 50 S to 30 S,
160 E to 180 E, `thetao`, surface-elevation crop:**

| Store | Chunking `[t, z, lat, lon]` | Amplification | Uncompressed fetched / wanted |
|---|---|---|---|
| ERA5 `0p7_6h` (**D43**, earlier sealed crop) | `(8,13,512,256)` | 26.2x | 29.88 / 1.14 GB |
| GLORYS `timeChunked.zarr` | `[1,1,512,2048]` | **72.52x** | 36.74 / 0.507 GB |
| GLORYS `geoChunked.zarr` | `[2081,1,16,16]` | **2.02x** | **1.023 / 0.507 GB** |

**This bears on D43: a laptop-feasible long regional ocean record does exist.** But only in one of
the two layouts. The asset names describe which dimension is chunked narrowly, not which query
they make cheap: `geoChunked.zarr` is time-deep and costs 2.02x for many times over a small region,
while `timeChunked.zarr` costs 72.52x. Taking the wrong asset registers the hostile store and
records a false negative against D43.

**Persisted probe digests:** `timeChunked` = `f9a45764fcf52c2a`, `geoChunked` =
`ccdb0625e7e8fb1d`. The earlier exploratory digests were produced before D67's live path was
repaired and were never persisted; these records were produced by `probe_store` itself after the
fix and contain the exact crop, variable structure and coordinate declaration.

**Store facts, measured:** dims `time=12227, latitude=2041, longitude=4320, elevation=50`; coverage
1993-01-01 to 2026-06-23, daily, at 1/12 degree. The current asset exposes 11 data variables;
the probe prices `thetao` and does not imply that every variable has the same storage encoding.

**The vertical axis is named `elevation`, not `depth`,** with negative-metre values
`-5727.917 .. -0.49402499198913574`. `KNOWN_VERTICAL_DIMENSIONS` now carries `elevation`.
Making that declaration executable found D68: the request path accepted integer levels only.
`CropSpec` and the strict HTTP model now preserve integer ERA5 values while admitting exact finite
floats, and the frontend loads the registered GLORYS crop defaults instead of carrying 850 hPa
across the store change.

**D67 live acceptance.** Before the fix, `assess_access_pattern` raised `MemoryError` in dask's
`slice_slices_and_integers`. After the fix, both real assets completed in 14 seconds in one
metadata-only command. The estimator applies the materialiser's xarray indexers to coordinate
arrays, maps selected labels to source positions, and counts exact chunk ids. The offline
regression replaces `select()` with a function that raises `MemoryError` and proves costing does
not call it.

**Registration.** `src/data_layer/glorys_store.py` loads the two checked-in records and registers
`glorys_phy_my_0p083deg_p1d` with `access="anonymous"`, `vertical_dim="elevation"`, the selected
probe digest, product/dataset/licence identity and the sealed acquisition defaults. Importing the
module performs no network access. An AST test scans every runtime module and finds no import of
`copernicusmarine`.

**Verification.** Focused backend and contract run: **257 passed, 2 skipped** before the
documentation inventory was updated; its only two failures were the expected missing-module and
stale-count guards. After reconciliation those guards pass. Frontend production build:
`tsc` clean, Vite **1,395 modules transformed**, emitted JS/CSS. Full-suite result is recorded in
the architecture execution ledger after the final run.

**Claim boundary.** Metadata only: no ocean data has been transferred, cropped or analysed. A
measured chunk layout is not evidence that a crop is scientifically useful. GLORYS breaks no
analysis assumption - per R17 it is a source, not a second domain, and must not be presented as
evidence that the abstraction generalises.

---

## TG12.1a - D70, readiness that declares what it is about (2026-08-28, `ed-dev`) - **COMPLETE**

Found by use rather than by test: GLORYS was selected in Acquire, `Probe store` and `Inspect` both
completed correctly, and the researcher asked what to do next. The honest answer is nothing -
materialisation is CLI-only by design and TG12.1's claim boundary says no analysis consumes an
ocean crop - but two things at that point were wrong rather than absent.

**Reproduced before the fix**, against a manifest carrying GLORYS's own selection:

```
spec = {"variables": ["thetao"], "levels": [-0.49402499198913574], "vertical_dim": "elevation"}
assess_manifest_readiness(spec) ->
    structurally_eligible = False
    missing_variables     = ['t', 'q', 'u', 'v', 'z']
    required_level_hpa    = 850,  level_available = False
```

`int(-0.49402499198913574)` is `0`; the coercion raised nothing. The verdict therefore reported an
ocean crop as an atmospheric candidate that had fallen short.

**Fixed.** Levels compared as numbers with no coercion; `applicable` derived from the crop's own
declared vertical axis (absent means `level`, per D63), so no store name is hardcoded; the
not-applicable text states that the variable and level rows describe what T5.2 requires rather
than anything this crop failed to supply; the Acquire panel renders that instead of the verdict;
and `Inspect` now says before the transfer that materialising is where this store currently stops.

**Verified.**

*   `test_readiness_refuses_to_judge_a_crop_it_does_not_describe` asserts both halves: the GLORYS
    crop reports `applicable=False`, `vertical_dim='elevation'` and an uncoerced level comparison,
    **and** an ERA5 crop is unchanged in every field (`applicable=True`,
    `not_applicable_reason=None`, `structurally_eligible=True`, `level_available=True`).
*   `test_regional_forecast.py`, `test_stores.py` and `test_acquisitions_api.py`: 63 passed.
*   Frontend `tsc --noEmit` clean; production build emitted 1,395 modules and real JS/CSS,
    matching the module count recorded for TG12.1.

**Claim boundary.** This changes what the workbench *says* about a crop. It transfers no ocean
data, and it does not create an analysis route for GLORYS - it makes the absence of one explicit
before a transfer is paid for rather than after. D62, D68 and D70 are one pattern recorded three
times: the registry describing a store more confidently than the code behind it could deliver.

---

## TG12.1b - D72, the generated command uses the store it names (2026-08-29, `ed-dev`) - **COMPLETE**

**Reproduced before the fix.** The Acquire panel's GLORYS materialisation command supplied
`--levels -0.49402499198913574`. The CLI parsed every level with `int()`, raising before it could
inspect or materialise anything. It also constructed `CropSpec` directly, so even an integral
ocean coordinate selected ERA5's default `level` dimension rather than GLORYS's registered
`elevation` dimension.

**Fixed.** `_parse_level` is integer-first: `850` remains the integer `850`, preserving existing
ERA5 content keys, while a non-integral finite literal remains a float. The CLI now calls
`crop_for_store`, the registry-aware constructor already used by the HTTP path. A non-numeric
coordinate is refused as `InvalidParameterError` with the accepted forms named.

**Focused verification:**

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py src/tests/test_zarr_source.py -q
79 passed, 1 skipped, 5 warnings in 258.93s
```

The acceptance asserts the exact GLORYS fractional value and `vertical_dim="elevation"`, then
asserts that ERA5's four pressure levels remain integers on `vertical_dim="level"`. No remote
store was opened and no field value was transferred.

**Claim boundary.** This verifies the command construction and parsing boundary. It does not
show that a 51 GB ocean crop has been materialised or that any analysis path consumes one.

---

## TG12.1c - D71, portable immutable evidence publication (2026-08-29, `ed-dev`) - **COMPLETE**

**Measured defect.** With pytest's base directory pinned to the repository drive, the two
independent-overlap cases failed at `os.link` with `[WinError 1] Incorrect function`. `D:` is
exFAT and does not support hard links. The original ledger wording said all five private writers
failed this way; source inspection corrected that claim before the fix was recorded. Only
`era5_overlap` called `os.link` unconditionally. Gate run, gate campaign, evaluation run and
evaluation job already selected Windows rename, but duplicated the same scientific guarantee.

**Implemented boundary.** `src/core/publication.py:publish_new_bytes` creates a random temporary
beside the target, writes the complete byte payload, flushes and `fsync`s it, and then exposes it
with one atomic no-replace namespace operation:

* Windows: `rename`, which refuses an existing destination and works on the deployed exFAT drive;
* Linux: `renameat2(RENAME_NOREPLACE)`;
* macOS: `renamex_np(RENAME_EXCL)`;
* remaining POSIX fallback: hard-link publication, which is one atomic create-if-absent operation.

There is no existence-check-plus-rename path and no overwrite fallback. A filesystem unable to
supply the contract raises rather than weakening receipt immutability. All five writers now call
this primitive and retain their domain-specific `DataSourceError` / `ForecastContractError`
translation outside it.

**Acceptance on the deployed volume:**

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_publication.py -q
4 passed, 1 warning in 7.53s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_cds_source.py -q
19 passed, 1 warning in 36.49s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_publication.py src/tests/test_cds_source.py src/tests/test_gate_run.py src/tests/test_gate_campaign.py src/tests/test_evaluation_run.py src/tests/test_evaluation_job.py -q
51 passed, 1 warning in 66.64s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 131.20s
```

`test_publication_race_has_exactly_one_complete_winner` launches eight spawned processes against
one path under `.pytest-basetemp` on `D:`. Exactly one publishes its complete unique 53,248-byte
payload and seven receive `FileExistsError`; no temporary survives. Separate cases prove an
existing target remains byte-identical and an injected unsupported namespace operation leaves
neither target nor temporary. The primitive reports `windows-rename-no-replace`, so the
acceptance record identifies the mechanism it actually exercised.

**Whole-platform verification:**

```text
> .\.venv\Scripts\python.exe -m pytest -q
2694 passed, 2 skipped, 1 xfailed, 6 warnings in 1500.69s (0:25:00)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0

> cd frontend
> npm run build
✓ 1395 modules transformed.
dist/index.html                      0.65 kB │ gzip:     0.43 kB
dist/assets/index-CYmlBRdW.css      28.76 kB │ gzip:     5.98 kB
dist/assets/index-CyL6ppRC.js   10,148.38 kB │ gzip: 3,045.59 kB
✓ built in 1m 55s
```

The two skips are the explicit opt-in live-GCS read and live store probe. The xfail is the
retained strict historical transform case. The frontend's single 10.15 MB minified JavaScript
chunk remains a measured usability debt; this slice does not call it acceptable merely because
the build passed.

**Claim boundary.** The tests establish complete-or-absent visibility, no replacement under an
eight-process race, and successful publication on the deployed exFAT filesystem. They establish
process-crash atomicity of the receipt namespace transition. They do **not** establish survival
of a sudden power loss on every filesystem/storage device, and no such durability claim is made.

---

## TG12.2a - D69, explicit per-sample presence and the masked-lag decision (2026-08-29, `ed-dev`) - **COMPLETE**

**Measured decision before inference.** A controlled Argo-like union clock was constructed from
20 floats, 146 ten-day cycles and a distinct 12-hour surfacing offset for each float. It contains
2,920 clock rows and 380 ordered channel pairs. Every pair has `n_effective=0`, longest contiguous
joint-presence run 0, and no admissible requested lag. Candidate 1 (operate within maximal
contiguous joint-presence runs) therefore cannot recover an injected coupling at this sampling
shape without first binning or interpolating observations and inventing simultaneity. Candidate 2
was selected: intermittently present records refuse frame-lag inference until a physical-time
estimator exists.

**Implemented contract.** `ChannelSeries.present` is an optional, exact boolean `(time, channel)`
mask shared by all measures. It is declared rather than inferred from `NaN`: finite values where
presence is false are contradictions and are refused; a non-finite value where presence is true
remains the distinct observed-but-invalid state. `non_stationary_support` and the mask are
enforced in both directions at the tabular/domain boundary. Present counts enter channel records,
lineage, the channel API and UI. Splits slice the mask and recompute partition viability, including
an embargo entirely inside a gap. `PartitionIdentity` hashes the exact presence pattern without
reading measure values, so two held-out records differing only in sampling cannot collide.

**Inference boundary.** `masked_frame_lag_assessment` reports pairwise effective N, longest joint
run and admissible lags without compacting the clock. Presence-aware `decorrelation_frames` uses
only pairs genuinely separated by each physical frame lag and refuses if no lag has enough pairs.
The shift-null primitive can take a declared joint overlap before shifting, holding N constant for
every surrogate, but this does not license a frame interpretation of the compacted positions.
Consequently `cross_scale_dependency` refuses a partial mask before producing a statistic, bias
warning, null or significance claim. An all-true mask takes the literal accepted path and is
asserted result-identical to no mask.

**Verification:**

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_presence.py -q
14 passed, 2 warnings in 3.59s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_presence.py src/tests/test_channel_series.py src/tests/test_cross_scale.py src/tests/test_domain_gate.py src/tests/test_preregistration.py src/tests/test_tabular_domain.py src/tests/test_channels_api.py src/tests/test_analysis_api.py src/tests/test_preregistration_api.py src/tests/test_mining_api.py src/tests/test_cross_domain_api.py src/tests/test_frontend_contract.py -q
386 passed, 5 warnings in 249.57s (0:04:09)

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 184.81s (0:03:04)

> .\.venv\Scripts\python.exe tools/audit_docs.py
defects              : 72 defined, 70 fixed, partial ['D18'], open ['D43']
test functions       : 2362
stale inventory rows : none
claimed suite totals : architecture (2708, 1) / roadmap (2708, 1)
RESULT               : ok

> .\.venv\Scripts\python.exe -m pytest -q
2708 passed, 2 skipped, 1 xfailed, 6 warnings in 1806.08s (0:30:06)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

The two skips remain the explicit opt-in live-GCS read and live store probe; no public data was
fetched by this slice. The expected xfail is the retained historical transform case. Frontend
production build remained clean at 1,395 modules (`10,148.45 kB`, gzip `3,045.62 kB`).

**Claim boundary.** TG12.2a makes intermittent observation support representable, content-bound
and impossible to pass silently into the existing frame-lag estimator. It does not ingest Argo,
construct simultaneous profiles, estimate dependence in physical time, or claim that sparse
asynchronous floats support lagged inference. The refusal is the scientifically supported result.

---

## TG12.1d - D73, transform-derived acquisition planning (2026-08-29, `ed-dev`) - **COMPLETE**

The acquisition planner is metadata-only and transform-owned. Four-level DTCWT with
`near_sym_b`/`qshift_b` derives a coarsest parent margin of 97 pixels, native margin of 7,
absolute minimum 240 x 240 and R13 recommended minimum 512 x 512. Four-level SWT/db2 derives
margin 23, absolute minimum 47 x 47 and recommended minimum 256 x 256. Tests compare these
values directly with the registered implementations' support functions; no copied filter length
is accepted as an oracle.

The 384 x 320 fixture request expands symmetrically to exactly 512 x 512 native cells, from
latitude 36..547 and longitude 104..615, and is re-priced through the coordinate-only chunk
counter. A source too small to supply the threshold says so. A transform lacking a support
callback is refused before source access. Plan identity moves with the transform configuration
or exact coordinate bytes and does not move when only field values change. Explicit
materialisation below the recommendation is refused before constructing a field selection.

The API/UI contract pins a non-default SWT/db3/reflect request at depth three through Inspect and
into the generated CLI. Acquire renders both thresholds, per-level support and valid interiors,
suggested bounds, revised bytes/amplification and the plan digest, and applies the recommendation
in one action. The production build passes, but visual/assistive inspection is **NOT RUN**: the
in-app browser bootstrap succeeded and then reported that no browser was available.

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_crop_planner.py src/tests/test_frontend_contract.py src/tests/test_zarr_source.py -q
160 passed, 1 skipped, 5 warnings in 31.27s

> cd frontend
> npm run build
✓ 1395 modules transformed.
dist/index.html                      0.65 kB | gzip:     0.43 kB
dist/assets/index-DmSYtdi0.css      28.88 kB | gzip:     6.00 kB
dist/assets/index-a_MGaxn3.js   10,154.57 kB | gzip: 3,047.07 kB
✓ built in 1m 15s

> .\.venv\Scripts\python.exe tools/audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 73 defined, 71 fixed, partial ['D18'], open ['D43']
test functions       : 2370
stale inventory rows : none
claimed suite totals : architecture (2716, 1) / roadmap (2716, 1)
RESULT               : ok

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
19 passed, 1 warning in 110.86s (0:01:50)

> .\.venv\Scripts\python.exe -m pytest -q
2716 passed, 2 skipped, 1 xfailed, 6 warnings in 1462.14s (0:24:22)

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 29   FAIL 0   NOT_YET_RUNNABLE 0
```

The two skips remain the explicit opt-in live-GCS read and live store probe. No public field
values were fetched. The xfail is the retained historical transform case. The warnings are the
six pre-existing SQLAlchemy, multipart, Python-version-support and empty-slice warnings; none is
new to this slice. The 10.15 MB minified frontend chunk remains explicit performance/usability
debt rather than being called acceptable because compilation passed.

**Claim boundary.** The plan verifies transform support, native-coordinate feasibility and
predicted chunk cost for one exact request. Its recommended threshold applies the named
128-parent-cell R13 span policy; it is not an empirical power calculation and does not establish
stationarity, physical relevance, successful transfer, valid returned field values or a future
scientific finding.

---

## TG12.2b-d - Irregular profiles, declared reductions and the Argo GDAC seam (2026-08-29, `ed-dev`) - **COMPLETE**

The profile acquisition pathway introduces a `ProfileSpec` to content-address non-gridded profiles in a given region, time window, and depth range. `ProfileCollection` preserves the exact scatter representation, without grid assumptions, asserting that atmospheric structures cannot simply be forced onto ocean profiles. 

The `ProfileReduction` registry provides explicitly declared methodologies (like per-float reduction and depth-bin aggregation), which dictate whether non-stationary support is preserved or whether aggregation causes an identity shift. The real Argo GDAC query runs through an isolated seam, strictly decoupled from the core, emitting actual profile data rather than fixtures, and rightfully declining operations missing necessary domain parameters. The UI (`ProfileAcquisition.tsx`) renders these declared irregularities honestly and appropriately blocks analysis requests that would otherwise interpolate or invent data.

**Verification, reconciled after the interrupted handoff.** The original block claimed
`2726 passed, 2 skipped` but the checked-in run log showed a documentation failure and the new
opt-in Argo case necessarily made a third skip. That claimed full-suite line is withdrawn rather
than laundered by editing its number. Focused replay passed, and the acceptance that matters for
TG12.2d was rerun explicitly:

```text
> $env:SPECTRALEARTH_RUN_LIVE_ARGO='1'
> $env:SPECTRALEARTH_ALLOW_NETWORK='1'
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_profiles.py::test_live_argo_gdac_small_query_when_network_is_explicitly_enabled
1 passed, 2 warnings in 14.47s
```

The bounded live response produced multiple floats, unequal per-float sample counts and partial
presence. The production frontend also rebuilt successfully after the profile path was retained.

**Claim boundary.** The reduction registration is fully documented and is demonstrably reproducible; it does not claim to validate any specific scientific assumption or cross-scale finding. The dataset simply represents recorded evidence of domain irregularities natively preserved within the analysis pipeline.

---

## TG13.1-13.2 - TESS light curves and executable violation coverage (2026-08-29, `ed-dev`) - **IMPLEMENTED; LIVE MAST OPEN**

The fake three-point source left by the interrupted agent was removed. Focused acceptance creates
checksum-valid calibrated SPOC-shaped FITS bytes, proves exact TIC/sector product preflight,
mutates flux without changing length and observes a different collection digest, refuses byte
caps before download, round-trips immutable publication, checks atomic onboarding and exercises
the HTTP claim boundary. Acquisition coverage is generated by joining declarations to available
paths, and every known violation has a path.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_photometry.py src/tests/test_acquisitions_api.py src/tests/test_domain_onboarding.py src/tests/test_geometry_registry.py
72 passed, 2 warnings in 5.18s
```

The official MAST metadata request exceeded its 45-second bound during live inspection. The
adapter reported a timeout rather than an empty catalogue and no FITS value was fetched. Live
MAST acceptance therefore remains OPEN.

## TG14.1-14.2 - File-first ingress and fixed representation audit (2026-08-29, `ed-dev`) - **COMPLETE FOR CSV/TSV SAMPLE TABLES**

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_dataset_ingress.py
6 passed, 2 warnings in 4.02s

> cd frontend
> npm run build
✓ 1398 modules transformed.
dist/assets/index-DdKE8XX3.js  10,180.69 kB | gzip: 3,052.47 kB
✓ built in 55.59s
```

The planted audit has 240 independent samples, two raw features, one PCA component, one declared
nuisance, 199 fixed permutations and three tests per partition. The planted feature survives BY
correction on both generate and held-out confirmation; the unrelated feature has lower held-out
MI. Changed file bytes and an edited plan are refused before compute. This is an association
audit, not a certificate, feature recommendation or conditional-information claim.

### Finished-tree verification (2026-08-30)

The two failures found by the first full replay were contract/bookkeeping defects: the inventory
still named the pre-slice test-function total, and the light-curve capabilities service was not
called by the UI. Both were corrected before this finished-tree replay.

```text
> .\.venv\Scripts\python.exe -m pytest -q
2741 passed, 4 skipped, 1 xfailed, 6 warnings in 1469.28s (0:24:29)

> cd frontend
> npm run build
✓ 1398 modules transformed.
dist/assets/index-D4D53_cW.js  10,181.00 kB | gzip: 3,052.25 kB
✓ built in 1m 18s

> .\.venv\Scripts\python.exe tools\audit_docs.py
undocumented modules : none
undocumented routes  : none
test functions       : 2397
stale inventory rows : none
claimed suite totals : architecture (2741, 1) / roadmap (2741, 1)
RESULT               : ok
```

The four skips are intentionally opt-in live GCS, store-probe, Argo and TESS/MAST checks. The
Argo check was separately exercised live above; the bounded TESS/MAST acceptance remains open.
The warnings and oversized frontend chunk remain explicit maintenance debt.

---

## TG15.1-15.2 - Dataset capability routing and explained UI gating (2026-08-30, `ed-dev`) - **COMPLETE**

`DatasetCapabilityProfile` binds operation decisions to an exact file, plan, collection or
reduction identity. The same backend registry now routes declared sample tables, admitted channel
records, planned grid crops, acquired Argo reductions and acquired TESS collections. The shell
keeps unavailable paths visible and renders the returned reason rather than silently greying or
hiding them. Backend entry points retain their own refusals.

The implementation review found one safety defect in G14: grouped and ordered declarations could
reach a row-random generate/confirm split. That path is now refused before plan creation. Grouped
data name group-held-out confirmation as the missing recipe; ordered data name blocked and
embargoed confirmation.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_dataset_ingress.py src/tests/test_channels_api.py src/tests/test_profiles.py src/tests/test_photometry.py src/tests/test_zarr_source.py -x
109 passed, 3 skipped, 5 warnings in 23.08s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_frontend_contract.py -x
94 passed, 5 warnings in 7.80s

> cd frontend
> npm run build
✓ 1399 modules transformed.
dist/assets/index-BUwU39Jz.js  10,185.28 kB | gzip: 3,052.89 kB
✓ built in 53.36s

> .\.venv\Scripts\python.exe -m pytest -q
2745 passed, 4 skipped, 1 xfailed, 6 warnings in 1423.99s (0:23:43)
```

The four skips and one xfail have the same explicit meanings as the preceding finished-tree run.
Capability availability means only that declared and verified prerequisites are present; it is
not evidence that the operation ran, succeeded or established a scientific claim.

---

## TG16.0 - Representation-structure admission and benchmark prerequisites (2026-08-30, `ed-dev`) - **COMPLETE**

The first G16 admission is independent-sample only, through the same authoritative refusal now
used by the G14 representation plan. Grouped and ordered data remain unavailable until their
respective null and confirmation schemes exist. The paired benchmark family freezes all nine
declared planted/safeguard cases plus the later operation-level calibration and power thresholds.
No G16 scientific operation was registered; the premature G15 `association_redundancy` entry was
removed.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_dataset_ingress.py
9 passed, 2 warnings in 4.41s

> .\.venv\Scripts\python.exe -m src.benchmarks
PASS 31   FAIL 0   NOT_YET_RUNNABLE 0
```

The focused benchmark/ingress run covering both files initially reported 55 passes and one
failure because the strengthened grouped refusal no longer contained the existing public phrase
`group-held-out confirmation`. The final wording preserves that phrase and adds the benchmarked
null requirement; the rerun above passes. The full backend suite was not rerun for this bounded
prerequisite slice; the clean 2745-pass G15 run remains the latest full-suite evidence.

---

## TG16.1 - Redundancy Structure Audit (2026-08-30, `ed-dev`) - **COMPLETE**

The registered operation now freezes the complete two-candidate group family, estimator,
equiprobable discretisation, conditional-permutation nulls, seeds, alpha and
Benjamini-Yekutieli correction before enumeration. Its structure map distinguishes supported
redundancy, supported complementarity and unresolved pairs. The joint test preserves the XOR
control despite weak singleton information, while the response explicitly denies a
partial-information decomposition, feature-removal instruction or representation recommendation.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_representation_structure.py src/tests/test_dataset_ingress.py
15 passed, 2 warnings in 4.55s

> .\.venv\Scripts\python.exe -m src.benchmarks --name representation_structure_planted --name representation_structure_safeguards
PASS 4   FAIL 0   NOT_YET_RUNNABLE 0
```

The benchmark command ran the frozen 200-replication power/calibration family. Exact-duplicate and
noisy-copy redundancy power was 1.00 and 0.94; complementary and XOR power was 1.00 each; the
independent false-claim rate was 0.02; and the four one-candidate conditioning safeguards made no
pair claim. The full backend suite was not rerun; the clean 2745-pass G15 run remains the latest
full-suite evidence.

---

## TG16.2 - Conditional-Information Audit (2026-08-30, `ed-dev`) - **COMPLETE**

The registered operation freezes the complete raw-feature family, the single researcher-declared
nuisance, equiprobable Miller-Madow conditional mutual information, overlap/effective-support and
linear conditional-model admission, conditional-randomisation draws, seed, alpha and global
Benjamini-Yekutieli correction. The null reconstructs target from the fitted target/nuisance
relationship plus permuted residuals instead of globally permuting target. Responses report only
conditional association and expressly decline confounding, collider, post-treatment and causal
interpretations.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_conditional_information.py src/tests/test_dataset_ingress.py src/tests/test_representation_structure.py
21 passed, 2 warnings in 4.68s

> .\.venv\Scripts\python.exe -m src.benchmarks --name representation_structure_planted --name representation_structure_safeguards
PASS 6   FAIL 0   NOT_YET_RUNNABLE 0
```

The benchmark command ran the frozen 200-replication family. Every applicable construction met
the support/model admission rule. Signal-survival power and collider conditional-association
detection were 1.00. Nuisance-only and conditional-null false-claim rates were 0.055 and 0.045,
below the frozen 0.075 ceiling. The collider detection is a safeguard against causal overwording,
not evidence that conditioning removed a confounder. The full backend suite was not rerun; the
clean 2745-pass G15 run remains the latest full-suite evidence.

---

## TG16.3 - Stable-Subspace Generation (2026-08-30, `ed-dev`) - **COMPLETE**

The registered operation seals the complete compact linear family, generate/confirmation split,
generate-only standardisation, supervised covariance objective and optional nuisance penalty,
ridge values, seeded block-power optimiser, restarts, perturbations, target-permutation refits,
alpha and global Benjamini-Yekutieli correction. Results identify spans by projectors, report
multi-seed perturbation distances, and keep the reserved confirmation partition unopened.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_stable_subspace.py src/tests/test_dataset_ingress.py
15 passed, 2 warnings in 4.50s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_benchmarks.py -k "gating_lookup or g16_benchmark_pair"
2 passed, 45 deselected, 1 warning in 1.97s

> .\.venv\Scripts\python.exe -m src.benchmarks --name representation_structure_planted --name representation_structure_safeguards
PASS 8   FAIL 0   NOT_YET_RUNNABLE 0
```

The first frozen run measured an independent false-candidate rate of 0.085, above the declared
0.075 ceiling, and therefore failed. The original 2% perturbation was too weak to distinguish a
chance direction from a scientifically stable span. The recipe was corrected before completion
to 10% perturbations and a 0.10 projector-distance ceiling. On the final 200-replication family,
exact-duplicate, noisy-copy and complementary linear candidate rates were 1.00 each, nonlinear
XOR was 0.025, and the independent false-candidate rate was 0.055. Every one-feature case made no
compact-subspace claim. The full backend suite was not rerun; the clean 2745-pass G15 run remains
the latest full-suite evidence.

---

## TG16.4 - Held-out and Nuisance-region Confirmation (2026-08-30, `ed-dev`) - **COMPLETE**

The confirmation workflow now freezes every TG16.3 family member's projector/application basis,
generate-only preprocessing, candidate state, generate-derived nuisance-region cuts, correction
unit, permutation ensemble and thresholds before held-out outcomes are used. Confirmation applies
those values unchanged, corrects the complete searched family, refuses inadequate frozen-region
overlap, and spends the content-and-index-bound partition once through the programme-wide durable
held-out ledger. Its strongest result is internal replication within one dataset, not external
certification, evidence storage, rung movement, optimality or causality.

```text
> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_stable_subspace.py
10 passed, 2 warnings in 4.56s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_stable_subspace.py src/tests/test_dataset_ingress.py src/tests/test_representation_structure.py src/tests/test_conditional_information.py
31 passed, 2 warnings in 5.19s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_benchmarks.py -k "gating_lookup or g16_benchmark_pair"
2 passed, 45 deselected, 1 warning in 1.97s

> .\.venv\Scripts\python.exe -m src.benchmarks --name representation_structure_planted --name representation_structure_safeguards
PASS 10   FAIL 0   NOT_YET_RUNNABLE 0
```

The paired gate ran 200 frozen replications with 156 generate and 68 confirmation rows.
Exact-duplicate, noisy-copy and complementary-linear internal-replication rates were 1.000 each;
nonlinear XOR was 0.005 and the independent null was 0.000, below the 0.075 ceiling. Every
one-feature case remained outside the compact family. No acceptance setting was changed after
this run. The full backend suite was not rerun; the clean 2745-pass G15 run remains the latest
full-suite evidence.

---

## TG16.5 - External Certification Seam and UI Integration (2026-08-30, `ed-dev`) - **COMPLETE**

Published candidate definitions now retain exact TG16.4 lineage and source preprocessing/span.
A separate no-adaptation contract binds them to a different content-addressed target and declared
acquisition provenance before target access. Certification verifies the independently published
contract digest, spends the whole target once, applies every source value unchanged, corrects the
complete transfer family and emits the only `external_replication_receipt` in the programme. The
receipt certifies the executed test and recorded provenance, not the truth of an independence
declaration, optimality, causality or population-wide transportability.

The integration audit also found eight committed TG16.1-TG16.4 routes that the UI could not call.
The generic-file workbench now exposes the complete progressive structure programme, including
the three TG16.5 routes. Argo's measure choice updates the acquisition projection as well as the
reduction; acquisition domain/path survive rerenders and browser refresh; Argo/TESS receipts now
say explicitly that an acquired dataset is not automatically an evidence study. MAST metadata
long-polling uses a 90-second bounded transport window and retries one transient transport failure.

```text
> npm run build
✓ built in 1m 6s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_frontend_contract.py src/tests/test_stable_subspace.py src/tests/test_photometry.py src/tests/test_profiles.py
121 passed, 2 skipped, 5 warnings in 8.00s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_stable_subspace.py
13 passed, 2 warnings in 4.86s

> .\.venv\Scripts\python.exe -m src.benchmarks --name representation_structure_planted --name representation_structure_safeguards
PASS 12   FAIL 0   NOT_YET_RUNNABLE 0

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_photometry.py src/tests/test_profiles.py src/tests/test_acquisitions_api.py src/tests/test_frontend_contract.py src/tests/test_stable_subspace.py src/tests/test_benchmarks.py -k "not suite_results and not no_null_benchmark_ever_fails and not every_benchmark_passes_its_declared_gates"
177 passed, 2 skipped, 1 deselected, 5 warnings in 306.15s

> .\.venv\Scripts\python.exe -m pytest -q src/tests/test_documentation.py
19 passed, 1 warning in 367.50s
```

Across 200 frozen replications, exact-duplicate, noisy-copy and complementary-linear external
receipt rates were 1.000 each. XOR, the independent null and all one-feature safeguards were
0.000. The two live-network tests remained opt-in and were skipped; the TESS retry itself is
covered with a deterministic first-timeout/second-success test. The full suite was not rerun at
the user's request; the clean 2745-pass G15 run remains the latest full-suite evidence.

---

## TG17.0 - Four-domain Flagship Contract and Known-answer Benchmarks (2026-08-30, `ed-dev`) - **COMPLETE**

The flagship contract is frozen before any configurable experiment operation exists. Reanalysis,
Argo, TESS and order book cross four distinct source/shape contracts; calendar-aligned and
scale/shape-aligned questions remain separate; week, three-month and six-month presets form the
initial duration family. Order book intentionally uses a licensed user-supplied content-addressed
record and keeps `lag_policy="none"`, so an apparent event order exercises a mandatory precedence
refusal instead of acquiring atmospheric semantics through the common workbench.

The two registered benchmarks cover six independent-construction cases: shared calendar event,
scale-shifted motif, same-window null, shared-gap alias, inadmissible precedence and an unaffordable
24,576-member family. Later operation acceptance is frozen at 200 replications, alpha 0.05,
maximum null rejection 0.075, minimum planted detection 0.80, maximum 10,000 family members,
4 GiB planned bytes and one hour planned runtime. TG17.0 registers no experiment capability and
emits no scientific result or evidence.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_benchmarks.py -k "g17 or gating_lookup" -q
4 passed, 46 deselected, 1 warning in 2.12s

> .\.venv\Scripts\python.exe -m src.benchmarks --name multidomain_flagship_planted --name multidomain_flagship_safeguards
PASS 2   FAIL 0   NOT_YET_RUNNABLE 0
```

The focused gate measured a maximum 21,600-second peak separation for the shared event (inside
one 43,200-second Argo interval), minimum native peak contrast 0.966, an 11.875-fold planted native
width range, 0.994 shared-gap coverage correlation, the exact planted domain event order, and the
24,576 versus 10,000 resource refusal. The full suite was not rerun at the user's request; the
clean 2745-pass G15 run remains the latest full-suite evidence.

---

## TG17.1 - Versioned Experiment Manifest and Composer Preflight (2026-08-30, `ed-dev`) - **COMPLETE**

The first operable G17 product slice uses one immutable `CrossDomainExperimentSpec` from saved
flagship recipe through API validation, content-addressed draft revision and metadata preflight.
The canonical JSON bytes derive both manifest digest and future run identity. Exact UTC boundaries
replace ambiguous duration labels, and all three duration windows form one 288-member family.

Preflight uses no network and opens no measurement values. It retains reanalysis grid extent,
Argo sparse point support and TESS sector-bounded support as different coverage facts, reports
access/bytes/gap and expected-sample basis per exact window, and refuses the unbound local order-
book record. A complete-coverage policy refuses unresolved sparse/sector coverage; it does not
silently narrow the study. The visible Composer saves and reloads the exact manifest across browser
refresh and leaves Run disabled until TG17.6. It neither creates evidence nor moves a claim rung.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_experiment_manifest.py -q
12 passed, 2 warnings in 5.24s

> .\.venv\Scripts\python.exe -m pytest src\tests\test_experiment_manifest.py src\tests\test_frontend_contract.py -q
107 passed, 5 warnings in 8.51s

> npm run build
✓ 1401 modules transformed; built in 1m 31s (TypeScript and Vite production bundle)
```

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

---

## TG17.2 - Canonical Structural-Trajectory Contract (2026-08-30, `ed-dev`) - **COMPLETE**

The immutable `StructuralTrajectory` retains exact native interval support, validity, semantic
identity, units, structural/native scale mapping, native record identity and locator,
adapter/version/config digests, assumption violations and reconstructable per-value lineage.
Executable adapter declarations constrain channels to benchmarked definitions and declare axes,
roles, invariances, consumed information, clock/support, missing-data behaviour, legitimate null,
leakage risks and refused operations.

Reanalysis, Argo and TESS known-answer records enter the same domain-blind mining seam. Focused
conformance covers exact reconstruction, immutable raw/native separation, and mandatory refusal
of semantic leakage and undeclared clock/support changes. The manifest-bound Composer preview
shows the native clock/support, unit, scale, adapter/native digests and limits while stating that
these are deterministic fixtures rather than acquired data. Live translation and scientific
execution remain unavailable.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_structural_trajectory.py src\tests\test_experiment_manifest.py src\tests\test_frontend_contract.py -q
118 passed, 5 warnings in 8.48s

> cd frontend && npm run build
✓ 1401 modules transformed; built in 1m 11s (TypeScript and Vite production bundle)
```

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

## TG17.3 - Adapter Registry, Schema-Driven Controls and Conformance Kit (2026-08-30, `ed-dev`) - **COMPLETE**

Acquisition plus structural translation is now a registered `DomainExperimentAdapter` rather than
an orchestrator switch. One registration carries the domain declaration, a typed `ControlSchema`,
the acquisition planner, translator configuration, materializer, structural declaration and
translator, capability derivation, null builder and provenance renderer. `preflight_manifest` lost
its literal `source_plans` table and its `channel_table:local` special case and resolves every
coverage row through the registered adapter.

Window arithmetic stayed in the framework: an adapter declares support kind, native cadence,
exactness, access and cost per day, and `plan_windows` derives expected samples, bytes and gap
status identically for every domain. Reanalysis and the bespoke family register from `src/adapters`;
Argo and TESS register from `extensions/` through the same public seam.

The conformance kit runs ten checks and executes what a declaration claims rather than trusting
it. `INVARIANCE_PROBES` applies each declared invariance to the native record and compares every
canonical channel; an invariance with no registered probe reports `NOT_PROBED` rather than `PASS`.
All four flagship adapters pass 11 of 11 checks against their deterministic known-answer records,
visibly through `POST /api/v1/experiment-composer/adapters/{adapter_id}/conformance`.

Order book became one declaration of a bespoke record family rather than a finance adapter. Its
fence is TG8.4's rule imported unchanged - detection may create a required declaration and may
never satisfy one - so an observed irregular clock obliges `irregular_sampling` on an onboarded
domain, an aggregate footprint obliges `aggregated_values`, a flat record is refused under a
domain declaring richer axes, and `lag_policy="none"` adds `precedence` to the refused operations
by construction. A bespoke domain is added by declaration alone with no code.

That is explicitly not treated as the acceptance evidence. The acceptance test installs a synthetic
fifth adapter with different structural mathematics - a monotone rank channel - from a module the
application never imports, and it reaches the registry, the control schema, the conformance kit and
the domain-blind mining seam with no edit to the orchestrator, the generic API routes or the UI.

Three defects were forced out. TG17.2's `assert_structural_conformance` reconstructed values from a
hardcoded standardized-level formula and compared every configuration digest against `{"ddof": 0}`,
which failed the fifth adapter for having different and correct arithmetic; `LINEAGE_RECONSTRUCTORS`
now dispatches on the declared operation and an unregistered operation fails rather than passes.

**D74**: the route-count guard had matched the words "those routes" written into architecture.md
by TG17.1 and had been unable to parse its own claim since. **D75**, found by fixing D74: the
guard enumerated a hand-maintained list of ten source files and four mounted routers were missing
from it — `profiles`, `lightcurves`, `ingress` and `experiment_composer` — so 32 served endpoints,
including the whole TG16 ingress surface, were invisible to every check in that file. architecture.md
claimed 75 routes; 107 are served, and the section 3.12 table was missing 21 rows. The list is
deleted rather than corrected: routes now come from the application object. Both are fixed and the
table is complete.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_adapter_registry.py src\tests\test_structural_trajectory.py src\tests\test_experiment_manifest.py src\tests\test_frontend_contract.py src\tests\test_domain_onboarding.py src\tests\test_acquisitions_api.py -q
180 passed, 5 warnings in 6.94s

> .\.venv\Scripts\python.exe -m pytest src\tests\test_documentation.py -q
19 passed, 1 warning in 469.84s

> cd frontend && npm run build
1402 modules transformed; built in 51.34s (TypeScript and Vite production bundle)
```

No live archive is acquired, no cross-domain statistic runs, no evidence is written and no claim
rung moves. The `source_binding` control makes the deterministic known-answer binding a visible
manifest-recorded choice; the live binding refuses by naming TG17.6.

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

## TG17.4 - Clock, Support and Coverage Semantics (2026-08-31, `ed-dev`) - **COMPLETE**

Cross-domain comparison is interval arithmetic over half-open `[start, end)` support. Two records
that arrive as arrays of equal length look aligned and are not; a row index is a position in a
file and nothing in either file says the two were written on the same clock.

The load-bearing guarantee is that **changing row density alone cannot manufacture support**.
Rewriting every fixture with sixty times as many rows over identical support leaves occupied
duration, overlap, governing scale and effective sample size unchanged. Effective sample size is
overlap duration over the coarser of the two native scales; raw row counts are reported in every
row and used by nothing. `[a, b)` and `[b, c)` abut and do not overlap.

Only `exact_support_overlap` runs without being named, and it transforms nothing. Every widening,
snapping or carrying kernel is frozen in the manifest's `AlignmentPolicy`, travels inside the
manifest digest, must be admitted by every participating adapter through `admissible_kernels`,
takes no framework-default parameter, and reports in seconds the overlap it created rather than
observed. A value-inventing kernel is refused over a domain declaring `irregular_sampling` or
`aggregated_values`. No registered adapter admits `carry_forward`.

Calendar mode may not silently search normalized scale ratios and scale/shape mode may not emit
simultaneity, precedence or causal language; the manifest refuses the mismatch where the search
is declared. Scale/shape correspondences retain both native durations. `elapsed_seconds` refuses
a naive local timestamp, and a daylight-saving day is 82,800 seconds rather than 86,400.

Preflight binds the kernel and reports pairs as bounded by the window where metadata cannot
establish an overlap, rather than offering a number that would later turn out to have been a
guess. `POST /api/v1/experiment-composer/manifests/alignment` measures the known-answer records'
actual support, stating that binding in every response, and the coverage view draws it by time
rather than by index.

The six adversarial fixtures each align as declared or refuse by name: unequal cadence, abutting
boundary, daylight-saving day, sparse profile, interrupted light curve, non-stationary support.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_structural_alignment.py src\tests\test_frontend_contract.py src\tests\test_adapter_registry.py src\tests\test_experiment_manifest.py src\tests\test_structural_trajectory.py src\tests\test_domain_onboarding.py src\tests\test_acquisitions_api.py src\tests\test_benchmarks.py src\tests\test_cross_domain.py src\tests\test_documentation.py -q
362 passed, 5 warnings in 539.08s (0:08:59)

> cd frontend && npm run build
1403 modules transformed; built in 55.38s (TypeScript and Vite production bundle)

> git diff --check
(clean)
```

No live archive is acquired, no cross-domain statistic runs, no evidence is written and no claim
rung moves.

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

## TG17.5 - Multi-Domain Family Accounting and Domain-Legitimate Nulls (2026-08-31, `ed-dev`) - **COMPLETE**

A four-domain study is a search over domain combinations, windows, channels, scales,
relationships, lags, representations and motifs. Every one of those is a knob that can be turned
after seeing a result, so every one of them is declared, and the family is the product of all of
them counted once. `src/core/experiment_family.py` builds that product as a `SearchSpecification`
over eight axes and prices it before acquisition. The inline product in `preflight_manifest` and
its copy in the browser are gone; a family size that could be read two ways is a family size that
will be.

Domain combinations are unioned across declared arities rather than multiplied: pairs and triples
of four domains are 6 + 4 = 10 members, because a member is one combination.

**The check found what it exists to find.** Priced for the first time, the flagship's 288 declared
tests need about **35,953 surrogates** under Benjamini-Yekutieli at alpha 0.05. The frozen
acceptance policy declares **200**, at which the largest affordable family is **four**. That
study could have run to completion, cost the full amount and been arithmetically incapable of
rejecting anything - reporting nothing for an arithmetic reason indistinguishable afterwards from
a clean negative. It is logged as **D76**, and the manifest gained the `ConfirmationPolicy` that
makes the two legitimate answers separable: a `confirmatory_only` study is priced at its complete
declared family and refused when it cannot resolve it, and a `generate_then_confirm` study must
name the held-out partition it will confirm on and how many members. The flagship declares the
second. Its generate stage is labelled in every payload as producing candidates and not claims.
**D77** fell out of the same pass: the flagship's null had carried a `preserve_gaps` parameter
that nothing read since TG17.1.

A pairwise screen never shrinks the correction unit. `correct_over_candidates` refuses the number
of survivors by name and permits it only against a named held-out partition, because a
confirmatory family is legitimately small only when it was frozen before that partition was
opened.

Two of the four flagship domains declare no justified lag policy. They take part in structural
association; the 240 declared members that pair them at a precedence relationship are reported
unavailable, and the declared family size is unchanged - a family narrowed to what survived is a
family chosen after looking.

Nulls became declared objects with a comparison mode and a named list of what they preserve and
destroy. A calendar null is refused for a scale/shape question where the question is declared.
Four admissible families are registered - the plain clock shift, a whole-cycle shift that keeps
seasonal phase, a within-group shift that never moves a value across a declared group boundary,
and a scale/shape partner reassignment that alters no record at all - and none of their parameters
has a framework default. `global_value_shuffle` is **registered and refused**: every domain can
execute it, which is exactly why it must be refusable by name rather than quietly absent. Which
nulls a domain's support can carry is declared per adapter and bound against every participating
adapter in preflight.

The TG17.0 fixtures calibrate at the frozen family level - six pairs, one correction, 999
replications, the declared null applied through `bind_null`, and a support-weighted statistic that
inherits TG17.4's density invariance. The planted `shared_calendar_event` is confirmed on **6 of
6** pairs. `same_window_unrelated`, `gap_alias` and `inadmissible_precedence` each reject **0 of
6**: an identical outer interval and a shared observation gap do not become shared structure.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_experiment_family.py src\tests\test_structural_alignment.py src\tests\test_frontend_contract.py src\tests\test_adapter_registry.py src\tests\test_experiment_manifest.py src\tests\test_structural_trajectory.py src\tests\test_domain_onboarding.py src\tests\test_acquisitions_api.py src\tests\test_benchmarks.py src\tests\test_cross_domain.py src\tests\test_family_accounting.py src\tests\test_preregistration.py src\tests\test_documentation.py -q
513 passed, 5 warnings in 556.20s (0:09:16)

> cd frontend && npm run build
built in 1m 17s (TypeScript and Vite production bundle)

> git diff --check
(clean)
```

No live archive is acquired, no confirmatory statistic runs, no evidence is written and no claim
rung moves. A calibration on fixtures with known answers is not a result about any domain.

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

## TG17.6 - Content-Addressed, Resumable Experiment Orchestrator (2026-08-31, `ed-dev`) - **COMPLETE**

A four-domain study is a long job over remote archives, and long jobs get interrupted. Every
interruption offers the same recovery - start again, and run whatever is available this time - and
taking it substitutes a smaller experiment for the declared one while producing a receipt
indistinguishable from a study that always intended to be that size. `src/core/experiment_run.py`
executes a frozen manifest as one state machine, held as a transition **table**, so what a run was
allowed to do next is one dictionary rather than a chain of branches.

**Run identity is the manifest.** It is a content address over the schema and the manifest digest
and nothing else - no clock, no UUID, no machine - so executing an identical manifest *is* the same
run. `POST /api/v1/experiment-runs` resumes rather than creates, and the Composer has no "new run"
control because there is no such operation.

**Every step is content-addressed.** A step key is the digest of the run, the stage, the component
and the digests of that step's declared inputs; a completed step is published immutably through
`publish_new_bytes` and replayed from disk. That is the no-duplicate-acquisition guarantee and the
drift check at once: a changed native artefact re-keys its translation rather than being paired
with a stale one. Operational failures are deliberately *not* published under a step address, since
a timeout is a fact about a network at a moment and not a function of the declared inputs - and
publishing it there would make a successful retry look like one address disagreeing with itself.

**A retry may not author a new plan.** `retry` re-executes only the components recorded `FAILED` or
`TIMED_OUT`, and refuses a manifest whose digest differs from the frozen one. `REFUSED` and
`MISSING` are the archive answering rather than failing, so they are terminal: the remedy is
`editable_copy`, which writes a new mutable draft and leaves the frozen run's journal unchanged.
There is no `unfreeze` - a frozen run edited after seeing how it went is a plan chosen with
knowledge of the result.

**Partial acquisition is the frozen policy's decision.** `decide_stage` is a pure function of the
manifest and the component statuses, auditable without reconstructing a run: `complete_required`
refuses, `partial_permitted` admits only above its declared minimum fraction, and the receipt names
every missing component either way. A refusal outranks a failure and a failure outranks a missing
component, because "we could not ask" is not "the answer is no".

**Progress cannot leak an unopened result.** `ComponentOutcome` carries a status, a digest, a
bounded work estimate and a remediation string, and has nowhere to put a measurement value - so a
progress feed watched during `MINING` reports that mining is happening and cannot report what it
found. `runs/<run_id>/journal.jsonl` is the only state: append-only, fsync'd per line, folded on
read, with a torn tail skipped rather than raised.

`src/core/run_workers.py` registers the stage-worker suites instead of accepting behaviour from a
request, and everything registered today **acquires nothing**. `fixture_dry_run` rehearses a frozen
plan end to end; `fixture_transient_failure` times out each acquisition component once and
completes it on retry, reading "first attempt" from the run's own journal so the rehearsal behaves
identically across a restart or four separate HTTP requests. Eight routes and
`frontend/src/components/RunMonitor.tsx` put the machine in the browser, with Retry and "open an
editable copy" as two buttons that are never both live.

**Two things the widened verification set found.** `test_analysis_api.py`'s TG11.1 acceptance test
asserted `len(names) == 13` before posting every sequence and cross-domain benchmark over HTTP;
TG17.0 registered two more, so the assertion aborted the test *before the HTTP call* and the two
four-domain flagship benchmarks were never exercised across the API boundary that test exists to
exercise (**D78**, fixed by deriving the count with a floor and naming them). And the shared
`client` fixture gained the filesystem half of D24's reasoning: persisting routes fall back to
`data/` when nothing binds them, which for a content-addressed run means an unbound test posting a
manifest would resume, and then advance, whatever real run that manifest already had.

**Acceptance met.** A killed run resumes without re-requesting the coverage it had acquired; a torn
journal tail is skipped and the run still resumes; re-executing an identical complete manifest
returns the same run identity and byte-identical artefact digests and requests nothing. A timed-out
acquisition leaves the run `FAILED` with a remediation and is retried over HTTP, re-executing only
that component. A retry carrying a different manifest is refused by digest. A permanent refusal is
terminal, is not retryable, and is answered with an editable copy that leaves the frozen run
unchanged.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_experiment_run.py src\tests\test_experiment_manifest.py src\tests\test_experiment_family.py src\tests\test_frontend_contract.py src\tests\test_structural_alignment.py src\tests\test_adapter_registry.py src\tests\test_structural_trajectory.py src\tests\test_domain_onboarding.py src\tests\test_acquisitions_api.py src\tests\test_benchmarks.py src\tests\test_cross_domain.py src\tests\test_family_accounting.py src\tests\test_preregistration.py src\tests\test_api_infrastructure.py src\tests\test_analysis_api.py src\tests\test_channels_api.py src\tests\test_cross_domain_api.py src\tests\test_evidence_api.py src\tests\test_findings_api.py src\tests\test_mining_api.py src\tests\test_preregistration_api.py src\tests\test_profiles.py src\tests\test_reviews_api.py src\tests\test_documentation.py -q
811 passed, 1 skipped, 5 warnings in 677.37s (0:11:17)

> cd frontend && npm run build
built in 1m 5s (TypeScript and Vite production bundle)

> git diff --check
(clean)
```

No live archive is acquired, no confirmatory statistic runs, no evidence is written and no claim
rung moves. The registered suites are rehearsals, and a rehearsal that completes is not a result.

The full suite was not rerun at the user's request; the clean 2745-pass G15 run remains the latest
full-suite evidence.

## TG17.7 - The Guided Path, the Ladder, and the First Browser Test (2026-08-31, `ed-dev`) - **COMPLETE**

The workbench before this slice was four acquisition surfaces and a Composer whose panels could be
visited in any order. Nothing about that was broken, and that is the problem: the order of
operations *is* the scientific discipline. A family priced after acquisition is priced knowing what
the data looked like. A null admitted after the statistic exists is not a null. A UI that permits
those in any order has not made an error - it has made the error **undetectable**, because no
receipt can distinguish an experiment that was declared from one that was assembled.

`src/core/composer_path.py` therefore holds the workflow as a registry rather than a layout.
`COMPOSER_PATH` carries the seven steps, each a `PathStep` deciding its own status from the
manifest, and `compose_state` returns exactly **one** `next_action`. The browser renders that; it
does not compute it. Statuses are three-valued, because "you have not done this" and "this cannot
be done yet" are different sentences and only one is the researcher's move. A blocked step keeps
its tab and its reason: the whole flagship blocks at preflight naming `order_book`, rather than
dropping the domain and reporting a complete three-domain study. `STAGE_LADDER` names acquired
material, an executed run, a finding and admitted evidence, each with what it is **not** and its
own gate; this surface moves a researcher across the first two and structurally cannot move them
across the last two.

Three things the path made honest. Duration presets resolve on the server in calendar terms and
apply as the explicit instants they resolved to - writing them as fixed day counts was caught in
test, because 182 days from the flagship's own anchor is 2026-07-02, so pressing the preset that
described your own window would have moved its boundary and re-addressed the manifest. The domain
menu filters nothing and returns a domain with no declared observation unselectable with the
reason, since an adapter says how a domain is translated and not what is measured, in which units,
in which role or from which record. `POST /path/state` looks for a run at the manifest's content
address and never opens one, because `RunStore.open` publishes a frozen manifest and a read of
where a draft stands must not be the thing that freezes it.

**The first browser test in this repository.** Every other check here reads source or calls HTTP,
and neither proves a page renders. `frontend/playwright.config.ts` serves the real API and the real
frontend and drives Chromium; `frontend/e2e/composer-path.spec.ts` walks the path through roles and
visible names only - no CSS class, no test id - because a test that clicks `.btn-primary` proves the
DOM has a div, not that a person could declare an experiment. The complete flagship is deliberately
**not** what is executed: `order_book` is bespoke and metadata cannot plan its coverage, so the
browser drives into that refusal, resolves it through the visible domain menu, and runs the
resulting three-domain plan.

**Two defects, both found by evidence that did not exist before.**

*   **D79**, found by the browser within minutes. The domain menu's checkbox was bound to the
    server's echo of the selection rather than to the manifest, so it snapped back to its old value
    and for ~200ms reported the **opposite** of the choice just made. Every source-level and HTTP
    test passed throughout, because the manifest and the payload were both correct and nothing in
    this repository rendered anything.
*   **D80**, found by the first full-suite run since TG15. A test asserted that the composer
    contract still called running an experiment unavailable - true at TG17.1, false from TG17.6,
    and the assertion did not fail, it **held the stale claim in place** while the run contract on
    the next router described the state machine that ran it.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests -q -p no:randomly
3106 passed, 2 failed, 4 skipped, 1 xfailed in 2617.52s (0:43:37)
  the two failures are D80 and the stale 2745 test-count claim this run replaced

> .\.venv\Scripts\python.exe -m pytest src\tests\test_composer_path.py src\tests\test_experiment_manifest.py src\tests\test_frontend_contract.py src\tests\test_experiment_run.py src\tests\test_documentation.py -q
304 passed in 354.32s (0:05:54)          (after both fixes; includes the full documentation audit)

> cd frontend && npx playwright test
11 passed (52.2s)                        (Chromium, real API and real frontend)

> cd frontend && npm run build
built in 49.17s                          (TypeScript and Vite production bundle)
```

`test_composer_path.py` 49 tests; `test_frontend_contract.py` 127 -> 144; seven new routes,
119 -> 126. Ledger D1-D80, 78 fixed.

No archive is acquired, no statistic runs, no finding is recorded and no evidence is admitted.

Every status this slice reports is a fact about a declaration.

---

## TG17.8 Scientific comparison views — COMPLETE (2026-08-31, `ed-dev`)

Seven linked views over one manifest, held in a registry ordered by ordinal, each declaring its
axes, its legend, what it may conclude and what it may not. Every earlier G17 slice refuses a bad
**declaration**; this one refuses a bad **picture**, which is harder, because a picture is
persuasive before it is read.

The refusals are structural rather than advisory:

*   `Axis` raises `MagnitudeEquivalenceError` when a `native_magnitude` coordinate is given more
    than one domain — at **construction**, so a view that would put two units on one ruler never
    finishes being built and cannot reach a browser, an export or a screenshot. There is no
    plotting call to police.
*   A coverage cell is a named state (`COVERED`/`SPARSE`/`ABSENT`/`REFUSED`) in the payload *and*
    in the component, which draws from `CELL_STYLE` with no numeric path into it. Absent support
    has no width to be zero, so it cannot be read as a measured zero.
*   `Mark` requires exactly one of an artefact digest and a reason it has none.
*   `register_encoding` refuses a role duplicating another's colour, marker *or* word, so the
    candidate/confirmation distinction survives for a reader who cannot use colour.

**What may be declared is not what may be drawn.** `calendar_aligned` admits `causality` as a
declarable relationship and that stays correct — a study holding an external intervention design
may test it. No view may draw it, because every alignment computed here is observational and the
design that licenses the arrow has no field in the manifest. A manifest declaring `causality`
still gets its matrix cells, occupied by the refusal and its reason.

`ViewContext.results_exist` requires a `MINING/` artefact rather than trusting
`state == "COMPLETE"`: TG17.6 lets a run complete under `partial_permitted` with mining components
missing by name, and without this a run that mined nothing would render its matrix as measured and
empty.

**Found by the guards, not by inspection:**

*   `test_every_api_method_is_reachable_from_the_ui` failed on **four** service methods with no
    caller — the contract, the legend, the single-view render and the reading check were served
    and invisible. They are now the legend, a *What these views will not draw* disclosure, a
    per-view refresh, and a control that asks the server whether a reading can be drawn.
*   Two `<details>` panels were exposed as groups with **no accessible name**, because a
    `<details>` name is not computed from its `<summary>`. Found by role+name locators timing out
    in the browser; fixed with explicit `aria-label`s.
*   One full-suite browser run failed on the motif view where the helper waited only for the first
    view to paint. The helper now waits for the seventh, so each test's assumption that the whole
    set is present is stated once rather than raced on.

```text
> .\.venv\Scripts\python.exe -m pytest src\tests\test_comparison_views.py src\tests\test_frontend_contract.py ^
    src\tests\test_experiment_manifest.py src\tests\test_composer_path.py ^
    src\tests\test_experiment_run.py src\tests\test_documentation.py -q -p no:randomly
387 passed in 341.21s (0:05:41)          (includes the full documentation audit)

> cd frontend && npx playwright test
30 passed (1.9m)                         (Chromium, real API and real frontend; 19 are TG17.8's)

> cd frontend && npm run build
built in 53.63s                          (TypeScript and Vite production bundle)
```

`test_comparison_views.py` 65 test functions (73 runs with parametrisation);
`test_frontend_contract.py` 144 -> 154; six new routes, 126 -> 132. The full suite has **not**
been rerun since TG17.7's 3106, so that figure remains the last measured one rather than a current
one; the documented inventory total is 2822.

No archive is acquired, no statistic runs, no finding is recorded and no evidence is admitted.

---

## TG17.9 Portable receipt, methods report and evidence handoff — COMPLETE (2026-08-31, `ed-dev`)

The completed-run export is now a separate immutable object from TG17.6's live receipt. It seals
the exact manifest, preflight, acquisition/source identities, freeze-time adapter contracts and
environment, role-separated artefacts, inference declaration, event sequence, results, refusals,
methods report and explicit evidence-category census. Registry-derived context is captured when
the run freezes. Legacy runs are exportable only if current re-derivation agrees with their
recorded preflight digest.

Import verifies the outer content address, exact field set, manifest-derived run identity, every
journal transition and step component, the receipt reconstructed from those events, and the
Markdown digest. Recomputing an outer digest after forging the embedded state therefore does not
pass. Replay writes no run, study or EvidenceBundle.

The Composer and Platform & evidence render the same generated capability snapshot. The
documentation guard imports it and requires every registered operation, adapter, refusal and
receipt field to be explained in `architecture.md`. The rendered handoff displays absent
hypothesis/evidence/replication/promotion categories before navigating to a separate study draft;
it makes no evidence call.

**D81.** The first real browser import rejected its own untouched export because Python wrote an
integral JSON number as `1.0` and JavaScript re-emitted it as `1`. Canonical hashing now normalises
integral JSON numbers. The focused regression and the Chromium export/replay path both pass.

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_experiment_receipt.py -q
21 passed in 35.33s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_frontend_contract.py ^
    src/tests/test_experiment_run.py -q -p no:randomly
237 passed in 66.35s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
20 passed                         (verified in the widened targeted run; all documentation checks green)

> cd frontend && npx playwright test
33 passed (2.2m)                  (real API, real frontend, Chromium; 3 are TG17.9)

> cd frontend && npm run build
1,408 modules transformed; production bundle clean

> git diff --check
clean
```

Four new routes bring the served surface from 132 to 136. The inventory is 2,847 test functions.
The full backend suite has not been rerun since TG17.7, so 3,106 remains the last measured full
figure. No live archive is acquired, no scientific statistic is run, no finding is recorded and
no evidence is admitted by TG17.9.

---

## TG17.10 Flagship qualification and no-glue release gate — APPARATUS GATE DELIVERED, RELEASE WITHHELD (2026-09-01, `ed-dev`)

The frozen matrix is three declared durations crossed with the two comparison modes, derived from
the single flagship recipe rather than a second configuration. Each cell keeps explicit UTC
boundaries, the complete-family correction and `duration_selected_before_results`, and binds the
order-book observation by the content digest of its known-answer record. `REFUSED` is a distinct
cell and gate state from `FAIL`, and a refused cell opens no run.

**The matrix does not go green, and that is the measurement.** `order_book.bespoke_record` declares
that it cannot carry `scale_partner_reassignment`, so the four-domain quartet cannot be qualified
in scale/shape mode at all. Three cells are refused before execution with their reason recorded,
rendered and asserted. The Composer answers the same way in the browser rather than switching mode
silently and failing later at execution.

Two gates pass (`offline_matrix` is not one of them; it is `REFUSED`). `browser_no_glue` and
`synthetic_fifth_adapter` are recorded `NOT_RUN` because a backend rehearsal must not award itself
a gate measured by a rendered browser or a source-edit audit, even though both of those pass.
`calendar_calibration` and `live_sources` are `NOT_RUN`; `scale_shape_calibration` is
`NOT_IMPLEMENTED`. `scientist_actions` is `NOT_MEASURED` throughout. The verdict is
`NOT_RELEASEABLE` and `verify_qualification_record` refuses a `RELEASEABLE` verdict carrying any
non-passing gate.

**D83.** TG17.10's first attempt widened the *framework default* `admissible_nulls` in four places
instead of declaring the null per domain, silently overruling the order-book adapter's documented
refusal and pre-admitting the null for any future adapter. Every targeted suite, the production
build and the full 35-test browser suite were green. Only `test_experiment_family.py`'s pinned
per-domain declaration objected, in a full-suite run. Reverted; the three admitting domains
declare it individually with reasons.

**D82.** A held-out confirmation partition is spent exactly once and the frozen flagship ships one
default name, so any plan composed in the browser was executable at most once per deployment and
the refusal told the scientist to declare a partition the form could not declare. The analysis
step now carries an explicit *Held-out confirmation partition* control.

```text
> .\.venv\Scripts\python.exe -m pytest src/tests -q -p no:randomly
3240 passed, 4 skipped, 1 xfailed in 2380.86s (0:39:40)

> .\.venv\Scripts\python.exe -m pytest src/tests/test_experiment_qualification.py ^
    src/tests/test_experiment_family.py -q
77 passed in 431.39s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py ^
    src/tests/test_frontend_contract.py -q
181 passed in 274.25s

> cd frontend && npx playwright test          (after rm -rf .e2e-state)
35 passed (2.7m)                  (real API, real frontend, Chromium; 2 are TG17.10)

> cd frontend && npm run build
1,409 modules transformed; production bundle clean

> git diff --check
clean
```

The clean **3,240** replaces TG17.7's 3,106 as the last measured full-suite figure. Two new routes
bring the served surface from 136 to 138; the inventory is 2,867 test functions. No live archive is
acquired, no scientific statistic is run, no finding is recorded, no evidence is admitted and
nothing is released by TG17.10. Five of the seven registered gates remain unpassed.

---

## T4C.6 acquisition attempted and stopped at D84 (2026-09-01, `ed-dev`)

**The three long-standing external prerequisites are now satisfied.** `cdsapi` 0.7.7 is installed
into `.venv` from `requirements-cds.txt`; a CDS personal access token is configured as
`CDSAPI_URL`/`CDSAPI_KEY` in `.env.local`, which `start_platform.ps1` promotes into the process
environment; the ERA5 licence for `reanalysis-era5-pressure-levels` has been accepted; and
`SPECTRALEARTH_ALLOW_NETWORK` was already set. The campaign preflight returns
`status: READY_FOR_CANARY` with `blockers: []`.

The direct regional route also resolves D43's amplification: the full five-year record is 2.52 GB
across 60 monthly shards against the catalogued route's 29.88 GB, the canary is 18 MB, and the
9.75 GiB total requirement passes against 941 GiB free on `D:`.

The preflight's own claim boundary still holds and is worth restating: `READY_FOR_CANARY` proves
local contract, capacity, dependency, configuration-presence and consent only. It does not prove
credential validity, licence acceptance or service availability. **No live CDS request has been
made, so the token and licence remain unproven.**

**Acquisition stopped before any data transfer, at D84.** Materialising the WeatherBench overlap
was refused on crop geometry. The first attempt supplied no `TransformSupportRequest` and was
correctly refused against the 4-level DTCWT default; that was a caller error, not a defect. Supplying
the campaign's frozen transform (`swt`, `db2`, 3 scales) produced the real refusal: the 161x161
crop retains a 139 px valid interior against `MIN_VALID_INTERIOR` of 128 and passes
`gate_campaign`, while `minimum_crop_size` rounds the 150 px raw minimum up to 256 and refuses.

Reading the estimator settled what the constant is standing in for. `transfer_entropy` consumes
1-D series; `energy_density[t, s] = sum(coefficient**2) / values.size` collapses space to one
scalar per frame per scale. The joint histogram's samples are frames: 4,382/216 = 20.3 per cell on
train and 2,914/216 = 13.5 on test, both above the 5.0 minimum. Crop size beyond edge exclusion
therefore governs the precision of the per-frame scalar, not the sample count -- a power
criterion, not a validity one. It biases toward the null, so it cannot forge a PASS, but it can
forge a FAIL that is really inadequate power, and the frozen decision rule's own FAIL/INVALID
distinction cannot currently be made. Recorded as D84; fix specified as roadmap T4C.5i.

```text
> .\.venv\Scripts\python.exe -m pip install -r requirements-cds.txt
Successfully installed cdsapi-0.7.7 ecmwf-datastores-client-0.5.3 multiurl-0.3.9

> .\.venv\Scripts\python.exe -m src.analysis_engine.gate_campaign review ^
    --campaign campaigns/t4c6_nz_era5_temperature_850_v1.json
campaign_sha256 84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975
network_used false

> .\.venv\Scripts\python.exe -m src.analysis_engine.gate_campaign preflight ...
status READY_FOR_CANARY; blockers []; cdsapi_available true;
credentials.configuration_present true; secret_values_inspected false;
remote_validity_or_licence_acceptance_proven false;
network_consent_enabled true; client_constructed false; network_used false

> materialise(weatherbench_overlap, analysis=swt/db2/3)
FieldTooSmallError: SWT level-3 ... needs at least (256, 256) ... field is (161, 161)
```

Network use in this session was limited to opening the WeatherBench store to read coordinate and
chunk metadata. No measurement values were transferred, no shard was downloaded, no cache was
written and no CDS request was issued. The defect ledger moves to D1-D84 with two open entries,
D43 and D84.

## T4C.5i steps 1-4 -- spatial sampling adequacy (2026-09-01, `ed-dev`)

`src/analysis_engine/spatial_power.py` derives what `MIN_VALID_INTERIOR` was standing in for
(D84): spatial decorrelation and effective sample size per interior, an attenuation curve measured
by nested sub-cropping, an extrapolation to an unlimited crop, and the minimum detectable effect
of the declared design. Steps 5-8 are not started, so D84 stays open.

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spatial_power.py -q
41 passed, 1 warning in 4.64s                     (36 test functions)

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spatial_power.py ^
    src/tests/test_cross_scale.py src/tests/test_scale_signature.py ^
    src/tests/test_statistics.py src/tests/test_gate_campaign.py ^
    src/tests/test_gate_run.py src/tests/test_preregistration.py -q
196 passed, 1 warning in 54.55s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
20 passed, 2 warnings in 252.38s

> git diff --check
(clean)
```

For the frozen campaign -- 36 tests, Benjamini-Yekutieli at alpha 0.05, 4,999 circular shifts --
the derived detection rank is `k = 0`: the observation must exceed **every** surrogate. The
p-value floor of 1/5000 clears the required raw level of 3.327e-4 by a factor of 1.66, so the
design detects, but with no margin for a single exceedance. A test pins that boundary to the
gate's own `screen` over the declared family, and the rejection flips across it and nowhere else.

No confidence interval is attached to the minimum detectable effect, and two attempts to attach
one are recorded in `roadmap.md` as errors rather than removed silently. The surrogate seed is
preregistered, so the ensemble is frozen and the order statistic is the decision boundary itself
rather than an estimate of it. The measured miscalibration that exposed this: at rank 1 a
bootstrap resample can never exceed the sample maximum, and four independent ensembles of 4,999
draws all fell above the interval's upper limit; at rank 10, coverage was 3 in 20.

The full backend suite has **not** been rerun since these additions, so 3106 remains the last
measured full-suite figure. No network was used. The defect ledger is unchanged at D1-D84 with
D43 and D84 open.

## T4C.5i step 5 -- refusing on the derived quantity, and defect D85 (2026-09-01, `ed-dev`)

Step 5 replaces the crop-size constant in the refusal path with the derived quantities from steps
1-4. `spatial_power_refusal` returns one verdict and, when it is `INVALID`, names the deficit in
the units of what caused it plus which of `REMEDY_AXES` -- crop size, frame count, scale count --
would close it. `crop_for_effect` inverts the attenuation fit already reported rather than adding a
second model, and refuses to name a crop where a number would be an invention: a target above the
unlimited-crop ceiling, a saturated largest sub-crop, or an area exponent showing the decorrelation
length still growing with the window. `family_for_effect` reports the family that would have
detected the effect and marks it `admissible_after_seeing_data: False`.

**Defect D85, found while wiring this and confirmed against the checked-in preregistration.**
`_shift_null` draws its circular shifts with replacement from `admissible_shifts`, so requesting
4,999 surrogates always returns 4,999 numbers. The exact test's reference set is the distinct
admissible shifts the record contains, and its attainable p-value is bounded by `1 / (1 + D)`
however many draws are taken. Measured on the frozen campaign's own split:

```
$ .venv/Scripts/python.exe -c "... frames_for_resolution ..."
train 4382 frames, theiler 24: 4329 distinct shifts, attainable p 2.309e-4, required 3.327e-4, resolves
test  2914 frames, theiler 24: 2861 distinct shifts, attainable p 3.494e-4, required 3.327e-4, does not
test  2914 frames, theiler  1: 2912 distinct shifts, attainable p 3.433e-4, required 3.327e-4, does not
      -> about 3007 frames would supply the 3005 distinct shifts required
```

Confirmed independently through the repository's own screening machinery rather than this module's
arithmetic, by feeding the best attainable p-value of each partition into `screen` over the
declared 36-test family:

```
train theiler 1  D 4380 p 0.00022826 q 0.034304 sig True
train theiler 24 D 4329 p 0.00023095 q 0.034708 sig True
test  theiler 1  D 2912 p 0.00034329 q 0.051591 sig False
test  theiler 24 D 2861 p 0.00034941 q 0.052510 sig False
```

**The frozen T4C.6 campaign cannot replicate on its confirmatory half at any effect size**, and
`check_power` reports it as adequately powered because it counts the 4,999 requested draws against
the 3,005 required rather than counting the reference set. The margin is small -- q = 0.0516
against alpha = 0.05 -- and on the wrong side. The campaign file is left frozen and unedited;
re-freezing it is the recorded supersession specified as T4C.5i step 8, not a repair, and D85 is
recorded as open.

**Wiring.** `review_gate_campaign` reports the audit as `scientific_design.surrogate_resolution`
with a `resolvable` flag; `preflight_gate_campaign` refuses on it before any transfer; and
`preflight_cached_gate` refuses on it for the `real_era5_gate` role. The order is deliberate: a
frozen campaign that cannot resolve its own family must stay loadable and reviewable, or the defect
could not be recorded against it. The audit uses the most favourable Theiler window of one frame,
so a partition that fails it cannot be rescued by any window, which is what makes refusing before
acquisition safe. The `test_gate_campaign.py` orchestration fixture was lengthened from 120 to 200
frames -- its 47-frame confirmatory partition genuinely could not resolve its own two-test family,
and a fixture is not evidence.

**Step 5 is not finished.** The attenuation half of the adjudication -- converting a FAIL into an
INVALID inside `run_cached_gate` -- needs the sub-cropped attenuation curve, which means
re-decomposing the train partition at six crop sizes. That is the same computation step 7's receipt
fields need, and the two are built together rather than twice.

Commands and results:

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_spatial_power.py -q
58 passed, 1 warning in 4.82s

$ .venv/Scripts/python.exe -m pytest src/tests/test_spatial_power.py src/tests/test_cross_scale.py \
    src/tests/test_scale_signature.py src/tests/test_statistics.py src/tests/test_gate_campaign.py \
    src/tests/test_gate_run.py src/tests/test_preregistration.py -q
215 passed, 1 warning in 50.73s

$ .venv/Scripts/python.exe -m pytest src/tests/test_lag_policy_registry.py -q
21 passed, 1 warning in 4.79s
```

No network was used and no data was acquired. The full backend suite has **not** been rerun since
these additions, so **3106** remains the last measured full-suite figure. The defect ledger moves
to D1-D85 with three open entries: D43, D84 and now D85.

## T4C.5i step 6 -- demoting the two constants (2026-09-01, `ed-dev`)

Step 6 leaves `MIN_VALID_INTERIOR` and `RECOMMENDED_VALID_PARENT_SIDE` in place, and leaves both
at 128. What changed is their standing and their reach. Each now declares, in code and in every
payload that reports it, that it is a judgement about how much uncontaminated span makes a spatial
statistic comfortable to look at rather than a derived power criterion, and each names
`analysis_engine/spatial_power.py` as the thing that answers the question it stood in for. The
power-of-two rounding left the refusal path entirely and is reported separately as an operational
convention that nothing is refused on.

Measured thresholds before and after:

```
$ .venv/Scripts/python.exe -c "... minimum_crop_size / dyadic_crop_size ..."
levels  refusal threshold  dyadic convention   (was: refused on the dyadic figure)
  1            142               256
  2            168               256
  3            220               256
  4            324               512
  5            532              1024

$ .venv/Scripts/python.exe -c "... crop_planner.assess_shape ..."
swt   db2 level 4 : requirement 174, dyadic 256
dtcwt     level 4 : requirement 352, dyadic 512
```

**What this closes, and what it does not.** D84's *contradiction* is closed. The frozen T4C.6 crop
is 161 px at db2 SWT level 3; the accumulated support contaminates 11 px per side, leaving a 139 px
valid interior, so `gate_campaign` admitted it against the 128 px heuristic while the planner
raised its own raw 150 px requirement to 256 and refused the same crop -- describing 256 to the
caller as *statistically recommended*. With the rounding gone the planner's threshold is 150 px,
both gates admit the crop, and that is pinned as a test against the defect's own case:

```
$ .venv/Scripts/python.exe -c "... assess_shape(161, 161, swt/db2/level 3) ..."
valid interior 139x139, requirement 150, dyadic 256, verdict recommended, meets True
```

What is **not** closed is whether 139 px of interior is *enough*. That is a power question, the
derived quantities exist in `spatial_power.py`, and the FAIL/INVALID adjudication that consumes
them is step 5's deferred attenuation half. **D84 remains open**, now on the adjudication rather
than on the disagreement, and D85 is untouched by this step.

The refusal wording was corrected throughout: `FieldTooSmallError` now reads *R13 heuristic
interior*, states that the threshold is a judgement rather than a derivation, and points at the
derived criterion. The API catalogue reports `r13_minimum_crop` as the requirement and adds
`r13_dyadic_operational_crop` for the rounding, with `r13_legacy_note` saying which is which. The
acquisition UI labels the tile *Heuristic minimum*, shows the dyadic convention beneath it marked
*not gated on*, renders the threshold's own `limitation` text, and no longer calls a crop that
clears it *scientifically recommended*.

Commands and results:

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_crop_planner.py src/tests/test_zarr_source.py -q
72 passed, 1 skipped, 5 warnings in 25.79s

$ .venv/Scripts/python.exe -m pytest src/tests/test_frontend_contract.py -q
161 passed, 5 warnings in 8.66s

$ .venv/Scripts/python.exe -m pytest src/tests/test_cds_source.py src/tests/test_imports.py \
    src/tests/test_gate_run.py src/tests/test_gate_campaign.py src/tests/test_spatial_power.py -q
136 passed, 2 warnings in 1275.76s

$ .venv/Scripts/python.exe -m pytest src/tests/test_documentation.py -q
20 passed, 2 warnings in 315.70s

$ npm run build            # frontend
1409 modules transformed, built in 1m 23s
```

No network was used and no data was acquired. The full backend suite has **not** been rerun since
these changes, so **3106** remains the last measured full-suite figure. The ledger is unchanged at
D1-D85 with three open entries: D43, D84 and D85.

## T4C.5i step 7 -- publishing the derivation, and the FAIL/INVALID boundary (2026-09-01, `ed-dev`)

Steps 1-4 produced derived spatial-power quantities that nothing consulted. Step 7 runs them
inside `run_cached_gate`, publishes every one of them in the receipt, and lets them decide whether
an absence is a negative finding -- which is also step 5's deferred attenuation half, deferred
precisely because it needed this curve.

`run_cached_gate` now takes one further bounded pass over the train partition. The receipt gains
a `spatial_power` block -- per-scale median decorrelation length per axis and effective sample
count, the attenuation curve over concentric sub-crops, the sweep-derived minimum detectable
effect, the `spatial_power_refusal` record with its remedy axes, and a `limitations` list -- and a
`power_adjudication` block naming the rule that produced the scientific verdict. The gate's own
verdict is left exactly as `evaluate_replication_gate` returned it, so the receipt records the
protocol decision and the power decision separately rather than presenting one as the other.

**The boundary.** In the real gate role a FAIL survives as a negative finding only where the
derived record returns ADEQUATE. Where it returns INVALID, could not be measured, or could not
reproduce the sweep's surrogate ensemble, the run is INVALID and names its deficit and remedy
axis. A PASS is never downgraded and the receipt says why: spatial imprecision attenuates toward
the null, so an undersized crop cannot manufacture a positive -- only an absence that belongs to
the instrument. Under synthetic acceptance the block is computed and published in full and
adjudicates nothing.

**Three things recorded in the receipt rather than in a docstring.**

*   The audited test is the **train** partition's smallest-p case. Selecting what to audit after
    seeing the held-out result is the move the split exists to prevent.
*   Where two scales lose different margins to the same filter, the curve is measured on the
    window both interiors can supply, so its largest row is *not* the sweep's own estimate. Both
    numbers are published and neither is adjusted into the other; the adjustment between them
    would be a correction nothing measured. On the synthetic fixture the interiors are 60 and 54
    px and the matched window is 54.
*   A decimated family's shared sub-crop size is a coefficient-count match, not a shared area.
    SWT -- which the frozen T4C.6 campaign uses -- is undecimated and unaffected, and the record
    says so for the families that are.

**Two seams added rather than shortcuts taken.** `cross_scale.shift_null_ensemble` and
`surrogate_seed` let the audit reconstruct the ensemble the sweep actually used: the minimum
detectable effect is an order statistic, the sweep keeps only that ensemble's summary, and an
audit that reseeded would be characterising a different null and reporting it as this study's
decision boundary. The receipt records whether the reconstruction matched the published summary,
and a mismatch is INVALID rather than a quiet threshold. `spatial_power.StreamedAttenuation`
builds the curve one frame at a time, because the array form would need 677 MB per scale on the
frozen crop before the orientations are counted; a test asserts it **equals** `attenuation_curve`
rather than approximating it.

Commands and results:

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_gate_run.py -q
6 passed, 1 warning in 13.34s

$ .venv/Scripts/python.exe -m pytest src/tests/test_spatial_power.py -q
63 passed, 1 warning in 5.47s

$ .venv/Scripts/python.exe -m pytest src/tests/test_cross_scale.py src/tests/test_spatial_power.py \
    src/tests/test_gate_run.py src/tests/test_gate_campaign.py src/tests/test_imports.py \
    src/tests/test_preregistration.py -q
199 passed, 2 warnings in 1182.67s

$ .venv/Scripts/python.exe -m pytest src/tests/test_gate_campaign.py src/tests/test_preregistration.py \
    src/tests/test_frontend_contract.py src/tests/test_imports.py src/tests/test_scale_signature.py -q
298 passed, 5 warnings in 796.84s

$ .venv/Scripts/python.exe -m pytest src/tests/test_documentation.py -q
20 passed, 2 warnings in 258.62s
```

The 298-test run predates the `limitations` field and the module-docstring change; the 199-test
run and the documentation audit were both taken against the final tree.

**What this does not establish.** No real gate has run. No atmospheric absence has been adjudicated
by the new boundary, and no measured attenuation curve for the frozen 161 px ERA5 crop exists --
every branch is exercised on the synthetic acceptance fixture and on unit-level records. D84's
remaining half, whether 139 px of valid interior is *enough*, now has an apparatus that will answer
it rather than an unanswered question, but the answer waits on acquisition, which step 8's campaign
supersession and D85 still block.

No network was used and no data was acquired. The full backend suite has **not** been rerun since
these changes, so **3106** remains the last measured full-suite figure. The ledger is unchanged at
D1-D85 with three open entries: D43, D84 and D85. No frontend change was needed: gate receipts are
not surfaced by the API or the UI, so the receipt is the artefact this step delivers.

## T4C.5i step 8 -- retiring a frozen campaign by checked supersession (2026-09-01, `ed-dev`)

D85 established that `t4c6_nz_era5_temperature_850_v1` could not resolve its own declared family
on its confirmatory partition. A frozen design that cannot reach its own decision cannot simply
be corrected in place: editing it destroys the record that the original rule existed, and leaves
a reader unable to tell a correction from a result-driven revision. Step 8 records the retirement
as a third immutable artifact instead.

`CampaignSupersession` names both campaigns by content hash and states its reasons as **checks**.
Recording it runs every check against both campaigns: a reason is admissible only where the
superseded campaign genuinely fails it and the successor genuinely passes. Properties the
predecessor already held are declared under `preserved` and must hold for **both**. A reason
naming a check the registry does not implement is refused outright. The consequence is that the
record cannot be written for a defect that was not real, cannot claim a repair that did not
happen, cannot be a rename, and cannot quietly drop a design property it was not repairing.

The successor `t4c6-nz-era5-temperature-850-campaign-v2` extends the record to six whole calendar
years, 2018--2023: 8,764 frames, a 5,258/3,498 split, 3,496 distinct admissible shifts against
the 3,005 required. Crop, variable, level, transform, the family of 36, lags, embargo, seed,
correction, canary and WeatherBench overlap are unchanged, and a test asserts that rather than
trusting the diff.

**The minimal repair was refused by the record's own checks.** This is the substantive finding of
the step. `frames_for_resolution` reported that 3,007 confirmatory frames would close D85, and it
was right about the question it was asked -- it audits the most favourable Theiler window of one
frame, because the window is derived from a series that does not exist before acquisition. The
sweep sets that window from the measured temporal decorrelation of the series under test. Measured
here:

```
record                              frames  train  test   max Theiler window still resolving
2022-12-31 (superseded v1)            7304   4382  2914   -- (does not resolve at all)
2023-02-27 (the minimal repair)       7536   4521  3007   1
2023-12-31 (successor v2)             8764   5258  3498   245
```

A design built to the reported number resolves at a window of one frame and nothing larger, so it
would have reproduced D85 at run time after the 2.8 GB transfer rather than before it. The
`resolution_margin` check states the margin as an admissibility condition against a declared floor
of 16 frames -- four days at this cadence -- and a test pins that the minimal design is refused by
it. The check is a *designed margin, not a prediction*: the guarantee remains the run-time refusal,
which re-runs the resolution audit at the measured window and returns INVALID.

The supersession bites at the **acquisition** boundary, not the reading boundary.
`review_gate_campaign` still loads the retired v1 and still reports its defect -- otherwise the
defect could not be recorded against the artifact it belongs to -- while
`preflight_gate_campaign(..., supersessions=[...])` and `gate_campaign preflight --supersession`
refuse to spend on it. `gate_campaign review-supersession` re-runs every check against both
campaigns and prints the outcomes side by side.

Commands and results:

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_gate_campaign.py -q
19 passed, 1 warning in 7.46s

$ .venv/Scripts/python.exe -m src.analysis_engine.gate_campaign review-supersession \
    --supersession campaigns/t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json
  reasons: surrogate_resolution   superseded=False successor=True
           resolution_margin(16)  superseded=False successor=True
  preserved: whole_annual_cycles  superseded=True  successor=True
  deferred_to_run: D84, D85, D43
  network_used: false
```

Artifact hashes, published immutably and pinned by test:

```
superseded campaign   84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975
successor campaign    c66284d619d7439638ec5e1886671894df4d12708ab3cdced245d7e5f80fa23c
supersession record   054592339d99141a1560b8fbecad6daaf02844a9813513331b982722f10e6711
```

**What this does not establish**, as the record's own `deferred_to_run` block says and the review
republishes. **D84 is not closed**: the crop is carried through unchanged, and whether 139 px of
valid interior suffices is a question about a field that does not exist, left to step 7's
`power_adjudication` at run time. **D85 is not closed**: the declared margin is a design decision,
not a measurement of the window the sweep will derive. **D43 is untouched**: no data has been
acquired for either campaign, no network was used, and the successor is a design rather than a
record. The successor's own preflight on this machine reports BLOCKED on the absent `cdsapi`
dependency, absent CDS credential configuration and disabled network consent -- unchanged from
v1, and unrelated to the supersession. A supersession is not a result and does not by itself
license the successor's acquisition.

The full backend suite has **not** been rerun, so **3106** remains the last measured full-suite
figure. The ledger is unchanged at D1-D85 with three open entries: D43, D84 and D85. **No part
of this line is surfaced in the API or the UI:** there is no gate or campaign route, no client
method and no component that reads a campaign, a supersession or a gate receipt. The T4C
artifacts are CLI-and-file only, and the frontend's preregistration and evidence panels belong
to the separate cross-domain line in `roadmap_cross_domain.md`.

## T4C.5j -- the gate record served read-only (2026-09-01, `ed-dev`)

The T4C line had no API route, no client method and no component. Every campaign, supersession
and receipt was CLI-and-file only, so the two distinctions the line exists to draw -- a retired
design that must not be acquired, and an inadequately powered absence that is INVALID rather
than FAIL -- could be checked only by knowing which file to open. `src/api/gate.py` serves seven
GET routes and `frontend/src/components/GateRecordView.tsx` renders them under Review.

The surface is read-only as a property of the routing table: a test collects every method served
under `/api/v1/gate` and asserts the set is exactly `{"GET"}`. There is no preflight route and no
acquisition route, and the four refusals are served as data and rendered, so a reader who cannot
find an acquire button is told why rather than left to infer the apparatus is unfinished.

Retirement is derived from content. A campaign is RETIRED here if and only if a supersession in
the store names it by fingerprint -- the same comparison `preflight_gate_campaign` refuses on --
so the surface and the spend agree by construction. A test installs all three artifacts under
different file names and asserts the retirement survives; another installs v1 and v2 with no
supersession and asserts both are ACTIVE while v1 still reports `resolvable: false`, because not
being retired is not being sound. A tampered envelope is listed as unreadable rather than
dropped, and the surface summary reports it.

Commands and results:

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_gate_api.py -q
13 passed, 2 warnings in 10.05s

$ .venv/Scripts/python.exe -m pytest src/tests/test_frontend_contract.py -q
167 passed, 5 warnings in 7.23s

$ .venv/Scripts/python.exe -m pytest src/tests/test_documentation.py -q
20 passed, 2 warnings in 253.77s

$ cd frontend && npm run build
tsc clean; 1410 modules transformed; built in 1m 11s
```

Served against the repository's own store, `GET /api/v1/gate` reports 2 campaigns, 1
supersession, 1 retired campaign, 0 receipts and `measurement_status: NOT_YET_MEASURED`;
`/gate/campaigns` labels `...-v1` RETIRED with `resolvable: false` and `...-v2` ACTIVE with
`resolvable: true`; `/gate/supersessions/...v1-to-v2` re-runs both reasons and reports
`superseded=false successor=true` for each, with `deferred_to_run` carrying D84, D85 and D43.

**What this does not establish.** It adds no science and closes no defect. D43, D84 and D85
remain open and unchanged. Nothing here reads a field or touches a network. **The receipt route
has never served a real receipt, because no gate has run**; the receipt tests use a fabricated
transport fixture that authenticates, which is a test of what the route shows a reviewer and is
not evidence about the atmosphere. The panel compiles and builds; as with every surface after
T3.5.25, its **rendered appearance has not been inspected in a browser** and no screenshot
exists in this repository. The full backend suite has not been rerun, so **3106** remains the
last measured full-suite figure; the inventory total is now 2967 by AST count.

## T4C.5k -- the first live ERA5 acquisition, stopped at the cross-route gate (2026-09-01, ed-dev)

The mandatory acquisition order was executed live against campaign v2 for the first time. Steps
1 and 2 passed; **step 3 failed and stopped the campaign**, so the 3.03 GB full record was never
requested. Every figure below was measured on this machine on this date.

**Preflight (zero network).** `python -m src.analysis_engine.gate_campaign preflight --campaign
campaigns/t4c6_nz_era5_temperature_850_v2.json --supersession
campaigns/t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json ...` returned
`status: READY_FOR_CANARY`, rc 0, `blockers: []`, `network_used: false`,
`client_constructed: false`. Credentials were reported present from the
`CDSAPI_URL`/`CDSAPI_KEY` environment with `secret_values_inspected: false`; no secret value was
read or displayed at any point in this work. Storage: 6.11 GB working requirement against 1,012
GB free on `D:`. Full record upper bound 3,025,332,704 bytes over 72 monthly shards; canary
18,436,160 bytes over 1.

**Step 0, the recorded probe.** `store_probe.probe_store` opened
`gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr` with `chunks={}` and
transferred no values. Recorded as digest `1a28d5980c97a38e` in `data/store_probes/`. Chunks are
`(1, 13, 721, 1440)` float32, 53,988,480 bytes each; 93,544 time steps. Amplification for the
eight-frame overlap crop: **520.7x**. This confirms the original D43 diagnosis at the
0.25-degree store, and by the same arithmetic the 8,764-frame record would cost roughly 473 GB
through this route -- which is why the record comes from CDS.

**Step 1, `materialise_weatherbench_overlap`.** 8 frames, 1 level, 161x161. Transferred
**237,476,444 bytes in 61.9 s** for 829,472 bytes of wanted values -- 286x realised against the
520.7x chunk-arithmetic bound, the difference being compression. Content hash
`fb7f6a261ae3bb1941c3b28eb6b892c2`, cache key `a5ff06f93a0e8f6f`.

**Step 2, `materialise_cds_canary`.** **The first live CDS request in this programme's
history.** Request ID `34e8fd4c-e007-4d16-9d95-e9289f057c84`, accepted then successful in 26 s
of queue, 339 KB downloaded, 45.4 s wall total. Content hash
`417c005f9355e1d7a42f9301e44285e6`, cache key `a4f2304e05985fd7`, shape
`{time: 8, latitude: 161, level: 1, longitude: 161}`. This establishes that the credential
configuration, licence acceptance, download, shard integrity check, NetCDF normalisation and
canonical Zarr conversion all work end to end against the live service. It establishes nothing
about the atmosphere: it is two days of one variable.

**Step 3, `require_canary_overlap_PASS_or_stop` -- FAILED, campaign stopped.** Receipt
`5acf466052195974adc21637724cc47984c3843919e0ae3904efe22b5272224d`, written immutably to
`data/era5_cache/a4f2304e05985fd7.overlap.json`, records `passed: false` with
`coordinates_exact: true`, `units_compatible: true`, `network_used: false` and
`maximum_frames_per_source_in_memory: 8`. For `t`: `max_abs_error` 7.324e-4 K, `rmse` 3.399e-4,
`mean_abs_error` 2.856e-4, `mismatch_count` 62,163 of 207,368 against `atol` 1e-4 and `rtol`
1e-6. **The full acquisition was not started.**

**Why it failed, tested rather than assumed.** The CDS route delivers 850 hPa temperature
quantised to 2.44140625e-4 K (2^-12); the WeatherBench route to 3.0517578125e-5 K (2^-15).
Neither file carries `scale_factor`/`add_offset`; both are float32. The declared 1e-4 K
tolerance is therefore 0.41 of the primary route's own representation step. The difference
between the routes is commensurate with the primary route's encoding rather than with the
atmosphere, on four independent checks: it is bounded at **exactly 3.000 CDS quanta**; its mean is flat across latitude (-0.620 to -0.454 quanta) and
longitude (-0.618 to -0.425); it is uncorrelated with the field value (-0.0002) and with the
spatial gradient (0.0008), where regridding or interpolation error would track the gradient
strongly; and registration is exact, a one-cell roll raising the maximum difference from 7.3e-4
K to 12.6 K in longitude or 30.8 K in latitude. The residual -0.533 quantum mean bias varies
per frame from -0.133 to -0.883, which is the signature of per-field GRIB packing.

**The magnitude is not fully explained, and that is recorded rather than smoothed over.**
Co-quantising a single underlying value at the two observed steps bounds the difference at
`q_cds/2 + q_wb/2` = 1.37e-4 K; adding the per-frame offset reaches roughly 1.5 quanta.
The observation is 3.000. Misregistration and interpolation are excluded and the scale is
set by the CDS encoding, but about a factor of two remains unaccounted for. This is a
second reason not to pick a constant now: a tolerance chosen to clear 7.324e-4 K would
also clear whatever is producing the unexplained part.

**Correction to the figures first published above.** The "3.000 quanta" and the "factor of
about two unexplained" were artefacts of pooling eight frames that are on *different* lattices.
Each CDS frame is packed separately: frames 0, 1, 4 and 5 of the canary sit on 2^-10 K and
frames 2, 3, 6 and 7 on 2^-9 K, each exactly. Measured against its own frame's step the worst
disagreement is **0.72 steps**, not 3.0, and nothing is unaccounted for. The pooled 2^-12 K
"quantum" was the interference pattern between two lattices, not a step either route uses.

**The independent window, acquired to derive a criterion without tuning it to its own test.**
A second canary at 2019-06-01/02 -- different season, different year, outside the campaign, in
its own directories -- cost 341,999 bytes from CDS and 237,738,102 bytes from WeatherBench. All
eight of its frames sit on 2^-9 K, and its signed error lies wholly within +/-0.5 steps
(min -0.4375, max +0.3125): exact round-to-nearest re-quantisation of the same numbers, with no
residual at all. It disagrees *more* in Kelvin than the gate window does -- 8.545e-4 against
7.324e-4 -- while being the cleaner of the two, which is the clearest available demonstration
that Kelvin is the wrong unit for this question. Both windows are `expver 0001`, so this is
final ERA5 against final ERA5, not ERA5 against ERA5T.

**Recorded as D86, and repaired by changing the unit rather than the number.** Widening `DEFAULT_ATOL` to admit
7.324e-4 K would have tuned the criterion to the first measurement it was ever asked to judge,
making the independent-route check ornamental. The repair was a criterion derived from the two
routes' declared encodings and frozen into the campaign envelope -- which is the second half of
D86, since the constant lived in module code and could therefore be changed without a
supersession, outside the preregistration discipline that governs everything else in this line.
Both halves were carried out in the continuation section below.

**What this does not establish.** No multi-year record was acquired. No crop was frozen. T4C.6
has not run and no gate verdict exists. D43 is **not closed**: its acquisition route is now
proven live for an eight-frame canary, but the multi-year record and its cross-route check
remain unrun. D84 and D85 are
untouched. No test suite was rerun as part of this work, so **3106** remains the last measured
full-suite figure.


## T4C.5k continued -- the encoding-relative criterion and campaign v3 (2026-09-02, ed-dev)

**The criterion.** `era5_overlap.encoding_step` returns the largest binary step on which every
value of a frame lies exactly, or `None`. `verify_cached_era5_overlap` gains
`criterion="encoding_relative"` with `steps_allowed`, judging each value against its own
frame's step. Receipts are criterion-keyed, so the earlier FAIL is preserved beside the PASS
rather than overwritten: a cache is a fact about what the archive returned, a verdict is a
judgement under a stated rule.

**A refusal that could never fire, found by testing it.** The first `encoding_step` accepted any
binary lattice. Every float32 value lies on one -- the one its own exponent defines -- so the
function would always succeed and would judge an unpacked route against its representation
error. It now requires the step to be at least two bits coarser than float32's spacing in that
range. Measured against the real caches: the CDS frames report 2^-10 and 2^-9; **both
WeatherBench windows report `None`**, correctly, because their values sit exactly at the float32
ulp. The criterion is therefore applied to the primary route only.

**Measured verdicts under the new criterion**, on caches already on disk:

| window | worst error | steps allowed | verdict |
| --- | --- | --- | --- |
| gate 2018-01-01/02 | 0.71875 steps | 1.0 | PASS, `mismatch_count` 0 |
| independent 2019-06-01/02 | 0.4375 steps | 1.0 | PASS, `mismatch_count` 0 |

Receipt `0ccf02243ef9b93c1fae7afb384ed4012b2d1127936e891b7b979bf3a6424de9`, schema
`era5-independent-overlap/v2`. The manifest now records `independent_overlap_check` FAIL and
`independent_overlap_check_encoding_relative` PASS side by side.

**Campaign v3, frozen as a checked supersession of v2.** Campaign sha256
`1c96e49b6717bf0d99da9f5048cea84f7a9af28b60d9c0c173a93056c0a0e24b`; supersession sha256
`cdf49d81bb8d9b2b6cc4bf3f46ecd4cd2310d93206a724fef047ac7887363905`. Its single reason cites D86
and re-runs `overlap_criterion_declared` against both designs -- v2 fails it, v3 passes. v2's
six whole calendar years and its D85 repair are recorded as preserved; D84, D85 and D43 are
carried as deferred to the run. **v1 and v2 keep the exact fingerprints they were sealed under**
(`84f7b53f...` and `c66284d6...`), verified after the schema change and pinned by test, because
the optional field is omitted from the mapping when absent.

**Preflight.** `preflight` on v3 with both supersessions returns `READY_FOR_CANARY`, rc 0,
`blockers: []`, with `overlap_criterion` `{name: encoding_relative, steps_allowed: 1.0}` and
`canary_cache_present: true`, `weatherbench_overlap_cache_present: true`,
`full_cache_present: false`. Preflight on **v2 is now refused outright** as retired by the new
supersession. A campaign with no declared criterion reports `BLOCKED` naming D86.

**Tests.** `test_cds_source.py` 14 -> 20 functions (25 collected, all passing);
`test_gate_campaign.py` 17 -> 22 (24 collected); `test_frontend_contract.py` 167 -> 168.
Final combined run against the tree as it stands, of `test_gate_api.py test_gate_campaign.py
test_cds_source.py test_frontend_contract.py test_regional_forecast.py test_imports.py`:
**291 passed** in 793.26 s, exit code 0. `test_documentation.py` **20 passed** in 246.37 s
against the same tree (an earlier documentation run of the same 20 took 274.63 s; the 246.37 s
is the one that judged the tree as it now stands). `git diff --check` clean. Frontend production build clean at **1,410
modules**. Inventory total 2967 -> 2979.

One earlier combined run reported 297 passed, but it was started before the residual-tolerance
fix and the magnitude test were added and therefore describes a tree that no longer exists; the
291 above is the figure for the current one.

**What this does not establish.** The 8,764-frame record has not been acquired; the full-cache
overlap has not run; no crop has been frozen against real data; T4C.6 has not executed and no
gate verdict exists. The canary is two days of one variable at one level, and a criterion
validated on two two-day windows is validated on two two-day windows. D43, D84 and D85 remain
open. The gate panel's new column compiles and builds but its **rendered appearance has not been
inspected in a browser**. The full backend suite has not been rerun, so **3106** remains the
last measured full-suite figure.

## T4C.5m -- the multi-year record, and the first gate verdict (D43 closed; D87 found and fixed)

**Preflight before spending.** `gate_campaign preflight` on campaign v3 with the v2 supersession
supplied: `status READY_FOR_CANARY`, `blockers []`, `overlap_criterion {name:
encoding_relative, steps_allowed: 1.0}`, `campaign_sha256
1c96e49b6717bf0d99da9f5048cea84f7a9af28b60d9c0c173a93056c0a0e24b`, exit code 0. The step-3
receipt on disk reads `passed: true` at **0.71875** steps with zero mismatches over 207,368
values.

**Step 4, `materialise_full_cds_record`.** 72/72 monthly CDS shards. `content_key
a07c23ec89f953c1`, `content_hash e488f5c3d480f072c834dceae1eeea2a`, `shape {time: 8764, level:
1, latitude: 161, longitude: 161}`, `bytes_transferred 338,905,211` (**338.905 MB**),
`wall_seconds 5,853.7`, `cache_hit false`, exit code 0. Against a 3,025,332,704-byte preflight
upper bound, which assumes float32 x2 with no compression credit; the checked quantity is the
frame count, verified exactly against the complete expected calendar at conversion.

**Step 5, `require_full_cache_overlap_PASS_or_stop`.** `passed True`, `max_error_in_steps
0.71875` against `steps_allowed 1.0`, `encoding_steps [2^-10, 2^-9]`, `max_abs_error
7.32421875e-4`, `rmse 3.398889796464777e-4`, `mismatch_count 0`, `value_count 207,368`,
`receipt_sha256 bdfd9c8a00a41d4b3dcd95c244cedd17c1078f675ef758f04fa2a2e0c494b0dc`. Identical to
the canary's figure because the record's first eight frames come through the same route; this
establishes the 72-shard concatenation, not a new fact about agreement.

**D87, found by running it.** Step 6 refused the record: `independent WeatherBench overlap
evidence is not a recorded PASS`. The manifest holds `independent_overlap_check_encoding_relative
= PASS` and `independent_overlap_check = NOT RUN`, and `validate_overlap_evidence` read only the
unsuffixed fields. Fixing that surfaced a second layer -- `CachedFieldReader.source_provenance`
hardcoded the same three names, so the evidence never reached the gate. Both fixed. Checked in
both directions on the real manifest before re-running: admitted under `encoding_relative` at
0.71875 steps, and **refused** under `absolute` with `not a recorded PASS under the absolute
criterion`.

**Step 6, `preflight_and_run_T4C.6`.** Run through `run_campaign_gate`, so the criterion came
from the frozen envelope. `campaign 1c96e49b`, `plan dd9fc47c25df1b32ca18e6c01873421854063d5add
ddb977d1026b66f1b7d2f5`, `crop a07c23ec89f953c1`, `criterion encoding_relative`.
**`scientific_verdict PASS`**, `gate.verdict PASS`, `problems []`, **10 replicated links**,
`wall_seconds 753.9`. Receipt at `data/gate_receipts/t4c6-nz-era5-temperature-850-v3.json`
(128 KB), `schema cross-scale-gate-receipt/v1`, with `preflight.overlap_criterion
encoding_relative`. Representative link `2->1@3` (18 h): train observed 0.021457982912114204
nats against surrogate mean 0.0019260827780943593, effect size 10.814231699653089, p 0.0006, q
0.015028413108460698; test observed 0.018884626612145183 against 0.0035527403383150834, effect
size 5.457797843730858, p 0.0012, q 0.02576299390021834; 4,999 surrogates, Theiler window 6,
Benjamini-Yekutieli. `power_adjudication` `power_applied false`: *"a PASS verdict is not an
absence, and spatial imprecision biases toward the null, so the derived power record cannot
overturn it."*

**A tripwire fired as designed.** `test_the_repository_store_serves_the_real_campaigns_and_their_retirement`
asserted `measurement_status == "NOT_YET_MEASURED"` with the note *"if this ever fails, a
receipt exists and the docs must say so"*. It failed. Updated to `MEASURED`, and the test now
also asserts the receipt is served from the checked-in store and that its `plan_sha256` binds to
campaign v3's plan.

**Tests.** `test_cds_source.py` 20 -> 22 functions; `test_gate_campaign.py` 22 -> 25.
`test_gate_campaign.py` **27 passed** in 15.26 s. `test_gate_api.py` **13 passed** in 17.39 s
after the tripwire update. Combined `test_cds_source.py test_gate_run.py test_zarr_source.py
test_gate_api.py`: **106 passed, 1 skipped, 1 failed** in 66.12 s -- the single failure was the
tripwire above, re-run green afterwards. Inventory total 2979 -> 2984.

**What this does not establish.** The verdict adjudicates the frozen T4C.6 relationship family on
this exact crop only -- one variable, one level, one region, six years -- and is not causality,
universality, forecast skill or operational readiness. The independent cross-route check covers
**eight of the 8,764 frames**, the window the frozen design specifies; the other 8,756 are
guaranteed structurally rather than against a second archive, and no mid-record independent
window has been acquired. **D84 and D85 remain open** and did not gate this result. The gate
panel's agreement-rule column still has **not been inspected in a browser**. The full backend
suite has not been rerun since these changes, so **3106** remains the last measured full-suite
figure, and the documentation audit has not yet been re-run against this tree.

## T4C.5n -- the mid-record audit window

**Acquisition.** WeatherBench `era5_0p25_6h`, 2021-07-01T00:00 to 2021-07-02T18:00, same region,
level, variable and eight-frame length as the frozen window; only the dates move. `content_key
19c03cdcde90ceb2`, **237,731,669 bytes** in **62.9 s**.

**Audit.** `passed True`, `role audit`, `max_error_in_steps` **0.46875** against `steps_allowed
1.0`, `encoding_steps [2^-9]` with all eight frames on that step, `max_abs_error
9.1552734375e-4`, `rmse 4.1482507906808234e-4`, `mismatch_count 0`, `value_count 207,368`,
`receipt_sha256 0f32c89a600b9759133df68a2b0943ef7e459a086f4acfd814fe83390dc65a8f`. Receipt at
`data/era5_cache/a07c23ec89f953c1.overlap.encoding_relative.midrecord2021.json`.

**The Kelvin inversion, reproduced on new data.** The audit window disagrees more in Kelvin than
the gate window (9.155e-4 against 7.324e-4) and less in steps (0.469 against 0.719). This is the
first reproduction of that inversion on a window acquired *after* the criterion was frozen, so
it cannot be an artefact of the derivation.

**A defect caught by the tests during this slice.** The new `label` parameter collided with an
existing loop variable of the same name inside `verify_cached_era5_overlap`, so every receipt
bound as `..._encoding_relative_independent` -- evidence filed under the wrong name by a change
that otherwise looked correct. Caught because the tests assert on manifest field names rather
than only on `passed`. Loop variable renamed to `side`.

**Tests.** `test_cds_source.py` 22 -> 24 functions. Clean run of `test_cds_source.py
test_gate_campaign.py test_gate_api.py test_gate_run.py test_zarr_source.py`: **136 passed, 1
skipped** in 95.02 s. Inventory total 2984 -> 2986.

**Full backend suite.** **3362 passed, 4 skipped, 1 xfailed** in 2,289.91 s (38:09), exit code 0.
This supersedes **3106** as the last measured full-suite figure. It was launched before the
T4C.5n label work, so it measures the tree as of the D87 fix; the 136-test clean run above covers
the T4C.5n changes, and the full suite has not been re-run since them.

**What this does not establish.** Two independently verified windows out of 8,764 frames. The
audit authorises nothing and does not revisit the gate verdict. D84 and D85 remain open. The
gate panel's agreement-rule column still has not been inspected in a browser.


## T4D.1 -- located maxima (`src/analysis_engine/spectral_feature.py`)

**Tests.** `src/tests/test_spectral_feature.py`, 15 test functions, **17 passed** in 1.84 s.

**What was measured.** A Gaussian blob planted at (32.0, 32.0), (30.4, 41.7) and (25.5, 25.5) is
recovered to better than one pixel in each case, and the parabolic refinement beats the integer
peak it starts from on the deliberately worst case at (30.5, 41.5). At scale 3 a blob at (1, 1)
is found with `interior=False` and refused with `interior=True`, so the R13 exclusion is the mask
and not a blind spot. Two frames, one with a blob ten times the other's amplitude: with the
threshold fitted over the record only frame 0 reports a detection, and a frozen threshold
re-supplied reproduces the fitted result exactly while a threshold of 1e6 yields nothing.

**The design change the tests forced, with its numbers.** Two acceptance cases failed under a
strict local-maximum rule: a blob at a half-pixel position produces two exactly equal samples and
strictness rejects both. Rewritten to regional maxima: a 4x4 flat top is now one detection at
(31.5, 31.5) with `plateau_pixels == 16`, and a blob at (32.0, 31.5) is found at exactly (32.0,
31.5) with `plateau_pixels == 2`. A plateau adjoining a strictly higher pixel is not reported at
all.

## T4D.2 -- tracks, and D88

**Tests.** `src/tests/test_spectral_tracking.py`, 19 test functions, **19 passed** in 3.18 s.

**D88, measured before it was fixed.** A delta at (64, 64) of a 128x128 field, SWT:

```
            LH offset      HL offset      HH offset     accumulated support / 2
haar  L4    (+0, +4)       (+4, +8)       (+4, +0)       7.5
db2   L1    (+2, +1)       (+1, +2)       (+1, +1)       1.5
db2   L2    (+6, +4)       (+4, +6)       (+4, +4)       4.5
db2   L3    (+14, +10)     (+10, +14)     (+10, +10)    10.5
db2   L4    (+30, +22)     (+22, +30)     (+22, +22)    22.5
db3   L4    (+60, +36)     (+36, +60)     (+36, +36)    37.5
```

The delta's argmax is a filter-peak artefact, so the decisive measurement is the centroid of
`|band|` for a symmetric Gaussian blob at (64, 64), which for a symmetric structure must sit at
the structure's centre if the band is registered:

```
             centroid          minus analysis_delay
haar L1     (64.50, 64.50)     (64.00, 64.00)
haar L2     (65.50, 65.50)     (64.00, 64.00)
haar L3     (67.50, 67.50)     (64.00, 64.00)
haar L4     (71.50, 71.50)     (64.00, 64.00)
db2  L4     (97.54, 88.65)     (75.04, 66.15)
db3  L4     (79.83, 96.62)     (42.33, 59.12)
```

Haar returns to the blob's centre exactly at every level. db2 and db3 do not, and the residual
grows with the level's dilation -- the non-linear-phase half of the defect, which no shift can
remove. `analysis_delay("haar", 1..5) = 0.5, 1.5, 3.5, 7.5, 15.5`;
`analysis_delay("db2", 1..4) = 1.5, 4.5, 10.5, 22.5`;
`is_linear_phase = {haar: True, db2: False, db3: False}`.

**The acceptance run.** `advected_vortex_sequence` (128x128, 24 frames, sigma 5.00 -> 13.54 cells,
velocity (1.5, 2.5) cells/step, doubling 16 steps -- all recorded before this code existed), SWT
haar at 5 levels, scales 4 and 5, `threshold_sigma=4.0`, `MotionBounds(max_doublings=0.5)`,
Hungarian association at alpha 0.05:

```
                              measured v        predicted v      err   transverse worst
L4 HL  len=24  t0= 0.0   (1.823, 2.526)   (1.871, 2.500)   0.0544   col  0.6668
L4 LH  len=24  t0= 0.0   (1.505, 2.852)   (1.500, 2.871)   0.0197   row  0.3521
L5 HL  len=15  t0= 9.0   (1.835, 2.532)   (1.940, 2.500)   0.1093   col  0.3447
L5 LH  len=11  t0= 9.0   (1.506, 2.807)   (1.500, 2.900)   0.0935   row  0.2323
```

4 tracks, 4 births, 1 death, 24 frames. The prediction column is advection plus
`(sigma_end - sigma_start) / (t_end - t_start)` on the axis that band high-passes, with no fitted
parameter. Worst velocity error **0.109 cells/step**; worst transverse position error **0.667 px**,
inside the roadmap's 1 px on the coordinate for which it is a claim about the structure.

**What this does not establish.** A detail-coefficient maximum is a flank, not a centre, so the
longitudinal coordinate of every track above is offset by about the structure's width and grows
with it; the roadmap's "< 1 px position" is met by the field-space path (`4D.position`, 0.052
cells) and not by this one. Nothing here is calibrated against a null: the threshold is
`sigma x RMS` and no significance is reported. The vortex is synthetic; no track has been produced
from ERA5. `4D.tracking` in the benchmark suite is graded through `src/core/extraction`, not
through this module, and its status is unchanged by this slice.

**Neighbourhood run.** `test_spectral_tracking.py test_spectral_feature.py test_coefficient_field.py
test_wavelet_bank.py test_tracking.py test_cross_scale.py test_scale_signature.py
test_stationary.py test_transforms.py test_benchmarks.py`: **387 passed, 1 xfailed** in 325.75 s.
Inventory total 2986 -> 3020.

**Full backend suite, after T4D.1, T4D.2 and D88.** **3400 passed, 4 skipped, 1 xfailed** in
2,165.95 s (36:05), exit code 0. This supersedes 3362 as the last measured full-suite figure and
measures the current tree. The arithmetic reconciles exactly: 3362 (as of the D87 fix) + 2
(T4C.5n's `test_cds_source.py` additions, which the 3362 run predated) + 17 (T4D.1) + 19 (T4D.2)
= 3400. Documentation audit inside that run: 20 passed.


## T4D.3 -- the sentences, and D89

**Tests.** `src/tests/test_spectral_narrative.py`, 25 test functions, **25 passed** in 2.22 s.
`src/tests/test_translation.py` gains one parametrised function for D89: **64 passed** (48 test
functions).

**The narrative, in full, of the T4D.2 acceptance pass.** Same field, same tracker settings, same
four tracks. This is the whole rendered output, not an excerpt:

```
Track 0 (advected_vortex_sequence, amplitude; level 4, LH): the coefficient maximum was followed
across 24 of 24 searched frames, consecutively, 0.00 to 23.00 frames. It moved 74.2 cells toward
increasing row and increasing col, at a mean 3.22 cells/frames. It stayed at level 4 throughout,
so this track reports no change of scale; which bands were excited, and when, is a statement about
the set of tracks rather than about this one. Its peak coefficient magnitude went from 3.7372 to
2.5576, a change of -31.6%, which is a change of -53.2% in coefficient energy -- energy being the
square, so the two figures are not interchangeable.

Track 1 (level 4, HL): ... 71.6 cells ... 3.12 cells/frames ... 3.7528 to 2.5242, -32.7%, -54.8%.
Track 2 (level 5, LH): 11 of 24 frames, 9.00 to 19.00 ... 6.3793 to 7.5824, +18.9%, +41.3%.
Track 3 (level 5, HL): 15 of 24 frames, 9.00 to 23.00 ... 6.3965 to 7.4860, +17.0%, +37.0%.

4 bands produced tracks. The earliest was L4/HL at 0.00 and the latest L5/LH at 9.00, a separation
of 9.00 in the clock's units. That ordering is a candidate precursor relationship between two
bands of one record: it is co-occurrence with a recorded sign of the time offset, it was not
tested against a null, and it is not a structure moving up the bank -- the transform is redundant,
so both bands respond at once and no merge is claimed.
```

**The measurement that replaces "dominant scale doubled".** The vortex widens from sigma 5.00 to
sigma 13.54 cells. No single track records that, since each holds one dyadic level and therefore
has a scale velocity of exactly zero. What is measured, across the population:

```
band     first seen   frames   magnitude first -> last     change
L4/LH        0.0        24        3.7372 -> 2.5576        -31.6%
L4/HL        0.0        24        3.7528 -> 2.5242        -32.7%
L5/LH        9.0        11        6.3793 -> 7.5824        +18.9%
L5/HL        9.0        15        6.3965 -> 7.4860        +17.0%
```

Both fine bands weaken and both coarse bands strengthen, and the coarse bands are not excited
until frame 9. That is the growth, expressed as an ordering of two bands over one record, which
is the strongest form the evidence supports.

**The compass, checked against arithmetic rather than against itself.** One displacement of
(+1 row, 0 col) on two grids that differ only in the sign of `dy` gives bearings of 0.0 and 180.0
degrees -- north and south -- which is the ERA5 case, since ERA5 stores rows north to south. A
displacement of (+1, +1) at `lat0 = 60` gives **26.6 degrees**, not 45: the east component is
shortened by `cos(60.125 deg)` before the bearing is taken, so omitting the cosine would rotate
the answer and change the compass word. A `cartesian` grid is refused every compass word and gets
axis-relative wording instead. A track with a net displacement of exactly zero is given no
bearing at all.

**Magnitude against energy.** A track whose magnitude goes 1.0 -> 1.43 renders "a change of 43.0%,
which is a change of 104.5% in coefficient energy". The roadmap's example sentence named the
second under the first's number.

**D89, and how it was found.** T4D.3's causal guard was written as `\bword\b` over the rendered
sentence, matching the programme's existing guard in `src/core/translation.py`. The test that puts
a caller's own dataset name into a sentence -- `dataset="co2_causes_warming"` -- failed, DID NOT
RAISE. An underscore is a word character, so the boundary sits at the ends of the identifier and
not at its underscores. Both guards now flatten punctuation to spaces before matching. The
remaining limit is asserted, not assumed: `co2causeswarming` still passes, and `causeway` still
passes, which is the point -- a substring match would catch the first and refuse the second.
`test_translation.py` was re-run after the change: **64 passed**, no existing expectation moved.

**What this does not establish.** Every sentence above describes synthetic data. No narrative has
been produced from ERA5, and the candidate precursor relationship reported is one ordering
observed once in one record: it was not tested against a null, it has no significance attached,
and it is not evidence that the ordering recurs. The guard refuses causal *vocabulary*; it cannot
refuse a causal *reading*, and the entitlement that travels with every narrative is the only thing
that addresses that.

**Neighbourhood run.** `test_spectral_narrative.py test_spectral_tracking.py
test_spectral_feature.py test_tracking.py test_claim_ladder.py test_translation.py
test_five_outputs.py test_coefficient_field.py`: **278 passed** in 35.38 s (before the D89 test was
added). Inventory total 3020 -> 3046.

**Full backend suite, after T4D.3 and D89.** **3429 passed, 4 skipped, 1 xfailed** in 2,197.90 s
(36:37), exit code 0. This supersedes 3400 as the last measured full-suite figure and measures the
current tree. The arithmetic reconciles exactly: 3400 + 25 (T4D.3's `test_spectral_narrative.py`)
+ 4 (D89's one parametrised function in `test_translation.py`, over the four words it is
parametrised on) = 3429. The documentation audit was also run on its own against the finished
documents: **20 passed** in 276.06 s, exit code 0. `git diff --check` clean.


## T4E.1 -- the constellations, and D90

**Tests.** `src/tests/test_spectral_constellation.py`, 45 test functions, **54 passed** in 2.53 s.
Neighbourhood run (`test_spectral_constellation.py test_spectral_narrative.py
test_spectral_tracking.py test_spectral_feature.py test_tracking.py test_constellation.py
test_invariance.py test_motif.py test_coefficient_field.py`): **370 passed** in 75.48 s.

**The pass, over the vortex T4D.2 tracked.** Same field, same tracker settings, same four tracks.
24 searched frames, 4 tracks, **135 constellations**: 87 pairs and 48 triples. The count is
checked frame by frame against the combinatorics of the pass's own census, so it cannot quietly be
a sample. By band:

```
L4/LH+L4/HL              24     L4/HL+L5/LH              11
L4/HL+L5/HL              15     L4/HL+L5/LH+L5/HL        11
L4/LH+L4/HL+L5/HL        15     L4/LH+L4/HL+L5/LH        11
L4/LH+L5/HL              15     L4/LH+L5/LH              11
                                L4/LH+L5/LH+L5/HL        11
                                L5/LH+L5/HL              11
```

**Three of the eight TG3.3 relations are measurable; five refuse by name.** Measurable:
`distance`, `relative_scale`, `succession`. Refused, with the field each one lacks:
`temporal_lag` and `co_occurrence` (no `temporal_scale`), `direction` and `convergence` (no
`orientation`), `containment` (no `extent`). Every refusal is in the receipt with its reason,
rather than dropped from the declaration.

**The two halves, at frame 9.** The left pair of columns is the graph -- dimensionless, and the
only half a match or a cluster may read. The right is carried, in this record's own units:

```
pair             distance  rel_scale |  separation   bearing    raw   sigma   onset
L4/LH+L4/HL        1.4447      1.00  |  11.56 cells   315.4   1.000   1.000   0.0 (bound)
L4/LH+L5/LH        0.2046      0.50  |   2.32 cells    88.7   1.755   0.554   9.0 (bound)
L4/LH+L5/HL        1.1887      0.50  |  13.45 cells   321.3   1.760   0.556   9.0 (bound)
L4/HL+L5/LH        1.1716      0.50  |  13.26 cells   128.1   1.756   0.555   9.0 (bound)
L4/HL+L5/HL        0.2026      0.50  |   2.29 cells   352.9   1.761   0.556   9.0 (bound)
L5/LH+L5/HL        0.9356      1.00  |  14.97 cells   314.3   1.003   1.003   0.0
```

**What each column is not.** `L4/LH+L4/HL` is one vortex seen through two orientations of one
level, and its 11.56 cells is entirely **flank geometry** -- a detail maximum sits about one
analysing width off the structure's centre, so a separation is between two flanks and not between
two structures. The raw and band-normalised strength ratios **disagree about the sign of the
comparison**: raw 1.755 says the coarse band is stronger by three quarters, and each strength
divided by its own band's RMS says 0.554, the weaker of the two. The band RMS is not estimated
here -- it comes back exactly from the detection's own `threshold / threshold_sigma`, and the test
checks that for all 318 nodes. Every offset marked *(bound)* has a left-censored end: both level-4
tracks were alive in the first searched frame, so the record began before they did.
`L5/LH+L5/HL`, born together at frame 9, is the one uncensored pair, and it carries no bound note.

**`succession` is not the ordering that carries information.** It is asserted false for every
ordered pair of every one of the 135 constellations, because the two observations are in the same
frame by construction. The informative ordering is between onsets, and it is carried separately.

**D90, and how it was found.** The first attempt to build a graph from these features returned
`distance` in the refusals rather than in the edges: *"'11.5579 cells vs 8 parent-grid px' ... the
separation is in 'cells' and the scales are in 'parent-grid px', so the quotient is not a number
of scale lengths."* The check is right and the units were wrong -- the bank is undecimated and
T4D.1 maps every level onto the parent grid, so a parent-grid pixel is a cell. The failure was
**silent**, because a refusal is recorded on the graph rather than raised: a mining pass would
have run with the geometry missing and reported patterns built from `relative_scale` and
`succession` alone. Fixed in `_scale_quantity`; the regression is pinned on the two unit names
themselves rather than on the symptom, `distance` is asserted measured on every pair of the pass,
and a separate test builds a scale genuinely in metres beside a location in cells and asserts it
**still refuses**, so the fix cannot be read as a weakening of the rule.

**What this does not establish.** Every number above is from synthetic data; no constellation has
been extracted from ERA5. There is no null, no support count and no significance anywhere in this
slice: a constellation is one observation of one arrangement in one searched frame. Recurrence is
T4E.3's clustering and T4E.4's support threshold, invariant matching is TG3.4's, and none of the
three has run over this output. The four tracks are four bands of one vortex, so the "population"
here is one structure seen four ways rather than four structures.

**Full backend suite, after T4E.1 and D90.** **3483 passed, 4 skipped, 1 xfailed** in 2,058.00 s
(34:17), exit code 0. This supersedes 3429 as the last measured full-suite figure and measures the
current tree. The arithmetic reconciles exactly: 3429 + 54 (T4E.1's
`test_spectral_constellation.py`, 45 test functions of which two are parametrised over five and
six cases) = 3483. D90's fix added no test case of its own; its regressions live inside that file.
`git diff --check` clean.

## T4E.2 -- the invariant signature, and the benchmark with no axis

`src/analysis_engine/spectral_invariance.py`, verified by `src/tests/test_spectral_invariance.py`
(45 test functions, 53 cases, 33.9 s). Everything below is measured on this tree.

### What the toggle costs, on the vortex pass

`compare_scale_modes` runs the same 135-constellation pass with scale invariance off and on:

| | scale-specific | scale-invariant |
|---|---|---|
| geometry keys on | `distance` (TG3.3) | `shape_ratio` (TG3.4 `relative_geometry`) |
| signed | **135** | **48** |
| refused | 0 | **87**, all cardinality 2 |
| axis refused | 87 (every pair) | 0 |
| invariant to | translation, rotation, reflection | translation, rotation, reflection, **rescaling** |
| crosses a domain boundary | no -- scales in cells | yes -- every entry dimensionless |

Turning the universality hook on costs every pair and nothing else: a scale-free shape is a ratio
between separations, and a pair has one separation whose ratio to itself is 1 for every pair in
every domain. The comparison deliberately stops there. It does **not** report how many distinct
configurations each mode sees, because that is a count of clusters, a cluster needs a tolerance,
and a tolerance calibrated rather than chosen is T4E.3.

### Invariance, measured on exact geometry

The reference is a scalene triangle -- no symmetry, so exactly one correspondence recovers it --
placed by hand so the truth carries no noise. In **both** modes the signature vector is identical
to floating-point precision (`abs=1e-9`, `1e-8` after a rotation) under:

*   translation by (311, -207) cells;
*   rotation by 17, 90 and 233.5 degrees;
*   reflection;
*   all five non-identity relabellings of the three members.

Two further measurements separate the modes:

*   a **uniform** rescaling by 0.5, 2.0 and 7.5 leaves the scale-invariant vector unchanged;
*   an estimator that **misses** a rescaling -- positions doubled, recorded scales left where they
    were -- moves the scale-specific geometry by exactly a factor of two and leaves the
    scale-invariant vector unchanged. That is TG3.4's measured drift reproduced as arithmetic
    rather than re-measured: on the real pipeline the miss is 4.8% at `scale_factor=3` against a
    3.3% noise floor, which is why `distance` is not registered as rescaling-invariant.

### The finding: `planted_configuration` is equilateral, so it has no principal axis

The specification asks for "bearings measured relative to the constellation's own principal
axis". The benchmark this programme supplies for invariance -- the one TG3.4's `4E.invariance`
gate runs on -- is an equilateral triangle, whose position covariance is isotropic. Measured over
**24 field-noise realisations** of the same planting, through the real extraction pipeline:

| quantity | measured |
|---|---|
| anisotropy `lambda_1 / lambda_2` | **1.0077 to 1.0421** |
| recovered axis angle | **0.78 to 158.08 degrees** |
| circular standard deviation of that angle (mod 180) | **~49.8 degrees** |
| shape ratios over the same replicates | reproduce to **0.218%** |

The shape is stable to a fifth of a percent and the axis it implies is uniform noise. A bearing
block written without a guard would have reported a confident angle for every one of those 24
realisations, all of them different, on the exact configuration nominated for testing invariance.

`AXIS_ISOTROPY_FLOOR = 1.0421` is that measurement -- the largest anisotropy a known-isotropic
configuration produced under noise -- in the same spirit as TG3.4's `calibrate_match_tolerance`:
an operating point is what the noise did, not what an author thought reasonable.
`calibrate_axis_admission` re-measures it, and a test asserts the module's constant is the number
the measurement produces. Clearing the floor is a **minimum, not a precision claim**: a
configuration just above it still has a poorly determined angle. The vortex triples clear it by
two orders of magnitude -- smallest observed anisotropy **85.22**, median 183.28 -- which is why
the bearings on this record are usable at all, and the reason is that the four tracks are the
flanks of one vortex and lie nearly on a line.

### What each block is, and is not

*   **Geometry** -- one value per unordered pair, from TG3.4's matcher rather than recomputed.
    Both matchers refuse a configuration outright rather than recording a hole, so there is no
    partial shape.
*   **Bearings** -- folded to [0, 90] degrees, because an edge is unordered and an axis has no
    sign. The fold makes the signature invariant to reflection as well as rotation. That is a
    consequence of the grid declaring no orientation (`has_orientation: False` on every 4D
    feature), not a preference; telling a configuration from its mirror image would need an
    orientation convention this record does not have. At three members the bearings are a
    *function of* the geometry block -- three separations determine a triangle up to similarity
    and reflection -- so they add no degree of freedom and are kept as the readable form.
*   **Strength** -- each member's magnitude over its own band's RMS, then over the geometric mean
    of the members'. The band normalisation is not optional: T4E.1 measured the raw and
    band-normalised ratios disagreeing about *which member is stronger*. A member with no
    recorded band RMS is refused.
*   **Scale** -- absolute, in cells, inside the scale-specific vector; ratios only inside the
    scale-invariant one. This is the second place the toggle bites, and it is the difference
    between "does this recur at *this* scale?" and "does this recur at *other* scales?". It is
    also why only one mode crosses a domain boundary, as a fact about the vector rather than a
    policy about it: a length in cells has no ratio to a length in metres. The absolute scales
    are carried beside the vector in both modes.

### The canonical order

The vector is minimised lexicographically over all node correspondences (six at most). A test
exhibits why sorting each block separately would be wrong: two configurations that are the same
triangle with the same three strengths *attached to different vertices* agree on sorted geometry
and on sorted strengths, and disagree on the minimised vector, because no single correspondence
makes both blocks true at once.

### What this does not establish

Nothing here is a match, a cluster, a support count, a null or a p-value. Every configuration
signed is synthetic: the vortex sequence and hand-placed triangles. Nothing has been signed from
ERA5 or from any second domain, so the cross-domain comparability of the scale-invariant mode is
a property of its units and not a demonstration. The universality hook is **exercised and not
evidenced**: the tracked bank has two levels, so every triple's scale block is one of two ratio
patterns and "does this configuration recur at *other* scales?" has almost no room to be answered
on this record. And a bearing that clears the isotropy floor is admitted, not certified -- the
floor says the elongation is not noise, not that the angle is accurate.

**Full backend suite, after T4E.2.** **3536 passed, 4 skipped, 1 xfailed** in 2,787.05 s (46:27),
exit code 0. This supersedes 3483 as the last measured full-suite figure and measures the current
tree. The arithmetic reconciles exactly: 3483 + 53 (T4E.2's `test_spectral_invariance.py`, 45 test
functions of which three are parametrised) = 3536. T4E.2 found no defect, so the ledger is
unchanged at D1-D90. `git diff --check` clean.

## T4E.3 -- approximate attributed-graph matching with a measured radius

`src/analysis_engine/spectral_clustering.py`, verified by
`src/tests/test_spectral_clustering.py` (16 test functions). Everything below was measured on
2026-09-04 against this tree; no network was used.

The metric declares all four T4E.2 blocks -- geometry, bearings, relative strength and scale --
and publishes their weights and dimensionless component reductions. The calibration takes three
known measurements of the same physical scalene configuration, measures all three pair distances,
and retains their maximum in TG3.4's `MatchTolerance`. With equal declared block weights, the
measured scale-specific radius is **0.2061925217**. The metric digest and signature family are
bound into that tolerance, so changing a weight, mode, cardinality, axis availability or
scale-unit contract makes the measurement inadmissible rather than silently reusing it.

The acceptance pass supplies four presentations not used as four independent discoveries: the
base triangle, a translation, a 73-degree rotation, and a held presentation with position,
strength and spatial-scale perturbations bounded at 10%. All four land in **one cluster**. When
position and member scales are both doubled, the scale-specific distance is **0.3333333333**,
outside its 0.2062 measured radius, and the two presentations form **two clusters**. In
scale-invariant mode the doubled presentation is identical under the declared comparable blocks
and the two form **one cluster**. This meets both halves of the roadmap acceptance.

Complete-link behaviour is pinned separately with A-B and B-C each inside the measured radius
while A-C is outside: the output is a cluster of two plus a singleton, never one chained cluster.
Reversing input order leaves the full receipt, membership and pattern IDs identical. Each pattern
reports an aligned centroid, the calibrated tolerance radius and the observed member radius; the
receipt states that its raw member count is not minimum-support mining, recurrence evidence, a
p-value or a discovery.

**D92 found and fixed.** The first acceptance run compared T4E.2's canonical vectors component by
component. Exact canonicalisation is valid for exact invariance but discontinuous under noise: one
10% replicate put a different edge first, attaching the strength and scale blocks to different
physical vertices. The scale-block RMS became **0.4743** instead of about 0.10, the calibrated
radius widened to **0.3418**, and the doubled absolute scale at 0.3333 incorrectly joined in
scale-specific mode. Approximate distance now minimises over every valid node correspondence
(at most six), moving edge and node attributes together; centroids align to a deterministic medoid
before averaging. A direct regression starts on the replicate whose exact canonical order flips
and measures its aligned scale-block disagreement below 0.11.

Recorded commands:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_clustering.py -q
................                                                         [100%]
16 passed, 1 warning in 2.38s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_clustering.py src/tests/test_spectral_invariance.py src/tests/test_spectral_constellation.py -q
........................................................................ [ 58%]
...................................................                      [100%]
123 passed, 1 warning in 25.80s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py src/tests/test_spectral_clustering.py -q
.............................................                            [100%]
45 passed, 2 warnings in 260.67s (0:04:20)
```

The warnings are SQLAlchemy's existing `declarative_base()` deprecation from
`src/database/session.py:55` and Starlette's existing `python_multipart` pending deprecation,
not T4E.3 failures. The full backend suite was **not** rerun; its
last measured figure remains **3536 passed, 4 skipped, 1 xfailed** from T4E.2 and predates the
TG17.11-TG17.14 and T4E.3 changes.

## T4E.4 -- bounded minimum-support mining

`src/analysis_engine/spectral_mining.py`, verified by
`src/tests/test_spectral_mining.py` (14 test functions). Measured on 2026-09-04 with no network.

The acceptance catalogue has two T4E.3 clusters: one contains five distinct constellation keys
and one contains two. At the declared inclusive minimum support of three, exactly the
five-occurrence pattern is `SUPPORTED` and the two-occurrence pattern is
`PRUNED_BELOW_MINIMUM`. Both remain in the receipt with their support, support unit, centroid,
calibrated tolerance and observed radius. At minimum five the first remains admitted; above all
observed counts the result is complete with zero supported patterns rather than failing or
returning an ambiguous empty list.

The scan orders candidates by decreasing support, then prunes the entire remaining tail at its
first miss. Support is counted over distinct `signature.key` identities; inserting the same
constellation twice is refused with the reason that duplication would manufacture support.
Reversing all seven input observations leaves pattern IDs, counts and decisions unchanged.

Both hard budgets are mandatory. A catalogue of two candidates under `max_candidates=1` refuses
before scanning and reports `partial_result: false`. Injected monotonic-clock overruns during
identity/support preflight and mid-scan both raise the client-safe `MiningBudgetExceededError` and
return no result. The successful receipt carries the exact budget and names elapsed time
`unasserted`, keeping an operational measurement out of deterministic claims. Its claim boundary
states that minimum support is not recurrence significance, a null test, a p-value, predictive
evidence or a discovery.

Recorded command:

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_mining.py src/tests/test_spectral_clustering.py -q
..............................                                           [100%]
30 passed, 1 warning in 3.03s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_mining.py src/tests/test_spectral_clustering.py src/tests/test_spectral_invariance.py src/tests/test_spectral_constellation.py -q
........................................................................ [ 52%]
.................................................................        [100%]
137 passed, 1 warning in 29.60s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
.............................                                            [100%]
29 passed, 2 warnings in 263.70s (0:04:23)
```

The warning is the existing SQLAlchemy `declarative_base()` deprecation. Full backend was not
rerun; **3536 passed, 4 skipped, 1 xfailed** remains the last full measurement and predates this
slice.

---

# The TG17.11 - TG18.5 backfill (recorded 2026-09-04)

Eight phases reached a terminal state between 2026-09-02 and 2026-09-04 without an entry in this
file. `roadmap.md` §10.2 requires recorded command output here for every phase claiming
completion, and the requirement went unguarded: nothing in `test_documentation.py` compared the
roadmap's completed phases against this file's headings, so the gap widened silently for eight
phases while the atmospheric line (T4C, T4D, T4E) continued to be recorded correctly.

**The output below was recorded contemporaneously and is transcribed, not re-measured.** Each
entry names the commit that recorded it. Re-measuring today would produce a different and less
honest record: it would attribute today's tree to a phase that closed against an earlier one. The
one entry measured today is TG18.5's own, which closes against this tree.

A guard now exists (`test_every_completed_cross_domain_phase_has_a_verification_entry`), so a
phase cannot again be marked done in the roadmap while this file is silent about it.

## TG17.11 Scale/shape mining calibration — DONE, GATE REFUSED (2026-09-03, `ed-dev`)

Transcribed from `2fe3e0c`, `a68ccf8`, `4d49ec6`, `0c0a22b`, `951f169`, `81ac6ed`.

`scale_shape_calibration` was the only qualification gate reading `NOT_IMPLEMENTED`. That sentence
is now false, and what replaced it is not a pass.

**The finding is two bounds that do not meet.** Below, every member of a k-pairing family sits at a
p-value floor of `1/k`, so under Benjamini-Yekutieli at alpha 0.05 no family smaller than **105**
pairings can reject even when every member is a perfect planted match. Above,
`reassign_scale_partners` enumerates exactly and refuses above **8** pairings rather than assume a
sampler's uniformity. There is therefore no inventory size at which the null *as the qualification
manifests declare it* — drawn, with a replication count — can produce a rejection. What can is the
exact partner test, which the registered calibration is built on and no declared manifest requests.
Both G17 candidate families are refused before size is reached: the quartet all-pairs admits one
distinguishable reassignment, the post-D83 triple none.

The gate reads `REFUSED`, the status reserved for a declared scientific limit that blocks release
exactly as a failure does. The record states `calibration_executed_here: false` and asserts only
quantities it computed; a guard fails if any power key appears in it. **A calibrated method the
declared plans cannot reach, a method that does not exist, and a method that ran and failed are
three different facts**, and the gate now distinguishes them.

D91 was found and fixed in the first slice: the declared families answered a different question
than the one the null poses.

```text
> pytest src/tests/test_documentation.py src/tests/test_experiment_qualification.py ^
    src/tests/test_experiment_family.py -q
114 passed

> pytest src/tests/test_frontend_contract.py -q
181 passed

> cd frontend && npx playwright test e2e/flagship-qualification.spec.ts
2 passed          (rendered ledger shows REFUSED)

> cd frontend && npm run build
passed

Three mutations, each caught: awarding the gate PASS fails two guards; hard-coding the
applicability verdict with a power number fails two; replacing the null's own refusal text
with a fixed string fails one.

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

## TG17.12 Calendar calibration recorded into its release gate — DONE (2026-09-03, `ed-dev`)

Transcribed from `015ae48`.

`calendar_calibration` was the last of TG17.10's seven gates whose `NOT_RUN` was true of the record
and false of the world. `calibrate_family` runs the frozen calendar family on the TG17.0 fixtures
and has always passed; nothing carried that measurement into the qualification record.

**The gate still does not run it.** A calibration is a scientific measurement and the gate is a
release gate. What this adds is the channel, `src/core/calibration_record.py`. A recording binds two
digests and reads as unrun if either moves: the declared contract (cases and their frozen rejection
counts, family size, alpha, correction, replications, channel, null family, seed), digested from
`CALIBRATION_CASES` rather than restated so a relaxed expectation cannot leave a stale pass agreeing
with it; and the source of the four modules that decide what the measurement is, with line endings
normalised.

**Four outcomes, three blocking.** Absent, unbound from contract and unbound from source all read
`NOT_RUN`, because a recording made against something else is a measurement of a different thing
rather than a weaker pass. Cases that missed their frozen answers read `FAIL`.

Neither digest is tamper-evidence against an editor of this repository, and the source binding
covers four files rather than the whole import graph. That boundary is stated rather than hidden:
the backstop is that the live calibration already runs in the suite every pass, and a guard in
`test_experiment_family.py` compares it case by case against what the gate is being told.

**The contrast with TG17.11, computed rather than asserted.** Both nulls have a p-value floor and
buy it differently. Scale/shape buys it with domains: a k-pairing family cannot go below `1/k`, its
draw refuses above 8, its correction needs 105. Calendar buys it with computation: the floor is
`1/(1+replications)` and there is no enumeration ceiling, because the surrogates are clock shifts
the record itself supports. Both configurations are checked against the real correction — the
calibration family at 999 against 293 required, the declared manifest plan at 4 corrected members
and 200 replications against 166 — and the guard asserts the inequality, not the numbers.

```text
Recorded at 999 replications on a family of six:
  shared_calendar_event         rejects 6 of 6 after correction
  same_window_unrelated         rejects 0
  gap_alias                     rejects 0
  inadmissible_precedence       rejects 0
all_met is true, so the gate reads PASS - the first of the seven to clear - and the
verdict is unmoved at NOT_RELEASEABLE, with a guard saying so.

> pytest src/tests/test_calibration_record.py -q
12 passed

> pytest (documentation + qualification suites) -q
64 passed

> drift backstop against the live calibration
passed

> cd frontend && npx tsc --noEmit
clean

> cd frontend && npm run build
passed in 59.75s

Gate assembles in 22 ms warm, guarded below one second.

Four mutations, all caught: dropping the source binding, treating a missing recording as a
pass, collapsing FAIL into NOT_RUN, and removing the line-ending normalisation each fail
exactly one guard.

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

## TG18.0 Rendered baseline and interaction inventory — DONE (2026-09-02, `ed-dev`)

A declaration phase, and it is recorded here because what it declared became load-bearing two days
later. TG18.0 preserved the four distinct product modes found in the rendered baseline: the
interactive instrument (Spectral Transforms), the guided commitment workflow (Composer), the
read-only claim surface (Findings), and the trust and qualification surface (Platform & evidence).
Its constraint is that *a change which improves one by making another ambiguous is not a successful
redesign.*

**That constraint was unguarded from 2026-09-02 until 2026-09-04.** No test in the repository could
have detected the four modes converging. TG18.5's second slice supplied the guard, and the design of
that guard is the reason the constraint is a property of the modes *together* rather than four
separate checks — four per-mode assertions could each pass while the modes collapsed onto one
another. See the TG18.5 entry below.

```text
No command output: this phase changed no code. Its verification is the guard that
TG18.5 slice 2 added retrospectively (product-modes.spec.ts, 10 collected), recorded below.
```

## TG18.1 Shared instrument foundation — DONE (2026-09-03, `ed-dev`)

Transcribed from `62fcc0e`, `85cb027`.

One stable application frame, and then the rendered inspection that the fifth slice had recorded as
`NOT RUN` because the shared browser runtime exposed no session. **That was a tooling gap, not an
absent capability**: the repository already carried Playwright and a Chromium binary, so
`frontend/e2e/narrow-width.spec.ts` now runs beside the existing acceptance specs under the same
config, at 320, 375, 414 and 768 CSS pixels across all four product modes.

It found two defects that the source contract and the production build had both passed, neither
visible above the compact breakpoints:

* The shared canvas rule `.workspace-main > *` also matched the two `sr-only` children. Overriding
  their one-pixel clipped box with a real width and `margin-inline: auto` let those absolutely
  positioned elements escape the workspace's clipping, so **every workspace scrolled horizontally by
  14 px at a 320 px viewport**. The rule now excludes `.sr-only`.
* The below-480px reflow collapsed `grid-cols-2..5` to one explicit track but left `col-span-*`
  children alone. A child spanning two columns of a one-column grid makes the browser create an
  implicit second track, **so the two-column layout returned while the declared template still read
  as one**. Spans are now released with the tracks.

A third, smaller finding: the 11/12 px metadata floor covered `text-[10px]` and `text-[11px]` but
never `text-[9px]`, which Acquire, the lineage nodes and the capability profile all use. Raised,
with a source assertion.

The spec measures the laid-out document rather than restating CSS: document scroll width, content
past the right edge that no ancestor scrolls, computed font size on every text-owning element,
rendered control height, header containment, and both the used track count and the rendered row
occupancy of every grid.

## TG18.2 Scientific visualization workspace — DONE (2026-09-03, `ed-dev`)

Transcribed from `c888f64`, `b297b7f`, `4ddc21e`, `48000da`.

Coordinated plot focus and exact-value reading, and three commitments that are contracts rather
than options. A figure ships a **real text and table equivalent**, not a caption. **Comparability is
a stated contract, not a colour-scale option** — a shared scale is admissible or it is refused with
its reason. **A figure states the domain a claim was fitted over**, so an extrapolated reading
cannot be mistaken for an interpolated one.

Accessible resizable panes across all three gridded comparisons, with pointer, keyboard, bounds,
reset and narrow-screen stacking. Publication HTML export for every heatmap and line chart;
publication sheets preserve vector figures, captions, missingness, scale provenance, fit domains,
uncertainty and assumptions. **Export performs no new scientific analysis and does not mutate live
plots** — the property that keeps a reading sheet a record rather than a second instrument.

```text
> cd frontend && npx playwright test          (after removing .e2e-state)
104 passed          (Chromium)

> pytest (focused frontend + documentation) -q
202 passed

> cd frontend && npm run build
passed, 1,416 modules

> git diff --check
clean apart from line-ending notices

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

## TG18.3 Guided research journey — DONE (2026-09-03, `ed-dev`)

Transcribed from `f423acb`.

A global `Acquire -> Inspect -> Design -> Run -> Compare -> Admit -> Report` journey. Each shell
blocker provides exactly one legitimate remediation rather than a dead end. **The Composer remains
the authority for scientific status and next actions** — the journey routes, it does not adjudicate
— and the existing claim ladder remains separate and visible beside it. All eight legacy gridded
tools stay reachable with distinct text and visual treatment, and journey location clears when the
researcher enters a non-journey legacy workspace.

```text
> cd frontend && npx playwright test          (after removing .e2e-state)
109 passed          (Chromium)

> pytest (frontend + documentation) -q
204 passed

> cd frontend && npm run build
passed, 1,417 modules

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

## TG18.4 Responsive and assistive-technology acceptance — DONE (2026-09-03, `ed-dev`)

Transcribed from `ac0a978`.

Eleven rendered Chromium acceptance checks at desktop (1440x900), laptop (1024x768) and narrow
(375x667) layouts. The skip link is the first Tab stop and carries the shared focus indicator;
fragment navigation and real workspace changes focus the named heading; the compact drawer returns
focus to its trigger after Escape. A 640 CSS-pixel/device-scale-two reflow inspection stands in for
a 1280-pixel viewport at 200% zoom, with no horizontal document scroll. Contrast is **computed from
rendered foreground and composited background** at the normal- and large-text thresholds rather than
read off the stylesheet. `prefers-reduced-motion` is emulated and the resulting durations measured.
A colour-removal pass leaves location, both blockers, both remediations and the claim-ladder
boundary named in text and semantics.

**It fixed a real defect the first keyboard run found.** StrictMode mount replay focused the hidden
workspace heading and stole the initial Tab stop; focus routing now compares the actual previous and
current workspace.

**Claim boundary.** This is bounded browser engineering acceptance. It is not a screen-reader audit
and not a WCAG conformance certification, both of which `roadmap_cross_domain.md` §7 keeps
explicitly out of scope. TG11.6 remains the source-level contract.

```text
> cd frontend && npx playwright test          (after removing .e2e-state)
120 passed          (Chromium)

> pytest src/tests/test_frontend_contract.py src/tests/test_documentation.py -q
206 passed          (test_frontend_contract.py collects 179; Python inventory 3161)

> cd frontend && npm run build
passed, 1,417 modules

> git diff --check
clean apart from line-ending notices

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

## TG18.5 UI qualification gate — COMPLETE (2026-09-04, `ed-dev`)

Five slices. The phase's own declaration set the conditions, and the close-out below checks the
delivery against them rather than restating them.

**Slice 1 — served-workspace reachability.** `ui-qualification.spec.ts` asserts the shell serves
exactly the qualified inventory in order, that all twenty workspaces open from a clean browser and
name themselves, that every reachable journey destination lands on an inventoried workspace rather
than the fallback heading, and that a clean browser disables nothing and claims no reason it is not
entitled to. **The slice found a coverage hole rather than a defect**, and the distinction matters:
`ResearchJourney.tsx` holds its seven destination identifiers separately from `WORKFLOW_NAV`, and
Inspect and Admit are blocked in a clean browser, so their destinations were never clicked by any
test in the repository. Renaming the journey's `domainWorkbench` target was confirmed to pass the
entire rendered suite — TG18.3's own journey tests included — while sending a researcher who had
selected a record to a heading reading "Scientific workbench workspace". Nothing was broken; nothing
was guarding it either.

**Slice 2 — one representative path per product mode.** `product-modes.spec.ts` walks the
characteristic path of each of TG18.0's four modes at 1440 and 1920 CSS pixels, so the two viewport
specs now span 320 to 1920. **The load-bearing assertion is not any of the four paths.** TG18.0's
constraint is a property of the modes *together*, so each mode declares a signature — found by role
and accessible name, never a class or a test id — and the suite asserts every signature appears in
**exactly one** of the four. Four per-mode checks could each pass while the modes converged.

**Slice 3 — the two numbers `scientist_actions` refuses to invent.** The design turns on which of
the two an assertion may hold. An action count is deterministic, so it is asserted: **14 actions**
from a clean browser to a `COMPLETE` run of the frozen plan, and **3** to reach the preflight refusal
with **4** more to clear it. A wall-clock duration is not, so it is recorded and asserted by nothing:
how long a refusal takes to explain itself is a property of the machine that ran the suite, and a
gate turning on it would fail for reasons unrelated to the interface while passing on a fast machine
as the interface got slower. The committed recording carries **0.264 s** and **2.835 s**
as unasserted context, and those figures already differ from the 0.3 s and 4.74 s slice 3
first measured, because slice 4's full-suite run re-recorded them on a differently loaded
machine. A gate asserting either number would have failed on that alone. The
count is an **upper bound on the shortest route**, not a claim about a minimum, and the measurement
says so in its own claim boundary.

**Slice 4 — the evidence channel.** `qualification-reporter.ts` writes `measurements/browser_run.json`
and decides nothing; `src/core/browser_evidence.py` decides, and every decision can be a refusal.
Two bindings, and **the second is the one that was actually needed**: weakening a spec returns the
gate to `NOT_RUN`, and a *partial* run is refused, because running one spec is the normal way to work
on a test and Playwright reports it as `passed`. The reporter therefore records both the specs that
ran and the whole inventory it found, and the gate refuses when they disagree, naming every spec that
did not run. That refusal was verified **before** the first full run: a green four-test invocation
read `NOT_RUN` and listed the other fourteen specs.

The outcomes stay apart. Absent, stale or partial means the run has not happened *for this code* and
reads `NOT_RUN`; a run that happened and failed reads `FAIL`; only a complete, clean run of the suite
this checkout contains reads `PASS`. `browser_no_glue` reads **`PASS`**.

### The close-out (2026-09-04)

**1. The declared scope is delivered.** The phase declared five things. A clean-browser run exercises
one representative path through each product mode (slice 2); it captures named viewport artefacts
(slice 2); it asserts every served route remains reachable (slice 1); it records the action count and
the refusal-to-remediation measurement (slice 3); and a missing, stale or digest-mismatched artefact
reads `NOT_RUN` (slice 4). The constraint that the ledger "may ingest a measurement with its
provenance and may never synthesize one it did not receive" holds: `browser_evidence.py` computes no
count of its own and every field it publishes came from a recording or is a refusal.

**2. The phase's "nothing it measures is reported back inside the product" is narrower than what
shipped, and the declaration is amended rather than the code.** The trust surface renders every
gate's `detail`, so `browser_no_glue`'s cleared detail now states on screen that a rendered run of
all 15 specs passed 136 tests with 0 failed. That sentence was written against the risk of building a
UI-quality dashboard, and no such surface exists: the `scientist_actions` counts are declared in
`api.ts` and **read by no component**, so the 14/3/4 figures and both durations appear nowhere in the
product. What is on screen is one gate in the release registry that has always rendered. Suppressing
a cleared gate's basis while continuing to show every refusal's reason would make a `PASS` *less*
inspectable than a refusal, which inverts the property this programme is built on. The line actually
held is therefore: **no UI-quality surface, and no UI-quality measurement outside the release
registry's own gate verdict and the basis for it.**

**3. `qualification_plan()` is no longer uniformly cold, and a reader must not misread the
difference.** The plan used to return every gate `NOT_RUN` until something executed it. Three gates
now read recordings at assembly time — `browser_no_glue` `PASS`, `calendar_calibration` `PASS`,
`scale_shape_calibration` `REFUSED` — while `offline_matrix` and `restart_recovery` still read
`NOT_RUN` in the cold plan and are filled by an actual run. Both kinds of `NOT_RUN` block release
identically, but they mean different things: one is a measurement this checkout has not received, the
other is a run this call did not perform.

**4. Cross-domain condition 19 is half-met, and the phase does not clear it.**
`roadmap_cross_domain.md` §6.19 requires that a complete G17 capability pass *both* the clean-browser
no-glue test *and* the synthetic fifth-adapter test. The first now passes. `synthetic_fifth_adapter`
remains `NOT_RUN`, so G17 completion stays unclaimable, and TG18.5 must not be recorded as satisfying
19. The verdict is unmoved at `NOT_RELEASEABLE`, blocked by `offline_matrix`,
`scale_shape_calibration`, `synthetic_fifth_adapter` and `live_sources`.

**5. `roadmap.md` §10.2 was itself unguarded, which is how eight phases drifted out of this file.**
See the backfill above and the two guards added in `test_documentation.py`. This is the close-out's
substantive finding: the condition that no completion claim exists without recorded output was
enforced by habit alone, and habit failed for two days across seven phases without a single test
objecting.

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_browser_evidence.py ^
    src/tests/test_calibration_record.py src/tests/test_experiment_qualification.py ^
    src/tests/test_documentation.py src/tests/test_frontend_contract.py -q
261 passed, 5 warnings in 312.03s (0:05:12)

> cd frontend && npx playwright test          (after removing .e2e-state)
136 passed                 Chromium, 15 specs
recorded 2026-09-03T20:34:54.533Z, finished 20:42:39.556Z (7 m 45 s, unasserted)

  assistive-acceptance.spec.ts    11     narrow-width.spec.ts           19
  comparison-contract.spec.ts     12     product-modes.spec.ts          10
  comparison-views.spec.ts        19     publication-export.spec.ts      4
  composer-path.spec.ts           11     research-journey.spec.ts        5
  experiment-receipt.spec.ts       3     resizable-panes.spec.ts         6
  figure-data.spec.ts              9     scientist-actions.spec.ts       2
  flagship-qualification.spec.ts   2     ui-qualification.spec.ts        4
  validity-uncertainty.spec.ts    19     ------------------------------ 136

> cd frontend && npx tsc --noEmit
clean

> cd frontend && npm run build
1,417 modules transformed; built in 53.45s

> qualification_plan() gate states, cold
verdict: NOT_RELEASEABLE          assembled in 33 ms warm, guarded below one second
  offline_matrix           NOT_RUN     (filled by an executed matrix; REFUSED when run)
  restart_recovery         NOT_RUN     (filled by an executed rehearsal)
  browser_no_glue          PASS        <- TG18.5
  synthetic_fifth_adapter  NOT_RUN
  calendar_calibration     PASS        <- TG17.12
  scale_shape_calibration  REFUSED     <- TG17.11
  live_sources             NOT_RUN
browser_evidence: PASS   136 passed / 0 failed / 0 skipped, 15 specs, 32 artefacts
scientist_actions: MEASURED   14 actions, 3 to the refusal, 4 to clear it
adapter_specific_framework_edits: NOT_MEASURED   (belongs to synthetic_fifth_adapter)

> artefact manifest inspection
32 artefacts, keys exactly {name, bytes, sha256}; no image payload; 10,108 bytes total.
frontend/e2e/artifacts/ is gitignored and no artefact is tracked, so slice 2's constraint
on slice 4 holds: what reaches the ledger is a manifest and digest, never the images.

> four mutations against src/core/browser_evidence.py, module restored byte-identically
M1 a partial run accepted as a full one          1 failed, 15 passed
M2 the spec source binding dropped               1 failed, 15 passed
M3 a failed run reported as merely unrun         1 failed, 15 passed
M4 a drifted action count reported not refused   1 failed, 15 passed
restored: identical                              16 passed

Full backend not rerun; last measured figure remains 3536 (T4E.2).
```

```text
> two mutations against the documents the new guards read, restored byte-identically
M1 a completed phase loses its VERIFICATION.md entry   1 failed, 1 passed
M2 an entry exists while the roadmap says IN PROGRESS  1 failed, 1 passed
restored: VERIFICATION.md identical, roadmap_cross_domain.md identical
                                                       2 passed

The completeness guard was also written before the backfill and run against the tree as it
then stood, where it failed naming exactly TG17.11, TG17.12, TG18.0, TG18.1, TG18.2, TG18.3
and TG18.4 - the seven phases the backfill then supplied. A guard written after the entries
it demands would have proved nothing about whether it can see their absence.

> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py ^
    src/tests/test_frontend_contract.py -q
210 passed, 5 warnings in 265.81s (0:04:25)

The run before it failed on test_documented_test_counts_match_the_source, because the two
new guards had moved test_documentation.py from 27 test functions to 29 and section 7.4
still said 27. The inventory guard caught its own file drifting, which is the mechanism
working rather than an incident: 27 -> 29 and the Python total 3227 -> 3229.
```

**A discarded measurement, recorded because discarding one silently is the habit worth refusing.**
Slice 2's mutation M2 first reported 10 of 10 tests failing, which would have been evidence that the
guard was indiscriminate rather than sharp. Re-run in isolation it failed exactly 4 — the claim path
plus the uniqueness assertion at both viewports. The first run had begun while Vite was still
reloading the file the previous mutation had just restored. **A mutation that appears to kill
everything is evidence about the harness, not about the guard**, and the 10-of-10 figure is recorded
here as discarded rather than quietly replaced.


## TG17.13 The source-edit audit and the fifth-adapter gate - COMPLETE (2026-09-04, ed-dev)

Two slices. Slice 1 wrote the audit a test name had been standing in for; slice 2 built the channel
that carries it into the release record. `synthetic_fifth_adapter` reads `PASS` for the first time
since TG17.10 registered it.

**What slice 2 decided, and why it is not TG17.12's shape applied twice.** The gate has two halves
and only one of them is recorded. The source-edit audit is **read live** on every call: it is
exact, it reads committed source, it costs milliseconds, and a recording of a fact that can be
recomputed is only a way to be wrong later. The acceptance run is **recorded**, because it cannot
be read from the gate's side at all - TG17.3 requires the fifth adapter to be defined in a module
the application never imports, so it lives in `src/tests/test_adapter_registry.py` and nothing
under `src/core` may reach it. Copying the recording shape onto both halves would have been the
symmetric answer and the wrong one.

**Deciding sits on the reading side.** The test supplies apparatus only it owns - the adapter, the
native record, the window - and `measure_extension_conformance` performs all eight checks and
decides whether they passed. A test that deleted its own assertions changes nothing about what is
recorded. What it can still do is stop calling the recorder, and the answer to that is `NOT_RUN`.

**What the gate now says, recorded output:**

```text
> .\.venv\Scripts\python.exe -c "from src.core.experiment_qualification import
    qualification_plan ..."
NOT_RUN  offline_matrix
NOT_RUN  restart_recovery
PASS     browser_no_glue
PASS     synthetic_fifth_adapter
PASS     calendar_calibration
REFUSED  scale_shape_calibration
NOT_RUN  live_sources

A synthetic fifth domain with different mathematics - a monotone rank channel, not the shared
standardized level - reached the registry, the control schema, the conformance kit and the
domain-blind mining seam from a module the application never imports, passing all 8 checks
(recorded 2026-09-03T22:53:39.149657Z). No framework source names it, so installation required
0 framework edits across 17 generic surfaces. Standing adapter-specific glue in those surfaces
is 1 and is reported rather than asserted: frontend/src/components/AcquisitionView.tsx:243
renders a bespoke planner for reanalysis.
```

Three of seven gates clear. The verdict is unmoved at `NOT_RELEASEABLE`: `offline_matrix` and
`restart_recovery` are `NOT_RUN` in a cold plan, `scale_shape_calibration` is `REFUSED`, and
`live_sources` is the last unmeasured scientific gate.

**A guard passed because it could not see what it was checking, for the fifth time.** The
`defined_outside_the_application` check first compared `translate.__module__` against dotted package
prefixes. Under pytest's import mode that string is the bare `test_adapter_registry`, which starts
with none of them - and would have started with none of them whatever the module was called. It now
resolves the defining *file*, makes it repository-relative and checks it against directory roots; a
callable whose source cannot be located is a **refusal** rather than a pass, because an unanswerable
question is not a satisfied one. D64, D74, D75, slice 1's vacuous registry scan, and now this.

**The completeness guard TG18.5 built did its first real work.** Marking TG17.13 DONE in the roadmap
made `test_every_completed_cross_domain_phase_has_a_verification_entry` fail, naming TG17.13 and
nothing else, before this entry existed:

```text
E  AssertionError: these cross-domain phases are marked complete in roadmap_cross_domain.md but
   have no '## <phase> ' entry in VERIFICATION.md: TG17.13
1 failed, 3 passed, 25 deselected, 1 warning in 0.66s
```

**A mutation pass with no baseline cannot tell a killed mutant from a broken suite, and this one
reported a false CAUGHT before it was fixed.** The first run of the script produced:

```text
M1 staleness ignored                              CAUGHT      2 failed, 49 passed
M2 a missing required check tolerated             CAUGHT      2 failed, 49 passed
M3 defining_source falls back to the module name  NOT CAUGHT  49 passed
M4 DEFAULT widened into glue                      CAUGHT      4 failed, 45 passed
M5 another adapter accepted for measurement       CAUGHT      2 failed, 49 passed
```

The second run, after I added a test for the branch M3 mutates, reported M3 as `CAUGHT  1 failed,
50 passed` - and that was **wrong**. The single failure in that run was the new test itself, which
was broken: it asserted `__module__ == "builtins"` for a callable compiled from a string, and the
exec scope carried no `__name__`, so the attribute was `None`. Every mutation's count in both runs
was inflated by one failing test, and M3 was never actually killed. The script now runs an
unmutated baseline first and refuses to report unless it is green:

```text
BASELINE (unmutated)                              GREEN   51 passed, 2 warnings in 33.13s
M1 staleness ignored                              CAUGHT  1 failed, 50 passed
M2 a missing required check tolerated             CAUGHT  1 failed, 50 passed
M3 defining_source falls back to the module name  CAUGHT  2 failed, 49 passed
M4 DEFAULT widened into glue                      CAUGHT  1 failed, 50 passed*
M5 another adapter accepted for measurement       CAUGHT  1 failed, 50 passed
M6 the live audit inherited from the recording    CAUGHT  1 failed, 50 passed

all modules restored byte-identically
```

*M4 reported `4 failed, 47 passed`. It is the cross-check that the contract binding bites: widening
`GLUE_KINDS` in `extension_audit.py` moves the contract digest, which un-measures the recording,
which returns the gate to `NOT_RUN` - so a recording cannot be made green by widening the list it is
judged against.

**Fixing the broken test found a real defect in the code under it.** `inspect.getsourcefile` returns
a pseudo-filename such as `<no file>` for code compiled from a string whose module carries a loader,
and on Windows `Path("<no file>").resolve()` yields an absolute path inside the repository without
raising - so `defining_source` would have reported a located source that cannot be read, and
`defined_outside_the_application` would have passed on it. The resolved path must now be an existing
file.

**A second real defect, found by a wall-clock assertion I did not write.** TG17.12's
`test_the_gate_reads_the_recording_and_does_not_run_the_calibration` asserts that
`qualification_plan()` assembles in under a second, and it failed in a long combined run. The
reading was contention, but investigating it found that the plan was calling
`read_extension_conformance()` **twice** - once for the gate and once for the record - walking
seventeen sources twice for an identical answer and doubling an HTTP route's added cost. It is read
once and passed to the gate. Measured after the fix: 0.183s, 0.146s, 0.145s over three consecutive
assemblies.

**Verification, all actually run:**

```text
> .\.venv\Scripts\python.exe -m pytest src/tests/test_extension_evidence.py ^
    src/tests/test_experiment_qualification.py -q
51 passed, 2 warnings in 33.13s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_calibration_record.py ^
    src/tests/test_browser_evidence.py src/tests/test_adapter_registry.py -q
46 passed, 1 warning in 13.03s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_experiment_qualification.py ^
    src/tests/test_extension_audit.py src/tests/test_frontend_contract.py -q
217 passed, 5 warnings in 33.92s
```

The reported counts exceed the source function counts because several of these modules parametrise:
`test_extension_evidence.py` carries 25 test functions and `test_experiment_qualification.py` 22.

Nothing in this phase is evidence about the world. Both slices are statements about this
repository's source, and about whether one synthetic adapter reached the seams a third party's
adapter would have to reach. The full backend suite was not rerun; the last measured result remains
3,536 passed.

## TG17.14 Four-domain live-source acquisition — DONE, GATE PASS (2026-09-04, `ed-dev`)

The first run in this project's history in which real data was pulled from public archives in more
than one domain. Authorised explicitly by the maintainer; both locks satisfied.

```
> python -m src.core.live_source_evidence run --order-book-record ... ^
    --confirm-network-access I_AUTHORIZE_BOUNDED_ARCHIVE_REQUESTS

reanalysis       PASS   network_used=True     324 values     ERA5 t@850hPa via CDS, 9x9x1x4
argo_float       PASS   network_used=True     10,218 values  93 profiles, Argo GDAC
tess_lightcurve  PASS   network_used=True     18,279 values  MAST SPOC, TIC 261136679 sector 1
order_book       PASS   network_used=False    34,560 values  2,880 records, local, no network
GATE: PASS   reasons: []
```

```
> python -c "qualification_plan()"
offline_matrix             NOT_RUN      blocking=True
restart_recovery           NOT_RUN      blocking=True
browser_no_glue            PASS         blocking=False
synthetic_fifth_adapter    PASS         blocking=False
calendar_calibration       PASS         blocking=False
scale_shape_calibration    REFUSED      blocking=True
live_sources               PASS         blocking=False
VERDICT: NOT_RELEASEABLE
```

**Five of seven gates pass and the release is still refused.** That is the result, not a
disappointment: `scale_shape_calibration` is TG17.11's declared scientific limit, and reaching four
archives in four domains does not buy past it.

**Two defects found by first contact with real data.**

*   **D94.** The reanalysis probe republished `streaming_content_hash`'s 32-character cache key in a
    field named `sha256`. `_is_sha256` requires 64, so the gate returned `NOT_RUN` with
    `reanalysis PASS has no non-empty content-addressed record` — declining to credit a pass whose
    content binding it could not verify. Fixed by separating the two contracts:
    `streaming_content_sha256` returns the whole digest and `streaming_content_hash` is defined as
    its 32-character prefix, **unchanged in value**, so no frozen atmospheric receipt moved.
*   **D95.** SPOC emits one row per cadence including those with no photometry, and those carry a
    non-finite `TIME`. The real sector-1 product for TIC 261136679 has **815 of them in 20,076
    rows**. `LightCurveCollection` refused the lot, correctly. Untimestamped cadences are now
    dropped and counted, with `unclocked_samples_dropped` carried into the gate record.

```
> python -m pytest src/tests/test_photometry.py -q
12 passed, 1 skipped, 2 warnings in 4.03s

> python -m pytest src/tests/test_zarr_source.py src/tests/test_cds_source.py ^
    src/tests/test_live_source_evidence.py -q
109 passed, 1 skipped, 5 warnings in 65.61s

> python -m pytest src/tests/test_experiment_qualification.py -q
26 passed, 2 warnings in 30.23s
```

**A correction recorded rather than quietly fixed.** A first reading of D95 claimed the
duplicate-clock guard had been blinded by NaN — "the sixth guard to pass by not looking". That was
**wrong**, and the test written to pin it refused to pass. numpy sorts NaN to the end, so the
finite prefix was always compared correctly and `[1.0, nan, 1.0]` refused before this slice exactly
as after. The claim was removed from the code comment and the test rewritten to assert what is
true. D95 is one defect, not two.

**Two tests changed because the world changed, not to make them green.**
`test_qualification_record_is_self_hashed_and_tampering_is_detected` tampered by writing
`status = "PASS"` to the last gate; once `live_sources` actually passed, the tamper wrote the value
already there and the record verified — a tamper test failing for the one reason it must not. It
now writes a sentinel that cannot collide with a real status.
`test_http_plan_and_rehearsal_keep_the_unrun_gates_visible` asserted some gate remained `NOT_RUN`
after the rehearsal; the survivor is now the `REFUSED` one, and the assertion was rewritten to its
actual intent — the rehearsal must not present every gate as passing.

**MAST is intermittent, and this record is three successes in five attempts.** On 2026-09-04
`Mast.Caom.Filtered.Position` answered three times and timed out twice after two bounded 90-second
attempts. A timeout records `REFUSED` with `network_used: null` — not `false`, because after a
failed call whether bytes moved is unknown. Stated because a reader deciding whether to depend on
this gate should know the binding constraint is a metadata service rather than the science, and
because reporting only the passing run would misdescribe the archive.

**The order-book record is not in this repository and that is deliberate.** Its domain declares
that redistribution of raw depth is restricted, so the gate binds it by `sha256`
(`b2fd0c2cfe426cf8c3c142ca2195da9564a6e65517c3cf8287afa4968edfe94f`) and `data/market_records/`
carries a README with the source URL and exact transformation instead of the bytes.

Nothing here is a scientific result. The record's own `claim_boundary` says so: this qualifies
bounded acquisition and record binding in four domains, and is not evidence, replication or claim
promotion.

## Full backend suite, after T4F.1 and TG17.14 (2026-09-04, `ed-dev`)

```
> .\.venv\Scripts\python.exe -m pytest src/tests/ -q
3738 passed, 4 skipped, 1 xfailed, 6 warnings in 2704.67s (0:45:04)
exit 0
```

The first full-suite measurement since before TG17.11. It replaces **3536**, which was measured at
T4E.2 and had been correctly quoted as stale by every slice since. The four skips are the standing
opt-in live checks: GCS read, store probe, Argo acceptance and TESS/MAST acceptance.

**One qualification on how it was measured.** Documentation-only edits landed while the run was in
progress -- the TG17.15 phase and this file's own TG17.14 entry. No source changed during the run,
so the figure is a true count of the code, but the 29 documentation guards were re-run standalone
afterwards and passed rather than being trusted from inside the long run.

The count rose by 202 over T4E.2's 3536. Contributing test functions added this session: 18 for
T4F.1's event substrate, 3 for D95, 1 for D94, 1 for D93. The remainder predates this session and
was verified against targeted suites at the time, which is exactly the accounting the standing
instruction against quoting an unmeasured larger figure exists to force.

## TG17.15 slice 1 The estimand declared before it is answered (2026-09-04, `ed-dev`)

TG17.11 established that the scale/shape null cannot reject at a reachable inventory size. This
slice establishes why, and the why is not about compute.

```
> python -m pytest src/tests/test_correspondence_estimand.py -q
14 passed, 1 warning in 1.97s
```

**The measurement the decision rests on**, computed from the null's own enumerator rather than
written down:

```
k=4  |A|=9       partners=3  floor=0.2500  miscounted=0.1000000
k=5  |A|=44      partners=4  floor=0.2000  miscounted=0.0222222
k=6  |A|=265     partners=5  floor=0.1667  miscounted=0.0037594
k=7  |A|=1854    partners=6  floor=0.1429  miscounted=0.0005391
k=8  |A|=14833   partners=7  floor=0.1250  miscounted=0.0000674
```

The valid reassignments are the derangement numbers and grow factorially; the partners one member
can receive are `k - 1`. A member's statistic depends only on its partner, so 14,833 draws produce
seven distinct statistic values. Reading the reassignment count as the reference-set size is
**anticonservative by more than a factor of a thousand at k = 8, in the direction that makes a null
easier to reject**. That is the most likely error in any reimplementation and is now pinned by
test rather than left as a warning.

**Two consequences, both measured.** Raising `MAX_REASSIGNABLE_PAIRINGS` cannot help: it raises the
reassignment count and leaves the partner count at `k - 1`, so the TG17.11 gap is structural rather
than computational. And the minimum resolvable joint family has **no margin at all** -- at 105, one
member a single step off the floor drops rejections to zero of 105, against 105 of 106 one size up.
Reaching 105 would not have produced a usable instrument.

**What the chosen estimand buys**, solved against the real correction:

```
tested m:      1    3    5    6   10   20
pool N:       19   36   45   48   58   71
```

and the property the joint estimand cannot have: at a pool of 48 with six tests, one member off the
floor rejects nothing; at 96 it rejects five of six. The test count is identical in both, so margin
was bought without multiplicity. Under `joint_structure` the only way to lower the floor is to grow
the family, which grows the correction burden with it.

Benjamini-Yekutieli is recorded with its reason rather than inherited: family members share a
partner inventory and are dependent, BY is valid under arbitrary dependence and BH is not.

`joint_structure` is registered **inadmissible** rather than omitted, so it is refused by name with
its reason attached; it is the natural first design and its failure is not visible from inside it.
`require_declared_estimand` refuses an undeclared estimand rather than defaulting, because the two
questions have different reference sets and can disagree on the same data.

This slice declares the question and answers nothing. Its claim boundary says it is not a
calibration, not a power analysis on real records, not a partner pool and not a result, and a guard
asserts no power key appears in its report. The gate is unchanged: `scale_shape_calibration` still
reads `REFUSED` and the verdict is still `NOT_RELEASEABLE`.

## Benchmark suite, verified by running (2026-09-04, `ed-dev`)

```
> python -m src.benchmarks
PASS 43   FAIL 0   NOT_YET_RUNNABLE 0
exit 0
```

Recorded because the figure had been quoted from the document rather than measured. **Thirteen of
the twenty-four registered benchmarks are null benchmarks**, where the correct answer is that there
is nothing there. Two of their reports are worth transcribing, because they are the instrument
catching what would otherwise be its most publishable result:

```
4C.r11_raw_is_deceptive   raw (un-anomalised) scale energies correlate at r = 0.999,
                          p = 2.02e-219 - a strong 'finding' that is purely the calendar.
                          This is the trap R11 exists for.

4F.refusal_calendar       unguarded, the same pipeline confirms 4 relationship(s) at
                          q = 0.0104 (strongest member naive p 4.5e-66, ESS-corrected
                          1.1e-12, so neither R12 nor the surrogate refuses it); with the
                          calendar fitted on train and removed from both partitions,
                          2 frozen and none confirmed, smallest q 0.105
```

## TG17.15 slice 2 The partner pool, and circularity made inexpressible (2026-09-04, `ed-dev`)

```
> python -m pytest src/tests/test_partner_pool.py -q
20 passed, 1 warning in 0.25s
```

The roadmap named this the slice most likely to go quietly wrong, and the reason is asymmetry: the
enumeration bound it replaces announced itself by refusing, whereas a badly curated pool returns a
confident number.

**The defence is structural, not disciplinary**, following R22's own precedent -- prose cannot
corrupt a claim level because the gates read typed fields, not because anyone is careful. Here,
there is no value in the module that is a function of two records. `build_partner_pool` reads
`RecordProfile` objects and never records, and a test asserts the profile's field set exactly, so a
similarity, distance or affinity field cannot be added without failing. Admission on resemblance to
the record under test is not a mistake this module can express.

**Mutation testing, four applied and four caught:**

```
CAUGHT  silently drop refused candidates instead of recording them
CAUGHT  allow native_seconds to be banded
CAUGHT  skip the check that the observed partner clears its own contract
CAUGHT  let a pool below the resolvable size through
uncaught: 0
```

**Acceptance pool**, six tested correspondences, 60 candidates offered:

```
admitted 58, refused 2, required 48, resolution floor 0.01695
  REFUSED TOO_SHORT  effective_sample_size = 20 against the partner's 400, ratio 0.05 outside [0.5, 2]
  REFUSED LEAKED     shares provenance 'srcLEFT' with the left member
  native_seconds     pool 6,092 to 141,160 s, partner 22,334 s at the 22nd percentile
```

The duration spread is deliberate: `native_seconds` is recorded and never banded, because
scale/shape mode exists to compare shapes across native durations and a pool banded on duration
would refuse the comparison the mode is for.

**What the receipt refuses to claim.** Passing every declared band is a **necessary condition for
exchangeability and not a sufficient one**. It establishes that no declared marginal visibly
violates it; an unmeasured property may still differ systematically and the pool cannot know. That
sentence is in the receipt and asserted by test, because a pool that reads as a proof of
exchangeability would be more dangerous than no pool at all.

No null was run and no calibration was performed. The gate is unchanged:
`scale_shape_calibration` still reads `REFUSED` and the verdict is still `NOT_RELEASEABLE`.

## TG17.15 slice 3 The exact pool-substitution null, and a defect found in its own first draft (2026-09-04, `ed-dev`)

```
> python -m pytest src/tests/test_pool_substitution_null.py -q
37 passed, 1 warning in 0.54s

> python -m pytest src/tests/test_correspondence_estimand.py src/tests/test_partner_pool.py \
      src/tests/test_pool_substitution_null.py -q
80 passed, 1 warning in 6.31s
```

**Mutation testing, ten applied and ten caught:**

```
CAUGHT  drop the observation from its own reference set (denominator N instead of N+1)
CAUGHT  stop counting ties as evidence against rejection
CAUGHT  silently skip an alternative whose record payload is missing
CAUGHT  let a non-finite statistic through instead of refusing
CAUGHT  default the orientation instead of requiring it to be declared
CAUGHT  allow the family to be narrowed after the pools were sealed
CAUGHT  allow the same correspondence to be counted twice
CAUGHT  allow the observed partner to sit inside its own pool
CAUGHT  skip the determinism check on the statistic
CAUGHT  report the all-genuine case as if it were the family's real power
uncaught: 0
```

**The defect the acceptance run found, and the correction it forced.** The first `resolution()`
reported `every_member_can_reject_at_its_own_floor`, which is what `minimum_pool_size` is defined
against. That question places *every* member at its floor simultaneously -- the most favourable
world there is -- and reporting it alone showed a green light for a family that cannot produce a
finding:

```
declared family m = 6, admitted pool N = 58, required N = 48, floor = 1/59 = 0.01695

family: 6 members, 0 rejected after benjamini_yekutieli at alpha 0.05
  L0   p = 0.01695  q = 0.08305  no  (at floor)
  L1   p = 0.01695  q = 0.08305  no  (at floor)
  L2   p = 0.01695  q = 0.08305  no  (at floor)
  L3   p = 0.96610  q = 1.00000  no
  L4   p = 0.23729  q = 0.87203  no
  L5   p = 0.54237  q = 1.00000  no
resolution: every member can reject at its own floor = True, worst floor 0.01695
            sparsest detectable: 5 of 6 genuine (83%); a half-genuine family of this
            size would need N = 97 rather than the 48 it was sized for
```

Three correspondences sat at the exact floor and none was rejected. Members that do not correspond
consume the Benjamini-Yekutieli step-up ranks the genuine ones need, so a pool sized by
`minimum_pool_size` is sized for a world in which nothing fails. `sparsest_detectable_count` now
measures the fewest genuine members these actual pools could ever reject, and the family receipt
carries it in words:

```
powered_for: this family can produce a rejection only if at least 5 of its 6 declared
correspondences are genuine. Members that do not correspond consume the Benjamini-Yekutieli
step-up ranks the genuine ones would need
```

`minimum_pool_size_for_detected_fraction` sizes a pool for that world before acquisition, and the
gap is large:

```
m = 6, fraction 1.000 -> N = 48
m = 6, fraction 0.833 -> N = 58
m = 6, fraction 0.500 -> N = 97
m = 6, fraction 0.167 -> N = 293
```

**Refusals, each by name:**

```
narrowed family          NullRefusal: expected exactly the 6 correspondences these pools were
                         sealed for. Narrowing the declared family ... is selection
undeclared orientation   InvalidParameterError: expected one of ['larger_is_more_similar',
                         'smaller_is_more_similar'], declared explicitly
missing payload          NullRefusal: r0002 is in the sealed pool but absent from the records
                         offered ... anticonservative and leaves no trace
```

**What was checked about the arithmetic, and what was not.** Statistics drawn i.i.d. are
exchangeable with their pool *by construction*, so the rank is uniform on the attainable grid:

```
40,000 trials, N = 58
  P(p <= 0.05) = 0.0338   nominal 0.05, attainable grid step 0.01695
  mean p       = 0.5081   uniform on the grid has mean 0.5085
```

The rate sits below nominal because the attainable grid is coarse: `0.0338 = 2/59` is the largest
grid point at or below 0.05. This checks the **arithmetic only**. It does not establish that a
curated pool of real records is exchangeable, and it is not a calibration. No false-positive rate
has been measured for this null on records with no planted correspondence; T4C.5h remains the
standing proof that a null can preserve exactly the property it is named after and still get the
distribution wrong, at a measured 0.765 against a nominal 0.05. That is slice 4.

The gate is unchanged: `scale_shape_calibration` still reads `REFUSED` and the verdict is still
`NOT_RELEASEABLE`.

**The full suite, and one failure that is not this slice's.**

```
> python -m pytest src/tests -q -p no:randomly
3 failed, 3815 passed, 4 skipped, 1 xfailed in 3739.00s (1:02:19)

FAILED test_acquisitions_api.py::test_a_server_restart_marks_active_cds_work_interrupted...
FAILED test_documentation.py::test_every_source_module_appears_in_architecture
FAILED test_documentation.py::test_documented_test_counts_match_the_source
```

The two documentation failures are stale: that run began at 17:56, before this slice's sections
were written. The dedicated re-run afterwards passed all 29 documentation guards in 10:24.

The acquisitions failure is **not attributable to this slice and was not reproduced**. It passed
alone (41.7s), as a whole file (13 passed), and with all 124 test modules collected but only it
selected. `test_acquisitions_api.py` is the *first* file pytest collects, so nothing added here
executed before it, and this slice touches no API, database or threading code. The mechanism is
visible in the test itself: `_wait_for_job` polls 200 times at 10 ms, a **two-second wall-clock
budget** on a background worker thread, inside a suite that ran for 62 minutes. That is a
timing-sensitive test rather than a defect in what it tests, it is pre-existing, and it is recorded
here rather than fixed inside an unrelated slice.

Gate re-read directly after the documentation pass rather than assumed:

```
Scale/shape null calibration a REFUSED
verdict: NOT_RELEASEABLE
```

## TG17.15 slice 4 The calibration, an expectation that was the defect, and four mutation gaps (2026-09-05, `ed-dev`)

```
> python -m pytest src/tests/test_pool_calibration.py -q
50 passed, 1 warning in 77.69s (0:01:17)
```

**The recorded calibration.** Five declared cases at 200 realisations each, `m = 6`, pools of 61
to 470, run through the real `correspondence_family` with the real `shape_recurrence` statistic.
Ran in 1,351 s (22:31).

```
calibrated: True | failing: [] | uncertified: []

case                       refus  family-wise  1-sided upper   uncorrected     rank 1      at floor,
                                                                per member              not rejected
planted_correspondence       0     200/200         --         1200/1200      1189/1200         0
no_correspondence            0       1/200       0.0235         55/1200         4/1200         3
shared_grid_alias            0       2/200       0.0311         40/1200         3/1200         1
clean_partner_noisy_pool     1       0/199       0.0149         54/1194         5/1194         5
unresolvable_inventory     200        --           --              --             --          --

realisations 200 | needed to certify alpha 59 | uniformity resolution 0.0962
```

Every family-wise bound clears alpha 0.05, and `unresolvable_inventory` refused 200 of 200 rather
than scoring. The whole calibration was run again after the mutation gaps below were closed, in
1,487 s, and **every figure above reproduced exactly** -- the fixes changed what is checked, not
what was measured. The eight ladder rungs published one identical `inventory_sha256`. **This null does not repeat T4C.5h**, whose surrogate preserved exactly the property
its method was named after and still measured a family-wise rate of 0.765 against a nominal 0.05.

**The tail is not the whole check.** Under exchangeability the observation's rank among its `N`
alternatives is uniform on `{1, ..., N + 1}` *exactly*, so the entire distribution is predicted in
advance rather than only its 5% tail -- a rate can look nominal while the distribution is wrong.
Measured on **one member per realisation**, because members of a family share a candidate
inventory and are dependent, which is the case Benjamini-Yekutieli was chosen for and also the
reason a pooled Kolmogorov-Smirnov p-value would not be a test:

```
                            independent (n = 200)          pooled diagnostic (n = 1200)
no_correspondence           mean 0.5214  KS 0.0650  p 0.35   mean 0.5104  KS 0.0326
shared_grid_alias           mean 0.5423  KS 0.0727  p 0.23   mean 0.5095  KS 0.0261
clean_partner_noisy_pool    mean 0.5218  KS 0.0684  p 0.30   mean 0.5086  KS 0.0273
```

**Slice 2's claim boundary, asked for a number.** It states that passing every declared band is a
*necessary* condition for exchangeability and not a sufficient one. Two adversarial nulls attack
exactly that: `shared_grid_alias` gives every record the same strong artefact keyed to position
within its own cycle -- a shared instrument cadence, which is not a shared shape -- and
`clean_partner_noisy_pool` draws the observed partner systematically cleaner than the inventory
while still inside every band, so the alternatives its own contract admits are noisier than it is
and its statistic should ride higher. Both hold the declared rate. The sentence survives contact
with a measurement; it is not thereby proved for real records, and the claim boundary still says so.

**The defect: the first recorded run failed, and the expectation was what was wrong.**

```
calibrated: False | failing: ['planted_correspondence']

planted_correspondence   memb=1199/1200   rank1=1189/1200   pools [61, 470]
declared expectation: {'outcome': 'rejects', 'minimum_detection': 1.0, ...}
```

`minimum_detection: 1.0` demanded that every one of 1,200 members reject. Eleven members did not
have the true partner at rank 1: with pools of up to 470 alternatives a chance candidate will
occasionally outrank a real correspondence. **Requiring otherwise is requiring a test with no
type-II error**, which is the point-estimate mistake already fixed for the error rates, left
standing in the opposite direction after they were fixed. It is replaced by two criteria derived
from what the case is rather than from what a perfect run would look like:

* the **statistic** must rank the true partner first for at least 90% of members, judged on a
  one-sided *lower* confidence bound -- the mirror of how an error rate is judged on an upper one;
* every member it did rank first must reject: `maximum_unresolved_at_floor = 0`, parameter-free,
  isolating the failure worth catching, which is being short of pool rather than short of effect.

The rerun's measured numbers are **identical** -- 1,199 of 1,200 and 1,189 of 1,200 -- and only the
verdict moved, which is what should happen when the expectation rather than the run was at fault.
On the new criteria the rank-one lower bound is 0.9849 against the declared 0.90, and no member
ranked first failed to reject. The failed run is recorded here rather than replaced by the one that
passed.

**Detection is a curve, and it falls off a cliff.** The inventory, the left members and every
declared marginal are held fixed across rungs -- enforced by digest -- so a rung differs from its
neighbour in the planted correlation and nothing else:

```
planted w   member   family   uncorrected   at rank 1
   1.00      1.000    1.000      1.000        0.992
   0.95      0.996    1.000      1.000        0.658
   0.92      0.958    1.000      1.000        0.346
   0.88      0.354    0.600      0.988        0.125
   0.84      0.062    0.225      0.817        0.083
   0.80      0.037    0.175      0.575        0.054
   0.60      0.004    0.025      0.158        0.013
   0.00      0.000    0.000      0.050        0.004
```

At `w = 0.88`, **98.8% of members have an uncorrected p at or under 0.05 and 35.4% survive
correction**. That gap is not noise; it is slice 3's `sparsest_detectable_count` prediction
confirmed by measurement. When every member corresponds, Benjamini-Yekutieli divides at rank 6 and
a pool of 48 suffices; when the family is mixed, a surviving member must clear
`alpha / (m * H_m) = 0.0034`, which no pool below 293 can reach. So every rung reports
`members_at_their_floor_that_did_not_reject`: a member the statistic ranked first that still fails
was **short of pool, not short of effect**, and `minimum_pool_size_for_detected_fraction` already
names the pool it would have needed. The bottom rung is a true null reached through the ladder
rather than through the case, and the two agree.

**A cost of the admission contract that nothing had measured.** `AdmissionContract` refuses to band
`native_seconds`, because bounding it would refuse the comparison scale/shape mode exists to make.
But `cadence_seconds` *is* banded, and cadence is native duration divided by a bounded row density,
so the cadence band narrows native duration **transitively** and nothing said so:

```
offered 1500 candidates spanning 4.05 decades of native duration
admitted per pool  229  268  348  299  396  243   (yield 15% to 26%)
decades admitted  1.18 1.33 1.47 1.42 1.45 0.95
```

A pool is far more native-scale homogeneous than the inventory it was drawn from. That helps
exchangeability and constrains the mode's reach; either way it is now reported by
`admission_yield` and guarded by a test rather than left to be inferred.

**Mutation testing found four gaps before it found none.**

```
first pass:  11 CAUGHT, 4 MISSED
  MISSED  stop holding a scoring case to its refusal ceiling
  MISSED  stop counting members ranked first that did not reject
  MISSED  redraw the inventory between rungs of the effect ladder
  MISSED  infer the phase window from the pair instead of declaring it

second pass: 15 CAUGHT, 0 MISSED
```

Three of the four were real. The refusal ceiling and the at-floor-unresolved count were never the
*binding* reason in any test -- every case that exercised them failed for another reason first --
so both are now tested directly against the decision rule. The ladder's "held fixed" claim was
enforced by digest on `build_realisation` and only described on `detection_profile`; each rung now
publishes an `inventory_sha256` and a test requires them equal.

The fourth was an **equivalent mutant**, and is recorded as one rather than worked around.
`_shared_phase` caps the comparison at each record's own declared support, so passing
`cycles=SPAN_CYCLES` and inferring the shorter record's span give the same number for every fixture
this module builds -- the test asserting they agreed **could not fail**. The number was never
wrong; the check was vacuous. The declaration is now something that can fail: `_statistic` refuses
a record that does not cover the declared window, so a fixture that silently got shorter is refused
rather than compared over a window nobody declared.

**What is not claimed.** A measured false-positive rate and detection curve for this null on
records whose answers were fixed before the method ran. It is evidence that the arithmetic and the
exchangeability hold together **on these fixtures**. It is not evidence that a pool of real records
is exchangeable: that rests on properties the fixtures were given by construction, and a real
inventory would have to be shown to have them.

**The full backend suite, and the flake that was contention after all.**

```
> python -m pytest src/tests -q
3868 passed, 4 skipped, 1 xfailed, 6 warnings in 2481.74s (0:41:21)
```

Nothing failed. `test_acquisitions_api.py::test_a_server_restart_marks_active_cds_work_interrupted_
for_explicit_resume` -- recorded under slice 3 as timing-sensitive and not reproducible in
isolation, and failing again in slice 3's rerun -- **passed here**. Both runs where it failed were
heavily contended by concurrent calibration work; this one had the machine to itself. That is the
behaviour a test polling a background worker on a two-second wall-clock budget would show, and it
is now the third piece of evidence for that reading rather than an assumption.

One precision about this figure: the suite began at 09:06 and two files were tidied at the margin
while it ran -- a docstring line, an `__all__` rewrap and an import rewrap, in
`pool_calibration.py` and its test. Nothing else in the tree imports either file, and
`test_pool_calibration.py` was rerun in full afterwards at 50 passed. So the suite figure covers
every module, and the tidied bytes are covered by that separate run rather than by this one.

The gate is unchanged, and deliberately so -- carrying this measurement into it is slice 5's work,
by checked supersession rather than by editing the old refusal away:

```
Scale/shape null calibration and planted power REFUSED
verdict: NOT_RELEASEABLE
```

## TG17.15 slice 5 A refusal superseded rather than deleted, and a gate that still refuses (2026-09-05, `ed-dev`)

```
> python -m pytest src/tests/test_calibration_record.py src/tests/test_experiment_qualification.py -q
59 passed, 2 warnings in 63.67s (0:01:03)
```

**The recording.** `calibrate_pool_substitution` run deliberately and written to
`calibration/scale_shape_calibration.json`, bound to its declared contract and to the source of the
six modules that decide what it measures. 1,031 s for the five cases and the eight-rung ladder.

```
recorded_utc 2026-09-04T22:41:28.066528Z | all_met True | seed 20260904 | realisations 200

case                        refus  fam-wise 1-sided up   rank1 lo       KS       p        pools
planted_correspondence          0   200/200     1.0000     0.9849   0.9929   0.000    [61, 470]
no_correspondence               0     1/200     0.0235     0.0011   0.0650   0.352    [61, 470]
shared_grid_alias               0     2/200     0.0311     0.0007   0.0727   0.229    [61, 470]
clean_partner_noisy_pool        1     0/199     0.0149     0.0017   0.0684   0.296    [57, 443]
unresolvable_inventory      200/200        --         --         --       --      --           --

ladder rungs 8 | inventory digests 1
```

Every figure reproduces slice 4's recorded run exactly, on a different day and in a separate
process. The planted row's `KS 0.9929, p 0.000` is the alternative and not the null: that case is
supposed to depart from uniformity, and a reader is entitled to see the number rather than a blank.

**The reproduction witness, which is what makes a 1,031-second measurement checkable in seconds.**
The calendar calibration is cheap enough that the live measurement runs in the test suite on every
pass and a guard compares it against the recording. This one is not, and pretending otherwise would
have been the easy lie. `calibrate_case` runs realisation `i` at `seed + i`, so a three-realisation
run at the recorded seed is not a *similar* measurement to the recorded one -- it is its leading
prefix, exactly. The recording carries a digest of those realisations per case, and the suite
recomputes all five:

```
witness planted_correspondence     seed 20260904 realisations 3 5fb8c2de6809f563
witness no_correspondence          seed 20260904 realisations 3 0551a40745b11881
witness shared_grid_alias          seed 20260904 realisations 3 55881cfdd34627aa
witness clean_partner_noisy_pool   seed 20260904 realisations 3 4fbb8db56ae0eb0d
witness unresolvable_inventory     seed 20260904 realisations 3 863a41d4047acd08
```

`no_correspondence`'s digest was computed independently, in a separate process before the recording
existed, and is the same. The case that refuses every realisation witnesses `["REFUSED", ...]`
rather than the empty list -- an empty digest agrees with every other run that also produced
nothing, which is the one thing a witness must not do. This is a **smaller** claim than the
calendar recording's, and the module says so in those words rather than implying an equivalence.

**The supersession, recomputed rather than remembered.**

```
> python -c "from src.core.calibration_record import scale_shape_supersession; ..."
supersession SUPERSEDED
still_true True {'minimum_resolvable_family': 105, 'largest_drawable_inventory': 8}
minimum_pool_size 48
```

`scale_shape_supersession` does not quote TG17.11's claim. It recomputes it from the two primitives
that claim turned on, using the same functions the gate's applicability section uses, and a guard
asserts the two agree. Three outcomes, and the third is the whole point of the word "checked":

* `SUPERSEDED` -- the old claim still holds when recomputed, and a bound successor recording passed.
* `NOT_SUPERSEDED` -- the successor is absent, stale or failed; the old refusal stands alone.
* `VOID` -- the predecessor's claim has **stopped being true**. It was retired on its own terms
  rather than superseded, and the record must be re-derived rather than kept.

A guard drives it into `VOID` by lifting `MAX_REASSIGNABLE_PAIRINGS` past the resolvable size and
asserts the record says the limit removed itself. Nothing in this repository produces `VOID` today,
which is exactly why it needed a test: a supersession nobody recomputes is a sentence about the
past that keeps agreeing with itself.

**The gate, read directly. It still refuses, and that is the result.**

```
SUPERSEDED. The superseded claim (TG17.11 slice 5,
src.benchmarks.shape_fixtures:calibrate_shape_family) is recomputed on every plan and still holds:
no family smaller than 105 pairings can reject under benjamini_yekutieli at alpha 0.05, the
declared null draws only from inventories of at most 8, and 2 of 2 declared domain families are
refused by the null before size is reached. Superseded for applicability by TG17.15 slices 1-4
(per_correspondence, src.benchmarks.pool_calibration:calibrate_pool_substitution), recorded
2026-09-04T22:41:28.066528Z: 6 correspondences need a pool of 48, which the calibration's own pools
reach. Blocked by declared_inference (decidable here); pool_exchangeability_on_real_records (not
decidable here).

status: REFUSED | verdict: NOT_RELEASEABLE
```

**Both blockers are computed, and one of them is admitted to be undecidable here.**
`declared_inference` is read back from the six frozen manifests rather than restated: every one
declares `scale_partner_reassignment` at 200 replications, and the calibrated method is exact pool
substitution, which none of them requests. `pool_exchangeability_on_real_records` carries
`decidable_here: false`, and a guard asserts it **survives a passing recording** -- it is the
failure mode this phase named in advance as the silently-violable one, and it is therefore the one
most likely to be quietly dropped once everything else goes green. Each blocker states what would
discharge it, and a guard fails a blocker that does not, because a blocker with no discharge
condition is a complaint.

TG17.11 kept three facts apart: a calibrated method the declared plans cannot reach, a method that
does not exist, and a method that ran and failed. This slice adds a fourth -- a calibrated,
applicable, **recorded** method that answers a question no declared plan asks -- and refuses to
report the measurement it can make in place of the one it cannot.

**Mutation testing: one gap, and it was the one a slow measurement hides.**

```
first pass:  14 CAUGHT, 1 MISSED
  MISSED  drop the witness from what a recording carries

second pass: 15 CAUGHT, 0 MISSED
```

Every guard read the recording already on disk, so a break in the code that *writes* one would have
passed the entire suite and surfaced only after the next fifty-minute re-record -- by which point
the recording it produced would be a receipt with an unusable witness. `_trim_case` is now
exercised directly on a one-realisation outcome costing about a second, and a guard also asserts
that what a recording is written with and what the gate reads back are the same shape. The fourteen
caught include awarding the gate a `PASS` now that a calibration passes, dropping the
exchangeability blocker once everything else is green, remembering the predecessor's claim instead
of recomputing it, calling it superseded while the predecessor is void, letting an absent successor
supersede, asserting the declared null instead of reading it from the manifests, and reading a
recording without binding it to the source it was measured against.

**Documentation guards.**

```
> python -m pytest src/tests/test_documentation.py -q
29 passed
```

**Full backend suite.**

```
3893 passed, 4 skipped, 1 xfailed, 6 warnings in 3859.63s (1:04:19)
```

The verdict is unmoved: `NOT_RELEASEABLE`. Scale/shape mode is now a calibrated method waiting on a
declared plan and a curated inventory, rather than a method that could not work. That is a better
position than TG17.11 left it in, and it is not a release.

## T4F.2 -- counted sequences, and the denominator they are counted over (2026-09-05, `ed-dev`)

`src/analysis_engine/spectral_sequences.py`, verified by `src/tests/test_spectral_sequences.py`
(33 test functions). Measured on 2026-09-05 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_sequences.py -q
33 passed, 1 warning in 7.67s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_sequences.py src/tests/test_spectral_events.py src/tests/test_spectral_mining.py src/tests/test_spectral_clustering.py -q
81 passed, 1 warning in 12.08s
```

T4F.1 built the ordered substrate and its own record said it counts nothing. This task does the
counting the phase exists for. `A4 -> A8 -> B8 -> C16` is now a chain with a support and a
confidence attached rather than a shape the record merely permits.

**The acceptance record, printed rather than described.** Three configurations, a chain planted
four times at frames 0, 6, 12 and 18 with two frames between links, a declared window of `[1, 3]`
frames and a minimum support of three:

```
=== grid searched frames 0..25, window [1, 3] frames, minimum support 3
sequence     support eligible   occurrs confidence  trunc  unobs status
A -> B             4        4         4      1.000      0      0 SUPPORTED
B -> C             4        4         4      1.000      0      0 SUPPORTED
C -> A             3        4         4      0.750      0      0 SUPPORTED
A -> B -> C        4        4         4      1.000      0      0 SUPPORTED
B -> C -> A        3        3         4      1.000      1      0 SUPPORTED
C -> A -> B        3        3         4      1.000      1      0 SUPPORTED
candidates examined 18, pruned by antimonotonicity 18, supported 6
```

**The same data on a record two frames shorter, which is the whole point of the task.**

```
=== grid searched frames 0..23, window [1, 3] frames, minimum support 3
(the two rows that move; A -> B and B -> C are unchanged at 4/4 and 1.000)
C -> A             3        3         4      1.000      1      0 SUPPORTED
A -> B -> C        3        3         4      1.000      1      0 SUPPORTED
```

`C -> A` reads **0.750** on the longer record and **1.000** on the shorter one, and both are
correct. On the 26-frame grid the last C's window `[23, 25]` was searched and held no A: that is a
miss, and it belongs in the denominator. On the 24-frame grid the same window runs past the last
frame anyone looked at, so that C was never given the chance the ratio is about; it is censored,
excluded, and reported as `trunc 1`. Counting it as a failure to be followed would have priced the
end of the record as a property of the pattern. The naive figure -- every antecedent in the
denominator regardless -- is 0.750 in both cases, and in the second case it is measuring where the
file stops.

`A -> B -> C` shows the other half: a three-link chain needs a window twice as long to have been
searched, so the last anchor is censored for the triple while the same anchor remains admissible
for the pair. A confidence that used one eligibility rule for every length would silently prefer
short chains.

**What is refused rather than defaulted.** A window is mandatory, because without declared lags
every pair of occurrences anywhere in the record is a transition. Its minimum lag is strictly
positive, so two events on one frame can never form a step -- T4F.1 refused to sort simultaneity
into an order and this module refuses to count one. A window that admits no lag on the grid's own
cadence lattice is refused rather than returning zeros, because a zero from an unreachable window
measures the window and not the record. A lag declared in one unit against frames declared in
another is refused rather than converted. Without a cadence, eligibility is undecidable, so the
support count stands and the ratio is `None` under `COVERAGE_UNDECLARED` rather than computed
against a denominator nobody checked.

**The pruning rule is proved, not assumed.** Extending a sequence lengthens the window an
antecedent must have observed, so the eligible set can only shrink; and a chain that completes for
`s + (q,)` completes for `s` by taking its prefix. Support is therefore antimonotone under
extension. The suite asserts that inequality directly over every extension the sweep reached, and
asserts the identity `examined + pruned == exhaustive`: 18 + 18 = 36 = 3^2 + 3^3. T4E.4's
`MiningBudget` is reused rather than a second budget declared, and either overrun refuses the
whole sweep with `partial_result: false`.

**Recurrence, and the several ways it is not a period.** Spans are tallied on the cadence lattice,
which is the finest interval the pass can resolve. All three patterns report `INTERVAL_REPEATS`
with a modal interval of 6 frames seen three times, concentration 1.00, one distinct interval. A
span crossing unobserved time *bounds* an interval from above rather than measuring it -- an
unseen occurrence inside would split it in two -- so it is excluded, counted as excluded, and kept
out of the reported longest gap. The receipt publishes `admissible_lattice_values` (25 on the
longer grid), because among few admissible values a repeat is expected under no structure at all,
and a concentration without that denominator invites over-reading. Four statuses keep the failures
apart: `NO_MEASURED_SPAN`, `TOO_FEW_MEASURED_SPANS`, `NO_REPEATED_INTERVAL`, `INTERVAL_REPEATS`.

**Mutation testing: 33 mutations, two batches, and the first batch was too easy.**

```
batch one:  17 CAUGHT, 0 MISSED
batch two:  12 CAUGHT, 4 MISSED
  MISSED  give a long chain the same eligibility window as a short one
  MISSED  admit a sequence one occurrence short of the declared minimum
  MISSED  take the concentration over every gap rather than over the measured ones
  MISSED  count the distinct intervals over the lattice rather than what was seen

after closing the gaps: 17 CAUGHT + 16 CAUGHT, 0 MISSED
```

A clean first pass is not evidence of good guards; it is evidence that the mutations were easy.
The second batch aimed at the quantities nothing obviously read and at the off-by-ones, and four
survived. The first is the substantive one: nothing asserted that a longer chain requires a longer
observed window, because the antimonotonicity guard is an inequality and equality satisfies it.
That mutation would have made every chain, at every length, eligible on a pair's window -- the
exact bias this task exists to prevent, passing a suite that appeared to test for it.

**One defect in this task's own code, found by printing the receipt rather than by a test.**
`candidates_pruned_by_antimonotonicity` counted the *prefixes* that failed rather than the
*candidates*, and multiplied by the level width. At the first level those two are the same number,
so the arithmetic agreed with itself exactly where it was tested and nowhere else: the receipt read
`pruned 0` while eighteen extensions had genuinely never been built. The acceptance guard had
encoded the same wrong formula, and the mutation aimed at that line was caught only because it
disagreed with an expectation that was itself wrong. Fixed, re-anchored so the mutation now
reintroduces exactly this defect, and the suite now asserts the identity `examined + pruned ==
exhaustive`, which no single wrong formula can satisfy. **It gets no defect number**: it never
reached a commit and no recorded result depended on it. It is written down because the way it
surfaced -- reading an actual receipt, after a green suite and a clean mutation pass -- is the part
worth keeping.

**One method was added to T4F.1's grid rather than to this module.**
`ObservationGrid.observation_of_window` answers whether a *proposed* window was wholly searched,
which `coverage_between` cannot: a span runs between two occurrences known to be on the grid, while
a window is offered by a lag and can therefore run off the end of what the pass looked at. The grid
is what knows what was looked at, so the question belongs there. `WINDOW_TRUNCATED_BY_RECORD` is
kept apart from `WINDOW_SPANS_UNOBSERVED_TIME` because the two bias a confidence differently -- one
is the record's edge, the other is a hole in the middle -- and collapsing them would lose the
distinction that makes the exclusion defensible.

**Nothing here is significance, and both receipts say so.** Support and confidence are not a base
rate, a lift, a surrogate comparison, a p-value, a precursor or a cause. A repeated interval is not
a period, a frequency or an oscillation, and no null was drawn over any of it. Those begin at
T4F.3.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed, 2 warnings in 305.58s (0:05:05)
```

Re-run standalone after the documentation edits rather than trusted from inside the long run.

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/ -q
3926 passed, 4 skipped, 1 xfailed, 6 warnings in 3510.91s (0:58:30)
exit 0
```

## T4F.3 -- precursor tests, and the first null this phase draws (2026-09-05, `ed-dev`)

`src/analysis_engine/spectral_precursors.py`, verified by `src/tests/test_spectral_precursors.py`
(47 test functions, 52 pytest cases: one of them is parametrised six ways). Measured on 2026-09-05 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_precursors.py -q
52 passed, 1 warning in 16.98s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_precursors.py
    src/tests/test_spectral_events.py src/tests/test_spectral_sequences.py
    src/tests/test_spectral_mining.py src/tests/test_spectral_clustering.py -q
133 passed, 1 warning in 45.28s
```

T4F.2 counted and its own receipt said it claimed nothing. This task asks whether a count means
anything. It is the roadmap's central question and the first null Phase 4F draws.

**The acceptance record, printed rather than described.** Two patterns on a 150-frame grid; A
every ten frames, B two frames behind each one, and three further B occurrences in a stretch A
never enters. Window `[1, 3]` frames, 99 circular-shift surrogates, seed 20260905, alpha 0.05,
Benjamini-Yekutieli:

```
rule       sup  elig     conf     base    lift lift 95%              p        q  status
A->B        10    10    1.000    0.259    3.87 [2.79, 3.87]     0.0100   0.0300  PRECURSOR_SIGNATURE
         corrected lift 3.93 (null mean confidence 0.254), overlapping windows 0, six figures yes
B->A         0    13    0.000    0.184    0.00 [0.00, 1.24]     1.0000   1.0000  NOT_DISTINGUISHED
         corrected lift 0.00 (null mean confidence 0.193), overlapping windows 0, six figures no
```

**The measurement this task exists to make.** The same unchanged record, the same lift, two nulls:

```
=== B in four dense blocks, A one clump of four sitting just in front of the first
circular_antecedent_shift      conf 1.000 base 0.190 lift 5.25  p 0.1100 q 0.1100  NOT_DISTINGUISHED
   null mean confidence 0.187, corrected lift 5.35
uniform_antecedent_relocation  conf 1.000 base 0.190 lift 5.25  p 0.0100 q 0.0100  PRECURSOR_SIGNATURE
   null mean confidence 0.185, corrected lift 5.40
```

Nothing about the data differs between those two lines. The lift is identical, the base rate is
identical, the ensemble size and the seed are identical. The scattering null destroys the
antecedent's own bursting, so a clump of four that happens to sit in front of a dense block beats
it on the shape of its own occurrence times; the shifting null moves the clump as a clump, and
roughly one rotation in nine lands it in front of some block, which is what p = 0.11 is
measuring. The choice of null is the hypothesis, and here it decides the finding. The uniform
null is provided for exactly this reason and its receipt carries the word **anti-conservative**
and the sentence "it should not be used to support a claim".

**The base rate is a window probability and it is measured.** Confidence is the chance that a
declared window following an occurrence contains the consequent, so dividing it by the fraction
of *frames* carrying the consequent would give a lift that grows with the width of the window and
with nothing else. `window_base_rate` drops the **same window** at every searched position whose
window was wholly observed -- the eligibility rule that decides an antecedent's admissibility,
applied to the reference -- and publishes the rate with the number of positions it was estimated
over. The reference is not purged of the antecedent's own occurrences: purging would make the
denominator depend on which rule is being tested, so two rules sharing a consequent would be
divided by different numbers. Including them pulls lift toward 1 for a common antecedent, which
is the conservative direction, and the choice is stated in the receipt rather than left implicit.

**A lag chosen by the data is a search, and the null is over the choice.** `src/core/precedence.py`
found this at the band level: the maximum of several lagged statistics is not one statistic. Two
families are therefore reported and each is corrected against its own size -- every (antecedent,
consequent, window) triple, all of them reported so that no selection precedes the correction; and
one selected lag per pair, referenced to the distribution of the **maximum across the declared
family**. That maximum has a null because one draw is shared across the family per antecedent, and
the sharing is held to by test rather than asserted: the selected null must equal the elementwise
maximum of the per-lag ones, and a window's ensemble must be identical whether it was declared
first or second in the family.

**A design that could not have rejected anything is refused before anything is counted.** With
`n` surrogates a p-value cannot fall below `1 / (1 + n)`, and rank 1 of an `m`-test family under
Benjamini-Yekutieli needs a raw p below `alpha / (m * H_m)`. `check_power` is asked first, and an
ensemble too small for the declared family raises. This is not T4C.5i's boundary applied late: an
under-powered study duly reports an absence indistinguishable from a real one, and the cleanest
way to honour "an inadequately powered absence is not a negative finding" is never to produce the
absence. The refusal is a fact about the declared design, before the record is read. It fires in
the suite on its own: a run at alpha 0.02 with 99 surrogates and two tests is refused with the
arithmetic, and 149 surrogates named as the requirement.

**Nothing is reimplemented.** The counting and its eligibility rule are T4F.2's `count_sequence`,
made public in this task so that a second denominator cannot exist -- two would agree on the
planted case and diverge exactly at the record's edge, which is the case the denominator was
written for, and the suite asserts the two agree figure for figure. The empirical p-value is
`significance.surrogate_p_value`; the correction and the power check are
`multiple_comparisons.adjust` and `check_power`; R9's six figures are carried by the programme's
own `AssociationFigures`, so a rule that cannot satisfy that contract is refused by the contract
rather than by a check in this file.

**Mutation testing: 49 mutations in two batches, 27 + 22, all caught after the gaps were closed.**

```
batch one:  19 CAUGHT, 8 MISSED
  MISSED  let the base rate depend on which antecedent is being tested        (equivalent mutant)
  MISSED  give each lag its own draw, so the maximum has no null              (equivalent mutant)
  MISSED  count one frame too many inside a proposed window                   (script scope)
  MISSED  take a Wald interval, which is too narrow exactly where support is small
  MISSED  correct against the tests that were reported rather than the family declared
  MISSED  call a rule a precursor on its raw p-value
  MISSED  relocate uniformly with replacement, so the surrogate loses occurrences
  MISSED  make the surrogate-corrected lift a second copy of the plain lift

batch two (the five real gaps re-aimed, plus fourteen new):  18 CAUGHT, 4 MISSED
  MISSED  relocate uniformly with replacement          (the fixture made a collision unlikely)
  MISSED  hide how many of the interval's trials shared a window   (no anchors a window apart)
  MISSED  let an unstated seed default to something                (bad anchor)
  MISSED  count one frame too many inside a proposed window        (script scope, again)

after closing every gap and re-anchoring: 27 CAUGHT + 22 CAUGHT, 0 MISSED
```

Three of batch one's misses were faults in the mutations rather than in the guards, and they are
recorded as such rather than quietly dropped. Two were **equivalent mutants**: trimming the last
frame from the base-rate universe removes a position whose window was already truncated and so
changes no number, and re-seeding the generator changes every draw consistently rather than
giving each lag its own. The third was a **scope error in the harness** -- an
`observation_of_window` mutation was run against the precursor and event suites but not against
T4F.2's, which is where that method's guard actually lives; with the third file added it is caught
at once, twice over.

The five that were real gaps are the ones worth naming. **The Wald interval is the substantive
one.** Nothing asserted the interval was the *score* interval rather than the normal
approximation, and the existing test compared widths, which a hybrid preserves. It matters
precisely here: a precursor rule is a rule where every eligible trial succeeded, and a Wald
interval on a perfect record has **no width at all** -- it would report that the record had
settled the question exactly. The test now holds the bounds to the score interval's defining
property, that the observation sits `z` standard errors from each of them, which no
approximation satisfies. **Correcting on the reported tests rather than the declared family** and
**calling a rule a precursor on its raw p-value** both survived because the planted fixture is too
clean: every member returned a number and the one rejection cleared both thresholds. Closing them
needed two fixtures built for the purpose -- a family containing a member that returns no p-value
at all, and a family whose every member clears alpha raw and none of them clears it corrected.
The remaining two, drawing the relocation with replacement and printing the plain lift twice,
were quantities nothing had read.

**One design consequence worth recording.** The power refusal and the raw-versus-corrected test
collide by construction: at rank 1 the corrected threshold is exactly the ratio the power check
demands, so a p-value at the ensemble's floor is always rejected under a design powered to reject
anything. The discriminating case therefore has to sit strictly above the floor, and the fixture
that produces it -- two lags on one pair, p = 0.02 and 0.04, both clearing alpha = 0.05 raw and
neither clearing q = 0.06 -- is named in the test so that a later edit which loses it fails loudly
rather than silently testing nothing.

**One change to an earlier task's code.** `ObservationGrid.observation_of_window` was a linear
scan over the grid's frames. It is called for every antecedent of every rule of every surrogate,
which took the smallest useful family from 0.00 s to 3.09 s, and a real study to minutes. It now
bisects a float copy of the frames built once in `__post_init__`. The classification it returns is
unchanged, T4F.1's and T4F.2's suites pass untouched, and two mutations aimed at the new
arithmetic -- one frame too many, and treating the window's own endpoints as outside it -- are
caught by T4F.2's existing guard on a window with a single missing frame in it.

**What is refused rather than defaulted.** A lag family must be a declared `LagFamily`, because a
family assembled at the call site is a family whose size the correction never sees; it is refused
empty, refused with a window declared twice, refused with two units in it, and refused in a unit
the grid does not use. The seed and the ensemble size are required arguments, because an ensemble
nobody can redraw is an ensemble nobody can check. A grid without a cadence is refused with the
three things it costs named. A pattern preceding itself is refused and sent to T4F.2's
`recurrence_report`, which has a denominator built for that question. An ensemble larger than the
record's distinct admissible rotations is refused rather than drawn with replacement, because the
same rotation twice prices one null draw as several. A declared family that will not fit its
budget is refused whole before any counting, with `partial_result: false`.

**Nothing here is causal.** A rejected rule is a `PRECURSOR_SIGNATURE`, and the boundary refuses
*cause*, *driver*, *mechanism*, *trigger*, *forecast* and *intervention* by name (R7). It also
states two things a p-value invites a reader to forget: the null is a statement about alignment,
so both patterns following a third thing this record does not contain is entirely consistent with
a rejection; and the base rate is estimated from this record alone, so a rule is a statement about
this record and not about the world. A test asserts that no status this module can emit contains a
term `claim_ladder.OUTSIDE_THE_LADDER` names.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
3978 passed, 4 skipped, 1 xfailed, 6 warnings in 3129.85s (0:52:09)
exit 0
```

## T4F.4 -- bidirectional queries, and the ranking that decides the answer (2026-09-05, `ed-dev`)

`src/analysis_engine/spectral_queries.py`, verified by `src/tests/test_spectral_queries.py`
(58 test functions, 58 pytest cases). Measured 2026-09-05 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_queries.py -q
58 passed, 1 warning in 2.63s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_queries.py
    src/tests/test_spectral_precursors.py src/tests/test_spectral_events.py
    src/tests/test_spectral_sequences.py src/tests/test_spectral_mining.py
    src/tests/test_spectral_clustering.py src/tests/test_spectral_invariance.py -q
245 passed, 1 warning in 51.08s
```

T4F.3 produced a corrected table. This task makes it askable from either end. The design is one
sentence -- both directions are one selection over that table and nothing is recomputed -- and
everything below is what that sentence rules out.

**The acceptance record, printed rather than described.** Six patterns on a 220-frame grid. The
target `B` occurs at eighteen irregular times; `A` sits two frames in front of every third of
them; `C` is common, evenly spread on every seventh frame, and related to nothing; `D` spans
`B`'s scale range; `E` is coarser than `B`; `F` is fine but every occurrence is censored by the
record's end. Window `[1, 3]` frames, 199 circular-shift surrogates, seed 20260905, alpha 0.05,
Benjamini-Yekutieli, and a family of four declared before the record was read:

```
=== the ordering, measured from the catalogue's own scales (units: cells)
  A  scale   2.520   range [  2.0,   4.0]
  B  scale  20.159   range [ 16.0,  32.0]
  C  scale   3.557   range [  3.0,   5.0]
  D  scale  20.000   range [ 10.0,  40.0]
  E  scale  80.635   range [ 64.0, 128.0]
  F  scale   1.260   range [  1.0,   2.0]
  not orderable: A/C, A/F, B/D

=== the declared family: four antecedents, one target, one lag, 199 shifts
rule   supp elig   conf    base   lift      p       q   status
A->B     6    6   1.000  0.249   4.02  0.0050  0.0417  PRECURSOR_SIGNATURE
C->B     9   31   0.290  0.249   1.17  0.3300  1.0000  NOT_DISTINGUISHED_FROM_NULL
D->B     0    4   0.000  0.249   0.00  1.0000  1.0000  NOT_DISTINGUISHED_FROM_NULL
E->B     0    4   0.000  0.249   0.00  1.0000  1.0000  NOT_DISTINGUISHED_FROM_NULL
```

**The measurement this task exists to make.** The same query, the same report, two rankings:

```
=== top-down on B, ranked by q_value -> PRECURSOR_SIGNATURES_FOUND
    rows naming it 4; withheld: 1 overlapping scale, 1 wrong side
    1. A  supp  6  lift  4.02  q 0.0417  PRECURSOR_SIGNATURE
    2. C  supp  9  lift  1.17  q 1.0000  NOT_DISTINGUISHED_FROM_NULL

=== top-down on B, ranked by support -> PRECURSOR_SIGNATURES_FOUND
    rows naming it 4; withheld: 1 overlapping scale, 1 wrong side
    1. C  supp  9  lift  1.17  q 1.0000  NOT_DISTINGUISHED_FROM_NULL
    2. A  supp  6  lift  4.02  q 0.0417  PRECURSOR_SIGNATURE
```

The roadmap words this task's top-down question as *what fine configurations most commonly
preceded it*. Answered literally, by how often the counterpart was followed by the target, the
first answer is a pattern the null did not distinguish from chance -- it wins because it occurs
more often, not because it precedes anything. Ranked by the corrected p-value the first answer
is the planted precursor. Nothing about the data differs between those two blocks. **The choice
of ranking decides the answer**, as the choice of null decided the finding one task earlier.
Every entry therefore carries its rank under *every* key, `rankings_disagree` is published on
the result, and the frequency key carries the sentence saying that it is not a measure of
association and that the reader should look at which entries the null distinguished before
reading the order.

**A query is a view, and a view is not a test.** The shortcut is to re-run the inference
restricted to the pattern being asked about -- one target, a handful of counterparts, a family
of one instead of four -- and watch every q-value fall. That is choosing the family after seeing
the record, which is the failure `precursor_report` avoids one level down by reporting every
member of the declared family whether it looked interesting or not. Nothing here recomputes a
p-value, a q-value or a status: the suite asserts every returned figure is identical to the
report's own row, and the family the correction was paid on is published beside every answer
with its declared size (4) next to an answer of 2. The cost of the shortcut is measured rather
than warned about: the same rule, the same record, the same seed and the same null give
`p = 0.0050` either way, and `q = 0.0417` in the declared family against `q = 0.0050` in a
family narrowed to it alone. A caller who genuinely wants the narrower family must declare it
before the record is read, by passing `pairs=` to `precursor_report`.

**Coarse and fine are measured, and the record is allowed to refuse them.** `scale_ordering`
reads the catalogue's own member scales in the catalogue's own units and builds an interval
order: a pattern occupies the range of scales its members actually spanned, and one pattern is
finer than another only when those ranges are disjoint. Ranges that overlap -- or merely touch,
since sharing a scale is sharing a scale -- are not orderable, and naming one of them the coarser
would be an ordering taken from a sort rather than from the record, which is the fabrication
`spectral_events` refuses when it publishes simultaneous events unordered. The relation is
partial by construction, the pairs it cannot order are published, and on this record three of
fifteen pairs are unorderable. In the top-down answer above, `D` is withheld because its range
spans the target's and `E` because it is coarser rather than finer; both are counted, and
counted apart from each other, so an answer of two out of four cannot be mistaken for an answer
of two out of two. The statistic is the geometric mean of every member node's scale and the range
spans every member rather than the first, which the suite holds to on a pattern whose members
disagree.

**Cardinality is not scale.** A five-node constellation is bigger than a three-node one, not
coarser. Two catalogues carrying the same scales at cardinality three and four produce an
identical ordering.

**Under scale invariance there is nothing to order, and the refusal is exact.** T4E.2's
universality hook divides every signature's scales by their own geometric mean, so that statistic
is exactly 1.0 for every pattern in such a catalogue and an ordering built on it would order
floating-point residue. A separate test asserts that exactness to 1e-12 on three unrelated scale
triples, so the refusal rests on a measured fact rather than on caution. That mode buys
cross-domain comparability by discarding absolute scale, and a direction query is a question
about absolute scale; the module says which one was asked for.

**Both directions are one engine.** The direction fixes two things and nothing else: which role
the target plays in the rule, and which side of the ordering the counterpart must sit on. A
top-down entry on the target and the matching bottom-up entry on the antecedent are asserted to
be the *same row object*, by identity, so a fix to one direction cannot fail to reach the other.
What is not true is that one direction's answer can be read off the other's: confidence divides
by a different denominator each way, so `A -> B` and `B -> A` are two hypotheses, both declared
and both corrected, and `directional_pair` puts them side by side rather than deriving one from
the other.

**An empty answer says which kind of empty it is.** Nothing distinguished from the null, nothing
on the required side of the ordering, and nothing on that side that could be measured at all are
three different findings and only the first is a negative result; each has its own status and
each is exercised on a record built for it. A censored antecedent is returned rather than dropped,
carries no rank under any key -- its support is zero because there was nothing to count, and
ranking that zero against a measured one would put a censored row in a league table of
associations -- and sorts below every measured entry.

**Mutation testing: 61 mutations in two batches, 37 + 24, all caught after the gaps were closed.**

```
batch one:  31 CAUGHT, 6 MISSED
  MISSED  call touching scale ranges orderable
  MISSED  publish the per-lag family's size beside a selected-lag answer
  MISSED  read the eligible count where the support count was asked for
  MISSED  take the leading entry from the bottom of the ranking
  MISSED  return the entries in table order rather than in the order that was asked for
  MISSED  drop the report's own boundary from the query receipt

batch two (the six re-aimed, plus eighteen new):  19 CAUGHT, 5 MISSED
  MISSED  report the whole family as the rows that named the target
  MISSED  publish the counterpart's scale range as the target's
  MISSED  word the top-down question as a search for something coarser
  MISSED  estimate a pattern's scale range from its first member alone
  MISSED  claim the family was declared after the record was read

after closing every gap: 37 CAUGHT + 24 CAUGHT, 0 MISSED
```

Every one of the eleven was a real gap; none was an equivalent mutant and none was a fault in the
harness. Four of them share a cause worth naming: **the acceptance fixture was too tidy.** Every
rule in it named the target, so reporting the whole family as the rows naming the target changed
no number; every member of a pattern carried identical scales, so estimating the range from the
first member alone changed no range; the decoy had both more support and more eligible trials
than the planted rule, so ranking by the eligible count reproduced the support order exactly.
Closing those needed records built for the purpose -- a family containing a rule that names
something else, a pattern whose members occupy different scales, and a two-lag family in which
support and eligibility disagree about the order (`[9, 6, 5, 0]` by support against `[31, 6, 31,
6]` eligible).

The other seven were assertions that were weaker than they looked. Two ranges that *touch* at one
scale were never tried, so nothing distinguished a strict boundary from an inclusive one -- and
the strict one is right, because two patterns sharing a scale are not separated by the record.
`top_by` was checked only for disagreeing with itself across keys, which survives being taken
from the bottom of the ranking rather than the top. The receipt's copy of the report's own claim
boundary was checked by substring, which survives being replaced by that substring. The question
text each direction publishes was compared with the constant it came from, so both moved
together; it is now compared with the side the direction actually selects. The rest --
`target_scale_range`, `declared_before_the_record_was_read`, and the two families' separate
sizes -- were quantities nothing had read.

**Sorting the unmeasured.** One batch-one miss deserves its own line because it is nearly an
equivalent mutant and is not one. Removing the "unranked last" term from the entry sort leaves
the remaining key ordering measured entries identically, so it is invisible on every fixture
whose entries were all measured. It becomes visible only on an answer that mixes a measured
counterpart with a censored one, where the censored row would otherwise sort *first*. That
record is now in the suite.

**What is refused rather than defaulted.** There is no default direction, because "what preceded
this" and "what follows this" are two hypotheses about the same pair and a silently chosen one
would answer a question nobody asked. A target the declared family never asked about in the
required role is refused by name rather than answered with an empty list, which would read as
"nothing preceded it" when the truth is that nothing asked. An ordering that does not carry a
scale range for every pattern the report tested is refused, because a query answering from a
subset it never mentions lies about its denominator. A support ranking of the selected-lag
family is refused by name, because a selected-lag row is one pair's strongest declared lag and
the count belongs to the lag that won rather than to the pair. A catalogue with no scale unit,
with two scale units, with a non-positive scale, or with no patterns at all is refused, as is a
query handed something that is not a completed report.

**Nothing here is causal, and nothing here is a direction of influence.** The claim boundary
refuses *cause*, *driver*, *mechanism*, *trigger*, *forecast* and *intervention* by name (R7),
carries the report's own boundary unaltered beside it, states that a query is a view of a test
that has already been corrected and not a second test, and states the thing the words most
invite a reader to forget: **top-down names a query whose target is the coarser of the two
patterns, and asserts nothing about which of them acts on the other.** A test asserts that no
status this module can emit contains a term `claim_ladder.OUTSIDE_THE_LADDER` names.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
4036 passed, 4 skipped, 1 xfailed, 6 warnings in 3665.97s (1:01:05)
exit code 0
```

## T4F.5 -- evidence projection, and the flank that makes a pixel a lie (2026-09-06, `ed-dev`)

`src/analysis_engine/spectral_projection.py`, verified by `src/tests/test_spectral_projection.py`
(67 test functions, 67 pytest cases). Measured 2026-09-06 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_projection.py -q
67 passed, 1 warning in 3.16s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_projection.py
    src/tests/test_spectral_queries.py src/tests/test_spectral_precursors.py
    src/tests/test_spectral_sequences.py src/tests/test_spectral_events.py
    src/tests/test_spectral_clustering.py src/tests/test_spectral_invariance.py
    src/tests/test_spectral_constellation.py src/tests/test_spectral_mining.py
    src/tests/test_spectral_feature.py src/tests/test_spectral_tracking.py
    src/tests/test_spectral_narrative.py -q
427 passed, 1 warning in 50.89s
```

T4F.3 produced a corrected table and T4F.4 made it askable from either end. Both answer in
pattern identities, and an integer is not evidence. This task says where on the parent grid, in
which frames, at what value of the field, and over which occurrences.

**The acceptance record, and why it is graded across all of it.** The T4D.2 benchmark
`advected_vortex_sequence`: one Gaussian vortex on a known straight trajectory, growing on a
known schedule, over 24 frames of a 128x128 crop, whose centre cell the benchmark records for
every frame. Decomposed with an undecimated haar bank at levels 4 and 5, detected at 4 sigma,
tracked, and enumerated as T4E.1 pairs. The projection is then graded against the recorded
centre in **every** frame rather than at a chosen one, because the interesting result is where
it stops working.

```
=== the map, from the record itself
  grid                cartesian 128x128, 31 km cells
  coordinates         PROJECTED_OFFSET_FROM_CROP_ORIGIN -- "this grid declares a metric but no
                      latitude, so these are distances from the crop's own cell (0, 0) and not
                      a position on the Earth"
  vertical            NO_VERTICAL_COORDINATE
  time                FRAME_ON_THE_RECORD_CLOCK -- no epoch, so a frame and not a date
  filter support      level 4: 16 cells   level 5: 32 cells      (the dyadic octave labels are
                                                                  8 and 16 -- different numbers)
```

**A coefficient maximum is not a pixel, and the record proves it rather than the docstring
asserting it.** A detail wavelet is derivative-like, so a symmetric blob has no maximum at its
centre and two on its flanks, about one analysing width out along the axis the band high-passes.
Measured over every occurrence in the record:

```
=== offset from the planted centre, by band
band     n   mean offset / vortex width   reach   holds the planted cell
L4/LH   50            1.079                 8     frames 0-8
L4/HL   50            1.080                 8     frames 0-6, 8-9
L5/LH   33            1.277                16     frames 9-19  (every frame it detects)
L5/HL   41            1.248                16     frames 9-23  (every frame it detects)

the peak cell equals the planted cell in 0 of 174 member projections
the offsets run from 6.06 to 15.15 cells
```

So the roadmap's acceptance -- *for a planted synthetic precursor, projection recovers the exact
grid cells that were planted* -- is met, and it is met by the footprint and never by the peak.
Projecting the maximum onto one pixel would have been wrong by six to fifteen cells in every
frame of the record while printing two decimal places.

**And it is met with a boundary in it, which is the finding.** A footprint reaches half the
filter support, and the flank offset grows with the structure. Level 4 reaches eight cells and
holds the planted cell out to frame 9; from frame 10, when the vortex is wider than eight cells,
it loses it in all fifteen remaining frames. Level 5 reaches sixteen and never loses it. The
suite checks containment against that geometry rather than against the code -- a member whose
offset is more than a cell inside the reach must contain the planted cell and one more than a
cell outside it must not, with the five frames that sit within a cell of the boundary left to
the measurement, since there the answer turns on where the planted half-cell rounds. **A
coefficient at a level far finer than the structure points away from it by more than its own
reach, and evidence should be read at the level that resolves the thing.**

**An intersection is not a location either, and the measurement corrected the prose.** The first
version of this module said the intersection of two members is "where those offsets cannot all
be pointing away from". The record says otherwise:

```
L5/LH + L5/HL   (one level, two axes)          11 occurrences, planted cell inside  11 / 11
L4/LH + L5/LH   (one orientation, two levels)  11 occurrences, planted cell inside   0 / 11
```

Two bands that resolve different axes have flanks pointing in different directions and their
overlap straddles what excited them; two bands of one orientation have flanks pointing the same
way and their overlap sits beside it, non-empty and wrong. Both counts are pinned in the suite,
the sentence in the receipt now says both, and an occurrence whose members all share an
orientation carries a `common_caution` saying so. This is the second time in this phase that a
measurement has corrected a sentence that sounded right.

**Each grid, level and clock says only what it can.** A `latlon` grid gives degrees in its own
convention, with a longitude past a closing grid's seam and the cell it names both brought back
inside the grid's own range; a `cartesian` grid gives northing and easting in metres and names
them a distance from the crop's own origin rather than a place; a `pixel` grid gives cells and
refuses degrees by name. The task asks for pressure levels: a level set without the coordinate
it is a value of is reported as a number, because calling it hectopascals is the assumption
TG1.5 exists to refuse, and this record declares no vertical coordinate at all. A frame is a
frame unless the *record* carries a calendar -- a decomposition's own clock is
`FieldSequence.times_seconds` by construction, so the field cannot know whether its numbers are
dates and the record it came from can.

**The field's values and the anomaly magnitudes are inputs.** Values under a footprint come from
the record the coefficients were taken from, refused if its length, grid or clock disagree with
the decomposition; an anomaly comes from a supplied anomaly record. A raw value is never called
an anomaly and this record's own time mean is never quietly subtracted to make one, because
which baseline was removed is a decision belonging to whoever removed it (R11). With nothing
supplied the footprint says the values are unknown and states that they are not zero and not the
coefficient.

**The historical instances are the ones the rule was counted on.** The per-anchor decision moved
into `spectral_sequences.anchor_verdicts`, which `count_sequence` now tallies and this module
lists; `window_completions` is likewise the one place a window's completions are found. A second
implementation would have agreed on the planted case and diverged exactly at the record's edge,
which is the case the eligibility rule was written for. On the acceptance record:

```
=== 1 -> 2 at lag [1, 3] frames, 113 surrogates, seed 20260906, family of two
supp 8   elig 9   occurrences 11   truncated 2   unobserved 0
conf 0.8889   base 0.3419   lift 2.600   p 0.0088   q 0.0263   PRECURSOR_SIGNATURE

  anchor   verdict                           completions
     6.0   SUPPORTING                        [8.0]
    13.0   SUPPORTING                        [15.0]
    ...
    60.0   ELIGIBLE_AND_NOT_FOLLOWED         []          (the consequent at 64 is four frames
                                                          out, and the window reaches three)
   118.0   INELIGIBLE_TRUNCATED_BY_RECORD    [119.0]     (followed, and still not support)
   119.0   INELIGIBLE_TRUNCATED_BY_RECORD    []
```

The anchor at 118 is the one worth reading. Something did follow it, one frame later, and it is
still not support: its window reaches to 121 and the record stops at 119, so nobody watched the
rest of it. Counting it would price the tail of every record as a success whenever the
consequent happens to be common, which is the asymmetry T4F.2's denominator was built to refuse.
With frames 62 and 63 unsearched instead, the anchor at 60 stops being an eligible failure and
becomes undecidable -- `INELIGIBLE_SPANS_UNOBSERVED_TIME` -- and the two are different findings:
one lowers the confidence, the other leaves the denominator. Every enumerated total is
reconciled against the figures the rule published before any of this is shown, and a series that
produces different ones is refused by name rather than displayed.

**Per-scale contribution is a share of this pattern's own members and of nothing else,** because
the bank is undecimated and therefore redundant: per-scale coefficient energies do not partition
the field's variance, and a structure straddling two levels appears in both shares. The suite
holds the shares to the squared magnitudes of the pattern's own member nodes over their total,
on a pattern that genuinely spans two levels so that dividing by the largest rather than by the
total would be visible.

**Mutation testing: 67 mutations in two batches, 43 + 24, all caught but one shown to be
equivalent.**

```
batch one:  32 CAUGHT, 11 MISSED
  MISSED  end the footprint one cell further out
  MISSED  do not record that the crop cut the footprint
  MISSED  treat a one-column footprint at the seam as wrapping
  MISSED  take the union of the footprints rather than the intersection
  MISSED  keep the column overlap when the rows do not overlap at all
  MISSED  order a pattern's occurrences by track rather than by time
  MISSED  share the magnitudes rather than their squares
  MISSED  divide each share by the largest rather than by the total
  MISSED  count a truncated anchor that happened to complete as support
  MISSED  report an unobserved window as truncated by the record
  MISSED  list completions from a window one frame wider than the one counted

batch two (24 new, aimed where the first batch did not):  16 CAUGHT, 6 MISSED
  MISSED  read every scale's support off the first level          <- equivalent, see below
  MISSED  test containment against only the first segment of a wrapped footprint
  MISSED  clip a column past the seam instead of wrapping it
  MISSED  truncate a sub-pixel row to a cell instead of rounding it
  MISSED  project the consequent's pattern at the antecedent's frame
  MISSED  project a member measured at a scale this decomposition does not carry

after closing every real gap: 43 CAUGHT + 23 CAUGHT, 1 equivalent
```

**The equivalent one, stated rather than counted as a kill.** `_interior_halfwidth(field, scale,
position)` takes its level from `int(scale)` whenever the scale label parses as an integer and
falls back to `position` only when it does not; `spectral_tracking._scale_quantity` refuses any
scale label that is not a positive integer level, so nothing that can produce a track -- and
therefore nothing that can produce a constellation to project -- can reach that fallback.
Passing `position + 1` is the correct contract of the helper and the mutation to `1` cannot be
observed. It is recorded here rather than closed with a test that would have to construct a
field the rest of the phase refuses.

**What the sixteen real gaps were.** Five were the fixture being too tidy again, in a new way:
every occurrence on the acceptance record is of one advecting vortex, so its members always
overlap, no anchor was ever both truncated and followed, no consequent ever sat just outside the
declared window, no unobserved window ever failed to complete, and the catalogue's members were
already in time order when they reached the projection. Those were closed by building the cases
-- a consequent four frames after an anchor and another inside a truncated window, two unsearched
frames that make an eligible failure undecidable, a catalogue handed to the index with its
members reversed, and an intersection taken between two footprints seventeen frames apart on one
trajectory. The rest were assertions weaker than they looked: a footprint's side length was
allowed a one-cell range so widening it stayed inside, containment was never tested across the
two segments of a wrapped footprint, the sub-pixel rounding of a centre cell was only ever
compared against itself, the intersection was never asserted to lie *inside* its members, a
share was only checked to sum to one on a pattern that had a single scale, and the antecedent
end of an instance was never checked to be the antecedent's pattern.

**What is refused rather than defaulted.** A detection whose positions were never aligned to the
parent grid is refused by name (D88): an unaligned index sits at the filter's anchor rather than
where it responded, by tens of cells at coarse levels, so a footprint drawn from it would be in
the wrong place by an amount that grows with scale. A constellation that dropped the features it
was built from cannot be projected, because nothing else can say which filter's support its
nodes have. A member measured at a scale this decomposition does not carry is refused rather
than given the nearest one. A catalogue whose members have no constellation in the supplied
extraction is refused with the count and the first missing key, because a projection that
skipped them would understate the evidence while looking complete. A selected-lag row is refused
where a counted rule is required. And a signature is refused with the reason: it carries no
position, by design, which is what makes it comparable.

**Nothing is claimed beyond a footprint.** The claim boundary refuses *cause*, *driver*,
*mechanism*, *trigger*, *forecast* and *intervention* by name (R7), carries T4F.3's own boundary
beside it on a rule projection, and states the thing this task most invites a reader to forget:
a footprint is **where a coefficient of this transform was, not where a structure is**, and the
field values reported under it are the record's own values at those cells rather than evidence
that the coefficient measured them. No verdict, status or basis name this module emits contains
a term `claim_ladder.OUTSIDE_THE_LADDER` names.

**The half of the acceptance that has not run.** "On real ERA5, a `recognised` pattern from
T4F.6 projects onto a physically sensible footprint -- verified by eye" needs T4F.6, which does
not exist, and a real-ERA5 mining pass, which has not been run. It is outstanding, and T4F.6 is
where it comes due.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed, 2 warnings in 1645.10s (0:27:25)
# re-run on the final tree, sharing the machine with the full suite above; an
# earlier standalone run of the same 29 guards took 273.08s (0:04:33)
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
4103 passed, 4 skipped, 1 xfailed, 6 warnings in 2524.94s (0:42:04)
exit code 0
```

## T4F.6 -- the physical gate, and what it takes to keep one able to fail (2026-09-06, `ed-dev`)

`src/analysis_engine/spectral_reference.py`, verified by `src/tests/test_spectral_reference.py`
(87 test functions, 87 pytest cases). Measured 2026-09-06 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_reference.py -q
87 passed, 1 warning in 3.55s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_reference.py
    src/tests/test_spectral_projection.py src/tests/test_spectral_queries.py
    src/tests/test_spectral_precursors.py src/tests/test_spectral_sequences.py
    src/tests/test_spectral_events.py src/tests/test_spectral_clustering.py
    src/tests/test_spectral_invariance.py src/tests/test_spectral_constellation.py
    src/tests/test_spectral_mining.py src/tests/test_spectral_feature.py
    src/tests/test_spectral_tracking.py src/tests/test_spectral_narrative.py -q
514 passed, 1 warning in 97.06s (0:01:37)
```

R10 says that discovering the already-known is the validation signal: if the top-ranked mined
precursors are not recognisable, the prior is that the pipeline is broken rather than that
physics has been overturned. That makes this task a gate, and a gate is worth exactly what it
can fail. Everything below is about protecting that.

**The record, as the bridge reads it.** The same `advected_vortex_sequence` benchmark T4D.2
through T4F.5 are graded on, this time carried on a six-hourly calendar so that a lag in frames
has an hour to become.

```
=== the record, as the bridge reads it
  grid                cartesian grid 128x128, dy=31000.0 m, dx=31000.0 m
  observable          amplitude@no-declared-level
  cell                31.00 to 31.00 km   (ratio 1.000)
  cadence             6 hours per frame, measured from the record's own timestamps
  filter support      level 4: 16 cells   level 5: 32 cells
```

**Every mined pattern, measured off the footprints T4F.5 drew.** The horizontal scale is the
filter support, not the dyadic octave label; the two differ by a factor of two and an entry
declares which convention its envelope is in, so a catalogue read in the wrong one is wrong by
exactly that factor and looks entirely reasonable.

```
  id  bands                 support (cells)  scale (km)      separation (km)
  1   L4/HL + L4/LH          16 to 16         496 to 496      526.2 to 612.6
  2   L4/HL + L5/HL          16 to 32         496 to 992       26.5 to  71.1
  3   L4/HL + L5/LH          16 to 32         496 to 992      410.9 to 558.3
  4   L4/LH + L4/HL          16 to 16         496 to 496      268.4 to 523.4
  5   L4/LH + L5/HL          16 to 32         496 to 992      416.9 to 639.6
  6   L4/LH + L5/LH          16 to 32         496 to 992       43.3 to  72.2
  7   L5/LH + L5/HL          32 to 32         992 to 992      464.1 to 588.8
```

**The labels, against two declared entries.** `mesoscale-vortex-pair` is 300-700 km with a 3-24
hour lead, two members and a 200-700 km separation; `planetary-wave-train` is 3000-8000 km with
a 72-240 hour lead, and is a declared decoy -- nothing that size was planted here.

```
  1   recognised     consistent=mesoscale-vortex-pair    excluded=planetary-wave-train
  2   unrecognised   consistent=-                        excluded=both
  3   recognised     consistent=mesoscale-vortex-pair    excluded=planetary-wave-train
  4   recognised     consistent=mesoscale-vortex-pair    excluded=planetary-wave-train
  5   recognised     consistent=mesoscale-vortex-pair    excluded=planetary-wave-train
  6   unrecognised   consistent=-                        excluded=both
  7   unrecognised   consistent=-                        excluded=both
  discrimination  DISCRIMINATING -- 7 of 7 patterns were excluded by at least one entry
  digest (draft)   6f0670f808283c14d982ebfc672ce359...
  digest (frozen)  6f0670f808283c14d982ebfc672ce359...   <- signing does not move it
```

Patterns 2 and 6 are excluded by **separation alone**: they clear the scale envelope and their
members sit 26 to 72 km apart, far inside a 200 km floor. Pattern 7 is excluded by scale alone.
Both facts are asserted rather than observed, by re-labelling against the same entry with the
criterion removed and taking the difference.

**All three verdicts, on one report.** The same seven patterns, the same 120-frame planted event
series and the same corrected table, adjudicated under four declarations.

```
  envelopes that fit, top 1        PASS      4G may start: True
        rank 1 of the q_value ranking is rule 1 -> 2, whose antecedent is consistent with
        mesoscale-vortex-pair
  the same, ranked by support      FAIL      4G may start: False
        no antecedent among the top 1 by support is consistent with any declared phenomenon, on
        a catalogue that excluded 10 pattern-entry pairs and so was capable of recognising one.
        The standing interpretation of a FAIL is that the pipeline is broken (R10), and Phase
        4G does not start
  only the decoy entry             FAIL      4G may start: False
        no antecedent among the top 2 by q_value is consistent with any declared phenomenon, on
        a catalogue that excluded 7 pattern-entry pairs [...]
  an envelope nothing falls outside INVALID   4G may start: False
        the catalogue excluded no pattern on any criterion, so it could not have failed
        anything and a PASS from it would record the width of its envelopes rather than the
        content of the record
        the cross-reference could not have failed, so it cannot have passed either. This is not
        a FAIL: nothing has been learned about the pipeline
        every labelled pattern is consistent with every entry, so these labels separate nothing
```

Three things in that block are the point of the task. **The ranking key decides the verdict** --
T4F.4 measured that a q-value ranking and a support ranking put different rules first, and at a
declared top-N of one that difference is the whole gate -- so the key, the top-N, the catalogue
digest and a required naming of the documented event are hashed into a `GateDeclaration` before
the run. **A catalogue that excluded nothing returns INVALID, not PASS**, because recognition by
imprecision is this task's own failure mode. And **INVALID is not FAIL**: it licenses nothing,
says so in its own reason, and is T4C.5i's boundary arriving at the physical gate.

**The measurement that bounds the whole exercise, taken from the acquired record.** T4C.5k's
ERA5 crop is 161x161 at 0.25 degrees, latitudes -60 to -20 and longitudes 140 to 180. Its
meridional spacing is constant; its zonal spacing is not.

```
  observable          t@850 pressure_hpa
  cell                13.899 to 27.799 km   (ratio 2.000)
  zonal variation     61.08%   aspect ratio 1.3054
  (GridSpec.anisotropy's own warning: "The zonal metric varies by 61.1% across this lat/lon
   patch (13899 m at one edge, 26122 m at the other).")

  a footprint there:
    level 3    8 cells     111.2 to    222.4 km
    level 4   16 cells     222.4 to    444.8 km
    level 5   32 cells     444.8 to    889.6 km
    level 6   64 cells     889.6 to   1779.1 km
    level 7  128 cells    1779.1 to   3558.2 km
```

A scale in cells is therefore a **range** of kilometres spanning a factor of two on that crop,
before any measurement error at all, and an envelope narrower than that range cannot exclude
anything there. The gate does not assume this away: it counts the exclusions the catalogue
actually made and refuses to pass when there were none.

**The shipped reference, and why it is a draft.**

```
  status draft   entries 4   digest 27ad0d61c2b02bdc47f08e9515b77fa9...
    extratropical-cyclone-thermal-couplet   500-2000 km    6-48 h    checkable here
    frontal-wave                            200- 800 km    6-24 h    checkable here
    blocking-onset                         2000-6000 km   48-168 h   checkable here
    upper-level-pv-precursor                800-3000 km   12-72 h    UNASSESSABLE: needs
                                                                     pv@315 K or z@500 hPa
  overlapping pairs published beside the labels: 5 of the 6 possible pairs
```

Every entry carries a real citation (Sinclair 1995; Sanders and Gyakum 1980; Rex 1950; Hoskins,
McIntyre and Robertson 1985), and **none of the envelopes is a number quoted from any of those
papers** -- they are the implementer's first reading, which is exactly why the catalogue's status
is `draft` and why `physical_gate` refuses to adjudicate under it. Choosing which phenomena count
as recognisable, and in what ranges, is the scientific content of this gate; this module will
not make that choice on anyone's behalf.

The fourth entry is in the catalogue *because* this record cannot check it. Classical
cyclogenesis is an upper-level disturbance overtaking a low-level baroclinic zone, and a
single-level record of 850 hPa temperature has no upper level in it. That entry is
`unassessable` here for every pattern, permanently, and calling it `unrecognised` instead would
record the contents of a crop as a fact about the atmosphere.

**A lead in hours needs this record's clock.** A window is a lag on the event series' own
observation grid; a cadence is a property of the decomposed record. `lead_for` converts one with
the other only after checking that every frame the series searched is a frame of the record. The
120-frame planted series is not the 24-frame record, so its lead is refused by name, and a
series built on the record's own frames converts a 1-to-3-frame window to 6 to 18 hours.

**Mutation testing: 77 mutations in four batches, 76 killed and one shown unreachable.**
Every survivor was a real gap and five of them shared one cause -- every grid in the suite was
isotropic or had `dlat == dlon`, so "the cell's shorter side" and "the zonal side" were the same
number and nothing distinguished them; the `overlapping_pairs` conjunction had no pair agreeing
on exactly one of scale and lead; and no entry's cardinality or separation criterion had ever
been the sole reason for an exclusion. The others were untested refusals (`cardinality` below
two, an empty projection, a series naming no searched frames) and two unexercised gate branches
(a rule whose figure the report never measured, and a report carrying no measured figure at
all). All twelve were closed with real cases and re-run:

```
> mutate_t4f6.py       55/67 killed
> mutate_t4f6b.py      12/12 killed   (the survivors, after the tests that bind them)
> mutate_t4f6c.py       3/4  killed   (the rewritten length-unit path)
> mutate_t4f6d.py       5/6  killed   (the three receipt corrections below)
> mutate_t4f6d.py       6/6  killed   (after the test that binds the sixth)
```

The fourth batch-c mutant is **unreachable rather than missed**, and is recorded as such. Reading
the module back found that a physical grid's length was converted with a silent fallback -- an
unrecognised unit was treated as though it were already kilometres -- which is the exact
assumption this module refuses everywhere else, so it became a refusal. Both physical geometries
this programme ships declare `m`, so no `GridSpec` that can be built without registering a new
geometry can reach that refusal, and a test for it would have to mutate a global registry to
exist. The kilometre spellings were removed for the same reason: a conversion nothing can
exercise is not one to ship, and a grid declaring kilometres is now refused too.

**Three corrections from reading the module back, all of them about what a receipt says.**
`criteria_assessed` counted the observable check, which the decision rule excludes, so a receipt
could show a two beside a declared floor of two when a single envelope had actually been
decided; it is now `substantive_criteria_decided` and publishes the floor next to itself.
`criterion_exclusions` counted pattern-entry pairs rather than criteria, which the gate's own
FAIL reason had said correctly and the receipt key had not; it is now
`pattern_entry_exclusions`. And the "this label separates nothing" caution required that no
entry be unassessable, which made it quietly narrower than its own docstring -- an entry the
record cannot check discriminates in neither direction, so it can no longer rescue a label that
separated nothing. Each is bound by a test and by a mutant.

**Refusals this module makes.** An entry with no citation, no observable, a scale not in
kilometres, a lead not in hours, a negative lead, a cardinality below two or an unknown scale
convention. A catalogue that is empty or names one entry twice. A declaration with no documented
period, an unknown ranking key or a top-N that is not a positive integer. A gate under a draft
catalogue. A pixel grid asked for kilometres, and a physical grid whose length is declared in
anything but metres -- unreachable for the two geometries this programme ships, and a refusal
rather than a fallback because the only fallback available is to assume the unit. A record with
no calendar, or with uneven frames, asked for hours. A pattern with nothing on the map. A geography that is not one, a report that is
not one, a variable the record cannot name.

**The claim boundary.** A `recognised` label states consistency with a declared envelope and
never identity; every consistent entry is listed rather than the nearest chosen, and a pattern
consistent with every entry is flagged as separating nothing. The gate's boundary refuses
*cause*, *driver*, *mechanism*, *trigger*, *forecast* and *intervention* by name, states that
`PASS` licenses the start of Phase 4G and claims nothing about the atmosphere, and states that
`INVALID` must never be read or reported as a `FAIL`.

**Not met, and not runnable yet.** The acceptance -- "on a real ERA5 period containing a
documented cyclogenesis event, the mining pass ranks a `recognised` pattern corresponding to it
in the top results" -- **has not been run**, so T4F.6 is `PARTIAL` and **Phase 4G remains
gated**. Three things are missing and none of them is code:

1.  a maintainer must review, correct and freeze a reference catalogue, the shipped one being
    explicitly a draft;
2.  a documented cyclogenesis event on the 2018-2023 record must be declared with its source,
    which is the `period_justification` the declaration requires and does not invent;
3.  the mining pass itself -- T4D through T4F.5 over the real 8,764-frame record, on anomalies
    against a training-only climatology per R11 -- has never been run, on this record or any
    other.

T4F.5's own second acceptance clause, a real-ERA5 `recognised` pattern projecting onto a
physically sensible footprint, waits on the same run.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed, 2 warnings in 285.02s (0:04:45)
# re-run on the final tree, after the figures above were filled in
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
4190 passed, 4 skipped, 1 xfailed, 6 warnings in 2728.53s (0:45:28)
exit code 0
```

## T4F.7 -- where a rule holds, where it does not, and where it never took the test (2026-09-07, `ed-dev`)

`src/analysis_engine/spectral_regions.py`, verified by `src/tests/test_spectral_regions.py`
(54 test functions, 54 pytest cases). Measured 2026-09-07 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_regions.py -q
54 passed, 1 warning in 8.08s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_regions.py
    src/tests/test_spectral_reference.py src/tests/test_spectral_projection.py
    src/tests/test_spectral_queries.py src/tests/test_spectral_precursors.py
    src/tests/test_spectral_sequences.py src/tests/test_spectral_events.py
    src/tests/test_spectral_clustering.py src/tests/test_spectral_invariance.py
    src/tests/test_spectral_constellation.py src/tests/test_spectral_mining.py
    src/tests/test_spectral_feature.py src/tests/test_spectral_tracking.py
    src/tests/test_spectral_narrative.py -q
568 passed, 1 warning in 60.43s (0:01:00)
```

R14 says a pattern that holds in only one region is a local quirk until shown otherwise:
"we found a thing about the Alps" and "we found a thing about the atmosphere" are both valuable
and they are not the same claim. This task re-tests a rule in declared held-out regions and
labels it `general` or `regional`.

**The acceptance record is built to give four different answers at once.** One 128x184 field of
260 frames, carrying two structures whose orientations a detail bank separates cleanly: a zonal
ridge (sy=2, sx=24) that excites `L3/HL` and a meridional one that excites `L3/LH`. Five boxes,
and four constructions:

```
  A  (16-56, 16-56)     discovery   zonal at e+1, meridional at e+3     the chain
  B  (16-56, 72-112)    held out    the same, identically built         the chain
  C  (16-56, 128-168)   held out    zonal at e+2, meridional at e+5     the chain, longer lag
  D  (72-112, 16-56)    held out    meridional at e+0, zonal at e+6     both, in the wrong order
  E  (72-112, 72-112)   held out    nothing at all
```

Episode starts are jittered off any lattice on purpose: a periodic schedule is reproduced exactly
by every rotation that is a multiple of its period, which would put a mode of the shifting null
on the observed value and make a real effect look like chance.

**The four answers.**

```
  region  role       outcome                 occurrences  support  lift   q
  A       discovery  HOLDS                        76 / 13   38/76   3.89   0.029   (not in the family)
  B       held out   HOLDS                        81 / 21   55/81   4.47   0.0346
  C       held out   HOLDS                        23 / 20   14/23   3.72   0.0346
  D       held out   DOES_NOT_HOLD                24 / 23    0/24   0.00   1.0
  E       held out   NOT_ASSESSABLE                0 /  0       -      -      -

  verdict: regional -- held in 2 of the 3 held-out regions that could be assessed,
                       and did not hold in D
  corrected over 4 declared held-out regions, of which 3 returned a p-value
```

Declaring three held-out boxes (`B`, `C`, `E`) rather than four returns **`general`** on the
same record and the same figures. That is a different declared design, not a narrowing, and it is
why the partition carries a digest: a region set chosen after the outcomes are known is a
different design from the one that was declared, and the digest is what makes those two tellable
apart.

**`E` did not fail. It never took the test.** This is the distinction the whole task turns on,
and it has two forms, kept apart: a region carrying no occurrence of the antecedent at all, and
one carrying the antecedent but never the consequent -- where the base rate is zero, no lift
exists, and the honest statement is that the configuration was absent rather than that the
association was. Rendering either as `regional` turns an absence of data into evidence of
locality, which is exactly how a fact about a crop becomes a fact about the atmosphere.

**A held-out region may supply occurrences but may not help define a pattern.** A centroid is
fitted to whatever it was clustered from, so a clustering that saw the held-out regions has
defined the thing being re-tested using the data it is being re-tested on -- R6's leak with a map
in place of a calendar. `match_into_catalogue` holds every centroid and every calibrated radius
fixed and admits a signature only if the radius contains it; `identity_leakage` counts the
*fitted* members that came from held-out ground and refuses `general` when there are any.

**Building that measured something worth recording: T4E.2's signature cannot tell two band
orientations apart, and that is by construction.** It is invariant to rotation, which is what
T4E.2 was for. On this record the same `L3/HL` signatures sit a median **0.447** from their own
centroid and **0.585** from the `L3/LH` one, both far inside a calibrated radius of **0.959** --
an `HL` configuration matches the `LH` centroid about as readily as its own. A catalogue whose
patterns differ only by band orientation therefore cannot be matched into by signature distance,
and the suite's own catalogue is declared rather than fitted for that reason, with `fitted=()`
saying so.

**Membership is decided on T4F.5's footprints, and there are three ways of not having one.**
A maximum sits about one analysing width from what excited it, so placing a configuration by that
cell puts it in the wrong box at a boundary. Of 1,063 configurations:

```
  281  wholly inside exactly one declared box   (A 89, B 102, C 43, D 47, E 0)
  364  straddling two declared boxes            -- drawn closer than the footprints reach
  397  reaching out of the one box they touch   -- into ground nobody declared
   21  outside every declared box
```

The three failures are counted apart because they say different things about the *declaration*,
and none of them is quietly given to whichever region held more of it.

**Independence is published, never assumed.** The gap between every pair of regions is reported
in cells and in kilometres. With no declared decorrelation length the status is `UNESTABLISHED`
and `general` is refused; with one declared, the held-out regions closer to the discovery region
than it are named -- at 1,500 km on this record that is `B`, `D` and `E`, and not `C` at 2,232 km.
Closeness is judged on the **widest** kilometre reading the grid supports, which matters on a
lat/lon crop where a gap in cells is a range: the suite pins that on a spherical regrid where the
two ends of the range straddle the declared length and disagree.

**Physiography is declared with a source**, for the same reason T4F.6 requires a citation: a
field of one variable at one level carries no coastline and no orography, so anything the module
said about the surface would be invention. `general` is refused when no assessed region declares
one, and refused again when every assessed region declares the same class -- R14 asks for similar
*and* dissimilar ground, and three similar boxes do not answer it.

**Mutation testing: 47 mutations, 41 killed on the first pass and all six survivors closed.**
Every survivor was real. Two came from the fixture being tidier than the world: every grid in the
suite was isotropic, so "the widest kilometre reading" and "the narrowest" were the same number
and nothing distinguished them, and a pixel grid was never run through the independence check at
all. Two were unexercised branches of the matcher -- taking the nearest pattern rather than the
first, and refusing a signature of another family. One was a footprint test loose enough to pass
for a box with one bound taken from a maximum. And one was a grid check reachable only when no
region carries anything, since `events_from_catalogue` refuses a bad grid on every other path.
All six were closed with real cases and re-run:

```
> mutate_t4f7.py       41/47 killed
> mutate_t4f7b.py       6/6  killed   (the survivors, after the tests that bind them)
```

**Refusals this module makes.** A physiography with no source. A region with an unknown role, a
reversed box or a negative one. A partition with no discovery region, two of them, no held-out
region, overlapping boxes or two regions of one name. A grid that is not an `ObservationGrid`. A
pattern asked to precede itself. An empty catalogue offered to the matcher, which would be
clustering under another name. Anything that is not a `ConstellationSet`, a `RegionPartition`, a
`Geography` or a `Generalisation`.

**The claim boundary.** `general` says the rule was measured again in at least two held-out
regions, cleared the null in every one that could be assessed, and did so across declared and
differing physiography -- and nothing else. It refuses *cause*, *driver*, *mechanism*, *trigger*,
*forecast* and *intervention* by name, states that it is not a claim about anywhere the rule has
not been tested, and states that `unassessable` licenses nothing at all. The T4F.3 boundary the
figures inherit travels with it.

**Not claimed.** This has been exercised on a synthetic record only. No atmospheric region has
been compared with any other, and applying it to the acquired ERA5 record waits on the same
mining pass T4F.6's gate waits on.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed, 2 warnings in 465.89s (0:07:45)
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
4244 passed, 4 skipped, 1 xfailed, 6 warnings in 3468.18s (0:57:48)
exit code 0
```

## T4F.8 -- a proposal is a test, or it is a suggestion (2026-09-07, `ed-dev`)

`src/analysis_engine/spectral_proposals.py`, verified by `src/tests/test_spectral_proposals.py`
(50 test functions, 50 pytest cases). Measured 2026-09-07 with no network.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_proposals.py -q
50 passed, 1 warning in 7.16s

> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_proposals.py
    src/tests/test_spectral_regions.py src/tests/test_spectral_reference.py
    src/tests/test_spectral_projection.py src/tests/test_spectral_queries.py
    src/tests/test_spectral_precursors.py src/tests/test_spectral_sequences.py
    src/tests/test_spectral_events.py src/tests/test_spectral_clustering.py
    src/tests/test_spectral_invariance.py src/tests/test_spectral_constellation.py
    src/tests/test_spectral_mining.py src/tests/test_spectral_feature.py
    src/tests/test_spectral_tracking.py src/tests/test_spectral_narrative.py -q
618 passed, 1 warning in 106.51s (0:01:46)
```

**The platform already proposed follow-ups, and both of them were optimisers.**
`_propose_numerical_followup` saw a positive correlation between a parameter and a metric and
proposed a sweep of larger values of that parameter; `_propose_categorical_followup` fixed the
best-performing category and re-ran everything else. The direction in each is chosen so as to
improve the metric, which means **no outcome of the proposed run would retract the finding that
prompted it**. That is a useful thing to run and it is not a test, and a procedure that only ever
produces confirmations is not closing a loop, it is closing a circle. Both now say so in their
own docstrings and point at this module.

**A proposal that cannot come back negative is refused.** Every re-test carries a named
statistic, a direction and a threshold: the finding is retracted if the Wilson upper bound on the
re-test's own confidence, at the declared alpha, falls at or below the base rate that same re-test
measures -- that is, if the antecedent did no better than dropping the same window anywhere on the
lattice. The threshold is then checked against the range that statistic can attain, and three ways
it fails are pinned: a base rate of zero, where a Wilson upper bound is strictly positive at any
number of trials so no experiment could ever satisfy the condition; a base rate the ground could
not produce at all; and a threshold outside [0, 1], which every outcome meets and which therefore
distinguishes nothing.

**The prediction is digested before the record is read.** The digest covers the design alone --
the rule, the target region and its partition digest, the prediction, the refutation, the required
occurrences and the declared alpha, correction, null and ensemble size. It excludes the status,
the lead, the signatory and the date, because none of those was declared in advance. The suite
pins that both ways: signing does not move the digest, and neither does reading the same design
through a different record (one with a calendar, one without, giving a measured lead and a
refusal), while changing the target, the alpha, the correction, the null or the ensemble size each
does. As in T4F.6 the code will not sign -- `register(registered_by=..., registered_on=...)` needs
a name and a date, and a blank one is refused.

**Which rule gets which kind, and why there are two.**

```
  rule status                    re-test        power proposal
  PRECURSOR_SIGNATURE            proposed       refused: no absence to interpret
  NOT_DISTINGUISHED_FROM_NULL    refused        proposed
```

A confirmation proposed for a rule the record could not distinguish from its own surrogates would
manufacture a discovery out of a negative result, and it is the easiest way for a proposal engine
to look productive. But a tool that proposes follow-ups only for the things that worked has
publication bias built into it, so a negative gets a proposal too -- of the other kind. **A power
proposal carries no prediction and no refutation, and both are `None` rather than filled in with
something plausible**: nothing about the alignment is being asserted, so there is nothing an
outcome could retract. It can convert an absence that means nothing into one that means something
and it can confirm nothing at all. It is refused when the study that produced the negative already
carried the occurrences an effect of the declared size needs, because that is a real negative and
asking for more data until it changes is chasing it.

The effect size a power proposal is sized on is a **declared** doubling of the base rate, not the
confidence the record happened to show -- sizing a study on the effect that record produced is
sizing it on noise.

**Power is computed from quantities the record can be read for without performing the test.** The
quantity under test is the alignment between antecedent and consequent. The design quantities are
not: how many of the antecedent's occurrences have a wholly observed window, and how often the
consequent falls in a window dropped anywhere on the lattice, are properties of the record, and
neither counts the pair. Both helpers were **promoted out of `spectral_precursors`**
(`eligible_anchors`, `wilson_interval`) rather than copied, so there is one definition of each; the
promotion was behaviour-preserving and that suite's 52 tests were re-run to say so.

**Separability is required in both directions, and neither direction is the redundant one.**

```
  detectable   the interval around the predicted effect excludes the base rate  -> can confirm
  refutable    the interval around the base rate excludes the predicted effect  -> can retract
```

Sweeping every pair of proportions to two decimal places: **2,052 pairs have some `n` that could
detect the effect and could not retract it, and 1,973 have some `n` the other way about.** Neither
condition subsumes the other, so a design sized on one of them alone is systematically too small
about half the time. The suite pins one case of each -- 323 to 327 detectable and not refutable at
0.35 against 0.30, and 195 refutable and not detectable at 0.54 against 0.47 -- and pins the
margin at n=24, predicted 0.5, base rate 0.3, where the null interval's upper bound sits at
0.50004, four hundred-thousandths above the prediction. Reading either interval from the wrong end
would have sized that study at 24 rather than 25.

**The design interval is taken at `p * n` and not at a whole number of occurrences.** Rounding
there is not a rounding error: it makes separability **non-monotone in `n`**, so a study of 336
trials fails a separation that 335 passes, and a search for the smallest sufficient design returns
an arbitrary member of a jagged set. This was measured on the first implementation, which returned
357 where 335 would have done. With the rounding removed the suite asserts monotonicity outright
over five pairs and asserts the search returns the smallest sufficient `n`.

**The proposals are made about rules the record actually produced.** The T4F.7 acceptance record
is re-used because it already carries both things this task needs:

```
  region  rule status                    confidence  base rate  eligible  proposal
  A       PRECURSOR_SIGNATURE                 0.500      0.128        76   the finding
  B       PRECURSOR_SIGNATURE                 0.679      0.152        81   TESTABLE  (needs 8)
  C       PRECURSOR_SIGNATURE                 0.609      0.163        23   UNDERPOWERED with no
                                                                          record supplied
  D       NOT_DISTINGUISHED_FROM_NULL         0.000      0.163        24   power: needs 32
```

**A ground that was offered and could not be read borrows nothing.** A target supplying no record
at all and a target supplying a record with no readable base rate are different situations, and
conflating them substitutes the discovery region's own figure for ground that refused to give one.
The first is `UNDERPOWERED` and names the acquisition it needs; the second is refused, for both
kinds of proposal. **This was a real defect**, found by mutation testing and fixed here.

**Closing the loop.** `followup_experiment_config` returns the same `parameter_matrix` shape the
experiment engine already accepts -- so the platform can run its own follow-up -- with the rule,
the window and the target region added to the parent's matrix, the retraction condition stated in
words in the description, and the whole proposal record in the metadata. The parent's matrix is
not mutated. A refused proposal gets neither a config nor a registration; an underpowered one gets
both, because it is a design somebody may want to fund.

**Refusals this module makes.** A rule that is not a `PrecursorRule` or a partition that is not a
`RegionPartition`. A correction this programme does not implement, at proposal and at
registration. A target region the partition does not declare, or the discovery region itself. A
catalogue whose identities leaked into held-out ground. A refutation condition no outcome could
satisfy. A ground that was supplied and gave no base rate. A target confidence that is not a
proportion. A registration with no author or no date. A refused proposal offered for registration
or for a runnable config. A parent config that is not a mapping.

**Mutation testing: 58 mutations, 57 killed and one argued equivalent.**

```
> mutate_t4f8.py        46/55 killed
> mutate_t4f8b.py       11 mutants -- the eight survivors after the tests that bind them,
                        plus three for the defect they exposed -- 10/11, then 1/1 on the last
```

Nine survived the first pass. Eight were real: two refutation branches no test constructed, two
uncovered ends of the two design intervals, one power-proposal guard that ignored the surrogate
ensemble, one digest that could have covered the status and the lead without any test noticing,
one that could have dropped the target region from the design, and one correction check at
registration. All eight are now bound. The ninth is **equivalent**: an early return in
`required_occurrences` for a prediction no better than the base rate, whose absence changes
nothing because the search below reaches the ceiling and returns the same `None`. It is kept
because it says so in one line rather than after seventeen doublings, and the duplicate copy of
that same comparison in `propose_re_test` was removed so there is one place it is decided.

**The claim boundary.** A proposal is a design, not a result. Every figure marked `predicted_` is
what the discovery would imply if it holds on the new ground and none of them has been measured
there. It refuses *cause*, *driver*, *mechanism*, *trigger*, *forecast* and *intervention* by
name, states that registering one makes it traceable rather than true, states that `TESTABLE` is
not a prediction that the re-test will succeed, and states that a power proposal can confirm
nothing. The T4F.3 boundary the predicted figures inherit travels with it.

**Not claimed.** No proposal has been run. This module produces designs; whether the platform
executes its own proposals is Phase 4G's question. Nothing here has been applied to the acquired
ERA5 record, which waits on the same mining pass T4F.6's gate waits on.

**Documentation guards.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_documentation.py -q
29 passed, 2 warnings in 496.51s (0:08:16)
```

**Full backend suite.**

```
> .\.venv\Scripts\python.exe -m pytest src/tests -q
4294 passed, 4 skipped, 1 xfailed, 6 warnings in 3991.25s (1:06:31)
exit code 0
```

## The T4F.6 gate run, attempted (2026-09-07, `ed-dev`) -- D96

The maintainer asked for the gate run. It cannot be performed, and the reason is not either of
the two declarations it was known to be waiting on. **T4E.3's identity clustering does not scale
to a real record, by about ten orders of magnitude.** This entry records the attempt, what it
produced, and the measurement that stopped it.

### What was produced, and stands

**The declaration** -- `data/mining_declarations/t4f6-tasman-mining-declaration.json`, written before
the record was read, sha256 `a649afccda2e3f4025ccb7af3123a3cbae58ada143772a7aea45affd181fcaac`.
It fixes the detection threshold, the co-presence cap, the split, the climatology model and the
transform depth, and lists what it does not fix.

Two choices in it are worth naming because both were nearly made the wrong way round:

*   **The threshold.** At sigma 3.5 -- the value every fixture in this programme uses -- a real
    atmospheric frame holds 25 co-present tracks and `extract_constellations` refuses against
    its default cap of 24. At sigma 4.0 it fits. Choosing 4.0 for that reason would have let
    the software's budget decide what counts as a feature. The cap was raised instead; the
    threshold was left where it was.
*   **The transform depth.** The programme's existing configuration is four levels. On this grid
    the bridge measures 13.899 to 27.799 km per cell, so four levels reach 445 km, and three of
    the catalogue's four entries would have come back unrecognised for a reason with nothing to
    do with the atmosphere. Level 8 is refused by R13 (support 256 px on a 161-cell field), so
    seven is the deepest this record admits, and levels 3 to 7 are what the catalogue's own
    envelopes reach:

```
  level  support     reach on this grid     catalogue entry            envelope
    3     8 cells     111 -  222 km         frontal-wave                200-800   levels 3-5
    4    16 cells     222 -  445 km         cyclone thermal couplet     500-2000  levels 5-7
    5    32 cells     445 -  890 km         upper-level PV precursor    800-3000  unassessable:
    6    64 cells     890 - 1779 km                                               single level
    7   128 cells    1779 - 3558 km         blocking-onset             2000-6000  larger than
    8   256 cells    REFUSED by R13                                               the record
```

`blocking-onset` is **unassessable on this crop permanently**, and for a physical reason rather
than a software one: the crop is 2,226 to 4,184 km wide zonally and 4,453 km meridionally, so
the upper half of its envelope does not fit inside the record at all.

**The training-only climatology (R11)** -- fitted on the 5,844 frames of 2018-2021 alone,
streaming one frame at a time so the record is never resident. Rank 9 of 9, condition number
3.237, `fitted_on_all_frames: False`.

```
  train  raw var 47.281 -> anomaly var 10.330   78.2% removed   (40 sampled frames)
  test   raw var 49.568 -> anomaly var 10.744   78.3% removed   (40 sampled frames)
```

Those two numbers agreeing is the check that matters: a climatology overfitted to its training
years removes visibly less out of sample, and this one does not.

**The declared configuration runs.** 600 frames of real anomalies at levels 3-7 decompose in
15.9 s, float32 coefficients are accepted downstream, and tracking takes 11.2 s for 9,248
features in 6,688 tracks.

### D96 -- `cluster_signatures` is O(n^3.7) and has never been run on more than a few dozen points

`cluster_signatures` is complete-linkage agglomerative clustering implemented directly: every
iteration enumerates all pairs of surviving clusters, recomputes every cross-distance from
scratch, and merges one pair. Measured on real signatures from this record:

```
      n   seconds   patterns   largest   measured exponent
     50      5.52          4        25
    100     39.65          7        33        2.84
    200    347.37         10        74        3.13
```

Extrapolated at exponent 3:

```
  n =    86,562  (600 frames)          2.8e10 s     ~900 years
  n =   843,113  (5,844 train frames)  2.6e13 s     ~800,000 years
```

**A first measurement of this defect was pessimistic by about a factor of two and is corrected
here rather than quietly replaced.** It reported 7.51 / 60.58 / 766.94 s and an exponent of 3.66.
The tolerance had been calibrated from three arbitrary signatures, which is not what
`calibrate_signature_tolerance` is for: it takes repeated measurements of *one physical
configuration* and returns their largest pairwise distance as a noise floor. Calibrated properly
-- from seven successive observations of one track pair, which is what "the same configuration
observed again" means on a real record -- the tolerance is **0.4797** against the 0.8387 first
used. A tolerance that is too large merges more, and this algorithm's cost grows with cluster
size, so the error ran in exactly the direction of the defect being claimed. The defect stands
unchanged; the figures did not.

600 frames of this record produce 86,562 cardinality-2 constellations; the training period
produces about 843,000. Every test in this repository clusters tens of points, no test clusters
more than a few dozen, and **no document states the algorithm's cost anywhere**. It was never
characterised, so it was never known to be the binding constraint.

This is not a compute budget that a longer run would clear. An optimal exact complete-linkage
implementation is O(n^2 log n) with O(n^2) memory, which at n = 843,000 is 7e11 pairs and
infeasible on any machine this programme targets. **The identity step needs a different
algorithm at this scale, not a faster version of this one** -- blocking or bucketing on the
signature space, or a streaming leader assignment -- and since a change there changes what a
"pattern" *is*, it is a scientific component and needs its own task, its own acceptance, and a
demonstration that it agrees with the present definition wherever the present one can be run.

Two budgets were also found to bind, and both refuse rather than sample, which is correct:

```
  max_nodes_per_frame   45 observed in 600 frames against a raised cap of 48
  max_constellations    ~843,000 projected against a default of 200,000
```

### An observation, not yet a finding

At n = 200 the clustering returned **4 patterns**, and at n = 50 it returned 3, against a
calibrated tolerance of 0.8387. A catalogue of four patterns drawn from thirty-one thousand
configurations is barely discriminating, and it points the same way as T4F.7's measurement that
T4E.2's signature is invariant to rotation by construction and cannot separate band
orientations. Whether the tolerance calibration is too permissive on real data is a separate
question from D96 and has not been established here -- n only reached 200.

### What this means for the gate

**Phase 4G remains gated, and the reason has changed.** It was recorded as waiting on three
things: a maintainer-frozen catalogue, a documented event declared with its source, and a mining
pass that had never been run. The first two still wait on the maintainer and are unchanged. The
third is not merely unperformed: **on the current implementation it cannot be performed**, and
the roadmap's wording understated that because nobody had measured the identity step.

Nothing here adjudicates anything. No pattern was labelled, no gate verdict was reached, and the
draft catalogue remains unsigned.

## D97 -- the identity calibration has no valid input on a real record (2026-09-07, `ed-dev`)

Measured while deciding how to fix D96. It is recorded before any fix because it changes what
the fix should be: D96 -- the identity step being unusably slow -- turns out to be a symptom of
this, and making the clustering faster would have delivered the same unfounded answer sooner.

### What the calibration asks for, and what a real record can give it

`calibrate_signature_tolerance` takes `replicates` -- *"at least two measurements of the same
physical configuration"* -- and returns their largest pairwise distance, describing it as a
measured noise floor with an approximate single-comparison **false-rejection** rate of
1/(pairs+1). That is a sound procedure when replicates exist. Every fixture in this programme
plants the same structure repeatedly and varies only the noise, so every test passes.

**A real atmospheric record contains no replicates.** The atmosphere is never in the same state
twice. The closest thing available is the same tracked configuration at successive frames, and
that is not a replicate either -- it is the same structure six hours older. Measured on the
acquired record, the distance between two observations of one track set grows monotonically with
the gap between them:

```
  gap (frames)      n     median      q95      max
        1        1686     0.2359   0.6612   1.1478
        2         983     0.3148   0.6989   1.1125
        3         280     0.3176   0.7291   1.0497
        4          99     0.3673   0.8229   0.9240
        5          36     0.4902   0.8717   0.9091
```

That gradient is physical evolution. Whatever is handed to the calibration on a real record, the
number that comes back is dominated by how much the atmosphere changed between looks, not by how
precisely the signature measures.

### The error rate that was never computed

Against pairs drawn from *different* track sets (median 0.5451, q95 1.1100), even the tightest
same-configuration sample -- adjacent frames only -- overlaps almost completely:

```
  radius                          same-configuration split   different-set pairs admitted
  1.1478  (max of adjacent)                  0.00%                      97.08%
  0.6612  (q95 of adjacent)                  5.04%                      61.69%
  0.2359  (median of adjacent)              50.00%                      11.88%
```

There is no good radius. At a 5% false-split rate the radius admits **62%** of pairs drawn from
different configurations; to admit only 12% it must split half of all same-configuration pairs.
The standardised separation of the two distributions is 1.172.

**Confirmed on three 480-frame slices spanning the record**, so this is a property of the
instrument on this data and not of one window:

```
  slice          separation   admitted at a 5% false-split radius
  frames    0- 480   1.172                61.69%
  frames 4000-4480   0.879                69.27%
  frames 7500-7980   1.620                60.20%
```

### One caveat, stated plainly

"Different track sets" is **not** the same as "different pattern types". Grouping distinct
occurrences into one pattern is precisely what the clustering is *for*, so a pair from two track
sets falling inside the radius is not by itself an error, and the admission figures above are
therefore not a false-acceptance rate. What the measurement does establish is narrower and still
decisive: the spread between observations of *one* configuration is comparable to the spread
across the whole population, so any radius calibrated from the former necessarily admits most of
the latter.

### Why this gates the phase

Because most pairs lie inside the calibrated radius, the tolerance graph on this record is a
single connected component from about a thousand configurations upward, and the clustering
returns a few very large patterns -- 25 patterns from 8,000 configurations, averaging 320
members. Any precursor rule mined on those identities is a statement about very large,
heterogeneous categories, and the T4F.6 gate would be adjudicating patterns whose discrimination
has never been measured.

This also explains three earlier observations that were recorded separately and never connected:
T4F.7's measurement that T4E.2's signature is invariant to rotation by construction and cannot
separate band orientations; and the fact that both the T4F.5 and T4F.7 suites built their
catalogues by **declaration** rather than by clustering, because clustering would not separate
what those tests needed.

**Not claimed.** No fix is attempted here. Whether the signature can be made more discriminating,
and whether identity should be declared rather than discovered, are T4E.6 and T4E.7's questions.
Nothing about the acquired record's physics is claimed either: this is a measurement of the
instrument, taken on real data.

## T4E.6 -- a radius has two error rates, and this programme measured one (2026-09-07, `ed-dev`)

`src/analysis_engine/spectral_clustering.py`, verified by
`src/tests/test_spectral_discrimination.py` (22 test functions). Closes the first half of D97.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_discrimination.py -q
22 passed, 1 warning in 7.91s

> .\.venv\Scripts\python.exe -m pytest src/tests -q
4348 passed, 4 skipped, 1 xfailed, 6 warnings in 2540.82s (0:42:20)
exit code 0
```

The first attempt at that full figure was **discarded rather than published**: the laptop slept
eight hours into it and resumed the next morning. The run survived and was still working when
checked -- 19.97 s of CPU in a 20 s window -- but several tests here judge against wall-clock
budgets, and a suspension can produce a spurious pass as readily as a spurious failure. It was
stopped and re-run on an awake machine. That earlier run also found a real defect and is the
reason `data/mining_declarations/` exists: the T4F.6 mining declaration had been written into
`data/gate_receipts/`, which the read-only T4C gate record serves, and the store correctly
reported itself unreadable rather than skipping a file whose schema it did not know.

`calibrate_signature_tolerance` returned a radius and stated a **false-rejection** rate. The
complementary rate -- how often that radius calls two configurations that are not the same one a
single pattern -- was never computed, and it is the rate that decides whether a pattern means
anything. It cannot be computed from replicates alone, because a set of replicates contains no
example of two different things. So the calibration now takes an optional `contrast` population
and returns a `Discrimination` on every tolerance it produces: both rates and both distance
distributions where a contrast was supplied, and a **named refusal** where it was not.

**The refusal is the design.** `ADMISSION_RATE_NOT_MEASURED` is not zero and is not small; the
record omits the keys it did not measure rather than emitting nulls, because a key present with
a null invites a reader to treat it as a measurement of nothing.

**`best_operating_point` returns `None` when no radius holds both rates at the level asked for**,
and that is the answer this task exists to make reachable. It is a fact about the signature
rather than about the search.

### What it measured on the acquired record, and the two things that came out

Calibrating from the longest run of repeated observations -- which is what the pipeline naturally
reaches for, being where the most replicate pairs are:

```
  radius 0.641357   false-split 0.00%   admitted 21.47%   separation 2.535
  same        min 0.0316  q05 0.0912  median 0.3537  q95 0.6356  max 0.6414
  different   min 0.0076  q05 0.3048  median 0.8840  q95 1.0898  max 1.1790

  both rates <= 5%    NONE EXISTS
  both rates <= 10%   NONE EXISTS
  both rates <= 20%   radius 0.5380  (split 10.7%, admitted 14.9%)
  best achievable     radius 0.5380, worst rate 14.9%
```

**First: the radius is arbitrary.** Repeating the calibration on 25 different configurations from
the same record, with the same metric and the same instrument:

```
  radius      min 0.1683   median 0.6999   max 1.1220     a factor of 6.7
  admission   min  8.9%    median  21.5%   max  84.5%
  separation  min 0.681    median  2.420   max  4.108
```

The longest run sits at the 48th percentile of those radii and the 52nd of those admission rates,
so the pipeline's natural choice is not biased -- it is simply arbitrary, which is worse than
biased because nothing signals it. A radius that moves by a factor of nearly seven depending on
which configuration happened to supply the replicates is a property of the sample and not of the
instrument.

**Second: the one rate the calibration did report is a tautology at its own operating point.**
The radius *is* the largest replicate-pair distance, so no replicate pair can exceed it and the
measured false-split rate at the calibrated radius is identically **0.00%** -- on every one of
those 25 runs, and on any input whatever. The `1/(pairs+1)` figure the basis quotes is an assumed
rate for *future* comparisons under a distributional assumption, not a measurement of this
record. Both facts are pinned by tests, so a later calibration that reports a real, non-tautological
split rate will break them and have to replace them.

### What this does not do

**It does not move the radius.** T4E.6 measures what the existing calibration produces; choosing
a better radius is T4E.7, and the suite asserts that supplying a contrast leaves both the value
and the replicate distances unchanged.

It also does not settle what "different" means. The contrast population here is drawn from other
track sets, and two different track sets may legitimately be the same *kind* of configuration --
which is what clustering is for. So the admission figures are an upper bound on what a labelled
contrast would give, and the suite says so rather than reporting them as a false-acceptance rate.
On the repository's own tracked fixture the same overlap appears and no radius reaches both rates
at a fifth; that is pinned too, and if a future signature makes it fail, the signature improved
and the assertion should be tightened rather than removed.

**Mutation testing: 19 mutations, 19 killed.** Two survived a first pass and both were real: a
distance falling exactly on the radius was not tested on either side, and the operating-point
search returned the only qualifying radius on a population where "first" and "smallest" could not
be told apart. Both are now bound by populations built to distinguish them.

## T4E.7 -- a radius measured against a null, and a record that will not give one (2026-09-08, `ed-dev`)

`src/analysis_engine/spectral_null_calibration.py`, verified by
`src/tests/test_spectral_null_calibration.py` (77 test functions, 93 cases). Closes the second
limb of D97 as a *method*; does **not** close D97, and opens D98.

```
> .\.venv\Scripts\python.exe -m pytest src/tests/test_spectral_null_calibration.py -q
93 passed, 1 warning in 27.96s
```

The replicate calibration has no valid input on a real record (D97) and the one rate it reported
is zero by construction (T4E.6). This asks a question the record can answer instead: how many
pairs of signatures land close together, and how many would land that close if the record
contained no recurring configuration at all? The second number is a null. The difference is what
recurrence has to explain.

### The acceptance, on a synthetic record with a planted identity

Twelve configurations, each observed ten times with 2% jitter; nineteen null populations of the
same size with no configuration observed twice. The calibration sees only distances.

```
  status    MEASURED          radius 0.033979
  separation statistic 0.1593, p = 0.05 (the floor at 19 members), band 0.0595
  recurrence fraction 0.0696       true fraction 0.0756
  published  contamination 0.0000   relative loss 0.0699   admission 0.0000

  against the labels it never saw:
    0 of 6,600 pairs that are NOT the same configuration are admitted
    462 of 540 pairs that ARE the same configuration are grouped (85.6%)
```

The refusal path is exercised on the same machinery with every configuration observed exactly
once: `OBSERVED_DOES_NOT_DEPART_FROM_NULL`, no radius.

### Four design decisions, each forced by a measurement that contradicted the first draft

**A corrected per-radius sweep cannot reject anything.** A surrogate p-value floors at
`1/(1+n)`; Benjamini-Yekutieli over 64 radii needs a raw p near `alpha / (64 * H_64)`, about
`1.6e-4`. `multiple_comparisons.required_surrogates` puts that at **7,588 surrogates** for the
acquired record's sweep, each one a full pipeline re-run over a whole surrogate record. The first
implementation did exactly this and returned a clean-looking negative on a synthetic record whose
planted grouping was obvious. The sweep is now **one maximum statistic** -- the largest excess
anywhere on the curve, with its null distribution built leave-one-out from the ensemble -- which
is a simultaneous band across every radius at once and owes no correction.

**A grid spaced evenly in quantile cannot see the tail.** Its first point sits at `1/n_radii`,
above the entire same-configuration mass whenever recurrence is under about 1.5% of pairs. The
grid is geometric in quantile, and `n_radii` now defaults to whatever resolves the declared
target rate: a sweep of 64 points over 4,000 pairs steps by **13.9%**, so a target of 10% was
unreachable for a reason that had nothing to do with the record, and passing a grid too coarse
for the target is now refused outright with the count that would suffice.

**The mixture fraction read in the bulk is noise.** A record with `C` distinct configurations
supplies only `C` draws of the unrelated-pair distribution however many signatures it contains,
so `F_obs` there carries an error of order `1/sqrt(C)` while the fraction being estimated is of
order `1/C`. Read at the null's median, the estimator returned **negative** fractions
(-0.4353, -0.7208, -0.6211, -1.0133 across four tuning points) on a synthetic record whose
planted grouping the separation test had just detected cleanly. It is now read only where the
excess clears the band and the null fraction is at or below a declared cap of 0.25.

**The contamination rate is an estimate, not a bound.** An earlier draft of this module claimed
it was an upper bound. Measured against labels on a planted synthetic, at **17 of 73 radii** the
published rate sat *below* the truth, by up to 0.11 even inside the region a radius is chosen
from. It substitutes the null's close-pair fraction for the record's own unrelated one, and the
two differ by however well the null describes the record. It matched the truth exactly at the
radius actually chosen (0.0000 against 0.0000) and wanders in the bulk, where no operating point
is ever taken. The limitation is pinned by a test that fails if the rate ever becomes a bound.

### No absolute split rate is published, because none is identifiable

It would be `1 - F_same(r)` with `F_same = excess / pi + F_null`. The fraction is read where the
same-configuration component is assumed saturated, so where it has not saturated the estimate
returns `pi * F_same` and every split rate derived from it is optimistic. Measured on a planted
synthetic against labels the module never saw, the first implementation published **1.91%
against a true 12.53%**. An error rate optimistic by six-fold is worse than an absent one. This
is the successor to T4E.6's finding that the old rate was zero by construction: the fix is not a
better estimate of that quantity but the statement that the quantity cannot be had from a record
with neither labels nor replicates.

### A refusal that nothing could reach, removed

`NO_RECURRENCE_FRACTION_ESTIMABLE` was a fifth status until mutation testing asked what would
produce it. The largest observed distance puts `F_obs` at 1 while `F_null` can be at most 1, so
the maximum excess is non-negative on **every** input; it is zero only when every null distance
lies inside the record's own range, and then every leave-one-out statistic is non-negative too
and the p-value is 1. Confirmed over 500 random ensembles: the smallest maximum excess seen was
0.1539 and no trial was significant with a non-positive one. The status was removed rather than
left as a refusal nobody could ever see.

### Where the estimator works, measured rather than asserted

Eleven synthetic records, each with a planted grouping, sweeping the number of recurring
configurations and how often each is observed. `pi` is the true mixture fraction. `R` recurring configurations are each observed `k` times
alongside `S` configurations observed once, for `N` signatures in all; the jitter is 0.02 unless
a row says otherwise, and every null member is `N` configurations observed once each.

```
  R    k    S |    N     pi | status                              pi_hat  spread  radius
  5   40    0 |  200 0.1960 | RECURRENCE_FRACTION_UNSTABLE        0.1517  0.3171       -
 10   20    0 |  200 0.0955 | RECURRENCE_FRACTION_UNSTABLE        0.1520  0.4867       -
 20   10    0 |  200 0.0452 | RECURRENCE_FRACTION_UNSTABLE        0.0805  0.1191       -
 40    5    0 |  200 0.0201 | OBSERVED_DOES_NOT_DEPART_FROM_NULL       -       -       -
 20   10  200 |  400 0.0113 | NO_RADIUS_HOLDS_BOTH_RATES          0.0435  0.0273       -
 50    8    0 |  400 0.0175 | OBSERVED_DOES_NOT_DEPART_FROM_NULL       -       -       -
100    4    0 |  400 0.0075 | NO_RADIUS_HOLDS_BOTH_RATES          0.0402  0.0055       -
 25   16    0 |  400 0.0376 | MEASURED  (jitter 0.02)             0.0377  0.0199  0.0462
 25   16    0 |  400 0.0376 | MEASURED  (jitter 0.005)            0.0399  0.0227  0.0115
 25   16    0 |  400 0.0376 | MEASURED  (jitter 0.08)             0.0331  0.0097  0.1311
  0    0  400 |  400 0.0000 | OBSERVED_DOES_NOT_DEPART_FROM_NULL       -       -       -
```

**No row publishes a wrong radius.** Where it measures, the fraction lands within 0.005 of the
truth; everywhere else it refuses, and the pure-null record refuses on the separation test. The
unstable rows are the regime where a record contains too few distinct configurations for its own
unrelated-pair distribution to be sampled, which is a real limitation and is refused rather than
averaged over.

### On the acquired record: no radius, and the reason is specific

480 frames of the acquired 8,764-frame ERA5 record, levels 3-7, sigma 3.5, cardinality-2
signatures. The null is nineteen `spatiotemporal_phase` surrogates of the anomaly sequence put
through the identical pipeline -- decomposition, tracking, constellations, signing -- so it
carries the record's full 3D power spectrum and its temporal autocorrelation and no coherent
feature that could recur.

```
  record        69,580 signatures      null members 13,264 - 19,364, median 16,090
                                        6,000 pairs        40,000 pairs
  status                     RECURRENCE_FRACTION_UNSTABLE   RECURRENCE_FRACTION_UNSTABLE
  radius                                            none                           none
  maximum excess                    0.0916 at r = 0.8271           0.0858 at r = 0.9347
  simultaneous band                               0.0387                         0.0398
  whole-sweep p                                     0.05                           0.05
  radii clearing the band                      13 of 77                       12 of 97
  smallest of them             r = 0.3707, F_null 0.2515      r = 0.3926, F_null 0.2768
  a corrected sweep would need            7,588 surrogates             10,004 surrogates
```

**The two budgets agree, so the refusal is about the record and not about the sample.** A 6.7x
increase in sampled pairs moved the band from 0.0387 to 0.0398 and changed nothing else.

**In the close-pair tail the excess is negative.** The record has *fewer* near-identical
signature pairs than its own surrogate null, at every radius up to a null fraction of about 0.05:

```
  radius    F_obs    F_null   excess
  0.0342   0.0072   0.0091  -0.0019
  0.0614   0.0170   0.0226  -0.0056
  0.0876   0.0302   0.0360  -0.0058
  0.1019   0.0400   0.0430  -0.0030
  0.1207   0.0532   0.0520  +0.0011
```

Phase randomisation produces a homogeneous field whose features are generic and therefore alike;
the record's are diverse. The excess only becomes positive from about `r = 0.12` and only clears
the band from `r = 0.37`, where the null already admits a quarter of its pairs -- which is where
a difference between two *feature populations* would show, not where recurrence would.

**And the null is not clean**, which the calibration now says by name: the record produced 69,580
signatures and the median surrogate 16,090, a factor of **4.3**. The null was meant to destroy
recurrence and keep the features; producing a quarter of them means it removed the structure the
features are found in. This is D98. The warning is only visible because population sizes travel
with the distances: both populations had been sampled to a common number of pairs, so no pair
count could have shown it.

### How much recurrence this record could contain at all

Under the strictest available reading of "the same configuration" -- the same tracked
constellation, observed again at a later frame -- measured directly on the record:

```
  signatures                        69,580
  distinct tracked constellations   64,153
  observations per constellation    max 8, median 1, mean 1.08
  observed more than once            4,444  (6.9% of constellations)
  same-configuration pairs           6,838 of 2,420,653,410
  ceiling on the mixture fraction    2.825e-06
```

That ceiling is **13,700 times smaller than the simultaneous band**. Under this reading, exact
recurrence is undetectable by pair-distance mixture on this record by four orders of magnitude,
and it is arithmetic rather than an implementation limit: the fraction is quadratically small
because the record produces many signatures and each configuration is seen about once.

**This does not settle the broader question**, and must not be read as though it did. Clustering
exists to find that *different* tracked constellations are the same recurring kind, and the
fraction of pairs that are the same kind is not bounded by the number above. What the acquired
record shows is that the strict reading is hopeless at this scale, and that the only excess the
record does exhibit against this null sits in the bulk where a feature-population difference
would sit.

**Mutation testing: 38 mutations, 38 killed**, across two batches. Twelve survived the first pass
and eleven of them were real gaps in the suite -- boundaries exactly on the band and exactly on
the cap, a null member tying the observed fraction, the contamination rate's exact value rather
than an inequality, the loss's floor at zero, the best-achievable being the smallest worst rate,
a spread exactly at the tolerance, a distance exactly at a radius, and duplicate radii. The
twelfth is the removed status above.

## T4E.8 slice 1 -- the record's own labels, and the radius they give (2026-09-08, `ed-dev`)

Receipt: `data/identity_calibration/t4e8-replicate-census.json`. No module changed; this slice is
a measurement that reversed the task's premise before any of it was built.

T4E.8 was specified as *build a better null*. Before building one, the acquired record was asked
what its own labels say, because a signature carries `track_ids` and a `time`, so the strictest
reading of identity -- the same tracked constellation, observed again -- is **already labelled in
the record**. Labels need no null and no mixture: both error rates are counts.

### The record has replicates, and `calibrate_signature_tolerance` says it has none

Its docstring reads "an atmospheric record contains no replicates at all". The 480-frame slice
holds 69,580 signatures over 64,153 distinct tracked constellations, of which **4,444 are
observed more than once**, giving **6,838 within-key pairs**. The claim is wrong in letter. It is
right in effect, and this slice measures why.

### The labelled discrimination bounds anything a null could achieve

At the best balanced operating point over all 6,838 pairs, against 20,000 sampled unrelated ones:

```
  weighting                 AUC    radius   grouped(same)   admitted(different)
  declared 1,1,1,1       0.7827    0.3621          0.7164                0.2883
  geometry only          0.7995    0.4060          0.7314                0.2679
```

**A null cannot beat labels.** With perfect ground truth the record groups 73% of genuine repeats
while admitting 27% of unrelated pairs, so no null -- however clean -- could have produced a
defensible radius under this reading. **D98 is real and is not the binding constraint**, and
T4E.8 as specified could not have reached its acceptance.

### What the within-key distance is actually measuring

It is monotone in how far the tracks physically moved, over pairs one frame (6 h) apart:

```
  furthest track moved      n    median   q0.95      max   unrelated admitted at median
        0 cells            32    0.0488  0.3634   0.3951   0.0111
      <= 1 cell           454    0.1055  0.6118   0.8746   0.0432
      <= 3 cells         1951    0.2208  0.6835   1.1573   0.1113
      <= 6 cells         2852    0.3208  0.7069   1.2068   0.2356
       > 6 cells          138    0.2947  0.8036   1.0229   0.1927
```

Median displacement over all within-key pairs is 3.25 cells. So the strict reading conflates *one
state measured twice* with *an evolving system observed twice*, and the second dominates: this is
physical evolution, exactly as the docstring warned, and it is the reason the radius is
indefensible -- not the null.

### The band flips more often than not

**A node changed wavelet band in 4,160 of the 6,838 within-key pairs -- 60.8%.** The same two
tracked features, one frame apart, are more often than not detected in different bands. Signing
with `scale_invariant=False` puts the band into the `scales` block, so this instability enters the
comparable vector directly. That is **D99**.

### Once both are excluded, a radius exists on the acquired record

The stratum that is a noise floor rather than a trajectory -- one frame apart, no band change,
furthest track moved at most one cell -- holds 124 pairs, 1.8% of the within-key set:

```
  stratum                        n      AUC   median   r@90% recall   unrelated admitted
  clean (noise floor)          124   0.9692   0.0472         0.1400               0.0633
  clean, weights 1,1,0,1       124   0.9726   0.0211         0.0972               0.0449
  same cell, band flipped       64   0.7274   0.3485         0.4339               0.3716
  evolving                    6650   0.7797   0.3055         0.6096               0.5613
```

**This is the first defensible identity radius this programme has measured on the acquired
record**: 0.0972, grouping 90% of genuine stationary repeats while admitting 4.5% of unrelated
pairs, both rates absolute counts against the record's own labels, with no null and no mixture
anywhere in the derivation.

Three limitations travel with it and none is cosmetic. The stratum holds **124 pairs**, so the
90th percentile rests on about a dozen observations and its tail is coarse. The stratum is a
**biased** sample of "the same configuration" by construction -- it is a noise floor, so the
radius admits near-stationary repeats and will reject a configuration that recurs after moving,
which is the thing clustering exists to find. And the band-flipped row shows that two observations
of the same two tracks, in the same place, one frame apart, are as far apart as unrelated pairs
whenever the band changes: **a detection instability wearing the costume of a signature
difference.**

### The strengths block carries no discrimination

Weighted alone over the 6,650 evolving pairs it scores an **AUC of 0.5053** -- a coin flip -- while
contributing 4.3% of same-pair squared distance against 1.3% of different-pair, so it widens the
gap it is weighted to close. Dropping it improves every stratum measured, though within the clean
stratum the improvement (0.9692 to 0.9726 on 124 pairs) is inside the noise and is not claimed.
That is **D100**.

Two things measured here are **not** findings and are recorded so they are not read as such. The
`bearings` block reports zero on every pair because this family is cardinality 2 with
`has_bearings: False` -- the block does not exist, and an earlier draft of the census reported its
absence as a measurement. And single-block weightings re-minimise the node alignment over that
block alone, so "scales only" scoring an AUC of 0.378 is an artefact of re-alignment rather than
a measurement of the scales block; the per-block shares above are taken under the full metric's
alignment, which is the only way they mean anything.


## T4E.8 slice 2 ? spatial identity, independently measured error rates, and unmet acceptance (2026-09-08)

Implemented an opt-in `spatial_geometry` signing mode through `SIGNATURE_MODES`. It compares
pairwise spatial separations and admitted bearings in a declared record/grid and source;
detector scales and band-normalised strengths remain carried. The existing scalar and
accelerated clustering contracts consume the new comparable blocks and refuse old-family
radii. No existing scientific declaration was signed or silently amended.

### Acquired-record audit

The first design was written before the candidate's source values were read. It is explicitly
exploratory and informed by slice 1. The saved climatology is fitted on the existing training
period; only frames 0:1440 are opened, not the 2022-2023 forecast-test period. The first window
sets the radius and the next two apply it unchanged. Track-key labels are proxies for continuity,
not independent ground truth for an unchanged spatial configuration or a recurring physical kind.
All-pair error rates below refer to these proxies. Distinct-key negatives can share a track.

Captured terminal output:

```
$ .venv/Scripts/python.exe -m tools.audit_spatial_identity --output data/identity_calibration/t4e8-spatial-audit.json
Design sha256: 10509a4e3e11aa9f79f6e03e88730ccd5cfe1945c04929bd925d9c5a0eb134ed
Window 0:480: 69580 configurations, 6838 repeat pairs
Window 480:960: 51395 configurations, 3652 repeat pairs
Window 960:1440: 52765 configurations, 5607 repeat pairs
Receipt: data/identity_calibration/t4e8-spatial-audit.json; DISCRIMINATION_CRITERIA_NOT_MET; 84.08 s
```

A descriptive amendment adds the monotone empirical-feasibility diagnostic after inspecting
those results. It changes no window, threshold, weight or acceptance. The final code also binds
the comparable family to the feature source as well as the declared grid scope; reusing a scope
string cannot cross a dataset boundary. All non-test Python sources and the audit script are
hashed before execution and checked unchanged before publishing the final receipt.

```
$ .venv/Scripts/python.exe -m tools.audit_spatial_identity --design data/identity_calibration/t4e8-spatial-design-v2.json --output data/identity_calibration/t4e8-spatial-audit-v2.json
Design sha256: fb1b64fbf4cf41db46768a4a100ebf74df6746add60141f590727d0f0ab54448
Window 0:480: 69580 configurations, 6838 repeat pairs
Window 480:960: 51395 configurations, 3652 repeat pairs
Window 960:1440: 52765 configurations, 5607 repeat pairs
Receipt: data/identity_calibration/t4e8-spatial-audit-v2.json; DISCRIMINATION_CRITERIA_NOT_MET; 144.52 s
```

That run was taken from an uncommitted tree, so its receipt records `code_dirty: true` and
cannot be cited against a revision. The slice was committed as `b902b87` and the identical
design re-run against the clean tree, which is the citable receipt:

```
$ .venv/Scripts/python.exe -m tools.audit_spatial_identity --design data/identity_calibration/t4e8-spatial-design-v2.json --output data/identity_calibration/t4e8-spatial-audit-v2-clean.json
Design sha256: fb1b64fbf4cf41db46768a4a100ebf74df6746add60141f590727d0f0ab54448
Window 0:480: 69580 configurations, 6838 repeat pairs
Window 480:960: 51395 configurations, 3652 repeat pairs
Window 960:1440: 52765 configurations, 5607 repeat pairs
Receipt: data/identity_calibration/t4e8-spatial-audit-v2-clean.json; DISCRIMINATION_CRITERIA_NOT_MET; 80.99 s
```

`code_dirty` is `false` and `code_revision` is `b902b87dc5318de0a3750162042c162efccc37ec`.
`design_sha256`, `source_declaration_sha256` and `metric_digest` are unchanged. Comparing the
two receipts field by field, the only differences anywhere are `code_dirty`, `code_revision`
and the four `elapsed_seconds` timings; every window's census, candidate errors, AUCs, strata
and feasibility diagnostic are bit-for-bit identical. The verdict is unchanged and
`approved_mining_radius` remains `null`. The faster wall clock is contention, not a
measurement difference. The figures tabulated below were extracted from the first receipt and
hold unchanged for the clean one.

**T4E.8 slice 3 (2026-09-08): a declared target, and the reproduction that proves the
declaration changed no measurement.**

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_target_declaration.py -q
21 passed   # 18 test functions, 21 parametrised cases
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_target_declaration.py     src/tests/test_spectral_spatial_identity.py src/tests/test_spectral_invariance.py     src/tests/test_spectral_clustering.py src/tests/test_spectral_regions.py -q
190 passed
$ .venv/Scripts/python.exe -m tools.mutate_identity_target
baselines_passed: true; mutants: 12; killed: 12; survivors: []
```

The twelve mutations all weaken the declaration or its admissibility rule, which is logic a
passing run never exercises. `admissibility_never_refuses`, `kind_recurrence_admits_proxy_labels`,
`absent_target_falls_through`, `absent_evidence_falls_through`, `proxy_wording_reworded`,
`caveat_never_published`, `proxy_claims_independence`, `declared_boundary_ignored`,
`catalogue_family_unchecked`, `unmatched_counted_as_an_identity`, `catalogue_type_unchecked` and
`negative_budget_unbounded` were each killed by a failing assertion, with no timeout or
collection failure counted as a kill. `negative_budget_unbounded` survived a first run because
no test covered the budget; rather than delete an untested guard, the budget was made an
argument and the refusal pinned, which is recorded here because the first figure was 11 of 12.

The refusals, all raised before any source value is opened:

```
$ .venv/Scripts/python.exe -m tools.audit_spatial_identity --design <no identity_target> ...
MissingParameterError: Missing required parameter 'identity_target' for 'an identity audit'.
Required parameters for 'an identity audit': kind_recurrence, spatial_persistence, track_continuity.

$ ... --design <identity_target: kind_recurrence, evidence_class: record_derived_proxy> ...
InvalidParameterError: Parameter 'evidence_class' = 'record_derived_proxy' is invalid: expected
evidence admissible for target 'kind_recurrence': external_reference. record_derived_proxy labels
are labels computed from the same record and pipeline whose identity is under test, so a
definition validated against them is validated against itself; choose admissible labels rather
than relaxing this pairing.

$ ... --design <identity_target: spatial_persistance> ...
UnknownNameError: Unknown identity target 'spatial_persistance'. Did you mean
'spatial_persistence'? Available: kind_recurrence, spatial_persistence, track_continuity.
```

The reproduction. `t4e8-spatial-design-v3.json` is the slice-2 design with `identity_target`
and `evidence_class` added and nothing else changed:

```
$ .venv/Scripts/python.exe -m tools.audit_spatial_identity --design data/identity_calibration/t4e8-spatial-design-v3.json --output data/identity_calibration/t4e8-spatial-audit-v3.json
Design sha256: e93b5e3246e07688ba5affe007c464a3b9dcd0b6e6e30e002bb618010f5a0464
Identity target: spatial_persistence via record_derived_proxy
Caveat: Repeated tracked keys bound how far geometry drifts over a track's lifetime; genuine morphological change is scored as a split, not as an error.
Window 0:480: 69580 configurations, 6838 repeat pairs
Window 480:960: 51395 configurations, 3652 repeat pairs
Window 960:1440: 52765 configurations, 5607 repeat pairs
Receipt: data/identity_calibration/t4e8-spatial-audit-v3.json; DISCRIMINATION_CRITERIA_NOT_MET; 140.61 s
```

`code_dirty` is `false` at revision `3fc491a`. Compared field by field against
`t4e8-spatial-audit-v2-clean.json`, every window's census, candidate errors, AUCs, strata and
feasibility diagnostic are **bit-for-bit identical**, at the same frozen radius
`0.13807521070069662` and the same `DISCRIMINATION_CRITERIA_NOT_MET` verdict with
`approved_mining_radius` `null`. Only `elapsed_seconds`, the new `identity_declaration` block,
and the expected provenance fields -- design, design hash, source hashes, revision -- differ.
Naming what was already being measured moved no measurement, which is the whole of what this
slice claims.

**Not delivered.** The specification's limb 3 asked for the declaration to be surfaced in the
interface. Nothing in `src/api` or `frontend/src` reads identity-calibration receipts, so there
is no view to extend and none was invented; the declaration is published in the receipt instead.
`kind_recurrence` is exercised and tested through the library's `catalogue_labels`, but is
unevaluable from the audit tool, which refuses every evidence class but `record_derived_proxy`
by name because no serialisation for a signed `PatternCatalogue` exists. No target is chosen and
no radius is approved.

**T4E.8 slice 4 (2026-09-09): the identity declaration rendered, refusals included.**

`PLAN.md` section 5 requires that the interface expose the scientific contract rather than
operate the backend, that a refusal rank equal to a value, and that evidence be **rendered**
rather than described. The last of those is why this entry carries a browser run.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_api.py -q
21 passed   # 16 test functions, 21 parametrised cases

$ .venv/Scripts/python.exe -m pytest src/tests/test_frontend_contract.py -q
185 passed

$ cd frontend && ./node_modules/.bin/tsc --noEmit -p tsconfig.json
(clean)

$ cd frontend && npm run build
built in 1m 22s

$ cd frontend && ./node_modules/.bin/playwright test e2e/identity-declaration.spec.ts
  ok 1 every declared target is drawn with what it does not license (4.7s)
  ok 2 the circular pairing is on screen as a refusal, with its reason (2.8s)
  ok 3 a refusal is drawn at the weight of an admission, not as an error (3.7s)
  ok 4 an admitted pairing still shows the caveat it owes (3.1s)
  ok 5 receipts show their claim boundary and that no radius is approved (3.1s)
  ok 6 a receipt written before the declaration existed is labelled, not hidden (2.6s)
  ok 7 a receipt opens whole rather than in fragments (3.5s)
  ok 8 the panel offers no control that chooses a target (2.9s)
  8 passed (36.3s)
```

Test 3 is the one the section was written for. "Equal weight" is a claim about a picture, so it
is measured as one: the refused and admitted cells' bounding boxes are compared and required to
agree within four pixels. Test 8 is its complement -- the panel must contain no `select` and no
`form`, because the server serves no route that would accept a choice, and the refusals are
rendered from the server's own list rather than implied by an absence of buttons.

Two contract guards failed while wiring this and both were right.
`test_no_served_route_is_unreachable_from_the_ui` refused `/api/v1/identity/audits/*` until the
panel actually consumed it, which is why a receipt can be opened whole rather than only
summarised. `test_the_qualification_gate_inventories_every_served_workspace` refused the new
workspace until it was added to the qualification inventory in `e2e/ui-qualification.spec.ts`.

This closes gap 1 of `PLAN.md` section 5. Gaps 2 and 3 remain open, and **no WCAG level is
claimed**: this run is a rendered functional inspection, not an assistive-technology audit.

**The cost of this slice, recorded rather than absorbed.** The full browser suite was run after
the change: **140 passed, 4 failed, 10.3 h**. The failures are not a broken panel. Recorded
evidence in this programme is bound to the source it was measured against, and this slice added
`e2e/identity-declaration.spec.ts`, edited `e2e/ui-qualification.spec.ts` and edited
`frontend/src/App.tsx`. The release plan therefore reports:

```
Clean-browser no-glue path: NOT_RUN
  The source of identity-declaration.spec.ts, ui-qualification.spec.ts has changed since the
  run was recorded.
Synthetic fifth-adapter no-edit test: NOT_RUN
  The source of frontend/src/App.tsx has changed since the run was recorded.
```

`flagship-qualification.spec.ts` then fails because it asserts the gate text, which now reads
NOT_RUN where it read PASS. Confirmed in isolation: 1 failed, 1 passed in 1.1 m. This is the
binding working, and is the same behaviour TG17.14 demonstrated when editing an acquisition
module returned `live_sources` to NOT_RUN and invalidated a passing record.

**Restoring them requires a fresh full-suite run and its recording; it has not been done.** The
G17 verdict was already `NOT_RELEASEABLE` on `scale_shape_calibration`, so this changes no
release decision, but it does mean two gates that read PASS now read NOT_RUN and the tables in
`architecture.md` and `roadmap_cross_domain.md` say so. The remaining 3 of the 4 suite failures
were not isolated to a cause; the run predates no clean baseline for this suite size, so they
are reported as measured rather than attributed.

**T4E.24 (2026-09-11): adopted, measured, and the tolerance turns out not to be the constraint.**

The declaration was adopted by the maintainer on 2026-09-11 and recorded in
`data/identity_calibration/t4e24-false-absence-adoption.json`, binding the declaration by
content hash `ae0e1792...` rather than by revision, because it was untracked when adopted. The
declaration itself is unchanged -- adopted declarations in this store are never mutated, and
editing one would break the binding that makes the adoption checkable.

**The generating code is committed.** T4E.20 and T4E.21 wrote their receipts from scripts that
were never committed, so neither can be re-run from this repository.
`src/benchmarks/false_absence.py` and `tools/measure_false_absence.py` close that gap for this
slice.

```
$ .venv/Scripts/python.exe -m tools.measure_false_absence
      --output measurements/t4e24_false_absence.json

gate PASSED   median 4 features/frame (declared band [4,10]), max 8, over 360 scenes
60 configurations x 6 scenes | 414 features | 2,484 trials | 1,493 s

marginal false absence rate      0.3724     (recovery 0.6276)
extracted matching no planting   15 over 360 scenes
median offset over recovered     0.406 cells

presence counts (features seen in 0..6 of 6 scenes)
   0 -> 121     1 -> 12     2 -> 20     3 -> 8     4 -> 10     5 -> 15     6 -> 228

admission at k = S - a, counts over the observed features
   a=0  k=6   228/414 = 0.5507
   a=1  k=5   243/414 = 0.5870      ABSENCES_TOLERATED = 1, as T4E.13 fixed it
   a=2  k=4   253/414 = 0.6111
   a=3  k=3   261/414 = 0.6304

dispersion   observed variance 7.352   binomial 1.402   bootstrap band [1.241, 1.584]
             OVER-DISPERSED
```

**The consequence claim, which is the half this slice may be cited for.** A threefold relaxation
of the tolerance -- from "present in every scene" to "present in three of six" -- moves admission
by **eight percentage points**. It cannot do more, because **121 of 414 features are recovered in
no scene at all**. The distribution leaves a tolerance almost nothing to act on: 228 features at
6 of 6, 121 at 0 of 6, and only **65 of 414 (15.7%)** in between. Coverage is the binding
constraint, not `a`.

The declaration named this arm in advance and named it as the worse one: *"a subpopulation of
features is invisible in EVERY scene and no tolerance recovers them. That is not a criterion
problem, it is a coverage problem, and it is worse, because it is silent. A criterion cannot fail
on evidence that never reaches it."*

**The mechanism claim, which is the weak half and was declared weak before measuring.** The
over-dispersion is large and the predicted concentration is confirmed. It was also close to built
in: the design holds scale, amplitude and neighbours fixed across the six scenes, which is
exactly what produces concentration. That caveat was written into the declaration before the
numbers existed and is not softened now.

**The declaration's stated mechanism is corrected.** It attributed concentration to *"suppression
[depending] on a feature's scale and amplitude relative to its neighbours."* The breakdown says
the driver is amplitude against the detection cut, very nearly alone:

```
recovery by peak-to-background ratio   8-14: 0.144   14-20: 0.595   20-26: 0.873   26-32: 0.919
recovery by scale (cells)             <2.5: 0.675    2.5-4: 0.689    4-6: 0.560     6-12: 0.532
recovery by nearest neighbour (cells) 8-15: 0.535   15-25: 0.649   25-40: 0.625     40+: 0.677

never seen (0/6), n=121   median ratio 12.3   median sigma 4.18   median neighbour 27.3 cells
always seen (6/6), n=228  median ratio 24.8   median sigma 3.38   median neighbour 30.5 cells
```

Amplitude moves recovery by 0.78 across its range; neighbour separation moves it by 0.14 across
a fivefold range. The features that vanish are the **faint** ones, not the crowded ones. This
sharpens the built-in caveat rather than softening it: amplitude is precisely one of the
quantities held constant across the six scenes, so a feature under the cut is under it in all six
by construction.

**A choice the declaration did not fix, recorded as a choice.** Where the detection cut comes
from. Both options were measured on the same probe before either was adopted:

```
calibrated on   gate median   recovery   extracted matching no planting (48 scenes)
background          5          0.940                    63
scene               4          0.774                     0
```

Calibrating on the bare background is the cleaner null -- the plantings do not enter the
surrogates that set their own threshold -- but it admits more than one noise peak per scene while
claiming family-wise control, and it does not reproduce T4E.21. Calibrating on the **scene** is
what the pipeline does on the real record, reproduces T4E.21's loss, and is the less flattering
of the two. It was chosen for the first two reasons; that it is also the harsher one is stated so
the choice cannot later read as a convenience.

**Disclosed: the gate passed at the bottom of its band again.** Median 4 against the record's 7,
the same disclosure T4E.21 carried. It forbids claiming the rate has been measured at the
record's own density. It bites less here than there, because competition turned out not to be
the mechanism -- but that is an argument, and it is labelled as one.

**Which way the bound runs.** Identical geometry is the most favourable case for recovery that
exists; real recurrence carries jitter, drift and evolution, each of which can only reduce it. So
0.6276 is an **upper** bound on recall and 0.3724 a **lower** bound on false absence. The real
figure is worse.

**Acceptance.** All four conditions met: the coded gate passed (1); the full presence
distribution and the admission counts at `a = 0, 1, 2, 3` are reported rather than a mean (2);
the mechanism and consequence claims are reported separately, the mechanism claim carrying its
declared caveat (3); and the two named outcomes did separate, so condition 4's permission to
return nothing was not needed.

**What is not licensed.** No tolerance is chosen and no proportion of `S` is proposed -- R20
forbids the horse race, and choosing a tolerance against this number is a separate declaration
owing its own blindness claim. Nothing in the extractor changed. T4E.13 is not superseded. No
reserved seed block (720-735, 880-895) and no frame of the 2022-2023 forecast-test period was
read. D96, D97, D98, D99 and D100 remain open and T4E.8's acquired-record acceptance is
untouched. The rate belongs to 850 hPa relative vorticity over this crop under this planting and
is quoted for no other domain, though the obligation to measure it transfers to every domain that
adopts the criterion.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_false_absence.py -q
27 passed

$ .venv/Scripts/python.exe -m pytest src/tests/test_false_absence.py
      src/tests/test_feature_extraction.py src/tests/test_documentation.py -q
1 failed, 120 passed in 344.45s
```

The one failure was `test_documented_test_counts_match_the_source`: the inventory total still
read 4315 against an actual 4342. It was corrected to 4342 rather than the test weakened.
**`tools/audit_docs.py` reported `RESULT: ok` on the same tree**, because it checks per-file
inventory rows and undocumented modules but never the table's own total. The audit is the weaker
of the two checks and the test caught what it missed.

**TG19.3 (2026-09-11): a signed reference resolves by digest, or refuses by name.**

An engineering slice. It makes no claim about any world, so it carries no declaration, no
adoption and no prediction. `src/data_layer/signed_reference.py`, wired into
`tools/restate_position_acceptance.py`.

**The gap, and it was not hypothetical.** T4E.17 signed the IBTrACS catalogue on terms naming a
specific population, bound it by sha256, and deliberately did not commit 35.5 MB of third-party
data. What the design records is the URL, the digest and the byte count. What it records **no**
path. So nothing in this repository knew where to look, nothing failed loudly when the file was
missing, and the file spent a day in a session scratchpad under `%TEMP%` where it survived by
luck. Losing it would have **voided the signature**, not merely cost a download: IBTrACS v04r01 is
a living archive, so a fresh copy is a different catalogue on which T4E.17's terms do not hold.

That is the lesson T4E.27 drew about a catalogue radius of `0.00`, applied to a file instead of a
field: **a missing input should be refused by name, not discovered later.**

**Three bindings, checked in order, because a chain is only as strong as the link nobody checks.**

1. **Signature against design.** `signs_sha256` in the signature file, against the design's own
   digest. If the design was edited after it was signed, the terms recorded are not the terms
   signed and everything downstream inherits the drift. Nothing had ever checked this.
2. **Design against data.** The digest the file must reproduce.
3. **Byte count first**, as a cheap pre-check, so a truncated download is named before 35 MB are
   hashed to reach the same conclusion.

```
$ python -c "from src.data_layer.signed_reference import resolve; ..."
name                 ibtracs_sp_v04r01
path                 data/catalogues/ibtracs.SP.list.v04r01.csv
available            True
signature_verified   True
bytes                35,482,417 expected, 35,482,417 observed
digest               matches the signed 631f76b9...
citations            2, carried with the resolution
```

**A refusal is a result, not an exception by default.** `resolve()` returns a record saying what
failed and what would lift it; `require()` raises for a caller that cannot proceed. Both carry the
source URL and the expected digest, so a reader who has lost the file learns where to get it and
what it must hash to in the same breath as learning it is gone.

**A digest mismatch is reported as a *different* reference, never a damaged one.** The refusal
says so in terms: the signed population, base rates and claim boundary do not extend to it, and
using it would evaluate against an unsigned catalogue. There is no fallback path to an unverified
copy, and nothing here reaches a network -- acquiring the file is the maintainer's act.

**A refusal that could not tell "missing" from "present" was itself the defect.** The first
version of `restate_position_acceptance.py` stated in prose that the catalogue was absent. It had
no way to check, and it was wrong: the file existed. That clause is now resolved rather than
asserted, and the tool's receipt carries the resolution beside the refusal. **The committed
T4E.27 receipt still says the file was not present on this machine. It was accurate when it was
written and is superseded rather than edited**, which is how this programme records corrections.

What the refusal now says is the part that was always true and is the only part that still is:
condition 2 needs the distance to the third-nearest feature, the committed receipt carries only
the nearest, and recovering it takes a re-run of the join. That re-run is its own work and is not
done here -- but it is no longer blocked on a missing file.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_signed_reference.py -q
16 passed
```

The tests pin each way silence could return: an absent file naming path, URL and digest; a
truncated one named by size without hashing; a different edition refused as different; a design
edited after signature; a signature carrying no digest; a design carrying no digest; an unsigned
design allowed and reported as unverified rather than failed; and `require()` raising by name
instead of returning an unverified path.

**What this does not do.** It downloads nothing, adopts nothing, and changes no default. It does
not lift T4E.27's condition 2, which needs the join re-run. It does not edit the signed design to
record the path -- that would break the sha256 the signature rests on, so the path lives in the
registry beside it. And it verifies bindings, not contents: that the file is the one that was
signed says nothing about whether the terms signed were the right ones.

**TG19.2 (2026-09-11): the join's bar, on the wire and on screen, in parts.**

An engineering slice. It makes no claim about any world, so it carries no declaration, no
adoption and no prediction. `src/api/identity.py` (two routes),
`frontend/src/components/PositionToleranceView.tsx`, `frontend/e2e/position-tolerance.spec.ts`.

**Why it exists.** T4E.27 built a tolerance that publishes its parts, and it was reachable only by
importing a Python module. A researcher could read what bar this programme used and could not see
what theirs would be. T4E.27 also found two things a researcher needs and neither was visible:
the bar was wrong for three reasons, *and* replacing it changed nothing. A bar that arrives as a
single number can only be accepted or rejected; one whose components, provenance, exclusions and
refusals are on screen can be disagreed with specifically.

**Two routes, both GET, both computing rather than deciding.** `/identity/tolerance/components`
serves the contract before any observation is supplied -- what the parts are, who supplies each,
how they combine, what is excluded, and why a missing uncertainty is refused.
`/identity/tolerance` computes one bar, and with an optional separation also reports whether it
is admitted and what the justified components fail to explain. No route stores a tolerance,
approves a join, or records an acceptance, and none was added.

**`null` is a third answer and the wire carries it as one.** A refused tolerance returns
`admitted: null`, never `false`, because *we could not say* and *no* are different answers and a
client that conflated them would count a missing catalogue uncertainty as a failed detection --
the exact error T4E.27 exists to correct.

```
$ curl '/api/v1/identity/tolerance?catalogue_radius_km=11.12&separation_km=99.98&observation=OWEN'
total_km 13.81   admitted false   unexplained_residual_km 86.17

$ curl '/api/v1/identity/tolerance?catalogue_radius_km=0.0&separation_km=74.91&observation=LINDA'
total_km null    admitted null    unexplained_residual_km null
```

**Rendered evidence, because PLAN section 5's acceptance is a claim about what a researcher can
see.** Six Chromium tests against the real API and the real frontend:

```
$ cd frontend && ./node_modules/.bin/playwright test e2e/position-tolerance.spec.ts
  ok 1 the bar arrives in parts, each with where it came from (17.2s)
  ok 2 what the bar leaves out is on screen at the weight of what it includes (2.7s)
  ok 3 a missing catalogue uncertainty is refused by name, not scored as a miss (3.0s)
  ok 4 a refusal renders as a result, not as an error state (2.8s)
  ok 5 a computed bar shows its total and the residual it does not explain (2.9s)
  ok 6 the panel states what it will not do, from the server own list (2.7s)
  6 passed (57.3s)
```

Test 3 is the one the panel was written for: LINDA's reported radius in the acquired record is
exactly 0.00, the case that made one storm of eighteen unpassable however good the extraction
was. On screen it renders as a refusal naming its reason, the verdict element is absent
entirely rather than showing a miss, and test 4 requires the surrounding result section to still
render with no error banner -- a bar that could not be built is an answer the instrument is
entitled to give. Test 6 asserts no control matches `accept|approve|save|record|apply`, because
the panel's only control is arithmetic.

**A pre-existing defect this slice found, and it is not this slice's.** Adding a workspace made
`test_the_qualification_gate_inventories_every_served_workspace` fail -- and the drift it
reported was **`Study trail`**, not the new panel. T4E.23 added that workspace on 2026-09-10 and
never added it to `e2e/ui-qualification.spec.ts`; the spec's last commit is T4E.8 slice 4's. So a
served workspace has been outside the qualification gate since then, and the guard that catches
exactly this was not run when it shipped. Both entries are now listed, in shell order. **A
workspace outside the qualification inventory is an unqualified surface that reads as a qualified
one**, which is the failure mode the guard exists for.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_frontend_contract.py src/tests/test_identity_api.py -q
214 passed

$ cd frontend && ./node_modules/.bin/tsc --noEmit -p tsconfig.json
(clean)
```

**What this costs, recorded rather than absorbed.** This slice edits `frontend/src/App.tsx` and
`e2e/ui-qualification.spec.ts`, and recorded evidence in this programme is bound to the source it
was measured against. The G17 release plan will therefore return `browser_no_glue` and
`synthetic_fifth_adapter` to `NOT_RUN`, as it did for T4E.8 slice 4 and as TG17.14 demonstrated
when editing an acquisition module invalidated a passing record. The G17 verdict was already
`NOT_RELEASEABLE` on `scale_shape_calibration`, so no release decision changes. Restoring those
two gates needs a fresh full browser run and its recording, and that has not been done.

**What is not claimed.** No WCAG level: this is a rendered functional inspection, not an
assistive-technology audit. Nothing is adopted, no default changed, and no tolerance approved for
any pipeline. The panel computes a bar for whatever numbers a reader types; it says nothing about
whether that bar is right for their catalogue, and the excluded component is on screen precisely
so they can see what it does not cover.

**TG19.1 (2026-09-11): the coverage question, made runnable by someone else.**

An engineering slice, not a scientific one: it makes no claim about any world, so it carries no
declaration, no adoption and no prediction. `src/benchmarks/coverage_report.py`,
`tools/coverage_report.py`, `survival_by_cardinality` in `src/benchmarks/false_absence.py`.

**Why it exists.** T4E.24 through T4E.27 measured what this extractor loses -- 37% of features
present never extracted, 85% of triples never surviving six scenes, a cut that cannot be relaxed
to fix it. Those numbers became *readable* through T4E.22's measurement API and T4E.23's study
trail. They were not *runnable*. A researcher arriving with their own field and their own
extractor could read what this programme measured about its own instrument and could not ask the
same question of theirs. That is the difference between a lab notebook and an instrument.

**What was generalised, and what stayed put.** The T4E.24 measurement was bound to ERA5 shards,
a vorticity sign convention and one extractor. The mechanism was never atmospheric: plant known
structure, extract, count what came back. `coverage_report` now takes a `background_for(i)`
callable, any **registered** extractor by name, and the caller's own declaration of what the
field is -- domain, dataset, variable, units, which R19 refuses a comparison without. The
atmospheric run is unchanged and its receipts are untouched.

**`survival_by_cardinality` turns a derivation into a capability.** T4E.24's second addendum
computed pair and triple survival by hand in a one-off script, because
`ALLOWED_CARDINALITIES` is `(2, 3)` and whole-configuration survival measures something nothing
downstream consumes. That arithmetic is now a function, tested, and reported by every coverage
run. Checked against the committed receipt, it reproduces the addendum exactly: pairs 394/1362
intact and 649/1362 assemblable, triples 418/2805 and 856/2805.

```
$ python -m tools.coverage_report --list-extractors
local_maximum    Local maxima above a surrogate-calibrated frame-maximum threshold, ...

$ python -m tools.coverage_report --output r.json --source fbm --configurations 3 --surrogates 199
VERDICT: EVIDENCE_NOT_REPRESENTATIVE
  median 3 features per frame is outside the declared band [4, 10] taken from the record
```

**The first run refused, and that is the instrument working.** The default band is the one the
atmospheric work declared, and it is wrong for a fractional-Brownian field -- which the tool's own
help says before it is run: *"the default is the one the atmospheric work declared and is
probably wrong for you"*. A caller who states a band their field warrants gets a measurement; a
caller who does not gets a refusal naming the reason. What is deliberately absent is any path
where the background is adjusted until the number improves.

```
$ python -m tools.coverage_report --output r.json --source fbm --configurations 3 \
      --surrogates 199 --median-band 2 10
VERDICT MEASURED   gate median 3   pairs intact 0.181   triples intact 0.079

$ python -m tools.coverage_report --output r.json --source netcdf \
      --path data/cds_downloads/t4e18_vorticity --variable vo --negate \
      --domain atmosphere --configurations 10 --surrogates 199
VERDICT MEASURED   gate median 4   pairs intact 0.257   triples intact 0.107   52.6 s
```

The second runs with **no atmospheric data at all** -- a synthetic field with a declared Hurst
exponent, for a caller who has no record of their own. The third runs the same question over the
acquired record through the generic path and lands near T4E.24's triple figure of 0.149, at a
tenth the configurations and a fifth the surrogates, which is the agreement a smaller sample
should give and not an independent confirmation of it.

**What every report carries.** The extractor's registered capabilities, including its declared
`shape_model`, because a coverage number means something different for an instrument that assumes
isotropy. Which field the cut was taken from, with T4E.25's measured 3.53x attached, so a caller
choosing `background` sees what the choice is worth. That identical geometry across scenes makes
recall an **upper** bound. And a claim boundary saying the report is a property of an instrument
and never of a world -- it plants what it then looks for, so it can say nothing about whether
such structure exists in any real field.

**What this does not do.** It adopts nothing, changes no default, and alters no extractor. It
does not make the atmospheric numbers transferable: a coverage figure belongs to the field and
the planting it was measured on, and the module refuses to imply otherwise. And it is not yet on
the wire or on screen -- reports land in the measurement store and are served by T4E.22's
existing routes, which is reach, not a user interface.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_coverage_report.py -q
17 passed
```

**T4E.27 (2026-09-11): the bar was wrong, and the bar was not what was carrying the failure.**

Adopted 2026-09-11 (`t4e27-position-tolerance-adoption.json`, binding the declaration by sha256
`f4ef59f6...`) and measured the same day. `measurements/t4e27_restated_acceptance.json`,
`src/analysis_engine/position_tolerance.py`, `tools/restate_position_acceptance.py`.

T4E.21 named this work and deferred it in writing. T4E.18's acceptance had failed at 2 of 18 and
1 of 18 against a bar of 9, and had stood as a failure ever since against a tolerance nobody had
argued for.

**Three reasons the original bar was wrong, none of which is that it failed.** A **category
error**: the bar was the catalogue's per-observation radius, an agency's bound on its own
uncertainty about its own quantity, never a bound on the separation between two different
quantities. An **unsatisfiable region**: the radii run 0.00 to 145.23 km and LINDA's is exactly
0.00, so one of eighteen storms could not have passed however good the extraction was -- a
missing agency report written down as a number, in a programme whose named-refusal rule had been
applied to its outputs and not to this input. And an **empirical miss**: T4E.19 measured the
correlation between offset and radius at +0.020, range straddling zero.

**The restated bar is computed, not chosen** -- the quadrature sum of the agency's reported
radius and the extractor's own localisation error, 0.295 cells or 8.19 km, as T4E.21 measured it
on known ground truth for a different purpose. The vorticity-versus-surface-centre separation is
deliberately **not** a component, so the residual measures it.

```
storm        lat   nearest   radius  tolerance  residual  admitted
FEHI       -36.8     69.74     8.90      12.10      57.6   no
GITA       -38.6    113.72    89.38      89.76      24.0   no
HOLA       -31.5    252.29    21.99      23.46     228.8   no
LINDA      -23.8     74.91     0.00    REFUSED         -   could not say
IRIS       -22.7    152.69    15.13      17.20     135.5   no
JOSIE      -21.5    189.65   145.23     145.46      44.2   no
OWEN       -20.8     99.98    11.12      13.81      86.2   no
PENNY      -20.0     51.20    11.12      13.81      37.4   no
OMA        -28.2    162.24    14.82      16.93     145.3   no
SARAI      -20.3   2055.91    24.57      25.89    2030.0   no
UESI       -37.5     60.81   103.47     103.80     -43.0   YES
GRETEL     -31.2    127.10   106.85     107.17      19.9   no
ANA        -22.3   1581.20    34.91      35.86    1545.3   no
LUCAS      -22.7     29.93    61.55      62.09     -32.2   YES
NIRAN      -28.4   1241.04   112.31     112.61    1128.4   no
UNNAMED    -30.1    159.45    22.24      23.70     135.7   no
RUBY       -30.7    107.00    11.12      13.81      93.2   no
SETH       -21.0    111.72    10.38      13.22      98.5   no

condition 1 restated: 2 admitted of 17 judged, 1 refused, needed 9 -- NOT MET
condition 2 restated: REFUSED BY NAME, not evaluable on the available evidence
median unexplained residual: 93.19 km
```

**The prediction was half right, and the half that was wrong was derivable.** The declaration
predicted the acceptance would still fail while *improving* on 2 of 18. It still fails. It did
not improve at all -- the same two storms pass, and no others. The reason is arithmetic that
should have been done in advance: adding 8.19 km in quadrature to radii of 11 to 145 km is very
nearly inert. It moves a bar of 11.12 to 13.81 and a bar of 89.38 to 89.76, against separations
of 50 to 250 km. **The estimator component could never have changed a verdict here, and saying so
required no measurement.** That is the fifth time in this programme a derivable fact has been
left underived, and it is recorded rather than smoothed over.

**So the finding is sharper than either outcome the declaration anticipated.** The bar was wrong
for three good reasons *and* replacing it with a defensible one changes nothing. What was
carrying the failure is not the tolerance. It is the separations themselves: a median nearest
feature of 127 km, with three storms -- SARAI at 2056 km, ANA at 1581, NIRAN at 1241 -- that no
tolerance worth the name will ever admit.

**The two storms that do pass, pass for the wrong reason.** UESI and LUCAS are admitted because
their *catalogue radii* are 103 and 62 km -- the agency was highly uncertain about where those
centres were. A join that closes because the reference is vague is not evidence that the
instrument found the cyclone, and counting it as one would be the same category error in the
opposite direction.

**The residual localises what remains, and it does not fit T4E.21's hypothesis.** The median
unexplained separation is **93.2 km**. T4E.21's leading candidate -- that an 850 hPa vorticity
maximum and a surface centre are different quantities, worth about 34 km -- cannot account for
that. So on this population the quantity difference is **not** the dominant term, and the honest
conclusion is that T4E.18's acceptance population and T4E.19's diagnostic population are not the
same problem. T4E.19 worked on 154 interior observations paired within 200 km with a dateline
group separated out, and got a median of 33.8 km. T4E.18's acceptance takes the deepest
observation per storm with no such filtering, and gets 127 km. Restating the bar exposed that
rather than repairing it.

**The acceptance curve, reported for inspection and not for selection.**

```
tolerance km     5    10    15    20    25    30    35    40    50    65
admitted /17     0     0     0     0     0     1     1     1     1     3

tolerance km    80   100   125   150   200   300   500
admitted /17     4     5     8     9    13    14    14
```

Nine of seventeen -- the declared majority -- first arrives at **150 km**, roughly an order of
magnitude beyond anything the two justified components support, and the curve then saturates at
14 of 17 no matter how far it is pushed. The declared bar decides the verdict; this is here so
the bar can be argued with specifically rather than merely accepted, and choosing a point from it
now would be the horse race R20 forbids.

**Condition 2 is refused by name, and this slice therefore meets its own acceptance only in
part.** Condition 2 asks for three or more features inside tolerance. The committed receipt
carries each storm's nearest distance and a count of features inside the *original* radius, but
not the distance to the third-nearest, so the count at any other bar is not recoverable from it.
Recomputing it needs the IBTrACS CSV, which T4E.17 bound by sha256, did not commit, and which is
not present on this machine. Approximating it was available and was refused. **The shortfall is
reported rather than absorbed:** this task's own acceptance condition 3 asked for both
conditions, and one of them was not delivered.

**The instrument, which was the point of the task.** `PositionTolerance` publishes each
component with its provenance, names what it deliberately excludes, and refuses an unusable input
by name -- returning `None` from `admits`, never `False`, because *we could not say* and *no* are
different answers and conflating them counts a missing agency report as a failed detection. A
refused observation leaves the denominator rather than scoring against the instrument. A bar
built this way can be disagreed with in parts, which is what lets a question this programme did
not think of be asked with the same instrument.

**What is not licensed.** No mining radius, no tolerance adopted into any pipeline, no part of
T4E.8's acquired-record acceptance discharged, and nothing settled about whether the record
supports `kind_recurrence`. `measurements/t4e18_acceptance.json` stands as measured with its own
correction beside it; this restatement is recorded alongside it, never in place of it. The
population, conditions, extractor, null and alpha are all unchanged, and nothing was adopted from
T4E.26. D96 through D100 remain open. No reserved seed block and no frame of the 2022-2023
forecast-test period was read.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_position_tolerance.py -q
17 passed
```

One test failed while writing this and was itself the error: it asserted that an infinite reported
radius should be accepted as a value rather than refused. An infinite bar admits every
separation, which is worse than having no bar at all, so the module was right and the test was
wrong. The test was corrected and the module's refusal wording was tightened to say why a
non-finite radius is refused as well as a zero one.

**T4E.26 (2026-09-11): peeling the null works, manufactures most of its own contamination, and
changes what a detection claims.**

Adopted 2026-09-11 (`t4e26-peeled-null-adoption.json`, binding the declaration by sha256
`3e374738...`) and measured the same day. One round, `alpha = 0.05`, T4E.24's 360 scenes under the
same root seed. `measurements/t4e26_peeled_null.json`. Three ensembles per scene -- the scene's
own, the residual's, and the bare background's -- 71 minutes.

**Acceptance condition 2 first.** Round zero reproduces T4E.24 and T4E.25 exactly: feature
recovery 0.6276, the presence distribution identical key by key (121/12/20/8/10/15/228),
configuration coverage 0.1667 and 0.0833, spurious 0.0417 per scene. Every feature had a usable
width, so nothing was left unsubtracted for want of one.

```
thresholds, in units of the background's own robust width (median over 360 scenes)
   round zero 15.17      peeled 8.64      oracle 4.37

fraction of the oracle gap closed, by decile over 360 scenes
   0.000  0.396  0.488  0.532  0.587  0.637  0.701  0.736  0.798  0.896  1.041

stage        gate med   feature recovery   recoverable    intact      spurious/scene
round zero       4          0.6276         0.1667(10/60)  0.0833(5/60)    0.042
peeled           6          0.8237         0.6000(36/60)  0.3000(18/60)   0.275

features never recovered in any scene:  121 -> 39
newly recovered trials 487, trials lost 0
median peak-to-background ratio: newly recovered 12.51, already found at round zero 24.02
```

**The prediction held on all three limbs, against bars fixed by T4E.25 before the probe that
informed it existed.** Intact-configuration coverage rose **+0.2167** against a 0.15 bar.
Contamination stayed at 0.275 per scene against a bar of 1.0. And the newly recovered plantings
are **half the brightness** of those round zero already had -- 12.51 against 24.02 -- so peeling
reached the faint population rather than re-finding the bright one. Nothing was lost: no trial
recovered at round zero went missing after peeling.

**The declared failure mode is real, and it is most of the contamination.** Of 99 spurious
features, **83 (83.8%) fall within two fitted widths of something that was peeled** -- they are
lobes the subtraction created and the lowered cut then reported. And they concentrate exactly
where the declaration said they would, on the asymmetric features an isotropic fit cannot
represent:

```
stretch of the planting nearest each manufactured feature
   1.0 -> 18      1.5 -> 11      2.5 -> 54
```

So the procedure's cost is not a diffuse rise in background noise. It is a specific, predictable
artefact at stretched features, arising from `local_maximum_extractor`'s declared
`isotropic_gaussian_on_a_flat_baseline` shape model meeting features that are not isotropic. Only
16 of 99 spurious features are ordinary contamination.

**THE REPORTED SIGNIFICANCE MEANS SOMETHING ELSE, and this belongs here rather than in a
footnote.** `NullCalibration` states its hypothesis as no peak exceeding the strongest peak of a
field with **the same power spectrum** as the frame. A peeled ensemble has the residual's
spectrum. So every p-value taken through a peeled cut answers a different question, and the
coverage gained above was gained **partly by changing what a detection claims**. Whether the
residual's spectrum is the better null for the question actually being asked -- whether *this*
peak is distinguishable from the background it sits on, rather than from a field that includes
itself -- is an argument, is labelled as one, and is not settled here. A successor that adopts a
peeled null must state the new hypothesis explicitly.

**What the numbers do and do not say about the coverage problem.** Peeling more than triples
intact-configuration coverage and cuts never-seen features from 121 to 39. It does not solve the
problem: **70% of configurations still hold at least one feature the extractor never recovers**,
and the gap's first decile is 0.000 -- in a tenth of scenes one round closes nothing at all. The
top decile exceeds 1.0, meaning some peeled cuts fall *below* the background oracle; that is
reported rather than clipped, because a peeled null is not bounded by the oracle and pretending
otherwise would hide an overshoot.

**Disclosed: the gate median moved from 4 to 6.** Both are inside the declared band, and 6 is
*closer* to the record's own 7 than round zero's 4 was. So the peeled stage is the better density
match to the record -- which is worth stating plainly, and is also the first time in this
sequence that a gate reading has moved toward the record rather than sitting at the bottom of its
band.

**The blindness claim and its limit, restated because it governs how this may be cited.** A
one-scene feasibility probe preceded the declaration and informed the prediction, which is
therefore **not blind** and is not reported as if it were. The two acceptance bars are T4E.25's
and predate the probe. These are T4E.24's scenes on their **fourth inspection**. Nothing here is
confirmatory: a pass is a failure to fail on the evidence that motivated the question, not a
validation.

**What is not licensed.** No peeled null is adopted, no default is changed, and nothing in
`local_maximum_extractor`, `NullCalibration` or the frame-maximum statistic is altered. No second
round was run and none is reported. No other null construction was evaluated (R20). The oracle is
a ceiling, not an achievable operating point. The 3.53x figure from T4E.25 and the artefact rate
here belong to 850 hPa relative vorticity over this crop under this planting and are quoted for
no other domain -- though a domain adopting a peeled null owes its own artefact measurement,
because that rate depends on how badly its extractor's declared shape model fits its features.
D96 through D100 remain open, T4E.8's acquired-record acceptance is untouched, and no reserved
seed block or frame of the 2022-2023 forecast-test period was read.

**The bound still runs the same way.** Identical geometry across the six scenes remains the most
favourable case, so every figure above is optimistic and the truth under jitter, drift and
evolution is worse.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_peeled_null.py -q
17 passed
```

One test failed while writing this and was itself the error: it asserted a peeled baseline of 5.0
against a flat field that had no peak planted in it, so the code was right and the test was
describing a field it had not built. The test was corrected rather than the module.

**T4E.25 (2026-09-11): the cut is not a lever, and the reason is not the one predicted.**

Adopted 2026-09-11 (`t4e25-coverage-contamination-adoption.json`, binding the declaration by
sha256 `ce3126fb...`) and measured the same day. 60 configurations at `S = 6`, T4E.24's scenes
replanted under the same root seed with the family-wise level `alpha` swept over the four
declared values. `measurements/t4e25_coverage_contamination.json`.

**Acceptance condition 2 first, because the declaration says nothing else may be interpreted
until it passes.** At `alpha = 0.05` the sweep reproduces T4E.24 **bit-for-bit**: marginal false
absence 0.3723832528 against 0.3723832528, 2,484 trials, 15 spurious, gate median 4, the presence
distribution identical key by key, the admission rates identical, and the addendum's
configuration figures at 10 and 5. The pairing is real, not asserted.

```
alpha  gate  feature   configurations      configurations   spurious   spurious   median ratio
       med   recovery  recoverable         intact           per scene  fraction   of recovered
0.05    4     0.6276    0.1667 (10/60)      0.0833 (5/60)     0.042      0.0095      24.0
0.10    5     0.6522    0.1833 (11/60)      0.1000 (6/60)     0.047      0.0104      23.7
0.25    5     0.6864    0.3000 (18/60)      0.1000 (6/60)     0.058      0.0122      23.2
0.50    5     0.7182    0.3500 (21/60)      0.1500 (9/60)     0.075      0.0149      22.8

step          d intact   d recoverable   d feature   spurious/scene   newly    median ratio
                                          recovery                    trials   of the new
0.05 -> 0.10   +0.0167     +0.0167        +0.0246    0.042 -> 0.047     61        16.9
0.10 -> 0.25   +0.0000     +0.1167        +0.0342    0.047 -> 0.058     85        14.0
0.25 -> 0.50   +0.0500     +0.0500        +0.0318    0.058 -> 0.075     79        13.1
```

No alpha was excluded by the gate. Recovery was monotone in alpha at every step, as the design
assumed; no trial was recovered at a lower alpha and lost at a higher one.

**The declared outcome held. The declared mechanism did not, and the difference matters.**

The prediction was: *"Contamination rises faster than configuration coverage at every step, and
no declared alpha brings intact-configuration coverage above 0.50 while keeping spurious features
below one per scene."*

The second clause **holds decisively**. Intact-configuration coverage reaches 0.15 at
`alpha = 0.50` -- a tenfold relaxation of the declared error level -- against a bar of 0.50. No
step met the declared falsification condition of a 0.15 absolute rise, so the verdict is
`PREDICTION_HELD`.

The first clause is **wrong**. Contamination barely moved: 0.042 to 0.075 spurious features per
scene across the whole sweep, never above 1.5% of everything extracted. Coverage rose more in
absolute terms than contamination did. The prediction was right about where the sweep ends and
wrong about what stops it, and reporting only the verdict would hide that.

**What actually stops it, measured rather than asserted.** A post-hoc diagnostic over 8 scenes,
adopting nothing and reporting no operating point, took the threshold in units of the
background's own robust width, from the scene's null and from the bare background's null:

```
alpha    planted-scene null    bare-background null
0.05          15.47 sigma            4.39 sigma
0.50          12.95 sigma            3.77 sigma

span 0.05 -> 0.50:  planted x1.195      background x1.165
the planted-scene null sits x3.53 above the bare-background null at alpha = 0.05
```

Two things follow, and only one of them was in the prediction.

**The tail is steep, as predicted.** A tenfold change in alpha moves the threshold by about 20%,
on the planted scene and on the bare background alike. That half of the predicted mechanism is
confirmed. But a threshold that moves 20% does not admit a flood of noise -- which is why
contamination stayed flat, and why the second half of the mechanism was wrong.

**The dominant term was not in the prediction at all.** The cut on a scene containing signal sits
**3.53x higher** than the cut on that scene's own background, because the plantings' power enters
every surrogate of the planted field. Against a level shift of 3.5x, a lever with 1.2x of travel
is not a lever. Alpha is not what sets this threshold; the signal is.

**This is the null behaving as declared, not a defect.** The hypothesis `NullCalibration` states
is *a structureless field with this power spectrum*, and the planted field's power spectrum
includes the plantings. Phase-randomising a field that contains coherent structure spreads that
structure's power across the frame and raises its own maxima. The result is conservative by
construction. It is the same effect recorded at T4E.24's calibration choice, where background
calibration gave recovery 0.940 against 0.774 -- now quantified as a threshold ratio rather than
inferred from a recall difference.

**What this licenses.** That coverage is not recoverable by relaxing the declared error level
within this scheme: a tenfold relaxation buys 9 percentage points of feature recovery and leaves
**65% of configurations still holding a permanently invisible feature**. T4E.24's coverage figure
is therefore a property of the detection scheme across a declared range, not an artefact of the
0.05 operating point -- which is exactly what this slice was declared to establish.

**What it does not license.** No operating point, no alpha, no change to any default (R20). No
change to the extractor, to `NullCalibration`, or to the frame-maximum statistic, and no
replacement for any of them. It does not establish that a scene-calibrated cut is the wrong
choice -- it is the choice the pipeline makes on the real record, and this slice measured its
price, not its correctness. The 3.53x figure is a diagnostic over 8 scenes with no error bar and
adjudicates nothing. Nothing here touches the atmosphere, T4E.8's acquired-record acceptance, or
D96 through D100. The reserved seed blocks and the 2022-2023 forecast-test period were not read.

**The bound still runs the same way.** These are T4E.24's scenes, so geometry is identical across
the six and the figures remain the favourable case. Real recurrence carries jitter, drift and
evolution; the true coverage is worse than every number above.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_coverage_contamination.py -q
12 passed
```

**T4E.24 second addendum (2026-09-11): the pipeline builds pairs and triples, so those are the
numbers that bind.**

Exact arithmetic on the committed T4E.24 receipt. No new measurement, no prediction, nothing
adjudicated -- and a correction to the first addendum's framing rather than to its figures.

The first addendum read the receipt per **configuration** and found that 50 of 60 hold a feature
recovered in no scene at all. That is true and its arithmetic stands, but it measures something
the pipeline never asks for. `ALLOWED_CARDINALITIES` in `spectral_constellation.py` is `(2, 3)`:
constellations are **pairs and triples**, and a fourth node is refused by name as the beginning
of frequent-subgraph mining. So the quantity a criterion actually consumes is not whether a
7-feature configuration survives whole. It is how many of its pairs and triples do.

```
                       intact in all 6 scenes      assemblable in at least one scene
pairs   (k = 2)          394 / 1362 = 0.2893            649 / 1362 = 0.4765
triples (k = 3)          418 / 2805 = 0.1490            856 / 2805 = 0.3052
```

A pair or triple is counted intact when every one of its members was recovered in all six scenes,
and assemblable when every member was recovered in at least one. Both are exact counts over
`C(n, k)` per configuration, summed.

**This is less severe than the configuration reading and still severe.** **85% of planted triples
never survive all six scenes, and 70% cannot be assembled in even one** -- on identical geometry,
which is the most favourable case available. The first addendum's 8.3% was the right arithmetic
for the wrong unit, and reporting it without this figure beside it would overstate the problem in
one direction while leaving the load-bearing number underived.

**That the unit was wrong was derivable from the code the whole time.** `ALLOWED_CARDINALITIES`
is a module constant with a comment explaining itself. This is the fourth time in this programme
that a derivable fact was left underived, and the first addendum was written specifically to
avoid that failure -- which is worth recording rather than quietly fixing.

**What it does not say.** It does not adjudicate any criterion. Candidate 2 matched cliques
across scenes and a partial-recurrence criterion tolerates absence by construction, so what any
given rule needs from a triple is its own question. It supersedes nothing: the feature-level rate
and the configuration-level reading are both still what they were. And the bound runs the same
way -- identical geometry is the favourable case, so the real figures are worse.

**T4E.24 addendum (2026-09-11): the same receipt, read at the level the criterion works on.**

This is **exact arithmetic on the committed receipt**, not a new measurement. It has no separate
blindness claim, declares no prediction, and adjudicates nothing. It is recorded because the
consequence was derivable from evidence already in hand, and this programme has three times met
the lesson that a derivable fact left underived is a fact nobody has.

The headline figure of T4E.24 is a rate over **features**. The identity path does not consume
features in isolation -- it consumes **configurations**, and a constellation missing two of its
seven members is a different constellation. Read per configuration, the same 414 features say:

```
60 configurations, 3 to 10 features each (median 7)

configurations with every feature seen at least once, anywhere    10/60 = 0.167
configurations with every feature seen in all six scenes           5/60 = 0.083

never-seen features per configuration
   0 -> 10 configurations      1 -> 18      2 -> 12      3 -> 9
   4 ->  4 configurations      5 ->  6      6 ->  1

fraction of a configuration that is present in all six scenes:  mean 0.572, median 0.528
```

**50 of 60 planted configurations contain at least one feature the extractor recovered in no
scene at all.** For those, no tolerance over scenes helps and no criterion over scenes can see
the planted configuration, because the object it would have to match was never assembled in any
of the six. The feature-level ceiling of 0.63 is therefore not the ceiling that binds: at the
level the criterion operates, on the most favourable geometry available, **8.3%** of planted
configurations survive intact across all six scenes and **16.7%** are even recoverable in
principle.

**What this does not say.** It does not say any particular criterion requires whole
configurations -- candidate 2 matched cliques across scenes, and a partial-recurrence criterion
tolerates absence by construction, so the requirement differs per criterion and none is
adjudicated here. It does not supersede the feature-level rate, which remains the quantity the
declaration asked for. It adds no evidence: every number above is a different reading of rows
already in `measurements/t4e24_false_absence.json`, and the bound still runs the same way --
identical geometry is the favourable case, so the real figures are worse.

**T4E.23: the study trail is on screen, with rendered evidence.** `StudyTrailView.tsx`, a new
"Study trail" tab, and `frontend/e2e/study-trail.spec.ts` -- **six tests passing in Chromium
against the real API and the real frontend**, with three screenshots captured under
`frontend/e2e/artifacts/`. PLAN section 5's acceptance for an interface slice is rendered
evidence with refusals demonstrated on screen rather than described, and this is that evidence
rather than a claim about it.

What the panel draws, and why each rule exists because of something this programme did:

* **A study is the chain it ran** -- declared, adopted or signed, measured -- so which result
  answered which question is not reconstructed from filenames. Thirteen studies render.
* **A verdict never appears without what it may not be used for.** The boundary renders inside
  the result, not beneath it.
* **A correction is a badge on the study.** T4E.17's three amendments and superseded signature,
  T4E.18's `CORRECTION_2026_09_10`, T4E.20's `GATE_CORRECTION` are all visible without opening
  anything, because a corrected record that reads as current is the dangerous case.
* **A question with no answer is shown, not filtered.** T4E.16 renders as *declared, not
  measured* with its withdrawal mark, and T4E.7 likewise. A view that hid T4E.16 would hide the
  cheapest result the programme produced.
* **The surface states what it will not do**, from the server's own refusal list, rather than
  implying it by an absence of buttons.

**A defect the rendering caught that the API tests could not.** The first `BOUNDARY_KEYS` list
omitted `boundary` and `acceptance_boundary` -- the plainest names of all -- so seven older
measurements rendered as *"no stated boundary; this record predates the convention"* when the
clause was right there in the record. **A viewer that under-reports a boundary is worse than one
that omits the field: it makes a false statement about the evidence, on screen.** Every
measurement now reports its boundary; the count of false "predates the convention" claims went
from seven to zero.

Two test defects were also caught by running rather than by reading: a substring selector that
matched `declared_before_measurement` instead of the *Declared* column heading, and -- after the
boundary fix put more text on the buttons -- a `close` control that matched four elements
because several boundaries contain "no closure of D96".

**T4E.22: the results become reviewable.** Until this slice `/api/v1/identity` served
`data/identity_calibration` and nothing else, so a reader could see that a study had been
*declared* and never what it *measured*. Every offset, falsified prediction and corrected
diagnosis lived in `measurements/` and in git, where no interface could reach them. **A
declaration without its result is a promise; a result without its declaration is an assertion;
only the pair is evidence.**

Three additions, all read-only and all in the router's existing idiom:

* **`GET /measurements` and `/measurements/{name}`** serve the measurement store with each
  record's verdict attached, and with **what it may not be used for attached rather than
  beside it**. Fifteen different key names have been used for that clause across this
  programme's records; all fifteen are collected rather than normalised, because renaming keys
  in committed evidence to suit a viewer would be rewriting evidence to fit its display.
* **`GET /studies`** joins each task into the chain the work actually runs -- declaration,
  adoption or signature, measurement, outcome -- so a reader is not left reconstructing from
  filenames which result answered which question, or whether the question was fixed before the
  answer was known.
* **Summaries now carry a verdict and its corrections.** A record corrected or superseded in
  the open is marked in the *summary*, because the summary is what a reader sees first and **a
  corrected record that reads as current is the dangerous case**.

Two states are shown rather than filtered, and both are load-bearing. A **declaration with no
measurement** is a question fixed and deliberately unanswered -- T4E.16 was withdrawn before
adoption by derivation, and a view that hid it would hide the cheapest result the programme
produced. A **measurement with no stated boundary** predates the convention and is listed by
name rather than passed over.

**A defect caught in verification, not in review.** The first study key split on the leading
separator, which made `t4e19_positional_error.json` a study of its own and left every
declaration reading as unanswered -- a view worse than none. It now takes the leading `t4eNN`
token under either convention, and a test pins both spellings.

**What this slice is not.** It is the API half. `IdentityDeclarationView.tsx` renders the
admissibility matrix and does not yet render studies or measurements, and PLAN section 5's
acceptance for an interface slice requires **rendered evidence captured in VERIFICATION.md,
with refusals demonstrated on screen rather than described**. That evidence does not exist for
this surface, so no interface claim is made here beyond what a client can now fetch.

**T4E.21: competition does not explain the gap, and the declared prediction is falsified.** 247
vortices planted across 39 frames at the record's own feature density, every centre known by
construction. `measurements/t4e21_faithful_background.json`.

**The gate is code this time, not prose.** `src/benchmarks/synthetic_backgrounds.py` implements
all three declared conditions -- median inside the band, no frame above the ceiling, and a median
strictly above zero -- and five tests pin it, including the exact case T4E.20 let through. The
redundant zero check is kept deliberately: a redundant condition that names the failure it was
written for is worth more than a tidy one that does not.

```
gate PASSED: median 4 features/frame (declared band [4,10], record median 7), max 9

planted 247 | recovered 167 | NEVER RECOVERED 80 (32%) | spurious 3

symmetric at real density   0.295 cells (8.2 km)
T4E.20 quiet background     0.370 cells (10.3 km)
real record                 1.630 cells (33.8 km)
by stretch: 1.0 -> 0.295    1.5 -> 0.338    2.5 -> 0.633
```

**The prediction said the offset would rise toward 1.63 cells at real density. It fell.** The
declaration named this outcome in advance: neither asymmetry alone nor competition explains the
real offset.

**What density does cost is recall, not accuracy.** Recovery collapses from 91% on the quiet
background to **68%** here -- 80 of 247 plantings never found at all -- while the features that
do survive are placed no worse. Suppression removes the maximum that would have marked a centre;
it does not displace the ones that remain. That separation was not predicted and is the
substantive finding.

**Disclosed: the gate passed at the bottom of its band.** Median 4 against the record's 7, so
competition is under-represented even though the declared band was met. That forbids claiming
density has been tested at the record's own level. It permits the observation that going from 0
to 4 features per frame did not raise the offset at all, which makes it implausible that 4 to 7
would triple it -- an argument, and labelled as one.

**About 25 km of the real 33.8 km remains unexplained.** Neither the estimator's asymmetry bias
at realistic parameters nor competition accounts for it. On this evidence the leading remaining
candidate is that **an 850 hPa relative-vorticity maximum and an agency's reported surface centre
are not the same quantity** -- which would make about 34 km an intrinsic cost of this comparison
rather than an error to be fixed, and would make restating the T4E.18 acceptance against a
defensible tolerance the correct response. That restatement is its own declaration and is not
made here. T4E.19's storm-type test found no dependence on `NATURE`, which is evidence *against*
a transition-driven separation, so this candidate is not yet comfortable either.

**T4E.20: the centroid is pulled toward the broader side, and now it is demonstrated rather than
hinted at.** 397 vortices planted at known centres on phase-randomised real frames, sweeping
scale, amplitude and asymmetry. `measurements/t4e20_synthetic_centre.json`.

**The gate failed by its own declared criterion, and the code said otherwise.** The declaration
required a background yielding feature counts comparable to the record and named "hundreds, **or
none**" as disqualifying. The phase-randomised backgrounds yield a median of **0** features where
the record yields **7**; the coded check tested only an upper bound and never implemented the
"or none" half. So the synthetic problem is *easier* than the real one -- a planted vortex faces
no competition where a real frame has seven features and a suppression rule between them.

That cuts toward the conclusion rather than away from it: an easier regime that still reaches the
observed offset at adverse parameters makes the mechanism more credible. What it forbids is the
quantitative claim that the estimator explains 33.8 km. What it supports is that the mechanism is
real and can reach that size.

```
does the offset point ALONG the stretch axis?   (0 = along it, 45 = unrelated)
  stretch 1.0   45.3 deg  n=107      offset by stretch:  1.0 -> 0.370 cells
  stretch 1.5   34.7      n=107                          1.5 -> 0.427
  stretch 2.5   18.9      n=112                          2.5 -> 0.841

symmetric, by scale:      sigma 2 -> 9 gives 0.234, 0.288, 0.580, 0.557 cells
symmetric, by amplitude:  ratio 8 -> 32 gives 0.686, 0.436, 0.207 cells
scale recovery:           planted 2.00/3.67/6.00/9.00 -> 2.07/3.79/6.01/8.93
at the record's medians:  0.368 cells = 10.2 km      (real record: 1.63 cells = 33.8 km)
at adverse parameters:    4.952 cells = 137.7 km
not found at all:         35 of 397 (8.8%), counted rather than dropped
```

**Cause A's directional prediction is confirmed cleanly**: monotone from unrelated at symmetry to
strongly aligned at 2.5x stretch. **Its magnitude prediction is confirmed too.** And the
estimator *sizes* a feature almost exactly while mislocating it, so this is a centroid problem
and not a scale one -- which also rules out the diverged-scale defect fixed earlier as an
explanation.

**What it does not settle**: that cause A accounts for the *whole* real offset. At the record's
median parameters it gives 10.2 km against 33.8 observed, and the failed gate forbids treating
the synthetic figure as a like-for-like prediction. Nor does it establish that the extractor
should be changed -- a displaced centroid on asymmetric features is a known property of windowed
centroids, and whether a better estimator exists for this field, and what it would cost
elsewhere, is separate declared work. The post-hoc southwest displacement from T4E.19 is
untouched: the stretch axis was drawn uniformly here, so this design cannot see a fixed
geographic bearing and does not claim to.

**T4E.19: the offset decomposed, and none of the three declared causes survives.** Predictions
were fixed before the measurement precisely because three causes that all produce "about 35 km"
are indistinguishable by magnitude. 176 interior observations, 162 paired within 200 km, 154
core after separating the dateline group. `measurements/t4e19_positional_error.json`.

```
core offset: median 33.8 km  q75 54.7  q90 92.2   against a catalogue radius of 15.2
             inside their own radius: 37 of 154        ratio 2.22

prediction tests (Pearson, leave-one-storm-out range; n=154 from 16 STORMS)
  A estimator bias   r(offset, feature sigma) = +0.143   [+0.025, +0.186]
  B catalogue uncert r(offset, radius)        = +0.020   [-0.033, +0.096]
  C physical         r(offset, wind)          = -0.136   [-0.241, +0.066]
  C physical         r(offset, latitude)      = +0.089   [+0.033, +0.114]  wrong sign
  C physical         by storm type: ET 36.2  MX 35.9  SS 32.7  TS 34.0 km
```

**Cause B is ruled out.** The offset does not track the agencies' own disagreement about where
the centre is, so this is *not* a case of comparing at the wrong tolerance -- which had been the
outcome that would have required no code at all.

**Cause C is not supported**, and the sharpest test is storm type: a transitioning or subtropical
system shows the same offset as a tropical one to within 3.5 km. The latitude term runs the
*opposite* way to the prediction and the intensity term straddles zero.

**Cause A survives in sign only** -- the one correlation that stays on one side of zero across
every leave-one-storm-out fit, explaining about 2% of the variance. Far too weak to carry the
explanation.

**So the declared answer is that the three causes are not separated at this sample size**, which
condition 3 anticipated and required to be reported rather than resolved by picking the most
plausible.

**What it does settle**: the offset is about 34 km, roughly 1.6 grid cells, and is largely
indifferent to storm scale, intensity, type and latitude. Whatever produces it is a property of
the extraction rather than of the catalogue or the storms.

**Fenced off as post-hoc and not adjudicated**: in grid cells the feature sits on average 0.598
south and 0.463 west of the catalogue position -- a systematic displacement of about 0.76 cells
with a mean absolute displacement of 1.634, so roughly half the offset is systematic and half is
scatter. It is *not* a fixed coordinate shift (the coefficient of variation is 0.755 in km
against 0.737 in cells; an indexing error would cluster tightly and does not). This pattern was
not among the declared three and was noticed in the data that would have to test it, so it is a
hypothesis for its own declaration -- testable on synthetic cyclone-like fields where the true
centre is known by construction.

**CORRECTION (2026-09-10): the diagnosis above was wrong, and is superseded rather than edited
away.** It claimed the extractor does not find the cyclone and named the phase-randomised
frame-maximum calibration as the cause. Both claims fail on measurement. The temperature case
was measured -- threshold 306.74 K against a field maximum of 302.5 K -- and generalised to
vorticity without ever being checked there.

**The calibration clears comfortably.** On the GITA frame the observed maximum is 1.5175e-3
against phase-randomised surrogate maxima of 1.76e-4 to 2.55e-4 -- a factor of six. Measured per
storm at the catalogue cell +/- three cells, **11 of 18 are accepted**: above threshold, a local
maximum, localised, and not off-frame.

**A second thing that was missed.** Of the five registered surrogate methods, `aaft`, `iaaft`,
`circular_shift` and `block_bootstrap` all preserve the marginal distribution, so the surrogate
frame maximum *equals* the observed maximum and the test has **no power whatever**. Only
`phase_randomise` can reject at all. That is a property of a frame-maximum statistic and is
worth knowing before any of them is proposed as a replacement.

**And raw-field extraction is better than the SWT planes, not worse** -- the reverse of what the
earlier four-storm comparison suggested.

```
                     nearest km (median)   inside radius   features/frame
raw field                        52.1          3 of 18       0-13, med 7
SWT planes                      127.1          2 of 18      73-153, med 123
```

**The two real failure modes.** Four storms sit at longitude **179.0 to 179.8** and are refused by
the R13 off-frame rule, because their own window at 2 sigma overruns the crop's eastern boundary
at 180.0; their nearest features are 1207, 2028, 2465 and 3685 km away. The crop stops at the
dateline because the request layer refuses dateline-crossing requests by design -- and Fiji,
Tonga and Samoa sit exactly there. For every storm *not* at the dateline the nearest feature is
16.6 to 99.3 km, median about 35, against a catalogue radius of 15.3: **one to two grid cells, a
factor of two to three, not an order of magnitude**.

**The verdict is unchanged -- acceptance still fails, 3 of 18 against a bar of 9 -- but the
diagnosis is what a next step would be built on**, and building a replacement calibration on the
wrong cause would have failed for a reason nobody had measured. What stands from the original
record: the variable carries the signal, the latitude band is not the cause, and the acquisition
is sound at 48 shards and 5,844 frames.

**T4E.18 acquired and FAILED its acceptance. The variable carries the signal; the extractor does
not find it.** 48 shards, **5,844 frames matching the expected calendar exactly**, 336.9 MB, 60
minutes of CDS queue, lat -58..-18, each shard carrying its own digest.
`measurements/t4e18_acceptance.json`.

```
condition 1  nearest feature inside the catalogue radius   2 of 18   (needed 9)  FAILED
condition 2  three features inside the radius              1 of 18   (needed 9)  FAILED
condition 3  features per frame 73-153, median 123                               MET
condition 4  refusals by name                                                    MET
nearest feature km: min 29.9  median 127.1  max 2055.9
```

**Three things are established and are not in doubt.** The variable carries the signal: measured
on the field before any extraction, GITA sits at the **0.02nd percentile** of its frame with the
single most cyclonic cell **one grid cell** from the catalogue position, and the declared sign
convention is correct. The crop is no longer the problem: the first probe sampled each storm's
*first* in-box observation, which is always where it entered the box and therefore always at the
boundary the extractor refuses features at -- **a sampling artefact, corrected**; re-sampled at
each storm's deepest interior observation, latitudes -20.0 to -38.6, it still fails. And the
representation is not the problem: raw extraction yields **3 to 7** features per frame against
73 to 153 through the SWT planes, and lands *further* from the storm in three of four cases.

**What it points at is the extractor's calibration.** `local_maximum_extractor` admits a maximum
only if it clears a threshold calibrated from the frame maxima of **phase-randomised
surrogates**, which preserve the power spectrum -- and for smooth geophysical fields those
surrogates routinely produce maxima as large as the observation's. On 850 hPa temperature the
threshold came out at **306.74 K against a field maximum of 302.5 K**, so nothing could clear it
at all. On vorticity, GITA -- the most cyclonic cell in its frame by a wide margin -- yields four
raw features, none within 2,900 km. The extractor's own docstring says it assumes an isotropic
peak on a flat baseline and that the honest response to a field it does not suit is **a second
registered extractor, not a special case inside this one**.

**The finding: the identity path's front end does not detect the phenomenon the signed catalogue
labels, in a field where that phenomenon is unambiguous and dominant.** That is a property of the
instrument -- not of the atmosphere, the catalogue, the crop or the variable.

**What it does not establish.** Not that vorticity is the wrong variable; the signal is
measurably present and correctly signed, and nothing extracted it. Not that the acquisition was
wasted -- it is what made the diagnosis possible. Not that a second extractor would succeed,
which is untested and would be its own declared work. `kind_recurrence` remains unevaluable, now
for a reason one layer further in than it was this morning.

**The variable works. The crop does not.** Two months of 850 hPa relative vorticity were
acquired as a probe before the declared full request -- 14 MB and about 100 seconds against
230 MB and 1.6 hours -- and tested against acceptance condition 1.
`measurements/t4e18_vorticity_probe.json`.

**Confirmed: a cyclone is a strong, correctly-signed, localised signal in this variable.** GITA
sits at the **0.02nd percentile** of its frame and the frame's single most cyclonic cell is
**one grid cell** from the catalogue position; HOLA at the 0.16th, LINDA 2.32nd, IRIS 1.20th.
The sign convention declared before acquisition is right.

**Failed anyway, and not because of the variable.** No storm of four had a feature inside its
catalogue radius. These storms sit at latitude -20.2 to -21.1 and the crop's northern edge is
**-20.0**, so they are one to four cells from the boundary, where the extractor refuses features
by design -- 34 rejected as `outside_valid_interior` on the GITA frame alone.

```
SP cyclone observations, lon 140-180, 2018-2021:   917
  inside the crop's lat band [-60,-20]:            210
  NORTH of the crop, excluded entirely:            707   (77%)
  in-crop median latitude -23.3; 74 within 2 deg of the edge
```

**The record was built for a New Zealand forecast experiment, not for cyclone identity**, and the
two purposes want different domains.

**Moving the box north fixes the geometry and costs the kind label.**

```
box (lat)      obs  w/radius  storms  interior   pairs     NATURE base rate
-60..-20       210      176      18       113    10,721          0.571
-45..-5        908      684      30       683   164,450          0.872
-40..0         900      680      30       678   163,880          0.875
```

At the same 161-pixel shape and 40-degree span the crop planner already assessed, the population
goes from 18 storms to 30 and from 113 interior observations to 683. But the `NATURE` base rate
rises to **0.872**, because at tropical latitudes almost every system is `TS`: a rule answering
"same kind" to everything would be right 87% of the time. **The label that discriminates usefully
in the southern box is close to degenerate in the northern one.**

**This is a decision, not an optimisation.** Choosing whichever box flatters a later result is the
horse race every declaration in this sequence forbids, so the trade-off is recorded and put to
the maintainer rather than resolved. And it is not free: the T4E.17 catalogue was **signed** on
terms naming this crop and a base rate of 0.571, so a different crop is a different evidence base
and the signature would have to be re-given rather than carried over.

**The full acquisition was not run.** Acquiring 230 MB onto a crop that cannot support the
evaluation is the mistake the probe exists to prevent.

**T4E.18 declared: acquire a variable in which a cyclone centre is an extractable feature.**
`data/identity_calibration/t4e18-vorticity-acquisition-design.json`, declared before any request
and **not yet acquired** -- the maintainer authorised the acquisition on 2026-09-10 and the
credentials it needs are not on this machine.

**The variable is `vorticity`** -- ERA5 relative vorticity, distinct from `potential_vorticity`
-- at 850 hPa, the same level, crop and grid as the existing record. A cyclone is a compact
near-isotropic extremum in it, an order of magnitude above a background of ~1e-5 s^-1, which is
the shape the registered extractor declares it assumes.

**Mean sea level pressure would be more commensurate and is not chosen, for a reason that is a
constraint rather than a preference.** IBTrACS records minimum central pressure directly, so
MSLP is the variable closest to the catalogue's own definition of a centre. But `cds_source.py`
refuses any dataset except `reanalysis-era5-pressure-levels` by design, and MSLP is single-level.
The chosen variable is the best available *within the machinery as it stands*, not the best in
principle, and if vorticity fails its acceptance then MSLP is the next candidate and the
acquisition layer has to be extended to reach it.

**The sign convention is declared before the data exists.** `local_maximum_extractor` finds
maxima; in the southern hemisphere cyclonic rotation is **negative** relative vorticity. A
maximum-finder on raw vorticity in this crop would locate anticyclones and miss every catalogue
storm. The field is therefore negated before extraction. Negation is a transformation this
programme chose, not a property of the data, and declaring it now is what stops it becoming a
knob turned after a disappointing result. The crop lies wholly south of the equator, so one sign
applies throughout; a crop spanning the equator is not covered.

**The window stops at 2021-12-31, deliberately.** Not acquiring 2022-2023 makes the
forecast-test reservation **physical** rather than a matter of policy: a frame that does not
exist cannot be opened by accident, by a refactor, or by someone who has not read the
constraint. The cost is a second request later, accepted knowingly -- this session has repeatedly
found that reservations enforced in code outlast reservations recorded in prose.

**Acceptance is declared before the data arrives**, which is the correction to what went wrong in
T4E.17. That catalogue was checked for independence, for the source of its radius and for the
adequacy of its population, and never for whether the record's variable could *see* what the
catalogue labels. The conditions now: the join must close by an order of magnitude, at least half
the storms must have three features inside their own catalogue radius, the extractor must be
neither starved nor swamped, and refusals are counted by name.

**What it still needs from the maintainer**: CDS credentials (`~/.cdsapirc` or CDSAPI_URL /
CDSAPI_KEY -- never pasted into a conversation and never recorded in this repository), a
one-time ERA5 licence acceptance on the CDS account, and the explicit network consent the layer
requires as a separate act.

**The join cannot be made, and the blocker has moved.** With the catalogue signed and the
extractor fixed, the remaining question was geometric: is there a constellation for a storm to
be the identity *of*? Measured in `measurements/t4e17_join_feasibility.json`.

```
nearest feature to the storm:  min 12.3 km   median 153.9 km   max 949.1 km
catalogue radius:              median 15.2 km
features per frame:            42 to 86 across the ten SWT planes

storms with >=3 features within the neighbourhood (any plane / same plane)
   25 km :  0 / 0      100 km :  1 / 0      400 km : 15 / 10
   50 km :  0 / 0      200 km :  7 / 3              of 18 storms
```

**At the catalogue's own declared radius there is usually no feature at all** -- the nearest is a
median ten times that radius away. Only 2 of 18 storms have any feature within 25 km and **none
has three**, which cardinality 3 requires. It is not a shortage of features; they are simply not
where the storms are. Widening the neighbourhood to 200-400 km until a triple appears would
substitute a radius *we* chose for the one the catalogue supplies, and "matched at its own
declared radius" is exactly what `external_reference` evidence means.

**The cause is the record's variable.** Features here come from **850 hPa temperature** -- thermal
structure. IBTrACS positions are cyclone centres, defined operationally by wind and pressure. A
warm core and a circulation centre need not coincide, and under shear or extratropical
transition they can be hundreds of kilometres apart. This is derivable in hindsight and was not
derived in advance; it is a property of the variable the record holds.

**What this does not show.** Not that the catalogue is inadequate -- it is signed, independent
and correctly specified, and nothing here bears on its labels. Not that the signature or any
criterion fails; none was run. Not that `kind_recurrence` is unevaluable in principle -- only
that it is unevaluable **against this record**, whose single variable is 850 hPa temperature.

**What would unblock it** is a record variable in which a cyclone centre is an extractable
feature: relative vorticity, mean sea level pressure, or 10 m wind. That is a new CDS
acquisition and a new record, and it is the maintainer's decision rather than a consequence of
this measurement.

**A diverged scale is now refused by name rather than acted on (`core/extraction.py`).** Found
while probing whether the signed catalogue could be joined to the record at all. `_localise`
corrects the integral estimator for the fraction of a Gaussian its window captures, and that
correction is a **fixed point that diverges**: a larger sigma captures less of itself, dividing
by the smaller capture returns a larger sigma. The upstream finite check does not catch it,
because a runaway here is a large *finite* number rather than an infinity.

On a real ERA5 SWT `level_1/HH` plane it produced **sigma 72,404 cells on a 161-cell frame**,
then asked `_disc_amplitude` for a **42.5 TiB** index array and killed the pass. **Every frame of
the acquired record was unextractable because of it**, which silently blocked the identity path
on real data -- not just the `kind_recurrence` join that exposed it.

The refusal category `unmeasurable_scale` already existed and was simply unreachable. A feature
wider than the frame was not measured by the frame, so a scale past the frame's own extent is
refused and counted. The bound is generous by construction: a Gaussian whose width equals the
frame is already unmeasurable from it, and real features on these planes are single-digit cells.

With the guard in place that frame yields **55 features** across the SWT planes, 9 of them
refused by name. Three regression tests pin it -- the guard firing on the path that actually
diverged (a collapsing refined amplitude), the refusal being counted rather than ending the
pass, and a clean Gaussian still extracting with zero `unmeasurable_scale` rejections, so the
bound cannot be trimming real features.

**Full suite after the change: 4,736 passed, 6 failed, all six checked.** Five are pre-existing
and were confirmed by re-running with the change stashed: `t4e_identity_certified` fails
identically at split 0.4667 (the documented T4E.9 FAIL), `test_imports` is that same benchmark
through the API, and three `test_browser_evidence` failures are the unrecorded browser evidence
PLAN already lists as owed. The sixth was the test inventory, and is corrected here.

**T4E.17 external catalogue (2026-09-10): SIGNED. The primary target is evaluable.**

**SIGNED by the maintainer on 2026-09-10**, at design sha256 `c692ea19...` -- the amended
design, not the original. `data/identity_calibration/t4e17-external-catalogue-signature.json`
records the act; it does not perform it, and the binding to a content hash means the terms
signed cannot drift from the terms recorded.

**What the signature makes true: `kind_recurrence` is evaluable from the tool for the first
time.** Its admissible evidence is `external_reference` alone; none has existed in this
programme through T4E.7 to T4E.16, seven measured criteria and one withdrawn. The primary
scientific target has been unevaluable throughout, and is not any more.

The terms signed are the amended ones -- 176 observations from **18 storms**, `NATURE`
adjudicating at a base rate of **0.571**, a per-observation radius from inter-agency spread with
a median of 15.2 km, and 34 single-agency observations refused by name. The superseded figures
(210 observations, 20 storms, 0.391) stay visible in the design, because the correction ran in
the direction that would have flattered a later result.

**Four obligations come with it**: the publisher's citation (Gahtan et al. 2024, DOI
10.25921/82ty-9e16, with Knapp et al. 2010); the catalogue is never committed and a digest that
fails to reproduce invalidates any evaluation built on it; every rate carries a storm-clustered
interval, because the unit of independence is the storm and not the pair; and **only the
two-sided error rates may be reported**, since accuracy is meaningless at a base rate of 0.571.

**Signing evaluates no criterion.** A rule for this evidence needs its own declaration, written
before these base rates shape it, and it can borrow nothing from the synthetic sequence.

```
T4E.17 -- IBTrACS v04r01, South Pacific subset, sha256 631f76b9..., 35,482,417 bytes
matching domain: the record's own crop, lat -60..-20, lon 140..180, 850 hPa
development window: 2018-01-01 .. 2021-12-31   (2022-2023 stays closed)

in-box observations at synoptic hours   210     from 20 distinct storms
cross-storm pairs (kind_recurrence)  20,241     within-storm pairs excluded
same kind by NATURE                   7,907     base rate 0.391
same kind by USA_SSHS                 3,679     base rate 0.182
storms per season 2018-2021           8, 4, 2, 6
```

**Why this is the primary target and why it has never been evaluable.** `kind_recurrence`
admits `external_reference` evidence *alone* -- "a signed, frozen catalogue matched at its own
declared radius" -- and no such catalogue has existed in this programme. Every measurement to
date has been on record-derived proxies or synthetic scenes. Seven criteria were declared and
measured on synthetic partitions and an eighth withdrawn; none of it touched this.

**The admissibility question is the whole difficulty, and most candidates fail it.** Blocking
indices, IMILAST-style track intercomparisons and most atmospheric-river catalogues are computed
**from reanalysis**. Against an ERA5 record they are record-derived proxies wearing a
catalogue's name, and admitting one would reintroduce precisely the circularity
`external_reference` exists to exclude. IBTrACS passes because it merges operational best-track
data from the meteorological agencies -- BoM, JMA, NHC, the Shanghai Typhoon Institute and
others -- assigned by forecasters and post-season review, not by any algorithm run over ERA5.

**The residual dependence is disclosed rather than assumed.** Best-track analysts use whatever
guidance was operationally available, which can include model fields. The labels are independent
of *this* record and of ERA5 as reprocessed here; they are not independent of numerical weather
prediction in general. That is weaker than a purely observational catalogue would give.

**The unit of independence is the storm, not the observation.** There are 20,241 pairs but only
**20 storms**, and successive six-hourly positions of one cyclone are strongly correlated. Any
interval computed as though the pairs were independent would be wrong by roughly the square root
of the clustering factor. Every rate must carry a storm-clustered interval, and the effective
sample size is nearer 20 than 20,241. This is stated first because it is the single easiest way
for a result here to be overclaimed.

**Within-storm pairs are excluded by construction.** They bear on `spatial_persistence`, a
different declared target with different admissible evidence. Counting them here would answer
the easier question and report it as the harder one.

**The kind label is fixed before any signature is computed.** `NATURE` -- the catalogue's own
storm-type classification -- adjudicates, because `kind_recurrence` asks what a system *is*;
`USA_SSHS` is an intensity ordinal and is recorded as declared characterisation only. Both base
rates were measured before declaring, so choosing the more favourable one afterwards would be a
horse race. `MX` (agencies disagreed) and `NR` (not reported) are **refusals by name**, not
classes: treating the catalogue's own uncertainty as ground truth would corrupt every rate built
on it.

**The prior odds here are nothing like the synthetic ones.** About 1:1.6 on `NATURE` and 1:4.5
on `USA_SSHS`, against **1:399** in the synthetic partitions. The 36-fold specificity shortfall
that dominated seven synthetic candidates is substantially a property of that design rather than
of the identity question, and no rate measured here may be compared directly to the synthetic
ones.

**The data is bound by digest and is not committed.** 35.5 MB of third-party data does not
belong in this repository -- the same discipline the acquired record and the market records are
held to. A re-download that fails to reproduce `631f76b9...` invalidates the evaluation and must
say so.

**2022-2023 stays closed.** The catalogue lists 393 in-box observations from 13 storms there.
That number comes from the **catalogue**, not the record: no ERA5 frame of the forecast-test
period was opened, and none may be. It is disclosed because it was seen.

**What this does NOT do.** It evaluates no criterion -- a rule for this evidence needs its own
declaration, written before these numbers shape it. It says nothing about any of the seven
synthetic candidates, none of which transfers here. It covers one basin, one 40-degree box, four
years and 20 storms, with a coarse six-value kind label two of whose values are refusals. It
approves no mining radius, discharges nothing of T4E.8's acceptance, and closes none of D96 to
D100. And the catalogue supplies identity, not physics: a criterion agreeing with it has agreed
with the contributing agencies' operational judgements, not with the atmosphere.

**Amended before signature, and the check was worth doing.** Two requirements in the design as
first written do not survive contact with the catalogue.

**The radius had no source.** The design said to take the matching tolerance from "the
catalogue's own reported uncertainty for the contributing agency". IBTrACS has **174 columns and
none of them reports position uncertainty**, so that requirement could not have been implemented
and signing it would have committed the maintainer to a matching parameter with no legitimate
source. What replaces it is supplied by the catalogue and chosen by nobody: independent agencies
report their own position for the same observation, and the radius is the greatest distance from
the catalogue's position to any contributing agency's. In this basin and window the agencies are
USA (186 observations), BOM (102), WELLINGTON (83) and NADI (67).

```
radius = max distance from catalogue position to any contributing agency
  median 15.2 km   q90 47.2   q95 89.4   max 145.2
  record grid cell at 40S: 21.3 km longitude, 27.8 km latitude
coverage: 176 of 210 observations have two or more agencies; 34 have one
```

The median radius is **below one grid cell**, so matching is tight at grid scale for half the
observations; q95 is about four cells, so the radius varies and must travel per observation
rather than be summarised -- which is what "matched at its own declared radius" requires anyway.
The **34 single-agency observations are refused by name**: the catalogue supplies no radius for
them and they are not given a default.

**The adjudicating base rate was wrong, in the flattering direction.** Applying that refusal and
the design's own MX/NR refusal leaves **176 observations from 18 storms**, and the `NATURE` base
rate is **0.571, not 0.391** -- same-kind pairs are the *majority*. A rule answering "same kind"
to everything would be right 57.1% of the time, so accuracy is a meaningless summary here and
only the two-sided error rates may be reported. `USA_SSHS` sits at 0.227 and would make any
criterion look better; **switching to it now, having seen both, is exactly the horse race the
design forbids**, and `NATURE` stays adjudicating on the principle that `kind_recurrence` asks
what a system *is* rather than how strong it is.

The original figures are superseded in place, not edited out. The check cost one verification
pass and saved a signature on an unimplementable design plus a base rate wrong by 0.18 in the
direction that would have made any later result look better than it was.

**Ten tests hold the design**: that it is not signed by code, that the data is bound by digest
and absent from the repository, that a reanalysis-derived catalogue would be circular and is
refused for that reason, that the unit of independence is the storm, that within-storm pairs
belong to a different target, that the kind label was fixed before any signature was computed,
that agency disagreement is refused rather than classed, that the forecast-test period is not
opened, that no criterion is evaluated, and that the publisher's citation requirement travels
with the data.

**T4E.16 candidate 5 (2026-09-10): WITHDRAWN BEFORE ADOPTION, by derivation. Never measured.**

**The design.** The bar on a consistent set is set by a within-partition permutation surrogate,
per group size: admit a set of size `m` only if its diameter is tighter than the tightest set of
size `m` found in any of 199 surrogate replicates. A max statistic, so family-wise error across
every set of that size is controlled at 1/200 with **no alpha divided by anything** -- which is
what made candidate 1 inert. The bar falls with `m` automatically, because large coincidental
sets are rarer under the surrogate than small ones, and nobody chooses the rate at which it
falls. That is the constraint the four measured results jointly fix: **the evidence a group
carries must scale with the group.**

**Withdrawn, because the surrogate can reassemble the motif.** It redistributes the partition's
*actual* configurations. When the `S` motif copies land in `S` distinct scenes they form the
same consistent set with **exactly the same diameter**, so the strictly-tighter rule then
rejects the motif. The rate is combinatorial -- `S!/S^S` = 0.0154 at `S` = 6, confirmed by
simulation at 0.0166, 0.0168 and 0.0153 for `m` = 20, 84 and 220 -- and over 199 replicates that
is **0.9642, 0.9657 and 0.9535**. Candidate 5 would have failed conditions 1 and 5 at every
richness, with about 96% probability, for a reason having nothing to do with the signature. It
is a permutation null invalidated by its own permutations, the known failure of that
construction when the signal is a small set of near-duplicates and `S` is small.

**It cannot simply be repaired.** A valid null must *generate* fresh distractor scenes rather
than redistribute existing ones -- but then every pairwise distance changes per replicate, the
cached matrix that made the design runnable is worthless, and the cost returns to the measured
189 hours at richness 12 alone. **A feasible null is invalid here; a valid null is infeasible.**
Widening `S` would drive `S!/S^S` down fast, but `S` = 6 is the frozen partition size, and
changing it to rescue a criterion would be tuning the evidence to the rule.

**The declaration had labelled the point honestly, which is why it was checked.** Its derivation
section said of the large-`m` argument: *an ARGUMENT, NOT A PROOF ... if some surrogate replicate
produces a size-6 set tighter than the motif, the motif is rejected, and nothing derivable
excludes that.* Labelling it as argued rather than proved is what made it the next thing to
compute.

**What was not done.** Not adopted, not measured, not quietly altered into a variant that
passes. No surrogate was run on the T4E.14 evidence or on any block. The declaration is retained
unedited, superseded rather than deleted, because a design killed by derivation is part of the
record.

**The multiplicity position is unchanged.** A declaration withdrawn before adoption and before
measurement adds nothing to the accumulated multiplicity, because nothing was tested. **Seven
criteria have been measured in this sequence, not eight**, and both reservations -- 720-735 and
880-895 -- remain unspent.

**Two things survive and are kept**, because the next design meets the same arithmetic. The cost
finding: the clique enumeration is free at every richness and the matching is the entire
expense, at 0.8, 13.5 and 95.3 seconds per partition at richness 6, 9 and 12. And
`PooledDistances`, which computes a partition's distances once and keeps the metric's refusals
**as refusals** rather than as numbers -- optimisation must preserve the named refusal.

**What this does NOT license.** Not that size-scaled evidence is the wrong idea: the constraint
the four measured results fix is untouched, and what failed is one null, not the principle. Not
that the signature is inadequate -- no measurement was taken. Not that permutation surrogates
are wrong in general; this one is invalid *here*, at `S` = 6, against a signal of `S`
near-duplicates. And nothing about `kind_recurrence`, D96 to D100, a mining radius, or
recurrence across separated epochs.

**What is verified here is a derivation and an absence.** Six tests hold it: the withdrawal
status and the absence of any measurement or adoption record, the reassembly rate computed both
analytically as `S!/S^S` and by simulation, the record of why the design cannot simply be
repaired, the multiplicity position, `PooledDistances` keeping a refusal as NaN so it can never
leak in as a distance, and the cost finding that outlives the candidate.

**T4E.15 candidate 4 (2026-09-10): adopted and FALSIFIED.**

Adopted at sha256 `b1c3546795fb7ed68e4a758ca32fe7d6534b4bb7ec8f2bf4c09e5df1d92815f0`.

```
CANDIDATE 4 -- partial presence, 36 partitions (totals by richness and j)
false split 1.0000 on 32 of 36; recovered anything on 4
false admission 1.0000 on those 32; 0.5385 to 0.9143 on the other four
hallucinated presence 0.0000 to 0.9118, above the 0.10 bound on all but one
null admitted 3, 29, 87 of 122, 553, 1450 proposed   (required: zero)

CROSS-CHECK -- total recurrence, where candidate 2 admits exactly 15
rich  block      admitted   true  recovered   split    admit
6     100-105          20     15         15  0.0000   0.2500
6     200-205          33     15         15  0.0000   0.5455
9     200-205          44     15         15  0.0000   0.6591
12    300-305          84     15         15  0.0000   0.8214
null admitted 6, 8, 52 of 116, 595, 1486 proposed     (required: zero)
```

**FALSIFIED on five of six conditions, on both evidences.** Recall fails outright: false split
1.0000 on 32 of 36 partial-presence partitions, with the motif recovered on only four. Admission
fails everywhere -- there is no partition at any presence level or richness inside the 0.10
bound. Hallucinated presence reaches 0.9118, so the rule claims recurrence in scenes holding
nothing at up to 91% of its admissions. Both nulls admit where the answer is zero. Only the
refusal condition holds.

**The mechanism, and it is precisely backwards.** Closure rewards isolation. The consistent
group of a node is the maximal agreeing set, so a node whose only partner is one other node
forms a closed group of **two** and is admitted trivially -- the rule is most permissive exactly
where the evidence for an identity is weakest. Meanwhile a motif configuration in a present
scene usually *is* the mutual nearest neighbour of something unrelated in an absent scene, and
that single loose end breaks closure, so the motif group is rejected. It discards the strongest
evidence and admits the weakest.

**The cross-check settles what the derivation could not.** On total-recurrence evidence closure
recovers the motif -- which is arithmetic, derived before the run -- but admits 20 to 84 pairs
per block against candidate 2's 15, and breaks the null there too. So closure is not a
differently-shaped filter than `k = S`; it is a **weaker** one. It fails even where candidate 2
succeeds.

**A derivation that should have been made and was not.** That closure admits pairs trivially was
derivable before adoption. The declaration's feasibility section derived only that the criterion
*can* admit the answer -- that it is not inert the way candidate 1 was -- and never asked what
the rule does at the extremes of its own domain, where a group of two makes closure vacuous.
This is the third time this programme has met that lesson from a different direction: candidate
1's tail model was too conservative at its extreme, candidate 4's closure too permissive at its
own, and both were derivable in advance. **A feasibility check must ask what a rule does at the
smallest and largest cases it admits, not only whether it can reach the right answer.**

**What the blindness claim still bought.** It was honoured: nothing moved after the numbers
appeared, and no parameter was added to rescue anything -- there was none to add. So this is a
real falsification of closure as declared, not a rule that failed to survive its own tuning.

**What this does NOT license.** Not that the signature is inadequate -- it is unchanged from
candidate 2, which passes at `k = S`. Not that partition structure is the wrong resource. Not
that partial recurrence is undetectable: one rule was tested, and its failure mechanism is now
understood well enough to state what a successor must not do -- **it must not treat a group of
two as evidence on the same terms as a group of six**. Not that a parameterised closure rule
would fail; it might not, and it is a different candidate needing its own declaration, one that
could no longer claim blindness because this result has been seen. No mining radius, no
discharge of T4E.8's acceptance, no closure of D96 to D100, nothing about `kind_recurrence`, and
nothing about recurrence across genuinely separated epochs.

**The declaration below is retained unedited.** It is superseded by this outcome, not corrected
by it: its refusal to predict was right, its derivation of the cross-check's recall was right,
and its ceiling on what a failure licenses was written before the failure happened. What it
missed is recorded above rather than quietly repaired.

**T4E.15 candidate 4 (2026-09-10): declared, fixed in code, and measured nowhere.**

`data/identity_calibration/t4e15-closure-declaration.json`, status
`declared_before_measurement`.

**The criterion.** Form the mutual nearest-neighbour matching as candidate C does. A set of
configurations, at most one per scene, is CONSISTENT when every member is the mutual nearest
neighbour of every other, as in candidate 2. It is CLOSED when no member has a mutual nearest
neighbour outside the set. Admit the pairs inside sets that are consistent and closed, whatever
their span.

**There is no `k`, and no parameter of any kind** -- no clique size, no span threshold, no
radius, no ratio, no fitted model, no estimated normaliser. A group of three is admitted on the
same terms as a group of six, which is what makes it a criterion *for* partial recurrence rather
than a criterion with its tolerance widened. Every falsified candidate in this sequence carried
a number that could be moved after the fact; this one carries none, which is why the declaration
can forbid adding one outright.

**What closure reads that a span threshold cannot** is the *absence* of outside partners. A
threshold sees only how far a group reaches and is blind to what its members do elsewhere in the
partition. A coincidental group whose members also match configurations outside it is not a
coherent identity however far it reaches; a true group matching nothing outside itself is one
however short it is. That is what candidate 2 was actually exploiting -- never the number 6, but
a conjunction with no loose ends.

**A design was discarded by derivation and is recorded rather than forgotten.** Ranking groups
by span and admitting the widest is the most direct reading of the maintainer's own phrasing,
and it is dead on arrival: candidate 3 established that coincidental groups reach `S - 1 = 5`,
so at `j = 3` the widest group in the partition is a coincidence and the motif is rejected
outright -- false split 1.0000 before the rule runs. Catching that in advance is the discipline
candidate 1's failure installed.

**Derivable: it cannot be inert on total-recurrence evidence.** A group spanning all `S` scenes
uses a partner in every other scene, and the matching gives at most one partner per scene, so
those are *all* of each member's partners and the motif group is closed by construction. So
candidate 4 admits at least the motif there and cannot fail the way candidate 1 did.

**Not derivable, and the whole substance of the measurement:** whether the motif group is closed
on *partial*-presence evidence. A motif configuration in a present scene may be the mutual
nearest neighbour of something unrelated in an absent scene, and closure would then reject the
motif. Nothing measured so far bears on how often that happens.

**The honest worst case is stated in advance.** Candidate 4 may admit nothing on partial-presence
evidence. That would be a complete result -- and it is *not* candidate 1's failure, which was
inert everywhere including where the answer was easy. **No prediction is offered**, because
unlike candidate 3 there is no basis for one, and a guess dressed as a prediction is worth
nothing when checked.

**Recall is the informative half here, for the first time.** It is counted against C(j,2) and
not C(S,2), and it is genuinely at risk rather than arithmetic. Hallucinated presence -- pairs
touching a scene that holds nothing -- is bounded and reported as a condition of its own,
because a rule that claims recurrence in an empty window is worse for mining than one that
misses a real occurrence, and a pooled admission rate hides which is happening.

**What is verified here is an absence.** Ten tests hold the structure: closure broken by a
single loose end into another scene, closure admitting only a subset of what consistency admits
(so it cannot rescue recall consistency did not have), the criterion carrying no tunable
parameter -- asserted from its signature, not its prose -- the discarded span-ranking design
recorded with its derivation, the worst case and the refusal to predict, recall counted against
C(j,2), hallucinated presence as a condition of its own, and **no candidate 4 measurement or
adoption record anywhere in the repository**.

**Blindness and its limit.** Fixed before anything was measured on the T4E.14 evidence and
before the `k` profile's rungs at `k = 4` and `k = 3` were seen. The declaration was authored
before rung `k = 4` landed but committed after it, so the git history does not prove that
interval; the claim rests on the recorded ordering and is written down as such.

**Nothing is licensed.** No result exists. Blocks 880-895 have never been built and are refused
unconditionally; 720-735 could confirm nothing here, being total-recurrence partitions.

**T4E.14 partial-presence evidence (2026-09-10): built and audited, all five conditions met.**

Declared in `t4e14-partial-presence-design.json`, adopted as `EVIDENCE_CONSTRUCTION_AND_AUDIT_ONLY`,
audited in `measurements/t4e14_partial_presence_audit.json`.

```
T4E.14 AUDIT -- 39 partial-presence partitions built and checked
condition 1  presence counts                 MET
condition 2  no leakage by configuration count MET
condition 3  labels refused not guessed      MET, 0 refusals
condition 4  absent scenes are ordinary      MET
condition 5  reproducible                    MET
null 850-855 holds nothing anywhere          MET

j        true pairs   example planting patterns
3            3        [0,2,5] [1,3,4] [0,4,5] [2,4,5]
4            6        [0,1,3,5] [0,1,4,5] [0,1,3,4] [0,1,2,5]
5           10        [1,2,3,4,5] [0,2,3,4,5] [0,1,2,3,4] x2

configurations per scene: 20 / 84 / 220 at richness 6 / 9 / 12,
identical whether or not the scene holds the motif
```

**Why this is evidence and not a rule.** Candidates B, C, D, 1, 2 and 3 all ran on partitions
where the motif sits in *every* scene. Recurrence there is total, so no criterion has ever been
shown a scene in which a true configuration is genuinely absent -- candidate 3's false-split
column was 0.0000 partly for that reason, and its falsification, though sound, was measured on
one side only. A criterion built for partial recurrence and measured where recurrence is total
cannot fail for the right reason or pass for the right reason.

**What the audit checked, and what it deliberately did not.** It checked the test bed, not any
rule. No candidate was run on these partitions and nothing here adjudicates one.

**Two construction choices that could each have rigged a later result.** The planting subset is
uniformly random rather than contiguous: these scenes carry no ordering, so a contiguous run
would introduce temporal structure the generator does not have, and a criterion could then score
well by discovering the planting rule instead of the motif. And absent scenes hold the same
number of configurations as present ones -- 20, 84, 220 -- so presence cannot be read off the
size of a scene for free.

**The population moves with `j` and travels with the partition.** A criterion must recover
C(j,2) pairs: 3, 6 and 10, against 15 in the total-recurrence partitions. An error rate divided
by 15 on this evidence would be the quietest possible wrong number.

**Derived in advance, and not findings.** Candidate 2 requires a group spanning every scene, so
at `j < S` no such group containing the motif exists and its false split is 1.0000 here *before
it runs*. Candidate 3 recovers the motif only at `j = 5`. Both were stated in the design
declaration; neither is a discovery about a passing or a falsified candidate.

**A new reservation, made at the only honest moment.** Partial-presence blocks 880-895 were
reserved before a single partial-presence scene existed, and are refused **unconditionally** --
`ReservedPartialPresenceScene`, with no `confirmatory` flag, because no amendment opening them
exists and a flag that could open them would be a door left ajar. Blocks 720-735 stay separately
reserved and are *total*-recurrence partitions, which is exactly why a new reservation was
needed: the property under test is absent from them.

**What this does not repair.** Epoch-to-epoch recurrence. The blocks are disjoint seed ranges
over independent draws, so scenes within a block differ from one another exactly as scenes
across blocks do, and "across partitions" is not a distinct phenomenon in this generator.
Partial presence is fixed here; separated epochs remain owed by the acquired-record path, and
nothing built on these partitions may be reported as evidence about them.

**These partitions are development evidence permanently** from 2026-09-10. A criterion measured
on them produces development evidence, and no later renaming makes them fresh. Confirmatory
evidence for a partial-recurrence criterion needs 880-895, which have never been built.

**T4E.13 k profile (2026-09-10): completed, and it adjudicates nothing.**

`measurements/t4e13_k_profile.json`. Characterisation of how the criterion degrades as the
tolerance widens, declared as such before it ran.

```
T4E.13 k PROFILE -- development blocks, totals over the four blocks
k    rich    admitted     motif  false adm      null
6    6             60        60     0.0000         0
6    9             60        60     0.0000         0
6    12            60        60     0.0000         0
5    6             60        60     0.0000         0
5    9             80        60     0.2500         0
5    12           154        60     0.6104         0
4    6           129        60     0.5349         6
4    9           374        60     0.8396        87
4    12          834        60     0.9281       132
3    6           336        60     0.8214        72
3    9          1196        60     0.9498       314
3    12         3293        60     0.9818       778
```

**The motif column never moves.** 60 pairs at every rung and every richness -- four blocks of
15 -- which is the monotonicity the T4E.13 declaration derived before any of this ran:
admissions are non-decreasing as `k` falls, so recall is inherited and is not a finding at any
rung. Everything informative is in the other three columns.

**The null breaks between `k = 5` and `k = 4`.** At `k = 6` and `k = 5` it admits nothing at any
richness. At `k = 4` it admits 6, 87 and 132 pairs, and at `k = 3`, 72, 314 and 778 -- where
nothing recurs at all. So the one-scene-wide margin is not merely where specificity against
*structured* coincidence runs out; just below it the criterion begins manufacturing identity out
of noise, which is a different and worse failure.

**False admission climbs monotonically** from 0.0000 to 0.9818, and at `k = 3`, richness 12,
3,293 pairs are admitted of which 60 are true. Nothing below `k = S` is recoverable by widening
further, which the same monotonicity already guaranteed.

**This adjudicates nothing.** R20 forbids the horse race and the T4E.13 declaration forbids it
by name: the criterion under evaluation was `k = S - 1` and it is falsified on its own
conditions. A `k` made attractive by this sweep would be a further candidate needing its own
declaration and its own evidence, and it could not claim its structural choice was blind,
because this profile has now been seen. That consequence was recorded in advance of running it.

**T4E.13 candidate 3 (2026-09-09): adopted and FALSIFIED.**

Adopted at sha256 `55631fa2df03cda7e1d62304abebc362582e722a4e7177e5fb14cd3a5858b99c` as a
development experiment; measured in `measurements/t4e13_candidate_3.json`.

```
CANDIDATE 3, k = 5 of 6 -- development blocks, adopted at sha256 55631fa2...
rich  block      configs   Cprops  admitted   motif   split    admit  shortfall
6     100-105         20      121        15      15  0.0000   0.0000       0.00
6     200-205         20      121        15      15  0.0000   0.0000       0.00
6     300-305         20      152        15      15  0.0000   0.0000       0.00
6     400-405         20      136        15      15  0.0000   0.0000       0.00
9     100-105         84      568        15      15  0.0000   0.0000       0.00
9     200-205         84      485        25      15  0.0000   0.4000       6.00
9     300-305         84      585        25      15  0.0000   0.4000       6.00
9     400-405         84      566        15      15  0.0000   0.0000       0.00
12    100-105        220     1446        25      15  0.0000   0.4000       6.00
12    200-205        220     1440        25      15  0.0000   0.4000       6.01
12    300-305        220     1590        75      15  0.0000   0.8000      36.05
12    400-405        220     1488        29      15  0.0000   0.4828       8.43

null: 0 admitted at richness 6, 9 and 12, of 116, 595 and 1486 proposed
```

**FALSIFIED on conditions 3 and 2.** False admission reaches 0.4000 at richness 9 and 0.8000
at richness 12, against a declared bound of 0.10 at every richness individually -- four to eight
times over -- and the shortfall widens with richness rather than staying non-increasing, which
the same condition forbids separately. Condition 2 fails as well: on block 300-305 the matched
fraction goes 0.0500, 0.0198, 0.0227, rising at the richest level instead of falling. Three
blocks fall as required; the condition is stated per block, not as an average, so one is enough.

**The declaration predicted the direction, before the run and in writing.** Coincidental groups
do reach `S - 1`, and the greedy probe under-counted them. What the measurement adds is the
size: at richness 12 on block 300-305, 75 pairs admitted where 15 are true.

**Condition 4 held, and it is worth recording precisely because the candidate failed.** Where
nothing recurs, tolerating one absence still admits **nothing** -- 0 of 1486 proposed pairs at
richness 12. The relaxation did not break the null; it broke the planted blocks, where
coincidences have real structure to be coincidental with.

**Condition 1 is arithmetic and is not a finding.** The 0.0000 false split everywhere was
derived before the run: admissions are non-decreasing as `k` falls, so candidate 3 inherits
candidate 2's recall by monotonicity. The declaration said this in advance so it could not be
reported as evidence afterwards, and it is not being.

**What the blindness claim bought.** `k`, its rule and every structural choice were committed at
sha256 `55631fa2...` before any measurement at any `k` below `S`, and nothing moved after the
numbers appeared. The claim is now spent -- it cannot be made again for these scenes -- and what
it purchased is that this is a real falsification rather than a criterion that failed to survive
its own tuning. That is the whole return on the slice, and it is a smaller return than a pass
would have been.

**The margin is exactly one scene wide.** Consistency across the whole partition excludes
coincidences; consistency across all but one scene does not. By the same monotonicity the
profile's remaining rungs cannot rescue anything -- `k = 4` and `k = 3` admit supersets of
`k = 5` -- so the profile characterises how fast it degrades and adjudicates nothing.

**Blocks 720-735 stay closed and now will not be spent on this.** The adoption made spending
them conditional on candidate 3 meeting its conditions, and it did not. That conditionality was
recorded before the numbers existed, which is the only reason it is worth anything now.

**What this does NOT license.** Not that the signature is inadequate -- it is unchanged from
candidate 2, which passed; what changed is one integer. Not that partition structure is the
wrong resource -- it is the resource candidate 2 used successfully. Not that partial recurrence
cannot be detected: this tests one relaxation, the minimal one, and a criterion built *for*
partial recurrence is a different object from one that merely tolerates it. And specifically not
that `k = S` is the right operating point for an acquired record -- candidate 2's own limitation
stands untouched, so what this establishes is that the obvious repair does not work, not that
the problem has gone away. Raising `k` back towards `S` is forbidden by the declaration and this
outcome does not unlock it. No mining radius, no discharge of T4E.8's acceptance, no closure of
D96 to D100, and nothing about `kind_recurrence`, which still has no catalogue.

**The declaration below is retained unedited.** It is superseded by this outcome, not corrected
by it: its prediction was right, its derivation of condition 1 was right, and its ceiling on
what a failure licenses was written before the failure happened.

**T4E.13 candidate 3 (2026-09-09): declared, fixed in code, and measured nowhere.**

`data/identity_calibration/t4e13-partial-recurrence-declaration.json`, sha256
`55631fa2df03cda7e1d62304abebc362582e722a4e7177e5fb14cd3a5858b99c`, committed at `9b40b65`;
status `declared_before_measurement`. Candidate 2's criterion at `k = S - 1` by the rule *tolerate
exactly one absence*.

**What is being claimed, exactly.** That the value of `k`, the rule fixing it, and every
structural choice around it were committed before any measurement at any `k` below the partition
size, on any block, planted or null. Candidate 2 could not claim that -- its `k = S` followed a
feasibility probe -- and that gap is the one thing its confirmatory pass could not close.

**What was already known, disclosed rather than hidden.** Candidate 2's exact run shows no
coincidental group reaching 6 anywhere; the greedy probe found some reaching 5 at richness 9.
So `k = S - 1` sits **at** the observed coincidental ceiling. That points away from tuning
rather than towards it: the safe choice was `k = S`, which is already known to work, and this is
the value most likely to fail. What remains unknown is how many coincidental groups reach 5,
which is the only quantity the error rate depends on.

**Two results are arithmetic, not evidence, and are derived before the fact.** Admissions are
non-decreasing as `k` falls, so candidate 3 admits a superset of candidate 2 on every partition.
Candidate 2's false split is 0.0000 everywhere, so candidate 3's is 0.0000 everywhere **before
it is run** -- acceptance condition 1 is passed by arithmetic and carries no evidential weight.
The same monotonicity says the criterion cannot be inert the way candidate 1 was, which admitted
nothing anywhere and was therefore never tested. The entire empirical content is on the
admission side, and the declaration says so in those words rather than discovering it later.

**The relaxation is small and its transfer value is small with it.** On a long record `S` is
large and `S - 1` is nearly as strict as `S`, so tolerating one absence does not solve the
transfer problem candidate 2 named -- it is one rung on that ladder. A criterion that transfers
will need `k` as a proportion of `S`, and the proportion will need its own evidence.

**What is verified here is an absence.** Eight tests hold the structure: `k` derived as a
function of the partition size rather than the literal 5, unequal partitions refused rather than
pooled, monotonicity in `k` checked on a toy metric rather than assumed, the declaration's
disclosure and its arithmetic-not-evidence clause asserted against the file, the `k` profile's
adjudicates-nothing status asserted against the code, and -- the one that matters --
**no candidate 3 measurement, adoption record or measuring call exists anywhere in the
repository**, asserted rather than intended. A single run at any `k` below `S` before adoption
would spend the only property this candidate has, silently.

**Nothing is licensed by any of it.** No result exists. Partitions 720-735 have never been
generated and are refused in code under `confirmatory=True`; this declaration does not open them
and adopting it will not. No mining radius, no discharge of T4E.8's acceptance, no closure of
D96 to D100, and nothing about `kind_recurrence`, which still has no catalogue.

**T4E.12 candidate 2, CONFIRMATORY (2026-09-09): reproduces on partitions 700-715.**

Authorised by `t4e12-confirmatory-amendment.json`, which opened two of the four reserved
partitions and kept the other two. Partitions 700-705 and 710-715 were built for the first and
only time for this run and are inspected data permanently.

```
CONFIRMATORY -- partitions built for the first and only time
rich  block      configs   C props  admitted    motif    split     admit
6     700-705         20       148        15       15   0.0000   0.0000
6     710-715         20       122        15       15   0.0000   0.0000
9     700-705         84       546        15       15   0.0000   0.0000
9     710-715         84       530        15       15   0.0000   0.0000
12    700-705        220      1518        15       15   0.0000   0.0000
12    710-715        219      1376        15       15   0.0000   0.0000

matched fraction 1/m at every level, including 1/219 where one configuration
was refused as incomparable. Two such refusals in total.
```

**It reproduces exactly.** Every partition at every richness admits 15 pairs -- C(6,2), the
motif's complete clique -- and nothing else, out of 122 to 1518 pairs candidate C proposes. Both
error rates are 0.0000 throughout. So candidate 2's development result was not an artefact of
the four blocks it was developed on.

**The reserved evidence was split rather than spent.** The maintainer opened two of the four
partitions and kept 720-725 and 730-735, which have never been generated and are refused **in
code** even under `confirmatory=True` -- `STILL_RESERVED_CONFIRMATORY_SEEDS`, enforced rather
than intended, because intention has already been shown insufficient in this programme.

**What a confirmatory pass here still cannot establish.** Not that a blind structural choice
would have produced it: `k = S` was chosen after a probe showed coincidental groups topping out
at 5 on the development blocks, and fresh scenes cannot retrospectively make that choice blind.
Not anything about rejection where nothing recurs -- **no confirmatory null was declared**, so
the null evidence remains development only. Not anything about partial recurrence, since `k = S`
rejects a configuration absent from one scene outright. Not anything about recurrence *across*
partitions, which the mining machinery needs. And no mining radius, no discharge of T4E.8's
acceptance, no closure of D96 to D100, and nothing about `kind_recurrence`, which still has no
catalogue.

**The maintainer's framing was fixed before these numbers existed and is unchanged by them**:
candidate 2 is a strong diagnostic of whether the signature can sustain coherent identity, not a
defensible real-world recurrence rule. What has been strengthened is the diagnostic; what has
not changed is that it is one.

**T4E.12 candidate 2 (2026-09-09): all four acceptance conditions met on development
evidence.**

Declared in `t4e12-consistency-declaration.json` (sha256 `639485af...`), committed at that hash
before any development block was evaluated, and adopted in
`t4e12-consistency-adoption.json` as a **development experiment only**, with the reserved
confirmatory blocks explicitly withheld by the maintainer.

```
rich  block      configs   C props  admitted    split    admit  shortfall
6     100-105         20       121        15   0.0000   0.0000      x0.00
6     200-205         20       121        15   0.0000   0.0000      x0.00
6     300-305         20       152        15   0.0000   0.0000      x0.00
6     400-405         20       136        15   0.0000   0.0000      x0.00
9     (four blocks)   84   485-585        15   0.0000   0.0000      x0.00
12    (four blocks)  220 1440-1590        15   0.0000   0.0000      x0.00

NULL rich=6    C proposed 116     ADMITTED 0
NULL rich=9    C proposed 595     ADMITTED 0
NULL rich=12   C proposed 1486    ADMITTED 0

matched fraction by richness: 0.05000, 0.01190, 0.00455  (exactly 1/m)
incomparable configurations refused: 0, 0, 4
```

**All four acceptance conditions met, on development evidence.** Every block at every
richness admits exactly 15 pairs -- C(6,2), the motif's complete clique -- and nothing else.

**Which half is informative.** The recall half was very nearly guaranteed: the scenes are built
with the motif in every scene, this criterion admits configurations present in every scene, and
candidate C had already recovered every motif pair in every block, so a partition-spanning group
exists **by construction**. **The rejection half is the finding** -- zero false admissions, and
a null admitting none of 116, 595 or 1486 proposed pairs, where candidate C returned 116 and
candidate D returned 62.

**The maintainer's ceiling, fixed before the numbers were known.** *"Candidate 2 looks like a
strong diagnostic of whether the signature can sustain coherent identity. It is not yet a
defensible real-world recurrence rule."* This is evidence that the signature **can** sustain
coherent identity across a partition, and is not reported as a recurrence criterion.

**The confirmatory blocks were withheld by decision and remain clean**, because `k` was
influenced by the feasibility probe on these same blocks. That is stricter than the declaration
asked. **No confirmatory evidence for this candidate exists or will exist under this adoption.**

**What it does not establish**: nothing about partial recurrence, since `k = S` rejects a
configuration absent from one scene outright; nothing about recurrence *across* partitions,
which the mining machinery needs; no mining radius, no discharge of T4E.8's acceptance, no
closure of D96 to D100; and nothing about `kind_recurrence`, which still has no catalogue.

**T4E.12 candidate 2 (2026-09-09): declared and implemented; NOT MEASURED.** *SUPERSEDED the
same day by the entry above, which records the adoption and the measurement. Kept unedited as
the record of the state the declaration was in before it was adopted; its statements about
adoption and measurement are no longer true.*

The criterion is `mutual_nearest_neighbour_consistency_across_the_partition`: a set of
configurations, at most one per scene, is consistent when every member is the mutual nearest
neighbour of every other, and only pairs inside sets spanning at least `k` scenes are admitted,
with `k` the partition size. Declared in `t4e12-consistency-declaration.json` (sha256
`639485af...`). The declaration is **not adopted**, so no development block has been evaluated
under it and no error rate for it exists in this document or anywhere else.

**A feasibility probe was run before the declaration was written and is disclosed in it.** This
is the standing lesson from candidate 1, which failed two conditions derivable in advance.

```
clique sizes reached by mutual-nearest-neighbour consistency, six-scene partitions
                        motif          non-motif spread
100-105 rich 6     6,6,6,6,6,6    1:21  2:53  3:32  4:8
100-105 rich 9     6,6,6,6,6,6    1:55  2:249 3:150 4:44
300-305 rich 6     6,6,6,6,6,6    1:7   2:51  3:41  4:15
300-305 rich 9     6,6,6,6,6,6    1:56  2:245 3:141 4:51 5:5
NULL    rich 6              --    1:22  2:51  3:46  4:1
NULL    rich 9              --    1:47  2:260 3:156 4:41
```

The motif reached a clique of 6 in every scene of every block; no non-motif configuration
exceeded 5; the null reached 4 at most. That establishes the criterion is not inert. It does not
establish error rates, and the probe used a greedy clique walk where the criterion uses an exact
enumeration, so its group sizes are a lower bound rather than the criterion's own output. The
probe also informed the choice of `k`, which the declaration states plainly.

What has been verified is mechanics, on constructed partner graphs: a fully agreeing set is
found whole; a set that does not close is cut back to what agrees; the largest agreeing subset
is found exactly where a greedy walk could return a smaller one; a group may not hold two
configurations from the same scene; only pairs inside groups spanning enough scenes are
admitted, and demanding more scenes than the partition holds admits nothing rather than
erroring; and a group of fewer than two scenes is refused as meaningless.

The reserved confirmatory blocks were not built. No mining radius is approved and no defect is
closed.

**T4E.12 candidate 1 (2026-09-09): it admits nothing, and it did not test the
signature.**

Declared in `t4e12-multiplicity-declaration.json` (sha256 `f0c45d92...`), committed at that hash
before any development block was evaluated, and adopted separately in
`t4e12-multiplicity-adoption.json`.

```
rich   block      pairs refused  C proposed  admitted   split
6      100-105       15      15         121         0     n/a
6      200-205       15      15         121         0     n/a
6      300-305       15      15         152         0     n/a
6      400-405       15      15         136         0     n/a
9      100-105       15       0         568         0  1.0000
9      200-205       15       0         485         0  1.0000
9      300-305       15       0         585         0  1.0000
9      400-405       15       0         566         0  1.0000
12     100-105       15       0        1446         0  1.0000
12     200-205       15       0        1440         0  1.0000
12     300-305       15       0        1590         0  1.0000
12     400-405       15       0        1488         0  1.0000

NULL rich=9    admitted 0, per scene pair 0.0000   (alpha = 0.05 expected)
NULL rich=12   admitted 0, per scene pair 0.0000   (alpha = 0.05 expected)

motif tail probability against the bound, sampled in four scene pairs
  100-105 rich 9    6.47e-5 vs 7.09e-6    x9.1
  100-105 rich 12   6.22e-6 vs 1.03e-6    x6.0
  300-305 rich 9    2.50e-5 vs 7.09e-6    x3.5
  300-305 rich 12   1.59e-5 vs 1.04e-6    x15.3
```

**Falsified on acceptance condition 1**: where the model could be fitted, nothing was admitted
and the false split is 1.0 against an accepted 0.10.

**Richness 6 was never evaluable**, and that was derivable before adoption: 50 exceedances at a
1st-percentile threshold needs at least 5,000 distances, hence at least 71 configurations, hence
at least nine features. Six features give 400 and 4. Refused by name at every scene pair.

**The null carries the important reading, and it was fixed in advance.** If the tail model held,
admissions per scene pair would equal `alpha` whatever the signature is like. Nominal 0.05,
measured 0.0000: the fitted model under-admits relative to its own nominal rate, so the failure
is the model's and **the signature was not cleanly tested**.

**A gap in the declaration's diagnostic**: it named "far above alpha" as indicting the model and
never named "far below", which is the direction that occurred.

**How short, and what is not claimed**: the motif's tail probability sits 9.1x, 6.0x, 3.5x and
15.3x above the bound in four sampled scene pairs. No trend in richness is claimed -- one block
improves and the other worsens.

**A condition found and named while measuring**: at twelve features a few configurations are so
nearly isotropic that their principal axis is refused, so they carry no bearing block and the
metric will not compare them with anything that does. None at six or nine features; 1, 1 and 2
of 1,320 in three of four blocks at twelve. `comparable_subset` refuses them by name and counts
them. This also qualifies the T4E.12 scaling entry below, which ran on block 100-105 alone and
would have failed on three of the four blocks at twelve features.

The reserved confirmatory blocks were not built. No mining radius is approved and no defect is
closed.

**T4E.12 candidate 1 (2026-09-09): declared and implemented; NOT MEASURED.** *SUPERSEDED
the same day by the entry above, which records the adoption and the measurement. Kept unedited
because it is the record of the state the declaration was in before it was adopted; every
statement below was true when written and the ones about adoption and measurement are no
longer true now.*

The criterion is `multiplicity_aware_tail_admission` -- candidate C's pairs, admitted only when
a distance as small as their own would arise with probability at most `alpha / m^2` under a tail
model fitted to that scene pair's own distances. Declared in
`t4e12-multiplicity-declaration.json` (sha256 `f0c45d92...`). The declaration is **not adopted**,
so no development block has been evaluated under it and no error rate for it exists in this
document or anywhere else.

The T4E.12 acceptance clause was **amended in the open** before this candidate was declared. As
first written it required a candidate to be "a change to what the signature carries"; the
scaling measurement shows what is indicted is a decision rule that does not know how many
comparisons it is making, and multiplicity is a property of the procedure. The four numbered
acceptance conditions are unchanged; only the admissible class of candidate is widened.

What has been verified is mechanics, on constructed scenes and a synthetic gamma sample:

- the admission bound is `alpha / m^2` and falls by more than two orders of magnitude between
  20 and 220 configurations, by construction rather than by calibration;
- the admitted set is unchanged when every distance is rescaled, so the criterion is scale-free
  as candidates C and D were;
- the tail probability is monotone in the distance and equals the exceedance rate at the
  threshold itself;
- a thin population, too few exceedances, or an all-NaN sample raises `TailModelRefused` rather
  than admitting on a fit that does not exist;
- the criterion admits no pair candidate C did not propose, and an empty scene returns no model
  rather than a fitted one;
- and the derivation behind not declaring the obvious candidate is recorded as a test: a
  nearest-neighbour matching returns at most one pair per configuration whatever the weights
  are, so reweighting cannot move the matched fraction that acceptance condition 2 is about.

The reserved confirmatory blocks were not built. No mining radius is approved and no defect is
closed.

**T4E.12 scaling diagnostic (2026-09-09): recall survives richness; the rule's match
count does not.**

A diagnostic, not a criterion. `measure_richness_scaling` varies the feature count, which the
generator now accepts as a parameter defaulting to the frozen six, so every existing caller
builds exactly the scenes it built before. Candidate D is a probe of the signature and is
already falsified; nothing is adopted and no operating point is reported. Six scenes per
richness, 15 scene pairs.

| features | configs | candidate pairs | matches | split | admission | pair rate | required | shortfall |
|---|---|---|---|---|---|---|---|---|
| 6 | 20 | 6,000 | 70 | **0.0000** | 0.7857 | 9.19e-3 | 2.79e-4 | **x33** |
| 9 | 84 | 105,840 | 321 | **0.0000** | 0.9533 | 2.89e-3 | 1.58e-5 | **x184** |
| 12 | 220 | 726,000 | 720 | **0.0000** | 0.9792 | 9.71e-4 | 2.30e-6 | **x423** |

**Recall is untouched by richness.** False split is 0.0000 at every level: the planted motif is
still the mutual nearest neighbour when competing against 219 rivals rather than 19.

**The rule matches a constant fraction of what it is given** -- 0.233, 0.255, 0.218 of
configurations -- because a nearest-neighbour matching returns at most one pair per
configuration. False admissions grow linearly with the configuration count while true
correspondences stay at one per scene pair, so the admission fraction climbs to 0.979.

**The per-pair rate falls as 1/m where a fixed admission bound demands 1/m^2,** and the
shortfall widens from x33 to x423. `required_pair_rate` is the arithmetic behind the last two
columns and approves nothing.

The reserved confirmatory blocks were not built. No mining radius is approved and no defect is
closed.

**T4E.11 diagnostic (2026-09-09): what the false admissions are. The ground truth is
sound and the constraint is specificity.**

A diagnostic, not a criterion: `decompose_matched_pairs` adopts nothing, chooses no threshold
and reports no operating point. Run after four candidates had been falsified against a ground
truth no declaration had examined. Pooled over four planted blocks, 60 scene pairs, 20
configurations per scene.

| class | available | C matched | rate | D matched | rate |
|---|---|---|---|---|---|
| **motif** | 60 | 60 | **100%** | 60 | **100%** |
| shared_2 | 1620 | 89 | 5.49% | 54 | 3.33% |
| shared_1 | 1620 | 36 | 2.22% | 17 | 1.05% |
| **crossed_2** | 1080 | **0** | **0.000%** | **0** | **0.000%** |
| crossed_1 | 10800 | 170 | 1.57% | 80 | 0.74% |
| unrelated | 8820 | 175 | 1.98% | 93 | 1.05% |
| *null, unrelated* | 6000 | 116 | 1.93% | 62 | **1.03%** |

`shared_n` pairs hold the SAME motif vertices -- the same physical features replanted -- and
`crossed_n` pairs hold motif vertices that are not the same ones. Separating those two is the
point; a count of shared features cannot.

**The hypothesis fails.** Only 71 of candidate D's 244 non-motif matches (29%) hold the same
motif vertices, and `shared_2` is enriched just 3.2x over unrelated. The ground truth is not
scoring real recurrence as error to any material degree.

**Three positive findings.** 60 of 60 motif pairs recovered by both criteria. The background
rate does not notice whether a motif is present (1.054% planted against 1.033% null), so the
null's retention was never a null-specific artefact. `crossed_2` is 0 of 1080: the full motif
is never matched to a configuration holding two of its own three features.

**The binding constraint, as a number.** 400 candidate pairs per scene pair against one true
positive -- prior odds 1:399. Candidate D's per-pair false rate is 244/23,940 = 1.019%,
specificity 98.98%. The declared 0.10 admission bound requires 0.028%: a 36-fold reduction. No
threshold on these distances supplies it, which is what opens T4E.12.

The reserved confirmatory blocks were not built. No mining radius is approved and no defect is
closed.

**T4E.11 candidate D (2026-09-09): the margin costs no recall, rejects too little, and
the distributions overlap.**

Declared in `t4e11-ratio-margin-declaration.json` (sha256 `b2ba2b4e...`), committed at that hash
before any development block was evaluated, and adopted separately in
`t4e11-ratio-margin-adoption.json`. `tau = 0.8` comes from Lowe (2004), IJCV 60(2) -- taken for
its external provenance, explicitly not because it was expected to be optimal here.

```
block           pairs  matches  C match  motif  found    split    admit
100-105            15       70      121     15     15   0.0000   0.7857
200-205            15       67      121     15     15   0.0000   0.7761
300-305            15       86      152     15     15   0.0000   0.8256
400-405            15       81      136     15     15   0.0000   0.8148
NULL 500-505       15       62      116      0      0      n/a   1.0000

false split range   : [0.0, 0.0]
null retention      : 0.5345   (accepted: at most 0.10)
refusals            : 0

diagnostic, NOT an operating point
  motif ratios, four blocks : 0.020 to 0.293
  null block ratios         : 0.240 to 0.996
```

**The margin costs nothing in recall.** Split stayed at 0.0000 on every block and match counts
fell on every block, so both directions of the structural prediction made before measurement
held exactly: what the margin discarded was entirely non-motif. The one condition it was
declared against is the one that fails -- it removed 54 of candidate C's 116 null matches and
kept 62 where the correct answer is none.

**The declared diagnostic answers the question the declaration posed.** Not "was `tau` badly
placed" but "do the distributions overlap": **they overlap**, 0.240 against 0.293. The contrast
is genuinely informative and 0.8 is far too permissive for this signature, but no threshold
separates recurrence from coincidence cleanly. The overlap is narrow, which is what makes it
dangerous; a `tau` chosen to sit inside it would be fitted to blocks now inspected four times
over, and the declaration forbids reporting such a value as an operating point.

**Two limitations, recorded as limitations rather than results.** Retention was measured against
the single declared null partition, so its stability across nulls is untested. The diagnostic
stores extremes rather than distributions, so the mass of the overlap is unknown from this run.

**The declaration diagnosed its own failure correctly, which the two before it did not.** Its
falsification field was narrowed on purpose after candidates B and C both over-reached, and
licenses only that at `tau = 0.8` this contrast does not separate in these scenes -- exactly
and only what the measurement supports.

The reserved confirmatory blocks were not built. No mining radius is approved -- there is still
no radius -- and no defect is closed.

**T4E.11 candidate C (2026-09-09): the ordering transfers; the rule cannot decline.**

Declared in `t4e11-mutual-nearest-neighbour-declaration.json` before measurement and adopted
separately. Two corrections were made to the declaration *before* adoption, after external
review the maintainer sought: the one-to-one assumption was restated at the instance level,
because the draft wording had misdescribed the criterion as permitting one match per scene pair
when mutual nearest-neighbour returns a partial matching that can hold many; and the scope was
made explicit as a domain-agnostic mechanism carrying atmospheric evidence only. The superseded
draft (sha256 `836812a0...`) was never adopted and nothing was measured under it; the adopted
declaration is `6b7105cc...`.

```
block         pairs  matches  motif  found    split    admit
100-105          15      121     15     15   0.0000   0.8760
200-205          15      121     15     15   0.0000   0.8760
300-305          15      152     15     15   0.0000   0.9013
400-405          15      136     15     15   0.0000   0.8897
NULL 500-505     15      116      0      0      n/a   1.0000

split range     : [0.0, 0.0]
admission range : [0.8760, 0.9013]
```

**The recall half is exactly what the task wanted, and nothing before it managed.** Split is
0.0000 in every block: every construction-labelled motif pair is recovered, across blocks whose
distance magnitudes differ by 1.876x. A frozen radius could not do that (T4E.10) and a
normalised one could not either (candidate B). **The ordering transfers where the magnitudes do
not**, which is a real finding about this signature and survives the candidate's failure.

**The rejection half fails completely.** Mutual nearest-neighbour always returns a match, so it
matches whatever is mutually nearest whether or not anything recurs. The null block -- same
feature count, same family, nothing repeated -- yields **116 matches where the correct answer is
none**, a false-admission rate of 1.0000. The planted blocks return 120 to 150 matches of which
15 are motifs. The candidate is falsified in exactly the manner its declaration named in
advance.

**A correction to the declaration's escalation, recorded rather than edited away.** It said a
failure would mean the question is not answerable by a criterion of this shape on this
signature, and that the next move would be the signature rather than a fourth criterion. That
does not follow. The shape did not fail; the ordering transferred perfectly. What failed is
that the rule is incomplete, having no rejection test -- and a rejection test can itself be
rank-based and scale-free, comparing the nearest distance to the second-nearest within one
scene pair. Trying that is candidate D and needs its own declaration under R20; adding it here
and re-measuring is precisely the move the sequential discipline forbids.

**A pattern, recorded because it is about this programme's own method.** This is the second
consecutive declaration whose *what would falsify this* reasoning was wrong in the same
direction. Candidate B predicted failure would show the shape moving rather than the scale; it
did not. Candidate C predicted failure would implicate the signature; it does not. Both
acceptance conditions were correct and both results are unambiguous, so the declarations earned
their keep. But that field has now over-reached twice and should be read as a hypothesis about
the failure rather than a conclusion licensed in advance.

The reserved confirmatory blocks were not built. No mining radius is approved -- there is no
radius -- and no defect is closed.

**T4E.11 candidate B (2026-09-09): adopted before measurement, falsified by it.**

Declared in `t4e11-normalised-distance-declaration.json` (sha256 `ee9715ea...`) before anything
was measured, and adopted by the maintainer on 2026-09-09 in a separate record so the declared
artifact keeps its hash. Development evaluation only; the reserved confirmatory blocks were not
built.

```
block         normaliser   raw mean  norm mean   norm med
100-105         0.065395   0.004575     0.0700     0.0653
200-205         0.061142   0.006941     0.1135     0.1037
300-305         0.061610   0.008580     0.1393     0.1317
400-405         0.066280   0.008585     0.1295     0.1292

raw mean spread        : 1.876x
normalised mean spread : 1.991x
```

**The candidate is falsified.** Normalising did not collapse the spread it was adopted to
collapse; it left it marginally wider.

```
normaliser spread across blocks : 1.084x
same-config mean spread         : 1.876x
normaliser / same-config mean   : 14.29, 8.81, 7.18, 7.72
correlation(normaliser, same-config mean) over 4 blocks : -0.210
```

The normaliser barely moves while the quantity it was meant to track moves a lot, sits seven to
fourteen times above that quantity, and does not correlate with it. A median over every
configuration's nearest cross-scene neighbour is dominated by the unrelated majority -- 114 of
120 configurations per block are not the motif -- so it measures the nearest-**unrelated**
distance, a different regime from the same-configuration one the radius operates in.

**A correction to the declaration's own reasoning, recorded rather than edited away.** The
declaration said failure would show the *shape* was moving rather than the scale. That is not
established: the same-configuration scale still moves 1.876x, and this particular label-free
normaliser simply does not track it. It also described the normaliser as "taken from the
close-pair regime the radius operates in", which was wrong in exactly the way the measurement
exposed. Whether another label-free normaliser would track it is open and needs its own
declaration; trying variants until one works is what R20's sequential discipline exists to
prevent. The declaration file is byte-identical so it still verifies against the hash it was
adopted under, and the outcome and this correction live in
`t4e11-normalised-distance-adoption.json`.

**T4E.8 slice 5 (2026-09-09): the identity target decided, and the role recorded as data.**

The maintainer's declared position: **`kind_recurrence` is the primary scientific target;
`track_continuity` and `spatial_persistence` are diagnostics for it.** The machinery could not
express that -- a declaration carried one target and one evidence class, so a diagnostic
measurement and the target it informs were recorded identically. `DECLARATION_ROLES` and the
`role` / `diagnostic_for` arguments close that gap.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_target_declaration.py     src/tests/test_identity_api.py -q
50 passed

>>> declare_identity_target("spatial_persistence", "record_derived_proxy",
...                         role="diagnostic", diagnostic_for="kind_recurrence")
role                : diagnostic
diagnostic_for      : kind_recurrence
diagnostic_boundary : A result here is diagnostic. It does not license 'kind_recurrence',
                      whose own acceptance and admissible evidence are unchanged by anything
                      measured under 'spatial_persistence'.
```

The rule that matters is that a role changes what is claimed from a result and **never** what
evidence is admissible:

```
kind_recurrence + record_derived_proxy, role=diagnostic
  -> InvalidParameterError: ... validated against itself ...   (circularity is not launderable)
spatial_persistence + proxy, role=diagnostic, no diagnostic_for
  -> MissingParameterError: 'diagnostic_for'
spatial_persistence, diagnostic_for=spatial_persistence
  -> InvalidParameterError: a measurement is not a diagnostic for itself
kind_recurrence + external_reference, role=primary, diagnostic_for=spatial_persistence
  -> InvalidParameterError: naming both leaves it unclear which one a result is about
```

**The cost of the decision, recorded with it.** `kind_recurrence` admits only
`external_reference`, so the primary target requires a reviewed, cited catalogue and a
serialisation the audit tool can load, and neither exists;
`audit_spatial_identity.py` refuses every other evidence class by name. The primary target is
therefore **unevaluable from the tool today**, and the catalogue moves onto the critical path,
where it was already a T4F.9 prerequisite. The diagnostics stay measurable and are worth
measuring, and a result under either is not progress toward the target.

**T4E.10 (2026-09-09): no operating point transfers, and the reason is not the estimator.**

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_operating_point.py -q
21 passed   # 15 test functions, 21 parametrised cases
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_certification.py -q
18 passed in 402.18s
```

**The support requirement, in closed form.** For the k-th smallest of n samples the covered
proportion is `Beta(k, n-k+1)`, so a bound of coverage p at confidence gamma exists only where
some k satisfies `P(Beta(k, n-k+1) >= p) >= gamma`. The maximum gives `1 - p**n >= gamma`, so
`n >= log(1-gamma)/log(p)`:

```
coverage  confidence  required n
0.90      0.90        22
0.95      0.95        59
0.90      0.50         7
0.99      0.90       230
```

T4E.9 calibrated on **15** pairs at 90/90. The guarantee was unavailable before any radius was
computed, and `nonparametric_tolerance_bound` refuses there rather than issuing one:

```
support of 15 cannot carry coverage 0.900 at confidence 0.900; 22 observations are required
and no order statistic of 15 supplies it. A radius issued here would carry no guarantee at all.
```

**Every estimator, calibrated then frozen, applied to partitions it never saw.**

```
below_requirement: 15 calibration pairs (required 22)
  empirical_quantile             DOES_NOT_HOLD  r=0.006563  split 0.4667  admission 0.0000
  bootstrap_upper                DOES_NOT_HOLD  r=0.007706  split 0.2667  admission 0.0000
  nonparametric_tolerance_bound  REFUSED

above_requirement: 28 calibration pairs (required 22)
  empirical_quantile             DOES_NOT_HOLD  r=0.006563  split 0.4667  admission 0.0000
  bootstrap_upper                DOES_NOT_HOLD  r=0.006835  split 0.4000  admission 0.0000
  nonparametric_tolerance_bound  DOES_NOT_HOLD  r=0.008468  split 0.2000  admission 0.0000
```

The tolerance bound is much the best of the three -- 20.0% against the quantile's 46.7% -- and
still fails the 10% bound. Admissions are **zero** everywhere, so the failure is entirely on the
split side.

**Why sufficient support does not rescue it.** Four blocks of six scenes, one generator,
identical parameters:

```
block 100-105  pairs 15  mean 0.004575  median 0.004268  max 0.008468
block 200-205  pairs 15  mean 0.006941  median 0.006341  max 0.010923
block 300-305  pairs 15  mean 0.008580  median 0.008115  max 0.012859
block 400-405  pairs 15  mean 0.008585  median 0.008562  max 0.016097

block mean min/max 0.004575 / 0.008585   ratio 1.88   sd 0.001895
```

A tolerance bound is distribution-free but not assumption-free: it covers the population its
sample was drawn from. These blocks are not one population. A radius calibrated on the first
therefore carries no guarantee about the second **at any support** -- and T4E.9's calibration
block is the tightest of the four, which is exactly why its frozen radius was too small
everywhere else.

**What this settles and what it does not.** It settles that the live half of D97 is not "find a
better estimator": a frozen absolute radius is the wrong object when disjoint partitions of one
generator differ by 1.88x, and disjoint windows of a real atmospheric record will differ more,
not less. It reaches D96 too, because that defect's workload is set by the radius and a radius
that cannot be frozen is a workload that cannot be predicted. It approves no mining radius,
closes no defect, and is measured on synthetic scenes, so it is a property of the estimator and
the partitions rather than of the atmosphere. A REFUSED outcome above is correct behaviour and
is recorded as such: scoring it as a failure would reward the estimator that answers anyway.

**T4E.9 (2026-09-08): the T4E identity path against an answer known by construction.**

The first measurement of `spectral_constellation` -> `spectral_invariance` ->
`spectral_clustering` against certified ground truth. `src/benchmarks/` had never touched that
path (D101). Six planted scenes calibrate a radius and freeze it; six more planted scenes and
six null scenes -- same feature count, same family, nothing recurring -- evaluate it. Six
features choose three, so each scene yields exactly 20 configurations of which exactly one is
the motif. Only cross-scene pairs are formed.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_identity_certification.py -q
14 passed in 171.63s

frozen radius            : 0.006563   (90% recall on 15 calibration motif pairs, then frozen)
planted evaluation       : motif pairs 15, other pairs 5985
                           AUC 1.0000  split 0.4667  admission 0.0000
null evaluation          : motif pairs 0 (unmeasured, not zero), other pairs 6000
                           admission 0.0000
feasible radius exists   : True, at 0.010801 -> split 0.0667, admission 0.0000
OUTCOME                  : FAIL
```

**The result separates into two halves and only one of them fails.** The identity definition
discriminates perfectly: AUC 1.0, and **zero admissions across 11,985 different-configuration
pairs** spanning both the planted evaluation and the null partition. That is the question D101
asked -- can the T4E path recover an answer known by construction -- and the answer is yes.

What fails is the radius. Calibrated at 90% recall on 15 cross-scene motif pairs and frozen,
it splits 46.7% of motif pairs in a partition it had never seen, against a declared 10% bound.
A descriptive amendment added after that first run -- changing no window, threshold, weight or
acceptance criterion, in the same manner as T4E.8 slice 2's -- reports that a feasible radius
does exist, at 0.010801, giving split 6.7% and admission 0%.

The suite, measured after the benchmark was registered:

```
$ .venv/Scripts/python.exe -m src.benchmarks
PASS 43   FAIL 1   NOT_YET_RUNNABLE 0
FAILED: one or more benchmarks did not reproduce their known answer.
```

`summarise(run_all())` agrees at 43/1/0. The suite is red, and it is meant to be: the FAIL is
`4E.identity_certified` reporting a measured result, not a broken build. Recording it as
`NOT_YET_RUNNABLE` to keep the column green would be hiding a finding behind a status.

Two staleness corrections found while recording this. `architecture.md` section 7.1 had claimed
**29 PASS** for this suite; the true figure before T4E.9 was 43 PASS, 0 FAIL.
`test_benchmark_status_in_docs_matches_a_real_run` uses `re.search`, so it checks only the
first such triple in the file and had never guarded that second copy. Both are now corrected
and the gap in the guard is recorded beside the row it let drift.

**Why this matters for D97.** The failure is measured with perfect labels, total separation and
no atmosphere at all. So `calibrate at a recall quantile, then freeze` does not transfer at this
support even when the answer is certain, which is a property of the procedure rather than of the
record. D97's second limb is now isolated from every atmospheric confound. The benchmark reports
FAIL rather than silence, D101 is closed by the measurement existing, and no mining radius is
approved by any of it.

**Mutual k-NN alignment and its missing reference (2026-09-08, exploratory, not a phase task).**

Apparatus written while auditing the Platonic Representation Hypothesis (arXiv:2405.07987v5),
which compares two representations by reducing each to a kernel and measuring the mean
intersection of the k-NN sets they induce. Nothing here is trained, downloaded or evaluated;
the acceptance is entirely synthetic and no real-model measurement has been run.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_representation_alignment.py -q
25 passed   # 19 test functions, 25 parametrised cases
```

**What the paper's own text contains.** Read in full -- v5 PDF, 27 pages, 110,001 characters,
case-insensitive over the text layer, which includes figure captions but not text baked into
plot images:

```
baseline      0     shuffl*       0     chance        0     null          0
surrogate     0     untrained     0     significan*   0     confidence    0
error bar     1     (Figure 2 only)
permut*       3     (all weight-space symmetry and citations; no permutation test)
control       3     (all three in the bibliography)
random*       7     (includes 1 randomly initialised ResNet-50 among the 78 vision models,
                     plotted as a category in the Figure 2 UMAP)
```

**The floor, computed from the settings the paper does state.** Appendix C declares k = 10
nearest neighbours over 1000 Places-365 images for vision-vision, and k = 10 over 1024 WIT
samples for cross-modal. Two independent uniform k-subsets of the other n-1 points intersect
in k^2/(n-1) on average, so the metric's chance level is k/(n-1):

```
cross-modal (WIT)           n=1024  k=10   chance = 0.00978   measured 0.16 = 16.4x chance
vision-vision (Places-365)  n=1000  k=10   chance = 0.01001
```

Section 6 of the paper asks whether 0.16 "is indicative of strong alignment with the remaining
gap being 'noise' or does it signify poor alignment with major differences left to explain."
Against the uniform floor the answer is available in closed form and **supports the paper**:
the measured value is about sixteen times chance. This is recorded as a result, not as a
criticism withdrawn -- the floor was never stated, and a number whose reference is absent
cannot be read even when the reading turns out to be favourable.

**A wrong claim of this programme's own, corrected in place.** The first version of the
acceptance test asserted that strong kernel structure would lift the permutation null well
above k/(n-1). It does not: under a uniform permutation the right-hand neighbour sets land
uniformly whatever structure they carry. Measured over 120-200 permutations:

```
structure     null_mean   analytic k/(n-1)   ratio
clustered     0.05622     0.05587            1.006
gaussian      0.05567     0.05587            0.997
duplicated    0.05616     0.05587            1.005
```

The closed form is therefore a reliable estimate of the null *mean*, which makes it more useful
than first claimed. It supplies no spread, so it is still not a threshold, and it cannot see
pairing structure that is real but not semantic. WIT is drawn from Wikipedia, where images and
captions from one article share a subject; two unrelated representations of such a sample would
align above the uniform floor. Only a permuted pairing removes that residue, and that
measurement has not been run.

Extracted from the final receipt (printed by the verification extraction command):

```
window       signatures repeats legacy_AUC spatial_AUC fixed_split fixed_admission admission_at_all_repeat_r90
   0:480     69580    6838  0.7866  0.8877  0.293946  0.104800  0.332800
 480:960     51395    3652  0.8288  0.8947  0.288061  0.093600  0.319500
 960:1440    52765    5607  0.8043  0.8952  0.256287  0.108950  0.305100
```

The frozen stationary radius is `0.13807521070069662`. In the calibration window the stationary
population grows from 124 band-stable pairs to 188 pairs (169 distinct contributing track keys),
with 19 observations in its upper decile. Those are dependent pairs; no iid confidence interval
or effective-sample-size claim is made. Stationary band-flipped pairs in that window have AUC
0.9471 under spatial geometry, against 0.9489 for band-stable pairs. This removes the earlier
large band penalty at fixed/near-stationary positions, not the positional effects of detector
flanks or physical evolution.

The final diagnostic finds **no radius meeting both empirical 10% bounds on the all-repeat
proxy populations in any window**. At the smallest radius admitting at least 90% of repeated-key
pairs, the different-key admission is 33.28%, 31.95% and 30.51%. This follows from monotonicity
of admission in radius; it is not evidence that all possible identity definitions or independent
physical labels would fail. `approved_mining_radius` is `null`. T4E.8 acceptance, D97 and the
pilot remain unresolved. The original signatures/declaration still carry D99/D100; the opt-in
mode removes those comparable attributes but is not an accepted pipeline replacement.

### Analytical and mutation evidence

The early focused run found one backward-compatible error-contract regression (the unknown-mode
exception changed type); restoring `InvalidParameterError` exposed the legacy test's required
explanatory wording, which was restored too. These failures were corrected rather than the
legacy test being weakened. A subsequent broad focused attempt recorded 133 passing cases and
that one wording failure before the final corrected regression below.

The new acceptance file has 24 test functions / 45 parametrised cases. They check exact 3-4-5
separation, nonconstant two-node identity, band/magnitude independence, translation/rotation/
reflection/permutation invariance, missing carried strength, empty blocks, scope and source
refusal, unsupported periodic geometry, old-radius refusal, scalar/accelerated agreement,
absolute counts, ties, empty populations, invalid inputs and monotone radius feasibility.

```
$ .venv/Scripts/python.exe -m tools.mutate_spatial_identity
baseline: BASELINE_PASS
band_normalisation: KILLED
constant_pair_shape: KILLED
strengths_comparable: KILLED
scales_comparable: KILLED
node_order_not_canonical: KILLED
source_not_bound: KILLED
baseline: BASELINE_PASS
mixed_families_compared: KILLED
carried_scales_reenter_point: KILLED
baseline: BASELINE_PASS
split_radius_tie: KILLED
admission_radius_tie: KILLED
auc_ties_count_as_wins: KILLED
recall_rounds_down: KILLED
feasibility_always_true: KILLED
```

All three unmutated module-reload baselines passed all 45 cases. All 13 targeted mutations
were killed by test failures, with no timeout or collection failure counted as a kill.
`measurements/t4e8_spatial_mutations.json` preserves each subprocess's captured output.
The harness changes modules in isolated interpreter namespaces and never edits source files.
This is a bounded audit of these changes, not exhaustive mutation coverage and not completion
of T4E.5's separate unfinished mutation programme.

### Documentation and regression

```
$ .venv/Scripts/python.exe tools/audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 100 defined, 92 fixed, partial ['D18'], open ['D84', 'D85', 'D96', 'D97', 'D98', 'D99', 'D100']
test functions       : 4034
stale inventory rows : none
claimed suite totals : architecture (4441, 1) / roadmap (4441, 1)
RESULT               : ok
```

Re-run 2026-09-08 after T4E.9's specification opened D101. The block above is the earlier
dated capture and is left as it was measured.

```
$ .venv/Scripts/python.exe tools/audit_docs.py
undocumented modules : none
undocumented routes  : none
defects              : 101 defined, 92 fixed, partial ['D18'], open ['D84', 'D85', 'D96', 'D97', 'D98', 'D99', 'D100', 'D101']
test functions       : 4071
stale inventory rows : none
claimed suite totals : architecture (4441, 1) / roadmap (4441, 1)
RESULT               : ok
```

The full-suite count remains the dated T4E.7 measurement, not an arithmetic increment from
focused tests. No frontend implementation changed and no fresh rendered acceptance is claimed.
