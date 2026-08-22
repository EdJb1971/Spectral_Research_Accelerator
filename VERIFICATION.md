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
autograd path. The tiny residual model is not Adam's model, is not trained to skill and supplies
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

The actual professor/laboratory architecture, weights, history semantics and declared training
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
behaviour only. No professor/Adam repository, configuration, coordinate hash or normalisation
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
evidence are synthetic fixtures. No real professor/Adam configuration, data crop, statistics,
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
and deterministic receipt binding. They do not show that Adam's laptop or HPC allocation was
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

`LICENSE.md` now records Edward Jonathan Bentley as owner and gives Adam Frank Bentley a named,
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
