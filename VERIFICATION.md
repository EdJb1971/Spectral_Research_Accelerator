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

## Phase 3.5 progress

**Fixed: 16 of 25 known defects** — D2, D3, D4, D5, D6, D7, D9, D10, D11, D19, D20, D21,
D22, D23, D24, D25.

**Still outstanding:**

| Defect | Task | Note |
|---|---|---|
| **D1** | T3.5.6 | **The big one.** DTCWT is still Haar in disguise; strict-xfail test standing by. |
| D8 | T4C.5 | No FDR control. Demonstrated live: 9 runs produced 9 "hypotheses". |
| D12 | T3.5.12 | Unseeded RNG — provenance cannot reproduce its own runs. |
| D13 | T3.5.13 | Metric-unaware gradients; PSD in pixel wavenumber on a lat/lon grid. |
| D14 | T3.5.14 | Ten generic 500s discard the exception. |
| D15 | T3.5.15 | Registries; the "seams" are still if/elif chains. |
| D16 | T3.5.16 | float32 forced in `PhysicalField`. |
| D17 | T3.5.20 | Per-bin Python loops over full arrays. |
| D18 | T3.5.21 | CUDA-only device probe, no MPS. |

Also outstanding: T3.5.7 (SWT), T3.5.8 (Alembic), T3.5.17 (ground-truth benchmark suite),
T3.5.18 (Zarr/ERA5), T3.5.19 (Executor seam), browser-based UI verification, and wiring the
new health/list endpoints into the frontend.
