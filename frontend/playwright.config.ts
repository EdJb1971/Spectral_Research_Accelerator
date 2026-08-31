import { defineConfig } from '@playwright/test';

/**
 * TG17.7 acceptance: the guided path driven in a real browser.
 *
 * Every other check in this repository reads source or calls HTTP. Neither proves the page
 * renders - `tsc` passed and every backend test was green while the hypothesis card would have
 * thrown "Objects are not valid as a React child" on first paint (the defect that made
 * `test_frontend_contract.py` exist). This config closes that gap by serving the real API and
 * the real frontend and driving them with Chromium.
 *
 * The backend is pointed at a scratch state directory, for the same reason the pytest `client`
 * fixture is: routes that persist fall back to `data/` when nothing binds them, and a run
 * identity is the content address of its manifest - so an unbound browser test posting the
 * flagship would resume, and then advance, whatever real run that plan already had.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:3000',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // Backslashes: this is handed to cmd.exe, which does not resolve `.venv/Scripts/...`.
      command: '.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000',
      cwd: '..',
      url: 'http://127.0.0.1:8000/api/v1/health',
      timeout: 180_000,
      reuseExistingServer: true,
      env: {
        EXPERIMENT_RUN_DIR: '.e2e-state/experiment_runs',
        EXPERIMENT_MANIFEST_DIR: '.e2e-state/experiment_manifests',
      },
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 3000',
      url: 'http://127.0.0.1:3000',
      timeout: 180_000,
      reuseExistingServer: true,
    },
  ],
});
