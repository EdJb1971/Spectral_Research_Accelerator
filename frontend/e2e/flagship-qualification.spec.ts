import { expect, test, Page } from '@playwright/test';

const COMPOSER = 'Experiment Composer';

async function openComposer(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: COMPOSER, exact: true }).click();
  await expect(page.getByRole('tablist', { name: 'Experiment composition path' })).toBeVisible();
}

async function step(page: Page, name: RegExp) {
  await page.getByRole('tab', { name }).click();
}

test('the generated release gate runs offline but refuses to certify unrun science', async ({ page }) => {
  test.slow();
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const gate = page.getByRole('region', { name: 'G17 flagship release qualification' });
  await expect(gate.getByRole('table', {
    name: 'Duration and comparison mode qualification matrix' })).toBeVisible();
  await expect(gate.getByRole('row')).toHaveCount(7);
  await expect(gate).toContainText('NOT_RELEASEABLE');

  await gate.getByRole('button', { name: 'Run offline qualification' }).click();
  await expect(gate.getByRole('status')).toContainText('Offline apparatus checks finished', {
    timeout: 30_000 });
  await expect(gate).toContainText('3 of 6 deterministic cells passed');
  // The three refused cells name the declaration that refused them, on screen.
  await expect(gate).toContainText('order_book.bespoke_record');
  await expect(gate).toContainText('Six-cell known-answer apparatus matrix: REFUSED');
  await expect(gate).toContainText('Single remote-failure restart recovery: PASS');
  // TG17.11: a registered calibration now exists, so the gate is no longer unimplemented.
  // It is still blocking, and REFUSED is what the rendered ledger must show.
  await expect(gate).toContainText('Scale/shape null calibration and planted power: REFUSED');
  await expect(gate).toContainText('Four-domain live-source tail: NOT_RUN');
  await expect(gate).toContainText('NOT_RELEASEABLE');
});

test('clean-browser four-domain path needs no JSON, and says why one mode is closed', async ({ page }) => {
  test.slow();
  // Cleared once, not on every navigation: an init script would also wipe the saved draft
  // pointer at the reload below, and the refresh would then be a new browser, not a resume.
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  let scientistActions = 0;
  await openComposer(page);

  // Asking for the mode this quartet cannot support must be answered in the browser, in words,
  // rather than by a silent switch that fails later at execution. `order_book.bespoke_record`
  // declines the scale/shape null, so no complete plan exists for these four domains.
  await page.getByRole('radio', { name: 'Scale/shape recurrence' }).click(); scientistActions++;
  await expect(page.getByRole('status')
    .filter({ hasText: 'do not yet admit a complete scale_shape_aligned plan' })).toBeVisible();
  await expect(page.getByRole('radio', { name: 'Calendar-aligned co-occurrence' })).toBeChecked();

  // Bind the independently supplied fourth record by content identity through its generated
  // adapter control. No filename, CSV conversion, terminal or handwritten manifest is involved.
  await step(page, /3\. Observation/); scientistActions++;
  const orderBook = page.getByRole('group', { name: 'Order book' });
  await orderBook.locator('summary').click(); scientistActions++;
  await orderBook.getByLabel('Record').fill('4'.repeat(64)); scientistActions++;

  await step(page, /5\. Analysis/); scientistActions++;
  await expect(page.getByRole('radio', { name: /independent_native_clock_shift/ })).toBeChecked();

  // The frozen flagship's confirmation partition is spent by the plan that opened it. An edited
  // plan is a different plan and declares its own, in the browser, with no manifest editing.
  await page.getByLabel('Held-out confirmation partition')
    .fill('g17_no_glue_calendar_heldout_v1'); scientistActions++;

  await step(page, /4\. Preflight/); scientistActions++;
  await page.getByRole('button', { name: 'Inspect metadata coverage' }).click(); scientistActions++;
  await expect(page.getByRole('tabpanel')).toContainText('Metadata preflight: PARTIAL');
  await expect(page.getByRole('tabpanel')).not.toContainText('REFUSED');

  // Save the exact revision so refresh reconstructs it rather than relying on component state.
  await step(page, /1\. Question/); scientistActions++;
  await page.getByRole('button', { name: 'Save draft' }).click(); scientistActions++;
  await expect(page.getByRole('status')
    .filter({ hasText: 'Refresh will reload this draft' })).toBeVisible();

  await step(page, /6\. Freeze and run/); scientistActions++;
  await page.getByRole('button', { name: 'Open or resume the run' }).click(); scientistActions++;
  await page.getByRole('button', { name: 'Execute the frozen plan' }).click(); scientistActions++;
  await expect(page.getByRole('status').filter({ hasText: 'not evidence' })).toBeVisible();

  // Refresh is deliberately not counted as a scientific action. Reopening addresses the same
  // saved manifest and must say it resumed rather than silently starting another experiment.
  await page.reload();
  await page.getByRole('button', { name: COMPOSER, exact: true }).click();
  // Wait for the saved revision itself, not for the panel: opening a run before the draft has
  // been reloaded would post the untouched default and honestly report a second plan.
  await step(page, /5\. Analysis/);
  await expect(page.getByLabel('Held-out confirmation partition'))
    .toHaveValue('g17_no_glue_calendar_heldout_v1');
  await step(page, /6\. Freeze and run/);
  await page.getByRole('button', { name: 'Open or resume the run' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Resumed run' })).toBeVisible();

  await step(page, /7\. Interpret/);
  await expect(page.getByRole('tabpanel')).toContainText('COMPLETE');
  await page.getByRole('button', { name: 'Export verified bundle' }).click();
  await expect(page.getByRole('tabpanel')).toContainText('VERIFIED: bundle');
  await expect(page.getByRole('tabpanel')).toContainText('measured results: ABSENT');
  expect(scientistActions).toBe(13);
});
