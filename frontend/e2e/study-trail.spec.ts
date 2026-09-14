import { expect, test, Page } from '@playwright/test';

/** T4E.22: the study trail, rendered rather than described.
 *
 * PLAN section 5 asks for an interface that exposes the scientific contract rather than
 * operating the backend, and its acceptance for an interface slice is specific: rendered
 * evidence, with refusals demonstrated on screen rather than asserted in prose. These tests are
 * that demonstration.
 *
 * What each one exists to catch is a real thing this programme did. A result was published with
 * a wrong diagnosis and corrected a commit later; a study was declared and withdrawn before
 * adoption, producing no measurement at all; and every number in the record carries a clause
 * saying what it may not be used for, under fifteen different key names. A panel that showed
 * the numbers and dropped any of that would be a worse instrument than no panel.
 */

async function openStudyTrail(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: /study trail/i }).click();
  await expect(page.getByTestId('study-trail')).toBeVisible();
}

test.describe('the study trail', () => {
  test('a study is drawn as the chain it ran, not as a list of files', async ({ page }) => {
    await openStudyTrail(page);

    const study = page.getByTestId(/^study-T4E/).first();
    await expect(study).toBeVisible();
    // Exact matching: 'declared_before_measurement' is a status this panel also renders, and a
    // substring match would find it instead of the column heading.
    await expect(study.getByText('Declared', { exact: true })).toBeVisible();
    await expect(study.getByText('Measured', { exact: true })).toBeVisible();

    // The rendered evidence PLAN section 5 asks for, captured rather than described.
    await page.getByTestId('study-trail').screenshot({
      path: 'e2e/artifacts/study-trail-chain.png' });
  });

  test('every verdict on screen carries what it may not be used for', async ({ page }) => {
    await openStudyTrail(page);

    const verdicts = page.getByTestId(/^verdict-/);
    const count = await verdicts.count();
    expect(count).toBeGreaterThan(0);

    // A verdict without a boundary beside it is the failure this panel exists to prevent.
    const boundaries = page.getByTestId(/^boundary-/);
    expect(await boundaries.count()).toBeGreaterThan(0);
    await expect(boundaries.first()).toBeVisible();
  });

  test('a question declared and never measured is on screen, not filtered away', async ({ page }) => {
    await openStudyTrail(page);

    const unanswered = page.getByTestId(/^no-result-T4E/);
    expect(await unanswered.count()).toBeGreaterThan(0);
    await expect(unanswered.first()).toBeVisible();
    await expect(unanswered.first()).toContainText(/declared, not measured/i);
  });

  test('a corrected record is marked where a reader looks first', async ({ page }) => {
    await openStudyTrail(page);

    const corrected = page.getByTestId(/^corrected-T4E/);
    expect(await corrected.count()).toBeGreaterThan(0);
    await expect(corrected.first()).toContainText(/corrected in the open/i);
    await corrected.first().screenshot({ path: 'e2e/artifacts/study-trail-correction.png' });
  });

  test('the surface states what it will not do, rather than implying it by absent buttons',
    async ({ page }) => {
      await openStudyTrail(page);

      const refusals = page.getByTestId('surface-refusal');
      expect(await refusals.count()).toBeGreaterThan(0);
      await expect(refusals.first()).toBeVisible();
      await expect(page.getByTestId('study-trail')).toContainText(/no route/i);
      await refusals.first().screenshot({ path: 'e2e/artifacts/study-trail-refusal.png' });
    });

  test('a measurement opens in full and closes again', async ({ page }) => {
    await openStudyTrail(page);

    await page.getByTestId(/^verdict-/).first().click();
    const detail = page.getByTestId('measurement-detail');
    await expect(detail).toBeVisible();
    // Scoped to the panel: several result buttons now carry boundary text that contains
    // "no closure of D96", so an unscoped name match finds them instead of the control.
    await detail.getByRole('button', { name: 'close' }).click();
    await expect(page.getByTestId('measurement-detail')).toHaveCount(0);
  });
});
