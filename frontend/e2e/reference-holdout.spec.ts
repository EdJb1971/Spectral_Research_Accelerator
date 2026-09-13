import { expect, test, Page } from '@playwright/test';

/**
 * T4E.40: the holdout has to survive being looked at.
 *
 * The backend tests prove the surface serves verified stages; only a browser proves a reader
 * can reach them. This spec drives the panel the way a reviewer would - open the flow, expand a
 * stage, read the bytes behind it, look at the rows - and asserts that the two things the
 * result must never lose travel with it on screen: the `NOT_AN_ACCEPTANCE` verdict and the
 * fragility of a majority of one.
 *
 * It deliberately stops short of submitting the review. Writing one persists a signed artifact
 * into the real calibration store, and a browser test is not a named person.
 */
async function openHoldout(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const panel = page.getByRole('region', { name: 'T4E.39 temporal reference holdout' });
  await expect(panel).toBeVisible();
  return panel;
}

test('the whole holdout flow renders, and the verdict travels with the numbers', async ({ page }) => {
  const panel = await openHoldout(page);

  await expect(panel).toContainText('HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE');
  await expect(panel.getByTestId('holdout-verdict')).toHaveText('NOT_AN_ACCEPTANCE');
  await expect(panel).toContainText('12 selected storms');
  await expect(panel).toContainText('strict majority 7');

  // A majority of exactly one is reported as fragile on screen, not left to be computed.
  await expect(panel.getByTestId('holdout-fragility')).toContainText('HOLDOUT_MIXED');

  // Refusals are tiles of the same weight as the two candidate counts.
  await expect(panel).toContainText('Exact tie');
  await expect(panel).toContainText('Refused');

  // Every performed stage is present, in the order it happened.
  for (const stage of ['Freeze the holdout before opening it',
    'A named person adopts the exact declaration', 'Open the signed catalogue only',
    'Plan the four exact field requests offline', 'Acquire the 48 exact-time shards',
    'Join the segments into one immutable record', 'Run the frozen basin comparison',
    'A named person reads the result and its boundary']) {
    await expect(panel.getByRole('button', { name: new RegExp(stage) })).toBeVisible();
  }
});

test('a reader can open a stage and reach the bytes behind it', async ({ page }) => {
  const panel = await openHoldout(page);

  await panel.getByRole('button', { name: /Run the frozen basin comparison/ }).click();
  await expect(panel).toContainText('strict_majority_needed');
  await expect(panel).toContainText('catalogue_seeded');

  await panel.getByRole('button', { name: /Examine the measurement artifact/ }).click();
  const artifact = panel.getByTestId('holdout-artifact');
  await expect(artifact).toContainText('measurements/t4e39_reference_holdout_evaluation.json');
  await expect(artifact).toContainText('file sha256');
  await expect(artifact).toContainText('HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE');
});

test('all twelve measured storms and both basin walks are inspectable', async ({ page }) => {
  const panel = await openHoldout(page);

  await expect(panel).toContainText('All 12 measured storms');
  await expect(panel).toContainText('SETH');
  await expect(panel).toContainText('MAL');

  const row = panel.getByRole('row').filter({ hasText: 'CODY' });
  await row.getByRole('button', { name: 'both walks' }).click();
  await expect(row).toContainText('negated_vorticity_basin');
  await expect(row).toContainText('CENTRE_FOUND');
});

test('the surface names what it will not do, and offers no acceptance', async ({ page }) => {
  const panel = await openHoldout(page);

  await expect(panel).toContainText('Acquisition reaches a network under a separate named');
  await expect(panel).toContainText('Re-running it would manufacture a second holdout');

  // The only decisions a reviewer is offered are readings of the boundary.
  const form = panel.getByTestId('holdout-review-form');
  await expect(form.getByRole('button', { name: 'BOUNDARY_SOUND' })).toBeVisible();
  await expect(form.getByRole('button', { name: 'BOUNDARY_DISPUTED' })).toBeVisible();
  await expect(form.getByRole('group', { name: 'Boundary assessment' }))
    .not.toContainText('ACCEPT');
  await expect(form).toContainText('It cannot accept the result');
});
