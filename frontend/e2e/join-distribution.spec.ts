import { expect, test, Page } from '@playwright/test';

/**
 * T4E.29 acceptance, in a real browser: the distribution is on screen, not described.
 *
 * T4E.18's correction stated that away from the dateline the nearest extracted feature is
 * "16.6 to 99.3 km", and read that as "a factor of two to three, not an order of magnitude".
 * The median was right and the range was not: SETH sits 315.1 km out and HOLA 247.7, neither
 * near a boundary, and GRETEL yields no feature at all. The claim was written, reviewed and
 * committed, and it stood for a day because the record held one number per storm and because
 * the aggregate was taken over a subset nobody named.
 *
 * These tests read the rendered page rather than the source, because "a scientist can see it"
 * is a claim about what appears on a screen.
 *
 * What the picture must contain: every distance rather than the nearest; the storm that yielded
 * nothing, present and labelled rather than absent; an exclusion applied with the excluded rows
 * still listed and their aggregate computed at equal weight; and the boundary the measurement
 * was recorded under.
 *
 * What it must not contain: any control that records, approves or accepts anything.
 */

const WORKSPACE = 'Match-distance results';

async function openJoin(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: new RegExp(`^${WORKSPACE}`) }).click();
  await expect(page.getByTestId('join-distribution')).toBeVisible();
}

test('every storm is a row, including the one that yielded no feature', async ({ page }) => {
  await openJoin(page);

  // Eighteen storms were sampled. A view that plotted only those with a distance would show 17
  // and look complete, which is the failure this panel exists to prevent.
  await expect(page.getByTestId('join-distribution').locator('li[data-testid^="join-row-"]'))
    .toHaveCount(18);

  const gretel = page.getByTestId('no-feature-GRETEL');
  await expect(gretel).toBeVisible();
  await expect(gretel).toContainText('no feature was extracted');
  await expect(gretel).toContainText('stays in every denominator');
});

test('the outliers the record excluded without saying so are visible as rows', async ({ page }) => {
  await openJoin(page);

  // Neither is near a boundary, and both break the range the correction published.
  await expect(page.getByTestId('join-row-SETH')).toContainText('315.1');
  await expect(page.getByTestId('join-row-HOLA')).toContainText('247.7');
});

test('an exclusion keeps its rows on screen and reports both aggregates', async ({ page }) => {
  await openJoin(page);

  await page.getByTestId('exclude-longitude').fill('178');

  // The kept maximum is the number that refutes the published range, and it is one glance away.
  await expect(page.getByTestId('aggregate-kept')).toContainText('315.1');
  await expect(page.getByTestId('aggregate-excluded')).toContainText('3685.3');

  // Excluded storms do not disappear: they are marked and still listed.
  await expect(page.getByTestId('excluded-JOSIE')).toBeVisible();
  await expect(page.getByTestId('join-row-JOSIE')).toBeVisible();
  await expect(page.getByTestId('exclusion-note')).toContainText('about that population');
});

test('the extraction pass is named, and naming a different one changes the numbers', async ({ page }) => {
  await openJoin(page);

  // The raw pass yields 0 to 13 features per frame; the SWT pass 73 to 153. A panel that read
  // whichever was stored first is the error T4E.27 made.
  await expect(page.getByTestId('join-row-FEHI')).toContainText('11 features');

  await page.getByTestId('population').selectOption('swt_planes');
  await expect(page.getByTestId('join-row-FEHI')).toContainText('140 features');
});

test('the panel carries the boundary the measurement was recorded under', async ({ page }) => {
  await openJoin(page);

  await expect(page.getByTestId('join-boundary')).toContainText('regenerate');
  await expect(page.getByTestId('join-refusal').first()).toBeVisible();
});

test('nothing here records, approves or accepts', async ({ page }) => {
  await openJoin(page);

  const names = await page.getByTestId('join-distribution').getByRole('button').allTextContents();
  for (const name of names) {
    expect(name).not.toMatch(/accept|approve|save|record|apply|adopt/i);
  }
});
