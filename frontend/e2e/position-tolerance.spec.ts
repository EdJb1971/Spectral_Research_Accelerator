import { expect, test, Page } from '@playwright/test';

/**
 * TG19.2 acceptance, in a real browser: the bar is on screen in parts, not described.
 *
 * T4E.18 judged a catalogue join against the catalogue's own per-observation radius and failed.
 * T4E.27 showed that bar was wrong for three reasons -- it bounds the wrong quantity, one
 * storm's radius was exactly 0.00 and could never have been satisfied, and the offset does not
 * track the radius at all -- and that replacing it with a defensible one changed nothing.
 *
 * Both halves matter to a researcher and neither was visible: the tolerance existed only as a
 * Python module. PLAN section 5 requires the interface to expose the scientific contract rather
 * than operate the backend, and that is a claim about what someone can *see*, so these tests
 * read the rendered page rather than the source.
 *
 * Four things the picture must contain. The components with their provenance, so the bar can be
 * disagreed with specifically. What the bar deliberately leaves out, at the weight of what it
 * includes, because that decides what its residual means. A missing catalogue uncertainty
 * refused by name rather than scored as a miss. And the residual itself, named as the measure of
 * the excluded component.
 *
 * One thing it must not contain: any control that records an acceptance. Computing a bar is
 * arithmetic; deciding a join is a scientific act, and the server serves no route that would
 * accept one.
 */

const WORKSPACE = 'Position matching';

async function openTolerance(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: new RegExp(`^${WORKSPACE}`) }).click();
  await expect(page.getByTestId('position-tolerance')).toBeVisible();
}

test('the bar arrives in parts, each with where it came from', async ({ page }) => {
  await openTolerance(page);

  const components = page.getByTestId('tolerance-components');
  await expect(components).toBeVisible();
  await expect(components).toContainText('catalogue_uncertainty');
  await expect(components).toContainText('estimator_localisation');
  // Provenance, not just names: a component a reader cannot trace is one they can only accept.
  await expect(components).toContainText('T4E.21');
  await expect(components).toContainText('quadrature');
});

test('what the bar leaves out is on screen at the weight of what it includes', async ({ page }) => {
  await openTolerance(page);

  const excluded = page.getByTestId('tolerance-excluded');
  await expect(excluded).toBeVisible();
  await expect(excluded).toContainText('surface centre');
  await expect(excluded).toContainText('fit the bar to the result');
});

test('a missing catalogue uncertainty is refused by name, not scored as a miss', async ({ page }) => {
  await openTolerance(page);

  // LINDA's reported radius in the acquired record is exactly 0.00 -- the case that made one
  // storm of eighteen unpassable however good the extraction was.
  await page.getByLabel('Observation').fill('LINDA');
  await page.getByLabel('Catalogue radius (km)').fill('0');
  await page.getByLabel('Separation (km, optional)').fill('74.91');
  await page.getByRole('button', { name: 'Compute' }).click();

  const refusal = page.getByTestId('tolerance-refusal');
  await expect(refusal).toBeVisible();
  await expect(refusal).toContainText('Refused');
  await expect(refusal).toContainText('missing report');
  // The verdict is withheld rather than returned as a failure.
  await expect(page.getByTestId('tolerance-no-verdict')).toContainText('different answers');
  await expect(page.getByTestId('tolerance-verdict')).toHaveCount(0);
});

test('a refusal renders as a result, not as an error state', async ({ page }) => {
  await openTolerance(page);

  await page.getByLabel('Catalogue radius (km)').fill('0');
  await page.getByRole('button', { name: 'Compute' }).click();
  await expect(page.getByTestId('tolerance-refusal')).toBeVisible();

  // The panel still shows its result section and no error banner: the bar could not be built,
  // which is an answer the instrument is entitled to give.
  await expect(page.getByTestId('tolerance-result')).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('a computed bar shows its total and the residual it does not explain', async ({ page }) => {
  await openTolerance(page);

  await page.getByLabel('Observation').fill('OWEN');
  await page.getByLabel('Catalogue radius (km)').fill('11.12');
  await page.getByLabel('Separation (km, optional)').fill('99.98');
  await page.getByRole('button', { name: 'Compute' }).click();

  await expect(page.getByTestId('tolerance-total')).toContainText('13.81');
  await expect(page.getByTestId('tolerance-verdict')).toContainText('outside the bar');
  const residual = page.getByTestId('tolerance-residual');
  await expect(residual).toContainText('86.17');
  await expect(residual).toContainText('the component the bar leaves out');
});

test('the panel states what it will not do, from the server own list', async ({ page }) => {
  await openTolerance(page);

  const refusals = page.getByTestId('tolerance-refusals');
  await expect(refusals).toBeVisible();
  await expect(refusals).toContainText('scientific declaration');

  // Nothing here records an acceptance or approves a join. The only control is arithmetic.
  const buttons = page.getByTestId('position-tolerance').getByRole('button');
  const names = await buttons.allTextContents();
  for (const name of names) {
    expect(name).not.toMatch(/accept|approve|save|record|apply/i);
  }
});
