import { expect, test, Page } from '@playwright/test';

/**
 * T4E.8 slice 4 acceptance, in a real browser: the refusal is on screen, not described.
 *
 * PLAN section 5 states the requirement this panel exists to meet -- the interface must expose
 * the scientific contract rather than operate the backend, and a refusal ranks equal to a
 * value. That is a claim about what a researcher can *see*, so source checks cannot settle it
 * and these tests read the rendered page.
 *
 * Three things the picture must contain. The inadmissible pairing -- recurrence of a physical
 * kind judged against labels drawn from the same pipeline -- rendered with its reason, because
 * a panel showing only the workable combinations hides the cell that matters most. An admitted
 * pairing that still owes a caveat, shown carrying it. And a receipt written before the
 * declaration existed, labelled as undeclared rather than quietly omitted.
 *
 * One thing it must not contain: any control that chooses. Choosing an identity target is a
 * scientific act, and the server serves no route that would accept one.
 */

const WORKSPACE = 'Identity definition';

/**
 * The primary label only. This entry belongs to the gridded field line, so its accessible name
 * also carries the "Legacy · ..." sublabel TG18.3 requires; matching the whole name here would
 * couple reachability to that wording.
 */
async function openIdentity(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: new RegExp(`^${WORKSPACE}`) }).click();
  await expect(
    page.getByRole('heading', { name: 'What identity is meant to recognise' })
  ).toBeVisible();
}

test.describe('the identity declaration, rendered', () => {
  test('every declared target is drawn with what it does not license', async ({ page }) => {
    await openIdentity(page);
    for (const target of ['track_continuity', 'spatial_persistence', 'kind_recurrence']) {
      await expect(page.getByText(target, { exact: false }).first()).toBeVisible();
    }
    await expect(page.getByText('Does not license:').first()).toBeVisible();
  });

  test('the circular pairing is on screen as a refusal, with its reason', async ({ page }) => {
    await openIdentity(page);
    const cell = page.getByTestId('cell-kind_recurrence-record_derived_proxy');
    await expect(cell).toBeVisible();
    await expect(cell.getByTestId('refused')).toBeVisible();
    await expect(cell).toContainText('validated against itself');
  });

  test('a refusal is drawn at the weight of an admission, not as an error', async ({ page }) => {
    await openIdentity(page);
    const refused = page.getByTestId('cell-kind_recurrence-record_derived_proxy');
    const admitted = page.getByTestId('cell-kind_recurrence-external_reference');
    await expect(refused).toBeVisible();
    await expect(admitted).toBeVisible();
    const refusedBox = await refused.boundingBox();
    const admittedBox = await admitted.boundingBox();
    expect(refusedBox).not.toBeNull();
    expect(admittedBox).not.toBeNull();
    // Same column, comparable width: the refusal is a peer of the admission on the page.
    expect(Math.abs((refusedBox!.width) - (admittedBox!.width))).toBeLessThan(4);
  });

  test('an admitted pairing still shows the caveat it owes', async ({ page }) => {
    await openIdentity(page);
    const cell = page.getByTestId('cell-track_continuity-record_derived_proxy');
    await expect(cell.getByTestId('admitted')).toBeVisible();
    await expect(cell).toContainText('agreement with the tracker');
  });

  test('receipts show their claim boundary and that no radius is approved', async ({ page }) => {
    await openIdentity(page);
    await expect(page.getByTestId('approved-radius').first()).toContainText(
      'no approved mining radius'
    );
  });

  test('a receipt written before the declaration existed is labelled, not hidden', async ({
    page,
  }) => {
    await openIdentity(page);
    await expect(page.getByTestId('undeclared').first()).toContainText(
      'No declared identity target'
    );
  });

  test('a receipt opens whole rather than in fragments', async ({ page }) => {
    await openIdentity(page);
    await page.getByTestId('open-t4e8-spatial-audit-v3.json').click();
    await expect(page.getByTestId('open-audit-body')).toBeVisible();
    await expect(page.getByTestId('open-audit-body')).toContainText('identity_declaration');
  });

  test('the panel offers no control that chooses a target', async ({ page }) => {
    await openIdentity(page);
    await expect(page.locator('select')).toHaveCount(0);
    await expect(page.locator('form')).toHaveCount(0);
    await expect(
      page.getByRole('heading', { name: 'What this surface will not do' })
    ).toBeVisible();
    await expect(page.getByText('does not choose', { exact: false }).first()).toBeVisible();
  });
});
