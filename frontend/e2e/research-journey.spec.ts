import { expect, test, Page } from '@playwright/test';

/** TG18.3 rendered acceptance: the global map navigates; Composer still decides science. */

function journey(page: Page) {
  return page.getByRole('navigation', { name: 'Guided research journey' });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(journey(page)).toBeVisible();
});

test('the complete journey is visible and explicitly separate from the claim ladder', async ({ page }) => {
  const map = journey(page);
  await expect(map.getByRole('listitem')).toHaveCount(7);
  await expect(map.getByRole('listitem')).toHaveText([
    /1Acquire/, /2Inspect/, /3Design/, /4Run/, /5Compare/, /6Admit/, /7Report/,
  ]);
  await expect(map).toContainText('Navigation only');
  await expect(map).toContainText('does not advance or replace the claim ladder');
});

test('every shell-level blocker names and performs one legitimate remediation', async ({ page }) => {
  const map = journey(page);
  const blocked = map.locator('.is-blocked');
  await expect(blocked).toHaveCount(2);
  await expect(blocked.nth(0)).toContainText('Blocked: no record is selected for inspection.');
  await expect(blocked.nth(0).getByRole('button')).toHaveText(
    'Next legitimate action: Acquire a record');
  await expect(blocked.nth(1)).toContainText('Blocked: no study is selected for evidence admission.');
  await expect(blocked.nth(1).getByRole('button')).toHaveText(
    'Next legitimate action: Open Composer and save a study');

  await blocked.nth(1).getByRole('button').click();
  await expect(page.getByRole('heading', { name: 'Experiment Composer' })).toBeVisible();
  await expect(page.getByRole('tab', { selected: true })).toHaveText(/1\. Question/);
});

test('design, run and compare hand off to Composer without inventing a status', async ({ page }) => {
  const map = journey(page);
  await map.getByRole('button', { name: 'Open Run' }).click();
  await expect(page.getByRole('tab', { selected: true })).toHaveText(/6\. Freeze and run/);
  await expect(page.getByRole('status').filter({ hasText: /Next legitimate action|Blocked before/ }))
    .toHaveCount(1);

  await map.getByRole('button', { name: 'Open Compare' }).click();
  await expect(page.getByRole('tab', { selected: true })).toHaveText(/7\. Interpret/);
  const panel = page.getByRole('tabpanel');
  for (const rung of ['Acquired material', 'Executed run', 'Finding', 'Admitted evidence']) {
    await expect(panel).toContainText(rung);
  }
  await expect(map).toContainText('does not advance or replace the claim ladder');
});

test('report is a direct journey destination even when the findings set is empty', async ({ page }) => {
  await journey(page).getByRole('button', { name: 'Open Report' }).click();
  await expect(page.getByRole('heading', { name: 'Findings', exact: true })).toBeVisible();
  await expect(page.locator('#workspace-heading')).toHaveText('Findings workspace');
  await expect(journey(page).getByRole('button', { name: 'Open Report' }))
    .toHaveAttribute('aria-current', 'step');
});

test('legacy gridded tools remain reachable and carry a distinct visual and text treatment', async ({ page }) => {
  const workflow = page.getByRole('navigation', { name: 'Scientific workflow' });
  const legacy = workflow.locator('[data-workflow-line="legacy-gridded"]');
  await expect(legacy).toHaveCount(8);
  await expect(legacy.first()).toContainText('Legacy · Gridded field line');
  await expect(legacy.first()).toHaveCSS('box-shadow', /rgba?\(245, 158, 11/);

  await legacy.filter({ hasText: 'Synthetic generator' }).click();
  await expect(page.getByRole('heading', { name: 'Synthetic Field Generator' })).toBeVisible();
  await expect(journey(page)).toBeVisible();
});
