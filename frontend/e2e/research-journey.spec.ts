import { expect, test, Page } from '@playwright/test';

/** TG18.3 rendered acceptance: the global map navigates; Composer still decides science. */

function journey(page: Page) {
  return page.getByRole('navigation', { name: 'Guided research journey' });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(journey(page)).toBeVisible();
});

test('the complete journey is visible and explains what each step means', async ({ page }) => {
  const map = journey(page);
  await expect(map.getByRole('listitem')).toHaveCount(7);
  await expect(map.getByRole('listitem')).toHaveText([
    /1Acquire/, /2Inspect/, /3Design/, /4Run/, /5Compare/, /6Validate/, /7Share/,
  ]);
  await expect(map).toContainText('move from data to a traceable result');
  await expect(map).toContainText('does not by itself prove a scientific claim');
});

test('every shell-level blocker names and performs one legitimate remediation', async ({ page }) => {
  const map = journey(page);
  const blocked = map.locator('.is-blocked');
  await expect(blocked).toHaveCount(2);
  await expect(blocked.nth(0)).toContainText('Select a data record before inspecting it.');
  await expect(blocked.nth(0).getByRole('button')).toHaveText('Acquire a record');
  await expect(blocked.nth(1)).toContainText('Select or create a study before adding evidence.');
  await expect(blocked.nth(1).getByRole('button')).toHaveText('Open Composer and save a study');

  await blocked.nth(1).getByRole('button').click();
  await expect(page.getByRole('heading', { name: 'New experiment' })).toBeVisible();
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
  await expect(map).toContainText('does not by itself prove a scientific claim');
});

test('report is a direct journey destination even when the findings set is empty', async ({ page }) => {
  await journey(page).getByRole('button', { name: 'Read findings' }).click();
  await expect(page.getByRole('heading', { name: 'Findings', exact: true })).toBeVisible();
  await expect(page.locator('#workspace-heading')).toHaveText('Findings workspace');
  await expect(journey(page).getByRole('button', { name: 'Read findings' }))
    .toHaveAttribute('aria-current', 'step');
});

test('specialised gridded-field tools remain reachable without development labels', async ({ page }) => {
  const workflow = page.getByRole('navigation', { name: 'Scientific workflow' });
  const specialised = workflow.locator('[data-workflow-line="gridded-field"]');
  await expect(specialised).toHaveCount(12);
  await expect(specialised.first()).toContainText('Gridded field line');
  await expect(specialised.first()).not.toContainText('Legacy');

  await specialised.filter({ hasText: 'Synthetic data' }).click();
  await expect(page.getByRole('heading', { name: 'Synthetic Field Generator' })).toBeVisible();
  await expect(journey(page)).toBeVisible();
});

test('the landing dashboard exposes the main user actions', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Research dashboard' })).toBeVisible();
  const actions = page.getByRole('region', { name: 'Common actions' });
  await expect(actions.getByRole('button', { name: /Start a new experiment/ })).toBeVisible();
  await expect(actions.getByRole('button', { name: /Add or find data/ })).toBeVisible();
  await expect(actions.getByRole('button', { name: /Read findings/ })).toBeVisible();
  await expect(page.getByText('Why pytest studies are not listed as studies')).toHaveCount(0);
});

test('a past study can be opened directly in the round table', async ({ page }) => {
  await page.route('**/api/v1/findings/studies', async (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify([{
      file: 'demo-study.json', readable: true, study_id: 'demo-study', revision: 1,
      hypothesis: 'A saved study that needs critical review', rung: 'FINDING', blocked: false,
    }]),
  }));
  await page.reload();

  const study = page.getByRole('list', { name: 'Archive records' })
    .getByRole('listitem').filter({ hasText: 'demo-study' });
  await expect(study.getByRole('button', { name: 'Discuss' })).toBeVisible();
  await study.getByRole('button', { name: 'Discuss' }).click();

  await expect(page.getByRole('heading', { name: 'Expert round table', exact: true })).toBeVisible();
  await expect(page.getByLabel('Study to discuss')).toHaveValue('demo-study');
});
