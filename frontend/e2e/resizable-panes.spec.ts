import { expect, test, Page } from '@playwright/test';

/**
 * TG18.2 acceptance: compared figures can yield canvas to one another without changing the
 * scientific object being shown. This is a rendered interaction contract: source inspection can
 * see a separator, but cannot establish that moving it changes two laid-out panes, preserves both
 * Plotly traces, or disappears when a one-column reading order is required.
 */

function workflowNav(page: Page) {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function openSyntheticPair(page: Page) {
  await page.goto('/');
  await workflowNav(page).getByRole('button', { name: /^Synthetic data/ }).click();
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });
  return page.getByRole('separator', { name: 'Resize the clean and perturbed synthetic field panes' });
}

test('the equal split states its presentation-only boundary', async ({ page }) => {
  const separator = await openSyntheticPair(page);
  await expect(separator).toHaveAttribute('aria-valuenow', '50');
  await expect(separator).toHaveAttribute(
    'aria-valuetext', 'First figure 50 percent; second figure 50 percent');
  await expect(separator.locator('xpath=..')).toContainText(
    'Pane width changes presentation only. Both figures remain present');
});

test('arrow keys resize, accelerated keys step farther, and Enter restores equality', async ({ page }) => {
  const separator = await openSyntheticPair(page);
  await separator.focus();
  await page.keyboard.press('ArrowRight');
  await expect(separator).toHaveAttribute('aria-valuenow', '55');
  await page.keyboard.press('Shift+ArrowRight');
  await expect(separator).toHaveAttribute('aria-valuenow', '65');
  await page.keyboard.press('Enter');
  await expect(separator).toHaveAttribute('aria-valuenow', '50');
});

test('Home and End expose the declared 25–75 percent bounds', async ({ page }) => {
  const separator = await openSyntheticPair(page);
  await separator.focus();
  await page.keyboard.press('Home');
  await expect(separator).toHaveAttribute('aria-valuenow', '25');
  await page.keyboard.press('End');
  await expect(separator).toHaveAttribute('aria-valuenow', '75');
});

test('dragging changes both rendered pane widths and double-click restores them', async ({ page }) => {
  const separator = await openSyntheticPair(page);
  const pair = separator.locator('xpath=..');
  const panes = pair.locator('.resizable-figure-pair__pane');
  const pairBox = await pair.boundingBox();
  const handleBox = await separator.boundingBox();
  if (!pairBox || !handleBox) throw new Error('Resizable pair was not laid out');

  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + 40);
  await page.mouse.down();
  await page.mouse.move(pairBox.x + pairBox.width * 0.7, handleBox.y + 40, { steps: 4 });
  await page.mouse.up();
  await expect(separator).toHaveAttribute('aria-valuenow', /^(69|70|71)$/);

  const first = await panes.nth(0).boundingBox();
  const second = await panes.nth(1).boundingBox();
  expect(first!.width).toBeGreaterThan(second!.width);

  await separator.dblclick();
  await expect(separator).toHaveAttribute('aria-valuenow', '50');
});

test('resizing preserves both Plotly data arrays and their shared scale', async ({ page }) => {
  const separator = await openSyntheticPair(page);
  const read = () => page.evaluate(() => ['fig-clean-field', 'fig-perturbed-field'].map((id) => {
    const plot = document.getElementById(id) as any;
    return { z: plot?.data?.[0]?.z, zmin: plot?.data?.[0]?.zmin, zmax: plot?.data?.[0]?.zmax };
  }));
  const before = await read();
  await separator.focus();
  await page.keyboard.press('Shift+ArrowLeft');
  const after = await read();
  expect(after).toEqual(before);
  expect(after[0].zmin).toBe(after[1].zmin);
  expect(after[0].zmax).toBe(after[1].zmax);
});

test('a narrow viewport restores one-column document order and removes the separator', async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 900 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open workspace menu' }).click();
  await workflowNav(page).getByRole('button', { name: /^Synthetic data/ }).click();
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });

  const pair = page.locator('.resizable-figure-pair').first();
  const panes = pair.locator('.resizable-figure-pair__pane');
  await expect(page.getByRole('separator', {
    name: 'Resize the clean and perturbed synthetic field panes',
  })).toBeHidden();
  const first = await panes.nth(0).boundingBox();
  const second = await panes.nth(1).boundingBox();
  expect(second!.y).toBeGreaterThan(first!.y + first!.height - 1);
  expect(Math.abs(first!.width - second!.width)).toBeLessThan(2);
});
