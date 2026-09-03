import { readFile } from 'node:fs/promises';
import { expect, test, Page } from '@playwright/test';

/**
 * TG18.2 acceptance: a publication export is the rendered vector figure plus the reading
 * contract, not a screenshot stripped of units and qualifications. Downloads are opened and
 * inspected here; seeing a button is not evidence that it preserved what the page said.
 */

function workflowNav(page: Page) {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function generateField(page: Page) {
  await page.goto('/');
  await workflowNav(page).getByRole('button', { name: /^Synthetic generator/ }).click();
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });
}

async function downloadHtml(page: Page, figureId: string) {
  const figure = page.locator(`#${figureId}`).locator('xpath=ancestor::figure');
  const pending = page.waitForEvent('download');
  await figure.getByRole('button', { name: 'Publication HTML' }).click();
  const download = await pending;
  const path = await download.path();
  if (!path) throw new Error('Browser did not retain the publication download');
  return { html: await readFile(path, 'utf8'), filename: download.suggestedFilename() };
}

test('every rendered figure family exposes the same publication action', async ({ page }) => {
  await generateField(page);
  const heatmaps = page.locator('figure').filter({ has: page.locator('.js-plotly-plot') });
  await expect(heatmaps).toHaveCount(2);
  await expect(heatmaps.getByRole('button', { name: 'Publication HTML' })).toHaveCount(2);

  await workflowNav(page).getByRole('button', { name: /^Diagnostics/ }).click();
  await page.getByRole('button', { name: 'Run Benchmarking diagnostics' }).click();
  await expect(page.locator('#fig-psd.js-plotly-plot')).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('#fig-psd').locator('xpath=ancestor::figure')
    .getByRole('button', { name: 'Publication HTML' })).toBeVisible();
});

test('the heat-map sheet carries the vector, caption and fully evaluated contract', async ({ page }) => {
  await generateField(page);
  const shape = await page.evaluate(() => {
    const values = (document.getElementById('fig-clean-field') as any)?.data?.[0]?.z || [];
    return { rows: values.length, columns: values[0]?.length || 0 };
  });
  const { html, filename } = await downloadHtml(page, 'fig-clean-field');
  expect(filename).toMatch(/^Generated_Clean_Field__F__\d{8}T\d{6}Z\.html$/);
  expect(html).toContain('data:image/svg+xml');
  expect(html).toContain('Synthetic analytical field; dimensionless');
  expect(html).toContain('Figure reading contract');
  expect(html).toContain(`${shape.rows} rows x ${shape.columns} columns`);
  expect(html).toContain('supplied, shared across panels');
  // This value is evaluated for the export even when the deferred on-screen disclosure stayed
  // closed; a publication sheet must not turn uninspected missingness into an em dash.
  expect(html).toContain(`all ${shape.rows * shape.columns} samples are finite`);
  expect(html).toContain('performs no scientific analysis');
  expect(html).not.toContain('<script');
});

test('the spectral sheet carries the fit domain, uncertainty and assumptions verbatim', async ({ page }) => {
  await generateField(page);
  await workflowNav(page).getByRole('button', { name: /^Diagnostics/ }).click();
  await page.getByRole('button', { name: 'Run Benchmarking diagnostics' }).click();
  await expect(page.locator('#fig-psd.js-plotly-plot')).toBeVisible({ timeout: 60_000 });

  const { html } = await downloadHtml(page, 'fig-psd');
  expect(html).toContain('fit domain');
  expect(html).toContain('spectral exponent');
  expect(html).toContain('Producer statements and qualifications');
  expect(html).toContain('isotropy');
  expect(html).toContain('no break point is fitted or tested');
  expect(html).toContain('fitted model itself is not drawn');
});

test('publication export does not mutate the live Plotly trace', async ({ page }) => {
  await generateField(page);
  const before = await page.evaluate(() => {
    const plot = document.getElementById('fig-clean-field') as any;
    return { data: plot?.data, layout: plot?.layout };
  });
  await downloadHtml(page, 'fig-clean-field');
  const after = await page.evaluate(() => {
    const plot = document.getElementById('fig-clean-field') as any;
    return { data: plot?.data, layout: plot?.layout };
  });
  expect(after).toEqual(before);
});
