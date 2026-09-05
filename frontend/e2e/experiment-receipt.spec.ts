import { expect, test, Page, APIRequestContext } from '@playwright/test';

/** TG17.9: the portable receipt is operable through rendered controls and remains below evidence. */

async function exportedFixture(request: APIRequestContext) {
  const recipeResponse = await request.get('/api/v1/experiment-composer/recipes/g17-flagship-calendar');
  const recipe = await recipeResponse.json();
  recipe.canonical_manifest.observations = recipe.canonical_manifest.observations
    .filter((row: { domain: string }) => row.domain !== 'order_book');
  const openedResponse = await request.post('/api/v1/experiment-runs', {
    data: recipe.canonical_manifest,
  });
  expect(openedResponse.ok()).toBeTruthy();
  const opened = await openedResponse.json();
  if (opened.receipt.state !== 'COMPLETE') {
    const execution = await request.post(`/api/v1/experiment-runs/${opened.run_id}/execute`, {
      data: { worker_suite: 'fixture_dry_run' },
    });
    expect(execution.ok()).toBeTruthy();
  }
  const exported = await request.post(
    `/api/v1/experiment-receipts/runs/${opened.run_id}/export`);
  expect(exported.ok()).toBeTruthy();
  return exported.json();
}

async function openInterpret(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Experiment Composer', exact: true }).click();
  await page.getByRole('tab', { name: /7\. Interpret/ }).click();
  await expect(page.getByRole('heading', { name: 'Immutable receipt and evidence handoff' }))
    .toBeVisible();
}

test('Platform & evidence explains every registered receipt field and refusal', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const trust = page.getByRole('region', { name: 'G17 experiment lineage and receipts' });
  await expect(trust).toBeVisible();
  await expect(trust).toContainText('verify_and_replay_bundle');
  await expect(trust).toContainText('automatic_evidence_admission');
  await trust.getByText('Every explained receipt field').click();
  for (const field of ['manifest', 'adapters', 'events', 'results', 'evidence_handoff']) {
    await expect(trust.getByText(field, { exact: true })).toBeVisible();
  }
});

test('deleting UI state and importing the bundle reconstructs the same completed run',
  async ({ page, request }) => {
    const exported = await exportedFixture(request);
    await openInterpret(page);
    await page.evaluate(() => localStorage.clear());
    await page.getByLabel('Choose experiment replay bundle').setInputFiles({
      name: `${exported.bundle_sha256}.json`,
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(exported.bundle)),
    });
    const verified = page.getByText(/VERIFIED: bundle/);
    await expect(verified).toBeVisible();
    await expect(verified).toContainText(exported.bundle.run_receipt.run_id);
    await expect(verified).toContainText('COMPLETE');
    await expect(page.getByText(/UI state discarded/)).toBeVisible();
  });

test('the reviewable handoff shows absences and opens a separate evidence draft',
  async ({ page, request }) => {
    const exported = await exportedFixture(request);
    await openInterpret(page);
    await page.getByLabel('Choose experiment replay bundle').setInputFiles({
      name: 'receipt.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(exported.bundle)),
    });
    await expect(page.getByText('admitted evidence: ABSENT')).toBeVisible();
    await expect(page.getByText('claim promotion: ABSENT')).toBeVisible();
    await page.getByRole('button', { name: 'Open a separate evidence-study draft' }).click();
    await expect(page.getByRole('heading', { name: 'Evidence record', exact: true })).toBeVisible();
    await expect(page.getByLabel('Study ID')).toHaveValue(exported.bundle.run_receipt.study_id);
  });
