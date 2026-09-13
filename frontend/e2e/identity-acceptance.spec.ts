import { expect, test } from '@playwright/test';

/**
 * T4E.41: a bar that only exists in a file is the problem it was written to fix.
 *
 * These tests check the two things that make the panel worth having: that a condition nothing
 * has been addressed to renders as plainly as one that is met, and that what would NOT count
 * is on screen beside what would.
 */
test('the acceptance renders every condition and what would not discharge it', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const panel = page.getByRole('region', { name: 'T4E.8 acceptance' });
  await expect(panel).toBeVisible();

  await expect(panel.getByTestId('acceptance-verdict')).toHaveText('INSUFFICIENT_EVIDENCE');
  await expect(panel).toContainText('0 of 8 conditions met');
  await expect(panel).toContainText('code may accept: false');
  await expect(panel).toContainText('kind_recurrence');

  for (const id of ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']) {
    await expect(panel.getByRole('button', { name: new RegExp(`^${id}`) })).toBeVisible();
  }

  const exclusions = panel.getByTestId('acceptance-exclusions');
  await expect(exclusions).toContainText('finite deterministic majority');
  await expect(exclusions).toContainText('development-only pass');
});

test('a condition states what would license it and what would not', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const panel = page.getByRole('region', { name: 'T4E.8 acceptance' });

  await panel.getByRole('button', { name: /^C3/ }).click();
  await expect(panel).toContainText('Licensed by');
  await expect(panel).toContainText('Not sufficient');
  // The result this programme just produced is named on screen as not clearing the bar.
  await expect(panel).toContainText('deterministic majority over a finite census');
});

test('the record the bar was set against is verified on screen', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const panel = page.getByRole('region', { name: 'T4E.8 acceptance' });

  await expect(panel).toContainText('7 artefacts, all verified: true');
  await expect(panel).toContainText('t4e39_reference_holdout_evaluation.json');
  await expect(panel).toContainText('t4e8-spatial-audit-v3.json');
});

test('the bar can be signed where it is read, and refuses a signature it was not given', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const panel = page.getByRole('region', { name: 'T4E.8 acceptance' });
  const form = panel.getByTestId('acceptance-adopt-form');
  await expect(form).toBeVisible();

  // The declaration a signer affirms having read is reachable from the same panel.
  await panel.getByTestId('acceptance-full-text').locator('summary').click();
  await expect(panel).toContainText('What this declaration is not');
  await expect(panel).toContainText('It measures nothing, adjudicates no candidate');

  // Nothing is filled in for the signer. A pre-filled affirmation would be the instrument
  // affirming on a person's behalf, which is the one thing this panel must not do.
  await expect(form.getByRole('textbox', { name: 'Adopter name' })).toHaveValue('');
  await expect(form.getByRole('textbox', { name: 'Adopter role' })).toHaveValue('');
  await expect(form.getByRole('textbox', { name: 'Adoption affirmation' })).toHaveValue('');

  // Driven to a refusal on purpose: a browser test that signed this would forge a signature.
  await form.getByRole('textbox', { name: 'Adopter name' }).fill('Playwright');
  await form.getByRole('textbox', { name: 'Adopter role' }).fill('automated test');
  await form.getByRole('textbox', { name: 'Adoption affirmation' }).fill('sure, adopt it');
  await form.getByRole('button', { name: 'Sign the acceptance bar' }).click();

  await expect(panel.getByTestId('acceptance-refusal')).toContainText('affirmation');
  await expect(panel.getByTestId('acceptance-verdict')).toHaveText('INSUFFICIENT_EVIDENCE');
  await expect(panel).toContainText('the bar is DRAFTED_NOT_ADOPTED');
});

test('signing the bar is stated as fixing a standard, not as accepting anything', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const form = page.getByRole('region', { name: 'T4E.8 acceptance' })
    .getByTestId('acceptance-adopt-form');

  await expect(form).toContainText('It does not accept T4E.8');
  await expect(form).toContainText('meet a condition or reopen any spent population');
});

test('the signer is remembered between adoptions, and the affirmation never is', async ({ page }) => {
  await page.goto('/');
  // A signer who has adopted before: name and role restored, nothing else.
  await page.evaluate(() => localStorage.setItem(
    'spectralearth.signer.identity', JSON.stringify({ name: 'Ed Bentley', role: 'maintainer' })));
  await page.reload();
  await page.getByRole('button', { name: 'Platform & evidence', exact: true }).click();
  const form = page.getByRole('region', { name: 'T4E.8 acceptance' })
    .getByTestId('acceptance-adopt-form');

  await expect(form.getByRole('textbox', { name: 'Adopter name' })).toHaveValue('Ed Bentley');
  await expect(form.getByRole('textbox', { name: 'Adopter role' })).toHaveValue('maintainer');
  // The act itself is never restored: a remembered affirmation would be the browser affirming.
  await expect(form.getByRole('textbox', { name: 'Adoption affirmation' })).toHaveValue('');
  await expect(form.getByTestId('remember-signer')).toBeChecked();
  await expect(form).toContainText('The affirmation is never saved');

  // A refused attempt stores nothing new, so a failed signature cannot seed an identity.
  await form.getByRole('textbox', { name: 'Adoption affirmation' }).fill('nope');
  await form.getByRole('button', { name: 'Sign the acceptance bar' }).click();
  await expect(page.getByRole('region', { name: 'T4E.8 acceptance' })
    .getByTestId('acceptance-refusal')).toContainText('affirmation');
  const stored = await page.evaluate(
    () => localStorage.getItem('spectralearth.signer.identity'));
  expect(JSON.parse(stored ?? '{}')).toEqual({ name: 'Ed Bentley', role: 'maintainer' });
  expect(stored).not.toContain('affirmation');

  // Unticking forgets on this browser.
  await form.getByTestId('remember-signer').uncheck();
  await expect.poll(async () => page.evaluate(
    () => localStorage.getItem('spectralearth.signer.identity'))).toBeNull();
});
