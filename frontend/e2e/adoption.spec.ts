import { expect, test, Page } from '@playwright/test';

/**
 * T4E.32/T4E.33 acceptance, in a real browser: signing is an act a person performs here.
 *
 * The rule is unchanged -- code does not sign a scientific declaration for a person -- and these
 * tests are mostly about what the surface refuses, because that is where the rule now lives. A
 * maintainer who reads a declaration, types their name and types the affirmation has signed it;
 * a form that filled any of that in would be signing on their behalf while appearing to ask.
 *
 * Nothing here signs anything. Every test drives the form to a REFUSAL, because a passing test
 * that wrote an adoption into the calibration store would be a test that forged a signature.
 */

async function openAdoption(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: /^Adopt declarations/ }).click();
  await expect(page.getByTestId('adoption')).toBeVisible();
}

test('an unsigned declaration offers to be adopted and a signed one names who signed it',
  async ({ page }) => {
    await openAdoption(page);

    await expect(page.getByTestId('sign-t4e28-join-rerun-declaration.json')).toBeVisible();
    await expect(page.getByTestId('adopted-t4e27-position-tolerance-declaration.json'))
      .toContainText('adopted by');
  });

test('the digest being signed is shown beside the form', async ({ page }) => {
  await openAdoption(page);
  await page.getByTestId('sign-t4e28-join-rerun-declaration.json').click();

  // A signature that reaches a different text than the one on screen is the failure the binding
  // exists to prevent.
  await expect(page.getByTestId('signing-digest')).toContainText('binds sha256');
});

test('the form supplies no default for the name, the reason or the affirmation',
  async ({ page }) => {
    await openAdoption(page);
    await page.getByTestId('sign-t4e28-join-rerun-declaration.json').click();

    await expect(page.getByTestId('adopted-by')).toHaveValue('');
    await expect(page.getByTestId('affirmation')).toHaveValue('');
    await expect(page.getByTestId('why')).toHaveValue('');
  });

test('a wrong affirmation is refused and nothing is written', async ({ page }) => {
  await openAdoption(page);
  await page.getByTestId('sign-t4e28-join-rerun-declaration.json').click();
  await page.getByTestId('adopted-by').fill('A Real Person');
  await page.getByTestId('adopted-as').fill('A_TEST');
  await page.getByTestId('what-was-adopted').fill('the declaration as written');
  await page.getByTestId('affirmation').fill('yes I adopt it');
  await page.getByTestId('sign-submit').click();

  await expect(page.getByTestId('sign-refusal')).toContainText('typed exactly');
  await expect(page.getByTestId('sign-done')).toHaveCount(0);
});

test('a placeholder name is refused as not being a person', async ({ page }) => {
  await openAdoption(page);
  await page.getByTestId('sign-t4e28-join-rerun-declaration.json').click();
  await page.getByTestId('adopted-by').fill('maintainer');
  await page.getByTestId('adopted-as').fill('A_TEST');
  await page.getByTestId('what-was-adopted').fill('the declaration as written');
  await page.getByTestId('affirmation').fill('I have read this declaration and I adopt it');
  await page.getByTestId('sign-submit').click();

  await expect(page.getByTestId('sign-refusal')).toContainText('is not a person');
});

test('the convening control states the cost before anything can be spent', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /^Recorded review/ }).click();
  await page.getByLabel('Published study ID').fill('t4e28-join-rerun');
  await page.getByRole('button', { name: 'Reload exact revision' }).click();

  await expect(page.getByTestId('call-count')).toContainText('8 paid calls');
  await expect(page.getByTestId('seat-candidate_synthesis')).toBeVisible();
  // The authorisation is a separate control, so the run button alone cannot spend anything.
  await expect(page.getByTestId('convene')).toBeDisabled();
  await page.getByTestId('authorise').check();
  await expect(page.getByTestId('convene')).toBeEnabled();
});

test('convening without a key on the server refuses and says nothing was sent',
  async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /^Recorded review/ }).click();
    await page.getByLabel('Published study ID').fill('t4e28-join-rerun');
    await page.getByRole('button', { name: 'Reload exact revision' }).click();
    await page.getByTestId('authorise').check();
    await page.getByTestId('convene').click();

    await expect(page.getByTestId('convene-refusal')).toContainText('Nothing was sent');
  });
