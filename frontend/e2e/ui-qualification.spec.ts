import { expect, test, Page } from '@playwright/test';

/**
 * TG18.5 UI qualification gate — served-workspace reachability.
 *
 * This gate is a test suite and nothing else. No part of it is reported back inside the product:
 * a UI that grades itself on screen is a claim about the UI, and the only claims this programme
 * publishes are about the science.
 *
 * The bounded claim: every workspace the shell serves opens from a clean browser and names
 * itself, and nothing opens under a name the inventory does not contain. That is apparatus
 * reachability. It is not evidence that any workspace computes anything correctly, and it must
 * never be read as one.
 */

/**
 * The inventory is written out rather than derived from the page, so that a workspace silently
 * disappearing fails here instead of quietly shrinking a self-derived list to match itself.
 * `test_frontend_contract.py` holds the other end against `WORKFLOW_NAV` in App.tsx.
 */
const SERVED_WORKSPACES = [
  'Acquire data',
  'Cross-domain analysis',
  'Preregistration',
  'Synthetic generator',
  'Meteorological data',
  'Boundary-condition lab',
  'Spectral transforms',
  'Diagnostics',
  'Structure mining',
  'Cross-domain record',
  'Automated hypotheses',
  'Evidence record',
  'Experiment Composer',
  'Legacy parameter sweeps',
  'Forecast evaluation',
  'Recorded review',
  'Atmospheric gate record',
  'Identity declaration',
  'Study trail',
  'Position tolerance',
  'Join distribution',
  'Research archive',
  'Findings',
  'Platform & evidence',
] as const;

/** App.tsx falls back to this when an active tab matches no navigation entry. */
const UNINVENTORIED_FALLBACK = 'Scientific workbench';

function workflowNav(page: Page) {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

/**
 * The primary label only. A legacy entry's accessible name also carries its "Legacy · ..." line,
 * which TG18.3 requires; matching the whole name here would couple reachability to that wording.
 */
function workspaceEntries(page: Page) {
  return workflowNav(page).getByRole('button');
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(workflowNav(page)).toBeVisible();
});

test('the shell serves exactly the qualified workspace inventory, in order', async ({ page }) => {
  const labels = await workspaceEntries(page).evaluateAll(
    nodes => nodes.map(node => node.querySelector('span.block')?.textContent?.trim() ?? ''));

  expect(labels).toEqual([...SERVED_WORKSPACES]);
});

test('every served workspace opens from a clean browser and names itself', async ({ page }) => {
  test.slow();
  const heading = page.locator('#workspace-heading');
  const entries = workspaceEntries(page);
  const reached: string[] = [];

  for (let index = 0; index < SERVED_WORKSPACES.length; index += 1) {
    const expected = SERVED_WORKSPACES[index];
    const entry = entries.nth(index);

    await entry.click();
    await expect(heading).toHaveText(`${expected} workspace`);
    // Reaching a workspace and being told you are there are separate facts.
    await expect(entry).toHaveAttribute('aria-current', 'page');
    reached.push(expected);
  }

  expect(reached).toEqual([...SERVED_WORKSPACES]);
});

test('every journey destination lands on an inventoried workspace', async ({ page }) => {
  // The journey holds its own workspace identifiers, separate from WORKFLOW_NAV. A rename on
  // either side is a navigation that silently arrives at App.tsx's fallback name instead of the
  // workspace the researcher asked for, and the shell would still look fine.
  const heading = page.locator('#workspace-heading');
  const stages = page.getByRole('navigation', { name: 'Guided research journey' })
    .getByRole('listitem');
  const stageCount = await stages.count();
  expect(stageCount).toBe(7);
  const landed: string[] = [];

  for (let index = 0; index < stageCount; index += 1) {
    await page.goto('/');
    // A blocked stage offers its remediation instead of its own action; both are destinations.
    const action = stages.nth(index).getByRole('button').first();
    await action.click();

    const text = (await heading.textContent())?.replace(/ workspace$/, '') ?? '';
    expect(text).not.toBe(UNINVENTORIED_FALLBACK);
    expect(SERVED_WORKSPACES).toContain(text);
    landed.push(text);
  }

  expect(landed).toHaveLength(7);
});


test('a clean browser disables no workspace, and a disabled one would carry its reason', async ({ page }) => {
  const entries = workspaceEntries(page);
  const count = await entries.count();
  expect(count).toBe(SERVED_WORKSPACES.length);

  for (let index = 0; index < count; index += 1) {
    await expect(entries.nth(index)).toBeEnabled();
  }

  // Unreachability is only legitimate when it is explained. With no capability profile selected
  // nothing may be disabled, so the described-by channel is asserted structurally: no entry
  // claims a reason it is not entitled to.
  const described = await entries.evaluateAll(
    nodes => nodes.filter(node => node.getAttribute('aria-describedby')).length);
  expect(described).toBe(0);
});
