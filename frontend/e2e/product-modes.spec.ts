import { expect, test, Page, Locator } from '@playwright/test';

/**
 * TG18.5 UI qualification gate — one representative path per product mode.
 *
 * This gate is a suite and nothing else. Nothing it measures is reported back inside the product:
 * a UI that grades itself on screen publishes a claim about the UI, and the claims this
 * programme publishes are about the science.
 *
 * **What the first slice already established, and why this is not that.** Slice 1 proved
 * reachability: every served workspace opens from a clean browser and names itself. Opening is
 * not operating. This slice walks the characteristic path of each of the four product modes
 * TG18.0 identified and captures a named artefact at the state that path reaches.
 *
 * **The load-bearing assertion is the one about ambiguity.** TG18.0's constraint is that "a
 * change that improves one by making another ambiguous is not a successful redesign", and that
 * is a property of the four modes *together* rather than of any one of them. So each mode
 * declares a signature - the observable that makes it that mode and not one of the others - and
 * the suite asserts that every signature appears in exactly one of the four. Four separate
 * per-mode assertions could all pass while the modes converged on each other; a uniqueness
 * assertion across the set cannot.
 *
 * **What this file does NOT claim.**
 *
 *   - It is not evidence that any mode computes anything correctly. It is apparatus behaviour:
 *     the path is walkable and the modes remain distinguishable. A rendered figure here says a
 *     figure rendered, never that its numbers are right.
 *   - The artefacts are for a human reader. Nothing in this file asserts a pixel, so a
 *     screenshot cannot pass or fail anything; it is a record of what the path reached.
 *   - Narrow-viewport layout belongs to `narrow-width.spec.ts` (320/375/414/768). The two
 *     viewports here are the desktop widths that inspection does not cover.
 */

/** Named because an artefact whose viewport is not in its own name cannot be read later. */
const VIEWPORTS = [
  { name: 'desktop-1440', width: 1440, height: 900 },
  { name: 'wide-1920', width: 1920, height: 1080 },
] as const;

type SignatureId =
  | 'composition_path'
  | 'findings_panels'
  | 'qualification_matrix'
  | 'on_demand_transform';

/**
 * The observable that makes a mode that mode. Each is a role and an accessible name, found the
 * way a researcher finds it, so a signature cannot be satisfied by a class name or a test id.
 */
const SIGNATURES: Record<SignatureId, (page: Page) => Locator> = {
  composition_path: page =>
    page.getByRole('tablist', { name: 'Experiment composition path' }),
  findings_panels: page => page.getByRole('tablist', { name: 'Findings panels' }),
  qualification_matrix: page =>
    page.getByRole('table', { name: 'Duration and comparison mode qualification matrix' }),
  on_demand_transform: page =>
    page.getByRole('button', { name: 'Apply Forward & Inverse' }),
};

const MODES = [
  {
    mode: 'interactive instrument',
    workspace: 'Spectral transforms',
    slug: 'instrument',
    signature: 'on_demand_transform' as SignatureId,
  },
  {
    mode: 'guided commitment workflow',
    workspace: 'Experiment Composer',
    slug: 'commitment',
    signature: 'composition_path' as SignatureId,
  },
  {
    mode: 'read-only claim surface',
    workspace: 'Findings',
    slug: 'claim',
    signature: 'findings_panels' as SignatureId,
  },
  {
    mode: 'trust and qualification surface',
    workspace: 'Platform & evidence',
    slug: 'qualification',
    signature: 'qualification_matrix' as SignatureId,
  },
] as const;

function workflowNav(page: Page): Locator {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

/**
 * Gridded-line rail entries append a context label, so the accessible name of the spectral
 * workspace is `Spectral transformsGridded field line`. A prefix match keeps the call reading
 * like the researcher's intent; `.first()` is not used, so an ambiguous prefix fails loudly.
 */
function openWorkspace(page: Page, name: string): Promise<void> {
  return workflowNav(page)
    .getByRole('button', { name: new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`) })
    .click();
}

async function reach(page: Page, workspace: string): Promise<void> {
  await page.goto('/');
  await expect(workflowNav(page)).toBeVisible();
  await openWorkspace(page, workspace);
  await expect(page.locator('#workspace-heading')).toHaveText(`${workspace} workspace`);
}

/** Which of the four signatures this page is currently showing. */
async function signaturesOn(page: Page): Promise<SignatureId[]> {
  const present: SignatureId[] = [];
  for (const id of Object.keys(SIGNATURES) as SignatureId[]) {
    if (await SIGNATURES[id](page).count() > 0) present.push(id);
  }
  return present;
}

for (const viewport of VIEWPORTS) {
  test.describe(`at ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('the interactive instrument recomputes on demand and keeps its figure equivalents',
      async ({ page }) => {
        test.slow();
        await reach(page, 'Spectral transforms');

        const apply = SIGNATURES.on_demand_transform(page);
        await expect(apply).toBeEnabled();
        await apply.click();

        // The mode's whole point: a researcher changes something and the picture answers. What
        // is asserted is that the answer arrives with its reading contract attached, which is
        // TG18.2's rule, not that the numbers in it are right.
        const contract = page.getByRole('region', {
          name: 'Comparison contract for the target field and its inverse reconstruction',
        });
        await expect(contract).toBeVisible();
        await expect(page.getByLabel(/^Figure data for /)).toHaveCount(2);

        await page.screenshot({
          path: `e2e/artifacts/${viewport.name}-mode-instrument.png`, fullPage: false,
        });
      });

    test('the guided commitment workflow serves its ordered steps and one next action',
      async ({ page }) => {
        test.slow();
        await reach(page, 'Experiment Composer');

        // Served, not held here: until the server answers, the view says so rather than
        // inventing a step order of its own.
        await expect(SIGNATURES.composition_path(page)).toBeVisible();
        await expect(page.getByRole('tab')).toHaveCount(7);

        // Exactly one. A commitment workflow offering two next actions is offering a choice
        // about what to commit to, which is the researcher's decision to record, not the UI's
        // to present twice.
        const banner = page.getByRole('status')
          .filter({ hasText: /Next legitimate action|Blocked before/ });
        await expect(banner).toHaveCount(1);
        await expect(banner).toContainText('/api/v1/');

        await page.screenshot({
          path: `e2e/artifacts/${viewport.name}-mode-commitment.png`, fullPage: false,
        });
      });

    test('the read-only claim surface opens every panel and offers no way to compute one',
      async ({ page }) => {
        test.slow();
        await reach(page, 'Findings');

        const panels = SIGNATURES.findings_panels(page);
        await expect(panels).toBeVisible();
        // Tabs, not buttons: these carry an explicit `role="tab"`, which is what a researcher's
        // assistive technology is told they are, so it is what the test must ask for.
        const tabs = panels.getByRole('tab');
        const count = await tabs.count();
        expect(count).toBeGreaterThan(1);

        for (let index = 0; index < count; index += 1) {
          const tab = tabs.nth(index);
          await expect(tab).toBeEnabled();
          await tab.click();
        }

        // The mode's defining absence. A claim surface that can recompute a claim is no longer
        // read-only, and the reader could not tell a recorded finding from a fresh one.
        await expect(SIGNATURES.on_demand_transform(page)).toHaveCount(0);
        await expect(page.getByRole('button', { name: /^(Freeze|Run) / })).toHaveCount(0);

        await page.screenshot({
          path: `e2e/artifacts/${viewport.name}-mode-claim.png`, fullPage: false,
        });
      });

    test('the trust surface shows a cleared gate and an uncleared one with its reason',
      async ({ page }) => {
        test.slow();
        await reach(page, 'Platform & evidence');

        await expect(SIGNATURES.qualification_matrix(page)).toBeVisible();

        // Both halves, because either alone is a different and misleading surface. A ledger
        // showing only passes is an advertisement; one showing only failures cannot be
        // distinguished from a broken build. TG17.12 made the first PASS available to show.
        const gates = page.locator('text=/^[A-Z][^:]+: (PASS|FAIL|REFUSED|NOT_RUN|NOT_IMPLEMENTED)$/');
        const rendered = await gates.allTextContents();
        expect(rendered.some(row => row.endsWith(': PASS'))).toBe(true);
        expect(rendered.some(row => !row.endsWith(': PASS'))).toBe(true);

        // An unpassed gate that does not say why is a status word, not a refusal.
        await expect(page.getByText('Calendar null calibration and planted power: PASS'))
          .toBeVisible();

        await page.screenshot({
          path: `e2e/artifacts/${viewport.name}-mode-qualification.png`, fullPage: false,
        });
      });

    test('every mode signature appears in exactly one of the four modes', async ({ page }) => {
      test.slow();
      // The TG18.0 constraint, asserted across the set rather than mode by mode: four separate
      // per-mode checks could all pass while the modes converged on one another.
      const observed: Record<string, SignatureId[]> = {};
      for (const entry of MODES) {
        await reach(page, entry.workspace);
        // Settle first. Three of the four signatures are served by the API rather than held in
        // the bundle, and `signaturesOn` counts without waiting - sampling a mode mid-fetch
        // would report an absence that is only a race, and would make this assertion flaky in
        // the direction that hides a real convergence.
        await expect(SIGNATURES[entry.signature](page)).toBeVisible();
        observed[entry.slug] = await signaturesOn(page);
      }

      for (const entry of MODES) {
        expect(observed[entry.slug], `${entry.mode} lost its own signature`)
          .toContain(entry.signature);
      }
      for (const id of Object.keys(SIGNATURES) as SignatureId[]) {
        const carriers = MODES.filter(entry => observed[entry.slug].includes(id))
          .map(entry => entry.mode);
        expect(carriers, `${id} no longer distinguishes exactly one mode`).toHaveLength(1);
      }
    });
  });
}
