import { expect, test, Page, Locator } from '@playwright/test';

/**
 * TG18.2 acceptance: comparability is a stated contract, not a colour-scale convenience.
 *
 * Two heat maps side by side are an invitation to compare them, and Plotly autoscales each panel
 * to its own extremes unless told otherwise. Before this slice `zRange` was supplied at exactly
 * one call site in the frontend, so the original-versus-inverse-reconstruction pair - which *is*
 * an error judgement - was drawn on two independent scales, leaving any amplitude loss visible
 * only in two small colour-bar ranges.
 *
 * The suite has two halves for a stated reason. The decision function is exercised directly
 * because several of its branches - mismatched units, mismatched quantity, an undeclared
 * relationship - are not reachable through the current UI, where every declared pair happens to
 * agree. Testing them only through the screen would leave the refusals unverified until some
 * future call site needed them, which is when a silent wrong answer would be most expensive. The
 * rendered half then checks that the decision actually reaches the figures.
 *
 * The direct half imports the module through the Vite dev server that `playwright.config.ts`
 * starts, so it runs the same source the page runs, with no second toolchain and no build step of
 * its own.
 */

const MODULE = '/src/components/FigureComparison.tsx';

type Contract = {
  scale: { shared: boolean; range?: [number, number]; reason: string };
  cells: { linked: boolean; rows?: number; columns?: number; reason: string };
};

async function decide(page: Page, panels: unknown[]): Promise<Contract> {
  return page.evaluate(async ({ modulePath, input }) => {
    const module = await import(/* @vite-ignore */ modulePath);
    return module.buildComparisonContract(input);
  }, { modulePath: MODULE, input: panels });
}

const FIELD_A = [[0, 1], [2, 3]];
const FIELD_B = [[0, 5], [-2, 1]];

test.describe('the decision, and the reasons it gives', () => {
  test.beforeEach(async ({ page }) => { await page.goto('/'); });

  test('same quantity and units share one range spanning every panel', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, units: 'K', quantity: 'temperature' },
      { key: 'b', title: 'B', data: FIELD_B, units: 'K', quantity: 'temperature' },
    ]);
    expect(contract.scale.shared).toBe(true);
    // The union, not one panel's extremes: a shared range that clipped the other panel would
    // hide exactly the difference it was adopted to reveal.
    expect(contract.scale.range).toEqual([-2, 5]);
    expect(contract.scale.reason).toContain('temperature');
    expect(contract.cells.linked).toBe(true);
  });

  test('different units refuse, because raw magnitudes never share an axis', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, units: 'K', quantity: 'temperature' },
      { key: 'b', title: 'B', data: FIELD_B, units: 'degC', quantity: 'temperature' },
    ]);
    expect(contract.scale.shared).toBe(false);
    expect(contract.scale.reason).toContain('different units');
    expect(contract.scale.reason).toContain('K');
    expect(contract.scale.reason).toContain('degC');
  });

  test('different quantities refuse, because there is no shared meaning to normalise', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, units: null, quantity: 'temperature' },
      { key: 'b', title: 'B', data: FIELD_B, units: null, quantity: 'vorticity' },
    ]);
    expect(contract.scale.shared).toBe(false);
    expect(contract.scale.reason).toContain('different quantities');
  });

  test('an undeclared relationship refuses rather than assuming one', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A },
      { key: 'b', title: 'B', data: FIELD_B },
    ]);
    // The safe default. Two unitless fields tell us nothing by being equally unitless, so a pair
    // becomes comparable only through an explicit claim in the source.
    expect(contract.scale.shared).toBe(false);
    expect(contract.scale.reason).toContain('does not declare which quantity');
  });

  test('a panel with no finite sample refuses, having no limits to share', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, quantity: 'q' },
      { key: 'b', title: 'B', data: [[null, null], [null, null]], quantity: 'q' },
    ]);
    expect(contract.scale.shared).toBe(false);
    expect(contract.scale.reason).toContain('no finite sample');
  });

  test('scale and cell correspondence are decided separately', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, quantity: 'q' },
      { key: 'b', title: 'B', data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], quantity: 'q' },
    ]);
    // Differing shapes do not prevent a shared colour range - the magnitudes are still the same
    // quantity - but they do prevent one index naming the same sample in both.
    expect(contract.scale.shared).toBe(true);
    expect(contract.cells.linked).toBe(false);
    expect(contract.cells.reason).toContain('differ in shape');
  });

  test('a lone panel is not silently declared comparable', async ({ page }) => {
    const contract = await decide(page, [
      { key: 'a', title: 'A', data: FIELD_A, units: 'K', quantity: 'temperature' },
    ]);
    expect(contract.scale.shared).toBe(false);
    expect(contract.cells.linked).toBe(false);
  });
});

function workflowNav(page: Page): Locator {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function generateField(page: Page) {
  await page.goto('/');
  await workflowNav(page).getByRole('button', { name: /^Synthetic generator/ }).click();
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });
}

test.describe('the decision reaching the figures', () => {
  test('the contract is stated where the pictures are, without opening anything', async ({ page }) => {
    await generateField(page);
    const notice = page.locator('section.figure-comparison').first();
    // Never inside a disclosure: it governs how the panels below may be read, so a reader who
    // opens nothing still gets it.
    await expect(notice).toBeVisible();
    await expect(notice).toContainText('Shared colour range');
    await expect(notice).toContainText('Cell addressing is linked');
  });

  test('the shared range is actually applied to every trace', async ({ page }) => {
    await generateField(page);
    const applied = await page.evaluate(() => ['fig-clean-field', 'fig-perturbed-field'].map((id) => {
      const div = document.getElementById(id) as any;
      return { id, zmin: div?.data?.[0]?.zmin, zmax: div?.data?.[0]?.zmax };
    }));
    for (const trace of applied) {
      expect(typeof trace.zmin, `${trace.id} has no explicit lower limit`).toBe('number');
      expect(typeof trace.zmax, `${trace.id} has no explicit upper limit`).toBe('number');
    }
    // Identical limits are the whole point: a difference between the pictures is now a
    // difference between the fields rather than a difference between two autoscales.
    expect(applied[0].zmin).toBe(applied[1].zmin);
    expect(applied[0].zmax).toBe(applied[1].zmax);
  });

  test('the reconstruction pair no longer autoscales independently', async ({ page }) => {
    await generateField(page);
    await workflowNav(page).getByRole('button', { name: /^Spectral transforms/ }).click();
    await page.getByRole('button', { name: 'Apply Forward & Inverse' }).click();
    await expect(page.locator('#fig-reconstructed-field.js-plotly-plot'))
      .toBeVisible({ timeout: 60_000 });

    const applied = await page.evaluate(() => ['fig-target-field', 'fig-reconstructed-field']
      .map((id) => {
        const div = document.getElementById(id) as any;
        return { id, zmin: div?.data?.[0]?.zmin, zmax: div?.data?.[0]?.zmax };
      }));
    expect(typeof applied[0].zmin).toBe('number');
    expect(applied[0].zmin).toBe(applied[1].zmin);
    expect(applied[0].zmax).toBe(applied[1].zmax);
  });

  test('one address inspects every linked panel at once', async ({ page }) => {
    await generateField(page);
    const panels = page.locator('details.figure-data');
    await expect(panels).toHaveCount(2);
    await panels.nth(0).locator('summary').click();
    await panels.nth(1).locator('summary').click();

    await panels.nth(0).getByLabel(/^Row index/).fill('9');
    await panels.nth(0).getByLabel(/^Column index/).fill('4');

    // The second panel follows without being touched. Linked addressing is the readable form of
    // the same claim the shared scale makes: these two grids correspond.
    await expect(panels.nth(1).getByLabel(/^Row index/)).toHaveValue('9');
    await expect(panels.nth(1).getByLabel(/^Column index/)).toHaveValue('4');
    await expect(panels.nth(1).getByRole('status')).toContainText('row 9, column 4');

    // And it is symmetric: driving the second panel moves the first.
    await panels.nth(1).getByLabel(/^Row index/).fill('2');
    await expect(panels.nth(0).getByLabel(/^Row index/)).toHaveValue('2');
  });

  test('a pair of differing shapes keeps its shared scale but refuses linked addressing', async ({ page }) => {
    await generateField(page);
    await workflowNav(page).getByRole('button', { name: /^Boundary-condition lab/ }).click();
    await page.getByRole('button', { name: 'Apply & Analyze Artefacts' }).click();
    await expect(page.getByRole('heading',
      { name: 'Artefact Gradients & Error Profiles by Boundary Distance' }))
      .toBeVisible({ timeout: 30_000 });

    const notice = page.locator('section.figure-comparison').first();
    await expect(notice).toContainText('Shared colour range');
    // Padding changes the grid, so one index no longer names the same sample. The contract says
    // which of the two claims survives rather than quietly dropping both or keeping both.
    await expect(notice).toContainText('Cell addressing is not linked');
    await expect(notice).toContainText('differ in shape');
  });
});
