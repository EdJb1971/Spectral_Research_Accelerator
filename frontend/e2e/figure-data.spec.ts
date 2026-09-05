import { expect, test, Page, Locator } from '@playwright/test';

/**
 * TG18.2 acceptance: every visual encoding has a text/table equivalent, in a real browser.
 *
 * The gridded panels carried an `sr-only` caption describing the *shape* of the data - "heat map
 * with 128 rows and 128 columns" - and nothing else. The values themselves were reachable only
 * through a Plotly hover tooltip: mouse-only, ephemeral, and absent from any exported or printed
 * copy. This suite drives both figure families with real backend data and checks the equivalent
 * against the figure it claims to equal.
 *
 * The load-bearing assertion is the agreement check. A text panel that renders plausible numbers
 * unrelated to the trace would satisfy every structural assertion here, so the exact sample is
 * read out of the live Plotly trace and compared with what the panel printed.
 */

const SIX_FIGURES = (value: number) => Number(value.toPrecision(6)).toString();

function workflowNav(page: Page): Locator {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function openWorkspace(page: Page, name: string) {
  await workflowNav(page).getByRole('button', { name: new RegExp(`^${name}`) }).click();
  await expect(page.locator('#workspace-heading')).toHaveText(`${name} workspace`);
}

/** Generate the deterministic analytical field the gridded panels operate on. */
async function generateField(page: Page) {
  await page.goto('/');
  await openWorkspace(page, 'Synthetic generator');
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });
}

function figureData(page: Page, name: string): Locator {
  return page.locator(`details[aria-label="Figure data for ${name}"]`);
}

test.describe('the heat map equivalent', () => {
  test('states units, support, colour range provenance and missingness', async ({ page }) => {
    await generateField(page);
    const panel = figureData(page, 'Generated Clean Field (F)');
    await expect(panel).toHaveCount(1);
    await panel.locator('summary').click();

    for (const label of ['Value units', 'Grid', 'Horizontal axis', 'Horizontal support',
      'Vertical axis', 'Vertical support', 'Colour range shown', 'Valid interior',
      'Missing samples']) {
      await expect(panel.getByText(label, { exact: true })).toBeVisible();
    }

    // Normalization is a claim about comparability, so the panel says where the limits came
    // from. A range derived from one panel cannot be read across two.
    await expect(panel).toContainText(/derived from this panel alone|supplied, shared across panels/);
    // Missingness is stated either way; silence would leave a gap indistinguishable from a
    // flat field.
    await expect(panel).toContainText(/samples are finite|are not finite and are drawn as gaps/);
  });

  test('the exact value it prints is the value the figure holds', async ({ page }) => {
    await generateField(page);
    const panel = figureData(page, 'Generated Clean Field (F)');
    await panel.locator('summary').click();

    const readout = panel.getByRole('status');
    for (const [row, column] of [[0, 0], [7, 11], [23, 5]]) {
      await panel.getByLabel(/^Row index/).fill(String(row));
      await panel.getByLabel(/^Column index/).fill(String(column));

      // Straight out of the live trace: this is the number the heat map is drawing.
      const encoded = await page.evaluate(({ r, c }) => {
        const div = document.getElementById('fig-clean-field') as any;
        return div?.data?.[0]?.z?.[r]?.[c];
      }, { r: row, c: column });
      expect(typeof encoded, `no trace sample at ${row},${column}`).toBe('number');

      await expect(readout).toContainText(SIX_FIGURES(encoded as number));
      await expect(readout).toContainText(`row ${row}, column ${column}`);
    }
  });

  test('an index outside the field refuses rather than inventing a sample', async ({ page }) => {
    await generateField(page);
    const panel = figureData(page, 'Generated Clean Field (F)');
    await panel.locator('summary').click();

    const rows = await page.evaluate(() => {
      const div = document.getElementById('fig-clean-field') as any;
      return div?.data?.[0]?.z?.length ?? 0;
    });
    expect(rows).toBeGreaterThan(0);

    // The control clamps to the field rather than accepting an index it cannot answer, so the
    // readout always describes a real sample.
    await panel.getByLabel(/^Row index/).fill(String(rows + 500));
    await expect(panel.getByLabel(/^Row index/)).toHaveValue(String(rows - 1));
    await expect(panel.getByRole('status')).toContainText(`row ${rows - 1}`);
  });

  test('it declines to derive a statistic, and says so', async ({ page }) => {
    await generateField(page);
    const panel = figureData(page, 'Generated Clean Field (F)');
    await panel.locator('summary').click();

    await expect(panel).toContainText('derives no summary statistic');
    // The refusal has to hold in the markup, not only in the sentence describing it.
    const labels = await panel.locator('dt').allInnerTexts();
    for (const forbidden of ['Mean', 'Median', 'Slope', 'Correlation', 'Standard deviation']) {
      expect(labels, `${forbidden} is a statistic this panel must not author`)
        .not.toContain(forbidden);
    }
  });

  test('a field too large to enumerate says that, rather than showing a partial table', async ({ page }) => {
    await generateField(page);
    const panel = figureData(page, 'Generated Clean Field (F)');
    await panel.locator('summary').click();
    await expect(panel).toContainText('are not tabulated cell by cell');
    await expect(panel.locator('table')).toHaveCount(0);
  });
});

test.describe('the line chart equivalent', () => {
  test.beforeEach(async ({ page }) => {
    await generateField(page);
    await openWorkspace(page, 'Boundary-condition lab');
    await page.getByRole('button', { name: 'Apply & Analyze Artefacts' }).click();
    // By role: the figure heading and the new table caption both carry this text, and a bare
    // text match would be ambiguous the moment the equivalent exists.
    await expect(page.getByRole('heading',
      { name: 'Artefact Gradients & Error Profiles by Boundary Distance' }))
      .toBeVisible({ timeout: 30_000 });
  });

  test('every point behind the figure is tabulated with its series and axes', async ({ page }) => {
    const panel = figureData(page, 'Artefact Gradients & Error Profiles by Boundary Distance');
    await panel.locator('summary').click();

    const table = panel.locator('table');
    await expect(table).toHaveCount(1);
    await expect(table.locator('thead th')).toHaveText([
      'series', 'index', 'Euclidean Distance from Boundary Grid Edge', 'Mean Value', 'drawn',
    ]);

    // Both series are present: a table showing one of two series would be a partial equivalent
    // presented as a complete one.
    await expect(table).toContainText('Mean Spatial Gradient Magnitude');
    await expect(table).toContainText('Mean Abs Error Profiles');

    // The equivalent is complete only if it holds one row per plotted point. Counting against
    // the live traces is what separates "a table exists" from "the table is the figure".
    const plotted = await page.evaluate(() => {
      const names = ['Mean Spatial Gradient Magnitude', 'Mean Abs Error Profiles'];
      let total = 0;
      for (const div of Array.from(document.querySelectorAll('.js-plotly-plot')) as any[]) {
        for (const trace of div?.data ?? []) {
          if (names.includes(trace.name)) total += Math.min(trace.x.length, trace.y.length);
        }
      }
      return total;
    });
    expect(plotted, 'neither trace is on the page').toBeGreaterThan(0);
    await expect(table.locator('tbody tr')).toHaveCount(plotted);
  });

  test('it states the axis scale and what that scale drops', async ({ page }) => {
    const panel = figureData(page, 'Artefact Gradients & Error Profiles by Boundary Distance');
    await panel.locator('summary').click();

    await expect(panel.getByText('Horizontal axis', { exact: true })).toBeVisible();
    await expect(panel.getByText('Vertical axis', { exact: true })).toBeVisible();
    await expect(panel.getByText('Points drawn', { exact: true })).toBeVisible();
    // Linear here, so the parenthetical must say linear rather than nothing: an unstated scale
    // is how a log plot gets read as a linear one.
    await expect(panel).toContainText('(linear)');
    await expect(panel).toContainText('derives no summary statistic');
  });

  test('the values it prints are the values the figure plots', async ({ page }) => {
    const panel = figureData(page, 'Artefact Gradients & Error Profiles by Boundary Distance');
    await panel.locator('summary').click();

    const first = await page.evaluate(() => {
      const divs = Array.from(document.querySelectorAll('.js-plotly-plot')) as any[];
      for (const div of divs) {
        const trace = div?.data?.find((t: any) => t.name === 'Mean Spatial Gradient Magnitude');
        if (trace) return { x: trace.x[0], y: trace.y[0] };
      }
      return null;
    });
    expect(first, 'the gradient trace is not on the page').not.toBeNull();

    const firstRow = panel.locator('tbody tr').first();
    await expect(firstRow).toContainText('Mean Spatial Gradient Magnitude');
    await expect(firstRow).toContainText(SIX_FIGURES(first!.x as number));
    await expect(firstRow).toContainText(SIX_FIGURES(first!.y as number));
  });
});

test('the data panels are named, so a disclosure is not announced as a bare triangle', async ({ page }) => {
  await generateField(page);
  const panels = page.locator('details.figure-data');
  const count = await panels.count();
  expect(count, 'no figure data panel rendered at all').toBeGreaterThan(0);
  for (let index = 0; index < count; index += 1) {
    const name = await panels.nth(index).getAttribute('aria-label');
    expect(name && name.trim().length, 'a figure data disclosure has no accessible name')
      .toBeTruthy();
  }
});
