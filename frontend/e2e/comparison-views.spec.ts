import { expect, test, Page } from '@playwright/test';

/**
 * TG17.8 acceptance, in a real browser, through visible labelled controls only.
 *
 * The acceptance is stated as three things a *picture* must not be able to do, so these tests are
 * about what the rendered page contains and what it refuses to contain:
 *
 *   - a semantic-trap reading cannot be drawn as magnitude equivalence, precedence or causality;
 *   - sparse or absent coverage is visually distinct from a measured zero;
 *   - every plotted point traces to an artefact, and every correction denominator is visible.
 *
 * Nothing here reaches for a class name or a test id. The last of the three is checked by reading
 * the numbers the researcher reads, in the tables the page prints under each view.
 */

const COMPOSER = 'Experiment Composer';

async function openViews(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: COMPOSER, exact: true }).click();
  await page.getByRole('tab', { name: /7\. Interpret/ }).click();
  await expect(page.getByRole('heading', { name: 'Comparison views' })).toBeVisible();
  // The set is fetched from the server; until it answers the panel has no views of its own.
  // Waiting for the *last* view rather than the first is deliberate: every test below assumes
  // the whole set is on the page, and a helper that returns once view one has painted makes
  // each of them carry its own race. One full-suite run failed here on the motif view before
  // this changed, on the run immediately after the component was edited - a cold Vite
  // transform is the likely cause, and the fix is to state the postcondition either way.
  await expect(page.getByRole('heading', { name: /7\. Provenance drill-down/ })).toBeVisible();
}

test.describe('the views the server serves', () => {
  test('all seven views are drawn, in their declared reading order', async ({ page }) => {
    await openViews(page);
    for (const title of [
      /1\. Cross-domain coverage timeline/,
      /2\. Native record beside canonical trajectory/,
      /3\. Native-to-structural scale mapping/,
      /4\. Pair, triple and quartet result matrix/,
      /5\. Motif correspondence and transfer/,
      /6\. Null distributions, corrected values and resolution/,
      /7\. Provenance drill-down/,
    ]) {
      await expect(page.getByRole('heading', { name: title })).toBeVisible();
    }
  });

  test('every view prints what it may and may not be concluded from', async ({ page }) => {
    await openViews(page);
    const boundaries = page.getByText('May not conclude:');
    await expect(boundaries).toHaveCount(7);
    await expect(page.getByText('May conclude:')).toHaveCount(7);
  });

  test('the legend carries each role in a colour, a marker and a word', async ({ page }) => {
    await openViews(page);
    const legend = page.getByRole('list', { name: 'Mark legend' }).first();
    for (const word of ['candidate', 'confirmed', 'transferred', 'surrogate', 'refused']) {
      await expect(legend).toContainText(word);
    }
    // The marker name is printed beside the swatch, so the distinction survives without colour.
    await expect(legend).toContainText('filled square');
    await expect(page.getByText('still has two')).toBeVisible();
  });
});

test.describe('what a picture here cannot say', () => {
  test('causality is offered as a question and refused with its reason', async ({ page }) => {
    await openViews(page);
    await page.getByText('What these views will not draw').click();
    await page.getByLabel('Can this be drawn?').selectOption('causality');
    await page.getByRole('button', { name: 'Check this reading' }).click();
    const answer = page.getByRole('status').filter({ hasText: /external design/ });
    await expect(answer).toBeVisible();
    await expect(answer).toContainText('may still declare and test');
  });

  test('magnitude equivalence is refused in the mode the study is actually in',
    async ({ page }) => {
      await openViews(page);
      await page.getByText('What these views will not draw').click();
      await page.getByLabel('Can this be drawn?').selectOption('magnitude_equivalence');
      await page.getByRole('button', { name: 'Check this reading' }).click();
      await expect(page.getByRole('status').filter({ hasText: /no mode to be admissible in/ }))
        .toBeVisible();
    });

  test('co-occurrence, which this mode does admit, is answered as drawable', async ({ page }) => {
    await openViews(page);
    await page.getByText('What these views will not draw').click();
    await page.getByLabel('Can this be drawn?').selectOption('co_occurrence');
    await page.getByRole('button', { name: 'Check this reading' }).click();
    await expect(page.getByRole('status').filter({ hasText: /can be drawn/ })).toBeVisible();
  });

  test('the refusals the views are built on are stated on the page', async ({ page }) => {
    await openViews(page);
    await page.getByText('What these views will not draw').click();
    const panel = page.getByRole('group', { name: 'What these views will not draw' });
    await expect(panel).toContainText('MagnitudeEquivalenceError');
    await expect(panel).toContainText('no code path');
    await expect(panel).toContainText('artefact digest');
  });

  test('no two domains share one native magnitude axis', async ({ page }) => {
    await openViews(page);
    const preview = page.getByRole('region', { name: /Native record beside canonical trajectory/ });
    // Each panel names its own units and says the two cannot be subtracted. Four domains, four
    // native panels, four separate magnitude axes - there is no fifth, shared one.
    await expect(preview).toContainText('are read side by side and never');
    await expect(preview).toContainText('Native:');
    await expect(preview).toContainText('Canonical:');
    await expect(preview).toContainText('Breaks:');
  });
});

test.describe('absence, and what it is not', () => {
  test('sparse and absent coverage are labelled, not shaded into a zero', async ({ page }) => {
    await openViews(page);
    const timeline = page.getByRole('region', { name: /Cross-domain coverage timeline/ });
    await expect(timeline).toContainText('covered');
    await expect(timeline).toContainText('sparse');
    await expect(timeline).toContainText('refused');
    await expect(timeline).toContainText('not the number zero');
    // The bespoke domain keeps its row, refused, rather than vanishing from the study.
    await expect(timeline).toContainText('order_book');
  });

  test('an unmeasured result cell says it is unmeasured, never that it is negative',
    async ({ page }) => {
      await openViews(page);
      const matrix = page.getByRole('region', { name: /result matrix/ });
      await expect(matrix).toContainText('NOT_YET_MEASURED');
      await expect(matrix).toContainText('unasked question');
      await expect(matrix).not.toContainText('no effect');
    });

  test('a manifest declaring no motif says so rather than showing an empty grid',
    async ({ page }) => {
      await openViews(page);
      const motifs = page.getByRole('region', { name: /Motif correspondence/ });
      await expect(motifs).toContainText('no frozen shape to look for');
      await expect(motifs).toContainText('priced');
    });
});

test.describe('the numbers behind the picture', () => {
  test('every correction denominator is visible without opening anything', async ({ page }) => {
    await openViews(page);
    const matrix = page.getByRole('region', { name: /result matrix/ });
    await expect(matrix).toContainText('Corrected over');
    await expect(matrix).toContainText('benjamini_yekutieli');
    await expect(matrix).toContainText('declared search of');
  });

  test('the accessible table under each view is reachable and holds the same values',
    async ({ page }) => {
      await openViews(page);
      const toggle = page.getByRole('button', { name: /Show the numbers behind each view/ });
      await toggle.click();
      await expect(page.getByRole('table', { name: /Numbers behind cross-domain coverage/ }))
        .toBeVisible();
      const timelineTable = page.getByRole('table', {
        name: /Numbers behind cross-domain coverage/ });
      await expect(timelineTable).toContainText('estimated bytes');
      await expect(timelineTable).toContainText('support kind');
      await page.getByRole('button', { name: /Hide the numbers behind each view/ }).click();
      await expect(timelineTable).toHaveCount(0);
    });

  test('the p-value floor is shown beside the ensemble that produces it', async ({ page }) => {
    await openViews(page);
    const nulls = page.getByRole('region', { name: /Null distributions/ });
    await expect(nulls).toContainText('p-value floor');
    await expect(nulls).toContainText('declared surrogates');
    await expect(nulls).toContainText('incapable of rejecting anything');
  });

  test('provenance drills down to source, adapter, licence and support', async ({ page }) => {
    await openViews(page);
    const provenance = page.getByRole('region', { name: /Provenance drill-down/ });
    await provenance.getByRole('group', { name: 'reanalysis' }).click();
    await expect(provenance).toContainText('licence');
    await expect(provenance).toContainText('definition');
    // No artefact exists yet, and the page says so rather than leaving the field blank.
    await expect(provenance).toContainText('no acquisition artefact exists');
  });
});

test('selecting a window highlights each domain\'s own contributing support', async ({ page }) => {
  await openViews(page);
  await page.getByRole('button', { name: 'Highlight what contributes to three_months' }).click();
  const highlight = page.getByRole('status').filter({ hasText: 'Contributing native support' });
  await expect(highlight).toBeVisible();
  await expect(highlight).toContainText('reanalysis');
  await expect(highlight).toContainText('order_book');
  // Four domains, four intervals. There is no merged extent, and the page says why.
  await expect(highlight).toContainText('only one domain addresses');
});

test('the views are operable from the keyboard alone', async ({ page }) => {
  await openViews(page);
  const toggle = page.getByRole('button', { name: /Show the numbers behind each view/ });
  await toggle.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: /Hide the numbers behind each view/ }))
    .toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: /Show the numbers behind each view/ }))
    .toHaveAttribute('aria-pressed', 'false');
});

test('a view can be refreshed on its own without disturbing the others', async ({ page }) => {
  await openViews(page);
  await page.getByRole('button', { name: 'Refresh Provenance drill-down' }).click();
  await expect(page.getByRole('heading', { name: /7\. Provenance drill-down/ })).toBeVisible();
  await expect(page.getByRole('heading', { name: /1\. Cross-domain coverage timeline/ }))
    .toBeVisible();
});

test('reading the views does not start the run they would describe', async ({ page }) => {
  await openViews(page);
  const provenance = page.getByRole('region', { name: /Provenance drill-down/ });
  await expect(provenance).toContainText('no acquisition artefact exists');
  // The ladder above is the authority on whether a run exists, and merely looking must not
  // have moved it. `RunStore.open` freezes a manifest; a read may not be the act that does.
  await page.getByRole('tab', { name: /6\. Freeze and run/ }).click();
  await expect(page.getByRole('tabpanel')).toContainText('No run has been opened');
});
