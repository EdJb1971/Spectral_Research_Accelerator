import { expect, test, Page } from '@playwright/test';

/**
 * TG17.7 acceptance, in a real browser, through visible labelled controls only.
 *
 * Nothing here reaches for a CSS class, a test id or an internal state hook. Every step is found
 * the way a researcher finds it - by its role and its visible name - because a test that clicks
 * `.btn-primary` proves the DOM has a div, not that a person could declare an experiment.
 *
 * The manifest driven here is the TG17.0 flagship with `order_book` removed **through the
 * domain menu**, which is itself part of the acceptance: the bespoke domain has no public
 * archive, so metadata cannot plan its coverage and the whole flagship blocks at preflight by
 * design. The interesting path is therefore the one where the researcher meets that refusal, is
 * told which domain and why, and resolves it with a visible control.
 */

const COMPOSER = 'New experiment';

async function openComposer(page: Page) {
  await page.goto('/');
  await page.getByRole('button', { name: COMPOSER, exact: true }).click();
  // `exact`, because `name` is a substring match and the shell's own sr-only workspace heading
  // reads "Experiment Composer workspace". Both are headings, so the unqualified locator
  // resolves to two the moment the panel renders and strict mode fails the run - intermittently,
  // since which of the two exists first depends on when the served path answers. Found by the
  // TG18.5 full-suite measurement, where it failed once and passed on isolated re-run.
  await expect(page.getByRole('heading', { name: COMPOSER, exact: true })).toBeVisible();
  // The path is fetched from the server; until it answers, the view says so rather than
  // guessing at a step order of its own.
  await expect(page.getByRole('tablist', { name: 'Experiment composition path' })).toBeVisible();
}

async function step(page: Page, name: string) {
  await page.getByRole('tab', { name }).click();
}

test.describe('the guided path', () => {
  test('the seven steps are served, ordered and all reachable', async ({ page }) => {
    await openComposer(page);
    const tabs = page.getByRole('tab');
    await expect(tabs).toHaveCount(7);
    await expect(tabs).toHaveText([
      /1\. Question/, /2\. Domains/, /3\. Observation/, /4\. Preflight/,
      /5\. Analysis/, /6\. Freeze and run/, /7\. Interpret/,
    ]);
    // A blocked step keeps its tab. Hiding it would hide the reason it is blocked, which is
    // the part a researcher needs.
    for (const name of [/4\. Preflight/, /7\. Interpret/]) {
      await expect(page.getByRole('tab', { name })).toBeEnabled();
    }
  });

  test('exactly one next action is offered, and it names its own route', async ({ page }) => {
    await openComposer(page);
    const banner = page.getByRole('status').filter({ hasText: /Next legitimate action|Blocked before/ });
    await expect(banner).toHaveCount(1);
    await expect(banner).toContainText('/api/v1/');
    await banner.getByRole('button', { name: 'Go to step' }).click();
    // Following the named action lands on the step that owns it, not on a general page.
    await expect(page.getByRole('tab', { selected: true })).toBeVisible();
  });

  test('the path is operable from the keyboard alone', async ({ page }) => {
    await openComposer(page);
    await page.getByRole('tab', { name: /1\. Question/ }).focus();
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/1\. Question/);
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/2\. Domains/);
    await expect(page.getByRole('tabpanel')).toContainText('what does each one break');
    await page.keyboard.press('End');
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/7\. Interpret/);
    await page.keyboard.press('Home');
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/1\. Question/);
  });

  test('the researcher keeps their place across a browser refresh', async ({ page }) => {
    await openComposer(page);
    await step(page, /5\. Analysis/);
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/5\. Analysis/);
    await page.reload();
    await page.getByRole('button', { name: COMPOSER, exact: true }).click();
    await expect(page.getByRole('tab', { selected: true })).toHaveText(/5\. Analysis/);
  });
});

test.describe('what the path refuses to hide', () => {
  test('the flagship blocks at preflight, naming the domain and the reason', async ({ page }) => {
    await openComposer(page);
    await step(page, /4\. Preflight/);
    await page.getByRole('button', { name: 'Inspect metadata coverage' }).click();
    const panel = page.getByRole('tabpanel');
    await expect(panel).toContainText('order_book');
    await expect(panel).toContainText(/no public archive|content-addressed local record/);
    // The refusal is visible, and the domain has not quietly vanished from the study.
    await expect(panel).toContainText('REFUSED');
  });

  test('every domain shows what it breaks before it is chosen', async ({ page }) => {
    await openComposer(page);
    await step(page, /2\. Domains/);
    const panel = page.getByRole('tabpanel');
    await expect(panel).toContainText('Breaks:');
    await expect(panel).toContainText('irregular_sampling');
    await expect(panel.getByRole('checkbox')).toHaveCount(4);
    for (const domain of ['reanalysis', 'argo_float', 'tess_lightcurve', 'order_book']) {
      await expect(panel.getByRole('checkbox', { name: `Include ${domain}` })).toBeVisible();
    }
  });

  test('the ladder separates a run from a finding and from admitted evidence', async ({ page }) => {
    await openComposer(page);
    await step(page, /7\. Interpret/);
    const panel = page.getByRole('tabpanel');
    for (const rung of ['Acquired material', 'Executed run', 'Finding', 'Admitted evidence']) {
      await expect(panel).toContainText(rung);
    }
    await expect(panel).toContainText('Downloading four archives is not an experiment about them');
    await expect(panel).toContainText('findings instrument');
  });

  test('an empty panel reads as an unasked question, never as a clean result', async ({ page }) => {
    await openComposer(page);
    await step(page, /7\. Interpret/);
    await expect(page.getByRole('tabpanel')).toContainText('not a null result');
    await step(page, /4\. Preflight/);
    await expect(page.getByRole('tabpanel')).toContainText('an unasked question, not a clean');
  });

  test('a destructive action asks a second time and says what is lost', async ({ page }) => {
    await openComposer(page);
    await step(page, /1\. Question/);
    const panel = page.getByRole('tabpanel');
    await panel.getByRole('button', { name: /Discard this browser/ }).click();
    const dialog = page.getByRole('alertdialog');
    await expect(dialog).toContainText('Saved revisions are immutable');
    await dialog.getByRole('button', { name: 'Keep it' }).click();
    await expect(dialog).toHaveCount(0);
  });
});

test('the whole executable plan is declared and run through visible controls', async ({ page }) => {
  test.slow();
  await openComposer(page);

  // 2. Drop the bespoke domain that metadata cannot plan for. This is the researcher meeting
  //    the preflight refusal and answering it, rather than the UI hiding the domain.
  await step(page, /2\. Domains/);
  await page.getByRole('checkbox', { name: 'Include order_book' }).uncheck();
  await expect(page.getByRole('tabpanel')).toContainText('3 of at least 2 domains selected');

  // 3. Apply a duration preset. What is applied is the pair of instants the server resolved.
  await step(page, /3\. Observation/);
  await page.getByRole('button', { name: 'Apply the week preset' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'The preset is a label' })).toBeVisible();

  // 4. Preflight now answers rather than refusing.
  await step(page, /4\. Preflight/);
  await page.getByRole('button', { name: 'Inspect metadata coverage' }).click();
  await expect(page.getByRole('tabpanel')).not.toContainText('order_book');
  await expect(page.getByRole('tab', { name: /4\. Preflight/ })).toBeVisible();

  // 5. Price the complete declared family, before anything is acquired.
  await step(page, /5\. Analysis/);
  await page.getByRole('button', { name: 'Price the declared family' }).click();
  await expect(page.getByRole('status').filter({ hasText: /family is priced|cannot reject anything/ }))
    .toBeVisible();

  // 6. Read the plan back in its own sentences, then open the run at the manifest's address.
  await step(page, /6\. Freeze and run/);
  await page.getByRole('button', { name: 'Render the preregistration summary' }).click();
  const summary = page.getByRole('tabpanel');
  await expect(summary).toContainText('A frozen plan is never edited after seeing how it went');
  await expect(summary).toContainText('benjamini_yekutieli');

  await page.getByRole('button', { name: 'Open or resume the run' }).click();
  await expect(page.getByRole('status').filter({ hasText: /Opened run|Resumed run/ })).toBeVisible();

  // The rehearsal suite must say, in the browser, that it acquires nothing.
  await expect(summary).toContainText('rehearsal, not data');

  await page.getByRole('button', { name: 'Execute the frozen plan' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'not evidence' })).toBeVisible();

  // 7. A completed run moves two rungs of the ladder and no more.
  await step(page, /7\. Interpret/);
  const interpret = page.getByRole('tabpanel');
  await expect(interpret).toContainText('COMPLETE');
  await expect(interpret).toContainText('A completed run is an executed plan, not admitted evidence');
});

test('re-opening the same plan resumes the same run rather than starting a second', async ({ page }) => {
  test.slow();
  await openComposer(page);
  await step(page, /2\. Domains/);
  await page.getByRole('checkbox', { name: 'Include order_book' }).uncheck();

  await step(page, /6\. Freeze and run/);
  await page.getByRole('button', { name: 'Open or resume the run' }).click();
  const first = page.getByRole('status').filter({ hasText: /Opened run|Resumed run/ });
  await expect(first).toBeVisible();
  const firstText = await first.innerText();
  const runId = firstText.match(/run ([0-9a-f]{32})/)?.[1];
  expect(runId, 'the message must name the run identity').toBeTruthy();

  // A browser refresh is the case this whole guarantee exists for.
  await page.reload();
  await page.getByRole('button', { name: COMPOSER, exact: true }).click();
  await step(page, /2\. Domains/);
  await page.getByRole('checkbox', { name: 'Include order_book' }).uncheck();
  await step(page, /6\. Freeze and run/);
  await page.getByRole('button', { name: 'Open or resume the run' }).click();
  await expect(page.getByRole('status').filter({ hasText: `Resumed run ${runId}` })).toBeVisible();
});
