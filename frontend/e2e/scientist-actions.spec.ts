import { expect, test, Page, Locator } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';


/**
 * TG18.5 UI qualification gate — the two numbers `scientist_actions` refuses to invent.
 *
 * TG17.10 shipped `NOT_RELEASEABLE` with `scientist_actions` reading `NOT_MEASURED` for the
 * action count and for `refusal_explanation_time_seconds`, because a deterministic backend
 * rehearsal cannot observe a researcher and must not award itself someone else's evidence.
 * This file measures both in a rendered browser, and is careful about which of them is the
 * kind of thing an assertion may hold.
 *
 * **An action count is deterministic, so it is asserted.** Every action below is one activation
 * of one visible control, found by role and accessible name. Change the path and the count
 * changes and the assertion fails, which is the point: the number is a property of the route a
 * researcher walks, and it is not allowed to drift silently.
 *
 * **A wall-clock duration is not, so it is recorded and never asserted.** How long a refusal
 * takes to explain itself on a loaded machine is a property of that machine. Asserting it would
 * make the gate fail for reasons that have nothing to do with the interface, and - worse - would
 * make it pass on a fast machine while the interface got slower. It is written into the
 * measurement as context a human can read, explicitly marked as unasserted.
 *
 * **What the count is, and what it is not.** It is the number of actions on the *declared
 * representative path*, enumerated in this file. It is an upper bound on the shortest route and
 * not a claim about the minimum: another researcher could reach the same state differently.
 * The measurement says so in its own words rather than leaving a reader to assume otherwise.
 *
 * `adapter_specific_framework_edits`, the third `NOT_MEASURED` field, is deliberately not
 * touched here. It is a source-edit audit belonging to the `synthetic_fifth_adapter` gate, and
 * no browser can observe it.
 */

const COMPOSER = 'New experiment';

/**
 * Where the measurement lands: committed, unlike the gitignored `e2e/artifacts/`, because a
 * release gate has to be able to read it. Resolved from the working directory, which is the
 * convention the screenshot paths in this suite already rely on; this package is ESM, so there
 * is no `__dirname` to resolve from instead.
 */
const MEASUREMENT = resolve(process.cwd(), '..', 'measurements', 'scientist_actions.json');

interface ActionRecord {
  ordinal: number;
  action: string;
  at_ms: number;
}

/**
 * Every interaction a researcher performs goes through here, so the count is produced by the
 * walk itself rather than maintained by hand beside it. Nothing reaches for a class or a test
 * id: an action that could not be found by its role and its visible name is not an action a
 * researcher could have taken.
 */
class Researcher {
  readonly actions: ActionRecord[] = [];
  private readonly started = Date.now();

  constructor(private readonly page: Page) {}

  private record(action: string): void {
    this.actions.push({
      ordinal: this.actions.length + 1,
      action,
      at_ms: Date.now() - this.started,
    });
  }

  get count(): number {
    return this.actions.length;
  }

  /** Opening the application is not an action taken inside it. */
  async open(): Promise<void> {
    await this.page.goto('/');
    await expect(this.page.getByRole('navigation', { name: 'Scientific workflow' }))
      .toBeVisible();
  }

  async click(locator: Locator, action: string): Promise<void> {
    await locator.click();
    this.record(action);
  }

  async uncheck(locator: Locator, action: string): Promise<void> {
    await locator.uncheck();
    this.record(action);
  }

  async openComposer(): Promise<void> {
    await this.click(
      this.page.getByRole('button', { name: COMPOSER, exact: true }), `open ${COMPOSER}`);
    // Waiting for the served path is not an action; the researcher does nothing while it loads.
    await expect(this.page.getByRole('heading', { name: COMPOSER, exact: true })).toBeVisible();
    await expect(this.page.getByRole('tablist', { name: 'Experiment composition path' }))
      .toBeVisible();
  }

  async step(name: RegExp, label: string): Promise<void> {
    await this.click(this.page.getByRole('tab', { name }), `go to step ${label}`);
  }

  async button(name: string): Promise<void> {
    await this.click(this.page.getByRole('button', { name }), name);
  }
}

/**
 * The two measurements are written together, so a reader never sees one without the other and
 * cannot mistake the recorded seconds for something that passed.
 */
const measurement: Record<string, unknown> = {
  schema: 'scientist-actions/v1',
  definition: 'visible researcher actions from a clean browser session',
  counted: 'one activation of one visible control, located by role and accessible name',
  not_counted: 'opening the application, waiting for a served response, reading, and assertions',
  claim_boundary:
    'Actions on the declared representative path, which is an upper bound on the shortest route '
    + 'and not a claim about the minimum. Wall-clock durations are recorded as context and are '
    + 'asserted by nothing: they are properties of the machine that ran this, not of the '
    + 'interface. Nothing here is evidence about any scientific result.',
};

test.afterAll(() => {
  mkdirSync(dirname(MEASUREMENT), { recursive: true });
  writeFileSync(MEASUREMENT, `${JSON.stringify(measurement, null, 2)}\n`, 'utf-8');
});

test('a complete plan is declared and run in a counted number of visible actions',
  async ({ page }) => {
    test.slow();
    const researcher = new Researcher(page);
    await researcher.open();

    await researcher.openComposer();

    // The bespoke domain metadata cannot plan for, dropped by the researcher rather than
    // hidden by the interface.
    await researcher.step(/2\. Domains/, '2. Domains');
    await researcher.uncheck(
      page.getByRole('checkbox', { name: 'Include order_book' }), 'exclude order_book');

    await researcher.step(/3\. Observation/, '3. Observation');
    await researcher.button('Apply the week preset');

    await researcher.step(/4\. Preflight/, '4. Preflight');
    await researcher.button('Inspect metadata coverage');
    await expect(page.getByRole('tabpanel')).not.toContainText('order_book');

    await researcher.step(/5\. Analysis/, '5. Analysis');
    await researcher.button('Price the declared family');

    await researcher.step(/6\. Freeze and run/, '6. Freeze and run');
    await researcher.button('Render the preregistration summary');
    await researcher.button('Open or resume the run');
    await expect(page.getByRole('status').filter({ hasText: /Opened run|Resumed run/ }))
      .toBeVisible();
    await researcher.button('Execute the frozen plan');
    await expect(page.getByRole('status').filter({ hasText: 'not evidence' })).toBeVisible();

    await researcher.step(/7\. Interpret/, '7. Interpret');
    await expect(page.getByRole('tabpanel')).toContainText('COMPLETE');

    // Frozen. A path that grows an action fails here rather than growing quietly.
    expect(researcher.count).toBe(14);

    measurement.declared_plan_to_completed_run = {
      actions: researcher.count,
      asserted: true,
      reached: 'a COMPLETE run of the frozen plan, which is an executed plan and not evidence',
      timeline: researcher.actions,
      elapsed_seconds_unasserted: researcher.actions[researcher.actions.length - 1].at_ms / 1000,
    };
  });

test('a refusal names its remediation, and reaching it is a counted number of actions',
  async ({ page }) => {
    test.slow();
    const researcher = new Researcher(page);
    await researcher.open();

    await researcher.openComposer();
    await researcher.step(/4\. Preflight/, '4. Preflight');
    await researcher.button('Inspect metadata coverage');

    // The refusal, before the clock starts: a researcher cannot begin remediating something
    // they have not yet been shown.
    const panel = page.getByRole('tabpanel');
    await expect(panel).toContainText('REFUSED');
    await expect(panel).toContainText('order_book');
    await expect(panel).toContainText(/no public archive|content-addressed local record/);
    const refusalSeen = Date.now();
    const actionsToRefusal = researcher.count;

    // The remediation the refusal names, taken through visible controls only.
    await researcher.step(/2\. Domains/, '2. Domains');
    await researcher.uncheck(
      page.getByRole('checkbox', { name: 'Include order_book' }), 'exclude order_book');
    await researcher.step(/4\. Preflight/, '4. Preflight');
    await researcher.button('Inspect metadata coverage');

    // Remediated: the same control that refused now answers, and the refusal is gone rather
    // than merely quieter.
    await expect(panel).not.toContainText('order_book');
    const remediated = Date.now();

    expect(actionsToRefusal).toBe(3);
    expect(researcher.count - actionsToRefusal).toBe(4);

    measurement.refusal_to_remediation = {
      refusal: 'the flagship preflight refuses order_book, which has no public archive to plan',
      actions_to_reach_the_refusal: actionsToRefusal,
      actions_from_refusal_to_remediation: researcher.count - actionsToRefusal,
      asserted: true,
      // Recorded, and asserted by nothing. See this file's header for why.
      refusal_explanation_time_seconds_unasserted: (remediated - refusalSeen) / 1000,
      remediation: 'exclude the bespoke domain through the domain menu',
      timeline: researcher.actions,
    };
  });
