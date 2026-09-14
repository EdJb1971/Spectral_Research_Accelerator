import { expect, test, Page, Locator } from '@playwright/test';

/**
 * TG18.1 acceptance: the narrow-width rendered inspection, in a real browser.
 *
 * `test_frontend_contract.py` reads the source and `npm run build` type-checks it. Neither can
 * see a rendered pixel, so neither can answer the only question this task asks: does the shell
 * actually hold together at 320 px, or does a heading, a status pill and a menu button quietly
 * push the document sideways? Every assertion below is a measurement taken from the laid-out
 * document, not a restatement of a CSS rule.
 *
 * What this file deliberately does NOT claim:
 *
 *   - It is not an accessibility conformance audit. TG18.4 owns rendered assistive-technology
 *     acceptance and TG11.6 owns the source contract; measuring a font size is not certifying
 *     a contrast ratio with a screen reader in the loop.
 *   - It does not cover the sticky research-context strip. That section renders only when a
 *     record or study is selected, and no workspace reached here selects one without acquiring
 *     or committing something. Its pinning is covered at source level only, and is recorded as
 *     such rather than counted as rendered evidence.
 *   - It does not cover Plotly label floors. Those axes exist only once a transform has run
 *     against real data, which this inspection does not do.
 */

const VIEWPORTS = [
  { name: '320-floor', width: 320, height: 640 },
  { name: '375-phone', width: 375, height: 667 },
  { name: '414-phone-large', width: 414, height: 896 },
  { name: '768-tablet', width: 768, height: 1024 },
] as const;

/** One workspace per product mode identified by the TG18.0 baseline, plus the acquisition entry. */
const WORKSPACES = [
  'Import & acquire data',
  'Spectral transforms',
  'New experiment',
  'Findings',
  'System health & validation',
] as const;

/** The compact-width breakpoint below which the workflow rail is a drawer rather than a column. */
const DRAWER_MAX_WIDTH = 1023;

function menuTrigger(page: Page): Locator {
  return page.getByRole('button', { name: /^(Open|Close) workspace menu$/ }).first();
}

function workflowNav(page: Page): Locator {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

/**
 * Rail entries in the gridded line append a context label, so the accessible name of the
 * spectral workspace is `Spectral transformsGridded field line`. Matching on a prefix keeps the
 * test reading like the researcher's intent while an ambiguity check keeps the prefix honest.
 */
function workspaceButton(page: Page, name: string): Locator {
  return workflowNav(page).getByRole('button', {
    name: new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`),
  });
}

async function openWorkspace(page: Page, name: string) {
  await menuTrigger(page).click();
  await expect(workspaceButton(page, name)).toHaveCount(1);
  await workspaceButton(page, name).click();
  await expect(page.locator('#workspace-heading')).toHaveText(`${name} workspace`);
  // The drawer closes itself on selection; waiting for that keeps a measurement from being
  // taken while a fixed overlay still covers the workspace.
  await expect(workflowNav(page)).toBeHidden();
}

/**
 * Elements wider than the viewport are a defect only when nothing is scrolling them. A table or
 * a code block inside its own `overflow-x: auto` container is the intended treatment, so an
 * ancestor that scrolls or clips absolves the child.
 */
async function unscrolledOverflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const limit = window.innerWidth + 1;
    const offenders: string[] = [];
    const main = document.querySelector('#workspace-main');
    if (!main) return ['#workspace-main is not in the document'];
    for (const element of Array.from(main.querySelectorAll<HTMLElement>('*'))) {
      const rect = element.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;
      if (rect.right <= limit) continue;
      let contained = false;
      for (let a = element.parentElement; a && a !== document.body; a = a.parentElement) {
        const overflowX = getComputedStyle(a).overflowX;
        if (overflowX === 'auto' || overflowX === 'scroll' || overflowX === 'hidden') {
          contained = true;
          break;
        }
      }
      if (contained) continue;
      offenders.push(
        `${element.tagName.toLowerCase()}.${element.className.toString().slice(0, 60)} ` +
        `right=${Math.round(rect.right)} > ${limit}`);
      if (offenders.length >= 8) break;
    }
    return offenders;
  });
}

/** Visible elements that own text, reported when their rendered size falls below the floor. */
async function textBelowFloor(page: Page, floorPx: number): Promise<string[]> {
  return page.evaluate((floor) => {
    const offenders: string[] = [];
    const main = document.querySelector('#workspace-main');
    if (!main) return ['#workspace-main is not in the document'];
    for (const element of Array.from(main.querySelectorAll<HTMLElement>('*'))) {
      const ownText = Array.from(element.childNodes)
        .filter(node => node.nodeType === Node.TEXT_NODE)
        .map(node => node.textContent || '')
        .join('')
        .trim();
      if (!ownText) continue;
      const rect = element.getBoundingClientRect();
      // Screen-reader-only text is clipped to a 1px box on purpose and has no rendered size to
      // defend; it is excluded rather than silently passed.
      if (rect.width <= 1 || rect.height <= 1) continue;
      const size = parseFloat(getComputedStyle(element).fontSize);
      if (size >= floor - 0.01) continue;
      offenders.push(`${element.tagName.toLowerCase()} "${ownText.slice(0, 40)}" = ${size}px`);
      if (offenders.length >= 8) break;
    }
    return offenders;
  }, floorPx);
}

/** Text-entry controls below the declared touch target height. */
async function shortTextControls(page: Page, floorPx: number): Promise<string[]> {
  return page.evaluate((floor) => {
    const offenders: string[] = [];
    const selector =
      '#workspace-main input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="hidden"]), ' +
      '#workspace-main select, #workspace-main textarea';
    for (const control of Array.from(document.querySelectorAll<HTMLElement>(selector))) {
      const rect = control.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;
      if (rect.height >= floor - 0.5) continue;
      offenders.push(`${control.tagName.toLowerCase()}[${(control as HTMLInputElement).type || 'n/a'}] = ${rect.height}px`);
      if (offenders.length >= 8) break;
    }
    return offenders;
  }, floorPx);
}

for (const viewport of VIEWPORTS) {
  test.describe(`at ${viewport.width}x${viewport.height}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('every product mode lays out without pushing the document sideways', async ({ page }) => {
      await page.goto('/');
      for (const workspace of WORKSPACES) {
        await openWorkspace(page, workspace);

        const documentScrolls = await page.evaluate(() => ({
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
        }));
        expect(documentScrolls.scrollWidth,
          `${workspace} scrolls horizontally at ${viewport.width}px`)
          .toBeLessThanOrEqual(documentScrolls.clientWidth + 1);

        expect(await unscrolledOverflow(page),
          `${workspace} has content past the right edge that nothing scrolls`).toEqual([]);

        await page.screenshot({
          path: `e2e/artifacts/${viewport.name}-${workspace.replace(/\W+/g, '-').toLowerCase()}.png`,
          fullPage: false,
        });
      }
    });

    test('rendered metadata and text-entry controls hold their declared floors', async ({ page }) => {
      await page.goto('/');
      for (const workspace of WORKSPACES) {
        await openWorkspace(page, workspace);
        expect(await textBelowFloor(page, 11),
          `${workspace} renders workspace text below the 11px floor`).toEqual([]);
        expect(await shortTextControls(page, 40),
          `${workspace} renders a text-entry control below the 40px minimum`).toEqual([]);
      }
    });

    test('the header keeps brand, connection state and menu inside the viewport', async ({ page }) => {
      await page.goto('/');
      const header = page.locator('header.instrument-header');
      await expect(header).toBeVisible();

      await expect(page.getByRole('heading', { level: 1, name: 'SpectralEarth' })).toBeVisible();
      await expect(menuTrigger(page)).toBeVisible();

      // Below 400px the connection control is an icon. Losing its label is acceptable; losing
      // its accessible name would make the only indicator of backend state unnameable.
      const connection = page.locator('.connection-status');
      await expect(connection).toBeVisible();
      const connectionName = await connection.evaluate(
        node => node.getAttribute('aria-label') || node.textContent || '');
      expect(connectionName.trim().length,
        'the connection control has no accessible name').toBeGreaterThan(0);
      if (viewport.width < 400) {
        await expect(page.locator('.connection-status .connection-label')).toBeHidden();
      }

      const overflowing = await header.evaluate((node, limit) => {
        const offenders: string[] = [];
        for (const element of Array.from(node.querySelectorAll<HTMLElement>('*'))) {
          const rect = element.getBoundingClientRect();
          if (rect.width === 0 && rect.height === 0) continue;
          if (rect.right > limit + 1) offenders.push(`${element.tagName.toLowerCase()} right=${Math.round(rect.right)}`);
        }
        return offenders;
      }, viewport.width);
      expect(overflowing, `header content overflows ${viewport.width}px`).toEqual([]);
    });
  });
}

test.describe('the compact workflow drawer', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('opening it scrims the workspace, locks scroll and moves focus into the rail', async ({ page }) => {
    await page.goto('/');
    await expect(workflowNav(page)).toBeHidden();
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');

    await menuTrigger(page).click();
    await expect(workflowNav(page)).toBeVisible();
    await expect(page.locator('.workflow-nav-scrim')).toBeVisible();
    expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden');

    // The drawer is a fixed overlay, not a block pushed into the page flow.
    expect(await workflowNav(page).evaluate(node => getComputedStyle(node).position)).toBe('fixed');
    // Focus lands on the workspace the researcher is already in, not on the first item.
    await expect(workflowNav(page).locator('[aria-current="page"]')).toBeFocused();

    await page.screenshot({ path: 'e2e/artifacts/375-phone-drawer-open.png' });
  });

  test('Escape closes it and returns focus to the trigger', async ({ page }) => {
    await page.goto('/');
    await menuTrigger(page).click();
    await expect(workflowNav(page)).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(workflowNav(page)).toBeHidden();
    await expect(menuTrigger(page)).toBeFocused();
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');
  });

  test('the scrim closes it and returns focus to the trigger', async ({ page }) => {
    await page.goto('/');
    await menuTrigger(page).click();
    await expect(workflowNav(page)).toBeVisible();

    // The drawer is `min(22rem, 100vw)` wide, so at 375px it covers all but a narrow strip of
    // the scrim. Clicking the left of the scrim would be intercepted by the drawer itself;
    // aiming at the exposed strip is what a researcher can actually hit. Below 352px no strip
    // is exposed at all and outside-click dismissal is unavailable by construction - Escape,
    // the trigger and selection remain, which is why none of them is optional.
    const railRight = await workflowNav(page).evaluate(node => node.getBoundingClientRect().right);
    const width = page.viewportSize()!.width;
    expect(width - railRight, 'no scrim strip is exposed beside the drawer').toBeGreaterThan(8);
    await page.mouse.click(Math.round((railRight + width) / 2), 200);

    await expect(workflowNav(page)).toBeHidden();
    await expect(menuTrigger(page)).toBeFocused();
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');
  });

  test('Tab stays inside the drawer while it is open', async ({ page }) => {
    await page.goto('/');
    await menuTrigger(page).click();
    await expect(workflowNav(page)).toBeVisible();

    const insideNav = () => page.evaluate(() =>
      !!document.activeElement?.closest('#scientific-workflow-nav'));

    // Enough presses to walk past the last control several times over; a leak shows up the
    // moment focus lands on the header or the workspace behind the scrim.
    for (let press = 0; press < 40; press += 1) {
      await page.keyboard.press('Tab');
      expect(await insideNav(), `focus left the drawer after ${press + 1} Tab presses`).toBe(true);
    }
    await page.keyboard.down('Shift');
    for (let press = 0; press < 10; press += 1) {
      await page.keyboard.press('Tab');
      expect(await insideNav(), `Shift+Tab left the drawer after ${press + 1} presses`).toBe(true);
    }
    await page.keyboard.up('Shift');
  });

  test('choosing a workspace closes it and unlocks the page', async ({ page }) => {
    await page.goto('/');
    await openWorkspace(page, 'Findings');
    await expect(page.locator('.workflow-nav-scrim')).toHaveCount(0);
    expect(await page.evaluate(() => document.body.style.overflow)).not.toBe('hidden');
  });

  test('the rail is a column again above the drawer breakpoint', async ({ page }) => {
    await page.setViewportSize({ width: DRAWER_MAX_WIDTH + 1, height: 900 });
    await page.goto('/');
    await expect(workflowNav(page)).toBeVisible();
    expect(await workflowNav(page).evaluate(node => getComputedStyle(node).position)).toBe('sticky');
    await expect(menuTrigger(page)).toBeHidden();
  });
});

test.describe('one-column reflow below 480px', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('inherited multi-column workspace grids collapse to a single track', async ({ page }) => {
    await page.goto('/');
    let inspected = 0;
    for (const workspace of WORKSPACES) {
      await openWorkspace(page, workspace);
      const multiTrack = await page.evaluate(() => {
        const offenders: string[] = [];
        let seen = 0;
        const selector = '#workspace-main .grid.grid-cols-2, #workspace-main .grid.grid-cols-3, ' +
          '#workspace-main .grid.grid-cols-4, #workspace-main .grid.grid-cols-5';
        for (const grid of Array.from(document.querySelectorAll<HTMLElement>(selector))) {
          const rect = grid.getBoundingClientRect();
          if (rect.width === 0 && rect.height === 0) continue;
          seen += 1;
          // The used value, not the declared one: an implicit track created by a `col-span`
          // child counts here exactly as a declared column would, which is the point.
          const tracks = getComputedStyle(grid).gridTemplateColumns.trim().split(/\s+/).length;
          if (tracks > 1) offenders.push(`${grid.className.toString().slice(0, 60)} -> ${tracks} tracks`);

          // And the rendered result, because a single track is only worth asserting if it
          // actually puts every child on its own row.
          const rows = new Map<number, number>();
          for (const child of Array.from(grid.children)) {
            const box = (child as HTMLElement).getBoundingClientRect();
            if (box.width === 0 && box.height === 0) continue;
            if (getComputedStyle(child as HTMLElement).display === 'contents') continue;
            const top = Math.round(box.top);
            rows.set(top, (rows.get(top) || 0) + 1);
          }
          for (const [top, count] of rows) {
            if (count > 1) {
              offenders.push(`${grid.className.toString().slice(0, 40)} -> ${count} children share row y=${top}`);
              break;
            }
          }
        }
        return { offenders, seen };
      });
      inspected += multiTrack.seen;
      expect(multiTrack.offenders,
        `${workspace} keeps a multi-column grid below 480px`).toEqual([]);
    }
    // A reflow rule that inspected nothing has demonstrated nothing.
    expect(inspected, 'no fixed-column workspace grid was rendered at all').toBeGreaterThan(0);
  });
});
