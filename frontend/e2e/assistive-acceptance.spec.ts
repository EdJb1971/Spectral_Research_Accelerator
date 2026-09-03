import { expect, test, Page } from '@playwright/test';

/**
 * TG18.4 rendered accessibility acceptance.
 *
 * This is an engineering inspection in Chromium, not WCAG certification and not a screen-reader
 * audit. It complements TG11.6's source contract by exercising the rendered keyboard route,
 * focus indicator, semantic journey, reflow, contrast, reduced-motion preference and text cues.
 */

const LAYOUTS = [
  { name: 'desktop', width: 1440, height: 900, compact: false },
  { name: 'laptop', width: 1024, height: 768, compact: false },
  { name: 'narrow', width: 375, height: 667, compact: true },
] as const;

function workflowNav(page: Page) {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function restartKeyboardRoute(page: Page) {
  await expect.poll(() => page.evaluate(() => document.activeElement?.tagName)).toBe('BODY');
  await page.keyboard.press('Tab');
}

for (const layout of LAYOUTS) {
  test.describe(`${layout.name} keyboard route at ${layout.width}x${layout.height}`, () => {
    test.use({ viewport: { width: layout.width, height: layout.height } });

    test('starts with the skip link and exposes a visible focus indicator', async ({ page }) => {
      await page.goto('/');
      await restartKeyboardRoute(page);

      const skip = page.getByRole('link', { name: 'Skip to workspace' });
      await expect(skip).toBeFocused();
      await expect(skip).toBeVisible();
      const focusStyle = await skip.evaluate((node) => {
        const style = getComputedStyle(node);
        return { width: parseFloat(style.outlineWidth), style: style.outlineStyle };
      });
      expect(focusStyle.style).not.toBe('none');
      expect(focusStyle.width).toBeGreaterThanOrEqual(3);

      await page.keyboard.press('Enter');
      await expect(page.locator('#workspace-heading')).toBeFocused();
      await expect(page.locator('#workspace-main')).toBeInViewport();
    });

    test('follows the shell order and returns focus after workspace navigation', async ({ page }) => {
      await page.goto('/');
      await restartKeyboardRoute(page);
      await page.keyboard.press('Tab');

      if (layout.compact) {
        const menu = page.getByRole('button', { name: 'Open workspace menu' });
        await expect(menu).toBeFocused();
        await page.keyboard.press('Enter');
        await expect(workflowNav(page).locator('[aria-current="page"]')).toBeFocused();
        await page.keyboard.press('Escape');
        await expect(menu).toBeFocused();
        await page.keyboard.press('Enter');
      } else {
        await expect(workflowNav(page).getByRole('button').first()).toBeFocused();
      }

      const findings = workflowNav(page).getByRole('button', { name: /^Findings$/ });
      await findings.focus();
      await page.keyboard.press('Enter');
      await expect(page.locator('#workspace-heading')).toHaveText('Findings workspace');
      await expect(page.locator('#workspace-heading')).toBeFocused();
    });
  });
}

test.describe('zoom-equivalent reflow', () => {
  // A 1280 CSS-pixel browser zoomed to 200% exposes a 640 CSS-pixel layout viewport. Device
  // scale factor 2 also makes screenshots retain the corresponding physical-pixel density.
  test.use({ viewport: { width: 640, height: 720 }, deviceScaleFactor: 2 });

  test('keeps the complete journey and document inside the effective viewport at 200%', async ({ page }) => {
    await page.goto('/');
    const journey = page.getByRole('navigation', { name: 'Guided research journey' });
    await expect(journey).toBeVisible();
    await expect(journey.getByRole('button')).toHaveCount(7);
    const metrics = await page.evaluate(() => ({
      client: document.documentElement.clientWidth,
      scroll: document.documentElement.scrollWidth,
      stageColumns: getComputedStyle(document.querySelector('.research-journey__stages')!)
        .gridTemplateColumns.trim().split(/\s+/).length,
    }));
    expect(metrics.client).toBe(640);
    expect(metrics.scroll).toBeLessThanOrEqual(metrics.client + 1);
    expect(metrics.stageColumns).toBe(4);
    await page.screenshot({ path: 'e2e/artifacts/200-percent-zoom-equivalent.png' });
  });
});

test('key shell text meets rendered contrast thresholds', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');

  const results = await page.evaluate(() => {
    type RGB = [number, number, number];
    const parse = (value: string): [number, number, number, number] => {
      const parts = value.match(/[\d.]+/g)?.map(Number) || [];
      return [parts[0] || 0, parts[1] || 0, parts[2] || 0, parts[3] ?? 1];
    };
    const over = (top: [number, number, number, number], bottom: RGB): RGB => [
      top[0] * top[3] + bottom[0] * (1 - top[3]),
      top[1] * top[3] + bottom[1] * (1 - top[3]),
      top[2] * top[3] + bottom[2] * (1 - top[3]),
    ];
    const luminance = (rgb: RGB) => {
      const values = rgb.map(value => {
        const channel = value / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2];
    };
    const contrast = (a: RGB, b: RGB) => {
      const [lighter, darker] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (lighter + 0.05) / (darker + 0.05);
    };
    const background = (element: Element): RGB => {
      const chain: Element[] = [];
      for (let node: Element | null = element; node; node = node.parentElement) chain.unshift(node);
      let colour: RGB = [2, 6, 23];
      for (const node of chain) colour = over(parse(getComputedStyle(node).backgroundColor), colour);
      return colour;
    };
    const selectors = [
      '.instrument-header h1',
      '.research-journey__heading > div > p:first-child',
      '.research-journey__boundary',
      '.research-journey__stage-name',
      '.research-journey__reason',
      '.research-journey__stage button',
    ];
    return selectors.map(selector => {
      const element = document.querySelector(selector)!;
      const style = getComputedStyle(element);
      const foreground = parse(style.color).slice(0, 3) as RGB;
      const ratio = contrast(foreground, background(element));
      const large = parseFloat(style.fontSize) >= 24
        || (parseFloat(style.fontSize) >= 18.66 && Number(style.fontWeight) >= 700);
      return { selector, ratio, required: large ? 3 : 4.5 };
    });
  });

  for (const result of results) {
    expect(result.ratio, `${result.selector} contrast ${result.ratio.toFixed(2)}:1`)
      .toBeGreaterThanOrEqual(result.required);
  }
});

test('reduced-motion preference removes meaningful animation and transition time', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  const timings = await page.evaluate(() => {
    const selectors = ['.connection-status', '.research-journey__stage button', '.skip-link'];
    return selectors.map(selector => {
      const style = getComputedStyle(document.querySelector(selector)!);
      return { selector, animation: style.animationDuration, transition: style.transitionDuration };
    });
  });
  const milliseconds = (duration: string) => duration.endsWith('ms')
    ? Number.parseFloat(duration) : Number.parseFloat(duration) * 1000;
  for (const timing of timings) {
    expect(milliseconds(timing.animation), `${timing.selector} animation`).toBeLessThanOrEqual(0.01);
    expect(milliseconds(timing.transition), `${timing.selector} transition`).toBeLessThanOrEqual(0.01);
  }
});

test('journey location and blockers remain named when colour is removed', async ({ page }) => {
  await page.goto('/');
  await page.addStyleTag({ content: `
    *, *::before, *::after {
      color: CanvasText !important;
      background: Canvas !important;
      border-color: CanvasText !important;
      box-shadow: none !important;
    }
  ` });
  const journey = page.getByRole('navigation', { name: 'Guided research journey' });
  await expect(journey.getByText('Blocked: no record is selected for inspection.')).toBeVisible();
  await expect(journey.getByText('Blocked: no study is selected for evidence admission.')).toBeVisible();
  await expect(journey.getByRole('button', { name: 'Open Acquire' })).toHaveAttribute('aria-current', 'step');
  await expect(journey.getByRole('button', { name: /Next legitimate action: Acquire a record/ }))
    .toBeVisible();
  await expect(journey.getByText('This map does not advance or replace the claim ladder.'))
    .toBeVisible();
});

test('the journey exposes ordered navigation and blocker meaning to the accessibility tree', async ({ page }) => {
  await page.goto('/');
  const journey = page.getByRole('navigation', { name: 'Guided research journey' });
  await expect(journey.getByRole('listitem')).toHaveCount(7);
  await expect(journey.getByRole('button')).toHaveCount(7);
  await expect(journey.getByRole('button', { name: 'Open Acquire' })).toHaveAttribute('aria-current', 'step');
  await expect(journey.getByRole('button', { name: /Next legitimate action/ })).toHaveCount(2);
  const accessibleNames = await journey.getByRole('button').evaluateAll(nodes => nodes.map(node =>
    node.getAttribute('aria-label') || node.textContent?.trim().replace(/\s+/g, ' ')));
  expect(accessibleNames).toEqual([
    'Open Acquire', 'Next legitimate action: Acquire a record', 'Open Design', 'Open Run',
    'Open Compare', 'Next legitimate action: Open Composer and save a study', 'Open Report',
  ]);
});
