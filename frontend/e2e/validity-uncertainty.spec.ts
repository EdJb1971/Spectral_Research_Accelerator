import { expect, test, Page, Locator } from '@playwright/test';

/**
 * TG18.2 acceptance: a figure states the domain a claim was taken over, and the uncertainty
 * attached to it.
 *
 * The power spectral density chart draws every wavenumber bin the transform produced. The
 * exponents quoted for it are fitted over `[k_min, k_max]` with `n_points` of those bins, and the
 * backend also returns a standard error, an R-squared, a weighting scheme and an explicit
 * `assumptions` list - including that no break point is fitted or tested, and that anisotropy has
 * been averaged away over annuli. Before this slice the figure showed the whole curve, the
 * exponent appeared in a card below it, and the assumptions were returned by the API, typed in
 * `api.ts`, and rendered nowhere at all.
 *
 * As in `comparison-contract.spec.ts`, the suite has two halves. The decision function is
 * exercised directly because its refusals - a degenerate band, a band off the drawn extent, a
 * value with no uncertainty supplied - are not reachable through the current UI, where the
 * backend's own fit always lands on screen. A refusal first exercised by the call site that
 * needed it is a refusal nobody has checked. The rendered half then confirms the decision reaches
 * the figure, including the part that must survive a logarithmic axis.
 */

const MODULE = '/src/components/FigureValidity.tsx';

type Contract = {
  domains: { marked: boolean; clipped: boolean; markedFrom: number; markedTo: number; reason: string }[];
  uncertainties: { stated: boolean; bounded: boolean; warn: boolean; text: string }[];
  any: boolean;
};

async function decide(page: Page, input: unknown): Promise<Contract> {
  return page.evaluate(async ({ modulePath, args }) => {
    const module = await import(/* @vite-ignore */ modulePath);
    return module.buildValidityContract(args);
  }, { modulePath: MODULE, args: input });
}

async function shapesFor(page: Page, input: unknown, axes: unknown) {
  return page.evaluate(async ({ modulePath, args, axisOptions }) => {
    const module = await import(/* @vite-ignore */ modulePath);
    return module.validityShapes(module.buildValidityContract(args), axisOptions);
  }, { modulePath: MODULE, args: input, axisOptions: axes });
}

const BAND = {
  axis: 'x', from: 2, to: 8, label: 'Fit domain', units: 'rad/m', pointsUsed: 12,
  claim: 'Fitted over this band only.',
};

test.describe('the validity decision, and the reasons it gives', () => {
  test.beforeEach(async ({ page }) => { await page.goto('/'); });

  test('a band inside the drawn extent is marked, and says what lies outside it', async ({ page }) => {
    const contract = await decide(page, { domains: [BAND], drawn: { x: [1, 20] } });
    expect(contract.domains[0].marked).toBe(true);
    expect(contract.domains[0].clipped).toBe(false);
    expect(contract.domains[0].reason).toContain('2 to 8 rad/m');
    expect(contract.domains[0].reason).toContain('12 samples');
    // The load-bearing sentence: without it, a curve drawn past the band reads as described by it.
    expect(contract.domains[0].reason).toContain('drawn but were not used');
  });

  test('a band reaching past the figure is clipped to what is drawn, and says so', async ({ page }) => {
    const contract = await decide(page, {
      domains: [{ ...BAND, from: 0, to: 8 }], drawn: { x: [1, 20] },
    });
    expect(contract.domains[0].marked).toBe(true);
    expect(contract.domains[0].clipped).toBe(true);
    // Clamped rather than merely described: the band on screen must be the band in the prose,
    // and a zero lower bound has no place on a logarithmic axis at all.
    expect(contract.domains[0].markedFrom).toBe(1);
    expect(contract.domains[0].markedTo).toBe(8);
    expect(contract.domains[0].reason).toContain('reaches past');
    // The sentence that does the real work must survive clipping - this is the branch the
    // platform's own spectra actually take, since the fit runs to Nyquist and the plotted bins
    // stop short of it.
    expect(contract.domains[0].reason).toContain('drawn but were not used');
  });

  test('a band entirely off the figure refuses to be marked', async ({ page }) => {
    const contract = await decide(page, {
      domains: [{ ...BAND, from: 40, to: 80 }], drawn: { x: [1, 20] },
    });
    expect(contract.domains[0].marked).toBe(false);
    expect(contract.domains[0].reason).toContain('lies entirely');
    // Silence here would be indistinguishable from a fit that spanned the whole figure.
    expect(contract.domains[0].reason).toContain('applies to samples this figure does not show');
  });

  test('a degenerate band refuses rather than drawing a zero-width mark', async ({ page }) => {
    const contract = await decide(page, { domains: [{ ...BAND, from: 8, to: 8 }] });
    expect(contract.domains[0].marked).toBe(false);
    expect(contract.domains[0].reason).toContain('encloses nothing');
  });

  test('a band with a non-finite limit refuses', async ({ page }) => {
    const contract = await decide(page, { domains: [{ ...BAND, to: null }] });
    expect(contract.domains[0].marked).toBe(false);
    expect(contract.domains[0].reason).toContain('not both finite');
  });

  test('a value with an uncertainty is stated as a bound', async ({ page }) => {
    const contract = await decide(page, {
      uncertainties: [{ label: 'β', value: -1.6667, uncertainty: 0.04, basis: 'least squares' }],
    });
    expect(contract.uncertainties[0].bounded).toBe(true);
    expect(contract.uncertainties[0].warn).toBe(false);
    expect(contract.uncertainties[0].text).toContain('±');
  });

  test('a value with no uncertainty supplied warns and says why that matters', async ({ page }) => {
    const contract = await decide(page, {
      uncertainties: [{ label: 'β', value: -1.6667, basis: 'least squares' }],
    });
    expect(contract.uncertainties[0].stated).toBe(true);
    expect(contract.uncertainties[0].bounded).toBe(false);
    expect(contract.uncertainties[0].warn).toBe(true);
    // The reason api.ts gives for typing the standard error as load-bearing rather than optional.
    expect(contract.uncertainties[0].text).toContain('cannot be compared against a reference');
  });

  test('a value the producer never produced is not printed as a number', async ({ page }) => {
    const contract = await decide(page, {
      uncertainties: [{ label: 'β', value: null, basis: 'fit did not converge' }],
    });
    expect(contract.uncertainties[0].stated).toBe(false);
    expect(contract.uncertainties[0].text).toContain('was not produced');
    expect(contract.uncertainties[0].text).not.toContain('NaN');
  });

  test('a band on a logarithmic axis is projected into log coordinates', async ({ page }) => {
    const linear = await shapesFor(page, { domains: [BAND], drawn: { x: [1, 20] } }, {});
    const logged = await shapesFor(page, { domains: [BAND], drawn: { x: [1, 20] } }, { logX: true });
    expect(linear[0].x0).toBe(2);
    // Plotly reads shape coordinates on a log axis as log10 of the value. Passing the raw
    // wavenumber would put the band somewhere else entirely on exactly the figures - log-log
    // spectra - where a fit domain matters most.
    expect(logged[0].x0).toBeCloseTo(Math.log10(2), 10);
    expect(logged[0].x1).toBeCloseTo(Math.log10(8), 10);
  });

  test('an unmarked band produces no shape at all', async ({ page }) => {
    const shapes = await shapesFor(page, {
      domains: [{ ...BAND, from: 40, to: 80 }], drawn: { x: [1, 20] },
    }, {});
    expect(shapes).toHaveLength(0);
  });

  test('membership distinguishes "outside the band" from "no band declared"', async ({ page }) => {
    const inside = await page.evaluate(async ({ modulePath, band }) => {
      const module = await import(/* @vite-ignore */ modulePath);
      const withBand = module.buildValidityContract({ domains: [band], drawn: { x: [1, 20] } });
      const without = module.buildValidityContract({});
      return {
        within: module.insideMarkedDomains(4, 0, withBand),
        beyond: module.insideMarkedDomains(15, 0, withBand),
        undeclared: module.insideMarkedDomains(4, 0, without),
      };
    }, { modulePath: MODULE, band: BAND });
    expect(inside.within).toBe(true);
    expect(inside.beyond).toBe(false);
    // Not `false`: a sample outside a declared band and a sample on a figure with no band are
    // different states, and collapsing them would put a bare "no" against every point.
    expect(inside.undeclared).toBeNull();
  });
});

test.describe('the slope record translated into the vocabulary', () => {
  test.beforeEach(async ({ page }) => { await page.goto('/'); });

  test('every part of the fit the backend returned is carried, verbatim where it is a claim', async ({ page }) => {
    const translated = await page.evaluate(async ({ modulePath }) => {
      const module = await import(/* @vite-ignore */ modulePath);
      return module.slopeValidity({
        slope_beta: -1.6667,
        slope_standard_error: 0.042,
        r_squared: 0.987,
        n_points: 24,
        k_min: 2,
        k_max: 30,
        weighting: 'counts',
        assumptions: ['isotropy: power is averaged over annuli', 'a single power law holds'],
        regime_interpretation: 'Consistent with a Kolmogorov inertial range.',
      }, { seriesLabel: 'Forecast PSD', kUnits: 'rad/m' });
    }, { modulePath: MODULE });

    expect(translated.domains[0].from).toBe(2);
    expect(translated.domains[0].to).toBe(30);
    expect(translated.domains[0].pointsUsed).toBe(24);
    expect(translated.domains[0].claim).toContain('counts weighting');
    expect(translated.uncertainties[0].uncertainty).toBe(0.042);
    expect(translated.uncertainties[0].basis).toContain('0.987');
    // Verbatim: a paraphrased assumption is a different assumption (R22, R23).
    expect(translated.uncertainties[0].qualifiers).toEqual([
      'isotropy: power is averaged over annuli', 'a single power law holds']);
    // The interpretation is deliberately not carried into the figure. It is already displayed
    // where the exponent is quoted, and a conclusion repeated beside a picture hardens into a
    // caption.
    expect(JSON.stringify(translated)).not.toContain('Kolmogorov');
  });

  test('a fit that did not run is carried as a refusal, not as NaN', async ({ page }) => {
    const contract = await page.evaluate(async ({ modulePath }) => {
      const module = await import(/* @vite-ignore */ modulePath);
      // What the backend returns when fewer than four usable bins fall in the band.
      const translated = module.slopeValidity({
        slope_beta: NaN, slope_standard_error: NaN, r_squared: NaN,
        n_points: 2, k_min: 2, k_max: 30, weighting: 'counts', assumptions: [],
      }, { seriesLabel: 'Forecast PSD', kUnits: 'rad/m' });
      return module.buildValidityContract({ ...translated, drawn: { x: [1, 40] } });
    }, { modulePath: MODULE });
    expect(contract.uncertainties[0].stated).toBe(false);
    expect(contract.uncertainties[0].text).toContain('was not produced');
    // The band is still marked: which bins the backend looked at is true regardless of whether
    // a usable exponent came out of them.
    expect(contract.domains[0].marked).toBe(true);
  });
});

function workflowNav(page: Page): Locator {
  return page.getByRole('navigation', { name: 'Scientific workflow' });
}

async function runDiagnostics(page: Page) {
  await page.goto('/');
  await workflowNav(page).getByRole('button', { name: /^Synthetic generator/ }).click();
  await page.getByRole('button', { name: 'Generate Analytical Field' }).click();
  await expect(page.locator('#fig-clean-field.js-plotly-plot')).toBeVisible({ timeout: 30_000 });
  await workflowNav(page).getByRole('button', { name: /^Diagnostics/ }).click();
  await page.getByRole('button', { name: 'Run Benchmarking diagnostics' }).click();
  await expect(page.locator('#fig-psd.js-plotly-plot')).toBeVisible({ timeout: 60_000 });
}

test.describe('the decision reaching the spectral figure', () => {
  test('the fit domain is stated with the figure, without opening anything', async ({ page }) => {
    await runDiagnostics(page);
    const notice = page.locator('section.figure-validity').first();
    // Never inside a disclosure: it governs how the curve above may be read.
    await expect(notice).toBeVisible();
    await expect(notice).toContainText('fit domain');
    await expect(notice).toContainText('drawn but were not used');
  });

  test('the exponent appears with its standard error, next to the curve it came from', async ({ page }) => {
    await runDiagnostics(page);
    const notice = page.locator('section.figure-validity').first();
    await expect(notice).toContainText('spectral exponent');
    await expect(notice).toContainText('±');
    await expect(notice).toContainText('R²');
  });

  test('the assumptions the backend returned are shown verbatim', async ({ page }) => {
    await runDiagnostics(page);
    const notice = page.locator('section.figure-validity').first();
    // These strings were typed in api.ts and rendered nowhere before this slice. They are the
    // difference between an exponent and an exponent one can argue with.
    await expect(notice).toContainText('isotropy');
    await expect(notice).toContainText('no break point is fitted or tested');
  });

  test('the page states that the fitted model itself is not drawn', async ({ page }) => {
    await runDiagnostics(page);
    await expect(page.locator('section.figure-validity').first())
      .toContainText('fitted model itself is not drawn');
  });

  test('the band reaches the figure as a shape', async ({ page }) => {
    await runDiagnostics(page);
    const shapes = await page.evaluate(() => {
      const div = document.getElementById('fig-psd') as any;
      return (div?.layout?.shapes || []).map((shape: any) => ({
        type: shape.type, x0: shape.x0, x1: shape.x1, yref: shape.yref,
      }));
    });
    expect(shapes.length).toBeGreaterThan(0);
    for (const shape of shapes) {
      expect(shape.type).toBe('rect');
      // Full height of the plotting area, so the band reads as a restriction on the abscissa
      // rather than as a box around some region of the curve.
      expect(shape.yref).toBe('paper');
      expect(Number.isFinite(shape.x0)).toBe(true);
      expect(shape.x1).toBeGreaterThan(shape.x0);
    }
  });

  test('the figure data table marks which points fall in the declared domain', async ({ page }) => {
    await runDiagnostics(page);
    const panel = page.locator('#fig-psd').locator('xpath=ancestor::figure')
      .locator('details.figure-data');
    await panel.locator('summary').click();
    // The text equivalent of the shading: a reader who cannot see the band still needs to know
    // which points the exponent was taken over.
    await expect(panel.getByRole('columnheader', { name: 'in declared domain' })).toBeVisible();
    await expect(panel.locator('tbody tr').first()).toBeVisible();
  });
});
