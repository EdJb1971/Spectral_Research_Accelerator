import { formatNumber } from './FigureData';

/**
 * Declared validity and stated uncertainty for a figure (TG18.2).
 *
 * ## The gap this closes
 *
 * A power-spectrum figure is drawn across every wavenumber the transform produced. The slope
 * fitted to it is not: the backend fits over `[k_min, k_max]`, using `n_points` of the bins, under
 * a `weighting` scheme, and returns `slope_standard_error`, `r_squared` and an explicit
 * `assumptions` list saying - among other things - that no break point is fitted or tested and
 * that anisotropy has been averaged away. Until this module, the figure showed the whole curve,
 * the exponent appeared in a card some distance below it, and the assumptions were returned by
 * the API, typed in `api.ts`, and rendered nowhere at all. A reader saw a spectrum and a beta
 * beside it, and the only available reading was that the exponent describes the curve on screen.
 *
 * The band a fit was taken over, and the uncertainty attached to what came out of it, belong to
 * the same family as the missingness already in the figure contract: they are things the picture
 * cannot tell you, so the picture must say them. They are not a second panel beside the
 * contract - they extend it.
 *
 * ## What this module may and may not do
 *
 * Every number here is transcribed from what the analysis layer returned. `k_min` and `k_max` are
 * backend values drawn as a band on an axis that already exists; the standard error, the point
 * count, the weighting and the assumption strings are carried through, the strings verbatim
 * (R22, R23), because backend-authored claim language is the claim and paraphrase is authorship.
 *
 * What this module deliberately does **not** do is draw the fitted power law, or an envelope
 * around it at plus or minus one standard error. Both are the obvious next move and both would
 * require the view to evaluate a model at every plotted abscissa. A curve rendered by the browser
 * is indistinguishable on screen from measured data, and an envelope is a confidence statement
 * with a shape - which is the mean-under-the-plot temptation of the previous slice wearing better
 * clothes. The band and the numbers are stated; the model stays where it was computed. That
 * boundary is printed on the page, not merely recorded here.
 *
 * ## The precedent this generalises, and what was deliberately left alone
 *
 * `Heatmap2D` already had one instance of this idea - `validInset`, the dashed box marking the
 * region far enough from the boundary to be free of the padding artefact, with its own prop, its
 * own annotation and its own sentence in the cell inspector. It is the same concept: a declared
 * sub-region within which a stated claim holds, drawn rather than cropped. That one-off is what
 * suggested the general shape here.
 *
 * It has not been retrofitted onto this module, and that is a decision rather than an omission.
 * Its only caller is `DTCWTScientificView`, which draws six panels per level and already states
 * the inset once, at level scope, in a banner above them; routing it through here would repeat
 * that statement six times and trade a clear page for a uniform one. An inset is also the
 * *intersection* of two axis bands, so the per-axis shading below would draw a cross where the
 * heat map correctly draws a box. Generalising a working special case into props no caller wants
 * is how a vocabulary stops earning its keep, so `validInset` stays as it is until a second
 * figure needs a validity region of its own.
 */

/** A declared sub-range of one axis within which a stated claim holds. */
export interface ValidityDomain {
  /** Which axis the band restricts. */
  axis: 'x' | 'y';
  from: number;
  to: number;
  /** Short name for the band, e.g. "Fit domain". View-authored framing, not a claim. */
  label: string;
  /** The producer's own statement of what holds inside. Rendered verbatim. */
  claim: string;
  units?: string | null;
  /** Samples the producer actually used inside the band, when it reported one. */
  pointsUsed?: number | null;
}

/** A value the analysis layer produced, with whatever uncertainty it stood behind. */
export interface StatedUncertainty {
  label: string;
  value: number;
  /** Absent or non-finite means the producer supplied none - which is stated, not hidden. */
  uncertainty?: number | null;
  units?: string | null;
  /** How the producer obtained the number. Rendered verbatim. */
  basis: string;
  /** Further producer-authored qualifiers, e.g. a fit's assumptions. Verbatim, never edited. */
  qualifiers?: string[];
}

export interface DomainDecision {
  domain: ValidityDomain;
  /** Whether the band can honestly be drawn on this figure. */
  marked: boolean;
  /** Marked, but reaching past what the figure draws. */
  clipped: boolean;
  /**
   * The span actually marked, clamped to what the figure draws. Equal to the declared range
   * unless `clipped`. Held here rather than recomputed at draw time so the band on screen is the
   * one the prose describes, and so a limit the axis cannot represent - a zero lower bound on a
   * logarithmic axis, which is where fitted spectra are read - never reaches Plotly.
   */
  markedFrom: number;
  markedTo: number;
  reason: string;
}

export interface UncertaintyDecision {
  statement: StatedUncertainty;
  /** The value itself is a finite number the producer stood behind. */
  stated: boolean;
  /** An uncertainty accompanies it. */
  bounded: boolean;
  text: string;
  warn: boolean;
}

export interface ValidityContract {
  domains: DomainDecision[];
  uncertainties: UncertaintyDecision[];
  /** Whether there is anything at all to state. */
  any: boolean;
}

/** The extent a figure actually draws on one axis, so a band can be checked against it. */
export interface DrawnExtent { x?: [number, number]; y?: [number, number]; }

const axisName = (axis: 'x' | 'y') => (axis === 'x' ? 'horizontal' : 'vertical');

function withUnits(units?: string | null): string {
  return units ? ` ${units}` : '';
}

/**
 * Decide what may be drawn and what must be said instead. Every refusal returns a reason written
 * for the page: a band that cannot be marked is more dangerous silently absent than stated, since
 * its absence is indistinguishable from a fit that spanned the whole figure.
 */
export function buildValidityContract(
  { domains = [], uncertainties = [], drawn = {} }: {
    domains?: ValidityDomain[];
    uncertainties?: StatedUncertainty[];
    drawn?: DrawnExtent;
  },
): ValidityContract {
  const domainDecisions = domains.map<DomainDecision>((domain) => {
    const { from, to, axis, label, units, pointsUsed } = domain;
    const span = `${formatNumber(from)} to ${formatNumber(to)}${withUnits(units)}`;
    const used = pointsUsed != null && Number.isFinite(pointsUsed)
      ? `, over ${pointsUsed} sample${pointsUsed === 1 ? '' : 's'}` : '';

    if (!Number.isFinite(from) || !Number.isFinite(to)) {
      return {
        domain,
        marked: false,
        clipped: false,
        markedFrom: from,
        markedTo: to,
        reason: `${label} is not marked on this figure: its limits are not both finite numbers, `
          + 'so there is no band to draw. Read the figure as carrying no declared domain.',
      };
    }
    if (from >= to) {
      return {
        domain,
        marked: false,
        clipped: false,
        markedFrom: from,
        markedTo: to,
        reason: `${label} is not marked on this figure: the declared range ${span} does not run `
          + 'upwards, so it encloses nothing.',
      };
    }

    const extent = drawn[axis];
    if (extent && Number.isFinite(extent[0]) && Number.isFinite(extent[1])) {
      const [low, high] = extent;
      if (to < low || from > high) {
        return {
          domain,
          marked: false,
          clipped: false,
          markedFrom: from,
          markedTo: to,
          reason: `${label} is not marked on this figure: the declared range ${span} lies entirely `
            + `outside the ${axisName(axis)} extent this figure draws `
            + `(${formatNumber(low)} to ${formatNumber(high)}), so no part of it is on screen. `
            + 'The claim below applies to samples this figure does not show.',
        };
      }
      if (from < low || to > high) {
        return {
          domain,
          marked: true,
          clipped: true,
          markedFrom: Math.max(from, low),
          markedTo: Math.min(to, high),
          // The clipping note is added to the standard sentence rather than replacing it. The
          // part that matters most to a reader - that the curve continues past the band and the
          // claim does not - is true whether or not the band also runs off the figure, and an
          // earlier draft lost it in exactly the case the platform actually produces.
          reason: `${label}: ${span}${used}, marked on the ${axisName(axis)} axis. Samples outside `
            + 'the band are drawn but were not used, so the claim below does not describe them. '
            + 'The declared band also reaches past the extent this figure draws '
            + `(${formatNumber(low)} to ${formatNumber(high)}), so the marking shows only the `
            + 'part that is on screen.',
        };
      }
    }

    return {
      domain,
      marked: true,
      clipped: false,
      markedFrom: from,
      markedTo: to,
      reason: `${label}: ${span}${used}, marked on the ${axisName(axis)} axis. Samples outside the `
        + 'band are drawn but were not used, so the claim below does not describe them.',
    };
  });

  const uncertaintyDecisions = uncertainties.map<UncertaintyDecision>((statement) => {
    const { label, value, uncertainty, units } = statement;
    if (!Number.isFinite(value)) {
      return {
        statement,
        stated: false,
        bounded: false,
        warn: true,
        text: `${label} was not produced for this figure, so there is no value to read against it.`,
      };
    }
    const bounded = uncertainty != null && Number.isFinite(uncertainty);
    if (!bounded) {
      return {
        statement,
        stated: true,
        bounded: false,
        warn: true,
        // The reason api.ts gives for typing the standard error as load-bearing rather than
        // decorative: an exponent without one cannot be tested against a reference exponent.
        text: `${label} = ${formatNumber(value)}${withUnits(units)}, with no uncertainty supplied. `
          + 'A value quoted without one cannot be compared against a reference value, because '
          + 'nothing says whether the difference is larger than the spread.',
      };
    }
    return {
      statement,
      stated: true,
      bounded: true,
      warn: false,
      text: `${label} = ${formatNumber(value)} ± ${formatNumber(uncertainty as number)}`
        + `${withUnits(units)}.`,
    };
  });

  return {
    domains: domainDecisions,
    uncertainties: uncertaintyDecisions,
    any: domainDecisions.length > 0 || uncertaintyDecisions.length > 0,
  };
}

/**
 * Plotly shapes for the bands that may honestly be drawn.
 *
 * On a logarithmic axis Plotly reads shape coordinates as log10 of the data value, not as the
 * value. Passing the raw wavenumber would place the band somewhere else entirely on exactly the
 * figures that need it most, since a power-law fit is read on log-log axes - so the axis type is
 * a required input here rather than an optional refinement.
 */
export function validityShapes(
  contract: ValidityContract,
  { logX = false, logY = false, colour = '#fbbf24' }:
    { logX?: boolean; logY?: boolean; colour?: string } = {},
): any[] {
  return contract.domains.filter((decision) => decision.marked).map((decision) => {
    const { axis } = decision.domain;
    const log = axis === 'x' ? logX : logY;
    const project = (value: number) => (log ? Math.log10(value) : value);
    const from = project(decision.markedFrom);
    const to = project(decision.markedTo);
    const across = axis === 'x'
      ? { x0: from, x1: to, y0: 0, y1: 1, yref: 'paper' as const }
      : { y0: from, y1: to, x0: 0, x1: 1, xref: 'paper' as const };
    return {
      type: 'rect' as const,
      layer: 'below' as const,
      fillcolor: 'rgba(251,191,36,0.07)',
      line: { color: colour, width: 1, dash: 'dash' as const },
      ...across,
    };
  });
}

/**
 * Whether a point lies inside every band that was marked. `null` when no band was marked, which
 * is the difference between "this sample was not used" and "no domain was declared" - two states
 * that must not collapse into one blank column.
 */
export function insideMarkedDomains(
  x: number, y: number, contract: ValidityContract,
): boolean | null {
  const marked = contract.domains.filter((decision) => decision.marked);
  if (marked.length === 0) return null;
  return marked.every((decision) => {
    const value = decision.domain.axis === 'x' ? x : y;
    if (!Number.isFinite(value)) return false;
    return value >= decision.domain.from && value <= decision.domain.to;
  });
}

/**
 * The stated domain and uncertainty, placed with the figure rather than inside its disclosure.
 *
 * A restricted domain changes how the whole picture may be read - the same reason the comparison
 * contract is not hidden either. A reader who opens nothing must still learn that the exponent
 * quoted for this curve was taken from part of it.
 */
export function FigureValidityNotice({ contract, label }: {
  contract: ValidityContract;
  label: string;
}) {
  if (!contract.any) return null;
  const refused = contract.domains.some((decision) => !decision.marked)
    || contract.uncertainties.some((decision) => decision.warn);
  return (
    <section aria-label={`Validity and uncertainty for ${label}`}
      className={`figure-validity rounded-lg border px-3 py-2 text-[11px] space-y-1 ${
        refused
          ? 'border-amber-500/25 bg-amber-500/5 text-amber-200/90'
          : 'border-teal-500/25 bg-teal-500/5 text-teal-200/90'}`}>
      {contract.domains.map((decision) => (
        <div key={decision.domain.label} className="space-y-0.5">
          <p>{decision.reason}</p>
          <p className="text-slate-400">{decision.domain.claim}</p>
        </div>
      ))}
      {contract.uncertainties.map((decision) => (
        <div key={decision.statement.label} className="space-y-0.5">
          <p className={decision.warn ? 'text-amber-300' : undefined}>{decision.text}</p>
          <p className="text-slate-400">{decision.statement.basis}</p>
          {(decision.statement.qualifiers || []).map((qualifier) => (
            <p key={qualifier} className="text-slate-500 pl-3 border-l border-slate-700">
              {qualifier}
            </p>
          ))}
        </div>
      ))}
      <p className="text-slate-500 pt-1 border-t border-slate-800">
        The band and these numbers are transcribed from the analysis layer. The fitted model
        itself is not drawn, and no uncertainty envelope is drawn around it: a curve rendered by
        this view would be indistinguishable on screen from measured data.
      </p>
    </section>
  );
}

/**
 * The translation from one backend `SlopeAnalysis` record into the vocabulary above.
 *
 * It lives here rather than at the call site so the mapping is written once and can be exercised
 * directly, and so no view is tempted to decide for itself which parts of a fit record are worth
 * showing. Everything the analysis layer chose to return about the fit is carried: the band, the
 * bins used, the weighting, the standard error, the goodness of fit, and the assumption strings -
 * the strings unedited, because a paraphrased assumption is a different assumption.
 *
 * `regime_interpretation` is deliberately *not* pulled in. It is the analysis layer's reading of
 * what the exponent means, it is already displayed where the exponent is quoted, and repeating a
 * conclusion beside a figure is how a tentative reading hardens into a caption.
 */
export function slopeValidity(
  analysis: {
    slope_beta: number;
    slope_standard_error?: number;
    r_squared: number;
    n_points?: number;
    k_min?: number;
    k_max?: number;
    weighting?: string;
    assumptions?: string[];
  },
  { seriesLabel, kUnits }: { seriesLabel: string; kUnits?: string | null },
): { domains: ValidityDomain[]; uncertainties: StatedUncertainty[] } {
  const domains: ValidityDomain[] = [];
  if (analysis.k_min != null && analysis.k_max != null) {
    domains.push({
      axis: 'x',
      from: analysis.k_min,
      to: analysis.k_max,
      label: `${seriesLabel} fit domain`,
      units: kUnits ?? null,
      pointsUsed: analysis.n_points ?? null,
      claim: `The exponent quoted for ${seriesLabel} was fitted over this band only`
        + `${analysis.weighting ? `, with ${analysis.weighting} weighting` : ''}. It is not a `
        + 'description of the curve outside it.',
    });
  }
  return {
    domains,
    uncertainties: [
      {
        label: `${seriesLabel} spectral exponent β`,
        value: analysis.slope_beta,
        uncertainty: analysis.slope_standard_error ?? null,
        basis: `Least-squares power-law fit reported by the analysis layer, R² `
          + `${formatNumber(analysis.r_squared)}`
          + `${analysis.n_points != null ? ` over ${analysis.n_points} spectral bins` : ''}.`,
        qualifiers: analysis.assumptions || [],
      },
    ],
  };
}
