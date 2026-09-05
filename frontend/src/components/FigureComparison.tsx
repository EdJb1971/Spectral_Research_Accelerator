import { useMemo, useState } from 'react';
import { formatNumber } from './FigureData';

/**
 * The comparison contract for a group of gridded panels (TG18.2).
 *
 * ## Why this is a contract rather than a colour-scale option
 *
 * Two heat maps side by side are an invitation to compare them, and Plotly gives each panel its
 * own colour range derived from its own extremes unless told otherwise. A reconstruction that
 * lost most of its amplitude therefore renders as a near-identical picture beside its original:
 * the difference survives only in two small colour-bar tick ranges, which is exactly where a
 * reader is not looking. Before this module, `zRange` was supplied at exactly one of the
 * platform's call sites, so every other pair - including original against inverse reconstruction,
 * which *is* an error judgement - was drawn on independent scales.
 *
 * Sharing a scale is not always the fix, though, and a control that simply forced one would trade
 * a silent visual error for a louder one. Two fields in different units must never share a raw
 * magnitude axis (R19-R21), and two fields that are not the same measured quantity have no shared
 * meaning to normalise. So the group does not "apply a shared scale"; it *decides whether one is
 * admissible*, states the decision and its reason on the page, and lets the panels follow.
 *
 * ## What the call site must declare, and why it cannot be inferred
 *
 * Units alone cannot establish that two panels measure the same thing - two unitless synthetic
 * fields tell us nothing by being equally unitless. The relationship is knowledge the call site
 * has and a generic component does not: F and F-hat are the same quantity because one is the
 * reconstruction of the other. A caller therefore declares a `quantity` key, and panels share a
 * scale only when that key and their units both agree. The default - no declaration - refuses,
 * so a pair becomes comparable only by an explicit, reviewable claim in the source.
 *
 * Cell correspondence is a second, stricter question: addressing row 7 column 11 in both panels
 * means the same sample only if the grids have the same shape. Scale comparability and cell
 * correspondence are decided and reported separately, because a pair can honestly have one
 * without the other.
 */

export interface ComparisonPanel {
  /** Stable key, used to address the panel's decision. */
  key: string;
  title: string;
  data: number[][];
  units?: string | null;
  /**
   * The measured quantity this panel holds. Two panels are candidates for a shared scale only
   * when they declare the same key. Omitting it refuses comparison, which is the safe default:
   * an undeclared relationship is not evidence of one.
   */
  quantity?: string;
}

export interface ScaleDecision {
  shared: boolean;
  range?: [number, number];
  reason: string;
}

export interface CellDecision {
  linked: boolean;
  rows?: number;
  columns?: number;
  reason: string;
}

export interface ComparisonContract {
  scale: ScaleDecision;
  cells: CellDecision;
}

function describeUnits(units?: string | null): string {
  return units ? units : 'no declared units';
}

function finiteExtent(data: number[][]): { min: number; max: number; any: boolean } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let any = false;
  for (const row of data) {
    for (const value of row) {
      if (!Number.isFinite(value)) continue;
      any = true;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return { min, max, any };
}

/**
 * Decide, and say why. Every branch returns a reason written for the page rather than for a log,
 * because the reason is the part a researcher needs in order to know how to read the pictures.
 */
export function buildComparisonContract(panels: ComparisonPanel[]): ComparisonContract {
  const shapes = panels.map((panel) => ({
    key: panel.key,
    rows: panel.data?.length ?? 0,
    columns: panel.data?.[0]?.length ?? 0,
  }));
  const sameShape = shapes.length > 1
    && shapes.every((shape) => shape.rows === shapes[0].rows && shape.columns === shapes[0].columns)
    && shapes[0].rows > 0 && shapes[0].columns > 0;

  const cells: CellDecision = sameShape
    ? {
      linked: true,
      rows: shapes[0].rows,
      columns: shapes[0].columns,
      reason: `Cell addressing is linked: every panel is ${shapes[0].rows} by ${shapes[0].columns}, `
        + 'so one row and column index names the same sample in each.',
    }
    : {
      linked: false,
      reason: shapes.length > 1
        ? 'Cell addressing is not linked: the panels differ in shape ('
          + shapes.map((shape) => `${shape.rows} by ${shape.columns}`).join(', ')
          + '), so one index does not name the same sample in each.'
        : 'Cell addressing is not linked: there is only one panel in this group.',
    };

  if (panels.length < 2) {
    return { scale: { shared: false, reason: 'A single panel has nothing to be comparable with.' }, cells };
  }

  const quantities = panels.map((panel) => panel.quantity);
  if (quantities.some((quantity) => !quantity)) {
    return {
      scale: {
        shared: false,
        reason: 'Not comparable: at least one panel does not declare which quantity it holds, and '
          + 'an undeclared relationship is not evidence of one. Each panel keeps its own colour '
          + 'range, so the pictures must not be read against each other.',
      },
      cells,
    };
  }
  if (new Set(quantities).size > 1) {
    return {
      scale: {
        shared: false,
        reason: `Not comparable: the panels hold different quantities (${
          Array.from(new Set(quantities)).join(', ')}). There is no shared meaning to normalise, `
          + 'so each panel keeps its own colour range.',
      },
      cells,
    };
  }

  const units = panels.map((panel) => panel.units ?? null);
  if (new Set(units.map((unit) => unit ?? '')).size > 1) {
    return {
      scale: {
        shared: false,
        reason: `Not comparable: the panels declare different units (${
          units.map(describeUnits).join(', ')}). Raw magnitudes in different units must never `
          + 'share one axis, so each panel keeps its own colour range.',
      },
      cells,
    };
  }

  const extents = panels.map((panel) => finiteExtent(panel.data));
  if (extents.some((extent) => !extent.any)) {
    return {
      scale: {
        shared: false,
        reason: 'Not comparable: at least one panel holds no finite sample, so there are no '
          + 'limits to share.',
      },
      cells,
    };
  }

  const min = Math.min(...extents.map((extent) => extent.min));
  const max = Math.max(...extents.map((extent) => extent.max));
  return {
    scale: {
      shared: true,
      range: [min, max],
      reason: `Shared colour range ${formatNumber(min)} to ${formatNumber(max)}, applied to every `
        + `panel. All panels declare the same quantity (${quantities[0]}) and their units agree `
        + `(${units[0] ? units[0] : 'none declared'}), and the range spans every panel's finite `
        + 'samples, so a difference between the pictures is a difference in the fields.',
    },
    cells,
  };
}

/** The decision, stated above the panels it governs. Never inside a disclosure: it says how the
 *  pictures below may be read, so a reader who never opens anything still gets it. */
export function FigureComparisonNotice({ contract, label }: {
  contract: ComparisonContract;
  label: string;
}) {
  const { scale, cells } = contract;
  return (
    <section aria-label={`Comparison contract for ${label}`}
      className={`figure-comparison rounded-lg border px-3 py-2 text-[11px] space-y-1 ${
        scale.shared
          ? 'border-teal-500/25 bg-teal-500/5 text-teal-200/90'
          : 'border-amber-500/25 bg-amber-500/5 text-amber-200/90'}`}>
      <p>{scale.reason}</p>
      <p className="text-slate-400">{cells.reason}</p>
    </section>
  );
}

/**
 * One address shared by every panel in a linked group, so inspecting a sample in one figure
 * inspects the corresponding sample in the others. Returned unlinked when the grids differ,
 * because moving a shared cursor across panels that do not correspond would be the lie the
 * contract exists to prevent.
 */
export function useLinkedAddress(contract: ComparisonContract) {
  const [address, setAddress] = useState({ row: 0, column: 0 });
  return useMemo(
    () => (contract.cells.linked ? { address, onAddressChange: setAddress } : {}),
    [contract.cells.linked, address]);
}
