import React, { useId } from 'react';

/**
 * The text/table equivalent every figure owes (TG18.2).
 *
 * A Plotly figure encodes numbers as pixels and hands them back only through a hover tooltip:
 * mouse-only, ephemeral, and gone from any printed or exported copy. Until now the gridded
 * panels carried an `sr-only` caption describing the *shape* of the data - "heat map with 128
 * rows and 128 columns" - which tells a reader that a figure exists, not what it says. This
 * module supplies the missing half.
 *
 * ## What belongs here, and what deliberately does not
 *
 * G18 changes presentation and navigation only: it may not recompute, summarize, promote or
 * reinterpret a scientific value. That rules out the obvious temptation - a tidy row of mean,
 * median, slope and correlation under every plot - because those are new statistics authored by
 * the browser, and a number invented by a view is indistinguishable, once it is on screen, from
 * one the analysis layer stands behind.
 *
 * The line drawn instead is: **transcribe what the figure already encodes, and state what the
 * figure could not encode.**
 *
 *   - A cell's value is the encoding itself, made readable and keyboard-reachable rather than
 *     mouse-only. It is transcription.
 *   - An axis range and a colour-bar range are already drawn on the figure. Repeating them in
 *     text is transcription; that is why the colour range is labelled as the range *shown on
 *     this figure* rather than as the range *of the data*, and why it says whether the limits
 *     were supplied for comparison or derived from this panel alone.
 *   - Non-finite samples and points dropped by a logarithmic axis are what the encoding
 *     silently omits. A gap in a line and an uncoloured cell look identical to absence of
 *     structure, so the count is stated. This is missingness, not summary.
 *   - A mean is none of those things. It appears nowhere on the figure, and it is not here.
 *
 * That boundary is printed in the panel itself rather than kept in this comment, so a reader
 * who wonders why there is no mean is answered on the page.
 */

/** One stated fact about the figure's contract. `value` of `null` renders as an em dash. */
export interface FigureFact {
  label: string;
  value: string | null;
  /** Rendered in amber: something the figure could not show, rather than a neutral property. */
  warn?: boolean;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : String(value);
  // Six significant figures matches the hover template the figures already use, so the text
  // equivalent and the tooltip cannot disagree about the same sample.
  return Number(value.toPrecision(6)).toString();
}

/**
 * The disclosure wrapper. `<details>` groups carry an explicit accessible name: an unnamed one
 * is announced only as "disclosure triangle", which is the TG17.8 defect that recurred in a
 * second view during TG17.10 (D82).
 */
export function FigureDataDisclosure({ name, summary, children, onOpenChange }: {
  name: string;
  summary: string;
  children: React.ReactNode;
  /** Lets a figure defer an expensive scan of its own data until the panel is actually opened. */
  onOpenChange?: (open: boolean) => void;
}) {
  return (
    <details aria-label={name}
      onToggle={(event) => onOpenChange?.((event.currentTarget as HTMLDetailsElement).open)}
      className="figure-data w-full mt-2 rounded-lg border border-slate-800 bg-slate-950/60">
      <summary className="cursor-pointer px-3 py-2 text-xs text-slate-300 select-none">
        {summary}
      </summary>
      <div className="px-3 pb-3 space-y-3">{children}</div>
    </details>
  );
}

/** The figure's stated contract: units, axes, support, normalization, missingness. */
export function FigureContract({ facts, boundary }: {
  facts: FigureFact[];
  boundary: string;
}) {
  return (
    <>
      <dl className="figure-contract grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
        {facts.map((fact) => (
          <div key={fact.label} className="contents">
            <dt className="text-slate-500">{fact.label}</dt>
            <dd className={`font-mono ${fact.warn ? 'text-amber-400' : 'text-slate-300'}`}>
              {fact.value === null ? '—' : fact.value}
            </dd>
          </div>
        ))}
      </dl>
      <p className="text-[11px] text-slate-500">{boundary}</p>
    </>
  );
}

/**
 * The tabular equivalent. Conventions match `AccessibleTable` in the comparison views - caption,
 * column scope, monospaced cells, em dash for an absent value - so a researcher meets one table
 * idiom across the platform rather than two.
 */
export function FigureTable({ caption, columns, rows, note }: {
  caption: string;
  columns: string[];
  rows: (string | number | null)[][];
  note?: string;
}) {
  return (
    <div className="overflow-auto max-h-80 rounded border border-slate-900">
      <table className="w-full text-xs border-collapse">
        <caption className="text-left text-[11px] text-slate-400 px-2 py-1">{caption}</caption>
        <thead className="sticky top-0 bg-slate-950">
          <tr className="text-slate-400">
            {columns.map((column) => (
              <th key={column} scope="col"
                className="text-left font-normal border-b border-slate-800 px-2 py-1">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="text-slate-200">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="border-b border-slate-900 px-2 py-1 font-mono">
                  {cell === null || cell === undefined || cell === '' ? '—' : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className="px-2 py-1 text-[11px] text-amber-400">{note}</p>}
    </div>
  );
}

/**
 * Exact-value inspection for a gridded field, by index rather than by pointer.
 *
 * A field of any research size cannot be tabulated cell by cell - a 721x1440 ERA5 crop is over a
 * million samples - so the equivalent for a heat map is addressed rather than enumerated. The
 * researcher names a row and a column and reads the exact sample back, with its coordinates,
 * its units and whether it falls inside the valid interior. That is what the hover tooltip
 * gives a mouse, offered to a keyboard and preserved in a printed copy.
 */
export function CellInspector({ data, coords, units, validInset, xLabel, yLabel }: {
  data: number[][];
  coords?: Record<string, number[]>;
  units?: string | null;
  validInset: number;
  xLabel?: string;
  yLabel?: string;
}) {
  const rowId = useId();
  const columnId = useId();
  const [row, setRow] = React.useState(0);
  const [column, setColumn] = React.useState(0);

  const rowCount = data.length;
  const columnCount = data[0]?.length ?? 0;
  const inRange = row >= 0 && row < rowCount && column >= 0 && column < columnCount;
  const sample = inRange ? data[row][column] : undefined;

  const xCoordinate = coords?.lon?.[column] ?? coords?.x?.[column];
  const yCoordinate = coords?.lat?.[row] ?? coords?.y?.[row];
  // The valid interior is an inset in native samples, so validity is an index question and is
  // answered the same way whether or not coordinates were supplied.
  const insideValid = validInset <= 0 || (
    row >= validInset && row < rowCount - validInset &&
    column >= validInset && column < columnCount - validInset);

  const clamp = (value: string, limit: number) => {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return 0;
    return Math.min(Math.max(parsed, 0), Math.max(limit - 1, 0));
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor={rowId} className="text-[11px] text-slate-500">
            Row index (0 to {Math.max(rowCount - 1, 0)})
          </label>
          <input id={rowId} type="number" min={0} max={Math.max(rowCount - 1, 0)} value={row}
            onChange={(event) => setRow(clamp(event.target.value, rowCount))}
            className="w-28 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-slate-200" />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor={columnId} className="text-[11px] text-slate-500">
            Column index (0 to {Math.max(columnCount - 1, 0)})
          </label>
          <input id={columnId} type="number" min={0} max={Math.max(columnCount - 1, 0)} value={column}
            onChange={(event) => setColumn(clamp(event.target.value, columnCount))}
            className="w-28 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-slate-200" />
        </div>
      </div>
      <output aria-live="polite"
        className="block rounded border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs">
        {!inRange || sample === undefined ? (
          <span className="text-slate-400">
            No sample at row {row}, column {column}: the field is {rowCount} by {columnCount}.
          </span>
        ) : (
          <span className="space-y-1 block">
            <span className="block font-mono text-slate-100">
              {Number.isFinite(sample) ? formatNumber(sample) : 'not a finite number'}
              {units && Number.isFinite(sample) ? ` ${units}` : ''}
            </span>
            <span className="block text-[11px] text-slate-400">
              row {row}, column {column}
              {yCoordinate !== undefined
                ? ` · ${yLabel || 'vertical'} ${formatNumber(yCoordinate)}` : ''}
              {xCoordinate !== undefined
                ? ` · ${xLabel || 'horizontal'} ${formatNumber(xCoordinate)}` : ''}
            </span>
            {validInset > 0 && (
              <span className={`block text-[11px] ${insideValid ? 'text-slate-400' : 'text-amber-400'}`}>
                {insideValid
                  ? `Inside the valid interior (${validInset}-sample inset).`
                  : `Outside the valid interior: within ${validInset} samples of the boundary, `
                    + 'so this sample carries the boundary artefact the inset marks.'}
              </span>
            )}
          </span>
        )}
      </output>
    </div>
  );
}
