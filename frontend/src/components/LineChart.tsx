import React, { useId, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import {
  FigureContract, FigureDataDisclosure, FigureFact, FigureTable, formatNumber,
} from './FigureData';

/** Beyond this many rows the table is capped and says so; silent truncation is the failure. */
const MAX_TABULATED_POINTS = 2000;

interface LineSeries {
  name: string;
  x: number[];
  y: number[];
  color?: string;
}

interface LineChartProps {
  series: LineSeries[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  /** DOM id, so FigureExport can render this exact plot to PNG/SVG. */
  divId?: string;
  /** Log axes: a power-law spectrum is a straight line only on log-log, and reading a slope
   *  off linear axes is how a Kolmogorov cascade gets mistaken for something else. */
  logX?: boolean;
  logY?: boolean;
}

export const LineChart: React.FC<LineChartProps> = ({
  series,
  title,
  xLabel,
  yLabel,
  divId,
  logX,
  logY,
}) => {
  const figureTitleId = useId();
  const [dataOpen, setDataOpen] = useState(false);
  // The empty-figure guard below runs after these hooks, so the contract is computed against a
  // safe list rather than assuming a caller supplied one.
  const safeSeries = series ?? [];

  /**
   * What the axes silently drop.
   *
   * A logarithmic axis is not a redrawing of the same data: Plotly discards every non-positive
   * sample, and it does so without a mark. A power spectrum that hit zero in three bins and a
   * power spectrum that was never sampled there produce the same picture, and the second is a
   * gap in coverage while the first is a measurement. The count of dropped samples is therefore
   * part of the figure's contract, not a diagnostic. Non-finite samples are counted for the
   * same reason - a break in a line reads as absence of structure.
   */
  const omissions = useMemo(() => {
    if (!dataOpen) return null;
    return safeSeries.map((item) => {
      let nonFinite = 0;
      let droppedByLogY = 0;
      let droppedByLogX = 0;
      const count = Math.min(item.x.length, item.y.length);
      for (let index = 0; index < count; index += 1) {
        const x = item.x[index];
        const y = item.y[index];
        if (!Number.isFinite(x) || !Number.isFinite(y)) { nonFinite += 1; continue; }
        if (logY && y <= 0) droppedByLogY += 1;
        if (logX && x <= 0) droppedByLogX += 1;
      }
      return { name: item.name, count, nonFinite, droppedByLogY, droppedByLogX };
    });
  }, [safeSeries, dataOpen, logX, logY]);

  const totalPoints = safeSeries.reduce(
    (sum, item) => sum + Math.min(item.x.length, item.y.length), 0);
  const droppedTotal = omissions
    ? omissions.reduce((sum, o) => sum + o.nonFinite + o.droppedByLogX + o.droppedByLogY, 0)
    : 0;

  const tableRows = useMemo(() => {
    if (!dataOpen) return [];
    const rows: (string | number | null)[][] = [];
    for (const item of safeSeries) {
      const count = Math.min(item.x.length, item.y.length);
      for (let index = 0; index < count && rows.length < MAX_TABULATED_POINTS; index += 1) {
        const x = item.x[index];
        const y = item.y[index];
        const drawn = Number.isFinite(x) && Number.isFinite(y)
          && !(logY && y <= 0) && !(logX && x <= 0);
        rows.push([
          item.name,
          index,
          Number.isFinite(x) ? formatNumber(x) : 'not finite',
          Number.isFinite(y) ? formatNumber(y) : 'not finite',
          drawn ? 'yes' : 'no',
        ]);
      }
    }
    return rows;
  }, [safeSeries, dataOpen, logX, logY]);

  const axisDescription = (label: string | undefined, log: boolean | undefined) => {
    const base = label || 'not labelled';
    return log ? `${base} (logarithmic; non-positive samples are not drawn)` : `${base} (linear)`;
  };

  const facts: FigureFact[] = [
    { label: 'Series', value: safeSeries.map((item) => item.name).join(', ') || 'none' },
    { label: 'Points drawn', value: `${totalPoints - droppedTotal} of ${totalPoints}`,
      warn: droppedTotal > 0 },
    { label: 'Horizontal axis', value: axisDescription(xLabel, logX) },
    { label: 'Vertical axis', value: axisDescription(yLabel, logY) },
    ...(omissions || []).filter(
      (o) => o.nonFinite || o.droppedByLogX || o.droppedByLogY).map((o) => ({
      label: `Omitted from ${o.name}`,
      value: [
        o.nonFinite ? `${o.nonFinite} not finite` : null,
        o.droppedByLogX ? `${o.droppedByLogX} non-positive on a log horizontal axis` : null,
        o.droppedByLogY ? `${o.droppedByLogY} non-positive on a log vertical axis` : null,
      ].filter(Boolean).join('; '),
      warn: true,
    })),
  ];

  if (!series || series.length === 0 || series.every(s => s.x.length === 0)) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 flex items-center justify-center h-64 text-slate-500 w-full">
        No data to display
      </div>
    );
  }

  const plotData = series.map((s, idx) => ({
    x: s.x,
    y: s.y,
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: s.name,
    line: {
      color: s.color || `hsl(${(idx * 137.5) % 360}, 70%, 50%)`,
      width: 2,
    },
    marker: {
      size: 6,
    },
  }));

  return (
    <figure className="bg-slate-900 border border-slate-800 rounded-lg p-4 relative w-full"
      aria-labelledby={figureTitleId}>
      <h3 id={figureTitleId} className={title ? "text-sm font-semibold text-slate-300 mb-2" : "sr-only"}>
        {title || 'Line chart'}
      </h3>
      <div className="w-full overflow-hidden rounded">
        <Plot
          data={plotData}
          layout={{
            autosize: true,
            margin: { t: 20, r: 20, b: 40, l: 50 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', size: 11 },
            xaxis: {
              title: xLabel ? { text: xLabel } : undefined,
              type: logX ? 'log' : undefined,
              gridcolor: '#1e293b',
              zeroline: false,
            },
            yaxis: {
              title: yLabel ? { text: yLabel } : undefined,
              type: logY ? 'log' : undefined,
              gridcolor: '#1e293b',
              zeroline: false,
            },
            legend: {
              orientation: 'h',
              y: -0.2,
              x: 0.5,
              xanchor: 'center',
              font: { color: '#cbd5e1' },
            },
          }}
          config={{
            displaylogo: false,
            toImageButtonOptions: { format: 'png', filename: title || 'figure', scale: 2 },
          }}
          divId={divId}
          useResizeHandler={true}
          className="w-full h-80"
        />
      </div>
      <figcaption className="sr-only">
        {series.length} series: {series.map(item => item.name).join(', ')}.
        {xLabel ? ` Horizontal axis: ${xLabel}${logX ? ', logarithmic scale' : ''}.` : ''}
        {yLabel ? ` Vertical axis: ${yLabel}${logY ? ', logarithmic scale' : ''}.` : ''}
        {' '}Exact point values are available in the figure data panel that follows.
      </figcaption>

      <FigureDataDisclosure
        name={`Figure data for ${title || 'line chart'}`}
        summary="Figure data: exact values, axis scales and omitted points"
        onOpenChange={setDataOpen}>
        <FigureContract facts={facts} boundary={
          'This panel transcribes what the figure encodes and names what it could not encode. '
          + 'It derives no summary statistic - no mean, slope or fitted exponent - because a '
          + 'number authored by a view is indistinguishable on screen from one the analysis layer '
          + 'stands behind.'} />
        <FigureTable
          caption={`Every point behind ${title || 'this figure'}`}
          columns={['series', 'index', xLabel || 'x', yLabel || 'y', 'drawn']}
          rows={tableRows}
          note={totalPoints > MAX_TABULATED_POINTS
            ? `Showing the first ${MAX_TABULATED_POINTS} of ${totalPoints} points. The rest are `
              + 'in the figure but not in this table.'
            : undefined} />
      </FigureDataDisclosure>
    </figure>
  );
};
