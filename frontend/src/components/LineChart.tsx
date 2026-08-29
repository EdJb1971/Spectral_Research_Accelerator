import React, { useId } from 'react';
import Plot from 'react-plotly.js';

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
            font: { color: '#94a3b8', size: 10 },
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
      </figcaption>
    </figure>
  );
};
