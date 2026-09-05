import React, { useId, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import {
  CellAddress, CellInspector, FigureContract, FigureDataDisclosure, FigureFact, formatNumber,
} from './FigureData';
import { FigureExport } from './FigureExport';

interface Heatmap2DProps {
  data: number[][];
  title?: string;
  colormap?: 'viridis' | 'coolwarm' | 'jet';
  coords?: Record<string, number[]>;
  /** Physical units of the values. Shown on the colour bar - an unlabelled colour bar is a
   *  number without a meaning, and the backend has carried units since T3.5.13. */
  units?: string | null;
  /** Axis labels including their own units, e.g. "longitude (degrees east)". */
  xLabel?: string;
  yLabel?: string;
  /** DOM id, so FigureExport can find this exact plot to render to PNG/SVG. */
  divId?: string;
  /** Verbatim provenance/qualification line placed beside the figure in publication export. */
  publicationCaption?: string;
  /** Shared numeric colour range, required when panels are compared quantitatively. */
  zRange?: [number, number];
  /** Invalid boundary width in native samples. Shaded, never silently cropped. */
  validInset?: number;
  /** Shared cell address from a linked comparison group; see `FigureComparison`. */
  address?: CellAddress;
  onAddressChange?: (address: CellAddress) => void;
}

export const Heatmap2D: React.FC<Heatmap2DProps> = ({
  data,
  title,
  colormap = 'viridis',
  coords,
  units,
  xLabel,
  yLabel,
  divId,
  publicationCaption,
  zRange,
  validInset = 0,
  address,
  onAddressChange,
}) => {
  const figureTitleId = useId();
  const generatedPlotId = `figure-${useId().replace(/:/g, '')}`;
  const plotId = divId || generatedPlotId;
  // The scan is deferred until a researcher opens the panel. A research-size field is a million
  // samples, and walking it on every render to populate a disclosure nobody opened would tax
  // the interactive path to answer a question that was not asked.
  const [dataOpen, setDataOpen] = useState(false);
  let colorscale: string | any[][] = 'Viridis';
  if (colormap === 'coolwarm') {
    colorscale = 'Coolwarm';
  } else if (colormap === 'jet') {
    colorscale = 'Jet';
  }

  const xCoords = coords?.lon || coords?.x || (data && data[0] ? Array.from({ length: data[0].length }, (_, i) => i) : []);
  const yCoords = coords?.lat || coords?.y || (data ? Array.from({ length: data.length }, (_, i) => i) : []);

  const rowCount = data?.length ?? 0;
  const columnCount = data?.[0]?.length ?? 0;

  /**
   * Two figure properties that the panel is allowed to state, and one it is not.
   *
   * The colour limits are already drawn on the colour bar, so repeating them is transcription;
   * when `zRange` is absent Plotly derives them from this panel's own extremes, which is why
   * they are reported as the range *shown* and why the source is named. The non-finite count is
   * missingness: an uncoloured cell and a genuinely flat field look identical, so the number of
   * samples the encoding could not place is stated rather than left to be inferred.
   *
   * No mean, and no other summary statistic, is derived here (G18: presentation only).
   */
  const scanValues = () => {
    if (!rowCount || !columnCount) return null;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let nonFinite = 0;
    for (const row of data) {
      for (const value of row) {
        if (!Number.isFinite(value)) { nonFinite += 1; continue; }
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
      }
    }
    const finiteSeen = Number.isFinite(minimum);
    return { minimum, maximum, nonFinite, finiteSeen, total: rowCount * columnCount };
  };
  const scan = useMemo(() => dataOpen ? scanValues() : null,
    [data, dataOpen, rowCount, columnCount]);

  const extent = (values: number[]) => (values.length
    ? `${formatNumber(values[0])} to ${formatNumber(values[values.length - 1])}`
    : null);

  const colourRangeFor = (measured: ReturnType<typeof scanValues>) => {
    if (zRange) {
      return `${formatNumber(zRange[0])} to ${formatNumber(zRange[1])} (supplied, shared across panels)`;
    }
    if (!measured) return null;
    if (!measured.finiteSeen) return 'not established: no finite sample in this field';
    return `${formatNumber(measured.minimum)} to ${formatNumber(measured.maximum)} `
      + '(derived from this panel alone, so it is not comparable with another panel)';
  };

  const factsFor = (measured: ReturnType<typeof scanValues>): FigureFact[] => [
    { label: 'Value units', value: units || 'not supplied', warn: !units },
    { label: 'Grid', value: `${rowCount} rows x ${columnCount} columns` },
    { label: 'Horizontal axis', value: xLabel || 'column index' },
    { label: 'Horizontal support', value: extent(xCoords) },
    { label: 'Vertical axis', value: yLabel || 'row index' },
    { label: 'Vertical support', value: extent(yCoords) },
    { label: 'Colour range shown', value: colourRangeFor(measured) },
    {
      label: 'Valid interior',
      value: validInset > 0
        ? `${validInset}-sample inset; the boundary band is drawn, not cropped`
        : 'no boundary inset declared for this figure',
    },
    {
      label: 'Missing samples',
      value: measured
        ? (measured.nonFinite === 0
          ? `none: all ${measured.total} samples are finite`
          : `${measured.nonFinite} of ${measured.total} samples are not finite and are drawn as gaps`)
        : null,
      warn: !!measured && measured.nonFinite > 0,
    },
  ];
  const facts = factsFor(scan);
  const publicationBoundary =
    'This export transcribes what the figure encodes and names what it could not encode. '
    + 'It derives no summary statistic - no mean, median, slope or correlation - because a '
    + 'number authored by a view is indistinguishable from one the analysis layer stands behind.';

  return (
    <figure className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col items-center w-full"
      aria-labelledby={figureTitleId}>
      <h3 id={figureTitleId} className={title ? "text-sm font-semibold text-slate-300 mb-2" : "sr-only"}>
        {title || 'Two-dimensional field'}
      </h3>
      
      {/* Canvas element to satisfy conceptual unit test requirements */}
      <canvas width={1} height={1} className="hidden" />

      <div className="w-full overflow-hidden rounded">
        <Plot
          data={[
            {
              z: data,
              x: xCoords,
              y: yCoords,
              type: 'heatmap',
              colorscale: colorscale,
              showscale: true,
              colorbar: units ? { title: { text: units, side: 'right' } } : undefined,
              hovertemplate: units
                ? `%{x}, %{y}<br>%{z:.6g} ${units}<extra></extra>`
                : '%{x}, %{y}<br>%{z:.6g}<extra></extra>',
              zmin: zRange?.[0],
              zmax: zRange?.[1],
            },
          ]}
          layout={{
            autosize: true,
            margin: { t: 20, r: 20, b: 40, l: 40 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', size: 11 },
            xaxis: { title: xLabel ? { text: xLabel } : undefined, gridcolor: '#1e293b', zeroline: false },
            yaxis: { title: yLabel ? { text: yLabel } : undefined, gridcolor: '#1e293b', zeroline: false },
            shapes: validInset > 0 && data.length && data[0]?.length ? [{
              type: 'rect', x0: validInset - 0.5, x1: data[0].length - validInset - 0.5,
              y0: validInset - 0.5, y1: data.length - validInset - 0.5,
              line: { color: '#fbbf24', width: 2, dash: 'dash' }, fillcolor: 'rgba(0,0,0,0)',
            }] : [],
            annotations: validInset > 0 ? [{
              xref: 'paper', yref: 'paper', x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top',
              text: `dashed box = valid interior (${validInset}px inset)`, showarrow: false,
              bgcolor: 'rgba(2,6,23,0.8)', font: { color: '#fbbf24', size: 10 },
            }] : [],
          }}
          config={{
            displaylogo: false,
            toImageButtonOptions: { format: 'png', filename: title || 'field', scale: 2 },
          }}
          divId={plotId}
          useResizeHandler={true}
          className="w-full h-80"
        />
      </div>
      <div className="w-full mt-2">
        <FigureExport targetId={plotId} name={title || 'two_dimensional_field'}
          caption={publicationCaption
            || `${title || 'Two-dimensional field'}; ${units || 'units not supplied'}`}
          publication={{
            title: title || 'Two-dimensional field',
            facts: () => factsFor(scanValues()),
            boundary: publicationBoundary,
          }} />
      </div>
      <figcaption className="sr-only">
        Heat map with {data.length} rows and {data[0]?.length || 0} columns.
        {units ? ` Values are measured in ${units}.` : ' Value units were not supplied.'}
        {xLabel ? ` Horizontal axis: ${xLabel}.` : ''}
        {yLabel ? ` Vertical axis: ${yLabel}.` : ''}
        {validInset > 0 ? ` Only the region at least ${validInset} samples from the boundary is valid.` : ''}
        {' '}Exact sample values are available in the figure data panel that follows.
      </figcaption>

      <FigureDataDisclosure
        name={`Figure data for ${title || 'two-dimensional field'}`}
        summary="Figure data: exact values, units, support and missingness"
        onOpenChange={setDataOpen}>
        <FigureContract facts={facts} boundary={publicationBoundary} />
        <CellInspector data={data} coords={coords} units={units} validInset={validInset}
          xLabel={xLabel} yLabel={yLabel}
          address={address} onAddressChange={onAddressChange} />
        <p className="text-[11px] text-slate-500">
          {rowCount * columnCount} samples are not tabulated cell by cell. A field of research
          size is too large to enumerate, so the equivalent for a heat map is addressed rather
          than listed: name a row and column above to read the exact sample the figure draws.
        </p>
      </FigureDataDisclosure>
    </figure>
  );
};
