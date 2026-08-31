import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, Table2, Waypoints } from 'lucide-react';
import * as types from '../types/api';
import { apiService } from '../services/api';

/**
 * TG17.8 The linked comparison views, rendered from the served payload.
 *
 * Nothing in this file decides what a view may conclude, which axes it is drawn on, or whether a
 * reading is admissible. Those are scientific facts and they live on the server, for the reason
 * every G17 slice has given: a boundary a component computes is a second boundary, and the one
 * a reader sees would not be the one a receipt could show them.
 *
 * Two rules this file *does* enforce, because they are rendering decisions and can only be made
 * here:
 *
 *  1. A coverage cell is drawn from its named state, never from a number. There is deliberately
 *     no numeric path into `CELL_STYLE`: absent support has no width to be zero, so it cannot be
 *     mistaken for a measured zero by anyone reading the picture.
 *  2. A role is drawn in all three of its channels - the colour swatch, the marker glyph and the
 *     word. The colour comes from the payload and the word is printed beside it, so a reader who
 *     cannot use the colour loses nothing that carries meaning.
 *
 * Every view prints its accessible table under the visual, from `payload.table`, which the
 * server builds from the same values it built the body from.
 */

const CELL_STYLE: Record<string, { box: string; label: string }> = {
  COVERED: { box: 'bg-sky-600/70 border-sky-400/50', label: 'covered' },
  SPARSE: { box: 'bg-sky-900/50 border-sky-700/60 [background-image:repeating-linear-gradient(45deg,transparent,transparent_3px,rgba(56,189,248,0.35)_3px,rgba(56,189,248,0.35)_6px)]', label: 'sparse' },
  ABSENT: { box: 'bg-slate-900 border-slate-700 border-dashed', label: 'absent' },
  REFUSED: { box: 'bg-rose-950/60 border-rose-600/60', label: 'refused' },
};

function CoverageCell({ cell }: { cell: any }) {
  const style = CELL_STYLE[cell.state] || CELL_STYLE.ABSENT;
  return (
    <td className="p-1 align-middle">
      <div className={`h-6 rounded border ${style.box} flex items-center justify-center`}
           title={cell.reason}>
        {/* The state is printed, not only shaded. A pattern alone would leave the distinction
            between absent and refused to a reader's eye for texture. */}
        <span className="text-[9px] uppercase tracking-wider text-slate-100/90">{style.label}</span>
      </div>
    </td>
  );
}

export function Legend({ encodings }: { encodings: types.ComparisonEncoding[] }) {
  return (
    <ul aria-label="Mark legend" className="flex flex-wrap gap-3 text-xs text-slate-300">
      {encodings.map((item) => (
        <li key={item.role} className="flex items-center gap-1.5"
            title={item.definition}>
          <span aria-hidden className="inline-block h-3 w-3 rounded-sm border border-slate-600"
                style={{ background: item.colour }} />
          <span className="text-slate-400">{item.marker.split('_').join(' ')}</span>
          <span className="text-slate-100">{item.word}</span>
          {!item.admits_claim && (
            <span className="text-[10px] text-amber-300/80">(no claim)</span>
          )}
        </li>
      ))}
    </ul>
  );
}

export function AccessibleTable({ table, caption }: {
  table: types.ComparisonTable;
  caption: string;
}) {
  if (!table.columns.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <caption className="text-left text-[11px] text-slate-400 pb-1">{caption}</caption>
        <thead>
          <tr className="text-slate-400">
            {table.columns.map((column) => (
              <th key={column} scope="col"
                  className="text-left font-normal border-b border-slate-800 px-2 py-1">
                {column.split('_').join(' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, index) => (
            <tr key={index} className="text-slate-200">
              {table.columns.map((column) => (
                <td key={column} className="border-b border-slate-900 px-2 py-1 font-mono">
                  {row[column] === null || row[column] === undefined || row[column] === ''
                    ? '—' : String(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** What the view may and may not be read as, printed with the view rather than in a footnote. */
export function ClaimBoundary({ view }: { view: types.ComparisonView }) {
  return (
    <div className="text-xs border border-slate-800 rounded p-2 bg-slate-900/40 space-y-1">
      <p className="text-slate-300">
        <span className="text-emerald-300">May conclude: </span>{view.may_conclude}
      </p>
      <p className="text-slate-300">
        <span className="text-rose-300">May not conclude: </span>
        {view.may_not_conclude.join('; ')}.
      </p>
      <p className="text-slate-500">{view.mode_forbids}</p>
    </div>
  );
}

function CoverageTimelineBody({ view, onSelectWindow }: {
  view: types.ComparisonView;
  onSelectWindow: (window: string) => void;
}) {
  const body = view.body;
  return (
    <div className="space-y-2">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-400">
            <th scope="col" className="text-left font-normal px-2">domain</th>
            {body.windows.map((window: any) => (
              <th key={window.name} scope="col" className="font-normal px-1">
                <button type="button" onClick={() => onSelectWindow(window.name)}
                        className="underline decoration-dotted hover:text-sky-300"
                        aria-label={`Highlight what contributes to ${window.name}`}>
                  {window.name}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.rows.map((row: any) => (
            <tr key={row.domain}>
              <th scope="row" className="text-left font-normal text-slate-200 px-2 py-1">
                {row.domain}
                <span className="block text-[10px] text-slate-500">{row.support_kind}</span>
              </th>
              {row.cells.map((cell: any) => (
                <CoverageCell key={cell.window} cell={cell} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-slate-400">{body.zero_distinction}</p>
    </div>
  );
}

function NativePreviewBody({ view }: { view: types.ComparisonView }) {
  return (
    <div className="grid gap-2 md:grid-cols-2">
      {view.body.rows.map((row: any) => (
        <div key={row.domain} className="border border-slate-800 rounded p-2 space-y-1">
          <p className="text-slate-100 text-sm">{row.domain}</p>
          <p className="text-xs text-slate-300">
            Native: {row.native.measure} in <span className="font-mono">{row.native.units}</span>
            {' '}({row.native.semantics})
          </p>
          <p className="text-xs text-slate-300">
            Canonical: {row.canonical.role}, {row.canonical.axis.kind.split('_').join(' ')}
          </p>
          {row.canonical.violations.length > 0 && (
            <p className="text-xs text-amber-200">Breaks: {row.canonical.violations.join(', ')}</p>
          )}
          <p className="text-[11px] text-slate-500">{row.why_not}</p>
        </div>
      ))}
    </div>
  );
}

function ScaleMappingBody({ view }: { view: types.ComparisonView }) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-slate-400">{view.body.mapping_note}</p>
      {view.body.rows.map((row: any) => (
        <div key={row.coordinate} className="border border-slate-800 rounded p-2">
          <p className="text-xs text-slate-200">coordinate {row.coordinate}</p>
          <ul className="text-xs text-slate-300 mt-1 space-y-0.5">
            {row.domains.map((entry: any) => (
              <li key={entry.domain}>
                {entry.domain}:{' '}
                {entry.resolvable
                  ? <span className="font-mono">{entry.native_duration_seconds} s native</span>
                  : <span className="text-amber-200">{entry.why_not}</span>}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function ResultMatrixBody({ view }: { view: types.ComparisonView }) {
  const correction = view.body.correction;
  return (
    <div className="space-y-2">
      <p className="text-xs text-slate-300">
        Corrected over <span className="font-mono">{correction.denominator}</span> members by{' '}
        {correction.method} at alpha <span className="font-mono">{correction.alpha}</span>, from a
        declared search of <span className="font-mono">{correction.declared_search_members}</span>.
      </p>
      <p className="text-[11px] text-slate-500">{correction.why_visible}</p>
      <ul className="space-y-1">
        {view.body.cells.map((cell: any, index: number) => (
          <li key={index} className="text-xs flex items-start gap-2">
            <span aria-hidden className="inline-block h-3 w-3 mt-0.5 rounded-sm shrink-0"
                  style={{ background: cell.mark.colour }} />
            <span className="font-mono text-slate-200">{cell.domains.join(' + ')}</span>
            <span className="text-slate-400">{cell.relationship}</span>
            <span className={cell.admissible ? 'text-slate-300' : 'text-rose-300'}>
              {cell.status}
            </span>
            <span className="text-slate-500">{cell.reason}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MotifBody({ view }: { view: types.ComparisonView }) {
  if (view.body.unavailable_reason) {
    return (
      <p className="text-xs text-slate-300 flex items-start gap-2">
        <Ban className="w-4 h-4 text-amber-300 shrink-0" aria-hidden />
        {view.body.unavailable_reason}
      </p>
    );
  }
  return (
    <div className="space-y-1">
      {view.body.rows.map((row: any, index: number) => (
        <p key={index} className="text-xs text-slate-300">
          <span className="font-mono text-slate-100">{row.motif}</span> onto {row.target_domain}:{' '}
          {row.status}
        </p>
      ))}
      <p className="text-[11px] text-slate-500">{view.body.transfer_note}</p>
    </div>
  );
}

function NullBody({ view }: { view: types.ComparisonView }) {
  const resolution = view.body.resolution;
  return (
    <div className="space-y-2">
      {view.body.nulls.map((item: any) => (
        <div key={item.name} className="border border-slate-800 rounded p-2 text-xs space-y-0.5">
          <p className="text-slate-100">{item.name} — {item.method}</p>
          <p className="text-slate-300">Preserves: {item.preserves.join(', ') || '—'}</p>
          <p className="text-slate-300">Destroys: {item.destroys.join(', ') || '—'}</p>
          <p className="text-slate-400">{item.distribution.status}: {item.distribution.reason}</p>
        </div>
      ))}
      <div className="border border-slate-800 rounded p-2 text-xs space-y-0.5">
        <p className="text-slate-200">
          p-value floor <span className="font-mono">{resolution.p_value_floor}</span> from{' '}
          <span className="font-mono">{resolution.declared_surrogates}</span> declared surrogates;{' '}
          <span className="font-mono">{resolution.surrogates_required}</span> required.
        </p>
        {!resolution.affordable && (
          <p className="text-rose-300 flex items-start gap-1">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" aria-hidden />
            {resolution.warning}
          </p>
        )}
        <p className="text-slate-500">{resolution.note}</p>
      </div>
    </div>
  );
}

function ProvenanceBody({ view }: { view: types.ComparisonView }) {
  return (
    <div className="space-y-2">
      {/* Each panel is labelled explicitly: a `<details>` is exposed as a group whose accessible
          name is *not* computed from its `<summary>`, so an unlabelled one is a region a
          screen-reader user meets with no name at all. */}
      {view.body.rows.map((row: any) => (
        <details key={row.domain} aria-label={row.domain}
                 className="border border-slate-800 rounded p-2 text-xs">
          <summary className="cursor-pointer text-slate-100">{row.domain}</summary>
          <dl className="mt-1 grid grid-cols-[10rem_1fr] gap-x-2 gap-y-0.5 text-slate-300">
            <dt className="text-slate-500">source</dt>
            <dd className="font-mono break-all">
              {row.source.source_id} @ {row.source.source_version}
            </dd>
            <dt className="text-slate-500">licence</dt>
            <dd>{row.source.licence}</dd>
            <dt className="text-slate-500">adapter</dt>
            <dd className="font-mono break-all">
              {row.adapter.adapter_id} @ {row.adapter.adapter_version}
            </dd>
            <dt className="text-slate-500">definition</dt>
            <dd className="font-mono break-all">{row.adapter.definition_sha256}</dd>
            <dt className="text-slate-500">support</dt>
            <dd>{row.support.support_kind}</dd>
            <dt className="text-slate-500">artefact</dt>
            <dd className="font-mono break-all">
              {row.artefact.sha256 || <span className="font-sans text-amber-200">
                {row.artefact.reason}</span>}
            </dd>
          </dl>
        </details>
      ))}
      <p className="text-[11px] text-slate-500 font-mono break-all">
        manifest {view.body.manifest_sha256}
      </p>
    </div>
  );
}

function ViewBody({ view, onSelectWindow }: {
  view: types.ComparisonView;
  onSelectWindow: (window: string) => void;
}) {
  switch (view.view_id) {
    case 'coverage_timeline': return <CoverageTimelineBody view={view} onSelectWindow={onSelectWindow} />;
    case 'native_preview': return <NativePreviewBody view={view} />;
    case 'scale_mapping': return <ScaleMappingBody view={view} />;
    case 'result_matrix': return <ResultMatrixBody view={view} />;
    case 'motif_correspondence': return <MotifBody view={view} />;
    case 'null_and_correction': return <NullBody view={view} />;
    case 'provenance_drilldown': return <ProvenanceBody view={view} />;
    // A view registered on the server that this build has no drawing for still renders: its
    // table, its legend and its boundary are all in the payload. Falling through to nothing
    // would make a new view look like a broken one.
    default: return null;
  }
}

/**
 * What these views will not draw, and a control that asks.
 *
 * The two reading lists are the whole of TG17.0's semantic-trap acceptance made operable. A
 * manifest may legitimately *declare* `causality` in calendar mode - a study holding an external
 * intervention design can test it - and no view here may *draw* it, because the design that
 * licenses the arrow has no field in the manifest. Serving both lists, and the reason, is what
 * stops that distinction from living in someone's memory.
 */
export function RefusalPanel({ contract, mode }: {
  contract: types.ComparisonViewsContract | null;
  mode: string;
}) {
  const [reading, setReading] = useState('');
  const [check, setCheck] = useState<types.ComparisonReadingCheck | null>(null);

  if (!contract) return null;
  const candidates = [
    ...(contract.readings.declarable_by_mode[mode] || []),
    ...contract.readings.never_admissible,
  ];

  const ask = async () => {
    if (!reading) return;
    try {
      setCheck(await apiService.comparisonCheckReading(mode, reading));
    } catch (problem) {
      setCheck(null);
    }
  };

  return (
    <details aria-label="What these views will not draw"
             className="border border-slate-800 rounded p-2 text-xs space-y-2">
      <summary className="cursor-pointer text-slate-200">What these views will not draw</summary>
      <ul className="list-disc pl-5 text-slate-300 space-y-0.5 mt-2">
        {Object.entries(contract.refusals).map(([name, why]) => (
          <li key={name}><span className="text-slate-100">{name.split('_').join(' ')}</span>: {why}</li>
        ))}
      </ul>
      <p className="text-slate-400">{contract.readings.why_two_lists}</p>
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <label htmlFor="reading-check" className="text-slate-300">Can this be drawn?</label>
        <select id="reading-check" value={reading} onChange={(event) => setReading(event.target.value)}
                className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-100">
          <option value="">Choose a reading</option>
          {candidates.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
        <button type="button" onClick={ask} disabled={!reading}
                className="border border-slate-700 rounded px-2 py-1 text-slate-200
                           hover:border-sky-600 disabled:opacity-50">
          Check this reading
        </button>
      </div>
      {check && (
        <p role="status" className={check.renderable ? 'text-emerald-300' : 'text-rose-300'}>
          {check.renderable
            ? `${check.reading} can be drawn in ${check.mode} mode. ${check.claim_boundary || ''}`
            : check.reason}
        </p>
      )}
    </details>
  );
}

export function LinkedSelectionPanel({ selection }: {
  selection: types.ComparisonLinkedSelection | null;
}) {
  if (!selection) return null;
  return (
    <div role="status" className="border border-sky-800/60 bg-sky-950/30 rounded p-2 space-y-1">
      <p className="text-xs text-slate-200 flex items-center gap-1.5">
        <Waypoints className="w-3.5 h-3.5" aria-hidden />
        Contributing native support for <span className="font-mono">{selection.window}</span>
      </p>
      <ul className="text-xs space-y-0.5">
        {selection.contributions.map((row) => (
          <li key={row.domain} className={row.contributes ? 'text-slate-200' : 'text-slate-500'}>
            <span className="font-mono">{row.domain}</span>: {row.state} — {row.reason}
          </li>
        ))}
      </ul>
      <p className="text-[11px] text-slate-400">{selection.why_not_merged}</p>
    </div>
  );
}

export function ComparisonViews({ manifest }: {
  manifest: types.CrossDomainExperimentManifest | null;
}) {
  const [views, setViews] = useState<types.ComparisonViewSet | null>(null);
  const [contract, setContract] = useState<types.ComparisonViewsContract | null>(null);
  const [legend, setLegend] = useState<
    { encodings: types.ComparisonEncoding[]; why_three_channels: string } | null>(null);
  const [selection, setSelection] = useState<types.ComparisonLinkedSelection | null>(null);
  const [showTables, setShowTables] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    apiService.comparisonViewsContract().then(setContract).catch(() => setContract(null));
    apiService.comparisonEncodings()
      .then((body) => setLegend({ encodings: body.encodings,
                                  why_three_channels: body.why_three_channels }))
      .catch(() => setLegend(null));
  }, []);

  const refresh = useCallback(async () => {
    if (!manifest) return;
    setBusy(true);
    setError('');
    try {
      setViews(await apiService.comparisonRenderAll(manifest));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    } finally {
      setBusy(false);
    }
  }, [manifest]);

  // The selection belongs to the manifest it was computed against. Clearing it when the plan
  // changes is not tidiness: a highlight left over from a previous plan would be pointing at
  // support that plan no longer declares.
  useEffect(() => { setSelection(null); refresh(); }, [refresh]);

  // One view at a time, for the case a researcher has changed nothing but wants to see this
  // panel again against the run that has moved on since the set was fetched.
  const refreshOne = async (viewId: string) => {
    if (!manifest) return;
    try {
      const fresh = await apiService.comparisonRenderView(viewId, manifest);
      setViews((current) => current && {
        ...current,
        views: current.views.map((row) => (row.view_id === viewId ? fresh : row)),
      });
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    }
  };

  const selectWindow = async (window: string) => {
    if (!manifest) return;
    try {
      setSelection(await apiService.comparisonLinkedSelection(manifest, window));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem));
    }
  };

  if (!manifest) {
    return <p className="text-sm text-slate-400">Load a manifest to inspect its comparisons.</p>;
  }

  return (
    <div className="space-y-4" aria-busy={busy}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-xs text-slate-400">
          Mode <span className="font-mono">{views?.mode || manifest.mode}</span>. Nothing here has
          been measured; these views show what the plan declares and what the run produced.
        </p>
        <button type="button" onClick={() => setShowTables((value) => !value)}
                aria-pressed={showTables}
                className="text-xs border border-slate-700 rounded px-2 py-1 flex items-center gap-1
                           text-slate-200 hover:border-sky-600">
          <Table2 className="w-3.5 h-3.5" aria-hidden />
          {showTables ? 'Hide the numbers behind each view' : 'Show the numbers behind each view'}
        </button>
      </div>

      {error && <p role="alert" className="text-xs text-rose-300">{error}</p>}

      {legend && (
        <div className="space-y-1">
          <Legend encodings={legend.encodings} />
          <p className="text-[11px] text-slate-500">{legend.why_three_channels}</p>
        </div>
      )}

      <RefusalPanel contract={contract} mode={views?.mode || manifest.mode} />

      <LinkedSelectionPanel selection={selection} />

      {(views?.views || []).map((view) => (
        <section key={view.view_id} aria-labelledby={`view-${view.view_id}`}
                 className="border border-slate-800 rounded p-3 space-y-2">
          <header className="space-y-0.5">
            <div className="flex items-start justify-between gap-2">
              <h4 id={`view-${view.view_id}`} className="text-sm text-slate-100">
                {view.ordinal}. {view.title}
              </h4>
              <button type="button" onClick={() => refreshOne(view.view_id)}
                      aria-label={`Refresh ${view.title}`}
                      className="text-[11px] border border-slate-700 rounded px-2 py-0.5
                                 text-slate-300 hover:border-sky-600 shrink-0">
                Refresh
              </button>
            </div>
            <p className="text-xs text-slate-400">{view.question}</p>
            <p className="text-[11px] text-slate-500">
              Axes: {view.axes.map((axis) => `${axis.name} (${axis.kind.split('_').join(' ')})`)
                .join(', ')}
            </p>
          </header>
          <Legend encodings={view.legend} />
          <ViewBody view={view} onSelectWindow={selectWindow} />
          {showTables && (
            <AccessibleTable table={view.table}
                             caption={`Numbers behind ${view.title.toLowerCase()}`} />
          )}
          <ClaimBoundary view={view} />
        </section>
      ))}
    </div>
  );
}
