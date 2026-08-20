import React, { useRef, useState } from 'react';
import { Upload, Loader2, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

/**
 * Load a user-supplied field into the platform (T3.5.24).
 *
 * Two steps, deliberately. An ERA5 NetCDF is `(time, level, lat, lon)` — there is no single
 * field in it. Picking `[0, 0]` silently would import a slice the researcher did not choose,
 * and every statistic computed afterwards would describe that arbitrary timestep. So the file
 * is inspected first, the extra dimensions are shown, and nothing is loaded until they are
 * pinned.
 *
 * Origin is never assumed. An uploaded file reports `is_simulated: null` unless the file
 * itself declares otherwise — the platform did not produce this data and claiming it is
 * observational would be inventing a fact.
 */

interface FieldImportProps {
  onLoaded: (result: types.ImportFieldResponse) => void;
  onError?: (message: string) => void;
}

export const FieldImport: React.FC<FieldImportProps> = ({ onLoaded, onError }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [report, setReport] = useState<types.ImportInspectResponse | null>(null);
  const [variable, setVariable] = useState<string>('');
  const [selection, setSelection] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);

  const reset = () => {
    setFile(null);
    setReport(null);
    setVariable('');
    setSelection({});
    if (inputRef.current) inputRef.current.value = '';
  };

  const choose = async (chosen: File | null) => {
    if (!chosen) return;
    setFile(chosen);
    setReport(null);
    setBusy(true);
    try {
      const inspected = await apiService.importInspect(chosen);
      setReport(inspected);
      setVariable(inspected.default_variable);
      // Index 0 is offered as a *starting value the researcher can see and change*, not as a
      // silent default: the difference is that it appears on screen and is recorded in the
      // provenance of whatever is loaded.
      const first = inspected.variables[inspected.default_variable];
      setSelection(Object.fromEntries(Object.keys(first?.extra_dims || {}).map(d => [d, 0])));
    } catch (e: any) {
      onError?.(e.message);
      reset();
    } finally {
      setBusy(false);
    }
  };

  const load = async () => {
    if (!file) return;
    setBusy(true);
    try {
      onLoaded(await apiService.importField(file, variable, selection));
    } catch (e: any) {
      onError?.(e.message);
    } finally {
      setBusy(false);
    }
  };

  const chosenVar = report?.variables?.[variable];
  const extraDims = chosenVar?.extra_dims || {};

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
      <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center gap-2">
        <Upload className="w-4 h-4 text-teal-400" /> Import a field
      </h3>
      <p className="text-[11px] text-slate-500 leading-relaxed">
        NetCDF (<code>.nc</code>), zipped Zarr (<code>.zarr.zip</code>), CSV or JSON — including
        anything this platform exported. The file is inspected before anything is loaded.
      </p>

      <input
        ref={inputRef}
        type="file"
        accept=".nc,.nc4,.netcdf,.cdf,.csv,.json,.zip"
        onChange={(e) => choose(e.target.files?.[0] || null)}
        className="block w-full text-[11px] text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border file:border-slate-700 file:text-[11px] file:font-semibold file:bg-slate-950 file:text-teal-400 hover:file:bg-slate-900"
      />

      {busy && (
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <Loader2 className="w-3.5 h-3.5 animate-spin" /> Reading file…
        </div>
      )}

      {report && (
        <div className="space-y-3">
          <div className="text-[10px] font-mono bg-slate-950 border border-slate-800 rounded p-2 text-slate-400 space-y-0.5">
            <div className="flex justify-between"><span>format</span><span className="text-slate-300">{report.format}</span></div>
            <div className="flex justify-between"><span>size</span><span className="text-slate-300">{(report.bytes / 1e6).toFixed(2)} MB</span></div>
            <div className="flex justify-between"><span>sha256</span><span className="text-slate-300">{report.content_hash.slice(0, 16)}…</span></div>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1">Variable</label>
            <select
              value={variable}
              onChange={(e) => {
                setVariable(e.target.value);
                const next = report.variables[e.target.value];
                setSelection(Object.fromEntries(
                  Object.keys(next?.extra_dims || {}).map(d => [d, 0])));
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 font-mono"
            >
              {Object.entries(report.variables).map(([name, v]) => (
                <option key={name} value={name}>
                  {name} [{v.dims.join(', ')}]{v.units ? ` — ${v.units}` : ''}
                </option>
              ))}
            </select>
          </div>

          {Object.keys(extraDims).length > 0 ? (
            <div className="space-y-2">
              <div className="text-[11px] text-amber-400 flex gap-2 leading-relaxed">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span>
                  This variable is {chosenVar?.dims.length}D. Pin every non-spatial axis — the
                  index you choose is recorded in the provenance, because a statistic computed
                  from an unnamed slice describes a timestep nobody selected.
                </span>
              </div>
              {Object.entries(extraDims).map(([dim, size]) => (
                <div key={dim}>
                  <label className="text-xs text-slate-400 flex justify-between mb-1">
                    <span className="font-mono">{dim}</span>
                    <span className="text-slate-500">
                      {selection[dim] ?? 0} of 0…{size - 1}
                      {report.coords?.[dim]?.[selection[dim] ?? 0] !== undefined &&
                        ` (${report.coords[dim][selection[dim] ?? 0]})`}
                    </span>
                  </label>
                  <input
                    type="range" min={0} max={size - 1} step={1}
                    value={selection[dim] ?? 0}
                    onChange={(e) => setSelection({ ...selection, [dim]: parseInt(e.target.value, 10) })}
                    className="w-full accent-teal-500"
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-emerald-400 flex gap-2">
              <CheckCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>Already 2D — nothing to select.</span>
            </div>
          )}

          <button
            onClick={load}
            disabled={busy}
            className="w-full bg-teal-600 hover:bg-teal-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-sm"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            Load into field buffer
          </button>
        </div>
      )}
    </div>
  );
};
