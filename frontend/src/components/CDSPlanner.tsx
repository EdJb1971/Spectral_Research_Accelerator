/** Metadata-only browser planner for the production Copernicus CDS acquisition route. */
import { useEffect, useState } from 'react';
import { Calculator, CheckCircle, Loader2, ShieldCheck } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface CDSPlannerProps { onError?: (message: string) => void }

const numberList = (value: string) => value.split(',').map((item) => Number(item.trim()))
  .filter(Number.isFinite);
const bytes = (value: number) => value >= 1e9
  ? `${(value / 1e9).toFixed(2)} GB`
  : `${(value / 1e6).toFixed(1)} MB`;

export default function CDSPlanner({ onError }: CDSPlannerProps) {
  const [capabilities, setCapabilities] = useState<types.CDSCapabilities | null>(null);
  const [request, setRequest] = useState<types.CDSPlanRequest | null>(null);
  const [plan, setPlan] = useState<types.CDSPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;
    apiService.cdsCapabilities().then((value) => {
      if (!live) return;
      setCapabilities(value); setRequest(value.defaults);
    }).catch((cause) => {
      const message = cause instanceof Error ? cause.message : String(cause);
      if (live) setError(message); onError?.(message);
    });
    return () => { live = false; };
  }, [onError]);

  const revise = (patch: Partial<types.CDSPlanRequest>) => {
    setRequest((current) => current ? { ...current, ...patch } : current);
    setPlan(null); setError('');
  };

  const inspect = async () => {
    if (!request) return;
    setBusy(true); setPlan(null); setError('');
    try { setPlan(await apiService.planCDS(request)); }
    catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message); onError?.(message);
    } finally { setBusy(false); }
  };

  if (!capabilities || !request) return <div className="mt-4 rounded-xl border border-slate-800
    bg-slate-950/40 p-4 text-xs text-slate-400">
    {error || <span className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" />
      Loading CDS planning contract…</span>}
  </div>;

  return <section className="mt-4 rounded-xl border border-teal-500/25 bg-teal-500/[0.035] p-4"
    aria-labelledby="cds-planner-heading">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h3 id="cds-planner-heading" className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <Calculator className="h-4 w-4 text-teal-400" /> CDS regional request planner
        </h3>
        <p className="mt-1 max-w-4xl text-xs leading-relaxed text-slate-400">
          Validate the exact ERA5 pressure-level request before any queued transfer. The server
          computes its immutable identity, monthly work units, frame geometry and conservative
          storage ceiling.
        </p>
      </div>
      <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/25
        bg-emerald-500/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
        <ShieldCheck className="h-3.5 w-3.5" /> Planning uses no network
      </span>
    </div>

    <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[1.15fr_0.85fr]">
      <div className="rounded-lg border border-slate-800 bg-slate-950/45 p-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <span className="block text-xs text-slate-400">Variables</span>
            <div className="mt-2 flex flex-wrap gap-2">
              {capabilities.variables.map((variable) => <label key={variable.id}
                className="flex cursor-pointer items-center gap-2 rounded-md border border-slate-800
                  bg-slate-900 px-2.5 py-2 text-xs text-slate-300">
                <input type="checkbox" className="accent-teal-500"
                  checked={request.variables.includes(variable.id)}
                  onChange={(event) => revise({ variables: event.target.checked
                    ? [...request.variables, variable.id]
                    : request.variables.filter((name) => name !== variable.id) })} />
                <span className="font-mono text-teal-300">{variable.id}</span>
                <span className="text-slate-500">{variable.cds_name.split('_').join(' ')}</span>
              </label>)}
            </div>
          </div>
          <label className="text-xs text-slate-400">Start date
            <input type="date" value={request.date_start}
              onChange={(event) => revise({ date_start: event.target.value })}
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 text-slate-200" />
          </label>
          <label className="text-xs text-slate-400">End date
            <input type="date" value={request.date_end}
              onChange={(event) => revise({ date_end: event.target.value })}
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 text-slate-200" />
          </label>
          <label className="text-xs text-slate-400 sm:col-span-2">UTC hours (comma separated)
            <input value={request.hours_utc.join(', ')}
              onChange={(event) => revise({ hours_utc: numberList(event.target.value) })}
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 font-mono text-slate-200" />
          </label>
          {([['lat_min', 'Latitude south'], ['lat_max', 'Latitude north'],
            ['lon_min', 'Longitude west'], ['lon_max', 'Longitude east']] as const)
            .map(([key, label]) => <label key={key} className="text-xs text-slate-400">{label}
              <input type="number" step={request.grid_degrees} value={request[key]}
                onChange={(event) => revise({ [key]: Number(event.target.value) })}
                className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 font-mono text-slate-200" />
            </label>)}
          <label className="text-xs text-slate-400">Pressure levels (hPa)
            <input value={request.pressure_levels.join(', ')}
              onChange={(event) => revise({ pressure_levels: numberList(event.target.value) })}
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 font-mono text-slate-200" />
          </label>
          <label className="text-xs text-slate-400">Grid spacing (degrees)
            <input type="number" min="0.01" step="0.01" value={request.grid_degrees}
              onChange={(event) => revise({ grid_degrees: Number(event.target.value) })}
              className="mt-1 w-full rounded border border-slate-800 bg-slate-950 p-2 font-mono text-slate-200" />
          </label>
          <label className="text-xs text-slate-400 sm:col-span-2">Analysis depth: {request.n_levels_analysis}
            <input type="range" min="1" max="6" value={request.n_levels_analysis}
              onChange={(event) => revise({ n_levels_analysis: Number(event.target.value) })}
              className="mt-1 w-full accent-teal-500" />
          </label>
        </div>
        <button type="button" onClick={() => void inspect()} disabled={busy}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded bg-teal-600 p-2.5
            text-xs font-semibold text-white hover:bg-teal-500 disabled:opacity-50">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Calculator className="h-4 w-4" />}
          Validate plan — no network
        </button>
        {error && <p role="alert" className="mt-3 rounded border border-rose-500/30 bg-rose-500/5
          p-2 text-xs leading-relaxed text-rose-300">{error}</p>}
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-950/45 p-4">
        {!plan ? <div className="flex h-full min-h-56 flex-col items-center justify-center text-center">
          <Calculator className="h-8 w-8 text-slate-700" />
          <p className="mt-2 text-sm text-slate-400">No request validated yet</p>
          <p className="mt-1 max-w-sm text-xs text-slate-600">Planning reads configuration only.
            It does not submit a CDS request, reserve disk, or acquire a value.</p>
        </div> : <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
            <CheckCircle className="h-4 w-4" /> Request is structurally valid
          </div>
          <p className={plan.network_used ? 'text-xs text-rose-300' : 'text-xs text-emerald-300'}>
            {plan.network_used ? 'Unexpected network activity reported' : 'Plan verified: no network used'}
          </p>
          <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4">
            {[['Frames', plan.storage_estimate.frames.toLocaleString()],
              ['Monthly shards', plan.storage_estimate.shards],
              ['Grid', `${plan.storage_estimate.latitude_points_upper_bound}×${plan.storage_estimate.longitude_points_upper_bound}`],
              ['Storage ceiling', bytes(plan.storage_estimate.artifact_bytes_upper_bound)]]
              .map(([label, value]) => <div key={label} className="rounded bg-slate-900 p-2">
                <span className="block text-[10px] uppercase tracking-wide text-slate-500">{label}</span>
                <strong className="mt-1 block text-sm text-slate-200">{value}</strong>
              </div>)}
          </div>
          <div className="rounded border border-slate-800 p-3 text-xs">
            <p className="text-slate-400">Analysis geometry</p>
            <p className={plan.analysis_geometry.status.startsWith('MEETS')
              ? 'mt-1 text-emerald-300' : 'mt-1 text-amber-300'}>
              {plan.analysis_geometry.status.split('_').join(' ')}
            </p>
            {!plan.analysis_geometry.status.startsWith('MEETS') &&
              <p className="mt-1 text-slate-500">The acquisition plan is valid, but this generic
                analysis-depth heuristic refuses the selected spatial extent.</p>}
          </div>
          <details className="rounded border border-slate-800 p-3 text-xs text-slate-400">
            <summary className="cursor-pointer font-medium text-slate-300">
              Monthly work units ({plan.monthly_shards.length})
            </summary>
            <div className="mt-2 max-h-40 overflow-auto font-mono text-[10px]">
              {plan.monthly_shards.map((shard) => <p key={shard.request_sha256}>
                {shard.year}-{String(shard.month).padStart(2, '0')} · {shard.days.length} days · {shard.filename}
              </p>)}
            </div>
          </details>
          <div className="text-[10px] leading-relaxed text-slate-500">
            <p className="break-all font-mono">request sha256 {plan.request_sha256}</p>
            <p className="mt-2">{plan.storage_estimate.basis}; no compression credit assumed.</p>
            <p className="mt-2 text-amber-300">{plan.next_action}</p>
            <p className="mt-2">{plan.claim_boundary}</p>
          </div>
        </div>}
      </div>
    </div>
  </section>;
}
