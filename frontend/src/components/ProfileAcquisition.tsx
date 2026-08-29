/** TG12.2: bounded irregular-profile acquisition and declared reduction. */
import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, Database, Loader2, Search } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  source: types.ProfileSource;
  onError?: (message: string) => void;
  onCapability?: (profile: types.DatasetCapabilityProfile | null) => void;
}

const fallback: types.ProfileSpecRequest = {
  source: 'argo_gdac_erddap', time_start: '2026-01-01', time_end: '2026-03-01',
  lat_min: -46, lat_max: -34, lon_min: 165, lon_max: 180,
  pressure_min_dbar: 0, pressure_max_dbar: 200,
  variables: ['temperature'], max_profiles: 200,
};

export const ProfileAcquisition: React.FC<Props> = ({ source, onError, onCapability }) => {
  const [capabilities, setCapabilities] = useState<types.ProfileCapabilities | null>(null);
  const [spec, setSpec] = useState<types.ProfileSpecRequest>({
    ...fallback, ...source.defaults, source: source.name,
  });
  const [reductionName, setReductionName] = useState<types.ProfileReductionRequest['name']>(
    'per_float_at_pressure');
  const [variable, setVariable] = useState('temperature');
  const [pressure, setPressure] = useState(100);
  const [maxGap, setMaxGap] = useState(25);
  const [binEdges, setBinEdges] = useState('0,50,100,200');
  const [plan, setPlan] = useState<types.ProfileQueryPlan | null>(null);
  const [result, setResult] = useState<types.ProfileAcquisitionResponse | null>(null);
  const [busy, setBusy] = useState<'inspect' | 'acquire' | null>(null);

  useEffect(() => {
    let live = true;
    apiService.profileCapabilities().then((value) => { if (live) setCapabilities(value); })
      .catch((error) => onError?.(error instanceof Error ? error.message : String(error)));
    return () => { live = false; };
  }, [onError]);

  const reduction = useMemo<types.ProfileReductionRequest>(() => ({
    name: reductionName,
    configuration: reductionName === 'per_float_at_pressure'
      ? { variable, pressure_dbar: pressure, max_interpolation_gap_dbar: maxGap }
      : { variable, bin_edges_dbar: binEdges.split(',').map(Number).filter(Number.isFinite) },
  }), [reductionName, variable, pressure, maxGap, binEdges]);

  const payload = useMemo(() => ({ spec, reduction }), [spec, reduction]);

  const revise = (change: Partial<types.ProfileSpecRequest>) => {
    setSpec((current) => ({ ...current, ...change }));
    setPlan(null); setResult(null); onCapability?.(null);
  };
  const reviseReduction = (action: () => void) => {
    action(); setPlan(null); setResult(null); onCapability?.(null);
  };
  const fail = (error: unknown) => onError?.(error instanceof Error ? error.message : String(error));

  const inspect = async () => {
    setBusy('inspect'); setResult(null);
    try { setPlan(await apiService.inspectProfiles(payload)); }
    catch (error) { setPlan(null); fail(error); }
    finally { setBusy(null); }
  };
  const acquire = async () => {
    setBusy('acquire');
    try { const acquired = await apiService.acquireProfiles(payload); setResult(acquired);
      onCapability?.(acquired.capability_profile); }
    catch (error) { setResult(null); fail(error); }
    finally { setBusy(null); }
  };

  return <div className="space-y-5">
    {capabilities && !capabilities.network_enabled && <div role="status"
      className="border border-amber-900/60 bg-amber-950/20 rounded-lg p-3 text-xs text-amber-200">
      Live profiles are disabled server-side. Set <code>{capabilities.network_env_var}=1</code>{' '}
      before starting the backend. A browser click never enables archive access.
    </div>}

    <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
        <header>
          <h3 className="text-sm font-semibold text-slate-100">3. Profile query</h3>
          <p className="text-[11px] text-slate-500 mt-1">{source.description}</p>
          <a href={capabilities?.source_doi} target="_blank" rel="noreferrer"
            className="text-[10px] text-teal-400 underline">Argo GDAC DOI and citation identity</a>
        </header>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs text-slate-400">Start
            <input type="date" value={spec.time_start.slice(0, 10)}
              onChange={(e) => revise({ time_start: e.target.value })}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
          </label>
          <label className="text-xs text-slate-400">End
            <input type="date" value={spec.time_end.slice(0, 10)}
              onChange={(e) => revise({ time_end: e.target.value })}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
          </label>
          {([['lat_min', 'Lat min'], ['lat_max', 'Lat max'], ['lon_min', 'Lon min'],
             ['lon_max', 'Lon max']] as const).map(([key, label]) =>
            <label key={key} className="text-xs text-slate-400">{label}
              <input type="number" value={spec[key]}
                onChange={(e) => revise({ [key]: Number(e.target.value) })}
                className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2 font-mono" />
            </label>)}
          <label className="text-xs text-slate-400">Pressure min (dbar)
            <input type="number" value={spec.pressure_min_dbar}
              onChange={(e) => revise({ pressure_min_dbar: Number(e.target.value) })}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
          </label>
          <label className="text-xs text-slate-400">Pressure max (dbar)
            <input type="number" value={spec.pressure_max_dbar}
              onChange={(e) => revise({ pressure_max_dbar: Number(e.target.value) })}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
          </label>
        </div>
        <label className="text-xs text-slate-400 block">Acquisition safety cap (profiles)
          <input type="number" min={2} max={500} value={spec.max_profiles}
            onChange={(e) => revise({ max_profiles: Number(e.target.value) })}
            className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
          <span className="text-[10px] text-slate-600">Preflight refuses before values when exceeded.</span>
        </label>

        <div className="border-t border-slate-800 pt-4 space-y-3">
          <h4 className="text-xs font-semibold text-slate-200">4. Declared reduction</h4>
          <label className="text-xs text-slate-400 block">Reduction
            <select value={reductionName}
              onChange={(e) => reviseReduction(() => setReductionName(e.target.value as typeof reductionName))}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
              {(capabilities?.reductions ?? []).map((entry) =>
                <option key={entry.name} value={entry.name}>{entry.name}</option>)}
            </select>
          </label>
          <label className="text-xs text-slate-400 block">Measure
            <select value={variable} onChange={(e) => reviseReduction(() => setVariable(e.target.value))}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
              {source.variables.map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
          {reductionName === 'per_float_at_pressure' ? <div className="grid grid-cols-2 gap-2">
            <label className="text-xs text-slate-400">Target (dbar)
              <input type="number" value={pressure}
                onChange={(e) => reviseReduction(() => setPressure(Number(e.target.value)))}
                className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
            </label>
            <label className="text-xs text-slate-400">Maximum bracket (dbar)
              <input type="number" value={maxGap}
                onChange={(e) => reviseReduction(() => setMaxGap(Number(e.target.value)))}
                className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
            </label>
          </div> : <label className="text-xs text-slate-400 block">Bin edges (dbar)
            <input value={binEdges} onChange={(e) => reviseReduction(() => setBinEdges(e.target.value))}
              className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2 font-mono" />
          </label>}
          <p className="text-[10px] text-slate-500">
            Per-float preserves floats starting/stopping. Depth-bin means pool float identity
            and explicitly add <code>aggregated_values</code>. They are different scientific records.
          </p>
        </div>

        <button type="button" onClick={() => void inspect()} disabled={busy !== null}
          className="w-full bg-teal-700 hover:bg-teal-600 disabled:opacity-50 rounded p-2 text-xs font-semibold flex justify-center gap-2">
          {busy === 'inspect' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          Inspect metadata first
        </button>
      </section>

      <div className="xl:col-span-2 space-y-4">
        {plan ? <section className={`rounded-xl border p-5 ${plan.within_profile_cap
          ? 'border-emerald-700/40 bg-emerald-950/10' : 'border-rose-700/40 bg-rose-950/10'}`}>
          <div className="flex gap-2 items-start">
            {plan.within_profile_cap ? <CheckCircle2 className="text-emerald-400 shrink-0" />
              : <AlertTriangle className="text-rose-400 shrink-0" />}
            <div>
              <h4 className="text-sm font-semibold text-slate-100">
                {plan.candidate_profiles} profiles across {plan.candidate_platforms} floats
              </h4>
              <p className="text-xs text-slate-400 mt-1">cap {plan.max_profiles} · no profile values fetched</p>
            </div>
          </div>
          {!plan.within_profile_cap && <p className="text-xs text-rose-300 mt-3">
            Narrow the region or time span, or deliberately raise the cap. Acquisition is refused
            before the level response.</p>}
          <details className="mt-3 text-[10px] text-slate-500">
            <summary className="cursor-pointer">Candidate profile coordinates and request identity</summary>
            {plan.profiles.slice(0, 12).map((row) => <p key={row.profile_id} className="font-mono mt-1">
              {row.time} · {row.platform_id} · {row.latitude.toFixed(3)}, {row.longitude.toFixed(3)}
            </p>)}
            {plan.profiles_withheld > 0 && <p>{plan.profiles_withheld} additional candidates withheld from preview.</p>}
            <p className="font-mono break-all mt-2">request {plan.request_sha256}</p>
            <p>{plan.claim_boundary}</p>
          </details>
          <button type="button" onClick={() => void acquire()}
            disabled={busy !== null || !plan.within_profile_cap || plan.candidate_profiles < 1}
            className="mt-4 bg-teal-600 hover:bg-teal-500 disabled:opacity-50 rounded px-4 py-2 text-xs font-semibold flex gap-2">
            {busy === 'acquire' ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Database className="w-4 h-4" />} Acquire, QC and reduce
          </button>
        </section> : <section className="min-h-[220px] bg-slate-900 border border-slate-800 rounded-xl flex flex-col items-center justify-center text-slate-500">
          <Search className="w-9 h-9 mb-2" /><p className="text-sm">No profile query inspected yet</p>
          <p className="text-xs mt-1">The index preflight always precedes profile-level transfer.</p>
        </section>}

        {result && <>
          <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-slate-100">Immutable acquisition record</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3 text-center text-xs">
              <div className="bg-slate-950 rounded p-2"><span className="block text-slate-500">profiles</span>{result.collection.n_profiles}</div>
              <div className="bg-slate-950 rounded p-2"><span className="block text-slate-500">floats</span>{result.collection.n_platforms}</div>
              <div className="bg-slate-950 rounded p-2"><span className="block text-slate-500">frames</span>{result.reduction.n_frames}</div>
              <div className="bg-slate-950 rounded p-2"><span className="block text-slate-500">channels</span>{result.reduction.n_channels}</div>
            </div>
            <p className="text-[10px] font-mono text-slate-500 break-all mt-3">
              collection {result.collection.collection_sha256}</p>
            <p className="text-[10px] font-mono text-slate-500 break-all">
              reduction {String(result.reduction.reduction.reduction_sha256)}</p>
            <p className="text-[10px] text-slate-500 mt-2">
              QC: {String(result.collection.qc_policy.name)} · publication {result.publication.publication}</p>
          </section>
          <section role="status" className={`rounded-xl border p-5 ${result.analysis_readiness.frame_lag_admissible
            ? 'border-emerald-600/40 bg-emerald-950/10' : 'border-amber-600/50 bg-amber-950/20'}`}>
            <div className="flex gap-2 items-start">
              {result.analysis_readiness.frame_lag_admissible
                ? <CheckCircle2 className="text-emerald-400 shrink-0" />
                : <AlertTriangle className="text-amber-400 shrink-0" />}
              <div><h4 className="text-sm font-semibold text-slate-100">
                Analysis decision: {result.analysis_readiness.decision.split('_').join(' ')}</h4>
                <p className="text-xs text-slate-300 mt-2">{result.analysis_readiness.reason}</p>
                <p className="text-[10px] font-mono text-amber-300 mt-2">
                  {result.analysis_readiness.basis.join(' · ')}</p>
              </div>
            </div>
          </section>
          <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-slate-100">Reduction consequences</h4>
            <p className="text-xs text-slate-400 mt-2">
              Derived violations: {Object.keys((result.reduction.derived_declaration.violations ?? {}) as object).join(', ') || 'none'}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3">
              {result.reduction.channel_records.slice(0, 12).map((row) => <div key={String(row.scale)}
                className="border border-slate-800 rounded p-2 text-[10px]">
                <span className="text-slate-200">{String(row.scale)}</span>
                {row.present_count !== undefined && <span className="text-slate-500 block">
                  present {String(row.present_count)} · absent {String(row.absent_count)}</span>}
              </div>)}
            </div>
            <p className="text-[10px] text-slate-500 mt-3">{result.claim_boundary}</p>
          </section>
        </>}
      </div>
    </div>
  </div>;
};

export default ProfileAcquisition;
