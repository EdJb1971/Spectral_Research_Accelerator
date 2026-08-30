import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, FileLock2, Loader2, Save, Search } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

const SAVED_DRAFT_KEY = 'spectralearth.g17.composer.draft';

function utcInput(value: string): string {
  return value.replace(/Z$/, '').slice(0, 16);
}

function utcValue(value: string): string {
  return new Date(value.endsWith('Z') ? value : `${value}:00Z`).toISOString();
}

export default function ExperimentComposer({ onSelectStudy }: { onSelectStudy?: (id: string) => void }) {
  const [manifest, setManifest] = useState<types.CrossDomainExperimentManifest | null>(null);
  const [identity, setIdentity] = useState<string>('');
  const [preflight, setPreflight] = useState<types.ExperimentPreflight | null>(null);
  const [contract, setContract] = useState<Record<string, any> | null>(null);
  const [recipes, setRecipes] = useState<{ recipe_id: string; title: string }[]>([]);
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    apiService.getExperimentComposerContract().then(setContract).catch(() => setContract(null));
    apiService.listExperimentRecipes().then((body) => setRecipes(body.recipes)).catch(() => setRecipes([]));
    const draft = localStorage.getItem(SAVED_DRAFT_KEY);
    const request = draft ? apiService.loadExperimentDraft(draft) : apiService.getFlagshipRecipe();
    request.then((body) => {
      setManifest(body.canonical_manifest);
      setIdentity(body.manifest_sha256);
    }).catch((error: Error) => {
      setMessage(`${draft ? 'Saved draft' : 'Flagship recipe'} could not be loaded: ${error.message}`);
      if (draft) localStorage.removeItem(SAVED_DRAFT_KEY);
    });
  }, []);

  const declaredMembers = useMemo(() => {
    if (!manifest) return 0;
    const n = manifest.observations.length;
    return (n * (n - 1) / 2) * manifest.family.channels.length * manifest.family.scales.length
      * manifest.windows.length * manifest.family.relationships.length;
  }, [manifest]);

  const updateWindow = (index: number, field: 'start_utc' | 'end_utc', value: string) => {
    if (!manifest) return;
    const windows = manifest.windows.map((row, rowIndex) => rowIndex === index
      ? { ...row, [field]: utcValue(value) } : row);
    setManifest({ ...manifest, windows });
    setIdentity(''); setPreflight(null);
  };

  const validate = async () => {
    if (!manifest) return;
    setBusy('validate'); setMessage('');
    try {
      const result = await apiService.validateExperimentManifest(manifest);
      setManifest(result.canonical_manifest); setIdentity(result.manifest_sha256);
      setMessage('Manifest is valid and has a stable content identity.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const save = async () => {
    if (!manifest) return;
    setBusy('save'); setMessage('');
    try {
      const result = await apiService.saveExperimentDraft(manifest.study_id, manifest);
      const authenticated = await apiService.loadExperimentManifest(result.manifest_sha256);
      localStorage.setItem(SAVED_DRAFT_KEY, result.draft_id);
      setManifest(authenticated.canonical_manifest); setIdentity(authenticated.manifest_sha256);
      onSelectStudy?.(manifest.study_id);
      setMessage('Saved as an immutable manifest revision. Refresh will reload this draft.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const inspect = async () => {
    if (!manifest) return;
    setBusy('preflight'); setMessage('');
    try {
      const result = await apiService.preflightExperimentManifest(manifest);
      setPreflight(result); setIdentity(result.manifest_sha256);
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  if (!manifest) return <div className="text-sm text-slate-400">Loading the saved experiment manifest…</div>;

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <FileLock2 className="w-5 h-5 text-teal-400" /> Experiment Composer
        </h2>
        <p className="text-sm text-slate-400 mt-1">One versioned manifest drives coverage planning now and the runner and receipt in later G17 slices.</p>
        <p className="text-xs text-slate-500 mt-1">{recipes.length} saved recipe available · {contract?.claim_boundary || 'Loading capability boundary…'}</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="font-semibold text-slate-100">1. Scientific question</h3>
          <div>
            <label htmlFor="composer-study" className="text-xs text-slate-400">Study identity</label>
            <input id="composer-study" value={manifest.study_id} onChange={(event) => {
              setManifest({ ...manifest, study_id: event.target.value }); setIdentity(''); setPreflight(null);
            }} className="mt-1 w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm" />
          </div>
          <fieldset>
            <legend className="text-xs text-slate-400 mb-2">Comparison mode</legend>
            {(['calendar_aligned', 'scale_shape_aligned'] as const).map((mode) => (
              <label key={mode} className="flex gap-2 text-sm text-slate-200 mb-2">
                <input type="radio" name="composer-mode" checked={manifest.mode === mode}
                  onChange={() => {
                    setManifest({ ...manifest, mode,
                      scale_normalization: mode === 'scale_shape_aligned'
                        ? (manifest.scale_normalization || { method: 'native_scale_ratio', reference: 'within_domain' }) : null });
                    setIdentity(''); setPreflight(null);
                  }} /> {mode === 'calendar_aligned' ? 'Calendar-aligned co-occurrence' : 'Scale/shape recurrence'}
              </label>
            ))}
          </fieldset>
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-200">
            {manifest.mode === 'calendar_aligned'
              ? 'Shared UTC support permits co-occurrence only. It does not establish precedence or causality.'
              : 'Normalized shape comparison does not establish simultaneity, precedence or comparable magnitude.'}
          </div>
          <div>
            <label htmlFor="coverage-policy" className="text-xs text-slate-400">Coverage rule</label>
            <select id="coverage-policy" value={manifest.coverage_policy.requirement}
              onChange={(event) => {
                const requirement = event.target.value as 'complete_required' | 'partial_permitted';
                setManifest({ ...manifest, coverage_policy: { requirement,
                  minimum_fraction: requirement === 'complete_required' ? 1 : 0.5 } });
                setIdentity(''); setPreflight(null);
              }} className="mt-1 w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm">
              <option value="complete_required">Complete required</option>
              <option value="partial_permitted">Declared partial permitted</option>
            </select>
          </div>
        </section>

        <section className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="font-semibold text-slate-100">2. Explicit UTC windows</h3>
          <p className="text-xs text-slate-500">Preset names are labels only; these exact boundaries enter one corrected family.</p>
          {manifest.windows.map((window, index) => (
            <div key={window.name} className="border border-slate-800 rounded-lg p-3">
              <div className="text-xs font-semibold text-teal-300 mb-2">{window.name.split('_').join(' ')}</div>
              <div className="grid grid-cols-2 gap-2">
                <label className="text-[11px] text-slate-400">Start
                  <input aria-label={`${window.name} start UTC`} type="datetime-local" value={utcInput(window.start_utc)}
                    onChange={(event) => updateWindow(index, 'start_utc', event.target.value)}
                    className="block mt-1 w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs" />
                </label>
                <label className="text-[11px] text-slate-400">End
                  <input aria-label={`${window.name} end UTC`} type="datetime-local" value={utcInput(window.end_utc)}
                    onChange={(event) => updateWindow(index, 'end_utc', event.target.value)}
                    className="block mt-1 w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs" />
                </label>
              </div>
            </div>
          ))}
        </section>

        <section className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="font-semibold text-slate-100">3. Declared quartet and family</h3>
          {manifest.observations.map((row) => (
            <div key={row.domain} className="border-b border-slate-800 pb-2">
              <div className="text-sm text-slate-200">{row.label}</div>
              <div className="text-xs text-slate-500">{row.measure} · {row.units} · {row.acquisition.source_id}</div>
            </div>
          ))}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Family members</span>{declaredMembers}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Null replications</span>{manifest.nulls[0]?.replications}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Correction</span>{manifest.correction}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Cap</span>{manifest.resource_caps.maximum_family_members}</div>
          </div>
        </section>
      </div>

      <section className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap gap-3 items-center">
          <button onClick={validate} disabled={!!busy} className="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-sm">Validate identity</button>
          <button onClick={save} disabled={!!busy} className="px-4 py-2 rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-50 text-sm flex gap-2 items-center"><Save className="w-4 h-4" /> Save draft</button>
          <button onClick={inspect} disabled={!!busy} className="px-4 py-2 rounded bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-sm flex gap-2 items-center">
            {busy === 'preflight' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />} Inspect metadata coverage
          </button>
          <button disabled title="TG17.6 delivers the authenticated resumable runner" className="px-4 py-2 rounded bg-slate-800 text-slate-500 text-sm cursor-not-allowed">Run experiment — not available yet</button>
        </div>
        {identity && <p className="font-mono text-[11px] text-slate-500 mt-3 break-all">manifest sha256 {identity}</p>}
        {message && <p role="status" className="text-sm text-slate-300 mt-3">{message}</p>}
      </section>

      {preflight && (
        <section className={`border rounded-xl p-5 ${preflight.status === 'REFUSED' ? 'border-amber-500/40 bg-amber-500/5' : 'border-emerald-500/40 bg-emerald-500/5'}`}>
          <h3 className="font-semibold flex gap-2 items-center">
            {preflight.status === 'REFUSED' ? <AlertTriangle className="w-5 h-5 text-amber-400" /> : <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
            Metadata preflight: {preflight.status}
          </h3>
          <p className="text-xs text-slate-400 mt-1">No network used and no measurement values opened.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 mt-4">
            {preflight.coverage.map((row) => (
              <div key={row.domain} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                <div className="flex justify-between text-sm"><span>{row.label || row.domain}</span><span className="text-xs text-slate-400">{row.status}</span></div>
                <p className="text-xs text-slate-500 mt-2">{row.support_kind} · {row.native_cadence_seconds ? `${row.native_cadence_seconds}s native cadence` : 'native irregular clock'}</p>
                <p className="text-xs text-slate-400 mt-2">{row.reason}</p>
              </div>
            ))}
          </div>
          {preflight.refusals.length > 0 && <ul className="mt-4 text-xs text-amber-200 list-disc pl-5">
            {preflight.refusals.map((row, index) => <li key={index}>{row.domain ? `${row.domain}: ` : ''}{row.reason}</li>)}
          </ul>}
          <p className="text-xs text-slate-500 mt-4">{preflight.claim_boundary}</p>
        </section>
      )}
    </div>
  );
}
