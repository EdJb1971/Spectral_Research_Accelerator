import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, Calculator, FileLock2, Loader2, Save, Search, ShieldCheck, Waves } from 'lucide-react';
import { apiService } from '../services/api';
import { AdapterControlPanel } from './AdapterControls';
import { AlignmentKernelPicker, CoverageTimeline } from './CoverageTimeline';
import { FamilyExpansionPanel, NullFamilyPicker } from './FamilyPlan';
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
  const [preview, setPreview] = useState<types.StructuralTrajectoryPreview | null>(null);
  const [contract, setContract] = useState<Record<string, any> | null>(null);
  const [recipes, setRecipes] = useState<{ recipe_id: string; title: string }[]>([]);
  const [adapters, setAdapters] = useState<types.DomainExperimentAdapterDescription[]>([]);
  const [conformance, setConformance] = useState<types.AdapterConformanceReport[]>([]);
  const [kernels, setKernels] = useState<types.AlignmentKernelDescription[]>([]);
  const [alignment, setAlignment] = useState<types.AlignmentReport | null>(null);
  const [nullFamilies, setNullFamilies] = useState<types.NullFamilyList | null>(null);
  const [expansion, setExpansion] = useState<types.FamilyExpansion | null>(null);
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    apiService.getExperimentComposerContract().then(setContract).catch(() => setContract(null));
    apiService.listExperimentRecipes().then((body) => setRecipes(body.recipes)).catch(() => setRecipes([]));
    apiService.listExperimentAdapters().then((body) => setAdapters(body.adapters)).catch(() => setAdapters([]));
    apiService.listAlignmentKernels().then((body) => setKernels(body.kernels)).catch(() => setKernels([]));
    apiService.listNullFamilies().then(setNullFamilies).catch(() => setNullFamilies(null));
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

  // TG17.5 removed the product that used to be computed here. It was the third copy of the
  // family formula in this repository and it ignored the arities, lags, representations and
  // motifs the manifest now declares - so the browser could show a smaller family than the
  // receipt, which is the one number a researcher must not be able to read two ways.
  const declaredFamily = useMemo(
    () => expansion || (preflight ? preflight.family : null), [expansion, preflight]);

  const adapterFor = (adapterId: string) => adapters.find((row) => row.adapter_id === adapterId);

  const updateAdapterParameters = (domain: string, parameters: Record<string, any>) => {
    if (!manifest) return;
    setManifest({ ...manifest, observations: manifest.observations.map((row) => row.domain === domain
      ? { ...row, adapter: { ...row.adapter, parameters } } : row) });
    setIdentity(''); setPreflight(null); setPreview(null); setConformance([]); setAlignment(null);
  };

  const checkConformance = async () => {
    if (!manifest) return;
    setBusy('conformance'); setMessage('');
    try {
      const reports = await Promise.all(manifest.observations.map((row) =>
        apiService.runAdapterConformance(row.adapter.adapter_id, row.adapter.parameters)));
      setConformance(reports);
      const failing = reports.filter((row) => !row.conformant).length;
      setMessage(failing
        ? `${failing} adapter(s) do not honour their own declaration.`
        : 'Every declared adapter contract was executed against its known-answer record.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const updateWindow = (index: number, field: 'start_utc' | 'end_utc', value: string) => {
    if (!manifest) return;
    const windows = manifest.windows.map((row, rowIndex) => rowIndex === index
      ? { ...row, [field]: utcValue(value) } : row);
    setManifest({ ...manifest, windows });
    setIdentity(''); setPreflight(null); setPreview(null);
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

  const updateAlignment = (policy: types.AlignmentPolicy) => {
    if (!manifest) return;
    setManifest({ ...manifest, alignment: policy });
    setIdentity(''); setPreflight(null); setAlignment(null);
  };

  const priceFamily = async () => {
    if (!manifest) return;
    setBusy('family'); setMessage('');
    try {
      const result = await apiService.experimentFamily(manifest);
      setExpansion(result);
      setMessage(result.correction.affordable
        ? 'The declared family is priced. Affordability is not evidence.'
        : 'This declaration cannot reject anything at the declared ensemble.');
    } catch (error: any) { setExpansion(null); setMessage(error.message); }
    finally { setBusy(''); }
  };

  const updateNull = (method: string, parameters: Record<string, any>) => {
    if (!manifest) return;
    const cleaned = Object.fromEntries(
      Object.entries(parameters).filter(([, value]) => value !== undefined && value !== ''));
    setManifest({ ...manifest, nulls: manifest.nulls.map((row, index) =>
      index === 0 ? { ...row, method, parameters: cleaned } : row) });
    setIdentity(''); setPreflight(null); setExpansion(null);
  };

  const showSharedSupport = async () => {
    if (!manifest) return;
    setBusy('alignment'); setMessage('');
    try {
      const result = await apiService.experimentAlignment(manifest);
      setAlignment(result); setIdentity(result.manifest_sha256);
      setMessage(result.status === 'REFUSED'
        ? 'These records do not share enough support for every declared pair.'
        : 'Shared support measured from the declared supports. Row indices were not compared.');
    } catch (error: any) { setAlignment(null); setMessage(error.message); }
    finally { setBusy(''); }
  };

  const previewRepresentation = async () => {
    if (!manifest) return;
    setBusy('representation'); setMessage('');
    try {
      const result = await apiService.previewStructuralTrajectories(manifest);
      setPreview(result); setIdentity(result.manifest_sha256);
      setMessage('Canonical contract reconstructed and checked on deterministic known-answer records.');
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
              setManifest({ ...manifest, study_id: event.target.value }); setIdentity(''); setPreflight(null); setPreview(null);
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
                    setIdentity(''); setPreflight(null); setPreview(null);
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
                setIdentity(''); setPreflight(null); setPreview(null);
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
          <h3 className="font-semibold text-slate-100">3. Declared domains and family</h3>
          <p className="text-xs text-slate-500">Every control below is rendered from its adapter's
            registered schema. This panel contains no per-domain form: a newly registered adapter
            appears here with its own controls without this view being edited.</p>
          {manifest.observations.map((row) => (
            <details key={row.domain} className="border border-slate-800 rounded-lg p-3">
              <summary className="cursor-pointer text-sm text-slate-200">{row.label}
                <span className="block text-xs text-slate-500">{row.measure} · {row.units} · {row.acquisition.source_id}</span>
              </summary>
              <div className="mt-3">
                <AdapterControlPanel adapter={adapterFor(row.adapter.adapter_id)}
                  parameters={row.adapter.parameters}
                  onChange={(next) => updateAdapterParameters(row.domain, next)} />
              </div>
            </details>
          ))}
          <div className="border-t border-slate-800 pt-3">
            <h4 className="text-sm text-slate-200 mb-1">Alignment</h4>
            <p className="text-xs text-slate-500 mb-2">Support is compared as
              <span className="font-mono"> [start, end)</span> intervals, never as row indices.
              Only the default kernel transforms nothing; anything else must be named here and
              admitted by every domain above.</p>
            <AlignmentKernelPicker kernels={kernels} policy={manifest.alignment}
              onChange={updateAlignment} />
            <p className="text-[11px] text-slate-500 mt-2">
              {contract?.mode_forbids?.[manifest.mode] || ''}
            </p>
          </div>
          <div className="border-t border-slate-800 pt-3">
            <h4 className="text-sm text-slate-200 mb-1">Null</h4>
            <p className="text-xs text-slate-500 mb-2">A null belongs to a comparison mode and
              to a domain. What each family preserves is declared, because a surrogate that
              destroys the autocorrelation, the cyclic phase or the gaps is one no domain here
              emits — and a p-value measured against it is optimistic in a way the result does
              not show.</p>
            <NullFamilyPicker list={nullFamilies} mode={manifest.mode}
              method={manifest.nulls[0]?.method || ''}
              parameters={manifest.nulls[0]?.parameters || {}} onChange={updateNull} />
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Family members</span>{declaredFamily ? declaredFamily.family_size.toLocaleString() : 'price it'}</div>
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
          <button onClick={previewRepresentation} disabled={!!busy} className="px-4 py-2 rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-sm flex gap-2 items-center">
            {busy === 'representation' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />} Inspect structural contract
          </button>
          <button onClick={priceFamily} disabled={!!busy} className="px-4 py-2 rounded bg-violet-700 hover:bg-violet-600 disabled:opacity-50 text-sm flex gap-2 items-center">
            {busy === 'family' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Calculator className="w-4 h-4" />} Price the declared family
          </button>
          <button onClick={showSharedSupport} disabled={!!busy} className="px-4 py-2 rounded bg-sky-700 hover:bg-sky-600 disabled:opacity-50 text-sm flex gap-2 items-center">
            {busy === 'alignment' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Waves className="w-4 h-4" />} Show shared support
          </button>
          <button onClick={checkConformance} disabled={!!busy} className="px-4 py-2 rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-sm flex gap-2 items-center">
            {busy === 'conformance' ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />} Check adapter conformance
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

      {declaredFamily && (
        <section className={`border rounded-xl p-5 ${declaredFamily.correction.affordable
          ? 'border-violet-500/40 bg-violet-500/5' : 'border-rose-500/40 bg-rose-500/5'}`}>
          <h3 className="font-semibold flex gap-2 items-center mb-3">
            <Calculator className="w-5 h-5 text-violet-400" /> Declared family
          </h3>
          <FamilyExpansionPanel expansion={declaredFamily} />
        </section>
      )}

      {alignment && (
        <section className={`border rounded-xl p-5 ${alignment.status === 'REFUSED' ? 'border-amber-500/40 bg-amber-500/5' : 'border-sky-500/40 bg-sky-500/5'}`}>
          <h3 className="font-semibold flex gap-2 items-center mb-3">
            <Waves className="w-5 h-5 text-sky-400" /> Shared support
          </h3>
          <CoverageTimeline report={alignment} />
        </section>
      )}

      {conformance.length > 0 && (
        <section className="border border-indigo-500/40 bg-indigo-500/5 rounded-xl p-5">
          <h3 className="font-semibold flex gap-2 items-center"><ShieldCheck className="w-5 h-5 text-indigo-400" /> Adapter conformance</h3>
          <p className="text-xs text-indigo-100/70 mt-1">Each declared invariance, refusal and null was executed against a deterministic known-answer record. A claim with no executable probe is shown as NOT_PROBED rather than as passing.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
            {conformance.map((report) => (
              <div key={report.adapter_id} className="bg-slate-950/70 border border-slate-800 rounded-lg p-4">
                <div className="flex justify-between gap-2 text-sm">
                  <span className="font-semibold text-slate-100">{report.domain}</span>
                  <span className={report.conformant ? 'text-emerald-300 text-xs' : 'text-amber-300 text-xs'}>
                    {report.conformant ? 'conformant' : 'NOT conformant'}
                  </span>
                </div>
                <ul className="mt-2 space-y-1">
                  {report.checks.map((check) => (
                    <li key={check.check} className="text-[11px] flex gap-2">
                      <span className={check.status === 'PASS' ? 'text-emerald-400'
                        : check.status === 'FAIL' ? 'text-red-400' : 'text-slate-500'}>{check.status}</span>
                      <span className="text-slate-400">{check.check}</span>
                    </li>
                  ))}
                </ul>
                <p className="text-[10px] text-slate-500 mt-2">{report.claim_boundary}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {preview && (
        <section className="border border-cyan-500/40 bg-cyan-500/5 rounded-xl p-5">
          <h3 className="font-semibold flex gap-2 items-center"><Activity className="w-5 h-5 text-cyan-400" /> Canonical StructuralTrajectory preview</h3>
          <p className="text-xs text-cyan-100/70 mt-1">Deterministic known-answer records, not acquired observations. All three enter <span className="font-mono">{preview.mining_interface}</span> without a domain branch.</p>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 mt-4">
            {preview.trajectories.map((row) => {
              const valid = row.valid_mask.filter(Boolean).length;
              return <div key={row.domain} className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 space-y-2">
                <div className="flex justify-between gap-2"><span className="font-semibold text-slate-100">{row.domain}</span><span className="text-[11px] text-cyan-300">{valid}/{row.valid_mask.length} valid</span></div>
                <p className="text-xs text-slate-400">{row.native_semantics} · {row.native_units}</p>
                <p className="text-xs text-slate-300">{row.channel} · {row.channel_units}</p>
                <div className="h-2 rounded bg-slate-800 overflow-hidden" title="Validity coverage on the unchanged native clock">
                  <div className="h-full bg-cyan-500" style={{ width: `${100 * valid / row.valid_mask.length}%` }} />
                </div>
                <dl className="text-[11px] text-slate-500 space-y-1">
                  <div><dt className="inline text-slate-400">Support </dt><dd className="inline">{row.support_start_seconds.length} native [start, end) intervals</dd></div>
                  <div><dt className="inline text-slate-400">Scale </dt><dd className="inline">1 structural = {row.structural_scales[0].native_value}s native</dd></div>
                  <div className="break-all"><dt className="inline text-slate-400">Adapter </dt><dd className="inline font-mono">{row.adapter.definition_sha256}</dd></div>
                  <div className="break-all"><dt className="inline text-slate-400">Native </dt><dd className="inline font-mono">{row.native_record.content_sha256} · retained</dd></div>
                </dl>
                {row.assumption_violations.length > 0 && <p className="text-[11px] text-amber-300">Limits: {row.assumption_violations.join(', ')}</p>}
              </div>;
            })}
          </div>
          <p className="text-xs text-slate-500 mt-4">{preview.claim_boundary}</p>
        </section>
      )}
    </div>
  );
}
