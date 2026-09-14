import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, Calculator, FileLock2, Loader2, Play, Save,
         Search, ShieldCheck, Waves } from 'lucide-react';
import { apiService } from '../services/api';
import { AdapterControlPanel } from './AdapterControls';
import { AlignmentKernelPicker, CoverageTimeline } from './CoverageTimeline';
import { ComparisonViews } from './ComparisonViews';
import { ConfirmButton, DomainMenuPanel, ManifestInspector, NextActionBanner, PathStepper,
         PreregistrationSummaryPanel, StageLadder, StepStatusBadge,
         WindowPresetPicker } from './ComposerPath';
import { FamilyExpansionPanel, NullFamilyPicker } from './FamilyPlan';
import { RunProgressPanel, RunWorkerSuitePicker } from './RunMonitor';
import { ExperimentReceiptPanel } from './ExperimentReceipt';
import * as types from '../types/api';

/**
 * TG17.7 One guided workbench over one manifest.
 *
 * The step this view shows, and the single action it offers next, both come from
 * `POST /experiment-composer/path/state`. This component holds no precondition of its own. That
 * is not tidiness: the order of operations *is* the scientific discipline - the family is priced
 * before acquisition because a family priced afterwards is priced knowing what the data looked
 * like - and a UI that let those happen in any order would not have made an error, it would have
 * made the error undetectable.
 *
 * The step a researcher is on is remembered across navigation and refresh, beside the saved
 * draft, because losing your place in a seven-step commitment is how a step gets skipped.
 */

const SAVED_DRAFT_KEY = 'spectralearth.g17.composer.draft';
const ACTIVE_STEP_KEY = 'spectralearth.g17.composer.step';

function utcInput(value: string): string {
  return value.replace(/Z$/, '').slice(0, 16);
}

function utcValue(value: string): string {
  return new Date(value.endsWith('Z') ? value : `${value}:00Z`).toISOString();
}

export default function ExperimentComposer({ onSelectStudy, onEvidenceHandoff, requestedStep }: {
  onSelectStudy?: (id: string) => void;
  onEvidenceHandoff?: (id: string) => void;
  requestedStep?: string;
}) {
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
  const [runContract, setRunContract] = useState<types.RunContract | null>(null);
  const [run, setRun] = useState<types.RunIdentity | null>(null);
  const [receipt, setReceipt] = useState<types.RunReceipt | null>(null);
  const [suite, setSuite] = useState('fixture_dry_run');
  const [pathState, setPathState] = useState<types.ComposerPathState | null>(null);
  const [pathError, setPathError] = useState('');
  const [domainMenu, setDomainMenu] = useState<types.ComposerDomainMenu | null>(null);
  const [presets, setPresets] = useState<types.ComposerWindowPresets | null>(null);
  const [prereg, setPrereg] = useState<types.ComposerPreregistrationSummary | null>(null);
  const [envelope, setEnvelope] = useState<types.ComposerManifestEnvelope | null>(null);
  const [activeStep, setActiveStep] = useState<string>(
    () => localStorage.getItem(ACTIVE_STEP_KEY) || 'question');
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');

  const selectStep = (stepId: string) => {
    setActiveStep(stepId);
    localStorage.setItem(ACTIVE_STEP_KEY, stepId);
  };

  // The global journey chooses only which served panel to inspect. It supplies no status and
  // cannot make the step actionable; the path response below remains the sole scientific judge.
  useEffect(() => {
    if (requestedStep) selectStep(requestedStep);
  }, [requestedStep]);

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

  // Where the plan stands is asked of the server after every change to the plan, because a
  // status this component computed would be a second opinion about a scientific precondition.
  // It is metadata-only, opens no archive and - deliberately - does not open a run.
  const refreshPath = useCallback((spec: types.CrossDomainExperimentManifest) => {
    apiService.composerPathState(spec)
      .then((state) => { setPathState(state); setPathError(''); })
      .catch((error: Error) => { setPathState(null); setPathError(error.message); });
  }, []);

  useEffect(() => { if (manifest) refreshPath(manifest); }, [manifest, refreshPath]);

  useEffect(() => {
    if (!manifest) return;
    apiService.composerDomainMenu(manifest.observations.map((row) => row.domain))
      .then(setDomainMenu).catch(() => setDomainMenu(null));
  }, [manifest?.observations]);

  useEffect(() => {
    if (!manifest || manifest.windows.length === 0) return;
    apiService.composerWindowPresets(manifest.windows[0].start_utc,
                                     manifest.windows[0].stride_seconds)
      .then(setPresets).catch(() => setPresets(null));
  }, [manifest?.windows?.[0]?.start_utc, manifest?.windows?.[0]?.stride_seconds]);

  // TG17.5 removed the product that used to be computed here. It was the third copy of the
  // family formula in this repository and it ignored the arities, lags, representations and
  // motifs the manifest now declares - so the browser could show a smaller family than the
  // receipt, which is the one number a researcher must not be able to read two ways.
  const declaredFamily = useMemo(
    () => expansion || (preflight ? preflight.family : null), [expansion, preflight]);

  const adapterFor = (adapterId: string) => adapters.find((row) => row.adapter_id === adapterId);

  const stepStatus = (stepId: string) =>
    pathState?.steps.find((row) => row.step_id === stepId) || null;

  const invalidate = (next: types.CrossDomainExperimentManifest) => {
    setManifest(next);
    setIdentity(''); setPreflight(null); setPreview(null); setConformance([]);
    setAlignment(null); setExpansion(null); setPrereg(null); setEnvelope(null);
  };

  const updateAdapterParameters = (domain: string, parameters: Record<string, any>) => {
    if (!manifest) return;
    invalidate({ ...manifest, observations: manifest.observations.map((row) => row.domain === domain
      ? { ...row, adapter: { ...row.adapter, parameters } } : row) });
  };

  // Adding a domain inserts the observation the menu supplied; the browser never invents one.
  // Dropping below two is refused here as well as by the manifest, so the reason is visible
  // before the request rather than as a validation error afterwards.
  const toggleDomain = (domain: string, selected: boolean) => {
    if (!manifest || !domainMenu) return;
    if (!selected) {
      if (manifest.observations.length <= domainMenu.minimum_domains) {
        setMessage(`A cross-domain comparison needs at least ${domainMenu.minimum_domains} domains. `
          + 'Removing this one would leave a study that is not the study you declared.');
        return;
      }
      invalidate({ ...manifest,
        observations: manifest.observations.filter((row) => row.domain !== domain) });
      return;
    }
    const option = domainMenu.domains.find((row) => row.domain === domain);
    if (!option?.observation) {
      setMessage(option?.unavailable_reason || 'This domain declares no observation to add.');
      return;
    }
    invalidate({ ...manifest, observations: [...manifest.observations, option.observation as any] });
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
    invalidate({ ...manifest, windows: manifest.windows.map((row, rowIndex) => rowIndex === index
      ? { ...row, [field]: utcValue(value) } : row) });
  };

  // A preset names a duration; what is applied, stored and hashed is the pair of explicit
  // instants the server resolved it to.
  const applyPreset = (preset: types.ComposerWindowPreset) => {
    if (!manifest) return;
    invalidate({ ...manifest, windows: manifest.windows.map((row) => row.name === preset.preset
      ? { ...row, start_utc: preset.start_utc, end_utc: preset.end_utc } : row) });
    setMessage(`${preset.preset.split('_').join(' ')} resolved to ${preset.start_utc} → ${preset.end_utc}. `
      + 'The preset is a label; those instants are what the manifest stores.');
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

  // Clone-to-edit. The saved revision is immutable, so this changes which draft pointer this
  // browser follows; it does not alter the revision that was cloned.
  const cloneToEdit = async () => {
    if (!manifest) return;
    setBusy('clone'); setMessage('');
    try {
      const cloneId = `${manifest.study_id}_v2`;
      const next = { ...manifest, study_id: cloneId };
      const result = await apiService.saveExperimentDraft(cloneId, next);
      localStorage.setItem(SAVED_DRAFT_KEY, result.draft_id);
      setManifest(result.canonical_manifest); setIdentity(result.manifest_sha256);
      setMessage(`Editing a clone saved as ${cloneId}. The revision it was cloned from is immutable `
        + 'and unchanged.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const discardDraft = () => {
    localStorage.removeItem(SAVED_DRAFT_KEY);
    setMessage('This browser no longer follows a saved draft. The saved revisions themselves are '
      + 'immutable and were not deleted.');
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
    invalidate({ ...manifest, alignment: policy });
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

  // A held-out confirmation partition is spent exactly once, so a plan edited in the browser
  // needs its own. Without this control the only composable plan is the flagship default, whose
  // partition the first run to open it consumes -- and the second scientist would be told to
  // "declare a new partition" with no way to declare one (D82).
  const updateHeldOutPartition = (value: string) => {
    if (!manifest) return;
    invalidate({ ...manifest, confirmation: { ...manifest.confirmation,
      held_out_partition: value.trim() || null } });
  };

  const updateNull = (method: string, parameters: Record<string, any>) => {
    if (!manifest) return;
    const cleaned = Object.fromEntries(
      Object.entries(parameters).filter(([, value]) => value !== undefined && value !== ''));
    invalidate({ ...manifest, nulls: manifest.nulls.map((row, index) =>
      index === 0 ? { ...row, method, parameters: cleaned } : row) });
  };

  const switchMode = (mode: 'calendar_aligned' | 'scale_shape_aligned') => {
    if (!manifest) return;
    const relationships = contract?.mode_relationships?.[mode] || [];
    const selectedAdapters = manifest.observations.map((row) => row.adapter.adapter_id);
    const family = nullFamilies?.families.find((row) => row.admissible
      && row.modes.includes(mode)
      && selectedAdapters.every((adapterId) => row.admitted_by.includes(adapterId)));
    if (!relationships.length || !family) {
      setMessage(`The registered contracts do not yet admit a complete ${mode} plan for every selected domain.`);
      return;
    }
    invalidate({ ...manifest, mode,
      scale_normalization: mode === 'scale_shape_aligned'
        ? (manifest.scale_normalization || { method: 'native_scale_ratio', reference: 'within_domain' }) : null,
      family: { ...manifest.family, relationships: [relationships[0]] },
      nulls: manifest.nulls.map((row, index) => index === 0
        ? { ...row, method: family.name, parameters: {} } : row),
      notes: { ...manifest.notes, claim_boundary: mode === 'calendar_aligned'
        ? 'co-occurrence only'
        : 'shape recurrence only; no simultaneity, precedence or causality' },
    });
  };

  const showPreregistration = async () => {
    if (!manifest) return;
    setBusy('prereg'); setMessage('');
    try {
      setPrereg(await apiService.composerPreregistrationSummary(manifest));
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const exportManifest = async () => {
    if (!manifest) return;
    setBusy('export'); setMessage('');
    try {
      const result = await apiService.composerExportManifest(manifest);
      setEnvelope(result);
      setMessage(`Envelope for manifest ${result.manifest_sha256} rendered below. It carries the plan `
        + 'and no data.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const importManifest = async (text: string) => {
    setBusy('import'); setMessage('');
    try {
      const result = await apiService.composerImportManifest(JSON.parse(text));
      invalidate(result.canonical_manifest);
      setIdentity(result.manifest_sha256);
      setMessage(`Imported manifest ${result.manifest_sha256}. Its body was checked against its own digest.`);
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  // Opening a run is posting the manifest. There is no "create": the identity is the content
  // address of the plan, so this resumes whatever run that plan already has, including after a
  // refresh, and the response says which of the two happened.
  const openRun = async () => {
    if (!manifest) return;
    setBusy('run'); setMessage('');
    try {
      const contract = runContract || await apiService.experimentRunContract();
      setRunContract(contract);
      const opened = await apiService.openExperimentRun(manifest);
      setRun(opened); setReceipt(opened.receipt); setIdentity(opened.manifest_sha256);
      setMessage(opened.resumed
        ? `Resumed run ${opened.run_id}. An identical manifest is the same run, not a second one.`
        : `Opened run ${opened.run_id} at the content address of this manifest.`);
      refreshPath(manifest);
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const refreshRun = async (next: types.RunReceipt) => {
    setReceipt(next);
    const progress = await apiService.experimentRunProgress(next.run_id);
    setRun((current) => current ? { ...current, progress, receipt: next } : current);
    if (manifest) refreshPath(manifest);
  };

  // Re-reads the run from disk rather than from this tab's memory. A run can advance in another
  // tab, in another process, or after this browser was closed, and the journal on the server is
  // the only thing that knows.
  const reloadRun = async () => {
    if (!run) return;
    setBusy('reload'); setMessage('');
    try {
      await refreshRun(await apiService.experimentRunReceipt(run.run_id));
      setMessage('Reloaded from the run journal on the server.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const executeRun = async () => {
    if (!run) return;
    setBusy('execute'); setMessage('');
    try {
      await refreshRun(await apiService.executeExperimentRun(run.run_id, suite));
      setMessage('The declared plan was executed. A completed run is an executed plan, not evidence.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  // Offered only for an operational failure. A refusal is answered with an editable copy, and the
  // server enforces that distinction whatever this button does.
  const retryRun = async () => {
    if (!run) return;
    setBusy('retry'); setMessage('');
    try {
      await refreshRun(await apiService.retryExperimentRun(run.run_id, suite));
      setMessage('Only the components that failed operationally were re-executed; the rest were replayed.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const cancelRun = async () => {
    if (!run) return;
    setBusy('cancel'); setMessage('');
    try {
      await refreshRun(await apiService.cancelExperimentRun(run.run_id, 'cancelled from the composer'));
      setMessage('Cancelled. A cancelled run is terminal and cannot be restarted.');
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
  };

  const copyRefusedRun = async () => {
    if (!run || !manifest) return;
    setBusy('copy'); setMessage('');
    try {
      const copy = await apiService.experimentRunEditableCopy(run.run_id, `${manifest.study_id}_copy`);
      setMessage(`Editable draft ${copy.draft_id} written. The refused run is unchanged: ${copy.note}`);
    } catch (error: any) { setMessage(error.message); }
    finally { setBusy(''); }
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

  const current = stepStatus(activeStep);
  const panel = (stepId: string, body: JSX.Element) => activeStep === stepId ? (
    <section role="tabpanel" id={`composer-panel-${stepId}`}
             aria-labelledby={`composer-tab-${stepId}`} aria-busy={!!busy}
             className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      {current && (
        <header className="border-b border-slate-800 pb-3">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="font-semibold text-slate-100">{current.ordinal}. {current.title}</h3>
            <StepStatusBadge status={current.status} />
          </div>
          <p className="text-sm text-slate-300 mt-2">{current.question}</p>
          <p className="text-xs text-slate-500 mt-1">Settles: {current.settles}</p>
          <p className="text-xs text-slate-400 mt-1">{current.reason}</p>
          <p className="text-[11px] text-amber-300/80 mt-1">{current.claim_boundary}</p>
        </header>
      )}
      {body}
    </section>
  ) : null;

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <FileLock2 className="w-5 h-5 text-teal-400" aria-hidden="true" /> New experiment
        </h2>
        <p className="text-sm text-slate-400 mt-1">One versioned manifest, walked in the order the
          server declares. Every step below is a commitment, and the next one is named for you.</p>
        <p className="text-xs text-slate-500 mt-1">{recipes.length} saved recipe available · {contract?.claim_boundary || 'Loading capability boundary…'}</p>
      </div>

      {pathError && (
        <div role="alert" className="rounded-xl border border-rose-500/40 bg-rose-500/5 p-4 text-sm text-rose-200">
          The guided path could not be read from the server, so this view does not know what may
          legitimately be done next: {pathError}
        </div>
      )}

      {pathState ? (
        <>
          <PathStepper steps={pathState.steps} active={activeStep} onSelect={selectStep} />
          <NextActionBanner state={pathState} onGo={selectStep} />
        </>
      ) : !pathError && (
        <p role="status" className="text-sm text-slate-400">Reading where this plan stands…</p>
      )}

      {panel('question', (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-4">
            <div>
              <label htmlFor="composer-study" className="text-xs text-slate-400">Study identity</label>
              <input id="composer-study" value={manifest.study_id}
                     onChange={(event) => invalidate({ ...manifest, study_id: event.target.value })}
                     className="mt-1 w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm
                                focus:outline-none focus:ring-2 focus:ring-teal-400" />
            </div>
            <fieldset>
              <legend className="text-xs text-slate-400 mb-2">Comparison mode</legend>
              {(['calendar_aligned', 'scale_shape_aligned'] as const).map((mode) => (
                <label key={mode} className="flex gap-2 text-sm text-slate-200 mb-2">
                  <input type="radio" name="composer-mode" checked={manifest.mode === mode}
                    className="focus:outline-none focus:ring-2 focus:ring-teal-400"
                    onChange={() => switchMode(mode)}
                  /> {mode === 'calendar_aligned' ? 'Calendar-aligned co-occurrence' : 'Scale/shape recurrence'}
                </label>
              ))}
            </fieldset>
            <div>
              <label htmlFor="coverage-policy" className="text-xs text-slate-400">Coverage rule</label>
              <select id="coverage-policy" value={manifest.coverage_policy.requirement}
                onChange={(event) => {
                  const requirement = event.target.value as 'complete_required' | 'partial_permitted';
                  invalidate({ ...manifest, coverage_policy: { requirement,
                    minimum_fraction: requirement === 'complete_required' ? 1 : 0.5 } });
                }} className="mt-1 w-full bg-slate-950 border border-slate-700 rounded px-3 py-2 text-sm
                              focus:outline-none focus:ring-2 focus:ring-teal-400">
                <option value="complete_required">Complete required</option>
                <option value="partial_permitted">Declared partial permitted</option>
              </select>
            </div>
          </div>
          <div className="space-y-3">
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-200">
              {manifest.mode === 'calendar_aligned'
                ? 'Shared UTC support permits co-occurrence only. It does not establish precedence or causality.'
                : 'Normalized shape comparison does not establish simultaneity, precedence or comparable magnitude.'}
            </div>
            <p className="text-[11px] text-slate-500">{contract?.mode_forbids?.[manifest.mode] || ''}</p>
            <div className="flex flex-wrap gap-2 pt-2">
              <button onClick={validate} disabled={!!busy}
                      className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs
                                 focus:outline-none focus:ring-2 focus:ring-teal-400">Validate identity</button>
              <button onClick={save} disabled={!!busy}
                      className="px-3 py-1.5 rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-50 text-xs
                                 flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
                <Save className="w-3.5 h-3.5" aria-hidden="true" /> Save draft
              </button>
              <button onClick={cloneToEdit} disabled={!!busy}
                      className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs
                                 focus:outline-none focus:ring-2 focus:ring-teal-400">Clone to edit</button>
              <ConfirmButton label="Discard this browser's draft pointer"
                             confirmLabel="This browser will stop following the saved draft. Saved revisions are immutable and stay."
                             onConfirm={discardDraft} disabled={!!busy} />
            </div>
          </div>
        </div>
      ))}

      {panel('domains', (
        <DomainMenuPanel menu={domainMenu} onToggle={toggleDomain} busy={!!busy}
                         selectedDomains={manifest.observations.map((row) => row.domain)} />
      ))}

      {panel('observation', (
        <div className="space-y-5">
          <div>
            <h4 className="text-sm text-slate-200 mb-2">Explicit UTC windows</h4>
            <p className="text-xs text-slate-500 mb-3">Preset names are labels only; these exact
              boundaries enter one corrected family.</p>
            <WindowPresetPicker presets={presets} onApply={applyPreset} busy={!!busy} />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
              {manifest.windows.map((window, index) => (
                <div key={window.name} className="border border-slate-800 rounded-lg p-3">
                  <div className="text-xs font-semibold text-teal-300 mb-2">{window.name.split('_').join(' ')}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="text-[11px] text-slate-400">Start
                      <input aria-label={`${window.name} start UTC`} type="datetime-local" value={utcInput(window.start_utc)}
                        onChange={(event) => updateWindow(index, 'start_utc', event.target.value)}
                        className="block mt-1 w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs
                                   focus:outline-none focus:ring-2 focus:ring-teal-400" />
                    </label>
                    <label className="text-[11px] text-slate-400">End
                      <input aria-label={`${window.name} end UTC`} type="datetime-local" value={utcInput(window.end_utc)}
                        onChange={(event) => updateWindow(index, 'end_utc', event.target.value)}
                        className="block mt-1 w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs
                                   focus:outline-none focus:ring-2 focus:ring-teal-400" />
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className="text-sm text-slate-200 mb-1">Domain-native measures and roles</h4>
            <p className="text-xs text-slate-500 mb-3">Every control below is rendered from its
              adapter's registered schema. This panel contains no per-domain form: a newly
              registered adapter appears here with its own controls without this view being edited.</p>
            {/* A `<details>` is exposed as a group whose accessible name is not computed from
                its `<summary>`, so each panel names itself explicitly (the TG17.8 a11y defect). */}
            {manifest.observations.map((row) => (
              <details key={row.domain} aria-label={row.label}
                       className="border border-slate-800 rounded-lg p-3 mb-2">
                <summary className="cursor-pointer text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-400">
                  {row.label}
                  <span className="block text-xs text-slate-500">{row.measure} · {row.units} · {row.acquisition.source_id}</span>
                </summary>
                <div className="mt-3">
                  <AdapterControlPanel adapter={adapterFor(row.adapter.adapter_id)}
                    parameters={row.adapter.parameters}
                    onChange={(next) => updateAdapterParameters(row.domain, next)} />
                </div>
              </details>
            ))}
          </div>
        </div>
      ))}

      {panel('preflight', (
        <div className="space-y-4">
          <button onClick={inspect} disabled={!!busy}
                  className="px-4 py-2 rounded bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-sm
                             flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
            {busy === 'preflight'
              ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
              : <Search className="w-4 h-4" aria-hidden="true" />} Inspect metadata coverage
          </button>
          {!preflight && <p className="text-sm text-slate-400">No preflight has been run against
            this revision of the manifest. An empty panel is an unasked question, not a clean
            result.</p>}
          {preflight && (
            <div className={`border rounded-xl p-5 ${preflight.status === 'REFUSED' ? 'border-amber-500/40 bg-amber-500/5' : 'border-emerald-500/40 bg-emerald-500/5'}`}>
              <h4 className="font-semibold flex gap-2 items-center">
                {preflight.status === 'REFUSED'
                  ? <AlertTriangle className="w-5 h-5 text-amber-400" aria-hidden="true" />
                  : <CheckCircle2 className="w-5 h-5 text-emerald-400" aria-hidden="true" />}
                Metadata preflight: {preflight.status}
              </h4>
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
            </div>
          )}
        </div>
      ))}

      {panel('analysis', (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div>
              <h4 className="text-sm text-slate-200 mb-1">Alignment</h4>
              <p className="text-xs text-slate-500 mb-2">Support is compared as
                <span className="font-mono"> [start, end)</span> intervals, never as row indices.
                Only the default kernel transforms nothing; anything else must be named here and
                admitted by every domain above.</p>
              <AlignmentKernelPicker kernels={kernels} policy={manifest.alignment}
                onChange={updateAlignment} />
            </div>
            <div>
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
          </div>
          <div>
            <label htmlFor="composer-heldout" className="text-sm text-slate-200">
              Held-out confirmation partition
            </label>
            <p className="text-xs text-slate-500 mt-1 mb-2">Opened once, by one run. A plan that
              edits the frozen default must name its own partition here; reusing a spent one is
              refused rather than quietly re-confirmed.</p>
            <input id="composer-heldout" type="text" className="w-full max-w-md bg-slate-950 border border-slate-700
                   rounded px-2 py-1.5 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-teal-400"
              value={manifest.confirmation.held_out_partition || ''}
              onChange={(event) => updateHeldOutPartition(event.target.value)} />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Family members</span>{declaredFamily ? declaredFamily.family_size.toLocaleString() : 'price it'}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Null replications</span>{manifest.nulls[0]?.replications}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Correction</span>{manifest.correction}</div>
            <div className="bg-slate-950 rounded p-3"><span className="text-slate-500 block">Cap</span>{manifest.resource_caps.maximum_family_members}</div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={priceFamily} disabled={!!busy}
                    className="px-3 py-1.5 rounded bg-violet-700 hover:bg-violet-600 disabled:opacity-50 text-xs
                               flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
              {busy === 'family' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                                 : <Calculator className="w-3.5 h-3.5" aria-hidden="true" />} Price the declared family
            </button>
            <button onClick={showSharedSupport} disabled={!!busy}
                    className="px-3 py-1.5 rounded bg-sky-700 hover:bg-sky-600 disabled:opacity-50 text-xs
                               flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
              {busy === 'alignment' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                                    : <Waves className="w-3.5 h-3.5" aria-hidden="true" />} Show shared support
            </button>
            <button onClick={checkConformance} disabled={!!busy}
                    className="px-3 py-1.5 rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-xs
                               flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
              {busy === 'conformance' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                                      : <ShieldCheck className="w-3.5 h-3.5" aria-hidden="true" />} Check adapter conformance
            </button>
            <button onClick={previewRepresentation} disabled={!!busy}
                    className="px-3 py-1.5 rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-xs
                               flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
              {busy === 'representation' ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                                         : <Activity className="w-3.5 h-3.5" aria-hidden="true" />} Inspect structural contract
            </button>
          </div>
          {!declaredFamily && <p className="text-sm text-slate-400">The declared family has not been
            priced against this revision. An unpriced family is an unknown cost, not a small one.</p>}
        </div>
      ))}

      {panel('freeze_and_run', (
        <div className="space-y-4">
          <button onClick={showPreregistration} disabled={!!busy}
                  className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs
                             focus:outline-none focus:ring-2 focus:ring-teal-400">
            Render the preregistration summary
          </button>
          <PreregistrationSummaryPanel summary={prereg} />
          <button onClick={openRun} disabled={!!busy}
                  className="px-4 py-2 rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-50 text-sm
                             flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
            {busy === 'run' ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                            : <Play className="w-4 h-4" aria-hidden="true" />} Open or resume the run
          </button>
          {run && receipt && runContract && (
            <div className="border border-blue-500/40 bg-blue-500/5 rounded-xl p-5">
              <h4 className="font-semibold flex gap-2 items-center mb-1">
                <Play className="w-5 h-5 text-blue-400" aria-hidden="true" /> Orchestrated run
              </h4>
              <p className="text-xs text-blue-100/70 mb-4">
                The run identity is the manifest, so refreshing this page resumes the same run rather
                than starting a second one. Completed steps are replayed from their content addresses
                and are never requested again.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 items-end">
                <RunWorkerSuitePicker suites={runContract.worker_suites} selected={suite}
                                      onSelect={setSuite} disabled={!!busy} />
                <div className="flex gap-2">
                  <button onClick={executeRun} disabled={!!busy}
                          className="px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-50 text-xs
                                     focus:outline-none focus:ring-2 focus:ring-teal-400">
                    Execute the frozen plan
                  </button>
                  <button onClick={reloadRun} disabled={!!busy}
                          className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs
                                     focus:outline-none focus:ring-2 focus:ring-teal-400">
                    Reload from the journal
                  </button>
                </div>
                <p className="text-[11px] text-slate-500">
                  Not yet available: {runContract.not_yet_available.join(', ')}.
                </p>
              </div>
              <RunProgressPanel progress={run.progress} receipt={receipt}
                                machine={runContract.state_machine} busy={!!busy}
                                onRetry={retryRun} onCancel={cancelRun} onEditableCopy={copyRefusedRun} />
            </div>
          )}
          {!run && <p className="text-sm text-slate-400">No run has been opened at this manifest's
            content address. Opening one is resumable and can never produce a second run for the
            same plan.</p>}
        </div>
      ))}

      {panel('interpret', (
        <div className="space-y-4">
          {pathState && <StageLadder ladder={pathState.ladder} />}
          {receipt ? (
            <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4 space-y-2">
              <p className="text-sm text-slate-200">Run {receipt.run_id} is {receipt.state} with{' '}
                {Object.keys(receipt.artefacts).length} completed artefact(s).</p>
              {receipt.missing_components.length > 0 && (
                <ul className="text-xs text-amber-200 list-disc pl-5">
                  {receipt.missing_components.map((row, index) => (
                    <li key={index}>{row.stage}/{row.component}: {row.status} — {row.remediation || row.detail}</li>
                  ))}
                </ul>
              )}
              <p className="text-[11px] text-slate-500">{receipt.claim_boundary}</p>
            </div>
          ) : (
            <p className="text-sm text-slate-400">There is no receipt to interpret yet. An empty
              interpretation panel is an experiment that has not run, not a null result.</p>
          )}
          {/* TG17.8. The views are shown whether or not a run exists, because most of what they
              make inspectable - coverage, native support, the scale mapping, the correction
              denominator - is a fact about the declaration, and it is worth reading *before*
              committing to the plan rather than after. Each view says which of its cells has
              not been measured and why. */}
          <div className="pt-2 border-t border-slate-800 space-y-3">
            <h4 className="text-sm font-semibold text-slate-100">Comparison views</h4>
            <ComparisonViews manifest={manifest} />
          </div>
          <ExperimentReceiptPanel runId={receipt?.run_id} runState={receipt?.state}
                                  onEvidenceHandoff={onEvidenceHandoff} />
        </div>
      ))}

      <section className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        {identity && <p className="font-mono text-[11px] text-slate-500 break-all">manifest sha256 {identity}</p>}
        {message && <p role="status" className="text-sm text-slate-300">{message}</p>}
        <ManifestInspector manifest={manifest} envelope={envelope}
                           onExport={exportManifest} onImport={importManifest} busy={!!busy} />
      </section>

      {declaredFamily && (
        <section className={`border rounded-xl p-5 ${declaredFamily.correction.affordable
          ? 'border-violet-500/40 bg-violet-500/5' : 'border-rose-500/40 bg-rose-500/5'}`}>
          <h3 className="font-semibold flex gap-2 items-center mb-3">
            <Calculator className="w-5 h-5 text-violet-400" aria-hidden="true" /> Declared family
          </h3>
          <FamilyExpansionPanel expansion={declaredFamily} />
        </section>
      )}

      {alignment && (
        <section className={`border rounded-xl p-5 ${alignment.status === 'REFUSED' ? 'border-amber-500/40 bg-amber-500/5' : 'border-sky-500/40 bg-sky-500/5'}`}>
          <h3 className="font-semibold flex gap-2 items-center mb-3">
            <Waves className="w-5 h-5 text-sky-400" aria-hidden="true" /> Shared support
          </h3>
          <CoverageTimeline report={alignment} />
        </section>
      )}

      {conformance.length > 0 && (
        <section className="border border-indigo-500/40 bg-indigo-500/5 rounded-xl p-5">
          <h3 className="font-semibold flex gap-2 items-center"><ShieldCheck className="w-5 h-5 text-indigo-400" aria-hidden="true" /> Adapter conformance</h3>
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
          <h3 className="font-semibold flex gap-2 items-center"><Activity className="w-5 h-5 text-cyan-400" aria-hidden="true" /> Canonical StructuralTrajectory preview</h3>
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
