/** TG16: progressive representation-structure workflow over one declared sample table. */
import { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ExternalLink, Loader2, Lock, Play } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

type Json = Record<string, any>;

const short = (value: unknown) => typeof value === 'string' ? `${value.slice(0, 16)}…` : '—';

const Result: React.FC<{ title: string; value: Json | null; rows?: any[] }> = ({ title, value, rows }) =>
  value ? <div className="border border-emerald-700/30 bg-emerald-950/10 rounded p-3 text-xs">
    <h4 className="font-semibold text-emerald-200 flex gap-2 items-center">
      <CheckCircle2 className="w-4 h-4" /> {title}
    </h4>
    {rows?.map((row, index) => <p key={String(row.label ?? row.candidate ?? index)}
      className="mt-1 text-slate-300">
      <span className="font-mono">{String(row.label ?? row.candidate ?? `member ${index + 1}`)}</span>
      {' · '}{String(row.outcome ?? (row.survives_both ? 'supported' : 'unresolved'))}
    </p>)}
    <p className="mt-2 text-[10px] text-slate-500">{String(value.claim_boundary ?? '')}</p>
  </div> : null;

interface Props {
  file: File;
  declaration: types.SampleTableDeclaration;
  capability: types.DatasetCapabilityProfile;
  handoff?: types.SampleTablePlanningHandoff | null;
  handoffPlan?: Record<string, any> | null;
  onError?: (message: string) => void;
}

export const RepresentationStructureProgramme: React.FC<Props> = ({
  file, declaration, capability, handoff = null, handoffPlan = null, onError,
}) => {
  const [busy, setBusy] = useState('');
  const [structurePlan, setStructurePlan] = useState<Json | null>(null);
  const [structure, setStructure] = useState<Json | null>(null);
  const [conditionalPlan, setConditionalPlan] = useState<Json | null>(null);
  const [conditional, setConditional] = useState<Json | null>(null);
  const [subspacePlan, setSubspacePlan] = useState<Json | null>(null);
  const [generation, setGeneration] = useState<Json | null>(null);
  const [confirmationSeal, setConfirmationSeal] = useState<Json | null>(null);
  const [confirmation, setConfirmation] = useState<Json | null>(null);
  const [published, setPublished] = useState<Json | null>(null);
  const [externalFile, setExternalFile] = useState<File | null>(null);
  const [acquisitionId, setAcquisitionId] = useState('');
  const [acquisitionSource, setAcquisitionSource] = useState('');
  const [acquiredAt, setAcquiredAt] = useState('');
  const [transferSeal, setTransferSeal] = useState<Json | null>(null);
  const [externalReceipt, setExternalReceipt] = useState<Json | null>(null);

  const fail = (error: unknown) => onError?.(error instanceof Error ? error.message : String(error));
  const run = async (name: string, action: () => Promise<void>) => {
    setBusy(name); try { await action(); } catch (error) { fail(error); } finally { setBusy(''); }
  };
  const hasNuisance = useMemo(() => Object.values(declaration.roles).includes('nuisance'),
    [declaration]);
  const replicated = confirmation?.internally_replicated_candidates ?? [];

  useEffect(() => {
    if (!handoff || !handoffPlan) return;
    if (handoff.operation.id === 'redundancy_structure_audit') {
      setStructurePlan(handoffPlan); setStructure(null);
    } else if (handoff.operation.id === 'conditional_information_audit') {
      setConditionalPlan(handoffPlan); setConditional(null);
    } else if (handoff.operation.id === 'stable_subspace_generation') {
      setSubspacePlan(handoffPlan); setGeneration(null); setConfirmationSeal(null);
      setConfirmation(null); setPublished(null);
    }
  }, [handoff, handoffPlan]);

  const runStructure = () => run('structure', async () => {
    const plan = structurePlan ?? await apiService.planRedundancyStructure(file, declaration);
    setStructurePlan(plan);
    setStructure(await apiService.runRedundancyStructure(file, plan));
  });
  const runConditional = () => run('conditional', async () => {
    const plan = conditionalPlan ?? await apiService.planConditionalInformation(file, declaration);
    setConditionalPlan(plan);
    setConditional(await apiService.runConditionalInformation(file, plan));
  });
  const planSubspace = () => run('plan', async () => {
    setSubspacePlan(await apiService.planStableSubspace(file, declaration));
    setGeneration(null); setConfirmationSeal(null); setConfirmation(null); setPublished(null);
  });
  const generate = () => run('generate', async () => {
    if (!subspacePlan) return;
    setGeneration(await apiService.generateStableSubspace(file, subspacePlan));
  });
  const freeze = () => run('freeze', async () => {
    if (!subspacePlan || !generation) return;
    setConfirmationSeal(await apiService.freezeStableSubspace(file, subspacePlan, generation));
  });
  const confirm = () => run('confirm', async () => {
    if (!confirmationSeal) return;
    setConfirmation(await apiService.confirmStableSubspace(file, confirmationSeal.seal_sha256));
  });
  const publish = () => run('publish', async () => {
    if (!confirmationSeal || replicated.length < 1) return;
    setPublished(await apiService.publishStableSubspace(
      confirmationSeal.seal_sha256, String(replicated[0].label)));
  });
  const freezeTransfer = () => run('transfer-freeze', async () => {
    if (!published || !externalFile) return;
    const probe = await apiService.probeGenericFile(externalFile);
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest(
      'SHA-256', await externalFile.arrayBuffer())))
      .map((value) => value.toString(16).padStart(2, '0')).join('');
    setTransferSeal(await apiService.freezeExternalSubspace(
      [published.candidate_sha256], digest, probe.n_rows, declaration,
      { acquisition_id: acquisitionId, acquired_at: acquiredAt,
        source: acquisitionSource, independent_of_origin: true }));
  });
  const certify = () => run('certify', async () => {
    if (!transferSeal || !externalFile) return;
    setExternalReceipt(await apiService.certifyExternalSubspace(
      externalFile, transferSeal.seal_sha256));
  });

  const available = (name: string) => capability.operations[name]?.available === true;
  const Button: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & { working: string }> =
    ({ working, children, ...props }) => <button {...props} disabled={busy !== '' || props.disabled}
      className="rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 px-3 py-2 text-xs font-semibold flex gap-2 items-center">
      {busy === working ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
      {children}
    </button>;

  return <section className="border border-indigo-500/30 bg-indigo-500/5 rounded-xl p-5 space-y-5">
    <header>
      <h3 className="text-base font-semibold text-indigo-100">Representation structure programme</h3>
      <p className="text-xs text-slate-400 mt-1">Use one declaration throughout. Each stage
        freezes its complete family before enumeration; no result removes a feature or chooses
        a representation for you.</p>
    </header>

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="border border-slate-800 rounded-lg p-4 space-y-3">
        <h4 className="text-sm font-semibold">1. Redundancy and complementarity</h4>
        <p className="text-[11px] text-slate-500">Map duplicate, complementary, and unresolved
          pair structure across every declared feature.</p>
        <Button working="structure" onClick={() => void runStructure()}
          disabled={!available('redundancy_structure_audit')}>
          {structurePlan ? 'Audit prepared plan' : 'Plan and audit'}</Button>
        {structurePlan && <p className="font-mono text-[10px] text-slate-500">
          plan {short(structurePlan.plan_sha256)}</p>}
        {!available('redundancy_structure_audit') && <p className="text-[10px] text-amber-400">
          {capability.operations.redundancy_structure_audit?.reason}</p>}
        <Result title="Candidate structure map" value={structure} rows={structure?.pairs} />
      </div>

      <div className="border border-slate-800 rounded-lg p-4 space-y-3">
        <h4 className="text-sm font-semibold">2. Conditional information</h4>
        <p className="text-[11px] text-slate-500">Requires exactly one declared nuisance and
          reports conditional association—not confounding removal.</p>
        <Button working="conditional" onClick={() => void runConditional()}
          disabled={!hasNuisance || !available('conditional_information_audit')}>
          {conditionalPlan ? 'Audit prepared plan' : 'Plan and audit'}</Button>
        {conditionalPlan && <p className="font-mono text-[10px] text-slate-500">
          plan {short(conditionalPlan.plan_sha256)}</p>}
        {(!hasNuisance || !available('conditional_information_audit')) &&
          <p className="text-[10px] text-amber-400">
            {capability.operations.conditional_information_audit?.reason}</p>}
        <Result title="Conditional-information audit" value={conditional}
          rows={conditional?.candidates} />
      </div>
    </div>

    <div className="border border-slate-800 rounded-lg p-4 space-y-3">
      <h4 className="text-sm font-semibold">3. Generate and internally confirm a stable subspace</h4>
      <p className="text-[11px] text-slate-500">The confirmation rows are reserved at Plan,
        opened once at Confirm, and tested against the complete generated family.</p>
      <div className="flex flex-wrap gap-2">
        <Button working="plan" onClick={() => void planSubspace()}
          disabled={!available('stable_subspace_generation')}>Plan</Button>
        <Button working="generate" onClick={() => void generate()} disabled={!subspacePlan}>Generate</Button>
        <Button working="freeze" onClick={() => void freeze()} disabled={!generation}>Freeze confirmation</Button>
        <Button working="confirm" onClick={() => void confirm()} disabled={!confirmationSeal}>Confirm once</Button>
      </div>
      {subspacePlan && <p className="font-mono text-[10px] text-slate-500">plan {short(subspacePlan.plan_sha256)}</p>}
      <Result title="Generate candidates" value={generation}
        rows={generation?.candidate_compact_stable_subspaces} />
      {confirmationSeal && <p className="text-[10px] text-amber-300 flex gap-2"><Lock className="w-3 h-3" />
        seal {short(confirmationSeal.seal_sha256)} · publish this digest before Confirm</p>}
      <Result title="Internal replication" value={confirmation}
        rows={confirmation?.subspaces} />
    </div>

    <div className="border border-slate-800 rounded-lg p-4 space-y-3">
      <h4 className="text-sm font-semibold flex gap-2 items-center"><ExternalLink className="w-4 h-4" />
        4. External certification</h4>
      <p className="text-[11px] text-slate-500">Publish one internally replicated definition,
        then bind it to independently acquired target bytes and provenance before opening them.
        This first recipe allows no scaling, schema, unit, or span adaptation.</p>
      <Button working="publish" onClick={() => void publish()} disabled={replicated.length < 1}>
        Publish first replicated candidate</Button>
      {published && <p className="font-mono text-[10px] text-slate-500">candidate {short(published.candidate_sha256)}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <label className="text-xs text-slate-400">Independent target CSV
          <input type="file" accept=".csv,.tsv,.txt,text/csv" onChange={(event) => {
            setExternalFile(event.target.files?.[0] ?? null); setTransferSeal(null); setExternalReceipt(null);
          }} className="mt-1 block w-full text-xs" /></label>
        <label className="text-xs text-slate-400">Acquisition identifier
          <input value={acquisitionId} onChange={(event) => setAcquisitionId(event.target.value)}
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2" /></label>
        <label className="text-xs text-slate-400">Archive / source
          <input value={acquisitionSource} onChange={(event) => setAcquisitionSource(event.target.value)}
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2" /></label>
        <label className="text-xs text-slate-400">Acquired at
          <input type="datetime-local" value={acquiredAt} onChange={(event) => setAcquiredAt(event.target.value)}
            className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2" /></label>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button working="transfer-freeze" onClick={() => void freezeTransfer()}
          disabled={!published || !externalFile || !acquisitionId.trim() ||
            !acquisitionSource.trim() || !acquiredAt.trim()}>Freeze target contract</Button>
        <Button working="certify" onClick={() => void certify()} disabled={!transferSeal}>
          Open target and certify once</Button>
      </div>
      {transferSeal && <p className="text-[10px] text-amber-300">transfer seal {short(transferSeal.seal_sha256)} · publish before opening</p>}
      <Result title="External certification receipt" value={externalReceipt}
        rows={externalReceipt?.subspaces} />
    </div>
  </section>;
};

export default RepresentationStructureProgramme;
