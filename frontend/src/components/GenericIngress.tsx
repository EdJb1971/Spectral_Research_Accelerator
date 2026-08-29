/** G14: file-first ingress with explicit roles and a sealed generate/confirm audit. */
import { useMemo, useState } from 'react';
import { FileSearch, Loader2, Play, ShieldCheck, Upload } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';
import DatasetCapabilityProfile from './DatasetCapabilityProfile';

const ROLES: types.SampleRole[] = [
  'ignore', 'sample_id', 'target', 'nuisance', 'feature', 'group', 'ordering',
];

const GenericIngress: React.FC<{ onError?: (message: string) => void;
  onCapability?: (profile: types.DatasetCapabilityProfile | null) => void }> = ({ onError, onCapability }) => {
  const [file, setFile] = useState<File | null>(null);
  const [probe, setProbe] = useState<types.FileProbe | null>(null);
  const [roles, setRoles] = useState<Record<string, types.SampleRole>>({});
  const [units, setUnits] = useState<Record<string, string>>({});
  const [relationship, setRelationship] = useState<'independent' | 'grouped' | 'ordered'>('independent');
  const [plan, setPlan] = useState<types.RepresentationAuditPlan | null>(null);
  const [result, setResult] = useState<types.RepresentationAuditResult | null>(null);
  const [capability, setCapability] = useState<types.DatasetCapabilityProfile | null>(null);
  const [busy, setBusy] = useState<'probe' | 'plan' | 'audit' | null>(null);

  const featureCount = useMemo(() => Object.values(roles).filter((role) => role === 'feature').length,
    [roles]);

  const fail = (error: unknown) => onError?.(error instanceof Error ? error.message : String(error));

  const inspect = async () => {
    if (!file) return;
    setBusy('probe'); setPlan(null); setResult(null); setCapability(null); onCapability?.(null);
    try {
      const found = await apiService.probeGenericFile(file);
      setProbe(found);
      setRoles(Object.fromEntries(found.columns.map((column) => [column.name, 'ignore'])));
      setUnits(Object.fromEntries(found.columns.filter((column) => column.storage_type === 'numeric')
        .map((column) => [column.name, 'dimensionless'])));
    } catch (error) { fail(error); } finally { setBusy(null); }
  };

  const freeze = async () => {
    if (!file || !probe) return;
    setBusy('plan'); setResult(null);
    try {
      const profile = await apiService.genericFileCapabilities(file, {
        roles, units, sample_relationship: relationship,
      });
      setCapability(profile); onCapability?.(profile);
      if (!profile.operations.representation_audit?.available) return;
      setPlan(await apiService.planRepresentationAudit(file, {
        roles, units, sample_relationship: relationship,
      }, { pcaComponents: Math.max(1, Math.min(3, featureCount)), permutations: 4999 }));
    } catch (error) { fail(error); } finally { setBusy(null); }
  };

  const run = async () => {
    if (!file || !plan) return;
    setBusy('audit');
    try { setResult(await apiService.runRepresentationAudit(file, plan)); }
    catch (error) { fail(error); } finally { setBusy(null); }
  };

  return <section className="border border-teal-500/30 bg-teal-500/5 rounded-xl p-5 space-y-4">
    <div className="flex gap-3 items-start">
      <Upload className="w-6 h-6 text-teal-400 shrink-0" />
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Load a file. Let&apos;s analyse it.</h2>
        <p className="text-xs text-slate-400 mt-1">The probe reads shape and storage facts only.
          You declare target, nuisance and feature meaning; names are never interpreted.</p>
      </div>
    </div>
    <div className="flex flex-col sm:flex-row gap-2">
      <label className="flex-1 text-xs text-slate-400">CSV or TSV sample table
        <input type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values"
          onChange={(event) => { setFile(event.target.files?.[0] ?? null); setProbe(null); setPlan(null); setResult(null); setCapability(null); onCapability?.(null); }}
          className="mt-1 block w-full text-xs file:bg-slate-800 file:text-slate-200 file:border-0 file:rounded file:px-3 file:py-2" />
      </label>
      <button type="button" onClick={() => void inspect()} disabled={!file || busy !== null}
        className="self-end bg-teal-600 hover:bg-teal-500 disabled:opacity-50 rounded px-4 py-2 text-xs font-semibold flex gap-2">
        {busy === 'probe' ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileSearch className="w-4 h-4" />}
        Probe file
      </button>
    </div>

    {probe && <>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="bg-slate-950/60 rounded p-2"><span className="block text-slate-500">Rows</span>{probe.n_rows}</div>
        <div className="bg-slate-950/60 rounded p-2"><span className="block text-slate-500">Columns</span>{probe.n_columns}</div>
        <div className="bg-slate-950/60 rounded p-2"><span className="block text-slate-500">Meaning inferred</span>No</div>
      </div>
      <div className="overflow-x-auto border border-slate-800 rounded-lg">
        <table className="w-full text-xs">
          <thead className="bg-slate-950 text-slate-500"><tr>
            <th className="text-left p-2">Column</th><th className="text-left p-2">Storage facts</th>
            <th className="text-left p-2">Declared role</th><th className="text-left p-2">Units</th>
          </tr></thead>
          <tbody>{probe.columns.map((column) => <tr key={column.name} className="border-t border-slate-800">
            <td className="p-2 font-mono text-slate-200">{column.name}</td>
            <td className="p-2 text-slate-500">{column.storage_type} · {column.missing_count} missing · {column.distinct_count} distinct</td>
            <td className="p-2"><label className="sr-only" htmlFor={`role-${column.name}`}>Role for {column.name}</label>
              <select id={`role-${column.name}`} value={roles[column.name] ?? 'ignore'}
                onChange={(event) => { setRoles({ ...roles, [column.name]: event.target.value as types.SampleRole }); setPlan(null); setResult(null); setCapability(null); onCapability?.(null); }}
                className="bg-slate-950 border border-slate-700 rounded p-1.5">
                {ROLES.map((role) => <option key={role}>{role}</option>)}
              </select></td>
            <td className="p-2">{column.storage_type === 'numeric' && roles[column.name] !== 'ignore' ?
              <label className="sr-only" htmlFor={`unit-${column.name}`}>Units for {column.name}</label> : null}
              {column.storage_type === 'numeric' && roles[column.name] !== 'ignore' &&
                <input id={`unit-${column.name}`} value={units[column.name] ?? ''}
                  onChange={(event) => { setUnits({ ...units, [column.name]: event.target.value }); setPlan(null); setResult(null); setCapability(null); onCapability?.(null); }}
                  className="w-32 bg-slate-950 border border-slate-700 rounded p-1.5" />}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="flex flex-col sm:flex-row gap-3 items-end">
        <label className="text-xs text-slate-400">How are rows related?
          <select value={relationship} onChange={(event) => { setRelationship(event.target.value as typeof relationship); setPlan(null); setResult(null); setCapability(null); onCapability?.(null); }}
            className="mt-1 block bg-slate-950 border border-slate-700 rounded p-2">
            <option value="independent">Independent samples</option>
            <option value="grouped">Grouped samples</option>
            <option value="ordered">Ordered observations</option>
          </select>
        </label>
        <button type="button" onClick={() => void freeze()} disabled={busy !== null || featureCount < 1}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded px-4 py-2 text-xs font-semibold flex gap-2">
          {busy === 'plan' ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
          Determine paths and freeze
        </button>
      </div>
      <p className="text-[10px] text-slate-500">{probe.claim_boundary}</p>
    </>}

    {capability && <DatasetCapabilityProfile profile={capability} />}

    {plan && <div className="border border-indigo-500/30 bg-indigo-500/5 rounded-lg p-4 text-xs">
      <div className="flex justify-between gap-3 items-start">
        <div><h3 className="font-semibold text-indigo-200">Family sealed before enumeration</h3>
          <p className="text-slate-400 mt-1">{plan.n_tests_per_partition} candidates · {plan.permutations} permutations · {plan.correction}</p>
          <p className="font-mono text-[10px] text-slate-600 mt-1 break-all">{plan.plan_sha256}</p></div>
        <button type="button" onClick={() => void run()} disabled={busy !== null}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded px-4 py-2 font-semibold flex gap-2 shrink-0">
          {busy === 'audit' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          Analyse
        </button>
      </div>
      <p className="text-slate-500 mt-2">{plan.candidates.join(' · ')}</p>
    </div>}

    {result && <div className="border border-emerald-500/30 bg-emerald-500/5 rounded-lg p-4 space-y-3">
      <div><h3 className="font-semibold text-emerald-200">Candidate structural representations</h3>
        <p className="text-xs text-slate-400">Effective feature dimension {result.structure.effective_dimension.toFixed(2)} · generate {result.partitions.generate_n} · held-out confirm {result.partitions.confirm_n}</p></div>
      <div className="space-y-2">{result.candidates.map((candidate) => <div key={candidate.candidate}
        className={`rounded border p-3 text-xs ${candidate.survives_both ? 'border-emerald-500/40' : 'border-slate-800'}`}>
        <div className="flex justify-between gap-2"><strong>{candidate.candidate}</strong>
          <span className={candidate.survives_both ? 'text-emerald-300' : 'text-slate-500'}>
            {candidate.survives_both ? 'survived both partitions' : 'not confirmed'}</span></div>
        <p className="text-slate-500 mt-1">MI generate {candidate.generate_mi_nats.toFixed(4)} (q {candidate.generate_q_value.toPrecision(3)}) · confirm {candidate.confirm_mi_nats.toFixed(4)} (q {candidate.confirm_q_value.toPrecision(3)})</p>
      </div>)}</div>
      <p className="text-[10px] text-slate-500">{result.claim_boundary}</p>
    </div>}
  </section>;
};

export default GenericIngress;
