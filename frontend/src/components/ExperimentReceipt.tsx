/** TG17.9: portable replay and explicit evidence handoff, never automatic admission. */

import { useEffect, useRef, useState } from 'react';
import { Download, FileCheck2, FileText, Loader2, ShieldCheck, Upload } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';


function download(name: string, type: string, value: string) {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; anchor.click();
  URL.revokeObjectURL(url);
}

interface Props {
  runId?: string;
  runState?: string;
  trustOnly?: boolean;
  onEvidenceHandoff?: (studyId: string) => void;
}

export function ExperimentReceiptPanel({ runId, runState, trustOnly = false,
                                         onEvidenceHandoff }: Props) {
  const [capabilities, setCapabilities] = useState<types.ExperimentReceiptCapabilities | null>(null);
  const [replay, setReplay] = useState<types.ExperimentReceiptReplay | null>(null);
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState('');
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    apiService.experimentReceiptCapabilities().then(setCapabilities)
      .catch((error: Error) => setMessage(`Receipt trust surface unavailable: ${error.message}`));
  }, []);

  const exportBundle = async () => {
    if (!runId) return;
    setBusy('export'); setMessage('');
    try {
      const result = await apiService.exportExperimentReceipt(runId);
      download(`${result.bundle_sha256}.json`, 'application/json', JSON.stringify(result.bundle, null, 2));
      setReplay(await apiService.replayExperimentReceipt(result.bundle));
      setMessage('Immutable bundle exported and replayed with verified identity.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally { setBusy(''); }
  };

  const exportMethods = async () => {
    if (!runId) return;
    setBusy('methods'); setMessage('');
    try {
      download(`${runId}-methods.md`, 'text/markdown', await apiService.experimentMethodsReport(runId));
      setMessage('Methods and limitations report exported.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally { setBusy(''); }
  };

  const importBundle = async (file?: File) => {
    if (!file) return;
    setBusy('replay'); setMessage('');
    try {
      const body = JSON.parse(await file.text()) as types.ExperimentReplayBundle;
      const reconstructed = await apiService.replayExperimentReceipt(body);
      setReplay(reconstructed);
      setMessage('UI state discarded: the run identity, receipt and conclusions were reconstructed from the bundle.');
    } catch (error) {
      setReplay(null); setMessage(error instanceof Error ? error.message : String(error));
    } finally { setBusy(''); if (input.current) input.current.value = ''; }
  };

  if (!capabilities) return <p role="status" className="text-xs text-slate-500">{message || 'Loading receipt trust surface…'}</p>;

  return <section className="border border-teal-500/30 bg-teal-500/5 rounded-xl p-5 space-y-4"
                  aria-labelledby={trustOnly ? 'receipt-trust-title' : 'receipt-export-title'}>
    <div>
      <h3 id={trustOnly ? 'receipt-trust-title' : 'receipt-export-title'}
          className="font-semibold text-slate-100 flex items-center gap-2">
        <FileCheck2 className="w-4 h-4 text-teal-400" aria-hidden="true" />
        {trustOnly ? 'Experiment lineage and receipts' : 'Immutable receipt and evidence handoff'}
      </h3>
      <p className="text-xs text-slate-500 mt-1">{capabilities.claim_boundary}</p>
    </div>

    <ol className="flex flex-wrap gap-2 text-[11px]" aria-label="Experiment receipt lineage">
      {capabilities.lineage.map((item, index) => <li key={item}
        className="px-2 py-1 rounded bg-slate-950 border border-slate-800">
        {index + 1}. {item}
      </li>)}
    </ol>

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs">
      <div>
        <h4 className="font-semibold text-slate-300 mb-2">Registered operations</h4>
        <ul className="space-y-1.5">
          {capabilities.operations.map((row) => <li key={row.name} className="text-slate-400">
            <code className="text-teal-300">{row.name}</code> — {row.effect}. Evidence write: <strong>no</strong>.
          </li>)}
        </ul>
      </div>
      <div>
        <h4 className="font-semibold text-slate-300 mb-2">Structural refusals</h4>
        <ul className="space-y-1.5">
          {capabilities.refusals.map((row) => <li key={row.name} className="text-slate-400">
            <strong className="text-amber-300">{row.name}</strong> — {row.reason}
          </li>)}
        </ul>
      </div>
    </div>

    <details className="text-xs" aria-label="Every field in the portable receipt">
      <summary className="cursor-pointer text-slate-300 font-semibold">Every explained receipt field</summary>
      <dl className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-3">
        {capabilities.receipt_fields.map((row) => <div key={row.name}
          className="rounded bg-slate-950 border border-slate-800 p-2">
          <dt className="font-mono text-teal-300">{row.name}</dt><dd className="text-slate-500">{row.meaning}</dd>
        </div>)}
      </dl>
    </details>

    {!trustOnly && <>
      <div className="flex flex-wrap gap-2">
        <button onClick={() => void exportBundle()} disabled={busy !== '' || runState !== 'COMPLETE'}
          className="px-3 py-1.5 rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-40 text-xs flex gap-2 items-center">
          {busy === 'export' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
          Export verified bundle
        </button>
        <button onClick={() => void exportMethods()} disabled={busy !== '' || runState !== 'COMPLETE'}
          className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-xs flex gap-2 items-center">
          <FileText className="w-3.5 h-3.5" /> Export methods report
        </button>
        <button onClick={() => input.current?.click()} disabled={busy !== ''}
          className="px-3 py-1.5 rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40 text-xs flex gap-2 items-center">
          {busy === 'replay' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
          Verify and replay bundle
        </button>
        <input ref={input} type="file" accept="application/json,.json" className="sr-only"
               aria-label="Choose experiment replay bundle" onChange={(event) => void importBundle(event.target.files?.[0])} />
      </div>
      {runState !== 'COMPLETE' && <p className="text-xs text-amber-300">A completed experiment export is available only
        in COMPLETE state. The live run receipt still preserves this run’s current limitations.</p>}
    </>}

    {replay && !trustOnly && <div className="rounded border border-emerald-500/30 bg-emerald-500/5 p-4 space-y-3">
      <p className="text-sm text-emerald-200 flex gap-2 items-center"><ShieldCheck className="w-4 h-4" />
        {replay.integrity}: bundle {replay.bundle_sha256.slice(0, 16)}… reconstructs run{' '}
        <code>{replay.run_receipt.run_id}</code> in state {replay.run_receipt.state}.</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
        {replay.evidence_handoff.categories.map((row) => <div key={row.category}
          className={`rounded p-2 border ${row.status === 'PRESENT' ? 'border-emerald-500/20 text-emerald-300' : 'border-slate-700 text-slate-500'}`}>
          {row.category.replace(/_/g, ' ')}: {row.status}
        </div>)}
      </div>
      <p className="text-xs text-slate-500">{replay.evidence_handoff.claim_boundary}</p>
      {onEvidenceHandoff && replay.evidence_handoff.eligible_actions.includes('open_reviewable_study_draft') &&
        <button onClick={() => onEvidenceHandoff(replay.evidence_handoff.proposed_study_id)}
          className="px-3 py-1.5 rounded bg-violet-700 hover:bg-violet-600 text-xs">
          Open a separate evidence-study draft
        </button>}
    </div>}
    {message && <p role="status" className="text-xs text-slate-300">{message}</p>}
  </section>;
}

export default ExperimentReceiptPanel;
