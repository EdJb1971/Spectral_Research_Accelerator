import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Archive, Beaker, BookOpen, Database, FileCheck2, Landmark, Play,
  RefreshCw, Search, ShieldCheck,
} from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

type ArchiveKind = 'study' | 'run' | 'gate' | 'evaluation' | 'acquisition' | 'validation';

interface ArchiveEntry {
  id: string;
  kind: ArchiveKind;
  classification: string;
  title: string;
  description: string;
  status: string;
  detail?: string;
  target: string;
  studyId?: string;
}

interface Props {
  onNavigate: (target: string, studyId?: string) => void;
  onError?: (message: string) => void;
}

const FILTERS: Array<{ key: 'all' | ArchiveKind; label: string }> = [
  { key: 'all', label: 'All records' },
  { key: 'study', label: 'Published studies' },
  { key: 'run', label: 'Experiment runs' },
  { key: 'gate', label: 'Gate receipts' },
  { key: 'evaluation', label: 'Forecast evaluations' },
  { key: 'acquisition', label: 'Acquisitions' },
  { key: 'validation', label: 'Validation fixtures' },
];

const KIND_META: Record<ArchiveKind, { label: string; colour: string }> = {
  study: { label: 'Scientific evidence', colour: 'text-teal-200 border-teal-500/30 bg-teal-500/10' },
  run: { label: 'Experiment run', colour: 'text-blue-200 border-blue-500/30 bg-blue-500/10' },
  gate: { label: 'Gate receipt', colour: 'text-emerald-200 border-emerald-500/30 bg-emerald-500/10' },
  evaluation: { label: 'Evaluation receipt', colour: 'text-cyan-200 border-cyan-500/30 bg-cyan-500/10' },
  acquisition: { label: 'Acquisition record', colour: 'text-amber-200 border-amber-500/30 bg-amber-500/10' },
  validation: { label: 'Validation fixture', colour: 'text-violet-200 border-violet-500/30 bg-violet-500/10' },
};

function KindIcon({ kind }: { kind: ArchiveKind }) {
  const className = 'h-4 w-4';
  if (kind === 'study') return <BookOpen className={className} aria-hidden="true" />;
  if (kind === 'run') return <Play className={className} aria-hidden="true" />;
  if (kind === 'gate') return <Landmark className={className} aria-hidden="true" />;
  if (kind === 'evaluation') return <FileCheck2 className={className} aria-hidden="true" />;
  if (kind === 'acquisition') return <Database className={className} aria-hidden="true" />;
  return <Beaker className={className} aria-hidden="true" />;
}

function entriesFrom(
  studies: types.StudySummary[], runs: types.RunContract,
  gates: types.GateReceiptIndex, evaluations: types.EvaluationReport[],
  probes: types.ZarrProbeLedgerResponse, cdsJobs: types.CDSJobList,
  benchmarks: types.BenchmarkResponse[],
): ArchiveEntry[] {
  return [
    ...studies.map((row): ArchiveEntry => ({
      id: row.study_id || row.file,
      kind: 'study', classification: row.readable ? 'SCIENTIFIC EVIDENCE' : 'UNREADABLE RECORD',
      title: row.study_id || row.file,
      description: row.readable
        ? (row.hypothesis || 'Published evidence bundle')
        : (row.refused_because || 'This bundle could not be verified.'),
      status: row.readable ? (row.rung || 'UNASSESSED') : 'REFUSED',
      detail: row.readable && row.revision !== undefined
        ? `revision ${row.revision} · ${row.blocked ? 'claim ladder capped' : 'no recorded cap'}` : undefined,
      target: 'findings', studyId: row.study_id || undefined,
    })),
    ...runs.runs.map((row): ArchiveEntry => ({
      id: row.run_id, kind: 'run', classification: 'EXPERIMENT RUN', title: row.title,
      description: `${row.study_id} · content-addressed manifest ${row.manifest_sha256.slice(0, 12)}…`,
      status: row.state, detail: 'An executed plan is not automatically admitted evidence.',
      target: 'experimentComposer', studyId: row.study_id,
    })),
    ...gates.receipts.map((row): ArchiveEntry => ({
      id: row.receipt_id, kind: 'gate', classification: 'GATE RECEIPT',
      title: row.study_id || row.receipt_id,
      description: `${row.evidence_role || 'unclassified evidence role'} · replication gate ${row.gate_verdict}`,
      status: row.scientific_verdict, detail: row.power_applied
        ? 'The scientific verdict includes the declared power adjudication.'
        : 'Power adjudication was not applicable to this verdict.',
      // A gate's study label is not proof that Findings has a published evidence bundle.
      target: 'gate',
    })),
    ...evaluations.map((row): ArchiveEntry => ({
      id: row.report_id, kind: 'evaluation', classification: 'EVALUATION RECEIPT',
      title: row.report_id,
      description: `${row.scope.evaluation_role} · ${row.scope.split} · ${row.scope.variables.join(', ')}`,
      status: row.readiness.scientific_skill,
      detail: row.readiness.reason, target: 'evaluation',
    })),
    ...cdsJobs.jobs.filter((job) => job.state === 'COMPLETE' && job.acquisition_record)
      .map((job): ArchiveEntry => ({
        id: job.acquisition_record!.record_sha256,
        kind: 'acquisition', classification: 'ACQUISITION RECORD',
        title: `ERA5 · ${job.request.date_start} to ${job.request.date_end}`,
        description: `${job.acquisition_record!.completed_shards} verified monthly shards · ${
          job.request.variables.join(', ')} · request ${job.request_sha256.slice(0, 12)}…`,
        status: 'COMPLETE',
        detail: 'Transfer and integrity provenance only; not analysis or evidence.',
        target: 'acquire',
      })),
    ...probes.probes.map((row): ArchiveEntry => ({
      id: row.digest, kind: 'acquisition', classification: 'ACQUISITION RECORD',
      title: row.uri, description: row.outcome_means,
      status: row.outcome.toUpperCase(), detail: `${row.evidence_means} · ${row.probed_on}`,
      target: 'acquire',
    })),
    ...benchmarks.map((row): ArchiveEntry => ({
      id: row.name, kind: 'validation', classification: 'VALIDATION FIXTURE',
      title: row.name, description: row.description,
      status: row.is_null ? 'KNOWN NULL' : 'KNOWN ANSWER',
      detail: `Software acceptance fixture · gates: ${row.gates.join(', ')}`,
      target: 'platform',
    })),
  ];
}

export default function ResearchArchive({ onNavigate, onError }: Props) {
  const [entries, setEntries] = useState<ArchiveEntry[]>([]);
  const [filter, setFilter] = useState<'all' | ArchiveKind>('all');
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [studies, runs, gates, evaluations, probes, cdsJobs, benchmarks] = await Promise.all([
        apiService.listStudies(), apiService.experimentRunContract(), apiService.listGateReceipts(),
        apiService.listEvaluationReports(), apiService.zarrProbes(), apiService.listCDSJobs(),
        apiService.listBenchmarks(),
      ]);
      setEntries(entriesFrom(studies, runs, gates, evaluations, probes, cdsJobs, benchmarks));
    } catch (error) {
      onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [onError]);

  useEffect(() => { void load(); }, [load]);

  const counts = useMemo(() => Object.fromEntries(
    (Object.keys(KIND_META) as ArchiveKind[]).map((kind) => [
      kind, entries.filter((entry) => entry.kind === kind).length,
    ]),
  ) as Record<ArchiveKind, number>, [entries]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => (filter === 'all' || entry.kind === filter)
      && (!needle || [entry.title, entry.description, entry.status, entry.classification]
        .some((value) => value.toLowerCase().includes(needle))));
  }, [entries, filter, query]);

  const researchRecords = counts.study + counts.run + counts.gate + counts.evaluation;

  return (
    <div className="max-w-[112rem] space-y-6 animate-fadeIn" aria-busy={busy}>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-xl font-bold text-white">
            <Archive className="h-5 w-5 text-teal-400" aria-hidden="true" /> Research Archive
          </h2>
          <p className="mt-1 max-w-4xl text-sm leading-relaxed text-slate-400">
            One index over persisted scientific records and software-validation evidence. Record
            classes remain separate: a passing fixture is not a published study, and a completed
            run is not automatically admitted evidence.
          </p>
        </div>
        <button type="button" onClick={() => void load()} disabled={busy}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800
                     px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-50">
          <RefreshCw className={`h-4 w-4 ${busy ? 'animate-spin' : ''}`} aria-hidden="true" />
          Refresh index
        </button>
      </header>

      <section className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Archive summary">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <span className="text-xs uppercase tracking-wide text-slate-500">Research records</span>
          <strong className="mt-1 block text-2xl font-semibold text-slate-100">{researchRecords}</strong>
        </div>
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-emerald-300/70">Gate receipts</span>
          <strong className="mt-1 block text-2xl font-semibold text-emerald-200">{counts.gate}</strong>
        </div>
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-amber-300/70">Acquisition records</span>
          <strong className="mt-1 block text-2xl font-semibold text-amber-200">{counts.acquisition}</strong>
        </div>
        <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-violet-300/70">Known-answer fixtures</span>
          <strong className="mt-1 block text-2xl font-semibold text-violet-200">{counts.validation}</strong>
        </div>
      </section>

      <section className="instrument-notice" aria-label="Test evidence boundary">
        <ShieldCheck className="instrument-notice__icon h-4 w-4" aria-hidden="true" />
        <div>
          <h3 className="text-sm font-semibold text-amber-100">Why pytest studies are not listed as studies</h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">
            Automated tests write evidence bundles and experiment runs into isolated temporary
            stores. They prove implementation behavior and are discarded after the test. The
            known-answer catalogue below is inspectable as validation evidence, but it receives no
            scientific claim rung and never enters Findings automatically.
          </p>
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/45 p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div role="tablist" aria-label="Archive record class" className="flex flex-wrap gap-2">
            {FILTERS.map((item) => (
              <button key={item.key} type="button" role="tab" aria-selected={filter === item.key}
                onClick={() => setFilter(item.key)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium ${filter === item.key
                  ? 'border-teal-500/40 bg-teal-500/10 text-teal-200'
                  : 'border-slate-800 bg-slate-950/50 text-slate-400 hover:border-slate-700 hover:text-slate-200'}`}>
                {item.label}{item.key !== 'all' ? ` · ${counts[item.key]}` : ` · ${entries.length}`}
              </button>
            ))}
          </div>
          <label className="relative block min-w-0 xl:w-80">
            <span className="sr-only">Search archive</span>
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-500"
              aria-hidden="true" />
            <input type="search" value={query} onChange={(event) => setQuery(event.target.value)}
              placeholder="Search IDs, status, or description"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 py-2 pl-9 pr-3
                         text-sm text-slate-200 placeholder:text-slate-600" />
          </label>
        </div>
      </section>

      {visible.length > 0 ? (
        <ul className="grid grid-cols-1 gap-3 xl:grid-cols-2" aria-label="Archive records">
          {visible.map((entry) => {
            const meta = KIND_META[entry.kind];
            return (
              <li key={`${entry.kind}:${entry.id}`}
                className="group rounded-xl border border-slate-800 bg-slate-900/55 p-4
                           transition-colors hover:border-slate-700">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <span className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1
                                      text-[10px] font-semibold uppercase tracking-wide ${meta.colour}`}>
                      <KindIcon kind={entry.kind} /> {meta.label}
                    </span>
                    <h3 className="mt-3 truncate text-sm font-semibold text-slate-100" title={entry.title}>
                      {entry.title}
                    </h3>
                  </div>
                  <span className="shrink-0 rounded-md bg-slate-950 px-2 py-1 font-mono text-[10px]
                                   text-slate-300 ring-1 ring-inset ring-slate-800">
                    {entry.status}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{entry.description}</p>
                {entry.detail && <p className="mt-2 text-xs leading-relaxed text-slate-500">{entry.detail}</p>}
                <div className="mt-4 flex items-center justify-between gap-3 border-t border-slate-800 pt-3">
                  <code className="truncate text-[10px] text-slate-600" title={entry.id}>{entry.id}</code>
                  <button type="button" onClick={() => onNavigate(entry.target, entry.studyId)}
                    className="shrink-0 rounded-md bg-slate-800 px-3 py-1.5 text-xs font-medium
                               text-slate-200 hover:bg-slate-700">
                    Inspect record
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="findings-empty">
          <div>
            <Archive className="mx-auto h-8 w-8 text-slate-600" aria-hidden="true" />
            <h3 className="mt-3 text-sm font-semibold text-slate-200">No records match this view</h3>
            <p className="mt-1 text-sm text-slate-500">Change the record class or clear the search.</p>
          </div>
        </div>
      )}
    </div>
  );
}
