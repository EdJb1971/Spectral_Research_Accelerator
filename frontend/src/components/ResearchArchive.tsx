import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Archive, Beaker, BookOpen, Database, FileCheck2, Landmark, MessageSquare, Play,
  Plus, RefreshCw, Search, ShieldCheck,
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
  { key: 'all', label: 'All activity' },
  { key: 'study', label: 'Studies' },
  { key: 'run', label: 'Experiment runs' },
  { key: 'gate', label: 'Decision records' },
  { key: 'evaluation', label: 'Evaluations' },
  { key: 'acquisition', label: 'Data imports' },
  { key: 'validation', label: 'System checks' },
];

const KIND_META: Record<ArchiveKind, { label: string; colour: string }> = {
  study: { label: 'Scientific evidence', colour: 'text-teal-200 border-teal-500/30 bg-teal-500/10' },
  run: { label: 'Experiment run', colour: 'text-blue-200 border-blue-500/30 bg-blue-500/10' },
  gate: { label: 'Decision record', colour: 'text-emerald-200 border-emerald-500/30 bg-emerald-500/10' },
  evaluation: { label: 'Evaluation receipt', colour: 'text-cyan-200 border-cyan-500/30 bg-cyan-500/10' },
  acquisition: { label: 'Acquisition record', colour: 'text-amber-200 border-amber-500/30 bg-amber-500/10' },
  validation: { label: 'System check', colour: 'text-violet-200 border-violet-500/30 bg-violet-500/10' },
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
      detail: `Known-answer system check · checks: ${row.gates.join(', ')}`,
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
            <Archive className="h-5 w-5 text-teal-400" aria-hidden="true" /> Research dashboard
          </h2>
          <p className="mt-1 max-w-4xl text-sm leading-relaxed text-slate-400">
            Start a new investigation, return to earlier experiments, inspect results, or ask an
            expert panel to challenge a published study. Everything saved by this installation is
            searchable below.
          </p>
        </div>
        <button type="button" onClick={() => void load()} disabled={busy}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800
                     px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-50">
          <RefreshCw className={`h-4 w-4 ${busy ? 'animate-spin' : ''}`} aria-hidden="true" />
          Refresh dashboard
        </button>
      </header>

      <section className="grid gap-3 md:grid-cols-3" aria-label="Common actions">
        <button type="button" onClick={() => onNavigate('experimentComposer')}
          className="group rounded-xl border border-teal-500/30 bg-teal-500/10 p-4 text-left hover:bg-teal-500/15">
          <Plus className="h-5 w-5 text-teal-300" aria-hidden="true" />
          <strong className="mt-3 block text-sm text-white">Start a new experiment</strong>
          <span className="mt-1 block text-xs text-slate-400">Define a question, select data, freeze the plan, and run it.</span>
        </button>
        <button type="button" onClick={() => onNavigate('acquire')}
          className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-left hover:border-slate-600">
          <Database className="h-5 w-5 text-amber-300" aria-hidden="true" />
          <strong className="mt-3 block text-sm text-white">Add or find data</strong>
          <span className="mt-1 block text-xs text-slate-400">Import a file or use one of the available data sources.</span>
        </button>
        <button type="button" onClick={() => onNavigate('findings')}
          className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-left hover:border-slate-600">
          <BookOpen className="h-5 w-5 text-blue-300" aria-hidden="true" />
          <strong className="mt-3 block text-sm text-white">Read findings</strong>
          <span className="mt-1 block text-xs text-slate-400">Review conclusions, limitations, and supporting evidence.</span>
        </button>
      </section>

      <section className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Archive summary">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <span className="text-xs uppercase tracking-wide text-slate-500">Studies &amp; runs</span>
          <strong className="mt-1 block text-2xl font-semibold text-slate-100">{researchRecords}</strong>
        </div>
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-emerald-300/70">Decisions</span>
          <strong className="mt-1 block text-2xl font-semibold text-emerald-200">{counts.gate}</strong>
        </div>
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-amber-300/70">Data imports</span>
          <strong className="mt-1 block text-2xl font-semibold text-amber-200">{counts.acquisition}</strong>
        </div>
        <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
          <span className="text-xs uppercase tracking-wide text-violet-300/70">System checks</span>
          <strong className="mt-1 block text-2xl font-semibold text-violet-200">{counts.validation}</strong>
        </div>
      </section>

      <section className="instrument-notice" aria-label="How records are classified">
        <ShieldCheck className="instrument-notice__icon h-4 w-4" aria-hidden="true" />
        <div>
          <h3 className="text-sm font-semibold text-amber-100">Results keep their scientific meaning</h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">
            A completed run is work performed, not automatically a supported conclusion. Studies,
            decisions, imported data, and system checks remain separate so you can see exactly what
            each record establishes.
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
                  <div className="flex shrink-0 gap-2">
                    {entry.kind === 'study' && entry.studyId && (
                      <button type="button" onClick={() => onNavigate('review', entry.studyId)}
                        className="inline-flex items-center gap-1.5 rounded-md border border-teal-500/30
                                   bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-200
                                   hover:bg-teal-500/20">
                        <MessageSquare className="h-3.5 w-3.5" aria-hidden="true" /> Discuss
                      </button>
                    )}
                    <button type="button" onClick={() => onNavigate(entry.target, entry.studyId)}
                      className="shrink-0 rounded-md bg-slate-800 px-3 py-1.5 text-xs font-medium
                                 text-slate-200 hover:bg-slate-700">
                      {entry.kind === 'run' ? 'Open experiment'
                        : entry.kind === 'study' ? 'View findings' : 'View details'}
                    </button>
                  </div>
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
