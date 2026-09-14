import { useEffect, useState } from 'react';
import { AlertTriangle, Ban, FileQuestion, FlaskConical, HelpCircle, Scale } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

/** T4E.22: the studies, their results, and what those results may not be used for.
 *
 * Until this panel a reader could see that a study had been *declared* and never what it
 * *measured*. Every offset, falsified prediction and corrected diagnosis lived in
 * `measurements/` and in git, where no interface reached them. A declaration without its result
 * is a promise; a result without its declaration is an assertion; only the pair is evidence.
 *
 * Four rules govern what follows, and each exists because of something this programme did.
 *
 * *A number never appears without what it may not be used for.* The boundary clause renders
 * inside the result, not beneath it, because it is the part a reader is most likely to skip and
 * the part this work most often needed.
 *
 * *A correction is a mark on the study, not a detail in its body.* Three records here were
 * superseded in the open after their first reading proved wrong. A corrected record that reads
 * as current is the dangerous case, so the mark is carried where a reader looks first.
 *
 * *A question with no answer is shown, not filtered.* One study was declared and withdrawn
 * before adoption by derivation -- it has a declaration, a withdrawal and no measurement, and it
 * is the cheapest result the programme produced. A view that hid it would hide that.
 *
 * *This panel offers no way to run anything.* Every route behind it is a GET, the server
 * publishes its own refusals, and they are rendered rather than implied by an absence of
 * buttons.
 */
export default function StudyTrailView({ onError }: { onError?: (m: string) => void }) {
  const [studies, setStudies] = useState<types.IdentityStudies | null>(null);
  const [measurements, setMeasurements] = useState<types.IdentityMeasurementIndex | null>(null);
  const [open, setOpen] = useState<types.IdentityMeasurementView | null>(null);
  const [status, setStatus] = useState('Loading the study trail…');

  useEffect(() => {
    Promise.all([apiService.identityStudies(), apiService.listIdentityMeasurements()])
      .then(([studyIndex, measurementIndex]) => {
        setStudies(studyIndex);
        setMeasurements(measurementIndex);
        setStatus('');
      })
      .catch((error: Error) => { setStatus(error.message); onError?.(error.message); });
  }, [onError]);

  if (status) {
    return <div className="p-4 text-sm text-slate-600 dark:text-slate-300">{status}</div>;
  }
  if (!studies || !measurements) return null;

  const byFile = new Map(measurements.measurements.map((m) => [m.file, m]));
  const asText = (value: unknown): string[] => {
    if (value == null) return [];
    if (Array.isArray(value)) return value.map((entry) => String(entry));
    if (typeof value === 'object') return Object.values(value as object).map((v) => String(v));
    return [String(value)];
  };

  return (
    <div className="p-4 space-y-6" data-testid="study-trail">
      <header className="space-y-1">
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <FlaskConical className="h-5 w-5" aria-hidden="true" />
          What was asked, what was answered, and what the answer does not license
        </h2>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          Each study is the chain the work actually runs: a question declared before it was
          measured, an adoption or signature, a measurement, and a verdict. A study with no
          measurement is a question deliberately left unanswered and is shown as one.
        </p>
      </header>

      {studies.studies.map((study) => (
        <section
          key={study.task}
          data-testid={`study-${study.task}`}
          className="rounded-lg border border-slate-200 dark:border-slate-700 p-3 space-y-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-semibold">{study.task}</h3>
            {study.has_a_result ? (
              <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-900
                               dark:bg-emerald-900 dark:text-emerald-100">
                measured
              </span>
            ) : (
              <span
                data-testid={`no-result-${study.task}`}
                className="flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-xs
                           text-amber-900 dark:bg-amber-900 dark:text-amber-100"
              >
                <HelpCircle className="h-3 w-3" aria-hidden="true" />
                declared, not measured
              </span>
            )}
            {study.corrected_or_superseded.length > 0 && (
              <span
                data-testid={`corrected-${study.task}`}
                className="flex items-center gap-1 rounded bg-rose-100 px-2 py-0.5 text-xs
                           text-rose-900 dark:bg-rose-900 dark:text-rose-100"
              >
                <AlertTriangle className="h-3 w-3" aria-hidden="true" />
                corrected in the open: {study.corrected_or_superseded.join(', ')}
              </span>
            )}
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <Column title="Declared" rows={study.declarations} />
            <Column title="Adopted or signed" rows={study.adoptions ?? []} />
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Measured
              </h4>
              {study.measurements.length === 0 && (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No measurement. The question was fixed and deliberately left unanswered.
                </p>
              )}
              {study.measurements.map((row) => {
                const summary = byFile.get(row.file) ?? row;
                return (
                  <button
                    key={row.file}
                    type="button"
                    onClick={() => apiService.identityMeasurement(row.file)
                      .then(setOpen)
                      .catch((e: Error) => { setStatus(e.message); onError?.(e.message); })}
                    className="w-full rounded border border-slate-200 dark:border-slate-700 p-2
                               text-left text-xs hover:bg-slate-50 dark:hover:bg-slate-800"
                  >
                    <span className="block font-mono">{row.file}</span>
                    {summary.verdict != null && (
                      <span data-testid={`verdict-${row.file}`} className="mt-1 block font-medium">
                        {String(summary.verdict).slice(0, 220)}
                      </span>
                    )}
                    {asText(summary.boundaries.map((b) => b.text).flat()).slice(0, 2).map(
                      (text, index) => (
                        <span
                          key={index}
                          data-testid={`boundary-${row.file}`}
                          className="mt-1 flex items-start gap-1 text-slate-600 dark:text-slate-300"
                        >
                          <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
                          <span>{text.slice(0, 200)}</span>
                        </span>
                      ))}
                    {summary.boundaries.length === 0 && (
                      <span className="mt-1 flex items-center gap-1 text-amber-700 dark:text-amber-300">
                        <FileQuestion className="h-3 w-3" aria-hidden="true" />
                        no stated boundary; this record predates the convention
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </section>
      ))}

      <section className="space-y-2 rounded-lg border border-slate-200 dark:border-slate-700 p-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <Scale className="h-4 w-4" aria-hidden="true" />
          What this surface will not do
        </h3>
        <ul className="space-y-1 text-xs text-slate-600 dark:text-slate-300">
          {studies.refusals.map((refusal) => (
            <li key={refusal} data-testid="surface-refusal" className="flex items-start gap-1">
              <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
              <span>{refusal}</span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {studies.declared_but_not_measured_note}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {measurements.boundary_note}
        </p>
      </section>

      {open && (
        <section
          data-testid="measurement-detail"
          className="rounded-lg border border-slate-300 dark:border-slate-600 p-3 space-y-2"
        >
          <div className="flex items-center justify-between">
            <h3 className="font-mono text-sm">{open.summary.file}</h3>
            <button type="button" onClick={() => setOpen(null)} className="text-xs underline">
              close
            </button>
          </div>
          <pre className="max-h-96 overflow-auto rounded bg-slate-50 p-2 text-xs
                          dark:bg-slate-900">
            {JSON.stringify(open.measurement, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}

function Column({ title, rows }: { title: string; rows: types.IdentityAuditSummary[] }) {
  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h4>
      {rows.length === 0 && (
        <p className="text-xs text-slate-500 dark:text-slate-400">none</p>
      )}
      {rows.map((row) => (
        <div
          key={row.file}
          className="rounded border border-slate-200 dark:border-slate-700 p-2 text-xs"
        >
          <span className="block font-mono">{row.file}</span>
          {row.status && <span className="mt-1 block">{String(row.status).slice(0, 160)}</span>}
        </div>
      ))}
    </div>
  );
}
