/** TG11.5: recorded argument beside the evidence, never part of the finding. */

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, MessageSquare, Receipt, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  selectedStudyId: string;
  onStudyId: (studyId: string) => void;
  onError?: (message: string) => void;
}

function Digest({ children }: { children: string }) {
  return <span className="font-mono break-all text-slate-500">{children}</span>;
}

export default function ReviewView({ selectedStudyId, onStudyId, onError }: Props) {
  const [surface, setSurface] = useState<types.ReviewSurface | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!selectedStudyId.trim()) {
      setSurface(null);
      return;
    }
    setBusy(true);
    try {
      setSurface(await apiService.getStudyReview(selectedStudyId.trim()));
    } catch (error) {
      setSurface(null);
      onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [selectedStudyId, onError]);

  useEffect(() => { void load(); }, [load]);

  return (
    <div className="space-y-5" aria-busy={busy}>
      <header>
        <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <MessageSquare size={18} aria-hidden="true" /> Recorded review
        </h3>
        <p className="text-sm text-slate-400 max-w-3xl mt-1">
          Inspect the adversarial exchange recorded beside one exact evidence revision. This
          workspace contains argument, not evidence and not permission to make a claim.
        </p>
      </header>

      <div className="flex flex-col sm:flex-row sm:items-end gap-2 max-w-3xl">
        <div className="flex-1">
          <label htmlFor="review-study-id"
            className="block text-xs uppercase tracking-wide text-slate-400 mb-1">
            Published study ID
          </label>
          <input id="review-study-id" type="text" value={selectedStudyId}
            onChange={(event) => onStudyId(event.target.value)}
            placeholder="Select a study in Findings or enter its ID"
            className="w-full rounded bg-slate-800 px-3 py-2 text-sm text-slate-100" />
        </div>
        <button type="button" onClick={() => void load()} disabled={busy || !selectedStudyId.trim()}
          className="flex items-center justify-center gap-2 rounded bg-slate-800 px-3 py-2 text-sm
                     text-slate-200 hover:bg-slate-700 disabled:opacity-50">
          <RefreshCw size={14} aria-hidden="true" /> Reload exact revision
        </button>
      </div>

      {!selectedStudyId.trim() && (
        <p className="rounded border border-slate-700 bg-slate-900/60 p-4 text-sm text-slate-400">
          No study selected. That is not the same as no review existing.
        </p>
      )}
      {busy && <p role="status" className="text-sm text-slate-400">Loading recorded review…</p>}

      {surface && (
        <>
          <section aria-label="Recorded-not-reproducible boundary"
            className="rounded border-2 border-amber-700/70 bg-amber-950/30 p-4">
            <h4 className="text-sm font-semibold text-amber-200">Recorded, not reproducible</h4>
            <p className="mt-2 text-sm text-amber-100">{surface.declaration}</p>
            <p className="mt-2 text-sm font-semibold text-amber-200">{surface.claim_boundary}</p>
            <p className="mt-3 text-xs text-amber-300/80">
              Bound to {surface.study_id}, bundle revision {surface.bundle_revision},{' '}
              <Digest>{surface.bundle_sha256}</Digest>
            </p>
          </section>

          {surface.absence_note && (
            <p className="rounded border border-slate-700 bg-slate-900/60 p-4 text-sm text-slate-400">
              {surface.absence_note}
            </p>
          )}

          {surface.unreadable.map((artifact) => (
            <div key={artifact.file} role="alert"
              className="rounded border border-red-800 bg-red-950/30 p-3 text-sm text-red-200">
              <AlertTriangle size={14} className="inline mr-2" aria-hidden="true" />
              {artifact.file} was refused: {artifact.refused_because}
            </div>
          ))}

          {surface.reviews.map((review) => (
            <article key={review.record.record_sha256}
              className="rounded border border-slate-700 bg-slate-900/60 p-4 space-y-5">
              <header>
                <h4 className="font-semibold text-slate-100">Recorded exchange</h4>
                <p className="text-xs text-slate-500 mt-1">
                  {review.file} · {review.record.calls.length} recorded calls · record{' '}
                  <Digest>{review.record.record_sha256}</Digest>
                </p>
              </header>
              <section aria-label="Recorded calls">
                <pre className="whitespace-pre-wrap break-words rounded bg-slate-950 p-4 text-xs
                                leading-relaxed text-slate-300 overflow-x-auto">
                  {review.rendered}
                </pre>
              </section>

              <section aria-label="Recorded round-robin outcomes" className="space-y-3">
                <h5 className="text-sm font-semibold text-slate-200">Round-robin outcome</h5>
                {review.outcomes.length === 0 ? (
                  <p className="text-sm italic text-slate-500">
                    No completed outcome recorded. That is not evidence that no exchange occurred.
                  </p>
                ) : review.outcomes.map((entry) => (
                  <pre key={entry.file}
                    className="whitespace-pre-wrap break-words rounded border border-slate-800
                               bg-slate-950 p-4 text-xs leading-relaxed text-slate-300">
                    {entry.rendered}
                  </pre>
                ))}
              </section>

              <section aria-label="Recorded review cost receipts" className="space-y-3">
                <h5 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Receipt size={14} aria-hidden="true" /> Route and token receipts
                </h5>
                {review.cost_receipts.length === 0 ? (
                  <p className="text-sm italic text-slate-500">
                    No cost receipt recorded. That is not the same as the review costing nothing.
                  </p>
                ) : review.cost_receipts.map(({ file, receipt }) => (
                  <div key={file} className="rounded border border-slate-800 bg-slate-950 p-3">
                    <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
                      <div><dt className="text-slate-500">Calls</dt>
                        <dd className="text-slate-200">{receipt.call_count}</dd></div>
                      <div><dt className="text-slate-500">Input tokens</dt>
                        <dd className="text-slate-200">{receipt.input_tokens}</dd></div>
                      <div><dt className="text-slate-500">Cached input tokens</dt>
                        <dd className="text-slate-200">{receipt.cached_input_tokens}</dd></div>
                      <div><dt className="text-slate-500">Output tokens</dt>
                        <dd className="text-slate-200">{receipt.output_tokens}</dd></div>
                      <div><dt className="text-slate-500">Total tokens</dt>
                        <dd className="text-slate-200">{receipt.total_tokens}</dd></div>
                      <div><dt className="text-slate-500">Measured cache-hit fraction</dt>
                        <dd className="text-slate-200">{receipt.cache_hit_fraction}</dd></div>
                    </dl>
                    <p className="mt-3 text-xs text-slate-500">
                      Token and route audit only; no price or review-quality claim. Receipt{' '}
                      <Digest>{receipt.receipt_sha256}</Digest>
                    </p>
                  </div>
                ))}
              </section>
            </article>
          ))}
        </>
      )}
    </div>
  );
}
