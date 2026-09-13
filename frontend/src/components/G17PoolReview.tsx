import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Database, FileSignature, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import type { G17PoolSurface } from '../types/api';
import { rememberSignerLabel, useSignerIdentity } from './useSignerIdentity';

export default function G17PoolReview() {
  const [state, setState] = useState<G17PoolSurface | null>(null);
  const [message, setMessage] = useState('Loading corrected pool evidence...');
  // A refusal is a result and must outlive the next reload. `message` is cleared by every
  // successful load, so a refusal parked there can be wiped by a refresh that lands after
  // it. Refusals get their own state, which nothing else clears.
  const [refusal, setRefusal] = useState('');
  const [decision, setDecision] = useState<'' | 'ESTABLISHED' | 'NOT_ESTABLISHED'>('');
  const [basis, setBasis] = useState('');
  const [unknowns, setUnknowns] = useState('');
  const [limitations, setLimitations] = useState('');
  const [boundary, setBoundary] = useState('');
  const { name, setName, role, setRole, remember, setRemember, persist } = useSignerIdentity();
  const [what, setWhat] = useState('');
  const [why, setWhy] = useState('');
  const [affirmation, setAffirmation] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setState(await apiService.g17Pool()); setMessage(''); }
    catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const writeReview = async () => {
    if (!decision) return;
    setBusy(true);
    try {
      await apiService.writeG17PoolReview({
        exchangeability: decision, basis,
        unmeasured_properties: unknowns.split('\n').map((value) => value.trim()).filter(Boolean),
        limitations, claim_boundary: boundary,
      });
      setMessage('Review written. It is not adopted.');
      await load();
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  };

  const adopt = async () => {
    setBusy(true);
    setRefusal('');
    try {
      await apiService.adoptG17PoolReview({
        adopted_by: name, adopted_as: role, what_was_adopted: what,
        affirmation, why: why || undefined,
      });
      persist();
      setMessage('Review adopted and verified against the live packet.');
      setAffirmation('');
      await load();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  };

  if (!state) return <p role="status" className="text-xs text-slate-500">{message}</p>;
  const request = state.review_request;

  return (
    <section className="border border-slate-800 bg-slate-900/50 p-5 space-y-4"
             aria-labelledby="g17-pool-title" data-testid="g17-pool">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="g17-pool-title" className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Database className="h-4 w-4 text-teal-400" aria-hidden="true" />
            G17 real-record pool
          </h3>
          <p className="mt-1 text-[11px] text-slate-500">
            Corrected per-TIC lineage, profile qualification, marginal admission, and human review.
          </p>
        </div>
        <button type="button" onClick={() => void load()} title="Refresh G17 pool evidence"
                className="p-2 text-slate-300 hover:text-white">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-px border border-slate-800 bg-slate-800 md:grid-cols-4">
        {[
          ['Qualified profiles', state.readiness.profile_count],
          ['Recorded refusals', state.qualification.refused_target_count],
          ['48-core', state.packet.core_size],
          ['Recommended subset', state.packet.recommended_size],
        ].map(([label, value]) => (
          <div key={label} className="bg-slate-950 p-3">
            <div className="text-lg font-semibold text-slate-100">{value}</div>
            <div className="text-[10px] uppercase text-slate-500">{label}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="border border-slate-800 p-3 text-xs space-y-2">
          <div className="flex justify-between"><span>Readiness</span><strong>{state.readiness.status}</strong></div>
          <div className="flex justify-between"><span>Exchangeability</span><strong>{state.packet.exchangeability}</strong></div>
          <div className="flex justify-between"><span>Alternatives per selected record</span>
            <strong>{state.packet.alternatives_per_recommended_record}</strong></div>
          <p className="break-all font-mono text-[10px] text-slate-500">{state.packet.packet_sha256}</p>
        </div>
        <div className="border border-slate-800 p-3 text-xs space-y-2">
          <div className="font-semibold text-slate-300">Surface coverage</div>
          {state.operations.map((item) => (
            <div key={item.operation} className="flex items-start gap-2">
              {item.available_in_ui
                ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />
                : <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />}
              <span>{item.operation}{item.reason ? `: ${item.reason}` : ''}</span>
            </div>
          ))}
        </div>
      </div>

      <details className="border border-slate-800 p-3 text-xs">
        <summary className="cursor-pointer font-semibold text-slate-300">57 recommended record IDs</summary>
        <p className="mt-2 break-words font-mono text-[10px] text-slate-500">
          {state.packet.recommended_record_ids.join(' · ')}
        </p>
      </details>
      <details className="border border-slate-800 p-3 text-xs">
        <summary className="cursor-pointer font-semibold text-slate-300">
          Eight qualification refusals and archived evidence
        </summary>
        <ul className="mt-2 space-y-1 text-slate-400">
          {state.qualification.refused_targets.map((item) => (
            <li key={item.tic_id}><span className="font-mono">TIC {item.tic_id}</span>: {item.reason}</li>
          ))}
        </ul>
        <p className="mt-2 text-slate-500">Archived: {state.archived_lineage_bug_artifacts.join(', ')}</p>
      </details>

      {state.review.status === 'NOT_WRITTEN' && (
        <div className="border border-amber-500/30 bg-amber-500/5 p-3 space-y-3">
          <h4 className="text-sm font-semibold text-amber-100">Human curation review</h4>
          <div className="flex gap-2" role="group" aria-label="Exchangeability decision">
            {request.human_inputs_required.exchangeability.map((value) => (
              <button key={value} type="button" onClick={() => setDecision(value as typeof decision)}
                      className={`px-3 py-1.5 text-xs border ${decision === value
                        ? 'border-teal-400 bg-teal-500/10 text-teal-200'
                        : 'border-slate-700 text-slate-400'}`}>{value}</button>
            ))}
          </div>
          <textarea aria-label="Scientific basis" value={basis} onChange={(event) => setBasis(event.target.value)}
                    placeholder="Scientific basis, authored by the reviewer" rows={3}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Unmeasured properties" value={unknowns}
                    onChange={(event) => setUnknowns(event.target.value)}
                    placeholder="One unmeasured property per line" rows={3}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Limitations" value={limitations}
                    onChange={(event) => setLimitations(event.target.value)} placeholder="Limitations" rows={2}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Claim boundary" value={boundary}
                    onChange={(event) => setBoundary(event.target.value)} placeholder="Claim boundary" rows={2}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <button type="button" disabled={busy || !decision} onClick={() => void writeReview()}
                  className="px-3 py-2 text-xs font-semibold bg-teal-700 text-white disabled:opacity-40">
            Write review without adopting
          </button>
        </div>
      )}

      {state.review.status === 'WRITTEN_NOT_ADOPTED' && (
        <div className="border border-slate-700 p-3 space-y-3">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <FileSignature className="h-4 w-4" /> Adopt written review
          </h4>
          <p className="break-all font-mono text-[10px] text-slate-500">{state.review.declaration_sha256}</p>
          <input aria-label="Reviewer name" value={name} onChange={(event) => setName(event.target.value)}
                 placeholder="Full name" className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <input aria-label="Reviewer role" value={role} onChange={(event) => setRole(event.target.value)}
                 placeholder="Review role" className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="What was adopted" value={what} onChange={(event) => setWhat(event.target.value)}
                    placeholder="What was adopted" rows={2}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Why adopted" value={why} onChange={(event) => setWhy(event.target.value)}
                    placeholder="Why you adopted it (optional)" rows={2}
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <label className="flex items-start gap-2 text-[11px] text-slate-500">
            <input type="checkbox" aria-label="Remember signer identity" checked={remember}
                   onChange={(event) => setRemember(event.target.checked)}
                   className="mt-0.5" data-testid="remember-signer" />
            <span>{rememberSignerLabel()}</span>
          </label>
          <label className="block text-xs text-slate-400">Type: {request.human_inputs_required.affirmation}
            <input aria-label="Adoption affirmation" value={affirmation}
                   onChange={(event) => setAffirmation(event.target.value)}
                   className="mt-1 w-full border border-slate-700 bg-slate-950 p-2" />
          </label>
          <button type="button" disabled={busy} onClick={() => void adopt()}
                  className="px-3 py-2 text-xs font-semibold bg-slate-100 text-slate-950 disabled:opacity-40">
            Sign and verify review
          </button>
          {refusal && (
            <p role="alert" data-testid="g17-adopt-refusal"
               className="border border-amber-500/40 bg-amber-500/5 p-2 text-[11px]
                          text-amber-100">
              {refusal}
            </p>
          )}
        </div>
      )}

      {state.review.status === 'ADOPTED' && (
        <p className="border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-emerald-200">
          Review adopted and bound to the current packet. Successor promotion remains a separate step.
        </p>
      )}
      <p role="status" className="text-xs text-slate-400">{message}</p>
      <p className="text-[10px] text-slate-500">{state.claim_boundary}</p>
    </section>
  );
}