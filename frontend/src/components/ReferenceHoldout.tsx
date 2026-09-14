import { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle, CheckCircle2, FileSignature, Lock, RefreshCw, Search, Wind,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { HoldoutArtifact, HoldoutRow, HoldoutRows, HoldoutSurface } from '../types/api';
import { rememberSignerLabel, useSignerIdentity } from './useSignerIdentity';

const CLASSIFICATION_LABEL: Record<string, string> = {
  VERTICAL_QUANTITY_SEPARATION_CANDIDATE: 'MSLP closer',
  CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE: 'Vorticity closer',
  EXACT_TIE: 'Exact tie',
  REFUSED: 'Refused',
};

const km = (value: number | null) => (value === null ? 'refused' : `${value.toFixed(1)} km`);

/** T4E.39: the whole holdout, in the order it was performed, with every stage examinable and
 *  the two human acts it ends in. The verdict is `NOT_AN_ACCEPTANCE` and no control here can
 *  change that, so the panel shows the boundary as prominently as the counts. */
export default function ReferenceHoldout() {
  const [state, setState] = useState<HoldoutSurface | null>(null);
  const [rows, setRows] = useState<HoldoutRows | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [artifact, setArtifact] = useState<HoldoutArtifact | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [message, setMessage] = useState('Loading holdout evidence...');
  // A refusal is a result and must outlive the next reload. `message` is cleared by every
  // successful load, so a refusal parked there can be wiped by a refresh that lands after
  // it. Refusals get their own state, which nothing else clears.
  const [refusal, setRefusal] = useState('');

  const [assessment, setAssessment] = useState<'' | 'BOUNDARY_SOUND' | 'BOUNDARY_DISPUTED'>('');
  const [basis, setBasis] = useState('');
  const [limits, setLimits] = useState('');
  const [limitations, setLimitations] = useState('');
  const [next, setNext] = useState('');
  const [boundary, setBoundary] = useState('');
  const { name, setName, role, setRole, remember, setRemember, persist } = useSignerIdentity();
  const [what, setWhat] = useState('');
  const [why, setWhy] = useState('');
  const [affirmation, setAffirmation] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [surface, measured] = await Promise.all([
        apiService.referenceHoldout(), apiService.referenceHoldoutRows()]);
      setState(surface);
      setRows(measured);
      setMessage('');
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const examine = async (target: string) => {
    if (artifact?.artifact === target) { setArtifact(null); return; }
    try { setArtifact(await apiService.referenceHoldoutArtifact(target)); }
    catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  };

  const writeReview = async () => {
    if (!assessment) return;
    setBusy(true);
    try {
      await apiService.writeHoldoutReview({
        boundary_assessment: assessment, basis,
        what_this_does_not_establish: limits.split('\n').map((v) => v.trim()).filter(Boolean),
        limitations, next_action: next, claim_boundary: boundary,
      });
      setMessage('Review written. It is not adopted, and the verdict is unchanged.');
      await load();
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  };

  const adopt = async () => {
    setBusy(true);
    setRefusal('');
    try {
      await apiService.adoptHoldoutReview({
        adopted_by: name, adopted_as: role, what_was_adopted: what,
        affirmation, why: why || undefined,
      });
      persist();
      setMessage('Review adopted and verified against the live result.');
      setAffirmation('');
      await load();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  };

  if (!state) return <p role="status" className="text-xs text-slate-500">{message}</p>;
  const counts = state.decision.counts ?? {};
  const request = state.review_request;

  return (
    <section className="border border-slate-800 bg-slate-900/50 p-5 space-y-4"
             aria-labelledby="holdout-title" data-testid="reference-holdout">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="holdout-title"
              className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Wind className="h-4 w-4 text-teal-400" aria-hidden="true" />
            Independent temporal holdout
          </h3>
          <p className="mt-1 text-[11px] text-slate-500">
            The 2022-2023 period, declared before it was opened, run once, and read by a person.
          </p>
        </div>
        <button type="button" onClick={() => void load()} title="Refresh holdout evidence"
                className="p-2 text-slate-300 hover:text-white">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div className="border border-slate-800 bg-slate-950 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-teal-300">{state.outcome}</div>
            <div className="mt-1 text-[11px] text-slate-500">
              {state.decision.population_denominator} selected storms · strict majority{' '}
              {state.decision.strict_majority_needed} · ties and refusals kept in the denominator
            </div>
          </div>
          <span data-testid="holdout-verdict"
                className="border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs
                           font-semibold uppercase tracking-wide text-amber-200">
            {state.verdict}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-px border border-slate-800 bg-slate-800 md:grid-cols-4">
          {Object.entries(CLASSIFICATION_LABEL).map(([key, label]) => (
            <div key={key} className="bg-slate-950 p-3">
              <div className="text-lg font-semibold text-slate-100">{counts[key] ?? 0}</div>
              <div className="text-[10px] uppercase text-slate-500">{label}</div>
            </div>
          ))}
        </div>

        {state.one_row_would_change_the_outcome && (
          <p data-testid="holdout-fragility"
             className="flex items-start gap-2 border border-amber-500/30 bg-amber-500/5 p-3
                        text-xs text-amber-100">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            This is the smallest majority the declared rule admits. One row changing side would
            have returned HOLDOUT_MIXED. That fragility is part of the result; it is not a reason
            to re-select, reweight or widen the population after the fact.
          </p>
        )}
      </div>

      <div className="space-y-px border border-slate-800 bg-slate-800">
        {state.stages.map((stage, index) => (
          <div key={stage.stage} className="bg-slate-950">
            <button type="button" onClick={() => setOpen(open === stage.stage ? null : stage.stage)}
                    aria-expanded={open === stage.stage}
                    className="flex w-full items-center gap-3 p-3 text-left hover:bg-slate-900">
              <span className="w-5 text-[10px] font-mono text-slate-600">{index + 1}</span>
              <span className="flex-1 text-xs text-slate-200">{stage.title}</span>
              <span className="text-[10px] uppercase text-slate-500">{stage.performed_by}</span>
              <span className="font-mono text-[10px] text-teal-300">{stage.status}</span>
            </button>
            {open === stage.stage && (
              <div className="border-t border-slate-800 p-3 space-y-2 text-xs">
                <pre className="overflow-x-auto bg-slate-900/60 p-2 text-[10px] text-slate-300">
                  {JSON.stringify(stage.facts, null, 2)}
                </pre>
                {stage.digest && (
                  <p className="break-all font-mono text-[10px] text-slate-500">{stage.digest}</p>
                )}
                <p className="text-slate-400">{stage.boundary}</p>
                {stage.artifact && (
                  <button type="button" onClick={() => void examine(stage.artifact as string)}
                          className="flex items-center gap-1.5 border border-slate-700 px-2 py-1
                                     text-[11px] text-slate-300 hover:text-white">
                    <Search className="h-3 w-3" aria-hidden="true" />
                    Examine the {stage.artifact} artifact
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {artifact && (
        <div className="border border-teal-500/30 bg-slate-950 p-3 space-y-2"
             data-testid="holdout-artifact">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-semibold text-teal-200">{artifact.path}</span>
            <button type="button" onClick={() => setArtifact(null)}
                    className="text-[11px] text-slate-400 hover:text-white">Close</button>
          </div>
          <p className="break-all font-mono text-[10px] text-slate-500">
            file sha256 {artifact.file_sha256}
          </p>
          <pre className="max-h-96 overflow-auto bg-slate-900/60 p-2 text-[10px] text-slate-300">
            {JSON.stringify(artifact.body, null, 2)}
          </pre>
        </div>
      )}

      {rows && (
        <div className="border border-slate-800">
          <div className="border-b border-slate-800 p-3 text-xs font-semibold text-slate-300">
            All {rows.rows.length} measured storms
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px]">
              <thead className="text-[10px] uppercase text-slate-500">
                <tr className="border-b border-slate-800">
                  <th className="p-2">Storm</th>
                  <th className="p-2">Time</th>
                  <th className="p-2">Catalogue to MSLP</th>
                  <th className="p-2">Catalogue to vorticity</th>
                  <th className="p-2">MSLP to vorticity</th>
                  <th className="p-2">Row</th>
                </tr>
              </thead>
              <tbody>
                {rows.rows.map((row: HoldoutRow) => {
                  const key = `${row.sid}-${row.time}`;
                  return (
                    <tr key={key} className="border-b border-slate-900 align-top">
                      <td className="p-2 text-slate-200">{row.storm}
                        <div className="font-mono text-[10px] text-slate-600">{row.sid}</div>
                      </td>
                      <td className="p-2 font-mono text-slate-400">{row.time}</td>
                      <td className="p-2 text-slate-300">{km(row.catalogue_to_mslp_km)}</td>
                      <td className="p-2 text-slate-300">{km(row.catalogue_to_vorticity_km)}</td>
                      <td className="p-2 text-slate-300">{km(row.mslp_to_vorticity_km)}</td>
                      <td className="p-2">
                        <span className={row.classification.startsWith('VERTICAL')
                          ? 'text-teal-300' : row.classification === 'REFUSED'
                            ? 'text-amber-300' : 'text-slate-300'}>
                          {CLASSIFICATION_LABEL[row.classification] ?? row.classification}
                        </span>
                        <button type="button"
                                onClick={() => setExpanded(expanded === key ? null : key)}
                                className="ml-2 text-[10px] text-slate-500 hover:text-white">
                          {expanded === key ? 'hide walks' : 'both walks'}
                        </button>
                        {expanded === key && (
                          <pre className="mt-2 max-h-64 overflow-auto bg-slate-900/60 p-2
                                          text-[10px] text-slate-300">
                            {JSON.stringify({
                              mslp_basin: row.mslp_basin,
                              negated_vorticity_basin: row.negated_vorticity_basin,
                            }, null, 2)}
                          </pre>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="border-t border-slate-800 p-3 text-[10px] text-slate-500">
            {Object.entries(rows.method).map(([key, value]) => `${key}: ${value}`).join(' · ')}
          </p>
        </div>
      )}

      <div className="border border-slate-800 p-3 text-xs space-y-2">
        <div className="font-semibold text-slate-300">Surface coverage</div>
        {state.operations.map((item) => (
          <div key={item.operation} className="flex items-start gap-2">
            {item.available_in_ui
              ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />
              : <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />}
            <span className="text-slate-400">
              {item.operation}{item.reason ? `: ${item.reason}` : ''}
            </span>
          </div>
        ))}
      </div>

      {state.review.status === 'NOT_WRITTEN' && (
        <div className="border border-amber-500/30 bg-amber-500/5 p-3 space-y-3"
             data-testid="holdout-review-form">
          <h4 className="text-sm font-semibold text-amber-100">Human result review</h4>
          <p className="text-[11px] text-amber-200/80">
            A review records a reading of this result and its stated boundary. It cannot accept
            the result: there is no acceptance to write, and the verdict stays {state.verdict}.
          </p>
          <div className="flex gap-2" role="group" aria-label="Boundary assessment">
            {request.human_inputs_required.boundary_assessment.map((value) => (
              <button key={value} type="button"
                      onClick={() => setAssessment(value as typeof assessment)}
                      className={`px-3 py-1.5 text-xs border ${assessment === value
                        ? 'border-teal-400 bg-teal-500/10 text-teal-200'
                        : 'border-slate-700 text-slate-400'}`}>{value}</button>
            ))}
          </div>
          <textarea aria-label="Scientific basis" value={basis} rows={3}
                    onChange={(event) => setBasis(event.target.value)}
                    placeholder="Your reading of this result, beyond its counts"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="What this does not establish" value={limits} rows={3}
                    onChange={(event) => setLimits(event.target.value)}
                    placeholder="One limit per line: what this result does not establish"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Limitations of this review" value={limitations} rows={2}
                    onChange={(event) => setLimitations(event.target.value)}
                    placeholder="The limits of this review itself"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Next action" value={next} rows={2}
                    onChange={(event) => setNext(event.target.value)}
                    placeholder="What you judge should happen next"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Claim boundary" value={boundary} rows={2}
                    onChange={(event) => setBoundary(event.target.value)}
                    placeholder="The claim boundary in your own words"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <button type="button" disabled={busy || !assessment} onClick={() => void writeReview()}
                  className="px-3 py-2 text-xs font-semibold bg-teal-700 text-white
                             disabled:opacity-40">
            Write review without adopting
          </button>
        </div>
      )}

      {state.review.status === 'WRITTEN_NOT_ADOPTED' && (
        <div className="border border-slate-700 p-3 space-y-3" data-testid="holdout-adopt-form">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <FileSignature className="h-4 w-4" /> Adopt written review
          </h4>
          <p className="break-all font-mono text-[10px] text-slate-500">
            {state.review.declaration_sha256}
          </p>
          <input aria-label="Reviewer name" value={name} placeholder="Full name"
                 onChange={(event) => setName(event.target.value)}
                 className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <input aria-label="Reviewer role" value={role} placeholder="Review role"
                 onChange={(event) => setRole(event.target.value)}
                 className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="What was adopted" value={what} rows={2}
                    onChange={(event) => setWhat(event.target.value)}
                    placeholder="What was adopted"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Why adopted" value={why} rows={2}
                    onChange={(event) => setWhy(event.target.value)}
                    placeholder="Why you adopted it (optional)"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <label className="flex items-start gap-2 text-[11px] text-slate-500">
            <input type="checkbox" aria-label="Remember signer identity" checked={remember}
                   onChange={(event) => setRemember(event.target.checked)}
                   className="mt-0.5" data-testid="remember-signer" />
            <span>{rememberSignerLabel()}</span>
          </label>
          <label className="block text-xs text-slate-400">
            Type: {request.human_inputs_required.affirmation}
            <input aria-label="Adoption affirmation" value={affirmation}
                   onChange={(event) => setAffirmation(event.target.value)}
                   className="mt-1 w-full border border-slate-700 bg-slate-950 p-2" />
          </label>
          <button type="button" disabled={busy} onClick={() => void adopt()}
                  className="px-3 py-2 text-xs font-semibold bg-slate-100 text-slate-950
                             disabled:opacity-40">
            Sign and verify review
          </button>
          {refusal && (
            <p role="alert" data-testid="holdout-adopt-refusal"
               className="border border-amber-500/40 bg-amber-500/5 p-2 text-[11px]
                          text-amber-100">
              {refusal}
            </p>
          )}
        </div>
      )}

      {state.review.status === 'ADOPTED' && (
        <div className="border border-emerald-500/30 bg-emerald-500/5 p-3 space-y-2 text-xs
                        text-emerald-100" data-testid="holdout-review-adopted">
          <div className="flex flex-wrap items-center gap-2">
            <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            <span className="font-semibold">
              {state.review.declaration?.boundary_assessment}
            </span>
            <span className="text-emerald-200/70">
              adopted by {state.review.adoption?.adopted_by} on{' '}
              {state.review.adoption?.adopted_on}
            </span>
          </div>
          <p className="text-emerald-200/80">
            {state.review.binds_current_result === false
              ? 'This review no longer binds the current result and must be re-read.'
              : `The review binds this exact result. The verdict remains ${state.verdict}.`}
          </p>
          <button type="button" onClick={() => void examine('review')}
                  className="flex items-center gap-1.5 border border-emerald-500/30 px-2 py-1
                             text-[11px] hover:text-white">
            <Search className="h-3 w-3" aria-hidden="true" /> Examine the review
          </button>
        </div>
      )}

      <p role="status" className="text-xs text-slate-400">{message}</p>
      <p className="text-[10px] text-slate-500">{state.claim_boundary}</p>
    </section>
  );
}
