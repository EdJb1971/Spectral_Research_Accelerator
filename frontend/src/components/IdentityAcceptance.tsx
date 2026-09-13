import { useCallback, useEffect, useState } from 'react';
import {
  CheckCircle2, CircleDashed, FileSignature, Gavel, RefreshCw, ShieldAlert, XCircle,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { IdentityAcceptanceState } from '../types/api';
import { rememberSignerLabel, useSignerIdentity } from './useSignerIdentity';

/** The exact words the adoption machinery requires. Shown so a signer can read what they are
 *  being asked to type; never pre-filled, because a pre-filled affirmation is the instrument
 *  affirming on a person's behalf. */
const REQUIRED_AFFIRMATION = 'I have read this declaration and I adopt it';

const STATUS_ICON: Record<string, JSX.Element> = {
  CONDITION_MET: <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />,
  CONDITION_NOT_MET: <XCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />,
  EVIDENCE_CLAIMED_NOT_ADOPTED: <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-amber-400" />,
  NO_EVIDENCE: <CircleDashed className="h-3.5 w-3.5 shrink-0 text-slate-500" />,
};

/** T4E.41: what a passing identity criterion must establish, and what would license it.
 *
 *  The panel is deliberately not a progress bar. `NO_EVIDENCE` is drawn at the same weight as
 *  `CONDITION_MET`, because a condition nothing has been addressed to is not a condition
 *  half-passed, and the declaration says conditions do not average. */
export default function IdentityAcceptance() {
  const [state, setState] = useState<IdentityAcceptanceState | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [message, setMessage] = useState('Loading the identity acceptance...');
  // A refusal is a result and must survive whatever else the panel does next. `message` is
  // cleared by every successful reload, so a refusal kept there can be wiped by a refresh
  // that happens to land after it -- which is how a signing refusal reached the DOM for two
  // renders and then vanished. Refusals live in their own state, which nothing else clears.
  const [refusal, setRefusal] = useState('');
  const { name, setName, role, setRole, remember, setRemember, persist } = useSignerIdentity();
  const [why, setWhy] = useState('');
  const [affirmation, setAffirmation] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setState(await apiService.identityAcceptance()); setMessage(''); }
    catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const adopt = async () => {
    if (!state) return;
    setBusy(true);
    setRefusal('');
    try {
      await apiService.signDeclaration({
        declaration: state.declaration_file,
        adopted_by: name,
        adopted_as: role,
        what_was_adopted: "the T4E.41 acceptance bar for T4E.8's identity criterion",
        affirmation,
        why: why || undefined,
      });
      persist();
      setMessage('The bar is adopted. No condition is met and T4E.8 is not accepted.');
      setAffirmation('');
      await load();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  };

  if (!state) return <p role="status" className="text-xs text-slate-500">{message}</p>;
  const bound = state.bound_evidence;

  return (
    <section className="border border-slate-800 bg-slate-900/50 p-5 space-y-4"
             aria-labelledby="acceptance-title" data-testid="identity-acceptance">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="acceptance-title"
              className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Gavel className="h-4 w-4 text-teal-400" aria-hidden="true" />
            T4E.8 acceptance
          </h3>
          <p className="mt-1 text-[11px] text-slate-500">
            What a passing identity criterion must establish for{' '}
            <span className="font-mono">{state.identity_target}</span>, and what would license it.
          </p>
        </div>
        <button type="button" onClick={() => void load()} title="Refresh acceptance state"
                className="p-2 text-slate-300 hover:text-white">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      <div className="border border-slate-800 bg-slate-950 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div data-testid="acceptance-verdict" className="text-sm font-semibold text-amber-200">
              {state.VERDICT}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {state.conditions_met} of {state.conditions_total} conditions met · the bar is{' '}
              {state.acceptance_adopted ? 'adopted' : state.declaration_status}
            </div>
          </div>
          <span className="border border-slate-700 px-3 py-1.5 text-[10px] uppercase
                           tracking-wide text-slate-400">
            code may accept: {String(state.code_may_emit_accepted)}
          </span>
        </div>
        <p className="text-[11px] text-slate-500">
          Conditions do not average. Software computes each one and stops; the acceptance itself
          is a person&apos;s act.
        </p>
      </div>

      <div className="space-y-px border border-slate-800 bg-slate-800">
        {state.conditions.map((condition) => (
          <div key={condition.id} className="bg-slate-950">
            <button type="button" aria-expanded={open === condition.id}
                    onClick={() => setOpen(open === condition.id ? null : condition.id)}
                    className="flex w-full items-center gap-3 p-3 text-left hover:bg-slate-900">
              <span className="w-6 font-mono text-[10px] text-slate-600">{condition.id}</span>
              {STATUS_ICON[condition.status] ?? STATUS_ICON.NO_EVIDENCE}
              <span className="flex-1 text-xs text-slate-200">{condition.name}</span>
              <span className="font-mono text-[10px] text-slate-500">{condition.status}</span>
            </button>
            {open === condition.id && (
              <div className="border-t border-slate-800 p-3 space-y-3 text-xs">
                <div>
                  <div className="text-[10px] uppercase text-slate-500">Requires</div>
                  <p className="text-slate-300">{condition.requires}</p>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-slate-500">Licensed by</div>
                  <p className="text-slate-300">{condition.licensed_by}</p>
                </div>
                <div className="border-l-2 border-amber-500/40 pl-3">
                  <div className="text-[10px] uppercase text-amber-400/80">Not sufficient</div>
                  <p className="text-amber-100/80">{condition.insufficient}</p>
                </div>
                {condition.why && <p className="text-slate-400">{condition.why}</p>}
                {condition.evidence.length > 0 && (
                  <pre className="overflow-x-auto bg-slate-900/60 p-2 text-[10px] text-slate-300">
                    {JSON.stringify(condition.evidence, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="border border-slate-800 p-3 text-xs space-y-2">
          <div className="font-semibold text-slate-300">
            The record this bar was set against
          </div>
          <p className="text-[11px] text-slate-500">
            {bound.checked} artefacts, all verified: {String(bound.all_verified)}
          </p>
          <ul className="space-y-1 text-[11px]">
            {bound.checks.map((check) => (
              <li key={check.artefact} className="flex items-start gap-2">
                {check.status === 'VERIFIED'
                  ? <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-emerald-400" />
                  : <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-rose-400" />}
                <span className="break-all font-mono text-slate-400">{check.artefact}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="border border-slate-800 p-3 text-xs space-y-2"
             data-testid="acceptance-exclusions">
          <div className="font-semibold text-slate-300">What does not discharge this</div>
          <ul className="space-y-1 text-[11px] text-slate-400">
            {state.what_does_not_discharge_this.map((item) => (
              <li key={item} className="flex items-start gap-2">
                <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-slate-600" aria-hidden="true" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <details className="border border-slate-800 p-3 text-xs" data-testid="acceptance-full-text">
        <summary className="cursor-pointer font-semibold text-slate-300">
          Read the declaration in full before signing it
        </summary>
        <div className="mt-3 space-y-3 text-[11px] text-slate-400">
          <p className="break-all font-mono text-[10px] text-slate-500">
            {state.declaration} · {state.declaration_sha256}
          </p>
          <div>
            <div className="text-[10px] uppercase text-slate-500">Why this exists</div>
            {Object.entries(state.why_this_exists).map(([key, value]) => (
              <p key={key} className="mt-1"><span className="text-slate-500">{key}: </span>{value}</p>
            ))}
          </div>
          <div>
            <div className="text-[10px] uppercase text-slate-500">What this declaration is not</div>
            <ul className="mt-1 space-y-1">
              {state.what_this_declaration_is_not.map((item) => <li key={item}>· {item}</li>)}
            </ul>
          </div>
          <div>
            <div className="text-[10px] uppercase text-slate-500">Target and scope</div>
            {Object.entries(state.target_and_scope).map(([key, value]) => (
              <p key={key} className="mt-1">
                <span className="text-slate-500">{key}: </span>{String(value)}
              </p>
            ))}
          </div>
          <div>
            <div className="text-[10px] uppercase text-slate-500">
              What acceptance would and would not license
            </div>
            {Object.entries(state.what_acceptance_would_and_would_not_license).map(
              ([key, value]) => (
                <p key={key} className="mt-1"><span className="text-slate-500">{key}: </span>{value}</p>
              ))}
          </div>
          <div>
            <div className="text-[10px] uppercase text-slate-500">Verdict semantics</div>
            {Object.entries(state.verdict_semantics ?? {}).map(([key, value]) => (
              <p key={key} className="mt-1">
                <span className="font-mono text-slate-500">{key}: </span>{String(value)}
              </p>
            ))}
          </div>
        </div>
      </details>

      {state.acceptance_adopted ? (
        <p data-testid="acceptance-adopted"
           className="border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-emerald-100">
          The bar is adopted by {state.adoption?.adopted_by} on {state.adoption?.adopted_on}.
          It fixes the standard the next candidate is measured against. It accepts nothing:
          the verdict is still {state.VERDICT} at {state.conditions_met} of{' '}
          {state.conditions_total}.
        </p>
      ) : (
        <div className="border border-slate-700 p-3 space-y-3" data-testid="acceptance-adopt-form">
          <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <FileSignature className="h-4 w-4" aria-hidden="true" /> Adopt the bar
          </h4>
          <p className="text-[11px] text-slate-500">
            Adopting fixes the standard for the next candidate. It does not accept T4E.8, license{' '}
            <span className="font-mono">{state.identity_target}</span>, meet a condition or reopen
            any spent population.
          </p>
          <input aria-label="Adopter name" value={name} placeholder="Full name"
                 onChange={(event) => setName(event.target.value)}
                 className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <input aria-label="Adopter role" value={role} placeholder="Role in this decision"
                 onChange={(event) => setRole(event.target.value)}
                 className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <textarea aria-label="Why adopted" value={why} rows={2}
                    onChange={(event) => setWhy(event.target.value)}
                    placeholder="Why you are fixing this bar (optional)"
                    className="w-full border border-slate-700 bg-slate-950 p-2 text-xs" />
          <label className="flex items-start gap-2 text-[11px] text-slate-500">
            <input type="checkbox" aria-label="Remember signer identity" checked={remember}
                   onChange={(event) => setRemember(event.target.checked)}
                   className="mt-0.5" data-testid="remember-signer" />
            <span>{rememberSignerLabel()}</span>
          </label>
          <label className="block text-xs text-slate-400">
            Type: {REQUIRED_AFFIRMATION}
            <input aria-label="Adoption affirmation" value={affirmation}
                   onChange={(event) => setAffirmation(event.target.value)}
                   className="mt-1 w-full border border-slate-700 bg-slate-950 p-2" />
          </label>
          <button type="button" disabled={busy} onClick={() => void adopt()}
                  className="px-3 py-2 text-xs font-semibold bg-slate-100 text-slate-950
                             disabled:opacity-40">
            Sign the acceptance bar
          </button>
          {refusal && (
            <p role="alert" data-testid="acceptance-refusal"
               className="border border-amber-500/40 bg-amber-500/5 p-2 text-[11px]
                          text-amber-100">
              {refusal}
            </p>
          )}
        </div>
      )}

      <p role="status" className="text-xs text-slate-400">{message}</p>
      <p className="text-[10px] text-slate-500">{state.claim_boundary}</p>
    </section>
  );
}
