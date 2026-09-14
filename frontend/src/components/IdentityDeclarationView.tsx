import { useEffect, useState } from 'react';
import { AlertTriangle, Ban, FileQuestion, HelpCircle, ShieldOff, Target } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

/** T4E.8 slice 4: what identity is meant to recognise, made visible before it is chosen.
 *
 * Slice 3 turned the identity target into a declared, refusable input. It is the most
 * consequential scientific choice in the atmospheric sequence -- continuity of a tracked
 * constellation, persistence of a spatial configuration, and recurrence of the same physical
 * kind are three different questions, and answering the third with labels drawn from the same
 * pipeline validates a definition against itself. Until this panel that choice was made by
 * hand-editing a JSON design, which is not a choice a researcher can make knowingly.
 *
 * Three rules govern what follows.
 *
 * *A refusal is a result, not an error state.* The inadmissible pairing is rendered at the same
 * weight as the admissible ones, with the server's own refusal text. Showing only the
 * combinations that work would hide the single cell a researcher most needs to understand.
 *
 * *An admitted pairing may still owe a warning.* `track_continuity` on tracked keys is admitted
 * and carries a caveat, because a high score there demonstrates agreement with the tracker
 * rather than independent identity. The caveat renders beside the admission, not beneath it.
 *
 * *This panel offers no way to choose.* The server serves no route that would answer one, and
 * choosing is a scientific act. The refusals are rendered from the server's list rather than
 * implied by an absence of buttons, so a reader who wonders where the control went is told.
 */
export default function IdentityDeclarationView({ onError }: { onError?: (m: string) => void }) {
  const [targets, setTargets] = useState<types.IdentityTargets | null>(null);
  const [audits, setAudits] = useState<types.IdentityAuditIndex | null>(null);
  const [open, setOpen] = useState<types.IdentityAuditView | null>(null);
  const [status, setStatus] = useState('Loading the identity declaration…');

  useEffect(() => {
    Promise.all([apiService.identityTargets(), apiService.listIdentityAudits()])
      .then(([targetSurface, auditIndex]) => {
        setTargets(targetSurface);
        setAudits(auditIndex);
        setStatus('');
      })
      .catch((error: Error) => { setStatus(error.message); onError?.(error.message); });
  }, [onError]);

  if (status) {
    return <div className="p-4 text-sm text-slate-600 dark:text-slate-300">{status}</div>;
  }
  if (!targets || !audits) return null;

  return (
    <div className="p-4 space-y-6" data-testid="identity-declaration">
      <header className="space-y-1">
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <Target className="h-5 w-5" aria-hidden="true" />
          What identity is meant to recognise
        </h2>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          {targets.choosing_is_not_automated}
        </p>
      </header>

      <section aria-labelledby="admissibility" className="space-y-3">
        <h3 id="admissibility" className="text-sm font-semibold uppercase tracking-wide">
          Targets, and the evidence that may validate each
        </h3>
        <ul className="space-y-3">
          {targets.targets.map((row) => (
            <li key={row.identity_target}
                className="rounded border border-slate-300 dark:border-slate-700 p-3 space-y-2">
              <div className="font-mono text-sm font-semibold">{row.identity_target}</div>
              <p className="text-sm">{row.recognises}</p>
              <p className="text-xs text-slate-600 dark:text-slate-300">
                <span className="font-semibold">Does not license: </span>{row.does_not_license}
              </p>
              <ul className="space-y-2">
                {row.evidence.map((cell) => (
                  <li key={cell.evidence_class}
                      data-testid={`cell-${row.identity_target}-${cell.evidence_class}`}
                      className={`rounded p-2 text-xs border ${cell.admitted
                        ? 'border-slate-300 dark:border-slate-700'
                        : 'border-amber-600 bg-amber-50 dark:bg-amber-950/40'}`}>
                    <div className="flex items-center gap-2 font-mono">
                      {cell.admitted
                        ? <span data-testid="admitted">admitted</span>
                        : <><Ban className="h-3.5 w-3.5" aria-hidden="true" />
                            <span data-testid="refused">refused</span></>}
                      <span>{cell.evidence_class}</span>
                      {!cell.independent_of_record && (
                        <span className="text-slate-500">(not independent of the record)</span>
                      )}
                    </div>
                    {cell.refusal && (
                      <p className="mt-1 text-amber-800 dark:text-amber-300">{cell.refusal}</p>
                    )}
                    {cell.caveat && (
                      <p className="mt-1 flex gap-1 text-slate-700 dark:text-slate-300">
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>{cell.caveat}</span>
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="audits" className="space-y-3">
        <h3 id="audits" className="text-sm font-semibold uppercase tracking-wide">
          Identity-calibration receipts
        </h3>
        {audits.audits.length === 0 ? (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            No identity audit has been run. This is an absence of runs, not an absence of findings.
          </p>
        ) : (
          <ul className="space-y-2">
            {audits.audits.map((audit) => (
              <li key={audit.file} data-testid={`audit-${audit.file}`}
                  className="rounded border border-slate-300 dark:border-slate-700 p-3 text-xs space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <button type="button" className="font-mono font-semibold underline"
                          data-testid={`open-${audit.file}`}
                          onClick={() => apiService.identityAudit(audit.file)
                            .then(setOpen)
                            .catch((error: Error) => { setStatus(error.message); onError?.(error.message); })}>
                    {audit.file}
                  </button>
                  {audit.status && <span className="font-mono">{audit.status}</span>}
                  <span data-testid="approved-radius" className="font-mono">
                    {audit.approved_mining_radius === null
                      ? 'no approved mining radius'
                      : `approved radius ${audit.approved_mining_radius}`}
                  </span>
                  {audit.code_dirty && (
                    <span className="text-amber-700 dark:text-amber-300">
                      measured from an uncommitted tree
                    </span>
                  )}
                </div>
                {audit.identity_declaration ? (
                  <div className="space-y-1">
                    <div className="font-mono">
                      target {audit.identity_declaration.identity_target} via{' '}
                      {audit.identity_declaration.evidence_class}
                    </div>
                    {audit.identity_declaration.caveat && (
                      <p className="flex gap-1">
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                        <span>{audit.identity_declaration.caveat}</span>
                      </p>
                    )}
                    <p className="text-slate-600 dark:text-slate-300">
                      <span className="font-semibold">Labels: </span>
                      {audit.identity_declaration.label_boundary}
                    </p>
                  </div>
                ) : (
                  <p data-testid="undeclared"
                     className="flex gap-1 text-slate-700 dark:text-slate-300">
                    <FileQuestion className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    <span>No identity target was declared for this earlier study.</span>
                  </p>
                )}
                {audit.claim_boundary && (
                  <p className="text-slate-600 dark:text-slate-300">
                    <span className="font-semibold">Claim boundary: </span>{audit.claim_boundary}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
        {audits.audits_without_a_declared_target.length > 0 && (
          <p className="flex gap-1 text-xs text-slate-600 dark:text-slate-300">
            <HelpCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>{audits.undeclared_note}</span>
          </p>
        )}
        {audits.unreadable.length > 0 && (
          <ul className="text-xs text-amber-700 dark:text-amber-300">
            {audits.unreadable.map((item) => (
              <li key={item.file}>{item.file}: {item.error}</li>
            ))}
          </ul>
        )}
      </section>

      {open && (
        <section aria-labelledby="open-audit" className="space-y-2">
          <h3 id="open-audit" className="text-sm font-semibold uppercase tracking-wide">
            {open.summary.file}, in full
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-300">
            The whole receipt, not a summary of it. A receipt read in fragments can lose the
            boundary that governs its numbers.
          </p>
          <pre data-testid="open-audit-body"
               className="max-h-96 overflow-auto rounded bg-slate-100 dark:bg-slate-900 p-3 text-xs">
            {JSON.stringify(open.audit, null, 2)}
          </pre>
          <button type="button" className="text-xs underline" onClick={() => setOpen(null)}>
            Close
          </button>
        </section>
      )}

      <section aria-labelledby="refusals" className="space-y-2">
        <h3 id="refusals" className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide">
          <ShieldOff className="h-4 w-4" aria-hidden="true" />
          What this surface will not do
        </h3>
        <ul className="list-disc pl-5 text-xs text-slate-600 dark:text-slate-300 space-y-1">
          {targets.refusals.map((refusal) => <li key={refusal}>{refusal}</li>)}
        </ul>
      </section>
    </div>
  );
}
