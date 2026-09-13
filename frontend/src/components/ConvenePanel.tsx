/* T4E.33: convening the round-robin from the surface rather than from a terminal.
 *
 * The protocol, the eight seats, the dissent register and the transport were built at TG7.1 and
 * TG7.2 and a terminal was the only way to reach them. For a single-maintainer research
 * instrument that is a real barrier: the point of the surface is to run the experiment, not to
 * describe one that must then be run somewhere else.
 *
 * What does NOT move into the surface is the decision. The number of paid calls is stated before
 * any is made, the authorisation is a separate deliberate control rather than part of the run
 * button, and the key lives in the server's environment and is never a field here -- so it is
 * never in a browser, a log, or a request body.
 *
 * The panel is shown seat by seat with what each must return, because a reviewer who cannot see
 * what was asked of the panel cannot judge what the panel said.
 */
import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, KeyRound, Play, Users } from 'lucide-react';
import { apiService } from '../services/api';
import type { PanelPlan, RoundRobinRun } from '../types/api';

interface Props {
  studyId: string;
  onError?: (message: string) => void;
  onRan?: () => void;
}

export default function ConvenePanel({ studyId, onError, onRan }: Props) {
  const [plan, setPlan] = useState<PanelPlan | null>(null);
  const [model, setModel] = useState('');
  const [authorised, setAuthorised] = useState(false);
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState('');
  const [ran, setRan] = useState<RoundRobinRun | null>(null);

  useEffect(() => {
    apiService.panelPlan()
      .then((next) => { setPlan(next); setModel(next.default_model); })
      .catch((error: Error) => onError?.(error.message));
  }, [onError]);

  const convene = useCallback(async () => {
    setBusy(true);
    setRefusal('');
    try {
      const result = await apiService.conveneRoundRobin(studyId, {
        i_authorise_paid_calls: authorised, model_id: model,
      });
      setRan(result);
      onRan?.();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [studyId, authorised, model, onRan]);

  if (!plan) return null;

  return (
    <section data-testid="convene-panel"
             className="space-y-3 rounded-lg border border-slate-300 p-3 dark:border-slate-600">
      <h4 className="flex items-center gap-2 text-sm font-semibold">
        <Users className="h-4 w-4" aria-hidden="true" />
        Start a round-table review
      </h4>

      <ol className="space-y-1 text-xs text-slate-600 dark:text-slate-300">
        {plan.roles.map((seat, index) => (
          <li key={seat.role} data-testid={`seat-${seat.role}`} className="flex gap-2">
            <span className="w-4 text-right text-slate-400">{index + 1}</span>
            <span className="w-52 font-medium capitalize">{seat.role.split('_').join(' ')}</span>
            <span className="text-slate-500">reviews {seat.expects.join(', ').split('_').join(' ')}</span>
          </li>
        ))}
      </ol>

      <p data-testid="call-count" className="text-xs text-slate-600 dark:text-slate-300">
        <strong>{plan.calls_if_every_turn_is_taken} paid calls</strong> if every turn is taken,{' '}
        {plan.calls_if_nothing_is_dissented_from} if nothing is dissented from. {plan.why_that_differs}
      </p>
      <p className="text-xs text-slate-500 dark:text-slate-400">{plan.cost_is_the_caller_s}</p>
      <p className="text-xs text-slate-500 dark:text-slate-400">
        {plan.one_model_in_every_seat_is_recorded_not_refused}
      </p>

      <label className="block space-y-1 text-sm">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Review model
        </span>
        <input data-testid="panel-model" value={model}
               onChange={(event) => setModel(event.target.value)}
               className="w-72 rounded border border-slate-300 px-2 py-1 dark:border-slate-600
                          dark:bg-slate-900" />
      </label>

      <p data-testid="key-state"
         className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
        <KeyRound className="h-3 w-3" aria-hidden="true" />
        {plan.key_present
          ? `A key is present in the server environment (${plan.key_variables.join(' or ')}).`
          : `No key is present in ${plan.key_variables.join(' or ')} on the server. `
            + 'The run will refuse and send nothing. The key is never entered here.'}
      </p>

      <label className="flex items-start gap-2 text-xs">
        <input type="checkbox" data-testid="authorise" checked={authorised}
               onChange={(event) => setAuthorised(event.target.checked)} className="mt-0.5" />
        <span>
          I authorise up to {plan.calls_if_every_turn_is_taken} paid calls to an external service
          on my own account, and I understand the bundle's contents are sent to it.
        </span>
      </label>

      <button type="button" data-testid="convene" disabled={busy || !authorised}
              onClick={() => void convene()}
              className="flex items-center gap-2 rounded bg-slate-900 px-3 py-1.5 text-sm
                         text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100
                         dark:text-slate-900">
        <Play className="h-3 w-3" aria-hidden="true" />
        {busy ? 'Round table in progress…' : 'Start round table'}
      </button>

      {refusal && (
        <p data-testid="convene-refusal" role="alert"
           className="flex items-start gap-1 rounded border border-amber-400 bg-amber-50 p-2
                      text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40
                      dark:text-amber-100">
          <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
          <span>{refusal}</span>
        </p>
      )}

      {ran && (
        <div data-testid="convene-result" className="space-y-1 rounded bg-slate-50 p-2 text-xs
                                                     dark:bg-slate-900">
          <p><strong>Discussion saved</strong> · {ran.turns_taken.length} contributions · {ran.total_tokens} tokens</p>
          <p className="flex items-start gap-1 text-slate-600 dark:text-slate-300">
            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
            <span>{ran.the_rung_was_copied_not_set}</span>
          </p>
          <p className="text-slate-500 dark:text-slate-400">{ran.claim_boundary}</p>
        </div>
      )}
    </section>
  );
}
