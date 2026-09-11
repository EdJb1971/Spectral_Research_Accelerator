/* T4E.34: composing a declaration, and committing it alone before anything is measured.
 *
 * T4E.30 surveyed every study here and found six that cannot be carried into an evidence bundle
 * -- and not one of them was refused for being declared after the fact. In every case the
 * declaration and the measurement entered git in the SAME commit. Those declarations were almost
 * certainly written first; git cannot separate them, so nothing downstream can check it.
 *
 * That is not a discipline problem. It is what happens when writing a declaration means opening
 * an editor in the middle of a working session: it gets saved with everything else. This form
 * exists to make the provable path the easy one, which is why the commit control is part of it
 * and not an afterthought.
 *
 * There is no template text here, no suggested prediction, no example claim boundary, and no
 * placeholder that could be mistaken for a starting point. Every word of the science is the
 * declarer's; what this surface contributes is structure and refusals. A prediction with no
 * stated falsifier is refused by the server, and the refusal is shown verbatim.
 *
 * Composing is not adopting. This writes a DRAFTED_NOT_ADOPTED declaration and nothing else; a
 * form that signed at the moment of drafting would make the signature meaningless.
 */
import { useCallback, useState } from 'react';
import { Ban, FilePlus2, GitCommitHorizontal, Plus, X } from 'lucide-react';
import { apiService } from '../services/api';
import type { ComposeGateEntry, ComposePrediction, ComposeResult } from '../types/api';

interface Props {
  /** A refusal from the server is a RESULT and renders inline, so this panel raises no error. */
  onComposed?: () => void;
}

const FIELDS: Array<{ key: string; label: string; hint: string; rows: number }> = [
  { key: 'artefact', label: 'Artefact — what is being made or measured, in one line',
    hint: '', rows: 1 },
  { key: 'why_this_exists', label: 'Why this exists',
    hint: 'A declaration that does not say why cannot be argued with later.', rows: 3 },
  { key: 'what_this_is_not', label: 'What this is NOT',
    hint: 'Stating this is how this programme has caught its own overreach.', rows: 3 },
  { key: 'the_inputs', label: 'The inputs, declared before the run',
    hint: 'Data, population, parameters. A run whose inputs were never declared cannot be '
        + 'reproduced or refused by name.', rows: 3 },
  { key: 'claim_boundary', label: 'Claim boundary — what this may not be used for',
    hint: 'No number here is recorded without one.', rows: 3 },
];

export default function DeclarationComposer({ onComposed }: Props) {
  const [text, setText] = useState<Record<string, string>>({});
  const [task, setTask] = useState('');
  const [declaredBy, setDeclaredBy] = useState('');
  const [gate, setGate] = useState<ComposeGateEntry[]>([
    { quantity: '', declared_value: '', tolerance: '' }]);
  const [predictions, setPredictions] = useState<ComposePrediction[]>([
    { name: '', statement: '', what_would_falsify_it: '' }]);
  const [commitAlone, setCommitAlone] = useState(true);
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState('');
  const [result, setResult] = useState<ComposeResult | null>(null);

  const submit = useCallback(async () => {
    setBusy(true);
    setRefusal('');
    setResult(null);
    try {
      const composed = await apiService.composeDeclaration({
        task, declared_by: declaredBy,
        artefact: text.artefact ?? '', why_this_exists: text.why_this_exists ?? '',
        what_this_is_not: text.what_this_is_not ?? '', the_inputs: text.the_inputs ?? '',
        claim_boundary: text.claim_boundary ?? '',
        gate, predictions, commit_it_alone: commitAlone,
      });
      setResult(composed);
      onComposed?.();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [task, declaredBy, text, gate, predictions, commitAlone, onComposed]);

  const field = (key: string, rows: number) => (
    rows === 1
      ? <input data-testid={`field-${key}`} value={text[key] ?? ''}
               onChange={(event) => setText({ ...text, [key]: event.target.value })}
               className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                          dark:border-slate-600 dark:bg-slate-900" />
      : <textarea data-testid={`field-${key}`} rows={rows} value={text[key] ?? ''}
                  onChange={(event) => setText({ ...text, [key]: event.target.value })}
                  className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                             dark:border-slate-600 dark:bg-slate-900" />
  );

  return (
    <section data-testid="declaration-composer"
             className="space-y-4 rounded-lg border border-slate-300 p-3 dark:border-slate-600">
      <header className="space-y-1">
        <h3 className="flex items-center gap-2 font-semibold">
          <FilePlus2 className="h-4 w-4" aria-hidden="true" />
          Declare a new study
        </h3>
        <p className="max-w-3xl text-xs text-slate-500 dark:text-slate-400">
          Nothing here is pre-filled. The structure and the refusals are the instrument's; every
          word of the science is yours. Composing writes a <strong>drafted, unsigned</strong>
          {' '}declaration — adopting it is a separate act.
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1">
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Task — e.g. T4E.34
          </span>
          <input data-testid="task" value={task} onChange={(e) => setTask(e.target.value)}
                 className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                            dark:border-slate-600 dark:bg-slate-900" />
        </label>
        <label className="space-y-1">
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Declared by — your name, not a role
          </span>
          <input data-testid="declared-by" value={declaredBy}
                 onChange={(e) => setDeclaredBy(e.target.value)}
                 className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                            dark:border-slate-600 dark:bg-slate-900" />
        </label>
      </div>

      {FIELDS.map((entry) => (
        <label key={entry.key} className="block space-y-1">
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            {entry.label}
          </span>
          {field(entry.key, entry.rows)}
          {entry.hint && (
            <span className="block text-xs text-slate-500 dark:text-slate-400">{entry.hint}</span>
          )}
        </label>
      ))}

      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          The gate — the figures the run will be judged against
        </h4>
        {gate.map((entry, index) => (
          <div key={index} className="flex flex-wrap items-center gap-2">
            <input data-testid={`gate-quantity-${index}`} placeholder="quantity"
                   value={entry.quantity}
                   onChange={(e) => setGate(gate.map((g, i) =>
                     i === index ? { ...g, quantity: e.target.value } : g))}
                   className="w-56 rounded border border-slate-300 px-2 py-1 text-sm
                              dark:border-slate-600 dark:bg-slate-900" />
            <input data-testid={`gate-value-${index}`} placeholder="declared value"
                   value={entry.declared_value}
                   onChange={(e) => setGate(gate.map((g, i) =>
                     i === index ? { ...g, declared_value: e.target.value } : g))}
                   className="w-40 rounded border border-slate-300 px-2 py-1 text-sm
                              dark:border-slate-600 dark:bg-slate-900" />
            <input data-testid={`gate-tolerance-${index}`} placeholder="tolerance (optional)"
                   value={entry.tolerance ?? ''}
                   onChange={(e) => setGate(gate.map((g, i) =>
                     i === index ? { ...g, tolerance: e.target.value } : g))}
                   className="w-44 rounded border border-slate-300 px-2 py-1 text-sm
                              dark:border-slate-600 dark:bg-slate-900" />
            {gate.length > 1 && (
              <button type="button" aria-label={`Remove gate entry ${index + 1}`}
                      onClick={() => setGate(gate.filter((_, i) => i !== index))}
                      className="text-slate-500 hover:text-slate-900">
                <X className="h-3 w-3" aria-hidden="true" />
              </button>
            )}
          </div>
        ))}
        <button type="button" data-testid="add-gate"
                onClick={() => setGate([...gate,
                  { quantity: '', declared_value: '', tolerance: '' }])}
                className="flex items-center gap-1 text-xs underline">
          <Plus className="h-3 w-3" aria-hidden="true" /> another gate figure
        </button>
      </div>

      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          The predictions — each with what would show it wrong
        </h4>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          A prediction that cannot fail is not a prediction. T4E.27 declared an improvement that
          was arithmetically impossible before the run, and it read as a risk that had been taken.
        </p>
        {predictions.map((entry, index) => (
          <div key={index} className="space-y-1 rounded border border-slate-200 p-2
                                      dark:border-slate-700">
            <div className="flex items-center gap-2">
              <input data-testid={`prediction-name-${index}`} placeholder="name"
                     value={entry.name}
                     onChange={(e) => setPredictions(predictions.map((p, i) =>
                       i === index ? { ...p, name: e.target.value } : p))}
                     className="w-44 rounded border border-slate-300 px-2 py-1 text-sm
                                dark:border-slate-600 dark:bg-slate-900" />
              {predictions.length > 1 && (
                <button type="button" aria-label={`Remove prediction ${index + 1}`}
                        onClick={() => setPredictions(predictions.filter((_, i) => i !== index))}
                        className="text-slate-500 hover:text-slate-900">
                  <X className="h-3 w-3" aria-hidden="true" />
                </button>
              )}
            </div>
            <textarea data-testid={`prediction-statement-${index}`} rows={2}
                      placeholder="what you predict"
                      value={entry.statement}
                      onChange={(e) => setPredictions(predictions.map((p, i) =>
                        i === index ? { ...p, statement: e.target.value } : p))}
                      className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                                 dark:border-slate-600 dark:bg-slate-900" />
            <textarea data-testid={`prediction-falsifier-${index}`} rows={2}
                      placeholder="what result would show this wrong"
                      value={entry.what_would_falsify_it}
                      onChange={(e) => setPredictions(predictions.map((p, i) =>
                        i === index ? { ...p, what_would_falsify_it: e.target.value } : p))}
                      className="w-full rounded border border-slate-300 px-2 py-1 text-sm
                                 dark:border-slate-600 dark:bg-slate-900" />
          </div>
        ))}
        <button type="button" data-testid="add-prediction"
                onClick={() => setPredictions([...predictions,
                  { name: '', statement: '', what_would_falsify_it: '' }])}
                className="flex items-center gap-1 text-xs underline">
          <Plus className="h-3 w-3" aria-hidden="true" /> another prediction
        </button>
      </div>

      <label className="flex items-start gap-2 text-xs">
        <input type="checkbox" data-testid="commit-alone" checked={commitAlone}
               onChange={(event) => setCommitAlone(event.target.checked)} className="mt-0.5" />
        <span>
          <strong>Commit this declaration on its own, now.</strong> Six of this repository's
          sixteen studies cannot be carried into an evidence bundle because their declaration and
          their measurement landed in the same commit, so nothing can prove which came first.
          Committing alone, before the run, is what makes the ordering provable. Other staged or
          modified files are left untouched.
        </span>
      </label>

      <button type="button" data-testid="compose-submit" disabled={busy}
              onClick={() => void submit()}
              className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white hover:bg-slate-700
                         disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900">
        {busy ? 'Writing…' : 'Compose the declaration'}
      </button>

      {refusal && (
        <p data-testid="compose-refusal" role="status"
           className="flex items-start gap-1 rounded border border-amber-400 bg-amber-50 p-2
                      text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40
                      dark:text-amber-100">
          <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
          <span>{refusal}</span>
        </p>
      )}

      {result && (
        <div data-testid="compose-result" role="status"
             className="space-y-1 rounded border border-emerald-400 bg-emerald-50 p-2 text-xs
                        dark:border-emerald-700 dark:bg-emerald-950/40">
          <p><strong>{result.written}</strong> — {result.status}</p>
          <p>{result.composing_is_not_adopting}</p>
          {result.commit?.committed && (
            <p data-testid="compose-commit" className="flex items-start gap-1">
              <GitCommitHorizontal className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
              <span>
                committed {result.commit.commit?.slice(0, 7)} at {result.commit.committed_at},
                containing {result.commit.files_in_commit.join(', ')}. {result.commit.why_alone}
              </span>
            </p>
          )}
          {result.commit && !result.commit.committed && (
            <p data-testid="compose-commit-refusal">{result.commit.refusal}</p>
          )}
          {result.not_committed && (
            <p data-testid="compose-not-committed">{result.not_committed}</p>
          )}
        </div>
      )}
    </section>
  );
}
