import { FormEvent, useEffect, useState } from 'react';
import { AlertTriangle, MessageCircle, Save, Send, ShieldCheck, Trash2 } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  studyId: string;
  glossary: string;
  onError?: (message: string) => void;
}

export default function FindingDiscussion({ studyId, glossary, onError }: Props) {
  const [context, setContext] = useState<types.FindingConversationContext | null>(null);
  const [turns, setTurns] = useState<types.ConversationTurn[]>([]);
  const [question, setQuestion] = useState('');
  const [authorised, setAuthorised] = useState(false);
  const [saveConfirmed, setSaveConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<types.SavedFindingDiscussion | null>(null);
  const [lastAnswer, setLastAnswer] = useState<types.FindingConversationAnswer | null>(null);

  useEffect(() => {
    setTurns([]);
    setQuestion('');
    setAuthorised(false);
    setSaveConfirmed(false);
    setSaved(null);
    setLastAnswer(null);
    setContext(null);
    if (!studyId || !glossary) return;
    apiService.getFindingConversationContext(studyId, glossary)
      .then(setContext)
      .catch((error) => onError?.(error instanceof Error ? error.message : String(error)));
  }, [studyId, glossary, onError]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!context || !text || !authorised || busy) return;
    setBusy(true);
    setQuestion('');
    try {
      const result = await apiService.askAboutFinding(studyId, {
        question: text,
        history: turns,
        glossary,
        model_id: context.model_id,
        expected_bundle_sha256: context.bundle_sha256,
        i_authorise_paid_call: true,
      });
      setTurns((existing) => [...existing,
        { role: 'user', text }, { role: 'assistant', text: result.answer }]);
      setLastAnswer(result);
      setAuthorised(false);
      setSaved(null);
      setSaveConfirmed(false);
    } catch (error) {
      setQuestion(text);
      onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    setTurns([]);
    setLastAnswer(null);
    setSaved(null);
    setSaveConfirmed(false);
  }

  async function save() {
    if (!context || turns.length < 2 || !saveConfirmed || busy) return;
    setBusy(true);
    try {
      const result = await apiService.saveFindingDiscussion(studyId, {
        turns, glossary, expected_bundle_sha256: context.bundle_sha256,
      });
      setSaved(result);
      setSaveConfirmed(false);
    } catch (error) {
      onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  if (!context) return null;

  return (
    <section className="mt-8 rounded-xl border border-teal-500/25 bg-slate-950/45 p-4"
      aria-label="Discuss this finding">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="flex items-center gap-2 text-base font-semibold text-slate-100">
            <MessageCircle className="h-4 w-4 text-teal-300" aria-hidden="true" />
            Discuss this finding
          </h4>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-400">
            Ask about the final finding with its full evidence, experiment runs, and formal
            round-table results in view. Each answer is freshly grounded in the current record.
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30
                         bg-emerald-500/10 px-2.5 py-1 text-xs font-semibold text-emerald-200">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" /> Private for now · not recorded
        </span>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Loaded: {context.what_is_loaded.join(', ')}. {context.formal_review_count} formal review
        {context.formal_review_count === 1 ? '' : 's'} and {context.run_count} experiment run
        {context.run_count === 1 ? '' : 's'} match this study. {context.claim_boundary}
      </p>

      {turns.length > 0 && (
        <div className="mt-4 space-y-3" aria-live="polite">
          {turns.map((turn, index) => (
            <div key={`${turn.role}-${index}`} className={`rounded-lg border p-3 ${turn.role === 'user'
              ? 'ml-6 border-slate-700 bg-slate-800/70'
              : 'mr-6 border-teal-900/60 bg-teal-950/20'}`}>
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                {turn.role === 'user' ? 'You' : 'Finding discussion'}
              </p>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{turn.text}</p>
            </div>
          ))}
        </div>
      )}

      {lastAnswer && (lastAnswer.records_used.length > 0 || lastAnswer.cautions.length > 0) && (
        <details className="mt-3 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
          <summary className="cursor-pointer text-xs text-slate-300">Sources and cautions for the latest answer</summary>
          {lastAnswer.records_used.length > 0 && (
            <p className="mt-2 text-xs text-slate-400">Records used: {lastAnswer.records_used.join(', ')}</p>
          )}
          {lastAnswer.cautions.map((caution) => (
            <p key={caution} className="mt-2 flex gap-2 text-xs text-amber-200/80">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" /> {caution}
            </p>
          ))}
        </details>
      )}

      {lastAnswer && lastAnswer.suggested_questions.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {lastAnswer.suggested_questions.map((suggestion) => (
            <button key={suggestion} type="button" onClick={() => setQuestion(suggestion)}
              className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300
                         hover:border-teal-500/40 hover:text-teal-200">
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {!context.key_present && (
        <p className="mt-4 rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200">
          Discussion needs a configured Gemini API key. Nothing will be sent until one is present
          and you approve a call.
        </p>
      )}

      <form onSubmit={(event) => void ask(event)} className="mt-4">
        <label htmlFor="finding-question" className="text-xs font-medium text-slate-300">
          Your question
        </label>
        <textarea id="finding-question" value={question}
          onChange={(event) => setQuestion(event.target.value)} rows={3}
          placeholder="For example: Which limitation matters most when interpreting this result?"
          className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm
                     text-slate-100 placeholder:text-slate-600 focus:border-teal-500 focus:outline-none" />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-start gap-2 text-xs text-slate-400">
            <input type="checkbox" checked={authorised}
              onChange={(event) => setAuthorised(event.target.checked)} className="mt-0.5" />
            <span>I approve one paid external model call for this question. The answer stays unsaved.</span>
          </label>
          <button type="submit"
            disabled={!question.trim() || !authorised || !context.key_present || busy}
            className="inline-flex items-center gap-2 rounded-lg bg-teal-600 px-4 py-2 text-sm
                       font-semibold text-white hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-40">
            <Send className="h-4 w-4" aria-hidden="true" /> {busy ? 'Working…' : 'Ask about the finding'}
          </button>
        </div>
      </form>

      {turns.length >= 2 && (
        <div className="mt-5 border-t border-slate-800 pt-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <label className="flex max-w-2xl items-start gap-2 text-xs text-slate-400">
              <input type="checkbox" checked={saveConfirmed}
                onChange={(event) => setSaveConfirmed(event.target.checked)} className="mt-0.5" />
              <span>I want to add this transcript to the study record. It will be labelled as
                interpretation, not evidence.</span>
            </label>
            <div className="flex gap-2">
              <button type="button" onClick={clear}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2
                           text-xs text-slate-300 hover:bg-slate-800">
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" /> Clear local chat
              </button>
              <button type="button" onClick={() => void save()}
                disabled={!saveConfirmed || busy || Boolean(saved)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-violet-500/30
                           bg-violet-500/10 px-3 py-2 text-xs font-semibold text-violet-200
                           hover:bg-violet-500/20 disabled:cursor-not-allowed disabled:opacity-40">
                <Save className="h-3.5 w-3.5" aria-hidden="true" />
                {saved ? 'Saved to study record' : 'Save discussion'}
              </button>
            </div>
          </div>
          {saved && <p className="mt-2 text-xs text-violet-300">Saved as {saved.file}.</p>}
        </div>
      )}
    </section>
  );
}
