/** TG11.3 — the evidence write path: record what was observed, and read back a rung nobody typed.
 *
 * This is the first panel in the application that writes anything bearing on a claim, so it is
 * the first one that could break rule R22. It does not have a control for the rung, and that is
 * not a UI decision: the request shapes have no field for one, the server recomputes the ladder
 * from the chain on every response, and the two things this panel displays as a verdict — the
 * rung and the precedence entry — are both computed elsewhere and rendered here whole.
 *
 * The append form deliberately keeps the head digest visible. An append states the revision it
 * extends, so a chain that moved under the researcher's editor is a refusal rather than a silent
 * overwrite, and showing the digest is what makes that refusal comprehensible when it happens.
 */

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, FilePlus2, Loader2, ShieldCheck } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  selectedRecord: types.ChannelRecordSelection | null;
  studyId: string;
  onStudyId: (value: string) => void;
  onError?: (message: string) => void;
}

const SHORT = (digest?: string) => (digest ? `${digest.slice(0, 12)}…` : '—');

const EvidenceView: React.FC<Props> = ({ selectedRecord, studyId, onStudyId, onError }) => {
  const [capabilities, setCapabilities] = useState<types.EvidenceCapabilities | null>(null);
  const [state, setState] = useState<types.EvidenceState | null>(null);
  const [wording, setWording] = useState<string | null>(null);

  const [hypothesisId, setHypothesisId] = useState('H1');
  const [statement, setStatement] = useState('');
  const [prediction, setPrediction] = useState('');

  const [category, setCategory] = useState<types.EvidenceCategory>('observations');
  const [status, setStatus] = useState<types.EvidenceStatus>('PASS');
  const [label, setLabel] = useState('');
  const [summary, setSummary] = useState('');
  const [payload, setPayload] = useState('{\n  "value": 0.0\n}');

  const [lags, setLags] = useState('2');
  const [surrogates, setSurrogates] = useState(499);
  const [busy, setBusy] = useState<'' | 'open' | 'append' | 'precedence' | 'head'>('');

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  useEffect(() => {
    apiService.getEvidenceCapabilities().then(setCapabilities).catch(fail);
  }, [fail]);

  const readHead = useCallback(async (id: string) => {
    if (!id) { setState(null); return; }
    setBusy('head');
    try {
      setState(await apiService.getEvidenceHead(id));
    } catch {
      // Not an error to report: a study that has not been opened yet simply has no chain.
      setState(null);
    } finally { setBusy(''); }
  }, []);

  useEffect(() => { readHead(studyId); }, [studyId, readHead]);

  const open = async () => {
    setBusy('open');
    setWording(null);
    try {
      const opened = await apiService.openStudy(studyId, hypothesisId, statement, prediction);
      setState(opened);
      setWording(opened.wording_note ?? null);
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  const append = async () => {
    if (!state) return;
    setBusy('append');
    try {
      const parsed = JSON.parse(payload);
      setState(await apiService.appendEvidence(state.study_id, {
        expected_head_sha256: state.head_sha256,
        category, label, status, summary, payload: parsed
      }));
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  const precedence = async () => {
    if (!state || !selectedRecord) return;
    setBusy('precedence');
    try {
      setState(await apiService.appendPrecedence(
        state.study_id, state.head_sha256,
        label || 'precedence over the selected record',
        selectedRecord, lags, surrogates));
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  const ladder = state?.ladder;
  const entries = state?.entries ?? (state?.entry ? [state.entry] : []);

  return (
    <section className="space-y-5 animate-fadeIn" aria-labelledby="evidence-title" aria-busy={busy !== ''}>
      <header>
        <h2 id="evidence-title" className="text-xl font-bold text-white flex items-center gap-2">
          <FilePlus2 className="text-teal-400 w-5 h-5" aria-hidden="true" /> Evidence record
        </h2>
        <p className="text-sm text-slate-400 max-w-4xl mt-1">
          Append what was observed to a hypothesis registered first. The rung is not something
          recorded here: it is recomputed by the claim ladder from the whole chain on every
          response, and no request this panel can send carries one (R22).
        </p>
        {capabilities && (
          <p className="text-xs text-slate-500 max-w-4xl mt-2">{capabilities.claim_boundary}</p>
        )}
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        {/* -------------------------------------------------------------- the hypothesis */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Register a hypothesis</p>
          <label className="block text-xs text-slate-400">Study identifier
            <input value={studyId} onChange={(e) => onStudyId(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <label className="block text-xs text-slate-400">Hypothesis identifier
            <input value={hypothesisId} onChange={(e) => setHypothesisId(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <label className="block text-xs text-slate-400">Statement
            <textarea value={statement} onChange={(e) => setStatement(e.target.value)} rows={2}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <label className="block text-xs text-slate-400">Prediction
            <textarea value={prediction} onChange={(e) => setPrediction(e.target.value)} rows={2}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <button type="button" onClick={open} disabled={busy !== '' || !studyId}
            className="w-full px-3 py-2 rounded bg-teal-500/20 text-teal-100 text-sm hover:bg-teal-500/30 disabled:opacity-40">
            {busy === 'open' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Open at revision zero
          </button>
          {wording && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded p-3">
              <p className="text-xs text-amber-200">{wording}</p>
            </div>
          )}
          <p className="text-[11px] text-slate-500">
            A hypothesis is registered before any evidence exists to favour it. Opening a study
            twice under one identifier is refused by the server.
          </p>
        </div>

        {/* ------------------------------------------------------------------ the append */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Record an observation</p>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Category
              <select value={category} onChange={(e) => setCategory(e.target.value as types.EvidenceCategory)}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200">
                {(capabilities?.categories ?? []).map((name) => (
                  <option key={name} value={name}>{name.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </label>
            <label className="text-xs text-slate-400">Status
              <select value={status} onChange={(e) => setStatus(e.target.value as types.EvidenceStatus)}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200">
                {(capabilities?.statuses ?? []).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="block text-xs text-slate-400">Label
            <input value={label} onChange={(e) => setLabel(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <label className="block text-xs text-slate-400">Summary
            <input value={summary} onChange={(e) => setSummary(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>
          <label className="block text-xs text-slate-400">Payload (JSON)
            <textarea value={payload} onChange={(e) => setPayload(e.target.value)} rows={4}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 font-mono text-xs text-slate-200" />
          </label>
          <p className="text-[11px] text-slate-500">
            Extending revision {state ? state.revision : '—'}, head <span className="font-mono">{SHORT(state?.head_sha256)}</span>.
          </p>
          <button type="button" onClick={append} disabled={busy !== '' || !state}
            className="w-full px-3 py-2 rounded bg-teal-500/20 text-teal-100 text-sm hover:bg-teal-500/30 disabled:opacity-40">
            {busy === 'append' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Append entry
          </button>

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Computed, not asserted</p>
            <p className="text-[11px] text-slate-500">
              {capabilities?.computed_not_accepted?.temporal_precedence
                ?? 'Temporal precedence is computed by the server.'}
            </p>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Lags in frames
                <input value={lags} onChange={(e) => setLags(e.target.value)}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
              </label>
              <label className="text-xs text-slate-400">Surrogates
                <input type="number" min={1} value={surrogates}
                  onChange={(e) => setSurrogates(Number(e.target.value))}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
              </label>
            </div>
            <button type="button" onClick={precedence}
              disabled={busy !== '' || !state || !selectedRecord}
              className="w-full px-3 py-2 rounded bg-slate-800 text-slate-200 text-sm hover:bg-slate-700 disabled:opacity-40">
              {busy === 'precedence' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Run precedence and record the answer
            </button>
            {!selectedRecord && (
              <p className="text-[11px] text-amber-300/70">
                Select and admit a channel record to run this; the analysis is computed over the
                record itself, not over a summary of it.
              </p>
            )}
          </div>
        </div>

        {/* ------------------------------------------------------------------ the ladder */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Where the chain stands</p>
          {!ladder && (
            <p className="text-sm text-slate-400">
              No chain read yet. Open a study, or type the identifier of one that exists.
            </p>
          )}
          {ladder && (
            <>
              <div>
                <p className="text-2xl font-bold text-white">{ladder.rung.replace(/_/g, ' ')}</p>
                <p className="text-xs text-slate-500">
                  revision {ladder.revision} · bundle <span className="font-mono">{SHORT(ladder.bundle_sha256)}</span>
                </p>
              </div>
              {state?.blocked ? (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded p-3 flex gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-300 shrink-0" aria-hidden="true" />
                  <p className="text-xs text-amber-200">
                    Capped below {ladder.unblocked_rung.replace(/_/g, ' ')}
                    {ladder.blocking_entries.length
                      ? ` by entr${ladder.blocking_entries.length === 1 ? 'y' : 'ies'} ${ladder.blocking_entries.join(', ')}`
                      : ''}. Negative evidence is load-bearing: no quantity of favourable
                    entries outvotes it.
                  </p>
                </div>
              ) : (
                <div className="bg-teal-500/10 border border-teal-500/20 rounded p-3 flex gap-2">
                  <CheckCircle2 className="w-4 h-4 text-teal-300 shrink-0" aria-hidden="true" />
                  <p className="text-xs text-teal-100">Nothing in the chain caps this bundle.</p>
                </div>
              )}
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Gates</p>
                <ul className="space-y-1">
                  {ladder.gates.map((gate) => (
                    <li key={gate.name} className="text-[11px] flex gap-2">
                      <span className={gate.satisfied ? 'text-teal-400' : 'text-slate-600'} aria-hidden="true">
                        {gate.satisfied ? '●' : '○'}
                      </span>
                      <span className={gate.satisfied ? 'text-slate-300' : 'text-slate-500'}>
                        {gate.requirement}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <p className="text-[11px] text-slate-500 flex gap-2">
                <ShieldCheck className="w-3 h-3 mt-0.5 shrink-0" aria-hidden="true" />
                {state?.rung_source}
              </p>
            </>
          )}
        </div>
      </div>

      {entries.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 overflow-x-auto">
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-3">The chain</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 text-left">
                <th className="pb-2 pr-3">#</th>
                <th className="pb-2 pr-3">Category</th>
                <th className="pb-2 pr-3">Status</th>
                <th className="pb-2 pr-3">Label</th>
                <th className="pb-2 pr-3">Recorded</th>
                <th className="pb-2">Entry digest</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.entry_sha256} className="border-t border-slate-800/70">
                  <td className="py-2 pr-3 text-slate-400">{entry.sequence}</td>
                  <td className="py-2 pr-3 text-slate-300">{entry.category.replace(/_/g, ' ')}</td>
                  <td className="py-2 pr-3 text-slate-300">{entry.status}</td>
                  <td className="py-2 pr-3 text-slate-300">{entry.label}</td>
                  <td className="py-2 pr-3 text-slate-500">{entry.recorded_at}</td>
                  <td className="py-2 font-mono text-slate-500">{SHORT(entry.entry_sha256)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[11px] text-slate-500 mt-3">
            Each entry carries the digest of the one before it, so the chain is append-only in a
            way that an edit would show rather than merely be discouraged from.
          </p>
        </div>
      )}
    </section>
  );
};

export default EvidenceView;
