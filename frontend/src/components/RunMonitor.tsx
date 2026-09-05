import * as types from '../types/api';

/**
 * The orchestrated run, watched from the browser (TG17.6).
 *
 * The reason this panel exists is the moment a long remote job is interrupted. What an operator
 * does in that moment decides whether the study that finishes is the study that was declared, and
 * the tempting action — start again, run whatever is available this time — is exactly the one
 * that quietly substitutes a smaller experiment. So the panel is built around three things:
 *
 * 1.  **The identity is the manifest.** The run id is printed beside the manifest digest, and a
 *     refresh re-opens the same run rather than starting another. There is no "new run" button,
 *     because there is no such operation: posting the plan again is resuming it.
 * 2.  **A retry and an edit are different remedies for different failures.** An operational
 *     failure offers Retry; a refusal offers an editable copy and does not offer Retry at all.
 *     The two buttons are never both live, because the whole point is that the answer to "the
 *     archive says no" is not "ask again".
 * 3.  **Nothing here shows a result.** The progress payload carries statuses, digests, a bounded
 *     work fraction and remediation text, and the backend type it comes from has nowhere to put a
 *     p-value. `results_visible` is displayed as a state of the run, not as a toggle on a value
 *     this panel is holding back.
 *
 * There is no domain branch in this file. Components arrive named by the run, so a fifth domain
 * appears in the grid without this panel learning what it is.
 */

const STATUS_TONE: Record<string, string> = {
  COMPLETE: 'bg-emerald-900/40 text-emerald-200 border-emerald-700/40',
  MISSING: 'bg-amber-900/40 text-amber-200 border-amber-700/40',
  REFUSED: 'bg-rose-900/50 text-rose-200 border-rose-700/40',
  FAILED: 'bg-orange-900/40 text-orange-200 border-orange-700/40',
  TIMED_OUT: 'bg-orange-900/40 text-orange-200 border-orange-700/40',
  PENDING: 'bg-slate-800/60 text-slate-400 border-slate-700/60'
};

const STATE_TONE: Record<string, string> = {
  COMPLETE: 'bg-emerald-900/40 text-emerald-200',
  REFUSED: 'bg-rose-900/50 text-rose-200',
  FAILED: 'bg-orange-900/40 text-orange-200',
  CANCELLED: 'bg-slate-800 text-slate-400'
};

function tone(map: Record<string, string>, key: string, fallback: string) {
  return map[key] || fallback;
}

/** The declared path, with the current state marked. Drawn from the backend's own table, so a
 *  state the server adds appears here without this file being edited. */
export function RunStateTrail({ machine, state }: { machine: types.RunStateMachine; state: string }) {
  const path = ['DRAFT', 'PREFLIGHTED', 'FROZEN', ...machine.work_stages, 'COMPLETE'];
  const reached = path.indexOf(state);
  return (
    <ol className="flex flex-wrap gap-1 text-[10px]" aria-label="declared run states">
      {path.map((step, index) => (
        <li key={step}
            aria-current={step === state ? 'step' : undefined}
            className={`px-2 py-0.5 rounded font-mono ${step === state
              ? 'bg-sky-700 text-white'
              : reached >= 0 && index < reached ? 'bg-slate-800 text-slate-400'
                : 'bg-slate-900 text-slate-600'}`}>
          {step}
        </li>
      ))}
      {!path.includes(state) && (
        <li aria-current="step" className={`px-2 py-0.5 rounded font-mono ${tone(STATE_TONE, state, 'bg-slate-800 text-slate-300')}`}>
          {state}
        </li>
      )}
    </ol>
  );
}

export function RunWorkerSuitePicker({ suites, selected, onSelect, disabled }: {
  suites: types.RunWorkerSuite[]; selected: string;
  onSelect: (name: string) => void; disabled?: boolean;
}) {
  return (
    <div className="space-y-2">
      <label htmlFor="run-worker-suite" className="text-xs text-slate-400">Worker suite</label>
      <select id="run-worker-suite" value={selected} disabled={disabled}
              onChange={(event) => onSelect(event.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs">
        {suites.map((suite) => (
          <option key={suite.name} value={suite.name}>{suite.name}</option>
        ))}
      </select>
      {suites.filter((suite) => suite.name === selected).map((suite) => (
        <p key={suite.name} className="text-[11px] text-slate-500">
          {suite.description}
          {suite.capabilities.acquires === false &&
            <span className="text-amber-300"> Acquires nothing: this is a rehearsal, not data.</span>}
        </p>
      ))}
    </div>
  );
}

export function RunProgressPanel({ progress, receipt, machine, onRetry, onCancel, onEditableCopy, busy }: {
  progress: types.RunProgress;
  receipt: types.RunReceipt;
  machine: types.RunStateMachine;
  onRetry?: () => void;
  onCancel?: () => void;
  onEditableCopy?: () => void;
  busy?: boolean;
}) {
  const work = progress.bounded_work;
  const percent = Math.round(100 * work.fraction);
  const terminal = machine.terminal_states.includes(progress.state);
  const refused = progress.state === 'REFUSED';

  return (
    <div className="space-y-4 text-xs">
      <div className="flex flex-wrap gap-3 items-center">
        <span className={`px-2 py-0.5 rounded text-[11px] ${tone(STATE_TONE, progress.state, 'bg-sky-900/40 text-sky-200')}`}>
          {progress.state}
        </span>
        <RunStateTrail machine={machine} state={progress.state} />
      </div>

      <div>
        <div className="flex justify-between text-[11px] text-slate-400">
          <span>{work.completed_steps} of {work.total_steps} declared steps</span>
          <span>{percent}%</span>
        </div>
        <div className="h-2 rounded bg-slate-800 overflow-hidden mt-1"
             role="progressbar" aria-valuenow={work.completed_steps}
             aria-valuemin={0} aria-valuemax={work.total_steps}>
          <div className="h-full bg-sky-500" style={{ width: `${percent}%` }} />
        </div>
        <p className="text-[10px] text-slate-500 mt-1">
          Bounded by the declared plan: every step was named before the run started.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {progress.stages.map((stage) => (
          <div key={stage.stage} className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
            <div className="font-mono text-[11px] text-slate-300">{stage.stage}</div>
            <ul className="mt-2 space-y-1">
              {stage.components.map((component) => (
                <li key={component.component} className="flex flex-wrap gap-2 items-baseline">
                  <span className={`px-1.5 py-0.5 rounded border text-[10px] ${tone(STATUS_TONE, component.status, 'bg-slate-800 text-slate-300 border-slate-700')}`}>
                    {component.status}
                  </span>
                  <span className="text-slate-300">{component.component}</span>
                  {component.reused && <span className="text-[10px] text-slate-500">replayed, not re-requested</span>}
                  {component.artifact_sha256 && (
                    <span className="font-mono text-[10px] text-slate-600 break-all">
                      {component.artifact_sha256.slice(0, 12)}
                    </span>
                  )}
                  {component.remediation && (
                    <span className="text-[10px] text-orange-200 basis-full">{component.remediation}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {receipt.missing_components.length > 0 && (
        <div className="border border-amber-500/40 bg-amber-500/5 rounded-lg p-3">
          <div className="text-[11px] text-amber-200">Components this run did not produce</div>
          <ul className="mt-1 space-y-1 text-[11px] text-amber-100/80 list-disc pl-4">
            {receipt.missing_components.map((row) => (
              <li key={`${row.stage}/${row.component}`}>
                <span className="font-mono">{row.stage}/{row.component}</span> — {row.status}. {row.detail}
              </li>
            ))}
          </ul>
        </div>
      )}

      {receipt.stage_decisions.length > 0 && (
        <ul className="space-y-1 text-[11px] text-slate-400">
          {receipt.stage_decisions.map((decision, index) => (
            <li key={`${decision.stage}-${index}`}>
              <span className="font-mono text-slate-300">{decision.stage}</span>
              {' '}<span className="text-slate-500">{decision.verdict}</span> — {decision.reason}
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap gap-2 items-center">
        <button onClick={onRetry} disabled={!progress.retryable || busy}
                title={progress.retryable
                  ? 'Re-executes only the components that failed operationally.'
                  : 'Only an operational failure can be retried.'}
                className="px-3 py-1.5 rounded bg-orange-700 hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed text-xs">
          Retry the operational failure
        </button>
        <button onClick={onEditableCopy} disabled={!refused || busy}
                title={refused
                  ? 'Writes a new editable draft. The refused run is left exactly as it is.'
                  : 'An editable copy is the remedy for a refusal.'}
                className="px-3 py-1.5 rounded bg-violet-700 hover:bg-violet-600 disabled:opacity-40 disabled:cursor-not-allowed text-xs">
          Open an editable copy
        </button>
        <button onClick={onCancel} disabled={terminal || busy}
                className="px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed text-xs">
          Cancel this run
        </button>
      </div>

      <dl className="text-[10px] text-slate-500 space-y-1">
        <div className="break-all"><dt className="inline text-slate-400">Run </dt><dd className="inline font-mono">{receipt.run_id}</dd></div>
        <div className="break-all"><dt className="inline text-slate-400">Manifest </dt><dd className="inline font-mono">{receipt.manifest_sha256}</dd></div>
        <div><dt className="inline text-slate-400">Coverage policy </dt><dd className="inline">{receipt.coverage_policy.requirement} at {receipt.coverage_policy.minimum_fraction}</dd></div>
        <div><dt className="inline text-slate-400">Results visible </dt><dd className="inline">{progress.results_visible ? 'yes, the run is complete' : 'no, the run has not completed'}</dd></div>
      </dl>

      <p className="text-[11px] text-slate-500">{progress.claim_boundary}</p>
      <p className="text-[11px] text-slate-500">{receipt.claim_boundary}</p>
    </div>
  );
}
