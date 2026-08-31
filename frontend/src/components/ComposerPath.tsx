import { useRef, useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, Circle, Download, FileJson, Lock,
         Upload } from 'lucide-react';
import * as types from '../types/api';

/**
 * TG17.7 The guided path, rendered from the served contract.
 *
 * Nothing in this file knows the order of the steps, what a step requires, or which action comes
 * next. It knows how to draw a step and how to draw the one action the server named. That is the
 * whole point: an order of operations held in a component is a second order of operations, and
 * the one a researcher follows would not be the one the receipt records.
 *
 * There is deliberately no domain branch here either, for the same reason there is none in the
 * adapter controls or the run monitor: a fifth domain reaches this surface by registering.
 */

const STATUS_STYLE: Record<string, string> = {
  SATISFIED: 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10',
  ACTION_REQUIRED: 'text-amber-200 border-amber-500/40 bg-amber-500/10',
  BLOCKED: 'text-rose-200 border-rose-500/40 bg-rose-500/10',
};

export function StepStatusBadge({ status }: { status: string }) {
  return (
    <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${
      STATUS_STYLE[status] || 'text-slate-300 border-slate-700 bg-slate-800'}`}>
      {status.split('_').join(' ')}
    </span>
  );
}

/** The seven steps as a tablist. Arrow keys move between them; every step stays reachable even
 *  when it is blocked, because hiding a step hides the reason it is blocked. */
export function PathStepper({ steps, active, onSelect }: {
  steps: types.ComposerStepState[];
  active: string;
  onSelect: (stepId: string) => void;
}) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  const move = (event: React.KeyboardEvent, index: number) => {
    const keys: Record<string, number> = {
      ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: steps.length - 1,
    };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const next = steps[(keys[event.key] + steps.length) % steps.length];
    onSelect(next.step_id);
    refs.current[next.step_id]?.focus();
  };

  return (
    <div role="tablist" aria-label="Experiment composition path"
         className="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
      {steps.map((step, index) => (
        <button key={step.step_id} role="tab" id={`composer-tab-${step.step_id}`}
                aria-selected={active === step.step_id}
                aria-controls={`composer-panel-${step.step_id}`}
                tabIndex={active === step.step_id ? 0 : -1}
                ref={(node) => { refs.current[step.step_id] = node; }}
                onKeyDown={(event) => move(event, index)}
                onClick={() => onSelect(step.step_id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs border transition-colors
                  focus:outline-none focus:ring-2 focus:ring-teal-400 ${
                  active === step.step_id
                    ? 'bg-slate-800 border-teal-500/60 text-white'
                    : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'}`}>
          {step.status === 'SATISFIED'
            ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" aria-hidden="true" />
            : step.status === 'BLOCKED'
              ? <Lock className="w-3.5 h-3.5 text-rose-400" aria-hidden="true" />
              : <Circle className="w-3.5 h-3.5 text-amber-400" aria-hidden="true" />}
          <span className="font-semibold">{step.ordinal}. {step.title}</span>
        </button>
      ))}
    </div>
  );
}

/** One action, named by the server. A banner that offered two would be a banner that had an
 *  opinion about which scientific commitment to make next. */
export function NextActionBanner({ state, onGo }: {
  state: types.ComposerPathState;
  onGo: (stepId: string) => void;
}) {
  const action = state.next_action;
  if (!action) {
    return (
      <div role="status" className="rounded-xl border border-slate-700 bg-slate-900 p-4 text-sm text-slate-300">
        Every step of the declared path is satisfied. That is an executed plan, not a finding.
      </div>
    );
  }
  return (
    <div role="status"
         className={`rounded-xl border p-4 ${action.blocked
           ? 'border-rose-500/40 bg-rose-500/5' : 'border-teal-500/40 bg-teal-500/5'}`}>
      <div className="flex items-start gap-3">
        {action.blocked
          ? <AlertTriangle className="w-5 h-5 text-rose-300 shrink-0" aria-hidden="true" />
          : <ArrowRight className="w-5 h-5 text-teal-300 shrink-0" aria-hidden="true" />}
        <div className="min-w-0">
          <p className="text-sm font-semibold text-white">
            {action.blocked ? 'Blocked before: ' : 'Next legitimate action: '}{action.label}
          </p>
          <p className="text-xs text-slate-300 mt-1">{action.why}</p>
          <p className="font-mono text-[10px] text-slate-500 mt-1 break-all">{action.route}</p>
        </div>
        <button onClick={() => onGo(action.step_id)}
                className="ml-auto shrink-0 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs
                           focus:outline-none focus:ring-2 focus:ring-teal-400">
          Go to step
        </button>
      </div>
    </div>
  );
}

/** Acquired material, an executed run, a finding, admitted evidence. Four separate things, shown
 *  together because the mistake is never in one of them - it is in the slide between them. */
export function StageLadder({ ladder }: { ladder: types.ComposerLadderRung[] }) {
  return (
    <ol className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
      {ladder.map((rung) => (
        <li key={rung.rung} className={`rounded-lg border p-3 ${rung.reached
          ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-slate-800 bg-slate-950/60'}`}>
          <div className="flex items-center gap-2">
            {rung.reached
              ? <CheckCircle2 className="w-4 h-4 text-emerald-400" aria-hidden="true" />
              : <Circle className="w-4 h-4 text-slate-600" aria-hidden="true" />}
            <span className="text-sm font-semibold text-slate-100">{rung.title}</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-2">{rung.is}</p>
          <p className="text-[11px] text-amber-300/80 mt-1">Not: {rung.is_not}</p>
          <p className="text-[11px] text-slate-500 mt-1">Gate: {rung.gate}</p>
          {!rung.reached && rung.why_not
            && <p className="text-[11px] text-slate-500 mt-1">Not reached: {rung.why_not}</p>}
        </li>
      ))}
    </ol>
  );
}

/** Every registered domain, with what it breaks. Nothing is filtered out: a row that cannot be
 *  chosen is disabled beside the backend's reason, because a control that vanishes when it
 *  becomes inadmissible is indistinguishable from one that was never offered. */
export function DomainMenuPanel({ menu, selectedDomains, onToggle, busy }: {
  menu: types.ComposerDomainMenu | null;
  /** Which domains the manifest declares. Deliberately **not** `row.selected` from the payload:
   *  the selection is a fact about the plan this browser is holding, and the menu is a
   *  catalogue. Driving the checkbox from the server's echo made it wait on a round-trip, so it
   *  snapped back to its old value and reported the opposite of what was just chosen until the
   *  reply landed (defect D79). The server still echoes `selected`; nothing renders from it. */
  selectedDomains: string[];
  onToggle: (domain: string, selected: boolean) => void;
  busy: boolean;
}) {
  if (!menu) return <p className="text-sm text-slate-400">Loading the registered domains…</p>;
  if (menu.domains.length === 0) {
    return <p className="text-sm text-slate-400">No domain adapter is registered in this
      instance. An empty menu is an empty registry, not an empty archive.</p>;
  }
  const selected = menu.domains.filter((row) => selectedDomains.includes(row.domain)).length;
  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">{menu.note}</p>
      <p className={`text-xs ${selected >= menu.minimum_domains ? 'text-slate-400' : 'text-amber-300'}`}>
        {selected} of at least {menu.minimum_domains} domains selected.
      </p>
      <ul className="space-y-2">
        {menu.domains.map((row) => (
          <li key={row.adapter_id}
              className={`rounded-lg border p-3 ${row.selectable
                ? 'border-slate-800 bg-slate-950/60' : 'border-slate-800 bg-slate-950/30 opacity-70'}`}>
            <label className="flex items-start gap-3 text-sm text-slate-200">
              <input type="checkbox" checked={selectedDomains.includes(row.domain)}
                     disabled={busy || !row.selectable}
                     aria-label={`Include ${row.domain}`}
                     onChange={(event) => onToggle(row.domain, event.target.checked)}
                     className="mt-1 focus:outline-none focus:ring-2 focus:ring-teal-400" />
              <span className="min-w-0">
                <span className="font-semibold">{row.domain}</span>
                <span className="block text-xs text-slate-400">{row.label}</span>
                <span className="block text-[11px] text-amber-300/90 mt-1">
                  Breaks: {row.breaks.length ? row.breaks.join(', ') : 'no declared assumption'}
                </span>
                <span className="block text-[11px] text-slate-500">
                  Licence {row.licence} · lag policy {row.lag_policy} · precedence{' '}
                  {row.precedence_admissible ? 'admissible' : 'not admissible'}
                </span>
                {!row.selectable && row.unavailable_reason
                  && <span className="block text-[11px] text-rose-300 mt-1">
                       Unavailable: {row.unavailable_reason}</span>}
              </span>
            </label>
          </li>
        ))}
      </ul>
      <p className="text-[11px] text-slate-500">{menu.claim_boundary}</p>
    </div>
  );
}

/** Presets resolved on the server, applied as the explicit instants they resolved to. */
export function WindowPresetPicker({ presets, onApply, busy }: {
  presets: types.ComposerWindowPresets | null;
  onApply: (preset: types.ComposerWindowPreset) => void;
  busy: boolean;
}) {
  if (!presets) return null;
  return (
    <div className="space-y-2">
      <p className="text-xs text-slate-500">{presets.note}</p>
      <div className="flex flex-wrap gap-2">
        {presets.presets.map((preset) => (
          <button key={preset.preset} disabled={busy} onClick={() => onApply(preset)}
                  aria-label={`Apply the ${preset.preset} preset`}
                  className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50
                             text-xs focus:outline-none focus:ring-2 focus:ring-teal-400">
            {preset.preset.split('_').join(' ')}
            <span className="block font-mono text-[10px] text-slate-500">
              {preset.start_utc} → {preset.end_utc}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

/** The plan in the plan's own sentences, generated from the bytes that are hashed. */
export function PreregistrationSummaryPanel({ summary }: {
  summary: types.ComposerPreregistrationSummary | null;
}) {
  if (!summary) {
    return <p className="text-sm text-slate-400">No preregistration summary has been rendered
      for this manifest yet.</p>;
  }
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
      <ol className="space-y-2 list-decimal pl-5">
        {summary.sentences.map((sentence, index) => (
          <li key={index} className="text-sm text-slate-200">{sentence}</li>
        ))}
      </ol>
      <p className="font-mono text-[10px] text-slate-500 mt-3 break-all">
        manifest sha256 {summary.manifest_sha256}
      </p>
      <p className="text-[11px] text-slate-500 mt-1">{summary.claim_boundary}</p>
    </div>
  );
}

/** The advanced inspector: disclosed, never hidden, and never required. Direct JSON editing is
 *  not a step on the path - this exists so a plan can be moved between instances and cited. */
export function ManifestInspector({ manifest, envelope, onExport, onImport, busy }: {
  manifest: Record<string, any>;
  envelope: types.ComposerManifestEnvelope | null;
  onExport: () => void;
  onImport: (text: string) => void;
  busy: boolean;
}) {
  const [pasted, setPasted] = useState('');
  return (
    <details className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
      <summary className="cursor-pointer text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-400">
        Advanced: inspect, export or import this manifest
      </summary>
      <p className="text-xs text-slate-500 mt-2">
        Composing an experiment never requires editing this. An imported envelope whose body
        disagrees with its digest is refused rather than loaded.
      </p>
      <div className="flex flex-wrap gap-2 mt-3">
        <button onClick={onExport} disabled={busy}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs
                           flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-teal-400">
          <Download className="w-3.5 h-3.5" aria-hidden="true" /> Export manifest envelope
        </button>
        <button onClick={() => onImport(pasted)} disabled={busy || !pasted.trim()}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs
                           flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-teal-400">
          <Upload className="w-3.5 h-3.5" aria-hidden="true" /> Import pasted envelope
        </button>
      </div>
      <label htmlFor="composer-envelope-import" className="block text-xs text-slate-400 mt-3">
        Machine-readable envelope to import
      </label>
      <textarea id="composer-envelope-import" value={pasted} rows={4}
                onChange={(event) => setPasted(event.target.value)}
                placeholder='{"schema": "composer-manifest-envelope/v1", ...}'
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded px-3 py-2
                           font-mono text-[11px] focus:outline-none focus:ring-2 focus:ring-teal-400" />
      {envelope && (
        <p className="font-mono text-[10px] text-slate-500 mt-2 break-all">
          <FileJson className="w-3 h-3 inline mr-1" aria-hidden="true" />
          exported envelope {envelope.envelope_sha256}
        </p>
      )}
      <pre className="mt-3 max-h-64 overflow-auto bg-slate-950 border border-slate-800 rounded p-3
                      text-[10px] text-slate-400">{JSON.stringify(manifest, null, 2)}</pre>
    </details>
  );
}

/** A destructive action asks twice. The second press is a different sentence, not the same one
 *  in red: what makes cancelling dangerous is that a cancelled run is terminal. */
export function ConfirmButton({ label, confirmLabel, onConfirm, disabled, className }: {
  label: string;
  confirmLabel: string;
  onConfirm: () => void;
  disabled?: boolean;
  className?: string;
}) {
  const [armed, setArmed] = useState(false);
  if (!armed) {
    return (
      <button onClick={() => setArmed(true)} disabled={disabled}
              className={className || 'px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs focus:outline-none focus:ring-2 focus:ring-rose-400'}>
        {label}
      </button>
    );
  }
  return (
    <span role="alertdialog" aria-label={label}
          className="inline-flex items-center gap-2 rounded border border-rose-500/50 bg-rose-500/10 px-2 py-1">
      <span className="text-[11px] text-rose-200">{confirmLabel}</span>
      <button onClick={() => { setArmed(false); onConfirm(); }} disabled={disabled}
              className="px-2 py-1 rounded bg-rose-700 hover:bg-rose-600 disabled:opacity-50 text-[11px]
                         focus:outline-none focus:ring-2 focus:ring-rose-300">
        Yes, do it
      </button>
      <button onClick={() => setArmed(false)}
              className="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-[11px]
                         focus:outline-none focus:ring-2 focus:ring-teal-400">
        Keep it
      </button>
    </span>
  );
}
