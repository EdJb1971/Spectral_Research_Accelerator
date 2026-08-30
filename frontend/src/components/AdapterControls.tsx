import * as types from '../types/api';

/**
 * Schema-driven controls for one registered domain adapter (TG17.3).
 *
 * The only switch in this file is on a control's declared `kind`. There is deliberately no
 * `domain === 'reanalysis'` branch anywhere: a fifth adapter reaches this form by registering
 * its `ControlSchema`, and this component never learns its name. TG17.3's review rule is
 * explicit that a hardcoded form in the generic composer fails, and the acceptance test
 * installs an adapter this file has never seen.
 *
 * Every control renders its declared help text. A control whose meaning lives only in the
 * adapter author's head cannot be operated by a researcher, so the backend refuses to register
 * one without help — and this component shows what that refusal bought.
 */
export function AdapterControl({ field, value, onChange }: {
  field: types.AdapterControlField;
  value: any;
  onChange: (next: any) => void;
}) {
  const id = `adapter-control-${field.name}`;
  const shared = 'mt-1 w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs';
  const current = value === undefined || value === null ? field.default : value;

  const input = () => {
    switch (field.kind) {
      case 'boolean':
        return <input id={id} type="checkbox" checked={Boolean(current)}
          onChange={(event) => onChange(event.target.checked)} className="mt-1" />;
      case 'enum':
        return <select id={id} value={String(current ?? '')} className={shared}
          onChange={(event) => onChange(event.target.value)}>
          {field.choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>{String(choice)}</option>))}
        </select>;
      case 'integer':
      case 'number':
        return <input id={id} type="number" value={current ?? ''} className={shared}
          min={field.minimum ?? undefined} max={field.maximum ?? undefined}
          step={field.kind === 'integer' ? 1 : 'any'}
          onChange={(event) => onChange(event.target.value === '' ? null
            : (field.kind === 'integer' ? parseInt(event.target.value, 10) : parseFloat(event.target.value)))} />;
      case 'utc_instant':
        return <input id={id} type="datetime-local" className={shared}
          value={current ? String(current).replace(/Z$/, '').slice(0, 16) : ''}
          onChange={(event) => onChange(event.target.value
            ? new Date(`${event.target.value}:00Z`).toISOString() : null)} />;
      case 'content_record':
        return <input id={id} type="text" value={current ?? ''} className={`${shared} font-mono`}
          placeholder="sha256 of a published record"
          onChange={(event) => onChange(event.target.value || null)} />;
      default:
        return <input id={id} type="text" value={current ?? ''} className={shared}
          onChange={(event) => onChange(event.target.value || null)} />;
    }
  };

  return (
    <div className="mb-3">
      <label htmlFor={id} className="text-[11px] text-slate-400">
        {field.label}{field.units ? ` (${field.units})` : ''}
        {field.required ? '' : <span className="text-slate-600"> · optional</span>}
      </label>
      {input()}
      <p id={`${id}-help`} className="text-[10px] text-slate-500 mt-1">{field.help}</p>
    </div>
  );
}

export function AdapterControlPanel({ adapter, parameters, onChange }: {
  adapter: types.DomainExperimentAdapterDescription | undefined;
  parameters: Record<string, any>;
  onChange: (next: Record<string, any>) => void;
}) {
  if (!adapter) {
    return <p className="text-[11px] text-amber-300">
      No adapter is registered for this domain, so its controls, coverage plan and translation
      are unavailable. This is a refusal, not an empty catalogue.
    </p>;
  }
  return (
    <div>
      {adapter.controls.fields.map((field) => (
        <AdapterControl key={field.name} field={field} value={parameters[field.name]}
          onChange={(next) => onChange({ ...parameters, [field.name]: next })} />
      ))}
      <p className="text-[10px] text-slate-600 break-all font-mono">
        {adapter.adapter_id} · {adapter.definition_sha256.slice(0, 16)}
      </p>
    </div>
  );
}
