import * as types from '../types/api';

/**
 * The declared family, shown before the freeze (TG17.5).
 *
 * The point of this panel is that the multiplication is visible while it can still be changed.
 * A researcher who adds a fifth domain or a fourth duration should see what it did to the
 * number of tests and to the surrogate ensemble the study would need, rather than learning
 * after four archives have been read that the pass is arithmetically incapable of rejecting
 * anything and reports nothing for a reason that is not the data.
 *
 * Two numbers are deliberately kept apart. `family_size` is what was declared and is the
 * correction unit. The confirmatory count is what a generate/confirm split corrects over, and
 * it is legitimate only because the held-out partition it names was closed while the candidates
 * were chosen — so the partition is printed beside it, not hidden behind the smaller number.
 *
 * There is no domain branch anywhere in this file. The axes, the costs and the null families
 * all arrive as declared rows, so a fifth domain reaches this panel without the panel learning
 * its name.
 */

const int = (value: number) => value.toLocaleString();

export function FamilyExpansionPanel({ expansion }: { expansion: types.FamilyExpansion }) {
  const correction = expansion.correction;
  const split = correction.stage === 'generate_then_confirm';

  return (
    <div className="space-y-4 text-xs">
      <div className="flex flex-wrap gap-3 items-baseline">
        <span className={`px-2 py-0.5 rounded text-[11px] ${correction.affordable
          ? 'bg-emerald-900/40 text-emerald-200' : 'bg-rose-900/50 text-rose-200'}`}>
          {correction.affordable ? 'R18: can reject' : 'R18: cannot reject anything'}
        </span>
        <span className="text-slate-200 font-mono">{expansion.in_human_terms}</span>
      </div>

      <table className="w-full text-left">
        <thead className="text-slate-500">
          <tr>
            <th className="py-1 pr-2">axis</th>
            <th className="py-1 pr-2">declared</th>
            <th className="py-1 pr-2">contributes</th>
            <th className="py-1">values</th>
          </tr>
        </thead>
        <tbody className="text-slate-300">
          {expansion.axes.map((axis) => (
            <tr key={axis.axis} className="border-t border-slate-800">
              <td className="py-1 pr-2 font-mono">{axis.axis}</td>
              <td className="py-1 pr-2">{axis.declared_values}</td>
              <td className="py-1 pr-2 font-mono text-slate-400">{axis.contributes}</td>
              <td className="py-1 text-slate-500">{axis.examples.join(', ')}
                {axis.declared_values > axis.examples.length ? ' …' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Figure label="declared tests" value={int(expansion.family_size)} />
        <Figure label="corrected over" value={int(correction.correction_unit_members)}
          hint={split ? `confirmed on ${correction.held_out_partition}` : 'every declared member'} />
        <Figure label="surrogates declared" value={int(correction.n_surrogates)} />
        <Figure label="surrogates required" value={int(correction.surrogates_required)}
          hint={`p-value floor ${correction.p_value_floor.toExponential(2)}`} />
      </div>

      {correction.warning && (
        <p className="text-rose-200 bg-rose-950/40 border border-rose-900 rounded p-2">
          {correction.warning}
        </p>
      )}

      {split && (
        <p className="text-amber-200 bg-amber-950/30 border border-amber-900/60 rounded p-2">
          {correction.note}
        </p>
      )}

      <div>
        <div className="text-slate-500 mb-1">what one more of each would cost, before the freeze</div>
        <table className="w-full text-left">
          <thead className="text-slate-500">
            <tr>
              <th className="py-1 pr-2">adding</th>
              <th className="py-1 pr-2">tests</th>
              <th className="py-1 pr-2">added</th>
              <th className="py-1 pr-2">surrogates needed</th>
              <th className="py-1">still affordable</th>
            </tr>
          </thead>
          <tbody className="text-slate-300">
            {expansion.expansion_cost.map((cost) => (
              <tr key={cost.axis} className="border-t border-slate-800">
                <td className="py-1 pr-2">{cost.adds}</td>
                <td className="py-1 pr-2 font-mono">{int(cost.family_size_before)} → {int(cost.family_size_after)}</td>
                <td className="py-1 pr-2 font-mono text-amber-300">+{int(cost.members_added)}</td>
                <td className="py-1 pr-2 font-mono">{int(cost.surrogates_required_after)}</td>
                <td className={`py-1 ${cost.affordable_after ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {cost.affordable_after ? 'yes' : 'no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="border-t border-slate-800 pt-2 space-y-1">
        <div className="text-slate-400">
          screen {int(expansion.screen_and_confirm.screen_family_size)} pairs · complete search{' '}
          {int(expansion.screen_and_confirm.complete_family_size)} · correction unit{' '}
          <span className="font-mono text-slate-200">{int(expansion.screen_and_confirm.correction_unit)}</span>
        </div>
        <p className="text-slate-500">{expansion.screen_and_confirm.rule}</p>
      </div>

      <div className="border-t border-slate-800 pt-2 space-y-1">
        <div className="text-slate-400">
          precedence unavailable for {expansion.precedence.unavailable_precedence_members} member(s)
          {expansion.precedence.domains_without_precedence_policy.length > 0 && (
            <span className="text-slate-500"> — no justified lag policy:{' '}
              <span className="font-mono">
                {expansion.precedence.domains_without_precedence_policy.join(', ')}
              </span>
            </span>
          )}
        </div>
        <p className="text-slate-500">{expansion.precedence.note}</p>
      </div>

      <p className="text-slate-500 border-t border-slate-800 pt-2">{expansion.claim_boundary}</p>
    </div>
  );
}

function Figure({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded p-2">
      <div className="text-slate-500">{label}</div>
      <div className="text-slate-100 font-mono text-sm">{value}</div>
      {hint && <div className="text-slate-500 mt-0.5">{hint}</div>}
    </div>
  );
}

/**
 * The declared null families, with what each preserves.
 *
 * A refused family renders disabled with its reason rather than being hidden. `global_value_shuffle`
 * is the case this exists for: every domain can execute it, so it will occur to everyone, and a
 * missing option reads as an oversight where a refused one reads as an answer.
 */
export function NullFamilyPicker({ list, mode, method, parameters, onChange }: {
  list: types.NullFamilyList | null;
  mode: string;
  method: string;
  parameters: Record<string, any>;
  onChange: (method: string, parameters: Record<string, any>) => void;
}) {
  const selected = method;
  const onSelect = (name: string) => onChange(name, {});
  if (!list) return null;
  return (
    <div className="space-y-2 text-xs">
      <div className="text-slate-500">{list.note}</div>
      {list.families.map((family) => {
        const wrongMode = !family.modes.includes(mode);
        const disabled = !family.admissible || wrongMode || family.admitted_by.length === 0;
        return (
          <label key={family.name}
            className={`block border rounded p-2 ${disabled
              ? 'border-slate-800 bg-slate-900/30 opacity-60'
              : 'border-slate-700 bg-slate-900/60 cursor-pointer'}`}>
            <div className="flex items-baseline gap-2">
              <input type="radio" name="null-family" disabled={disabled}
                checked={selected === family.name}
                onChange={() => onSelect(family.name)} />
              <span className="font-mono text-slate-200">{family.name}</span>
              <span className="text-slate-500">{family.modes.join(', ')}</span>
            </div>
            <div className="text-slate-400 mt-1">{family.definition}</div>
            <div className="mt-1 text-slate-500">
              preserves <span className="text-emerald-300">{family.preserves.join(', ')}</span>
              {' · '}destroys <span className="text-amber-300">{family.destroys.join(', ')}</span>
            </div>
            {family.parameters.length > 0 && (
              <div className="mt-1 space-y-1">
                {family.parameters.map((name) => (
                  <label key={name} className="flex gap-2 items-center text-slate-400">
                    <span className="font-mono">{name}</span>
                    <input type="number" disabled={disabled || selected !== family.name}
                      className="bg-slate-950 border border-slate-700 rounded px-2 py-0.5 w-40"
                      placeholder="no default — this is a scientific choice"
                      value={selected === family.name && parameters[name] !== undefined
                        ? parameters[name] : ''}
                      onChange={(event) => onChange(family.name, {
                        ...parameters,
                        [name]: event.target.value === '' ? undefined : Number(event.target.value),
                      })} />
                  </label>
                ))}
              </div>
            )}
            {!family.admissible && (
              <div className="text-rose-300 mt-1">refused: {family.inadmissible_reason}</div>
            )}
            {family.admissible && family.admitted_by.length === 0 && (
              <div className="text-slate-500 mt-1">no registered domain admits this null</div>
            )}
            {family.admissible && family.admitted_by.length > 0 && (
              <div className="text-slate-500 mt-1">
                admitted by <span className="font-mono">{family.admitted_by.join(', ')}</span>
              </div>
            )}
            {wrongMode && family.admissible && (
              <div className="text-slate-500 mt-1">
                not a null for {mode}: it answers a question this mode does not ask
              </div>
            )}
          </label>
        );
      })}
    </div>
  );
}
