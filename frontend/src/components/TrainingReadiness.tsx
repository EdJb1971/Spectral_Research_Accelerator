import type { TrainingRepresentationCatalogue } from '../types/api';

interface Props {
  catalogue: TrainingRepresentationCatalogue | null;
}

export function TrainingReadiness({ catalogue }: Props) {
  return (
    <section
      data-testid="training-representation-readiness"
      className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4"
    >
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Training-loop readiness — not merely analytical availability</h3>
          <p className="text-xs text-slate-400 mt-1">
            Accepted means batched (B,C,H,W), differentiable encode/inverse and tested backward
            flow. “Analysis only” must not be dropped into a model training script.
          </p>
        </div>
        {catalogue && (
          <span className="text-[11px] font-mono text-slate-400 bg-slate-950 border border-slate-800 rounded px-2 py-1 whitespace-nowrap">
            {catalogue.selected_spatial_shape.join('×')} · L{catalogue.selected_levels} · {catalogue.selected_wavelet}
          </span>
        )}
      </div>

      {!catalogue ? (
        <p className="text-xs text-amber-400">Capability evidence is unavailable; no training-readiness claim is being shown.</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {catalogue.representations.map((entry) => (
            <article key={entry.name} className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="text-sm font-semibold text-slate-200">{entry.label}</h4>
                  <p className="text-[11px] text-slate-500 mt-0.5">{entry.scientific_role}</p>
                </div>
                <span className={`text-[10px] uppercase tracking-wide font-semibold rounded px-2 py-1 ${entry.status === 'accepted' ? 'bg-emerald-950 text-emerald-400 border border-emerald-900' : 'bg-amber-950 text-amber-400 border border-amber-900'}`}>
                  {entry.status === 'accepted' ? 'training accepted' : 'analysis only'}
                </span>
              </div>

              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
                <div><dt className="text-slate-500">Coefficients</dt><dd className="text-slate-300">{entry.coefficient_ratio}</dd></div>
                <div><dt className="text-slate-500">Exact inverse</dt><dd className="text-slate-300">{entry.exact_inverse === null ? 'not accepted' : entry.exact_inverse ? 'yes' : 'no'}</dd></div>
                <div className="col-span-2"><dt className="text-slate-500">Shift behaviour</dt><dd className="text-slate-300">{entry.shift_behavior}</dd></div>
                <div className="col-span-2"><dt className="text-slate-500">Directional meaning</dt><dd className="text-slate-300">{entry.directionality}</dd></div>
                <div className="col-span-2"><dt className="text-slate-500">Boundary convention</dt><dd className="text-slate-300">{entry.boundary}</dd></div>
              </dl>

              {entry.selected_configuration && (
                <div className={`rounded-lg border p-3 ${entry.selected_configuration.coarsest_scale_has_valid_interior ? 'border-slate-800 bg-slate-900/60' : 'border-rose-900 bg-rose-950/30'}`}>
                  <p className="text-[11px] text-slate-300 mb-2">
                    <strong>{entry.selected_configuration.atlas_planes_per_input_channel
                      ? `${entry.selected_configuration.atlas_planes_per_input_channel} exact real atlas planes per input channel.`
                      : `${entry.selected_configuration.coefficient_channels_per_input_channel} coefficient channels per input channel.`}</strong>{' '}
                    {entry.selected_configuration.implementation_policy ?? entry.selected_configuration.display_contract}
                  </p>
                  <div className="grid grid-cols-3 gap-1 text-[10px] font-mono">
                    <span className="text-slate-500">level</span><span className="text-slate-500">edge / side</span><span className="text-slate-500">valid H×W</span>
                    {(entry.selected_configuration.valid_interior_halfwidth_by_level
                      ?? entry.selected_configuration.valid_interior_halfwidth_parent_px_by_level
                      ?? []).map((margin, index) => (
                      <div className="contents" key={index}>
                        <span className="text-slate-300">{index + 1}</span>
                        <span className="text-slate-300">{margin}px parent</span>
                        <span className={(entry.selected_configuration!.valid_interior_shape_by_level
                          ?? entry.selected_configuration!.valid_interior_native_shape_by_level
                          ?? [])[index].every(v => v > 0) ? 'text-slate-300' : 'text-rose-400'}>
                          {(entry.selected_configuration!.valid_interior_shape_by_level
                            ?? entry.selected_configuration!.valid_interior_native_shape_by_level
                            ?? [])[index].join('×')}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="text-[11px] leading-relaxed">
                <p className="text-slate-500">Limits that travel with the result</p>
                <ul className="list-disc pl-4 text-amber-300/90 mt-1 space-y-0.5">
                  {entry.limitations.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
              <div className="text-[10px] leading-relaxed border-t border-slate-800 pt-2">
                <p className="text-emerald-500">Verified: {entry.verified.length ? entry.verified.join('; ') : 'no accepted training backend'}</p>
                <p className="text-slate-500 mt-1">Not run: {entry.not_run.join('; ')}</p>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
