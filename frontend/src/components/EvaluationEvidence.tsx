import { useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle, FileCheck2, Upload } from 'lucide-react';
import { EvaluationReport } from '../types/api';

const n = (value: number | null, digits = 4) => value === null ? 'undefined' : Number(value).toPrecision(digits);
const shortHash = (value: unknown) => typeof value === 'string' ? `${value.slice(0, 12)}…${value.slice(-8)}` : 'not embedded';

export function EvaluationEvidence({ reports, importing, onImport }: {
  reports: EvaluationReport[];
  importing: boolean;
  onImport: (file: File) => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState<string>('');
  const selected = reports.find(r => r.report_id === selectedId) || reports[0] || null;
  const rankByKey = useMemo(() => new Map(
    (selected?.rank_diagnostics || []).map(r => [`${r.variable}:${r.lead_hours}`, r])
  ), [selected]);

  return <div className="space-y-6 animate-fadeIn">
    <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
      <div>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <FileCheck2 className="text-teal-400 w-5 h-5" /> Verified Forecast Evaluation
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          Persistence-relative FCN3 ensemble evidence. Every displayed number comes from a receipt
          whose nested hashes and lineage were re-verified by the server at import and read time.
        </p>
      </div>
      <label className="cursor-pointer bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold py-2 px-4 rounded-lg flex items-center gap-2">
        <Upload className="w-4 h-4" /> {importing ? 'Verifying…' : 'Import receipt'}
        <input type="file" accept="application/json,.json" className="hidden" disabled={importing}
          onChange={async e => { const file = e.target.files?.[0]; if (file) await onImport(file); e.target.value = ''; }} />
      </label>
    </div>

    {!selected ? <div className="border border-slate-800 bg-slate-900/40 rounded-xl p-8 text-center">
      <FileCheck2 className="w-8 h-8 text-slate-600 mx-auto mb-3" />
      <h3 className="text-sm font-semibold text-slate-300">No authenticated real evaluation receipt</h3>
      <p className="text-xs text-slate-500 mt-2 max-w-xl mx-auto leading-relaxed">
        Metrics and plots remain absent until a T5.6e receipt passes integrity, lineage, forecast-artifact,
        and accepted ERA5-source checks. Synthetic demonstrations are deliberately not shown here.
      </p>
    </div> : <>
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-3">
        <div className="xl:col-span-2 bg-slate-900/50 border border-emerald-500/20 rounded-xl p-4">
          <div className="flex gap-2 text-emerald-400 text-xs font-semibold"><CheckCircle className="w-4 h-4" /> Receipt integrity verified</div>
          <p className="text-[11px] text-slate-500 mt-2">Forecast artifact authenticated; truth manifest declares an accepted official ERA5 catalogue URI.</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">Holdout status</div>
          <div className={selected.readiness.independent_holdout ? 'text-emerald-400 mt-1 text-sm' : 'text-amber-400 mt-1 text-sm'}>
            {selected.readiness.independent_holdout ? 'Fresh post-2019 holdout' : 'Published-period diagnostic'}
          </div>
        </div>
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
          <div className="text-[10px] uppercase tracking-wider text-amber-500">Scientific claim</div>
          <div className="text-amber-300 mt-1 text-sm">Skill not established</div>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-3 lg:items-end bg-slate-900/40 border border-slate-800 rounded-xl p-4">
        <div className="flex-1">
          <label className="text-[10px] uppercase tracking-wider text-slate-500 block mb-1">Verified report</label>
          <select value={selected.report_id} onChange={e => setSelectedId(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-xs font-mono text-slate-300">
            {reports.map(r => <option key={r.report_id} value={r.report_id}>{r.scope.split_start} — {r.scope.split_end} · {r.report_id.slice(0, 12)}</option>)}
          </select>
        </div>
        <div className="text-[11px] font-mono text-slate-500">{selected.scope.initialization_count} initializations · {selected.scope.ensemble_members.length} members · {selected.scope.grid_shape.join('×')} grid</div>
      </div>

      <section className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-800">
          <h3 className="text-sm font-semibold text-slate-200">Matched-sample scores</h3>
          <p className="text-[11px] text-slate-500 mt-1">Cosine-latitude area weighted. Variables retain their own physical units and are never combined.</p>
        </div>
        <div className="overflow-x-auto"><table className="w-full text-[11px] font-mono">
          <thead className="text-slate-500 border-b border-slate-800"><tr>
            {['lead','variable','mean RMSE','persistence RMSE','MSE skill vs persistence','CRPS','spread RMS','spread / skill'].map(x => <th key={x} className="text-right first:text-left py-2 px-3 font-medium">{x}</th>)}
          </tr></thead>
          <tbody>{selected.metrics.map(row => <tr key={`${row.variable}:${row.lead_hours}`} className="border-b border-slate-800/60">
            <td className="py-2 px-3 text-slate-300">+{row.lead_hours} h</td><td className="py-2 px-3 text-right text-teal-300">{row.variable} [{row.unit}]</td>
            <td className="py-2 px-3 text-right">{n(row.ensemble_mean.rmse)}</td><td className="py-2 px-3 text-right">{n(row.persistence.rmse)}</td>
            <td className="py-2 px-3 text-right">{n(row.ensemble_mean.mse_skill_score_vs_persistence)}</td><td className="py-2 px-3 text-right">{n(row.crps)}</td>
            <td className="py-2 px-3 text-right">{n(row.spread_rms)}</td><td className="py-2 px-3 text-right">{n(row.spread_skill_ratio)}</td>
          </tr>)}</tbody>
        </table></div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {selected.metrics.map(row => { const rank = rankByKey.get(`${row.variable}:${row.lead_hours}`); return <div key={`rank:${row.variable}:${row.lead_hours}`} className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex justify-between text-xs"><span className="text-slate-300">Rank frequency · {row.variable} · +{row.lead_hours} h</span><span className="text-sky-400">diagnostic only</span></div>
          <div className="h-28 flex items-end gap-1 mt-4 border-b border-slate-700" aria-label={`Area-weighted rank frequencies for ${row.variable} at ${row.lead_hours} hours`}>
            {(rank?.area_weighted_frequency || []).map((v, i) => <div key={i} className="flex-1 bg-sky-500/60 min-w-[3px]" style={{height: `${Math.max(1, v * 100)}%`}} title={`rank ${i}: ${v.toPrecision(4)}`} />)}
          </div>
          <p className="text-[10px] text-slate-500 mt-2">Truth rank 0…M; ties are fractionally allocated. Shape is a calibration diagnostic, not a pass/fail test.</p>
        </div>; })}
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2 text-[11px]">
          <h3 className="text-sm font-semibold text-slate-200">Exact evaluation scope</h3>
          <dl className="grid grid-cols-[9rem_1fr] gap-y-1 text-slate-500"><dt>window</dt><dd className="text-slate-300">{selected.scope.split_start} — {selected.scope.split_end}</dd><dt>split</dt><dd>{selected.scope.split}</dd><dt>role</dt><dd>{selected.scope.evaluation_role}</dd><dt>pressure</dt><dd>{selected.scope.level_hpa} hPa</dd><dt>domain</dt><dd className="font-mono">{JSON.stringify(selected.scope.bounds)}</dd><dt>weighting</dt><dd>{selected.scope.area_weighting}</dd></dl>
        </div>
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2 text-[11px]">
          <h3 className="text-sm font-semibold text-slate-200">Content identities</h3>
          {Object.entries(selected.provenance).filter(([k]) => k !== 'forecast').map(([k,v]) => <div key={k} className="flex justify-between gap-3"><span className="text-slate-500">{k}</span><span className="font-mono text-slate-300" title={String(v)}>{shortHash(v)}</span></div>)}
          <div className="border-t border-slate-800 pt-2 text-slate-500">model: <span className="text-slate-300">{selected.provenance.forecast?.provider || 'not embedded'} {selected.provenance.forecast?.model || ''} {selected.provenance.forecast?.model_version || ''}</span></div>
          <div className="text-slate-500">checkpoint: <span className="font-mono text-slate-300" title={selected.provenance.forecast?.checkpoint_sha256}>{shortHash(selected.provenance.forecast?.checkpoint_sha256)}</span></div>
        </div>
      </section>

      <section className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
        <h3 className="text-xs font-semibold text-amber-300 flex items-center gap-2"><AlertTriangle className="w-4 h-4" /> Claim boundaries</h3>
        {selected.claim_boundaries.map((c, i) => <p key={i} className="text-[11px] text-amber-200/70 mt-2 leading-relaxed">{c}</p>)}
      </section>
    </>}
  </div>;
}
