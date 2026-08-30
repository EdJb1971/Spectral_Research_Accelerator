/** TG13.1 bounded TESS-SPOC acquisition. */
import { useEffect, useState } from 'react';
import { Download, Loader2, Search } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

const LightCurveAcquisition: React.FC<{ source: types.LightCurveSource;
  onError?: (message: string) => void;
  onCapability?: (profile: types.DatasetCapabilityProfile | null) => void }> = ({ source, onError, onCapability }) => {
  const [request, setRequest] = useState<types.LightCurveSpecRequest>({
    target_id: '261136679', sectors: [1], flux_column: 'PDCSAP_FLUX',
    quality_policy: 'quality_zero', max_products: 4, max_download_bytes: 64 * 1024 * 1024,
  });
  const [plan, setPlan] = useState<types.LightCurvePlan | null>(null);
  const [result, setResult] = useState<types.LightCurveAcquisitionResponse | null>(null);
  const [capabilities, setCapabilities] = useState<types.LightCurveCapabilities | null>(null);
  const [busy, setBusy] = useState<'inspect' | 'acquire' | null>(null);
  const fail = (error: unknown) => onError?.(error instanceof Error ? error.message : String(error));
  useEffect(() => { onCapability?.(null); }, [request]);
  useEffect(() => {
    let active = true;
    void apiService.lightCurveCapabilities()
      .then((value) => { if (active) setCapabilities(value); })
      .catch(fail);
    return () => { active = false; };
  }, []);
  const inspect = async () => { setBusy('inspect'); setPlan(null); setResult(null);
    try { setPlan(await apiService.inspectLightCurve(request)); } catch (error) { fail(error); } finally { setBusy(null); } };
  const acquire = async () => { setBusy('acquire'); setResult(null);
    try { const acquired = await apiService.acquireLightCurve(request); setResult(acquired);
      onCapability?.(acquired.capability_profile); } catch (error) { fail(error); } finally { setBusy(null); } };

  return <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
    <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
      <h3 className="text-sm font-semibold text-slate-200">3. TESS target and sectors</h3>
      <p className="text-[10px] text-slate-500">{source.archive} · {source.pipeline} calibrated LC products · opt-in via <code>{source.network_env_var}</code></p>
      <label className="text-xs text-slate-400 block">TIC identifier
        <input value={request.target_id} onChange={(e) => setRequest({ ...request, target_id: e.target.value })}
          className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 font-mono" /></label>
      <label className="text-xs text-slate-400 block">Sectors (comma separated)
        <input value={request.sectors.join(',')} onChange={(e) => setRequest({ ...request,
          sectors: e.target.value.split(',').map(Number).filter((v) => Number.isInteger(v) && v > 0) })}
          className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 font-mono" /></label>
      <label className="text-xs text-slate-400 block">Flux product
        <select value={request.flux_column} onChange={(e) => setRequest({ ...request,
          flux_column: e.target.value as types.LightCurveSpecRequest['flux_column'] })}
          className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2">
          <option>PDCSAP_FLUX</option><option>SAP_FLUX</option></select></label>
      <div className="flex gap-2">
        <button onClick={() => void inspect()} disabled={busy !== null || capabilities === null}
          className="flex-1 border border-slate-700 rounded p-2 text-xs flex justify-center gap-1">
          {busy === 'inspect' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />} Inspect metadata</button>
        <button onClick={() => void acquire()} disabled={busy !== null || !plan || !plan.within_byte_cap || !plan.within_product_cap}
          className="flex-1 bg-teal-600 disabled:opacity-50 rounded p-2 text-xs font-semibold flex justify-center gap-1">
          {busy === 'acquire' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />} Acquire</button>
      </div>
      {capabilities && <p className="text-[10px] text-slate-500">{capabilities.claim_boundary}</p>}
    </section>
    <section className="xl:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-5">
      {!plan && !result && <div className="min-h-48 flex items-center justify-center text-xs text-slate-500">Inspect resolves exact products and cost before any FITS value is transferred.</div>}
      {plan && <div className="space-y-3 text-xs"><h3 className="font-semibold text-slate-200">Metadata-only acquisition plan</h3>
        <p>{plan.target.tic_id} · ICRS {plan.target.ra_deg.toFixed(6)}, {plan.target.dec_deg.toFixed(6)}</p>
        <p className="text-slate-400">{plan.candidate_products} product(s) · {(plan.predicted_download_bytes / 1048576).toFixed(2)} MiB predicted</p>
        {plan.products.map((p) => <div key={p.data_uri} className="border border-slate-800 rounded p-2 font-mono text-[10px]">sector {p.sector} · {(p.size_bytes / 1048576).toFixed(2)} MiB · {p.filename}</div>)}
        <p className="text-[10px] text-slate-500">{plan.claim_boundary}</p></div>}
      {result && <div className="mt-4 border-t border-slate-800 pt-4 space-y-2 text-xs">
        <h3 className="font-semibold text-emerald-300">Immutable collection published</h3>
        <p>{result.collection.n_samples} samples · {result.collection.n_products} products · {result.collection.quality_flagged} flagged retained</p>
        <p className="font-mono text-[10px] break-all text-slate-500">{result.collection.collection_sha256}</p>
        {Object.entries(result.analysis_readiness).map(([k, v]) => <p key={k}><span className="text-slate-500">{k}:</span> {v}</p>)}
        <div className="border border-sky-700/40 bg-sky-950/20 rounded p-3 text-[11px] text-sky-200">
          <strong>Acquired dataset, not a study.</strong> The collection and capability profile
          are reproducible. They do not silently create an evidence study or masquerade as a
          generic channel record; only operations marked available above may consume it.
        </div>
        <p className="text-[10px] text-slate-500">{result.claim_boundary}</p></div>}
    </section>
  </div>;
};

export default LightCurveAcquisition;
