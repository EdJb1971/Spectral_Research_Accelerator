/** TG10.2: one acquisition surface, ordered domain -> acquisition -> selection. */
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle, CheckCircle, Cloud, Database, HardDrive, Loader2, Search, Server,
} from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';
import ChannelRecords from './ChannelRecords';

interface AcquisitionViewProps {
  onError?: (message: string) => void;
  selectedRecord?: types.ChannelRecordSelection | null;
  onSelectRecord?: (record: types.ChannelRecordSelection | null) => void;
}

const DEFAULT_CROP: types.ZarrCropRequest = {
  store: 'era5_0p25_6h', variables: ['temperature'],
  time_start: '2020-06-01', time_end: '2020-06-08',
  lat_min: -4, lat_max: 60, lon_min: 0, lon_max: 64,
  levels: [850, 700, 500, 300], n_levels_analysis: 4,
  analysis: {
    transform_family: 'dtcwt', wavelet: 'db2', boundary_mode: 'periodic',
    dtcwt_level1: 'near_sym_b', dtcwt_qshift: 'qshift_b',
  },
};

export const AcquisitionView: React.FC<AcquisitionViewProps> = ({
  onError, selectedRecord = null, onSelectRecord,
}) => {
  const [catalogue, setCatalogue] = useState<types.AcquisitionCatalogue | null>(null);
  const [zarrCatalogue, setZarrCatalogue] = useState<types.ZarrCatalogueResponse | null>(null);
  const [cached, setCached] = useState<types.ZarrCachedResponse | null>(null);
  const [probes, setProbes] = useState<types.ZarrProbeLedgerResponse | null>(null);
  const [probe, setProbe] = useState<types.ZarrProbeRecord | null>(null);
  const [inspection, setInspection] = useState<types.ZarrInspectResponse | null>(null);
  const [domainName, setDomainName] = useState('');
  const [acquisitionId, setAcquisitionId] = useState('');
  const [crop, setCrop] = useState<types.ZarrCropRequest>(DEFAULT_CROP);
  const [busy, setBusy] = useState(false);

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  useEffect(() => {
    let live = true;
    setBusy(true);
    Promise.all([
      apiService.listAcquisitions(), apiService.zarrCatalogue(),
      apiService.zarrCached(), apiService.zarrProbes(),
    ]).then(([acquisitions, stores, local, ledger]) => {
      if (!live) return;
      setCatalogue(acquisitions);
      setZarrCatalogue(stores);
      setCached(local);
      setProbes(ledger);
      const selectedDomain = selectedRecord
        ? acquisitions.domains.find((domain) => domain.name === selectedRecord.record.domain)
        : undefined;
      const first = selectedDomain || acquisitions.domains.find((domain) =>
        domain.acquisitions.some((item) => item.available));
      if (first) {
        setDomainName(first.name);
        const option = selectedRecord
          ? first.acquisitions.find((item) => item.shape === 'channel_table' && item.available)
          : first.acquisitions.find((item) => item.available);
        if (option) setAcquisitionId(option.id);
      }
    }).catch(fail).finally(() => { if (live) setBusy(false); });
    return () => { live = false; };
  }, [fail, selectedRecord]);

  const domain = useMemo(() => catalogue?.domains.find((row) => row.name === domainName),
    [catalogue, domainName]);
  const acquisition = useMemo(() => domain?.acquisitions.find((row) => row.id === acquisitionId),
    [domain, acquisitionId]);

  const chooseDomain = (name: string) => {
    const chosen = catalogue?.domains.find((row) => row.name === name);
    setDomainName(name);
    setAcquisitionId(chosen?.acquisitions.find((item) => item.available)?.id || '');
    setInspection(null);
    setProbe(null);
  };

  const chooseAcquisition = (option: types.AcquisitionOption) => {
    if (!option.available) return;
    setAcquisitionId(option.id);
    setInspection(null);
    setProbe(null);
    if (option.shape === 'grid_crop') {
      const base = option.store?.vertical_dim === 'level'
        ? DEFAULT_CROP
        : { ...DEFAULT_CROP, variables: [], levels: [] };
      const defaults = option.store?.extra?.acquisition_defaults || {};
      setCrop({ ...base, ...defaults, store: option.name });
    }
  };

  const runProbe = async () => {
    setBusy(true);
    try {
      const result = await apiService.zarrProbe({
        uri: crop.store, variables: crop.variables, crop,
      });
      setProbe(result.probe);
      setProbes(await apiService.zarrProbes());
    } catch (error) { fail(error); } finally { setBusy(false); }
  };

  const inspect = async () => {
    setBusy(true);
    setInspection(null);
    try { setInspection(await apiService.zarrInspect(crop)); }
    catch (error) { fail(error); }
    finally { setBusy(false); }
  };

  const reviseCrop = (patch: Partial<types.ZarrCropRequest>) => {
    setCrop((current) => ({ ...current, ...patch }));
    setInspection(null);
  };

  const reviseAnalysis = (patch: Partial<types.ZarrAnalysisRequest>) => {
    setCrop((current) => ({
      ...current, analysis: { ...current.analysis, ...patch },
    }));
    setInspection(null);
  };

  const applyRecommendedBounds = async () => {
    const bounds = inspection?.acquisition_plan.suggestions.recommended.bounds;
    if (!bounds) return;
    const planned = { ...crop, ...bounds };
    setCrop(planned);
    setInspection(null);
    setBusy(true);
    try { setInspection(await apiService.zarrInspect(planned)); }
    catch (error) { fail(error); }
    finally { setBusy(false); }
  };

  if (!catalogue || !domain) {
    return <div className="p-8 text-sm text-slate-400 flex items-center gap-2">
      <Loader2 className="w-4 h-4 animate-spin" /> Loading acquisition contracts…
    </div>;
  }

  const store = acquisition?.store;
  const verticalLabel = store?.vertical_dim === 'depth' ? 'Depths' :
    store?.vertical_dim ? `${store.vertical_dim} values` : 'Vertical selection';

  return (
    <div className="space-y-6 animate-fadeIn" aria-busy={busy}>
      <header>
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Database className="text-teal-400 w-5 h-5" /> Acquire
        </h2>
        <p className="text-sm text-slate-400 max-w-4xl">{catalogue.note}</p>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <label htmlFor="acquisition-domain" className="text-xs uppercase text-slate-500">
            1. Domain
          </label>
          <select id="acquisition-domain" value={domainName}
            onChange={(event) => chooseDomain(event.target.value)}
            className="mt-2 w-full bg-slate-950 border border-slate-800 rounded p-2 text-sm">
            {catalogue.domains.map((row) => <option key={row.name} value={row.name}>{row.name}</option>)}
          </select>
          <p className="text-xs text-slate-400 mt-3">{domain.description}</p>
          <p className="text-[10px] text-slate-600 mt-2">Licence: {domain.licence}</p>
        </div>

        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <h3 className="text-xs uppercase text-slate-500 mb-2">2. Acquisition</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {domain.acquisitions.map((option) => (
              <button key={option.id} type="button" disabled={!option.available}
                onClick={() => chooseAcquisition(option)}
                className={`text-left border rounded-lg p-3 ${acquisitionId === option.id
                  ? 'border-teal-500 bg-teal-500/10' : option.available
                    ? 'border-slate-800 hover:border-slate-600' : 'border-slate-900 opacity-60'}`}>
                <span className="text-sm text-slate-200 block">{option.name}</span>
                <span className="text-[10px] font-mono text-teal-400">{option.shape}</span>
                <span className="text-[10px] text-slate-500 block mt-1">
                  {option.available ? option.access_means : option.unavailable_reason}
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs">
        <p className="text-amber-300">{domain.domain_limits.attribution_caveat}</p>
        {domain.domain_limits.refuses.length > 0 && (
          <div className="mt-2 text-slate-400">
            This domain refuses: {domain.domain_limits.refuses.map((row) =>
              String(row.consequence || row.basis || Object.values(row)[0])).join('; ')}
          </div>
        )}
      </section>

      {acquisition?.shape === 'channel_table' && acquisition.available && (
        <ChannelRecords key={domain.name} domainName={domain.name} onError={onError}
          selectedRecord={selectedRecord?.record.domain === domain.name ? selectedRecord : null}
          onSelectRecord={onSelectRecord} />
      )}

      {acquisition?.shape === 'grid_crop' && acquisition.available && (
        <div className="space-y-5">
          {zarrCatalogue && !zarrCatalogue.network_enabled && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 flex gap-3 text-xs text-slate-400">
              <Server className="w-4 h-4 shrink-0" />
              <span>Network is off. Set <code className="text-teal-400">
                {zarrCatalogue.network_env_var}=1</code> before starting the backend. Cached crops remain local.</span>
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-semibold text-slate-200">3. Grid crop selection</h3>
              {store && <div className="text-[10px] text-slate-500 space-y-1">
                <p><Cloud className="inline w-3 h-3" /> {store.note}</p>
                <p>{store.access_means} · vertical axis {store.vertical_dim ?? 'none'}</p>
                <p>Chunking: {store.chunks.method_means}
                  {store.chunks.regional_amplification !== null
                    ? ` · ${store.chunks.regional_amplification}x recorded regional amplification` : ''}</p>
              </div>}

              <label className="text-xs text-slate-400 block">Variables (comma separated)
                <input value={crop.variables.join(',')}
                  onChange={(e) => reviseCrop({ variables: e.target.value.split(',').map(v => v.trim()).filter(Boolean) })}
                  className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2 font-mono" />
              </label>
              <div className="grid grid-cols-2 gap-2">
                <label className="text-xs text-slate-400">Start
                  <input type="date" value={crop.time_start} onChange={(e) => reviseCrop({ time_start: e.target.value })}
                    className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
                </label>
                <label className="text-xs text-slate-400">End
                  <input type="date" value={crop.time_end} onChange={(e) => reviseCrop({ time_end: e.target.value })}
                    className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2" />
                </label>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {([['lat_min', 'Lat min'], ['lat_max', 'Lat max'], ['lon_min', 'Lon min'], ['lon_max', 'Lon max']] as const)
                  .map(([key, label]) => <label key={key} className="text-xs text-slate-400">{label}
                    <input type="number" value={crop[key]}
                      onChange={(e) => reviseCrop({ [key]: Number(e.target.value) })}
                      className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2 font-mono" />
                  </label>)}
              </div>
              {store?.vertical_dim && <label className="text-xs text-slate-400 block">{verticalLabel}
                <input value={crop.levels.join(',')}
                  onChange={(e) => reviseCrop({ levels: e.target.value.split(',').map(Number).filter(Number.isFinite) })}
                  className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2 font-mono" />
              </label>}
              <div className="grid grid-cols-2 gap-2">
                <label className="text-xs text-slate-400">Analysis transform
                  <select value={crop.analysis.transform_family}
                    onChange={(e) => reviseAnalysis({
                      transform_family: e.target.value as 'swt' | 'dtcwt',
                    })}
                    className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
                    {Object.keys(zarrCatalogue?.analysis_transforms ?? { swt: {}, dtcwt: {} })
                      .map((name) => <option key={name} value={name}>{name.toUpperCase()}</option>)}
                  </select>
                </label>
                {crop.analysis.transform_family === 'swt' ?
                  <label className="text-xs text-slate-400">Wavelet
                    <select value={crop.analysis.wavelet}
                      onChange={(e) => reviseAnalysis({
                        wavelet: e.target.value as 'haar' | 'db2' | 'db3',
                      })}
                      className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
                      {['haar', 'db2', 'db3'].map((name) => <option key={name}>{name}</option>)}
                    </select>
                  </label> :
                  <label className="text-xs text-slate-400">Q-shift filters
                    <select value={crop.analysis.dtcwt_qshift}
                      onChange={(e) => reviseAnalysis({ dtcwt_qshift: e.target.value })}
                      className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
                      {['qshift_a', 'qshift_b', 'qshift_c', 'qshift_d']
                        .map((name) => <option key={name}>{name}</option>)}
                    </select>
                  </label>}
              </div>
              {crop.analysis.transform_family === 'dtcwt' ?
                <label className="text-xs text-slate-400 block">Level-1 filters
                  <select value={crop.analysis.dtcwt_level1}
                    onChange={(e) => reviseAnalysis({ dtcwt_level1: e.target.value })}
                    className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
                    {['near_sym_a', 'near_sym_b', 'legall']
                      .map((name) => <option key={name}>{name}</option>)}
                  </select>
                </label> :
                <label className="text-xs text-slate-400 block">Boundary mode
                  <select value={crop.analysis.boundary_mode}
                    onChange={(e) => reviseAnalysis({
                      boundary_mode: e.target.value as 'periodic' | 'reflect',
                    })}
                    className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-2">
                    {['periodic', 'reflect'].map((name) => <option key={name}>{name}</option>)}
                  </select>
                </label>}
              <label className="text-xs text-slate-400 block">
                Transform levels to support: {crop.n_levels_analysis}
                <input type="range" min="1" max="6" value={crop.n_levels_analysis}
                  onChange={(e) => reviseCrop({ n_levels_analysis: Number(e.target.value) })}
                  className="mt-1 w-full accent-teal-500" />
                <span className="text-[10px] text-slate-600">
                  Inspect derives both crop thresholds from this exact transform configuration.
                </span>
              </label>
              <div className="flex gap-2">
                <button onClick={() => void runProbe()} disabled={busy}
                  className="flex-1 border border-slate-700 rounded p-2 text-xs">Probe store</button>
                <button onClick={() => void inspect()} disabled={busy}
                  className="flex-1 bg-teal-600 rounded p-2 text-xs font-semibold flex justify-center gap-1">
                  {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />} Inspect
                </button>
              </div>
              {probes && <p className="text-[10px] text-slate-500">{probes.count} probes recorded;{' '}
                {probes.transcribed} transcribed.</p>}
              {probe && <div className="text-[10px] border border-slate-800 rounded p-2">
                <p>{probe.probed_on} · {probe.outcome_means}</p>
                {probe.refusal_detail && <p className="text-amber-300 mt-1">{probe.refusal_detail}</p>}
                <p className="text-slate-600 mt-1">{probe.evidence_means}</p>
              </div>}
            </div>

            <div className="xl:col-span-2 space-y-4">
              {inspection ? <>
                <div className={`rounded-xl p-5 border ${inspection.assessment.chunk_hostile
                  ? 'border-amber-500/30 bg-amber-500/5' : 'border-emerald-500/30 bg-emerald-500/5'}`}>
                  <div className="flex gap-2 items-center">
                    {inspection.assessment.chunk_hostile
                      ? <AlertTriangle className="text-amber-400" /> : <CheckCircle className="text-emerald-400" />}
                    <strong className="text-xl">{inspection.assessment.amplification.toFixed(1)}×</strong>
                    <span className="text-xs text-slate-400">amplification ·{' '}
                      {(inspection.assessment.bytes_fetched_estimate / 1e9).toFixed(2)} GB fetched for{' '}
                      {(inspection.assessment.bytes_wanted / 1e9).toFixed(2)} GB requested</span>
                  </div>
                  {inspection.assessment.warning && <p className="text-xs text-amber-300 mt-2">{inspection.assessment.warning}</p>}
                  {inspection.assessment.advice.map((line, i) => <p key={i} className="text-xs text-slate-400 mt-1">· {line}</p>)}
                </div>
                <div className={`rounded-xl p-5 border ${inspection.geometry.meets_recommended_minimum
                  ? 'border-emerald-500/30 bg-emerald-500/5'
                  : inspection.geometry.meets_absolute_minimum
                    ? 'border-amber-500/30 bg-amber-500/5'
                    : 'border-rose-500/30 bg-rose-500/5'}`}>
                  <div className="flex items-center gap-2">
                    {inspection.geometry.meets_recommended_minimum
                      ? <CheckCircle className="text-emerald-400 w-5 h-5" />
                      : <AlertTriangle className="text-amber-400 w-5 h-5" />}
                    <h4 className="text-sm font-semibold text-slate-100">
                      {String(inspection.geometry.analysis.transform_family).toUpperCase()} ·{' '}
                      {inspection.geometry.analysis.levels} levels ·{' '}
                      {inspection.geometry.verdict === 'recommended' ? 'scientifically recommended'
                        : inspection.geometry.verdict === 'technical_only'
                          ? 'technical support only' : 'insufficient support'}
                    </h4>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-3 text-center text-[11px]">
                    <div className="bg-slate-950/60 rounded p-2">
                      <span className="block text-slate-500">Current native crop</span>
                      <strong>{inspection.geometry.current_shape.join('×')}</strong>
                    </div>
                    <div className="bg-slate-950/60 rounded p-2">
                      <span className="block text-slate-500">Absolute minimum</span>
                      <strong>{inspection.geometry.absolute_minimum.shape.join('×')}</strong>
                    </div>
                    <div className="bg-slate-950/60 rounded p-2">
                      <span className="block text-slate-500">Recommended minimum</span>
                      <strong>{inspection.geometry.recommended_minimum.shape.join('×')}</strong>
                    </div>
                  </div>
                  {!inspection.geometry.meets_recommended_minimum && <p className="text-xs text-amber-200 mt-3">
                    A nonempty valid interior is not enough for a research result. Materialisation
                    is gated on the recommended threshold; changing transform or depth changes the
                    study and remains visible in the plan digest.
                  </p>}
                  {(() => {
                    const suggestion = inspection.acquisition_plan.suggestions.recommended;
                    if (!suggestion.feasible) return <p className="text-xs text-rose-300 mt-3">{suggestion.reason}</p>;
                    if (inspection.geometry.meets_recommended_minimum || !suggestion.bounds) return null;
                    return <div className="mt-3 border border-slate-700 rounded p-3 text-xs text-slate-300">
                      <p>Suggested native-grid bounds: lat {suggestion.bounds.lat_min}…{suggestion.bounds.lat_max};{' '}
                        lon {suggestion.bounds.lon_min}…{suggestion.bounds.lon_max}</p>
                      <p className="text-slate-500 mt-1">Result {suggestion.actual_shape?.join('×')} · estimated{' '}
                        {((suggestion.cost?.bytes_fetched_estimate ?? 0) / 1e9).toFixed(2)} GB fetched ·{' '}
                        {suggestion.cost?.amplification.toFixed(1)}× amplification</p>
                      <button type="button" onClick={() => void applyRecommendedBounds()} disabled={busy}
                        className="mt-2 bg-teal-600 hover:bg-teal-500 disabled:opacity-50 rounded px-3 py-2 font-semibold">
                        Use recommended bounds and re-inspect
                      </button>
                    </div>;
                  })()}
                  <details className="mt-3 text-[10px] text-slate-500">
                    <summary className="cursor-pointer">Per-level support and audit identity</summary>
                    {inspection.geometry.levels.map((level) => <p key={level.level} className="font-mono mt-1">
                      L{level.level}: support {level.support_parent_px}px · margin {level.margin_parent_px}px/side ·{' '}
                      native valid {level.valid_native_shape.join('×')}
                    </p>)}
                    <p className="font-mono mt-2 break-all">plan {inspection.acquisition_plan.plan_sha256}</p>
                    <p className="mt-1">{inspection.acquisition_plan.claim_boundary}</p>
                  </details>
                </div>
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                  <h4 className="text-sm text-slate-200 mb-2">Remote chunk structure</h4>
                  {Object.entries(inspection.structure.variables).map(([name, value]) =>
                    <p key={name} className="text-[10px] font-mono text-slate-400">{name}: shape [{value.shape.join(', ')}], chunks [{(value.chunks || []).join(', ')}]</p>)}
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded p-3">
                  <p className="text-[10px] text-slate-500">Materialise from the command line:</p>
                  <code className="text-[10px] text-teal-400 break-all">{inspection.cli}</code>
                  {!inspection.geometry.meets_recommended_minimum && <p className="text-[10px] text-amber-300 mt-2">
                    This command is complete and auditable, but it will refuse until the crop is
                    re-inspected at the recommended size.
                  </p>}
                </div>
                {/* Said before the transfer rather than discovered after it (D70): materialising
                    is the end of the road for a store no analysis path reads yet. */}
                {store && store.vertical_dim !== 'level' && <div className="border border-amber-900/60 bg-amber-950/20 rounded p-3">
                  <p className="text-[11px] text-amber-300">
                    Materialising is where this store currently stops. The crop will be cached,
                    content-addressed and reproducible from the command above, and no analysis
                    path reads a crop from it yet — the T5.2 regional-forecast route is
                    atmospheric and asks for ERA5 variables on a pressure level. Registration and
                    costing are what this store has so far; an analysis route for it is not
                    built.
                  </p>
                </div>}
              </> : <div className="min-h-[260px] bg-slate-900 border border-slate-800 rounded-xl flex flex-col items-center justify-center text-slate-500">
                <Search className="w-10 h-10 mb-2" /><p className="text-sm">No crop inspected yet</p>
              </div>}

              {cached && cached.count > 0 && <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                <h4 className="text-sm text-slate-200 flex gap-2"><HardDrive className="w-4 h-4" /> Materialised crops ({cached.count})</h4>
                {cached.crops.map((item) => <div key={item.content_key}
                  className="mt-2 border border-slate-800 rounded p-2 text-[10px] font-mono text-slate-400">
                  <p className="text-slate-200">{item.content_key}</p>
                  <p>{Object.entries(item.shape || {}).map(([key, value]) => `${key}=${value}`).join(' ')}</p>
                  {/* A crop from a non-pressure-level store is not a failed T5.2 candidate; the
                      question does not apply to it, and reporting the atmospheric verdict over
                      it read as though the crop had fallen short (D70). */}
                  {item.regional_forecast_readiness.applicable ? <>
                    <p className={item.regional_forecast_readiness.structurally_eligible
                      ? 'text-emerald-400' : 'text-amber-400'}>
                      {item.regional_forecast_readiness.structurally_eligible
                        ? 'T5.2 structure eligible' : 'T5.2 inputs incomplete'}
                    </p>
                    <p className="text-amber-300 font-sans">Prepared dataset: NO · train-only normalisation verified: NO · independent ERA5 cross-check: NOT RUN</p>
                    <p className="text-amber-300 font-sans">
                      cadence: {item.regional_forecast_readiness.cadence_verified
                        ? `${item.regional_forecast_readiness.expected_cadence_hours} h verified`
                        : 'NOT VERIFIED'} · physical lead labels: NOT AVAILABLE
                    </p>
                    <p className="text-amber-300 font-sans">
                      split contract: {item.regional_forecast_readiness.split_mode === 'calendar_boundaries'
                        ? `calendar (${item.regional_forecast_readiness.calendar_boundaries?.join(' → ')})`
                        : 'ratios (dates not frozen)'}
                    </p>
                    <p className="font-sans text-slate-600">{item.regional_forecast_readiness.claim_boundary}</p>
                  </> : <>
                    <p className="text-slate-400">
                      T5.2 readiness: NOT APPLICABLE · vertical axis {item.regional_forecast_readiness.vertical_dim}
                    </p>
                    <p className="font-sans text-slate-500">{item.regional_forecast_readiness.not_applicable_reason}</p>
                  </>}
                </div>)}
              </div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AcquisitionView;
