/** TG11.1 — read-only access to the existing cross-domain analysis engine. */

import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Ban, CheckCircle2, FlaskConical, Loader2 } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  selectedRecord: types.ChannelRecordSelection | null;
  onError?: (message: string) => void;
  onAcquire?: () => void;
}

const parseLags = (text: string): number[] => text.split(',')
  .map((value) => Number(value.trim())).filter((value) => Number.isFinite(value));

const DomainAnalysisView: React.FC<Props> = ({ selectedRecord, onError, onAcquire }) => {
  const precedenceAdmissible = Boolean(
    selectedRecord?.record.domain_limits.precedence_admissible);
  const [operation, setOperation] = useState<types.DomainAnalysisOperation>('association');
  const [lags, setLags] = useState('1');
  const [estimator, setEstimator] = useState('mutual_information');
  const [bins, setBins] = useState(4);
  const [surrogates, setSurrogates] = useState(499);
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(20260827);
  const [embargo, setEmbargo] = useState(1);
  const [trainRatio, setTrainRatio] = useState(0.6);
  const [studyId, setStudyId] = useState('workbench-domain-gate');
  const [policyParams, setPolicyParams] = useState('{}');
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<types.DomainAnalysisResponse | null>(null);
  const [capabilities, setCapabilities] = useState<types.DomainAnalysisCapabilities | null>(null);

  useEffect(() => {
    apiService.getDomainAnalysisCapabilities().then(setCapabilities).catch((error) => {
      onError?.(error instanceof Error ? error.message : String(error));
    });
  }, [onError]);

  useEffect(() => {
    setOperation(precedenceAdmissible ? 'precedence' : 'association');
    setResponse(null);
  }, [selectedRecord, precedenceAdmissible]);

  const parsedLags = useMemo(() => parseLags(lags), [lags]);
  const result = response?.result;
  const rows = Array.isArray(result?.results) ? result.results : [];
  const warnings = Array.isArray(result?.warnings) ? result.warnings : [];

  const run = async () => {
    if (!selectedRecord) return;
    setBusy(true);
    setResponse(null);
    try {
      const lagPolicyParams = JSON.parse(policyParams);
      if (!lagPolicyParams || Array.isArray(lagPolicyParams)
          || typeof lagPolicyParams !== 'object') {
        throw new Error('Lag-policy parameters must be a JSON object.');
      }
      const configuration: Record<string, any> = {
        lags: parsedLags, estimator, bins, n_surrogates: surrogates, alpha, seed,
        correction: 'benjamini_yekutieli', lag_policy_params: lagPolicyParams,
      };
      if (operation === 'domain_gate') {
        Object.assign(configuration, {
          study_id: studyId, train_ratio: trainRatio, embargo_frames: embargo,
          require_advection_floor: false,
        });
      }
      setResponse(await apiService.runDomainAnalysis(selectedRecord, operation, configuration));
    } catch (error) {
      onError?.(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  if (!selectedRecord) {
    return (
      <section className="space-y-4 animate-fadeIn" aria-labelledby="analysis-workbench-title">
        <header>
          <h2 id="analysis-workbench-title" className="text-xl font-bold text-white flex items-center gap-2">
            <FlaskConical className="text-teal-400 w-5 h-5" aria-hidden="true" /> Cross-domain analysis
          </h2>
        </header>
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-5 max-w-3xl">
          <p className="text-sm text-amber-200">Choose and admit a channel record before analysing it.</p>
          <p className="text-xs text-amber-300/70 mt-2">
            The full original file is retained so analysis never runs against the capped preview.
          </p>
          <button type="button" onClick={onAcquire}
            className="mt-4 px-3 py-2 rounded bg-amber-500/20 text-amber-100 text-sm hover:bg-amber-500/30">
            Go to Acquire
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-5 animate-fadeIn" aria-labelledby="analysis-workbench-title" aria-busy={busy}>
      <header>
        <h2 id="analysis-workbench-title" className="text-xl font-bold text-white flex items-center gap-2">
          <FlaskConical className="text-teal-400 w-5 h-5" aria-hidden="true" /> Cross-domain analysis
        </h2>
        <p className="text-sm text-slate-400 max-w-4xl mt-1">
          Run the accepted engine against the complete selected record. This is read-only compute:
          it records no evidence, moves no rung, and does not open held-out data.
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Selected record</p>
            <p className="text-sm text-slate-100 mt-1">{selectedRecord.record.source_name}</p>
            <p className="text-xs text-teal-400">{selectedRecord.record.domain} · {selectedRecord.record.n_rows} frames</p>
          </div>

          <fieldset className="space-y-2">
            <legend className="text-xs text-slate-400 mb-1">Operation</legend>
            {(['association', 'precedence', 'domain_gate'] as const).map((value) => {
              const refused = value === 'precedence' && !precedenceAdmissible;
              return <label key={value} className={`flex gap-2 text-sm ${refused ? 'text-slate-600' : 'text-slate-200'}`}>
                <input type="radio" name="analysis-operation" value={value}
                  checked={operation === value} disabled={refused}
                  onChange={() => { setOperation(value); setResponse(null); }} />
                <span>{value === 'domain_gate' ? 'Domain gate' : value[0].toUpperCase() + value.slice(1)}</span>
              </label>;
            })}
          </fieldset>

          {!precedenceAdmissible && (
            <div className="flex gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-3">
              <Ban className="w-4 h-4 shrink-0" aria-hidden="true" />
              <span>R21: this domain declares no admissible lag floor. Association remains available; precedence is refused.</span>
            </div>
          )}

          <label className="block text-xs text-slate-400">Lags in frames
            <input value={lags} onChange={(e) => setLags(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200"
              aria-describedby="lag-family-help" />
          </label>
          <p id="lag-family-help" className="text-[11px] text-slate-600">Comma-separated, positive, unique and increasing.</p>

          <label className="block text-xs text-slate-400">Estimator
            <select value={estimator} onChange={(e) => setEstimator(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200">
              <option value="mutual_information">Mutual information</option>
              <option value="transfer_entropy">Transfer entropy</option>
            </select>
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Bins
              <input type="number" min={2} value={bins} onChange={(e) => setBins(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <label className="text-xs text-slate-400">Surrogates
              <input type="number" min={1} value={surrogates} onChange={(e) => setSurrogates(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <label className="text-xs text-slate-400">Alpha
              <input type="number" min="0" max="1" step="0.01" value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <label className="text-xs text-slate-400">Seed
              <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
          </div>

          {operation === 'domain_gate' && <div className="space-y-3 border-t border-slate-800 pt-3">
            <label className="block text-xs text-slate-400">Study label
              <input value={studyId} onChange={(e) => setStudyId(e.target.value)}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Train ratio
                <input type="number" min="0" max="1" step="0.05" value={trainRatio}
                  onChange={(e) => setTrainRatio(Number(e.target.value))}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
              </label>
              <label className="text-xs text-slate-400">Embargo frames
                <input type="number" min={1} value={embargo} onChange={(e) => setEmbargo(Number(e.target.value))}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
              </label>
            </div>
          </div>}

          <label className="block text-xs text-slate-400">Lag-policy parameters (JSON)
            <textarea value={policyParams} onChange={(e) => setPolicyParams(e.target.value)} rows={2}
              className="mt-1 w-full font-mono bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>

          <button type="button" onClick={() => void run()} disabled={busy || parsedLags.length === 0}
            className="w-full flex justify-center items-center gap-2 rounded bg-teal-600 px-4 py-2 text-sm text-white hover:bg-teal-500 disabled:opacity-50">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <FlaskConical className="w-4 h-4" aria-hidden="true" />}
            {busy ? 'Running full record…' : 'Run read-only analysis'}
          </button>
        </div>

        <div className="xl:col-span-2 space-y-4">
          {!response && <div className="border border-dashed border-slate-800 rounded-xl p-8 text-sm text-slate-500">
            {capabilities?.claim_boundary || 'Loading the analysis claim boundary…'}
          </div>}

          {response && <>
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" aria-hidden="true" />
                <h3 className="font-semibold text-slate-100">Analysis completed</h3>
                {result?.verdict && <span className={`text-xs px-2 py-1 rounded border ${
                  result.verdict === 'PASS' ? 'text-emerald-300 border-emerald-500/30' :
                  result.verdict === 'FAIL' ? 'text-sky-300 border-sky-500/30' :
                  result.verdict === 'INVALID' ? 'text-amber-300 border-amber-500/30' :
                  'text-slate-300 border-slate-600'}`}>{result.verdict}</span>}
              </div>
              <dl className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div><dt className="text-slate-500">claim boundary</dt><dd className="text-slate-200">{result?.claim_boundary || response.operation}</dd></div>
                <div><dt className="text-slate-500">tests</dt><dd className="text-slate-200">{result?.n_tests ?? result?.train?.n_tests ?? '—'}</dd></div>
                <div><dt className="text-slate-500">significant</dt><dd className="text-slate-200">{result?.n_significant ?? result?.gate?.replicated?.length ?? '—'}</dd></div>
                <div><dt className="text-slate-500">correction</dt><dd className="text-slate-200">Benjamini–Yekutieli</dd></div>
              </dl>
              {response.source.content_sha256 !== selectedRecord.record.content_sha256 && (
                <p className="text-sm text-rose-300 flex gap-2">
                  <Ban className="w-4 h-4 shrink-0" aria-hidden="true" />
                  The analysed bytes are not the selected record's bytes. Discard this result and
                  re-admit the file.
                </p>)}
              {response.source.content_sha256 === selectedRecord.record.content_sha256
                && response.source.frames !== selectedRecord.record.n_rows && (
                <p className="text-sm text-amber-300 flex gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
                  The same file re-admitted to {response.source.frames} frames where the record
                  reports {selectedRecord.record.n_rows}. The two readings disagree; do not read
                  this result as being about the record on screen.
                </p>)}
              {(result?.n_significant === 0 || (result?.verdict === 'FAIL')) &&
                <p className="text-sm text-sky-300">There is nothing here in the declared family under this design.</p>}
              {result?.verdict === 'INVALID' &&
                <p className="text-sm text-amber-300">INVALID is not FAIL: the design itself did not hold, so this run is
                  no evidence of absence. Read the refusal below and re-declare the family before running it again.</p>}
            </div>

            {warnings.length > 0 && <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 space-y-2">
              {warnings.map((warning: string, index: number) => <p key={index} className="flex gap-2 text-xs text-amber-200">
                <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />{warning}
              </p>)}
            </div>}

            {rows.length > 0 && <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-slate-950/70 text-slate-500"><tr>
                  <th scope="col" className="text-left p-3">Family member</th>
                  <th scope="col" className="text-right p-3">Effect (nats)</th>
                  <th scope="col" className="text-right p-3">p</th>
                  <th scope="col" className="text-right p-3">q</th>
                  <th scope="col" className="text-left p-3">Decision</th>
                </tr></thead>
                <tbody>{rows.map((row: Record<string, any>, index: number) => <tr key={row.label || index} className="border-t border-slate-800">
                  <td className="p-3 font-mono text-slate-200">{row.label}</td>
                  <td className="p-3 text-right text-slate-300">{String(row.excess_nats)}</td>
                  <td className="p-3 text-right text-slate-300">{String(row.p_value)}</td>
                  <td className="p-3 text-right text-slate-300">{String(row.q_value)}</td>
                  <td className="p-3 text-slate-300">{row.significant ? 'significant after correction' : 'not significant'}</td>
                </tr>)}</tbody>
              </table>
            </div>}

            <p className="text-xs text-slate-500 border-l-2 border-slate-700 pl-3">{response.claim_boundary}</p>
          </>}
        </div>
      </div>
    </section>
  );
};

export default DomainAnalysisView;
