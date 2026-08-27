/** TG11.4 — structure mining: find a configuration that recurs, and pay for having looked.
 *
 * The panel is laid out in the order the science has to happen in, because the order is the
 * only thing that makes the result mean anything. A field is admitted and the server extracts
 * its features; the family that a mining pass would examine is priced *before* anything is
 * mined; the match tolerance is measured from replicates rather than typed; candidates are
 * mined on the training frames; the confirmatory family is frozen against held-out frames that
 * have not been opened; and only then are those frames spent, once.
 *
 * Two controls are deliberately absent, and neither absence is a UI decision.
 *
 * There is no way to enter a feature. A motif is a configuration of extracted features, so a
 * panel that let a researcher place three points would let them draw the shape they wanted
 * confirmed — and every number downstream would be computed correctly from a fabricated
 * premise. The request shapes have no field for one.
 *
 * There is no way to type a tolerance. It decides which configurations count as repeats, and a
 * wrong one is not subtle: measured the tempting way it came out fifty times too wide in this
 * tree's own benchmark, at which point every triangle matched every other. What travels is the
 * digest of a calibration the server performed.
 *
 * Everything shown as a verdict — the price, the candidates, the corrected q-values, the
 * invariance report — is rendered whole from the response. This panel computes nothing.
 */

import { useCallback, useEffect, useState } from 'react';
import { Boxes, Loader2, Lock, ShieldCheck, Sparkles } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  studyId: string;
  onError?: (message: string) => void;
}

type Busy = '' | 'admit' | 'price' | 'tolerance' | 'generate' | 'freeze' | 'confirm'
  | 'publish' | 'transfer' | 'audit';

const SHORT = (digest?: string | null) => (digest ? `${digest.slice(0, 12)}…` : '—');

const FIELD = 'mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200';
const CARD = 'bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4';
const BUTTON = 'w-full px-3 py-2 rounded bg-indigo-500/20 text-indigo-100 text-sm '
  + 'hover:bg-indigo-500/30 disabled:opacity-40';

const StructureMiningView: React.FC<Props> = ({ studyId, onError }) => {
  const [capabilities, setCapabilities] = useState<types.MiningCapabilities | null>(null);
  const [records, setRecords] = useState<{ record_id: string; declaration: Record<string, unknown> }[]>([]);
  const [admitted, setAdmitted] = useState<types.AdmittedRecord | null>(null);
  const [price, setPrice] = useState<types.FamilyPrice | null>(null);
  const [tolerance, setTolerance] = useState<types.ToleranceReceipt | null>(null);
  const [generated, setGenerated] = useState<types.MiningGeneration | null>(null);
  const [seal, setSeal] = useState<types.MiningSeal | null>(null);
  const [confirmation, setConfirmation] = useState<types.MiningConfirmation | null>(null);
  const [published, setPublished] = useState<types.PublishedMotif | null>(null);
  const [transfer, setTransfer] = useState<types.TransferReceipt | null>(null);
  const [audit, setAudit] = useState<types.InvarianceAudit | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [domain, setDomain] = useState('reanalysis');
  const [dataset, setDataset] = useState('');
  const [variable, setVariable] = useState('amplitude');
  const [representation, setRepresentation] = useState('identity');
  const [extraction, setExtraction] = useState('{\n  "n_surrogates": 199\n}');

  const [recordId, setRecordId] = useState('');
  const [replicateRecordId, setReplicateRecordId] = useState('');
  const [replicateFrames, setReplicateFrames] = useState('0,1,2,3');
  const [declaredAs, setDeclaredAs] = useState('');

  const [size, setSize] = useState(3);
  const [matcher, setMatcher] = useState('relative_geometry');
  const [surrogates, setSurrogates] = useState(199);
  const [trainRatio, setTrainRatio] = useState(0.5);
  const [embargo, setEmbargo] = useState(0);
  const [publishedDigest, setPublishedDigest] = useState('');

  const [targetRecordId, setTargetRecordId] = useState('');
  const [targetDomain, setTargetDomain] = useState('');
  const [referenceFrame, setReferenceFrame] = useState(0);
  const [presentationFrames, setPresentationFrames] = useState('4,5');
  const [presentationTransform, setPresentationTransform] = useState('rotation');

  const [busy, setBusy] = useState<Busy>('');

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  const refresh = useCallback(() => {
    apiService.listMiningRecords().then((body) => setRecords(body.records)).catch(fail);
  }, [fail]);

  useEffect(() => {
    apiService.getMiningCapabilities().then(setCapabilities).catch(fail);
    refresh();
  }, [fail, refresh]);

  const run = (): types.MiningRunRequest => ({
    record_id: recordId,
    tolerance_sha256: tolerance?.tolerance_sha256 ?? '',
    size, matcher, n_surrogates: surrogates,
    train_ratio: trainRatio, embargo_frames: embargo,
    study_id: studyId
  });

  const guard = async (which: Busy, work: () => Promise<void>) => {
    setBusy(which);
    try { await work(); } catch (error) { fail(error); } finally { setBusy(''); }
  };

  const admit = () => guard('admit', async () => {
    if (!file) return;
    const body = await apiService.admitRecord(file, domain, dataset, variable,
      representation, extraction);
    setAdmitted(body);
    setRecordId(body.record_id);
    refresh();
  });

  const priceIt = () => guard('price', async () => {
    const frames = admitted?.n_frames ?? 6;
    const features = admitted?.feature_counts?.[0] ?? 6;
    setPrice(await apiService.priceFamily(Math.max(2, Math.floor(frames * trainRatio)),
      features, size, surrogates));
  });

  const calibrate = () => guard('tolerance', async () => {
    const frames = replicateFrames.split(',').map((v) => Number(v.trim()))
      .filter((v) => Number.isFinite(v));
    setTolerance(await apiService.calibrateTolerance(
      replicateRecordId || recordId, frames, declaredAs));
  });

  const generate = () => guard('generate', async () => {
    setGenerated(await apiService.generateMotifs(run()));
  });

  const freeze = () => guard('freeze', async () => {
    setSeal(await apiService.freezeMotifs(run()));
  });

  const confirm = () => guard('confirm', async () => {
    if (!seal) return;
    setConfirmation(await apiService.confirmMotifs(
      seal.seal_sha256, publishedDigest || undefined));
  });

  const publish = () => guard('publish', async () => {
    if (!seal) return;
    setPublished(await apiService.publishMotif(seal.seal_sha256, seal.frozen_labels[0]));
  });

  const sendTransfer = () => guard('transfer', async () => {
    if (!published) return;
    setTransfer(await apiService.transferMotif(
      published.motif_sha256, published.motif_sha256, targetRecordId, targetDomain));
  });

  const auditMatchers = () => guard('audit', async () => {
    const frames = presentationFrames.split(',').map((v) => Number(v.trim()))
      .filter((v) => Number.isFinite(v));
    const replicates = replicateFrames.split(',').map((v) => Number(v.trim()))
      .filter((v) => Number.isFinite(v));
    setAudit(await apiService.auditInvariance(
      replicateRecordId || recordId, referenceFrame, replicates,
      frames.map((frame) => ({ frame, transforms: [presentationTransform] }))));
  });

  return (
    <section className="space-y-5 animate-fadeIn" aria-labelledby="mining-title">
      <header>
        <h2 id="mining-title" className="text-xl font-bold text-white flex items-center gap-2">
          <Boxes className="text-indigo-400 w-5 h-5" aria-hidden="true" /> Structure mining
        </h2>
        <p className="text-sm text-slate-400 max-w-4xl mt-1">
          Mine frames for a configuration that recurs, price the search before running it, and
          confirm what was frozen on frames it was not mined from. Features are extracted here
          from an admitted field — nothing on this panel can supply one, and the match
          tolerance is measured rather than chosen.
        </p>
        {capabilities && (
          <p className="text-xs text-slate-500 max-w-4xl mt-2">{capabilities.claim_boundary}</p>
        )}
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        {/* ------------------------------------------------------------ admit and price */}
        <div className={CARD}>
          <p className="text-xs uppercase tracking-wide text-slate-500">Admit a field</p>
          <label className="block text-xs text-slate-400">Frames (.npy, one stack)
            <input type="file" accept=".npy"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} className={FIELD} />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Domain
              <select value={domain} onChange={(e) => setDomain(e.target.value)} className={FIELD}>
                {(capabilities?.minable_domains ?? []).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
            <label className="text-xs text-slate-400">Variable
              <input value={variable} onChange={(e) => setVariable(e.target.value)} className={FIELD} />
            </label>
            <label className="text-xs text-slate-400">Dataset
              <input value={dataset} onChange={(e) => setDataset(e.target.value)} className={FIELD} />
            </label>
            <label className="text-xs text-slate-400">Representation
              <input value={representation} onChange={(e) => setRepresentation(e.target.value)} className={FIELD} />
            </label>
          </div>
          <label className="block text-xs text-slate-400">Extraction settings (JSON)
            <textarea value={extraction} onChange={(e) => setExtraction(e.target.value)} rows={3}
              className={`${FIELD} font-mono text-xs`} />
          </label>
          <button type="button" onClick={admit} disabled={busy !== '' || !file || !dataset}
            className={BUTTON}>
            {busy === 'admit' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Admit and extract
          </button>

          {admitted && (
            <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-2">
              <p className="text-xs text-slate-300">
                {admitted.n_frames} frames, <span className="font-mono">{SHORT(admitted.record_id)}</span>
              </p>
              <p className="text-xs text-slate-400">
                Features per frame: {admitted.feature_counts.join(', ')}
              </p>
              {!admitted.minable && (
                <p className="text-xs text-amber-200">{admitted.why_not_minable}</p>
              )}
              <p className="text-[11px] text-slate-500">{admitted.note}</p>
            </div>
          )}

          <label className="block text-xs text-slate-400">Record digest
            <input value={recordId} onChange={(e) => setRecordId(e.target.value)}
              className={`${FIELD} font-mono text-xs`} />
          </label>
          <p className="text-[11px] text-slate-500">
            {records.length} admitted {records.length === 1 ? 'field' : 'fields'} in this store.
          </p>

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Price it first</p>
            <button type="button" onClick={priceIt} disabled={busy !== ''} className={BUTTON}>
              {busy === 'price' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Price the search
            </button>
            {price && (
              <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
                <p className="text-xs text-slate-300">
                  Family of {price.generate.family_size}; {price.generate.n_surrogates} surrogates
                  give a p-value floor of {price.generate.p_value_floor.toFixed(4)} and about{' '}
                  {price.generate.surrogates_required} would be needed.
                </p>
                <p className="text-xs text-slate-400">
                  {price.split_is_not_optional
                    ? 'Unaffordable in one stage — the generate/confirm split is the only shape that can be paid for.'
                    : 'Affordable in one stage.'}
                </p>
                <p className="text-[11px] text-slate-500">{price.confirmatory_ceiling.reading}</p>
              </div>
            )}
          </div>
        </div>

        {/* --------------------------------------------------- calibrate, mine and freeze */}
        <div className={CARD}>
          <p className="text-xs uppercase tracking-wide text-slate-500">Measure the tolerance</p>
          <label className="block text-xs text-slate-400">Replicate record digest (blank = the record above)
            <input value={replicateRecordId} onChange={(e) => setReplicateRecordId(e.target.value)}
              className={`${FIELD} font-mono text-xs`} />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Replicate frames
              <input value={replicateFrames} onChange={(e) => setReplicateFrames(e.target.value)}
                className={FIELD} />
            </label>
            <label className="text-xs text-slate-400">Matcher
              <select value={matcher} onChange={(e) => setMatcher(e.target.value)} className={FIELD}>
                {(capabilities?.matchers ?? []).map((entry) => (
                  <option key={entry.name} value={entry.name}>{entry.name}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="block text-xs text-slate-400">These frames are replicates of…
            <input value={declaredAs} onChange={(e) => setDeclaredAs(e.target.value)}
              className={FIELD} placeholder="one configuration under different noise" />
          </label>
          <button type="button" onClick={calibrate}
            disabled={busy !== '' || !declaredAs || !(replicateRecordId || recordId)}
            className={BUTTON}>
            {busy === 'tolerance' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Calibrate
          </button>
          {tolerance && (
            <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
              <p className="text-xs text-slate-300">
                {tolerance.receipt.tolerance.value.toExponential(3)} over{' '}
                {tolerance.receipt.tolerance.n_replicates} replicates,{' '}
                <span className="font-mono">{SHORT(tolerance.tolerance_sha256)}</span>
              </p>
              <p className="text-[11px] text-slate-500">{tolerance.receipt.tolerance.basis}</p>
              <p className="text-[11px] text-amber-200/80">{tolerance.note}</p>
            </div>
          )}

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Mine the training frames</p>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Configuration size
                <select value={size} onChange={(e) => setSize(Number(e.target.value))} className={FIELD}>
                  {Object.keys(capabilities?.sizes ?? { 3: '' }).map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-slate-400">Surrogates
                <input type="number" value={surrogates} min={19}
                  onChange={(e) => setSurrogates(Number(e.target.value))} className={FIELD} />
              </label>
              <label className="text-xs text-slate-400">Train ratio
                <input type="number" step="0.05" value={trainRatio} min={0.1} max={0.9}
                  onChange={(e) => setTrainRatio(Number(e.target.value))} className={FIELD} />
              </label>
              <label className="text-xs text-slate-400">Embargo frames
                <input type="number" value={embargo} min={0}
                  onChange={(e) => setEmbargo(Number(e.target.value))} className={FIELD} />
              </label>
            </div>
            <button type="button" onClick={generate}
              disabled={busy !== '' || !recordId || !tolerance} className={BUTTON}>
              {busy === 'generate' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Generate candidates
            </button>
            {generated && (
              <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
                <p className="text-xs text-slate-300">
                  {generated.mining.n_candidates} candidates from a family of{' '}
                  {generated.mining.generate_family_size};{' '}
                  {generated.mining.intransitive_pairs} intransitive pairs.
                </p>
                <ul className="text-xs text-slate-400 space-y-1">
                  {generated.chosen.map((candidate) => (
                    <li key={candidate.label} className="font-mono">
                      {candidate.label} — support {candidate.support}/{generated.mining.scenes.length}
                    </li>
                  ))}
                </ul>
                <p className="text-[11px] text-slate-500">{generated.claim_boundary}</p>
              </div>
            )}
            <button type="button" onClick={freeze}
              disabled={busy !== '' || !generated} className={BUTTON}>
              {busy === 'freeze' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null}
              <Lock className="w-4 h-4 inline mx-1" aria-hidden="true" /> Freeze before opening
            </button>
          </div>
        </div>

        {/* ------------------------------------------------------- confirm, publish, audit */}
        <div className={CARD}>
          <p className="text-xs uppercase tracking-wide text-slate-500">Where it stands</p>
          {seal ? (
            <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
              <p className="text-xs text-slate-300">
                Sealed <span className="font-mono">{SHORT(seal.seal_sha256)}</span> —{' '}
                {seal.frozen_labels.length} frozen.
              </p>
              <ul className="text-xs text-slate-400 font-mono">
                {seal.frozen_labels.map((frozen) => <li key={frozen}>{frozen}</li>)}
              </ul>
              <p className="text-[11px] text-amber-200/80">{seal.publication_note}</p>
            </div>
          ) : (
            <p className="text-xs text-slate-500">
              Nothing is frozen yet. A confirmation runs only against a family that was fixed
              before the held-out frames were opened.
            </p>
          )}

          <label className="block text-xs text-slate-400">Published seal digest (optional)
            <input value={publishedDigest} onChange={(e) => setPublishedDigest(e.target.value)}
              className={`${FIELD} font-mono text-xs`} />
          </label>
          <button type="button" onClick={confirm} disabled={busy !== '' || !seal} className={BUTTON}>
            {busy === 'confirm' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null}
            <ShieldCheck className="w-4 h-4 inline mx-1" aria-hidden="true" /> Open the held-out frames once
          </button>

          {confirmation && (
            <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-2">
              <table className="w-full text-xs text-slate-300">
                <caption className="sr-only">Frozen motifs and their corrected q-values</caption>
                <thead>
                  <tr className="text-slate-500 text-left">
                    <th scope="col">Motif</th><th scope="col">p</th>
                    <th scope="col">q</th><th scope="col">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {confirmation.receipt.labels.map((label, index) => (
                    <tr key={label} className="border-t border-slate-800">
                      <td className="font-mono py-1">{label}</td>
                      <td>{confirmation.receipt.p_values[index].toFixed(4)}</td>
                      <td>{confirmation.receipt.adjusted[index].toFixed(4)}</td>
                      <td>{confirmation.receipt.supports[index]}/{confirmation.receipt.n_scenes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {confirmation.vacuous.length > 0 && (
                <p className="text-xs text-amber-200">
                  Vacuous, so not confirmed by anything: {confirmation.vacuous.join(', ')}
                </p>
              )}
              <p className="text-[11px] text-slate-500">{confirmation.receipt.claim_boundary}</p>
            </div>
          )}

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Publish and transfer</p>
            <button type="button" onClick={publish} disabled={busy !== '' || !seal} className={BUTTON}>
              {busy === 'publish' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Publish the definition
            </button>
            {published && (
              <p className="text-xs text-slate-400">
                <span className="font-mono">{SHORT(published.motif_sha256)}</span> — carries the
                origin licence: {published.origin_licence}
              </p>
            )}
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Target record digest
                <input value={targetRecordId} onChange={(e) => setTargetRecordId(e.target.value)}
                  className={`${FIELD} font-mono text-xs`} />
              </label>
              <label className="text-xs text-slate-400">Target domain
                <input value={targetDomain} onChange={(e) => setTargetDomain(e.target.value)}
                  className={FIELD} />
              </label>
            </div>
            <button type="button" onClick={sendTransfer}
              disabled={busy !== '' || !published || !targetRecordId || !targetDomain}
              className={BUTTON}>
              {busy === 'transfer' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null}
              <Sparkles className="w-4 h-4 inline mx-1" aria-hidden="true" /> Open the target once
            </button>
            {transfer && (
              <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
                <p className="text-xs text-slate-300">
                  {transfer.receipt.n_matches} matches across {transfer.receipt.support} of{' '}
                  {transfer.receipt.n_scenes} target scenes, from{' '}
                  {transfer.receipt.n_examined} configurations examined.
                </p>
                <p className="text-[11px] text-slate-500">{transfer.receipt.claim_boundary}</p>
              </div>
            )}
          </div>

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Audit the matchers</p>
            <div className="grid grid-cols-3 gap-3">
              <label className="text-xs text-slate-400">Reference
                <input type="number" value={referenceFrame} min={0}
                  onChange={(e) => setReferenceFrame(Number(e.target.value))} className={FIELD} />
              </label>
              <label className="text-xs text-slate-400">Presentations
                <input value={presentationFrames}
                  onChange={(e) => setPresentationFrames(e.target.value)} className={FIELD} />
              </label>
              <label className="text-xs text-slate-400">Transform
                <select value={presentationTransform}
                  onChange={(e) => setPresentationTransform(e.target.value)} className={FIELD}>
                  {(capabilities?.transforms ?? []).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </label>
            </div>
            <button type="button" onClick={auditMatchers}
              disabled={busy !== '' || !(replicateRecordId || recordId)} className={BUTTON}>
              {busy === 'audit' ? <Loader2 className="w-4 h-4 animate-spin inline" aria-hidden="true" /> : null} Measure declared invariance
            </button>
            {audit && (
              <div className="bg-slate-950/60 border border-slate-800 rounded p-3 space-y-1">
                {Object.entries(audit.reports).map(([name, report]) => (
                  <p key={name} className="text-xs text-slate-300">
                    <span className="font-mono">{name}</span> — declared{' '}
                    {report.declared.join(', ') || 'nothing'}; measured{' '}
                    {report.measured.join(', ') || 'nothing'}
                    {report.overclaimed.length > 0
                      ? ` — overclaimed ${report.overclaimed.join(', ')}`
                      : ''}
                  </p>
                ))}
                <p className="text-[11px] text-slate-500">
                  {audit.declared_transforms_are_the_callers}
                </p>
                <p className="text-[11px] text-slate-500">{audit.claim_boundary}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default StructureMiningView;
