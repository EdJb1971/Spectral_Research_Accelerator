/** TG11.2 — preregistration first: declare the family, then open the held-out data once (R18).
 *
 * The ordering this panel shows is not the ordering that is enforced. The server refuses a
 * confirmation against a partition with no seal, and refuses a second confirmation against a
 * partition already spent, whatever this file renders — a workflow enforced by which button is
 * enabled would defeat the module while appearing to use it. What this panel is for is making
 * the ordering *legible*: what is about to be bound, what it hashed to, and the fact that
 * opening the held-out partition is a thing you get to do once.
 */

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, CheckCircle2, Lock, Loader2, ShieldAlert } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  selectedRecord: types.ChannelRecordSelection | null;
  onError?: (message: string) => void;
  onAcquire?: () => void;
}

const parseLags = (text: string): number[] => text.split(',')
  .map((value) => Number(value.trim())).filter((value) => Number.isFinite(value));

const PreregistrationView: React.FC<Props> = ({ selectedRecord, onError, onAcquire }) => {
  const [capabilities, setCapabilities] = useState<types.PreregistrationCapabilities | null>(null);
  const [partition, setPartition] = useState<types.PartitionDescription | null>(null);
  const [seals, setSeals] = useState<types.SealSummary[]>([]);
  const [sealed, setSealed] = useState<types.SealResponse | null>(null);
  const [confirmation, setConfirmation] = useState<types.ConfirmationResponse | null>(null);

  const [studyId, setStudyId] = useState('');
  const [generateLags, setGenerateLags] = useState('1,2,3,4');
  const [confirmLags, setConfirmLags] = useState('2');
  const [generateSurrogates, setGenerateSurrogates] = useState(4999);
  const [confirmSurrogates, setConfirmSurrogates] = useState(499);
  const [estimator, setEstimator] = useState('mutual_information');
  const [trainRatio, setTrainRatio] = useState(0.6);
  const [embargo, setEmbargo] = useState(4);
  const [publishedSha, setPublishedSha] = useState('');
  const [verified, setVerified] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<'' | 'partition' | 'seal' | 'confirm' | 'verify'>('');

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  const refreshSeals = useCallback(() => {
    apiService.listSeals().then((listing) => setSeals(listing.seals)).catch(fail);
  }, [fail]);

  useEffect(() => {
    apiService.getPreregistrationCapabilities().then(setCapabilities).catch(fail);
    refreshSeals();
  }, [fail, refreshSeals]);

  useEffect(() => {
    setPartition(null);
    setSealed(null);
    setConfirmation(null);
  }, [selectedRecord]);

  const describe = async () => {
    if (!selectedRecord) return;
    setBusy('partition');
    setPartition(null);
    try {
      setPartition(await apiService.describePartition(selectedRecord, trainRatio, embargo));
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  const seal = async () => {
    if (!selectedRecord) return;
    setBusy('seal');
    setSealed(null);
    setConfirmation(null);
    try {
      const response = await apiService.sealFamily(
        selectedRecord, studyId,
        { lags: parseLags(generateLags), n_surrogates: generateSurrogates, estimator },
        { lags: parseLags(confirmLags), n_surrogates: confirmSurrogates },
        trainRatio, embargo);
      setSealed(response);
      refreshSeals();
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  // The check with weight, reachable without spending anything. A seal that no longer hashes
  // to its own contents, or that disagrees with the digest published before the partition was
  // opened, should be discoverable at any time and not only at the one moment it is used.
  const verify = async (sealSha256: string) => {
    setBusy('verify');
    try {
      const body = await apiService.readSeal(sealSha256, publishedSha.trim() || undefined);
      setVerified({
        ...verified,
        [sealSha256]: body.checked_against_publication
          ? 'Matches the published digest.'
          : 'Self-consistent. No published digest was supplied, so this is not evidence about itself.',
      });
    } catch (error) {
      setVerified({ ...verified, [sealSha256]: 'Refused. This seal does not verify.' });
      fail(error);
    } finally { setBusy(''); }
  };

  const confirm = async (sealSha256: string) => {
    if (!selectedRecord) return;
    setBusy('confirm');
    setConfirmation(null);
    try {
      const response = await apiService.confirmSeal(
        sealSha256, selectedRecord.file, publishedSha.trim() || undefined);
      setConfirmation(response);
      refreshSeals();
    } catch (error) { fail(error); } finally { setBusy(''); }
  };

  if (!selectedRecord) {
    return (
      <section className="space-y-4 animate-fadeIn" aria-labelledby="preregistration-title">
        <header>
          <h2 id="preregistration-title" className="text-xl font-bold text-white flex items-center gap-2">
            <Lock className="text-teal-400 w-5 h-5" aria-hidden="true" /> Preregistration
          </h2>
        </header>
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-5 max-w-3xl">
          <p className="text-sm text-amber-200">Choose and admit a channel record before declaring a family against it.</p>
          <p className="text-xs text-amber-300/70 mt-2">
            A seal binds itself to a partition of a specific record, so there is nothing to declare until one is selected.
          </p>
          <button type="button" onClick={onAcquire}
            className="mt-4 px-3 py-2 rounded bg-amber-500/20 text-amber-100 text-sm hover:bg-amber-500/30">
            Go to Acquire
          </button>
        </div>
      </section>
    );
  }

  const receipt = confirmation?.receipt;

  return (
    <section className="space-y-5 animate-fadeIn" aria-labelledby="preregistration-title" aria-busy={busy !== ''}>
      <header>
        <h2 id="preregistration-title" className="text-xl font-bold text-white flex items-center gap-2">
          <Lock className="text-teal-400 w-5 h-5" aria-hidden="true" /> Preregistration
        </h2>
        <p className="text-sm text-slate-400 max-w-4xl mt-1">
          Fix the confirmatory family before the held-out partition is opened (R18). Nothing here is a
          result: a seal is a promise about ordering, and a confirmation receipt is an input to the
          evidence write path rather than a claim.
        </p>
        {capabilities && (
          <p className="text-xs text-slate-500 max-w-4xl mt-2">
            <span className="text-slate-400">Once is per </span>{capabilities.once_is_per}
          </p>
        )}
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        {/* ---------------------------------------------------------------- declare */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Selected record</p>
            <p className="text-sm text-slate-100 mt-1">{selectedRecord.record.source_name}</p>
            <p className="text-xs text-teal-400">
              {selectedRecord.record.domain} · {selectedRecord.record.n_rows} frames
            </p>
          </div>

          <label className="block text-xs text-slate-400">Study label
            <input value={studyId} onChange={(e) => setStudyId(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-slate-400">Train fraction
              <input type="number" min="0.1" max="0.9" step="0.05" value={trainRatio}
                onChange={(e) => setTrainRatio(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <label className="text-xs text-slate-400">Embargo frames
              <input type="number" min={0} value={embargo}
                onChange={(e) => setEmbargo(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
          </div>

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Generate — mined on train</p>
            <label className="block text-xs text-slate-400">Lags in frames
              <input value={generateLags} onChange={(e) => setGenerateLags(e.target.value)}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-slate-400">Surrogates
                <input type="number" min={1} value={generateSurrogates}
                  onChange={(e) => setGenerateSurrogates(Number(e.target.value))}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
              </label>
              <label className="text-xs text-slate-400">Estimator
                <select value={estimator} onChange={(e) => setEstimator(e.target.value)}
                  className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200">
                  <option value="mutual_information">Mutual information</option>
                  <option value="transfer_entropy">Transfer entropy</option>
                </select>
              </label>
            </div>
          </div>

          <div className="border-t border-slate-800 pt-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-500">Confirm — frozen now, tested once</p>
            <label className="block text-xs text-slate-400">Lags in frames
              <input value={confirmLags} onChange={(e) => setConfirmLags(e.target.value)}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200"
                aria-describedby="confirm-narrowing-help" />
            </label>
            <label className="block text-xs text-slate-400">Surrogates
              <input type="number" min={1} value={confirmSurrogates}
                onChange={(e) => setConfirmSurrogates(Number(e.target.value))}
                className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200" />
            </label>
            <p id="confirm-narrowing-help" className="text-[11px] text-slate-600">
              A subset of the generated lags. The confirmatory family is not a cheaper family — it is a
              smaller one, written down before the held-out partition is opened, and that is what makes
              its correction legitimate.
            </p>
          </div>

          <div className="flex gap-2">
            <button type="button" onClick={describe} disabled={busy !== ''}
              className="px-3 py-2 rounded bg-slate-800 text-slate-200 text-sm hover:bg-slate-700 disabled:opacity-50">
              {busy === 'partition' ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : 'Inspect partition'}
            </button>
            <button type="button" onClick={seal} disabled={busy !== '' || !studyId.trim()}
              className="px-3 py-2 rounded bg-teal-500/20 text-teal-100 text-sm hover:bg-teal-500/30 disabled:opacity-50">
              {busy === 'seal' ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : 'Freeze the family'}
            </button>
          </div>
        </div>

        {/* ------------------------------------------------------------- what is bound */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">What a seal would bind</p>
          {!partition && <p className="text-sm text-slate-500">Inspect the partition to see it.</p>}
          {partition && (
            <>
              <p className="text-xs text-slate-500">{partition.claim_boundary}</p>
              <dl className="text-sm space-y-2">
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-400">Train frames</dt>
                  <dd className="text-slate-200">{partition.train.frames[0]}–{partition.train.frames[1]}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-400">Held-out frames</dt>
                  <dd className="text-slate-200">{partition.held_out.frames[0]}–{partition.held_out.frames[1]}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-400">Held-out digest</dt>
                  <dd className="text-slate-300 font-mono text-xs">{partition.held_out.digest.slice(0, 16)}…</dd>
                </div>
              </dl>
              {partition.already_opened && (
                <p className="text-sm text-rose-300 flex gap-2">
                  <Ban className="w-4 h-4 shrink-0" aria-hidden="true" />
                  This held-out partition has already been opened, at {partition.opened_record?.opened_at}.
                  It cannot be tested again, and a second seal over it would be refused.
                </p>
              )}
            </>
          )}

          {sealed && (
            <div className="border-t border-slate-800 pt-4 space-y-3">
              <p className="text-sm text-emerald-300 flex gap-2">
                <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden="true" />
                Frozen: {sealed.confirm_family_size} members, from a generated family of {sealed.generate_family_size}.
              </p>
              <p className="text-xs text-slate-400">
                Sealed at {sealed.sealed_at} by {sealed.sealed_at_source}
              </p>
              <p className="text-xs uppercase tracking-wide text-slate-500">Publish this digest</p>
              <code className="block text-xs font-mono text-teal-200 bg-slate-950 border border-slate-800 rounded p-2 break-all">
                {sealed.seal_sha256}
              </code>
              <p className="text-xs text-amber-300 flex gap-2">
                <ShieldAlert className="w-4 h-4 shrink-0" aria-hidden="true" />
                {sealed.publication}
              </p>
              <p className="text-xs text-slate-500">{sealed.claim_boundary}</p>
            </div>
          )}
        </div>

        {/* ----------------------------------------------------------------- confirm */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Seals, and what is left to spend</p>
          {seals.length === 0 && <p className="text-sm text-slate-500">No family has been frozen yet.</p>}
          <ul className="space-y-3">
            {seals.map((row) => (
              <li key={row.seal_sha256} className="border border-slate-800 rounded p-3 space-y-2">
                <p className="text-sm text-slate-200">{row.study_id || '(no study label)'}</p>
                <p className="text-xs font-mono text-slate-500 break-all">{row.seal_sha256.slice(0, 24)}…</p>
                <p className="text-xs text-slate-400">
                  {row.confirm_family_size} frozen of {row.generate_family_size} generated ·
                  held-out frames {row.held_out_frames[0]}–{row.held_out_frames[1]}
                </p>
                <button type="button" onClick={() => verify(row.seal_sha256)} disabled={busy !== ''}
                  className="px-3 py-1.5 rounded bg-slate-800 text-slate-200 text-xs hover:bg-slate-700 disabled:opacity-50">
                  Verify this seal
                </button>
                {verified[row.seal_sha256] && (
                  <p className="text-xs text-slate-400">{verified[row.seal_sha256]}</p>
                )}
                {row.spent ? (
                  <p className="text-xs text-slate-500 flex gap-2">
                    <Ban className="w-4 h-4 shrink-0" aria-hidden="true" />
                    Spent. This held-out data has been tested and will not be tested again.
                  </p>
                ) : (
                  <button type="button" onClick={() => confirm(row.seal_sha256)} disabled={busy !== ''}
                    className="px-3 py-2 rounded bg-rose-500/20 text-rose-100 text-sm hover:bg-rose-500/30 disabled:opacity-50">
                    {busy === 'confirm'
                      ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                      : 'Open the held-out partition once'}
                  </button>
                )}
              </li>
            ))}
          </ul>

          <label className="block text-xs text-slate-400 border-t border-slate-800 pt-3">
            Published seal digest (optional)
            <input value={publishedSha} onChange={(e) => setPublishedSha(e.target.value)}
              className="mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200 font-mono text-xs"
              aria-describedby="published-digest-help" />
          </label>
          <p id="published-digest-help" className="text-[11px] text-slate-600">
            A seal is not evidence about itself. Supply the digest as it was published somewhere you
            cannot rewrite, and the confirmation is checked against that instead of against the stored copy.
          </p>
        </div>
      </div>

      {/* ------------------------------------------------------------------- receipt */}
      {receipt && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <h3 className="text-lg font-semibold text-white">Confirmation receipt</h3>
            <span className="text-xs px-2 py-1 rounded border text-slate-300 border-slate-600">
              corrected over {receipt.correction_unit} frozen members
            </span>
            {confirmation?.checked_against_publication ? (
              <span className="text-xs px-2 py-1 rounded border text-emerald-300 border-emerald-500/30">
                checked against the published digest
              </span>
            ) : (
              <span className="text-xs px-2 py-1 rounded border text-amber-300 border-amber-500/30">
                self-consistency only — no published digest was supplied
              </span>
            )}
          </div>

          <p className="text-sm text-slate-400">
            {receipt.correction} at α={receipt.alpha}, assuming {receipt.dependence_assumption}.
            {' '}{receipt.n_rejected} of {receipt.labels.length} frozen members survived.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4">Member</th>
                  <th className="py-2 pr-4">p</th>
                  <th className="py-2 pr-4">adjusted</th>
                  <th className="py-2">survived</th>
                </tr>
              </thead>
              <tbody>
                {receipt.labels.map((label, index) => (
                  <tr key={label} className="border-t border-slate-800">
                    <td className="py-2 pr-4 font-mono text-xs text-slate-300">{label}</td>
                    <td className="py-2 pr-4 text-slate-300">{receipt.p_values[index]?.toFixed(4)}</td>
                    <td className="py-2 pr-4 text-slate-300">{receipt.adjusted[index]?.toFixed(4)}</td>
                    <td className="py-2 text-slate-300">{receipt.rejected[index] ? 'yes' : 'no'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {confirmation && confirmation.sweep_warnings.length > 0 && (
            <ul className="space-y-2">
              {confirmation.sweep_warnings.map((warning) => (
                <li key={warning} className="text-xs text-amber-300 flex gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />{warning}
                </li>
              ))}
            </ul>
          )}

          <p className="text-sm text-slate-400">{confirmation?.claim_boundary}</p>
          <p className="text-xs text-slate-500">
            The {receipt.generate_family_size} members mined on train are not corrected for here and are
            not claims. This receipt records no evidence and moves no rung; writing it into a bundle is a
            separate, deliberate act.
          </p>
        </div>
      )}
    </section>
  );
};

export default PreregistrationView;
