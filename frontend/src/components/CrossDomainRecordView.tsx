/** TG11.4b — the cross-domain record: two clocks, one exact intersection, lags in seconds.
 *
 * Everywhere else in this application a lag family is entered in frames. That is honest while
 * there is one record. Two domains sampled hourly and three-hourly have two frame sizes, and a
 * family declared in either one is a family the other domain cannot read — so every lag control
 * on this panel is in **seconds**, and the server converts them onto the exact common cadence
 * or refuses the ones that cadence cannot express.
 *
 * Three things this panel deliberately cannot offer.
 *
 * There is no resampling control. The two records are aligned by exact timestamp intersection;
 * if they share too few observations the request is refused. A nearest-neighbour or linear join
 * would manufacture values at times one source never observed and then let those values vote in
 * a lagged relationship. What is shown instead is how many native observations each side kept
 * and discarded, because a join that quietly kept a third of one record is a different study.
 *
 * There is no default unit. Each column must be given its semantics and its units before the
 * record can be read, because a channel table carries neither and R19 does not allow either to
 * be dropped. The statistic is dimensionless and compares no raw magnitude; what keeps that
 * true is that both magnitudes still say what they are, all the way into the receipt.
 *
 * There is nothing to tune at confirmation. The confirm control sends the two files and, if the
 * seal digest was published, that digest. The domains, the columns, the family, the split, the
 * ensemble, the correction and the seed are read back out of the seal.
 *
 * Every verdict — the price, the affordability, the candidates, the confirmed labels — is
 * rendered whole from the response. This panel computes nothing.
 */

import { useCallback, useEffect, useState } from 'react';
import { Clock, Globe, Loader2, Lock, ShieldCheck } from 'lucide-react';

import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  studyId: string;
  onError?: (message: string) => void;
}

type Busy = '' | 'align' | 'price' | 'partition' | 'generate' | 'seal' | 'confirm';

const FIELD = 'mt-1 w-full bg-slate-950 border border-slate-700 rounded p-2 text-slate-200';
const CARD = 'bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4';
const BUTTON = 'w-full px-3 py-2 rounded bg-indigo-500/20 text-indigo-100 text-sm '
  + 'hover:bg-indigo-500/30 disabled:opacity-40';
const LABEL = 'block text-xs uppercase tracking-wide text-slate-400';

const SHORT = (digest?: string | null) => (digest ? `${digest.slice(0, 12)}…` : '—');

const EXAMPLE_CHANNELS = '{\n  "column_name": { "semantics": "what it means", "units": "K" }\n}';

const CrossDomainRecordView: React.FC<Props> = ({ studyId, onError }) => {
  const [capabilities, setCapabilities] = useState<types.CrossDomainCapabilities | null>(null);
  const [aligned, setAligned] = useState<types.CrossDomainAligned | null>(null);
  const [price, setPrice] = useState<types.CrossDomainPrice | null>(null);
  const [partition, setPartition] = useState<types.CrossDomainPartition | null>(null);
  const [generated, setGenerated] = useState<types.CrossDomainGeneration | null>(null);
  const [seal, setSeal] = useState<types.CrossDomainSeal | null>(null);
  const [confirmation, setConfirmation] = useState<types.CrossDomainConfirmation | null>(null);

  const [firstFile, setFirstFile] = useState<File | null>(null);
  const [secondFile, setSecondFile] = useState<File | null>(null);
  const [firstDomain, setFirstDomain] = useState('');
  const [secondDomain, setSecondDomain] = useState('');
  const [firstTimeColumn, setFirstTimeColumn] = useState('t');
  const [secondTimeColumn, setSecondTimeColumn] = useState('t');
  const [firstChannels, setFirstChannels] = useState(EXAMPLE_CHANNELS);
  const [secondChannels, setSecondChannels] = useState(EXAMPLE_CHANNELS);

  const [name, setName] = useState('cross-domain');
  const [lagSeconds, setLagSeconds] = useState('10800, 21600, 32400, 43200');
  const [surrogates, setSurrogates] = useState(199);
  const [fraction, setFraction] = useState(0.55);
  const [publishedDigest, setPublishedDigest] = useState('');

  const [busy, setBusy] = useState<Busy>('');

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  useEffect(() => {
    apiService.getCrossDomainCapabilities().then(setCapabilities).catch(fail);
  }, [fail]);

  /** The reading of one side. Parsed here so a malformed declaration is a message beside the
   *  control that holds it rather than a 400 with no obvious owner. */
  const source = (domain: string, timeColumn: string, channels: string):
    types.CrossDomainSource => ({
      domain, time_column: timeColumn, channels: JSON.parse(channels)
    });

  const durations = (): number[] =>
    lagSeconds.split(',').map((piece) => Number(piece.trim())).filter((v) => !Number.isNaN(v));

  const both = (): [File, File, types.CrossDomainSource, types.CrossDomainSource] => {
    if (!firstFile || !secondFile) throw new Error('Both records are required to align.');
    return [firstFile, secondFile,
      source(firstDomain, firstTimeColumn, firstChannels),
      source(secondDomain, secondTimeColumn, secondChannels)];
  };

  const run = async (step: Busy, action: () => Promise<void>) => {
    setBusy(step);
    try {
      await action();
    } catch (error) {
      fail(error);
    } finally {
      setBusy('');
    }
  };

  const onAlign = () => run('align', async () => {
    const [a, b, one, two] = both();
    setAligned(await apiService.alignDomains(a, b, one, two, name));
  });

  const onPrice = () => run('price', async () => {
    const [a, b, one, two] = both();
    setPrice(await apiService.priceCrossDomainLags(a, b, one, two, name, durations(),
      surrogates));
  });

  const onPartition = () => run('partition', async () => {
    const [a, b, one, two] = both();
    setPartition(await apiService.describeCrossDomainPartition(a, b, one, two, name, fraction));
  });

  const onGenerate = () => run('generate', async () => {
    const [a, b, one, two] = both();
    setGenerated(await apiService.generateCrossDomain(a, b, one, two, name, durations(),
      studyId, surrogates, fraction));
  });

  const onSeal = () => run('seal', async () => {
    const [a, b, one, two] = both();
    setSeal(await apiService.sealCrossDomain(a, b, one, two, name, durations(), studyId,
      surrogates, fraction));
  });

  const onConfirm = () => run('confirm', async () => {
    if (!seal) throw new Error('Freeze a family before opening the held-out partition.');
    const [a, b] = both();
    setConfirmation(await apiService.confirmCrossDomain(seal.seal_sha256, a, b,
      publishedDigest || undefined));
  });

  const spinner = (step: Busy) => (busy === step
    ? <Loader2 className="w-4 h-4 animate-spin inline mr-2" /> : null);

  return (
    <div className="space-y-6" aria-busy={busy !== ''}>
      <header className="flex items-start gap-3">
        <Globe className="w-6 h-6 text-indigo-300 mt-1" />
        <div>
          <h2 className="text-xl text-slate-100">Cross-domain record</h2>
          <p className="text-sm text-slate-400 max-w-3xl">
            Two records from two declared domains, aligned on the timestamps they actually
            share. {capabilities?.claim_boundary}
          </p>
          {capabilities && (
            <p className="text-xs text-slate-500 mt-1">
              Alignment: {capabilities.alignment}. Interpolation: {capabilities.interpolation}.
              Lags declared in {capabilities.lags_declared_in}. At least{' '}
              {capabilities.minimum_common_observations} shared observations.
            </p>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ---------------------------------------------------------------- the two sides */}
        <section className={CARD}>
          <h3 className="text-slate-200 text-sm flex items-center gap-2">
            <Clock className="w-4 h-4" /> Two native clocks
          </h3>

          {[{
            id: 'first', label: 'First domain', file: firstFile, setFile: setFirstFile,
            domain: firstDomain, setDomain: setFirstDomain,
            column: firstTimeColumn, setColumn: setFirstTimeColumn,
            channels: firstChannels, setChannels: setFirstChannels,
          }, {
            id: 'second', label: 'Second domain', file: secondFile, setFile: setSecondFile,
            domain: secondDomain, setDomain: setSecondDomain,
            column: secondTimeColumn, setColumn: setSecondTimeColumn,
            channels: secondChannels, setChannels: setSecondChannels,
          }].map((side) => (
            <fieldset key={side.label} className="space-y-2 border-t border-slate-800 pt-3">
              <legend className={LABEL}>{side.label}</legend>
              <label htmlFor={`${side.id}-domain-file`} className="sr-only">{side.label} channel table</label>
              <input id={`${side.id}-domain-file`} type="file" accept=".csv,.tsv,.txt" className={FIELD}
                onChange={(event) => side.setFile(event.target.files?.[0] ?? null)} />
              <label htmlFor={`${side.id}-domain-name`} className="sr-only">{side.label} registered domain name</label>
              <input id={`${side.id}-domain-name`} className={FIELD} placeholder="registered domain name"
                value={side.domain} onChange={(e) => side.setDomain(e.target.value)} />
              <label htmlFor={`${side.id}-time-column`} className="sr-only">{side.label} time column</label>
              <input id={`${side.id}-time-column`} className={FIELD} placeholder="time column"
                value={side.column} onChange={(e) => side.setColumn(e.target.value)} />
              <label htmlFor={`${side.id}-channel-declarations`} className="sr-only">{side.label} channel semantics and units JSON</label>
              <textarea id={`${side.id}-channel-declarations`} className={`${FIELD} font-mono text-xs h-28`} value={side.channels}
                onChange={(e) => side.setChannels(e.target.value)} />
            </fieldset>
          ))}

          <p className="text-xs text-slate-500">
            Every column needs its semantics and its units. Neither is defaulted: the statistic
            compares no raw magnitude, and the receipt carries both operands' meanings back.
          </p>

          <label htmlFor="cross-domain-study-name" className="sr-only">Study name for this alignment</label>
          <input id="cross-domain-study-name" className={FIELD} placeholder="study name for this alignment"
            value={name} onChange={(e) => setName(e.target.value)} />
          <button className={BUTTON} disabled={busy !== ''} onClick={onAlign}>
            {spinner('align')}Align on shared timestamps
          </button>

          {aligned && (
            <div className="text-xs text-slate-400 space-y-1">
              <p>Common cadence {aligned.alignment.common_cadence_seconds} s over{' '}
                {aligned.alignment.n_common_observations} shared observations.</p>
              <p>Clock {SHORT(aligned.alignment.clock_sha256)}. Physical lag floor{' '}
                {aligned.alignment.physical_lag_floor_seconds} s.</p>
              {Object.entries(aligned.alignment.discarded_native_observations).map(
                ([domain, discarded]) => (
                  <p key={domain}>{domain}: kept{' '}
                    {aligned.alignment.retained_native_observations[domain]}, discarded{' '}
                    {discarded}.</p>
                ))}
              <p className="text-slate-500">{aligned.cross_domain_pairs.length} ordered pairs
                cross the boundary.</p>
            </div>
          )}
        </section>

        {/* ------------------------------------------------------------- family and split */}
        <section className={CARD}>
          <h3 className="text-slate-200 text-sm">Family, in seconds</h3>
          <label htmlFor="cross-domain-lags" className={LABEL}>Lag durations (seconds, comma separated)</label>
          <input id="cross-domain-lags" className={FIELD} value={lagSeconds}
            onChange={(e) => setLagSeconds(e.target.value)} />
          <label htmlFor="cross-domain-surrogates" className={LABEL}>Surrogates</label>
          <input id="cross-domain-surrogates" className={FIELD} type="number" value={surrogates}
            onChange={(e) => setSurrogates(Number(e.target.value))} />
          <label htmlFor="cross-domain-training-fraction" className={LABEL}>Training fraction</label>
          <input id="cross-domain-training-fraction" className={FIELD} type="number" step="0.05" value={fraction}
            onChange={(e) => setFraction(Number(e.target.value))} />

          <button className={BUTTON} disabled={busy !== ''} onClick={onPrice}>
            {spinner('price')}Price this family
          </button>

          {price && (
            <div className="text-xs text-slate-400 space-y-1">
              <p>{price.family.family_size} members: {price.family.n_pairs} crossing pairs at{' '}
                lag frames [{price.family.lag_frames.join(', ')}] on the common clock.</p>
              <p className={price.family.affordable ? 'text-emerald-300' : 'text-amber-300'}>
                {price.family.affordable
                  ? `Affordable at ${price.family.n_surrogates} surrogates.`
                  : `Not affordable at ${price.family.n_surrogates} surrogates: this family `
                    + `needs ${price.family.required_surrogates} to be capable of a rejection, `
                    + `and only ${price.family.affordable_member_count} members can be carried `
                    + `forward.`}
              </p>
              <p className="text-slate-500">{price.family.reading}</p>
            </div>
          )}

          <button className={BUTTON} disabled={busy !== ''} onClick={onPartition}>
            {spinner('partition')}Show the split
          </button>

          {partition && (
            <div className="text-xs text-slate-400 space-y-1">
              <p>Train {partition.train.n_frames} frames, held out{' '}
                {partition.held_out.n_frames} frames, embargo {partition.embargo_frames}{' '}
                (recommended {partition.recommended_embargo_frames}).</p>
              <p>Held-out partition {SHORT(partition.held_out.digest)}.</p>
              {partition.already_opened && (
                <p className="text-amber-300">This partition has already been opened. It
                  cannot be confirmed against again, under this seal or any other.</p>
              )}
            </div>
          )}

          <button className={BUTTON} disabled={busy !== ''} onClick={onGenerate}>
            {spinner('generate')}Sweep the training partition
          </button>

          {generated && (
            <div className="text-xs text-slate-400 space-y-1">
              <p>{generated.n_examined} members examined. Carried forward:</p>
              <ul className="space-y-1">
                {generated.candidates.map((candidate) => (
                  <li key={candidate.label} className="font-mono text-[11px] text-slate-300">
                    {candidate.label} r={candidate.correlation.toFixed(3)}
                  </li>
                ))}
              </ul>
              <p className="text-slate-500">{generated.claim_boundary}</p>
            </div>
          )}
        </section>

        {/* -------------------------------------------------------------- seal and confirm */}
        <section className={CARD}>
          <h3 className="text-slate-200 text-sm flex items-center gap-2">
            <Lock className="w-4 h-4" /> Freeze, then open once
          </h3>

          <button className={BUTTON} disabled={busy !== ''} onClick={onSeal}>
            {spinner('seal')}Freeze the confirmatory family
          </button>

          {seal && (
            <div className="text-xs text-slate-400 space-y-1">
              <p>Seal {SHORT(seal.seal_sha256)} at {seal.sealed_at}.</p>
              <p>{seal.confirm_family_size} of {seal.generate_family_size} members frozen.</p>
              <ul className="space-y-1">
                {seal.confirm_labels.map((label) => (
                  <li key={label} className="font-mono text-[11px] text-slate-300">{label}</li>
                ))}
              </ul>
              <p className="text-slate-500">{seal.publication}</p>
            </div>
          )}

          <label htmlFor="cross-domain-published-seal" className={LABEL}>Published seal digest (optional)</label>
          <input id="cross-domain-published-seal" className={FIELD} placeholder="the digest you published before sealing"
            value={publishedDigest} onChange={(e) => setPublishedDigest(e.target.value)} />

          <button className={BUTTON} disabled={busy !== '' || !seal} onClick={onConfirm}>
            {spinner('confirm')}Open the held-out partition
          </button>
          <p className="text-xs text-slate-500">
            This control sends the two records and nothing else. The domains, the columns, the
            family, the split, the ensemble and the seed all come out of the seal.
          </p>

          {confirmation && (
            <div className="text-xs text-slate-400 space-y-2">
              <p className="flex items-center gap-2 text-slate-200">
                <ShieldCheck className="w-4 h-4" />
                {confirmation.confirmed_labels.length === 0
                  ? 'Nothing was confirmed on the held-out partition.'
                  : `${confirmation.confirmed_labels.length} relationship(s) confirmed.`}
              </p>
              {(confirmation.receipt.relationships ?? []).map((row: any) => (
                <div key={row.label} className="border-t border-slate-800 pt-2">
                  <p className="font-mono text-[11px] text-slate-300">{row.label}</p>
                  <p>Lead of {row.lag_seconds} s. Statistic {row.statistic_units}.</p>
                  <p className="text-slate-500">
                    {row.driver_operand?.semantics} ({row.driver_operand?.units}) →{' '}
                    {row.driven_operand?.semantics} ({row.driven_operand?.units})
                  </p>
                </div>
              ))}
              {(confirmation.receipt.vacuous ?? []).length > 0 && (
                <p className="text-amber-300">
                  Vacuous: {(confirmation.receipt.vacuous as string[]).join(', ')}. The ensemble
                  could not have rejected these, so their silence says nothing.
                </p>
              )}
              <p className="text-slate-500">{confirmation.claim_boundary}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default CrossDomainRecordView;
