/**
 * TG9.2: the findings view.
 *
 * Phase G9's governing rule is visible in what this file does *not* contain: there is no
 * `toFixed`, no percent literal, and no arithmetic on any claim-bearing value. Every
 * scientific string displayed here was produced by the backend and is rendered verbatim.
 *
 * That is not stylistic. R9 says a bare confidence percentage must not be a renderable
 * component, and calls it a hard constraint on the frontend rather than only on the mining
 * code. A rule of that shape cannot be enforced by whoever writes the JSX remembering it. So
 * the backend refuses to serve a confidence without its five companions, `AssociationFigures`
 * cannot be constructed partially, and this component only ever prints `figures` through the
 * one string the backend already assembled. A percentage is not withheld here; it is
 * unobtainable.
 *
 * The same applies to entitlements. A `TranslationUnit` carries `rendered` and `licences`, and
 * this file always shows them together, because "this is a candidate precursor" without
 * "predictive utility is not shown" is a promotion performed by layout.
 */

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, BookOpen, FileText, Layers, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

interface Props {
  onError?: (message: string) => void;
  selectedStudyId?: string;
  onSelectStudy?: (studyId: string) => void;
}

/** One structural fact: its domain wording, inseparable from the bound that qualifies it. */
function Unit({ unit }: { unit: types.TranslationUnit }) {
  return (
    <li className="border-l-2 border-slate-700 pl-3 py-2">
      <p className="text-sm text-slate-200">{unit.rendered}</p>
      <p className="text-xs text-slate-400 mt-1 italic">{unit.licences}</p>
      <p className="text-[10px] font-mono text-slate-600 mt-1">{unit.structural_key}</p>
    </li>
  );
}

/**
 * A section of the finding. `emptyNote` is required rather than optional on purpose: an empty
 * section must say that nothing was recorded, because silence reads as reassurance and "no
 * contrary evidence was written down" is not "no contrary evidence exists".
 */
function Section(
  { title, units, emptyNote }:
  { title: string; units: types.TranslationUnit[]; emptyNote: string },
) {
  return (
    <section className="mb-5" aria-label={title}>
      <h4 className="text-xs uppercase tracking-wide text-slate-400 mb-2">{title}</h4>
      {units.length === 0
        ? <p className="text-sm text-slate-500 italic pl-3">{emptyNote}</p>
        : <ul className="space-y-1">{units.map((u) => <Unit key={u.structural_key} unit={u} />)}</ul>}
    </section>
  );
}

export default function FindingsView({
  onError, selectedStudyId = '', onSelectStudy,
}: Props) {
  const [domains, setDomains] = useState<types.DomainSummary[]>([]);
  const [studies, setStudies] = useState<types.StudySummary[]>([]);
  const [glossaryName, setGlossaryName] = useState<string>('');
  const [finding, setFinding] = useState<types.TranslatedFinding | null>(null);
  const [glossary, setGlossary] = useState<types.DomainGlossaryPayload | null>(null);
  const [outputs, setOutputs] = useState<Record<string, unknown> | null>(null);
  const [bundle, setBundle] = useState<Record<string, unknown> | null>(null);
  const [contract, setContract] = useState<types.OnboardingContract | null>(null);
  const [panel, setPanel] =
    useState<'finding' | 'refusals' | 'structural' | 'glossary' | 'onboarding'
             | 'bundle'>('finding');
  const [busy, setBusy] = useState(false);

  const fail = useCallback((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    if (onError) onError(message);
  }, [onError]);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const [domainRows, studyRows] = await Promise.all([
        apiService.listDomains(),
        apiService.listStudies(),
      ]);
      setDomains(domainRows);
      setStudies(studyRows);
      if (!glossaryName && domainRows.length > 0) setGlossaryName(domainRows[0].name);
      const readable = studyRows.find((row) => row.readable && row.study_id);
      if (!selectedStudyId && readable && readable.study_id) onSelectStudy?.(readable.study_id);
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  }, [fail, glossaryName, selectedStudyId, onSelectStudy]);

  useEffect(() => { void refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedStudyId || !glossaryName) return;
    setBusy(true);
    apiService.getTranslation(selectedStudyId, glossaryName)
      .then(setFinding)
      .catch(fail)
      .finally(() => setBusy(false));
  }, [selectedStudyId, glossaryName, fail]);

  useEffect(() => {
    if (panel === 'glossary' && glossaryName) {
      apiService.getGlossary(glossaryName).then(setGlossary).catch(fail);
    }
    if (panel === 'structural' && selectedStudyId) {
      apiService.getStudyOutputs(selectedStudyId).then(setOutputs).catch(fail);
    }
    if (panel === 'bundle' && selectedStudyId) {
      apiService.getStudy(selectedStudyId).then(setBundle).catch(fail);
    }
    if (panel === 'onboarding') {
      apiService.getOnboardingContract().then(setContract).catch(fail);
    }
  }, [panel, glossaryName, selectedStudyId, fail]);

  const panels: Array<{ key: typeof panel; label: string }> = [
    { key: 'finding', label: 'In domain words' },
    { key: 'refusals', label: 'What this domain refuses' },
    { key: 'structural', label: 'Untranslated claim state' },
    { key: 'glossary', label: 'The wording used' },
    { key: 'onboarding', label: 'How this domain was declared' },
    { key: 'bundle', label: 'The evidence itself' },
  ];

  return (
    <div className="space-y-4" aria-busy={busy}>
      <header className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Findings</h3>
          <p className="text-sm text-slate-400 max-w-3xl">
            A published study, stated in one domain&apos;s words. Every claim is shown with the
            bound that qualifies it. Nothing on this page is computed here: the wording, the
            figures and the refusals are all produced by the claim ladder and rendered as given.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          aria-label="Reload domains and studies"
          className="flex items-center gap-2 px-3 py-2 text-sm rounded bg-slate-800 hover:bg-slate-700
                     text-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-400"
        >
          <RefreshCw size={14} aria-hidden="true" /> Reload
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="lg:col-span-1 space-y-4">
          <div>
            <label htmlFor="findings-domain"
              className="block text-xs uppercase tracking-wide text-slate-400 mb-1">
              Domain vocabulary
            </label>
            <select
              id="findings-domain"
              value={glossaryName}
              onChange={(event) => setGlossaryName(event.target.value)}
              className="w-full bg-slate-800 text-slate-100 text-sm rounded px-2 py-2
                         focus:outline-none focus:ring-2 focus:ring-teal-400"
            >
              {domains.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.onboarding.complete ? d.domain : `${d.domain} (declaration incomplete)`}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500 mt-1">
              Changing this changes the words and nothing else. The facts below are identical in
              every vocabulary.
            </p>
          </div>

          <div>
            <h4 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
              <Layers size={12} className="inline mr-1" aria-hidden="true" /> Published studies
            </h4>
            <ul className="space-y-1" role="list">
              {studies.map((row) => (
                <li key={row.file}>
                  {row.readable && row.study_id ? (
                    <button
                      type="button"
                      onClick={() => onSelectStudy?.(row.study_id as string)}
                      aria-pressed={selectedStudyId === row.study_id}
                      className={`w-full text-left px-2 py-2 rounded text-sm focus:outline-none
                                  focus:ring-2 focus:ring-teal-400 ${
                        selectedStudyId === row.study_id
                          ? 'bg-teal-900/40 text-teal-200'
                          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                    >
                      <span className="block truncate">{row.study_id}</span>
                      <span className="block text-xs text-slate-400">
                        {row.rung}{row.blocked ? ' — capped' : ''}
                      </span>
                    </button>
                  ) : (
                    <div className="px-2 py-2 rounded text-sm bg-red-950/40 text-red-300">
                      <AlertTriangle size={12} className="inline mr-1" aria-hidden="true" />
                      <span>{row.file} could not be read</span>
                      <span className="block text-xs text-red-400/80">{row.refused_because}</span>
                    </div>
                  )}
                </li>
              ))}
              {studies.length === 0 && (
                <li className="text-sm text-slate-500 italic">
                  No studies published. That is not the same as no studies existing.
                </li>
              )}
            </ul>
          </div>
        </div>

        <div className="lg:col-span-3 bg-slate-900/60 rounded p-4 min-h-[24rem]">
          <div role="tablist" aria-label="Findings panels" className="flex gap-1 mb-4 flex-wrap">
            {panels.map((entry) => (
              <button
                key={entry.key}
                type="button"
                role="tab"
                aria-selected={panel === entry.key}
                onClick={() => setPanel(entry.key)}
                className={`px-3 py-1.5 text-xs rounded focus:outline-none focus:ring-2
                            focus:ring-teal-400 ${
                  panel === entry.key
                    ? 'bg-teal-900/50 text-teal-200'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
              >
                {entry.label}
              </button>
            ))}
          </div>

          {busy && <p className="text-sm text-slate-400">Loading…</p>}

          {panel === 'finding' && finding && (
            <div>
              <p className="text-xs font-mono text-slate-500 mb-4">
                {finding.study_id} · revision {finding.revision} · claim state{' '}
                {finding.summary_sha256.slice(0, 12)}
              </p>

              <Section title="What can be said" units={finding.claimable}
                emptyNote="Nothing may be claimed from this record." />

              {finding.figures_text && (
                <div className="mb-5 pl-3">
                  <h4 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
                    Strength of the association
                  </h4>
                  {/*
                    The one place a percentage appears in this component, and it is a single
                    string the backend assembled. Note what is *not* here: no read of
                    `figures.confidence`, no interpolation of `figures.support`. The moment a
                    view builds this line from parts, R9 depends on the view author remembering
                    to include the base rate - which is exactly the dependency it must not have.
                  */}
                  <p className="text-sm text-slate-200 font-mono">{finding.figures_text}</p>
                </div>
              )}

              {finding.unadmitted_reading && (
                <section
                  className="mb-5 border border-orange-800/60 rounded p-3 bg-orange-950/20"
                  aria-label="Reading not admitted by the selected domain"
                >
                  <h4 className="text-xs uppercase tracking-wide text-orange-300 mb-2">
                    <AlertTriangle size={12} className="inline mr-1" aria-hidden="true" />
                    The selected domain does not admit this reading
                  </h4>
                  <p className="text-sm text-orange-100/90">{finding.unadmitted_reading.note}</p>
                  <p className="text-xs text-orange-300/70 mt-2 italic">
                    {finding.unadmitted_reading.attribution_caveat}
                  </p>
                </section>
              )}

              <Section title="What cannot be said" units={finding.not_claimable}
                emptyNote="No rung above the one reached is defined." />
              <Section title="What argues against it" units={finding.contradicting}
                emptyNote="None recorded; that is not the same as none existing." />
              <Section title="What else could explain it" units={finding.alternatives}
                emptyNote="None that this record names; that is not the same as none existing." />
              <Section
                title="What to look at next"
                units={finding.next_observation ? [finding.next_observation] : []}
                emptyNote="None identified; that is not the same as none existing." />

              {finding.commentary.length > 0 && (
                <section className="mt-6 border border-amber-900/50 rounded p-3 bg-amber-950/20"
                  aria-label="Recorded commentary">
                  <h4 className="text-xs uppercase tracking-wide text-amber-400 mb-2">
                    Commentary — recorded, not reproducible, and it moved nothing above
                  </h4>
                  <ul className="space-y-1">
                    {finding.commentary.map((line) => (
                      <li key={line} className="text-sm text-amber-100/80">{line}</li>
                    ))}
                  </ul>
                </section>
              )}

              <details className="mt-6">
                <summary className="text-xs text-slate-400 cursor-pointer focus:outline-none
                                    focus:ring-2 focus:ring-teal-400 rounded">
                  <FileText size={12} className="inline mr-1" aria-hidden="true" />
                  The whole statement as rendered by the backend
                </summary>
                <pre className="mt-2 text-xs text-slate-300 whitespace-pre-wrap font-mono
                                bg-slate-950/60 p-3 rounded overflow-x-auto">
                  {finding.rendered_text}
                </pre>
              </details>
            </div>
          )}

          {panel === 'refusals' && (
            <div>
              {finding?.domain_limits ? (
                <>
                  <p className="text-sm text-slate-400 mb-3">
                    What this domain&apos;s own declaration forbids, and why. Each reason is the
                    one the analysis layer enforces, not a restatement of it.
                  </p>
                  <p className="text-xs text-amber-300/90 mb-4 italic border-l-2
                                border-amber-700 pl-2">
                    {finding.domain_limits.attribution_caveat}
                  </p>
                  <dl className="space-y-2">
                    <div className="flex gap-3 text-sm">
                      <dt className="text-slate-500 w-56 shrink-0">Lead-lag reading (R21)</dt>
                      <dd className={finding.domain_limits.precedence_admissible
                        ? 'text-slate-300' : 'text-orange-300'}>
                        {finding.domain_limits.precedence_admissible
                          ? 'admissible: this domain declares a lag floor'
                          : 'not admissible: this domain declares no lag floor'}
                      </dd>
                    </div>
                  </dl>
                  <h4 className="text-xs uppercase tracking-wide text-slate-400 mt-5 mb-2">
                    Declared refusals
                  </h4>
                  {finding.domain_limits.refuses.length === 0 ? (
                    <p className="text-sm text-slate-500 italic">
                      This domain declares no assumption violations. That means the inherited
                      assumptions were written against it, not that it is free of limits.
                    </p>
                  ) : (
                    <ul className="space-y-2">
                      {finding.domain_limits.refuses.map((refusal) => (
                        <li key={refusal.basis}
                          className="border-l-2 border-orange-800 pl-3 py-1">
                          <p className="text-xs font-mono text-orange-400">{refusal.basis}</p>
                          <p className="text-sm text-slate-300 mt-1">{refusal.consequence}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              ) : (
                <p className="text-sm text-slate-500 italic">
                  This vocabulary has no domain declaration registered behind it, so what it
                  refuses is unknown. That is not the same as it refusing nothing.
                </p>
              )}
            </div>
          )}

          {panel === 'structural' && outputs && (
            <div>
              <p className="text-sm text-slate-400 mb-2">
                The same finding in the programme&apos;s own vocabulary, so the domain wording can
                be checked against it. If these disagree, the translation is at fault.
              </p>
              <pre className="text-xs text-slate-300 font-mono bg-slate-950/60 p-3 rounded
                              overflow-x-auto max-h-[32rem]">
                {JSON.stringify(outputs, null, 2)}
              </pre>
            </div>
          )}

          {panel === 'glossary' && glossary && (
            <div>
              <p className="text-sm text-slate-400 mb-2">
                <BookOpen size={12} className="inline mr-1" aria-hidden="true" />
                Every word this domain uses for a structural term. Published so it can be
                audited: the wording is screened, but nothing can verify that it means to a
                practitioner what it appears to mean.
              </p>
              <dl className="text-xs space-y-1 max-h-[32rem] overflow-y-auto">
                {Object.entries(glossary.phrases).map(([term, phrase]) => (
                  <div key={term} className="flex gap-3 border-b border-slate-800 py-1">
                    <dt className="font-mono text-slate-500 w-1/3 shrink-0">{term}</dt>
                    <dd className="text-slate-300">{phrase}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {panel === 'onboarding' && contract && (
            <div>
              <p className="text-sm text-slate-400 mb-3">
                What every domain must declare before anything may be read in its words, and
                whether each registered domain satisfied all of it in one checked call. A domain
                marked incomplete was assembled from separate registrations, so its geometry and
                its declared violations were never checked against each other.
              </p>

              <section className="mb-5" aria-label="The onboarding contract">
                <h5 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
                  Required of every domain
                </h5>
                <dl className="text-xs space-y-1">
                  {contract.required.map((item) => (
                    <div key={item.requirement}
                      className="flex gap-3 border-b border-slate-800 py-1">
                      <dt className="font-mono text-slate-500 w-1/4 shrink-0">
                        {item.requirement}
                      </dt>
                      <dd className="text-slate-300">{item.why}</dd>
                    </div>
                  ))}
                </dl>
              </section>

              <section aria-label="Declared domains">
                <h5 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
                  Domains declared through the contract
                </h5>
                <ul className="text-xs space-y-2">
                  {contract.onboarded.map((row) => (
                    <li key={row.name} className="border border-slate-800 rounded p-2">
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="text-slate-200 font-medium">{row.name}</span>
                        <span className={row.complete ? 'text-teal-400' : 'text-amber-400'}>
                          {row.complete ? 'declared in full' : 'declaration incomplete'}
                        </span>
                      </div>
                      {row.onboarded_by && (
                        <p className="text-slate-500 mt-1">
                          declared by <span className="font-mono">{row.onboarded_by}</span>
                        </p>
                      )}
                      {row.geometry !== undefined && (
                        <p className="text-slate-500">
                          geometry:{' '}
                          <span className="font-mono">{row.geometry ?? 'none declared'}</span>
                        </p>
                      )}
                      {row.onboarding_sha256 && (
                        <p className="text-slate-600 font-mono break-all">
                          {row.onboarding_sha256}
                        </p>
                      )}
                      {row.missing.length > 0 && (
                        <p className="text-amber-400 mt-1">
                          missing: {row.missing.join(', ')}
                        </p>
                      )}
                      {row.note && <p className="text-amber-400/80 mt-1">{row.note}</p>}
                    </li>
                  ))}
                </ul>
              </section>

              <p className="text-xs text-amber-300/90 mt-4 border border-amber-900/50 rounded
                            p-2 bg-amber-950/20">
                {contract.attribution_caveat}
              </p>
            </div>
          )}

          {panel === 'bundle' && bundle && (
            <div>
              <p className="text-sm text-slate-400 mb-2">
                The evidence chain itself, before any wording was applied to it.
              </p>
              <pre className="text-xs text-slate-300 font-mono bg-slate-950/60 p-3 rounded
                              overflow-x-auto max-h-[32rem]">
                {JSON.stringify(bundle, null, 2)}
              </pre>
            </div>
          )}

          {panel === 'finding' && !finding && !busy && (
            <p className="text-sm text-slate-500 italic">
              Select a published study to see what it permits and what it does not.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
