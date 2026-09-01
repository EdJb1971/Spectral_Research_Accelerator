import { useEffect, useState } from 'react';
import { AlertTriangle, Archive, FileWarning, Landmark, ScrollText, ShieldOff } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

/** T4C.5j: the atmospheric gate record, made readable without opening a file.
 *
 * This panel exists because the two distinctions the T4C line was built to draw were the two
 * buried deepest in the filesystem: that a **retired** campaign is still readable but must not
 * be acquired, and that a FAIL is a negative finding only where the derived spatial-power record
 * shows the absence was detectable. A reviewer cannot check either one by being told they hold.
 *
 * Three rules govern everything below.
 *
 * *It offers no action.* There is no acquire, preflight, edit or re-freeze control, because the
 * server serves no route that would answer one. The refusals are rendered from the server's own
 * list rather than implied by an absence of buttons, so a reader who wonders where the acquire
 * button went is told.
 *
 * *A retirement is shown at the top of the design it retires.* A researcher who scrolls into a
 * campaign's calendar split and family size has already begun reading it as live, so the banner
 * must precede them. The retired design is still shown in full, with `resolvable: false` intact:
 * hiding it would erase the record of what was actually preregistered.
 *
 * *An empty receipt list is labelled as an absence of runs.* Rendered bare it reads as an
 * absence of findings, which is the opposite claim and the more attractive one.
 */
export default function GateRecordView({ onError }: { onError?: (message: string) => void }) {
  const [surface, setSurface] = useState<types.GateSurface | null>(null);
  const [campaigns, setCampaigns] = useState<types.GateCampaignSummary[]>([]);
  const [supersessions, setSupersessions] = useState<types.GateSupersessionSummary[]>([]);
  const [receipts, setReceipts] = useState<types.GateReceiptIndex | null>(null);
  const [openCampaign, setOpenCampaign] = useState<types.GateCampaignReview | null>(null);
  const [openSupersession, setOpenSupersession] = useState<types.GateSupersessionReview | null>(null);
  const [openReceipt, setOpenReceipt] = useState<types.GateReceiptView | null>(null);
  const [busy, setBusy] = useState(true);
  const [status, setStatus] = useState('Loading the gate record…');

  const fail = (error: Error) => { setStatus(error.message); onError?.(error.message); };

  useEffect(() => {
    Promise.all([apiService.gateSurface(), apiService.listGateCampaigns(),
                 apiService.listGateSupersessions(), apiService.listGateReceipts()])
      .then(([head, campaignIndex, supersessionIndex, receiptIndex]) => {
        setSurface(head);
        setCampaigns(campaignIndex.campaigns);
        setSupersessions(supersessionIndex.supersessions);
        setReceipts(receiptIndex);
        setStatus('');
      })
      .catch(fail)
      .finally(() => setBusy(false));
  }, []);

  const openCampaignRecord = async (campaignId: string) => {
    setBusy(true);
    setStatus(`Reading ${campaignId}…`);
    try {
      setOpenCampaign(await apiService.gateCampaignReview(campaignId));
      setOpenReceipt(null);
      setStatus('');
    } catch (error: any) { fail(error); } finally { setBusy(false); }
  };

  const openSupersessionRecord = async (supersessionId: string) => {
    setBusy(true);
    setStatus(`Re-running the stated checks against both campaigns…`);
    try {
      setOpenSupersession(await apiService.gateSupersessionReview(supersessionId));
      setStatus('');
    } catch (error: any) { fail(error); } finally { setBusy(false); }
  };

  const openReceiptRecord = async (receiptId: string) => {
    setBusy(true);
    setStatus(`Verifying receipt ${receiptId}…`);
    try {
      setOpenReceipt(await apiService.gateReceipt(receiptId));
      setOpenCampaign(null);
      setStatus('');
    } catch (error: any) { fail(error); } finally { setBusy(false); }
  };

  const design = openCampaign?.scientific_design ?? null;
  const split = (design?.calendar_split ?? {}) as Record<string, string>;
  const resolution = (design?.surrogate_resolution ?? {}) as Record<string, any>;

  return (
    <div className="space-y-5" aria-busy={busy}>

      {/* ------------------------------------------------------------------ what this holds */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3"
               aria-labelledby="gate-title">
        <div className="flex flex-wrap gap-3 justify-between items-start">
          <div>
            <h3 id="gate-title"
                className="text-sm font-semibold text-slate-200 flex gap-2 items-center">
              <Landmark className="w-4 h-4 text-sky-400" aria-hidden="true" />
              Atmospheric gate record (T4C)
            </h3>
            <p className="text-[11px] text-slate-500 mt-1 max-w-3xl">
              Preregistered ERA5 gate designs, the checked records that retire one, and the
              immutable receipts a run publishes. Read-only: this workspace serves no action
              that could acquire data, change a design or move a verdict.
            </p>
          </div>
          {surface && (
            <div className={`border rounded px-3 py-2 text-xs font-semibold ${
              surface.measurement_status === 'MEASURED'
                ? 'text-emerald-300 border-emerald-500/30 bg-emerald-500/5'
                : 'text-amber-200 border-amber-500/30 bg-amber-500/5'}`}>
              {surface.measurement_status}
            </div>
          )}
        </div>

        <p role="status" className="text-xs text-slate-400">{status}</p>

        {surface && (
          <>
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
              {[['Designs', surface.campaigns], ['Retired', surface.retired_campaigns],
                ['Supersessions', surface.supersessions], ['Receipts', surface.receipts]]
                .map(([label, value]) => (
                  <div key={String(label)} className="bg-slate-950/50 border border-slate-800 rounded p-2">
                    <dt className="text-slate-500">{label}</dt>
                    <dd className="text-slate-200 text-base font-semibold">{value}</dd>
                  </div>
                ))}
            </dl>
            <div className="border border-slate-800 rounded p-3 bg-slate-950/40">
              <p className="text-[11px] text-slate-400 font-semibold flex gap-2 items-center">
                <ShieldOff className="w-3.5 h-3.5 text-slate-500" aria-hidden="true" />
                What this surface will not do
              </p>
              <ul className="mt-2 space-y-1 text-[11px] text-slate-500 list-disc pl-5">
                {surface.refusals.map((line) => <li key={line}>{line}</li>)}
              </ul>
            </div>
            <p className="text-[11px] text-slate-500 italic">{surface.claim_boundary}</p>
            {surface.unreadable.length > 0 && (
              <p role="alert" className="text-[11px] text-rose-300 flex gap-2 items-start">
                <FileWarning className="w-3.5 h-3.5 mt-0.5 shrink-0" aria-hidden="true" />
                {surface.unreadable.length} file(s) in the store did not authenticate and are
                listed rather than dropped: {surface.unreadable.map((r) => r.file).join(', ')}
              </p>
            )}
          </>
        )}
      </section>

      {/* -------------------------------------------------------------------- the designs */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3"
               aria-labelledby="gate-campaigns-title">
        <h3 id="gate-campaigns-title" className="text-sm font-semibold text-slate-200">
          Preregistered designs
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]" aria-label="Preregistered gate campaigns">
            <thead><tr className="text-left text-slate-500 border-b border-slate-800">
              <th className="py-2">Campaign</th><th>Record</th><th>Frames</th>
              <th>Resolves its family</th><th>Status</th><th />
            </tr></thead>
            <tbody>
              {campaigns.map((row) => (
                <tr key={row.campaign_id} className="border-b border-slate-800/60">
                  <td className="py-2 pr-3 text-slate-300 font-mono">{row.campaign_id}</td>
                  <td className="pr-3 text-slate-400">{row.date_start} → {row.date_end}</td>
                  <td className="pr-3 text-slate-400">{row.expected_frames}</td>
                  <td className={`pr-3 font-semibold ${
                    row.resolvable ? 'text-emerald-300' : 'text-rose-300'}`}>
                    {row.resolvable ? 'yes' : 'no'}
                  </td>
                  <td className="pr-3">
                    <span className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${
                      row.status === 'RETIRED'
                        ? 'text-rose-300 border-rose-500/30 bg-rose-500/5'
                        : 'text-emerald-300 border-emerald-500/30 bg-emerald-500/5'}`}>
                      {row.status}
                    </span>
                  </td>
                  <td>
                    <button onClick={() => openCampaignRecord(row.campaign_id)}
                            className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px]
                                       focus:outline-none focus:ring-2 focus:ring-teal-400">
                      Read design
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {openCampaign && (
          <div className="border border-slate-800 rounded p-4 space-y-3 bg-slate-950/40">
            {/* The retirement precedes the design it retires. A reader who has reached the
                calendar split is already reading it as a live plan. */}
            {openCampaign.retired_by && (
              <div role="alert"
                   className="border border-rose-500/40 bg-rose-500/5 rounded p-3 space-y-1">
                <p className="text-xs font-semibold text-rose-200 flex gap-2 items-center">
                  <Archive className="w-4 h-4" aria-hidden="true" />
                  Retired — acquisition {openCampaign.retired_by.acquisition}
                </p>
                <p className="text-[11px] text-rose-100/80">{openCampaign.retired_by.statement}</p>
                <p className="text-[11px] text-slate-400">
                  Superseded by <span className="font-mono">
                    {openCampaign.retired_by.successor_campaign_id}</span> under{' '}
                  <span className="font-mono">{openCampaign.retired_by.supersession_id}</span>.
                </p>
              </div>
            )}
            <h4 className="text-xs font-semibold text-slate-200 font-mono">
              {openCampaign.campaign_id}
            </h4>
            <p className="text-[10px] text-slate-500 font-mono break-all">
              campaign sha256 {openCampaign.campaign_sha256}
            </p>
            <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-[11px]">
              {[['Train', `${split.train_start} → ${split.train_end}`],
                ['Embargo', `${split.embargo_start} → ${split.embargo_end}`],
                ['Confirmatory', `${split.test_start} → ${split.test_end}`],
                ['Declared family', `${design?.hypothesis_family_size} tests`],
                ['Distinct alignments (confirmatory)',
                 `${resolution.test?.distinct_shifts ?? '—'} of ${resolution.test?.shifts_required ?? '—'} required`],
                ['Resolves its family', design?.resolvable ? 'yes' : 'no']]
                .map(([label, value]) => (
                  <div key={String(label)} className="bg-slate-900/60 border border-slate-800 rounded p-2">
                    <dt className="text-slate-500">{label}</dt>
                    <dd className="text-slate-200 mt-0.5">{String(value)}</dd>
                  </div>
                ))}
            </dl>
            <p className="text-[11px] text-slate-400"><span className="text-slate-500">Decision rule:
              </span> {openCampaign.decision_rule}</p>
            <p className="text-[11px] text-slate-500 italic">{openCampaign.claim_boundary}</p>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------- the retirement record */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3"
               aria-labelledby="gate-supersessions-title">
        <h3 id="gate-supersessions-title"
            className="text-sm font-semibold text-slate-200 flex gap-2 items-center">
          <ScrollText className="w-4 h-4 text-violet-400" aria-hidden="true" />
          Retirements
        </h3>
        <p className="text-[11px] text-slate-500 max-w-3xl">
          A frozen design is not retired by assertion. Every stated reason is a check, and the
          server re-runs each one against both campaigns when you open it — so what you read
          below is the outcome now, not a claim the record made about itself when it was written.
          A reason is admissible only where the retired design fails it and the successor passes.
        </p>
        <ul className="space-y-2">
          {supersessions.map((row) => (
            <li key={row.supersession_id}
                className="flex flex-wrap gap-3 justify-between items-center border border-slate-800
                           rounded p-3 bg-slate-950/40">
              <div className="text-[11px]">
                <p className="text-slate-300 font-mono">{row.superseded_campaign_id} → {row.successor_campaign_id}</p>
                <p className="text-slate-500 mt-0.5">
                  {row.reason_count} checked reason(s) for {row.defects.join(', ')};{' '}
                  {row.preserved_count} preserved invariant(s); deferred to run:{' '}
                  {row.deferred_to_run.join(', ')}
                </p>
              </div>
              <button onClick={() => openSupersessionRecord(row.supersession_id)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px]
                                 focus:outline-none focus:ring-2 focus:ring-teal-400">
                Re-run the checks
              </button>
            </li>
          ))}
          {supersessions.length === 0 && (
            <li className="text-[11px] text-slate-500">No design has been retired.</li>
          )}
        </ul>

        {openSupersession && (
          <div className="border border-slate-800 rounded p-4 space-y-4 bg-slate-950/40">
            <p className="text-[10px] text-slate-500 font-mono break-all">
              supersession sha256 {openSupersession.supersession_sha256}
            </p>
            {[['Reasons the retirement stands on', openSupersession.reasons],
              ['Invariants the repair had to preserve', openSupersession.preserved]]
              .map(([heading, rows]) => (
                <div key={String(heading)}>
                  <h4 className="text-xs font-semibold text-slate-300">{String(heading)}</h4>
                  <div className="overflow-x-auto mt-2">
                    <table className="w-full text-[11px]" aria-label={String(heading)}>
                      <thead><tr className="text-left text-slate-500 border-b border-slate-800">
                        <th className="py-2">Check</th><th>Statement</th>
                        <th>Retired design</th><th>Successor</th>
                      </tr></thead>
                      <tbody>
                        {(rows as types.GateSupersessionFinding[]).map((finding) => (
                          <tr key={finding.check + finding.statement}
                              className="border-b border-slate-800/60 align-top">
                            <td className="py-2 pr-3 text-slate-300 font-mono">
                              {finding.check}
                              {finding.defect && <span className="text-slate-500"> ({finding.defect})</span>}
                            </td>
                            <td className="pr-3 text-slate-400 max-w-md">{finding.statement}</td>
                            {[finding.superseded, finding.successor].map((outcome, index) => (
                              <td key={index} className={`pr-3 font-semibold ${
                                outcome.passes ? 'text-emerald-300' : 'text-rose-300'}`}>
                                {outcome.passes ? 'passes' : 'fails'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}

            {/* The honesty field. Rendered as prominently as the reasons, because a retirement
                that looks like it closed its defects is the failure mode this record prevents. */}
            <div className="border border-amber-500/30 bg-amber-500/5 rounded p-3">
              <p className="text-xs font-semibold text-amber-200 flex gap-2 items-center">
                <AlertTriangle className="w-4 h-4" aria-hidden="true" />
                What this retirement does not settle
              </p>
              <ul className="mt-2 space-y-2 text-[11px] text-amber-100/80">
                {openSupersession.deferred_to_run.map((entry) => (
                  <li key={entry.defect}>
                    <span className="font-mono text-amber-200">{entry.defect}</span> — {entry.statement}
                    <span className="text-slate-400"> Adjudicated by {entry.adjudicated_by}.</span>
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-[11px] text-slate-500 italic">{openSupersession.claim_boundary}</p>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------------ receipts */}
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3"
               aria-labelledby="gate-receipts-title">
        <h3 id="gate-receipts-title" className="text-sm font-semibold text-slate-200">
          Published runs
        </h3>
        {receipts && receipts.status === 'NOT_YET_MEASURED' && (
          <p role="status"
             className="border border-amber-500/30 bg-amber-500/5 rounded p-3 text-[11px] text-amber-100/90">
            <span className="font-semibold text-amber-200">NOT_YET_MEASURED. </span>
            {receipts.statement}
          </p>
        )}
        {receipts && receipts.receipts.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]" aria-label="Published gate receipts">
              <thead><tr className="text-left text-slate-500 border-b border-slate-800">
                <th className="py-2">Receipt</th><th>Study</th><th>Role</th>
                <th>Gate verdict</th><th>Scientific verdict</th><th />
              </tr></thead>
              <tbody>
                {receipts.receipts.map((row) => (
                  <tr key={row.receipt_id} className="border-b border-slate-800/60">
                    <td className="py-2 pr-3 text-slate-300 font-mono">{row.receipt_id}</td>
                    <td className="pr-3 text-slate-400">{row.study_id}</td>
                    <td className="pr-3 text-slate-400">{row.evidence_role}</td>
                    <td className="pr-3 text-slate-300">{row.gate_verdict}</td>
                    <td className={`pr-3 font-semibold ${
                      row.scientific_verdict === row.gate_verdict
                        ? 'text-slate-300' : 'text-amber-200'}`}>
                      {row.scientific_verdict}
                      {row.power_applied && <span className="text-slate-500"> (power applied)</span>}
                    </td>
                    <td>
                      <button onClick={() => openReceiptRecord(row.receipt_id)}
                              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px]
                                         focus:outline-none focus:ring-2 focus:ring-teal-400">
                        Read receipt
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {openReceipt && (
          <div className="border border-slate-800 rounded p-4 space-y-3 bg-slate-950/40">
            <p className="text-[10px] text-slate-500 font-mono break-all">
              receipt sha256 {openReceipt.receipt.receipt_sha256} · integrity {openReceipt.integrity}
            </p>
            {/* Both verdicts, always, and which rule moved which. Showing the scientific verdict
                alone would hide the FAIL/INVALID boundary; showing the gate's alone would report
                an unpowered absence as a negative finding. */}
            <div className="grid sm:grid-cols-2 gap-3">
              <div className="bg-slate-900/60 border border-slate-800 rounded p-3">
                <p className="text-[11px] text-slate-500">Replication rule returned</p>
                <p className="text-sm font-semibold text-slate-200">{openReceipt.gate_verdict}</p>
              </div>
              <div className="bg-slate-900/60 border border-slate-800 rounded p-3">
                <p className="text-[11px] text-slate-500">Scientific verdict — read this one</p>
                <p className="text-sm font-semibold text-slate-200">{openReceipt.scientific_verdict}</p>
              </div>
            </div>
            <p className="text-[11px] text-slate-400">
              <span className="text-slate-500">
                {openReceipt.power_adjudication.power_applied
                  ? 'The derived spatial-power record moved this verdict: '
                  : 'The derived spatial-power record was published but adjudicated nothing: '}
              </span>
              {openReceipt.power_adjudication.reason}
            </p>
            <p className="text-[11px] text-slate-500 italic">{openReceipt.claim_boundary}</p>
          </div>
        )}
      </section>
    </div>
  );
}
