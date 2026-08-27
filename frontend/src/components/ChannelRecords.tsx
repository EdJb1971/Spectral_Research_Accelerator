/**
 * Domain Records — the ingestion seam for channel tables (TG8.4).
 *
 * Peer to the meteorological tab rather than a section inside it: one tab reads grids, this one
 * reads channel tables for any declared domain. That the interface is multi-domain is a
 * structural fact here, not a claim in a document.
 *
 * **Two steps, and this component chooses neither of them.** `inspect` reports what the file is
 * and which declared domains admit it; the researcher picks the clock column and the domain;
 * `read` loads it. The domain list is rendered with each refusal spelled out, so a researcher
 * sees why a domain is unavailable before they try it.
 *
 * **Nothing scientific is computed or formatted here.** The clock facts, the refusals, the
 * caveat and the preview note all arrive as strings or plain booleans from the backend and are
 * rendered as given — Phase G9's rule, applied to a tab G9 did not write. The one formatting
 * this file does is on a raw data value or a row count, neither of which is a claim.
 */

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, CheckCircle2, Database, Upload } from 'lucide-react';

import { LineChart } from './LineChart';
import { apiService } from '../services/api';
import * as types from '../types/api';

interface ChannelRecordsProps {
  onError?: (message: string) => void;
  /** When supplied by Acquire, the researcher has already chosen the domain. */
  domainName?: string;
  /** TG11.0: shell-owned context survives navigation between workflow panels. */
  selectedRecord?: types.ChannelRecordSelection | null;
  onSelectRecord?: (record: types.ChannelRecordSelection | null) => void;
}

export const ChannelRecords: React.FC<ChannelRecordsProps> = ({
  onError, domainName, selectedRecord = null, onSelectRecord,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [inspection, setInspection] = useState<types.ChannelInspection | null>(null);
  const [record, setRecord] = useState<types.ChannelRecord | null>(selectedRecord?.record ?? null);
  const [timeColumn, setTimeColumn] = useState<string>('');
  const [domain, setDomain] = useState<string>('');
  const [supports, setSupports] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);

  useEffect(() => { setRecord(selectedRecord?.record ?? null); }, [selectedRecord]);

  const fail = useCallback((error: unknown) => {
    onError?.(error instanceof Error ? error.message : String(error));
  }, [onError]);

  const reset = () => {
    setInspection(null);
    setRecord(null);
    onSelectRecord?.(null);
    setTimeColumn('');
    setDomain('');
    setSupports({});
  };

  const onChoose = async (chosen: File | null) => {
    setFile(chosen);
    reset();
    if (!chosen) return;
    setBusy(true);
    try {
      const report = await apiService.inspectChannelRecord(chosen);
      setInspection(report);
      setTimeColumn(report.time_column);
      // The first admitting domain is offered as a *starting value the researcher can see and
      // change*, never as a silent default: the difference is that it appears on screen and is
      // recorded in the provenance of whatever is loaded.
      const admitting = domainName
        ? report.domains.find((d) => d.name === domainName && d.admits)
        : report.domains.find((d) => d.admits);
      setDomain(admitting ? admitting.name : '');
    } catch (error) {
      fail(error);
      setFile(null);
    } finally {
      setBusy(false);
    }
  };

  const onRead = async () => {
    if (!file || !domain || !timeColumn) return;
    setBusy(true);
    try {
      const loaded = await apiService.readChannelRecord(file, domain, timeColumn,
        { supportParentPx: supports });
      setRecord(loaded);
      onSelectRecord?.({ record: loaded, file, timeColumn, supportParentPx: supports });
    } catch (error) {
      setRecord(null);
      onSelectRecord?.(null);
      fail(error);
    } finally {
      setBusy(false);
    }
  };

  const reInspect = async (column: string) => {
    setTimeColumn(column);
    setRecord(null);
    onSelectRecord?.(null);
    if (!file) return;
    setBusy(true);
    try {
      setInspection(await apiService.inspectChannelRecord(file, { timeColumn: column }));
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <header>
        <h3 className="text-lg font-semibold text-slate-100">Domain records</h3>
        <p className="text-sm text-slate-400 max-w-3xl">
          A channel table — one clock column and one column per channel — read under a domain
          that has declared what it is and what it breaks. What the file is, and what that
          obliges a domain to have declared, are reported here; which domain to read it under is
          your choice, and nothing is filled in on your behalf.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* -------------------------------------------------- the file and the two choices */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
            <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Upload className="w-4 h-4 text-teal-400" aria-hidden="true" /> Choose a record
            </h4>
            <label htmlFor="channel-file" className="sr-only">Channel record file</label>
            <input
              id="channel-file"
              type="file"
              accept=".csv,.tsv,.txt,text/csv"
              onChange={(event) => void onChoose(event.target.files?.[0] || null)}
              className="block w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3
                         file:rounded file:border-0 file:text-xs file:bg-slate-800
                         file:text-slate-200 hover:file:bg-slate-700"
            />
            <p className="text-xs text-slate-500">
              Delimited UTF-8 text. A gridded file (NetCDF, Zarr) belongs to the meteorological
              tab, which reads grids rather than channels.
            </p>
          </div>

          {inspection && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
              <h4 className="text-sm font-semibold text-slate-200">What this file is</h4>
              <dl className="text-xs space-y-1">
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">rows</dt>
                  <dd className="text-slate-200">{inspection.n_rows}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">columns</dt>
                  <dd className="text-slate-200 text-right">{inspection.columns.join(', ')}</dd>
                </div>
                {inspection.clock && (
                  <>
                    <div className="flex justify-between gap-3">
                      <dt className="text-slate-500">clock</dt>
                      <dd className="text-slate-200">
                        {inspection.clock.regular ? 'regularly sampled' : 'irregular'}
                      </dd>
                    </div>
                    {inspection.clock.cadence_seconds !== null && (
                      <div className="flex justify-between gap-3">
                        <dt className="text-slate-500">cadence (s)</dt>
                        <dd className="text-slate-200">{inspection.clock.cadence_seconds}</dd>
                      </div>
                    )}
                  </>
                )}
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">digest</dt>
                  <dd className="text-slate-500 font-mono break-all text-right">
                    {inspection.content_sha256.slice(0, 16)}…
                  </dd>
                </div>
              </dl>

              {!inspection.readable && inspection.refused_because && (
                <p className="text-xs text-amber-300 border border-amber-900/50 rounded p-2
                              bg-amber-950/20 flex gap-2">
                  <Ban size={14} className="shrink-0 mt-0.5" aria-hidden="true" />
                  <span>{inspection.refused_because}</span>
                </p>
              )}

              {inspection.readable && (
                <div>
                  <label htmlFor="channel-clock"
                    className="block text-xs uppercase tracking-wide text-slate-400 mb-1">
                    Clock column
                  </label>
                  <select
                    id="channel-clock"
                    value={timeColumn}
                    onChange={(event) => void reInspect(event.target.value)}
                    className="w-full bg-slate-800 text-slate-100 text-sm rounded px-2 py-2
                               focus:outline-none focus:ring-2 focus:ring-teal-400"
                  >
                    {inspection.candidate_time_columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </div>
              )}

              {inspection.required_violations.length > 0 && (
                <div className="text-xs">
                  <p className="text-slate-400 mb-1">
                    This file&apos;s shape requires the domain you choose to have already
                    declared:
                  </p>
                  <ul className="space-y-1">
                    {inspection.required_violations.map((violation) => (
                      <li key={violation} className="font-mono text-amber-300">{violation}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* -------------------------------------------------- the domains and their verdicts */}
        <div className="lg:col-span-2 space-y-4">
          {inspection && inspection.domains.length > 0 && (
            <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4"
              aria-label="Declared domains and whether they admit this record">
              <h4 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <Database className="w-4 h-4 text-teal-400" aria-hidden="true" />
                {domainName ? `Read it under ${domainName}` : 'Read it under which domain?'}
              </h4>
              <ul className="space-y-2">
                {inspection.domains.filter((row) => !domainName || row.name === domainName).map((row) => (
                  <li key={row.name}>
                    <label
                      className={`flex gap-3 items-start border rounded p-2 cursor-pointer ${
                        row.admits
                          ? 'border-slate-800 hover:border-teal-700'
                          : 'border-slate-900 opacity-70 cursor-not-allowed'
                      }`}
                    >
                      <input
                        type="radio"
                        name="channel-domain"
                        value={row.name}
                        checked={domain === row.name}
                        disabled={!row.admits}
                        onChange={() => {
                          setDomain(row.name);
                          setRecord(null);
                          onSelectRecord?.(null);
                        }}
                        className="mt-1"
                      />
                      <span className="text-xs flex-1">
                        <span className="flex items-center gap-2">
                          <span className="text-slate-200 font-medium">{row.name}</span>
                          {row.admits ? (
                            <CheckCircle2 size={13} className="text-teal-400"
                              aria-label="admits this record" />
                          ) : (
                            <Ban size={13} className="text-amber-400"
                              aria-label="refuses this record" />
                          )}
                        </span>
                        {row.refusals.map((refusal) => (
                          <span key={refusal} className="block text-amber-400/90 mt-1">
                            {refusal}
                          </span>
                        ))}
                        {row.admits && (
                          <span className="block text-slate-500 mt-1">
                            declares: {row.violations.join(', ') || 'no violations'}
                          </span>
                        )}
                      </span>
                    </label>
                  </li>
                ))}
              </ul>

              {inspection.readable && (
                <div className="mt-3 space-y-2">
                  {inspection.channel_columns.length > 0 && (
                    <details className="text-xs">
                      <summary className="text-slate-400 cursor-pointer">
                        Declare a channel as a window aggregate
                      </summary>
                      <p className="text-slate-500 mt-2">{inspection.aggregate_note}</p>
                      <div className="mt-2 space-y-1">
                        {inspection.channel_columns.map((channel) => (
                          <div key={channel} className="flex items-center gap-2">
                            <label htmlFor={`support-${channel}`}
                              className="text-slate-400 w-32 shrink-0 truncate">{channel}</label>
                            <input
                              id={`support-${channel}`}
                              type="number"
                              min={1}
                              step={1}
                              value={supports[channel] ?? 1}
                              onChange={(event) => {
                                const parsed = Number(event.target.value);
                                setSupports((current) => ({ ...current, [channel]: parsed }));
                                setRecord(null);
                                onSelectRecord?.(null);
                              }}
                              className="w-24 bg-slate-800 text-slate-100 rounded px-2 py-1
                                         focus:outline-none focus:ring-2 focus:ring-teal-400"
                            />
                            <span className="text-slate-600">clock samples</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  <button
                    type="button"
                    onClick={() => void onRead()}
                    disabled={busy || !domain || !timeColumn}
                    className="px-3 py-2 text-sm rounded bg-teal-700 hover:bg-teal-600
                               disabled:bg-slate-800 disabled:text-slate-600 text-white
                               focus:outline-none focus:ring-2 focus:ring-teal-400"
                  >
                    {busy ? 'Reading…' : 'Read this record'}
                  </button>
                </div>
              )}
            </section>
          )}

          {/* ---------------------------------------------------------------- the loaded record */}
          {record && (
            <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-4"
              aria-label="The loaded record">
              <div className="flex items-baseline justify-between gap-3">
                <h4 className="text-sm font-semibold text-slate-200">
                  {record.source_name} read under {record.domain}
                </h4>
                <span className="text-xs text-slate-500">
                  {record.n_rows} rows · {record.channels.length} channels
                </span>
              </div>

              <LineChart
                series={record.channels.map((channel) => ({
                  name: channel.name,
                  x: record.times_seconds,
                  y: channel.values,
                }))}
                title="Channel preview"
                xLabel="clock (s)"
                yLabel="value"
                divId="channel-record-preview"
              />
              <p className="text-xs text-slate-500">{record.preview_note}</p>
              {record.rows_withheld > 0 && (
                <p className="text-xs text-amber-300">
                  Showing the first {record.preview_rows} rows; {record.rows_withheld} were not
                  sent to the browser. The record itself was read whole and nothing was thinned.
                </p>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <section aria-label="Channels in this record">
                  <h5 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
                    Channels
                  </h5>
                  <ul className="text-xs space-y-1">
                    {record.channels.map((channel) => (
                      <li key={channel.name}
                        className="flex justify-between gap-3 border-b border-slate-800 py-1">
                        <span className="text-slate-200">{channel.name}</span>
                        <span className="text-slate-500">
                          {channel.is_aggregate
                            ? `aggregate over ${channel.support_parent_px} samples`
                            : 'instantaneous'}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <dl className="text-xs mt-3 space-y-1">
                    <div className="flex justify-between gap-3">
                      <dt className="text-slate-500">clock</dt>
                      <dd className="text-slate-200">
                        {record.clock.regular ? 'regularly sampled' : 'irregular (declared)'}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt className="text-slate-500">digest</dt>
                      <dd className="text-slate-500 font-mono break-all text-right">
                        {record.content_sha256}
                      </dd>
                    </div>
                  </dl>
                </section>

                <section aria-label="What this domain refuses">
                  <h5 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
                    What {record.domain} refuses
                  </h5>
                  <ul className="text-xs space-y-2">
                    {record.domain_limits.refuses.map((refusal) => (
                      <li key={refusal.basis} className="border border-slate-800 rounded p-2">
                        <span className="font-mono text-slate-500 block">{refusal.basis}</span>
                        <span className="text-slate-300">{refusal.consequence}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              </div>

              <p className="text-xs text-amber-300/90 border border-amber-900/50 rounded p-2
                            bg-amber-950/20 flex gap-2">
                <AlertTriangle size={14} className="shrink-0 mt-0.5" aria-hidden="true" />
                <span>{record.domain_limits.attribution_caveat}</span>
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChannelRecords;
