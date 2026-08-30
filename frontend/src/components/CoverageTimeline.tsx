import * as types from '../types/api';

/**
 * Actual coverage, drawn as support (TG17.4).
 *
 * Each row is one record's `[start, end)` support inside the window, positioned by time rather
 * than by index. A record with 1,344 rows and one two-day gap looks like a bar with a gap in
 * it, not like a longer bar. That is the whole point: the eye should see the same thing the
 * arithmetic sees, so that a sparse record cannot look like a dense one because it happens to
 * have as many lines.
 *
 * The row count is printed, greyed, next to the numbers that are evidence. It is the number a
 * reader would otherwise reach for, and showing it beside the effective sample size is cheaper
 * than explaining afterwards why it was the wrong one.
 */

const fmt = (seconds: number) => {
  if (!isFinite(seconds)) return '—';
  const days = seconds / 86400;
  if (Math.abs(days) >= 1) return `${days.toFixed(days >= 10 ? 0 : 1)} d`;
  const hours = seconds / 3600;
  if (Math.abs(hours) >= 1) return `${hours.toFixed(1)} h`;
  return `${Math.round(seconds)} s`;
};

const pct = (fraction: number) => `${(100 * fraction).toFixed(1)}%`;

export function CoverageBar({ row }: { row: types.SupportCoverage }) {
  const span = row.window_end_seconds - row.window_start_seconds || 1;
  return (
    <div className="h-4 w-full bg-slate-900 border border-slate-800 rounded relative overflow-hidden"
      title={`${row.gap_count} gap(s); largest ${fmt(row.largest_gap_seconds)}`}>
      {row.intervals.map(([start, end], index) => (
        <div key={index} className="absolute top-0 bottom-0 bg-sky-600/70"
          style={{
            left: `${(100 * (start - row.window_start_seconds)) / span}%`,
            width: `${Math.max(0.15, (100 * (end - start)) / span)}%`,
          }} />
      ))}
    </div>
  );
}

export function CoverageTimeline({ report }: { report: types.AlignmentReport }) {
  const manufactured = report.pairs.reduce(
    (total, pair) => total + (pair.manufactured_overlap_seconds || 0), 0);

  return (
    <div className="space-y-4 text-xs">
      <div className="flex flex-wrap gap-3 items-baseline">
        <span className={`px-2 py-0.5 rounded text-[11px] ${report.status === 'REFUSED'
          ? 'bg-rose-900/50 text-rose-200' : 'bg-emerald-900/40 text-emerald-200'}`}>
          {report.status}
        </span>
        <span className="text-slate-400">
          kernel <span className="font-mono text-slate-200">{report.kernel.name}</span>
          {report.kernel.manufactures_simultaneity
            ? <span className="ml-2 text-amber-300">manufactures simultaneity</span>
            : <span className="ml-2 text-slate-500">transforms nothing</span>}
        </span>
        <span className="text-slate-500">window {fmt(report.window_seconds)}</span>
        <span className="text-slate-500">records: {report.record_binding}</span>
      </div>

      <div>
        <div className="text-slate-400 mb-1">Support actually covered</div>
        {report.coverage.map((row) => (
          <div key={row.label} className="mb-2">
            <div className="flex justify-between text-[11px] mb-0.5">
              <span className="text-slate-200">{row.label}</span>
              <span className="text-slate-400">
                {pct(row.covered_fraction)} covered · {row.gap_count} gap
                {row.gap_count === 1 ? '' : 's'} · native scale {fmt(row.native_scale_seconds)}
                {row.support_is_stationary ? '' : ' · support not stationary'}
              </span>
            </div>
            <CoverageBar row={row} />
            <div className="text-[10px] text-slate-600 mt-0.5">
              {row.raw_row_count} rows ({row.valid_row_count} valid) — {row.rows_are_not_evidence}
            </div>
          </div>
        ))}
      </div>

      <div>
        <div className="text-slate-400 mb-1">Shared support, pairwise</div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead className="text-slate-500">
              <tr className="text-left">
                <th className="py-1 pr-3">pair</th>
                <th className="py-1 pr-3">shared</th>
                <th className="py-1 pr-3">governing scale</th>
                <th className="py-1 pr-3">effective samples</th>
                <th className="py-1 pr-3">created by kernel</th>
                <th className="py-1">rows (not used)</th>
              </tr>
            </thead>
            <tbody>
              {report.pairs.map((pair) => (
                <tr key={`${pair.left}|${pair.right}`}
                  className={`border-t border-slate-800 ${pair.status === 'REFUSED' ? 'text-rose-300' : 'text-slate-300'}`}>
                  <td className="py-1 pr-3">{pair.left} / {pair.right}</td>
                  <td className="py-1 pr-3">{fmt(pair.overlap_seconds)}</td>
                  <td className="py-1 pr-3">{fmt(pair.governing_scale_seconds)}</td>
                  <td className="py-1 pr-3" title={pair.effective_sample_size_basis}>
                    {pair.effective_sample_size.toFixed(1)}
                  </td>
                  <td className={`py-1 pr-3 ${pair.manufactured_overlap_seconds > 0 ? 'text-amber-300' : 'text-slate-600'}`}>
                    {fmt(pair.manufactured_overlap_seconds)}
                  </td>
                  <td className="py-1 text-slate-600">
                    {pair.row_counts_not_used.left} / {pair.row_counts_not_used.right}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {manufactured > 0 && (
          <p className="mt-2 text-[11px] text-amber-300">
            {fmt(manufactured)} of the shared support above was created by the declared kernel
            rather than observed in the records.
          </p>
        )}
      </div>

      {report.refusals.length > 0 && (
        <ul className="space-y-1 text-[11px] text-rose-300">
          {report.refusals.map((refusal, index) => (
            <li key={index}>
              {refusal.pair ? `${refusal.pair.join(' / ')}: ` : ''}
              {refusal.relationship ? `${refusal.relationship}: ` : ''}{refusal.reason}
            </li>
          ))}
        </ul>
      )}

      <p className="text-[11px] text-slate-500">{report.mode_forbids}</p>
      <p className="text-[11px] text-slate-500">{report.claim_boundary}</p>
    </div>
  );
}

/** The kernel a manifest freezes, chosen from what the registry offers and what every
 *  participating adapter admits. A kernel no adapter admits is shown disabled rather than
 *  hidden: a researcher should be able to see that the operation exists and that this
 *  combination of domains cannot use it. */
export function AlignmentKernelPicker({ kernels, policy, onChange }: {
  kernels: types.AlignmentKernelDescription[];
  policy: types.AlignmentPolicy;
  onChange: (next: types.AlignmentPolicy) => void;
}) {
  const selected = kernels.find((kernel) => kernel.name === policy.kernel);
  const shared = 'bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs';

  return (
    <div className="space-y-2 text-xs">
      <label className="block text-[11px] text-slate-400">
        Alignment kernel
        <select className={`${shared} mt-1 w-full`} value={policy.kernel}
          onChange={(event) => onChange({
            ...policy, kernel: event.target.value, parameters: {},
          })}>
          {kernels.map((kernel) => (
            <option key={kernel.name} value={kernel.name}
              disabled={kernel.admitted_by.length === 0}>
              {kernel.name}{kernel.admitted_by.length === 0 ? ' — no registered adapter admits this' : ''}
            </option>
          ))}
        </select>
      </label>

      {selected && (
        <p className="text-[11px] text-slate-500">
          {selected.summary}
          {selected.manufactures_simultaneity && (
            <span className="block mt-1 text-amber-300">
              This kernel creates shared support the records do not contain. The seconds it
              creates are reported separately.
            </span>
          )}
        </p>
      )}

      {selected?.required_parameters.map((name) => (
        <label key={name} className="block text-[11px] text-slate-400">
          {name}
          <input type="number" step="any" className={`${shared} mt-1 w-full`}
            value={policy.parameters[name] ?? ''}
            placeholder="no default — this is a scientific choice"
            onChange={(event) => onChange({
              ...policy,
              parameters: {
                ...policy.parameters,
                [name]: event.target.value === '' ? (undefined as any) : parseFloat(event.target.value),
              },
            })} />
        </label>
      ))}

      <div className="grid grid-cols-2 gap-2">
        <label className="block text-[11px] text-slate-400">
          minimum shared seconds
          <input type="number" step="any" className={`${shared} mt-1 w-full`}
            value={policy.minimum_overlap_seconds}
            onChange={(event) => onChange({
              ...policy, minimum_overlap_seconds: parseFloat(event.target.value) || 0,
            })} />
        </label>
        <label className="block text-[11px] text-slate-400">
          minimum effective samples
          <input type="number" step="any" className={`${shared} mt-1 w-full`}
            value={policy.minimum_effective_samples}
            onChange={(event) => onChange({
              ...policy, minimum_effective_samples: parseFloat(event.target.value) || 0,
            })} />
        </label>
      </div>
      {selected && selected.admitted_by.length > 0 && (
        <p className="text-[10px] text-slate-600">
          admitted by {selected.admitted_by.join(', ')}
        </p>
      )}
    </div>
  );
}
