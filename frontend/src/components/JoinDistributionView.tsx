/* T4E.29: the distribution, rather than its extremes.
 *
 * T4E.18's correction stated that away from the dateline the nearest extracted feature is
 * "16.6 to 99.3 km" -- "a factor of two to three, not an order of magnitude". The median was
 * right. The range was not: SETH sits 315.1 km out and HOLA 247.7, neither anywhere near a
 * boundary, and GRETEL yields no feature at all. The claim survived a review and an adoption
 * because the record held ONE NUMBER PER STORM, so nothing could contradict it, and because the
 * aggregate was taken over a subset that was never named.
 *
 * T4E.28 recorded the distributions. This panel renders them, and it renders the one operation
 * whose silent version produced that error: excluding a group and reporting what is left.
 *
 * Four rules, each from something this went wrong on.
 *
 * *No exclusion hides a row.* Excluded storms stay in the table, marked, and their aggregate is
 * computed at EQUAL WEIGHT beside the kept one rather than beneath it. An aggregate over a
 * filtered population is a statement about that population and nothing else, and the only way to
 * check such a statement is to see what was taken out.
 *
 * *A storm with no feature is drawn, not skipped.* It has no distance, so it cannot appear on a
 * distance axis -- which is exactly why it would vanish. It gets a row saying so and stays in
 * every denominator, because dropping it improves every aggregate by removing the worst case.
 *
 * *Every distance is plotted, not the nearest.* The nearest is one tick among many. A reader
 * looking at 140 ticks spread across a frame is looking at the thing that makes "the extractor
 * found the storm" false, and no summary statistic shows it.
 *
 * *The population must be named.* The record holds two extraction passes. Reading whichever is
 * stored first is the error T4E.27 made, so this panel asks and the server refuses if it does
 * not.
 *
 * It reads. It computes no verdict, stores nothing, and there is no route behind it that would.
 */
import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, CircleOff, Ruler, Scale } from 'lucide-react';
import { apiService } from '../services/api';
import type { JoinDistribution, JoinDistributionAggregate, JoinDistributionRow } from '../types/api';

interface Props {
  onError?: (message: string) => void;
}

const POPULATIONS = ['raw_field', 'swt_planes'] as const;

/** Distances span 16 km to 3,685 km, so a linear axis renders every real storm as one pixel. */
function position(km: number, max: number): number {
  const safe = Math.max(km, 1);
  return (Math.log10(safe) / Math.log10(Math.max(max, 10))) * 100;
}

function Aggregate({ title, value, tone }: {
  title: string; value: JoinDistributionAggregate | null; tone: string;
}) {
  if (!value) {
    return (
      <div className={`rounded border p-2 text-xs ${tone}`}>
        <h4 className="font-semibold">{title}</h4>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          No storms in this group.
        </p>
      </div>
    );
  }
  return (
    <div className={`rounded border p-2 text-xs ${tone}`} data-testid={`aggregate-${title}`}>
      <h4 className="font-semibold">{title}</h4>
      <dl className="mt-1 space-y-0.5">
        <div className="flex justify-between gap-2">
          <dt>storms</dt><dd className="font-mono">{value.storms}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt>nearest km</dt>
          <dd className="font-mono">
            {value.nearest_km
              ? `${value.nearest_km.min.toFixed(1)} – ${value.nearest_km.median.toFixed(1)} – ${value.nearest_km.max.toFixed(1)}`
              : 'no distances'}
          </dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt>≥1 inside radius</dt>
          <dd className="font-mono">{value.at_least_one_inside_radius} of {value.of}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt>≥3 inside radius</dt>
          <dd className="font-mono">{value.three_inside_radius} of {value.of}</dd>
        </div>
        {value.storms_with_no_feature > 0 && (
          <div className="flex justify-between gap-2 text-amber-700 dark:text-amber-300">
            <dt>no feature at all</dt>
            <dd className="font-mono">{value.storms_with_no_feature}</dd>
          </div>
        )}
      </dl>
      <p className="mt-1 text-slate-500 dark:text-slate-400">min – median – max</p>
    </div>
  );
}

function Row({ row, max }: { row: JoinDistributionRow; max: number }) {
  const radius = position(row.radius_km, max);
  return (
    <li
      data-testid={`join-row-${row.storm}`}
      className={`rounded border p-2 ${row.excluded
        ? 'border-dashed border-amber-400 bg-amber-50/60 dark:bg-amber-950/30'
        : 'border-slate-200 dark:border-slate-700'}`}
    >
      <div className="flex flex-wrap items-baseline gap-2 text-xs">
        <span className="w-20 font-mono font-semibold">{row.storm}</span>
        <span className="text-slate-500 dark:text-slate-400">
          lon {row.lon?.toFixed(1) ?? '—'}
        </span>
        <span className="text-slate-500 dark:text-slate-400">
          radius {row.radius_km.toFixed(1)} km
        </span>
        <span className="text-slate-500 dark:text-slate-400">
          {row.features} features
        </span>
        <span className="font-mono">
          nearest {row.nearest_km == null ? '—' : `${row.nearest_km.toFixed(1)} km`}
        </span>
        {row.excluded && (
          <span
            data-testid={`excluded-${row.storm}`}
            className="flex items-center gap-1 rounded bg-amber-100 px-1.5 py-0.5
                       text-amber-900 dark:bg-amber-900 dark:text-amber-100"
          >
            <Ban className="h-3 w-3" aria-hidden="true" />
            excluded, still counted here: {row.excluded_because}
          </span>
        )}
      </div>

      {row.no_feature ? (
        <p
          data-testid={`no-feature-${row.storm}`}
          className="mt-1 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-300"
        >
          <CircleOff className="h-3 w-3 shrink-0" aria-hidden="true" />
          {row.no_feature}. It has no distance to plot and stays in every denominator.
        </p>
      ) : (
        <div
          className="relative mt-2 h-6 rounded bg-slate-100 dark:bg-slate-800"
          role="img"
          aria-label={
            `${row.storm}: ${row.features} features at ${row.distances_km.length} distances from `
            + `${row.distances_km[0].toFixed(1)} to `
            + `${row.distances_km[row.distances_km.length - 1].toFixed(1)} km; the catalogue `
            + `radius is ${row.radius_km.toFixed(1)} km and ${row.inside_radius} features are inside it`
          }
        >
          <div
            className="absolute inset-y-0 bg-emerald-200/70 dark:bg-emerald-800/50"
            style={{ left: 0, width: `${Math.max(radius, 0.6)}%` }}
            title={`catalogue radius ${row.radius_km.toFixed(1)} km`}
          />
          {row.distances_km.map((km, index) => (
            <span
              key={index}
              className={`absolute top-1 h-4 w-px ${km <= row.radius_km
                ? 'bg-emerald-700 dark:bg-emerald-300'
                : 'bg-slate-500 dark:bg-slate-400'}`}
              style={{ left: `${position(km, max)}%` }}
            />
          ))}
        </div>
      )}
      <p className="sr-only" data-testid={`transcript-${row.storm}`}>
        {row.storm}: {row.features} features, nearest{' '}
        {row.nearest_km == null ? 'none' : `${row.nearest_km.toFixed(1)} km`}, catalogue radius{' '}
        {row.radius_km.toFixed(1)} km, {row.inside_radius} inside it.
      </p>
    </li>
  );
}

export default function JoinDistributionView({ onError }: Props) {
  const [population, setPopulation] = useState<string>('raw_field');
  const [threshold, setThreshold] = useState<string>('');
  const [data, setData] = useState<JoinDistribution | null>(null);
  const [status, setStatus] = useState('Loading the join…');

  const load = useCallback(async () => {
    try {
      const parsed = threshold.trim() === '' ? null : Number(threshold);
      const next = await apiService.joinDistribution({
        population,
        excludeLongitudeAtOrAbove: Number.isFinite(parsed as number) ? parsed : null,
      });
      setData(next);
      setStatus('');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      onError?.(message);
    }
  }, [population, threshold, onError]);

  useEffect(() => { void load(); }, [load]);

  if (status) {
    return <div className="p-4 text-sm text-slate-600 dark:text-slate-300">{status}</div>;
  }
  if (!data) return null;

  const max = data.rows.reduce(
    (running, row) => Math.max(running, row.distances_km[row.distances_km.length - 1] ?? 0), 10);

  return (
    <div className="p-4 space-y-4" data-testid="join-distribution">
      <header className="space-y-1">
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <Ruler className="h-5 w-5" aria-hidden="true" />
          Every distance in the join, and what an exclusion does to the answer
        </h2>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          {data.how_to_read_a_distance}
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-3 text-sm">
        <label className="space-y-1">
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Extraction pass
          </span>
          <select
            data-testid="population"
            value={population}
            onChange={(event) => setPopulation(event.target.value)}
            className="rounded border border-slate-300 px-2 py-1 dark:border-slate-600
                       dark:bg-slate-900"
          >
            {POPULATIONS.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Exclude longitude at or above
          </span>
          <input
            data-testid="exclude-longitude"
            value={threshold}
            onChange={(event) => setThreshold(event.target.value)}
            placeholder="none"
            inputMode="decimal"
            className="w-32 rounded border border-slate-300 px-2 py-1 dark:border-slate-600
                       dark:bg-slate-900"
          />
        </label>
        <p className="max-w-md text-xs text-slate-500 dark:text-slate-400">
          {data.population.description}
        </p>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <Aggregate title="everything" value={data.everything}
                   tone="border-slate-300 dark:border-slate-600" />
        <Aggregate title="kept" value={data.kept}
                   tone="border-emerald-300 dark:border-emerald-700" />
        <Aggregate title="excluded" value={data.excluded}
                   tone="border-amber-400 dark:border-amber-600" />
      </div>

      <p
        data-testid="exclusion-note"
        className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 p-2
                   text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40
                   dark:text-amber-100"
      >
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <span>
          {data.exclusion.why_they_are_still_listed}{' '}
          <strong>{data.exclusion.what_this_will_not_do}</strong>
        </span>
      </p>

      <ul className="space-y-2">
        {data.rows.map((row) => <Row key={row.storm} row={row} max={max} />)}
      </ul>

      <section className="space-y-2 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
        <h3 className="flex items-center gap-2 text-sm font-semibold">
          <Scale className="h-4 w-4" aria-hidden="true" />
          What this surface will not do
        </h3>
        {data.verdict && (
          <p data-testid="join-verdict" className="text-xs font-medium">{data.verdict}</p>
        )}
        {data.claim_boundary && (
          <p data-testid="join-boundary"
             className="flex items-start gap-1 text-xs text-slate-600 dark:text-slate-300">
            <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
            <span>{data.claim_boundary}</span>
          </p>
        )}
        <ul className="space-y-1 text-xs text-slate-600 dark:text-slate-300">
          {data.refusals.map((refusal) => (
            <li key={refusal} data-testid="join-refusal" className="flex items-start gap-1">
              <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
              <span>{refusal}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
