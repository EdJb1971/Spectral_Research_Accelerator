/* TG19.2: the join's bar, shown in parts.
 *
 * T4E.18 judged whether an extracted feature lands at a catalogue cyclone centre using the
 * catalogue's own per-observation radius, and failed. T4E.27 showed that bar was wrong for three
 * reasons -- it bounds the wrong quantity, one observation's radius was exactly 0.00 and could
 * never have been satisfied, and the offset does not track the radius at all -- and that
 * replacing it with a defensible one changed nothing.
 *
 * Both halves of that matter to a researcher and neither was visible. A bar that appears as a
 * single number can only be accepted or rejected; one that shows its components, where each came
 * from, what it deliberately leaves out and what it refuses can be disagreed with specifically.
 * This panel is that surface.
 *
 * It computes. It does not decide: nothing here stores a tolerance, approves a join or records
 * an acceptance, and the server serves no route that would.
 */
import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, Ruler, Scale } from 'lucide-react';
import { apiService } from '../services/api';
import type { PositionToleranceView as ToleranceView, ToleranceComponents } from '../types/api';

interface Props {
  onError?: (message: string) => void;
}

export default function PositionToleranceView({ onError }: Props) {
  const [components, setComponents] = useState<ToleranceComponents | null>(null);
  const [radius, setRadius] = useState<string>('11.12');
  const [separation, setSeparation] = useState<string>('99.98');
  const [observation, setObservation] = useState<string>('OWEN');
  const [result, setResult] = useState<ToleranceView | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiService.toleranceComponents()
      .then((data: ToleranceComponents) => { if (!cancelled) setComponents(data); })
      .catch((e: unknown) => onError?.(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; };
  }, [onError]);

  const compute = useCallback(async () => {
    setBusy(true);
    try {
      const parsed = radius.trim() === '' ? null : Number(radius);
      const sep = separation.trim() === '' ? null : Number(separation);
      setResult(await apiService.positionTolerance({
        catalogueRadiusKm: parsed, separationKm: sep, observation: observation || 'unnamed',
      }));
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [radius, separation, observation, onError]);

  useEffect(() => { void compute(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-6" data-testid="position-tolerance">
      <header className="space-y-2">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Ruler className="w-5 h-5" aria-hidden="true" />
          Position matching tolerance
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 max-w-3xl">
          How close a feature must land to a catalogue position to count as the same thing —
          shown in parts, so it can be argued with rather than only accepted. This panel computes
          a bar. It does not decide an acceptance, and no served route stores one.
        </p>
      </header>

      {components && (
        <section className="rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-3"
                 data-testid="tolerance-components">
          <h3 className="font-medium flex items-center gap-2">
            <Scale className="w-4 h-4" aria-hidden="true" /> What the bar is made of
          </h3>
          <ul className="space-y-2">
            {components.components.map((component) => (
              <li key={component.name}
                  className="text-sm border-l-2 border-slate-300 dark:border-slate-600 pl-3">
                <div className="font-mono">{component.name}</div>
                <div className="text-slate-600 dark:text-slate-300">{component.source}</div>
                <div className="text-xs text-slate-500">supplied by {component.supplied_by}</div>
              </li>
            ))}
          </ul>
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium">Combined by:</span> {components.combined_by}
          </p>

          {/* What a bar leaves out decides what its residual means, so it renders at the same
              weight as what it includes rather than as a footnote. */}
          <div className="rounded border border-amber-300 dark:border-amber-700
                          bg-amber-50 dark:bg-amber-950/40 p-3" data-testid="tolerance-excluded">
            <div className="font-medium flex items-center gap-2 text-sm">
              <AlertTriangle className="w-4 h-4" aria-hidden="true" /> Deliberately not included
            </div>
            <p className="text-sm mt-1">{components.excluded}</p>
          </div>

          <div className="rounded border border-slate-300 dark:border-slate-600 p-3"
               data-testid="tolerance-missing-uncertainty">
            <div className="font-medium text-sm">Why a missing uncertainty is refused</div>
            <p className="text-sm mt-1">{components.why_a_missing_uncertainty_is_refused}</p>
          </div>
        </section>
      )}

      <section className="rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-3">
        <h3 className="font-medium">Compute a bar</h3>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm space-y-1">
            <span className="block font-medium">Observation</span>
            <input className="w-full rounded border border-slate-300 dark:border-slate-600
                              bg-white dark:bg-slate-900 px-2 py-1"
                   value={observation} onChange={(e) => setObservation(e.target.value)} />
          </label>
          <label className="text-sm space-y-1">
            <span className="block font-medium">Catalogue radius (km)</span>
            <input className="w-full rounded border border-slate-300 dark:border-slate-600
                              bg-white dark:bg-slate-900 px-2 py-1"
                   inputMode="decimal" value={radius}
                   onChange={(e) => setRadius(e.target.value)} />
          </label>
          <label className="text-sm space-y-1">
            <span className="block font-medium">Separation (km, optional)</span>
            <input className="w-full rounded border border-slate-300 dark:border-slate-600
                              bg-white dark:bg-slate-900 px-2 py-1"
                   inputMode="decimal" value={separation}
                   onChange={(e) => setSeparation(e.target.value)} />
          </label>
        </div>
        <button type="button" onClick={() => void compute()} disabled={busy}
                className="rounded bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900
                           px-3 py-1.5 text-sm disabled:opacity-50">
          {busy ? 'Computing…' : 'Compute'}
        </button>
      </section>

      {result && (
        <section className="rounded-lg border border-slate-200 dark:border-slate-700 p-4 space-y-3"
                 data-testid="tolerance-result">
          <h3 className="font-medium">{result.observation}</h3>

          {/* A refusal renders at the weight of a value. It is not an error state: the bar
              could not be built, so nothing was judged, and saying "no" here would count a
              missing catalogue uncertainty against the instrument. */}
          {result.refused ? (
            <div className="rounded border border-slate-400 dark:border-slate-500
                            bg-slate-50 dark:bg-slate-800/60 p-3" data-testid="tolerance-refusal">
              <div className="font-medium flex items-center gap-2">
                <Ban className="w-4 h-4" aria-hidden="true" /> Refused — no bar was built
              </div>
              <p className="text-sm mt-1">{result.refusal}</p>
              {result.why_no_verdict && (
                <p className="text-sm mt-2" data-testid="tolerance-no-verdict">
                  {result.why_no_verdict}
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-2xl font-mono" data-testid="tolerance-total">
                {result.total_km?.toFixed(2)} km
              </div>
              <ul className="text-sm space-y-1">
                {result.components.map((component) => (
                  <li key={component.name} className="flex justify-between gap-4">
                    <span className="font-mono">{component.name}</span>
                    <span className="font-mono">
                      {component.km === null ? 'not supplied' : `${component.km.toFixed(2)} km`}
                    </span>
                  </li>
                ))}
              </ul>
              {result.separation_km !== null && (
                <div className="pt-2 border-t border-slate-200 dark:border-slate-700 text-sm">
                  <div data-testid="tolerance-verdict">
                    Separation {result.separation_km.toFixed(2)} km —{' '}
                    <span className="font-medium">
                      {result.admitted ? 'inside the bar' : 'outside the bar'}
                    </span>
                  </div>
                  {result.unexplained_residual_km !== null && (
                    <div className="mt-1" data-testid="tolerance-residual">
                      Unexplained residual{' '}
                      <span className="font-mono">
                        {result.unexplained_residual_km.toFixed(2)} km
                      </span>
                      {' '}— this is the measure of the component the bar leaves out.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <p className="text-xs text-slate-500" data-testid="tolerance-boundary">
            {result.what_is_not_included}
          </p>
        </section>
      )}

      {components && (
        <section className="rounded-lg border border-slate-200 dark:border-slate-700 p-4"
                 data-testid="tolerance-refusals">
          <h3 className="font-medium text-sm">What this surface will not do</h3>
          <ul className="mt-2 space-y-1 text-sm list-disc list-inside">
            {components.refusals.map((refusal) => <li key={refusal}>{refusal}</li>)}
          </ul>
        </section>
      )}
    </div>
  );
}
