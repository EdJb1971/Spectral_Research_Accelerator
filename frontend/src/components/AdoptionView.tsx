/* T4E.32: adoption, performed here instead of in a text editor.
 *
 * Every adoption in this repository was a hand-written JSON file. The rule that produced them
 * stands and is not relaxed -- code does not sign a scientific declaration for a person -- but
 * that rule was being enforced by the awkwardness of the medium, which is a poor place to keep a
 * principle. A maintainer who reads a declaration, types their name and types the affirmation has
 * signed it. Editing a file by hand is not more deliberate than that; it is only slower.
 *
 * Three things this panel does that a form usually does not.
 *
 * *It supplies no default for anything that is the scientific act.* Not the name, not the reason,
 * not the affirmation. A field pre-filled with something plausible would be signing on the
 * maintainer's behalf while appearing to ask.
 *
 * *It shows the digest being signed, next to the button.* An adoption binds a sha256, and a
 * signature that reaches a different text than the one on screen is the failure the binding
 * exists to prevent.
 *
 * *It reports drift as loudly as it reports adoption.* A declaration amended after signing is a
 * different declaration; the row says so rather than continuing to read as signed.
 */
import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Ban, CheckCircle2, FileSignature, PenLine } from 'lucide-react';
import { apiService } from '../services/api';
import type { DeclarationIndex, DeclarationRow } from '../types/api';
import DeclarationComposer from './DeclarationComposer';

interface Props {
  onError?: (message: string) => void;
}

export default function AdoptionView({ onError }: Props) {
  const [index, setIndex] = useState<DeclarationIndex | null>(null);
  const [open, setOpen] = useState<string>('');
  const [name, setName] = useState('');
  const [adoptedAs, setAdoptedAs] = useState('');
  const [what, setWhat] = useState('');
  const [why, setWhy] = useState('');
  const [affirmation, setAffirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState('');
  const [signed, setSigned] = useState('');

  const load = useCallback(async () => {
    try {
      setIndex(await apiService.listDeclarations());
    } catch (error) {
      onError?.(error instanceof Error ? error.message : String(error));
    }
  }, [onError]);

  useEffect(() => { void load(); }, [load]);

  const sign = useCallback(async () => {
    setBusy(true);
    setRefusal('');
    setSigned('');
    try {
      const result = await apiService.signDeclaration({
        declaration: open, adopted_by: name, adopted_as: adoptedAs,
        what_was_adopted: what, affirmation, why: why || undefined,
      });
      setSigned(`${result.adopted.file} — bound to ${result.adopted.adopts_sha256.slice(0, 16)}…`);
      setAffirmation('');
      await load();
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [open, name, adoptedAs, what, why, affirmation, load]);

  if (!index) return <div className="p-4 text-sm text-slate-400">Loading declarations…</div>;

  const unsigned = index.declarations.filter((row) => !row.adopted);
  const current = index.declarations.find((row) => row.declaration === open);

  const Row = ({ row }: { row: DeclarationRow }) => (
    <li data-testid={`declaration-${row.declaration}`}
        className="rounded border border-slate-200 p-2 text-xs dark:border-slate-700">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono">{row.declaration}</span>
        {row.task && <span className="text-slate-500">{row.task}</span>}
        {row.adopted ? (
          <span data-testid={`adopted-${row.declaration}`}
                className="flex items-center gap-1 rounded bg-emerald-100 px-1.5 py-0.5
                           text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100">
            <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
            adopted by {row.adopted_by} on {row.adopted_on}
          </span>
        ) : (
          <button type="button" data-testid={`sign-${row.declaration}`}
                  onClick={() => { setOpen(row.declaration); setRefusal(''); setSigned(''); }}
                  className="flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 text-white
                             hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900">
            <PenLine className="h-3 w-3" aria-hidden="true" />
            adopt this
          </button>
        )}
      </div>
      {row.status && <p className="mt-1 text-slate-500">{row.status}</p>}
      {row.signature_still_reaches_the_declaration === false && (
        <p data-testid={`drift-${row.declaration}`}
           className="mt-1 flex items-start gap-1 rounded bg-rose-100 p-1 text-rose-900
                      dark:bg-rose-950 dark:text-rose-100">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
          <span>{row.drift}</span>
        </p>
      )}
    </li>
  );

  return (
    <div className="p-4 space-y-4" data-testid="adoption">
      <header className="space-y-1">
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <FileSignature className="h-5 w-5" aria-hidden="true" />
          Declarations, and the ones nobody has signed
        </h2>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          {index.what_signing_means}
        </p>
        <p className="max-w-3xl text-xs text-slate-500 dark:text-slate-400">
          {index.what_this_surface_will_not_supply}
        </p>
      </header>

      {/* T4E.34. Signing a declaration was a button before writing one was; the composer closes
          the larger half, and its commit-alone control is what makes a later evidence bundle
          possible at all. */}
      <DeclarationComposer onComposed={() => void load()} />

      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Awaiting adoption ({unsigned.length})
        </h3>
        <ul className="space-y-2">{unsigned.map((row) => <Row key={row.declaration} row={row} />)}</ul>
      </section>

      {current && (
        <section data-testid="sign-form"
                 className="space-y-3 rounded-lg border-2 border-slate-300 p-3
                            dark:border-slate-600">
          <h3 className="font-semibold">Adopting {current.declaration}</h3>
          <p className="font-mono text-xs text-slate-500" data-testid="signing-digest">
            binds sha256 {current.declaration_sha256}
          </p>

          <label className="block space-y-1 text-sm">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Adopted by — your name, not a role
            </span>
            <input data-testid="adopted-by" value={name}
                   onChange={(event) => setName(event.target.value)}
                   className="w-full rounded border border-slate-300 px-2 py-1
                              dark:border-slate-600 dark:bg-slate-900" />
          </label>

          <label className="block space-y-1 text-sm">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Adopted as
            </span>
            <input data-testid="adopted-as" value={adoptedAs}
                   onChange={(event) => setAdoptedAs(event.target.value)}
                   placeholder="e.g. REPRODUCTION_GATE"
                   className="w-full rounded border border-slate-300 px-2 py-1
                              dark:border-slate-600 dark:bg-slate-900" />
          </label>

          <label className="block space-y-1 text-sm">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              What was adopted
            </span>
            <textarea data-testid="what-was-adopted" value={what} rows={3}
                      onChange={(event) => setWhat(event.target.value)}
                      className="w-full rounded border border-slate-300 px-2 py-1
                                 dark:border-slate-600 dark:bg-slate-900" />
          </label>

          <label className="block space-y-1 text-sm">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Why you adopted it (optional)
            </span>
            <textarea data-testid="why" value={why} rows={2}
                      onChange={(event) => setWhy(event.target.value)}
                      className="w-full rounded border border-slate-300 px-2 py-1
                                 dark:border-slate-600 dark:bg-slate-900" />
          </label>

          <label className="block space-y-1 text-sm">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Type: {index.required_affirmation}
            </span>
            <input data-testid="affirmation" value={affirmation}
                   onChange={(event) => setAffirmation(event.target.value)}
                   className="w-full rounded border border-slate-300 px-2 py-1
                              dark:border-slate-600 dark:bg-slate-900" />
          </label>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            A signature that can be produced by one click is one that can be produced by accident.
          </p>

          <button type="button" data-testid="sign-submit" disabled={busy} onClick={() => void sign()}
                  className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white
                             hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-100
                             dark:text-slate-900">
            {busy ? 'Signing…' : 'Sign this declaration'}
          </button>

          {refusal && (
            <p data-testid="sign-refusal" role="status"
               className="flex items-start gap-1 rounded border border-amber-400 bg-amber-50 p-2
                          text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950/40
                          dark:text-amber-100">
              <Ban className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
              <span>{refusal}</span>
            </p>
          )}
          {signed && (
            <p data-testid="sign-done" role="status"
               className="rounded border border-emerald-400 bg-emerald-50 p-2 text-xs
                          text-emerald-900 dark:border-emerald-700 dark:bg-emerald-950/40
                          dark:text-emerald-100">
              Written: {signed}
            </p>
          )}
        </section>
      )}

      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Adopted ({index.declarations.filter((row) => row.adopted).length})
        </h3>
        <ul className="space-y-2">
          {index.declarations.filter((row) => row.adopted)
            .map((row) => <Row key={row.declaration} row={row} />)}
        </ul>
      </section>
    </div>
  );
}
