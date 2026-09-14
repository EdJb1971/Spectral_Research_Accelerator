import { useCallback, useEffect, useState } from 'react';

/**
 * Who is signing, remembered between adoptions. What they are signing, never.
 *
 * A single-maintainer instrument asks the same person for the same two strings every time they
 * adopt anything, which is friction with no scientific content: a name and a role are
 * attribution, and retyping them proves nothing. The affirmation is different in kind. It is
 * the act itself -- the sentence that says this person read this declaration -- so it is
 * deliberately absent from everything below. A remembered affirmation would mean the browser
 * had affirmed, once, on behalf of every future adoption, which is exactly the substitution the
 * adoption machinery exists to prevent.
 *
 * The reason a declaration is adopted is not remembered either, for a narrower reason: a reason
 * belongs to one decision, and a cached one would silently attach last week's rationale to a
 * document it was never written about.
 *
 * Storage is per-browser and best-effort. It can be absent (a private window, cleared data) and
 * on some configurations reading it throws, so every access is guarded and the forms render
 * correctly with nothing stored.
 */
const STORAGE_KEY = 'spectralearth.signer.identity';

export interface SignerIdentity {
  name: string;
  role: string;
}

const EMPTY: SignerIdentity = { name: '', role: '' };

function read(): SignerIdentity {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw) as Partial<SignerIdentity>;
    return {
      name: typeof parsed.name === 'string' ? parsed.name : '',
      role: typeof parsed.role === 'string' ? parsed.role : '',
    };
  } catch {
    return EMPTY;
  }
}

/**
 * Returns the remembered signer, setters for the two fields, and whether to remember them.
 *
 * `remember` defaults to true only when something was already stored: a first-time signer is
 * not opted in on their behalf.
 */
export function useSignerIdentity() {
  const stored = read();
  const [name, setName] = useState(stored.name);
  const [role, setRole] = useState(stored.role);
  const [remember, setRemember] = useState(Boolean(stored.name || stored.role));

  useEffect(() => {
    if (remember) return;
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* storage unavailable */ }
  }, [remember]);

  /** Call after an adoption succeeds, so a refused attempt never stores anything. */
  const persist = useCallback(() => {
    if (!remember) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ name, role }));
    } catch { /* storage unavailable; the form still worked */ }
  }, [remember, name, role]);

  const forget = useCallback(() => {
    setName('');
    setRole('');
    setRemember(false);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* storage unavailable */ }
  }, []);

  return { name, setName, role, setRole, remember, setRemember, persist, forget };
}

/**
 * The checkbox every adoption form shows, with the exclusion stated rather than implied.
 * Rendered as a plain element so each panel keeps its own surrounding layout.
 */
export function rememberSignerLabel(): string {
  return 'Remember my name and role on this browser. The affirmation is never saved.';
}
