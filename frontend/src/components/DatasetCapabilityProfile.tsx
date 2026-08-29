import { Ban, CheckCircle2, HelpCircle } from 'lucide-react';

import * as types from '../types/api';

interface Props {
  profile: types.DatasetCapabilityProfile;
  compact?: boolean;
}

const DatasetCapabilityProfile: React.FC<Props> = ({ profile, compact = false }) => (
  <section className="border border-slate-700 bg-slate-950/60 rounded-xl p-4 space-y-3"
    aria-label="Admissible scientific paths">
    <header className="flex flex-wrap justify-between gap-2">
      <div>
        <h3 className="text-sm font-semibold text-slate-100">Admissible scientific paths</h3>
        <p className="text-[10px] text-slate-500">{profile.kind} · {profile.phase}
          {profile.domain ? ` · ${profile.domain}` : ''}</p>
      </div>
      <span className="font-mono text-[9px] text-slate-600">{profile.profile_sha256.slice(0, 12)}…</span>
    </header>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
      {profile.capabilities.map((capability) => <div key={capability.name}
        className="rounded border border-slate-800 px-2 py-1.5 text-[10px] flex gap-1.5 items-center">
        {capability.value === true ? <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
          : capability.value === false ? <Ban className="w-3 h-3 text-slate-600 shrink-0" />
            : <HelpCircle className="w-3 h-3 text-amber-400 shrink-0" />}
        <span className={capability.value === true ? 'text-slate-200' : 'text-slate-500'}>
          {capability.label}: {capability.value === null ? 'not established' : capability.value ? 'yes' : 'no'}
        </span>
      </div>)}
    </div>
    {!compact && <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
      {Object.entries(profile.operations).map(([key, decision]) => <div key={key}
        className={`rounded border p-3 text-xs ${decision.available
          ? 'border-emerald-800/60 bg-emerald-950/10' : 'border-slate-800 bg-slate-900/30'}`}>
        <div className="flex gap-2 items-center">
          {decision.available ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            : <Ban className="w-4 h-4 text-amber-400 shrink-0" />}
          <strong className={decision.available ? 'text-emerald-200' : 'text-slate-300'}>{decision.name}</strong>
          <span className="ml-auto uppercase text-[9px] text-slate-600">{decision.status.replace('_', ' ')}</span>
        </div>
        <p className="text-[11px] text-slate-500 mt-1 ml-6">{decision.reason}</p>
      </div>)}
    </div>}
    <p className="text-[9px] text-slate-600">{profile.claim_boundary}</p>
  </section>
);

export default DatasetCapabilityProfile;
