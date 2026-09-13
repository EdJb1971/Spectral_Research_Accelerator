import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2, Play, ShieldCheck } from 'lucide-react';
import { apiService } from '../services/api';
import * as types from '../types/api';

function statusColour(status: string) {
  if (status === 'PASS') return 'text-emerald-300 border-emerald-500/30 bg-emerald-500/5';
  if (status === 'FAIL') return 'text-rose-300 border-rose-500/30 bg-rose-500/5';
  return 'text-amber-200 border-amber-500/30 bg-amber-500/5';
}

/** TG17.10's release decision is rendered from the server's complete gate registry.
 *
 * A green deterministic matrix cannot hide an unrun live or scientific-calibration gate: the
 * verdict is server-computed and every blocking row stays visible beside the rehearsal rows.
 *
 * A refused cell keeps its reason on screen rather than showing a bare status word. The
 * scale/shape quartet is currently refused by the order-book adapter's own declaration, and a
 * reader who cannot see why would be left to assume the apparatus is broken.
 */
export function ExperimentQualificationPanel() {
  const [record, setRecord] = useState<types.ExperimentQualificationRecord | null>(null);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('Loading the qualification gate…');

  useEffect(() => {
    apiService.experimentQualificationPlan()
      .then((body) => { setRecord(body); setMessage(''); })
      .catch((error: Error) => setMessage(error.message));
  }, []);

  const rehearse = async () => {
    setRunning(true);
    setMessage('Preflighting six fixture-only manifests, executing the admissible ones and one '
      + 'restart recovery…');
    try {
      const body = await apiService.rehearseExperimentQualification();
      setRecord(body);
      setMessage(body.verdict === 'RELEASEABLE'
        ? 'Every registered gate passed.'
        : 'Offline apparatus checks finished. Unrun scientific and live gates still block release.');
    } catch (error: any) { setMessage(error.message); }
    finally { setRunning(false); }
  };

  if (!record) return <p role="status" className="text-xs text-slate-500">{message}</p>;
  const passed = record.matrix.filter((row) => row.status === 'PASS').length;
  const refused = record.matrix.filter((row) => row.status === 'REFUSED').length;

  return (
    <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4"
             aria-labelledby="qualification-title">
      <div className="flex flex-wrap justify-between gap-3 items-start">
        <div>
          <h3 id="qualification-title" className="text-sm font-semibold text-slate-200 flex gap-2 items-center">
            <ShieldCheck className="w-4 h-4 text-violet-400" aria-hidden="true" />
            Experiment workflow qualification
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">
            Three frozen durations × two scientifically separate modes. Fixture rehearsal and
            live-source acceptance are different gates, and a cell a domain's declaration refuses
            is neither a pass nor a broken apparatus.
          </p>
        </div>
        <div className={`border rounded px-3 py-2 text-xs font-semibold ${statusColour(
          record.verdict === 'RELEASEABLE' ? 'PASS' : 'NOT_RUN')}`}>
          {record.verdict}
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center">
        <button onClick={rehearse} disabled={running}
                className="px-3 py-2 rounded bg-violet-700 hover:bg-violet-600 disabled:opacity-50 text-xs
                           flex gap-2 items-center focus:outline-none focus:ring-2 focus:ring-teal-400">
          {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                   : <Play className="w-3.5 h-3.5" aria-hidden="true" />}
          Run offline qualification
        </button>
        <span role="status" className="text-xs text-slate-400">{message
          || `${passed}/${record.matrix.length} cells passed${refused ? `, ${refused} refused` : ''}`}</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[11px]" aria-label="Duration and comparison mode qualification matrix">
          <thead><tr className="text-left text-slate-500 border-b border-slate-800">
            <th className="py-2">Duration</th><th>Mode</th><th>Exact UTC interval</th>
            <th>Correction</th><th>Status</th>
          </tr></thead>
          <tbody>{record.matrix.map((row) => (
            <tr key={row.cell_id} className="border-b border-slate-800/60">
              <td className="py-2 text-slate-300">{row.duration.replace('_', ' ')}</td>
              <td className="font-mono text-slate-400">{row.mode}</td>
              <td className="font-mono text-slate-500">{row.start_utc} → {row.end_utc}</td>
              <td className="font-mono text-slate-500">{row.family_correction}</td>
              <td>
                <span className={row.status === 'PASS' ? 'text-emerald-300' : 'text-amber-300'}>
                  {row.status}</span>
                {row.refusals?.length ? <span className="block text-[10px] text-amber-200/80 mt-1">
                  {row.refusals.map((item) => item.reason).join(' ')}</span> : null}
              </td>
            </tr>
          ))}</tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
        {record.gates.map((gate) => (
          <div key={gate.gate_id} className={`border rounded-lg p-3 ${statusColour(gate.status)}`}>
            <div className="flex gap-2 items-center text-xs font-semibold">
              {gate.status === 'PASS'
                ? <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
                : <AlertTriangle className="w-3.5 h-3.5" aria-hidden="true" />}
              {gate.title}: {gate.status}
            </div>
            <p className="text-[10px] opacity-80 mt-1">{gate.detail}</p>
          </div>
        ))}
      </div>

      {record.recovery && <p className="text-[11px] text-slate-400">
        Recovery rehearsal: <span className="text-slate-200">{record.recovery.failed_component}</span>
        {' '}failed once; the restarted run retried it while every completed acquisition stayed at one attempt.
      </p>}
      <p className="text-[10px] text-slate-600 break-all font-mono">
        {record.record_kind} · {record.qualification_sha256}
      </p>
      <p className="text-[11px] text-slate-500">{record.claim_boundary}</p>
    </section>
  );
}

export default ExperimentQualificationPanel;
