import { useMemo, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Heatmap2D } from './Heatmap2D';

interface NativeLevel {
  level: number;
  native_shape: number[];
  valid_interior_halfwidth_parent_px: number;
  valid_interior_halfwidth_native_px: number;
  valid_interior_native_shape: number[];
  magnitude_by_orientation: number[][][];
}

interface DTCWTSummary {
  levels: number;
  feature_orientations_deg: number[];
  wavevector_orientations_deg: number[];
  measured_level1_wavevector_deg: number[];
  measured_qshift_wavevector_deg: number[];
  native_magnitude_maps: NativeLevel[];
  visualization_contract: Record<string, string>;
}

export function DTCWTScientificView({ summary }: { summary: DTCWTSummary | null }) {
  const [selectedLevel, setSelectedLevel] = useState(1);
  const level = summary?.native_magnitude_maps.find(item => item.level === selectedLevel)
    ?? summary?.native_magnitude_maps[0];
  const sharedMaximum = useMemo(() => {
    if (!level) return 1;
    return level.magnitude_by_orientation.flat(2).reduce(
      (maximum, value) => Math.max(maximum, value), 1e-30
    );
  }, [level]);

  if (!summary || !level) return null;
  const measured = level.level === 1
    ? summary.measured_level1_wavevector_deg
    : summary.measured_qshift_wavevector_deg;
  const valid = level.valid_interior_native_shape.every(value => value > 0);

  return (
    <section data-testid="dtcwt-scientific-view" className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">DTCWT directional evidence — native complex magnitude</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Six orientation responses are shown on this level's own decimated grid with one shared colour scale.
            The dashed inset is the only region supported without boundary extension.
          </p>
        </div>
        <label className="text-xs text-slate-400">
          Scale level
          <select value={level.level} onChange={event => setSelectedLevel(Number(event.target.value))}
            className="ml-2 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
            {summary.native_magnitude_maps.map(item => <option key={item.level} value={item.level}>L{item.level}</option>)}
          </select>
        </label>
      </div>

      <div className={`rounded-lg border p-3 text-xs ${valid ? 'border-amber-900 bg-amber-950/20 text-amber-200' : 'border-rose-800 bg-rose-950/40 text-rose-200'}`}>
        <div className="flex gap-2 items-start">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <p>
            L{level.level}: native {level.native_shape.join('×')}; edge inset {level.valid_interior_halfwidth_native_px}
            {' '}native samples ({level.valid_interior_halfwidth_parent_px} parent-grid pixels per side); valid interior{' '}
            <strong>{level.valid_interior_native_shape.join('×')}</strong>.
            {!valid && ' No uncontaminated interior remains: do not interpret these maps as regional feature evidence.'}
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[11px] text-left">
          <thead className="text-slate-500 border-b border-slate-800"><tr>
            <th className="py-2">band</th><th>feature axis</th><th>nominal wavevector</th><th>measured centre</th><th>display</th>
          </tr></thead>
          <tbody>{summary.feature_orientations_deg.map((feature, index) => <tr key={index} className="border-b border-slate-800/60">
            <td className="py-2 font-mono text-slate-300">{index}</td>
            <td className="text-teal-300">{feature.toFixed(1)}°</td>
            <td className="text-slate-300">{summary.wavevector_orientations_deg[index].toFixed(1)}°</td>
            <td className="text-slate-300">{measured[index].toFixed(1)}°</td>
            <td className="text-slate-400">|complex coefficient|</td>
          </tr>)}</tbody>
        </table>
      </div>

      {level.level === 1 && <p className="text-xs text-amber-300">
        Level 1 is less directionally precise (up to about 7.1° from nominal); use levels 2+ for orientation estimates when a valid interior exists.
      </p>}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {level.magnitude_by_orientation.map((map, index) => (
          <Heatmap2D key={index} data={map}
            title={`Band ${index} · feature ${summary.feature_orientations_deg[index]}°`}
            units="|coefficient|" xLabel="native column index" yLabel="native row index"
            zRange={[0, sharedMaximum]} validInset={level.valid_interior_halfwidth_native_px} />
        ))}
      </div>

      <p className="text-[11px] text-slate-500">
        No cross-level interpolation is used. Panel adjacency is layout only, not physical adjacency. Magnitude supports feature-strength comparison; phase is retained for reconstruction but is not displayed as if it were a scalar physical field.
      </p>
    </section>
  );
}
