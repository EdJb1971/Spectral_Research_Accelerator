import React, { useState } from 'react';
import { Download, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';

/**
 * Export controls for a field or a table (T3.5.23).
 *
 * Every button posts to the backend, which embeds the provenance **inside** the file: the
 * source, the seed, whether the data was simulated, the units and the grid. A CSV sitting in a
 * downloads folder with none of that is indistinguishable from any other CSV six months later,
 * which is exactly when it matters.
 *
 * NetCDF and Zarr are server-side because a browser cannot write either format; a hand-rolled
 * approximation would open in some tools and not others, which is worse than not offering it.
 * PNG and SVG are handled by the plot components instead, since only they know what is
 * currently on screen.
 */

const FIELD_FORMATS = [
  { id: 'csv', label: 'CSV', hint: 'Opens anywhere. Provenance in commented header lines.' },
  { id: 'json', label: 'JSON', hint: 'Nested provenance preserved exactly.' },
  { id: 'netcdf', label: 'NetCDF', hint: 'NetCDF4/HDF5 with coords, units and attrs — straight into xarray.' },
  { id: 'zarr', label: 'Zarr', hint: 'Chunked store, delivered as a zip.' },
] as const;

const TABLE_FORMATS = [
  { id: 'csv', label: 'CSV', hint: 'Provenance in commented header lines.' },
  { id: 'json', label: 'JSON', hint: 'Nested values preserved.' },
] as const;

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Revoked on the next tick rather than immediately: revoking synchronously can cancel the
  // download in some browsers before it has read the blob.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

interface FieldExportProps {
  field: number[][] | null;
  coords?: Record<string, number[]>;
  metadata?: Record<string, any>;
  variable?: string;
  units?: string | null;
  name: string;
  label?: string;
  onError?: (message: string) => void;
}

export const FieldExportBar: React.FC<FieldExportProps> = ({
  field, coords, metadata, variable = 'field', units, name, label = 'Export field', onError,
}) => {
  const [busy, setBusy] = useState<string | null>(null);
  const disabled = !field || field.length === 0;

  const run = async (format: string) => {
    if (!field) return;
    setBusy(format);
    try {
      const { blob, filename } = await apiService.exportField({
        field_data: field,
        format,
        coords: coords || {},
        metadata: metadata || {},
        variable,
        units: units ?? null,
        name,
      });
      triggerDownload(blob, filename);
    } catch (e: any) {
      onError?.(`Export failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
        <Download className="w-3 h-3" /> {label}
      </span>
      {FIELD_FORMATS.map(f => (
        <button
          key={f.id}
          onClick={() => run(f.id)}
          disabled={disabled || busy !== null}
          title={disabled ? 'Nothing to export yet' : f.hint}
          className="text-[10px] font-mono px-2 py-1 rounded border border-slate-800 bg-slate-950 text-slate-400 hover:text-teal-400 hover:border-teal-500/30 disabled:opacity-40 disabled:hover:text-slate-400 disabled:hover:border-slate-800 transition flex items-center gap-1"
        >
          {busy === f.id && <Loader2 className="w-2.5 h-2.5 animate-spin" />}
          {f.label}
        </button>
      ))}
    </div>
  );
};

interface TableExportProps {
  rows: Record<string, any>[];
  columns?: string[];
  metadata?: Record<string, any>;
  name: string;
  label?: string;
  onError?: (message: string) => void;
}

export const TableExportBar: React.FC<TableExportProps> = ({
  rows, columns, metadata, name, label = 'Export table', onError,
}) => {
  const [busy, setBusy] = useState<string | null>(null);

  const run = async (format: string) => {
    setBusy(format);
    try {
      const { blob, filename } = await apiService.exportTable({
        rows, format, columns: columns ?? null, metadata: metadata || {}, name,
      });
      triggerDownload(blob, filename);
    } catch (e: any) {
      onError?.(`Export failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
        <Download className="w-3 h-3" /> {label}
      </span>
      {TABLE_FORMATS.map(f => (
        <button
          key={f.id}
          onClick={() => run(f.id)}
          disabled={busy !== null}
          title={f.hint}
          className="text-[10px] font-mono px-2 py-1 rounded border border-slate-800 bg-slate-950 text-slate-400 hover:text-teal-400 hover:border-teal-500/30 disabled:opacity-40 transition flex items-center gap-1"
        >
          {busy === f.id && <Loader2 className="w-2.5 h-2.5 animate-spin" />}
          {f.label}
        </button>
      ))}
      {/* An empty table is still worth exporting: "nothing survived correction" is a real
          result, and refusing to save it would make the honest answer the unsaveable one. */}
      {rows.length === 0 && (
        <span className="text-[10px] text-slate-600">(0 rows — still exportable)</span>
      )}
    </div>
  );
};
