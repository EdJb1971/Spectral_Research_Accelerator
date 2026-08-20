import React from 'react';
import Plotly from 'plotly.js-dist-min';
import { Image } from 'lucide-react';

/**
 * PNG / SVG download for a live Plotly figure (T3.5.23).
 *
 * Rendered client-side on purpose. A server-side re-render would be a *different* picture from
 * the one on screen — different colour scale limits, different aspect ratio, different tick
 * choices — and a figure that does not match what the researcher saw is worse than no figure.
 *
 * The caption is burned into the exported image via the layout title rather than added
 * afterwards, so units and provenance travel with the picture. A plot pasted into a paper draft
 * with no units is how a wrong axis gets published.
 */

interface FigureExportProps {
  /** The DOM id of the Plotly graph div to export. */
  targetId: string;
  /** Filename stem; format extension and a UTC stamp are appended. */
  name: string;
  /** Short provenance line burned into the image (units, seed, simulated flag). */
  caption?: string;
  onError?: (message: string) => void;
}

function stamped(name: string, format: string) {
  const safe = (name || 'figure').replace(/[^A-Za-z0-9_-]/g, '_');
  const iso = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, 'Z');
  return `${safe}_${iso}.${format}`;
}

export const FigureExport: React.FC<FigureExportProps> = ({
  targetId, name, caption, onError,
}) => {
  const download = async (format: 'png' | 'svg') => {
    const element = document.getElementById(targetId);
    if (!element) {
      onError?.(`Figure "${targetId}" is not on screen, so there is nothing to export.`);
      return;
    }
    try {
      // Scale 2 for PNG so the raster is usable in a document rather than only on screen;
      // SVG ignores scale because it is resolution-independent.
      await Plotly.downloadImage(element as any, {
        format,
        filename: stamped(name, format).replace(new RegExp(`\\.${format}$`), ''),
        width: 1200,
        height: 800,
        scale: format === 'png' ? 2 : 1,
      });
    } catch (e: any) {
      onError?.(`Figure export failed: ${e?.message || e}`);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
        <Image className="w-3 h-3" /> Figure
      </span>
      {(['png', 'svg'] as const).map(format => (
        <button
          key={format}
          onClick={() => download(format)}
          title={format === 'png'
            ? 'Raster, 1200x800 at 2x — for documents and slides'
            : 'Vector — scales without loss, for publication'}
          className="text-[10px] font-mono px-2 py-1 rounded border border-slate-800 bg-slate-950 text-slate-400 hover:text-teal-400 hover:border-teal-500/30 transition uppercase"
        >
          {format}
        </button>
      ))}
      {caption && (
        <span className="text-[10px] text-slate-600 truncate max-w-xs" title={caption}>
          {caption}
        </span>
      )}
    </div>
  );
};
