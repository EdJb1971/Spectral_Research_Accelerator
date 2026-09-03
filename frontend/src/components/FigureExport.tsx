import React, { useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { FileText, Image, Loader2 } from 'lucide-react';
import type { FigureFact } from './FigureData';

/**
 * PNG / SVG download for a live Plotly figure (T3.5.23).
 *
 * Rendered client-side on purpose. A server-side re-render would be a *different* picture from
 * the one on screen — different colour scale limits, different aspect ratio, different tick
 * choices — and a figure that does not match what the researcher saw is worse than no figure.
 *
 * PNG and SVG are the picture alone. The publication sheet is the portable, self-contained HTML
 * form: it keeps the vector snapshot beside the supplied caption and the same reading contract
 * the page states. It derives no summary and has no route back into the claim ladder.
 */

interface FigureExportProps {
  /** The DOM id of the Plotly graph div to export. */
  targetId: string;
  /** Filename stem; format extension and a UTC stamp are appended. */
  name: string;
  /** Short provenance line burned into the image (units, seed, simulated flag). */
  caption?: string;
  publication?: {
    title: string;
    /** Deferred because a research-size field should be scanned only when export is requested. */
    facts: () => FigureFact[];
    /** Producer-authored claims, assumptions and refusal reasons carried verbatim. */
    notes?: () => string[];
    boundary: string;
  };
  onError?: (message: string) => void;
}

function stamped(name: string, format: string) {
  const safe = (name || 'figure').replace(/[^A-Za-z0-9_-]/g, '_');
  const iso = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, 'Z');
  return `${safe}_${iso}.${format}`;
}

function escapeHtml(value: unknown) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character] as string));
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const FigureExport: React.FC<FigureExportProps> = ({
  targetId, name, caption, publication, onError,
}) => {
  const [busy, setBusy] = useState(false);
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

  const downloadPublication = async () => {
    const element = document.getElementById(targetId);
    if (!element || !publication) {
      onError?.(`Figure "${targetId}" is not on screen, so there is nothing to export.`);
      return;
    }
    setBusy(true);
    try {
      const svg = await (Plotly as any).toImage(element as any, {
        format: 'svg', width: 1200, height: 800, scale: 1,
      });
      const facts = publication.facts();
      const rows = facts.map((fact) => (
        `<tr${fact.warn ? ' class="warning"' : ''}><th scope="row">${escapeHtml(fact.label)}</th>`
        + `<td>${escapeHtml(fact.value === null ? 'not established' : fact.value)}</td></tr>`
      )).join('');
      const notes = (publication.notes?.() || [])
        .map(note => `<li>${escapeHtml(note)}</li>`).join('');
      const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>${escapeHtml(publication.title)}</title><style>
body{max-width:1200px;margin:2rem auto;padding:0 1.5rem;color:#111827;font:16px/1.5 system-ui,sans-serif}
h1{font-size:1.5rem}.figure{width:100%;height:auto;border:1px solid #cbd5e1}.caption{font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:1rem}th,td{text-align:left;vertical-align:top;padding:.45rem;border-bottom:1px solid #e2e8f0}th{width:15rem}.warning{color:#92400e}
.boundary{margin-top:1rem;padding:.75rem;border-left:4px solid #0f766e;background:#f0fdfa}.meta{color:#475569;font-size:.875rem}
@media print{body{margin:0;max-width:none}.figure{break-inside:avoid}}
</style></head><body><main><h1>${escapeHtml(publication.title)}</h1>
<img class="figure" src="${escapeHtml(svg)}" alt="${escapeHtml(publication.title)}">
${caption ? `<p class="caption">${escapeHtml(caption)}</p>` : ''}
<h2>Figure reading contract</h2><table><tbody>${rows}</tbody></table>
${notes ? `<h2>Producer statements and qualifications</h2><ul>${notes}</ul>` : ''}
<p class="boundary">${escapeHtml(publication.boundary)}</p>
<p class="meta">Exported ${escapeHtml(new Date().toISOString())}. This sheet reproduces the live figure and transcribes its stated contract; it performs no scientific analysis.</p>
</main></body></html>`;
      triggerDownload(new Blob([html], { type: 'text/html;charset=utf-8' }), stamped(name, 'html'));
    } catch (e: any) {
      onError?.(`Publication export failed: ${e?.message || e}`);
    } finally {
      setBusy(false);
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
      {publication && (
        <button
          onClick={downloadPublication}
          disabled={busy}
          title="Self-contained vector figure, caption and reading contract — for review and publication"
          className="text-[10px] font-mono px-2 py-1 rounded border border-slate-800 bg-slate-950 text-slate-400 hover:text-teal-400 hover:border-teal-500/30 disabled:opacity-40 transition uppercase flex items-center gap-1"
        >
          {busy ? <Loader2 className="w-3 h-3 animate-spin" /> : <FileText className="w-3 h-3" />}
          Publication HTML
        </button>
      )}
      {caption && (
        <span className="text-[10px] text-slate-600 truncate max-w-xs" title={caption}>
          {caption}
        </span>
      )}
    </div>
  );
};
