/**
 * `plotly.js-dist-min` ships a bundled minified build with no type declarations, and
 * `@types/plotly.js-dist-min` is not a dependency here. Only `downloadImage` is used (by
 * FigureExport), so it is declared precisely rather than the whole module being cast to
 * `any` — a blanket `declare module` would silently accept a typo in any Plotly call.
 */
declare module 'plotly.js-dist-min' {
  export function downloadImage(
    graphDiv: HTMLElement | string,
    options: {
      format: 'png' | 'svg' | 'jpeg' | 'webp';
      filename?: string;
      width?: number;
      height?: number;
      scale?: number;
    },
  ): Promise<string>;
}
