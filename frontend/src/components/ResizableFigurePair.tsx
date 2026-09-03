import React, { KeyboardEvent, PointerEvent, ReactNode, useEffect, useRef, useState } from 'react';

/**
 * A presentation-only split for two scientific figures (TG18.2).
 *
 * The split changes how much canvas each figure receives; it never hides a pane, changes the
 * comparison contract, or touches either figure's data. At narrow widths CSS returns the pair to
 * document order as a one-column stack and removes the inapplicable separator.
 */
export function ResizableFigurePair({
  label,
  children,
}: {
  label: string;
  children: [ReactNode, ReactNode];
}) {
  const container = useRef<HTMLDivElement>(null);
  const [firstPercent, setFirstPercent] = useState(50);
  const clamp = (value: number) => Math.max(25, Math.min(75, value));

  // Plotly listens for the window resize event. A pane resize is not a window resize, so publish
  // the same layout notification after React has committed the new grid tracks.
  useEffect(() => {
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    return () => cancelAnimationFrame(frame);
  }, [firstPercent]);

  const setFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = container.current?.getBoundingClientRect();
    if (!bounds || bounds.width <= 0) return;
    setFirstPercent(clamp(((event.clientX - bounds.left) / bounds.width) * 100));
  };

  const beginResize = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setFromPointer(event);
  };

  const moveResize = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) setFromPointer(event);
  };

  const keyResize = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 5;
    let next: number | null = null;
    if (event.key === 'ArrowLeft') next = firstPercent - step;
    if (event.key === 'ArrowRight') next = firstPercent + step;
    if (event.key === 'Home') next = 25;
    if (event.key === 'End') next = 75;
    if (event.key === 'Enter' || event.key === ' ') next = 50;
    if (next === null) return;
    event.preventDefault();
    setFirstPercent(clamp(next));
  };

  return (
    <div
      ref={container}
      className="resizable-figure-pair"
      style={{ '--first-pane': `${firstPercent}%` } as React.CSSProperties}
      data-first-pane-percent={Math.round(firstPercent)}
    >
      <div className="resizable-figure-pair__pane">{children[0]}</div>
      <div
        className="resizable-figure-pair__separator"
        role="separator"
        aria-label={`Resize ${label}`}
        aria-orientation="vertical"
        aria-valuemin={25}
        aria-valuemax={75}
        aria-valuenow={Math.round(firstPercent)}
        aria-valuetext={`First figure ${Math.round(firstPercent)} percent; second figure ${Math.round(100 - firstPercent)} percent`}
        tabIndex={0}
        onPointerDown={beginResize}
        onPointerMove={moveResize}
        onDoubleClick={() => setFirstPercent(50)}
        onKeyDown={keyResize}
        title="Drag to resize; arrow keys adjust; Enter resets"
      >
        <span aria-hidden="true" />
      </div>
      <div className="resizable-figure-pair__pane">{children[1]}</div>
      <p className="resizable-figure-pair__note">
        Pane width changes presentation only. Both figures remain present and their data,
        scales and comparison contract are unchanged. Drag the divider, use arrow keys, or press
        Enter to restore equal widths.
      </p>
    </div>
  );
}

export default ResizableFigurePair;
