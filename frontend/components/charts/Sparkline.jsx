import React from "react";

import { INK, PRIMARY } from "./theme";

/**
 * A compact inline distribution for the per-column profile table.
 *
 * Plain SVG rather than recharts: this renders once per column, on the very
 * first screen after upload, before a run exists — the one place recharts
 * cannot be lazy-loaded away, since `Dashboard` (where every other chart
 * lives) has not mounted yet. A hand-drawn bar strip needs none of what
 * recharts provides here (axes, tooltips, responsive containers) and keeps
 * recharts out of the initial bundle entirely.
 */
export function Sparkline({ sparkline }) {
  const counts = sparkline?.counts || [];
  if (counts.length === 0) return <span style={{ color: INK.muted, fontSize: "0.75rem" }}>—</span>;
  const max = Math.max(...counts) || 1;
  const width = 110;
  const height = 28;
  const gap = 1;
  const barWidth = counts.length > 0 ? (width - gap * (counts.length - 1)) / counts.length : width;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {counts.map((count, index) => {
        const barHeight = Math.max(1, (count / max) * height);
        const x = index * (barWidth + gap);
        const y = height - barHeight;
        const opacity = 0.35 + 0.65 * (count / max);
        return <rect key={index} x={x} y={y} width={barWidth} height={barHeight} fill={PRIMARY} fillOpacity={opacity} rx={1} />;
      })}
    </svg>
  );
}
