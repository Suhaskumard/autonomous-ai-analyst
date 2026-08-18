import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import ChartFrame, { EmptyChart } from "./ChartFrame";
import {
  AXIS_PROPS,
  BAR_RADIUS,
  GRID_PROPS,
  INK,
  MAX_SERIES_ADJACENT,
  PRIMARY,
  TOOLTIP_STYLE,
  formatNumber,
  seriesColor,
} from "./theme";

/** Every model tried in one run, on the metric that selected the winner. */
export function ModelScores({ scores, problemType, selectedModel }) {
  const metric = problemType === "classification" ? "accuracy" : "rmse";
  const rows = Object.entries(scores || {})
    .filter(([, values]) => values && values[metric] !== undefined)
    .map(([model, values]) => ({ model, value: values[metric], selected: model === selectedModel }))
    .sort((a, b) => (metric === "accuracy" ? b.value - a.value : a.value - b.value));

  return (
    <ChartFrame
      title={`Model scores (${metric})`}
      caption={
        metric === "accuracy"
          ? "Higher is better. The selected model is outlined."
          : "Lower is better. The selected model is outlined."
      }
      tableHeaders={["Model", metric]}
      tableRows={rows.map((row) => [row.model, formatNumber(row.value, 4)])}
      height={Math.max(220, rows.length * 44 + 40)}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No model scores recorded." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }}>
            <CartesianGrid {...GRID_PROPS} vertical horizontal={false} />
            <XAxis
              type="number"
              {...AXIS_PROPS}
              domain={metric === "accuracy" ? [0, 1] : undefined}
              tickFormatter={(value) => (metric === "accuracy" ? `${Math.round(value * 100)}%` : formatNumber(value))}
            />
            <YAxis type="category" dataKey="model" {...AXIS_PROPS} width={140} />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(value) => [
                metric === "accuracy" ? `${(value * 100).toFixed(1)}%` : formatNumber(value, 4),
                metric,
              ]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={26} fill={PRIMARY} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

/**
 * Scores across several runs, one series per run.
 *
 * Capped at four series because that is what the palette validates on the
 * adjacent pairlist for grouped bars — the backend enforces the same ceiling.
 */
export function RunComparison({ comparison }) {
  if (!comparison) return null;

  if (comparison.comparable === false) {
    return (
      <ChartFrame title="Compare runs" caption="Scores side by side.">
        <EmptyChart message="These runs solve different problem types — an accuracy and an RMSE cannot share an axis. Compare runs of the same kind." />
      </ChartFrame>
    );
  }

  const runs = (comparison.runs || []).slice(0, MAX_SERIES_ADJACENT);
  const metric = comparison.metric || "accuracy";
  const models = Array.from(new Set(runs.flatMap((run) => Object.keys(run.scores || {}))));

  const rows = models.map((model) => {
    const row = { model };
    runs.forEach((run) => {
      const value = run.scores?.[model]?.[metric];
      if (value !== undefined) row[labelFor(run)] = value;
    });
    return row;
  });

  return (
    <ChartFrame
      title={`Compare runs (${metric})`}
      caption={`${runs.length} runs, per model. ${metric === "accuracy" ? "Higher" : "Lower"} is better.`}
      tableHeaders={["Model", ...runs.map(labelFor)]}
      tableRows={rows.map((row) => [row.model, ...runs.map((run) => formatNumber(row[labelFor(run)], 4))])}
      height={Math.max(260, rows.length * 46 + 60)}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No overlapping model scores across these runs." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }} barGap={2}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="model" {...AXIS_PROPS} interval={0} angle={-15} textAnchor="end" height={56} />
            <YAxis
              {...AXIS_PROPS}
              width={52}
              domain={metric === "accuracy" ? [0, 1] : undefined}
              tickFormatter={(value) => (metric === "accuracy" ? `${Math.round(value * 100)}%` : formatNumber(value))}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(value) => [metric === "accuracy" ? `${(value * 100).toFixed(1)}%` : formatNumber(value, 4)]}
            />
            {/* Two or more series: the legend is always present, so identity
                is never carried by colour alone. */}
            <Legend wrapperStyle={{ fontSize: "0.78rem", color: INK.secondary }} />
            {runs.map((run, index) => (
              <Bar
                key={run.run_key}
                dataKey={labelFor(run)}
                fill={seriesColor(index)}
                radius={BAR_RADIUS}
                maxBarSize={38}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

function labelFor(run) {
  return `${run.mode} · ${run.target ?? "?"} · ${run.run_key.slice(0, 6)}`;
}
