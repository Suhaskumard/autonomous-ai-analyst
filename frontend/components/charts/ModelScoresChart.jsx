import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ChartFrame, { EmptyChart } from "./ChartFrame";
import {
  AXIS_PROPS,
  BAR_RADIUS,
  GRID_PROPS,
  INK,
  MAX_SERIES_ADJACENT,
  PRIMARY,
  STATUS,
  TOOLTIP_STYLE,
  formatNumber,
  seriesColor,
} from "./theme";

const METRIC_LABEL = {
  f1_macro: "macro F1",
  rmse: "RMSE",
  accuracy: "accuracy",
  balanced_accuracy: "balanced accuracy",
};

/**
 * Every model tried, as a cross-validated mean with its spread.
 *
 * A single bar per model was a single-split score dressed up as a fact. The
 * bar is now the mean over the folds and the whisker is +/- one standard
 * deviation, so a model that only looks best because of one lucky fold is
 * visibly indistinguishable from its neighbours. The dashed line is what a
 * constant predictor scored on the very same folds: bars that do not clear it
 * learned nothing.
 */
export function ModelScores({ cvScores, holdoutScores, baselineCv, selectionMetric, selectedModel }) {
  const metric = selectionMetric || "f1_macro";
  const higherIsBetter = metric !== "rmse";
  const label = METRIC_LABEL[metric] || metric;

  const rows = Object.entries(cvScores || {})
    .map(([model, metrics]) => {
      const entry = metrics?.[metric];
      if (!entry || typeof entry.mean !== "number") return null;
      return {
        model,
        mean: entry.mean,
        std: entry.std ?? 0,
        folds: entry.folds?.length ?? 0,
        holdout: holdoutScores?.[model]?.[metric],
        selected: model === selectedModel,
      };
    })
    .filter(Boolean)
    .sort((a, b) => (higherIsBetter ? b.mean - a.mean : a.mean - b.mean));

  const baseline = baselineCv?.[metric]?.mean;

  return (
    <ChartFrame
      title={`Cross-validated ${label}`}
      caption={`Mean +/- 1 sd over ${rows[0]?.folds || 0} folds. ${
        higherIsBetter ? "Higher" : "Lower"
      } is better; the dashed line is a constant predictor on the same folds.`}
      tableHeaders={["Model", `CV ${label}`, "Spread", "Holdout"]}
      tableRows={rows.map((row) => [
        row.selected ? `${row.model} (selected)` : row.model,
        formatNumber(row.mean, 4),
        `+/- ${formatNumber(row.std, 4)}`,
        row.holdout === undefined ? "—" : formatNumber(row.holdout, 4),
      ])}
      height={Math.max(240, rows.length * 46 + 60)}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No cross-validated scores recorded." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 64, bottom: 4, left: 8 }}>
            <CartesianGrid {...GRID_PROPS} vertical horizontal={false} />
            <XAxis
              type="number"
              {...AXIS_PROPS}
              domain={higherIsBetter ? [0, 1] : undefined}
              tickFormatter={(value) => (higherIsBetter ? formatNumber(value, 2) : formatNumber(value))}
            />
            <YAxis type="category" dataKey="model" {...AXIS_PROPS} width={150} />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(value, _name, item) => [
                `${formatNumber(value, 4)} +/- ${formatNumber(item?.payload?.std, 4)}`,
                `CV ${label}`,
              ]}
            />
            {typeof baseline === "number" && (
              <ReferenceLine
                x={baseline}
                stroke={STATUS.warning}
                strokeDasharray="4 4"
                label={{ value: "baseline", fill: INK.secondary, fontSize: 11, position: "top" }}
              />
            )}
            <Bar dataKey="mean" radius={[0, 4, 4, 0]} maxBarSize={26}>
              {rows.map((row) => (
                // The selected model is the saturated one; the rest recede.
                <Cell key={row.model} fill={PRIMARY} fillOpacity={row.selected ? 1 : 0.45} />
              ))}
              <ErrorBar dataKey="std" direction="x" width={4} strokeWidth={1.5} stroke={INK.secondary} />
            </Bar>
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
