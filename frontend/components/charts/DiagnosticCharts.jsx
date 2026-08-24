import React from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import ChartFrame, { EmptyChart } from "./ChartFrame";
import {
  AXIS_PROPS,
  DIVERGING,
  GRID_PROPS,
  INK,
  PRIMARY,
  SEQUENTIAL,
  TOOLTIP_STYLE,
  divergingColor,
  formatNumber,
  sequentialColor,
} from "./theme";

/**
 * Where the model is wrong — the thing a single accuracy number cannot say.
 * Both matrices are hand-built SVG grids: a heatmap is a cell chart, not a
 * bar chart, and Recharts has no such form.
 */

export function ConfusionMatrix({ data }) {
  if (!data || !data.labels?.length) {
    return (
      <ChartFrame title="Confusion matrix" caption="Where predictions land against the truth.">
        <EmptyChart message="No confusion matrix for this run." />
      </ChartFrame>
    );
  }

  const { labels, matrix, truncated } = data;
  const max = Math.max(1, ...matrix.flat());
  const size = labels.length;
  const cell = size > 8 ? 34 : size > 5 ? 46 : 58;
  const gutter = 2; // the 2px surface gap between adjacent fills

  return (
    <ChartFrame
      title="Confusion matrix"
      caption={
        `Rows are the true class, columns the predicted one; the diagonal is correct. ` +
        `${data.total.toLocaleString()} held-out rows.` +
        (truncated ? " Showing the most common classes only." : "")
      }
      tableHeaders={["True \\ Predicted", ...labels]}
      tableRows={matrix.map((row, index) => [labels[index], ...row])}
      height={size * (cell + gutter) + 90}
    >
      <div className="matrix-scroll">
        <table className="matrix-grid">
          <thead>
            <tr>
              <th className="matrix-corner" scope="col">
                <span className="matrix-axis-label">true ↓ / predicted →</span>
              </th>
              {labels.map((label) => (
                <th key={label} scope="col" className="matrix-head" style={{ width: cell }}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex]}>
                <th scope="row" className="matrix-head matrix-row-head">
                  {labels[rowIndex]}
                </th>
                {row.map((value, columnIndex) => {
                  const fraction = value / max;
                  const correct = rowIndex === columnIndex;
                  return (
                    <td key={columnIndex} style={{ padding: gutter / 2 }}>
                      <div
                        className="matrix-cell"
                        style={{
                          background: value === 0 ? "rgba(255,255,255,0.03)" : sequentialColor(fraction),
                          // A 2px surface ring marks the diagonal without
                          // spending a second colour on it.
                          outline: correct ? `2px solid ${INK.primary}` : "none",
                          outlineOffset: "-2px",
                          height: cell,
                          color: fraction > 0.45 ? "#f8fafc" : INK.secondary,
                        }}
                        title={`True ${labels[rowIndex]}, predicted ${labels[columnIndex]}: ${value}`}
                      >
                        {value}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <Legend
          from={SEQUENTIAL[0]}
          to={SEQUENTIAL[SEQUENTIAL.length - 1]}
          left="0 rows"
          right={`${max} rows`}
        />
      </div>
    </ChartFrame>
  );
}

export function CorrelationHeatmap({ data }) {
  if (!data || !data.columns?.length) {
    return (
      <ChartFrame title="Correlation" caption="How numeric columns move together.">
        <EmptyChart message="Needs at least two varying numeric columns." />
      </ChartFrame>
    );
  }

  const { columns, matrix, truncated, strongest_pairs: strongest } = data;
  const cell = columns.length > 8 ? 32 : columns.length > 5 ? 40 : 52;
  const headline = strongest?.[0];

  return (
    <ChartFrame
      title="Correlation between numeric columns"
      caption={
        (headline
          ? `Strongest pair: ${headline.a} and ${headline.b} (r = ${formatNumber(headline.r, 2)}). `
          : "") + (truncated ? "Showing the most variable columns." : "Pearson r, −1 to +1.")
      }
      tableHeaders={["", ...columns]}
      tableRows={matrix.map((row, index) => [columns[index], ...row.map((value) => formatNumber(value, 2))])}
      height={columns.length * (cell + 2) + 100}
    >
      <div className="matrix-scroll">
        <table className="matrix-grid">
          <thead>
            <tr>
              <th className="matrix-corner" />
              {columns.map((column) => (
                <th key={column} scope="col" className="matrix-head" style={{ width: cell }}>
                  <span className="matrix-head-rotated" title={column}>
                    {column}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={columns[rowIndex]}>
                <th scope="row" className="matrix-head matrix-row-head">
                  {columns[rowIndex]}
                </th>
                {row.map((value, columnIndex) => (
                  <td key={columnIndex} style={{ padding: 1 }}>
                    <div
                      className="matrix-cell"
                      style={{
                        background: divergingColor(value),
                        height: cell,
                        color: Math.abs(value) > 0.55 ? "#f8fafc" : INK.secondary,
                        fontSize: cell < 40 ? "0.65rem" : "0.75rem",
                      }}
                      title={`${columns[rowIndex]} vs ${columns[columnIndex]}: r = ${formatNumber(value, 3)}`}
                    >
                      {formatNumber(value, 2)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <Legend from={DIVERGING.negative} via={DIVERGING.neutral} to={DIVERGING.positive} left="−1" right="+1" />
      </div>
    </ChartFrame>
  );
}

export function ResidualPlot({ data }) {
  if (!data || !data.points?.length) {
    return (
      <ChartFrame title="Residuals" caption="Prediction error against predicted value.">
        <EmptyChart message="No residuals for this run." />
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title="Residuals vs predicted"
      caption={
        `Each point is one held-out row; the line at zero is a perfect prediction. ` +
        `Mean error ${formatNumber(data.mean_residual, 3)}.` +
        (data.sampled ? ` Sampled ${data.points.length} of ${data.total.toLocaleString()} rows.` : "")
      }
      tableHeaders={["Predicted", "Actual", "Residual"]}
      tableRows={data.points
        .slice(0, 100)
        .map((point) => [
          formatNumber(point.predicted, 2),
          formatNumber(point.actual, 2),
          formatNumber(point.residual, 2),
        ])}
      height={300}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid {...GRID_PROPS} vertical />
          <XAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            {...AXIS_PROPS}
            tickFormatter={(value) => formatNumber(value)}
            label={{ value: "Predicted", position: "insideBottom", offset: -8, fill: INK.muted, fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="residual"
            name="Residual"
            {...AXIS_PROPS}
            width={56}
            tickFormatter={(value) => formatNumber(value)}
            label={{ value: "Residual", angle: -90, position: "insideLeft", fill: INK.muted, fontSize: 11 }}
          />
          <ZAxis range={[36, 36]} />
          <ReferenceLine y={0} stroke={INK.secondary} strokeWidth={2} />
          <Tooltip
            {...TOOLTIP_STYLE}
            cursor={{ strokeDasharray: "3 3", stroke: INK.axis }}
            formatter={(value, name) => [formatNumber(value, 3), name]}
          />
          <Scatter data={data.points} fill={PRIMARY} fillOpacity={0.65} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

function Legend({ from, via, to, left, right }) {
  const gradient = via ? `${from}, ${via}, ${to}` : `${from}, ${to}`;
  return (
    <div className="matrix-legend">
      <span>{left}</span>
      <span className="matrix-legend-ramp" style={{ background: `linear-gradient(to right, ${gradient})` }} />
      <span>{right}</span>
    </div>
  );
}
