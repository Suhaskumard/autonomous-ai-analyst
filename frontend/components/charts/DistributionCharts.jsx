import React from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import ChartFrame, { EmptyChart } from "./ChartFrame";
import { AXIS_PROPS, BAR_RADIUS, BAR_RADIUS_HORIZONTAL, GRID_PROPS, INK, PRIMARY, TOOLTIP_STYLE, formatNumber } from "./theme";

/**
 * The backend has always computed these payloads; nothing rendered them.
 * One measure per chart, so each is a single-series bar: one colour, no
 * legend (the title names the series), values on hover.
 */

export function NumericHistogram({ column, data }) {
  const rows = (data?.labels || []).map((label, index) => ({
    label,
    count: data.counts?.[index] ?? 0,
    // The backend labels bins as pandas intervals; keep the full text for the
    // tooltip and a short form for the axis.
    short: shortenBin(label),
  }));

  return (
    <ChartFrame
      title={`Distribution of ${column}`}
      caption="Counts per value range, from the training snapshot."
      tableHeaders={["Range", "Count"]}
      tableRows={rows.map((row) => [row.label, row.count])}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No numeric distribution available." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="short" {...AXIS_PROPS} interval="preserveStartEnd" />
            <YAxis {...AXIS_PROPS} width={44} tickFormatter={(value) => formatNumber(value)} />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(value) => [formatNumber(value), "Rows"]}
              labelFormatter={(_label, payload) => payload?.[0]?.payload?.label ?? ""}
            />
            <Bar dataKey="count" fill={PRIMARY} radius={BAR_RADIUS} maxBarSize={48} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

export function CategoryBar({ column, data }) {
  const rows = (data?.labels || []).map((label, index) => ({
    label,
    count: data.counts?.[index] ?? 0,
  }));

  return (
    <ChartFrame
      title={`Top values in ${column}`}
      caption="Most frequent categories in the training snapshot."
      tableHeaders={["Value", "Count"]}
      tableRows={rows.map((row) => [row.label, row.count])}
      height={Math.max(200, rows.length * 30 + 40)}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No categorical values available." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 44, bottom: 4, left: 8 }}>
            <CartesianGrid {...GRID_PROPS} vertical horizontal={false} />
            <XAxis type="number" {...AXIS_PROPS} tickFormatter={(value) => formatNumber(value)} />
            <YAxis type="category" dataKey="label" {...AXIS_PROPS} width={110} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(value) => [formatNumber(value), "Rows"]} />
            <Bar dataKey="count" fill={PRIMARY} radius={BAR_RADIUS_HORIZONTAL} maxBarSize={22}>
              {/* Selective direct labels: one per bar is few enough to read. */}
              <LabelList
                dataKey="count"
                position="right"
                fill={INK.secondary}
                fontSize={11}
                formatter={(value) => formatNumber(value)}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

export function FeatureImportanceChart({ data, method }) {
  const rows = (data || []).map((item) => ({ feature: item.feature, importance: item.importance }));
  const methodLabel =
    method === "shap"
      ? "SHAP values"
      : method === "native"
        ? "model-native importances (SHAP unavailable)"
        : "unavailable";

  return (
    <ChartFrame
      title="Key feature influencers"
      caption={`From ${methodLabel}. One-hot columns are summed back into their source column.`}
      tableHeaders={["Feature", "Importance"]}
      tableRows={rows.map((row) => [row.feature, formatNumber(row.importance, 4)])}
      height={Math.max(200, rows.length * 32 + 40)}
    >
      {rows.length === 0 ? (
        <EmptyChart message="No feature attribution available for this model." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }}>
            <CartesianGrid {...GRID_PROPS} vertical horizontal={false} />
            <XAxis type="number" {...AXIS_PROPS} tickFormatter={(value) => formatNumber(value, 3)} />
            <YAxis type="category" dataKey="feature" {...AXIS_PROPS} width={120} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(value) => [formatNumber(value, 4), "Importance"]} />
            <Bar dataKey="importance" fill={PRIMARY} radius={BAR_RADIUS_HORIZONTAL} maxBarSize={22}>
              <LabelList
                dataKey="importance"
                position="right"
                fill={INK.secondary}
                fontSize={11}
                formatter={(value) => formatNumber(value, 3)}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

function shortenBin(label) {
  const match = String(label).match(/^\(?\[?(-?[\d.]+)/);
  if (!match) return String(label).slice(0, 10);
  return formatNumber(Number(match[1]), 1);
}
