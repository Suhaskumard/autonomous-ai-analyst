import React from "react";
import { Scissors, Table2 } from "lucide-react";

import { formatNumber } from "./charts/theme";

/**
 * The fields the pipeline has always computed and stored, and that nothing
 * displayed: summary_stats, dropped_columns, class_names, the trained shape,
 * and the counted half of the quality report.
 *
 * Numeric and categorical columns get separate tables rather than one wide
 * table half-full of dashes — a mean has no meaning for a category, and a
 * "most common value" has none for a float.
 */
export default function DatasetSummary({ result }) {
  const entries = Object.entries(result?.summary_stats || {});
  if (entries.length === 0) return null;

  const rows = result?.row_count ?? 0;
  const quality = result?.quality_report || {};
  const dropped = result?.dropped_columns || [];
  const classNames = result?.class_names || [];

  const numeric = entries.filter(([, stats]) => stats?.type === "numeric");
  const categorical = entries.filter(([, stats]) => stats?.type !== "numeric");
  const missing = (count) => describeMissing(count, rows);

  return (
    <div className="card fade-in">
      <h2>
        <Table2 size={20} style={{ color: "var(--primary)" }} />
        What was trained on
      </h2>

      <dl className="facts-grid">
        <Fact label="Trained shape">
          <span className="tabular">
            {rows.toLocaleString()} × {result?.column_count ?? 0}
          </span>
          {" rows × columns"}
        </Fact>
        <Fact label="Target">
          <strong>{result?.target || "—"}</strong>
          <span className="fact-note">
            {result?.target_auto_detected === undefined
              ? ""
              : result.target_auto_detected
                ? " auto-detected"
                : " chosen by you"}
          </span>
        </Fact>
        <Fact label="Features used">{(result?.features || []).length}</Fact>
        <Fact label={result?.problem_type === "classification" ? "Classes" : "Predicted value"}>
          {classNames.length > 0 ? classNames.join(", ") : "continuous"}
        </Fact>
        <Fact label="Duplicate rows removed">{(quality.duplicate_rows ?? 0).toLocaleString()}</Fact>
        <Fact label="Training mode">{result?.mode || "—"}</Fact>
        <Fact label="Trained at">{formatTimestamp(result?.timestamp)}</Fact>
        <Fact label="Dataset fingerprint">
          {/* Artifacts are content-addressed: the same digest means the same
              file, which is why re-uploading it replays instead of retraining. */}
          <code className="fingerprint" title={result?.dataset_hash || ""}>
            {result?.dataset_hash ? `${result.dataset_hash.slice(0, 12)}…` : "—"}
          </code>
        </Fact>
      </dl>

      {dropped.length > 0 && (
        <p className="dropped-note">
          <Scissors size={13} />
          <span>
            Dropped before training: <strong>{dropped.join(", ")}</strong>. These columns were identifiers, constant,
            or free text — they carry no signal a model can generalise from.
          </span>
        </p>
      )}

      {numeric.length > 0 && (
        <StatTable
          caption="Numeric columns"
          headers={["Column", "Missing", "Mean", "Median", "Min", "Max", "Std dev"]}
          rows={numeric.map(([name, stats]) => [
            name,
            missing(stats.null_count),
            formatNumber(stats.mean, 3),
            formatNumber(stats.median, 3),
            formatNumber(stats.min, 3),
            formatNumber(stats.max, 3),
            formatNumber(stats.std, 3),
          ])}
        />
      )}

      {categorical.length > 0 && (
        <StatTable
          caption="Categorical columns"
          headers={["Column", "Missing", "Distinct values", "Most common"]}
          rows={categorical.map(([name, stats]) => [
            name,
            missing(stats.null_count),
            (stats.unique_count ?? 0).toLocaleString(),
            stats.most_frequent ?? "—",
          ])}
        />
      )}
    </div>
  );
}

function Fact({ label, children }) {
  return (
    <div className="fact">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function StatTable({ caption, headers, rows }) {
  return (
    <div className="summary-table-block">
      <div className="chart-table-wrap">
        <table className="chart-table">
          <caption className="summary-table-caption">
            {caption} ({rows.length})
          </caption>
          <thead>
            <tr>
              {headers.map((header) => (
                <th key={header} scope="col">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row[0]}>
                <th scope="row" className="summary-column-name">
                  {row[0]}
                </th>
                {row.slice(1).map((cell, index) => (
                  <td key={index} className="tabular">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** A bare count is unreadable without the denominator it came from. */
function describeMissing(count, rows) {
  const missing = count ?? 0;
  if (missing === 0) return "none";
  const pct = rows > 0 ? (missing / rows) * 100 : 0;
  return `${missing.toLocaleString()} (${pct < 0.1 ? "<0.1" : pct.toFixed(1)}%)`;
}

function formatTimestamp(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}
