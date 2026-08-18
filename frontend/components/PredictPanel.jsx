import React, { useMemo, useState } from "react";
import { AlertTriangle, Download, FileSpreadsheet, Loader2, Play, Rows3 } from "lucide-react";

import { formatNumber } from "./charts/theme";
import { predictDataset, predictSingleRow } from "../services/api";

/**
 * Inference results as a table, not a JSON dump.
 *
 * Two ways in: a CSV of many rows, or one row typed in — the second exists
 * because checking a single hypothetical should not require making a file.
 */
export default function PredictPanel({ runKey, features = [], problemType, target }) {
  const [tab, setTab] = useState("file");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [row, setRow] = useState(() => Object.fromEntries(features.map((feature) => [feature, ""])));

  const templateCsv = useMemo(() => (features.length ? `${features.join(",")}\n` : ""), [features]);

  const submitFile = async (event) => {
    event.preventDefault();
    const file = event.target.elements.predictFile?.files?.[0];
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      setResult(await predictDataset(runKey, file));
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const submitRow = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setResult(await predictSingleRow(runKey, row));
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const download = (content, filename) => {
    const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadPredictions = () => {
    if (!result?.rows?.length) return;
    const inputs = result.input_rows || [];
    const header = [...(result.features || []), `predicted_${result.target || "value"}`];
    const hasConfidence = result.rows.some((item) => item.confidence !== undefined);
    if (hasConfidence) header.push("confidence");

    const lines = [header.map(csvCell).join(",")];
    result.rows.forEach((item, index) => {
      const source = inputs[index] || {};
      const cells = (result.features || []).map((feature) => csvCell(source[feature]));
      cells.push(csvCell(item.prediction));
      if (hasConfidence) cells.push(item.confidence === undefined ? "" : item.confidence.toFixed(4));
      lines.push(cells.join(","));
    });
    download(`${lines.join("\n")}\n`, `predictions_${runKey.slice(0, 10)}.csv`);
  };

  return (
    <div className="card fade-in">
      <h2>
        <Play size={20} style={{ color: "var(--primary)" }} />
        Predictive inference
      </h2>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "16px" }}>
        Apply the trained model to new rows{target ? ` and predict ${target}` : ""}.
      </p>

      <div className="tab-row" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "file"}
          className={tab === "file" ? "tab tab-active" : "tab"}
          onClick={() => setTab("file")}
        >
          <FileSpreadsheet size={14} />
          Upload a CSV
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "row"}
          className={tab === "row" ? "tab tab-active" : "tab"}
          onClick={() => setTab("row")}
        >
          <Rows3 size={14} />
          Enter one row
        </button>
      </div>

      {tab === "file" ? (
        <form onSubmit={submitFile}>
          <div className="input-group">
            <label htmlFor="predict-file">Inference data (CSV)</label>
            <input
              id="predict-file"
              className="input"
              type="file"
              name="predictFile"
              accept=".csv,.tsv,.txt"
              required
              disabled={loading}
            />
            <p style={{ margin: "8px 0 0", fontSize: "0.78rem", color: "var(--text-dim)" }}>
              Required columns: {features.join(", ") || "—"}
            </p>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <button className="btn btn-primary" type="submit" disabled={loading} style={{ flex: 1 }}>
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
              Run predictions
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => download(templateCsv, `template_${runKey.slice(0, 10)}.csv`)}
              disabled={!templateCsv}
            >
              <Download size={14} />
              Template
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={submitRow}>
          <div className="row-grid">
            {features.map((feature) => (
              <div className="input-group" key={feature} style={{ margin: 0 }}>
                <label htmlFor={`row-${feature}`}>{feature}</label>
                <input
                  id={`row-${feature}`}
                  className="input"
                  value={row[feature] ?? ""}
                  onChange={(event) => setRow((current) => ({ ...current, [feature]: event.target.value }))}
                  disabled={loading}
                  required
                />
              </div>
            ))}
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading} style={{ width: "100%", marginTop: 16 }}>
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            Predict this row
          </button>
        </form>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {result?.rows?.length > 0 && (
        <div className="fade-in" style={{ marginTop: "24px" }}>
          <div className="results-head">
            <h3 style={{ fontSize: "0.98rem", margin: 0 }}>
              {result.rows.length.toLocaleString()} prediction{result.rows.length === 1 ? "" : "s"}
            </h3>
            <button type="button" className="btn btn-secondary btn-compact" onClick={downloadPredictions}>
              <Download size={13} />
              Download CSV
            </button>
          </div>

          {result.parse_warnings?.length > 0 && (
            <div className="notice notice-warning" role="status">
              <AlertTriangle size={14} />
              <div>
                {result.parse_warnings.map((warning, index) => (
                  <p key={index} style={{ margin: index ? "4px 0 0" : 0 }}>
                    {warning}
                  </p>
                ))}
              </div>
            </div>
          )}

          <div className="results-table-wrap">
            <table className="results-table">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  {(result.features || []).map((feature) => (
                    <th key={feature} scope="col">
                      {feature}
                    </th>
                  ))}
                  <th scope="col">Predicted {result.target || ""}</th>
                  {problemType !== "regression" && <th scope="col">Confidence</th>}
                </tr>
              </thead>
              <tbody>
                {result.rows.slice(0, 100).map((item, index) => (
                  <tr key={item.index ?? index}>
                    <td className="tabular" style={{ color: "var(--text-dim)" }}>
                      {index + 1}
                    </td>
                    {(result.features || []).map((feature) => (
                      <td key={feature} className="tabular">
                        {formatCell(result.input_rows?.[index]?.[feature])}
                      </td>
                    ))}
                    <td>
                      <strong>{formatCell(item.prediction)}</strong>
                    </td>
                    {problemType !== "regression" && (
                      <td className="tabular">
                        {item.confidence === undefined ? (
                          <span style={{ color: "var(--text-dim)" }}>n/a</span>
                        ) : (
                          <span title={describeProbabilities(item.class_probabilities)}>
                            {(item.confidence * 100).toFixed(1)}%
                          </span>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {result.rows.length > 100 && (
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: "var(--text-dim)" }}>
              Showing the first 100 rows. Download the CSV for all {result.rows.length.toLocaleString()}.
            </p>
          )}
          {result.confidences === null && problemType === "classification" && (
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: "var(--text-dim)" }}>
              This model is an ensemble vote, which has no calibrated probability to report.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function formatCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

function describeProbabilities(probabilities) {
  if (!probabilities) return "";
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => `${label}: ${(value * 100).toFixed(1)}%`)
    .join(" · ");
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
