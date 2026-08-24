import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Clock, FolderOpen, GitCompare, Loader2, RefreshCw, Trash2 } from "lucide-react";

import { RunComparison } from "./charts/ModelScoresChart";
import { VERDICT_COLOR, formatNumber } from "./charts/theme";
import { compareRuns, deleteRun, getInsights, getRunResult, listRuns } from "../services/api";

const MAX_COMPARE = 4;

const VERDICT_LABEL = {
  strong: "Healthy",
  fair: "Caveats",
  unreliable: "Low confidence",
};

/**
 * Past runs, backed by the registry.
 *
 * Before this, every result vanished on refresh. A run here can be reopened
 * into the dashboard, compared against up to three others, or deleted along
 * with its artifacts.
 */
export default function Workspace({ onOpenRun, activeRunKey, authStatus }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [busyKey, setBusyKey] = useState("");
  const [expandedKey, setExpandedKey] = useState("");
  // Fetched summaries, keyed by run. Re-expanding a row does not refetch.
  const [previews, setPreviews] = useState({});
  const [previewError, setPreviewError] = useState("");

  const controllerRef = useRef(null);

  const refresh = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError("");
    try {
      const body = await listRuns(50, { signal: controller.signal });
      setRuns(body.runs || []);
    } catch (err) {
      if (err?.name !== "AbortError") setError(err.message || "Could not load past runs.");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    // Abort the in-flight request if the panel unmounts mid-load.
    return () => controllerRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- authStatus is
    // a re-fetch trigger, not a value read inside refresh(): this panel
    // mounts before a signed-out visitor has a key, so its first request
    // 401s, and nothing else would tell it to try again once they sign in.
  }, [refresh, activeRunKey, authStatus]);

  const toggleSelected = (runKey) => {
    setComparison(null);
    setSelected((current) =>
      current.includes(runKey)
        ? current.filter((key) => key !== runKey)
        : current.length >= MAX_COMPARE
          ? current
          : [...current, runKey]
    );
  };

  const handleCompare = async () => {
    setError("");
    try {
      setComparison(await compareRuns(selected));
    } catch (err) {
      setError(err.message || "Could not compare those runs.");
    }
  };

  const handleOpen = async (runKey) => {
    setBusyKey(runKey);
    setError("");
    try {
      onOpenRun(await getRunResult(runKey));
    } catch (err) {
      setError(err.message || "Could not reopen that run.");
    } finally {
      setBusyKey("");
    }
  };

  /**
   * Expanding a row asks /api/insights for that run's summary. It is the
   * cheap read — health, scores, and attribution without the full dashboard
   * payload — so a past run can be judged before it is opened.
   */
  const togglePreview = async (runKey) => {
    setPreviewError("");
    if (expandedKey === runKey) {
      setExpandedKey("");
      return;
    }
    setExpandedKey(runKey);
    if (previews[runKey]) return;
    try {
      const insights = await getInsights(runKey);
      setPreviews((current) => ({ ...current, [runKey]: insights }));
    } catch (err) {
      setPreviewError(err.message || "Could not load that run's summary.");
    }
  };

  const handleDelete = async (runKey, label) => {
    // Deletion takes the model and every artifact with it — nothing here is
    // recoverable afterward, so the trash icon needs a deliberate second
    // click, not just a steady hand near a small target in a dense row.
    if (!window.confirm(`Delete ${label || "this run"}? This removes its model and artifacts and cannot be undone.`)) {
      return;
    }
    setBusyKey(runKey);
    setError("");
    try {
      await deleteRun(runKey);
      setSelected((current) => current.filter((key) => key !== runKey));
      setPreviews((current) => {
        const { [runKey]: _removed, ...rest } = current;
        return rest;
      });
      if (expandedKey === runKey) setExpandedKey("");
      setComparison(null);
      await refresh();
    } catch (err) {
      setError(err.message || "Could not delete that run.");
    } finally {
      setBusyKey("");
    }
  };

  return (
    <div className="card fade-in">
      <div className="workspace-head">
        <h2 className="panel-head-title">
          <FolderOpen size={20} style={{ color: "var(--primary)" }} />
          Dataset workspace
        </h2>
        <div style={{ display: "flex", gap: "8px" }}>
          {selected.length >= 2 && (
            <button type="button" className="btn btn-secondary" onClick={handleCompare}>
              <GitCompare size={14} />
              Compare {selected.length}
            </button>
          )}
          <button type="button" className="btn btn-secondary" onClick={refresh} disabled={loading}>
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}

      {runs.length === 0 && !loading ? (
        <p style={{ color: "var(--text-dim)", fontSize: "0.9rem", margin: "8px 0 0" }}>
          No past runs yet. Train a dataset and it will be listed here — results survive a refresh and a restart.
        </p>
      ) : (
        <div className="workspace-table-wrap">
          <table className="workspace-table">
            <thead>
              <tr>
                <th scope="col">
                  <span className="sr-only">Compare</span>
                </th>
                <th scope="col">Dataset</th>
                <th scope="col">Target</th>
                <th scope="col">Mode</th>
                <th scope="col">Model</th>
                <th scope="col">Health</th>
                <th scope="col">Size</th>
                <th scope="col">When</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <React.Fragment key={run.run_key}>
                <tr className={run.run_key === activeRunKey ? "workspace-row-active" : undefined}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.includes(run.run_key)}
                      onChange={() => toggleSelected(run.run_key)}
                      disabled={!selected.includes(run.run_key) && selected.length >= MAX_COMPARE}
                      aria-label={`Compare ${run.filename || run.run_key}`}
                    />
                  </td>
                  <th scope="row" className="workspace-name">
                    {run.filename || `${run.run_key.slice(0, 10)}…`}
                  </th>
                  <td>{run.target || "—"}</td>
                  <td>{run.mode}</td>
                  <td>{run.selected_model || "—"}</td>
                  <td>
                    {run.health_verdict ? (
                      <span className="verdict-pill" style={{ color: VERDICT_COLOR[run.health_verdict] }}>
                        {/* Colour never carries this alone — the word is the label. */}
                        <span
                          className="verdict-dot"
                          style={{ background: VERDICT_COLOR[run.health_verdict] }}
                          aria-hidden="true"
                        />
                        {VERDICT_LABEL[run.health_verdict] || run.health_verdict}
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="tabular">
                    {run.row_count?.toLocaleString()} × {run.column_count}
                  </td>
                  <td className="workspace-when">
                    <Clock size={12} />
                    {formatWhen(run.created_at)}
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: "6px", justifyContent: "flex-end" }}>
                      <button
                        type="button"
                        className="btn btn-secondary btn-compact"
                        onClick={() => togglePreview(run.run_key)}
                        aria-expanded={expandedKey === run.run_key}
                        aria-label={`Summary of ${run.filename || run.run_key}`}
                      >
                        {expandedKey === run.run_key ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                        Summary
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-compact"
                        onClick={() => handleOpen(run.run_key)}
                        disabled={busyKey === run.run_key}
                      >
                        {busyKey === run.run_key ? <Loader2 size={13} className="animate-spin" /> : "Open"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-danger btn-compact"
                        onClick={() => handleDelete(run.run_key, run.filename || run.run_key)}
                        disabled={busyKey === run.run_key}
                        aria-label={`Delete ${run.filename || run.run_key}`}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </td>
                </tr>
                {expandedKey === run.run_key && (
                  <tr className="workspace-preview-row">
                    <td colSpan={9}>
                      <RunPreview insights={previews[run.run_key]} error={previewError} />
                    </td>
                  </tr>
                )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {comparison && (
        <div style={{ marginTop: "24px" }}>
          <RunComparison comparison={comparison} />
        </div>
      )}
    </div>
  );
}

/**
 * A past run at a glance, from /api/insights — the endpoint that had no
 * client caller at all. Enough to decide whether a run is worth reopening:
 * the health verdict, what the selected model scored, and what drove it.
 */
function RunPreview({ insights, error }) {
  if (error) {
    return (
      <div className="notice notice-error" role="alert">
        {error}
      </div>
    );
  }
  if (!insights) {
    return (
      <p className="run-preview-loading">
        <Loader2 size={14} className="animate-spin" />
        Loading summary…
      </p>
    );
  }

  const health = insights.health || {};
  const scores = insights.scores?.[insights.model] || {};
  const topFeatures = (insights.feature_importance || []).slice(0, 5);
  const failed = Object.keys(insights.failed_models || {});
  const methodLabel = insights.explanation_method === "shap" ? "SHAP" : "model-native";

  return (
    <div className="run-preview">
      <div className="run-preview-block">
        <h4>Result health</h4>
        {health.verdict ? (
          <p style={{ color: VERDICT_COLOR[health.verdict] }}>{health.headline}</p>
        ) : (
          <p className="run-preview-dim">No verdict stored for this run.</p>
        )}
        <dl className="run-preview-scores">
          {Object.entries(scores).map(([metric, value]) => (
            <div key={metric}>
              <dt>{metric.replace(/_/g, " ")}</dt>
              <dd className="tabular">{formatNumber(value, 4)}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="run-preview-block">
        <h4>What drove it ({methodLabel})</h4>
        {topFeatures.length === 0 ? (
          <p className="run-preview-dim">No feature attribution stored for this run.</p>
        ) : (
          <ol className="run-preview-features">
            {topFeatures.map((item) => (
              <li key={item.feature}>
                <span>{item.feature}</span>
                <span className="tabular run-preview-dim">{formatNumber(item.importance, 3)}</span>
              </li>
            ))}
          </ol>
        )}
        {failed.length > 0 && <p className="run-preview-failed">Failed to train: {failed.join(", ")}</p>}
      </div>
    </div>
  );
}

function formatWhen(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`;
  if (minutes < 60 * 24 * 7) return `${Math.round(minutes / (60 * 24))}d ago`;
  return date.toLocaleDateString();
}
