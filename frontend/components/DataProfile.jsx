import React, { useMemo, useState } from "react";
import { AlertTriangle, Ban, ChevronRight, Crosshair, Database, Hash, Loader2, Type } from "lucide-react";

import { Sparkline } from "./charts/DistributionCharts";
import { formatNumber } from "./charts/theme";

/**
 * What is in this file, before anything is trained.
 *
 * The target picker lives in this same screen on purpose: choosing a target is
 * the one decision that decides whether a run is meaningful, and it should be
 * made while looking at the columns rather than from a bare dropdown.
 */
export default function DataProfile({ profile, mode, manualModel, onModeChange, onManualModelChange, onTrain, onCancel, isTraining }) {
  const columns = profile?.column_profiles || [];
  const [target, setTarget] = useState(() => {
    const suggested = columns.find((column) => column.name === profile?.suggested_target);
    return suggested?.usable_as_target ? suggested.name : "";
  });

  const selected = useMemo(() => columns.find((column) => column.name === target), [columns, target]);
  const usableCount = columns.filter((column) => column.usable_as_target).length;
  const parse = profile?.parse_report;

  return (
    <div className="card fade-in">
      <div className="profile-head">
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div className="icon-chip">
            <Database size={22} />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: "1.2rem" }}>{profile?.filename || "Dataset"}</h2>
            <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-dim)" }}>
              {`${(profile?.rows ?? 0).toLocaleString()} rows · ${profile?.columns ?? 0} columns · ${usableCount} usable as a target`}
            </p>
          </div>
        </div>
        <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={isTraining}>
          Choose a different file
        </button>
      </div>

      {parse?.warnings?.length > 0 && (
        <div className="notice notice-warning" role="status">
          <AlertTriangle size={16} />
          <div>
            {parse.warnings.map((warning, index) => (
              <p key={index} style={{ margin: index ? "4px 0 0" : 0 }}>
                {warning}
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="profile-table-wrap">
        <table className="profile-table">
          <caption className="sr-only">Per-column profile. Choose which column to predict.</caption>
          <thead>
            <tr>
              <th scope="col">Target</th>
              <th scope="col">Column</th>
              <th scope="col">Type</th>
              <th scope="col">Missing</th>
              <th scope="col">Unique</th>
              <th scope="col">Distribution</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((column) => (
              <tr key={column.name} className={target === column.name ? "profile-row-selected" : undefined}>
                <td>
                  <input
                    type="radio"
                    name="target-column"
                    id={`target-${column.name}`}
                    value={column.name}
                    checked={target === column.name}
                    disabled={!column.usable_as_target || isTraining}
                    onChange={() => setTarget(column.name)}
                    aria-label={`Predict ${column.name}`}
                  />
                </td>
                <th scope="row">
                  <label htmlFor={`target-${column.name}`} className="profile-column-name">
                    {column.name}
                  </label>
                </th>
                <td>
                  <span className="profile-type">
                    {column.dtype === "numeric" ? <Hash size={12} /> : <Type size={12} />}
                    {column.dtype}
                  </span>
                </td>
                <td className="tabular">
                  {column.missing_pct > 0 ? (
                    <span className={column.missing_pct > 40 ? "value-warning" : undefined}>
                      {column.missing_pct}%
                    </span>
                  ) : (
                    <span style={{ color: "var(--text-dim)" }}>none</span>
                  )}
                </td>
                <td className="tabular">
                  {column.unique.toLocaleString()}
                  <span style={{ color: "var(--text-dim)" }}> ({column.unique_pct}%)</span>
                </td>
                <td>
                  <Sparkline sparkline={column.sparkline} />
                </td>
                <td className="profile-note">
                  {!column.usable_as_target && (
                    <span className="profile-blocked">
                      <Ban size={12} />
                      {column.target_reason}
                    </span>
                  )}
                  {column.feature_note && <span className="profile-hint">{column.feature_note}</span>}
                  {column.usable_as_target && !column.feature_note && (
                    <span style={{ color: "var(--text-dim)" }}>{summarise(column)}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="profile-actions">
        <div className="input-group" style={{ margin: 0, minWidth: 220 }}>
          <label htmlFor="training-mode">Training mode</label>
          <select
            id="training-mode"
            className="input"
            value={mode}
            onChange={(event) => onModeChange(event.target.value)}
            disabled={isTraining}
          >
            <option value="auto">Auto-Pilot (fastest)</option>
            <option value="ensemble">Ensemble (vote of all models)</option>
            <option value="manual">Manual selection</option>
          </select>
        </div>

        {mode === "manual" && (
          <div className="input-group" style={{ margin: 0, minWidth: 220 }}>
            <label htmlFor="manual-model">Model</label>
            <select
              id="manual-model"
              className="input"
              value={manualModel}
              onChange={(event) => onManualModelChange(event.target.value)}
              disabled={isTraining}
            >
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="RandomForest">Random Forest</option>
              <option value="XGBoost">XGBoost</option>
              <option value="DecisionTree">Decision Tree</option>
              <option value="SVM">SVM</option>
            </select>
          </div>
        )}

        <div className="profile-target-summary">
          <Crosshair size={16} style={{ color: "var(--primary)" }} />
          {selected ? (
            <span>
              Predicting <strong>{selected.name}</strong> — {selected.target_reason}
            </span>
          ) : (
            <span style={{ color: "var(--text-dim)" }}>Pick a target column to continue.</span>
          )}
        </div>

        <button
          type="button"
          className="btn btn-primary"
          onClick={() => onTrain(target)}
          disabled={!target || isTraining}
        >
          {isTraining ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Training...
            </>
          ) : (
            <>
              Train on this dataset
              <ChevronRight size={18} />
            </>
          )}
        </button>
      </div>
    </div>
  );
}

function summarise(column) {
  if (column.dtype === "numeric" && column.stats) {
    return `${formatNumber(column.stats.min)} to ${formatNumber(column.stats.max)}, mean ${formatNumber(column.stats.mean)}`;
  }
  if (column.stats?.most_frequent) {
    return `most common: ${column.stats.most_frequent}`;
  }
  return "";
}
