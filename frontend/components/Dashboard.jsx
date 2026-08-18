import React from "react";
import {
  Award,
  Crosshair,
  FileWarning,
  History,
  Info,
  Lightbulb,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  Target,
  XCircle,
} from "lucide-react";

import DatasetSummary from "./DatasetSummary";
import { CategoryBar, FeatureImportanceChart, NumericHistogram } from "./charts/DistributionCharts";
import { ConfusionMatrix, CorrelationHeatmap, ResidualPlot } from "./charts/DiagnosticCharts";
import { ModelScores } from "./charts/ModelScoresChart";
import { VERDICT_COLOR, formatNumber } from "./charts/theme";

const VERDICTS = {
  strong: { icon: ShieldCheck, label: "Healthy result" },
  fair: { icon: ShieldAlert, label: "Usable with caveats" },
  unreliable: { icon: ShieldX, label: "Low confidence" },
};

// A score means nothing without the baseline it should beat. The backend
// computes the verdict; this renders it before anything else, so a run that
// scored below a constant predictor can never be read as "Optimized Model".
function HealthBanner({ health }) {
  if (!health?.verdict) return null;
  const config = VERDICTS[health.verdict] || VERDICTS.fair;
  const color = VERDICT_COLOR[health.verdict];
  const Icon = config.icon;
  const isPercent = health.metric === "accuracy";
  const format = (value) =>
    typeof value !== "number" ? "—" : isPercent ? `${(value * 100).toFixed(1)}%` : formatNumber(value, 4);

  return (
    <div className="card fade-in" style={{ borderLeft: `4px solid ${color}`, background: `${color}0f`, marginBottom: 24 }}>
      <div style={{ display: "flex", gap: "16px", alignItems: "flex-start" }}>
        <div style={{ color, flexShrink: 0, marginTop: 2 }}>
          <Icon size={28} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "12px", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: "1.1rem", color }}>{health.headline}</h2>
            <span className="banner-tag">{config.label}</span>
          </div>
          <p style={{ margin: "8px 0 12px", fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            {`${isPercent ? "Accuracy" : "RMSE"} ${format(health.score)} vs naive baseline ${format(health.baseline)}`}
            {isPercent ? " (always predicting the most common class)" : " (always predicting the mean)"}
          </p>
          {(health.reasons || []).length > 0 && (
            <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 6 }}>
              {health.reasons.map((reason, index) => (
                <li key={index} style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>
                  {reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

const PROVENANCE = {
  trained: "Freshly trained just now.",
  reused: "Replayed from cache — this dataset, mode, and target had already been trained.",
  reopened: "Reopened from the workspace — stored artifacts, not a new training run.",
};

// Without this, a cache replay and a fresh run look identical, and the
// "Analysing…" progress bar implies work that never happened.
function Provenance({ status, message }) {
  const note = PROVENANCE[status];
  if (!note) return null;
  return (
    <p className="provenance">
      <History size={13} />
      <span>{message || note}</span>
    </p>
  );
}

function MetricCard({ label, value, icon: Icon, color }) {
  return (
    <div className="stat-card fade-in">
      <div className="stat-icon" style={{ color: color || "var(--primary)" }}>
        {Icon && <Icon size={20} />}
      </div>
      <div className="stat-value" style={{ color }}>
        {typeof value === "number" ? formatNumber(value, 4) : value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function FailedModels({ failed }) {
  const entries = Object.entries(failed || {});
  if (entries.length === 0) return null;
  return (
    <div className="card fade-in" style={{ borderLeft: "4px solid #d03b3b" }}>
      <h2>
        <XCircle size={20} style={{ color: "#d03b3b" }} />
        Models that failed ({entries.length})
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {entries.map(([name, message]) => (
          <div key={name} className="failure-item">
            <div className="failure-name">{name}</div>
            <div className="failure-message">{message}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ParsingNotes({ report, quality }) {
  const messages = [...(report?.warnings || []), ...(quality?.warnings || [])];
  if (messages.length === 0) return null;
  return (
    <div className="card fade-in" style={{ borderLeft: "4px solid #fab219" }}>
      <h2>
        <FileWarning size={20} style={{ color: "#fab219" }} />
        How your file was read
      </h2>
      <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 8 }}>
        {messages.map((message, index) => (
          <li key={index} style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            {message}
          </li>
        ))}
      </ul>
      {report?.rows_parsed !== undefined && (
        <p style={{ margin: "12px 0 0", fontSize: "0.8rem", color: "var(--text-dim)" }}>
          {report.rows_parsed.toLocaleString()} rows parsed with delimiter{" "}
          <code>{report.delimiter === "\t" ? "\\t" : report.delimiter}</code>, encoding {report.encoding}.
        </p>
      )}
    </div>
  );
}

export default function Dashboard({ result }) {
  if (!result) return null;

  const performance = result.all_model_scores?.[result.selected_model] || {};
  const health = result.health || {};
  const unreliable = health.verdict === "unreliable";
  const diagnostics = result.diagnostics || {};
  const charts = result.charts || {};
  const histograms = Object.entries(charts.numeric_histograms || {});
  const categories = Object.entries(charts.categorical_bars || {});

  return (
    <div className="dashboard-root fade-in">
      <Provenance status={result.status} message={result.message} />
      <HealthBanner health={health} />

      <div className="stats-grid" style={{ marginBottom: 32 }}>
        <MetricCard
          label={unreliable ? "Selected model (low confidence)" : "Selected model"}
          value={result.selected_model}
          icon={Award}
          color={unreliable ? VERDICT_COLOR.unreliable : "var(--secondary)"}
        />
        <MetricCard label="Problem domain" value={result.problem_type} icon={Target} />
        <MetricCard label="Target column" value={result.target || "—"} icon={Crosshair} />
        {Object.entries(performance).map(([metric, value]) => (
          <MetricCard key={metric} label={metric.replace("_", " ")} value={value} icon={Info} />
        ))}
      </div>

      <div className="grid-2">
        <FeatureImportanceChart data={result.feature_importance} method={result.explanation_method} />
        <ModelScores
          scores={result.all_model_scores}
          problemType={result.problem_type}
          selectedModel={result.selected_model}
        />
      </div>

      <div className="grid-2">
        {result.problem_type === "classification" ? (
          <ConfusionMatrix data={diagnostics.confusion_matrix} />
        ) : (
          <ResidualPlot data={diagnostics.residuals} />
        )}
        <CorrelationHeatmap data={diagnostics.correlation} />
      </div>

      {(histograms.length > 0 || categories.length > 0) && (
        <div className="grid-2">
          {histograms.map(([column, data]) => (
            <NumericHistogram key={column} column={column} data={data} />
          ))}
          {categories.map(([column, data]) => (
            <CategoryBar key={column} column={column} data={data} />
          ))}
        </div>
      )}

      <div className="grid-2">
        <div className="card fade-in">
          <h2>
            <Lightbulb size={20} style={{ color: "#fab219" }} />
            Autonomous insights
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {(result.insights || []).map((insight, index) => (
              <div key={index} className="insight-item">
                <div style={{ color: "#fab219", flexShrink: 0 }}>•</div>
                {insight}
              </div>
            ))}
          </div>
        </div>
        <ParsingNotes report={result.parse_report} quality={result.quality_report} />
      </div>

      <DatasetSummary result={result} />

      <FailedModels failed={result.failed_models} />
    </div>
  );
}
