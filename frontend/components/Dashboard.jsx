import React from "react";
import { BarChart3, Lightbulb, Target, Award, Info, Crosshair, ShieldCheck, ShieldAlert, ShieldX, XCircle } from "lucide-react";

const VERDICTS = {
  strong: { color: "#34d399", icon: ShieldCheck, label: "Healthy result" },
  fair: { color: "#fbbf24", icon: ShieldAlert, label: "Usable with caveats" },
  unreliable: { color: "#f87171", icon: ShieldX, label: "Low confidence" },
};

// A score means nothing without the baseline it should beat. The backend
// computes the verdict; this renders it before anything else, so a run that
// scored below a constant predictor can never be read as "Optimized Model".
function HealthBanner({ health }) {
  if (!health || !health.verdict) return null;
  const config = VERDICTS[health.verdict] || VERDICTS.fair;
  const Icon = config.icon;
  const isPercent = health.metric === "accuracy";
  const format = (v) =>
    typeof v !== "number" ? "—" : isPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);

  return (
    <div
      className="card fade-in"
      style={{
        borderLeft: `4px solid ${config.color}`,
        background: `${config.color}0f`,
        marginBottom: "24px",
      }}
    >
      <div style={{ display: "flex", gap: "16px", alignItems: "flex-start" }}>
        <div style={{ color: config.color, flexShrink: 0, marginTop: "2px" }}>
          <Icon size={28} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "12px", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, fontSize: "1.1rem", color: config.color }}>{health.headline}</h2>
            <span style={{ fontSize: "0.8rem", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {config.label}
            </span>
          </div>
          <p style={{ margin: "8px 0 12px", fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            {health.metric === "accuracy" ? "Accuracy" : "RMSE"} <strong>{format(health.score)}</strong>
            {" vs naive baseline "}
            <strong>{format(health.baseline)}</strong>
            {health.metric === "accuracy"
              ? " (always predicting the most common class)"
              : " (always predicting the mean)"}
          </p>
          {(health.reasons || []).length > 0 && (
            <ul style={{ margin: 0, paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "6px" }}>
              {health.reasons.map((reason, idx) => (
                <li key={idx} style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>{reason}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function FailedModels({ failed }) {
  const entries = Object.entries(failed || {});
  if (entries.length === 0) return null;

  return (
    <div className="card fade-in" style={{ borderLeft: "4px solid #f87171" }}>
      <h2>
        <XCircle size={20} style={{ color: "#f87171" }} />
        Models That Failed ({entries.length})
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {entries.map(([name, error]) => (
          <div
            key={name}
            style={{
              padding: "12px",
              background: "rgba(248, 113, 113, 0.06)",
              border: "1px solid rgba(248, 113, 113, 0.2)",
              borderRadius: "12px",
            }}
          >
            <div style={{ fontWeight: 600, color: "#f8717b", marginBottom: "4px", fontSize: "0.9rem" }}>{name}</div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-dim)", wordBreak: "break-word" }}>{error}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureImportance({ data, method }) {
  if (!data || data.length === 0) return null;
  const maxImportance = Math.max(...data.map((f) => f.importance)) || 1;
  const methodLabel =
    method === "shap" ? "SHAP values" : method === "native" ? "model-native importances (SHAP unavailable)" : "";

  return (
    <div className="card fade-in">
      <h2>
        <BarChart3 size={20} style={{ color: "var(--primary)" }} />
        Key Feature Influencers
      </h2>
      {methodLabel && (
        <p style={{ margin: "-8px 0 16px", fontSize: "0.78rem", color: "var(--text-dim)" }}>
          Computed from {methodLabel}. One-hot columns are summed back into their source column.
        </p>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {data.map((f, i) => (
          <div key={i}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px", fontSize: "0.9rem" }}>
              <span style={{ color: "var(--text-primary)", fontWeight: "600" }}>{f.feature}</span>
              <span style={{ color: "var(--text-dim)" }}>{f.importance.toFixed(4)}</span>
            </div>
            <div className="progress-bar" style={{ height: "10px" }}>
              <div className="progress-inner" style={{ width: `${(f.importance / maxImportance) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon, color }) {
  return (
    <div className="stat-card fade-in">
      <div style={{
        display: "inline-flex",
        padding: "10px",
        borderRadius: "12px",
        background: "rgba(255,255,255,0.03)",
        marginBottom: "16px",
        color: color || "var(--primary)"
      }}>
        {Icon && <Icon size={20} />}
      </div>
      <div className="stat-value" style={{ color: color }}>
        {typeof value === "number" ? value.toFixed(4) : value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function Dashboard({ result }) {
  if (!result) return null;

  const performance = result.all_model_scores?.[result.selected_model] || {};
  const health = result.health || {};
  const unreliable = health.verdict === "unreliable";
  const modelColor = unreliable ? "#f87171" : "var(--secondary)";

  return (
    <div className="dashboard-root fade-in">
      <HealthBanner health={health} />

      <div className="stats-grid" style={{ marginBottom: "32px" }}>
        <MetricCard
          label={unreliable ? "Selected Model (low confidence)" : "Selected Model"}
          value={result.selected_model}
          icon={Award}
          color={modelColor}
        />
        <MetricCard label="Problem Domain" value={result.problem_type} icon={Target} />
        <MetricCard label="Target Column" value={result.target || "—"} icon={Crosshair} />
        {Object.entries(performance).map(([metric, val]) => (
          <MetricCard key={metric} label={metric.replace('_', ' ')} value={val} icon={Info} />
        ))}
      </div>

      <div className="grid-2">
        <FeatureImportance data={result.feature_importance} method={result.explanation_method} />

        <div className="card fade-in">
          <h2>
            <Lightbulb size={20} style={{ color: "#fbbf24" }} />
            Autonomous Insights
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {(result.insights || []).map((insight, idx) => (
              <div key={idx} style={{
                display: "flex",
                gap: "12px",
                padding: "12px",
                background: "rgba(255,255,255,0.02)",
                borderRadius: "12px",
                border: "1px solid var(--border-glass)",
                fontSize: "0.95rem",
                color: "var(--text-secondary)"
              }}>
                <div style={{ color: "#fbbf24", flexShrink: 0 }}>•</div>
                {insight}
              </div>
            ))}
          </div>
        </div>
      </div>

      <FailedModels failed={result.failed_models} />
    </div>
  );
}
