import React from "react";
import { BarChart3, Lightbulb, Target, Award, Info } from "lucide-react";

function FeatureImportance({ data }) {
  if (!data || data.length === 0) return null;
  const maxImportance = Math.max(...data.map(f => f.importance));
  
  return (
    <div className="card fade-in">
      <h2>
        <BarChart3 size={20} style={{ color: "var(--primary)" }} />
        Key Feature Influencers
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {data.map((f, i) => (
          <div key={i}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px", fontSize: "0.9rem" }}>
              <span style={{ color: "var(--text-primary)", fontWeight: "600" }}>{f.feature}</span>
              <span style={{ color: "var(--text-dim)" }}>{(f.importance * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-bar" style={{ height: "10px" }}>
              <div 
                className="progress-inner" 
                style={{ width: `${(f.importance / maxImportance) * 100}%` }} 
              />
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
  
  return (
    <div className="dashboard-root fade-in">
      <div className="stats-grid" style={{ marginBottom: "32px" }}>
        <MetricCard 
          label="Optimized Model" 
          value={result.selected_model} 
          icon={Award} 
          color="var(--secondary)" 
        />
        <MetricCard 
          label="Problem Domain" 
          value={result.problem_type} 
          icon={Target} 
        />
        {Object.entries(performance).map(([metric, val]) => (
          <MetricCard 
            key={metric} 
            label={metric.replace('_', ' ')} 
            value={val} 
            icon={Info}
          />
        ))}
      </div>

      <div className="grid-2">
        <FeatureImportance data={result.feature_importance} />
        
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
    </div>
  );
}
