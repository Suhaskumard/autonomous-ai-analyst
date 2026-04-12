import React, { useMemo, useState } from "react";
import { Play, Download, FileText, AlertTriangle, FileSpreadsheet, Loader2 } from "lucide-react";

export default function PredictPanel({ datasetHash, features = [], onPredict }) {
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async (e) => {
    e.preventDefault();
    const file = e.target.predictFile.files[0];
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const res = await onPredict(datasetHash, file);
      setPredictionResult(res);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const templateCsv = useMemo(() => {
    if (!features.length) return "";
    const header = features.join(",");
    const sample = features.map(() => "").join(",");
    return `${header}\n${sample}\n`;
  }, [features]);

  const downloadTemplate = () => {
    if (!templateCsv) return;
    const blob = new Blob([templateCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${datasetHash}_prediction_template.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card fade-in">
      <h2>
        <Play size={20} style={{ color: "var(--primary)" }} />
        Predictive Inference
      </h2>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginBottom: "20px" }}>
        Apply the optimized model to new, unseen data samples.
      </p>
      
      {features.length > 0 && (
        <div style={{ 
          background: "rgba(255,255,255,0.03)", 
          padding: "16px", 
          borderRadius: "16px", 
          marginBottom: "24px",
          border: "1px solid var(--border-glass)"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px", fontSize: "0.9rem", color: "var(--text-primary)" }}>
            <FileSpreadsheet size={16} />
            Schema Validation Required
          </div>
          <div style={{ 
            fontSize: "0.8rem", 
            color: "var(--text-dim)", 
            display: "flex", 
            flexWrap: "wrap", 
            gap: "8px",
            marginBottom: "16px"
          }}>
            {features.map((f, i) => (
              <span key={i} style={{ background: "rgba(255,255,255,0.05)", padding: "2px 8px", borderRadius: "100px" }}>
                {f}
              </span>
            ))}
          </div>
          <button 
            className="btn btn-secondary" 
            type="button" 
            onClick={downloadTemplate}
            style={{ width: "100%", fontSize: "0.8rem" }}
          >
            <Download size={14} />
            Download Validation Template
          </button>
        </div>
      )}

      <form onSubmit={handlePredict}>
        <div className="input-group">
          <label>Inference Data (CSV)</label>
          <input className="input" type="file" name="predictFile" accept=".csv" required disabled={loading} />
        </div>
        <button className="btn btn-primary" type="submit" disabled={loading} style={{ width: "100%" }}>
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Running Inference...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Predictions
            </>
          )}
        </button>
      </form>

      {error && (
        <div style={{ 
          marginTop: "16px", 
          padding: "12px", 
          borderRadius: "12px", 
          background: "rgba(239, 68, 68, 0.1)", 
          color: "#f87171",
          fontSize: "0.9rem",
          display: "flex",
          gap: "10px"
        }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {predictionResult && (
        <div className="fade-in" style={{ marginTop: "24px" }}>
          <h3 style={{ fontSize: "1rem", marginBottom: "12px", color: "var(--text-primary)" }}>
            <FileText size={16} />
            Inference Results
          </h3>
          <div style={{ 
            background: "#020617", 
            borderRadius: "12px", 
            padding: "16px", 
            maxHeight: "300px", 
            overflowY: "auto",
            border: "1px solid var(--border-glass)"
          }}>
            <pre style={{ 
              margin: 0, 
              fontSize: "0.85rem", 
              color: "#34d399",
              fontFamily: "'JetBrains Mono', monospace"
            }}>
              {JSON.stringify(predictionResult, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
