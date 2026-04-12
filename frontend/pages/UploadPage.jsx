import React, { useState } from "react";
import { Layers, Activity, AlertCircle, CheckCircle2 } from "lucide-react";
import UploadForm from "../components/UploadForm";
import Dashboard from "../components/Dashboard";
import ChatBox from "../components/ChatBox";
import PredictPanel from "../components/PredictPanel";
import { chatQuery, getUploadJobStatus, predictDataset, startUploadJob } from "../services/api";

export default function UploadPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e, mode, manualModel) => {
    e.preventDefault();
    const file = e.target.file.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("mode", mode);
    if (mode === "manual") formData.append("manual_model", manualModel);

    setLoading(true);
    setError("");
    setJobStatus(null);
    setResult(null);
    try {
      const start = await startUploadJob(formData);
      const jobId = start.job_id;

      let done = false;
      while (!done) {
        const status = await getUploadJobStatus(jobId);
        setJobStatus(status);
        if (status.state === "completed") {
          setResult(status.result);
          done = true;
        } else if (status.state === "failed") {
          throw new Error(status.error || "Job failed.");
        } else {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }
    } catch (err) {
      setError(err.message || "Failed to process dataset.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="header fade-in">
        <h1 className="title">AUTONOMOUS AI ANALYST</h1>
        <p className="subtitle">
          Experience the future of data science. Upload your dataset and let our advanced 
          AI perform end-to-end analysis, modeling, and deep exploration.
        </p>
      </header>

      <div className="fade-in">
        <UploadForm onSubmit={handleSubmit} isLoading={loading} />
      </div>

      {loading && (
        <div className="card fade-in" style={{ borderLeft: "4px solid var(--primary)" }}>
          <h2 style={{ fontSize: "1.2rem", color: "var(--text-primary)" }}>
            <Activity size={20} style={{ color: "var(--primary)" }} />
            Processing Pipeline
          </h2>
          <div style={{ marginBottom: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontSize: "0.9rem" }}>
              <span style={{ color: "var(--text-secondary)" }}>{jobStatus?.current_step || "Initializing..."}</span>
              <span style={{ fontWeight: "700", color: "var(--primary)" }}>{jobStatus?.progress ?? 0}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-inner" style={{ width: `${jobStatus?.progress ?? 0}%` }} />
            </div>
          </div>
          
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "12px" }}>
            {(jobStatus?.status_log || []).slice().reverse().map((item, idx) => (
              <div key={idx} style={{ 
                fontSize: "0.75rem", 
                padding: "8px 12px", 
                background: "rgba(255,255,255,0.03)", 
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                color: idx === 0 ? "var(--text-primary)" : "var(--text-dim)"
              }}>
                {idx === 0 ? <Activity size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                {item.step}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="card fade-in" style={{ borderLeft: "4px solid #ef4444", background: "rgba(239, 68, 68, 0.05)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "#f8717b" }}>
            <AlertCircle size={24} />
            <div>
              <h3 style={{ margin: 0, fontSize: "1rem", color: "#f8717b" }}>Analysis Failed</h3>
              <p style={{ margin: 0, fontSize: "0.9rem", color: "rgba(248, 113, 123, 0.8)" }}>{error}</p>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="fade-in">
          <Dashboard result={result} />
          
          <div className="grid-2">
            {result.dataset_hash && (
              <PredictPanel
                datasetHash={result.dataset_hash}
                features={result.features || []}
                onPredict={predictDataset}
              />
            )}
            {result.dataset_hash && (
              <ChatBox datasetHash={result.dataset_hash} onAsk={chatQuery} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
