import React, { useState } from "react";
import { Upload, Cpu, Zap, ChevronRight, Loader2, Crosshair } from "lucide-react";

const DELIMITERS = [",", "\t", ";", "|"];

// Read just enough of the file to show a target-column picker. The header is
// parsed in the browser so choosing a target needs no extra round trip.
function parseHeader(text) {
  const firstLine = text.split(/\r?\n/).find((line) => line.trim().length > 0);
  if (!firstLine) return [];
  const delimiter = DELIMITERS.reduce(
    (best, d) => (firstLine.split(d).length > firstLine.split(best).length ? d : best),
    ","
  );
  return firstLine
    .split(delimiter)
    .map((c) => c.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

export default function UploadForm({ onSubmit, isLoading }) {
  const [mode, setMode] = useState("auto");
  const [manualModel, setManualModel] = useState("RandomForest");
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [headerError, setHeaderError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    setColumns([]);
    setTargetColumn("");
    setHeaderError("");
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseHeader(String(reader.result || ""));
      if (parsed.length < 2) {
        setHeaderError("Could not read column names from this file's first line.");
        return;
      }
      setColumns(parsed);
    };
    reader.onerror = () => setHeaderError("Could not read the selected file.");
    reader.readAsText(file.slice(0, 65536));
  };

  return (
    <div className="card fade-in">
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "24px" }}>
        <div style={{
          padding: "10px",
          borderRadius: "12px",
          background: "rgba(99, 102, 241, 0.1)",
          color: "var(--primary)"
        }}>
          <Cpu size={24} />
        </div>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.25rem" }}>Initialize Analysis</h2>
          <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-dim)" }}>
            Configure your AI pipeline parameters
          </p>
        </div>
      </div>

      <form onSubmit={(e) => onSubmit(e, mode, manualModel, targetColumn)}>
        <div className="grid-2">
          <div className="input-group">
            <label htmlFor="dataset-file">Dataset File (CSV)</label>
            <div style={{ position: "relative" }}>
              <input
                id="dataset-file"
                type="file"
                name="file"
                accept=".csv,.tsv,.txt"
                className="input"
                required
                disabled={isLoading}
                onChange={handleFileChange}
                style={{ paddingLeft: "44px" }}
              />
              <Upload
                size={18}
                style={{
                  position: "absolute",
                  left: "16px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--text-dim)"
                }}
              />
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="training-mode">Training Mode</label>
            <div style={{ position: "relative" }}>
              <select
                id="training-mode"
                className="input"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                disabled={isLoading}
                style={{ paddingLeft: "44px", appearance: "none" }}
              >
                <option value="auto">Auto-Pilot (Fastest)</option>
                <option value="ensemble">Ensemble (Best Accuracy)</option>
                <option value="manual">Manual Selection</option>
              </select>
              <Zap
                size={18}
                style={{
                  position: "absolute",
                  left: "16px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--primary)"
                }}
              />
            </div>
          </div>
        </div>

        <div className="input-group" style={{ marginTop: "8px" }}>
          <label htmlFor="target-column">Target Column (what the model should predict)</label>
          <div style={{ position: "relative" }}>
            <select
              id="target-column"
              className="input"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              disabled={isLoading || columns.length === 0}
              style={{ paddingLeft: "44px", appearance: "none" }}
            >
              <option value="">
                {columns.length ? "Auto-detect (last column)" : "Choose a file first"}
              </option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <Crosshair
              size={18}
              style={{
                position: "absolute",
                left: "16px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--primary)"
              }}
            />
          </div>
          <p style={{ margin: "8px 0 0", fontSize: "0.78rem", color: "var(--text-dim)" }}>
            {headerError
              ? headerError
              : "Auto-detect takes the last column, which is often wrong. Pick the real target when you know it."}
          </p>
        </div>

        {mode === "manual" && (
          <div className="input-group fade-in" style={{ marginTop: '8px' }}>
            <label htmlFor="manual-model">Select Model Architecture</label>
            <select
              id="manual-model"
              className="input"
              value={manualModel}
              onChange={(e) => setManualModel(e.target.value)}
              disabled={isLoading}
            >
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="RandomForest">Random Forest</option>
              <option value="XGBoost">XGBoost</option>
              <option value="DecisionTree">Decision Tree</option>
              <option value="SVM">SVM</option>
            </select>
            <p style={{ margin: "8px 0 0", fontSize: "0.78rem", color: "var(--text-dim)" }}>
              Regression datasets use the matching regressor automatically.
            </p>
          </div>
        )}

        <div style={{ marginTop: '32px', display: 'flex', justifyContent: 'flex-end' }}>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading}
            style={{ width: "100%" }}
          >
            {isLoading ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                Initializing Pipeline...
              </>
            ) : (
              <>
                Start Autonomous Pipeline
                <ChevronRight size={20} />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
