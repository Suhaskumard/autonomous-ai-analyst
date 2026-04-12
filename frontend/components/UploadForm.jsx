import React, { useState } from "react";
import { Upload, Cpu, Zap, ChevronRight, Loader2 } from "lucide-react";

export default function UploadForm({ onSubmit, isLoading }) {
  const [mode, setMode] = useState("auto");
  const [manualModel, setManualModel] = useState("RandomForest");

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

      <form onSubmit={(e) => onSubmit(e, mode, manualModel)}>
        <div className="grid-2">
          <div className="input-group">
            <label>Dataset File (CSV)</label>
            <div style={{ position: "relative" }}>
              <input 
                type="file" 
                name="file" 
                className="input" 
                required 
                disabled={isLoading}
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
            <label>Training Mode</label>
            <div style={{ position: "relative" }}>
              <select 
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
        
        {mode === "manual" && (
          <div className="input-group fade-in" style={{ marginTop: '8px' }}>
            <label>Select Model Architecture</label>
            <select 
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
          </div>
        )}

        <div style={{ marginTop: '32px', display: 'flex', justifyContent: 'flex-end' }}>
          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={isLoading}
            style={{ width: "100%", sm: { width: "auto" } }}
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
