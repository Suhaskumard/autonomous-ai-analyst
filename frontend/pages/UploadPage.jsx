import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Loader2,
  MessageSquareText,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

import AccountBar from "../components/AccountBar";
import DataProfile from "../components/DataProfile";
import ErrorBoundary from "../components/ErrorBoundary";
import ThemeToggle from "../components/ThemeToggle";
import useUploadJob from "../hooks/useUploadJob";
import { chatQuery, generateReport, profileDataset } from "../services/api";

// None of these render until a run exists — the upload screen never needs
// them. Splitting them out is what actually shrinks the initial bundle:
// Dashboard and its chart tree pull in recharts, which was most of the single
// chunk Vite had been warning about since Phase 6 added the run registry.
// `lazy` means the browser fetches the chunk the first time result or run
// history renders, not before.
const Dashboard = lazy(() => import("../components/Dashboard"));
const PredictPanel = lazy(() => import("../components/PredictPanel"));
const ChatBox = lazy(() => import("../components/ChatBox"));
const ReportPanel = lazy(() => import("../components/ReportPanel"));
const RunActions = lazy(() => import("../components/RunActions"));
const Workspace = lazy(() => import("../components/Workspace"));

// One fallback for all five: the chunk is usually already warm from a
// previous run in the same session, so this shows for a beat on first load
// and not at all after.
function PanelFallback() {
  return (
    <div className="panel-loading">
      <Loader2 size={16} className="animate-spin" />
      <span>Loading…</span>
    </div>
  );
}

const HOW_IT_WORKS = [
  { icon: Upload, title: "Upload & profile", body: "Drop in a CSV — column types, missing values, and target candidates surface before anything trains." },
  { icon: Cpu, title: "Train & explain", body: "Auto-picks or lets you choose a model, then reports the scores and the features driving them." },
  { icon: MessageSquareText, title: "Ask & export", body: "Query the trained model in plain language, predict on new rows, and export a shareable report." },
];

/**
 * The flow: pick a file → look at what is in it and choose a target → train →
 * dashboard. The profile step exists so the target is an informed choice.
 */
export default function UploadPage() {
  // "local" on a single-operator install, where there is nothing to sign in to.
  const [authStatus, setAuthStatus] = useState("local");
  const [file, setFile] = useState(null);
  const [profile, setProfile] = useState(null);
  const [profiling, setProfiling] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [mode, setMode] = useState("auto");
  const [manualModel, setManualModel] = useState("RandomForest");
  const [tuningBudget, setTuningBudget] = useState(0);
  const [useSmote, setUseSmote] = useState(false);

  const { loading, status, result, error, errorKind, run, cancel, reset, showResult } = useUploadJob();
  const profileControllerRef = useRef(null);

  // A stable reference: an inline arrow here would give AccountBar a new
  // `onStatusChange` on every UploadPage re-render, which — through its own
  // useCallback dependency — reruns its identity-check effect on every
  // render too. That fired a fresh /api/auth/me on every status-poll tick
  // while training, not just once on mount.
  const handleAuthStatusChange = useCallback(({ status }) => setAuthStatus(status), []);

  // Reopening a run from a workspace list scrolled far down otherwise leaves
  // the viewport past the end of the (usually shorter) dashboard that just
  // replaced it, looking blank until the reader scrolls up by hand.
  useEffect(() => {
    if (result?.run_key) window.scrollTo(0, 0);
  }, [result?.run_key]);

  const handleFileChange = useCallback(async (event) => {
    const chosen = event.target.files?.[0];
    setProfile(null);
    setProfileError("");
    setFile(chosen || null);
    if (!chosen) return;

    profileControllerRef.current?.abort();
    const controller = new AbortController();
    profileControllerRef.current = controller;

    setProfiling(true);
    try {
      setProfile(await profileDataset(chosen, { signal: controller.signal }));
    } catch (err) {
      if (err?.name !== "AbortError") setProfileError(err.message || "Could not read that file.");
    } finally {
      if (!controller.signal.aborted) setProfiling(false);
    }
  }, []);

  const startTraining = useCallback(
    (targetColumn) => {
      if (!file) return;
      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", mode);
      if (mode === "manual") formData.append("manual_model", manualModel);
      if (targetColumn) formData.append("target_column", targetColumn);
      // Both are part of the cache key server-side, so changing either
      // retrains rather than replaying the previous artifact.
      formData.append("tuning_budget_seconds", String(tuningBudget));
      formData.append("use_smote", String(useSmote));
      run(formData);
    },
    [file, mode, manualModel, tuningBudget, useSmote, run]
  );

  const clearFile = useCallback(() => {
    profileControllerRef.current?.abort();
    setFile(null);
    setProfile(null);
    setProfileError("");
    reset();
  }, [reset]);

  return (
    <>
      <div className="topbar">
        <div className="topbar-brand">
          <span className="topbar-mark">
            <Sparkles size={17} />
          </span>
          Autonomous AI Analyst
        </div>
        <div className="topbar-actions">
          <AccountBar onStatusChange={handleAuthStatusChange} />
          <ThemeToggle />
        </div>
      </div>

      <div className="app-shell">
      <header className="header fade-in">
        <h1 className="title">AUTONOMOUS AI ANALYST</h1>
        <p className="subtitle">
          Upload a dataset, see exactly what is in it, then let the pipeline train, explain, and score a model you can
          actually check.
        </p>
      </header>

      {authStatus === "signed-out" && (
        // Everything below needs a credential, so showing the upload form here
        // would only produce a 401 the moment the user chose a file.
        <div className="card fade-in">
          <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Sign in to continue</h2>
          <p style={{ margin: "8px 0 0", fontSize: "0.9rem", color: "var(--text-secondary)" }}>
            This server has authentication enabled. Enter the API key you were issued above — it is shown once when the
            account is created and cannot be recovered afterwards. Your datasets, models, and conversations are visible
            only to your account.
          </p>
        </div>
      )}

      {authStatus !== "signed-out" && !profile && !result && (
        <div className="card fade-in">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <div className="icon-chip">
              <Cpu size={22} />
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: "1.2rem" }}>Start an analysis</h2>
              <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-dim)" }}>
                Choose a CSV. Nothing is trained until you pick a target column.
              </p>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="dataset-file">Dataset file (CSV)</label>
            <div style={{ position: "relative" }}>
              <input
                id="dataset-file"
                type="file"
                name="file"
                accept=".csv,.tsv,.txt"
                className="input"
                disabled={profiling}
                onChange={handleFileChange}
                style={{ paddingLeft: 44 }}
              />
              <Upload size={18} className="input-icon" />
            </div>
          </div>

          {profiling && (
            <p style={{ display: "flex", gap: 8, alignItems: "center", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
              <Loader2 size={16} className="animate-spin" />
              Reading {file?.name}…
            </p>
          )}
          {profileError && (
            <div className="notice notice-error" role="alert">
              <AlertCircle size={16} />
              {profileError}
            </div>
          )}
        </div>
      )}

      {authStatus !== "signed-out" && !profile && !result && (
        <div className="how-it-works fade-in">
          {HOW_IT_WORKS.map((step, index) => (
            <div className="how-step" key={step.title}>
              <span className="how-step-number">{String(index + 1).padStart(2, "0")}</span>
              <div className="icon-chip how-step-icon">
                <step.icon size={20} />
              </div>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          ))}
        </div>
      )}

      {profile && !result && (
        <ErrorBoundary title="The data profile could not be displayed">
          <DataProfile
            profile={profile}
            mode={mode}
            manualModel={manualModel}
            tuningBudget={tuningBudget}
            useSmote={useSmote}
            onModeChange={setMode}
            onManualModelChange={setManualModel}
            onTuningBudgetChange={setTuningBudget}
            onUseSmoteChange={setUseSmote}
            onTrain={startTraining}
            onCancel={clearFile}
            isTraining={loading}
          />
        </ErrorBoundary>
      )}

      {loading && (
        <div className="card fade-in" style={{ borderLeft: "4px solid var(--primary)" }}>
          <div className="progress-head">
            <h2 className="panel-head-title" style={{ fontSize: "1.1rem" }}>
              <Activity size={18} style={{ color: "var(--primary)" }} />
              Processing pipeline
            </h2>
            <button type="button" className="btn btn-secondary btn-compact" onClick={cancel}>
              <X size={13} />
              Stop watching
            </button>
          </div>

          <div style={{ marginBottom: 20 }}>
            <div className="progress-labels">
              <span style={{ color: "var(--text-secondary)" }}>{status?.current_step || "Initializing…"}</span>
              <span className="tabular" style={{ fontWeight: 700, color: "var(--primary)" }}>
                {status?.progress ?? 0}%
              </span>
            </div>
            <div className="progress-bar">
              <div className="progress-inner" style={{ width: `${status?.progress ?? 0}%` }} />
            </div>
          </div>

          <div className="step-grid">
            {(status?.status_log || [])
              .slice()
              .reverse()
              .map((item, index) => (
                <div key={index} className={index === 0 ? "step step-current" : "step"}>
                  {index === 0 ? <Activity size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
                  {item.step}
                </div>
              ))}
          </div>
        </div>
      )}

      {error && (
        <div className="card fade-in notice-card-error">
          <div className="banner-row">
            <div className="banner-icon" style={{ color: "#f8717b" }}>
              <AlertCircle size={22} />
            </div>
            <div className="banner-body">
              <h3 className="banner-title" style={{ fontSize: "1rem", color: "#f8717b" }}>{titleFor(errorKind)}</h3>
              <p style={{ margin: "4px 0 0", fontSize: "0.9rem", color: "rgba(248,113,123,0.85)" }}>{error}</p>
              {errorKind === "target" && (
                <p style={{ margin: "6px 0 0", fontSize: "0.85rem", color: "var(--text-dim)" }}>
                  Pick a different target column above and run again.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="fade-in">
          <div className="result-actions">
            <button type="button" className="btn btn-secondary" onClick={clearFile}>
              <Upload size={14} />
              Analyse another dataset
            </button>
          </div>

          <ErrorBoundary title="The results dashboard could not be displayed">
            <Suspense fallback={<PanelFallback />}>
              <Dashboard result={result} />
            </Suspense>
          </ErrorBoundary>

          <div className="grid-2">
            {result.run_key && (
              <ErrorBoundary title="The prediction panel could not be displayed">
                <Suspense fallback={<PanelFallback />}>
                  <PredictPanel
                    runKey={result.run_key}
                    features={result.features || []}
                    problemType={result.problem_type}
                    target={result.target}
                  />
                </Suspense>
              </ErrorBoundary>
            )}
            {result.run_key && (
              <ErrorBoundary title="The analyst chat could not be displayed">
                <Suspense fallback={<PanelFallback />}>
                  <ChatBox runKey={result.run_key} onAsk={chatQuery} />
                </Suspense>
              </ErrorBoundary>
            )}
          </div>

          {result.run_key && (
            <ErrorBoundary title="The report could not be displayed">
              <Suspense fallback={<PanelFallback />}>
                <ReportPanel runKey={result.run_key} onGenerate={generateReport} />
              </Suspense>
            </ErrorBoundary>
          )}

          {result.run_key && (
            <ErrorBoundary title="The share and export panel could not be displayed">
              <Suspense fallback={<PanelFallback />}>
                <RunActions runKey={result.run_key} />
              </Suspense>
            </ErrorBoundary>
          )}
        </div>
      )}

      <ErrorBoundary title="The workspace could not be displayed">
        <Suspense fallback={<PanelFallback />}>
          {/* Workspace mounts (and fetches) immediately on page load, before a
              signed-out visitor has entered a key — authStatus is otherwise
              unused here, but including it means signing in re-triggers the
              fetch instead of leaving the earlier 401 on screen forever. */}
          <Workspace onOpenRun={showResult} activeRunKey={result?.run_key} authStatus={authStatus} />
        </Suspense>
      </ErrorBoundary>
      </div>
    </>
  );
}

function titleFor(kind) {
  switch (kind) {
    case "target":
      return "Unusable target column";
    case "dataset":
      return "Dataset cannot be used";
    case "timeout":
      return "Still running";
    case "network":
      return "Could not reach the API";
    default:
      return "Analysis failed";
  }
}
