import React, { Suspense, lazy, useEffect, useState } from "react";
import { Eye, Loader2 } from "lucide-react";

import ErrorBoundary from "../components/ErrorBoundary";
import ThemeToggle from "../components/ThemeToggle";
import { getSharedRun } from "../services/api";

const Dashboard = lazy(() => import("../components/Dashboard"));

/**
 * What someone handed a link sees — Phase 10.
 *
 * The dashboard and nothing else: no upload control, no chat, no prediction
 * panel, no workspace. That is not a UI decision the server trusts — `GET
 * /api/share/{token}` returns only the stored result payload and every other
 * endpoint still requires a principal — but the page should not offer
 * controls that would 401 or 403 if pressed.
 */
export default function SharedRunPage({ token }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getSharedRun(token)
      .then((body) => {
        if (!cancelled) setResult(body);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || "This link is invalid or has expired.");
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <>
      <div className="topbar">
        <div className="topbar-brand">
          <span className="topbar-mark">
            <Eye size={17} />
          </span>
          Autonomous AI Analyst
        </div>
        <div className="topbar-actions">
          <ThemeToggle />
        </div>
      </div>

      <div className="app-shell">
      <header className="header fade-in">
        <h1 className="title" style={{ fontSize: "clamp(1.8rem, 4vw, 2.6rem)", display: "flex", alignItems: "center", justifyContent: "center", gap: 12 }}>
          <Eye size={30} style={{ color: "var(--primary)" }} />
          Shared analysis
        </h1>
        <p className="subtitle">
          A read-only view of one run. It expires, and whoever shared it can revoke it at any time.
        </p>
      </header>

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}

      {!result && !error && (
        <p className="panel-loading">
          <Loader2 size={16} className="animate-spin" />
          Loading the shared run…
        </p>
      )}

      {result && (
        <ErrorBoundary title="The shared dashboard could not be displayed">
          <Suspense
            fallback={
              <p className="panel-loading">
                <Loader2 size={16} className="animate-spin" />
                Loading…
              </p>
            }
          >
            <Dashboard result={result} />
          </Suspense>
        </ErrorBoundary>
      )}
      </div>
    </>
  );
}
