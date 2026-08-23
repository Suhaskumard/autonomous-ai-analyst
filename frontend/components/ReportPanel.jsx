import React, { useCallback, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { FileText, Loader2, Sparkles } from "lucide-react";

/**
 * One button, one full EDA sweep.
 *
 * Every number in the report is computed by the same deterministic tools the
 * analyst uses; the model is only asked to write the prose around results it
 * was handed. With no model configured the report still builds — findings
 * without narrative, which beats narrative without findings.
 */
export default function ReportPanel({ runKey, onGenerate }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const controllerRef = useRef(null);

  const generate = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError("");
    try {
      setReport(await onGenerate(runKey, { signal: controller.signal }));
    } catch (err) {
      if (err?.name !== "AbortError") setError(err?.message || "The report could not be generated.");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, [onGenerate, runKey]);

  return (
    <div className="card fade-in">
      <div className="workspace-head">
        <h2 className="panel-head-title">
          <FileText size={20} style={{ color: "var(--primary)" }} />
          Autonomous report
        </h2>
        <button type="button" className="btn btn-primary" onClick={generate} disabled={loading}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
          {report ? "Regenerate" : "Generate report"}
        </button>
      </div>

      {!report && !loading && !error && (
        <p className="subtle-text">
          Runs a full sweep — profiles, relationships, per-segment consistency, distributions, and the model
          result — and writes it up as one document you can copy out.
        </p>
      )}

      {loading && (
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", display: "flex", gap: 8 }}>
          <Loader2 size={16} className="animate-spin" />
          Profiling columns, computing correlations, and drawing charts…
        </p>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}

      {report && (
        <div className="report">
          <p className="report-meta">
            {report.dataset?.rows?.toLocaleString()} rows × {report.dataset?.columns} columns ·{" "}
            {report.dataset?.problem_type} · target <strong>{report.dataset?.target}</strong>
          </p>

          {report.narrative_error && (
            <div className="notice notice-warning" role="status">
              {report.narrative_error}
            </div>
          )}
          {report.narrative && (
            <div className="markdown-content report-narrative">
              <ReactMarkdown>{report.narrative}</ReactMarkdown>
            </div>
          )}

          {(report.sections || []).map((section) => (
            <section key={section.id} className="report-section">
              <h3>{section.title}</h3>
              {section.summary && <p className="report-summary">{section.summary}</p>}

              {(section.profiles || []).length > 0 && (
                <ul className="report-profiles">
                  {section.profiles.map((profile) => (
                    <li key={profile.column}>
                      <strong>{profile.column}</strong> — {profile.summary}
                    </li>
                  ))}
                </ul>
              )}

              {(section.figures || []).length > 0 && (
                <div className="report-figures">
                  {section.figures.map((figure, index) => (
                    <img
                      key={index}
                      src={`data:${figure.mime || "image/png"};base64,${figure.data}`}
                      alt={figure.caption || `${section.title} chart ${index + 1}`}
                    />
                  ))}
                </div>
              )}
            </section>
          ))}

          {report.markdown && (
            <details className="report-source">
              <summary>Markdown source (copy to share)</summary>
              <pre>
                <code>{report.markdown}</code>
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
