import React from "react";
import { AlertOctagon, RotateCcw } from "lucide-react";

/**
 * Catches render errors in the dashboard subtree.
 *
 * Without this, one malformed field in a result payload unmounts the whole app
 * and the user sees a blank page with the real cause only in the console.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("Dashboard render failed:", error, info?.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="card fade-in" style={{ borderLeft: "4px solid #f87171" }}>
        <div style={{ display: "flex", gap: "16px", alignItems: "flex-start" }}>
          <div style={{ color: "#f87171", flexShrink: 0, marginTop: "2px" }}>
            <AlertOctagon size={26} />
          </div>
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: 0, fontSize: "1.05rem", color: "#f8717b" }}>
              {this.props.title || "This panel could not be displayed"}
            </h2>
            <p style={{ margin: "8px 0 12px", fontSize: "0.9rem", color: "var(--text-secondary)" }}>
              The analysis itself completed — only this view failed to render. The rest of the page is unaffected.
            </p>
            <pre
              style={{
                margin: "0 0 16px",
                padding: "12px",
                background: "rgba(248, 113, 113, 0.06)",
                borderRadius: "10px",
                fontSize: "0.78rem",
                color: "var(--text-dim)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {String(this.state.error?.message || this.state.error)}
            </pre>
            <button className="btn btn-secondary" type="button" onClick={this.reset}>
              <RotateCcw size={14} />
              Try rendering again
            </button>
          </div>
        </div>
      </div>
    );
  }
}
