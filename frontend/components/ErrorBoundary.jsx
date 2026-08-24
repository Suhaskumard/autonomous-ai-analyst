import React from "react";
import { AlertOctagon, RotateCcw } from "lucide-react";

// A lazy-loaded chunk failing to fetch (a stale tab open across a deploy that
// rotated chunk hashes, a flaky network) throws a specific, recognizable
// error. Unlike a bad-data render error, resetting local state here cannot
// help: React caches the rejected import() promise on the lazy component
// forever, so re-rendering the same reference throws the identical error
// again, every time — "Try rendering again" would be a button that lies.
// Only a full reload re-fetches index.html and picks up current chunk URLs.
function isChunkLoadError(error) {
  const message = String(error?.message || error || "");
  return /Failed to fetch dynamically imported module|Loading chunk .* failed|ChunkLoadError|error loading dynamically imported module/i.test(
    message
  );
}

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

    const chunkError = isChunkLoadError(this.state.error);

    return (
      <div className="card fade-in" style={{ borderLeft: "4px solid #f87171" }}>
        <div className="banner-row">
          <div className="banner-icon" style={{ color: "#f87171" }}>
            <AlertOctagon size={26} />
          </div>
          <div className="banner-body">
            <h2 className="banner-title" style={{ color: "#f8717b" }}>
              {this.props.title || "This panel could not be displayed"}
            </h2>
            <p className="banner-desc">
              {chunkError
                ? "This part of the app updated since the page was loaded, so this piece failed to fetch. Reloading picks up the current version."
                : "The analysis itself completed — only this view failed to render. The rest of the page is unaffected."}
            </p>
            <pre className="error-boundary-detail">
              {String(this.state.error?.message || this.state.error)}
            </pre>
            {chunkError ? (
              <button className="btn btn-secondary" type="button" onClick={() => window.location.reload()}>
                <RotateCcw size={14} />
                Reload page
              </button>
            ) : (
              <button className="btn btn-secondary" type="button" onClick={this.reset}>
                <RotateCcw size={14} />
                Try rendering again
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }
}
