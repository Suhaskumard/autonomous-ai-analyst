import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Bot, Loader2, RefreshCw, Send, Sparkles, User } from "lucide-react";

import AnalystSteps from "./AnalystSteps";

const SUGGESTIONS = [
  "Which two features correlate most strongly, and does that hold within each segment?",
  "Which column has the most missing values?",
  "Show the distribution of the target",
];

/**
 * The analyst conversation.
 *
 * History lives on the server: this posts a question and a conversation id,
 * not a transcript. Before, the client sent back its own idea of what had been
 * said, so anything it dropped or rewrote became what the model believed.
 */
export default function ChatBox({ runKey, onAsk }) {
  const [query, setQuery] = useState("");
  const [meta, setMeta] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);
  const controllerRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, loading]);

  // Abort an in-flight question if the panel goes away.
  useEffect(() => () => controllerRef.current?.abort(), []);

  const ask = useCallback(
    async (text) => {
      const question = (text ?? query).trim();
      if (!question || loading) return;

      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      setHistory((current) => [...current, { role: "user", text: question }]);
      setQuery("");
      setError("");
      setLoading(true);

      try {
        const res = await onAsk(
          { run_key: runKey, query: question, conversation_id: conversationId },
          { signal: controller.signal }
        );
        setMeta(res.response);
        setConversationId(res.conversation_id || null);
        setHistory((current) => [
          ...current,
          { role: "ai", text: res.response.answer, steps: res.response.steps || [] },
        ]);
      } catch (err) {
        if (err?.name === "AbortError") return;
        setError(err?.message || "The analyst could not be reached.");
        setHistory((current) => current.slice(0, -1));
        setQuery(question);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [conversationId, loading, onAsk, query, runKey]
  );

  const startFresh = () => {
    controllerRef.current?.abort();
    setConversationId(null);
    setHistory([]);
    setMeta(null);
    setError("");
    setLoading(false);
  };

  return (
    <div className="card fade-in" style={{ padding: 0, overflow: "hidden" }}>
      <div className="chat-head">
        <h3 style={{ margin: 0, display: "flex", alignItems: "center", gap: "12px" }}>
          <Sparkles size={20} style={{ color: "var(--primary)" }} />
          Advanced AI Analyst
        </h3>
        <div className="chat-head-meta">
          {/* Model and execution location come from the backend, so the UI
              cannot advertise a capability that is not actually in play. */}
          <span>{meta?.llm_model || "Not yet asked"}</span>
          {meta?.data_access === "local-execution" && <span>Runs code locally on your rows</span>}
          {meta?.llm_enabled === false && <span>Disabled (no API key)</span>}
          {history.length > 0 && (
            <button type="button" className="chat-reset" onClick={startFresh}>
              <RefreshCw size={12} />
              New conversation
            </button>
          )}
        </div>
      </div>

      <div className="chat-window" style={{ border: "none", borderRadius: 0 }}>
        <div className="chat-messages">
          {history.length === 0 && (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <Bot size={32} />
              </div>
              <div>
                <p style={{ fontSize: "1.05rem", color: "var(--text-secondary)", marginBottom: 8 }}>
                  Ask me about this dataset.
                </p>
                <p style={{ maxWidth: 460, lineHeight: 1.6, margin: 0 }}>
                  I run pandas against your rows on this machine and show you the code and results behind every
                  number, so you can check them yourself.
                </p>
              </div>
              <div className="chat-suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button key={suggestion} type="button" onClick={() => ask(suggestion)} disabled={loading}>
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {history.map((message, index) => (
            <div key={index} className={`message ${message.role === "ai" ? "ai" : "user"}`}>
              <div style={{ display: "flex", gap: 12 }}>
                <div className={message.role === "ai" ? "chat-avatar chat-avatar-ai" : "chat-avatar"}>
                  {message.role === "ai" ? <Bot size={18} /> : <User size={18} />}
                </div>
                <div style={{ width: "100%", minWidth: 0 }}>
                  <AnalystSteps steps={message.steps} />
                  <div className="markdown-content" style={{ overflowX: "auto" }}>
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="message ai">
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <div className="chat-avatar chat-avatar-ai">
                  <Loader2 size={18} className="animate-spin" />
                </div>
                <span style={{ color: "var(--text-dim)", fontSize: "0.9rem" }}>
                  Choosing tools and running them against your rows…
                </span>
              </div>
            </div>
          )}

          {error && (
            <div className="notice notice-error" role="alert">
              {error}
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="chat-input-area">
          <input
            className="input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && ask()}
            placeholder="Ask a question about this dataset"
            aria-label="Ask the analyst"
            disabled={loading}
          />
          <button
            className="btn btn-primary"
            onClick={() => ask()}
            disabled={loading || !query.trim()}
            aria-label="Send"
            style={{ width: 48, height: 48, padding: 0, borderRadius: 12 }}
          >
            {loading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
}
