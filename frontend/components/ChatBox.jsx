import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Bot, User, Sparkles, Loader2 } from "lucide-react";

export default function ChatBox({ datasetHash, onAsk }) {
  const [query, setQuery] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history]);

  const ask = async () => {
    if (!query.trim() || loading) return;
    
    const userMsg = { role: "user", text: query };
    const newHistory = [...history, userMsg];
    setHistory(newHistory);
    setQuery("");
    setLoading(true);

    try {
      // Pass history to backend for multi-turn support
      const apiHistory = history.map(h => ({
        role: h.role === "ai" ? "assistant" : "user",
        content: h.text
      }));

      const res = await onAsk({ 
        dataset_hash: datasetHash, 
        query,
        history: apiHistory
      });

      const aiMsg = { role: "ai", text: res.response.answer };
      setHistory([...newHistory, aiMsg]);
    } catch (err) {
      setHistory(prev => [...prev, { role: "ai", text: "I apologize, but I encountered an error while processing your request. Please try again or re-upload the dataset." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card fade-in" style={{ padding: "0", overflow: "hidden" }}>
      <div style={{ 
        padding: "20px 32px", 
        borderBottom: "1px solid var(--border-glass)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(255, 255, 255, 0.02)"
      }}>
        <h3 style={{ margin: 0, display: "flex", alignItems: "center", gap: "12px" }}>
          <Sparkles size={20} className="text-primary" style={{ color: "var(--primary)" }} />
          Advanced AI Analyst
        </h3>
        <span style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>
          Gemini 1.5 Flash • Code Execution Enabled
        </span>
      </div>

      <div className="chat-window" style={{ border: "none", borderRadius: "0" }}>
        <div className="chat-messages">
          {history.length === 0 && (
            <div style={{ 
              textAlign: "center", 
              marginTop: "60px", 
              color: "var(--text-dim)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "20px"
            }}>
              <div style={{ 
                width: "64px", 
                height: "64px", 
                borderRadius: "20px", 
                background: "rgba(255,255,255,0.03)", 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center",
                border: "1px solid var(--border-glass)"
              }}>
                <Bot size={32} />
              </div>
              <div>
                <p style={{ fontSize: "1.1rem", color: "var(--text-secondary)", marginBottom: "8px" }}>
                  Hello! I'm your Autonomous AI Analyst.
                </p>
                <p style={{ maxWidth: "400px", lineHeight: "1.6" }}>
                  I can write Python code to analyze your data, create visualizations, and uncover deep patterns.
                </p>
              </div>
            </div>
          )}
          
          {history.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role === "ai" ? "ai" : "user"}`}>
              <div style={{ display: "flex", gap: "12px" }}>
                <div style={{ 
                  flexShrink: 0, 
                  width: "32px", 
                  height: "32px", 
                  borderRadius: "8px", 
                  background: msg.role === "ai" ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  {msg.role === "ai" ? <Bot size={18} /> : <User size={18} />}
                </div>
                <div className="markdown-content" style={{ overflowX: "auto", width: "100%" }}>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message ai">
              <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                <div style={{ 
                  flexShrink: 0, 
                  width: "32px", 
                  height: "32px", 
                  borderRadius: "8px", 
                  background: "rgba(255,255,255,0.05)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  <Loader2 size={18} className="animate-spin" />
                </div>
                <span style={{ color: "var(--text-dim)", fontSize: "0.9rem" }}>Analyst is writing code and processing results...</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-area">
          <input
            className="input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="Ask me to analyze something (e.g., 'Find outliers in the target column')"
            disabled={loading}
          />
          <button 
            className="btn btn-primary" 
            onClick={ask} 
            disabled={loading || !query.trim()}
            style={{ width: "48px", height: "48px", padding: "0", borderRadius: "12px" }}
          >
            {loading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
}
