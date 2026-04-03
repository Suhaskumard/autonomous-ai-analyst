import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react';
import './CopilotPage.css';

export default function CopilotPage() {
  const [messages, setMessages] = useState([
    { id: 1, type: 'bot', text: 'Hello! I am your Autonomous AI Analyst. How can I help you with your data today?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSuggest = (text) => {
    setInput(text);
  };

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMessage = { id: Date.now(), type: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Mock API call to backend chat route (which can be added later)
    setTimeout(() => {
      const responses = [
        "Based on the analysis, I recommend focusing on improving variable X.",
        "That's an interesting question. The data suggests a correlation, but more samples might be needed.",
        "I can generate a more detailed breakdown if you'd like. The current F1 score is strong.",
        "The model indicates that the peak anomalies occurred during the third quarter."
      ];
      const botResponse = { 
        id: Date.now() + 1, 
        type: 'bot', 
        text: responses[Math.floor(Math.random() * responses.length)]
      };
      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="chat-page animate-fade-in">
      <div className="chat-container glass-panel">
        <div className="chat-header">
          <Bot size={24} className="bot-icon" />
          <div>
            <h3>AI Copilot</h3>
            <span className="status-online">Autonomous Mode Active</span>
          </div>
        </div>
        
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.type}`}>
              <div className="message-bubble">
                {msg.type === 'bot' && <Bot size={16} className="msg-icon bot" />}
                {msg.type === 'user' && <User size={16} className="msg-icon user" />}
                <p>{msg.text}</p>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="message-wrapper bot">
              <div className="message-bubble typing">
                <Loader2 size={16} className="spinner" />
                <span>AI is analyzing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="suggested-questions">
          <button onClick={() => handleSuggest("Explain my model performance")}>Explain my model performance</button>
          <button onClick={() => handleSuggest("What should I improve?")}>What should I improve?</button>
          <button onClick={() => handleSuggest("Why is accuracy low?")}>Why is accuracy low?</button>
        </div>

        <form className="chat-input-form" onSubmit={handleSend}>
          <input 
            type="text" 
            placeholder="Ask Copilot a question about your data..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isTyping}
          />
          <button type="submit" disabled={!input.trim() || isTyping} className="send-btn">
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
