import { useNavigate } from 'react-router-dom';
import { Database, Zap, Sparkles, MoveRight } from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page animate-fade-in">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-badge">
          <Sparkles size={16} className="text-accent" />
          <span>Next-Generation Data Intelligence v2.0</span>
        </div>
        <h1 className="hero-title">
          Talk to your data.<br />
          <span className="gradient-text">Understand the future.</span>
        </h1>
        <p className="hero-subtitle">
          Upload any structured dataset and instantly unlock predictive models, advanced visualization, and an autonomous AI ready to answer your most difficult questions.
        </p>
        <div className="hero-actions">
          <button className="btn btn-primary btn-large" onClick={() => navigate('/upload')}>
            Start Analyzing <MoveRight size={20} />
          </button>
          <button className="btn btn-secondary btn-large" onClick={() => navigate('/integrations')}>
            View Integrations
          </button>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section glass-panel">
        <div className="stat-item">
          <h3>2.5x</h3>
          <p>Faster Time to Insight</p>
        </div>
        <div className="divider"></div>
        <div className="stat-item">
          <h3>99.9%</h3>
          <p>Prediction Accuracy Potential</p>
        </div>
        <div className="divider"></div>
        <div className="stat-item">
          <h3>100+</h3>
          <p>Supported Formats & Sources</p>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="section-title">Everything you need to scale</h2>
        <div className="features-grid">
          <div className="feature-card glass-panel">
            <div className="feature-icon"><Database size={24} /></div>
            <h3>Seamless Ingestion</h3>
            <p>Drag and drop CSVs or connect directly to AWS, Snowflake, or Postgres pipelines.</p>
          </div>
          <div className="feature-card glass-panel">
            <div className="feature-icon"><Zap size={24} /></div>
            <h3>Automated ML</h3>
            <p>Our autonomous engine profiles data, selects the best model, and optimizes parameters instantly.</p>
          </div>
          <div className="feature-card glass-panel">
            <div className="feature-icon"><Sparkles size={24} /></div>
            <h3>Conversational BI</h3>
            <p>Stop writing complex SQL queries. Ask questions in plain English and let the AI give you actual numbers.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
