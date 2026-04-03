import { BrowserRouter as Router, Routes, Route, NavLink, Link } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import Dashboard from './pages/Dashboard';
import CopilotPage from './pages/CopilotPage';
import HistoryPage from './pages/HistoryPage';
import IntegrationsPage from './pages/IntegrationsPage';
import SettingsPage from './pages/SettingsPage';
import './index.css';

function App() {
  return (
    <Router>
      <nav className="navbar">
        <Link to="/" className="brand-logo">Autono Analyst</Link>
        <div className="nav-links">
            <div className="nav-main-links">
                <NavLink to="/upload" className={({isActive}) => isActive ? "active" : ""}>Analyze</NavLink>
                <NavLink to="/dashboard" className={({isActive}) => isActive ? "active" : ""}>Dashboard</NavLink>
                <NavLink to="/copilot" className={({isActive}) => isActive ? "active" : ""}>AI Copilot</NavLink>
                <NavLink to="/history" className={({isActive}) => isActive ? "active" : ""}>History</NavLink>
                <NavLink to="/integrations" className={({isActive}) => isActive ? "active" : ""}>Integrations</NavLink>
            </div>
            <div className="nav-utils">
                <NavLink to="/settings" className="settings-link"><span className="icon">⚙️</span></NavLink>
            </div>
        </div>
      </nav>
      
      <main className="main-content">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/copilot" element={<CopilotPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/integrations" element={<IntegrationsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
