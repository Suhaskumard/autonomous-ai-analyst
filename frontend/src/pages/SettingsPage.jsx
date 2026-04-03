git add .import { useState } from 'react';
import { Settings, Key, User, Shield, Eye, EyeOff } from 'lucide-react';
import './SettingsPage.css';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('account');
  const [apiKeyVisible, setApiKeyVisible] = useState(false);

  const stripeKey = process.env.REACT_APP_STRIPE_KEY 

  return (
    <div className="settings-page animate-fade-in">
      <header className="page-header-simple">
        <h2><Settings className="header-icon" /> Preferences & Settings</h2>
        <p className="text-muted">Manage your account, billing, and developer keys.</p>
      </header>

      <div className="settings-layout">
        {/* Sidebar */}
        <div className="settings-sidebar glass-panel">
          <nav className="settings-nav">
            <button 
              className={`settings-nav-item ${activeTab === 'account' ? 'active' : ''}`}
              onClick={() => setActiveTab('account')}
            >
              <User size={18} /> Account Profile
            </button>
            <button 
              className={`settings-nav-item ${activeTab === 'api' ? 'active' : ''}`}
              onClick={() => setActiveTab('api')}
            >
              <Key size={18} /> API Keys
            </button>
            <button 
              className={`settings-nav-item ${activeTab === 'security' ? 'active' : ''}`}
              onClick={() => setActiveTab('security')}
            >
              <Shield size={18} /> Security
            </button>
          </nav>
        </div>

        {/* Content area */}
        <div className="settings-content glass-panel">
          {activeTab === 'account' && (
            <div className="tab-pane animate-fade-in">
              <h3>Account Profile</h3>
              <p className="text-muted mb-4">Manage your personal information.</p>
              
              <div className="form-group">
                <label>Name</label>
                <input type="text" className="form-input" defaultValue="Jane Doe" />
              </div>
              <div className="form-group">
                <label>Email Address</label>
                <input type="email" className="form-input" defaultValue="jane@example.com" />
              </div>
              <div className="form-group">
                <label>Organization</label>
                <input type="text" className="form-input" defaultValue="Acme Corp Data Team" />
              </div>
              
              <button className="btn btn-primary mt-4">Save Changes</button>
            </div>
          )}

          {activeTab === 'api' && (
            <div className="tab-pane animate-fade-in">
              <h3>Developer API Keys</h3>
              <p className="text-muted mb-4">Use these keys to access the Analyst via external applications.</p>
              
              <div className="api-key-container">
                <div className="key-details">
                  <span className="key-name">Production Key</span>
                  <div className="key-value-wrapper">
                    <code className="key-value">
                      {apiKeyVisible ? stripeKey : stripeKey.replace(/./g, '•')}
                    </code>
                    <button className="btn-icon" onClick={() => setApiKeyVisible(!apiKeyVisible)}>
                      {apiKeyVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              </div>
              
              <button className="btn btn-secondary mt-4">Generate New Key</button>
            </div>
          )}

          {activeTab === 'security' && (
            <div className="tab-pane animate-fade-in">
              <h3>Security Settings</h3>
              <p className="text-muted mb-4">Keep your account secure.</p>
              
              <div className="security-option">
                <div>
                  <h4>Two-Factor Authentication</h4>
                  <p className="text-muted text-sm">Add an extra layer of security to your account.</p>
                </div>
                <button className="btn btn-secondary">Enable 2FA</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
