import { useState } from 'react';
import { Cloud, Database, HardDrive, Loader2, CheckCircle2 } from 'lucide-react';
import './IntegrationsPage.css';

const INTEGRATIONS = [
  { id: 'aws', name: 'AWS S3', icon: <Cloud size={32} />, category: 'Cloud Storage', status: 'connected' },
  { id: 'snowflake', name: 'Snowflake', icon: <Database size={32} />, category: 'Data Warehouse', status: 'disconnected' },
  { id: 'postgres', name: 'PostgreSQL', icon: <HardDrive size={32} />, category: 'Relational DB', status: 'disconnected' },
  { id: 'bigquery', name: 'Google BigQuery', icon: <Database size={32} />, category: 'Data Warehouse', status: 'disconnected' }
];

export default function IntegrationsPage() {
  const [integrations, setIntegrations] = useState(INTEGRATIONS);
  const [loadingId, setLoadingId] = useState(null);

  const handleConnect = (id) => {
    setLoadingId(id);
    setTimeout(() => {
      setIntegrations(prev => 
        prev.map(init => init.id === id ? { ...init, status: 'connected' } : init)
      );
      setLoadingId(null);
    }, 1500);
  };

  const handleDisconnect = (id) => {
    setIntegrations(prev => 
      prev.map(init => init.id === id ? { ...init, status: 'disconnected' } : init)
    );
  };

  return (
    <div className="integrations-page animate-fade-in">
      <header className="page-header-simple">
        <h2><Cloud className="header-icon" /> Data Integrations</h2>
        <p className="text-muted">Connect your autonomous analyst directly to your data pipelines.</p>
      </header>

      <div className="integrations-grid">
        {integrations.map((integration) => (
          <div key={integration.id} className={`integration-card glass-panel ${integration.status === 'connected' ? 'active' : ''}`}>
            <div className="integration-header">
              <div className={`integration-icon ${integration.id}`}>
                {integration.icon}
              </div>
              <div className="integration-status">
                {integration.status === 'connected' ? (
                  <span className="status-badge completed"><CheckCircle2 size={12}/> Connected</span>
                ) : (
                  <span className="status-badge disconnected">Disconnected</span>
                )}
              </div>
            </div>
            
            <div className="integration-body">
              <h3>{integration.name}</h3>
              <p className="text-muted">{integration.category}</p>
            </div>

            <div className="integration-footer">
              {integration.status === 'connected' ? (
                <button className="btn btn-secondary w-full" onClick={() => handleDisconnect(integration.id)}>
                  Configure
                </button>
              ) : (
                <button 
                  className="btn btn-primary w-full" 
                  onClick={() => handleConnect(integration.id)}
                  disabled={loadingId === integration.id}
                >
                  {loadingId === integration.id ? <Loader2 className="spinner" size={18} /> : 'Connect'}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
