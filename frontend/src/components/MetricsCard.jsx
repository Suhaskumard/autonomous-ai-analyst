import { Activity } from 'lucide-react';
import './MetricsCard.css';

export default function MetricsCard({ title, value, icon, percentage, isPositive }) {
  return (
    <div className="metrics-card glass-panel">
      <div className="metrics-header">
        <span className="metrics-title">{title}</span>
        <div className="metrics-icon-wrapper">
          {icon || <Activity size={18} />}
        </div>
      </div>
      <div className="metrics-content">
        <div className="metrics-value">{value}</div>
        {percentage && (
          <div className={`metrics-percentage ${isPositive ? 'positive' : 'negative'}`}>
            {isPositive ? '+' : '-'}{percentage}%
          </div>
        )}
      </div>
    </div>
  );
}
