import { useEffect, useState } from 'react';
import { Clock, FileText, CheckCircle2, AlertCircle } from 'lucide-react';
import './HistoryPage.css';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Mock history data
    setHistory([
      { id: 1, dataset: 'Q3_Financials.csv', date: '2023-10-15', status: 'completed', accuracy: '94.2%', rows: '14,230' },
      { id: 2, dataset: 'Customer_Churn_Log.csv', date: '2023-10-14', status: 'completed', accuracy: '89.1%', rows: '8,400' },
      { id: 3, dataset: 'Server_Metrics_Sep.csv', date: '2023-10-10', status: 'failed', accuracy: '-', rows: '1.2M' },
      { id: 4, dataset: 'Marketing_Campaign_A.csv', date: '2023-10-05', status: 'completed', accuracy: '91.8%', rows: '45,000' },
      { id: 5, dataset: 'test_data.csv', date: 'Just now', status: 'completed', accuracy: '96.4%', rows: '5' }
    ]);
  }, []);

  return (
    <div className="history-page animate-fade-in">
      <header className="page-header-simple">
        <h2><Clock className="header-icon" /> Analysis History</h2>
        <p className="text-muted">Review past datasets and model runs</p>
      </header>

      <div className="glass-panel table-container">
        <table className="history-table">
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Date</th>
              <th>Rows Processed</th>
              <th>Status</th>
              <th>Model Accuracy</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item) => (
              <tr key={item.id}>
                <td className="dataset-cell">
                  <FileText size={16} className="text-muted"/> {item.dataset}
                </td>
                <td>{item.date}</td>
                <td>{item.rows}</td>
                <td>
                  <span className={`status-badge ${item.status}`}>
                    {item.status === 'completed' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                    {item.status}
                  </span>
                </td>
                <td className={item.status === 'completed' ? 'text-success' : 'text-muted'}>
                  {item.accuracy}
                </td>
                <td>
                  <button className="btn-text" disabled={item.status === 'failed'}>View details</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
