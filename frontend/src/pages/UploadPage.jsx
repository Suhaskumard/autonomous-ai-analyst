import { useNavigate } from 'react-router-dom';
import FileUploader from '../components/FileUploader';
import './UploadPage.css';

export default function UploadPage() {
  const navigate = useNavigate();

  const handleUploadSuccess = (response) => {
    // Store datasetRef in localStorage to persist across navigation
    localStorage.setItem('dataset_ref', response.dataset_ref);
    navigate('/dashboard');
  };

  return (
    <div className="upload-page">
      <div className="page-header animate-fade-in">
        <h2>Data intelligence, simplified.</h2>
        <p>Upload your dataset and let AI uncover the insights that matter.</p>
      </div>
      
      <FileUploader onUploadSuccess={handleUploadSuccess} />
    </div>
  );
}
