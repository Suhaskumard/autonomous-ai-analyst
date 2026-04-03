import { useState, useRef } from 'react';
import { UploadCloud, File, X, Loader2 } from 'lucide-react';
import { uploadFile } from '../services/api';
import './FileUploader.css';

export default function FileUploader({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setError(null);
    if (!selectedFile.name.endsWith('.csv')) {
      setError("Please upload a valid CSV file.");
      return;
    }
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      const response = await uploadFile(file);
      onUploadSuccess(response);
    } catch (err) {
      setError("Failed to upload file. Please ensure backend is running.");
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="file-uploader glass-panel animate-fade-in">
      <div 
        className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
      >
        <input 
          ref={inputRef} 
          type="file" 
          accept=".csv" 
          onChange={handleChange} 
          style={{ display: "none" }} 
        />
        
        {!file ? (
          <div className="drop-content">
            <UploadCloud size={48} className="upload-icon" />
            <p>Drag & Drop your dataset here</p>
            <span className="text-muted">or click to browse (.csv only)</span>
          </div>
        ) : (
          <div className="file-info" onClick={(e) => e.stopPropagation()}>
            <File size={32} className="file-icon" />
            <div className="file-details">
              <span className="file-name">{file.name}</span>
              <span className="file-size">{(file.size / 1024).toFixed(2)} KB</span>
            </div>
            <button className="remove-btn" onClick={() => setFile(null)}>
              <X size={20} />
            </button>
          </div>
        )}
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="uploader-actions">
        <button 
          className="btn upload-btn" 
          onClick={handleUpload} 
          disabled={!file || isUploading}
        >
          {isUploading ? (
            <><Loader2 className="spinner" size={18} /> Uploading...</>
          ) : (
            'Analyze Dataset'
          )}
        </button>
      </div>
    </div>
  );
}
