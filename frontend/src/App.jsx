import { useState, useEffect, useRef } from 'react';
import {
  Upload,
  FileText,
  Car,
  Brain,
  Send,
  X,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  User,
  DollarSign,
  Loader2,
  Sparkles,
  FileSearch,
} from 'lucide-react';
import { checkHealth, analyzeDocument, askQuestion } from './api';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [question, setQuestion] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [expandedCards, setExpandedCards] = useState({
    summary: true,
    buyer: true,
    vehicle: true,
    financial: true,
    validation: true,
    ocr: false,
  });
  
  const fileInputRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await checkHealth();
        setIsHealthy(true);
      } catch (err) {
        setIsHealthy(false);
      }
    };
    
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile);
      setResult(null);
      setError(null);
      setChatMessages([]);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
      setChatMessages([]);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setChatMessages([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setIsLoading(true);
    setError(null);
    setLoadingMessage('Initializing OCR engine...');
    
    try {
      // Simulate progress messages
      const messages = [
        'Loading DocFormer model...',
        'Processing PDF pages...',
        'Extracting text and layout...',
        'Running field extraction...',
        'Loading Qwen2.5 model...',
        'Analyzing contract details...',
        'Validating extracted data...',
        'Generating insights...',
      ];
      
      let messageIndex = 0;
      const messageInterval = setInterval(() => {
        if (messageIndex < messages.length) {
          setLoadingMessage(messages[messageIndex]);
          messageIndex++;
        }
      }, 3000);
      
      const data = await analyzeDocument(file);
      
      clearInterval(messageInterval);
      setResult(data);
      setIsLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
      setIsLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!file || !question.trim()) return;
    
    const userQuestion = question.trim();
    setQuestion('');
    setChatMessages((prev) => [...prev, { role: 'user', content: userQuestion }]);
    
    try {
      const response = await askQuestion(file, userQuestion);
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: response.answer },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I could not process your question. Please try again.' },
      ]);
    }
  };

  const toggleCard = (cardName) => {
    setExpandedCards((prev) => ({
      ...prev,
      [cardName]: !prev[cardName],
    }));
  };

  const formatCurrency = (value) => {
    if (!value) return 'N/A';
    const num = parseFloat(String(value).replace(/[^0-9.-]/g, ''));
    if (isNaN(num)) return value;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(num);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-title">
          <Car size={28} />
          <h1>Car Deal Document Intelligence</h1>
          <div className="header-badge">
            <span className="badge badge-blue">DocFormer OCR</span>
            <span className="badge badge-green">Qwen2.5 AI</span>
          </div>
        </div>
        <div className="health-indicator">
          <span className={`health-dot ${isHealthy ? '' : 'offline'}`}></span>
          {isHealthy ? 'Backend Connected' : 'Backend Offline'}
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Sidebar */}
        <aside className="sidebar">
          {/* Upload Section */}
          <section className="upload-section">
            <h2 className="section-title">
              <Upload size={18} />
              Upload Document
            </h2>
            
            <div
              className={`drop-zone ${isDragging ? 'dragover' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <FileSearch size={40} className="drop-zone-icon" />
              <p>
                Drop your PDF here or <span>browse</span>
              </p>
              <small>Supports car dealership contracts</small>
            </div>
            
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf"
              hidden
            />
            
            {file && (
              <div className="file-preview">
                <div className="file-info">
                  <FileText size={20} className="file-icon" />
                  <div>
                    <div className="file-name">{file.name}</div>
                    <div className="file-size">{formatFileSize(file.size)}</div>
                  </div>
                </div>
                <button className="remove-file" onClick={handleRemoveFile}>
                  <X size={18} />
                </button>
              </div>
            )}
            
            <button
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={!file || isLoading || !isHealthy}
            >
              {isLoading ? (
                <>
                  <Loader2 size={18} className="spinner-icon" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Analyze Document
                </>
              )}
            </button>
          </section>

          {/* Question Section */}
          <section className="question-section">
            <h2 className="section-title">
              <Brain size={18} />
              Ask Questions
            </h2>
            
            <div className="question-input-wrapper">
              <input
                type="text"
                className="question-input"
                placeholder="Ask about the contract..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                disabled={!file || !isHealthy}
              />
              <button
                className="btn-icon"
                onClick={handleAskQuestion}
                disabled={!file || !question.trim() || !isHealthy}
              >
                <Send size={18} />
              </button>
            </div>
            
            <div className="chat-messages">
              {chatMessages.map((msg, index) => (
                <div key={index} className={`chat-message ${msg.role}`}>
                  {msg.content}
                </div>
              ))}
            </div>
          </section>
        </aside>

        {/* Results Panel */}
        <section className="results-panel">
          {!result && !isLoading && !error && (
            <div className="results-placeholder">
              <FileText size={80} className="results-placeholder-icon" />
              <h3>No Document Analyzed</h3>
              <p>
                Upload a car dealership contract PDF and click "Analyze Document"
                to extract buyer information, vehicle details, financial data,
                and get AI-powered insights.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <h3>Processing Document</h3>
              <p>{loadingMessage}</p>
            </div>
          )}

          {error && (
            <div className="error-state">
              <XCircle size={48} />
              <h3>Analysis Failed</h3>
              <p>{error}</p>
              <button className="btn btn-secondary" onClick={handleAnalyze}>
                Retry
              </button>
            </div>
          )}

          {result && (
            <div className="results-grid">
              {/* Summary Card */}
              <div className="result-card">
                <div
                  className="result-card-header"
                  onClick={() => toggleCard('summary')}
                >
                  <div className="result-card-title">
                    <Sparkles size={20} />
                    Analysis Summary
                  </div>
                  {expandedCards.summary ? (
                    <ChevronUp size={20} />
                  ) : (
                    <ChevronDown size={20} />
                  )}
                </div>
                {expandedCards.summary && (
                  <div className="result-card-content">
                    <div className="summary-stats">
                      <div className="stat-item">
                        <div className="stat-value">{result.page_count || 1}</div>
                        <div className="stat-label">Pages</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-value">
                          {Object.keys(result.extracted_fields || {}).length}
                        </div>
                        <div className="stat-label">Fields Extracted</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-value">
                          {result.summary?.warnings?.length || 0}
                        </div>
                        <div className="stat-label">Warnings</div>
                      </div>
                    </div>
                    
                    {result.analysis?.insights && (
                      <div style={{ marginTop: '1rem' }}>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                          {result.analysis.insights}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Buyer Information */}
              {result.extracted_fields?.buyer_info && (
                <div className="result-card">
                  <div
                    className="result-card-header"
                    onClick={() => toggleCard('buyer')}
                  >
                    <div className="result-card-title">
                      <User size={20} />
                      Buyer Information
                    </div>
                    {expandedCards.buyer ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                  {expandedCards.buyer && (
                    <div className="result-card-content">
                      {Object.entries(result.extracted_fields.buyer_info).map(
                        ([key, value]) => (
                          <div className="field-row" key={key}>
                            <span className="field-label">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                            </span>
                            <span className="field-value">{value || 'N/A'}</span>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Vehicle Details */}
              {result.extracted_fields?.vehicle_info && (
                <div className="result-card">
                  <div
                    className="result-card-header"
                    onClick={() => toggleCard('vehicle')}
                  >
                    <div className="result-card-title">
                      <Car size={20} />
                      Vehicle Details
                    </div>
                    {expandedCards.vehicle ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                  {expandedCards.vehicle && (
                    <div className="result-card-content">
                      {Object.entries(result.extracted_fields.vehicle_info).map(
                        ([key, value]) => (
                          <div className="field-row" key={key}>
                            <span className="field-label">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                            </span>
                            <span className={`field-value ${key === 'vin' ? 'highlight' : ''}`}>
                              {value || 'N/A'}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Financial Details */}
              {result.extracted_fields?.financial_info && (
                <div className="result-card">
                  <div
                    className="result-card-header"
                    onClick={() => toggleCard('financial')}
                  >
                    <div className="result-card-title">
                      <DollarSign size={20} />
                      Financial Details
                    </div>
                    {expandedCards.financial ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                  {expandedCards.financial && (
                    <div className="result-card-content">
                      {Object.entries(result.extracted_fields.financial_info).map(
                        ([key, value]) => (
                          <div className="field-row" key={key}>
                            <span className="field-label">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                            </span>
                            <span className="field-value highlight">
                              {key.toLowerCase().includes('price') ||
                              key.toLowerCase().includes('amount') ||
                              key.toLowerCase().includes('payment') ||
                              key.toLowerCase().includes('total')
                                ? formatCurrency(value)
                                : value || 'N/A'}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Validation Results */}
              {result.analysis?.validation && (
                <div className="result-card">
                  <div
                    className="result-card-header"
                    onClick={() => toggleCard('validation')}
                  >
                    <div className="result-card-title">
                      <CheckCircle size={20} />
                      Validation Results
                    </div>
                    {expandedCards.validation ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                  {expandedCards.validation && (
                    <div className="result-card-content">
                      <div className="validation-results">
                        {Object.entries(result.analysis.validation).map(
                          ([field, data]) => (
                            <div className="validation-item" key={field}>
                              {data.valid ? (
                                <CheckCircle
                                  size={18}
                                  className="validation-icon valid"
                                />
                              ) : (
                                <XCircle
                                  size={18}
                                  className="validation-icon invalid"
                                />
                              )}
                              <span className="validation-field">
                                {field.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                              </span>
                              {data.message && (
                                <span className="validation-message">
                                  {data.message}
                                </span>
                              )}
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Warnings */}
              {result.summary?.warnings?.length > 0 && (
                <div className="result-card">
                  <div className="result-card-header">
                    <div className="result-card-title">
                      <AlertTriangle size={20} />
                      Warnings & Recommendations
                    </div>
                  </div>
                  <div className="result-card-content">
                    <div className="warnings-list">
                      {result.summary.warnings.map((warning, index) => (
                        <div
                          className={`warning-item ${
                            warning.severity === 'error' ? 'error' : ''
                          }`}
                          key={index}
                        >
                          <AlertTriangle size={18} className="warning-icon" />
                          <span className="warning-text">
                            {warning.message || warning}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* OCR Text */}
              {result.ocr_data?.text && (
                <div className="result-card">
                  <div
                    className="result-card-header"
                    onClick={() => toggleCard('ocr')}
                  >
                    <div className="result-card-title">
                      <FileText size={20} />
                      Extracted Text (OCR)
                    </div>
                    {expandedCards.ocr ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                  {expandedCards.ocr && (
                    <div className="result-card-content">
                      <pre className="ocr-text">{result.ocr_data.text}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App
