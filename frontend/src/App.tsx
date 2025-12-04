import { useState } from 'react'
import './App.css'
import { TemplateSelector, AISchemaBuilder, DataAugmenter } from './components'
import { Template } from './templates'

// Types
interface Job {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  created_at: string
  type: string
  result?: any  // Changed from outputs to result
  error?: string
}

function App() {
  const [activeTab, setActiveTab] = useState<'text' | 'image' | 'unified'>('text')
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)

  // Text generation state
  const [textSchema, setTextSchema] = useState(`{
  "name": {"type": "string", "description": "Full name"},
  "age": {"type": "integer", "minimum": 18, "maximum": 80},
  "email": {"type": "email"},
  "city": {"type": "string", "examples": ["New York", "London", "Tokyo"]}
}`)
  const [numSamples, setNumSamples] = useState(100)
  const [outputFormat, setOutputFormat] = useState<'json' | 'csv' | 'sql'>('json')

  // Image generation state
  const [imagePrompts, setImagePrompts] = useState('A futuristic warehouse with robots\nA modern office interior with computers\nAn outdoor scene with vehicles')
  const [numImages, setNumImages] = useState(10)
  const [resolution, setResolution] = useState<[number, number]>([512, 512])
  const [sceneType, setSceneType] = useState('indoor')
  const [annotations, setAnnotations] = useState<string[]>(['bbox'])

  const API_BASE = 'http://localhost:8000'

  // Generate text data
  const handleTextGeneration = async (e?: React.MouseEvent) => {
    e?.preventDefault()
    setLoading(true)
    try {
      const schema = JSON.parse(textSchema)
      const response = await fetch(`${API_BASE}/generate/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          schema,
          num_samples: numSamples,
          format: outputFormat
        })
      })
      const data = await response.json()
      pollJobStatus(data.job_id)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to generate text data. Check console for details.')
    } finally {
      setLoading(false)
    }
  }

  // Generate images
  const handleImageGeneration = async (e?: React.MouseEvent) => {
    e?.preventDefault()
    setLoading(true)
    try {
      const promptList = imagePrompts.split('\n').filter(p => p.trim())
      const response = await fetch(`${API_BASE}/generate/images`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompts: promptList,
          num_images: numImages,
          resolution,
          scene_type: sceneType,
          annotations
        })
      })
      const data = await response.json()
      pollJobStatus(data.job_id)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to generate images. Check console for details.')
    } finally {
      setLoading(false)
    }
  }

  // Generate unified dataset
  const handleUnifiedGeneration = async (e?: React.MouseEvent) => {
    e?.preventDefault()
    setLoading(true)
    try {
      const schema = JSON.parse(textSchema)
      const promptList = imagePrompts.split('\n').filter(p => p.trim())
      
      const response = await fetch(`${API_BASE}/generate/unified`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text_config: {
            schema,
            num_samples: numSamples,
            format: outputFormat
          },
          image_config: {
            prompts: promptList,
            num_images: numImages,
            resolution,
            scene_type: sceneType,
            annotations
          }
        })
      })
      const data = await response.json()
      pollJobStatus(data.job_id)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to generate unified dataset. Check console for details.')
    } finally {
      setLoading(false)
    }
  }

  // Poll job status
  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/status/${jobId}`)
        const job: Job = await response.json()
        
        // Ensure job_id is set
        if (!job.job_id) {
          job.job_id = jobId
        }
        
        setJobs(prev => {
          const existing = prev.findIndex(j => j.job_id === jobId)
          if (existing >= 0) {
            const updated = [...prev]
            updated[existing] = job
            return updated
          }
          return [job, ...prev]
        })

        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval)
        }
      } catch (error) {
        console.error('Error polling job:', error)
        clearInterval(interval)
      }
    }, 3000) // Increased to 3 seconds to reduce lag
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="nvidia-eye">
              <div className="eye-center"></div>
            </div>
            <h1>NVIDIA SDG</h1>
          </div>
          <p className="subtitle">Synthetic Data Generation Platform</p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-tabs">
        <button
          className={`tab ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <span className="tab-icon">📝</span>
          Text Data
        </button>
        <button
          className={`tab ${activeTab === 'image' ? 'active' : ''}`}
          onClick={() => setActiveTab('image')}
        >
          <span className="tab-icon">🖼️</span>
          Images
        </button>
        <button
          className={`tab ${activeTab === 'unified' ? 'active' : ''}`}
          onClick={() => setActiveTab('unified')}
        >
          <span className="tab-icon">🔄</span>
          Unified
        </button>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-grid">
          {/* Configuration Panel */}
          <div className="config-panel">
            <h2 className="panel-title">Configuration</h2>

            {/* Text Generation Form */}
            {activeTab === 'text' && (
              <div className="form-section">
                {/* Template Selector */}
                <TemplateSelector onSelect={(template) => {
                  setTextSchema(JSON.stringify(template.schema, null, 2))
                  setNumSamples(template.samples)
                  if (template.prompts) {
                    setImagePrompts(template.prompts.join('\n'))
                  }
                }} />

                {/* AI Schema Builder */}
                <AISchemaBuilder onSchemaGenerated={(schema) => {
                  setTextSchema(JSON.stringify(schema, null, 2))
                }} />

                {/* Data Augmenter */}
                <DataAugmenter onAugmented={(jobId) => {
                  pollJobStatus(jobId)
                }} />

                <div className="form-group">
                  <label>Data Schema (JSON)</label>
                  <textarea
                    className="textarea"
                    rows={10}
                    value={textSchema}
                    onChange={(e) => setTextSchema(e.target.value)}
                    placeholder='{"field": {"type": "string"}}'
                  />
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Number of Samples</label>
                    <input
                      type="number"
                      className="input"
                      value={numSamples}
                      onChange={(e) => setNumSamples(Number(e.target.value))}
                      min={1}
                      max={100000}
                    />
                  </div>

                  <div className="form-group">
                    <label>Output Format</label>
                    <select
                      className="select"
                      value={outputFormat}
                      onChange={(e) => setOutputFormat(e.target.value as any)}
                    >
                      <option value="json">JSON</option>
                      <option value="csv">CSV</option>
                      <option value="sql">SQL</option>
                    </select>
                  </div>
                </div>

                <button
                  type="button"
                  className="generate-btn"
                  onClick={handleTextGeneration}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : '⚡ Generate Text Data'}
                </button>
              </div>
            )}

            {/* Image Generation Form */}
            {activeTab === 'image' && (
              <div className="form-section">
                <div className="form-group">
                  <label>Image Prompts (one per line)</label>
                  <textarea
                    className="textarea"
                    rows={6}
                    value={imagePrompts}
                    onChange={(e) => setImagePrompts(e.target.value)}
                    placeholder="A futuristic warehouse&#10;Modern office interior&#10;Outdoor scene"
                  />
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Number of Images</label>
                    <input
                      type="number"
                      className="input"
                      value={numImages}
                      onChange={(e) => setNumImages(Number(e.target.value))}
                      min={1}
                      max={100}
                    />
                  </div>

                  <div className="form-group">
                    <label>Scene Type</label>
                    <select
                      className="select"
                      value={sceneType}
                      onChange={(e) => setSceneType(e.target.value)}
                    >
                      <option value="indoor">Indoor</option>
                      <option value="outdoor">Outdoor</option>
                      <option value="warehouse">Warehouse</option>
                      <option value="general">General</option>
                    </select>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Width</label>
                    <input
                      type="number"
                      className="input"
                      value={resolution[0]}
                      onChange={(e) => setResolution([Number(e.target.value), resolution[1]])}
                      step={64}
                    />
                  </div>

                  <div className="form-group">
                    <label>Height</label>
                    <input
                      type="number"
                      className="input"
                      value={resolution[1]}
                      onChange={(e) => setResolution([resolution[0], Number(e.target.value)])}
                      step={64}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label>Annotations</label>
                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={annotations.includes('bbox')}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setAnnotations([...annotations, 'bbox'])
                          } else {
                            setAnnotations(annotations.filter(a => a !== 'bbox'))
                          }
                        }}
                      />
                      Bounding Boxes
                    </label>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={annotations.includes('segmentation')}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setAnnotations([...annotations, 'segmentation'])
                          } else {
                            setAnnotations(annotations.filter(a => a !== 'segmentation'))
                          }
                        }}
                      />
                      Segmentation
                    </label>
                  </div>
                </div>

                <button
                  type="button"
                  className="generate-btn"
                  onClick={handleImageGeneration}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : '⚡ Generate Images'}
                </button>
              </div>
            )}

            {/* Unified Form */}
            {activeTab === 'unified' && (
              <div className="form-section">
                <div className="info-box">
                  <p>🔄 Generate both text data and images in one workflow.</p>
                </div>

                <h3 style={{marginTop: '1rem', marginBottom: '0.5rem', color: 'var(--nvidia-green)'}}>Text Configuration</h3>
                <div className="form-group">
                  <label>Data Schema (JSON)</label>
                  <textarea
                    className="textarea"
                    rows={6}
                    value={textSchema}
                    onChange={(e) => setTextSchema(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Number of Samples</label>
                  <input
                    type="number"
                    className="input"
                    value={numSamples}
                    onChange={(e) => setNumSamples(Number(e.target.value))}
                  />
                </div>

                <h3 style={{marginTop: '1.5rem', marginBottom: '0.5rem', color: 'var(--nvidia-green)'}}>Image Configuration</h3>
                <div className="form-group">
                  <label>Image Prompts</label>
                  <textarea
                    className="textarea"
                    rows={4}
                    value={imagePrompts}
                    onChange={(e) => setImagePrompts(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Number of Images</label>
                  <input
                    type="number"
                    className="input"
                    value={numImages}
                    onChange={(e) => setNumImages(Number(e.target.value))}
                  />
                </div>

                <button
                  type="button"
                  className="generate-btn"
                  onClick={handleUnifiedGeneration}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : '⚡ Generate Unified Dataset'}
                </button>
              </div>
            )}
          </div>

          {/* Jobs Panel */}
          <div className="jobs-panel">
            <h2 className="panel-title">Generation Jobs</h2>

            {jobs.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <p>No jobs yet</p>
                <p className="empty-subtitle">Generate data to see jobs here</p>
              </div>
            ) : (
              <div className="jobs-list">
                {jobs.map((job, index) => (
                  <div key={job.job_id || `job-${index}`} className={`job-card ${job.status}`}>
                    <div className="job-header">
                      <div>
                        <span className="job-type">{job.type}</span>
                        <span className={`job-status ${job.status}`}>
                          {job.status}
                        </span>
                      </div>
                      <span className="job-time">
                        {new Date(job.created_at).toLocaleTimeString()}
                      </span>
                    </div>

                    {job.status === 'running' && (
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{ width: `${job.progress}%` }}
                        >
                          <span className="progress-text">{job.progress.toFixed(0)}%</span>
                        </div>
                      </div>
                    )}

                    {job.status === 'completed' && job.result && (
                      <div className="job-outputs">
                        <p className="output-label">✓ Generated:</p>
                        {job.result.num_samples && (
                          <p>📝 {job.result.num_samples} samples</p>
                        )}
                        {job.result.num_images && (
                          <p>🖼️ {job.result.num_images} images</p>
                        )}
                        {job.result.file && (
                          <p className="output-file">📁 {job.result.file}</p>
                        )}
                        {job.result.images_dir && (
                          <p className="output-file">📁 {job.result.images_dir}</p>
                        )}
                        {job.result.generator && (
                          <p className="output-info">🔧 {job.result.generator}</p>
                        )}
                        <button 
                          type="button"
                          className="download-btn"
                          onClick={() => {
                            if (job.job_id) {
                              window.open(`${API_BASE}/download/${job.job_id}`, '_blank')
                            }
                          }}
                        >
                          ⬇️ Download Results
                        </button>
                      </div>
                    )}

                    {job.status === 'failed' && job.error && (
                      <div className="job-error">
                        <p>❌ Error: {job.error}</p>
                      </div>
                    )}

                    <div className="job-id">ID: {job.job_id?.slice(0, 8) || 'Unknown'}...</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by NVIDIA • Qwen2.5-1.5B • Stable Diffusion v1.5</p>
      </footer>
    </div>
  )
}

export default App