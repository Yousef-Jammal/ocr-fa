import { DATASET_TEMPLATES, Template } from './templates'

interface TemplateSelectorProps {
  onSelect: (template: Template) => void
}

export const TemplateSelector = ({ onSelect }: TemplateSelectorProps) => {
  return (
    <div className="template-section">
      <h3>📚 Quick Start Templates</h3>
      <div className="template-grid">
        {Object.entries(DATASET_TEMPLATES).map(([key, template]) => (
          <div
            key={key}
            className="template-card"
            onClick={() => onSelect(template)}
          >
            <div className="template-icon">{template.icon}</div>
            <h4>{template.name}</h4>
            <p>{template.description}</p>
            <div className="template-stats">
              <span>📊 {template.samples.toLocaleString()} samples</span>
              {template.prompts && <span>🖼️ {template.prompts.length} prompts</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

interface AISchemaBuilderProps {
  onSchemaGenerated: (schema: any) => void
}

export const AISchemaBuilder = ({ onSchemaGenerated }: AISchemaBuilderProps) => {
  const [description, setDescription] = useState('')
  const [generating, setGenerating] = useState(false)

  const generateSchema = async () => {
    if (!description.trim()) {
      alert('Please describe what data you need')
      return
    }

    setGenerating(true)
    try {
      const response = await fetch('http://localhost:8000/ai/generate-schema', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description })
      })
      const data = await response.json()
      onSchemaGenerated(data.schema)
      alert('✅ Schema generated successfully!')
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to generate schema. Check console for details.')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="ai-schema-builder">
      <h3>🤖 AI Schema Builder</h3>
      <p>Describe what data you need in plain English</p>
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Example: I need customer data with names, emails, ages between 18-65, purchase amounts, and signup dates"
        rows={3}
      />
      <button
        type="button"
        onClick={generateSchema}
        disabled={generating}
        className="ai-btn"
      >
        {generating ? '⏳ Generating...' : '✨ Generate Schema with AI'}
      </button>
    </div>
  )
}

interface DataAugmenterProps {
  onAugmented: (jobId: string) => void
}

export const DataAugmenter = ({ onAugmented }: DataAugmenterProps) => {
  const [file, setFile] = useState<File | null>(null)
  const [numSamples, setNumSamples] = useState(1000)
  const [augmenting, setAugmenting] = useState(false)

  const handleAugment = async () => {
    if (!file) {
      alert('Please select a CSV file')
      return
    }

    setAugmenting(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('num_samples', numSamples.toString())

      const response = await fetch(`http://localhost:8000/augment?num_samples=${numSamples}`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      onAugmented(data.job_id)
      alert('✅ Augmentation started!')
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to augment data. Check console for details.')
    } finally {
      setAugmenting(false)
    }
  }

  return (
    <div className="data-augmenter">
      <h3>🔄 Data Augmentation</h3>
      <p>Upload CSV, AI generates similar but unique data</p>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      {file && (
        <div className="file-info">
          📄 {file.name} ({(file.size / 1024).toFixed(2)} KB)
        </div>
      )}
      <div className="augment-controls">
        <label>
          Samples to generate:
          <input
            type="number"
            value={numSamples}
            onChange={(e) => setNumSamples(parseInt(e.target.value))}
            min={100}
            max={100000}
          />
        </label>
        <button
          type="button"
          onClick={handleAugment}
          disabled={augmenting || !file}
          className="augment-btn"
        >
          {augmenting ? '⏳ Augmenting...' : '🚀 Augment Data'}
        </button>
      </div>
    </div>
  )
}

import { useState } from 'react'
