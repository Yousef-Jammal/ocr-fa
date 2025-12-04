import { useState } from 'react'
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
