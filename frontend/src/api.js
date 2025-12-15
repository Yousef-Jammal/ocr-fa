import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large documents
});

/**
 * Check API health status
 */
export async function checkHealth() {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Backend is not available');
  }
}

/**
 * Full document analysis: OCR + Field Extraction + Qwen Analysis
 * @param {File} file - PDF file to analyze
 * @param {string} query - Optional query for analysis
 */
export async function analyzeDocument(file, query = null) {
  const formData = new FormData();
  formData.append('file', file);
  
  let url = '/analyze';
  if (query) {
    url += `?query=${encodeURIComponent(query)}`;
  }
  
  const response = await api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
}

/**
 * OCR extraction only (faster, no Qwen reasoning)
 * @param {File} file - PDF file to extract
 */
export async function extractDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/extract', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
}

/**
 * Analyze pre-extracted text
 * @param {string} text - Text to analyze
 * @param {string} query - Optional query
 */
export async function analyzeText(text, query = null) {
  const response = await api.post('/analyze-text', {
    text,
    query,
  });
  
  return response.data;
}

/**
 * Answer a specific question about the document
 * @param {File} file - PDF file
 * @param {string} question - Question to ask
 */
export async function askQuestion(file, question) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('question', question);
  
  const response = await api.post('/question', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
}

/**
 * Validate extracted fields
 * @param {Object} fields - Fields to validate
 */
export async function validateFields(fields) {
  const response = await api.post('/validate-fields', {
    fields,
  });
  
  return response.data;
}

export default api;
