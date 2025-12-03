# Synthetic Data Generator v2.0 🎯

A streamlined system for generating synthetic tabular data and images using **Qwen2.5-1.5B-Instruct** (text) and **Stable Diffusion v1.5** (images).

## Features

- **📊 Text Data Generation**: Schema-driven business data with type validation
- **🖼️ Image Generation**: Stable Diffusion with prompt-based synthesis
- **⚡ Unified Generation**: Combined text + image datasets
- **📋 Job Tracking**: Async background processing with progress monitoring
- **🎨 Modern UI**: Clean web interface with real-time updates

## Architecture

```
Backend (FastAPI):
- Qwen2.5-1.5B-Instruct for structured text generation
- Stable Diffusion v1.5 for image synthesis
- Lazy-loaded generators for fast startup
- CORS-enabled API with background task processing

Frontend (Vanilla HTML/CSS/JS):
- Schema builder with visual field editor
- Multi-prompt image generation
- Real-time job status updates
- Responsive design
```

## Setup

### 1. Backend Setup

```powershell
cd backend

# Activate Python 3.11 environment
.\venv311\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### 2. Frontend Setup

Open `frontend_v2/index.html` directly in your browser, or serve it:

```powershell
cd frontend_v2
python -m http.server 3000
```

Then visit `http://localhost:3000`

## Usage

### Via Web UI

1. **Text Data**:
   - Build your schema using the visual editor
   - Add fields with names and types (string, int, float, bool)
   - Set number of samples
   - Choose output format (JSON/CSV)
   - Click "Generate"

2. **Images**:
   - Enter prompts (one per line)
   - Set number of images and scene type
   - Optionally enable bounding box annotations
   - Click "Generate"

3. **Unified**:
   - Configure both text schema (JSON format)
   - Add image prompts
   - Generate combined dataset

4. **Jobs**:
   - Monitor all active/completed jobs
   - View progress and results
   - Auto-refreshes every 3 seconds

### Via API (Python)

```python
import requests

# Text generation
response = requests.post('http://localhost:8000/generate/text', json={
    "schema": {
        "customer_name": "string",
        "age": "int",
        "order_value": "float"
    },
    "num_samples": 100,
    "output_format": "json"
})
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/status/{job_id}').json()
print(f"Progress: {status['progress']}%")

# Image generation
response = requests.post('http://localhost:8000/generate/images', json={
    "prompts": [
        "A modern warehouse interior",
        "An outdoor retail storefront"
    ],
    "num_images": 10,
    "resolution": [512, 512],
    "annotations": ["bbox"]
})
```

### Via cURL

```bash
# Text generation
curl -X POST "http://localhost:8000/generate/text" \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {"name": "string", "age": "int"},
    "num_samples": 50,
    "output_format": "csv"
  }'

# Image generation
curl -X POST "http://localhost:8000/generate/images" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["A product on white background"],
    "num_images": 5,
    "resolution": [512, 512]
  }'
```

## API Endpoints

- `GET /` - API info and status
- `GET /health` - Health check with model status
- `POST /generate/text` - Generate tabular data
- `POST /generate/images` - Generate images with SD
- `POST /generate/unified` - Generate text + images
- `GET /status/{job_id}` - Check job status
- `GET /jobs` - List all jobs

## Output Structure

### Text Data
```
generated_data/
  text_data_20240101_120000/
    data.json          # Generated samples
    schema.json        # Schema definition
```

### Images
```
generated_data/
  images_20240101_120000/
    images/
      image_00000.png
      image_00001.png
      ...
    annotations/
      annotations.json  # COCO format (if enabled)
```

## Models

- **Text**: Qwen/Qwen2.5-1.5B-Instruct (3.09 GB)
  - First load: ~35 minutes (downloads model)
  - Subsequent loads: ~30 seconds (cached)
  - CUDA-enabled for GTX 1650 Ti

- **Images**: runwayml/stable-diffusion-v1-5 (~4 GB)
  - Runs at FP16 precision on CUDA
  - 30 inference steps per image
  - Guidance scale: 7.5

## Requirements

- Python 3.11 (venv311)
- CUDA-capable GPU (GTX 1650 Ti or better)
- ~10 GB disk space for models
- ~4-6 GB VRAM for generation

## Testing

```powershell
# Test API endpoints
cd backend
python test_v2_api.py

# Test Qwen model directly
python test_qwen.py

# Test imports
python test_imports.py
```

## Troubleshooting

**API won't start:**
- Check if venv311 is activated: `.\venv311\Scripts\activate`
- Verify CUDA: `python check_cuda.py`
- Check port 8000: `netstat -ano | findstr :8000`

**Model loading fails:**
- Check internet connection (first-time download)
- Check disk space (~10 GB needed)
- Check VRAM (~4 GB minimum)

**Frontend can't connect:**
- Verify API is running: visit `http://localhost:8000/health`
- Check CORS is enabled in main.py
- Try opening browser console for errors

## Performance

**Text Generation:**
- ~2-5 seconds per sample (Qwen2.5-1.5B on CUDA)
- Batch size: 1 (streaming generation)

**Image Generation:**
- ~8-12 seconds per image (SD v1.5 on CUDA)
- Resolution: 512x512 default
- Can generate up to 100 images per request

## Development

Built with:
- FastAPI 0.123.4
- Transformers 4.57.3
- Diffusers 0.35.2
- PyTorch 2.5.1+cu121
- NeMo 2.5.3 (base toolkit)

## License

This project uses:
- Qwen2.5-1.5B-Instruct: Apache 2.0
- Stable Diffusion v1.5: CreativeML Open RAIL-M
