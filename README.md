# Synthetic Data Generator (SDG) v2.0 🎯

A FastAPI-based service and lightweight frontend for generating synthetic tabular data and images. SDG uses Qwen2.5-1.5B-Instruct for structured text generation and Stable Diffusion v1.5 for image synthesis, and provides job tracking and a simple web UI for rapid prototyping.

## Overview

SDG helps teams create realistic, schema-driven tabular datasets and prompt-driven image datasets for testing, demos, and model evaluation. It supports:
- Schema-first text generation with type validation
- Prompt-based image generation (Stable Diffusion)
- Unified runs that pair text rows with generated images
- Asynchronous background jobs and progress monitoring
- A minimal frontend for building schemas and monitoring jobs

## Architecture

- Runtime: Python 3.11 (FastAPI)
- API Framework: FastAPI (uvicorn)
- Text Model: Qwen2.5-1.5B-Instruct (CUDA-enabled)
- Image Model: runwayml/stable-diffusion-v1-5 (Diffusers, FP16)
- Persistence: Local filesystem for generated outputs (generated_data/)
- Background processing: FastAPI background tasks / async workers
- Frontend: Static HTML/CSS/JS (vanilla)

## Base URL (Local dev)

```
http://localhost:8000
```

## Health and Status

### GET /health
Returns system health and model status.

Response example:
```json
{
  "status": "healthy",
  "text_model": "qwen2.5-instruct (loaded)",
  "image_model": "stable-diffusion-v1-5 (loaded)",
  "jobs_in_progress": 0
}
```

Status codes:
- 200 OK — Service is healthy
- 500 Internal Server Error — Service is degraded

---

## API Endpoints

All endpoints are mounted on the base URL above.

### GET /
API info and basic status.

### POST /generate/text
Generate tabular/text data from a schema.

Request body (JSON):
```json
{
  "schema": {
    "customer_name": "string",
    "age": "int",
    "order_value": "float"
  },
  "num_samples": 100,
  "output_format": "json"
}
```

Response (202 Accepted):
```json
{
  "job_id": "job_012345",
  "message": "Text generation job queued"
}
```

### POST /generate/images
Generate images from prompts.

Request body (JSON):
```json
{
  "prompts": [
    "A modern warehouse interior",
    "An outdoor retail storefront"
  ],
  "num_images": 10,
  "resolution": [512, 512],
  "annotations": ["bbox"]
}
```

Response (202 Accepted):
```json
{
  "job_id": "job_987654",
  "message": "Image generation job queued"
}
```

### POST /generate/unified
Generate combined text + images. Accepts a text schema and image prompts; returns a job id for a combined output package.

Request body example:
```json
{
  "schema": {"id": "int", "product": "string", "price": "float"},
  "num_samples": 50,
  "prompts": ["A product on white background"],
  "output_format": "json"
}
```

### GET /status/{job_id}
Check job progress and retrieve results when complete.

Response example:
```json
{
  "job_id": "job_012345",
  "status": "completed",
  "progress": 100,
  "output_path": "generated_data/text_data_2025-12-03_120000/"
}
```

### GET /jobs
List all jobs (queued, running, completed).

Response example:
```json
[
  {
    "job_id": "job_012345",
    "type": "text",
    "status": "completed",
    "created_at": "2025-12-03T12:00:00Z"
  }
]
```

---

## Usage

### Web UI
- Open frontend/index.html (or run a static server) and visit the site.
- Build a schema using the visual editor (field name + type).
- Set number of samples and output format (JSON/CSV).
- For images, enter prompts (one per line) and set image options.
- Submit a job and monitor its progress in the Jobs panel.

### API (Python)
```python
import requests

# Submit text generation job
r = requests.post('http://localhost:8000/generate/text', json={
    "schema": {"name": "string", "age": "int"},
    "num_samples": 50,
    "output_format": "csv"
})
job_id = r.json()["job_id"]

# Poll for status
status = requests.get(f'http://localhost:8000/status/{job_id}').json()
print(status)
```

### cURL
```bash
curl -X POST "http://localhost:8000/generate/text" \
  -H "Content-Type: application/json" \
  -d '{"schema":{"name":"string","age":"int"},"num_samples":50,"output_format":"csv"}'
```

---

## Output Layout

Text example:
generated_data/
  text_data_YYYYMMDD_HHMMSS/
    data.json
    schema.json

Images example:
generated_data/
  images_YYYYMMDD_HHMMSS/
    images/
      image_00000.png
    annotations/
      annotations.json  # optional COCO-like annotations

---

## Models & Performance

- Text: Qwen2.5-1.5B-Instruct
  - First load may download the model (~3.1 GB) and take time
  - Subsequent loads are faster with local caching
  - Typical throughput: ~2–5s per sample on CUDA (varies by GPU & batch size)

- Images: Stable Diffusion v1.5
  - FP16 on CUDA, default 30 inference steps, guidance 7.5
  - Typical throughput: ~8–12s per 512×512 image (varies by GPU)

Hardware guidance:
- CUDA-capable GPU recommended (GTX 1650 Ti or better)
- ~4–6 GB VRAM for minimal image runs
- ~10 GB disk for models

---

## Requirements

- Python 3.11
- CUDA toolkit (for GPU acceleration)
- pip dependencies: see backend/requirements.txt
- Optional: A GPU with NVIDIA drivers and compatible PyTorch build

---

## Local Setup

Backend:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
python -m http.server 3000
# then open http://localhost:3000
```

Testing:
```bash
cd backend
python test_v2_api.py
python test_qwen.py
python test_imports.py
```

---

## Environment Variables

Set these for production or advanced local config:

```bash
# Optional CUDA / device overrides
SDG_DEVICE=cuda
# Model cache / storage
MODEL_CACHE_DIR=/path/to/model_cache
# FastAPI recommended settings
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
```

---

## Troubleshooting

- API won't start: ensure Python venv is activated and uvicorn is installed; check port 8000 availability.
- Model download fails: confirm internet connection and free disk space (~10 GB).
- GPU issues: confirm CUDA drivers and compatible PyTorch build (check python check_cuda.py).
- Frontend can't connect: verify API health at /health and CORS in backend main.py.

---

## Authentication & Security

This PoC ships without authentication. For production consider:
- API key or JWT authentication
- Rate limiting and request validation
- Model access restrictions and logging
- Secrets management for API keys and model tokens

---

## Error Responses

Errors follow FastAPI standard:
```json
{"detail": "Error message"}
```
Common statuses:
- 200 OK — Success
- 201 Created — Resource created
- 202 Accepted — Job queued
- 400 Bad Request — Invalid input
- 404 Not Found — Resource missing
- 500 Internal Server Error — Server problem

---

## Deployment & Costs

This repository targets local and cloud development environments. If deploying to cloud:
- Containerize backend
- Use GPU-enabled hosts for model inference
- Consider storage for generated datasets (S3/Blob storage)
- Monitor costs for GPU instances and storage

Estimated local resource needs:
- Disk: ~10 GB
- VRAM: 4–6 GB (min)
- CPU + memory: depends on concurrency

---

## Development & Contributing

- FastAPI backend (backend/)
- Static frontend (frontend/)
- Tests under backend/
- To contribute, open issues or PRs describing the feature or bug. Follow repository code style and run tests locally.

---

## License & Models

- Project code: see LICENSE in repository
- Qwen2.5-1.5B-Instruct: Apache-2.0 (follow model terms)
- Stable Diffusion v1.5: CreativeML Open RAIL-M (follow model license and safety guidance)

---

## Support

If you run into issues:
1. Check interactive docs at http://localhost:8000/docs
2. Inspect backend logs for errors
3. Open an issue on the repository with reproduction steps and logs
