from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal, Set
import uuid
from datetime import datetime
import json
from pathlib import Path
import asyncio
import zipfile
import io

# Import generators
from nemo_generator import NeMoTextGenerator
from image_generator_v2 import ImageGenerator

app = FastAPI(
    title="Synthetic Data Generation API",
    description="Generate synthetic tabular data with Qwen2.5-1.5B and images with Stable Diffusion",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generators (lazy loading for faster startup)
text_gen = None
image_gen = None

def get_text_generator():
    global text_gen
    if text_gen is None:
        text_gen = NeMoTextGenerator()
    return text_gen

def get_image_generator():
    global image_gen
    if image_gen is None:
        image_gen = ImageGenerator()
    return image_gen

# Storage configuration
OUTPUT_DIR = Path("./generated_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking (in production, use Redis/DB)
jobs = {}

# ==================== Request Models with Enhanced Validation ====================

class TabularRequest(BaseModel):
    data_schema: Dict[str, Any] = Field(
        alias="schema", 
        description="Data schema definition with field types and constraints"
    )
    num_samples: int = Field(
        default=100, 
        ge=1, 
        le=100000,
        description="Number of samples to generate (1-100,000)"
    )
    format: Literal["json", "csv", "sql"] = Field(
        default="json",
        description="Output format"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional constraints on generated data"
    )

    class Config:
        populate_by_name = True
    
    @validator('data_schema')
    def validate_schema(cls, v):
        if not v:
            raise ValueError("Schema cannot be empty")
        if not isinstance(v, dict):
            raise ValueError("Schema must be a dictionary")
        return v
    
    @validator('num_samples')
    def validate_num_samples(cls, v):
        if v < 1:
            raise ValueError("Number of samples must be at least 1")
        if v > 100000:
            raise ValueError("Number of samples cannot exceed 100,000")
        return v

class ImageRequest(BaseModel):
    prompts: List[str] = Field(
        description="Image generation prompts (one per variation)",
        min_items=1,
        max_items=50
    )
    num_images: int = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Number of images to generate (1-100)"
    )
    resolution: tuple[int, int] = Field(
        default=(512, 512), 
        description="Image resolution (width, height) - must be multiples of 64"
    )
    scene_type: Optional[str] = Field(
        default="general", 
        description="Scene type hint for annotations"
    )
    annotations: List[str] = Field(
        default=["bbox"], 
        description="Annotation types: bbox, segmentation"
    )
    objects: Optional[List[str]] = Field(
        default=None, 
        description="Specific object classes for annotations"
    )
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError("At least one prompt is required")
        if len(v) > 50:
            raise ValueError("Maximum 50 prompts allowed")
        for prompt in v:
            if not prompt.strip():
                raise ValueError("Empty prompts are not allowed")
            if len(prompt) > 500:
                raise ValueError("Prompt length cannot exceed 500 characters")
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        width, height = v
        if width < 256 or height < 256:
            raise ValueError("Resolution must be at least 256x256")
        if width > 1024 or height > 1024:
            raise ValueError("Resolution cannot exceed 1024x1024")
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        return v
    
    @validator('annotations')
    def validate_annotations(cls, v):
        valid_types = {"bbox", "segmentation"}
        for ann_type in v:
            if ann_type not in valid_types:
                raise ValueError(f"Invalid annotation type: {ann_type}. Must be one of {valid_types}")
        return v

class UnifiedRequest(BaseModel):
    text_config: Optional[TabularRequest] = Field(
        default=None,
        description="Text generation configuration"
    )
    image_config: Optional[ImageRequest] = Field(
        default=None,
        description="Image generation configuration"
    )
    
    @validator('text_config', 'image_config')
    def validate_at_least_one(cls, v, values):
        # Check if at least one config is provided
        if 'text_config' in values and not values.get('text_config') and not v:
            raise ValueError("At least one of text_config or image_config must be provided")
        return v

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: str
    result: Optional[Dict[str, Any]] = None  # Changed from outputs
    error: Optional[str] = None

# ==================== Endpoints ====================

@app.get("/")
async def root():
    return {
        "service": "Synthetic Data Generation API",
        "version": "2.0.0",
        "description": "Generate synthetic tabular data and images",
        "models": {
            "text": "Qwen/Qwen2.5-1.5B-Instruct",
            "image": "Stable Diffusion v1.5"
        },
        "endpoints": {
            "health": "GET /health",
            "generate_text": "POST /generate/text",
            "generate_images": "POST /generate/images",
            "unified": "POST /generate/unified",
            "job_status": "GET /status/{job_id}",
            "list_jobs": "GET /jobs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "text_generator": "loaded" if text_gen else "not_loaded",
            "image_generator": "loaded" if image_gen else "not_loaded"
        }
    }

@app.post("/generate/text")
async def generate_text(request: TabularRequest, background_tasks: BackgroundTasks):
    """Generate synthetic tabular/text data using Qwen2.5-1.5B"""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "type": "tabular"
    }
    
    background_tasks.add_task(
        _generate_text_task,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "pending"}

@app.post("/generate/images")
async def generate_images(request: ImageRequest, background_tasks: BackgroundTasks):
    """Generate synthetic images using Stable Diffusion"""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "type": "images"
    }
    
    background_tasks.add_task(
        _generate_images_task,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "pending"}

@app.post("/generate/unified")
async def generate_unified(request: UnifiedRequest, background_tasks: BackgroundTasks):
    """Generate multi-modal dataset (text + images)"""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "type": "unified",
        "sub_jobs": {}
    }
    
    background_tasks.add_task(
        _generate_unified_task,
        job_id,
        request
    )
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id].copy()
    job_data["job_id"] = job_id  # Ensure job_id is always included
    return job_data

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "type": job_data.get("type"),
                "status": job_data.get("status"),
                "created_at": job_data.get("created_at")
            }
            for job_id, job_data in jobs.items()
        ]
    }

# ==================== Background Tasks ====================

async def _generate_text_task(job_id: str, request: TabularRequest):
    """Enhanced background task for text generation with real-time progress"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5.0
        
        # Get generator
        gen = get_text_generator()
        jobs[job_id]["progress"] = 15.0
        
        # Generate data with progress tracking
        result = await gen.generate(
            schema=request.data_schema,
            num_samples=request.num_samples,
            format=request.format,
            constraints=request.constraints
        )
        
        jobs[job_id]["progress"] = 80.0
        
        # Save output
        output_path = OUTPUT_DIR / f"{job_id}.{request.format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use newline='' for CSV to prevent extra blank lines on Windows
        open_mode = "w" if request.format != "csv" else "w"
        newline_param = {} if request.format != "csv" else {"newline": ""}
        
        with open(output_path, open_mode, **newline_param) as f:
            if request.format == "json":
                json.dump(result, f, indent=2)
            else:
                f.write(result)
        
        jobs[job_id]["progress"] = 95.0
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "result": {  # Changed from outputs to result
                "file": str(output_path),
                "num_samples": request.num_samples,
                "format": request.format
            }
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0.0
        })

async def _generate_images_task(job_id: str, request: ImageRequest):
    """Enhanced background task for image generation with real-time progress"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5.0
        
        # Get generator
        gen = get_image_generator()
        jobs[job_id]["progress"] = 10.0
        
        output_dir = OUTPUT_DIR / job_id / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images with progress tracking
        jobs[job_id]["progress"] = 15.0
        
        result = await gen.generate(
            prompts=request.prompts,  # ✅ FIXED: Added missing prompts parameter
            num_images=request.num_images,
            resolution=request.resolution,
            scene_type=request.scene_type,
            annotations=request.annotations,
            objects=request.objects,
            output_dir=output_dir
        )
        
        jobs[job_id]["progress"] = 95.0
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "result": result  # Changed from outputs to result
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0.0
        })

async def _generate_unified_task(job_id: str, request: UnifiedRequest):
    """Background task for unified generation"""
    try:
        jobs[job_id]["status"] = "running"
        sub_jobs = {}
        
        # Generate text if requested
        if request.text_config:
            text_job_id = f"{job_id}_text"
            jobs[text_job_id] = {
                "status": "pending",
                "progress": 0.0,
                "created_at": datetime.now().isoformat()
            }
            await _generate_text_task(text_job_id, request.text_config)
            sub_jobs["text"] = text_job_id
            jobs[job_id]["progress"] = 50.0
        
        # Generate images if requested
        if request.image_config:
            image_job_id = f"{job_id}_images"
            jobs[image_job_id] = {
                "status": "pending",
                "progress": 0.0,
                "created_at": datetime.now().isoformat()
            }
            await _generate_images_task(image_job_id, request.image_config)
            sub_jobs["images"] = image_job_id
            jobs[job_id]["progress"] = 100.0
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "sub_jobs": sub_jobs
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0.0
        })

# ==================== WebSocket Connection Manager ====================

class ConnectionManager:
    """Manage WebSocket connections for real-time job updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def broadcast_job_update(self, job_id: str, job_data: dict):
        if job_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(job_data)
                except:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.disconnect(dead, job_id)

manager = ConnectionManager()

# ==================== Download Endpoints ====================

@app.get("/download/{job_id}")
async def download_job_results(job_id: str):
    """Download results for a completed job as a zip file"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add result files
        if "result" in job:
            result = job["result"]
            
            # Text data
            if "file" in result:
                file_path = Path(result["file"])
                if file_path.exists():
                    zip_file.write(file_path, file_path.name)
            
            # Images
            if "images_dir" in result:
                images_dir = Path(result["images_dir"])
                if images_dir.exists():
                    for img_file in images_dir.glob("*.png"):
                        zip_file.write(img_file, f"images/{img_file.name}")
            
            # Annotations
            if "annotations_dir" in result:
                ann_dir = Path(result["annotations_dir"])
                if ann_dir.exists():
                    for ann_file in ann_dir.glob("*"):
                        zip_file.write(ann_file, f"annotations/{ann_file.name}")
            
            # Add metadata
            metadata = {
                "job_id": job_id,
                "created_at": job.get("created_at"),
                "result": result
            }
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}.zip"}
    )

@app.get("/download/{job_id}/file")
async def download_single_file(job_id: str):
    """Download single result file (for text/CSV generation)"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")
    
    if "result" not in job or "file" not in job["result"]:
        raise HTTPException(status_code=404, detail="No file available for this job")
    
    file_path = Path(job["result"]["file"])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )

# ==================== AI Schema Builder ====================

class SchemaGenerationRequest(BaseModel):
    description: str = Field(..., description="Natural language description of the data you need")

@app.post("/ai/generate-schema")
async def ai_generate_schema(request: SchemaGenerationRequest):
    """🤖 AI-powered schema generation from natural language description"""
    
    generator = get_text_generator()
    
    if not generator.model:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    # Create prompt for schema generation
    prompt = f"""Generate a JSON data schema for the following description:

Description: {request.description}

Create a schema with appropriate fields, types, and constraints. Output ONLY valid JSON in this format:
{{
  "field_name": {{"type": "string|integer|float|boolean|datetime|email", "description": "...", "examples": [...], "minimum": X, "maximum": Y}}
}}

Generate the schema now:"""
    
    try:
        messages = [
            {"role": "system", "content": "You are a data schema expert. Generate precise, well-structured JSON schemas from natural language descriptions. Output ONLY valid JSON without explanations."},
            {"role": "user", "content": prompt}
        ]
        
        text = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = generator.tokenizer([text], return_tensors="pt").to(generator.device)
        
        import torch
        with torch.no_grad():
            generated_ids = generator.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = generator.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            schema_json = response[start_idx:end_idx+1]
            schema = json.loads(schema_json)
            
            return {
                "schema": schema,
                "description": request.description,
                "generated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate valid schema")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema generation failed: {str(e)}")

# ==================== Data Augmentation ====================

from fastapi import UploadFile, File
import pandas as pd

@app.post("/augment")
async def augment_data(
    file: UploadFile = File(...),
    num_samples: int = 1000,
    background_tasks: BackgroundTasks = None
):
    """🔄 Smart data augmentation - upload CSV, get similar synthetic data"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "type": "augmentation",
        "status": "processing",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat()
    }
    
    # Schedule augmentation task
    background_tasks.add_task(_augment_data_task, job_id, file, num_samples)
    
    return {"job_id": job_id, "status": "processing"}

async def _augment_data_task(job_id: str, file: UploadFile, num_samples: int):
    """Background task for data augmentation"""
    
    try:
        jobs[job_id]["progress"] = 10.0
        
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        jobs[job_id]["progress"] = 20.0
        
        # Analyze schema from CSV
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                schema[col] = {
                    "type": "integer",
                    "minimum": int(df[col].min()),
                    "maximum": int(df[col].max())
                }
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = {
                    "type": "float",
                    "minimum": float(df[col].min()),
                    "maximum": float(df[col].max())
                }
            elif pd.api.types.is_bool_dtype(dtype):
                schema[col] = {"type": "boolean"}
            else:
                # String or object - extract examples
                unique_vals = df[col].unique()[:5].tolist()
                schema[col] = {
                    "type": "string",
                    "examples": [str(v) for v in unique_vals if pd.notna(v)]
                }
        
        jobs[job_id]["progress"] = 40.0
        
        # Generate synthetic data using the schema
        generator = get_text_generator()
        result = await generator.generate(
            schema=schema,
            num_samples=num_samples,
            format="csv",
            constraints=None
        )
        
        jobs[job_id]["progress"] = 80.0
        
        # Save augmented data
        output_path = OUTPUT_DIR / f"{job_id}.csv"
        with open(output_path, "w", newline='') as f:
            f.write(result)
        
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "file": str(output_path),
            "samples_generated": num_samples,
            "original_samples": len(df),
            "schema": schema
        }
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Augmentation error: {e}")

# ==================== WebSocket Endpoint ====================

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await manager.connect(websocket, job_id)
    
    try:
        # Send initial job status
        if job_id in jobs:
            await websocket.send_json(jobs[job_id])
        
        # Keep connection alive and send updates
        while True:
            # Wait for messages (or just keep alive)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send periodic updates
                if job_id in jobs:
                    await websocket.send_json(jobs[job_id])
            
            # Check if job is complete
            if job_id in jobs and jobs[job_id]["status"] in ["completed", "failed"]:
                await websocket.send_json(jobs[job_id])
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, job_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)