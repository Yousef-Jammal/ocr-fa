"""
Test the v2.0.0 API endpoints
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_text_generation():
    """Test text generation endpoint"""
    print("\n=== Testing /generate/text ===")
    
    schema = {
        "customer_name": "string",
        "age": "int",
        "email": "string",
        "order_value": "float",
        "is_premium": "bool"
    }
    
    constraints = {
        "age": {"min": 18, "max": 70},
        "order_value": {"min": 10.0, "max": 5000.0}
    }
    
    request_data = {
        "schema": schema,
        "num_samples": 5,
        "constraints": constraints,
        "output_format": "json"
    }
    
    response = requests.post(f"{BASE_URL}/generate/text", json=request_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Job ID: {result.get('job_id')}")
    
    # Poll for completion
    job_id = result.get('job_id')
    if job_id:
        print("\nPolling for completion...")
        for _ in range(30):
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            status = status_response.json()
            print(f"Progress: {status.get('progress', 0)}% - {status.get('status')}")
            
            if status.get('status') == 'completed':
                print("\n✓ Generation completed!")
                print(f"Output path: {status.get('result', {}).get('output_path')}")
                break
            elif status.get('status') == 'failed':
                print(f"\n✗ Generation failed: {status.get('error')}")
                break
            
            time.sleep(2)

def test_image_generation():
    """Test image generation endpoint"""
    print("\n=== Testing /generate/images ===")
    
    prompts = [
        "A modern warehouse interior with shelves and boxes, photorealistic",
        "An outdoor retail store front with signage, sunny day",
        "An indoor shopping mall with people walking, bright lighting"
    ]
    
    request_data = {
        "prompts": prompts,
        "num_images": 3,
        "resolution": [512, 512],
        "scene_type": "general",
        "annotations": ["bbox"]
    }
    
    response = requests.post(f"{BASE_URL}/generate/images", json=request_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Job ID: {result.get('job_id')}")
    
    # Poll for completion
    job_id = result.get('job_id')
    if job_id:
        print("\nPolling for completion...")
        for _ in range(30):
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            status = status_response.json()
            print(f"Progress: {status.get('progress', 0)}% - {status.get('status')}")
            
            if status.get('status') == 'completed':
                print("\n✓ Generation completed!")
                print(f"Images dir: {status.get('result', {}).get('images_dir')}")
                print(f"Num images: {status.get('result', {}).get('num_images')}")
                break
            elif status.get('status') == 'failed':
                print(f"\n✗ Generation failed: {status.get('error')}")
                break
            
            time.sleep(2)

def test_unified_generation():
    """Test unified generation endpoint"""
    print("\n=== Testing /generate/unified ===")
    
    request_data = {
        "text_config": {
            "schema": {
                "product_name": "string",
                "price": "float",
                "in_stock": "bool"
            },
            "num_samples": 3,
            "output_format": "json"
        },
        "image_config": {
            "prompts": ["A product on a white background"],
            "num_images": 2,
            "resolution": [512, 512]
        }
    }
    
    response = requests.post(f"{BASE_URL}/generate/unified", json=request_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Job ID: {result.get('job_id')}")

if __name__ == "__main__":
    try:
        # Test health first
        test_health()
        
        # Test individual endpoints
        test_text_generation()
        test_image_generation()
        
        # Test unified endpoint
        test_unified_generation()
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to API. Make sure the server is running:")
        print("  cd backend")
        print("  .\\venv311\\Scripts\\activate")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n✗ Error: {e}")
