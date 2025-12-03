"""Quick API test for Qwen text generation"""
import requests
import json
import time

# Test schema-driven generation
payload = {
    "schema": {
        "customer_name": {"type": "string", "description": "Full customer name"},
        "age": {"type": "integer", "minimum": 18, "maximum": 70},
        "email": {"type": "email"},
        "order_value": {"type": "float", "minimum": 10.0, "maximum": 5000.0},
        "is_premium": {"type": "boolean"}
    },
    "num_samples": 3,
    "format": "json"
}

print("Sending request to API...")
response = requests.post("http://localhost:8000/generate", json=payload)

if response.status_code == 200:
    result = response.json()
    job_id = result["job_id"]
    print(f"✓ Job created: {job_id}")
    
    # Poll for completion
    print("Waiting for generation...")
    for i in range(30):
        time.sleep(2)
        status_response = requests.get(f"http://localhost:8000/status/{job_id}")
        status_data = status_response.json()
        
        print(f"  Status: {status_data['status']} - Progress: {status_data.get('progress', 0):.1f}%")
        
        if status_data["status"] == "completed":
            print("\n✓ Generation completed!")
            print("\nOutputs:", json.dumps(status_data.get("outputs"), indent=2))
            break
        elif status_data["status"] == "failed":
            print(f"\n✗ Generation failed: {status_data.get('error')}")
            break
else:
    print(f"✗ Request failed: {response.status_code}")
    print(response.text)
