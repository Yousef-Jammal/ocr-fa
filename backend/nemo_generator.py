"""
Text generation using Qwen2.5-1.5B-Instruct for structured data
Generates synthetic text data with schema validation
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import random
from datetime import datetime
import torch

# Transformers for Qwen model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using fallback generator.")

class NeMoTextGenerator:
    """Generate synthetic text data using Qwen2.5-1.5B-Instruct"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._used_ids = set()  # Track unique IDs
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✓ {model_name} loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                print("Using fallback generator")
                self.model = None
    
    async def generate(
        self,
        schema: Dict[str, Any],
        num_samples: int,
        format: str = "json",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate synthetic data based on schema
        
        Args:
            schema: Data schema definition
            num_samples: Number of samples to generate
            format: Output format (json, csv, sql)
            constraints: Optional constraints on generated data
            
        Returns:
            Generated data in requested format
        """
        
        if self.model and TRANSFORMERS_AVAILABLE:
            return await self._generate_with_qwen(
                schema, num_samples, format, constraints
            )
        else:
            return await self._generate_fallback(
                schema, num_samples, format, constraints
            )
    
    async def _generate_with_qwen(
        self,
        schema: Dict[str, Any],
        num_samples: int,
        format: str,
        constraints: Optional[Dict[str, Any]]
    ) -> Any:
        """Generate using Qwen2.5 model with enhanced prompting"""
        
        samples = []
        
        # 🔥 ENHANCEMENT 1: Generate few-shot examples from schema
        few_shot_examples = self._generate_few_shot_examples(schema, num_examples=2)
        
        for i in range(num_samples):
            # Create enhanced prompt with examples
            prompt = self._create_enhanced_prompt(schema, constraints, few_shot_examples)
            
            # Generate with Qwen
            try:
                messages = [
                    {"role": "system", "content": "You are an expert synthetic data generator. Generate highly realistic, diverse, and contextually accurate structured data. Follow the schema precisely and ensure data relationships are logical. Output ONLY valid JSON without any explanations."},
                    {"role": "user", "content": prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                # 🔥 ENHANCEMENT 2: Better generation parameters for quality
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=768,  # Increased for complex schemas
                        temperature=0.8,  # Slightly higher for diversity
                        do_sample=True,
                        top_p=0.92,  # Adjusted for better quality
                        top_k=50,  # Add top_k sampling
                        repetition_penalty=1.1,  # Reduce repetition
                        no_repeat_ngram_size=3  # Prevent phrase repetition
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 🔥 ENHANCEMENT 3: Better parsing with validation
                sample = self._parse_llm_response(response, schema)
                
                # Validate sample quality
                if self._validate_sample_quality(sample, schema):
                    samples.append(sample)
                else:
                    print(f"Sample {i} failed quality check, regenerating...")
                    # Retry with different temperature
                    sample = self._generate_fallback_sample(schema, constraints)
                    samples.append(sample)
                    
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                # Fallback to template generation for this sample
                sample = self._generate_fallback_sample(schema, constraints)
                samples.append(sample)
        
        return self._format_output(samples, format)
    
    async def _generate_fallback(
        self,
        schema: Dict[str, Any],
        num_samples: int,
        format: str,
        constraints: Optional[Dict[str, Any]]
    ) -> Any:
        """Fallback generator using templates"""
        
        samples = []
        
        for i in range(num_samples):
            sample = {}
            
            for field_name, field_spec in schema.items():
                field_type = field_spec.get("type", "string")
                
                # Generate value based on type
                if field_type == "string":
                    sample[field_name] = self._generate_string(field_spec, constraints)
                elif field_type == "integer":
                    sample[field_name] = self._generate_integer(field_spec, constraints)
                elif field_type == "float":
                    sample[field_name] = self._generate_float(field_spec, constraints)
                elif field_type == "boolean":
                    sample[field_name] = random.choice([True, False])
                elif field_type == "datetime":
                    sample[field_name] = datetime.now().isoformat()
                elif field_type == "email":
                    sample[field_name] = f"user{i}@example.com"
                elif field_type == "url":
                    sample[field_name] = f"https://example.com/resource/{i}"
                else:
                    sample[field_name] = None
            
            samples.append(sample)
        
        return self._format_output(samples, format)
    
    def _generate_string(
        self,
        field_name: str,
        field_spec: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """🔥 ENHANCED: Generate realistic string values based on field context"""
        
        field_name = field_name.lower()
        examples = field_spec.get("examples", [])
        
        if examples:
            return random.choice(examples)
        
        # 🔥 Context-aware generation based on field name
        if "name" in field_name or "customer" in field_name or "user" in field_name:
            first_names = ["John", "Emma", "Michael", "Sophia", "William", "Olivia", "James", "Ava", "Robert", "Isabella"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
            if "full" in field_name or "holder" in field_name:
                return f"{random.choice(first_names)} {random.choice(last_names)}"
            elif "first" in field_name:
                return random.choice(first_names)
            elif "last" in field_name or "surname" in field_name:
                return random.choice(last_names)
            return f"{random.choice(first_names)} {random.choice(last_names)}"
        
        if "email" in field_name or "mail" in field_name:
            domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "example.com"]
            names = ["john.doe", "jane.smith", "user", "contact", "info"]
            return f"{random.choice(names)}{random.randint(1, 999)}@{random.choice(domains)}"
        
        if "phone" in field_name or "mobile" in field_name or "tel" in field_name:
            return f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        
        if "address" in field_name or "street" in field_name:
            streets = ["Main St", "Oak Ave", "Maple Dr", "Park Blvd", "Washington St", "First Ave"]
            return f"{random.randint(1, 9999)} {random.choice(streets)}"
        
        if "city" in field_name:
            cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
            return random.choice(cities)
        
        if "country" in field_name:
            countries = ["USA", "Canada", "UK", "Germany", "France", "Spain", "Italy", "Japan"]
            return random.choice(countries)
        
        if "state" in field_name or "province" in field_name:
            states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
            return random.choice(states)
        
        if "zip" in field_name or "postal" in field_name:
            return f"{random.randint(10000, 99999)}"
        
        if "product" in field_name or "item" in field_name:
            products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Camera", "Monitor", "Keyboard", "Mouse"]
            return random.choice(products)
        
        if "description" in field_name or "comment" in field_name or "note" in field_name:
            descriptions = [
                "High quality product with excellent features",
                "Premium service with fast delivery",
                "Reliable and efficient solution",
                "Customer satisfaction guaranteed",
                "Industry leading performance"
            ]
            return random.choice(descriptions)
        
        if "id" in field_name or "code" in field_name:
            return f"{random.choice(['CUS', 'PRD', 'ORD', 'INV', 'TXN'])}{random.randint(100000, 999999)}"
        
        if "currency" in field_name:
            return random.choice(["USD", "EUR", "GBP", "JPY", "CAD", "AUD"])
        
        if "merchant" in field_name or "vendor" in field_name or "store" in field_name:
            merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Apple Store", "Home Depot", "Costco", "Starbucks"]
            return random.choice(merchants)
        
        if "status" in field_name:
            return random.choice(["active", "pending", "completed", "cancelled", "processing"])
        
        # Default: generate readable text
        min_length = field_spec.get("min_length", 5)
        max_length = field_spec.get("maximum", 50)
        
        words = ["premium", "quality", "professional", "advanced", "efficient", "reliable", 
                 "innovative", "modern", "secure", "powerful", "flexible", "optimized"]
        
        result = []
        current_length = 0
        target_length = random.randint(min_length, min(max_length, 100))
        
        while current_length < target_length:
            word = random.choice(words)
            result.append(word)
            current_length += len(word) + 1
        
        return " ".join(result)[:target_length].strip()
    
    def _generate_integer(
        self,
        field_name: str,
        field_spec: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> int:
        """🔥 ENHANCED: Generate unique IDs for ID fields"""
        
        field_name_lower = field_name.lower()
        min_val = field_spec.get("minimum", 0)
        max_val = field_spec.get("maximum", 100)
        
        # Generate unique IDs for ID fields
        if "id" in field_name_lower and "_id" in field_name_lower:
            # Use the provided range, or default to 100000-999999
            id_min = max(min_val, 1)
            id_max = max(max_val, id_min + 1000)
            
            unique_id = random.randint(id_min, id_max)
            attempts = 0
            while unique_id in self._used_ids and attempts < 100:
                unique_id = random.randint(id_min, id_max)
                attempts += 1
            
            self._used_ids.add(unique_id)
            return unique_id
        
        return random.randint(min_val, max_val)
    
    def _generate_float(
        self,
        field_spec: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> float:
        """Generate float value"""
        
        min_val = field_spec.get("minimum", 0.0)
        max_val = field_spec.get("maximum", 100.0)
        
        return random.uniform(min_val, max_val)
    
    def _generate_datetime(self, field_name: str) -> str:
        """Generate realistic datetime based on field context"""
        from datetime import timedelta
        
        field_name_lower = field_name.lower()
        now = datetime.now()
        
        if "birth" in field_name_lower or "dob" in field_name_lower:
            # Generate birth dates between 1-120 years ago
            years_ago = random.randint(1, 120)
            date = now - timedelta(days=years_ago * 365)
            return date.strftime("%Y-%m-%d")
        
        if "last" in field_name_lower or "recent" in field_name_lower or "visit" in field_name_lower:
            # Recent dates within last 2 years
            days_ago = random.randint(0, 730)
            date = now - timedelta(days=days_ago)
            return date.strftime("%Y-%m-%d")
        
        return now.isoformat()
    
    def _generate_email(self, field_name: str) -> str:
        """Generate realistic email addresses"""
        domains = ["gmail.com", "yahoo.com", "outlook.com", "hospital.org", "clinic.com", "healthcare.net"]
        prefixes = ["contact", "emergency", "info", "patient", "admin", "support"]
        
        if "emergency" in field_name.lower():
            return f"{random.choice(prefixes)}{random.randint(1, 999)}@{random.choice(domains)}"
        
        first_names = ["john", "jane", "michael", "sarah", "david", "emma", "robert", "lisa"]
        last_names = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis"]
        
        return f"{random.choice(first_names)}.{random.choice(last_names)}{random.randint(1, 99)}@{random.choice(domains)}"
    
    def _create_prompt_from_schema(
        self,
        schema: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for Qwen model based on schema"""
        
        prompt = "Generate a single data record as valid JSON matching this schema:\n\n"
        prompt += "Schema:\n"
        
        for field_name, field_spec in schema.items():
            field_type = field_spec.get("type", "string")
            description = field_spec.get("description", "")
            
            prompt += f"  - {field_name} (type: {field_type})"
            
            if description:
                prompt += f": {description}"
            
            # Add constraints
            if field_type == "integer" or field_type == "float":
                if "minimum" in field_spec:
                    prompt += f" (min: {field_spec['minimum']}"
                if "maximum" in field_spec:
                    prompt += f", max: {field_spec['maximum']})"
                elif "minimum" in field_spec:
                    prompt += ")"
            
            if "examples" in field_spec:
                prompt += f" Examples: {', '.join(map(str, field_spec['examples'][:3]))}"
            
            prompt += "\n"
        
        if constraints:
            prompt += f"\nAdditional constraints: {json.dumps(constraints)}\n"
        
        prompt += "\nGenerate realistic, valid data as JSON. Output format: {\"field1\": value1, \"field2\": value2, ...}"
        
        return prompt
    
    def _create_enhanced_prompt(
        self,
        schema: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
        few_shot_examples: List[Dict[str, Any]]
    ) -> str:
        """🔥 ENHANCED: Create advanced prompt with few-shot learning"""
        
        prompt = "# Task: Generate Realistic Synthetic Data\n\n"
        prompt += "## Schema Definition:\n"
        
        for field_name, field_spec in schema.items():
            field_type = field_spec.get("type", "string")
            description = field_spec.get("description", "")
            
            prompt += f"### {field_name}\n"
            prompt += f"- Type: {field_type}\n"
            
            if description:
                prompt += f"- Description: {description}\n"
            
            # Add detailed constraints
            if field_type in ["integer", "float"]:
                if "minimum" in field_spec:
                    prompt += f"- Minimum: {field_spec['minimum']}\n"
                if "maximum" in field_spec:
                    prompt += f"- Maximum: {field_spec['maximum']}\n"
            
            if "examples" in field_spec and field_spec["examples"]:
                prompt += f"- Valid Examples: {', '.join(map(str, field_spec['examples'][:5]))}\n"
            
            if "pattern" in field_spec:
                prompt += f"- Pattern: {field_spec['pattern']}\n"
            
            prompt += "\n"
        
        # Add few-shot examples
        if few_shot_examples:
            prompt += "## Example Records (for reference):\n\n"
            for idx, example in enumerate(few_shot_examples, 1):
                prompt += f"Example {idx}:\n```json\n{json.dumps(example, indent=2)}\n```\n\n"
        
        if constraints:
            prompt += f"## Additional Constraints:\n{json.dumps(constraints, indent=2)}\n\n"
        
        prompt += "## Instructions:\n"
        prompt += "1. Generate ONE new data record following the schema exactly\n"
        prompt += "2. Ensure all data is realistic and contextually appropriate\n"
        prompt += "3. Maintain logical relationships between fields\n"
        prompt += "4. Use diverse values (don't repeat example data)\n"
        prompt += "5. Output ONLY the JSON object, no explanations\n\n"
        prompt += "Generate the record now:"
        
        return prompt
    
    def _generate_few_shot_examples(
        self,
        schema: Dict[str, Any],
        num_examples: int = 2
    ) -> List[Dict[str, Any]]:
        """🔥 NEW: Generate few-shot examples for better prompting"""
        
        examples = []
        for i in range(num_examples):
            example = self._generate_fallback_sample(schema, None)
            examples.append(example)
        return examples
    
    def _validate_sample_quality(
        self,
        sample: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> bool:
        """🔥 NEW: Validate generated sample meets quality standards"""
        
        if not sample:
            return False
        
        # Check all required fields present
        for field_name in schema.keys():
            if field_name not in sample:
                return False
        
        # Check field types and constraints
        for field_name, field_spec in schema.items():
            value = sample.get(field_name)
            field_type = field_spec.get("type", "string")
            
            # Type validation
            if field_type == "integer" and not isinstance(value, int):
                return False
            if field_type == "float" and not isinstance(value, (int, float)):
                return False
            if field_type == "boolean" and not isinstance(value, bool):
                return False
            if field_type == "string" and not isinstance(value, str):
                return False
            
            # Range validation
            if field_type in ["integer", "float"]:
                if "minimum" in field_spec and value < field_spec["minimum"]:
                    return False
                if "maximum" in field_spec and value > field_spec["maximum"]:
                    return False
        
        return True
    
    def _parse_llm_response(self, response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        
        # Try to extract JSON from response
        try:
            # Look for JSON object in response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                sample = json.loads(json_str)
                
                # Validate and cast types according to schema
                validated_sample = {}
                for field_name, field_spec in schema.items():
                    if field_name in sample:
                        value = sample[field_name]
                        field_type = field_spec.get("type", "string")
                        
                        try:
                            if field_type == "integer":
                                validated_sample[field_name] = int(value)
                            elif field_type == "float":
                                validated_sample[field_name] = float(value)
                            elif field_type == "boolean":
                                validated_sample[field_name] = bool(value)
                            else:
                                validated_sample[field_name] = str(value)
                        except (ValueError, TypeError):
                            # Use fallback value
                            validated_sample[field_name] = self._generate_field_value(field_name, field_spec, None)
                    else:
                        # Field missing, generate fallback
                        validated_sample[field_name] = self._generate_field_value(field_name, field_spec, None)
                
                return validated_sample
        except json.JSONDecodeError:
            pass
        
        # Fallback: generate from schema
        return self._generate_fallback_sample(schema, None)
    
    def _generate_fallback_sample(self, schema: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a sample using template logic when LLM fails"""
        sample = {}
        for field_name, field_spec in schema.items():
            sample[field_name] = self._generate_field_value(field_name, field_spec, constraints)
        return sample
    
    def _generate_field_value(self, field_name: str, field_spec: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> Any:
        """Generate a single field value based on spec"""
        field_type = field_spec.get("type", "string")
        
        if field_type == "string":
            return self._generate_string(field_name, field_spec, constraints)
        elif field_type == "integer":
            return self._generate_integer(field_name, field_spec, constraints)
        elif field_type == "float":
            return self._generate_float(field_spec, constraints)
        elif field_type == "boolean":
            return random.choice([True, False])
        elif field_type == "datetime":
            return self._generate_datetime(field_name)
        elif field_type == "email":
            return self._generate_email(field_name)
        elif field_type == "url":
            return f"https://example.com/resource/{random.randint(1000, 9999)}"
        else:
            return None
    
    def _format_output(self, samples: List[Dict[str, Any]], format: str) -> Any:
        """Format output data"""
        
        if format == "json":
            return samples
        
        elif format == "csv":
            if not samples:
                return ""
            
            # Create CSV
            import csv
            import io
            
            output = io.StringIO(newline='')  # Fix: prevent extra blank lines
            writer = csv.DictWriter(output, fieldnames=samples[0].keys())
            writer.writeheader()
            writer.writerows(samples)
            
            return output.getvalue()
        
        elif format == "sql":
            # Generate SQL INSERT statements
            if not samples:
                return ""
            
            table_name = "synthetic_data"
            columns = list(samples[0].keys())
            
            sql_statements = []
            for sample in samples:
                values = [f"'{v}'" if isinstance(v, str) else str(v) 
                         for v in sample.values()]
                sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
                sql_statements.append(sql)
            
            return "\n".join(sql_statements)
        
        else:
            return samples
