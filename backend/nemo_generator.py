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
        """Generate using Qwen2.5 model"""
        
        samples = []
        
        for i in range(num_samples):
            # Create prompt from schema
            prompt = self._create_prompt_from_schema(schema, constraints)
            
            # Generate with Qwen
            try:
                messages = [
                    {"role": "system", "content": "You are a data generation assistant. Generate realistic structured data exactly matching the provided schema. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Parse response into structured data
                sample = self._parse_llm_response(response, schema)
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
        field_spec: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Generate string value"""
        
        examples = field_spec.get("examples", [])
        if examples:
            return random.choice(examples)
        
        min_length = field_spec.get("min_length", 5)
        max_length = field_spec.get("max_length", 20)
        length = random.randint(min_length, max_length)
        
        # Generate random words
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "cat", "mouse", "bird", "fish", "tree", "house", "car", "book"]
        
        result = []
        current_length = 0
        while current_length < length:
            word = random.choice(words)
            result.append(word)
            current_length += len(word) + 1
        
        return " ".join(result)[:length]
    
    def _generate_integer(
        self,
        field_spec: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> int:
        """Generate integer value"""
        
        min_val = field_spec.get("minimum", 0)
        max_val = field_spec.get("maximum", 100)
        
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
                            validated_sample[field_name] = self._generate_field_value(field_spec, None)
                    else:
                        # Field missing, generate fallback
                        validated_sample[field_name] = self._generate_field_value(field_spec, None)
                
                return validated_sample
        except json.JSONDecodeError:
            pass
        
        # Fallback: generate from schema
        return self._generate_fallback_sample(schema, None)
    
    def _generate_fallback_sample(self, schema: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a sample using template logic when LLM fails"""
        sample = {}
        for field_name, field_spec in schema.items():
            sample[field_name] = self._generate_field_value(field_spec, constraints)
        return sample
    
    def _generate_field_value(self, field_spec: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> Any:
        """Generate a single field value based on spec"""
        field_type = field_spec.get("type", "string")
        
        if field_type == "string":
            return self._generate_string(field_spec, constraints)
        elif field_type == "integer":
            return self._generate_integer(field_spec, constraints)
        elif field_type == "float":
            return self._generate_float(field_spec, constraints)
        elif field_type == "boolean":
            return random.choice([True, False])
        elif field_type == "datetime":
            return datetime.now().isoformat()
        elif field_type == "email":
            return f"user{random.randint(1000, 9999)}@example.com"
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
