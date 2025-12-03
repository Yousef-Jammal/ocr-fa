"""
FIXED: Streamlined image generation using Stable Diffusion
Handles black image issues and corrupted outputs
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
from PIL import Image
import random
import torch
import gc

try:
    from diffusers import StableDiffusionPipeline
    STABLE_DIFFUSION_AVAILABLE = True
except ImportError:
    STABLE_DIFFUSION_AVAILABLE = False
    print("Warning: Stable Diffusion not available.")

class ImageGenerator:
    """Generate synthetic images using Stable Diffusion"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if STABLE_DIFFUSION_AVAILABLE:
            try:
                print(f"Loading {model_name}...")
                # 🔥 FIX: Use float32 for stability (prevents NaN/Inf)
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # float32 is more stable than float16
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                # Enable memory efficient attention if available
                if self.device == "cuda":
                    try:
                        self.pipeline.enable_attention_slicing()
                        self.pipeline.enable_vae_slicing()
                    except:
                        pass
                
                self.pipeline = self.pipeline.to(self.device)
                print(f"✓ Stable Diffusion loaded on {self.device} (float32 mode)")
            except Exception as e:
                print(f"Failed to load Stable Diffusion: {e}")
                self.pipeline = None
    
    async def generate(
        self,
        prompts: List[str],
        num_images: int,
        resolution: tuple[int, int] = (512, 512),
        scene_type: str = "general",
        annotations: List[str] = None,
        objects: Optional[List[str]] = None,
        output_dir: Path = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic images from prompts
        
        Args:
            prompts: List of text prompts for image generation
            num_images: Number of images to generate
            resolution: Image resolution (width, height)
            scene_type: Scene type hint (for annotation generation)
            annotations: Annotation types to generate (bbox, segmentation)
            objects: Optional specific objects for annotations
            output_dir: Output directory
            
        Returns:
            Dictionary with generation results
        """
        
        if self.pipeline and STABLE_DIFFUSION_AVAILABLE:
            return await self._generate_with_diffusion(
                prompts, num_images, resolution, scene_type, 
                annotations or [], objects, output_dir
            )
        else:
            return await self._generate_fallback(
                prompts, num_images, resolution, scene_type, 
                annotations or [], objects, output_dir
            )
    
    async def _generate_with_diffusion(
        self,
        prompts: List[str],
        num_images: int,
        resolution: tuple[int, int],
        scene_type: str,
        annotations: List[str],
        objects: Optional[List[str]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate using Stable Diffusion with black image fixes"""
        
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        all_annotations = []
        images_generated = 0
        errors = []
        
        for i in range(num_images):
            # Select or cycle through prompts
            prompt = prompts[i % len(prompts)]
            
            # 🔥 FIX 1: Enhance prompt to prevent black images
            enhanced_prompt = self._enhance_prompt(prompt, scene_type)
            
            # 🔥 FIX 2: Simplified negative prompt
            negative_prompt = "dark, black, monochrome, blurry, low quality"
            
            try:
                # 🔥 FIX 3: Clear GPU memory before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 🔥 FIX 4: Use different seed for each image
                generator = torch.Generator(device=self.device).manual_seed(42 + i)
                
                # Generate image with optimal settings
                with torch.no_grad():
                    # 🔥 FIX 5: Balanced settings for SD v1.5
                    output = self.pipeline(
                        enhanced_prompt,
                        negative_prompt=negative_prompt,
                        height=resolution[1],
                        width=resolution[0],
                        num_inference_steps=30,  # 30 is optimal for SD v1.5
                        guidance_scale=7.5,  # 7.5 is the sweet spot
                        generator=generator,
                        output_type="pil"
                    )
                    
                    image = output.images[0]
                
                # 🔥 FIX 7: Validate image is not black/corrupted
                if self._is_image_valid(image):
                    # Save image
                    img_path = images_dir / f"image_{i:05d}.png"
                    image.save(img_path, format="PNG", optimize=False)
                    images_generated += 1
                    
                    print(f"✓ Generated image {i+1}/{num_images}")
                    
                else:
                    # 🔥 FIX 8: Retry with different settings
                    print(f"⚠️ Image {i} is black/corrupted, retrying...")
                    image = await self._retry_generation(
                        enhanced_prompt, 
                        negative_prompt, 
                        resolution, 
                        i
                    )
                    
                    if image and self._is_image_valid(image):
                        img_path = images_dir / f"image_{i:05d}.png"
                        image.save(img_path, format="PNG")
                        images_generated += 1
                        print(f"✓ Retry successful for image {i+1}/{num_images}")
                    else:
                        # Use fallback
                        print(f"❌ Using fallback for image {i+1}")
                        image = self._generate_synthetic_image(resolution, scene_type)
                        img_path = images_dir / f"image_{i:05d}.png"
                        image.save(img_path)
                        images_generated += 1
                        errors.append(f"Image {i}: Used fallback due to generation failure")
                
                # Generate annotations if requested
                img_annotations = {
                    "image_id": i,
                    "file_name": img_path.name,
                    "width": resolution[0],
                    "height": resolution[1],
                    "prompt": prompt,
                    "scene_type": scene_type
                }
                
                if "bbox" in annotations:
                    img_annotations["bboxes"] = self._generate_bboxes(resolution, objects)
                
                if "segmentation" in annotations:
                    seg_mask = self._generate_segmentation(resolution)
                    seg_path = annotations_dir / f"seg_{i:05d}.png"
                    Image.fromarray(seg_mask).save(seg_path)
                    img_annotations["segmentation"] = str(seg_path)
                
                all_annotations.append(img_annotations)
                
            except Exception as e:
                error_msg = f"Error generating image {i}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                
                # Use fallback instead of skipping
                try:
                    image = self._generate_synthetic_image(resolution, scene_type)
                    img_path = images_dir / f"image_{i:05d}.png"
                    image.save(img_path)
                    images_generated += 1
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    continue
        
        # Save annotations in COCO format
        if all_annotations:
            coco_annotations = self._convert_to_coco(all_annotations)
            with open(annotations_dir / "annotations.json", "w") as f:
                json.dump(coco_annotations, f, indent=2)
        
        return {
            "num_images": images_generated,
            "requested": num_images,
            "images_dir": str(images_dir),
            "annotations_dir": str(annotations_dir) if annotations else None,
            "annotation_format": "COCO" if annotations else None,
            "scene_type": scene_type,
            "resolution": resolution,
            "prompts_used": prompts,
            "generator": "Stable Diffusion v1.5",
            "errors": errors if errors else None
        }
    
    def _enhance_prompt(self, prompt: str, scene_type: str) -> str:
        """
        🔥 FIX: Simplify prompt enhancement - too complex prompts confuse SD v1.5
        """
        # Keep it simple for better results
        lighting_map = {
            "indoor": "bright interior",
            "outdoor": "daylight",
            "warehouse": "bright lighting",
            "general": "well-lit"
        }
        
        lighting = lighting_map.get(scene_type, lighting_map["general"])
        
        # Minimal enhancement - let the user's prompt do the work
        enhanced = f"{prompt}, {lighting}, detailed"
        
        return enhanced
    
    def _is_image_valid(self, image: Image.Image) -> bool:
        """
        🔥 FIX: Check if image is not black or corrupted
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check for NaN or Inf
            if np.isnan(img_array).any() or np.isinf(img_array).any():
                return False
            
            # Check if image is too dark (black)
            mean_brightness = np.mean(img_array)
            if mean_brightness < 10:  # Almost completely black
                return False
            
            # Check for zero variance (solid color)
            if np.std(img_array) < 5:  # Almost no variation
                return False
            
            # Check if all pixels are the same
            unique_values = len(np.unique(img_array))
            if unique_values < 100:  # Too few unique values
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating image: {e}")
            return False
    
    async def _retry_generation(
        self,
        prompt: str,
        negative_prompt: str,
        resolution: tuple[int, int],
        attempt: int
    ) -> Optional[Image.Image]:
        """
        🔥 FIX: Retry generation with different settings
        """
        try:
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Use different seed
            generator = torch.Generator(device=self.device).manual_seed(1000 + attempt * 100)
            
            # Try with different parameters
            with torch.no_grad():
                output = self.pipeline(
                    prompt,
                    negative_prompt=negative_prompt,
                    height=resolution[1],
                    width=resolution[0],
                    num_inference_steps=40,
                    guidance_scale=8.0,
                    generator=generator,
                    output_type="pil"
                )
                
                return output.images[0]
        
        except Exception as e:
            print(f"Retry failed: {e}")
            return None
    
    async def _generate_fallback(
        self,
        prompts: List[str],
        num_images: int,
        resolution: tuple[int, int],
        scene_type: str,
        annotations: List[str],
        objects: Optional[List[str]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Fallback generator using procedural generation"""
        
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        all_annotations = []
        
        for i in range(num_images):
            # Generate synthetic image
            img = self._generate_synthetic_image(resolution, scene_type)
            img_path = images_dir / f"image_{i:05d}.png"
            img.save(img_path)
            
            # Generate annotations
            img_annotations = {
                "image_id": i,
                "file_name": img_path.name,
                "width": resolution[0],
                "height": resolution[1],
                "prompt": prompts[i % len(prompts)] if prompts else "synthetic",
                "scene_type": scene_type
            }
            
            if "bbox" in annotations:
                img_annotations["bboxes"] = self._generate_bboxes(resolution, objects)
            
            if "segmentation" in annotations:
                seg_mask = self._generate_segmentation(resolution)
                seg_path = annotations_dir / f"seg_{i:05d}.png"
                Image.fromarray(seg_mask).save(seg_path)
                img_annotations["segmentation"] = str(seg_path)
            
            all_annotations.append(img_annotations)
        
        # Save COCO format annotations
        if all_annotations:
            coco_annotations = self._convert_to_coco(all_annotations)
            with open(annotations_dir / "annotations.json", "w") as f:
                json.dump(coco_annotations, f, indent=2)
        
        return {
            "num_images": num_images,
            "images_dir": str(images_dir),
            "annotations_dir": str(annotations_dir) if annotations else None,
            "annotation_format": "COCO" if annotations else None,
            "scene_type": scene_type,
            "resolution": resolution,
            "generator": "Fallback (Procedural)"
        }
    
    def _generate_synthetic_image(self, resolution: tuple[int, int], scene_type: str) -> Image.Image:
        """Generate a procedural synthetic image (ENHANCED - more colorful)"""
        width, height = resolution
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate gradient based on scene type
        color_schemes = {
            "indoor": ([220, 200, 180], [140, 120, 100]),
            "outdoor": ([135, 206, 250], [60, 179, 113]),
            "warehouse": ([180, 180, 190], [80, 80, 90]),
            "general": ([200, 180, 220], [100, 120, 140])
        }
        
        colors = color_schemes.get(scene_type, color_schemes["general"])
        
        for y in range(height):
            ratio = y / height
            for c in range(3):
                img_array[y, :, c] = int(colors[0][c] * (1 - ratio) + colors[1][c] * ratio)
        
        # Add random colorful shapes
        num_objects = random.randint(5, 12)
        for _ in range(num_objects):
            x1 = random.randint(0, max(1, width - 100))
            y1 = random.randint(0, max(1, height - 100))
            w = random.randint(50, min(200, width - x1))
            h = random.randint(50, min(200, height - y1))
            
            # More vibrant colors
            color = [random.randint(80, 255) for _ in range(3)]
            img_array[y1:y1+h, x1:x1+w] = color
        
        return Image.fromarray(img_array)
    
    def _generate_bboxes(self, resolution: tuple[int, int], objects: Optional[List[str]] = None) -> List[Dict]:
        """Generate synthetic bounding boxes"""
        width, height = resolution
        num_boxes = random.randint(2, 6)
        
        default_objects = ["object", "item", "entity", "element"]
        object_classes = objects or default_objects
        
        bboxes = []
        for i in range(num_boxes):
            x = random.randint(0, max(1, width - 100))
            y = random.randint(0, max(1, height - 100))
            w = random.randint(50, min(150, width - x))
            h = random.randint(50, min(150, height - y))
            
            bboxes.append({
                "id": i,
                "category": random.choice(object_classes),
                "bbox": [x, y, w, h],
                "area": w * h,
                "confidence": random.uniform(0.85, 0.99)
            })
        
        return bboxes
    
    def _generate_segmentation(self, resolution: tuple[int, int]) -> np.ndarray:
        """Generate synthetic segmentation mask"""
        width, height = resolution
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add random segments
        num_segments = random.randint(3, 6)
        for i in range(1, num_segments + 1):
            x = random.randint(0, max(1, width - 100))
            y = random.randint(0, max(1, height - 100))
            w = random.randint(50, min(150, width - x))
            h = random.randint(50, min(150, height - y))
            
            mask[y:y+h, x:x+w] = i
        
        return mask
    
    def _convert_to_coco(self, annotations: List[Dict]) -> Dict:
        """Convert annotations to COCO format"""
        coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Collect unique categories
        categories = set()
        for ann in annotations:
            if "bboxes" in ann:
                for bbox in ann["bboxes"]:
                    categories.add(bbox["category"])
        
        # Add categories
        for i, cat in enumerate(sorted(categories)):
            coco["categories"].append({
                "id": i,
                "name": cat,
                "supercategory": "object"
            })
        
        # Add images and annotations
        ann_id = 0
        for img_ann in annotations:
            coco["images"].append({
                "id": img_ann["image_id"],
                "file_name": img_ann["file_name"],
                "width": img_ann["width"],
                "height": img_ann["height"]
            })
            
            if "bboxes" in img_ann:
                for bbox in img_ann["bboxes"]:
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_ann["image_id"],
                        "category_id": bbox["id"] % len(coco["categories"]),
                        "bbox": bbox["bbox"],
                        "area": bbox["area"],
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        return coco