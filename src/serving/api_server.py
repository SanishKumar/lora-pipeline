from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LoRA Fine-Tuner API", version="1.0.0")

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    use_lora: bool = True
    lora_scale: float = 1.0

class GenerationResponse(BaseModel):
    image: str  # base64 encoded
    seed: int
    generation_time: float

class LoRAInferenceService:
    def __init__(self, base_model_path: str, lora_path: Optional[str] = None):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load the diffusion pipeline"""
        try:
            logger.info("Loading base model...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LoRA weights if available
            if self.lora_path and os.path.exists(self.lora_path):
                logger.info(f"Loading LoRA weights from {self.lora_path}")
                self.pipeline.load_lora_weights(self.lora_path)
            
            # Enable memory efficient attention
            self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            logger.info("Pipeline loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise e
    
    def generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """Generate image from prompt"""
        import time
        start_time = time.time()
        
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
        else:
            request.seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(request.seed)
        
        try:
            # Generate image
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    cross_attention_kwargs={"scale": request.lora_scale} if request.use_lora else None
                )
            
            image = result.images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            generation_time = time.time() - start_time
            
            return GenerationResponse(
                image=img_str,
                seed=request.seed,
                generation_time=generation_time
            )
        
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
service = LoRAInferenceService(
    base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
    lora_path="models/lora"
)

@app.get("/")
async def root():
    return {"message": "LoRA Fine-Tuner API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": service.device}

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate an image from a text prompt"""
    try:
        response = service.generate_image(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "base_model": service.base_model_path,
        "lora_model": service.lora_path,
        "device": service.device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)