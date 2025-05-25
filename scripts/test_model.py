#!/usr/bin/env python3
"""
Model Testing Script
Usage: python scripts/test_model.py --lora models/lora --prompt "test prompt"
"""

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os

def main():
    parser = argparse.ArgumentParser(description="Test trained LoRA model")
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base model path")
    parser.add_argument("--lora", type=str, required=True, help="LoRA model path")
    parser.add_argument("--prompt", type=str, required=True, help="Test prompt")
    parser.add_argument("--output", type=str, default="test_output.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Loading pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    
    pipeline = pipeline.to(device)
    
    # Load LoRA
    if os.path.exists(args.lora):
        print(f"Loading LoRA from {args.lora}")
        pipeline.load_lora_weights(args.lora)
    else:
        print(f"Warning: LoRA path {args.lora} does not exist")
    
    # Enable optimizations
    pipeline.enable_attention_slicing()
    
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        print("xformers not available, using standard attention")
    
    # Generate image
    print(f"Generating image with prompt: '{args.prompt}'")
    torch.manual_seed(args.seed)
    
    with torch.autocast(device):
        image = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            cross_attention_kwargs={"scale": 1.0}
        ).images[0]
    
    # Save image
    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()