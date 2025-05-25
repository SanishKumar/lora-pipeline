#!/usr/bin/env python3
"""
LoRA Training Script
Usage: python scripts/train_lora.py --config configs/training_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.lora_trainer import LoRATrainer
import logging

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--output", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LoRA training...")
    
    try:
        trainer = LoRATrainer(args.config)
        
        # Override config if needed
        if args.output:
            trainer.config['training']['output_dir'] = args.output
        if args.resume:
            trainer.config['training']['resume_from_checkpoint'] = args.resume
        
        trainer.train()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()