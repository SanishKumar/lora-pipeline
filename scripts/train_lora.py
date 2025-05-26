#!/usr/bin/env python3
"""
LoRA Training Script

Usage:
  python scripts/train_lora.py --config configs/training_config.yaml [--resume <checkpoint>]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src/ to Python path so we can import our training modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from training.lora_trainer import LoRATrainer

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint or output dir to resume from")
    parser.add_argument("--output", type=str, default=None,
                        help="(Optional) override the output directory")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting LoRA training...")

    # Instantiate and run trainer
    trainer = LoRATrainer(config_path=args.config)
    if args.output:
        trainer.config["training"]["output_dir"] = args.output
    if args.resume:
        trainer.config["training"]["resume_from_checkpoint"] = args.resume

    trainer.train()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
