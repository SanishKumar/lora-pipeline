#!/usr/bin/env python3
"""
Data Preparation Script
Usage: python scripts/prepare_data.py --input data/raw/my_style --output data/processed
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.preprocessor import ImagePreprocessor
from preprocessing.data_collector import DatasetCollector
import logging

def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input", type=str, required=True, help="Input directory with raw images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--augment-factor", type=int, default=2, help="Number of augmented versions per image")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Check input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Count images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    logger.info(f"Found {len(image_files)} images in {input_path}")
    
    if len(image_files) == 0:
        logger.error("No images found! Make sure your images are in .jpg or .png format")
        return
    
    # Process images
    logger.info("Starting image preprocessing...")
    preprocessor = ImagePreprocessor(
        input_dir=str(input_path),
        output_dir=args.output,
        target_size=(args.size, args.size)
    )
    
    preprocessor.process_dataset(
        augment=args.augment,
        augment_factor=args.augment_factor
    )
    
    logger.info("Data preparation completed!")

if __name__ == "__main__":
    main()