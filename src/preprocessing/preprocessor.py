import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
from typing import Tuple, List
import json

class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str, target_size: Tuple[int, int] = (512, 512)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def resize_and_crop(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio and center crop"""
        # Calculate scaling factor
        w, h = image.size
        target_w, target_h = self.target_size
        
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        return image.crop((left, top, right, bottom))
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply random enhancements for data augmentation"""
        # Random brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(random.uniform(0.8, 1.2))
        
        # Random contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(random.uniform(0.8, 1.2))
        
        # Random saturation
        color = ImageEnhance.Color(image)
        image = color.enhance(random.uniform(0.8, 1.2))
        
        return image
    
    def process_dataset(self, augment: bool = True, augment_factor: int = 3):
        """Process entire dataset"""
        metadata = []
        
        for img_path in self.input_dir.glob('*.jpg'):
            try:
                # Load and process original
                image = Image.open(img_path).convert('RGB')
                processed = self.resize_and_crop(image)
                
                # Save original processed version
                output_path = self.output_dir / img_path.name
                processed.save(output_path, 'JPEG', quality=95)
                
                metadata.append({
                    'original': str(img_path),
                    'processed': str(output_path),
                    'size': self.target_size,
                    'augmented': False
                })
                
                # Create augmented versions
                if augment:
                    for i in range(augment_factor):
                        augmented = self.enhance_image(processed.copy())
                        aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        aug_path = self.output_dir / aug_name
                        augmented.save(aug_path, 'JPEG', quality=95)
                        
                        metadata.append({
                            'original': str(img_path),
                            'processed': str(aug_path),
                            'size': self.target_size,
                            'augmented': True,
                            'augmentation_id': i
                        })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed {len(metadata)} images total")

# Usage
if __name__ == "__main__":
    preprocessor = ImagePreprocessor("data/raw/custom_art_style", "data/processed")
    preprocessor.process_dataset(augment=True, augment_factor=2)