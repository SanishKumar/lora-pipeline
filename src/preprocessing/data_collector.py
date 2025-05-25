import os
import requests
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple
import logging

class DatasetCollector:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_from_urls(self, urls: List[str], style_name: str):
        """Download images from URLs"""
        style_dir = self.data_dir / style_name
        style_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    filename = f"{style_name}_{i:04d}.jpg"
                    filepath = style_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    self.logger.info(f"Downloaded: {filename}")
                else:
                    self.logger.warning(f"Failed to download: {url}")
            except Exception as e:
                self.logger.error(f"Error downloading {url}: {e}")
    
    def collect_local_images(self, source_dir: str, style_name: str):
        """Copy images from local directory"""
        source_path = Path(source_dir)
        style_dir = self.data_dir / style_name
        style_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for i, img_path in enumerate(source_path.glob('*')):
            if img_path.suffix.lower() in image_extensions:
                new_filename = f"{style_name}_{i:04d}{img_path.suffix}"
                new_path = style_dir / new_filename
                
                # Copy and potentially convert image
                img = Image.open(img_path)
                img = img.convert('RGB')
                img.save(new_path, 'JPEG', quality=95)
                
                self.logger.info(f"Processed: {new_filename}")

# Usage example
if __name__ == "__main__":
    collector = DatasetCollector()
    
    # Example: collect from local directory
    collector.collect_local_images("/path/to/your/art/images", "custom_art_style")