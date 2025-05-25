# LoRA Fine-Tuner Pipeline Usage Guide

## Quick Start

### 1. Prepare Your Data
```bash
# Put your style images in data/raw/my_style/
python scripts/prepare_data.py --input data/raw/my_style --output data/processed --augment