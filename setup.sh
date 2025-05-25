#!/bin/bash

echo "Setting up LoRA Fine-Tuner Pipeline..."

# Create directories
mkdir -p data/{raw,processed,augmented}
mkdir -p models/{base,lora,checkpoints}
mkdir -p logs
mkdir -p outputs
mkdir -p scripts

# Set permissions
chmod +x scripts/*.py
chmod +x setup.sh

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download base model (optional - will be downloaded automatically)
echo "Base model will be downloaded automatically during first run."

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your training images to data/raw/your_style_name/"
echo "2. Run: python scripts/prepare_data.py --input data/raw/your_style_name --output data/processed --augment"
echo "3. Run: python scripts/train_lora.py --config configs/training_config.yaml"
echo "4. Test: python scripts/test_model.py --lora models/lora --prompt 'your test prompt'"