import argparse
import logging
import math
import os
import random
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from peft import LoraConfig, get_peft_model, TaskType

check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class LoRATrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.accelerator = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        
    def setup_accelerator(self):
        """Initialize accelerator for distributed training"""
        project_config = ProjectConfiguration(
            project_dir=self.config['training']['output_dir'],
            logging_dir=self.config['training']['logging_dir']
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            mixed_precision=self.config['training']['mixed_precision'],
            log_with=self.config['training']['report_to'],
            project_config=project_config,
        )
        
        if self.accelerator.is_local_main_process:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )
    
    def load_models(self):
        """Load pretrained models"""
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="tokenizer",
            revision=self.config['model']['revision']
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="text_encoder",
            revision=self.config['model']['revision'],
            variant=self.config['model']['variant']
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="vae",
            revision=self.config['model']['revision'],
            variant=self.config['model']['variant']
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="unet",
            revision=self.config['model']['revision'],
            variant=self.config['model']['variant']
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="scheduler"
        )
        
        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
    
    def setup_lora(self):
        """Setup LoRA configuration"""
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['rank'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=self.config['lora']['dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type'],
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Enable training mode for LoRA parameters only
        self.unet.train()
        
        if self.config['training']['enable_xformers_memory_efficient_attention']:
            if is_xformers_available():
                import xformers
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available")
    
    def prepare_dataset(self):
        """Prepare training dataset"""
        # Image transforms
        train_transforms = transforms.Compose([
            transforms.Resize(self.config['dataset']['resolution'], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.config['dataset']['resolution']) if self.config['dataset']['center_crop'] else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if self.config['dataset']['random_flip'] else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[self.config['dataset']['image_column']]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = self.tokenize_captions(examples)
            return examples
        
        # Load dataset
        dataset = load_dataset("imagefolder", data_dir=self.config['dataset']['data_dir'])
        
        # Add captions (you may want to customize this)
        def add_captions(batch):
            batch['text'] = ['a beautiful artwork in the custom style'] * len(batch['image'])
            return batch
        
        dataset = dataset.map(add_captions, batched=True)
        
        # Preprocess dataset
        train_dataset = dataset["train"].with_transform(preprocess_train)
        
        return train_dataset
    
    def tokenize_captions(self, examples):
        """Tokenize captions"""
        captions = []
        for caption in examples[self.config['dataset']['caption_column']]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if len(caption) > 0 else "")
            else:
                captions.append("")
        
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def train(self):
        """Main training loop"""
        # Setup
        self.setup_accelerator()
        self.load_models()
        self.setup_lora()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config['training']['train_batch_size'],
            num_workers=0,
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # Scheduler
        lr_scheduler = get_scheduler(
            self.config['training']['lr_scheduler'],
            optimizer=optimizer,
            num_warmup_steps=self.config['training']['lr_warmup_steps'] * self.config['training']['gradient_accumulation_steps'],
            num_training_steps=self.config['training']['max_train_steps'] * self.config['training']['gradient_accumulation_steps'],
        )
        
        # Prepare everything with accelerator
        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        
        # Training loop
        global_step = 0
        first_epoch = 0
        
        progress_bar = tqdm(
            range(0, self.config['training']['max_train_steps']),
            initial=global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for epoch in range(first_epoch, self.config['training']['num_train_epochs']):
            self.unet.train()
            train_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample timesteps
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    
                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    
                    # Get model prediction
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    
                    # Get target
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
                    
                    # Calculate loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Gather losses across processes
                    avg_loss = self.accelerator.gather(loss.repeat(self.config['training']['train_batch_size'])).mean()
                    train_loss += avg_loss.item() / self.config['training']['gradient_accumulation_steps']
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % self.config['training']['checkpointing_steps'] == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.config['training']['output_dir'], f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step >= self.config['training']['max_train_steps']:
                    break
        
        # Save final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.accelerator.unwrap_model(self.unet)
            self.unet.save_pretrained(self.config['training']['output_dir'])
            logger.info(f"Saved LoRA weights to {self.config['training']['output_dir']}")
        
        self.accelerator.end_training()

# Usage
if __name__ == "__main__":
    trainer = LoRATrainer("configs/training_config.yaml")
    trainer.train()