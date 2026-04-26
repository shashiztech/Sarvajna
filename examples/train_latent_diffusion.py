"""
Training script for Latent Diffusion Model (text-to-image).

Usage:
    python examples/train_latent_diffusion.py --config configs/latent_diffusion_base.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from sarvanjna.core.config import TrainingConfig
from sarvanjna.models.vision import LatentDiffusionConfig
from sarvanjna.training.latent_diffusion_trainer import LatentDiffusionTrainer
from sarvanjna.data.image_text_dataset import ImageTextDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with images and captions')
    parser.add_argument('--output-dir', type=str, default='outputs/latent_diffusion', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    training_config = TrainingConfig.from_yaml(args.config)
    model_config = LatentDiffusionConfig()  # Default config
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}")
    train_dataset = ImageTextDataset(
        data_dir=args.data_dir,
        split='train',
        tokenizer_path='models/tokenizer.model',
        image_size=512,
        center_crop=True,
        random_flip=True,
    )
    
    val_dataset = ImageTextDataset(
        data_dir=args.data_dir,
        split='val',
        tokenizer_path='models/tokenizer.model',
        image_size=512,
        center_crop=True,
        random_flip=False,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=True if training_config.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=True if training_config.num_workers > 0 else False,
    )
    
    # Create model
    print("Creating Latent Diffusion Model")
    model = LatentDiffusionTrainer(
        model_config=model_config,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        max_steps=training_config.max_steps,
        use_ema=True,
    )
    
    print(f"Total parameters: {model.model.get_num_params():,}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='latent-diffusion-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=training_config.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=training_config.num_gpus if torch.cuda.is_available() else 'auto',
        strategy='ddp' if training_config.num_gpus > 1 else 'auto',
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        val_check_interval=training_config.val_check_interval,
        log_every_n_steps=training_config.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
    )
    
    # Train
    print("Starting training")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    
    print(f"Training complete! Checkpoints saved to {output_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
