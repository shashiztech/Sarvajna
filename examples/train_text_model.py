"""
Example training script for Text-to-Text model.

Usage:
    python examples/train_text_model.py --config configs/text_base.yaml
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader

from sarvanjna.core.config import Config
from sarvanjna.models.text.transformer import TransformerConfig
from sarvanjna.training.text_trainer import TextToTextTrainer
from sarvanjna.data.text_dataset import TextDataset
from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Text-to-Text model")
    parser.add_argument("--config", type=str, default="configs/text_base.yaml", help="Config file path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Path to SentencePiece model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Setup directories
    config.setup_directories()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TextDataset(
        data_path=Path(args.data_path),
        tokenizer=tokenizer,
        max_length=config.text.max_seq_length,
        task_type="instruction",
    )
    
    val_dataset = None
    if args.val_data_path:
        val_dataset = TextDataset(
            data_path=Path(args.val_data_path),
            tokenizer=tokenizer,
            max_length=config.text.max_seq_length,
            task_type="instruction",
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TextDataset.collate_fn,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=TextDataset.collate_fn,
            pin_memory=True,
        )
    
    # Create model config
    model_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_seq_length=config.text.max_seq_length,
    )
    
    # Create trainer module
    print("Initializing model...")
    model = TextToTextTrainer(
        config=model_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        gradient_clip=config.training.gradient_clip,
    )
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="text-model-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss" if val_loader else "train/loss",
            mode="min",
            save_top_k=config.training.keep_last_n_checkpoints,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    if val_loader:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=5,
                mode="min",
            )
        )
    
    # Setup logger
    logger = None
    if config.training.use_wandb:
        logger = WandbLogger(
            project=config.training.wandb_project,
            name="text-to-text-training",
            save_dir=output_dir,
        )
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nTraining complete! Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
