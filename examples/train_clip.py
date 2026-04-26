"""
Example script for training CLIP model.

Usage:
    python examples/train_clip.py --config configs/clip_base.yaml --data_dir data/image_text_pairs
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from sarvanjna.core.config import Config
from sarvanjna.data.image_text_dataset import ImageTextDataset
from sarvanjna.training.clip_trainer import CLIPTrainer
from sarvanjna.models.vision.clip import CLIPConfig
from sarvanjna.models.vision.vision_transformer import ViTConfig
from sarvanjna.models.text.transformer import TransformerConfig
from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer


def collate_fn(batch):
    """Collate function for CLIP training."""
    images = []
    input_ids = []
    attention_masks = []
    
    for item in batch:
        images.append(item['image'])
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
    
    import torch
    return {
        'image': torch.stack(images),
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer model path")
    parser.add_argument("--output_dir", type=str, default="outputs/clip", help="Output directory")
    parser.add_argument("--project_name", type=str, default="sarvanjna-clip", help="W&B project name")
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Training CLIP Model")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Load tokenizer
    tokenizer = SentencePieceTokenizer.load(args.tokenizer)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # Create CLIP config
    vision_config = ViTConfig(
        image_size=config.data.image_size,
        patch_size=config.model.vision.patch_size,
        d_model=config.model.vision.d_model,
        n_heads=config.model.vision.n_heads,
        n_layers=config.model.vision.n_layers,
        num_classes=None,
    )
    
    text_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.text.d_model,
        n_heads=config.model.text.n_heads,
        n_layers=config.model.text.n_layers,
        max_seq_length=config.text.max_length,
    )
    
    clip_config = CLIPConfig(
        vision_config=vision_config,
        text_config=text_config,
        embed_dim=config.model.embed_dim,
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ImageTextDataset(
        data_dir=args.data_dir,
        split='train',
        tokenizer=tokenizer,
        image_size=config.data.image_size,
        max_length=config.text.max_length,
        transform_type='train',
    )
    
    val_dataset = ImageTextDataset(
        data_dir=args.data_dir,
        split='val',
        tokenizer=tokenizer,
        image_size=config.data.image_size,
        max_length=config.text.max_length,
        transform_type='val',
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("\nCreating CLIP model...")
    model = CLIPTrainer(
        config=clip_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
    )
    
    num_params = model.model.get_num_params()
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="clip-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
    )
    
    # Setup logger
    logger = WandbLogger(
        project=args.project_name,
        save_dir=output_dir,
        log_model=True,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator="auto",
        devices="auto",
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
