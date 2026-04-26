"""
Example script for training Image VAE.

Usage:
    python examples/train_vae.py --config configs/vae_base.yaml --data_dir data/images
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sarvanjna.core.config import Config
from sarvanjna.training.vae_trainer import VAETrainer
from sarvanjna.models.vision.image_autoencoder import VAEConfig


class ImageDataset:
    """Simple image dataset for VAE training."""
    
    def __init__(self, data_dir: str, image_size: int = 256, split: str = 'train'):
        from PIL import Image
        import os
        
        self.data_dir = Path(data_dir) / split
        self.image_files = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))
        
        # Data augmentation for training
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {'image': image}


def main():
    parser = argparse.ArgumentParser(description="Train Image VAE")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/vae", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Training Image VAE")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Create VAE config
    vae_config = VAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=config.model.latent_channels,
        base_channels=config.model.base_channels,
        resolution=config.data.image_size,
        kl_weight=config.model.kl_weight,
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ImageDataset(
        data_dir=args.data_dir,
        image_size=config.data.image_size,
        split='train',
    )
    
    val_dataset = ImageDataset(
        data_dir=args.data_dir,
        image_size=config.data.image_size,
        split='val',
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating VAE model...")
    model = VAETrainer(
        config=vae_config,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    num_params = model.model.get_num_params()
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator="auto",
        devices="auto",
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=config.training.val_check_interval,
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
