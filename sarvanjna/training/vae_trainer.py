"""
PyTorch Lightning trainer for Image VAE.
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any

from ..models.vision.image_autoencoder import ImageAutoencoder, VAEConfig


class VAETrainer(pl.LightningModule):
    """
    Lightning module for training Image VAE.
    
    Trains variational autoencoder for image compression.
    """
    
    def __init__(
        self,
        config: VAEConfig,
        learning_rate: float = 4.5e-6,
        weight_decay: float = 0.0,
        betas: tuple = (0.5, 0.9),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        
        self.model = ImageAutoencoder(config)
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        return self.model(x, return_latent=True)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        
        # Forward pass
        outputs = self(images)
        loss = outputs['loss']
        recon_loss = outputs['recon_loss']
        kl_loss = outputs['kl_loss']
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/recon_loss', recon_loss, prog_bar=True, sync_dist=True)
        self.log('train/kl_loss', kl_loss, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images = batch['image']
        
        # Forward pass
        outputs = self(images)
        loss = outputs['loss']
        recon_loss = outputs['recon_loss']
        kl_loss = outputs['kl_loss']
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', recon_loss, prog_bar=True, sync_dist=True)
        self.log('val/kl_loss', kl_loss, prog_bar=True, sync_dist=True)
        
        # Optionally log sample images
        if batch_idx == 0:
            self.log_images(images, outputs['reconstruction'])
        
        return {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
    
    def log_images(self, original: torch.Tensor, reconstruction: torch.Tensor, num_images: int = 4):
        """Log original and reconstructed images to tensorboard."""
        if hasattr(self.logger, 'experiment'):
            # Take first few images
            original = original[:num_images]
            reconstruction = reconstruction[:num_images]
            
            # Clamp to [0, 1] for visualization
            original = torch.clamp(original, 0, 1)
            reconstruction = torch.clamp(reconstruction, 0, 1)
            
            # Log to tensorboard
            self.logger.experiment.add_images(
                'val/original',
                original,
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                'val/reconstruction',
                reconstruction,
                self.current_epoch,
            )
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        
        return optimizer
    
    def encode(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Encode images to latent space for inference."""
        self.eval()
        with torch.no_grad():
            return self.model.encode(images, sample=sample)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images for inference."""
        self.eval()
        with torch.no_grad():
            return self.model.decode(latents)
