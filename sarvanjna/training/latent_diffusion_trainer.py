"""
PyTorch Lightning trainer for Latent Diffusion Model.
"""

import torch
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sarvanjna.models.vision.latent_diffusion import LatentDiffusionModel, LatentDiffusionConfig


class LatentDiffusionTrainer(pl.LightningModule):
    """
    Lightning module for training Latent Diffusion Model.
    
    Handles:
    - Training loop with diffusion loss
    - Validation with image generation
    - Optimizer and learning rate scheduling
    """
    
    def __init__(
        self,
        model_config: LatentDiffusionConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Model
        self.model = LatentDiffusionModel(model_config)
        
        # EMA model (optional)
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LatentDiffusionModel(model_config)
            self.model_ema.load_state_dict(self.model.state_dict())
            self.model_ema.eval()
            for param in self.model_ema.parameters():
                param.requires_grad = False
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.ema_decay = ema_decay
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(images, input_ids, attention_mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Forward pass
        outputs = self(images, input_ids, attention_mask)
        loss = outputs['loss']
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        # Update EMA model
        if self.use_ema:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Use EMA model if available
        model = self.model_ema if self.use_ema else self.model
        
        # Forward pass
        outputs = model(images, input_ids, attention_mask)
        loss = outputs['loss']
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        # Generate sample images (first batch only)
        if batch_idx == 0:
            with torch.no_grad():
                # Take first 4 prompts
                sample_input_ids = input_ids[:4]
                sample_attention_mask = attention_mask[:4] if attention_mask is not None else None
                
                # Generate images
                generated_images = model.generate(
                    sample_input_ids,
                    sample_attention_mask,
                    num_inference_steps=20,  # Fast sampling for validation
                    guidance_scale=7.5,
                )
                
                # Log images
                if self.logger is not None:
                    # Convert from [-1, 1] to [0, 1]
                    generated_images = (generated_images + 1) / 2
                    
                    # Log to TensorBoard
                    if hasattr(self.logger.experiment, 'add_images'):
                        self.logger.experiment.add_images(
                            'val/generated',
                            generated_images,
                            self.global_step,
                        )
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Only optimize U-Net parameters (freeze VAE and text encoder)
        params = list(self.model.unet.parameters())
        
        optimizer = AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Cosine annealing with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=1e-7,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }
    
    def _update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.model_ema.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save EMA model state."""
        if self.use_ema:
            checkpoint['model_ema_state_dict'] = self.model_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load EMA model state."""
        if self.use_ema and 'model_ema_state_dict' in checkpoint:
            self.model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
